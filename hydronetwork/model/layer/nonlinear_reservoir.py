import keras
from keras import Layer, ops
from hydronetwork.model.layer.embedding import Embedding


@keras.saving.register_keras_serializable(package='Custom', name='NonlinearReservoirCell')
class NonlinearReservoirCell(Layer):
    """
    非线性水库模型的单个时间步计算单元。
    流域自由水状态是一个shape为(batch_size, m)的张量。m表示流域中垂向上分层的数量。
    该模型的计算过程如下：
    1. 将该时段的产流量求和后，加到自由水状态上。
        shape: (batch_size, m, n) -> (batch_size, m)
    2. 由产流量分别计算每一层的汇流量比例。这一步将使用m个神经网络，每个神经网络的输出是一个shape为(batch_size, 1)的张量。
        单个神经网络计算前后的shape变化为(batch_size, 1, n) -> (batch_size, 1)
        shape: (batch_size, m, n) -> (batch_size, m)
    3. 将汇流量比例与自由水状态、上一时段的汇流量相乘，得到本时段的汇流量。
        shape: (batch_size, m) * (batch_size, m) + (batch_size, m) * (batch_size, m) -> (batch_size, m)
    4. 更新自由水状态
        shape: (batch_size, m) - (batch_size, m) -> (batch_size, m)

    :param m: 流域中垂向上分层的数量，即土壤层数。该参数用于构建
    :param layer_units: 每层神经网络的神经元数量，注意最后一层神经元数量必须为1
    """

    def __init__(self,
                 m: int,
                 layer_units: list[int] = (32, 16, 1),
                 **kwargs):
        super().__init__(**kwargs)
        self.m = m
        self.layer_units = layer_units
        # 创建m个神经网络，每个神经网络的输出是一个shape为(batch_size, 1)的张量
        assert isinstance(m, int) and m > 0, "m必须为正整数"
        assert (isinstance(layer_units, list) or isinstance(layer_units, tuple)) and len(
            layer_units) == m, "layer_units必须为长度为m的列表或元组"
        assert layer_units[-1] == 1, "最后一层神经元数量必须为1"
        self.time_distributed_embedding = [Embedding(layer_units=layer_units, last_activation="sigmoid")
                                           for _ in range(m)
                                           ]

    def call(self,
             last_flow,  # shape: (batch_size, m)
             last_free_water,  # shape: (batch_size, m)
             runoff,  # shape: (batch_size, m, n)
             ):
        # 计算产流量之和
        runoff_sum = ops.sum(runoff, axis=-1)  # shape: (batch_size, m)
        # 将产流量之和加到自由水状态上
        free_water = last_free_water + runoff_sum  # shape: (batch_size, m)
        # 计算每一层的汇流量比例
        flow_list = []
        free_water_list = []
        for i in range(len(self.time_distributed_embedding)):  # 对每一层进行计算
            # 计算汇流量比例
            # shape: (batch_size, n) -> (batch_size, 1) -> (batch_size, )
            free_water_ratio = self.time_distributed_embedding[i](runoff[:, i, :]).squeeze()
            # 将汇流量比例加与自由水状态相乘，得到本时段第i层的汇流量
            # shape: (batch_size, ) * (batch_size, ) + (batch_size, ) * (batch_size, ) -> (batch_size, )
            flow_list.append(free_water[:, i] * free_water_ratio + last_flow[:, i] * (1 - free_water_ratio))
            # 更新自由水状态
            # shape: (batch_size, ) - (batch_size, ) -> (batch_size, )
            free_water_list.append(free_water[:, i] - flow_list[i])
        # 将flow_list和new_free_water_list转换为张量，list中有m个大小为(batch_size, )的张量
        # shape: [(batch_size, ), (batch_size, ), ..., (batch_size, )] -> (batch_size, m)
        flow = ops.stack(flow_list, axis=-1)
        free_water = ops.stack(free_water_list, axis=-1)
        # 将m层的汇流量相加，得到本时段的总汇流量
        return free_water, flow  # shape: (batch_size, m), (batch_size, m)

    def get_config(self):
        return {"m": self.m,
                "layer_units": self.layer_units
                }


# %% 测试NonlinearReservoirCell 已通过测试
# import numpy as np
# from hydronetwork.utils import tensor2numpy
#
# batch_size = 2
# m = 3
# n = 4
# # 设置last_free_water和runoff全为正数
# last_free_water = np.random.random((batch_size, m))
# last_flow = np.random.random((batch_size, m))
# runoff = np.random.random((batch_size, m, n))
#
# test_nonlinear_reservoir_cell = NonlinearReservoirCell(m=3, layer_units=[32, 16, 1])
# free_water, flow = test_nonlinear_reservoir_cell(last_flow, last_free_water, runoff)
# free_water = tensor2numpy(free_water)
# flow = tensor2numpy(flow)

# %% 连续时间步的非线性水库模型

@keras.saving.register_keras_serializable(package='Custom', name='NonlinearReservoirModel')
class NonlinearReservoirModel(Layer):
    """
    非线性水库模型，用于连续时间步的水量计算。
    该模型的计算过程如下：
    1. 初始化自由水状态，即流域中每一层的自由水量。
        shape: (batch_size, m)
    2. 逐个时间步计算自由水状态。
        shape: (batch_size, m, n) -> (batch_size, m)
    3. 将每一时间步的自由水状态保存到列表中。
    4. 返回所有时间步的自由水状态。

    :param m: 流域中垂向上分层的数量，即土壤层数。该参数用于构建
    :param layer_units: 每层神经网络的神经元数量，注意最后一层神经元数量必须为1
    :param horizon: 最后需要计算flow的时间步数，即最后horizon个时间步的flow会被输出。
    """

    def __init__(self,
                 m: int,
                 layer_units: list[int],
                 horizon: int,
                 **kwargs):
        super().__init__(**kwargs)
        self.m = m
        self.layer_units = layer_units
        self.n_steps = horizon
        self.cell = NonlinearReservoirCell(m=m, layer_units=layer_units)

    def call(self,
             runoff,  # shape: (batch_size, T, m, n)
             ):
        # 初始化自由水状态
        free_water = ops.ones((runoff.shape[0], self.m)) * 0.5  # shape: (batch_size, m)
        flow_i = ops.ones((runoff.shape[0], self.m)) * 0.5  # shape: (batch_size, m)
        flow_list = []
        # 连续计算
        for i in range(runoff.shape[1]):
            # free_water, flow_i shape: (batch_size, m), (batch_size, )
            free_water, flow_i = self.cell(last_free_water=free_water,
                                           runoff=runoff[:, i, :, :],
                                           last_flow=flow_i)
            if i >= runoff.shape[1] - self.n_steps:
                flow_list.append(ops.sum(flow_i, axis=-1))
        # 将flow_list转换为张量，list中有n_steps个大小为(batch_size, )的张量
        # shape: [(batch_size, ), (batch_size, ), ..., (batch_size, )] -> (batch_size, horizon)
        return ops.stack(flow_list, axis=-1)

# # %% 测试NonlinearReservoir
# import numpy as np
# from hydronetwork.utils import tensor2numpy
#
# batch_size = 1
# T = 100
# m = 3
# n = 4
# n_steps = 7
# # 设置runoff全为正数
# runoff = np.random.random((batch_size, T, m, n))
# model = NonlinearReservoirModel(m=3, layer_units=[32, 16, 1], horizon=7)
# flow = model(runoff)
# flow = tensor2numpy(flow)
