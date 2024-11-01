import keras
from keras import Layer, ops, KerasTensor
from hydronetwork.model.layer.embedding import Embedding


@keras.saving.register_keras_serializable(package='Custom', name='NonlinearReservoir')
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
    3. 将汇流量比例加与自由水状态相乘，得到本时段每一层的汇流量
        shape: (batch_size, m) * (batch_size, m) -> (batch_size, m)
    4. 更新自由水状态
        shape: (batch_size, m) - (batch_size, m) -> (batch_size, m)

    :param m: 流域中垂向上分层的数量，即土壤层数。该参数用于构建
    :param layer_units: 每层神经网络的神经元数量，注意最后一层神经元数量必须为1
    """

    def __init__(self,
                 m: int,
                 layer_units: list[int],
                 **kwargs):
        super(NonlinearReservoirCell, self).__init__(**kwargs)
        # 创建m个神经网络，每个神经网络的输出是一个shape为(batch_size, 1)的张量
        assert isinstance(m, int) and m > 0, "m必须为正整数"
        assert isinstance(layer_units, list) and len(layer_units) == m, "layer_units必须为长度为m的列表"
        assert layer_units[-1] == 1, "最后一层神经元数量必须为1"
        self.time_distributed_embedding = [
            Embedding(layer_units=layer_units, last_activation="sigmoid") for i in range(m)
        ]

    def call(self,
             last_free_water: KerasTensor,  # shape: (batch_size, m)
             runoff: KerasTensor,  # shape: (batch_size, m, n)
             ):
        # 计算产流量之和
        runoff_sum = ops.sum(runoff, axis=-1)  # shape: (batch_size, m)
        # 将产流量之和加到自由水状态上
        free_water = last_free_water + runoff_sum  # shape: (batch_size, m)
        # 计算每一层的汇流量比例
        flow = ops.zeros_like(free_water)  # shape: (batch_size, m)
        for i in range(len(self.time_distributed_embedding)):
            # 计算汇流量比例
            # shape: (batch_size, 1, n) -> (batch_size, 1, 1)
            free_water_ratio = self.time_distributed_embedding[i](runoff[:, i, :])
            # 去除冗余维度
            # shape: (batch_size, 1, 1) -> (batch_size, )
            free_water_ratio = ops.squeeze(free_water_ratio)
            # 将汇流量比例加与自由水状态相乘，得到本时段每一层的汇流量
            # shape: (batch_size, ) * (batch_size, ) -> (batch_size, )
            flow[:, i] = free_water[:, i] * free_water_ratio
            # 更新自由水状态
            # shape: (batch_size, ) - (batch_size, ) -> (batch_size, )
            free_water[:, i] = free_water[:, i] - flow[:, i]
        return free_water, flow  # shape: (batch_size, m), (batch_size, m)


# %% 测试单元 已通过测试
# import numpy as np
# from hydronetwork.utils import tensor2numpy
#
# batch_size = 2
# m = 3
# n = 4
# # 设置last_free_water和runoff全为正数
# last_free_water = np.random.random((batch_size, m))
# runoff = np.random.random((batch_size, m, n))
#
# test_nonlinear_reservoir_cell = NonlinearReservoirCell(m=3, layer_units=[32, 16, 1])
# free_water, flow = test_nonlinear_reservoir_cell(last_free_water, runoff)
# free_water = tensor2numpy(free_water)
# flow = tensor2numpy(flow)
