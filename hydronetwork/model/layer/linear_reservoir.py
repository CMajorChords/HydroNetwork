import keras
from keras import Layer, ops
from hydronetwork.model.layer.embedding import Embedding


@keras.saving.register_keras_serializable(package='Custom', name='LinearReservoirCell')
class LinearReservoirCell(Layer):
    """
    非线性水库模型的单个时间步计算单元。
    流域自由水状态是一个shape为(batch_size, m)的张量。m表示流域中垂向上分层的数量。
    该模型的计算过程如下：
    1. 将该时段的产流量通过三个网络分别得到一个汇流量特征 [batch_size, 1] * 3
    2. 将这三个特征通过一个线性层和激活函数得到一个汇流量比例 [batch_size, 1]
    3. flow = 所有成分的径流总量 * 汇流量比例 + （1-汇流量比例）* last_flow

    :param activation: 转换各个径流成分的激活函数，应该为relu或者
    :param layer_units: 每个网络的神经元数量

    """

    def __init__(self,
                 layer_units: list[int] = (32, 16, 1),
                 activation: str = "relu",
                 **kwargs):
        super().__init__(**kwargs)
        self.layer_units = layer_units
        self.activation = activation
        self.surface_runoff_net = Embedding(layer_units=layer_units, last_activation=activation)
        self.subsurface_runoff_net = Embedding(layer_units=layer_units, last_activation=activation)
        self.deep_runoff_net = Embedding(layer_units=layer_units, last_activation=activation)
        self.linear_layer = Embedding(layer_units=[3, 3, 1], last_activation="sigmoid")

    def call(self,
             runoff_surface,  # shape: (batch_size, n)
             runoff_middle,  # shape: (batch_size, n)
             runoff_deep,  # shape: (batch_size, n)
             last_flow,  # shape: (batch_size, 1)
             ):
        # 计算产流量之和
        runoff_sum = runoff_surface + runoff_middle + runoff_deep  # shape: (batch_size, n)
        runoff_sum = ops.sum(runoff_sum, axis=-1, keepdims=True)  # shape: (batch_size, 1)
        # 计算汇流量比例
        surface_features = self.surface_runoff_net(runoff_surface)  # shape: (batch_size, 1)
        middle_features = self.subsurface_runoff_net(runoff_middle)  # shape: (batch_size, 1)
        deep_features = self.deep_runoff_net(runoff_deep)  # shape: (batch_size, 1)
        # 将这三个比例通过一个线性层和激活函数得到一个汇流量比例 # shape: (batch_size, 3)
        runoff_feature = ops.concatenate([surface_features, middle_features, deep_features], axis=-1)
        flow_ratio = self.linear_layer(runoff_feature)  # shape: (batch_size, 1)

        # 计算汇流量
        return runoff_sum * flow_ratio + last_flow * (1 - flow_ratio)  # shape: (batch_size, 1)

    def get_config(self):
        return {'layer_units': self.layer_units,
                'activation': self.activation}


# %% 测试linearReservoirCell 已通过测试
# import numpy as np
# from hydronetwork.utils import tensor2numpy
#
# batch_size = 512
# n = 64
# layer_units = [32, 16, 1]
# activation = "relu"
# last_flow = np.random.random((batch_size, 1))
# linear_reservoir = LinearReservoirCell(layer_units=layer_units, activation=activation)
# runoff_surface = np.random.random((batch_size, n))
# runoff_middle = np.random.random((batch_size, n))
# runoff_deep = np.random.random((batch_size, n))
# flow = linear_reservoir(runoff_surface, runoff_middle, runoff_deep, last_flow)


# %% 线性水库模型
class LinearReservoirModel(Layer):
    def __init__(self,
                 layer_units: list[int] = (32, 16, 1),
                 activation: str = "relu",
                 **kwargs):
        super().__init__(**kwargs)
        self.layer_units = layer_units
        self.activation = activation
        self.linear_reservoir = LinearReservoirCell(layer_units=layer_units,
                                                    activation=activation, )

    def call(self,
             streamflow,  # shape: (batch_size, lookback)
             runoff_surface,  # shape: (batch_size, horizon, n)
             runoff_middle,  # shape: (batch_size, horizon, n)
             runoff_deep  # shape: (batch_size, horizon, n)
             ):
        # 计算horizon
        horizon = runoff_deep.shape[1]
        # 汇流边界条件
        flow = streamflow[:, -1].unsqueeze(-1)
        flow_list = []
        # 汇流
        for i in range(horizon):
            # 从horizon开始计算flow
            flow = self.linear_reservoir(
                runoff_surface=runoff_surface[:, i],
                runoff_middle=runoff_middle[:, i],
                runoff_deep=runoff_deep[:, i],
                last_flow=flow)
            flow_list.append(flow)  # shape: [batch_size, 1] * horizon
        return ops.concatenate(flow_list, axis=-1)  # shape: [batch_size, horizon]


# %% 测试LinearReservoirModel
# import numpy as np
# from hydronetwork.utils import tensor2numpy
#
# batch_size = 512
# n = 64
# T= 10
# layer_units = [32, 16, 1]
# activation = "relu"
# linear_reservoir_model = LinearReservoirModel(layer_units=layer_units, activation=activation)
# streamflow = np.random.random((batch_size, 10))
# runoff_surface = np.random.random((batch_size, 10, n))
# runoff_middle = np.random.random((batch_size, 10, n))
# runoff_deep = np.random.random((batch_size, 10, n))
#
# flow = linear_reservoir_model(streamflow, runoff_surface, runoff_middle, runoff_deep)
# flow = tensor2numpy(flow)
