# 新安江神经网络模型
import keras
from keras import Model
from hydronetwork.model.layer.linear_reservoir import LinearReservoirModel
from hydronetwork.model.xinanjiang_net.stored_full_runoff import StoredFullRunoffModel


@keras.saving.register_keras_serializable(package='Custom', name='XinAnJiangNet')
class XinAnJiangNet(Model):
    """
    新安江神经网络，搭配蓄满产流网络和线性水库模型。

    :param m: 流域中垂向上分层的数量，即土壤层数。
    :param n: 土壤纵向划分层数
    :param n_mix_steps: 产流模型中的水量混合头的计算次数
    :param horizon: 最后需要计算flow的时间步数，即最后n_steps个时间步的flow会被输出。
    :param water_capacity_max: 单个土块的最大蓄水容量
    :param layer_units: 非线性水库中每层神经网络的神经元数量，注意最后一层神经元数量必须为1
    """

    def __init__(self,
                 # 产流模型参数
                 n: int,
                 evap_net_units: list = (),
                 infilr_net_units: list = (),
                 tension_water_capacity_limit: [float, float] = (120, 200),
                 free_water_capacity_limit: [float, float] = (5, 50),
                 evap_percent_limit: float = 0.2,
                 middle_runoff_percent_limit: float = 0.65,
                 deep_and_middle_runoff_percent_limit: float = 0.7,
                 # 汇流模型参数
                 layer_units: list[int] = (32, 16, 1),
                 activation: str = "relu",
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.n = n
        self.evap_net_units = evap_net_units
        self.infilr_net_units = infilr_net_units
        self.tension_water_capacity_limit = tension_water_capacity_limit
        self.free_water_capacity_limit = free_water_capacity_limit
        self.evap_percent_limit = evap_percent_limit
        self.middle_runoff_percent_limit = middle_runoff_percent_limit
        self.deep_and_middle_runoff_percent_limit = deep_and_middle_runoff_percent_limit
        self.layer_units = layer_units
        self.activation = activation
        # 初始化产流模型
        self.stored_full_runoff = StoredFullRunoffModel(n=n,
                                                        evap_net_units=evap_net_units,
                                                        infilr_net_units=infilr_net_units,
                                                        tension_water_capacity_limit=tension_water_capacity_limit,
                                                        free_water_capacity_limit=free_water_capacity_limit,
                                                        evap_percent_limit=evap_percent_limit,
                                                        middle_runoff_percent_limit=middle_runoff_percent_limit,
                                                        deep_and_middle_runoff_percent_limit=deep_and_middle_runoff_percent_limit,
                                                        **kwargs)
        # 初始化汇流模型
        self.linear_reservoir = LinearReservoirModel(layer_units=layer_units,
                                                     activation=activation,
                                                     **kwargs)


    def call(self,
             inputs  # shape: ([batch_size, lookback], [batch_size, lookback+horizon, num_features+1])
             ):
        streamflow = inputs[0]  # shape: [batch_size, lookback]
        forcing = inputs[1]  # shape: [batch_size, lookback+horizon, num_features+1]
        # 计算产流
        runoff_surface, runoff_middle, runoff_deep = self.stored_full_runoff(forcing)
        # 只截取最后horizon个时间步的产流作为输入
        horizon = forcing.shape[1] - streamflow.shape[1]
        runoff_surface = runoff_surface[:, -horizon:, :]
        runoff_middle = runoff_middle[:, -horizon:, :]
        runoff_deep = runoff_deep[:, -horizon:, :]
        # 计算汇流
        flow = self.linear_reservoir(streamflow=streamflow,
                                     runoff_surface=runoff_surface,
                                     runoff_middle=runoff_middle,
                                     runoff_deep=runoff_deep)
        return flow

    def get_config(self):
        return {'n': self.n,
                'evap_net_units': self.evap_net_units,
                'infilr_net_units': self.infilr_net_units,
                'tension_water_capacity_limit': self.tension_water_capacity_limit,
                'free_water_capacity_limit': self.free_water_capacity_limit,
                'evap_percent_limit': self.evap_percent_limit,
                'middle_runoff_percent_limit': self.middle_runoff_percent_limit,
                'deep_and_middle_runoff_percent_limit': self.deep_and_middle_runoff_percent_limit,
                'layer_units': self.layer_units,
                'activation': self.activation}


# %% 测试XAJNet
# import numpy as np
# from hydronetwork.utils import tensor2numpy
#
# n = 37
# lookback = 365
# horizon = 4
# num_features = 3
# input_forcing = np.random.random((512, lookback + horizon, num_features + 1))
# input_streamflow = np.random.random((512, lookback))
# model = XinAnJiangNet(n=n)
# output = model([input_streamflow, input_forcing])
# output = tensor2numpy(output)
