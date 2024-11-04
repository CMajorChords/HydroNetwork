# 空间混合网络
import keras
from keras import Model
from hydronetwork.model.layer.nonlinear_reservoir import NonlinearReservoirModel as ZENConfluence
from hydronetwork.model.zonal_exchange_net.produce_runoff import RunoffProducingModel as ZENRunoffProducing


@keras.saving.register_keras_serializable(package='Custom', name='ZonalExchangeNet')
class ZonalExchangeNet(Model):
    """
    空间混合网络，用于计算流域中的汇流过程。
    分为产流和汇流两个部分，其中产流部分使用空间混合网络，汇流部分使用非线性水库模型。

    :param m: 流域中垂向上分层的数量，即土壤层数。
    :param n: 土壤纵向划分层数
    :param n_mix_steps: 产流模型中的水量混合头的计算次数
    :param horizon: 最后需要计算flow的时间步数，即最后n_steps个时间步的flow会被输出。
    :param water_capacity_max: 单个土块的最大蓄水容量
    :param layer_units: 非线性水库中每层神经网络的神经元数量，注意最后一层神经元数量必须为1
    """

    def __init__(self,
                 m: int,
                 n: int,
                 horizon: int,
                 n_mix_steps: int = 3,
                 water_capacity_max: int = 120,
                 layer_units: list[int] = (32, 16, 1),
                 **kwargs):
        super().__init__(**kwargs)
        self.m = m
        self.n = n
        self.n_mix_steps = n_mix_steps
        self.water_capacity_max = water_capacity_max
        self.layer_units = layer_units
        self.confluence = ZENConfluence(m=m, layer_units=layer_units, horizon=horizon)
        self.runoff_producing = ZENRunoffProducing(m=m, n=n, water_capacity_max=water_capacity_max)

    def call(self,
             # precipitation,  # size=batch_size*T
             # potential_evaporation,  # size=batch_size*T
             inputs  # size=batch_size*T*2
             ):
        precipitation = inputs[:, :, 0]
        potential_evaporation = inputs[:, :, 1]
        # 计算产流过程
        runoff = self.runoff_producing(precipitation, potential_evaporation)
        # 计算汇流过程
        return self.confluence(runoff)
