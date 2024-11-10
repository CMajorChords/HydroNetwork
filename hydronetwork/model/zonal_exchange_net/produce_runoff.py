import keras
from keras import Layer, ops
from hydronetwork.model.layer.embedding import Embedding
# from hydronetwork.model.zonal_exchange_net.mix_water import WaterMixHead
from hydronetwork.model.add_weight.constraints import NonNegAndSumLimit


@keras.saving.register_keras_serializable(package='Custom', name='RunoffProducingCell')
class RunoffProducingCell(Layer):
    """
    产流模型单个时间步的计算单元。
    产流模型的输入为：
    1.上一个时间步输出的土壤含水量，shape: [batch_size, m, n]
    2.本时间步的降雨量，shape: [batch_size, 1]
    3.本时间步与蒸发相关的因子，shape: [batch_size, num_features]
    产流模型自身增加的可学习参数为
        土壤蓄水容量，shape: [m, n]

    计算过程：
    1.前处理
        将降水量平均分为n*s份，s为水量混合步数，[batch_size, ] -> [batch_size, 1, n]
    2.水量混合
        扣除蒸散发计算：
            将与蒸发相关的特征时段初土壤含水量flatten，经过一个全连接层后reshape得到扣除蒸散发的土壤含水量比例
            [batch_size, m*n+num_features] -> [batch_size, m*n] -> [batch_size, m, n] # 使用sigmoid
            将该比例乘以时段初土壤含水量得到扣除蒸散发的土壤含水量，[batch_size, m, n] * [batch_size, m, n] -> [batch_size, m, n]
        降水（下渗）计算：
            使用水量混合头计算降水量下渗的比例，[batch_size, m, n] -> [batch_size, m, n] # 使用softmax
            乘以降水量得到降水在每个土块上的分配，[batch_size, m, n] * [batch_size, ] -> [batch_size, m, n]
        补充土壤含水量：
            将降水量和扣除蒸散发的土壤含水量相加得到补充后的土壤含水量，[batch_size, m, n] + [batch_size, m, n] -> [batch_size, m, n]
            对补充后的土壤含水量与土壤蓄水容量取最小值，得到时段末土壤含水量，[batch_size, m, n] -> [batch_size, m, n]
    3.水量平衡计算产流量
        计算每个土块的产流量，产流量=降水量（下渗）-（补充后的土壤含水量-扣除蒸散发的土壤含水量）
        [batch_size, m, n]= [batch_size, m, n] - （[batch_size, m, n] - [batch_size, m, n]）

    :param m: 土壤层数
    :param n: 土壤纵向划分层数
    :param n_mix_steps: 水量混合头中的计算步数，默认为3。
    :param water_capacity_max: 单个土块的最大蓄水容量
    """

    def __init__(self,
                 m: int,
                 n: int,
                 n_mix_steps: int = 1,
                 water_capacity_max=120,
                 **kwargs
                 ):
        super(RunoffProducingCell, self).__init__(**kwargs)
        self.m = m
        self.n = n
        self.n_mix_steps = n_mix_steps
        self.water_capacity_max = water_capacity_max
        # 土壤张力水容量, shape = [m, n]
        self.water_capacity = self.add_weight(shape=(m, n),
                                              trainable=True,
                                              constraint=NonNegAndSumLimit(max_value=water_capacity_max,
                                                                           axis=[-1, -2]),
                                              name='water_capacity'
                                              )
        # 蒸散发计算参数
        self.dense_evaporation = Embedding(layer_units=[m * n,
                                                        m * n,
                                                        m * n],
                                           last_activation='sigmoid')
        # 降水计算参数
        # self.head_infiltration = WaterMixHead(m=m, n=n,
        #                                       normalization='linear_normalize',
        #                                       n_mix_steps=n_mix_steps)
        self.dense_precipitation = Embedding(layer_units=[m * n,
                                                          m * n,
                                                          m * n],
                                             last_activation='softmax')

    def call(self,
             last_soil_water,  # [batch_size, m, n]
             precipitation,  # [batch_size, ]
             evaporation_features,  # [batch_size, num_features]
             ):
        """
        计算单个时间步的产流量

        :param last_soil_water: 上一个时间步的土壤含水量比例，size=batch_size*m*n
        :param precipitation: 本时间步的降雨量，size=batch_size
        :param evaporation_features: 本时间步和蒸散发相关的特征，size=batch_size
        :return: 每一层的产流量，size=batch_size*m
        """
        # 前处理
        # 将降水量除以(n*n_mix_steps)，得到每一切片每一次的降水量 [batch_size, ] -> [batch_size, 1, n]
        # precipitation_progress = precipitation / self.n / self.n_mix_steps  # [batch_size, ]
        # precipitation_progress = precipitation_progress.unsqueeze(-1).unsqueeze(-1)  # [batch_size, 1, 1]
        # precipitation_progress = ops.repeat(precipitation_progress,
        #                                     repeats=self.n,
        #                                     axis=-1)  # shape = [batch_size, 1, n]

        # 扣除蒸散发计算
        # 将与蒸发相关的特征时段初土壤含水量flatten，经过一个全连接层后reshape得到扣除蒸散发的土壤含水量比例
        evaporation_features = ops.concatenate(
            [last_soil_water.reshape((last_soil_water.shape[0], -1)),  # [batch_size, m*n]
             evaporation_features],  # [batch_size, num_features]
            axis=-1)  # [batch_size, m*n+num_features]
        # 将该比例乘以时段初土壤含水量得到扣除蒸散发的土壤含水量，[batch_size, m, n] * [batch_size, m, n] -> [batch_size, m, n]
        soil_water_ratio_evap = self.dense_evaporation(evaporation_features)  # [batch_size, m*n]
        soil_water_after_evap = soil_water_ratio_evap.reshape(last_soil_water.shape) * last_soil_water  # [batch_size, m, n]

        # 降水计算
        # 使用水量混合头计算降水后土壤水的含量，[batch_size, m, n] -> [batch_size, m, n]
        # soil_water_after_infiltration = self.head_infiltration(soil_water=soil_water_after_evap,
        #                                                        precipitation=precipitation_progress,
        #                                                        water_capacity=self.water_capacity.unsqueeze(0),
        #                                                        )
        precipitation_features = ops.concatenate([soil_water_after_evap.reshape((soil_water_after_evap.shape[0], -1)),
                                                  precipitation.unsqueeze(-1)],  # [batch_size, m*n+1]
                                                 axis=-1)
        soil_water_ratio_infiltration = self.dense_precipitation(precipitation_features)
        infiltration_distribution = soil_water_ratio_infiltration.reshape(last_soil_water.shape) * precipitation.unsqueeze(-1).unsqueeze(-1)  # [batch_size, m, n]

        # 补充土壤含水量
        soil_water_after_infiltration = soil_water_after_evap + infiltration_distribution
        soil_water = ops.minimum(soil_water_after_infiltration, self.water_capacity)  # [batch_size, m, n]

        # 计算产流量
        runoff = infiltration_distribution - (soil_water - soil_water_after_evap)  # [batch_size, m, n]
        return runoff, soil_water  # add evaporation for test

    def get_config(self):
        return {"m": self.m,
                "n": self.n,
                "n_mix_steps": self.n_mix_steps,
                "water_capacity_max": self.water_capacity_max}


# %%测试RunoffProducingCell
# import numpy as np
# from hydronetwork.utils import tensor2numpy
#
# m = 3
# n = 64
# n_mix_steps = 3
# batch_size = 1
# num_features = 5
# cell = RunoffProducingCell(m=m, n=n, n_mix_steps=n_mix_steps)
# last_soil_water = np.random.rand(batch_size, m, n)
# # 保证last_soil_water_ratio在0-10之间
# last_soil_water = np.clip(last_soil_water, 0, 10)
# precipitation = np.random.rand(batch_size, )
# evaporation_features = np.random.rand(batch_size, num_features)
# runoff, soil_water = cell(last_soil_water=last_soil_water,
#                           precipitation=precipitation,
#                           evaporation_features=evaporation_features)
# runoff = tensor2numpy(runoff)
# soil_water = tensor2numpy(soil_water)


# %%
@keras.saving.register_keras_serializable(package='Custom', name='RunoffProducingModel')
class RunoffProducingModel(Layer):
    """
    产流模型，用于计算整个时间序列的产流量
    :param m: 土壤层数
    :param n: 土壤纵向划分层数
    :param water_capacity_max: 单个土块的最大蓄水容量
    """

    def __init__(self,
                 m: int,
                 n: int,
                 n_mix_steps: int = 1,
                 water_capacity_max=120,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.m = m
        self.n = n
        self.n_mix_steps = n_mix_steps
        self.water_capacity_max = water_capacity_max
        self.cell = RunoffProducingCell(m=m, n=n, n_mix_steps=n_mix_steps, water_capacity_max=water_capacity_max)

    def call(self,
             precipitation,  # size=batch_size*T
             evaporation_features,  # size=batch_size*T
             ):
        """
        计算整个时间序列的产流量
        :param precipitation: 降雨量序列，size=batch_size*T
        :param evaporation_features: 蒸发量序列，size=batch_size*T
        :return: 产流量序列，size=batch_size*T*m
        """
        # 初始化土壤含水量比例
        soil_water = ops.ones((precipitation.shape[0], self.m, self.n)) * 0.5  # [batch_size, m, n]
        # 连续计算每个时间步的产流量
        runoff_list = []
        for t in range(precipitation.shape[1]):
            runoff, soil_water = self.cell(last_soil_water=soil_water,  # [batch_size, m, n]
                                           precipitation=precipitation[:, t],  # [batch_size, ]
                                           evaporation_features=evaporation_features[:, t])  # [batch_size, ]
            runoff_list.append(runoff)
        # 将产流量序列连接 shape：[[batch_size, m, n], [batch_size, m, n], ...] -> [batch_size, T, m, n]
        runoff = ops.stack(runoff_list, axis=1)
        return runoff

# %% 测试RunoffProducingCell 测试通过
# import numpy as np
# from hydronetwork.utils import tensor2numpy
#
# batchsize = 512
# T = 10
# m = 3
# n = 64
# n_mix_steps = 3
# model = RunoffProducingModel(m=m, n=n, n_mix_steps=n_mix_steps)
# precipitation = np.random.rand(batchsize, T)
# evaporation_features = np.random.rand(batchsize, T, 5)
# runoff = model(precipitation=precipitation, evaporation_features=evaporation_features)
# runoff = tensor2numpy(runoff)
