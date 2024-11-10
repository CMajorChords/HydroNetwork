import keras
from keras import Layer, ops
from hydronetwork.model.layer.embedding import Embedding
from hydronetwork.model.add_weight.constraints import NonNegAndSumLimit


@keras.saving.register_keras_serializable(package='Custom', name='StoredFullRunoffCell')
class StoredFullRunoffCell(Layer):
    """
    产流模型单个时间步单层土壤的计算单元。

    输入为：
    1.上一个时间步输出的土壤含水量，shape: [batch_size, n]
    2.本时间步的降雨量，shape: [batch_size, 1]
    3.本时间步与蒸发相关的因子，shape: [batch_size, num_features]

    参数：
    张力水容量：[batch_size, n]
    自由水容量：[batch_size, n]
    下渗神经网络：[batch_size, n], 最后一层用sigmoid
    蒸发神经网络：[batch_size, num_features]，最后一层用sigmoid

    产流：
    输入：降水[batch_size,], 上一时段土壤张力水含水量[batch_size, n], 本层蒸发相关因子[batch_size, num_features]
    将降水量和张力水容量相加得到补充后的张力水含水量，[batch_size, n] + [batch_size, n] -> [batch_size, n]
    使用地表面蒸发神经网络计算地表面蒸发，[batch_size, num_features + n] -> [batch_size, n]
    用ops.where计算地表面自由水，[batch_size, n] - [batch_size, n] -> [batch_size, n]
    扣除自由水后，剩余的水量即为张力水，[batch_size, n] - [batch_size, n] -> [batch_size, n]

    分水源
    使用torch.where计算地表面自由水，[batch_size, n] - [batch_size, n] -> [batch_size, n]
    扣除地面径流后，剩余的水量即为地表面自由水，[batch_size, n] - [batch_size, n] -> [batch_size, n]
    使用壤中流网络计算地表面中层产流比例，调整数值约束比例到一定界限，[batch_size, n] -> [batch_size, n]
    将中层产流比例乘以地表自由水得到地表面中层产流量，[batch_size, n] * [batch_size, n] -> [batch_size, n]
    使用下渗网络计算自由水下渗比例，调整数值约束比例到一定界限，[batch_size, n] -> [batch_size, n]
    将下渗比例乘以地表自由水得到地表面下渗量，[batch_size, n] * [batch_size, n] -> [batch_size, n]
    地表面自由水-地表面下渗量=产流量

    :param n: 土壤纵向划分层数
    :param evap_net_units: 蒸发神经网络的隐藏层单元数，将自动添加一个输出层单元数为n，默认为空
    :param infilr_net_units: 下渗神经网络的隐藏层单元数，将自动添加一个输出层单元数为n，默认为空
    :param tension_water_capacity_max: 土壤张力水容量最大值 该值将作为约束张力水容量矩阵之和的最大值
    :param free_water_capacity_max: 土壤自由水容量最大值 该值将作为约束自由水容量矩阵之和的最大值
    :param evap_percent_limit: 土壤蒸发占总水量的最大比例
    :param middle_runoff_percent_limit: 土壤中层产流占总水量的最大比例
    """

    def __init__(self,
                 n: int,
                 evap_net_units: list = (),
                 infilr_net_units: list = (),
                 tension_water_capacity_limit: [float, float] = (120, 200),
                 free_water_capacity_limit: [float, float] = (5, 50),
                 evap_percent_limit: float = 0.2,
                 middle_runoff_percent_limit: float = 0.65,
                 deep_and_middle_runoff_percent_limit: float = 0.7,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.n = n
        self.tension_water_capacity_limit = tension_water_capacity_limit
        self.free_water_capacity_limit = free_water_capacity_limit
        self.evap_net_units = evap_net_units
        self.infilr_net_units = infilr_net_units
        self.evap_percent_limit = evap_percent_limit
        self.middle_runoff_percent_limit = middle_runoff_percent_limit
        self.deep_and_middle_runoff_percent_limit = deep_and_middle_runoff_percent_limit
        # 张力水容量
        self.tension_water_capacity = self.add_weight(name='tension_water_capacity',
                                                      shape=(n,),
                                                      trainable=True,
                                                      constraint=NonNegAndSumLimit(
                                                          min_value=tension_water_capacity_limit[0],
                                                          max_value=tension_water_capacity_limit[1]),
                                                      )
        # 自由水容量
        self.free_water_capacity = self.add_weight(name='free_water_capacity',
                                                   shape=(n,),
                                                   trainable=True,
                                                   constraint=NonNegAndSumLimit(
                                                       min_value=free_water_capacity_limit[0],
                                                       max_value=free_water_capacity_limit[1]),
                                                   )
        # 神经网络
        self.evap_net = Embedding(layer_units=list(evap_net_units) + [n], last_activation='sigmoid')
        self.middle_runoff_net = Embedding(layer_units=list(evap_net_units) + [n], last_activation='sigmoid')

    def call(self,
             precipitation,  # [batch_size, ]
             evaporation_features,  # [batch_size, num_features]
             last_tension_soil_water,  # [batch_size, ]
             last_free_soil_water,  # [batch_size, ]
             ):
        """
        计算单个时间步的产流量

        :param precipitation: 本时间步的降雨量，[batch_size, ]
        :param evaporation_features: 本层和蒸散发相关的特征，[batch_size, num_features]
        :param last_tension_soil_water: 上一个时间步的土壤张力水含水量比例，[batch_size, ]
        :param last_free_soil_water: 上一个时间步的土壤自由水含水量比例，[batch_size, ]
        :return: 该层各组分产流量，[batch_size, n]
        """
        # 产流
        # 将降水平均分为各层，[batch_size, ] -> [batch_size, n]
        precipitation = ops.repeat(precipitation.unsqueeze(-1) / self.n, self.n, axis=-1)
        # 将降水量和地表面以上张力水容量相加得到补充后的地表面含水量，[batch_size, n] + [batch_size, n] -> [batch_size, n]
        tension_soil_water = last_tension_soil_water + precipitation
        # 使用地表面蒸发神经网络计算地表面蒸发百分比，[batch_size, num_features] -> [batch_size, n]
        evaporation_ratio = self.evap_net(evaporation_features)
        # 将蒸发百分比乘以地表面含水量，[batch_size, n] * [batch_size, n] -> [batch_size, n]
        tension_soil_water = tension_soil_water * (
                1 - self.evap_percent_limit + evaporation_ratio * self.evap_percent_limit)
        # 将蒸发后该层含水量超过张力水容量的部分作为自由水，[batch_size, n] - [batch_size, n] -> [batch_size, n]
        free_soil_water = ops.where(tension_soil_water > self.tension_water_capacity,
                                    tension_soil_water - self.tension_water_capacity,
                                    ops.zeros_like(tension_soil_water))
        tension_soil_water = ops.minimum(tension_soil_water, self.tension_water_capacity)

        # 分水源
        # 将超过自由水容量的部分作为地面径流
        runoff_surface = ops.where(free_soil_water > self.free_water_capacity,
                                   free_soil_water - self.free_water_capacity,
                                   ops.zeros_like(free_soil_water))  # [batch_size, n]
        free_soil_water = ops.minimum(free_soil_water, self.free_water_capacity)  # [batch_size, n]
        # 计算壤中流、地下径流
        runoff_middle_ratio = self.middle_runoff_net(free_soil_water) * self.middle_runoff_percent_limit
        runoff_middle = runoff_middle_ratio * free_soil_water
        runoff_deep = (self.deep_and_middle_runoff_percent_limit - runoff_middle_ratio) * free_soil_water
        # 更新自由水000000000000000000000000000000000
        free_soil_water = free_soil_water * (1 - self.deep_and_middle_runoff_percent_limit)

        return tension_soil_water, free_soil_water, runoff_surface, runoff_middle, runoff_deep

    def get_config(self):
        return {'n': self.n,
                'tension_water_capacity_limit': self.tension_water_capacity_limit,
                'free_water_capacity_limit': self.free_water_capacity_limit,
                'evap_net_units': self.evap_net_units,
                'infilr_net_units': self.infilr_net_units,
                'evap_percent_limit': self.evap_percent_limit,
                'middle_runoff_percent_limit': self.middle_runoff_percent_limit,
                'deep_and_middle_runoff_percent_limit': self.deep_and_middle_runoff_percent_limit}


# %%测试RunoffProducingLayer 已完成测试
# import numpy as np
# from hydronetwork.utils import tensor2numpy
#
# n = 37
#
# layer = StoredFullRunoffCell(n=n, )
#
# last_tension_soil_water = np.random.random((512, n))
# last_free_soil_water = np.random.random((512, n))
# precipitation = np.random.random((512,))
# evaporation_features = np.random.random((512, 10))
#
# tension_soil_water, free_soil_water, runoff_surface, runoff_middle, runoff_deep = layer(
#     precipitation=precipitation,
#     evaporation_features=evaporation_features,
#     last_tension_soil_water=last_tension_soil_water,
#     last_free_soil_water=last_free_soil_water)
#
# tension_soil_water = tensor2numpy(tension_soil_water)
# free_soil_water = tensor2numpy(free_soil_water)
# runoff_surface = tensor2numpy(runoff_surface)
# runoff_middle = tensor2numpy(runoff_middle)
# runoff_deep = tensor2numpy(runoff_deep)


# %% 蓄满产流模型

class StoredFullRunoffModel(Layer):
    """
    蓄满产流模型，根据蓄满产流计算产流量。


    首先设置边界条件
    last_tension_soil_water,  # [batch_size, ] 全部为0
    last_free_soil_water,  # [batch_size, ] 全部为0

    产流
    将边界条件送入产流模型，逐个时间步计算产流量，最后的产流量被concat为一个大小为[batch_size, timestep]的张量
    一般模型中, timestep = lookback + horizon
    """

    def __init__(self,
                 n: int,
                 evap_net_units: list = (),
                 infilr_net_units: list = (),
                 tension_water_capacity_limit: [float, float] = (120, 200),
                 free_water_capacity_limit: [float, float] = (5, 50),
                 evap_percent_limit: float = 0.2,
                 middle_runoff_percent_limit: float = 0.65,
                 deep_and_middle_runoff_percent_limit: float = 0.7,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.n = n
        self.tension_water_capacity_limit = tension_water_capacity_limit
        self.free_water_capacity_limit = free_water_capacity_limit
        self.evap_net_units = evap_net_units
        self.infilr_net_units = infilr_net_units
        self.evap_percent_limit = evap_percent_limit
        self.middle_runoff_percent_limit = middle_runoff_percent_limit
        self.deep_and_middle_runoff_percent_limit = deep_and_middle_runoff_percent_limit
        self.stored_full_runoff_cell = StoredFullRunoffCell(n=n,
                                                            evap_net_units=evap_net_units,
                                                            infilr_net_units=infilr_net_units,
                                                            tension_water_capacity_limit=tension_water_capacity_limit,
                                                            free_water_capacity_limit=free_water_capacity_limit,
                                                            evap_percent_limit=evap_percent_limit,
                                                            middle_runoff_percent_limit=middle_runoff_percent_limit,
                                                            deep_and_middle_runoff_percent_limit=deep_and_middle_runoff_percent_limit)

    def call(self,
             forcing,  # shape: [batch_size, lookback+horizon, num_features+1]
             ):
        # 将输入转化为历史径流和降水、蒸发特征
        precipitation = forcing[:, :, 0]  # [batch_size, lookback+horizon]
        evaporation_features = forcing[:, :, 1:]  # [batch_size, lookback+horizon, num_features]

        # 初始化边界条件
        tension_soil_water = ops.zeros((forcing.shape[0], self.n))  # [batch_size, n] 张力水初始值
        free_soil_water = ops.zeros((forcing.shape[0], self.n))  # [batch_size, n] 自由水初始值

        # 产流
        runoff_surface_list = []
        runoff_middle_list = []
        runoff_deep_list = []
        for i in range(precipitation.shape[1]):  # shape[1] = lookback+horizon
            tension_soil_water, free_soil_water, runoff_surface, runoff_middle, runoff_deep = self.stored_full_runoff_cell(
                precipitation=precipitation[:, i],  # [batch_size,]
                evaporation_features=evaporation_features[:, i],  # [batch_size, num_features]
                last_tension_soil_water=tension_soil_water,  # [batch_size, n]
                last_free_soil_water=free_soil_water,  # [batch_size, n]
            )
            runoff_surface_list.append(runoff_surface)  # [batch_size, n] * (lookback+horizon)
            runoff_middle_list.append(runoff_middle)  # [batch_size, n] * (lookback+horizon)
            runoff_deep_list.append(runoff_deep)  # [batch_size, n] * (lookback+horizon)

        # 将产流量拼接为一个张量
        runoff_surface = ops.stack(runoff_surface_list, axis=1)  # [batch_size, lookback+horizon, n]
        runoff_middle = ops.stack(runoff_middle_list, axis=1)  # [batch_size, lookback+horizon, n]
        runoff_deep = ops.stack(runoff_deep_list, axis=1)  # [batch_size, lookback+horizon, n]

        return runoff_surface, runoff_middle, runoff_deep

    def get_config(self):
        return {'n': self.n,
                'tension_water_capacity_limit': self.tension_water_capacity_limit,
                'free_water_capacity_limit': self.free_water_capacity_limit,
                'evap_net_units': self.evap_net_units,
                'infilr_net_units': self.infilr_net_units,
                'evap_percent_limit': self.evap_percent_limit,
                'middle_runoff_percent_limit': self.middle_runoff_percent_limit,
                'deep_and_middle_runoff_percent_limit': self.deep_and_middle_runoff_percent_limit}

# %%测试StoredFullRunoffModel
# import numpy as np
# import keras
# from hydronetwork.utils import tensor2numpy, reset
#
# n = 37
# evap_units = [64, 32]
# infilr_units = [64, 32]
# model = StoredFullRunoffModel(n=n,
#                               evap_net_units=evap_units,
#                               infilr_net_units=infilr_units)
#
# forcing = np.random.random((512, 30, 4))
#
# runoff_surface, runoff_middle, runoff_deep = model(forcing)
# runoff_surface = tensor2numpy(runoff_surface)
# runoff_middle = tensor2numpy(runoff_middle)
# runoff_deep = tensor2numpy(runoff_deep)
