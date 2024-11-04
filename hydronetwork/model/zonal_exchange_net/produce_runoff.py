import keras
from keras import Layer, ops
from hydronetwork.model.layer.embedding import Embedding
from hydronetwork.model.zonal_exchange_net.mix_water import WaterMixHead
from hydronetwork.model.add_weight.constraints import MinMaxL1Norm


@keras.saving.register_keras_serializable(package='Custom', name='RunoffProducingCell')
class RunoffProducingCell(Layer):
    """
    产流模型单个时间步的计算单元。
    产流模型的输入为：
    1.上一个时间步输出的土壤含水量比例，shape: [batch_size, m, n]
    2.本时间步的降雨量，shape: [batch_size, 1]
    3.本时间步的潜在蒸散发量，shape: [batch_size, 1]
    产流模型自身增加的可学习参数为
        土壤蓄水容量，shape: [m, n]

    计算过程：
    1.前处理
        将上一个时间步的土壤含水量比例乘以土壤蓄水容量，得到土壤含水量，
            土壤蓄水容量是一个可学习的参数矩阵，[batch_size, m, n] -> [batch_size, m, n]
        将降水量平均分为n*s份，s为水量混合步数，[batch_size, ] -> [batch_size, 1, n]
    2.水量混合
        时段末土壤含水量计算：
            使用水量混合头计算时段末土壤含水量比例，[batch_size, m, n] -> [batch_size, m, n] # 使用sigmoid
        降水计算：
            使用水量混合头计算降水量下渗的比例，[batch_size, m, n] -> [batch_size, m, n] # 使用softmax
            乘以降水量得到降水在每个土块上的分配，[batch_size, m, n] * [batch_size, ] -> [batch_size, m, n]
        蒸散发计算：
            将土壤蓄水容量flatten并与总潜在蒸散发concatenate，[batch_size, m, n] -> [batch_size, m*n+1]
            将concatenate后的向量输入到一个全连接层，最后一层的激活函数为softmax，[batch_size, m*n+1] -> [batch_size, m*n]
            将输出reshape为[batch_size, m, n]，[batch_size, m*n] -> [batch_size, m, n]
            计算每个土块的潜在蒸散发量，[batch_size, m, n] * [batch_size, m, n] -> [batch_size, m, n]
            使用水量混合头计算每个土块的蒸散发比例，[batch_size, m, n] -> [batch_size, m, n] # 使用sigmoid
            乘以潜在蒸发量得到每个土块的蒸散发量，[batch_size, m, n] * [batch_size, m, n] -> [batch_size, m, n]
    3.水量平衡计算
        计算每个土块的产流量，产流量=降水量（下渗）-蒸散发量-土壤含水量变化量=降水量-蒸散发量-(时段末土壤含水量-时段初土壤含水量)*土壤蓄水容量
        [batch_size, m, n] - [batch_size, m, n] - ([batch_size, m, n] - [batch_size, m, n]) * [batch_size, m, n]
        -> [batch_size, m, n]

    :param m: 土壤层数
    :param n: 土壤纵向划分层数
    :param n_mix_steps: 水量混合头中的计算步数，默认为3。
    :param water_capacity_max: 单个土块的最大蓄水容量
    """

    def __init__(self,
                 m: int,
                 n: int,
                 n_mix_steps: int = 3,
                 water_capacity_max=120,
                 **kwargs
                 ):
        super(RunoffProducingCell, self).__init__(**kwargs)
        self.m = m
        self.n = n
        self.n_mix_steps = n_mix_steps
        self.water_capacity_max = water_capacity_max
        # 土壤蓄水容量, shape = [m, n]
        self.water_capacity = self.add_weight(shape=(m, n),
                                              trainable=True,
                                              constraint=MinMaxL1Norm(min_value=0, max_value=water_capacity_max),
                                              )
        # 时段末土壤含水量计算参数
        self.head_water_ratio = WaterMixHead(m=m, n=n, normalization='sigmoid', n_mix_steps=n_mix_steps)
        # 降水计算参数
        self.head_infiltration = WaterMixHead(m=m, n=n, normalization='softmax', n_mix_steps=n_mix_steps)
        # 蒸散发计算参数
        self.distribute_potential_evaporation = Embedding(layer_units=[m * n + 1,
                                                                       m * n],
                                                          last_activation='softmax')
        self.head_evaporation_potential = WaterMixHead(m=m, n=n, normalization='softmax', n_mix_steps=n_mix_steps)

    def call(self,
             last_soil_water_ratio,  # [batch_size, m, n]
             precipitation,  # [batch_size, ]
             potential_evaporation,  # [batch_size, ]
             ):
        """
        计算单个时间步的产流量

        :param last_soil_water_ratio: 上一个时间步的土壤含水量比例，size=batch_size*m*n
        :param precipitation: 本时间步的降雨量，size=batch_size*1
        :param potential_evaporation: 本时间步的潜在蒸散发量，size=batch_size*1
        :return: 每一层的产流量，size=batch_size*m
        """
        # 前处理
        #
        # 将上一个时间步的土壤含水量比例乘以土壤蓄水容量，得到土壤含水量，
        last_soil_water = last_soil_water_ratio * self.water_capacity  # [batch_size, m, n]
        # 将降水量除以(n*n_mix_steps)，得到每一切片每一次的降水量 [batch_size, ] -> [batch_size, 1, n]
        precipitation_progress = precipitation / self.n / self.n_mix_steps  # [batch_size, ]
        precipitation_progress = precipitation_progress.unsqueeze(-1).unsqueeze(-1)  # [batch_size, 1, 1]
        precipitation_progress = ops.repeat(precipitation_progress,
                                            repeats=self.n,
                                            axis=-1)  # shape = [batch_size, 1, n]

        # 水量混合
        # 时段末土壤含水量计算,使用水量混合头计算时段末土壤含水量比例，[batch_size, m, n] -> [batch_size, m, n]
        soil_water_ratio = self.head_water_ratio(soil_water=last_soil_water, precipitation=precipitation_progress)
        #
        #
        # 降水计算
        # 使用水量混合头计算降水量下渗的比例，[batch_size, m, n] -> [batch_size, m, n]
        infiltration_ratio = self.head_infiltration(soil_water=last_soil_water, precipitation=precipitation_progress)
        # 乘以降水量得到降水在每个土块上的分配，
        # [batch_size, m, n] * [batch_size, ] -> [batch_size, m, n] * [batch_size, 1, 1] -> [batch_size, m, n]
        infiltration = infiltration_ratio * precipitation.unsqueeze(-1).unsqueeze(-1)
        #
        #
        # 蒸散发计算
        # 将土壤蓄水容量flatten并添加batch_size维度，[m, n] -> [m*n] -> [batch_size, m*n]
        soil_water_flatten = ops.repeat(ops.reshape(self.water_capacity, (1, -1)),  # [1, m*n]
                                        repeats=last_soil_water.shape[0],  # batch_size
                                        axis=0)  # [batch_size, m*n]
        # 将土壤蓄水容量flatten并与总潜在蒸散发concatenate，[batch_size, m*n] -> [batch_size, m*n+1]
        potential_evaporation = potential_evaporation.unsqueeze(-1)  # [batch_size, 1]
        evaporation_concat = ops.concatenate([soil_water_flatten, potential_evaporation],
                                             axis=-1)  # [batch_size, m*n+1]
        # 将concatenate后的向量输入到一个全连接层后reshape，[batch_size, m*n+1] -> [batch_size, m, n]
        potential_evaporation_ratio = self.distribute_potential_evaporation(evaporation_concat)  # [batch_size, m*n]
        potential_evaporation_ratio = potential_evaporation_ratio.reshape(last_soil_water.shape)  # [batch_size, m, n]
        # 计算每个土块的潜在蒸散发量，[batch_size, m, n] * [batch_size, m, n] -> [batch_size, m, n]
        potential_evaporation = potential_evaporation_ratio * potential_evaporation.unsqueeze(-1)  # [batch_size, m, n]
        # 使用水量混合头计算每个土块的蒸散发比例，[batch_size, m, n] -> [batch_size, m, n]
        evaporation_ratio = self.head_evaporation_potential(soil_water=last_soil_water,
                                                            precipitation=precipitation_progress)
        # 乘以潜在蒸发量得到每个土块的蒸散发量，[batch_size, m, n] * [batch_size, m, n] -> [batch_size, m, n]
        evaporation = evaporation_ratio * potential_evaporation  # [batch_size, m, n]

        # 计算产流量
        # 计算每个土块的产流量，产流量=降水量（下渗）-蒸散发量-土壤含水量变化量
        # [batch_size, m, n]
        runoff = infiltration - evaporation - (soil_water_ratio - last_soil_water_ratio) * self.water_capacity
        return runoff, soil_water_ratio  # add evaporation for test

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
# cell = RunoffProducingCell(m=m, n=n, n_mix_steps=n_mix_steps)
# last_soil_water_ratio = np.random.rand(batch_size, m, n)
# # 保证last_soil_water_ratio在0-1之间
# last_soil_water_ratio = np.clip(last_soil_water_ratio, 0, 1)
# precipitation = np.random.rand(batch_size, )
# potential_evaporation = np.random.rand(batch_size, )
# runoff, soil_water_ratio, evaporation = cell(last_soil_water_ratio=last_soil_water_ratio,
#                                              precipitation=precipitation,
#                                              potential_evaporation=potential_evaporation)
# runoff = tensor2numpy(runoff)
# soil_water_ratio = tensor2numpy(soil_water_ratio)
# evaporation = tensor2numpy(evaporation)
#
# # 验证水量平衡
# # 计算总蒸发
# print(np.sum(evaporation))
# # 计算土壤含水量变化
# print(np.sum((last_soil_water_ratio - soil_water_ratio) * cell.water_capacity))
# # 水量平衡的产流
# print(precipitation - np.sum(evaporation) - np.sum((soil_water_ratio - last_soil_water_ratio) * cell.water_capacity))
# # 计算总产流
# print(np.sum(runoff))


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
                 n_mix_steps: int = 3,
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
             potential_evaporation,  # size=batch_size*T
             ):
        """
        计算整个时间序列的产流量
        :param precipitation: 降雨量序列，size=batch_size*T
        :param potential_evaporation: 蒸发量序列，size=batch_size*T
        :return: 产流量序列，size=batch_size*T*m
        """
        # 初始化土壤含水量比例
        soil_water_ratio = ops.ones((precipitation.shape[0], self.m, self.n)) * 0.5  # [batch_size, m, n]
        # 连续计算每个时间步的产流量
        runoff_list = []
        for t in range(precipitation.shape[1]):
            runoff, soil_water_ratio = self.cell(last_soil_water_ratio=soil_water_ratio,  # [batch_size, m, n]
                                                 precipitation=precipitation[:, t],  # [batch_size, ]
                                                 potential_evaporation=potential_evaporation[:, t])  # [batch_size, ]
            runoff_list.append(runoff)
        # 将产流量序列连接 shape：[[batch_size, m, n], [batch_size, m, n], ...] -> [batch_size, T, m, n]
        runoff = ops.stack(runoff_list, axis=1)
        return runoff
# %% 测试RunoffProducingCell 测试通过
# import numpy as np
# from hydronetwork.utils import tensor2numpy
# batchsize = 512
# T = 10
# m = 3
# n = 64
# n_mix_steps = 3
# model = RunoffProducingModel(m=m, n=n, n_mix_steps=n_mix_steps)
# precipitation = np.random.rand(batchsize, T)
# potential_evaporation = np.random.rand(batchsize, T)
# runoff = model(precipitation=precipitation, potential_evaporation=potential_evaporation)
# runoff = tensor2numpy(runoff)
