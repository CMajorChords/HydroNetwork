import keras
from keras import Layer, ops, KerasTensor
from hydronetwork.model.ZonalExchangeNet.mix_water import WaterMixHead
from hydronetwork.model.add_weight.constraints import MinMaxL1Norm


@keras.saving.register_keras_serializable(package='Custom', name='RunoffProducingCell')
class RunoffProducingCell(Layer):
    """
    产流模型单个时间步的计算单元。
    产流模型的输入为：
    1.上一个时间步输出的土壤含水量比例，size=m*n，m为土壤层的数量，n为按照土壤蓄水容量将土壤在纵向上划分的层数
    2.本时间步的降雨量，size=1
    3.本时间步的潜在蒸散发量，size=1（可能是一些和潜在蒸散发量相关的变量，如温度、湿度，而不是直接的潜在蒸散发量，所以需要一个embedding层）

    计算过程：
    1.前处理
        1.1 将上一个时间步的土壤含水量比例乘以土壤蓄水容量，得到土壤含水量，
            土壤蓄水容量是一个可学习的参数矩阵，[batch_size, m, n] -> [batch_size, m, n]
        1.2 将降水量平均分为n*s份，s为水量混合步数，[batch_size, 1] -> [batch_size, 1, n]
        2.3 将降水量concat到土壤含水量上,这一步应该在每一次混合的过程中进行 [batch_size, m, n] -> [batch_size, m+1, n]
    2.水量混合
        2.1水量垂向混合：利用大小为2*n的卷积核将纵向上相邻的两层土壤含水量进行卷积运算(无padding, stride=1)，
            [batch_size, m+1, n] -> [batch_size, m, n]
        2.3水量横向混合：利用attention机制将m个土壤层的水量进行横向混合，
            [batch_size, m, n] -> [batch_size, m, n]
        2.4将最后得到的3个m*n矩阵归一化（对于土壤含水量，使用tanh。对于降水入渗和蒸发，使用softmax或者normalization，），得到本时间步的某个比例，
            [batch_size, m, n] -> [batch_size, m, n]
        2.5将以上3步重复三遍（3个head），得到3个m*n矩阵，分别代表土壤含水量比例、降水（入渗）分配比例、潜在蒸散发分配比例
            3 * [batch_size, m, n] -> 3 * [batch_size, m, n]
        2.6土壤含水量比例*土壤蓄水容量得到土壤含水量，降水分配比例*降水量得到土壤下渗量，潜在蒸散发分配比例*潜在蒸散发量得到土壤蒸散发量
            [batch_size, m, n] -> [batch_size, m, n]
    3.根据水量平衡，产流量 = 降水（入渗）量 - 土壤蒸散发量-（本时刻土壤含水量-上时刻土壤含水量）
        [batch_size, m, n] -> [batch_size, m, n]

    :param n_soil_layers: 土壤层数
    :param n_soil_divisions: 土壤纵向划分层数
    :param n_mix_steps: 水量混合步数，默认为3。每一步都包含一次垂向混合和一次横向混合。每一步共用同一套参数。
    """

    def __init__(self,
                 n_soil_layers: int,
                 n_soil_divisions: int,
                 water_capacity: KerasTensor,
                 n_mix_steps: int = 3,
                 **kwargs
                 ):
        super(RunoffProducingCell, self).__init__(**kwargs)
        # 设置土壤水蓄水容量矩阵，size=m*n，添加约束条件，使得所有元素都大于0
        self.n_soil_layers = n_soil_layers
        self.n_soil_divisions = n_soil_divisions
        self.water_capacity = water_capacity
        # 设置水量混合头部
        self.n_mix_steps = n_mix_steps
        self.water_mix_head = WaterMixHead(n_layers=3, n_divisions=3,
                                           activation="sigmoid",
                                           n_mix_steps=n_mix_steps)
        self.infiltration_mix = WaterMixHead(n_layers=3, n_divisions=3,
                                             activation="softmax",
                                             n_mix_steps=n_mix_steps)
        self.evaporation_mix = WaterMixHead(n_layers=3, n_divisions=3,
                                            activation="softmax",
                                            n_mix_steps=n_mix_steps)

    def call(self,
             last_soil_water_ratio: KerasTensor,
             precipitation: KerasTensor,
             evaporation: KerasTensor,
             ):
        """
        计算单个时间步的产流量

        :param last_soil_water_ratio: 上一个时间步的土壤含水量比例，size=batch_size*m*n
        :param precipitation: 本时间步的降雨量，size=batch_size*1
        :param evaporation: 本时间步的潜在蒸散发量，size=batch_size*1
        :return: 每一层的产流量，size=batch_size*m
        """
        # 前处理
        soil_water = last_soil_water_ratio * self.water_capacity  # shape = [batch_size, m, n]
        # 将降水量除以(n_soil_divisions*n_mix_steps)，得到每一切片每一次的降水量 [batch_size, ] -> [batch_size, 1, n]
        precipitation_progress = precipitation / self.n_soil_divisions / self.n_mix_steps  # [batch_size, 1, 1]
        precipitation_progress = ops.repeat(precipitation_progress, repeats=self.n_soil_divisions,
                                            axis=-1)  # shape = [batch_size, 1, n]
        # 水量混合
        soil_water_ratio = self.water_mix_head(soil_water=soil_water,
                                               precipitation=precipitation_progress)  # size=batch_size*m*n
        infiltration_ratio = self.infiltration_mix(soil_water=soil_water,
                                                   precipitation=precipitation_progress)  # size=batch_size*m*n
        evaporation_ratio = self.evaporation_mix(soil_water=soil_water,
                                                 precipitation=precipitation_progress)  # size=batch_size*m*n
        # 计算产流量
        infiltration = infiltration_ratio * precipitation  # [batch_size, m, n] * [batch_size, 1, n] -> [batch_size, m, n]
        evaporation = evaporation_ratio * evaporation  # size=batch_size*m*n
        return infiltration - evaporation - (
                soil_water_ratio - last_soil_water_ratio) * self.water_capacity  # size=batch_size*m*n


@keras.saving.register_keras_serializable(package='Custom', name='RunoffProducingModel')
class RunoffProducingModel(Layer):
    """
    产流模型，用于计算整个时间序列的产流量
    :param n_soil_layers: 土壤层数
    :param n_soil_divisions: 土壤纵向划分层数
    :param water_capacity_max: 单个土块的最大蓄水容量
    """

    def __init__(self,
                 n_soil_layers: int,
                 n_soil_divisions: int,
                 water_capacity_max=120,
                 **kwargs
                 ):
        super(RunoffProducingModel, self).__init__(**kwargs)
        self.shape_soil_water = (n_soil_layers, n_soil_divisions)
        self.water_capacity = self.add_weight(shape=(n_soil_layers, n_soil_divisions),
                                              trainable=True,
                                              constraint=MinMaxL1Norm(min_value=0, max_value=water_capacity_max),
                                              )
        self.cell = RunoffProducingCell(n_soil_layers=n_soil_layers,
                                        n_soil_divisions=n_soil_divisions,
                                        water_capacity=self.water_capacity)

    def call(self,
             rainfall: KerasTensor,  # size=batch_size*T
             evaporation: KerasTensor,  # size=batch_size*T
             ):
        """
        计算整个时间序列的产流量
        :param rainfall: 降雨量序列，size=batch_size*T
        :param evaporation: 蒸发量序列，size=batch_size*T
        :return: 产流量序列，size=batch_size*T*m
        """
        # 初始化土壤含水量比例
        soil_water_ratio = ops.ones(self.shape_soil_water)  # size=batch_size*m*n
        # 对每个时间步进行连续计算
        # 预分配产流量矩阵 size=batch_size*T*m*n
        runoff = ops.zeros((rainfall.shape[0], rainfall.shape[1], self.shape_soil_water[0], self.shape_soil_water[1]))
        for t in range(rainfall.shape[1]):
            runoff[:, t] = self.cell(last_soil_water_ratio=soil_water_ratio,
                                     precipitation=rainfall[:, t],
                                     evaporation=evaporation[:, t])
        return runoff


# %%
# 测试RunoffProducingCell
import torch

water_capacity = torch.rand(2, 4, 6, device='cuda')
precipitation = torch.rand(2, 1, 1, device='cuda')
evaporation = torch.rand(2, 1, 1, device='cuda')
last_soil_water_ratio = torch.rand(2, 4, 6, device='cuda')
# 保证last_soil_water_ratio在0-1之间
last_soil_water_ratio = torch.clamp(last_soil_water_ratio, 0, 1)
cell = RunoffProducingCell(n_soil_layers=4, n_soil_divisions=6, water_capacity=water_capacity)
# cell(last_soil_water_ratio, precipitation, evaporation)
result = cell(last_soil_water_ratio, precipitation, evaporation)

# 计算evaporation_ratio之和
# evaporation_ratio_sum = infiltration_ratio.sum(dim=(1, 2), keepdim=False)
# TODO soil_water_ratio所有的元素都变成了0.5，这是不对的

