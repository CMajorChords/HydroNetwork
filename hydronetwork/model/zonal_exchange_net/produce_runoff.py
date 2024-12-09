import keras
from keras import Layer, ops
from hydronetwork.model.layer.embedding import Embedding

@keras.saving.register_keras_serializable(package='Custom', name='RunoffProducingCell')
class RunoffProducingCell(Layer):
    """
    产流模型单个时间步的计算单元。
    产流模型的输入为：
    1.上一个时间步的径流R_{t-1}, shape: [batch_size, n]
    2.本时间步的降雨量P_t, shape: [batch_size, ]
    3.本时间步和蒸发相关的特征EvapFeatures_t, shape: [batch_size, num_features]
    4.上一个时间步的土壤含水量SW_{t-1}, shape: [batch_size, n]

    计算过程：
    遗忘：扣除本步的蒸发量和上一步的径流量。
    forget_gate = sigmoid(W_forget * [SW_{t-1}, EvapFeatures_t，R_{t-1}] + b_forget)
    计算遗忘后的土壤含水量：SW_t = evap_gate * SW_{t-1}

    输入：将降水融入土壤含水量中
    输入门：prcp_gate = sigmoid(W_prcp * [SW_t, P_t] + b_prcp)
    候选土壤含水量：SW_cand = tanh(W_SW * [SW_t, P_t] + b_SW)
    降水后的土壤含水量：SW_t = prcp_gate * SW_cand

    输出：根据土壤含水量计算径流
    径流门 runoff_gate = sigmoid(W_runoff * [SW_t, P_t] + b_runoff)
    本时间步的径流：R_t = runoff_gate * tanh(SW_t)


    :param units: 土壤含水量神经元数量
    """

    def __init__(self,
                 units: int,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.dense_forget_gate = Embedding(layer_units=[2 * units, units], last_activation='sigmoid')
        self.dense_input_gate = Embedding(layer_units=[2 * units, units], last_activation='sigmoid')
        self.dense_input_candidate = Embedding(layer_units=[2 * units, units], last_activation='tanh')
        self.dense_output_gate = Embedding(layer_units=[2 * units, units], last_activation='sigmoid')

    def call(self,
             last_runoff,  # [batch_size, units]
             last_soil_water,  # [batch_size, units]
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
        precipitation = ops.expand_dims(precipitation, axis=-1)  # [batch_size, 1]
        # 遗忘门
        forget_gate = self.dense_forget_gate(ops.concatenate(xs=[last_soil_water, evaporation_features, last_runoff],
                                                             axis=-1)
                                             )  # [batch_size, units + num_features + units]

        # 输入门
        input_gate = self.dense_input_gate(ops.concatenate([last_soil_water, precipitation], axis=-1))
        # 候选土壤含水量
        soil_water_candidate = self.dense_input_candidate(ops.concatenate([last_soil_water, precipitation], axis=-1))
        # 降水后的土壤含水量
        soil_water = input_gate * soil_water_candidate + forget_gate * last_soil_water

        # 输出门
        output_gate = self.dense_output_gate(ops.concatenate([soil_water, precipitation], axis=-1))
        # 本时间步的径流
        runoff = output_gate * soil_water
        return runoff, soil_water


# %%测试RunoffProducingCell
# import numpy as np
#
# batchsize = 512
# units = 32
# last_runoff = np.random.rand(batchsize, units)
# last_soil_water = np.random.rand(batchsize, units)
# precipitation = np.random.rand(batchsize)
# evaporation_features = np.random.rand(batchsize, 5)
# model = RunoffProducingCell(units=units)
# runoff, soil_water = model(last_runoff, last_soil_water, precipitation, evaporation_features)

# %%
@keras.saving.register_keras_serializable(package='Custom', name='RunoffProducingModel')
class RunoffProducingModel(Layer):
    """
    产流模型，用于计算整个时间序列的产流量
    :param units: 土壤含水量神经元数量
    """

    def __init__(self,
                 units: int,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.units = units
        self.runoff_producing_cell = RunoffProducingCell(units)

    def call(self,
             precipitation,  # [batch_size, T]
             evaporation_features,  # [batch_size, T, num_features]
             ):
        """
        计算整个时间序列的产流量
        :param precipitation:
        :param evaporation_features:
        :return: 每一层的产流量，size=batch_size*T*m
        """
        batch_size, time_steps, num_features = evaporation_features.shape
        runoff = []
        soil_water = []
        # 边界条件
        last_runoff = ops.zeros([batch_size, self.units])
        last_soil_water = ops.zeros([batch_size, self.units])
        for t in range(time_steps):
            runoff_t, soil_water_t = self.runoff_producing_cell(last_runoff=last_runoff,
                                                                last_soil_water=last_soil_water,
                                                                precipitation=precipitation[:, t],
                                                                evaporation_features=evaporation_features[:, t, :])
            runoff.append(runoff_t)  # [batch_size, units]
            soil_water.append(soil_water_t)  # [batch_size, units]
            last_runoff = runoff_t
            last_soil_water = soil_water_t
        # stack
        runoff = ops.stack(runoff, axis=1)  # [batch_size, T, units]
        soil_water = ops.stack(soil_water, axis=1)  # [batch_size, T, units]
        return runoff, soil_water


# %% 测试RunoffProducingCell 测试通过
# import numpy as np
#
# batch_size = 512
# T = 365
# units = 64
#
# precipitation = np.random.rand(batch_size, T)
# evaporation_features = np.random.rand(batch_size, T, 5)
# model = RunoffProducingModel(units=units)
# runoff, soil_water = model(precipitation, evaporation_features)
