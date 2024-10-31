# %%
from hydronetwork.model.ZonalExchangeNet.produce_runoff import RunoffProducingCell
import numpy as np

batch_size = 32
m = 4
n = 128
# 生成上一个时间步的土壤含水量比例，控制在0-1之间
last_soil_water_ratio = np.random.rand(batch_size, m, n)
last_soil_water_ratio = np.clip(last_soil_water_ratio, 0, 1)
# 生成本时间步的降雨量，控制在0-100之间
rainfall = np.random.rand(batch_size, 1) * 100
# 生成本时间步的潜在蒸散发量，控制在0-100之间
evaporation = np.random.rand(batch_size, 1) * 100

cell = RunoffProducingCell(n_soil_layers=m, n_soil_divisions=n)

# 测试cell
runoff = cell(last_soil_water_ratio, rainfall, evaporation)
