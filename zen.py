# ZonalExchangeNet: 使用神经网络建模降雨径流过程
# %load_ext autoreload
# %autoreload 2
from hydronetwork.data import camels_us as camels
from hydronetwork.dataset import get_dataset, split_timeseries

# %%数据加载
gauge_id = "11266500"
timeseries = camels.load_timeseries(gauge_id)
train_set, val_set = split_timeseries(timeseries, split_list=[0.8, 0.2])
# ['prcp(mm/day)', 'srad(W/m2)', 'tmax(C)', 'tmin(C)', 'vp(Pa)', 'streamflow']
dataset_config = {"lookback": 365,
                  "horizon": 30,
                  "batch_size": 512,
                  "features_bidirectional": ["prcp(mm/day)", "srad(W/m2)", "tmax(C)", "tmin(C)", "vp(Pa)"],
                  "target": "streamflow",
                  }
train_set, val_set = get_dataset(train_set, **dataset_config), get_dataset(val_set, **dataset_config)

# %%模型构建
from hydronetwork.model.ZonalExchangeNet.produce_runoff import RunoffProducingCell







# 产流模型

