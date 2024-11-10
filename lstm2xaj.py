# %%
from hydronetwork.model.EncoderDecoderLSTM import encoder_decoder_lstm
from hydronetwork.data import camels_us
from hydronetwork.dataset import split_timeseries, get_dataset
from hydronetwork.train import callback_for_features_selection, WarmupExponentialDecay
from hydronetwork.train import NSELoss, RMSELoss
from keras.api.optimizers import AdamW
from hydronetwork.utils import autoload
from hydronetwork.dataset.preprocessing import log_transform
from hydronetwork.dataset import Normalizer

autoload()

# %%加载数据
data = camels_us.load_timeseries(gauge_id='01013500', multi_process=True, unit="m^3/s")

lookback = 60
horizon = 7

# 特征工程
data['streamflow'] = log_transform(data['streamflow'])
normalizer = Normalizer()
data = normalizer.normalize(data)

# 划分数据集
train_data, test_data = split_timeseries(data, split_list=[0.8, 0.2])
train_dataset = get_dataset(timeseries=train_data,
                            lookback=lookback,
                            horizon=horizon,
                            features_bidirectional=['prcp(mm/day)', 'srad(W/m2)', 'tmax(C)', 'tmin(C)', 'vp(Pa)'],
                            features_lookback=['streamflow'],
                            target="streamflow",
                            batch_size=1024,
                            )
test_dataset = get_dataset(timeseries=test_data,
                           lookback=lookback,
                           horizon=horizon,
                           features_bidirectional=['prcp(mm/day)', 'srad(W/m2)', 'tmax(C)', 'tmin(C)', 'vp(Pa)'],
                           features_lookback=['streamflow'],
                           target="streamflow",
                           batch_size=1024,
                           )

# 设置参数
features_type = ['bidirectional', 'lookback']

model = encoder_decoder_lstm(lookback, horizon, features_type)

model.compile(optimizer=AdamW(learning_rate=WarmupExponentialDecay(dataset_length=len(train_dataset),
                                                                   initial_learning_rate=1e-4, )),
              loss=NSELoss(),
              metrics=[RMSELoss()],
              )
model.fit(train_dataset,
          epochs=100,
          verbose=1,
          shuffle=True,
          validation_data=test_dataset,
          callbacks=callback_for_features_selection(),
          )

# %%
from hydronetwork.evaluate.evaluate import predict
import numpy as np


def nse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true_mean = np.mean(y_true)
    return 1 - np.sum((y_pred - y_true) ** 2) / np.sum((y_true - y_true_mean) ** 2)


import pandas as pd

result = predict(model, test_dataset)
nse_df = {}
for i in range(0, len(result.columns), 2):
    nse_df[f"{int(0.5 * i)}_day"] = nse(result.iloc[:, i], result.iloc[:, i + 1])
nse_df = pd.Series(nse_df)
