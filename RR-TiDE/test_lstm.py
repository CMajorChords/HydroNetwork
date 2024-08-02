from hydronetwork.data.camels import loading_timeseries, load_attributes, get_gauge_id
from hydronetwork.dataset import split_timeseries, get_dataset
from torch.utils.data import DataLoader
from torch import nn
import torch

# %% 加载数据和数据集
gauge_ids = get_gauge_id(n=30)
data = load_timeseries(gauge_id=gauge_ids)
attributes = load_attributes()
data = split_timeseries(data, split_list=["1985-01-01", "1990-01-01"])
data = data[0]
dataset = get_dataset(
    timeseries=data,
    target="streamflow",
    lookback=80,
    horizon=7,
    attributes=attributes,
    features_bidirectional=["prcp(mm/day)",
                            "srad(W/m2)",
                            "tmax(C)",
                            "tmin(C)",
                            "vp(Pa)"],
)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
data_loader_test = DataLoader(dataset, batch_size=1, shuffle=False)


# %% 定义模型
# ---test forward---
class ResidualBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(ResidualBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layer1 = nn.Linear(in_features, out_features // 2)
        self.layer2 = nn.Linear(out_features // 2, out_features)
        # 为了维度对齐的线性层
        self.align = nn.Linear(in_features, out_features)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        y = self.layer1(x)
        y = self.leaky_relu(y)
        y = self.layer2(y)
        y = self.leaky_relu(y)
        y = y + self.align(x)
        return y


#%% 模拟一个batch
batch_timeseries, batch_attributes, batch_target = next(iter(data_loader))
# 定义各层
attributes_embedding = ResidualBlock(in_features=batch_attributes.shape[-1], out_features=5)
timeseries_embedding = ResidualBlock(in_features=batch_timeseries.shape[-1], out_features=5)
lstm_encoder = nn.LSTM(input_size=10, hidden_size=128, num_layers=2, batch_first=True)
cell_state_transform = ResidualBlock(in_features=128, out_features=64)
hidden_state_transform = ResidualBlock(in_features=128, out_features=64)
lstm_decoder = nn.LSTM(input_size=10, hidden_size=128, num_layers=2, batch_first=True)
output_transform = ResidualBlock(in_features=128, out_features=64)
# 前向传播
batch_attributes = attributes_embedding(batch_attributes)
batch_timeseries = timeseries_embedding(batch_timeseries)
# 给attributes加一个时间维度
batch_attributes = batch_attributes.unsqueeze(1).repeat(1, batch_timeseries.shape[1], 1)
# 拼接
batch_input = torch.cat([batch_timeseries, batch_attributes], dim=-1)
# 输入encoder
output_encoder, (hidden_state, cell_state) = lstm_encoder(batch_input)
# 转换
cell_state = cell_state_transform(cell_state.squeeze(0))
hidden_state = hidden_state_transform(hidden_state.squeeze(0))
# 解码
output_decoder, _ = lstm_decoder(batch_input, (hidden_state.unsqueeze(0), cell_state.unsqueeze(0)))