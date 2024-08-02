# 构建残差块，残差块由一个线性层和一个激活函数组成
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 ):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.residual_Linear = nn.Linear(in_features, out_features)
        self.activation = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        print(f'Input x shape: {x.shape}')  # 添加打印语句
        residual = self.residual_Linear(x)
        x = self.linear(self.activation(x))
        return self.bn(x + residual)
