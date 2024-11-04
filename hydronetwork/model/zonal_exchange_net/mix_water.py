import keras
from keras import layers, Layer, ops, activations
from hydronetwork.model.layer.attention import SelfAttention
from hydronetwork.model.layer.normalization import LinearNormalization, SoftmaxNormalization


@keras.saving.register_keras_serializable(package='Custom', name='WaterMixHead')
class WaterMixHead(Layer):
    """
    水量混合头部，用于将不同的水量进行混合。
    1.单次混合操作包括：
        1.1 concat：将降水量与土壤含水量进行拼接
            shape: (batch_size, m, n), (batch_size, 1, n) -> (batch_size, m+1, n)
        1.2 垂向混合：利用大小为2*n的卷积核将纵向上相邻的两层土壤含水量进行卷积运算(无padding, stride=1)
            shape: (batch_size, m+1, n) -> (batch_size, m, n)
        1.3 横向混合：利用attention机制将m个土壤层的水量进行横向混合，size=m*n，每层用不同的attention
            shape: (batch_size, m, n) -> (batch_size, m, n)
    2. 进行多次混合操作，每一次混合操作都包含一次垂向混合、一次通道混合和一次横向混合。每次混合参数共享。
    3. 最后使用激活函数进行归一化，让每一个元素都在0-1之间。

    :param m: 土壤层数
    :param n: 土壤纵向划分层数
    :param normalization: 归一化操作，可以是'sigmoid'、'softmax'、'linear_normalize'
    :param n_mix_steps: 水量混合步数，默认为3。每一步都包含一次垂向混合、一次通道混合和一次横向混合。每一步用的参数不共享。
    """

    def __init__(self,
                 m: int,
                 n: int,
                 normalization: str,
                 n_mix_steps: int = 3,
                 if_lateral_mix: bool = True,
                 **kwargs):
        super(WaterMixHead, self).__init__(**kwargs)
        self.m = m
        self.n = n
        self.n_mix_steps = n_mix_steps
        self.if_lateral_mix = if_lateral_mix
        self.conv = []
        self.attention = []
        self.relu = layers.ReLU()
        for i in range(n_mix_steps):
            self.conv.append(layers.DepthwiseConv1D(kernel_size=2))
            # if if_lateral_mix:
            #     self.attention.append([SelfAttention() for _ in range(m)])
        # 激活函数，这在确定百分比时很重要。
        assert normalization in ["sigmoid", "softmax", "linear_normalize"], "激活函数必须是sigmoid、softmax或normalization"
        self.normalization = normalization
        if normalization == "sigmoid":
            self.normalization = activations.sigmoid
        elif normalization == "softmax":
            self.normalization = SoftmaxNormalization(axis=[-1, -2])
        else:
            self.normalization = LinearNormalization(axis=[-1, -2], add_relu=True)

    # def build(self, input_shape):

    def call(self,
             soil_water,  # size=batch_size*m*n
             precipitation,  # size=batch_size*1*n
             ):
        for i in range(self.n_mix_steps):
            soil_water = ops.concatenate([soil_water, precipitation], axis=-2)  # size=batch_size*(m+1)*n
            # 垂向混合
            soil_water = self.conv[i](soil_water)  # size=batch_size*(m+1)*n -> size=batch_size*m*n
            # # 横向混合，需要首先将m*n转换为n*m
            # if self.if_lateral_mix:
            #     soil_water = ops.transpose(soil_water, axes=[0, 2, 1])  # size=batch_size*n*m
            #     for m in range(self.m):
            #         m_soil_water = soil_water[:, :, m].unsqueeze(-1)  # size=batch_size*n*1
            #         m_soil_water = self.attention[i][m](m_soil_water)  # size=batch_size*n*1
            #         soil_water[:, :, m] = m_soil_water.squeeze(-1)
            #     # 重新转换为m*n
            #     soil_water = ops.transpose(soil_water, axes=[0, 2, 1])  # size=batch_size*m*n
            # 激活函数，注意最后一次混合不需要激活函数
            if i < self.n_mix_steps - 1:
                soil_water = self.relu(soil_water)
        # 归一化
        return self.normalization(soil_water)

    def get_config(self):
        return {"m": self.m,
                "n": self.n,
                "normalization": self.normalization,
                "n_mix_steps": self.n_mix_steps,
                "if_lateral_mix": self.if_lateral_mix,
                }

# %%测试层 已通过测试
# import numpy as np
#
# soil_water = np.random.rand(2, 3, 4)
# precipitation = np.random.rand(2, 1, 4)
#
# layer_sigmoid = WaterMixHead(m=3, n=4, normalization="sigmoid")
# layer_softmax = WaterMixHead(m=3, n=4, normalization="softmax")
# layer_linear = WaterMixHead(m=3, n=4, normalization="linear_normalize")
#
# output_sigmoid = layer_sigmoid(soil_water, precipitation).cpu().detach().numpy()
# output_softmax = layer_softmax(soil_water, precipitation).cpu().detach().numpy()
# output_linear = layer_linear(soil_water, precipitation).cpu().detach().numpy()
