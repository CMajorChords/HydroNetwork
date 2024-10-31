import keras
from keras import layers, Layer, ops, activations
from hydronetwork.model.layer.attention import SelfAttention
from hydronetwork.model.layer.normalization import LinearNormalization, SoftmaxNormalization


@keras.saving.register_keras_serializable(package='Custom', name='WaterMixHead')
class WaterMixHead(Layer):
    """
    水量混合头部，用于将不同的水量进行混合。
    1.单次混合操作包括：
        1.1 垂向混合：利用大小为2*n的卷积核将纵向上相邻的两层土壤含水量进行卷积运算(无padding, stride=1)
            size=batch_size*(m+1)*n -> size=batch_size*m*n
        1.3 横向混合：利用attention机制将m个土壤层的水量进行横向混合，size=m*n
    2. 进行多次混合操作，每一次混合操作都包含一次垂向混合、一次通道混合和一次横向混合。每次混合参数并不共享。
    3. 最后使用激活函数进行归一化，让每一个元素都在0-1之间。

    :param n_layers: 土壤层数
    :param n_divisions: 土壤纵向划分层数
    :param activation: 激活函数，可以是'tanh'、'softmax'、'normalization'
    :param n_mix_steps: 水量混合步数，默认为3。每一步都包含一次垂向混合、一次通道混合和一次横向混合。每一步共用同一套参数。
    """

    def __init__(self,
                 n_layers: int,
                 n_divisions: int,
                 activation: str,
                 n_mix_steps: int = 3,
                 **kwargs):
        super(WaterMixHead, self).__init__(**kwargs)
        self.n_layers = n_layers
        self.n_divisions = n_divisions
        self.n_mix_steps = n_mix_steps

        self.conv = []
        self.attention = []
        self.relu = layers.ReLU()
        for i in range(n_mix_steps):
            self.conv.append(layers.DepthwiseConv1D(kernel_size=2))
            self.attention.append(SelfAttention())

        assert activation in ["sigmoid", "softmax", "linear_normalize"], "激活函数必须是sigmoid、softmax或normalization"
        self.activation = activation
        if activation == "sigmoid":
            self.normalization = activations.sigmoid
        elif activation == "softmax":
            self.normalization = SoftmaxNormalization(axis=[-1, -2])
        else:
            self.normalization = LinearNormalization(axis=[-1, -2])

    # def build(self, input_shape):

    def call(self,
             soil_water,  # size=batch_size*m*n
             precipitation,  # size=batch_size*1*n
             ):
        for i in range(self.n_mix_steps):
            soil_water = ops.concatenate([soil_water, precipitation], axis=-2)  # size=batch_size*(m+1)*n
            # 垂向混合
            soil_water = self.conv[i](soil_water)  # size=batch_size*(m+1)*n -> size=batch_size*m*n
            # 横向混合，需要首先将m*n转换为n*m
            soil_water = ops.transpose(soil_water, axes=[0, 2, 1])  # size=batch_size*n*m
            soil_water = self.attention[i](soil_water)  # size=batch_size*n*m
            soil_water = ops.transpose(soil_water, axes=[0, 2, 1])  # size=batch_size*m*n
            # 激活函数，注意最后一次混合不需要激活函数
            if i < self.n_mix_steps - 1:
                soil_water = self.relu(soil_water)
        # 归一化
        return self.normalization(soil_water)

    def get_config(self):
        return {"n_layers": self.n_layers,
                "n_divisions": self.n_divisions,
                "activation": self.activation,
                "n_mix_steps": self.n_mix_steps}

# %%测试层
import numpy as np

soil_water = np.random.random((2, 3, 4))
precipitation = np.ones((2, 1, 4))

layer_sigmoid = WaterMixHead(n_layers=3, n_divisions=4, activation="sigmoid")
layer_softmax = WaterMixHead(n_layers=3, n_divisions=4, activation="softmax")
layer_linear = WaterMixHead(n_layers=3, n_divisions=4, activation="linear_normalize")

output_sigmoid = layer_sigmoid(soil_water, precipitation)
output_softmax = layer_softmax(soil_water, precipitation)
output_linear = layer_linear(soil_water, precipitation)
