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
        self.normalization = LinearNormalization(axis=[-2, -1])

        # 动态调整混合的次数
        for i in range(n_mix_steps):
            setattr(self, f"conv{i}", layers.DepthwiseConv1D(kernel_size=2))
            setattr(self, f"attention{i}", SelfAttention())
            setattr(self, f"relu{i}", layers.ReLU())

        assert activation in ["tanh", "softmax", "linear_normalize"], "激活函数必须是tanh、softmax或normalization"
        self.activation = activation
        if activation == "tanh":
            self.normalization = activations.tanh
        elif activation == "softmax":
            self.normalization = SoftmaxNormalization(axis=[-2, -1])
        elif activation == "normalization":
            self.normalization = LinearNormalization(axis=[-2, -1])

    def call(self,
             soil_water,  # size=batch_size*(m+1)*n
             precipitation,  # size=batch_size*n
             ):
        for i in range(self.n_mix_steps):
            soil_water = ops.concatenate([soil_water, precipitation], axis=-2)  # size=batch_size*(m+1)*n
            # 垂向混合
            soil_water = getattr(self, f"conv{i}")(soil_water)  # size=batch_size*(m+1)*n -> size=batch_size*m*n
            # 横向混合，需要首先将m*n转换为n*m
            soil_water = ops.transpose(soil_water, perm=[0, 2, 1])  # size=batch_size*n*m
            soil_water = getattr(self, f"attention{i}")(soil_water)  # size=batch_size*n*m
            soil_water = ops.transpose(soil_water, perm=[0, 2, 1])  # size=batch_size*m*n
            # 激活函数，注意最后一次混合不需要激活函数
            if i < self.n_mix_steps - 1:
                soil_water = getattr(self, f"relu{i}")(soil_water)
        # 归一化
        return self.normalization(soil_water)

    def get_config(self):
        config = super(WaterMixHead, self).get_config()
        config.update({"n_layers": self.n_layers,
                       "n_divisions": self.n_divisions,
                       "activation": self.activation,
                       "n_mix_steps": self.n_mix_steps})
        return config
