# embedding
import keras
from keras import layers, Layer
from hydronetwork.model.layer.normalization import LinearNormalization


@keras.saving.register_keras_serializable(package='Custom', name='Embedding')
class Embedding(Layer):
    """
    使用线性层对特征进行embedding
    :param activation: 激活函数
    :param first_layer_units: 第一层神经元数量
    :param layer_units: 每层神经元数量
    :param last_activation: 最后一层激活函数，可选relu、sigmoid、softmax
    :param residual: 是否使用残差连接
    :param normalize: 是否对输出进行归一化
    """

    def __init__(self,
                 layer_units: list[int] = (32, 16, 1),
                 last_activation: str = "relu",
                 residual: bool = True,
                 normalize: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        # 检查神经元数量是否为int
        dense_layers = []
        # 按照layer_units创建神经网络，不要把最后一层的激活函数添加进去
        for i, unit in enumerate(layer_units[:-1]):
            assert isinstance(unit, int) and unit > 0, f"第{i}层神经元数量必须为正整数"
            dense_layers.append(layers.Dense(unit, activation="relu"))
        dense_layers.append(layers.Dense(layer_units[-1]))
        self.embedding = keras.Sequential(dense_layers)
        if residual:
            self.residual = layers.Dense(layer_units[-1])
        self.last_activation = layers.Activation(last_activation)
        if normalize:
            self.normalization = LinearNormalization(axis=-1)

    def call(self, features):
        if hasattr(self, 'residual'):
            output = self.embedding(features) + self.residual(features)
        else:
            output = self.embedding(features)
        output = self.last_activation(output)
        if hasattr(self, 'normalization'):
            output = self.normalization(output)
        return output

# %% 测试层 已通过测试

# [batch_size, num_features] -> [batch_size, 1]
# import numpy as np
# from hydronetwork.utils import tensor2numpy
#
# input = np.random.random((32, 10))
# embedding = Embedding()
# output = embedding(input)
# output = tensor2numpy(output)
