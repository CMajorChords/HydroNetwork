# 蒸发量embedding
import keras
from keras import layers, Layer


@keras.saving.register_keras_serializable(package='Custom', name='Embedding')
class Embedding(Layer):
    """
    使用先行层对特征进行embedding
    :param first_layer_units: 第一层神经元数量，必须为偶数
    """

    def __init__(self,
                 layer_units: list[int] = (32, 16, 1),
                 last_activation: bool = True,
                 **kwargs):
        super(Embedding, self).__init__(**kwargs)
        # 检查神经元数量是否为int
        dense_layers = []
        for i, unit in enumerate(layer_units):
            assert isinstance(unit, int) and unit > 0, f"第{i}层神经元数量必须为正整数"
            dense_layers.append(layers.Dense(unit, activation="relu"))
        # 如果最后一层不需要激活函数，则删除最后一层后再添加一个线性层
        if not last_activation:
            dense_layers.pop()
            dense_layers.append(layers.Dense(layer_units[-1]))
            self.residual = layers.Dense(layer_units[-1])
        else:
            self.residual = layers.Dense(1, activation="relu")
        self.embedding = keras.Sequential(dense_layers)

    def call(self, features):
        return self.embedding(features) + self.residual(features)
