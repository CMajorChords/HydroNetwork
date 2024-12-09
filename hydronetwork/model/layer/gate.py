# embedding
import keras
from keras import layers, Layer
from keras.api.utils import normalize


@keras.saving.register_keras_serializable(package='Custom', name='Gate')
class Gate(Layer):
    """
    门控层，对输入的特征进行门控
    :param units: 门控单元数量
    :param activation: 激活函数
    :param normalization: 是否对输出进行归一化
    """

    def __init__(self,
                 units: int,
                 activation: str = "sigmoid",
                 normalization: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense = layers.Dense(units, activation=activation)
        self.normalize = normalization

    def call(self, features):
        output = self.dense(features)
        if self.normalize:
            output = normalize(output, order=1, axis=-1)
        return output
