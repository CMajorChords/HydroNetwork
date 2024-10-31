# 各种attention机制
import keras
from keras import Layer, layers, ops


@keras.saving.register_keras_serializable(package='Custom', name='SelfAttention')
class SelfAttention(Layer):
    """
    自注意力机制，可选在时间步上进行自注意力机制，也可选在特征维度上进行自注意力机制。
    输入张量的维度为[batch_size, time_steps, num_features]

    :param attention_axis: 注意力机制作用的轴
    :return: 注意力机制作用后的矩阵，维度与输入矩阵相同
    """

    def __init__(self,
                 **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.attention = layers.Attention()
        self.num_features = None
        self.w_query = None
        self.w_key = None
        self.w_value = None

    def build(self, input_shape):
        num_features = input_shape[-1]
        self.num_features = num_features
        self.w_query = self.add_weight(name="w_query",
                                       shape=(num_features, num_features),
                                       trainable=True,
                                       dtype="float32")
        self.w_key = self.add_weight(name="w_key",
                                     shape=(num_features, num_features),
                                     trainable=True,
                                     dtype="float32")
        self.w_value = self.add_weight(name="w_value",
                                       shape=(num_features, num_features),
                                       trainable=True,
                                       dtype="float32")

    def call(self, inputs):
        # 计算query、key、value
        query = ops.dot(inputs,
                        self.w_query,
                        )  # [batch_size, time_steps, num_features] -> [batch_size, time_steps, num_features]
        key = ops.dot(inputs,
                      self.w_key,
                      )  # [batch_size, time_steps, num_features] -> [batch_size, time_steps, num_features]
        value = ops.dot(inputs,
                        self.w_value,
                        )  # [batch_size, time_steps, num_features] -> [batch_size, time_steps, num_features]

        # 直接得到注意力机制作用后的张量
        return self.attention([query, value, key])


# %% 测试层
# [batch_size, time_steps, num_features] -> [batch_size, time_steps, num_features]
# import numpy as np
#
# inputs = np.random.randn(32, 13, 4)
#
# outputs = SelfAttention()(inputs)
