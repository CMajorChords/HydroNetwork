# 对特征做归一化处理
import keras
from keras import Layer, ops


@keras.saving.register_keras_serializable(package='Custom', name='LinearNormalization')
class LinearNormalization(Layer):
    """
    线性归一化层，对输入的特征进行线性归一化
    :param axis: 归一化的轴，-1表示对最后一个维度进行归一化，{-1, -2}表示对倒数两个维度进行归一化
    """

    def __init__(self, axis, **kwargs):
        assert axis == -1 or axis == [-1, -2], "axis必须是-1或者[-1, -2]"
        super(LinearNormalization, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):  # inputs: [batch_size, feature_size]
        return inputs / ops.sum(inputs, axis=self.axis, keepdims=True)

    def get_config(self):
        return {'axis': self.axis}


@keras.saving.register_keras_serializable(package='Custom', name='SoftmaxNormalization')
class SoftmaxNormalization(Layer):
    """
    softmax归一化层，对输入的特征进行softmax归一化
    :param axis: 归一化的轴，-1表示对最后一个维度进行归一化，、[-1, -2]表示对倒数两个维度进行归一化
    """

    def __init__(self, axis, **kwargs):
        assert axis == -1 or axis == [-1, -2], "axis必须是-1或者[-1, -2]"
        super(SoftmaxNormalization, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):  # inputs: [batch_size, feature_size]
        exp_x = ops.exp(inputs - ops.max(inputs, axis=self.axis, keepdims=True))
        return exp_x / ops.sum(exp_x, axis=self.axis, keepdims=True)

    def get_config(self):
        return {'axis': self.axis}

# %%测试层
# import numpy as np
#
# # 测试LinearNormalization和SoftmaxNormalization
# linear_normalization_1d = LinearNormalization(axis=-1)
# softmax_normalization_1d = SoftmaxNormalization(axis=-1)
# linear_normalization_2d = LinearNormalization(axis=[-1, -2])
# softmax_normalization_2d = SoftmaxNormalization(axis=[-1, -2])
# # [batch_size, feature_size] or [batch_size, m, n]
# inputs_1d = np.random.randn(32, 10)
# inputs_2d = np.random.randn(3, 4, 5)
#
# # 测试1d
# outputs_linear_1d = linear_normalization_1d(inputs_1d).cpu()
# outputs_softmax_1d = softmax_normalization_1d(inputs_1d).cpu()
# # 将每一行的元素相加，应该得到1
# print(outputs_linear_1d.sum(axis=-1))
# print(outputs_softmax_1d.sum(axis=-1))
#
# # 测试2d
# outputs_linear_2d = linear_normalization_2d(inputs_2d).cpu()
# outputs_softmax_2d = softmax_normalization_2d(inputs_2d).cpu()
# # 将每一个元素的值相加，应该得到1
# print(outputs_linear_2d.sum(axis=(-1, -2)))
# print(outputs_softmax_2d.sum(axis=(-1, -2)))

