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
