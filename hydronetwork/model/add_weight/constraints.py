# 对矩阵参数进行约束
import keras
from keras import ops
from keras.src import backend


class MinMaxL1Norm(keras.constraints.Constraint):
    """
    MinMaxL1Norm约束，用于约束权重矩阵的L1范数在[min_value, max_value]之间。

    :param min_value: L1范数的最小值
    :param max_value: L1范数的最大值
    :param rate: 约束力度，取值在[0, 1]之间，0表示不进行约束，1表示完全约束
    :param axis: 约束的轴
    """
    def __init__(self, min_value=0.0, max_value=1.0, rate=1.0, axis=-1):
        self.min_value = min_value
        self.max_value = max_value
        self.rate = rate
        self.axis = axis

    def __call__(self, w):
        w = backend.convert_to_tensor(w)
        # norms = ops.sqrt(ops.sum(ops.square(w), axis=self.axis, keepdims=True))
        # 修改为L1范数
        norms = ops.sum(ops.abs(w), axis=self.axis, keepdims=True)
        desired = (
                self.rate * ops.clip(norms, self.min_value, self.max_value)
                + (1 - self.rate) * norms
        )
        return w * (desired / (backend.epsilon() + norms))

    def get_config(self):
        return {
            "min_value": self.min_value,
            "max_value": self.max_value,
            "rate": self.rate,
            "axis": self.axis,
        }
