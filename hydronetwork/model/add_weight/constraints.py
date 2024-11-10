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


class NonNegAndSumLimit(keras.constraints.Constraint):
    """
    NonNegAndSumLimit约束，用于约束权重矩阵的非负性与矩阵之和的上下界

    :param min_value: 矩阵之和的最小值
    :param max_value: 矩阵之和的最大值
    :param axis: 约束的轴
    :param rate: 约束力度，取值在[0, 1]之间，0表示不进行约束，1表示完全约束
    """

    def __init__(self,
                 min_value,
                 max_value,
                 axis=-1,
                 rate=1.0,
                 ):
        self.rate = rate
        self.axis = axis
        self.max_value = max_value
        self.min_value = min_value

    def __call__(self, w):
        w = backend.convert_to_tensor(w)
        # 非负性约束
        w = ops.relu(w)
        # 和的上下界约束
        total_sum = ops.sum(w, axis=self.axis, keepdims=True)
        # 计算缩放因子，使得矩阵之和在[min_value, max_value]之间
        scaling_factor = ops.where(
            total_sum < self.min_value,  # 如果总和小于 min_value
            x1=self.min_value / (backend.epsilon() + total_sum),  # 使用 min_value 缩放
            x2=ops.where(
                total_sum > self.max_value,  # 如果总和大于 max_value
                x1=self.max_value / (backend.epsilon() + total_sum),  # 使用 max_value 缩放
                x2=1.0  # 否则保持不变
            )
        )
        return w * (scaling_factor * self.rate + (1 - self.rate))

    def get_config(self):
        return {
            "rate": self.rate,
            "axis": self.axis,
            "max_value": self.max_value,
            "min_value": self.min_value,
        }


class SumL1Norm(keras.constraints.Constraint):
    """
    SumL1Norm约束，用于约束权重矩阵的L1范数和为1。

    :param rate: 约束力度，取值在[0, 1]之间，0表示不进行约束，1表示完全约束
    :param axis: 约束的轴
    """

    def __init__(self,
                 rate=1.0,
                 axis=0,
                 ):
        self.rate = rate
        self.axis = axis

    def __call__(self, w):
        w = backend.convert_to_tensor(w)
        # L1范数和为1约束
        norms = ops.sum(w, axis=self.axis, keepdims=True)
        return w / (backend.epsilon() + norms)

    def get_config(self):
        return {
            "rate": self.rate,
            "axis": self.axis,
        }
