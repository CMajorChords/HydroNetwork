# 定义多个水文模型常用的损失函数的类
from torch import Tensor
from torch.nn import Module
from hydronetwork.metric.torch import rmse, nse, kge


class RMSELoss(Module):
    """
    创建一个均方根误差损失函数。
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(y_pred: Tensor, y_true: Tensor) -> Tensor:
        return rmse(y_pred, y_true)


class NSELoss(Module):
    """
    创建一个Nash-Sutcliffe效率系数损失函数。由于NSE越接近1越好，所以这里的损失函数是1-NSE。
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(y_pred: Tensor, y_true: Tensor) -> Tensor:
        nse_result = nse(y_pred, y_true)
        return 1 - nse_result


class KGELoss(Module):
    """
    创建一个Kling-Gupta效率系数损失函数。
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(y_pred: Tensor, y_true: Tensor) -> Tensor:
        return kge(y_pred, y_true)
