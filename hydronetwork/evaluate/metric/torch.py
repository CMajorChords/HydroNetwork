# 定义多个水文模型常用的损失函数和类
from torch.nn import Module
from torch import Tensor
from torch.nn.functional import mse_loss
from numpy import ndarray, float64, float32, sqrt, sum
from pandas import Series
from typing import Union, Sequence


def rmse(y_pred: Union[Tensor, ndarray, Series, Sequence],
         y_true: Union[Tensor, ndarray, Series, Sequence],
         ) -> Union[Tensor, float64, float32]:
    """
    计算均方根误差，用于评估模型的预测结果。
    RMSE越接近0，表示模型的预测结果越准确。
    RMSE = sqrt(mean((y_pred - y_true)^2))

    :param y_pred: 预测的序列
    :param y_true: 真实的序列
    :return: 均方根误差
    """
    if isinstance(y_pred, Tensor):
        return sqrt(mse_loss(y_pred, y_true))
    else:
        return sqrt(mse(y_pred, y_true))


def nse(y_pred: Union[Tensor, ndarray, Series, Sequence],
        y_true: Union[Tensor, ndarray, Series, Sequence],
        ) -> Union[Tensor, float64, float32]:
    """
    计算Nash-Sutcliffe效率系数，用于评估模型的预测结果。
    NSE越接近1，表示模型的预测结果越准确。
    NSE = 1 - sum((y_pred - y_true)^2) / sum((y_true - mean(y_true))^2)

    :param y_pred: 预测的序列
    :param y_true: 真实的序列
    :return: Nash-Sutcliffe效率系数
    """
    return 1 - sum((y_pred - y_true) ** 2) / sum((y_true - y_true.mean()) ** 2)


def mse(y_pred: Union[Tensor, ndarray, Series, Sequence],
        y_true: Union[Tensor, ndarray, Series, Sequence],
        ) -> Union[Tensor, float64, float32]:
    """
    计算均方误差，用于评估模型的预测结果。
    MSE越接近0，表示模型的预测结果越准确。
    MSE = mean((y_pred - y_true)^2)

    :param y_pred: 预测的序列
    :param y_true: 真实的序列
    :return: 均方误差
    """
    if isinstance(y_pred, Tensor):
        return mse_loss(y_pred, y_true)
    else:
        return ((y_pred - y_true) ** 2).mean()


def kge(y_pred: Union[Tensor, ndarray, Series, Sequence],
        y_true: Union[Tensor, ndarray, Series, Sequence],
        ) -> Union[Tensor, float64, float32]:
    """
    计算Kling-Gupta效率系数，用于评估模型的预测结果。
    KGE越接近1，表示模型的预测结果越准确。
    KGE = 1-sqrt((r-1)^2+(alpha-1)^2+(beta-1)^2)

    :param y_pred: 预测的序列
    :param y_true: 真实的序列
    :return: Kling-Gupta效率系数
    """
    y_pred_mean = y_pred.mean()
    y_true_mean = y_true.mean()
    y_pred_std = y_pred.std()
    y_true_std = y_true.std()
    alpha = y_pred_std / y_true_std
    beta = y_pred_mean / y_true_mean
    r = sum((y_pred - y_pred_mean) * (y_true - y_true_mean)) / (y_pred_std * y_true_std)
    return 1 - sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


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
