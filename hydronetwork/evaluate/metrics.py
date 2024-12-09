import numpy as np


def mse(y_true,
        y_pred,
        ):
    """
    计算均方误差，用于评估模型的预测结果。
    MSE越接近0，表示模型的预测结果越准确。
    MSE = mean((y_pred - y_true)^2)

    :param y_true: 真实的序列，Tensor
    :param y_pred: 预测的序列，Tensor
    :return: 均方误差
    """
    return np.mean(np.square(y_pred - y_true))


def rmse(y_true,
         y_pred
         ):
    """
    计算均方根误差，用于评估模型的预测结果。
    RMSE越接近0，表示模型的预测结果越准确。
    RMSE = sqrt(mean((y_pred - y_true)^2))

    :param y_true: 真实的序列，Tensor
    :param y_pred: 预测的序列，Tensor
    :return: 均方根误差
    """
    return np.sqrt(mse(y_true, y_pred))


def nse(y_true,
        y_pred,
        ):
    """
    计算Nash-Sutcliffe效率系数，用于评估模型的预测结果。
    NSE越接近1，表示模型的预测结果越准确。
    NSE = 1 - sum((y_pred - y_true)^2) / sum((y_true - mean(y_true))^2)

    :param y_true: 真实的序列，Tensor
    :param y_pred: 预测的序列，Tensor
    :return: Nash-Sutcliffe效率系数
    """
    y_true_mean = np.mean(y_true, axis=-1, keepdims=True)
    return 1 - np.sum(np.square(y_pred - y_true)) / np.sum(np.square(y_true - y_true_mean))


def kge(y_true,
        y_pred,
        ):
    """
    计算Kling-Gupta效率系数，用于评估模型的预测结果。
    KGE越接近1，表示模型的预测结果越准确。
    KGE = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)

    :param y_true: 真实的序列，Tensor
    :param y_pred: 预测的序列，Tensor
    :return: Kling-Gupta效率系数
    """
    y_true_mean = np.mean(y_true)
    y_pred_mean = np.mean(y_pred)
    y_true_std = np.std(y_true)
    y_pred_std = np.std(y_pred)

    alpha = y_pred_std / y_true_std
    beta = y_pred_mean / y_true_mean
    r = np.sum((y_pred - y_pred_mean) * (y_true - y_true_mean)) / (y_pred_std * y_true_std)

    return 1 - np.sqrt(np.square(r - 1) + np.square(alpha - 1) + np.square(beta - 1))
