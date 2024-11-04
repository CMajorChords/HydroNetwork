from keras import ops


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
    return ops.mean(ops.square(y_pred - y_true))


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
    return ops.sqrt(mse(y_true, y_pred))


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
    y_true_mean = ops.mean(y_true)
    return ops.sum(ops.square(y_pred - y_true)) / ops.sum(ops.square(y_true - y_true_mean)) - 1  # 最小化这个值，所以加负号


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
    y_true_mean = ops.mean(y_true)
    y_pred_mean = ops.mean(y_pred)
    y_true_std = ops.std(y_true)
    y_pred_std = ops.std(y_pred)

    alpha = y_pred_std / y_true_std
    beta = y_pred_mean / y_true_mean
    r = ops.sum((y_pred - y_pred_mean) * (y_true - y_true_mean)) / (y_pred_std * y_true_std)

    return 1 - ops.sqrt(ops.square(r - 1) + ops.square(alpha - 1) + ops.square(beta - 1))
