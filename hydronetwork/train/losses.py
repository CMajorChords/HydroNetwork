from keras import ops
from keras import backend
from keras.api.losses import Loss


# %% functions
def mse(y_true, y_pred):
    """
    计算均方误差 (MSE)，支持批处理大小维度。
    注意，如果有多个batchsize的数据，本函数会将消除batchsize维度，将所有数据合并计算MSE。

    MSE = mean((y_pred - y_true)^2)
    越接近0，表示模型的预测结果越准确。

    :param y_true: 实际值，形状为 (batchsize, n) 的 Tensor
    :param y_pred: 预测值，形状为 (batchsize, n) 的 Tensor
    :return: 每个样本的 MSE 损失，形状为 (batchsize,)
    """
    return ops.mean(ops.square(y_pred - y_true)) # shape: 1


def rmse(y_true, y_pred):
    """
    计算均方根误差 (RMSE)
    返回值为每个样本的均方根误差。

    RMSE = sqrt(MSE)
    越接近0，表示模型的预测结果越准确。

    :param y_true: 实际值，形状为 (batchsize, n) 的 Tensor
    :param y_pred: 预测值，形状为 (batchsize, n) 的 Tensor
    :return: 每个样本的 RMSE 损失，形状为 (batchsize,)
    """
    return ops.sqrt(mse(y_true, y_pred))  # shape: 1


def nse(y_true,
        y_pred,
        ):
    """
    计算Nash-Sutcliffe效率系数，用于评估模型的预测结果。
    NSE越接近1，表示模型的预测结果越准确。
    NSE = 1 - sum((y_pred - y_true)^2) / sum((y_true - mean(y_true))^2)
    由于 loss 函数是最小化的，因此返回 1 - NSE
    注意，如果有多个batchsize的数据，本函数会将消除batchsize维度，将所有数据合并计算NSE。
    越接近0，表示模型的预测结果越准确。

    :param y_true: 真实的序列，Tensor
    :param y_pred: 预测的序列，Tensor
    :return: Nash-Sutcliffe效率系数
    """
    y_true_mean = ops.mean(y_true)
    return ops.sum(ops.square(y_pred - y_true)) / ops.sum(ops.square(y_true - y_true_mean)) # shape: 1


def kge(y_true, y_pred):
    """
    计算 Kling-Gupta 效率系数 (KGE)，支持 batchsize 维度。
    返回值为每个样本的 KGE 损失。

    KGE 越接近 1，表示模型预测效果越好。
    KGE = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)
    由于 loss 函数是最小化的，因此返回 1 - KGE

    :param y_true: 实际值，形状为 (batchsize, n) 的 Tensor
    :param y_pred: 预测值，形状为 (batchsize, n) 的 Tensor
    :return: 每个样本的 KGE 损失，形状为 (batchsize,)
    """
    y_true_mean = ops.mean(y_true)  # shape: 1
    y_pred_mean = ops.mean(y_pred)  # shape: 1

    y_true_std = ops.std(y_true)  # shape: 1
    y_pred_std = ops.std(y_pred)  # shape: 1

    # alpha = predicted_std / true_std
    alpha = y_pred_std / (y_true_std + backend.epsilon())  # 避免除 0，shape: 1

    # beta = predicted_mean / true_mean
    beta = y_pred_mean / (y_true_mean + backend.epsilon())  # 避免除 0，shape: 1

    # r = corr(y_pred, y_true)，通过协方差近似计算
    r_numerator = ops.sum((y_pred - y_pred_mean) * (y_true - y_true_mean))  # shape: 1
    r_denominator = y_pred_std * y_true_std  # 避免除 0，shape: 1

    r = r_numerator / r_denominator  # shape: (batchsize,)

    # KGE = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)
    result = ops.sqrt(ops.square(r - 1) + ops.square(alpha - 1) + ops.square(beta - 1))  # shape: 1

    return result  # shape: (batchsize,)


# %% classes

class MSELoss(Loss):
    def __init__(self, name="mse", reduction="sum_over_batch_size", dtype=None):
        super().__init__(reduction=reduction, name=name, dtype=dtype)

    def call(self, y_true, y_pred):
        """
        直接调用之前实现的 mse 函数
        """
        return mse(y_true, y_pred)


class RMSELoss(Loss):
    def __init__(self, name="rmse", reduction="sum_over_batch_size", dtype=None):
        super().__init__(reduction=reduction, name=name, dtype=dtype)

    def call(self, y_true, y_pred):
        """
        直接调用之前实现的 rmse 函数
        """
        return rmse(y_true, y_pred)


class NSELoss(Loss):
    def __init__(self, name="nse", reduction="sum_over_batch_size", dtype=None):
        super().__init__(reduction=reduction, name=name, dtype=dtype)

    def call(self, y_true, y_pred):
        """
        直接调用之前实现的 nse 函数
        """
        return nse(y_true, y_pred)


class KGELoss(Loss):
    def __init__(self, name="kge", reduction="sum_over_batch_size", dtype=None):
        super().__init__(reduction=reduction, name=name, dtype=dtype)

    def call(self, y_true, y_pred):
        """
        直接调用之前实现的 kge 函数
        """
        return kge(y_true, y_pred)
