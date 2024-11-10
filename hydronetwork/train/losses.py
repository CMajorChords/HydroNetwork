from keras import ops
from keras import backend
from keras.api.losses import Loss


# %% functions
def mse(y_true, y_pred):
    """
    计算均方误差 (MSE)，支持批处理大小维度。
    返回值为每个样本的均方误差。

    MSE = mean((y_pred - y_true)^2), 按最后一个维度 n 求mean

    :param y_true: 实际值，形状为 (batchsize, n) 的 Tensor
    :param y_pred: 预测值，形状为 (batchsize, n) 的 Tensor
    :return: 每个样本的 MSE 损失，形状为 (batchsize,)
    """
    squared_diff = ops.square(y_pred - y_true)  # shape: (batchsize, n)
    # loss应该除以batchsize
    batchsize = y_pred.shape[0]
    return ops.mean(squared_diff, axis=-1) / batchsize  # shape: (batchsize,)


def rmse(y_true, y_pred):
    """
    计算均方根误差 (RMSE)，支持 batchsize 维度。
    返回值为每个样本的均方根误差。

    RMSE = sqrt(MSE)

    :param y_true: 实际值，形状为 (batchsize, n) 的 Tensor
    :param y_pred: 预测值，形状为 (batchsize, n) 的 Tensor
    :return: 每个样本的 RMSE 损失，形状为 (batchsize,)
    """
    return ops.sqrt(mse(y_true, y_pred))  # shape: (batchsize,)


def nse(y_true, y_pred):
    """
    计算 Nash-Sutcliffe 效率系数 (NSE)，支持 batchsize 维度。
    返回值为每个样本的 NSE 损失。

    NSE 越接近 1，表示模型预测结果越准确。
    NSE = 1 - sum((y_pred - y_true)^2) / sum((y_true - mean(y_true))^2)
    由于loss函数是最小化的，因此返回 1 - NSE

    :param y_true: 实际值，形状为 (batchsize, n) 的 Tensor
    :param y_pred: 预测值，形状为 (batchsize, n) 的 Tensor
    :return: 每个样本的 NSE 损失，形状为(batchsize,)
    """
    y_true_mean = ops.mean(y_true, axis=-1, keepdims=True)  # shape: (batchsize, 1)

    numerator = ops.sum(ops.square(y_pred - y_true), axis=-1)  # shape: (batchsize,)
    denominator = ops.sum(ops.square(y_true - y_true_mean), axis=-1)  # shape: (batchsize,)

    # 避免 denominator 为 0 引发除零错误，直接加上epsilon
    denominator = denominator + backend.epsilon()  # shape: (batchsize,)

    # loss应该除以batchsize
    batchsize = y_pred.shape[0]
    return numerator / denominator / batchsize  # shape: (batchsize,)


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
    y_true_mean = ops.mean(y_true, axis=-1, keepdims=True)  # shape: (batchsize, 1)
    y_pred_mean = ops.mean(y_pred, axis=-1, keepdims=True)  # shape: (batchsize, 1)

    y_true_std = ops.std(y_true, axis=-1)  # shape: (batchsize,)
    y_pred_std = ops.std(y_pred, axis=-1)  # shape: (batchsize,)

    # alpha = predicted_std / true_std
    alpha = y_pred_std / (y_true_std + backend.epsilon())  # 避免除 0，shape: (batchsize,)

    # beta = predicted_mean / true_mean
    beta = y_pred_mean / (y_true_mean + backend.epsilon())  # 避免除 0，shape: (batchsize, 1)
    beta = ops.squeeze(beta, axis=-1)  # shape: (batchsize,)

    # r = corr(y_pred, y_true)，通过协方差近似计算
    r_numerator = ops.sum((y_pred - y_pred_mean) * (y_true - y_true_mean), axis=-1)  # shape: (batchsize,)
    r_denominator = y_pred_std * y_true_std  # 避免除 0，shape: (batchsize,)

    r = r_numerator / r_denominator  # shape: (batchsize,)

    # KGE = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)
    result = ops.sqrt(ops.square(r - 1) + ops.square(alpha - 1) + ops.square(beta - 1))  # shape: (batchsize,)

    # loss应该除以batchsize
    batchsize = y_pred.shape[0]
    return result / batchsize  # shape: (batchsize,)


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
