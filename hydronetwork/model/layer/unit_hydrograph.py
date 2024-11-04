# %% 对汇流量执行对角线单位线法
import keras
from keras import ops


@keras.saving.register_keras_serializable(package='Custom', name='diagonal_sum')
def diagonal_sum(matrix):
    """
    对batch中的每个矩阵执行反对角线错位相加
    :param matrix: size=batch_size*T*hydrograph_length
    :return: size=batch_size*(T+hydrograph_length-1)
    """
    batch_size, time_step, hydrograph_length = matrix.shape

    # 初始化 result
    result_shape = (batch_size, time_step + hydrograph_length - 1)
    result = ops.zeros(result_shape)

    # 将hydrograph_length维度上的元素反转，方便对角线错位相加
    matrix = ops.flip(matrix, axis=-2)

    for i in range(time_step + hydrograph_length - 1):
        result[:, i] = ops.sum(ops.diagonal(matrix, offset=i - time_step + 1, axis1=-2, axis2=-1), axis=-1)

    return result


# %% 测试对角线错位相加
# import numpy as np
#
# x = np.random.rand(2, 3, 4)
# print(f"x: {x}")
# y = diagonal_sum(x)
# print(f"y: {y.cpu()}")

# %%一个更好的单位先设置：脉冲响应函数
# 定义脉冲响应函数 h(x,t)
import numpy as np
def pulse_response(x, t, C, D):
    part1 = x / (2 * t * np.sqrt(np.pi * D * t))
    part2 = np.exp(-((C * t - x) ** 2) / (4 * D * t))
    return part1 * part2
