# %% 对汇流量执行对角线单位线法
from keras import ops


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
    print(matrix)

    for i in range(time_step + hydrograph_length - 1):
        print(ops.sum(ops.diagonal(matrix, offset=i - time_step + 1, axis1=-2, axis2=-1), axis=-1))
        result[:, i] = ops.sum(ops.diagonal(matrix, offset=i - time_step + 1, axis1=-2, axis2=-1), axis=-1)

    return result
