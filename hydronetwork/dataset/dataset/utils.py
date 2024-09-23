import numpy as np
from numpy import ndarray, stack
from typing import Iterable


def check_length(*args: list):
    """
    检查输入的多个list的长度是否一致。

    :param args: 需要检查的数组
    """
    length_set = set(len(arg) for arg in args)
    if len(length_set) != 1:
        raise ValueError("输入的数组长度不一致")


def shuffle_list(*args: list,
                 seed: int = None,
                 ):
    """
    随机打乱输入的多个list, 并保持list中的元素一一对应。

    :param args: 需要打乱的数组
    :param seed: 随机种子
    :return: 打乱后的数组
    """
    # 检查输入的数组长度是否一致
    check_length(*args)
    # 设置随机种子
    if seed is not None:
        np.random.seed(seed)
    # 打乱数组
    idx = np.random.permutation(len(args[0])).tolist()
    # 如果只有一个数组，直接返回打乱后的数组
    if len(args) == 1:
        return [args[0][i] for i in idx]
    # 如果有多个数组，返回打乱后的多个数组
    return ([arg[i] for i in idx] for arg in args)


def split_single_list(list_for_split: list, batch_size: int):
    """
    按照顺序将list分割为多个大小为batch_size的list。
    :param list_for_split: 需要分割的数组
    :param batch_size: 分割的大小
    :return: 分割后的list
    """
    length = len(list_for_split)
    return [list_for_split[i:min(i + batch_size, length)] for i in range(0, length, batch_size)]


def split_list_by_batch(*args: list,
                        batch_size: int,
                        ):
    """
    将list按照batch_size分割。并保证list中的元素一一对应。

    :param args: 需要分割的数组
    :param batch_size: 分割的大小
    :return: 分割后的list
    """
    # 检查输入的数组长度是否一致
    check_length(*args)
    # 分割数组
    # 如果只有一个数组，直接返回分割后的数组
    if len(args) == 1:
        return split_single_list(args[0], batch_size)
    return (split_single_list(arg, batch_size) for arg in args)


def stack_2d_slices(data: ndarray,
                    index: Iterable[int],
                    start_windows_length: int,
                    end_windows_length: int,
                    axis: int = 0):
    """
    将二维矩阵按照index和windows_length进行切片，并按照axis进行堆叠。注意axis是新建的维度。
    :param data: 2D矩阵
    :param index: 切片的索引
    :param start_windows_length: 切片的起始点距离索引的长度
    :param end_windows_length: 切片的结束点距离索引的长度
    :param axis: 新建的维度
    :return: 堆积切片形成的新3D矩阵
    """
    return stack([data[i + start_windows_length: i + end_windows_length]
                  for i in index],
                 axis=axis)
