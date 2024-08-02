# 面向对象编程matplotlib绘图工具
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def count_subplots(data: DataFrame,
                   ncols: int,
                   ) -> (int, int):
    """
    统计子图的行数和列数

    :param data: 需要绘制的数据
    :param ncols: 每行的子图数量，如果data中的列数不是ncols的整数倍，则最后一行的子图数量可能会少于ncols
    :return: 行数、列数
    """
    nrows = len(data.columns) // ncols
    # 如果列数不能整除ncols，则行数加1
    if len(data.columns) % ncols != 0:
        nrows += 1
    # 如果data中的列数小于ncols，则列数为data中的列数
    if len(data.columns) < ncols:
        ncols = len(data.columns)
    return nrows, ncols


def get_square_axes(nrows: int,
                    ncols: int,
                    length: int,
                    aspect: float = 1.2,
                    ) -> (Figure, Axes):
    """
    根据子图的行数和列数设置fig_size，使得子图的长宽比尽可能接近1
    :param nrows: 子图的行数
    :param ncols: 子图的列数
    :param length: 每个子图的边长
    :param aspect: 子图的长宽比
    :return: fig, axes
    """
    fig = plt.figure(figsize=(length * ncols * aspect, length * nrows))
    axes = fig.subplots(nrows, ncols)
    return fig, axes
