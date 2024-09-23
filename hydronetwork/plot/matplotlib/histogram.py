# 绘制直方图
from hydronetwork.plot.matplotlib.utils import count_subplots, get_square_axes
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Optional


def plot_hist_on_ax(ax: Axes,
                    data: DataFrame | Series,
                    index: int,
                    bins: int,
                    title: bool,
                    ):
    """
    绘制data中某一列的直方图

    :param ax: 用于绘制的子图
    :param data: 需要绘制的数据
    :param index: 需要绘制的数据的列索引
    :param bins: 直方图的柱子数量
    :param title: 是否显示标题
    """
    ax.hist(data.iloc[:, index], bins=bins)
    ax.set_title(data.columns[index]) if title else None


def hist(data: DataFrame | Series,
         show: bool = False,
         bins: int = 60,
         ncols: int = 3,
         length: int = 2,
         axes_title: bool = True,
         figure_title: Optional[str] = None,
         ) -> (Figure, Axes):
    """
    绘制data中每一列的直方图

    :param data: 需要绘制的数据
    :param show: 是否显示图像
    :param bins: 直方图的柱子数量
    :param ncols: 子图的列数
    :param length: 每个子图的边长
    :param axes_title: 子图的标题
    :param figure_title: 图像的标题
    :return: fig, axes
    """
    data = data.to_frame() if isinstance(data, Series) else data
    # 根据ncols计算nrows，并获取正方形的axes
    nrows, ncols = count_subplots(data, ncols=ncols)
    fig, axes = get_square_axes(ncols=ncols, nrows=nrows, length=length)
    if isinstance(axes, Axes):
        axes = [axes]
    else:
        axes = axes.ravel()

    # 隐藏多余的子图
    for ax in axes[len(data.columns):]:
        fig.delaxes(ax)
    # 循环绘制多个子图
    for i, ax in enumerate(axes[:len(data.columns)]):
        plot_hist_on_ax(ax=ax, data=data, index=i, bins=bins, title=axes_title)

    # 设置图像标题
    fig.suptitle(figure_title) if figure_title else None
    # 防止各个图之间重叠
    plt.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.1)
    # 返回图像对象和子图对象
    plt.show() if show else None
    return fig, axes
