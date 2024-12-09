from typing import Optional, Sequence
from pandas import DataFrame
import os
import pandas as pd


def combine_multi_dataframe_to_multiindex(path: [Sequence[str], str],
                                          index_col: str,
                                          level_0: Sequence[str],
                                          ignore_columns: Optional[Sequence[str]] = None,
                                          ignore_index: Optional[Sequence[str]] = None,
                                          ) -> DataFrame:
    """
    将多个DataFrame或Series合并为一个多级索引的DataFrame。主要用于多个流域模拟结果的合并。

    :param path: 多个DataFrame或Series的路径，可以是多个具体的文件路径，也可以是一个文件夹路径（将读取该文件夹下的所有csv文件）
    :param index_col: 索引列名
    :param level_0: 多级索引的第一级索引列表
    :param ignore_columns: 忽略的列名
    :param ignore_index: 忽略的索引名
    :return:  合并后的数据，一个列索引为多级索引的DataFrame
    """
    if isinstance(path, str):
        dataframe_list = []
        for i, file in enumerate(os.listdir(path)):
            if file.endswith('.csv'):
                dataframe_list.append(
                    pd.read_csv(os.path.join(path, file), index_col=index_col)
                )
    elif isinstance(path, Sequence):
        dataframe_list = [pd.read_csv(file, index_col=index_col) for file in path]
    else:
        raise ValueError('path参数应该是一个文件夹路径或者文件路径的列表')
    # 将多个DataFrame或Series合并为一个DataFrame
    combined_data = pd.concat(dataframe_list,
                              keys=level_0,
                              axis=1, )
    # 删除忽略的列和索引
    if ignore_columns is not None:
        combined_data = combined_data.drop(columns=ignore_columns, level=1)
    if ignore_index is not None:
        combined_data = combined_data.drop(index=ignore_index)
    return combined_data
