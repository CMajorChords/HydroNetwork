# 划分时间序列数据集为多个子集
from typing import Union
from pandas import DataFrame, Series, concat, to_datetime
from numpy import cumsum
from datetime import datetime
from tqdm.contrib.concurrent import process_map
from hydronetwork.dataset.check import check_multiindex, check_same_class


def split_ts_by_int(data: Union[DataFrame, Series],
                    split_list: [int],
                    ) -> [DataFrame]:
    """
    将时间序列数据集按照样本数量划分为多个子集
    :param data: 时间序列数据集。index为时间
    :param split_list: 用于划分数据集的列表。列表中整数表示每个子集的样本数量
    :return: 划分后的多个子集
    """
    # 数据检查
    if sum(split_list) > len(data):
        raise ValueError("各子集样本数量之和大于数据集样本数量")
    # 累加计算每个子集的起始和结束索引
    end_list = cumsum(split_list)
    start_list = [a - b for a, b in zip(end_list, split_list)]
    # 划分数据集
    split_data = map(lambda start, end: data.iloc[start:end], start_list, end_list)
    return split_data


def split_ts_by_datetime(data: Union[DataFrame, Series],
                         split_list: [datetime],
                         ) -> [DataFrame]:
    """
    将时间序列数据集按照时间点划分为多个子集
    :param data: 时间序列数据集。index为时间
    :param split_list: 用于划分数据集的列表。列表中时间表示时间点
    :return: 划分后的多个子集
    """
    # 数据检查
    split_list.sort()
    if split_list[-1] >= data.index[-1]:
        raise ValueError("划分时间点大于等于数据集最后一个时间")
    elif split_list[0] <= data.index[0]:
        raise ValueError("划分时间点小于等于数据集第一个时间")
    split_list = [data.index[0]] + split_list + [data.index[-1]]
    # 划分数据集
    start_list = split_list[:-1]
    end_list = split_list[1:]
    split_data = map(lambda start, end: data.loc[start:end], start_list, end_list)
    return split_data


def split_ts_by_float(data: Union[DataFrame, Series],
                      split_list: [float],
                      ) -> [DataFrame]:
    """
    将时间序列数据集按照样本数量占比划分为多个子集
    :param data: 时间序列数据集。index为时间
    :param split_list: 用于划分数据集的列表。列表中浮点数表示每个子集的样本数量占比
    :return: 划分后的多个子集
    """
    # 数据检查
    if sum(split_list) > 1:
        raise ValueError("各子集样本数量之和大于1")
    # 将占比转换为样本数量
    split_list = [int(percent * len(data)) for percent in split_list]
    if sum(split_list) < len(data):  # 如果样本数量之和小于数据集样本数量
        split_list[-1] += len(data) - sum(split_list)  # 将剩余样本数量加到最后一个子集
    return split_ts_by_int(data, split_list)


def split_ts_for_single_basin(data: Union[DataFrame, Series],
                              split_list: ([float], [int], [datetime], [str]),
                              ) -> [DataFrame]:
    """
    将单个流域的时间序列数据集按照样本数量或时间点划分为多个子集
    :param data: 单个流域的数据集。index为时间
    :param split_list: 用于划分数据集的列表。列表中浮点数表示每个子集的样本数量占比，整数表示每个子集的样本数量，时间表示时间点
    :return: 划分后的多个子集
    """
    # 数据检查
    if check_same_class(split_list, int):  # 如果划分列表元素为整数
        return split_ts_by_int(data, split_list)
    elif check_same_class(split_list, datetime):  # 如果划分列表元素为时间
        return split_ts_by_datetime(data, split_list)
    elif check_same_class(split_list, float):  # 如果划分列表元素为浮点数
        return split_ts_by_float(data, split_list)
    elif check_same_class(split_list, str):  # 如果划分列表元素为字符串
        split_list = to_datetime(split_list)
        return split_ts_by_datetime(data, split_list)
    else:
        raise ValueError("划分列表元素必须为int、datetime、float或表示时间的str")


def split_timeseries(data: [DataFrame, Series],
                     split_list: ([float], [int], [datetime], [str]),
                     ) -> [DataFrame]:
    """
    将单个流域或多个流域的时间序列数据集按照样本数量或时间点划分为多个子集
    :param data: 数据集。单个流域的数据集index为时间。多个流域的数据集index应为multiindex，第一层为流域id，第二层为时间
    :param split_list: 用于划分数据集的列表。列表中浮点数表示每个子集的样本数量占比，整数表示每个子集的样本数量，时间表示时间点
    :return: 划分后的多个子集
    """
    if check_multiindex(data):  # 如果数据集index为multiindex
        # 将数据集按流域id分组
        data_group_by_basin = {gauge_id: group_data.droplevel(level=0) for gauge_id, group_data in
                               data.groupby(level=0)}

        # 多进程处理每个流域
        len_gauge_id_list = len(data_group_by_basin)
        split_data = process_map(split_ts_for_single_basin,
                                 data_group_by_basin.values(),
                                 [split_list] * len_gauge_id_list,
                                 desc=f"正在划分{len_gauge_id_list}个流域的数据集",
                                 total=len_gauge_id_list,
                                 )  # 划分时间序列数据集为多个子集
        split_data = zip(*split_data)  # 将每个时间段的数据集合并成一个tuple

        # 将tuple中的数据集合并成一个DataFrame，index为multiindex，第一层为流域id，第二层为时间
        def concat_data(data_iter, keys):
            return concat(data_iter, keys=keys, axis=0)

        return map(concat_data, split_data, data_group_by_basin.keys())
    else:  # 如果数据集index为时间，即单个流域的数据集
        return split_ts_for_single_basin(data, split_list)
