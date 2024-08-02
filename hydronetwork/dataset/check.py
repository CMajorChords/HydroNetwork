# 创造检查数据的类和函数
from typing import Union, Sequence
from numpy import ndarray, isnan
from concurrent.futures import ProcessPoolExecutor
from pandas import concat
from pandas import DataFrame, Series, MultiIndex, DatetimeIndex


def check_nan_pandas(data: Union[DataFrame, Series]) -> Series:
    """
    检查数据中是否存在缺失值。

    :param data: 待检查的数据，
    :return: 检查结果, 返回存在缺失值列的缺失值数量
    """
    if isinstance(data, Series):
        data = data.to_frame()
    # 检查每一列是否存在缺失值并计算缺失值的数量
    nan_count = data.isnull().sum()
    nan_count.name = "nan_count"
    return nan_count[nan_count > 0]


def check_nan_numpy(data: Union[Sequence[ndarray], ndarray]) -> bool:
    """
    检查数据中是否存在缺失值。

    :param data: 待检查的数据，可以是多个ndarray组成的list或者tuple
    :return: 检查结果, 返回是否存在缺失值
    """
    return any((isnan(sub_data).any() for sub_data in data)) if isinstance(data, Sequence) else isnan(data).any()


def append_na_sequences(na_sequences: DataFrame,
                        na_sequence_start,
                        na_sequence_end,
                        na_sequence_length: int) -> DataFrame:
    """
    将连续缺失值的信息存入DataFrame。
    :param na_sequences: 存储连续缺失值的DataFrame
    :param na_sequence_start: 连续缺失值的起始时间
    :param na_sequence_end: 连续缺失值的结束时间
    :param na_sequence_length: 连续缺失值的长度
    :return: 存入连续缺失值信息后的DataFrame
    """
    # 将该段连续缺失值的信息存入DataFrame
    return concat([na_sequences, DataFrame({"start_time": [na_sequence_start],
                                            "end_time": [na_sequence_end],
                                            "length": [na_sequence_length]})],
                  axis=0,
                  ignore_index=True)


def check_consecutive_na_in_series(data: Series, threshold: int) -> DataFrame:
    """
    检查Series数据中是否存在连续缺失值。
    :param data: 要检查的数据
    :param threshold: 连续缺失值的阈值
    :return: 返回一个DataFrame，包含连续缺失值的起始时间、结束时间和长度
    """
    # 找到当前列的na值
    data_is_na = data.isna()
    # 得到当前列的长度
    time_length = len(data_is_na)
    # 初始化
    prev_is_na = False  # 用于记录前一个值是否为na
    na_sequences = DataFrame()  # 用于存储连续缺失值的信息
    na_sequence_start = None  # 用于记录连续缺失值的起始时间
    na_sequence_length = 0  # 用于记录连续缺失值的长度
    # 寻找连续缺失值
    for i, (time_index, is_na) in enumerate(data_is_na.items()):
        if is_na and not prev_is_na:  # 当前值为na，前一个值不为na，说明是连续缺失值的起始位置
            na_sequence_length = 1
            na_sequence_start = time_index
            # 判断当前值是否是最后一个值
            if i == time_length - 1 and threshold == 1:
                na_sequences = append_na_sequences(na_sequences, na_sequence_start, time_index, na_sequence_length)
        elif is_na and prev_is_na:  # 当前值为na，前一个值也为na，说明是连续缺失值的中间位置
            na_sequence_length += 1
            # 判断当前值是否是最后一个值
            if i == time_length - 1 and na_sequence_length >= threshold:
                na_sequences = append_na_sequences(na_sequences, na_sequence_start, time_index, na_sequence_length)
        elif not is_na and prev_is_na:  # 当前值不为na，前一个值为na，说明是连续缺失值的结束位置
            na_sequence_end = data_is_na.index[i - 1]
            # 将该段连续缺失值的信息存入DataFrame
            if na_sequence_length >= threshold:  # 连续缺失值的长度大于等于阈值
                na_sequences = append_na_sequences(na_sequences, na_sequence_start, na_sequence_end, na_sequence_length)
            # 重置
            na_sequence_start = None
            na_sequence_length = 0
        prev_is_na = is_na
    return na_sequences


def check_consecutive_na_in_dataframe(data: DataFrame, threshold: int) -> DataFrame:
    """
    检查数据中是否存在连续缺失值。
    :param data: 要检查的数据
    :param threshold: 连续缺失值的阈值
    :return: 返回一个DataFrame，其中index是一个multiindex，level为(列名, 起始时间)，columns为[结束时间，连续缺失值长度]
    """
    with ProcessPoolExecutor() as executor:
        result = concat(
            executor.map(check_consecutive_na_in_series,
                         [data[column] for column in data.columns],
                         [threshold] * len(data.columns)),
            keys=data.columns,
            axis=0,
            names=["column", "na_sequence_index"]
        )
    return result


def check_consecutive_nas(data: Union[DataFrame, Series], threshold: int = 10) -> DataFrame:
    """
    检查数据中是否存在连续缺失值。

    :param data: 要检查的数据
    :param threshold: 连续缺失值的阈值
    :return: 返回一个DataFrame，其中index是一个multiindex，level为(列名, 起始时间)，columns为[结束时间，连续缺失值长度]
    """
    if isinstance(data, DataFrame):
        check_result = check_consecutive_na_in_dataframe(data=data, threshold=threshold)
    else:
        check_result = check_consecutive_na_in_series(data=data, threshold=threshold)
    return check_result.astype({"start_time": "datetime64[ns]", "end_time": "datetime64[ns]", "length": "int"})


def check_multiindex(data: Union[DataFrame, Series]) -> bool:
    """
    检查数据集是否为包含多个level的multiindex。注意第一个level为"gauge_id"，第二个level为"datetime"。

    :param data: 待检查的数据集
    :return: 是否为multiindex
    """
    if isinstance(data.index, MultiIndex):
        if data.index.names == ["gauge_id", "datetime"]:
            return True
        else:
            raise ValueError("数据集index的level名称必须为['gauge_id', 'datetime']")
    elif isinstance(data.index, DatetimeIndex):
        return False
    else:
        raise ValueError("数据集index必须为MultiIndex或者DatetimeIndex")


def check_same_class(sequence: Sequence,
                     class_type: type) -> bool:
    """
    检查序列中的元素是否都是指定的类型。

    :param sequence: 待检查的序列
    :param class_type: 指定的类型
    :return: 是否都是指定的类型
    """
    return all(isinstance(element, class_type) for element in sequence)
