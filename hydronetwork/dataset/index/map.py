# 映射数据为符合模型的输入和输出
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from hydronetwork.dataset.check import check_nan_numpy
from typing import Union, Optional, List, Tuple, TypeAlias
from numpy import ndarray, expand_dims, vstack, tile, empty

DataFrame: TypeAlias = pd.DataFrame
Series: TypeAlias = pd.Series

def map_target_and_features_lookback(data: DataFrame,
                                     target: str,
                                     lookback: int,
                                     horizon: int,
                                     features_lookback: List[str],
                                     ) -> Tuple[ndarray, ndarray]:
    """
    将目标时间序列映射为输入和输出。用于所有特征的未来值都是未知的情况。

    :param data: 时间序列数据，可以是DataFrame或者Series，列索引不能为multiindex
    :param target: 目标时间序列的列名
    :param lookback: 过去时间步数
    :param horizon: 未来时间步数
    :param features_lookback: 只将lookback输入到网络的特征。注意绝大多数自回归都需要将target_col作为features_lookback的一部分
    :return: 目标时间序列的输入和输出
    """
    # 从DataFrame中提取目标时间序列和特征时间序列
    target = data[target].values
    features_lookback = data[features_lookback].values
    # 初始化输入和输出
    input_data = empty((0, lookback, features_lookback.shape[1]))
    output_data = empty((0, horizon))
    length = target.shape[0] - lookback - horizon + 1
    # 循环提取每个输入-输出对
    for i in range(length):
        i_input = features_lookback[i:i + lookback]
        i_output = target[i + lookback:i + lookback + horizon]
        if check_nan_numpy([i_input, i_output]):
            continue
        else:
            i_input = expand_dims(i_input, axis=0)
            i_output = expand_dims(i_output, axis=0)
            input_data = vstack((input_data, i_input))
            output_data = vstack((output_data, i_output))
    return input_data, output_data


def map_target_and_features_bidirectional(data: DataFrame,
                                          target: str,
                                          lookback: int,
                                          horizon: int,
                                          features_bidirectional: List[str],
                                          ) -> Tuple[ndarray, ndarray]:
    """
    将目标时间序列映射为输入和输出。用于全部特征的未来值都是已知的情况。

    :param data: 时间序列数据，可以是DataFrame或者Series，列索引不能为multiindex
    :param target: 目标时间序列的列名
    :param lookback: 过去时间步数
    :param horizon: 未来时间步数
    :param features_bidirectional: 将lookback和horizon全部输入到网络的特征。例如气象数据可以提前预测再输入到降水径流模型，时间列也是可以提前知道的
    :return: 目标时间序列的输入和输出
    """
    # 从DataFrame中提取目标时间序列和特征时间序列
    target = data[target].values
    features_bidirectional = data[features_bidirectional].values
    # 初始化输入和输出
    input_data = empty((0, lookback + horizon, features_bidirectional.shape[1]))
    output_data = empty((0, horizon))
    length = target.shape[0] - lookback - horizon + 1
    # 循环提取每个输入-输出对
    for i in range(length):
        i_input = features_bidirectional[i:i + lookback + horizon]
        i_output = target[i + lookback:i + lookback + horizon]
        if check_nan_numpy([i_input, i_output]):
            continue
        else:
            i_input = expand_dims(i_input, axis=0)
            i_output = expand_dims(i_output, axis=0)
            input_data = vstack((input_data, i_input))
            output_data = vstack((output_data, i_output))
    return input_data, output_data


def map_target_and_features_bidirectional_lookback(data: DataFrame,
                                                   target: str,
                                                   lookback: int,
                                                   horizon: int,
                                                   features_lookback: Optional[List[str]],
                                                   features_bidirectional: Optional[List[str]],
                                                   ) -> Tuple[ndarray, ndarray, ndarray]:
    """
    将目标时间序列映射为输入和输出。用于部分特征的未来值是未知的，部分特征的未来值是已知的情况。

    :param data: 时间序列数据，可以是DataFrame或者Series，列索引不能为multiindex
    :param target: 目标时间序列的列名
    :param lookback: 过去时间步数
    :param horizon: 未来时间步数
    :param features_lookback: 只将lookback输入到网络的特征。注意绝大多数自回归都需要将target_col作为features_lookback的一部分
    :param features_bidirectional: 将lookback和horizon全部输入到网络的特征。例如气象数据可以提前预测再输入到降水径流模型，时间列也是可以提前知道的
    :return: 根据features_lookback和features_bidirectional的不同，返回三个ndarray或者两个ndarray
    """
    # 从DataFrame中提取目标时间序列和特征时间序列
    target = data[target].values
    features_lookback = data[features_lookback].values
    features_bidirectional = data[features_bidirectional].values
    # 初始化输入和输出
    input_lookback = empty((0, lookback, features_lookback.shape[1]))
    input_bidirectional = empty((0, lookback + horizon, features_bidirectional.shape[1]))
    output_data = empty((0, horizon))
    length = target.shape[0] - lookback - horizon + 1
    # 循环提取每个输入-输出对
    for i in range(length):
        i_input_lookback = features_lookback[i:i + lookback]
        i_input_bidirectional = features_bidirectional[i:i + lookback + horizon]
        i_output = target[i + lookback:i + lookback + horizon]
        if check_nan_numpy([i_input_lookback, i_input_bidirectional, i_output]):
            continue
        else:
            i_input_lookback = expand_dims(i_input_lookback, axis=0)
            i_input_bidirectional = expand_dims(i_input_bidirectional, axis=0)
            i_output = expand_dims(i_output, axis=0)
            input_lookback = vstack((input_lookback, i_input_lookback))
            input_bidirectional = vstack((input_bidirectional, i_input_bidirectional))
            output_data = vstack((output_data, i_output))
    return input_lookback, input_bidirectional, output_data


def map_data_for_single_basin(timeseries_data: Union[DataFrame, Series],
                              target: str,
                              lookback: int,
                              horizon: int,
                              features_lookback: Optional[List[str]] = None,
                              features_bidirectional: Optional[List[str]] = None,
                              ) -> Union[Tuple[ndarray, ndarray, ndarray], Tuple[ndarray, ndarray]]:
    """
    将时间序列分别处理为目标时间序列和特征时间序列，用于构建Dataset。

    :param timeseries_data: 时间序列数据，可以是DataFrame或者Series，列索引不能为multiindex
    :param target: 目标时间序列的列名
    :param lookback: 过去时间步数
    :param horizon: 未来时间步数
    :param features_lookback: 只将lookback输入到网络的特征。注意绝大多数自回归都需要将target_col作为features_lookback的一部分
    :param features_bidirectional: 将lookback和horizon全部输入到网络的特征。例如气象数据可以提前预测再输入到降水径流模型，时间列也是可以提前知道的
    :return: 根据features_lookback和features_bidirectional的不同，返回三个ndarray或者两个ndarray
    """
    match (features_lookback is None, features_bidirectional is None):
        case (True, True):
            raise ValueError("过去特征和双向特征不能同时为空")
        case (False, False):
            if set(features_lookback) & set(features_bidirectional):
                raise ValueError("过去特征和双向特征不能有重复")
            return map_target_and_features_bidirectional_lookback(data=timeseries_data, target=target,
                                                                  lookback=lookback,
                                                                  horizon=horizon, features_lookback=features_lookback,
                                                                  features_bidirectional=features_bidirectional)
        case (False, True):
            return map_target_and_features_lookback(data=timeseries_data,
                                                    target=target,
                                                    features_lookback=features_lookback,
                                                    lookback=lookback,
                                                    horizon=horizon)
        case (True, False):
            return map_target_and_features_bidirectional(data=timeseries_data, target=target, lookback=lookback,
                                                         horizon=horizon,
                                                         features_bidirectional=features_bidirectional)


def map_data_for_basins(timeseries_data: DataFrame,
                        static_data: DataFrame,
                        target: str,
                        lookback: int,
                        horizon: int,
                        features_lookback: Optional[List[str]] = None,
                        features_bidirectional: Optional[List[str]] = None,
                        ) -> Union[Tuple[Tuple, Tuple, List, Tuple], Tuple[Tuple, List, Tuple]]:
    """
    处理多个流域的时间序列数据，用于构建Dataset。

    :param timeseries_data: 多个流域的时间序列数据，列索引应当为multiindex(流域id, 特征名)
    :param static_data: 静态特征，不随时间变化, 列索引为流域id，行索引为静态特征
    :param target: 目标时间序列的列名
    :param lookback: 过去时间步数
    :param horizon: 未来时间步数
    :param features_lookback: 只将lookback输入到网络的特征。注意绝大多数自回归都需要将target_col作为features_lookback的一部分
    :param features_bidirectional: 将lookback和horizon全部输入到网络的特征。例如气象数据可以提前预测再输入到降水径流模型，时间列也是可以提前知道的
    :return: 根据features_lookback和features_bidirectional的不同，返回三个ndarray或者两个ndarray
    """
    # 针对不同的features_lookback和features_bidirectional选择不同的函数进行处理
    match (features_lookback is None, features_bidirectional is None):
        case (True, True):
            raise ValueError("过去特征和双向特征不能同时为空")
        case (False, False):
            if set(features_lookback) & set(features_bidirectional):
                raise ValueError("过去特征和双向特征不能有重复")
            params = ((timeseries_id.droplevel(0, axis=0),  # 按照列索引的第一个level分组,第一个level为流域id。
                       target,
                       lookback,
                       horizon,
                       features_lookback,
                       features_bidirectional,)
                      for _, timeseries_id in timeseries_data.groupby(level=0))
            params = zip(*params)  # zip解包后的数据中每个tuple中是多个流域的一个参数
            # 多进程处理数据
            with ProcessPoolExecutor() as executor:
                input_lookback, input_bidirectional, output_data = zip(
                    *executor.map(map_target_and_features_bidirectional_lookback, *params))
            # 处理静态特征
            input_static = process_static_data(static_data, timeseries_data, output_data)
            return input_lookback, input_bidirectional, input_static, output_data
        case (False, True):
            params = ((timeseries_id.droplevel(0, axis=1),  # 按照列索引的第一个level分组
                       target,
                       lookback,
                       horizon,
                       features_lookback,)
                      for _, timeseries_id in timeseries_data.groupby(level=0))  # 每个tuple中是一个流域的多个参数
            params = zip(*params)  # zip解包后的数据中每个tuple中是多个流域的一个参数
            # 多进程处理数据
            with ProcessPoolExecutor() as executor:
                input_lookback, output_data = zip(*executor.map(map_target_and_features_lookback, *params))
            # 处理静态特征
            input_static = process_static_data(static_data, timeseries_data, output_data)
            return input_lookback, input_static, output_data
        case (True, False):
            params = ((timeseries_id.droplevel(0, axis=1),  # 按照列索引的第一个level分组
                       target,
                       lookback,
                       horizon,
                       features_bidirectional,)
                      for _, timeseries_id in timeseries_data.groupby(level=0))  # 每个tuple中是一个流域的多个参数
            params = zip(*params)  # zip解包后的数据中每个tuple中是多个流域的一个参数
            # 多进程处理数据
            with ProcessPoolExecutor() as executor:
                input_bidirectional, output_data = zip(*executor.map(map_target_and_features_bidirectional, *params))
            # 处理静态特征
            input_static = process_static_data(static_data, timeseries_data, output_data)
            return input_bidirectional, input_static, output_data


def process_static_data(static_data: DataFrame,
                        timeseries_data: DataFrame,
                        output_data: ndarray) -> List[ndarray]:
    """
    处理静态特征数据，将静态特征数据扩展为与时间序列数据相同的形状。

    :param static_data: 静态特征数据，列索引为流域id，行索引为静态特征
    :param timeseries_data: 时间序列数据，行索引应当为multiindex(流域id, datetime)
    :param output_data: 一个list，每个元素是一个流域的所有样本的输出
    :return:
    """
    if set(static_data.index) != set(timeseries_data.index.levels[0]):
        raise ValueError("静态特征的流域id和时间序列的流域id不一致")
    static_data = static_data.loc[timeseries_data.index.levels[0]].values
    num_samples_per_basin = [i.shape[0] for i in output_data]
    return [tile(static_data[basin_idx],
                 (num_single_basin_samples, 1)
                 )
            for basin_idx, num_single_basin_samples in enumerate(num_samples_per_basin)]
