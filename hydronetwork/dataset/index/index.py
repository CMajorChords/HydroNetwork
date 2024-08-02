from datetime import datetime
from typing import Union, List, Tuple, Optional
from pandas import DataFrame
from tqdm.contrib.concurrent import process_map
from hydronetwork.dataset.check import check_nan_numpy, check_multiindex


def index_single_basin_timeseries(timeseries: DataFrame,
                                  target: str,
                                  lookback: int,
                                  horizon: int,
                                  features_lookback: Optional[List[str]] = None,
                                  features_bidirectional: Optional[List[str]] = None,
                                  keep_index_label: bool = False,
                                  ) -> Union[List[int], List[datetime]]:
    """
    获取单个流域时间序列中无缺失值样本的索引。

    :param timeseries: 时间序列数据，可以是DataFrame或者Series。index必须是时间戳，可以是multiindex，其他的level会被保留
    :param target: 目标时间序列的列名
    :param lookback: 过去时间步数
    :param horizon: 未来时间步数
    :param features_lookback: 只将lookback输入到网络的特征。注意绝大多数自回归都需要将target_col作为features_lookback的一部分
    :param features_bidirectional: 将lookback和horizon全部输入到网络的特征。例如气象数据可以提前预测再输入到降水径流模型，时间列也是可以提前知道的
    :param keep_index_label: 如果为True，返回的索引为时间戳，否则为整数
    :return: 无缺失值样本的索引
    """
    # 从DataFrame中提取目标时间序列和特征时间序列
    target = timeseries[target].values
    index_list = []
    length = target.shape[0] - lookback - horizon + 1
    match (features_lookback is not None, features_bidirectional is not None):
        case (True, True):
            features_lookback = timeseries[features_lookback].values
            features_bidirectional = timeseries[features_bidirectional].values
            # 循环提取每个输入-输出对
            for i in range(length):
                i_input_lookback = features_lookback[i:i + lookback]
                i_input_bidirectional = features_bidirectional[i:i + lookback + horizon]
                i_output = target[i + lookback:i + lookback + horizon]
                if check_nan_numpy((i_input_lookback, i_input_bidirectional, i_output)):
                    continue
                index_list.append(i)
        case (True, False):
            features_lookback = timeseries[features_lookback].values
            # 循环提取每个输入-输出对
            for i in range(length):
                i_input = features_lookback[i:i + lookback]
                i_output = target[i + lookback:i + lookback + horizon]
                if check_nan_numpy((i_input, i_output)):
                    continue
                index_list.append(i)
        case (False, True):
            features_bidirectional = timeseries[features_bidirectional].values
            # 循环提取每个输入-输出对
            for i in range(length):
                i_input = features_bidirectional[i:i + lookback + horizon]
                i_output = target[i + lookback:i + lookback + horizon]
                if check_nan_numpy((i_input, i_output)):
                    continue
                index_list.append(i)
        case (False, False):
            raise ValueError("过去特征和双向特征不能同时为空")
    if keep_index_label:
        index_list = timeseries.index[index_list].to_list()
    return index_list


def index_multi_basin_timeseries(timeseries: DataFrame,
                                 target: str,
                                 lookback: int,
                                 horizon: int,
                                 attributes: Optional[DataFrame] = None,
                                 features_lookback: Optional[List[str]] = None,
                                 features_bidirectional: Optional[List[str]] = None,
                                 keep_index_label: bool = False,
                                 ) -> Union[
    List[int],
    List[tuple],
    Tuple[List[int], List[int]],
    Tuple[List[tuple], List[int]],
]:
    """
    获取时间序列中无缺失值样本的索引。可选包括静态属性数据。

    :param timeseries: 时间序列数据，可以是DataFrame或者Series，列索引为multiindex，第一个level为流域id，第二个level为时间戳
    :param target: 目标时间序列的列名
    :param lookback: 过去时间步数
    :param horizon: 未来时间步数
    :param attributes: 静态属性数据，可以是DataFrame或者Series，index必须是流域id
    :param features_lookback: 只将lookback输入到网络的特征。注意绝大多数自回归都需要将target_col作为features_lookback的一部分
    :param features_bidirectional: 将lookback和horizon全部输入到网络的特征。例如气象数据可以提前预测再输入到降水径流模型，时间列也是可以提前知道的
    :param keep_index_label: 如果为True，返回标签索引，否则返回整数索引
    :return: 无缺失值样本的索引，若attributes不为空，返回的是一个tuple，第一个元素是时间序列索引，第二个元素是属性索引
    """
    # 按照列索引的第一个level分组,第一个level为流域id。
    timeseries_id_list = [timeseries_id for _, timeseries_id in timeseries.groupby(level=0)]
    num_basins = len(timeseries_id_list)
    # 使用并行处理生成索引，datetime_index_list是一个list，每个元素是一个tuple，第一个元素是流域id，第二个元素是索引
    timeseries_indices_list = process_map(index_single_basin_timeseries,
                                          timeseries_id_list,
                                          [target] * num_basins,
                                          [lookback] * num_basins,
                                          [horizon] * num_basins,
                                          [features_lookback] * num_basins,
                                          [features_bidirectional] * num_basins,
                                          [True] * num_basins,
                                          desc=f"正在为{num_basins}个流域的有效数据生成索引",
                                          total=num_basins)
    # 将所有的时间序列索引合并
    timeseries_indices_list = sum(timeseries_indices_list, [])
    gauge_id_list, datetime_index_list = zip(*timeseries_indices_list)
    if not keep_index_label:
        timeseries_indices_list = list(map(timeseries.index.get_loc, timeseries_indices_list))
    if attributes is not None:
        attributes_indices_list = list(map(attributes.index.get_loc, gauge_id_list))
        return timeseries_indices_list, attributes_indices_list
    else:
        return timeseries_indices_list


def index_timeseries(timeseries: DataFrame,
                     target: str,
                     lookback: int,
                     horizon: int,
                     attributes: Optional[DataFrame] = None,
                     features_lookback: Optional[List[str]] = None,
                     features_bidirectional: Optional[List[str]] = None,
                     keep_index_label: bool = False,
                     ) -> Union[
    List[int],
    List[tuple],
    Tuple[List[int], List[int]],
    Tuple[List[tuple], List[int]],
]:
    """
    获取时间序列中无缺失值样本的索引。

    :param timeseries: 时间序列数据，可以是DataFrame或者Series，列索引为multiindex，第一个level为流域id，第二个level为时间戳
    :param target: 目标时间序列的列名
    :param lookback: 过去时间步数
    :param horizon: 未来时间步数
    :param attributes: 静态属性数据，可以是DataFrame或者Series，index必须是流域id，只在多流域数据集中有效
    :param features_lookback: 只将lookback输入到网络的特征。注意绝大多数自回归都需要将target_col作为features_lookback的一部分
    :param features_bidirectional: 将lookback和horizon全部输入到网络的特征。例如气象数据可以提前预测再输入到降水径流模型，时间列也是可以提前知道的
    :param keep_index_label: 如果为True，返回的索引为时间戳，否则为整数
    :return: 无缺失值样本的索引，若attributes不为空且时间序列为多流域数据，返回的是一个tuple，第一个元素是时间序列索引，第二个元素是属性索引。
    """
    if check_multiindex(timeseries):
        return index_multi_basin_timeseries(timeseries=timeseries, target=target, lookback=lookback, horizon=horizon,
                                            attributes=attributes, features_lookback=features_lookback,
                                            features_bidirectional=features_bidirectional,
                                            keep_index_label=keep_index_label)
    else:
        return index_single_basin_timeseries(timeseries=timeseries, target=target, lookback=lookback, horizon=horizon,
                                             features_lookback=features_lookback,
                                             features_bidirectional=features_bidirectional,
                                             keep_index_label=keep_index_label)
