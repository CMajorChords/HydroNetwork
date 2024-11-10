# 评估降雨径流模型的表现结果并将结果可视化
import os
import pandas as pd
from typing import Optional, Sequence
from hydronetwork.dataset.dataset import HydroDataset
from pandas import DataFrame
import numpy as np
from hydronetwork.dataset.preprocessing.interpolate import interpolate_nan
from torch import Tensor
from itertools import chain
from keras import Model


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


def predict(model: Model,
            test_dataset: HydroDataset,
            predict_index: Optional[Sequence[int]] = None,
            verbose: int = "auto",
            interpolate: bool = False,
            ) -> DataFrame:
    """
    将模型的预测结果转换为DataFrame，方便评估预测结果。
    :param model: 训练好的keras模型
    :param test_dataset: 测试数据集，用来获取测试的时间索引和真实值。
    :param predict_index: 预测的时间步长索引。默认为None，表示预测的所有时间步长。
    :param verbose: 是否打印模型的预测进度。
    :param interpolate: 是否对预测结果和真实值中不连续的时间步长进行插值。
    :return:预测结果的DataFrame，index为时间索引。
            若predict_index为None，则列为"1_pred", "1_true", "2_pred", "2_true", ...
            若predict_index不为None，则列为"{predict_index[0]}_pred", "{predict_index[0]}_true", ...
    """
    if test_dataset.shuffle:
        raise ValueError('乱序后的测试数据集无法找到原来的时间索引')
    # 获取预测值
    prediction = model.predict(test_dataset,
                               batch_size=test_dataset.batch_size,
                               verbose=verbose)  # shape: (num_sample, time_steps)
    # 获取测试数据集的时间索引
    datetime_index = test_dataset.time_index  # 如果是多流域数据集，time_index是一个MultiIndex(gauge_id, datetime)
    # 将test_dataset的有效时间索引展平为一维数组,每个索引预测的是lookback之后的值，所以预测的时间索引应该在原来的基础上加上lookback
    # valid_int_index = [item + test_dataset.lookback for sublist in test_dataset.timeseries_index for item in sublist]
    valid_int_index = [item + test_dataset.lookback for item in chain(*test_dataset.timeseries_index)]
    # 获取预测的时间索引
    predict_datetime_index = datetime_index[valid_int_index]
    # 获取真实值
    observation = np.zeros_like(prediction)
    for smple_index, int_index in enumerate(valid_int_index):
        # 填充真实值，每个索引预测的是lookback：lookback+horizon之间的值
        observation[smple_index] = test_dataset.target[int_index:int_index + test_dataset.horizon]
    # 创建DataFrame
    prediction_df = pd.DataFrame(prediction,
                                 index=predict_datetime_index,
                                 columns=[f'{i}_pred' for i in range(1, prediction.shape[1] + 1)])
    observation_df = pd.DataFrame(observation,
                                  index=predict_datetime_index,
                                  columns=[f'{i}_true' for i in range(1, observation.shape[1] + 1)])
    # 选择预测的时间步长
    if predict_index is not None:
        predict_index = [predict_step - 1 for predict_step in predict_index]
        prediction_df = prediction_df.iloc[:, predict_index]
        observation_df = observation_df.iloc[:, predict_index]
    # 合并并转变columns的顺序
    result_df = pd.concat([prediction_df, observation_df], axis=1)
    result_df = result_df.reindex(columns=sorted(result_df.columns))
    # 插值
    if interpolate:
        # 按照原来的时间索引设置DataFrame的索引，这回导致很多NaN值
        result_df = result_df.reindex(datetime_index)
        # 插值
        result_df = interpolate_nan(result_df)
    # 确保result_df中所有的值为ndarray而不是Tensor
    for column in result_df.columns:
        if isinstance(result_df[column].values[0], Tensor):
            result_df[column] = result_df[column].apply(lambda x: x.numpy())
    return result_df
