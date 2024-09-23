# 为降雨径流模拟深度学习模型准备dataset和dataloader
from typing import List, Optional, Union
from pandas import DataFrame
from hydronetwork.dataset.check import check_multiindex
from hydronetwork.dataset.dataset.hydro_dataset_multi_basin_with_attributes import HydroDatasetMultiBasinWithAttributes
from hydronetwork.dataset.dataset.hydro_dataset_multi_basin_without_attributes import \
    HydroDatasetMultiBasinWithoutAttributes
from hydronetwork.dataset.dataset.hydro_dataset_single_basin_with_attributes import \
    HydroDatasetSingleBasinWithAttributes
from hydronetwork.dataset.dataset.hydro_dataset_single_basin_without_attributes import \
    HydroDatasetSingleBasinWithoutAttributes
from typing import Union

HydroDataset = Union[
    HydroDatasetMultiBasinWithAttributes,
    HydroDatasetMultiBasinWithoutAttributes,
    HydroDatasetSingleBasinWithAttributes,
    HydroDatasetSingleBasinWithoutAttributes
]


def get_dataset(timeseries: DataFrame,
                lookback: int,
                horizon: int,
                target: str,
                attributes: Optional[DataFrame] = None,
                batch_size: int = 512,
                features_lookback: Optional[List[str]] = None,
                features_bidirectional: Optional[List[str]] = None,
                shuffle: bool = False,
                **kwargs,
                ) -> Union[HydroDatasetMultiBasinWithAttributes,
HydroDatasetMultiBasinWithoutAttributes,
HydroDatasetSingleBasinWithAttributes,
HydroDatasetSingleBasinWithoutAttributes
]:
    """
    根据输入数据的类型，返回不同的Dataset对象。

    :param timeseries: 时间序列数据，DataFrame。index必须是MultiIndex，level0为流域id，level1为时间戳
    :param target: 要预测的时间序列的列名
    :param lookback: 过去时间步数
    :param horizon: 未来时间步数
    :param attributes: 静态属性数据，可以是DataFrame或者Series。index必须是流域id
    :param batch_size: 批大小
    :param features_lookback: 只将lookback输入到网络的特征
    :param features_bidirectional: 将lookback和horizon全部输入到网络的特征
    :param shuffle: 是否打乱数据
    :return: HydroDatasetMultiBasinWithAttributes, HydroDatasetMultiBasinWithoutAttributes,
             HydroDatasetSingleBasinWithAttributes, HydroDatasetSingleBasinWithoutAttributes
    """
    # 检查timeseries是否是multiindex
    if isinstance(timeseries, DataFrame):
        # 如果是MultiIndex，则为多流域数据
        if check_multiindex(timeseries):
            if attributes is None:
                dataset = HydroDatasetMultiBasinWithoutAttributes(timeseries=timeseries,
                                                                  target=target,
                                                                  lookback=lookback,
                                                                  horizon=horizon,
                                                                  batch_size=batch_size,
                                                                  shuffle=shuffle,
                                                                  features_lookback=features_lookback,
                                                                  features_bidirectional=features_bidirectional,
                                                                  **kwargs)
            else:
                dataset = HydroDatasetMultiBasinWithAttributes(timeseries=timeseries, target=target, lookback=lookback,
                                                               horizon=horizon, attributes=attributes,
                                                               batch_size=batch_size,
                                                               shuffle=shuffle,
                                                               features_lookback=features_lookback,
                                                               features_bidirectional=features_bidirectional,
                                                               **kwargs)
        # 如果不是MultiIndex，则为单流域数据
        else:
            if attributes is None:
                dataset = HydroDatasetSingleBasinWithoutAttributes(timeseries=timeseries,
                                                                   target=target,
                                                                   lookback=lookback,
                                                                   horizon=horizon,
                                                                   batch_size=batch_size,
                                                                   shuffle=shuffle,
                                                                   features_lookback=features_lookback,
                                                                   features_bidirectional=features_bidirectional,
                                                                   **kwargs)
            else:
                dataset = HydroDatasetSingleBasinWithAttributes(timeseries=timeseries,
                                                                target=target,
                                                                lookback=lookback,
                                                                horizon=horizon,
                                                                attributes=attributes,
                                                                batch_size=batch_size,
                                                                shuffle=shuffle,
                                                                features_lookback=features_lookback,
                                                                features_bidirectional=features_bidirectional,
                                                                **kwargs)
    else:
        raise ValueError("timeseries必须是DataFrame")
    return dataset
