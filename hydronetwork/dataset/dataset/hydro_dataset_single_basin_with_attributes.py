from typing import List, Optional
import keras
from pandas import DataFrame
from hydronetwork.dataset.index import index_single_basin_timeseries
from hydronetwork.dataset.dataset.utils import shuffle_list, split_list_by_batch, stack_2d_slices
from numpy import ndarray


class HydroDatasetSingleBasinWithAttributes(keras.utils.PyDataset):
    """
    为降单个流域的时间序列数据和静态属性数据构建的Dataset类。
    """

    def __init__(self,
                 timeseries: DataFrame,
                 target: str,
                 lookback: int,
                 horizon: int,
                 attributes: DataFrame,
                 batch_size: int,
                 shuffle: bool,
                 features_lookback: Optional[List[str]] = None,
                 features_bidirectional: Optional[List[str]] = None,
                 **kwargs,
                 ):
        """
        初始化HydroDatasetBasins。

        :param timeseries: 时间序列数据，DataFrame。index必须是时间戳
        :param target: 要预测的时间序列的列名
        :param lookback: 过去时间步数
        :param horizon: 未来时间步数
        :param attributes: 静态属性数据，DataFrame。index必须是流域id
        :param batch_size: 批次大小
        :param features_lookback: 只将lookback输入到网络的特征
        :param features_bidirectional: 将lookback和horizon全部输入到网络的特征
        :param shuffle: 是否打乱数据
        :param kwargs: https://keras.io/api/models/model_training_apis/#fit-method
        """
        super().__init__(**kwargs)
        self.lookback = lookback
        self.horizon = horizon
        self.target = timeseries[target].values
        self.attributes = attributes.values
        self.num_static_features = len(attributes.columns)
        match features_lookback, features_bidirectional:
            case (None, None):
                raise ValueError("过去特征和双向特征不能同时为空")
            case (None, _):
                self.features_bidirectional = timeseries[features_bidirectional].values
                self.num_bidirectional_features = len(features_bidirectional)
                self.get_item_func = self.get_item_bidirectional
            case (_, None):
                self.features_lookback = timeseries[features_lookback].values
                self.num_lookback_features = len(features_lookback)
                self.get_item_func = self.get_item_lookback
            case (_, _):
                self.features_lookback = timeseries[features_lookback].values
                self.num_lookback_features = len(features_lookback)
                self.features_bidirectional = timeseries[features_bidirectional].values
                self.num_bidirectional_features = len(features_bidirectional)
                self.get_item_func = self.get_item_lookback_bidirectional
        # 检索时间序列数据有效索引
        timeseries_index \
            = index_single_basin_timeseries(timeseries=timeseries, target=target, lookback=lookback, horizon=horizon,
                                            features_lookback=features_lookback,
                                            features_bidirectional=features_bidirectional)
        # 打乱数据
        if shuffle:
            timeseries_index = shuffle_list(timeseries_index)
        self.timeseries_index = split_list_by_batch(timeseries_index, batch_size=batch_size)

    def __len__(self) -> int:
        """
        获取样本数量。

        :param self: HydroDatasetBasins类的实例
        :return: 样本数量
        """
        return len(self.timeseries_index)

    def __getitem__(self, idx: int):
        """
        获取单个样本的输入和target输出。

        :param self: HydroDatasetBasins类的实例
        :param idx: 样本的索引
        :return: 输入、attributes输入和target输出
        """
        return self.get_item_func(idx)

    def get_item_lookback(self, idx: int) -> ((ndarray, ndarray), ndarray):
        """
        获取单个样本的lookback输入和target输出。

        :param self: HydroDatasetBasins类的实例
        :param idx: 样本的索引
        :return: lookback输入、attributes输入和target输出
        """
        # 设置输入的时间序列数据
        input_timeseries = stack_2d_slices(data=self.features_lookback,
                                           index=self.timeseries_index[idx],
                                           start_windows_length=0,
                                           end_windows_length=self.lookback)
        output_target = stack_2d_slices(data=self.target,
                                        index=self.timeseries_index[idx],
                                        start_windows_length=self.lookback,
                                        end_windows_length=self.lookback + self.horizon,
                                        )
        return (input_timeseries, self.attributes), output_target

    def get_item_bidirectional(self, idx: int) -> ((ndarray, ndarray), ndarray):
        """
        获取单个样本的bidirectional输入和target输出。

        :param self: HydroDatasetBasins类的实例
        :param idx: 样本的索引
        :return: bidirectional输入、attributes输入和target输出
        """
        # 设置输入的时间序列数据
        input_timeseries = stack_2d_slices(data=self.features_bidirectional,
                                           index=self.timeseries_index[idx],
                                           start_windows_length=0,
                                           end_windows_length=self.lookback + self.horizon)
        output_target = stack_2d_slices(data=self.target,
                                        index=self.timeseries_index[idx],
                                        start_windows_length=self.lookback,
                                        end_windows_length=self.lookback + self.horizon,
                                        )
        return input_timeseries, self.attributes, output_target

    def get_item_lookback_bidirectional(self, idx: int) -> ((ndarray, ndarray, ndarray), ndarray):
        """
        获取单个样本的lookback和bidirectional输入和target输出。

        :param self: HydroDatasetBasins类的实例
        :param idx: 样本的索引
        :return: lookback、bidirectional输入、attributes输入和target输出
        """
        # 设置输入的时间序列数据
        input_timeseries_lookback = stack_2d_slices(data=self.features_lookback,
                                                    index=self.timeseries_index[idx],
                                                    start_windows_length=0,
                                                    end_windows_length=self.lookback)
        input_timeseries_bidirectional = stack_2d_slices(data=self.features_bidirectional,
                                                         index=self.timeseries_index[idx],
                                                         start_windows_length=0,
                                                         end_windows_length=self.lookback + self.horizon)
        output_target = stack_2d_slices(data=self.target,
                                        index=self.timeseries_index[idx],
                                        start_windows_length=self.lookback,
                                        end_windows_length=self.lookback + self.horizon,
                                        )
        return (input_timeseries_lookback, input_timeseries_bidirectional, self.attributes), output_target
