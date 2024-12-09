# 创建处理数据归一化的工具函数和类
from typing import Union, Optional, Tuple, List

from networkx.algorithms.flow import minimum_cut_value
from pandas import DataFrame, Series


def normalize(data: Union[DataFrame, Series],
              use_cols: Optional[Union[str, List[str]]] = None,
              scale: Optional[List[float]] = None
              ) -> Tuple[Union[DataFrame, Series], DataFrame]:
    """
    将数据映射到指定范围内，对时间序列数据进行归一化处理。

    :param data: 时间序列数据
    :param use_cols: 需要归一化的列
    :param scale: 归一化范围，默认为[0, 1]，即将数据映射到0-1之间
    :return: 归一化后的数据和归一化参数
    """
    # 如果指定了需要归一化的列，则只对指定的列进行归一化
    if use_cols is not None:
        data = data[use_cols]
    # 如果没有指定归一化范围，则使用0-1作为默认范围
    if scale is None:
        scale = [0, 1]
    scale_lower_bound: float = scale[0]
    scale_upper_bound: float = scale[1]
    # 获取数据中每列的最大值、最小值
    min_series: Union[Series, float] = data.min(axis=0)
    max_series: Union[Series, float] = data.max(axis=0)
    # 对数据进行归一化处理
    data: Union[DataFrame, Series] = (data - min_series) / (max_series - min_series)
    # 对归一化之后的数据进行放缩
    data: Union[DataFrame, Series] = data * (scale_upper_bound - scale_lower_bound) + scale_lower_bound
    # 将最大值和最小值拼接成一个Dataframe，分别储存最大值、最小值、归一化范围，scale_params的行数等于data的列数，即特征数
    if isinstance(data, Series):
        scale_params = Series(
            data={"min": min_series,
                  "max": max_series,
                  "scale_lower_bound": scale_lower_bound,
                  "scale_upper_bound": scale_upper_bound,
                  },
            name=data.name
        )
    else:
        scale_params = DataFrame({
            "min": min_series,
            "max": max_series,
            "scale_lower_bound": scale_lower_bound,
            "scale_upper_bound": scale_upper_bound
        })
    return data, scale_params


def denormalize(data: Union[DataFrame, Series],
                scale_params: DataFrame,
                use_col: str,
                ) -> Union[DataFrame, Series]:
    """
    将提供的数据（一般是模型的预测值）反归一化到原始数据范围内。
    所用的归一化参数是指定列的归一化参数。即不管data中有多少列，只用指定列的归一化参数。
    一般用于将模型的预测值反归一化到原始数据范围内。

    :param data:
    :param scale_params:
    :param use_col:
    :return:
    """
    # 获取归一化参数
    min_value = scale_params.loc[use_col, 'min']
    max_value = scale_params.loc[use_col, 'max']
    scale_lower_bound = scale_params.loc[use_col, 'scale_lower_bound']
    scale_upper_bound = scale_params.loc[use_col, 'scale_upper_bound']
    # 对数据进行还原处理
    data = (data - scale_lower_bound) / (scale_upper_bound - scale_lower_bound)
    data = data * (max_value - min_value) + min_value
    return data


def restore(data: Union[DataFrame, Series],
            scale_params: DataFrame,
            use_cols: Optional[Union[str, List[str]]] = None,
            ) -> Union[DataFrame, Series]:
    """
    将归一化后的数据还原到原始数据范围内。

    :param data: 归一化后的数据
    :param scale_params: 归一化参数，包含四列：最小值（min）、最大值（max）、归一化下界（scale_lower_bound）、归一化上界（scale_upper_bound）
    :param use_cols: 需要还原的列
    :return: 反归一化后的数据
    """
    # 如果指定了需要归一化的列，则只对指定的列进行归一化
    if use_cols is not None:
        try:
            data = data[use_cols]
            scale_params = scale_params.loc[use_cols]
        except KeyError:
            raise ValueError(
                f"数据的columns和归一化参数的index不一致，数据的columns为{use_cols}，归一化参数的index为{scale_params.index}")
    else:
        try:
            use_cols = data.columns if isinstance(data, DataFrame) else [data.name]
            scale_params = scale_params.loc[use_cols]
        except KeyError:
            scale_cols = scale_params.index
            raise ValueError(
                f"数据的columns和归一化参数的index不一致，数据的columns为{use_cols}，归一化参数的index为{scale_cols}")
    # 如果数据是DataFrame，即有多个特征，则需要判断数据的columns和归一化参数的index是否一致
    if isinstance(data, DataFrame) and not data.columns.equals(scale_params.index):
        raise ValueError('数据的columns和归一化参数的index不一致')
    # 如果数据是Series，即只有一个特征，则需要判断数据的name和归一化参数的index是否一致
    elif isinstance(data, Series) and (not data.name == scale_params.name):
        raise ValueError('数据的name和归一化参数的index不一致')
    # 获取归一化参数
    min_series: Union[DataFrame, Series] = scale_params['min']
    max_series: Union[DataFrame, Series] = scale_params['max']
    scale_lower_bound: Union[DataFrame, Series] = scale_params['scale_lower_bound']
    scale_upper_bound: Union[DataFrame, Series] = scale_params['scale_upper_bound']
    # 对数据进行还原处理
    data: Union[DataFrame, Series] = (data - scale_lower_bound) / (scale_upper_bound - scale_lower_bound)
    data: Union[DataFrame, Series] = data * (max_series - min_series) + min_series
    return data


class Normalizer:
    """
    用于对时间序列数据进行归一化处理的类
    :param scale: 归一化范围，默认为[0, 1]，即将数据映射到0-1之间
    """

    def __init__(self,
                 scale: Optional[List[float]] = None
                 ):
        """
        初始化Normalizer对象

        :param scale: 归一化范围，默认为[0, 1]，即将数据映射到0-1之间
        """
        self.scale = scale
        self.use_cols = None
        self.scale_params = None

    def normalize(self,
                  data: Union[DataFrame, Series],
                  use_cols: Optional[Union[str, List[str]]] = None,
                  ) -> Union[DataFrame, Series]:
        """
        对时间序列数据进行归一化处理

        :param data: 时间序列数据
        :param use_cols: 需要归一化的列
        :return: 归一化后的数据
        """
        self.use_cols = use_cols
        data, self.scale_params = normalize(data, use_cols=use_cols, scale=self.scale)
        return data

    def denormalize(self,
                    data: Union[DataFrame, Series],
                    use_col: str,
                    ) -> Union[DataFrame, Series]:
        """
        对所有时间序列数据只使用指定列的参数进行反归一化处理

        :param data: 时间序列数据，一般是模型的预测值
        :param use_col: 进行反归一化的所用参数列
        :return: 反归一化后的数据
        """
        return denormalize(data, self.scale_params, use_col)

    def restore(self,
                data: Union[DataFrame, Series],
                use_cols: Optional[Union[str, List[str]]] = None,
                ) -> Union[DataFrame, Series]:
        """
        对时间序列数据进行反归一化处理

        :param data: 归一化后的数据
        :param use_cols: 需要还原的列
        :return: 反归一化后的数据
        """
        return restore(data, self.scale_params, use_cols=use_cols)

    def to_csv(self, path: str):
        """
        将归一化参数保存为csv文件

        :param path: 保存路径
        """
        self.scale_params.to_csv(path)

    def from_csv(self, path: str):
        """
        从csv文件中读取归一化参数

        :param path: 读取路径
        """
        self.scale_params = DataFrame.from_csv(path)
