# 用于数据处理的多个插值办法
from pandas import DataFrame
from hydronetwork.dataset.check import check_multiindex


def interpolate_single_basin_nan(data: DataFrame,
                                 ) -> DataFrame:
    """
    对单个流域的DataFrame按照index: datetime进行线性插值。

    :param data: 单个流域的DataFrame, 索引为datetime
    :return: 插值后的DataFrame
    """
    return data.interpolate(method='time')


def interpolate_multi_basin_nan(data: DataFrame,
                                ) -> DataFrame:
    """
    对多个流域的DataFrame按照MultiIndex的第二级索引进行线性插值。

    :param data: 多个流域的DataFrame, 索引为MultiIndex，第一级索引为流域名，第二级索引为datetime
    :return: 插值后的DataFrame
    """
    return data.groupby(level=0).apply(interpolate_single_basin_nan)


def interpolate_nan(data: DataFrame,
                    ) -> DataFrame:
    """
    对DataFrame中的缺失值进行插值。

    :param data: DataFrame
    :return: 插值后的DataFrame
    """
    if check_multiindex(data):
        return interpolate_multi_basin_nan(data)
    else:
        return interpolate_single_basin_nan(data)
