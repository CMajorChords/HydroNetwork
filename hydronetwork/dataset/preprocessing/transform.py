# 创建处理数据分布转换及其逆变换的工具函数
from typing import Union
from pandas import DataFrame, Series
from numpy import log, exp, log1p, expm1
from scipy.stats import boxcox


def log_transform(data: Union[DataFrame, Series],
                  add_one: bool = True,
                  ) -> Union[DataFrame, Series]:
    """
    对数据进行log变换

    :param data: 要变换的数据，可以是DataFrame或Series
    :param add_one: 是否对数据加1再进行变换
    :return: 对数变换后的数据
    """
    return log1p(data) if add_one else log(data)


def log_inverse_transform(transformed_data: Union[DataFrame, Series],
                          add_one: bool = True,
                          ) -> Union[DataFrame, Series]:
    """
    对log变换后的数据进行逆变换

    :param transformed_data: log变换后的数据，可以是DataFrame或Series
    :param add_one: 数据取log变换前是否加了1
    :return: 逆变换后的数据
    """
    return expm1(transformed_data) if add_one else exp(transformed_data)


def box_cox_transform(data: Union[DataFrame, Series],
                      add_small_value: bool = False,
                      ) -> (Union[DataFrame, Series], Union[Series, float]):
    """
    对数据进行Box-Cox变换，注意Box-Cox变换要求数据都是正数。

    :param data: 要变换的数据，可以是DataFrame或Series
    :param add_small_value: 是否对数据加一个小值再进行变换
    :return: 变换后的数据和lambda值
    """
    if add_small_value:
        data += 1e-6
    if isinstance(data, DataFrame):  # 如果是DataFrame，对每一列进行变换
        transformed_data = DataFrame(index=data.index)
        lambda_values = Series(index=data.columns)
        for column in data.columns:
            transformed_data[column], lambda_values[column] = boxcox(data[column].values)
    else:  # 如果是Series，对整个Series进行变换
        transformed_data, lambda_values = boxcox(data.values)
        transformed_data = Series(transformed_data, index=data.index)
    return transformed_data, lambda_values


def box_cox_inverse_transform(transformed_data: Union[DataFrame, Series],
                              lambda_values: Union[Series, float],
                              add_small_value: bool = False,
                              ) -> Union[DataFrame, Series]:
    """
    对Box-Cox变换后的数据进行逆变换。

    :param transformed_data: Box-Cox变换后的数据，可以是DataFrame或Series
    :param lambda_values: Box-Cox变换时得到的lambda值，可以是DataFrame或Series
    :param add_small_value: 数据取Box-Cox变换前是否加了一个小值
    :return: 逆变换后的数据
    """

    def inverse_box_cox_single(single_column: Series, lambda_value: float) -> Series:
        if lambda_value == 0:
            return exp(single_column)
        else:
            return (single_column * lambda_value + 1) ** (1 / lambda_value)

    if isinstance(transformed_data, DataFrame):  # 如果是DataFrame，对每一列进行逆变换
        if not isinstance(lambda_values, Series):
            raise ValueError("如果transformed_data是DataFrame，lambda_values必须是Series")
        data = DataFrame(index=transformed_data.index)
        for column in transformed_data.columns:
            data[column] = inverse_box_cox_single(transformed_data[column], lambda_values[column])
    else:  # 如果是Series，对整个Series进行逆变换
        if not isinstance(lambda_values, float):
            raise ValueError("如果transformed_data是Series，lambda_values必须是float")
        data = inverse_box_cox_single(transformed_data, lambda_values)
    if add_small_value:
        data -= 1e-6
    return data


class Transformer:
    def __init__(self, data: Union[DataFrame, Series]):
        # 初始化数据
        self.add_one = None
        self.lambda_values = None
        self.add_small_value = None

    def log_transform(self,
                      data: Union[DataFrame, Series],
                      add_one: bool = True
                      ) -> Union[DataFrame, Series]:
        """
        对数据进行log变换
        :param data: 要变换的数据，可以是DataFrame或Series
        :param add_one: 是否对数据加1再进行变换
        """
        self.add_one = add_one
        return log_transform(data, add_one=add_one)

    def log_inverse_transform(self,
                              transformed_data: Union[DataFrame, Series]
                              ) -> Union[DataFrame, Series]:
        """
        对log变换后的数据进行逆变换
        :param transformed_data: log变换后的数据，可以是DataFrame或Series
        """
        return log_inverse_transform(transformed_data, add_one=self.add_one)

    def box_cox_transform(self,
                          data: Union[DataFrame, Series],
                          add_small_value: bool = False
                          ) -> (Union[DataFrame, Series], Union[Series, float]):
        """
        对数据进行Box-Cox变换，注意Box-Cox变换要求数据都是正数。
        :param data: 要变换的数据，可以是DataFrame或Series
        :param add_small_value: 是否对数据加一个小值再进行变换
        """
        self.add_small_value = add_small_value
        transformed_data, self.lambda_values = box_cox_transform(data, add_small_value=add_small_value)
        return transformed_data

    def box_cox_inverse_transform(self,
                                  transformed_data: Union[DataFrame, Series]
                                  ) -> Union[DataFrame, Series]:
        """
        对Box-Cox变换后的数据进行逆变换。
        :param transformed_data: Box-Cox变换后的数据，可以是DataFrame或Series
        """
        return box_cox_inverse_transform(transformed_data,
                                         self.lambda_values,
                                         add_small_value=self.add_small_value)
