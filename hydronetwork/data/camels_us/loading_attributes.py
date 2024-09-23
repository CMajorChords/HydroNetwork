from typing import List, Optional
import pandas as pd
from pandas import DataFrame
from tqdm.contrib.concurrent import thread_map
from concurrent.futures import ThreadPoolExecutor
import hydronetwork.data.camels_us.camels_us_params as params


def load_single_type_attributes(attr: str,
                                root_path: str = params.camels_root_path,
                                ) -> DataFrame:
    """
    加载CAMELS数据集中的单个属性种类的属性数据

    :param root_path: CAMELS数据集根目录
    :param attr: 属性种类名称
    :return: 单个属性种类的属性数据
    """
    path = root_path + f"/camels_{attr}.txt"
    dtype = {"gauge_id": str, "huc_02": str, } if attr == "name" else {"gauge_id": str, }
    return pd.read_csv(path,
                       index_col="gauge_id",
                       dtype=dtype,
                       sep=";",
                       )


def load_attributes(gauge_id_list: Optional[List[str]] = None,
                    attr_type_name_list: Optional[List[str]] = None,
                    root_path: str = params.camels_root_path,
                    use_attr_name: Optional[List[str]] = None,
                    tqdm: bool = True,
                    ) -> DataFrame:
    """
    加载CAMELS数据集中指定流域的属性数据

    :param gauge_id_list: 流域ID列表, 默认为None, 表示加载所有流域的属性数据
    :param attr_type_name_list: 属性种类名称列表
    :param root_path: CAMELS数据集根目录，默认为"camels_params"中的root_path
    :param use_attr_name: 使用的属性名称列表
    :param tqdm: 是否显示进度条
    :return: 加载指定属性种类的所有属性数据
    """
    # 首先加载指定属性种类的属性数据
    if attr_type_name_list is None:
        attr_type_name_list = params.attributes_type_name
    len_attr_type_name_list = len(attr_type_name_list)
    if tqdm:
        attributes = pd.concat(
            thread_map(load_single_type_attributes,
                       attr_type_name_list,
                       [root_path] * len_attr_type_name_list,
                       desc=f"正在加载{len_attr_type_name_list}类流域静态属性",
                       total=len_attr_type_name_list,
                       ),
            axis=1)
    else:
        with ThreadPoolExecutor() as executor:
            attributes = pd.concat(
                executor.map(load_single_type_attributes,
                             attr_type_name_list,
                             [root_path] * len_attr_type_name_list,
                             ),
                axis=1)
    # 删除忽略的属性
    if use_attr_name is None:
        use_attr_name = params.use_attribute_name
    for attribute in use_attr_name:  # 检查忽略的属性是否在属性数据中
        if attribute not in attributes.columns:
            raise ValueError(f"属性{attribute}不在属性数据中")
    attributes = attributes[use_attr_name]
    if gauge_id_list is not None:
        attributes = attributes.loc[gauge_id_list]
    return attributes
