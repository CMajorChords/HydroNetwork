# 创建用于处理camels数据集的工具函数
from os import cpu_count
from typing import List, Tuple, Union, Optional, Sequence
import hydronetwork.data.camels_us.camels_us_params as params
from hydronetwork.data.camels_us.loading_attributes import load_single_type_attributes

num_workers = cpu_count()


def get_gauge_id(root_path: str = params.camels_root_path,
                 get_huc_02: bool = False,
                 ignore_gauge_id: Optional[List[str]] = None,
                 n: Optional[int] = None,
                 ) -> Union[
    str,
    List[str],
    Tuple[str, str],
    Tuple[List[str], List[str]]
]:
    """
    获取指定流域ID。如果没有指定流域ID，则返回所有流域ID

    :param root_path: CAMELS数据集根目录
    :param get_huc_02: 是否获取HUC02编码
    :param n: 流域ID数量，默认为选择所有流域ID
    :param ignore_gauge_id: 忽略的流域ID列表
    :return: n个随机流域ID和对应的HUC02编码（可选）
    """
    camels_name = load_single_type_attributes("name", root_path)
    # 删除忽略的流域ID
    if ignore_gauge_id is not None:
        camels_name.drop(index=ignore_gauge_id, inplace=True)
    # 如果n为None，则返回所有流域ID, 否则返回n个随机流域ID
    data = camels_name if n is None else camels_name.sample(n)
    # 如果get_huc_02为True，则返回流域ID和对应的HUC02编码，否则只返回流域ID
    gauge_id_list = data.index.tolist()
    if get_huc_02:
        huc_02_list = data["huc_02"].tolist()
        return (gauge_id_list[0], huc_02_list[0]) if n == 1 else (gauge_id_list, huc_02_list)
    else:
        return gauge_id_list[0] if n == 1 else gauge_id_list


def split_list(sequence: Sequence, n: int) -> List[List]:
    """
    将容器分成n份, 如果不能整除，则前面的部分比后面的部分多1个元素
    :param sequence: 待分割的容器
    :param n: 分割的份数
    :return: 分割后的列表
    """
    k, m = divmod(len(sequence), n)
    return [sequence[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]
