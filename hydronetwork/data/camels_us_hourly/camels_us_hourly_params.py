from yaml import safe_load
from functools import cache
from pandas import Series
from typing import Optional
import random
import os
from hydronetwork.data.camels_us.loading_attributes import load_single_type_attributes

path = "configs/camels_us_hourly/CAMELS_HOURLY.yml"
with open(path, "r", encoding="utf-8") as f:
    config = safe_load(f)

# CAMELS_US_HOURLY数据集根目录
camels_us_hourly_root_path = config["camels_us_hourly_root_path"]


# CAMELS_US__HOURLY数据集中的流域ID
@cache
def get_gauge_id_list(id_type: str = "streamflow"):
    """
    获取CAMELS_US_HOURLY数据集中的流域ID

    :param id_type: ID类型，可选"streamflow"或"forcing"
    :return: 流域ID列表
    """
    if id_type == "streamflow":
        path_streamflow = camels_us_hourly_root_path + "/usgs_streamflow"
        filenames = [file for file in os.listdir(path_streamflow) if file.endswith('.csv')]
        return [file.split('-')[0] for file in filenames]
    elif id_type == "forcing":
        path_forcing = camels_us_hourly_root_path + "/nldas_hourly"
        filenames = [file for file in os.listdir(path_forcing) if file.endswith('.csv')]
        return [file.split('_')[0] for file in filenames]
    else:
        raise ValueError("id_type必须为streamflow或forcing")


def get_gauge_id(n: Optional[int] = None) -> str | list[str]:
    """
    随机获取n个流域ID，如果n为None，则获取所有流域ID
    :param n: 流域ID数量
    :return: 流域ID列表
    """
    gauge_id_list = get_gauge_id_list()
    if n is None:
        return gauge_id_list
    elif n == 1:
        # 随机获取一个流域ID(str)
        return random.choice(gauge_id_list)
    else:
        return random.sample(gauge_id_list, n)


@cache
def get_basins_area(source: str = "GAGESII") -> Series:
    """
    获取所有流域的面积

    :param source: 面积数据来源，可选"GAGESII"或"Geospatial Fabric"
    :return: 流域面积
    """
    if source == "GAGESII":
        source = "area_gages2"
    elif source == "Geospatial Fabric":
        source = "area_geospatial_fabric"
    else:
        raise ValueError("流域面积来源必须为GAGESII或Geospatial Fabric")
    topo = load_single_type_attributes("topo")
    return topo[source]


def get_basin_area(gauge_id: str, source: str = "GAGESII") -> float:
    """
    获取指定流域的面积

    :param gauge_id: 流域ID
    :param source: 面积数据来源，可选"GAGESII"或"Geospatial Fabric"
    :return: 流域面积
    """
    area = get_basins_area(source)
    return area[gauge_id]
