from typing import List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm.contrib.concurrent import thread_map
from rich import print
from xarray import Dataset
from pandas import DataFrame
import pandas as pd
from hydronetwork.data.camels_us.utils import split_list, num_workers
from hydronetwork.data.camels_us_hourly.camels_us_hourly_params import (camels_us_hourly_root_path,
                                                                        get_gauge_id_list,
                                                                        get_basin_area)


def load_single_basin_streamflow(gauge_id: str,
                                 root_path: str = camels_us_hourly_root_path,
                                 unit="m^3/s",
                                 ) -> DataFrame:
    """
    加载单个流域的流量数据

    :param gauge_id: 流域ID
    :param root_path: CAMELS数据集根目录，默认为"camels_params"中的root_path
    :param unit: 流量单位，默认为"m^3/s"，可以为"mm/h"
    :return: 单个流域的流量数据
    """
    # 读取数据
    path = root_path + "/usgs_streamflow/" + gauge_id + "-usgs-hourly.csv"
    data = pd.read_csv(path,
                       usecols=["date", "QObs(mm/h)"],
                       parse_dates=["date"],
                       index_col="date",
                       encoding="utf-8",
                       )
    # 单位换算，从mm/h转换为m^3/s
    if unit == "m^3/s":
        data["QObs(mm/h)"] = data["QObs(mm/h)"] * get_basin_area(gauge_id) / 3.6
    data.columns = ["streamflow"]
    data.columns.name = "streamflow"
    data.index.name = "datetime"
    return data


def load_basins_streamflow_with_threads(gauge_id_list: List[str],
                                        root_path: str,
                                        unit="m^3/s",
                                        tqdm: bool = True,
                                        ) -> DataFrame:
    """
    多线程加载CAMELS_US_HOURLY数据集中部分流域的流量数据

    :param gauge_id_list: 流域ID列表
    :param root_path: CAMELS数据集根目录，默认为"camels_params"中的root_path
    :param tqdm: 是否显示进度条
    :param unit: 流量单位，默认为"m^3/s"，可以为"mm/h"
    :return: 所有流域的流量数据
    """
    len_gauge_id_list = len(gauge_id_list)
    if tqdm:
        return pd.concat(
            thread_map(load_single_basin_streamflow,
                       gauge_id_list,
                       [root_path] * len_gauge_id_list,
                       [unit] * len_gauge_id_list,
                       desc=f"正在加载{len_gauge_id_list}个流域的流量数据",
                       total=len_gauge_id_list,
                       ),
            keys=gauge_id_list,
            names=["gauge_id", "datetime"],
            axis=0,
        )
    else:
        with ThreadPoolExecutor() as executor:
            return pd.concat(
                executor.map(load_single_basin_streamflow,
                             gauge_id_list,
                             [root_path] * len_gauge_id_list,
                             [unit] * len_gauge_id_list,
                             ),
                keys=gauge_id_list,
                names=["gauge_id", "datetime"],
                axis=0,
            )


def load_basins_streamflow(gauge_id_list: Optional[List[str]] = None,
                           root_path: str = camels_us_hourly_root_path,
                           unit: str = "m^3/s",
                           multi_process: bool = False,
                           to_xarray: bool = False,
                           ) -> Union[DataFrame, Dataset]:
    """
    加载CAMELS_US_HOURLY数据集中所有流域的流量数据

    :param gauge_id_list: 流域ID列表，如果为None，则加载所有流域的ID
    :param root_path: CAMELS数据集根目录，默认为"camels_params"中的root_path
    :param unit: 流量单位，默认为"m^3/s"，可以为"mm/h"
    :param multi_process: 是否使用多进程
    :param to_xarray: 是否转换为xarray格式
    :return: 所有流域的流量数据
    """
    # 如果没有指定流域ID列表，则加载所有流域的ID
    if gauge_id_list is None:
        gauge_id_list = get_gauge_id_list()
    # 判断使用多线程还是多进程, 并加载所有流域的流量数据
    if multi_process:
        # 先将流域ID列表和huc_02列表分成num_workers份
        gauge_id_lists = split_list(gauge_id_list, num_workers)
        # 使用多进程加载所有流域的流量数据
        with ProcessPoolExecutor() as executor:
            print(f"[bold]正在使用{num_workers}个进程加载{len(gauge_id_list)}个流域的流量数据[/bold]")
            result = pd.concat(
                executor.map(load_basins_streamflow_with_threads,
                             gauge_id_lists,
                             [root_path] * num_workers,  # 数据集根目录
                             [unit] * num_workers,  # 流量单位
                             [True] * num_workers,  # 显示进度条
                             ),
                axis=0,
            )
            print("[bold]加载完成！[/bold]")
    else:
        result = load_basins_streamflow_with_threads(gauge_id_list, root_path)
    if to_xarray:
        result = result.to_xarray()
    return result


def load_streamflow(gauge_id: Optional[Union[str, List[str]]] = None,
                    root_path: str = camels_us_hourly_root_path,
                    multi_process: bool = False,
                    to_xarray: bool = False,
                    unit="m^3/s",
                    ) -> Union[DataFrame, Dataset]:
    """
    加载CAMELS数据集中所有流域的流量数据

    :param gauge_id:流域ID，可以是str(表示单个流域id)，也可以是List[str](表示多个流域id)
    :param root_path:CAMELS数据集根目录，默认为"camels_params"中的root_path
    :param multi_process:是否使用多进程，仅对多个流域有效
    :param to_xarray:是否转换为xarray格式，仅对多个流域有效
    :param unit:流量单位，默认为"m^3/s"，可以为"mm/h"
    :return:指定流域的流量数据
    """
    if unit not in ("m^3/s", "mm/h"):
        raise ValueError("unit必须为m^3/s或mm/h")
    if gauge_id is None:
        return load_basins_streamflow(gauge_id_list=None,
                                      root_path=root_path,
                                      multi_process=multi_process,
                                      to_xarray=to_xarray,
                                      unit=unit, )
    elif isinstance(gauge_id, str):
        return load_single_basin_streamflow(gauge_id,
                                            root_path=root_path,
                                            unit=unit,
                                            )
    elif isinstance(gauge_id, list):
        return load_basins_streamflow(gauge_id_list=gauge_id,
                                      root_path=root_path,
                                      multi_process=multi_process,
                                      to_xarray=to_xarray,
                                      unit=unit, )
