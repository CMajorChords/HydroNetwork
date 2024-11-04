from typing import List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm.contrib.concurrent import thread_map
from xarray import Dataset
from pandas import DataFrame
from rich import print
import pandas as pd
from hydronetwork.data.camels_us_hourly.camels_us_hourly_params import camels_us_hourly_root_path, get_gauge_id_list
from hydronetwork.data.camels_us.utils import num_workers, split_list


def load_single_basin_forcing(gauge_id: str,
                              root_path: str = camels_us_hourly_root_path,
                              ) -> DataFrame:
    """
    加载单个流域的气象强迫数据

    :param gauge_id: 流域ID
    :param root_path: CAMELS数据集根目录，默认为"camels_params"中的root_path
    :return: 单个流域的气象强迫数据
    """
    # 根据降水数据来源获得数据路径 14400000_hourly_nldas.csv
    path = root_path + "/nldas_hourly/" + gauge_id + "_hourly_nldas.csv"
    # 读取并修改数据
    data = pd.read_csv(path,
                       index_col=["date"],
                       parse_dates=["date"],
                       encoding="utf-8",
                       )
    data.columns.name = "forcing_type"
    return data


def load_basins_forcing_with_threads(gauge_id_list: List[str],
                                     root_path: str,
                                     tqdm: bool = True,
                                     ) -> DataFrame:
    """
    多线程加载CAMELS_US_HOURLY数据集中部分流域的气象强迫数据

    :param gauge_id_list: 流域ID列表
    :param root_path: CAMELS数据集根目录，默认为"camels_params"中的root_path
    :param tqdm: 是否显示进度条
    :return: 所有流域的气象强迫数据
    """
    len_gauge_id_list = len(gauge_id_list)
    if tqdm:
        return pd.concat(
            thread_map(load_single_basin_forcing,
                       gauge_id_list,
                       [root_path] * len_gauge_id_list,
                       desc=f"正在加载{len_gauge_id_list}个流域的气象强迫数据",
                       total=len_gauge_id_list,
                       ),
            keys=gauge_id_list,
            names=["gauge_id", "datetime"],
            axis=0,
        )
    else:
        with ThreadPoolExecutor() as executor:
            return pd.concat(
                executor.map(load_single_basin_forcing,
                             gauge_id_list,
                             [root_path] * len_gauge_id_list,
                             ),
                keys=gauge_id_list,
                names=["gauge_id", "datetime"],
                axis=0,
            )


def load_basins_forcing(gauge_id_list: Optional[List[str]] = None,
                        root_path: str = camels_us_hourly_root_path,
                        multi_process: bool = False,
                        to_xarray: bool = False,
                        ) -> Union[DataFrame, Dataset]:
    """
    加载CAMELS_US_HOURLY数据集中所有流域的气象强迫数据

    :param gauge_id_list: 流域ID列表，如果为None，则加载所有流域的ID
    :param root_path: CAMELS数据集根目录，默认为"camels_params"中的root_path
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
            print(f"[bold]正在使用{num_workers}个进程加载{len(gauge_id_list)}个流域的气象强迫数据[/bold]")
            result = pd.concat(
                executor.map(load_basins_forcing_with_threads,
                             gauge_id_lists,
                             [root_path] * num_workers,  # 数据集根目录
                             ),
                axis=0,
            )
            print("[bold]加载完成！[/bold]")
        # result = pd.concat(process_map(load_basins_streamflow_with_threads,
        #                                gauge_id_lists, huc_02_lists,
        #                                [root_path] * num_workers,  # 数据集根目录
        #                                [False] * num_workers,  # 不显示进度条
        #                                desc=f"正在使用{num_workers}个进程加载{len(gauge_id_list)}个流域的流量数据",
        #                                total=num_workers
        #                                ),
        #                    axis=0,
        #                    )
    else:
        result = load_basins_forcing_with_threads(gauge_id_list, root_path)
    if to_xarray:
        result = result.to_xarray()
    return result


def load_forcing(gauge_id: Optional[Union[str, List[str]]] = None,
                 root_path: str = camels_us_hourly_root_path,
                 multi_process: bool = False,
                 to_xarray: bool = False,
                 astype: str = "float32",
                 ) -> Union[DataFrame, Dataset]:
    """
    加载CAMELS数据集中指定流域的气象强迫数据

    :param gauge_id:流域ID，可以是str(表示单个流域id)，也可以是List[str](表示多个流域id)
    :param root_path:CAMELS数据集根目录，默认为"camels_params"中的root_path
    :param multi_process:是否使用多进程，仅对多个流域有效
    :param to_xarray:是否转换为xarray格式，仅对多个流域有效
    :param astype:数据类型，默认为"float32"
    :return:指定流域的气象强迫数据
    """
    if gauge_id is None:
        return load_basins_forcing(gauge_id_list=None,
                                   root_path=root_path,
                                   multi_process=multi_process,
                                   to_xarray=to_xarray).astype(astype)
    elif isinstance(gauge_id, str):
        return load_single_basin_forcing(gauge_id=gauge_id,
                                         root_path=root_path).astype(astype)
    elif isinstance(gauge_id, list):
        return load_basins_forcing(gauge_id_list=gauge_id,
                                   root_path=root_path,
                                   multi_process=multi_process,
                                   to_xarray=to_xarray).astype(astype)
