from typing import List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm.contrib.concurrent import thread_map
from xarray import Dataset
from pandas import DataFrame
from rich import print
import pandas as pd
import hydronetwork.data.camels.camels_params as params
from hydronetwork.data.camels.loading_attributes import load_single_type_attributes
from hydronetwork.data.camels.loading_forcing import load_single_basin_forcing
from hydronetwork.data.camels.loading_streamflow import load_single_basin_streamflow
from hydronetwork.data.camels.utils import get_gauge_id, num_workers, split_list


def load_single_basin_timeseries(gauge_id: str,
                                 huc_02: Optional[str] = None,
                                 root_path: str = params.camels_root_path,
                                 source: str = params.forcing_source,
                                 ignore_columns: Optional[List[str]] = None,
                                 ) -> DataFrame:
    """
    加载单个流域的气象强迫和流量数据

    :param gauge_id: 流域ID
    :param huc_02: 流域的HUC02编码
    :param root_path: CAMELS数据集根目录，默认为"camels_params"中的root_path
    :param source: 降水数据来源, 默认为"daymet", 可选值为"daymet"、“maurer"和”nldas"
    :param ignore_columns: 忽略的列名列表
    :return: 单个流域的气象强迫和流量数据
    """
    forcing = load_single_basin_forcing(gauge_id, huc_02, root_path, source, ignore_columns)
    streamflow = load_single_basin_streamflow(gauge_id, huc_02, root_path)
    timeseries = pd.concat([forcing, streamflow], axis=1)
    timeseries.columns.name = "timeseries_type"
    return timeseries


def load_basins_timeseries_with_threads(gauge_id_list: List[str],
                                        huc_02_list: List[str],
                                        root_path: str,
                                        source: str,
                                        ignore_columns: Optional[List[str]],
                                        tqdm: bool = True,
                                        ) -> DataFrame:
    """
    多线程加载CAMELS数据集中部分流域的气象强迫和流量数据

    :param gauge_id_list: 流域ID列表
    :param huc_02_list: 流域的HUC02编码列表
    :param root_path: CAMELS数据集根目录，默认为"camels_params"中的root_path
    :param source: 降水数据来源, 默认为"daymet", 可选值为"daymet"、“maurer"和”nldas"
    :param ignore_columns: 忽略的列名列表
    :param tqdm: 是否显示进度条
    :return: 所有流域的气象强迫和流量数据
    """
    len_gauge_id_list = len(gauge_id_list)
    if tqdm:
        return pd.concat(
            thread_map(load_single_basin_timeseries,
                       gauge_id_list, huc_02_list,
                       [root_path] * len_gauge_id_list,
                       [source] * len_gauge_id_list,
                       [ignore_columns] * len_gauge_id_list,
                       desc=f"正在加载{len_gauge_id_list}个流域的气象强迫和流量数据",
                       total=len_gauge_id_list,
                       leave=True,
                       ),
            keys=gauge_id_list,
            names=["gauge_id", "datetime"],
            axis=0,
        )
    else:
        with ThreadPoolExecutor() as executor:
            return pd.concat(
                executor.map(load_single_basin_timeseries,
                             gauge_id_list, huc_02_list,
                             [root_path] * len_gauge_id_list,
                             [source] * len_gauge_id_list,
                             [ignore_columns] * len_gauge_id_list,
                             ),
                keys=gauge_id_list,
                names=["gauge_id", "datetime"],
                axis=0,
            )


def load_basins_timeseries(gauge_id_list: Optional[List[str]] = None,
                           root_path: str = params.camels_root_path,
                           source: str = params.forcing_source,
                           ignore_columns: Optional[List[str]] = None,
                           multi_process: bool = False,
                           to_xarray: bool = False,
                           ) -> Union[DataFrame, Dataset]:
    """
    加载CAMELS数据集中所有流域的气象强迫和流量数据

    :param gauge_id_list: 流域ID列表，如果为None，则加载所有流域的ID
    :param root_path: CAMELS数据集根目录，默认为"camels_params"中的root_path
    :param source: 降水数据来源, 默认为"daymet", 可选值为"daymet"、“maurer"和”nldas"
    :param ignore_columns: 忽略的列名列表
    :param multi_process: 是否使用多进程
    :param to_xarray: 是否转换为xarray格式
    :return: 所有流域的气象强迫和流量数据
    """
    # 如果没有指定流域ID列表，则加载所有流域的ID
    if gauge_id_list is None:
        gauge_id_list, huc_02_list = get_gauge_id(root_path, True)
    else:
        huc_02_list = load_single_type_attributes("name", root_path).loc[gauge_id_list, "huc_02"].tolist()
    # 判断使用多线程还是多进程, 并加载所有流域的气象强迫和流量数据
    if multi_process:
        # 将流域ID列表和huc_02列表分成num_workers份
        gauge_id_lists = split_list(gauge_id_list, num_workers)
        huc_02_lists = split_list(huc_02_list, num_workers)
        # 使用多进程加载所有流域的气象强迫和流量数据
        with ProcessPoolExecutor() as executor:
            print(f"[bold]正在使用{num_workers}个进程加载{len(gauge_id_list)}个流域的气象强迫和流量数据……[/bold]")
            result = pd.concat(
                executor.map(load_basins_timeseries_with_threads,
                             gauge_id_lists, huc_02_lists,
                             [root_path] * num_workers,
                             [source] * num_workers,
                             [ignore_columns] * num_workers,
                             [False] * num_workers,  # tqdm
                             ),
                axis=0,
            )
            print("[bold]加载完成！[/bold]")
        # result = pd.concat(process_map(load_basins_timeseries_with_threads,
        #                                gauge_id_lists, huc_02_lists,
        #                                [root_path] * num_workers,
        #                                [source] * num_workers,
        #                                [ignore_columns] * num_workers,
        #                                [False] * num_workers,  # tqdm
        #                                desc=f"正在使用{num_workers}个进程加载{len(gauge_id_list)}个流域的气象强迫和流量数据",
        #                                total=num_workers,
        #                                ),
        #                    axis=0,
        #                    )
    else:
        result = load_basins_timeseries_with_threads(gauge_id_list, huc_02_list, root_path, source, ignore_columns)
    if to_xarray:
        result = result.to_xarray()
    return result


def load_timeseries(gauge_id: Optional[Union[str, List[str]]] = None,
                    root_path: str = params.camels_root_path,
                    source: str = params.forcing_source,
                    ignore_columns: Optional[List[str]] = None,
                    multi_process: bool = False,
                    to_xarray: bool = False,
                    ) -> Union[DataFrame, Dataset]:
    """
    加载CAMELS数据集中所有流域的气象强迫和流量数据

    :param gauge_id: 流域ID，可以是str(表示单个流域id)，也可以是List[str](表示多个流域id)
    :param root_path: CAMELS数据集根目录，默认为"camels_params"中的root_path
    :param source: 降水数据来源, 默认为"daymet", 可选值为"daymet"、“maurer"和”nldas"
    :param ignore_columns: 忽略的列名列表
    :param multi_process: 是否使用多进程, 仅对多个流域有效
    :param to_xarray: 是否转换为xarray格式, 仅对多个流域有效
    :return: 指定流域的气象强迫和流量数据
    """
    if gauge_id is None:
        return load_basins_timeseries(gauge_id_list=None,
                                      root_path=root_path,
                                      source=source,
                                      ignore_columns=ignore_columns,
                                      multi_process=multi_process,
                                      to_xarray=to_xarray)
    elif isinstance(gauge_id, str):
        return load_single_basin_timeseries(gauge_id,
                                            root_path=root_path,
                                            source=source,
                                            ignore_columns=ignore_columns)
    elif isinstance(gauge_id, list):
        return load_basins_timeseries(gauge_id_list=gauge_id,
                                      root_path=root_path,
                                      source=source,
                                      ignore_columns=ignore_columns,
                                      multi_process=multi_process,
                                      to_xarray=to_xarray)
