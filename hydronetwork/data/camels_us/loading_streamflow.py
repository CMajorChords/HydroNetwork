from typing import List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm.contrib.concurrent import thread_map
from rich import print
from xarray import Dataset
from pandas import DataFrame
import pandas as pd
import hydronetwork.data.camels_us.camels_us_params as params
from hydronetwork.data.camels_us.loading_attributes import load_single_type_attributes
from hydronetwork.data.camels_us.utils import get_gauge_id, num_workers, split_list
from hydronetwork.data.camels_us_hourly.camels_us_hourly_params import get_basin_area


def load_single_basin_streamflow(gauge_id: str,
                                 huc_02: Optional[str] = None,
                                 root_path: str = params.camels_root_path,
                                 add_datetime: bool = True,
                                 unit="m^3/s",
                                 ) -> DataFrame:
    """
    加载单个流域的流量数据

    :param gauge_id: 流域ID
    :param huc_02: 流域的HUC02编码，如果没有指定，则从流域名称数据中获取
    :param root_path: CAMELS数据集根目录，默认为"camels_params"中的root_path
    :param add_datetime: 是否添加datetime列，若添加，将删去Year，Mnth和Day三列，并添加datetime列为索引
    :param unit: 流量单位，默认为"m^3/s"，可以为"mm/day"
    :return: 单个流域的流量数据
    """
    # 如果没有指定HUC02编码，则从流域名称数据中获取
    if huc_02 is None:
        huc_02 = load_single_type_attributes("name", root_path).loc[gauge_id, "huc_02"]
    # 读取数据
    path = root_path + (f"/basin_timeseries_v1p2_metForcing_obsFlow/"
                        f"basin_dataset_public_v1p2/usgs_streamflow/{huc_02}/{gauge_id}_streamflow_qc.txt")
    data = pd.read_csv(path,
                       header=None,
                       names=["gauge_id", "Year", "Mnth", "Day", "streamflow", "qc"],
                       sep=r"\s+",
                       )
    # 数据处理
    data["streamflow"] = data["streamflow"].where(data["qc"] != "M")  # 如果qc是M，则将流量数据设为NaN
    # 单位换算，从cfs转换为m^3/s，
    data["streamflow"] = data["streamflow"] * 0.028316846592
    # 如果unit为mm/h，则将流量数据转换为mm/h
    if unit not in ["m^3/s", "mm/day"]:
        raise ValueError("unit必须为m^3/s或mm/day，当前unit为{unit}")
    if unit == "mm/day":
        data["streamflow"] = data["streamflow"] * 3.6 * 24 / get_basin_area(gauge_id)
    data.drop(columns="qc", inplace=True)
    if add_datetime:  # 根据年月日形成datetime列
        data["datetime"] = pd.to_datetime(data[["Year", "Mnth", "Day"]].astype(str).agg("-".join, axis=1))
        data.set_index("datetime", inplace=True)
        data.drop(columns=["Year", "Mnth", "Day"], inplace=True)
    data.drop(columns="gauge_id", inplace=True)  # 删除gauge_id列
    data.columns.name = "streamflow"
    return data


def load_basins_streamflow_with_threads(gauge_id_list: List[str],
                                        huc_02_list: List[str],
                                        root_path: str,
                                        unit: str = "m^3/s",
                                        tqdm: bool = True,
                                        ) -> DataFrame:
    """
    多线程加载CAMELS数据集中部分流域的流量数据

    :param gauge_id_list: 流域ID列表
    :param huc_02_list: 流域的HUC02编码列表
    :param root_path: CAMELS数据集根目录，默认为"camels_params"中的root_path
    :param unit: 流量单位，默认为"m^3/s"，可以为"mm/day"
    :param tqdm: 是否显示进度条
    :return: 所有流域的流量数据
    """
    len_gauge_id_list = len(gauge_id_list)
    if tqdm:
        return pd.concat(
            thread_map(load_single_basin_streamflow,
                       gauge_id_list, huc_02_list,
                       [root_path] * len_gauge_id_list,
                       [True] * len_gauge_id_list,  # 添加datetime列
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
                             gauge_id_list, huc_02_list,
                             [root_path] * len_gauge_id_list,
                             [True] * len_gauge_id_list,  # 添加datetime列
                             [unit] * len_gauge_id_list,
                             ),
                keys=gauge_id_list,
                names=["gauge_id", "datetime"],
                axis=0,
            )


def load_basins_streamflow(gauge_id_list: Optional[List[str]] = None,
                           root_path: str = params.camels_root_path,
                           unit: str = "m^3/s",
                           multi_process: bool = False,
                           to_xarray: bool = False,
                           ) -> Union[DataFrame, Dataset]:
    """
    加载CAMELS数据集中所有流域的流量数据

    :param gauge_id_list: 流域ID列表，如果为None，则加载所有流域的ID
    :param root_path: CAMELS数据集根目录，默认为"camels_params"中的root_path
    :param unit: 流量单位，默认为"m^3/s"，可以为"mm/day"
    :param multi_process: 是否使用多进程
    :param to_xarray: 是否转换为xarray格式
    :return: 所有流域的流量数据
    """
    # 如果没有指定流域ID列表，则加载所有流域的ID
    if gauge_id_list is None:
        gauge_id_list, huc_02_list = get_gauge_id(root_path, True)
    else:
        huc_02_list = load_single_type_attributes("name", root_path).loc[gauge_id_list, "huc_02"].tolist()
    # 判断使用多线程还是多进程, 并加载所有流域的流量数据
    if multi_process:
        # 先将流域ID列表和huc_02列表分成num_workers份
        gauge_id_lists = split_list(gauge_id_list, num_workers)
        huc_02_lists = split_list(huc_02_list, num_workers)
        # 使用多进程加载所有流域的流量数据
        with ProcessPoolExecutor() as executor:
            print(f"[bold]正在使用{num_workers}个进程加载{len(gauge_id_list)}个流域的流量数据[/bold]")
            result = pd.concat(
                executor.map(load_basins_streamflow_with_threads,
                             gauge_id_lists, huc_02_lists,
                             [root_path] * num_workers,  # 数据集根目录
                             [unit] * num_workers,  # 流量单位
                             [False] * num_workers,  # 不显示进度条
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
        result = load_basins_streamflow_with_threads(gauge_id_list, huc_02_list, root_path, unit)
    if to_xarray:
        result = result.to_xarray()
    return result


def load_streamflow(gauge_id: Optional[Union[str, List[str]]] = None,
                    root_path: str = params.camels_root_path,
                    add_datetime: bool = True,
                    unit: str = "m^3/s",
                    multi_process: bool = False,
                    to_xarray: bool = False,
                    ) -> Union[DataFrame, Dataset]:
    """
    加载CAMELS数据集中所有流域的流量数据

    :param gauge_id:流域ID，可以是str(表示单个流域id)，也可以是List[str](表示多个流域id)
    :param root_path:CAMELS数据集根目录，默认为"camels_params"中的root_path
    :param add_datetime:是否添加datetime列，若添加，将删去Year，Mnth和Day三列，并添加datetime列为索引，仅对单个流域有效
    :param unit:流量单位，默认为"m^3/s"，可以为"mm/day"
    :param multi_process:是否使用多进程，仅对多个流域有效
    :param to_xarray:是否转换为xarray格式，仅对多个流域有效
    :return:指定流域的流量数据
    """
    if gauge_id is None:
        return load_basins_streamflow(gauge_id_list=None,
                                      root_path=root_path,
                                      multi_process=multi_process,
                                      unit=unit,
                                      to_xarray=to_xarray)
    elif isinstance(gauge_id, str):
        return load_single_basin_streamflow(gauge_id,
                                            root_path=root_path,
                                            unit=unit,
                                            add_datetime=add_datetime)
    elif isinstance(gauge_id, list):
        return load_basins_streamflow(gauge_id_list=gauge_id,
                                      root_path=root_path,
                                      multi_process=multi_process,
                                      unit=unit,
                                      to_xarray=to_xarray)
