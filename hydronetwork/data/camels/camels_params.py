from yaml import safe_load
from csv import reader

path = "configs/camels/CAMELS_US.yml"
with open(path, "r", encoding="utf-8") as f:
    config = safe_load(f)

# CAMELS数据集根目录
camels_root_path = config["camels_root_path"]

# CAMELS数据集中的属性种类名称
attributes_type_name = config["attributes_type_name"]

# CAMELS数据集中要使用的属性名称
use_attribute_name = config["attributes"]

# CAMELS数据集中的降水数据来源
forcing_source = config["forcing_source"]
# CAMELS降水数据中要忽略的列
if forcing_source == "daymet":
    ignore_precip_cols = config["ignore_precip_cols_daymet"]
elif forcing_source == "maurer" or forcing_source.lower == "nldas":
    ignore_precip_cols = config["ignore_precip_cols_maurer_and_nldas"]
else:
    raise ValueError("指定的降水数据来源不正确，只能选择daymet、maurer或nldas")


# CAMELS数据集中要忽略的流域id
def read_gauge_id_from_csv(csv_path):
    with open(csv_path, "r") as csv_file:
        csv_reader = reader(csv_file)
        gauge_id_list = list(csv_reader)
    return gauge_id_list


ignore_gauge_id = read_gauge_id_from_csv(config["ignore_gauge_id_list_path"])

# CAMELS数据集中的各类特征
features_bidirectional = ["prcp(mm/day)",
                          "srad(W/m2)",
                          "tmax(C)",
                          "tmin(C)",
                          "vp(Pa)",
                          ]
features_lookback = ["streamflow",
                     ]
