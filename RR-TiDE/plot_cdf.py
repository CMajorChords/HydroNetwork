import pandas as pd

# 提取数据
metric = "NSE"
path = ["data/Tide result/无特征投影层预报结果",
        "data/Tide result/有特征投影层预报结果/best",
        "data/Tide result/有特征投影层预报结果/rin",
        ]
# path_single_model = path[0]
# dataframe_list = []
# use_id = pd.read_csv("use_id.csv").values
# for n in range(1, 8):
#     csv_path = path_single_model + f"/metrics_daymet_pred_steps={n}.csv"
#     dataframe = pd.read_csv(csv_path, index_col="gauge_id")
#     dataframe = dataframe.loc[use_id.flatten()]
#     dataframe_list.append(dataframe[metric])
# columns = ["1st-day", "2nd-day", "3rd-day", "4th-day", "5th-day", "6th-day", "7th-day"]
# dataframe = pd.concat(dataframe_list, axis=1)
# dataframe.columns = columns
model_dataframe = []
for path_single_model in path:
    dataframe_list = []
    use_id = pd.read_csv("use_id.csv").values
    for n in range(1, 8):
        csv_path = path_single_model + f"/metrics_daymet_pred_steps={n}.csv"
        dataframe = pd.read_csv(csv_path, index_col="gauge_id")
        dataframe = dataframe.loc[use_id.flatten()]
        dataframe_list.append(dataframe[metric])
    columns = ["1st-day", "2nd-day", "3rd-day", "4th-day", "5th-day", "6th-day", "7th-day"]
    dataframe = pd.concat(dataframe_list, axis=1)
    dataframe.columns = columns
    model_dataframe.append(dataframe)
on = ["No FP, No RevIN", "FP + RevIN", "FP Only"]
model_dataframe = pd.concat(model_dataframe, keys=on, axis=1)
day_dataframe = []
for i, column in zip(range(1, 8), columns):
        day_dataframe.append(model_dataframe.xs(column, axis=1, level=1))
# %%绘图
from drafts.matplotlib import plot_multiple_cdfs
plot_multiple_cdfs(day_dataframe,
                   width="1.5 column",
                   height=1.5,
                   save_path="cdf",
                   titles=columns)