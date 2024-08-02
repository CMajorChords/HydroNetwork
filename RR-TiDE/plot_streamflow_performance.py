import pandas as pd
from drafts.matplotlib import plot_streamflow
from hydronetwork.plot.plot_params import color

gauge_id = "01013500"
# gauge_id = "01030500"
# gauge_id_list = []
# for i in range(1, 8):
#     data = pd.read_csv(f"data/Tide result/有特征投影层预报结果/best/results_daymet_pred_steps={i}.csv")
#     gauge_id_list.append(list(data["gauge_id"].unique()))
# # 找到这几个list中的交集
# gauge_id = gauge_id_list[0]
# for i in range(1, 7):
#     gauge_id = list(set(gauge_id).intersection(set(gauge_id_list[i])))
# metric = pd.read_csv("data/Tide result/有特征投影层预报结果/best/metrics_daymet_pred_steps=7.csv")
# metric = metric[metric["gauge_id"].isin(gauge_id)]
# # 找到nse第二高的那个gauge_id并print nse
# max_nse = metric[metric["NSE"] == metric["NSE"].nlargest(2).iloc[1]]
# print(max_nse)
# 获取数据
data = []
streamflow = load_streamflow(gauge_id)
for n in range(1, 8):
    path = f"data/Tide result/有特征投影层预报结果/best/results_daymet_pred_steps={n}.csv"
    dataframe = pd.read_csv(path)
    dataframe = dataframe[dataframe["gauge_id"] == int(gauge_id)]
    length = len(dataframe)
    dataframe.index = streamflow.index[-length:]
    dataframe.drop(columns="gauge_id", inplace=True)
    # 改名， true_target改为observed streamflow,predict_target改为predicted streamflow-n
    dataframe.rename(columns={"true_target": "observed streamflow",
                              f"predict_target": f"predicted streamflow-{n}"
                              },
                     inplace=True)
    data.append(dataframe)
data = pd.concat(data, axis=1)
# 删除多余的observed streamflow，只保留一个
data_without_observed_streamflow = data.drop(columns="observed streamflow")
data = pd.concat([data.iloc[:, 0], data_without_observed_streamflow], axis=1)
# %%绘制
data_1 = data.loc[:, ["observed streamflow",
                      "predicted streamflow-1",
                      "predicted streamflow-2", ]]
data_2 = data.loc[:, ["observed streamflow",
                      "predicted streamflow-3",
                      "predicted streamflow-4", ]]
data_3 = data.loc[:, ["observed streamflow",
                      "predicted streamflow-5",
                      "predicted streamflow-6",
                      "predicted streamflow-7", ]]
data_1 = data_1.loc[: "2014-10-1"]
data_2 = data_2.loc[: "2014-10-1"]
data_3 = data_3.loc[: "2014-10-1"]
# 注意除了observed streamflow之外，其他的流量过程颜色不能相同，所有的颜色全部采用深色，高级的颜色
# plot_streamflow(data=data_1,
#                 width="Double column (full width)",
#                 height=3,
#                 y_lim=[0, 7400],
#                 save_path="streamflow_1",
#                 color_list=[color[0], color[1], color[2]],
#                 add_symbol="(1)"
#                 )
# plot_streamflow(data=data_2,
#                 width="Double column (full width)",
#                 height=3,
#                 y_lim=[0, 7400],
#                 save_path="streamflow_2",
#                 color_list=[color[0], color[3], color[4]],
#                 add_symbol="(2)"
#                 )
# plot_streamflow(data=data_3,
#                 width="Double column (full width)",
#                 height=3,
#                 y_lim=[0, 7400],
#                 save_path="streamflow_3",
#                 color_list=[color[0], color[5], color[6], color[7]],
#                 add_symbol="(3)"
#                 )
data_list = [data_1, data_2, data_3]
color_lists = [
    [color[0], color[5], color[6], color[1]],
    [color[0], color[3], color[4], color[1]],
    [color[0], color[8], color[2], color[1]],
]

plot_streamflow(data_list=data_list, y_lim=[0, 20000], save_path="combined_streamflow", color_lists=color_lists,
                width="1.5 column", height=1.8)
