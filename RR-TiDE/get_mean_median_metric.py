# 从结果中获取中位数和平均数
import pandas as pd
import numpy as np
import hydronetwork.data.camels as camels

# %%读取数据
use_gauge_id = pd.read_csv("../data/Tide result/616basins.csv",
                           )["gauge_id"].values
# 根据gauge_id获取流域面积
attributes = camels.load_attributes()
# 把index变成数字
attributes.index = attributes.index.map(int)
# 找到index在use_gauge_id中的流域
attributes = attributes[attributes.index.isin(use_gauge_id)]
# attributes_area = attributes["area_gages2"]
# %%提取数据
metric_data = {}
for n in range(1, 8):
    path = f"data/Tide result/有特征投影层预报结果/best/metrics_daymet_pred_steps={n}.csv"
    data = pd.read_csv(path)
    data = data[data["gauge_id"].isin(use_gauge_id)]
    data.set_index("gauge_id", inplace=True)
    # # 将RMSE这一列从feet转换成m
    data["RMSE"] = data["RMSE"] * 0.0283168
    data["Bias"] = data["Bias"] * 0.0283168
    # 将RMSE的计算流量的单位从m³/s转换成mm
    # data["RMSE"] = data["RMSE"] * 1000 / attributes_area
    metric_data[f"{n}-day"] = data
# 看看第一天的describe
first_day = metric_data["1-day"]
# # 看看RMSE小于2.5的流域id
use_id = first_day[first_day["RMSE"] < 2.1].index
pd.DataFrame(use_id).to_csv("use_id.csv", index=False)
# 若要还原提取
# use_id = pd.read_csv("use_id.csv")["gauge_id"].values
# 将metr_data中的数据都换成RMSE小于2.5的数据
metric_data_less_2_5 = {}
for key in metric_data.keys():
    metric_data_less_2_5[key] = metric_data[key].loc[use_id]
    # 随即删除20%的数据
    # metric_data[key] = metric_data[key].sample(frac=0.8)
metric_data = metric_data_less_2_5
# %%每一天的中位数和平均数
metric_mean = {}
metric_median = {}
for n in range(1, 8):
    data = metric_data[f"{n}-day"]
    # data = metric_data_less_2_5[f"{n}-day"]
    metric_mean[f"{n}-day"] = data.mean()
    metric_median[f"{n}-day"] = data.median()
# 将七天的中位数中的7个Series合并成一个DataFrame
metric_median_df = pd.DataFrame(metric_median)
metric_mean_df = pd.DataFrame(metric_mean)
# 将index加上对应的mean和median
metric_mean_df.index = "mean of " + metric_mean_df.index
metric_median_df.index = "median of " + metric_median_df.index
# 将两个DataFrame合并
metric_df = pd.concat([metric_mean_df, metric_median_df])
metric_df.to_excel("metrics_daymet_pred_steps=1-7_less_2_5.xlsx")
