import pandas as pd
from drafts.matplotlib import plot_metrics

# %%读取数据
path = r"C:\Users\18313\Desktop\新建结果 2024.6.9\结果.xlsx"
metric_data = pd.read_excel(path, sheet_name="不同方法的比较")
# 根据metric这一列排序
ordered_metric = ["Median of NSE",
                  "Mean of NSE",
                  "Median of RMSE",
                  "Mean of RMSE",
                  "Median of Bias",
                  "Mean of Bias",
                  ]
metric_data = metric_data.set_index("Metric")
metric_data = metric_data.loc[ordered_metric]
#解散index
metric_data.reset_index(inplace=True)
# 将第0列和第一列设置为MultiIndex
metric_data.index = pd.MultiIndex.from_frame(metric_data.iloc[:, :2])
# 删除第0列和第一列
metric_data.drop(columns=["Metric", "Model"], inplace=True)
# %绘图
plot_metrics(data=metric_data,
             nline=2,
             width="1.5 column",
             height=1.8,
             save_path="figures",
             )
