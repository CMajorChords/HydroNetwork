import pandas as pd
from drafts.matplotlib import plot_streamflow

gauge_id = "01013500"
streamflow = load_streamflow(gauge_id)
n = 1
path_no_fp = f"data/Tide result/无特征投影层预报结果/results_daymet_pred_steps={4}.csv"
path_fp_only = f"data/Tide result/有特征投影层预报结果/rin/results_daymet_pred_steps={2}.csv"
path_fp_rin = f"data/Tide result/有特征投影层预报结果/best/results_daymet_pred_steps={n}.csv"
data_no_fp = pd.read_csv(path_no_fp, index_col="gauge_id")
data_fp_only = pd.read_csv(path_fp_only, index_col="gauge_id")
data_fp_rin = pd.read_csv(path_fp_rin, index_col="gauge_id")
data_no_fp = data_no_fp.loc[int(gauge_id)]
data_fp_only = data_fp_only.loc[int(gauge_id)]
data_fp_rin = data_fp_rin.loc[int(gauge_id)]
data_no_fp.index = streamflow.index[-len(data_no_fp):]
data_fp_only.index = streamflow.index[-len(data_fp_only):]
data_fp_rin.index = streamflow.index[-len(data_fp_rin):]
data_no_fp.rename(columns={"true_target": "observed streamflow",
                           "predict_target": "No FP, No RevIN",
                           },
                  inplace=True)
data_fp_only.rename(columns={"true_target": "observed streamflow",
                             "predict_target": "FP Only",
                             },
                    inplace=True)
data_fp_rin.rename(columns={"true_target": "observed streamflow",
                            "predict_target": "FP + RevIN",
                            },
                   inplace=True)
# 合并为一个DataFrame
data = pd.concat([data_no_fp["observed streamflow"],
                  data_no_fp["No FP, No RevIN"],
                  data_fp_only["FP Only"],
                  data_fp_rin["FP + RevIN"]], axis=1)
# 设置data的数值为float
data = data.astype(float)
data_slice = data
data_slice = data.loc["2012-9-01":"2014-4-1"]
# 绘制
plot_streamflow(data_list=[data_slice],
                # y_lim=[0, 7400],
                save_path="streamflow_projection_revin",
                # color_lists=[color["blue"], color["green"], color["red"]],
                width="1.5 column",
                height=2,
                share_x_label=False,
                legend_loc="upper right",
                )
