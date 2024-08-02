import pandas as pd
import scienceplots
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams, gridspec
from typing import List
from pandas import DataFrame, Series
from typing import Union, Optional
import hydronetwork.plot.plot_params as params
# from subprocess import call

rcParams['font.family'] = params.font
rcParams['savefig.dpi'] = params.dpi
rcParams['figure.dpi'] = 0.4 * params.dpi
rcParams['savefig.format'] = params.pic_format
rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=params.color)
plt.style.use(params.style)

path = "../data/Tide result/val_loss.xlsx"
data = pd.read_excel(path, index_col="Step")
# 改名 without features projection -> No FP, No RevIN
# with RevIN -> FP + RevIN
# without RevIN -> FP Only
# data = data.rename(columns={"without features projection": "No FP, No RevIN",
#                             "with RevIN": "FP + RevIN",
#                             "without RevIN": "FP Only"}
#                    )
# 删去without features projection
data = data.drop(columns="without features projection")
fig = plt.figure(figsize=(params.width["1.5 column"], 2))
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
data.plot(ax=ax)
ax.set_xlabel("Step")
ax.set_ylabel("Loss")
# 设置xlim为0-1800000
ax.set_xlim(0, 1800000)
# 设置体图片宽度
# 设置图例无框
ax.legend(frameon=False)
# 保存图片
plt.savefig("val_loss.JPEG")
plt.show()