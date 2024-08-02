from pandas import read_csv
from yaml import safe_load

# 读取配置文件
path = "configs/plot/plot.yml"
with open(path, encoding='utf-8') as file:
    plot_params = safe_load(file)

# 设置风格
style = plot_params["style"]

# 设置字体
font = plot_params["font"]


# 设置默认dpi
dpi = plot_params["dpi"]
# 设置图片格式
pic_format = plot_params["format"]
# 设置图片颜色
color = plot_params["color"]
# 设置图片大小
width = read_csv("configs/plot/width.csv", header=0, index_col=0)
width["Image Width (inch)"] = width["Image Width (mm)"] / 25.4
width = {
    "Minimal width": width.loc["Minimal size", "Image Width (inch)"],
    "Single column": width.loc["Single column", "Image Width (inch)"],
    "1.5 column": width.loc["1.5 column", "Image Width (inch)"],
    "Double column (full width)": width.loc["Double column (full width)", "Image Width (inch)"],
}
