import numpy as np
import matplotlib.pyplot as plt

# 定义非对称高斯变体函数
def skewed_gaussian(x, mu, a, sigma, p):
    return a * np.exp(-(np.abs(x - mu)**p) / (2 * sigma**2))

# 定义分段指数函数 (用于之前绘制作比较)
def piecewise_exponential(x, mu, a, k1, k2):
    return np.where(x <= mu, a * np.exp(k1 * (x - mu)), a * np.exp(k2 * (x - mu)))

# 参数设定
mu = 6      # 峰值位置
a = 1        # 峰值高度
sigma = 20    # 高斯分布的标准差
p = 2        # 对称模型控制参数 (>2表示向左偏态)
k1 = 1       # 用于对比的分段函数 - 上升段陡峭度
k2 = -0.2    # 用于对比的分段函数 - 下降段平缓度

# 生成采样点
x = np.linspace(1, 30, 300)

# 生成非对称高斯函数和分段指数函数的值
y_skewed_gaussian = skewed_gaussian(x, mu, a, sigma, p)
y_piecewise = piecewise_exponential(x, mu, a, k1, k2)

# 将所有采样点归一化，使得所有采样点和为1
y_skewed_gaussian /= np.sum(y_skewed_gaussian)
y_piecewise /= np.sum(y_piecewise)

# 绘制图像
plt.figure(figsize=(10, 6))

# 绘制非对称高斯曲线
plt.plot(x, y_skewed_gaussian, label='非对称高斯函数 (向左偏态)', color='r', linestyle='--', linewidth=2)

# 绘制分段指数函数作对比
plt.plot(x, y_piecewise, label='分段指数函数', color='b', linewidth=2)

# 添加图例、标题和标签
plt.title("非对称高斯曲线与分段指数函数 (峰值位置 x=15)", fontsize=16)
plt.xlabel("x", fontsize=14)
plt.ylabel("f(x)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)

# 显示图像
plt.show()