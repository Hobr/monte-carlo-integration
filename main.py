import time

import matplotlib.pyplot as plt
import numpy as np
from numba import jit

# 下标(x最小值)
bottom = 0
# 上标(x最大值)
top = 2 * np.pi
# 随机数(x)生成次数
times = 10**9
# 分层数(分层采样)
layers = 100
# 总执行次数
for_time = 3


# 函数式
## y(x) = 2sin(x) (x^3+ x^2+ 2x+ 3)
@jit(nopython=True, parallel=True)
def func(x):
    return np.multiply(
        np.multiply(np.sin(x), 2),
        np.add(np.add(np.power(x, 3), np.power(x, 2)), np.add(np.multiply(x, 2), 3)),
    )


# 常规实现+多线程
@jit(nopython=True, parallel=True)
def simple():
    # 均匀分布中生成x
    x = np.random.uniform(bottom, top, times)
    # 计算y
    y = func(x)
    # y的平均数*(top-bottom)
    dist = np.multiply(np.mean(y), np.subtract(top, bottom))
    return dist


# 重要性采样

# 分层采样

# 遗传算法

# AVX+SIMD

# Cython

# CUDA

for i in range(for_time):
    print(simple())
