from timeit import timeit

import numpy as np
from numba import jit,cuda

# 层数(分层采样)/代数(遗传算法)
layers = 10**4
# 变异率(遗传算法)
mutation = 0.2

# 总执行次数
for_time = 1
# 随机数生成次数
times = 10**9

# 下标(x最小值)
bottom = 0
# 上标(x最大值)
top = 2 * np.pi


# 函数式
## y(x) = 2sin(x) (x^3+ x^2+ 2x+ 3)
@jit(nopython=True, parallel=True)
def func(x):
    return np.multiply(
        np.multiply(np.sin(x), 2),
        np.add(np.add(np.power(x, 3), np.power(x, 2)), np.add(np.multiply(x, 2), 3)),
    )


# 常规实现
@jit(nopython=True, parallel=True)
def simple():
    # 在均匀分布中生成x
    x = np.random.uniform(bottom, top, times)
    # 计算y
    y = func(x)
    # y的平均数*(top-bottom) 积分中值
    dist = np.multiply(np.mean(y), np.subtract(top, bottom))

    print(dist)
    return dist


# 重要性采样
@jit(nopython=True, parallel=True)
def important():
    # 在均匀分布中生成x
    y = np.random.uniform(bottom, top, times)
    # 计算函数值与概率分布值的比例
    ## 概率分布函数: 1/2pi,即均匀分布
    percent = func(y) / (1 / (2 * np.pi))
    # 积分值
    dist = np.mean(percent)

    print(dist)
    return dist


# 分层采样
@jit(nopython=True, parallel=True)
def layer():
    dist = 0.0
    for i in np.arange(layers):
        # 单层随机数
        lay_sample = np.random.uniform(
            bottom + i * (top - bottom) / layers,
            bottom + (i + 1) * (top - bottom) / layers,
            times // layers,
        )
        # 单层积分
        lay_dist = np.mean(func(lay_sample))
        # 加权平均区间积分
        dist += lay_dist * (top - bottom) / layers

    print(dist)
    return dist


# 遗传算法
@jit(nopython=True, parallel=True)
def gene():
    pass


# Numba(CUDA)
@cuda.jit(device=True)
def numba_cuda():
    pass

# CuPy(CUDA)
def cupy_cuda():
    pass


print(
    "层数(分层采样)/代数(遗传算法):",
    layers,
    "\n变异率(遗传算法):",
    mutation,
    "\n总执行次数",
    for_time,
    "\n随机数生成次数",
    times,
)
print("===========CPU执行===========")
print("一般实现时间:", timeit("simple()", globals=globals(), number=for_time))
print("重要性采样时间:", timeit("important()", globals=globals(), number=for_time))
print("分层采样时间:", timeit("layer()", globals=globals(), number=for_time))
# print("遗传算法时间:", timeit("jit()", globals=globals(), number=for_time))
