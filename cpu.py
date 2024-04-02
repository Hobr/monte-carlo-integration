import time

import matplotlib.pyplot as plt
import numpy as np
from numba import jit

# 下标(x最小值)
bottom = 0
# 上标(x最大值)
top = 2 * np.pi


# 计算CPU时间
## 参数: 被测速函数, **函数参数
def calculate_cpu_time(func, *args, **kwargs):
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    cpu_time = end_time - start_time
    return cpu_time, result


# 函数式
## y = 2 * sin(x) * (x**3 + x**2 + 2 * x + 3)
@jit(nopython=True, nogil=True, parallel=True)
def vec_func(x):
    return np.multiply(
        np.multiply(np.sin(x), 2),
        np.add(np.add(np.power(x, 3), np.power(x, 2)), np.add(np.multiply(x, 2), 3)),
    )


# 常规实现
## 参数:随机数函数,最小值,最大值,样本量
@jit(nopython=True, nogil=True, parallel=True)
def simple(random_func, bottom, top, sample_num):
    # 在均匀分布中生成x
    random_x = np.random.uniform(bottom, top, sample_num)
    # 计算y/积分和
    integ_sum = random_func(random_x)
    # y的平均数*(top-bottom) 积分中值
    dist = np.multiply(np.mean(integ_sum), np.subtract(top, bottom))
    return dist


# 重要性采样
## 参数:随机数函数,最小值,最大值,样本量
@jit(nopython=True, nogil=True, parallel=True)
def important(random_func, bottom, top, sample_num):
    # 在均匀分布中生成x
    y = np.random.uniform(bottom, top, sample_num)
    # 计算函数值与概率分布值的比例
    ## 概率分布函数: 1/2pi,即均匀分布
    ## random_func(y) / (1 / (2 * np.pi))
    percent = np.divide(random_func(y), np.divide(1, np.multiply(2, np.pi)))
    # 积分值
    dist = np.mean(percent)
    return dist


# 分层采样
## 参数:随机数函数,最小值,最大值,样本量,分层层数
@jit(nopython=True, nogil=True, parallel=True)
def layer(random_func, bottom, top, sample_num, layers):
    dist = 0.0
    for i in np.arange(layers):
        # 单层随机数
        lay_sample = np.random.uniform(
            # bottom + i * (top - bottom) / layers
            np.add(bottom, np.multiply(i, np.divide(np.subtract(top, bottom), layers))),
            # bottom + (i + 1) * (top - bottom) / layers,
            np.add(
                bottom,
                np.multiply(np.add(i, 1), np.divide(np.subtract(top, bottom), layers)),
            ),
            # sample_num // layers,
            np.floor_divide(sample_num, layers),
        )
        # 单层积分
        lay_dist = np.mean(random_func(lay_sample))
        # 加权平均区间积分
        # dist += lay_dist * (top - bottom) / layers
        dist = np.add(
            dist, np.multiply(lay_dist, np.divide(np.subtract(top, bottom), layers))
        )
    return dist


# 总执行次数
total_run = 1
# 样本个数
sample_num = 10**9 + (7 * 10**8)
# 分层层数
layers = 10**4


print("总执行次数", total_run, ",样本个数", sample_num, ",分层层数:", layers)

for i in range(total_run):
    print("======== CPU第", i, "次执行 ========")
    cpu_time, result = calculate_cpu_time(simple, vec_func, bottom, top, sample_num)
    print("一般实现时间:", cpu_time, "值:", result)
    time.sleep(3)
    cpu_time, result = calculate_cpu_time(important, vec_func, bottom, top, sample_num)
    print("重要性采样时间:", cpu_time, "值:", result)
    time.sleep(3)
    cpu_time, result = calculate_cpu_time(
        layer, vec_func, bottom, top, sample_num, layers
    )
    print("分层采样时间:", cpu_time, "值:", result)
    time.sleep(3)
