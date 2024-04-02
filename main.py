import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from scipy import integrate


# 计算CPU时间
## 参数: 被测速函数, **函数参数
def calculate_cpu_time(func, *args, **kwargs):
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    cpu_time = end_time - start_time
    return cpu_time, result


# 普通函数式 无Numba
## y = 2 * sin(x) * (x**3 + x**2 + 2 * x + 3)
def dis_func(x):
    return 2 * np.sin(x) * (x**3 + x**2 + 2 * x + 3)


# 普通函数式 有Numba
## y = 2 * sin(x) * (x**3 + x**2 + 2 * x + 3)
@jit(nopython=True, nogil=True, parallel=True)
def enab_func(x):
    return 2 * np.sin(x) * (x**3 + x**2 + 2 * x + 3)


# 常规实现 无Numba
## 参数:随机数函数,最小值,最大值,样本量
def dis_simple(random_func, bottom, top, sample_num):
    # 在均匀分布中生成x
    random_x = np.random.uniform(bottom, top, sample_num)
    # 计算y/积分和
    integ_sum = random_func(random_x)
    # 积分中值
    dist = np.mean(integ_sum) * (top - bottom)
    return dist


# 常规实现 有Numba
## 参数:随机数函数,最小值,最大值,样本量
@jit(nopython=True, nogil=True, parallel=True)
def simple(random_func, bottom, top, sample_num):
    # 在均匀分布中生成x
    random_x = np.random.uniform(bottom, top, sample_num)
    # 计算y/积分和
    integ_sum = random_func(random_x)
    # 积分中值
    dist = np.mean(integ_sum) * (top - bottom)
    return dist


# 常规实现 有CUDA
## 参数:随机数函数,最小值,最大值,样本量,计算网格
def cuda_simple(bottom, top, sample_num):
    def func(x):
        return 2 * cp.sin(x) * (x**3 + x**2 + 2 * x + 3)

    # 在均匀分布中生成x
    random_x = cp.random.uniform(bottom, top, sample_num)
    # 计算y/积分和
    integ_sum = func(random_x)
    # 积分中值
    dist = cp.mean(integ_sum) * (top - bottom)
    return dist


# 重要性采样
## 参数:随机数函数,最小值,最大值,样本量
@jit(nopython=True, nogil=True, parallel=True)
def important(random_func, bottom, top, sample_num):
    # 在均匀分布中生成x
    y = np.random.uniform(bottom, top, sample_num)
    # 计算函数值与概率分布值的比例
    ## 概率分布函数: 1/2pi,即均匀分布
    percent = random_func(y) / (1 / (2 * np.pi))
    # 积分值
    dist = np.mean(percent)
    return dist


# 重要性采样 有CUDA
## 参数:随机数函数,最小值,最大值,样本量
def cuda_important(bottom, top, sample_num):
    def func(x):
        return 2 * cp.sin(x) * (x**3 + x**2 + 2 * x + 3)

    # 在均匀分布中生成x
    y = cp.random.uniform(bottom, top, sample_num)
    # 计算函数值与概率分布值的比例
    ## 概率分布函数: 1/2pi,即均匀分布
    percent = func(y) / (1 / (2 * cp.pi))
    # 积分值
    dist = cp.mean(percent)
    return dist


# 分层采样
## 参数:随机数函数,最小值,最大值,样本量,分层层数
@jit(nopython=True, nogil=True, parallel=True)
def layer(random_func, bottom, top, sample_num, layers):
    dist = 0.0
    # 减少重复计算
    width = (top - bottom) / layers
    for i in np.arange(layers):
        # 单层随机数
        lay_sample = np.random.uniform(
            bottom + i * width,
            bottom + (i + 1) * width,
            sample_num // layers,
        )
        # 单层积分
        lay_dist = np.mean(random_func(lay_sample))
        # 加权平均区间积分
        dist += lay_dist * width
    return dist


# 分层采样 有CUDA 非向量化
## 参数:随机数函数,最小值,最大值,样本量,分层层数
def cuda_for_layer(bottom, top, sample_num, layers):
    def func(x):
        return 2 * cp.sin(x) * (x**3 + x**2 + 2 * x + 3)

    dist = 0.0
    # 减少重复计算
    width = (top - bottom) / layers
    for i in cp.arange(layers):
        # 单层随机数
        lay_sample = cp.random.uniform(
            bottom + i * width,
            bottom + (i + 1) * width,
            sample_num // layers,
        )
        # 单层积分
        lay_dist = cp.mean(func(lay_sample))
        # 加权平均区间积分
        dist += lay_dist * width
    return dist


# 分层采样 有CUDA
## 参数:随机数函数,最小值,最大值,样本量,分层层数
def cuda_layer(bottom, top, sample_num, layers):
    def func(x):
        return 2 * cp.sin(x) * (x**3 + x**2 + 2 * x + 3)

    # 向量化
    # 创建存储每层随机数的数组
    lay_samples = cp.random.uniform(
        bottom + cp.arange(layers) * (top - bottom) / layers,
        bottom + (cp.arange(layers) + 1) * (top - bottom) / layers,
        (sample_num // layers, layers),
    )
    # 单层积分
    lay_dists = cp.mean(func(lay_samples), axis=0)
    # 加权平均区间积分
    dist = cp.sum(lay_dists * (top - bottom) / layers)
    return dist


# 总执行次数
total_run = 3
# 样本个数
sample_num = 10**8
# 分层层数
layers = 10**4

# 下标(x最小值)
bottom = 0
# 上标(x最大值)
top = 2 * np.pi
# 真实积分估值
about = integrate.quad(dis_func, bottom, top)

print("总执行次数", total_run, ",样本个数", sample_num, ",分层层数:", layers, ",正确值:", about)

for i in range(total_run):
    print("======== 第", i + 1, "次执行 ========")
    print("______________________________")
    print("一般方法Numba开启与否时的运行情况")
    print("______________________________")
    cpu_time, result = calculate_cpu_time(dis_simple, dis_func, bottom, top, sample_num)
    print("无Numba:", cpu_time, "值:", result)
    time.sleep(3)

    cpu_time, result = calculate_cpu_time(simple, enab_func, bottom, top, sample_num)
    print("有Numba:", cpu_time, "值:", result)
    time.sleep(3)

for i in range(total_run):
    print("======== 第", i + 1, "次执行 ========")
    print("______________________________")
    print("单个方法在不同样本量下的结果区别")
    print("______________________________")
    for i in range(4, 9):
        for_num = 10 ** (i + 1)
        for_layers = 10 ** (i // 2)
        print("样本个数", for_num)
        cpu_time, result = calculate_cpu_time(simple, enab_func, bottom, top, for_num)
        print("一般实现时间:", cpu_time, "值:", result)
        time.sleep(3)

        cpu_time, result = calculate_cpu_time(
            important, enab_func, bottom, top, for_num
        )
        print("重要性采样时间:", cpu_time, "值:", result)
        time.sleep(3)

        cpu_time, result = calculate_cpu_time(
            layer, enab_func, bottom, top, for_num, for_layers
        )
        print("分层抽样时间:", cpu_time, "层数:", for_layers, "值:", result)
        time.sleep(3)

        cpu_time, result = calculate_cpu_time(cuda_simple, bottom, 2 * cp.pi, for_num)
        print("CUDA 一般实现时间:", cpu_time, "值:", result)
        time.sleep(3)

        cpu_time, result = calculate_cpu_time(
            cuda_important, bottom, 2 * cp.pi, for_num
        )
        print("CUDA 重要性采样时间:", cpu_time, "值:", result)
        time.sleep(3)

        cpu_time, result = calculate_cpu_time(
            cuda_for_layer, bottom, 2 * cp.pi, for_num, layers
        )
        print("CUDA 分层 非向量化抽样时间:", cpu_time, "层数:", for_layers, "值:", result)
        time.sleep(3)

        cpu_time, result = calculate_cpu_time(
            cuda_layer, bottom, 2 * cp.pi, for_num, layers
        )
        print("CUDA 分层抽样时间:", cpu_time, "层数:", for_layers, "值:", result)
        time.sleep(3)
