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
## 参数:随机数函数,最小值,最大值,样本量
def cuda_simple(random_func, bottom, top, sample_num):
    pass


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
def cuda_important(random_func, bottom, top, sample_num):
    pass


# 分层采样
## 参数:随机数函数,最小值,最大值,样本量,分层层数
@jit(nopython=True, nogil=True, parallel=True)
def layer(random_func, bottom, top, sample_num, layers):
    dist = 0.0
    for i in np.arange(layers):
        # 单层随机数
        lay_sample = np.random.uniform(
            bottom + i * (top - bottom) / layers,
            bottom + (i + 1) * (top - bottom) / layers,
            sample_num // layers,
        )
        # 单层积分
        lay_dist = np.mean(random_func(lay_sample))
        # 加权平均区间积分
        dist += lay_dist * (top - bottom) / layers
    return dist


# 分层采样 有CUDA
## 参数:随机数函数,最小值,最大值,样本量,分层层数
def cuda_layer(random_func, bottom, top, sample_num, layers):
    pass


# 总执行次数
total_run = 1
# 样本个数
sample_num = 10**9
# 分层层数
layers = 10**4


print("总执行次数", total_run, ",样本个数", sample_num, ",分层层数:", layers)

for i in range(total_run):
    print("======== CPU第", i, "次执行 ========")
    print("______________第一阶段______________")
    # print("一般方法Numba开启与否时的运行情况")
    # cpu_time, result = calculate_cpu_time(dis_simple, dis_func, bottom, top, sample_num)
    # print("无Numba:", cpu_time, "值:", result)
    # time.sleep(3)

    cpu_time, result = calculate_cpu_time(simple, enab_func, bottom, top, sample_num)
    print("有Numba:", cpu_time, "值:", result)
    time.sleep(3)

    print("______________第二阶段______________")
    print("多个算法间的速度 结果 x y拟合")
    cpu_time, result = calculate_cpu_time(simple, enab_func, bottom, top, sample_num)
    print("一般实现时间:", cpu_time, "值:", result)
    time.sleep(3)

    cpu_time, result = calculate_cpu_time(important, enab_func, bottom, top, sample_num)
    print("重要性采样时间:", cpu_time, "值:", result)
    time.sleep(3)

    cpu_time, result = calculate_cpu_time(
        layer, enab_func, bottom, top, sample_num, layers
    )
    print("分层抽样时间:", cpu_time, "值:", result)
    time.sleep(3)

    # cpu_time, result = calculate_cpu_time(
    #    cuda_simple, enab_func, bottom, top, sample_num
    # )
    # print("CUDA 一般实现时间:", cpu_time, "值:", result)
    # time.sleep(3)

    # cpu_time, result = calculate_cpu_time(
    #    cuda_important, enab_func, bottom, top, sample_num
    # )
    # print("CUDA 重要性采样时间:", cpu_time, "值:", result)
    # time.sleep(3)

    # cpu_time, result = calculate_cpu_time(
    #    cuda_layer, enab_func, bottom, top, sample_num, layers
    # )
    # print("CUDA 分层抽样时间:", cpu_time, "值:", result)
    # time.sleep(3)

    print("______________第三阶段______________")
    print("单个方法在不同样本量下的结果区别")
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

        # cpu_time, result = calculate_cpu_time(
        #    cuda_simple, enab_func, bottom, top, for_num
        # )
        # print("CUDA 一般实现时间:", cpu_time, "值:", result)
        # time.sleep(3)

        # cpu_time, result = calculate_cpu_time(
        #    cuda_important, enab_func, bottom, top, for_num
        # )
        # print("CUDA 重要性采样时间:", cpu_time, "值:", result)
        # time.sleep(3)

        # cpu_time, result = calculate_cpu_time(
        #    cuda_layer, enab_func, bottom, top, for_num, for_layer
        # )
        # print("CUDA 分层抽样时间:", cpu_time, "层数:", for_layers, "值:", result)
        # time.sleep(3)
