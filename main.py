import time
from datetime import datetime

import cupy as cp
import matplotlib
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
## 参数:原函数,最小值,最大值,样本量
def dis_simple(funct, bottom, top, sample_num):
    # 在均匀分布中生成x
    random_x = np.random.uniform(bottom, top, sample_num)
    # 计算y/积分和
    integ_sum = funct(random_x)
    # 积分中值
    dist = np.mean(integ_sum) * (top - bottom)
    return dist


# 常规实现 有Numba
## 参数:原函数,最小值,最大值,样本量
@jit(nopython=True, nogil=True, parallel=True)
def simple(funct, bottom, top, sample_num):
    # 在均匀分布中生成x
    random_x = np.random.uniform(bottom, top, sample_num)
    # 计算y/积分和
    integ_sum = funct(random_x)
    # 积分中值
    dist = np.mean(integ_sum) * (top - bottom)
    return dist


# 常规实现 有CUDA
## 参数:原函数,最小值,最大值,样本量,计算网格
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
## 参数:原函数,最小值,最大值,样本量
@jit(nopython=True, nogil=True, parallel=True)
def important(funct, bottom, top, sample_num):
    # 在均匀分布中生成x
    y = np.random.uniform(bottom, top, sample_num)
    # 计算函数值与概率分布值的比例
    ## 概率分布函数: 1/2pi,即均匀分布
    percent = funct(y) / (1 / (2 * np.pi))
    # 积分值
    dist = np.mean(percent)
    return dist


# 重要性采样 有CUDA
## 参数:原函数,最小值,最大值,样本量
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
## 参数:原函数,最小值,最大值,样本量,分层层数
@jit(nopython=True, nogil=True, parallel=True)
def layer(funct, bottom, top, sample_num, layers):
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
        lay_dist = np.mean(funct(lay_sample))
        # 加权平均区间积分
        dist += lay_dist * width
    return dist


# 分层采样 有CUDA 非向量化
## 参数:原函数,最小值,最大值,样本量,分层层数
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
## 参数:原函数,最小值,最大值,样本量,分层层数
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


# 一般方法Numba开启与否时的运行情况
def diagram_1(dis, enab):
    print(dis, enab)
    plt.style.use("seaborn-v0_8")

    x = range(len(dis))

    plt.plot(x, dis, label="Disable")
    plt.plot(x, enab, label="Enable")

    plt.title("Numba开启与否对运行时间的影响", fontproperties=font)
    plt.ylabel("时间", fontproperties=font)

    plt.legend()

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"img/plot_{current_time}.png"
    plt.savefig(filename)
    plt.close()


## 同一样本量下不同方法速度对比
def diagram_2(speed, large):
    print(speed)
    plt.style.use("seaborn-v0_8")

    name = [
        "Normal"
        "Important"
        "Layer"
        "CUDA Normal"
        "CUDA Important"
        "CUDA Layer"
        "CUDA Vector Layer"
    ]

    plt.bar(name, speed)

    plt.title("样本量为" + str(large) + "时不同算法及运行方式的速度", fontproperties=font)

    for key, value in speed.items():
        speed[key] = round(value, 5)

    for i, val in enumerate(speed):
        plt.text(i, val, str(val), ha="center", va="bottom")

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"img/plot_{current_time}.png"
    plt.savefig(filename)
    plt.close()


## 单个方法在不同样本量下的结果区别
def diagram_3(input, lab):
    print(input, lab)
    plt.style.use("seaborn-v0_8")

    plt.plot(input, label=lab)

    plt.title("样本量不同时" + lab + "方式的结果对比", fontproperties=font)

    plt.legend()

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"img/plot_{current_time}.png"
    plt.savefig(filename)
    plt.close()


## 有无CUDA对比
def diagram_4(no_cuda, with_cuda):
    print(no_cuda, with_cuda)
    plt.style.use("seaborn-v0_8")

    plt.plot(no_cuda, label="Without CUDA")
    plt.plot(with_cuda, label="With CUDA")

    plt.title("CUDA对速度的影响", fontproperties=font)

    plt.legend()

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"img/plot_{current_time}.png"
    plt.savefig(filename)
    plt.close()


## CUDA分层采样有无向量化下速度的对比
def diagram_5(with_vec, no_vec):
    print(with_vec, no_vec)
    plt.style.use("seaborn-v0_8")

    plt.plot(with_vec, label="With Vector")
    plt.plot(no_vec, label="Without Vector")

    plt.title("运算向量化对计算的影响", fontproperties=font)

    plt.legend()

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"img/plot_{current_time}.png"
    plt.savefig(filename)
    plt.close()


# 统计
font = matplotlib.font_manager.FontProperties(fname="ref/font.otf")
## 一般方法Numba开启与否时的运行情况
dis_time_dict = []
enab_time_dict = []
## 同一样本量下不同方法速度对比
s_speed = []
## 单个方法在不同样本量下的结果区别
a_dict = []
b_dict = []
c_dict = []
d_dict = []
e_dict = []
f_dict = []
g_dict = []
## 有无CUDA对比
no_cuda = []
with_cuda = []
## CUDA分层采样有无向量化下速度的对比
with_vec = []
no_vec = []

# 下标(x最小值)
bottom = 0
# 上标(x最大值)
top = 2 * np.pi

# 总执行次数
total_run = 3
# 样本个数
sample_num = 10**5
# 分层层数
layers = 10**3

# 真实积分估值
about = integrate.quad(dis_func, bottom, top)

print("总执行次数", total_run, ",样本个数", sample_num, ",分层层数:", layers, ",正确值:", about)

for i in range(total_run):
    print("======== 第", i + 1, "次执行 ========")
    print("______________________________")
    print("一般方法Numba开启与否时的运行情况")
    print("______________________________")
    cpu_time, result = calculate_cpu_time(dis_simple, dis_func, bottom, top, sample_num)
    if i != 0:
        dis_time_dict.append(cpu_time)
    print("无Numba:", cpu_time, "值:", result)
    time.sleep(3)

    cpu_time, result = calculate_cpu_time(simple, enab_func, bottom, top, sample_num)
    if i != 0:
        enab_time_dict.append(cpu_time)
    print("有Numba:", cpu_time, "值:", result)
    time.sleep(3)

diagram_1(dis_time_dict, enab_time_dict)

for i in range(total_run):
    print("======== 第", i + 1, "次执行 ========")
    print("______________________________")
    print("单个方法在不同样本量下的结果区别")
    print("______________________________")
    for j in range(4, 7):
        for_num = 10 ** (j + 1)
        for_layers = 10 ** (j // 2)
        print("样本个数", for_num)
        cpu_time, result = calculate_cpu_time(simple, enab_func, bottom, top, for_num)
        s_speed.append(cpu_time)
        if i == 1:
            a_dict.append(cpu_time)
            no_cuda.append(cpu_time)

        print("一般实现时间:", cpu_time, "值:", result)
        time.sleep(3)

        cpu_time, result = calculate_cpu_time(
            important, enab_func, bottom, top, for_num
        )
        s_speed.append(cpu_time)
        if i == 1:
            b_dict.append(cpu_time)
            no_cuda.append(cpu_time)
        print("重要性采样时间:", cpu_time, "值:", result)
        time.sleep(3)

        cpu_time, result = calculate_cpu_time(
            layer, enab_func, bottom, top, for_num, for_layers
        )
        s_speed.append(cpu_time)
        if i == 1:
            c_dict.append(cpu_time)
            no_cuda.append(cpu_time)
        print("分层抽样时间:", cpu_time, "层数:", for_layers, "值:", result)
        time.sleep(3)

        cpu_time, result = calculate_cpu_time(cuda_simple, bottom, 2 * cp.pi, for_num)
        s_speed.append(cpu_time)
        if i == 1:
            d_dict.append(cpu_time)
            with_cuda.append(cpu_time)
        print("CUDA 一般实现时间:", cpu_time, "值:", result)
        time.sleep(3)

        cpu_time, result = calculate_cpu_time(
            cuda_important, bottom, 2 * cp.pi, for_num
        )
        s_speed.append(cpu_time)
        if i == 1:
            e_dict.append(cpu_time)
            with_cuda.append(cpu_time)
        print("CUDA 重要性采样时间:", cpu_time, "值:", result)
        time.sleep(3)

        cpu_time, result = calculate_cpu_time(
            cuda_for_layer, bottom, 2 * cp.pi, for_num, layers
        )
        s_speed.append(cpu_time)
        if i == 1:
            f_dict.append(cpu_time)
            no_vec.append(cpu_time)
        print("CUDA 分层 非向量化抽样时间:", cpu_time, "层数:", for_layers, "值:", result)
        time.sleep(3)

        cpu_time, result = calculate_cpu_time(
            cuda_layer, bottom, 2 * cp.pi, for_num, layers
        )
        s_speed.append(cpu_time)
        if i == 1:
            g_dict.append(cpu_time)
            with_cuda.append(cpu_time)
            with_vec.append(cpu_time)
        print("CUDA 分层抽样时间:", cpu_time, "层数:", for_layers, "值:", result)
        time.sleep(3)
        diagram_2(s_speed, for_num)
        s_speed.clear()

time.sleep(1)
diagram_3(a_dict, "Normal")
time.sleep(1)
diagram_3(b_dict, "Important")
time.sleep(1)
diagram_3(c_dict, "Layer")
time.sleep(1)
diagram_3(d_dict, "CUDA Normal")
time.sleep(1)
diagram_3(e_dict, "CUDA Important")
time.sleep(1)
diagram_3(f_dict, "CUDA Layer")
time.sleep(1)
diagram_3(g_dict, "CUDA Vector Layer")
time.sleep(1)
diagram_4(no_cuda, with_cuda)
time.sleep(1)
diagram_5(with_vec, no_vec)
