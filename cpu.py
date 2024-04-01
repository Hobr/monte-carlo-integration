from timeit import timeit
from time import sleep
import numpy as np
from numba import jit

# 总执行次数
for_time = 3
# 随机数个数
times = 10**9 + (7 * 10**8)
# 分层层数
layers = 10**5

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
    random_x = np.random.uniform(bottom, top, times)
    # 计算y/积分和
    integ_sum = func(random_x)
    # y的平均数*(top-bottom) 积分中值
    dist = np.multiply(np.mean(integ_sum), np.subtract(top, bottom))

    print(dist)
    return dist


# 重要性采样
@jit(nopython=True, parallel=True)
def important():
    # 在均匀分布中生成x
    y = np.random.uniform(bottom, top, times)
    # 计算函数值与概率分布值的比例
    ## 概率分布函数: 1/2pi,即均匀分布
    ## func(y) / (1 / (2 * np.pi))
    percent = np.divide(func(y), np.divide(1, np.multiply(2, np.pi)))
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
            # bottom + i * (top - bottom) / layers
            np.add(bottom, np.multiply(i, np.divide(np.subtract(top, bottom), layers))),
            # bottom + (i + 1) * (top - bottom) / layers,
            np.add(
                bottom,
                np.multiply(np.add(i, 1), np.divide(np.subtract(top, bottom), layers)),
            ),
            # times // layers,
            np.floor_divide(times, layers),
        )
        # 单层积分
        lay_dist = np.mean(func(lay_sample))
        # 加权平均区间积分
        # dist += lay_dist * (top - bottom) / layers
        dist = np.add(
            dist, np.multiply(lay_dist, np.divide(np.subtract(top, bottom), layers))
        )

    print(dist)
    return dist


print(
    "总执行次数",
    for_time,
    "\n随机数个数",
    times,
    "\n分层层数:",
    layers,
)
print("===========CPU执行===========")
print("一般实现时间:", timeit("simple()", globals=globals(), number=for_time))
sleep(3)
print("重要性采样时间:", timeit("important()", globals=globals(), number=for_time))
sleep(3)
print("分层采样时间:", timeit("layer()", globals=globals(), number=for_time))
sleep(3)
