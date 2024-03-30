import time

import matplotlib.pyplot as plt
import numpy as np
from numba import jit


# 函数式
## y(x) = 2sin(x) (x^3+ x^2+ 2x+ 3)
@jit(nopython=True)
def func(x):
    return np.multiply(
        np.multiply(np.sin(x), 2),
        np.add(np.add(np.power(x, 3), np.power(x, 2)), np.add(np.multiply(x, 2), 3)),
    )


# 蒙特卡洛
## 参数: 函数, 区间左, 区间右, 次数
@jit(nopython=True, parallel=True)
def monte(func, bottom, top, times):
    x = np.random.uniform(bottom, top, times)
    print(x)
    y = func(x)
    print(y)
    dist = np.multiply(np.mean(y), np.subtract(top, bottom))
    return dist


print(monte(func, 0, 2 * np.pi, 10**8))
