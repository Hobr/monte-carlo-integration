# [蒙特卡洛积分计算](https://github.com/Hobr/monte-carlo-integration/)

> 运用了一些黑魔法来使用蒙特卡洛法计算定积分

南京理工大学 2024年 基于解释性语言的高效科学计算 课程大作业

## 生成的结果

### 纵向

1. 无Numba
2. Numba
3. CUDA

### 横向

1. 一般算法
2. 重要性采样
3. 分层采样
4. CUDA 分层采样

### 展示内容

- 一般方法Numba开启与否时的运行情况
- 同一样本量下不同方法速度对比
- 单个方法在不同样本量下的结果区别
- 有无CUDA对比
- CUDA分层采样有无向量化下速度的对比
- 不同样本量下的值与真实值对比

## 使用

### Julia

```bash
export JULIA_PKG_SERVER=https://mirrors.ustc.edu.cn/julia
julia
]add Distributions JuliaFormatter
using JuliaFormatter
format_file("main.jl")
include("main.jl")
exit()
```

### Python

```bash
pip install -r requirements.txt
black *.py
isort *.py
python main.py
```

## 参考

<https://www.labri.fr/perso/nrougier/from-python-to-numpy/>
<https://www.zhihu.com/question/67310504>
<https://www.nvidia.cn/glossary/data-science/numpy/>
<https://developer.nvidia.com/blog/copperhead-data-parallel-python/>
<https://developer.nvidia.com/blog/seven-things-numba/>
<https://developer.nvidia.com/blog/gpu-accelerated-graph-analytics-python-numba/>
<https://numpy.org/doc/stable/user/absolute_beginners.html#plotting-arrays-with-matplotlib>
