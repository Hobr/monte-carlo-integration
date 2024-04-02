# 蒙特卡洛积分计算法

> 运用了许多黑魔法来使用蒙特卡洛法计算定积分

南京理工大学 2024年 基于解释性语言的高效科学计算 课程大作业

## 生成的结果

### 纵向

1. 无Numba 
2. Numba
3. Numba+CUDA
4. CuPy

### 横向

1. 无向量一般算法
2. 一般算法
3. 重要性采样
4. 分层采样

### 展示内容

1. numba开启与否时的运行情况
2. f(x)向量化与否时生成随机数用时
3. 单个方法下的x y拟合
4. 单个方法在不同样本量下的结果区别
5. 单个算法在CPU和GPU下的速度区别
6. 多个算法间的速度区别
7. 各算法结果与真实结果区别

## 使用

### Julia

```bash
export JULIA_PKG_SERVER=https://mirrors.ustc.edu.cn/julia
julia
]add Distributions JuliaFormatter
using JuliaFormatter
format_file("cpu.jl")
format_file("gpu.jl")
include("cpu.jl")
include("gpu.jl")
exit()
```

### Python

```bash
pip install -r requirements.txt
black *.py
isort *.py
python cpu.py
python gpu.py
```

## 参考

<https://www.labri.fr/perso/nrougier/from-python-to-numpy/>
<https://www.zhihu.com/question/67310504>
<https://www.nvidia.cn/glossary/data-science/numpy/>
<https://developer.nvidia.com/blog/numba-python-cuda-acceleration/>
<https://developer.nvidia.com/blog/copperhead-data-parallel-python/>
<https://github.com/ContinuumIO/gtc2017-numba>
<https://developer.nvidia.com/blog/seven-things-numba/>
<https://developer.nvidia.com/blog/gpu-accelerated-graph-analytics-python-numba/>
<https://numpy.org/doc/stable/reference/simd/build-options.html>
<https://numba.pydata.org/numba-doc/latest/cuda/overview.html#terminology>
<https://numpy.org/doc/stable/user/absolute_beginners.html#plotting-arrays-with-matplotlib>
<https://cuda.juliagpu.org/stable/>
