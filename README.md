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

1. 一般方法Numba开启与否时的运行情况
  - 关闭 dis_func(), dis_simple()
  - 开启 enab_func(), simple()

2. 多个算法间的速度 结果 x y拟合
  - CPU下三种方法 enab_func ,cuda...
  - GPU下三种方法 enab_func ,cuda...

3. 单个方法在不同样本量下的结果区别
  - 10**(9-2)

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
<https://developer.nvidia.com/blog/numba-python-cuda-acceleration/>
<https://developer.nvidia.com/blog/copperhead-data-parallel-python/>
<https://github.com/ContinuumIO/gtc2017-numba>
<https://developer.nvidia.com/blog/seven-things-numba/>
<https://developer.nvidia.com/blog/gpu-accelerated-graph-analytics-python-numba/>
<https://numpy.org/doc/stable/reference/simd/build-options.html>
<https://numba.pydata.org/numba-doc/latest/cuda/overview.html#terminology>
<https://numpy.org/doc/stable/user/absolute_beginners.html#plotting-arrays-with-matplotlib>
<https://cuda.juliagpu.org/stable/>
