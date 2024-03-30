# 蒙特卡洛积分计算法

> 运用了许多黑魔法来使用蒙特卡洛法计算定积分

南京理工大学 2024年 基于解释性语言的高效科学计算 课程大作业

## 实现

- 最简单实现
- 并行计算实现
- CPU特性加速
- AVX
- Cython
- 函数向量化
- Numba CUDA

## TODO

- 多线程
- 画图
- PyPy
- 性能
- PPT/Word

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
black main.py
isort main.py
python main.py
```

## 参考

<https://www.zhihu.com/question/67310504>
<https://www.nvidia.cn/glossary/data-science/numpy/>
<https://developer.nvidia.com/blog/numba-python-cuda-acceleration/>
<https://developer.nvidia.com/blog/copperhead-data-parallel-python/>
<https://github.com/ContinuumIO/gtc2017-numba>
<https://developer.nvidia.com/blog/seven-things-numba/>
<https://developer.nvidia.com/blog/gpu-accelerated-graph-analytics-python-numba/>
<https://numpy.org/doc/stable/reference/simd/build-options.html>
<https://numba.pydata.org/numba-doc/latest/cuda/overview.html#terminology>
