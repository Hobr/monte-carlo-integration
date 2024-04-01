# 蒙特卡洛积分计算法

> 运用了许多黑魔法来使用蒙特卡洛法计算定积分

南京理工大学 2024年 基于解释性语言的高效科学计算 课程大作业

## TODO

- 误差
- 画图
  - 不同方法速度对比
  - x
  - 不同方法得出的积分+正确值
  - 系统占用
- PyPy
- PPT/Word

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
