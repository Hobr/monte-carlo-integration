using Distributions

# 原函数
## y(x) = 2sin(x) (x^3+ x^2+ 2x+ 3)
function func(x)
    sinx = map(sin, x)
    return 2 * sinx * (x^3 + x^2 + 2x + 3)
end

# 常规实现
## 参数:原函数,最小值,最大值,样本量
function simple(func, bottom, top, times)
    random_x = rand(Uniform(bottom, top), times)
    integ_sum = func.(random_x)
    dist = mean(integ_sum) * (top - bottom)
    return dist
end

# 重要性采样
## 参数:原函数,最小值,最大值,样本量
function important(func, bottom, top, times)
    y = rand(Uniform(bottom, top), times)
    percent = func.(y) / (1 / (2 * pi))
    dist = mean(percent)
    return dist
end

# 分层采样
## 参数:原函数,最小值,最大值,样本量,分层数
function layer(func, bottom, top, times, layers)
    dist = 0.0
    width = (top - bottom) / layers
    for i = 0:layers-1
        lay_sample = rand(
            Uniform(bottom + i * width, bottom + (i + 1) * width),
            Int(times // layers),
        )
        lay_dist = mean(func.(lay_sample))
        dist += lay_dist * width
    end
    return dist
end

println(simple(func, 0, 2 * pi, 10^8))
println(important(func, 0, 2 * pi, 10^8))
println(layer(func, 0, 2 * pi, 10^8, 10^3))
