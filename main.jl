using Distributions

# 函数式
## y(x) = 2sin(x) (x^3+ x^2+ 2x+ 3)
function func(x)
    return 2 * sin(x) * (x^3 + x^2 + 2x + 3)
end

# 蒙特卡洛
## 参数: 函数, 区间左, 区间右, 次数
function monte(func, bottom, top, times)
    x = rand(Uniform(bottom, top), times)
    y = func.(x)
    dist = mean(y) * (top - bottom)
    return dist
end

println(monte(func, 0, 2 * pi, 10^8))

