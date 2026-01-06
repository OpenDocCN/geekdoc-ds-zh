# 拒绝采样究竟有多耗时？

> 原文：[`www.algorithm-archive.org/contents/box_muller/box_muller_rejection.html`](https://www.algorithm-archive.org/contents/box_muller/box_muller_rejection.html)

让我们想象我们想要一个最终包含 \( n \) 个粒子的高斯分布。使用笛卡尔 Box-Muller 方法，这很简单：以 \( n \) 个粒子开始初始分布，然后进行转换。只要我们从一个均匀分布的**圆**而不是均匀分布的**正方形**开始，极坐标 Box-Muller 方法也可以同样简单。也就是说，只要我们在事先进行拒绝采样，极坐标 Box-Muller 方法就会始终更有效。公平地说，有一些方法可以在不进行拒绝采样的情况下生成圆内的均匀分布点，但让我们假设在这个例子中我们需要拒绝采样。

这意味着有人需要为极坐标方法进行拒绝采样，这有时是一个痛苦的过程。这也意味着 Box-Muller 方法可以用来教授一些通用 GPU 计算的基本原理。请注意，由于这个问题的特殊性，本节的所有代码都将使用 Julia 编写，并使用 KernelAbstractions.jl 包，这使得我们可以在配置方式的基础上在 CPU 或 GPU 硬件上执行相同的内核。

首先，让我们考虑这种情况：我们将拒绝采样作为极坐标 Box-Muller 内核的一部分，而不是作为预处理步骤。在这种情况下，我们可以想象有两种不同的方式来编写我们的内核：

1.  有放回：在这种情况下，我们**绝对需要**最终在高斯分布中的点数，因此如果我们运行内核时发现一个点在单位圆外，我们将“重新投掷”以获得一个**确实**在圆内的新点。

1.  无放回：这意味着我们将从一个均匀分布的点开始，但最终得到一个有 \( n \) 个点的正态分布。在这种情况下，如果我们运行内核时发现一个点在单位圆外，我们只需通过将输出值设置为 NaN（或类似值）来忽略它。

好吧，首先是有放回的情况：

```
@kernel function polar_muller_replacement!(input_pts, output_pts, sigma, mu)
    tid = @index(Global, Linear)
    @inbounds r_0 = input_pts[tid, 1]² + input_pts[tid, 2]²

    while r_0 > 1 || r_0 == 0
        p1 = rand()*2-1
        p2 = rand()*2-1
        r_0 = p1² + p2²
    end

    @inbounds output_pts[tid,1] = sigma * input_pts[tid,1] *
                                  sqrt(-2 * log(r_0) / r_0) + mu
    @inbounds output_pts[tid,2] = sigma * input_pts[tid, 2] *
                                  sqrt(-2 * log(r_0) / r_0) + mu
end 
```

这个想法有很多原因是不好的。以下是一些原因：

1.  如果我们找到一个点在单位圆外，我们必须不断地寻找新点，直到我们**确实**找到一个在圆内的点。因为我们是在并行运行这个程序，其中每个线程一次转换一个点，一些线程可能永远找不到新点（如果我们真的很不走运）。

1.  要生成新点，我们需要重新生成一个均匀分布，但如果我们均匀分布不是随机的怎么办？如果它是一个网格（或类似的东西）呢？在这种情况下，我们真的不应该在圆内寻找新点，因为所有这些点都已经计算在内了。

1.  在一些并行平台（如 GPU）上，`rand()`函数有点棘手，可能不会直接工作。事实上，上面显示的实现只能在 CPU 上运行。

好吧，好吧。我认为没有人会期望一个内部有`while`循环的核能快。那么，没有替换的方法怎么样？如果我们完全忽略`while`循环，肯定没问题！但是，这种方法的问题稍微不那么直接，首先，代码：

```
@kernel function polar_muller_noreplacement!(input_pts, output_pts, sigma, mu)
    tid = @index(Global, Linear)
    @inbounds r_0 = input_pts[tid, 1]² + input_pts[tid, 2]²

    # this method is only valid for points within the unit circle
    if r_0 == 0 || r_0 > 1
        @inbounds output_pts[tid,1] = NaN
        @inbounds output_pts[tid,2] = NaN
    else
        @inbounds output_pts[tid,1] = sigma * input_pts[tid,1] *
                                      sqrt(-2 * log(r_0) / r_0) + mu
        @inbounds output_pts[tid,2] = sigma * input_pts[tid, 2] *
                                      sqrt(-2 * log(r_0) / r_0) + mu
    end

end 
```

要开始讨论为什么没有替换的极化核也是一个糟糕的想法，让我们回到蒙特卡洛章节，在那里我们将它嵌入到一个圆中，计算了其值。在那里，我们发现随机选择一个点落在单位圆内的概率为，如下面的视觉图中所示：

![图片](img/89c9f51f3a2b57a4095e1f959e9c27fb.png)

这意味着在圆内的点均匀分布将拒绝正方形上的*点*。这也意味着如果我们有一个特定的最终分布值，我们平均需要更多的输入值！

没问题！在这个假设的情况下，我们不需要*正好*这么多点，所以我们可以从*点*开始初始分布，对吧？

对的。这将在并行 CPU 硬件上运行得很好，但在 GPU 上仍然会有问题。

在 GPU 上，所有计算都是并行进行的，但有一个最小的并行单位，称为*warp*。warp 是能够并行执行的最小线程数，通常约为 32。这意味着如果有一个操作被队列化，所有 32 个线程将同时执行它。如果有 16 个线程需要执行某事，而其他 16 个线程需要执行其他事情，这将导致*warp 发散*，需要执行两个动作而不是一个：

![图片](img/a9dbaba2342dc4399148e466c87e8c58.png)

在这个图像中，每个奇数线程都需要执行粉红色的动作，而偶数线程需要执行蓝色的动作。这意味着将执行两个独立的并行任务，一个用于偶数线程，另一个用于奇数线程。这意味着如果队列中有*个分离的操作，所有线程完成工作可能需要*那么长的时间！这就是为什么内核中的`if`语句可能很危险！如果使用不当，它们可能导致 warp 中的某些线程执行不同的操作！

让我们想象上面的图像是更大数组的一部分，这样就有很多具有相同发散问题的 warp。在这种情况下，我们可以在之前对数组进行排序，以便所有偶数元素都在所有奇数元素之前。这意味着 warp 几乎肯定不会发散，因为队列中的元素都将属于同一类型并需要相同的操作。不幸的是，这要以一个昂贵的排序操作为代价。

如果我们查看上面的内核，我们实际上是在要求我们的线程做不同于其他人的事情，因为我们通常输入的是均匀随机分布，这意味着大多数 warp 将不得不排队进行 2 个并行操作而不是 1 个。

实际上，我们需要选择我们的“毒药”：

+   使用笛卡尔方法进行缓慢和操作

+   极坐标方法中的 warp 发散

唯一知道哪种方法更好的方法是进行基准测试，我们将在稍后展示，但还有一个最终场景我们应该考虑：将拒绝采样作为预处理步骤会怎样？这意味着我们预先初始化极坐标核，使其在单位圆内具有均匀分布的点。这意味着没有 warp 发散，因此我们可以得到两者的最佳结合，对吧？

嗯，其实不尽然。极坐标 Box—Muller 方法确实会更快，但再次强调：某处某人需要执行拒绝采样，如果我们把这一步骤包含到过程中，事情又会变得复杂起来。事实是，这个预处理步骤很难做对，可能需要单独的一章来详细说明。

在许多情况下，在操作之前花点时间确保后续操作快速是值得的，但在这个案例中，我们只有一个操作，而不是一系列操作。Box—Muller 方法通常只在模拟开始时使用一次，这意味着拒绝采样的预处理步骤可能过于冗余。

无论哪种情况，基准测试将展示我们在这里处理的真实情况：

| 方法 | CPU | GPU |
| --- | --- | --- |
| 笛卡尔 | ms | ms |
| 极坐标不替换 | ms | ms |
| 极坐标替换 | ms | NA |

这些基准测试是在 Nvidia GTX 970 GPU 和 Ryzen 3700X 16 核心 CPU 上运行的。对感兴趣的读者，代码可以在下面找到。对于这些基准测试，我们使用了 Julia 的内置基准测试套件`BenchmarkTools`，并确保使用`CUDA.@sync`同步 GPU 内核。我们还使用了输入点进行了测试。

在这里，我们看到结果出现了有趣的分歧。在 CPU 上，极坐标方法总是更快，但在 GPU 上，两种方法相当。我相信这是从 Box—Muller 方法中可以学到的最重要的教训：有时候，无论你如何努力优化你的代码，不同的硬件可以提供截然不同的结果！确保代码的实际性能如你所想，进行基准测试是极其重要的！

## 完整脚本

```
using KernelAbstractions
using CUDA

if has_cuda_gpu()
    using CUDAKernels
end

function create_grid(n, endpoints; AT = Array)

    grid_extents = endpoints[2] - endpoints[1]

    # number of points along any given axis
    # For 2D, we take the sqrt(n) and then round up
    axis_num = ceil(Int, sqrt(n))

    # we are now rounding n up to the nearest square if it was not already one
    if sqrt(n) != axis_num
       n = axis_num²
    end 

    # Distance between each point
    dx = grid_extents / (axis_num)

    # This is warning in the case that we do not have a square number
    if sqrt(n) != axis_num
        println("Cannot evenly divide ", n, " into 2 dimensions!")
    end 

    # Initializing the array, particles along the column, dimensions along rows
    a = AT(zeros(n, 2))

    # This works by firxt generating an N dimensional tuple with the number
    # of particles to be places along each dimension ((10,10) for 2D and n=100)
    # Then we generate the list of all CartesianIndices and cast that onto a
    # grid by multiplying by dx and subtracting grid_extents/2
    for i = 1:axis_num
        for j = 1:axis_num
            a[(i - 1) * axis_num + j, 1] = i * dx + endpoints[1]
            a[(i - 1) * axis_num + j, 2] = j * dx + endpoints[1]
        end
    end

    return a
end

function create_rand_dist(n, endpoints; AT = Array)
    grid_extents = endpoints[2] - endpoints[1]
    return(AT(rand(n,2)) * grid_extents .+ endpoints[1]) 
end

# This function reads in a pair of input points and performs the Cartesian
# Box--Muller transform
@kernel function polar_muller_noreplacement!(input_pts, output_pts, sigma, mu)
    tid = @index(Global, Linear)
    @inbounds r_0 = input_pts[tid, 1]² + input_pts[tid, 2]²

    # this method is only valid for points within the unit circle
    if r_0 == 0 || r_0 > 1
        @inbounds output_pts[tid,1] = NaN
        @inbounds output_pts[tid,2] = NaN
    else
        @inbounds output_pts[tid,1] = sigma * input_pts[tid,1] *
                                      sqrt(-2 * log(r_0) / r_0) + mu
        @inbounds output_pts[tid,2] = sigma * input_pts[tid, 2] *
                                      sqrt(-2 * log(r_0) / r_0) + mu
    end

end

@kernel function polar_muller_replacement!(input_pts, output_pts, sigma, mu)
    tid = @index(Global, Linear)
    @inbounds r_0 = input_pts[tid, 1]² + input_pts[tid, 2]²

    while r_0 > 1 || r_0 == 0
        p1 = rand()*2-1
        p2 = rand()*2-1
        r_0 = p1² + p2²
    end

    @inbounds output_pts[tid,1] = sigma * input_pts[tid,1] *
                                  sqrt(-2 * log(r_0) / r_0) + mu
    @inbounds output_pts[tid,2] = sigma * input_pts[tid, 2] *
                                  sqrt(-2 * log(r_0) / r_0) + mu
end

function polar_box_muller!(input_pts, output_pts, sigma, mu;
                           numthreads = 256, numcores = 4,
                           f = polar_muller_noreplacement!)
    if isa(input_pts, Array)
        kernel! = f(CPU(), numcores)
    else
        kernel! = f(CUDADevice(), numthreads)
    end
    kernel!(input_pts, output_pts, sigma, mu, ndrange=size(input_pts)[1])
end

@kernel function cartesian_kernel!(input_pts, output_pts, sigma, mu)
    tid = @index(Global, Linear)

    @inbounds r = sqrt(-2 * log(input_pts[tid,1]))
    @inbounds theta = 2 * pi * input_pts[tid, 2]

    @inbounds output_pts[tid,1] = sigma * r * cos(theta) + mu
    @inbounds output_pts[tid,2] = sigma * r * sin(theta) + mu
end

function cartesian_box_muller!(input_pts, output_pts, sigma, mu;
                               numthreads = 256, numcores = 4)
    if isa(input_pts, Array)
        kernel! = cartesian_kernel!(CPU(), numcores)
    else
        kernel! = cartesian_kernel!(CUDADevice(), numthreads)
    end

    kernel!(input_pts, output_pts, sigma, mu, ndrange=size(input_pts)[1])
end

function main()

    input_pts = create_rand_dist(4096²,[0,1])
    output_pts = create_rand_dist(4096²,[0,1])

    wait(cartesian_box_muller!(input_pts, output_pts, 1, 0))
    @time wait(cartesian_box_muller!(input_pts, output_pts, 1, 0))
    wait(polar_box_muller!(input_pts, output_pts, 1, 0))
    @time wait(polar_box_muller!(input_pts, output_pts, 1, 0))

    if has_cuda_gpu()
        input_pts = create_rand_dist(4096²,[0,1], AT = CuArray)
        output_pts = create_rand_dist(4096²,[0,1], AT = CuArray)

        wait(cartesian_box_muller!(input_pts, output_pts, 1, 0))
        CUDA.@time wait(cartesian_box_muller!(input_pts, output_pts, 1, 0))
        wait(polar_box_muller!(input_pts, output_pts, 1, 0))
        CUDA.@time wait(polar_box_muller!(input_pts, output_pts, 1, 0))
    end

end

main() 
```

## 许可证

##### 代码示例

代码示例采用 MIT 许可协议（可在[LICENSE.md](https://github.com/algorithm-archivists/algorithm-archive/blob/main/LICENSE.md)中找到）。

##### 文本

本章的文本由[James Schloss](https://github.com/leios)编写，并采用[Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/legalcode)许可协议。

[](https://creativecommons.org/licenses/by-sa/4.0/)

![](https://creativecommons.org/licenses/by-sa/4.0/)
