# 内存延迟

> 原文：[`en.algorithmica.org/hpc/cpu-cache/latency/`](https://en.algorithmica.org/hpc/cpu-cache/latency/)

尽管带宽是一个更复杂的概念，但它比延迟更容易观察和测量：你可以简单地执行一系列独立的读取或写入查询，调度器在事先知道这些查询的情况下，重新排序并重叠它们，隐藏它们的延迟，并最大化总吞吐量。

为了测量*延迟*，我们需要设计一个实验，在这个实验中，CPU 不能通过提前知道我们将请求的内存位置来作弊。确保这一点的 一种方法是通过生成一个大小为 $N$ 的随机排列，该排列对应于一个循环，然后反复遵循该排列：

```cpp
int p[N], q[N];   // generating a random permutation iota(p, p + N, 0); random_shuffle(p, p + N);   // this permutation may contain multiple cycles, // so instead we use it to construct another permutation with a single cycle int k = p[N - 1]; for (int i = 0; i < N; i++)  k = q[k] = p[i];   for (int t = 0; t < K; t++)  for (int i = 0; i < N; i++) k = q[k]; 
```

与线性迭代相比，以这种方式遍历数组的所有元素要慢得多——慢得多——多个数量级。这不仅使得 SIMD 成为不可能，而且还 使流水线停滞，创建了一个巨大的指令交通堵塞，所有指令都在等待从内存中获取单个数据。

这种性能反模式被称为*指针追查*，它在数据结构中非常常见，尤其是在使用大量堆分配对象及其指针的高级语言编写的那些数据结构中，这些指针对于动态类型是必要的。

![](img/679a67a9cc2e67d6e8233f2aef54db53.png)

当谈论延迟时，使用周期或纳秒而不是吞吐量单位更有意义，因此我们用它的倒数来替换这个图表：

![](img/37ff0412d79dea6330b801c37cc61f52.png)

注意，这两个图表上的悬崖不像带宽图表上那么明显。这是因为即使数组不能完全放入其中，我们仍然有可能命中缓存的前一层。

### [#](https://en.algorithmica.org/hpc/cpu-cache/latency/#theoretical-latency)理论延迟

更正式地说，如果缓存层次结构中有 $k$ 个级别，大小分别为 $s_i$，延迟分别为 $l_i$，那么，它们的期望延迟将不会等于最慢的访问，而是：

$$ E[L] = \frac{ s_1 \cdot l_1 + (s_2 - s_1) \cdot l_2 % + (s_3 - s_2) \cdot l_3 + \ldots + (N - s_k) \cdot l_{RAM} }{N} $$ 如果我们抽象掉所有发生在最慢缓存层之前的事情，我们可以将公式简化为仅仅这个：$$ E[L] = \frac{N \cdot l_{last} - C}{N} = l_{last} - \frac{C}{N} $$ 随着 $N$ 的增加，期望延迟逐渐接近 $l_{last}$，如果你足够用力地眯眼，吞吐量（倒数延迟）的图表应该大致看起来像是由几个转置和缩放的双曲线组成的：$$ \begin{aligned} E[L]^{-1} &= \frac{1}{l_{last} - \frac{C}{N}} \\ &= \frac{N}{N \cdot l_{last} - C} \\ &= \frac{1}{l_{last}} \cdot \frac{N + \frac{C}{l_{last}} - \frac{C}{l_{last}}}{N - \frac{C}{l_{last}}} \\ &= \frac{1}{l_{last}} \cdot \left(\frac{1}{N \cdot \frac{l_{last}}{C} - 1} + 1\right) \\ &= \frac{1}{k \cdot (x - x_0)} + y_0 \end{aligned} $$

要获取实际的延迟数值，我们可以迭代地应用第一个公式来推导$l_1$，然后是$l_2$，依此类推。或者，只需查看悬崖前的值——它们应该在大约 10-15%的真延迟范围内。

有更直接的方式来测量延迟，包括使用非临时读取，但这个基准更能代表实际的访问模式。

### [#](https://en.algorithmica.org/hpc/cpu-cache/latency/#frequency-scaling)频率缩放

与带宽类似，所有 CPU 缓存的延迟与其时钟频率成比例增加，而 RAM 则不然。如果我们通过开启超频来改变频率，我们也可以观察到这种差异。

![](img/52a8679535465aa0773ba3893cbcc533.png)

如果我们将它作为相对加速率来绘制，图表开始变得更有意义。

![](img/3448a82210f51d78cfa7ee4a30cc4db3.png)

你可能会期望对于完全适合 CPU 缓存的数组大小有 2 倍的速度，但对于存储在 RAM 中的数组则大致相等。但实际情况并非如此：即使是对于 RAM 访问，也存在一个小的、固定的延迟。这是因为 CPU 在将读取查询派遣到主内存之前，首先必须检查其缓存——为了节省可能需要的 RAM 带宽给其他进程。

内存延迟也受到虚拟内存实现和 RAM 特定时序的一些细节的影响，这些我们将在后面讨论。[← 内存带宽](https://en.algorithmica.org/hpc/cpu-cache/bandwidth/)[缓存行 →](https://en.algorithmica.org/hpc/cpu-cache/cache-lines/)
