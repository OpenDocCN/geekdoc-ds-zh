# 内存级并行性

> 原文：[`en.algorithmica.org/hpc/cpu-cache/mlp/`](https://en.algorithmica.org/hpc/cpu-cache/mlp/)

内存请求可以在时间上重叠：当您等待读取请求完成时，您可以发送几个其他请求，这些请求将与它并发执行。这是为什么线性迭代比指针跳跃快得多的主要原因：CPU 知道它需要获取下一个内存位置，并且会提前发送内存请求。

并发的内存操作数量很大但有限，并且对于不同类型的内存来说各不相同。在设计算法和特别是数据结构时，您可能想知道这个数字，因为它限制了您的计算可以实现的并行程度。

要理论上找到特定内存类型的这个极限，您可以将其延迟（获取缓存行的耗时）乘以其带宽（每秒获取的缓存行数），这将给出平均内存操作数：

![](img/140f45accd593c65f4ed6d9695db7d77.png)

L1/L2 缓存的延迟很小，因此不需要一个长的待处理请求管道，但较大的内存类型可以支持高达 25-40 个并发读取操作。

### [#](https://en.algorithmica.org/hpc/cpu-cache/mlp/#direct-experiment)直接实验

让我们尝试通过修改我们的指针追踪基准测试来更直接地测量可用的内存并行性，这样我们就可以并行地循环$D$个不同的周期，而不是只循环一个：

```cpp
const int M = N / D; int p[M], q[D][M];   for (int d = 0; d < D; d++) {  iota(p, p + M, 0); random_shuffle(p, p + M); k[d] = p[M - 1]; for (int i = 0; i < M; i++) k[d] = q[d][k[d]] = p[i]; }   for (int i = 0; i < M; i++)  for (int d = 0; d < D; d++) k[d] = q[d][k[d]]; 
```

将周期长度的总和固定在几个选定的大小，并尝试不同的$D$，我们得到略微不同的结果：

![](img/6d0a280245a088fb7de18b7aa488862e.png)

如预测的那样，L2 缓存的运行受到大约 6 个并发操作的限制，但较大的内存类型都在 13 到 17 之间达到最大值。由于逻辑寄存器存在冲突，您无法利用更多的内存通道。当通道数少于寄存器数时，您每条通道只能发出一个读取指令：

```cpp
dec     edx movsx   rdi, DWORD PTR q[0+rdi*4] movsx   rsi, DWORD PTR q[1048576+rsi*4] movsx   rcx, DWORD PTR q[2097152+rcx*4] movsx   rax, DWORD PTR q[3145728+rax*4] jne     .L9 
```

但当超过 15 时，您必须使用临时内存存储：

```cpp
mov     edx, DWORD PTR q[0+rdx*4] mov     DWORD PTR [rbp-128+rax*4], edx 
```

您并不总是能达到最大可能的内存并行级别，但对于大多数应用来说，十二个并发请求已经足够多了。[←内存共享](https://en.algorithmica.org/hpc/cpu-cache/sharing/)[预取→](https://en.algorithmica.org/hpc/cpu-cache/prefetching/)
