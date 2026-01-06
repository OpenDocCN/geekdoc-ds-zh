# 机器代码分析器

> 原文：[`en.algorithmica.org/hpc/profiling/mca/`](https://en.algorithmica.org/hpc/profiling/mca/)

*机器代码分析器* 是一个程序，它接受一小段汇编代码，并使用编译器可用的信息在特定的微架构上模拟其执行，并输出整个块的延迟和吞吐量，以及 CPU 内部各种资源的完美周期利用率。

### [#](https://en.algorithmica.org/hpc/profiling/mca/#using-llvm-mca)使用 `llvm-mca`

有许多不同的机器代码分析器，但我个人更喜欢 `llvm-mca`，你很可能可以通过包管理器与 `clang` 一起安装它。你也可以通过一个名为 [UICA](https://uica.uops.info) 的基于 Web 的工具或通过在 [Compiler Explorer](https://godbolt.org/) 中选择“分析”作为语言来访问它。

`llvm-mca` 做的是运行给定汇编代码片段的一组迭代次数，并计算每个指令的资源使用统计信息，这对于找出瓶颈位置非常有用。

我们将数组求和作为我们的简单示例：

```cpp
loop:
    addl (%rax), %edx
    addq $4, %rax
    cmpq %rcx, %rax
    jne	 loop 
```

这是使用 `llvm-mca` 对 Skylake 微架构的分析：

```cpp
Iterations:        100
Instructions:      400
Total Cycles:      108
Total uOps:        500

Dispatch Width:    6
uOps Per Cycle:    4.63
IPC:               3.70
Block RThroughput: 0.8 
```

首先，它输出循环和硬件的一般信息：

+   它“运行”了 100 次循环，总共执行了 400 条指令，耗时 108 周期，平均每周期执行 $\frac{400}{108} \approx 3.7$ 条指令（IPC）。

+   CPU 理论上每周期可以执行多达 6 条指令（调度宽度）。

+   理论上，每个周期平均可以在 0.8 个周期内执行（块倒数吞吐量）。

+   这里的“uOps”是指 CPU 将每条指令分解成的微操作（例如，融合加载加法由两个 uOps 组成）。

然后它继续提供关于每个单独指令的信息：

```cpp
Instruction Info:
[1]: uOps
[2]: Latency
[3]: RThroughput
[4]: MayLoad
[5]: MayStore
[6]: HasSideEffects (U)

[1]    [2]    [3]    [4]    [5]    [6]    Instructions:
 2      6     0.50    *                   addl	(%rax), %edx
 1      1     0.25                        addq	$4, %rax
 1      1     0.25                        cmpq	%rcx, %rax
 1      1     0.50                        jne	-11 
```

在指令表中没有什么是不存在的：

+   每条指令分解成多少个 uOps；

+   每条指令完成所需多少周期（延迟）；

+   考虑到可以同时执行多个副本，每条指令完成所需的周期数（平均倒数吞吐量）；

然后它输出可能最重要的部分——哪些指令在何时何地执行：

```cpp
Resource pressure by instruction:
[0]    [1]    [2]    [3]    [4]    [5]    [6]    [7]    [8]    [9]    Instructions:
 -      -     0.01   0.98   0.50   0.50    -      -     0.01    -     addl (%rax), %edx
 -      -      -      -      -      -      -     0.01   0.99    -     addq $4, %rax
 -      -      -     0.01    -      -      -     0.99    -      -     cmpq %rcx, %rax
 -      -     0.99    -      -      -      -      -     0.01    -     jne  -11 
```

由于对执行端口的竞争导致结构冒险，端口经常成为吞吐量导向型循环的瓶颈，此图表有助于诊断原因。它不会给你一个完美的周期 Gantt 图，但会给出每个指令使用的执行端口的聚合统计数据，这让你可以找出哪个端口过载。[← 程序模拟](https://en.algorithmica.org/hpc/profiling/simulation/)[基准测试 →](https://en.algorithmica.org/hpc/profiling/benchmarking/)
