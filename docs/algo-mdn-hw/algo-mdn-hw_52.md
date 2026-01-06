# RAM 与 CPU 缓存

> 原文：[`en.algorithmica.org/hpc/cpu-cache/`](https://en.algorithmica.org/hpc/cpu-cache/)

在上一章中，我们从理论角度研究了计算机内存，使用外部内存模型来估算内存密集型算法的性能。

虽然外部内存模型对于涉及硬盘和网络存储的计算来说大致准确，在这些情况下，内存值算术运算的成本与外部 I/O 操作的成本相比可以忽略不计，但对于缓存层次结构中的较低级别来说，这些操作的代价变得可以比较，因此它过于不精确。

为了对内存算法进行更精细的优化，我们必须开始考虑 CPU 缓存系统的许多具体细节。而不是研究大量枯燥的英特尔文档，其中包含干燥的规格和理论上可达到的限制，我们将通过运行许多小型的基准程序并使用类似于实际代码中经常出现的访问模式来实验性地估计这些参数。

### 实验设置

如前所述，我将在 Ryzen 7 4700U 上运行所有实验，这是一款“Zen 2”CPU，以下是其主要缓存相关规格：

+   8 个物理核心（无超线程技术）主频为 2GHz（在提升模式下可达 4.1GHz — 我们已禁用此功能）；

+   256K 的 8 路组相联 L1 数据缓存或每核心 32K；

+   4M 的 8 路组相联 L2 缓存或每核心 512K；

+   8M 的 16 路组相联 L3 缓存，共享于 8 个核心之间；

+   16GB（2x8G）的 DDR4 RAM，频率为 2667MHz。

您可以通过在 Linux 上运行`dmidecode -t cache`或`lshw -class memory`来比较您的硬件，或者在 Windows 上安装[CPU-Z](https://en.wikipedia.org/wiki/CPU-Z)。您还可以在[WikiChip](https://en.wikichip.org/wiki/amd/ryzen_7/4700u)和[7-CPU](https://www.7-cpu.com/cpu/Zen2.html)上找到有关 CPU 的更多详细信息。并非所有结论都适用于现有的所有 CPU 平台。

由于防止编译器优化掉未使用值存在困难，本文中的代码片段为了说明目的而略有简化。如果您想自己重现它们，请查看[代码仓库](https://github.com/sslotin/amh-code/tree/main/cpu-cache)。

### 致谢

本章灵感来源于 Igor Ostrovsky 的“[处理器缓存效果画廊](http://igoro.com/archive/gallery-of-processor-cache-effects/)”和 Ulrich Drepper 的“[程序员应了解的内存知识](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf)”，这两者都可以作为良好的辅助阅读材料。

[← 空间和时间局部性](https://en.algorithmica.org/hpc/external-memory/locality/)

[← 外部内存](https://en.algorithmica.org/hpc/external-memory/)[内存带宽 →](https://en.algorithmica.org/hpc/cpu-cache/bandwidth/)

[../SIMD 并行处理 →](https://en.algorithmica.org/hpc/simd/)
