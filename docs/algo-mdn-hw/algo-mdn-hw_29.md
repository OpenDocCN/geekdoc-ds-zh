# 获取准确结果

> 原文：[`en.algorithmica.org/hpc/profiling/noise/`](https://en.algorithmica.org/hpc/profiling/noise/)

有两个库算法实现并不罕见，每个都维护自己的基准测试代码，并声称比对方更快。这会让所有相关人员感到困惑，尤其是用户，他们必须在这两个之间做出某种选择。

这种情况通常不是由其作者的欺诈行为引起的；他们只是对“更快”的定义不同，而且确实，定义和使用单一的性能指标通常非常有问题。

### [#](https://en.algorithmica.org/hpc/profiling/noise/#measuring-the-right-thing)正确测量事物

有许多事情可能会引入基准的偏差。

**不同的数据集。**有许多算法的性能在某些方面依赖于数据集的分布。为了定义，例如，最快的排序、最短路径或二分搜索算法是什么，你必须固定算法运行的数据集。

这有时甚至适用于处理单个输入的算法。例如，给 GCD 实现提供连续数字不是一个好主意，因为它使得分支非常可预测：

```cpp
// don't do this int checksum = 0;   for (int a = 0; a < 1000; a++)  for (int b = 0; b < 1000; b++) checksum ^= gcd(a, b); 
```

然而，如果我们随机采样这些相同的数字，分支预测变得困难得多，尽管处理相同的输入，但顺序改变后，基准测试需要更长的时间：

```cpp
int a[1000], b[1000];   for (int i = 0; i < 1000; i++)  a[i] = rand() % 1000, b[i] = rand() % 1000;   int checksum = 0;   for (int t = 0; t < 1000; t++)  for (int i = 0; i < 1000; i++) checksum += gcd(a[i], b[i]); 
```

尽管对于大多数情况来说，最合理的做法是随机均匀地采样数据，但许多现实世界的应用具有远非均匀的分布，因此不能只选择一个。一般来说，一个好的基准应该针对特定应用，并尽可能使用代表你实际用例的数据集。

**多个目标。**一些算法设计问题有多个关键目标。例如，除了高度依赖于键的分布外，哈希表还需要仔细平衡：

+   内存使用，

+   添加查询的延迟，

+   正成员查询的延迟，

+   负成员查询的延迟。

选择哈希表实现的方法是尝试将多个变体放入应用程序中。

**延迟与吞吐量。**人们常常忽视的另一个方面是，执行时间可以以多种方式定义，即使是针对单个查询。

当你编写如下代码时：

```cpp
for (int i = 0; i < N; i++)  q[i] = rand();   int checksum = 0;   for (int i = 0; i < N; i++)  checksum ^= lower_bound(q[i]); 
```

然后测量整个过程并除以迭代次数，你实际上是在测量查询的*吞吐量*——它在单位时间内可以处理多少操作。由于交错，这通常小于单独处理一个操作所需的时间。

为了测量实际的*延迟*，你需要引入调用之间的依赖关系：

```cpp
for (int i = 0; i < N; i++)  checksum ^= lower_bound(checksum ^ q[i]); 
```

这通常在可能存在流水线停滞问题的算法中影响最大，例如，在比较有分支和无分支算法时。

**冷缓存。**另一个偏差来源是**冷缓存效应**，当所需数据尚未在缓存中时，内存读取最初会花费更长的时间。

这可以通过在开始测量之前进行一次**预热运行**来解决：

```cpp
// warm-up run  volatile checksum = 0;   for (int i = 0; i < N; i++)  checksum ^= lower_bound(q[i]);     // actual run  clock_t start = clock(); checksum = 0;   for (int i = 0; i < N; i++)  checksum ^= lower_bound(q[i]); 
```

如果答案验证比仅仅计算某种校验和更复杂，有时将预热运行与答案验证结合起来也很方便。

**过度优化。**有时基准测试完全是错误的，因为编译器只是优化了被基准测试的代码。为了防止编译器走捷径，你需要添加校验和，或者将其打印到某个地方，或者添加`volatile`限定符，这也会防止任何类型的循环迭代交错。

对于只写入数据的算法，你可以使用`__sync_synchronize()`内建函数来添加内存栅栏，并防止编译器累积更新。

### [#](https://en.algorithmica.org/hpc/profiling/noise/#reducing-noise) 减少噪声

我们所描述的问题会在测量中产生**偏差**：它们始终使一个算法相对于另一个算法具有优势。基准测试中可能存在其他类型的问题，导致不可预测的偏差或完全随机的噪声，从而增加**方差**。

这些类型的问题是由副作用和某种外部噪声引起的，主要由于噪声邻居和 CPU 频率缩放：

+   如果你基准测试一个计算密集型算法，使用`perf stat`来测量其性能（以周期为单位）：这样它将独立于时钟频率，而时钟频率的波动通常是噪声的主要来源。

+   否则，将核心频率设置为预期的值，并确保没有任何东西干扰它。在 Linux 上，你可以使用`cpupower`（例如，使用`sudo cpupower frequency-set -g powersave`将其设置为最小或使用`sudo cpupower frequency-set -g ondemand`来启用超频）。我使用一个[方便的 GNOME shell 扩展](https://extensions.gnome.org/extension/1082/cpufreq/)，它有一个单独的按钮来执行此操作。

+   如果适用，关闭超线程并将作业附加到特定核心。确保系统上没有其他作业运行，关闭网络，并尽量避免摆弄鼠标。

你无法完全消除噪声和偏差。甚至一个程序的名字也可能影响其速度：可执行文件的名字最终会出现在环境变量中，环境变量最终会出现在调用栈上，因此名字的长度会影响栈对齐，这可能导致由于跨越缓存行或内存页面边界而减慢数据访问速度。

在指导优化和尤其是向他人报告结果时，考虑噪声是很重要的。除非你期待有 2 倍以上的改进，否则对待所有微基准测试的方式应与 A/B 测试相同。

当您在一台笔记本电脑上运行程序不到一秒钟时，性能的±5%波动是完全正常的。因此，如果您想决定是否撤销或保留潜在的+1%改进，请运行它直到达到统计显著性，这可以通过计算方差和 p 值来确定。

### [#](https://en.algorithmica.org/hpc/profiling/noise/#further-reading)进一步阅读

感兴趣的读者可以探索 Dror Feitelson 编制的这个全面的[实验计算机科学资源列表](https://www.cs.huji.ac.il/w~feit/exp/related.html)，也许可以从 Todd Mytkowicz 等人的“[在不做任何明显错误的情况下产生错误数据](http://eecs.northwestern.edu/~robby/courses/322-2013-spring/mytkowicz-wrong-data.pdf)”开始。

您还可以观看 Emery Berger 关于如何进行统计性能评估的[这场精彩演讲](https://www.youtube.com/watch?v=r-TLSBdHe1A)。[← 基准测试](https://en.algorithmica.org/hpc/profiling/benchmarking/)[../算术 →](https://en.algorithmica.org/hpc/arithmetic/)
