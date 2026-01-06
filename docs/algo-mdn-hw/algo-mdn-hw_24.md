# 对于排序算法，我们想知道它进行了多少次比较。

> 译文：[`en.algorithmica.org/hpc/profiling/instrumentation/`](https://en.algorithmica.org/hpc/profiling/instrumentation/)

对于一个哈希函数，我们对其输入的平均长度感兴趣；

为了达到更高的精度，你可以在循环中重复调用函数，只计时一次，然后将总时间除以迭代次数：

```cpp
clock_t start = clock(); do_something(); float seconds = float(clock() - start) / CLOCKS_PER_SEC; printf("do_something() took %.4f", seconds); 
```

这里有一个细微差别，那就是你不能用这种方法来测量特别快的函数的执行时间，因为`clock`函数返回当前时间戳（以微秒为单位，$10^{-6}$），并且它本身完成也需要几百纳秒。所有其他与时间相关的实用工具都具有至少微秒级的粒度，这在底层优化的世界中是永恒的。

### [#](https://en.algorithmica.org/hpc/profiling/instrumentation/#event-sampling)事件抽样

```cpp
#include <stdio.h> #include <time.h>  const int N = 1e6;   int main() {  clock_t start = clock();   for (int i = 0; i < N; i++) clock(); // benchmarking the clock function itself  float duration = float(clock() - start) / CLOCKS_PER_SEC; printf("%.2fns per iteration\n", 1e9 * duration / N);   return 0; } 
```

*仪器*是一个过于复杂的术语，意味着将计时器和其他跟踪代码插入到程序中。最简单的例子是在类 Unix 系统中使用`time`实用工具来测量整个程序的执行持续时间。

原文：[`en.algorithmica.org/hpc/profiling/instrumentation/`](https://en.algorithmica.org/hpc/profiling/instrumentation/)

仪器也可以用来收集其他类型的信息，这些信息可以提供关于特定算法性能的有用见解。例如：

+   对于一个二叉树，我们关心其大小和高度；

+   更一般地，我们想知道程序中哪些部分需要优化。编译器和 IDEs 附带了一些工具，可以自动计时指定的函数，但通过手动使用语言提供的任何与时间交互的方法来做会更稳健：

+   添加计数器有一个缺点，那就是它会引入开销，尽管你可以通过只对一小部分调用进行随机操作来几乎完全缓解这个问题：

如果采样率足够小，每次调用剩余的唯一开销将是随机数生成和条件检查。有趣的是，我们可以通过一些统计技巧进一步优化它。

从数学上讲，我们在这里所做的就是反复从[伯努利分布](https://en.wikipedia.org/wiki/Bernoulli_distribution)（$p$等于采样率）中进行抽样，直到我们得到一个成功。还有一个分布可以告诉我们需要多少次伯努利抽样迭代才能得到第一个正值，称为[几何分布](https://en.wikipedia.org/wiki/Geometric_distribution)。我们可以从它中进行抽样，并使用该值作为递减计数器：

```cpp
void query() {  if (rand() % 100 == 0) { // update statistics } // main logic } 
```

你还需要确保没有任何东西被缓存、被编译器优化掉或受到类似副作用的影响。这是一个单独且非常复杂的话题，我们将在本章末尾进行更详细的讨论。

以类似的方式，我们可以在代码中插入计数器来计算这些特定算法的统计数据。

```cpp
void query() {  static next_sample = geometric_distribution(sample_rate); if (next_sample--) { next_sample = geometric_distribution(sample_rate); // ... } // ... } 
```

这样我们就可以消除每次调用时都需要采样新随机数的需要，只有在选择计算统计数据时才重置计数器。

这种技术通常被大型项目中的库算法开发者频繁使用，以收集性能分析数据，同时尽量不影响最终程序的性能。[← ../性能分析](https://en.algorithmica.org/hpc/profiling/)[统计性能分析 →](https://en.algorithmica.org/hpc/profiling/events/)
