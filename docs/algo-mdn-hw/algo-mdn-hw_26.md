# 程序模拟

> 原文：[`en.algorithmica.org/hpc/profiling/simulation/`](https://en.algorithmica.org/hpc/profiling/simulation/)

分析的最后一个方法（或者更确切地说，是一组方法）不是通过实际运行程序来收集数据，而是通过使用专用工具来模拟应该发生的情况。

这样的分析器有很多子类别，它们在模拟计算哪个方面有所不同。在这篇文章中，我们将重点关注 缓存 和 分支预测，并使用 [Cachegrind](https://valgrind.org/docs/manual/cg-manual.html) 来实现这一点，它是 [Valgrind](https://valgrind.org/) 的一部分，Valgrind 是一个用于内存泄漏检测和一般内存调试的成熟工具。

### [#](https://en.algorithmica.org/hpc/profiling/simulation/#profiling-with-cachegrind) 使用 Cachegrind 进行分析

Cachegrind 实质上检查二进制文件中的“有趣”指令——执行内存读取/写入和条件/间接跳转的指令——并用代码替换它们，这些代码使用软件数据结构模拟相应的硬件操作。因此，它不需要访问源代码，可以与已编译的程序一起工作，并且可以在任何这样的程序上运行：

```cpp
valgrind --tool=cachegrind --branch-sim=yes ./run #       also simulate branch prediction ^   ^ any command, not necessarily one process 
```

它对所有的二进制文件进行测量，运行它们，并输出一个类似于 perf stat 的摘要：

```cpp
I   refs:      483,664,426
I1  misses:          1,858
LLi misses:          1,788
I1  miss rate:        0.00%
LLi miss rate:        0.00%

D   refs:      115,204,359  (88,016,970 rd   + 27,187,389 wr)
D1  misses:      9,722,664  ( 9,656,463 rd   +     66,201 wr)
LLd misses:         72,587  (     8,496 rd   +     64,091 wr)
D1  miss rate:         8.4% (      11.0%     +        0.2%  )
LLd miss rate:         0.1% (       0.0%     +        0.2%  )

LL refs:         9,724,522  ( 9,658,321 rd   +     66,201 wr)
LL misses:          74,375  (    10,284 rd   +     64,091 wr)
LL miss rate:          0.0% (       0.0%     +        0.2%  )

Branches:       90,575,071  (88,569,738 cond +  2,005,333 ind)
Mispredicts:    19,922,564  (19,921,919 cond +        645 ind)
Mispred rate:         22.0% (      22.5%     +        0.0%   ) 
```

我们向 Cachegrind 提供了与 上一节 中完全相同的示例代码：我们创建了一个包含一百万个随机整数的数组，对其进行排序，然后在上面执行一百万次二分搜索。Cachegrind 显示的数字与 perf 显示的数字大致相同，但 perf 测量的内存读取和分支的数量略有增加，这是由于 推测执行 的原因：它们实际上在硬件中发生，并因此增加硬件计数器，但被丢弃并不影响实际性能，因此在模拟中被忽略。

Cachegrind 只模拟缓存的第一级（`D1` 表示数据，`I1` 表示指令）和最后一级（`LL`，统一），其特性是从系统中推断出来的。它不会以任何方式限制你，因为你也可以从命令行设置它们，例如，模拟 L2 缓存：`--LL=<size>,<associativity>,<line size>`。

它似乎只是让我们的程序变慢了，并没有提供任何 `perf stat` 无法提供的信息。为了从它那里获得比仅仅摘要信息更多的信息，我们可以检查一个包含分析信息的特殊文件，它默认以 `cachegrind.out.<pid>` 的名称在相同的目录下导出。它是可读的，但通常通过 `cg_annotate` 命令来读取：

```cpp
cg_annotate cachegrind.out.4159404 --show=Dr,D1mr,DLmr,Bc,Bcm #                                    ^ we are only interested in data reads and branches 
```

首先它显示了运行过程中使用的参数，包括缓存系统的特性：

```cpp
I1 cache:         32768 B, 64 B, 8-way associative
D1 cache:         32768 B, 64 B, 8-way associative
LL cache:         8388608 B, 64 B, direct-mapped 
```

它对 L3 缓存的模拟并不完全准确：它不是统一的（总共 8M，但单个核心只能看到 4M）并且也是 16 路关联的，但我们将暂时忽略这一点。

接下来，它输出一个类似于`perf report`的每个函数摘要：

```cpp
Dr         D1mr      DLmr Bc         Bcm         file:function
--------------------------------------------------------------------------------
19,951,476 8,985,458    3 41,902,938 11,005,530  ???:query()
24,832,125   585,982   65 24,712,356  7,689,480  ???:void std::__introsort_loop<...>
16,000,000        60    3  9,935,484    129,044  ???:random_r
18,000,000         2    1  6,000,000          1  ???:random
 4,690,248    61,999   17  5,690,241  1,081,230  ???:setup()
 2,000,000         0    0          0          0  ???:rand 
```

你可以看到在排序阶段有很多分支预测错误，而且在二分搜索期间也有很多 L1 缓存未命中和分支预测错误。我们无法通过 perf 获取这些信息——它只会告诉我们整个程序的这些计数。

Cachegrind 另一个很棒的功能是对源代码的逐行注释。为此，你需要用调试信息（`-g`）编译程序，并明确告诉`cg_annotate`要注释哪些源文件，或者只需传递`--auto=yes`选项，这样它就会注释它能到达的所有内容（包括标准库源代码）。

整个源代码到分析的过程因此会是这样的：

```cpp
g++ -O3 -g sort-and-search.cc -o run valgrind --tool=cachegrind --branch-sim=yes --cachegrind-out-file=cachegrind.out ./run cg_annotate cachegrind.out --auto=yes --show=Dr,D1mr,DLmr,Bc,Bcm 
```

由于 glibc 的实现不是最易读的，为了说明目的，我们用我们自己的二分搜索替换`lower_bound`，它将被这样注释：

```cpp
Dr         D1mr      DLmr Bc         Bcm  .         .    .          .         .  int binary_search(int x) { 0         0    0          0         0      int l = 0, r = n - 1; 0         0    0 20,951,468 1,031,609      while (l < r) { 0         0    0          0         0          int m = (l + r) / 2; 19,951,468 8,991,917   63 19,951,468 9,973,904          if (a[m] >= x)  .         .    .          .         .              r = m; .         .    .          .         .          else 0         0    0          0         0              l = m + 1; .         .    .          .         .      } .         .    .          .         .      return l; .         .    .          .         .  } 
```

不幸的是，Cachegrind 只跟踪内存访问和分支。当瓶颈是由其他因素引起时，我们需要其他模拟工具。[← 统计分析](https://en.algorithmica.org/hpc/profiling/events/)[机器代码分析器](https://en.algorithmica.org/hpc/profiling/mca/)
