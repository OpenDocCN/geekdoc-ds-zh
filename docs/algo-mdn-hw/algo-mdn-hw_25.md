# 统计配置

> 原文：[`en.algorithmica.org/hpc/profiling/events/`](https://en.algorithmica.org/hpc/profiling/events/)

配置是一种相当繁琐的配置方法，尤其是如果你对程序中的多个小部分感兴趣。即使可以通过工具部分自动化，它仍然无法帮助你收集一些细粒度的统计数据，因为它固有的开销。

另一种对程序进行配置的较少侵入性方法是随机地在程序执行过程中中断其执行，并查看指令指针的位置。指针在每个函数块中停止的次数将与执行这些函数所花费的总时间大致成比例。通过这种方式，你还可以获取一些其他有用的信息，例如通过检查调用栈来找出哪些函数被哪些函数调用。

这原则上可以通过仅用 `gdb` 和 `ctrl+c` 在随机间隔停止程序来完成，但现代 CPU 和操作系统提供了用于此类配置的特殊实用程序。

### [#](https://en.algorithmica.org/hpc/profiling/events/#hardware-events)硬件事件

硬件 *性能计数器* 是集成到微处理器中的特殊寄存器，可以存储某些与硬件相关的活动的计数。由于它们基本上只是连接有激活线的二进制计数器，因此它们在微芯片上添加成本低廉。

每个性能计数器都连接到电路的一个大子集，并且可以配置为在特定的硬件事件上增加，例如分支预测错误或缓存未命中。你可以在程序开始时重置计数器，运行程序，并在结束时输出其存储的值，它将等于在整个执行过程中触发某个事件的精确次数。

你还可以通过在它们之间进行多路复用来跟踪多个事件，也就是说，在偶数间隔停止程序并重新配置计数器。在这种情况下，结果将不会是精确的，而是一个统计近似。这里的一个细微差别是，仅仅通过增加采样频率来提高其准确性是不可行的，因为这会过多地影响性能，从而扭曲分布，因此为了收集多个统计数据，你需要运行程序更长的时间。

总体而言，事件驱动的统计配置通常是诊断性能问题的最有效和最简单的方法。

### [#](https://en.algorithmica.org/hpc/profiling/events/#profiling-with-perf)使用 perf 进行配置

依赖于上述描述的事件采样技术的性能分析工具被称为 *统计分析器*。有很多这样的工具，但在这本书中我们将主要使用的是 [perf](https://perf.wiki.kernel.org/)，它是随 Linux 内核一起提供的统计分析器。在非 Linux 系统上，您可以使用英特尔提供的 [VTune](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/vtune-profiler.html#gs.cuc0ks)，它为我们提供了类似的功能。虽然它是专有软件，但可以免费使用，并且您需要每 90 天更新一次社区许可证，而 perf 则是自由免费的。

Perf 是一个基于命令行的应用程序，它根据程序的实时执行生成报告。它不需要源代码，并且可以分析非常广泛的应用程序，包括涉及多个进程和与操作系统交互的应用程序。

为了解释的目的，我编写了一个小程序，它创建了一个包含一百万个随机整数的数组，对其进行排序，然后在上面执行一百万次二分查找：

```cpp
void setup() {
    for (int i = 0; i < n; i++)
        a[i] = rand();
    std::sort(a, a + n);
}

int query() {
    int checksum = 0;
    for (int i = 0; i < n; i++) {
        int idx = std::lower_bound(a, a + n, rand()) - a;
        checksum += idx;
    }
    return checksum;
} 
```

编译完成后（`g++ -O3 -march=native example.cc -o run`），我们可以使用 `perf stat ./run` 来运行它，该命令会在执行过程中输出基本性能事件的计数：

```cpp
 Performance counter stats for './run':

        646.07 msec task-clock:u               # 0.997 CPUs utilized          
             0      context-switches:u         # 0.000 K/sec                  
             0      cpu-migrations:u           # 0.000 K/sec                  
         1,096      page-faults:u              # 0.002 M/sec                  
   852,125,255      cycles:u                   # 1.319 GHz (83.35%)
    28,475,954      stalled-cycles-frontend:u  # 3.34% frontend cycles idle (83.30%)
    10,460,937      stalled-cycles-backend:u   # 1.23% backend cycles idle (83.28%)
   479,175,388      instructions:u             # 0.56  insn per cycle         
                                               # 0.06  stalled cycles per insn (83.28%)
   122,705,572      branches:u                 # 189.925 M/sec (83.32%)
    19,229,451      branch-misses:u            # 15.67% of all branches (83.47%)

   0.647801770 seconds time elapsed
   0.647278000 seconds user
   0.000000000 seconds sys 
```

您可以看到，执行耗时 0.53 秒或 852M 个周期，在有效 1.32 GHz 的时钟频率下，执行了 479M 条指令。还有 122.7M 条分支，其中 15.7% 被误判。

您可以使用 `perf list` 获取所有支持的事件列表，然后使用 `-e` 选项指定您想要的事件列表。例如，为了诊断二分查找，我们主要关心缓存未命中：

```cpp
> perf stat -e cache-references,cache-misses ./run

91,002,054      cache-references:u                                          
44,991,746      cache-misses:u      # 49.440 % of all cache refs 
```

单独的 `perf stat` 只是为整个程序设置性能计数器。它可以告诉你分支误判的总数，但不会告诉你它们发生在哪里，更不用说为什么会发生。

要尝试我们之前讨论的停止世界方法，我们需要使用 `perf record <cmd>`，它记录分析数据并将其作为 `perf.data` 文件导出，然后调用 `perf report` 来检查它。我强烈建议您亲自尝试，因为最后一个命令是交互式的，并且具有丰富的色彩，但对于那些现在无法做到的人，我会尽力描述它。

当您调用 `perf report` 时，它首先显示一个类似于 `top` 的交互式报告，告诉您哪些函数花费了多长时间：

```cpp
Overhead  Command  Shared Object        Symbol
  63.08%  run      run                  [.] query
  24.98%  run      run                  [.] std::__introsort_loop<...>
   5.02%  run      libc-2.33.so         [.] __random
   3.43%  run      run                  [.] setup
   1.95%  run      libc-2.33.so         [.] __random_r
   0.80%  run      libc-2.33.so         [.] rand 
```

注意，对于每个函数，只列出了其**开销**，而不是总运行时间（例如，`setup`包括`std::__introsort_loop`，但只计算其自身的开销，为 3.43%）。有工具可以将 perf 报告转换为[火焰图](https://www.brendangregg.com/flamegraphs.html)，使其更清晰。你还需要考虑可能的内联，显然这里发生的就是`std::lower_bound`的内联。Perf 还跟踪共享库（如`libc`）以及通常任何其他派生的进程：如果你愿意，可以用 perf 启动一个网页浏览器，看看里面发生了什么。

接下来，你可以“放大”查看这些函数中的任何一个，并且除了其他功能之外，它还会提供显示其与相关热图相关的反汇编代码。例如，这里是`query`的反汇编代码：

```cpp
 │20: → call   rand@plt
       │      mov    %r12,%rsi
       │      mov    %eax,%edi
       │      mov    $0xf4240,%eax
       │      nop    
       │30:   test   %rax,%rax
  4.57 │    ↓ jle    52
       │35:   mov    %rax,%rdx
  0.52 │      sar    %rdx
  0.33 │      lea    (%rsi,%rdx,4),%rcx
  4.30 │      cmp    (%rcx),%edi
 65.39 │    ↓ jle    b0
  0.07 │      sub    %rdx,%rax
  9.32 │      lea    0x4(%rcx),%rsi
  0.06 │      dec    %rax
  1.37 │      test   %rax,%rax
  1.11 │    ↑ jg     35
       │52:   sub    %r12,%rsi
  2.22 │      sar    $0x2,%rsi
  0.33 │      add    %esi,%ebp
  0.20 │      dec    %ebx
       │    ↑ jne    20 
```

在左侧列中是指令指针在特定行上停止的频率。你可以看到我们大约 65%的时间花在了跳转指令上，因为它前面有一个比较运算符，这表明控制流在这里等待这个比较结果。

由于如流水线和乱序执行等复杂性，“现在”在现代 CPU 中不是一个定义良好的概念，因此数据略有不准确，因为指令指针稍微向前漂移。指令级数据仍然有用，但在单个周期级别，我们需要切换到更精确的方法。[← 仪器](https://en.algorithmica.org/hpc/profiling/instrumentation/)[程序模拟 →](https://en.algorithmica.org/hpc/profiling/simulation/)
