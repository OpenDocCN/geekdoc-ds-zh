# 基准测试

> 原文：[`en.algorithmica.org/hpc/profiling/benchmarking/`](https://en.algorithmica.org/hpc/profiling/benchmarking/)

大多数优秀的软件工程实践以某种方式解决了加快*开发周期*的问题：你希望编译软件更快（构建系统）、尽快捕捉到错误（静态分析、持续集成）、一旦新版本准备好就发布（持续部署），以及尽可能少地延迟地响应用户反馈（敏捷开发）。

性能工程也不例外。如果你做得正确，它也应该类似于一个循环：

1.  运行程序并收集指标。

1.  确定瓶颈在哪里。

1.  移除瓶颈并回到步骤 1。

在本节中，我们将讨论基准测试，并讨论一些实用的技术，这些技术可以使这个循环更短，并帮助你更快地迭代。大部分建议都来自本书的编写工作，因此你可以在本书的[代码仓库](https://github.com/sslotin/ahm-code)中找到许多描述的设置的实例。

### [#](https://en.algorithmica.org/hpc/profiling/benchmarking/#benchmarking-inside-c)C++ 内部基准测试

存在几种编写基准测试代码的方法。可能最流行的一种是将你想要比较的几个相同语言的实现包含在一个文件中，从 `main` 函数中单独调用它们，并在同一源文件中计算你想要的全部指标。

这种方法的缺点是，你需要编写大量的样板代码，并且为每个实现重复它，但可以通过元编程部分地抵消这一点。例如，当你正在基准测试多个 gcd 实现时，你可以使用这个高阶函数显著减少基准测试代码：

```cpp
const int N = 1e6, T = 1e9 / N; int a[N], b[N];   void timeit(int (*f)(int, int)) {  clock_t start = clock();   int checksum = 0;   for (int t = 0; t < T; t++) for (int i = 0; i < n; i++) checksum ^= f(a[i], b[i]);  float seconds = float(clock() - start) / CLOCKS_PER_SEC;   printf("checksum: %d\n", checksum); printf("%.2f ns per call\n", 1e9 * seconds / N / T); }   int main() {  for (int i = 0; i < N; i++) a[i] = rand(), b[i] = rand();  timeit(std::gcd); timeit(my_gcd); timeit(my_another_gcd); // ...  return 0; } 
```

这是一个低开销的方法，让你能够运行更多的实验，并从它们中获得更准确的结果。../noise。你仍然需要执行一些重复操作，但它们可以通过框架在很大程度上自动化，其中最受欢迎的选择是 C++ 的 [Google 基准库](https://github.com/google/benchmark)。一些编程语言也有方便的内置基准测试工具：在此特别提一下 [Python 的 timeit 函数](https://docs.python.org/3/library/timeit.html)和 [Julia 的 @benchmark 宏](https://github.com/JuliaCI/BenchmarkTools.jl)。

尽管在执行速度方面*高效*，但 C 和 C++ 并不是最*高效*的语言，尤其是在分析方面。当你的算法依赖于某些参数，如输入大小，并且你需要从每个实现中收集不止一个数据点时，你真的希望将你的基准测试代码与外部环境集成，并使用其他工具来分析结果。

### [#](https://en.algorithmica.org/hpc/profiling/benchmarking/#splitting-up-implementations)拆分实现

提高模块化和可重用性的一个方法是将所有测试和统计分析代码与算法的实际实现分开，并且确保不同版本在单独的文件中实现，但具有相同的接口。

在 C/C++中，你可以通过创建一个包含函数接口及其所有基准测试代码的`main`中的单个头文件（例如，`gcd.hh`）来实现这一点：

```cpp
int gcd(int a, int b); // to be implemented  // for data structures, you also need to create a setup function // (unless the same preprocessing step for all versions would suffice)  int main() {  const int N = 1e6, T = 1e9 / N; int a[N], b[N]; // careful: local arrays are allocated on the stack and may cause stack overflow // for large arrays, allocate with "new" or create a global array  for (int i = 0; i < N; i++) a[i] = rand(), b[i] = rand();   int checksum = 0;   clock_t start = clock();   for (int t = 0; t < T; t++) for (int i = 0; i < n; i++) checksum += gcd(a[i], b[i]);  float seconds = float(clock() - start) / CLOCKS_PER_SEC;   printf("%d\n", checksum); printf("%.2f ns per call\n", 1e9 * seconds / N / T);  return 0; } 
```

然后，为每个算法版本创建许多实现文件（例如，`v1.cc`、`v2.cc`等，或者如果适用，一些有意义的名称），它们都包含那个单独的头文件：

```cpp
#include "gcd.hh"  int gcd(int a, int b) {  if (b == 0) return a; else return gcd(b, a % b); } 
```

做这件事的整个目的是能够从命令行中测试特定的算法版本，而不需要触摸任何源代码文件。为此，你可能还希望公开它可能具有的任何参数——例如，通过解析命令行参数：

```cpp
int main(int argc, char* argv[]) {  int N = (argc > 1 ? atoi(argv[1]) : 1e6); const int T = 1e9 / N;   // ... } 
```

另一种方法是使用 C 风格的全局定义，然后在编译时通过`-D N=...`标志传递它们：

```cpp
#ifndef N #define N 1000000 #endif  const int T = 1e9 / N; 
```

这样你可以利用编译时的常量，这可能对某些算法的性能非常有帮助，但代价是每次你想更改参数时都必须重新构建程序，这会显著增加你在一系列参数值上收集指标所需的时间。

### [#](https://en.algorithmica.org/hpc/profiling/benchmarking/#makefiles)Makefiles

将源文件拆分可以让你使用像[Make](https://en.wikipedia.org/wiki/Make_(software))这样的缓存构建系统来加速编译。

我通常在我的项目中携带这个 Makefile 的版本：

```cpp
compile = g++ -std=c++17 -O3 -march=native -Wall   %: %.cc gcd.hh  $(compile) $< -o $@   %.s: %.cc gcd.hh  $(compile) -S -fverbose-asm $< -o $@   %.run: %  @./$<   .PHONY: %.run 
```

现在，你可以使用`make example`来编译`example.cc`，并自动运行它使用`make example.run`。

你也可以在 Makefile 中添加用于计算统计的脚本，或者与`perf stat`调用结合使用，使分析自动化。

### [#](https://en.algorithmica.org/hpc/profiling/benchmarking/#jupyter-notebooks)Jupyter Notebooks

为了加快高级分析的速度，你可以创建一个 Jupyter 笔记本，在其中放置所有脚本并执行所有绘图。

添加一个用于基准测试实现的包装器很方便，它只返回一个标量结果：

```cpp
def bench(source, n=2**20):  !make -s {source} if _exit_code != 0: raise Exception("Compilation failed") res = !./{source} {n} {q} duration = float(res[0].split()[0]) return duration 
```

然后你可以用它来编写干净的统计分析代码：

```cpp
ns = list(int(1.17**k) for k in range(30, 60)) baseline = [bench('std_lower_bound', n=n) for n in ns] results = [bench('my_binary_search', n=n) for n in ns]   # plotting relative speedup for different array sizes import matplotlib.pyplot as plt   plt.plot(ns, [x / y for x, y in zip(baseline, results)]) plt.show() 
```

一旦建立，这个工作流程会让你迭代得更快，并专注于优化算法本身。[← 机器代码分析器](https://en.algorithmica.org/hpc/profiling/mca/)[获取准确结果 →](https://en.algorithmica.org/hpc/profiling/noise/)
