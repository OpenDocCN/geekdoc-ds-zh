# 情境优化

> 原文：[`en.algorithmica.org/hpc/compilation/situational/`](https://en.algorithmica.org/hpc/compilation/situational/)

通过`-O2`和`-O3`启用的大多数编译器优化都保证要么改进性能，至少不会严重损害性能。那些不包括在`-O3`中的优化要么不是严格符合标准的，或者非常具体，需要程序员提供一些额外的输入来帮助决定使用它们是否有益。

让我们讨论一下这本书中我们已经多次提到的最常用的那些。

### [#](https://en.algorithmica.org/hpc/compilation/situational/#loop-unrolling)循环展开

循环展开默认是禁用的，除非循环在编译时已知迭代次数是一个小的常数——在这种情况下，它将被替换为一个完全无跳转的、重复的指令序列。可以通过`-funroll-loops`标志全局启用，这将展开所有在编译时或在循环入口时可以确定迭代次数的循环。

你也可以使用一个 pragma 来针对特定的循环：

```cpp
#pragma GCC unroll 4 for (int i = 0; i < n; i++) {  // ... } 
```

循环展开会使二进制文件更大，并且可能或可能不会使其运行更快。不要狂热地使用它。

### [#](https://en.algorithmica.org/hpc/compilation/situational/#function-inlining)函数内联

内联最好留给编译器来决定，但你可以通过`inline`关键字来影响它：

```cpp
inline int square(int x) {  return x * x; } 
```

如果编译器认为潜在的性能提升不值得，可能会忽略这个提示。你可以通过添加`always_inline`属性来强制内联：

```cpp
#define FORCE_INLINE inline __attribute__((always_inline)) 
```

还有`-finline-limit=n`选项，它允许你设置内联函数大小的特定阈值（以指令数量计）。它的 Clang 等价项是`-inline-threshold`。

### [#](https://en.algorithmica.org/hpc/compilation/situational/#likeliness-of-branches)分支的可能性

分支的可能性可以通过`[[likely]]`和`[[unlikely]]`属性在`if`和`switch`中提示：

```cpp
int factorial(int n) {  if (n > 1) [[likely]] return n * factorial(n - 1); else [[unlikely]] return 1; } 
```

这是一个只在 C++20 中出现的全新功能。在此之前，有编译器特定的内建函数被用来封装条件表达式。在较老的 GCC 中的相同示例：

```cpp
int factorial(int n) {  if (__builtin_expect(n > 1, 1)) return n * factorial(n - 1); else return 1; } 
```

当你需要指导编译器走向正确的方向时，还有很多其他类似的情况，但我们将稍后当它们变得更加相关时再讨论。

### [#](https://en.algorithmica.org/hpc/compilation/situational/#profile-guided-optimization)基于配置文件优化

将所有这些元数据添加到源代码中是繁琐的。人们已经讨厌编写 C++，即使不需要这样做。

并不是总是很明显某些优化是有益的还是无益的。为了决定分支重排、函数内联或循环展开，我们需要回答如下问题：

+   这个分支被取用的频率是多少？

+   这个函数被调用的频率是多少？

+   这个循环的平均迭代次数是多少？

幸运的是，我们有一种方法可以自动提供这种现实世界的信息。

*基于性能分析的优化*（PGO，也称为“pogo”，因为它更容易发音且更有趣）是一种使用 性能分析数据 来提高性能的技术，这种性能是仅通过静态分析无法实现的。简单来说，它包括在程序中感兴趣的地方添加计时器和计数器，使用真实数据编译并运行程序，然后再次编译，但这次提供测试运行中的额外信息。

整个过程都由现代编译器自动化完成。例如，`-fprofile-generate` 标志会让 GCC 在程序中插入性能分析代码：

```cpp
g++ -fprofile-generate [other flags] source.cc -o binary 
```

在我们运行程序之后——最好是使用尽可能代表实际用例的输入——它将创建一些 `*.gcda` 文件，这些文件包含测试运行的日志数据，之后我们可以重新构建程序，但现在添加 `-fprofile-use` 标志：

```cpp
g++ -fprofile-use [other flags] source.cc -o binary 
```

它通常可以将大型代码库的性能提高 10-20%，因此它通常被包含在性能关键项目的构建过程中。这是投资于可靠的基准测试代码的另一个原因。[← 标志和目标](https://en.algorithmica.org/hpc/compilation/flags/)[契约编程 →](https://en.algorithmica.org/hpc/compilation/contracts/)
