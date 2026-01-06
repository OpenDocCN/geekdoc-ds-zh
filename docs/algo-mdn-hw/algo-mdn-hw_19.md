# 标志和目标

> 原文：[`en.algorithmica.org/hpc/compilation/flags/`](https://en.algorithmica.org/hpc/compilation/flags/)

从编译器获得高性能的第一步是请求它，这通过超过一百种不同的编译器选项、属性和指令来完成。

### [#](https://en.algorithmica.org/hpc/compilation/flags/#optimization-levels) 优化级别

GCC 中有 4 个半主要优化级别用于提高速度：

+   `-O0` 是默认选项，它不进行任何优化（尽管从某种意义上说，它确实进行了优化：优化编译时间）。

+   `-O1`（也称为 `-O`）执行一些“低垂的果实”优化，几乎不影响编译时间。

+   `-O2` 启用所有已知具有很少或没有负面影响的优化，并且完成时间合理（这是大多数项目用于生产构建的优化级别）。

+   `-O3` 执行非常激进的优化，几乎启用了 GCC 中实现的几乎所有*正确*的优化。

+   `-Ofast` 执行了 `-O3` 所有的操作，并额外包含了一些可能会破坏严格标准兼容性的优化标志，但这种方式不会对大多数应用造成关键性的影响（例如，浮点运算可能会重新排列，导致尾数偏移几个位）。

还有许多其他优化标志，甚至不在 `-Ofast` 中，因为它们非常特定，默认启用它们更有可能损害性能而不是提高性能——我们将在下一节中讨论其中的一些。

### [#](https://en.algorithmica.org/hpc/compilation/flags/#specifying-targets) 指定目标

下一步你可能想要做的是告诉编译器更多关于此代码预期在哪些计算机上运行的信息：平台集合越小，越好。默认情况下，它将生成可以在任何相对较新的（>2000）x86 CPU 上运行的二进制文件。最简单的方法是通过传递 `-march` 标志来指定确切的微架构：`-march=haswell`。如果你在将运行二进制文件的同一台计算机上编译，可以使用 `-march=native` 进行自动检测。

指令集通常是向后兼容的，所以通常只需要使用你需要的最老微架构的名称就足够了。一种更稳健的方法是列出 CPU 保证具有的特定功能：`-mavx2`，`-mpopcnt`。当你只想针对特定机器调整程序，而不使用可能在不兼容 CPU 上崩溃的指令时，可以使用`-mtune`标志（默认情况下，`-march=x` 也隐含 `-mtune=x`）。

这些选项也可以通过使用编译器指令（pragmas）而不是编译标志来指定一个编译单元：

```cpp
#pragma GCC optimize("O3")
#pragma GCC target("avx2") 
```

当你需要优化单个高性能过程而不增加整个项目的构建时间时，这很有用。

### [#](https://en.algorithmica.org/hpc/compilation/flags/#multiversioned-functions) 多版本函数

有时候你可能也希望在单个库中提供几个特定架构的实现。你可以使用基于属性的语法在编译时自动选择多版本函数：

```cpp
__attribute__(( target("default") )) // fallback implementation
int popcnt(int x) {
    int s = 0;
    for (int i = 0; i < 32; i++)
        s += (x>>i&1);
    return s;
}

__attribute__(( target("popcnt") )) // used if popcnt flag is enabled
int popcnt(int x) {
    return __builtin_popcount(x);
} 
```

在 Clang 中，你不能使用预处理指令从源代码中设置目标和优化标志，但你可以使用属性，就像在 GCC 中一样。[← 编译阶段](https://en.algorithmica.org/hpc/compilation/stages/)[情境优化 →](https://en.algorithmica.org/hpc/compilation/situational/)
