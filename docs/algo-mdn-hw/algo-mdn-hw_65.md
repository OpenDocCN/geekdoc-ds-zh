# 内联函数和向量类型

> 原文：[`en.algorithmica.org/hpc/simd/intrinsics/`](https://en.algorithmica.org/hpc/simd/intrinsics/)

使用 SIMD 最低级的方式是直接使用汇编向量指令——它们与它们的标量等效物完全相同——但我们不会这样做。相反，我们将使用映射到这些指令的现代 C/C++编译器中可用的*内联*函数。

在本节中，我们将介绍它们的语法基础，并在本章的其余部分广泛使用它们来完成一些真正有趣的事情。

## [#](https://en.algorithmica.org/hpc/simd/intrinsics/#setup)设置

要使用 x86 内联函数，我们需要做一些前期准备工作。

首先，我们需要确定硬件支持哪些扩展。在 Linux 上，你可以通过调用`cat /proc/cpuinfo`来获取信息，在其他平台上，你最好去[WikiChip](https://en.wikichip.org/wiki/WikiChip)网站，使用 CPU 的名称进行查找。在任一情况下，都应该有一个`flags`部分，列出了所有支持的向量扩展的代码。

此外，还有一个特殊的[CPUID](https://en.wikipedia.org/wiki/CPUID)汇编指令，允许你查询有关 CPU 的各种信息，包括特定向量扩展的支持情况。它主要用于在运行时获取此类信息，以避免为每个微架构分发单独的二进制文件。其输出信息以特征掩码的形式非常密集地返回，因此编译器提供了内置方法来理解它。以下是一个示例：

```cpp
#include <iostream>
using namespace std;

int main() {
    cout << __builtin_cpu_supports("sse") << endl;
    cout << __builtin_cpu_supports("sse2") << endl;
    cout << __builtin_cpu_supports("avx") << endl;
    cout << __builtin_cpu_supports("avx2") << endl;
    cout << __builtin_cpu_supports("avx512f") << endl;

    return 0;
} 
```

其次，我们需要包含一个包含所需内联函数子集的头文件。类似于 GCC 中的`<bits/stdc++.h>`，有一个`<x86intrin.h>`头文件包含了所有这些函数，所以我们只需使用它。

最后，我们需要通知编译器目标 CPU 实际上支持这些扩展。这可以通过`#pragma GCC target(...)`（就像我们之前做的那样[../]）或编译器选项中的`-march=...`标志来实现。如果你在同一台机器上编译和运行代码，你可以设置`-march=native`来自动检测微架构。

在所有后续的代码示例中，假设它们以以下行开始：

```cpp
#pragma GCC target("avx2")
#pragma GCC optimize("O3")

#include <x86intrin.h>
#include <bits/stdc++.h>

using namespace std; 
```

在本章中，我们将重点关注 AVX2 和之前的 SIMD 扩展，这些扩展应该适用于 95%以上的桌面和服务器计算机，尽管一般原则在 AVX512、Arm Neon 和其他 SIMD 架构上也同样适用。

### [#](https://en.algorithmica.org/hpc/simd/intrinsics/#simd-registers)SIMD 寄存器

SIMD 扩展之间最显著的区分是更宽的寄存器支持：

+   SSE（1999）增加了 16 个 128 位的寄存器，称为`xmm0`到`xmm15`。

+   AVX（2011）增加了 16 个 256 位的寄存器，称为`ymm0`到`ymm15`。

+   AVX512（2017）增加了 16 个 512 位的寄存器，称为`zmm0`到`zmm15`。

如您从命名和 512 位已经占满整个缓存行这一事实中可以猜到的，x86 设计者并不打算在近期内添加更宽的寄存器。

C/C++编译器实现了特殊的*向量类型*，这些类型指的是存储在这些寄存器中的数据：

+   128 位的`__m128`, `__m128d`和`__m128i`类型分别用于单精度浮点数、双精度浮点数和各种整数数据；

+   256 位的`__m256`, `__m256d`, `__m256i`;

+   512 位的`__m512`, `__m512d`, `__m512i`。

寄存器本身可以存储任何类型的数据：这些类型仅用于类型检查。您可以通过与转换任何其他类型相同的方式将向量变量转换为另一个向量类型，而且这不会花费您任何代价。

### [#](https://en.algorithmica.org/hpc/simd/intrinsics/#simd-intrinsics)SIMD 内联函数

*内联函数*只是 C 风格的函数，它们对这些向量数据类型进行操作，通常是通过简单地调用相关的汇编指令。

例如，这里是一个使用 AVX 内联函数将两个 64 位浮点数数组相加的循环：

```cpp
double a[100], b[100], c[100];

// iterate in blocks of 4,
// because that's how many doubles can fit into a 256-bit register
for (int i = 0; i < 100; i += 4) {
    // load two 256-bit segments into registers
    __m256d x = _mm256_loadu_pd(&a[i]);
    __m256d y = _mm256_loadu_pd(&b[i]);

    // add 4+4 64-bit numbers together
    __m256d z = _mm256_add_pd(x, y);

    // write the 256-bit result into memory, starting with c[i]
    _mm256_storeu_pd(&c[i], z);
} 
```

使用 SIMD 的主要挑战是将数据放入连续的固定大小块中，以便加载到寄存器中。在上面的代码中，如果数组长度不能被块大小整除，我们可能会遇到一般性问题。对此有两种常见的解决方案：

1.  我们可以通过迭代最后一个不完整的段来“超出”范围。为了确保我们不会通过尝试从或写入我们不拥有的内存区域而引发段错误，我们需要将数组填充到最近的块大小（通常使用一些“中性”元素，例如零）。

1.  少做一次迭代，并在最后写一个小循环来正常计算余数（使用标量操作）。

人类更喜欢#1，因为它更简单，生成的代码更少，而编译器更喜欢#2，因为它们实际上没有其他合法的选项。

### [#](https://en.algorithmica.org/hpc/simd/intrinsics/#instruction-references)指令参考

大多数 SIMD 内联函数遵循类似于`_mm<size>_<action>_<type>`的命名约定，并对应于一个类似命名的汇编指令。一旦您习惯了汇编命名约定，它们就相对容易理解，尽管有时它们的名称似乎是由猫在键盘上走动生成的（解释：[punpcklqdq](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig_expand=3037,3009,4870,4870,4872,4875,833,879,874,849,848,6715,4845,6046,3853,288,6570,6527,6527,90,7307,6385,5993,2692,6946,6949,5456,6938,5456,1021,3007,514,518,4875,7253,7183,3892,5135,5260,5259,6385,3915,4027,3873,7401&techs=AVX,AVX2&text=punpcklqdq)))。

这里有一些额外的例子，以便您了解其精髓：

+   `_mm_add_epi16`：将两个 16 位*扩展打包整数*的 128 位向量相加，或者简单地说，就是`short`类型。

+   `_mm256_acos_pd`：对 4 个*打包双精度浮点数*进行逐元素计算$\arccos$。

+   `_mm256_broadcast_sd`: 将内存位置中的一个`double`类型的数据广播（复制）到结果向量的所有 4 个元素。

+   `_mm256_ceil_pd`: 将每个 4 个`double`类型的数据向上取整到最近的整数。

+   `_mm256_cmpeq_epi32`: 比较两组各 8 个打包的`int`类型的数据，并返回一个掩码，其中包含相等的元素对。

+   `_mm256_blendv_ps`: 根据掩码从两个向量中选择元素。

如你所猜，内联函数的数量组合上非常庞大，而且除此之外，一些指令也有立即数——因此它们的内联函数需要编译时常量参数：例如，浮点比较指令[有 32 种不同的修饰符](https://stackoverflow.com/questions/16988199/how-to-choose-avx-compare-predicate-variants)。

由于某种原因，有一些操作对寄存器中存储的数据类型是无关的，但只接受特定的向量类型（通常是 32 位浮点数）——你只需将其转换为和从它转换以使用该内联函数。为了简化本章的示例，我们将主要使用 256 位 AVX2 寄存器中的 32 位整数（`epi32`）。

对于 x86 SIMD 内联函数，一个非常有用的参考资料是[英特尔内联函数指南](https://software.intel.com/sites/landingpage/IntrinsicsGuide/)，它按类别和扩展分组，包含描述、伪代码、相关的汇编指令以及它们在英特尔微架构上的延迟和吞吐量。你可能想要将该页面添加到书签中。

当你知道存在一个特定的指令，只想查找其名称或性能信息时，英特尔参考很有用。当你不知道它是否存在时，这个[速查表](https://db.in.tum.de/~finis/x86%20intrinsics%20cheat%20sheet%20v1.0.pdf)可能做得更好。

**指令选择**。请注意，编译器并不一定选择你指定的确切指令。类似于我们之前讨论的标量`c = a + b`，也存在一个融合的向量加法指令，因此，而不是每个循环周期使用 2+1+1=4 条指令，编译器[重写了上面的代码](https://godbolt.org/z/dMz8E5Ye8)，使用 3 条指令的块，如下所示：

```cpp
vmovapd ymm1, YMMWORD PTR a[rax]
vaddpd  ymm0, ymm1, YMMWORD PTR b[rax]
vmovapd YMMWORD PTR c[rax], ymm0 
```

有时，尽管这种情况很少见，这种编译器干扰会使事情变得更糟，因此始终检查汇编代码（它们通常以“v”开头）并仔细查看生成的向量指令是一个好主意。

此外，一些内联函数并不映射到单个指令，而是映射到一系列指令，作为方便的快捷方式：广播和提取是一个显著的例子。

### [#](https://en.algorithmica.org/hpc/simd/intrinsics/#gcc-vector-extensions)GCC 向量扩展

如果你觉得 C 内联函数的设计很糟糕，你并不孤单。我花费了数百小时编写 SIMD 代码并阅读英特尔内联函数指南，我仍然不能记得是否需要输入`_mm256`或`__m256`。

内联函数不仅难以使用，而且既不便携也不易于维护。在优秀的软件中，你不想为每个 CPU 维护不同的程序：你希望一次实现，以架构无关的方式进行。

有一天，GNU 项目的编译器工程师们也有同样的想法，并开发了一种定义你自己的向量类型的方法，这些类型感觉更像数组，并且对一些操作符进行了重载，以匹配相关的指令。

在 GCC 中，你可以这样定义一个将 8 个整数打包到 256 位（32 字节）寄存器的向量：

```cpp
typedef int v8si __attribute__ (( vector_size(32) ));
// type ^   ^ typename          size in bytes ^ 
```

不幸的是，这并不是 C 或 C++标准的一部分，因此不同的编译器使用不同的语法来实现这一点。

在命名约定方面，通常会将元素的大小和类型包含在类型名称中：在上面的例子中，我们定义了一个“8 个有符号整数的向量”。但你可以选择任何你想要的名称，比如`vec`、`reg`或者任何其他名称。唯一你不希望做的是将其命名为`vector`，因为这会由于`std::vector`的存在而造成很多混淆。

使用这些类型的主要优势是，对于许多操作，你可以使用正常的 C++运算符，而不是查找相关的内联函数。

```cpp
v4si a = {1, 2, 3, 5};
v4si b = {8, 13, 21, 34};

v4si c = a + b;

for (int i = 0; i < 4; i++)
    printf("%d\n", c[i]);

c *= 2; // multiply by scalar

for (int i = 0; i < 4; i++)
    printf("%d\n", c[i]); 
```

使用向量类型，我们可以极大地简化之前使用内联函数实现的“a + b”循环：

```cpp
typedef double v4d __attribute__ (( vector_size(32) ));
v4d a[100/4], b[100/4], c[100/4];

for (int i = 0; i < 100/4; i++)
    c[i] = a[i] + b[i]; 
```

如你所见，向量扩展与内联函数的噩梦相比要干净得多。它们的缺点是，有些我们可能想要做的事情无法用原生 C++结构表达，所以我们仍然需要内联函数来处理它们。幸运的是，这不是一个排他性的选择，因为向量类型支持零成本转换为`_mm`类型，并返回：

```cpp
v8f x;
int mask = _mm256_movemask_ps((__m256) x) 
```

对于不同的语言，也存在许多第三方库，它们提供了类似的特性来编写可移植的 SIMD 代码，并实现了一些功能，总体上比内联函数和内置向量类型更容易使用。C++中的显著例子包括[Highway](https://github.com/google/highway)、[Expressive Vector Engine](https://github.com/jfalcou/eve)、[Vector Class Library](https://github.com/vectorclass/version2)和[xsimd](https://github.com/xtensor-stack/xsimd)。

建议使用一个成熟的 SIMD 库，因为它可以极大地提高开发者的体验。然而，在这本书中，我们将尽量接近硬件，主要直接使用内联函数，偶尔在可以简化的情况下切换到向量扩展。

* * *

1.  AVX512 还增加了 8 个所谓的*掩码寄存器*，分别命名为`k0`到`k7`，用于掩码和混合数据。我们不会涉及这些内容，而主要使用 AVX2 和之前的标准。↩︎ [← ../SIMD Parallelism](https://en.algorithmica.org/hpc/simd/)[移动数据 →](https://en.algorithmica.org/hpc/simd/moving/)
