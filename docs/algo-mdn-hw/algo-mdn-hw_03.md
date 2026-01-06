# 编程语言

> 原文：[`en.algorithmica.org/hpc/complexity/languages/`](https://en.algorithmica.org/hpc/complexity/languages/)

如果你正在阅读这本书，那么在你的计算机科学之旅中，你一定有过一个时刻，你开始关心你代码的效率。

我的经历是在高中时期，当我意识到制作网站和进行*有用的*编程并不能让你进入大学，于是进入了令人兴奋的算法编程奥林匹克竞赛的世界。我是一名不错的程序员，尤其是对于一个高中生来说，但我从未真正想过我的代码执行需要多少时间。但突然间，这开始变得很重要：每个问题现在都有一个严格的时间限制。我开始计算我的操作次数。一秒钟你能做多少？

我对计算机架构的了解不多，无法回答这个问题。但我也并不需要正确的答案——我需要一个经验法则。我的思考过程是：“2-3GHz 意味着每秒执行 20 到 30 亿条指令，在一个简单的循环中，我还需要增加循环计数器，检查循环结束条件，进行数组索引等操作，所以让我们为每个有用的操作增加 3-5 条指令的空间”最终我使用了$5 \cdot 10⁸$作为估算值。这些陈述都不正确，但计算我的算法需要多少操作，并将其除以这个数字，对于我的用例来说是一个很好的经验法则。

当然，真正的答案要复杂得多，并且高度依赖于你心中的“操作”类型。对于像指针追踪这样的操作，它可能低至$10⁷$，而对于向量指令集加速的线性代数，它可能高达$10^{11}$。为了展示这些显著的不同，我们将使用不同语言实现的矩阵乘法案例研究——并深入探讨计算机如何执行它们。

## [#](https://en.algorithmica.org/hpc/complexity/languages/#types-of-languages)语言类型

在最低级别上，计算机执行由二进制编码的*指令*组成的*机器代码*，这些指令用于控制 CPU。它们是特定的、古怪的，并且需要大量的智力努力才能与之合作，因此人们在创建计算机之后做的第一件事之一就是创建*编程语言*，这些语言抽象出计算机操作的一些细节，以简化编程过程。

编程语言本质上只是一个接口。用其编写的任何程序都只是更高级的、更优雅的表示形式，但最终仍需要在某些点上将其转换为 CPU 上执行的机器代码——并且有几种不同的方法可以做到这一点：

+   从程序员的视角来看，有两种类型的语言：*编译型*，在执行前进行预处理，和*解释型*，在运行时使用一个称为*解释器*的独立程序执行。

+   从计算机的角度来看，也存在两种类型的语言：*原生*，它直接执行机器代码，和 *托管*，它依赖于某种 *运行时* 来执行。

由于在解释器中运行机器代码没有意义，这总共形成了三种类型的语言：

+   解释型语言，例如 Python、JavaScript 或 Ruby。

+   带有运行时的编译型语言，例如 Java、C# 或 Erlang（以及在其虚拟机上工作的语言，如 Scala、F# 或 Elixir）。

+   编译型原生语言，例如 C、Go 或 Rust。

执行计算机程序没有“正确”的方式：每种方法都有其自身的优势和劣势。解释器和虚拟机提供了灵活性，并使一些高级编程特性成为可能，例如动态类型、运行时代码修改和自动内存管理，但这些都会带来一些不可避免的性能折衷，我们将在下面讨论。

### [#](https://en.algorithmica.org/hpc/complexity/languages/#interpreted-languages)解释型语言

下面是一个纯 Python 中定义的 $1024 \times 1024$ 矩阵乘法的示例：

```cpp
import time
import random

n = 1024

a = [[random.random()
      for row in range(n)]
      for col in range(n)]

b = [[random.random()
      for row in range(n)]
      for col in range(n)]

c = [[0
      for row in range(n)]
      for col in range(n)]

start = time.time()

for i in range(n):
    for j in range(n):
        for k in range(n):
            c[i][j] += a[i][k] * b[k][j]

duration = time.time() - start
print(duration) 
```

这段代码运行了 630 秒。这超过了 10 分钟！

让我们尝试将这个数字放在一个合适的视角中。运行它的 CPU 的时钟频率为 1.4GHz，意味着它每秒执行 $1.4 \cdot 10⁹$ 个周期，整个计算总共接近 $10^{15}$ 个周期，最内层循环中大约每乘法执行 880 个周期。

如果你考虑到 Python 需要做什么来确定程序员的意思，这并不奇怪：

+   它解析表达式 `c[i][j] += a[i][k] * b[k][j]`；

+   尝试确定 `a`、`b` 和 `c` 是什么，并在包含类型信息的特殊哈希表中查找它们的名称；

+   理解 `a` 是一个列表，获取其 `[]` 操作符，检索 `a[i]` 的指针，发现它也是一个列表，再次获取其 `[]` 操作符，获取 `a[i][k]` 的指针，然后是元素本身；

+   查找其类型，确定它是一个 `float`，并获取实现 `*` 操作符的方法；

+   对 `b` 和 `c` 执行相同的事情，最后将结果加赋给 `c[i][j]`。

当然，广泛使用的语言（如 Python）的解释器已经进行了很好的优化，并且可以在重复执行相同代码时跳过一些步骤。但是，由于语言设计，一些相当显著的开销是不可避免的。如果我们去掉所有这些类型检查和指针追踪，也许我们可以将每乘法的周期比接近 1，或者接近原生乘法的“成本”？

### [#](https://en.algorithmica.org/hpc/complexity/languages/#managed-languages)托管语言

与之前相同的矩阵乘法过程，但使用 Java 实现：

```cpp
import java.util.Random;

public class Matmul {
    static int n = 1024;
    static double[][] a = new double[n][n];
    static double[][] b = new double[n][n];
    static double[][] c = new double[n][n];

    public static void main(String[] args) {
        Random rand = new Random();

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                a[i][j] = rand.nextDouble();
                b[i][j] = rand.nextDouble();
                c[i][j] = 0;
            }
        }

        long start = System.nanoTime();

        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < n; k++)
                    c[i][j] += a[i][k] * b[k][j];

        double diff = (System.nanoTime() - start) * 1e-9;
        System.out.println(diff);
    }
} 
```

现在它运行需要 10 秒，相当于每次乘法大约 13 个 CPU 周期——比 Python 快 63 倍。考虑到我们需要从内存中非顺序地读取`b`的元素，运行时间大致符合预期。

Java 是一种*编译型*，但不是*原生型*语言。程序首先编译成*字节码*，然后由虚拟机（JVM）进行解释。为了达到更高的性能，代码中经常执行的部分，例如最内层的`for`循环，在运行时被编译成机器码，然后几乎无开销地执行。这种技术被称为*即时编译*。

JIT 编译不是语言本身的功能，而是其实现的功能。还有一个名为[PyPy](https://www.pypy.org/)的 Python JIT 编译版本，它执行上述代码需要大约 12 秒，而且没有任何修改。

### [#](https://en.algorithmica.org/hpc/complexity/languages/#compiled-languages)编译型语言

现在轮到 C 了：

```cpp
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define n 1024
double a[n][n], b[n][n], c[n][n];

int main() {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i][j] = (double) rand() / RAND_MAX;
            b[i][j] = (double) rand() / RAND_MAX;
        }
    }

    clock_t start = clock();

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                c[i][j] += a[i][k] * b[k][j];

    float seconds = (float) (clock() - start) / CLOCKS_PER_SEC;
    printf("%.4f\n", seconds);

    return 0;
} 
```

如果你用`gcc -O3`编译它，它需要 9 秒。

这看起来并不像是一个巨大的改进——Java 和 PyPy 超过 1-3 秒的优势可以归因于 JIT 编译的额外时间——但我们还没有充分利用一个更好的 C 编译器生态系统。如果我们添加`-march=native`和`-ffast-math`标志，时间突然下降到 0.6 秒！

这里发生的事情是我们向编译器传达了我们正在运行的 CPU 的确切模型(`-march=native`)，并给它自由去重新排列浮点运算(`-ffast-math`)，因此它利用了这一点，并使用向量化来实现这个加速。

并不是不可能调整 PyPy 和 Java 的 JIT 编译器以实现相同性能而不对源代码进行重大修改，但显然对于直接编译成原生代码的语言来说更容易。

### [#](https://en.algorithmica.org/hpc/complexity/languages/#blas)BLAS

最后，让我们看看一个专家优化的实现能做什么。我们将测试一个广泛使用的优化线性代数库，称为[OpenBLAS](https://www.openblas.net/)。使用它的最简单方法是回到 Python，并从`numpy`中调用它：

```cpp
import time
import numpy as np

n = 1024

a = np.random.rand(n, n)
b = np.random.rand(n, n)

start = time.time()

c = np.dot(a, b)

duration = time.time() - start
print(duration) 
```

现在它需要大约 0.12 秒：比自动向量化 C 版本快 5 倍，比我们最初的 Python 实现快 5250 倍！

你通常不会看到如此戏剧性的改进。目前，我们还没有准备好告诉你这是如何实现的。OpenBLAS 中密集矩阵乘法的实现通常是针对*每个*架构定制的 5000 行手写汇编代码（[查看代码](https://github.com/xianyi/OpenBLAS/blob/develop/kernel/x86_64/dgemm_kernel_16x2_haswell.S)）。在后面的章节中，我们将逐一解释所有相关技术，然后返回到这个例子，并使用不到 40 行的 C 代码开发我们自己的 BLAS 级实现。

### [#](https://en.algorithmica.org/hpc/complexity/languages/#takeaway)总结

这里的关键教训是，使用本地、低级语言并不一定能带来性能；但它确实能让你对性能拥有*控制*。

与“每秒 N 次操作”的简化说法相辅相成，许多程序员也存在一种误解，认为使用不同的编程语言会在那个数字上产生某种乘数效应。以这种方式思考并将语言[比较](https://benchmarksgame-team.pages.debian.net/benchmarksgame/index.html)性能并不太有意义：编程语言本质上只是工具，它们以方便的抽象为代价，减少了*一些*对性能的控制。无论执行环境如何，程序员仍然主要需要利用硬件提供的机遇。[←现代硬件](https://en.algorithmica.org/hpc/complexity/hardware/)[→计算机架构](https://en.algorithmica.org/hpc/architecture/)
