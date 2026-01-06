# 矩阵乘法

> 原文：[`en.algorithmica.org/hpc/algorithms/matmul/`](https://en.algorithmica.org/hpc/algorithms/matmul/)

在这个案例研究中，我们将设计和实现几个矩阵乘法算法。

我们从简单的“for-for-for”算法开始，并逐步改进它，最终得到一个比 BLAS 库性能快 50 倍，且 C 代码行数少于 40 行的版本。

所有实现都是使用 GCC 13 编译，并在 2GHz 的[Zen 2](https://en.wikichip.org/wiki/amd/microarchitectures/zen_2) CPU 上运行。

## [#](https://en.algorithmica.org/hpc/algorithms/matmul/#baseline)基线

将$l \times n$矩阵$A$乘以$n \times m$矩阵$B$的结果定义为$l \times m$矩阵$C$，使得：

$$ C_{ij} = \sum_{k=1}^{n} A_{ik} \cdot B_{kj} $$

为了简单起见，我们只考虑*方阵*，其中$l = m = n$。

要实现矩阵乘法，我们可以简单地将这个定义转换成代码，但我们将使用一维数组（即矩阵）而不是二维数组，以明确指针算术：

```cpp
void matmul(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                c[i * n + j] += a[i * n + k] * b[k * n + j];
} 
```

由于后面的原因，我们将只使用$48$的倍数矩阵大小进行基准测试，但实现对所有其他大小都是正确的。我们还特别使用 32 位浮点数，尽管所有实现都可以很容易地推广到其他数据类型和操作。

使用`g++ -O3 -march=native -ffast-math -funroll-loops`编译，原始方法在~16.7 秒内乘以大小为$n = 1920 = 48 \times 40$的两个矩阵。为了更直观地说明，这大约是$\frac{1920³}{16.7 \times 10⁹} \approx 0.42$个每纳秒的有效操作（GFLOPS），或者大约每个乘法 5 个 CPU 周期，看起来还不是很好。

## [#](https://en.algorithmica.org/hpc/algorithms/matmul/#transposition)转置

通常，当优化处理大量数据的算法时——$1920² \times 3 \times 4 \approx 42$ MB 显然是大量数据，因为它无法装入任何 CPU 缓存——在优化算术之前，应该始终从内存开始，因为它更有可能是瓶颈。

字段$C_{ij}$可以看作是矩阵$A$的第$i$行和矩阵$B$的第$j$列的点积。当我们在上面的内层循环中增加`k`时，我们正在顺序读取矩阵`a`，但我们迭代`b`的列时跳过了$n$个元素，这不如顺序迭代快。

一种著名的优化方法，用于解决此问题是将矩阵$B$存储在*列主序*顺序中——或者，在矩阵乘法之前，将其*转置*。这需要$O(n²)$额外的操作，但确保在内层循环中顺序读取：

```cpp
void matmul(const float *a, const float *_b, float *c, int n) {
    float *b = new float[n * n];

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            b[i * n + j] = _b[j * n + i];

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                c[i * n + j] += a[i * n + k] * b[j * n + k]; // <- note the indices
} 
```

这段代码运行时间为 ~12.4 秒，大约快 30%。

如我们稍后将看到的，转置它比仅仅的顺序内存读取有更多重要的好处。

## [#](https://en.algorithmica.org/hpc/algorithms/matmul/#vectorization) 向量化

现在我们所做的只是顺序读取 `a` 和 `b` 的元素，将它们相乘，并将结果加到一个累加变量中，我们可以使用 SIMD 指令来加速整个过程。使用 GCC 向量类型 来实现它相当直接——我们可以 对齐内存 矩阵行，用零填充它们，然后像计算任何其他 归约 一样计算乘加：

```cpp
// a vector of 256 / 32 = 8 floats
typedef float vec __attribute__ (( vector_size(32) ));

// a helper function that allocates n vectors and initializes them with zeros
vec* alloc(int n) {
    vec* ptr = (vec*) std::aligned_alloc(32, 32 * n);
    memset(ptr, 0, 32 * n);
    return ptr;
}

void matmul(const float *_a, const float *_b, float *c, int n) {
    int nB = (n + 7) / 8; // number of 8-element vectors in a row (rounded up)

    vec *a = alloc(n * nB);
    vec *b = alloc(n * nB);

    // move both matrices to the aligned region
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i * nB + j / 8][j % 8] = _a[i * n + j];
            b[i * nB + j / 8][j % 8] = _b[j * n + i]; // <- b is still transposed
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            vec s{}; // initialize the accumulator with zeros

            // vertical summation
            for (int k = 0; k < nB; k++)
                s += a[i * nB + k] * b[j * nB + k];

            // horizontal summation
            for (int k = 0; k < 8; k++)
                c[i * n + j] += s[k];
        }
    }

    std::free(a);
    std::free(b);
} 
```

对于 $n = 1920$ 的性能现在大约是 2.3 GFLOPS——或者比转置但未向量化的版本高大约 4 倍。

![](img/00d5459969621e683c4e77b5a7a95845.png)

这种优化看起来既不复杂也不特定于矩阵乘法。为什么编译器不能自己 自动向量化 内循环呢？

实际上可以；阻止这一点的唯一因素是 `c` 可能与 `a` 或 `b` 发生重叠。为了排除这种可能性，你可以通过向它添加 `__restrict__` 关键字来通知编译器你保证 `c` 不与任何东西 别名：

```cpp
void matmul(const float *a, const float *_b, float * __restrict__ c, int n) {
    // ...
} 
```

手动和自动向量化实现的表现大致相同。

## [#](https://en.algorithmica.org/hpc/algorithms/matmul/#memory-efficiency) 内存效率

有趣的是，实现效率取决于问题大小。

首先，性能（定义为每秒有用操作的次数）随着循环管理和水平减少的开销增加而增加。然后，大约在 $n=256$ 时，随着矩阵不再适合缓存（$2 \times 256² \times 4 = 512$ KB 是 L2 缓存的容量），性能开始平稳下降，性能瓶颈由 内存带宽 造成。

![](img/975e5f9450e0945c80efa3a5e008db91.png)

还有趣的是，原始实现主要与非向量化的转置版本相当——甚至略好，因为它不需要执行转置。

有些人可能会认为通过进行顺序读取会有一些通用的性能提升，因为我们正在获取更少的缓存行，但这并不是事实：读取 `b` 的第一列确实需要更多时间，但接下来的 15 列读取将与第一列位于相同的缓存行中，所以它们无论如何都会被缓存——除非矩阵如此之大以至于连 `n * cache_line_size` 字节都放不进缓存，这对于任何实际矩阵大小来说都不是问题。

相反，性能在只有少数特定的矩阵大小上下降，这是由于缓存关联性的影响：当$n$是 2 的大幂次的倍数时，我们正在获取`b`的地址，这些地址很可能映射到相同的缓存行，这减少了有效的缓存大小。这解释了$n = 1920 = 2⁷ \times 3 \times 5$时的 30%性能下降，你可以看到对于$1536 = 2⁹ \times 3$的更明显的下降：它大约比$n=1535$慢 3 倍。

所以，出人意料的是，转置矩阵并不能帮助缓存——而且在原始标量实现中，我们实际上并不是由内存带宽瓶颈所限制。但我们的向量化实现确实如此，所以让我们专注于其 I/O 效率。

## [#](https://en.algorithmica.org/hpc/algorithms/matmul/#register-reuse)寄存器重用

使用类似 Python 的符号来引用子矩阵，为了计算单元格$C[x][y]$，我们需要计算$A[x][:]$和 B[:][y]$的点积，这需要获取$2n$个元素，即使我们以列主序存储$B$。

要计算$C[x:x+2][y:y+2]$，即$C$的$2 \times 2$子矩阵，我们需要从$A$中获取两行和从$B$中获取两列，即$A[x:x+2][:]$和$B[:][y:y+2]$，总共包含$4n$个元素，来更新*四个*元素，而不是*一个*——这在 I/O 效率方面是$\frac{2n / 1}{4n / 4} = 2$倍更好。

为了避免多次获取数据，我们需要并行遍历这些行和列，并计算所有可能的$2 \times 2$乘积组合。以下是一个概念验证：

```cpp
void kernel_2x2(int x, int y) {
    int c00 = 0, c01 = 0, c10 = 0, c11 = 0;

    for (int k = 0; k < n; k++) {
        // read rows
        int a0 = a[x][k];
        int a1 = a[x + 1][k];

        // read columns
        int b0 = b[k][y];
        int b1 = b[k][y + 1];

        // update all combinations
        c00 += a0 * b0;
        c01 += a0 * b1;
        c10 += a1 * b0;
        c11 += a1 * b1;
    }

    // write the results to C
    c[x][y]         = c00;
    c[x][y + 1]     = c01;
    c[x + 1][y]     = c10;
    c[x + 1][y + 1] = c11;
} 
```

现在，我们可以简单地调用这个内核来处理$C$的所有$2 \times 2$子矩阵，但我们不会费心去评估它：尽管这个算法在 I/O 操作方面更好，但它仍然无法击败我们的基于 SIMD 的实现。相反，我们将扩展这种方法，并立即开发一个类似的*向量化*内核。

## [#](https://en.algorithmica.org/hpc/algorithms/matmul/#designing-the-kernel)内核设计

而不是从头开始设计一个计算$C$的$h \times w$子矩阵的内核，我们将声明一个函数，使用$A$的$l$到$r$列和$B$的$l$到$r$行来*更新*它。目前来看，这似乎是一个过度泛化的方法，但这个函数接口将在以后证明是有用的。

为了确定$h$和$w$，我们有几个性能考虑因素：

+   通常，为了计算$h \times w$子矩阵，我们需要获取$2 \cdot n \cdot (h + w)$个元素。为了优化 I/O 效率，我们希望$\frac{h \cdot w}{h + w}$的比率很高，这可以通过大而接近正方形的子矩阵来实现。

+   我们希望使用所有现代 x86 架构上可用的 [FMA](https://en.wikipedia.org/wiki/FMA_instruction_set)（“融合乘加”）指令。正如你从名称中可以猜到的，它一次在 8 元素向量上执行 `c += a * b` 操作——这是点积的核心——从而避免了分别执行向量乘法和加法。

+   为了更好地利用这个指令，我们希望利用 指令级并行性。在 Zen 2 上，`fma` 指令的延迟为 5，吞吐量为 2，这意味着我们需要同时执行至少 $5 \times 2 = 10$ 个指令来饱和其执行端口。

+   我们希望避免寄存器溢出（将数据在寄存器和内存之间移动超过必要次数），我们只有 $16$ 个逻辑向量寄存器可以用作累加器（减去那些我们需要存储临时值的寄存器）。

由于这些原因，我们选择了 $6 \times 16$ 的内核。这样，我们一次处理 $96$ 个元素，这些元素存储在 $6 \times 2 = 12$ 个向量寄存器中。为了有效地更新它们，我们使用以下程序：

```cpp
// update 6x16 submatrix C[x:x+6][y:y+16]
// using A[x:x+6][l:r] and B[l:r][y:y+16]
void kernel(float *a, vec *b, vec *c, int x, int y, int l, int r, int n) {
    vec t[6][2]{}; // will be zero-filled and stored in ymm registers

    for (int k = l; k < r; k++) {
        for (int i = 0; i < 6; i++) {
            // broadcast a[x + i][k] into a register
            vec alpha = vec{} + a[(x + i) * n + k]; // converts to a broadcast
            // multiply b[k][y:y+16] by it and update t[i][0] and t[i][1]
            for (int j = 0; j < 2; j++)
                t[i][j] += alpha * b[(k * n + y) / 8 + j]; // converts to an fma
        }
    }

    // write the results back to C
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 2; j++)
            c[((x + i) * n + y) / 8 + j] += t[i][j];
} 
```

我们需要 `t` 以便编译器将这些元素存储在向量寄存器中。我们只需更新它们在 `c` 中的最终目标，但不幸的是，编译器将它们重新写回内存，导致速度降低（将 `__restrict__` 关键字包裹起来也没有帮助）。

在展开这些循环并将 `b` 从 `i` 循环中提升出来（`b[(k * n + y) / 8 + j]` 不依赖于 `i`，可以在所有 6 次迭代中加载一次并重复使用），编译器生成的内容更接近以下内容：

```cpp
for (int k = l; k < r; k++) {
    __m256 b0 = _mm256_load_ps((__m256*) &b[k * n + y];
    __m256 b1 = _mm256_load_ps((__m256*) &b[k * n + y + 8];

    __m256 a0 = _mm256_broadcast_ps((__m128*) &a[x * n + k]);
    t00 = _mm256_fmadd_ps(a0, b0, t00);
    t01 = _mm256_fmadd_ps(a0, b1, t01);

    __m256 a1 = _mm256_broadcast_ps((__m128*) &a[(x + 1) * n + k]);
    t10 = _mm256_fmadd_ps(a1, b0, t10);
    t11 = _mm256_fmadd_ps(a1, b1, t11);

    // ...
} 
```

我们使用了 $12+3=15$ 个向量寄存器和总共 $6 \times 3 + 2 = 20$ 条指令来执行 $16 \times 6 = 96$ 次更新。假设没有其他瓶颈，我们应该达到 `_mm256_fmadd_ps` 的吞吐量。

注意，这个内核是架构特定的。如果没有 `fma`，或者其吞吐量/延迟不同，或者 SIMD 宽度为 128 或 512 位，我们将会做出不同的设计选择。多平台 BLAS 实现包含了 [许多内核](https://github.com/xianyi/OpenBLAS/tree/develop/kernel)，每个都是手动用汇编编写的，并针对特定架构进行了优化。

其余的实现很简单。类似于之前的向量化实现，我们只需将矩阵移动到内存对齐数组中，并调用内核而不是最内层循环：

```cpp
void matmul(const float *_a, const float *_b, float *_c, int n) {
    // to simplify the implementation, we pad the height and width
    // so that they are divisible by 6 and 16 respectively
    int nx = (n + 5) / 6 * 6;
    int ny = (n + 15) / 16 * 16;

    float *a = alloc(nx * ny);
    float *b = alloc(nx * ny);
    float *c = alloc(nx * ny);

    for (int i = 0; i < n; i++) {
        memcpy(&a[i * ny], &_a[i * n], 4 * n);
        memcpy(&b[i * ny], &_b[i * n], 4 * n); // we don't need to transpose b this time
    }

    for (int x = 0; x < nx; x += 6)
        for (int y = 0; y < ny; y += 16)
            kernel(a, (vec*) b, (vec*) c, x, y, 0, n, ny);

    for (int i = 0; i < n; i++)
        memcpy(&_c[i * n], &c[i * ny], 4 * n);

    std::free(a);
    std::free(b);
    std::free(c);
} 
```

这提高了基准性能，但仅提高了 ~40%：

![图片](img/f2657a7ecd6c529f6e6452ee1e44234e.png)

在较小的数组上，速度提升更高（2-3 倍），这表明仍然存在内存带宽问题：

![图片](img/40a8fec92b43c74fb7f4f8f73db525c7.png)

现在，如果你已经阅读了关于缓存无关算法的部分，你就会知道解决这类问题的通用方法是将所有矩阵分成四部分，执行八次递归的块矩阵乘法，并仔细组合结果。这个方案在实践中是可行的，但存在一些递归开销，而且它也不允许我们微调算法，因此，我们将采用不同的、更简单的方法。

## [#](https://en.algorithmica.org/hpc/algorithms/matmul/#blocking)阻塞

与分而治之技巧的*缓存感知*替代方案是*缓存阻塞*：将数据分成可以放入缓存的块，并逐个处理它们。如果我们有多层缓存，我们可以进行分层阻塞：首先选择一个适合 L3 缓存的块数据，然后将其分成适合 L2 缓存的块，依此类推。这种方法需要事先知道缓存大小，但通常更容易实现，并且在实践中也更快。

与数组相比，缓存阻塞在矩阵上做起来不那么简单，但基本思路是这样的：

+   选择适合 L3 缓存的$B$的子矩阵（例如，其列的一个子集）。

+   选择适合 L2 缓存的$A$的子矩阵（例如，其行的一个子集）。

+   选择适合 L1 缓存的$B$的子矩阵（例如，其行的一个子集）。

+   使用内核更新$C$的相关子矩阵。

这里有一个由 Jukka Suomela 提供的良好[可视化](https://jukkasuomela.fi/cache-blocking-demo/)（它展示了多种不同的方法；你感兴趣的是最后一种）。

注意，开始这个过程选择矩阵$B$的决定并非任意。在内核执行期间，我们读取$A$的元素比读取$B$的元素慢得多：我们只获取并广播$A$的一个元素，然后与$B$的 16 个元素相乘。因此，我们希望$B$在 L1 缓存中，而$A$可以留在 L2 缓存中，而不是反过来。

这听起来很复杂，但我们可以通过仅仅增加三个外层`for`循环来实现它，这些循环共同被称为*宏内核*（而更新 6x16 子矩阵的高度优化的低级函数被称为*微内核*）：

```cpp
const int s3 = 64;  // how many columns of B to select
const int s2 = 120; // how many rows of A to select 
const int s1 = 240; // how many rows of B to select

for (int i3 = 0; i3 < ny; i3 += s3)
    // now we are working with b[:][i3:i3+s3]
    for (int i2 = 0; i2 < nx; i2 += s2)
        // now we are working with a[i2:i2+s2][:]
        for (int i1 = 0; i1 < ny; i1 += s1)
            // now we are working with b[i1:i1+s1][i3:i3+s3]
            // and we need to update c[i2:i2+s2][i3:i3+s3] with [l:r] = [i1:i1+s1]
            for (int x = i2; x < std::min(i2 + s2, nx); x += 6)
                for (int y = i3; y < std::min(i3 + s3, ny); y += 16)
                    kernel(a, (vec*) b, (vec*) c, x, y, i1, std::min(i1 + s1, n), ny); 
```

缓存阻塞完全消除了内存瓶颈：

![](img/e03a7a28939cc2aa9d85d80e659b63ad.png)

性能不再（显著）受问题大小的影响：

![](img/3c3d2bcb24bee8e8ec7020c3651d3570.png)

注意，$1536$ 处的下降趋势仍然存在：缓存关联性仍然影响性能。为了减轻这一点，我们可以调整步长常数或在内布局中插入空隙，但现在我们不会去麻烦做这些。

## [#](https://en.algorithmica.org/hpc/algorithms/matmul/#optimization)优化

为了接近性能极限，我们需要一些额外的优化：

+   移除内存分配，并直接在传递给函数的数组上操作。请注意，我们不需要对`a`做任何事情，因为我们一次只读取一个元素，并且我们可以使用一个非对齐的 `store`来操作`c`，因为我们很少使用它，所以我们唯一关心的是读取`b`。

+   移除`std::min`，使得大小参数（大部分情况下）保持不变，并可以被编译器嵌入到机器代码中（这也让编译器能够更有效地展开微内核循环，并避免运行时检查）。

+   手动使用 12 个向量变量重写微内核（编译器似乎难以将它们保持在寄存器中，首先将它们写入临时内存位置，然后再写入$C$）。

这些优化方法简单但实现起来相当繁琐，所以我们不会在文章中列出[代码](https://github.com/sslotin/amh-code/blob/main/matmul/v5-unrolled.cc)。它还需要做更多的工作来有效地支持“奇怪”的矩阵大小，这就是为什么我们只为大小是$48 = \frac{6 \cdot 16}{\gcd(6, 16)}$的倍数的矩阵运行基准测试。

这些单独的小改进累积起来，又带来了 50%的性能提升：

![](img/f3b30c45c12e557f364590624dbeadfe.png)

我们实际上离理论性能极限并不远——这可以通过 SIMD 宽度乘以`fma`指令吞吐量乘以时钟频率来计算：

$$ \underbrace{8}_{SIMD} \cdot \underbrace{2}_{thr.} \cdot \underbrace{2 \cdot 10⁹}_{cycles/sec} = 32 \; GFLOPS \;\; (3.2 \cdot 10^{10}) $$

与一些实际库进行比较，例如[OpenBLAS](https://www.openblas.net/)，更具代表性。最懒惰的方法是简单地从 NumPy 中调用矩阵乘法。由于 Python 可能存在一些轻微的开销，但最终可以达到理论极限的 80%，这似乎是合理的（20%的开销是可以接受的：矩阵乘法并不是 CPU 的唯一用途）。

![](img/3df0f282b3daecc24455ce8a340c9b27.png)

我们已经达到了约 93%的 BLAS 性能和约 75%的理论性能极限，这对于本质上只有 40 行 C 代码来说是非常了不起的。

有趣的是，整个操作可以简化为一个深度嵌套的`for`循环，并达到 BLAS 级别的性能（假设我们处于 2050 年，并使用 GCC 版本 35，它最终停止了寄存器溢出的错误）：

```cpp
for (int i3 = 0; i3 < n; i3 += s3)
    for (int i2 = 0; i2 < n; i2 += s2)
        for (int i1 = 0; i1 < n; i1 += s1)
            for (int x = i2; x < i2 + s2; x += 6)
                for (int y = i3; y < i3 + s3; y += 16)
                    for (int k = i1; k < i1 + s1; k++)
                        for (int i = 0; i < 6; i++)
                            for (int j = 0; j < 2; j++)
                                c[x * n / 8 + i * n / 8 + y / 8 + j]
                                += (vec{} + a[x * n + i * n + k])
                                   * b[n / 8 * k + y / 8 + j]; 
```

还有一种方法可以执行渐近更少的算术运算——斯特拉斯算法——但它有一个很大的常数因子，并且仅对[非常大的矩阵](https://arxiv.org/pdf/1605.01078.pdf)（$n > 4000$）有效，在这些情况下，我们通常不得不使用多进程或某些近似降维方法。

## [#](https://en.algorithmica.org/hpc/algorithms/matmul/#generalizations)推广

FMA 也支持 64 位浮点数，但它不支持整数：你需要分别执行加法和乘法，这会导致性能下降。如果你可以保证所有中间结果都可以精确地表示为 32 位或 64 位浮点数（这通常是情况），那么直接将它们转换为浮点数再转换回来可能更快。

这种方法也可以应用于一些看起来相似的计算。一个例子是定义为“min-plus 矩阵乘法”的计算：

$$ (A \circ B)_{ij} = \min_{1 \le k \le n} (A_{ik} + B_{kj}) $$

它也被称为“距离积”，因为它具有图解释：当应用于自身 $(D \circ D)$ 时，结果是所有成对顶点在完全连接加权图中的最短路径矩阵，该图由边权重矩阵 $D$ 指定。

距离积的一个有趣之处在于，如果我们迭代这个过程并计算

$$ D_2 = D \circ D \\ D_4 = D_2 \circ D_2 \\ D_8 = D_4 \circ D_4 \\ \ldots $$

…我们可以在 $O(\log n)$ 步中找到所有对最短路径：

```cpp
for (int l = 0; l < logn; l++)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                d[i][j] = min(d[i][j], d[i][k] + d[k][j]); 
```

这需要 $O(n³ \log n)$ 次操作。如果我们以特定的顺序执行这两个边松弛操作，我们只需一次遍历就能完成，这被称为 [Floyd-Warshall 算法](https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm)：

```cpp
for (int k = 0; k < n; k++)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            d[i][j] = min(d[i][j], d[i][k] + d[k][j]); 
```

有趣的是，将距离积向量化并在 $O(n³ \log n)$ 总操作中执行它 $O(\log n)$ 次（或可能更少）([或可能更少](https://arxiv.org/pdf/1904.01210.pdf))，比天真地执行 $O(n³)$ 次操作的 Floyd-Warshall 算法要快，尽管快得并不多。

作为练习，尝试加快这个“for-for-for”计算。这比矩阵乘法案例更难，因为现在迭代之间存在逻辑依赖关系，你需要按特定顺序执行更新，但仍然可以设计[一个类似的内核和块迭代顺序](https://github.com/sslotin/amh-code/blob/main/floyd/blocked.cc)，从而实现 30-50 倍的总速度提升。

## [#](https://en.algorithmica.org/hpc/algorithms/matmul/#acknowledgements)致谢

最终算法最初由 Kazushige Goto 设计，它是 GotoBLAS 和 OpenBLAS 的基础。作者本人更详细地描述了它，在 “[Anatomy of High-Performance Matrix Multiplication](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf)”。

展示风格受到了 Jukka Suomela 的 “[Programming Parallel Computers](http://ppc.cs.aalto.fi/)” 课程的影响，该课程包含一个关于加速距离积的[类似案例研究](http://ppc.cs.aalto.fi/ch2/)。[← 使用 SIMD 的前缀和](https://en.algorithmica.org/hpc/algorithms/prefix/)[→ 数据结构案例研究](https://en.algorithmica.org/hpc/data-structures/)
