# Montgomery 乘法

> 原文：[`en.algorithmica.org/hpc/number-theory/montgomery/`](https://en.algorithmica.org/hpc/number-theory/montgomery/)

毫不奇怪，模运算中的大量计算通常花费在计算模运算上，这和 一般整数除法一样慢，通常需要 15-20 个周期，具体取决于操作数的大小。

处理这种麻烦的最佳方式是完全避免模运算，推迟或用 预测 来替换它，这可以在计算模和时做到，例如：

```cpp
const int M = 1e9 + 7;

// input: array of n integers in the [0, M) range
// output: sum modulo M
int slow_sum(int *a, int n) {
    int s = 0;
    for (int i = 0; i < n; i++)
        s = (s + a[i]) % M;
    return s;
}

int fast_sum(int *a, int n) {
    int s = 0;
    for (int i = 0; i < n; i++) {
        s += a[i]; // s < 2 * M
        s = (s >= M ? s - M : s); // will be replaced with cmov
    }
    return s;
}

int faster_sum(int *a, int n) {
    long long s = 0; // 64-bit integer to handle overflow
    for (int i = 0; i < n; i++)
        s += a[i]; // will be vectorized
    return s % M;
} 
```

然而，有时你只有一系列模乘法，没有好的方法可以避免计算除法的余数——除了使用 整数除法技巧，这需要一个常数模数和一些预计算。

但还有一种专门为模运算设计的技巧，称为 *Montgomery 乘法*。

### [#](https://en.algorithmica.org/hpc/number-theory/montgomery/#montgomery-space)Montgomery 空间

Montgomery 乘法通过首先将乘数转换到 *Montgomery 空间*，在那里模乘法可以以较低的成本进行，然后在需要它们的实际值时再转换回来。与一般的整数除法方法不同，Montgomery 乘法在执行单个模数减少时并不高效，只有在存在一系列模运算时才变得有价值。

该空间由模数 $n$ 和一个与 $n$ 互质的正整数 $r \ge n$ 定义。该算法涉及模运算和除以 $r$，因此实际上，$r$ 被选择为 $2^{32}$ 或 $2^{64}$，这样这些操作可以通过右移和位与来完成。

**定义**。Montgomery 空间中一个数 $x$ 的 *代表* $\bar x$ 定义为

$$ \bar{x} = x \cdot r \bmod n $$

计算这种转换涉及到一个乘法和模运算——这是一个昂贵的操作，我们最初就想优化掉，这就是为什么我们只在将数字转换到和从 Montgomery 空间转换的开销值得时才使用这种方法，而不是用于一般的模乘法。

在 Montgomery 空间内，加法、减法和检查相等性是按常规进行的：

$$ x \cdot r + y \cdot r \equiv (x + y) \cdot r \bmod n $$ 然而，这在乘法中并不成立。将 Montgomery 空间中的乘法表示为 $*$，将“常规”乘法表示为 $\cdot$，我们期望的结果是：$$ \bar{x} * \bar{y} = \overline{x \cdot y} = (x \cdot y) \cdot r \bmod n $$ 但在 Montgomery 空间中的常规乘法结果是：$$ \bar{x} \cdot \bar{y} = (x \cdot y) \cdot r \cdot r \bmod n $$ 因此，Montgomery 空间中的乘法定义为 $$ \bar{x} * \bar{y} = \bar{x} \cdot \bar{y} \cdot r^{-1} \bmod n $$

这意味着，在我们在 Montgomery 空间中正常乘以两个数之后，我们需要通过乘以 $r^{-1}$ 并取模来 *化简* 结果——并且有一种有效的方法来完成这个特定的操作。

### [#](https://en.algorithmica.org/hpc/number-theory/montgomery/#montgomery-reduction)Montgomery reduction

假设 $r=2^{32}$，模 $n$ 是 32 位，而我们需要化简的数 $x$ 是 64 位（两个 32 位数的乘积）。我们的目标是计算 $y = x \cdot r^{-1} \bmod n$。

由于 $r$ 与 $n$ 互质，我们知道在 $0, n)$ 范围内存在两个数 $r^{-1}$ 和 $n^\prime$，使得

$$ r \cdot r^{-1} + n \cdot n^\prime = 1 $$

并且 $r^{-1}$ 和 $n^\prime$ 都可以计算，例如，使用 [扩展欧几里得算法。

使用这个恒等式，我们可以将 $r \cdot r^{-1}$ 表示为 $(1 - n \cdot n^\prime)$，并将 $x \cdot r^{-1}$ 写作

$$ \begin{aligned} x \cdot r^{-1} &= x \cdot r \cdot r^{-1} / r \\ &= x \cdot (1 - n \cdot n^{\prime}) / r \\ &= (x - x \cdot n \cdot n^{\prime} ) / r \\ &\equiv (x - x \cdot n \cdot n^{\prime} + k \cdot r \cdot n) / r &\pmod n &\;\;\text{(对于任何整数 $k$)} \\ &\equiv (x - (x \cdot n^{\prime} - k \cdot r) \cdot n) / r &\pmod n \end{aligned} $$ 现在，如果我们选择 $k$ 为 $\lfloor x \cdot n^\prime / r \rfloor$（$x \cdot n^\prime$ 乘积的上 64 位），它将相互抵消，而 $(k \cdot r - x \cdot n^{\prime})$ 将简单地等于 $x \cdot n^{\prime} \bmod r$（$x \cdot n^\prime$ 的下 32 位），这意味着：$$ x \cdot r^{-1} \equiv (x - x \cdot n^{\prime} \bmod r \cdot n) / r $$

算法本身只是评估这个公式，执行两次乘法来计算 $q = x \cdot n^{\prime} \bmod r$ 和 $m = q \cdot n$，然后从 $x$ 中减去它，并将结果右移以除以 $r$。

需要处理的是，结果可能不在 $[0, n)$ 的范围内；但既然

$$ x < n \cdot n < r \cdot n \implies x / r < n $$ 和 $$ m = q \cdot n < r \cdot n \implies m / r < n $$ 确保了 $$ -n < (x - m) / r < n $$

因此，我们只需检查结果是否为负，如果是，则将其加上 $n$，得到以下算法：

```cpp
typedef __uint32_t u32;
typedef __uint64_t u64;

const u32 n = 1e9 + 7, nr = inverse(n, 1ull << 32);

u32 reduce(u64 x) {
    u32 q = u32(x) * nr;      // q = x * n' mod r
    u64 m = (u64) q * n;      // m = q * n
    u32 y = (x - m) >> 32;    // y = (x - m) / r
    return x < m ? y + n : y; // if y < 0, add n to make it be in the [0, n) range
} 
```

最后这个检查相对便宜，但仍然在关键路径上。如果我们对结果在 $[0, 2 \cdot n - 2]$ 范围内而不是 $0, n)$ 范围内没有问题，我们可以将其删除，并无条件地将 $n$ 添加到结果中：

```cpp
u32 reduce(u64 x) {
    u32 q = u32(x) * nr;
    u64 m = (u64) q * n;
    u32 y = (x - m) >> 32;
    return y + n
} 
```

我们也可以将计算图中的 `>> 32` 操作提前一步，并计算 $\lfloor x / r \rfloor - \lfloor m / r \rfloor$ 而不是 $(x - m) / r$。这是正确的，因为 $x$ 和 $m$ 的低 32 位无论如何都是相同的

$$ m = x \cdot n^\prime \cdot n \equiv x \pmod r $$

但为什么我们自愿选择执行两次右移而不是一次呢？这是因为对于`((u64) q * n) >> 32`，我们需要执行一个 32 位乘法并取结果的最高 32 位（x86 的`mul`指令[已经写入一个单独的寄存器，所以这并不需要任何开销），而另一个右移`x >> 32`不在关键路径上。

```cpp
u32 reduce(u64 x) {
    u32 q = u32(x) * nr;
    u32 m = ((u64) q * n) >> 32;
    return (x >> 32) + n - m;
} 
```

Montgomery 乘法相较于其他模数缩减方法的主要优势之一是它不需要非常大的数据类型：它只需要一个$r \times r$的乘法，该乘法提取结果的下$r$位和上$r$位，这在大多数硬件上[有特殊支持](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig_expand=7395,7392,7269,4868,7269,7269,1820,1835,6385,5051,4909,4918,5051,7269,6423,7410,150,2138,1829,1944,3009,1029,7077,519,5183,4462,4490,1944,5055,5012,5055&techs=AVX,AVX2&text=mul)，这也使得它很容易推广到 SIMD 和更大的数据类型：

```cpp
typedef __uint128_t u128;

u64 reduce(u128 x) const {
    u64 q = u64(x) * nr;
    u64 m = ((u128) q * n) >> 64;
    return (x >> 64) + n - m;
} 
```

注意，使用通用的整数除法技巧无法进行 128 位对 64 位的模数运算：编译器[回退](https://godbolt.org/z/fbEE4v4qr)到调用一个慢速的[长算术库函数](https://github.com/llvm-mirror/compiler-rt/blob/69445f095c22aac2388f939bedebf224a6efcdaf/lib/builtins/udivmodti4.c#L22)来支持它。

### [#](https://en.algorithmica.org/hpc/number-theory/montgomery/#faster-inverse-and-transform) 更快的逆运算和转换

Montgomery 乘法本身很快，但它需要一些预计算：

+   计算$n$对$r$取模以得到$n'$，

+   将一个数*转换*到 Montgomery 空间，

+   将一个数从 Montgomery 空间*转换*出来。

最后一个操作已经通过我们刚刚实现的`reduce`过程高效地执行了，但前两个操作可以稍微优化。

**计算逆元** $n' = n^{-1} \bmod r$ 可以通过利用$r$是 2 的幂的事实，并使用以下恒等式来更快地完成，而不是使用扩展欧几里得算法：

$$ a \cdot x \equiv 1 \bmod 2^k \implies a \cdot x \cdot (2 - a \cdot x) \equiv 1 \bmod 2^{2k} $$ 证明：$$ \begin{aligned} a \cdot x \cdot (2 - a \cdot x) &= 2 \cdot a \cdot x - (a \cdot x)² \\ &= 2 \cdot (1 + m \cdot 2^k) - (1 + m \cdot 2^k)² \\ &= 2 + 2 \cdot m \cdot 2^k - 1 - 2 \cdot m \cdot 2^k - m² \cdot 2^{2k} \\ &= 1 - m² \cdot 2^{2k} \\ &\equiv 1 \bmod 2^{2k}. \end{aligned} $$

我们可以从$x = 1$作为$a$模$2¹$的逆元开始，并应用这个恒等式正好$\log_2 r$次，每次将逆元的位数翻倍——这有点类似于牛顿法。

**转换**一个数到 Montgomery 空间可以通过乘以$r$并按通常的方式计算模数来完成，但我们可以利用这个关系：

$$ \bar{x} = x \cdot r \bmod n = x \cdot r² $$

将一个数转换到该空间中只是一个乘以$r²$的操作。因此，我们可以预先计算$r² \bmod n$并执行乘法和减少操作——这实际上可能更快或更慢，因为将一个数乘以$r=2^{k}$可以通过左移实现，而乘以$r² \bmod n$则不能。

### [完整实现](https://en.algorithmica.org/hpc/number-theory/montgomery/#complete-implementation)

将所有内容包装在一个单独的`constexpr`结构体中是很方便的：

```cpp
struct Montgomery {
    u32 n, nr;

    constexpr Montgomery(u32 n) : n(n), nr(1) {
        // log(2^32) = 5
        for (int i = 0; i < 5; i++)
            nr *= 2 - n * nr;
    }

    u32 reduce(u64 x) const {
        u32 q = u32(x) * nr;
        u32 m = ((u64) q * n) >> 32;
        return (x >> 32) + n - m;
        // returns a number in the [0, 2 * n - 2] range
        // (add a "x < n ? x : x - n" type of check if you need a proper modulo)
    }

    u32 multiply(u32 x, u32 y) const {
        return reduce((u64) x * y);
    }

    u32 transform(u32 x) const {
        return (u64(x) << 32) % n;
        // can also be implemented as multiply(x, r^2 mod n)
    }
}; 
```

为了测试其性能，我们可以将蒙哥马利乘法插入到二进制指数运算中：

```cpp
constexpr Montgomery space(M);

int inverse(int _a) {
    u64 a = space.transform(_a);
    u64 r = space.transform(1);

    #pragma GCC unroll(30)
    for (int l = 0; l < 30; l++) {
        if ( (M - 2) >> l & 1 )
            r = space.multiply(r, a);
        a = space.multiply(a, a);
    }

    return space.reduce(r);
} 
```

而使用编译器生成的快速模运算技巧的普通二进制指数运算每次`inverse`调用需要大约 170ns，这个实现需要大约 166ns，如果我们省略`transform`和`reduce`，则可以降低到大约 158ns（一个合理的用例是将`inverse`用作更大模运算子程序的一部分）。这是一个小的改进，但蒙哥马利乘法对于 SIMD 应用和更大的数据类型变得更加有利。

**练习。** 实现高效的*模*矩阵乘法矩阵乘法。[← 扩展欧几里得算法](https://en.algorithmica.org/hpc/number-theory/euclid-extended/)[→ 外部存储](https://en.algorithmica.org/hpc/external-memory/)
