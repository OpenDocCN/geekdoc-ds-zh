# 掩码和混合

> 原文：[`en.algorithmica.org/hpc/simd/masking/`](https://en.algorithmica.org/hpc/simd/masking/)

SIMD 编程的一个较大挑战是它的控制流选项非常有限——因为应用于向量的操作对所有元素都是相同的。

这使得通常可以用`if`或任何其他类型的分支轻易解决的问题变得更加困难。在 SIMD 中，它们必须通过各种无分支编程技术来处理，而这些技术并不总是那么容易应用。

### [#](https://en.algorithmica.org/hpc/simd/masking/#masking)掩码

使计算无分支的主要方式是通过*预测*——计算两个分支的结果，然后使用某种算术技巧或特殊的“条件移动”指令：

```cpp
for (int i = 0; i < N; i++)
    a[i] = rand() % 100;

int s = 0;

// branch:
for (int i = 0; i < N; i++)
    if (a[i] < 50)
        s += a[i];

// no branch:
for (int i = 0; i < N; i++)
    s += (a[i] < 50) * a[i];

// also no branch:
for (int i = 0; i < N; i++)
    s += (a[i] < 50 ? a[i] : 0); 
```

为了向量化这个循环，我们需要两个新的指令：

+   `_mm256_cmpgt_epi32`，它比较两个向量中的整数，如果第一个元素大于第二个元素，则产生全一的掩码，否则产生全零的掩码。

+   `_mm256_blendv_epi8`，它根据提供的掩码将两个向量的值进行混合（组合）。

通过掩码和混合向量的元素，使得只有选定的子集受到计算的影响，我们可以以类似于条件移动的方式执行预测：

```cpp
const reg c = _mm256_set1_epi32(49);
const reg z = _mm256_setzero_si256();
reg s = _mm256_setzero_si256();

for (int i = 0; i < N; i += 8) {
    reg x = _mm256_load_si256( (reg*) &a[i] );
    reg mask = _mm256_cmpgt_epi32(x, c);
    x = _mm256_blendv_epi8(x, z, mask);
    s = _mm256_add_epi32(s, x);
} 
```

（为了简洁起见，省略了水平求和和考虑数组余数等次要细节。）

这通常是在 SIMD 中执行预测的方式，但并不总是最优的方法。我们可以利用混合值中有一个是零的事实，并用位运算`and`与掩码而不是混合：

```cpp
const reg c = _mm256_set1_epi32(50);
reg s = _mm256_setzero_si256();

for (int i = 0; i < N; i += 8) {
    reg x = _mm256_load_si256( (reg*) &a[i] );
    reg mask = _mm256_cmpgt_epi32(c, x);
    x = _mm256_and_si256(x, mask);
    s = _mm256_add_epi32(s, x);
} 
```

这个循环执行速度略快，因为在特定的 CPU 上，向量`and`操作比`blend`操作少一个周期。

几种其他指令支持将掩码作为输入，其中最值得注意的是：

+   `_mm256_blend_epi32`内联函数是一个`blend`，它使用 8 位整数掩码而不是向量（这就是为什么它没有`v`在结尾）。

+   `_mm256_maskload_epi32`和`_mm256_maskstore_epi32`内联函数，它们一次从内存中加载/存储 SIMD 块并与掩码进行`and`操作。

我们还可以使用内置的向量类型进行预测：

```cpp
vec *v = (vec*) a;
vec s = {};

for (int i = 0; i < N / 8; i++)
    s += (v[i] < 50 ? v[i] : 0); 
```

所有这些版本在约 13 GFLOPS 的速度下工作，因为本例非常简单，编译器可以完全自行将循环向量化。让我们继续看一些更复杂的例子，这些例子不能自动向量化。

### [#](https://en.algorithmica.org/hpc/simd/masking/#searching)搜索

在下一个例子中，我们需要在数组中找到一个特定的值并返回其位置（即`std::find`）：

```cpp
const int N = (1<<12);
int a[N];

int find(int x) {
    for (int i = 0; i < N; i++)
        if (a[i] == x)
            return i;
    return -1;
} 
```

为了基准测试`find`函数，我们用从$0$到$(N - 1)$的数字填充数组，然后重复搜索随机元素：

```cpp
for (int i = 0; i < N; i++)
    a[i] = i;

for (int t = 0; t < K; t++)
    checksum ^= find(rand() % N); 
```

标量版本提供了大约 4 GFLOPS 的性能。这个数字包括了我们没有必要处理的元素，所以在大脑中将这个数字除以 2（我们预期需要检查的元素的比例）。

为了向量化它，我们需要将包含其元素的向量与搜索值进行比较，以产生一个掩码，然后以某种方式检查这个掩码是否为零。如果不是，所需的元素就在这个 8 元素块中的某个地方。

要检查掩码是否为零，我们可以使用`_mm256_movemask_ps`内建函数，它从向量的每个 32 位元素中取出第一个位，并从中生成一个 8 位整数掩码。然后我们可以检查这个掩码是否非零——如果是的话，也可以立即使用`ctz`指令获取索引：

```cpp
int find(int needle) {
    reg x = _mm256_set1_epi32(needle);

    for (int i = 0; i < N; i += 8) {
        reg y = _mm256_load_si256( (reg*) &a[i] );
        reg m = _mm256_cmpeq_epi32(x, y);
        int mask = _mm256_movemask_ps((__m256) m);
        if (mask != 0)
            return i + __builtin_ctz(mask);
    }

    return -1;
} 
```

这个版本提供了大约 20 GFLOPS 或大约比标量版本快 5 倍。它只在热循环中使用 3 条指令：

```cpp
vpcmpeqd  ymm0, ymm1, YMMWORD PTR a[0+rdx*4]
vmovmskps eax, ymm0
test      eax, eax
je        loop 
```

检查一个向量是否为零是一个常见的操作，在 SIMD 中有一个类似于`test`的操作我们可以使用：

```cpp
int find(int needle) {
    reg x = _mm256_set1_epi32(needle);

    for (int i = 0; i < N; i += 8) {
        reg y = _mm256_load_si256( (reg*) &a[i] );
        reg m = _mm256_cmpeq_epi32(x, y);
        if (!_mm256_testz_si256(m, m)) {
            int mask = _mm256_movemask_ps((__m256) m);
            return i + __builtin_ctz(mask);
        }
    }

    return -1;
} 
```

我们仍然使用`movemask`来稍后执行`ctz`，但现在热循环现在少了一条指令：

```cpp
vpcmpeqd ymm0, ymm1, YMMWORD PTR a[0+rdx*4]
vptest   ymm0, ymm0
je       loop 
```

这并没有显著提高性能，因为`vptest`和`vmovmskps`的吞吐量都是 1，并且无论我们在循环中做什么，都会成为计算的瓶颈。

为了克服这个限制，我们可以以 16 个元素为一组进行迭代，并使用位运算`or`将两个 256 位 AVX2 寄存器的独立比较结果组合起来：

```cpp
int find(int needle) {
    reg x = _mm256_set1_epi32(needle);

    for (int i = 0; i < N; i += 16) {
        reg y1 = _mm256_load_si256( (reg*) &a[i] );
        reg y2 = _mm256_load_si256( (reg*) &a[i + 8] );
        reg m1 = _mm256_cmpeq_epi32(x, y1);
        reg m2 = _mm256_cmpeq_epi32(x, y2);
        reg m = _mm256_or_si256(m1, m2);
        if (!_mm256_testz_si256(m, m)) {
            int mask = (_mm256_movemask_ps((__m256) m2) << 8)
                     +  _mm256_movemask_ps((__m256) m1);
            return i + __builtin_ctz(mask);
        }
    }

    return -1;
} 
```

移除这个障碍后，性能现在达到大约 34 GFLOPS。但为什么不是 40 呢？难道不应该快两倍吗？

下面是如何在汇编中查看循环的一次迭代的示例：

```cpp
vpcmpeqd ymm2, ymm1, YMMWORD PTR a[0+rdx*4]
vpcmpeqd ymm3, ymm1, YMMWORD PTR a[32+rdx*4]
vpor     ymm0, ymm3, ymm2
vptest   ymm0, ymm0
je       loop 
```

每次迭代，我们需要执行 5 条指令。虽然所有相关执行端口的吞吐量允许在平均一个周期内完成这些操作，但我们不能这样做，因为这个特定 CPU（Zen 2）的解码宽度是 4。因此，性能受到限制，只有其可能性能的 5/9。

为了减轻这一点，我们可以在每次迭代中再次加倍我们处理的 SIMD 块的数量：

```cpp
unsigned get_mask(reg m) {
    return _mm256_movemask_ps((__m256) m);
}

reg cmp(reg x, int *p) {
    reg y = _mm256_load_si256( (reg*) p );
    return _mm256_cmpeq_epi32(x, y);
}

int find(int needle) {
    reg x = _mm256_set1_epi32(needle);

    for (int i = 0; i < N; i += 32) {
        reg m1 = cmp(x, &a[i]);
        reg m2 = cmp(x, &a[i + 8]);
        reg m3 = cmp(x, &a[i + 16]);
        reg m4 = cmp(x, &a[i + 24]);
        reg m12 = _mm256_or_si256(m1, m2);
        reg m34 = _mm256_or_si256(m3, m4);
        reg m = _mm256_or_si256(m12, m34);
        if (!_mm256_testz_si256(m, m)) {
            unsigned mask = (get_mask(m4) << 24)
                          + (get_mask(m3) << 16)
                          + (get_mask(m2) << 8)
                          +  get_mask(m1);
            return i + __builtin_ctz(mask);
        }
    }

    return -1;
} 
```

现在显示的吞吐量为 43 GFLOPS——或者大约比原始标量实现快 10 倍。

将每个周期扩展到 64 个值并不能有所帮助：当遇到条件时，小数组会因为所有这些额外的`movemask`而受到开销的影响，而大数组无论如何都会因为内存带宽成为瓶颈。

### [#](https://en.algorithmica.org/hpc/simd/masking/#counting-values) 计数值

作为最后的练习，让我们找到数组中一个值的计数，而不仅仅是它的第一次出现：

```cpp
int count(int x) {
    int cnt = 0;
    for (int i = 0; i < N; i++)
        cnt += (a[i] == x);
    return cnt;
} 
```

为了向量化它，我们只需要将比较掩码转换为每个元素为 0 或 1，并计算总和：

```cpp
const reg ones = _mm256_set1_epi32(1);

int count(int needle) {
    reg x = _mm256_set1_epi32(needle);
    reg s = _mm256_setzero_si256();

    for (int i = 0; i < N; i += 8) {
        reg y = _mm256_load_si256( (reg*) &a[i] );
        reg m = _mm256_cmpeq_epi32(x, y);
        m = _mm256_and_si256(m, ones);
        s = _mm256_add_epi32(s, m);
    }

    return hsum(s);
} 
```

两种实现都产生了大约 15 GFLOPS：编译器可以完全自行向量化第一种实现。

但是，编译器无法发现的一个技巧是，当重新解释为整数时，所有位都是 1 的掩码是负一。因此，我们可以跳过与最低位进行与操作的部分，并直接使用掩码本身，然后只需对最终结果取反：

```cpp
int count(int needle) {
    reg x = _mm256_set1_epi32(needle);
    reg s = _mm256_setzero_si256();

    for (int i = 0; i < N; i += 8) {
        reg y = _mm256_load_si256( (reg*) &a[i] );
        reg m = _mm256_cmpeq_epi32(x, y);
        s = _mm256_add_epi32(s, m);
    }

    return -hsum(s);
} 
```

这并不提高该特定架构的性能，因为吞吐量实际上是由更新`s`的操作瓶颈所限制的：存在对前一次迭代的依赖，因此循环不能比每个 CPU 周期一次迭代更快地执行。如果我们把累加器分成两部分，我们可以利用指令级并行性： 

```cpp
int count(int needle) {
    reg x = _mm256_set1_epi32(needle);
    reg s1 = _mm256_setzero_si256();
    reg s2 = _mm256_setzero_si256();

    for (int i = 0; i < N; i += 16) {
        reg y1 = _mm256_load_si256( (reg*) &a[i] );
        reg y2 = _mm256_load_si256( (reg*) &a[i + 8] );
        reg m1 = _mm256_cmpeq_epi32(x, y1);
        reg m2 = _mm256_cmpeq_epi32(x, y2);
        s1 = _mm256_add_epi32(s1, m1);
        s2 = _mm256_add_epi32(s2, m2);
    }

    s1 = _mm256_add_epi32(s1, s2);

    return -hsum(s1);
} 
```

它现在提供约 22 GFLOPS 的性能，这是可能达到的最高水平。

当将此代码适配为较短的数据类型时，请注意累加器可能会溢出。为了解决这个问题，可以添加另一个更大的累加器，并定期停止循环，将本地累加器中的值添加到它，然后重置本地累加器。例如，对于 8 位整数，这意味着创建另一个内部循环，该循环执行 $\lfloor \frac{256-1}{8} \rfloor = 15$ 次迭代。[← 减法操作](https://en.algorithmica.org/hpc/simd/reduction/)[寄存器内洗牌 →](https://en.algorithmica.org/hpc/simd/shuffling/)
