# SIMD 的 Argmin

> 原文：[`en.algorithmica.org/hpc/algorithms/argmin/`](https://en.algorithmica.org/hpc/algorithms/argmin/)

计算数组的*最小值*是易于向量化，因为它与其他任何归约操作没有区别：在 AVX2 中，你只需要使用一个方便的`_mm256_min_epi32`内建函数作为内部操作。它在一个周期内计算两个 8 元素向量的最小值——甚至比标量情况更快，标量情况至少需要一个比较和一个条件跳转。

找到最小元素(*argmin*)的*索引*要困难得多，但仍然可以非常高效地将其向量化。在本节中，我们设计了一个算法，该算法以计算最小值的速度（几乎）计算 argmin，比原始标量方法快 15 倍左右。

### [#](https://en.algorithmica.org/hpc/algorithms/argmin/#scalar-baseline)标量基线

对于我们的基准测试，我们创建一个随机的 32 位整数数组，然后反复尝试找到其中最小值的索引（如果不是唯一的，则是第一个）：

```cpp
const int N = (1 << 16);
alignas(32) int a[N];

for (int i = 0; i < N; i++)
    a[i] = rand(); 
```

为了说明，我们假设$N$是 2 的幂，并且对所有实验进行$N=2^{13}$，这样内存带宽就不是问题。

在标量情况下实现 argmin，我们只需要维护索引而不是最小值：

```cpp
int argmin(int *a, int n) {
    int k = 0;

    for (int i = 0; i < n; i++)
        if (a[i] < a[k])
            k = i;

    return k;
} 
```

它的工作效率大约在 1.5 GFLOPS——这意味着平均每秒处理$1.5 \times 10⁹$个值，或者每个周期大约处理 0.75 个值（CPU 的时钟频率为 2GHz）。

让我们将其与`std::min_element`进行比较：

```cpp
int argmin(int *a, int n) {
    int k = std::min_element(a, a + n) - a;
    return k;
} 
```

GCC 提供的版本大约是 0.28 GFLOPS——显然，编译器无法穿透所有抽象。这是永远不要使用 STL 的另一个提醒。

### [#](https://en.algorithmica.org/hpc/algorithms/argmin/#vector-of-indices)索引向量

向量化标量实现的问题在于后续迭代之间存在依赖关系。当我们优化数组求和时，我们遇到了相同的问题，我们通过将数组分成 8 个切片来解决它，每个切片代表其索引的子集，这些索引具有相同的余数模 8。我们也可以在这里应用同样的技巧，只不过我们还要考虑数组索引。

当我们拥有连续元素及其索引的向量时，我们可以使用 predication 并行处理它们：

```cpp
typedef __m256i reg;

int argmin(int *a, int n) {
    // indices on the current iteration
    reg cur = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    // the current minimum for each slice
    reg min = _mm256_set1_epi32(INT_MAX);
    // its index (argmin) for each slice
    reg idx = _mm256_setzero_si256();

    for (int i = 0; i < n; i += 8) {
        // load a new SIMD block
        reg x = _mm256_load_si256((reg*) &a[i]);
        // find the slices where the minimum is updated
        reg mask = _mm256_cmpgt_epi32(min, x);
        // update the indices
        idx = _mm256_blendv_epi8(idx, cur, mask);
        // update the minimum (can also similarly use a "blend" here, but min is faster)
        min = _mm256_min_epi32(x, min);
        // update the current indices
        const reg eight = _mm256_set1_epi32(8);
        cur = _mm256_add_epi32(cur, eight);       // 
        // can also use a "blend" here, but min is faster
    }

    // find the argmin in the "min" register and return its real index

    int min_arr[8], idx_arr[8];

    _mm256_storeu_si256((reg*) min_arr, min);
    _mm256_storeu_si256((reg*) idx_arr, idx);

    int k = 0, m = min_arr[0];

    for (int i = 1; i < 8; i++)
        if (min_arr[i] < m)
            m = min_arr[k = i];

    return idx_arr[k];
} 
```

它的工作效率大约在 8-8.5 GFLOPS。迭代之间仍然存在一些依赖关系，因此我们可以通过考虑每个迭代超过 8 个元素并利用指令级并行来优化它。

这将大大提高性能，但不足以匹配计算最小值的速度（约 24 GFLOPS），因为还有一个瓶颈。在每次迭代中，我们需要一个加载融合的比较、一个加载融合的最小值、一个混合和一个加法——总共 4 条指令来处理 8 个元素。由于这个 CPU（Zen 2）的解码宽度仅为 4，即使我们设法消除了所有其他瓶颈，性能仍然会受限于 8 × 2 = 16 GFLOPS。

相反，我们将切换到另一种需要每个元素更少指令的方法。

### [#](https://en.algorithmica.org/hpc/algorithms/argmin/#branches-arent-scary)分支并不可怕

当我们运行标量版本时，我们多久更新一次最小值？

直觉告诉我们，如果所有值都是独立随机抽取的，那么下一个元素小于所有前一个元素的事件不应该很频繁。更精确地说，它等于处理元素数量的倒数。因此，满足`a[i] < a[k]`条件的预期次数等于调和级数的和：

$$ \frac{1}{2} + \frac{1}{3} + \frac{1}{4} + \ldots + \frac{1}{n} = O(\ln(n)) $$

因此，对于一百个元素的数组，最小值大约更新 5 次，对于一千个元素的数组更新 7 次，而对于一百万个元素的数组则只需更新 14 次——当作为所有新最小值检查的分数来看时，这并不算多。

编译器可能无法自己解决这个问题，所以让我们明确提供这个信息：

```cpp
int argmin(int *a, int n) {
    int k = 0;

    for (int i = 0; i < n; i++)
        if (a[i] < a[k]) [[unlikely]]
            k = i;

    return k;
} 
```

编译器优化了机器代码布局，现在 CPU 能够以大约 2 GFLOPS 的速度执行循环——这比非提示循环的 1.5 GFLOPS 略有提高，但幅度不大。

这里是想法：如果我们整个计算过程中只更新最小值大约十几次，我们可以丢弃所有的向量混合和索引更新，只需维护最小值并定期检查它是否已更改。在这个检查中，我们可以使用我们想要的任何慢速方法来更新 argmin，因为它只会被调用几次。

要使用 SIMD 实现它，我们每个迭代只需要进行一次向量加载、一次比较和一次测试是否为零：

```cpp
int argmin(int *a, int n) {
    int min = INT_MAX, idx = 0;

    reg p = _mm256_set1_epi32(min);

    for (int i = 0; i < n; i += 8) {
        reg y = _mm256_load_si256((reg*) &a[i]); 
        reg mask = _mm256_cmpgt_epi32(p, y);
        if (!_mm256_testz_si256(mask, mask)) { [[unlikely]]
            for (int j = i; j < i + 8; j++)
                if (a[j] < min)
                    min = a[idx = j];
            p = _mm256_set1_epi32(min);
        }
    }

    return idx;
} 
```

它已经以约 8.5 GFLOPS 的速度运行，但现在循环被`testz`指令所限制，该指令的吞吐量仅为一次。解决方案是加载两个连续的 SIMD 块并使用它们的最小值，这样`testz`就可以一次有效地处理 16 个元素：

```cpp
int argmin(int *a, int n) {
    int min = INT_MAX, idx = 0;

    reg p = _mm256_set1_epi32(min);

    for (int i = 0; i < n; i += 16) {
        reg y1 = _mm256_load_si256((reg*) &a[i]);
        reg y2 = _mm256_load_si256((reg*) &a[i + 8]);
        reg y = _mm256_min_epi32(y1, y2);
        reg mask = _mm256_cmpgt_epi32(p, y);
        if (!_mm256_testz_si256(mask, mask)) { [[unlikely]]
            for (int j = i; j < i + 16; j++)
                if (a[j] < min)
                    min = a[idx = j];
            p = _mm256_set1_epi32(min);
        }
    }

    return idx;
} 
```

这个版本以约 10 GFLOPS 的速度运行。为了消除其他障碍，我们可以做两件事：

+   将块大小增加到 32 个元素，以允许更多的指令级并行性。

+   优化局部 argmin：我们不需要计算其确切位置，只需保存块的索引，然后在结束时只查找一次。这使得我们只需在每次正检查时计算最小值，并将其广播到向量中，这更简单且更快。

实现了这两个优化后，性能增加到惊人的 ~22 GFLOPS：

```cpp
int argmin(int *a, int n) {
    int min = INT_MAX, idx = 0;

    reg p = _mm256_set1_epi32(min);

    for (int i = 0; i < n; i += 32) {
        reg y1 = _mm256_load_si256((reg*) &a[i]);
        reg y2 = _mm256_load_si256((reg*) &a[i + 8]);
        reg y3 = _mm256_load_si256((reg*) &a[i + 16]);
        reg y4 = _mm256_load_si256((reg*) &a[i + 24]);
        y1 = _mm256_min_epi32(y1, y2);
        y3 = _mm256_min_epi32(y3, y4);
        y1 = _mm256_min_epi32(y1, y3);
        reg mask = _mm256_cmpgt_epi32(p, y1);
        if (!_mm256_testz_si256(mask, mask)) { [[unlikely]]
            idx = i;
            for (int j = i; j < i + 32; j++)
                min = (a[j] < min ? a[j] : min);
            p = _mm256_set1_epi32(min);
        }
    }

    for (int i = idx; i < idx + 31; i++)
        if (a[i] == min)
            return i;

    return idx + 31;
} 
```

这几乎达到了极限，因为仅仅计算最小值本身在大约 24-25 GFLOPS 的速度下工作。

所有这些喜欢分支的 SIMD 实现的唯一问题是它们依赖于最小值很少更新。这在随机输入分布中是正确的，但在最坏的情况下不是。如果我们用一个递减的数字序列填充数组，最后一个实现的性能将下降到大约 2.7 GFLOPS — 慢了大约 10 倍（尽管仍然比标量代码快，因为我们只计算每个块的最小值）。

解决这个问题的一种方法是与快速排序类似的随机算法做同样的事情：自己 shuffle 输入并随机顺序遍历数组。这让你避免了这种最坏情况惩罚，但由于 RNG- 和 内存-相关的问题，实现起来很棘手。有一个更简单的解决方案。

### [#](https://en.algorithmica.org/hpc/algorithms/argmin/#find-the-minimum-then-find-the-index) 找到最小值，然后找到索引

我们知道 how to calculate the minimum of an array 快速以及 how to find an element in an array 快速 — 那么，我们为什么不在分别计算最小值后再去寻找它呢？

```cpp
int argmin(int *a, int n) {
    int needle = min(a, n);
    int idx = find(a, n, needle);
    return idx;
} 
```

如果我们最优地实现这两个子程序（检查相关文章），对于随机数组性能将是 ~18 GFLOPS，对于递减数组是 ~12 GFLOPS — 这是有道理的，因为我们预计要分别读取数组 1.5 和 2 次。这本身并不糟糕 — 至少我们避免了 10 倍的最坏情况性能惩罚 — 但问题是，这种受惩罚的性能也转化为更大的数组，当我们受限于 内存带宽 而不是计算时。

幸运的是，我们 already know how to fix it. 我们可以 split the array into blocks of fixed size $B$ 并在这些块上计算最小值，同时维护全局最小值。当新块上的最小值低于全局最小值时，我们更新它，并记住全局最小值当前所在的块号。处理完整个数组后，我们只需回到那个块，扫描其 $B$ 个元素以找到 argmin。

这样我们只处理 $(N + B)$ 个元素，而且不必牺牲 ½ 或 ⅓ 的性能：

```cpp
const int B = 256;

// returns the minimum and its first block
pair<int, int> approx_argmin(int *a, int n) {
    int res = INT_MAX, idx = 0;
    for (int i = 0; i < n; i += B) {
        int val = min(a + i, B);
        if (val < res) {
            res = val;
            idx = i;
        }
    }
    return {res, idx};
}

int argmin(int *a, int n) {
    auto [needle, base] = approx_argmin(a, n);
    int idx = find(a + base, B, needle);
    return base + idx;
} 
```

这导致了最终实现的结果，对于随机数组和递减数组分别是 ~22 和 ~19 GFLOPS。

完整的实现，包括 `min()` 和 `find()`，大约有 100 行代码。 [查看这里](https://github.com/sslotin/amh-code/blob/main/argmin/combined.cc)，尽管它还远未达到生产级别。

### [#](https://en.algorithmica.org/hpc/algorithms/argmin/#summary) 总结

Here are the results combined for all implementations:

```cpp
algorithm    rand   decr   reason for the performance difference
-----------  -----  -----  -------------------------------------------------------------
std          0.28   0.28   
scalar       1.54   1.89   efficient branch prediction
+ hinted     1.95   0.75   wrong hint
index        8.17   8.12
simd         8.51   1.65   scalar-based argmin on each iteration
+ ilp        10.22  1.74   ^ same
+ optimized  22.44  2.70   ^ same, but faster because there are less inter-dependencies
min+find     18.21  12.92  find() has to scan the entire array
+ blocked    22.23  19.29  we still have an optional horizontal minimum every B elements 
```

对这些结果持保留态度：测量结果相当嘈杂，它们仅针对两种输入分布、特定数组大小（$N=2^{13}$，L1 缓存的大小）、特定架构（Zen 2）和特定且略过时的编译器（GCC 9.3）进行了测试——编译器的优化也非常容易受到基准测试代码中微小变化的影响。

仍然有一些小事情可以优化，但潜在的提升不到 10%，所以我没有费心去做。有一天我可能会鼓起勇气，将算法优化到理论极限，处理不能被块大小整除的数组大小和非对齐的内存情况，然后在许多架构上正确地重新运行基准测试，包括 p 值等。如果有人在我之前做了这件事，请[提醒我](http://sereja.me/)。

### [#](https://en.algorithmica.org/hpc/algorithms/argmin/#acknowledgements)致谢

第一个基于索引的 SIMD 算法最初是由 Wojciech Muła 在 2018 年设计的。[原文链接](http://0x80.pl/notesen/2018-10-03-simd-index-of-min.html)。

感谢 Zach Wegner[指出](https://twitter.com/zwegner/status/1491520929138151425)使用内联函数手动实现 Muła 算法时性能有所提升（我最初使用了 GCC 向量类型）。

在发表后，我发现[BQN](https://mlochbaum.github.io/BQN/)的创造者[Marshall Lochbaum](https://www.aplwiki.com/wiki/Marshall_Lochbaum)在 2019 年开发 Dyalog APL 时设计了一个[非常相似的算法](https://forums.dyalog.com/viewtopic.php?f=13&t=1579&sid=e2cbd69817a17a6e7b1f76c677b1f69e#p6239)。请更加关注数组编程语言的世界！[← 整数分解](https://en.algorithmica.org/hpc/algorithms/factorization/)[使用 SIMD 进行前缀和 →](https://en.algorithmica.org/hpc/algorithms/prefix/)
