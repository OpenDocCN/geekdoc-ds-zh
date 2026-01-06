# 寄存器内洗牌

> 原文：[`en.algorithmica.org/hpc/simd/shuffling/`](https://en.algorithmica.org/hpc/simd/shuffling/)

掩码允许您仅对向量元素的一部分应用操作。这是一种非常有效且常用的数据操作技术，但在许多情况下，您需要执行更高级的操作，这些操作涉及在向量寄存器内部排列值，而不是仅仅将它们与其他向量混合。

问题在于，在硬件中为每个可能的使用案例添加一个单独的元素洗牌指令是不切实际的。尽管如此，我们可以添加一个通用的排列指令，该指令接受排列的索引并使用预计算的查找表来生成这些索引。

这种通用想法可能过于抽象，所以让我们直接跳到例子。

### [#](https://en.algorithmica.org/hpc/simd/shuffling/#shuffles-and-popcount)洗牌和计数人口

*计数人口*，也称为*汉明重量*，是二进制字符串中`1`位的数量。

这是一个常用的操作，因此 x86 上有一个单独的指令用于计算单词的计数人口：

```cpp
const int N = (1<<12); int a[N];   int popcnt() {  int res = 0; for (int i = 0; i < N; i++) res += __builtin_popcount(a[i]); return res; } 
```

它还支持 64 位整数，将总吞吐量提高了一倍：

```cpp
int popcnt_ll() {  long long *b = (long long*) a; int res = 0; for (int i = 0; i < N / 2; i++) res += __builtin_popcountl(b[i]); return res; } 
```

只需要两个指令：负载融合的计数人口和加法。它们都具有高吞吐量，因此代码在每周期处理大约$8+8=16$字节，因为它受限于这个 CPU 上的 4 位解码宽度。

这些指令大约在 2008 年添加到 x86 CPU 中，与 SSE4 一起。让我们暂时回到向量化成为事实之前，尝试通过其他方式实现计数人口。

常规的方法是逐位遍历二进制字符串：

```cpp
__attribute__ (( optimize("no-tree-vectorize") )) int popcnt() {  int res = 0; for (int i = 0; i < N; i++) for (int l = 0; l < 32; l++) res += (a[i] >> l & 1); return res; } 
```

如预期的那样，它每周期略快于 1/8 字节——大约是 0.2。

我们可以尝试以字节为单位进行处理，而不是逐个位，通过预计算一个包含单个字节计数人口的 256 元素*查找表*，然后在迭代数组的原始字节时查询它：

```cpp
struct Precalc {  alignas(64) char counts[256];   constexpr Precalc() : counts{} { for (int m = 0; m < 256; m++) for (int i = 0; i < 8; i++) counts[m] += (m >> i & 1); } };   constexpr Precalc P;   int popcnt() {  auto b = (unsigned char*) a; // careful: plain "char" is signed int res = 0; for (int i = 0; i < 4 * N; i++) res += P.counts[b[i]]; return res; } 
```

现在它每周期处理大约 2 个字节，如果我们切换到 16 位单词（`unsigned short`），则上升到约 2.7。

与`popcnt`指令相比，这种解决方案仍然非常慢，但现在它可以进行向量化。我们不会尝试通过 gather 指令来加速它，而是采取另一种方法：将查找表的大小缩小到足以放入寄存器中，然后使用特殊的[pshufb](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=pshuf&techs=AVX,AVX2&expand=6331)指令并行查找其值。

在 128 位 SSE3 中引入的原始`pshufb`需要两个寄存器：包含 16 个字节值的查找表和一个包含 16 个 4 位索引（0 到 15）的向量，指定每个位置要选择的字节。在 256 位 AVX2 中，而不是 32 字节的查找表和尴尬的 5 位索引，我们有一个指令可以独立地对两个 128 位通道执行相同的洗牌操作。

因此，对于我们的用例，我们创建了一个 16 字节的查找表，其中包含每个半字节的计数，重复两次：

```cpp
const reg lookup = _mm256_setr_epi8(  /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2, /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3, /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3, /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4,   /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2, /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3, /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3, /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4 ); 
```

现在，为了计算向量的计数，我们将每个字节拆分为低半字节和高半字节，然后使用此查找表检索它们的计数。唯一剩下的事情是仔细地将它们相加：

```cpp
const reg low_mask = _mm256_set1_epi8(0x0f);   int popcnt() {  int k = 0;   reg t = _mm256_setzero_si256();   for (; k + 15 < N; k += 15) { reg s = _mm256_setzero_si256();  for (int i = 0; i < 15; i += 8) { reg x = _mm256_load_si256( (reg*) &a[k + i] );  reg l = _mm256_and_si256(x, low_mask); reg h = _mm256_and_si256(_mm256_srli_epi16(x, 4), low_mask);   reg pl = _mm256_shuffle_epi8(lookup, l); reg ph = _mm256_shuffle_epi8(lookup, h);   s = _mm256_add_epi8(s, pl); s = _mm256_add_epi8(s, ph); }   t = _mm256_add_epi64(t, _mm256_sad_epu8(s, _mm256_setzero_si256())); }   int res = hsum(t);   while (k < N) res += __builtin_popcount(a[k++]);   return res; } 
```

此代码每个周期处理大约 30 字节。理论上，内循环可以执行 32 次，但我们必须每 15 次迭代停止一次，因为 8 位计数器可能会溢出。

`pshufb`指令在某些 SIMD 算法中非常关键，以至于 Wojciech Muła——提出这个算法的人——将其作为他的[Twitter 昵称](https://twitter.com/pshufb)。你可以更快地计算计数：查看他的[GitHub 仓库](https://github.com/WojciechMula/sse-popcount)中不同的向量化计数实现，以及他的[最新论文](https://arxiv.org/pdf/1611.07612.pdf)，其中详细解释了当前的最佳实践。

### [#](https://en.algorithmica.org/hpc/simd/shuffling/#permutations-and-lookup-tables)排列和查找表

本章的最后一个重要示例是`filter`。它是一个非常重要的数据处理原语，它接受一个数组作为输入，并只输出满足给定谓词的元素（保持其原始顺序）。

在单线程标量情况下，通过维护一个每次写入时递增的计数器可以简单地实现：

```cpp
int a[N], b[N];   int filter() {  int k = 0;   for (int i = 0; i < N; i++) if (a[i] < P) b[k++] = a[i];   return k; } 
```

为了进行向量化，我们将使用 `_mm256_permutevar8x32_epi32` 内置函数。它接受一个值向量并使用索引向量单独选择它们。尽管名称如此，它并不*排列*值，而是*复制*它们以形成一个新的向量：结果中允许有重复。

我们算法的一般思想如下：

+   在数据向量上计算谓词——在这种情况下，这意味着执行比较以获取掩码；

+   使用`movemask`指令获取一个标量 8 位掩码；

+   使用此掩码索引查找表，该查找表返回一个排列，将满足谓词的元素移动到向量的开头（保持其原始顺序）；

+   使用`_mm256_permutevar8x32_epi32`内置函数排列值。

+   将整个排列向量写入缓冲区——它可能有一些尾随垃圾，但它的前缀是正确的；

+   计算标量掩码的计数并移动缓冲区指针；

首先，我们需要预先计算排列：

```cpp
struct Precalc {  alignas(64) int permutation[256][8];   constexpr Precalc() : permutation{} { for (int m = 0; m < 256; m++) { int k = 0; for (int i = 0; i < 8; i++) if (m >> i & 1) permutation[m][k++] = i; } } };   constexpr Precalc T; 
```

然后，我们可以实现算法本身：

```cpp
const reg p = _mm256_set1_epi32(P);   int filter() {  int k = 0;   for (int i = 0; i < N; i += 8) { reg x = _mm256_load_si256( (reg*) &a[i] );  reg m = _mm256_cmpgt_epi32(p, x); int mask = _mm256_movemask_ps((__m256) m); reg permutation = _mm256_load_si256( (reg*) &T.permutation[mask] );  x = _mm256_permutevar8x32_epi32(x, permutation); _mm256_storeu_si256((reg*) &b[k], x);  k += __builtin_popcount(mask); }   return k; } 
```

向量化版本需要一些工作来实现，但它比标量版本快 6-7 倍（对于`P`的值无论是低还是高，速度提升略低，因为分支变得可预测）。

![图片](img/057627d1402dd58e05fbb7be94527260.png)

循环性能仍然相对较低——每次迭代需要 4 个 CPU 周期——因为在特定的 CPU（Zen 2）上，`movemask`、`permute`和`store`的吞吐量较低，并且都必须通过相同的执行端口（P2）。在大多数其他 x86 CPU 上，你可以期望它大约快 2 倍。

在 AVX-512 上，过滤操作也可以实现得相当快：它有一个特殊的“[压缩](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig_expand=7395,7392,7269,4868,7269,7269,1820,1835,6385,5051,4909,4918,5051,7269,6423,7410,150,2138,1829,1944,3009,1029,7077,519,5183,4462,4490,1944,1395&text=_mm512_mask_compress_epi32)”指令，该指令接受一个数据向量和掩码，并将未掩码的元素连续写入。这在依赖于各种过滤子例程的算法中，如快速排序，有着巨大的影响。[← 掩码和混合](https://en.algorithmica.org/hpc/simd/masking/)[自动向量化和多指令单数据（SPMD）→](https://en.algorithmica.org/hpc/simd/auto-vectorization/)
