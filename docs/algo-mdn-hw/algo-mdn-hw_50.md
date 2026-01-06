# 无缓存感知算法

> 原文：[`en.algorithmica.org/hpc/external-memory/oblivious/`](https://en.algorithmica.org/hpc/external-memory/oblivious/)

在外部存储模型的背景下，有两种类型的有效算法：

+   *缓存感知*算法对已知的 $B$ 和 $M$ 有效。

+   *无缓存感知*算法对任何 $B$ 和 $M$ 都有效。

例如，外部归并排序是一个缓存感知但不是无缓存感知的算法：我们需要知道系统的内存特性，即可用内存与块大小的比率，以找到正确的 $k$ 来执行 $k$-路归并排序。

无缓存感知算法很有趣，因为它们会自动对所有缓存层次结构中的内存级别变得最优，而不仅仅是它们被特别调整的那个级别。在这篇文章中，我们考虑了它们在矩阵计算中的一些应用。

## [#](https://en.algorithmica.org/hpc/external-memory/oblivious/#matrix-transposition)矩阵转置

假设我们有一个大小为 $N \times N$ 的方阵 $A$，我们需要对其进行转置。按照定义的朴素方法可能会这样做：

```cpp
for (int i = 0; i < n; i++)  for (int j = 0; j < i; j++) swap(a[j * N + i], a[i * N + j]); 
```

在这里，我们使用了一个指向内存区域开头的单指针，而不是 2 维数组，以便更明确地描述其内存操作。

该代码的 I/O 复杂度为 $O(N²)$，因为写入不是顺序的。如果你尝试交换迭代变量，情况将相反，但结果将是相同的。

### [#](https://en.algorithmica.org/hpc/external-memory/oblivious/#algorithm)算法

*无缓存感知*算法依赖于以下块矩阵恒等式：

$$ \begin{pmatrix} A & B \\ C & D \end{pmatrix}^T= \begin{pmatrix} A^T & C^T \\ B^T & D^T \end{pmatrix} $$

它允许我们使用分治法递归地解决问题：

1.  将输入矩阵分成 4 个更小的矩阵。

1.  递归地转置每一个。

1.  通过交换角结果矩阵来组合结果。

在矩阵上实现分治法比在数组上要复杂一些，但主要思想是相同的。我们不想显式地复制子矩阵，而是想使用对它们的“视图”，并在数据开始适合 L1 缓存时切换到朴素方法（或者如果你事先不知道，可以选择像 $32 \times 32$ 这样的小尺寸）。我们还需要仔细处理当 $n$ 为奇数时无法将矩阵分成 4 个相等子矩阵的情况。

```cpp
void transpose(int *a, int n, int N) {  if (n <= 32) { for (int i = 0; i < n; i++) for (int j = 0; j < i; j++) swap(a[i * N + j], a[j * N + i]); } else { int k = n / 2;   transpose(a, k, N); transpose(a + k, k, N); transpose(a + k * N, k, N); transpose(a + k * N + k, k, N);  for (int i = 0; i < k; i++) for (int j = 0; j < k; j++) swap(a[i * N + (j + k)], a[(i + k) * N + j]);  if (n & 1) for (int i = 0; i < n - 1; i++) swap(a[i * N + n - 1], a[(n - 1) * N + i]); } } 
```

该算法的 I/O 复杂度为 $O(\frac{N²}{B})$，因为我们只需要在每次归并阶段触摸大约一半的内存块，这意味着在每个阶段我们的问题都变得更小。

将此代码适应非方阵的一般情况留给读者作为练习。

## [#](https://en.algorithmica.org/hpc/external-memory/oblivious/#matrix-multiplication)矩阵乘法

接下来，让我们考虑一个稍微复杂一些的问题：矩阵乘法。

$$ C_{ij} = \sum_k A_{ik} B_{kj} $$

原始算法只是将其定义直接转换成代码：

```cpp
// don't forget to initialize c[][] with zeroes for (int i = 0; i < n; i++)  for (int j = 0; j < n; j++) for (int k = 0; k < n; k++) c[i * n + j] += a[i * n + k] * b[k * n + j]; 
```

它需要访问总共 $O(N³)$ 个块，因为每个标量乘法都需要单独的块读取。

一个著名的优化是首先转置 $B$：

```cpp
for (int i = 0; i < n; i++)  for (int j = 0; j < i; j++) swap(b[j][i], b[i][j]) // ^ or use our faster transpose from before  for (int i = 0; i < n; i++)  for (int j = 0; j < n; j++) for (int k = 0; k < n; k++) c[i * n + j] += a[i * n + k] * b[j * n + k]; // <- note the indices 
```

不论是使用原始方法还是我们之前开发的缓存无关方法进行转置，当一个矩阵被转置后进行矩阵乘法，其时间复杂度会达到 $O(N³/B + N²)$，因为所有的内存访问现在都是顺序的。

看起来我们无法做得更好，但事实并非如此。

### [#](https://en.algorithmica.org/hpc/external-memory/oblivious/#algorithm)算法

缓存无关矩阵乘法依赖于与转置基本相同的技巧。我们需要将数据分割，直到它适合最低的缓存（即，$N² \leq M$）。对于矩阵乘法，这相当于使用以下公式：

$$ \begin{pmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \\ \end{pmatrix} \begin{pmatrix} B_{11} & B_{12} \\ B_{21} & B_{22} \\ \end{pmatrix} = \begin{pmatrix} A_{11} B_{11} + A_{12} B_{21} & A_{11} B_{12} + A_{12} B_{22}\\ A_{21} B_{11} + A_{22} B_{21} & A_{21} B_{12} + A_{22} B_{22}\\ \end{pmatrix} $$

尽管实现起来稍微困难一些，因为我们现在有总共 8 次递归矩阵乘法：

```cpp
void matmul(const float *a, const float *b, float *c, int n, int N) {  if (n <= 32) { for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) for (int k = 0; k < n; k++) c[i * N + j] += a[i * N + k] * b[k * N + j]; } else { int k = n / 2;   // c11 = a11 b11 + a12 b21 matmul(a,     b,         c, k, N); matmul(a + k, b + k * N, c, k, N);  // c12 = a11 b12 + a12 b22 matmul(a,     b + k,         c + k, k, N); matmul(a + k, b + k * N + k, c + k, k, N);  // c21 = a21 b11 + a22 b21 matmul(a + k * N,     b,         c + k * N, k, N); matmul(a + k * N + k, b + k * N, c + k * N, k, N);  // c22 = a21 b12 + a22 b22 mul(a + k * N,     b + k,         c + k * N + k, k, N); mul(a + k * N + k, b + k * N + k, c + k * N + k, k, N);   if (n & 1) { for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) for (int k = (i < n - 1 && j < n - 1) ? n - 1 : 0; k < n; k++) c[i * N + j] += a[i * N + k] * b[k * N + j]; } } } 
```

由于这里还有许多其他因素在起作用，我们不会对这个实现进行基准测试，而是只在外部内存模型中进行其理论性能分析。

### [#](https://en.algorithmica.org/hpc/external-memory/oblivious/#analysis)分析

算法的算术复杂度保持不变，因为递归

$$ T(N) = 8 \cdot T(N/2) + \Theta(N²) $$

通过 $T(N) = \Theta(N³)$ 得到解决。

看起来我们还没有“征服”什么，但让我们来考虑它的 I/O 复杂度：

$$ T(N) = \begin{cases} O(\frac{N²}{B}) & N \leq \sqrt M & \text{(我们只需要读取它)} \\ 8 \cdot T(N/2) + O(\frac{N²}{B}) & \text{否则} \end{cases} $$ 该递归主要由 $O((\frac{N}{\sqrt M})³)$ 的基本案例主导，意味着总复杂度是 $$ T(N) = O\left(\frac{(\sqrt{M})²}{B} \cdot \left(\frac{N}{\sqrt M}\right)³\right) = O\left(\frac{N³}{B\sqrt{M}}\right) $$

这比仅仅 $O(\frac{N³}{B})$ 要好，而且好得多。

### [#](https://en.algorithmica.org/hpc/external-memory/oblivious/#strassen-algorithm)Strassen 算法

在类似于 Karatsuba 算法的精神下，矩阵乘法可以被分解成涉及 7 次大小为 $\frac{n}{2}$ 的矩阵乘法，主定理告诉我们这样的分治算法的时间复杂度会是 $O(n^{\log_2 7}) \approx O(n^{2.81})$，在外部内存模型中也有类似的渐近复杂度。

这种技术被称为 Strassen 算法，同样将每个矩阵分成 4 部分：

$$ \begin{pmatrix} C_{11} & C_{12} \\ C_{21} & C_{22} \\ \end{pmatrix} =\begin{pmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \\ \end{pmatrix} \begin{pmatrix} B_{11} & B_{12} \\ B_{21} & B_{22} \\ \end{pmatrix} $$ 但随后它计算了$\frac{N}{2} \times \frac{N}{2}$矩阵的中间乘积并将它们组合以得到矩阵$C$：$$ \begin{aligned} M_1 &= (A_{11} + A_{22})(B_{11} + B_{22}) & C_{11} &= M_1 + M_4 - M_5 + M_7 \\ M_2 &= (A_{21} + A_{22}) B_{11} & C_{12} &= M_3 + M_5 \\ M_3 &= A_{11} (B_{21} - B_{22}) & C_{21} &= M_2 + M_4 \\ M_4 &= A_{22} (B_{21} - B_{11}) & C_{22} &= M_1 - M_2 + M_3 + M_6 \\ M_5 &= (A_{11} + A_{12}) B_{22} \\ M_6 &= (A_{21} - A_{11}) (B_{11} + B_{12}) \\ M_7 &= (A_{12} - A_{22}) (B_{21} + B_{22}) \end{aligned} $$

如果你愿意，可以通过简单的替换来验证这些公式。

就我所知，主流的优化线性代数库中没有使用斯特拉斯算法，尽管有一些[原型实现](https://arxiv.org/pdf/1605.01078.pdf)对于大于 2000 左右的矩阵是有效的。

这种技术可以，并且实际上已经多次扩展，通过考虑更多的子矩阵乘积来进一步降低渐进复杂度。截至 2020 年，当前的世界纪录是$O(n^{2.3728596})$。你能否在$O(n²)$或至少$O(n² \log^k n)$时间内乘法矩阵是一个未解决的问题。

## [#](https://en.algorithmica.org/hpc/external-memory/oblivious/#further-reading)进一步阅读

为了获得坚实的理论观点，可以考虑阅读 Erik Demaine 的[《Cache-Oblivious Algorithms and Data Structures》](https://erikdemaine.org/papers/BRICS2002/paper.pdf)。[←驱逐策略](https://en.algorithmica.org/hpc/external-memory/policies/)[空间和时间局部性→](https://en.algorithmica.org/hpc/external-memory/locality/)
