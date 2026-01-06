# 整数分解

> 原文：[`en.algorithmica.org/hpc/algorithms/factorization/`](https://en.algorithmica.org/hpc/algorithms/factorization/)

将整数分解为素数的问题是计算数论的核心。它至少从公元前 3 世纪以来就被[研究](https://www.cs.purdue.edu/homes/ssw/chapter3.pdf)，并且已经开发出许多[方法](https://en.wikipedia.org/wiki/Category:Integer_factorization_algorithms)，这些方法对不同的输入都是有效的。

在这个案例研究中，我们特别考虑的是*字长*整数的分解：那些在$10⁹$和$10^{18}$的数量级上。对于这本书来说并不典型，在这一章中，你实际上可以学习到一个渐近上更好的算法：我们从一个基本方法开始，并逐渐构建到$O(\sqrt[4]{n})$时间的*Pollard 的 rho 算法*，并将其优化到可以以 0.3-0.4ms 的时间分解 60 位的半素数，并且比之前的最先进技术快大约 3 倍。

### [#](https://en.algorithmica.org/hpc/algorithms/factorization/#benchmark)基准测试

对于所有方法，我们将实现一个`find_factor`函数，它接受一个正整数$n$并返回它的任何非平凡除数（如果该数是素数则返回`1`）：

```cpp
// I don't feel like typing "unsigned long long" each time typedef __uint16_t u16; typedef __uint32_t u32; typedef __uint64_t u64; typedef __uint128_t u128;   u64 find_factor(u64 n); 
```

要找到完整的分解，你可以将其应用于$n$，将其减少，并继续进行，直到无法找到新的因子：

```cpp
vector<u64> factorize(u64 n) {  vector<u64> factorization; do { u64 d = find_factor(n); factorization.push_back(d); n /= d; } while (d != 1); return factorization; } 
```

在每次移除因子之后，问题变得相当小，因此完整分解的最坏情况运行时间等于`find_factor`调用的最坏情况运行时间。

对于许多分解算法，包括本节中介绍的那些，运行时间与较小的素数因子成比例。因此，为了提供最坏情况的输入，我们使用*半素数*：两个相同数量级的素数$p \le q$的乘积。我们通过两个随机$\lfloor k / 2 \rfloor$-位素数的乘积来生成一个$k$位的半素数。

由于一些算法本质上是随机的，我们也可以容忍一小部分（小于 1%）的假阴性错误（当`find_factor`返回`1`而数字$n$是合数时），尽管这个比率可以通过不造成显著性能损失来降低到几乎为零。

### [#](https://en.algorithmica.org/hpc/algorithms/factorization/#trial-division)试除法

最基本的方法是尝试$n$以下的所有整数作为除数：

```cpp
u64 find_factor(u64 n) {  for (u64 d = 2; d < n; d++) if (n % d == 0) return d; return 1; } 
```

我们可以注意到，如果$n$被除以$d < \sqrt n$，那么它也会被除以$\frac{n}{d} > \sqrt n$，因此没有必要单独检查它。这让我们可以提前停止试除法，并且只需检查不超过$\sqrt n$的潜在除数：

```cpp
u64 find_factor(u64 n) {  for (u64 d = 2; d * d <= n; d++) if (n % d == 0) return d; return 1; } 
```

在我们的基准测试中，$n$是一个半素数，我们总是找到较小的除数，因此$O(n)$和$O(\sqrt n)$的实现表现相同，并且能够每秒分解大约 2k 个 30 位数的数——而分解一个单独的 60 位数则需要整整 20 秒。

### [#](https://en.algorithmica.org/hpc/algorithms/factorization/#lookup-table)查找表

现在，你可以在 Linux 终端或 Google 搜索栏中输入`factor 57`来获取任何数的因数分解。但在计算机发明之前，使用*因数分解表*更为实用：包含前 N 个数的因数分解的特殊书籍。

我们也可以使用这种方法在编译时间计算这些查找表。为了节省空间，我们只需要存储一个数的最小除数。由于最小的除数不超过$\sqrt n$，我们只需要为每个 16 位整数分配一个字节：

```cpp
template <int N = (1<<16)> struct Precalc {  unsigned char divisor[N];   constexpr Precalc() : divisor{} { for (int i = 0; i < N; i++) divisor[i] = 1; for (int i = 2; i * i < N; i++) if (divisor[i] == 1) for (int k = i * i; k < N; k += i) divisor[k] = i; } };   constexpr Precalc P{};   u64 find_factor(u64 n) {  return P.divisor[n]; } 
```

使用这种方法，我们每秒可以处理 300 万个 16 位整数，尽管对于更大的数字，它可能会变慢。虽然计算和存储前$2^{16}$个数的除数只需要几毫秒和 64KB 的内存，但它对于更大的输入扩展性不好。

### [#](https://en.algorithmica.org/hpc/algorithms/factorization/#wheel-factorization)轮式因数分解

为了节省纸张空间，计算机时代之前的因数分解表通常排除了能被 2 和 5 整除的数，使得因数分解表的大小减少了 0.4 倍。在十进制数制中，你可以快速判断一个数是否能被 2 或 5 整除（通过查看它的最后一位数字），并在可能的情况下，将数 n 除以 2 或 5，最终到达因数分解表中的某个条目。

我们可以通过首先检查一个数是否能被 2 整除，然后只考虑奇数除数来应用类似的技巧进行试除法：

```cpp
u64 find_factor(u64 n) {  if (n % 2 == 0) return 2; for (u64 d = 3; d * d <= n; d += 2) if (n % d == 0) return d; return 1; } 
```

由于需要执行的除法次数减少了 50%，这个算法的速度快了一倍。

这种方法可以扩展：如果数不能被 3 整除，我们也可以忽略所有 3 的倍数，对于所有其他除数也是如此。问题是，随着要排除的质数的增加，迭代仅遍历不能被它们整除的数变得更加不直接，因为它们遵循不规则的模式——除非质数的数量很少。

例如，如果我们考虑 2、3 和 5，那么在最初的 90 个数字中，我们只需要检查：

```cpp
(1,) 7, 11, 13, 17, 19, 23, 29,
31, 37, 41, 43, 47, 49, 53, 59,
61, 67, 71, 73, 77, 79, 83, 89…

```

你可以注意到一个模式：序列每 30 个数重复一次。这并不奇怪，因为余数模$2 \times 3 \times 5 = 30$就是我们需要用来确定一个数是否能被 2、3 或 5 整除的所有余数。这意味着我们只需要检查每 30 个数中具有特定余数的 8 个数，从而按比例提高性能：

```cpp
u64 find_factor(u64 n) {  for (u64 d : {2, 3, 5}) if (n % d == 0) return d; u64 offsets[] = {0, 4, 6, 10, 12, 16, 22, 24}; for (u64 d = 7; d * d <= n; d += 30) { for (u64 offset : offsets) { u64 x = d + offset; if (n % x == 0) return x; } } return 1; } 
```

如预期的那样，它比简单的试除法快 3.75 倍，每秒处理约 7.6k 个 30 位数字。通过考虑更多的质数，性能可以进一步提高，但回报正在减少：添加一个新的质数 p 可以减少迭代次数的 1/p，但将跳转列表的大小增加 p 倍，需要成比例更多的内存。

### [#](https://en.algorithmica.org/hpc/algorithms/factorization/#precomputed-primes)预计算质数

如果我们继续增加轮分解中的质数数量，最终可以排除所有合数，并只检查质数因子。在这种情况下，我们不需要这个偏移数组，只需要质数数组：

```cpp
const int N = (1 << 16);   struct Precalc {  u16 primes[6542]; // # of primes under N=2¹⁶  constexpr Precalc() : primes{} { bool marked[N] = {}; int n_primes = 0;   for (int i = 2; i < N; i++) { if (!marked[i]) { primes[n_primes++] = i; for (int j = 2 * i; j < N; j += i) marked[j] = true; } } } };   constexpr Precalc P{};   u64 find_factor(u64 n) {  for (u16 p : P.primes) if (n % p == 0) return p; return 1; } 
```

这种方法使我们能够每秒处理近 20k 个 30 位整数，但它不适用于更大的（64 位）数，除非它们有小的（$< 2^{16}$）因子。

注意，这实际上是一种渐近优化：在最初的$n$个数中，有$O(\frac{n}{\ln n})$个质数，所以这个算法执行$O(\frac{\sqrt n}{\ln \sqrt n})$次操作，而轮分解只消除了一部分但常数大的除数。如果我们将其扩展到 64 位数，并预先计算$2^{32}$以下的每个质数（存储这需要几百兆字节的内存），相对速度将增加一个因子$\frac{\ln \sqrt{n²}}{\ln \sqrt n} = 2 \cdot \frac{1/2}{1/2} \cdot \frac{\ln n}{\ln n} = 2$。

所有试除法的变体，包括这个，都受整数除法速度的限制，如果我们事先知道除数并允许一些额外的预计算，则可以对其进行优化。Lemire 除法检查是合适的：

```cpp
// ...precomputation is the same as before, // but we store the reciprocal instead of the prime number itself u64 magic[6542]; // for each prime i: magic[n_primes++] = u64(-1) / i + 1;   u64 find_factor(u64 n) {  for (u64 m : P.magic) if (m * n < m) return u64(-1) / m + 1; return 1; } 
```

这使得算法的速度提高了约 18 倍：我们现在每秒可以分解**约 350k**个 30 位数的数，这实际上是我们对这个数范围内最有效的算法。虽然通过并行执行这些检查与 SIMD 可能进一步优化，但我们将在那里停止，并尝试不同的、渐近上更好的方法。

### [#](https://en.algorithmica.org/hpc/algorithms/factorization/#pollards-rho-algorithm)Pollard 的ρ算法

Pollard 的ρ算法是一种随机化的$O(\sqrt[4]{n})$整数分解算法，它利用了[生日悖论](https://en.wikipedia.org/wiki/Birthday_problem)：

> 只需要从 1 到$n$之间抽取$d = \Theta(\sqrt{n})$个随机数，以高概率得到一个冲突。

其背后的推理是，每个增加的元素$d$有$\frac{d}{n}$的机会与其他元素冲突，这意味着预期的冲突数是$\frac{d²}{n}$。如果$d$渐近小于$\sqrt n$，那么这个比率随着$n \to \infty$增长到零，否则增长到无穷大。

考虑某个函数$f(x)$，它将余数$x \in 0, n)$映射到$n$的另一个余数，从数论的角度来看，这似乎是随机的。具体来说，我们将使用$f(x) = x² + 1 \bmod n$，这对于我们的目的来说足够随机。

现在，考虑一个图，其中每个数字顶点 $x$ 都有一条指向 $f(x)$ 的边。这样的图被称为*函数图*。在函数图中，任何元素的“轨迹”——如果我们从这个元素开始并继续跟随边，我们所走的路径——最终会形成一个循环（因为顶点集是有限的，在某个时刻，我们必须回到已经访问过的顶点）。

![图片

一个元素的轨迹类似于希腊字母 ρ（rho），这也是算法的名字由来。

考虑某个特定元素 $x_0$ 的轨迹：

$$ x_0, \; f(x_0), \; f(f(x_0)), \; \ldots $$

让我们从这个序列中再构造一个序列，通过将每个元素对 $ p $ 取模，其中 $ p $ 是 $ n $ 的最小质数除数。

**引理。** 在序列变成循环之前，其期望长度是 $ O(\sqrt[4]{n}) $。

**证明：** 由于 $ p $ 是最小的除数，$ p \leq \sqrt n $。每次我们跟随一条新边，实际上我们生成一个介于 $ 0 $ 和 $ p $ 之间的随机数（我们将 $ f $ 视为一个“确定性的随机”函数）。生日悖论指出，我们只需要生成 $ O(\sqrt p) = O(\sqrt[4]{n}) $ 个数字直到我们得到一个碰撞并因此进入循环。

由于我们不知道 $ p $，这个模 $-p$ 序列只是想象中的，但如果在这个序列中找到一个循环——即 $ i $ 和 $ j $ 使得

如果 $ f^i(x_0) \equiv f^j(x_0) \pmod p $，那么我们也可以找到 $ p $ 本身，即 $$ p = \gcd(|f^i(x_0) - f^j(x_0)|, n) $$ 算法本身只是使用这个最大公约数（GCD）技巧和 Floyd 的“[龟兔赛跑](https://en.wikipedia.org/wiki/Cycle_detection#Floyd's_tortoise_and_hare)”算法找到这个循环和 $ p $：我们维护两个指针 $ i $ 和 $ j = 2i $ 并检查 $$ \gcd(|f^i(x_0) - f^j(x_0)|, n) \neq 1 $$

这相当于比较 $f^i(x_0)$ 和 $f^j(x_0)$ 在模 $p$ 下的值。由于 $j$（兔子）的增长速度是 $i$（乌龟）的两倍，它们的差值在每次迭代中增加 $1$，最终将等于（或为）循环长度，此时 $i$ 和 $j$ 指向相同的元素。正如我们半个页面前所证明的，达到循环只需要 $O(\sqrt[4]{n})$ 次迭代：

```cpp
u64 f(u64 x, u64 mod) {  return ((u128) x * x + 1) % mod; }   u64 diff(u64 a, u64 b) {  // a and b are unsigned and so is their difference, so we can't just call abs(a - b) return a > b ? a - b : b - a; }   const u64 SEED = 42;   u64 find_factor(u64 n) {  u64 x = SEED, y = SEED, g = 1; while (g == 1) { x = f(f(x, n), n); // advance x twice y = f(y, n);       // advance y once g = gcd(diff(x, y)); } return g; } 
```

虽然它只处理大约 ~25k 个 30 位整数——这几乎是通过快速除法技巧检查每个质数速度的 15 倍慢——但它显著优于所有 $\tilde{O}(\sqrt n)$ 的 60 位数字算法，每秒分解大约 90 个。

### [#](https://en.algorithmica.org/hpc/algorithms/factorization/#pollard-brent-algorithm)Pollard-Brent 算法

Floyd 的循环查找算法有一个问题，即它移动迭代器比必要的多：至少有一半的顶点被较慢的迭代器多访问了一次。

解决这个问题的一种方法是将快速迭代器访问的值$x_i$记住，并且每两次迭代，就使用$x_i$和$x_{\lfloor i / 2 \rfloor}$的差来计算 GCD。但也可以使用不同的原理而不需要额外的内存：乌龟不是在每次迭代中都移动，而是在迭代次数成为 2 的幂时重置为快速迭代器的值。这样，我们可以在保持使用相同的 GCD 技巧来比较$x_i$和$x_{2^{\lfloor \log_2 i \rfloor}}$的同时，节省额外的迭代次数：

```cpp
u64 find_factor(u64 n) {  u64 x = SEED;  for (int l = 256; l < (1 << 20); l *= 2) { u64 y = x; for (int i = 0; i < l; i++) { x = f(x, n); if (u64 g = gcd(diff(x, y), n); g != 1) return g; } }   return 1; } 
```

注意，我们还设置了迭代的上限，以确保算法在合理的时间内完成，并且如果$n$最终被证明是素数，则返回`1`。

实际上，这并没有提高性能，甚至使算法慢了约 1.5 倍，这可能与$x$过时有关。它大部分时间都在计算 GCD，而没有推进迭代器——实际上，由于这个原因，该算法当前的时间复杂度是$O(\sqrt[4]{n} \log n)$。

我们将优化 GCD 调用的次数，而不是直接优化 GCD 本身（见../gcd）。我们可以利用以下事实：如果$a$和$b$中的一个包含因子$p$，那么$a \cdot b \bmod n$也将包含它，因此我们不需要计算$\gcd(a, n)$和$\gcd(b, n)$，而是可以计算$\gcd(a \cdot b \bmod n, n)$。这样，我们可以将 GCD 的计算分组为$M = O(\log n)$，从而从渐近式中移除$\log n$：

```cpp
const int M = 1024;   u64 find_factor(u64 n) {  u64 x = SEED;  for (int l = M; l < (1 << 20); l *= 2) { u64 y = x, p = 1; for (int i = 0; i < l; i += M) { for (int j = 0; j < M; j++) { y = f(y, n); p = (u128) p * diff(x, y) % n; } if (u64 g = gcd(p, n); g != 1) return g; } }   return 1; } 
```

现在它每秒执行 425 次因式分解，瓶颈在于模数的速度。

### [#](https://en.algorithmica.org/hpc/algorithms/factorization/#optimizing-the-modulo)优化模数

最后一步是应用 Montgomery 乘法。由于模数是常数，我们可以在 Montgomery 空间中执行所有计算——推进迭代器、乘法，甚至计算 GCD——因为在这个空间中减少是便宜的：

```cpp
struct Montgomery {  u64 n, nr;  Montgomery(u64 n) : n(n) { nr = 1; for (int i = 0; i < 6; i++) nr *= 2 - n * nr; }   u64 reduce(u128 x) const { u64 q = u64(x) * nr; u64 m = ((u128) q * n) >> 64; return (x >> 64) + n - m; }   u64 multiply(u64 x, u64 y) { return reduce((u128) x * y); } };   u64 f(u64 x, u64 a, Montgomery m) {  return m.multiply(x, x) + a; }   const int M = 1024;   u64 find_factor(u64 n, u64 x0 = 2, u64 a = 1) {  Montgomery m(n); u64 x = SEED;  for (int l = M; l < (1 << 20); l *= 2) { u64 y = x, p = 1; for (int i = 0; i < l; i += M) { for (int j = 0; j < M; j++) { x = f(x, m); p = m.multiply(p, diff(x, y)); } if (u64 g = gcd(p, n); g != 1) return g; } }   return 1; } 
```

此实现每秒可以处理大约 3k 个 60 位整数，这比[PARI](https://pari.math.u-bordeaux.fr/) / [SageMath 的`factor`](https://doc.sagemath.org/html/en/reference/structure/sage/structure/factorization.html) / `cat semiprimes.txt | time factor`测量的速度要快约 3 倍。

### [#](https://en.algorithmica.org/hpc/algorithms/factorization/#further-improvements)进一步改进

**优化。** 在我们实现的 Pollard 算法中，仍有大量的优化潜力：

+   我们可能可以使用更好的循环查找算法，利用图是随机的这一事实。例如，我们进入循环的可能性很小（循环的长度和进入循环之前走过的路径的长度在期望上应该是相等的，因为我们进入循环之前，我们独立地选择走过的路径的顶点），所以我们可能只需要在开始使用 GCD 技巧进行试验之前推进迭代器一段时间。

+   我们当前的方法受限于迭代器的推进（Montgomery 乘法的延迟远高于其吞吐量），在等待其完成的同时，我们可以使用之前的数据进行不止一次的试验。

+   如果我们并行运行具有不同种子的 $p$ 个独立算法实例，并在其中一个找到答案时停止，它将快 $\sqrt p$ 倍（推理类似于生日悖论；试着证明它）。我们不需要使用多个核心：有很多未被挖掘的指令级并行性，因此我们可以在同一线程上并发运行两个或三个相同的操作，或者使用 SIMD 指令并行执行 4 或 8 次乘法。

我不会对看到 3 倍改进和大约 10k/sec 的吞吐量感到惊讶。如果你[实现](https://github.com/sslotin/amh-code/tree/main/factor)了这些想法中的某些，请[告诉我](http://sereja.me/)。

**错误。**在实用实现中，我们还需要处理可能的错误。我们当前的实现对于 60 位整数有 0.7%的错误率，如果数字更小，错误率会更高。这些错误主要来自三个来源：

+   简单地没有找到周期（算法本质上是随机的，没有保证一定会找到）。在这种情况下，我们需要执行素性测试，并可选择重新开始。

+   `p` 变量变为零（因为 $p$ 和 $q$ 都可能进入乘积）。随着我们减小输入的大小或增加常数 `M`，这种情况变得越来越可能。在这种情况下，我们需要重新启动过程，或者（更好）回滚最后 $M$ 次迭代并逐一进行试验。

+   Montgomery 乘法中的溢出。我们当前的实现对这些比较宽松，如果 $n$ 很大，我们需要添加更多的 `x > mod ? x - mod : x` 类型的语句来处理溢出。

**更大的数字。**如果我们排除使用我们之前实现的算法的小数和具有小素因子的数，这些问题就变得不那么重要了。一般来说，最佳方法应该取决于数字的大小：

+   小于 $2^{16}$：使用查找表；

+   小于 $2^{32}$：使用带有快速可除性检查的预计算素数列表；

+   小于 $2^{64}$ 或左右：使用带有 Montgomery 乘法的 Pollard 的 rho 算法；

+   小于 $10^{50}$：切换到[Lenstra 椭圆曲线分解](https://en.wikipedia.org/wiki/Lenstra_elliptic-curve_factorization)；

+   小于 $10^{100}$：切换到[二次筛法](https://en.wikipedia.org/wiki/Quadratic_sieve)；

+   大于 $10^{100}$：切换到[通用数域筛法](https://en.wikipedia.org/wiki/General_number_field_sieve)。

最后三种方法与我们之前所做的方法非常不同，需要更多的高级数论知识，它们值得一篇（或一整门大学课程）的专门文章。[← 二进制最大公约数](https://en.algorithmica.org/hpc/algorithms/gcd/)[使用 SIMD 求最小值 →](https://en.algorithmica.org/hpc/algorithms/argmin/)
