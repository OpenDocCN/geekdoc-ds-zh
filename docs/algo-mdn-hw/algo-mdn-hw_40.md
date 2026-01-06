# 二进制指数运算

> 原文：[`en.algorithmica.org/hpc/number-theory/exponentiation/`](https://en.algorithmica.org/hpc/number-theory/exponentiation/)

在模运算（以及计算代数的一般情况）中，你经常需要将一个数提高到 $n$ 次幂——为了进行模除法，执行素性测试，或者计算某些组合值——而你通常希望花费少于 $\Theta(n)$ 次操作来计算它。

*二进制指数运算*，也称为*平方根指数运算*，是一种允许使用 $O(\log n)$ 次乘法来计算 $n$ 次幂的方法，依赖于以下观察：

$$ \begin{aligned} a^{2k} &= (a^k)² \\ a^{2k + 1} &= (a^k)² \cdot a \end{aligned} $$ 为了计算 $a^n$，我们可以递归地计算 $a^{\lfloor n / 2 \rfloor}$，然后平方它，如果 $n$ 是奇数，则可选地乘以 $a$，对应以下递归：$$ a^n = f(a, n) = \begin{cases} 1, && n = 0 \\ f(a, \frac{n}{2})², && 2 \mid n \\ f(a, n - 1) \cdot a, && 2 \nmid n \end{cases} $$

由于 $n$ 在每次递归转换中至少减半，这个递归的深度和总乘法次数将最多为 $O(\log n)$。

### [#](https://en.algorithmica.org/hpc/number-theory/exponentiation/#recursive-implementation)递归实现

由于我们已经有了一个递归，很自然地，我们将算法实现为一个匹配情况的递归函数：

```cpp
const int M = 1e9 + 7; // modulo typedef unsigned long long u64;   u64 binpow(u64 a, u64 n) {  if (n == 0) return 1; if (n % 2 == 1) return binpow(a, n - 1) * a % M; else { u64 b = binpow(a, n / 2); return b * b % M; } } 
```

在我们的基准测试中，我们使用 $n = m - 2$，这样我们就可以计算 $a$ 模 $m$ 的乘法逆元：

```cpp
u64 inverse(u64 a) {  return binpow(a, M - 2); } 
```

我们使用 $m = 10⁹+7$，这是一个在组合问题中计算校验和常用的模值，用于竞技编程——因为它是一个质数（允许通过二进制指数运算进行逆运算），足够大，不会在加法中溢出 `int`，不会在乘法中溢出 `long long`，并且容易输入为 `1e9 + 7`。

由于我们在代码中将它用作编译时常数，编译器可以优化模运算，通过将其替换为乘法（即使它不是编译时常数，通过手动计算一次魔法常数并用于快速化简仍然更便宜）。

执行路径——以及因此的运行时间——取决于 $n$ 的值。对于这个特定的 $n$，基线实现每次调用大约需要 330 纳秒。由于递归引入了一些开销，将其实现展开为迭代过程是有意义的。

### [#](https://en.algorithmica.org/hpc/number-theory/exponentiation/#iterative-implementation)迭代实现

$a^n$ 的结果可以表示为 $a$ 的某些 2 的幂的乘积——那些对应于 $n$ 的二进制表示中的 1 的幂。例如，如果 $n = 42 = 32 + 8 + 2$，那么

$$ a^{42} = a^{32+8+2} = a^{32} \cdot a⁸ \cdot a² $$

为了计算这个乘积，我们可以遍历$n$的位，同时维护两个变量：$a^{2^k}$的值和考虑$n$的$k$个最低位后的当前乘积。在每一步中，如果$n$的第$k$位被设置，则将当前乘积乘以$a^{2^k}$，并且在任何情况下，将$a^k$平方以得到$a^{2^k \cdot 2} = a^{2^{k+1}}$，这将用于下一次迭代。

```cpp
u64 binpow(u64 a, u64 n) {  u64 r = 1;  while (n) { if (n & 1) r = res * a % M; a = a * a % M; n >>= 1; }  return r; } 
```

迭代实现的每次调用大约需要 180 纳秒。繁重的计算保持不变；改进主要来自于减少的依赖链：`a = a * a % M`需要在循环继续之前完成，而现在它可以与`r = res * a % M`并发执行。

性能还受益于$n$是一个常数，使得所有分支可预测，并让调度器提前知道需要执行什么。然而，编译器并没有利用这一点，也没有展开`while(n) n >>= 1`循环。我们可以将其重写为一个执行 30 次常量迭代的`for`循环：

```cpp
u64 inverse(u64 a) {  u64 r = 1;  #pragma GCC unroll(30) for (int l = 0; l < 30; l++) { if ( (M - 2) >> l & 1 ) r = r * a % M; a = a * a % M; }   return r; } 
```

这迫使编译器只生成我们需要的指令，从而再节省 10 纳秒，使总运行时间约为 170 纳秒。

注意，性能不仅取决于$n$的二进制长度，还取决于二进制中 1 的数量。如果$n$是$2^{30}$，由于我们不需要执行任何旁路乘法，所以大约可以节省 20 纳秒。[←模运算](https://en.algorithmica.org/hpc/number-theory/modular/)[扩展欧几里得算法→](https://en.algorithmica.org/hpc/number-theory/euclid-extended/)
