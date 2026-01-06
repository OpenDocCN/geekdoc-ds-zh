# 扩展欧几里得算法

> 原文：[`en.algorithmica.org/hpc/number-theory/euclid-extended/`](https://en.algorithmica.org/hpc/number-theory/euclid-extended/)

费马定理允许我们通过 $O(\log n)$ 次操作中的二进制指数运算来计算模乘法逆元，但它只适用于素数模数。存在它的一个推广，[欧拉定理](https://en.wikipedia.org/wiki/Euler%27s_theorem)，指出如果 $m$ 和 $a$ 互质，则 $$ a^{\phi(m)} \equiv 1 \pmod m $$

其中 $\phi(m)$ 是 [欧拉函数](https://en.wikipedia.org/wiki/Euler%27s_totient_function)，定义为小于 $m$ 的正整数 $x$ 中与 $m$ 互质的数的个数。在 $m$ 是素数的情况下，所有 $m - 1$ 个余数都是互质的，因此 $\phi(m) = m - 1$，从而得到费马定理。

这使得我们可以通过知道 $\phi(m)$ 来计算 $a$ 的逆元为 $a^{\phi(m) - 1}$，但反过来计算它并不快：通常需要获得 $m$ 的因式分解来执行它。存在一个更通用的方法，它通过修改欧几里得算法来实现。

### [#](https://en.algorithmica.org/hpc/number-theory/euclid-extended/#algorithm)算法

*扩展欧几里得算法*，除了找到 $g = \gcd(a, b)$，还找到整数 $x$ 和 $y$，使得

$$ a \cdot x + b \cdot y = g $$ 这就解决了在将 $b$ 替换为 $m$ 和 $g$ 替换为 $1$ 的情况下寻找模逆元的问题：$$ a^{-1} \cdot a + k \cdot m = 1 $$

注意，如果 $a$ 与 $m$ 不互质，则没有解，因为 $a$ 和 $m$ 的任何整数组合都无法得到不是它们最大公约数倍数的任何数。

算法也是递归的：它计算 $\gcd(b, a \bmod b)$ 的系数 $x’$ 和 $y’$，并恢复原始数对的解。如果我们有一个数对 $(b, a \bmod b)$ 的解 $(x’， y’)$

$$ b \cdot x' + (a \bmod b) \cdot y' = g $$ 那么，为了得到初始输入的解，我们可以将 $(a \bmod b)$ 重新写为 $(a - \lfloor \frac{a}{b} \rfloor \cdot b)$ 并将其代入上述方程：$$ b \cdot x' + (a - \Big \lfloor \frac{a}{b} \Big \rfloor \cdot b) \cdot y' = g $$ 现在我们重新排列项，按 $a$ 和 $b$ 分组，得到 $$ a \cdot \underbrace{y'}_x + b \cdot \underbrace{(x' - \Big \lfloor \frac{a}{b} \Big \rfloor \cdot y')}_y = g $$

将其与初始表达式进行比较，我们可以推断出我们可以直接使用 $a$ 和 $b$ 的系数作为初始的 $x$ 和 $y$。

### [#](https://en.algorithmica.org/hpc/number-theory/euclid-extended/#implementation)实现

我们将算法实现为一个递归函数。由于它的输出不是单个整数，而是三个整数，所以我们通过引用传递系数：

```cpp
int gcd(int a, int b, int &x, int &y) {
    if (a == 0) {
        x = 0;
        y = 1;
        return b;
    }
    int x1, y1;
    int d = gcd(b % a, a, x1, y1);
    x = y1 - (b / a) * x1;
    y = x1;
    return d;
} 
```

要计算逆元，我们只需传递$a$和$m$，然后返回算法找到的$x$系数。由于我们传递了两个正数，其中一个系数将是正的，另一个是负的（哪个是负的取决于迭代次数是奇数还是偶数），因此我们需要可选地检查$x$是否为负，并加上$m$以获得正确的余数：

```cpp
int inverse(int a) {
    int x, y;
    gcd(a, M, x, y);
    if (x < 0)
        x += M;
    return x;
} 
```

它在 160 纳秒内完成——比使用二进制指数运算求逆数快 10 纳秒。为了进一步优化它，我们可以将其转换为迭代形式——这需要 135 纳秒：

```cpp
int inverse(int a) {
    int b = M, x = 1, y = 0;
    while (a != 1) {
        y -= b / a * x;
        b %= a;
        swap(a, b);
        swap(x, y);
    }
    return x < 0 ? x + M : x;
} 
```

注意，与二进制指数运算不同，运行时间取决于$a$的值。例如，对于这个特定的$m$值（$10⁹ + 7$），最坏的情况是 564400443，算法执行了 37 次迭代，耗时 250 纳秒。

**练习**。尝试将相同的技巧应用于二进制最大公约数（除非你在优化方面比我更好，否则它不会带来性能提升）。[← 二进制指数运算](https://en.algorithmica.org/hpc/number-theory/exponentiation/)[蒙哥马利乘法 →](https://en.algorithmica.org/hpc/number-theory/montgomery/)
