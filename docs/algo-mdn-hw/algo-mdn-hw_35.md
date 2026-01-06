# 快速逆平方根

> 原文：[`en.algorithmica.org/hpc/arithmetic/rsqrt/`](https://en.algorithmica.org/hpc/arithmetic/rsqrt/)

浮点数 $\frac{1}{\sqrt x}$ 的逆平方根用于计算归一化向量，而这些向量又广泛应用于各种模拟场景，例如计算机图形学（例如，用于确定入射角和反射角以模拟光照）。$$ \hat{v} = \frac{\vec v}{\sqrt {v_x² + v_y² + v_z²}} $$

直接计算逆平方根——首先计算平方根，然后除以 $1$——非常慢，因为这两个操作即使是在硬件中实现也是慢的。

但有一个出人意料的好近似算法，它利用了浮点数在内存中存储的方式。事实上，它非常好，以至于它已经被 [硬件实现](https://www.felixcloutier.com/x86/rsqrtps)，因此算法本身对于软件工程师来说不再相关，但我们仍然要探讨它，因为它具有内在的美感和巨大的教育价值。

除了方法本身之外，其创建的历史也非常有趣。它归功于一家游戏工作室 *id Software*，他们在 1999 年的标志性游戏 *Quake III Arena* 中使用了它，尽管显然，它是通过一系列“我从一个人那里学到了它，那个人又从另一个人那里学到了它”的过程得到的，似乎最终结束在威廉·卡汉（William Kahan）身上（同一个人负责 IEEE 754 和卡汉求和算法）。

它在 2005 年左右在游戏开发社区中变得流行，当时他们发布了游戏的源代码。以下是 [其中的相关摘录](https://github.com/id-Software/Quake-III-Arena/blob/master/code/game/q_math.c#L552)，包括注释：

```cpp
float Q_rsqrt(float number) {  long i; float x2, y; const float threehalfs = 1.5F;   x2 = number * 0.5F; y  = number; i  = * ( long * ) &y;                       // evil floating point bit level hacking i  = 0x5f3759df - ( i >> 1 );               // what the fuck? y  = * ( float * ) &i; y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration //  y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed  return y; } 
```

我们将逐步解释它是如何工作的，但首先，我们需要稍微绕个弯。

### [#](https://en.algorithmica.org/hpc/arithmetic/rsqrt/#approximate-logarithm)近似对数

在计算机（或至少是负担得起的计算器）成为日常用品之前，人们使用对数表来计算乘法及相关操作——通过查找 $a$ 和 $b$ 的对数，将它们相加，然后找到结果的逆对数。

$$ a \times b = 10^{\log a + \log b} = \log^{-1}(\log a + \log b) $$ 当使用恒等式计算 $\frac{1}{\sqrt x}$ 时，你也可以做同样的技巧：$$ \log \frac{1}{\sqrt x} = - \frac{1}{2} \log x $$

快速逆平方根基于这个恒等式，因此它需要非常快速地计算 $x$ 的对数。结果发现，它可以通过仅仅将 32 位 `float` 重新解释为整数来近似。

回忆 浮点数按顺序存储符号位（对于正值等于零，即我们的情况）、指数 $e_x$ 和尾数 $m_x$，这对应于

$$ x = 2^{e_x} \cdot (1 + m_x) $$ 它的对数因此是 $$ \log_2 x = e_x + \log_2 (1 + m_x) $$ 由于 $m_x \in 0, 1)$，右侧的对数可以通过 $$ \log_2 (1 + m_x) \approx m_x $$ 来近似。近似在区间的两端是精确的，但为了考虑平均情况，我们需要通过一个小常数 $\sigma$ 来移动它，因此 $$ \log_2 x = e_x + \log_2 (1 + m_x) \approx e_x + m_x + \sigma $$ 现在，考虑到这个近似并定义 $L=2^{23}$（`float` 中的尾数位数）和 $B=127$（指数偏差），当我们将 $x$ 的位模式重新解释为整数 $I_x$ 时，我们实际上得到 $$ \begin{aligned} I_x &= L \cdot (e_x + B + m_x) \\ &= L \cdot (e_x + m_x + \sigma +B-\sigma ) \\ &\approx L \cdot \log_2 (x) + L \cdot (B-\sigma ) \end{aligned} $$

（将一个整数乘以 $L=2^{23}$ 等同于将其左移 23 位。）

当你调整 $\sigma$ 以最小化均方误差时，这会导致一个令人惊讶的准确近似。

![

将浮点数 $x$ 重新解释为整数（蓝色）与其缩放和移位的对数（灰色）相比

现在，从近似中表达对数，我们得到

$$ \log_2 x \approx \frac{I_x}{L} - (B - \sigma) $$

很酷。现在，我们刚才在哪里？哦，是的，我们想要计算平方根的倒数。

### [#](https://en.algorithmica.org/hpc/arithmetic/rsqrt/#approximating-the-result)结果近似

要使用恒等式 $\log_2 y = - \frac{1}{2} \log_2 x$ 来计算 $y = \frac{1}{\sqrt x}$，我们可以将其代入我们的近似公式，得到

$$ \frac{I_y}{L} - (B - \sigma) \approx - \frac{1}{2} ( \frac{I_x}{L} - (B - \sigma) ) $$ 解出 $I_y$：$$ I_y \approx \frac{3}{2} L (B - \sigma) - \frac{1}{2} I_x $$

结果表明，我们甚至不需要最初计算对数：上面的公式只是一个常数减去 $x$ 的整数重新解释的一半。它在代码中写成：

```cpp
i = * ( long * ) &y; i = 0x5f3759df - ( i >> 1 ); 
```

我们在第一行将 `y` 重新解释为整数，然后将其代入第二行的公式，其中第一个项是魔法数 $\frac{3}{2} L (B - \sigma) = \mathtt{0x5F3759DF}$，而第二个项是通过二进制移位而不是除法来计算的。

### [#](https://en.algorithmica.org/hpc/arithmetic/rsqrt/#iterating-with-newtons-method)使用牛顿法迭代

接下来，我们有一对用牛顿法编写的迭代，其中 $f(y) = \frac{1}{y²} - x$ 和一个非常好的初始值。其更新规则是

$$ f'(y) = - \frac{2}{y³} \implies y_{i+1} = y_{i} (\frac{3}{2} - \frac{x}{2} y_i²) = \frac{y_i (3 - x y_i²)}{2} $$

这在代码中写成

```cpp
x2 = number * 0.5F; y  = y * ( threehalfs - ( x2 * y * y ) ); 
```

初始近似值非常准确，以至于仅一次迭代就足以满足游戏开发的需求。在第一次迭代后，它就落在了正确答案的 99.8%范围内，并且可以通过进一步迭代来提高精度——这正是硬件所做的事情：[x86 指令](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig_expand=3037,3009,5135,4870,4870,4872,4875,833,879,874,849,848,6715,4845,6046,3853,288,6570,6527,6527,90,7307,6385,5993&text=rsqrt&techs=AVX,AVX2)执行其中的一些操作，并保证相对误差不超过$1.5 \times 2^{-12}$。

### [进一步阅读](https://en.algorithmica.org/hpc/arithmetic/rsqrt/#further-reading)

[关于快速逆平方根的维基百科文章](https://en.wikipedia.org/wiki/Fast_inverse_square_root#Floating-point_representation). [← 牛顿法](https://en.algorithmica.org/hpc/arithmetic/newton/)[整数数字 →](https://en.algorithmica.org/hpc/arithmetic/integer/)
