# 浮点数

> 原文：[`en.algorithmica.org/hpc/arithmetic/float/`](https://en.algorithmica.org/hpc/arithmetic/float/)

浮点运算的用户应该得到一个这样的智商钟形曲线图——因为这就是它与大多数人通常的关系：

+   初级程序员到处使用它，就像它是某种魔法无限精度数据类型一样。

+   然后他们发现 `0.1 + 0.2 != 0.3` 或者类似的其他怪异现象，感到恐慌，开始认为每次计算都会添加一些随机的误差项，并且多年来完全避免使用任何真实的数据类型。

+   然后他们最终鼓起勇气，阅读了 IEEE-754 浮点数的工作原理规范，并开始适当地使用它们。

不幸的是，太多的人仍然处于第二阶段，产生了关于浮点运算的各种误解——认为它本质上是不精确和不稳定的，并且比整数运算慢。

![](img/fbc618b7fb6320b8680a6a63a76c34a3.png)

但这些都只是神话。由于专门的指令，浮点运算通常比整数运算要快，并且实数表示在舍入方面彻底标准化，遵循简单和确定的规则，允许你可靠地管理计算误差。

事实上，它如此可靠，以至于一些高级编程语言，最著名的是 JavaScript，根本不提供整数类型。在 JavaScript 中，只有一个 `number` 类型，它内部以 64 位 `double` 存储的，由于浮点运算的工作方式，所有介于 $-2^{53}$ 和 $2^{53}$ 之间的整数以及涉及它们的计算结果都可以精确存储，因此从程序员的视角来看，几乎没有实际需要单独的整数类型。

一个显著的例外是当你需要使用数字进行位运算时，通常不支持这种操作的 *浮点单元*（负责浮点数运算的协处理器）通常不支持。在这种情况下，它们需要被转换为整数，这在 JavaScript 兼容的浏览器中非常频繁地使用，以至于 arm [添加了一个特殊的“FJCVTZS”指令](https://community.arm.com/developer/ip-products/processors/b/processors-ip-blog/posts/armv8-a-architecture-2016-additions)，代表“将浮点 JavaScript 转换为有符号定点，舍入到零”，并且确实按照它所说的那样做——将实数精确转换为整数，就像 JavaScript 一样——这是软件-硬件反馈循环中一个有趣的例子。

但除非你是使用真实类型专门模拟整数运算的 JavaScript 开发者，否则你可能需要一个更深入的关于浮点运算的指南，因此我们将从更广泛的主题开始。

## [#](https://en.algorithmica.org/hpc/arithmetic/float/#real-number-representations)实数表示

如果你需要处理实数（非整数）数字，你有几种选择，适用性各不相同。在直接跳到浮点数之前，这是本文的大部分内容，我们想要讨论可用的替代方案以及背后的动机——毕竟，避免浮点运算的人确实有他们的理由。

### [#](https://en.algorithmica.org/hpc/arithmetic/float/#symbolic-expressions) 符号表达式

第一种也是最繁琐的方法是存储产生结果的代数表达式，而不是存储结果本身。

这里有一个简单的例子。在某些应用中，例如计算几何，除了加、减和乘数字外，你还需要进行不四舍五入的除法，产生一个有理数，它可以由两个整数的比例精确表示：

```cpp
struct r {
    int x, y;
};

r operator+(r a, r b) { return {a.x * b.y + a.y * b.x, a.y * b.y}; }
r operator*(r a, r b) { return {a.x * b.x, a.y * b.y}; }
r operator/(r a, r b) { return {a.x * b.x, a.y * b.y}; }
bool operator<(r a, r b) { return a.x * b.y < b.x * a.y; }
// ...and so on, you get the idea 
```

这个比例可以被约简，这将使这种表示变得独特：

```cpp
struct r {
    int x, y;
    r(int x, int y) : x(x), y(y) {
        if (y < 0)
            x = -x, y = -y;
        int g = gcd(x, y);
        x /= g;
        y /= g;
    }
}; 
```

这就是像 WolframAlpha 和 SageMath 这样的 *计算机代数* 系统是如何工作的：它们仅操作符号表达式，并避免将任何内容作为实值评估。

使用这种方法，你可以获得绝对精度，当你的范围有限，例如只支持有理数时，它效果很好。但是，这需要巨大的计算成本，因为通常你需要以某种方式存储导致结果的所有操作的历史，并在每次执行新操作时考虑这些历史——随着历史的增长，这很快就会变得不可行。

### [#](https://en.algorithmica.org/hpc/arithmetic/float/#fixed-point-numbers) 固定点数

另一种方法是坚持使用整数，但将它们视为乘以一个固定的常数。这本质上等同于改变更大规模测量单位的单位。

由于某些值无法精确表示，这使得计算不精确：你需要将结果四舍五入到最近的表示值。

这种方法在金融软件中常用，在这些软件中，你需要一个简单的方法来管理舍入误差，以确保最终数字相加。例如，纳斯达克在其股票列表中使用美元的 $\frac{1}{10000}$ 作为其基本单位，这意味着在所有交易中，你都能获得小数点后精确到 4 位的精度。

```cpp
struct money {
    uint v; // 1/10000th of a dollar
};

std::string to_string(money) {
    return std::format("${0}.{1:04d}", v / 10000, v % 10000);
}

money operator*(money x, money y) { return {x.v * y.v / 10000}; } 
```

除了引入舍入误差外，更大的问题是，当缩放常数放置不当时，它们变得无用。如果你正在处理的数字太大，则内部整数值将溢出，如果数字太小，它们将被四舍五入到零。有趣的是，这种情况曾经 [成为问题](https://www.wsj.com/articles/berkshire-hathaways-stock-price-is-too-much-for-computers-11620168548) 在纳斯达克，当伯克希尔哈撒韦的股价接近 $\frac{2^{32} - 1}{10000}$ = $429,496.7295$ 时，它再也无法适应一个无符号 32 位整数。

这个问题使得定点算术在需要同时使用小数和大数的应用中基本不适用，例如，评估某些物理方程式：

$$ E = m c² $$

在这个特定的例子中，$m$ 通常与质子的质量（$1.67 \cdot 10^{-27}$ kg）处于同一数量级，而 $c$ 是光速（$3 \cdot 10⁹$ m/s）。

### [#](https://en.algorithmica.org/hpc/arithmetic/float/#floating-point-numbers)浮点数

在大多数数值应用中，我们主要关注相对误差。我们希望我们的计算结果与真实值相差不超过，比如说，$0.01\%$，我们并不真正关心这 $0.01\%$ 在绝对单位中等于多少。

浮点数通过存储一定数量的最高有效数字和数字的量级来解决这一问题。更精确地说，它们使用一个整数（称为*尾数*或*小数部分*）并通过某个固定基数的指数进行缩放——最常见的是 2 或 10。例如：

$$ 1.2345 = \underbrace{12345}_\text{小数部分} \times {\underbrace{10}_\text{基数}\!\!\!\!} ^{\overbrace{-4}^\text{指数}} $$

计算机在固定长度的二进制词上操作，因此当为硬件设计浮点数格式时，你希望有一个固定长度的二进制格式，其中一些位专门用于尾数（以获得更高的精度），一些位用于指数（以获得更大的范围）。

这个手工实现的浮点数示例希望传达这个想法：

```cpp
// DIY floating-point number
struct fp {
    int m; // mantissa
    int e; // exponent
}; 
```

这样我们就可以用 $\pm \; m \times 2^e$ 的形式来表示数字，其中 $m$ 和 $e$ 都是有限的可能为负的整数——这分别对应于负数或小数。这些数字的分布非常不均匀：在 $[0, 1)$ 范围内的数字数量与在 $[1, +\infty)$ 范围内的数字数量大致相同。

注意，这些表示对于某些数字不是唯一的。例如，数字 $1$ 可以表示为

$$ 1 \times 2⁰ = 2 \times 2^{-1} = 256 \times 2^{-8} $$

以及在 28 种其他不会溢出尾数的方式中。

这在某些应用中可能会出现问题，例如比较或散列。为了解决这个问题，我们可以使用某种约定来*规范化*这些表示。在十进制中，[标准形式](https://en.wikipedia.org/wiki/Scientific_notation)是将逗号放在第一个数字之后（`6.022e23`），对于二进制，我们可以做同样的事情：

$$ 42 = 10101_2 = 1.0101_2 \times 2⁵ $$ 注意，按照这个规则，第一个位总是 1。显式存储它是多余的，所以我们假装它在那里，只存储其他位，这些位对应于 $0, 1)$ 范围内的某个有理数。现在可以表示的数字集合大致为 $$ \{ \pm \; (1 + m) \cdot 2^e \; | \; m = \frac{x}{2^{32}}, \; x \in [0, 2^{32}) \} $$

由于 $m$ 现在是一个非负值，我们将将其作为无符号整数处理，并添加一个单独的布尔字段来表示数字的符号：

```cpp
struct fp {
    bool s;     // sign: "0" for "+", "1" for "-" 
    unsigned m; // mantissa
    int e;      // exponent
}; 
```

现在，让我们尝试实现一些算术运算——例如，乘法——使用我们手工制作的浮点数。使用新的公式，结果应该是：

$$ \begin{aligned} c &= a \cdot b \\ &= (s_a \cdot (1 + m_a) \cdot 2^{e_a}) \cdot (s_b \cdot (1 + m_b) \cdot 2^{e_b}) \\ &= s_a \cdot s_b \cdot (1 + m_a) \cdot (1 + m_b) \cdot 2^{e_a} \cdot 2^{e_b} \\ &= \underbrace{s_a \cdot s_b}_{s_c} \cdot (1 + \underbrace{m_a + m_b + m_a \cdot m_b}_{m_c}) \cdot 2^{\overbrace{e_a + e_b}^{e_c}} \end{aligned} $$

现在的分组看起来计算起来很简单，但有两大要点：

+   新的尾数现在在 $[0, 3)$ 范围内。我们需要检查它是否大于 $1$ 并规范化表示，应用以下公式：$1 + m = (1 + 1) + (m - 1) = (1 + \frac{m - 1}{2}) \cdot 2$。

+   结果数字可能（并且很可能）不能精确表示，因为缺乏精度。我们需要两倍的位来考虑 $m_a \cdot m_b$ 项，我们在这里能做的最好的事情是将它四舍五入到最接近的可表示数字。

由于我们需要一些额外的位来正确处理尾数溢出问题，我们将从 $m$ 中保留一个位，从而将其限制在 $[0,2^{31})$ 范围内。

```cpp
fp operator*(fp a, fp b) {
    fp c;
    c.s = a.s ^ b.s;
    c.e = a.e + b.e;

    uint64_t x = a.m, y = b.m; // casting to wider types
    uint64_t m = (x << 31) + (y << 31) + x * y; // 62- or 63-bit intermediate result
    if (m & (1<<62)) { // checking for overflow
        m -= (1<<62); // m -= 1;
        m >>= 1;
        c.e++;
    }
    m += (1<<30); // "rounding" by adding 0.5 to a value that will be floored next
    c.m = m >> 31;

    return c;
} 
```

许多需要更高精度级别应用软件使用类似的软件浮点算术。但当然，你不想每次想要乘以两个实数时都执行 10 条或更多的指令，所以现代 CPU 上，浮点算术通常在硬件中实现——由于其复杂性，通常作为独立的协处理器。

x86 的 *浮点单元*（通常称为 x87）有独立的寄存器和自己的小型指令集，支持内存操作、基本算术、三角函数以及一些常见操作，如对数、指数和平方根。为了使这些操作正确地协同工作，需要澄清浮点数表示的一些额外细节——我们将在 [下一节 中进行说明。[← ../Arithmetic](https://en.algorithmica.org/hpc/arithmetic/)[IEEE 754 浮点数 →](https://en.algorithmica.org/hpc/arithmetic/ieee-754/)
