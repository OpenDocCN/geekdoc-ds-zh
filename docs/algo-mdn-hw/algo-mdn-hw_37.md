# 整数除法

> 原文：[`en.algorithmica.org/hpc/arithmetic/division/`](https://en.algorithmica.org/hpc/arithmetic/division/)

与其他算术运算相比，除法在 x86 和通用计算机上工作得非常糟糕。浮点数和整数除法在硬件中实现起来非常困难。电路在 ALU 中占用大量空间，计算有多个阶段，因此`div`及其相关指令通常需要 10-20 个周期才能完成，对于较小的数据类型，延迟略低。

### [#](https://en.algorithmica.org/hpc/arithmetic/division/#division-and-modulo-in-x86) x86 中的除法和取模

由于没有人愿意为单独的取模操作重复所有这些混乱，`div`指令同时服务于这两个目的。要执行 32 位整数除法，你需要将被除数*特别*放入`eax`寄存器，并用除数作为其唯一操作数调用`div`。之后，商将被存储在`eax`中，余数将被存储在`edx`中。

唯一的注意事项是，被除数实际上需要存储在*两个*寄存器中，`eax`和`edx`：这种机制使得可以进行 64 位除以 32 位，甚至 128 位除以 64 位的除法，类似于 128 位乘法的工作方式。在执行常规的 32 位除以 32 位有符号除法时，我们需要将`eax`扩展到 64 位，并将其高位存储在`edx`中：

```cpp
div(int, int):
    mov  eax, edi
    cdq
    idiv esi
    ret 
```

对于无符号除法，你可以将`edx`设置为零，这样它就不会干扰：

```cpp
div(unsigned, unsigned):
    mov  eax, edi
    xor  edx, edx
    div  esi
    ret 
```

在这两种情况下，除了`eax`中的商之外，你还可以通过`edx`访问余数：

```cpp
mod(unsigned, unsigned):
    mov  eax, edi
    xor  edx, edx 
    div  esi
    mov  eax, edx
    ret 
```

你也可以将 128 位整数（存储在`rdx:rax`中）除以一个 64 位整数：

```cpp
div(u128, u64):
    ; a = rdi + rsi, b = rdx
    mov  rcx, rdx
    mov  rax, rdi
    mov  rdx, rsi
    div  edx 
    ret 
```

被除数的高位应该小于除数，否则会发生溢出。由于这个限制，要使编译器自己生成此代码是[困难的](https://danlark.org/2020/06/14/128-bit-division/)：如果你将一个 128 位整数类型除以一个 64 位整数，编译器会将其包裹在额外的检查中，这些检查实际上可能是多余的。

### [#](https://en.algorithmica.org/hpc/arithmetic/division/#division-by-constants) 常数除法

整数除法非常慢，即使在硬件中完全实现，但在某些情况下，如果除数是常数，可以避免。一个众所周知的例子是除以 2 的幂，这可以替换为一个周期的二进制移位：二进制 GCD 算法是这种技术的令人愉快的展示。

在一般情况下，有一些巧妙的小技巧可以将除法替换为乘法，但需要一些预计算。所有这些技巧都基于以下想法。考虑将一个浮点数$x$除以另一个已知的浮点数$y$的任务。我们可以计算一个常数

$$ d \approx y^{-1} $$ 然后，在运行时，我们将计算 $$ x / y = x \cdot y^{-1} \approx x \cdot d $$

$\frac{1}{y}$ 的结果最多偏离 $\epsilon$，乘法 $x \cdot d$ 只会增加另一个 $\epsilon$，因此最多偏离 $2 \epsilon + \epsilon² = O(\epsilon)$，这对于浮点数情况是可以容忍的。

### [#](https://en.algorithmica.org/hpc/arithmetic/division/#barrett-reduction)巴雷特除法

如何将这个技巧推广到整数？计算 `int d = 1 / y` 似乎不起作用，因为它将只是零。我们能做的最好的事情是将它表示为

$$ d = \frac{m}{2^s} $$ 然后找到一个“神奇”的数字 $m$ 和一个二进制移位 $s$，使得对于所有 `x` 在范围内，`x / y == (x * m) >> s`。$$ \lfloor x / y \rfloor = \lfloor x \cdot y^{-1} \rfloor = \lfloor x \cdot d \rfloor = \lfloor x \cdot \frac{m}{2^s} \rfloor $$

可以证明这样的配对总是存在的，并且编译器实际上会通过自己执行类似的优化。每次它们遇到对常数的除法时，它们都会用乘法和二进制移位来替换它。以下是除以 `(10⁹ + 7)` 的 `unsigned long long` 生成的汇编代码：

```cpp
;  input (rdi): x
; output (rax): x mod (m=1e9+7)
mov    rax, rdi
movabs rdx, -8543223828751151131  ; load magic constant into a register
mul    rdx                        ; perform multiplication
mov    rax, rdx
shr    rax, 29                    ; binary shift of the result 
```

这种技术被称为**巴雷特除法**，它被称为“除法”是因为它主要用于模运算，可以通过这个公式用一次除法、一次乘法和一次减法来替换：

$$ r = x - \lfloor x / y \rfloor \cdot y $$

此方法需要一些预计算，包括执行一次实际除法。因此，这只有在您执行的不是一次而是几次除法，并且所有这些除法都有相同的常数除数时才有益。

### [#](https://en.algorithmica.org/hpc/arithmetic/division/#why-it-works)为什么它有效

为什么总是存在这样的 $m$ 和 $s$，更不用说如何找到它们了。但给定一个固定的 $s$，直觉告诉我们，为了使 $2^s$ 能够抵消，$m$ 应该尽可能接近 $2^s/y$。因此有两个自然的选择：$\lfloor 2^s/y \rfloor$ 和 $\lceil 2^s/y \rceil$。第一个不适用，因为如果你用

$$ \Bigl \lfloor \frac{x \cdot \lfloor 2^s/y \rfloor}{2^s} \Bigr \rfloor $$ 然后对于任何整数 $\frac{x}{y}$，其中 $y$ 不是偶数，结果将严格小于真实值。这仅留下另一种情况，$m = \lceil 2^s/y \rceil$。现在，让我们尝试推导出计算结果的上下限：$$ \lfloor x / y \rfloor = \Bigl \lfloor \frac{x \cdot m}{2^s} \Bigr \rfloor = \Bigl \lfloor \frac{x \cdot \lceil 2^s /y \rceil}{2^s} \Bigr \rfloor $$ 让我们从 $m$ 的界限开始：$$ 2^s / y \le \lceil 2^s / y \rceil < 2^s / y + 1 $$ 现在对于整个表达式：$$ x / y - 1 < \Bigl \lfloor \frac{x \cdot \lceil 2^s /y \rceil}{2^s} \Bigr \rfloor < x / y + x / 2^s $$

我们可以看到结果落在大小为 $(1 + \frac{x}{2^s})$ 的范围内，并且如果这个范围对于所有可能的 $x / y$ 总是恰好有一个整数，那么算法将保证给出正确答案。结果证明，我们可以总是将 $s$ 设置得足够高以实现这一点。

这里最坏的情况是什么？如何选择 $x$ 和 $y$ 使得 $(x/y - 1, x/y + x / 2^s)$ 范围内包含两个整数？我们可以看到整数比不适用，因为左边界不包括在内，并且假设 $x/2^s < 1$，只有 $x/y$ 本身会在范围内。最坏的情况实际上是 $x/y$ 接近 $1$ 但不超过它的值。对于 $n$ 位整数，这是可能的最大整数除以第一个最大整数：

$$ \begin{aligned} x = 2^n - 2 \\ y = 2^n - 1 \end{aligned} $$

在这种情况下，下限将是 $(\frac{2^n-2}{2^n-1} - 1)$，上限将是 $(\frac{2^n-2}{2^n-1} + \frac{2^n-2}{2^s})$。左边界尽可能接近一个整数，整个范围的大小是可能的最大整数的第二大小。而且这里是关键点：如果 $s \ge n$，那么这个范围内只包含一个整数，即 $1$，因此算法将始终返回它。

### [#](https://en.algorithmica.org/hpc/arithmetic/division/#lemire-reduction)Lemire 降阶法

Barrett 降阶法稍微复杂一些，并且由于它是间接计算的，因此为模运算生成了一个长度指令序列。有一种新的 ([2019](https://arxiv.org/pdf/1902.01961.pdf)) 方法，在某些情况下对于模运算来说更简单且实际上更快。它还没有一个传统的名字，但我打算称它为 [Lemire](https://lemire.me/blog/) 降阶法。

这里是主要思想。考虑一些整数分数的浮点表示：

$$ \frac{179}{6} = 11101.1101010101\ldots = 29\tfrac{5}{6} \approx 29.83 $$

我们如何“解剖”它以获取所需的各个部分？

+   要得到整数部分（29），我们只需在点之前截断或舍入它。

+   要得到小数部分（⅚），我们只需取点后面的部分。

+   要得到余数（5），我们可以将小数部分乘以除数。

现在，对于 32 位整数，我们可以将 $s = 64$ 并查看我们在乘法和移位方案中进行的计算：

$$ \lfloor x / y \rfloor = \Bigl \lfloor \frac{x \cdot m}{2^s} \Bigr \rfloor = \Bigl \lfloor \frac{x \cdot \lceil 2^s /y \rceil}{2^s} \Bigr \rfloor $$

我们在这里真正做的是将 $x$ 乘以一个浮点常数（$x \cdot m$），然后截断结果（$\lfloor \frac{\cdot}{2^s} \rfloor$）。

如果我们取的不是最高位而是最低位呢？这会对应于小数部分——如果我们将其乘以 $y$ 并截断结果，这将正好是余数：

$$ r = \Bigl \lfloor \frac{ (x \cdot \lceil 2^s /y \rceil \bmod 2^s) \cdot y }{2^s} \Bigr \rfloor $$

这方法完美无缺，因为我们在这里所做的工作可以解释为仅仅是三次链式浮点数乘法，其总相对误差为 $O(\epsilon)$。由于 $\epsilon = O(\frac{1}{2^s})$ 且 $s = 2n$，误差将始终小于一，因此结果将是精确的。

```cpp
uint32_t y;

uint64_t m = uint64_t(-1) / y + 1; // ceil(2^64 / y)

uint32_t mod(uint32_t x) {
    uint64_t lowbits = m * x;
    return ((__uint128_t) lowbits * y) >> 64; 
}

uint32_t div(uint32_t x) {
    return ((__uint128_t) m * x) >> 64;
} 
```

我们还可以通过使用以下事实来检查 $x$ 是否能被 $y$ 整除：如果除法的余数为零，当且仅当分数部分（$m \cdot x$ 的低 64 位）不超过 $m$（否则，当乘以 $y$ 并右移 64 位时，它将变成一个非零数）。

```cpp
bool is_divisible(uint32_t x) {
    return m * x < m;
} 
```

这种方法的唯一缺点是它需要比原始大小大四倍的整型来执行乘法，而其他简化方法只需双精度即可。

还有一种通过仔细操作中间结果的一半来计算 64x64 模的方法；实现细节留给读者作为练习。

### [#](https://en.algorithmica.org/hpc/arithmetic/division/#further-reading)进一步阅读

查看关于更通用优化整数除法实现的[libdivide](https://github.com/ridiculousfish/libdivide)和[GMP](https://gmplib.org/)。

值得阅读的是[Hacker’s Delight](https://www.amazon.com/Hackers-Delight-2nd-Henry-Warren/dp/0321842685)，其中有一整章专门介绍整数除法。[← 整数](https://en.algorithmica.org/hpc/arithmetic/integer/)[→ 数论](https://en.algorithmica.org/hpc/number-theory/)
