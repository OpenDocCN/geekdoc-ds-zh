# 合同编程

> 原文：[`en.algorithmica.org/hpc/compilation/contracts/`](https://en.algorithmica.org/hpc/compilation/contracts/)

在“安全”的语言如 Java 和 Rust 中，你通常对每个可能的操作和每个可能的输入都有明确定义的行为。有一些事情是*未定义的*，比如哈希表中键的顺序或`std::vector`的增长因子，但这些通常是些微不足道的小细节，留给实现以备未来可能获得性能提升。

相比之下，C 和 C++将未定义行为的概念提升到了另一个层次。某些操作在编译或运行时不会引发错误，但它们是不被*允许的*——在程序员和编译器之间存在一种*合同*，即如果出现未定义行为，编译器在法律上被允许做任何事情，包括炸毁你的显示器或格式化你的硬盘。但编译器工程师对此不感兴趣。相反，未定义行为被用来保证没有边缘情况，并帮助优化。

### [#](https://en.algorithmica.org/hpc/compilation/contracts/#why-undefined-behavior-exists)未定义行为存在的原因

有两组主要的行为会导致未定义行为：

+   几乎肯定是不小心引入的 bug 的操作，如除以零、取消对空指针的引用或从未初始化的内存中读取。你希望在测试过程中尽早捕获这些，因此崩溃或有一些非确定性行为比它们总是执行固定的回退操作（如返回零）要好。

    你可以使用*sanitizers*编译和运行程序以在早期捕获未定义行为。在 GCC 和 Clang 中，你可以使用`-fsanitize=undefined`标志，并且一些臭名昭著的会导致未定义行为的操作将被仪器化以在运行时检测它。

+   在不同平台上，某些操作会有略微不同的可观察行为。例如，将整数左移超过 31 位的结果是未定义的，因为执行此操作的指令在 Arm 和 x86 CPU 上的实现方式不同。如果你标准化一种特定的行为，那么为其他平台编译的所有程序将不得不多花费几个周期来检查这种边缘情况，因此最好将其留为未定义。

    有时，当某些平台特定的行为有合法的使用场景时，而不是将其声明为未定义，它可以被留为*实现定义的*。例如，右移负整数的结果取决于平台：它要么向右移动零，要么向右移动一（例如，将`11010110 = -42`右移一位可能意味着`01101011 = 107`或`11101011 = -21`，这两种情况都是现实的）。

将某些内容指定为未定义行为而不是实现定义的行为也有助于编译器进行优化。考虑有符号整数溢出的情况。在几乎所有架构上，有符号整数的溢出方式与无符号整数相同，`INT_MAX + 1 == INT_MIN`，然而，根据 C++ 标准，这是未定义行为。这是故意的：如果你不允许有符号整数溢出，那么对于 `int`，`(x + 1) > x` 总是保证为真，但对于 `unsigned int` 则不是，因为 `(x + 1)` 可能会溢出。对于有符号类型，这允许编译器优化掉这样的检查。

作为更自然发生的例子，考虑一个具有整数控制变量的循环的情况。现代 C++ 和像 Rust 这样的语言鼓励程序员使用无符号整数 (`size_t` / `usize`)，而 C 程序员则固执地继续使用 `int`。为了理解为什么，考虑以下 `for` 循环：

```cpp
for (unsigned int i = 0; i < n; i++) {
    // ...
} 
```

这个循环执行了多少次？从技术上讲有两个有效的答案：$n$ 和无穷大，后者是 $n$ 超过 $2^{32}$ 的情况，此时 $i$ 每 $2^{32}$ 次迭代就会重置为零。虽然前者可能是程序员假设的，但为了符合语言规范，编译器仍然必须插入额外的运行时检查并考虑两种情况，这些情况应该被优化不同。同时，`int` 版本将恰好执行 $n$ 次迭代，因为有符号溢出的可能性已经被定义为不存在。

### [#](https://en.algorithmica.org/hpc/compilation/contracts/#removing-corner-cases) 移除角落案例

“安全”的编程风格通常涉及进行大量的运行时检查，但它们不必以性能为代价。

例如，Rust 在索引数组和其他随机访问结构时著名地使用了边界检查。在 C++ STL 中，`vector` 和 `array` 有一个“不安全”的 `[]` 操作符和一个“安全”的 `.at()` 方法，大致如下：

```cpp
T at(size_t k) {
    if (k >= size())
        throw std::out_of_range("Array index exceeds its size");
    return _memory[k];
} 
```

有趣的是，这些检查在运行时很少真正执行，因为编译器通常可以在编译时证明每个访问都在边界内。例如，当从 1 迭代到数组大小并在每一步索引第 $i$ 个元素时，不可能发生任何非法操作，因此边界检查可以安全地优化掉。

### [#](https://en.algorithmica.org/hpc/compilation/contracts/#assumptions) 假设

当编译器无法证明角落案例不存在，但你可以时，可以使用未定义行为的机制提供这些额外信息。

Clang 有一个有用的 `__builtin_assume` 函数，你可以在其中放置一个保证为真的语句，编译器将使用这个假设进行优化。在 GCC 中，你可以使用 `__builtin_unreachable` 来做同样的事情：

```cpp
void assume(bool pred) {
    if (!pred)
        __builtin_unreachable();
} 
```

例如，你可以在上面的例子中的 `at` 之前放置 `assume(k < vector.size())`，然后边界检查将被优化掉。

将 `assume` 与 `assert` 和 `static_assert` 结合起来以查找错误也是非常有用的：你可以在调试构建中使用相同的函数来检查先决条件，然后在生产构建中利用它们来提高性能。

### [#](https://en.algorithmica.org/hpc/compilation/contracts/#arithmetic)算术

边界情况是你应该记住的事情，尤其是在优化算术时。

对于 浮点运算，这并不是一个很大的问题，因为你可以通过 `-ffast-math` 标志（它也包含在 `-Ofast` 中）来禁用严格的标准化遵守。你几乎不得不这样做，因为否则编译器除了以与源代码相同的顺序执行算术运算外，别无他法进行优化。

对于整数运算，这不同，因为结果总是必须精确的。考虑除以 2 的情况：

```cpp
unsigned div_unsigned(unsigned x) {
    return x / 2;
} 
```

一个广为人知的优化是将它替换为单个右移（`x >> 1`）：

```cpp
shr eax 
```

这当然对所有 *正数* 是正确的，但对于一般情况又如何呢？

```cpp
int div_signed(int x) {
    return x / 2;
} 
```

如果 `x` 是负数，那么简单的移位操作就不起作用了——无论移位是在零位还是符号位上执行：

+   如果我们移入零，我们会得到一个非负结果（符号位为零）。

+   如果我们移入符号位，那么舍入将朝向负无穷大而不是零（`-5 / 2` 将等于 `-3` 而不是 `-2`）^(1)。

因此，对于一般情况，我们必须插入一些辅助手段来使其工作：

```cpp
mov  ebx, eax
shr  ebx, 31    ; extract the sign bit
add  eax, ebx   ; add 1 to the value if it is negative to ensure rounding towards zero
sar  eax        ; this one shifts in sign bits 
```

当只有正数情况是我们所期望的，我们也可以使用 `assume` 机制来消除负 `x` 的可能性，并避免处理这个边界情况：

```cpp
int div_assume(int x) {
    assume(x >= 0);
    return x / 2;
} 
```

尽管在这个特定情况下，可能最好的语法来表示我们只期望非负数是使用无符号整数类型。

由于这样的细微差别，通常在中间函数中展开代数并手动简化算术，而不是依赖编译器来做，这样做往往是有益的。

### [#](https://en.algorithmica.org/hpc/compilation/contracts/#memory-aliasing)内存别名

编译器在优化涉及内存读取和写入的操作方面相当糟糕。这是因为它们通常没有足够的上下文来确保优化是正确的。

考虑以下示例：

```cpp
void add(int *a, int *b, int n) {
    for (int i = 0; i < n; i++)
        a[i] += b[i];
} 
```

由于这个循环的每次迭代都是独立的，它可以并行执行并 向量化。但从技术上讲，它真的可以吗？

如果数组 `a` 和 `b` 交叉，可能会出现问题。考虑 `b == a + 1` 的情况，即如果 `b` 只是 `a` 从第二个元素开始的内存视图。在这种情况下，下一次迭代依赖于前一次迭代，唯一的正确解决方案是顺序执行循环。即使程序员知道这种情况不可能发生，编译器也必须检查这种可能性。

这就是为什么我们有 `const` 和 `restrict` 关键字。第一个强制我们不会用指针变量修改内存，第二个是告诉编译器内存保证不会被别名化的方式。

```cpp
void add(int * __restrict__ a, const int * __restrict__ b, int n) {
    for (int i = 0; i < n; i++)
        a[i] += b[i];
} 
```

这些关键字单独使用也是为了自文档化的好主意。

### [#](https://en.algorithmica.org/hpc/compilation/contracts/#c-contracts)C++ 合同

合同编程是一种使用较少但非常强大的技术。

有一个后期提案建议将设计-by-contract 以合同属性的形式添加到 C++ 标准[http://www.hellenico.gr/cpp/w/cpp/language/attributes/contract.html]，这些属性在功能上等同于我们手工制作的、针对特定编译器的 `assume`：

```cpp
T at(size_t k) [[ expects: k < n ]] {
    return _memory[k];
} 
```

有 3 种类型的属性——`expects`、`ensures` 和 `assert`——分别用于在函数中指定前置和后置条件以及可以在程序中的任何地方放置的一般断言。

不幸的是，这个令人兴奋的新特性[尚未最终标准化](https://www.reddit.com/r/cpp/comments/cmk7ek/what_happened_to_c20_contracts/)，更不用说在主要的 C++ 编译器中实现了。但也许，几年后，我们能够写出这样的代码：

```cpp
bool is_power_of_two(int m) {
    return m > 0 && (m & (m - 1) == 0);
}

int mod_power_of_two(int x, int m)
    [[ expects: x >= 0 ]]
    [[ expects: is_power_of_two(m) ]]
    [[ ensures r: r >= 0 && r < m ]]
{
    int r = x & (m - 1);
    [[ assert: r = x % m ]];
    return r;
} 
```

其他面向性能的语言，如 [Rust](https://docs.rs/contracts/latest/contracts/) 和 [D](https://dlang.org/spec/contracts.html)，也提供了某些形式的合同编程。

一个通用的、与语言无关的建议是始终检查编译器生成的汇编代码（../stages），如果它不是你所期望的，那么尝试思考可能限制编译器进行优化的边界情况。

* * *

1.  有趣的事实：在 Python 中，由于某种原因，对负数进行整数除法会将结果向下取整，所以 `-5 // 2 = -3` 等同于 `-5 >> 1 = -3`。我怀疑吉多·范罗苏姆在最初设计语言时并没有考虑到这种优化，但从理论上讲，一个包含许多除以二的操作的 JIT 编译 Python 程序可能比类似的 C++ 程序更快。↩︎ [← 情境优化](https://en.algorithmica.org/hpc/compilation/situational/)[预计算 →](https://en.algorithmica.org/hpc/compilation/precalc/)
