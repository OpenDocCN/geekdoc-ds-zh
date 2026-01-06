# 无分支编程

> 原文：[`en.algorithmica.org/hpc/pipelining/branchless/`](https://en.algorithmica.org/hpc/pipelining/branchless/)

如我们在上一节中所述，CPU 无法有效预测的分支是昂贵的，因为它们可能导致分支预测错误后长时间流水线停滞以获取新指令。在本节中，我们将讨论消除分支的方法。

### [#](https://en.algorithmica.org/hpc/pipelining/branchless/#predication)预测

我们将继续之前开始的同一种案例研究——我们创建一个随机数字数组，并计算所有小于 50 的元素的总和：

```cpp
for (int i = 0; i < N; i++)
    a[i] = rand() % 100;

volatile int s;

for (int i = 0; i < N; i++)
    if (a[i] < 50)
        s += a[i]; 
```

我们的目标是消除由`if`语句引起的分支。我们可以尝试这样消除它：

```cpp
for (int i = 0; i < N; i++)
    s += (a[i] < 50) * a[i]; 
```

循环现在每个元素需要大约 7 个周期，而不是原来的大约 14 个周期。此外，如果我们改变`50`为其他阈值，性能仍然保持不变，因此它不依赖于分支概率。

但是等等…难道不应该仍然有一个分支吗？`(a[i] < 50)`是如何映射到汇编中的？

在汇编中不存在布尔类型，也没有基于比较结果返回 1 或 0 的指令，但我们可以间接地这样计算：`(a[i] - 50) >> 31`。这个技巧依赖于整数的二进制表示，特别是以下事实：如果表达式`a[i] - 50`是负数（意味着`a[i] < 50`），那么结果的最高位将被设置为 1，然后我们可以通过右移来提取它。

```cpp
mov  ebx, eax   ; t = x
sub  ebx, 50    ; t -= 50
sar  ebx, 31    ; t >>= 31
imul  eax, ebx   ; x *= t 
```

实现整个序列的另一种更复杂的方法是将符号位转换为掩码，然后使用位运算`and`而不是乘法：`((a[i] - 50) >> 31 - 1) & a[i]`。考虑到与其他指令不同，`imul`需要 3 个周期，这使得整个序列快一个周期。

```cpp
mov  ebx, eax   ; t = x
sub  ebx, 50    ; t -= 50
sar  ebx, 31    ; t >>= 31
; imul  eax, ebx ; x *= t
sub  ebx, 1     ; t -= 1 (causing underflow if t = 0)
and  eax, ebx   ; x &= t 
```

注意，从编译器的角度来看，这种优化在技术上是不正确的：对于 50 个最低的可表示整数——那些在$[-2^{31}, - 2^{31} + 49]$范围内的整数——结果将因为下溢而出错。我们知道所有数字都在 0 到 100 之间，这种情况不会发生，但编译器不知道。

但编译器实际上选择做不同的事情。它没有使用这个算术技巧，而是使用了一个特殊的`cmov`（“条件移动”）指令，该指令根据条件（使用标志寄存器计算和检查，与跳转相同）分配一个值。

```cpp
mov     ebx, 0      ; cmov doesn't support immediate values, so we need a zero register
cmp     eax, 50
cmovge  eax, ebx    ; eax = (eax >= 50 ? eax : ebx=0) 
```

因此，上面的代码实际上更接近使用三元运算符，如下所示：

```cpp
for (int i = 0; i < N; i++)
    s += (a[i] < 50 ? a[i] : 0); 
```

这两种变体都经过编译器优化，并产生以下汇编代码：

```cpp
 mov     eax, 0
    mov     ecx, -4000000
loop:
    mov     esi, dword ptr [rdx + a + 4000000]  ; load a[i]
    cmp     esi, 50
    cmovge  esi, eax                            ; esi = (esi >= 50 ? esi : eax=0)
    add     dword ptr [rsp + 12], esi           ; s += esi
    add     rdx, 4
    jnz     loop                                ; "iterate while rdx is not zero" 
```

这种通用技术被称为*预测*，它大致等同于以下代数技巧：

$$ x = c \cdot a + (1 - c) \cdot b $$

这样你可以消除分支，但这也意味着要评估 *两个* 分支以及 `cmov` 本身。因为评估“>=”分支不花费任何代价，所以性能与分支版本中的“总是是”情况完全相同。

### [#](https://en.algorithmica.org/hpc/pipelining/branchless/#when-predication-is-beneficial)当预测有益时

使用预测消除了控制冒险，但引入了数据冒险。仍然会有流水线停顿，但这是一个更便宜的停顿：你只需要等待 `cmov` 解决，而不是在预测错误的情况下清空整个流水线。

然而，有许多情况下，保持分支代码不变会更有效率。这种情况发生在计算 *两个* 分支的成本超过了只计算 *一个* 分支的惩罚。

在我们的例子中，当分支可以被预测的概率超过 ~75% 时，分支代码获胜。

![](img/2310bb67e7bb9ec6ac10dff44f8498e4.png)

这个 75% 的阈值通常被编译器用作确定是否使用 `cmov` 的启发式方法。不幸的是，这个概率通常在编译时是未知的，因此需要以几种方式之一提供：

+   我们可以使用基于配置文件优化，它将自行决定是否使用预测。

+   我们可以使用可能性属性和编译器特定的内建函数来提示分支的可能性：GCC 中的 `__builtin_expect_with_probability` 和 Clang 中的 `__builtin_unpredictable`。

+   我们可以使用三元运算符或各种算术技巧重写分支代码，这相当于程序员和编译器之间的一种隐式合同：如果程序员这样编写代码，那么它可能意味着要无分支。

“正确的方法”是使用分支提示，但不幸的是，对它们的支持不足。目前，这些提示似乎在编译器后端决定是否使用 `cmov` 时已经丢失了。[这些提示似乎丢失了](https://bugs.llvm.org/show_bug.cgi?id=40027)。有一些进展[朝着使其成为可能的方向](https://discourse.llvm.org/t/rfc-cmov-vs-branch-optimization/6040)，但目前还没有好的方法可以强制编译器生成无分支代码，所以有时最好的希望就是简单地写一小段汇编代码。

### [#](https://en.algorithmica.org/hpc/pipelining/branchless/#larger-examples)更大的例子

**字符串**。简化来说，`std::string` 由一个指向堆上某个位置的以 null 结尾的 `char` 数组（也称为“C-string”）的指针和一个包含字符串大小的整数组成。

字符串的一个常见值是空字符串——这也是它的默认值。你还需要以某种方式处理它们，并且习惯性的方法是分配`nullptr`作为指针和`0`作为字符串大小，然后在涉及字符串的每个过程的开始处检查指针是否为空或大小是否为零。

然而，这需要单独的分支，这是代价高昂的（除非大多数字符串要么是空的，要么是非空的）。为了移除检查以及分支，我们可以分配一个“零 C 字符串”，这只是一个在某个地方分配的零字节，然后简单地将所有空字符串指向那里。现在所有涉及空字符串的字符串操作都必须读取这个无用的零字节，但这仍然比分支预测错误便宜得多。

**二分搜索。**标准的二分搜索可以无分支实现，在小型数组（适合缓存）上它比分支的`std::lower_bound`快约 4 倍：

```cpp
int lower_bound(int x) {
    int *base = t, len = n;
    while (len > 1) {
        int half = len / 2;
        base += (base[half - 1] < x) * half; // will be replaced with a "cmov"
        len -= half;
    }
    return *base;
} 
```

除了更复杂之外，它还有一个轻微的缺点，即它可能进行更多的比较（常数 $\lceil \log_2 n \rceil$ 而不是 $\lfloor \log_2 n \rfloor$ 或 $\lceil \log_2 n \rceil$），并且不能预测未来的内存读取（这充当预取，因此在大数组上会损失性能）。

通常，通过隐式或显式地*填充*数据结构，使它们的操作需要恒定的迭代次数，从而实现无分支。有关更复杂的示例，请参阅该文章。

**数据并行编程。**无分支编程对于 SIMD 应用非常重要，因为这些应用一开始就没有分支。

在我们的数组求和示例中，从累加器中移除`volatile`类型限定符允许编译器向量化循环：

```cpp
/* volatile */ int s = 0;

for (int i = 0; i < N; i++)
    if (a[i] < 50)
        s += a[i]; 
```

现在它每元素大约需要 0.3 秒，这主要是受内存瓶颈的影响。

编译器通常能够将没有分支或迭代间依赖的任何循环进行向量化——以及一些特定的微小偏差，例如归约或只包含一个 if-without-else 的简单循环。更复杂的任何内容的向量化都是一个非常非平凡的问题，可能涉及各种技术，如掩码和寄存器内排列。[← 分支的成本](https://en.algorithmica.org/hpc/pipelining/branching/)[指令表 →](https://en.algorithmica.org/hpc/pipelining/tables/)
