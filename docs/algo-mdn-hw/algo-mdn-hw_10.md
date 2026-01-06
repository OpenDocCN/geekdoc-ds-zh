# 机器代码布局

> 原文：[`en.algorithmica.org/hpc/architecture/layout/`](https://en.algorithmica.org/hpc/architecture/layout/)

计算机工程师喜欢将 CPU 的 流水线 在心理上分为两部分：*前端*，在这里指令从内存中取出并解码，以及*后端*，在这里它们被调度并最终执行。通常，性能瓶颈在于执行阶段，因此，本书的大部分努力都将致力于优化后端。

但有时情况可能相反，当前端没有足够快地将指令提供给后端以使其饱和时。这可能由许多原因造成，所有这些原因最终都与机器代码在内存中的布局有关，并以轶事的方式影响性能，例如删除未使用的代码、交换“if”分支，甚至改变函数声明的顺序，导致性能要么提高要么下降。

### [#](https://en.algorithmica.org/hpc/architecture/layout/#cpu-front-end)CPU 前端

在机器代码被转换为指令，并且 CPU 理解程序员想要什么之前，它首先需要经过两个我们感兴趣的重要阶段：*取指*和*解码*。

在**取指**阶段，CPU 简单地从主内存中加载一个固定大小的字节数组，其中包含一些指令的二进制编码。这个块的大小在 x86 上通常是 32 字节，尽管在不同的机器上可能会有所不同。一个重要的细微差别是，这个块必须对齐：块地址必须是其大小的倍数（在我们的例子中是 32B）。

接下来是**解码**阶段：CPU 查看这个字节数组，丢弃所有在指令指针之前的部分，并将剩余的部分拆分为指令。机器指令使用可变数量的字节进行编码：像 `inc rax` 这样简单且非常常见的指令只需要一个字节，而一些编码了常量和修改行为前缀的晦涩指令可能需要多达 15 个字节。因此，从 32 字节块中可以解码出可变数量的指令，但不超过称为*解码宽度*的特定机器相关限制。在我的 CPU（一个 [Zen 2](https://en.wikichip.org/wiki/amd/microarchitectures/zen_2)）上，解码宽度是 4，这意味着在每个周期内，最多可以解码并传递给下一阶段的指令数量为 4。

这些阶段以流水线的方式工作：如果 CPU 能够预测（或 预测）它需要的下一个指令块，那么取指阶段不需要等待当前块中的最后一个指令被解码，就可以立即加载下一个指令。

### [#](https://en.algorithmica.org/hpc/architecture/layout/#code-alignment)代码对齐

在其他条件相同的情况下，编译器通常更喜欢较短的机器代码指令，因为这样可以在单个 32B 获取块中放入更多的指令，同时也因为这样可以减少二进制文件的大小。但有时相反的情况更可取，这是因为取出的指令块必须对齐。

假设你需要执行一个从 32B 对齐块的最后一个字节开始的指令序列。你可能能够无额外延迟地执行第一条指令，但对于随后的指令，你必须等待一个额外的周期来执行另一个指令检索。如果代码块是对齐在 32B 边界上，那么最多可以有 4 条指令被解码并随后并发执行（除非它们特别长或相互依赖）。

考虑到这一点，编译器通常会进行看似有害的优化：它们有时会偏好较长的机器代码指令，甚至插入无用的指令^(1)，以便将关键跳转位置对齐到合适的 2 的幂次边界。

在 GCC 中，你可以使用`-falign-labels=n`标志来指定特定的对齐策略，[替换](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html) `-labels` 为 `-function`、`-loops` 或 `-jumps`，如果你想要更具有选择性。在`-O2`和`-O3`优化级别上，它默认启用——没有设置特定的对齐，在这种情况下，它使用一个（通常是合理的）机器相关的默认值。

### [#](https://en.algorithmica.org/hpc/architecture/layout/#instruction-cache)指令缓存

指令的存储和检索主要使用与数据相同的内存系统，除了可能将缓存的下层替换为单独的*指令缓存*（因为你不希望随机的数据读取将处理它的代码踢出）。

指令缓存对于以下情况至关重要：

+   不确定你将要执行哪些指令，并且需要以低延迟获取下一个块，

+   或者正在执行一系列冗长但快速处理的指令，并且需要高带宽。

因此，内存系统可能成为大型机器代码程序的瓶颈。这种考虑限制了之前讨论的优化技术的适用性：

+   内联函数并不总是最优的，因为它减少了代码共享并增加了二进制文件的大小，需要更多的指令缓存。

+   展开循环只有在一定程度上是有益的，即使编译时已知迭代次数：在某个点上，CPU 将不得不从主内存中获取指令和数据，在这种情况下，它很可能会因为内存带宽而成为瓶颈。

+   大量的 代码对齐 会增加二进制文件的大小，再次需要更多的指令缓存。相比从主内存中取指令并等待，多一个周期的取指令时间是一个小的惩罚。

另一个方面是，将频繁使用的指令序列放置在相同的 缓存行 和 内存页 上，可以提高 缓存局部性。为了提高指令缓存利用率，你应该将热代码与热代码组合，将冷代码与冷代码组合，并在可能的情况下移除死代码（未使用的代码）。如果你想进一步探索这个想法，可以查看 Facebook 的 [二进制优化和布局工具](https://engineering.fb.com/2018/06/19/data-infrastructure/accelerate-large-scale-applications-with-bolt/)，该工具最近已被 [合并](https://github.com/llvm/llvm-project/commit/4c106cfdf7cf7eec861ad3983a3dd9a9e8f3a8ae) 到 LLVM 中。

### [#](https://en.algorithmica.org/hpc/architecture/layout/#unequal-branches) 不等分支

假设由于某种原因，你需要一个辅助函数来计算整数区间的长度。它接受两个参数，$x$ 和 $y$，但为了方便，它可能对应于 $[x, y]$ 或 $[y, x]$，这取决于哪个不为空。在纯 C 语言中，你可能写成这样：

```cpp
int length(int x, int y) {  if (x > y) return x - y; else return y - x; } 
```

在 x86 汇编中，你可以以更多的方式实现它，这明显影响了性能。让我们先尝试直接将此代码映射到汇编中：

```cpp
length:  cmp  edi, esi jle  less ; x > y sub  edi, esi mov  eax, edi done:  ret less:  ; x <= y sub  esi, edi mov  eax, esi jmp  done 
```

虽然初始的 C 语言代码看起来非常对称，但汇编版本则不然。这导致了一个有趣的现象：一个分支可以比另一个分支稍微快一点执行：如果 `x > y`，那么 CPU 可以直接执行 `cmp` 和 `ret` 之间的 5 条指令，如果函数对齐，这些指令都会一次性被取到；而如果是 `x <= y` 的情况，则需要额外的两个跳转。

可以合理地假设 `x > y` 的情况是不太可能的（为什么有人会计算反转区间的长度？），更像是几乎从未发生的异常。我们可以检测这种情况，并简单地交换 `x` 和 `y`：

```cpp
int length(int x, int y) {  if (x > y) swap(x, y); return y - x; } 
```

汇编代码将像这样，就像通常的 if-without-else 模式一样：

```cpp
length:  cmp  edi, esi jle  normal     ; if x <= y, no swap is needed, and we can skip the xchg xchg edi, esi normal:  sub  esi, edi mov  eax, esi ret 
```

现在的总指令长度是 6，比之前的 8 短。但仍然没有针对我们假设的情况进行优化：如果我们认为 `x > y` 永远不会发生，那么加载 `xchg edi, esi` 指令（它永远不会被执行）就是浪费。我们可以通过将其移出正常执行路径来解决这个问题：

```cpp
length:  cmp  edi, esi jg   swap normal:  sub  esi, edi mov  eax, esi ret swap:  xchg edi, esi jmp normal 
```

这种技术在处理异常情况时非常实用，在高级代码中，你可以给编译器一个 提示，表明某个分支比另一个分支更有可能：

```cpp
int length(int x, int y) {  if (x > y) [[unlikely]] swap(x, y); return y - x; } 
```

这种优化只有在你知道分支很少被取用的情况下才是有益的。当情况不是这样时，有其他方面比代码布局更重要，这迫使编译器避免任何分支——在这种情况下，通过替换为特殊的“条件移动”指令来实现，大致对应于三元表达式`(x > y ? y - x : x - y)`或调用`abs(x - y)`：

```cpp
length:  mov   edx, edi mov   eax, esi sub   edx, esi sub   eax, edi cmp   edi, esi cmovg eax, edx  ; "mov if edi > esi" ret 
```

消除分支是一个重要的话题，我们将在下一章的很大一部分内容中更详细地讨论它。

* * *

1.  这样的指令被称为空操作，或 NOP 指令。在 x86 架构中，执行空操作的“官方方法”是`xchg rax, rax`（交换一个寄存器与其自身）：CPU 会识别它，并且不会在执行阶段之外额外消耗周期，除了解码阶段。`nop`缩写映射到相同的机器代码。↩︎ [← 间接分支](https://en.algorithmica.org/hpc/architecture/indirect/)[../指令级并行处理 →](https://en.algorithmica.org/hpc/pipelining/)
