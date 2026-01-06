# 循环和条件语句

> 原文：[`en.algorithmica.org/hpc/architecture/loops/`](https://en.algorithmica.org/hpc/architecture/loops/)

让我们考虑一个稍微复杂一些的例子：

```cpp
loop:  add  edx, DWORD PTR [rax] add  rax, 4 cmp  rax, rcx jne  loop 
```

它计算了一个 32 位整数数组的总和，就像简单的 `for` 循环一样。

循环的“主体”是 `add edx, DWORD PTR [rax]`：这条指令从迭代器 `rax` 加载数据并将其加到累加器 `edx` 上。接下来，我们使用 `add rax, 4` 将迭代器向前移动 4 个字节。然后，发生了一个稍微复杂一些的事情。

### [#](https://en.algorithmica.org/hpc/architecture/loops/#jumps) 跳转

汇编语言没有 if-s、for-s、函数或其他高级语言中的控制流结构。它所拥有的只是 `goto`，或者称为“跳转”，这是在低级编程领域中的叫法。

**跳转**将指令指针移动到由其操作数指定的位置。这个位置可以是内存中的绝对地址，相对于当前地址，甚至可以在运行时计算。为了避免直接管理这些地址的麻烦，你可以用字符串后跟 `:` 标记任何指令，然后使用这个字符串作为标签，当转换为机器代码时，这个标签会被替换为该指令的相对地址。

标签可以是任何字符串，但编译器并不富有创意，[通常](https://godbolt.org/z/T45x8GKa5) 只使用源代码中的行号和带有签名的函数名来选择标签名称。

**无条件**跳转 `jmp` 只能用来实现 `while (true)` 类型的循环或将程序的一部分连接起来。一系列**条件**跳转用于实现实际的控制流。

有理由认为这些条件是在某处计算为 `bool` 类型的值，并将它们作为操作数传递给条件跳转：毕竟，这是在编程语言中是如何工作的。但在硬件中并不是这样实现的。条件操作使用一个特殊的 `FLAGS` 寄存器，首先需要通过执行执行某些检查的指令来填充它。

在我们的例子中，`cmp rax, rcx` 比较迭代器 `rax` 与数组结束指针 `rcx`。这会更新 `FLAGS` 寄存器，现在它可以由 `jne loop` 使用，它查找那里的一定位，以确定两个值是否相等，然后要么跳回到开始，要么继续到下一个指令，从而打破循环。

### [#](https://en.algorithmica.org/hpc/architecture/loops/#loop-unrolling) 循环展开

你可能已经注意到了上面的循环有一个很大的开销来处理单个元素。在每次循环中，只有一个有用的指令被执行，其他 3 个指令是在增加迭代器并尝试确定我们是否已经完成。

我们可以做的就是对循环进行展开，将迭代分组在一起——相当于在 C 语言中编写如下内容：

```cpp
for (int i = 0; i < n; i += 4) {  s += a[i]; s += a[i + 1]; s += a[i + 2]; s += a[i + 3]; } 
```

在汇编语言中，它看起来可能像这样：

```cpp
loop:  add  edx, [rax] add  edx, [rax+4] add  edx, [rax+8] add  edx, [rax+12] add  rax, 16 cmp  rax, rsi jne  loop 
```

现在我们只需要 3 条循环控制指令来处理 4 个有用的指令（从效率的$\frac{1}{4}$提升到$\frac{4}{7}$），并且这可以继续减少开销，几乎接近于零。

在实践中，展开循环并不总是对性能必要，因为现代处理器实际上并不是逐条执行指令，而是维护一个待执行指令队列，这样两个独立的操作就可以并发执行，无需等待对方完成。

这也是我们的情况：展开循环的实际加速不会是四倍，因为增加计数器和检查是否完成的操作与循环体是独立的，并且可以安排与它同时运行。但仍然可能有益于请求编译器在一定程度上展开它。

### [#](https://en.algorithmica.org/hpc/architecture/loops/#an-alternative-approach)另一种方法

你不必显式使用`cmp`或类似的指令来执行条件跳转。许多其他指令要么读取要么修改`FLAGS`寄存器，有时作为副产品启用可选的异常检查。

例如，`add`指令总是设置一系列标志，表示结果是否为零、是否为负、是否发生溢出或下溢等。利用这一机制，编译器通常会生成如下循环：

```cpp
 mov  rax, -100  ; replace 100 with the array size loop:  add  edx, DWORD PTR [rax + 100 + rcx] add  rax, 4 jnz  loop       ; checks if the result is zero 
```

这段代码对人类来说读起来有点困难，但在重复部分中指令数量少了一条，这可能会对性能产生有意义的影响。[← 汇编语言](https://en.algorithmica.org/hpc/architecture/assembly/)[函数和递归 →](https://en.algorithmica.org/hpc/architecture/functions/)
