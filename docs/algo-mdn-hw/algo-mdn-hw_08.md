# 函数和递归

> 原文：[`en.algorithmica.org/hpc/architecture/functions/`](https://en.algorithmica.org/hpc/architecture/functions/)

在汇编中“调用函数”时，你需要跳转到其开始处，然后跳转回来。但随后出现了两个重要问题：

1.  如果调用者将数据存储在调用者相同的寄存器中怎么办？

1.  “返回”在哪里？

通过在内存中有一个专门的位置，我们可以在这里写入在调用函数之前需要返回的所有信息，这两个问题都可以得到解决。这个位置被称为*栈*。

### [#](https://en.algorithmica.org/hpc/architecture/functions/#the-stack)栈

硬件栈的工作方式与软件栈相同，并且类似地仅由两个指针实现：

+   *基指针*标记栈的起始位置，并且传统上存储在`rbp`中。

+   *栈指针*标记栈的最后一个元素，并且传统上存储在`rsp`中。

当你需要调用一个函数时，你需要将所有局部变量推送到栈上（你也可以在其他情况下这样做；例如，当你用完寄存器时），推送当前指令指针，然后跳转到函数的开始处。当从函数退出时，你查看存储在栈顶的指针，跳转到那里，然后仔细将存储在栈上的所有变量读回到它们的寄存器中。

你可以使用常规的内存操作和跳转来实现这些功能，但由于其频繁使用，为此有 4 个特殊指令：

+   `push`在栈指针处写入数据并递减它。

+   `pop`从栈指针读取数据并递增它。

+   `call`将后续指令的地址放在栈顶并跳转到标签。

+   `ret`从栈顶读取返回地址并跳转到它。

如果它们不是实际的硬件指令，你会称它们为“语法糖”——它们只是这两个指令片段的融合等效：

```cpp
; "push rax" sub rsp, 8 mov QWORD PTR[rsp], rax   ; "pop rax" mov rax, QWORD PTR[rsp] add rsp, 8   ; "call func" push rip ; <- instruction pointer (although accessing it like that is probably illegal) jmp func   ; "ret" pop  rcx ; <- choose any unused register jmp rcx 
```

`rbp`和`rsp`之间的内存区域被称为*栈帧*，通常函数的局部变量存储在这里。它在程序开始时预先分配，如果你在栈上推送的数据超过了其容量（Linux 默认为 8MB），你会遇到*栈溢出*错误。因为现代操作系统实际上只有在读取或写入其地址空间时才会给你内存页面，所以你可以自由地指定一个非常大的栈大小，这更像是对可以使用多少栈内存的限制，而不是每个程序都必须使用的固定数量。

### [#](https://en.algorithmica.org/hpc/architecture/functions/#calling-conventions)调用约定

开发编译器和操作系统的那些人最终制定了一套[约定](https://wiki.osdev.org/Calling_Conventions)，用于如何编写和调用函数。这些约定使得一些重要的软件工程奇迹成为可能，例如将编译拆分为独立的单元、重用已编译的库，甚至可以用不同的编程语言编写。

考虑以下 C 语言的示例：

```cpp
int square(int x) {  return x * x; }   int distance(int x, int y) {  return square(x) + square(y); } 
```

按照惯例，一个函数应该在其`rdi`、`rsi`、`rdx`、`rcx`、`r8`、`r9`（如果不够，则其余部分在栈上）中接收参数，将返回值放入`rax`，然后返回。因此，作为简单单参数函数的`square`可以这样实现：

```cpp
square:             ; x = edi, ret = eax  imul edi, edi mov  eax, edi ret 
```

每次我们从`distance`调用它时，我们只需要做一些麻烦来保留其局部变量：

```cpp
distance:           ; x = rdi/edi, y = rsi/esi, ret = rax/eax  push rdi push rsi call square     ; eax = square(x) pop  rsi pop  rdi   mov  ebx, eax   ; save x² mov  rdi, rsi   ; move new x=y   push rdi push rsi call square     ; eax = square(x=y) pop  rsi pop  rdi   add  eax, ebx   ; x² + y² ret 
```

还有更多细微之处，但在这里我们不会深入探讨，因为这本书是关于性能的，而处理函数调用的最佳方式实际上是从一开始就避免它们。

### [#](https://en.algorithmica.org/hpc/architecture/functions/#inlining) 内联

将数据在栈之间移动为这些小型函数创建了明显的开销。你必须这样做的原因是，通常情况下，你不知道被调用者是否正在修改你存储局部变量的寄存器。但是当你能够访问`square`的代码时，你可以通过将数据存储在你知道不会被修改的寄存器中来解决这个问题。

```cpp
distance:  call square mov  ebx, eax mov  edi, esi call square add  eax, ebx ret 
```

这比之前好，但我们仍然隐式地访问栈内存：你需要在每次函数调用时推送和弹出指令指针。在像这样的简单情况下，我们可以通过将调用者的代码缝合到被调用者中并解决寄存器冲突来内联函数调用。在我们的例子中：

```cpp
distance:  imul edi, edi       ; edi = x² imul esi, esi       ; esi = y² add  edi, esi mov  eax, edi       ; there is no "add eax, edi, esi", so we need a separate mov ret 
```

这与优化编译器从这段代码中产生的结果非常接近——只是它们使用了 lea 技巧来使生成的机器代码序列小几字节：

```cpp
distance:  imul edi, edi       ; edi = x² imul esi, esi       ; esi = y² lea  eax, [rdi+rsi] ; eax = x² + y² ret 
```

在这种情况下，函数内联显然是有益的，编译器通常自动执行这一操作，但也有一些情况不适合内联——我们将在稍后讨论它们这些情况。

### [#](https://en.algorithmica.org/hpc/architecture/functions/#tail-call-elimination) 尾调用消除

当被调用者不进行任何其他函数调用，或者至少这些调用不是递归的时，内联操作很简单。让我们来看一个更复杂的例子。考虑以下阶乘的递归计算：

```cpp
int factorial(int n) {  if (n == 0) return 1; return factorial(n - 1) * n; } 
```

等价的汇编代码：

```cpp
; n = edi, ret = eax factorial:  test edi, edi   ; test if a value is zero jne  nonzero    ; (the machine code of "cmp rax, 0" would be one byte longer) mov  eax, 1     ; return 1 ret nonzero:  push edi        ; save n to use later in multiplication sub  edi, 1 call factorial  ; call f(n - 1) pop  edi imul eax, edi ret 
```

如果函数是递归的，通过重构它，仍然经常可以使其“无调用”。这种情况发生在函数是**尾递归**时，即它在进行递归调用后立即返回。由于调用后不需要执行任何操作，因此也不需要在栈上存储任何东西，递归调用可以安全地替换为跳转到开始处——实际上将函数转换成了一个循环。

要使我们的`factorial`函数成为尾递归，我们可以向它传递一个“当前乘积”参数：

```cpp
int factorial(int n, int p = 1) {  if (n == 0) return p; return factorial(n - 1, p * n); } 
```

然后，这个函数可以很容易地折叠成一个循环：

```cpp
; assuming n > 0 factorial:  mov  eax, 1 loop:  imul eax, edi sub  edi, 1 jne  loop ret 
```

递归之所以可能慢，主要原因是它需要读写数据到栈中，而迭代和尾递归算法则不需要。这个概念在函数式编程中非常重要，因为在函数式编程中没有循环，你只能使用函数。如果没有尾调用消除，函数式程序将需要更多的时间和内存来执行。[← 循环和条件](https://en.algorithmica.org/hpc/architecture/loops/)[间接分支 →](https://en.algorithmica.org/hpc/architecture/indirect/)
