# 间接分支

> 原文：[`en.algorithmica.org/hpc/architecture/indirect/`](https://en.algorithmica.org/hpc/architecture/indirect/)

在汇编过程中，所有标签都被转换为地址（绝对或相对），然后编码成跳转指令。

您还可以通过存储在寄存器中的非常量值进行跳转，这被称为*计算跳转*：

```cpp
jmp rax 
```

这有几个与动态语言和实现更复杂控制流相关的有趣应用。

### [#](https://en.algorithmica.org/hpc/architecture/indirect/#multiway-branch)多路分支

如果您已经忘记了`switch`语句的作用，这里有一个用于计算美国评分系统 GPA 的小子程序：

```cpp
switch (grade) {  case 'A': return 4.0; break; case 'B': return 3.0; break; case 'C': return 2.0; break; case 'D': return 1.0; break; case 'E': case 'F': return 0.0; break; default: return NAN; } 
```

我个人不记得上一次在非教育环境中使用`switch`语句是什么时候了。一般来说，`switch`语句相当于一系列的“if, else if, else if, else if...”等等，因此许多语言甚至没有它们。尽管如此，这样的控制流结构对于实现解析器、解释器和其它状态机来说是非常重要的，这些通常由一个`while (true)`循环和一个`switch (state)`语句组成。

当我们控制变量可以取的值的范围时，我们可以使用以下技巧，利用计算跳转。我们不是创建$n$个条件分支，而是创建一个*分支表*，其中包含指向可能的跳转位置的指针/偏移量，然后只需用`state`变量索引它，该变量在$[0, n)$范围内取值。

当值密集地打包在一起时（不一定严格按顺序，但表格中必须有空白字段是值得的），编译器会使用这种技术。它也可以通过*计算跳转*显式实现：

```cpp
void weather_in_russia(int season) {  static const void* table[] = {&&winter, &&spring, &&summer, &&fall}; goto *table[season];   winter: printf("Freezing\n"); return; spring: printf("Dirty\n"); return; summer: printf("Dry\n"); return; fall: printf("Windy\n"); return; } 
```

基于`switch`的代码并不总是对编译器优化来说简单直接，因此在状态机的上下文中，`goto`语句经常被直接使用。`glibc`的 I/O 相关部分充满了这样的例子。

### [#](https://en.algorithmica.org/hpc/architecture/indirect/#dynamic-dispatch)动态分派

间接分支对于实现运行时多态也是非常重要的。

考虑一个陈词滥调的例子，当我们有一个具有虚拟`.speak()`方法的`Animal`抽象类，以及两个具体实现：一个吠叫的`Dog`和一个喵喵叫的`Cat`：

```cpp
struct Animal {  virtual void speak() { printf("<abstract animal sound>\n");} };   struct Dog {  void speak() override { printf("Bark\n"); } };   struct Cat {  void speak() override { printf("Meow\n"); } }; 
```

我们想要创建一个动物，并且在不事先知道其类型的情况下，调用其`.speak()`方法，该方法应该以某种方式调用正确的实现：

```cpp
Dog sparkles; Cat mittens;   Animal *catdog = (rand() & 1) ? &sparkles : &mittens; catdog->speak(); 
```

实现这种行为有许多方法，但 C++使用*虚方法表*来实现。

对于`Animal`的所有具体实现，编译器都会填充它们的所有方法（即它们的指令序列），以确保所有类都具有相同的长度（通过在`ret`指令后插入一些填充指令来实现），然后只需将它们顺序地写入指令内存中的某个位置。然后，它会在结构体（即所有实例）中添加一个*运行时类型信息*字段，这本质上只是指向类虚拟方法的正确实现的内存区域中的偏移量。

使用虚拟方法调用时，该偏移字段是从结构体的实例中获取的，然后使用它进行常规函数调用，利用的事实是每个派生类的所有方法和其他字段都具有完全相同的偏移量。

当然，这会增加一些开销：

+   你可能还需要额外花费大约 15 个周期，原因与分支预测错误相同。

+   编译器很可能无法内联函数调用本身。

+   类的大小会增加几个字节左右（这取决于具体实现）。

+   二进制文件本身会稍微增大一点。

由于这些原因，在性能关键的应用中通常避免使用运行时多态。[← 函数和递归](https://en.algorithmica.org/hpc/architecture/functions/)[机器代码布局 →](https://en.algorithmica.org/hpc/architecture/layout/)
