# 外部内存模型

> 原文：[`en.algorithmica.org/hpc/external-memory/model/`](https://en.algorithmica.org/hpc/external-memory/model/)

为了推理内存绑定算法的性能，我们需要开发一个成本模型，它对昂贵的块 I/O 操作更敏感，但又不至于过于严格，仍然有用。

### [#](https://en.algorithmica.org/hpc/external-memory/model/#cache-aware-model)缓存感知模型

在标准 RAM 模型中，我们忽略了基本操作完成所需时间不等的事实。最重要的是，它不区分不同类型内存上的操作，将 RAM 读取大约需要 50ns 的时间等同于 HDD 读取大约需要 5ms 的时间，或者说大约是$10⁵$倍。

在精神上类似，在外部内存模型中，我们简单地忽略所有非 I/O 操作。更具体地说，我们考虑一个缓存层次结构级别，并假设以下关于硬件和问题的内容：

+   数据集的大小是$N$，并且全部存储在*外部*内存中，我们可以以$B$个元素为一个单元的时间（读取整个块和仅读取一个元素所需时间相同）进行读写。

+   我们可以在*内部*内存中存储$M$个元素，这意味着我们可以存储多达$\left \lfloor \frac{M}{B} \right \rfloor$个块。

+   我们只关心 I/O 操作：在读取和写入之间的任何计算都是免费的。

+   我们还假设$N \gg M \gg B$。

在这个模型中，我们衡量算法的性能是通过其高级*I/O 操作*，或*IOPS*——也就是说，在执行过程中读入或写入外部内存的总块数。

我们将主要关注内部内存是 RAM 而外部内存是 SSD 或 HDD 的情况，尽管我们将开发的底层分析技术适用于缓存层次结构的任何层。在这些设置下，合理的块大小$B$约为 1MB，内部内存大小$M$通常是几 GB，而$N$高达几 TB。

### [#](https://en.algorithmica.org/hpc/external-memory/model/#array-scan)数组扫描

作为简单的例子，当我们通过逐个元素迭代数组来计算其和时，我们隐式地以$O(B)$个元素的大小加载它，在外部内存模型中，我们逐个处理这些块：

$$ \underbrace{a_1, a_2, a_3,} _ {B_1} \underbrace{a_4, a_5, a_6,} _ {B_2} \ldots \underbrace{a_{n-3}, a_{n-2}, a_{n-1}} _ {B_{m-1}} $$ 因此，在外部内存模型中，求和和其他线性数组扫描的复杂度是 $$ SCAN(N) \stackrel{\text{def}}{=} O\left(\left \lceil \frac{N}{B} \right \rceil \right) \; \text{IOPS} $$

你可以像这样显式地实现外部数组扫描：

```cpp
FILE *input = fopen("input.bin", "rb");   const int M = 1024; int buffer[M], sum = 0;   // while the file is not fully processed while (true) {  // read up to M of 4-byte elements from the input stream int n = fread(buffer, 4, M, input); //  ^ the number of elements that were actually read  // if we can't read any more elements, finish if (n == 0) break;  // sum elements in-memory for (int i = 0; i < n; i++) sum += buffer[i]; }   fclose(input); printf("%d\n", sum); 
```

注意，在大多数情况下，操作系统会自动进行缓冲。即使数据只是从普通文件重定向到标准输入，操作系统也会缓冲其流，并以约 4KB（默认）的块读取。 [← 虚拟内存](https://en.algorithmica.org/hpc/external-memory/virtual/)[外部排序 →](https://en.algorithmica.org/hpc/external-memory/sorting/)
