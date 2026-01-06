# 计算机架构

> 原文：[`en.algorithmica.org/hpc/architecture/`](https://en.algorithmica.org/hpc/architecture/)

当我开始学习如何自己优化程序时，我犯的一个大错误是主要依赖于经验方法。由于不理解计算机的实际工作原理，我会半随机地交换嵌套循环，重新排列算术运算，组合分支条件，手动内联函数，并遵循我从其他人那里听到的所有 sorts of 其他性能提示，盲目地希望有所改进。

不幸的是，这就是大多数程序员处理优化的方式。关于性能的大部分书籍并没有教你如何从定性角度推理软件性能。相反，它们给你提供关于某些实现方法的泛泛建议——而一般的性能直觉显然是不够的。

如果我在做算法编程之前学习了计算机架构，可能节省了我数十甚至数百个小时。所以，即使大多数人对此不感兴趣，我们也将花费前几章的时间来研究 CPU 的工作原理，并从学习汇编开始。

[← 编程语言](https://en.algorithmica.org/hpc/complexity/languages/)

[← ../复杂度模型](https://en.algorithmica.org/hpc/complexity/)[指令集架构 →](https://en.algorithmica.org/hpc/architecture/isa/)

[../指令级并行 →](https://en.algorithmica.org/hpc/pipelining/)
