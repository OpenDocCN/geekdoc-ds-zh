# 编译

> 原文：[`en.algorithmica.org/hpc/compilation/`](https://en.algorithmica.org/hpc/compilation/)

学习汇编语言的主要好处不是能够用它来编写程序，而是理解编译代码执行过程中发生的事情及其性能影响。

在极少数情况下，我们确实需要切换到手写汇编以获得最大性能，但大多数时候，编译器能够自行生成接近最优的代码。当它们做不到这一点时，通常是因为程序员对问题的了解超过了从源代码中可以推断出的信息，但未能将这额外信息传达给编译器。

在本章中，我们将讨论如何让编译器精确地完成我们想要的工作，以及如何收集有用的信息来指导进一步的优化。

[← 吞吐量计算](https://en.algorithmica.org/hpc/pipelining/throughput/)

[← ../指令级并行](https://en.algorithmica.org/hpc/pipelining/)[编译阶段 →](https://en.algorithmica.org/hpc/compilation/stages/)

[../性能分析 →](https://en.algorithmica.org/hpc/profiling/)
