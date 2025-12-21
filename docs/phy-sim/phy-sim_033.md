# 滑动 Dirichlet 边界条件

> 原文：[`phys-sim-book.github.io/lec6-slip_DBC.html`](https://phys-sim-book.github.io/lec6-slip_DBC.html)

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.css">

尽管它们可能在时间步的初始阶段感到满意，但一般的滑动 Dirichlet 边界条件（DBC）提出了独特的挑战。与粘性 DBC 不同，它们不能直接使用自由度消除法来解决，主要是因为它们的约束雅可比矩阵不包含单位矩阵块。为了应对这种复杂性，我们可以采用基变换策略。

在深入研究更普遍的场景之前，首先考察一种特定的滑动 DBC 类型是有洞察力的：那些与轴对齐的。理解这个特定案例将为解决更广泛的滑动 DBC 范围奠定基础。
