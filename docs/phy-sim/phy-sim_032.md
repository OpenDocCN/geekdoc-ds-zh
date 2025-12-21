# 摘要

> 原文：[`phys-sim-book.github.io/lec5.4-summary.html`](https://phys-sim-book.github.io/lec5.4-summary.html)

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.css">

在本节中，我们探讨了 Dirichlet 边界条件（DBC），这是优化时间积分器的重要组成部分，并将它们作为简单的线性等式约束来呈现。存在两种类型的 DBC：**粘性**和**滑动**。粘性 DBC 使某些节点固定，从而固定其位置，而滑动 DBC 则限制节点在平面或直线内的移动。

我们专注于在时间步开始时已经满足粘性 DBC（Dirichlet Boundary Conditions）的案例。在这种情况下，**自由度消除法**被证明是有效的。这项技术修改了增量势的梯度和 Hessian 矩阵，确保结果搜索方向保持在可行空间内。

在接下来的讲座中，我们将深入探讨处理滑动 DBC（Dirichlet Boundary Conditions）的方法，并展示将它们有效纳入优化问题的方法。
