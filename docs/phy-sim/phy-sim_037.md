# 摘要

> 原文：[`phys-sim-book.github.io/lec6.4-summary.html`](https://phys-sim-book.github.io/lec6.4-summary.html)

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.css">

本节已证明，通过改变变量基，可以使用自由度（DOF）消除法有效地管理一般滑移 Dirichlet 边界条件（DBC），就像轴对齐的滑移 DBC 一样。

虽然奇异值分解（SVD）可以用来找到一般线性等式约束的基，但这种方法可能不适用于大型或复杂的约束。尽管如此，还是可以开发出针对节点滑移 DBC 约束的特定程序性例程来计算基。

目前，我们的关注点一直是在模拟框架内维护已经满足的 DBC。展望未来，讨论将转向探索点与解析表面之间的摩擦接触。此外，我们还将重新审视那些在时间步开始时未满足 DBC 的场景，深入研究更复杂的情况。
