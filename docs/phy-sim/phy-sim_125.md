# 摘要

> 原文：[`phys-sim-book.github.io/lec24.4-summary.html`](https://phys-sim-book.github.io/lec24.4-summary.html)

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.css">

在本节中，我们讨论了基于隐式接触预测（IPC）实现三维接触处理方法的主要技术细节。

在三维空间中，距离和摩擦基础的计算变得更加复杂。这些计算依赖于点-三角形和边-边的基本对，类似于在二维空间中使用的点-边对。

对于仅 C0 连续的边-边距离，需要一个额外的平滑减少到零的磨光函数。这个函数与屏障能量密度函数相乘，以实现 C1 连续性，从而可以使用基于梯度的优化方法。

由于三维空间中基本对的数量显著增加，因此在广相阶段通常使用空间哈希或边界体积层次结构（BVH）等空间数据结构来过滤候选对象，在计算距离或执行 CCD 之前。
