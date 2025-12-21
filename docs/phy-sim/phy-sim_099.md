# 摘要

> 原文：[`phys-sim-book.github.io/lec19.4-summary.html`](https://phys-sim-book.github.io/lec19.4-summary.html)

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.css">

基于时间和空间离散化的弱形式，我们探讨了在线性有限元设置下计算质量矩阵、变形梯度和弹性力的方法，所有这些都与我们在第无逆弹性节中的实现相一致。

在线性有限元中，世界空间坐标 x 被近似为 X 的分段线性函数。这种近似，x^(X)，在每个三角形内部是线性函数，并且在边缘上是 C0-连续的。通过使用两个参数 β 和 γ 来表示每个三角形上的点，我们可以识别出插值三角形顶点位移的线性形状函数，并推导出变形梯度 F。然后可以通过对 β 和 γ 进行积分来计算质量矩阵的条目和弹性项。 
