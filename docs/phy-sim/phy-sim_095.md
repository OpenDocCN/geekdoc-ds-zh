# 线性有限元

> 原文：[`phys-sim-book.github.io/lec19-linear_FEM.html`](https://phys-sim-book.github.io/lec19-linear_FEM.html)

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.css">

从连续设置中的控制方程中，我们使用后向欧拉时间积分规则推导出离散化弱形式系统（nd 个方程）：

Ma^bΔt2xb∣^n−(xb∣^n−1+hVb∣^n−1)​=∫∂Ω0Na^(X)Ti^(X,tn)ds(X)−∫Ω0Na^,j(X)Pi^j(X,tn)dX.​(19.1) 在本章中，我们将首先讨论线性有限元中的形状函数 Na^。这一探索将帮助我们理解无逆弹性中详细描述的底层实现。

我们将特别关注单纯形有限元。在二维中，2-单纯形是一个三角形，我们在这本书中一直使用三角形网格将固体域离散化成不相交的三角形元素集合。

> ****定义 19.1（单纯形）****
> 
> 一个 n-单纯形是一个具有 n+1 个顶点的几何对象，存在于 n 维空间中。它不能适应任何较小维度的空间。
