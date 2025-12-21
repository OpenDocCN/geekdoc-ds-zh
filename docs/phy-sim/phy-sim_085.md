# 弱形式的离散化

> 原文：[`phys-sim-book.github.io/lec17-disc_weak_form.html`](https://phys-sim-book.github.io/lec17-disc_weak_form.html)

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.css">

在本讲座中，我们将对动量守恒方程的弱形式（暂时忽略体力）在空间和时间上进行离散化，以达到离散形式——在第一讲中引入的方程组。

我们将首先关注特定的时间点，t=tn。从动量守恒方程的弱形式（方程 (16.3.6)），我们有：∫Ω0R0(X)Qin(X)Ain(X)dX=∫∂Ω0Qin(X)Tin(X)ds(X)−∫Ω0Qi,jn(X)Pijn(X)dX,​(17.1) 对于任意的 Qn(X)，其中上标 n 表示在 t=tn 时测量的量。这里：

+   R 和 T 由模拟设置指定，

+   P 可以通过自由度 x 通过本构关系计算得出，

+   A=∂t2/∂x2 是 x 的二阶时间导数，并且

+   Q 是一个任意向量场。
