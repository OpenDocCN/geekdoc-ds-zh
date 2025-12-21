# 摘要

> 原文：[`phys-sim-book.github.io/lec10.3-summary.html`](https://phys-sim-book.github.io/lec10.3-summary.html)

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.css">

在本案例研究中，我们在模拟对象和斜面之间实现了半隐式摩擦，适应了任意方向和位置。在 IPC 的优化时间积分框架中，摩擦也使用势能进行建模。关键区别在于，对于半隐式离散化，每个时间步开始时预先计算了法向力大小和切向算子。

在下一讲中，我们将介绍移动边界条件。这涉及到障碍物或边界节点以规定的方式移动，主动地将动力学注入场景中。
