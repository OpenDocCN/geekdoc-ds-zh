# 应力及其导数

> 原文：[`phys-sim-book.github.io/lec14-stress_and_derivatives.html`](https://phys-sim-book.github.io/lec14-stress_and_derivatives.html)

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.css">

在介绍了标准应变能之后，我们现在继续对它们相对于世界空间坐标 x 的微分进行讨论，以模拟真实的弹性行为。然而，首先建立这些坐标 x 和变形梯度 F 之间的显式关系是很重要的。这种关系在很大程度上取决于特定的离散化选择。

在我们深入探讨离散化之前，我们应该了解如何计算应变能函数 Ψ 对 F 的导数。这些导数与应力的概念基本相关联，应力是理解材料在变形下行为的一个关键要素。
