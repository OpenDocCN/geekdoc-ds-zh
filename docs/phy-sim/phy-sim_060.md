# 概述

> 原文：[`phys-sim-book.github.io/lec11.3-summary.html`](https://phys-sim-book.github.io/lec11.3-summary.html)

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.css">

我们引入了惩罚方法来处理移动边界条件，同时防止相互穿透。涉及的关键策略包括：

+   在 DBC 节点上增加额外的弹簧势能以增强增量势能；

+   根据需要自适应地增加惩罚刚度；

+   对于那些足够接近其目标的边界条件节点，消除自由度（DOFs）；

+   确保在收敛点满足所有边界条件（BCs）。

为了解决我们在压缩质量-弹簧弹性正方形案例研究中观察到的反演伪影，应用屏障型弹性势能是必不可少的。当应用这些势能时，我们的移动边界条件的惩罚方法起着至关重要的作用，因为直接指定边界条件节点仍然可能导致反演。在下一章中，我们将探讨超弹性模型，这些模型在实际应用中比质量-弹簧系统更受欢迎。
