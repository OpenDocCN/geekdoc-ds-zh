# 摘要

> 原文：[`phys-sim-book.github.io/lec9.4-summary.html`](https://phys-sim-book.github.io/lec9.4-summary.html)

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.css">

我们引入了库仑摩擦模型，该模型通过切向空间中的静摩擦力和动摩擦力非平滑地惩罚接触点的剪切运动。

为了将摩擦力整合到优化时间积分器中，我们首先平滑地近似动态-静态过渡。这使得仅使用节点速度自由度就能唯一确定摩擦力。

然后，我们应用了一种半隐式离散化方法，该方法在上一时间步固定法向力大小λ和切向算子 T，增强了摩擦力的可积性。

为了实现具有完全隐式摩擦的解，我们执行了不动点迭代。这些迭代在半隐式时间积分和λ和 T 的更新之间交替。

在下一讲中，我们将探讨一个涉及斜坡上具有变化摩擦系数的方形案例研究。
