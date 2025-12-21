# 2D 摩擦自接触*

> 原文：[`phys-sim-book.github.io/lec22-2d_self_fric.html`](https://phys-sim-book.github.io/lec22-2d_self_fric.html)



在本讲座中，我们基于我们在 案例研究：2D 自接触 中的 2D 自接触实现来实现 2D 摩擦。本部分的可执行 Python 项目可以在 [`github.com/phys-sim-book/solid-sim-tutorial`](https://github.com/phys-sim-book/solid-sim-tutorial) 找到。

为了简化，我们将专注于实现摩擦的半隐式版本。这意味着法向力大小 λ 和切向算子 T 将被离散化到最后一步，并且我们将在每个时间步长中只进行一次优化求解，而不进行进一步固定点迭代，这些迭代收敛到具有完全隐式摩擦的解（摩擦接触），位于 `8_self_friction` 文件夹下。在 `simulators/8_self_friction` 文件夹下可以找到 [MUDA](https://github.com/MuGdxy/muda) GPU 实现。[`github.com/phys-sim-book/solid-sim-tutorial-gpu`](https://github.com/phys-sim-book/solid-sim-tutorial-gpu) 也可以找到。结合 IPC 中的平滑近似静态-动态摩擦过渡，将摩擦引入优化时间积分框架就像添加额外的势能一样简单。
