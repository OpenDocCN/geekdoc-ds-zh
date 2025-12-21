# 案例研究：方形在斜坡上*

> 原文：[`phys-sim-book.github.io/lec10-square_on_slope.html`](https://phys-sim-book.github.io/lec10-square_on_slope.html)

`<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.css">`

在本节中，基于我们从摩擦接触中学到的知识，我们在优化时间积分框架内实现了斜坡的摩擦接触。我们首先将用于方形下降案例研究中水平地面的接触模型扩展，以适应任意方向和位置的斜坡。

在此扩展之后，我们实现了斜坡的摩擦，通过模拟一个弹性方形掉落到其上进行了测试。根据摩擦系数 μ，方形要么在斜坡上的各个点停止，要么继续滑动。

本节的可执行 Python 项目可在 `4_friction` 文件夹下的 [`github.com/phys-sim-book/solid-sim-tutorial`](https://github.com/phys-sim-book/solid-sim-tutorial) 找到。[MUDA](https://github.com/MuGdxy/muda) GPU 实现可在 `simulators/4_friction` 文件夹下的 [`github.com/phys-sim-book/solid-sim-tutorial-gpu`](https://github.com/phys-sim-book/solid-sim-tutorial-gpu) 找到。
