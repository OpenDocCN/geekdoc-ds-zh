# 案例研究：2D 质量弹簧*

> 原文：[`phys-sim-book.github.io/lec4-2d_mass_spring.html`](https://phys-sim-book.github.io/lec4-2d_mass_spring.html)

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.css">

到目前为止，我们已经完成了基于优化的固体模拟框架的高级介绍。在本讲中，我们详细阐述了如何使用 [Python3](https://www.python.org/) (CPU) 和 [MUDA](https://github.com/MuGdxy/muda) (GPU) 实现一个简单的二维弹性动力学模拟器。

本书中有 Python CPU 和 MUDA GPU 实现的章节将在标题后标记一个 *。所有 Python 和 MUDA 的实现可以在 [`github.com/phys-sim-book/solid-sim-tutorial`](https://github.com/phys-sim-book/solid-sim-tutorial) 和 [`github.com/phys-sim-book/solid-sim-tutorial-gpu`](https://github.com/phys-sim-book/solid-sim-tutorial-gpu) 分别找到。本节的可执行项目位于这些存储库的 `/1_mass_spring` 文件夹中。
