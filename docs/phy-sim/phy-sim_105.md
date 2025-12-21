# 案例研究：2D 自接触*

> 原文：[`phys-sim-book.github.io/lec21-2d_self_contact.html`](https://phys-sim-book.github.io/lec21-2d_self_contact.html)

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.css">

我们已经完成了将线性有限元与弹塑动力学和摩擦接触的弱形式推导相连接。现在，是时候看看这些概念如何在代码中实现了。在本讲中，我们将基于我们的 Python 开发的案例研究：无反演弹性弹性模拟，实现基于 2D 无摩擦自接触。

本节的可执行 Python 项目可以在`7_self_contact`文件夹下的[`github.com/phys-sim-book/solid-sim-tutorial`](https://github.com/phys-sim-book/solid-sim-tutorial)找到。[MUDA](https://github.com/MuGdxy/muda) GPU 实现可以在`simulators/7_self_contact`文件夹下的[`github.com/phys-sim-book/solid-sim-tutorial-gpu`](https://github.com/phys-sim-book/solid-sim-tutorial-gpu)找到。我们将在下一讲中实现摩擦自接触。
