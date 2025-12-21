# 案例研究：无反演弹性*

> 原文：[`phys-sim-book.github.io/lec15-inv_free_elasticity.html`](https://phys-sim-book.github.io/lec15-inv_free_elasticity.html)



在本章末尾，我们将前几节课中介绍的 Neo-Hookean 模型应用于模拟无反演弹性固体。本节的可执行 Python 项目可在[`github.com/phys-sim-book/solid-sim-tutorial`](https://github.com/phys-sim-book/solid-sim-tutorial)下的`6_inv_free`文件夹中找到。在`simulators/6_inv_free`文件夹下可以找到[MUDA](https://github.com/MuGdxy/muda) GPU 实现。与在质量-弹簧模型中将弹性离散到弹簧上不同，我们将 Neo-Hookean 模型离散到三角形元素上，应用链式法则根据变形梯度 F 和世界空间节点位置 x 之间的关系计算弹性力，然后开发了一种基于根查找的方法来过滤线性搜索的初始步长，以确保非反演。
