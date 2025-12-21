# 案例研究：带有球体碰撞器的 2D 沙子*

> 原文：[`phys-sim-book.github.io/lec29-mpm_elastic_case_study.html`](https://phys-sim-book.github.io/lec29-mpm_elastic_case_study.html)



**本讲座作者：[杨昌宇](https://changyu.io/), 加州大学洛杉矶分校*

到目前为止，我们已经介绍了构建完整且物理上合理的 MPM 模拟系统所需的所有核心组件——包括材料离散化、时间积分和数据传输方案。

在本案例研究中，我们将这些组件组合起来模拟 2D 中两个相互碰撞的弹性块。我们首先通过设置模拟，包括材料属性和数据结构定义。然后，我们实现粒子-网格（PIC）传输来处理粒子与网格之间的动量交换。

本例作为使用弹性材料快速入门 MPM 管道的演示，不涉及塑性或复杂的边界条件。

在实现方面，我们使用 NumPy 和[Taichi](https://docs.taichi-lang.org/)作为我们的编程框架。Taichi 在 CPU 和 GPU 上提供高效的并行性，更重要的是，它支持 sparse data structures)，这对于高性能 MPM 网格计算至关重要。

本节的可执行 Python 项目可以在`10_mpm_elasticity`文件夹下的[`github.com/phys-sim-book/solid-sim-tutorial`](https://github.com/phys-sim-book/solid-sim-tutorial)找到。
