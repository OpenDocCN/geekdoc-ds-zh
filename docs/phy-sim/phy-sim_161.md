# 基于位置的动力系统框架

> 原文：[`phys-sim-book.github.io/lec31-position_based_dynamics.html`](https://phys-sim-book.github.io/lec31-position_based_dynamics.html)

``

**本讲座作者：[Žiga Kovačič](https://zzigak.github.io)，康奈尔大学**

本节介绍了基于位置的动力系统（PBD）。在物理模拟领域，存在几种主导范式，每种范式都有不同的抽象级别来描述运动。

最经典的方法是基于力的，它直接模拟牛顿第二定律。在基于力的范式下，内部和外部力被累积来确定粒子的加速度，然后通过数值积分来找到新的速度和位置。虽然这些方法在物理上准确，但由于需要解决大型非线性系统，它们可能会遭受昂贵的计算成本。

基于位置的方法完全绕过速度和加速度层，直接作用于粒子的位置。核心思想是通过一组几何约束来定义系统的行为。模拟循环首先根据每个粒子的当前速度预测其新的位置。然后，迭代求解器直接调整这些预测位置，以确保所有约束都得到满足。这个过程通过约束投影代替了力的显式积分，导致高度高效且视觉上可信的模拟，这使得 PBD 成为经典动力学框架的有效替代方案。

本节受到[[Bender et al. 2017]](bibliography.html#bender2017survey)和[[Macklin & Muller 2013]](bibliography.html#macklin2013position)关于 PBD 的令人难以置信的笔记的启发。
