# 案例研究：带有球体碰撞器的二维沙子*

> 原文：[`phys-sim-book.github.io/lec30-mpm_sand_case_study.html`](https://phys-sim-book.github.io/lec30-mpm_sand_case_study.html)



**作者：[杨昌宇](https://changyu.io/), 加州大学洛杉矶分校**

基于前一章——二维中两个弹性块的碰撞，其中我们使用 PIC 转移方案实现了最小化材料点方法 (MPM) 模拟——本案例研究展示了如何通过最小的额外努力，将系统扩展以创建更高级的沙子模拟。

与前一章不同，其中粒子在规则网格上采样，这里我们使用 Poisson-disk 采样来初始化材料点。这有助于减少混叠伪影和结构噪声，在颗粒模拟中产生更真实的物理行为。

在本案例研究中，我们使用 Drucker-Prager 弹塑性 [[Klar 等人 2016]](bibliography.html#klar2016drucker) 本构模型来模拟沙子，这使得我们能够捕捉不可恢复的变形和内部摩擦——这是颗粒材料行为的关键特征。

我们在域内放置了一个静态球体碰撞器，它通过摩擦接触与下落的沙粒相互作用。碰撞器的边界使用**距离函数（SDF）**表示，并强制执行接触约束和库仑摩擦。

我们通过结合 APIC 转移方案扩展了原始的 PIC 方案，以实现更高的精度和减少数值耗散。

本节的可执行 Python 项目可在[`github.com/phys-sim-book/solid-sim-tutorial`](https://github.com/phys-sim-book/solid-sim-tutorial)下的`11_mpm_sand`文件夹中找到。
