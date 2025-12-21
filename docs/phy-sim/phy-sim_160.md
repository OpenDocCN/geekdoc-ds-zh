# 摘要

> 原文：[`phys-sim-book.github.io/lec30.4-summary.html`](https://phys-sim-book.github.io/lec30.4-summary.html)



我们已经将我们的弹性 MPM 框架扩展到通过整合**泊松盘采样**、**APIC 转移**、**Drucker-Prager 弹塑性**和**基于 SDF 的摩擦接触**来模拟颗粒材料。APIC 方案通过捕捉局部仿射运动，在提高精度的同时保持稳定性，而 Drucker-Prager 模型则实现了逼真的塑性流动和压力相关屈服。引入了一个使用符号距离函数（SDF）的静态球体碰撞器，允许平滑且稳健地实施接触和摩擦约束。这些改进共同使得对经历大变形和飞溅的沙子进行稳定且物理上合理的模拟成为可能。
