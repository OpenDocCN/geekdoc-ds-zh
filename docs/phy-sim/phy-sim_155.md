# 摘要

> 原文：[`phys-sim-book.github.io/lec29.3-summary.html`](https://phys-sim-book.github.io/lec29.3-summary.html)



我们成功实现了一个最小化但完整的 2D 材料点法（MPM）模拟，其中包括两个碰撞的弹性块。这个设置展示了 MPM 的核心流程，包括粒子采样、数据结构初始化、基于 PIC（粒子图像法）的传输方案，以及在 Hencky 应变空间中基于 StVK 本构模型的弹性变形。

尽管简单，这个例子却捕捉了 MPM（Material Point Method，材料点法）的关键方面。它作为一个干净且可扩展的基础，用于构建更复杂的 MPM 系统。

在下一讲中，我们将通过引入 APIC 传输方案、Drucker-Prager 塑性和基于 SDF（距离场）的边界处理，在这个框架的基础上模拟 2D 沙子与静态球体碰撞器相互作用，从而实现具有摩擦接触的颗粒材料的真实建模。
