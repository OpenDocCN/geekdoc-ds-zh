# 简化模型与可逆性

> 原文：[`phys-sim-book.github.io/lec13.3-simp_model_inversion.html`](https://phys-sim-book.github.io/lec13.3-simp_model_inversion.html)



> ****定义 13.3.1（协变线性弹性）**** 为了使线性弹性具有旋转感知性同时保持其简单性，我们可以引入一个基本旋转 Rn 并构建一个能量密度函数ΨLC(F)=Ψlin((Rn)TF)，惩罚 F 与这个固定 Rn 之间的任何偏差。这被称为协变线性弹性。

ΨLC(F)相对于 F 保持为二次能量，对于动态模拟非常有用。在每次时间步长 n+1 优化的开始，我们计算 Rn 作为 Fn 最近的旋转：Rn=argRmin∥Fn−R∥F2，s.t.RT=R−1 且 det(R)=1。(13.3.1) 如前所述，解由 Fn 上的极分解给出，并且使用极 SVD Fn=UnΣn(Vn)T，我们有 Rn=Un(Vn)T。然而，由于 Rn 在优化过程中不随 F 变化，协变线性弹性仍然不是旋转不变的。因此，它不适合大变形。

对于旋转不变的弹性模型，计算机图形学领域的实践者为了视觉计算的目的已经简化了它们。例如，为了更有效的计算，只保留能量密度函数中的μ项而忽略λ项：ΨR(F)=4μ∥FTF−I∥F2，或ΨARAP(F)=μi∑d(σi−1)2 等。(13.3.2) 这里ΨARAP(F)被称为**尽可能刚体（As-Rigid-As-Possible，ARAP）**能量，它在形状建模、布料模拟和表面参数化等方面得到广泛应用。与 ARAP 相比，ΨR(F)是 F 的高阶多项式，可以不执行 F 上的昂贵 SVD 来计算。

在本讲座中我们查看的所有应变能密度函数中，除了 Neo-Hookean，其他都是在整个域 Rd×d 上定义的。Neo-Hookean 能量密度函数定义在{F | F∈Rd×d, det(F)>0}上。就像在 IPC 中防止相互穿透的势能一样，ΨNH(F)也是一种势能，当 det(F)接近 0 时趋于无穷大，提供任意大的弹性力以防止**反转**（det(F)≤0）。

允许 det(F)≤0 的应变能密度函数也称为**可逆弹性模型**。它们容易处理（不需要线搜索滤波），但不保证非反转。设计一个提供合理大反转阻力的可逆弹性能量在计算机图形学研究领域引起了大量关注 [[Stomakhin et al. 2012]](bibliography.html#stomakhin2012energetically) [[Smith et al. 2018]](bibliography.html#smith2018stable)。
