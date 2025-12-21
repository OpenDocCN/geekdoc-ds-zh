# 变形梯度与粒子状态更新

> 原文：[`phys-sim-book.github.io/lec26.4-particle_state_update.html`](https://phys-sim-book.github.io/lec26.4-particle_state_update.html)



在 MPM 中，每个粒子携带一个变形梯度 Fp​，它追踪粒子附着的局部材料体积随时间拉伸、旋转或剪切的情况。它在基于本构模型计算应力中起着核心作用。

在每个时间步的开始，网格位于一个规则、未变形的晶格上。令 xin​ 为时间 tn 时网格节点 i 的位置。通过观察网格在时间步内局部移动的情况来更新变形梯度，假设给定的网格速度场 v^in+1​。

### 变形梯度更新

最常见的方法是使用一阶近似来计算 Fpn+1：

Fpn+1​=(I+Δti∑​vin+1​∇(wipn​)T)Fpn​.

在**MLS-MPM** [[Hu et al. 2018]](bibliography.html#hu2018moving)和**APIC** [[Jiang et al. 2015]](bibliography.html#jiang2015affine)中，可以使用已经计算过的仿射速度矩阵 Cp​，在 P2G 和 G2P 转换期间更紧凑地重写变形梯度更新。

回忆一下局部仿射速度场：

vp​(x)=vp​+Cp​(x−xp​).

从这个结果中，网格运动诱导出局部速度梯度 Cp​，并且变形梯度可以更新为：

Fpn+1​=(I+ΔtCp​)Fpn​.(3)

这样就避免了显式评估 ∇wip​ 的需要，而是使用已经聚集在 Cp​ 中的仿射行为。

**塑性流动**也在此阶段应用，将更新的变形梯度投影回由材料的屈服准则定义的可接受空间。这个过程称为**回映射**，确保材料遵守塑性极限，将在下一讲中详细讨论。

### 位置更新

那么在时间 tn+1 时，粒子位置随后被推进：

xpn+1​=xpn​+Δtvpn+1​.
