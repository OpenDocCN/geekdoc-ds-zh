# 压力

> 原文：[`phys-sim-book.github.io/lec14.1-stress.html`](https://phys-sim-book.github.io/lec14.1-stress.html)

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.css">

压力是一个张量场，类似于变形梯度 F，它在整个固体材料的域上定义。它量化了材料物体所经历的内部压力和张力。压力与应变（或 F）之间的关系是通过所谓的**本构关系**建立的。这种关系概述了材料如何响应各种变形。

本构关系的常见例子是一维中的胡克定律，它在弹性条件下适用于许多传统材料。在**超弹性材料**的背景下，这种关系由应变能函数 Ψ(F) 特定地定义。

> ****定义 14.1.1（超弹性材料）.**** 超弹性材料是那些弹性固体，其**第一个 Piola-Kirchhoff 应力** P 可以通过应变能密度函数 Ψ(F) 导出，即 P=∂F/∂Ψ。(14.1.1) 用指标表示，这意味着 Pij=∂Fij/∂Ψ。P 是一个与 F 相同维度的离散小矩阵。

在研究材料在应力下的行为时，使用了各种定义，其中**柯西应力**在工程背景下尤为普遍。柯西应力，表示为 σ(⋅,t):Ωt→Rd×d，可以通过以下关系与第一个 Piola-Kirchhoff 应力张量 P 相联系：σ=J1PFT=det(F)1∂F/∂ΨFT。

从应变能函数 Ψ(F) 计算出 P 对于不需要奇异值分解（SVD）的能量模型来说相对简单，例如 Neo-Hookean 模型。然而，像 ARAP（As-Rigid-As-Possible）这样的通用各向同性弹性模型通常依赖于主拉伸或最接近旋转矩阵的计算，这需要 SVD。当确定 ∂F/∂P 时，这种计算变得特别复杂和资源密集，这对于隐式时间积分至关重要。

我们提出了一种有效的方法，该方法利用了由 [[Stomakhin 等人 2012]](bibliography.html#stomakhin2012energetically) 引入的稀疏结构，来计算一般各向同性弹性材料的第一个 Piola-Kirchhoff 应力张量 P 及其导数 ∂F/∂P（无论是作为张量还是微分 δP）。这种方法利用了符号软件包，我们将特别讨论在 *Mathematica* 中的实现。在 *Maple* 或其他软件中的实现同样简单，遵循相同的概念框架。对于计算机图形学中常用的导数计算的更深入探索，请参阅 [[Schroeder 2022]](bibliography.html#schroeder2022practical) 的作品。

需要注意的是，所讨论的计算策略也可以应用于对角空间中的其他导数，类似于 ∂F∂P。例如，在某些模型中，**柯西应力** τ 被优先考虑，而不是第一 Piola-Kirchhoff 应力 P。柯西应力表示为：τ=Uτ^UT，其中 τ^ 是一个对角应力度量，每个条目都是奇异值 Σ 的函数。计算 ∂F∂τ 的方法与 P 相似。
