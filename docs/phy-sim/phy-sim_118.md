# 质量矩阵

> 原文：[`phys-sim-book.github.io/lec23.2-mass_matrix.html`](https://phys-sim-book.github.io/lec23.2-mass_matrix.html)



回想一下，质量矩阵可以计算为 Mab = ∑∫Ω e∈T R(X,0)Na(X)Nb(X)dX，其中 Ωe0 表示四面体 e 的材料空间。将积分变量从 X 变换为 (β,γ,τ) 得到 = ∫Ωe0 R(X,0)Na(X)Nb(X)dX ∫0¹ ∫0^(1−τ) ∫0^(1−β−τ) R(β,γ,τ,0)Na(β,γ,τ)Nb(β,γ,τ) det(∂(β,γ,τ)/∂X) dγdβdτ。

对于具有顶点 X1, X2, X3 和 X4 的元素 e，det(∂(β,γ,τ)/∂X) = |det([X2−X1, X3−X1, X4−X1])| = 6Ve，其中 Ve 是四面体 e 的体积。

在这里，我们将省略一致质量矩阵中每个元素的详细推导。假设密度 R 均匀，对于集中质量矩阵，Ma lump = ∑e∈T(a) 4RVe 和 Mab lump = 0 (a ≠ b)，其中 T(a) 表示与节点 a 相连的四面体集合。换句话说，每个四面体的质量均匀分布在它的 4 个节点上，这在直观上类似于二维情况。
