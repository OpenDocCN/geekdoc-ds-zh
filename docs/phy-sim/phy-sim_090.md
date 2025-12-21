# 引入边界条件

> 原文：[`phys-sim-book.github.io/lec18.1-incorporate_BC.html`](https://phys-sim-book.github.io/lec18.1-incorporate_BC.html)



在我们推导的弱形式（见方程 (16.3.6)）中，有一个边界项 ∫∂Ω0 Qi(X,t)Ti(X,t)ds(X)，它描述了从外部对固体边界的力作用。

如果没有 Dirichlet 边界条件，整个边界将使用 **Neumann 边界条件** 处理，其中边界力作为问题设置的一部分指定。回想一下，我们讨论了 Dirichlet 边界条件，其中边界位移是直接规定的。在实践中，外部力作用于 Dirichlet 边界，以确保它们的位移精确地匹配规定的值，这些力直接从这些位移计算得出。

在固体模拟问题中，边界可以是 Dirichlet 边界或 Neumann 边界，这可以通过强形式中的更一般问题公式描述：R(X,0)∂t∂V(X,t)=∇X⋅P(X,t)+R(X,0)Aext(X,t)，∀X∈Ω0 和 t≥0；x=xD(X,t)，∀X∈ΓD 和 t≥0；P(X,t)N(X)=TN(X,t)，∀X∈ΓN 和 t≥0。（18.1.1）

这里 ΓN 和 ΓD 分别是 Neumann 和 Dirichlet 边界，ΓN∪ΓD=∂Ω0，ΓN∩ΓD=∅，xD 和 TN 是已知的。在我们推导动量守恒的弱形式（见方程 (18.1.1)，第一行）之后，边界项 ∫∂Ω0 Qi(X,t)Ti(X,t)ds(X) 可以分别考虑 Dirichlet 和 Neumann 边界：∫∂Ω0 Qi(X,t)Ti(X,t)ds(X)=∫ΓD Qi(X,t)TD∣i(X,t)ds(X)+∫ΓN Qi(X,t)TN∣i(X,t)ds(X)。

对于 Neumann 边界，由于提供了牵引力 TN(X,t)，上述积分在离散化后可以直接计算。然而，对于 Dirichlet 边界，TD(X,t) 在我们解决问题之前仍然是未知的。因此，一种直接的方法是将 Dirichlet 边界上的牵引力作为未知数，并求解包含离散化弱形式方程和 Dirichlet 边界条件的系统。

> ***备注 18.1.1（优化形式）***。在 优化形式 中，势能不包括任何 Dirichlet 边界，实际上忽略了弱形式中的边界积分。这是有效的，因为 Dirichlet 边界条件将通过线性等式约束强制执行，相应的离散化弱形式方程将被覆盖。
