# 解的准确性

> [原文链接](https://phys-sim-book.github.io/lec7.3-sol_accuracy.html)

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.css">

那么，我们为什么可以解方程 (7.2.2) 来近似求解方程 (7.2.1) 的解呢？与 Dirichlet 边界条件 类似，在方程 (7.2.1) 的解 \(x^*\) 处，以下 KKT 条件都成立：∇E(x∗)−i,j∑​γij​∇dij​(x∗)=0,∀i,j, dij​(x∗)≥0, γij​≥0, γij​dij​(x∗)=0.​(7.3.1) 而在方程 (7.2.2) 的局部最优 \(x'\) 处，我们有 ∇E(x′)+h2∇Pb​(x′)=0,(7.3.2) 这等价于 ∇E(x′)+h2i,j∑​Ai​d^∇b(dij​(x′))=0 和 ∇E(x′)+h2i,j∑​Ai​d^∂d∂b​(dij​(x′))∇dij​(x′)=0，如果我们代入 \(\nabla b(d_{ij})\) 的表达式。令 \(\gamma_{ij}' = -h² A_i \hat{d} \frac{\partial b}{\partial d}(d_{ij}(x'))\)，我们可以进一步将方程 (7.3.2) 重写为 ∇E(x′)−i,j∑​γij′​∇dij​(x′)=0，如果我们把 \(\gamma_{ij}'\) 作为对偶变量。现在，由于障碍函数提供了任意大的排斥力以避免相互穿透，我们知道对于所有 \(i,j\)，\(d_{ij}(x') \geq 0\)。此外，对于所有 \(i,j\)，\(\gamma_{ij}' \geq 0\) 也成立，因为 \(\frac{\partial b}{\partial d} \leq 0\) 是构造的。这意味着在 \(x'\) 处，我们具有动量平衡，没有相互穿透，接触力只推不拉。

在我们的模拟中，在 \( x' \) 处唯一没有严格满足的 Karush-Kuhn-Tucker (KKT) 条件是互补松弛条件。这源于我们的障碍近似函数的方式。具体来说，我们有一个情况，其中 \( \gamma_{ij} > 0 \Longleftrightarrow 0 < d_{ij} < \hat{d} \)，表示基于固体和障碍物之间的距离激活接触力。

随着阈值 \( \hat{d} \) 的降低，接触力仅在固体更接近时才会激活（如图 7.2.1 所示）。这种调整导致互补松弛误差减少，这在一定程度上是可以控制的。然而，需要注意的是，这种控制是有代价的：计算效率可能会降低。这是因为较小的 \( \hat{d} \) 值导致更尖锐的目标函数，这通常需要更多的牛顿迭代来求解。因此，在模拟的准确性（即遵守 KKT 条件）和所需的计算资源之间存在着权衡。
