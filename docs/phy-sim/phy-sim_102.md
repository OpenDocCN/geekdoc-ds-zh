# 固体-障碍物接触

> 原文：[`phys-sim-book.github.io/lec20.2-obstacle_contact.html`](https://phys-sim-book.github.io/lec20.2-obstacle_contact.html)

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.css">

回想一下，我们使用保守力模型来近似接触牵引力 TC，允许它直接根据固体的当前配置进行评估。这导致了一个接触势能：

PC=∫ΓC21b(X2∈ΓC−N(X)min∥x(X,t)−x(X2,t)∥,d^)ds(X),

其中 b() 是势垒能量密度函数，N(X) 是围绕 X 的一个无穷小区域，在该区域内忽略接触以保持理论上的严谨性。

对于模拟固体和碰撞障碍物之间的正常接触（目前忽略自接触），PC 可以写成更简单的形式 PC=∫ΓS21b(X2∈ΓOmin∥x(X,t)−x(X2,t)∥,d^)ds(X)+∫ΓO21b(X2∈ΓSmin∥x(X,t)−x(X2,t)∥,d^)ds(X)=∫ΓSb(X2∈ΓOmin∥x(X,t)−x(X2,t)∥,d^)ds(X)=∫ΓSb(dPO(x(X,t),O),d^)ds(X)。这里 ΓS 和 ΓO 分别是模拟固体和障碍物的边界，dPO(x(X,t),O)=minX2∈ΓO∥x(X,t)−x(X2,t)∥ 是点-障碍物距离，从两个项简化为一个单一项是由于连续设置中的对称性。使用三角形离散化，∫ΓSb(dPO(x(X,t),O),d^)ds(X)≈e∈T∑∫∂Ωe0∩ΓSb(dPO(x(X,t),O),d^)ds(X)。(20.2.1) 类似于 Neumann 边界的推导，对于任何边界节点 a^，如果有 2 个入射三角形，让我们看看其中一个积分。不失一般性，我们可以假设 Na^=β (Xa^ 对应于三角形 e 中的 X2)，并且 X3 是 e 在边界边上的另一个节点。然后，将积分变量切换到 β，我们得到 =∫∂Ωe0∩ΓSb(dPO(x(βX2+(1−β)X3,t),O),d^)∂β∂sds(X)。(20.2.2) 由于 b() 和 dPO() 都是高度非线性的函数，我们无法得到方程 (20.2.2) 的封闭形式表达式。如果我们取两个端点 X2 和 X3 作为求积点，并且两者都有权重 21，我们可以将积分近似为 ≈∫01b(dPO(x(βX2+(1−β)X3,t),O),d^)∂β∂sds(X)21b(dPO(x(X2,t),O),d^)∂β∂sds(X)+21b(dPO(x(X3,t),O),d^)∂β∂sds(X)。(20.2.3) 然后，整个边界积分可以近似为 ∫ΓSb(dPO(x(X,t),O),d^)ds(X)≈a^∑2∥Xa^−Xa^−1∥+∥Xa^−Xa^+1∥b(dPO(xa^,O),d^)，假设 Xa^−1 和 Xa^+1 是 Xa^ 在边界上的两个邻居。这正是现在在 Filter Line Search 中实现的内容。

> ***备注 20.2.1（线段积分的求积选择）***。选择线段积分的两个端点（β=0,1）作为求积点（方程 (20.2.3)）并不是一个常见的方案。通常，高斯求积会使用 β=63±3。选择 β=0,1 的优点是它会导致全局求积点更少，从而降低计算成本，因为相邻的边共享端点。

要了解 PC 如何与离散弱形式中的边界积分（方程(20.1)）相连接，让我们对离散化接触势（方程(20.2.1)）关于 xa^的导数进行求导：==−∂xa^∂(∑e∈T∫∂Ωe0∩ΓSb(dPO(x(X,t),O),d^)ds(X))e∈T∑∫∂Ωe0∩ΓS−∂x∂b(dPO(x(X,t),O),d^)∂xa^∂xds(X)e∈T∑∫∂Ωe0∩ΓS−∂x∂b(dPO(x(X,t),O),d^)Na^​(X)ds(X)．然后我们也验证了 TC(X,t)=−∂x∂b(dPO(x(X,t),O),d^)在这里。
