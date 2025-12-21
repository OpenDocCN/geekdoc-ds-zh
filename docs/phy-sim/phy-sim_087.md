# 离散时间

> [原文：https://phys-sim-book.github.io/lec17.2-discrete_time.html](https://phys-sim-book.github.io/lec17.2-discrete_time.html)

``

时间离散化将 A 与我们的自由度（DOF）x 相联系。在连续设置中，A(X,t)=∂t2/∂2x(X,t)。现在，让我们将时间划分为小的区间，t0,t2,…,tn,…，正如第一章所讨论的。使用有限差分公式，我们可以方便地用 x 来近似 A。

例如，使用向后欧拉：An(X)Vn(X)=tn−tn−1Vn(X)−Vn−1(X)，=tn−tn−1xn(X)−xn−1(X)，这给我们：An(X)=Δt2xn(X)−(xn−1(X)+ΔtVn−1(X))，其中 Δt=tn−tn−1。将这个关系应用于样本点，代入方程 (17.1.3)，我们得到：Ma^bΔt2xb|i^n−(xb|i^n−1+ΔtVb|i^n−1) = ∫∂Ω0 Na(X)Ti(X,tn)ds(X)−∫Ω0 Na^,j(X)Pi^j(X,tn)dX。(17.2.1)

然后，通过应用质量集中和零牵引边界条件，即 T(X,t)=0，我们最终看到方程 (17.2.1) 是第一讲中 离散时空 的离散形式向后欧拉时间积分的第 (a^d+i^)- 行：M(xn+1−(xn+Δtvn))−Δt2f(xn+1)=0，其中弹性力 f(x) 是通过评估：−∫Ω0 Na^,j(X)Pi^j(X,t)dX 得到的，这将在下一章中讨论。
