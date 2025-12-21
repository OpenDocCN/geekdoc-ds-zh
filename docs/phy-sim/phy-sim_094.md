# 摘要

> 原文：[`phys-sim-book.github.io/lec18.5-summary.html`](https://phys-sim-book.github.io/lec18.5-summary.html)

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.css">

我们已经讨论了纽曼边界条件、狄利克雷边界条件以及在连续设置中的摩擦接触，以完成一个严格的问题表述。将所有内容以强形式结合，对于所有 t≥0：

R(X,0)∂t∂V(X,t)=∇X⋅P(X,t)+R(X,0)Aext(X,t),x=xD(X,t),P(X,t)N(X)=TN(X,t)+TC(X,t)+TF(X,t),ϕ(X,t):Ω0→Ωt 是双射,TF(X,t)=β∈RdargminβTVF(X,t)s.t.∥β∥≤μ∥TC(X,t)∥  和  β⋅N(X,t)=0,​∀ X∈Ω0;∀ X∈ΓD;∀ X∈ΓN;∀ X∈Ω0;∀ X∈ΓC.​(18.5.1)

在推导动量方程的弱形式后，边界积分项可以如下分离：

∫∂Ω0Qi(X,t)Ti(X,t)ds(X)=∫ΓDQi(X,t)TD∣i(X,t)ds(X)+∫ΓNQi(X,t)TN∣i(X,t)ds(X)+∫ΓCQi(X,t)TC∣i(X,t)ds(X)+∫ΓCQi(X,t)TF∣i(X,t)ds(X)．(18.5.2)

在这里，只给出了纽曼力 TN(X,t)，而所有其他边界力在解耦系统后都可以确定。幸运的是，狄利克雷边界条件可以在优化框架中直接作为线性等式约束来实施。摩擦接触力 TC(X,t) 和 TF(X,t) 都可以平滑地近似为具有可控误差的守恒力。

在下一章中，我们将讨论使用有限元方法（FEM）离散弱形式，将本章的推导与离散模拟方法联系起来。
