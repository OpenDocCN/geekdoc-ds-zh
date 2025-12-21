# 摩擦力

> 原文：[`phys-sim-book.github.io/lec18.4-friction_force.html`](https://phys-sim-book.github.io/lec18.4-friction_force.html)



类似于摩擦接触，在库仑约束下最大化耗散率，定义了单位面积变化的摩擦力：

TF(X,t)=β∈RdargminβTVF(X,t)s.t.∥β∥≤μ∥TC(X,t)∥  和  β⋅N(X,t)=0.​(18.4.1)

这里，VF(X,t)=V(X,t)−V(X2,t) 是 X 和最近点 X2（X2=argminX2∈ΓC−N(X)∥X−X2∥）之间的相对滑动速度，μ 是摩擦系数，TC 是单位面积的法向接触力，N 是法向方向。

这等价于：

TF(X,t)=−μ∥TC(X,t)∥ f(∥VF(X,t)∥) s(VF(X,t)),(18.4.2)

当 ∥VF∥>0 时，s(VF)=∥VF∥VF，而当 ∥VF∥=0 时，s(VF)取与 N(X,t) 正交的任意单位向量。

此外，摩擦尺度函数 f 在 VF 上也是非光滑的，因为当 ∥VF∥>0 时，f(∥VF∥)=1，而当 ∥VF∥=0 时，f(∥VF∥)∈[0,1]。这些非光滑性质可能会严重阻碍甚至破坏基于梯度的优化的收敛。这里摩擦-速度关系的光滑化方法与摩擦接触中采用的方法相同。
