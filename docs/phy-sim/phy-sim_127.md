# 刚体动力学

> 原文：[`phys-sim-book.github.io/lec25.1-rigid_body_dynamics.html`](https://phys-sim-book.github.io/lec25.1-rigid_body_dynamics.html)



在深入研究 ABD 之前，让我们回顾一下模拟刚体动力学传统方法。

让我们从回顾牛顿力学开始——特别是，对于位置 x(t)处的粒子 m 的牛顿第二定律：

dtdxmdtdv=v=f.

现在，计算 dtd(mx×v)：

dtd(mx×v)=mdtdx×v+mx×dtdv=mv×v+x×f=0+x×f=x×f.

在这里，mx×v 被称为**角动量**，而 x×f 被称为**扭矩**。

对于刚体 B，我们可以对其体积进行积分：

dtd(∫Bρx×vdx)=∫Bρx×f(x)dx，其中ρ=ρ(x)是 B 在位置 x 的质量密度。

现在，让 c 表示 B 的质量中心（COM）。那么：

dtd(∫Bρc×vdx)=∫Bρc×dtdvdx=∫Bρc×f(x)dx.

结合两者：

dtd(∫Bρ(x−c)×vdx)=∫Bρ(x−c)×f(x)dx.

由于∫Bρ(x−c)×vcdx=(∫Bρ(x−c)dx)×vc=0×vc=0，我们可以进一步简化：

dtd(∫Bρ(x−c)×(v−vc)dx)=∫Bρ(x−c)×f(x)dx，其中右侧是关于质心的扭矩τ。

将右侧表示为扭矩τc，计算左侧：

dtd(∫Bρ(x−c)×(v−vC)dx)=dtd(∫Bρ(x−c)×(ω×(x−c))dx)=dtd(∫Bρ(⟨x−c,x−c⟩ω−⟨x−c,ω⟩(x−c))dx)=IBcω.

这里，**惯性张量**IBc 定义为：

IBc:=∫Bρ(∥x−c∥22I−(x−c)(x−c)T)dx.

由于 x 依赖于 t，所以 IBc 也依赖于 t。设 R(t)为旋转矩阵（B 从时间 0 到 t 的旋转）。很容易验证：

IBc(t)=R(t)IBc(0)R(t)^T.

求导得到：

τc=ω×(Icω)+Icdtdω.

因此，刚体动力学定律变为：

dtd(mvc)ω×(Icω)+Icdtdω=f=τc.(25.1.1)

这些方程类似于牛顿第二定律。第一个描述了力如何影响刚体的线性动量，而第二个描述了扭矩如何影响其角动量。

使用简单的显式积分器，我们可以通过以下方式模拟刚体运动：

vn+1xn+1ωn+1q^n+1q^n=vn+Δtmfn,=xn+Δtvn,=ωn+Δt(Ic)^−1(τc−ω×(Icω)),=qn+2Δtωqn,=∣q^n+1∣q^n+1.(25.1.2)

这里，q 是表示身体旋转的单位四元数。其更新基于一阶近似：

q−q=e2Δtωq−q=(cos2∥ω∥Δt−1+∥ω∥ωsin2∥ω∥Δt)q=(Θ(Δt²)+2Δtω)q，这导致：Δt→0limΔtq−q=21wq，

其中 w=(ωx,ωy,ωz,0)；q 是旋转向量Δtω和 q 表示的旋转的合成。

> ***备注 25.1.1（旋转矢量到四元数）***. 角度为θ，绕由单位向量 u=(ux,uy,uz)=uxi+uyj+uzk 定义的轴旋转可以表示为四元数 q=e2θ(uxi+uyj+uzk)=cos2θ+(uxi+uyj+uzk)sin2θ=cos2θ+usin2θ
