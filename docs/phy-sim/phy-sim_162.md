# 初步了解

> 原文：[`phys-sim-book.github.io/lec31.1-pbd_preliminaries.html`](https://phys-sim-book.github.io/lec31.1-pbd_preliminaries.html)

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.css">

要充分理解 PBD 方法，首先理解其灵感来源的经典拉格朗日动力学公式是至关重要的。由 N 个粒子组成的动态系统的状态由它们的个体质量 mi，位置 pi∈R3 和速度 vi∈R3 描述。该系统的演化由从牛顿第二定律导出的两个一阶常微分方程控制：v˙i=mi1​fi（32.1.1）p˙i=vi（32.1.2）其中 fi 是粒子 i 上所有力的总和。

Rigid bodies require additional attributes to describe their rotational state: an orientation quaternion qi∈H, an angular velocity ωi∈R3, and an inertia tensor Ii∈R3×3. The rotational motion is then described by the Newton-Euler equations: ω˙i=Ii−1(τi−(ωi×(Iiωi)))（32.1.3）q˙i=21ω^iqi（32.1.4）where τi is the net torque and ω^i is the quaternion [0,ωi].

To simulate this evolution, these continuous equations are discretized using a numerical integrator. The Symplectic Euler method updates the velocity first, then uses this new velocity to update the position, improving stability over standard explicit Euler. For a particle, with time step Δt, the update is: vi(t0+Δt)=vi(t0)+Δtmi1fi(t0)(32.1.5) pi(t0+Δt)=xi(t0)+Δtvi(t0+Δt)(32.1.6) This procedure is applied analogously for rigid body states.

> ***备注 32.1.1（四元数归一化）***。由于数值积分误差，四元数 qi 可能从单位长度漂移。在每次积分步骤之后重新归一化四元数对于保持有效的旋转状态是至关重要的。

Finally, interactions and physical limits are modeled using **holonomic constraints**, which depend only on positions and orientations, but not velocities! Constraints are kinematic restrictions in the form of equations and inequalities that constrain the relative motion of bodies. An **equality (bilateral)** constraint takes the form Cj(p,q)=0, while an **inequality (unilateral)** constraint is Cj(p,q)≥0. In classical dynamics, these are satisfied by computing constraint forces and adding them to fi in Equation (32.1.1). It is this specific mechanism for handling constraints that PBD fundamentally changes.
