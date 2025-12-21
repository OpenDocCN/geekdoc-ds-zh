# Drucker-Prager 弹塑性

> 原文：[`phys-sim-book.github.io/lec30.1-drucker_prager.html`](https://phys-sim-book.github.io/lec30.1-drucker_prager.html)



**Drucker-Prager** 塑性模型被广泛用于模拟沙子和土壤等颗粒材料。它通过引入一个**摩擦角**来推广 von Mises 模型，该摩擦角决定了材料相对于正应力的剪切应力承受能力。从物理上讲，这对应于颗粒之间的库仑摩擦：当剪切应力超过基于压力的摩擦相关的界限时，材料发生屈服。

在应力空间中，Drucker-Prager 屈服面呈**锥形**，有三个不同的案例来处理：

+   **情况 I（弹性）**：应力严格位于锥内，没有发生塑性变形。

+   **情况 II（膨胀）**：应力对应于材料体积膨胀（正迹）的配置，没有施加阻力——这映射到锥尖。

+   **情况 III（剪切）**：应力位于锥外，但有压缩压力，必须投影回锥表面。

此模型最好在**对数应变（Hencky 应变）**空间中使用变形梯度的奇异值分解来实现，这我们在上一节中已经介绍过。

> **示例 31.1.1（Drucker-Prager 屈服准则，对数应变公式）**。我们定义**对数应变**为：
> 
> ϵ=log(Σ)+d1log(Jvol)I。
> 
> 偏心部分是：
> 
> ϵ^=ϵ−d1tr(ϵ)I。
> 
> 塑性乘子计算如下：
> 
> Δγ=∥ϵ^∥+2μ(dλ+2μ)⋅tr(ϵ)⋅α。
> 
> 这里，α=32⋅3−sinϕ2sinϕ 是从摩擦角 ϕ 导出的 Drucker-Prager 摩擦系数。
> 
> 然后我们应用返回映射：
> 
> +   如果 tr(ϵ)>0（情况 II），我们将投影到锥尖：设置 ϵ=0。
> +   
> +   如果 Δγ≤0，我们位于锥内（情况 I）：无变化。
> +   
> +   否则（情况 III），我们将投影回锥表面：
> +   
> ϵn+1=ϵ−∥ϵ^∥Δγϵ^。
> 
> 最后，我们计算更新的奇异值：
> 
> ΣEn+1=exp(ϵn+1)，
> 
> 并重建弹性变形：
> 
> FEn+1=Udiag(ΣEn+1)VT。
> 
> **示例 31.1.2（带体积校正的 Drucker-Prager 塑性）**。
> 
> 在沙子等颗粒材料中，如果处理不当，体积膨胀可能导致**非物理体积增益**。标准的 Drucker-Prager 投影在膨胀（正迹）下将应力映射到锥尖，这对应于**无应力状态**。然而，这可能在新的平衡形状中“锁定”膨胀配置，这是不切实际的。
> 
> 这种效应可能导致在颗粒经历弹性膨胀后进行塑性投影时持续体积膨胀。任何未来的压缩都会产生人工的弹性惩罚，导致材料响应不正确。
> 
> 为了纠正这个问题，我们遵循 [[Tampubolon 等人 2017]](bibliography.html#tampubolon2017multi) 描述的体积修正处理方法，通过引入每个粒子的标量累加器 vvol，来跟踪由塑性投影引起的**对数体积变化**：
> 
> vvoln+1 = vvoln − logdetFEn+1 + logdetFtrial.
> 
> 这项修正自然地通过调整回映射前的应变而整合到**对数应变公式**中：
> 
> ϵcorrected = ϵtrial + dvvol I,
> 
> 其中 d 是空间维度。这允许未来的压缩**抵消之前的体积增加**，而不是被弹性抵抗。在下面的代码中，`diff_log_J` 提供了这个体积修正项，它是通过对数行列式差的累积来计算的。

**实现 31.1.1（Drucker-Prager 弹塑性回映射，simulator.py）**。

```py
@ti.func
def Drucker_Prager_return_mapping(F, diff_log_J):
    dim = ti.static(F.n)
    sin_phi = ti.sin(friction_angle_in_degrees/ 180.0 * ti.math.pi)
    friction_alpha = ti.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)
    U, sig_diag, V = ti.svd(F)
    sig = ti.Vector([ti.max(sig_diag[i,i], 0.05) for i in ti.static(range(dim))])
    epsilon = ti.log(sig)
    epsilon += diff_log_J / dim # volume correction treatment
    trace_epsilon = epsilon.sum()
    shifted_trace = trace_epsilon
    if shifted_trace >= 0:
        for d in ti.static(range(dim)):
            epsilon[d] = 0.
    else:
        epsilon_hat = epsilon - (trace_epsilon / dim)
        epsilon_hat_norm = ti.sqrt(epsilon_hat.dot(epsilon_hat)+1e-8)
        delta_gamma = epsilon_hat_norm + (dim * lam + 2\. * mu) / (2\. * mu) * (shifted_trace) * friction_alpha
        epsilon -= (ti.max(delta_gamma, 0) / epsilon_hat_norm) * epsilon_hat
    sig_out = ti.exp(epsilon)
    for d in ti.static(range(dim)):
        sig_diag[d,d] = sig_out[d]
    return U @ sig_diag @ V.transpose() 
```

回映射在粒子状态更新期间强制执行：

**实现 31.1.2（粒子状态更新，simulator.py）**。

```py
@ti.kernel
def update_particle_state():
    for p in range(N_particles):
        # trial elastic deformation gradient
        F_tr = F[p]
        # apply return mapping to correct the trial elastic state, projecting the stress induced by F_tr
        # back onto the yield surface, following the direction specified by the plastic flow rule.
        new_F = Drucker_Prager_return_mapping(F_tr, diff_log_J[p])
        # track the volume change incurred by return mapping to correct volume, following https://dl.acm.org/doi/10.1145/3072959.3073651 sec 4.3.4
        diff_log_J[p] += -ti.log(new_F.determinant()) + ti.log(F_tr.determinant()) 
        F[p] = new_F
        # advection
        x[p] += dt * v[p] 
```
