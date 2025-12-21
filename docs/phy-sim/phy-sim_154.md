# 粒子-单元格转移

> 原文：[`phys-sim-book.github.io/lec29.2-pic_transfer.html`](https://phys-sim-book.github.io/lec29.2-pic_transfer.html)



在每个模拟步骤的开始，在累积新的粒子到网格转移之前，必须清除网格。

**实现 30.2.1 (重置网格，simulator.py).**

```py
def reset_grid():
    # after each transfer, the grid is reset
    grid_m.fill(0)
    grid_v.fill(0) 
```

我们采用使用变形梯度的奇异值分解 (SVD) 在对数 (Hencky) 应变空间中表述的 Saint Venant-Kirchhoff (StVK) 本构模型。

**实现 30.2.2 (Stvk Hencky 弹性，simulator.py).**

```py
@ti.func
def StVK_Hencky_PK1_2D(F):
    U, sig, V = ti.svd(F)
    inv_sig = sig.inverse()
    e = ti.Matrix([[ti.log(sig[0, 0]), 0], [0, ti.log(sig[1, 1])]])
    return U @ (2 * mu * inv_sig @ e + lam * e.trace() * inv_sig) @ V.transpose() 
```

在粒子到网格 (P2G) 转移期间，我们使用二次 B 样条插值将每个粒子的质量、动量和内力分布到其相邻的网格节点。此过程遵循 PIC (粒子在单元格中) 公式，其中粒子速度直接转移到网格，而不存储仿射速度场。

**实现 30.2.3 (PIC 粒子到网格 (P2G) 转移，simulator.py).**

```py
@ti.kernel
def particle_to_grid_transfer():
    for p in range(N_particles):
        base = (x[p] / dx - 0.5).cast(int)
        fx = x[p] / dx - base.cast(float)
        # quadratic B-spline interpolating functions (Section 26.2)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        # gradient of the interpolating function (Section 26.2)
        dw_dx = [fx - 1.5, 2 * (1.0 - fx), fx - 0.5]

        P = StVK_Hencky_PK1_2D(F[p])
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                weight = w[i][0] * w[j][1]
                grad_weight = ti.Vector([(1\. / dx) * dw_dx[i][0] * w[j][1], 
                                          w[i][0] * (1\. / dx) * dw_dx[j][1]])

                grid_m[base + offset] += weight * m[p] # mass transfer
                grid_v[base + offset] += weight * m[p] * v[p] # momentum Transfer, PIC formulation
                # internal force (stress) transfer
                fi = -vol[p] * P @ F[p].transpose() @ grad_weight
                grid_v[base + offset] += dt * fi 
```

在粒子到网格 (P2G) 转移之后，我们归一化网格动量以获得节点速度，并通过将速度置零来强制执行域边缘附近的 Dirichlet 边界条件。

**实现 30.2.4 (网格更新，simulator.py).**

```py
@ti.kernel
def update_grid():
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] = grid_v[i, j] / grid_m[i, j] # extract updated nodal velocity from transferred nodal momentum

            # Dirichlet BC near the bounding box
            if i <= 3 or i > grid_size - 3 or j <= 2 or j > grid_size - 3:
                grid_v[i, j] = 0 
```

在网格到粒子 (G2P) 转移期间，我们从背景网格中收集更新的速度，并使用从插值函数导出的速度梯度来计算弹性变形梯度更新。

**实现 30.2.5 (PIC 网格到粒子 (G2P) 转移，simulator.py).**

```py
@ti.kernel
def grid_to_particle_transfer():
    for p in range(N_particles):
        base = (x[p] / dx - 0.5).cast(int)
        fx = x[p] / dx - base.cast(float)
        # quadratic B-spline interpolating functions (Section 26.2)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        # gradient of the interpolating function (Section 26.2)
        dw_dx = [fx - 1.5, 2 * (1.0 - fx), fx - 0.5]

        new_v = ti.Vector.zero(float, 2)
        v_grad_wT = ti.Matrix.zero(float, 2, 2)
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                weight = w[i][0] * w[j][1]
                grad_weight = ti.Vector([(1\. / dx) * dw_dx[i][0] * w[j][1], 
                                          w[i][0] * (1\. / dx) * dw_dx[j][1]])

                new_v += weight * grid_v[base + offset]
                v_grad_wT += grid_v[base + offset].outer_product(grad_weight)

        v[p] = new_v
        F[p] = (ti.Matrix.identity(float, 2) + dt * v_grad_wT) @ F[p] 
```

最后，通过使用辛欧拉时间积分进行对流，更新粒子位置。

**实现 30.2.6 (粒子状态更新，simulator.py).**

```py
@ti.kernel
def update_particle_state():
    for p in range(N_particles):
        x[p] += dt * v[p] # advection 
```

一个完整的 MPM 模拟步骤包括以下阶段：

**实现 30.2.7 (MPM 的一次完整时间步，simulator.py).**

```py
def step():
    # a single time step of the Material Point Method (MPM) simulation
    reset_grid()
    particle_to_grid_transfer()
    update_grid()
    grid_to_particle_transfer()
    update_particle_state() 
```

![](img/1bcb904b717ed3313178f48f22cea71e.png)

**图 30.2.1.** **2D 中两个碰撞的弹性块的时间序列**。红色和蓝色块以相反的速度相互接近，在碰撞时发生大的弹性变形，并反弹以恢复形状。该模拟在 MPM 框架下展示了对称动量交换和弹性能量存储。
