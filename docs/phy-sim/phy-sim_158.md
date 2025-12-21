# 基于 SDF 的球体碰撞器

> 原文：[`phys-sim-book.github.io/lec30.2-sphere_sdf.html`](https://phys-sim-book.github.io/lec30.2-sphere_sdf.html)



在有符号距离部分，我们介绍了固体几何形状的**解析表示**——其中像球体、盒子和半空间这样的形状是通过它们坐标上的数学表达式定义的。在那里引入的一个强大抽象是**有符号距离函数（SDF）**。此函数评估空间中任意一点的几何表面有符号距离：负值表示点在物体内部，正值表示外部，零表示正好在表面上。

这个概念自然地转化为在模拟框架如 MPM 中的**碰撞检测和边界条件执行**。

### 使用解析 SDF 表示碰撞器

考虑一个**二维球体**（圆形）的中心为 c=(cx, cy)和半径 r。其 SDF 定义为：

ϕ(x)=∥x−c∥−r.

+   如果ϕ(x)<0，则该点位于球体**内部**。

+   如果ϕ(x)=0，则该点位于球体的**表面**上。

+   如果ϕ(x)>0，则该点位于球体**外部**。

这个定义允许我们通过在每个网格节点上评估 SDF，在整个模拟域内统一应用**接触边界条件**。

**实现 31.2.1（具有摩擦接触的球体 SDF 碰撞器，simulator.py）。**

```py
 # a sphere SDF as boundary condition
            sphere_center = ti.Vector([0.5, 0.5])
            sphere_radius = 0.05 + dx # add a dx-gap to avoid penetration
            if (x_i - sphere_center).norm() < sphere_radius:
                normal = (x_i - sphere_center).normalized()
                diff_vel = -grid_v[i, j]
                dotnv = normal.dot(diff_vel)
                dotnv_frac = dotnv * (1.0 - sdf_friction)
                grid_v[i, j] += diff_vel * sdf_friction + normal * dotnv_frac 
```
