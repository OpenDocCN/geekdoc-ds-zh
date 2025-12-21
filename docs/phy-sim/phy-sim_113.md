# 预计算法向和切向信息

> 原文：[`phys-sim-book.github.io/lec22.2-precompute.html`](https://phys-sim-book.github.io/lec22.2-precompute.html)



为了使时间离散化的摩擦力可积，我们必须显式地离散化某些法向和切向信息。这些信息只需要在每个时间步长的开始计算一次，因为它们将在每次优化过程中保持不变。

首先，我们需要使用 xn 计算每个点-边缘对的 λn。回想一下，我们使用了平方距离作为势函数的输入，因此 λn 应该使用链式法则如下计算：

λa^n,en=21Aa^(-∂dPE∂b(dsqPE(xa^n,e),d²))=21Aa^(-∂dsqPE∂b(dsqPE(xa^n,e),d²)∂dPE∂dsqPE))=21Aa^(-∂dsqPE∂b(dsqPE(xa^n,e),d²))2dPE.​

根据以平方距离作为输入的缩放势函数（方程 (21.3.1))，我们可以推导出

∂dsq∂b(dsq,d²)={8κd^(d²¹lnd²dsq+dsq1(d²dsq−1))0if d<d^;if d≥d^.​

> ***备注 22.2.1.*** 在我们的半隐式摩擦设置中，点-边缘对的集合在每个时间步长中是固定的，并且与法向接触对的集合不同。摩擦的集合只包含那些 dPE(xa^n,e)<d^ 的对，并且这不会随着当前时间步长中的优化变量 xn+1 而改变。

对于切向信息，关键在于保持边缘上最近点的法向和重心坐标不变。对于第 k 个点-边缘对，如果我们用 p、e0 和 e1 表示点和边缘的节点索引，那么我们可以写出相对滑动速度为

vk=(I−nnT)(vp−((1−r)ve0+rve1)),

其中 r=argminc∥xp−((1−c)xe0+cxe1)∥ 是重心坐标，n=(xp−((1−r)xe0+rxe1))/∥xp−((1−r)xe0+rxe1)∥ 是边缘的法向。在这里我们可以看到，r 和 n 都依赖于 x，所以直接积分 vk 是非平凡的。通过使用 xn 计算 n 和 r，我们得到半隐式相对滑动速度

vˉk=(I−nn(nn)T)(vp−((1−rn)ve0+rnve1)),

现在只有速度依赖于 xn+1，这使得积分和微分都变得简单。如果我们表示 v^k=vp−((1−rn)ve0+rnve1)，我们得到

∂v^k∂vˉk=(I−nn(nn)T)和∂[xpT,xe0T,xe1T]T∂v^k=h¹[I(I(rn−1)I−rnI)].

### 代码

接下来，让我们看看代码。实现 22.2.1 计算了给定点-边缘节点位置的最接近点的重心坐标和法向。思路是将 xp 垂直投影到边缘上。

**实现 22.2.1（计算接触点和法向，PointEdgeDistance.py）。**

```py
# compute normal and the parameterization of the closest point on the edge
def tangent(p, e0, e1):
    e = e1 - e0
    ratio = np.dot(e, p - e0) / np.dot(e, e)
    if ratio < 0:    # point(p)-point(e0) expression
        n = p - e0
    elif ratio > 1:  # point(p)-point(e1) expression
        n = p - e1
    else:            # point(p)-line(e0e1) expression
        n = p - ((1 - ratio) * e0 + ratio * e1)
    return [n / np.linalg.norm(n), ratio] 
```

然后，实现 22.2.2 遍历所有距离小于 d^ 的非碰撞点-边缘对，计算 λ，并调用上述函数计算 n 和 r。

如同在摩擦接触中所述，这些代码行在`time_integrator.py`的每个时间步开始时执行，并且每个摩擦对的信息被存储并传递给我们将要讨论的能量、梯度和 Hessian 计算函数。

**实现 22.2.2（半隐式摩擦预计算，BarrierEnergy.py）**。

```py
 # self-contact
    mu_lambda_self = []
    dhat_sqr = dhat * dhat
    for xI in bp:
        for eI in be:
            if xI != eI[0] and xI != eI[1]: # do not consider a point and its incident edge
                d_sqr = PE.val(x[xI], x[eI[0]], x[eI[1]])
                if d_sqr < dhat_sqr:
                    s = d_sqr / dhat_sqr
                    # since d_sqr is used, need to divide by 8 not 2 here for consistency to linear elasticity
                    # also, lambda = -\partial b / \partial d = -(\partial b / \partial d²) * (\partial d² / \partial d)
                    mu_lam = mu * -0.5 * contact_area[xI] * dhat * (kappa / 8 * (math.log(s) / dhat_sqr + (s - 1) / d_sqr)) * 2 * math.sqrt(d_sqr)
                    [n, r] = PE.tangent(x[xI], x[eI[0]], x[eI[1]]) # normal and closest point parameterization on the edge
                    mu_lambda_self.append([xI, eI[0], eI[1], mu_lam, n, r]) 
```
