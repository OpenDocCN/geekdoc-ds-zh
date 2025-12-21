# 势垒能量及其导数

> 原文：[`phys-sim-book.github.io/lec21.3-barrier_and_derivatives.html`](https://phys-sim-book.github.io/lec21.3-barrier_and_derivatives.html)

`<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.css">`

在实现了点-边距离函数之后，我们可以遍历所有点-边对来组装总的势垒能量及其导数。这些将被用于在时间步优化中求解搜索方向。

由于使用了平方距离，因此在这里我们将势垒函数重新缩放为 b(d²,d²)={8κd^(d²/d²−1)ln(d²/d²)0 d<d² d≥d²}，(21.3.1) 以确保 ∂(d²/d²)²/∂²b(d²,d²)=κd² 仍然成立。类似于弹性，s=d/d² 可以被视为应变度量，那么在 s=1 时能量密度（每单位面积）函数 b 对 s 的二阶导数将对应于杨氏模量乘以厚度 d²，这使得 κ 在物理上具有意义且便于设置。

基于方程 (20.3.1)，我们可以推导出势垒势能的梯度和 Hessian 为 ∇Pb(x)∇²Pb(x) = 2a^∑(Aa^e∈E−I(Xa^))∑(∂d/∂b(d(xa^,e),d²))∇x(d(xa^,e)) 和 = 2a^∑(Aa^e∈E−I(Xa^))∑[(∂²d²/∂²b(d(xa^,e),d²))∇x(d(xa^,e))∇x(d(xa^,e))T + (∂d/∂b(d(xa^,e),d²))∇x²(d(xa^,e))]，其中 Aa^ = 2∥Xa^−Xa^−1∥ + ∥Xa^−Xa^+1∥，并且我们省略了平方点-边距离函数（d(xa^,e)表示 dsqPE(xa^,e)）的上标和下标。

势垒接触势能的能量、梯度和 Hessian 的实现如下：

**实现 21.3.1（势垒能量计算，BarrierEnergy.py）**。

```py
 # self-contact
    dhat_sqr = dhat * dhat
    for xI in bp:
        for eI in be:
            if xI != eI[0] and xI != eI[1]: # do not consider a point and its incident edge
                d_sqr = PE.val(x[xI], x[eI[0]], x[eI[1]])
                if d_sqr < dhat_sqr:
                    s = d_sqr / dhat_sqr
                    # since d_sqr is used, need to divide by 8 not 2 here for consistency to linear elasticity:
                    sum += 0.5 * contact_area[xI] * dhat * kappa / 8 * (s - 1) * math.log(s) 
```

**实现 21.3.2（势垒能量梯度计算，BarrierEnergy.py）**。

```py
 # self-contact
    dhat_sqr = dhat * dhat
    for xI in bp:
        for eI in be:
            if xI != eI[0] and xI != eI[1]: # do not consider a point and its incident edge
                d_sqr = PE.val(x[xI], x[eI[0]], x[eI[1]])
                if d_sqr < dhat_sqr:
                    s = d_sqr / dhat_sqr
                    # since d_sqr is used, need to divide by 8 not 2 here for consistency to linear elasticity:
                    local_grad = 0.5 * contact_area[xI] * dhat * (kappa / 8 * (math.log(s) / dhat_sqr + (s - 1) / d_sqr)) * PE.grad(x[xI], x[eI[0]], x[eI[1]])
                    g[xI] += local_grad[0:2]
                    g[eI[0]] += local_grad[2:4]
                    g[eI[1]] += local_grad[4:6] 
```

**实现 21.3.3（势垒能量 Hessian 计算，BarrierEnergy.py）**。

```py
 # self-contact
    dhat_sqr = dhat * dhat
    for xI in bp:
        for eI in be:
            if xI != eI[0] and xI != eI[1]: # do not consider a point and its incident edge
                d_sqr = PE.val(x[xI], x[eI[0]], x[eI[1]])
                if d_sqr < dhat_sqr:
                    d_sqr_grad = PE.grad(x[xI], x[eI[0]], x[eI[1]])
                    s = d_sqr / dhat_sqr
                    # since d_sqr is used, need to divide by 8 not 2 here for consistency to linear elasticity:
                    local_hess = 0.5 * contact_area[xI] * dhat * utils.make_PSD(kappa / (8 * d_sqr * d_sqr * dhat_sqr) * (d_sqr + dhat_sqr) * np.outer(d_sqr_grad, d_sqr_grad) \
                        + (kappa / 8 * (math.log(s) / dhat_sqr + (s - 1) / d_sqr)) * PE.hess(x[xI], x[eI[0]], x[eI[1]]))
                    index = [xI, eI[0], eI[1]]
                    for nI in range(0, 3):
                        for nJ in range(0, 3):
                            for c in range(0, 2):
                                for r in range(0, 2):
                                    IJV[0].append(index[nI] * 2 + r)
                                    IJV[1].append(index[nJ] * 2 + c)
                                    IJV[2] = np.append(IJV[2], local_hess[nI * 2 + r, nJ * 2 + c]) 
```
