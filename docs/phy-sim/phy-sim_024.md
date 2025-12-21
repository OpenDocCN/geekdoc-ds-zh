# 优化时间积分器

> 原文：[`phys-sim-book.github.io/lec4.4-opt_time_integrator.html`](https://phys-sim-book.github.io/lec4.4-opt_time_integrator.html)



在建立了评估任意配置的增量势的能力之后，我们现在将注意力转向优化时间积分器的实现。这个积分器对于最小化增量势至关重要，进而更新节点位置和速度。此实现遵循 算法 3.3.1 中概述的方法：

**实现 4.4.1 (time_integrator.py).**

```py
import copy
from cmath import inf

import numpy as np
import numpy.linalg as LA
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

import InertiaEnergy
import MassSpringEnergy

def step_forward(x, e, v, m, l2, k, h, tol):
    x_tilde = x + v * h     # implicit Euler predictive position
    x_n = copy.deepcopy(x)

    # Newton loop
    iter = 0
    E_last = IP_val(x, e, x_tilde, m, l2, k, h)
    p = search_dir(x, e, x_tilde, m, l2, k, h)
    while LA.norm(p, inf) / h > tol:
        print('Iteration', iter, ':')
        print('residual =', LA.norm(p, inf) / h)

        # line search
        alpha = 1
        while IP_val(x + alpha * p, e, x_tilde, m, l2, k, h) > E_last:
            alpha /= 2
        print('step size =', alpha)

        x += alpha * p
        E_last = IP_val(x, e, x_tilde, m, l2, k, h)
        p = search_dir(x, e, x_tilde, m, l2, k, h)
        iter += 1

    v = (x - x_n) / h   # implicit Euler velocity update
    return [x, v]

def IP_val(x, e, x_tilde, m, l2, k, h):
    return InertiaEnergy.val(x, x_tilde, m) + h * h * MassSpringEnergy.val(x, e, l2, k)     # implicit Euler

def IP_grad(x, e, x_tilde, m, l2, k, h):
    return InertiaEnergy.grad(x, x_tilde, m) + h * h * MassSpringEnergy.grad(x, e, l2, k)   # implicit Euler

def IP_hess(x, e, x_tilde, m, l2, k, h):
    IJV_In = InertiaEnergy.hess(x, x_tilde, m)
    IJV_MS = MassSpringEnergy.hess(x, e, l2, k)
    IJV_MS[2] *= h * h    # implicit Euler
    IJV = np.append(IJV_In, IJV_MS, axis=1)
    H = sparse.coo_matrix((IJV[2], (IJV[0], IJV[1])), shape=(len(x) * 2, len(x) * 2)).tocsr()
    return H

def search_dir(x, e, x_tilde, m, l2, k, h):
    projected_hess = IP_hess(x, e, x_tilde, m, l2, k, h)
    reshaped_grad = IP_grad(x, e, x_tilde, m, l2, k, h).reshape(len(x) * 2, 1)
    return spsolve(projected_hess, -reshaped_grad).reshape(len(x), 2) 
```

在这里 `step_forward()` 实质上是投影牛顿法（算法 3.3.1）的直接翻译，并且我们将增量势值（`IP_val()`）、梯度（`IP_grad()`）和海森矩阵（`IP_hess()`）评估作为单独的函数实现，以提高清晰度。

在计算搜索方向时，我们利用了来自 [Scipy 库](https://scipy.org/) 的线性求解器，该求解器擅长处理稀疏矩阵。值得注意的是，此求解器接受压缩稀疏行（CSR）格式的矩阵。选择这种格式和求解器是由它们在处理效率和内存使用方面的效率驱动的，这在处理计算模拟中经常遇到的大规模稀疏矩阵问题时尤其有利。
