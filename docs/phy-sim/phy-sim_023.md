# 质量弹簧势能

> 原文：[`phys-sim-book.github.io/lec4.3-mass_spring_energy.html`](https://phys-sim-book.github.io/lec4.3-mass_spring_energy.html)

`<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.css">`

在本案例研究中，我们专注于将质量-弹簧弹性势能纳入我们的系统。质量-弹簧弹性的概念类似于将网格的每条边都视为一个弹簧。这种方法受到胡克定律的启发，使我们能够以下列方式对边 e 上的势能进行公式化：

Pe(x)=l2/21k(l2∥x1−x2∥2−1)2,(4.3.1)

在这里，x1 和 x2 分别代表边两端当前的位置。变量 l 表示边的原始长度，k 是控制弹簧刚度的参数。值得注意的是，当两端之间的距离 ∥x1−x2∥ 等于原始长度 l 时，势能 Pe(x) 达到其全局最小值 0，表示没有施加力。

该公式的关键方面是在开头包含 l2。这类似于在整个固体上积分弹簧能量，并选择边作为求积点。这种积分有助于保持刚度行为与参数 k 之间的一致关系，无论网格分辨率的变化。

与标准弹簧能量公式相比，另一个偏差是我们避免了平方根运算。我们直接使用 ∥x1−x2∥²，使我们的模型本质上是多项式的。这种简化产生了更简洁的梯度和对角线表达式：

∂x1/∂Pe(x)=−∂x2/∂Pe(x)=2k(l2∥x1−x2∥2−1)(x1−x2),

∂x1²/∂2Pe(x)=∂x2²/∂2Pe(x)=−∂x1∂x2/∂2Pe(x)=−∂x2∂x1/∂2Pe(x)=l2/4k(x1−x2)(x1−x2)^T+2k(l2∥x1−x2∥2−1)I=l2/2k(2(x1−x2)(x1−x2)^T+(∥x1−x2∥2−l2)I)。

系统的总势能，表示为 P(x)，可以通过对所有边的势能求和得到。这使用方程 (4.3.1) 计算。因此，总势能表示为：P(x)=∑ePe(x)，其中求和是对网格中所有边进行的。

**实现 4.3.1（MassSpringEnergy.py）**。

```py
import numpy as np
import utils

def val(x, e, l2, k):
    sum = 0.0
    for i in range(0, len(e)):
        diff = x[e[i][0]] - x[e[i][1]]
        sum += l2[i] * 0.5 * k[i] * (diff.dot(diff) / l2[i] - 1) ** 2
    return sum

def grad(x, e, l2, k):
    g = np.array([[0.0, 0.0]] * len(x))
    for i in range(0, len(e)):
        diff = x[e[i][0]] - x[e[i][1]]
        g_diff = 2 * k[i] * (diff.dot(diff) / l2[i] - 1) * diff
        g[e[i][0]] += g_diff
        g[e[i][1]] -= g_diff
    return g

def hess(x, e, l2, k):
    IJV = [[0] * (len(e) * 16), [0] * (len(e) * 16), np.array([0.0] * (len(e) * 16))]
    for i in range(0, len(e)):
        diff = x[e[i][0]] - x[e[i][1]]
        H_diff = 2 * k[i] / l2[i] * (2 * np.outer(diff, diff) + (diff.dot(diff) - l2[i]) * np.identity(2))
        H_local = utils.make_PSD(np.block([[H_diff, -H_diff], [-H_diff, H_diff]]))
        # add to global matrix
        for nI in range(0, 2):
            for nJ in range(0, 2):
                indStart = i * 16 + (nI * 2 + nJ) * 4
                for r in range(0, 2):
                    for c in range(0, 2):
                        IJV[0][indStart + r * 2 + c] = e[i][nI] * 2 + r
                        IJV[1][indStart + r * 2 + c] = e[i][nJ] * 2 + c
                        IJV[2][indStart + r * 2 + c] = H_local[nI * 2 + r, nJ * 2 + c]
    return IJV 
```

在处理质量-弹簧能量的 Hessian 矩阵时，一个关键考虑因素是其非对称正定（SPD）性质。为了解决这个问题，采用了一种特定的修改：我们中和了对应于每条边的局部 Hessian 的负特征值。在将这些局部 Hessian 纳入全局矩阵之前进行此操作。这个过程涉及将负特征值设为零，从而确保生成的全局 Hessian 矩阵更接近所需的 SPD 属性。这种修改对于牛顿法至关重要。

**实现 4.3.2（正定投影）**。

```py
import numpy as np
import numpy.linalg as LA

def make_PSD(hess):
    [lam, V] = LA.eigh(hess)    # Eigen decomposition on symmetric matrix
    # set all negative Eigenvalues to 0
    for i in range(0, len(lam)):
        lam[i] = max(0, lam[i])
    return np.matmul(np.matmul(V, np.diag(lam)), np.transpose(V)) 
```
