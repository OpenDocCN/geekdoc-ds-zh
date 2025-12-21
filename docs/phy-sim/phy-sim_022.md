# 惯性项

> 原文：[`phys-sim-book.github.io/lec4.2-inertia.html`](https://phys-sim-book.github.io/lec4.2-inertia.html)



对于惯性项，给定 $ \tilde{x}^n = x^n + h v^n $，我们有 \[ E_I(x) = \frac{1}{2}\|x - \tilde{x}^n \|_M², \quad \nabla E_I(x) = M(x - \tilde{x}^n), \quad \text{and} \quad \nabla² E_I(x) = M, \] 这一点易于实现：

**实现 4.2.1 (InertiaEnergy.py)**。

```py
import numpy as np

def val(x, x_tilde, m):
    sum = 0.0
    for i in range(0, len(x)):
        diff = x[i] - x_tilde[i]
        sum += 0.5 * m[i] * diff.dot(diff)
    return sum

def grad(x, x_tilde, m):
    g = np.array([[0.0, 0.0]] * len(x))
    for i in range(0, len(x)):
        g[i] = m[i] * (x[i] - x_tilde[i])
    return g

def hess(x, x_tilde, m):
    IJV = [[0] * (len(x) * 2), [0] * (len(x) * 2), np.array([0.0] * (len(x) * 2))]
    for i in range(0, len(x)):
        for d in range(0, 2):
            IJV[0][i * 2 + d] = i * 2 + d
            IJV[1][i * 2 + d] = i * 2 + d
            IJV[2][i * 2 + d] = m[i]
    return IJV 
```

函数`val()`、`grad()`和`hess()`被设计用来计算惯性项的不同组成部分。具体来说：

+   `val()`: 计算惯性项的值。

+   `grad()`: 计算惯性项的梯度。

+   `hess()`: 确定惯性项的 Hessian 矩阵。

关于 Hessian 矩阵，采用了一种内存高效的策略。而不是分配一个大的二维数组来存储 Hessian 矩阵的所有条目，只保留非零条目。这是通过使用由三个列表组成的`IJV`结构实现的：

1.  **行索引**：标识每个非零条目的行位置。

1.  **列索引**：指示每个非零条目的列位置。

1.  **值**：在指定行和列的实际非零值。

此方法显著降低了与下游处理相关的内存使用和计算成本。
