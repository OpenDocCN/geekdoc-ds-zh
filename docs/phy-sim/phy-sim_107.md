# 点-边距离

> 原文：[`phys-sim-book.github.io/lec21.2-point_edge_dist.html`](https://phys-sim-book.github.io/lec21.2-point_edge_dist.html)



接下来，我们计算点-边距离及其导数。这些将被用于求解接触力。对于一个节点 p 和一个边 e0e1，它们的平方距离定义为

dsqPE(p,e0,e1)=λmin∥p−((1−λ)e0+λe1)∥2s.t.λ∈[0,1],

这是 p 和 e0e1 上任意点之间的最近平方距离。

> ***备注 21.2.1（距离计算优化）***。在这里，我们使用平方无符号距离来评估接触能。这种方法避免了取平方根，这可能会使导数的表达式复杂化，并在计算过程中增加数值舍入误差。此外，无符号距离可以直接扩展到共维数对，例如点-点对，这在模拟二维中的粒子接触时很有用。它们也不像有符号距离那样在有大位移时出现锁定问题。

幸运的是，dsqPE(p,e0,e1)相对于自由度（DOFs）是一个分段光滑函数：dsqPE(p,e0,e1)=⎩⎨⎧∥p−e0∥2if (e1−e0)⋅(p−e0)<0,∥p−e1∥2if (e1−e0)⋅(p−e0)>∥e1−e0∥2,∥e1−e0∥21(det([p−e0,e1−e0]))2otherwise,​(21.2.1) 其中，平滑表达式可以通过检查节点是否位于边的正交补中来确定。有了这些平滑表达式，我们可以对每个表达式进行微分，从而获得点-边距离函数的导数。实现如下：

**实现 21.2.1（点-边距离计算（省略 Hessian），PointEdgeDistance.py）**。

```py
import numpy as np

import distance.PointPointDistance as PP
import distance.PointLineDistance as PL

def val(p, e0, e1):
    e = e1 - e0
    ratio = np.dot(e, p - e0) / np.dot(e, e)
    if ratio < 0:    # point(p)-point(e0) expression
        return PP.val(p, e0)
    elif ratio > 1:  # point(p)-point(e1) expression
        return PP.val(p, e1)
    else:            # point(p)-line(e0e1) expression
        return PL.val(p, e0, e1)

def grad(p, e0, e1):
    e = e1 - e0
    ratio = np.dot(e, p - e0) / np.dot(e, e)
    if ratio < 0:    # point(p)-point(e0) expression
        g_PP = PP.grad(p, e0)
        return np.reshape([g_PP[0:2], g_PP[2:4], np.array([0.0, 0.0])], (1, 6))[0]
    elif ratio > 1:  # point(p)-point(e1) expression
        g_PP = PP.grad(p, e1)
        return np.reshape([g_PP[0:2], np.array([0.0, 0.0]), g_PP[2:4]], (1, 6))[0]
    else:            # point(p)-line(e0e1) expression
        return PL.grad(p, e0, e1) 
```

可以验证，点-边距离函数在所有地方都是 C1-连续的，包括不同段之间的接口。对于点-点情况，我们有：

**实现 21.2.2（点-点距离计算，PointPointDistance.py）**。

```py
import numpy as np

def val(p0, p1):
    e = p0 - p1
    return np.dot(e, e)

def grad(p0, p1):
    e = p0 - p1
    return np.reshape([2 * e, -2 * e], (1, 4))[0]

def hess(p0, p1):
    H = np.array([[0.0] * 4] * 4)
    H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 2
    H[0, 2] = H[1, 3] = H[2, 0] = H[3, 1] = -2
    return H 
```

对于点-线情况，距离评估可以按以下方式实现，导数可以使用符号微分工具获得。

**实现 21.2.3（点-线距离计算（省略 Hessian），PointLineDistance.py）**。

```py
import numpy as np

def val(p, e0, e1):
    e = e1 - e0
    numerator = e[1] * p[0] - e[0] * p[1] + e1[0] * e0[1] - e1[1] * e0[0]
    return numerator * numerator / np.dot(e, e)

def grad(p, e0, e1):
    g = np.array([0.0] * 6)
    t13 = -e1[0] + e0[0]
    t14 = -e1[1] + e0[1]
    t23 = 1.0 / (t13 * t13 + t14 * t14)
    t25 = ((e0[0] * e1[1] + -(e0[1] * e1[0])) + t14 * p[0]) + -(t13 * p[1])
    t24 = t23 * t23
    t26 = t25 * t25
    t27 = (e0[0] * 2.0 + -(e1[0] * 2.0)) * t24 * t26
    t26 *= (e0[1] * 2.0 + -(e1[1] * 2.0)) * t24
    g[0] = t14 * t23 * t25 * 2.0
    g[1] = t13 * t23 * t25 * -2.0
    t24 = t23 * t25
    g[2] = -t27 - t24 * (-e1[1] + p[1]) * 2.0
    g[3] = -t26 + t24 * (-e1[0] + p[0]) * 2.0
    g[4] = t27 + t24 * (p[1] - e0[1]) * 2.0
    g[5] = t26 - t24 * (p[0] - e0[0]) * 2.0
    return g 
```
