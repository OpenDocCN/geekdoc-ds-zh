# 线性三角形元素

> 原文：[`phys-sim-book.github.io/lec15.1-linear_tri_elem.html`](https://phys-sim-book.github.io/lec15.1-linear_tri_elem.html)



在之前的讨论中，我们学习了如何计算 Ψ 及其对 F 的导数。然而，对于模拟，我们需要 ∂x/∂Ψ 和 ∂x2/∂2Ψ。这需要我们清楚地理解 F(x)，因为它允许我们使用链式法则有效地推导出这些关于 x 的导数。

在二维模拟中，我们通常将固体域划分为非退化的三角形元素。假设映射 x=ϕ(X) 在每个三角形内是线性的，从而保持变形梯度 F 恒定。参考示例 12.2.1，对于一个由顶点 X1、X2、X3 定义的三角形，我们得到以下方程：x2−x1=F(X2−X1)和 x3−x1=F(X3−X1)，其中 xi 表示三角形顶点的世界空间坐标。这种关系导致 F 的表达式：F=[x2−x1,x3−x1][X2−X1,X3−X1]−1。(15.1.1) 方程 (15.1.1) 显示，这里推导出的 F，通过三角形边 X2−X1 和 X3−X1 的线性组合将三角形内的任何线段映射到其世界空间对应物。这个公式的更一般和严格的推导将在后续章节中给出。

一旦建立了 F(x)，我们就可以计算每个三角形关于 x 的导数，如下所示：∂[x1T,x2T,x3T]T/∂[F11,F21,F12,F22]T=−B11−B21 0−B12−B22 0 0−B11−B21 0−B12−B22 B11 0B12 0 0B11 0B12 B21 0B22 0 0B21 0B22 ，其中 B=[X2−X1,X3−X1]−1 表示由从第二个顶点减去第一个顶点和第三个顶点形成的矩阵的逆。这个矩阵 B 可以在初始化时与其他属性（如每个三角形的体积和 Lame 参数）一起预计算。

**实现 15.1.1（元素信息预计算，simulator.py）。**

```py
# rest shape basis, volume, and lame parameters
vol = [0.0] * len(e)
IB = [np.array([[0.0, 0.0]] * 2)] * len(e)
for i in range(0, len(e)):
    TB = [x[e[i][1]] - x[e[i][0]], x[e[i][2]] - x[e[i][0]]]
    vol[i] = np.linalg.det(np.transpose(TB)) / 2
    IB[i] = np.linalg.inv(np.transpose(TB))
mu_lame = [0.5 * E / (1 + nu)] * len(e)
lam = [E * nu / ((1 + nu) * (1 - 2 * nu))] * len(e) 
```

杨氏模量和泊松比：

```py
E = 1e5         # Young's modulus
nu = 0.4        # Poisson's ratio 
```

在这里，`e` 不再存储所有边元素，如质量-弹簧模型中那样，而是代表所有三角形元素，这些元素可以通过修改网格代码生成，如下所示：

**实现 15.1.2（按三角形顶点索引组装，square_mesh.py）。**

```py
 # connect the nodes with triangle elements
    e = []
    for i in range(0, n_seg):
        for j in range(0, n_seg):
            # triangulate each cell following a symmetric pattern:
            if (i % 2)^(j % 2):
                e.append([i * (n_seg + 1) + j, (i + 1) * (n_seg + 1) + j, i * (n_seg + 1) + j + 1])
                e.append([(i + 1) * (n_seg + 1) + j, (i + 1) * (n_seg + 1) + j + 1, i * (n_seg + 1) + j + 1])
            else:
                e.append([i * (n_seg + 1) + j, (i + 1) * (n_seg + 1) + j, (i + 1) * (n_seg + 1) + j + 1])
                e.append([i * (n_seg + 1) + j, (i + 1) * (n_seg + 1) + j + 1, i * (n_seg + 1) + j + 1]) 
```

三角形以对称模式排列，可以通过绘制三条边来渲染：

**实现 15.1.3（绘制三角形，simulator.py）。**

```py
 pygame.draw.aaline(screen, (0, 0, 255), screen_projection(x[eI[0]]), screen_projection(x[eI[1]]))
        pygame.draw.aaline(screen, (0, 0, 255), screen_projection(x[eI[1]]), screen_projection(x[eI[2]]))
        pygame.draw.aaline(screen, (0, 0, 255), screen_projection(x[eI[2]]), screen_projection(x[eI[0]])) 
```
