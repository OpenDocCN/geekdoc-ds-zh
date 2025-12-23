# 2.6\. 练习题

> 原文：[`mmids-textbook.github.io/chap02_ls/exercises/roch-mmids-ls-exercises.html`](https://mmids-textbook.github.io/chap02_ls/exercises/roch-mmids-ls-exercises.html)

## 2.6.1\. 预习工作表#

*(有 Claude、Gemini 和 ChatGPT 的帮助)*

**第 2.2 节**

**E2.2.1** 判断集合 \(U = \{(x, y, z) \in \mathbb{R}³ : x + 2y - z = 0\}\) 是否是 \(\mathbb{R}³\) 的线性子空间。

**E2.2.2** 判断向量 \(\mathbf{u}_1 = (1, 1, 1)\)，\(\mathbf{u}_2 = (1, 0, -1)\)，和 \(\mathbf{u}_3 = (2, 1, 0)\) 是否线性无关。

**E2.2.3** 找到子空间 \(U = \{(x, y, z) \in \mathbb{R}³ : x - y + z = 0\}\) 的基。

**E2.2.4** 找到子空间 \(U = \{(x, y, z, w) \in \mathbb{R}⁴ : x + y = 0, z = w\}\) 的维度。

**E2.2.5** 验证向量 \(\mathbf{u}_1 = (1/\sqrt{2}, 1/\sqrt{2})\) 和 \(\mathbf{u}_2 = (1/\sqrt{2}, -1/\sqrt{2})\) 是否构成一个正交归一列表。

**E2.2.6** 给定正交基 \(\mathbf{q}_1 = (1/\sqrt{2}, 1/\sqrt{2})\)，\(\mathbf{q}_2 = (1/\sqrt{2}, -1/\sqrt{2})\)，求向量 \(\mathbf{w} = (1, 0)\) 的正交展开。

**E2.2.7** 判断矩阵 \(A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}\) 是否非奇异。

**E2.2.8** 解方程组 \(A\mathbf{x} = \mathbf{b}\)，其中 \(A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}\) 且 \(\mathbf{b} = (5, 11)\)。

**E2.2.9** 给定向量 \(\mathbf{w}_1 = (1, 0, 1)\) 和 \(\mathbf{w}_2 = (0, 1, 1)\)，将向量 \(\mathbf{v} = (2, 3, 5)\) 表示为 \(\mathbf{w}_1\) 和 \(\mathbf{w}_2\) 的线性组合。

**E2.2.10** 验证向量 \(\mathbf{u} = (1, 2, 3)\) 和 \(\mathbf{v} = (4, -2, 1)\) 是否正交。

**E2.2.11** 找到矩阵 \(B = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}\) 的零空间。

**E2.2.12** 给定 \(\mathbb{R}³\) 的基 \(\{\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3\}\)，将向量 \(\mathbf{v} = (4, 5, 6)\) 表示为基向量的线性组合。

**E2.2.13** 判断向量 \(\mathbf{u}_1 = (1, 2, 3)\)，\(\mathbf{u}_2 = (2, -1, 0)\)，和 \(\mathbf{u}_3 = (1, 8, 6)\) 是否线性无关。

**E2.2.14** 验证向量 \(\mathbf{u} = (2, 1)\) 和 \(\mathbf{v} = (1, -3)\) 的柯西-施瓦茨不等式。

**E2.2.15** 使用矩阵求逆法解方程组 \(2x + y = 3\) 和 \(x - y = 1\)。

**第 2.3 节**

**E2.3.1** 设 \(Q = \begin{pmatrix} \frac{1}{\sqrt{2}} & \frac{-1}{\sqrt{6}} \\ 0 & \frac{2}{\sqrt{6}} \\ \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{6}} \end{pmatrix}\)。\(Q\) 是一个正交矩阵吗？

**E2.3.2** 判断：向量在子空间上的正交投影总是严格短于原始向量。

**E2.3.3** 对于向量 \(\mathbf{v} = (2, 3)\) 和由 \(\mathbf{u} = (1, 1)\) 生成的线性子空间 \(U\)，找到 \(\mathrm{proj}_{U} \mathbf{v}\) 的正交投影。

**E2.3.4** 设 \(U = \mathrm{span}((1, 1, 0))\) 和 \(\mathbf{v} = (1, 2, 3)\)。计算 \(\|\mathbf{v}\|²\) 和 \(\|\mathrm{proj}_U \mathbf{v}\|²\)。

**E2.3.5** 找到 \(\mathbf{v} = (1, 2, 1)\) 在由 \(\mathbf{u} = (1, 1, 0)\) 张成的子空间上的正交投影。

**E2.3.6** 设 \(U = \mathrm{span}((1, 0, 1), (0, 1, 1))\)。找到 \(U^\perp\) 的一个基。

**E2.3.7** 设 \(\mathbf{v} = (1, 2, 3)\) 和 \(U = \mathrm{span}((1, 0, 1), (0, 1, 1))\)。计算 \(\mathrm{proj}_U \mathbf{v}\)。

**E2.3.8** 设 \(\mathbf{v} = (1, 2, 3)\) 和 \(U = \mathrm{span}((1, 0, 1), (0, 1, 1))\)。将 \(\mathbf{v}\) 分解为其在 \(U\) 上的正交投影和一个在 \(U^\perp\) 上的向量。

**E2.3.9** 设 \(\mathbf{v} = (1, 2, 3)\) 和 \(U = \mathrm{span}((1, 0, 1), (0, 1, 1))\)。验证毕达哥拉斯定理：\(\|\mathbf{v}\|² = \|\mathrm{proj}_U \mathbf{v}\|² + \|\mathbf{v} - \mathrm{proj}_U \mathbf{v}\|²\)。

**E2.3.10** 设 \(A = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}\)。\(P = AA^T\) 是否是一个正交投影矩阵？

**E2.3.11** 设 \(\mathbf{u}_1 = \begin{pmatrix} 2 \\ 1 \\ -2 \end{pmatrix}\) 和 \(\mathbf{u}_2 = \begin{pmatrix} 1 \\ 1 \\ 1 \end{pmatrix}\)。计算 \(\mathrm{proj}_{\mathbf{u}_1} \mathbf{u}_2\)。

**E2.3.12** 找到在 \(\mathbb{R}³\) 中由 \(\mathbf{u} = (1, 1, 0)\) 张成的子空间的正交补。

**E2.3.13** 设 \(W = \mathrm{span}\left((1, 1, 0)\right)\)。找到 \(W^\perp\) 的一个基。

**E2.3.14** 设 \(A = \begin{pmatrix} 1 & 2 \\ 0 & 1 \\ 1 & 0 \end{pmatrix}\) 和 \(\mathbf{b} = \begin{pmatrix} 3 \\ 1 \\ 2 \end{pmatrix}\)。建立正则方程以求解线性最小二乘问题 \(A\mathbf{x} \approx \mathbf{b}\)。

**E2.3.15** 求解正则方程 \(A^T A \mathbf{x} = A^T \mathbf{b}\) 对于 \(A = \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}\) 和 \(\mathbf{b} = \begin{pmatrix} 3 \\ 1 \end{pmatrix}\)。

**E2.3.16** 设 \(\mathbf{a} = (1, 2, 1)\) 和 \(\mathbf{b} = (1, 0, -1)\)。找到 \(\mathbf{a}\) 在 \(\mathrm{span}(\mathbf{b})\) 上的正交投影。

**第 2.4 节**

**E2.4.1** 设 \(\mathbf{a}_1 = (1, 0)\) 和 \(\mathbf{a}_2 = (1, 1)\)。应用 Gram-Schmidt 算法找到 \(\mathrm{span}(\mathbf{a}_1, \mathbf{a}_2)\) 的一个正交基。

**E2.4.2** 使用 Gram-Schmidt 算法确定矩阵 \(A = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}\) 的 QR 分解。

**E2.4.3** 将基 \(\{(1, 1), (1, 0)\}\) 通过 Gram-Schmidt 算法转换为一个正交基。

**E2.4.4** 给定向量 \(a_1 = (1, 1, 1)\)，\(a_2 = (1, 0, -1)\)，应用 Gram-Schmidt 算法得到 \(\mathrm{span}(a_1, a_2)\) 的一个正交基。

**E2.4.5** 对于来自 E2.4.4 的向量 \(a_1\) 和 \(a_2\)，找到矩阵 \(A = [a_1\ a_2]\) 的 QR 分解。

**E2.4.6** 使用回代法求解方程组 \(R\mathbf{x} = \begin{pmatrix} 1 & 1 \\ 0 & 2 \end{pmatrix} \mathbf{x} = \begin{pmatrix} 1 \\ 2 \end{pmatrix}\)。

**E2.4.7** 设 \(R = \begin{pmatrix} 2 & -1 \\ 0 & 3 \end{pmatrix}\) 和 \(\mathbf{b} = (4, 3)\)。使用回代法求解方程组 \(R\mathbf{x} = \mathbf{b}\)。

**E2.4.8** 给定矩阵 \(A = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}\) 和向量 \(\mathbf{b} = (1, 2)\)，使用 QR 分解求解最小二乘问题 \(\min_{x \in \mathbb{R}²} \|A\mathbf{x} - \mathbf{b}\|\)。

**E2.4.9** 设 \(\mathbf{z} = (1, -1, 0)\)。找到一个豪斯霍尔德反射矩阵 \(H\)，该矩阵在 \(\mathbf{z}\) 正交的超平面上进行反射。

**E2.4.10** 找到一个豪斯霍尔德反射矩阵，该矩阵在矩阵 \(A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}\) 的第一列下方引入零。

**E2.4.11** 验证 E2.4.10 中的豪斯霍尔德反射矩阵 \(H_1\) 是正交和对称的。

**E2.4.12** 应用 E2.4.10 中的豪斯霍尔德反射矩阵 \(H_1\) 到 E2.4.10 中的矩阵 \(A\)，并验证它引入了第一列下方的零。

**E2.4.13** 使用 E2.4.10 中的豪斯霍尔德反射矩阵 \(H_1\)，找到 E2.4.10 中的矩阵 \(A\) 的 QR 分解。

**E2.4.14** 验证 E2.4.13 中的 QR 分解满足 \(A = QR\) 且 \(Q\) 是正交的。

**E2.4.15** 设 \(A = \begin{pmatrix} 3 & 1 \\ 4 & 2 \end{pmatrix}\)。找到一个豪斯霍尔德反射矩阵 \(H_1\)，使得 \(H_1 A\) 的第一列的第二项为零。

**E2.4.16** 设 \(A = \begin{pmatrix} 1 & 1 \\ 1 & -1 \\ 0 & 1 \end{pmatrix}\) 和 \(\mathbf{b} = (2, 0, 1)\)。为与 \(A\) 和 \(\mathbf{b}\) 相关的线性最小二乘问题设置正则方程 \(A^TA\mathbf{x} = A^T\mathbf{b}\)。

**E2.4.17** 使用 QR 分解求解线性最小二乘问题 \(A\mathbf{x} = \mathbf{b}\)，其中 \(A = \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}\) 和 \(\mathbf{b} = \begin{pmatrix} 2 \\ 0 \end{pmatrix}\)。

**第 2.5 节**

**E2.5.1** 给定数据点 \((x_1, y_1) = (1, 2)\)，\((x_2, y_2) = (2, 4)\)，\((x_3, y_3) = (3, 5)\)，和 \((x_4, y_4) = (4, 7)\)，找到使最小二乘准则 \(\sum_{i=1}⁴ (y_i - \beta_0 - \beta_1 x_i)²\) 最小的系数 \(\beta_0\) 和 \(\beta_1\)。

**E2.5.2** 对于 E2.5.1 中的数据点，计算线性回归模型的拟合值和残差。

**E2.5.3** 给定数据点 \((x_1, y_1) = (1, 3)\)，\((x_2, y_2) = (2, 5)\)，和 \((x_3, y_3) = (3, 8)\)，构建用于找到最小二乘线的 \(A\) 和 \(\mathbf{y}\)。

**E2.5.4** 对于 E2.5.3 中的数据，计算正则方程 \(A^T A \boldsymbol{\beta} = A^T \mathbf{y}\)。

**E2.5.5** 解 E2.5.4 中的正则方程，找到最小二乘线的系数 \(\beta_0\) 和 \(\beta_1\)。

**E2.5.6** 使用 E2.5.3 中的数据和 E2.5.5 中的拟合线，计算每个数据点的残差。

**E2.5.7** 给定数据点 \((x_1, y_1) = (-1, 2)\)，\((x_2, y_2) = (0, 1)\)，和 \((x_3, y_3) = (1, 3)\)，构建拟合二次多项式（次数为 2）的矩阵 \(A\)。

**E2.5.8** 假设我们有一个包含 \(n = 100\) 个观察值的样本，并且通过有放回重采样进行自助法。一个特定观察值被包含在给定自助样本中的概率是多少？（参见在线补充材料。）

**E2.5.9** 给定矩阵 \(A = \begin{pmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{pmatrix}\) 和向量 \(\mathbf{y} = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}\)，求解最小二乘问题中的系数 \(\boldsymbol{\beta}\)。

**E2.5.10** 对于多项式回归问题 \(y = \beta_0 + \beta_1 x + \beta_2 x²\)，给定点 \((0, 1)\)，\((1, 2)\)，\((2, 5)\)，形成矩阵 \(A\)。

**E2.5.11** 对于线性拟合 \(y = 2x + 1\) 在数据点 \((1, 3)\)，\((2, 5)\)，\((3, 7)\) 上的残差平方和（RSS）是多少？

**E2.5.12** 对于多项式回归 \(y = \beta_0 + \beta_1 x + \beta_2 x²\) 在点 \((1, 1)\)，\((2, 4)\)，\((3, 9)\) 上，找到正则方程。

## 2.6.2\. 问题#

**2.1** 证明 \(\mathbf{0}\) 是任何（非空）线性子空间的一个元素。\(\lhd\)

**2.2** 证明 \(\mathrm{null}(B)\) 是线性子空间。\(\lhd\)

**2.3** 设对所有 \(j \in [m]\)，\(\beta_j \neq 0\)。证明 \(\mathrm{span}(\beta_1 \mathbf{w}_1, \ldots, \beta_m \mathbf{w}_m) = \mathrm{span}(\mathbf{w}_1, \ldots, \mathbf{w}_m)\)。\(\lhd\)

**2.4** （改编自[Sol]）假设 \(U_1\) 和 \(U_2\) 是向量空间 \(V\) 的线性子空间。证明 \(U_1 \cap U_2\) 是 \(V\) 的线性子空间。\(U_1 \cup U_2\) 是否总是 \(V\) 的线性子空间？\(\lhd\)

**2.5** 设 \(\mathcal{U}, \mathcal{V}\) 是 \(V\) 的线性子空间，使得 \(\mathcal{U} \subseteq \mathcal{V}\)。证明 \(\mathrm{dim}(\mathcal{U}) \leq \mathrm{dim}(\mathcal{V})\)。[*提示：补全基。*] \(\lhd\)

**2.6** （改编自[Axl]）证明如果 \(\{\mathbf{v}_1,\ldots,\mathbf{v}_n\}\) 张成 \(U\)，那么列表

$$ \{\mathbf{v}_1-\mathbf{v}_2, \mathbf{v}_2-\mathbf{v}_3,\ldots,\mathbf{v}_{n-1}-\mathbf{v}_n,\mathbf{v}_n\}, $$

通过从每个向量（除了最后一个）中减去以下向量得到。\(\lhd\)

**2.7** （改编自[Axl]）证明如果 \(\{\mathbf{v}_1,\ldots,\mathbf{v}_n\}\) 是线性无关的，那么列表

$$ \{\mathbf{v}_1-\mathbf{v}_2, \mathbf{v}_2-\mathbf{v}_3,\ldots,\mathbf{v}_{n-1}-\mathbf{v}_n,\mathbf{v}_n\}, $$

通过从每个向量（除了最后一个）中减去以下向量得到。\(\lhd\)

**2.7** (改编自[Axl]) 假设 \(\{\mathbf{v}_1,\ldots,\mathbf{v}_n\}\) 在 \(U\) 中线性无关，并且 \(\mathbf{w} \in U\)。证明如果 \(\{\mathbf{v}_1 + \mathbf{w},\ldots, \mathbf{v}_n + \mathbf{w}\}\) 线性相关，那么 \(\mathbf{w} \in \mathrm{span}(\mathbf{v}_1,\ldots,\mathbf{v}_n)\)。 \(\lhd\)

**2.9** 证明 \(U^\perp\) 是一个线性子空间。 \(\lhd\)

**2.10** 设 \(U \subseteq V\) 是一个线性子空间，并且设 \(\mathbf{v} \in U\)。证明 \(\mathrm{proj}_U \mathbf{v} = \mathbf{v}\)。 \(\lhd\)

**2.11** 设 \(A \in \mathbb{R}^{n \times n}\) 是一个方阵。证明，如果对于任何 \(\mathbf{b} \in \mathbb{R}^n\) 存在一个唯一的 \(\mathbf{x} \in \mathbb{R}^n\) 使得 \(A \mathbf{x} = \mathbf{b}\)，那么 \(A\) 是非奇异的。 [提示：考虑 \(\mathbf{b} = \mathbf{0}\)。] \(\lhd\)

**2.13** 证明，如果 \(B \in \mathbb{R}^{n \times m}\) 和 \(C \in \mathbb{R}^{m \times p}\)，那么 \((BC)^T = C^T B^T\)。 [提示：检查各项是否匹配。] \(\lhd\)

**2.12** 设 \(A \in \mathbb{R}^{n\times m}\) 是一个 \(n\times m\) 矩阵，其列线性无关。证明 \(m\leq n\)。\(\lhd\)

**2.15** 判断向量 \(\mathbf{v} = (0,0,1)\) 是否在由该基张成的子空间中。

$$\begin{split} \mathbf{q}_1 = \frac{1}{\sqrt{2}} \begin{pmatrix} 1\\ 0\\ 1 \end{pmatrix}, \quad \mathbf{q}_2 = \frac{1}{\sqrt{3}} \begin{pmatrix} -1\\ 1\\ 1 \end{pmatrix}. \end{split}$$

\(\lhd\)

**2.14** 给定一个向量 \(\mathbf{v} = (1, 2, 3)\) 和由 \(\mathbf{u}_1 = (1, 0, 0)\) 和 \(\mathbf{u}_2 = (0, 1, 0)\) 张成的平面，找到 \(v\) 在该平面上的正交投影。

**2.18** 给定正交基 \(\{(1/\sqrt{2}, 1/\sqrt{2}), (-1/\sqrt{2}, 1/\sqrt{2})\}\)，找到向量 \(\mathbf{v} = (3, 3)\) 在由该基张成的子空间上的投影。

**2.16** 找到向量 \(\mathbf{v} = (4, 3, 0)\) 在 \(\mathbb{R}³\) 中由 \(\mathbf{u} = (1, 1, 1)\) 张成的直线上的正交投影。 \(\lhd\)

**2.18** 设 \(Q, W \in \mathbb{R}^{n \times n}\) 是可逆的。证明 \((Q W)^{-1} = W^{-1} Q^{-1}\) 和 \((Q^T)^{-1} = (Q^{-1})^T\). \(\lhd\)

**2.17** 证明 *Reducing a Spanning List Lemma*。 \(\lhd\)

**2.19** 证明对于任何 \(\mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3 \in \mathbb{R}^n\) 和 \(\beta \in \mathbb{R}\)：

a) \(\langle \mathbf{x}_1, \mathbf{x}_2 \rangle = \langle \mathbf{x}_2, \mathbf{x}_1 \rangle\)

b) \(\langle \beta \,\mathbf{x}_1 + \mathbf{x}_2, \mathbf{x}_3 \rangle = \beta \,\langle \mathbf{x}_1,\mathbf{x}_3\rangle + \langle \mathbf{x}_2,\mathbf{x}_3\rangle\)

c) \(\|\mathbf{x}_1\|² = \langle \mathbf{x}_1, \mathbf{x}_1 \rangle\)

\(\lhd\)

**2.22** 使用为 Householder 反射引入的符号，定义

$$ \tilde{\mathbf{z}}_1 = \frac{\|\mathbf{y}_1\| \,\mathbf{e}_1^{(n)} + \mathbf{y}_1}{\| \|\mathbf{y}_1\| \,\mathbf{e}_1^{(n)} + \mathbf{y}_1\|} \quad \text{and} \quad \tilde{H}_1 = I_{n\times n} - 2\tilde{\mathbf{z}}_1 \tilde{\mathbf{z}}_1^T. $$

a) 证明 \(\tilde{H}_1 \mathbf{y}_1 = - \|\mathbf{y}_1\| \,\mathbf{e}_1^{(n)}\)。

b) 在 \(\mathbf{y}_1 = \|\mathbf{y}_1\| \,\mathbf{e}_1^{(n)}\) 的情况下计算矩阵 \(\tilde{H}_1\)。 \(\lhd\)

**2.22** 证明两个正交矩阵 \(Q_1\) 和 \(Q_2\) 的乘积也是正交的。 \(\lhd\)

**2.23** 建立等式 \((U^\perp)^\perp = U\)。 \(\lhd\)

**2.24** 计算 \(U^\perp \cap U\) 并证明你的答案。 \(\lhd\)

**2.25** （改编自 [Sol]）如果 \(\mathbf{x}, \mathbf{y} \in \mathbb{R}^m\) 且 \(\|\mathbf{x}\| = \|\mathbf{y}\|\)，编写一个算法来找到一个正交矩阵 \(Q\) 使得 \(Q\mathbf{x} = \mathbf{y}\)。 \(\lhd\)

**2.26** （改编自 [Sol]）假设 \(A \in \mathbb{R}^{m \times n}\) 的秩为 \(m\)，且 \(m \leq n\)。设

$$ A^T = Q R $$

be the QR decomposition of \(A^T\) obtained by the Gram-Schmidt algorithm. Provide a solution \(\mathbf{x}\) to the underdetermined system \(A\mathbf{x} = \mathbf{b}\) in terms of \(Q\) and \(R\). [*提示:* 首先尝试平方情况。然后通过添加 \(0\) 来猜测并检查一般情况的解。] \(\lhd\)

**2.27** （改编自 [Sol]）设 \(A \in \mathbb{R}^{m \times m}\) 具有满列秩，并且 \(L \in \mathbb{R}^{m \times m}\) 是一个具有正对角元素的 lower triangular 矩阵，使得 \(A^T A = L L^T\)（这被称为 Cholesky 分解）。

a) 通过证明其列线性无关来证明 \(L^T\) 是可逆的。

b) 定义 \(Q = A(L^T)^{-1}\)。证明 \(Q\) 的列构成一个正交归一列表。

c) 使用 (b) 中的矩阵 \(Q\) 给出 \(A\) 的 QR 分解。确保显示 \(R\) 具有所需的结构。

\(\lhd\)

**2.28** （改编自 [Sol]）假设 \(A\in\mathbb{R}^{m\times n}\) 具有满列秩，且 \(A = QR\) 是通过 Gram-Schmidt 算法获得的 QR 分解。证明 \(P_0 = I_{m\times m} - QQ^T\) 是 \(A^T\) 的零空间的投影矩阵。[*提示:* 检查 *正交投影定理* 中的几何特征。] \(\lhd\)

**2.29** 假设我们考虑 \(\mathbf{a} \in \mathbb{R}^n\) 作为 \(n \times 1\) 矩阵。明确写出其 QR 分解。 \(\lhd\)

**2.30** （改编自 [Sol]）证明一个矩阵 \(A \in \mathbb{R}^{m \times n}\) 如果其列线性无关，则可以分解为 \(A = QL\)，其中 \(L\) 是下三角矩阵。[*提示:* 修改我们的程序以获得 QR 分解。] \(\lhd\)

**2.31** （改编自 [Sol]）假设 \(A \in \mathbb{R}^{n \times p}\)，\(B \in \mathbb{R}^{m \times p}\)，\(\mathbf{a} \in \mathbb{R}^n\)，和 \(\mathbf{b} \in \mathbb{R}^m\)。找到任何 \(\mathbf{x}\) 满足 \(\|A\mathbf{x} - \mathbf{a}\|² + \|B\mathbf{x} - \mathbf{b}\|²\) 最小的线性方程组。[*提示:* 将问题重写为线性最小二乘问题。] \(\lhd\)

**2.32** 设 \(P\) 为一个投影矩阵。证明：

a) \(P² = P\)

b) \(P^T = P\)

c) 检查上述两个命题对于 \(\mathbb{R}³\) 中 \(\mathbf{u} = (1, 0, 1)\) 张成的投影。

\(\lhd\)

**2.33** 证明对于任何 \(\mathbf{x}_1, \ldots, \mathbf{x}_m, \mathbf{y}_1, \ldots, \mathbf{y}_\ell, \in \mathbb{R}^n\),

$$ \left\langle \sum_{i=1}^m \mathbf{x}_i, \sum_{j=1}^\ell \mathbf{y}_j \right\rangle = \sum_{i=1}^m \sum_{j=1}^\ell \langle \mathbf{x}_i,\mathbf{y}_j\rangle. $$

\(\lhd\)

**2.34** 设 \(A \in \mathbb{R}^{n\times m}\) 为 \(n\times m\) 矩阵，其中 \(n \geq m\)，设 \(\mathbf{b} \in \mathbb{R}^n\) 为一个向量。设

$$ \mathbf{p}^* = \arg\min_{\mathbf{p} \in U} \|\mathbf{p} - \mathbf{b}\|, $$

and \(\mathbf{x}^*\) be such that \(\mathbf{p}^* = A \mathbf{x}^*\). Construct an \(A\) and a \(\mathbf{b}\) such that \(\mathbf{x}^*\) is *not* unique. \(\lhd\)

**2.35** 设 \(H_k \in \mathbb{R}^{n \times n}\) 为形式

$$\begin{split} H_k = \begin{pmatrix} I_{(k-1)\times (k-1)} & \mathbf{0} \\ \mathbf{0} & F_k \end{pmatrix} \end{split}$$

where

$$ F_k = I_{(n-k+1) \times (n-k+1)} - 2 \mathbf{z}_k \mathbf{z}_k^T, $$

for some unit vector \(\mathbf{z}_k \in \mathbb{R}^{n - k + 1}\). Show that \(H_k\) is an orthogonal matrix. \(\lhd\)

**2.36** 使用豪斯霍尔德反射的符号，设 \(A\) 的列线性无关。

a) 假设 \(\mathbf{z}_1 \neq \mathbf{0}\)，证明 \(\mathbf{y}_2 \neq \mathbf{0}\)。

b) 假设 \(\mathbf{z}_1, \mathbf{z}_2 \neq \mathbf{0}\)，证明 \(\mathbf{y}_3 \neq \mathbf{0}\)。 \(\lhd\)

**2.37** 设 \(R \in \mathbb{R}^{n \times m}\)，其中 \(n \geq m\)，为上三角矩阵，对角线上的元素非零。证明 \(R\) 的列线性无关。 \(\lhd\)

**2.38** 考虑输入数据 \(\{(\mathbf{x}_i, y_i)\}_{i=1}^n\) 上的线性回归问题，其中 \(\mathbf{x}_i \in \mathbb{R}^d\) 且 \(y_i \in \mathbb{R}\) 对所有 \(i\) 成立。

a) 如果存在 \(d+1\) 个形式为

$$\begin{split} \begin{pmatrix} 1 \\ \mathbf{x}_i \end{pmatrix} \end{split}$$

that form an independent list.

b) 在 \(d=1\) 的情况下简化上述条件，即当 \(x_i\) 为实值时。

\(\lhd\)

**2.39** 设 \(U\) 为 \(\mathbb{R}^n\) 的线性子空间，并假设 \(\mathbf{q}_1, \ldots, \mathbf{q}_k\) 是 \(U\) 的正交基。设 \(\mathbf{v} \in \mathbb{R}^n\)。

a) 证明 \(\mathbf{v}\) 在子空间 \(U^\perp\)（即 \(U\) 的正交补）上的正交投影是 \(\mathbf{v} - \mathrm{proj}_U \mathbf{v}\)。[*提示:* 使用投影的几何特征。]

b) 设 \(Q\) 为列向量为 \(\mathbf{q}_1, \ldots, \mathbf{q}_k\) 的矩阵。用 \(Q\) 表示子空间 \(U^\perp\) 上的投影矩阵。[*提示:* 使用 a)。]

\(\lhd\)

**2.40** 证明以下命题，该命题被称为 *子空间交集引理*。设 \(\mathcal{S}_1\) 和 \(\mathcal{S}_2\) 是 \(\mathbb{R}^d\) 的线性子空间，并设

$$ \mathcal{S}_1 + \mathcal{S}_2 = \{\mathbf{x}_1 + \mathbf{x}_2 \,:\, \forall \mathbf{x}_1 \in \mathcal{S}_1, \mathbf{x}_2 \in \mathcal{S}_2\}. $$

然后，它成立

$$ \mathrm{dim}(\mathcal{S}_1 + \mathcal{S}_2) = \mathrm{dim}(\mathcal{S}_1) + \mathrm{dim}(\mathcal{S}_2) - \mathrm{dim}(\mathcal{S}_1 \cap \mathcal{S}_2). $$

[*提示:* 考虑 \(\mathcal{S}_1 \cap \mathcal{S}_2\) 的一个基并将其扩展为 \(\mathcal{S}_1\) 和 \(\mathcal{S}_2\) 的基。证明得到的向量列表是线性无关的。] \(\lhd\)

**2.41** 设 \(\mathcal{U}, \mathcal{V} \subseteq \mathbb{R}^d\) 是子空间，且 \(\mathrm{dim}(\mathcal{U}) + \mathrm{dim}(\mathcal{V}) > d\)。使用问题 2.40 证明在交集 \(\mathcal{U} \cap \mathcal{V}\) 中存在一个非零向量。 \(\lhd\)

**2.42** 证明对于 \(\mathcal{V} = \mathbb{R}^d\) 的任意线性子空间 \(\mathcal{S}_1, \ldots, \mathcal{S}_m\)，它成立

$$ \mathrm{dim}\left(\bigcap_{k=1}^m \mathcal{S}_k\right) \geq \sum_{k=1}^m \mathrm{dim}\left(\mathcal{S}_k\right) - (m-1) \,\mathrm{dim}(\mathcal{V}). $$

[*提示:* 使用问题 2.40 中的 *子空间交集引理* 和归纳法。] \(\lhd\)

**2.43** 设 \(\mathcal{W}\) 是 \(\mathbb{R}^d\) 的一个线性子空间，并且设 \(\mathbf{w}_1,\ldots,\mathbf{w}_k\) 是 \(\mathcal{W}\) 的一个正交基。证明存在一个包含 \(\mathbf{w}_i\) 的 \(\mathbb{R}^d\) 的正交基。 \(\lhd\)

**2.44** 设 \(\mathcal{U}, \mathcal{V}\) 是 \(V\) 的线性子空间，且 \(\mathcal{U} \subseteq \mathcal{V}\)。证明如果 \(\mathrm{dim}(\mathcal{U}) = \mathrm{dim}(\mathcal{V})\)，则 \(\mathcal{U} = \mathcal{V}\)。[*提示:* 完成基。] \(\lhd\)

**2.45** 设 \(\mathcal{U}, \mathcal{V}\) 是 \(V\) 的线性子空间，且 \(\mathcal{U} \subseteq \mathcal{V}\)。证明如果 \(\mathrm{dim}(\mathcal{U}) < \mathrm{dim}(\mathcal{V})\)，则存在一个 \(\mathbf{u} \in \mathcal{V}\) 使得 \(\mathbf{u} \notin \mathcal{U}\)。[*提示:* 完成基。] \(\lhd\)

**2.46** 设 \(\mathcal{Z} \subseteq \mathcal{W}\) 是线性子空间，且 \(\mathrm{dim}(\mathcal{Z}) < \mathrm{dim}(\mathcal{W})\)。证明存在一个单位向量 \(\mathbf{w} \in \mathcal{W}\) 与 \(\mathcal{Z}\) 正交。 \(\lhd\)

**2.47** 设 \(\mathcal{W} = \mathrm{span}(\mathbf{w}_1,\ldots,\mathbf{w}_\ell)\) 且 \(\mathbf{z} \in \mathcal{W}\) 的范数为单位。证明存在一个包含 \(\mathbf{z}\) 的 \(\mathcal{W}\) 的正交基。 \(\lhd\)

**2.48** 设 \(\{\mathbf{u}_1,\ldots,\mathbf{u}_m\}\) 是一个线性无关的列表。证明对于任意的非零 \(\beta_1,\ldots,\beta_m \in \mathbb{R}\)，列表 \(\{\beta_1\mathbf{u}_1,\ldots,\beta_m\mathbf{u}_m\}\) 也是线性无关的。 \(\lhd\)

**2.49** 设 \(\mathcal{Z}\) 是 \(\mathbb{R}^n\) 的一个线性子空间，并且设 \(\mathbf{v} \in \mathbb{R}^n\)。证明 \(\|\mathrm{proj}_{\mathcal{Z}}\mathbf{v}\|_2 \leq \|\mathbf{v}\|_2\)。 \(\lhd\)

## 2.6.1\. 预习工作表#

*(在克劳德、双子星和 ChatGPT 的帮助下)*

**第 2.2 节**

**E2.2.1** 判断集合 \(U = \{(x, y, z) \in \mathbb{R}³ : x + 2y - z = 0\}\) 是否是 \(\mathbb{R}³\) 的一个线性子空间。

**E2.2.2** 判断向量 \(\mathbf{u}_1 = (1, 1, 1)\)，\(\mathbf{u}_2 = (1, 0, -1)\)，和 \(\mathbf{u}_3 = (2, 1, 0)\) 是否线性无关。

**E2.2.3** 找到子空间 \(U = \{(x, y, z) \in \mathbb{R}³ : x - y + z = 0\}\) 的一个基。

**E2.2.4** 找到子空间 \(U = \{(x, y, z, w) \in \mathbb{R}⁴ : x + y = 0, z = w\}\) 的维度。

**E2.2.5** 验证向量 \(\mathbf{u}_1 = (1/\sqrt{2}, 1/\sqrt{2})\)，\(\mathbf{u}_2 = (1/\sqrt{2}, -1/\sqrt{2})\) 是否构成一个正交列表。

**E2.2.6** 给定正交基 \(\mathbf{q}_1 = (1/\sqrt{2}, 1/\sqrt{2})\)，\(\mathbf{q}_2 = (1/\sqrt{2}, -1/\sqrt{2})\)，求向量 \(\mathbf{w} = (1, 0)\) 的正交展开。

**E2.2.7** 判断矩阵 \(A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}\) 是否非奇异。

**E2.2.8** 解方程组 \(A\mathbf{x} = \mathbf{b}\)，其中 \(A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}\) 和 \(\mathbf{b} = (5, 11)\)。

**E2.2.9** 给定向量 \(\mathbf{w}_1 = (1, 0, 1)\) 和 \(\mathbf{w}_2 = (0, 1, 1)\)，将向量 \(\mathbf{v} = (2, 3, 5)\) 表示为 \(\mathbf{w}_1\) 和 \(\mathbf{w}_2\) 的线性组合。

**E2.2.10** 验证向量 \(\mathbf{u} = (1, 2, 3)\) 和 \(\mathbf{v} = (4, -2, 1)\) 是否正交。

**E2.2.11** 找到矩阵 \(B = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}\) 的零空间。

**E2.2.12** 给定 \(\mathbb{R}³\) 的基 \(\{\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3\}\)，将向量 \(\mathbf{v} = (4, 5, 6)\) 表示为基向量的线性组合。

**E2.2.13** 判断向量 \(\mathbf{u}_1 = (1, 2, 3)\)，\(\mathbf{u}_2 = (2, -1, 0)\)，和 \(\mathbf{u}_3 = (1, 8, 6)\) 是否线性无关。

**E2.2.14** 验证向量 \(\mathbf{u} = (2, 1)\) 和 \(\mathbf{v} = (1, -3)\) 的柯西-施瓦茨不等式。

**E2.2.15** 使用矩阵求逆法解方程组 \(2x + y = 3\) 和 \(x - y = 1\)。

**第 2.3 节**

**E2.3.1** 设 \(Q = \begin{pmatrix} \frac{1}{\sqrt{2}} & \frac{-1}{\sqrt{6}} \\ 0 & \frac{2}{\sqrt{6}} \\ \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{6}} \end{pmatrix}\)。\(Q\) 是一个正交矩阵吗？

**E2.3.2** 对或错：向量在子空间上的正交投影总是严格短于原始向量。

**E2.3.3** 对于向量 \(\mathbf{v} = (2, 3)\) 和由 \(\mathbf{u} = (1, 1)\) 生成的线性子空间 \(U\)，找到正交投影 \(\mathrm{proj}_{U} \mathbf{v}\)。

**E2.3.4** 设 \(U = \mathrm{span}((1, 1, 0))\) 和 \(\mathbf{v} = (1, 2, 3)\)。计算 \(\|\mathbf{v}\|²\) 和 \(\|\mathrm{proj}_U \mathbf{v}\|²\)。

**E2.3.5** 找到 \(\mathbf{v} = (1, 2, 1)\) 在由 \(\mathbf{u} = (1, 1, 0)\) 张成的子空间上的正交投影。

**E2.3.6** 设 \(U = \mathrm{span}((1, 0, 1), (0, 1, 1))\)。找到 \(U^\perp\) 的一个基。

**E2.3.7** 设 \(\mathbf{v} = (1, 2, 3)\) 和 \(U = \mathrm{span}((1, 0, 1), (0, 1, 1))\)。计算 \(\mathrm{proj}_U \mathbf{v}\)。

**E2.3.8** 设 \(\mathbf{v} = (1, 2, 3)\) 和 \(U = \mathrm{span}((1, 0, 1), (0, 1, 1))\)。将 \(\mathbf{v}\) 分解为其在 \(U\) 上的正交投影和一个在 \(U^\perp\) 上的向量。

**E2.3.9** 设 \(\mathbf{v} = (1, 2, 3)\) 和 \(U = \mathrm{span}((1, 0, 1), (0, 1, 1))\)。验证毕达哥拉斯定理：\(\|\mathbf{v}\|² = \|\mathrm{proj}_U \mathbf{v}\|² + \|\mathbf{v} - \mathrm{proj}_U \mathbf{v}\|²\)。

**E2.3.10** 设 \(A = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}\)。\(P = AA^T\) 是否是一个正交投影矩阵？

**E2.3.11** 设 \(\mathbf{u}_1 = \begin{pmatrix} 2 \\ 1 \\ -2 \end{pmatrix}\) 和 \(\mathbf{u}_2 = \begin{pmatrix} 1 \\ 1 \\ 1 \end{pmatrix}\)。计算 \(\mathrm{proj}_{\mathbf{u}_1} \mathbf{u}_2\)。

**E2.3.12** 找到 \(\mathbb{R}³\) 中由 \(\mathbf{u} = (1, 1, 0)\) 张成的子空间的正交补。

**E2.3.13** 设 \(W = \mathrm{span}\left((1, 1, 0)\right)\)。找到 \(W^\perp\) 的一个基。

**E2.3.14** 设 \(A = \begin{pmatrix} 1 & 2 \\ 0 & 1 \\ 1 & 0 \end{pmatrix}\) 和 \(\mathbf{b} = \begin{pmatrix} 3 \\ 1 \\ 2 \end{pmatrix}\)。设置正则方程以求解线性最小二乘问题 \(A\mathbf{x} \approx \mathbf{b}\)。

**E2.3.15** 求解正则方程 \(A^T A \mathbf{x} = A^T \mathbf{b}\) 对于 \(A = \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}\) 和 \(\mathbf{b} = \begin{pmatrix} 3 \\ 1 \end{pmatrix}\)。

**E2.3.16** 设 \(\mathbf{a} = (1, 2, 1)\) 和 \(\mathbf{b} = (1, 0, -1)\)。找到 \(\mathbf{a}\) 在 \(\mathrm{span}(\mathbf{b})\) 上的正交投影。

**第 2.4 节**

**E2.4.1** 设 \(\mathbf{a}_1 = (1, 0)\) 和 \(\mathbf{a}_2 = (1, 1)\)。应用 Gram-Schmidt 算法找到 \(\mathrm{span}(\mathbf{a}_1, \mathbf{a}_2)\) 的一个正交基。

**E2.4.2** 使用 Gram-Schmidt 算法确定矩阵 \(A = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}\) 的 QR 分解。

**E2.4.3** 应用 Gram-Schmidt 算法将基 \(\{(1, 1), (1, 0)\}\) 转换为一个正交基。

**E2.4.4** 给定向量 \(a_1 = (1, 1, 1)\)，\(a_2 = (1, 0, -1)\)，应用 Gram-Schmidt 算法得到 \(\mathrm{span}(a_1, a_2)\) 的一个正交基。

**E2.4.5** 对于 E2.4.4 中的向量 \(a_1\) 和 \(a_2\)，找到矩阵 \(A = [a_1\ a_2]\) 的 QR 分解。

**E2.4.6** 使用回代法求解方程组 \(R\mathbf{x} = \begin{pmatrix} 1 & 1 \\ 0 & 2 \end{pmatrix} \mathbf{x} = \begin{pmatrix} 1 \\ 2 \end{pmatrix}\)。

**E2.4.7** 设 \(R = \begin{pmatrix} 2 & -1 \\ 0 & 3 \end{pmatrix}\) 和 \(\mathbf{b} = (4, 3)\)。使用回代法求解方程组 \(R\mathbf{x} = \mathbf{b}\)。

**E2.4.8** 给定矩阵 \(A = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}\) 和向量 \(\mathbf{b} = (1, 2)\)，使用 QR 分解求解最小二乘问题 \(\min_{x \in \mathbb{R}²} \|A\mathbf{x} - \mathbf{b}\|\)。

**E2.4.9** 令 \(\mathbf{z} = (1, -1, 0)\)。找到通过 \(\mathbf{z}\) 正交的超平面的 Householder 反射矩阵 \(H\)。

**E2.4.10** 找到 Householder 反射矩阵，使其在矩阵 \(A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}\) 的第一列下方引入零。

**E2.4.11** 验证 E2.4.10 中的 Householder 反射矩阵 \(H_1\) 是正交的和对称的。

**E2.4.12** 将 E2.4.10 中的 Householder 反射矩阵 \(H_1\) 应用到 E2.4.10 中的矩阵 \(A\) 上，并验证它引入了第一列下方的零。

**E2.4.13** 使用 E2.4.10 中的 Householder 反射矩阵 \(H_1\)，找到 E2.4.10 中的矩阵 \(A\) 的 QR 分解。

**E2.4.14** 验证 E2.4.13 中的 QR 分解满足 \(A = QR\) 且 \(Q\) 是正交的。

**E2.4.15** 令 \(A = \begin{pmatrix} 3 & 1 \\ 4 & 2 \end{pmatrix}\)。找到一个 Householder 反射矩阵 \(H_1\)，使得 \(H_1 A\) 的第一列的第二项为零。

**E2.4.16** 令 \(A = \begin{pmatrix} 1 & 1 \\ 1 & -1 \\ 0 & 1 \end{pmatrix}\) 和 \(\mathbf{b} = (2, 0, 1)\)。为与 \(A\) 和 \(\mathbf{b}\) 相关的线性最小二乘问题设置正则方程 \(A^TA\mathbf{x} = A^T\mathbf{b}\)。

**E2.4.17** 使用 QR 分解求解线性最小二乘问题 \(A\mathbf{x} = \mathbf{b}\)，其中 \(A = \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}\) 和 \(\mathbf{b} = \begin{pmatrix} 2 \\ 0 \end{pmatrix}\)。

**第 2.5 节**

**E2.5.1** 给定数据点 \((x_1, y_1) = (1, 2)\)，\((x_2, y_2) = (2, 4)\)，\((x_3, y_3) = (3, 5)\)，和 \((x_4, y_4) = (4, 7)\)，找到使最小二乘准则 \(\sum_{i=1}⁴ (y_i - \beta_0 - \beta_1 x_i)²\) 最小的系数 \(\beta_0\) 和 \(\beta_1\)。

**E2.5.2** 对于 E2.5.1 中的数据点，计算线性回归模型的拟合值和残差。

**E2.5.3** 给定数据点 \((x_1, y_1) = (1, 3)\)，\((x_2, y_2) = (2, 5)\)，和 \((x_3, y_3) = (3, 8)\)，构建用于找到最小二乘线的 \(A\) 和 \(\mathbf{y}\)。

**E2.5.4** 对于 E2.5.3 中的数据，计算正则方程 \(A^T A \boldsymbol{\beta} = A^T \mathbf{y}\)。

**E2.5.5** 解 E2.5.4 中的正则方程，找到最小二乘线的系数 \(\beta_0\) 和 \(\beta_1\)。

**E2.5.6** 使用 E2.5.3 中的数据和 E2.5.5 中拟合的直线，计算每个数据点的残差。

**E2.5.7** 给定数据点 \((x_1, y_1) = (-1, 2)\)，\((x_2, y_2) = (0, 1)\)，和 \((x_3, y_3) = (1, 3)\)，构建拟合二次多项式（次数为 2）的矩阵 \(A\)。

**E2.5.8** 假设我们有一个 \(n = 100\) 的观测样本，并且通过有放回重采样进行自助法。一个特定观测值被包含在给定自助样本中的概率是多少？（参见在线补充材料。）

**E2.5.9** 给定矩阵 \(A = \begin{pmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{pmatrix}\) 和向量 \(\mathbf{y} = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}\)，求解最小二乘问题中的系数 \(\boldsymbol{\beta}\)。

**E2.5.10** 对于多项式回归问题 \(y = \beta_0 + \beta_1 x + \beta_2 x²\)，在点 \((0, 1)\)，\((1, 2)\)，\((2, 5)\) 上形成矩阵 \(A\)。

**E2.5.11** 对于线性拟合 \(y = 2x + 1\) 在数据点 \((1, 3)\)，\((2, 5)\)，\((3, 7)\) 上的残差平方和（RSS）是多少？

**E2.5.12** 对于在点 \((1, 1)\)，\((2, 4)\)，\((3, 9)\) 上的多项式回归 \(y = \beta_0 + \beta_1 x + \beta_2 x²\)，找到正则方程。

## 2.6.2\. 问题#

**2.1** 证明 \(\mathbf{0}\) 是任何（非空）线性子空间的一个元素。 \(\lhd\)

**2.2** 证明 \(\mathrm{null}(B)\) 是一个线性子空间。 \(\lhd\)

**2.3** 对于所有 \(j \in [m]\)，令 \(\beta_j \neq 0\)。证明 \(\mathrm{span}(\beta_1 \mathbf{w}_1, \ldots, \beta_m \mathbf{w}_m) = \mathrm{span}(\mathbf{w}_1, \ldots, \mathbf{w}_m).\) \(\lhd\)

**2.4** （改编自[Sol]）假设 \(U_1\) 和 \(U_2\) 是向量空间 \(V\) 的线性子空间。证明 \(U_1 \cap U_2\) 是 \(V\) 的一个线性子空间。\(U_1 \cup U_2\) 是否总是 \(V\) 的一个线性子空间？ \(\lhd\)

**2.5** 令 \(\mathcal{U}, \mathcal{V}\) 是 \(V\) 的线性子空间，使得 \(\mathcal{U} \subseteq \mathcal{V}\)。证明 \(\mathrm{dim}(\mathcal{U}) \leq \mathrm{dim}(\mathcal{V})\)。[*提示:* 完成基。] \(\lhd\)

**2.6** （改编自[Axl]）证明如果 \(\{\mathbf{v}_1,\ldots,\mathbf{v}_n\}\) 张成 \(U\)，那么该列表也张成 \(U\)。

$$ \{\mathbf{v}_1-\mathbf{v}_2, \mathbf{v}_2-\mathbf{v}_3,\ldots,\mathbf{v}_{n-1}-\mathbf{v}_n,\mathbf{v}_n\}, $$

通过从每个向量（除了最后一个）中减去以下向量得到。 \(\lhd\)

**2.7** （改编自[Axl]）证明如果 \(\{\mathbf{v}_1,\ldots,\mathbf{v}_n\}\) 是线性无关的，那么该列表也是线性无关的。

$$ \{\mathbf{v}_1-\mathbf{v}_2, \mathbf{v}_2-\mathbf{v}_3,\ldots,\mathbf{v}_{n-1}-\mathbf{v}_n,\mathbf{v}_n\}, $$

通过从每个向量（除了最后一个）中减去以下向量得到。 \(\lhd\)

**2.8** （改编自[Axl]）假设 \(\{\mathbf{v}_1,\ldots,\mathbf{v}_n\}\) 在 \(U\) 中是线性无关的，且 \(\mathbf{w} \in U\)。证明如果 \(\{\mathbf{v}_1 + \mathbf{w},\ldots, \mathbf{v}_n + \mathbf{w}\}\) 是线性相关的，那么 \(\mathbf{w} \in \mathrm{span}(\mathbf{v}_1,\ldots,\mathbf{v}_n)\)。 \(\lhd\)

**2.9** 证明 \(U^\perp\) 是一个线性子空间。 \(\lhd\)

**2.10** 设 \(U \subseteq V\) 是一个线性子空间，并且 \(\mathbf{v} \in U\)。证明 \(\mathrm{proj}_U \mathbf{v} = \mathbf{v}\)。 \(\lhd\)

**2.11** 设 \(A \in \mathbb{R}^{n \times n}\) 是一个 \(n \times n\) 的方阵。证明，如果对于任意的 \(\mathbf{b} \in \mathbb{R}^n\) 存在唯一的 \(\mathbf{x} \in \mathbb{R}^n\) 使得 \(A \mathbf{x} = \mathbf{b}\)，则 \(A\) 是非奇异的。[提示：考虑 \(\mathbf{b} = \mathbf{0}\)。] \(\lhd\)

**2.12** 证明，如果 \(B \in \mathbb{R}^{n \times m}\) 和 \(C \in \mathbb{R}^{m \times p}\)，则 \((BC)^T = C^T B^T\). [提示：检查项是否匹配。] \(\lhd\)

**2.13** 设 \(A \in \mathbb{R}^{n\times m}\) 是一个 \(n\times m\) 矩阵，其列线性无关。证明 \(m\leq n\).\(\lhd\)

**2.14** 向量 \(\mathbf{v} = (0,0,1)\) 是否在由以下向量张成的空间中？

$$\begin{split} \mathbf{q}_1 = \frac{1}{\sqrt{2}} \begin{pmatrix} 1\\ 0\\ 1 \end{pmatrix}, \quad \mathbf{q}_2 = \frac{1}{\sqrt{3}} \begin{pmatrix} -1\\ 1\\ 1 \end{pmatrix}. \end{split}$$

\(\lhd\)

**2.15** 给定向量 \(\mathbf{v} = (1, 2, 3)\) 和由 \(\mathbf{u}_1 = (1, 0, 0)\) 和 \(\mathbf{u}_2 = (0, 1, 0)\) 张成的平面，求 \(v\) 在该平面上的正交投影。

**2.16** 给定正交基 \(\{(1/\sqrt{2}, 1/\sqrt{2}), (-1/\sqrt{2}, 1/\sqrt{2})\}\)，求向量 \(\mathbf{v} = (3, 3)\) 在由该基张成的子空间上的投影。

**2.17** 求 \(\mathbf{v} = (4, 3, 0)\) 在 \(\mathbb{R}³\) 中由 \(\mathbf{u} = (1, 1, 1)\) 张成的直线上的正交投影。 \(\lhd\)

**2.18** 设 \(Q, W \in \mathbb{R}^{n \times n}\) 是可逆的。证明 \((Q W)^{-1} = W^{-1} Q^{-1}\) 和 \((Q^T)^{-1} = (Q^{-1})^T\). \(\lhd\)

**2.19** 证明 *Reducing a Spanning List Lemma*。 \(\lhd\)

**2.20** 证明对于任意的 \(\mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3 \in \mathbb{R}^n\) 和 \(\beta \in \mathbb{R}\)：

a) \(\langle \mathbf{x}_1, \mathbf{x}_2 \rangle = \langle \mathbf{x}_2, \mathbf{x}_1 \rangle\)

b) \(\langle \beta \,\mathbf{x}_1 + \mathbf{x}_2, \mathbf{x}_3 \rangle = \beta \,\langle \mathbf{x}_1,\mathbf{x}_3\rangle + \langle \mathbf{x}_2,\mathbf{x}_3\rangle\)

c) \(\|\mathbf{x}_1\|² = \langle \mathbf{x}_1, \mathbf{x}_1 \rangle\)

\(\lhd\)

**2.21** 使用为 Householder 反射引入的符号，定义

$$ \tilde{\mathbf{z}}_1 = \frac{\|\mathbf{y}_1\| \,\mathbf{e}_1^{(n)} + \mathbf{y}_1}{\| \|\mathbf{y}_1\| \,\mathbf{e}_1^{(n)} + \mathbf{y}_1\|} \quad \text{和} \quad \tilde{H}_1 = I_{n\times n} - 2\tilde{\mathbf{z}}_1 \tilde{\mathbf{z}}_1^T. $$

a) 证明 \(\tilde{H}_1 \mathbf{y}_1 = - \|\mathbf{y}_1\| \,\mathbf{e}_1^{(n)}\).

b) 在 \(\mathbf{y}_1 = \|\mathbf{y}_1\| \,\mathbf{e}_1^{(n)}\) 的情况下计算矩阵 \(\tilde{H}_1\). \(\lhd\)

**2.22** 证明两个正交矩阵 \(Q_1\) 和 \(Q_2\) 的乘积也是正交的。 \(\lhd\)

**2.23** 证明 \((U^\perp)^\perp = U\). \(\lhd\)

**2.24** 计算 \(U^\perp \cap U\) 并证明你的答案。 \(\lhd\)

**2.25** (改编自 [Sol]) 如果 \(\mathbf{x}, \mathbf{y} \in \mathbb{R}^m\) 且 \(\|\mathbf{x}\| = \|\mathbf{y}\|\)，编写一个算法来找到一个正交矩阵 \(Q\) 使得 \(Q\mathbf{x} = \mathbf{y}\)。\(\lhd\)

**2.26** (改编自 [Sol]) 假设 \(A \in \mathbb{R}^{m \times n}\) 的秩为 \(m\)，其中 \(m \leq n\)。令

$$ A^T = Q R $$

是通过 Gram-Schmidt 算法获得的 \(A^T\) 的 QR 分解。以 \(Q\) 和 \(R\) 为条件，提供一个解 \(\mathbf{x}\) 来解决欠定系统 \(A\mathbf{x} = \mathbf{b}\)。[*提示:* 首先尝试平方情况。然后通过添加 \(0\) 来猜测和检查一般情况的解。] \(\lhd\)

**2.27** (改编自 [Sol]) 令 \(A \in \mathbb{R}^{m \times m}\) 具有满列秩，并且假设 \(L \in \mathbb{R}^{m \times m}\) 是一个具有正对角元素的 lower triangular 矩阵，使得 \(A^T A = L L^T\)（这被称为 Cholesky 分解）。

a) 通过证明其列线性无关来证明 \(L^T\) 是可逆的。

b) 定义 \(Q = A(L^T)^{-1}\)。证明 \(Q\) 的列形成一个正交列表。

c) 使用 (b) 中的矩阵 \(Q\) 给出 \(A\) 的 QR 分解。确保显示 \(R\) 具有所需的结构。

\(\lhd\)

**2.28** (改编自 [Sol]) 假设 \(A\in\mathbb{R}^{m\times n}\) 具有满列秩，并且 \(A = QR\) 是通过 Gram-Schmidt 算法获得的 QR 分解。证明 \(P_0 = I_{m\times m} - QQ^T\) 是 \(A^T\) 的零空间的投影矩阵。[*提示:* 检查 *正交投影定理* 中的几何特征。] \(\lhd\)

**2.29** 假设我们考虑 \(\mathbf{a} \in \mathbb{R}^n\) 作为 \(n \times 1\) 矩阵。明确写出它的 QR 分解。\(\lhd\)

**2.30** (改编自 [Sol]) 证明一个矩阵 \(A \in \mathbb{R}^{m \times n}\) 的列线性无关可以被分解为 \(A = QL\)，其中 \(L\) 是下三角矩阵。[*提示:* 修改我们的程序以获得 QR 分解。] \(\lhd\)

**2.31** (改编自 [Sol]) 假设 \(A \in \mathbb{R}^{n \times p}\), \(B \in \mathbb{R}^{m \times p}\), \(\mathbf{a} \in \mathbb{R}^n\), 和 \(\mathbf{b} \in \mathbb{R}^m\). 找到一个线性方程组，它被任何使 \(\|A\mathbf{x} - \mathbf{a}\|² + \|B\mathbf{x} - \mathbf{b}\|²\) 最小的 \(\mathbf{x}\) 满足。[*提示:* 将问题重写为线性最小二乘问题。] \(\lhd\)

**2.32** 令 \(P\) 为一个投影矩阵。证明：

a) \(P² = P\)

b) \(P^T = P\)

c) 检查上述两个关于在 \(\mathbb{R}³\) 中投影到 \(\mathbf{u} = (1, 0, 1)\) 的张量上的断言。

\(\lhd\)

**2.33** 证明对于任何 \(\mathbf{x}_1, \ldots, \mathbf{x}_m, \mathbf{y}_1, \ldots, \mathbf{y}_\ell, \in \mathbb{R}^n\),

$$ \left\langle \sum_{i=1}^m \mathbf{x}_i, \sum_{j=1}^\ell \mathbf{y}_j \right\rangle = \sum_{i=1}^m \sum_{j=1}^\ell \langle \mathbf{x}_i,\mathbf{y}_j\rangle. $$

\(\lhd\)

**2.34** 设 \(A \in \mathbb{R}^{n\times m}\) 是一个 \(n\times m\) 的矩阵，其中 \(n \geq m\)，且设 \(\mathbf{b} \in \mathbb{R}^n\) 是一个向量。设

$$ \mathbf{p}^* = \arg\min_{\mathbf{p} \in U} \|\mathbf{p} - \mathbf{b}\|, $$

以及 \(\mathbf{x}^*\) 使得 \(\mathbf{p}^* = A \mathbf{x}^*\)。构造一个 \(A\) 和一个 \(\mathbf{b}\)，使得 \(\mathbf{x}^*\) 不是唯一的。 \(\lhd\)

**2.35** 设 \(H_k \in \mathbb{R}^{n \times n}\) 是一个形式为

$$\begin{split} H_k = \begin{pmatrix} I_{(k-1)\times (k-1)} & \mathbf{0} \\ \mathbf{0} & F_k \end{pmatrix} \end{split}$$

其中

$$ F_k = I_{(n-k+1) \times (n-k+1)} - 2 \mathbf{z}_k \mathbf{z}_k^T, $$

对于某个单位向量 \(\mathbf{z}_k \in \mathbb{R}^{n - k + 1}\)。证明 \(H_k\) 是一个正交矩阵。 \(\lhd\)

**2.36** 使用 Householder 反射的符号，设 \(A\) 有线性无关的列。

a) 假设 \(\mathbf{z}_1 \neq \mathbf{0}\)，证明 \(\mathbf{y}_2 \neq \mathbf{0}\)。

b) 假设 \(\mathbf{z}_1, \mathbf{z}_2 \neq \mathbf{0}\)，证明 \(\mathbf{y}_3 \neq \mathbf{0}\)。 \(\lhd\)

**2.37** 设 \(R \in \mathbb{R}^{n \times m}\)，其中 \(n \geq m\)，是上三角矩阵，对角线上的元素非零。证明 \(R\) 的列是线性无关的。 \(\lhd\)

**2.38** 考虑输入数据 \(\{(\mathbf{x}_i, y_i)\}_{i=1}^n\) 上的线性回归问题，其中 \(\mathbf{x}_i \in \mathbb{R}^d\) 且 \(y_i \in \mathbb{R}\) 对所有 \(i\) 都成立。

a) 如果存在 \(d+1\) 个形式为

$$\begin{split} \begin{pmatrix} 1 \\ \mathbf{x}_i \end{pmatrix} \end{split}$$

形成一个独立的列表。

b) 在 \(d=1\) 的情况下简化前面的条件，即当 \(x_i\) 是实值时。

\(\lhd\)

**2.39** 设 \(U\) 是 \(\mathbb{R}^n\) 的一个线性子空间，并假设 \(\mathbf{q}_1, \ldots, \mathbf{q}_k\) 是 \(U\) 的一个正交基。设 \(\mathbf{v} \in \mathbb{R}^n\)。

a) 证明 \(\mathbf{v}\) 在子空间 \(U^\perp\) 上的正交投影（即 \(U\) 的正交补）是 \(\mathbf{v} - \mathrm{proj}_U \mathbf{v}\)。[*提示:* 使用投影的几何特征。]

b) 设 \(Q\) 是列向量为 \(\mathbf{q}_1, \ldots, \mathbf{q}_k\) 的矩阵。用 \(Q\) 表示到子空间 \(U^\perp\) 的投影矩阵。[*提示:* 使用 a)。]

\(\lhd\)

**2.40** 证明以下命题，该命题被称为 *子空间交集引理*。设 \(\mathcal{S}_1\) 和 \(\mathcal{S}_2\) 是 \(\mathbb{R}^d\) 的线性子空间，并设

$$ \mathcal{S}_1 + \mathcal{S}_2 = \{\mathbf{x}_1 + \mathbf{x}_2 \,:\, \forall \mathbf{x}_1 \in \mathcal{S}_1, \mathbf{x}_2 \in \mathcal{S}_2\}. $$

然后，它成立

$$ \mathrm{dim}(\mathcal{S}_1 + \mathcal{S}_2) = \mathrm{dim}(\mathcal{S}_1) + \mathrm{dim}(\mathcal{S}_2) - \mathrm{dim}(\mathcal{S}_1 \cap \mathcal{S}_2). $$

[*提示:* 考虑 \(\mathcal{S}_1 \cap \mathcal{S}_2\) 的一个基，并将其扩展为 \(\mathcal{S}_1\) 和 \(\mathcal{S}_2\) 的基。证明得到的向量列表是线性无关的。] \(\lhd\)

**2.41** 设 \(\mathcal{U}, \mathcal{V} \subseteq \mathbb{R}^d\) 是子空间，且 \(\mathrm{dim}(\mathcal{U}) + \mathrm{dim}(\mathcal{V}) > d\)。使用问题 2.40 证明在 \(\mathcal{U} \cap \mathcal{V}\) 的交集中存在一个非零向量。 \(\lhd\)

**2.42** 证明，对于 \(\mathcal{V} = \mathbb{R}^d\) 的任何线性子空间 \(\mathcal{S}_1, \ldots, \mathcal{S}_m\)，都有

$$ \mathrm{dim}\left(\bigcap_{k=1}^m \mathcal{S}_k\right) \geq \sum_{k=1}^m \mathrm{dim}\left(\mathcal{S}_k\right) - (m-1) \,\mathrm{dim}(\mathcal{V}). $$

[*提示:* 使用问题 2.40 中的 *子空间交集引理* 和归纳法。] \(\lhd\)

**2.43** 设 \(\mathcal{W}\) 是 \(\mathbb{R}^d\) 的一个线性子空间，且 \(\mathbf{w}_1,\ldots,\mathbf{w}_k\) 是 \(\mathcal{W}\) 的一个正交基。证明存在一个包含 \(\mathbf{w}_i\) 的 \(\mathbb{R}^d\) 的正交基。 \(\lhd\)

**2.44** 设 \(\mathcal{U}, \mathcal{V}\) 是 \(V\) 的线性子空间，且 \(\mathcal{U} \subseteq \mathcal{V}\)。证明如果 \(\mathrm{dim}(\mathcal{U}) = \mathrm{dim}(\mathcal{V})\)，则 \(\mathcal{U} = \mathcal{V}\)。[*提示:* 完成基。] \(\lhd\)

**2.45** 设 \(\mathcal{U}, \mathcal{V}\) 是 \(V\) 的线性子空间，且 \(\mathcal{U} \subseteq \mathcal{V}\)。证明如果 \(\mathrm{dim}(\mathcal{U}) < \mathrm{dim}(\mathcal{V})\)，则存在一个 \(\mathbf{u} \in \mathcal{V}\) 使得 \(\mathbf{u} \notin \mathcal{U}\)。[*提示:* 完成基。] \(\lhd\)

**2.46** 设 \(\mathcal{Z} \subseteq \mathcal{W}\) 是线性子空间，且 \(\mathrm{dim}(\mathcal{Z}) < \mathrm{dim}(\mathcal{W})\)。证明存在一个与 \(\mathcal{Z}\) 正交的单位向量 \(\mathbf{w} \in \mathcal{W}\)。 \(\lhd\)

**2.47** 设 \(\mathcal{W} = \mathrm{span}(\mathbf{w}_1,\ldots,\mathbf{w}_\ell)\) 且 \(\mathbf{z} \in \mathcal{W}\) 的范数为单位。证明存在一个包含 \(\mathbf{z}\) 的 \(\mathcal{W}\) 的正交基。 \(\lhd\)

**2.48** 设 \(\{\mathbf{u}_1,\ldots,\mathbf{u}_m\}\) 是一个独立列表。证明对于任何非零的 \(\beta_1,\ldots,\beta_m \in \mathbb{R}\)，列表 \(\{\beta_1\mathbf{u}_1,\ldots,\beta_m\mathbf{u}_m\}\) 也是独立的。 \(\lhd\)

**2.49** 设 \(\mathcal{Z}\) 是 \(\mathbb{R}^n\) 的一个线性子空间，且 \(\mathbf{v} \in \mathbb{R}^n\)。证明 \(\|\mathrm{proj}_{\mathcal{Z}}\mathbf{v}\|_2 \leq \|\mathbf{v}\|_2\)。 \(\lhd\)
