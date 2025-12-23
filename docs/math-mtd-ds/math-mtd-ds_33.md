# 4.7\. 练习题#

> 原文：[`mmids-textbook.github.io/chap04_svd/exercises/roch-mmids-svd-exercises.html`](https://mmids-textbook.github.io/chap04_svd/exercises/roch-mmids-svd-exercises.html)

## 4.7.1\. 热身练习表#

*(在克劳德、双子星和 ChatGPT 的帮助下)*

**章节 4.2**

**E4.2.1** 计算矩阵 \(A = \begin{pmatrix} 1 & 2 & 3\\ 0 & 1 & 1\\ 1 & 3 & 4 \end{pmatrix}\) 的秩。

**E4.2.2** 设 \(A = \begin{pmatrix} 1 & 2\\ 3 & 4 \end{pmatrix}\) 和 \(B = \begin{pmatrix} 1 & 0\\ 0 & 1 \end{pmatrix}\)。计算 \(\mathrm{rk}(A)\)，\(\mathrm{rk}(B)\) 和 \(\mathrm{rk}(A+B)\)。

**E4.2.3** 找出矩阵 \(A = \begin{pmatrix} 3 & 1\\ 1 & 3 \end{pmatrix}\) 的特征值和对应的特征向量。

**E4.2.4** 验证 E4.2.3 中的特征向量是正交的。

**E4.2.5** 写出矩阵 \(A\) 从 E4.2.3 的谱分解。

**E4.2.6** 判断矩阵 \(A = \begin{pmatrix} 2 & -1\\ -1 & 2 \end{pmatrix}\) 是正定、正半定还是都不是。

**E4.2.7** 计算 \(\mathbf{u} = (1, 2, 3)\) 和 \(\mathbf{v} = (4, 5)\) 的外积 \(\mathbf{u}\mathbf{v}^T\)。

**E4.2.8** 将矩阵 \(A = \begin{pmatrix} 1 & 2 & 3\\ 2 & 4 & 6\\ 3 & 6 & 9 \end{pmatrix}\) 写成秩为 1 的矩阵之和。

**E4.2.9** 设 \(A = \begin{pmatrix} 1 & 2 \\ 3 & 6 \end{pmatrix}\)。\(A\) 的秩是多少？

**E4.2.10** 设 \(\mathbf{u} = \begin{pmatrix} 1 \\ 2 \end{pmatrix}\) 和 \(\mathbf{v} = \begin{pmatrix} 3 \\ 4 \end{pmatrix}\)。计算外积 \(\mathbf{u} \mathbf{v}^T\)。

**E4.2.11** 设 \(A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}\)。找出 \(A\) 的特征值和特征向量。

**E4.2.12** 设 \(A = \begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix}\)。验证 \(\mathbf{v} = \begin{pmatrix} 1 \\ 1 \end{pmatrix}\) 是 \(A\) 的一个特征向量，并找出相应的特征值。

**E4.2.12** 设 \(A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}\)。找出 \(A\) 的列空间的一个基。

**E4.2.14** 设 \(A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}\)。找出 \(A\) 的零空间的基。

**E4.2.15** 设 \(A = \begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix}\)。\(A\) 是正半定吗？

**E4.2.16** 判断函数 \(f(x, y) = x² + y² + xy\) 是否是凸函数。

**E4.2.17** 函数 \(f(x, y) = x² - y²\) 是凸函数吗？

**E4.2.18** 检查函数 \(f(x, y) = e^{x+y}\) 的凸性。

**E4.2.19** 检查函数 \(f(x, y) = \log(x² + y² + 1)\) 的凸性。

**E4.2.20** 函数 \(f(x, y) = xy\) 是凸函数吗？

**章节 4.3**

**E4.3.1** 设 \(\boldsymbol{\alpha}_1 = (1, 2)\) 和 \(\boldsymbol{\alpha}_2 = (-2, 1)\)。找出 \(\mathbb{R}²\) 中一个单位向量 \(\mathbf{w}_1\)，使得 \(\|A\mathbf{w}_1\|²\) 最大，其中 \(A\) 是行向量为 \(\boldsymbol{\alpha}_1^T\) 和 \(\boldsymbol{\alpha}_2^T\) 的矩阵。

**E4.3.2** 令 \(\mathbf{u} = (1/\sqrt{2}, 1/\sqrt{2})\) 和 \(\mathbf{v} = (1/\sqrt{2}, -1/\sqrt{2})\)。计算外积 \(\mathbf{u}\mathbf{v}^T\)。

**E4.3.3** 令 \(A = \begin{pmatrix} 3 & 0 \\ 0 & 2 \end{pmatrix}\)。求 \(A\) 的奇异值分解。

**E4.3.4** 令 \(A = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}\)。求 \(A\) 的紧凑奇异值分解。

**E4.3.5** 令 \(A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}\) 如 E4.3.5 中所述。求 \(A^TA\) 的特征值和特征向量。

**E4.3.6** 令 \(A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}\) 如 E4.3.5 中所述。求 \(AA^T\) 的特征值和特征向量。

**E4.3.7** 令 \(A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}\) 如 E4.3.5 中所述。求 \(A\) 的紧凑奇异值分解。

**E4.3.8** 令 \(A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}\) 如 E4.3.5 中所述。求 \(A\) 的完全奇异值分解。

**E4.3.9** 令 \(A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}\) 如 E4.3.5 中所述。求 \(A\) 的四个基本子空间中的每一个的正交基。

**E4.3.10** 给定数据点 \(\boldsymbol{\alpha}_1 = (1,2)\)，\(\boldsymbol{\alpha}_2 = (2,1)\)，\(\boldsymbol{\alpha}_3 = (-1,1)\)，和 \(\boldsymbol{\alpha}_4 = (3,-1)\)，计算矩阵 \(A\) 以及矩阵 \(A^TA\) 和 \(AA^T\)。

**E4.3.11** 求 E4.3.10 中的矩阵 \(A^TA\) 的特征值和特征向量。

**E4.3.12** 计算 E4.3.10 中的矩阵 \(A\) 的紧凑奇异值分解。

**E4.3.13** 从 E4.3.12 计算矩阵 \(U_1 \Sigma_1 V_1^T\) 并验证它等于 E4.3.10 中的 \(A\)。

**E4.3.14** 基于 E4.3.12，验证关系 \(A^T \mathbf{u}_i = \sigma_i \mathbf{v}_i\) 对于 \(i=1,2\)。

**E4.3.15** 基于 E4.3.12，验证关系 \(A^TA\mathbf{v}_i = \sigma_i² \mathbf{v}_i\) 和 \(AA^T\mathbf{u}_i = \sigma_i² \mathbf{u}_i\) 对于 \(i=1,2\)。

**E4.3.16** 基于 E4.3.12，找到维度为 \(k=2\) 的最佳逼近子空间，并计算到该子空间的平方距离之和。

**E4.3.17** 基于 E4.3.12，找到维度为 \(k=1\) 的最佳逼近子空间，并计算到该子空间的平方距离之和。

**E4.3.18** 基于 E4.3.17，计算通过截断奇异值分解得到的 \(k=1\) 的矩阵 \(Z\)。

**第 4.4 节**

**E4.4.1** 令 \(A = \begin{pmatrix} 3 & 0 \\ 0 & 1 \end{pmatrix}\)。计算 \(A²\)，\(A³\)，和 \(A⁴\)。你观察到了什么模式？

**E4.4.2** 令 \(B = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}\)。计算 \(B²\) 和 \(B³\)。如果你继续计算 \(B\) 的高次幂，你期望会发生什么？

**E4.4.3** 给定对称矩阵 \(A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}\) 和向量 \(\mathbf{x} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}\)，计算 \(k = 1, 2, 3\) 时 \(A^k \mathbf{x}\) 以及 \(\frac{A^k \mathbf{x}}{\|A^k \mathbf{x}\|}\)。

**E4.4.4** 给定对称矩阵 \(A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}\) 和向量 \(\mathbf{x} = \begin{pmatrix} 0 \\ 1 \end{pmatrix}\)，计算 \(A^k \mathbf{x}\) 对于 \(k = 1, 2, 3\) 和 \(\frac{A^k \mathbf{x}}{\|A^k \mathbf{x}\|}\)。

**E4.4.5** 在矩阵 \(A = \begin{pmatrix} 2 & 0 \\ 0 & 1 \end{pmatrix}\) 和向量 \(\mathbf{x} = \begin{pmatrix} 1 \\ 1 \end{pmatrix}\) 的条件下，计算 \(A^k \mathbf{x}\) 对于 \(k = 1, 2, 3\)。

**E4.4.6** 令 \(A = \begin{pmatrix} 4 & 1 \\ 1 & 4 \end{pmatrix}\)。求 \(A\) 的特征值和特征向量。

**E4.4.7** 令 \(A\) 如 E4.4.6 所示。使用 \(A\) 的谱分解来计算 \(A²\) 和 \(A³\)。

**E4.4.8** 令 \(A\) 如 E4.4.6 所示。令 \(\mathbf{x} = \begin{pmatrix} 2 \\ 1 \end{pmatrix}\)。计算 \(A² \mathbf{x}\) 和 \(A³ \mathbf{x}\)。随着幂的增加，这些向量的方向有什么特点？

**E4.4.9** 令 \(A = \begin{pmatrix} 1 & 2 \\ 0 & 1 \end{pmatrix}\)。计算 \(A^TA\)。通过计算其特征值来确定 \(A^TA\) 是否是正半定的。

**第 4.5 节**

**E4.5.1** 给定一个加载向量 \(\boldsymbol{\phi}_1 = \left(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}\right)\)，验证它满足单位范数约束。

**E4.5.2** 给定两个加载向量 \(\boldsymbol{\phi}_1 = \left(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}\right)\) 和 \(\boldsymbol{\phi}_2 = \left(\frac{1}{\sqrt{2}}, -\frac{1}{\sqrt{2}}\right)\)，验证它们是正交的。

**E4.5.3** 给定一个中心化数据矩阵 \(\tilde{X} = \begin{pmatrix} -1 & 1 \\ 1 & -1 \end{pmatrix}\) 和一个加载向量 \(\boldsymbol{\phi}_1 = \left(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}\right)\)，计算第一个主成分得分 \(t_{i1}\)。

**E4.5.4** 给定一个中心化数据矩阵 \(\tilde{X} = \begin{pmatrix} -1 & 1 \\ 1 & -1 \end{pmatrix}\) 和两个加载向量 \(\boldsymbol{\phi}_1 = \left(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}\right)\) 和 \(\boldsymbol{\phi}_2 = \left(\frac{1}{\sqrt{2}}, -\frac{1}{\sqrt{2}}\right)\)，计算第一个和第二个主成分得分 \(t_{i1}\) 和 \(t_{i2}\)。

**E4.5.5** 给定来自 E4.5.4 的第一个和第二个主成分得分，验证它们是不相关的。

**E4.5.6** 给定一个数据矩阵 \(X = \begin{pmatrix} 1 & 3 \\ -1 & -3\end{pmatrix}\)，计算中心化数据矩阵 \(\tilde{X}\)。

**E4.5.7** 给定数据矩阵

\[\begin{split} X = \begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{pmatrix}, \end{split}\]

计算均值中心化的数据矩阵。

**E4.5.8** 给定来自 E4.5.7 的中心化数据矩阵的奇异值分解，提取第一个主成分加载向量 \(\boldsymbol{\phi}_1\)。

**E4.5.9** 给定中心化数据矩阵 \(\tilde{X}\) 和来自 E4.5.7 和 E4.5.8 的第一个主成分加载向量 \(\boldsymbol{\phi}_1\)，计算第一个主成分得分 \(t_{i1}\)。

**E4.5.10** 给定一个具有中心列的数据矩阵 \(X\)，证明 \(X\) 的样本协方差矩阵可以表示为 \(\frac{1}{n-1} X^T X\)。

**E4.5.11** 给定第一个主成分的加载向量 \(\boldsymbol{\phi}_1 = (0.8, 0.6)\)，找到第二个主成分的加载向量 \(\boldsymbol{\phi}_2\)。

**E4.5.12** 给定数据点 \(\mathbf{x}_1 = (1, 0)\), \(\mathbf{x}_2 = (0, 1)\), 和 \(\mathbf{x}_3 = (-1, 0)\)，计算第一个主成分向量。

**第 4.6 节**

**E4.6.1** 计算 \(A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{pmatrix}\) 的 Frobenius 范数。

**E4.6.2** 计算 \(A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}\) 的奇异值分解。

**E4.6.3** 设 \(A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}\)。计算 \(A\) 的 Frobenius 范数 \(\|A\|_F\)。

**E4.6.4** 设 \(A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}\)。计算 \(A\) 的诱导 2-范数 \(\|A\|_2\)。

**E4.6.5** 设 \(A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}\) 和 \(B = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}\)。计算 \(\|A - B\|_F\)。

**E4.6.6** 设 \(A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}\) 和 \(A_1 = \sigma_1 \mathbf{u}_1 \mathbf{v}_1^T\) 是 \(A\) 的秩为 1 的截断 SVD。计算 \(\|A - A_1\|_F\)。

**E4.6.7** 设 \(A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}\) 和 \(A_1 = \sigma_1 \mathbf{u}_1 \mathbf{v}_1^T\) 是 \(A\) 的秩为 1 的截断 SVD。计算 \(\|A - A_1\|_2\)。

**E4.6.8** 设 \(A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}\)，\(\mathbf{b} = \begin{pmatrix} 5 \\ 10 \end{pmatrix}\)，和 \(\lambda = 1\)。计算 \(\min_{\mathbf{x} \in \mathbb{R}²} \|A \mathbf{x} - \mathbf{b}\|_2² + \lambda \|\mathbf{x}\|_2²\) 的岭回归解 \(\mathbf{x}^{**}\)。

**E4.6.9** 设 \(A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}\)。计算 \(\|A^{-1}\|_2\)。

**E4.6.10** 对于 \(A = \begin{pmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{pmatrix}\)，\(b = \begin{pmatrix} 1 \\ 2 \\ 2 \end{pmatrix}\)，和 \(\lambda = 1\)，计算岭回归解。

## 4.7.2\. 问题#

**4.1** 设 \(Q \in \mathbb{R}^{n \times n}\) 是一个正交矩阵。使用 *通过谱分解定理的 SVD* 来计算 \(Q\) 的 SVD。在这种情况下，紧凑型 SVD 和完整 SVD 有区别吗？\(\lhd\)

**4.2** 设 \(W \subseteq \mathbb{R}^m\) 是一个超平面。证明存在一个单位向量 \(\mathbf{z} \in \mathbb{R}^m\)，使得

\[ \mathbf{w} \in W \iff \langle \mathbf{z}, \mathbf{w} \rangle = 0. \]

\(\lhd\)

**4.3** 构造一个矩阵 \(A \in \mathbb{R}^{n \times n}\)，使得最大化问题存在多个解。

\[ \mathbf{v}_1 \in \arg\max\{\|A \mathbf{v}\|:\|\mathbf{v}\| = 1\}. \]

\(\lhd\)

**4.4** 证明正定矩阵的 *正定性描述* 的一个类似命题。\(\lhd\)

**4.5** （改编自 [Sol]）证明 \(\|A\|_2 = \|\Sigma\|_2\)，其中 \(A = U \Sigma V^T\) 是 A 的奇异值分解。 \(\lhd\)

**4.6** 设 \(A\), \(U \Sigma V^T\), \(B\) 如**幂迭代引理**中所述。进一步假设 \(\sigma_2 > \sigma_3\)，且 \(\mathbf{y} \in \mathbb{R}^m\) 满足 \(\langle \mathbf{v}_1, \mathbf{y} \rangle = 0\) 和 \(\langle \mathbf{v}_2, \mathbf{y} \rangle > 0\)。证明当 \(k \to +\infty\) 时，\(\frac{B^{k} \mathbf{y}}{\|B^{k} \mathbf{y}\|} \to \mathbf{v}_2\)。你将如何找到这样的 \(\mathbf{y}\)? \(\lhd\)

**4.7** 设 \(A \in \mathbb{R}^{n \times m}\)。使用**柯西-施瓦茨不等式**来证明

\[ \|A\|_2 = \max \left\{ \mathbf{x}^T A \mathbf{y} \,:\, \|\mathbf{x}\| = \|\mathbf{y}\| = 1 \right\}. \]

\(\lhd\)

**4.8** 设 \(A \in \mathbb{R}^{n \times m}\).

a) 证明 \(\|A\|_F² = \sum_{j=1}^m \|A \mathbf{e}_j\|²\)。

b) 使用 (a) 和**柯西-施瓦茨不等式**来证明 \(\|A\|_2 \leq \|A\|_F\).

c) 给出一个例子，使得 \(\|A\|_F = \sqrt{n} \|A\|_2\). \(\lhd\)

**4.9** 使用**柯西-施瓦茨不等式**来证明对于任何 \(A, B\)，当 \(AB\) 定义良好时，有

\[ \|A B \|_F \leq \|A\|_F \|B\|_F. \]

\(\lhd\)

**4.10** (*注意*: 指的是在线补充材料。) 设 \(A \in \mathbb{R}^{n \times n}\) 是非奇异的，其奇异值分解为 \(A = U \Sigma V^T\)，其中奇异值满足 \(\sigma_1 \geq \cdots \geq \sigma_n > 0\)。证明

\[ \min_{\mathbf{x} \neq \mathbf{0}} \frac{\|A \mathbf{x}\|}{\|\mathbf{x}\|} = \min_{\mathbf{y} \neq \mathbf{0}} \frac{\|\mathbf{y}\|}{\|A^{-1}\mathbf{y}\|} = \sigma_n = 1/\|A^+\|_2. \]

\(\lhd\)

**4.11** 设 \(X \in \mathbb{R}^{n \times d}\) 是一个矩阵，其行向量为 \(\mathbf{x}_1^T,\ldots,\mathbf{x}_n^T\)。将以下求和式用 \(X\) 的矩阵形式表示：

\[ \frac{1}{n} \sum_{i=1}^n \mathbf{x}_i \mathbf{x}_i^T. \]

证明你的答案。 \(\lhd\)

**4.12** 设 \(A \in \mathbb{R}^{d \times d}\) 是一个对称矩阵。证明 \(A \preceq M I_{d \times d}\) 当且仅当 \(A\) 的特征值不超过 \(M\)。类似地，\(m I_{d \times d} \preceq A\) 当且仅当 \(A\) 的特征值至少为 \(m\)。[*提示:* 注意到 \(A\) 的特征向量也是单位矩阵 \(I_{d \times d}\) 的特征向量。] \(\lhd\)

**4.13** 证明**正半定情况下的幂迭代引理**。 \(\lhd\)

**4.14** 回忆一下，矩阵 \(A = (a_{i,j})_{i,j} \in \mathbb{R}^n\) 的迹 \(\mathrm{tr}(A)\) 是其对角线元素的和，即 \(\mathrm{tr}(A) = \sum_{i=1}^n a_{i,i}\)。

a) 证明对于任何 \(A \in \mathbb{R}^{n \times m}\) 和 \(B \in \mathbb{R}^{m \times n}\)，有 \(\mathrm{tr}(AB) = \mathrm{tr}(BA)\)。

b) 使用 (a) 来证明，如果对称矩阵 \(A\) 的谱分解为 \(\sum_{i=1}^n \lambda_i \mathbf{q}_i \mathbf{q}_i^T\)，那么

\[ \mathrm{tr}(A) = \sum_{i=1}^n \lambda_i. \]

\(\lhd\)

**4.15** (改编自 [Sol]) 假设 \(A \in \mathbb{R}^{m \times n}\) 和 \(B \in \mathbb{R}^{n \times m}\)。证明 \(\|A\|²_F = \mathrm{tr}(A^T A)\) 和 \(\mathrm{tr}(A B) = \mathrm{tr}(B A)\)，其中记住矩阵 \(C\) 的迹 \(\mathrm{tr}(C)\) 是 \(C\) 对角元素的和。 \(\lhd\)

**4.16** (改编自 [Str]) 设 \(A \in \mathbb{R}^{n \times k}\) 和 \(B \in \mathbb{R}^{k \times m}\)。证明 \(AB\) 的零空间包含 \(B\) 的零空间。 \(\lhd\)

**4.17** (改编自 [Str]) 设 \(A \in \mathbb{R}^{n \times m}\)。证明 \(A^T A\) 与 \(A\) 有相同的零空间。 \(\lhd\)

**4.18** (改编自 [Str]) 设 \(A \in \mathbb{R}^{m \times r}\) 和 \(B \in \mathbb{R}^{r \times n}\)，它们的秩都是 \(r\)。

a) 证明 \(A^T A\)、\(B B^T\) 和 \(A^T A B B^T\) 都是可逆的。

b) 证明 \(r = \mathrm{rk}(A^T A B B^T) \leq \mathrm{rk}(AB)\)。

c) 推论出 \(\mathrm{rk}(AB) = r\)。

\(\lhd\)

**4.19** 对于任何 \(n \geq 1\)，给出一个矩阵 \(A \in \mathbb{R}^{n \times n}\) 的例子，使得 \(A \neq I_{n \times n}\) 且 \(A² = I_{n \times n}\)。 \(\lhd\)

**4.20** 设 \(A \in \mathbb{R}^{n \times k}\) 和 \(B \in \mathbb{R}^{r \times m}\)，并且设 \(D \in \mathbb{R}^{k \times r}\) 为对角矩阵。假设 \(k < r\)。计算 \(AD\) 的列和 \(DB\) 的行。 \(\lhd\)

**4.21** 计算 \(A = [-1]\) 的奇异值分解。 \(\lhd\)

**4.22** 在这个练习中，我们展示一个矩阵可以有许多不同的奇异值分解。设 \(Q, W \in \mathbb{R}^{n \times n}\) 为正交矩阵。给出 \(Q\) 的四个不同的奇异值分解，一个其中 \(V = I\)，一个其中 \(V = Q\)，一个其中 \(V = Q^T\)，一个其中 \(V = W\)。确保检查所有奇异值分解的要求。奇异值是否改变？ \(\lhd\)

**4.23** (改编自 [Sol]) 证明向矩阵中添加一行不能减少其最大的奇异值。 \(\lhd\)

**4.24** 设 \(\mathbf{v}_1,\ldots,\mathbf{v}_n \in \mathbb{R}^n\) 为一个正交基。固定 \(1 \leq k < n\)。设 \(Q_1\) 为列向量为 \(\mathbf{v}_1,\ldots,\mathbf{v}_k\) 的矩阵，设 \(Q_2\) 为列向量为 \(\mathbf{v}_{k+1},\ldots,\mathbf{v}_n\) 的矩阵。证明

\[ Q_1 Q_1^T + Q_2 Q_2^T = I_{n \times n}. \]

[*提示:* 两边乘以 \(\mathbf{e}_i\).] \(\lhd\)

**4.25** (*注意*: 指的是在线补充材料。) 设 \(Q \in \mathbb{R}^{n \times n}\) 为一个正交矩阵。计算其条件数

\[ \kappa_2(Q) = \|Q\|_2 \|Q^{-1}\|_2. \]

\(\lhd\)

**4.26** 证明奇异值分解关系引理。 \(\lhd\)

**4.27** 设 \(A = \sum_{j=1}^r \sigma_j \mathbf{u}_j \mathbf{v}_j^T\) 是 \(A \in \mathbb{R}^{n \times m}\) 的奇异值分解，其中 \(\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0\)。

定义

\[ B = A - \sigma_1 \mathbf{u}_1 \mathbf{v}_1^T. \]

证明

\[ \mathbf{v}_2 \in \arg\max\{\|B \mathbf{v}\|:\|\mathbf{v}\| = 1\}. \]

\(\lhd\)

**4.28** (改编自 [Sol]) \(A \in \mathbb{R}^{n \times n}\) 的稳定秩定义为

\[ \mathrm{StableRk}(A) = \frac{\|A\|_F²}{\|A\|_2²}. \]

a) 证明如果矩阵 \(A\) 的所有列都是相同的非零向量 \(\mathbf{v} \in \mathbb{R}^n\)，那么 \(\mathrm{StableRk}(A) = 1\)。

b) 证明当 \(A\) 的列是正交归一时，则 \(\mathrm{StableRk}(A) = n\)。

c) \(\mathrm{StableRk}(A)\) 是否总是整数？证明你的答案。

d) 更一般地，证明

\[ 1 \leq \mathrm{StableRk}(A) \leq n. \]

e) 证明在一般情况下，以下等式成立

\[ \mathrm{StableRk}(A) \leq \mathrm{Rk}(A). \]

[*提示：使用*矩阵范数和奇异值引理*。] \(\lhd\)

**4.29** 设 \(A \in \mathbb{R}^{n \times n}\) 是一个具有满奇异值分解 \(A = U \Sigma V^T\) 的方阵。

a) 证明以下公式的正确性

\[ A = (U V^T) (V \Sigma V^T). \]

b) 设

\[ Q = U V^T, \qquad S = V \Sigma V^T. \]

证明 \(Q\) 是正交的，且 \(S\) 是正半定的。形式为 \(A = Q S\) 的分解称为极分解。\(\lhd\)

**4.30** 设 \(A \in \mathbb{R}^{n \times m}\) 是一个具有满奇异值分解 \(A = U \Sigma V^T\) 的矩阵，其中奇异值满足 \(\sigma_1 \geq \cdots \geq \sigma_m \geq 0\)。定义 \(m\) 维单位球 \(\mathbb{S}^{m-1} = \{\mathbf{x} \in \mathbb{R}^m\,:\,\|\mathbf{x}\| = 1\}\)。证明

\[ \sigma_m² = \min_{\mathbf{x} \in \mathbb{S}^{m-1}} \|A \mathbf{x}\|²\. \]

\(\lhd\)

**4.31** 设 \(A \in \mathbb{R}^{n \times m}\) 是一个具有列 \(\mathbf{a}_1,\ldots,\mathbf{a}_m\) 的矩阵。证明对于所有 \(i\)

\[ \|\mathbf{a}_i\|_2 \leq \|A\|_2. \]

\(\lhd\)

**4.32** 设 \((U_i, V_i)\)，\(i = 1, \ldots, n\) 是取值在 \(\mathbb{R}²\) 中的独立同分布随机向量。假设 \(\mathbb{E}[U_1²], \mathbb{E}[V_1²] < +\infty\)。

a) 证明

\[ \mathbb{E}\left[\frac{1}{n} \sum_{i=1}^n U_i \right] = \mathbb{E}[U_1]. \]

b) 证明

\[ \mathbb{E}\left[\frac{1}{n-1} \sum_{i=1}^n \left(U_i - \frac{1}{n} \sum_{j=1}^n U_j\right)²\right] = \mathrm{Var}[U_1]. \]

c) 证明

\[ \mathbb{E}\left[\frac{1}{n-1} \sum_{i=1}^n \left(U_i - \frac{1}{n} \sum_{j=1}^n U_j\right) \left(V_i - \frac{1}{n} \sum_{k=1}^n V_k\right) \right] = \mathrm{Cov}[U_1, V_1]. \]

从字面上讲，上述每个方程的左侧（更精确地说，是在期望内部）是右侧的一个所谓的无偏估计量。这些估计量分别称为样本均值、样本方差和样本协方差。\(\lhd\)

**4.33** 给出在数据矩阵 \(A \in \mathbb{R}^{n \times m}\) 和一个列正交归一的矩阵 \(W\) 的条件下，最佳 \(k\) 维逼近子空间的一个等价矩阵表示。[*提示：使用 Frobenius 范数。] \(\lhd\)

**4.34** 设 \(\mathbf{q}_1,\ldots,\mathbf{q}_m\) 是 \(\mathbb{R}^n\) 中的一个正交归一列表，并考虑矩阵

\[ M = \sum_{i=1}^m \mathbf{q}_i \mathbf{q}_i^T. \]

a) 假设 \(m = n\)。通过计算 \(\mathbf{e}_i^T M \mathbf{e}_j\) 对于所有 \(i,j\) 来证明 \(M = I_{n \times n}\)。

b) 假设 \(m \leq n\)。\(M\) 的几何解释是什么？基于你的答案，给出 (a) 的第二个证明。

\(\lhd\)

**4.35** 设 \(A \in \mathbb{R}^{n \times m}\) 是一个具有紧奇异值分解 \(A = U \Sigma V^T\) 的矩阵。设 \(A^+ = V \Sigma^{-1} U^T\)（称为伪逆）。证明 \(A A^+ A = A\) 和 \(A^+ A A^+ = A^+\)。\(\lhd\)

**4.36** 设 \(A \in \mathbb{R}^{n \times n}\) 是一个具有正交特征向量 \(\mathbf{q}_1,\ldots,\mathbf{q}_n\) 和相应的特征值 \(\lambda_1 \geq \cdots \geq \lambda_n\) 的对称矩阵。

a) 计算 \(A^T A\) 的谱分解。

b) 使用 (a) 来用 \(A\) 的两个特征值来计算 \(\|A\|_2\)。\(\lhd\)

## 4.7.1\. 热身练习表#

*(在 Claude、Gemini 和 ChatGPT 的帮助下)*

**第 4.2 节**

**E4.2.1** 计算矩阵 \(A = \begin{pmatrix} 1 & 2 & 3\\ 0 & 1 & 1\\ 1 & 3 & 4 \end{pmatrix}\) 的秩。

**E4.2.2** 设 \(A = \begin{pmatrix} 1 & 2\\ 3 & 4 \end{pmatrix}\) 和 \(B = \begin{pmatrix} 1 & 0\\ 0 & 1 \end{pmatrix}\)。计算 \(\mathrm{rk}(A)\)，\(\mathrm{rk}(B)\) 和 \(\mathrm{rk}(A+B)\)。

**E4.2.3** 找到矩阵 \(A = \begin{pmatrix} 3 & 1\\ 1 & 3 \end{pmatrix}\) 的特征值和对应的特征向量。

**E4.2.4** 验证 E4.2.3 中的特征向量是正交的。

**E4.2.5** 写出 E4.2.3 中矩阵 \(A\) 的谱分解。

**E4.2.6** 确定矩阵 \(A = \begin{pmatrix} 2 & -1\\ -1 & 2 \end{pmatrix}\) 是正定矩阵、正半定矩阵还是都不是。

**E4.2.7** 对于 \(\mathbf{u} = (1, 2, 3)\) 和 \(\mathbf{v} = (4, 5)\)，计算外积 \(\mathbf{u}\mathbf{v}^T\)。

**E4.2.8** 将矩阵 \(A = \begin{pmatrix} 1 & 2 & 3\\ 2 & 4 & 6\\ 3 & 6 & 9 \end{pmatrix}\) 写成秩为 1 的矩阵之和。

**E4.2.9** 设 \(A = \begin{pmatrix} 1 & 2 \\ 3 & 6 \end{pmatrix}\)。\(A\) 的秩是多少？

**E4.2.10** 设 \(\mathbf{u} = \begin{pmatrix} 1 \\ 2 \end{pmatrix}\) 和 \(\mathbf{v} = \begin{pmatrix} 3 \\ 4 \end{pmatrix}\)。计算外积 \(\mathbf{u} \mathbf{v}^T\)。

**E4.2.11** 设 \(A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}\)。找到 \(A\) 的特征值和特征向量。

**E4.2.12** 设 \(A = \begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix}\)。验证 \(\mathbf{v} = \begin{pmatrix} 1 \\ 1 \end{pmatrix}\) 是 \(A\) 的一个特征向量，并找到相应的特征值。

**E4.2.13** 设 \(A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}\)。找到 \(A\) 的列空间的基。

**E4.2.14** 设 \(A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}\)。找到 \(A\) 的零空间的基。

**E4.2.15** 设 \(A = \begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix}\)。\(A\) 是正半定矩阵吗？

**E4.2.16** 确定函数 \(f(x, y) = x² + y² + xy\) 是否是凸函数。

**E4.2.17** 函数 \(f(x, y) = x² - y²\) 是凸函数吗？

**E4.2.18** 检查函数 \(f(x, y) = e^{x+y}\) 的凸性。

**E4.2.19** 检查函数 \(f(x, y) = \log(x² + y² + 1)\) 的凸性。

**E4.2.20** 函数 \(f(x, y) = xy\) 是凸函数吗？

**第 4.3 节**

**E4.3.1** 设 \(\boldsymbol{\alpha}_1 = (1, 2)\) 和 \(\boldsymbol{\alpha}_2 = (-2, 1)\)。找到 \(\mathbb{R}²\) 中最大化 \(\|A\mathbf{w}_1\|²\) 的单位向量 \(\mathbf{w}_1\)，其中 \(A\) 是行向量 \(\boldsymbol{\alpha}_1^T\) 和 \(\boldsymbol{\alpha}_2^T\) 的矩阵。

**E4.3.2** 设 \(\mathbf{u} = (1/\sqrt{2}, 1/\sqrt{2})\) 和 \(\mathbf{v} = (1/\sqrt{2}, -1/\sqrt{2})\)。计算外积 \(\mathbf{u}\mathbf{v}^T\)。

**E4.3.3** 设 \(A = \begin{pmatrix} 3 & 0 \\ 0 & 2 \end{pmatrix}\)。求 \(A\) 的奇异值分解。

**E4.3.4** 设 \(A = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}\)。求 \(A\) 的紧凑奇异值分解。

**E4.3.5** 设 \(A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}\)。求 \(A^TA\) 的特征值和特征向量。

**E4.3.6** 设 \(A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}\) 如 E4.3.5 所示。求 \(AA^T\) 的特征值和特征向量。

**E4.3.7** 设 \(A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}\) 如 E4.3.5 所示。求 \(A\) 的紧凑奇异值分解。

**E4.3.8** 设 \(A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}\) 如 E4.3.5 所示。求 \(A\) 的完整奇异值分解。

**E4.3.9** 设 \(A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}\) 如 E4.3.5 所示。找到 \(A\) 的四个基本子空间中的每一个的正交基。

**E4.3.10** 给定数据点 \(\boldsymbol{\alpha}_1 = (1,2)\)，\(\boldsymbol{\alpha}_2 = (2,1)\)，\(\boldsymbol{\alpha}_3 = (-1,1)\)，和 \(\boldsymbol{\alpha}_4 = (3,-1)\)，计算矩阵 \(A\) 和矩阵 \(A^TA\) 以及 \(AA^T\)。

**E4.3.11** 从 E4.3.10 中求矩阵 \(A^TA\) 的特征值和特征向量。

**E4.3.12** 从 E4.3.10 中计算矩阵 \(A\) 的紧凑奇异值分解。

**E4.3.13** 从 E4.3.12 中计算矩阵 \(U_1 \Sigma_1 V_1^T\) 并验证它等于 E4.3.10 中的 \(A\)。

**E4.3.14** 基于 E4.3.12，验证关系 \(A^T \mathbf{u}_i = \sigma_i \mathbf{v}_i\) 对于 \(i=1,2\) 成立。

**E4.3.15** 基于 E4.3.12，验证关系 \(A^TA\mathbf{v}_i = \sigma_i² \mathbf{v}_i\) 和 \(AA^T\mathbf{u}_i = \sigma_i² \mathbf{u}_i\) 对于 \(i=1,2\) 成立。

**E4.3.16** 基于 E4.3.12，找到维度为 \(k=2\) 的最佳逼近子空间，并计算到该子空间的平方距离之和。

**E4.3.17** 基于 E4.3.12，找到维度为 \(k=1\) 的最佳逼近子空间，并计算到该子空间的平方距离之和。

**E4.3.18** 基于 E4.3.17，计算通过截断奇异值分解得到的 \(k=1\) 的矩阵 \(Z\)。

**第 4.4 节**

**E4.4.1** 设 \(A = \begin{pmatrix} 3 & 0 \\ 0 & 1 \end{pmatrix}\)。计算 \(A²\)、\(A³\) 和 \(A⁴\)。你观察到了什么模式？

**E4.4.2** 设 \(B = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}\)。计算 \(B²\) 和 \(B³\)。如果你继续计算 \(B\) 的高次幂，你期望会发生什么？

**E4.4.3** 给定对称矩阵 \(A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}\) 和向量 \(\mathbf{x} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}\)，计算 \(A^k \mathbf{x}\) 对于 \(k = 1, 2, 3\) 和 \(\frac{A^k \mathbf{x}}{\|A^k \mathbf{x}\|}\)。

**E4.4.4** 给定对称矩阵 \(A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}\) 和向量 \(\mathbf{x} = \begin{pmatrix} 0 \\ 1 \end{pmatrix}\)，计算 \(A^k \mathbf{x}\) 对于 \(k = 1, 2, 3\) 和 \(\frac{A^k \mathbf{x}}{\|A^k \mathbf{x}\|}\)。

**E4.4.5** 给定矩阵 \(A = \begin{pmatrix} 2 & 0 \\ 0 & 1 \end{pmatrix}\) 和向量 \(\mathbf{x} = \begin{pmatrix} 1 \\ 1 \end{pmatrix}\)，计算 \(A^k \mathbf{x}\) 对于 \(k = 1, 2, 3\)。

**E4.4.6** 令 \(A = \begin{pmatrix} 4 & 1 \\ 1 & 4 \end{pmatrix}\)。找到 \(A\) 的特征值和特征向量。

**E4.4.7** 令 \(A\) 如 E4.4.6 所示。使用 \(A\) 的谱分解计算 \(A²\) 和 \(A³\)。

**E4.4.8** 令 \(A\) 如 E4.4.6 所示。令 \(\mathbf{x} = \begin{pmatrix} 2 \\ 1 \end{pmatrix}\)。计算 \(A² \mathbf{x}\) 和 \(A³ \mathbf{x}\)。随着幂的增加，这些向量的方向有什么特点？

**E4.4.9** 令 \(A = \begin{pmatrix} 1 & 2 \\ 0 & 1 \end{pmatrix}\)。计算 \(A^TA\)。通过计算其特征值来确定 \(A^TA\) 是否是正半定矩阵。

**第 4.5 节**

**E4.5.1** 给定一个载荷向量 \(\boldsymbol{\phi}_1 = \left(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}\right)\)，验证它满足单位范数约束。

**E4.5.2** 给定两个载荷向量 \(\boldsymbol{\phi}_1 = \left(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}\right)\) 和 \(\boldsymbol{\phi}_2 = \left(\frac{1}{\sqrt{2}}, -\frac{1}{\sqrt{2}}\right)\)，验证它们是正交的。

**E4.5.3** 给定一个中心化数据矩阵 \(\tilde{X} = \begin{pmatrix} -1 & 1 \\ 1 & -1 \end{pmatrix}\) 和一个载荷向量 \(\boldsymbol{\phi}_1 = \left(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}\right)\)，计算第一个主成分得分 \(t_{i1}\)。

**E4.5.4** 给定一个中心化数据矩阵 \(\tilde{X} = \begin{pmatrix} -1 & 1 \\ 1 & -1 \end{pmatrix}\) 和两个载荷向量 \(\boldsymbol{\phi}_1 = \left(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}\right)\) 和 \(\boldsymbol{\phi}_2 = \left(\frac{1}{\sqrt{2}}, -\frac{1}{\sqrt{2}}\right)\)，计算第一和第二个主成分得分 \(t_{i1}\) 和 \(t_{i2}\)。

**E4.5.5** 给定 E4.5.4 中的第一和第二个主成分得分，验证它们是不相关的。

**E4.5.6** 给定数据矩阵 \(X = \begin{pmatrix} 1 & 3 \\ -1 & -3\end{pmatrix}\)，计算中心化数据矩阵 \(\tilde{X}\)。

**E4.5.7** 给定数据矩阵

\[\begin{split} X = \begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{pmatrix}, \end{split}\]

计算均值中心化数据矩阵。

**E4.5.8** 给定 E4.5.7 中中心化数据矩阵的奇异值分解，提取第一个主成分载荷向量 \(\boldsymbol{\phi}_1\)。

**E4.5.9** 给定中心化的数据矩阵 \(\tilde{X}\) 和来自 E4.5.7 和 E4.5.8 的第一个主成分载荷向量 \(\boldsymbol{\phi}_1\)，计算第一个主成分得分 \(t_{i1}\)。

**E4.5.10** 给定一个具有中心化列的数据矩阵 \(X\)，证明 \(X\) 的样本协方差矩阵可以表示为 \(\frac{1}{n-1} X^T X\)。

**E4.5.11** 给定第一个主成分的载荷向量 \(\boldsymbol{\phi}_1 = (0.8, 0.6)\)，找到第二个主成分的载荷向量 \(\boldsymbol{\phi}_2\)。

**E4.5.12** 给定数据点 \(\mathbf{x}_1 = (1, 0)\)，\(\mathbf{x}_2 = (0, 1)\)，和 \(\mathbf{x}_3 = (-1, 0)\)，计算第一个主成分向量。

**第 4.6 节**

**E4.6.1** 计算矩阵 \(A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{pmatrix}\) 的 Frobenius 范数。

**E4.6.2** 计算 \(A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}\) 的奇异值分解。

**E4.6.3** 设 \(A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}\)。计算 Frobenius 范数 \(\|A\|_F\)。

**E4.6.4** 设 \(A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}\)。计算诱导的 2-范数 \(\|A\|_2\)。

**E4.6.5** 设 \(A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}\) 和 \(B = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}\)。计算 \(\|A - B\|_F\)。

**E4.6.6** 设 \(A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}\) 和 \(A_1 = \sigma_1 \mathbf{u}_1 \mathbf{v}_1^T\) 是 \(A\) 的秩-1 截断奇异值分解。计算 \(\|A - A_1\|_F\)。

**E4.6.7** 设 \(A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}\) 和 \(A_1 = \sigma_1 \mathbf{u}_1 \mathbf{v}_1^T\) 是 \(A\) 的秩-1 截断奇异值分解。计算 \(\|A - A_1\|_2\)。

**E4.6.8** 设 \(A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}\)，向量 \(\mathbf{b} = \begin{pmatrix} 5 \\ 10 \end{pmatrix}\)，以及 \(\lambda = 1\)。计算岭回归解 \(\mathbf{x}^{**}\) 到 \(\min_{\mathbf{x} \in \mathbb{R}²} \|A \mathbf{x} - \mathbf{b}\|_2² + \lambda \|\mathbf{x}\|_2²\)。

**E4.6.9** 设 \(A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}\)。计算 \(\|A^{-1}\|_2\)。

**E4.6.10** 计算以下矩阵 \(A = \begin{pmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{pmatrix}\)，向量 \(b = \begin{pmatrix} 1 \\ 2 \\ 2 \end{pmatrix}\)，以及 \(\lambda = 1\) 的岭回归解。

## 4.7.2\. 问题#

**4.1** 设 \(Q \in \mathbb{R}^{n \times n}\) 是一个正交矩阵。使用 *通过谱分解定理的奇异值分解* 来计算 \(Q\) 的奇异值分解。在这种情况下，紧凑型奇异值分解和完整奇异值分解之间有区别吗？\(\lhd\)

**4.2** 设 \(W \subseteq \mathbb{R}^m\) 是一个超平面。证明存在一个单位向量 \(\mathbf{z} \in \mathbb{R}^m\)，使得

\[ \mathbf{w} \in W \iff \langle \mathbf{z}, \mathbf{w} \rangle = 0. \]

\(\lhd\)

**4.3** 构造一个矩阵 \(A \in \mathbb{R}^{n \times n}\)，使得最大化问题存在多个解。

\[ \mathbf{v}_1 \in \arg\max\{\|A \mathbf{v}\|:\|\mathbf{v}\| = 1\}. \]

\(\lhd\)

**4.4** 证明正定矩阵的 *正定性特征* 的一个类似命题。 \(\lhd\)

**4.5** (改编自 [Sol]) 证明 \(\|A\|_2 = \|\Sigma\|_2\)，其中 \(A = U \Sigma V^T\) 是 A 的奇异值分解。 \(\lhd\)

**4.6** 设 \(A\), \(U \Sigma V^T\), \(B\) 如 *幂迭代引理* 中所述。进一步假设 \(\sigma_2 > \sigma_3\) 且 \(\mathbf{y} \in \mathbb{R}^m\) 同时满足 \(\langle \mathbf{v}_1, \mathbf{y} \rangle = 0\) 和 \(\langle \mathbf{v}_2, \mathbf{y} \rangle > 0\)。证明当 \(k \to +\infty\) 时，\(\frac{B^{k} \mathbf{y}}{\|B^{k} \mathbf{y}\|} \to \mathbf{v}_2\)。你将如何找到这样的 \(\mathbf{y}\)? \(\lhd\)

**4.7** 设 \(A \in \mathbb{R}^{n \times m}\)。使用 *柯西-施瓦茨不等式* 证明

\[ \|A\|_2 = \max \left\{ \mathbf{x}^T A \mathbf{y} \,:\, \|\mathbf{x}\| = \|\mathbf{y}\| = 1 \right\}. \]

\(\lhd\)

**4.8** 设 \(A \in \mathbb{R}^{n \times m}\)。

a) 证明 \(\|A\|_F² = \sum_{j=1}^m \|A \mathbf{e}_j\|²\)。

b) 使用 (a) 和 *柯西-施瓦茨不等式* 证明 \(\|A\|_2 \leq \|A\|_F\)。

c) 给出一个例子，使得 \(\|A\|_F = \sqrt{n} \|A\|_2\)。 \(\lhd\)

**4.9** 使用 *柯西-施瓦茨不等式* 证明，对于任何 \(A, B\)，当 \(AB\) 有定义时，有

\[ \|A B \|_F \leq \|A\|_F \|B\|_F. \]

\(\lhd\)

**4.10** (*注意*: 指的是在线补充材料。) 设 \(A \in \mathbb{R}^{n \times n}\) 是非奇异的，其奇异值分解为 \(A = U \Sigma V^T\)，其中奇异值满足 \(\sigma_1 \geq \cdots \geq \sigma_n > 0\)。证明

\[ \min_{\mathbf{x} \neq \mathbf{0}} \frac{\|A \mathbf{x}\|}{\|\mathbf{x}\|} = \min_{\mathbf{y} \neq \mathbf{0}} \frac{\|\mathbf{y}\|}{\|A^{-1}\mathbf{y}\|} = \sigma_n = 1/\|A^+\|_2. \]

\(\lhd\)

**4.11** 设 \(X \in \mathbb{R}^{n \times d}\) 是一个矩阵，其行分别为 \(\mathbf{x}_1^T,\ldots,\mathbf{x}_n^T\)。将以下和写成以 \(X\) 为矩阵形式：

\[ \frac{1}{n} \sum_{i=1}^n \mathbf{x}_i \mathbf{x}_i^T. \]

证明你的答案。 \(\lhd\)

**4.12** 设 \(A \in \mathbb{R}^{d \times d}\) 是一个对称矩阵。证明 \(A \preceq M I_{d \times d}\) 当且仅当 \(A\) 的特征值不超过 \(M\)。类似地，\(m I_{d \times d} \preceq A\) 当且仅当 \(A\) 的特征值至少为 \(m\)。[*提示:* 观察到 \(A\) 的特征向量也是单位矩阵 \(I_{d \times d}\) 的特征向量。] \(\lhd\)

**4.13** 证明 *正半定情况下的幂迭代引理*. \(\lhd\)

**4.14** 回忆一下，矩阵 \(A = (a_{i,j})_{i,j} \in \mathbb{R}^n\) 的迹 \(\mathrm{tr}(A)\) 是其对角线元素的和，即 \(\mathrm{tr}(A) = \sum_{i=1}^n a_{i,i}\)。

a) 证明，对于任何 \(A \in \mathbb{R}^{n \times m}\) 和 \(B \in \mathbb{R}^{m \times n}\)，有 \(\mathrm{tr}(AB) = \mathrm{tr}(BA)\)。

b) 使用 a) 和 *柯西-施瓦茨不等式* 证明，如果对称矩阵 \(A\) 的谱分解为 \(\sum_{i=1}^n \lambda_i \mathbf{q}_i \mathbf{q}_i^T\)，那么

\[ \mathrm{tr}(A) = \sum_{i=1}^n \lambda_i. \]

\(\lhd\)

**4.15** （改编自 [Sol]）假设 \(A \in \mathbb{R}^{m \times n}\) 和 \(B \in \mathbb{R}^{n \times m}\)。证明 \(\|A\|²_F = \mathrm{tr}(A^T A)\) 和 \(\mathrm{tr}(A B) = \mathrm{tr}(B A)\)，其中回忆一下，矩阵 \(C\) 的迹，记为 \(\mathrm{tr}(C)\)，是 \(C\) 对角元素的和。\(\lhd\)

**4.16** （改编自 [Str]）设 \(A \in \mathbb{R}^{n \times k}\) 和 \(B \in \mathbb{R}^{k \times m}\)。证明 \(AB\) 的零空间包含 \(B\) 的零空间。\(\lhd\)

**4.17** （改编自 [Str]）设 \(A \in \mathbb{R}^{n \times m}\)。证明 \(A^T A\) 与 \(A\) 有相同的零空间。\(\lhd\)

**4.18** （改编自 [Str]）设 \(A \in \mathbb{R}^{m \times r}\) 和 \(B \in \mathbb{R}^{r \times n}\)，两者均为秩 \(r\)。

a) 证明 \(A^T A\)、\(B B^T\) 和 \(A^T A B B^T\) 都是可逆的。

b) 证明 \(r = \mathrm{rk}(A^T A B B^T) \leq \mathrm{rk}(AB)\)。

c) 推论出 \(\mathrm{rk}(AB) = r\)。

\(\lhd\)

**4.19** 对于任何 \(n \geq 1\)，给出一个矩阵 \(A \in \mathbb{R}^{n \times n}\) 的例子，使得 \(A \neq I_{n \times n}\) 且 \(A² = I_{n \times n}\)。\(\lhd\)

**4.20** 设 \(A \in \mathbb{R}^{n \times k}\) 和 \(B \in \mathbb{R}^{r \times m}\)，且 \(D \in \mathbb{R}^{k \times r}\) 是一个对角矩阵。假设 \(k < r\)。计算 \(AD\) 的列和 \(DB\) 的行。\(\lhd\)

**4.21** 计算 \(A = [-1]\) 的奇异值分解。\(\lhd\)

**4.22** 在这个练习中，我们展示了矩阵可以有许多不同的奇异值分解（SVD）。设 \(Q, W \in \mathbb{R}^{n \times n}\) 是正交矩阵。给出 \(Q\) 的四个不同的 SVD，一个其中 \(V = I\)，一个其中 \(V = Q\)，一个其中 \(V = Q^T\)，以及一个其中 \(V = W\)。确保检查 SVD 的所有要求。奇异值会改变吗？\(\lhd\)

**4.23** （改编自 [Sol]）证明向矩阵中添加一行不能减少其最大的奇异值。\(\lhd\)

**4.24** 设 \(\mathbf{v}_1,\ldots,\mathbf{v}_n \in \mathbb{R}^n\) 是一个正交基。固定 \(1 \leq k < n\)。设 \(Q_1\) 是列向量为 \(\mathbf{v}_1,\ldots,\mathbf{v}_k\) 的矩阵，设 \(Q_2\) 是列向量为 \(\mathbf{v}_{k+1},\ldots,\mathbf{v}_n\) 的矩阵。证明

\[ Q_1 Q_1^T + Q_2 Q_2^T = I_{n \times n}. \]

[*提示：将两边乘以 \(\mathbf{e}_i\).] \(\lhd\)

**4.25** （*注意：指在线补充材料。*）设 \(Q \in \mathbb{R}^{n \times n}\) 是一个正交矩阵。计算其条件数

\[ \kappa_2(Q) = \|Q\|_2 \|Q^{-1}\|_2. \]

\(\lhd\)

**4.26** 证明奇异值分解关系引理。\(\lhd\)

**4.27** 设 \(A = \sum_{j=1}^r \sigma_j \mathbf{u}_j \mathbf{v}_j^T\) 是 \(A \in \mathbb{R}^{n \times m}\) 的奇异值分解，其中 \(\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0\)。

定义

\[ B = A - \sigma_1 \mathbf{u}_1 \mathbf{v}_1^T. \]

证明

\[ \mathbf{v}_2 \in \arg\max\{\|B \mathbf{v}\|:\|\mathbf{v}\| = 1\}. \]

\(\lhd\)

**4.28** （改编自 [Sol]）\(A \in \mathbb{R}^{n \times n}\) 的稳定秩定义为

\[ \mathrm{StableRk}(A) = \frac{\|A\|_F²}{\|A\|_2²}. \]

a) 证明，如果 \(A\) 的所有列都是相同的非零向量 \(\mathbf{v} \in \mathbb{R}^n\)，则 \(\mathrm{StableRk}(A) = 1\)。

b) 证明当 \(A\) 的列是正交归一时，则 \(\mathrm{StableRk}(A) = n\)。

c) \(\mathrm{StableRk}(A)\) 是否总是整数？证明你的答案。

d) 更一般地，证明

\[ 1 \leq \mathrm{StableRk}(A) \leq n. \]

e) 证明在一般情况下，以下等式成立

\[ \mathrm{StableRk}(A) \leq \mathrm{Rk}(A). \]

[*提示：使用矩阵范数和奇异值引理。]*\(\lhd\)

**4.29** 设 \(A \in \mathbb{R}^{n \times n}\) 是一个具有满奇异值分解 \(A = U \Sigma V^T\) 的方阵。

a) 证明以下公式的合理性

\[ A = (U V^T) (V \Sigma V^T). \]

b) 设

\[ Q = U V^T, \qquad S = V \Sigma V^T. \]

证明 \(Q\) 是正交的，且 \(S\) 是正半定的。形式为 \(A = Q S\) 的分解称为极分解。\(\lhd\)

**4.30** 设 \(A \in \mathbb{R}^{n \times m}\) 是一个具有满奇异值分解 \(A = U \Sigma V^T\) 的矩阵，其中奇异值满足 \(\sigma_1 \geq \cdots \geq \sigma_m \geq 0\)。定义 \(m\) 维单位球 \(\mathbb{S}^{m-1} = \{\mathbf{x} \in \mathbb{R}^m\,:\,\|\mathbf{x}\| = 1\}\)。证明

\[ \sigma_m² = \min_{\mathbf{x} \in \mathbb{S}^{m-1}} \|A \mathbf{x}\|²\. \]

\(\lhd\)

**4.31** 设 \(A \in \mathbb{R}^{n \times m}\) 是一个矩阵，其列向量为 \(\mathbf{a}_1,\ldots,\mathbf{a}_m\)。证明对于所有 \(i\)

\[ \|\mathbf{a}_i\|_2 \leq \|A\|_2. \]

\(\lhd\)

**4.32** 设 \((U_i, V_i)\)，\(i = 1, \ldots, n\) 是在 \(\mathbb{R}²\) 中取值的独立同分布随机向量。假设 \(\mathbb{E}[U_1²], \mathbb{E}[V_1²] < +\infty\)。

a) 证明

\[ \mathbb{E}\left[\frac{1}{n} \sum_{i=1}^n U_i \right] = \mathbb{E}[U_1]. \]

b) 证明

\[ \mathbb{E}\left[\frac{1}{n-1} \sum_{i=1}^n \left(U_i - \frac{1}{n} \sum_{j=1}^n U_j\right)²\right] = \mathrm{Var}[U_1]. \]

c) 证明

\[ \mathbb{E}\left[\frac{1}{n-1} \sum_{i=1}^n \left(U_i - \frac{1}{n} \sum_{j=1}^n U_j\right) \left(V_i - \frac{1}{n} \sum_{k=1}^n V_k\right) \right] = \mathrm{Cov}[U_1, V_1]. \]

在文字上，上述每个方程的左侧（更精确地说，是在期望内部）是右侧的一个所谓的无偏估计量。这些估计量分别被称为样本均值、样本方差和样本协方差。\(\lhd\)

**4.33** 给出数据矩阵 \(A \in \mathbb{R}^{n \times m}\) 和一个列向量正交的矩阵 \(W\) 的最佳 \(k\) 维逼近子空间的一个等价矩阵表示。[提示：使用 Frobenius 范数。]\(\lhd\)

**4.34** 设 \(\mathbf{q}_1,\ldots,\mathbf{q}_m\) 是 \(\mathbb{R}^n\) 中的一个正交列表，并考虑矩阵

\[ M = \sum_{i=1}^m \mathbf{q}_i \mathbf{q}_i^T. \]

a) 假设\(m = n\)。通过计算所有\(i,j\)的\(\mathbf{e}_i^T M \mathbf{e}_j\)来证明\(M = I_{n \times n}\)。

b) 假设\(m \leq n\)。\(M\)的几何解释是什么？根据你的答案，给出(a)的第二个证明。

\(\lhd\)

**4.35** 设\(A \in \mathbb{R}^{n \times m}\)为一个具有紧奇异值分解\(A = U \Sigma V^T\)的矩阵。设\(A^+ = V \Sigma^{-1} U^T\)（称为伪逆）。证明\(A A^+ A = A\)和\(A^+ A A^+ = A^+\)。\(\lhd\)

**4.36** 设\(A \in \mathbb{R}^{n \times n}\)为一个具有正交归一特征向量\(\mathbf{q}_1,\ldots,\mathbf{q}_n\)和相应的特征值\(\lambda_1 \geq \cdots \geq \lambda_n\)的对称矩阵。

a) 计算\(A^T A\)的谱分解。

b) 使用(a)中的方法，用\(A\)的两个特征值来计算\(\|A\|_2\)。\(\lhd\)
