# 8.6\. 练习题#

> 原文：[`mmids-textbook.github.io/chap08_nn/exercises/roch-mmids-nn-exercises.html`](https://mmids-textbook.github.io/chap08_nn/exercises/roch-mmids-nn-exercises.html)

## 8.6.1\. 预习工作表#

*(with help from Claude, Gemini, and ChatGPT)*

**第 8.2 节**

**E8.2.1** 设 \(A = \begin{pmatrix} 2 & 1 \\ 0 & -1 \end{pmatrix}\)。计算向量 \(\text{vec}(A)\)。

**E8.2.2** 设 \(\mathbf{a} = (2, -1, 3)\) 和 \(\mathbf{b} = (1, 0, -2)\)。计算哈达玛积 \(a \odot b\)。

**E8.2.3** 设 \(A = \begin{pmatrix} 1 & 2 \\ -1 & 0 \end{pmatrix}\) 和 \(B = \begin{pmatrix} 3 & -1 \\ 2 & 1 \end{pmatrix}\)。计算克罗内克积 \(A \otimes B\)。

**E8.2.4** 设 \(A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}\) 和 \(B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}\)。计算哈达玛积 \(A \odot B\)。

**E8.2.5** 设 \(A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}\) 和 \(B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}\)。计算克罗内克积 \(A \otimes B\)。

**E8.2.6** 设 \(\mathbf{f}(x, y) = (x²y, \sin(xy), e^{x+y})\)。计算 \(\mathbf{f}\) 在点 \((1, \frac{\pi}{2})\) 处的雅可比矩阵。

**E8.2.7** 设 \(\mathbf{f}(x, y) = (x² + y², xy)\) 和 \(\mathbf{g}(u, v) = (uv, u + v)\)。计算 \(\mathbf{g} \circ \mathbf{f}\) 在点 \((1, 2)\) 处的雅可比矩阵。

**E8.2.8** 设 \(\mathbf{a} = (1, 2, 3)\)，\(\mathbf{b} = (4, 5, 6)\)，和 \(\mathbf{c} = (7, 8, 9)\)。计算 \(\mathbf{a}^T(\mathbf{b} \odot \mathbf{c})\) 和 \(\mathbf{1}^T(\mathbf{a} \odot \mathbf{b} \odot \mathbf{c})\) 并验证它们相等。

**E8.2.9** 设 \(A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}\) 和 \(B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}\)。计算 \((A \otimes B)^T\) 和 \(A^T \otimes B^T\) 并验证它们相等。

**E8.2.10** 设 \(\mathbf{g}(x, y) = (x²y, x + y)\)。计算雅可比矩阵 \(J_{\mathbf{g}}(x, y)\)。

**E8.2.11** 设 \(f(x, y, z) = x² + y² + z²\)。计算 \(f\) 在点 \((1, 2, 3)\) 处的梯度。

**E8.2.12** 设 \(f(x) = 2x³ - x\) 和 \(g(y) = y² + 1\)。计算复合函数 \(g \circ f\) 在 \(x = 1\) 处的雅可比矩阵。

**E8.2.13** 设 \(f(x, y) = xy\) 和 \(\mathbf{g}(x, y) = (x², y²)\)。计算 \(f \circ \mathbf{g}\) 在点 \((1, 2)\) 处的雅可比矩阵。

**E8.2.14** 设 \(A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}\) 和 \(\mathbf{b} = \begin{pmatrix} 5 \\ 6 \end{pmatrix}\)。定义函数 \(\mathbf{f}(\mathbf{x}) = A \mathbf{x} + \mathbf{b}\)。计算 \(\mathbf{f}\) 在任意点 \(\mathbf{x} \in \mathbb{R}²\) 处的雅可比矩阵。

**E8.2.15** 设 \(f(x) = \sin(x)\)。定义函数 \(\mathbf{g}(x, y, z) = (f(x), f(y), f(z))\)。计算 \(\mathbf{g}\) 在点 \((\frac{\pi}{2}, \frac{\pi}{4}, \frac{\pi}{6})\) 处的雅可比矩阵。

**E8.2.16** 使用 PyTorch 计算 \(f(x) = x³ - 4x\) 在 \(x = 2\) 处的梯度。提供 PyTorch 代码和结果。

**第 8.3 节**

**E8.3.1** 设 \(A = \begin{pmatrix} 1 & -1 \\ 0 & 2 \end{pmatrix}\) 和 \(B = \begin{pmatrix} 2 & 1 \\ -1 & 3 \end{pmatrix}\)。计算 \(AB\) 需要进行多少次基本运算（加法和乘法）？

**E8.3.2** 设 \(A = \begin{pmatrix} 1 & -1 \\ 0 & 2 \end{pmatrix}\) 和 \(\mathbf{v} = (2, -1)\)。计算 \(A\mathbf{v}\) 需要进行多少次基本运算（加法和乘法）？

**E8.3.3** 设 \(\ell(\hat{\mathbf{y}}) = \|\hat{\mathbf{y}}\|²\) 其中 \(\hat{\mathbf{y}} = (\hat{y}_1, \hat{y}_2)\)。计算 \(J_{\ell}(\hat{\mathbf{y}})\)。

**E8.3.4** 设 \(\mathbf{g}_0(\mathbf{z}_0) = \begin{pmatrix} 1 & 2 \\ -1 & 0 \end{pmatrix} \mathbf{z}_0\) 和 \(\ell(\hat{\mathbf{y}}) = \|\hat{\mathbf{y}}\|²\)。计算 \(f(\mathbf{x}) = \ell(\mathbf{g}_0(\mathbf{x}))\) 在 \(\mathbf{x} = (1, -1)\) 处的梯度 \(\nabla f(\mathbf{x})\)。

**E8.3.5** 设 \(\mathbf{g}_0\) 和 \(\ell\) 如 E8.3.4 所示。设 \(\mathbf{g}_1(\mathbf{z}_1) = \begin{pmatrix} -1 & 0 \\ 1 & 1 \end{pmatrix} \mathbf{z}_1\)。使用反向传播法计算 \(f(\mathbf{x}) = \ell(\mathbf{g}_1(\mathbf{g}_0(\mathbf{x})))\) 在 \(\mathbf{x} = (1, -1)\) 处的梯度 \(\nabla f(\mathbf{x})\)。

**E8.3.6** 设 \(\mathbf{g}_0(\mathbf{x}, \mathbf{w}_0) = \mathcal{W}_0 \mathbf{x}\) 其中 \(\mathcal{W}_0 = \begin{pmatrix} w_0 & w_1 \\ w_2 & w_3 \end{pmatrix}\) 和 \(\mathbf{w}_0 = (w_0, w_1, w_2, w_3)\)。设 \(\mathbf{x} = (-1, 1)\) 为固定值。仅通过直接计算必要的偏导数（即，不使用文本中的公式）来计算关于 \(\mathbf{w}_0\) 的雅可比矩阵 \(J_{\mathbf{g}_0}(\mathbf{x}, \mathbf{w}_0)\)，然后与文本中的公式进行比较。

**E8.3.7** 设 \(g_1(\mathbf{z}_1, \mathbf{w}_1) = \mathcal{W}_1 \mathbf{z}_1\) 其中 \(\mathcal{W}_1 = \begin{pmatrix} w_4 & w_5\end{pmatrix}\) 和 \(\mathbf{w}_1 = (w_4, w_5)\)。直接计算必要的偏导数（即，不使用文本中的公式），分别对 \(\mathbf{z}_1\) 和 \(\mathbf{w}_1\) 计算 \(J_{g_1}(\mathbf{z}_1, \mathbf{w}_1)\)，然后与文本中的公式进行比较。

**E8.3.8** 设 \(h(\mathbf{w}) = g_1(\mathbf{g}_0(\mathbf{x}, \mathbf{w}_0), \mathbf{w}_1)\) 其中 \(\mathbf{g}_0\) 和 \(g_1\) 如 E8.3.6 和 E8.3.7 所示，且 \(\mathbf{w} = (\mathbf{w}_0, \mathbf{w}_1) = (w_0, w_1, w_2, w_3, w_4, w_5)\)。设 \(\mathbf{x} = (-1, 1)\) 为固定值。通过直接计算必要的偏导数（即，不使用文本中的公式）来计算 \(J_h(\mathbf{w})\)。

**E8.3.9** 设 \(f(\mathbf{w}) = \ell(g_1(\mathbf{g}_0(\mathbf{x}, \mathbf{w}_0), \mathbf{w}_1))\)，其中 \(\ell(\hat{y}) = \hat{y}²\)，\(\mathbf{g}_0\) 和 \(g_1\) 如 E8.3.6 和 E8.3.7 中所述，且 \(\mathbf{w} = (\mathbf{w}_0, \mathbf{w}_1) = (w_0, w_1, w_2, w_3, w_4, w_5)\)。设 \(\mathbf{x} = (-1, 1)\) 为固定值。通过直接计算必要的偏导数（即，不使用文本中的公式）来计算 \(J_f(\mathbf{w})\)，然后与文本中的公式进行比较。

**第 8.4 节**

**E8.4.1** 给定一个包含 5 个样本的数据集，计算全梯度下降步和批大小为 2 的期望随机梯度下降步。假设单个样本梯度为：\(\nabla f_{\mathbf{x}_1, y_1}(\mathbf{w}) = (1, 2)\)，\(\nabla f_{\mathbf{x}_2, y_2}(\mathbf{w}) = (-1, 1)\)，\(\nabla f_{\mathbf{x}_3, y_3}(\mathbf{w}) = (0, -1)\)，\(\nabla f_{\mathbf{x}_4, y_4}(\mathbf{w}) = (2, 0)\)，和 \(\nabla f_{\mathbf{x}_5, y_5}(\mathbf{w}) = (1, 1)\)。

**E8.4.2** 计算对于 \(\mathbf{z} = (1, -2, 0, 3)\) 的 softmax 函数 \(\boldsymbol{\gamma}(\mathbf{z})\)。

**E8.4.3** 计算概率分布 \(\mathbf{p} = (0.2, 0.3, 0.5)\) 和 \(\mathbf{q} = (0.1, 0.4, 0.5)\) 之间的 Kullback-Leibler 散度。

**E8.4.4** 在单特征线性回归中，单个样本 \((x, y)\) 的损失函数由以下公式给出

\[ \ell(w, b; x, y) = (wx + b - y)². \]

计算梯度 \(\frac{\partial \ell}{\partial w}\) 和 \(\frac{\partial \ell}{\partial b}\)。

**E8.4.5** 假设我们有一个包含三个样本的数据集：\((x_1, y_1) = (2, 3)\)，\((x_2, y_2) = (-1, 0)\)，和 \((x_3, y_3) = (1, 2)\)。我们想要对线性回归进行批大小为 2 的迷你批随机梯度下降。如果第一个随机选择的迷你批是 \(B = \{1, 3\}\)，假设学习率为 \(\alpha = 0.1\)，计算参数 \(w\) 和 \(b\) 的 SGD 更新，假设模型初始化为 \(w = 1\) 和 \(b = 0\)。

**E8.4.6** 对于单样本线性回归问题 \((\mathbf{x}, y) = ((1, 2), 3)\)，计算损失函数 \(f(\mathbf{w}) = (\mathbf{x}^T \mathbf{w} - y)²\) 在 \(\mathbf{w} = (0, 0)\) 处的梯度。

**E8.4.7** 考虑单个样本 \((x, y)\) 的逻辑回归损失函数，其中 \(x \in \mathbb{R}\) 且 \(y \in \{0, 1\}\):

\[ \ell(w; x, y) = -y \log(\sigma(wx)) - (1-y) \log(1 - \sigma(wx)), \]

其中 \(\sigma(z) = \frac{1}{1 + e^{-z}}\) 是 sigmoid 函数。计算关于 \(w\) 的梯度 \(\nabla \ell(w; x, y)\)。

**E8.4.8** 考虑一个有三个类别 (\(K = 3\)) 的多项式逻辑回归问题。给定一个输入向量 \(\mathbf{x} = (1, -1)\) 和一个权重矩阵

\[\begin{split} W = \begin{pmatrix} 1 & 2 \\ -1 & 0 \\ 0 & 1 \end{pmatrix}, \end{split}\]

计算 softmax 输出 \(\boldsymbol{\gamma}(W\mathbf{x})\)。

**E8.4.9** 对于具有单个样本 \((\mathbf{x}, \mathbf{y}) = ((1, 2), (0, 0, 1))\) 和 \(K = 3\) 个类别的多项式逻辑回归问题，计算在 \(W = \begin{pmatrix} 0 & 0 \\ 0 & 0 \\ 0 & 0 \end{pmatrix}\) 处损失函数 \(f(\mathbf{w}) = -\sum_{i=1}^K y_i \log \gamma_i(W\mathbf{x})\) 的梯度。

**E8.4.10** 对于具有两个样本 \((\mathbf{x}_1, y_1) = ((1, 2), 3)\) 和 \((\mathbf{x}_2, y_2) = ((4, -1), 2)\) 的线性回归问题，计算在 \(\mathbf{w} = (0, 0)\) 处的完整梯度。

**E8.4.11** 对于具有两个样本 \((\mathbf{x}_1, \mathbf{y}_1) = ((1, 2), (0, 0, 1))\) 和 \((\mathbf{x}_2, \mathbf{y}_2) = ((4, -1), (1, 0, 0))\) 的多项式逻辑回归问题，以及 \(K = 3\) 个类别，计算在 \(W = \begin{pmatrix} 0 & 0 \\ 0 & 0 \\ 0 & 0 \end{pmatrix}\) 处的完整梯度。

**E8.4.12** 在一个二元分类问题中，逻辑回归模型预测两个样本的概率为 0.8 和 0.3。如果这些样本的真实标签分别是 1 和 0，计算平均交叉熵损失。

**E8.4.13** 在一个有四个类别的多类别分类问题中，一个模型预测一个样本的概率分布为 \((0.1, 0.2, 0.3, 0.4)\)。如果真实标签是第三个类别，那么这个样本的交叉熵损失是多少？

**第 8.5 节**

**E8.5.1** 计算以下 \(t\) 值的 sigmoid 函数 \(\sigma(t)\)：\(1, -1, 2\)。

**E8.5.2** 计算以下 \(t\) 值的 sigmoid 函数的导数 \(\sigma'(t)\)：\(1, -1, 2\)。

**E8.5.3** 给定向量 \(\mathbf{z} = (1, -1, 2)\)，计算 \(\bsigma(\mathbf{z})\) 和 \(\bsigma'(\mathbf{z})\)。

**E8.5.4** 给定矩阵 \(W = \begin{pmatrix} 1 & -1 \\ 2 & 0 \end{pmatrix}\) 和向量 \(\mathbf{x} = (1, 2)\)，计算 \(\bsigma(W\mathbf{x})\)。

**E8.5.5** 给定矩阵 \(W = \begin{pmatrix} 1 & -1 \\ 2 & 0 \end{pmatrix}\) 和向量 \(\mathbf{x} = (1, 2)\)，计算 \(\bsigma(W\mathbf{x})\) 的雅可比矩阵。

**E8.5.6** 给定向量 \(\mathbf{y} = (0, 1)\) 和 \(\mathbf{z} = (0.3, 0.7)\)，计算交叉熵损失 \(H(\mathbf{y}, \mathbf{z})\)。

**E8.5.7** 给定向量 \(\mathbf{y} = (0, 1)\) 和 \(\mathbf{z} = (0.3, 0.7)\)，计算交叉熵损失 \(\nabla H(\mathbf{y}, \mathbf{z})\) 的梯度。

**E8.5.8** 给定向量 \(\mathbf{w} = (1, 2, -1, 0)\) 和 \(\mathbf{z} = (1, 2)\)，计算 \(\mathbb{A}_2[\mathbf{w}]\) 和 \(\mathbb{B}_2[\mathbf{z}]\)。

## 8.6.2\. 问题#

**8.1** 考虑仿射映射

\[ \mathbf{f}(\mathbf{x}) = A \mathbf{x} + \mathbf{b}, \]

其中 \(A \in \mathbb{R}^{m \times d}\) 且 \(\mathbf{b} = (b_1, \ldots, b_m) \in \mathbb{R}^m\)。设 \(S \subseteq \mathbb{R}^m\) 为一个凸集。证明以下集合是凸的：

\[ T = \left\{ \mathbf{x} \in \mathbb{R}^d \,:\, \mathbf{f}(\mathbf{x}) \in S \right\}. \]

\(\lhd\)

**8.2** （改编自[Khu]）考虑向量值函数 \(\mathbf{f} = (f_1, \ldots, f_d) : \mathbb{R}^d \to \mathbb{R}^d\)，定义为

\[ f_i(\mathbf{x}) = x_i³, \]

对于所有 \(\mathbf{x} \in \mathbb{R}^d\) 和所有 \(i = 1, \ldots, d\)。

a) 计算所有 \(\mathbf{x}\) 的雅可比矩阵 \(\mathbf{J}_\mathbf{f}(\mathbf{x})\)。

b) 当 \(\mathbf{J}_\mathbf{f}(\mathbf{x})\) 是可逆的时候？

c) 当 \(\mathbf{J}_\mathbf{f}(\mathbf{x})\) 是正半定的时候？\(\lhd\)

**8.3** 设 \(A = (a_{i,j})_{i,j=1}^n \in \mathbb{R}^{n \times n}\) 为一个对称矩阵。

a) 设 \(\mathbf{v} = (v_1, \ldots, v_n) \in \mathbb{R}^n\) 为 \(A\) 的一个特征向量，其特征值为 \(\lambda\)。设 \(v_{i}\) 为 \(\mathbf{v}\) 中绝对值最大的元素，即 \(i \in \arg\max_j |v_j|\)。定义向量 \(\mathbf{w} = (w_1, \ldots, w_n)\) 如下

\[ w_j = \frac{v_j}{v_{i}}, \qquad j=1, \ldots, n. \]

证明

\[ \lambda - a_{i,i} = \sum_{j \neq i} a_{i,j} w_j. \]

b) 使用(a)来证明，对于 \(A\) 的任意特征值 \(\lambda\)，存在 \(i\) 使得

\[ |\lambda - a_{i,i}| \leq \sum_{j \neq i} |a_{i,j}|. \]

\(\lhd\)

**8.4** 一个对称矩阵 \(A = (a_{i,j})_{i,j=1}^n \in \mathbb{R}^{n \times n}\) 如果其对角线上的元素非负，并且对于所有 \(i\)

\[ a_{i,i} \geq \sum_{j \neq i} |a_{i,j}|, \]

即，每个对角元素都大于或等于其行中其他元素的绝对值之和。使用问题 8.3 来证明这样的矩阵是正半定的。\(\lhd\)

**8.5** 考虑多项式逻辑回归。设

\[ R = I_{K \times K} \otimes \mathbf{x}^T, \]

和

\[ S = \mathrm{diag}\left( \bgamma \left( \mathbf{v} \right) \right) - \bgamma \left( \mathbf{v} \right) \, \bgamma \left( \mathbf{v} \right)^T \]

其中

\[ \mathbf{v} = \bgamma \left( \bfg_0 (\mathbf{x}, \mathbf{w}) \right). \]

a) 证明

\[ \nabla f(\mathbf{w}) = \Gamma \left( \bgamma \left( \bfg_0 (\mathbf{x}, \mathbf{w}) \right) \right) \]

其中

\[ \Gamma (\mathbf{u}) = R (\mathbf{u} - \mathbf{y}). \]

b) 使用**链式法则**来证明

\[ H_f (\mathbf{w}) = R^T S R. \]

c) 使用(b)和**克罗内克积的性质**来证明

\[ H_f (\mathbf{w}) = \left( \mathrm{diag} \left( \bgamma \left( \mathcal{W} \mathbf{x} \right) \right) - \bgamma \left( \mathcal{W} \mathbf{x} \right) \, \bgamma \left( \mathcal{W} \mathbf{x} \right)^T \right) \otimes \mathbf{x} \mathbf{x}^T. \]

\(\lhd\)

**8.6** 考虑多项式逻辑回归。使用问题 8.4 和 8.5 来证明目标函数是凸的。[提示：只需证明 \(S\)（在问题 8.5 中定义）是对角占优的。为什么？]\(\lhd\)

**8.7** 证明**克罗内克积性质引理**的部分 a)。\(\lhd\)

**8.8** 证明**克罗内克积性质引理**的部分 b)。\(\lhd\)

**8.9** 证明**克罗内克积性质引理**的部分 c)和 d)。\(\lhd\)

**8.10** 证明**克罗内克积性质引理**的部分 e)。\(\lhd\)

**8.11** 设 \(A\) 和 \(B\) 分别是大小为 \(n \times n\) 和 \(m \times m\) 的对称矩阵。

a) 证明 \(A \otimes B\) 是对称的。[提示：使用问题 8.10。]

b) 用 \(A\) 和 \(B\) 的特征向量和特征值来计算 \(A \otimes B\) 的特征向量和特征值。[提示：尝试 \(A\) 和 \(B\) 的特征向量的克罗内克积。]

c) 回忆一下，对称矩阵的行列式是其特征值的乘积。证明

\[ \mathrm{det}(A \otimes B) = \mathrm{det}(A)^n \,\mathrm{det}(B)^m. \]

\(\lhd\)

**8.12** 用 \(\mathrm{tr}(A)\) 和 \(\mathrm{tr}(B)\) 来计算 \(\mathrm{tr}(A \otimes B)\)。证明你的答案。 \(\lhd\)

**8.13** a) 证明如果 \(D_1\) 和 \(D_2\) 是方对角矩阵，那么 \(D_1 \otimes D_2\) 也是。 

b) 证明如果 \(Q_1\) 和 \(Q_2\) 有正交归一列，那么 \(Q_1 \otimes Q_2\) 也是。 \(\lhd\)

**8.14** 设 \(A_1 = U_1 \Sigma_1 V_1^T\) 和 \(A_2 = U_2 \Sigma_2 V_2^T\) 分别是 \(A_1, A_2 \in \mathbb{R}^{n \times n}\) 的满秩奇异值分解。

a) 计算一个 \(A_1 \otimes A_2\) 的满秩奇异值分解。[提示：使用问题 8.13。]

b) 证明 \(A_1 \otimes A_2\) 的秩是 \(\mathrm{rk}(A_1) \,\mathrm{rk}(A_2)\)。 \(\lhd\)

**8.15** 设 \(P_1\) 和 \(P_2\) 是转移矩阵。

a) 设 \(\bpi_1\) 和 \(\bpi_2\)（作为行向量）分别是 \(P_1\) 和 \(P_2\) 的平稳分布。证明 \(\bpi_1 \otimes \bpi_2\) 是 \(P_1 \otimes P_2\) 的平稳分布。

b) 假设 \(P_1\) 和 \(P_2\) 都是既约的又是懒惰的。证明 \(P_1 \otimes P_2\) 也是这样。 \(\lhd\)

**8.16** 设 \(\mathbf{u}\) 为列向量，\(A, B\) 为可以形成矩阵乘积 \(AB\) 的矩阵。

a) 设 \(\mathbf{a}_1^T, \ldots, \mathbf{a}_n^T\) 为 \(A\) 的行。证明

\[\begin{split} A \otimes \mathbf{u} = \begin{pmatrix} \mathbf{u} \mathbf{a}_1^T\\ \vdots\\ \mathbf{u} \mathbf{a}_n^T \end{pmatrix} \end{split}\]

b) 证明**克罗内克积性质引理**的第 f) 部分。 \(\lhd\)

**8.17** 证明**连续函数的复合引理**。 \(\lhd\)

**8.18** 考虑映射 \(X \mathbf{z}\) 作为矩阵 \(X \in \mathbb{R}^{n \times m}\) 的元素的函数。具体来说，对于固定的 \(\mathbf{z} \in \mathbb{R}^m\)，令 \((\mathbf{x}^{(i)})^T\) 为 \(X\) 的第 \(i\) 行，我们定义函数

\[\begin{split} \mathbf{f}(\mathbf{x}) = X \mathbf{z} = \begin{pmatrix} (\mathbf{x}^{(1)})^T \mathbf{z} \\ \vdots\\ (\mathbf{x}^{(n)})^T \mathbf{z} \end{pmatrix} \end{split}\]

其中 \(\mathbf{x} = \mathrm{vec}(X^T) = (\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)})\). 证明 \(\mathbf{f}\) 在 \(\mathbf{x}\) 上是线性的，即 \(\mathbf{f}(\mathbf{x} + \mathbf{x}') = \mathbf{f}(\mathbf{x}) + \mathbf{f}(\mathbf{x}')\).

\(\lhd\)

**8.19** 设 \(f(x_1, x_2) = \sin(x_1² + x_2) + \cos(x_1 x_2)\)。通过定义适当的函数 \(\mathbf{g}\) 和 \(h\) 使得 \(f = h \circ \mathbf{g}\)，使用 **链式法则** 计算 \(f\) 的梯度。

\(\lhd\)

**8.20** 考虑函数 \(f(x_1, x_2, x_3) = \sqrt{x_1 + x_2² + \exp(x_3)}\)。通过定义合适的函数 \(\mathbf{g}\) 和 \(h\) 使得 \(f = h \circ \mathbf{g}\)，使用 **链式法则** 在点 \((1, 2, 0)\) 处找到 \(f\) 的梯度。

\(\lhd\)

**8.21** 考虑函数 \(f(x_1, x_2, x_3) = (x_1 + x_2²)³ + \sin(x_2 x_3)\)。通过定义合适的函数 \(\mathbf{g}\) 和 \(h\) 使得 \(f = h \circ \mathbf{g}\)，使用 **链式法则** 计算 \(f\) 的梯度。

\(\lhd\)

**8.22** 对于 \(i=1, \ldots, n\)，设 \(f_i : D_i \to \mathbb{R}\)，其中 \(D_i \subseteq \mathbb{R}\)，是关于单变量的连续可微实值函数。考虑定义在 \(D_1 \times \cdots \times D_n\) 上的向量值函数 \(\mathbf{f} : D_1 \times \cdots \times D_n \to \mathbb{R}^n\)，如下所示

\[ \mathbf{f}(\mathbf{x}) = (f_1(\mathbf{x}), \ldots, f_n(\mathbf{x})) = (f_1(x_1), \ldots, f_n(x_n)). \]

对于 \(\mathbf{x} = (x_1, \ldots, x_n)\) 满足 \(x_i\) 是 \(D_i\) 的内点对所有 \(i\) 成立，计算 \(J_{\mathbf{f}}(\mathbf{x})\) 的雅可比矩阵。

\(\lhd\)

**8.23** 设 \(f\) 是一个取矩阵 \(A = (a_{i,j})_{i,j} \in \mathbb{R}^{n \times n}\) 作为输入的实值函数。假设 \(f\) 在 \(A\) 的每个元素上都是连续可微的。考虑以下矩阵导数

\[\begin{split} \frac{\partial f(A)}{\partial A} = \begin{pmatrix} \frac{\partial f(A)}{\partial a_{1,1}} & \cdots & \frac{\partial f(A)}{\partial a_{1,n}}\\ \vdots & \ddots & \vdots\\ \frac{\partial f(A)}{\partial a_{n,1}} & \cdots & \frac{\partial f(A)}{\partial a_{n,n}} \end{pmatrix}. \end{split}\]

a) 证明对于任意的 \(B \in \mathbb{R}^{n \times n}\),

\[ \frac{\partial \,\mathrm{tr}(B^T A)}{\partial A} = B. \]

b) 证明对于任意的 \(\mathbf{x}, \mathbf{y} \in \mathbb{R}^d\),

\[ \frac{\partial \,\mathbf{x}^T A \mathbf{y}}{\partial A} = \mathbf{x} \mathbf{y}^T. \]

\(\lhd\)

**8.24** 设 \(A = (a_{i,j})_{i \in [n], j \in [m]} \in \mathbb{R}^{n \times m}\) 和 \(B = (b_{i,j})_{i \in [p], j \in [q]} \in \mathbb{R}^{p \times q}\) 是任意矩阵。它们的克罗内克积，记为 \(A \otimes B \in \mathbb{R}^{np \times mq}\)，是以下分块形式的矩阵

\[\begin{split} A \otimes B = \begin{pmatrix} a_{1,1} B & \cdots & a_{1,m} B \\ \vdots & \ddots & \vdots \\ a_{n,1} B & \cdots & a_{n,m} B \end{pmatrix}. \end{split}\]

克罗内克积满足以下性质（这些性质由分块公式得出，但不需要证明）：1) 如果 \(A, B, C, D\) 是可以形成矩阵乘积 \(AC\) 和 \(BD\) 的矩阵，那么 \((A \otimes B)\,(C \otimes D) = (AC) \otimes (BD)\)；2) \(A \otimes B\) 的转置是 \((A \otimes B)^T = A^T \otimes B^T\).

a) 证明如果 \(D_1\) 和 \(D_2\) 是方对角矩阵，那么 \(D_1 \otimes D_2\) 也是。

b) 证明如果 \(Q_1\) 和 \(Q_2\) 有正交归一列，那么 \(Q_1 \otimes Q_2\) 也有。

c) 设 \(A_1 = U_1 \Sigma_1 V_1^T\) 和 \(A_2 = U_2 \Sigma_2 V_2^T\) 分别是 \(A_1, A_2 \in \mathbb{R}^{n \times n}\) 的满 SVD。计算 \(A_1 \otimes A_2\) 的满 SVD。

d) 设 \(A_1\) 和 \(A_2\) 如 c)所述。证明 \(A_1 \otimes A_2\) 的秩为 \(\mathrm{rk}(A_1) \,\mathrm{rk}(A_2)\)。

\(\lhd\)

## 8.6.1\. 热身练习表#

*(在 Claude，Gemini 和 ChatGPT 的帮助下)*

**第 8.2 节**

**E8.2.1** 设 \(A = \begin{pmatrix} 2 & 1 \\ 0 & -1 \end{pmatrix}\)。计算向量 \(\text{vec}(A)\)。

**E8.2.2** 设 \(\mathbf{a} = (2, -1, 3)\) 和 \(\mathbf{b} = (1, 0, -2)\)。计算 Hadamard 积 \(a \odot b\)。

**E8.2.3** 设 \(A = \begin{pmatrix} 1 & 2 \\ -1 & 0 \end{pmatrix}\) 和 \(B = \begin{pmatrix} 3 & -1 \\ 2 & 1 \end{pmatrix}\)。计算 Kronecker 积 \(A \otimes B\)。

**E8.2.4** 设 \(A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}\) 和 \(B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}\)。计算 Hadamard 积 \(A \odot B\)。

**E8.2.5** 设 \(A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}\) 和 \(B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}\)。计算 Kronecker 积 \(A \otimes B\)。

**E8.2.6** 设 \(\mathbf{f}(x, y) = (x²y, \sin(xy), e^{x+y})\)。计算 \(\mathbf{f}\) 在点 \((1, \frac{\pi}{2})\) 处的雅可比矩阵。

**E8.2.7** 设 \(\mathbf{f}(x, y) = (x² + y², xy)\) 和 \(\mathbf{g}(u, v) = (uv, u + v)\)。计算 \(\mathbf{g} \circ \mathbf{f}\) 在点 \((1, 2)\) 处的雅可比矩阵。

**E8.2.8** 设 \(\mathbf{a} = (1, 2, 3)\)，\(\mathbf{b} = (4, 5, 6)\)，和 \(\mathbf{c} = (7, 8, 9)\)。计算 \(\mathbf{a}^T(\mathbf{b} \odot \mathbf{c})\) 和 \(\mathbf{1}^T(\mathbf{a} \odot \mathbf{b} \odot \mathbf{c})\) 并验证它们相等。

**E8.2.9** 设 \(A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}\) 和 \(B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}\)。计算 \((A \otimes B)^T\) 和 \(A^T \otimes B^T\) 并验证它们相等。

**E8.2.10** 设 \(\mathbf{g}(x, y) = (x²y, x + y)\)。计算 \(J_{\mathbf{g}}(x, y)\)。

**E8.2.11** 设 \(f(x, y, z) = x² + y² + z²\)。计算 \(f\) 在点 \((1, 2, 3)\) 处的梯度。

**E8.2.12** 设 \(f(x) = 2x³ - x\) 和 \(g(y) = y² + 1\)。计算复合函数 \(g \circ f\) 在 \(x = 1\) 处的雅可比矩阵。

**E8.2.13** 设 \(f(x, y) = xy\) 和 \(\mathbf{g}(x, y) = (x², y²)\)。计算 \(f \circ \mathbf{g}\) 在点 \((1, 2)\) 处的雅可比矩阵。

**E8.2.14** 设 \(A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}\) 和 \(\mathbf{b} = \begin{pmatrix} 5 \\ 6 \end{pmatrix}\)。定义函数 \(\mathbf{f}(\mathbf{x}) = A \mathbf{x} + \mathbf{b}\)。计算 \(\mathbf{f}\) 在任意点 \(\mathbf{x} \in \mathbb{R}²\) 处的雅可比矩阵。

**E8.2.15** 设 \(f(x) = \sin(x)\)。定义函数 \(\mathbf{g}(x, y, z) = (f(x), f(y), f(z))\)。计算 \(\mathbf{g}\) 在点 \((\frac{\pi}{2}, \frac{\pi}{4}, \frac{\pi}{6})\) 处的雅可比矩阵。

**E8.2.16** 使用 PyTorch 计算 \(f(x) = x³ - 4x\) 在 \(x = 2\) 处的梯度。提供 PyTorch 代码和结果。

**第 8.3 节**

**E8.3.1** 设 \(A = \begin{pmatrix} 1 & -1 \\ 0 & 2 \end{pmatrix}\) 和 \(B = \begin{pmatrix} 2 & 1 \\ -1 & 3 \end{pmatrix}\)。计算 \(AB\) 需要进行多少次基本运算（加法和乘法）？

**E8.3.2** 设 \(A = \begin{pmatrix} 1 & -1 \\ 0 & 2 \end{pmatrix}\) 和 \(\mathbf{v} = (2, -1)\)。计算 \(A\mathbf{v}\) 需要进行多少次基本运算（加法和乘法）？

**E8.3.3** 设 \(\ell(\hat{\mathbf{y}}) = \|\hat{\mathbf{y}}\|²\) 其中 \(\hat{\mathbf{y}} = (\hat{y}_1, \hat{y}_2)\)。计算 \(J_{\ell}(\hat{\mathbf{y}})\)。

**E8.3.4** 设 \(\mathbf{g}_0(\mathbf{z}_0) = \begin{pmatrix} 1 & 2 \\ -1 & 0 \end{pmatrix} \mathbf{z}_0\) 和 \(\ell(\hat{\mathbf{y}}) = \|\hat{\mathbf{y}}\|²\)。计算 \(f(\mathbf{x}) = \ell(\mathbf{g}_0(\mathbf{x}))\) 和 \(\mathbf{x} = (1, -1)\) 时的 \(\nabla f(\mathbf{x})\)。

**E8.3.5** 设 \(\mathbf{g}_0\) 和 \(\ell\) 如 E8.3.4 所述。设 \(\mathbf{g}_1(\mathbf{z}_1) = \begin{pmatrix} -1 & 0 \\ 1 & 1 \end{pmatrix} \mathbf{z}_1\)。使用逆模式计算 \(f(\mathbf{x}) = \ell(\mathbf{g}_1(\mathbf{g}_0(\mathbf{x})))\) 和 \(\mathbf{x} = (1, -1)\) 时的 \(\nabla f(\mathbf{x})\)。

**E8.3.6** 设 \(\mathbf{g}_0(\mathbf{x}, \mathbf{w}_0) = \mathcal{W}_0 \mathbf{x}\) 其中 \(\mathcal{W}_0 = \begin{pmatrix} w_0 & w_1 \\ w_2 & w_3 \end{pmatrix}\) 和 \(\mathbf{w}_0 = (w_0, w_1, w_2, w_3)\)。设 \(\mathbf{x} = (-1, 1)\) 为固定点。仅通过直接计算必要的偏导数（即不使用文本中的公式），计算 \(\mathbf{g}_0\) 关于 \(\mathbf{w}_0\) 的雅可比 \(J_{\mathbf{g}_0}(\mathbf{x}, \mathbf{w}_0)\)，然后与文本中的公式进行比较。

**E8.3.7** 设 \(g_1(\mathbf{z}_1, \mathbf{w}_1) = \mathcal{W}_1 \mathbf{z}_1\) 其中 \(\mathcal{W}_1 = \begin{pmatrix} w_4 & w_5\end{pmatrix}\) 和 \(\mathbf{w}_1 = (w_4, w_5)\)。通过直接计算必要的偏导数（即不使用文本中的公式），计算 \(J_{g_1}(\mathbf{z}_1, \mathbf{w}_1)\) 关于 \(\mathbf{z}_1\) 和 \(\mathbf{w}_1\) 的雅可比，然后与文本中的公式进行比较。

**E8.3.8** 设 \(h(\mathbf{w}) = g_1(\mathbf{g}_0(\mathbf{x}, \mathbf{w}_0), \mathbf{w}_1)\)，其中 \(\mathbf{g}_0\) 和 \(g_1\) 如 E8.3.6 和 E8.3.7 所述，且 \(\mathbf{w} = (\mathbf{w}_0, \mathbf{w}_1) = (w_0, w_1, w_2, w_3, w_4, w_5)\)。设 \(\mathbf{x} = (-1, 1)\) 为固定值。通过直接计算必要的偏导数（即，不使用文本中的公式）来计算 \(J_h(\mathbf{w})\)。

**E8.3.9** 设 \(f(\mathbf{w}) = \ell(g_1(\mathbf{g}_0(\mathbf{x}, \mathbf{w}_0), \mathbf{w}_1))\)，其中 \(\ell(\hat{y}) = \hat{y}²\)，\(\mathbf{g}_0\) 和 \(g_1\) 如 E8.3.6 和 E8.3.7 所述，且 \(\mathbf{w} = (\mathbf{w}_0, \mathbf{w}_1) = (w_0, w_1, w_2, w_3, w_4, w_5)\)。设 \(\mathbf{x} = (-1, 1)\) 为固定值。通过直接计算必要的偏导数（即，不使用文本中的公式）来计算 \(J_f(\mathbf{w})\)，然后与文本中的公式进行比较。

**第 8.4 节**

**E8.4.1** 给定一个包含 5 个样本的数据集，计算全梯度下降步骤和批大小为 2 的期望随机梯度下降步骤。假设单个样本的梯度为：\(\nabla f_{\mathbf{x}_1, y_1}(\mathbf{w}) = (1, 2)\)，\(\nabla f_{\mathbf{x}_2, y_2}(\mathbf{w}) = (-1, 1)\)，\(\nabla f_{\mathbf{x}_3, y_3}(\mathbf{w}) = (0, -1)\)，\(\nabla f_{\mathbf{x}_4, y_4}(\mathbf{w}) = (2, 0)\)，和 \(\nabla f_{\mathbf{x}_5, y_5}(\mathbf{w}) = (1, 1)\)。

**E8.4.2** 计算 \(\mathbf{z} = (1, -2, 0, 3)\) 的 softmax 函数 \(\boldsymbol{\gamma}(\mathbf{z})\)。

**E8.4.3** 计算概率分布 \(\mathbf{p} = (0.2, 0.3, 0.5)\) 和 \(\mathbf{q} = (0.1, 0.4, 0.5)\) 之间的 Kullback-Leibler 散度。

**E8.4.4** 在具有单个特征的线性回归中，单个样本 \((x, y)\) 的损失函数由下式给出

\[ \ell(w, b; x, y) = (wx + b - y)². \]

计算梯度 \(\frac{\partial \ell}{\partial w}\) 和 \(\frac{\partial \ell}{\partial b}\)。

**E8.4.5** 假设我们有一个包含三个样本的数据集：\((x_1, y_1) = (2, 3)\)，\((x_2, y_2) = (-1, 0)\)，和 \((x_3, y_3) = (1, 2)\)。我们想要对线性回归进行小批量随机梯度下降，批大小为 2。如果第一个随机选择的小批量是 \(B = \{1, 3\}\)，假设学习率为 \(\alpha = 0.1\)，计算参数 \(w\) 和 \(b\) 的 SGD 更新。模型初始化为 \(w = 1\) 和 \(b = 0\)。

**E8.4.6** 对于单个样本 \((\mathbf{x}, y) = ((1, 2), 3)\) 的线性回归问题，计算损失函数 \(f(\mathbf{w}) = (\mathbf{x}^T \mathbf{w} - y)²\) 在 \(\mathbf{w} = (0, 0)\) 处的梯度。

**E8.4.7** 考虑单个样本 \((x, y)\) 的逻辑回归损失函数，其中 \(x \in \mathbb{R}\) 且 \(y \in \{0, 1\}\)：

\[ \ell(w; x, y) = -y \log(\sigma(wx)) - (1-y) \log(1 - \sigma(wx)), \]

其中 \(\sigma(z) = \frac{1}{1 + e^{-z}}\) 是 sigmoid 函数。计算关于 \(w\) 的梯度 \(\nabla \ell(w; x, y)\)。

**E8.4.8** 考虑一个有三个类别 (\(K = 3\)) 的多项式逻辑回归问题。给定输入向量 \(\mathbf{x} = (1, -1)\) 和权重矩阵

\[\begin{split} W = \begin{pmatrix} 1 & 2 \\ -1 & 0 \\ 0 & 1 \end{pmatrix}, \end{split}\]

计算 \(\boldsymbol{\gamma}(W\mathbf{x})\) 的 softmax 输出。

**E8.4.9** 对于具有单个样本 \((\mathbf{x}, \mathbf{y}) = ((1, 2), (0, 0, 1))\) 和 \(K = 3\) 个类别的多项式逻辑回归问题，计算损失函数 \(f(\mathbf{w}) = -\sum_{i=1}^K y_i \log \gamma_i(W\mathbf{x})\) 在 \(W = \begin{pmatrix} 0 & 0 \\ 0 & 0 \\ 0 & 0 \end{pmatrix}\) 处的梯度。

**E8.4.10** 对于具有两个样本 \((\mathbf{x}_1, y_1) = ((1, 2), 3)\) 和 \((\mathbf{x}_2, y_2) = ((4, -1), 2)\) 的线性回归问题，计算 \(\mathbf{w} = (0, 0)\) 时的完整梯度。

**E8.4.11** 对于具有两个样本 \((\mathbf{x}_1, \mathbf{y}_1) = ((1, 2), (0, 0, 1))\) 和 \((\mathbf{x}_2, \mathbf{y}_2) = ((4, -1), (1, 0, 0))\) 的多项式逻辑回归问题，以及 \(K = 3\) 个类别，计算 \(W = \begin{pmatrix} 0 & 0 \\ 0 & 0 \\ 0 & 0 \end{pmatrix}\) 时的完整梯度。

**E8.4.12** 在一个二元分类问题中，逻辑回归模型预测了两个样本的概率为 0.8 和 0.3。如果这些样本的真实标签分别是 1 和 0，计算平均交叉熵损失。

**E8.4.13** 在一个有四个类别的多类别分类问题中，一个模型预测了一个样本的概率分布：\((0.1, 0.2, 0.3, 0.4)\)。如果真实标签是第三个类别，那么这个样本的交叉熵损失是多少？

**第 8.5 节**

**E8.5.1** 计算以下 \(t\) 值的 sigmoid 函数 \(\sigma(t)\)：\(1, -1, 2\).

**E8.5.2** 计算以下 \(t\) 值的 sigmoid 函数的导数 \(\sigma'(t)\)：\(1, -1, 2\).

**E8.5.3** 给定向量 \(\mathbf{z} = (1, -1, 2)\)，计算 \(\bsigma(\mathbf{z})\) 和 \(\bsigma'(\mathbf{z})\).

**E8.5.4** 给定矩阵 \(W = \begin{pmatrix} 1 & -1 \\ 2 & 0 \end{pmatrix}\) 和向量 \(\mathbf{x} = (1, 2)\)，计算 \(\bsigma(W\mathbf{x})\).

**E8.5.5** 给定矩阵 \(W = \begin{pmatrix} 1 & -1 \\ 2 & 0 \end{pmatrix}\) 和向量 \(\mathbf{x} = (1, 2)\)，计算 \(\bsigma(W\mathbf{x})\) 的雅可比矩阵。

**E8.5.6** 给定向量 \(\mathbf{y} = (0, 1)\) 和 \(\mathbf{z} = (0.3, 0.7)\)，计算交叉熵损失 \(H(\mathbf{y}, \mathbf{z})\).

**E8.5.7** 给定向量 \(\mathbf{y} = (0, 1)\) 和 \(\mathbf{z} = (0.3, 0.7)\)，计算交叉熵损失 \(\nabla H(\mathbf{y}, \mathbf{z})\) 的梯度。

**E8.5.8** 给定向量 \(\mathbf{w} = (1, 2, -1, 0)\) 和 \(\mathbf{z} = (1, 2)\)，计算 \(\mathbb{A}_2[\mathbf{w}]\) 和 \(\mathbb{B}_2[\mathbf{z}]\)。

## 8.6.2\. 问题#

**第 8.1 节** 考虑仿射映射

\[ \mathbf{f}(\mathbf{x}) = A \mathbf{x} + \mathbf{b}, \]

其中 \(A \in \mathbb{R}^{m \times d}\) 且 \(\mathbf{b} = (b_1, \ldots, b_m) \in \mathbb{R}^m\)。设 \(S \subseteq \mathbb{R}^m\) 是一个凸集。证明以下集合是凸的：

\[ T = \left\{ \mathbf{x} \in \mathbb{R}^d \,:\, \mathbf{f}(\mathbf{x}) \in S \right\}. \]

\(\lhd\)

**8.2** （改编自 [Khu]）考虑向量值函数 \(\mathbf{f} = (f_1, \ldots, f_d) : \mathbb{R}^d \to \mathbb{R}^d\)，定义为

\[ f_i(\mathbf{x}) = x_i³, \]

对于所有 \(\mathbf{x} \in \mathbb{R}^d\) 和所有 \(i = 1, \ldots, d\)。

a) 计算所有 \(\mathbf{x}\) 的雅可比矩阵 \(\mathbf{J}_\mathbf{f}(\mathbf{x})\)。

b) 当 \(\mathbf{J}_\mathbf{f}(\mathbf{x})\) 是可逆的吗？

c) 当 \(\mathbf{J}_\mathbf{f}(\mathbf{x})\) 是正定半定吗？\(\lhd\)

**8.3** 设 \(A = (a_{i,j})_{i,j=1}^n \in \mathbb{R}^{n \times n}\) 是一个对称矩阵。

a) 设 \(\mathbf{v} = (v_1, \ldots, v_n) \in \mathbb{R}^n\) 是 \(A\) 的一个特征向量，其特征值为 \(\lambda\)。设 \(v_{i}\) 是 \(\mathbf{v}\) 中绝对值最大的元素，即 \(i \in \arg\max_j |v_j|\)。定义向量 \(\mathbf{w} = (w_1, \ldots, w_n)\) 如下

\[ w_j = \frac{v_j}{v_{i}}, \qquad j=1, \ldots, n. \]

证明

\[ \lambda - a_{i,i} = \sum_{j \neq i} a_{i,j} w_j. \]

b) 使用 (a) 来证明，对于 \(A\) 的任何特征值 \(\lambda\)，存在 \(i\) 使得

\[ |\lambda - a_{i,i}| \leq \sum_{j \neq i} |a_{i,j}|. \]

\(\lhd\)

**8.4** 一个对称矩阵 \(A = (a_{i,j})_{i,j=1}^n \in \mathbb{R}^{n \times n}\) 如果其对角线上的元素非负，并且对于所有 \(i\)

\[ a_{i,i} \geq \sum_{j \neq i} |a_{i,j}|, \]

即，每个对角元素都大于或等于其行中其他元素的绝对值之和。使用问题 8.3 来证明这样的矩阵是正定半定的。\(\lhd\)

**8.5** 考虑多项式逻辑回归。设

\[ R = I_{K \times K} \otimes \mathbf{x}^T, \]

和

\[ S = \mathrm{diag}\left( \bgamma \left( \mathbf{v} \right) \right) - \bgamma \left( \mathbf{v} \right) \, \bgamma \left( \mathbf{v} \right)^T \]

其中

\[ \mathbf{v} = \bgamma \left( \bfg_0 (\mathbf{x}, \mathbf{w}) \right). \]

a) 证明

\[ \nabla f(\mathbf{w}) = \Gamma \left( \bgamma \left( \bfg_0 (\mathbf{x}, \mathbf{w}) \right) \right) \]

其中

\[ \Gamma (\mathbf{u}) = R (\mathbf{u} - \mathbf{y}). \]

b) 使用 **链式法则** 来证明

\[ H_f (\mathbf{w}) = R^T S R. \]

c) 使用 (b) 和 **克罗内克积的性质** 来证明

\[ H_f (\mathbf{w}) = \left( \mathrm{diag} \left( \bgamma \left( \mathcal{W} \mathbf{x} \right) \right) - \bgamma \left( \mathcal{W} \mathbf{x} \right) \, \bgamma \left( \mathcal{W} \mathbf{x} \right)^T \right) \otimes \mathbf{x} \mathbf{x}^T. \]

\(\lhd\)

**8.6** 考虑多项式逻辑回归。使用问题 8.4 和 8.5 来证明目标函数是凸的。[提示：只需证明 \(S\)（在问题 8.5 中定义）是对角占优的。为什么？]\(\lhd\)

**8.7** 证明 *克罗内克积性质引理* 的部分 a)。\(\lhd\)

**8.8** 证明 *克罗内克积性质引理* 的部分 b)。\(\lhd\)

**8.9** 证明 *克罗内克积性质引理* 的部分 c) 和 d)。\(\lhd\)

**8.10** 证明 *克罗内克积性质引理* 的部分 e)。\(\lhd\)

**8.11** 设 \(A\) 和 \(B\) 分别是大小为 \(n \times n\) 和 \(m \times m\) 的对称矩阵。

a) 证明 \(A \otimes B\) 是对称的。[提示：使用问题 8.10。]\(\lhd\)

b) 用 \(A\) 和 \(B\) 的特征向量与特征值表示 \(A \otimes B\) 的特征向量与特征值。[提示：尝试 \(A\) 和 \(B\) 的特征向量的克罗内克积。]

c) 回忆对称矩阵的行列式是其特征值的乘积。证明 \(\lhd\)

\[ \mathrm{det}(A \otimes B) = \mathrm{det}(A)^n \,\mathrm{det}(B)^m. \]

\(\lhd\)

**8.12** 用 \(\mathrm{tr}(A)\) 和 \(\mathrm{tr}(B)\) 表示 \(\mathrm{tr}(A \otimes B)\)。证明你的答案。 \(\lhd\)

**8.13** a) 证明如果 \(D_1\) 和 \(D_2\) 是方对角矩阵，那么 \(D_1 \otimes D_2\) 也是。\(\lhd\)

b) 证明如果 \(Q_1\) 和 \(Q_2\) 有正交列，那么 \(Q_1 \otimes Q_2\) 也有正交列。\(\lhd\)

**8.14** 设 \(A_1 = U_1 \Sigma_1 V_1^T\) 和 \(A_2 = U_2 \Sigma_2 V_2^T\) 分别是 \(A_1, A_2 \in \mathbb{R}^{n \times n}\) 的全奇异值分解。

a) 计算 \(A_1 \otimes A_2\) 的完全奇异值分解。[提示：使用问题 8.13。]

b) 证明 \(A_1 \otimes A_2\) 的秩是 \(\mathrm{rk}(A_1) \,\mathrm{rk}(A_2)\)。\(\lhd\)

**8.15** 设 \(P_1\) 和 \(P_2\) 是转移矩阵。

a) 设 \(\bpi_1\) 和 \(\bpi_2\)（作为行向量）分别是 \(P_1\) 和 \(P_2\) 的平稳分布。证明 \(\bpi_1 \otimes \bpi_2\) 是 \(P_1 \otimes P_2\) 的平稳分布。

b) 假设 \(P_1\) 和 \(P_2\) 都是不可约的且懒惰的。证明 \(P_1 \otimes P_2\) 也是这样。\(\lhd\)

**8.16** 设 \(\mathbf{u}\) 是一个列向量，\(A, B\) 是可以形成矩阵乘积 \(AB\) 的矩阵。

a) 设 \(\mathbf{a}_1^T, \ldots, \mathbf{a}_n^T\) 是 \(A\) 的行。\(\lhd\)

\[\begin{split} A \otimes \mathbf{u} = \begin{pmatrix} \mathbf{u} \mathbf{a}_1^T\\ \vdots\\ \mathbf{u} \mathbf{a}_n^T \end{pmatrix}. \end{split}\]

b) 证明 *克罗内克积性质引理* 的部分 f)。\(\lhd\)

**8.17** 证明 *连续函数的复合引理*。\(\lhd\)

**8.18** 考虑映射 \(X \mathbf{z}\) 作为矩阵 \(X \in \mathbb{R}^{n \times m}\) 的元素的函数。具体来说，对于固定的 \(\mathbf{z} \in \mathbb{R}^m\)，令 \((\mathbf{x}^{(i)})^T\) 为 \(X\) 的第 \(i\) 行，我们定义函数

\[\begin{split} \mathbf{f}(\mathbf{x}) = X \mathbf{z} = \begin{pmatrix} (\mathbf{x}^{(1)})^T \mathbf{z} \\ \vdots\\ (\mathbf{x}^{(n)})^T \mathbf{z} \end{pmatrix} \end{split}\]

其中 \(\mathbf{x} = \mathrm{vec}(X^T) = (\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)})\)。证明 \(\mathbf{f}\) 在 \(\mathbf{x}\) 上是线性的，即 \(\mathbf{f}(\mathbf{x} + \mathbf{x}') = \mathbf{f}(\mathbf{x}) + \mathbf{f}(\mathbf{x}')\)。

\(\lhd\)

**8.19** 设 \(f(x_1, x_2) = \sin(x_1² + x_2) + \cos(x_1 x_2)\)。通过定义适当的函数 \(\mathbf{g}\) 和 \(h\) 使得 \(f = h \circ \mathbf{g}\)，使用**链式法则**求 \(f\) 的梯度。

\(\lhd\)

**8.20** 考虑函数 \(f(x_1, x_2, x_3) = \sqrt{x_1 + x_2² + \exp(x_3)}\)。通过定义合适的函数 \(\mathbf{g}\) 和 \(h\) 使得 \(f = h \circ \mathbf{g}\)，使用**链式法则**求 \(f\) 在点 \((1, 2, 0)\) 处的梯度。

\(\lhd\)

**8.21** 考虑函数 \(f(x_1, x_2, x_3) = (x_1 + x_2²)³ + \sin(x_2 x_3)\)。通过定义合适的函数 \(\mathbf{g}\) 和 \(h\) 使得 \(f = h \circ \mathbf{g}\)，使用**链式法则**求 \(f\) 的梯度。

\(\lhd\)

**8.22** 对于 \(i=1, \ldots, n\)，设 \(f_i : D_i \to \mathbb{R}\)，其中 \(D_i \subseteq \mathbb{R}\)，是一个关于单变量的连续可微实值函数。考虑定义在 \(D_1 \times \cdots \times D_n \to \mathbb{R}^n\) 的向量值函数 \(\mathbf{f}\)：

\[ \mathbf{f}(\mathbf{x}) = (f_1(\mathbf{x}), \ldots, f_n(\mathbf{x})) = (f_1(x_1), \ldots, f_n(x_n)). \]

对于 \(\mathbf{x} = (x_1, \ldots, x_n)\) 且 \(x_i\) 是 \(D_i\) 的内部点（对所有 \(i\)），计算 \(J_{\mathbf{f}}(\mathbf{x})\)。

\(\lhd\)

**8.23** 设 \(f\) 是一个以矩阵 \(A = (a_{i,j})_{i,j} \in \mathbb{R}^{n \times n}\) 为输入的实值函数。假设 \(f\) 在 \(A\) 的每个元素上都是连续可微的。考虑以下矩阵导数

\[\begin{split} \frac{\partial f(A)}{\partial A} = \begin{pmatrix} \frac{\partial f(A)}{\partial a_{1,1}} & \cdots & \frac{\partial f(A)}{\partial a_{1,n}}\\ \vdots & \ddots & \vdots\\ \frac{\partial f(A)}{\partial a_{n,1}} & \cdots & \frac{\partial f(A)}{\partial a_{n,n}} \end{pmatrix}. \end{split}\]

a) 证明，对于任意 \(B \in \mathbb{R}^{n \times n}\)，

\[ \frac{\partial \,\mathrm{tr}(B^T A)}{\partial A} = B. \]

b) 证明，对于任意 \(\mathbf{x}, \mathbf{y} \in \mathbb{R}^d\)，

\[ \frac{\partial \,\mathbf{x}^T A \mathbf{y}}{\partial A} = \mathbf{x} \mathbf{y}^T. \]

\(\lhd\)

**8.24** 设 \(A = (a_{i,j})_{i \in [n], j \in [m]} \in \mathbb{R}^{n \times m}\) 和 \(B = (b_{i,j})_{i \in [p], j \in [q]} \in \mathbb{R}^{p \times q}\) 是任意矩阵。它们的克罗内克积，记为 \(A \otimes B \in \mathbb{R}^{np \times mq}\)，是以下分块形式的矩阵

\[\begin{split} A \otimes B = \begin{pmatrix} a_{1,1} B & \cdots & a_{1,m} B \\ \vdots & \ddots & \vdots \\ a_{n,1} B & \cdots & a_{n,m} B \end{pmatrix}. \end{split}\]

克朗内克积满足以下性质（这些性质由分块公式得出，但不需要证明）：1) 如果 \(A, B, C, D\) 是可以形成矩阵乘积 \(AC\) 和 \(BD\) 的矩阵，那么 \((A \otimes B)\,(C \otimes D) = (AC) \otimes (BD)\)；2) \(A \otimes B\) 的转置是 \((A \otimes B)^T = A^T \otimes B^T\)。

a) 证明如果 \(D_1\) 和 \(D_2\) 是方对角矩阵，那么 \(D_1 \otimes D_2\) 也是。

b) 证明如果 \(Q_1\) 和 \(Q_2\) 有正交归一列，那么 \(Q_1 \otimes Q_2\) 也有。

c) 设 \(A_1 = U_1 \Sigma_1 V_1^T\) 和 \(A_2 = U_2 \Sigma_2 V_2^T\) 分别是 \(A_1, A_2 \in \mathbb{R}^{n \times n}\) 的满奇异值分解。计算 \(A_1 \otimes A_2\) 的满奇异值分解。

d) 设 \(A_1\) 和 \(A_2\) 如 c) 所述。证明 \(A_1 \otimes A_2\) 的秩是 \(\mathrm{rk}(A_1) \,\mathrm{rk}(A_2)\)。

\(\lhd\)
