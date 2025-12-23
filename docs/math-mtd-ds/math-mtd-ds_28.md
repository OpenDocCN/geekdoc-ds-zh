# 4.2. 背景知识：矩阵秩和谱分解的回顾#

> 原文：[`mmids-textbook.github.io/chap04_svd/02_spectral/roch-mmids-svd-spectral.html`](https://mmids-textbook.github.io/chap04_svd/02_spectral/roch-mmids-svd-spectral.html)

我们将需要线性代数的两个额外概念，矩阵的秩和谱定理。

## 4.2.1. 矩阵的秩#

回想一下，矩阵 \(A\) 的列空间的维度被称为 \(A\) 的列秩。类似地，\(A\) 的行秩是其行空间的维度。实际上，这两个秩的概念由 *行秩等于列秩定理*（\(\idx{row rank equals column rank theorem}\xdi\)）所等同。我们将在下文中给出该定理的证明。我们简单地将 \(A\) 的行秩和列秩称为秩，用 \(\mathrm{rk}(A)\) 表示。 \(\idx{rank}\xdi\)

*证明思路:* *(行秩等于列秩)* 将 \(A\) 写成矩阵分解 \(BC\)，其中 \(B\) 的列构成 \(\mathrm{col}(A)\) 的基。然后 \(C\) 的行必然构成 \(\mathrm{row}(A)\) 的一个生成集。因此，由于 \(B\) 的列数和 \(C\) 的行数相等，我们得出行秩小于或等于列秩。将相同的论证应用于转置矩阵，得到结论。

回想前一章中的以下观察。

**观察 D1:** 任何 \(\mathbb{R}^n\) 的线性子空间 \(U\) 都有一个基。

**观察 D2:** 如果 \(U\) 和 \(V\) 是满足 \(U \subseteq V\) 的线性子空间，那么 \(\mathrm{dim}(U) \leq \mathrm{dim}(V)\)。

**观察 D3:** \(\mathrm{span}(\mathbf{u}_1,\ldots,\mathbf{u}_m)\) 的维度最多为 \(m\)。

*证明:* *(行秩等于列秩)* 假设 \(A\) 的列秩为 \(r\)。根据上面的 *观察 D1*，存在一个基 \(\mathbf{b}_1,\ldots, \mathbf{b}_r \in \mathbb{R}^n\)，使得 \(\mathrm{col}(A)\) 的基为 \(\mathbf{b}_1,\ldots, \mathbf{b}_r\)，并且我们知道 \(r \leq n\)，根据 *观察 D2*。也就是说，对于每个 \(j\)，令 \(\mathbf{a}_{j} = A_{\cdot,j}\) 为 \(A\) 的第 \(j\) 列，我们可以写成

\[ \mathbf{a}_{j} = \sum_{\ell=1}^r \mathbf{b}_\ell c_{\ell j} \]

对于某些 \(c_{\ell j}\) 的值。令 \(B\) 为列由 \(\mathbf{b}_1, \ldots, \mathbf{b}_r\) 组成的矩阵，令 \(C\) 为元素 \((C)_{\ell j} = c_{\ell j}\)，\(\ell=1,\ldots,r\)，\(j=1,\ldots,m\) 的矩阵。那么上述方程可以重新写成矩阵分解 \(A = BC\)。确实，根据我们之前关于矩阵乘积的观察，\(A\) 的列是 \(B\) 的列的线性组合，系数来自 \(C\) 的对应列。

关键点是以下内容：\(C\) 必然有 \(r\) 行。令 \(\boldsymbol{\alpha}_{i}^T = A_{i,\cdot}\) 为 \(A\) 的第 \(i\) 行，\(\mathbf{c}_{\ell}^T = C_{\ell,\cdot}\) 为 \(C\) 的第 \(\ell\) 行。使用我们关于矩阵乘积的行表示的替代表示，分解等价于

\[ \boldsymbol{\alpha}_{i}^T = \sum_{\ell=1}^r b_{i\ell} \mathbf{c}_\ell^T, \quad i=1,\ldots, n, \]

其中 \(b_{i\ell} = (\mathbf{b}_i)_\ell = (B)_{i\ell}\) 是 \(B\) 的第 \(\ell\) 列的第 \(i\) 个元素。换句话说，\(A\) 的行是 \(C\) 的行的线性组合，系数来自 \(B\) 的对应行。特别是，\(\mathcal{C} = \{\mathbf{c}_{j}:j=1,\ldots,r\}\) 是 \(A\) 的行空间的生成集，即 \(A\) 的每一行都可以写成 \(\mathcal{C}\) 的线性组合。换句话说，\(\mathrm{row}(A) \subseteq \mathrm{span}(\mathcal{C})\).

因此，矩阵 \(A\) 的行秩最多为 \(r\)，根据 *观察 D2*，\(A\) 的列秩。

将相同的论点应用于 \(A^T\)，这交换了列和行的角色，得到 \(A\) 的列秩（即 \(A^T\) 的行秩）最多是 \(A\) 的行秩（即 \(A^T\) 的列秩）。因此，这两个秩的概念必须相等。（我们再次通过 *观察 D2* 推导出 \(r \leq m\)。）\(\square\)

**示例**: **（继续）** 我们说明了定理的证明。继续前面的例子，设 \(A\) 是一个矩阵，其列向量为 \(\mathbf{w}_1 = (1,0,1)\)，\(\mathbf{w}_2 = (0,1,1)\)，和 \(\mathbf{w}_3 = (1,-1,0)\)

\[\begin{split} A = \begin{pmatrix} 1 & 0 & 1\\ 0 & 1 & -1\\ 1 & 1 & 0 \end{pmatrix}. \end{split}\]

我们知道 \(\mathbf{w}_1\) 和 \(\mathbf{w}_2\) 构成了 \(\mathrm{col}(A)\) 的基。我们使用它们来构造我们的矩阵 \(B\)

\[\begin{split} B = \begin{pmatrix} 1 & 0\\ 0 & 1\\ 1 & 1 \end{pmatrix}. \end{split}\]

回忆一下 \(\mathbf{w}_3 = \mathbf{w}_1 - \mathbf{w}_2\). 因此矩阵 \(C\) 是

\[\begin{split} C = \begin{pmatrix} 1 & 0 & 1\\ 0 & 1 & -1 \end{pmatrix}. \end{split}\]

事实上，\(C\) 的第 \(j\) 列给出了产生 \(A\) 的第 \(j\) 列的 \(B\) 的列的线性组合中的系数。检查 \(A = B C\)。\(\lhd\)

**数值角**: 在 Numpy 中，可以使用函数 `numpy.linalg.matrix_rank` 计算矩阵的秩。我们将在本章后面看到如何使用奇异值分解来计算它（这是 `LA.matrix_rank` 所做的）。让我们尝试上面的例子。

```py
w1 = np.array([1., 0., 1.])
w2 = np.array([0., 1., 1.])
w3 = np.array([1., -1., 0.])
A = np.stack((w1, w2, w3), axis=-1)
print(A) 
```

```py
[[ 1\.  0\.  1.]
 [ 0\.  1\. -1.]
 [ 1\.  1\.  0.]] 
```

我们计算矩阵 `A` 的秩。

```py
LA.matrix_rank(A) 
```

```py
2 
```

我们这次只取 `A` 的前两列来形成 `B`。

```py
B = np.stack((w1, w2),axis=-1)
print(B) 
```

```py
[[1\. 0.]
 [0\. 1.]
 [1\. 1.]] 
```

```py
LA.matrix_rank(B) 
```

```py
2 
```

回忆一下，在 Numpy 中，`@` 用于矩阵乘法。

```py
C = np.array([[1., 0., 1.],[0., 1., -1.]])
print(C) 
```

```py
[[ 1\.  0\.  1.]
 [ 0\.  1\. -1.]] 
```

```py
LA.matrix_rank(C) 
```

```py
2 
```

```py
print(B @ C) 
```

```py
[[ 1\.  0\.  1.]
 [ 0\.  1\. -1.]
 [ 1\.  1\.  0.]] 
```

\(\unlhd\)

**示例**: 设 \(A \in \mathbb{R}^{n \times k}\) 和 \(B \in \mathbb{R}^{k \times m}\)。那么我们声称

\[ \mathrm{rk}(AB) \leq \mathrm{rk}(A). \]

事实上，\(AB\) 的列是 \(A\) 的列的线性组合。因此 \(\mathrm{col}(AB) \subseteq \mathrm{col}(A)\)。通过 *观察 D2* 得出这个结论。\(\lhd\)

**示例**: 设 \(A \in \mathbb{R}^{n \times k}\) 和 \(B \in \mathbb{R}^{n \times m}\)。那么我们声称

\[ \mathrm{rk}(A + B) \leq \mathrm{rk}(A) + \mathrm{rk}(B). \]

事实上，\(A + B\) 的列是 \(A\) 和 \(B\) 的列的线性组合。设 \(\mathbf{u}_1,\ldots,\mathbf{u}_h\) 是 \(\mathrm{col}(A)\) 的一个基，设 \(\mathbf{v}_1,\ldots,\mathbf{v}_{\ell}\) 是 \(\mathrm{col}(B)\) 的一个基，根据 *观察 D1*。然后，我们得出

\[ \mathrm{col}(A + B) \subseteq \mathrm{span}(\mathbf{u}_1,\ldots,\mathbf{u}_h,\mathbf{v}_1,\ldots,\mathbf{v}_{\ell}). \]

根据 *观察 D2*，可以得出

\[ \mathrm{rk}(A + B) \leq \mathrm{dim}(\mathrm{span}(\mathbf{u}_1,\ldots,\mathbf{u}_h,\mathbf{v}_1,\ldots,\mathbf{v}_{\ell})). \]

根据 *观察 D3*，右侧的长度最多是生成向量的长度，即 \(h+\ell\)。但是，根据构造，\(\mathrm{rk}(A) = h\) 和 \(\mathrm{rk}(B) = \ell\)，所以我们完成了。 \(\lhd\)

**例证：** **(秩-零度定理的证明)** \(\idx{rank-nullity theorem}\xdi\) 设 \(A \in \mathbb{R}^{n \times m}\)。回忆一下，\(A\) 的列空间 \(\mathrm{col}(A) \subseteq \mathbb{R}^n\) 是其列的生成空间。我们计算其正交补。根据定义，\(A\) 的列，我们用 \(\mathbf{a}_1,\ldots,\mathbf{a}_m\) 表示，形成 \(\mathrm{col}(A)\) 的一个生成列表。因此，如果 \(\mathbf{u} \in \mathrm{col}(A)^\perp \subseteq \mathbb{R}^n\)，则

\[ \mathbf{a}_i^T\mathbf{u} = \langle \mathbf{u}, \mathbf{a}_i \rangle = 0, \quad \forall i=1,\ldots,m. \]

事实上，这意味着对于任何 \(\mathbf{v} \in \mathrm{col}(A)\)，例如 \(\mathbf{v} = \beta_1 \mathbf{a}_1 + \cdots + \beta_m \mathbf{a}_m\)，我们有

\[ \left\langle \mathbf{u}, \sum_{i=1}^m \beta_i \mathbf{a}_i \right\rangle = \sum_{i=1}^m \beta_i \langle \mathbf{u}, \mathbf{a}_i \rangle = 0. \]

上述 \(m\) 个条件可以写成矩阵形式

\[ A^T \mathbf{u} = \mathbf{0}. \]

即，\(A\) 的列空间的正交补是 \(A^T\) 的零空间

\[ \mathrm{col}(A)^\perp = \mathrm{null}(A^T). \]

将相同的论点应用于 \(A^T\) 的列空间，可以得出

\[ \mathrm{col}(A^T)^\perp = \mathrm{null}(A), \]

注意到 \(\mathrm{null}(A) \subseteq \mathbb{R}^m\)。四个线性子空间 \(\mathrm{col}(A)\)，\(\mathrm{col}(A^T)\)，\(\mathrm{null}(A)\) 和 \(\mathrm{null}(A^T)\) 被称为 \(A\) 的基本子空间。我们已经证明了

\[ \mathrm{col}(A) \oplus \mathrm{null}(A^T) = \mathbb{R}^n \quad \text{和} \quad \mathrm{col}(A^T) \oplus \mathrm{null}(A) = \mathbb{R}^m \]

根据行秩等于列秩定理，\(\mathrm{dim}(\mathrm{col}(A)) = \mathrm{dim}(\mathrm{col}(A^T))\)。此外，根据我们之前关于直接和维度的观察，我们有

\[ n = \mathrm{dim}(\mathrm{col}(A)) + \mathrm{dim}(\mathrm{null}(A^T)) = \mathrm{dim}(\mathrm{col}(A^T)) + \mathrm{dim}(\mathrm{null}(A^T)) \]

和

\[ m = \mathrm{dim}(\mathrm{col}(A^T)) + \mathrm{dim}(\mathrm{null}(A)) = \mathrm{dim}(\mathrm{col}(A)) + \mathrm{dim}(\mathrm{null}(A)). \]

因此，我们得出

\[ \mathrm{dim}(\mathrm{null}(A)) = m - \mathrm{rk}(A) \]

和

\[ \mathrm{dim}(\mathrm{null}(A^T)) = n - \mathrm{rk}(A). \]

这些公式被称为 *秩-零空间定理*。\(A\) 的零空间的维度称为 \(A\) 的零度。 \(\lhd\)

**外积和秩为 1 的矩阵** 设 \(\mathbf{u} = (u_1,\ldots,u_n) \in \mathbb{R}^n\) 和 \(\mathbf{v} = (v_1,\ldots,v_m) \in \mathbf{R}^m\) 是两个列向量。它们的外积 \(\idx{outer product}\xdi\) 定义为如下矩阵

\[\begin{split} \mathbf{u} \mathbf{v}^T = \begin{pmatrix} u_1 v_1 & u_1 v_2 & \cdots & u_1 v_m\\ u_2 v_1 & u_2 v_2 & \cdots & u_2 v_m\\ \vdots & \vdots & \ddots & \vdots\\ u_n v_1 & u_n v_2 & \cdots & u_n v_m \end{pmatrix} = \begin{pmatrix} | & & | \\ v_{1} \mathbf{u} & \ldots & v_{m} \mathbf{u} \\ | & & | \end{pmatrix}. \end{split}\]

这与内积 \(\mathbf{u}^T \mathbf{v}\) 不相同，内积要求 \(n = m\) 并产生一个标量。

如果 \(\mathbf{u}\) 和 \(\mathbf{v}\) 都不为零，矩阵 \(\mathbf{u} \mathbf{v}^T\) 的秩为 1。实际上，它的列都是同一个向量 \(\mathbf{u}\) 的倍数。因此，由 \(\mathbf{u} \mathbf{v}^T\) 的列张成的列空间是一维的。反之，根据秩的定义，任何秩为 1 的矩阵都可以写成这种形式。

我们已经看到了许多不同的矩阵乘积的解释。这里还有另一种解释。设 \(A = (a_{ij})_{i,j} \in \mathbb{R}^{n \times k}\) 和 \(B = (b_{ij})_{i,j} \in \mathbb{R}^{k \times m}\)。用 \(\mathbf{a}_1,\ldots,\mathbf{a}_k\) 表示 \(A\) 的列，用 \(\mathbf{b}_1^T,\ldots,\mathbf{b}_k^T\) 表示 \(B\) 的行。

然后

\[\begin{align*} A B &= \begin{pmatrix} \sum_{j=1}^k a_{1j} b_{j1} & \sum_{j=1}^k a_{1j} b_{j2} & \cdots & \sum_{j=1}^k a_{1j} b_{jm}\\ \sum_{j=1}^k a_{2j} b_{j1} & \sum_{j=1}^k a_{2j} b_{j2} & \cdots & \sum_{j=1}^k a_{2j} b_{jm}\\ \vdots & \vdots & \ddots & \vdots\\ \sum_{j=1}^k a_{nj} b_{j1} & \sum_{j=1}^k a_{nj} b_{j2} & \cdots & \sum_{j=1}^k a_{nj} b_{jm} \end{pmatrix}\\ &= \sum_{j=1}^k \begin{pmatrix} a_{1j} b_{j1} & a_{1j} b_{j2} & \cdots & a_{1j} b_{jm}\\ a_{2j} b_{j1} & a_{2j} b_{j2} & \cdots & a_{2j} b_{jm}\\ \vdots & \vdots & \ddots & \vdots\\ a_{nj} b_{j1} & a_{nj} b_{j2} & \cdots & a_{nj} b_{jm} \end{pmatrix}\\ &= \sum_{j=1}^k \mathbf{a}_j \mathbf{b}_j^T. \end{align*}\]

换句话说，矩阵乘积 \(AB\) 可以解释为 \(k\) 个秩为 1 的矩阵之和，每个矩阵都是 \(A\) 的一个列与 \(B\) 的对应行的外积。

因为和的秩至多为秩的和（如前一个例子所示），所以 \(AB\) 的秩至多为 \(k\)。这与乘积的秩至多为秩的最小值的事实一致（如前一个例子所示）。

## 4.2.2\. 特征值和特征向量#

回忆一下特征值\(\idx{eigenvalue}\xdi\) 和特征向量\(\idx{eigenvector}\xdi\) 的概念。我们在 \(\mathbb{R}^d\) 上工作。

**定义：** **（特征值和特征向量）** 设 \(A \in \mathbb{R}^{d \times d}\) 为一个方阵。那么 \(\lambda \in \mathbb{R}\) 是 \(A\) 的一个特征值，如果存在一个非零向量 \(\mathbf{x} \neq \mathbf{0}\) 使得

\[ A \mathbf{x} = \lambda \mathbf{x}. \]

向量 \(\mathbf{x}\) 被称为特征向量。 \(\natural\)

如下例所示，并非每个矩阵都有一个（实）特征值。

**例：** **（没有实特征值）** 设 \(d = 2\) 并令

\[\begin{split} A = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}. \end{split}\]

对于 \(\lambda\) 来说是特征值，必须存在一个非零特征向量 \(\mathbf{x} = (x_1, x_2)\) 使得

\[ A \mathbf{x} = \lambda \mathbf{x} \]

或者换一种说法，\(- x_2 = \lambda x_1\) 和 \(x_1 = \lambda x_2\)。将这些方程代入对方程，必须得到 \(- x_2 = \lambda² x_2\) 和 \(x_1 = - \lambda² x_1\)。因为 \(x_1, x_2\) 不能同时为 \(0\)，\(\lambda\) 必须满足方程 \(\lambda² = -1\)，而对于这个方程没有实数解。 \(\lhd\)

通常，\(A \in \mathbb{R}^{d \times d}\) 至多有 \(d\) 个不同的特征值。

**引理：** **（特征值的数量）** \(\idx{number of eigenvalues lemma}\xdi\) 设 \(A \in \mathbb{R}^{d \times d}\) 且 \(\lambda_1, \ldots, \lambda_m\) 是 \(A\) 的不同特征值，对应特征向量 \(\mathbf{x}_1, \ldots, \mathbf{x}_m\)。那么 \(\mathbf{x}_1, \ldots, \mathbf{x}_m\) 是线性无关的。特别是，\(m \leq d\)。 \(\flat\)

*证明：* 假设通过反证法，\(\mathbf{x}_1, \ldots, \mathbf{x}_m\) 是线性相关的。根据 *线性相关引理*，存在 \(k \leq m\) 使得

\[ \mathbf{x}_k \in \mathrm{span}(\mathbf{x}_1, \ldots, \mathbf{x}_{k-1}) \]

其中 \(\mathbf{x}_1, \ldots, \mathbf{x}_{k-1}\) 是线性无关的。特别是，存在 \(a_1, \ldots, a_{k-1}\) 使得

\[ \mathbf{x}_k = a_1 \mathbf{x}_1 + \cdots + a_{k-1} \mathbf{x}_{k-1}. \]

以两种方式变换上述方程：（1）两边乘以 \(\lambda_k\)，（2）应用 \(A\)。然后减去得到的方程。这导致

\[ \mathbf{0} = a_1 (\lambda_k - \lambda_1) \mathbf{x}_1 + \cdots + a_{k-1} (\lambda_k - \lambda_{k-1}) \mathbf{x}_{k-1}. \]

因为 \(\lambda_i\) 是不同的，且 \(\mathbf{x}_1, \ldots, \mathbf{x}_{k-1}\) 是线性无关的，所以必须有 \(a_1 = \cdots = a_{k-1} = 0\)。但这意味着 \(\mathbf{x}_k = \mathbf{0}\)，这是矛盾的。

对于第二个断言，如果有超过 \(d\) 个不同的特征值，那么根据第一个断言，会有超过 \(d\) 个对应的线性无关的特征向量，这是矛盾的。 \(\square\)

**示例：** **（对角矩阵（和相似矩阵）**）回忆一下，我们使用 \(\mathrm{diag}(\lambda_1,\ldots,\lambda_d)\) 表示对角矩阵，其对角元素为 \(\lambda_1,\ldots,\lambda_d\)。在 *特征值数量引理* 中的上界可以通过具有不同对角元素的对角矩阵来实现，例如 \(A = \mathrm{diag}(\lambda_1, \ldots, \lambda_d)\)。然后，每个标准基向量 \(\mathbf{e}_i\) 就是一个特征向量。

\[ A \mathbf{e}_i = \lambda_i \mathbf{e}_i. \]

更一般地，设 \(A\) 与具有不同对角元素的对角矩阵 \(D = \mathrm{diag}(\lambda_1, \ldots, \lambda_d)\) 相似，即存在一个非奇异矩阵 \(P\)，使得

\[ A = P D P^{-1}. \]

设 \(\mathbf{p}_1, \ldots, \mathbf{p}_d\) 是 \(P\) 的列。注意，因为 \(P\) 的列构成了 \(\mathbb{R}^d\) 的一个基，向量 \(\mathbf{c} = P^{-1} \mathbf{x}\) 的元素是 \(\mathbf{p}_i\) 的唯一线性组合的系数，等于 \(\mathbf{x}\)。实际上，\(P \mathbf{c} = \mathbf{x}\)。因此，\(A \mathbf{x}\) 可以被认为是：（1）用 \(\mathbf{p}_1, \ldots, \mathbf{p}_d\) 的基表示 \(\mathbf{x}\)，（2）并将 \(\mathbf{p}_i\) 缩放为相应的 \(\lambda_i\)。特别是，\(\mathbf{p}_i\) 是 \(A\) 的特征向量，因为，根据上述内容，\(P^{-1} \mathbf{p}_i = \mathbf{e}_i\)，所以

\[ A \mathbf{p}_i = P D P^{-1} \mathbf{p}_i = P D \mathbf{e}_i = P (\lambda_i \mathbf{e}_i) = \lambda_i \mathbf{p}_i. \]

\(\lhd\)

**数值角落：** 在 Numpy 中，可以使用 `numpy.linalg.eig` 计算矩阵的特征值和特征向量。[`numpy.linalg.eig`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html)。

```py
A = np.array([[2.5, -0.5], [-0.5, 2.5]])
w, v = LA.eig(A)
print(w)
print(v) 
```

```py
[3\. 2.]
[[ 0.70710678  0.70710678]
 [-0.70710678  0.70710678]] 
```

在这里，`w` 是数组中的特征值，而 `v` 的列是相应的特征向量。

\(\unlhd\)

**一些矩阵代数** 我们将需要一些关于矩阵的有用观察。

一个（不一定为方阵）的矩阵 \(D \in \mathbb{R}^{k \times r}\) 是对角矩阵，如果其非对角元素为零。也就是说，\(i \neq j\) 意味着 \(D_{ij} =0\)。请注意，对角矩阵不一定是方阵，并且对角元素允许为零。

乘以一个对角矩阵的矩阵具有非常特定的效果。设 \(A \in \mathbb{R}^{n \times k}\) 和 \(B \in \mathbb{R}^{r \times m}\)。我们在这里关注 \(k \geq r\) 的情况。矩阵乘积 \(A D\) 产生一个矩阵，其列是 \(A\) 的列的线性组合，其中系数来自 \(D\) 的对应列。但 \(D\) 的列最多只有一个非零元素，即对角元素。因此，\(AD\) 的列实际上是 \(A\) 的列的倍数。

\[\begin{split} AD = \begin{pmatrix} | & & | \\ d_{11} \mathbf{a}_1 & \ldots & d_{rr} \mathbf{a}_r \\ | & & | \end{pmatrix} \end{split}\]

其中 \(\mathbf{a}_1,\ldots,\mathbf{a}_k\) 是 \(A\) 的列，\(d_{11},\ldots,d_{rr}\) 是 \(D\) 的对角线元素。

类似地，\(D B\) 的行是 \(B\) 的行的线性组合，其中系数来自 \(D\) 的对应行。\(D\) 的行最多只有一个非零元素，即对角线元素。在这种情况下，当 \(k \geq r\) 时，行 \(r+1,\ldots,k\) 必定只有零项，因为那里没有对角线元素。因此，\(DB\) 的前 \(r\) 行是 \(B\) 的行的倍数，接下来的 \(k-r\) 行是 \(\mathbf{0}\)。

\[\begin{split} DB = \begin{pmatrix} \horz & d_{11} \mathbf{b}_1^T & \horz \\ & \vdots &\\ \horz & d_{rr} \mathbf{b}_r^T & \horz\\ \horz & \mathbf{0} & \horz\\ & \vdots &\\ \horz & \mathbf{0} & \horz \end{pmatrix} \end{split}\]

其中 \(\mathbf{b}_1^T,\ldots,\mathbf{b}_r^T\) 是 \(B\) 的行。

**示例：** 以下特殊情况将在本章的后面部分有用。假设 \(D, F \in \mathbb{R}^{n \times n}\) 都是方阵对角矩阵。那么 \(D F\) 是对角元素为 \(d_{11} f_{11}, \ldots,d_{nn} f_{nn}\) 的矩阵。\(\lhd\)

**谱定理** 当 \(A\) 是对称矩阵时，一个显著的结果是存在一个由 \(A\) 的特征向量构成的 \(\mathbb{R}^d\) 的正交基。我们将在下一章证明这个结果。

**知识检查：** 设 \(A \in \mathbb{R}^{d \times d}\) 是对称矩阵。证明对于 \(\mathbb{R}^d\) 中的任意 \(\mathbf{u}\) 和 \(\mathbf{v}\)，以下等式成立：

\[ \langle \mathbf{u}, A \mathbf{v} \rangle = \langle A \mathbf{u}, \mathbf{v} \rangle. \]

\(\checkmark\)

在正式陈述结果之前，我们做一些观察。设 \(A \in \mathbb{R}^{d \times d}\) 是对称矩阵。假设 \(\mathbf{v}_i\) 和 \(\mathbf{v}_j\) 分别对应于不同的特征值 \(\lambda_i\) 和 \(\lambda_j\) 的特征向量。那么以下量可以用两种方式表示

\[ \langle \mathbf{v}_i, A \mathbf{v}_j \rangle = \langle \mathbf{v}_i, \lambda_j \mathbf{v}_j \rangle = \lambda_j \langle \mathbf{v}_i, \mathbf{v}_j \rangle \]

并且，由于 \(A\) 的对称性，

\[ \langle \mathbf{v}_i, A \mathbf{v}_j \rangle = \mathbf{v}_i^T A \mathbf{v}_j = \mathbf{v}_i^T A^T \mathbf{v}_j = (A \mathbf{v}_i)^T \mathbf{v}_j = \langle A \mathbf{v}_i, \mathbf{v}_j \rangle = \langle \lambda_i \mathbf{v}_i, \mathbf{v}_j \rangle = \lambda_i \langle \mathbf{v}_i, \mathbf{v}_j \rangle. \]

相减，

\[ (\lambda_j - \lambda_i) \langle \mathbf{v}_i, \mathbf{v}_j \rangle = 0 \]

并且利用 \(\lambda_i \neq \lambda_j\)

\[ \langle \mathbf{v}_i, \mathbf{v}_j \rangle = 0. \]

即，\(\mathbf{v}_i\) 和 \(\mathbf{v}_j\) 必定是正交的。

我们证明了：

**引理** 设 \(A \in \mathbb{R}^{d \times d}\) 是对称矩阵。假设 \(\mathbf{v}_i\) 和 \(\mathbf{v}_j\) 是对应于不同特征值的特征向量。那么 \(\mathbf{v}_i\) 和 \(\mathbf{v}_j\) 是正交的。\(\flat\)

这个引理给出了一个不同的证明——在对称情况下——特征值的数量最多为 \(d\)，因为成对的正交向量是线性无关的。

事实上：

**定理** **(谱定理**) \(\idx{谱定理}\xdi\) 设 \(A \in \mathbb{R}^{d \times d}\) 为一个对称矩阵，即 \(A^T = A\)。那么 \(A\) 有 \(d\) 个正交特征向量 \(\mathbf{q}_1, \ldots, \mathbf{q}_d\)，对应（不一定不同的）实特征值 \(\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d\)。此外，\(A\) 可以表示为矩阵分解

\[ A = Q \Lambda Q^T = \sum_{i=1}^d \lambda_i \mathbf{q}_i \mathbf{q}_i^T \]

其中 \(Q\) 的列是 \(\mathbf{q}_1, \ldots, \mathbf{q}_d\)，且 \(\Lambda = \mathrm{diag}(\lambda_1, \ldots, \lambda_d)\)。\(\sharp\)

我们将这种分解称为 \(A\) 的谱分解\(\idx{谱分解}\xdi\)。

注意，这种分解确实产生了 \(A\) 的特征向量。对于任意的 \(j\)，我们有

\[ A \mathbf{q}_j = \sum_{i=1}^d \lambda_i \mathbf{q}_i \mathbf{q}_i^T \mathbf{q}_j = \lambda_j \mathbf{q}_j, \]

其中我们使用了正交性，即当 \(i \neq j\) 时 \(\mathbf{q}_i^T \mathbf{q}_j = 0\)，当 \(i = j\) 时 \(\mathbf{q}_i^T \mathbf{q}_j = 1\)。上述方程恰好表明 \(\mathbf{q}_j\) 是 \(A\) 的一个特征向量，对应的特征值为 \(\lambda_j\)。由于我们已经找到了 \(d\) 个特征值（可能重复），我们已经找到了所有特征值。

设 \(\lambda_1, \lambda_2, \ldots, \lambda_d\) 是由谱定理保证存在的特征值。我们首先论证不存在其他特征值。实际上，假设 \(\mu \neq \lambda_1, \lambda_2, \ldots, \lambda_d\) 是一个特征值，对应的特征向量为 \(\mathbf{p}\)。我们已经看到 \(\mathbf{p}\) 与特征向量 \(\mathbf{q}_1, \ldots, \mathbf{q}_d\) 正交。由于后者构成了 \(\mathbb{R}^d\) 的正交基，这种情况不可能发生，我们得到了一个矛盾。

一些特征值 \(\lambda_1, \lambda_2, \ldots, \lambda_d\) 可以重复，也就是说，存在 \(i, j\) 使得 \(\lambda_{i} = \lambda_{j}\)。例如，如果 \(A = I_{d \times d}\) 是单位矩阵，那么特征值对所有 \(i \in [d]\) 都是 \(\lambda_i = 1\)。

对于 \(A\) 的一个固定特征值 \(\lambda\)，具有特征值 \(\lambda\) 的特征向量集合满足

\[ A \mathbf{v} = \lambda \mathbf{v} = \lambda I_{d \times d} \mathbf{v} \]

或者，重新排列，

\[ (A - \lambda I_{d \times d})\mathbf{v} = \mathbf{0}. \]

换句话说，它是集合 \(\mathrm{null}(A - \lambda I_{d \times d})\)。

在 *谱定理* 中对应相同特征值 \(\lambda\) 的特征向量可以被 \(\mathrm{null}(A - \lambda I_{d \times d})\) 子空间的任意正交基所替代。但是，除了这种自由度之外，我们还有以下特征：设 \(\mu_1,\ldots,\mu_f\) 是 \(\lambda_1, \lambda_2, \ldots, \lambda_d\) 中的唯一值

\[ \mathbb{R}^d = \mathrm{null}(A - \mu_1 I_{d \times d}) \oplus \mathrm{null}(A - \mu_2 I_{d \times d}) \oplus \cdots \oplus \mathrm{null}(A - \mu_f I_{d \times d}), \]

其中我们使用了对于任何 \(\mathbf{u} \in \mathrm{null}(A - \mu_i I_{d \times d})\) 和 \(\mathbf{v} \in \mathrm{null}(A - \mu_j I_{d \times d})\)（其中 \(i \neq j\)），我们有 \(\mathbf{u}\) 与 \(\mathbf{v}\) 正交的事实。

我们特别证明了在 *谱定理* 中的特征值序列是唯一的（计数重复）。

两个矩阵 \(B, D \in \mathbb{R}^{d \times d}\) 是 [相似](https://en.wikipedia.org/wiki/Matrix_similarity)\(\idx{相似}\xdi\) 的，如果存在一个可逆矩阵 \(P\) 使得 \(B = P^{-1} D P\)。可以证明 \(B\) 和 \(D\) 对应于相同的线性映射，但表示在不同的基中。当 \(P = Q\) 是一个正交矩阵时，变换简化为 \(B = Q^T D Q\)。

因此，关于谱分解的另一种思考方式是，它表达了任何对称矩阵都通过正交变换相似于对角矩阵的事实。在这个基中，相应的线性映射由对角矩阵表示，这个基是特征向量基。

**示例：** **(2 \times 2 对称矩阵的特征分解)** 最简单的非平凡情况是 \(2 \times 2\) 对称矩阵

\[\begin{split} A = \begin{pmatrix} a & b\\ b & d \end{pmatrix}. \end{split}\]

我们推导出一个逐步的配方来计算其特征值和特征向量。

如前所述，一个特征值 \(\lambda\) 对应于非空 \(\mathrm{null}(A - \lambda I_{2 \times 2})\)，相应的特征向量求解

\[\begin{split} \mathbf{0} = (A - \lambda I_{2 \times 2})\mathbf{v} = \begin{pmatrix}a - \lambda & b\\ b & d - \lambda\end{pmatrix}. \end{split}\]

换句话说，矩阵 \(\begin{pmatrix}a - \lambda & b\\ b & d - \lambda\end{pmatrix}\) 的列线性相关。我们已经看到，检查这一点的其中一种方法就是计算行列式，在 \(2 \times 2\) 的情况下，这仅仅是

\[\begin{split} \mathrm{det}\left[\begin{pmatrix}a - \lambda & b\\ b & d - \lambda\end{pmatrix}\right] = (a - \lambda ) (d - \lambda) - b². \end{split}\]

这是一个关于 \(\lambda\) 的二次多项式，称为矩阵 \(A\) 的特征多项式。

特征多项式的根，即

\[ 0 = (a - \lambda ) (d - \lambda) - b² = \lambda² - (a + d)\lambda + (ad - b²), \]

是

\[ \lambda_{1} = \frac{(a + d) + \sqrt{(a + d)² - 4(ad - b²)}}{2} \]

和

\[ \lambda_{2} = \frac{(a + d) - \sqrt{(a + d)² - 4(ad - b²)}}{2}. \]

展开平方根中的表达式

\[ (a + d)² - 4(ad - b²) = a² + d² - 2ad + 4b² = (a - d)² + 4b², \]

我们看到，对于任何 \(a, b, d\)，平方根是定义良好的（即产生实数值）。

剩下的任务是找到对应的特征向量 \(\mathbf{v}_{1} = (v_{1,1}, v_{1,2})\) 和 \(\mathbf{v}_2 = (v_{2,1}, v_{2,2})\)，通过求解 \(2 \times 2\) 的线性方程组

\[\begin{split} \begin{pmatrix} a - \lambda_i & b \\ b & d - \lambda_i \end{pmatrix} \begin{pmatrix} v_{i,1} \\ v_{i,2} \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix} \end{split}\]

这保证了有解。当 \(\lambda_1 = \lambda_2\) 时，需要找到两个线性无关的解。

这里有一个数值示例。考虑矩阵

\[\begin{split} A = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}. \end{split}\]

特征多项式方程是

\[ \lambda² - 6\lambda + 8 = 0. \]

特征值是

\[ \lambda_{1}, \lambda_{2} = \frac{6 \pm \sqrt{36 - 4(9-1))}}{2} = \frac{6 \pm \sqrt{4}}{2} = 4, 2. \]

然后我们求解特征向量。对于 \(\lambda_1 = 4\)

\[\begin{split} \begin{pmatrix} 3 - \lambda_1 & 1 \\ 1 & 3 - \lambda_1 \end{pmatrix} \begin{pmatrix} v_{1,1} \\ v_{1,2} \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix} \Leftrightarrow \begin{cases} - v_{1,1} + v_{1,2} = 0\\ v_{1,1} - v_{1,2} = 0 \end{cases} \end{split}\]

因此，在归一化后，我们取

\[\begin{split} \mathbf{v}_1 = \frac{1}{\sqrt{2}}\begin{pmatrix}1 \\ 1\end{pmatrix}. \end{split}\]

对于 \(\lambda_2 = 2\):

\[\begin{split} \begin{pmatrix} 3 - \lambda_2 & 1 \\ 1 & 3 - \lambda_2 \end{pmatrix} \begin{pmatrix} v_{2,1} \\ v_{2,2} \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix} \Leftrightarrow \begin{cases} v_{1,1} + v_{1,2} = 0\\ v_{1,1} + v_{1,2} = 0 \end{cases} \end{split}\]

因此，在归一化后，我们取

\[\begin{split} \mathbf{v}_2 = \frac{1}{\sqrt{2}}\begin{pmatrix}1 \\ -1\end{pmatrix}. \end{split}\]

这些是特征向量的事实可以通过手工检查（试试看！）来验证。 \(\lhd\)

更一般地，对于任何 \(A \in \mathbb{R}^{d \times d}\) 的矩阵，特征多项式 \(\mathrm{det}(A - \lambda I_{d \times d})\) 的根是 \(A\) 的特征值。我们将在下一节推导一个更有效的数值方法来计算它们。

**正定矩阵的情况** 对称矩阵的特征值虽然为实数，但可能是负数。然而，存在一个重要的特殊情况，即特征值都是非负的，即正定矩阵。 \(\idx{正定矩阵}\xdi\)

**定理** **(正定性的特征)** \(\idx{正定性的特征}\xdi\) 设 \(A \in \mathbb{R}^{d \times d}\) 为一个对称矩阵，并且设 \(A = Q \Lambda Q^T\) 为 \(A\) 的谱分解，其中 \(\Lambda = \mathrm{diag}(\lambda_1, \ldots, \lambda_d)\)。那么 \(A \succeq 0\) 当且仅当其特征值 \(\lambda_1, \ldots, \lambda_d\) 是非负的。 \(\sharp\)

*证明:* 假设 \(A \succeq 0\)。设 \(\mathbf{q}_i\) 是 \(A\) 的一个特征向量，对应特征值 \(\lambda_i\)。那么

\[ \langle \mathbf{q}_i, A \mathbf{q}_i \rangle = \langle \mathbf{q}_i, \lambda_i \mathbf{q}_i \rangle = \lambda_i \]

根据正半定矩阵的定义，这必须是非负的。

在另一个方向上，假设 \(\lambda_1, \ldots, \lambda_d \geq 0\)。那么，通过外积形式的谱分解

\[ \langle \mathbf{x}, A \mathbf{x} \rangle = \mathbf{x}^T \left(\sum_{i=1}^d \lambda_i \mathbf{q}_i \mathbf{q}_i^T\right) \mathbf{x} = \sum_{i=1}^d \lambda_i \mathbf{x}^T\mathbf{q}_i \mathbf{q}_i^T\mathbf{x} = \sum_{i=1}^d \lambda_i \langle \mathbf{q}_i, \mathbf{x}\rangle² \]

这必然是非负的。 \(\square\)

同样地，一个对称矩阵是正定的\(\idx{正定矩阵}\xdi\)当且仅当它的所有特征值都是严格正的。证明基本上是相同的。

**KNOWLEDGE CHECK:** 通过修改上面的证明来证明这个最后的陈述。 \(\checkmark\)

回想一下，正半定性的一个重要应用是作为凸性的特征\(\idx{凸函数}\xdi\)。这里有一些例子。

**EXAMPLE:** **(通过 Hessian 特征值实现的凸性**) 考虑以下函数

\[ f(x, y) = \frac{3}{2} x² + xy + \frac{3}{2} y² + 5x - 2y + 1. \]

为了证明它是凸的，我们计算其 Hessian

\[\begin{split} H_f(x,y) = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix} \end{split}\]

对于所有 \(x,y\)。根据先前的例子，其特征值是 \(2\) 和 \(4\)，都是严格正的。这通过 *二阶凸性条件* 证明了该陈述。 \(\lhd\)

**EXAMPLE:** **(对数凹性**) 如果 \(-\log f\) 是凸的，那么函数 \(f :\mathbb{R}^d \to \mathbb{R}\) 被称为对数凹的。换句话说，对于所有 \(\mathbf{x}, \mathbf{y} \in \mathbb{R}^d\) 和 \(\alpha \in (0,1)\)

\[ - \log f((1-\alpha)\mathbf{x} + \alpha \mathbf{y}) \leq - (1-\alpha) \log f(\mathbf{x}) - \alpha \log f(\mathbf{y}), \]

这等价于

\[ \log f((1-\alpha)\mathbf{x} + \alpha \mathbf{y}) \geq (1-\alpha) \log f(\mathbf{x}) + \alpha \log f(\mathbf{y}) , \]

或者，因为 \(a \log b = \log b^a\) 并且对数函数是严格递增的，

\[ f((1-\alpha)\mathbf{x} + \alpha \mathbf{y}) \geq f(\mathbf{x})^{1-\alpha} f(\mathbf{y})^{\alpha}. \]

我们将在课程中稍后看到，在 \(\mathbb{R}^d\) 上的多变量高斯向量 \(\mathbf{X}\) 具有均值 \(\bmu \in \mathbb{R}^d\) 和正定协方差矩阵 \(\bSigma \in \mathbb{R}^{d \times d}\) 的概率密度函数 (PDF)

\[ f_{\bmu, \bSigma}(\mathbf{x}) = \frac{1}{(2\pi)^{d/2} \,|\bSigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \bmu)^T \bSigma^{-1} (\mathbf{x} - \bmu)\right) \]

其中 \(|A|\) 是 \(A\) 的 [行列式](https://en.wikipedia.org/wiki/Determinant)，在对称矩阵的情况下，它仅仅是其特征值的乘积（包括重复的特征值）。我们声称这个概率密度函数是对数凹的。

从定义来看，

\[\begin{align*} &- \log f_{\bmu, \bSigma}(\mathbf{x})\\ &= \frac{1}{2}(\mathbf{x} - \bmu)^T \bSigma^{-1} (\mathbf{x} - \bmu) + \log (2\pi)^{d/2} \,|\bSigma|^{1/2}\\ &= \frac{1}{2}\mathbf{x}^T \bSigma^{-1} \mathbf{x} - \bmu^T \bSigma^{-1} \mathbf{x} + \left[\frac{1}{2}\bmu^T \bSigma^{-1} \bmu + \log (2\pi)^{d/2} \,|\bSigma|^{1/2}\right]. \end{align*}\]

设 \(P = \bSigma^{-1}\)，\(\mathbf{q} = - \bmu^T \bSigma^{-1}\) 和 \(r = \frac{1}{2}\bmu^T \bSigma^{-1} \bmu + \log (2\pi)^{d/2} \,|\bSigma|^{1/2}\)。

之前的一个例子表明，如果 \(\bSigma^{-1}\) 是正半定，则 PDF 是对数凹的。由于 \(\bSigma\) 是正定的，根据假设，\(\bSigma = Q \Lambda Q^T\) 具有特征分解，其中 \(\Lambda\) 的所有对角元素都是严格正的。因此，\(\bSigma^{-1} = Q \Lambda^{-1} Q^T\)，其中 \(\Lambda^{-1}\) 的对角元素是 \(\Lambda\) 的逆元素——因此也是严格正的。特别是，\(\bSigma^{-1}\) 是正半定的。\(\lhd\)

**NUMERICAL CORNER:** 因此，我们可以通过使用 `numpy.linalg.eig` 计算其特征值来检查矩阵是否是正半定的。[`numpy.linalg.eig`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html)。

```py
A = np.array([[1, -1], [-1, 1]])
w, v = LA.eig(A)
print(w) 
```

```py
[2\. 0.] 
```

```py
B = np.array([[1, -2], [-2, 1]])
z, u = LA.eig(B)
print(z) 
```

```py
[ 3\. -1.] 
```

**KNOWLEDGE CHECK:** 以下哪个矩阵是正半定的？

\[\begin{split} A = \begin{pmatrix} 1 & -1\\ -1 & 1 \end{pmatrix} \qquad B = \begin{pmatrix} 1 & -2\\ -2 & 1 \end{pmatrix} \end{split}\]

a) 两者

b) \(A\)

c) \(B\)

d) 都不是

\(\checkmark\)

\(\unlhd\)

***自我评估测验*** *(有 Claude、Gemini 和 ChatGPT 的帮助)*

**1** 矩阵 \(A \in \mathbb{R}^{n \times m}\) 的秩是什么？

a) \(A\) 中非零条目的数量

b) \(A\) 的行空间的维度

c) \(A\) 的零空间的维度

d) \(A\) 的迹

**2** 关于矩阵 \(A \in \mathbb{R}^{n \times m}\) 的秩，以下哪个是正确的？

a) \(\mathrm{rk}(A) \leq \min\{n,m\}\)

b) \(\mathrm{rk}(A) \geq \max\{n,m\}\)

c) \(\mathrm{rk}(A) = \mathrm{rk}(A^T)\) 仅当 \(A\) 是对称的

d) \(\mathrm{rk}(A) = \mathrm{rk}(A^T)\) 仅当 \(A\) 是方阵

**3** 设 \(A \in \mathbb{R}^{n \times k}\) 和 \(B \in \mathbb{R}^{k \times m}\)。一般情况下，以下哪个是正确的？

a) \(\mathrm{rk}(AB) \leq \mathrm{rk}(A)\)

b) \(\mathrm{rk}(AB) \geq \mathrm{rk}(A)\)

c) \(\mathrm{rk}(AB) = \mathrm{rk}(A)\)

d) \(\mathrm{rk}(AB) = \mathrm{rk}(B)\)

**4** 设 \(A \in \mathbb{R}^{d \times d}\) 是对称的。根据谱定理，以下哪个是正确的？

a) \(A\) 至多有 \(d\) 个不同的特征值

b) \(A\) 恰好有 \(d\) 个不同的特征值

c) \(A\) 至少有 \(d\) 个不同的特征值

d) \(A\) 的不同特征值的数量与 \(d\) 无关

**5** 关于两个向量 \(\mathbf{u}\) 和 \(\mathbf{v}\) 的外积，以下哪个是正确的？

a) 它是一个标量。

b) 它是一个向量。

c) 它是一个秩为一的矩阵

d) 它是一个秩为零的矩阵

1 的答案：b. 理由：文本声明“\(A\) 的行秩和列秩简单地[是]秩，我们用 \(\mathrm{rk}(A)\) 表示。”

2 的答案：a. 理由：在行秩等于列秩定理中，文本声明“\(A\) 的行秩等于 \(A\) 的列秩。此外，\(\mathrm{rk}(A) \leq \min\{n,m\}\)。”

3 的答案：a. 理由：文本表明“\(AB\) 的列是 \(A\) 的列的线性组合。因此 \(\mathrm{col}(AB) \subseteq \mathrm{col}(A)\)。根据观察 D2，可以得出这个结论。”

4 的答案：a. 理由：谱定理声明“对称矩阵 \(A\) 有 \(d\) 个正交归一特征向量 \(\mathbf{q}_1, \ldots, \mathbf{q}_d\)，对应（不一定不同的）实特征值 \(\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d\)。”

5 的答案：c. 理由：文本定义了外积并声明“如果 \(\mathbf{u}\) 和 \(\mathbf{v}\) 都不为零，矩阵 \(\mathbf{u} \mathbf{v}^T\) 的秩为 1。”

## 4.2.1\. 矩阵的秩#

回想一下，\(A\) 的列空间的维度称为 \(A\) 的列秩。类似地，\(A\) 的行秩是其行空间的维度。实际上，这两个秩的概念通过 *行秩等于列秩定理*\(\idx{row rank equals column rank theorem}\xdi\) 是相同的。我们将在下面给出该定理的证明。我们简单地将 \(A\) 的行秩和列秩称为秩，我们用 \(\mathrm{rk}(A)\) 表示。\(\idx{rank}\xdi\)

*证明思路:* *(行秩等于列秩)* 将 \(A\) 写成矩阵分解 \(BC\)，其中 \(B\) 的列构成 \(\mathrm{col}(A)\) 的基。然后 \(C\) 的行必然构成 \(\mathrm{row}(A)\) 的一个生成集。因此，因为 \(B\) 的列数和 \(C\) 的行数相等，我们得出行秩小于或等于列秩。将相同的论证应用于转置矩阵，得出结论。

回想一下前一章中的以下观察。

**观察 D1:** \(\mathbb{R}^n\) 的任何线性子空间 \(U\) 都有一个基。

**观察 D2:** 如果 \(U\) 和 \(V\) 是满足 \(U \subseteq V\) 的线性子空间，那么 \(\mathrm{dim}(U) \leq \mathrm{dim}(V)\)。

**观察 D3:** \(\mathrm{span}(\mathbf{u}_1,\ldots,\mathbf{u}_m)\) 的维度最多为 \(m\)。

*证明:* *(行秩等于列秩)* 假设 \(A\) 的列秩为 \(r\)。根据上面的 *观察 D1*，存在一个基 \(\mathbf{b}_1,\ldots, \mathbf{b}_r \in \mathbb{R}^n\)，它是 \(\mathrm{col}(A)\) 的基，并且我们知道根据 *观察 D2*，\(r \leq n\)。也就是说，对于每个 \(j\)，令 \(\mathbf{a}_{j} = A_{\cdot,j}\) 为 \(A\) 的第 \(j\) 列，我们可以写成

\[ \mathbf{a}_{j} = \sum_{\ell=1}^r \mathbf{b}_\ell c_{\ell j} \]

对于某些 \(c_{\ell j}\) 的值。设 \(B\) 是列由 \(\mathbf{b}_1, \ldots, \mathbf{b}_r\) 组成的矩阵，设 \(C\) 是元素为 \((C)_{\ell j} = c_{\ell j}\)，\(\ell=1,\ldots,r\)，\(j=1,\ldots,m\) 的矩阵。那么上述方程可以重写为矩阵分解 \(A = BC\)。确实，根据我们之前关于矩阵-矩阵乘法的观察，\(A\) 的列是 \(B\) 的列的线性组合，系数来自 \(C\) 的对应列。

关键点如下：\(C\) 必然有 \(r\) 行。设 \(\boldsymbol{\alpha}_{i}^T = A_{i,\cdot}\) 为 \(A\) 的第 \(i\) 行，\(\mathbf{c}_{\ell}^T = C_{\ell,\cdot}\) 为 \(C\) 的第 \(\ell\) 行。使用我们关于矩阵-矩阵乘法的行表示法，分解等价于

\[ \boldsymbol{\alpha}_{i}^T = \sum_{\ell=1}^r b_{i\ell} \mathbf{c}_\ell^T, \quad i=1,\ldots, n, \]

其中 \(b_{i\ell} = (\mathbf{b}_i)_\ell = (B)_{i\ell}\) 是 \(B\) 的第 \(\ell\) 列的第 \(i\) 个元素。换句话说，\(A\) 的行是 \(C\) 的行的线性组合，系数来自 \(B\) 的对应行。特别是，\(\mathcal{C} = \{\mathbf{c}_{j}:j=1,\ldots,r\}\) 是 \(A\) 的行空间的生成集，即 \(A\) 的每一行都可以写成 \(\mathcal{C}\) 的线性组合。换句话说，\(\mathrm{row}(A) \subseteq \mathrm{span}(\mathcal{C})\)。

因此，\(A\) 的行秩至多为 \(r\)，\(A\) 的列秩，根据 *观察 D2*。

将相同的论点应用于 \(A^T\)，它交换了列和行的角色，得出 \(A\) 的列秩（即 \(A^T\) 的行秩）至多为 \(A\) 的行秩（即 \(A^T\) 的列秩）。因此，这两个秩的概念必须相等。（我们再次通过 *观察 D2* 推导出 \(r \leq m\)。）\(\square\)

**EXAMPLE:** **(continued)** 我们说明了定理的证明。继续先前的例子，设 \(A\) 是列由 \(\mathbf{w}_1 = (1,0,1)\)，\(\mathbf{w}_2 = (0,1,1)\)，和 \(\mathbf{w}_3 = (1,-1,0)\) 组成的矩阵

\[\begin{split} A = \begin{pmatrix} 1 & 0 & 1\\ 0 & 1 & -1\\ 1 & 1 & 0 \end{pmatrix}. \end{split}\]

我们知道 \(\mathbf{w}_1\) 和 \(\mathbf{w}_2\) 构成了 \(\mathrm{col}(A)\) 的基。我们使用它们来构造我们的矩阵 \(B\)

\[\begin{split} B = \begin{pmatrix} 1 & 0\\ 0 & 1\\ 1 & 1 \end{pmatrix}. \end{split}\]

回想一下，\(\mathbf{w}_3 = \mathbf{w}_1 - \mathbf{w}_2\)。因此，矩阵 \(C\) 是

\[\begin{split} C = \begin{pmatrix} 1 & 0 & 1\\ 0 & 1 & -1 \end{pmatrix}. \end{split}\]

事实上，\(C\) 的第 \(j\) 列给出了产生 \(A\) 的第 \(j\) 列的 \(B\) 的列的线性组合中的系数。检查 \(A = B C\)。\(\lhd\)

**NUMERICAL CORNER:** 在 Numpy 中，可以使用函数 `numpy.linalg.matrix_rank` 计算矩阵的秩。我们将在本章后面看到如何使用奇异值分解来计算它（这是 `LA.matrix_rank` 所做的）。让我们尝试上面的例子。

```py
w1 = np.array([1., 0., 1.])
w2 = np.array([0., 1., 1.])
w3 = np.array([1., -1., 0.])
A = np.stack((w1, w2, w3), axis=-1)
print(A) 
```

```py
[[ 1\.  0\.  1.]
 [ 0\.  1\. -1.]
 [ 1\.  1\.  0.]] 
```

我们计算 `A` 的秩。

```py
LA.matrix_rank(A) 
```

```py
2 
```

我们这次只取 `A` 的前两列来形成 `B`。

```py
B = np.stack((w1, w2),axis=-1)
print(B) 
```

```py
[[1\. 0.]
 [0\. 1.]
 [1\. 1.]] 
```

```py
LA.matrix_rank(B) 
```

```py
2 
```

回忆一下，在 Numpy 中，`@` 用于矩阵乘法。

```py
C = np.array([[1., 0., 1.],[0., 1., -1.]])
print(C) 
```

```py
[[ 1\.  0\.  1.]
 [ 0\.  1\. -1.]] 
```

```py
LA.matrix_rank(C) 
```

```py
2 
```

```py
print(B @ C) 
```

```py
[[ 1\.  0\.  1.]
 [ 0\.  1\. -1.]
 [ 1\.  1\.  0.]] 
```

\(\unlhd\)

**EXAMPLE:** 令 \(A \in \mathbb{R}^{n \times k}\) 和 \(B \in \mathbb{R}^{k \times m}\)。那么我们声称：

\[ \mathrm{rk}(AB) \leq \mathrm{rk}(A). \]

事实上，\(AB\) 的列是 \(A\) 的列的线性组合。因此 \(\mathrm{col}(AB) \subseteq \mathrm{col}(A)\)。根据 *观察 D2*，这个结论成立。 \(\lhd\)

**EXAMPLE:** 令 \(A \in \mathbb{R}^{n \times k}\) 和 \(B \in \mathbb{R}^{n \times m}\)。那么我们声称：

\[ \mathrm{rk}(A + B) \leq \mathrm{rk}(A) + \mathrm{rk}(B). \]

事实上，\(A + B\) 的列是 \(A\) 和 \(B\) 的列的线性组合。设 \(\mathbf{u}_1,\ldots,\mathbf{u}_h\) 是 \(\mathrm{col}(A)\) 的一个基，设 \(\mathbf{v}_1,\ldots,\mathbf{v}_{\ell}\) 是 \(\mathrm{col}(B)\) 的一个基，根据 *观察 D1*。然后，我们得出：

\[ \mathrm{col}(A + B) \subseteq \mathrm{span}(\mathbf{u}_1,\ldots,\mathbf{u}_h,\mathbf{v}_1,\ldots,\mathbf{v}_{\ell}). \]

根据 *观察 D2*，可以得出：

\[ \mathrm{rk}(A + B) \leq \mathrm{dim}(\mathrm{span}(\mathbf{u}_1,\ldots,\mathbf{u}_h,\mathbf{v}_1,\ldots,\mathbf{v}_{\ell})). \]

根据 *观察 D3*，右边至多是生成集的长度，即 \(h+\ell\)。但是根据构造，\(\mathrm{rk}(A) = h\) 和 \(\mathrm{rk}(B) = \ell\)，所以我们完成了。 \(\lhd\)

**EXAMPLE:** **（秩-零度定理的证明）** \(\idx{rank-nullity theorem}\xdi\) 令 \(A \in \mathbb{R}^{n \times m}\)。回忆一下，\(A\) 的列空间 \(\mathrm{col}(A) \subseteq \mathbb{R}^n\) 是其列的生成空间。我们计算其正交补。根据定义，\(A\) 的列，我们用 \(\mathbf{a}_1,\ldots,\mathbf{a}_m\) 表示，形成 \(\mathrm{col}(A)\) 的生成集。因此，如果且仅当 \(\mathbf{u} \in \mathrm{col}(A)^\perp \subseteq \mathbb{R}^n\) 时，\(\mathbf{u}\) 属于 \(\mathrm{col}(A)^\perp\)：

\[ \mathbf{a}_i^T\mathbf{u} = \langle \mathbf{u}, \mathbf{a}_i \rangle = 0, \quad \forall i=1,\ldots,m. \]

确实，这表明对于任何 \(\mathbf{v} \in \mathrm{col}(A)\)，例如 \(\mathbf{v} = \beta_1 \mathbf{a}_1 + \cdots + \beta_m \mathbf{a}_m\)，我们有：

\[ \left\langle \mathbf{u}, \sum_{i=1}^m \beta_i \mathbf{a}_i \right\rangle = \sum_{i=1}^m \beta_i \langle \mathbf{u}, \mathbf{a}_i \rangle = 0. \]

上述 \(m\) 个条件可以写成矩阵形式：

\[ A^T \mathbf{u} = \mathbf{0}. \]

那就是，\(A\) 的列空间的正交补是 \(A^T\) 的零空间

\[ \mathrm{col}(A)^\perp = \mathrm{null}(A^T). \]

将相同的论点应用于 \(A^T\) 的列空间，可以得出

\[ \mathrm{col}(A^T)^\perp = \mathrm{null}(A), \]

注意到 \(\mathrm{null}(A) \subseteq \mathbb{R}^m\)。这四个线性子空间 \(\mathrm{col}(A)\)，\(\mathrm{col}(A^T)\)，\(\mathrm{null}(A)\) 和 \(\mathrm{null}(A^T)\) 被称为 \(A\) 的**基本子空间**。我们已经证明

\[ \mathrm{col}(A) \oplus \mathrm{null}(A^T) = \mathbb{R}^n \quad \text{和} \quad \mathrm{col}(A^T) \oplus \mathrm{null}(A) = \mathbb{R}^m \]

根据行秩等于列秩定理，\(\mathrm{dim}(\mathrm{col}(A)) = \mathrm{dim}(\mathrm{col}(A^T))\)。此外，根据我们之前关于直接和维度的观察，我们有

\[ n = \mathrm{dim}(\mathrm{col}(A)) + \mathrm{dim}(\mathrm{null}(A^T)) = \mathrm{dim}(\mathrm{col}(A^T)) + \mathrm{dim}(\mathrm{null}(A^T)) \]

并且

\[ m = \mathrm{dim}(\mathrm{col}(A^T)) + \mathrm{dim}(\mathrm{null}(A)) = \mathrm{dim}(\mathrm{col}(A)) + \mathrm{dim}(\mathrm{null}(A)). \]

因此我们得出结论：

\[ \mathrm{dim}(\mathrm{null}(A)) = m - \mathrm{rk}(A) \]

并且

\[ \mathrm{dim}(\mathrm{null}(A^T)) = n - \mathrm{rk}(A). \]

这些公式被称为**秩-零度定理**。\(A\) 的零空间的维度被称为 \(A\) 的**零度**。 \(\lhd\)

**外积和秩为 1 的矩阵** 设 \(\mathbf{u} = (u_1,\ldots,u_n) \in \mathbb{R}^n\) 和 \(\mathbf{v} = (v_1,\ldots,v_m) \in \mathbf{R}^m\) 是两个列向量。它们的**外积**定义为矩阵

\[\begin{split} \mathbf{u} \mathbf{v}^T = \begin{pmatrix} u_1 v_1 & u_1 v_2 & \cdots & u_1 v_m\\ u_2 v_1 & u_2 v_2 & \cdots & u_2 v_m\\ \vdots & \vdots & \ddots & \vdots\\ u_n v_1 & u_n v_2 & \cdots & u_n v_m \end{pmatrix} = \begin{pmatrix} | & & | \\ v_{1} \mathbf{u} & \ldots & v_{m} \mathbf{u} \\ | & & | \end{pmatrix}. \end{split}\]

这与内积 \(\mathbf{u}^T \mathbf{v}\) 不应混淆，它要求 \(n = m\) 并产生一个标量。

如果 \(\mathbf{u}\) 和 \(\mathbf{v}\) 都不为零，矩阵 \(\mathbf{u} \mathbf{v}^T\) 的秩为 1。确实，它的列都是同一个向量 \(\mathbf{u}\) 的倍数。因此，由 \(\mathbf{u} \mathbf{v}^T\) 的列张成的列空间是一维的。反之，任何秩为 1 的矩阵都可以通过秩的定义写成这种形式。

我们已经看到了许多不同的矩阵乘积的解释。这里还有另一个。设 \(A = (a_{ij})_{i,j} \in \mathbb{R}^{n \times k}\) 和 \(B = (b_{ij})_{i,j} \in \mathbb{R}^{k \times m}\)。用 \(\mathbf{a}_1,\ldots,\mathbf{a}_k\) 表示 \(A\) 的列，用 \(\mathbf{b}_1^T,\ldots,\mathbf{b}_k^T\) 表示 \(B\) 的行。

然后

\[\begin{align*} A B &= \begin{pmatrix} \sum_{j=1}^k a_{1j} b_{j1} & \sum_{j=1}^k a_{1j} b_{j2} & \cdots & \sum_{j=1}^k a_{1j} b_{jm}\\ \sum_{j=1}^k a_{2j} b_{j1} & \sum_{j=1}^k a_{2j} b_{j2} & \cdots & \sum_{j=1}^k a_{2j} b_{jm}\\ \vdots & \vdots & \ddots & \vdots\\ \sum_{j=1}^k a_{nj} b_{j1} & \sum_{j=1}^k a_{nj} b_{j2} & \cdots & \sum_{j=1}^k a_{nj} b_{jm} \end{pmatrix}\\ &= \sum_{j=1}^k \begin{pmatrix} a_{1j} b_{j1} & a_{1j} b_{j2} & \cdots & a_{1j} b_{jm}\\ a_{2j} b_{j1} & a_{2j} b_{j2} & \cdots & a_{2j} b_{jm}\\ \vdots & \vdots & \ddots & \vdots\\ a_{nj} b_{j1} & a_{nj} b_{j2} & \cdots & a_{nj} b_{jm} \end{pmatrix}\\ &= \sum_{j=1}^k \mathbf{a}_j \mathbf{b}_j^T. \end{align*}\]

换句话说，矩阵乘积 \(AB\) 可以解释为 \(k\) 个秩为 1 的矩阵之和，每个矩阵都是 \(A\) 的一个列向量与 \(B\) 的对应行向量的外积。

因为和的秩至多为各秩之和（如前一个示例所示），所以 \(AB\) 的秩至多为 \(k\)。这与乘积的秩至多为秩的最小值的事实一致（如前一个示例所示）。

## 4.2.2\. 特征值和特征向量#

回顾特征值\(\idx{eigenvalue}\xdi\) 和特征向量\(\idx{eigenvector}\xdi\) 的概念。我们在 \(\mathbb{R}^d\) 上工作。

**定义** **(特征值和特征向量)** 设 \(A \in \mathbb{R}^{d \times d}\) 为一个方阵。那么，如果存在一个非零向量 \(\mathbf{x} \neq \mathbf{0}\)，使得 \(\lambda \in \mathbb{R}\) 是 \(A\) 的一个特征值，则称 \(\lambda\) 为 \(A\) 的特征值。

\[ A \mathbf{x} = \lambda \mathbf{x}. \]

向量 \(\mathbf{x}\) 被称为特征向量。\(\natural\)

如下例所示，并非每个矩阵都有（实数）特征值。

**示例** **(没有实数特征值)** 设 \(d = 2\) 并令

\[\begin{split} A = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}. \end{split}\]

对于 \(\lambda\) 是一个特征值，必须存在一个非零特征向量 \(\mathbf{x} = (x_1, x_2)\)，使得

\[ A \mathbf{x} = \lambda \mathbf{x} \]

或者换一种说法，\(- x_2 = \lambda x_1\) 和 \(x_1 = \lambda x_2\)。将这些方程代入对方程，必须得到 \(- x_2 = \lambda² x_2\) 和 \(x_1 = - \lambda² x_1\)。因为 \(x_1, x_2\) 不能同时为 \(0\)，所以 \(\lambda\) 必须满足方程 \(\lambda² = -1\)，而这个方程没有实数解。\(\lhd\)

一般情况下，\(A \in \mathbb{R}^{d \times d}\) 至多有 \(d\) 个不同的特征值。

**引理** **(特征值的数量)** \(\idx{number of eigenvalues lemma}\xdi\) 设 \(A \in \mathbb{R}^{d \times d}\) 且 \(\lambda_1, \ldots, \lambda_m\) 是 \(A\) 的不同特征值，对应特征向量分别为 \(\mathbf{x}_1, \ldots, \mathbf{x}_m\)。那么 \(\mathbf{x}_1, \ldots, \mathbf{x}_m\) 是线性无关的。特别是，\(m \leq d\)。\(\flat\)

*证明:* 假设通过反证法，\(\mathbf{x}_1, \ldots, \mathbf{x}_m\) 是线性相关的。根据 *线性相关性引理*，存在 \(k \leq m\) 使得

\[ \mathbf{x}_k \in \mathrm{span}(\mathbf{x}_1, \ldots, \mathbf{x}_{k-1}) \]

其中 \(\mathbf{x}_1, \ldots, \mathbf{x}_{k-1}\) 是线性无关的。特别是，存在 \(a_1, \ldots, a_{k-1}\) 使得

\[ \mathbf{x}_k = a_1 \mathbf{x}_1 + \cdots + a_{k-1} \mathbf{x}_{k-1}. \]

以两种方式变换上述方程： (1) 将两边乘以 \(\lambda_k\)， (2) 应用 \(A\)。然后减去得到的方程。这导致

\[ \mathbf{0} = a_1 (\lambda_k - \lambda_1) \mathbf{x}_1 + \cdots + a_{k-1} (\lambda_k - \lambda_{k-1}) \mathbf{x}_{k-1}. \]

因为 \(\lambda_i\) 是不同的，且 \(\mathbf{x}_1, \ldots, \mathbf{x}_{k-1}\) 是线性无关的，我们必须有 \(a_1 = \cdots = a_{k-1} = 0\)。但这意味着 \(\mathbf{x}_k = \mathbf{0}\)，这是矛盾的。

对于第二个断言，如果有超过 \(d\) 个不同的特征值，那么根据第一个断言，将会有超过 \(d\) 个对应的线性无关的特征向量，这是矛盾的。 \(\square\)

**EXAMPLE:** **(对角（和相似）矩阵)** 回想一下，我们用 \(\mathrm{diag}(\lambda_1,\ldots,\lambda_d)\) 表示对角矩阵，其对角元素为 \(\lambda_1,\ldots,\lambda_d\)。在 *特征值数量引理* 中的上界可以通过具有不同对角元素的矩阵来实现，例如，对角矩阵 \(A = \mathrm{diag}(\lambda_1, \ldots, \lambda_d)\)。然后，每个标准基向量 \(\mathbf{e}_i\) 就是一个特征向量

\[ A \mathbf{e}_i = \lambda_i \mathbf{e}_i. \]

更一般地，设 \(A\) 与具有不同对角元素的矩阵 \(D = \mathrm{diag}(\lambda_1, \ldots, \lambda_d)\) 相似，即存在一个非奇异矩阵 \(P\) 使得

\[ A = P D P^{-1}. \]

设 \(\mathbf{p}_1, \ldots, \mathbf{p}_d\) 为 \(P\) 的列。注意，因为 \(P\) 的列构成了 \(\mathbb{R}^d\) 的一个基，向量 \(\mathbf{c} = P^{-1} \mathbf{x}\) 的元素是 \(\mathbf{p}_i\) 的唯一线性组合的系数，等于 \(\mathbf{x}\)。实际上，\(P \mathbf{c} = \mathbf{x}\)。因此，\(A \mathbf{x}\) 可以理解为： (1) 用 \(\mathbf{p}_1, \ldots, \mathbf{p}_d\) 的基表示 \(\mathbf{x}\)， (2) 并将 \(\mathbf{p}_i\) 按相应的 \(\lambda_i\) 缩放。特别是，\(\mathbf{p}_i\) 是 \(A\) 的特征向量，因为，根据上述内容，\(P^{-1} \mathbf{p}_i = \mathbf{e}_i\)，所以

\[ A \mathbf{p}_i = P D P^{-1} \mathbf{p}_i = P D \mathbf{e}_i = P (\lambda_i \mathbf{e}_i) = \lambda_i \mathbf{p}_i. \]

\(\lhd\)

**数值角落:** 在 Numpy 中，可以使用 `numpy.linalg.eig` 计算矩阵的特征值和特征向量。[`numpy.linalg.eig`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html)。

```py
A = np.array([[2.5, -0.5], [-0.5, 2.5]])
w, v = LA.eig(A)
print(w)
print(v) 
```

```py
[3\. 2.]
[[ 0.70710678  0.70710678]
 [-0.70710678  0.70710678]] 
```

在上面，`w` 是数组中的特征值，而 `v` 的列是对应的特征向量。

\(\unlhd\)

**一些矩阵代数** 我们将需要一些关于矩阵的有用观察。

一个（不一定为方阵）的矩阵 \(D \in \mathbb{R}^{k \times r}\) 如果其非对角元素为零，则是对角矩阵。也就是说，如果 \(i \neq j\)，则 \(D_{ij} =0\)。注意，对角矩阵不一定是方阵，并且对角元素可以是零。

将一个矩阵与对角矩阵相乘具有非常特定的效果。设 \(A \in \mathbb{R}^{n \times k}\) 和 \(B \in \mathbb{R}^{r \times m}\)。我们在这里关注 \(k \geq r\) 的情况。矩阵乘积 \(A D\) 产生一个矩阵，其列是 \(A\) 的列的线性组合，系数来自 \(D\) 的对应列。但 \(D\) 的列最多只有一个非零元素，即对角元素。因此，\(AD\) 的列实际上是 \(A\) 的列的倍数。

\[\begin{split} AD = \begin{pmatrix} | & & | \\ d_{11} \mathbf{a}_1 & \ldots & d_{rr} \mathbf{a}_r \\ | & & | \end{pmatrix} \end{split}\]

其中 \(\mathbf{a}_1,\ldots,\mathbf{a}_k\) 是 \(A\) 的列，\(d_{11},\ldots,d_{rr}\) 是 \(D\) 的对角元素。

同样，\(D B\) 的行是 \(B\) 的行的线性组合，系数来自 \(D\) 的对应行。\(D\) 的行最多只有一个非零元素，即对角元素。在这种情况下，如果 \(k \geq r\)，行 \(r+1,\ldots,k\) 必然只有零元素，因为那里没有对角元素。因此，\(DB\) 的前 \(r\) 行是 \(B\) 的行的倍数，接下来的 \(k-r\) 行是 \(\mathbf{0}\)。

\[\begin{split} DB = \begin{pmatrix} \horz & d_{11} \mathbf{b}_1^T & \horz \\ & \vdots &\\ \horz & d_{rr} \mathbf{b}_r^T & \horz\\ \horz & \mathbf{0} & \horz\\ & \vdots &\\ \horz & \mathbf{0} & \horz \end{pmatrix} \end{split}\]

其中 \(\mathbf{b}_1^T,\ldots,\mathbf{b}_r^T\) 是 \(B\) 的行。

**示例:** 以下特殊情况将在本章的后面部分有用。假设 \(D, F \in \mathbb{R}^{n \times n}\) 都是方阵对角矩阵。那么 \(D F\) 是对角元素为 \(d_{11} f_{11}, \ldots,d_{nn} f_{nn}\) 的矩阵。\(\lhd\)

**谱定理** 当 \(A\) 是对称矩阵时，一个显著的结果是存在一个由 \(A\) 的特征向量组成的 \(\mathbb{R}^d\) 的正交基。我们将在下一章证明这个结果。

**知识检查:** 设 \(A \in \mathbb{R}^{d \times d}\) 是对称矩阵。证明对于 \(\mathbb{R}^d\) 中的任意 \(\mathbf{u}\) 和 \(\mathbf{v}\)，都有

\[ \langle \mathbf{u}, A \mathbf{v} \rangle = \langle A \mathbf{u}, \mathbf{v} \rangle. \]

\(\checkmark\)

在正式陈述结果之前，我们做一些观察。设 \(A \in \mathbb{R}^{d \times d}\) 为对称矩阵。假设 \(\mathbf{v}_i\) 和 \(\mathbf{v}_j\) 分别对应于不同的特征值 \(\lambda_i\) 和 \(\lambda_j\) 的特征向量。那么以下量可以用两种方式表示

\[ \langle \mathbf{v}_i, A \mathbf{v}_j \rangle = \langle \mathbf{v}_i, \lambda_j \mathbf{v}_j \rangle = \lambda_j \langle \mathbf{v}_i, \mathbf{v}_j \rangle \]

并且，由于 \(A\) 的对称性，

\[ \langle \mathbf{v}_i, A \mathbf{v}_j \rangle = \mathbf{v}_i^T A \mathbf{v}_j = \mathbf{v}_i^T A^T \mathbf{v}_j = (A \mathbf{v}_i)^T \mathbf{v}_j = \langle A \mathbf{v}_i, \mathbf{v}_j \rangle = \langle \lambda_i \mathbf{v}_i, \mathbf{v}_j \rangle = \lambda_i \langle \mathbf{v}_i, \mathbf{v}_j \rangle. \]

相减，

\[ (\lambda_j - \lambda_i) \langle \mathbf{v}_i, \mathbf{v}_j \rangle = 0 \]

并利用 \(\lambda_i \neq \lambda_j\)

\[ \langle \mathbf{v}_i, \mathbf{v}_j \rangle = 0. \]

即，\(\mathbf{v}_i\) 和 \(\mathbf{v}_j\) 必定是正交的。

我们证明了：

**引理** 设 \(A \in \mathbb{R}^{d \times d}\) 为对称矩阵。假设 \(\mathbf{v}_i\) 和 \(\mathbf{v}_j\) 是对应于不同特征值的特征向量。那么 \(\mathbf{v}_i\) 和 \(\mathbf{v}_j\) 是正交的。\(\flat\)

这个引理给出了一个不同的证明——在对称情况下——特征值的数量最多为 \(d\)，因为成对的正交向量是线性无关的。

事实上：

**定理** **(谱定理**) \(\idx{谱定理}\xdi\) 设 \(A \in \mathbb{R}^{d \times d}\) 为对称矩阵，即 \(A^T = A\)。则 \(A\) 有 \(d\) 个正交归一特征向量 \(\mathbf{q}_1, \ldots, \mathbf{q}_d\)，对应（不一定互异）的实特征值 \(\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d\)。此外，\(A\) 可以表示为矩阵分解

\[ A = Q \Lambda Q^T = \sum_{i=1}^d \lambda_i \mathbf{q}_i \mathbf{q}_i^T \]

其中 \(Q\) 的列是 \(\mathbf{q}_1, \ldots, \mathbf{q}_d\)，而 \(\Lambda = \mathrm{diag}(\lambda_1, \ldots, \lambda_d)\)。\(\sharp\)

我们将这种分解称为 \(A\) 的谱分解\(\idx{谱分解}\xdi\)。

注意，这种分解确实产生了 \(A\) 的特征向量。对于任何 \(j\)，我们有

\[ A \mathbf{q}_j = \sum_{i=1}^d \lambda_i \mathbf{q}_i \mathbf{q}_i^T \mathbf{q}_j = \lambda_j \mathbf{q}_j, \]

其中我们使用了正交性，即当 \(i \neq j\) 时 \(\mathbf{q}_i^T \mathbf{q}_j = 0\)，当 \(i = j\) 时 \(\mathbf{q}_i^T \mathbf{q}_j = 1\)。上述方程恰好说明 \(\mathbf{q}_j\) 是 \(A\) 的一个特征向量，对应的特征值为 \(\lambda_j\)。由于我们已经找到了 \(d\) 个特征值（可能重复），我们已经找到了所有特征值。

设 \(\lambda_1, \lambda_2, \ldots, \lambda_d\) 为由**谱定理**保证存在的特征值。我们首先论证不存在其他特征值。实际上，假设 \(\mu \neq \lambda_1, \lambda_2, \ldots, \lambda_d\) 是一个特征值，其对应的特征向量为 \(\mathbf{p}\)。我们已经看到 \(\mathbf{p}\) 与特征向量 \(\mathbf{q}_1, \ldots, \mathbf{q}_d\) 正交。由于后者构成了 \(\mathbb{R}^d\) 的一个正交基，这种情况不可能发生，因此我们得到了一个矛盾。

一些特征值 \(\lambda_1, \lambda_2, \ldots, \lambda_d\) 可以重复，也就是说，存在 \(i, j\) 使得 \(\lambda_{i} = \lambda_{j}\)。例如，如果 \(A = I_{d \times d}\) 是单位矩阵，那么特征值对所有 \(i \in [d]\) 都是 \(\lambda_i = 1\)。

对于 \(A\) 的一个固定特征值 \(\lambda\)，具有特征值 \(\lambda\) 的特征向量集合满足

\[ A \mathbf{v} = \lambda \mathbf{v} = \lambda I_{d \times d} \mathbf{v} \]

或者，重新排列，

\[ (A - \lambda I_{d \times d})\mathbf{v} = \mathbf{0}. \]

换句话说，它是 \(\mathrm{null}(A - \lambda I_{d \times d})\) 的集合。

**谱定理**中对应相同特征值 \(\lambda\) 的特征向量可以用 \(\mathrm{null}(A - \lambda I_{d \times d})\) 的任意正交基来代替。但是，除了这种自由度之外，我们还有以下特征描述：设 \(\mu_1,\ldots,\mu_f\) 是 \(\lambda_1, \lambda_2, \ldots, \lambda_d\) 中的唯一值

\[ \mathbb{R}^d = \mathrm{null}(A - \mu_1 I_{d \times d}) \oplus \mathrm{null}(A - \mu_2 I_{d \times d}) \oplus \cdots \oplus \mathrm{null}(A - \mu_f I_{d \times d}), \]

在这里，我们使用了这样一个事实：对于任何 \(\mathbf{u} \in \mathrm{null}(A - \mu_i I_{d \times d})\) 和 \(\mathbf{v} \in \mathrm{null}(A - \mu_j I_{d \times d})\)，其中 \(i \neq j\)，我们有 \(\mathbf{u}\) 与 \(\mathbf{v}\) 正交。

我们特别证明了**谱定理**中特征值的序列是唯一的（包括重复的）。

两个矩阵 \(B, D \in \mathbb{R}^{d \times d}\) 如果存在一个可逆矩阵 \(P\) 使得 \(B = P^{-1} D P\)，则称它们是[相似](https://en.wikipedia.org/wiki/Matrix_similarity)\(\idx{相似}\xdi\)的。可以证明 \(B\) 和 \(D\) 对应相同的线性映射，但表示在不同的基下。当 \(P = Q\) 是一个正交矩阵时，变换简化为 \(B = Q^T D Q\)。

因此，另一种思考谱分解的方式是，它表达了任何对称矩阵都可以通过正交变换相似于对角矩阵的事实。在这个基下，相应的线性映射由对角矩阵表示，这个基是特征向量基。

**示例：** **(2×2 对称矩阵的特征分解)** 最简单的非平凡情况是 \(2 \times 2\) 对称矩阵

\[\begin{split} A = \begin{pmatrix} a & b\\ b & d \end{pmatrix}. \end{split}\]

我们推导出计算其特征值和特征向量的逐步方法。

如前所述，特征值 \(\lambda\) 对应于非空 \(\mathrm{null}(A - \lambda I_{2 \times 2})\)，相应的特征向量解

\[\begin{split} \mathbf{0} = (A - \lambda I_{2 \times 2})\mathbf{v} = \begin{pmatrix}a - \lambda & b\\ b & d - \lambda\end{pmatrix}. \end{split}\]

换句话说，矩阵 \(\begin{pmatrix}a - \lambda & b\\ b & d - \lambda\end{pmatrix}\) 的列是线性相关的。我们已经看到，检查这一点的办法之一是计算行列式，在 \(2 \times 2\) 的情况下，这很简单

\[\begin{split} \mathrm{det}\left[\begin{pmatrix}a - \lambda & b\\ b & d - \lambda\end{pmatrix}\right] = (a - \lambda ) (d - \lambda) - b². \end{split}\]

这是一个关于 \(\lambda\) 的二次多项式，称为矩阵 \(A\) 的特征多项式。

特征多项式的根，即方程的解

\[ 0 = (a - \lambda ) (d - \lambda) - b² = \lambda² - (a + d)\lambda + (ad - b²), \]

是

\[ \lambda_{1} = \frac{(a + d) + \sqrt{(a + d)² - 4(ad - b²)}}{2} \]

和

\[ \lambda_{2} = \frac{(a + d) - \sqrt{(a + d)² - 4(ad - b²)}}{2}. \]

展开平方根中的表达式

\[ (a + d)² - 4(ad - b²) = a² + d² - 2ad + 4b² = (a - d)² + 4b², \]

我们看到，对于任何 \(a, b, d\)，平方根都是良好定义的（即产生实数值）。

剩下的任务是找到对应的特征向量 \(\mathbf{v}_{1} = (v_{1,1}, v_{1,2})\) 和 \(\mathbf{v}_2 = (v_{2,1}, v_{2,2})\)，通过解 \(2 \times 2\) 的线性方程组

\[\begin{split} \begin{pmatrix} a - \lambda_i & b \\ b & d - \lambda_i \end{pmatrix} \begin{pmatrix} v_{i,1} \\ v_{i,2} \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix} \end{split}\]

这是有解保证的。当 \(\lambda_1 = \lambda_2\) 时，需要找到两个线性无关的解。

这里有一个数值示例。考虑矩阵

\[\begin{split} A = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}. \end{split}\]

特征多项式方程是

\[ \lambda² - 6\lambda + 8 = 0. \]

特征值是

\[ \lambda_{1}, \lambda_{2} = \frac{6 \pm \sqrt{36 - 4(9-1))}}{2} = \frac{6 \pm \sqrt{4}}{2} = 4, 2. \]

然后我们解特征向量。对于 \(\lambda_1 = 4\)

\[\begin{split} \begin{pmatrix} 3 - \lambda_1 & 1 \\ 1 & 3 - \lambda_1 \end{pmatrix} \begin{pmatrix} v_{1,1} \\ v_{1,2} \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix} \Leftrightarrow \begin{cases} - v_{1,1} + v_{1,2} = 0\\ v_{1,1} - v_{1,2} = 0 \end{cases} \end{split}\]

因此，在归一化后，我们取

\[\begin{split} \mathbf{v}_1 = \frac{1}{\sqrt{2}}\begin{pmatrix}1 \\ 1\end{pmatrix}. \end{split}\]

对于 \(\lambda_2 = 2\):

\[\begin{split} \begin{pmatrix} 3 - \lambda_2 & 1 \\ 1 & 3 - \lambda_2 \end{pmatrix} \begin{pmatrix} v_{2,1} \\ v_{2,2} \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix} \Leftrightarrow \begin{cases} v_{1,1} + v_{1,2} = 0\\ v_{1,1} + v_{1,2} = 0 \end{cases} \end{split}\]

因此，在归一化之后，我们取

\[\begin{split} \mathbf{v}_2 = \frac{1}{\sqrt{2}}\begin{pmatrix}1 \\ -1\end{pmatrix}. \end{split}\]

这些是特征向量的事实可以通过手工检查（试试看！）来验证。 \(\lhd\)

更一般地，对于任何矩阵 \(A \in \mathbb{R}^{d \times d}\)，特征多项式 \(\mathrm{det}(A - \lambda I_{d \times d})\) 的根是 \(A\) 的特征值。我们将在下一节推导一个更有效的数值方法来计算它们。

**正半定矩阵的情况** 对称矩阵的特征值虽然为实数，但可能是负数。然而，存在一个重要的特殊情况，即特征值都是非负的，这就是正半定矩阵。\(\idx{正半定矩阵}\xdi\)

**定理** **(正半定性的刻画)** \(\idx{正半定性的刻画}\xdi\) 设 \(A \in \mathbb{R}^{d \times d}\) 为对称矩阵，并且 \(A = Q \Lambda Q^T\) 是 \(A\) 的谱分解，其中 \(\Lambda = \mathrm{diag}(\lambda_1, \ldots, \lambda_d)\)。那么 \(A \succeq 0\) 当且仅当其特征值 \(\lambda_1, \ldots, \lambda_d\) 都是非负的。\(\sharp\)

*证明* 假设 \(A \succeq 0\)。设 \(\mathbf{q}_i\) 是 \(A\) 的一个特征向量，对应的特征值为 \(\lambda_i\)。那么

\[ \langle \mathbf{q}_i, A \mathbf{q}_i \rangle = \langle \mathbf{q}_i, \lambda_i \mathbf{q}_i \rangle = \lambda_i \]

根据正半定矩阵的定义，这必须是非负的。

在相反的方向上，假设 \(\lambda_1, \ldots, \lambda_d \geq 0\)。那么，根据外积形式的谱分解

\[ \langle \mathbf{x}, A \mathbf{x} \rangle = \mathbf{x}^T \left(\sum_{i=1}^d \lambda_i \mathbf{q}_i \mathbf{q}_i^T\right) \mathbf{x} = \sum_{i=1}^d \lambda_i \mathbf{x}^T\mathbf{q}_i \mathbf{q}_i^T\mathbf{x} = \sum_{i=1}^d \lambda_i \langle \mathbf{q}_i, \mathbf{x}\rangle² \]

这必然是非负的。\(\square\)

同样，一个对称矩阵是正定的\(\idx{正定矩阵}\xdi\) 当且仅当所有特征值都是严格正的。证明基本上是相同的。

**知识检查**：通过修改上面的证明来证明这个最后的陈述。 \(\checkmark\)

回想一下，正半定性的一个重要应用是作为凸性的刻画\(\idx{凸函数}\xdi\)。这里有一些例子。

**示例** **(通过 Hessian 的特征值来证明凸性)** 考虑函数

\[ f(x, y) = \frac{3}{2} x² + xy + \frac{3}{2} y² + 5x - 2y + 1. \]

为了证明它是凸的，我们计算其 Hessian

\[\begin{split} H_f(x,y) = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix} \end{split}\]

对于所有的 \(x,y\)。根据先前的例子，其特征值是 \(2\) 和 \(4\)，都是严格正的。这通过 *二阶凸性条件* 证明了我们的断言。 \(\lhd\)

**示例**: **(对数凹性)** 如果说一个函数 \(f :\mathbb{R}^d \to \mathbb{R}\) 是对数凹的，那么 \(-\log f\) 是凸的。换句话说，对于所有的 \(\mathbf{x}, \mathbf{y} \in \mathbb{R}^d\) 和 \(\alpha \in (0,1)\)

\[ - \log f((1-\alpha)\mathbf{x} + \alpha \mathbf{y}) \leq - (1-\alpha) \log f(\mathbf{x}) - \alpha \log f(\mathbf{y}), \]

这等价于

\[ \log f((1-\alpha)\mathbf{x} + \alpha \mathbf{y}) \geq (1-\alpha) \log f(\mathbf{x}) + \alpha \log f(\mathbf{y}) , \]

或者，因为 \(a \log b = \log b^a\) 并且对数函数是严格递增的，

\[ f((1-\alpha)\mathbf{x} + \alpha \mathbf{y}) \geq f(\mathbf{x})^{1-\alpha} f(\mathbf{y})^{\alpha}. \]

我们将在课程中稍后看到，在 \(\mathbb{R}^d\) 上的多元高斯向量 \(\mathbf{X}\) 具有均值 \(\bmu \in \mathbb{R}^d\) 和正定协方差矩阵 \(\bSigma \in \mathbb{R}^{d \times d}\)，其概率密度函数（PDF）

\[ f_{\bmu, \bSigma}(\mathbf{x}) = \frac{1}{(2\pi)^{d/2} \,|\bSigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \bmu)^T \bSigma^{-1} (\mathbf{x} - \bmu)\right) \]

其中 \(|A|\) 是 \(A\) 的 [行列式](https://en.wikipedia.org/wiki/Determinant)，在对称矩阵的情况下，它仅仅是其特征值的乘积（如果有重复）。我们声称这个 PDF 是对数凹的。

从定义来看，

\[\begin{align*} &- \log f_{\bmu, \bSigma}(\mathbf{x})\\ &= \frac{1}{2}(\mathbf{x} - \bmu)^T \bSigma^{-1} (\mathbf{x} - \bmu) + \log (2\pi)^{d/2} \,|\bSigma|^{1/2}\\ &= \frac{1}{2}\mathbf{x}^T \bSigma^{-1} \mathbf{x} - \bmu^T \bSigma^{-1} \mathbf{x} + \left[\frac{1}{2}\bmu^T \bSigma^{-1} \bmu + \log (2\pi)^{d/2} \,|\bSigma|^{1/2}\right]. \end{align*}\]

令 \(P = \bSigma^{-1}\)，\(\mathbf{q} = - \bmu^T \bSigma^{-1}\) 和 \(r = \frac{1}{2}\bmu^T \bSigma^{-1} \bmu + \log (2\pi)^{d/2} \,|\bSigma|^{1/2}\)。

一个先前的例子表明，如果 \(\bSigma^{-1}\) 是正半定的，那么 PDF 是对数凹的。由于假设 \(\bSigma\) 是正定的，\(\bSigma = Q \Lambda Q^T\) 有一个特征分解，其中 \(\Lambda\) 的所有对角线元素都是严格正的。因此，\(\bSigma^{-1} = Q \Lambda^{-1} Q^T\)，其中 \(\Lambda^{-1}\) 的对角线元素是 \(\Lambda\) 的元素的倒数——因此也是严格正的。特别是，\(\bSigma^{-1}\) 是正半定的。 \(\lhd\)

**数值角**: 因此，我们可以通过使用 `numpy.linalg.eig`（[链接](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html)）计算其特征值来检查一个矩阵是否是正半定的。

```py
A = np.array([[1, -1], [-1, 1]])
w, v = LA.eig(A)
print(w) 
```

```py
[2\. 0.] 
```

```py
B = np.array([[1, -2], [-2, 1]])
z, u = LA.eig(B)
print(z) 
```

```py
[ 3\. -1.] 
```

**知识检查**: 这些矩阵中哪一个是正半定的？

\[\begin{split} A = \begin{pmatrix} 1 & -1\\ -1 & 1 \end{pmatrix} \qquad B = \begin{pmatrix} 1 & -2\\ -2 & 1 \end{pmatrix} \end{split}\]

a) 两个

b) \(A\)

c) \(B\)

d) 都不是

\(\checkmark\)

\(\unlhd\)

***自我评估测验*** *(由 Claude、Gemini 和 ChatGPT 协助)*

**1** 矩阵\(A \in \mathbb{R}^{n \times m}\)的秩是什么？

a) \(A\)中非零项的数量。

b) \(A\)的行空间的维度。

c) \(A\)的零空间的维度。

d) \(A\)的迹

**2** 关于矩阵\(A \in \mathbb{R}^{n \times m}\)的秩，以下哪个是正确的？

a) \(\mathrm{rk}(A) \leq \min\{n,m\}\)

b) \(\mathrm{rk}(A) \geq \max\{n,m\}\)

c) 只有当\(A\)是对称时，\(\mathrm{rk}(A) = \mathrm{rk}(A^T)\)

d) 只有当\(A\)是方阵时，\(\mathrm{rk}(A) = \mathrm{rk}(A^T)\)

**3** 设\(A \in \mathbb{R}^{n \times k}\)和\(B \in \mathbb{R}^{k \times m}\)。以下哪个在一般情况下是正确的？

a) \(\mathrm{rk}(AB) \leq \mathrm{rk}(A)\)

b) \(\mathrm{rk}(AB) \geq \mathrm{rk}(A)\)

c) \(\mathrm{rk}(AB) = \mathrm{rk}(A)\)

d) \(\mathrm{rk}(AB) = \mathrm{rk}(B)\)

**4** 设\(A \in \mathbb{R}^{d \times d}\)是对称的。根据谱定理，以下哪个是正确的？

a) \(A\)至多有\(d\)个不同的特征值

b) \(A\)恰好有\(d\)个不同的特征值

c) \(A\)至少有\(d\)个不同的特征值

d) \(A\)的不同特征值的数量与\(d\)无关

**5** 关于两个向量\(\mathbf{u}\)和\(\mathbf{v}\)的外积，以下哪个是正确的？

a) 它是一个标量。

b) 它是一个向量。

c) 它是一个秩为 1 的矩阵。

d) 它是一个秩为零的矩阵。

1 题的答案：b. 理由：文本声明“\(A\)的行秩和列秩[就是]秩，我们用\(\mathrm{rk}(A)\)表示。”

2 题的答案：a. 理由：文本在行秩等于列秩定理中声明“\(A\)的行秩等于\(A\)的列秩。此外，\(\mathrm{rk}(A) \leq \min\{n,m\}\)。”

3 题的答案：a. 理由：文本显示“\(AB\)的列是\(A\)的列的线性组合。因此\(\mathrm{col}(AB) \subseteq \mathrm{col}(A)\)。根据观察 D2，这个结论成立。”

4 题的答案：a. 理由：谱定理指出“对称矩阵\(A\)有\(d\)个正交归一的特征向量\(\mathbf{q}_1, \ldots, \mathbf{q}_d\)，对应的不一定是不同的实特征值\(\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d\)。”

5 题的答案：c. 理由：文本定义了外积并声明“如果\(\mathbf{u}\)和\(\mathbf{v}\)不为零，矩阵\(\mathbf{u} \mathbf{v}^T\)的秩为 1。”
