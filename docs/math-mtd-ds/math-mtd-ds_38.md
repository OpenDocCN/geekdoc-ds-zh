# 5.3\. 特征值的变分表征#

> 原文：[`mmids-textbook.github.io/chap05_specgraph/03_extremal/roch-mmids-specgraph-extremal.html`](https://mmids-textbook.github.io/chap05_specgraph/03_extremal/roch-mmids-specgraph-extremal.html)

在本节中，我们通过某些优化问题来表征特征值，这个想法我们在与奇异值分解密切相关的背景下已经遇到过。这种变分表征在应用中非常有用，正如我们在本章后面将要看到的。我们首先给出一个使用这一想法的谱定理的证明。

## 5.3.1\. 谱定理的证明#

我们将需要以下块矩阵公式。假设

\[\begin{split} A = \begin{pmatrix} A_{11} \\ A_{21} \end{pmatrix} \quad \text{和} \quad B = \begin{pmatrix} B_{11} & B_{12} \end{pmatrix} \end{split}\]

其中 \(A_{11} \in \mathbb{R}^{n_1\times p}\)，\(A_{21} \in \mathbb{R}^{n_2\times p}\)，\(B_{11} \in \mathbb{R}^{p\times n_1}\)，\(B_{12} \in \mathbb{R}^{p\times n_2}\)，然后

\[\begin{split} A B = \begin{pmatrix} A_{11} B_{11} & A_{11} B_{12} \\ A_{21} B_{11} & A_{21} B_{12} \end{pmatrix}. \end{split}\]

事实上，这仅仅是之前遇到的 \(2 \times 2\) 块矩阵乘法公式的特例，其中块 \(A_{12}, A_{22}, B_{21}, B_{22}\) 是空的。

*证明思路（谱定理）：* 类似于我们使用豪斯霍尔德变换来“在对角线下添加零”，这里我们将使用一系列正交变换来在对角线上下都添加零。回想一下，两个矩阵 \(C, D \in \mathbb{R}^{d \times d}\) 如果存在一个可逆矩阵 \(P\) 使得 \(C = P^{-1} D P\)，则它们是[相似](https://en.wikipedia.org/wiki/Matrix_similarity)的，并且当 \(P = W\) 是一个正交矩阵时，变换简化为 \(C = W^T D W\)。

我们构造一系列正交矩阵 \(\hat{W}_1,\ldots, \hat{W}_d\)，并逐步计算 \(W_1^T A W_1\)，\(W_2^T W_1^T A W_1 W_2\)，依此类推，使得

\[ \Lambda = W_d^T \cdots W_2^T W_1^T A W_1 W_2 \cdots W_d \]

是对角矩阵。然后矩阵 \(Q\) 简单地是 \(W_1 W_2 \cdots W_d\)（检查它！）。

为了定义这些矩阵，我们使用一个贪婪序列来最大化二次型 \(\langle \mathbf{v}, A \mathbf{v}\rangle\)。这个二次型如何与特征值相关？回想一下，对于一个单位特征向量 \(\mathbf{v}\) 和特征值 \(\lambda\)，我们有 \(\langle \mathbf{v}, A \mathbf{v}\rangle = \langle \mathbf{v}, \lambda \mathbf{v}\rangle = \lambda\)。

*证明：* *(谱定理)* \(\idx{spectral theorem}\xdi\) 我们通过归纳法进行证明。我们已经遇到过特征向量是优化问题的驻点的想法。实际上，这个证明使用了那个想法。

***第一个特征向量：*** 令 \(A_1 = A\)。定义

\[ \mathbf{v}_1 \in \arg\max\{\langle \mathbf{v}, A_1 \mathbf{v}\rangle:\|\mathbf{v}\| = 1\} \]

这是由**极值定理**保证的，并进一步定义

\[ \lambda_1 = \max\{\langle \mathbf{v}, A_1 \mathbf{v}\rangle:\|\mathbf{v}\| = 1\}. \]

将 \(\mathbf{v}_1\) 完成为 \(\mathbb{R}^d\) 的正交归一基 \(\mathbf{v}_1, \hat{\mathbf{v}}_2, \ldots, \hat{\mathbf{v}}_d\)，并形成分块矩阵

\[ \hat{W}_1 = \begin{pmatrix} \mathbf{v}_1 & \hat{V}_1 \end{pmatrix} \]

其中 \(\hat{V}_1\) 的列向量是 \(\hat{\mathbf{v}}_2, \ldots, \hat{\mathbf{v}}_d\)。注意 \(\hat{W}_1\) 是通过构造正交的。

***向对角化迈进一步：*** 我们接下来将展示 \(\hat{W}_1\) 通过相似变换使我们更接近对角矩阵。首先要注意的是

\[\begin{align*} \hat{W}_1^T A_1 \hat{W}_1 &= \begin{pmatrix} \mathbf{v}_1^T \\ \hat{V}_1^T \end{pmatrix} A_1 \begin{pmatrix} \mathbf{v}_1 & \hat{V}_1 \end{pmatrix}\\ &= \begin{pmatrix} \mathbf{v}_1^T \\ \hat{V}_1^T \end{pmatrix} \begin{pmatrix} A_1 \mathbf{v}_1 & A_1 \hat{V}_1 \end{pmatrix}\\ &= \begin{pmatrix} \mathbf{v}_1^T A_1 \mathbf{v}_1 & \mathbf{v}_1^T A_1 \hat{V}_1\\ \hat{V}_1^T A_1 \mathbf{v}_1 & \hat{V}_1^T A_1 \hat{V}_1 \end{pmatrix}\\ &= \begin{pmatrix} \lambda_1 & \mathbf{w}_1^T \\ \mathbf{w}_1 & A_2 \end{pmatrix} \end{align*}\]

其中 \(\mathbf{w}_1 = \hat{V}_1^T A_1 \mathbf{v}_1\) 且 \(A_2 = \hat{V}_1^T A_1 \hat{V}_1\).

关键的断言是 \(\mathbf{w}_1 = \mathbf{0}\)。这可以通过矛盾论证得出。确实，假设 \(\mathbf{w}_1 \neq \mathbf{0}\) 并考虑单位向量（检查 \(\|\mathbf{z}\| = 1\)\)

\[\begin{split} \mathbf{z} = \hat{W}_1 \frac{1}{\sqrt{1 + \delta² \|\mathbf{w}_1\|²}} \begin{pmatrix} 1 \\ \delta \mathbf{w}_1 \end{pmatrix} \end{split}\]

这实现了目标值（检查它！）

\[\begin{align*} \mathbf{z}^T A_1 \mathbf{z} &= \frac{1}{1 + \delta² \|\mathbf{w}_1\|²} \begin{pmatrix} 1 \\ \delta \mathbf{w}_1 \end{pmatrix}^T \begin{pmatrix} \lambda_1 & \mathbf{w}_1^T \\ \mathbf{w}_1 & A_2 \end{pmatrix} \begin{pmatrix} 1 \\ \delta \mathbf{w}_1 \end{pmatrix}\\ &= \frac{1}{1 + \delta² \|\mathbf{w}_1\|²} \left( \lambda_1 + 2 \delta \|\mathbf{w}_1\|² + \delta² \mathbf{w}_1^T A_2 \mathbf{w}_1 \right). \end{align*}\]

根据几何级数的和，对于 \(\varepsilon \in (0,1)\)，

\[ \frac{1}{1 + \varepsilon²} = 1 - \varepsilon² + \varepsilon⁴ + \cdots. \]

因此，对于足够小的 \(\delta > 0\)，

\[\begin{align*} \mathbf{z}^T A_1 \mathbf{z} &\approx (\lambda_1 + 2 \delta \|\mathbf{w}_1\|² + \delta² \mathbf{w}_1^T A_2 \mathbf{w}_1) (1 - \delta² \|\mathbf{w}_1\|²)\\ &\approx \lambda_1 + 2 \delta \|\mathbf{w}_1\|² + C \delta²\\ &> \lambda_1 \end{align*}\]

其中 \(C \in \mathbb{R}\) 依赖于 \(\mathbf{w}_1\) 和 \(A_2\)，并且我们忽略了涉及 \(\delta³, \delta⁴, \delta⁵, \ldots\) 的“高阶”项，当 \(\delta\) 很小时，这些项的总贡献可以忽略不计。这给出了所需的矛盾——也就是说，它将意味着存在一个向量能够实现比最优 \(\mathbf{v}_1\) 更好的目标值。（*练习：使这个论证严谨。*）

因此，让 \(W_1 = \hat{W}_1\)，

\[\begin{split} W_1^T A_1 W_1 = \begin{pmatrix} \lambda_1 & \mathbf{0} \\ \mathbf{0} & A_2 \end{pmatrix}. \end{split}\]

最后注意，\(A_2 = \hat{V}_1^T A_1 \hat{V}_1\) 是对称的

\[ A_2^T = (\hat{V}_1^T A_1 \hat{V}_1)^T = \hat{V}_1^T A_1^T \hat{V}_1 = \hat{V}_1^T A_1 \hat{V}_1 = A_2 \]

由 \(A_1\) 本身的对称性。

***归纳的下一步：*** 将相同的论证应用于对称矩阵 \(A_2 \in \mathbb{R}^{(d-1)\times (d-1)}\)，令 \(\hat{W}_2 = (\mathbf{v}_2\ \hat{V}_2) \in \mathbb{R}^{(d-1)\times (d-1)}\) 为相应的正交矩阵，并通过方程定义 \(\lambda_2\) 和 \(A_3\)

\[\begin{split} \hat{W}_2^T A_2 \hat{W}_2 = \begin{pmatrix} \lambda_2 & \mathbf{0} \\ \mathbf{0} & A_3 \end{pmatrix}. \end{split}\]

现在定义分块矩阵

\[\begin{split} W_2 = \begin{pmatrix} 1 & \mathbf{0}\\ \mathbf{0} & \hat{W}_2 \end{pmatrix}. \end{split}\]

注意到（检查它！）

\[\begin{split} W_2^T W_1^T A_1 W_1 W_2 = W_2^T \begin{pmatrix} \lambda_1 & \mathbf{0} \\ \mathbf{0} & A_2 \end{pmatrix} W_2 = \begin{pmatrix} \lambda_1 & \mathbf{0}\\ \mathbf{0} & \hat{W}_2^T A_2 \hat{W}_2 \end{pmatrix} =\begin{pmatrix} \lambda_1 & 0 & \mathbf{0} \\ 0 & \lambda_2 & \mathbf{0} \\ \mathbf{0} & \mathbf{0} & A_3 \end{pmatrix}. \end{split}\]

通过类似的归纳方法给出断言。 \(\square\)

关于证明的一些评论：

1- 通过向量微积分使用拉格朗日乘数法在 \(\max\{\langle \mathbf{v}, A_1 \mathbf{v}\rangle:\|\mathbf{v}\| = 1\}\) 上，可以更直观地理解 \(\mathbf{w}_1 = \mathbf{0}\) 的事实，以看到 \(A_1 \mathbf{v}_1\) 必须与 \(\mathbf{v}_1\) 成正比。因此，根据 \(\hat{V}_1^T\) 的构造，\(\mathbf{w}_1 = \hat{V}_1^T A_1 \mathbf{v}_1 = \mathbf{0}\)。实际上，定义拉格朗日函数

\[ L(\mathbf{v}, \lambda) = \langle \mathbf{v}, A_1 \mathbf{v}\rangle - \lambda(\|\mathbf{v}\|² - 1). \]

局部最大值 \(\mathbf{v}_1\) 的一阶必要条件是（检查它！）

\[ \nabla_{\mathbf{v}} L(\mathbf{v}_1, \lambda_1) = 2A_1 \mathbf{v}_1 - 2\lambda_1 \mathbf{v}_1 = \mathbf{0} \]\[ \nabla_{\lambda} L(\mathbf{v}_1, \lambda_1) = \|\mathbf{v}_1\|² - 1 = 0. \]

从第一个条件，我们有

\[ A_1 \mathbf{v}_1 = \lambda_1 \mathbf{v}_1. \]

这表明 \(A_1 \mathbf{v}_1\) 与 \(\mathbf{v}_1\) 成正比，正如所声称的。

2- 通过构造，向量 \(\mathbf{v}_2\)（即 \(\hat{W}_2\) 的第一列）是

\[ \mathbf{v}_2 \in \arg\max\{\langle \mathbf{v}, A_2 \mathbf{v}\rangle:\|\mathbf{v}\| = 1\}. \]

注意，根据 \(A_2\) 的定义（以及 \(A_1 = A\) 的事实），

\[ \mathbf{v}^T A_2 \mathbf{v} = \mathbf{v}^T \hat{V}_1^T A_1 \hat{V}_1 \mathbf{v} = (\hat{V}_1 \mathbf{v})^T \,A \,(\hat{V}_1 \mathbf{v}). \]

因此，我们可以将解 \(\mathbf{v}_2\) 视为指定 \(\hat{V}_1\) 的列的优线性组合——这些列形成了与 \(\mathbf{v}_1\) 正交的向量空间 \(\mathrm{span}(\mathbf{v}_1)^\perp\) 的基。本质上，\(\mathbf{v}_2\) 解决了与 \(\mathbf{v}_1\) 相同的问题，*但仅限于 \(\mathrm{span}(\mathbf{v}_1)^\perp\)*。我们将在下面回到这个问题。

## 5.3.2\. 变分特征：特殊情况#

我们从定义开始。

**定义** **(瑞利商)** \(\idx{瑞利商}\xdi\) 设 \(A \in \mathbb{R}^{d \times d}\) 为对称矩阵。瑞利商定义为

\[ \mathcal{R}_A(\mathbf{u}) = \frac{\langle \mathbf{u}, A \mathbf{u} \rangle}{\langle \mathbf{u}, \mathbf{u} \rangle} \]

该公式适用于 \(\mathbb{R}^{d}\) 中任何 \(\mathbf{u} \neq \mathbf{0}\)。\(\natural\)

为了看到与谱分解的联系，设 \(\mathbf{v}\) 是 \(A\) 的一个（不一定为单位）特征向量，其特征值为 \(\lambda\)。可以证明 \(\mathcal{R}_A(\mathbf{v}) = \lambda\)。（试试看！）实际上，回想一下，我们之前已经证明了 \(A\) 的特征向量是 \(\mathcal{R}_A\) 的驻点。

在陈述一般变分特征之前，我们证明几个特殊情况。在整个过程中，设 \(A \in \mathbb{R}^{d \times d}\) 为对称矩阵，其谱分解为 \(A = \sum_{i=1}^d \lambda_i \mathbf{v}_i \mathbf{v}_i^T\)，其中 \(\lambda_1 \geq \cdots \geq \lambda_d\)。

*最大的特征值:* 由于 \(\mathbf{v}_1, \ldots, \mathbf{v}_d\) 形成了 \(\mathbb{R}^d\) 的一个正交基，任何非零向量 \(\mathbf{u}\) 可以写成 \(\mathbf{u} = \sum_{i=1}^d \langle \mathbf{u}, \mathbf{v}_i \rangle \mathbf{v}_i\)，并且根据正交列表的性质和内积的双线性，可以得出

\[ \langle \mathbf{u}, \mathbf{u} \rangle = \|\mathbf{u}\|² = \left\|\sum_{i=1}^d \langle \mathbf{u}, \mathbf{v}_i \rangle \mathbf{v}_i\right\|² = \sum_{i=1}^d \langle \mathbf{u}, \mathbf{v}_i \rangle² \]\[ \langle \mathbf{u}, A \mathbf{u} \rangle = \left\langle \mathbf{u}, \sum_{i=1}^d \langle \mathbf{u}, \mathbf{v}_i \rangle A \mathbf{v}_i \right\rangle = \left\langle \mathbf{u}, \sum_{i=1}^d \langle \mathbf{u}, \mathbf{v}_i \rangle \lambda_i \mathbf{v}_i \right\rangle = \sum_{i=1}^d \lambda_i \langle \mathbf{u}, \mathbf{v}_i \rangle². \]

因此，

\[ \mathcal{R}_A(\mathbf{u}) = \frac{\langle \mathbf{u}, A \mathbf{u} \rangle}{\langle \mathbf{u}, \mathbf{u} \rangle} = \frac{\sum_{i=1}^d \lambda_i \langle \mathbf{u}, \mathbf{v}_i \rangle²}{\sum_{i=1}^d \langle \mathbf{u}, \mathbf{v}_i \rangle²} \leq \lambda_1 \frac{\sum_{i=1}^d \langle \mathbf{u}, \mathbf{v}_i \rangle²}{\sum_{i=1}^d \langle \mathbf{u}, \mathbf{v}_i \rangle²} = \lambda_1, \]

其中我们使用了 \(\lambda_1 \geq \cdots \geq \lambda_d\) 和 \(\langle \mathbf{u}, \mathbf{v}_i \rangle² \geq 0\) 的事实。此外，\(\mathcal{R}_A(\mathbf{v}_1) = \lambda_1\)。因此，我们得出

\[ \lambda_1 = \max_{\mathbf{u} \neq \mathbf{0}} \mathcal{R}_A(\mathbf{u}). \]

*最小的特征值:* 从相反的方向进行论证，我们得到最小特征值的特征。使用与之前相同的符号，我们有

\[ \mathcal{R}_A(\mathbf{u}) = \frac{\langle \mathbf{u}, A \mathbf{u} \rangle}{\langle \mathbf{u}, \mathbf{u} \rangle} = \frac{\sum_{i=1}^d \lambda_i \langle \mathbf{u}, \mathbf{v}_i \rangle²}{\sum_{i=1}^d \langle \mathbf{u}, \mathbf{v}_i \rangle²} \geq \lambda_d \frac{\sum_{i=1}^d \langle \mathbf{u}, \mathbf{v}_i \rangle²}{\sum_{i=1}^d \langle \mathbf{u}, \mathbf{v}_i \rangle²} = \lambda_d, \]

其中，我们再次使用了 \(\lambda_1 \geq \cdots \geq \lambda_d\) 和 \(\langle \mathbf{u}, \mathbf{v}_i \rangle² \geq 0\) 的事实。此外，\(\mathcal{R}_A(\mathbf{v}_d) = \lambda_d\)。因此，我们得出

\[ \lambda_d = \min_{\mathbf{u} \neq \mathbf{0}} \mathcal{R}_A(\mathbf{u}). \]

*第二小的特征值:* 为了挑选出第二小的特征值，我们按照上述方法进行论证，但将优化限制在空间 \(\mathcal{V}_{d-1} = \mathrm{span}(\mathbf{v}_1,\ldots,\mathbf{v}_{d-1})\) 中。实际上，如果 \(\mathbf{u}\) 在线性子空间 \(\mathcal{V}_{d-1}\) 中，它可以写成 \(\mathbf{u} = \sum_{i=1}^{d-1} \langle \mathbf{u}, \mathbf{v}_i \rangle \mathbf{v}_i\)（因为 \(\mathbf{v}_1,\ldots,\mathbf{v}_{d-1}\) 形成了它的正交归一基；为什么？）并且由此得出

\[ \langle \mathbf{u}, \mathbf{u} \rangle = \sum_{i=1}^{d-1} \langle \mathbf{u}, \mathbf{v}_i \rangle² \]\[ \langle \mathbf{u}, A \mathbf{u} \rangle = \left\langle \mathbf{u}, \sum_{i=1}^{d-1} \langle \mathbf{u}, \mathbf{v}_i \rangle \lambda_i \mathbf{v}_i \right\rangle = \sum_{i=1}^{d-1} \lambda_i \langle \mathbf{u}, \mathbf{v}_i \rangle². \]

因此，

\[ \mathcal{R}_A(\mathbf{u}) = \frac{\langle \mathbf{u}, A \mathbf{u} \rangle}{\langle \mathbf{u}, \mathbf{u} \rangle} = \frac{\sum_{i=1}^{d-1} \lambda_i \langle \mathbf{u}, \mathbf{v}_i \rangle²}{\sum_{i=1}^{d-1} \langle \mathbf{u}, \mathbf{v}_i \rangle²} \geq \lambda_{d-1} \frac{\sum_{i=1}^{d-1} \langle \mathbf{u}, \mathbf{v}_i \rangle²}{\sum_{i=1}^{d-1} \langle \mathbf{u}, \mathbf{v}_i \rangle²} = \lambda_{d-1}, \]

其中我们使用了 \(\lambda_1 \geq \cdots \geq \lambda_{d-1}\) 和 \(\langle \mathbf{u}, \mathbf{v}_i \rangle² \geq 0\) 的事实。此外，\(\mathcal{R}_A(\mathbf{v}_{d-1}) = \lambda_{d-1}\) 并且当然 \(\mathbf{v}_{d-1} \in \mathcal{V}_{d-1}\)。因此，我们得出

\[ \lambda_{d-1} = \min_{\mathbf{0} \neq \mathbf{u} \in \mathcal{V}_{d-1}} \mathcal{R}_A(\mathbf{u}). \]

现在什么是 \(\mathcal{V}_{d-1}\) 呢？它由正交列表 \(\mathbf{v}_1,\ldots,\mathbf{v}_{d-1}\) 张成，其中每个都与 \(\mathbf{v}_d\) 正交。所以（为什么？）

\[ \mathcal{V}_{d-1} = \mathrm{span}(\mathbf{v}_d)^\perp. \]

因此，等价地，

\[ \lambda_{d-1} = \min\left\{\mathcal{R}_A(\mathbf{u})\,:\ \mathbf{u} \neq \mathbf{0}, \langle \mathbf{u}, \mathbf{v}_d\rangle = 0 \right\}. \]

事实上，对于任何 \(\mathbf{u} \neq \mathbf{0}\)，我们可以通过定义 \(\mathbf{z} = \mathbf{u}/\|\mathbf{u}\|\) 来归一化它，并且我们注意到

\[ \mathcal{R}_A(\mathbf{u}) = \frac{\langle \mathbf{u}, A \mathbf{u}\rangle}{\langle \mathbf{u},\mathbf{u}\rangle} = \frac{\langle \mathbf{u}, A \mathbf{u}\rangle}{\|\mathbf{u}\|²} = \left\langle \frac{\mathbf{u}}{\|\mathbf{u}\|}, A \frac{\mathbf{u}}{\|\mathbf{u}\|}\right\rangle = \langle \mathbf{z}, A \mathbf{z}\rangle. \]

因此，

\[ \lambda_{d-1} = \min\left\{\langle \mathbf{z}, A \mathbf{z}\rangle\,:\ \|\mathbf{z}\|=1, \langle \mathbf{z}, \mathbf{v}_d\rangle = 0 \right\}. \]

## 5.3.3\. 一般陈述：Courant-Fischer#

在陈述一个一般结果之前，我们给出一个额外的例子。

*第二个最小的特征值（取两个）：* 有趣的是，第二个最小的特征值有一个第二特征。实际上，如果我们将优化限制在空间 \(\mathcal{W}_{2} = \mathrm{span}(\mathbf{v}_{d-1},\mathbf{v}_d)\) 上，那么。如果 \(\mathbf{u}\) 在线性子空间 \(\mathcal{W}_{2}\) 中，它可以写成 \(\mathbf{u} = \sum_{i=d-1}^{d} \langle \mathbf{u}, \mathbf{v}_i \rangle \mathbf{v}_i\)（因为 \(\mathbf{v}_{d-1},\mathbf{v}_{d}\) 形成了它的正交基），并且由此得出

\[ \langle \mathbf{u}, \mathbf{u} \rangle = \sum_{i=d-1}^{d} \langle \mathbf{u}, \mathbf{v}_i \rangle² \]\[ \langle \mathbf{u}, A \mathbf{u} \rangle = \left\langle \mathbf{u}, \sum_{i=d-1}^{d} \langle \mathbf{u}, \mathbf{v}_i \rangle \lambda_i \mathbf{v}_i \right\rangle = \sum_{i=d-1}^{d} \lambda_i \langle \mathbf{u}, \mathbf{v}_i \rangle². \]

因此，

\[ \mathcal{R}_A(\mathbf{u}) = \frac{\langle \mathbf{u}, A \mathbf{u} \rangle}{\langle \mathbf{u}, \mathbf{u} \rangle} = \frac{\sum_{i=d-1}^{d} \lambda_i \langle \mathbf{u}, \mathbf{v}_i \rangle²}{\sum_{i=d-1}^{d} \langle \mathbf{u}, \mathbf{v}_i \rangle²} \leq \lambda_{d-1} \frac{\sum_{i=d-1}^{d} \langle \mathbf{u}, \mathbf{v}_i \rangle²}{\sum_{i=d-1}^{d} \langle \mathbf{u}, \mathbf{v}_i \rangle²} = \lambda_{d-1}, \]

其中我们使用了 \(\lambda_{d-1} \geq \lambda_{d}\) 和 \(\langle \mathbf{u}, \mathbf{v}_i \rangle² \geq 0\) 的事实。此外，\(\mathcal{R}_A(\mathbf{v}_{d-1}) = \lambda_{d-1}\)。因此，我们已建立

\[ \lambda_{d-1} = \max_{\mathbf{0} \neq \mathbf{u} \in \mathcal{W}_{2}} \mathcal{R}_A(\mathbf{u}). \]

**定理** **(Courant-Fischer)** \(\idx{Courant-Fischer 定理}\xdi\) 设 \(A \in \mathbb{R}^{d \times d}\) 为一个对称矩阵，其谱分解为 \(A = \sum_{i=1}^d \lambda_i \mathbf{v}_i \mathbf{v}_i^T\)，其中 \(\lambda_1 \geq \cdots \geq \lambda_d\)。对于每个 \(k = 1,\ldots,d\)，定义子空间

\[ \mathcal{V}_k = \mathrm{span}(\mathbf{v}_1, \ldots, \mathbf{v}_k) \quad\text{和}\quad \mathcal{W}_{d-k+1} = \mathrm{span}(\mathbf{v}_k, \ldots, \mathbf{v}_d). \]

然后，对于所有 \(k = 1,\ldots,d\)，

\[ \lambda_k = \min_{\mathbf{u} \in \mathcal{V}_k} \mathcal{R}_A(\mathbf{u}) = \max_{\mathbf{u} \in \mathcal{W}_{d-k+1}} \mathcal{R}_A(\mathbf{u}), \]

这些被称为局部公式。此外，我们还有以下最小-最大（或全局）公式，它们不依赖于谱分解的选择：对于所有 \(k = 1,\ldots,d\)，

\[ \lambda_k = \max_{\mathrm{dim}(\mathcal{V}) = k} \min_{\mathbf{u} \in \mathcal{V}} \mathcal{R}_A(\mathbf{u}) = \min_{\mathrm{dim}(\mathcal{W}) = d-k+1} \max_{\mathbf{u} \in \mathcal{W}} \mathcal{R}_A(\mathbf{u}). \]

\(\sharp\)

*证明思路:* 对于局部公式，我们将 \(\mathcal{V}_k\) 中的向量展开到基 \(\mathbf{v}_1,\ldots,\mathbf{v}_k\) 中，并使用 \(\mathcal{R}_A(\mathbf{v}_i) = \lambda_i\) 和特征值按非递增顺序排列的事实。全局公式由维度论证得出。

**示例:** **(第三小特征值)** 可以使用 Courant-Fischer 定理恢复前一小节中的特殊情况。还可以得到一些新的有趣情况。接下来，我们将给出第三小特征值的特征描述。设 \(A \in \mathbb{R}^{d \times d}\) 为一个对称矩阵，其谱分解为 \(A = \sum_{i=1}^d \lambda_i \mathbf{v}_i \mathbf{v}_i^T\)，其中 \(\lambda_1 \geq \cdots \geq \lambda_d\)。在 Courant-Fischer 定理中使用 \(k=d-2\) 得到

\[ \lambda_{d-2} = \min_{\mathbf{u} \in \mathcal{V}_{d-2}} \mathcal{R}_A(\mathbf{u}), \]

其中

\[ \mathcal{V}_{d-2} = \mathrm{span}(\mathbf{v}_1, \ldots, \mathbf{v}_{d-2}). \]

可以看出

\[ \mathrm{span}(\mathbf{v}_1, \ldots, \mathbf{v}_{d-2}) = \mathrm{span}(\mathbf{v}_{d-1}, \mathbf{v}_{d})^\perp. \]

因此，我们最终得到第三小特征值的以下特征描述

\[ \lambda_{d-2} = \min\{\mathcal{R}_A(\mathbf{u})\,:\, \mathbf{0} \neq \mathbf{u}, \langle \mathbf{u},\mathbf{v}_d\rangle = 0, \langle \mathbf{u},\mathbf{v}_{d-1}\rangle = 0\}. \]

使用与 \(\lambda_{d-1}\) 相同的论证，我们得到

\[ \lambda_{d-2} = \min\left\{\langle \mathbf{z}, A \mathbf{z}\rangle\,:\ \|\mathbf{z}\|=1, \langle \mathbf{z}, \mathbf{v}_d\rangle = 0, \langle \mathbf{z}, \mathbf{v}_{d-1}\rangle = 0 \right\}. \]

\(\lhd\)

现在我们给出 Courant-Fischer 定理的证明。局部公式是从推导上述特殊情况相同的论证中得出的，因此省略了通用证明（但尝试证明它！）全局公式需要新的想法。

*证明:* *(Courant-Fischer)* \(\idx{Courant-Fischer 定理}\xdi\) 由于\(\mathcal{V}_k\)的维度为\(k\)，根据局部公式，可以得出

\[ \lambda_k = \min_{\mathbf{u} \in \mathcal{V}_k} \mathcal{R}_A(\mathbf{u}) \leq \max_{\mathrm{dim}(\mathcal{V}) = k} \min_{\mathbf{u} \in \mathcal{V}} \mathcal{R}_A(\mathbf{u}). \]

设\(\mathcal{V}\)为任意维度为\(k\)的子空间。因为\(\mathcal{W}_{d-k+1}\)的维度为\(d - k + 1\)，所以我们有\(\dim(\mathcal{V}) + \mathrm{dim}(\mathcal{W}_{d-k+1}) > d\)，并且在\(\mathcal{V} \cap \mathcal{W}_{d-k+1}\)的交集中必须存在非零向量\(\mathbf{u}_0\)（证明它！）。

然后，根据另一个局部公式，我们有

\[ \lambda_k = \max_{\mathbf{u} \in \mathcal{W}_{d-k+1}} \mathcal{R}_A(\mathbf{u}) \geq \mathcal{R}_A(\mathbf{u}_0) \geq \min_{\mathbf{u} \in \mathcal{V}} \mathcal{R}_A(\mathbf{u}). \]

由于这个不等式对任何维度为\(k\)的子空间都成立，所以我们有

\[ \lambda_k \geq \max_{\mathrm{dim}(\mathcal{V}) = k} \min_{\mathbf{u} \in \mathcal{V}} \mathcal{R}_A(\mathbf{u}). \]

结合上述反向的不等式，得出所声称的结论。另一个全局公式以类似的方式证明。\(\square\)

***自我评估测验*** *(由 Claude, Gemini 和 ChatGPT 协助)*

**1** 根据对称矩阵\(A\)的最大特征值\(\lambda_1\)的变分特征，以下哪个是正确的？

a) \(\lambda_1 = \min_{\mathbf{u} \neq 0} R_A(\mathbf{u})\)

b) \(\lambda_1 = \max_{\mathbf{u} \neq 0} R_A(\mathbf{u})\)

c) \(\lambda_1 = \min_{\|\mathbf{u}\| = 0} R_A(\mathbf{u})\)

d) \(\lambda_1 = \max_{\|\mathbf{u}\| = 0} R_A(\mathbf{u})\)

**2** 设\(\mathcal{V}_{d-1} = \mathrm{span}(\mathbf{v}_1, \ldots, \mathbf{v}_{d-1})\)，其中\(\mathbf{v}_1, \ldots, \mathbf{v}_d\)是具有特征值\(\lambda_1 \geq \cdots \geq \lambda_d\)的对称矩阵\(A\)的特征向量。以下哪个描述了第二小的特征值\(\lambda_{d-1}\)？

a) \(\lambda_{d-1} = \min_{0 \neq \mathbf{u} \in \mathcal{V}_{d-1}} R_A(\mathbf{u})\)

b) \(\lambda_{d-1} = \max_{0 \neq \mathbf{u} \in \mathcal{V}_{d-1}} R_A(\mathbf{u})\)

c) \(\lambda_{d-1} = \min_{\|\mathbf{u}\| = 0, \mathbf{u} \in \mathcal{V}_{d-1}} R_A(\mathbf{u})\)

d) \(\lambda_{d-1} = \max_{\|\mathbf{u}\| = 0, \mathbf{u} \in \mathcal{V}_{d-1}} R_A(\mathbf{u})\)

**3** 设 \(\mathcal{W}_2 = \mathrm{span}(\mathbf{v}_{d-1}, \mathbf{v}_d)\)，其中 \(\mathbf{v}_1, \ldots, \mathbf{v}_d\) 是对称矩阵 \(A\) 的特征向量，其特征值为 \(\lambda_1 \geq \cdots \geq \lambda_d\)。以下哪个选项描述了第二个最小的特征值 \(\lambda_{d-1}\)？

a) \(\lambda_{d-1} = \min_{0 \neq \mathbf{u} \in \mathcal{W}_2} R_A(\mathbf{u})\)

b) \(\lambda_{d-1} = \max_{0 \neq \mathbf{u} \in \mathcal{W}_2} R_A(\mathbf{u})\)

c) \(\lambda_{d-1} = \min_{\|\mathbf{u}\| = 0, \mathbf{u} \in \mathcal{W}_2} R_A(\mathbf{u})\)

d) \(\lambda_{d-1} = \max_{\|\mathbf{u}\| = 0, \mathbf{u} \in \mathcal{W}_2} R_A(\mathbf{u})\)

**4** 根据 Courant-Fischer 定理，以下哪个是对称矩阵 \(A\) 的第 \(k\) 个特征值 \(\lambda_k\) 的全局公式？

a) \(\lambda_k = \min_{\mathbf{u} \in \mathcal{V}_k} R_A(\mathbf{u}) = \max_{\mathbf{u} \in \mathcal{W}_{d-k+1}} R_A(\mathbf{u})\)

b) \(\lambda_k = \max_{\mathbf{u} \in \mathcal{V}_k} R_A(\mathbf{u}) = \min_{\mathbf{u} \in \mathcal{W}_{d-k+1}} R_A(\mathbf{u})\)

c) \(\lambda_k = \max_{\mathrm{dim}(\mathcal{V}) = k} \min_{\mathbf{u} \in \mathcal{V}} R_A(\mathbf{u}) = \min_{\mathrm{dim}(\mathcal{W}) = d-k+1} \max_{\mathbf{u} \in \mathcal{W}} R_A(\mathbf{u})\)

d) \(\lambda_k = \max_{\mathrm{dim}(\mathcal{V}) = k} \max_{\mathbf{u} \in \mathcal{V}} R_A(\mathbf{u}) = \min_{\mathrm{dim}(\mathcal{W}) = d-k+1} \min_{\mathbf{u} \in \mathcal{W}} R_A(\mathbf{u})\)

**5** Courant-Fischer 定理中局部公式和全局公式的主要区别是什么？

a) 局部公式比全局公式更容易计算。

b) 局部公式依赖于特定的特征向量选择，而全局公式则不然。

c) 局部公式仅适用于对称矩阵，而全局公式适用于任何矩阵。

d) 局部公式提供了特征值的上界，而全局公式提供了下界。

1 的答案：b. 理由：文本确立了 \(\lambda_1 = \max_{\mathbf{u} \neq 0} R_A(\mathbf{u})\)。

2 的答案：a. 理由：文本确立了 \(\lambda_{d-1} = \min_{0 \neq \mathbf{u} \in \mathcal{V}_{d-1}} R_A(\mathbf{u})\)。

3 的答案：b. 理由：文本确立了 \(\lambda_{d-1} = \max_{0 \neq \mathbf{u} \in \mathcal{W}_2} R_A(\mathbf{u})\)。

4 的答案：c. 理由：Courant-Fischer 定理指出，对称矩阵 \(A\) 的第 \(k\) 个特征值的全局公式是 \(\lambda_k = \max_{\mathrm{dim}(\mathcal{V}) = k} \min_{\mathbf{u} \in \mathcal{V}} R_A(\mathbf{u}) = \min_{\mathrm{dim}(\mathcal{W}) = d-k+1} \max_{\mathbf{u} \in \mathcal{W}} R_A(\mathbf{u})\)。

5 的答案：b. 理由：文本强调全局公式“不依赖于谱分解的选择”，与依赖于特定一组特征向量的局部公式不同。

## 5.3.1\. 证明谱定理#

我们将需要以下分块矩阵公式。假设

\[\begin{split} A = \begin{pmatrix} A_{11} \\ A_{21} \end{pmatrix} \quad \text{和} \quad B = \begin{pmatrix} B_{11} & B_{12} \end{pmatrix} \end{split}\]

其中 \(A_{11} \in \mathbb{R}^{n_1\times p}\)，\(A_{21} \in \mathbb{R}^{n_2\times p}\)，\(B_{11} \in \mathbb{R}^{p\times n_1}\)，\(B_{12} \in \mathbb{R}^{p\times n_2}\)，那么

\[\begin{split} A B = \begin{pmatrix} A_{11} B_{11} & A_{11} B_{12} \\ A_{21} B_{11} & A_{21} B_{12} \end{pmatrix}. \end{split}\]

实际上，这仅仅是之前遇到的 \(2 \times 2\) 分块矩阵乘法公式的特例，其中 \(A_{12}, A_{22}, B_{21}, B_{22}\) 是空块。

*证明思路（谱定理）：* 类似于我们如何使用豪斯霍尔德变换来“在主对角线下添加零”，这里我们将使用一系列正交变换来在主对角线以上和以下都添加零。回想一下，两个矩阵 \(C, D \in \mathbb{R}^{d \times d}\) 是 [相似](https://en.wikipedia.org/wiki/Matrix_similarity) 的，如果存在一个可逆矩阵 \(P\) 使得 \(C = P^{-1} D P\)，并且当 \(P = W\) 是一个正交矩阵时，变换简化为 \(C = W^T D W\)。

我们构造一系列正交矩阵 \(\hat{W}_1,\ldots, \hat{W}_d\)，并逐步计算 \(W_1^T A W_1\)，\(W_2^T W_1^T A W_1 W_2\)，依此类推，以便

\[ \Lambda = W_d^T \cdots W_2^T W_1^T A W_1 W_2 \cdots W_d \]

是对角矩阵。然后矩阵 \(Q\) 简单地是 \(W_1 W_2 \cdots W_d\)（检查一下！）。

为了定义这些矩阵，我们使用一个贪婪序列来最大化二次型 \(\langle \mathbf{v}, A \mathbf{v}\rangle\)。这个二次型如何与特征值相关？回想一下，对于一个单位特征向量 \(\mathbf{v}\) 和特征值 \(\lambda\)，我们有 \(\langle \mathbf{v}, A \mathbf{v}\rangle = \langle \mathbf{v}, \lambda \mathbf{v}\rangle = \lambda\)。

*证明：* *(谱定理)* \(\idx{spectral theorem}\xdi\) 我们通过归纳法进行证明。我们已经遇到了这样的想法，即特征向量是优化问题的驻点。实际上，证明正是使用了这个想法。

***第一个特征向量：*** 令 \(A_1 = A\)。定义

\[ \mathbf{v}_1 \in \arg\max\{\langle \mathbf{v}, A_1 \mathbf{v}\rangle:\|\mathbf{v}\| = 1\} \]

这是由 *极值定理* 保证的，并进一步定义

\[ \lambda_1 = \max\{\langle \mathbf{v}, A_1 \mathbf{v}\rangle:\|\mathbf{v}\| = 1\}. \]

将 \(\mathbf{v}_1\) 补充为 \(\mathbb{R}^d\) 的一个正交基，\(\mathbf{v}_1, \hat{\mathbf{v}}_2, \ldots, \hat{\mathbf{v}}_d\)，并形成分块矩阵

\[ \hat{W}_1 = \begin{pmatrix} \mathbf{v}_1 & \hat{V}_1 \end{pmatrix} \]

其中 \(\hat{V}_1\) 的列是 \(\hat{\mathbf{v}}_2, \ldots, \hat{\mathbf{v}}_d\)。注意 \(\hat{W}_1\) 是通过构造正交的。

***向对角化迈进一步：*** 我们接下来将展示 \(\hat{W}_1\) 通过相似变换使我们更接近对角矩阵。首先要注意的是

\[\begin{align*} \hat{W}_1^T A_1 \hat{W}_1 &= \begin{pmatrix} \mathbf{v}_1^T \\ \hat{V}_1^T \end{pmatrix} A_1 \begin{pmatrix} \mathbf{v}_1 & \hat{V}_1 \end{pmatrix}\\ &= \begin{pmatrix} \mathbf{v}_1^T \\ \hat{V}_1^T \end{pmatrix} \begin{pmatrix} A_1 \mathbf{v}_1 & A_1 \hat{V}_1 \end{pmatrix}\\ &= \begin{pmatrix} \mathbf{v}_1^T A_1 \mathbf{v}_1 & \mathbf{v}_1^T A_1 \hat{V}_1\\ \hat{V}_1^T A_1 \mathbf{v}_1 & \hat{V}_1^T A_1 \hat{V}_1 \end{pmatrix}\\ &= \begin{pmatrix} \lambda_1 & \mathbf{w}_1^T \\ \mathbf{w}_1 & A_2 \end{pmatrix} \end{align*}\]

其中 \(\mathbf{w}_1 = \hat{V}_1^T A_1 \mathbf{v}_1\) 和 \(A_2 = \hat{V}_1^T A_1 \hat{V}_1\).

关键的断言是 \(\mathbf{w}_1 = \mathbf{0}\)。这可以通过矛盾论证得出。确实，假设 \(\mathbf{w}_1 \neq \mathbf{0}\) 并考虑单位向量（检查 \(\|\mathbf{z}\| = 1\)）

\[\begin{split} \mathbf{z} = \hat{W}_1 \frac{1}{\sqrt{1 + \delta² \|\mathbf{w}_1\|²}} \begin{pmatrix} 1 \\ \delta \mathbf{w}_1 \end{pmatrix} \end{split}\]

它实现了目标值（检查它！）

\[\begin{align*} \mathbf{z}^T A_1 \mathbf{z} &= \frac{1}{1 + \delta² \|\mathbf{w}_1\|²} \begin{pmatrix} 1 \\ \delta \mathbf{w}_1 \end{pmatrix}^T \begin{pmatrix} \lambda_1 & \mathbf{w}_1^T \\ \mathbf{w}_1 & A_2 \end{pmatrix} \begin{pmatrix} 1 \\ \delta \mathbf{w}_1 \end{pmatrix}\\ &= \frac{1}{1 + \delta² \|\mathbf{w}_1\|²} \left( \lambda_1 + 2 \delta \|\mathbf{w}_1\|² + \delta² \mathbf{w}_1^T A_2 \mathbf{w}_1 \right). \end{align*}\]

通过 [几何级数的和](https://en.wikipedia.org/wiki/Geometric_series)，对于 \(\varepsilon \in (0,1)\)，

\[ \frac{1}{1 + \varepsilon²} = 1 - \varepsilon² + \varepsilon⁴ + \cdots. \]

因此，对于足够小的 \(\delta > 0\)，

\[\begin{align*} \mathbf{z}^T A_1 \mathbf{z} &\approx (\lambda_1 + 2 \delta \|\mathbf{w}_1\|² + \delta² \mathbf{w}_1^T A_2 \mathbf{w}_1) (1 - \delta² \|\mathbf{w}_1\|²)\\ &\approx \lambda_1 + 2 \delta \|\mathbf{w}_1\|² + C \delta²\\ &> \lambda_1 \end{align*}\]

其中 \(C \in \mathbb{R}\) 依赖于 \(\mathbf{w}_1\) 和 \(A_2\)，并且我们忽略了包含 \(\delta³, \delta⁴, \delta⁵, \ldots\) 的“高阶”项，当 \(\delta\) 很小时，这些项的总贡献可以忽略不计。这给出了所需的矛盾——也就是说，它将意味着存在一个向量能够达到比最优 \(\mathbf{v}_1\) 更好的严格目标值。（*练习：使这个论证严谨。*）

因此，让 \(W_1 = \hat{W}_1\)，

\[\begin{split} W_1^T A_1 W_1 = \begin{pmatrix} \lambda_1 & \mathbf{0} \\ \mathbf{0} & A_2 \end{pmatrix}. \end{split}\]

最后注意 \(A_2 = \hat{V}_1^T A_1 \hat{V}_1\) 是对称的

\[ A_2^T = (\hat{V}_1^T A_1 \hat{V}_1)^T = \hat{V}_1^T A_1^T \hat{V}_1 = \hat{V}_1^T A_1 \hat{V}_1 = A_2 \]

由 \(A_1\) 本身的对称性。

***归纳的下一步：*** 将相同的论点应用于对称矩阵 \(A_2 \in \mathbb{R}^{(d-1)\times (d-1)}\)，令 \(\hat{W}_2 = (\mathbf{v}_2\ \hat{V}_2) \in \mathbb{R}^{(d-1)\times (d-1)}\) 为相应的正交矩阵，并通过以下方程定义 \(\lambda_2\) 和 \(A_3\)：

\[\begin{split} \hat{W}_2^T A_2 \hat{W}_2 = \begin{pmatrix} \lambda_2 & \mathbf{0} \\ \mathbf{0} & A_3 \end{pmatrix}. \end{split}\]

现在定义分块矩阵

\[\begin{split} W_2 = \begin{pmatrix} 1 & \mathbf{0}\\ \mathbf{0} & \hat{W}_2 \end{pmatrix}. \end{split}\]

注意到（检查它！）

\[\begin{split} W_2^T W_1^T A_1 W_1 W_2 = W_2^T \begin{pmatrix} \lambda_1 & \mathbf{0} \\ \mathbf{0} & A_2 \end{pmatrix} W_2 = \begin{pmatrix} \lambda_1 & \mathbf{0}\\ \mathbf{0} & \hat{W}_2^T A_2 \hat{W}_2 \end{pmatrix} =\begin{pmatrix} \lambda_1 & 0 & \mathbf{0} \\ 0 & \lambda_2 & \mathbf{0} \\ \mathbf{0} & \mathbf{0} & A_3 \end{pmatrix}. \end{split}\]

通过类似的归纳方法可以得到结论。 \(\square\)

关于证明的一些评论：

1- 通过向量微积分使用拉格朗日乘数法在 \(\max\{\langle \mathbf{v}, A_1 \mathbf{v}\rangle:\|\mathbf{v}\| = 1\}\) 上进行操作，可以直观地理解 \(\mathbf{w}_1 = \mathbf{0}\) 的事实，因为 \(A_1 \mathbf{v}_1\) 必须与 \(\mathbf{v}_1\) 成比例。因此，根据 \(\hat{V}_1^T\) 的构造，\(\mathbf{w}_1 = \hat{V}_1^T A_1 \mathbf{v}_1 = \mathbf{0}\)。实际上，定义拉格朗日函数

\[ L(\mathbf{v}, \lambda) = \langle \mathbf{v}, A_1 \mathbf{v}\rangle - \lambda(\|\mathbf{v}\|² - 1). \]

局部极大值 \(\mathbf{v}_1\) 的一阶必要条件是（检查它！）

\[ \nabla_{\mathbf{v}} L(\mathbf{v}_1, \lambda_1) = 2A_1 \mathbf{v}_1 - 2\lambda_1 \mathbf{v}_1 = \mathbf{0} \]\[ \nabla_{\lambda} L(\mathbf{v}_1, \lambda_1) = \|\mathbf{v}_1\|² - 1 = 0. \]

从第一个条件，我们有

\[ A_1 \mathbf{v}_1 = \lambda_1 \mathbf{v}_1. \]

这表明 \(A_1 \mathbf{v}_1\) 与 \(\mathbf{v}_1\) 成比例，正如所声称的那样。

2- 通过构造，向量 \(\mathbf{v}_2\)（即 \(\hat{W}_2\) 的第一列）是以下方程的解

\[ \mathbf{v}_2 \in \arg\max\{\langle \mathbf{v}, A_2 \mathbf{v}\rangle:\|\mathbf{v}\| = 1\}. \]

注意，根据 \(A_2\) 的定义（以及 \(A_1 = A\) 的事实），

\[ \mathbf{v}^T A_2 \mathbf{v} = \mathbf{v}^T \hat{V}_1^T A_1 \hat{V}_1 \mathbf{v} = (\hat{V}_1 \mathbf{v})^T \,A \,(\hat{V}_1 \mathbf{v}). \]

因此，我们可以将解 \(\mathbf{v}_2\) 视为指定 \(\hat{V}_1\) 的列的优线性组合——这些列形成 \(\mathrm{span}(\mathbf{v}_1)^\perp\) 空间的基，该空间是垂直于 \(\mathbf{v}_1\) 的向量的空间。本质上，\(\mathbf{v}_2\) 解决了与 \(\mathbf{v}_1\) 相同的问题，*但限制在 \(\mathrm{span}(\mathbf{v}_1)^\perp\)*。我们将在下面回到这个问题。

## 5.3.2\. 变分特征：特殊情况#

我们从定义开始。

**定义** **（雷利商）** \(\idx{Rayleigh quotient}\xdi\) 设 \(A \in \mathbb{R}^{d \times d}\) 是一个对称矩阵。雷利商定义为

\[ \mathcal{R}_A(\mathbf{u}) = \frac{\langle \mathbf{u}, A \mathbf{u} \rangle}{\langle \mathbf{u}, \mathbf{u} \rangle} \]

这是对 \(\mathbb{R}^{d}\) 中任何 \(\mathbf{u} \neq \mathbf{0}\) 定义的。 \(\natural\)

为了开始看到与谱分解的联系，设 \(\mathbf{v}\) 是 \(A\) 的一个（不一定为单位）特征向量，其特征值为 \(\lambda\)。可以证明 \(\mathcal{R}_A(\mathbf{v}) = \lambda\)。（试一试！）实际上，回想一下，我们之前已经证明了 \(A\) 的特征向量是 \(\mathcal{R}_A\) 的驻点。

在陈述一般的变分特征之前，我们证明几个特殊情况。在整个过程中，设 \(A \in \mathbb{R}^{d \times d}\) 是一个对称矩阵，其谱分解为 \(A = \sum_{i=1}^d \lambda_i \mathbf{v}_i \mathbf{v}_i^T\)，其中 \(\lambda_1 \geq \cdots \geq \lambda_d\)。

*最大特征值:* 由于 \(\mathbf{v}_1, \ldots, \mathbf{v}_d\) 构成了 \(\mathbb{R}^d\) 的一个正交归一基，任何非零向量 \(\mathbf{u}\) 可以表示为 \(\mathbf{u} = \sum_{i=1}^d \langle \mathbf{u}, \mathbf{v}_i \rangle \mathbf{v}_i\)，并且根据正交归一列表的性质和内积的双线性，我们得到

\[ \langle \mathbf{u}, \mathbf{u} \rangle = \|\mathbf{u}\|² = \left\|\sum_{i=1}^d \langle \mathbf{u}, \mathbf{v}_i \rangle \mathbf{v}_i\right\|² = \sum_{i=1}^d \langle \mathbf{u}, \mathbf{v}_i \rangle² \]\[ \langle \mathbf{u}, A \mathbf{u} \rangle = \left\langle \mathbf{u}, \sum_{i=1}^d \langle \mathbf{u}, \mathbf{v}_i \rangle A \mathbf{v}_i \right\rangle = \left\langle \mathbf{u}, \sum_{i=1}^d \langle \mathbf{u}, \mathbf{v}_i \rangle \lambda_i \mathbf{v}_i \right\rangle = \sum_{i=1}^d \lambda_i \langle \mathbf{u}, \mathbf{v}_i \rangle². \]

因此，

\[ \mathcal{R}_A(\mathbf{u}) = \frac{\langle \mathbf{u}, A \mathbf{u} \rangle}{\langle \mathbf{u}, \mathbf{u} \rangle} = \frac{\sum_{i=1}^d \lambda_i \langle \mathbf{u}, \mathbf{v}_i \rangle²}{\sum_{i=1}^d \langle \mathbf{u}, \mathbf{v}_i \rangle²} \leq \lambda_1 \frac{\sum_{i=1}^d \langle \mathbf{u}, \mathbf{v}_i \rangle²}{\sum_{i=1}^d \langle \mathbf{u}, \mathbf{v}_i \rangle²} = \lambda_1, \]

其中我们使用了 \(\lambda_1 \geq \cdots \geq \lambda_d\) 和 \(\langle \mathbf{u}, \mathbf{v}_i \rangle² \geq 0\) 的事实。此外，\(\mathcal{R}_A(\mathbf{v}_1) = \lambda_1\)。因此，我们得到

\[ \lambda_1 = \max_{\mathbf{u} \neq \mathbf{0}} \mathcal{R}_A(\mathbf{u}). \]

*最小特征值:* 从相反的方向进行论证，我们得到最小特征值的特征。使用与之前相同的符号，我们有

\[ \mathcal{R}_A(\mathbf{u}) = \frac{\langle \mathbf{u}, A \mathbf{u} \rangle}{\langle \mathbf{u}, \mathbf{u} \rangle} = \frac{\sum_{i=1}^d \lambda_i \langle \mathbf{u}, \mathbf{v}_i \rangle²}{\sum_{i=1}^d \langle \mathbf{u}, \mathbf{v}_i \rangle²} \geq \lambda_d \frac{\sum_{i=1}^d \langle \mathbf{u}, \mathbf{v}_i \rangle²}{\sum_{i=1}^d \langle \mathbf{u}, \mathbf{v}_i \rangle²} = \lambda_d, \]

在这里，我们再次使用了 \(\lambda_1 \geq \cdots \geq \lambda_d\) 以及 \(\langle \mathbf{u}, \mathbf{v}_i \rangle² \geq 0\) 的性质。此外，\(\mathcal{R}_A(\mathbf{v}_d) = \lambda_d\)。因此，我们已经建立了

\[ \lambda_d = \min_{\mathbf{u} \neq \mathbf{0}} \mathcal{R}_A(\mathbf{u}). \]

*第二小的特征值:* 为了挑选出第二小的特征值，我们按照上述方法进行论证，但将优化限制在空间 \(\mathcal{V}_{d-1} = \mathrm{span}(\mathbf{v}_1,\ldots,\mathbf{v}_{d-1})\)。实际上，如果 \(\mathbf{u}\) 在线性子空间 \(\mathcal{V}_{d-1}\) 中，它可以写成 \(\mathbf{u} = \sum_{i=1}^{d-1} \langle \mathbf{u}, \mathbf{v}_i \rangle \mathbf{v}_i\)（因为 \(\mathbf{v}_1,\ldots,\mathbf{v}_{d-1}\) 形成了它的正交基；为什么？）并且由此得出

\[ \langle \mathbf{u}, \mathbf{u} \rangle = \sum_{i=1}^{d-1} \langle \mathbf{u}, \mathbf{v}_i \rangle² \]\[ \langle \mathbf{u}, A \mathbf{u} \rangle = \left\langle \mathbf{u}, \sum_{i=1}^{d-1} \langle \mathbf{u}, \mathbf{v}_i \rangle \lambda_i \mathbf{v}_i \right\rangle = \sum_{i=1}^{d-1} \lambda_i \langle \mathbf{u}, \mathbf{v}_i \rangle². \]

因此，

\[ \mathcal{R}_A(\mathbf{u}) = \frac{\langle \mathbf{u}, A \mathbf{u} \rangle}{\langle \mathbf{u}, \mathbf{u} \rangle} = \frac{\sum_{i=1}^{d-1} \lambda_i \langle \mathbf{u}, \mathbf{v}_i \rangle²}{\sum_{i=1}^{d-1} \langle \mathbf{u}, \mathbf{v}_i \rangle²} \geq \lambda_{d-1} \frac{\sum_{i=1}^{d-1} \langle \mathbf{u}, \mathbf{v}_i \rangle²}{\sum_{i=1}^{d-1} \langle \mathbf{u}, \mathbf{v}_i \rangle²} = \lambda_{d-1}, \]

在这里，我们使用了 \(\lambda_1 \geq \cdots \geq \lambda_{d-1}\) 以及 \(\langle \mathbf{u}, \mathbf{v}_i \rangle² \geq 0\) 的性质。此外，\(\mathcal{R}_A(\mathbf{v}_{d-1}) = \lambda_{d-1}\) 并且当然 \(\mathbf{v}_{d-1} \in \mathcal{V}_{d-1}\)。因此，我们已经建立了

\[ \lambda_{d-1} = \min_{\mathbf{0} \neq \mathbf{u} \in \mathcal{V}_{d-1}} \mathcal{R}_A(\mathbf{u}). \]

现在什么是 \(\mathcal{V}_{d-1}\)？它是由正交列表 \(\mathbf{v}_1,\ldots,\mathbf{v}_{d-1}\) 张成的，每个都与 \(\mathbf{v}_d\) 正交。所以（为什么？）

\[ \mathcal{V}_{d-1} = \mathrm{span}(\mathbf{v}_d)^\perp. \]

因此，等价地，

\[ \lambda_{d-1} = \min\left\{\mathcal{R}_A(\mathbf{u})\,:\ \mathbf{u} \neq \mathbf{0}, \langle \mathbf{u}, \mathbf{v}_d\rangle = 0 \right\}. \]

事实上，对于任何 \(\mathbf{u} \neq \mathbf{0}\)，我们可以通过定义 \(\mathbf{z} = \mathbf{u}/\|\mathbf{u}\|\) 来归一化它，并且我们注意到

\[ \mathcal{R}_A(\mathbf{u}) = \frac{\langle \mathbf{u}, A \mathbf{u}\rangle}{\langle \mathbf{u},\mathbf{u}\rangle} = \frac{\langle \mathbf{u}, A \mathbf{u}\rangle}{\|\mathbf{u}\|²} = \left\langle \frac{\mathbf{u}}{\|\mathbf{u}\|}, A \frac{\mathbf{u}}{\|\mathbf{u}\|}\right\rangle = \langle \mathbf{z}, A \mathbf{z}\rangle. \]

因此，

\[ \lambda_{d-1} = \min\left\{\langle \mathbf{z}, A \mathbf{z}\rangle\,:\ \|\mathbf{z}\|=1, \langle \mathbf{z}, \mathbf{v}_d\rangle = 0 \right\}. \]

## 5.3.3\. 一般性陈述：Courant-Fischer#

在陈述一个一般结果之前，我们再举一个例子。

*第二个最小的特征值（取两个）：* 有趣的是，第二个最小的特征值还有一个特征。实际上，如果我们将优化限制在空间 \(\mathcal{W}_{2} = \mathrm{span}(\mathbf{v}_{d-1},\mathbf{v}_d)\) 上，那么。如果 \(\mathbf{u}\) 在线性子空间 \(\mathcal{W}_{2}\) 中，它可以写成 \(\mathbf{u} = \sum_{i=d-1}^{d} \langle \mathbf{u}, \mathbf{v}_i \rangle \mathbf{v}_i\)（因为 \(\mathbf{v}_{d-1},\mathbf{v}_{d}\) 形成了它的正交基），并且由此得出

\[ \langle \mathbf{u}, \mathbf{u} \rangle = \sum_{i=d-1}^{d} \langle \mathbf{u}, \mathbf{v}_i \rangle² \]\[ \langle \mathbf{u}, A \mathbf{u} \rangle = \left\langle \mathbf{u}, \sum_{i=d-1}^{d} \langle \mathbf{u}, \mathbf{v}_i \rangle \lambda_i \mathbf{v}_i \right\rangle = \sum_{i=d-1}^{d} \lambda_i \langle \mathbf{u}, \mathbf{v}_i \rangle². \]

因此，

\[ \mathcal{R}_A(\mathbf{u}) = \frac{\langle \mathbf{u}, A \mathbf{u} \rangle}{\langle \mathbf{u}, \mathbf{u} \rangle} = \frac{\sum_{i=d-1}^{d} \lambda_i \langle \mathbf{u}, \mathbf{v}_i \rangle²}{\sum_{i=d-1}^{d} \langle \mathbf{u}, \mathbf{v}_i \rangle²} \leq \lambda_{d-1} \frac{\sum_{i=d-1}^{d} \langle \mathbf{u}, \mathbf{v}_i \rangle²}{\sum_{i=d-1}^{d} \langle \mathbf{u}, \mathbf{v}_i \rangle²} = \lambda_{d-1}, \]

其中我们使用了 \(\lambda_{d-1} \geq \lambda_{d}\) 以及 \(\langle \mathbf{u}, \mathbf{v}_i \rangle² \geq 0\) 的性质。此外，\(\mathcal{R}_A(\mathbf{v}_{d-1}) = \lambda_{d-1}\)。因此，我们得到

\[ \lambda_{d-1} = \max_{\mathbf{0} \neq \mathbf{u} \in \mathcal{W}_{2}} \mathcal{R}_A(\mathbf{u}). \]

**定理** **(Courant-Fischer)** \(\idx{Courant-Fischer 定理}\xdi\) 设 \(A \in \mathbb{R}^{d \times d}\) 是一个对称矩阵，其谱分解为 \(A = \sum_{i=1}^d \lambda_i \mathbf{v}_i \mathbf{v}_i^T\)，其中 \(\lambda_1 \geq \cdots \geq \lambda_d\)。对于每个 \(k = 1,\ldots,d\)，定义子空间

\[ \mathcal{V}_k = \mathrm{span}(\mathbf{v}_1, \ldots, \mathbf{v}_k) \quad\text{和}\quad \mathcal{W}_{d-k+1} = \mathrm{span}(\mathbf{v}_k, \ldots, \mathbf{v}_d). \]

然后，对于所有 \(k = 1,\ldots,d\)，

\[ \lambda_k = \min_{\mathbf{u} \in \mathcal{V}_k} \mathcal{R}_A(\mathbf{u}) = \max_{\mathbf{u} \in \mathcal{W}_{d-k+1}} \mathcal{R}_A(\mathbf{u}), \]

这些被称为局部公式。此外，我们还有以下最小-最大（或全局）公式，这些公式不依赖于谱分解的选择：对于所有 \(k = 1,\ldots,d\)，

\[ \lambda_k = \max_{\mathrm{dim}(\mathcal{V}) = k} \min_{\mathbf{u} \in \mathcal{V}} \mathcal{R}_A(\mathbf{u}) = \min_{\mathrm{dim}(\mathcal{W}) = d-k+1} \max_{\mathbf{u} \in \mathcal{W}} \mathcal{R}_A(\mathbf{u}). \]

\(\sharp\)

*证明思路:* 对于局部公式，我们将 \(\mathcal{V}_k\) 中的向量展开到基 \(\mathbf{v}_1,\ldots,\mathbf{v}_k\) 中，并使用 \(\mathcal{R}_A(\mathbf{v}_i) = \lambda_i\) 和特征值非递增的已知事实。全局公式则由维度论证得出。

**示例:** **(第三小特征值)** 可以使用 *Courant-Fischer 定理* 恢复前一小节中的特殊情况。还可以得到一些新的有趣情况。接下来，我们将给出第三小特征值的描述。设 \(A \in \mathbb{R}^{d \times d}\) 为一个具有谱分解 \(A = \sum_{i=1}^d \lambda_i \mathbf{v}_i \mathbf{v}_i^T\) 的对称矩阵，其中 \(\lambda_1 \geq \cdots \geq \lambda_d\)。使用 *Courant-Fischer 定理* 中的 \(k=d-2\) 得到

\[ \lambda_{d-2} = \min_{\mathbf{u} \in \mathcal{V}_{d-2}} \mathcal{R}_A(\mathbf{u}), \]

其中

\[ \mathcal{V}_{d-2} = \mathrm{span}(\mathbf{v}_1, \ldots, \mathbf{v}_{d-2}). \]

可以看出，

\[ \mathrm{span}(\mathbf{v}_1, \ldots, \mathbf{v}_{d-2}) = \mathrm{span}(\mathbf{v}_{d-1}, \mathbf{v}_{d})^\perp. \]

因此，我们最终得到以下关于第三小特征值的描述

\[ \lambda_{d-2} = \min\{\mathcal{R}_A(\mathbf{u})\,:\, \mathbf{0} \neq \mathbf{u}, \langle \mathbf{u},\mathbf{v}_d\rangle = 0, \langle \mathbf{u},\mathbf{v}_{d-1}\rangle = 0\}. \]

使用与 \(\lambda_{d-1}\) 相同的论证，我们还可以得到

\[ \lambda_{d-2} = \min\left\{\langle \mathbf{z}, A \mathbf{z}\rangle\,:\ \|\mathbf{z}\|=1, \langle \mathbf{z}, \mathbf{v}_d\rangle = 0, \langle \mathbf{z}, \mathbf{v}_{d-1}\rangle = 0 \right\}. \]

\(\lhd\)

我们现在给出 *Courant-Fischer 定理* 的证明。局部公式与推导上述特殊情况所用的相同论证一致，因此省略一般证明（但尝试证明它！）全局公式需要新的想法。

*证明:* *(Courant-Fischer)* \(\idx{Courant-Fischer theorem}\xdi\) 由于 \(\mathcal{V}_k\) 的维度为 \(k\)，根据局部公式，可以得出

\[ \lambda_k = \min_{\mathbf{u} \in \mathcal{V}_k} \mathcal{R}_A(\mathbf{u}) \leq \max_{\mathrm{dim}(\mathcal{V}) = k} \min_{\mathbf{u} \in \mathcal{V}} \mathcal{R}_A(\mathbf{u}). \]

设 \(\mathcal{V}\) 为任意维度为 \(k\) 的子空间。因为 \(\mathcal{W}_{d-k+1}\) 的维度为 \(d - k + 1\)，所以我们有 \(\dim(\mathcal{V}) + \mathrm{dim}(\mathcal{W}_{d-k+1}) > d\)，并且在 \(\mathcal{V} \cap \mathcal{W}_{d-k+1}\) 的交集中必须存在非零向量 \(\mathbf{u}_0\)（证明它！）。

我们通过另一个局部公式得到：

\[ \lambda_k = \max_{\mathbf{u} \in \mathcal{W}_{d-k+1}} \mathcal{R}_A(\mathbf{u}) \geq \mathcal{R}_A(\mathbf{u}_0) \geq \min_{\mathbf{u} \in \mathcal{V}} \mathcal{R}_A(\mathbf{u}). \]

由于这个不等式对任何维度为 \(k\) 的子空间都成立，我们有

\[ \lambda_k \geq \max_{\mathrm{dim}(\mathcal{V}) = k} \min_{\mathbf{u} \in \mathcal{V}} \mathcal{R}_A(\mathbf{u}). \]

结合上述反向不等式，得出结论。另一个全局公式也以类似方式证明。\(\square\)

***自我评估测验*** *(由克莱德、双子星和 ChatGPT 协助)*

**1** 根据对称矩阵 \(A\) 的最大特征值 \(\lambda_1\) 的变分特征，以下哪个是正确的？

a) \(\lambda_1 = \min_{\mathbf{u} \neq 0} R_A(\mathbf{u})\)

b) \(\lambda_1 = \max_{\mathbf{u} \neq 0} R_A(\mathbf{u})\)

c) \(\lambda_1 = \min_{\|\mathbf{u}\| = 0} R_A(\mathbf{u})\)

d) \(\lambda_1 = \max_{\|\mathbf{u}\| = 0} R_A(\mathbf{u})\)

**2** 设 \(\mathcal{V}_{d-1} = \mathrm{span}(\mathbf{v}_1, \ldots, \mathbf{v}_{d-1})\)，其中 \(\mathbf{v}_1, \ldots, \mathbf{v}_d\) 是具有特征值 \(\lambda_1 \geq \cdots \geq \lambda_d\) 的对称矩阵 \(A\) 的特征向量。以下哪个描述了第二个最小的特征值 \(\lambda_{d-1}\)？

a) \(\lambda_{d-1} = \min_{0 \neq \mathbf{u} \in \mathcal{V}_{d-1}} R_A(\mathbf{u})\)

b) \(\lambda_{d-1} = \max_{0 \neq \mathbf{u} \in \mathcal{V}_{d-1}} R_A(\mathbf{u})\)

c) \(\lambda_{d-1} = \min_{\|\mathbf{u}\| = 0, \mathbf{u} \in \mathcal{V}_{d-1}} R_A(\mathbf{u})\)

d) \(\lambda_{d-1} = \max_{\|\mathbf{u}\| = 0, \mathbf{u} \in \mathcal{V}_{d-1}} R_A(\mathbf{u})\)

**3** 设 \(\mathcal{W}_2 = \mathrm{span}(\mathbf{v}_{d-1}, \mathbf{v}_d)\)，其中 \(\mathbf{v}_1, \ldots, \mathbf{v}_d\) 是具有特征值 \(\lambda_1 \geq \cdots \geq \lambda_d\) 的对称矩阵 \(A\) 的特征向量。以下哪个描述了第二个最小的特征值 \(\lambda_{d-1}\)？

a) \(\lambda_{d-1} = \min_{0 \neq \mathbf{u} \in \mathcal{W}_2} R_A(\mathbf{u})\)

b) \(\lambda_{d-1} = \max_{0 \neq \mathbf{u} \in \mathcal{W}_2} R_A(\mathbf{u})\)

c) \(\lambda_{d-1} = \min_{\|\mathbf{u}\| = 0, \mathbf{u} \in \mathcal{W}_2} R_A(\mathbf{u})\)

d) \(\lambda_{d-1} = \max_{\|\mathbf{u}\| = 0, \mathbf{u} \in \mathcal{W}_2} R_A(\mathbf{u})\)

**4** 根据 *Courant-Fischer 定理*，以下哪个是对称矩阵 \(A\) 的第 \(k\) 个特征值 \(\lambda_k\) 的全局公式？

a) \(\lambda_k = \min_{\mathbf{u} \in \mathcal{V}_k} R_A(\mathbf{u}) = \max_{\mathbf{u} \in \mathcal{W}_{d-k+1}} R_A(\mathbf{u})\)

b) \(\lambda_k = \max_{\mathbf{u} \in \mathcal{V}_k} R_A(\mathbf{u}) = \min_{\mathbf{u} \in \mathcal{W}_{d-k+1}} R_A(\mathbf{u})\)

c) \(\lambda_k = \max_{\mathrm{dim}(\mathcal{V}) = k} \min_{\mathbf{u} \in \mathcal{V}} R_A(\mathbf{u}) = \min_{\mathrm{dim}(\mathcal{W}) = d-k+1} \max_{\mathbf{u} \in \mathcal{W}} R_A(\mathbf{u})\)

d) \(\lambda_k = \max_{\mathrm{dim}(\mathcal{V}) = k} \max_{\mathbf{u} \in \mathcal{V}} R_A(\mathbf{u}) = \min_{\mathrm{dim}(\mathcal{W}) = d-k+1} \min_{\mathbf{u} \in \mathcal{W}} R_A(\mathbf{u})\)

**5** Courant-Fischer 定理中局部公式和全局公式的主要区别是什么？

a) 局部公式比全局公式更容易计算。

b) 局部公式依赖于特定的特征向量选择，而全局公式则不依赖于。

c) 局部公式仅适用于对称矩阵，而全局公式适用于任何矩阵。

d) 局部公式提供了特征值的上界，而全局公式提供了下界。

答案 1：b. 证明：文本确立了 \(\lambda_1 = \max_{\mathbf{u} \neq 0} R_A(\mathbf{u})\)。

答案 2：a. 证明：文本确立了 \(\lambda_{d-1} = \min_{0 \neq \mathbf{u} \in \mathcal{V}_{d-1}} R_A(\mathbf{u})\)。

答案 3：b. 证明：文本确立了 \(\lambda_{d-1} = \max_{0 \neq \mathbf{u} \in \mathcal{W}_2} R_A(\mathbf{u})\)。

答案 4：c. 证明：Courant-Fischer 定理指出，第 \(k\) 个特征值的全局公式是 \(\lambda_k = \max_{\mathrm{dim}(\mathcal{V}) = k} \min_{\mathbf{u} \in \mathcal{V}} R_A(\mathbf{u}) = \min_{\mathrm{dim}(\mathcal{W}) = d-k+1} \max_{\mathbf{u} \in \mathcal{W}} R_A(\mathbf{u})\)。

答案 5：b. 证明：文本强调全局公式“不依赖于谱分解的选择”，这与依赖于特定一组特征向量的局部公式不同。
