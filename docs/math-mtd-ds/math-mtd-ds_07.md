# 1.5\. 练习题

> 原文：[`mmids-textbook.github.io/chap01_intro/exercises/roch-mmids-intro-exercises.html`](https://mmids-textbook.github.io/chap01_intro/exercises/roch-mmids-intro-exercises.html)

## 1.5.1\. 预习工作表#

*(在 Claude、Gemini 和 ChatGPT 的帮助下)*

**第 1.2 节**

**E1.2.1** 计算向量 \(\mathbf{x} = (6, 8)\) 的欧几里得范数。

**E1.2.2** 计算向量 \(\mathbf{u} = (1, 2, 3)\) 和 \(\mathbf{v} = (4, 5, 6)\) 的内积。

**E1.2.3** 找到矩阵 \(A = \begin{pmatrix}1 & 2 & 3 \\ 4 & 5 & 6\end{pmatrix}\) 的转置。

**E1.2.4** 计算矩阵 \(A = \begin{pmatrix}1 & 2 \\ 3 & 4\end{pmatrix}\) 的弗罗贝尼乌斯范数。

**E1.2.5** 验证矩阵 \(A = \begin{pmatrix}2 & 0 \\ 0 & 3\end{pmatrix}\) 是否是对称的。

**E1.2.6** 设 \(A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}\) 和 \(B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}\)。计算 \(AB\) 和 \(BA\)。

**E1.2.7** 设 \(f(x, y) = x² + xy + y²\)。计算偏导数 \(\frac{\partial f}{\partial x}\) 和 \(\frac{\partial f}{\partial y}\)。

**E1.2.8** 给定函数 \(g(x, y) = x² y + xy³\)，计算其偏导数 \(\frac{\partial g}{\partial x}\) 和 \(\frac{\partial g}{\partial y}\)。

**E1.2.9** 设 \(f(x) = x³ - 3x² + 2x\)。使用 *泰勒定理* 在 \(a = 1\) 附近找到 \(f\) 的线性近似，并带有二阶误差项。

**E1.2.10** 计算离散随机变量 \(X\) 的期望值 \(\E[X]\)，其中 \(\P(X = 1) = 0.2\)，\(\P(X = 2) = 0.5\)，和 \(\P(X = 3) = 0.3\)。

**E1.2.11:** 计算离散随机变量 \(X\) 的方差 \(\mathrm{Var}[X]\)，其中 \(\P(X = 1) = 0.4\)，\(\P(X = 2) = 0.6\)，和 \(\E[X] = 1.6\)。

**E1.2.12** 设 \(X\) 和 \(Y\) 是随机变量，其期望值 \(\mathbb{E}[X] = 2\)，\(\mathbb{E}[Y] = 3\)，方差 \(\mathrm{Var}[X] = 4\)，和 \(\mathrm{Var}[Y] = 9\)。在 \(X\) 和 \(Y\) 相互独立的情况下，计算 \(\mathbb{E}[2X + Y - 1]\) 和 \(\mathrm{Var}[2X + Y - 1]\)。

**E1.2.13** 设 \(X\) 为一个随机变量，其期望值 \(\mathbb{E}[X] = 3\) 和方差 \(\mathrm{Var}[X] = 4\)。使用 *切比雪夫不等式* 来界定 \(\P[|X - 3| \geq 4]\)。

**E1.2.14** 一个随机变量 \(X\) 的均值 \(\mu = 5\) 和方差 \(\sigma² = 4\)。使用 *切比雪夫不等式* 来找到 \(X\) 距离其均值超过 3 个单位概率的上界。

**E1.2.15** 给定随机向量 \(\bX = (X_1, X_2)\)，其中 \(X_1\) 和 \(X_2\) 是相互独立的正态分布随机变量，求 \(\bX\) 的协方差矩阵。

**E1.2.16** 设 \(\bX = \begin{pmatrix} X_1 \\ X_2 \end{pmatrix}\) 是一个随机向量，其 \(\mathbb{E}[X_1] = 1\)，\(\mathbb{E}[X_2] = 2\)，方差 \(\text{Var}[X_1] = 3\)，方差 \(\text{Var}[X_2] = 4\)，和协方差 \(\text{Cov}[X_1, X_2] = 1\)。写出 \(\bX\) 的均值向量 \(\bmu_\bX\) 和协方差矩阵 \(\bSigma_\bX\)。

**E1.2.17** 设 \(\bX = \begin{pmatrix} X_1 \\ X_2 \end{pmatrix}\) 是一个随机向量，其均值 \(\bmu_\bX = \begin{pmatrix} 1 \\ 2 \end{pmatrix}\) 和协方差矩阵 \(\bSigma_\bX = \begin{pmatrix} 3 & 1 \\ 1 & 4 \end{pmatrix}\)。设 \(A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}\)。计算 \(\mathbb{E}[A\bX]\) 和 \(\text{Cov}[A\bX]\)。

**E1.2.18** 设 \(X_1, X_2, X_3\) 是相互独立的随机变量，\(\mathbb{E}[X_1] = 1\)，\(\mathbb{E}[X_2] = 2\)，\(\mathbb{E}[X_3] = 3\)，\(\text{Var}[X_1] = 4\)，\(\text{Var}[X_2] = 5\)，和 \(\text{Var}[X_3] = 6\)。计算 \(\mathbb{E}[X_1 + X_2 + X_3]\) 和 \(\text{Var}[X_1 + X_2 + X_3]\)。

**E1.2.19** 设 \(A = \begin{pmatrix} 2 & -1 \\ -1 & 2 \end{pmatrix}\)。证明 \(A\) 是正定的。

**第 1.3 节**

**E1.3.1** 给定数据点 \(\mathbf{x}_1 = (1, 2)\)，\(\mathbf{x}_2 = (-1, 1)\)，\(\mathbf{x}_3 = (0, -1)\)，和 \(\mathbf{x}_4 = (2, 0)\)，以及划分 \(C_1 = \{1, 4\}\)，\(C_2 = \{2, 3\}\)，计算最优簇代表 \(\boldsymbol{\mu}_1^*\) 和 \(\boldsymbol{\mu}_2^*\)。

**E1.3.2** 对于来自 E1.3.1 的数据点和划分，计算 \(k\)-means 目标值 \(\mathcal{G}(C_1, C_2)\)。

**E1.3.3** 给定簇代表 \(\boldsymbol{\mu}_1 = (0, 1)\)，\(\boldsymbol{\mu}_2 = (1, -1)\)，\(\boldsymbol{\mu}_3 = (-2, 0)\)，以及数据点 \(\mathbf{x}_1 = (1, 1)\)，\(\mathbf{x}_2 = (-1, -1)\)，\(\mathbf{x}_3 = (2, -2)\)，\(\mathbf{x}_4 = (-2, 1)\)，和 \(\mathbf{x}_5 = (0, 0)\)，找到最优划分 \(C_1, C_2, C_3\)。

**E1.3.4** 对于来自 E1.3.3 的数据点和簇代表，计算 \(k\)-means 目标值 \(G(C_1, C_2, C_3; \boldsymbol{\mu}_1, \boldsymbol{\mu}_2, \boldsymbol{\mu}_3)\)。

**E1.3.5** 证明对于任意两个簇代表 \(\boldsymbol{\mu}_1, \boldsymbol{\mu}_2 \in \mathbb{R}^d\) 和任意数据点 \(\mathbf{x} \in \mathbb{R}^d\)，不等式 \(\|\mathbf{x} - \boldsymbol{\mu}_1\|² \leq \|\mathbf{x} - \boldsymbol{\mu}_2\|²\) 等价于 \(2(\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)^T \mathbf{x} \leq \|\boldsymbol{\mu}_2\|² - \|\boldsymbol{\mu}_1\|²\)。

**E1.3.6** 给定簇分配矩阵

$$\begin{split} Z = \begin{bmatrix} 1 & 0 & 0\\ 0 & 1 & 0\\ 0 & 0 & 1\\ 1 & 0 & 0\\ 0 & 1 & 0 \end{bmatrix} \end{split}$$

以及簇代表矩阵

$$\begin{split} U = \begin{bmatrix} 1 & 2\\ -1 & 0\\ 0 & -2 \end{bmatrix}, \end{split}$$

计算矩阵乘积 \(ZU\)。

**E1.3.7** 对于任意两个矩阵 \(A, B \in \mathbb{R}^{n \times m}\)，证明 \(\|A + B\|_F² = \|A\|_F² + \|B\|_F² + 2 \sum_{i=1}^n \sum_{j=1}^m A_{ij} B_{ij}\)。

**E1.3.8** 证明对于任意矩阵 \(A \in \mathbb{R}^{n \times m}\) 和任意标量 \(c \in \mathbb{R}\)，它满足 \(\|cA\|_F = |c| \|A\|_F\).

**E1.3.9** 对二次函数 \(q(x) = 3x² - 6x + 5\) 完全平方。\(q\) 的最小值是多少，在哪里取得？

**E1.3.10** 设 \(f_1(x) = x² + 1\) 和 \(f_2(y) = 2y² - 3\)。定义 \(h(x, y) = f_1(x) + f_2(y)\)。找到 \(h(x, y)\) 的全局最小值。

**E1.3.11** 计算矩阵的 Frobenius 范数

$$\begin{split} A = \begin{bmatrix} 2 & -1 \\ 0 & 3 \end{bmatrix}. \end{split}$$

**第 1.4 节**

**E1.4.1** 设 \(X_1, \dots, X_d\) 是在 \([-1/2, 1/2]\) 上均匀分布的独立随机变量。计算 \(\mathbb{E}[X_i]\) 和 \(\text{Var}[X_i]\)。

**E1.4.2** 设 \(\bX = (X_1, \dots, X_d)\) 为一个 \(d\)-维向量，其中每个 \(X_i\) 是在 \([-1/2, 1/2]\) 上均匀分布的独立随机变量。计算 \(\mathbb{E}[\|\bX\|²]\)。

**E1.4.3** 设 \(\bX = (X_1, \dots, X_d)\) 为一个 \(d\)-维向量，其中每个 \(X_i\) 是一个独立的正态分布随机变量。计算 \(\mathbb{E}[\|\bX\|²]\)。

**E1.4.4** 设 \(X_1, X_2, \ldots, X_d\) 是独立随机变量，对于所有 \(i\) 有 \(E[X_i] = \mu\) 和 \(Var[X_i] = \sigma²\)。找到 \(\E[\sum_{i=1}^d X_i]\) 和 \(\mathrm{Var}[\sum_{i=1}^d X_i]\)。

**E1.4.5** 已知随机变量 \(X\) 的期望值 \(\E[X] = 1\) 和方差 \(\mathrm{Var}[X] = 4\)，使用 *切比雪夫不等式* 来界定 \(\P[|X - 1| \geq 3]\)。

**E1.4.6** 设 \(\bX = (X_1, \ldots, X_d)\) 为一个具有独立分量的随机向量，每个分量遵循 \(U[-\frac{1}{2}, \frac{1}{2}]\)。对于 \(d = 1, 2, 3\)，显式地计算 \(\P[\|\bX\| \leq \frac{1}{2}]\)。[提示：使用几何论证。]

**E1.4.7** 已知随机变量 \(X\) 的期望值 \(\E[X] = 0\) 和方差 \(\mathrm{Var}[X] = 1\)，使用 *切比雪夫不等式* 来界定 \(\P[|X| \geq 2\sqrt{d}]\) 对于 \(d = 1, 10, 100\)。

**E1.4.8** 设 \(\bX = (X_1, \ldots, X_d)\) 为一个具有独立分量，每个分量遵循 \(U[-\frac{1}{2}, \frac{1}{2}]\) 的随机向量。使用 *切比雪夫不等式* 为 \(d = 1, 10, 100\) 找到一个上界 \(\P[\|\bX\| \geq \frac{1}{2}]\)。

## 1.5.2\. 问题#

**1.1** (改编自 [VLMS]) 回想一下，对于相同维度的两个向量 \(\mathbf{x}\) 和 \(\mathbf{y}\)，它们的内积是 \(\mathbf{x}^T \mathbf{y}\)。一个向量被称为非负的，如果 *所有* 的分量都是 \(\geq 0\)。

a) 证明两个非负向量的内积必然 \(\geq 0\)。

b) 假设两个非负向量的内积是 *零*。你能说些什么关于它们的稀疏模式，即它们的哪些分量是零或非零。\(\lhd\)

**1.2** (改编自 [VLMS]) 布尔 \(n\)-向量是指所有分量要么是 \(0\) 要么是 \(1\) 的向量。这样的向量用于编码 \(n\) 个条件中的每一个是否成立，其中 \(a_i = 1\) 表示条件 \(i\) 成立。

a) 同样信息的另一种常见编码使用 \(-1\) 和 \(1\) 两个值作为元素。例如，布尔向量 \((0,1,1,0)\) 使用这种替代编码将写作 \((-1, 1, 1, -1)\)。假设 \(\mathbf{x}\) 是一个布尔向量，而 \(\mathbf{y}\) 是一个使用 \(-1\) 和 \(1\) 的值编码相同信息的向量。使用向量符号表示 \(\mathbf{y}\) 关于 \(\mathbf{x}\) 的表达式。同样，使用向量符号表示 \(\mathbf{x}\) 关于 \(\mathbf{y}\) 的表达式。

b) 假设 \(\mathbf{x}\) 和 \(\mathbf{y}\) 是布尔 \(n\)-向量。给出它们平方欧几里得距离 \(\|\mathbf{x} - \mathbf{y}\|_2²\) 的简单文字描述，并从数学上证明它。\(\lhd\)

**1.3** (改编自 [VLMS]) 如果 \(\mathbf{a}\) 是一个向量，那么 \(\mathbf{a}_{r:s}\) 是一个大小为 \(s - r + 1\) 的向量，其元素为 \(a_r,\ldots,a_s\)，即 \(\mathbf{a}_{r:s} = (a_r,\ldots,a_s)\)。向量 \(\mathbf{a}_{r:s}\) 被称为切片。作为一个更具体的例子，如果 \(\mathbf{z}\) 是 \(4\)-向量 \((1, -1, 2, 0)\)，那么切片 \(\mathbf{z}_{2:3} = (-1, 2)\)。假设 \(T\)-向量 \(\mathbf{x}\) 表示一个时间序列或信号。这个量

$$ \mathcal{D}(\mathbf{x})=(x_1 - x_2)² + (x_2 - x_3)² + \cdots+(x_{T-1} - x_T)², $$

信号相邻值的差的和被称为信号的狄利克雷能量。狄利克雷能量是时间序列粗糙度或波动性的度量。

a) 使用切片用向量符号表达 \(\mathcal{D}(\mathbf{x})\)。[提示：注意 \(\mathcal{D}(\mathbf{x})\) 与两个向量之间的平方欧几里得距离的相似性。]

b) \(\mathcal{D}(\mathbf{x})\) 可以有多小？哪些信号 \(\mathbf{x}\) 有这个狄利克雷能量的最小值？

c) 找到一个绝对值不超过一的信号 \(\mathbf{x}\)，使其 \(\mathcal{D}(\mathbf{x})\) 的值尽可能大。给出达到的狄利克雷能量值。\(\lhd\)

**1.4** (改编自 [VLMS]) 长度为 \(n\) 的向量可以表示字典中每个单词在文档中出现的次数。例如，\((25,2,0)\) 表示第一个字典单词出现了 \(25\) 次，第二个单词两次，第三个单词一次也没有。假设 \(n\)-向量 \(\mathbf{w}\) 是与文档和 \(n\) 个单词的字典相关的词频向量。为了简单起见，我们将假设文档中的所有单词都在字典中。

a) \(\mathbf{1}^T \mathbf{w}\) 是什么？这里 \(\mathbf{1}\) 是适当大小的全一向量。

b) \(w_{282} = 0\) 的含义是什么？

c) 设 \(\mathbf{h}\) 是给出词频直方图的 \(n\)-向量，即 \(h_i\) 是文档中单词 \(i\) 的词频比例。使用向量符号将 \(\mathbf{h}\) 用 \(\mathbf{w}\) 表达。(你可以假设文档中至少有一个单词。) \(\lhd\)

**1.5** (改编自 [VLMS]) 假设 \(\mathbf{a}\) 和 \(\mathbf{b}\) 是相同大小的向量。三角不等式表明 \(\|\mathbf{a} + \mathbf{b}\|_2 \leq \|\mathbf{a}\|_2 + \|\mathbf{b}\|_2\)。证明我们也有 \(\|\mathbf{a} + \mathbf{b}\|_2 \geq \|\mathbf{a}\|_2 - \|\mathbf{b}\|_2\)。 \(\lhd\)

**1.6** (改编自 [VLMS]) 验证以下恒等式对于任何相同大小的两个向量 \(\mathbf{a}\) 和 \(\mathbf{b}\) 都成立。

a) \((\mathbf{a}+\mathbf{b})^T(\mathbf{a} - \mathbf{b})=\|\mathbf{a}\|_2² - \|\mathbf{b}\|_2²\).

b) \(\|\mathbf{a} + \mathbf{b}\|_2² + \|\mathbf{a} - \mathbf{b}\|_2² = 2(\|\mathbf{a}\|_2² + \|\mathbf{b}\|_2²)\)。这被称为*平行四边形法则*。 \(\lhd\)

**1.7** (改编自 [VLMS]) \(n\) 维向量 \(\mathbf{x}\) 的均方根 (RMS) 值定义为

$$ \mathrm{rms}(\mathbf{x}) = \sqrt{\frac{x_1² + \cdots + x_n²}{n}} = \frac{\|\mathbf{x}\|_2}{\sqrt{n}}. $$

a) 证明一个向量的至少一个元素的绝对值至少与向量的均方根值一样大。

b) 对于一个 \(n\) 维向量 \(\mathbf{x}\)，令 \(\mathrm{avg}(\mathbf{x}) = \mathbf{1}^T \mathbf{x}/n\) 和

$$ \mathrm{std}(\mathbf{x}) = \frac{\|\mathbf{x} - \mathrm{avg}(\mathbf{x}) \mathbf{1}\|_2}{\sqrt{n}}. $$

建立恒等式

$$ \mathrm{rms}(\mathbf{x})² = \mathrm{avg}(\mathbf{x})² + \mathrm{std}(\mathbf{x})². $$

c) 使用(b)证明 \(|\mathrm{avg}(\mathbf{x})| \leq \mathrm{rms}(\mathbf{x})\).

d) 使用(b)证明 \(\mathrm{std}(\mathbf{x}) \leq \mathrm{rms}(\mathbf{x})\)。 \(\lhd\)

**1.8** (改编自 [VLMS]) 假设向量 \(\mathbf{x}_1,\ldots,\mathbf{x}_N\) 使用 \(k\)-means 进行聚类，组代表为 \(\mathbf{z}_1,\ldots,\mathbf{z}_k\)。

a) 假设原始向量 \(\mathbf{x}_i\) 是非负的，即它们的元素是非负的。解释为什么代表向量 \(\mathbf{z}_j\) 也是非负的。

b) 假设原始向量 \(\mathbf{x}_i\) 表示比例，即它们的元素是非负的，并且总和为 1。解释为什么代表向量 \(\mathbf{z}_j\) 也表示比例，即它们的元素是非负的，并且总和为 1。

c) 假设原始向量 \(\mathbf{x}_i\) 是布尔值，即它们的元素是 \(0\) 或 \(1\)。解释 \((\mathbf{z}_j)_i\)，即第 \(j\) 个组代表向量的第 \(i\) 个元素的含义。 \(\lhd\)

**1.9** (改编自 [VLMS]) 将向量集合聚类成 \(k = 2\) 个组称为 \(2\)-way 分区，因为我们正在将向量分成 \(2\) 个组，索引集为 \(G_1\) 和 \(G_2\)。假设我们在 \(n\) 个向量 \(\mathbf{x}_1,\ldots,\mathbf{x}_N\) 上运行 \(k = 2\) 的 \(k\)-means。证明存在一个非零向量 \(\mathbf{w}\) 和一个标量 \(v\)，使得

$$ \mathbf{w}^T \mathbf{x}_i + v \geq 0, \forall i \in G_1, \quad \mathbf{w}^T \mathbf{x}_i + v \leq 0, \forall i \in G_2. $$

换句话说，仿射函数 \(f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + v\) 在第一组上大于或等于零，在第二组上小于或等于零。这被称为两组的线性分离。[*提示:* 考虑函数 \(\|\mathbf{x} - \mathbf{z}_1\|_2² - \|\mathbf{x} - \mathbf{z}_2\|_2²\)，其中 \(\mathbf{z}_1\) 和 \(\mathbf{z}_2\) 是组代表。] \(\lhd\)

**1.10** (改编自 [ASV]) 设 \(0 < p \leq 1\)。一个随机变量 \(X\) 如果其可能值为 \(\{1, 2, 3, \ldots\}\)，并且满足 \(\mathbb{P}(X = k) = (1 - p)^{k - 1}p\) 对于正整数 \(k\)，则 \(X\) 具有几何分布，成功参数为 \(p\)。其期望值为 \(1/p\)，方差为 \((1 - p)/p²\)。

设 \(X\) 是参数为 \(p = 1/6\) 的几何随机变量。

a) 使用 *马尔可夫不等式* 来找到 \(\mathbb{P}(X \geq 16)\) 的上界。

b) 使用 *切比雪夫不等式* 来找到 \(\mathbb{P}(X \geq 16)\) 的上界。

c) 使用几何级数的和来显式计算 \(\mathbb{P}(X \geq 16)\) 的概率，并与你推导的上界进行比较。 \(\lhd\)

**1.11** (改编自 [ASV]) 设 \(0 < \lambda < +\infty\)。一个随机变量 \(X\) 如果 \(X\) 的密度函数为 \(f(x) = \lambda e^{-\lambda x}\)，对于 \(x \geq 0\) 和 \(0\) 否则，则 \(X\) 具有参数为 \(\lambda\) 的指数分布。其期望值为 \(1/\lambda\)，方差为 \(1/\lambda²\)。

设 \(X\) 是参数为 \(\lambda = 1/2\) 的指数随机变量。

a) 使用 *马尔可夫不等式* 来给出 \(X\) 大于 \(15\) 的概率上界。

b) 使用 *切比雪夫不等式* 来找到 \(\mathbb{P}(X > 6)\) 的上界。

c) 使用积分来显式计算 \(\mathbb{P}(X > 6)\) 的概率，并与你推导的上界进行比较。 \(\lhd\)

**1.12** (改编自 [ASV]) 假设 \(X\) 是一个非负随机变量，且 \(\mathbb{E}[X] = 10\)。

a) 使用 *马尔可夫不等式* 来给出 \(X\) 大于 \(15\) 的概率上界。

b) 假设我们还知道 \(\mathrm{Var}(X) = 3\)。使用 *切比雪夫不等式* 来给出比部分 (a) 更好的 \(\mathbb{P}(X > 15)\) 的上界。

c) 假设 \(Y_1,Y_2,\ldots,Y_{300}\) 是与 \(X\) 具有相同分布的独立同分布随机变量，因此，特别是 \(\mathbb{E}(Y_i) = 10\) 和 \(\mathrm{Var}(Y_i) = 3\)。使用 *切比雪夫不等式* 来给出 \(\sum_{i=1}^{300} Y_i\) 大于 \(3060\) 的概率上界。 \(\lhd\)

**1.13** (改编自 [ASV]) 假设我们有一组独立同分布的随机变量 \(X_1, X_2, X_3,\ldots\)，它们的期望值 \(\mathbb{E}[X_1] = \mu\) 和方差 \(\mathrm{Var}(X_1) = \sigma²\) 是有限的。令 \(S_n = X_1 +\cdots+X_n\)。证明对于任何固定的 \(\varepsilon > 0\) 和 \(1/2 < \alpha < 1\)，我们有

$$ \lim_{n \to +\infty} \mathbb{P}\left[\left|\frac{S_n - n\mu}{n^\alpha}\right| < \varepsilon\right] = 1. $$

\(\lhd\)

**1.14** (改编自 [ASV]) 通过模仿 *大数定律* 的证明，证明以下变体。假设我们有一些随机变量 \(X_1, X_2, \ldots\)，每个都有有限的均值 \(\mathbb{E}[X_i] = \mu\) 和方差 \(\mathrm{Var}(X_i) = \sigma²\)。进一步假设当 \(|i - j| \geq 2\) 时，\(\mathrm{Cov}(X_i,X_j) = 0\)，并且存在一个常数 \(c > 0\)，使得对于所有 \(i\)，\(|\mathrm{Cov}(X_i, X_{i+1})| < c\)。设 \(S_n =X_1 +\cdots+X_n\)。那么对于任何固定的 \(\varepsilon>0\)，我们有

$$ \lim_{n\to +\infty} \mathbb{P}\left[\left|\frac{S_n}{n} - \mu\right|\right]<\varepsilon=1. $$

\(\lhd\)

**1.15** (改编自 [ASV]) 通过模仿 *切比雪夫不等式* 的证明，证明以下变体。设 \(X\) 是一个具有有限均值 \(\mu\) 的随机变量，并且对于某个 \(s > 0\)，\(\mathbb{E}[\exp(s(X - \mu))] < +\infty\)。那么对于 \(c > 0\)，我们有

$$ \mathbb{P}(X \geq \mu + c) \leq \frac{\mathbb{E}[\exp(s(X - \mu)]}{e^{s c}}. $$

仔细证明你的答案。\(\lhd\)

**1.16** (改编自 [ASV]) 回想一下，实值随机变量 \(Z\) 的累积分布函数 (CDF) 是函数 \(F_Z(z) = \mathbb{P}[Z \leq z]\)，对于所有 \(z \in \mathbb{R}\)。设 \(X_1, X_2, \ldots , X_n\) 是具有相同累积分布函数 \(F\) 的独立随机变量。用最小值和最大值表示为

$$ Z = \min(X_1,X_2,\ldots,X_n) \quad\text{和}\quad W = \max(X_1,X_2,\ldots,X_n). $$

找到 \(Z\) 和 \(W\) 的累积分布函数 \(F_Z\) 和 \(F_W\)。\(\lhd\)

**1.17** (改编自 [ASV]) 假设 \(X_1, X_2, \ldots\) 是具有参数 \(\lambda = 1\) 的指数分布的独立同分布随机变量（参见上面的练习 1.11），并设 \(M_n = \max(X_1,\ldots,X_n)\)。证明对于任何 \(x \in \mathbb{R}\)，我们有

$$ \lim_{n \to +\infty} \mathbb{P}(M_n - \ln n \leq x) = \exp(-e^{-x}). $$

右侧是 Gumbel 分布的累积分布函数，它是极值分布的一个例子。[提示：使用练习 1.16 计算出 \(\mathbb{P}(M_n \leq \ln n + x)\) 的显式值，然后求 \(n \to +\infty\) 时的极限。] \(\lhd\)

**1.18** (改编自 [ASV]) 设 \(A\) 和 \(B\) 是两个不相交的事件。它们在什么条件下是独立的？\(\lhd\)

**1.19** (改编自 [ASV]) 我们从集合 \(\{10, 11, 12, \ldots , 99\}\) 中均匀随机选择一个数字。

a) 设 \(X\) 为所选数字的第一个数字，\(Y\) 为第二个数字。证明 \(X\) 和 \(Y\) 是独立的随机变量。

b) 设 \(X\) 为所选数字的第一个数字，\(Z\) 为两个数字的和。证明 \(X\) 和 \(Z\) 不是独立的。

\(\lhd\)

**1.20** (改编自 [ASV]) 假设 \(X\) 和 \(Y\) 具有联合概率密度函数 \(f(x,y) = 2e^{-(x+2y)}\) 对于 \(x > 0, y > 0\) 和其他地方为 \(0\)。证明 \(X\) 和 \(Y\) 是独立的随机变量，并给出它们的边缘分布。[*提示:* 回忆一下问题 1.11 中的指数分布。] \(\lhd\)

**1.21** (改编自 [ASV]) 假设 \(X,Y\) 具有联合概率密度函数

$$ f(x,y) = c\exp\left[-\frac{x²}{2} - \frac{(x - y)²}{2}\right], $$

对于 \(x,y \in \mathbb{R}²\)，对于某个常数 \(c > 0\)。

a) 求常数 \(c\) 的值。[*提示:* 积分的顺序很重要。你可以不进行复杂的积分来完成这个任务。具体来说，回忆一下对于任何 \(\mu \in \mathbb{R}\) 和 \(\sigma > 0\)，都有

$$ \int_{-\infty}^{+\infty} \frac{1}{\sqrt{2 \pi \sigma²}} \exp\left(-\frac{(z - \mu)²}{2\sigma²}\right) \mathrm{d} z= 1. $$

]

b) 求 \(X\) 和 \(Y\) 的边缘密度函数。[*提示:* 你可以不进行复杂的积分来完成这个任务。完成平方。]

c) 判断 \(X\) 和 \(Y\) 是否独立。证明你的答案。 \(\lhd\)

**1.22** (改编自 [ASV]) 设 \(p(x, y)\) 是 \((X, Y)\) 的联合概率质量函数。假设存在两个函数 \(a\) 和 \(b\)，使得对于 \(X\) 的所有可能值 \(x\) 和 \(Y\) 的所有可能值 \(y\)，都有 \(p(x,y) = a(x)b(y)\)。证明 \(X\) 和 \(Y\) 是独立的。不要假设 \(a\) 和 \(b\) 是概率质量函数。 \(\lhd\)

**1.23** (改编自 [BHK]) a) 设 \(X\) 和 \(Y\) 是在 \([-1/2 , 1/2]\) 上具有均匀分布的独立随机变量。计算 \(\mathbb{E}[X]\)，\(\mathbb{E}[X²]\)，\(\mathrm{Var}[X²]\)，\(\mathbb{E}[X - Y]\)，\(\mathbb{E}[XY]\)，和 \(\mathbb{E}[(X - Y)²]\)。

b) 两个在单位 d 维立方体 \(\mathcal{C} = [-1/2,1/2]^d\) 内随机生成的点之间的期望平方欧几里得距离是多少？\(\lhd\)

**1.24** (改编自 [BHK]) a) 对于任意 \(a > 0\)，给出一个非负随机变量 \(X\) 的概率分布，使得

$$ \mathbb{P}[X \geq a] = \frac{\mathbb{E}[X]}{a}. $$

b) 证明对于任何 \(c > 0\)，存在一个分布使得 *切比雪夫不等式* 是紧的，即 \( \mathbb{P}[|X - \mathbb{E}[X]| \geq c] = \frac{\mathrm{Var}[X]}{c²}. \)

[*提示:* 选择一个关于 \(0\) 对称的分布。] \(\lhd\)

**1.25** (改编自 [BHK]) 设 \(X_1, X_2, \ldots , X_n\) 是具有均值 \(\mu\) 和方差 \(\sigma²\) 的独立同分布随机变量。设

$$ \overline{X}_n = \frac{1}{n} \sum_{i=1}^n X_i, $$

是样本均值。假设使用样本均值估计方差如下

$$ S²_n = \frac{1}{n} \sum_{i=1}^n \left(X_i - \overline{X}_n\right)². $$

计算 \(\mathbb{E}(S²_n)\)。[*提示:* 将 \(X_i - \overline{X}_n\) 替换为 \((X_i - \mu) - (\overline{X}_n - \mu)\)。] \(\lhd\)

**1.26** 设 \(f\) 和 \(g\) 在 \(x\) 处有导数，且 \(\alpha\) 和 \(\beta\) 是常数。从导数的定义出发证明

$$ [\alpha f(x) + \beta g(x)]' = \alpha f'(x) + \beta g'(x). $$

\(\lhd\)

**1.27** 设 \(Z_1 \sim \mathrm{N}(\mu_1, \sigma_1²)\) 和 \(Z_2 \sim \mathrm{N}(\mu_2, \sigma_2²)\) 是独立的正态变量，均值分别为 \(\mu_1\)，\(\mu_2\)，方差分别为 \(\sigma_1²\) 和 \(\sigma_2²\)。回忆 \(Z_1 + Z_2\) 仍然是正态分布。

a) \(Z_1 + Z_2\) 的均值和方差是什么？

b) 设 \(\mathbf{X}_1, \mathbf{X}_2, \mathbf{Y}_1\) 是独立的球形 \(d\) 维高斯分布，均值为 \(-w \mathbf{e}_1\)，方差为 \(1\)。设 \(\mathbf{Y}_2\) 是一个独立的球形 \(d\) 维高斯分布，均值为 \(w \mathbf{e}_1\)，方差为 \(1\)。显式计算 \(\mathbb{E}[\|\mathbf{X}_1 - \mathbf{X}_2\|²]\) 和 \(\mathbb{E}[\|\mathbf{Y}_1 - \mathbf{Y}_2\|²]\)。 \(\lhd\)

**1.28** 假设我们给出了 \(\mathbb{R}^d\) 中的 \(n\) 个向量 \(\mathbf{x}_1,\ldots,\mathbf{x}_n\) 和一个划分 \(C_1, \ldots, C_k \subseteq [n]\)。设 \(n_i = |C_i|\) 为簇 \(C_i\) 的大小，并设

$$ \boldsymbol{\mu}_i^* = \frac{1}{n_i} \sum_{j\in C_i} \mathbf{x}_j $$

为 \(C_i\) 的质心，对于 \(i=1,\ldots,k\)。

a) 证明

$$ \sum_{j \in C_i} \|\mathbf{x}_j - \boldsymbol{\mu}_i^*\|² = \left(\sum_{j \in C_i} \|\mathbf{x}_j\|²\right) - n_i\|\boldsymbol{\mu}_i^*\|². $$

b) 证明

$$\begin{split} \|\boldsymbol{\mu}_i^*\|² = \frac{1}{n_i²}\left(\sum_{j \in C_i} \|\mathbf{x}_j\|² + \sum_{\substack{j, \ell \in C_i\\j \neq \ell}} \mathbf{x}_j^T\mathbf{x}_\ell\right). \end{split}$$

c) 证明

$$\begin{split} \sum_{\substack{j, \ell \in C_i\\j \neq \ell}} \|\mathbf{x}_j - \mathbf{x}_\ell\|² = 2(n_i-1)\sum_{j \in C_i} \|\mathbf{x}_j\|² - 2 \sum_{\substack{j, \ell \in C_i\\j \neq \ell}} \mathbf{x}_j^T\mathbf{x}_\ell. \end{split}$$

d) 结合 a)，b），c）来证明在所有 \([n]\) 的划分 \(C_1, \ldots, C_k\) 上最小化 \(k\)-means 目标函数 \(\mathcal{G}(C_1,\ldots,C_k)\) 等价于最小化

$$ \sum_{i=1}^k \frac{1}{2 |C_i|} \sum_{j,\ell \in C_i} \|\mathbf{x}_j - \mathbf{x}_\ell\|². $$

\(\lhd\)

**1.29** 假设 \(A \in \mathbb{R}^{n \times m}\) 的行由 \(\mathbf{r}_1,\ldots,\mathbf{r}_n \in \mathbb{R}^m\) 的转置给出，且 \(A\) 的列由 \(\mathbf{c}_1,\ldots,\mathbf{c}_m \in \mathbb{R}^n\) 给出。给出 \(A^T A\) 和 \(A A^T\) 的元素表达式。 \(\lhd\)

**1.30** 使用 \(\bSigma_\bX\) 的矩阵形式给出**协方差正定性**的证明。 \(\lhd\)

**1.31** 使用**柯西-施瓦茨不等式**来证明相关系数位于 \([-1,1]\) 范围内。 \(\lhd\)

**1.32** 设 \(f(x,y) = g(x) + h(y)\)，其中 \(x, y \in \mathbb{R}\) 且 \(g,h\) 是实值连续可微函数。用 \(g\) 和 \(h\) 的导数来计算 \(f\) 的梯度。 \(\lhd\)

**1.33** 设 \(A = [a]\) 是一个 \(1\times 1\) 的正定矩阵。证明 \(a > 0\)。 \(\lhd\)

**1.34** 证明正定矩阵的对角元素必然是正的。 \(\lhd\)

**1.35** 设 \(A, B \in \mathbb{R}^{n \times m}\) 且 \(c \in \mathbb{R}\)。证明

a) \((A + B)^T = A^T + B^T\)

b) \((c A)^T = c A^T\)

\(\lhd\)

## 1.5.1\. 热身练习表#

*(在克劳德、双子星和 ChatGPT 的帮助下)*

**第 1.2 节**

**E1.2.1** 计算向量 \(\mathbf{x} = (6, 8)\) 的欧几里得范数。

**E1.2.2** 计算向量 \(\mathbf{u} = (1, 2, 3)\) 和 \(\mathbf{v} = (4, 5, 6)\) 的内积 \(\langle \mathbf{u}, \mathbf{v} \rangle\)。

**E1.2.3** 找到矩阵 \(A = \begin{pmatrix}1 & 2 & 3 \\ 4 & 5 & 6\end{pmatrix}\) 的转置。

**E1.2.4** 计算矩阵 \(A = \begin{pmatrix}1 & 2 \\ 3 & 4\end{pmatrix}\) 的 Frobenius 范数。

**E1.2.5** 验证矩阵 \(A = \begin{pmatrix}2 & 0 \\ 0 & 3\end{pmatrix}\) 是否是对称的。

**E1.2.6** 设 \(A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}\) 和 \(B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}\)。计算 \(AB\) 和 \(BA\)。

**E1.2.7** 设 \(f(x, y) = x² + xy + y²\)。计算偏导数 \(\frac{\partial f}{\partial x}\) 和 \(\frac{\partial f}{\partial y}\)。

**E1.2.8** 给定函数 \(g(x, y) = x² y + xy³\)，计算其偏导数 \(\frac{\partial g}{\partial x}\) 和 \(\frac{\partial g}{\partial y}\)。

**E1.2.9** 设 \(f(x) = x³ - 3x² + 2x\)。使用 *泰勒定理* 在 \(a = 1\) 附近找到 \(f\) 的线性近似，并带有二阶误差项。

**E1.2.10** 计算离散随机变量 \(X\) 的期望 \(\E[X]\)，其中 \(\P(X = 1) = 0.2\)，\(\P(X = 2) = 0.5\)，和 \(\P(X = 3) = 0.3\)。

**E1.2.11:** 计算离散随机变量 \(X\) 的方差 \(\mathrm{Var}[X]\)，其中 \(\P(X = 1) = 0.4\)，\(\P(X = 2) = 0.6\)，和 \(\E[X] = 1.6\)。

**E1.2.12** 设 \(X\) 和 \(Y\) 是随机变量，满足 \(\mathbb{E}[X] = 2\)，\(\mathbb{E}[Y] = 3\)，\(\mathrm{Var}[X] = 4\)，和 \(\mathrm{Var}[Y] = 9\)。假设 \(X\) 和 \(Y\) 是独立的，计算 \(\mathbb{E}[2X + Y - 1]\) 和 \(\mathrm{Var}[2X + Y - 1]\)。

**E1.2.13** 设 \(X\) 是一个随机变量，满足 \(\mathbb{E}[X] = 3\) 和 \(\mathrm{Var}[X] = 4\)。使用 *切比雪夫不等式* 来界定 \(\P[|X - 3| \geq 4]\)。

**E1.2.14** 随机变量 \(X\) 的均值 \(\mu = 5\) 和方差 \(\sigma² = 4\)。使用 *切比雪夫不等式* 来找到 \(X\) 距离其均值超过 3 个单位概率的上界。

**E1.2.15** 给定随机向量 \(\bX = (X_1, X_2)\)，其中 \(X_1\) 和 \(X_2\) 是独立的正态标准随机变量，\(\bX\) 的协方差矩阵是什么？

**E1.2.16** 设 \(\bX = \begin{pmatrix} X_1 \\ X_2 \end{pmatrix}\) 是一个随机向量，其 \(\mathbb{E}[X_1] = 1\)，\(\mathbb{E}[X_2] = 2\)，\(\text{Var}[X_1] = 3\)，\(\text{Var}[X_2] = 4\)，和 \(\text{Cov}[X_1, X_2] = 1\)。写出 \(\bX\) 的均值向量 \(\bmu_\bX\) 和协方差矩阵 \(\bSigma_\bX\)。

**E1.2.17** 设 \(\bX = \begin{pmatrix} X_1 \\ X_2 \end{pmatrix}\) 是一个随机向量，其均值 \(\bmu_\bX = \begin{pmatrix} 1 \\ 2 \end{pmatrix}\) 和协方差矩阵 \(\bSigma_\bX = \begin{pmatrix} 3 & 1 \\ 1 & 4 \end{pmatrix}\)。设 \(A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}\)。计算 \(\mathbb{E}[A\bX]\) 和 \(\text{Cov}[A\bX]\)。

**E1.2.18** 设 \(X_1, X_2, X_3\) 是独立的随机变量，其 \(\mathbb{E}[X_1] = 1\)，\(\mathbb{E}[X_2] = 2\)，\(\mathbb{E}[X_3] = 3\)，\(\text{Var}[X_1] = 4\)，\(\text{Var}[X_2] = 5\)，和 \(\text{Var}[X_3] = 6\)。计算 \(\mathbb{E}[X_1 + X_2 + X_3]\) 和 \(\text{Var}[X_1 + X_2 + X_3]\)。

**E1.2.19** 设 \(A = \begin{pmatrix} 2 & -1 \\ -1 & 2 \end{pmatrix}\)。证明 \(A\) 是正定的。

**第 1.3 节**

**E1.3.1** 给定数据点 \(\mathbf{x}_1 = (1, 2)\)，\(\mathbf{x}_2 = (-1, 1)\)，\(\mathbf{x}_3 = (0, -1)\)，和 \(\mathbf{x}_4 = (2, 0)\)，以及分区 \(C_1 = \{1, 4\}\)，\(C_2 = \{2, 3\}\)，计算最优聚类代表 \(\boldsymbol{\mu}_1^*\) 和 \(\boldsymbol{\mu}_2^*\)。

**E1.3.2** 对于来自 E1.3.1 的数据点和分区，计算 \(k\)-means 目标值 \(\mathcal{G}(C_1, C_2)\)。

**E1.3.3** 给定聚类代表 \(\boldsymbol{\mu}_1 = (0, 1)\)，\(\boldsymbol{\mu}_2 = (1, -1)\)，和 \(\boldsymbol{\mu}_3 = (-2, 0)\)，以及数据点 \(\mathbf{x}_1 = (1, 1)\)，\(\mathbf{x}_2 = (-1, -1)\)，\(\mathbf{x}_3 = (2, -2)\)，\(\mathbf{x}_4 = (-2, 1)\)，和 \(\mathbf{x}_5 = (0, 0)\)，找到最优分区 \(C_1, C_2, C_3\)。

**E1.3.4** 对于来自 E1.3.3 的数据点和聚类代表，计算 \(k\)-means 目标值 \(G(C_1, C_2, C_3; \boldsymbol{\mu}_1, \boldsymbol{\mu}_2, \boldsymbol{\mu}_3)\)。

**E1.3.5** 证明对于任意两个聚类代表 \(\boldsymbol{\mu}_1, \boldsymbol{\mu}_2 \in \mathbb{R}^d\) 和任意数据点 \(\mathbf{x} \in \mathbb{R}^d\)，不等式 \(\|\mathbf{x} - \boldsymbol{\mu}_1\|² \leq \|\mathbf{x} - \boldsymbol{\mu}_2\|²\) 等价于 \(2(\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)^T \mathbf{x} \leq \|\boldsymbol{\mu}_2\|² - \|\boldsymbol{\mu}_1\|²\)。

**E1.3.6** 给定聚类分配矩阵

$$\begin{split} Z = \begin{bmatrix} 1 & 0 & 0\\ 0 & 1 & 0\\ 0 & 0 & 1\\ 1 & 0 & 0\\ 0 & 1 & 0 \end{bmatrix} \end{split}$$

以及聚类代表矩阵

$$\begin{split} U = \begin{bmatrix} 1 & 2\\ -1 & 0\\ 0 & -2 \end{bmatrix}, \end{split}$$

计算矩阵乘积 \(ZU\)。

**E1.3.7** 对于任意两个矩阵 \(A, B \in \mathbb{R}^{n \times m}\)，证明 \(\|A + B\|_F² = \|A\|_F² + \|B\|_F² + 2 \sum_{i=1}^n \sum_{j=1}^m A_{ij} B_{ij}\)。

**E1.3.8** 证明对于任何 \(A \in \mathbb{R}^{n \times m}\) 矩阵和任何标量 \(c \in \mathbb{R}\)，都有 \(\|cA\|_F = |c| \|A\|_F\)。

**E1.3.9** 完成二次函数 \(q(x) = 3x² - 6x + 5\) 的平方。\(q\) 的最小值是多少，在哪里取得？

**E1.3.10** 设 \(f_1(x) = x² + 1\) 和 \(f_2(y) = 2y² - 3\)。定义 \(h(x, y) = f_1(x) + f_2(y)\)。找到 \(h(x, y)\) 的全局最小值。

**E1.3.11** 计算矩阵的弗罗贝尼乌斯范数

$$\begin{split} A = \begin{bmatrix} 2 & -1 \\ 0 & 3 \end{bmatrix}. \end{split}$$

**第 1.4 节**

**E1.4.1** 设 \(X_1, \dots, X_d\) 是在 \([-1/2, 1/2]\) 上均匀分布的独立随机变量。计算 \(\mathbb{E}[X_i]\) 和 \(\text{Var}[X_i]\)。

**E1.4.2** 设 \(\bX = (X_1, \dots, X_d)\) 是一个 \(d\)-维向量，其中每个 \(X_i\) 是在 \([-1/2, 1/2]\) 上均匀分布的独立随机变量。计算 \(\mathbb{E}[\|\bX\|²]\)。

**E1.4.3** 设 \(\bX = (X_1, \dots, X_d)\) 是一个 \(d\)-维向量，其中每个 \(X_i\) 是独立的正态分布随机变量。计算 \(\mathbb{E}[\|\bX\|²]\)。

**E1.4.4** 设 \(X_1, X_2, \ldots, X_d\) 是具有 \(E[X_i] = \mu\) 和 \(Var[X_i] = \sigma²\) 的独立随机变量，对所有 \(i\) 都成立。求 \(\E[\sum_{i=1}^d X_i]\) 和 \(\mathrm{Var}[\sum_{i=1}^d X_i]\)。

**E1.4.5** 给定一个随机变量 \(X\)，其 \(\E[X] = 1\) 和 \(\mathrm{Var}[X] = 4\)，使用 *切比雪夫不等式* 来界定 \(\P[|X - 1| \geq 3]\)。

**E1.4.6** 设 \(\bX = (X_1, \ldots, X_d)\) 是一个具有独立分量的随机向量，每个分量遵循 \(U[-\frac{1}{2}, \frac{1}{2}]\)。对于 \(d = 1, 2, 3\)，显式地计算 \(\P[\|\bX\| \leq \frac{1}{2}]\)。[提示：使用几何论证。]

**E1.4.7** 给定一个随机变量 \(X\)，其 \(\E[X] = 0\) 和 \(\mathrm{Var}[X] = 1\)，使用 *切比雪夫不等式* 来界定 \(\P[|X| \geq 2\sqrt{d}]\) 对于 \(d = 1, 10, 100\)。

**E1.4.8** 设 \(\bX = (X_1, \ldots, X_d)\) 是一个具有独立分量的随机向量，每个分量遵循 \(U[-\frac{1}{2}, \frac{1}{2}]\)。使用 *切比雪夫不等式* 为 \(d = 1, 10, 100\) 找到 \(\P[\|\bX\| \geq \frac{1}{2}]\) 的上界。

## 1.5.2\. 问题#

**1.1** (改编自 [VLMS]) 回想一下，对于相同维度的两个向量 \(\mathbf{x}\) 和 \(\mathbf{y}\)，它们的内积是 \(\mathbf{x}^T \mathbf{y}\)。一个向量被称为非负的，如果 *所有* 的条目都是 \(\geq 0\)。

a) 证明两个非负向量的内积必然 \(\geq 0\)。

b) 假设两个非负向量的内积为零。你能说些什么关于它们的稀疏模式，即它们的哪些条目是零或非零。\(\lhd\)

**1.2** (改编自 [VLMS]) 一个布尔 \(n\)-向量是指所有条目要么是 \(0\) 要么是 \(1\) 的向量。这样的向量用于编码 \(n\) 个条件中的每一个是否成立，其中 \(a_i = 1\) 表示条件 \(i\) 成立。

a) 同样信息的另一种编码使用两个值 \(-1\) 和 \(1\) 来表示条目。例如，布尔向量 \((0,1,1,0)\) 使用这种替代编码将写作 \((-1, 1, 1, -1)\)。假设 \(\mathbf{x}\) 是一个布尔向量，而 \(\mathbf{y}\) 是一个使用 \(-1\) 和 \(1\) 的值来编码相同信息的向量。请用向量符号表示 \(\mathbf{y}\) 关于 \(\mathbf{x}\) 的表达式。同样，也请用向量符号表示 \(\mathbf{x}\) 关于 \(\mathbf{y}\) 的表达式。

b) 假设 \(\mathbf{x}\) 和 \(\mathbf{y}\) 是布尔 \(n\) 维向量。给出它们平方欧几里得距离 \(\|\mathbf{x} - \mathbf{y}\|_2²\) 的简单文字描述，并从数学上证明它。\(\lhd\)

**1.3** (改编自 [VLMS]) 如果 \(\mathbf{a}\) 是一个向量，那么 \(\mathbf{a}_{r:s}\) 是大小为 \(s - r + 1\) 的向量，条目为 \(a_r,\ldots,a_s\)，即 \(\mathbf{a}_{r:s} = (a_r,\ldots,a_s)\)。向量 \(\mathbf{a}_{r:s}\) 被称为切片。作为一个更具体的例子，如果 \(\mathbf{z}\) 是 \(4\) 维向量 \((1, -1, 2, 0)\)，那么切片 \(\mathbf{z}_{2:3} = (-1, 2)\)。假设 \(T\) 维向量 \(\mathbf{x}\) 表示时间序列或信号。量

$$ \mathcal{D}(\mathbf{x})=(x_1 - x_2)² + (x_2 - x_3)² + \cdots+(x_{T-1} - x_T)², $$

信号相邻值的差的和，被称为信号的狄利克雷能量。狄利克雷能量是时间序列粗糙度或波动性的度量。

a) 使用切片将 \(\mathcal{D}(\mathbf{x})\) 表达为向量符号。[提示：注意 \(\mathcal{D}(\mathbf{x})\) 与两个向量之间的平方欧几里得距离之间的相似性。]

b) \(\mathcal{D}(\mathbf{x})\) 可以有多小？具有狄利克雷能量最小值的信号 \(\mathbf{x}\) 是什么？

c) 找到一个信号 \(\mathbf{x}\)，其条目绝对值不超过一，并且 \(\mathcal{D}(\mathbf{x})\) 的值尽可能大。给出所达到的狄利克雷能量的值。\(\lhd\)

**1.4** (改编自 [VLMS]) 长度为 \(n\) 的向量可以表示字典中每个单词在文档中出现的次数。例如，\((25,2,0)\) 表示第一个字典单词出现了 \(25\) 次，第二个单词两次，第三个单词一次也没有。假设 \(n\) 维向量 \(\mathbf{w}\) 是与文档和 \(n\) 个单词的字典相关联的词频向量。为了简单起见，我们将假设文档中的所有单词都在字典中。

a) \(\mathbf{1}^T \mathbf{w}\) 是什么？在这里，\(\mathbf{1}\) 是适当大小的全一向量。

b) \(w_{282} = 0\) 的意义是什么？

c) 设 \(\mathbf{h}\) 是 \(n\) 维向量，它给出了单词计数的直方图，即 \(h_i\) 是文档中单词 \(i\) 的比例。请用向量符号表示 \(\mathbf{h}\) 关于 \(\mathbf{w}\) 的表达式。（你可以假设文档中至少有一个单词。）\(\lhd\)

**1.5** (改编自 [VLMS]) 假设 \(\mathbf{a}\) 和 \(\mathbf{b}\) 是相同大小的向量。三角不等式表明 \(\|\mathbf{a} + \mathbf{b}\|_2 \leq \|\mathbf{a}\|_2 + \|\mathbf{b}\|_2\)。证明我们也有 \(\|\mathbf{a} + \mathbf{b}\|_2 \geq \|\mathbf{a}\|_2 - \|\mathbf{b}\|_2\)。 \(\lhd\)

**1.6** (改编自 [VLMS]) 验证以下恒等式对于任何相同大小的两个向量 \(\mathbf{a}\) 和 \(\mathbf{b}\) 都成立。

a) \((\mathbf{a}+\mathbf{b})^T(\mathbf{a} - \mathbf{b})=\|\mathbf{a}\|_2² - \|\mathbf{b}\|_2²\).

b) \(\|\mathbf{a} + \mathbf{b}\|_2² + \|\mathbf{a} - \mathbf{b}\|_2² = 2(\|\mathbf{a}\|_2² + \|\mathbf{b}\|_2²)\)。这被称为 *平行四边形法则*。 \(\lhd\)

**1.7** (改编自 [VLMS]) \(n\)-向量 \(\mathbf{x}\) 的均方根 (RMS) 值定义为

$$ \mathrm{rms}(\mathbf{x}) = \sqrt{\frac{x_1² + \cdots + x_n²}{n}} = \frac{\|\mathbf{x}\|_2}{\sqrt{n}}. $$

a) 证明向量中至少有一个元素的绝对值不小于该向量的 RMS 值。

b) 对于 \(n\)-向量 \(\mathbf{x}\)，令 \(\mathrm{avg}(\mathbf{x}) = \mathbf{1}^T \mathbf{x}/n\) 和

$$ \mathrm{std}(\mathbf{x}) = \frac{\|\mathbf{x} - \mathrm{avg}(\mathbf{x}) \mathbf{1}\|_2}{\sqrt{n}}. $$

建立以下恒等式

$$ \mathrm{rms}(\mathbf{x})² = \mathrm{avg}(\mathbf{x})² + \mathrm{std}(\mathbf{x})². $$

c) 使用 (b) 证明 \(|\mathrm{avg}(\mathbf{x})| \leq \mathrm{rms}(\mathbf{x})\)。

d) 使用 (b) 证明 \(\mathrm{std}(\mathbf{x}) \leq \mathrm{rms}(\mathbf{x})\)。 \(\lhd\)

**1.8** (改编自 [VLMS]) 假设向量 \(\mathbf{x}_1,\ldots,\mathbf{x}_N\) 使用 \(k\)-means 算法进行聚类，组代表向量为 \(\mathbf{z}_1,\ldots,\mathbf{z}_k\)。

a) 假设原始向量 \(\mathbf{x}_i\) 是非负的，即它们的元素是非负的。解释为什么代表向量 \(\mathbf{z}_j\) 也是非负的。

b) 假设原始向量 \(\mathbf{x}_i\) 表示比例，即它们的元素是非负的，且总和为 1。解释为什么代表向量 \(\mathbf{z}_j\) 也表示比例，即它们的元素是非负的，且总和为 1。

c) 假设原始向量 \(\mathbf{x}_i\) 是布尔值，即它们的元素是 \(0\) 或 \(1\)。给出 \((\mathbf{z}_j)_i\) 的解释，即第 \(j\) 个组代表向量的第 \(i\) 个元素。 \(\lhd\)

**1.9** (改编自 [VLMS]) 将一组向量聚类成 \(k = 2\) 个组称为 \(2\)-way 分区，因为我们把向量分成了 \(2\) 个组，分别用索引集 \(G_1\) 和 \(G_2\) 表示。假设我们在 \(n\) 个向量 \(\mathbf{x}_1,\ldots,\mathbf{x}_N\) 上运行 \(k\)-means 算法，其中 \(k = 2\)。证明存在一个非零向量 \(\mathbf{w}\) 和一个标量 \(v\)，使得

$$ \mathbf{w}^T \mathbf{x}_i + v \geq 0, \forall i \in G_1, \quad \mathbf{w}^T \mathbf{x}_i + v \leq 0, \forall i \in G_2. $$

换句话说，仿射函数 \(f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + v\) 在第一组上大于或等于零，在第二组上小于或等于零。这被称为两组的线性分离。[提示：考虑函数 \(\|\mathbf{x} - \mathbf{z}_1\|_2² - \|\mathbf{x} - \mathbf{z}_2\|_2²\)，其中 \(\mathbf{z}_1\) 和 \(\mathbf{z}_2\) 是组代表。] \(\lhd\)

**1.10** (改编自 [ASV]) 设 \(0 < p \leq 1\)。一个随机变量 \(X\) 如果其可能值为 \(\{1, 2, 3, \ldots\}\)，并且 \(X\) 满足 \(\mathbb{P}(X = k) = (1 - p)^{k - 1}p\) 对于正整数 \(k\)，则 \(X\) 具有成功参数 \(p\) 的几何分布。其期望值为 \(1/p\)，方差为 \((1 - p)/p²\)。

设 \(X\) 是参数为 \(p = 1/6\) 的几何随机变量。

a) 使用 *马尔可夫不等式* 找到 \(\mathbb{P}(X \geq 16)\) 的上界。

b) 使用 *切比雪夫不等式* 找到 \(\mathbb{P}(X \geq 16)\) 的上界。

c) 使用几何级数的和显式计算概率 \(\mathbb{P}(X \geq 16)\) 并与您推导的上界进行比较。 \(\lhd\)

**1.11** (改编自 [ASV]) 设 \(0 < \lambda < +\infty\)。一个随机变量 \(X\) 如果 \(X\) 的密度函数为 \(f(x) = \lambda e^{-\lambda x}\)，对于 \(x \geq 0\) 和 \(0\) 否则，则 \(X\) 具有参数 \(\lambda\) 的指数分布。其期望值为 \(1/\lambda\)，方差为 \(1/\lambda²\)。

设 \(X\) 是参数为 \(\lambda = 1/2\) 的指数随机变量。

a) 使用 *马尔可夫不等式* 找到 \(\mathbb{P}(X > 6)\) 的上界。

b) 使用 *切比雪夫不等式* 找到 \(\mathbb{P}(X > 6)\) 的上界。

c) 使用积分显式计算概率 \(\mathbb{P}(X > 6)\) 并与您推导的上界进行比较。 \(\lhd\)

**1.12** (改编自 [ASV]) 假设 \(X\) 是一个非负随机变量，且 \(\mathbb{E}[X] = 10\)。

a) 使用 *马尔可夫不等式* 给出 \(X\) 大于 \(15\) 的概率的上界。

b) 假设我们还知道 \(\mathrm{Var}(X) = 3\)。使用 *切比雪夫不等式* 给出比部分 (a) 更好的 \(\mathbb{P}(X > 15)\) 的上界。

c) 假设 \(Y_1,Y_2,\ldots,Y_{300}\) 是与 \(X\) 具有相同分布的独立同分布随机变量，因此，特别是 \(\mathbb{E}(Y_i) = 10\) 和 \(\mathrm{Var}(Y_i) = 3\)。使用 *切比雪夫不等式* 给出 \(\sum_{i=1}^{300} Y_i\) 大于 \(3060\) 的概率的上界。 \(\lhd\)

**1.13** (改编自 [ASV]) 假设我们有一些独立同分布的随机变量 \(X_1, X_2, X_3,\ldots\)，它们具有有限的均值 \(\mathbb{E}[X_1] = \mu\) 和方差 \(\mathrm{Var}(X_1) = \sigma²\)。设 \(S_n = X_1 +\cdots+X_n\)。证明对于任何固定的 \(\varepsilon > 0\) 和 \(1/2 < \alpha < 1\)，我们有

$$ \lim_{n \to +\infty} \mathbb{P}\left[\left|\frac{S_n - n\mu}{n^\alpha}\right| < \varepsilon\right] = 1. $$

\(\lhd\)

**1.14** (改编自 [ASV]) 通过模仿 *大数定律* 的证明，证明以下变体。假设我们有一些随机变量 \(X_1, X_2, \ldots\)，每个都有有限的均值 \(\mathbb{E}[X_i] = \mu\) 和方差 \(\mathrm{Var}(X_i) = \sigma²\)。进一步假设当 \(|i - j| \geq 2\) 时，\(\mathrm{Cov}(X_i,X_j) = 0\)，并且存在一个常数 \(c > 0\)，使得对于所有 \(i\)，\(|\mathrm{Cov}(X_i, X_{i+1})| < c\)。令 \(S_n =X_1 +\cdots+X_n\)。那么对于任何固定的 \(\varepsilon>0\)，我们有

$$ \lim_{n\to +\infty} \mathbb{P}\left[\left|\frac{S_n}{n} - \mu\right|\right]<\varepsilon=1. $$

\(\lhd\)

**1.15** (改编自 [ASV]) 通过模仿 *切比雪夫不等式* 的证明，证明以下变体。令 \(X\) 为具有有限均值 \(\mu\) 的随机变量，并且对于某个 \(s > 0\)，\(\mathbb{E}[\exp(s(X - \mu)] < +\infty\)。那么对于 \(c > 0\)，我们有

$$ \mathbb{P}(X \geq \mu + c) \leq \frac{\mathbb{E}[\exp(s(X - \mu)]}{e^{s c}}. $$

仔细证明你的答案。\(\lhd\)

**1.16** (改编自 [ASV]) 回忆一下，实值随机变量 \(Z\) 的累积分布函数 (CDF) 是函数 \(F_Z(z) = \mathbb{P}[Z \leq z]\)，对于所有 \(z \in \mathbb{R}\)。令 \(X_1, X_2, \ldots , X_n\) 是具有相同的累积分布函数 \(F\) 的独立随机变量。用最小值和最大值表示

$$ Z = \min(X_1,X_2,\ldots,X_n) \quad\text{和}\quad W = \max(X_1,X_2,\ldots,X_n). $$

找到 \(Z\) 和 \(W\) 的累积分布函数 \(F_Z\) 和 \(F_W\)。\(\lhd\)

**1.17** (改编自 [ASV]) 假设 \(X_1, X_2, \ldots\) 是具有参数 \(\lambda = 1\) 的指数分布的独立同分布随机变量（参见上文的练习 1.11），令 \(M_n = \max(X_1,\ldots,X_n)\)。证明对于任何 \(x \in \mathbb{R}\)，我们有

$$ \lim_{n \to +\infty} \mathbb{P}(M_n - \ln n \leq x) = \exp(-e^{-x}). $$

右侧是 Gumbel 分布的累积分布函数，是极值分布的一个例子。[提示：使用练习 1.16 来显式计算 \(\mathbb{P}(M_n \leq \ln n + x)\)，然后求 \(n \to +\infty\) 时的极限。] \(\lhd\)

**1.18** (改编自 [ASV]) 设 \(A\) 和 \(B\) 是两个不相交的事件。它们在什么条件下是独立的？\(\lhd\)

**1.19** (改编自 [ASV]) 我们从集合 \(\{10, 11, 12, \ldots , 99\}\) 中均匀随机选择一个数字。

a) 令 \(X\) 为所选数字的第一个数字，\(Y\) 为第二个数字。证明 \(X\) 和 \(Y\) 是独立的随机变量。

b) 令 \(X\) 为所选数字的第一个数字，\(Z\) 为两个数字的和。证明 \(X\) 和 \(Z\) 不是独立的。

\(\lhd\)

**1.20** (改编自 [ASV]) 假设 \(X\) 和 \(Y\) 的联合概率密度函数为 \(f(x,y) = 2e^{-(x+2y)}\)，对于 \(x > 0, y > 0\) 和其他地方为 \(0\)。证明 \(X\) 和 \(Y\) 是独立的随机变量，并给出它们的边缘分布。[*提示:* 回忆一下问题 1.11 中的指数分布。] \(\lhd\)

**1.21** (改编自 [ASV]) 假设 \(X,Y\) 有联合概率密度函数

$$ f(x,y) = c\exp\left[-\frac{x²}{2} - \frac{(x - y)²}{2}\right], $$

对于 \(x,y \in \mathbb{R}²\)，对于某个常数 \(c > 0\)。

a) 求常数 \(c\) 的值。[*提示:* 积分的顺序很重要。你可以不进行复杂的积分来完成这个任务。具体来说，回忆一下对于任何 \(\mu \in \mathbb{R}\) 和 \(\sigma > 0\)，以下等式成立

$$ \int_{-\infty}^{+\infty} \frac{1}{\sqrt{2 \pi \sigma²}} \exp\left(-\frac{(z - \mu)²}{2\sigma²}\right) \mathrm{d} z= 1. $$

]

b) 找到 \(X\) 和 \(Y\) 的边缘密度函数。[*提示:* 你可以不进行复杂的积分来完成这个任务。完成平方。]

c) 判断 \(X\) 和 \(Y\) 是否独立。证明你的答案。 \(\lhd\)

**1.22** (改编自 [ASV]) 设 \(p(x, y)\) 是 \((X, Y)\) 的联合概率质量函数。假设存在两个函数 \(a\) 和 \(b\)，使得对于 \(X\) 的所有可能值 \(x\) 和 \(Y\) 的所有可能值 \(y\)，都有 \(p(x,y) = a(x)b(y)\)。证明 \(X\) 和 \(Y\) 是独立的。不要假设 \(a\) 和 \(b\) 是概率质量函数。\(\lhd\)

**1.23** (改编自 [BHK]) a) 设 \(X\) 和 \(Y\) 是在 \([-1/2 , 1/2]\) 上具有均匀分布的独立随机变量。计算 \(\mathbb{E}[X]\)，\(\mathbb{E}[X²]\)，\(\mathrm{Var}[X²]\)，\(\mathbb{E}[X - Y]\)，\(\mathbb{E}[XY]\)，和 \(\mathbb{E}[(X - Y)²]\)。

b) 两个在单位 d 维立方体 \(\mathcal{C} = [-1/2,1/2]^d\) 内随机生成的点之间的期望平方欧几里得距离是多少？\(\lhd\)

**1.24** (改编自 [BHK]) a) 对于任意 \(a > 0\)，给出一个非负随机变量 \(X\) 的概率分布，使得

$$ \mathbb{P}[X \geq a] = \frac{\mathbb{E}[X]}{a}. $$

b) 证明对于任何 \(c > 0\)，存在一个分布使得 *切比雪夫不等式* 是紧的，即 \( \mathbb{P}[|X - \mathbb{E}[X]| \geq c] = \frac{\mathrm{Var}[X]}{c²}. \)

[*提示:* 选择一个关于 \(0\) 对称的分布。] \(\lhd\)

**1.25** (改编自 [BHK]) 设 \(X_1, X_2, \ldots , X_n\) 是具有均值 \(\mu\) 和方差 \(\sigma²\) 的独立同分布随机变量。设

$$ \overline{X}_n = \frac{1}{n} \sum_{i=1}^n X_i, $$

作为样本均值。假设使用样本均值来估计方差如下

$$ S²_n = \frac{1}{n} \sum_{i=1}^n \left(X_i - \overline{X}_n\right)². $$

计算 \(\mathbb{E}(S²_n)\)。[*提示:* 将 \(X_i - \overline{X}_n\) 替换为 \((X_i - \mu) - (\overline{X}_n - \mu)\)。] \(\lhd\)

**1.26** 设 \(f\) 和 \(g\) 在 \(x\) 处有导数，设 \(\alpha\) 和 \(\beta\) 是常数。从导数的定义出发证明：

$$ [\alpha f(x) + \beta g(x)]' = \alpha f'(x) + \beta g'(x). $$

\(\lhd\)

**1.27** 设 \(Z_1 \sim \mathrm{N}(\mu_1, \sigma_1²)\) 和 \(Z_2 \sim \mathrm{N}(\mu_2, \sigma_2²)\) 是独立的正态变量，均值分别为 \(\mu_1\) 和 \(\mu_2\)，方差分别为 \(\sigma_1²\) 和 \(\sigma_2²\)。回忆一下，\(Z_1 + Z_2\) 仍然是正态分布。

a) \(Z_1 + Z_2\) 的均值和方差是什么？

b) 设 \(\mathbf{X}_1, \mathbf{X}_2, \mathbf{Y}_1\) 是独立的 \(d\) 维球面高斯变量，均值为 \(-w \mathbf{e}_1\)，方差为 \(1\)。设 \(\mathbf{Y}_2\) 是一个独立的 \(d\) 维球面高斯变量，均值为 \(w \mathbf{e}_1\)，方差为 \(1\)。显式地计算 \(\mathbb{E}[\|\mathbf{X}_1 - \mathbf{X}_2\|²]\) 和 \(\mathbb{E}[\|\mathbf{Y}_1 - \mathbf{Y}_2\|²]\)。 \(\lhd\)

**1.28** 假设我们给出了 \(\mathbb{R}^d\) 中的 \(n\) 个向量 \(\mathbf{x}_1,\ldots,\mathbf{x}_n\) 和一个划分 \(C_1, \ldots, C_k \subseteq [n]\)。设 \(n_i = |C_i|\) 为簇 \(C_i\) 的大小，并设

$$ \boldsymbol{\mu}_i^* = \frac{1}{n_i} \sum_{j\in C_i} \mathbf{x}_j $$

为 \(C_i\) 的质心，对于 \(i=1,\ldots,k\)。

a) 证明：

$$ \sum_{j \in C_i} \|\mathbf{x}_j - \boldsymbol{\mu}_i^*\|² = \left(\sum_{j \in C_i} \|\mathbf{x}_j\|²\right) - n_i\|\boldsymbol{\mu}_i^*\|². $$

b) 证明：

$$\begin{split} \|\boldsymbol{\mu}_i^*\|² = \frac{1}{n_i²}\left(\sum_{j \in C_i} \|\mathbf{x}_j\|² + \sum_{\substack{j, \ell \in C_i\\j \neq \ell}} \mathbf{x}_j^T\mathbf{x}_\ell\right). \end{split}$$

c) 证明：

$$\begin{split} \sum_{\substack{j, \ell \in C_i\\j \neq \ell}} \|\mathbf{x}_j - \mathbf{x}_\ell\|² = 2(n_i-1)\sum_{j \in C_i} \|\mathbf{x}_j\|² - 2 \sum_{\substack{j, \ell \in C_i\\j \neq \ell}} \mathbf{x}_j^T\mathbf{x}_\ell. \end{split}$$

d) 将 a), b), c) 结合起来证明在所有 \([n]\) 的划分 \(C_1, \ldots, C_k\) 上最小化 \(k\)-means 目标函数 \(\mathcal{G}(C_1,\ldots,C_k)\) 等价于最小化

$$ \sum_{i=1}^k \frac{1}{2 |C_i|} \sum_{j,\ell \in C_i} \|\mathbf{x}_j - \mathbf{x}_\ell\|². $$

\(\lhd\)

**1.29** 假设 \(A \in \mathbb{R}^{n \times m}\) 的行由 \(\mathbf{r}_1,\ldots,\mathbf{r}_n \in \mathbb{R}^m\) 的转置给出，而 \(A\) 的列由 \(\mathbf{c}_1,\ldots,\mathbf{c}_m \in \mathbb{R}^n\) 给出。用这些向量的形式给出 \(A^T A\) 和 \(A A^T\) 的元素表达式。 \(\lhd\)

**1.30** 使用 \(\bSigma_\bX\) 的矩阵形式给出协方差正定性的证明。 \(\lhd\)

**1.31** 使用 **柯西-施瓦茨不等式** 证明相关系数位于 \([-1,1]\) 之间。 \(\lhd\)

**1.32** 设 \(f(x,y) = g(x) + h(y)\)，其中 \(x, y \in \mathbb{R}\) 且 \(g,h\) 是实值连续可微函数。计算 \(f\) 的梯度，用 \(g\) 和 \(h\) 的导数表示。\(\lhd\)

**1.33** 设 \(A = [a]\) 是一个 \(1\times 1\) 的正定矩阵。证明 \(a > 0\)。\(\lhd\)

**1.34** 证明正定矩阵的对角元素必然是正的。\(\lhd\)

**1.35** 设 \(A, B \in \mathbb{R}^{n \times m}\) 且 \(c \in \mathbb{R}\)。证明：

a) \((A + B)^T = A^T + B^T\)

b) \((c A)^T = c A^T\)

\(\lhd\)
