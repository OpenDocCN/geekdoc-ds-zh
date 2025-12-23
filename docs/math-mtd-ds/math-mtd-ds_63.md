# 8.2\. 背景：雅可比、链式法则以及自动微分的简要介绍

> 原文：[`mmids-textbook.github.io/chap08_nn/02_chain/roch-mmids-nn-chain.html`](https://mmids-textbook.github.io/chap08_nn/02_chain/roch-mmids-nn-chain.html)

我们引入多个变量的向量值函数的雅可比以及在此更一般设置下的**链式法则**。我们还简要介绍了自动微分。我们首先介绍一些额外的矩阵代数。

## 8.2.1\. 更多的矩阵代数：Hadamard 和 Kronecker 积#

首先，我们介绍 Hadamard 积$\idx{Hadamard 积}\xdi$和除法$\idx{Hadamard 除法}\xdi$。相同维度的两个矩阵（或向量）的 Hadamard 积，$A = (a_{i,j})_{i \in [n], j \in [m]}, B = (b_{i,j})_{i\in [n], j \in [m]} \in \mathbb{R}^{n \times m}$，定义为逐元素乘积

$$ A \odot B = (a_{i,j} b_{i,j})_{i\in [n], j \in [m]}. $$

同样，Hadamard 除法定义为逐元素除法

$$ A \oslash B = (a_{i,j} / b_{i,j})_{i\in [n], j \in [m]} $$

在这里我们假设对于所有的 $i,j$，都有 $b_{i,j} \neq 0$。

**示例：** 作为一个说明性的例子，

$$\begin{split} \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ \end{bmatrix} \odot \begin{bmatrix} 0 & 5 \\ 6 & 7 \\ \end{bmatrix} = \begin{bmatrix} 1 \times 0 & 2 \times 5\\ 3 \times 6 & 4 \times 7\\ \end{bmatrix} = \begin{bmatrix} 0 & 10\\ 18 & 28\\ \end{bmatrix}. \end{split}$$

$\lhd$

回想一下，$\mathbf{1}$ 是全一向量，并且对于 $\mathbf{x} = (x_1,\ldots,x_n) \in \mathbb{R}^n$，$\mathrm{diag}(\mathbf{x}) \in \mathbb{R}^{n \times n}$ 是对角线元素为 $x_1,\ldots,x_n$ 的对角矩阵。

**引理** **(Hadamard 积的性质)** $\idx{Hadamard 积的性质}\xdi$ 设 $\mathbf{a} = (a_1,\ldots,a_n), \mathbf{b} = (b_1,\ldots,b_n), \mathbf{c} = (c_1,\ldots,c_n) \in \mathbb{R}^n$。那么以下性质成立：

a) $\mathrm{diag}(\mathbf{a}) \, \mathbf{b} = \mathrm{diag}(\mathbf{a} \odot \mathbf{b})$;

b) $\mathbf{a}^T(\mathbf{b} \odot \mathbf{c}) = \mathbf{1}^T(\mathbf{a} \odot \mathbf{b} \odot \mathbf{c})$

并且，当 $a_i \neq 0$ 对于所有的 $i$ 成立时，以下性质也成立：

c) $\mathrm{diag}(\mathbf{a}) \, (\mathbf{b} \oslash \mathbf{a}) = \mathrm{diag}(\mathbf{b})$;

d) $\mathbf{a}^T \, (\mathbf{b} \oslash \mathbf{a}) = \mathbf{1}^T \mathbf{b}$.

$\flat$

**证明：** a) 对角矩阵与向量的乘积产生一个新的向量，其原始坐标被相应的对角线元素相乘。这证明了第一个命题。

b) 我们有

$$ \mathbf{a}^T \, (\mathbf{b} \odot \mathbf{c}) = \sum_{i=1}^n a_i (b_i c_i) = \mathbf{1}^T (\mathbf{a} \odot \mathbf{b} \odot \mathbf{c}). $$

c) 和 d) 分别由 a) 和 b) 推出。

$\square$

其次，我们介绍克罗内克积$\idx{克罗内克积}\xdi$。设 $A = (a_{i,j})_{i \in [n], j \in [m]} \in \mathbb{R}^{n \times m}$ 和 $B = (b_{i,j})_{i \in [p], j \in [q]} \in \mathbb{R}^{p \times q}$ 是任意矩阵。它们的克罗内克积，记作 $A \otimes B \in \mathbb{R}^{np \times mq}$，是以下形式的矩阵

$$\begin{split} A \otimes B = \begin{pmatrix} a_{1,1} B & \cdots & a_{1,m} B \\ \vdots & \ddots & \vdots \\ a_{n,1} B & \cdots & a_{n,m} B \end{pmatrix}. \end{split}$$

**示例**：以下是一个来自 [维基百科](https://en.wikipedia.org/wiki/Kronecker_product#Examples) 的简单说明性示例：

$$\begin{split} \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ \end{bmatrix} \otimes \begin{bmatrix} 0 & 5 \\ 6 & 7 \\ \end{bmatrix} = \begin{bmatrix} 1 \begin{bmatrix} 0 & 5 \\ 6 & 7 \\ \end{bmatrix} & 2 \begin{bmatrix} 0 & 5 \\ 6 & 7 \\ \end{bmatrix} \\ 3 \begin{bmatrix} 0 & 5 \\ 6 & 7 \\ \end{bmatrix} & 4 \begin{bmatrix} 0 & 5 \\ 6 & 7 \\ \end{bmatrix} \\ \end{bmatrix} = \begin{bmatrix} 0 & 5 & 0 & 10 \\ 6 & 7 & 12 & 14 \\ 0 & 15 & 0 & 20 \\ 18 & 21 & 24 & 28 \end{bmatrix}. \end{split}$$

$\lhd$

**示例**：**（外积）** $\idx{外积}\xdi$ 这是另一个我们之前遇到的例子，两个向量 $\mathbf{u} = (u_1,\ldots,u_n) \in \mathbb{R}^n$ 和 $\mathbf{v} = (v_1,\ldots, v_m) \in \mathbb{R}^m$ 的外积。回忆一下，外积按块形式定义为 $n \times m$ 矩阵

$$ \mathbf{u} \mathbf{v}^T = \begin{pmatrix} v_1 \mathbf{u} & \cdots & v_m \mathbf{u} \end{pmatrix} = \mathbf{v}^T \otimes \mathbf{u}. $$

等价地，

$$\begin{split} \mathbf{u} \mathbf{v}^T = \begin{pmatrix} u_1 \mathbf{v}^T\\ \vdots\\ u_n \mathbf{v}^T \end{pmatrix} = \mathbf{u} \otimes \mathbf{v}^T. \end{split}$$

$\lhd$

**示例**：**（续）** 在前面的例子中，克罗内克积是可交换的（即，我们有 $\mathbf{v}^T \otimes \mathbf{u} = \mathbf{u} \otimes \mathbf{v}^T$）。在一般情况下并非如此。回到上面的第一个例子，请注意

$$\begin{split} \begin{bmatrix} 0 & 5 \\ 6 & 7 \\ \end{bmatrix} \otimes \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ \end{bmatrix} = \begin{bmatrix} 0 \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ \end{bmatrix} & 5 \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ \end{bmatrix} \\ 6 \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ \end{bmatrix} & 7 \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ \end{bmatrix} \\ \end{bmatrix} = \begin{bmatrix} 0 & 0 & 5 & 10 \\ 0 & 0 & 15 & 20 \\ 6 & 12 & 7 & 14 \\ 18 & 24 & 21 & 28 \end{bmatrix}. \end{split}$$

你可以检查，这与我们按相反顺序得到的结果不同。$\lhd$

以下引理的证明留给读者作为练习。

**引理** **（克罗内克积的性质）** $\idx{克罗内克积的性质}\xdi$ 克罗内克积具有以下性质：

a) 如果 $B, C$ 是相同维度的矩阵

$$ A \otimes (B + C) = A \otimes B + A \otimes C \quad \text{并且}\quad (B + C) \otimes A = B \otimes A + C \otimes A. $$

b) 如果 $A, B, C, D$ 是可以形成矩阵乘积 $AC$ 和 $BD$ 的矩阵，那么

$$ (A \otimes B)\,(C \otimes D) = (AC) \otimes (BD). $$

c) 如果 $A, C$ 是相同维度的矩阵，且 $B, D$ 是相同维度的矩阵，那么

$$ (A \otimes B)\odot(C \otimes D) = (A\odot C) \otimes (B\odot D). $$

d) 如果 $A,B$ 是可逆的，那么

$$ (A \otimes B)^{-1} = A^{-1} \otimes B^{-1}. $$

e) $A \otimes B$ 的转置是

$$ (A \otimes B)^T = A^T \otimes B^T. $$

f) 如果 $\mathbf{u}$ 是一个列向量，且 $A, B$ 是可以形成矩阵乘积 $AB$ 的矩阵，那么

$$ (\mathbf{u} \otimes A) B = \mathbf{u} \otimes (AB) \quad\text{并且}\quad (A \otimes \mathbf{u}) B = (AB) \otimes \mathbf{u}. $$

同样地，

$$ A \,(\mathbf{u}^T \otimes B) = \mathbf{u}^T \otimes (AB) \quad\text{并且}\quad A \,(B \otimes \mathbf{u}^T) = (AB) \otimes \mathbf{u}^T. $$

$\flat$

## 8.2.2\. 雅可比矩阵#

回想一下，实变函数的导数是函数相对于变量变化的速率。另一种说法是 $f'(x)$ 是 $f$ 在 $x$ 处的切线斜率。形式上，可以在 $x$ 的邻域内通过以下线性函数来近似 $f(x)$

$$ f(x + h) = f(x) + f'(x) h + r(h), $$

其中 $r(h)$ 相对于 $h$ 是可以忽略的，

$$ \lim_{h\to 0} \frac{r(h)}{h} = 0. $$

事实上，定义

$$ r(h) = f(x + h) - f(x) - f'(x) h. $$

然后根据导数的定义

$$ \lim_{h\to 0} \frac{r(h)}{h} = \lim_{h\to 0} \frac{f(x + h) - f(x) - f'(x) h}{h} = \lim_{h\to 0} \left[\frac{f(x + h) - f(x)}{h} - f'(x) \right] = 0. $$

对于向量值函数，我们有以下推广。设 $\mathbf{f} = (f_1, \ldots, f_m) : D \to \mathbb{R}^m$，其中 $D \subseteq \mathbb{R}^d$，并且设 $\mathbf{x} \in D$ 是 $D$ 的一个内点。我们说 $\mathbf{f}$ 在 $\mathbf{x}$ 处可微$\idx{可微}\xdi$，如果存在一个矩阵 $A \in \mathbb{R}^{m \times d}$ 使得

$$ \mathbf{f}(\mathbf{x}+\mathbf{h}) = \mathbf{f}(\mathbf{x}) + A \mathbf{h} + \mathbf{r}(\mathbf{h}) $$

其中

$$ \lim_{\mathbf{h} \to 0} \frac{\|\mathbf{r}(\mathbf{h})\|_2}{\|\mathbf{h}\|_2} = 0. $$

矩阵 $\mathbf{f}'(\mathbf{x}) = A$ 被称为 $\mathbf{f}$ 在 $\mathbf{x}$ 处的微分$\idx{微分}\xdi$，并且我们看到仿射映射 $\mathbf{f}(\mathbf{x}) + A \mathbf{h}$ 提供了 $\mathbf{f}$ 在 $\mathbf{x}$ 附近的近似。

我们在这里不会推导出完整的理论。在 $\mathbf{f}$ 的每个分量在 $\mathbf{x}$ 的邻域内具有连续偏导数的情形下，微分存在且等于定义如下的雅可比矩阵。

**定义** **(雅可比)** $\idx{雅可比}\xdi$ 设 $\mathbf{f} = (f_1, \ldots, f_m) : D \to \mathbb{R}^m$ 其中 $D \subseteq \mathbb{R}^d$，并且设 $\mathbf{x}_0 \in D$ 是 $D$ 的一个内部点，其中对于所有 $i, j$，$\frac{\partial f_j (\mathbf{x}_0)}{\partial x_i}$ 存在。在 $\mathbf{x}_0$ 处 $\mathbf{f}$ 的雅可比是 $m \times d$ 矩阵

$$\begin{split} J_{\mathbf{f}}(\mathbf{x}_0) = \begin{pmatrix} \frac{\partial f_1 (\mathbf{x}_0)}{\partial x_1} & \ldots & \frac{\partial f_1 (\mathbf{x}_0)}{\partial x_d}\\ \vdots & \ddots & \vdots\\ \frac{\partial f_m (\mathbf{x}_0)}{\partial x_1} & \ldots & \frac{\partial f_m (\mathbf{x}_0)}{\partial x_d} \end{pmatrix}. \end{split}$$

$\natural$

**定理** **(微分和雅可比)** $\idx{微分和雅可比定理}\xdi$ 设 $\mathbf{f} = (f_1, \ldots, f_m) : D \to \mathbb{R}^m$ 其中 $D \subseteq \mathbb{R}^d$，并且设 $\mathbf{x}_0 \in D$ 是 $D$ 的一个内部点。假设对于所有 $i, j$，$\frac{\partial f_j (\mathbf{x}_0)}{\partial x_i}$ 存在并且在一个以 $\mathbf{x}_0$ 为中心的开球内是连续的。那么在 $\mathbf{x}_0$ 处的微分等于 $\mathbf{f}$ 在 $\mathbf{x}_0$ 处的雅可比。 $\sharp$

回想一下，对于任何 $A, B$，当 $AB$ 定义良好时，有 $\|A B \|_F \leq \|A\|_F \|B\|_F$。这特别适用于 $B$ 是一个列向量时，此时 $\|B\|_F$ 是其欧几里得范数。

**证明**：根据**平均值定理**，对于每个 $i$，存在 $\xi_{\mathbf{h},i} \in (0,1)$ 使得

$$ f_i(\mathbf{x}_0+\mathbf{h}) = f_i(\mathbf{x}_0) + \nabla f_i(\mathbf{x}_0 + \xi_{\mathbf{h},i} \mathbf{h})^T \mathbf{h}. $$

定义

$$\begin{split} \tilde{J}(\mathbf{h}) = \begin{pmatrix} \frac{\partial f_1 (\mathbf{x}_0 + \xi_{\mathbf{h},1} \mathbf{h})}{\partial x_1} & \ldots & \frac{\partial f_1 (\mathbf{x}_0 + \xi_{\mathbf{h},1} \mathbf{h})}{\partial x_d}\\ \vdots & \ddots & \vdots\\ \frac{\partial f_m (\mathbf{x}_0 + \xi_{\mathbf{h},m} \mathbf{h})}{\partial x_1} & \ldots & \frac{\partial f_m (\mathbf{x}_0 + \xi_{\mathbf{h},m} \mathbf{h})}{\partial x_d} \end{pmatrix}. \end{split}$$

因此我们有

$$ \mathbf{f}(\mathbf{x}_0+\mathbf{h}) - \mathbf{f}(\mathbf{x}_0) - J_{\mathbf{f}}(\mathbf{x}_0)\,\mathbf{h} = \tilde{J}(\mathbf{h}) \,\mathbf{h} - J_{\mathbf{f}}(\mathbf{x}_0)\,\mathbf{h} = \left(\tilde{J}(\mathbf{h}) - J_{\mathbf{f}}(\mathbf{x}_0)\right)\,\mathbf{h}. $$

当 $\mathbf{h}$ 趋向于 $0$ 时取极限，我们得到

$$\begin{align*} \lim_{\mathbf{h} \to 0} \frac{\|\mathbf{f}(\mathbf{x}_0+\mathbf{h}) - \mathbf{f}(\mathbf{x}_0) - J_{\mathbf{f}}(\mathbf{x}_0)\,\mathbf{h}\|_2}{\|\mathbf{h}\|_2} &= \lim_{\mathbf{h} \to 0} \frac{\|\left(\tilde{J}(\mathbf{h}) - J_{\mathbf{f}}(\mathbf{x}_0)\right)\,\mathbf{h}\|_2}{\|\mathbf{h}\|_2}\\ &\leq \lim_{\mathbf{h} \to 0} \frac{\|\tilde{J}(\mathbf{h}) - J_{\mathbf{f}}(\mathbf{x}_0)\|_F \|\mathbf{h}\|_2}{\|\mathbf{h}\|_2}\\ &= 0, \end{align*}$$

通过偏导数的连续性。$\square$

**示例：** 向量值函数的一个例子是

$$\begin{split} \mathbf{g}(x_1,x_2) = \begin{pmatrix} g_1(x_1,x_2)\\ g_2(x_1,x_2)\\ g_3(x_1,x_2) \end{pmatrix} = \begin{pmatrix} 3 x_1²\\ x_2\\ x_1 x_2 \end{pmatrix}. \end{split}$$

其 Jacobian 是

$$\begin{split} J_{\mathbf{g}}(x_1, x_2) = \begin{pmatrix} \frac{\partial g_1 (x_1, x_2)}{\partial x_1} & \frac{\partial g_1 (x_1, x_2)}{\partial x_2}\\ \frac{\partial g_2 (x_1, x_2)}{\partial x_1} & \frac{\partial g_2 (x_1, x_2)}{\partial x_2}\\ \frac{\partial g_3 (x_1, x_2)}{\partial x_1} & \frac{\partial g_3 (x_1, x_2)}{\partial x_2} \end{pmatrix} = \begin{pmatrix} 6 x_1 & 0\\ 0 & 1\\ x_2 & x_1 \end{pmatrix}. \end{split}$$

$\lhd$

**示例：** **（梯度与 Jacobian）** 对于一个连续可微的实值函数 $f : D \to \mathbb{R}$，其 Jacobian 简化为行向量

$$ J_{f}(\mathbf{x}_0) = \left(\frac{\partial f (\mathbf{x}_0)}{\partial x_1}, \ldots, \frac{\partial f (\mathbf{x}_0)}{\partial x_d}\right)^T = \nabla f(\mathbf{x}_0)^T $$

其中 $\nabla f(\mathbf{x}_0)$ 是 $f$ 在 $\mathbf{x}_0$ 处的梯度。$\lhd$

**示例：** **（Hessian 和 Jacobian）** 对于一个二阶连续可微的实值函数 $f : D \to \mathbb{R}$，其梯度的 Jacobian 是

$$\begin{split} J_{\nabla f}(\mathbf{x}_0) = \begin{pmatrix} \frac{\partial² f(\mathbf{x}_0)}{\partial x_1²} & \cdots & \frac{\partial² f(\mathbf{x}_0)}{\partial x_d \partial x_1}\\ \vdots & \ddots & \vdots\\ \frac{\partial² f(\mathbf{x}_0)}{\partial x_1 \partial x_d} & \cdots & \frac{\partial² f(\mathbf{x}_0)}{\partial x_d²} \end{pmatrix}, \end{split}$$

即，$f$ 在 $\mathbf{x}_0$ 处的 Hessian（转置的，但这没有关系；为什么？）。$\lhd$

**示例：** **（参数曲线和 Jacobian）** 考虑参数曲线 $\mathbf{g}(t) = (g_1(t), \ldots, g_d(t)) \in \mathbb{R}^d$，其中 $t$ 在 $\mathbb{R}$ 的某个闭区间内。假设 $\mathbf{g}(t)$ 在 $t$ 处连续可微，即其每个分量都是。

然后

$$\begin{split} J_{\mathbf{g}}(t) = \begin{pmatrix} g_1'(t)\\ \vdots\\ g_m'(t) \end{pmatrix} = \mathbf{g}'(t). \end{split}$$

$\lhd$

**示例：** **（仿射映射）** 设 $A = (a_{i,j})_{i,j} \in \mathbb{R}^{m \times d}$ 和 $\mathbf{b} = (b_1,\ldots,b_m) \in \mathbb{R}^{m}$。定义向量值函数 $\mathbf{f} = (f_1, \ldots, f_m) : \mathbb{R}^d \to \mathbb{R}^m$ 为

$$ \mathbf{f}(\mathbf{x}) = A \mathbf{x} + \mathbf{b}. $$

这是一个仿射映射。特别地，在线性映射的情况下，当 $\mathbf{b} = \mathbf{0}$ 时，

$$ \mathbf{f}(\mathbf{x} + \mathbf{y}) = A(\mathbf{x} + \mathbf{y}) = A\mathbf{x} + A\mathbf{y} = \mathbf{f}(\mathbf{x}) + \mathbf{f}(\mathbf{y}). $$

用 $\boldsymbol{\alpha}_1^T,\ldots,\boldsymbol{\alpha}_m^T$ 表示 $A$ 的行。

我们计算 $\mathbf{f}$ 在 $\mathbf{x}$ 处的 Jacobian。注意，

$$\begin{align*} \frac{\partial f_i (\mathbf{x})}{\partial x_j} &= \frac{\partial}{\partial x_j}[\boldsymbol{\alpha}_i^T \mathbf{x} + b_i]\\ &= \frac{\partial}{\partial x_j}\left[\sum_{\ell=1}^m a_{i,\ell} x_{\ell} + b_i\right]\\ &= a_{i,j}. \end{align*}$$

所以

$$ J_{\mathbf{f}}(\mathbf{x}) = A. $$

$\lhd$

以下重要的例子是雅可比的一个不太直接的应用。

引入矩阵 $A = (a_{i,j})_{i,j} \in \mathbb{R}^{n \times m}$ 的向量化 $\mathrm{vec}(A) \in \mathbb{R}^{nm}$ 作为向量是有用的，作为

$$ \mathrm{vec}(A) = (a_{1,1},\ldots,a_{n,1},a_{1,2},\ldots,a_{n,2},\ldots,a_{1,m},\ldots,a_{n,m}). $$

那就是通过将 $A$ 的列堆叠在一起得到的。

**示例：** **(关于其矩阵的线性映射的雅可比)** 我们对上一个例子采取不同的方法。在数据科学应用中，计算线性映射 $X \mathbf{z}$ 的雅可比（相对于矩阵 $X \in \mathbb{R}^{n \times m}$）将是有用的。具体来说，对于固定的 $\mathbf{z} \in \mathbb{R}^{m}$，令 $(\mathbf{x}^{(i)})^T$ 为 $X$ 的第 $i$ 行，我们定义函数

$$\begin{split} \mathbf{f}(\mathbf{x}) = X \mathbf{z} = \begin{pmatrix} (\mathbf{x}^{(1)})^T\\ \vdots\\ (\mathbf{x}^{(n)})^T \end{pmatrix} \mathbf{z} = \begin{pmatrix} (\mathbf{x}^{(1)})^T \mathbf{z} \\ \vdots\\ (\mathbf{x}^{(n)})^T \mathbf{z} \end{pmatrix} \end{split}$$

其中 $\mathbf{x} = \mathrm{vec}(X^T) = (\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)})$.

为了计算雅可比，让我们看看与 $\mathbf{x}^{(k)}$ 中的变量相对应的列，即列 $\alpha_k = (k-1) m + 1$ 到 $\beta_k = k m$。注意，只有 $\mathbf{f}$ 的第 $k$ 个分量依赖于 $\mathbf{x}^{(k)}$，所以 $J_{\mathbf{f}}(\mathbf{x})$ 的行 $\neq k$ 对应的列是 $0$。

另一方面，行 $k$ 是我们之前公式中仿射映射梯度的公式中的 $\mathbf{z}^T$。因此，$J_{\mathbf{f}}(\mathbf{x})$ 的列 $\alpha_k$ 到 $\beta_k$ 可以写成 $\mathbf{e}_k \mathbf{z}^T$，其中这里的 $\mathbf{e}_k \in \mathbb{R}^{n}$ 是 $\mathbb{R}^{n}$ 的第 $k$ 个标准基向量。

因此，$J_{\mathbf{f}}(\mathbf{x})$ 可以写成块形式：

$$ J_{\mathbf{f}}(\mathbf{x}) = \begin{pmatrix} \mathbf{e}_1 \mathbf{z}^T & \cdots & \mathbf{e}_{n}\mathbf{z}^T \end{pmatrix} = I_{n\times n} \otimes \mathbf{z}^T =: \mathbb{B}_{n}[\mathbf{z}], $$

其中最后一个等式是一个定义。 $\lhd$

我们还需要一个额外的细节。

**示例：** **(关于其输入和矩阵的线性映射的雅可比)** 再次考虑线性映射 $X \mathbf{z}$ - 这次是作为矩阵 $X \in \mathbb{R}^{n \times m}$ 和向量 $\mathbf{z} \in \mathbb{R}^{m}$ 的函数。也就是说，再次令 $(\mathbf{x}^{(i)})^T$ 为 $X$ 的第 $i$ 行，我们定义函数

$$\begin{split} \mathbf{g}(\mathbf{z}, \mathbf{x}) = X \mathbf{z} = \begin{pmatrix} (\mathbf{x}^{(1)})^T\\ \vdots\\ (\mathbf{x}^{(n)})^T \end{pmatrix} \mathbf{z} = \begin{pmatrix} (\mathbf{x}^{(1)})^T \mathbf{z} \\ \vdots\\ (\mathbf{x}^{(n)})^T \mathbf{z} \end{pmatrix} \end{split}$$

如前所述 $\mathbf{x} = \mathrm{vec}(X^T) = (\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)})$。

为了计算雅可比矩阵，我们将其视为一个分块矩阵，并使用前两个例子。$J_{\mathbf{f}}(\mathbf{z}, \mathbf{x})$ 中对应于 $\mathbf{z}$ 中的变量的列，即列 $1$ 到 $m$，是

$$\begin{split} X = \begin{pmatrix} (\mathbf{x}^{(1)})^T\\ \vdots\\ (\mathbf{x}^{(n)})^T \end{pmatrix} =: \mathbb{A}_{n}[\mathbf{x}]. \end{split}$$

$J_{\mathbf{f}}(\mathbf{z}, \mathbf{x})$ 的列对应于 $\mathbf{x}$ 中的变量，即列 $m + 1$ 到 $m + nm$，是矩阵 $\mathbb{B}_{n}[\mathbf{z}]$。注意，在 $\mathbb{A}_{n}[\mathbf{x}]$ 和 $\mathbb{B}_{n}[\mathbf{z}]$ 中，下标 $n$ 表示矩阵的行数。列数由 $n$ 和输入向量的尺寸决定：

+   $\mathbb{A}_{n}[\mathbf{x}]$ 的长度除以 $n$；

+   $\mathbb{B}_{n}[\mathbf{z}]$ 的长度乘以 $n$。

因此 $J_{\mathbf{f}}(\mathbf{z}, \mathbf{x})$ 可以写成分块形式

$$ J_{\mathbf{f}}(\mathbf{z}, \mathbf{x}) = \begin{pmatrix} \mathbb{A}_{n}[\mathbf{x}] & \mathbb{B}_{n}[\mathbf{z}] \end{pmatrix}. $$

$\lhd$

**EXAMPLE:** **(逐元素函数)** 设 $f : D \to \mathbb{R}$，其中 $D \subseteq \mathbb{R}$，是一个关于单变量的连续可微实值函数。对于 $n \geq 2$，考虑将 $f$ 应用到向量 $\mathbf{x} \in \mathbb{R}^n$ 的每个元素上，即令 $\mathbf{f} : D^n \to \mathbb{R}^n$，具有

$$ \mathbf{f}(\mathbf{x}) = (f_1(\mathbf{x}), \ldots, f_n(\mathbf{x})) = (f(x_1), \ldots, f(x_n)). $$

$\mathbf{f}$ 的雅可比矩阵可以从单变量情况的导数 $f'$ 计算得出。确实，令 $\mathbf{x} = (x_1,\ldots,x_n)$ 使得对于所有 $i$，$x_i$ 是 $D$ 的内点，

$$ \frac{\partial f_j(\mathbf{x})}{\partial x_j} = f'(x_j), $$

当 $\ell \neq j$ 时

$$ \frac{\partial f_\ell(\mathbf{x})}{\partial x_j} =0, $$

因为 $f_\ell(\mathbf{x})$ 实际上并不依赖于 $x_j$。换句话说，雅可比矩阵的第 $j$ 列是 $f'(x_j) \,\mathbf{e}_j$，其中再次 $\mathbf{e}_{j}$ 是 $\mathbb{R}^{n}$ 中的第 $j$ 个标准基向量。

因此 $J_{\mathbf{f}}(\mathbf{x})$ 是对角矩阵，其对角元素为 $f'(x_j)$，$j=1, \ldots, n$，我们用

$$ J_{\mathbf{f}}(\mathbf{x}) = \mathrm{diag}(f'(x_1),\ldots,f'(x_n)). $$

$\lhd$

## 8.2.3\. 链式法则的推广#

正如我们所看到的，函数通常是通过简单函数的复合得到的。我们将使用向量符号 $\mathbf{h} = \mathbf{g} \circ \mathbf{f}$ 表示函数 $\mathbf{h}(\mathbf{x}) = \mathbf{g} (\mathbf{f} (\mathbf{x}))$。

**引理** **(连续函数的复合)** $\idx{连续函数复合引理}\xdi$ 设 $\mathbf{f} : D_1 \to \mathbb{R}^m$，其中 $D_1 \subseteq \mathbb{R}^d$，以及设 $\mathbf{g} : D_2 \to \mathbb{R}^p$，其中 $D_2 \subseteq \mathbb{R}^m$。假设 $\mathbf{f}$ 在 $\mathbf{x}_0$ 处连续，且 $\mathbf{g}$ 在 $\mathbf{f}(\mathbf{x}_0)$ 处连续。那么 $\mathbf{g} \circ \mathbf{f}$ 在 $\mathbf{x}_0$ 处连续。 $\flat$

**链式法则** 给出了复合函数雅可比的公式。

**定理** **(链式法则)** $\idx{链式法则}\xdi$ 设 $\mathbf{f} : D_1 \to \mathbb{R}^m$，其中 $D_1 \subseteq \mathbb{R}^d$，以及设 $\mathbf{g} : D_2 \to \mathbb{R}^p$，其中 $D_2 \subseteq \mathbb{R}^m$。假设 $\mathbf{f}$ 在 $D_1$ 的内点 $\mathbf{x}_0$ 处连续可微，且 $\mathbf{g}$ 在 $D_2$ 的内点 $\mathbf{f}(\mathbf{x}_0)$ 处连续可微。那么

$$ J_{\mathbf{g} \circ \mathbf{f}}(\mathbf{x}_0) = J_{\mathbf{g}}(\mathbf{f}(\mathbf{x}_0)) \,J_{\mathbf{f}}(\mathbf{x}_0) $$

作为矩阵的乘积。 $\sharp$

直观上，雅可比提供了函数在点附近的线性近似。线性映射的复合对应于相关矩阵的乘积。同样，复合函数的雅可比是雅可比矩阵的乘积。

*证明：* 为了避免混淆，我们将 $\mathbf{f}$ 和 $\mathbf{g}$ 视为具有不同名称的变量函数，具体来说，

$$ \mathbf{f}(\mathbf{x}) = (f_1(x_1,\ldots,x_d),\ldots,f_m(x_1,\ldots,x_d)) $$

以及

$$ \mathbf{g}(\mathbf{y}) = (g_1(y_1,\ldots,y_m),\ldots,g_p(y_1,\ldots,y_m)). $$

我们将链式法则应用于参数向量曲线上的实值函数。也就是说，我们考虑

$$ h_i(\mathbf{x}) = g_i(\mathbf{f}(\mathbf{x})) = g_i(f_1(x_1,\ldots,x_j,\ldots,x_d),\ldots,f_m(x_1,\ldots,x_j,\ldots,x_d)) $$

仅作为 $x_j$ 的函数，其他所有 $x_i$ 均固定。

我们得到

$$ \frac{\partial h_i(\mathbf{x}_0)}{\partial x_j} = \sum_{k=1}^m \frac{\partial g_i(\mathbf{f}(\mathbf{x}_0))} {\partial y_k} \frac{\partial f_k(\mathbf{x}_0)}{\partial x_j} $$

其中，与之前一样，符号 $\frac{\partial g_i} {\partial y_k}$ 表示 $g_i$ 对其第 $k$ 个分量的偏导数。以矩阵形式，该命题成立。 $\square$

**示例：** **(仿射映射继续)** 设 $A \in \mathbb{R}^{m \times d}$ 和 $\mathbf{b} \in \mathbb{R}^{m}$。再次定义向量值函数 $\mathbf{f} : \mathbb{R}^d \to \mathbb{R}^m$ 为 $\mathbf{f}(\mathbf{x}) = A \mathbf{x} + \mathbf{b}$。此外，对于 $C \in \mathbb{R}^{p \times m}$ 和 $\mathbf{d} \in \mathbb{R}^{p}$，定义 $\mathbf{g} : \mathbb{R}^m \to \mathbb{R}^p$ 为 $\mathbf{g}(\mathbf{y}) = C \mathbf{y} + \mathbf{d}$。

然后

$$ J_{\mathbf{g} \circ \mathbf{f}}(\mathbf{x}) = J_{\mathbf{g}}(\mathbf{f}(\mathbf{x})) \,J_{\mathbf{f}}(\mathbf{x}) = C A, $$

对于所有 $\mathbf{x} \in \mathbb{R}^d$。

这与观察结果一致

$$ \mathbf{g} \circ \mathbf{f} (\mathbf{x}) = \mathbf{g} (\mathbf{f} (\mathbf{x})) = C( A\mathbf{x} + \mathbf{b} ) + \mathbf{d} = CA \mathbf{x} + (C\mathbf{b} + \mathbf{d}). $$

$\lhd$

**示例：** 假设我们想要计算函数的梯度

$$ f(x_1, x_2) = 3 x_1² + x_2 + \exp(x_1 x_2). $$

我们可以直接应用**链式法则**，但为了说明即将出现的观点，我们将 $f$ 视为“更简单”的向量值函数的复合。具体来说，让

$$\begin{split} \mathbf{g}(x_1,x_2) = \begin{pmatrix} 3 x_1²\\ x_2\\ x_1 x_2 \end{pmatrix} \qquad h(y_1,y_2,y_3) = y_1 + y_2 + \exp(y_3). \end{split}$$

然后 $f(x_1, x_2) = h(\mathbf{g}(x_1, x_2)) = h \circ \mathbf{g}(x_1, x_2)$.

通过**链式法则**，我们可以通过首先计算 $\mathbf{g}$ 和 $h$ 的雅可比矩阵来计算 $f$ 的梯度。我们已经计算了 $\mathbf{g}$ 的雅可比矩阵

$$\begin{split} J_{\mathbf{g}}(x_1, x_2) = \begin{pmatrix} 6 x_1 & 0\\ 0 & 1\\ x_2 & x_1 \end{pmatrix}. \end{split}$$

$h$ 的雅可比矩阵为

$$ J_h(y_1, y_2, y_3) = \begin{pmatrix} \frac{\partial h(y_1, y_2, y_3)}{\partial y_1} & \frac{\partial h(y_1, y_2, y_3)}{\partial y_2} & \frac{\partial h(y_1, y_2, y_3)}{\partial y_3} \end{pmatrix} = \begin{pmatrix} 1 & 1 & \exp(y_3) \end{pmatrix}. $$

**链式法则**规定

$$\begin{align*} \nabla f(x_1, x_2)^T &= J_f(x_1, x_2)\\ &= J_h(\mathbf{g}(x_1,x_2)) \, J_{\mathbf{g}}(x_1, x_2)\\ &= \begin{pmatrix} 1 & 1 & \exp(g_3(x_1, x_2)) \end{pmatrix} \begin{pmatrix} 6 x_1 & 0\\ 0 & 1\\ x_2 & x_1 \end{pmatrix}\\ &= \begin{pmatrix} 1 & 1 & \exp(x_1 x_2) \end{pmatrix} \begin{pmatrix} 6 x_1 & 0\\ 0 & 1\\ x_2 & x_1 \end{pmatrix}\\ &= \begin{pmatrix} 6 x_1 + x_2 \exp(x_1 x_2) & 1 + x_1 \exp(x_1 x_2) \end{pmatrix}. \end{align*}$$

你可以直接检查（即，不进行复合）这确实是正确的梯度（转置）。

或者，像我们在其证明中所做的那样“展开”**链式法则**是有益的。具体来说，

$$\begin{align*} \frac{\partial f (x_1, x_2)}{\partial x_1} &= \sum_{i=1}³ \frac{\partial h(\mathbf{g}(x_1,x_2))}{\partial y_i} \frac{\partial g_i (x_1, x_2)}{\partial x_1}\\ &= \frac{\partial h(\mathbf{g}(x_1,x_2))}{\partial y_1} \frac{\partial g_1 (x_1, x_2)}{\partial x_1} + \frac{\partial h(\mathbf{g}(x_1,x_2))}{\partial y_2} \frac{\partial g_2 (x_1, x_2)}{\partial x_1} + \frac{\partial h(\mathbf{g}(x_1,x_2))}{\partial y_3} \frac{\partial g_3 (x_1, x_2)}{\partial x_1}\\ &= 1 \cdot 6x_1 + 1 \cdot 0 + \exp(g_3(x_1, x_2)) \cdot x_2\\ &= 6 x_1 + x_2 \exp(x_1 x_2). \end{align*}$$

注意，这相当于将 $J_h(\mathbf{g}(x_1,x_2))$ 乘以 $J_{\mathbf{g}}(x_1, x_2)$ 的第一列。

类似地

$$\begin{align*} \frac{\partial f (x_1, x_2)}{\partial x_2} &= \sum_{i=1}³ \frac{\partial h(\mathbf{g}(x_1,x_2))}{\partial y_i} \frac{\partial g_i (x_1, x_2)}{\partial x_2}\\ &= \frac{\partial h(\mathbf{g}(x_1,x_2))}{\partial y_1} \frac{\partial g_1 (x_1, x_2)}{\partial x_2} + \frac{\partial h(\mathbf{g}(x_1,x_2))}{\partial y_2} \frac{\partial g_2 (x_1, x_2)}{\partial x_2} + \frac{\partial h(\mathbf{g}(x_1,x_2))}{\partial y_3} \frac{\partial g_3 (x_1, x_2)}{\partial x_2}\\ &= 1 \cdot 0 + 1 \cdot 1 + \exp(g_3(x_1, x_2)) \cdot x_1\\ &= 1 + x_1 \exp(x_1 x_2). \end{align*}$$

这相当于将 $J_h(\mathbf{g}(x_1,x_2))$ 乘以 $J_{\mathbf{g}}(x_1, x_2)$ 的第二列。 $\lhd$

**CHAT & LEARN** 雅可比行列式在多元积分变量变换中有着重要的应用。请你的心仪 AI 聊天机器人解释这一应用，并提供一个使用雅可比行列式在双重积分变量变换中的例子。 $\ddagger$

## 8.2.4\. PyTorch 中自动微分简介#

我们展示了如何在 PyTorch 中使用 [自动微分](https://en.wikipedia.org/wiki/Automatic_differentiation)$\idx{自动微分}\xdi$ 来计算梯度。

引用 [维基百科](https://en.wikipedia.org/wiki/Automatic_differentiation):

> 在数学和计算机代数中，自动微分（AD），也称为算法微分或计算微分，是一组用于数值评估由计算机程序指定的函数导数的技巧。AD 利用了这样一个事实：无论计算机程序多么复杂，它都执行一系列基本的算术运算（加法、减法、乘法、除法等）和基本函数（指数、对数、正弦、余弦等）。通过将这些运算反复应用链式法则，可以自动计算任意阶的导数，精确到工作精度，并且使用的算术运算次数最多只比原始程序多一个很小的常数因子。自动微分与符号微分和数值微分（有限差分法）不同。符号微分可能导致代码效率低下，并且面临将计算机程序转换为单个表达式的困难，而数值微分可能在离散化过程中引入舍入误差和消去。

**PyTorch 中的自动微分** 我们将使用[PyTorch](https://pytorch.org/tutorials/)。它使用[tensors](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)$\idx{tensor}\xdi$，在许多方面与 NumPy 数组的行为相似。参见[这里](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)以获得快速介绍。我们首先初始化张量。在这里，每个张量对应一个单独的实变量。使用选项`requires_grad=True`，我们指示这些是以后将计算梯度的变量。我们初始化张量到将计算导数的值。如果需要在不同的值处计算导数，我们需要重复此过程。函数`[.backward()]`使用反向传播计算梯度，我们将在后面返回。偏导数可以通过`[.grad]`访问。

**数值角:** 这可以通过一个例子更好地理解。

```py
x1 = torch.tensor(1.0, requires_grad=True)
x2 = torch.tensor(2.0, requires_grad=True) 
```

我们定义函数。请注意，我们使用`torch.exp`，这是 PyTorch 实现的（逐元素）指数函数。此外，与 NumPy 一样，PyTorch 允许使用`**`来[取幂](https://pytorch.org/docs/stable/generated/torch.pow.html)。[这里](https://pytorch.org/docs/stable/name_inference.html)是 PyTorch 中张量操作列表。

```py
f = 3 * (x1 ** 2) + x2 + torch.exp(x1 * x2)

f.backward()

print(x1.grad)  # df/dx1
print(x2.grad)  # df/dx2 
```

```py
tensor(20.7781)
tensor(8.3891) 
```

输入参数也可以是向量，这允许考虑大量变量的函数。在这里，我们使用 `torch.sum` 来取参数的和。

```py
z = torch.tensor([1., 2., 3.], requires_grad=True)

g = torch.sum(z ** 2)
g.backward()

print(z.grad)  # gradient is (2 z_1, 2 z_2, 2 z_3) 
```

```py
tensor([2., 4., 6.]) 
```

这里是数据科学环境中另一个典型的例子。

```py
X = torch.randn(3, 2)  # Random dataset (features)
y = torch.tensor([[1., 0., 1.]])  # Dataset (labels)
theta = torch.ones(2, 1, requires_grad=True)  # Parameter assignment

predict = X @ theta  # Classifier with parameter vector theta
loss = torch.sum((predict - y)**2)  # Loss function
loss.backward()  # Compute gradients

print(theta.grad)  # gradient of loss 
```

```py
tensor([[29.7629],
        [31.4817]]) 
```

**CHAT & LEARN** 请你的 AI 聊天机器人解释如何使用 PyTorch 计算二阶导数（有点棘手）。请提供可以应用于先前示例的代码。 ([在 Colab 中打开](https://colab.research.google.com/github/MMiDS-textbook/MMiDS-textbook.github.io/blob/main/just_the_code/roch_mmids_chap_nn_notebook.ipynb)) $\ddagger$

$\unlhd$

**在 PyTorch 中实现梯度下降** 而不是明确指定梯度函数，我们可以使用 PyTorch 自动计算它。接下来将这样做。注意，下降更新是在 `with torch.no_grad()` 中完成的（[torch.no_grad()](https://pytorch.org/docs/stable/generated/torch.no_grad.html)），这确保更新操作本身不被跟踪用于梯度计算。在这里，输入 `x0` 以及输出 `xk.numpy(force=True)` 都是 NumPy 数组。函数 `torch.Tensor.numpy()` 将 PyTorch 张量转换为 NumPy 数组（有关 `force=True` 选项的解释，请参阅文档）。此外，引用 ChatGPT：

> 在给定的代码中，`.item()` 用于从一个张量中提取标量值。在 PyTorch 中，当你对张量执行操作时，你会得到张量作为结果，即使结果是一个单一的标量值。`.item()` 用于从张量中提取这个标量值。

```py
def gd_with_ad(f, x0, alpha=1e-3, niters=int(1e6)):
    xk = torch.tensor(x0, requires_grad=True, dtype=torch.float)

    for _ in range(niters):
        value = f(xk)
        value.backward()

        with torch.no_grad():  
            xk -= alpha * xk.grad

        xk.grad.zero_()

    return xk.numpy(force=True), f(xk).item() 
```

**数值角**：我们回顾一个先前的例子。

```py
def f(x):
    return x**3

print(gd_with_ad(f, 2, niters=int(1e4))) 
```

```py
(array(0.03277362, dtype=float32), 3.5202472645323724e-05) 
```

```py
print(gd_with_ad(f, -2, niters=100)) 
```

```py
(array(-4.9335055, dtype=float32), -120.07894897460938) 
```

$\unlhd$

**CHAT & LEARN** 简要介绍了自动微分与符号微分和数值微分不同。请你的 AI 聊天机器人详细解释这三种计算导数方法的区别。 $\ddagger$

***自我评估测验*** *(由 Claude, Gemini 和 ChatGPT 协助)*

**1** 设 $A \in \mathbb{R}^{n \times m}$ 和 $B \in \mathbb{R}^{p \times q}$。Kronecker 积 $A \otimes B$ 的维度是什么？

a) $n \times m$

b) $p \times q$

c) $np \times mq$

d) $nq \times mp$

**2** 如果 $f: \mathbb{R}^d \rightarrow \mathbb{R}^m$ 是一个连续可微的函数，那么在其定义域内部点 $x_0$ 处的雅可比矩阵 $J_f(x_0)$ 是什么？

a) 一个表示 $f$ 在 $x_0$ 处的变化率的标量。

b) 一个在 $\mathbb{R}^m$ 中的向量，表示 $f$ 在 $x_0$ 处的最速上升方向。

c) 一个 $m \times d$ 的矩阵，表示 $f$ 在 $x_0$ 处的分量函数的偏导数。

d) $f$ 在 $x_0$ 处的 Hessian 矩阵。

**3** 在链式法则的背景下，如果 $f: \mathbb{R}² \to \mathbb{R}³$ 和 $g: \mathbb{R}³ \to \mathbb{R}$，那么雅可比矩阵 $J_{g \circ f}(x)$ 的维度是多少？

a) $3 \times 2$

b) $1 \times 3$

c) $2 \times 3$

d) $1 \times 2$

**4** 让 $\mathbf{f} : D_1 \to \mathbb{R}^m$，其中 $D_1 \subseteq \mathbb{R}^d$，并且让 $\mathbf{g} : D_2 \to \mathbb{R}^p$，其中 $D_2 \subseteq \mathbb{R}^m$。假设 $\mathbf{f}$ 在 $D_1$ 的内点 $\mathbf{x}_0$ 处连续可微，且 $\mathbf{g}$ 在 $D_2$ 的内点 $\mathbf{f}(\mathbf{x}_0)$ 处连续可微。根据链式法则，以下哪个是正确的？

a) $J_{\mathbf{g} \circ \mathbf{f}}(\mathbf{x}_0) = J_{\mathbf{f}}(\mathbf{x}_0) \, J_{\mathbf{g}}(\mathbf{f}(\mathbf{x}_0))$

b) $J_{\mathbf{g} \circ \mathbf{f}}(\mathbf{x}_0) = J_{\mathbf{g}}(\mathbf{f}(\mathbf{x}_0)) \, J_{\mathbf{f}}(\mathbf{x}_0)$

c) $J_{\mathbf{g} \circ \mathbf{f}}(\mathbf{x}_0) = J_{\mathbf{f}}(\mathbf{g}(\mathbf{x}_0)) \, J_{\mathbf{g}}(\mathbf{x}_0)$

d) $J_{\mathbf{g} \circ \mathbf{f}}(\mathbf{x}_0) = J_{\mathbf{g}}(\mathbf{x}_0) \, J_{\mathbf{f}}(\mathbf{g}(\mathbf{x}_0))$

**5** 设 $A \in \mathbb{R}^{m \times d}$ 和 $\mathbf{b} \in \mathbb{R}^{m}$。定义向量值函数 $\mathbf{f} : \mathbb{R}^d \to \mathbb{R}^m$ 为 $\mathbf{f}(\mathbf{x}) = A \mathbf{x} + \mathbf{b}$。$\mathbf{f}$ 在 $\mathbf{x}_0$ 处的雅可比矩阵是什么？

a) $J_{\mathbf{f}}(\mathbf{x}_0) = A^T$

b) $J_{\mathbf{f}}(\mathbf{x}_0) = A \mathbf{x}_0 + \mathbf{b}$

c) $J_{\mathbf{f}}(\mathbf{x}_0) = A$

d) $J_{\mathbf{f}}(\mathbf{x}_0) = \mathbf{b}$

1 题的答案：c. 理由：文本定义了 Kronecker 积为一个具有 $np \times mq$ 维度的分块矩阵。

2 题的答案：c. 理由：文本定义了向量值函数的雅可比矩阵为偏导数矩阵。

3 题的答案：d. 理由：复合函数 $g \circ f$ 将 $\mathbb{R}² \to \mathbb{R}$ 映射，因此雅可比矩阵 $J_{g \circ f}(x)$ 是 $1 \times 2$。

4 题的答案：b. 理由：文本中提到：“链式法则给出了复合函数雅可比矩阵的公式。[…] 假设 $\mathbf{f}$ 在 $D_1$ 的内点 $\mathbf{x}_0$ 处连续可微，且 $\mathbf{g}$ 在 $D_2$ 的内点 $\mathbf{f}(\mathbf{x}_0)$ 处连续可微。那么

$$ J_{\mathbf{g} \circ \mathbf{f}}(\mathbf{x}_0) = J_{\mathbf{g}}(\mathbf{f}(\mathbf{x}_0)) \,J_{\mathbf{f}}(\mathbf{x}_0) $$

作为矩阵的乘积。”

5 题的答案：c. 理由：文本中提到：“设 $A \in \mathbb{R}^{m \times d}$ 和 $\mathbf{b} = (b_1,\ldots,b_m) \in \mathbb{R}^{m}$。定义向量值函数 $\mathbf{f} = (f_1, \ldots, f_m) : \mathbb{R}^d \to \mathbb{R}^m$ 为 $\mathbf{f}(\mathbf{x}) = A \mathbf{x} + \mathbf{b}$。[…] 因此 $J_{\mathbf{f}}(\mathbf{x}) = A.$” 

## 8.2.1\. 更多的矩阵代数：Hadamard 和 Kronecker 积#

首先，我们引入 Hadamard 积$\idx{Hadamard product}\xdi$ 和除法$\idx{Hadamard division}\xdi$. 两个相同维度的矩阵（或向量）的 Hadamard 积定义为元素乘积

$$ A \odot B = (a_{i,j} b_{i,j})_{i\in [n], j \in [m]}. $$

类似地，Hadamard 除法定义为元素除法

$$ A \oslash B = (a_{i,j} / b_{i,j})_{i\in [n], j \in [m]} $$

其中我们假设对于所有 $i,j$，$b_{i,j} \neq 0$。

**示例:** 作为一个说明性的例子，

$$\begin{split} \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ \end{bmatrix} \odot \begin{bmatrix} 0 & 5 \\ 6 & 7 \\ \end{bmatrix} = \begin{bmatrix} 1 \times 0 & 2 \times 5\\ 3 \times 6 & 4 \times 7\\ \end{bmatrix} = \begin{bmatrix} 0 & 10\\ 18 & 28\\ \end{bmatrix}. \end{split}$$

$\lhd$

回想一下，$\mathbf{1}$ 是全一向量，并且对于 $\mathbf{x} = (x_1,\ldots,x_n) \in \mathbb{R}^n$，$\mathrm{diag}(\mathbf{x}) \in \mathbb{R}^{n \times n}$ 是对角线元素为 $x_1,\ldots,x_n$ 的对角矩阵。

**引理** **(Hadamard 积的性质)** $\idx{Hadamard product properties}\xdi$ 设 $\mathbf{a} = (a_1,\ldots,a_n), \mathbf{b} = (b_1,\ldots,b_n), \mathbf{c} = (c_1,\ldots,c_n) \in \mathbb{R}^n$. 则以下成立：

a) $\mathrm{diag}(\mathbf{a}) \, \mathbf{b} = \mathrm{diag}(\mathbf{a} \odot \mathbf{b})$;

b) $\mathbf{a}^T(\mathbf{b} \odot \mathbf{c}) = \mathbf{1}^T(\mathbf{a} \odot \mathbf{b} \odot \mathbf{c})$

并且，假设对于所有 $i$，$a_i \neq 0$，以下也成立：

c) $\mathrm{diag}(\mathbf{a}) \, (\mathbf{b} \oslash \mathbf{a}) = \mathrm{diag}(\mathbf{b})$;

d) $\mathbf{a}^T \, (\mathbf{b} \oslash \mathbf{a}) = \mathbf{1}^T \mathbf{b}$.

$\flat$

*证明:* a) 对角矩阵与向量的乘积产生一个新的向量，其原始坐标乘以相应的对角线元素。这证明了第一个命题。

b) 我们有

$$ \mathbf{a}^T \, (\mathbf{b} \odot \mathbf{c}) = \sum_{i=1}^n a_i (b_i c_i) = \mathbf{1}^T (\mathbf{a} \odot \mathbf{b} \odot \mathbf{c}). $$

c) 和 d) 分别由 a) 和 b) 得出。

$\square$

第二，我们引入 Kronecker 积$\idx{Kronecker product}\xdi$. 设 $A = (a_{i,j})_{i \in [n], j \in [m]} \in \mathbb{R}^{n \times m}$ 和 $B = (b_{i,j})_{i \in [p], j \in [q]} \in \mathbb{R}^{p \times q}$ 是任意矩阵。它们的 Kronecker 积，记为 $A \otimes B \in \mathbb{R}^{np \times mq}$，是以下形式的矩阵

$$\begin{split} A \otimes B = \begin{pmatrix} a_{1,1} B & \cdots & a_{1,m} B \\ \vdots & \ddots & \vdots \\ a_{n,1} B & \cdots & a_{n,m} B \end{pmatrix}. \end{split}$$

**示例:** 这里是一个来自 [维基百科](https://en.wikipedia.org/wiki/Kronecker_product#Examples) 的简单说明性例子：

$$\begin{split} \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ \end{bmatrix} \otimes \begin{bmatrix} 0 & 5 \\ 6 & 7 \\ \end{bmatrix} = \begin{bmatrix} 1 \begin{bmatrix} 0 & 5 \\ 6 & 7 \\ \end{bmatrix} & 2 \begin{bmatrix} 0 & 5 \\ 6 & 7 \\ \end{bmatrix} \\ 3 \begin{bmatrix} 0 & 5 \\ 6 & 7 \\ \end{bmatrix} & 4 \begin{bmatrix} 0 & 5 \\ 6 & 7 \\ \end{bmatrix} \\ \end{bmatrix} = \begin{bmatrix} 0 & 5 & 0 & 10 \\ 6 & 7 & 12 & 14 \\ 0 & 15 & 0 & 20 \\ 18 & 21 & 24 & 28 \end{bmatrix}. \end{split}$$

$\lhd$

**例题** **(外积)** $\idx{外积}\xdi$ 这里是另一个我们之前遇到的例子，两个向量 $\mathbf{u} = (u_1,\ldots,u_n) \in \mathbb{R}^n$ 和 $\mathbf{v} = (v_1,\ldots, v_m) \in \mathbb{R}^m$ 的外积。回忆一下，外积以块形式定义为 $n \times m$ 矩阵

$$ \mathbf{u} \mathbf{v}^T = \begin{pmatrix} v_1 \mathbf{u} & \cdots & v_m \mathbf{u} \end{pmatrix} = \mathbf{v}^T \otimes \mathbf{u}. $$

相当于，

$$\begin{split} \mathbf{u} \mathbf{v}^T = \begin{pmatrix} u_1 \mathbf{v}^T\\ \vdots\\ u_n \mathbf{v}^T \end{pmatrix} = \mathbf{u} \otimes \mathbf{v}^T. \end{split}$$

$\lhd$

**例题** **(继续)** 在前面的例子中，克罗内克积是可交换的（即，我们有 $\mathbf{v}^T \otimes \mathbf{u} = \mathbf{u} \otimes \mathbf{v}^T$）。在一般情况下并不总是这样。回到上面的第一个例子，注意

$$\begin{split} \begin{bmatrix} 0 & 5 \\ 6 & 7 \\ \end{bmatrix} \otimes \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ \end{bmatrix} = \begin{bmatrix} 0 \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ \end{bmatrix} & 5 \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ \end{bmatrix} \\ 6 \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ \end{bmatrix} & 7 \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ \end{bmatrix} \\ \end{bmatrix} = \begin{bmatrix} 0 & 0 & 5 & 10 \\ 0 & 0 & 15 & 20 \\ 6 & 12 & 7 & 14 \\ 18 & 24 & 21 & 28 \end{bmatrix}. \end{split}$$

你可以检查这和我们之前相反顺序得到的结果是不同的。$\lhd$

以下引理的证明留给读者作为练习。

**引理** **(克罗内克积的性质)** $\idx{克罗内克积的性质}\xdi$ 克罗内克积具有以下性质：

a) 如果 $B, C$ 是相同维度的矩阵

$$ A \otimes (B + C) = A \otimes B + A \otimes C \quad \text{和}\quad (B + C) \otimes A = B \otimes A + C \otimes A. $$

b) 如果 $A, B, C, D$ 是可以形成矩阵乘积 $AC$ 和 $BD$ 的矩阵，那么

$$ (A \otimes B)\,(C \otimes D) = (AC) \otimes (BD). $$

c) 如果 $A, C$ 是相同维度的矩阵，且 $B, D$ 是相同维度的矩阵，那么

$$ (A \otimes B)\odot(C \otimes D) = (A\odot C) \otimes (B\odot D). $$

d) 如果 $A,B$ 是可逆的，那么

$$ (A \otimes B)^{-1} = A^{-1} \otimes B^{-1}. $$

e) $A \otimes B$ 的转置是

$$ (A \otimes B)^T = A^T \otimes B^T. $$

f) 如果 $\mathbf{u}$ 是一个列向量，且 $A, B$ 是可以形成矩阵乘积 $AB$ 的矩阵，那么

$$ (\mathbf{u} \otimes A) B = \mathbf{u} \otimes (AB) \quad\text{和}\quad (A \otimes \mathbf{u}) B = (AB) \otimes \mathbf{u}. $$

同样地，

$$ A \,(\mathbf{u}^T \otimes B) = \mathbf{u}^T \otimes (AB) \quad\text{和}\quad A \,(B \otimes \mathbf{u}^T) = (AB) \otimes \mathbf{u}^T. $$

$\flat$

## 8.2.2\. 雅可比矩阵#

回想一下，实变函数的导数是函数相对于变量变化的改变率。另一种说法是 $f'(x)$ 是 $f$ 在 $x$ 处的切线斜率。形式上，可以在 $x$ 的邻域内通过以下方式近似 $f(x)$

$$ f(x + h) = f(x) + f'(x) h + r(h), $$

其中 $r(h)$ 与 $h$ 相比可以忽略不计

$$ \lim_{h\to 0} \frac{r(h)}{h} = 0. $$

事实上，定义

$$ r(h) = f(x + h) - f(x) - f'(x) h. $$

然后根据导数的定义

$$ \lim_{h\to 0} \frac{r(h)}{h} = \lim_{h\to 0} \frac{f(x + h) - f(x) - f'(x) h}{h} = \lim_{h\to 0} \left[\frac{f(x + h) - f(x)}{h} - f'(x) \right] = 0. $$

对于向量值函数，我们有以下推广。设 $\mathbf{f} = (f_1, \ldots, f_m) : D \to \mathbb{R}^m$ 其中 $D \subseteq \mathbb{R}^d$，并且设 $\mathbf{x} \in D$ 是 $D$ 的一个内部点。我们说 $\mathbf{f}$ 在 $\mathbf{x}$ 处是可微的$\idx{可微}\xdi$，如果存在一个矩阵 $A \in \mathbb{R}^{m \times d}$ 使得

$$ \mathbf{f}(\mathbf{x}+\mathbf{h}) = \mathbf{f}(\mathbf{x}) + A \mathbf{h} + \mathbf{r}(\mathbf{h}) $$

其中

$$ \lim_{\mathbf{h} \to 0} \frac{\|\mathbf{r}(\mathbf{h})\|_2}{\|\mathbf{h}\|_2} = 0. $$

矩阵 $\mathbf{f}'(\mathbf{x}) = A$ 被称为 $\mathbf{f}$ 在 $\mathbf{x}$ 处的微分$\idx{微分}\xdi$，我们看到仿射映射 $\mathbf{f}(\mathbf{x}) + A \mathbf{h}$ 提供了在 $\mathbf{x}$ 的邻域内对 $\mathbf{f}$ 的近似。

我们在这里不会推导完整的理论。在 $\mathbf{f}$ 的每个分量在 $\mathbf{x}$ 的邻域内具有连续偏导数的情形下，微分存在，并且等于下一个定义的雅可比矩阵。

**定义** **(雅可比矩阵)** $\idx{雅可比}\xdi$ 设 $\mathbf{f} = (f_1, \ldots, f_m) : D \to \mathbb{R}^m$ 其中 $D \subseteq \mathbb{R}^d$，并且设 $\mathbf{x}_0 \in D$ 是 $D$ 的一个内部点，其中对于所有 $i, j$，$\frac{\partial f_j (\mathbf{x}_0)}{\partial x_i}$ 都存在。在 $\mathbf{x}_0$ 处 $\mathbf{f}$ 的雅可比矩阵是一个 $m \times d$ 的矩阵

$$\begin{split} J_{\mathbf{f}}(\mathbf{x}_0) = \begin{pmatrix} \frac{\partial f_1 (\mathbf{x}_0)}{\partial x_1} & \ldots & \frac{\partial f_1 (\mathbf{x}_0)}{\partial x_d}\\ \vdots & \ddots & \vdots\\ \frac{\partial f_m (\mathbf{x}_0)}{\partial x_1} & \ldots & \frac{\partial f_m (\mathbf{x}_0)}{\partial x_d} \end{pmatrix}. \end{split}$$

$\natural$

**定理** **(微分和雅可比)** $\idx{微分和雅可比定理}\xdi$ 设 $\mathbf{f} = (f_1, \ldots, f_m) : D \to \mathbb{R}^m$ 其中 $D \subseteq \mathbb{R}^d$，并且设 $\mathbf{x}_0 \in D$ 是 $D$ 的一个内点。假设对于所有 $i, j$，$\frac{\partial f_j (\mathbf{x}_0)}{\partial x_i}$ 存在且在 $\mathbf{x}_0$ 附近的某个开球内是连续的。那么在 $\mathbf{x}_0$ 处的微分等于 $\mathbf{f}$ 在 $\mathbf{x}_0$ 处的雅可比矩阵。 $\sharp$

回想一下，对于任何 $A, B$，当 $AB$ 被良好定义时，它都满足 $\|A B \|_F \leq \|A\|_F \|B\|_F$。这特别适用于 $B$ 是一个列向量时，在这种情况下，$\|B\|_F$ 是其欧几里得范数。

*证明* 通过**平均值定理**，对于每个 $i$，存在 $\xi_{\mathbf{h},i} \in (0,1)$ 使得

$$ f_i(\mathbf{x}_0+\mathbf{h}) = f_i(\mathbf{x}_0) + \nabla f_i(\mathbf{x}_0 + \xi_{\mathbf{h},i} \mathbf{h})^T \mathbf{h}. $$

定义

$$\begin{split} \tilde{J}(\mathbf{h}) = \begin{pmatrix} \frac{\partial f_1 (\mathbf{x}_0 + \xi_{\mathbf{h},1} \mathbf{h})}{\partial x_1} & \ldots & \frac{\partial f_1 (\mathbf{x}_0 + \xi_{\mathbf{h},1} \mathbf{h})}{\partial x_d}\\ \vdots & \ddots & \vdots\\ \frac{\partial f_m (\mathbf{x}_0 + \xi_{\mathbf{h},m} \mathbf{h})}{\partial x_1} & \ldots & \frac{\partial f_m (\mathbf{x}_0 + \xi_{\mathbf{h},m} \mathbf{h})}{\partial x_d} \end{pmatrix}. \end{split}$$

因此我们有

$$ \mathbf{f}(\mathbf{x}_0+\mathbf{h}) - \mathbf{f}(\mathbf{x}_0) - J_{\mathbf{f}}(\mathbf{x}_0)\,\mathbf{h} = \tilde{J}(\mathbf{h}) \,\mathbf{h} - J_{\mathbf{f}}(\mathbf{x}_0)\,\mathbf{h} = \left(\tilde{J}(\mathbf{h}) - J_{\mathbf{f}}(\mathbf{x}_0)\right)\,\mathbf{h}. $$

当 $\mathbf{h}$ 趋向于 $0$ 时取极限，我们得到

$$\begin{align*} \lim_{\mathbf{h} \to 0} \frac{\|\mathbf{f}(\mathbf{x}_0+\mathbf{h}) - \mathbf{f}(\mathbf{x}_0) - J_{\mathbf{f}}(\mathbf{x}_0)\,\mathbf{h}\|_2}{\|\mathbf{h}\|_2} &= \lim_{\mathbf{h} \to 0} \frac{\|\left(\tilde{J}(\mathbf{h}) - J_{\mathbf{f}}(\mathbf{x}_0)\right)\,\mathbf{h}\|_2}{\|\mathbf{h}\|_2}\\ &\leq \lim_{\mathbf{h} \to 0} \frac{\|\tilde{J}(\mathbf{h}) - J_{\mathbf{f}}(\mathbf{x}_0)\|_F \|\mathbf{h}\|_2}{\|\mathbf{h}\|_2}\\ &= 0, \end{align*}$$

通过偏导数的连续性。 $\square$

**示例**：一个向量值函数的例子是

$$\begin{split} \mathbf{g}(x_1,x_2) = \begin{pmatrix} g_1(x_1,x_2)\\ g_2(x_1,x_2)\\ g_3(x_1,x_2) \end{pmatrix} = \begin{pmatrix} 3 x_1²\\ x_2\\ x_1 x_2 \end{pmatrix}. \end{split}$$

它的雅可比矩阵是

$$\begin{split} J_{\mathbf{g}}(x_1, x_2) = \begin{pmatrix} \frac{\partial g_1 (x_1, x_2)}{\partial x_1} & \frac{\partial g_1 (x_1, x_2)}{\partial x_2}\\ \frac{\partial g_2 (x_1, x_2)}{\partial x_1} & \frac{\partial g_2 (x_1, x_2)}{\partial x_2}\\ \frac{\partial g_3 (x_1, x_2)}{\partial x_1} & \frac{\partial g_3 (x_1, x_2)}{\partial x_2} \end{pmatrix} = \begin{pmatrix} 6 x_1 & 0\\ 0 & 1\\ x_2 & x_1 \end{pmatrix}. \end{split}$$

$\lhd$

**示例：** **（梯度与雅可比矩阵）** 对于一个连续可微的实值函数 $f : D \to \mathbb{R}$，雅可比矩阵简化为行向量

$$ J_{f}(\mathbf{x}_0) = \left(\frac{\partial f (\mathbf{x}_0)}{\partial x_1}, \ldots, \frac{\partial f (\mathbf{x}_0)}{\partial x_d}\right)^T = \nabla f(\mathbf{x}_0)^T $$

其中 $\nabla f(\mathbf{x}_0)$ 是 $f$ 在 $\mathbf{x}_0$ 处的梯度。$\lhd$

**示例：** **（Hessian 与雅可比矩阵）** 对于一个二阶连续可微的实值函数 $f : D \to \mathbb{R}$，其梯度的雅可比矩阵为

$$\begin{split} J_{\nabla f}(\mathbf{x}_0) = \begin{pmatrix} \frac{\partial² f(\mathbf{x}_0)}{\partial x_1²} & \cdots & \frac{\partial² f(\mathbf{x}_0)}{\partial x_d \partial x_1}\\ \vdots & \ddots & \vdots\\ \frac{\partial² f(\mathbf{x}_0)}{\partial x_1 \partial x_d} & \cdots & \frac{\partial² f(\mathbf{x}_0)}{\partial x_d²} \end{pmatrix}, \end{split}$$

即，$f$ 在 $\mathbf{x}_0$ 处的 Hessian（转置，但这没有关系；为什么？）。$\lhd$

**示例：** **（参数曲线与雅可比矩阵）** 考虑参数曲线 $\mathbf{g}(t) = (g_1(t), \ldots, g_d(t)) \in \mathbb{R}^d$，其中 $t$ 在 $\mathbb{R}$ 的某个闭区间内。假设 $\mathbf{g}(t)$ 在 $t$ 处连续可微，即其每个分量都是。

然后

$$\begin{split} J_{\mathbf{g}}(t) = \begin{pmatrix} g_1'(t)\\ \vdots\\ g_m'(t) \end{pmatrix} = \mathbf{g}'(t). \end{split}$$

$\lhd$

**示例：** **（仿射映射）** 设 $A = (a_{i,j})_{i,j} \in \mathbb{R}^{m \times d}$ 和 $\mathbf{b} = (b_1,\ldots,b_m) \in \mathbb{R}^{m}$。定义向量值函数 $\mathbf{f} = (f_1, \ldots, f_m) : \mathbb{R}^d \to \mathbb{R}^m$ 为

$$ \mathbf{f}(\mathbf{x}) = A \mathbf{x} + \mathbf{b}. $$

这是一个仿射映射。特别地，在线性映射的情况下，当 $\mathbf{b} = \mathbf{0}$ 时，

$$ \mathbf{f}(\mathbf{x} + \mathbf{y}) = A(\mathbf{x} + \mathbf{y}) = A\mathbf{x} + A\mathbf{y} = \mathbf{f}(\mathbf{x}) + \mathbf{f}(\mathbf{y}). $$

用 $\boldsymbol{\alpha}_1^T,\ldots,\boldsymbol{\alpha}_m^T$ 表示 $A$ 的行。

我们计算 $\mathbf{f}$ 在 $\mathbf{x}$ 处的雅可比矩阵。注意，

$$\begin{align*} \frac{\partial f_i (\mathbf{x})}{\partial x_j} &= \frac{\partial}{\partial x_j}[\boldsymbol{\alpha}_i^T \mathbf{x} + b_i]\\ &= \frac{\partial}{\partial x_j}\left[\sum_{\ell=1}^m a_{i,\ell} x_{\ell} + b_i\right]\\ &= a_{i,j}. \end{align*}$$

所以

$$ J_{\mathbf{f}}(\mathbf{x}) = A. $$

$\lhd$

以下重要示例是雅可比矩阵的一个不那么直接的应用。

将矩阵 $A = (a_{i,j})_{i,j} \in \mathbb{R}^{n \times m}$ 的向量化 $\mathrm{vec}(A) \in \mathbb{R}^{nm}$ 介绍为向量将是有用的。

$$ \mathrm{vec}(A) = (a_{1,1},\ldots,a_{n,1},a_{1,2},\ldots,a_{n,2},\ldots,a_{1,m},\ldots,a_{n,m}). $$

那就是通过将 $A$ 的列堆叠在一起得到的。

**示例：** **(关于其矩阵的线性映射的雅可比矩阵)** 我们对前面的例子采取不同的方法。在数据科学应用中，计算线性映射 $X \mathbf{z}$ 的雅可比矩阵——相对于矩阵 $X \in \mathbb{R}^{n \times m}$ 将是有用的。具体来说，对于固定的 $\mathbf{z} \in \mathbb{R}^{m}$，令 $(\mathbf{x}^{(i)})^T$ 为 $X$ 的第 $i$ 行，我们定义函数

$$\begin{split} \mathbf{f}(\mathbf{x}) = X \mathbf{z} = \begin{pmatrix} (\mathbf{x}^{(1)})^T\\ \vdots\\ (\mathbf{x}^{(n)})^T \end{pmatrix} \mathbf{z} = \begin{pmatrix} (\mathbf{x}^{(1)})^T \mathbf{z} \\ \vdots\\ (\mathbf{x}^{(n)})^T \mathbf{z} \end{pmatrix} \end{split}$$

其中 $\mathbf{x} = \mathrm{vec}(X^T) = (\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)})$.

为了计算雅可比矩阵，让我们看看与 $\mathbf{x}^{(k)}$ 中的变量相对应的列，即列 $\alpha_k = (k-1) m + 1$ 到 $\beta_k = k m$。注意，只有 $\mathbf{f}$ 的第 $k$ 个分量依赖于 $\mathbf{x}^{(k)}$，所以 $J_{\mathbf{f}}(\mathbf{x})$ 的行 $\neq k$ 对应的列是 $0$。

另一方面，行 $k$ 是我们之前公式中关于仿射映射梯度的公式中的 $\mathbf{z}^T$。因此，一种写 $J_{\mathbf{f}}(\mathbf{x})$ 的列 $\alpha_k$ 到 $\beta_k$ 的方法是 $\mathbf{e}_k \mathbf{z}^T$，其中这里的 $\mathbf{e}_k \in \mathbb{R}^{n}$ 是 $\mathbb{R}^{n}$ 的第 $k$ 个标准基向量。

因此 $J_{\mathbf{f}}(\mathbf{x})$ 可以写成块形式。

$$ J_{\mathbf{f}}(\mathbf{x}) = \begin{pmatrix} \mathbf{e}_1 \mathbf{z}^T & \cdots & \mathbf{e}_{n}\mathbf{z}^T \end{pmatrix} = I_{n\times n} \otimes \mathbf{z}^T =: \mathbb{B}_{n}[\mathbf{z}], $$

其中最后一个等式是一个定义。 $\lhd$

我们还需要一个额外的技巧。

**示例：** **(关于输入和矩阵的线性映射的雅可比矩阵)** 再次考虑线性映射 $X \mathbf{z}$——这次作为矩阵 $X \in \mathbb{R}^{n \times m}$ 和向量 $\mathbf{z} \in \mathbb{R}^{m}$ 的函数。也就是说，再次令 $(\mathbf{x}^{(i)})^T$ 为 $X$ 的第 $i$ 行，我们定义函数

$$\begin{split} \mathbf{g}(\mathbf{z}, \mathbf{x}) = X \mathbf{z} = \begin{pmatrix} (\mathbf{x}^{(1)})^T\\ \vdots\\ (\mathbf{x}^{(n)})^T \end{pmatrix} \mathbf{z} = \begin{pmatrix} (\mathbf{x}^{(1)})^T \mathbf{z} \\ \vdots\\ (\mathbf{x}^{(n)})^T \mathbf{z} \end{pmatrix} \end{split}$$

其中，如前所述 $\mathbf{x} = \mathrm{vec}(X^T) = (\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)})$。

要计算雅可比矩阵，我们将其视为一个分块矩阵，并使用前两个例子。$J_{\mathbf{f}}(\mathbf{z}, \mathbf{x})$ 对应于 $\mathbf{z}$ 中的变量的列，即 $1$ 到 $m$ 的列，是

$$\begin{split} X = \begin{pmatrix} (\mathbf{x}^{(1)})^T\\ \vdots\\ (\mathbf{x}^{(n)})^T \end{pmatrix} =: \mathbb{A}_{n}[\mathbf{x}]. \end{split}$$

$J_{\mathbf{f}}(\mathbf{z}, \mathbf{x})$ 对应于 $\mathbf{x}$ 中的变量的列，即 $m + 1$ 到 $m + nm$ 的列，是矩阵 $\mathbb{B}_{n}[\mathbf{z}]$。注意，在 $\mathbb{A}_{n}[\mathbf{x}]$ 和 $\mathbb{B}_{n}[\mathbf{z}]$ 中，下标 $n$ 表示矩阵的行数。列数由 $n$ 和输入向量的尺寸决定：

+   $\mathbb{A}_{n}[\mathbf{x}]$ 的长度除以 $n$；

+   $\mathbb{B}_{n}[\mathbf{z}]$ 的长度乘以 $n$。

因此，$J_{\mathbf{f}}(\mathbf{z}, \mathbf{x})$ 可以写成分块形式

$$ J_{\mathbf{f}}(\mathbf{z}, \mathbf{x}) = \begin{pmatrix} \mathbb{A}_{n}[\mathbf{x}] & \mathbb{B}_{n}[\mathbf{z}] \end{pmatrix}. $$

$\lhd$

**示例：** **（逐元素函数）** 设 $f : D \to \mathbb{R}$，其中 $D \subseteq \mathbb{R}$，是一个关于单变量的连续可微实值函数。对于 $n \geq 2$，考虑将 $f$ 应用到向量 $\mathbf{x} \in \mathbb{R}^n$ 的每个元素上，即设 $\mathbf{f} : D^n \to \mathbb{R}^n$，使得

$$ \mathbf{f}(\mathbf{x}) = (f_1(\mathbf{x}), \ldots, f_n(\mathbf{x})) = (f(x_1), \ldots, f(x_n)). $$

$\mathbf{f}$ 的雅可比矩阵可以从单变量情况的导数 $f'$ 计算得出。实际上，让 $\mathbf{x} = (x_1,\ldots,x_n)$ 使得对于所有 $i$，$x_i$ 是 $D$ 的内点，

$$ \frac{\partial f_j(\mathbf{x})}{\partial x_j} = f'(x_j), $$

当 $\ell \neq j$

$$ \frac{\partial f_\ell(\mathbf{x})}{\partial x_j} =0, $$

因为 $f_\ell(\mathbf{x})$ 实际上不依赖于 $x_j$。换句话说，雅可比矩阵的第 $j$ 列是 $f'(x_j) \,\mathbf{e}_j$，其中再次 $\mathbf{e}_{j}$ 是 $\mathbb{R}^{n}$ 中的第 $j$ 个标准基向量。

因此，$J_{\mathbf{f}}(\mathbf{x})$ 是对角矩阵，其对角元素是 $f'(x_j)$，$j=1, \ldots, n$，我们用

$$ J_{\mathbf{f}}(\mathbf{x}) = \mathrm{diag}(f'(x_1),\ldots,f'(x_n)). $$

$\lhd$

## 8.2.3\. 链式法则的推广#

正如我们所看到的，函数通常是通过更简单函数的组合得到的。我们将使用向量表示法 $\mathbf{h} = \mathbf{g} \circ \mathbf{f}$ 来表示函数 $\mathbf{h}(\mathbf{x}) = \mathbf{g} (\mathbf{f} (\mathbf{x}))$。

**引理** **（连续函数的复合）** $\idx{连续函数复合引理}\xdi$ 设 $\mathbf{f} : D_1 \to \mathbb{R}^m$，其中 $D_1 \subseteq \mathbb{R}^d$，设 $\mathbf{g} : D_2 \to \mathbb{R}^p$，其中 $D_2 \subseteq \mathbb{R}^m$。假设 $\mathbf{f}$ 在 $\mathbf{x}_0$ 处连续，且 $\mathbf{g}$ 在 $\mathbf{f}(\mathbf{x}_0)$ 处连续。那么 $\mathbf{g} \circ \mathbf{f}$ 在 $\mathbf{x}_0$ 处连续。 $\flat$

**链式法则** 给出了复合函数雅可比矩阵的公式。

**定理** **（链式法则）** $\idx{链式法则}\xdi$ 设 $\mathbf{f} : D_1 \to \mathbb{R}^m$，其中 $D_1 \subseteq \mathbb{R}^d$，设 $\mathbf{g} : D_2 \to \mathbb{R}^p$，其中 $D_2 \subseteq \mathbb{R}^m$。假设 $\mathbf{f}$ 在 $D_1$ 的内点 $\mathbf{x}_0$ 处可连续微分，且 $\mathbf{g}$ 在 $D_2$ 的内点 $\mathbf{f}(\mathbf{x}_0)$ 处可连续微分。那么

$$ J_{\mathbf{g} \circ \mathbf{f}}(\mathbf{x}_0) = J_{\mathbf{g}}(\mathbf{f}(\mathbf{x}_0)) \,J_{\mathbf{f}}(\mathbf{x}_0) $$

作为矩阵的乘积。 $\sharp$

直观上，雅可比矩阵提供了函数在一点邻域内的线性近似。线性映射的复合对应于相关矩阵的乘积。同样，复合函数的雅可比矩阵是雅可比矩阵的乘积。

**证明：** 为了避免混淆，我们考虑 $\mathbf{f}$ 和 $\mathbf{g}$ 是变量名称不同的函数，具体来说，

$$ \mathbf{f}(\mathbf{x}) = (f_1(x_1,\ldots,x_d),\ldots,f_m(x_1,\ldots,x_d)) $$

和

$$ \mathbf{g}(\mathbf{y}) = (g_1(y_1,\ldots,y_m),\ldots,g_p(y_1,\ldots,y_m)). $$

我们应用实值函数在参数向量曲线上的**链式法则**。也就是说，我们考虑

$$ h_i(\mathbf{x}) = g_i(\mathbf{f}(\mathbf{x})) = g_i(f_1(x_1,\ldots,x_j,\ldots,x_d),\ldots,f_m(x_1,\ldots,x_j,\ldots,x_d)) $$

仅作为 $x_j$ 的函数，其他所有 $x_i$ 均固定。

我们得到

$$ \frac{\partial h_i(\mathbf{x}_0)}{\partial x_j} = \sum_{k=1}^m \frac{\partial g_i(\mathbf{f}(\mathbf{x}_0))} {\partial y_k} \frac{\partial f_k(\mathbf{x}_0)}{\partial x_j} $$

其中，与之前一样，符号 $\frac{\partial g_i} {\partial y_k}$ 表示 $g_i$ 对其第 $k$ 个分量的偏导数。在矩阵形式中，该命题成立。 $\square$

**例：** **（仿射映射继续）** 设 $A \in \mathbb{R}^{m \times d}$ 和 $\mathbf{b} \in \mathbb{R}^{m}$。再次定义向量值函数 $\mathbf{f} : \mathbb{R}^d \to \mathbb{R}^m$ 为 $\mathbf{f}(\mathbf{x}) = A \mathbf{x} + \mathbf{b}$。此外，对于 $C \in \mathbb{R}^{p \times m}$ 和 $\mathbf{d} \in \mathbb{R}^{p}$，定义 $\mathbf{g} : \mathbb{R}^m \to \mathbb{R}^p$ 为 $\mathbf{g}(\mathbf{y}) = C \mathbf{y} + \mathbf{d}$。

然后

$$ J_{\mathbf{g} \circ \mathbf{f}}(\mathbf{x}) = J_{\mathbf{g}}(\mathbf{f}(\mathbf{x})) \,J_{\mathbf{f}}(\mathbf{x}) = C A, $$

对于所有 $\mathbf{x} \in \mathbb{R}^d$.

这与观察一致

$$ \mathbf{g} \circ \mathbf{f} (\mathbf{x}) = \mathbf{g} (\mathbf{f} (\mathbf{x})) = C( A\mathbf{x} + \mathbf{b} ) + \mathbf{d} = CA \mathbf{x} + (C\mathbf{b} + \mathbf{d}). $$

$\lhd$

**示例：** 假设我们想要计算函数的梯度

$$ f(x_1, x_2) = 3 x_1² + x_2 + \exp(x_1 x_2). $$

我们可以直接应用链式法则，但为了说明即将出现的观点，我们将 $f$ 视为“更简单”的向量值函数的组合。具体来说，设

$$\begin{split} \mathbf{g}(x_1,x_2) = \begin{pmatrix} 3 x_1²\\ x_2\\ x_1 x_2 \end{pmatrix} \qquad h(y_1,y_2,y_3) = y_1 + y_2 + \exp(y_3). \end{split}$$

然后 $f(x_1, x_2) = h(\mathbf{g}(x_1, x_2)) = h \circ \mathbf{g}(x_1, x_2)$.

根据链式法则，我们可以通过首先计算 $\mathbf{g}$ 和 $h$ 的雅可比矩阵来计算 $f$ 的梯度。我们已计算了 $\mathbf{g}$ 的雅可比矩阵

$$\begin{split} J_{\mathbf{g}}(x_1, x_2) = \begin{pmatrix} 6 x_1 & 0\\ 0 & 1\\ x_2 & x_1 \end{pmatrix}. \end{split}$$

$h$ 的雅可比矩阵是

$$ J_h(y_1, y_2, y_3) = \begin{pmatrix} \frac{\partial h(y_1, y_2, y_3)}{\partial y_1} & \frac{\partial h(y_1, y_2, y_3)}{\partial y_2} & \frac{\partial h(y_1, y_2, y_3)}{\partial y_3} \end{pmatrix} = \begin{pmatrix} 1 & 1 & \exp(y_3) \end{pmatrix}. $$

链式法则规定

$$\begin{align*} \nabla f(x_1, x_2)^T &= J_f(x_1, x_2)\\ &= J_h(\mathbf{g}(x_1,x_2)) \, J_{\mathbf{g}}(x_1, x_2)\\ &= \begin{pmatrix} 1 & 1 & \exp(g_3(x_1, x_2)) \end{pmatrix} \begin{pmatrix} 6 x_1 & 0\\ 0 & 1\\ x_2 & x_1 \end{pmatrix}\\ &= \begin{pmatrix} 1 & 1 & \exp(x_1 x_2) \end{pmatrix} \begin{pmatrix} 6 x_1 & 0\\ 0 & 1\\ x_2 & x_1 \end{pmatrix}\\ &= \begin{pmatrix} 6 x_1 + x_2 \exp(x_1 x_2) & 1 + x_1 \exp(x_1 x_2) \end{pmatrix}. \end{align*}$$

您可以直接检查（即，无需组合）这确实是正确的梯度（转置）。

或者，像我们在其证明中所做的那样，“展开”链式法则是有益的。具体来说，

$$\begin{align*} \frac{\partial f (x_1, x_2)}{\partial x_1} &= \sum_{i=1}³ \frac{\partial h(\mathbf{g}(x_1,x_2))}{\partial y_i} \frac{\partial g_i (x_1, x_2)}{\partial x_1}\\ &= \frac{\partial h(\mathbf{g}(x_1,x_2))}{\partial y_1} \frac{\partial g_1 (x_1, x_2)}{\partial x_1} + \frac{\partial h(\mathbf{g}(x_1,x_2))}{\partial y_2} \frac{\partial g_2 (x_1, x_2)}{\partial x_1} + \frac{\partial h(\mathbf{g}(x_1,x_2))}{\partial y_3} \frac{\partial g_3 (x_1, x_2)}{\partial x_1}\\ &= 1 \cdot 6x_1 + 1 \cdot 0 + \exp(g_3(x_1, x_2)) \cdot x_2\\ &= 6 x_1 + x_2 \exp(x_1 x_2). \end{align*}$$

注意这对应于将 $J_h(\mathbf{g}(x_1,x_2))$ 乘以 $J_{\mathbf{g}}(x_1, x_2)$ 的第一列。

同样

$$\begin{align*} \frac{\partial f (x_1, x_2)}{\partial x_2} &= \sum_{i=1}³ \frac{\partial h(\mathbf{g}(x_1,x_2))}{\partial y_i} \frac{\partial g_i (x_1, x_2)}{\partial x_2}\\ &= \frac{\partial h(\mathbf{g}(x_1,x_2))}{\partial y_1} \frac{\partial g_1 (x_1, x_2)}{\partial x_2} + \frac{\partial h(\mathbf{g}(x_1,x_2))}{\partial y_2} \frac{\partial g_2 (x_1, x_2)}{\partial x_2} + \frac{\partial h(\mathbf{g}(x_1,x_2))}{\partial y_3} \frac{\partial g_3 (x_1, x_2)}{\partial x_2}\\ &= 1 \cdot 0 + 1 \cdot 1 + \exp(g_3(x_1, x_2)) \cdot x_1\\ &= 1 + x_1 \exp(x_1 x_2). \end{align*}$$

这相当于将 $J_h(\mathbf{g}(x_1,x_2))$ 乘以 $J_{\mathbf{g}}(x_1, x_2)$ 的第二列。 $\lhd$

**CHAT & LEARN** 雅可比行列式在多元积分变量变换中有着重要的应用。请你的心仪 AI 聊天机器人解释这一应用，并提供一个使用雅可比行列式在双重积分变量变换中的例子。 $\ddagger$

## 8.2.4\. PyTorch 中自动微分的简要介绍#

我们展示了在 PyTorch 中使用 [自动微分](https://en.wikipedia.org/wiki/Automatic_differentiation)$\idx{自动微分}\xdi$ 来计算梯度的方法。

引用 [维基百科](https://en.wikipedia.org/wiki/Automatic_differentiation):

> 在数学和计算机代数中，自动微分（AD），也称为算法微分或计算微分，是一组用于数值评估由计算机程序指定的函数导数的技巧。AD 利用这样一个事实：无论计算机程序多么复杂，它都执行一系列基本的算术运算（加法、减法、乘法、除法等）和基本函数（指数、对数、正弦、余弦等）。通过将这些运算反复应用链式法则，可以自动计算任意阶的导数，精确到工作精度，并且使用的算术运算次数最多只比原始程序多一个很小的常数因子。自动微分与符号微分和数值微分（有限差分法）不同。符号微分可能导致代码效率低下，并且面临将计算机程序转换为单个表达式的困难，而数值微分可能在离散化过程中引入舍入误差和消去。

**PyTorch 中的自动微分** 我们将使用 [PyTorch](https://pytorch.org/tutorials/)。它使用 [张量](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)$\idx{tensor}\xdi$，在许多方面与 NumPy 数组的行为相似。见[这里](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)的快速介绍。我们首先初始化张量。这里每个张量对应一个单独的实变量。通过选项`[`requires_grad=True`](https://pytorch.org/docs/stable/generated/torch.Tensor.requires_grad.html#torch.Tensor.requires_grad)，我们表明这些是之后将计算梯度的变量。我们初始化张量到将计算导数的值。如果需要在不同的值处计算导数，我们需要重复此过程。函数`[``.backward()`](https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html)`使用反向传播计算梯度，我们稍后会回到这一点。偏导数可以通过`[``.grad`](https://pytorch.org/docs/stable/generated/torch.Tensor.grad.html)`访问。

**数值角**: 这通过一个例子更容易理解。

```py
x1 = torch.tensor(1.0, requires_grad=True)
x2 = torch.tensor(2.0, requires_grad=True) 
```

我们定义了函数。请注意，我们使用了`[`torch.exp`](https://pytorch.org/docs/stable/generated/torch.exp.html)，这是 PyTorch 对（逐元素）指数函数的实现。此外，与 NumPy 类似，PyTorch 允许使用 `**` 来进行[取幂](https://pytorch.org/docs/stable/generated/torch.pow.html)。[这里](https://pytorch.org/docs/stable/name_inference.html)是 PyTorch 中张量操作的列表。

```py
f = 3 * (x1 ** 2) + x2 + torch.exp(x1 * x2)

f.backward()

print(x1.grad)  # df/dx1
print(x2.grad)  # df/dx2 
```

```py
tensor(20.7781)
tensor(8.3891) 
```

输入参数也可以是向量，这允许考虑大量变量的函数。这里我们使用`[`torch.sum`](https://pytorch.org/docs/stable/generated/torch.sum.html#torch.sum)`来对参数求和。

```py
z = torch.tensor([1., 2., 3.], requires_grad=True)

g = torch.sum(z ** 2)
g.backward()

print(z.grad)  # gradient is (2 z_1, 2 z_2, 2 z_3) 
```

```py
tensor([2., 4., 6.]) 
```

在数据科学背景下，这里还有一个典型的例子。

```py
X = torch.randn(3, 2)  # Random dataset (features)
y = torch.tensor([[1., 0., 1.]])  # Dataset (labels)
theta = torch.ones(2, 1, requires_grad=True)  # Parameter assignment

predict = X @ theta  # Classifier with parameter vector theta
loss = torch.sum((predict - y)**2)  # Loss function
loss.backward()  # Compute gradients

print(theta.grad)  # gradient of loss 
```

```py
tensor([[29.7629],
        [31.4817]]) 
```

**聊天与学习** 向你喜欢的 AI 聊天机器人解释如何使用 PyTorch 计算二阶导数（有点棘手）。请求代码，你可以将其应用于前面的例子。([在 Colab 中打开](https://colab.research.google.com/github/MMiDS-textbook/MMiDS-textbook.github.io/blob/main/just_the_code/roch_mmids_chap_nn_notebook.ipynb)) $\ddagger$

$\unlhd$

**在 PyTorch 中实现梯度下降** 我们可以不用显式指定梯度函数，而是使用 PyTorch 自动计算它。接下来将这样做。注意，下降更新是在 `with torch.no_grad()` 内部完成的，这确保更新操作本身不被跟踪用于梯度计算。这里输入 `x0` 以及输出 `xk.numpy(force=True)` 都是 NumPy 数组。函数 `torch.Tensor.numpy()` 将 PyTorch 张量转换为 NumPy 数组（有关 `force=True` 选项的解释，请参阅文档）。此外，引用 ChatGPT：

> 在给定的代码中，`.item()` 用于从张量中提取标量值。在 PyTorch 中，当你对张量执行操作时，你会得到张量作为结果，即使结果是一个单一的标量值。`.item()` 用于从张量中提取这个标量值。

```py
def gd_with_ad(f, x0, alpha=1e-3, niters=int(1e6)):
    xk = torch.tensor(x0, requires_grad=True, dtype=torch.float)

    for _ in range(niters):
        value = f(xk)
        value.backward()

        with torch.no_grad():  
            xk -= alpha * xk.grad

        xk.grad.zero_()

    return xk.numpy(force=True), f(xk).item() 
```

**数值角** 我们回顾一个先前的例子。

```py
def f(x):
    return x**3

print(gd_with_ad(f, 2, niters=int(1e4))) 
```

```py
(array(0.03277362, dtype=float32), 3.5202472645323724e-05) 
```

```py
print(gd_with_ad(f, -2, niters=100)) 
```

```py
(array(-4.9335055, dtype=float32), -120.07894897460938) 
```

$\unlhd$

**CHAT & LEARN** 本节简要提到自动微分与符号微分和数值微分不同。请你的首选 AI 聊天机器人详细解释这三种计算导数方法的区别。 $\ddagger$

***自我评估测验*** *(由 Claude、Gemini 和 ChatGPT 协助)*

**1** 设 $A \in \mathbb{R}^{n \times m}$ 和 $B \in \mathbb{R}^{p \times q}$。Kronecker 积 $A \otimes B$ 的维度是什么？

a) $n \times m$

b) $p \times q$

c) $np \times mq$

d) $nq \times mp$

**2** 如果 $f: \mathbb{R}^d \rightarrow \mathbb{R}^m$ 是一个在定义域内点 $x_0$ 处连续可微的函数，那么它在 $x_0$ 处的雅可比 $J_f(x_0)$ 是什么？

a) 表示 $f$ 在 $x_0$ 处变化率的标量。

b) 在 $x_0$ 处 $f$ 的最速上升方向的 $\mathbb{R}^m$ 向量。

c) $f$ 在 $x_0$ 处的分量函数的偏导数构成的 $m \times d$ 矩阵。

d) $f$ 在 $x_0$ 处的 Hessian 矩阵。

**3** 在链式法则的背景下，如果 $f: \mathbb{R}² \to \mathbb{R}³$ 且 $g: \mathbb{R}³ \to \mathbb{R}$，那么 $J_{g \circ f}(x)$ 的雅可比矩阵的维度是多少？

a) $3 \times 2$

b) $1 \times 3$

c) $2 \times 3$

d) $1 \times 2$

**4** 设 $\mathbf{f} : D_1 \to \mathbb{R}^m$，其中 $D_1 \subseteq \mathbb{R}^d$，设 $\mathbf{g} : D_2 \to \mathbb{R}^p$，其中 $D_2 \subseteq \mathbb{R}^m$。假设 $\mathbf{f}$ 在 $D_1$ 的内点 $\mathbf{x}_0$ 处连续可微，且 $\mathbf{g}$ 在 $\mathbf{f}(\mathbf{x}_0)$ 处，即 $D_2$ 的内点处连续可微。根据链式法则，以下哪个是正确的？

a) $J_{\mathbf{g} \circ \mathbf{f}}(\mathbf{x}_0) = J_{\mathbf{f}}(\mathbf{x}_0) \, J_{\mathbf{g}}(\mathbf{f}(\mathbf{x}_0))$

b) $J_{\mathbf{g} \circ \mathbf{f}}(\mathbf{x}_0) = J_{\mathbf{g}}(\mathbf{f}(\mathbf{x}_0)) \, J_{\mathbf{f}}(\mathbf{x}_0)$

c) $J_{\mathbf{g} \circ \mathbf{f}}(\mathbf{x}_0) = J_{\mathbf{f}}(\mathbf{g}(\mathbf{x}_0)) \, J_{\mathbf{g}}(\mathbf{x}_0)$

d) $J_{\mathbf{g} \circ \mathbf{f}}(\mathbf{x}_0) = J_{\mathbf{g}}(\mathbf{x}_0) \, J_{\mathbf{f}}(\mathbf{g}(\mathbf{x}_0))$

**5** 设 $A \in \mathbb{R}^{m \times d}$ 和 $\mathbf{b} \in \mathbb{R}^{m}$。定义向量值函数 $\mathbf{f} : \mathbb{R}^d \to \mathbb{R}^m$ 为 $\mathbf{f}(\mathbf{x}) = A \mathbf{x} + \mathbf{b}$。$\mathbf{f}$ 在 $\mathbf{x}_0$ 处的雅可比矩阵是什么？

a) $J_{\mathbf{f}}(\mathbf{x}_0) = A^T$

b) $J_{\mathbf{f}}(\mathbf{x}_0) = A \mathbf{x}_0 + \mathbf{b}$

c) $J_{\mathbf{f}}(\mathbf{x}_0) = A$

d) $J_{\mathbf{f}}(\mathbf{x}_0) = \mathbf{b}$

1 题的答案：c. 理由：文本将克罗内克积定义为具有 $np \times mq$ 维度的分块矩阵。

2 题的答案：c. 理由：文本将向量值函数的雅可比定义为偏导数矩阵。

3 题的答案：d. 理由：复合函数 $g \circ f$ 将 $\mathbb{R}² \to \mathbb{R}$ 映射，因此雅可比矩阵 $J_{g \circ f}(x)$ 是 $1 \times 2$。

4 题的答案：b. 理由：文本中提到：“链式法则给出了复合函数雅可比的公式。[……]假设 $\mathbf{f}$ 在 $D_1$ 的内部点 $\mathbf{x}_0$ 处连续可微，且 $\mathbf{g}$ 在 $\mathbf{f}(\mathbf{x}_0)$ 处，即 $D_2$ 的内部点处连续可微。那么”

$$ J_{\mathbf{g} \circ \mathbf{f}}(\mathbf{x}_0) = J_{\mathbf{g}}(\mathbf{f}(\mathbf{x}_0)) \, J_{\mathbf{f}}(\mathbf{x}_0) $$

作为矩阵的产物。”

5 题的答案：c. 理由：文本中提到：“设 $A \in \mathbb{R}^{m \times d}$ 和 $\mathbf{b} = (b_1,\ldots,b_m) \in \mathbb{R}^{m}$。定义向量值函数 $\mathbf{f} = (f_1, \ldots, f_m) : \mathbb{R}^d \to \mathbb{R}^m$ 为 $\mathbf{f}(\mathbf{x}) = A \mathbf{x} + \mathbf{b}$。$$ \text{所以 } J_{\mathbf{f}}(\mathbf{x}) = A.$$”
