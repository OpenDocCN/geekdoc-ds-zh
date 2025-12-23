# 8.7\. 在线补充材料

> 原文：[`mmids-textbook.github.io/chap08_nn/supp/roch-mmids-nn-supp.html`](https://mmids-textbook.github.io/chap08_nn/supp/roch-mmids-nn-supp.html)

## 8.7.1\. 测验、解答、代码等.#

### 8.7.1.1\. 仅代码#

下面可以访问本章代码的交互式 Jupyter 笔记本（推荐使用 Google Colab）。鼓励您对其进行实验。一些建议的计算练习散布其中。笔记本也可以作为幻灯片查看。

+   [笔记本](https://github.com/MMiDS-textbook/MMiDS-textbook.github.io/blob/main/just_the_code/roch_mmids_chap_nn_notebook.ipynb) ([在 Colab 中打开](https://colab.research.google.com/github/MMiDS-textbook/MMiDS-textbook.github.io/blob/main/just_the_code/roch_mmids_chap_nn_notebook.ipynb))

+   [幻灯片](https://raw.githack.com/MMiDS-textbook/MMiDS-textbook.github.io/main/just_the_code/roch_mmids_chap_nn_notebook_slides.slides.html)

### 8.7.1.2\. 自我评估测验#

通过以下链接可以访问自我评估测验的更全面网络版本。

+   [第 8.2 节](https://raw.githack.com/MMiDS-textbook/MMiDS-textbook.github.io/main/quizzes/self-assessment/quiz_8_2.html)

+   [第 8.3 节](https://raw.githack.com/MMiDS-textbook/MMiDS-textbook.github.io/main/quizzes/self-assessment/quiz_8_3.html)

+   [第 8.4 节](https://raw.githack.com/MMiDS-textbook/MMiDS-textbook.github.io/main/quizzes/self-assessment/quiz_8_4.html)

+   [第 8.5 节](https://raw.githack.com/MMiDS-textbook/MMiDS-textbook.github.io/main/quizzes/self-assessment/quiz_8_5.html)

### 8.7.1.3\. 自动测验#

可以在此处访问本章的自动生成的测验（推荐使用 Google Colab）。

+   [自动测验](https://github.com/MMiDS-textbook/MMiDS-textbook.github.io/blob/main/quizzes/auto_quizzes/roch-mmids-nn-autoquiz.ipynb) ([在 Colab 中打开](https://colab.research.google.com/github/MMiDS-textbook/MMiDS-textbook.github.io/blob/main/quizzes/auto_quizzes/roch-mmids-nn-autoquiz.ipynb))

### 8.7.1.4\. 奇数编号预热练习的解答#

*(在克劳德、双子星和 ChatGPT 的帮助下)*

**E8.2.1** 通过堆叠矩阵 $A$ 的列来获得向量化：$\text{vec}(A) = (2, 0, 1, -1)$。

**E8.2.3**

$$\begin{split} A \otimes B = \begin{pmatrix} 1 \cdot B & 2 \cdot B \\ -1 \cdot B & 0 \cdot B \end{pmatrix} = \begin{pmatrix} 3 & -1 & 6 & -2 \\ 2 & 1 & 4 & 2 \\ -3 & 1 & -6 & 2 \\ -2 & -1 & -4 & -2 \end{pmatrix}. \end{split}$$

**E8.2.5**

$$\begin{split} A \otimes B = \begin{pmatrix} 1 \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix} & 2 \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix} \\ 3 \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix} & 4 \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix} \end{pmatrix} = \begin{pmatrix} 5 & 6 & 10 & 12 \\ 7 & 8 & 14 & 16 \\ 15 & 18 & 20 & 24 \\ 21 & 24 & 28 & 32 \end{pmatrix}. \end{split}$$

**E8.2.7** 首先，计算 $\mathbf{f}$ 和 $\mathbf{g}$ 的雅可比矩阵：

$$\begin{split} J_{\mathbf{f}}(x, y) = \begin{pmatrix} 2x & 2y \\ y & x \end{pmatrix}, \quad J_{\mathbf{g}}(u, v) = \begin{pmatrix} v & u \\ 1 & 1 \end{pmatrix}. \end{split}$$

然后，根据链式法则，

$$\begin{split} J_{\mathbf{g} \circ \mathbf{f}}(1, 2) = J_{\mathbf{g}}(\mathbf{f}(1, 2)) \, J_{\mathbf{f}}(1, 2) = J_{\mathbf{g}}(5, 2) \, J_{\mathbf{f}}(1, 2) = \begin{pmatrix} 2 & 5 \\ 1 & 1 \end{pmatrix} \begin{pmatrix} 2 & 4 \\ 2 & 1 \end{pmatrix} = \begin{pmatrix} 14 & 13 \\ 4 & 5 \end{pmatrix}. \end{split}$$

**E8.2.9** 从 E8.2.5，我们有

$$\begin{split} A \otimes B = \begin{pmatrix} 5 & 6 & 10 & 12 \\ 7 & 8 & 14 & 16 \\ 15 & 18 & 20 & 24 \\ 21 & 24 & 28 & 32 \end{pmatrix} \end{split}$$

所以，

$$\begin{split}(A \otimes B)^T = \begin{pmatrix} 5 & 7 & 15 & 21 \\ 6 & 8 & 18 & 24 \\ 10 & 14 & 20 & 28 \\ 12 & 16 & 24 & 32 \end{pmatrix}. \end{split}$$

现在，

$$\begin{split} A^T = \begin{pmatrix} 1 & 3 \\ 2 & 4 \end{pmatrix}, \quad B^T = \begin{pmatrix} 5 & 7 \\ 6 & 8 \end{pmatrix} \end{split}$$

所以，

$$\begin{split} A^T \otimes B^T = \begin{pmatrix} 1 \begin{pmatrix} 5 & 7 \\ 6 & 8 \end{pmatrix} & 3 \begin{pmatrix} 5 & 7 \\ 6 & 8 \end{pmatrix} \\ 2 \begin{pmatrix} 5 & 7 \\ 6 & 8 \end{pmatrix} & 4 \begin{pmatrix} 5 & 7 \\ 6 & 8 \end{pmatrix} \end{pmatrix} = \begin{pmatrix} 5 & 7 & 15 & 21 \\ 6 & 8 & 18 & 24 \\ 10 & 14 & 20 & 28 \\ 12 & 16 & 24 & 32 \end{pmatrix}. \end{split}$$

我们看到 $(A \otimes B)^T = A^T \otimes B^T$，正如克罗内克积的性质所预期的那样。

**E8.2.11**

$$\begin{split} \nabla f(x, y, z) = \begin{pmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \\ \frac{\partial f}{\partial z} \end{pmatrix} = \begin{pmatrix} 2x \\ 2y \\ 2z \end{pmatrix}. \end{split}$$

所以，

$$\begin{split} \nabla f(1, 2, 3) = \begin{pmatrix} 2 \\ 4 \\ 6 \end{pmatrix}. \end{split}$$

**E8.2.13** 首先，计算 $f$ 的梯度以及 $\mathbf{g}$ 的雅可比矩阵：

$$\begin{split} \nabla f(x, y) = \begin{pmatrix} y \\ x \end{pmatrix}, \quad J_{\mathbf{g}}(x, y) = \begin{pmatrix} 2x & 0 \\ 0 & 2y \end{pmatrix}. \end{split}$$

然后，根据链式法则，

$$\begin{split} J_{f \circ \mathbf{g}}(1, 2) = \nabla f(\mathbf{g}(1, 2))^T \, J_{\mathbf{g}}(1, 2) = \nabla f(1, 4)^T \, J_{\mathbf{g}}(1, 2) = \begin{pmatrix} 4 & 1 \end{pmatrix} \begin{pmatrix} 2 & 0 \\ 0 & 4 \end{pmatrix} = \begin{pmatrix} 8 & 4 \end{pmatrix}. \end{split}$$

**E8.2.15** $\mathbf{g}$ 的雅可比矩阵为

$$\begin{split} J_{\mathbf{g}}(x, y, z) = \begin{pmatrix} f'(x) & 0 & 0 \\ 0 & f'(y) & 0 \\ 0 & 0 & f'(z) \end{pmatrix} \end{split}$$

其中 $f'(x) = \cos(x).$ 因此，

$$\begin{split} J_{\mathbf{g}}(\frac{\pi}{2}, \frac{\pi}{4}, \frac{\pi}{6}) = \begin{pmatrix} \cos(\frac{\pi}{2}) & 0 & 0 \\ 0 & \cos(\frac{\pi}{4}) & 0 \\ 0 & 0 & \cos(\frac{\pi}{6}) \end{pmatrix} = \begin{pmatrix} 0 & 0 & 0 \\ 0 & \frac{\sqrt{2}}{2} & 0 \\ 0 & 0 & \frac{\sqrt{3}}{2} \end{pmatrix}. \end{split}$$

**E8.3.1** $AB$ 的每个元素是 $A$ 的某一行与 $B$ 的某一列的点积，这需要 2 次乘法和 1 次加法。由于 $AB$ 有 4 个元素，总的操作次数是 $4 \times 3 = 12$。

**E8.3.3** 我们有

$$ \ell(\hat{\mathbf{y}}) = \hat{y}_1² + \hat{y}_2² $$

因此，偏导数为 $\frac{\partial \ell}{\partial \hat{y}_1} = 2 \hat{y}_1$ 和 $\frac{\partial \ell}{\partial \hat{y}_2} = 2 \hat{y}_2$。

$$ J_{\ell}(\hat{\mathbf{y}}) = 2 \hat{\mathbf{y}}^T. $$

**E8.3.5** 从 E8.3.4，我们得到 $\mathbf{z}_1 = \mathbf{g}_0(\mathbf{z}_0) = (-1, -1)$。然后，$\mathbf{z}_2 = \mathbf{g}_1(\mathbf{z}_1) = (1, -2)$ 和 $f(\mathbf{x}) = \ell(\mathbf{z}_2) = 5$。根据**链式法则**，

$$\begin{split} \nabla f(\mathbf{x})^T = J_f(\mathbf{x}) = J_{\ell}(\mathbf{z}_1) \,J_{\mathbf{g}_1}(\mathbf{z}_1) \,J_{\mathbf{g}_0}(\mathbf{z}_0) = 2 \mathbf{z}_2^T \begin{pmatrix} -1 & 0 \\ 1 & 1 \end{pmatrix} \begin{pmatrix} 1 & 2 \\ -1 & 0 \end{pmatrix} = (-10, -4) \begin{pmatrix} 1 & 2 \\ -1 & 0 \end{pmatrix} = (6, -20). \end{split}$$

**E8.3.7** 我们有

$$ g_1(\mathbf{z}_1, \mathbf{w}_1) = w_4 z_{1,1} + w_5 z_{1,2} $$

因此，通过计算所有偏导数，

$$ J_{g_1}(\mathbf{z}_1, \mathbf{w}_1) = \begin{pmatrix} w_4 & w_5 & z_{1,1} & z_{1,2} \end{pmatrix} = \begin{pmatrix} \mathbf{w}_1^T & \mathbf{z}_1^T \end{pmatrix} = \begin{pmatrix} W_1 & I_{1 \times 1} \otimes \mathbf{z}_1^T \end{pmatrix}. $$

使用文本中的符号，$A_1 = W_1$ 和 $B_1 = \mathbf{z}_1^T = I_{1 \times 1} \otimes \mathbf{z}_1^T$.

**E8.3.9** 我们有

$$ f(\mathbf{w}) = (w_4 (- w_0 + w_1) + w_5 (-w_2 + w_3))² $$

因此，根据**链式法则**，偏导数为

$$ \frac{\partial f}{\partial w_0} = 2(w_4 (- w_0 + w_1) + w_5 (-w_2 + w_3)) (-w_4) $$$$ \frac{\partial f}{\partial w_1} = 2(w_4 (- w_0 + w_1) + w_5 (-w_2 + w_3)) (w_4) $$$$ \frac{\partial f}{\partial w_2} = 2(w_4 (- w_0 + w_1) + w_5 (-w_2 + w_3)) (-w_5) $$$$ \frac{\partial f}{\partial w_3} = 2(w_4 (- w_0 + w_1) + w_5 (-w_2 + w_3)) (w_5) $$$$ \frac{\partial f}{\partial w_4} = 2(w_4 (- w_0 + w_1) + w_5 (-w_2 + w_3)) (-w_0 + w_1) $$$$ \frac{\partial f}{\partial w_5} = 2(w_4 (- w_0 + w_1) + w_5 (-w_2 + w_3)) (-w_2 + w_3). $$

此外，根据 E8.3.7，

$$\begin{split} z_2 = g_1(\mathbf{z}_1, \mathbf{w}_1) = W_1 \mathbf{z}_1 = \begin{pmatrix} w_4 & w_5\end{pmatrix} \begin{pmatrix}- w_0 + w_1\\-w_2 + w_3\end{pmatrix} = w_4 (- w_0 + w_1) + w_5 (-w_2 + w_3). \end{split}$$

通过基本递归和 E8.3.3 和 E8.3.8 的结果，

$$\begin{align*} J_f(\mathbf{w}) &= J_{\ell}(h(\mathbf{w})) \,J_{h}(\mathbf{w}) = 2 z_2 \begin{pmatrix} A_1 B_0 & B_1\end{pmatrix}\\ &= 2 (w_4 (- w_0 + w_1) + w_5 (-w_2 + w_3)) (-w_4, w_4, -w_5, w_5, -w_0 + w_1, -w_2 + w_3). \end{align*}$$

**E8.4.1** 完整的梯度下降步长为：

$$ \frac{1}{5} \sum_{i=1}⁵ \nabla f_{\mathbf{x}_i, y_i}(w) = \frac{1}{5}((1, 2) + (-1, 1) + (0, -1) + (2, 0) + (1, 1)) = (\frac{3}{5}, \frac{3}{5}). $$

批大小为 2 的期望 SGD 步长为：

$$ \mathbb{E} [\frac{1}{2} \sum_{i\in B} \nabla f_{\mathbf{x}_i, y_i}(w)] = \frac{1}{5} \sum_{i=1}⁵ \nabla f_{x_i, y_i}(w) = (\frac{3}{5}, \frac{3}{5}), $$

这等于完整的梯度下降步长，正如在“期望 SGD 步长”引理中证明的那样。

**E8.4.3**

$$\begin{align*} \mathrm{KL}(\mathbf{p} \| \mathbf{q}) &= \sum_{i=1}³ p_i \log \frac{p_i}{q_i} \\ &= 0.2 \log \frac{0.2}{0.1} + 0.3 \log \frac{0.3}{0.4} + 0.5 \log \frac{0.5}{0.5} \\ &\approx 0.2 \cdot 0.6931 + 0.3 \cdot (-0.2877) + 0.5 \cdot 0 \\ &\approx 0.0525. \end{align*}$$

**E8.4.5** SGD 更新由

$$\begin{align*} w &\leftarrow w - \frac{\alpha}{|B|} \sum_{i \in B} \frac{\partial \ell}{\partial w}(w, b; x_i, y_i), \\ b &\leftarrow b - \frac{\alpha}{|B|} \sum_{i \in B} \frac{\partial \ell}{\partial b}(w, b; x_i, y_i). \end{align*}$$

将值代入，我们得到

$$\begin{align*} w &\leftarrow 1 - \frac{0.1}{2} (2 \cdot 2(2 \cdot 1 + 0 - 3) + 2 \cdot 1(1 \cdot 1 + 0 - 2)) = 1.3, \\ b &\leftarrow 0 - \frac{0.1}{2} (2(2 \cdot 1 + 0 - 3) + 2(1 \cdot 1 + 0 - 2)) = 0.3. \end{align*}$$

**E8.4.7**

$$\begin{align*} \nabla \ell(w; x, y) &= -\frac{y}{\sigma(wx)} \sigma'(wx) x + \frac{1-y}{1-\sigma(wx)} \sigma'(wx) x \\ &= -yx(1 - \sigma(wx)) + x(1-y)\sigma(wx) \\ &= x(\sigma(wx) - y). \end{align*}$$

我们使用了$\sigma'(z) = \sigma(z)(1-\sigma(z))$的事实。

**E8.4.9** 首先，我们计算$\mathbf{z}_1 = W\mathbf{x} = \begin{pmatrix} 0 & 0 \\ 0 & 0 \\ 0 & 0 \end{pmatrix} (1, 2) = (0, 0, 0)$。然后，$\hat{\mathbf{y}} = \boldsymbol{\gamma}(\mathbf{z}_1) = (\frac{1}{3}, \frac{1}{3}, \frac{1}{3})$。从文本中，我们有：

$$\begin{split} \nabla f(\mathbf{w}) = (\boldsymbol{\gamma}(W\mathbf{x}) - \mathbf{y}) \otimes \mathbf{x} = (\hat{\mathbf{y}} - \mathbf{y}) \otimes \mathbf{x} = (\frac{1}{3}, \frac{1}{3}, -\frac{2}{3}) \otimes (1, 2) = \begin{pmatrix} \frac{1}{3} & \frac{2}{3} \\ \frac{1}{3} & \frac{2}{3} \\ -\frac{2}{3} & -\frac{4}{3} \end{pmatrix}. \end{split}$$

**E8.4.11** 首先，我们计算各个梯度：

$$\begin{align*} \nabla f_{\mathbf{x}_1, \mathbf{y}_1}(W) &= (\boldsymbol{\gamma}(W \mathbf{x}_1) - \mathbf{y}_1) \otimes x_1 = (\frac{1}{3}, \frac{1}{3}, -\frac{2}{3}) \otimes (1, 2) = \begin{pmatrix} \frac{1}{3} & \frac{2}{3} \\ \frac{1}{3} & \frac{2}{3} \\ -\frac{2}{3} & -\frac{4}{3} \end{pmatrix}, \\ \nabla f_{\mathbf{x}_2, \mathbf{y}_2}(W) &= (\boldsymbol{\gamma}(W\mathbf{x}_2) - \mathbf{y}_2) \otimes x_2 = (-\frac{2}{3}, \frac{1}{3}, \frac{1}{3}) \otimes (4, -1) = \begin{pmatrix} -\frac{8}{3} & \frac{2}{3} \\ \frac{4}{3} & -\frac{1}{3} \\ \frac{4}{3} & -\frac{1}{3} \end{pmatrix}. \end{align*}$$

然后，完整的梯度为：

$$\begin{split} \frac{1}{2} (\nabla f_{\mathbf{x}_1, \mathbf{y}_1}(W) + \nabla f_{\mathbf{x}_2, \mathbf{y}_2}(W)) = \frac{1}{2} \left(\begin{pmatrix} \frac{1}{3} & \frac{2}{3} \\ \frac{1}{3} & \frac{2}{3} \\ -\frac{2}{3} & -\frac{4}{3} \end{pmatrix} + \begin{pmatrix} -\frac{8}{3} & \frac{2}{3} \\ \frac{4}{3} & -\frac{1}{3} \\ \frac{4}{3} & -\frac{1}{3} \end{pmatrix}\right) = \begin{pmatrix} -\frac{11}{6} & \frac{2}{3} \\ \frac{5}{6} & \frac{1}{6} \\ \frac{1}{3} & -\frac{5}{6} \end{pmatrix}. \end{split}$$

**E8.4.13** 交叉熵损失由以下公式给出

$$ -\log(0.3) \approx 1.204. $$

**E8.5.1** $\sigma(1) = \frac{1}{1 + e^{-1}} \approx 0.73$ $\sigma(-1) = \frac{1}{1 + e^{1}} \approx 0.27$ $\sigma(2) = \frac{1}{1 + e^{-2}} \approx 0.88$

**E8.5.3** $\bsigma(\mathbf{z}) = (\bsigma(1), \bsigma(-1), \bsigma(2)) \approx (0.73, 0.27, 0.88)$ $\bsigma'(\mathbf{z}) = (\bsigma'(1), \bsigma'(-1), \bsigma'(2)) \approx (0.20, 0.20, 0.10)$

**E8.5.5** $W\mathbf{x} = \begin{pmatrix} -1 \\ 4 \end{pmatrix}$，因此 $\sigma(W\mathbf{x}) \approx (0.27, 0.98)$，$J_\bsigma(W\mathbf{x}) = \mathrm{diag}(\bsigma(W\mathbf{x}) \odot (1 - \bsigma(W\mathbf{x}))) \approx \begin{pmatrix} 0.20 & 0 \\ 0 & 0.02 \end{pmatrix}$

**E8.5.7** $\nabla H(\mathbf{y}, \mathbf{z}) = (-\frac{y_1}{z_1}, -\frac{y_2}{z_2}) = (-\frac{0}{0.3}, -\frac{1}{0.7}) \approx (0, -1.43)$

### 8.7.1.5\. 学习成果#

+   定义雅可比矩阵并使用它来计算向量值函数的微分。

+   陈述并应用广义链式法则来计算函数复合的雅可比矩阵。

+   使用矩阵的 Hadamard 和 Kronecker 积进行计算。

+   描述自动微分的用途及其相对于符号和数值微分的优势。

+   在 PyTorch 中实现自动微分以计算向量值函数的梯度。

+   推导多层递进函数的链式法则并将其应用于计算梯度。

+   比较自动微分的正向和反向模式的计算复杂度。

+   定义递进函数并识别它们的关键属性。

+   推导递进函数雅可比矩阵的基本递归公式。

+   实现反向传播算法以有效地计算渐进函数的梯度。

+   分析反向传播算法的计算复杂度，以矩阵-向量乘法的数量来衡量。

+   描述随机梯度下降（SGD）算法，并解释它与标准梯度下降有何不同。

+   从损失函数的梯度推导出随机梯度下降的更新规则。

+   证明预期的 SGD 步长等于完整梯度下降步长。

+   评估在真实世界数据集上使用随机梯度下降训练的模型性能。

+   定义多层感知器（MLP）架构，并描述每个层中仿射映射和激活函数的作用。

+   使用对角矩阵和克罗内克积的性质计算 sigmoid 激活函数的雅可比矩阵。

+   在一个小型 MLP 示例中应用链式法则来计算损失函数相对于权重的梯度。

+   使用前向和反向传递泛化具有任意层数的 MLP 的梯度计算。

+   使用 PyTorch 实现神经网络的训练。

$\aleph$

## 8.7.2\. 其他部分#

### 8.7.2.1\. 另一个例子：线性回归#

我们给出了另一个渐进函数和反向传播及随机梯度下降应用的实例。

**计算梯度** 当我们从分类的角度出发，已经激励了上一节中引入的框架，它也立即适用于回归设置。分类和回归都是监督学习的实例。

我们首先计算单个样本的梯度。这里 $\mathbf{x} \in \mathbb{R}^d$，但 $y$ 是一个实值结果变量。我们回顾线性回归的案例，其中损失函数是

$$ \ell(z) = (z - y)² $$

并且回归函数只有输入层和输出层，没有隐藏层（即 $L=0$），

$$ h(\mathbf{w}) = \sum_{j=1}^d w_{j} x_{j} = \mathbf{x}^T\mathbf{w}, $$

其中 $\mathbf{w} \in \mathbb{R}^{d}$ 是参数。回想一下，我们可以通过向输入添加一个常数项（一个不依赖于输入的项）来包含一个常数项。为了简化符号，我们假设如果需要，这个预处理已经完成。

最后，单个样本的目标函数是

$$ f(\mathbf{w}) = \ell(h(\mathbf{w})) = \left(\sum_{j=1}^d w_{j} x_{j} - y\right)² = \left(\mathbf{x}^T\mathbf{w} - y\right)²\. $$

使用前一小节中的符号，在这种情况下，前向传递如下：

*初始化：*

$$\mathbf{z}_0 := \mathbf{x}.$$

*前向层循环：*

$$\begin{align*} \hat{y} := z_1 := g_0(\mathbf{z}_0,\mathbf{w}_0) &= \sum_{j=1}^d w_{0,j} z_{0,j} = \mathbf{z}_0^T \mathbf{w}_0 \end{align*}$$$$\begin{align*} \begin{pmatrix} A_0 & B_0 \end{pmatrix} := J_{g_0}(\mathbf{z}_0,\mathbf{w}_0) &= ( w_{0,1},\ldots, w_{0,d},z_{0,1},\ldots,z_{0,d} )^T = \begin{pmatrix}\mathbf{w}_0^T & \mathbf{z}_0^T\end{pmatrix}, \end{align*}$$

其中 $\mathbf{w}_0 := \mathbf{w}$。

*损失：*

$$\begin{align*} z_2 &:= \ell(z_1) = (z_1 - y)²\\ p_2 &:= \frac{\mathrm{d}}{\mathrm{d} z_1} {\ell}(z_1) = 2 (z_1 - y). \end{align*}$$

反向传播过程是：

*反向层循环：*

$$\begin{align*} \mathbf{q}_0 := B_0^T p_1 &= 2 (z_1 - y) \, \mathbf{z}_0. \end{align*}$$

*输出：*

$$ \nabla f(\mathbf{w}) = \mathbf{q}_0. $$

如前所述，实际上没有必要计算 $A_0$ 和 $\mathbf{p}_0$。

**`Advertising` 数据集和最小二乘解** 我们回到 `Advertising` 数据集。

```py
data = pd.read_csv('advertising.csv')
data.head() 
```

|  | Unnamed: 0 | TV | radio | newspaper | sales |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | 230.1 | 37.8 | 69.2 | 22.1 |
| 1 | 2 | 44.5 | 39.3 | 45.1 | 10.4 |
| 2 | 3 | 17.2 | 45.9 | 69.3 | 9.3 |
| 3 | 4 | 151.5 | 41.3 | 58.5 | 18.5 |
| 4 | 5 | 180.8 | 10.8 | 58.4 | 12.9 |

```py
n = len(data.index)
print(n) 
```

```py
200 
```

我们首先使用之前详细说明的平方最小二乘法来计算解。我们使用 `numpy.column_stack` 向特征向量添加一列。

```py
TV = data['TV'].to_numpy()
radio = data['radio'].to_numpy()
newspaper = data['newspaper'].to_numpy()
sales = data['sales'].to_numpy()
features = np.stack((TV, radio, newspaper), axis=-1)
A = np.column_stack((np.ones(n), features))
coeff = mmids.ls_by_qr(A, sales)
print(coeff) 
```

```py
[ 2.93888937e+00  4.57646455e-02  1.88530017e-01 -1.03749304e-03] 
```

均方误差（MSE）是：

```py
np.mean((A @ coeff - sales)**2) 
```

```py
2.7841263145109365 
```

**使用 PyTorch 解决问题** 我们将使用 PyTorch 来实现前面的方法。我们首先将数据转换为 PyTorch 张量。然后使用 `torch.utils.data.TensorDataset` 创建数据集。最后，`torch.utils.data.DataLoader` 提供了加载数据的批处理工具。我们取大小为 `BATCH_SIZE = 64` 的迷你批次，并在每次传递时对样本进行随机排列（使用选项 `shuffle=True`）。

```py
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

features_tensor = torch.tensor(features, dtype=torch.float32)
sales_tensor = torch.tensor(sales, dtype=torch.float32).view(-1, 1)

BATCH_SIZE = 64
train_dataset = TensorDataset(features_tensor, sales_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) 
```

现在我们构建我们的模型。它只是一个从 $\mathbb{R}³$ 到 $\mathbb{R}$ 的仿射映射。请注意，没有必要通过添加 $1$s 来预处理输入。PyTorch 会自动添加一个常数项（或“偏置变量”）（除非选择选项 [`bias=False`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)）。

```py
model = nn.Sequential(
    nn.Linear(3, 1)  # 3 input features, 1 output value
) 
```

最后，我们准备好在损失函数上运行我们选择的优化方法，这些方法将在下面指定。有很多 [优化器](https://pytorch.org/docs/stable/optim.html#algorithms) 可用。（参见这篇 [文章](https://hackernoon.com/demystifying-different-variants-of-gradient-descent-optimization-algorithm-19ae9ba2e9bc)，了解许多常见优化器的简要说明。）这里我们使用 SGD 作为优化器。损失函数是均方误差（MSE）。快速教程在这里 [这里](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)。

```py
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-5) 
```

选择合适的数据遍历次数（即 epoch 数）需要一些实验。这里$10⁴$就足够了。但为了节省时间，我们只运行了$100$个 epoch。从结果中你可以看到，这还不够。在每次遍历中，我们计算当前模型的输出，使用`backward()`获取梯度，然后使用`step()`进行下降更新。我们还需要首先重置梯度（否则它们会默认累加）。

```py
epochs = 100
for epoch in range(epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}") 
```

最终参数和损失如下：

<details class="hide above-input"><summary aria-label="Toggle hidden content">显示代码单元格源代码 隐藏代码单元格源代码</summary>

```py
weights = model[0].weight.detach().numpy()
bias = model[0].bias.detach().numpy()
print("Weights:", weights)
print("Bias:", bias) 
```</details>

```py
Weights: [[0.05736413 0.11314777 0.08020781]]
Bias: [-0.02631279] 
```

<details class="hide above-input"><summary aria-label="Toggle hidden content">显示代码单元格源代码 隐藏代码单元格源代码</summary>

```py
# Evaluate the model
model.eval()
with torch.no_grad():
    total_loss = 0
    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()

    print(f"Mean Squared Error on Training Set: {total_loss  /  len(train_loader)}") 
```</details>

```py
Mean Squared Error on Training Set: 7.885213494300842 
```

### 8.7.2.2\. 卷积神经网络#

我们回到 Fashion MNIST 数据集。使用为图像定制的神经网络，可以做得比我们之前更好，这种神经网络被称为[卷积神经网络](https://cs231n.github.io/convolutional-networks/)。从[Wikipedia](https://en.wikipedia.org/wiki/Convolutional_neural_network)：

> 在深度学习中，卷积神经网络（CNN，或 ConvNet）是一类深度神经网络，最常用于分析视觉图像。它们也被称为基于共享权重架构和平移不变性特征的平移不变或空间不变人工神经网络（SIANN）。

更多背景信息可以在斯坦福的[CS231n](http://cs231n.github.io/)的出色[模块](http://cs231n.github.io/convolutional-networks/)中找到。我们的 CNN 将是[卷积层](http://cs231n.github.io/convolutional-networks/#conv)和[池化层](http://cs231n.github.io/convolutional-networks/#pool)的组合。

**CHAT & LEARN** 卷积神经网络（CNN）在图像分类方面非常强大。请你的心仪 AI 聊天机器人解释 CNN 的基本概念，包括卷积层和池化层。$\ddagger$

```py
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

train_dataset = datasets.FashionMNIST(root='./data', train=True, 
                               download=True, transform=transforms.ToTensor())
test_dataset = datasets.FashionMNIST(root='./data', train=False, 
                              download=True, transform=transforms.ToTensor())

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() 
                      else ("mps" if torch.backends.mps.is_available() 
                            else "cpu"))
print("Using device:", device) 
```

```py
Using device: mps 
```

新模型如下。

```py
model = nn.Sequential(
    # First convolution, operating upon a 28x28 image
    nn.Conv2d(1, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    # Second convolution, operating upon a 14x14 image
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    # Third convolution, operating upon a 7x7 image
    nn.Conv2d(32, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    # Flatten the tensor
    nn.Flatten(),

    # Fully connected layer
    nn.Linear(32 * 3 * 3, 10),
).to(device) 
```

我们进行训练和测试。

```py
loss_fn = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters()) 
```

```py
mmids.training_loop(train_loader, model, loss_fn, optimizer, device) 
```

```py
Epoch 1/3 
```

```py
Epoch 2/3 
```

```py
Epoch 3/3 
```

```py
mmids.test(test_loader, model, loss_fn, device) 
```

```py
Test error: 88.6% accuracy 
```

注意更高的准确率。

最后，我们尝试原始的 MNIST 数据集。我们使用相同的 CNN。

```py
train_dataset = datasets.MNIST(root='./data', train=True, 
                                      download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='./data', train=False, 
                                     download=True, transform=transforms.ToTensor())

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) 
```

```py
loss_fn = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters()) 
```

```py
mmids.training_loop(train_loader, model, loss_fn, optimizer, device) 
```

```py
Epoch 1/3 
```

```py
Epoch 2/3 
```

```py
Epoch 3/3 
```

```py
mmids.test(test_loader, model, loss_fn, device) 
```

```py
Test error: 98.6% accuracy 
```

注意在这个（容易的 - 如你所见）数据集上的非常高的准确率。

## 8.7.3\. 额外证明#

**拉格朗日乘数条件证明** 我们首先证明**拉格朗日乘数：一阶必要条件**。我们遵循优秀的教科书 [[Ber](http://www.athenasc.com/nonlinbook.html)，第 4.1 节]。证明使用了雅可比矩阵的概念。

*证明思路:* 我们通过在目标函数中对约束进行惩罚，将问题简化为一个无约束优化问题。然后我们应用无约束的**一阶必要条件**。

*证明:* *(拉格朗日乘数法：一阶必要条件)* 我们通过在目标函数中惩罚约束条件将问题简化为一个无约束优化问题。我们还添加了一个正则化项，以确保 $\mathbf{x}^*$ 在一个邻域内是唯一的局部最小值。具体来说，对于每个非负整数 $k$，考虑目标函数

$$ F^k(\mathbf{x}) = f(\mathbf{x}) + \frac{k}{2} \|\mathbf{h}(\mathbf{x})\|² + \frac{\alpha}{2} \|\mathbf{x} - \mathbf{x}^*\|² $$

对于某个正常数 $\alpha > 0$。注意，随着 $k$ 的增大，惩罚变得更为重要，因此强制约束变得更为可取。证明分几个步骤进行。

我们首先考虑一个版本，该版本在 $\mathbf{x}^*$ 的邻域内最小化 $F^k$，受到约束。因为 $\mathbf{x}^*$ 是在 $\mathbf{h}(\mathbf{x}) = \mathbf{0}$ 的约束下 $f$ 的局部最小值，存在 $\delta > 0$，使得对于所有在 $\mathbf{x}^*$ 邻域内的可行 $\mathbf{x}$，都有 $f(\mathbf{x}^*)\leq f(\mathbf{x})$。

$$ \mathscr{X} = B_{\delta}(\mathbf{x}^*) = \{\mathbf{x}:\|\mathbf{x} - \mathbf{x}^*\| \leq \delta\}. $$

**引理** **(步骤 I：在 $\mathbf{x}^*$ 邻域内解决惩罚问题)** 对于 $k \geq 1$，设 $\mathbf{x}^k$ 是最小化问题的全局最小值

$$ \min_{\mathbf{x} \in \mathscr{X}} F^k(\mathbf{x}). $$

a) 序列 $\{\mathbf{x}^k\}_{k=1}^{+\infty}$ 收敛到 $\mathbf{x}^*$。

b) 对于足够大的 $k$，$\mathbf{x}^k$ 是目标函数 $F^k$ 的局部最小值，*没有任何约束*。

$\flat$

*证明:* 集合 $\mathscr{X}$ 是闭集且有界，$F^k$ 是连续的。因此，根据*极值定理*，序列 $\{\mathbf{x}^k\}_{k=1}^{+\infty}$ 是有定义的。设 $\bar{\mathbf{x}}$ 是 $\{\mathbf{x}^k\}_{k=1}^{+\infty}$ 的任意极限点。我们证明 $\bar{\mathbf{x}} = \mathbf{x}^*$。这将意味着 a)。这也意味着 b)，因为当 $k$ 足够大时，$\mathbf{x}^k$ 必须是 $\mathscr{X}$ 的内点。

设 $-\infty < m \leq M < + \infty$ 是 $\mathscr{X}$ 上函数 $f$ 的最小值和最大值，根据*极值定理*存在。那么，对于所有 $k$，根据 $\mathbf{x}^k$ 的定义以及 $\mathbf{x}^*$ 是可行解的事实

$$\begin{align*} (*) \qquad f(\mathbf{x}^k) &+ \frac{k}{2} \|\mathbf{h}(\mathbf{x}^k)\|² + \frac{\alpha}{2} \|\mathbf{x}^k - \mathbf{x}^*\|²\\ &\leq f(\mathbf{x}^*) + \frac{k}{2} \|\mathbf{h}(\mathbf{x}^*)\|² + \frac{\alpha}{2} \|\mathbf{x}^* - \mathbf{x}^*\|² = f(\mathbf{x}^*). \end{align*}$$

重新排列给出

$$ \|\mathbf{h}(\mathbf{x}^k)\|² \leq \frac{2}{k} \left[f(\mathbf{x}^*) - f(\mathbf{x}^k) - \frac{\alpha}{2} \|\mathbf{x}^k - \mathbf{x}^*\|²\right] \leq \frac{2}{k} \left[ f(\mathbf{x}^*) - m\right]. $$

因此，$\lim_{k \to \infty} \|\mathbf{h}(\mathbf{x}^k)\|² = 0$，由于 $\mathbf{h}$ 和 Frobenius 范数的连续性，这导致 $\|\mathbf{h}(\bar{\mathbf{x}})\|² = 0$，即 $\mathbf{h}(\bar{\mathbf{x}}) = \mathbf{0}$。换句话说，任何极限点 $\bar{\mathbf{x}}$ 都是可行的。

除了可行之外，$\bar{\mathbf{x}} \in \mathscr{X}$ 因为那个约束集是闭合的。因此，根据 $\mathscr{X}$ 的选择，我们有 $f(\mathbf{x}^*) \leq f(\bar{\mathbf{x}})$。此外，根据 $(*)$，我们得到

$$ f(\mathbf{x}^*) \leq f(\bar{\mathbf{x}}) + \frac{\alpha}{2} \|\bar{\mathbf{x}} - \mathbf{x}^*\|² \leq f(\mathbf{x}^*). $$

这只有在 $\|\bar{\mathbf{x}} - \mathbf{x}^*\|² = 0$ 或换句话说，$\bar{\mathbf{x}} = \mathbf{x}^*$ 的情况下才可能。这证明了引理。 $\square$

**引理** **(第二步：应用无约束必要条件)** 设 $\{\mathbf{x}^k\}_{k=1}^{+\infty}$ 是前面引理中的序列。

a) 对于足够大的 $k$，向量 $\nabla h_i(\mathbf{x}^k)$，$i=1,\ldots,\ell$，是线性无关的。

b) 设 $\mathbf{J}_{\mathbf{h}}(\mathbf{x})$ 为 $\mathbf{h}$ 的雅可比矩阵，即其行是 $\nabla h_i(\mathbf{x})^T$ 的向量，$i=1,\ldots,\ell$。那么

$$ \nabla f(\mathbf{x}^*) + \mathbf{J}_{\mathbf{h}}(\mathbf{x}^*)^T \blambda^* = \mathbf{0} $$

其中

$$ \blambda^* = - (\mathbf{J}_{\mathbf{h}}(\mathbf{x}^*) \, \mathbf{J}_{\mathbf{h}}^T(\mathbf{x}^*))^{-1} \mathbf{J}_{\mathbf{h}}^T(\mathbf{x}^*) \nabla f(\mathbf{x}^*). $$

$\flat$

*证明：* 根据前面的引理，对于足够大的 $k$，$\mathbf{x}^k$ 是 $F^k$ 的无约束局部极小值。因此，根据（无约束的）*一阶必要条件*，它成立，

$$ \nabla F^k(\mathbf{x}^k) = \mathbf{0}. $$

为了计算 $F^k$ 的梯度，我们注意到

$$ \|\mathbf{h}(\mathbf{x})\|² = \sum_{i=1}^\ell (h_i(\mathbf{x}))². $$

偏导数是

$$ \frac{\partial}{\partial x_j} \|\mathbf{h}(\mathbf{x})\|² = \sum_{i=1}^\ell \frac{\partial}{\partial x_j} (h_i(\mathbf{x}))² = \sum_{i=1}^\ell 2 h_i(\mathbf{x}) \frac{\partial h_i(\mathbf{x})}{\partial x_j}, $$

通过 *链式法则*。因此，以向量形式，

$$ \nabla \|\mathbf{h}(\mathbf{x})\|² = 2 \mathbf{J}_{\mathbf{h}}(\mathbf{x})^T \mathbf{h}(\mathbf{x}). $$

项 $\|\mathbf{x} - \mathbf{x}^*\|²$ 可以重写为二次函数

$$ \|\mathbf{x} - \mathbf{x}^*\|² = \frac{1}{2}\mathbf{x}^T (2 I_{d \times d}) \mathbf{x} - 2 (\mathbf{x}^*)^T \mathbf{x} + (\mathbf{x}^*)^T \mathbf{x}^*. $$

使用之前的公式，其中 $P = 2 I_{d \times d}$（这是一个对称矩阵），$\mathbf{q} = -2 \mathbf{x}^*$ 和 $r = (\mathbf{x}^*)^T \mathbf{x}^*$，我们得到

$$ \nabla \|\mathbf{x} - \mathbf{x}^*\|² = 2\mathbf{x} -2 \mathbf{x}^*. $$

所以，把所有东西放在一起，

$$ (**) \qquad \mathbf{0} = \nabla F^k(\mathbf{x}^k) = \nabla f(\mathbf{x}^k) + \mathbf{J}_{\mathbf{h}}(\mathbf{x}^k)^T (k \mathbf{h}(\mathbf{x}^k)) + \alpha(\mathbf{x}^k - \mathbf{x}^*). $$

根据之前的引理，$\mathbf{x}^k \to \mathbf{x}^*$，$\nabla f(\mathbf{x}^k) \to \nabla f(\mathbf{x}^*)$，以及 $\mathbf{J}_{\mathbf{h}}(\mathbf{x}^k) \to \mathbf{J}_{\mathbf{h}}(\mathbf{x}^*)$ 当 $k \to +\infty$。

因此，需要推导 $k \mathbf{h}(\mathbf{x}^k)$ 的极限。根据假设，$\mathbf{J}_{\mathbf{h}}(\mathbf{x}^*)^T$ 的列是线性无关的。这意味着对于任何单位向量 $\mathbf{z} \in \mathbb{R}^\ell$

$$ \mathbf{z}^T \mathbf{J}_{\mathbf{h}}(\mathbf{x}^*) \,\mathbf{J}_{\mathbf{h}}(\mathbf{x}^*)^T \mathbf{z} = \|\mathbf{J}_{\mathbf{h}}(\mathbf{x}^*)^T \mathbf{z}\|² > 0 $$

否则，我们将有 $\mathbf{J}_{\mathbf{h}}(\mathbf{x}^*)^T \mathbf{z} = \mathbf{0}$，这与线性无关的假设相矛盾。根据**极值定理**，存在 $\beta > 0$ 使得

$$ \mathbf{z}^T \mathbf{J}_{\mathbf{h}}(\mathbf{x}^*) \,\mathbf{J}_{\mathbf{h}}(\mathbf{x}^*)^T \mathbf{z} \geq \beta $$

对于所有单位向量 $\mathbf{z} \in \mathbb{R}^\ell$。由于 $\mathbf{J}_{\mathbf{h}}(\mathbf{x}^k) \to \mathbf{J}_{\mathbf{h}}(\mathbf{x}^*)$，根据之前的引理，对于足够大的 $k$ 和任意单位向量 $\mathbf{z} \in \mathbb{R}^\ell$，

$$ |\mathbf{z}^T \mathbf{J}_{\mathbf{h}}(\mathbf{x}^*)\, \mathbf{J}_{\mathbf{h}}(\mathbf{x}^*)^T \mathbf{z} - \mathbf{z}^T \mathbf{J}_{\mathbf{h}}(\mathbf{x}^k) \,\mathbf{J}_{\mathbf{h}}(\mathbf{x}^k)^T \mathbf{z}| \leq \| \mathbf{J}_{\mathbf{h}}(\mathbf{x}^*) \,\mathbf{J}_{\mathbf{h}}(\mathbf{x}^*)^T - \mathbf{J}_{\mathbf{h}}(\mathbf{x}^k) \,\mathbf{J}_{\mathbf{h}}(\mathbf{x}^k)^T \|_F \leq \frac{\beta}{2}. $$

这意味着

$$ \mathbf{z}^T \mathbf{J}_{\mathbf{h}}(\mathbf{x}^k) \,\mathbf{J}_{\mathbf{h}}(\mathbf{x}^k)^T \mathbf{z} \geq \frac{\beta}{2}, $$

因此，通过与上述相同的论证，当 $k$ 足够大时，$\mathbf{J}_{\mathbf{h}}(\mathbf{x}^k)^T$ 的列是线性无关的，并且 $\mathbf{J}_{\mathbf{h}}(\mathbf{x}^k) \,\mathbf{J}_{\mathbf{h}}(\mathbf{x}^k)^T$ 是可逆的。这证明了 a）。

回到 $(**)$，将两边乘以 $(\mathbf{J}_{\mathbf{h}}(\mathbf{x}^k)\, \mathbf{J}_{\mathbf{h}}(\mathbf{x}^k)^T)^{-1} \mathbf{J}_{\mathbf{h}}(\mathbf{x}^k)$，并取极限 $k \to +\infty$，经过重新排列后我们得到

$$ k \mathbf{h}(\mathbf{x}^k) \to - (\mathbf{J}_{\mathbf{h}}(\mathbf{x}^*) ^T \mathbf{J}_{\mathbf{h}}(\mathbf{x}^*))^{-1} \mathbf{J}_{\mathbf{h}}(\mathbf{x}^*) \nabla f(\mathbf{x}^*) = \blambda^*. $$

将其代入后得到

$$ \nabla f(\mathbf{x}^*) + \mathbf{J}_{\mathbf{h}}(\mathbf{x}^*)^T \blambda^* = \mathbf{0} $$

如所声称。这证明了 b）。$\square$

结合引理可以建立定理。$\square$

接下来，我们证明 *拉格朗日乘数：二阶充分条件*。再次，我们遵循 [[Ber](http://www.athenasc.com/nonlinbook.html)，第 4.2 节]。我们需要以下引理。证明可以跳过。

*证明思路:* 我们考虑一个略微修改的问题

$$\begin{align*} &\text{min} f(\mathbf{x}) + \frac{c}{2} \|\mathbf{h}(\mathbf{x})\|²\\ &\text{s.t.}\ \mathbf{h}(\mathbf{x}) = \mathbf{0} \end{align*}$$

它具有相同的局部极小值。将 *二阶充分条件* 应用到修改后问题的拉格朗日函数上，当 $c$ 足够大时，得到结果。

**引理** 设 $P$ 和 $Q$ 是 $\mathbb{R}^{n \times n}$ 中的对称矩阵。假设 $Q$ 是正半定矩阵，且 $P$ 在 $Q$ 的零空间上是正定的，即对于所有 $\mathbf{w} \neq \mathbf{0}$ 满足 $\mathbf{w}^T Q \mathbf{w} = \mathbf{0}$ 的 $\mathbf{w}$，都有 $\mathbf{w}^T P \mathbf{w} > 0$。那么存在一个标量 $\bar{c} \geq 0$，使得对于所有 $c > \bar{c}$，$P + c Q$ 是正定的。 $\flat$

*证明：*(引理)* 我们通过反证法来论证。假设存在一个非负的、递增的、发散的序列 $\{c_k\}_{k=1}^{+\infty}$ 和一个单位向量序列 $\{\mathbf{x}^k\}_{k=1}^{+\infty}$，

$$ (\mathbf{x}^k)^T (P + c_k Q) \mathbf{x}^k \leq 0 $$

对于所有 $k$。因为序列是有界的，所以它有一个极限点 $\bar{\mathbf{x}}$。不失一般性，假设 $\mathbf{x}^k \to \bar{\mathbf{x}}$，当 $k \to \infty$。由于 $c_k \to +\infty$ 并且根据假设 $(\mathbf{x}^k)^T Q \mathbf{x}^k \geq 0$，我们必须有 $(\bar{\mathbf{x}})^T Q \bar{\mathbf{x}} = 0$，否则 $(\mathbf{x}^k)^T (P + c_k Q) \mathbf{x}^k$ 将会发散。因此，根据陈述中的假设，必须有 $(\bar{\mathbf{x}})^T P \bar{\mathbf{x}} > 0$。这与上面的不等式相矛盾。 $\square$

*证明：*(拉格朗日乘数：二阶充分条件)* 我们考虑修改后的问题

$$\begin{align*} &\text{min} f(\mathbf{x}) + \frac{c}{2} \|\mathbf{h}(\mathbf{x})\|²\\ &\text{s.t.}\ \mathbf{h}(\mathbf{x}) = \mathbf{0}. \end{align*}$$

它具有与原问题相同的局部极小值，因为对于可行向量，目标函数中的附加项为零。这个额外的项将允许我们使用之前的引理。为了方便起见，我们定义

$$ g_c(\mathbf{x}) = f(\mathbf{x}) + \frac{c}{2} \|\mathbf{h}(\mathbf{x})\|². $$

修改后问题的拉格朗日函数是

$$ L_c(\mathbf{x}, \blambda) = g_c(\mathbf{x}) + \mathbf{h}(\mathbf{x})^T \blambda. $$

我们将应用 *二阶充分条件* 来最小化 $L_c$ 关于 $\mathbf{x}$ 的问题。我们只表示与变量 $\mathbf{x}$ 相关的 Hessian，记为 $\nabla²_{\mathbf{x},\mathbf{x}}$。

回忆一下在证明 *拉格朗日乘数：一阶必要条件* 时的内容，

$$ \nabla \|\mathbf{h}(\mathbf{x})\|² = 2 \mathbf{J}_{\mathbf{h}}(\mathbf{x})^T \mathbf{h}(\mathbf{x}). $$

为了计算该函数的 Hessian 矩阵，我们注意到

$$\begin{align*} \frac{\partial}{\partial x_i}\left( \frac{\partial}{\partial x_j} \|\mathbf{h}(\mathbf{x})\|²\right) &= \frac{\partial}{\partial x_i}\left( \sum_{k=1}^\ell 2 h_k(\mathbf{x}) \frac{\partial h_k(\mathbf{x})}{\partial x_j}\right)\\ &= 2 \sum_{k=1}^\ell\left( \frac{\partial h_k(\mathbf{x})}{\partial x_i} \frac{\partial h_k(\mathbf{x})}{\partial x_j} + h_k(\mathbf{x}) \frac{\partial² h_k(\mathbf{x})}{\partial x_i \partial x_j} \right)\\ &= 2 \left(\mathbf{J}_{\mathbf{h}}(\mathbf{x})^T \,\mathbf{J}_{\mathbf{h}}(\mathbf{x}) + \sum_{k=1}^\ell h_k(\mathbf{x}) \, \mathbf{H}_{h_k}(\mathbf{x})\right)_{i,j}. \end{align*}$$

因此

$$ \nabla_{\mathbf{x}} L_c(\mathbf{x}^*, \blambda^*) = \nabla f(\mathbf{x}^*) + c \mathbf{J}_{\mathbf{h}}(\mathbf{x}^*)^T \mathbf{h}(\mathbf{x}^*) + \mathbf{J}_{\mathbf{h}}(\mathbf{x}^*)^T \blambda^* $$

和

$$ \nabla²_{\mathbf{x},\mathbf{x}} L_c(\mathbf{x}^*, \blambda^*) = \mathbf{H}_{f}(\mathbf{x}^*) + c \left(\mathbf{J}_{\mathbf{h}}(\mathbf{x}^*)^T \,\mathbf{J}_{\mathbf{h}}(\mathbf{x}^*) + \sum_{k=1}^\ell h_k(\mathbf{x}^*) \, \mathbf{H}_{h_k}(\mathbf{x}^*)\right) + \sum_{k=1}^\ell \lambda^*_k \, \mathbf{H}_{h_k}(\mathbf{x}^*). $$

根据定理的假设，这可以简化为

$$ \nabla_{\mathbf{x}} L_c(\mathbf{x}^*, \blambda^*) = \mathbf{0} $$

和

$$ \nabla²_{\mathbf{x},\mathbf{x}} L_c(\mathbf{x}^*, \blambda^*) =\underbrace{\left\{ \mathbf{H}_{f}(\mathbf{x}^*) + \sum_{k=1}^\ell \lambda^*_k \, \mathbf{H}_{h_k}(\mathbf{x}^*) \right\}}_{P} + c \underbrace{\left\{\mathbf{J}_{\mathbf{h}}(\mathbf{x}^*)^T \,\mathbf{J}_{\mathbf{h}}(\mathbf{x}^*) \right\}}_{Q}, $$

其中，对于任意的 $\mathbf{v}$ 满足 $\mathbf{J}_{\mathbf{h}}(\mathbf{x}^*) \,\mathbf{v} = \mathbf{0}$（这本身意味着 $\mathbf{v}^T Q \mathbf{v} = \mathbf{0}$），进一步有 $\mathbf{v}^T P \mathbf{v} > 0$。此外，由于

$$ \mathbf{w}^T Q \mathbf{w} = \mathbf{w}^T \mathbf{J}_{\mathbf{h}}(\mathbf{x}^*)^T \,\mathbf{J}_{\mathbf{h}}(\mathbf{x}^*) \mathbf{w} = \left\|\mathbf{J}_{\mathbf{h}}(\mathbf{x}^*) \mathbf{w}\right\|² \geq 0 $$

对于任意的 $\mathbf{w} \in \mathbb{R}^d$。前一个引理允许我们取 $c$ 足够大，使得 $\nabla²_{\mathbf{x},\mathbf{x}} L_c(\mathbf{x}^*, \blambda^*) \succ \mathbf{0}$.

因此，对于该问题，在 $\mathbf{x}^*$ 处满足无约束的**二阶充分条件**。

$$ \min_{\mathbf{x} \in \mathbb{R}^d} L_c(\mathbf{x}, \blambda^*). $$

那就是存在 $\delta > 0$ 使得

$$ L_c(\mathbf{x}^*, \blambda^*) < L_c(\mathbf{x}, \blambda^*), \qquad \forall \mathbf{x} \in B_{\delta}(\mathbf{x}^*) \setminus \{\mathbf{x}^*\}. $$

将其限制在修改后的约束问题的可行向量上（即那些满足 $\mathbf{h}(\mathbf{x}) = \mathbf{0}$ 的向量），简化后得到

$$ f(\mathbf{x}^*) < f(\mathbf{x}), \qquad \forall \mathbf{x} \in \{\mathbf{x} : \mathbf{h}(\mathbf{x}) = \mathbf{0}\} \cap (B_{\delta}(\mathbf{x}^*) \setminus \{\mathbf{x}^*\}). $$

因此，$\mathbf{x}^*$ 是修改后的约束问题的严格局部极小值（进而也是原始约束问题的严格局部极小值）。这就完成了证明。$\square$

## 8.7.1\. 测验、解答、代码等.#

### 8.7.1.1\. 仅代码#

下面可以访问本章代码的交互式 Jupyter 笔记本（推荐使用 Google Colab）。鼓励您对其进行尝试。一些建议的计算练习散布其中。笔记本也可作为幻灯片查看。

+   [笔记本](https://github.com/MMiDS-textbook/MMiDS-textbook.github.io/blob/main/just_the_code/roch_mmids_chap_nn_notebook.ipynb) ([在 Colab 中打开](https://colab.research.google.com/github/MMiDS-textbook/MMiDS-textbook.github.io/blob/main/just_the_code/roch_mmids_chap_nn_notebook.ipynb))

+   [幻灯片](https://raw.githack.com/MMiDS-textbook/MMiDS-textbook.github.io/main/just_the_code/roch_mmids_chap_nn_notebook_slides.slides.html)

### 8.7.1.2\. 自我评估测验#

通过以下链接可以获取更全面的自我评估测验的网络版本。

+   [第 8.2 节](https://raw.githack.com/MMiDS-textbook/MMiDS-textbook.github.io/main/quizzes/self-assessment/quiz_8_2.html)

+   [第 8.3 节](https://raw.githack.com/MMiDS-textbook/MMiDS-textbook.github.io/main/quizzes/self-assessment/quiz_8_3.html)

+   [第 8.4 节](https://raw.githack.com/MMiDS-textbook/MMiDS-textbook.github.io/main/quizzes/self-assessment/quiz_8_4.html)

+   [第 8.5 节](https://raw.githack.com/MMiDS-textbook/MMiDS-textbook.github.io/main/quizzes/self-assessment/quiz_8_5.html)

### 8.7.1.3\. 自动测验#

本章自动生成的测验可以在此访问（推荐使用 Google Colab）。

+   [自动测验](https://github.com/MMiDS-textbook/MMiDS-textbook.github.io/blob/main/quizzes/auto_quizzes/roch-mmids-nn-autoquiz.ipynb) ([在 Colab 中打开](https://colab.research.google.com/github/MMiDS-textbook/MMiDS-textbook.github.io/blob/main/quizzes/auto_quizzes/roch-mmids-nn-autoquiz.ipynb))

### 8.7.1.4\. 奇数练习题的解答#

*(在 Claude、Gemini 和 ChatGPT 的帮助下)*

**E8.2.1** 向量化是通过堆叠 $A$ 的列获得的：$\text{vec}(A) = (2, 0, 1, -1)$。

**E8.2.3**

$$\begin{split} A \otimes B = \begin{pmatrix} 1 \cdot B & 2 \cdot B \\ -1 \cdot B & 0 \cdot B \end{pmatrix} = \begin{pmatrix} 3 & -1 & 6 & -2 \\ 2 & 1 & 4 & 2 \\ -3 & 1 & -6 & 2 \\ -2 & -1 & -4 & -2 \end{pmatrix}. \end{split}$$

**E8.2.5**

$$\begin{split} A \otimes B = \begin{pmatrix} 1 \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix} & 2 \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix} \\ 3 \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix} & 4 \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix} \end{pmatrix} = \begin{pmatrix} 5 & 6 & 10 & 12 \\ 7 & 8 & 14 & 16 \\ 15 & 18 & 20 & 24 \\ 21 & 24 & 28 & 32 \end{pmatrix}. \end{split}$$

**E8.2.7** 首先，计算 $\mathbf{f}$ 和 $\mathbf{g}$ 的雅可比矩阵：

$$\begin{split} J_{\mathbf{f}}(x, y) = \begin{pmatrix} 2x & 2y \\ y & x \end{pmatrix}, \quad J_{\mathbf{g}}(u, v) = \begin{pmatrix} v & u \\ 1 & 1 \end{pmatrix}. \end{split}$$

然后，根据链式法则，

$$\begin{split} J_{\mathbf{g} \circ \mathbf{f}}(1, 2) = J_{\mathbf{g}}(\mathbf{f}(1, 2)) \, J_{\mathbf{f}}(1, 2) = J_{\mathbf{g}}(5, 2) \, J_{\mathbf{f}}(1, 2) = \begin{pmatrix} 2 & 5 \\ 1 & 1 \end{pmatrix} \begin{pmatrix} 2 & 4 \\ 2 & 1 \end{pmatrix} = \begin{pmatrix} 14 & 13 \\ 4 & 5 \end{pmatrix}. \end{split}$$

**E8.2.9** 从 E8.2.5，我们有

$$\begin{split} A \otimes B = \begin{pmatrix} 5 & 6 & 10 & 12 \\ 7 & 8 & 14 & 16 \\ 15 & 18 & 20 & 24 \\ 21 & 24 & 28 & 32 \end{pmatrix} \end{split}$$

所以，

$$\begin{split}(A \otimes B)^T = \begin{pmatrix} 5 & 7 & 15 & 21 \\ 6 & 8 & 18 & 24 \\ 10 & 14 & 20 & 28 \\ 12 & 16 & 24 & 32 \end{pmatrix}. \end{split}$$

现在，

$$\begin{split} A^T = \begin{pmatrix} 1 & 3 \\ 2 & 4 \end{pmatrix}, \quad B^T = \begin{pmatrix} 5 & 7 \\ 6 & 8 \end{pmatrix} \end{split}$$

所以，

$$\begin{split} A^T \otimes B^T = \begin{pmatrix} 1 \begin{pmatrix} 5 & 7 \\ 6 & 8 \end{pmatrix} & 3 \begin{pmatrix} 5 & 7 \\ 6 & 8 \end{pmatrix} \\ 2 \begin{pmatrix} 5 & 7 \\ 6 & 8 \end{pmatrix} & 4 \begin{pmatrix} 5 & 7 \\ 6 & 8 \end{pmatrix} \end{pmatrix} = \begin{pmatrix} 5 & 7 & 15 & 21 \\ 6 & 8 & 18 & 24 \\ 10 & 14 & 20 & 28 \\ 12 & 16 & 24 & 32 \end{pmatrix}. \end{split}$$

我们看到 $(A \otimes B)^T = A^T \otimes B^T$，正如克罗内克积的性质所预期的那样。

**E8.2.11**

$$\begin{split} \nabla f(x, y, z) = \begin{pmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \\ \frac{\partial f}{\partial z} \end{pmatrix} = \begin{pmatrix} 2x \\ 2y \\ 2z \end{pmatrix}. \end{split}$$

所以，

$$\begin{split} \nabla f(1, 2, 3) = \begin{pmatrix} 2 \\ 4 \\ 6 \end{pmatrix}. \end{split}$$

**E8.2.13** 首先，计算 $f$ 的梯度以及 $\mathbf{g}$ 的雅可比矩阵：

$$\begin{split} \nabla f(x, y) = \begin{pmatrix} y \\ x \end{pmatrix}, \quad J_{\mathbf{g}}(x, y) = \begin{pmatrix} 2x & 0 \\ 0 & 2y \end{pmatrix}. \end{split}$$

然后，根据链式法则，

$$\begin{split} J_{f \circ \mathbf{g}}(1, 2) = \nabla f(\mathbf{g}(1, 2))^T \, J_{\mathbf{g}}(1, 2) = \nabla f(1, 4)^T \, J_{\mathbf{g}}(1, 2) = \begin{pmatrix} 4 & 1 \end{pmatrix} \begin{pmatrix} 2 & 0 \\ 0 & 4 \end{pmatrix} = \begin{pmatrix} 8 & 4 \end{pmatrix}. \end{split}$$

**E8.2.15** $\mathbf{g}$ 的雅可比矩阵为

$$\begin{split} J_{\mathbf{g}}(x, y, z) = \begin{pmatrix} f'(x) & 0 & 0 \\ 0 & f'(y) & 0 \\ 0 & 0 & f'(z) \end{pmatrix} \end{split}$$

其中 $f'(x) = \cos(x).$ 因此，

$$\begin{split} J_{\mathbf{g}}(\frac{\pi}{2}, \frac{\pi}{4}, \frac{\pi}{6}) = \begin{pmatrix} \cos(\frac{\pi}{2}) & 0 & 0 \\ 0 & \cos(\frac{\pi}{4}) & 0 \\ 0 & 0 & \cos(\frac{\pi}{6}) \end{pmatrix} = \begin{pmatrix} 0 & 0 & 0 \\ 0 & \frac{\sqrt{2}}{2} & 0 \\ 0 & 0 & \frac{\sqrt{3}}{2} \end{pmatrix}. \end{split}$$

**E8.3.1** $AB$ 的每个元素是 $A$ 的一个行与 $B$ 的一个列的点积，这需要 2 次乘法和 1 次加法。由于 $AB$ 有 4 个元素，所以总的操作数是 $4 \times 3 = 12$。

**E8.3.3** 我们有

$$ \ell(\hat{\mathbf{y}}) = \hat{y}_1² + \hat{y}_2² $$

因此，偏导数为 $\frac{\partial \ell}{\partial \hat{y}_1} = 2 \hat{y}_1$ 和 $\frac{\partial \ell}{\partial \hat{y}_2} = 2 \hat{y}_2$ 和

$$ J_{\ell}(\hat{\mathbf{y}}) = 2 \hat{\mathbf{y}}^T. $$

**E8.3.5** 从 E8.3.4，我们得到 $\mathbf{z}_1 = \mathbf{g}_0(\mathbf{z}_0) = (-1, -1)$。然后，$\mathbf{z}_2 = \mathbf{g}_1(\mathbf{z}_1) = (1, -2)$ 和 $f(\mathbf{x}) = \ell(\mathbf{z}_2) = 5$。根据 **链式法则**，

$$\begin{split} \nabla f(\mathbf{x})^T = J_f(\mathbf{x}) = J_{\ell}(\mathbf{z}_1) \,J_{\mathbf{g}_1}(\mathbf{z}_1) \,J_{\mathbf{g}_0}(\mathbf{z}_0) = 2 \mathbf{z}_2^T \begin{pmatrix} -1 & 0 \\ 1 & 1 \end{pmatrix} \begin{pmatrix} 1 & 2 \\ -1 & 0 \end{pmatrix} = (-10, -4) \begin{pmatrix} 1 & 2 \\ -1 & 0 \end{pmatrix} = (6, -20). \end{split}$$

**E8.3.7** 我们有

$$ g_1(\mathbf{z}_1, \mathbf{w}_1) = w_4 z_{1,1} + w_5 z_{1,2} $$

因此，通过计算所有偏导数，

$$ J_{g_1}(\mathbf{z}_1, \mathbf{w}_1) = \begin{pmatrix} w_4 & w_5 & z_{1,1} & z_{1,2} \end{pmatrix} = \begin{pmatrix} \mathbf{w}_1^T & \mathbf{z}_1^T \end{pmatrix} = \begin{pmatrix} W_1 & I_{1 \times 1} \otimes \mathbf{z}_1^T \end{pmatrix}. $$

使用文本中的符号，$A_1 = W_1$ 和 $B_1 = \mathbf{z}_1^T = I_{1 \times 1} \otimes \mathbf{z}_1^T$.

**E8.3.9** 我们有

$$ f(\mathbf{w}) = (w_4 (- w_0 + w_1) + w_5 (-w_2 + w_3))² $$

因此，根据 **链式法则**，偏导数为

$$ \frac{\partial f}{\partial w_0} = 2(w_4 (- w_0 + w_1) + w_5 (-w_2 + w_3)) (-w_4) $$$$ \frac{\partial f}{\partial w_1} = 2(w_4 (- w_0 + w_1) + w_5 (-w_2 + w_3)) (w_4) $$$$ \frac{\partial f}{\partial w_2} = 2(w_4 (- w_0 + w_1) + w_5 (-w_2 + w_3)) (-w_5) $$$$ \frac{\partial f}{\partial w_3} = 2(w_4 (- w_0 + w_1) + w_5 (-w_2 + w_3)) (w_5) $$$$ \frac{\partial f}{\partial w_4} = 2(w_4 (- w_0 + w_1) + w_5 (-w_2 + w_3)) (-w_0 + w_1) $$$$ \frac{\partial f}{\partial w_5} = 2(w_4 (- w_0 + w_1) + w_5 (-w_2 + w_3)) (-w_2 + w_3). $$

此外，根据 E8.3.7，

$$\begin{split} z_2 = g_1(\mathbf{z}_1, \mathbf{w}_1) = W_1 \mathbf{z}_1 = \begin{pmatrix} w_4 & w_5\end{pmatrix} \begin{pmatrix}- w_0 + w_1\\-w_2 + w_3\end{pmatrix} = w_4 (- w_0 + w_1) + w_5 (-w_2 + w_3). \end{split}$$

通过基本递归和 E8.3.3 和 E8.3.8 中的结果，

$$\begin{align*} J_f(\mathbf{w}) &= J_{\ell}(h(\mathbf{w})) \,J_{h}(\mathbf{w}) = 2 z_2 \begin{pmatrix} A_1 B_0 & B_1\end{pmatrix}\\ &= 2 (w_4 (- w_0 + w_1) + w_5 (-w_2 + w_3)) (-w_4, w_4, -w_5, w_5, -w_0 + w_1, -w_2 + w_3). \end{align*}$$

**E8.4.1** 完整的梯度下降步骤是：

$$ \frac{1}{5} \sum_{i=1}⁵ \nabla f_{\mathbf{x}_i, y_i}(w) = \frac{1}{5}((1, 2) + (-1, 1) + (0, -1) + (2, 0) + (1, 1)) = (\frac{3}{5}, \frac{3}{5}). $$

批大小为 2 的期望 SGD 步骤是：

$$ \mathbb{E} [\frac{1}{2} \sum_{i\in B} \nabla f_{\mathbf{x}_i, y_i}(w)] = \frac{1}{5} \sum_{i=1}⁵ \nabla f_{x_i, y_i}(w) = (\frac{3}{5}, \frac{3}{5}), $$

这与全梯度下降步骤相等，正如在“期望 SGD 步骤”引理中证明的那样。

**E8.4.3**

$$\begin{align*} \mathrm{KL}(\mathbf{p} \| \mathbf{q}) &= \sum_{i=1}³ p_i \log \frac{p_i}{q_i} \\ &= 0.2 \log \frac{0.2}{0.1} + 0.3 \log \frac{0.3}{0.4} + 0.5 \log \frac{0.5}{0.5} \\ &\approx 0.2 \cdot 0.6931 + 0.3 \cdot (-0.2877) + 0.5 \cdot 0 \\ &\approx 0.0525. \end{align*}$$

**E8.4.5** SGD 更新由以下给出

$$\begin{align*} w &\leftarrow w - \frac{\alpha}{|B|} \sum_{i \in B} \frac{\partial \ell}{\partial w}(w, b; x_i, y_i), \\ b &\leftarrow b - \frac{\alpha}{|B|} \sum_{i \in B} \frac{\partial \ell}{\partial b}(w, b; x_i, y_i). \end{align*}$$

将值代入，我们得到

$$\begin{align*} w &\leftarrow 1 - \frac{0.1}{2} (2 \cdot 2(2 \cdot 1 + 0 - 3) + 2 \cdot 1(1 \cdot 1 + 0 - 2)) = 1.3, \\ b &\leftarrow 0 - \frac{0.1}{2} (2(2 \cdot 1 + 0 - 3) + 2(1 \cdot 1 + 0 - 2)) = 0.3. \end{align*}$$

**E8.4.7**

$$\begin{align*} \nabla \ell(w; x, y) &= -\frac{y}{\sigma(wx)} \sigma'(wx) x + \frac{1-y}{1-\sigma(wx)} \sigma'(wx) x \\ &= -yx(1 - \sigma(wx)) + x(1-y)\sigma(wx) \\ &= x(\sigma(wx) - y). \end{align*}$$

我们使用了这样一个事实：$\sigma'(z) = \sigma(z)(1-\sigma(z))$.

**E8.4.9** 首先，我们计算 $\mathbf{z}_1 = W\mathbf{x} = \begin{pmatrix} 0 & 0 \\ 0 & 0 \\ 0 & 0 \end{pmatrix} (1, 2) = (0, 0, 0)$。然后，$\hat{\mathbf{y}} = \boldsymbol{\gamma}(\mathbf{z}_1) = (\frac{1}{3}, \frac{1}{3}, \frac{1}{3})$。从文本中，我们有：

$$\begin{split} \nabla f(\mathbf{w}) = (\boldsymbol{\gamma}(W\mathbf{x}) - \mathbf{y}) \otimes \mathbf{x} = (\hat{\mathbf{y}} - \mathbf{y}) \otimes \mathbf{x} = (\frac{1}{3}, \frac{1}{3}, -\frac{2}{3}) \otimes (1, 2) = \begin{pmatrix} \frac{1}{3} & \frac{2}{3} \\ \frac{1}{3} & \frac{2}{3} \\ -\frac{2}{3} & -\frac{4}{3} \end{pmatrix}. \end{split}$$

**E8.4.11** 首先，我们计算各个梯度：

$$\begin{align*} \nabla f_{\mathbf{x}_1, \mathbf{y}_1}(W) &= (\boldsymbol{\gamma}(W \mathbf{x}_1) - \mathbf{y}_1) \otimes x_1 = (\frac{1}{3}, \frac{1}{3}, -\frac{2}{3}) \otimes (1, 2) = \begin{pmatrix} \frac{1}{3} & \frac{2}{3} \\ \frac{1}{3} & \frac{2}{3} \\ -\frac{2}{3} & -\frac{4}{3} \end{pmatrix}, \\ \nabla f_{\mathbf{x}_2, \mathbf{y}_2}(W) &= (\boldsymbol{\gamma}(W\mathbf{x}_2) - \mathbf{y}_2) \otimes x_2 = (-\frac{2}{3}, \frac{1}{3}, \frac{1}{3}) \otimes (4, -1) = \begin{pmatrix} -\frac{8}{3} & \frac{2}{3} \\ \frac{4}{3} & -\frac{1}{3} \\ \frac{4}{3} & -\frac{1}{3} \end{pmatrix}. \end{align*}$$

然后，完整的梯度为：

$$\begin{split} \frac{1}{2} (\nabla f_{\mathbf{x}_1, \mathbf{y}_1}(W) + \nabla f_{\mathbf{x}_2, \mathbf{y}_2}(W)) = \frac{1}{2} \left(\begin{pmatrix} \frac{1}{3} & \frac{2}{3} \\ \frac{1}{3} & \frac{2}{3} \\ -\frac{2}{3} & -\frac{4}{3} \end{pmatrix} + \begin{pmatrix} -\frac{8}{3} & \frac{2}{3} \\ \frac{4}{3} & -\frac{1}{3} \\ \frac{4}{3} & -\frac{1}{3} \end{pmatrix}\right) = \begin{pmatrix} -\frac{11}{6} & \frac{2}{3} \\ \frac{5}{6} & \frac{1}{6} \\ \frac{1}{3} & -\frac{5}{6} \end{pmatrix}. \end{split}$$

**E8.4.13** 交叉熵损失由

$$ -\log(0.3) \approx 1.204. $$

**E8.5.1** $\sigma(1) = \frac{1}{1 + e^{-1}} \approx 0.73$ $\sigma(-1) = \frac{1}{1 + e^{1}} \approx 0.27$ $\sigma(2) = \frac{1}{1 + e^{-2}} \approx 0.88$

**E8.5.3** $\bsigma(\mathbf{z}) = (\bsigma(1), \bsigma(-1), \bsigma(2)) \approx (0.73, 0.27, 0.88)$ $\bsigma'(\mathbf{z}) = (\bsigma'(1), \bsigma'(-1), \bsigma'(2)) \approx (0.20, 0.20, 0.10)$

**E8.5.5** $W\mathbf{x} = \begin{pmatrix} -1 \\ 4 \end{pmatrix}$，因此 $\sigma(W\mathbf{x}) \approx (0.27, 0.98)$，$J_\bsigma(W\mathbf{x}) = \mathrm{diag}(\bsigma(W\mathbf{x}) \odot (1 - \bsigma(W\mathbf{x}))) \approx \begin{pmatrix} 0.20 & 0 \\ 0 & 0.02 \end{pmatrix}$

**E8.5.7** $\nabla H(\mathbf{y}, \mathbf{z}) = (-\frac{y_1}{z_1}, -\frac{y_2}{z_2}) = (-\frac{0}{0.3}, -\frac{1}{0.7}) \approx (0, -1.43)$

### 8.7.1.5\. 学习成果#

+   定义雅可比矩阵并使用它来计算向量值函数的微分。

+   陈述并应用广义链式法则来计算函数复合的雅可比矩阵。

+   使用矩阵的 Hadamard 和 Kronecker 积进行计算。

+   描述自动微分的用途及其相对于符号和数值微分的优势。

+   在 PyTorch 中实现自动微分以计算向量值函数的梯度。

+   推导多层递进函数的链式法则并将其应用于计算梯度。

+   从计算复杂性的角度比较和对比自动微分的前向和反向模式。

+   定义递进函数并识别它们的关键属性。

+   推导递进函数雅可比矩阵的基本递归公式。

+   实现反向传播算法以高效计算渐进函数的梯度。

+   从矩阵-向量乘数的数量来分析反向传播算法的计算复杂度。

+   描述随机梯度下降（SGD）算法，并解释它与标准梯度下降有何不同。

+   从损失函数的梯度推导出随机梯度下降的更新规则。

+   证明期望的 SGD 步长等于完整梯度下降步长。

+   在真实世界数据集上评估使用随机梯度下降（SGD）训练的模型性能。

+   定义多层感知器（MLP）的架构，并描述每个层中仿射映射和激活函数的作用。

+   使用对角矩阵和克罗内克积的性质计算 sigmoid 激活函数的雅可比矩阵。

+   在一个小型 MLP 示例中应用链式法则来计算损失函数相对于权重的梯度。

+   使用正向和反向传播，泛化具有任意层数的多层感知器（MLP）的梯度计算。

+   使用 PyTorch 实现神经网络的训练。

$\aleph$

### 8.7.1.1\. 仅代码#

下面可以访问包含本章代码的交互式 Jupyter 笔记本（推荐使用 Google Colab）。鼓励您对其进行实验。一些建议的计算练习散布在其中。笔记本也可以作为幻灯片查看。

+   [笔记本](https://github.com/MMiDS-textbook/MMiDS-textbook.github.io/blob/main/just_the_code/roch_mmids_chap_nn_notebook.ipynb) ([在 Colab 中打开](https://colab.research.google.com/github/MMiDS-textbook/MMiDS-textbook.github.io/blob/main/just_the_code/roch_mmids_chap_nn_notebook.ipynb))

+   [幻灯片](https://raw.githack.com/MMiDS-textbook/MMiDS-textbook.github.io/main/just_the_code/roch_mmids_chap_nn_notebook_slides.slides.html)

### 8.7.1.2\. 自我评估测验#

通过以下链接可以获取更全面的自我评估测验的网页版本。

+   [第 8.2 节](https://raw.githack.com/MMiDS-textbook/MMiDS-textbook.github.io/main/quizzes/self-assessment/quiz_8_2.html)

+   [第 8.3 节](https://raw.githack.com/MMiDS-textbook/MMiDS-textbook.github.io/main/quizzes/self-assessment/quiz_8_3.html)

+   [第 8.4 节](https://raw.githack.com/MMiDS-textbook/MMiDS-textbook.github.io/main/quizzes/self-assessment/quiz_8_4.html)

+   [第 8.5 节](https://raw.githack.com/MMiDS-textbook/MMiDS-textbook.github.io/main/quizzes/self-assessment/quiz_8_5.html)

### 8.7.1.3\. 自动测验#

可以在此处访问本章的自动生成的测验（推荐使用 Google Colab）。

+   [自动测验](https://github.com/MMiDS-textbook/MMiDS-textbook.github.io/blob/main/quizzes/auto_quizzes/roch-mmids-nn-autoquiz.ipynb) ([在 Colab 中打开](https://colab.research.google.com/github/MMiDS-textbook/MMiDS-textbook.github.io/blob/main/quizzes/auto_quizzes/roch-mmids-nn-autoquiz.ipynb))

### 8.7.1.4\. 奇数编号的预热练习的解答#

*(在 Claude、Gemini 和 ChatGPT 的帮助下)*

**E8.2.1** 向量化是通过堆叠 $A$ 的列得到的：$\text{vec}(A) = (2, 0, 1, -1)$。

**E8.2.3**

$$\begin{split} A \otimes B = \begin{pmatrix} 1 \cdot B & 2 \cdot B \\ -1 \cdot B & 0 \cdot B \end{pmatrix} = \begin{pmatrix} 3 & -1 & 6 & -2 \\ 2 & 1 & 4 & 2 \\ -3 & 1 & -6 & 2 \\ -2 & -1 & -4 & -2 \end{pmatrix}. \end{split}$$

**E8.2.5**

$$\begin{split} A \otimes B = \begin{pmatrix} 1 \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix} & 2 \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix} \\ 3 \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix} & 4 \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix} \end{pmatrix} = \begin{pmatrix} 5 & 6 & 10 & 12 \\ 7 & 8 & 14 & 16 \\ 15 & 18 & 20 & 24 \\ 21 & 24 & 28 & 32 \end{pmatrix}. \end{split}$$

**E8.2.7** 首先，计算 $\mathbf{f}$ 和 $\mathbf{g}$ 的雅可比矩阵：

$$\begin{split} J_{\mathbf{f}}(x, y) = \begin{pmatrix} 2x & 2y \\ y & x \end{pmatrix}, \quad J_{\mathbf{g}}(u, v) = \begin{pmatrix} v & u \\ 1 & 1 \end{pmatrix}. \end{split}$$

然后，根据链式法则，

$$\begin{split} J_{\mathbf{g} \circ \mathbf{f}}(1, 2) = J_{\mathbf{g}}(\mathbf{f}(1, 2)) \, J_{\mathbf{f}}(1, 2) = J_{\mathbf{g}}(5, 2) \, J_{\mathbf{f}}(1, 2) = \begin{pmatrix} 2 & 5 \\ 1 & 1 \end{pmatrix} \begin{pmatrix} 2 & 4 \\ 2 & 1 \end{pmatrix} = \begin{pmatrix} 14 & 13 \\ 4 & 5 \end{pmatrix}. \end{split}$$

**E8.2.9** 从 E8.2.5，我们有

$$\begin{split} A \otimes B = \begin{pmatrix} 5 & 6 & 10 & 12 \\ 7 & 8 & 14 & 16 \\ 15 & 18 & 20 & 24 \\ 21 & 24 & 28 & 32 \end{pmatrix} \end{split}$$

所以，

$$\begin{split}(A \otimes B)^T = \begin{pmatrix} 5 & 7 & 15 & 21 \\ 6 & 8 & 18 & 24 \\ 10 & 14 & 20 & 28 \\ 12 & 16 & 24 & 32 \end{pmatrix}. \end{split}$$

现在，

$$\begin{split} A^T = \begin{pmatrix} 1 & 3 \\ 2 & 4 \end{pmatrix}, \quad B^T = \begin{pmatrix} 5 & 7 \\ 6 & 8 \end{pmatrix} \end{split}$$

所以，

$$\begin{split} A^T \otimes B^T = \begin{pmatrix} 1 \begin{pmatrix} 5 & 7 \\ 6 & 8 \end{pmatrix} & 3 \begin{pmatrix} 5 & 7 \\ 6 & 8 \end{pmatrix} \\ 2 \begin{pmatrix} 5 & 7 \\ 6 & 8 \end{pmatrix} & 4 \begin{pmatrix} 5 & 7 \\ 6 & 8 \end{pmatrix} \end{pmatrix} = \begin{pmatrix} 5 & 7 & 15 & 21 \\ 6 & 8 & 18 & 24 \\ 10 & 14 & 20 & 28 \\ 12 & 16 & 24 & 32 \end{pmatrix}. \end{split}$$

我们看到 $(A \otimes B)^T = A^T \otimes B^T$，正如 Kronecker 积的性质所预期的那样。

**E8.2.11**

$$\begin{split} \nabla f(x, y, z) = \begin{pmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \\ \frac{\partial f}{\partial z} \end{pmatrix} = \begin{pmatrix} 2x \\ 2y \\ 2z \end{pmatrix}. \end{split}$$

因此，

$$\begin{split} \nabla f(1, 2, 3) = \begin{pmatrix} 2 \\ 4 \\ 6 \end{pmatrix}. \end{split}$$

**E8.2.13** 首先，计算 $f$ 的梯度以及 $\mathbf{g}$ 的雅可比矩阵：

$$\begin{split} \nabla f(x, y) = \begin{pmatrix} y \\ x \end{pmatrix}, \quad J_{\mathbf{g}}(x, y) = \begin{pmatrix} 2x & 0 \\ 0 & 2y \end{pmatrix}. \end{split}$$

然后，根据**链式法则**，

$$\begin{split} J_{f \circ \mathbf{g}}(1, 2) = \nabla f(\mathbf{g}(1, 2))^T \, J_{\mathbf{g}}(1, 2) = \nabla f(1, 4)^T \, J_{\mathbf{g}}(1, 2) = \begin{pmatrix} 4 & 1 \end{pmatrix} \begin{pmatrix} 2 & 0 \\ 0 & 4 \end{pmatrix} = \begin{pmatrix} 8 & 4 \end{pmatrix}. \end{split}$$

**E8.2.15** $\mathbf{g}$ 的雅可比矩阵为

$$\begin{split} J_{\mathbf{g}}(x, y, z) = \begin{pmatrix} f'(x) & 0 & 0 \\ 0 & f'(y) & 0 \\ 0 & 0 & f'(z) \end{pmatrix} \end{split}$$

其中 $f'(x) = \cos(x).$ 因此，

$$\begin{split} J_{\mathbf{g}}(\frac{\pi}{2}, \frac{\pi}{4}, \frac{\pi}{6}) = \begin{pmatrix} \cos(\frac{\pi}{2}) & 0 & 0 \\ 0 & \cos(\frac{\pi}{4}) & 0 \\ 0 & 0 & \cos(\frac{\pi}{6}) \end{pmatrix} = \begin{pmatrix} 0 & 0 & 0 \\ 0 & \frac{\sqrt{2}}{2} & 0 \\ 0 & 0 & \frac{\sqrt{3}}{2} \end{pmatrix}. \end{split}$$

**E8.3.1** 矩阵 $AB$ 的每个元素是 $A$ 的某一行与 $B$ 的某一列的点积，这需要 2 次乘法和 1 次加法。由于 $AB$ 有 4 个元素，所以总的操作次数是 $4 \times 3 = 12$。

**E8.3.3** 我们有

$$ \ell(\hat{\mathbf{y}}) = \hat{y}_1² + \hat{y}_2² $$

因此，偏导数为 $\frac{\partial \ell}{\partial \hat{y}_1} = 2 \hat{y}_1$ 和 $\frac{\partial \ell}{\partial \hat{y}_2} = 2 \hat{y}_2$ 和

$$ J_{\ell}(\hat{\mathbf{y}}) = 2 \hat{\mathbf{y}}^T. $$

**E8.3.5** 从 E8.3.4，我们得到 $\mathbf{z}_1 = \mathbf{g}_0(\mathbf{z}_0) = (-1, -1)$。然后，$\mathbf{z}_2 = \mathbf{g}_1(\mathbf{z}_1) = (1, -2)$ 且 $f(\mathbf{x}) = \ell(\mathbf{z}_2) = 5$。根据**链式法则**，

$$\begin{split} \nabla f(\mathbf{x})^T = J_f(\mathbf{x}) = J_{\ell}(\mathbf{z}_1) \,J_{\mathbf{g}_1}(\mathbf{z}_1) \,J_{\mathbf{g}_0}(\mathbf{z}_0) = 2 \mathbf{z}_2^T \begin{pmatrix} -1 & 0 \\ 1 & 1 \end{pmatrix} \begin{pmatrix} 1 & 2 \\ -1 & 0 \end{pmatrix} = (-10, -4) \begin{pmatrix} 1 & 2 \\ -1 & 0 \end{pmatrix} = (6, -20). \end{split}$$

**E8.3.7** 我们有

$$ g_1(\mathbf{z}_1, \mathbf{w}_1) = w_4 z_{1,1} + w_5 z_{1,2} $$

因此，通过计算所有偏导数，

$$ J_{g_1}(\mathbf{z}_1, \mathbf{w}_1) = \begin{pmatrix} w_4 & w_5 & z_{1,1} & z_{1,2} \end{pmatrix} = \begin{pmatrix} \mathbf{w}_1^T & \mathbf{z}_1^T \end{pmatrix} = \begin{pmatrix} W_1 & I_{1 \times 1} \otimes \mathbf{z}_1^T \end{pmatrix}. $$

使用文本中的符号，$A_1 = W_1$ 和 $B_1 = \mathbf{z}_1^T = I_{1 \times 1} \otimes \mathbf{z}_1^T$.

**E8.3.9** 我们有

$$ f(\mathbf{w}) = (w_4 (- w_0 + w_1) + w_5 (-w_2 + w_3))² $$

因此，根据**链式法则**，偏导数是

$$ \frac{\partial f}{\partial w_0} = 2(w_4 (- w_0 + w_1) + w_5 (-w_2 + w_3)) (-w_4) $$$$ \frac{\partial f}{\partial w_1} = 2(w_4 (- w_0 + w_1) + w_5 (-w_2 + w_3)) (w_4) $$$$ \frac{\partial f}{\partial w_2} = 2(w_4 (- w_0 + w_1) + w_5 (-w_2 + w_3)) (-w_5) $$$$ \frac{\partial f}{\partial w_3} = 2(w_4 (- w_0 + w_1) + w_5 (-w_2 + w_3)) (w_5) $$$$ \frac{\partial f}{\partial w_4} = 2(w_4 (- w_0 + w_1) + w_5 (-w_2 + w_3)) (-w_0 + w_1) $$$$ \frac{\partial f}{\partial w_5} = 2(w_4 (- w_0 + w_1) + w_5 (-w_2 + w_3)) (-w_2 + w_3). $$

此外，根据 E8.3.7，

$$\begin{split} z_2 = g_1(\mathbf{z}_1, \mathbf{w}_1) = W_1 \mathbf{z}_1 = \begin{pmatrix} w_4 & w_5\end{pmatrix} \begin{pmatrix}- w_0 + w_1\\-w_2 + w_3\end{pmatrix} = w_4 (- w_0 + w_1) + w_5 (-w_2 + w_3). \end{split}$$

通过基本递归和 E8.3.3 以及 E8.3.8 的结果，

$$\begin{align*} J_f(\mathbf{w}) &= J_{\ell}(h(\mathbf{w})) \,J_{h}(\mathbf{w}) = 2 z_2 \begin{pmatrix} A_1 B_0 & B_1\end{pmatrix}\\ &= 2 (w_4 (- w_0 + w_1) + w_5 (-w_2 + w_3)) (-w_4, w_4, -w_5, w_5, -w_0 + w_1, -w_2 + w_3). \end{align*}$$

**E8.4.1** 完整的梯度下降步骤是：

$$ \frac{1}{5} \sum_{i=1}⁵ \nabla f_{\mathbf{x}_i, y_i}(w) = \frac{1}{5}((1, 2) + (-1, 1) + (0, -1) + (2, 0) + (1, 1)) = (\frac{3}{5}, \frac{3}{5}). $$

批大小为 2 的期望随机梯度下降步骤是：

$$ \mathbb{E} [\frac{1}{2} \sum_{i\in B} \nabla f_{\mathbf{x}_i, y_i}(w)] = \frac{1}{5} \sum_{i=1}⁵ \nabla f_{x_i, y_i}(w) = (\frac{3}{5}, \frac{3}{5}), $$

这等于完整的梯度下降步骤，如“期望随机梯度下降步骤”引理中证明的。

**E8.4.3**

$$\begin{align*} \mathrm{KL}(\mathbf{p} \| \mathbf{q}) &= \sum_{i=1}³ p_i \log \frac{p_i}{q_i} \\ &= 0.2 \log \frac{0.2}{0.1} + 0.3 \log \frac{0.3}{0.4} + 0.5 \log \frac{0.5}{0.5} \\ &\approx 0.2 \cdot 0.6931 + 0.3 \cdot (-0.2877) + 0.5 \cdot 0 \\ &\approx 0.0525. \end{align*}$$

**E8.4.5** 随机梯度下降的更新由

$$\begin{align*} w &\leftarrow w - \frac{\alpha}{|B|} \sum_{i \in B} \frac{\partial \ell}{\partial w}(w, b; x_i, y_i), \\ b &\leftarrow b - \frac{\alpha}{|B|} \sum_{i \in B} \frac{\partial \ell}{\partial b}(w, b; x_i, y_i). \end{align*}$$

将值代入，我们得到

$$\begin{align*} w &\leftarrow 1 - \frac{0.1}{2} (2 \cdot 2(2 \cdot 1 + 0 - 3) + 2 \cdot 1(1 \cdot 1 + 0 - 2)) = 1.3, \\ b &\leftarrow 0 - \frac{0.1}{2} (2(2 \cdot 1 + 0 - 3) + 2(1 \cdot 1 + 0 - 2)) = 0.3. \end{align*}$$

**E8.4.7**

$$\begin{align*} \nabla \ell(w; x, y) &= -\frac{y}{\sigma(wx)} \sigma'(wx) x + \frac{1-y}{1-\sigma(wx)} \sigma'(wx) x \\ &= -yx(1 - \sigma(wx)) + x(1-y)\sigma(wx) \\ &= x(\sigma(wx) - y). \end{align*}$$

我们使用了$\sigma'(z) = \sigma(z)(1-\sigma(z))$的事实。

**E8.4.9** 首先，我们计算 $\mathbf{z}_1 = W\mathbf{x} = \begin{pmatrix} 0 & 0 \\ 0 & 0 \\ 0 & 0 \end{pmatrix} (1, 2) = (0, 0, 0)$。然后，$\hat{\mathbf{y}} = \boldsymbol{\gamma}(\mathbf{z}_1) = (\frac{1}{3}, \frac{1}{3}, \frac{1}{3})$。从文本中，我们有：

$$\begin{split} \nabla f(\mathbf{w}) = (\boldsymbol{\gamma}(W\mathbf{x}) - \mathbf{y}) \otimes \mathbf{x} = (\hat{\mathbf{y}} - \mathbf{y}) \otimes \mathbf{x} = (\frac{1}{3}, \frac{1}{3}, -\frac{2}{3}) \otimes (1, 2) = \begin{pmatrix} \frac{1}{3} & \frac{2}{3} \\ \frac{1}{3} & \frac{2}{3} \\ -\frac{2}{3} & -\frac{4}{3} \end{pmatrix}. \end{split}$$

**E8.4.11** 首先，我们计算各个梯度：

$$\begin{align*} \nabla f_{\mathbf{x}_1, \mathbf{y}_1}(W) &= (\boldsymbol{\gamma}(W \mathbf{x}_1) - \mathbf{y}_1) \otimes x_1 = (\frac{1}{3}, \frac{1}{3}, -\frac{2}{3}) \otimes (1, 2) = \begin{pmatrix} \frac{1}{3} & \frac{2}{3} \\ \frac{1}{3} & \frac{2}{3} \\ -\frac{2}{3} & -\frac{4}{3} \end{pmatrix}, \\ \nabla f_{\mathbf{x}_2, \mathbf{y}_2}(W) &= (\boldsymbol{\gamma}(W\mathbf{x}_2) - \mathbf{y}_2) \otimes x_2 = (-\frac{2}{3}, \frac{1}{3}, \frac{1}{3}) \otimes (4, -1) = \begin{pmatrix} -\frac{8}{3} & \frac{2}{3} \\ \frac{4}{3} & -\frac{1}{3} \\ \frac{4}{3} & -\frac{1}{3} \end{pmatrix}. \end{align*}$$

然后，完整的梯度是：

$$\begin{split} \frac{1}{2} (\nabla f_{\mathbf{x}_1, \mathbf{y}_1}(W) + \nabla f_{\mathbf{x}_2, \mathbf{y}_2}(W)) = \frac{1}{2} \left(\begin{pmatrix} \frac{1}{3} & \frac{2}{3} \\ \frac{1}{3} & \frac{2}{3} \\ -\frac{2}{3} & -\frac{4}{3} \end{pmatrix} + \begin{pmatrix} -\frac{8}{3} & \frac{2}{3} \\ \frac{4}{3} & -\frac{1}{3} \\ \frac{4}{3} & -\frac{1}{3} \end{pmatrix}\right) = \begin{pmatrix} -\frac{11}{6} & \frac{2}{3} \\ \frac{5}{6} & \frac{1}{6} \\ \frac{1}{3} & -\frac{5}{6} \end{pmatrix}. \end{split}$$

**E8.4.13** 交叉熵损失由以下公式给出

$$ -\log(0.3) \approx 1.204. $$

**E8.5.1** $\sigma(1) = \frac{1}{1 + e^{-1}} \approx 0.73$ $\sigma(-1) = \frac{1}{1 + e^{1}} \approx 0.27$ $\sigma(2) = \frac{1}{1 + e^{-2}} \approx 0.88$

**E8.5.3** $\bsigma(\mathbf{z}) = (\bsigma(1), \bsigma(-1), \bsigma(2)) \approx (0.73, 0.27, 0.88)$ $\bsigma'(\mathbf{z}) = (\bsigma'(1), \bsigma'(-1), \bsigma'(2)) \approx (0.20, 0.20, 0.10)$

**E8.5.5** $W\mathbf{x} = \begin{pmatrix} -1 \\ 4 \end{pmatrix}$，所以 $\sigma(W\mathbf{x}) \approx (0.27, 0.98)$，$J_\bsigma(W\mathbf{x}) = \mathrm{diag}(\bsigma(W\mathbf{x}) \odot (1 - \bsigma(W\mathbf{x}))) \approx \begin{pmatrix} 0.20 & 0 \\ 0 & 0.02 \end{pmatrix}$

**E8.5.7** $\nabla H(\mathbf{y}, \mathbf{z}) = (-\frac{y_1}{z_1}, -\frac{y_2}{z_2}) = (-\frac{0}{0.3}, -\frac{1}{0.7}) \approx (0, -1.43)$

### 8.7.1.5\. 学习成果#

+   定义雅可比矩阵并使用它来计算向量值函数的微分。

+   状态并应用广义链式法则来计算函数复合的雅可比矩阵。

+   进行矩阵的 Hadamard 积和克罗内克积的计算。

+   描述自动微分的用途及其与符号微分和数值微分相比的优势。

+   在 PyTorch 中实现自动微分以计算向量值函数的梯度。

+   推导多层渐进函数的链式法则，并将其应用于计算梯度。

+   比较自动微分的正向和反向模式的计算复杂度。

+   定义渐进函数并识别其关键属性。

+   推导渐进函数雅可比矩阵的基本递归公式。

+   实现反向传播算法以高效地计算渐进函数的梯度。

+   分析反向传播算法的计算复杂度，从矩阵-向量乘数的数量来考虑。

+   描述随机梯度下降（SGD）算法，并解释它与标准梯度下降的不同之处。

+   从损失函数的梯度推导出随机梯度下降的更新规则。

+   证明期望的 SGD 步骤等于完整梯度下降步骤。

+   评估在真实世界数据集上使用随机梯度下降训练的模型性能。

+   定义多层感知器（MLP）架构，并描述每一层中仿射映射和激活函数的作用。

+   利用对角矩阵和克罗内克积的性质计算 sigmoid 激活函数的雅可比矩阵。

+   在一个小型 MLP 示例中应用链式法则计算损失函数相对于权重的梯度。

+   将梯度计算推广到具有任意层数的 MLP，使用正向和反向传递。

+   使用 PyTorch 实现神经网络的训练。

$\aleph$

## 8.7.2. Additional sections#

### 8.7.2.1. 另一个例子：线性回归#

我们给出另一个关于渐进函数以及反向传播和随机梯度下降应用的实例。

**计算梯度** 当我们从分类的角度出发，激励了上一节中引入的框架，它也立即适用于回归设置。分类和回归都是监督学习的实例。

我们首先计算单个样本的梯度。这里 $\mathbf{x} \in \mathbb{R}^d$，但 $y$ 是一个实值结果变量。我们回顾线性回归的案例，其中损失函数是

$$ \ell(z) = (z - y)² $$

并且回归函数只有输入层和输出层，没有隐藏层（即 $L=0$），

$$ h(\mathbf{w}) = \sum_{j=1}^d w_{j} x_{j} = \mathbf{x}^T\mathbf{w}, $$

其中 $\mathbf{w} \in \mathbb{R}^{d}$ 是参数。回想一下，我们可以通过在输入中添加一个常数项（一个不依赖于输入的项）来包含一个 $1$。为了简化符号，我们假设如果需要，这个预处理已经完成。

最后，单个样本的目标函数是

$$ f(\mathbf{w}) = \ell(h(\mathbf{w})) = \left(\sum_{j=1}^d w_{j} x_{j} - y\right)² = \left(\mathbf{x}^T\mathbf{w} - y\right)²\. $$

使用前一小节中的符号，这种情况下的前向传播是：

*初始化：*

$$\mathbf{z}_0 := \mathbf{x}.$$

*前向层循环：*

$$\begin{align*} \hat{y} := z_1 := g_0(\mathbf{z}_0,\mathbf{w}_0) &= \sum_{j=1}^d w_{0,j} z_{0,j} = \mathbf{z}_0^T \mathbf{w}_0 \end{align*}$$$$\begin{align*} \begin{pmatrix} A_0 & B_0 \end{pmatrix} := J_{g_0}(\mathbf{z}_0,\mathbf{w}_0) &= ( w_{0,1},\ldots, w_{0,d},z_{0,1},\ldots,z_{0,d} )^T = \begin{pmatrix}\mathbf{w}_0^T & \mathbf{z}_0^T\end{pmatrix}, \end{align*}$$

其中 $\mathbf{w}_0 := \mathbf{w}$。

*损失：*

$$\begin{align*} z_2 &:= \ell(z_1) = (z_1 - y)²\\ p_2 &:= \frac{\mathrm{d}}{\mathrm{d} z_1} {\ell}(z_1) = 2 (z_1 - y). \end{align*}$$

反向传播是：

*反向层循环：*

$$\begin{align*} \mathbf{q}_0 := B_0^T p_1 &= 2 (z_1 - y) \, \mathbf{z}_0. \end{align*}$$

*输出：*

$$ \nabla f(\mathbf{w}) = \mathbf{q}_0. $$

正如我们之前提到的，实际上没有必要计算 $A_0$ 和 $\mathbf{p}_0$。

**`Advertising` 数据集和最小二乘解** 我们回到 `Advertising` 数据集。

```py
data = pd.read_csv('advertising.csv')
data.head() 
```

|  | Unnamed: 0 | TV | radio | newspaper | sales |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | 230.1 | 37.8 | 69.2 | 22.1 |
| 1 | 2 | 44.5 | 39.3 | 45.1 | 10.4 |
| 2 | 3 | 17.2 | 45.9 | 69.3 | 9.3 |
| 3 | 4 | 151.5 | 41.3 | 58.5 | 18.5 |
| 4 | 5 | 180.8 | 10.8 | 58.4 | 12.9 |

```py
n = len(data.index)
print(n) 
```

```py
200 
```

我们首先使用之前详细说明的最小二乘法来计算解。我们使用 `numpy.column_stack` 在特征向量中添加一列 $1$。

```py
TV = data['TV'].to_numpy()
radio = data['radio'].to_numpy()
newspaper = data['newspaper'].to_numpy()
sales = data['sales'].to_numpy()
features = np.stack((TV, radio, newspaper), axis=-1)
A = np.column_stack((np.ones(n), features))
coeff = mmids.ls_by_qr(A, sales)
print(coeff) 
```

```py
[ 2.93888937e+00  4.57646455e-02  1.88530017e-01 -1.03749304e-03] 
```

均方误差（MSE）是：

```py
np.mean((A @ coeff - sales)**2) 
```

```py
2.7841263145109365 
```

**使用 PyTorch 解决问题** 我们将使用 PyTorch 来实现前面的方法。我们首先将数据转换为 PyTorch 张量。然后使用 `torch.utils.data.TensorDataset` 创建数据集。最后，`torch.utils.data.DataLoader` 提供了加载数据以进行批处理的工具。我们取大小为 `BATCH_SIZE = 64` 的迷你批次，并在每次传递时对样本进行随机排列（使用选项 `shuffle=True`）。

```py
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

features_tensor = torch.tensor(features, dtype=torch.float32)
sales_tensor = torch.tensor(sales, dtype=torch.float32).view(-1, 1)

BATCH_SIZE = 64
train_dataset = TensorDataset(features_tensor, sales_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) 
```

现在我们构建我们的模型。它只是一个从 $\mathbb{R}³$ 到 $\mathbb{R}$ 的仿射映射。请注意，没有必要通过添加 $1$ 来预先处理输入。PyTorch（除非选择选项[`bias=False`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)）会自动添加一个常数项（或“偏置变量”）。

```py
model = nn.Sequential(
    nn.Linear(3, 1)  # 3 input features, 1 output value
) 
```

最后，我们准备好在我们的损失函数上运行我们选择的优化方法，这些方法将在下面指定。有许多[优化器](https://pytorch.org/docs/stable/optim.html#algorithms)可供选择。（参见这篇[文章](https://hackernoon.com/demystifying-different-variants-of-gradient-descent-optimization-algorithm-19ae9ba2e9bc)，了解许多常见优化器的简要说明。）这里我们使用 SGD 作为优化器。损失函数是均方误差（MSE）。快速教程在这里[这里](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)。

```py
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-5) 
```

选择合适的数据遍历次数（即 epoch）需要一些实验。这里 $10⁴$ 就足够了。但为了节省时间，我们只运行 $100$ 个 epoch。正如您将从结果中看到的那样，这并不够。在每次遍历中，我们计算当前模型的输出，使用 `backward()` 获取梯度，然后使用 `step()` 执行下降更新。我们还需要首先重置梯度（否则它们会默认累加）。

```py
epochs = 100
for epoch in range(epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}") 
```

最终的参数和损失如下：

<details class="hide above-input"><summary aria-label="Toggle hidden content">显示代码单元格源代码 隐藏代码单元格源代码</summary>

```py
weights = model[0].weight.detach().numpy()
bias = model[0].bias.detach().numpy()
print("Weights:", weights)
print("Bias:", bias) 
```</details>

```py
Weights: [[0.05736413 0.11314777 0.08020781]]
Bias: [-0.02631279] 
```

<details class="hide above-input"><summary aria-label="Toggle hidden content">显示代码单元格源代码 隐藏代码单元格源代码</summary>

```py
# Evaluate the model
model.eval()
with torch.no_grad():
    total_loss = 0
    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()

    print(f"Mean Squared Error on Training Set: {total_loss  /  len(train_loader)}") 
```</details>

```py
Mean Squared Error on Training Set: 7.885213494300842 
```

### 8.7.2.2\. 卷积神经网络#

我们回到 Fashion MNIST 数据集。使用专门为图像设计的神经网络，即[卷积神经网络](https://cs231n.github.io/convolutional-networks/)，可以获得比我们之前更好的结果。根据[Wikipedia](https://en.wikipedia.org/wiki/Convolutional_neural_network)：

> 在深度学习中，卷积神经网络（CNN，或 ConvNet）是一类深度神经网络，最常用于分析视觉图像。它们也被称为基于共享权重架构和平移不变性特征的平移不变或空间不变的人工神经网络（SIANN）。

更多背景信息可以在斯坦福大学的[CS231n](http://cs231n.github.io/)提供的这个优秀的[模块](http://cs231n.github.io/convolutional-networks/)中找到。我们的卷积神经网络将是由[卷积层](http://cs231n.github.io/convolutional-networks/#conv)和[池化层](http://cs231n.github.io/convolutional-networks/#pool)组成的。

**CHAT & LEARN** 卷积神经网络（CNNs）在图像分类中非常强大。请你的喜欢的 AI 聊天机器人解释 CNN 的基本概念，包括卷积层和池化层。 $\ddagger$

```py
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

train_dataset = datasets.FashionMNIST(root='./data', train=True, 
                               download=True, transform=transforms.ToTensor())
test_dataset = datasets.FashionMNIST(root='./data', train=False, 
                              download=True, transform=transforms.ToTensor())

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() 
                      else ("mps" if torch.backends.mps.is_available() 
                            else "cpu"))
print("Using device:", device) 
```

```py
Using device: mps 
```

新的模型如下。

```py
model = nn.Sequential(
    # First convolution, operating upon a 28x28 image
    nn.Conv2d(1, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    # Second convolution, operating upon a 14x14 image
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    # Third convolution, operating upon a 7x7 image
    nn.Conv2d(32, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    # Flatten the tensor
    nn.Flatten(),

    # Fully connected layer
    nn.Linear(32 * 3 * 3, 10),
).to(device) 
```

我们进行训练和测试。

```py
loss_fn = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters()) 
```

```py
mmids.training_loop(train_loader, model, loss_fn, optimizer, device) 
```

```py
Epoch 1/3 
```

```py
Epoch 2/3 
```

```py
Epoch 3/3 
```

```py
mmids.test(test_loader, model, loss_fn, device) 
```

```py
Test error: 88.6% accuracy 
```

注意更高的准确性。

最后，我们尝试原始的 MNIST 数据集。我们使用相同的 CNN。

```py
train_dataset = datasets.MNIST(root='./data', train=True, 
                                      download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='./data', train=False, 
                                     download=True, transform=transforms.ToTensor())

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) 
```

```py
loss_fn = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters()) 
```

```py
mmids.training_loop(train_loader, model, loss_fn, optimizer, device) 
```

```py
Epoch 1/3 
```

```py
Epoch 2/3 
```

```py
Epoch 3/3 
```

```py
mmids.test(test_loader, model, loss_fn, device) 
```

```py
Test error: 98.6% accuracy 
```

注意在这个（容易的 - 如我们所见）数据集上的非常高的准确性。

### 8.7.2.1. 另一个例子：线性回归#

我们给出了渐进函数和反向传播及随机梯度下降应用的另一个具体例子。

**计算梯度** 虽然我们是从分类的角度来激励上一节中引入的框架，但它也立即适用于回归设置。分类和回归都是监督学习的实例。

我们首先计算单个样本的梯度。这里 $\mathbf{x} \in \mathbb{R}^d$ 仍然适用，但 $y$ 是一个实值结果变量。我们回顾线性回归的案例，其中损失函数是

$$ \ell(z) = (z - y)² $$

并且回归函数只有输入层和输出层，没有隐藏层（即 $L=0$），

$$ h(\mathbf{w}) = \sum_{j=1}^d w_{j} x_{j} = \mathbf{x}^T\mathbf{w}, $$

其中 $\mathbf{w} \in \mathbb{R}^{d}$ 是参数。回想一下，我们可以通过在输入中添加一个常数项（不依赖于输入的项）来包含一个常数项。为了简化符号，我们假设如果需要，已经完成了预处理。

最后，单个样本的目标函数是

$$ f(\mathbf{w}) = \ell(h(\mathbf{w})) = \left(\sum_{j=1}^d w_{j} x_{j} - y\right)² = \left(\mathbf{x}^T\mathbf{w} - y\right)²\. $$

使用前一小节中的符号，本例中的前向传播如下：

*初始化:*

$$\mathbf{z}_0 := \mathbf{x}.$$

*前向层循环:*

$$\begin{align*} \hat{y} := z_1 := g_0(\mathbf{z}_0,\mathbf{w}_0) &= \sum_{j=1}^d w_{0,j} z_{0,j} = \mathbf{z}_0^T \mathbf{w}_0 \end{align*}$$$$\begin{align*} \begin{pmatrix} A_0 & B_0 \end{pmatrix} := J_{g_0}(\mathbf{z}_0,\mathbf{w}_0) &= ( w_{0,1},\ldots, w_{0,d},z_{0,1},\ldots,z_{0,d} )^T = \begin{pmatrix}\mathbf{w}_0^T & \mathbf{z}_0^T\end{pmatrix}, \end{align*}$$

其中 $\mathbf{w}_0 := \mathbf{w}$。

*损失:*

$$\begin{align*} z_2 &:= \ell(z_1) = (z_1 - y)²\\ p_2 &:= \frac{\mathrm{d}}{\mathrm{d} z_1} {\ell}(z_1) = 2 (z_1 - y). \end{align*}$$

反向传播如下：

*反向层循环:*

$$\begin{align*} \mathbf{q}_0 := B_0^T p_1 &= 2 (z_1 - y) \, \mathbf{z}_0. \end{align*}$$

*输出:*

$$ \nabla f(\mathbf{w}) = \mathbf{q}_0. $$

正如我们之前提到的，实际上没有必要计算 $A_0$ 和 $\mathbf{p}_0$。

**`Advertising` 数据集和最小二乘解** 我们回到 `Advertising` 数据集。

```py
data = pd.read_csv('advertising.csv')
data.head() 
```

|  | 未命名: 0 | 电视 | 收音机 | 报纸 | 销售额 |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | 230.1 | 37.8 | 69.2 | 22.1 |
| 1 | 2 | 44.5 | 39.3 | 45.1 | 10.4 |
| 2 | 3 | 17.2 | 45.9 | 69.3 | 9.3 |
| 3 | 4 | 151.5 | 41.3 | 58.5 | 18.5 |
| 4 | 5 | 180.8 | 10.8 | 58.4 | 12.9 |

```py
n = len(data.index)
print(n) 
```

```py
200 
```

我们首先使用我们之前详细说明的平方最小二乘法来计算解决方案。我们使用 `numpy.column_stack` 向特征向量添加一列。

```py
TV = data['TV'].to_numpy()
radio = data['radio'].to_numpy()
newspaper = data['newspaper'].to_numpy()
sales = data['sales'].to_numpy()
features = np.stack((TV, radio, newspaper), axis=-1)
A = np.column_stack((np.ones(n), features))
coeff = mmids.ls_by_qr(A, sales)
print(coeff) 
```

```py
[ 2.93888937e+00  4.57646455e-02  1.88530017e-01 -1.03749304e-03] 
```

均方误差（MSE）是：

```py
np.mean((A @ coeff - sales)**2) 
```

```py
2.7841263145109365 
```

**使用 PyTorch 解决问题** 我们将使用 PyTorch 来实现前面的方法。我们首先将数据转换为 PyTorch 张量。然后使用 `torch.utils.data.TensorDataset` 创建数据集。最后，`torch.utils.data.DataLoader` 提供了用于批量加载数据的实用工具。我们取大小为 `BATCH_SIZE = 64` 的迷你批次，并在每一轮中对样本进行随机排列（使用选项 `shuffle=True`）。

```py
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

features_tensor = torch.tensor(features, dtype=torch.float32)
sales_tensor = torch.tensor(sales, dtype=torch.float32).view(-1, 1)

BATCH_SIZE = 64
train_dataset = TensorDataset(features_tensor, sales_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) 
```

现在我们构建我们的模型。它只是一个从 $\mathbb{R}³$ 到 $\mathbb{R}$ 的仿射映射。请注意，没有必要通过添加 $1$ 来预处理输入。PyTorch（除非选择选项 `bias=False`）会自动添加一个常数项（或“偏置变量”）。

```py
model = nn.Sequential(
    nn.Linear(3, 1)  # 3 input features, 1 output value
) 
```

最后，我们准备好在我们的损失函数上运行我们选择的优化方法，这些方法将在下面指定。有许多可用的[优化器](https://pytorch.org/docs/stable/optim.html#algorithms)。（参见这篇[文章](https://hackernoon.com/demystifying-different-variants-of-gradient-descent-optimization-algorithm-19ae9ba2e9bc)，了解许多常见优化器的简要说明。）在这里，我们使用 SGD 作为优化器。损失函数是均方误差（MSE）。快速教程[在这里](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)。

```py
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-5) 
```

选择合适的数据遍历次数（即训练轮数）需要一些实验。这里 $10⁴$ 足够了。但为了节省时间，我们只运行 $100$ 轮。正如您将从结果中看到的那样，这并不够。在每一轮中，我们计算当前模型的输出，使用 `backward()` 获取梯度，然后使用 `step()` 执行下降更新。我们还需要首先重置梯度（否则它们会默认累加）。

```py
epochs = 100
for epoch in range(epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}") 
```

最终参数和损失如下：

<details class="hide above-input"><summary aria-label="Toggle hidden content">显示代码单元格源代码 隐藏代码单元格源代码</summary>

```py
weights = model[0].weight.detach().numpy()
bias = model[0].bias.detach().numpy()
print("Weights:", weights)
print("Bias:", bias) 
```

```py
Weights: [[0.05736413 0.11314777 0.08020781]]
Bias: [-0.02631279] 
```

<details class="hide above-input"><summary aria-label="Toggle hidden content">显示代码单元格源代码 隐藏代码单元格源代码</summary>

```py
# Evaluate the model
model.eval()
with torch.no_grad():
    total_loss = 0
    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()

    print(f"Mean Squared Error on Training Set: {total_loss  /  len(train_loader)}") 
```

```py
Mean Squared Error on Training Set: 7.885213494300842 
```

### 8.7.2.2\. 卷积神经网络#

我们回到 Fashion MNIST 数据集。使用专门针对图像的神经网络，即[卷积神经网络](https://cs231n.github.io/convolutional-networks/)，可以做得比我们之前更好。从[Wikipedia](https://en.wikipedia.org/wiki/Convolutional_neural_network)：

> 在深度学习中，卷积神经网络（CNN，或 ConvNet）是一类深度神经网络，最常用于分析视觉图像。它们也被称为基于共享权重架构和平移不变性特征的平移不变或空间不变人工神经网络（SIANN）。

更多背景信息可以在斯坦福的[CS231n](http://cs231n.github.io/)的这篇优秀的[模块](http://cs231n.github.io/convolutional-networks/)中找到。我们的 CNN 将是[卷积层](http://cs231n.github.io/convolutional-networks/#conv)和[池化层](http://cs231n.github.io/convolutional-networks/#pool)的组合。

**CHAT & LEARN** 卷积神经网络（CNNs）在图像分类方面非常强大。请你的 AI 聊天机器人解释 CNN 的基本概念，包括卷积层和池化层。$\ddagger$

```py
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

train_dataset = datasets.FashionMNIST(root='./data', train=True, 
                               download=True, transform=transforms.ToTensor())
test_dataset = datasets.FashionMNIST(root='./data', train=False, 
                              download=True, transform=transforms.ToTensor())

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() 
                      else ("mps" if torch.backends.mps.is_available() 
                            else "cpu"))
print("Using device:", device) 
```

```py
Using device: mps 
```

新模型如下。

```py
model = nn.Sequential(
    # First convolution, operating upon a 28x28 image
    nn.Conv2d(1, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    # Second convolution, operating upon a 14x14 image
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    # Third convolution, operating upon a 7x7 image
    nn.Conv2d(32, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    # Flatten the tensor
    nn.Flatten(),

    # Fully connected layer
    nn.Linear(32 * 3 * 3, 10),
).to(device) 
```

我们进行训练和测试。

```py
loss_fn = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters()) 
```

```py
mmids.training_loop(train_loader, model, loss_fn, optimizer, device) 
```

```py
Epoch 1/3 
```

```py
Epoch 2/3 
```

```py
Epoch 3/3 
```

```py
mmids.test(test_loader, model, loss_fn, device) 
```

```py
Test error: 88.6% accuracy 
```

注意更高的准确率。

最后，我们尝试了原始的 MNIST 数据集。我们使用了相同的卷积神经网络。

```py
train_dataset = datasets.MNIST(root='./data', train=True, 
                                      download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='./data', train=False, 
                                     download=True, transform=transforms.ToTensor())

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) 
```

```py
loss_fn = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters()) 
```

```py
mmids.training_loop(train_loader, model, loss_fn, optimizer, device) 
```

```py
Epoch 1/3 
```

```py
Epoch 2/3 
```

```py
Epoch 3/3 
```

```py
mmids.test(test_loader, model, loss_fn, device) 
```

```py
Test error: 98.6% accuracy 
```

注意在这个（容易的——结果证明）数据集上的非常高的准确率。

## 8.7.3\. 其他证明#

**拉格朗日乘数条件证明** 我们首先证明*拉格朗日乘数：一阶必要条件*。我们遵循优秀的教科书[[Ber](http://www.athenasc.com/nonlinbook.html)，第 4.1 节]。证明使用了雅可比矩阵的概念。

*证明思路：* 我们通过惩罚目标函数中的约束条件将问题简化为一个无约束优化问题。然后我们应用无约束的*一阶必要条件*。

*证明：* *(拉格朗日乘数：一阶必要条件)* 我们通过惩罚目标函数中的约束条件将问题简化为一个无约束优化问题。我们还添加了一个正则化项，以确保$\mathbf{x}^*$是邻域中唯一的局部最小值。具体来说，对于每个非负整数$k$，考虑以下目标函数

$$ F^k(\mathbf{x}) = f(\mathbf{x}) + \frac{k}{2} \|\mathbf{h}(\mathbf{x})\|² + \frac{\alpha}{2} \|\mathbf{x} - \mathbf{x}^*\|² $$

对于某个正常数$\alpha > 0$。注意，随着$k$的增大，惩罚变得更为显著，因此，强制约束变得更为可取。证明分几个步骤进行。

我们首先考虑一个版本，它最小化$F^k$，同时约束它位于$\mathbf{x}^*$的邻域内。因为$\mathbf{x}^*$是受$\mathbf{h}(\mathbf{x}) = \mathbf{0}$约束的$f$的局部最小值，存在$\delta > 0$，使得对于所有在$\mathbf{x}^*$邻域内的可行$\mathbf{x}$，都有$f(\mathbf{x}^*)\leq f(\mathbf{x})$。

$$ \mathscr{X} = B_{\delta}(\mathbf{x}^*) = \{\mathbf{x}:\|\mathbf{x} - \mathbf{x}^*\| \leq \delta\}. $$

**引理** **(步骤 I：在 $\mathbf{x}^*$ 的邻域内求解惩罚问题)** 对于 $k \geq 1$，设 $\mathbf{x}^k$ 是最小化问题的全局最小值

$$ \min_{\mathbf{x} \in \mathscr{X}} F^k(\mathbf{x}). $$

a) 序列 $\{\mathbf{x}^k\}_{k=1}^{+\infty}$ 收敛到 $\mathbf{x}^*$。

b) 对于足够大的 $k$，$\mathbf{x}^k$ 是目标函数 $F^k$ 的局部最小值，没有任何约束。

$\flat$

*证明* 集合 $\mathscr{X}$ 是闭合且有界的，且 $F^k$ 是连续的。因此，根据**极值定理**，序列 $\{\mathbf{x}^k\}_{k=1}^{+\infty}$ 是有定义的。设 $\bar{\mathbf{x}}$ 是 $\{\mathbf{x}^k\}_{k=1}^{+\infty}$ 的任意极限点。我们证明 $\bar{\mathbf{x}} = \mathbf{x}^*$。这将意味着 a)。这也意味着 b)，因为当 $k$ 足够大时，$\mathbf{x}^k$ 必须是 $\mathscr{X}$ 的内点。

设 $-\infty < m \leq M < + \infty$ 是 $\mathscr{X}$ 上函数 $f$ 的最小值和最大值，根据**极值定理**存在。那么，对于所有 $k$，根据 $\mathbf{x}^k$ 的定义以及 $\mathbf{x}^*$ 是可行解的事实

$$\begin{align*} (*) \qquad f(\mathbf{x}^k) &+ \frac{k}{2} \|\mathbf{h}(\mathbf{x}^k)\|² + \frac{\alpha}{2} \|\mathbf{x}^k - \mathbf{x}^*\|²\\ &\leq f(\mathbf{x}^*) + \frac{k}{2} \|\mathbf{h}(\mathbf{x}^*)\|² + \frac{\alpha}{2} \|\mathbf{x}^* - \mathbf{x}^*\|² = f(\mathbf{x}^*). \end{align*}$$

重新排列得到

$$ \|\mathbf{h}(\mathbf{x}^k)\|² \leq \frac{2}{k} \left[f(\mathbf{x}^*) - f(\mathbf{x}^k) - \frac{\alpha}{2} \|\mathbf{x}^k - \mathbf{x}^*\|²\right] \leq \frac{2}{k} \left[ f(\mathbf{x}^*) - m\right]. $$

因此，$\lim_{k \to \infty} \|\mathbf{h}(\mathbf{x}^k)\|² = 0$，根据 $\mathbf{h}$ 和 Frobenius 范数的连续性，这意味着 $\|\mathbf{h}(\bar{\mathbf{x}})\|² = 0$，即 $\mathbf{h}(\bar{\mathbf{x}}) = \mathbf{0}$。换句话说，任何极限点 $\bar{\mathbf{x}}$ 都是可行解。

除了是可行解外，$\bar{\mathbf{x}} \in \mathscr{X}$ 因为该约束集是闭合的。因此，根据 $\mathscr{X}$ 的选择，我们有 $f(\mathbf{x}^*) \leq f(\bar{\mathbf{x}})$。此外，根据 $(*)$，我们得到

$$ f(\mathbf{x}^*) \leq f(\bar{\mathbf{x}}) + \frac{\alpha}{2} \|\bar{\mathbf{x}} - \mathbf{x}^*\|² \leq f(\mathbf{x}^*). $$

这只有在 $\|\bar{\mathbf{x}} - \mathbf{x}^*\|² = 0$ 或换句话说，$\bar{\mathbf{x}} = \mathbf{x}^*$ 的情况下才可能。这证明了引理。 $\square$

**引理** **(步骤 II：应用无约束必要条件)** 设 $\{\mathbf{x}^k\}_{k=1}^{+\infty}$ 是前一个引理中的序列。

a) 对于足够大的 $k$，向量 $\nabla h_i(\mathbf{x}^k)$，$i=1,\ldots,\ell$，是线性无关的。

b) 设 $\mathbf{J}_{\mathbf{h}}(\mathbf{x})$ 为 $\mathbf{h}$ 的雅可比矩阵，即其行向量是 $\nabla h_i(\mathbf{x})^T$，$i=1,\ldots,\ell$ 的矩阵。然后

$$ \nabla f(\mathbf{x}^*) + \mathbf{J}_{\mathbf{h}}(\mathbf{x}^*)^T \blambda^* = \mathbf{0} $$

其中

$$ \blambda^* = - (\mathbf{J}_{\mathbf{h}}(\mathbf{x}^*) \, \mathbf{J}_{\mathbf{h}}^T(\mathbf{x}^*))^{-1} \mathbf{J}_{\mathbf{h}}^T(\mathbf{x}^*) \nabla f(\mathbf{x}^*). $$

$\flat$

*证明:* 根据前面的引理，对于足够大的 $k$，$\mathbf{x}^k$ 是 $F^k$ 的无约束局部极小值。因此，根据（无约束的）*一阶必要条件*，有

$$ \nabla F^k(\mathbf{x}^k) = \mathbf{0}. $$

为了计算 $F^k$ 的梯度，我们注意到

$$ \|\mathbf{h}(\mathbf{x})\|² = \sum_{i=1}^\ell (h_i(\mathbf{x}))². $$

偏导数是

$$ \frac{\partial}{\partial x_j} \|\mathbf{h}(\mathbf{x})\|² = \sum_{i=1}^\ell \frac{\partial}{\partial x_j} (h_i(\mathbf{x}))² = \sum_{i=1}^\ell 2 h_i(\mathbf{x}) \frac{\partial h_i(\mathbf{x})}{\partial x_j}, $$

通过 *链式法则*。因此，以向量形式，

$$ \nabla \|\mathbf{h}(\mathbf{x})\|² = 2 \mathbf{J}_{\mathbf{h}}(\mathbf{x})^T \mathbf{h}(\mathbf{x}). $$

项 $\|\mathbf{x} - \mathbf{x}^*\|²$ 可以重写为二次函数

$$ \|\mathbf{x} - \mathbf{x}^*\|² = \frac{1}{2}\mathbf{x}^T (2 I_{d \times d}) \mathbf{x} - 2 (\mathbf{x}^*)^T \mathbf{x} + (\mathbf{x}^*)^T \mathbf{x}^*. $$

使用之前的公式，其中 $P = 2 I_{d \times d}$（这是一个对称矩阵），$\mathbf{q} = -2 \mathbf{x}^*$ 和 $r = (\mathbf{x}^*)^T \mathbf{x}^*$，我们得到

$$ \nabla \|\mathbf{x} - \mathbf{x}^*\|² = 2\mathbf{x} -2 \mathbf{x}^*. $$

因此，将所有这些放在一起，

$$ (**) \qquad \mathbf{0} = \nabla F^k(\mathbf{x}^k) = \nabla f(\mathbf{x}^k) + \mathbf{J}_{\mathbf{h}}(\mathbf{x}^k)^T (k \mathbf{h}(\mathbf{x}^k)) + \alpha(\mathbf{x}^k - \mathbf{x}^*). $$

根据前面的引理，$\mathbf{x}^k \to \mathbf{x}^*$，$\nabla f(\mathbf{x}^k) \to \nabla f(\mathbf{x}^*)$，以及 $\mathbf{J}_{\mathbf{h}}(\mathbf{x}^k) \to \mathbf{J}_{\mathbf{h}}(\mathbf{x}^*)$ 当 $k \to +\infty$。

因此，剩下要推导 $k \mathbf{h}(\mathbf{x}^k)$ 的极限。根据假设，$\mathbf{J}_{\mathbf{h}}(\mathbf{x}^*)^T$ 的列是线性无关的。这意味着对于任何单位向量 $\mathbf{z} \in \mathbb{R}^\ell$

$$ \mathbf{z}^T \mathbf{J}_{\mathbf{h}}(\mathbf{x}^*) \,\mathbf{J}_{\mathbf{h}}(\mathbf{x}^*)^T \mathbf{z} = \|\mathbf{J}_{\mathbf{h}}(\mathbf{x}^*)^T \mathbf{z}\|² > 0 $$

否则，我们会有 $\mathbf{J}_{\mathbf{h}}(\mathbf{x}^*)^T \mathbf{z} = \mathbf{0}$，这与线性无关的假设相矛盾。根据 *极值定理*，存在 $\beta > 0$ 使得

$$ \mathbf{z}^T \mathbf{J}_{\mathbf{h}}(\mathbf{x}^*) \,\mathbf{J}_{\mathbf{h}}(\mathbf{x}^*)^T \mathbf{z} \geq \beta $$

对于所有单位向量 $\mathbf{z} \in \mathbb{R}^\ell$。由于 $\mathbf{J}_{\mathbf{h}}(\mathbf{x}^k) \to \mathbf{J}_{\mathbf{h}}(\mathbf{x}^*)$，根据之前的引理，对于足够大的 $k$ 和任意单位向量 $\mathbf{z} \in \mathbb{R}^\ell$，

$$ |\mathbf{z}^T \mathbf{J}_{\mathbf{h}}(\mathbf{x}^*)\, \mathbf{J}_{\mathbf{h}}(\mathbf{x}^*)^T \mathbf{z} - \mathbf{z}^T \mathbf{J}_{\mathbf{h}}(\mathbf{x}^k) \,\mathbf{J}_{\mathbf{h}}(\mathbf{x}^k)^T \mathbf{z}| \leq \| \mathbf{J}_{\mathbf{h}}(\mathbf{x}^*) \,\mathbf{J}_{\mathbf{h}}(\mathbf{x}^*)^T - \mathbf{J}_{\mathbf{h}}(\mathbf{x}^k) \,\mathbf{J}_{\mathbf{h}}(\mathbf{x}^k)^T \|_F \leq \frac{\beta}{2}. $$

这意味着

$$ \mathbf{z}^T \mathbf{J}_{\mathbf{h}}(\mathbf{x}^k) \,\mathbf{J}_{\mathbf{h}}(\mathbf{x}^k)^T \mathbf{z} \geq \frac{\beta}{2}, $$

因此，通过上述相同的论证，当 $k$ 足够大时，$\mathbf{J}_{\mathbf{h}}(\mathbf{x}^k)^T$ 的列线性无关，且 $\mathbf{J}_{\mathbf{h}}(\mathbf{x}^k) \,\mathbf{J}_{\mathbf{h}}(\mathbf{x}^k)^T$ 是可逆的。这证明了 a)。

回到 $(**)$，两边乘以 $(\mathbf{J}_{\mathbf{h}}(\mathbf{x}^k)\, \mathbf{J}_{\mathbf{h}}(\mathbf{x}^k)^T)^{-1} \mathbf{J}_{\mathbf{h}}(\mathbf{x}^k)$，并取极限 $k \to +\infty$，经过重新排列后得到

$$ k \mathbf{h}(\mathbf{x}^k) \to - (\mathbf{J}_{\mathbf{h}}(\mathbf{x}^*) ^T \mathbf{J}_{\mathbf{h}}(\mathbf{x}^*))^{-1} \mathbf{J}_{\mathbf{h}}(\mathbf{x}^*) \nabla f(\mathbf{x}^*) = \blambda^*. $$

将变量代回，我们得到

$$ \nabla f(\mathbf{x}^*) + \mathbf{J}_{\mathbf{h}}(\mathbf{x}^*)^T \blambda^* = \mathbf{0} $$

如所声称。这证明了 b)。 $\square$

结合引理可以建立定理。 $\square$

接下来，我们证明**拉格朗日乘数：二阶充分条件**。再次，我们遵循 [[Ber](http://www.athenasc.com/nonlinbook.html), 第 4.2 节]。我们需要以下引理。证明可以跳过。

**证明思路：** 我们考虑一个略微修改的问题

$$\begin{align*} &\text{min} f(\mathbf{x}) + \frac{c}{2} \|\mathbf{h}(\mathbf{x})\|²\\ &\text{s.t.}\ \mathbf{h}(\mathbf{x}) = \mathbf{0} \end{align*}$$

它具有相同的局部极小值。将**二阶充分条件**应用于修改后问题的拉格朗日函数，当 $c$ 足够大时给出结果。

**引理** 设 $P$ 和 $Q$ 是 $\mathbb{R}^{n \times n}$ 中的对称矩阵。假设 $Q$ 是正半定矩阵，且 $P$ 在 $Q$ 的零空间上是正定的，即对于所有 $\mathbf{w} \neq \mathbf{0}$ 满足 $\mathbf{w}^T Q \mathbf{w} = \mathbf{0}$ 的 $\mathbf{w}$，有 $\mathbf{w}^T P \mathbf{w} > 0$。那么存在一个标量 $\bar{c} \geq 0$，使得对于所有 $c > \bar{c}$，$P + c Q$ 是正定的。 $\flat$

*证明：* *(引理)* 我们通过反证法进行论证。假设存在一个非负的、递增的、发散的序列$\{c_k\}_{k=1}^{+\infty}$和一个单位向量序列$\{\mathbf{x}^k\}_{k=1}^{+\infty}$，使得

$$ (\mathbf{x}^k)^T (P + c_k Q) \mathbf{x}^k \leq 0 $$

对于所有$k$。因为序列是有界的，所以它有一个极限点$\bar{\mathbf{x}}$。假设不失一般性，$\mathbf{x}^k \to \bar{\mathbf{x}}$，当$k \to \infty$。由于$c_k \to +\infty$并且根据假设$(\mathbf{x}^k)^T Q \mathbf{x}^k \geq 0$，我们必须有$(\bar{\mathbf{x}})^T Q \bar{\mathbf{x}} = 0$，否则$(\mathbf{x}^k)^T (P + c_k Q) \mathbf{x}^k$将会发散。因此，根据陈述中的假设，$(\bar{\mathbf{x}})^T P \bar{\mathbf{x}} > 0$。这与上面的不等式相矛盾。$\square$

*证明：* *(拉格朗日乘数：二阶充分条件)* 我们考虑修改后的问题

$$\begin{align*} &\text{min} f(\mathbf{x}) + \frac{c}{2} \|\mathbf{h}(\mathbf{x})\|²\\ &\text{s.t.}\ \mathbf{h}(\mathbf{x}) = \mathbf{0}. \end{align*}$$

由于目标函数中的附加项对于可行向量来说是零，因此它具有与原问题相同的局部极小值。这个额外的项将允许我们使用之前的引理。为了方便起见，我们定义

$$ g_c(\mathbf{x}) = f(\mathbf{x}) + \frac{c}{2} \|\mathbf{h}(\mathbf{x})\|². $$

修改后问题的拉格朗日函数为

$$ L_c(\mathbf{x}, \blambda) = g_c(\mathbf{x}) + \mathbf{h}(\mathbf{x})^T \blambda. $$

我们将应用**二阶充分条件**来求解在$\mathbf{x}$上最小化$L_c$的问题。我们用$\nabla²_{\mathbf{x},\mathbf{x}}$表示仅关于变量$\mathbf{x}$的 Hessian 矩阵。

回忆从**拉格朗日乘数：一阶必要条件**的证明中，我们知道

$$ \nabla \|\mathbf{h}(\mathbf{x})\|² = 2 \mathbf{J}_{\mathbf{h}}(\mathbf{x})^T \mathbf{h}(\mathbf{x}). $$

为了计算该函数的 Hessian 矩阵，我们注意到

$$\begin{align*} \frac{\partial}{\partial x_i}\left( \frac{\partial}{\partial x_j} \|\mathbf{h}(\mathbf{x})\|²\right) &= \frac{\partial}{\partial x_i}\left( \sum_{k=1}^\ell 2 h_k(\mathbf{x}) \frac{\partial h_k(\mathbf{x})}{\partial x_j}\right)\\ &= 2 \sum_{k=1}^\ell\left( \frac{\partial h_k(\mathbf{x})}{\partial x_i} \frac{\partial h_k(\mathbf{x})}{\partial x_j} + h_k(\mathbf{x}) \frac{\partial² h_k(\mathbf{x})}{\partial x_i \partial x_j} \right)\\ &= 2 \left(\mathbf{J}_{\mathbf{h}}(\mathbf{x})^T \,\mathbf{J}_{\mathbf{h}}(\mathbf{x}) + \sum_{k=1}^\ell h_k(\mathbf{x}) \, \mathbf{H}_{h_k}(\mathbf{x})\right)_{i,j}. \end{align*}$$

因此

$$ \nabla_{\mathbf{x}} L_c(\mathbf{x}^*, \blambda^*) = \nabla f(\mathbf{x}^*) + c \mathbf{J}_{\mathbf{h}}(\mathbf{x}^*)^T \mathbf{h}(\mathbf{x}^*) + \mathbf{J}_{\mathbf{h}}(\mathbf{x}^*)^T \blambda^* $$

和

$$ \nabla²_{\mathbf{x},\mathbf{x}} L_c(\mathbf{x}^*, \blambda^*) = \mathbf{H}_{f}(\mathbf{x}^*) + c \left(\mathbf{J}_{\mathbf{h}}(\mathbf{x}^*)^T \,\mathbf{J}_{\mathbf{h}}(\mathbf{x}^*) + \sum_{k=1}^\ell h_k(\mathbf{x}^*) \, \mathbf{H}_{h_k}(\mathbf{x}^*)\right) + \sum_{k=1}^\ell \lambda^*_k \, \mathbf{H}_{h_k}(\mathbf{x}^*). $$

根据定理的假设，这简化为

$$ \nabla_{\mathbf{x}} L_c(\mathbf{x}^*, \blambda^*) = \mathbf{0} $$

和

$$ \nabla²_{\mathbf{x},\mathbf{x}} L_c(\mathbf{x}^*, \blambda^*) =\underbrace{\left\{ \mathbf{H}_{f}(\mathbf{x}^*) + \sum_{k=1}^\ell \lambda^*_k \, \mathbf{H}_{h_k}(\mathbf{x}^*) \right\}}_{P} + c \underbrace{\left\{\mathbf{J}_{\mathbf{h}}(\mathbf{x}^*)^T \,\mathbf{J}_{\mathbf{h}}(\mathbf{x}^*) \right\}}_{Q}, $$

其中，对于任何满足 $\mathbf{J}_{\mathbf{h}}(\mathbf{x}^*) \,\mathbf{v} = \mathbf{0}$（这本身意味着 $\mathbf{v}^T Q \mathbf{v} = \mathbf{0}$）的 $\mathbf{v}$，都有 $\mathbf{v}^T P \mathbf{v} > 0$。此外，由于

$$ \mathbf{w}^T Q \mathbf{w} = \mathbf{w}^T \mathbf{J}_{\mathbf{h}}(\mathbf{x}^*)^T \,\mathbf{J}_{\mathbf{h}}(\mathbf{x}^*) \mathbf{w} = \left\|\mathbf{J}_{\mathbf{h}}(\mathbf{x}^*) \mathbf{w}\right\|² \geq 0 $$

对于任何 $\mathbf{w} \in \mathbb{R}^d$。前述引理允许我们取 $c$ 足够大，使得 $\nabla²_{\mathbf{x},\mathbf{x}} L_c(\mathbf{x}^*, \blambda^*) \succ \mathbf{0}$。

因此，对于该问题，在 $\mathbf{x}^*$ 处满足无约束的**二阶充分条件**。

$$ \min_{\mathbf{x} \in \mathbb{R}^d} L_c(\mathbf{x}, \blambda^*). $$

即存在 $\delta > 0$ 使得

$$ L_c(\mathbf{x}^*, \blambda^*) < L_c(\mathbf{x}, \blambda^*), \qquad \forall \mathbf{x} \in B_{\delta}(\mathbf{x}^*) \setminus \{\mathbf{x}^*\}. $$

将此限制为修改后的约束问题的可行向量（即那些满足 $\mathbf{h}(\mathbf{x}) = \mathbf{0}$ 的向量）意味着在简化后

$$ f(\mathbf{x}^*) < f(\mathbf{x}), \qquad \forall \mathbf{x} \in \{\mathbf{x} : \mathbf{h}(\mathbf{x}) = \mathbf{0}\} \cap (B_{\delta}(\mathbf{x}^*) \setminus \{\mathbf{x}^*\}). $$

因此，$\mathbf{x}^*$ 是修改后的约束问题的严格局部极小值（并且由此，原始约束问题也是如此）。这就完成了证明。 $\square$
