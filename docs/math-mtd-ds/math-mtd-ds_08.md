# 1.6\. 在线补充材料#

> 原文：[`mmids-textbook.github.io/chap01_intro/supp/roch-mmids-intro-supp.html`](https://mmids-textbook.github.io/chap01_intro/supp/roch-mmids-intro-supp.html)

## 1.6.1\. 问答、解决方案、代码等。#

### 1.6.1.1\. 仅代码。#

一个包含本章代码的交互式 Jupyter 笔记本可以在此处访问（推荐使用 Google Colab）。鼓励您对其进行实验。一些建议的计算练习散布在其中。笔记本也可以作为幻灯片查看。

+   [笔记本](https://github.com/MMiDS-textbook/MMiDS-textbook.github.io/blob/main/just_the_code/roch_mmids_chap_intro_notebook.ipynb) ([在 Colab 中打开](https://colab.research.google.com/github/MMiDS-textbook/MMiDS-textbook.github.io/blob/main/just_the_code/roch_mmids_chap_intro_notebook.ipynb))

+   [幻灯片](https://raw.githack.com/MMiDS-textbook/MMiDS-textbook.github.io/main/just_the_code/roch_mmids_chap_intro_notebook_slides.slides.html)

### 1.6.1.2\. 自我评估问答。#

通过以下链接可以访问更全面的自我评估问答的网页版本。

+   [第 1.2 节](https://raw.githack.com/MMiDS-textbook/MMiDS-textbook.github.io/main/quizzes/self-assessment/quiz_1_2.html)

+   [第 1.3 节](https://raw.githack.com/MMiDS-textbook/MMiDS-textbook.github.io/main/quizzes/self-assessment/quiz_1_3.html)

+   [第 1.4 节](https://raw.githack.com/MMiDS-textbook/MMiDS-textbook.github.io/main/quizzes/self-assessment/quiz_1_4.html)

### 1.6.1.3\. 自动问答。#

可以在此处访问本章的自动生成的问答（推荐使用 Google Colab）。

+   [自动问答](https://github.com/MMiDS-textbook/MMiDS-textbook.github.io/blob/main/quizzes/auto_quizzes/roch-mmids-intro-autoquiz.ipynb) ([在 Colab 中打开](https://colab.research.google.com/github/MMiDS-textbook/MMiDS-textbook.github.io/blob/main/quizzes/auto_quizzes/roch-mmids-intro-autoquiz.ipynb))

### 1.6.1.4\. 奇数练习题的解决方案。#

*(在克劳德、双子座和 ChatGPT 的帮助下)*

E1.2.1 的答案和解释：欧几里得范数 \(\|\mathbf{x}\|_2\) 定义为

$$ \|\mathbf{x}\|_2 = \sqrt{x_1² + x_2²} = \sqrt{6² + 8²} = \sqrt{36 + 64} = \sqrt{100} = 10. $$

E1.2.3 的答案和解释：通过交换行和列得到转置矩阵 \(A^T\)

$$\begin{split} A^T = \begin{pmatrix}1 & 4 \\ 2 & 5 \\ 3 & 6\end{pmatrix} \end{split}$$

E1.2.5 的答案和解释：如果一个矩阵 \(A\) 满足 \(A = A^T\)，则该矩阵是对称的

$$\begin{split} A^T = \begin{pmatrix}2 & 0 \\ 0 & 3\end{pmatrix} = A. \end{split}$$

因此，\(A\) 是对称的。

对 E1.2.7 的答案和解释：\(\frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x+h, y) - f(x, y)}{h} = \lim_{h \to 0} \frac{(x+h)² + (x+h)y + y² - (x² + xy + y²)}{h} = \lim_{h \to 0} \frac{2xh + h² + hy}{h} = 2x + y\)。

\(\frac{\partial f}{\partial y} = \lim_{h \to 0} \frac{f(x, y+h) - f(x, y)}{h} = \lim_{h \to 0} \frac{x² + x(y+h) + (y+h)² - (x² + xy + y²)}{h} = \lim_{h \to 0} \frac{xh + 2yh + h²}{h} = x + 2y\)。

对 E1.2.9 的答案和解释：根据 **泰勒定理**，\(f(x) = f(a) + (x - a)f'(a) + \frac{1}{2}(x - a)²f''(\xi)\) 对于某个在 \(a\) 和 \(x\) 之间的 \(\xi\)。这里，\(f(1) = 1³ - 3 \cdot 1² + 2 \cdot 1 = 0\)，\(f'(x) = 3x² - 6x + 2\)，所以 \(f'(1) = 3 - 6 + 2 = -1\)，并且 \(f''(x) = 6x - 6\)。因此，对于某个在 \(1\) 和 \(x\) 之间的 \(\xi\)，\(f(x) = 0 + (x - 1)(-1) + \frac{1}{2}(x - 1)²(6\xi - 6)\)。

对 E1.2.11 的答案和解释：首先，计算 \(\E[X²]\)

$$ \E[X²] = \sum_{x} x² \cdot P(X = x) = 1² \cdot 0.4 + 2² \cdot 0.6 = 0.4 + 4 \cdot 0.6 = 0.4 + 2.4 = 2.8. $$

然后，使用公式 \(\mathrm{Var}[X] = \E[X²] - (\E[X])²\)

$$ \mathrm{Var}[X] = 2.8 - (1.6)² = 2.8 - 2.56 = 0.24. $$

对 E1.2.13 的答案和解释：根据 **切比雪夫不等式**，对于任何 \(\alpha > 0\)，\(\P[|X - \mathbb{E}[X]| \geq \alpha] \leq \frac{\mathrm{Var}[X]}{\alpha²}\)。这里，\(\alpha = 4\)，所以 \(\P[|X - 3| \geq 4] \leq \frac{4}{4²} = \frac{1}{4}\)。

对 E1.2.15 的答案和解释：\(X\) 的协方差矩阵是 \(\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}\)。由于 \(X_1\) 和 \(X_2\) 是独立的，它们的协方差为零。由于它们是标准正态分布，每个的方差都是 1。

对 E1.2.17 的答案和解释：\(\mathbb{E}[AX] = A\mathbb{E}[X] = A\mu_X = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} \begin{pmatrix} 1 \\ 2 \end{pmatrix} = \begin{pmatrix} 5 \\ 11 \end{pmatrix}\)，\(\text{Cov}[AX] = A\text{Cov}[X]A^T = A\Sigma_XA^T = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} \begin{pmatrix} 3 & 1 \\ 1 & 4 \end{pmatrix} \begin{pmatrix} 1 & 3 \\ 2 & 4 \end{pmatrix} = \begin{pmatrix} 11 & 19 \\ 19 & 35 \end{pmatrix}\)

对 E1.2.19 的答案和解释：对于任何非零向量 \(z = \begin{pmatrix} z_1 \\ z_2 \end{pmatrix}\)，我们有

$$\begin{align*} z^TAz &= \begin{pmatrix} z_1 & z_2 \end{pmatrix} \begin{pmatrix} 2 & -1 \\ -1 & 2 \end{pmatrix} \begin{pmatrix} z_1 \\ z_2 \end{pmatrix} \\ &= \begin{pmatrix} z_1 & z_2 \end{pmatrix} \begin{pmatrix} 2z_1 - z_2 \\ -z_1 + 2z_2 \end{pmatrix} \\ &= 2z_1² - 2z_1z_2 + 2z_2² \\ &= z_1² + (z_1 - z_2)² + z_2² > 0 \end{align*}$$

由于 \((z_1 - z_2)² \geq 0\)，并且 \(z_1² > 0\) 或 \(z_2² > 0\)（因为 \(z \neq 0\)）。因此，\(A\) 是正定的。

对 E1.3.1 的答案和解释：\(\boldsymbol{\mu}_1^* = \frac{1}{2}(\mathbf{x}_1 + \mathbf{x}_4) = (\frac{3}{2}, 1)\)，\(\boldsymbol{\mu}_2^* = \frac{1}{2}(\mathbf{x}_2 + \mathbf{x}_3) = (-\frac{1}{2}, 0)\)。

对 E1.3.3 的回答和证明：\(C_1 = \{1, 5\}\)，\(C_2 = \{3\}\)，\(C_3 = \{2, 4\}\)。

对 E1.3.5 的回答和证明：展开平方范数并消去项得到等价的不等式。

对 E1.3.7 的回答和证明：展开 \(\|A + B\|_F² = \sum_{i=1}^n \sum_{j=1}^m (A_{ij} + B_{ij})²\) 并简化。

对 E1.3.9 的回答和证明：\(q(x) = 3(x-1)² + 2\)，最小值在\(x = 1\)时为 2。

对 E1.3.11 的回答和证明：\(\|A\|_F = \sqrt{14}\)。

对 E1.4.1 的回答和证明：由于\(X_i\)在\([-1/2, 1/2]\)上均匀分布，其概率密度函数为\(f_{X_i}(x) = 1\)，当\(x \in [-1/2, 1/2]\)时，否则为 0。因此，

$$\mathbb{E}[X_i] = \int_{-1/2}^{1/2} x f_{X_i}(x) dx = \int_{-1/2}^{1/2} x dx = 0$$

和

$$\text{Var}[X_i] = \mathbb{E}[X_i²] - (\mathbb{E}[X_i])² = \int_{-1/2}^{1/2} x² dx = \frac{1}{12}.$$

对 E1.4.3 的回答和证明：我们有

$$\mathbb{E}[\|X\|²] = \mathbb{E}\left[\sum_{i=1}^d X_i²\right] = \sum_{i=1}^d \mathbb{E}[X_i²] = \sum_{i=1}^d 1 = d.$$

对 E1.4.5 的回答和证明：根据**切比雪夫不等式**，\(P[|X - 1| \geq 3] \leq \frac{\mathrm{Var}[X]}{3²} = \frac{4}{9}\)。

对 E1.4.7 的回答和证明：根据**切比雪夫不等式**，\(\P[|X| \geq 2\sqrt{d}] \leq \frac{\mathrm{Var}[X]}{(2\sqrt{d})²} = \frac{1}{4d}\)。对于\(d = 1\)，\(\P[|X| \geq 2] \leq \frac{1}{4}\)。对于\(d = 10\)，\(P[|X| \geq 2\sqrt{10}] \leq \frac{1}{40}\)。对于\(d = 100\)，\(\P[|X| \geq 20] \leq \frac{1}{400}\)。

### 1.6.1.5\. 学习成果#

+   计算向量和矩阵的范数，以及向量之间的内积。

+   利用范数和内积的性质，包括柯西-施瓦茨不等式和三角不等式。

+   执行基本的矩阵运算，包括加法、标量乘法、乘法和转置。

+   计算矩阵-向量积和矩阵-矩阵积，并解释它们的几何和代数意义。

+   解释下降方向的概念及其与优化的关系。

+   使用泰勒定理来近似函数并分析这些近似中的误差项。

+   定义并计算多变量函数的偏导数和梯度。

+   陈述并应用切比雪夫不等式来量化随机变量围绕其均值集中的集中度。

+   计算随机变量和向量的期望、方差和协方差。

+   定义并给出正半定矩阵的例子。

+   解释随机向量协方差矩阵的性质和意义。

+   描述并从球面高斯分布中生成样本。

+   编写 Python 代码来计算范数、内积和执行矩阵运算。

+   使用 Python 模拟随机变量，计算它们的统计属性，并可视化分布和大量法则。

+   将 k-means 聚类问题表述为优化问题。

+   描述 k-means 算法及其寻找聚类中心和分配的迭代步骤。

+   推导固定划分的最优聚类代表（质心）。

+   推导固定代表的最优划分（聚类分配）。

+   分析 k-means 算法的收敛性质，并理解其在寻找全局最优解方面的局限性。

+   将 k-means 算法应用于现实世界的数据集，例如企鹅数据集，并解释结果。

+   将 k-means 目标函数表示为矩阵形式，并将其与矩阵分解联系起来。

+   理解数据预处理步骤（如标准化）对于 k-means 的重要性。

+   认识到 k-means 的局限性，例如其对初始化的敏感性以及需要提前指定聚类数目的需求。

+   讨论确定 k-means 聚类中最佳聚类数目的挑战。

+   理解使用 k-means 算法聚类高维数据时的挑战，并认识到维度增加如何导致噪声压倒信号，使得区分聚类变得困难。

+   理解高维空间中测度集中的概念，特别是这样一个反直觉的事实：高维立方体的绝大部分体积集中在其角部，使其看起来像“尖锐的球体”。

+   将切比雪夫不等式应用于推导从高维立方体中随机选择一点落在内切球内的概率，并理解随着维度增加，这个概率如何降低。

+   分析标准正态向量的范数在高维空间中的行为，并使用切比雪夫不等式证明，尽管联合概率密度函数在原点处达到最大值，但范数很可能接近维度的平方根。

+   在高维设置中使用切比雪夫不等式推导概率界限。

+   模拟高维数据以经验验证理论结果。

\(\aleph\)

## 1.6.2\. 其他部分#

### 1.6.2.1\. 高维聚类的一个更严格的分析#

在本可选部分，我们给出前一小节所述现象的一个正式陈述。

**定理** 设 \(\mathbf{X}_1, \mathbf{X}_2, \mathbf{Y}_1\) 是独立的球形 \(d\)-维高斯分布，均值为 \(-w_d \mathbf{e}_1\)，方差为 \(1\)，其中 \(\{w_d\}\) 是 \(d\) 中的单调序列。设 \(\mathbf{Y}_2\) 是独立的球形 \(d\)-维高斯分布，均值为 \(w_d \mathbf{e}_1\)，方差为 \(1\)。令 \(\Delta_d = \|\mathbf{Y}_1 - \mathbf{Y}_2\|² - \|\mathbf{X}_1 - \mathbf{X}_2\|²\)，当 \(d \to +\infty\) 时

$$\begin{split} \frac{\mathbb{E}[\Delta_d]}{\sqrt{\mathrm{Var}[\Delta_d]}} \to \begin{cases} 0, & \text{如果 $w_d \ll d^{1/4}$}\\ +\infty, & \text{如果 $w_d \gg d^{1/4}$} \end{cases} \end{split}$$

其中 \(w_d \ll d^{1/4}\) 表示 \(w_d/d^{1/4} \to 0\)。\(\sharp\)

这个比率被称为[信噪比](https://en.wikipedia.org/wiki/Signal-to-noise_ratio)。

为了证明这个命题，我们需要以下性质。

**引理** 设 \(W\) 是关于零对称的实值随机变量，即 \(W\) 和 \(-W\) 是同分布的。那么对于所有奇数 \(k\)，\(\mathbb{E}[W^k] = 0\)。\(\flat\)

*证明:* 通过对称性，

$$ \mathbb{E}[W^k] = \mathbb{E}[(-W)^k] = \mathbb{E}[(-1)^k W^k] = - \mathbb{E}[W^k]. $$

满足这个方程的唯一方法是 \(\mathbb{E}[W^k] = 0\)。\(\square\)

返回到命题的证明：

*证明思路:* *(定理)* 通过期望的线性，对 \(\mathbb{E}[\Delta_d]\) 贡献的唯一坐标是第一个，而所有坐标都对 \(\mathrm{Var}[\Delta_d]\) 贡献。更具体地说，计算表明前者是 \(c_0 w²\)，而后者是 \(c_1 w² + c_2 d\)，其中 \(c_0, c_1, c_2\) 是常数。

*证明:* *(命题)* 写 \(w := w_d\) 和 \(\Delta := \Delta_d\) 以简化符号。有两个步骤：

*(1) \(\Delta\) 的期望：* 根据定义，随机变量 \(X_{1,i} - X_{2,i}\)，\(i = 1,\ldots, d\)，和 \(Y_{1,i} - Y_{2,i}\)，\(i = 2,\ldots, d\)，是同分布的。因此，根据期望的线性，我们有

$$\begin{align*} \mathbb{E}[\Delta] &= \sum_{i=1}^d \mathbb{E}[(Y_{1,i} - Y_{2,i})²] - \sum_{i=1}^d \mathbb{E}[(X_{1,i} - X_{2,i})²]\\ &= \mathbb{E}[(Y_{1,1} - Y_{2,1})²] - \mathbb{E}[(X_{1,1} - X_{2,1})²]. \end{align*}$$

进一步，我们可以写出 \(Y_{1,1} - Y_{1,2} \sim (Z_1 -w) - (Z_2+w)\) 其中 \(Z_1, Z_2 \sim N(0,1)\) 是独立的，其中这里 \(\sim\) 表示分布上的等价。因此，我们有

$$\begin{align*} \mathbb{E}[(Y_{1,1} - Y_{2,1})²] &= \mathbb{E}[(Z_1 - Z_2 - 2w)²]\\ &= \mathbb{E}[(Z_1 - Z_2)²] - 4w \,\mathbb{E}[Z_1 - Z_2] + 4 w². \end{align*}$$

类似地，\(X_{1,1} - X_{1,2} \sim Z_1 - Z_2\) 因此 \(\mathbb{E}[(X_{1,1} - X_{2,1})²] = \mathbb{E}[(Z_1 - Z_2)²]\)。由于 \(\mathbb{E}[Z_1 - Z_2] = 0\)，我们最终得到 \(\mathbb{E}[\Delta] = 4 w²\)。

*(2) \(\Delta\) 的方差：* 使用 (1) 中的观察结果和坐标的独立性，我们得到

$$\begin{align*} \mathrm{Var}[\Delta] &= \sum_{i=1}^d \mathrm{Var}[(Y_{1,i} - Y_{2,i})²] + \sum_{i=1}^d \mathrm{Var}[(X_{1,i} - X_{2,i})²]\\ &= \mathrm{Var}[(Z_1 - Z_2 - 2w)²] + (2d-1) \,\mathrm{Var}[(Z_1 - Z_2)²]. \end{align*}$$

通过**求和的方差**，

$$\begin{align*} \mathrm{Var}[(Z_1 - Z_2 - 2w)²] &= \mathrm{Var}[(Z_1 - Z_2)² - 4w(Z_1 - Z_2) + 4w²]\\ &= \mathrm{Var}[(Z_1 - Z_2)² - 4w(Z_1 - Z_2)]\\ &= \mathrm{Var}[(Z_1 - Z_2)²] + 16 w² \mathrm{Var}[Z_1 - Z_2]\\ &\quad - 8w \,\mathrm{Cov}[(Z_1 - Z_2)², Z_1 - Z_2]. \end{align*}$$

因为 \(Z_1\) 和 \(Z_2\) 是独立的，\(\mathrm{Var}[Z_1 - Z_2] = \mathrm{Var}[Z_1] + \mathrm{Var}[Z_2] = 2\)。此外，随机变量 \((Z_1 - Z_2)\) 是对称的，所以

$$\begin{align*} \mathrm{Cov}[(Z_1 - Z_2)², Z_1 - Z_2] &= \mathbb{E}[(Z_1 - Z_2)³]\\ & \quad - \mathbb{E}[(Z_1 - Z_2)²] \,\mathbb{E}[Z_1 - Z_2]\\ &= 0. \end{align*}$$

最后，

$$ \mathrm{Var}[\Delta] = 32 w² + 2d \,\mathrm{Var}[(Z_1 - Z_2)²] $$

*整合一切：*

$$ \frac{\mathbb{E}[\Delta]}{\sqrt{\mathrm{Var}[\Delta]}} = \frac{4 w²}{\sqrt{32 w² + 2d \,\mathrm{Var}[(Z_1 - Z_2)²]}}. $$

当 \(d \to +\infty\) 时，得出结论。 \(\square\)

## 1.6.1\. 测验、解答、代码等。#

### 1.6.1.1\. 仅代码#

本章中的代码可以在此处访问交互式 Jupyter 笔记本（推荐使用 Google Colab）。鼓励您对其进行尝试。一些建议的计算练习散布在其中。笔记本也可以作为幻灯片查看。

+   [笔记本](https://github.com/MMiDS-textbook/MMiDS-textbook.github.io/blob/main/just_the_code/roch_mmids_chap_intro_notebook.ipynb) ([在 Colab 中打开](https://colab.research.google.com/github/MMiDS-textbook/MMiDS-textbook.github.io/blob/main/just_the_code/roch_mmids_chap_intro_notebook.ipynb))

+   [幻灯片](https://raw.githack.com/MMiDS-textbook/MMiDS-textbook.github.io/main/just_the_code/roch_mmids_chap_intro_notebook_slides.slides.html)

### 1.6.1.2\. 自我评估测验#

通过以下链接可以获取更广泛的自我评估测验的网络版本。

+   [第 1.2 节](https://raw.githack.com/MMiDS-textbook/MMiDS-textbook.github.io/main/quizzes/self-assessment/quiz_1_2.html)

+   [第 1.3 节](https://raw.githack.com/MMiDS-textbook/MMiDS-textbook.github.io/main/quizzes/self-assessment/quiz_1_3.html)

+   [第 1.4 节](https://raw.githack.com/MMiDS-textbook/MMiDS-textbook.github.io/main/quizzes/self-assessment/quiz_1_4.html)

### 1.6.1.3\. 自动测验#

本章的自动测验可以在此处访问（推荐使用 Google Colab）。

+   [自动测验](https://github.com/MMiDS-textbook/MMiDS-textbook.github.io/blob/main/quizzes/auto_quizzes/roch-mmids-intro-autoquiz.ipynb) ([在 Colab 中打开](https://colab.research.google.com/github/MMiDS-textbook/MMiDS-textbook.github.io/blob/main/quizzes/auto_quizzes/roch-mmids-intro-autoquiz.ipynb))

### 1.6.1.4\. 奇数编号预热练习的解答#

*(有 Claude、Gemini 和 ChatGPT 的帮助)*

E1.2.1 的答案和证明：欧几里得范数 \(\|\mathbf{x}\|_2\) 定义为

$$ \|\mathbf{x}\|_2 = \sqrt{x_1² + x_2²} = \sqrt{6² + 8²} = \sqrt{36 + 64} = \sqrt{100} = 10. $$

E1.2.3 的答案和证明：转置 \(A^T\) 是通过交换行和列得到的

$$\begin{split} A^T = \begin{pmatrix}1 & 4 \\ 2 & 5 \\ 3 & 6\end{pmatrix} \end{split}$$

E1.2.5 的答案和证明：一个矩阵 \(A\) 是对称的，如果 \(A = A^T\)

$$\begin{split} A^T = \begin{pmatrix}2 & 0 \\ 0 & 3\end{pmatrix} = A. \end{split}$$

因此，\(A\) 是对称的。

E1.2.7 的答案和证明：\(\frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x+h, y) - f(x, y)}{h} = \lim_{h \to 0} \frac{(x+h)² + (x+h)y + y² - (x² + xy + y²)}{h} = \lim_{h \to 0} \frac{2xh + h² + hy}{h} = 2x + y\).

\(\frac{\partial f}{\partial y} = \lim_{h \to 0} \frac{f(x, y+h) - f(x, y)}{h} = \lim_{h \to 0} \frac{x² + x(y+h) + (y+h)² - (x² + xy + y²)}{h} = \lim_{h \to 0} \frac{xh + 2yh + h²}{h} = x + 2y\).

E1.2.9 的答案和证明：根据 *泰勒定理*，\(f(x) = f(a) + (x - a)f'(a) + \frac{1}{2}(x - a)²f''(\xi)\) 对于某个在 \(a\) 和 \(x\) 之间的 \(\xi\)。这里，\(f(1) = 1³ - 3 \cdot 1² + 2 \cdot 1 = 0\)，\(f'(x) = 3x² - 6x + 2\)，所以 \(f'(1) = 3 - 6 + 2 = -1\)，并且 \(f''(x) = 6x - 6\)。因此，\(f(x) = 0 + (x - 1)(-1) + \frac{1}{2}(x - 1)²(6\xi - 6)\) 对于某个在 \(1\) 和 \(x\) 之间的 \(\xi\)。

E1.2.11 的答案和证明：首先，计算 \(\E[X²]\)

$$ \E[X²] = \sum_{x} x² \cdot P(X = x) = 1² \cdot 0.4 + 2² \cdot 0.6 = 0.4 + 4 \cdot 0.6 = 0.4 + 2.4 = 2.8. $$

然后，使用公式 \(\mathrm{Var}[X] = \E[X²] - (\E[X])²\)

$$ \mathrm{Var}[X] = 2.8 - (1.6)² = 2.8 - 2.56 = 0.24. $$

E1.2.13 的答案和证明：根据 *切比雪夫不等式*，\(\P[|X - \mathbb{E}[X]| \geq \alpha] \leq \frac{\mathrm{Var}[X]}{\alpha²}\) 对于任何 \(\alpha > 0\)。这里，\(\alpha = 4\)，所以 \(\P[|X - 3| \geq 4] \leq \frac{4}{4²} = \frac{1}{4}\)。

E1.2.15 的答案和证明：\(X\) 的协方差矩阵是 \(\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}\)。由于 \(X_1\) 和 \(X_2\) 是独立的，它们的协方差为零。由于它们是标准正态分布，每个的方差都是 1。

E1.2.17 的答案和证明：\(\mathbb{E}[AX] = A\mathbb{E}[X] = A\mu_X = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} \begin{pmatrix} 1 \\ 2 \end{pmatrix} = \begin{pmatrix} 5 \\ 11 \end{pmatrix}\)，\(\text{Cov}[AX] = A\text{Cov}[X]A^T = A\Sigma_XA^T = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} \begin{pmatrix} 3 & 1 \\ 1 & 4 \end{pmatrix} \begin{pmatrix} 1 & 3 \\ 2 & 4 \end{pmatrix} = \begin{pmatrix} 11 & 19 \\ 19 & 35 \end{pmatrix}\)

E1.2.19 的答案和证明：对于任何非零向量 \(z = \begin{pmatrix} z_1 \\ z_2 \end{pmatrix}\)，我们有

$$\begin{align*} z^TAz &= \begin{pmatrix} z_1 & z_2 \end{pmatrix} \begin{pmatrix} 2 & -1 \\ -1 & 2 \end{pmatrix} \begin{pmatrix} z_1 \\ z_2 \end{pmatrix} \\ &= \begin{pmatrix} z_1 & z_2 \end{pmatrix} \begin{pmatrix} 2z_1 - z_2 \\ -z_1 + 2z_2 \end{pmatrix} \\ &= 2z_1² - 2z_1z_2 + 2z_2² \\ &= z_1² + (z_1 - z_2)² + z_2² > 0 \end{align*}$$

由于 \((z_1 - z_2)² \geq 0\)，且 \(z_1² > 0\) 或 \(z_2² > 0\)（因为 \(z \neq 0\)）。因此，\(A\) 是正定的。

对 E1.3.1 的答案和解释：\(\boldsymbol{\mu}_1^* = \frac{1}{2}(\mathbf{x}_1 + \mathbf{x}_4) = (\frac{3}{2}, 1)\)，\(\boldsymbol{\mu}_2^* = \frac{1}{2}(\mathbf{x}_2 + \mathbf{x}_3) = (-\frac{1}{2}, 0)\)。

对 E1.3.3 的答案和解释：\(C_1 = \{1, 5\}\)，\(C_2 = \{3\}\)，\(C_3 = \{2, 4\}\)。

对 E1.3.5 的答案和解释：展开平方范数并消去项得到等价的不等式。

对 E1.3.7 的答案和解释：展开 \(\|A + B\|_F² = \sum_{i=1}^n \sum_{j=1}^m (A_{ij} + B_{ij})²\) 并简化。

对 E1.3.9 的答案和解释：\(q(x) = 3(x-1)² + 2\)，在 \(x = 1\) 处的最小值为 2。

对 E1.3.11 的答案和解释：\(\|A\|_F = \sqrt{14}\)。

对 E1.4.1 的答案和解释：由于 \(X_i\) 在 \([-1/2, 1/2]\) 上均匀分布，其概率密度函数为 \(f_{X_i}(x) = 1\) 对于 \(x \in [-1/2, 1/2]\) 且其他情况为 0。因此，

$$\mathbb{E}[X_i] = \int_{-1/2}^{1/2} x f_{X_i}(x) dx = \int_{-1/2}^{1/2} x dx = 0$$

和

$$\text{Var}[X_i] = \mathbb{E}[X_i²] - (\mathbb{E}[X_i])² = \int_{-1/2}^{1/2} x² dx = \frac{1}{12}.$$

对 E1.4.3 的答案和解释：我们有

$$\mathbb{E}[\|X\|²] = \mathbb{E}\left[\sum_{i=1}^d X_i²\right] = \sum_{i=1}^d \mathbb{E}[X_i²] = \sum_{i=1}^d 1 = d.$$

对 E1.4.5 的答案和解释：根据**切比雪夫不等式**，\(P[|X - 1| \geq 3] \leq \frac{\mathrm{Var}[X]}{3²} = \frac{4}{9}\)。

对 E1.4.7 的答案和解释：根据**切比雪夫不等式**，\(\P[|X| \geq 2\sqrt{d}] \leq \frac{\mathrm{Var}[X]}{(2\sqrt{d})²} = \frac{1}{4d}\)。对于 \(d = 1\)，\(\P[|X| \geq 2] \leq \frac{1}{4}\)。对于 \(d = 10\)，\(P[|X| \geq 2\sqrt{10}] \leq \frac{1}{40}\)。对于 \(d = 100\)，\(\P[|X| \geq 20] \leq \frac{1}{400}\)。

### 1.6.1.5\. 学习成果#

+   计算向量和矩阵的范数，以及向量之间的内积。

+   利用范数和内积的性质，包括柯西-施瓦茨不等式和三角不等式。

+   执行基本的矩阵运算，包括加法、标量乘法、乘法和转置。

+   计算矩阵-向量积和矩阵-矩阵积，并解释它们的几何和代数意义。

+   解释下降方向的概念及其与优化的关系。

+   使用泰勒定理来近似函数并分析这些近似中的误差项。

+   定义并计算多变量函数的偏导数和梯度。

+   陈述并应用切比雪夫不等式来量化随机变量围绕其均值的集中度。

+   计算随机变量和向量的期望、方差和协方差。

+   定义并给出正定矩阵的例子。

+   解释随机向量协方差矩阵的性质和意义。

+   描述并从球形高斯分布中生成样本。

+   编写 Python 代码来计算范数、内积并执行矩阵运算。

+   使用 Python 模拟随机变量，计算它们的统计特性，并可视化分布和大量定律。

+   将 k-means 聚类问题表述为优化问题。

+   描述 k-means 算法及其寻找聚类中心和分配的迭代步骤。

+   推导固定划分的最优聚类代表（质心）。

+   推导固定代表的最优划分（聚类分配）。

+   分析 k-means 算法的收敛性质，并理解其在寻找全局最优解方面的局限性。

+   将 k-means 算法应用于现实世界的数据集，例如企鹅数据集，并解释结果。

+   以矩阵形式表达 k-means 目标函数，并将其与矩阵分解联系起来。

+   理解数据预处理步骤（如标准化）对于 k-means 的重要性。

+   认识到 k-means 的局限性，例如对初始化的敏感性以及需要提前指定聚类数量。

+   讨论确定 k-means 聚类中最佳聚类数量的挑战。

+   理解使用 k-means 算法聚类高维数据的挑战，并认识到维度增加如何导致噪声压倒信号，使得区分聚类变得困难。

+   理解高维空间中测度集中的概念，特别是大多数高维立方体的体积集中在其角落的令人费解的事实，使其看起来像“尖刺球”。

+   将切比雪夫不等式应用于推导从高维立方体中随机选择一点落在内切球内的概率，并理解随着维度增加，这个概率如何降低。

+   分析标准正态向量范数在高维空间中的行为，并使用切比雪夫不等式证明，尽管联合概率密度函数在原点处最大，但范数高度可能接近维度的平方根。

+   使用切比雪夫不等式在多维设置中推导概率界限。

+   模拟高维数据以经验验证理论结果。

\(\aleph\)

### 1.6.1.1\. 仅代码#

下面可以访问一个包含本章代码的交互式 Jupyter 笔记本（推荐使用 Google Colab）。鼓励您对其进行尝试。一些建议的计算练习散布在其中。笔记本也可以作为幻灯片查看。

+   [笔记本](https://github.com/MMiDS-textbook/MMiDS-textbook.github.io/blob/main/just_the_code/roch_mmids_chap_intro_notebook.ipynb) ([在 Colab 中打开](https://colab.research.google.com/github/MMiDS-textbook/MMiDS-textbook.github.io/blob/main/just_the_code/roch_mmids_chap_intro_notebook.ipynb))

+   [幻灯片](https://raw.githack.com/MMiDS-textbook/MMiDS-textbook.github.io/main/just_the_code/roch_mmids_chap_intro_notebook_slides.slides.html)

### 1.6.1.2\. 自我评估测验#

通过以下链接可以访问更全面的自我评估测验的网页版本。

+   [第 1.2 节](https://raw.githack.com/MMiDS-textbook/MMiDS-textbook.github.io/main/quizzes/self-assessment/quiz_1_2.html)

+   [第 1.3 节](https://raw.githack.com/MMiDS-textbook/MMiDS-textbook.github.io/main/quizzes/self-assessment/quiz_1_3.html)

+   [第 1.4 节](https://raw.githack.com/MMiDS-textbook/MMiDS-textbook.github.io/main/quizzes/self-assessment/quiz_1_4.html)

### 1.6.1.3\. 自动测验#

本章自动生成的测验可以在以下链接中访问（推荐使用 Google Colab）。

+   [自动测验](https://github.com/MMiDS-textbook/MMiDS-textbook.github.io/blob/main/quizzes/auto_quizzes/roch-mmids-intro-autoquiz.ipynb) ([在 Colab 中打开](https://colab.research.google.com/github/MMiDS-textbook/MMiDS-textbook.github.io/blob/main/quizzes/auto_quizzes/roch-mmids-intro-autoquiz.ipynb))

### 1.6.1.4\. 奇数编号预热练习的解答#

*(在 Claude、Gemini 和 ChatGPT 的帮助下)*

E1.2.1 的答案和解释：欧几里得范数 \(\|\mathbf{x}\|_2\) 由以下公式给出

$$ \|\mathbf{x}\|_2 = \sqrt{x_1² + x_2²} = \sqrt{6² + 8²} = \sqrt{36 + 64} = \sqrt{100} = 10. $$

E1.2.3 的答案和解释：矩阵 \(A\) 的转置 \(A^T\) 是通过交换行和列得到的

$$\begin{split} A^T = \begin{pmatrix}1 & 4 \\ 2 & 5 \\ 3 & 6\end{pmatrix} \end{split}$$

E1.2.5 的答案和解释：如果 \(A = A^T\)，则矩阵 \(A\) 是对称的

$$\begin{split} A^T = \begin{pmatrix}2 & 0 \\ 0 & 3\end{pmatrix} = A. \end{split}$$

因此，\(A\) 是对称的。

E1.2.7 的答案和解释：\(\frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x+h, y) - f(x, y)}{h} = \lim_{h \to 0} \frac{(x+h)² + (x+h)y + y² - (x² + xy + y²)}{h} = \lim_{h \to 0} \frac{2xh + h² + hy}{h} = 2x + y\).

\(\frac{\partial f}{\partial y} = \lim_{h \to 0} \frac{f(x, y+h) - f(x, y)}{h} = \lim_{h \to 0} \frac{x² + x(y+h) + (y+h)² - (x² + xy + y²)}{h} = \lim_{h \to 0} \frac{xh + 2yh + h²}{h} = x + 2y\).

答案和证明 E1.2.9：根据**泰勒定理**，\(f(x) = f(a) + (x - a)f'(a) + \frac{1}{2}(x - a)²f''(\xi)\)，其中 \(\xi\) 在 \(a\) 和 \(x\) 之间。这里，\(f(1) = 1³ - 3 \cdot 1² + 2 \cdot 1 = 0\)，\(f'(x) = 3x² - 6x + 2\)，所以 \(f'(1) = 3 - 6 + 2 = -1\)，并且 \(f''(x) = 6x - 6\)。因此，对于某个在 \(1\) 和 \(x\) 之间的 \(\xi\)，\(f(x) = 0 + (x - 1)(-1) + \frac{1}{2}(x - 1)²(6\xi - 6)\)。

答案和证明 E1.2.11：首先，计算 \(\E[X²]\)。

$$ \E[X²] = \sum_{x} x² \cdot P(X = x) = 1² \cdot 0.4 + 2² \cdot 0.6 = 0.4 + 4 \cdot 0.6 = 0.4 + 2.4 = 2.8. $$

然后，使用公式 \(\mathrm{Var}[X] = \E[X²] - (\E[X])²\)

$$ \mathrm{Var}[X] = 2.8 - (1.6)² = 2.8 - 2.56 = 0.24. $$

答案和证明 E1.2.13：根据**切比雪夫不等式**，对于任何 \(\alpha > 0\)，\(\P[|X - \mathbb{E}[X]| \geq \alpha] \leq \frac{\mathrm{Var}[X]}{\alpha²}\)。这里，\(\alpha = 4\)，所以 \(\P[|X - 3| \geq 4] \leq \frac{4}{4²} = \frac{1}{4}\)。

答案和证明 E1.2.15：\(X\) 的协方差矩阵为 \(\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}\)。由于 \(X_1\) 和 \(X_2\) 是独立的，它们的协方差为零。由于它们是标准正态分布，每个的方差都是 1。

答案和证明 E1.2.17：\(\mathbb{E}[AX] = A\mathbb{E}[X] = A\mu_X = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} \begin{pmatrix} 1 \\ 2 \end{pmatrix} = \begin{pmatrix} 5 \\ 11 \end{pmatrix}\)，\(\text{Cov}[AX] = A\text{Cov}[X]A^T = A\Sigma_XA^T = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} \begin{pmatrix} 3 & 1 \\ 1 & 4 \end{pmatrix} \begin{pmatrix} 1 & 3 \\ 2 & 4 \end{pmatrix} = \begin{pmatrix} 11 & 19 \\ 19 & 35 \end{pmatrix}\)

答案和证明 E1.2.19：对于任何非零向量 \(z = \begin{pmatrix} z_1 \\ z_2 \end{pmatrix}\)，我们有

$$\begin{align*} z^TAz &= \begin{pmatrix} z_1 & z_2 \end{pmatrix} \begin{pmatrix} 2 & -1 \\ -1 & 2 \end{pmatrix} \begin{pmatrix} z_1 \\ z_2 \end{pmatrix} \\ &= \begin{pmatrix} z_1 & z_2 \end{pmatrix} \begin{pmatrix} 2z_1 - z_2 \\ -z_1 + 2z_2 \end{pmatrix} \\ &= 2z_1² - 2z_1z_2 + 2z_2² \\ &= z_1² + (z_1 - z_2)² + z_2² > 0 \end{align*}$$

由于 \((z_1 - z_2)² \geq 0\)，并且 \(z_1² > 0\) 或 \(z_2² > 0\)（因为 \(z \neq 0\)）。因此，\(A\) 是正定的。

答案和证明 E1.3.1：\(\boldsymbol{\mu}_1^* = \frac{1}{2}(\mathbf{x}_1 + \mathbf{x}_4) = (\frac{3}{2}, 1)\)，\(\boldsymbol{\mu}_2^* = \frac{1}{2}(\mathbf{x}_2 + \mathbf{x}_3) = (-\frac{1}{2}, 0)\)。

答案和证明 E1.3.3：\(C_1 = \{1, 5\}\)，\(C_2 = \{3\}\)，\(C_3 = \{2, 4\}\)。

答案和证明 E1.3.5：展开平方范数并消去项得到等价的不等式。

答案和证明 E1.3.7：展开 \(\|A + B\|_F² = \sum_{i=1}^n \sum_{j=1}^m (A_{ij} + B_{ij})²\) 并简化。

答案和证明 E1.3.9：\(q(x) = 3(x-1)² + 2\)，最小值在 \(x = 1\) 时为 2。

答案和证明 E1.3.11：\(\|A\|_F = \sqrt{14}\)。

对 E1.4.1 的答案和解释：由于 \(X_i\) 在 \([-1/2, 1/2]\) 上均匀分布，其概率密度函数为 \(f_{X_i}(x) = 1\)，对于 \(x \in [-1/2, 1/2]\) 且其他情况为 0。因此，

$$\mathbb{E}[X_i] = \int_{-1/2}^{1/2} x f_{X_i}(x) dx = \int_{-1/2}^{1/2} x dx = 0$$

和

$$\text{Var}[X_i] = \mathbb{E}[X_i²] - (\mathbb{E}[X_i])² = \int_{-1/2}^{1/2} x² dx = \frac{1}{12}.$$

对 E1.4.3 的答案和解释：我们有

$$\mathbb{E}[\|X\|²] = \mathbb{E}\left[\sum_{i=1}^d X_i²\right] = \sum_{i=1}^d \mathbb{E}[X_i²] = \sum_{i=1}^d 1 = d.$$

对 E1.4.5 的答案和解释：根据**切比雪夫不等式**，\(P[|X - 1| \geq 3] \leq \frac{\mathrm{Var}[X]}{3²} = \frac{4}{9}\)。

对 E1.4.7 的答案和解释：根据**切比雪夫不等式**，\(\P[|X| \geq 2\sqrt{d}] \leq \frac{\mathrm{Var}[X]}{(2\sqrt{d})²} = \frac{1}{4d}\)。对于 \(d = 1\)，\(\P[|X| \geq 2] \leq \frac{1}{4}\)。对于 \(d = 10\)，\(P[|X| \geq 2\sqrt{10}] \leq \frac{1}{40}\)。对于 \(d = 100\)，\(\P[|X| \geq 20] \leq \frac{1}{400}\)。

### 1.6.1.5\. 学习成果#

+   计算向量和矩阵的范数，以及向量之间的内积。

+   利用范数和内积的性质，包括柯西-施瓦茨不等式和三角不等式。

+   执行基本的矩阵运算，包括加法、标量乘法、乘法和转置。

+   计算矩阵-向量积和矩阵-矩阵积，并解释它们的几何和代数意义。

+   解释下降方向的概念及其与优化的关系。

+   使用泰勒定理来近似函数并分析这些近似中的误差项。

+   定义并计算多变量函数的偏导数和梯度。

+   陈述并应用切比雪夫不等式来量化随机变量围绕其均值集中的集中度。

+   计算随机变量和向量的期望、方差和协方差。

+   定义并给出正半定矩阵的例子。

+   解释随机向量协方差矩阵的性质和意义。

+   描述并从球面高斯分布中生成样本。

+   编写 Python 代码来计算范数、内积和执行矩阵运算。

+   使用 Python 模拟随机变量，计算它们的统计属性，并可视化分布和大量定律。

+   将 k-means 聚类问题表述为优化问题。

+   描述 k-means 算法及其寻找聚类中心和分配的迭代步骤。

+   推导固定划分的最优聚类代表（质心）。

+   推导固定代表的最优划分（聚类分配）。

+   分析 k-means 算法的收敛性质，并理解其在寻找全局最优解方面的局限性。

+   将 k-means 算法应用于实际数据集，例如企鹅数据集，并解释结果。

+   将 k-means 目标函数表示为矩阵形式，并将其与矩阵分解联系起来。

+   理解数据预处理步骤（如标准化）对于 k-means 的重要性。

+   认识到 k-means 的局限性，例如其对初始化的敏感性以及需要提前指定簇的数量。

+   讨论确定 k-means 聚类中最佳簇数量的挑战。

+   理解使用 k-means 算法对高维数据进行聚类的挑战，并认识到维度增加如何导致噪声压倒信号，使得区分簇变得困难。

+   理解高维空间中测度集中的概念，特别是大多数高维立方体的体积都集中在其角落，使其看起来像“尖锐球体”这一令人费解的事实。

+   应用切比雪夫不等式推导从高维立方体中随机选择一点落在内切球内的概率，并理解随着维度增加，这个概率如何降低。

+   分析标准正态向量范数在高维空间中的行为，并使用切比雪夫不等式证明，尽管联合概率密度函数在原点处达到最大值，但范数很可能接近维度的平方根。

+   在高维设置中使用切比雪夫不等式推导概率界限。

+   通过模拟高维数据来经验性地验证理论结果。

\(\aleph\)

## 1.6.2\. 其他部分#

### 1.6.2.1\. 高维聚类更严格的分析#

在本可选部分，我们给出前一小节所述现象的一个正式陈述。

**定理** 设 \(\mathbf{X}_1, \mathbf{X}_2, \mathbf{Y}_1\) 是独立的球形 \(d\)-维高斯分布，均值为 \(-w_d \mathbf{e}_1\)，方差为 \(1\)，其中 \(\{w_d\}\) 是 \(d\) 中的单调序列。设 \(\mathbf{Y}_2\) 是独立的球形 \(d\)-维高斯分布，均值为 \(w_d \mathbf{e}_1\)，方差为 \(1\)。令 \(\Delta_d = \|\mathbf{Y}_1 - \mathbf{Y}_2\|² - \|\mathbf{X}_1 - \mathbf{X}_2\|²\)，当 \(d \to +\infty\) 时

$$\begin{split} \frac{\mathbb{E}[\Delta_d]}{\sqrt{\mathrm{Var}[\Delta_d]}} \to \begin{cases} 0, & \text{如果 $w_d \ll d^{1/4}$}\\ +\infty, & \text{如果 $w_d \gg d^{1/4}$} \end{cases} \end{split}$$

其中 \(w_d \ll d^{1/4}\) 表示 \(w_d/d^{1/4} \to 0\)。 \(\sharp\)

这个比率被称为[信噪比](https://en.wikipedia.org/wiki/Signal-to-noise_ratio)。

为了证明这一主张，我们需要以下性质。

**引理** 设 \(W\) 是关于零对称的实值随机变量，即 \(W\) 和 \(-W\) 是同分布的。那么对于所有奇数 \(k\)，\(\mathbb{E}[W^k] = 0\)。\(\flat\)

*证明：* 通过对称性，

$$ \mathbb{E}[W^k] = \mathbb{E}[(-W)^k] = \mathbb{E}[(-1)^k W^k] = - \mathbb{E}[W^k]. $$

满足这个方程的唯一方法是 \(\mathbb{E}[W^k] = 0\)。\(\square\)

返回到命题的证明：

*证明思路：* *(定理)* 由于期望的线性性质，只有第一个坐标对 \(\mathbb{E}[\Delta_d]\) 有贡献，而所有坐标都对 \(\mathrm{Var}[\Delta_d]\) 有贡献。更具体地说，计算表明前者是 \(c_0 w²\)，而后者是 \(c_1 w² + c_2 d\)，其中 \(c_0, c_1, c_2\) 是常数。

*证明：* *(命题)* 写 \(w := w_d\) 和 \(\Delta := \Delta_d\) 以简化符号。有两个步骤：

*(1) \(\Delta\) 的期望：* 根据定义，随机变量 \(X_{1,i} - X_{2,i}\)，\(i = 1,\ldots, d\)，和 \(Y_{1,i} - Y_{2,i}\)，\(i = 2,\ldots, d\)，具有相同的分布。因此，根据期望的线性性质，

$$\begin{align*} \mathbb{E}[\Delta] &= \sum_{i=1}^d \mathbb{E}[(Y_{1,i} - Y_{2,i})²] - \sum_{i=1}^d \mathbb{E}[(X_{1,i} - X_{2,i})²]\\ &= \mathbb{E}[(Y_{1,1} - Y_{2,1})²] - \mathbb{E}[(X_{1,1} - X_{2,1})²]. \end{align*}$$

此外，我们可以写出 \(Y_{1,1} - Y_{1,2} \sim (Z_1 -w) - (Z_2+w)\)，其中 \(Z_1, Z_2 \sim N(0,1)\) 是独立的，其中这里的 \(\sim\) 表示分布上的等价。因此，我们有

$$\begin{align*} \mathbb{E}[(Y_{1,1} - Y_{2,1})²] &= \mathbb{E}[(Z_1 - Z_2 - 2w)²]\\ &= \mathbb{E}[(Z_1 - Z_2)²] - 4w \,\mathbb{E}[Z_1 - Z_2] + 4 w². \end{align*}$$

类似地，\(X_{1,1} - X_{1,2} \sim Z_1 - Z_2\)，所以 \(\mathbb{E}[(X_{1,1} - X_{2,1})²] = \mathbb{E}[(Z_1 - Z_2)²]\)。由于 \(\mathbb{E}[Z_1 - Z_2] = 0\)，我们最终得到 \(\mathbb{E}[\Delta] = 4 w²\)。

*(2) \(\Delta\) 的方差：* 使用 (1) 中的观察结果和坐标的独立性，我们得到

$$\begin{align*} \mathrm{Var}[\Delta] &= \sum_{i=1}^d \mathrm{Var}[(Y_{1,i} - Y_{2,i})²] + \sum_{i=1}^d \mathrm{Var}[(X_{1,i} - X_{2,i})²]\\ &= \mathrm{Var}[(Z_1 - Z_2 - 2w)²] + (2d-1) \,\mathrm{Var}[(Z_1 - Z_2)²]. \end{align*}$$

根据方差的性质，

$$\begin{align*} \mathrm{Var}[(Z_1 - Z_2 - 2w)²] &= \mathrm{Var}[(Z_1 - Z_2)² - 4w(Z_1 - Z_2) + 4w²]\\ &= \mathrm{Var}[(Z_1 - Z_2)² - 4w(Z_1 - Z_2)]\\ &= \mathrm{Var}[(Z_1 - Z_2)²] + 16 w² \mathrm{Var}[Z_1 - Z_2]\\ &\quad - 8w \,\mathrm{Cov}[(Z_1 - Z_2)², Z_1 - Z_2]. \end{align*}$$

因为 \(Z_1\) 和 \(Z_2\) 是独立的，\(\mathrm{Var}[Z_1 - Z_2] = \mathrm{Var}[Z_1] + \mathrm{Var}[Z_2] = 2\)。此外，随机变量 \((Z_1 - Z_2)\) 是对称的，所以

$$\begin{align*} \mathrm{Cov}[(Z_1 - Z_2)², Z_1 - Z_2] &= \mathbb{E}[(Z_1 - Z_2)³]\\ & \quad - \mathbb{E}[(Z_1 - Z_2)²] \,\mathbb{E}[Z_1 - Z_2]\\ &= 0. \end{align*}$$

最后，

$$ \mathrm{Var}[\Delta] = 32 w² + 2d \,\mathrm{Var}[(Z_1 - Z_2)²] $$

*将所有内容综合起来：*

$$ \frac{\mathbb{E}[\Delta]}{\sqrt{\mathrm{Var}[\Delta]}} = \frac{4 w²}{\sqrt{32 w² + 2d \,\mathrm{Var}[(Z_1 - Z_2)²]}}. $$

取 \(d \to +\infty\) 得到断言。 \(\square\)

### 1.6.2.1\. 高维聚类更严格的分析#

在本可选部分，我们给出前一小节所述现象的一个正式陈述。

**定理** 设 \(\mathbf{X}_1, \mathbf{X}_2, \mathbf{Y}_1\) 是独立的球形 \(d\)-维高斯分布，均值为 \(-w_d \mathbf{e}_1\)，方差为 \(1\)，其中 \(\{w_d\}\) 是 \(d\) 中的单调序列。设 \(\mathbf{Y}_2\) 是独立的球形 \(d\)-维高斯分布，均值为 \(w_d \mathbf{e}_1\)，方差为 \(1\)。然后，令 \(\Delta_d = \|\mathbf{Y}_1 - \mathbf{Y}_2\|² - \|\mathbf{X}_1 - \mathbf{X}_2\|²\)，当 \(d \to +\infty\) 时

$$\begin{split} \frac{\mathbb{E}[\Delta_d]}{\sqrt{\mathrm{Var}[\Delta_d]}} \to \begin{cases} 0, & \text{如果 $w_d \ll d^{1/4}$}\\ +\infty, & \text{如果 $w_d \gg d^{1/4}$} \end{cases} \end{split}$$

其中 \(w_d \ll d^{1/4}\) 表示 \(w_d/d^{1/4} \to 0\). \(\sharp\)

这个比率就是所说的信噪比。[信号与噪声比](https://en.wikipedia.org/wiki/Signal-to-noise_ratio)。

为了证明这个断言，我们需要以下性质。

**引理** 设 \(W\) 是关于零对称的实值随机变量，即 \(W\) 和 \(-W\) 具有相同的分布。那么对于所有奇数 \(k\)，\(\mathbb{E}[W^k] = 0\)。 \(\flat\)

*证明：* 通过对称性，

$$ \mathbb{E}[W^k] = \mathbb{E}[(-W)^k] = \mathbb{E}[(-1)^k W^k] = - \mathbb{E}[W^k]. $$

满足此方程的唯一方法是 \(\mathbb{E}[W^k] = 0\). \(\square\)

回到断言的证明：

*证明思路：* *(定理)* 对 \(\mathbb{E}[\Delta_d]\) 贡献的唯一坐标是第一个，因为期望的线性性质，而所有坐标都对 \(\mathrm{Var}[\Delta_d]\) 贡献。更具体地说，计算表明前者是 \(c_0 w²\)，而后者是 \(c_1 w² + c_2 d\)，其中 \(c_0, c_1, c_2\) 是常数。

*证明：* *(断言)* 将 \(w := w_d\) 和 \(\Delta := \Delta_d\) 写出来以简化符号。有两个步骤：

*(1) \(\Delta\) 的期望：* 根据定义，随机变量 \(X_{1,i} - X_{2,i}\)，\(i = 1,\ldots, d\)，和 \(Y_{1,i} - Y_{2,i}\)，\(i = 2,\ldots, d\)，具有相同的分布。因此，根据期望的线性性质，

$$\begin{align*} \mathbb{E}[\Delta] &= \sum_{i=1}^d \mathbb{E}[(Y_{1,i} - Y_{2,i})²] - \sum_{i=1}^d \mathbb{E}[(X_{1,i} - X_{2,i})²]\\ &= \mathbb{E}[(Y_{1,1} - Y_{2,1})²] - \mathbb{E}[(X_{1,1} - X_{2,1})²]. \end{align*}$$

此外，我们可以写出 \(Y_{1,1} - Y_{1,2} \sim (Z_1 -w) - (Z_2+w)\)，其中 \(Z_1, Z_2 \sim N(0,1)\) 是独立的，其中 \(\sim\) 表示分布上的等价。因此，我们有

$$\begin{align*} \mathbb{E}[(Y_{1,1} - Y_{2,1})²] &= \mathbb{E}[(Z_1 - Z_2 - 2w)²]\\ &= \mathbb{E}[(Z_1 - Z_2)²] - 4w \,\mathbb{E}[Z_1 - Z_2] + 4 w². \end{align*}$$

同样，\(X_{1,1} - X_{1,2} \sim Z_1 - Z_2\)，因此 \(\mathbb{E}[(X_{1,1} - X_{2,1})²] = \mathbb{E}[(Z_1 - Z_2)²]\)。由于 \(\mathbb{E}[Z_1 - Z_2] = 0\)，我们最终得到 \(\mathbb{E}[\Delta] = 4 w²\)。

*(2) \(\Delta\) 的方差：* 利用(1)中的观察结果和坐标的独立性，我们得到

$$\begin{align*} \mathrm{Var}[\Delta] &= \sum_{i=1}^d \mathrm{Var}[(Y_{1,i} - Y_{2,i})²] + \sum_{i=1}^d \mathrm{Var}[(X_{1,i} - X_{2,i})²]\\ &= \mathrm{Var}[(Z_1 - Z_2 - 2w)²] + (2d-1) \,\mathrm{Var}[(Z_1 - Z_2)²]. \end{align*}$$

根据方差的和，

$$\begin{align*} \mathrm{Var}[(Z_1 - Z_2 - 2w)²] &= \mathrm{Var}[(Z_1 - Z_2)² - 4w(Z_1 - Z_2) + 4w²]\\ &= \mathrm{Var}[(Z_1 - Z_2)² - 4w(Z_1 - Z_2)]\\ &= \mathrm{Var}[(Z_1 - Z_2)²] + 16 w² \mathrm{Var}[Z_1 - Z_2]\\ &\quad - 8w \,\mathrm{Cov}[(Z_1 - Z_2)², Z_1 - Z_2]. \end{align*}$$

因为 \(Z_1\) 和 \(Z_2\) 是独立的，\(\mathrm{Var}[Z_1 - Z_2] = \mathrm{Var}[Z_1] + \mathrm{Var}[Z_2] = 2\)。此外，随机变量 \((Z_1 - Z_2)\) 是对称的，所以

$$\begin{align*} \mathrm{Cov}[(Z_1 - Z_2)², Z_1 - Z_2] &= \mathbb{E}[(Z_1 - Z_2)³]\\ & \quad - \mathbb{E}[(Z_1 - Z_2)²] \,\mathbb{E}[Z_1 - Z_2]\\ &= 0. \end{align*}$$

最后，

$$ \mathrm{Var}[\Delta] = 32 w² + 2d \,\mathrm{Var}[(Z_1 - Z_2)²] $$

*将所有内容综合起来：*

$$ \frac{\mathbb{E}[\Delta]}{\sqrt{\mathrm{Var}[\Delta]}} = \frac{4 w²}{\sqrt{32 w² + 2d \,\mathrm{Var}[(Z_1 - Z_2)²]}}. $$

当 \(d \to +\infty\) 时，得到所声称的结果。 \(\square\)
