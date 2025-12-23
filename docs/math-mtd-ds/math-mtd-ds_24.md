# 3.7\. 练习题#

> 原文：[`mmids-textbook.github.io/chap03_opt/exercises/roch-mmids-opt-exercises.html`](https://mmids-textbook.github.io/chap03_opt/exercises/roch-mmids-opt-exercises.html)

## 3.7.1\. 预习工作表#

*(由 Claude, Gemini 和 ChatGPT 协助)*

**第 3.2 节**

**E3.2.1** 计算函数 \(f(x_1, x_2) = 3x_1² - 2x_1x_2 + 4x_2² - 5x_1 + 2x_2\) 在点 \((1, -1)\) 处的梯度。

**E3.2.2** 求函数 \(f(x_1, x_2, x_3) = 2x_1x_2 + 3x_2x_3 - x_1² - 4x_3²\) 的 Hessian 矩阵。

**E3.2.3** 计算函数 \(f(x_1, x_2) = \sin(x_1) \cos(x_2)\) 在点 \((\frac{\pi}{4}, \frac{\pi}{3})\) 处的偏导数 \(\frac{\partial f}{\partial x_1}\) 和 \(\frac{\partial f}{\partial x_2}\)。

**E3.2.4** 设 \(f(x_1, x_2) = e^{x_1} + x_1x_2 - x_2²\) 和 \(\mathbf{g}(t) = (t², t)\)。使用**链式法则**计算 \((f \circ \mathbf{g})'(1)\)。

**E3.2.5** 验证函数 \(f(x_1, x_2) = x_1³ + 3x_1x_2² - 2x_2³\) 在点 \((1, 2)\) 处的**Hessian 矩阵**的**对称性**。

**E3.2.6** 求函数 \(f(x_1, x_2, x_3) = 2x_1 - 3x_2 + 4x_3\) 在点 \((1, -2, 3)\) 处的梯度。

**E3.2.7** 计算函数 \(f(x_1, x_2) = x_1² \sin(x_2)\) 的二阶偏导数。

**E3.2.8** 计算二次函数 \(f(x_1, x_2) = 3x_1² + 2x_1x_2 - x_2² + 4x_1 - 2x_2\) 在点 \((1, -1)\) 处的梯度。

**E3.2.9** 求函数 \(f(x_1, x_2, x_3) = x_1² + 2x_2² + 3x_3² - 2x_1x_2 + 4x_1x_3 - 6x_2x_3\) 的 Hessian 矩阵。

**E3.2.10** 对函数 \(f(x_1, x_2) = x_1² + x_2²\) 在连接点 \((0, 0)\) 和 \((1, 2)\) 的线段上应用**多元平均值定理**。

**E3.2.11** 求函数 \(f(x, y) = x³y² - 2xy³ + y⁴\) 的偏导数 \(\frac{\partial f}{\partial x}\) 和 \(\frac{\partial f}{\partial y}\)。

**E3.2.12** 计算函数 \(f(x, y) = \ln(x² + 2y²)\) 在点 \((1, 2)\) 处的梯度 \(\nabla f(x, y)\)。

**E3.2.13** 求函数 \(g(x, y) = \sin(x) \cos(y)\) 的 Hessian 矩阵。

**E3.2.14** 如果 \(p(x, y, z) = x²yz + 3xyz²\)，求 \(p\) 的所有二阶偏导数。

**E3.2.15** 验证函数 \(q(x, y) = x³ - 3xy²\) 满足拉普拉斯方程：\(\frac{\partial² q}{\partial x²} + \frac{\partial² q}{\partial y²} = 0\).

**E3.2.16** 考虑函数 \(s(x, y) = x² + 4xy + 5y²\)。使用**平均值定理**（其中 \(\xi = 0\)）通过点 \((1, 1)\) 近似 \(s(1.1, 0.9)\)。

**E3.2.17** 一个粒子沿着路径 \(\mathbf{c}(t) = (t², t³)\) 运动。点 \((x, y)\) 的温度由 \(u(x, y) = e^{-x² - y²}\) 给出。求粒子在时间 \(t = 1\) 时温度变化的速率。

**E3.2.18** 对函数 \(f(x, y) = x² + y²\) 和 \(\mathbf{g}(t) = (t, \sin t)\) 应用**链式法则**求 \(\frac{d}{dt} f(\mathbf{g}(t))\).

**E3.2.19** 使用**链式法则**求\(f(x, y) = xy\)和\(\mathbf{g}(t) = (t², \cos t)\)的导数\(\frac{d}{dt} f(\mathbf{g}(t))\)。

**第 3.3 节**

**E3.3.1** 考虑函数\(f(x_1, x_2) = x_1² + 2x_2²\)。找出所有使得\(\nabla f(x_1, x_2) = 0\)的点\((x_1, x_2)\)。

**E3.3.2** 设\(f(x_1, x_2) = x_1² - x_2²\)。求\(f\)在点\((1, 1)\)沿方向\(\mathbf{v} = (\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}})\)的方向导数。

**E3.3.3** 考虑函数\(f(x_1, x_2) = x_1² + 2x_1x_2 + x_2²\)。求\(f\)在点\((1, 1)\)沿方向\(\mathbf{v} = (\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}})\)的第二方向导数。

**E3.3.4** 考虑优化问题\(\min_{x_1, x_2} x_1² + x_2²\)，受限于\(x_1 + x_2 = 1\)。写出拉格朗日函数\(L(x_1, x_2, \lambda)\)。

**E3.3.5** 对于 E3.3.4 中的优化问题，找出满足一阶必要条件的所有点\((x_1, x_2, \lambda)\)。

**E3.3.6** 对于 E3.3.4 中的优化问题，检查点\((\frac{1}{2}, \frac{1}{2}, -1)\)处的二阶充分条件。

**E3.3.7** 考虑函数\(f(x_1, x_2, x_3) = x_1² + x_2² + x_3²\)。找出满足优化问题\(\min_{x_1, x_2, x_3} f(x_1, x_2, x_3)\)受限于\(x_1 + 2x_2 + 3x_3 = 6\)的一阶必要条件的所有点\((x_1, x_2, x_3)\)。

**E3.3.8** 对于 E3.3.7 中的优化问题，检查点\((\frac{1}{2}, 1, \frac{3}{2}, -1)\)处的二阶充分条件。

**E3.3.9** 设\(f: \mathbb{R}² \to \mathbb{R}\)由\(f(x_1, x_2) = x_1³ - 3x_1x_2²\)定义。确定向量\(\mathbf{v} = (1, 1)\)是否是\(f\)在点\((1, 0)\)的下降方向。

**E3.3.10** 设\(f: \mathbb{R}² \to \mathbb{R}\)由\(f(x_1, x_2) = x_1² + x_2² - 2x_1 - 4x_2 + 5\)定义。求\(f\)的 Hessian 矩阵。

**E3.3.11** 设\(f: \mathbb{R}² \to \mathbb{R}\)由\(f(x_1, x_2) = -x_1² - x_2²\)定义。计算\(f\)在点\((0, 0)\)沿方向\(\mathbf{v} = (1, 0)\)的第二方向导数。

**E3.3.12** 计算函数\(f(x,y) = x² + y²\)在点\((1,1)\)沿方向\(\mathbf{v} = (1,1)\)的方向导数。

**E3.3.13** 确定函数\(f(x,y) = x³ + y³ - 3xy\)在点\((1,1)\)处的 Hessian 矩阵。

**第 3.4 节**

**E3.4.1** 找出点\((2, 3)\)和\((4, 5)\)在\(\alpha = 0.3\)条件下的凸组合。

**E3.4.2** 判断以下集合是否是凸集：

\[ S = \{(x_1, x_2) \in \mathbb{R}² : x_1² + x_2² \leq 1\}. \]

**E3.4.3** 设\(S_1 = \{x \in \mathbb{R}² : x_1 + x_2 \leq 1\}\)和\(S_2 = \{x \in \mathbb{R}² : x_1 - x_2 \leq 1\}\)。直接证明\(S_1 \cap S_2\)是一个凸集。

**E3.4.4** 使用 Hessian 矩阵验证函数\(f(\mathbf{x}) = \log(e^{x_1} + e^{x_2})\)是凸函数。

**E3.4.5** 找出函数\(f(x) = x² + 2x + 1\)的全局最小值。

**E3.4.6** 考虑函数 \(f(x) = |x|\)。证明 \(f\) 是凸的但在 \(x = 0\) 处不可微。

**E3.4.7** 验证函数 \(f(x, y) = x² + 2y²\) 在 \(m = 2\) 时是强凸的。

**E3.4.8** 找到点 \(x = (1, 2)\) 投影到集合 \(D = \{(x_1, x_2) \in \mathbb{R}² : x_1 + x_2 = 1\}\) 的投影。

**E3.4.9** 设 \(f(x) = x⁴ - 2x² + 1\)。判断 \(f\) 是否是凸的。

**E3.4.10** 设 \(f(x) = e^x\)。判断 \(f\) 是否是对数凹的，即 \(\log f\) 是凹的。

**E3.4.11** 设 \(f(x) = x² - 2x + 2\)。判断 \(f\) 是否是强凸的。

**E3.4.12** 设 \(A = \begin{pmatrix} 1 & 2 \\ 2 & 5 \end{pmatrix}\) 和 \(\mathbf{b} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}\)。找到使 \(\|A\mathbf{x} - \mathbf{b}\|_2²\) 最小的向量 \(x\)。

**E3.4.13** 给定集合 \(D = \{(x, y) \in \mathbb{R}² : x² + y² < 4\}\)，判断 \(D\) 是否是凸集。

**E3.4.14** 判断集合 \(D = \{(x, y) \in \mathbb{R}² : x + y \leq 1\}\) 是否是凸集。

**E3.4.15** 判断集合 \(D = \{(x, y) \in \mathbb{R}² : x² - y² \leq 1\}\) 是否是凸集。

**第 3.5 节**

**E3.5.1** 给定函数 \(f(x, y) = x² + 4y²\)，找到在点 \((1, 1)\) 处的最速下降方向。

**E3.5.2** 考虑函数 \(f(x) = x² + 4x + 5\)。计算 \(f'\)(x) 并找到在 \(x_0 = 1\) 处的最速下降方向。

**E3.5.3** 给定函数 \(f(x) = x³ - 6x² + 9x + 2\)，从 \(x_0 = 0\) 开始，进行两次梯度下降迭代，步长为 \(\alpha = 0.1\)。

**E3.5.4** 设 \(f(x) = x²\)。证明 \(f\) 对于某个 \(L > 0\) 是 \(L\)-光滑的。

**E3.5.5** 设 \(f(x) = x²\)。从 \(x_0 = 2\) 开始，进行一步梯度下降，步长为 \(\alpha = 0.1\)。

**E3.5.6** 考虑 \(2\)-光滑函数 \(f(x) = x²\)。从 \(x_0 = 3\) 开始，计算梯度下降法经过一步迭代且步长为 \(\alpha = \frac{1}{2}\) 后得到的点 \(x_1\)。

**E3.5.7** 验证函数 \(f(x) = 2x² + 1\) 是 \(4\)-强凸的。

**E3.5.8** 对于全局最小值为 \(x^* = 0\) 的 \(4\)-强凸函数 \(f(x) = 2x² + 1\)，计算 \(\frac{1}{2m} |f'(1)|²\) 并将其与 \(f(1) - f(x^*)\) 进行比较。

**E3.5.9** 设 \(f(x) = x⁴\)。\(f\) 对于某个 \(L > 0\) 是 \(L\)-光滑的吗？证明你的答案。

**E3.5.10** 设 \(f(x) = x³\)。\(f\) 对于某个 \(m > 0\) 是 \(m\)-强凸的吗？证明你的答案。

**E3.5.11** 设 \(f(x) = x² + 2x + 1\)。证明 \(f\) 对于某个 \(m > 0\) 是 \(m\)-强凸的。

**E3.5.12** 设 \(f(x) = x² + 2x + 1\)。从 \(x_0 = -3\) 开始，进行一步梯度下降，步长为 \(\alpha = 0.2\)。

**E3.5.13** 给定函数 \(f(x, y) = x² + y²\)，从 \((1, 1)\) 开始，进行两步梯度下降，步长为 \(\alpha = 0.1\)。

**E3.5.14** 对于一个二次函数\(f(x) = ax² + bx + c\)，其中\(a > 0\)，推导出梯度下降在一步内收敛到最小值的\(\alpha\)值。

**第 3.6 节**

**E3.6.1** 计算概率为\(p = 0.25\)的事件的对数几率。

**E3.6.2** 给定特征\(\mathbf{x} = (2, -1)\)和系数\(\boldsymbol{\alpha} = (0.5, -0.3)\)，计算线性组合\(\boldsymbol{\alpha}^T \mathbf{x}\)。

**E3.6.3** 对于数据点\(\mathbf{x} = (1, 3)\)，标签为\(b = 1\)，系数为\(\boldsymbol{\alpha} = (-0.2, 0.4)\)，使用 sigmoid 函数计算概率\(p(\mathbf{x}; \boldsymbol{\alpha})\)。

**E3.6.4** 计算单个数据点\(\mathbf{x} = (1, 2)\)的交叉熵损失，标签为\(b = 0\)，给定\(\boldsymbol{\alpha} = (0.3, -0.4)\)。

**E3.6.5** 证明逻辑函数\(\sigma(z) = \frac{1}{1 + e^{-z}}\)满足\(\sigma'(z) = \sigma(z)(1 - \sigma(z))\)。

**E3.6.6** 给定特征向量\(\boldsymbol{\alpha}_1 = (1, 2)\)，\(\boldsymbol{\alpha}_2 = (-1, 1)\)，\(\boldsymbol{\alpha}_3 = (0, -1)\)，以及相应的标签\(b_1 = 1\)，\(b_2 = 0\)，\(b_3 = 1\)，计算\(x = (1, 1)\)时的逻辑回归目标函数\(\ell(\mathbf{x}; A, \mathbf{b})\)。

**E3.6.7** 对于与 E3.6.6 相同的资料，计算逻辑回归目标函数梯度\(\nabla \ell(\mathbf{x}; A, b)\)在\(\mathbf{x} = (1, 1)\)处的值。

**E3.6.8** 对于与 E3.6.6 相同的资料，计算逻辑回归目标函数在\(\mathbf{x} = (1, 1)\)处的 Hessian 矩阵\(\mathbf{H}_\ell(\mathbf{x}; A, \mathbf{b})\)。

**E3.6.9** 对于与 E3.6.6 相同的资料，从\(\mathbf{x}⁰ = (0, 0)\)开始，执行一步梯度下降，步长为\(\beta = 0.1\)。

**E3.6.10** 对于与 E3.6.6 相同的资料，计算逻辑回归目标函数的平滑参数\(L\)。

## 3.7.2\. 问题#

**3.1** 设\(f : \mathbb{R} \to\mathbb{R}\)是二阶连续可微的，设\(\mathbf{a}_1, \mathbf{a}_2\)是\(\mathbb{R}^d\)中的向量，设\(b_1, b_2 \in \mathbb{R}\)。考虑以下定义在\(\mathbb{R}^d\)中的实值函数\(\mathbf{x} \in \mathbb{R}^d\)：

\[ g(\mathbf{x}) = \frac{1}{2} f(\mathbf{a}_1^T \mathbf{x} + b_1) + \frac{1}{2} f(\mathbf{a}_2^T \mathbf{x} + b_2). \]

a) 用\(f\)的导数\(f'\)表示\(g\)的梯度（通过向量形式，我们指的是不能单独写出\(\nabla g(\mathbf{x})\)的每个元素）。

b) 用\(f\)的一阶导数\(f'\)和二阶导数\(f''\)表示\(g\)的 Hessian 矩阵（通过矩阵形式，我们指的是不能单独写出\(\mathbf{H}_g (\mathbf{x})\)的每个元素或其列）。

**3.2** 使用方向导数的定义给出 *下降方向和方向导数引理* 的另一种证明。[提示：模仿单变量情况的证明。]\(\lhd\)

**3.3** (改编自 [CVX]) 证明如果 \(S_1\) 和 \(S_2\) 是 \(\mathbb{R}^{m+n}\) 中的凸集，那么它们的部分和

\[ S = \{(\mathbf{x}, \mathbf{y}_1+\mathbf{y}_2) \,|\, \mathbf{x} \in \mathbb{R}^m, \mathbf{y}_1, \mathbf{y}_2 \in \mathbb{R}^n, (\mathbf{x},\mathbf{y}_1) \in S_1, (\mathbf{x},\mathbf{y}_2)\in S_2\}. \]

\(\lhd\)

**3.4** \(\mathbf{z}_1, \ldots, \mathbf{z}_m \in \mathbb{R}^d\) 的凸组合是形如的线性组合

\[ \mathbf{w} = \sum_{i=1}^m \alpha_i \mathbf{z}_i \]

其中对所有 \(i\)，\(\alpha_i \geq 0\) 且 \(\sum_{i=1}^m \alpha_i = 1\)。证明一个集合是凸的当且仅当它包含其所有元素的凸组合。[提示：使用 \(m\) 的归纳法。] \(\lhd\)

**3.5** 一个集合 \(C \subseteq \mathbb{R}^d\) 是一个锥，如果对于所有 \(\mathbf{x} \in C\) 和所有 \(\alpha \geq 0\)，\(\alpha \mathbf{x} \in C\)。\(\mathbf{z}_1, \ldots, \mathbf{z}_m \in \mathbb{R}^d\) 的对偶组合是形如的线性组合

\[ \mathbf{w} = \sum_{i=1}^m \alpha_i \mathbf{z}_i \]

其中对所有 \(i\)，\(\alpha_i \geq 0\)。证明一个集合 \(C\) 是一个凸锥当且仅当它包含其所有元素的对偶组合。 \(\lhd\)

**3.6** 证明任何 \(\mathbb{R}^d\) 的线性子空间是凸的。 \(\lhd\)

**3.7** 证明以下集合

\[ \mathbb{R}^d_+ = \{\mathbf{x} \in \mathbb{R}^d\,:\, \mathbf{x} \geq \mathbf{0}\} \]

是凸的。 \(\lhd\)

**3.8** 设 \(f : \mathbb{R}^d \to \mathbb{R}\) 是一个严格凸函数。进一步假设存在一个全局最小值 \(\mathbf{x}^*\)。证明它是唯一的。 \(\lhd\)

**3.9** 证明 \(\mathbf{S}_+^n\) 是一个凸锥。 \(\lhd\)

**3.10** 从 *保持凸性的运算引理* 证明 (a)-(e)。 \(\lhd\)

**3.11** 设 \(f \,:\, \mathbb{R}^d \to \mathbb{R}\) 是一个函数。函数的上凸包是集合

\[ \mathbf{epi} f = \{(\mathbf{x}, y)\,:\, \mathbf{x} \in \mathbb{R}^d, y \geq f(\mathbf{x})\}. \]

证明 \(f\) 是凸的当且仅当 \(\mathbf{epi} f\) 是一个凸集。 \(\lhd\)

**3.12** 设 \(f_1, \ldots, f_m : \mathbb{R}^d \to \mathbb{R}\) 是凸函数，且 \(\beta_1, \ldots, \beta_m \geq 0\)。证明

\[ f(\mathbf{x}) = \sum_{i=1}^m \beta_i f_i(\mathbf{x}) \]

是凸的。 \(\lhd\)

**3.13** 设 \(f_1, f_2 : \mathbb{R}^d \to \mathbb{R}\) 是凸函数。证明逐点最大函数

\[ f(\mathbf{x}) = \max\{f_1(\mathbf{x}), f_2(\mathbf{x})\} \]

是凸的。[提示：首先证明 \(\max\{\alpha + \beta, \eta + \phi\} \leq \max\{\alpha, \eta\} + \max\{\beta, \phi\}\)。]\(\lhd\)

**3.14** 证明以下复合定理：如果 \(f : \mathbb{R}^d \to \mathbb{R}\) 是凸的，且 \(g : \mathbb{R} \to \mathbb{R}\) 是凸的且非递减的，那么复合 \(h = g \circ f\) 是凸的。 \(\lhd\)

**3.15** 证明 \(f(x) = e^{\beta x}\)，其中 \(x, \beta \in \mathbb{R}\)，是凸的。 \(\lhd\)

**3.16** 设 \(f : \mathbb{R} \to \mathbb{R}\) 是二阶连续可微的。证明如果对于所有 \(x \in \mathbb{R}\)，有 \(|f''(x)| \leq L\)，则 \(f\) 是 \(L\)-平滑的。 \(\lhd\)

**3.17** 对于 \(a \in [-1,1]\) 和 \(b \in \{0,1\}\)，设 \(\hat{f}(x, a) = \sigma(x a)\) 其中

\[ \sigma(t) = \frac{1}{1 + e^{-t}} \]

是一个由 \(x \in \mathbb{R}\) 参数化的分类器。对于数据集 \(a_i \in [-1,1]\) 和 \(b_i \in \{0,1\}\)，\(i=1,\ldots,n\)，令交叉熵损失为

\[ \mathcal{L}(x, \{(a_i, b_i)\}_{i=1}^n) = \frac{1}{n} \sum_{i=1}^n \ell(x, a_i, b_i) \]

在哪里

\[ \ell(x, a, b) = - b \log(\hat{f}(x, a)) - (1-b) \log(1 - \hat{f}(x, a)). \]

a) 证明对于所有 \(t \in \mathbb{R}\)，\(\sigma'(t) = \sigma(t)(1- \sigma(t))\)。

b) 使用 a) 证明

\[ \frac{\partial}{\partial x} \mathcal{L}(x, \{(a_i, b_i)\}_{i=1}^n) = - \frac{1}{n} \sum_{i=1}^n (b_i - \hat{f}(x,a_i)) \,a_i. \]

c) 使用 b) 证明

\[ \frac{\partial²}{\partial x²} \mathcal{L}(x, \{(a_i, b_i)\}_{i=1}^n) = \frac{1}{n} \sum_{i=1}^n \hat{f}(x,a_i) \,(1 - \hat{f}(x,a_i)) \,a_i². \]

d) 使用 c) 证明，对于任何数据集 \(\{(a_i, b_i)\}_{i=1}^n\)，\(\mathcal{L}\) 作为 \(x\) 的函数是凸的且 \(1\)-平滑。 \(\lhd\)

**3.18** 证明强凸函数的**二次界**。[*提示:* 修改平滑函数的**二次界**的证明。] \(\lhd\)

**3.19** 证明对于所有 \(x > 0\)，\(\log x \leq x - 1\)。[*提示:* 计算 \(s(x) = x - 1 - \log x\) 的导数和 \(s(1)\) 的值。] \(\lhd\)

**3.20** 考虑线性函数

\[ f(\mathbf{x}) = \mathbf{q}^T \mathbf{x} + r \]

在哪里 \(\mathbf{x} = (x_1, \ldots, x_d), \mathbf{q} = (q_1, \ldots, q_d) \in \mathbb{R}^d\) 和 \(r \in \mathbb{R}\)。固定一个值 \(c \in \mathbb{R}\)。假设

\[ f(\mathbf{x}_0) = f(\mathbf{x}_1) = c. \]

证明

\[ \nabla f(\mathbf{x}_0)^T(\mathbf{x}_1 - \mathbf{x}_0) = 0. \]

\(\lhd\)

**3.21** 设 \(f : D_1 \to \mathbb{R}\)，其中 \(D_1 \subseteq \mathbb{R}^d\)，设 \(\mathbf{g} : D_2 \to \mathbb{R}^d\)，其中 \(D_2 \subseteq \mathbb{R}\)。假设 \(f\) 在 \(\mathbf{g}(t_0)\) 处连续可微，\(\mathbf{g}(t_0)\) 是 \(D_1\) 的内点，且 \(\mathbf{g}\) 在 \(t_0\) 处连续可微，\(t_0\) 是 \(D_2\) 的内点。进一步假设存在一个值 \(c \in \mathbb{R}\) 使得

\[ f \circ \mathbf{g}(t) = c, \qquad \forall t \in D_2. \]

证明 \(\mathbf{g}'(t_0)\) 与 \(\nabla f(\mathbf{g}(t_0))\) 正交。 \(\lhd\)

**3.22** 固定一个分割 \(C_1,\ldots,C_k\) 的 \([n]\)。在 \(k\)-means 目标下，其成本是

\[ \mathcal{G}(C_1,\ldots,C_k) = \min_{\boldsymbol{\mu}_1,\ldots,\boldsymbol{\mu}_k \in \mathbb{R}^d} G(C_1,\ldots,C_k; \boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_k) \]

在哪里

\[\begin{align*} G(C_1,\ldots,C_k; \boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_k) &= \sum_{i=1}^k \sum_{j \in C_i} \|\mathbf{x}_j - \boldsymbol{\mu}_i\|² = \sum_{j=1}^n \|\mathbf{x}_j - \boldsymbol{\mu}_{c(j)}\|², \end{align*}\]

其中 \(\boldsymbol{\mu}_i \in \mathbb{R}^d\)，是簇 \(C_i\) 的中心。

a) 假设 \(\boldsymbol{\mu}_i^*\) 是 \(f\) 的全局最小值。

\[ F_i(\boldsymbol{\mu}_i) = \sum_{j \in C_i} \|\mathbf{x}_j - \boldsymbol{\mu}_i\|², \]

对于每个 \(i\)。证明 \(\boldsymbol{\mu}_1^*, \ldots, \boldsymbol{\mu}_k^*\) 是 \(G(C_1,\ldots,C_k; \boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_k)\) 的全局最小值。

b) 计算 \(F_i(\boldsymbol{\mu}_i)\) 的梯度。

c) 找到 \(F_i(\boldsymbol{\mu}_i)\) 的驻点。

d) 计算 \(F_i(\boldsymbol{\mu}_i)\) 的 Hessian 矩阵。

e) 证明 \(F_i(\boldsymbol{\mu}_i)\) 是凸函数。

f) 计算 \(G(C_1,\ldots,C_k; \boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_k)\) 的全局最小值。证明你的答案。 \(\lhd\)

**3.23** 考虑二次函数

\[ f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^T P \mathbf{x} + \mathbf{q}^T \mathbf{x} + r \]

其中 \(P\) 是对称的。证明如果 \(P\) 是正定的，那么 \(f\) 是强凸的。 \(\lhd\)

**3.24** 闭半空间是形如的集合

\[ K = \{\mathbf{x} \in \mathbb{R}^d\,:\,\mathbf{a}^T \mathbf{x} \leq b\}, \]

对于某个 \(\mathbf{a} \in \mathbb{R}^d\) 和 \(b \in \mathbb{R}\)。

a) 证明 \(K\) 是凸函数。

b) 证明存在 \(\mathbf{x}_0 \in \mathbb{R}^d\) 使得

\[ K = \{\mathbf{x} \in \mathbb{R}^d\,:\,\mathbf{a}^T (\mathbf{x}-\mathbf{x}_0) \leq 0\}. \]

\(\lhd\)

**3.25** 超平面是形如的集合

\[ H = \{\mathbf{x} \in \mathbb{R}^d\,:\,\mathbf{a}^T \mathbf{x} = b\}, \]

对于某个 \(\mathbf{a} \in \mathbb{R}^d\) 和 \(b \in \mathbb{R}\)。

a) 证明 \(H\) 是凸函数。

b) 假设 \(b = 0\)。计算 \(H^\perp\) 并证明你的答案。 \(\lhd\)

**3.26** 回想一下，\(\ell¹\)-范数定义为

\[ \|\mathbf{x}\|_1 = \sum_{i=1}^d |x_i|. \]

a) 证明 \(\ell¹\)-范数满足三角不等式。

b) 证明闭的 \(\ell¹\) 球

\[ \{\mathbf{x} \in \mathbb{R}^d\,:\,\|\mathbf{x}\|_1 \leq r\}, \]

对于某个 \(r > 0\)，是凸函数。 \(\lhd\)

**3.27** 洛伦兹锥是形如的集合

\[ C = \left\{ (\mathbf{x},t) \in \mathbb{R}^d \times \mathbb{R}\,:\, \|\mathbf{x}\|_2 \leq t \right\}. \]

证明 \(C\) 是凸函数。 \(\lhd\)

**3.28** 多面体是形如的集合

\[ K = \left\{ \mathbf{x} \in \mathbb{R}^d\,:\, \mathbf{a}_i^T \mathbf{x} \leq b_i, i=1,\ldots,m, \text{ and } \mathbf{c}_j^T \mathbf{x} = d_j, j=1,\ldots,n \right\}, \]

对于某个 \(\mathbf{a}_i \in \mathbb{R}^d\)，\(b_i \in \mathbb{R}\)，\(i=1,\ldots,m\) 和 \(\mathbf{c}_j \in \mathbb{R}^d\)，\(d_j \in \mathbb{R}\)，\(j=1,\ldots,n\)。证明 \(K\) 是凸函数。 \(\lhd\)

**3.29** \(\mathbb{R}^d\) 中的概率单纯形是形如的集合

\[ \Delta = \left\{ \mathbf{x} \in \mathbb{R}^d\,:\, \mathbf{x} \geq \mathbf{0} \text{ and } \mathbf{x}^T \mathbf{1} = 1 \right\}. \]

证明如下

\(\Delta\) 是凸集。 \(\lhd\)

**3.30** 考虑函数

\[ f^*(\mathbf{y}) = \sup_{\mathbf{x} \in \mathbb{R}^d} \left\{ \mathbf{y}^T\mathbf{x} - \|\mathbf{x}\|_2 \right\}. \]

a) 证明如果 \(\|\mathbf{y}\|_2 \leq 1\)，则 \(f^*(\mathbf{y}) = 0\)。

b) 证明如果 \(\|\mathbf{y}\|_2 > 1\)，则 \(f^*(\mathbf{y}) = +\infty\)。 \(\lhd\)

**3.31** 设 \(f, g : D \to \mathbb{R}\) 对于某个 \(D \subseteq \mathbb{R}^d\)，并设 \(\mathbf{x}_0\) 是 \(D\) 的一个内部点。假设 \(f\) 和 \(g\) 在 \(\mathbf{x}_0\) 处是连续可微的。设 \(h(\mathbf{x}) = f(\mathbf{x}) g(\mathbf{x})\) 对于 \(D\) 中的所有 \(\mathbf{x}\)。计算 \(\nabla h(\mathbf{x}_0)\)。[*提示:* 你可以使用相应的单变量结果。] \(\lhd\)

**3.32** 设 \(f : D \to \mathbb{R}\) 对于某个 \(D \subseteq \mathbb{R}^d\)，并设 \(\mathbf{x}_0\) 是 \(D\) 的一个内部点，其中 \(f(\mathbf{x}_0) \neq 0\)。假设 \(f\) 在 \(\mathbf{x}_0\) 处是连续可微的。计算 \(\nabla (1/f)\)。[*提示:* 你可以使用相应的单变量结果而不需要证明。] \(\lhd\)

**3.33** 设 \(f, g : D \to \mathbb{R}\) 对于某个 \(D \subseteq \mathbb{R}^d\)，并设 \(\mathbf{x}_0\) 是 \(D\) 的一个内部点。假设 \(f\) 和 \(g\) 在 \(\mathbf{x}_0\) 处是二阶连续可微的。设 \(h(\mathbf{x}) = f(\mathbf{x}) g(\mathbf{x})\) 对于 \(D\) 中的所有 \(\mathbf{x}\)。计算 \(\mathbf{H}_h (\mathbf{x}_0)\)。 \(\lhd\)

**3.34** (改编自 [Rud]) 设 \(f : D \to \mathbb{R}\) 对于某个凸开集 \(D \subseteq \mathbb{R}^d\)。假设 \(f\) 在 \(D\) 的每一点上都是连续可微的，并且进一步 \(\frac{\partial f}{\partial x_1}(\mathbf{x}_0) = 0\) 对于所有 \(\mathbf{x}_0 \in D\)。证明 \(f\) 只依赖于 \(x_2,\ldots,x_d\)。 \(\lhd\)

**3.35** (改编自 [Rud]) 设 \(\mathbf{g} : D \to \mathbb{R}^d\) 对于某个开集 \(D \subseteq \mathbb{R}\)。假设 \(\mathbf{g}\) 在 \(D\) 的每一点上都是连续可微的，并且进一步

\[ \|\mathbf{g}(t)\|² = 1, \qquad \forall t \in D. \]

证明对于所有 \(t \in D\)，有 \(\mathbf{g}'(t)^T \mathbf{g}(t) = 0\)。[*提示:* 使用复合函数。] \(\lhd\)

**3.36** (改编自 [Khu]) 设 \(f : D \to \mathbb{R}\) 对于某个开集 \(D \subseteq \mathbb{R}^d\)。假设 \(f\) 在 \(D\) 的每一点上都是连续可微的，并且进一步

\[ f(t \mathbf{x}) = t^n f(\mathbf{x}), \]

对于任何 \(\mathbf{x} \in D\) 和任何标量 \(t\)，使得 \(t \mathbf{x} \in D\)。在这种情况下，\(f\) 被称为是 \(n\) 次齐次的。证明对于所有 \(\mathbf{x} \in D\)，我们有

\[ \mathbf{x}^T \nabla f(\mathbf{x}) = n f(\mathbf{x}). \]

\(\lhd\)

**3.37** (改编自 [Khu]) 在 \(\mathbf{0}\) 的邻域内对函数应用 *泰勒定理*。

\[ f(x_1, x_2) = \exp(x_2 \sin x_1). \]

[*提示:* 你可以使用单变量函数导数的标准公式，无需证明。] \(\lhd\)

**3.38** (改编自 [Khu]) 在 \(\mathbf{0}\) 的邻域内对函数应用 *泰勒定理*。

\[ f(x_1, x_2) = \cos(x_1 x_2). \]

[*提示:* 你可以使用单变量函数导数的标准公式，无需证明。] \(\lhd\)

**3.39** (改编自 [Khu]) 在 \(\mathbf{0}\) 的邻域内对函数应用 *泰勒定理*。

\[ f(x_1, x_2, x_3) = \sin(e^{x_1} + x_2² + x_3³). \]

[*提示:* 你可以使用单变量函数导数的标准公式，无需证明。] \(\lhd\)

**3.40** (改编自 [Khu]) 考虑函数

\[ f(x_1, x_2) = (x_2 - x_1²)(x_2 - 2 x_1²). \]

a) 计算 \(f\) 在 \((0,0)\) 处的梯度和国值。

b) 证明 \((0,0)\) 不是 \(f\) 的严格局部极小值。[*提示:* 考虑形式为 \(x_1 = t^\alpha\) 和 \(x_2 = t^\beta\) 的参数曲线。]

c) 设 \(\mathbf{g}(t) = \mathbf{a} t\)，其中 \(\mathbf{a} = (a_1,a_2) \in \mathbb{R}²\) 是一个非零向量。证明 \(t=0\) 是 \(h(t) = f(\mathbf{g}(t))\) 的严格局部极小值。 \(\lhd\)

**3.41** 假设 \(f : \mathbb{R}^d \to \mathbb{R}\) 是凸函数。设 \(A \in \mathbb{R}^{d \times m}\) 和 \(\mathbf{b} \in \mathbb{R}^d\)。证明

\[ g(\mathbf{x}) = f(A \mathbf{x} + \mathbf{b}) \]

在 \(\mathbb{R}^m\) 上是凸的。 \(\lhd\)

**3.42** (改编自 [CVX]) 对于 \(\mathbf{x} = (x_1,\ldots,x_n)\in\mathbb{R}^n\)，设

\[ x_{[1]} \geq x_{[2]} \geq \cdots \geq x_{[n]}, \]

是 \(\mathbf{x}\) 在非递增顺序下的坐标。设 \(1 \leq r \leq n\) 是一个整数，且 \(w_1 \geq w_2 \geq \cdots \geq w_r \geq 0\)。证明

\[ f(\mathbf{x}) = \sum_{i=1}^r w_i x_{[i]}, \]

是凸的。 \(\lhd\)

**3.43** 对于固定的 \(\mathbf{z} \in \mathbb{R}^d\)，设

\[ f(\mathbf{x}) = \|\mathbf{x} - \mathbf{z}\|_2². \]

a) 计算 \(f\) 的梯度和国值。

b) 证明 \(f\) 是强凸的。

c) 对于有限集 \(Z \subseteq \mathbb{R}^d\)，设

\[ g(\mathbf{x}) = \max_{\mathbf{z} \in Z} \|\mathbf{x} - \mathbf{z}\|_2². \]

使用 *问题 3.13* 来证明 \(g\) 是凸的。 \(\lhd\)

**3.44** 设 \(S_i\), \(i \in I\)，是 \(\mathbb{R}^d\) 的凸子集集合。这里 \(I\) 可以是无限的。证明

\[ \bigcap_{i \in I} S_i, \]

是凸的。 \(\lhd\)

**3.45** a) 设 \(f_i\), \(i \in I\)，是定义在 \(\mathbb{R}^d\) 上的凸实值函数集合。使用 *问题 3.11, 3.44* 来证明

\[ g(\mathbf{x}) = \sup_{i \in I} f_i(\mathbf{x}), \]

是凸的。

b) 设 \(f : \mathbb{R}^{d+f} \to \mathbb{R}\) 是一个凸函数，且 \(S \subseteq \mathbb{R}^{f}\) 是一个（不一定凸的）集合。使用 (a) 来证明该函数

\[ g(\mathbf{x}) = \sup_{\mathbf{y} \in S} f(\mathbf{x},\mathbf{y}), \]

是凸的。你可以假设对于所有 \(\mathbf{x} \in \mathbb{R}^d\)，\(g(\mathbf{x}) < +\infty\)。 \(\lhd\)

**3.46** 使用 *问题 3.45* 来证明矩阵的最大特征值是所有对称矩阵集上的凸函数。[提示：参见 *Rayleigh Quotient* 例子。首先证明 \(\langle \mathbf{x}, A\mathbf{x}\rangle\) 是 \(A\) 的线性函数。] \(\lhd\)

**3.47** 设 \(f : \mathbb{R}^d \to \mathbb{R}\) 是一个凸函数。证明全局最优解集是凸集。 \(\lhd\)

**3.48** 考虑对称 \(n \times n\) 矩阵的集合

\[ \mathbf{S}^n = \left\{ X \in \mathbb{R}^{n \times n}\,:\, X = X^T \right\}. \]

a) 证明 \(\mathbf{S}^n\) 是 \(n \times n\) 矩阵向量空间中的线性子空间，即对于所有 \(X_1, X_2 \in \mathbf{S}^n\) 和 \(\alpha \in \mathbb{R}\)

\[ X_1 + X_2 \in \mathbf{S}^n, \]

和

\[ \alpha X_1 \in \mathbf{S}^n. \]

b) 证明存在 \(\mathbf{S}^n\) 的一个大小为 \(d = {n \choose 2} + n\) 的生成集，即一组对称矩阵 \(X_1,\ldots,X_d \in \mathbf{S}^n\)，使得任何矩阵 \(Y \in \mathbf{S}^n\) 可以写成线性组合

\[ Y = \sum_{i=1}^d \alpha_i X_i, \]

对于某些 \(\alpha_1,\ldots,\alpha_d \in \mathbb{R}\)。

c) 证明你在 b) 中构造的矩阵 \(X_1,\ldots,X_d \in \mathbf{S}^n\) 在线性无关的意义上是线性无关的。

\[ \sum_{i=1}^d \alpha_i X_i = \mathbf{0}_{n \times n} \quad \implies \quad \alpha_1 = \cdots = \alpha_d = 0. \]

\(\lhd\)

**3.49** 设 \(f : D \to \mathbb{R}\) 是定义在集合上的凸函数。

\[ D = \{\mathbf{x} = (x_1,\ldots,x_d) \in \mathbb{R}^d: x_i \geq 0, \forall i\}. \]

a) 证明 \(D\) 是凸集。

b) 证明凸集上的凸函数的 *一阶最优性条件* 简化为

\[ \frac{\partial f(\mathbf{x}⁰)}{\partial x_i} \geq 0, \qquad \forall i \in [d] \]

和

\[ \frac{\partial f(\mathbf{x}⁰)}{\partial x_i} = 0, \qquad \text{if $x_i⁰ > 0$}. \]

[*提示：在定理的条件中选择正确的 \(\mathbf{y}\)。] \(\lhd\)

**3.50** 设 \(f : \mathbb{R}^d \to \mathbb{R}\) 是二阶连续可微且 \(m\)-强凸的，其中 \(m>0\)。

a) 证明

\[ f(\mathbf{y}_n) \to +\infty, \]

沿着任何序列 \((\mathbf{y}_n)\) 满足 \(\|\mathbf{y}_n\| \to +\infty\) 当 \(n \to +\infty\)。

b) 证明，对于任何 \(\mathbf{x}⁰\)，集合

\[ C = \{\mathbf{x} \in \mathbb{R}^d\,:\,f(\mathbf{x}) \leq f(\mathbf{x}⁰)\}, \]

是闭且有界的。

c) 证明存在唯一的全局最优解 \(\mathbf{x}^*\) 对于 \(f\)。 \(\lhd\)

**3.51** 假设 \(f : \mathbb{R}^d \to \mathbb{R}\) 是 \(L\)-平滑和 \(m\)-强凸的，并且在 \(\mathbf{x}^*\) 处有全局最优解。从任意点 \(\mathbf{x}⁰\) 开始应用梯度下降法，步长为 \(\alpha = 1/L\)。

a) 证明 \(\mathbf{x}^t \to \mathbf{x}^*\) 当 \(t \to +\infty\)。[*提示：使用 \(\mathbf{x}^*\) 处的 *强凸函数的二次界*。]

b) 给出 \(\|\mathbf{x}^t - \mathbf{x}^*\|\) 的一个上界，该上界依赖于 \(t, m, L, f(\mathbf{x}⁰)\) 和 \(f(\mathbf{x}^*)\)。

\(\lhd\)

**3.52** 设 \(f : \mathbb{R}^d \to \mathbb{R}\) 是二阶连续可微的。

a) 证明

\[ \nabla f(\mathbf{y}) = \nabla f(\mathbf{x}) + \int_{0}¹ \mathbf{H}_f(\mathbf{x} + \xi \mathbf{p}) \,\mathbf{p} \,\mathrm{d}\xi, \]

其中 \(\mathbf{p} = \mathbf{y} - \mathbf{x}\)。在这里，向量值函数的积分是逐项进行的。[提示：设 \(g_i(\xi) = \frac{\partial}{\partial x_i} f(\mathbf{x} + \xi \mathbf{p})\)。微积分基本定理表明 \(g_i(1) - g_i(0) = \int_{0}¹ g_i'(\xi) \,\mathrm{d}\xi.\)]

b) 假设 \(f\) 是 \(L\)-平滑的。使用 (a) 证明 \(f\) 的梯度在意义上是 \(L\)-Lipschitz。

\[ \|\nabla f(\mathbf{y}) - \nabla f(\mathbf{x})\|_2 \leq L \|\mathbf{y} - \mathbf{x}\|_2, \qquad \forall \mathbf{x}, \mathbf{y} \in \mathbb{R}^d. \]

\(\lhd\)

**3.53** 考虑二次函数

\[ f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^T P \mathbf{x} + \mathbf{q}^T \mathbf{x} + r, \]

其中 \(P\) 是一个对称矩阵。使用**一阶凸性条件**证明 \(f\) 是凸的当且仅当 \(P\) 是正半定矩阵。 \(\lhd\)

**3.54** 设

\[ \ell(\mathbf{x}; A, \mathbf{b}) = \frac{1}{n} \sum_{i=1}^n \left\{- b_i \log(\sigma(\boldsymbol{\alpha}_i^T \mathbf{x})) - (1-b_i) \log(1- \sigma(\boldsymbol{\alpha}_i^T \mathbf{x}))\right\}. \]

是逻辑回归中的损失函数，其中 \(A \in \mathbb{R}^{n \times d}\) 的行是 \(\boldsymbol{\alpha}_i^T\)。证明 \(\frac{\partial f}{\partial x_1}\) 和 \(\frac{\partial f}{\partial x_2}\) 在点 \((\frac{\pi}{4}, \frac{\pi}{3})\)。

\[ \mathbf{H}_{\ell}(\mathbf{x}; A, \mathbf{b}) \preceq \frac{1}{4n} A^T A. \]

[提示：阅读**逻辑回归平滑性**引理的证明。] \(\lhd\)

## 3.7.1\. 热身练习表#

*(有 Claude、Gemini 和 ChatGPT 的帮助)*

**第 3.2 节**

**E3.2.1** 计算函数 \(f(x_1, x_2) = 3x_1² - 2x_1x_2 + 4x_2² - 5x_1 + 2x_2\) 在点 \((1, -1)\) 的梯度。

**E3.2.2** 找到函数 \(f(x_1, x_2, x_3) = 2x_1x_2 + 3x_2x_3 - x_1² - 4x_3²\) 的 Hessian 矩阵。

**E3.2.3** 计算函数 \(f(x_1, x_2) = \sin(x_1) \cos(x_2)\) 在点 \((\frac{\pi}{4}, \frac{\pi}{3})\) 的偏导数。

**E3.2.4** 设 \(f(x_1, x_2) = e^{x_1} + x_1x_2 - x_2²\) 和 \(\mathbf{g}(t) = (t², t)\)。使用**链式法则**计算 \((f \circ \mathbf{g})'(1)\)。

**E3.2.5** 验证函数 \(f(x_1, x_2) = x_1³ + 3x_1x_2² - 2x_2³\) 在点 \((1, 2)\) 的**Hessian 矩阵**的**对称性**。

**E3.2.6** 找到函数 \(f(x_1, x_2, x_3) = 2x_1 - 3x_2 + 4x_3\) 在点 \((1, -2, 3)\) 的梯度。

**E3.2.7** 计算函数 \(f(x_1, x_2) = x_1² \sin(x_2)\) 的二阶偏导数。

**E3.2.8** 计算二次函数 \(f(x_1, x_2) = 3x_1² + 2x_1x_2 - x_2² + 4x_1 - 2x_2\) 在点 \((1, -1)\) 处的梯度。

**E3.2.9** 求函数 \(f(x_1, x_2, x_3) = x_1² + 2x_2² + 3x_3² - 2x_1x_2 + 4x_1x_3 - 6x_2x_3\) 的海森矩阵。

**E3.2.10** 对函数 \(f(x_1, x_2) = x_1² + x_2²\) 在连接点 \((0, 0)\) 和 \((1, 2)\) 的线段上应用多元均值定理。

**E3.2.11** 求函数 \(f(x, y) = x³y² - 2xy³ + y⁴\) 的偏导数 \(\frac{\partial f}{\partial x}\) 和 \(\frac{\partial f}{\partial y}\)。

**E3.2.12** 计算函数 \(f(x, y) = \ln(x² + 2y²)\) 在点 \((1, 2)\) 处的梯度 \(\nabla f(x, y)\)。

**E3.2.13** 求函数 \(g(x, y) = \sin(x) \cos(y)\) 的海森矩阵。

**E3.2.14** 如果 \(p(x, y, z) = x²yz + 3xyz²\)，求 \(p\) 的所有二阶偏导数。

**E3.2.15** 验证函数 \(q(x, y) = x³ - 3xy²\) 满足拉普拉斯方程：\(\frac{\partial² q}{\partial x²} + \frac{\partial² q}{\partial y²} = 0\)。

**E3.2.16** 考虑函数 \(s(x, y) = x² + 4xy + 5y²\)。使用均值定理（\(\xi = 0\)）通过点 \((1, 1)\) 近似 \(s(1.1, 0.9)\)。

**E3.2.17** 一个粒子沿着路径 \(\mathbf{c}(t) = (t², t³)\) 运动。点 \((x, y)\) 的温度由 \(u(x, y) = e^{-x² - y²}\) 给出。求粒子在时间 \(t = 1\) 时温度变化率。

**E3.2.18** 对 \(f(x, y) = x² + y²\) 和 \(\mathbf{g}(t) = (t, \sin t)\) 应用链式法则求 \(\frac{d}{dt} f(\mathbf{g}(t))\)。

**E3.2.19** 对 \(f(x, y) = xy\) 和 \(\mathbf{g}(t) = (t², \cos t)\) 应用链式法则求 \(\frac{d}{dt} f(\mathbf{g}(t))\)。

**第 3.3 节**

**E3.3.1** 考虑函数 \(f(x_1, x_2) = x_1² + 2x_2²\)。找出所有满足 \(\nabla f(x_1, x_2) = 0\) 的点 \((x_1, x_2)\)。

**E3.3.2** 设 \(f(x_1, x_2) = x_1² - x_2²\)。求函数 \(f\) 在点 \((1, 1)\) 处沿方向 \(\mathbf{v} = (\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}})\) 的方向导数。

**E3.3.3** 考虑函数 \(f(x_1, x_2) = x_1² + 2x_1x_2 + x_2²\)。求函数 \(f\) 在点 \((1, 1)\) 处沿方向 \(\mathbf{v} = (\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}})\) 的二阶方向导数。

**E3.3.4** 考虑优化问题 \(\min_{x_1, x_2} x_1² + x_2²\) 在约束 \(x_1 + x_2 = 1\) 下。写出拉格朗日函数 \(L(x_1, x_2, \lambda)\)。

**E3.3.5** 对于 E3.3.4 中的优化问题，找出所有满足一阶必要条件的点 \((x_1, x_2, \lambda)\)。

**E3.3.6** 对于 E3.3.4 中的优化问题，检查点 \((\frac{1}{2}, \frac{1}{2}, -1)\) 处的二阶充分条件。

**E3.3.7** 考虑函数 \(f(x_1, x_2, x_3) = x_1² + x_2² + x_3²\)。找出满足优化问题 \(\min_{x_1, x_2, x_3} f(x_1, x_2, x_3)\) 且受约束 \(x_1 + 2x_2 + 3x_3 = 6\) 的所有点 \((x_1, x_2, x_3)\)。

**E3.3.8** 对于 E3.3.7 中的优化问题，在点 \((\frac{1}{2}, 1, \frac{3}{2}, -1)\) 处检查二阶充分条件。

**E3.3.9** 设 \(f: \mathbb{R}² \to \mathbb{R}\) 由 \(f(x_1, x_2) = x_1³ - 3x_1x_2²\) 定义。确定向量 \(\mathbf{v} = (1, 1)\) 是否是 \(f\) 在点 \((1, 0)\) 处的下降方向。

**E3.3.10** 设 \(f: \mathbb{R}² \to \mathbb{R}\) 由 \(f(x_1, x_2) = x_1² + x_2² - 2x_1 - 4x_2 + 5\) 定义。找出 \(f\) 的 Hessian 矩阵。

**E3.3.11** 设 \(f: \mathbb{R}² \to \mathbb{R}\) 由 \(f(x_1, x_2) = -x_1² - x_2²\) 定义。计算 \(f\) 在点 \((0, 0)\) 处沿方向 \(\mathbf{v} = (1, 0)\) 的二阶方向导数。

**E3.3.12** 计算函数 \(f(x,y) = x² + y²\) 在点 \((1,1)\) 处沿方向 \(\mathbf{v} = (1,1)\) 的方向导数。

**E3.3.13** 确定 \(f(x,y) = x³ + y³ - 3xy\) 在点 \((1,1)\) 处的 Hessian 矩阵。

**第 3.4 节**

**E3.4.1** 找出点 \((2, 3)\) 和 \((4, 5)\) 在 \(\alpha = 0.3\) 下的凸组合。

**E3.4.2** 判断以下集合是否是凸集：

\[ S = \{(x_1, x_2) \in \mathbb{R}² : x_1² + x_2² \leq 1\}. \]

**E3.4.3** 设 \(S_1 = \{x \in \mathbb{R}² : x_1 + x_2 \leq 1\}\) 和 \(S_2 = \{x \in \mathbb{R}² : x_1 - x_2 \leq 1\}\)。直接证明 \(S_1 \cap S_2\) 是一个凸集。

**E3.4.4** 使用 Hessian 矩阵验证函数 \(f(\mathbf{x}) = \log(e^{x_1} + e^{x_2})\) 是凸函数。

**E3.4.5** 找出函数 \(f(x) = x² + 2x + 1\) 的全局最小值。

**E3.4.6** 考虑函数 \(f(x) = |x|\)。证明 \(f\) 是凸函数但在 \(x = 0\) 处不可微。

**E3.4.7** 验证函数 \(f(x, y) = x² + 2y²\) 在 \(m = 2\) 的情况下是强凸的。

**E3.4.8** 找出点 \(x = (1, 2)\) 在集合 \(D = \{(x_1, x_2) \in \mathbb{R}² : x_1 + x_2 = 1\}\) 上的投影。

**E3.4.9** 设 \(f(x) = x⁴ - 2x² + 1\)。判断 \(f\) 是否是凸函数。

**E3.4.10** 设 \(f(x) = e^x\)。判断 \(f\) 是否是对数凹的，即 \(\log f\) 是凹函数。

**E3.4.11** 设 \(f(x) = x² - 2x + 2\)。判断 \(f\) 是否是强凸函数。

**E3.4.12** 设 \(A = \begin{pmatrix} 1 & 2 \\ 2 & 5 \end{pmatrix}\) 和 \(\mathbf{b} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}\)。找出使 \(\|A\mathbf{x} - \mathbf{b}\|_2²\) 最小的向量 \(\mathbf{x}\)。

**E3.4.13** 给定集合 \(D = \{(x, y) \in \mathbb{R}² : x² + y² < 4\}\)，判断 \(D\) 是否是凸集。

**E3.4.14** 判断集合 \(D = \{(x, y) \in \mathbb{R}² : x + y \leq 1\}\) 是否是凸集。

**E3.4.15** 判断集合 \(D = \{(x, y) \in \mathbb{R}² : x² - y² \leq 1\}\) 是否是凸集。

**第 3.5 节**

**E3.5.1** 给定函数 \(f(x, y) = x² + 4y²\)，找到点 \((1, 1)\) 处的最速下降方向。

**E3.5.2** 考虑函数 \(f(x) = x² + 4x + 5\)。计算导数 \(f'(x)\) 并找到在 \(x_0 = 1\) 处的最速下降方向。

**E3.5.3** 给定函数 \(f(x) = x³ - 6x² + 9x + 2\)，从 \(x_0 = 0\) 开始，进行两步梯度下降，步长 \(\alpha = 0.1\)。

**E3.5.4** 设 \(f(x) = x²\)。证明 \(f\) 对于某个 \(L > 0\) 是 \(L\)-平滑的。

**E3.5.5** 设 \(f(x) = x²\)。从 \(x_0 = 2\) 开始，进行一步梯度下降，步长 \(\alpha = 0.1\)。

**E3.5.6** 考虑 \(2\)-平滑函数 \(f(x) = x²\)。从 \(x_0 = 3\) 开始，计算步长 \(\alpha = \frac{1}{2}\) 的一次梯度下降迭代后得到的点 \(x_1\)。

**E3.5.7** 验证函数 \(f(x) = 2x² + 1\) 是 \(4\)-强凸的。

**E3.5.8** 对于全局最小值 \(x^* = 0\) 的 \(4\)-强凸函数 \(f(x) = 2x² + 1\)，计算 \(\frac{1}{2m} |f'(1)|²\) 并将其与 \(f(1) - f(x^*)\) 进行比较。

**E3.5.9** 设 \(f(x) = x⁴\)。\(f\) 对于某个 \(L > 0\) 是 \(L\)-平滑的吗？证明你的答案。

**E3.5.10** 设 \(f(x) = x³\)。\(f\) 对于某个 \(m > 0\) 是 \(m\)-强凸的吗？证明你的答案。

**E3.5.11** 设 \(f(x) = x² + 2x + 1\)。证明 \(f\) 对于某个 \(m > 0\) 是 \(m\)-强凸的。

**E3.5.12** 设 \(f(x) = x² + 2x + 1\)。从 \(x_0 = -3\) 开始，进行一步梯度下降，步长 \(\alpha = 0.2\)。

**E3.5.13** 给定函数 \(f(x, y) = x² + y²\)，从 \((1, 1)\) 开始，进行两步梯度下降，步长 \(\alpha = 0.1\)。

**E3.5.14** 对于一个二次函数 \(f(x) = ax² + bx + c\) ，其中 \(a > 0\)，推导出梯度下降在一步内收敛到最小值的 \(\alpha\) 值。

**第 3.6 节**

**E3.6.1** 计算概率为 \(p = 0.25\) 的事件的对数几率。

**E3.6.2** 给定特征 \(\mathbf{x} = (2, -1)\) 和系数 \(\boldsymbol{\alpha} = (0.5, -0.3)\)，计算线性组合 \(\boldsymbol{\alpha}^T \mathbf{x}\)。

**E3.6.3** 对于数据点 \(\mathbf{x} = (1, 3)\) ，标签 \(b = 1\) 和系数 \(\boldsymbol{\alpha} = (-0.2, 0.4)\)，使用 sigmoid 函数计算概率 \(p(\mathbf{x}; \boldsymbol{\alpha})\)。

**E3.6.4** 计算标签为 \(b = 0\) 的单个数据点 \(\mathbf{x} = (1, 2)\) 的交叉熵损失，给定 \(\boldsymbol{\alpha} = (0.3, -0.4)\)。

**E3.6.5** 证明逻辑函数 \(\sigma(z) = \frac{1}{1 + e^{-z}}\) 满足 \(\sigma'(z) = \sigma(z)(1 - \sigma(z))\)。

**E3.6.6** 给定特征向量 \(\boldsymbol{\alpha}_1 = (1, 2)\)，\(\boldsymbol{\alpha}_2 = (-1, 1)\)，\(\boldsymbol{\alpha}_3 = (0, -1)\)，以及相应的标签 \(b_1 = 1\)，\(b_2 = 0\)，\(b_3 = 1\)，计算 \(x = (1, 1)\) 时的逻辑回归目标函数 \(\ell(\mathbf{x}; A, \mathbf{b})\)。

**E3.6.7** 对于与 E3.6.6 相同的数据，计算逻辑回归目标函数 \(\nabla \ell(\mathbf{x}; A, b)\) 在 \(\mathbf{x} = (1, 1)\) 处的梯度。

**E3.6.8** 对于与 E3.6.6 相同的数据，计算逻辑回归目标函数 \(\mathbf{H}_\ell(\mathbf{x}; A, \mathbf{b})\) 在 \(\mathbf{x} = (1, 1)\) 处的 Hessian。

**E3.6.9** 对于与 E3.6.6 相同的数据，从 \(\mathbf{x}⁰ = (0, 0)\) 开始，执行一步梯度下降，步长 \(\beta = 0.1\)。

**E3.6.10** 对于与 E3.6.6 相同的数据，计算逻辑回归目标函数的光滑性参数 \(L\)。

## 3.7.2\. 问题#

**3.1** 设 \(f : \mathbb{R} \to\mathbb{R}\) 是二阶连续可微的，设 \(\mathbf{a}_1, \mathbf{a}_2\) 是 \(\mathbb{R}^d\) 中的向量，设 \(b_1, b_2 \in \mathbb{R}\)。考虑以下 \(\mathbb{R}^d\) 中的实值函数 \(\mathbf{x}\)：

\[ g(\mathbf{x}) = \frac{1}{2} f(\mathbf{a}_1^T \mathbf{x} + b_1) + \frac{1}{2} f(\mathbf{a}_2^T \mathbf{x} + b_2). \]

a) 用 \(f\) 的导数 \(f'\) 表示 \(g\) 的梯度，并以向量形式给出。（通过向量形式，我们指的是不能单独写出 \(\nabla g(\mathbf{x})\) 的每个元素。）

b) 用 \(f\) 的一阶导数 \(f'\) 和二阶导数 \(f''\) 表示 \(g\) 的 Hessian，并以矩阵形式给出。（通过矩阵形式，我们指的是不能单独写出 \(\mathbf{H}_g (\mathbf{x})\) 的每个元素或其列。）

**3.2** 使用方向导数的定义，给出**下降方向和方向导数引理**的另一种证明。[提示：模仿单变量情况的证明。] \(\lhd\)

**3.3** （改编自 [CVX]）证明如果 \(S_1\) 和 \(S_2\) 是 \(\mathbb{R}^{m+n}\) 中的凸集，那么它们的部分和也是凸集。

\[ S = \{(\mathbf{x}, \mathbf{y}_1+\mathbf{y}_2) \,|\, \mathbf{x} \in \mathbb{R}^m, \mathbf{y}_1, \mathbf{y}_2 \in \mathbb{R}^n, (\mathbf{x},\mathbf{y}_1) \in S_1, (\mathbf{x},\mathbf{y}_2)\in S_2\}. \]

\(\lhd\)

**3.4** \(\mathbf{z}_1, \ldots, \mathbf{z}_m \in \mathbb{R}^d\) 的凸组合是以下形式的线性组合

\[ \mathbf{w} = \sum_{i=1}^m \alpha_i \mathbf{z}_i \]

其中对所有 \(i\)，\(\alpha_i \geq 0\)，且 \(\sum_{i=1}^m \alpha_i = 1\)。证明一个集合是凸集当且仅当它包含其所有元素的凸组合。[提示：使用对 \(m\) 的归纳法。] \(\lhd\)

**3.5** 如果 \(C \subseteq \mathbb{R}^d\) 是一个锥，对于所有 \(\mathbf{x} \in C\) 和所有 \(\alpha \geq 0\)，\(\alpha \mathbf{x} \in C\)，则 \(C\) 是一个锥。\(\mathbf{z}_1, \ldots, \mathbf{z}_m \in \mathbb{R}^d\) 的锥组合是以下形式的线性组合

\[ \mathbf{w} = \sum_{i=1}^m \alpha_i \mathbf{z}_i \]

其中对所有 \(i\)，\(\alpha_i \geq 0\)。证明一个集合 \(C\) 是一个凸锥当且仅当它包含其所有元素的锥组合。 \(\lhd\)

**3.6** 证明 \(\mathbb{R}^d\) 的任何线性子空间都是凸的。 \(\lhd\)

**3.7** 证明集合

\[ \mathbb{R}^d_+ = \{\mathbf{x} \in \mathbb{R}^d\,:\, \mathbf{x} \geq \mathbf{0}\} \]

是凸的。 \(\lhd\)

**3.8** 设 \(f : \mathbb{R}^d \to \mathbb{R}\) 是一个严格凸函数。进一步假设存在一个全局最小值 \(\mathbf{x}^*\)。证明它是唯一的。 \(\lhd\)

**3.9** 证明 \(\mathbf{S}_+^n\) 是一个凸锥。 \(\lhd\)

**3.10** 从 *保持凸性运算引理* 证明 (a)-(e)。 \(\lhd\)

**3.11** 设 \(f \,:\, \mathbb{R}^d \to \mathbb{R}\) 是一个函数。\(f\) 的图像是集合

\[ \mathbf{epi} f = \{(\mathbf{x}, y)\,:\, \mathbf{x} \in \mathbb{R}^d, y \geq f(\mathbf{x})\}. \]

证明 \(f\) 是凸的当且仅当 \(\mathbf{epi} f\) 是一个凸集。 \(\lhd\)

**3.12** 设 \(f_1, \ldots, f_m : \mathbb{R}^d \to \mathbb{R}\) 是凸函数，并设 \(\beta_1, \ldots, \beta_m \geq 0\)。证明

\[ f(\mathbf{x}) = \sum_{i=1}^m \beta_i f_i(\mathbf{x}) \]

是凸的。 \(\lhd\)

**3.13** 设 \(f_1, f_2 : \mathbb{R}^d \to \mathbb{R}\) 是凸函数。证明点值最大函数

\[ f(\mathbf{x}) = \max\{f_1(\mathbf{x}), f_2(\mathbf{x})\} \]

是凸的。[提示：首先证明 \(\max\{\alpha + \beta, \eta + \phi\} \leq \max\{\alpha, \eta\} + \max\{\beta, \phi\}\]。]\(\lhd\)

**3.14** 证明以下复合定理：如果 \(f : \mathbb{R}^d \to \mathbb{R}\) 是凸的，且 \(g : \mathbb{R} \to \mathbb{R}\) 是凸的且非递减的，那么复合 \(h = g \circ f\) 是凸的。 \(\lhd\)

**3.15** 证明 \(f(x) = e^{\beta x}\)，其中 \(x, \beta \in \mathbb{R}\)，是凸的。 \(\lhd\)

**3.16** 设 \(f : \mathbb{R} \to \mathbb{R}\) 是二阶连续可微的。证明如果对所有 \(x \in \mathbb{R}\)，有 \(|f''(x)| \leq L\)，则 \(f\) 是 \(L\)-平滑的。 \(\lhd\)

**3.17** 对于 \(a \in [-1,1]\) 和 \(b \in \{0,1\}\)，令 \(\hat{f}(x, a) = \sigma(x a)\) 其中

\[ \sigma(t) = \frac{1}{1 + e^{-t}} \]

是一个由 \(x \in \mathbb{R}\) 参数化的分类器。对于数据集 \(a_i \in [-1,1]\) 和 \(b_i \in \{0,1\}\)，\(i=1,\ldots,n\)，令交叉熵损失为

\[ \mathcal{L}(x, \{(a_i, b_i)\}_{i=1}^n) = \frac{1}{n} \sum_{i=1}^n \ell(x, a_i, b_i) \]

其中

\[ \ell(x, a, b) = - b \log(\hat{f}(x, a)) - (1-b) \log(1 - \hat{f}(x, a)). \]

a) 证明 \(\sigma'(t) = \sigma(t)(1- \sigma(t))\) 对于所有 \(t \in \mathbb{R}\) 成立。

b) 使用 a) 证明

\[ \frac{\partial}{\partial x} \mathcal{L}(x, \{(a_i, b_i)\}_{i=1}^n) = - \frac{1}{n} \sum_{i=1}^n (b_i - \hat{f}(x,a_i)) \,a_i. \]

c) 使用 b) 证明

\[ \frac{\partial²}{\partial x²} \mathcal{L}(x, \{(a_i, b_i)\}_{i=1}^n) = \frac{1}{n} \sum_{i=1}^n \hat{f}(x,a_i) \,(1 - \hat{f}(x,a_i)) \,a_i². \]

d) 使用 c) 证明，对于任何数据集 \(\{(a_i, b_i)\}_{i=1}^n\)，\(\mathcal{L}\) 作为 \(x\) 的函数是凸的且 \(1\)-平滑。 \(\lhd\)

**3.18** 证明强凸函数的**二次界**。[*提示:* 修改平滑函数的**二次界**的证明。] \(\lhd\)

**3.19** 证明对所有 \(x > 0\)，有 \(\log x \leq x - 1\)。[*提示:* 计算 \(s(x) = x - 1 - \log x\) 的导数和 \(s(1)\) 的值。] \(\lhd\)

**3.20** 考虑一个仿射函数

\[ f(\mathbf{x}) = \mathbf{q}^T \mathbf{x} + r \]

其中 \(\mathbf{x} = (x_1, \ldots, x_d), \mathbf{q} = (q_1, \ldots, q_d) \in \mathbb{R}^d\) 和 \(r \in \mathbb{R}\)。固定一个值 \(c \in \mathbb{R}\)。假设

\[ f(\mathbf{x}_0) = f(\mathbf{x}_1) = c. \]

证明

\[ \nabla f(\mathbf{x}_0)^T(\mathbf{x}_1 - \mathbf{x}_0) = 0. \]

\(\lhd\)

**3.21** 设 \(f : D_1 \to \mathbb{R}\)，其中 \(D_1 \subseteq \mathbb{R}^d\)，设 \(\mathbf{g} : D_2 \to \mathbb{R}^d\)，其中 \(D_2 \subseteq \mathbb{R}\)。假设 \(f\) 在 \(\mathbf{g}(t_0)\) 处连续可微，\(\mathbf{g}(t_0)\) 是 \(D_1\) 的内点，且 \(\mathbf{g}\) 在 \(t_0\) 处连续可微，\(t_0\) 是 \(D_2\) 的内点。进一步假设存在一个值 \(c \in \mathbb{R}\)，使得

\[ f \circ \mathbf{g}(t) = c, \qquad \forall t \in D_2. \]

证明 \(\mathbf{g}'(t_0)\) 与 \(\nabla f(\mathbf{g}(t_0))\) 正交。 \(\lhd\)

**3.22** 固定一个 \([n]\) 的划分 \(C_1,\ldots,C_k\)。在 \(k\)-均值目标下，其成本是

\[ \mathcal{G}(C_1,\ldots,C_k) = \min_{\boldsymbol{\mu}_1,\ldots,\boldsymbol{\mu}_k \in \mathbb{R}^d} G(C_1,\ldots,C_k; \boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_k) \]

其中

\[\begin{align*} G(C_1,\ldots,C_k; \boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_k) &= \sum_{i=1}^k \sum_{j \in C_i} \|\mathbf{x}_j - \boldsymbol{\mu}_i\|² = \sum_{j=1}^n \|\mathbf{x}_j - \boldsymbol{\mu}_{c(j)}\|², \end{align*}\]

\(\boldsymbol{\mu}_i \in \mathbb{R}^d\)，是聚类 \(C_i\) 的中心。

a) 假设 \(\boldsymbol{\mu}_i^*\) 是

\[ F_i(\boldsymbol{\mu}_i) = \sum_{j \in C_i} \|\mathbf{x}_j - \boldsymbol{\mu}_i\|², \]

对于每个 \(i\)。证明 \(\boldsymbol{\mu}_1^*, \ldots, \boldsymbol{\mu}_k^*\) 是 \(G(C_1,\ldots,C_k; \boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_k)\) 的全局最小值。

b) 计算函数 \(F_i(\boldsymbol{\mu}_i)\) 的梯度。

c) 找到函数 \(F_i(\boldsymbol{\mu}_i)\) 的驻点。

d) 计算 \(F_i(\boldsymbol{\mu}_i)\) 的 Hessian 矩阵。

e) 证明 \(F_i(\boldsymbol{\mu}_i)\) 是凸的。

f) 计算函数 \(G(C_1,\ldots,C_k; \boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_k)\) 的全局最小值。证明你的答案。 \(\lhd\)

**3.23** 考虑一个二次函数

\[ f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^T P \mathbf{x} + \mathbf{q}^T \mathbf{x} + r \]

其中 \(P\) 是对称的。证明如果 \(P\) 是正定的，那么 \(f\) 是强凸的。 \(\lhd\)

**3.24** 闭半空间是形如的集合

\[ K = \{\mathbf{x} \in \mathbb{R}^d\,:\,\mathbf{a}^T \mathbf{x} \leq b\}, \]

对于某个 \(\mathbf{a} \in \mathbb{R}^d\) 和 \(b \in \mathbb{R}\)。

a) 证明 \(K\) 是凸的。

b) 证明存在 \(\mathbf{x}_0 \in \mathbb{R}^d\) 使得

\[ K = \{\mathbf{x} \in \mathbb{R}^d\,:\,\mathbf{a}^T (\mathbf{x}-\mathbf{x}_0) \leq 0\}. \]

\(\lhd\)

**3.25** �超平面是形如

\[ H = \{\mathbf{x} \in \mathbb{R}^d\,:\,\mathbf{a}^T \mathbf{x} = b\}, \]

对于某个 \(\mathbf{a} \in \mathbb{R}^d\) 和 \(b \in \mathbb{R}\)。

a) 证明 \(H\) 是凸集。

b) 假设 \(b = 0\)。计算 \(H^\perp\) 并证明你的答案。 \(\lhd\)

**3.26** 回忆 \(\ell¹\)-范数定义为

\[ \|\mathbf{x}\|_1 = \sum_{i=1}^d |x_i|. \]

a) 证明 \(\ell¹\)-范数满足三角不等式。

b) 证明闭的 \(\ell¹\) 球

\[ \{\mathbf{x} \in \mathbb{R}^d\,:\,\|\mathbf{x}\|_1 \leq r\}, \]

对于某个 \(r > 0\)，是凸集。 \(\lhd\)

**3.27** 洛伦兹锥是集合

\[ C = \left\{ (\mathbf{x},t) \in \mathbb{R}^d \times \mathbb{R}\,:\, \|\mathbf{x}\|_2 \leq t \right\}. \]

证明 \(C\) 是凸集。 \(\lhd\)

**3.28** 多面体是形如

\[ K = \left\{ \mathbf{x} \in \mathbb{R}^d\,:\, \mathbf{a}_i^T \mathbf{x} \leq b_i, i=1,\ldots,m, \text{ and } \mathbf{c}_j^T \mathbf{x} = d_j, j=1,\ldots,n \right\}, \]

对于某些 \(\mathbf{a}_i \in \mathbb{R}^d\)，\(b_i \in \mathbb{R}\)，\(i=1,\ldots,m\) 和 \(\mathbf{c}_j \in \mathbb{R}^d\)，\(d_j \in \mathbb{R}\)，\(j=1,\ldots,n\)。证明 \(K\) 是凸集。 \(\lhd\)

**3.29** \(\mathbb{R}^d\) 中的概率单纯形是集合

\[ \Delta = \left\{ \mathbf{x} \in \mathbb{R}^d\,:\, \mathbf{x} \geq \mathbf{0} \text{ and } \mathbf{x}^T \mathbf{1} = 1 \right\}. \]

证明

\(\Delta\) 是凸集。 \(\lhd\)

**3.30** 考虑函数

\[ f^*(\mathbf{y}) = \sup_{\mathbf{x} \in \mathbb{R}^d} \left\{ \mathbf{y}^T\mathbf{x} - \|\mathbf{x}\|_2 \right\}. \]

a) 证明如果 \(\|\mathbf{y}\|_2 \leq 1\)，则 \(f^*(\mathbf{y}) = 0\)。

b) 证明如果 \(\|\mathbf{y}\|_2 > 1\)，则 \(f^*(\mathbf{y}) = +\infty\)。 \(\lhd\)

**3.31** 设 \(f, g : D \to \mathbb{R}\) 对于某个 \(D \subseteq \mathbb{R}^d\)，并且设 \(\mathbf{x}_0\) 是 \(D\) 的一个内点。假设 \(f\) 和 \(g\) 在 \(\mathbf{x}_0\) 处连续可微。对于 \(D\) 中的所有 \(\mathbf{x}\)，令 \(h(\mathbf{x}) = f(\mathbf{x}) g(\mathbf{x})\)。计算 \(\nabla h(\mathbf{x}_0)\)。[*提示：你可以利用相应的单变量结果。*\] \(\lhd\)

**3.32** 设 \(f : D \to \mathbb{R}\) 对于某个 \(D \subseteq \mathbb{R}^d\)，并且设 \(\mathbf{x}_0\) 是 \(D\) 的一个内点，其中 \(f(\mathbf{x}_0) \neq 0\)。假设 \(f\) 在 \(\mathbf{x}_0\) 处连续可微。计算 \(\nabla (1/f)\)。[*提示：你可以利用相应的单变量结果而不需要证明。*\] \(\lhd\)

**3.33** 对于某个 \(D \subseteq \mathbb{R}^d\) 和某个内点 \(\mathbf{x}_0\)，设 \(f, g : D \to \mathbb{R}\)。假设 \(f\) 和 \(g\) 在 \(\mathbf{x}_0\) 处是二阶连续可微的。对于 \(D\) 中的所有 \(\mathbf{x}\)，令 \(h(\mathbf{x}) = f(\mathbf{x}) g(\mathbf{x})\)。计算 \(\mathbf{H}_h (\mathbf{x}_0)\)。 \(\lhd\)

**3.34** (改编自 [Rud]) 对于某个凸开集 \(D \subseteq \mathbb{R}^d\)，设 \(f : D \to \mathbb{R}\)。假设 \(f\) 在 \(D\) 的每一点上都是连续可微的，并且进一步 \(\frac{\partial f}{\partial x_1}(\mathbf{x}_0) = 0\) 对于 \(D\) 中的所有 \(\mathbf{x}_0\) 成立。证明 \(f\) 只依赖于 \(x_2,\ldots,x_d\)。 \(\lhd\)

**3.35** (改编自 [Rud]) 对于某个开集 \(D \subseteq \mathbb{R}\)，设 \(\mathbf{g} : D \to \mathbb{R}^d\)。假设 \(\mathbf{g}\) 在 \(D\) 的每一点上都是连续可微的，并且进一步

\[ \|\mathbf{g}(t)\|² = 1, \qquad \forall t \in D. \]

证明对于所有 \(t \in D\)，有 \(\mathbf{g}'(t)^T \mathbf{g}(t) = 0\)。[*提示:* 使用复合函数。] \(\lhd\)

**3.36** (改编自 [Khu]) 对于某个开集 \(D \subseteq \mathbb{R}^d\)，设 \(f : D \to \mathbb{R}\)。假设 \(f\) 在 \(D\) 的每一点上都是连续可微的，并且进一步

\[ f(t \mathbf{x}) = t^n f(\mathbf{x}), \]

对于 \(D\) 中的任意 \(\mathbf{x}\) 和任意标量 \(t\)，使得 \(t \mathbf{x} \in D\)。在这种情况下，\(f\) 被称为是 \(n\) 次齐次的。证明对于 \(D\) 中的所有 \(\mathbf{x}\)，我们有

\[ \mathbf{x}^T \nabla f(\mathbf{x}) = n f(\mathbf{x}). \]

\(\lhd\)

**3.37** (改编自 [Khu]) 在 \(\mathbf{0}\) 的邻域内对函数应用 **泰勒定理**。

\[ f(x_1, x_2) = \exp(x_2 \sin x_1). \]

[*提示:* 可以使用单变量函数的导数标准公式，无需证明。] \(\lhd\)

**3.38** (改编自 [Khu]) 在 \(\mathbf{0}\) 的邻域内对函数应用 **泰勒定理**。

\[ f(x_1, x_2) = \cos(x_1 x_2). \]

[*提示:* 可以使用单变量函数的导数标准公式，无需证明。] \(\lhd\)

**3.39** (改编自 [Khu]) 在 \(\mathbf{0}\) 的邻域内对函数应用 **泰勒定理**。

\[ f(x_1, x_2, x_3) = \sin(e^{x_1} + x_2² + x_3³). \]

[*提示:* 可以使用单变量函数的导数标准公式，无需证明。] \(\lhd\)

**3.40** (改编自 [Khu]) 考虑函数

\[ f(x_1, x_2) = (x_2 - x_1²)(x_2 - 2 x_1²). \]

a) 计算 \(f\) 在 \((0,0)\) 处的梯度和国值。

b) 证明 \((0,0)\) 不是 \(f\) 的严格局部极小值。[*提示:* 考虑形式为 \(x_1 = t^\alpha\) 和 \(x_2 = t^\beta\) 的参数曲线。]

c) 令 \(\mathbf{g}(t) = \mathbf{a} t\)，其中 \(\mathbf{a} = (a_1,a_2) \in \mathbb{R}²\) 是一个非零向量。证明 \(t=0\) 是 \(h(t) = f(\mathbf{g}(t))\) 的严格局部极小值。 \(\lhd\)

**3.41** 假设 \(f : \mathbb{R}^d \to \mathbb{R}\) 是凸函数。设 \(A \in \mathbb{R}^{d \times m}\) 和 \(\mathbf{b} \in \mathbb{R}^d\)。证明

\[ g(\mathbf{x}) = f(A \mathbf{x} + \mathbf{b}) \]

是 \(\mathbb{R}^m\) 上的凸集。 \(\lhd\)

**3.42** （改编自 [CVX]）对于 \(\mathbf{x} = (x_1,\ldots,x_n)\in\mathbb{R}^n\)，设

\[ x_{[1]} \geq x_{[2]} \geq \cdots \geq x_{[n]}, \]

\(\mathbf{x}\) 在非递增顺序下的坐标。设 \(1 \leq r \leq n\) 为一个整数，且 \(w_1 \geq w_2 \geq \cdots \geq w_r \geq 0\)。证明：

\[ f(\mathbf{x}) = \sum_{i=1}^r w_i x_{[i]}, \]

是凸集。 \(\lhd\)

**3.43** 对于一个固定的 \(\mathbf{z} \in \mathbb{R}^d\)，设

\[ f(\mathbf{x}) = \|\mathbf{x} - \mathbf{z}\|_2². \]

a) 计算 \(f\) 的梯度和对偶。

b) 证明 \(f\) 是强凸的。

c) 对于一个有限集 \(Z \subseteq \mathbb{R}^d\)，设

\[ g(\mathbf{x}) = \max_{\mathbf{z} \in Z} \|\mathbf{x} - \mathbf{z}\|_2². \]

使用 *问题 3.13* 来证明 \(g\) 是凸的。 \(\lhd\)

**3.44** 设 \(S_i\), \(i \in I\)，是 \(\mathbb{R}^d\) 的凸子集的集合。这里 \(I\) 可以是无限的。证明

\[ \bigcap_{i \in I} S_i, \]

是凸的。 \(\lhd\)

**3.45** a) 设 \(f_i\), \(i \in I\)，是 \(\mathbb{R}^d\) 上的凸实值函数的集合。使用 *问题 3.11, 3.44* 来证明

\[ g(\mathbf{x}) = \sup_{i \in I} f_i(\mathbf{x}), \]

是凸的。

b) 设 \(f : \mathbb{R}^{d+f} \to \mathbb{R}\) 为一个凸函数，且设 \(S \subseteq \mathbb{R}^{f}\) 为一个（不一定凸的）集合。使用 (a) 来证明函数

\[ g(\mathbf{x}) = \sup_{\mathbf{y} \in S} f(\mathbf{x},\mathbf{y}), \]

是凸的。你可以假设对于所有 \(\mathbf{x} \in \mathbb{R}^d\)，\(g(\mathbf{x}) < +\infty\)。 \(\lhd\)

**3.46** 使用 *问题 3.45* 来证明矩阵的最大特征值是所有对称矩阵集合上的凸函数。[提示：参见 *Rayleigh Quotient* 例子。首先证明 \(\langle \mathbf{x}, A\mathbf{x}\rangle\) 是 \(A\) 的线性函数。] \(\lhd\)

**3.47** 设 \(f : \mathbb{R}^d \to \mathbb{R}\) 为一个凸函数。证明全局最小值集合是凸的。 \(\lhd\)

**3.48** 考虑对称的 \(n \times n\) 矩阵集合

\[ \mathbf{S}^n = \left\{ X \in \mathbb{R}^{n \times n}\,:\, X = X^T \right\}. \]

a) 证明 \(\mathbf{S}^n\) 是 \(n \times n\) 矩阵向量空间的一个线性子空间，即对于所有 \(X_1, X_2 \in \mathbf{S}^n\) 和 \(\alpha \in \mathbb{R}\)

\[ X_1 + X_2 \in \mathbf{S}^n, \]

和

\[ \alpha X_1 \in \mathbf{S}^n. \]

b) 证明存在 \(\mathbf{S}^n\) 的一个大小为 \(d = {n \choose 2} + n\) 的生成集，即一组对称矩阵 \(X_1,\ldots,X_d \in \mathbf{S}^n\)，使得任何矩阵 \(Y \in \mathbf{S}^n\) 可以写成

\[ Y = \sum_{i=1}^d \alpha_i X_i, \]

对于某些 \(\alpha_1,\ldots,\alpha_d \in \mathbb{R}\)。

c) 证明你在 b) 中构造的矩阵 \(X_1,\ldots,X_d \in \mathbf{S}^n\) 在线性无关的意义上，

\[ \sum_{i=1}^d \alpha_i X_i = \mathbf{0}_{n \times n} \quad \implies \quad \alpha_1 = \cdots = \alpha_d = 0. \]

\(\lhd\)

**3.49** 设 \(f : D \to \mathbb{R}\) 是定义在集合

\[ D = \{\mathbf{x} = (x_1,\ldots,x_d) \in \mathbb{R}^d: x_i \geq 0, \forall i\}. \]

a) 证明 \(D\) 是凸集。

b) 证明凸集上的凸函数的*一阶最优性条件*简化为

\[ \frac{\partial f(\mathbf{x}⁰)}{\partial x_i} \geq 0, \qquad \forall i \in [d] \]

和

\[ \frac{\partial f(\mathbf{x}⁰)}{\partial x_i} = 0, \qquad \text{if $x_i⁰ > 0$}. \]

[*提示:* 在定理的条件中选择正确的 \(\mathbf{y}\)。] \(\lhd\)

**3.50** 设 \(f : \mathbb{R}^d \to \mathbb{R}\) 是二阶连续可微的，并且 \(m\)-强凸，其中 \(m>0\)。

a) 证明

\[ f(\mathbf{y}_n) \to +\infty, \]

沿着任何满足 \(\|\mathbf{y}_n\| \to +\infty\) 当 \(n \to +\infty\) 的序列 \((\mathbf{y}_n)\)。

b) 证明，对于任意 \(\mathbf{x}⁰\)，集合

\[ C = \{\mathbf{x} \in \mathbb{R}^d\,:\,f(\mathbf{x}) \leq f(\mathbf{x}⁰)\}, \]

是闭集且有界。

c) 证明存在唯一的全局最小值 \(\mathbf{x}^*\) 的 \(f\)。\(\lhd\)

**3.51** 假设 \(f : \mathbb{R}^d \to \mathbb{R}\) 是 \(L\)-光滑和 \(m\)-强凸的，并且在 \(\mathbf{x}^*\) 处有全局最小值。从任意点 \(\mathbf{x}⁰\) 开始应用梯度下降法，步长为 \(\alpha = 1/L\)。

a) 证明 \(\mathbf{x}^t \to \mathbf{x}^*\) 当 \(t \to +\infty\)。[*提示:* 在 \(\mathbf{x}^*\) 处使用强凸函数的*二次界*。]

b) 给出 \(\|\mathbf{x}^t - \mathbf{x}^*\|\) 的一个上界，它依赖于 \(t, m, L, f(\mathbf{x}⁰)\) 和 \(f(\mathbf{x}^*)\)。

\(\lhd\)

**3.52** 设 \(f : \mathbb{R}^d \to \mathbb{R}\) 是二阶连续可微的。

a) 证明

\[ \nabla f(\mathbf{y}) = \nabla f(\mathbf{x}) + \int_{0}¹ \mathbf{H}_f(\mathbf{x} + \xi \mathbf{p}) \,\mathbf{p} \,\mathrm{d}\xi, \]

其中 \(\mathbf{p} = \mathbf{y} - \mathbf{x}\)。这里向量值函数的积分是逐项进行的。[提示: 令 \(g_i(\xi) = \frac{\partial}{\partial x_i} f(\mathbf{x} + \xi \mathbf{p})\)。微积分基本定理表明 \(g_i(1) - g_i(0) = \int_{0}¹ g_i'(\xi) \,\mathrm{d}\xi.\)]

b) 假设 \(f\) 是 \(L\)-光滑的。使用 (a) 证明 \(f\) 的梯度在 \(L\)-Lipschitz 意义上是 \(L\)-Lipschitz。

\[ \|\nabla f(\mathbf{y}) - \nabla f(\mathbf{x})\|_2 \leq L \|\mathbf{y} - \mathbf{x}\|_2, \qquad \forall \mathbf{x}, \mathbf{y} \in \mathbb{R}^d. \]

\(\lhd\)

**3.53** 考虑二次函数

\[ f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^T P \mathbf{x} + \mathbf{q}^T \mathbf{x} + r, \]

其中 \(P\) 是一个对称矩阵。使用 *一阶凸性条件* 来证明 \(f\) 是凸函数当且仅当 \(P\) 是正半定矩阵。 \(\lhd\)

**3.54** 设

\[ \ell(\mathbf{x}; A, \mathbf{b}) = \frac{1}{n} \sum_{i=1}^n \left\{- b_i \log(\sigma(\boldsymbol{\alpha}_i^T \mathbf{x})) - (1-b_i) \log(1- \sigma(\boldsymbol{\alpha}_i^T \mathbf{x}))\right\}. \]

为逻辑回归中的损失函数，其中 \(A \in \mathbb{R}^{n \times d}\) 的行是 \(\boldsymbol{\alpha}_i^T\)。证明

\[ \mathbf{H}_{\ell}(\mathbf{x}; A, \mathbf{b}) \preceq \frac{1}{4n} A^T A. \]

[提示：阅读 *逻辑回归的平滑性* 公式的证明。] \(\lhd\)
