# 8.3\. 人工智能构建块 1：反向传播#

> 原文：[`mmids-textbook.github.io/chap08_nn/03_backprop/roch-mmids-nn-backprop.html`](https://mmids-textbook.github.io/chap08_nn/03_backprop/roch-mmids-nn-backprop.html)

我们发展了自动微分的基本数学基础。我们限制自己在一个特殊的环境中：多层递进函数。许多重要的分类器都采用序列组合的形式，其中参数对每个组合层都是特定的。我们展示了如何系统地应用**链式法则**到这样的函数上。我们还给出了一些例子。

## 8.3.1\. 前向与反向#

我们从一个固定参数的例子开始，以说明问题。假设 \(f : \mathbb{R}^d \to \mathbb{R}\) 可以表示为 \(L+1\) 个向量值函数 \(\bfg_i : \mathbb{R}^{n_i} \to \mathbb{R}^{n_{i+1}}\) 和一个实值函数 \(\ell : \mathbb{R}^{n_{L+1}} \to \mathbb{R}\) 的组合，如下所示

\[ f(\mathbf{x}) = \ell \circ \bfg_{L} \circ \bfg_{L-1} \circ \cdots \circ \bfg_1 \circ \bfg_0(\mathbf{x}) = \ell(\bfg_{L}(\bfg_{L-1}(\cdots \bfg_1(\bfg_0(\mathbf{x}))\cdots))). \]

这里 \(n_0 = d\) 是输入维度。我们还让 \(n_{L+1} = K\) 成为输出维度。考虑

\[ h(\mathbf{x}) = \bfg_{L}(\bfg_{L-1}(\cdots \bfg_1(\bfg_0(\mathbf{x}))\cdots)) \]

作为预测函数（即回归或分类函数），并将 \(\ell\) 视为损失函数。

首先观察函数 \(f\) 本身可以通过以下方式递归地从内部开始计算，这是直截了当的：

\[\begin{align*} \mathbf{z}_0 &:= \mathbf{x}\\ \mathbf{z}_1 &:= \bfg_0(\mathbf{z}_0)\\ \mathbf{z}_2 &:= \bfg_1(\mathbf{z}_1)\\ \vdots\\ \mathbf{z}_L &:= \bfg_{L-1}(\mathbf{z}_{L-1})\\ \hat{\mathbf{y}} := \mathbf{z}_{L+1} &:= \bfg_{L}(\mathbf{z}_{L})\\ f(\mathbf{x}) &:= \ell(\hat{\mathbf{y}}). \end{align*}\]

预见到神经网络的环境，我们感兴趣的主要应用，我们将 \(\mathbf{z}_0 = \mathbf{x}\) 称为“输入层”，\(\hat{\mathbf{y}} = \mathbf{z}_{L+1} = \bfg_{L}(\mathbf{z}_{L})\) 称为“输出层”，并将 \(\mathbf{z}_{1} = \bfg_0(\mathbf{z}_0), \ldots, \mathbf{z}_L = \bfg_{L-1}(\mathbf{z}_{L-1})\) 称为“隐藏层”。特别是，\(L\) 是隐藏层的数量。

**示例：** 我们将在本小节中始终使用以下运行示例。我们假设每个 \(\bfg_i\) 是一个线性映射，即 \(\bfg_i(\mathbf{z}_i) = \mathcal{W}_{i} \mathbf{z}_i\)，其中 \(\mathcal{W}_{i} \in \mathbb{R}^{n_{i+1} \times n_i}\) 是一个固定、已知的矩阵。还假设 \(\ell : \mathbb{R}^K \to \mathbb{R}\) 定义为

\[ \ell(\hat{\mathbf{y}}) = \frac{1}{2} \|\mathbf{y} - \hat{\mathbf{y}}\|², \]

对于一个固定的、已知的向量 \(\mathbf{y} \in \mathbb{R}^{K}\)。

以递归方式从内部开始计算 \(f\)，如上所示

\[\begin{align*} \mathbf{z}_0 &:= \mathbf{x}\\ \mathbf{z}_1 &:= \mathcal{W}_{0} \mathbf{z}_0 = \mathcal{W}_{0} \mathbf{x}\\ \mathbf{z}_2 &:= \mathcal{W}_{1} \mathbf{z}_1 = \mathcal{W}_{1} \mathcal{W}_{0} \mathbf{x}\\ \vdots\\ \mathbf{z}_L &:= \mathcal{W}_{L-1} \mathbf{z}_{L-1} = \mathcal{W}_{L-1} \cdots \mathcal{W}_{1} \mathcal{W}_{0} \mathbf{x}\\ \hat{\mathbf{y}} := \mathbf{z}_{L+1} &:= \mathcal{W}_{L} \mathbf{z}_{L} = \mathcal{W}_{L} \mathcal{W}_{L-1} \cdots \mathcal{W}_{1} \mathcal{W}_{0} \mathbf{x}\\ f(\mathbf{x}) &:= \ell(\hat{\mathbf{y}}) = \frac{1}{2}\|\mathbf{y} - \hat{\mathbf{y}}\|² = \frac{1}{2}\left\|\mathbf{y} - \mathcal{W}_{L} \mathcal{W}_{L-1} \cdots \mathcal{W}_{1} \mathcal{W}_{0} \mathbf{x}\right\|². \end{align*}\]

从本质上讲，我们是在将观察到的结果 \(\mathbf{y}\) 与基于输入 \(\mathbf{x}\) 的预测 \(\mathcal{W}_{L} \mathcal{W}_{L-1} \cdots \mathcal{W}_{1} \mathcal{W}_{0} \mathbf{x}\) 进行比较。

在本节中，我们探讨如何计算相对于 \(\mathbf{x}\) 的梯度。（实际上，我们更感兴趣的是计算相对于参数的梯度，即矩阵 \(\mathcal{W}_{0}, \ldots, \mathcal{W}_{L}\) 的元素，这是一个我们将在本节后面再次讨论的任务。我们还将对更复杂的——特别是非线性的——预测函数感兴趣。）\(\lhd\)

**数值角**：为了使问题更具体，我们考虑一个特定的例子。我们将使用 `torch.linalg.vector_norm`（[链接](https://pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html)）在 PyTorch 中计算欧几里得范数。假设 \(d=3\), \(L=1\), \(n_1 = 2\), 和 \(K = 2\)，我们有以下选择：

```py
x = torch.tensor([1.,0.,-1.], requires_grad=True)
y = torch.tensor([0.,1.])
W0 = torch.tensor([[0.,1.,-1.],[2.,0.,1.]])
W1 = torch.tensor([[-1.,0.],[2.,-1.]])

z0 = x
z1 = W0 @ z0
z2 = W1 @ z1
f = 0.5 * (torch.linalg.vector_norm(y-z2) ** 2)

print(z0) 
```

```py
tensor([ 1.,  0., -1.], requires_grad=True) 
```

```py
print(z1) 
```

```py
tensor([1., 1.], grad_fn=<MvBackward0>) 
```

```py
print(z2) 
```

```py
tensor([-1.,  1.], grad_fn=<MvBackward0>) 
```

```py
print(f) 
```

```py
tensor(0.5000, grad_fn=<MulBackward0>) 
```

\(\unlhd\)

**正向模式** \(\idx{forward mode}\xdi\) 我们准备应用**链式法则**

\[ \nabla f(\mathbf{x})^T = J_{f}(\mathbf{x}) = J_{\ell}(\mathbf{z}_{L+1}) J_{\bfg_L}(\mathbf{z}_L) J_{\bfg_{L-1}}(\mathbf{z}_{L-1}) \cdots J_{\bfg_1}(\mathbf{z}_1) J_{\bfg_0}(\mathbf{x}) \]

其中 \(\mathbf{z}_{i}\) 如上所述，我们使用了 \(\mathbf{z}_0 = \mathbf{x}\)。这里的矩阵乘积是定义良好的。实际上，\(J_{g_i}(\mathbf{z}_i)\) 的大小是 \(n_{i+1} \times n_{i}\)（即输出数量乘以输入数量），而 \(J_{g_{i-1}}(\mathbf{z}_{i-1})\) 的大小是 \(n_{i} \times n_{i-1}\)——因此维度是兼容的。

因此，我们可以像对 \(f\) 本身那样递归地计算 \(\nabla f(\mathbf{x})^T\)。实际上，我们可以同时计算这两个。这被称为正向模式：

\[\begin{align*} \mathbf{z}_0 &:= \mathbf{x}\\ \mathbf{z}_1 &:= \bfg_0(\mathbf{z}_0), \quad F_0 := J_{\bfg_0}(\mathbf{z}_0)\\ \mathbf{z}_2 &:= \bfg_1(\mathbf{z}_1), \quad F_1 := J_{\bfg_1}(\mathbf{z}_1)\, F_0\\ \vdots\\ \mathbf{z}_L &:= \bfg_{L-1}(\mathbf{z}_{L-1}), \quad F_{L-1} := J_{\bfg_{L-1}}(\mathbf{z}_{L-1})\, F_{L-2}\\ \hat{\mathbf{y}} := \mathbf{z}_{L+1} &:= \bfg_{L}(\mathbf{z}_{L}), \quad F_{L} := J_{\bfg_{L}}(\mathbf{z}_{L})\, F_{L-1}\\ f(\mathbf{x}) &:= \ell(\hat{\mathbf{y}}), \quad \nabla f(\mathbf{x})^T := J_{\ell}(\hat{\mathbf{y}}) F_L. \end{align*}\]

**示例:** **(继续)** 我们将此过程应用于正在运行的例子。线性映射\(\bfg_i(\mathbf{z}_i) = \mathcal{W}_{i} \mathbf{z}_i\)的雅可比矩阵是矩阵\(\mathcal{W}_{i}\)，正如我们之前所看到的。也就是说，对于任何\(\mathbf{z}_i\)，\(J_{\bfg_i}(\mathbf{z}_i) = \mathcal{W}_{i}\)。为了计算\(\ell\)的雅可比矩阵，我们将其重写为一个二次函数

\[\begin{align*} \ell(\hat{\mathbf{y}}) &= \frac{1}{2} \|\mathbf{y} - \hat{\mathbf{y}}\|²\\ &= \frac{1}{2} \mathbf{y}^T\mathbf{y} - \frac{1}{2} 2 \mathbf{y}^T\hat{\mathbf{y}} + \frac{1}{2}\hat{\mathbf{y}}^T \hat{\mathbf{y}}\\ &= \frac{1}{2} \hat{\mathbf{y}}^T I_{n_{L+1} \times n_{L+1}}\hat{\mathbf{y}} + (-\mathbf{y})^T\hat{\mathbf{y}} + \frac{1}{2} \mathbf{y}^T\mathbf{y}. \end{align*}\]

从先前的例子中，

\[ J_\ell(\hat{\mathbf{y}})^T = \nabla \ell(\hat{\mathbf{y}}) = \frac{1}{2}\left[I_{n_{L+1} \times n_{L+1}} + I_{n_{L+1} \times n_{L+1}}^T\right]\, \hat{\mathbf{y}} + (-\mathbf{y}) = \hat{\mathbf{y}} - \mathbf{y}. \]

将所有内容综合起来，我们得到

\[\begin{align*} F_0 &:= J_{\bfg_0}(\mathbf{z}_0) = \mathcal{W}_{0}\\ F_1 &:= J_{\bfg_1}(\mathbf{z}_1)\, F_0 = \mathcal{W}_{1} F_0 = \mathcal{W}_{1} \mathcal{W}_{0}\\ \vdots\\ F_{L-1} &:= J_{\bfg_{L-1}}(\mathbf{z}_{L-1})\, F_{L-2} = \mathcal{W}_{L-1} F_{L-2}= \mathcal{W}_{L-1} \cdots \mathcal{W}_{1} \mathcal{W}_{0}\\ F_{L} &:= J_{\bfg_{L}}(\mathbf{z}_{L})\, F_{L-1} = \mathcal{W}_{L} F_{L-1} = \mathcal{W}_{L} \mathcal{W}_{L-1} \cdots \mathcal{W}_{1} \mathcal{W}_{0}\\ \nabla f(\mathbf{x})^T &:= J_{\ell}(\hat{\mathbf{y}}) F_L = (\hat{\mathbf{y}} - \mathbf{y})^T F_L = (\hat{\mathbf{y}} - \mathbf{y})^T \mathcal{W}_{L} \mathcal{W}_{L-1} \cdots \mathcal{W}_{1} \mathcal{W}_{0}\\ &= (\mathcal{W}_{L} \mathcal{W}_{L-1} \cdots \mathcal{W}_{1} \mathcal{W}_{0} \mathbf{x} - \mathbf{y})^T \mathcal{W}_{L} \mathcal{W}_{L-1} \cdots \mathcal{W}_{1} \mathcal{W}_{0}. \end{align*}\]

\(\lhd\)

**数值角:** 我们回到我们的具体例子。在 PyTorch 中使用`.T`将列向量转换为行向量会引发错误，因为它仅适用于 2D 张量。相反，可以使用`[`torch.unsqueeze](https://pytorch.org/docs/stable/generated/torch.unsqueeze.html)`。下面，`(z2 - y).unsqueeze(0)`给`z2 - y`添加了一个维度，使其成为一个形状为\((1, 2)\)的 2D 张量。

```py
with torch.no_grad():
    F0 = W0
    F1 = W1 @ F0
    grad_f = (z2 - y).unsqueeze(0) @ F1

print(F0) 
```

```py
tensor([[ 0.,  1., -1.],
        [ 2.,  0.,  1.]]) 
```

```py
print(F1) 
```

```py
tensor([[ 0., -1.,  1.],
        [-2.,  2., -3.]]) 
```

```py
print(grad_f) 
```

```py
tensor([[ 0.,  1., -1.]]) 
```

我们可以使用 AD 来验证我们得到相同的结果。

```py
f.backward()
print(x.grad) 
```

```py
tensor([ 0.,  1., -1.]) 
```

\(\unlhd\)

**知识检查：** 通过对以下表达式求梯度直接得到：

\[ f(\mathbf{x}) = \frac{1}{2}\|\mathbf{y} - \mathcal{W}_{L} \mathcal{W}_{L-1} \cdots \mathcal{W}_{1} \mathcal{W}_{0} \mathbf{x}\|². \]

\(\checkmark\)

**逆模式** \(\idx{reverse mode}\xdi\) 我们刚才描述的对应于在 **链式法则** 公式中执行矩阵乘法

\[ \nabla f(\mathbf{x})^T = J_{f}(\mathbf{x}) = J_{\ell}(\hat{\mathbf{y}}) J_{\bfg_L}(\mathbf{z}_L) J_{\bfg_{L-1}}(\mathbf{z}_{L-1}) \cdots J_{\bfg_1}(\mathbf{z}_1) J_{\bfg_0}(\mathbf{x}). \]

*从右到左*。但还有另一种方法来做这件事——从左到右！这被称为逆模式。首先，我们像之前一样递归地计算 \(f\)，特别是我们计算 \(\mathbf{z}_i\)，然后从后往前计算

\[\begin{align*} G_{L+1} &:= J_{\ell}(\hat{\mathbf{y}})\\ G_{L} &:= G_{L+1}\,J_{g_{L}}(\mathbf{z}_{L})\\ &\vdots\\ G_1 &:= G_2 \, J_{\bfg_1}(\mathbf{z}_1)\\ \nabla f(\mathbf{x})^T &:= G_1 \, J_{\bfg_0}(\mathbf{z}_0) \end{align*}\]

这可能看起来只是对我们之前算法的微小修改。实际上并非如此——它要快得多！为了看到这一点，我们需要计算实现每个模式所需的基本操作数。

首先进行一个简单的计算：将一个 \(n \times m\) 矩阵 \(A = (a_{ij})_{i,j}\) 与一个 \(m \times \ell\) 矩阵 \(B= (b_{ij})_{i,j}\) 相乘需要多少个基本操作（即加法和乘法）？让我们问问 ChatGPT。这是答案：

“为了计算将一个 \(n \times m\) 矩阵与一个 \(m \times \ell\) 矩阵相乘所需的基本操作数，考虑结果 \(n \times \ell\) 矩阵中每个元素的涉及的计算。

结果矩阵的每个元素是第一个矩阵的某一行与第二个矩阵的某一列的点积。如果第一个矩阵表示为 \(A\)，第二个矩阵表示为 \(B\)，那么结果矩阵 \(C\) 中位置 \(i, j\) 的元素计算如下：

\[ c_{ij} = a_{i1}b_{1j} + a_{i2}b_{2j} + \ldots + a_{im}b_{mj} \]

这个计算涉及 \(m\) 次乘法和 \(m - 1\) 次加法，对于每个元素总共是 \(m + (m - 1) = 2m - 1\) 个基本操作。

由于矩阵 \(C\) 有 \(n \times \ell\) 个元素，整个矩阵 \(C\) 所需的基本操作总数是：

\[ (n \ell)(2m - 1) = 2nm\ell - n\ell \]

因此，将一个 \(n \times m\) 矩阵与一个 \(m \times \ell\) 矩阵相乘需要 \(2nm\ell - n\ell\) 个基本操作。”

回到计算梯度的过程，我们计算每个模式所需的基本操作数。为了简化计算，我们假设 \(n_0 = d\)，\(n_1 = n_2 = \cdots = n_{L} = m\)，\(n_{L+1} = K\)。

*正向:* 矩阵 \(F_0 = J_{\bfg_0}(\mathbf{z}_0)\) 的维度是 \(m \times d\)。矩阵 \(F_1\)，作为 \(J_{\bfg_1}(\mathbf{z}_1) \in \mathbb{R}^{m \times m}\) 和 \(F_0 \in \mathbb{R}^{m \times d}\) 的乘积，其维度是 \(m \times d\)；因此，计算它需要 \(m (2m-1) d\) 次操作。对于 \(F_2, \ldots, F_{L-1}\) 也是如此（检查一下！）！通过类似的考虑，矩阵 \(F_L\) 的维度是 \(K \times d\)，计算它需要 \(K (2m-1) d\) 次操作。最后，\(\nabla f(\mathbf{x})^T = J_{\ell}(\mathbf{z}_{L+1}) F_L \in \mathbb{R}^{1 \times d}\) 并需要 \((2K-1) d\) 次操作。总的来说，操作次数是

\[ (L-1) m (2m-1) d + K (2m-1) d + (2K-1) d. \]

如果我们将 \(K\) 视为一个小的常数并忽略较小的阶数项，这大约是 \(2 L m² d\)。

*反向:* 矩阵 \(G_{L+1} = J_{\ell}(\mathbf{z}_{L+1})\) 的维度是 \(1 \times K\)。矩阵 \(G_{L}\)，作为 \(G_{L+1} \in \mathbb{R}^{1 \times K}\) 和 \(J_{g_{L}}(\mathbf{z}_{L}) \in \mathbb{R}^{K \times m}\) 的乘积，其维度是 \(1 \times m\)；因此，计算它需要 \((2K-1) m\) 次操作。矩阵 \(G_{L-1}\)，作为 \(G_{L} \in \mathbb{R}^{1 \times m}\) 和 \(J_{g_{L-1}}(\mathbf{z}_{L-1}) \in \mathbb{R}^{m \times m}\) 的乘积，其维度是 \(1 \times m\)；因此，计算它需要 \((2m-1) m\) 次操作。对于 \(G_{L-2}, \ldots, G_{1}\) 也是如此（检查一下！）！通过类似的考虑，\(\nabla f(\mathbf{x})^T = G_1 \, J_{\bfg_0}(\mathbf{z}_0) \in \mathbb{R}^{1 \times d}\) 并需要 \((2m-1) d\) 次操作。总的来说，操作次数是

\[ (2K-1) m + (L-1) (2m-1) m + (2m-1) d. \]

这大约是 \(2 L m² + 2 m d\) – 这可以比 \(2 L m² d\) 小得多！换句话说，反向模式方法可以快得多。特别注意的是，反向模式中的所有计算都是矩阵-向量乘积（或者更精确地说，是行向量-矩阵乘积），而不是矩阵-矩阵乘积。

**示例:** **(继续)** 我们将反向模式方法应用于先前的例子。我们得到

\[\begin{align*} G_{L+1} &:= J_{\ell}(\hat{\mathbf{y}}) = (\hat{\mathbf{y}} - \mathbf{y})^T\\ G_{L} &:= G_{L+1}\,J_{g_{L}}(\mathbf{z}_{L}) = G_{L+1} \mathcal{W}_{L} = (\hat{\mathbf{y}} - \mathbf{y})^T \mathcal{W}_{L} \\ \vdots\\ G_1 &:= G_2 \, J_{\bfg_1}(\mathbf{z}_1) = G_2 \mathcal{W}_{1} = [(\hat{\mathbf{y}} - \mathbf{y})^T \mathcal{W}_{L} \cdots \mathcal{W}_{2}] \mathcal{W}_{1} \\ \nabla f(\mathbf{x})^T &:= G_1 \, J_{\bfg_0}(\mathbf{z}_0) = [(\hat{\mathbf{y}} - \mathbf{y})^T \mathcal{W}_{L} \cdots \mathcal{W}_{2}\mathcal{W}_{1}] \mathcal{W}_{0}\\ &= (\mathcal{W}_{L} \mathcal{W}_{L-1} \cdots \mathcal{W}_{1} \mathcal{W}_{0} \mathbf{x} - \mathbf{y})^T \mathcal{W}_{L} \mathcal{W}_{L-1} \cdots \mathcal{W}_{1} \mathcal{W}_{0}, \end{align*}\]

这与我们的先前计算相匹配。注意，所有计算都涉及将行向量乘以矩阵。 \(\lhd\)

**数值角**: 我们尝试我们的特定例子。

```py
with torch.no_grad():
    G2 = (z2 - y).unsqueeze(0)
    G1 = G2 @ W1
    grad_f = G1 @ W0

print(G2) 
```

```py
tensor([[-1.,  0.]]) 
```

```py
print(G1) 
```

```py
tensor([[1., 0.]]) 
```

```py
print(grad_f) 
```

```py
tensor([[ 0.,  1., -1.]]) 
```

我们确实再次得到了相同的答案。

\(\unlhd\)

为了提供更多关于通过反向模式获得的节省的洞察，考虑以下简单的计算。设 \(A, B \in \mathbb{R}^{n \times n}\) 和 \(\mathbf{v} \in \mathbb{R}^n\)。假设我们想要计算 \(\mathbf{v}^T B A\)。通过矩阵乘法的 [结合律](https://en.wikipedia.org/wiki/Associative_property)，有两种方法来做这件事：计算 \(\mathbf{v}^{T}(BA)\)（即，首先计算 \(BA\) 然后乘以 \(\mathbf{v}^T\)；或者计算 \((\mathbf{v}^T B) A\)。第一种方法需要 \(n²(2n-1) + n(2n-1)\) 次操作，而第二种方法只需要 \(2n(2n-1)\)。后者要小得多，因为当 \(n\) 很大时，\(2 n³\)（第一种方法中的主导项）的增长速度比 \(4 n²\)（第二种方法中的主导项）快得多。

为什么会发生这种情况？理解这一点的其中一种方法是考虑输出 \(\mathbf{v}^T B A\) 是 \(A\) 的行的一个 *线性组合* – 实际上是一个非常具体的线性组合。在第一种方法中，我们计算 \(BA\)，这给我们 \(n\) 个不同的 \(A\) 的行的线性组合 – 没有一个是我们想要的 – 然后我们通过乘以 \(\mathbf{v}^T\) 来计算所需的线性组合。这是浪费的。在第二种方法中，我们立即计算我们寻求的特定线性组合的系数 – \(\mathbf{v}^T B\) – 然后通过右乘 \(A\) 来计算这个线性组合。

虽然我们在本小节中考察的设置具有启发性，但它并不完全是我们想要的。在机器学习环境中，每个“层” \(\bfg_i\) 都有参数（在我们的运行示例中，是 \(\mathcal{W}_{i}\) 的条目）并且我们希望针对这些参数进行优化。为此，我们需要关于参数的梯度，而不是输入 \(\mathbf{x}\)。在下一个小节中，我们将考虑当前设置的推广，即递进函数，这将使我们能够做到这一点。符号变得更加复杂，但基本思想保持不变。

## 8.3.2\. 递进函数#

如前所述，虽然将预测函数 \(h\)（例如，分类器）定义为输入数据 \(\mathbf{x}\in \mathbb{R}^{d}\) 的函数可能看起来很自然，但在拟合数据时，我们最终感兴趣的将 \(h\) 视为需要调整的参数 \(\mathbf{w} \in \mathbb{R}^r\) 的函数 – 在一个固定的数据集上。因此，在本节中，输入 \(\mathbf{x}\) 是固定的，而参数向量 \(\mathbf{w}\) 现在是可变的。

**第一个例子** 我们使用前一小节中的例子来说明主要思想。也就是说，假设 \(d=3\)，\(L=1\)，\(n_1 = 2\)，\(K = 2\)。固定一个数据样本 \(\mathbf{x} = (x_1,x_2,x_3) \in \mathbb{R}³, \mathbf{y} = (y_1, y_2) \in \mathbb{R}²\)。对于 \(i=0, 1\)，我们使用以下记号

\[\begin{split} \mathcal{W}_{0} = \begin{pmatrix} w_0 & w_1 & w_2\\ w_3 & w_4 & w_5 \end{pmatrix} \quad \text{and} \quad \mathcal{W}_{1} = \begin{pmatrix} w_6 & w_7\\ w_8 & w_9 \end{pmatrix}. \end{split}\]

让

\[ \ell(\hat{\mathbf{y}}) = \frac{1}{2} \|\mathbf{y} - \hat{\mathbf{y}}\|² = \frac{1}{2}(y_1 - \hat{y}_1)² + \frac{1}{2}(y_2 - \hat{y}_2)². \]

我们将“层”函数 \(\bfg_i\) 的记号改为反映它现在是一个两个（连接的）向量的函数：来自前一层的输入 \(\mathbf{z}_i = (z_{i,1},\ldots,z_{i,n_i})\) 和一个特定层的参数集 \(\mathbf{w}_i\)。也就是说，

\[\begin{split} \bfg_i(\mathbf{z}_i, \mathbf{w}_i) = \mathcal{W}_{i} \mathbf{z}_i = \begin{pmatrix} (\mathbf{w}_i^{(1)})^T\\ (\mathbf{w}_i^{(2)})^T \end{pmatrix} \mathbf{z}_i \end{split}\]

的梯度，其中 \(\mathbf{w}_i = (\mathbf{w}_i^{(1)}, \mathbf{w}_i^{(2)})\) 是 \(\mathcal{W}_{i}\) 的行的连接（作为列向量）。另一种说法是

\[ \mathbf{w}_i = \mathrm{vec}(\mathcal{W}_{i}^T), \]

其中我们取转置将行转换为列。更具体地说，

\[ \bfg_0(\mathbf{z}_0, \mathbf{w}_0) = \mathcal{W}_{0} \mathbf{z}_{0} \quad\text{with}\quad \mathbf{w}_0 = (w_0, w_1, w_2, w_3, w_4, w_5) \]

（即，\(\mathbf{w}_0^{(1)} = (w_0, w_1, w_2)\) 和 \(\mathbf{w}_0^{(2)} = (w_3, w_4, w_5)\)）和

\[ \bfg_1(\mathbf{z}_1, \mathbf{w}_1) = \mathcal{W}_{1} \mathbf{z}_{1} \quad\text{with}\quad \mathbf{w}_1 = (w_6, w_7, w_8, w_9) \]

（即，\(\mathbf{w}_1^{(1)} = (w_6, w_7)\) 和 \(\mathbf{w}_1^{(2)} = (w_8, w_9)\)）。

我们试图计算

\[\begin{align*} f(\mathbf{w}) &= \ell(\bfg_1(\bfg_0(\mathbf{x},\mathbf{w}_0),\mathbf{w}_1))\\ &= \frac{1}{2} \|\mathbf{y} - \mathcal{W}_{1} \mathcal{W}_{0} \mathbf{x}\|²\\ &= \frac{1}{2}\left(y_1 - w_6(w_0 x_1 + w_1 x_2 + w_2 x_3) - w_7(w_3 x_1 + w_4 x_2 + w_5 x_3)\right)²\\ & \qquad + \frac{1}{2}\left(y_2 - w_8(w_0 x_1 + w_1 x_2 + w_2 x_3) - w_9(w_3 x_1 + w_4 x_2 + w_5 x_3)\right)². \end{align*}\]

通过反向应用**链式法则**，正如我们在前一小节中所证明的那样——但这次我们对参数求梯度

\[ \mathbf{w} := (\mathbf{w}_0, \mathbf{w}_1) = (w_0,w_1,\ldots,w_9). \]

注意记号中的一个关键变化：我们现在相应地认为 \(f\) 是 \(\mathbf{w}\) 的函数；\(\mathbf{x}\) 的作用是隐含的。

另一方面，当我们刚刚表示我们只关心相对于前者的梯度时，现在认为 \(\bfg_i\) 是其自身参数和来自前一层的输入的函数，这似乎是反直觉的。但是，正如我们将看到的，结果是我们确实需要两个相对于的雅可比矩阵，因为前一层的输入实际上依赖于前一层的参数。例如，\(\bfg_1(\mathbf{z}_1, \mathbf{w}_1) = \mathcal{W}_{1} \mathbf{z}_{1}\)，其中 \(\mathbf{z}_{1} = \bfg_0(\mathbf{z}_0, \mathbf{w}_0) = \mathcal{W}_{0} \mathbf{z}_{0}\)。

回想一下，我们已经在先前的例子中计算了所需的雅可比矩阵 \(J_{\bfg_0}\) 和 \(J_{\bfg_1}\)。我们也计算了 \(\ell\) 的雅可比矩阵 \(J_{\ell}\)。在这个时候，应用链式法则并推断 \(f\) 的梯度是

\[ J_{\ell}(\bfg_1(\bfg_0(\mathbf{x},\mathbf{w}_0),\mathbf{w}_1)) \,J_{\bfg_1}(\bfg_0(\mathbf{x},\mathbf{w}_0),\mathbf{w}_1) \,J_{\bfg_0}(\mathbf{x}, \mathbf{w}_0). \]

但这是不正确的。首先，维度不匹配！例如，\(J_{\bfg_0} \in \mathbb{R}^{2 \times 9}\)，因为 \(\bfg_0\) 有 \(2\) 个输出和 \(9\) 个输入（即 \(z_{0,1}, z_{0,2}, z_{0,3}, w_0, w_1, w_2, w_3, w_4, w_5\)），而 \(J_{\bfg_1} \in \mathbb{R}^{2 \times 6}\)，因为 \(\bfg_1\) 有 \(2\) 个输出和 \(6\) 个输入（即 \(z_{1,1}, z_{1,2}, w_6, w_7, w_8, w_9\)）。那么出了什么问题？

函数 \(f\) 实际上并不是由函数 \(\ell\)、\(\bfg_1\) 和 \(\bfg_0\) 的简单组合。确实，相对于要微分的是逐步引入的参数，每一层都注入自己的附加参数，这些参数不是从前一层获得的。因此，我们不能像前一小节中发生的那样，将 \(f\) 的梯度写成雅可比矩阵的简单乘积。

但并非一切尽失。我们下面将展示，我们仍然可以逐步应用链式法则，同时考虑到每一层的附加参数。借鉴前一小节，我们首先向前计算 \(f\) 和雅可比矩阵，然后向后计算梯度 \(\nabla f\)。我们使用背景部分中的符号 \(\mathbb{A}_{n}[\mathbf{x}]\) 和 \(\mathbb{B}_{n}[\mathbf{z}]\)。

在正向阶段，我们计算 \(f\) 本身和所需的雅可比矩阵：

\[\begin{align*} &\mathbf{z}_0 := \mathbf{x}\\ & = (x_1, x_2, x_3)\\ &\mathbf{z}_1 := \bfg_0(\mathbf{z}_0, \mathbf{w}_0) = \mathcal{W}_{0} \mathbf{z}_{0}\\ &= \begin{pmatrix} (\mathbf{w}_0^{(1)})^T\mathbf{x}\\ (\mathbf{w}_0^{(2)})^T\mathbf{x}\end{pmatrix} = \begin{pmatrix} w_0 x_1 + w_1 x_2 + w_2 x_3\\ w_3 x_1 + w_4 x_2 + w_5 x_3 \end{pmatrix}\\ &J_{\bfg_0}(\mathbf{z}_0, \mathbf{w}_0) := \begin{pmatrix} \mathbb{A}_{2}[\mathbf{w}_0] & \mathbb{B}_{2}[\mathbf{z}_0] \end{pmatrix} = \begin{pmatrix} \mathcal{W}_{0} & I_{2\times 2} \otimes \mathbf{z}_0^T \end{pmatrix}\\ &= \begin{pmatrix} w_0 & w_1 & w_2 & x_1 & x_2 & x_3 & 0 & 0 & 0\\ w_3 & w_4 & w_5 & 0 & 0 & 0 & x_1 & x_2 & x_3 \end{pmatrix} \end{align*}\]\[\begin{align*} &\hat{\mathbf{y}} := \mathbf{z}_2 := \bfg_1(\mathbf{z}_1, \mathbf{w}_1) = \mathcal{W}_{1} \mathbf{z}_{1}\\ &= \begin{pmatrix} w_6 z_{1,1} + w_7 z_{1,2}\\ w_8 z_{1,1} + w_9 z_{1,2} \end{pmatrix}\\ &= \begin{pmatrix} w_6 (\mathbf{w}_0^{(1)})^T\mathbf{x} + w_7 (\mathbf{w}_0^{(2)})^T\mathbf{x}\\ w_8 (\mathbf{w}_0^{(1)})^T\mathbf{x} + w_9 (\mathbf{w}_0^{(2)})^T\mathbf{x} \end{pmatrix}\\ &= \begin{pmatrix} w_6(w_0 x_1 + w_1 x_2 + w_2 x_3) + w_7(w_3 x_1 + w_4 x_2 + w_5 x_3)\\ w_8(w_0 x_1 + w_1 x_2 + w_2 x_3) + w_9(w_3 x_1 + w_4 x_2 + w_5 x_3) \end{pmatrix}\\ &J_{\bfg_1}(\mathbf{z}_1, \mathbf{w}_1):= \begin{pmatrix} \mathbb{A}_{2}[\mathbf{w}_1] & \mathbb{B}_{2}[\mathbf{z}_1] \end{pmatrix} = \begin{pmatrix} \mathcal{W}_{1} & I_{2\times 2} \otimes \mathbf{z}_1^T \end{pmatrix}\\ &= \begin{pmatrix} w_6 & w_7 & z_{1,1} & z_{1,2} & 0 & 0\\ w_8 & w_9 & 0 & 0 & z_{1,1} & z_{1,2} \end{pmatrix}\\ &= \begin{pmatrix} w_6 & w_7 & (\mathbf{w}_0^{(1)})^T\mathbf{x} & (\mathbf{w}_0^{(2)})^T\mathbf{x} & 0 & 0\\ w_8 & w_9 & 0 & 0 & (\mathbf{w}_0^{(1)})^T\mathbf{x} & (\mathbf{w}_0^{(2)})^T\mathbf{x} \end{pmatrix} \end{align*}\]\[\begin{align*} &f(\mathbf{x}) := \ell(\hat{\mathbf{y}}) = \frac{1}{2} \|\mathbf{y} - \hat{\mathbf{y}}\|²\\ &= \frac{1}{2}\left(y_1 - w_6(w_0 x_1 + w_1 x_2 + w_2 x_3) - w_7(w_3 x_1 + w_4 x_2 + w_5 x_3)\right)²\\ & \qquad + \frac{1}{2}\left(y_2 - w_8(w_0 x_1 + w_1 x_2 + w_2 x_3) - w_9(w_3 x_1 + w_4 x_2 + w_5 x_3)\right)²\\ &J_{\ell}(\hat{\mathbf{y}}) = (\hat{\mathbf{y}} - \mathbf{y})^T\\ &= \begin{pmatrix} w_6 (\mathbf{w}_0^{(1)})^T\mathbf{x} + w_7 (\mathbf{w}_0^{(2)})^T\mathbf{x} - y_1 & w_8 (\mathbf{w}_0^{(1)})^T\mathbf{x} + w_9 (\mathbf{w}_0^{(2)})^T\mathbf{x} - y_2 \end{pmatrix}. \end{align*}\]

我们现在计算 \(f\) 关于 \(\mathbf{w}\) 的梯度。我们从 \(\mathbf{w}_1 = (w_6, w_7, w_8, w_9)\) 开始。对于这一步，我们将 \(f\) 视为 \(\ell(\bfg_1(\mathbf{z}_1, \mathbf{w}_1))\) 的复合函数。在这里，\(\mathbf{z}_1\) 不依赖于 \(\mathbf{w}_1\)，因此可以认为在这个计算中是固定的。根据**链式法则**

\[\begin{align*} \frac{\partial f(\mathbf{w})}{\partial w_6} &= \frac{\partial \ell(\bfg_1(\mathbf{z}_1, \mathbf{w}_1))}{\partial w_6} = \frac{\partial \ell(\hat{\mathbf{y}})}{\partial \hat{y}_1} \frac{\partial g_{1,1}(\mathbf{z}_1, \mathbf{w}_1)}{\partial w_6} + \frac{\partial \ell(\hat{\mathbf{y}})}{\partial \hat{y}_2} \frac{\partial g_{1,2}(\mathbf{z}_1, \mathbf{w}_1)}{\partial w_6} = (\hat{y}_1 - y_1) z_{1,1} \end{align*}\]

我们使用了这样一个事实，即 \(g_{1,2}(\mathbf{z}_1, \mathbf{w}_1) = w_8 z_{1,1} + w_9 z_{1,2}\) 不依赖于 \(w_6\)，因此 \(\frac{\partial g_{1,2}(\mathbf{z}_1, \mathbf{w}_1)}{\partial w_6} = 0\)。同样地

\[\begin{align*} \frac{\partial f(\mathbf{w})}{\partial w_7} &= \frac{\partial \ell(\bfg_1(\mathbf{z}_1, \mathbf{w}_1))}{\partial w_7} =\frac{\partial \ell(\hat{\mathbf{y}})}{\partial \hat{y}_1} \frac{\partial g_{1,1}(\mathbf{z}_1, \mathbf{w}_1)}{\partial w_7} + \frac{\partial \ell(\hat{\mathbf{y}})}{\partial \hat{y}_2} \frac{\partial g_{1,2}(\mathbf{z}_1, \mathbf{w}_1)}{\partial w_7} = (\hat{y}_1 - y_1) z_{1,2}\\ \frac{\partial f(\mathbf{w})}{\partial w_8} &= \frac{\partial \ell(\bfg_1(\mathbf{z}_1, \mathbf{w}_1))}{\partial w_8} =\frac{\partial \ell(\hat{\mathbf{y}})}{\partial \hat{y}_1} \frac{\partial g_{1,1}(\mathbf{z}_1, \mathbf{w}_1)}{\partial w_8} + \frac{\partial \ell(\hat{\mathbf{y}})}{\partial \hat{y}_2} \frac{\partial g_{1,2}(\mathbf{z}_1, \mathbf{w}_1)}{\partial w_8} = (\hat{y}_2 - y_2) z_{1,1}\\ \frac{\partial f(\mathbf{w})}{\partial w_9} &= \frac{\partial \ell(\bfg_1(\mathbf{z}_1, \mathbf{w}_1))}{\partial w_9} =\frac{\partial \ell(\hat{\mathbf{y}})}{\partial \hat{y}_1} \frac{\partial g_{1,1}(\mathbf{z}_1, \mathbf{w}_1)}{\partial w_9} + \frac{\partial \ell(\hat{\mathbf{y}})}{\partial \hat{y}_2} \frac{\partial g_{1,2}(\mathbf{z}_1, \mathbf{w}_1)}{\partial w_9} = (\hat{y}_2 - y_2) z_{1,2}. \end{align*}\]

以矩阵形式表示，这是

\[\begin{align*} &\begin{pmatrix}\frac{\partial f(\mathbf{w})}{\partial w_6} & \frac{\partial f(\mathbf{w})}{\partial w_7} & \frac{\partial f(\mathbf{w})}{\partial w_8} & \frac{\partial f(\mathbf{w})}{\partial w_9} \end{pmatrix}\\ &= J_{\ell}(\hat{\mathbf{y}}) \,\mathbb{B}_{2}[\mathbf{z}_1]\\ &= (\hat{\mathbf{y}} - \mathbf{y})^T (I_{2\times 2} \otimes \mathbf{z}_1^T)\\ &= (\hat{\mathbf{y}} - \mathbf{y})^T \otimes \mathbf{z}_1^T\\ &= \begin{pmatrix} (\hat{y}_1 - y_1) z_{1,1} & (\hat{y}_1 - y_1) z_{1,2} & (\hat{y}_2 - y_2) z_{1,1} & (\hat{y}_2 - y_2) z_{1,2} \end{pmatrix} \end{align*}\]

在最后一行我们使用了**克罗内克积的性质（f）**。

为了计算关于 \(\mathbf{w}_0 = (w_0, w_1, \ldots, w_5)\) 的偏导数，我们首先需要计算关于 \(\mathbf{z}_1 = (z_{1,1}, z_{1,2})\) 的偏导数，因为 \(f\) 通过它依赖于 \(\mathbf{w}_0\)。对于这个计算，我们再次将 \(f\) 视为 \(\ell(\bfg_1(\mathbf{z}_1, \mathbf{w}_1))\) 的复合，但这次我们的重点是变量 \(\mathbf{z}_1\)。我们得到

\[\begin{align*} \frac{\partial f(\mathbf{w})}{\partial z_{1,1}} &= \frac{\partial \ell(\bfg_1(\mathbf{z}_1, \mathbf{w}_1))}{\partial z_{1,1}}\\ &= \frac{\partial \ell(\hat{\mathbf{y}})}{\partial \hat{y}_1} \frac{\partial g_{1,1}(\mathbf{z}_1, \mathbf{w}_1)}{\partial z_{1,1}} + \frac{\partial \ell(\hat{\mathbf{y}})}{\partial \hat{y}_2} \frac{\partial g_{1,2}(\mathbf{z}_1, \mathbf{w}_1)}{\partial z_{1,1}}\\ &= (\hat{y}_1 - y_1) w_6 + (\hat{y}_2 - y_2) w_8 \end{align*}\]

和

\[\begin{align*} \frac{\partial f(\mathbf{w})}{\partial z_{1,2}} &= \frac{\partial \ell(\bfg_1(\mathbf{z}_1, \mathbf{w}_1))}{\partial z_{1,2}}\\ &= \frac{\partial \ell(\hat{\mathbf{y}})}{\partial \hat{y}_1} \frac{\partial g_{1,1}(\mathbf{z}_1, \mathbf{w}_1)}{\partial z_{1,2}} + \frac{\partial \ell(\hat{\mathbf{y}})}{\partial \hat{y}_2} \frac{\partial g_{1,2}(\mathbf{z}_1, \mathbf{w}_1)}{\partial z_{1,2}}\\ &= (\hat{y}_1 - y_1) w_7 + (\hat{y}_2 - y_2) w_9. \end{align*}\]

以矩阵形式，这可以表示为

\[\begin{align*} &\begin{pmatrix}\frac{\partial f(\mathbf{w})}{\partial z_{1,1}} & \frac{\partial f(\mathbf{w})}{\partial z_{1,2}} \end{pmatrix}\\ &= J_{\ell}(\hat{\mathbf{y}}) \,\mathbb{A}_{2}[\mathbf{w}_1]\\ &= (\hat{\mathbf{y}} - \mathbf{y})^T \mathcal{W}_1\\ &= \begin{pmatrix} (\hat{y}_1 - y_1) w_6 + (\hat{y}_2 - y_2) w_8 & (\hat{y}_1 - y_1) w_7 + (\hat{y}_2 - y_2) w_9 \end{pmatrix}. \end{align*}\]

向量 \(\left(\frac{\partial f(\mathbf{w})}{\partial z_{1,1}}, \frac{\partial f(\mathbf{w})}{\partial z_{1,2}}\right)\) 被称为伴随向量。

我们现在计算 \(f\) 关于 \(\mathbf{w}_0 = (w_0, w_1, \ldots, w_5)\) 的梯度。对于这一步，我们将 \(f\) 视为 \(\ell(\bfg_1(\mathbf{z}_1, \mathbf{w}_1))\) 作为 \(\mathbf{z}_1\) 的函数和 \(\bfg_0(\mathbf{z}_0, \mathbf{w}_0)\) 作为 \(\mathbf{w}_0\) 的函数的复合。在这里，\(\mathbf{z}_0\) 不依赖于 \(\mathbf{w}_0\)，因此可以认为在这个计算中是固定的。根据**链式法则**

\[\begin{align*} \frac{\partial f(\mathbf{w})}{\partial w_0} &= \frac{\partial \ell(\bfg_1(\bfg_0(\mathbf{z}_0, \mathbf{w}_0), \mathbf{w}_1))}{\partial w_0}\\ &= \frac{\partial \ell(\bfg_1(\mathbf{z}_1, \mathbf{w}_1))}{\partial z_{1,1}} \frac{\partial g_{0,1}(\mathbf{z}_0, \mathbf{w}_0)}{\partial w_0} + \frac{\partial \ell(\bfg_1(\mathbf{z}_1, \mathbf{w}_1))}{\partial z_{1,2}} \frac{\partial g_{0,2}(\mathbf{z}_0, \mathbf{w}_0)}{\partial w_0}\\ &= ((\hat{y}_1 - y_1) w_6 + (\hat{y}_2 - y_2) w_8) z_{0,1}\\ &= ((\hat{y}_1 - y_1) w_6 + (\hat{y}_2 - y_2) w_8) x_{1}, \end{align*}\]

其中我们使用了 \(g_{0,2}(\mathbf{z}_0, \mathbf{w}_0) = w_3 z_{0,1} + w_4 z_{0,2} + w_5 z_{0,3}\) 不依赖于 \(w_0\) 的这一事实，因此 \(\frac{\partial g_{0,2}(\mathbf{z}_0, \mathbf{w}_0)}{\partial w_0} = 0\)。

类似地（检查一下！）

\[\begin{align*} \frac{\partial f(\mathbf{w})}{\partial w_1} &= ((\hat{y}_1 - y_1) w_6 + (\hat{y}_2 - y_2) w_8) x_{2}\\ \frac{\partial f(\mathbf{w})}{\partial w_2} &= ((\hat{y}_1 - y_1) w_6 + (\hat{y}_2 - y_2) w_8) x_{3}\\ \frac{\partial f(\mathbf{w})}{\partial w_3} &= ((\hat{y}_1 - y_1) w_7 + (\hat{y}_2 - y_2) w_9) x_{1}\\ \frac{\partial f(\mathbf{w})}{\partial w_4} &= ((\hat{y}_1 - y_1) w_7 + (\hat{y}_2 - y_2) w_9) x_{2}\\ \frac{\partial f(\mathbf{w})}{\partial w_5} &= ((\hat{y}_1 - y_1) w_7 + (\hat{y}_2 - y_2) w_9) x_{3}. \end{align*}\]

以矩阵形式，这可以表示为

\[\begin{align*} &\begin{pmatrix}\frac{\partial f(\mathbf{w})}{\partial w_0} & \frac{\partial f(\mathbf{w})}{\partial w_1} & \frac{\partial f(\mathbf{w})}{\partial w_2} & \frac{\partial f(\mathbf{w})}{\partial w_3} & \frac{\partial f(\mathbf{w})}{\partial w_4} & \frac{\partial f(\mathbf{w})}{\partial w_5} \end{pmatrix}\\ &= J_{\ell}(\hat{\mathbf{y}}) \,\mathbb{A}_{2}[\mathbf{w}_1] \,\mathbb{B}_{2}[\mathbf{z}_0]\\ &= (\hat{\mathbf{y}} - \mathbf{y})^T \mathcal{W}_1 (I_{2\times 2} \otimes \mathbf{z}_0^T)\\ &= ((\hat{\mathbf{y}} - \mathbf{y})^T \mathcal{W}_1) \otimes \mathbf{x}^T\\ &= \begin{pmatrix} ((\hat{y}_1 - y_1) w_6 + (\hat{y}_2 - y_2) w_8) x_{1} & \cdots & ((\hat{y}_1 - y_1) w_7 + (\hat{y}_2 - y_2) w_9) x_{3} \end{pmatrix} \end{align*}\]

其中我们在最后一行使用了 *克罗内克积的性质（f）*。

总结一下，

\[ \nabla f (\mathbf{w})^T = \begin{pmatrix} (\hat{\mathbf{y}} - \mathbf{y})^T \otimes (\mathcal{W}_{0} \mathbf{x})^T & ((\hat{\mathbf{y}} - \mathbf{y})^T \mathcal{W}_1) \otimes \mathbf{x}^T \end{pmatrix}. \]

**数值角:** 我们回到前一小节的具体例子。这次矩阵 `W0` 和 `W1` 需要计算偏导数。

```py
x = torch.tensor([1.,0.,-1.])
y = torch.tensor([0.,1.])
W0 = torch.tensor([[0.,1.,-1.],[2.,0.,1.]], requires_grad=True)
W1 = torch.tensor([[-1.,0.],[2.,-1.]], requires_grad=True)

z0 = x
z1 = W0 @ z0
z2 = W1 @ z1
f = 0.5 * (torch.linalg.vector_norm(y-z2) ** 2)

print(z0) 
```

```py
tensor([ 1.,  0., -1.]) 
```

```py
print(z1) 
```

```py
tensor([1., 1.], grad_fn=<MvBackward0>) 
```

```py
print(z2) 
```

```py
tensor([-1.,  1.], grad_fn=<MvBackward0>) 
```

```py
print(f) 
```

```py
tensor(0.5000, grad_fn=<MulBackward0>) 
```

我们使用 AD（自动微分）来计算梯度 \(\nabla f(\mathbf{w})\)。

```py
f.backward() 
```

```py
print(W0.grad) 
```

```py
tensor([[ 1.,  0., -1.],
        [ 0.,  0., -0.]]) 
```

```py
print(W1.grad) 
```

```py
tensor([[-1., -1.],
        [-0., -0.]]) 
```

这些是矩阵导数的形式

\[\begin{split} \frac{\partial f}{\partial \mathcal{W}_0} = \begin{pmatrix} \frac{\partial f}{\partial w_0} & \frac{\partial f}{\partial w_1} & \frac{\partial f}{\partial w_2} \\ \frac{\partial f}{\partial w_3} & \frac{\partial f}{\partial w_4} & \frac{\partial f}{\partial w_5} \end{pmatrix} \quad\text{和}\quad \frac{\partial f}{\partial \mathcal{W}_1} = \begin{pmatrix} \frac{\partial f}{\partial w_6} & \frac{\partial f}{\partial w_7} \\ \frac{\partial f}{\partial w_8} & \frac{\partial f}{\partial w_9} \end{pmatrix}. \end{split}\]

我们使用我们的公式来验证它们是否与这些结果匹配。我们需要克罗内克积，在 PyTorch 中通过 `torch.kron` 实现[`torch.kron`](https://pytorch.org/docs/stable/generated/torch.kron.html)。

```py
with torch.no_grad():
    grad_W0 = torch.kron((z2 - y).unsqueeze(0) @ W1, z0.unsqueeze(0))
    grad_W1 = torch.kron((z2 - y).unsqueeze(0), z1.unsqueeze(0))

print(grad_W0) 
```

```py
tensor([[ 1.,  0., -1.,  0.,  0., -0.]]) 
```

```py
print(grad_W1) 
```

```py
tensor([[-1., -1.,  0.,  0.]]) 
```

注意这次这些结果是以向量化的形式（即通过连接行获得）写出的。但它们与 AD 输出相匹配。

\(\unlhd\)

**一般设置** \(\idx{渐进函数}\xdi\) 更一般地，我们有 \(L+2\) 层。输入层是 \(\mathbf{z}_0 := \mathbf{x}\)，我们将其称为层 \(0\)。隐藏层 \(i\)，\(i=1,\ldots,L\)，由一个连续可微的函数 \(\mathbf{z}_i := \bfg_{i-1}(\mathbf{z}_{i-1}, \mathbf{w}_{i-1})\) 定义，这次它接受两个向量值输入：一个来自 \((i-1)\) 层的向量 \(\mathbf{z}_{i-1} \in \mathbb{R}^{n_{i-1}}\) 和一个特定于 \(i\) 层的参数向量 \(\mathbf{w}_{i-1} \in \mathbb{R}^{r_{i-1}}\)

\[ \bfg_{i-1} = (g_{i-1,1},\ldots,g_{i-1,n_{i}}) : \mathbb{R}^{n_{i-1} + r_{i-1}} \to \mathbb{R}^{n_{i}}. \]

\(\bfg_{i-1}\) 的输出 \(\mathbf{z}_i\) 是一个在 \(\mathbb{R}^{n_{i}}\) 中的向量，它作为输入传递给 \((i+1)\) 层。输出层是 \(\mathbf{z}_{L+1} := \bfg_{L}(\mathbf{z}_{L}, \mathbf{w}_{L})\)，我们也将它称为层 \(L+1\)。

对于 \(i = 1,\ldots,L+1\)，令

\[ \overline{\mathbf{w}}^{i-1} = (\mathbf{w}_0,\mathbf{w}_1,\ldots,\mathbf{w}_{i-1}) \in \mathbb{R}^{r_0 + r_1+\cdots+r_{i-1}} \]

是前 \(i\) 层参数的连接（不包括没有参数的输入层）作为一个向量在 \(\mathbb{R}^{r_0+r_1+\cdots+r_{i-1}}\) 中。然后层 \(i\) 作为参数的输出是组合

\[ \mathcal{O}_{i-1}(\overline{\mathbf{w}}^{i-1}) = \bfg_{i-1}(\mathcal{O}_{i-2}(\overline{\mathbf{w}}^{i-2}), \mathbf{w}_{i-1}) = \bfg_{i-1}(\bfg_{i-2}(\cdots \bfg_1(\bfg_0(\mathbf{x},\mathbf{w}_0),\mathbf{w}_1), \cdots, \mathbf{w}_{i-2}), \mathbf{w}_{i-1}) \in \mathbb{R}^{n_{i}}, \]

对于 \(i = 2, \ldots, L+1\)。当 \(i=1\) 时，我们有

\[ \mathcal{O}_{0}(\overline{\mathbf{w}}^{0}) = \bfg_{0}(\mathbf{x}, \mathbf{w}_0). \]

注意到函数 \(\mathcal{O}_{i-1}\) 依赖于输入 \(\mathbf{x}\) —— 在这个设置中我们并不将其视为变量。为了简化符号，我们没有明确表示对 \(\mathbf{x}\) 的依赖。

令 \(\mathbf{w} := \overline{\mathbf{w}}^{L}\)，最终的输出是

\[ \bfh(\mathbf{w}) = \mathcal{O}_{L}(\overline{\mathbf{w}}^{L}). \]

展开组合，这可以写成另一种形式

\[ \bfh(\mathbf{w}) = \bfg_{L}(\bfg_{L-1}(\cdots \bfg_1(\bfg_0(\mathbf{x},\mathbf{w}_0),\mathbf{w}_1), \cdots, \mathbf{w}_{L-1}), \mathbf{w}_{L}). \]

再次，我们没有明确表示对 \(\mathbf{x}\) 的依赖。

作为最后一步，我们有一个损失函数 \(\ell : \mathbb{R}^{n_{L+1}} \to \mathbb{R}\)，它接受最后一层的输出并衡量与给定标签 \(\mathbf{y} \in \Delta_K\) 的拟合度。我们将在下面看到一些示例。最终的函数是

\[ f(\mathbf{w}) = \ell(\bfh(\mathbf{w})) \in \mathbb{R}. \]

我们寻求计算 \(f(\mathbf{w})\) 关于参数 \(\mathbf{w}\) 的梯度，以便应用梯度下降法。

**示例：** **（继续）** 我们回到前一小节中的运行示例。也就是说，\(\bfg_i(\mathbf{z}_i, \mathbf{w}_i) = \mathcal{W}_{i} \mathbf{z}_i\) 其中 \(\mathcal{W}_{i} \in \mathbb{R}^{n_{i+1} \times n_i}\) 的元素被视为参数，我们令 \(\mathbf{w}_i = \mathrm{vec}(\mathcal{W}_{i}^T)\)。还假设 \(\ell : \mathbb{R}^K \to \mathbb{R}\) 定义为

\[ \ell(\hat{\mathbf{y}}) = \frac{1}{2} \|\mathbf{y} - \hat{\mathbf{y}}\|², \]

对于一个固定的、已知的向量 \(\mathbf{y} \in \mathbb{R}^{K}\)。

递归计算 \(f\) 给出

\[\begin{align*} \mathbf{z}_0 &:= \mathbf{x}\\ \mathbf{z}_1 &:= \mathcal{O}_0(\overline{\mathbf{w}}⁰) = \bfg_0(\mathbf{z}_0, \mathbf{w}_0) = \mathcal{W}_{0} \mathbf{z}_0 = \mathcal{W}_{0} \mathbf{x}\\ \mathbf{z}_2 &:= \mathcal{O}_1(\overline{\mathbf{w}}¹) = \bfg_1(\mathbf{z}_1, \mathbf{w}_1) = \mathcal{W}_{1} \mathbf{z}_1 = \mathcal{W}_{1} \mathcal{W}_{0} \mathbf{x}\\ \vdots\\ \mathbf{z}_L &:= \mathcal{O}_{L-1}(\overline{\mathbf{w}}^{L-1}) = \bfg_{L-1}(\mathbf{z}_{L-1}, \mathbf{w}_{L-1}) = \mathcal{W}_{L-1} \mathbf{z}_{L-1} = \mathcal{W}_{L-1} \cdots \mathcal{W}_{1} \mathcal{W}_{0} \mathbf{x}\\ \hat{\mathbf{y}} := \mathbf{z}_{L+1} &:= \mathcal{O}_{L}(\overline{\mathbf{w}}^{L}) = \bfg_{L}(\mathbf{z}_{L}, \mathbf{w}_{L}) = \mathcal{W}_{L} \mathbf{z}_{L} = \mathcal{W}_{L} \mathcal{W}_{L-1} \cdots \mathcal{W}_{1} \mathcal{W}_{0} \mathbf{x}\\ f(\mathbf{x}) &:= \ell(\hat{\mathbf{y}}) = \frac{1}{2}\|\mathbf{y} - \hat{\mathbf{y}}\|² = \frac{1}{2}\left\|\mathbf{y} - \mathcal{W}_{L} \mathcal{W}_{L-1} \cdots \mathcal{W}_{1} \mathcal{W}_{0} \mathbf{x}\right\|². \end{align*}\]

\(\lhd\)

**应用链式法则** 回想一下，链式法则的关键洞察是，为了计算复合函数如 \(\bfh(\mathbf{w})\) 的梯度——无论其多么复杂——只需要分别计算中间函数的雅可比矩阵，然后进行矩阵乘法。在本节中，我们将计算渐进情况下的必要雅可比矩阵。

将基本复合步骤重新写为将方便

\[ \mathcal{O}_{i}(\overline{\mathbf{w}}^{i}) = \bfg_{i}(\mathcal{O}_{i-1}(\overline{\mathbf{w}}^{i-1}), \mathbf{w}_{i}) = \bfg_{i}(\mathcal{I}_{i}(\overline{\mathbf{w}}^{i})) \in \mathbb{R}^{n_{i+1}}, \]

其中层 \(i+1\) 的输入（既包括层特定参数也包括前一层输出）是

\[ \mathcal{I}_{i}(\overline{\mathbf{w}}^{i}) = \left( \mathcal{O}_{i-1}(\overline{\mathbf{w}}^{i-1}), \mathbf{w}_{i} \right) \in \mathbb{R}^{n_{i} + r_{i}}, \]

对于 \(i = 1, \ldots, L\)。当 \(i=0\) 时，我们有

\[ \mathcal{I}_{0}(\overline{\mathbf{w}}^{0}) = \left(\mathbf{z}_0, \mathbf{w}_0 \right) = \left(\mathbf{x}, \mathbf{w}_0 \right). \]

应用链式法则我们得到

\[ J_{\mathcal{O}_{i}}(\overline{\mathbf{w}}^{i}) = J_{\bfg_i}(\mathcal{I}_{i}(\overline{\mathbf{w}}^{i})) \,J_{\mathcal{I}_{i}}(\overline{\mathbf{w}}^{i}). \]

首先，的雅可比矩阵

\[ \mathcal{I}_{i}(\overline{\mathbf{w}}^{i}) = \left( \mathcal{O}_{i-1}(\overline{\mathbf{w}}^{i-1}), \mathbf{w}_i \right) \]

具有简单的分块对角结构

\[\begin{split} J_{\mathcal{I}_{i}}(\overline{\mathbf{w}}^{i}) = \begin{pmatrix} J_{\mathcal{O}_{i-1}}(\overline{\mathbf{w}}^{i-1}) & 0 \\ 0 & I_{r_i \times r_i} \end{pmatrix} \in \mathbb{R}^{(n_{i} + r_{i})\times(r_0 + \cdots + r_i)} \end{split}\]

因为 \(\mathcal{I}_{i}\) 的第一个块组件 \(\mathcal{O}_{i-1}(\overline{\mathbf{w}}^{i-1})\) 不依赖于 \(\mathbf{w}_i\)，而 \(\mathcal{I}_{i}\) 的第二个块组件 \(\mathbf{w}_i\) 不依赖于 \(\overline{\mathbf{w}}^{i-1}\)。观察到一个相当大的矩阵，其列数特别是随着 \(i\) 的增长而增长。最后一个公式适用于 \(i \geq 1\)。当 \(i=0\) 时，我们有 \(\mathcal{I}_{0}(\overline{\mathbf{w}}^{0}) = \left(\mathbf{x}, \mathbf{w}_0\right)\)，所以

\[\begin{split} J_{\mathcal{I}_{0}}(\overline{\mathbf{w}}^{0}) = \begin{pmatrix} \mathbf{0}_{d \times r_0} \\ I_{r_0 \times r_0} \end{pmatrix} \in \mathbb{R}^{(d+ r_0) \times r_0}. \end{split}\]

我们同样将 \(\bfg_i(\mathbf{z}_i, \mathbf{w}_i)\) 的雅可比矩阵进行划分，即，我们将它划分为对应于对 \(\mathbf{z}_{i}\) 的偏导数（相应的块用 \(A_i\) 表示）和对应于 \(\mathbf{w}_i\) 的偏导数（相应的块用 \(B_i\) 表示）

\[ J_{\bfg_i}(\mathbf{z}_i, \mathbf{w}_i) = \begin{pmatrix} A_i & B_i \end{pmatrix} \in \mathbb{R}^{n_{i+1} \times (n_i + r_i)}, \]

在 \((\mathbf{z}_i, \mathbf{w}_i) = \mathcal{I}_{i}(\overline{\mathbf{w}}^{i}) = (\mathcal{O}_{i-1}(\overline{\mathbf{w}}^{i-1}), \mathbf{w}_i)\) 处评估。注意，\(A_i\) 和 \(B_i\) 取决于函数 \(\bfg_i\) 的细节，该函数通常相对简单。我们将在下一节给出示例。

将上述内容代入，我们得到

\[\begin{split} J_{\mathcal{O}_{i}}(\overline{\mathbf{w}}^{i}) = \begin{pmatrix} A_i & B_i \end{pmatrix} \,\begin{pmatrix} J_{\mathcal{O}_{i-1}}(\overline{\mathbf{w}}^{i-1}) & 0 \\ 0 & I_{r_i \times r_i} \end{pmatrix}. \end{split}\]

这导致递归

\[ J_{\mathcal{O}_{i}}(\overline{\mathbf{w}}^{i}) = \begin{pmatrix} A_i \, J_{\mathcal{O}_{i-1}}(\overline{\mathbf{w}}^{i-1}) & B_i \end{pmatrix} \in \mathbb{R}^{n_{i+1}\times(r_0 + \cdots + r_i)} \]

从中可以计算出 \(\mathbf{h}(\mathbf{w})\) 的雅可比矩阵。像 \(J_{\mathcal{I}_{i}}\) 一样，\(J_{\mathcal{O}_{i}}\) 也是一个大矩阵。我们将这个矩阵方程称为 *基本递归*。

基本情况 \(i=0\) 是

\[\begin{split} J_{\mathcal{O}_{0}}(\overline{\mathbf{w}}^{0}) = \begin{pmatrix} A_0 & B_0 \end{pmatrix}\begin{pmatrix} \mathbf{0}_{d \times r_0} \\ I_{r_0 \times r_0} \end{pmatrix} = B_0. \end{split}\]

最后，再次使用**链式法则**。

\[\begin{align*} \nabla {f(\mathbf{w})} &= J_{f}(\mathbf{w})^T\\ &= [J_{\ell}(\bfh(\mathbf{w})) \,J_{\bfh}(\mathbf{w})]^T\\ &= J_{\bfh}(\mathbf{w})^T \,\nabla {\ell}(\bfh(\mathbf{w}))\\ &= J_{\mathcal{O}_{L}}(\overline{\mathbf{w}}^{L})^T \,\nabla {\ell}(\mathcal{O}_{L}(\overline{\mathbf{w}}^{L})). \end{align*}\]

矩阵 \(J_{\mathcal{O}_{L}}(\overline{\mathbf{w}}^{L})\) 是使用上述递归方法计算的，而 \(\nabla {\ell}\) 取决于函数 \(\ell\)。

**反向传播** \(\idx{backpropagation}\xdi\) 我们利用基本的递归方法来计算 \(\bfh\) 的梯度。正如我们所见，有两种方法可以做到这一点。直接应用递归是其中之一，但它需要许多矩阵-矩阵乘法。最初的几步是

\[ J_{\mathcal{O}_{0}}(\overline{\mathbf{w}}^{0}) = B_0, \]\[ J_{\mathcal{O}_{1}}(\overline{\mathbf{w}}^{1}) = \begin{pmatrix} A_1 J_{\mathcal{O}_{0}}(\overline{\mathbf{w}}^{0}) & B_1 \end{pmatrix} \]\[ J_{\mathcal{O}_{2}}(\overline{\mathbf{w}}^{2}) = \begin{pmatrix} A_2 \, J_{\mathcal{O}_{1}}(\overline{\mathbf{w}}^{1}) & B_2 \end{pmatrix}, \]

等等。

相反，正如对输入 \(\mathbf{x}\) 求导的情况一样，也可以反向运行递归。后一种方法可能要快得多，因为，正如我们下面将要详细说明的，它只涉及矩阵-向量乘积。从末尾开始，即从以下方程开始：

\[ \nabla {f}(\mathbf{w}) = J_{\bfh}(\mathbf{w})^T \,\nabla {\ell}(\bfh(\mathbf{w})). \]

注意到 \(\nabla {\ell}(\bfh(\mathbf{w}))\) 是一个向量——不是一个矩阵。然后使用上述递归方法扩展矩阵 \(J_{\bfh}(\mathbf{w})\)。

\[\begin{align*} \nabla {f}(\mathbf{w}) &= J_{\bfh}(\mathbf{w})^T \,\nabla {\ell}(\bfh(\mathbf{w}))\\ &= J_{\mathcal{O}_{L}}(\overline{\mathbf{w}}^{L})^T \,\nabla {\ell}(\bfh(\mathbf{w}))\\ &= \begin{pmatrix} A_L \, J_{\mathcal{O}_{L-1}}(\overline{\mathbf{w}}^{L-1}) & B_L \end{pmatrix}^T \,\nabla {\ell}(\bfh(\mathbf{w}))\\ &= \begin{pmatrix} J_{\mathcal{O}_{L-1}}(\overline{\mathbf{w}}^{L-1})^T A_L^T \\ B_L^T \end{pmatrix} \,\nabla {\ell}(\bfh(\mathbf{w}))\\ &= \begin{pmatrix} J_{\mathcal{O}_{L-1}}(\overline{\mathbf{w}}^{L-1})^T \left\{ A_L^T \,\nabla {\ell}(\bfh(\mathbf{w}))\right\} \\ B_L^T \,\nabla {\ell}(\bfh(\mathbf{w})) \end{pmatrix}. \end{align*}\]

关键在于，两个表达式 \(A_L^T \,\nabla {\ell}(\bfh(\mathbf{w}))\) 和 \(B_L^T \,\nabla {\ell}(\bfh(\mathbf{w}))\) 都是**矩阵-向量乘积**。这种模式在递归的下一个层次上持续存在。请注意，这假设我们首先已经预计算了 \(\bfh(\mathbf{w})\)。

在下一个层次，我们使用基本的递归方法扩展矩阵 \(J_{\mathcal{O}_{L-1}}(\overline{\mathbf{w}}^{L-1})^T\)。

\[\begin{align*} \nabla {f}(\mathbf{w}) &= \begin{pmatrix} J_{\mathcal{O}_{L-1}}(\overline{\mathbf{w}}^{L-1})^T \left\{ A_L^T \,\nabla {\ell}(\bfh(\mathbf{w}))\right\} \\ B_L^T \,\nabla {\ell}(\bfh(\mathbf{w})) \end{pmatrix}\\ &= \begin{pmatrix} \begin{pmatrix} A_{L-1} \, J_{\mathcal{O}_{L-2}}(\overline{\mathbf{w}}^{L-2}) & B_{L-1} \end{pmatrix}^T \left\{ A_L^T \,\nabla {\ell}(\bfh(\mathbf{w}))\right\} \\ B_L^T \,\nabla {\ell}(\bfh(\mathbf{w})) \end{pmatrix}\\ &= \begin{pmatrix} \begin{pmatrix} J_{\mathcal{O}_{L-2}}(\overline{\mathbf{w}}^{L-2})\,A_{L-1}^T \\ B_{L-1}^T \end{pmatrix} \left\{ A_L^T \,\nabla {\ell}(\bfh(\mathbf{w}))\right\} \\ B_L^T \,\nabla {\ell}(\bfh(\mathbf{w})) \end{pmatrix}\\ &= \begin{pmatrix} J_{\mathcal{O}_{L-2}}(\overline{\mathbf{w}}^{L-2})\left\{A_{L-1}^T \left\{ A_L^T \,\nabla {\ell}(\bfh(\mathbf{w}))\right\}\right\} \\ B_{L-1}^T \left\{ A_L^T \,\nabla {\ell}(\bfh(\mathbf{w}))\right\}\\ B_L^T \,\nabla {\ell}(\bfh(\mathbf{w})) \end{pmatrix}. \end{align*}\]

通过归纳法继续推导给出了 \(f\) 的梯度的另一种公式。

事实上，下一级给出

\[\begin{align*} \nabla {f}(\mathbf{w}) &= \begin{pmatrix} J_{\mathcal{O}_{L-3}}(\overline{\mathbf{w}}^{L-3})\left\{A_{L-2}^T \left\{A_{L-1}^T \left\{ A_L^T \,\nabla {\ell}(\bfh(\mathbf{w}))\right\}\right\}\right\} \\ B_{L-2}^T \left\{A_{L-1}^T \left\{ A_L^T \,\nabla {\ell}(\bfh(\mathbf{w}))\right\}\right\} \\ B_{L-1}^T \left\{ A_L^T \,\nabla {\ell}(\bfh(\mathbf{w}))\right\}\\ B_L^T \,\nabla {\ell}(\bfh(\mathbf{w})) \end{pmatrix}. \end{align*}\]

等等。观察发现，实际上我们并不需要计算大的矩阵 \(J_{\mathcal{O}_{i}}\) ——只需要计算向量序列 \(B_L^T \,\nabla {\ell}(\bfh(\mathbf{w}))\), \(B_{L-1}^T \left\{ A_L^T \,\nabla {\ell}(\bfh(\mathbf{w}))\right\}\), \(B_{L-2}^T \left\{A_{L-1}^T \left\{ A_L^T \,\nabla {\ell}(\bfh(\mathbf{w}))\right\}\right\}, \)等等。

这些公式可能看起来有些繁琐，但它们具有直观的形式。矩阵 \(A_i\) 是雅可比矩阵 \(J_{\bfg_i}\) 的子矩阵，仅对应于对 \(\mathbf{z}_i\) 的偏导数，即来自前一层的输入。矩阵 \(B_i\) 是雅可比矩阵 \(J_{\bfg_i}\) 的子矩阵，仅对应于对 \(\mathbf{w}_i\) 的偏导数，即特定层的参数。为了计算与 \((i+1)\) 层的参数 \(\mathbf{w}_i\) 对应的 \(\nabla f\) 的子向量，我们从最后一个开始，通过乘以相应的 \(A_j^T\) 对前一层的输入进行反复求导，直到达到 \(i+1\) 层，此时我们通过乘以 \(B_i^T\) 对特定层的参数进行偏导。这个过程在这里停止，因为它之前的层不依赖于 \(\mathbf{w}_i\)，因此其完全影响 \(f\) 已经被考虑在内。

换句话说，我们需要计算

\[ \mathbf{p}_{L} := A_L^T \,\nabla {\ell}(\bfh(\mathbf{w})), \]

和

\[ \mathbf{q}_{L} := B_L^T \,\nabla {\ell}(\bfh(\mathbf{w})), \]

然后

\[ \mathbf{p}_{L-1} := A_{L-1}^T \mathbf{p}_{L} = A_{L-1}^T \left\{ A_L^T \,\nabla {\ell}(\bfh(\mathbf{w}))\right\} \]

和

\[ \mathbf{q}_{L-1} := B_{L-1}^T \mathbf{p}_{L} = B_{L-1}^T \left\{ A_L^T \,\nabla {\ell}(\bfh(\mathbf{w}))\right\}, \]

然后

\[ \mathbf{p}_{L-2} := A_{L-2}^T \mathbf{p}_{L-1} = A_{L-2}^T \left\{A_{L-1}^T \left\{ A_L^T \,\nabla {\ell}(\bfh(\mathbf{w}))\right\}\right\} \]

和

\[ \mathbf{q}_{L-2} := B_{L-2}^T \mathbf{p}_{L-1} = B_{L-2}^T \left\{A_{L-1}^T \left\{ A_L^T \,\nabla {\ell}(\bfh(\mathbf{w}))\right\} \right\}, \]

以此类推。这些 \(\mathbf{p}_i\) 被称为伴随项；它们对应于 \(f\) 对 \(\mathbf{z}_i\) 的偏导数向量。

需要注意的一个细节是。矩阵 \(A_i, B_i\) 依赖于层 \(i-1\) 的输出。为了计算它们，我们首先进行正向传播，即，我们令 \(\mathbf{z}_0 = \mathbf{x}\) 然后

\[ \mathbf{z}_1 = \mathcal{O}_{0}(\overline{\mathbf{w}}^{0}) = \bfg_0(\mathbf{z}_0, \mathbf{w}_0), \]\[ \mathbf{z}_2 = \mathcal{O}_{1}(\overline{\mathbf{w}}^{1}) = \bfg_1(\mathcal{O}_{0}(\overline{\mathbf{w}}^{0}), \mathbf{w}_1) = \bfg_1(\mathbf{z}_1, \mathbf{w}_1), \]

以此类推。在正向传播过程中，我们也会沿途计算 \(A_i, B_i\)。

我们现在给出完整的算法，它涉及两个过程。在正向传播或正向传播步骤中，我们计算以下内容。

*初始化:*

\[\mathbf{z}_0 := \mathbf{x}\]

*正向层循环:* 对于 \(i = 0, 1,\ldots,L\)，

\[\begin{align*} \mathbf{z}_{i+1} &:= \bfg_i(\mathbf{z}_i, \mathbf{w}_i)\\ \begin{pmatrix} A_i & B_i \end{pmatrix} &:= J_{\bfg_i}(\mathbf{z}_i, \mathbf{w}_i) \end{align*}\]

*损失:*

\[\begin{align*} z_{L+2} &:= \ell(\mathbf{z}_{L+1})\\ \mathbf{p}_{L+1} &:= \nabla {\ell}(\mathbf{z}_{L+1}). \end{align*}\]

在反向传播或反向传播步骤中，我们计算以下内容。

*反向层循环:* 对于 \(i = L,\ldots,1, 0\)，

\[\begin{align*} \mathbf{p}_{i} &:= A_i^T \mathbf{p}_{i+1}\\ \mathbf{q}_{i} &:= B_i^T \mathbf{p}_{i+1} \end{align*}\]

*输出:*

\[ \nabla f(\mathbf{w}) = (\mathbf{q}_0,\mathbf{q}_1,\ldots,\mathbf{q}_L). \]

注意，实际上我们不需要计算 \(A_0\) 和 \(\mathbf{p}_0\)。

**示例:** **(继续)** 我们将算法应用于我们的运行示例。从前面的计算中，对于 \(i = 0, 1,\ldots,L\)，雅可比矩阵是

\[\begin{align*} J_{\bfg_i}(\mathbf{z}_i, \mathbf{w}_i) &= \begin{pmatrix} \mathbb{A}_{n_{i+1}}[\mathbf{w}_i] & \mathbb{B}_{n_{i+1}}[\mathbf{z}_i] \end{pmatrix}\\ &= \begin{pmatrix} \mathcal{W}_i & I_{n_{i+1} \times n_{i+1}} \otimes \mathbf{z}_i^T \end{pmatrix}\\ &=: \begin{pmatrix} A_i & B_i \end{pmatrix} \end{align*}\]

和

\[ J_{\ell}(\hat{\mathbf{y}}) = (\hat{\mathbf{y}} - \mathbf{y})^T. \]

利用**克罗内克积的性质**，我们得到

\[ \mathbf{p}_{L} := A_L^T \,\nabla {\ell}(\bfh(\mathbf{w})) = \mathcal{W}_L^T (\hat{\mathbf{y}} - \mathbf{y}) \]\[\begin{align*} \mathbf{q}_{L} &:= B_L^T \,\nabla {\ell}(\bfh(\mathbf{w})) = (I_{n_{L+1} \times n_{L+1}} \otimes \mathbf{z}_L^T)^T (\hat{\mathbf{y}} - \mathbf{y}) = (\hat{\mathbf{y}} - \mathbf{y}) \otimes \mathbf{z}_L\\ &= (\hat{\mathbf{y}} - \mathbf{y}) \otimes \mathcal{W}_{L-1} \cdots \mathcal{W}_{1} \mathcal{W}_{0} \mathbf{x} \end{align*}\]\[ \mathbf{p}_{L-1} := A_{L-1}^T \mathbf{p}_{L} = \mathcal{W}_{L-1}^T \mathcal{W}_L^T (\hat{\mathbf{y}} - \mathbf{y}) \]\[\begin{align*} \mathbf{q}_{L-1} &:= B_{L-1}^T \mathbf{p}_{L} = (I_{n_{L} \times n_{L}} \otimes \mathbf{z}_{L-1}^T)^T \mathcal{W}_L^T (\hat{\mathbf{y}} - \mathbf{y}) = \mathcal{W}_L^T (\hat{\mathbf{y}} - \mathbf{y}) \otimes \mathbf{z}_{L-1}\\ &= \mathcal{W}_L^T (\hat{\mathbf{y}} - \mathbf{y}) \otimes \mathcal{W}_{L-2} \cdots \mathcal{W}_{1} \mathcal{W}_{0} \mathbf{x} \end{align*}\]\[ \mathbf{p}_{L-2} := A_{L-2}^T \mathbf{p}_{L-1} = \mathcal{W}_{L-2}^T \mathcal{W}_{L-1}^T \mathcal{W}_L^T (\hat{\mathbf{y}} - \mathbf{y}) \]\[\begin{align*} \mathbf{q}_{L-2} &:= B_{L-2}^T \mathbf{p}_{L-1} = (I_{n_{L-1} \times n_{L-1}} \otimes \mathbf{z}_{L-2}^T)^T \mathcal{W}_{L-1}^T \mathcal{W}_L^T (\hat{\mathbf{y}} - \mathbf{y}) = \mathcal{W}_{L-1}^T \mathcal{W}_L^T (\hat{\mathbf{y}} - \mathbf{y}) \otimes \mathbf{z}_{L-2}\\ &= \mathcal{W}_{L-1}^T \mathcal{W}_L^T (\hat{\mathbf{y}} - \mathbf{y}) \otimes \mathcal{W}_{L-3} \cdots \mathcal{W}_{1} \mathcal{W}_{0} \mathbf{x} \end{align*}\]

等等。按照模式，最后一步是

\[ \mathbf{p}_1 := \mathcal{W}_{1}^T \cdots \mathcal{W}_{L-1}^T \mathcal{W}_L^T (\hat{\mathbf{y}} - \mathbf{y}) \]\[ \mathbf{q}_0 := B_{0}^T \mathbf{p}_{1} = \mathcal{W}_{1}^T \cdots \mathcal{W}_{L-1}^T \mathcal{W}_L^T (\hat{\mathbf{y}} - \mathbf{y}) \otimes \mathbf{x}. \]

这些计算与我们之前推导的 \(L=1\) 的情况一致（检查一下！）。\(\lhd\)

**CHAT & LEARN** 反向传播的效率对深度学习的成功至关重要。向您最喜欢的 AI 聊天机器人询问反向传播的历史及其在现代深度学习发展中的作用。 \(\ddagger\)

***自我评估测验*** *(在 Claude, Gemini 和 ChatGPT 的帮助下)*

**1** 在反向传播算法中，‘前向传递’计算了什么？

a) 每一层 \(i\) 的伴随 \(\mathbf{p}_i\)。

b) 每一层 \(i\) 参数的梯度 \(\mathbf{q}_i\)。

c) 每一层 \(i\) 的函数值 \(\mathbf{z}_i\) 和雅可比矩阵 \(A_i, B_i\)。

d) 关于所有参数的最终梯度 \(\nabla f(\mathbf{w})\)。

**2** 反向传播算法中‘反向传递’的目的是什么？

a) 从输入 \(\mathbf{x}\) 计算每一层 \(i\) 的函数值 \(\mathbf{z}_i\)。

b) 使用基本递归计算每一层 \(i\) 的雅可比矩阵 \(A_i, B_i\)。

c) 使用基本递归计算每层 \(i\) 的伴随矩阵 \(\mathbf{p}_i\) 和梯度 \(\mathbf{q}_i\)。

d) 为了计算渐进函数的最终输出 \(\ell(\mathbf{z}_{L+1})\)。

**3** 后向传播算法在层数 \(L\) 和矩阵维度 \(m\) 方面的计算复杂度是什么？

a) \(\approx Lm\)

b) \(\approx Lm²\)

c) \(\approx Lm²d\)

d) \(\approx Lm³d\)

**4** 在渐进函数的背景下，矩阵 \(A_i\) 和 \(B_i\) 的意义是什么？

a) 它们分别代表了层函数相对于输入和参数的雅可比矩阵。

b) 它们是前向传播过程中计算的中间值。

c) 它们是在后向传播算法中使用的伴随矩阵。

d) 它们是每层的参数矩阵。

**5** 在渐进函数的背景下，以下哪个选项最能描述向量 \(\mathbf{w}_i\) 的作用？

a) 第 \(i\) 层的输入。

b) 第 \(i\) 层的输出。

c) 第 \(i\) 层特有的参数。

d) 将所有层到 \(i\) 的参数连接起来。

1 的答案：c. 理由：本节介绍了前向传播步骤，该步骤计算“以下内容：初始化：\(\mathbf{z}_0 := \mathbf{x}\) 前向层循环：对于 \(i=0,1,\dots,L\)，\(\mathbf{z}_{i+1} := \mathbf{g}_i(\mathbf{z}_i, \mathbf{w}_i)\) \((A_i,B_i) := J_{\mathbf{g}_i}(\mathbf{z}_i, \mathbf{w}_i)\) 损失：\(\mathbf{z}_{L+2} := \ell(\mathbf{z}_{L+1})\)”

2 的答案：c. 理由：反向传播过程描述如下：“反向层循环：对于 \(i=L,\dots,1,0\)，\(\mathbf{p}_i := A_i^T \mathbf{p}_{i+1}\) \(\mathbf{q}_i := B_i^T \mathbf{p}_{i+1}\) 输出：\(\nabla f(\mathbf{w}) = (\mathbf{q}_0, \mathbf{q}_1, \dots, \mathbf{q}_L)\)。”

3 的答案：b. 理由：文本推导出反向模式的操作数大约是 \(2Lm²\)，并指出“这大约是 \(2Lm²\) – 这可以比 \(2Lm²d\) 小得多！”

4 的答案：a. 理由：文本将 \(A_i\) 和 \(B_i\) 定义为雅可比矩阵 \(J_{\mathbf{g}_i}(\mathbf{z}_i, \mathbf{w}_i)\) 对应于相对于 \(\mathbf{z}_i\) 和 \(\mathbf{w}_i\) 的偏导数的块。

5 的答案：c. 理由：文本解释道：“在机器学习背景下，每个“层”\(\mathbf{g}_i\)都有参数（在我们的例子中，是\(\mathcal{W}_i\)的条目），我们试图优化这些参数。”

## 8.3.1\. 前向传播与后向传播#

我们从一个固定参数的例子开始，以说明问题。假设 \(f : \mathbb{R}^d \to \mathbb{R}\) 可以表示为 \(L+1\) 个向量值函数 \(\bfg_i : \mathbb{R}^{n_i} \to \mathbb{R}^{n_{i+1}}\) 和一个实值函数 \(\ell : \mathbb{R}^{n_{L+1}} \to \mathbb{R}\) 的组合，如下所示

\[ f(\mathbf{x}) = \ell \circ \bfg_{L} \circ \bfg_{L-1} \circ \cdots \circ \bfg_1 \circ \bfg_0(\mathbf{x}) = \ell(\bfg_{L}(\bfg_{L-1}(\cdots \bfg_1(\bfg_0(\mathbf{x}))\cdots))). \]

这里 \(n_0 = d\) 是输入维度。我们还让 \(n_{L+1} = K\) 成为输出维度。考虑

\[ h(\mathbf{x}) = \bfg_{L}(\bfg_{L-1}(\cdots \bfg_1(\bfg_0(\mathbf{x}))\cdots)) \]

作为预测函数（即回归或分类函数），并将 \(\ell\) 视为损失函数。

首先观察函数 \(f\) 本身可以通过以下方式递归地从内部开始计算：

\[\begin{align*} \mathbf{z}_0 &:= \mathbf{x}\\ \mathbf{z}_1 &:= \bfg_0(\mathbf{z}_0)\\ \mathbf{z}_2 &:= \bfg_1(\mathbf{z}_1)\\ \vdots\\ \mathbf{z}_L &:= \bfg_{L-1}(\mathbf{z}_{L-1})\\ \hat{\mathbf{y}} := \mathbf{z}_{L+1} &:= \bfg_{L}(\mathbf{z}_{L})\\ f(\mathbf{x}) &:= \ell(\hat{\mathbf{y}}). \end{align*}\]

预测神经网络（我们的主要应用兴趣）的设置，我们将 \(\mathbf{z}_0 = \mathbf{x}\) 称为“输入层”，\(\hat{\mathbf{y}} = \mathbf{z}_{L+1} = \bfg_{L}(\mathbf{z}_{L})\) 称为“输出层”，以及 \(\mathbf{z}_{1} = \bfg_0(\mathbf{z}_0), \ldots, \mathbf{z}_L = \bfg_{L-1}(\mathbf{z}_{L-1})\) 称为“隐藏层”。特别是，\(L\) 是隐藏层的数量。

**示例：** 我们将在本小节中始终使用以下运行示例。我们假设每个 \(\bfg_i\) 是一个线性映射，即 \(\bfg_i(\mathbf{z}_i) = \mathcal{W}_{i} \mathbf{z}_i\)，其中 \(\mathcal{W}_{i} \in \mathbb{R}^{n_{i+1} \times n_i}\) 是一个固定、已知的矩阵。还假设 \(\ell : \mathbb{R}^K \to \mathbb{R}\) 被定义为

\[ \ell(\hat{\mathbf{y}}) = \frac{1}{2} \|\mathbf{y} - \hat{\mathbf{y}}\|², \]

对于一个固定的、已知的向量 \(\mathbf{y} \in \mathbb{R}^{K}\)。

递归地计算 \(f\)，从内部开始，如上所述，给出以下结果

\[\begin{align*} \mathbf{z}_0 &:= \mathbf{x}\\ \mathbf{z}_1 &:= \mathcal{W}_{0} \mathbf{z}_0 = \mathcal{W}_{0} \mathbf{x}\\ \mathbf{z}_2 &:= \mathcal{W}_{1} \mathbf{z}_1 = \mathcal{W}_{1} \mathcal{W}_{0} \mathbf{x}\\ \vdots\\ \mathbf{z}_L &:= \mathcal{W}_{L-1} \mathbf{z}_{L-1} = \mathcal{W}_{L-1} \cdots \mathcal{W}_{1} \mathcal{W}_{0} \mathbf{x}\\ \hat{\mathbf{y}} := \mathbf{z}_{L+1} &:= \mathcal{W}_{L} \mathbf{z}_{L} = \mathcal{W}_{L} \mathcal{W}_{L-1} \cdots \mathcal{W}_{1} \mathcal{W}_{0} \mathbf{x}\\ f(\mathbf{x}) &:= \ell(\hat{\mathbf{y}}) = \frac{1}{2}\|\mathbf{y} - \hat{\mathbf{y}}\|² = \frac{1}{2}\left\|\mathbf{y} - \mathcal{W}_{L} \mathcal{W}_{L-1} \cdots \mathcal{W}_{1} \mathcal{W}_{0} \mathbf{x}\right\|². \end{align*}\]

本质上，我们是在比较基于输入 \(\mathbf{x}\) 的预测 \(\mathcal{W}_{L} \mathcal{W}_{L-1} \cdots \mathcal{W}_{1} \mathcal{W}_{0} \mathbf{x}\) 与观察到的结果 \(\mathbf{y}\)。

在本节中，我们将探讨如何计算相对于 \(\mathbf{x}\) 的梯度。（实际上，我们更感兴趣的是计算相对于参数的梯度，即矩阵 \(\mathcal{W}_{0}, \ldots, \mathcal{W}_{L}\) 的元素，我们将在本节的后面部分回到这个任务。我们还将对更复杂的——特别是非线性的——预测函数感兴趣。）\(\lhd\)

**数值角落：** 为了使事情更加具体，我们考虑一个特定的例子。我们将使用 \[`torch.linalg.vector_norm`](https://pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html) 在 PyTorch 中计算欧几里得范数。假设 \(d=3\)，\(L=1\)，\(n_1 = 2\)，和 \(K = 2\)，我们有以下选择：

```py
x = torch.tensor([1.,0.,-1.], requires_grad=True)
y = torch.tensor([0.,1.])
W0 = torch.tensor([[0.,1.,-1.],[2.,0.,1.]])
W1 = torch.tensor([[-1.,0.],[2.,-1.]])

z0 = x
z1 = W0 @ z0
z2 = W1 @ z1
f = 0.5 * (torch.linalg.vector_norm(y-z2) ** 2)

print(z0) 
```

```py
tensor([ 1.,  0., -1.], requires_grad=True) 
```

```py
print(z1) 
```

```py
tensor([1., 1.], grad_fn=<MvBackward0>) 
```

```py
print(z2) 
```

```py
tensor([-1.,  1.], grad_fn=<MvBackward0>) 
```

```py
print(f) 
```

```py
tensor(0.5000, grad_fn=<MulBackward0>) 
```

\(\unlhd\)

**正向模式** \(\idx{forward mode}\xdi\) 我们准备应用**链式法则**

\[ \nabla f(\mathbf{x})^T = J_{f}(\mathbf{x}) = J_{\ell}(\mathbf{z}_{L+1}) J_{\bfg_L}(\mathbf{z}_L) J_{\bfg_{L-1}}(\mathbf{z}_{L-1}) \cdots J_{\bfg_1}(\mathbf{z}_1) J_{\bfg_0}(\mathbf{x}) \]

其中，\(\mathbf{z}_{i}\) 如上所述，我们使用了 \(\mathbf{z}_0 = \mathbf{x}\)。这里的矩阵乘积是有定义的。确实，\(J_{g_i}(\mathbf{z}_i)\) 的大小是 \(n_{i+1} \times n_{i}\)（即输出数量乘以输入数量），而 \(J_{g_{i-1}}(\mathbf{z}_{i-1})\) 的大小是 \(n_{i} \times n_{i-1}\)——因此维度是兼容的。

因此，像对 \(f\) 本身那样，我们可以递归地计算 \(\nabla f(\mathbf{x})^T\)。实际上，我们可以同时计算这两个。这被称为正向模式：

\[\begin{align*} \mathbf{z}_0 &:= \mathbf{x}\\ \mathbf{z}_1 &:= \bfg_0(\mathbf{z}_0), \quad F_0 := J_{\bfg_0}(\mathbf{z}_0)\\ \mathbf{z}_2 &:= \bfg_1(\mathbf{z}_1), \quad F_1 := J_{\bfg_1}(\mathbf{z}_1)\, F_0\\ \vdots\\ \mathbf{z}_L &:= \bfg_{L-1}(\mathbf{z}_{L-1}), \quad F_{L-1} := J_{\bfg_{L-1}}(\mathbf{z}_{L-1})\, F_{L-2}\\ \hat{\mathbf{y}} := \mathbf{z}_{L+1} &:= \bfg_{L}(\mathbf{z}_{L}), \quad F_{L} := J_{\bfg_{L}}(\mathbf{z}_{L})\, F_{L-1}\\ f(\mathbf{x}) &:= \ell(\hat{\mathbf{y}}), \quad \nabla f(\mathbf{x})^T := J_{\ell}(\hat{\mathbf{y}}) F_L. \end{align*}\]

**示例：** **（继续）** 我们将此过程应用于运行示例。线性映射 \(\bfg_i(\mathbf{z}_i) = \mathcal{W}_{i} \mathbf{z}_i\) 的雅可比矩阵是矩阵 \(\mathcal{W}_{i}\)，正如我们在之前的例子中所看到的。也就是说，对于任何 \(\mathbf{z}_i\)，\(J_{\bfg_i}(\mathbf{z}_i) = \mathcal{W}_{i}\)。为了计算 \(\ell\) 的雅可比矩阵，我们将它重写为一个二次函数

\[\begin{align*} \ell(\hat{\mathbf{y}}) &= \frac{1}{2} \|\mathbf{y} - \hat{\mathbf{y}}\|²\\ &= \frac{1}{2} \mathbf{y}^T\mathbf{y} - \frac{1}{2} 2 \mathbf{y}^T\hat{\mathbf{y}} + \frac{1}{2}\hat{\mathbf{y}}^T \hat{\mathbf{y}}\\ &= \frac{1}{2} \hat{\mathbf{y}}^T I_{n_{L+1} \times n_{L+1}}\hat{\mathbf{y}} + (-\mathbf{y})^T\hat{\mathbf{y}} + \frac{1}{2} \mathbf{y}^T\mathbf{y}. \end{align*}\]

从之前的例子中，

\[ J_\ell(\hat{\mathbf{y}})^T = \nabla \ell(\hat{\mathbf{y}}) = \frac{1}{2}\left[I_{n_{L+1} \times n_{L+1}} + I_{n_{L+1} \times n_{L+1}}^T\right]\, \hat{\mathbf{y}} + (-\mathbf{y}) = \hat{\mathbf{y}} - \mathbf{y}. \]

将所有这些放在一起，我们得到

\[\begin{align*} F_0 &:= J_{\bfg_0}(\mathbf{z}_0) = \mathcal{W}_{0}\\ F_1 &:= J_{\bfg_1}(\mathbf{z}_1)\, F_0 = \mathcal{W}_{1} F_0 = \mathcal{W}_{1} \mathcal{W}_{0}\\ \vdots\\ F_{L-1} &:= J_{\bfg_{L-1}}(\mathbf{z}_{L-1})\, F_{L-2} = \mathcal{W}_{L-1} F_{L-2}= \mathcal{W}_{L-1} \cdots \mathcal{W}_{1} \mathcal{W}_{0}\\ F_{L} &:= J_{\bfg_{L}}(\mathbf{z}_{L})\, F_{L-1} = \mathcal{W}_{L} F_{L-1} = \mathcal{W}_{L} \mathcal{W}_{L-1} \cdots \mathcal{W}_{1} \mathcal{W}_{0}\\ \nabla f(\mathbf{x})^T &:= J_{\ell}(\hat{\mathbf{y}}) F_L = (\hat{\mathbf{y}} - \mathbf{y})^T F_L = (\hat{\mathbf{y}} - \mathbf{y})^T \mathcal{W}_{L} \mathcal{W}_{L-1} \cdots \mathcal{W}_{1} \mathcal{W}_{0}\\ &= (\mathcal{W}_{L} \mathcal{W}_{L-1} \cdots \mathcal{W}_{1} \mathcal{W}_{0} \mathbf{x} - \mathbf{y})^T \mathcal{W}_{L} \mathcal{W}_{L-1} \cdots \mathcal{W}_{1} \mathcal{W}_{0}. \end{align*}\]

\(\lhd\)

**数值角落:** 我们回到我们的具体例子。在 PyTorch 中使用 `.T` 将列向量转换为行向量会引发错误，因为它仅适用于 2D 张量。相反，可以使用 `torch.unsqueeze`。下面，`(z2 - y).unsqueeze(0)` 向 `z2 - y` 添加一个维度，使其成为一个形状为 \((1, 2)\) 的 2D 张量。

```py
with torch.no_grad():
    F0 = W0
    F1 = W1 @ F0
    grad_f = (z2 - y).unsqueeze(0) @ F1

print(F0) 
```

```py
tensor([[ 0.,  1., -1.],
        [ 2.,  0.,  1.]]) 
```

```py
print(F1) 
```

```py
tensor([[ 0., -1.,  1.],
        [-2.,  2., -3.]]) 
```

```py
print(grad_f) 
```

```py
tensor([[ 0.,  1., -1.]]) 
```

我们可以使用 AD 来验证我们得到相同的结果。

```py
f.backward()
print(x.grad) 
```

```py
tensor([ 0.,  1., -1.]) 
```

\(\unlhd\)

**知识检查:** 通过对以下表达式求梯度来直接获得最后一个表达式

\[ f(\mathbf{x}) = \frac{1}{2}\|\mathbf{y} - \mathcal{W}_{L} \mathcal{W}_{L-1} \cdots \mathcal{W}_{1} \mathcal{W}_{0} \mathbf{x}\|². \]

\(\checkmark\)

**反向模式** \(\idx{reverse mode}\xdi\) 我们刚才描述的对应于在 *链式法则* 公式中执行矩阵乘法

\[ \nabla f(\mathbf{x})^T = J_{f}(\mathbf{x}) = J_{\ell}(\hat{\mathbf{y}}) J_{\bfg_L}(\mathbf{z}_L) J_{\bfg_{L-1}}(\mathbf{z}_{L-1}) \cdots J_{\bfg_1}(\mathbf{z}_1) J_{\bfg_0}(\mathbf{x}). \]

*从右到左*。但还有另一种方法来做这件事——从左到右！这被称为反向模式。首先，我们像之前一样递归地计算 \(f\)，特别是我们计算 \(\mathbf{z}_i\)，然后反向计算

\[\begin{align*} G_{L+1} &:= J_{\ell}(\hat{\mathbf{y}})\\ G_{L} &:= G_{L+1}\,J_{g_{L}}(\mathbf{z}_{L})\\ &\vdots\\ G_1 &:= G_2 \, J_{\bfg_1}(\mathbf{z}_1)\\ \nabla f(\mathbf{x})^T &:= G_1 \, J_{\bfg_0}(\mathbf{z}_0) \end{align*}\]

这可能看起来是对我们之前算法的微小修改。实际上并非如此——它要快得多！为了看到这一点，我们需要计算实现每种模式所涉及的基本操作的数量。

首先进行一个简单的计算：要乘以一个 \(n \times m\) 的矩阵 \(A = (a_{ij})_{i,j}\) 和一个 \(m \times \ell\) 的矩阵 \(B= (b_{ij})_{i,j}\)，需要多少基本运算（即加法和乘法）？让我们问问 ChatGPT。这是答案：

“为了计算将一个 \(n \times m\) 的矩阵乘以一个 \(m \times \ell\) 的矩阵所需的基本运算数，考虑结果 \(n \times \ell\) 矩阵每个元素的运算过程。

结果矩阵的每个元素是第一个矩阵的一行与第二个矩阵的一列的点积。如果第一个矩阵表示为 \(A\)，第二个矩阵表示为 \(B\)，那么结果矩阵 \(C\) 中位置 \(i, j\) 的元素计算如下：

\[ c_{ij} = a_{i1}b_{1j} + a_{i2}b_{2j} + \ldots + a_{im}b_{mj} \]

这个计算涉及 \(m\) 次乘法和 \(m - 1\) 次加法，因此每个元素总共需要 \(m + (m - 1) = 2m - 1\) 个基本运算。

由于矩阵 \(C\) 有 \(n \times \ell\) 个元素，整个矩阵 \(C\) 所需的基本运算总数为：

\[ (n \ell)(2m - 1) = 2nm\ell - n\ell \]

因此，乘以一个 \(n \times m\) 的矩阵和一个 \(m \times \ell\) 的矩阵需要 \(2nm\ell - n\ell\) 个基本运算。”

返回到计算梯度，我们计算每个模式的所需基本运算数。为了简化计算，我们假设 \(n_0 = d\)，\(n_1 = n_2 = \cdots = n_{L} = m\)，\(n_{L+1} = K\)。

*正向计算:* 矩阵 \(F_0 = J_{\bfg_0}(\mathbf{z}_0)\) 的维度为 \(m \times d\)。矩阵 \(F_1\)，作为 \(J_{\bfg_1}(\mathbf{z}_1) \in \mathbb{R}^{m \times m}\) 和 \(F_0 \in \mathbb{R}^{m \times d}\) 的乘积，其维度为 \(m \times d\)；因此，计算它需要 \(m (2m-1) d\) 次运算。对于 \(F_2, \ldots, F_{L-1}\) 也是如此（检查一下！）通过类似的考虑，矩阵 \(F_L\) 的维度为 \(K \times d\)，计算它需要 \(K (2m-1) d\) 次运算。最后，\(\nabla f(\mathbf{x})^T = J_{\ell}(\mathbf{z}_{L+1}) F_L \in \mathbb{R}^{1 \times d}\) 并需要 \((2K-1) d\) 次运算来计算。总体上，运算次数为

\[ (L-1) m (2m-1) d + K (2m-1) d + (2K-1) d. \]

如果将 \(K\) 视为一个小的常数并忽略较小的阶数项，这大约是 \(2 L m² d\)。

*逆向：* 矩阵 \(G_{L+1} = J_{\ell}(\mathbf{z}_{L+1})\) 的维度为 \(1 \times K\)。矩阵 \(G_{L}\)，作为 \(G_{L+1} \in \mathbb{R}^{1 \times K}\) 和 \(J_{g_{L}}(\mathbf{z}_{L}) \in \mathbb{R}^{K \times m}\) 的乘积，其维度为 \(1 \times m\)；因此需要 \((2K-1) m\) 次操作来计算。矩阵 \(G_{L-1}\)，作为 \(G_{L} \in \mathbb{R}^{1 \times m}\) 和 \(J_{g_{L-1}}(\mathbf{z}_{L-1}) \in \mathbb{R}^{m \times m}\) 的乘积，其维度为 \(1 \times m\)；因此需要 \((2m-1) m\) 次操作来计算。对于 \(G_{L-2}, \ldots, G_{1}\) 也是如此（请检查！）通过类似的考虑，\(\nabla f(\mathbf{x})^T = G_1 \, J_{\bfg_0}(\mathbf{z}_0) \in \mathbb{R}^{1 \times d}\) 并需要 \((2m-1) d\) 次操作来计算。总体上，操作次数为

\[ (2K-1) m + (L-1) (2m-1) m + (2m-1) d. \]

这大约是 \(2 L m² + 2 m d\) – 这可以比 \(2 L m² d\) 小得多！换句话说，逆向模式方法可以快得多。特别是请注意，逆向模式中的所有计算都是矩阵-向量乘法（或者更精确地说，是行向量-矩阵乘法），而不是矩阵-矩阵乘法。

**示例：** **（继续）** 我们将逆向模式方法应用于之前的示例。我们得到

\[\begin{align*} G_{L+1} &:= J_{\ell}(\hat{\mathbf{y}}) = (\hat{\mathbf{y}} - \mathbf{y})^T\\ G_{L} &:= G_{L+1}\,J_{g_{L}}(\mathbf{z}_{L}) = G_{L+1} \mathcal{W}_{L} = (\hat{\mathbf{y}} - \mathbf{y})^T \mathcal{W}_{L} \\ \vdots\\ G_1 &:= G_2 \, J_{\bfg_1}(\mathbf{z}_1) = G_2 \mathcal{W}_{1} = [(\hat{\mathbf{y}} - \mathbf{y})^T \mathcal{W}_{L} \cdots \mathcal{W}_{2}] \mathcal{W}_{1} \\ \nabla f(\mathbf{x})^T &:= G_1 \, J_{\bfg_0}(\mathbf{z}_0) = [(\hat{\mathbf{y}} - \mathbf{y})^T \mathcal{W}_{L} \cdots \mathcal{W}_{2}\mathcal{W}_{1}] \mathcal{W}_{0}\\ &= (\mathcal{W}_{L} \mathcal{W}_{L-1} \cdots \mathcal{W}_{1} \mathcal{W}_{0} \mathbf{x} - \mathbf{y})^T \mathcal{W}_{L} \mathcal{W}_{L-1} \cdots \mathcal{W}_{1} \mathcal{W}_{0}, \end{align*}\]

这与我们的先前计算相匹配。请注意，所有计算都涉及行向量与矩阵的乘法。 \(\lhd\)

**数值角：** 我们尝试我们的特定示例。

```py
with torch.no_grad():
    G2 = (z2 - y).unsqueeze(0)
    G1 = G2 @ W1
    grad_f = G1 @ W0

print(G2) 
```

```py
tensor([[-1.,  0.]]) 
```

```py
print(G1) 
```

```py
tensor([[1., 0.]]) 
```

```py
print(grad_f) 
```

```py
tensor([[ 0.,  1., -1.]]) 
```

我们确实再次得到了相同的答案。

\(\unlhd\)

为了更深入地了解通过反向模式获得的节省，考虑以下简单的计算。设 \(A, B \in \mathbb{R}^{n \times n}\) 和 \(\mathbf{v} \in \mathbb{R}^n\)。假设我们想要计算 \(\mathbf{v}^T B A\)。通过矩阵乘法的 [结合律](https://en.wikipedia.org/wiki/Associative_property)，有两种方法可以做到这一点：计算 \(\mathbf{v}^{T}(BA)\)（即，首先计算 \(BA\) 然后乘以 \(\mathbf{v}^T\)；或者计算 \((\mathbf{v}^T B) A\)。第一种方法需要 \(n²(2n-1) + n(2n-1)\) 次操作，而第二种方法只需要 \(2n(2n-1)\)。后者要小得多，因为当 \(n\) 很大时，\(2 n³\)（第一种方法中的主导项）的增长速度比 \(4 n²\)（第二种方法中的主导项）快得多。

为什么会发生这种情况？理解这一点的其中一个方法是将输出 \(\mathbf{v}^T B A\) 视为 \(A\) 的行的一个 *线性组合* – 实际上是一个非常具体的线性组合。在第一种方法中，我们计算 \(BA\)，这给我们 \(n\) 个不同的 \(A\) 的行的线性组合 – 没有一个是我们想要的 – 然后我们通过乘以 \(\mathbf{v}^T\) 来计算所需的线性组合。这是浪费的。在第二种方法中，我们立即计算我们寻求的特定线性组合的系数 – \(\mathbf{v}^T B\) – 然后通过右乘 \(A\) 来计算那个线性组合。

虽然我们在本小节中考察的设置很有启发性，但它并不完全符合我们的要求。在机器学习环境中，每个“层” \(\bfg_i\) 都有参数（在我们的例子中，是 \(\mathcal{W}_{i}\) 的项），我们寻求对这些参数进行优化。为此，我们需要关于参数的梯度，而不是输入 \(\mathbf{x}\) 的梯度。在下一个小节中，我们考虑当前设置的推广，即渐进函数，这将使我们能够做到这一点。符号变得更加复杂，但基本思想保持不变。

## 8.3.2\. 渐进函数#

如前所述，虽然将预测函数 \(h\)（例如，分类器）定义为输入数据 \(\mathbf{x}\in \mathbb{R}^{d}\) 的函数看起来很自然，但在拟合数据时，我们最终感兴趣的是将 \(h\) 视为需要调整的参数 \(\mathbf{w} \in \mathbb{R}^r\) 的函数 – 在一个固定的数据集上。因此，在本节中，输入 \(\mathbf{x}\) 是固定的，而参数向量 \(\mathbf{w}\) 现在是可变的。

**第一个例子** 我们使用前一小节中的例子来说明主要思想。也就是说，假设 \(d=3\)，\(L=1\)，\(n_1 = 2\)，和 \(K = 2\)。固定一个数据样本 \(\mathbf{x} = (x_1,x_2,x_3) \in \mathbb{R}³, \mathbf{y} = (y_1, y_2) \in \mathbb{R}²\)。对于 \(i=0, 1\)，我们使用以下符号

\[\begin{split} \mathcal{W}_{0} = \begin{pmatrix} w_0 & w_1 & w_2\\ w_3 & w_4 & w_5 \end{pmatrix} \quad \text{and} \quad \mathcal{W}_{1} = \begin{pmatrix} w_6 & w_7\\ w_8 & w_9 \end{pmatrix}. \end{split}\]

并且令

\[ \ell(\hat{\mathbf{y}}) = \frac{1}{2} \|\mathbf{y} - \hat{\mathbf{y}}\|² = \frac{1}{2}(y_1 - \hat{y}_1)² + \frac{1}{2}(y_2 - \hat{y}_2)². \]

我们将“层”函数 \(\bfg_i\) 的表示法进行更改，以反映它现在是一个两个（连接的）向量的函数：来自前一层的输入 \(\mathbf{z}_i = (z_{i,1},\ldots,z_{i,n_i})\) 和一个特定层的参数集 \(\mathbf{w}_i\)。也就是说，

\[\begin{split} \bfg_i(\mathbf{z}_i, \mathbf{w}_i) = \mathcal{W}_{i} \mathbf{z}_i = \begin{pmatrix} (\mathbf{w}_i^{(1)})^T\\ (\mathbf{w}_i^{(2)})^T \end{pmatrix} \mathbf{z}_i \end{split}\]

其中 \(\mathbf{w}_i = (\mathbf{w}_i^{(1)}, \mathbf{w}_i^{(2)})\)，这是 \(\mathcal{W}_{i}\) 的行（作为列向量）的连接。另一种说法是

\[ \mathbf{w}_i = \mathrm{vec}(\mathcal{W}_{i}^T), \]

其中我们取转置是为了将行转换为列。更具体地说，

\[ \bfg_0(\mathbf{z}_0, \mathbf{w}_0) = \mathcal{W}_{0} \mathbf{z}_{0} \quad\text{with}\quad \mathbf{w}_0 = (w_0, w_1, w_2, w_3, w_4, w_5) \]

（即，\(\mathbf{w}_0^{(1)} = (w_0, w_1, w_2)\) 和 \(\mathbf{w}_0^{(2)} = (w_3, w_4, w_5)\)）和

\[ \bfg_1(\mathbf{z}_1, \mathbf{w}_1) = \mathcal{W}_{1} \mathbf{z}_{1} \quad\text{with}\quad \mathbf{w}_1 = (w_6, w_7, w_8, w_9) \]

（即，\(\mathbf{w}_1^{(1)} = (w_6, w_7)\) 和 \(\mathbf{w}_1^{(2)} = (w_8, w_9)\)）。

我们试图计算

\[\begin{align*} f(\mathbf{w}) &= \ell(\bfg_1(\bfg_0(\mathbf{x},\mathbf{w}_0),\mathbf{w}_1))\\ &= \frac{1}{2} \|\mathbf{y} - \mathcal{W}_{1} \mathcal{W}_{0} \mathbf{x}\|²\\ &= \frac{1}{2}\left(y_1 - w_6(w_0 x_1 + w_1 x_2 + w_2 x_3) - w_7(w_3 x_1 + w_4 x_2 + w_5 x_3)\right)²\\ & \qquad + \frac{1}{2}\left(y_2 - w_8(w_0 x_1 + w_1 x_2 + w_2 x_3) - w_9(w_3 x_1 + w_4 x_2 + w_5 x_3)\right)². \end{align*}\]

通过应用反向的**链式法则**，正如我们在前一小节中所证明的那样——但这次我们对参数求梯度

\[ \mathbf{w} := (\mathbf{w}_0, \mathbf{w}_1) = (w_0,w_1,\ldots,w_9). \]

注意符号的一个关键变化：我们现在相应地认为 \(f\) 是 \(\mathbf{w}\) 的函数；\(\mathbf{x}\) 的作用是隐含的。

另一方面，当我们刚刚说我们只关心与前者相关的梯度时，现在将 \(\bfg_i\) 视为既与其自身参数又与前一层输入相关的函数，这似乎有些反直觉。但是，正如我们将要看到的，我们确实需要关于两者的雅可比矩阵，因为前一层的输入实际上依赖于前一层的参数。例如，\(\bfg_1(\mathbf{z}_1, \mathbf{w}_1) = \mathcal{W}_{1} \mathbf{z}_{1}\)，其中 \(\mathbf{z}_{1} = \bfg_0(\mathbf{z}_0, \mathbf{w}_0) = \mathcal{W}_{0} \mathbf{z}_{0}\)。

回想一下，我们已经在之前的例子中计算了所需的雅可比矩阵 \(J_{\bfg_0}\) 和 \(J_{\bfg_1}\)。我们还计算了 \(\ell\) 的雅可比矩阵 \(J_{\ell}\)。在这个时候，应用**链式法则**并推断出 \(f\) 的梯度是

\[ J_{\ell}(\bfg_1(\bfg_0(\mathbf{x},\mathbf{w}_0),\mathbf{w}_1)) \,J_{\bfg_1}(\bfg_0(\mathbf{x},\mathbf{w}_0),\mathbf{w}_1) \,J_{\bfg_0}(\mathbf{x}, \mathbf{w}_0). \]

但这并不正确。首先，维度不匹配！例如，\(J_{\bfg_0} \in \mathbb{R}^{2 \times 9}\)，因为\(\bfg_0\)有\(2\)个输出和\(9\)个输入（即，\(z_{0,1}, z_{0,2}, z_{0,3}, w_0, w_1, w_2, w_3, w_4, w_5\)），而\(J_{\bfg_1} \in \mathbb{R}^{2 \times 6}\)，因为\(\bfg_1\)有\(2\)个输出和\(6\)个输入（即，\(z_{1,1}, z_{1,2}, w_6, w_7, w_8, w_9\)）。那么问题出在哪里呢？

函数 \(f\) 实际上并不是由函数 \(\ell\)、\(\bfg_1\) 和 \(\bfg_0\) 的简单组合。事实上，我们要对参数进行微分，这些参数是逐步引入的，每一层都注入它自己的附加参数，而这些参数并不是从前一层次获得的。因此，我们不能像前一小节中那样简单地将雅可比矩阵的乘积作为 \(f\) 的梯度来写。

但并非一切都已失去。我们下面将展示，我们仍然可以逐步应用**链式法则**，同时考虑到每一层的附加参数。借鉴前一小节的方法，我们首先向前计算 \(f\) 和雅可比矩阵，然后向后计算梯度 \(\nabla f\)。我们使用背景部分中的符号 \(\mathbb{A}_{n}[\mathbf{x}]\) 和 \(\mathbb{B}_{n}[\mathbf{z}]\)。

在正向阶段，我们计算 \(f\) 本身和所需的雅可比矩阵：

\[\begin{align*} &\mathbf{z}_0 := \mathbf{x}\\ & = (x_1, x_2, x_3)\\ &\mathbf{z}_1 := \bfg_0(\mathbf{z}_0, \mathbf{w}_0) = \mathcal{W}_{0} \mathbf{z}_{0}\\ &= \begin{pmatrix} (\mathbf{w}_0^{(1)})^T\mathbf{x}\\ (\mathbf{w}_0^{(2)})^T\mathbf{x}\end{pmatrix} = \begin{pmatrix} w_0 x_1 + w_1 x_2 + w_2 x_3\\ w_3 x_1 + w_4 x_2 + w_5 x_3 \end{pmatrix}\\ &J_{\bfg_0}(\mathbf{z}_0, \mathbf{w}_0) := \begin{pmatrix} \mathbb{A}_{2}[\mathbf{w}_0] & \mathbb{B}_{2}[\mathbf{z}_0] \end{pmatrix} = \begin{pmatrix} \mathcal{W}_{0} & I_{2\times 2} \otimes \mathbf{z}_0^T \end{pmatrix}\\ &= \begin{pmatrix} w_0 & w_1 & w_2 & x_1 & x_2 & x_3 & 0 & 0 & 0\\ w_3 & w_4 & w_5 & 0 & 0 & 0 & x_1 & x_2 & x_3 \end{pmatrix} \end{align*}\]\[\begin{align*} &\hat{\mathbf{y}} := \mathbf{z}_2 := \bfg_1(\mathbf{z}_1, \mathbf{w}_1) = \mathcal{W}_{1} \mathbf{z}_{1}\\ &= \begin{pmatrix} w_6 z_{1,1} + w_7 z_{1,2}\\ w_8 z_{1,1} + w_9 z_{1,2} \end{pmatrix}\\ &= \begin{pmatrix} w_6 (\mathbf{w}_0^{(1)})^T\mathbf{x} + w_7 (\mathbf{w}_0^{(2)})^T\mathbf{x}\\ w_8 (\mathbf{w}_0^{(1)})^T\mathbf{x} + w_9 (\mathbf{w}_0^{(2)})^T\mathbf{x} \end{pmatrix}\\ &= \begin{pmatrix} w_6(w_0 x_1 + w_1 x_2 + w_2 x_3) + w_7(w_3 x_1 + w_4 x_2 + w_5 x_3)\\ w_8(w_0 x_1 + w_1 x_2 + w_2 x_3) + w_9(w_3 x_1 + w_4 x_2 + w_5 x_3) \end{pmatrix}\\ &J_{\bfg_1}(\mathbf{z}_1, \mathbf{w}_1):= \begin{pmatrix} \mathbb{A}_{2}[\mathbf{w}_1] & \mathbb{B}_{2}[\mathbf{z}_1] \end{pmatrix} = \begin{pmatrix} \mathcal{W}_{1} & I_{2\times 2} \otimes \mathbf{z}_1^T \end{pmatrix}\\ &= \begin{pmatrix} w_6 & w_7 & z_{1,1} & z_{1,2} & 0 & 0\\ w_8 & w_9 & 0 & 0 & z_{1,1} & z_{1,2} \end{pmatrix}\\ &= \begin{pmatrix} w_6 & w_7 & (\mathbf{w}_0^{(1)})^T\mathbf{x} & (\mathbf{w}_0^{(2)})^T\mathbf{x} & 0 & 0\\ w_8 & w_9 & 0 & 0 & (\mathbf{w}_0^{(1)})^T\mathbf{x} & (\mathbf{w}_0^{(2)})^T\mathbf{x} \end{pmatrix} \end{align*}\]\[\begin{align*} &f(\mathbf{x}) := \ell(\hat{\mathbf{y}}) = \frac{1}{2} \|\mathbf{y} - \hat{\mathbf{y}}\|²\\ &= \frac{1}{2}\left(y_1 - w_6(w_0 x_1 + w_1 x_2 + w_2 x_3) - w_7(w_3 x_1 + w_4 x_2 + w_5 x_3)\right)²\\ & \qquad + \frac{1}{2}\left(y_2 - w_8(w_0 x_1 + w_1 x_2 + w_2 x_3) - w_9(w_3 x_1 + w_4 x_2 + w_5 x_3)\right)²\\ &J_{\ell}(\hat{\mathbf{y}}) = (\hat{\mathbf{y}} - \mathbf{y})^T\\ &= \begin{pmatrix} w_6 (\mathbf{w}_0^{(1)})^T\mathbf{x} + w_7 (\mathbf{w}_0^{(2)})^T\mathbf{x} - y_1 & w_8 (\mathbf{w}_0^{(1)})^T\mathbf{x} + w_9 (\mathbf{w}_0^{(2)})^T\mathbf{x} - y_2 \end{pmatrix}. \end{align*}\]

我们现在计算 \(f\) 关于 \(\mathbf{w}\) 的梯度。我们首先从 \(\mathbf{w}_1 = (w_6, w_7, w_8, w_9)\) 开始。对于这一步，我们将 \(f\) 视为 \(\ell(\bfg_1(\mathbf{z}_1, \mathbf{w}_1))\) 的复合函数。在这里，\(\mathbf{z}_1\) 不依赖于 \(\mathbf{w}_1\)，因此可以认为在这个计算中是固定的。根据 **链式法则**

\[\begin{align*} \frac{\partial f(\mathbf{w})}{\partial w_6} &= \frac{\partial \ell(\bfg_1(\mathbf{z}_1, \mathbf{w}_1))}{\partial w_6} = \frac{\partial \ell(\hat{\mathbf{y}})}{\partial \hat{y}_1} \frac{\partial g_{1,1}(\mathbf{z}_1, \mathbf{w}_1)}{\partial w_6} + \frac{\partial \ell(\hat{\mathbf{y}})}{\partial \hat{y}_2} \frac{\partial g_{1,2}(\mathbf{z}_1, \mathbf{w}_1)}{\partial w_6} = (\hat{y}_1 - y_1) z_{1,1} \end{align*}\]

其中我们使用了 \(g_{1,2}(\mathbf{z}_1, \mathbf{w}_1) = w_8 z_{1,1} + w_9 z_{1,2}\) 不依赖于 \(w_6\) 的这一事实，因此 \(\frac{\partial g_{1,2}(\mathbf{z}_1, \mathbf{w}_1)}{\partial w_6} = 0\)。同样

\[\begin{align*} \frac{\partial f(\mathbf{w})}{\partial w_7} &= \frac{\partial \ell(\bfg_1(\mathbf{z}_1, \mathbf{w}_1))}{\partial w_7} =\frac{\partial \ell(\hat{\mathbf{y}})}{\partial \hat{y}_1} \frac{\partial g_{1,1}(\mathbf{z}_1, \mathbf{w}_1)}{\partial w_7} + \frac{\partial \ell(\hat{\mathbf{y}})}{\partial \hat{y}_2} \frac{\partial g_{1,2}(\mathbf{z}_1, \mathbf{w}_1)}{\partial w_7} = (\hat{y}_1 - y_1) z_{1,2}\\ \frac{\partial f(\mathbf{w})}{\partial w_8} &= \frac{\partial \ell(\bfg_1(\mathbf{z}_1, \mathbf{w}_1))}{\partial w_8} =\frac{\partial \ell(\hat{\mathbf{y}})}{\partial \hat{y}_1} \frac{\partial g_{1,1}(\mathbf{z}_1, \mathbf{w}_1)}{\partial w_8} + \frac{\partial \ell(\hat{\mathbf{y}})}{\partial \hat{y}_2} \frac{\partial g_{1,2}(\mathbf{z}_1, \mathbf{w}_1)}{\partial w_8} = (\hat{y}_2 - y_2) z_{1,1}\\ \frac{\partial f(\mathbf{w})}{\partial w_9} &= \frac{\partial \ell(\bfg_1(\mathbf{z}_1, \mathbf{w}_1))}{\partial w_9} =\frac{\partial \ell(\hat{\mathbf{y}})}{\partial \hat{y}_1} \frac{\partial g_{1,1}(\mathbf{z}_1, \mathbf{w}_1)}{\partial w_9} + \frac{\partial \ell(\hat{\mathbf{y}})}{\partial \hat{y}_2} \frac{\partial g_{1,2}(\mathbf{z}_1, \mathbf{w}_1)}{\partial w_9} = (\hat{y}_2 - y_2) z_{1,2}. \end{align*}\]

以矩阵形式表示，这为

\[\begin{align*} &\begin{pmatrix}\frac{\partial f(\mathbf{w})}{\partial w_6} & \frac{\partial f(\mathbf{w})}{\partial w_7} & \frac{\partial f(\mathbf{w})}{\partial w_8} & \frac{\partial f(\mathbf{w})}{\partial w_9} \end{pmatrix}\\ &= J_{\ell}(\hat{\mathbf{y}}) \,\mathbb{B}_{2}[\mathbf{z}_1]\\ &= (\hat{\mathbf{y}} - \mathbf{y})^T (I_{2\times 2} \otimes \mathbf{z}_1^T)\\ &= (\hat{\mathbf{y}} - \mathbf{y})^T \otimes \mathbf{z}_1^T\\ &= \begin{pmatrix} (\hat{y}_1 - y_1) z_{1,1} & (\hat{y}_1 - y_1) z_{1,2} & (\hat{y}_2 - y_2) z_{1,1} & (\hat{y}_2 - y_2) z_{1,2} \end{pmatrix} \end{align*}\]

其中我们在最后一行使用了**克罗内克积的性质（f）**。

要计算相对于 \(\mathbf{w}_0 = (w_0, w_1, \ldots, w_5)\) 的偏导数，我们首先需要计算相对于 \(\mathbf{z}_1 = (z_{1,1}, z_{1,2})\) 的偏导数，因为 \(f\) 通过它依赖于 \(\mathbf{w}_0\)。为此计算，我们再次将 \(f\) 视为 \(\ell(\bfg_1(\mathbf{z}_1, \mathbf{w}_1))\) 的复合，但这次我们的重点是变量 \(\mathbf{z}_1\)。我们得到

\[\begin{align*} \frac{\partial f(\mathbf{w})}{\partial z_{1,1}} &= \frac{\partial \ell(\bfg_1(\mathbf{z}_1, \mathbf{w}_1))}{\partial z_{1,1}}\\ &= \frac{\partial \ell(\hat{\mathbf{y}})}{\partial \hat{y}_1} \frac{\partial g_{1,1}(\mathbf{z}_1, \mathbf{w}_1)}{\partial z_{1,1}} + \frac{\partial \ell(\hat{\mathbf{y}})}{\partial \hat{y}_2} \frac{\partial g_{1,2}(\mathbf{z}_1, \mathbf{w}_1)}{\partial z_{1,1}}\\ &= (\hat{y}_1 - y_1) w_6 + (\hat{y}_2 - y_2) w_8 \end{align*}\]

和

\[\begin{align*} \frac{\partial f(\mathbf{w})}{\partial z_{1,2}} &= \frac{\partial \ell(\bfg_1(\mathbf{z}_1, \mathbf{w}_1))}{\partial z_{1,2}}\\ &= \frac{\partial \ell(\hat{\mathbf{y}})}{\partial \hat{y}_1} \frac{\partial g_{1,1}(\mathbf{z}_1, \mathbf{w}_1)}{\partial z_{1,2}} + \frac{\partial \ell(\hat{\mathbf{y}})}{\partial \hat{y}_2} \frac{\partial g_{1,2}(\mathbf{z}_1, \mathbf{w}_1)}{\partial z_{1,2}}\\ &= (\hat{y}_1 - y_1) w_7 + (\hat{y}_2 - y_2) w_9. \end{align*}\]

以矩阵形式，这可以表示为

\[\begin{align*} &\begin{pmatrix}\frac{\partial f(\mathbf{w})}{\partial z_{1,1}} & \frac{\partial f(\mathbf{w})}{\partial z_{1,2}} \end{pmatrix}\\ &= J_{\ell}(\hat{\mathbf{y}}) \,\mathbb{A}_{2}[\mathbf{w}_1]\\ &= (\hat{\mathbf{y}} - \mathbf{y})^T \mathcal{W}_1\\ &= \begin{pmatrix} (\hat{y}_1 - y_1) w_6 + (\hat{y}_2 - y_2) w_8 & (\hat{y}_1 - y_1) w_7 + (\hat{y}_2 - y_2) w_9 \end{pmatrix}. \end{align*}\]

向量 \(\left(\frac{\partial f(\mathbf{w})}{\partial z_{1,1}}, \frac{\partial f(\mathbf{w})}{\partial z_{1,2}}\right)\) 被称为伴随向量。

现在，我们计算 \(f\) 相对于 \(\mathbf{w}_0 = (w_0, w_1, \ldots, w_5)\) 的梯度。为此步骤，我们将 \(f\) 视为 \(\ell(\bfg_1(\mathbf{z}_1, \mathbf{w}_1))\) 作为 \(\mathbf{z}_1\) 的函数，以及 \(\bfg_0(\mathbf{z}_0, \mathbf{w}_0)\) 作为 \(\mathbf{w}_0\) 的函数的复合。在这里，\(\mathbf{z}_0\) 不依赖于 \(\mathbf{w}_0\)，因此可以在此计算中视为固定。根据**链式法则**

\[\begin{align*} \frac{\partial f(\mathbf{w})}{\partial w_0} &= \frac{\partial \ell(\bfg_1(\bfg_0(\mathbf{z}_0, \mathbf{w}_0), \mathbf{w}_1))}{\partial w_0}\\ &= \frac{\partial \ell(\bfg_1(\mathbf{z}_1, \mathbf{w}_1))}{\partial z_{1,1}} \frac{\partial g_{0,1}(\mathbf{z}_0, \mathbf{w}_0)}{\partial w_0} + \frac{\partial \ell(\bfg_1(\mathbf{z}_1, \mathbf{w}_1))}{\partial z_{1,2}} \frac{\partial g_{0,2}(\mathbf{z}_0, \mathbf{w}_0)}{\partial w_0}\\ &= ((\hat{y}_1 - y_1) w_6 + (\hat{y}_2 - y_2) w_8) z_{0,1}\\ &= ((\hat{y}_1 - y_1) w_6 + (\hat{y}_2 - y_2) w_8) x_{1}, \end{align*}\]

其中我们使用了 \(g_{0,2}(\mathbf{z}_0, \mathbf{w}_0) = w_3 z_{0,1} + w_4 z_{0,2} + w_5 z_{0,3}\) 不依赖于 \(w_0\) 的性质，因此 \(\frac{\partial g_{0,2}(\mathbf{z}_0, \mathbf{w}_0)}{\partial w_0} = 0\).

同样（检查一下！）

\[\begin{align*} \frac{\partial f(\mathbf{w})}{\partial w_1} &= ((\hat{y}_1 - y_1) w_6 + (\hat{y}_2 - y_2) w_8) x_{2}\\ \frac{\partial f(\mathbf{w})}{\partial w_2} &= ((\hat{y}_1 - y_1) w_6 + (\hat{y}_2 - y_2) w_8) x_{3}\\ \frac{\partial f(\mathbf{w})}{\partial w_3} &= ((\hat{y}_1 - y_1) w_7 + (\hat{y}_2 - y_2) w_9) x_{1}\\ \frac{\partial f(\mathbf{w})}{\partial w_4} &= ((\hat{y}_1 - y_1) w_7 + (\hat{y}_2 - y_2) w_9) x_{2}\\ \frac{\partial f(\mathbf{w})}{\partial w_5} &= ((\hat{y}_1 - y_1) w_7 + (\hat{y}_2 - y_2) w_9) x_{3}. \end{align*}\]

以矩阵形式表示，这是

\[\begin{align*} &\begin{pmatrix}\frac{\partial f(\mathbf{w})}{\partial w_0} & \frac{\partial f(\mathbf{w})}{\partial w_1} & \frac{\partial f(\mathbf{w})}{\partial w_2} & \frac{\partial f(\mathbf{w})}{\partial w_3} & \frac{\partial f(\mathbf{w})}{\partial w_4} & \frac{\partial f(\mathbf{w})}{\partial w_5} \end{pmatrix}\\ &= J_{\ell}(\hat{\mathbf{y}}) \,\mathbb{A}_{2}[\mathbf{w}_1] \,\mathbb{B}_{2}[\mathbf{z}_0]\\ &= (\hat{\mathbf{y}} - \mathbf{y})^T \mathcal{W}_1 (I_{2\times 2} \otimes \mathbf{z}_0^T)\\ &= ((\hat{\mathbf{y}} - \mathbf{y})^T \mathcal{W}_1) \otimes \mathbf{x}^T\\ &= \begin{pmatrix} ((\hat{y}_1 - y_1) w_6 + (\hat{y}_2 - y_2) w_8) x_{1} & \cdots & ((\hat{y}_1 - y_1) w_7 + (\hat{y}_2 - y_2) w_9) x_{3} \end{pmatrix} \end{align*}\]

其中我们在最后一行使用了克罗内克积的性质（f）。

总结起来，

\[ \nabla f (\mathbf{w})^T = \begin{pmatrix} (\hat{\mathbf{y}} - \mathbf{y})^T \otimes (\mathcal{W}_{0} \mathbf{x})^T & ((\hat{\mathbf{y}} - \mathbf{y})^T \mathcal{W}_1) \otimes \mathbf{x}^T \end{pmatrix}. \]

**数值角:** 我们回到前一小节的具体例子。这次矩阵 `W0` 和 `W1` 需要计算偏导数。

```py
x = torch.tensor([1.,0.,-1.])
y = torch.tensor([0.,1.])
W0 = torch.tensor([[0.,1.,-1.],[2.,0.,1.]], requires_grad=True)
W1 = torch.tensor([[-1.,0.],[2.,-1.]], requires_grad=True)

z0 = x
z1 = W0 @ z0
z2 = W1 @ z1
f = 0.5 * (torch.linalg.vector_norm(y-z2) ** 2)

print(z0) 
```

```py
tensor([ 1.,  0., -1.]) 
```

```py
print(z1) 
```

```py
tensor([1., 1.], grad_fn=<MvBackward0>) 
```

```py
print(z2) 
```

```py
tensor([-1.,  1.], grad_fn=<MvBackward0>) 
```

```py
print(f) 
```

```py
tensor(0.5000, grad_fn=<MulBackward0>) 
```

我们使用 AD 计算梯度 \(\nabla f(\mathbf{w})\).

```py
f.backward() 
```

```py
print(W0.grad) 
```

```py
tensor([[ 1.,  0., -1.],
        [ 0.,  0., -0.]]) 
```

```py
print(W1.grad) 
```

```py
tensor([[-1., -1.],
        [-0., -0.]]) 
```

这些是以矩阵导数的形式编写的

\[\begin{split} \frac{\partial f}{\partial \mathcal{W}_0} = \begin{pmatrix} \frac{\partial f}{\partial w_0} & \frac{\partial f}{\partial w_1} & \frac{\partial f}{\partial w_2} \\ \frac{\partial f}{\partial w_3} & \frac{\partial f}{\partial w_4} & \frac{\partial f}{\partial w_5} \end{pmatrix} \quad\text{and}\quad \frac{\partial f}{\partial \mathcal{W}_1} = \begin{pmatrix} \frac{\partial f}{\partial w_6} & \frac{\partial f}{\partial w_7} \\ \frac{\partial f}{\partial w_8} & \frac{\partial f}{\partial w_9} \end{pmatrix}. \end{split}\]

我们使用我们的公式来验证它们是否与这些结果匹配。我们需要克罗内克积，在 PyTorch 中通过 `torch.kron` 实现[torch.kron](https://pytorch.org/docs/stable/generated/torch.kron.html)。

```py
with torch.no_grad():
    grad_W0 = torch.kron((z2 - y).unsqueeze(0) @ W1, z0.unsqueeze(0))
    grad_W1 = torch.kron((z2 - y).unsqueeze(0), z1.unsqueeze(0))

print(grad_W0) 
```

```py
tensor([[ 1.,  0., -1.,  0.,  0., -0.]]) 
```

```py
print(grad_W1) 
```

```py
tensor([[-1., -1.,  0.,  0.]]) 
```

注意这次这些结果是以向量化的形式（即通过连接行获得）写出的。但它们与 AD 输出相匹配。

\(\unlhd\)

**通用设置** \(\idx{渐进函数}\xdi\) 更普遍地，我们有 \(L+2\) 层。输入层是 \(\mathbf{z}_0 := \mathbf{x}\)，我们将其称为层 \(0\)。隐藏层 \(i\)，\(i=1,\ldots,L\)，由一个连续可微的函数 \(\mathbf{z}_i := \bfg_{i-1}(\mathbf{z}_{i-1}, \mathbf{w}_{i-1})\) 定义，这次它接受 *两个向量值输入*：一个来自 \((i-1)\) 层的向量 \(\mathbf{z}_{i-1} \in \mathbb{R}^{n_{i-1}}\) 和一个属于 \(i\) 层的参数向量 \(\mathbf{w}_{i-1} \in \mathbb{R}^{r_{i-1}}\)

\[ \bfg_{i-1} = (g_{i-1,1},\ldots,g_{i-1,n_{i}}) : \mathbb{R}^{n_{i-1} + r_{i-1}} \to \mathbb{R}^{n_{i}}. \]

\(\bfg_{i-1}\) 的输出 \(\mathbf{z}_i\) 是一个 \(\mathbb{R}^{n_{i}}\) 中的向量，它作为输入传递给 \((i+1)\) 层。输出层是 \(\mathbf{z}_{L+1} := \bfg_{L}(\mathbf{z}_{L}, \mathbf{w}_{L})\)，我们也将它称为层 \(L+1\)。

对于 \(i = 1,\ldots,L+1\)，设

\[ \overline{\mathbf{w}}^{i-1} = (\mathbf{w}_0,\mathbf{w}_1,\ldots,\mathbf{w}_{i-1}) \in \mathbb{R}^{r_0 + r_1+\cdots+r_{i-1}} \]

是前 \(i\) 层参数（不包括没有参数的输入层）的连接，作为一个向量在 \(\mathbb{R}^{r_0+r_1+\cdots+r_{i-1}}\) 中。然后层 \(i\) 的输出 *作为参数的函数* 是组合

\[ \mathcal{O}_{i-1}(\overline{\mathbf{w}}^{i-1}) = \bfg_{i-1}(\mathcal{O}_{i-2}(\overline{\mathbf{w}}^{i-2}), \mathbf{w}_{i-1}) = \bfg_{i-1}(\bfg_{i-2}(\cdots \bfg_1(\bfg_0(\mathbf{x},\mathbf{w}_0),\mathbf{w}_1), \cdots, \mathbf{w}_{i-2}), \mathbf{w}_{i-1}) \in \mathbb{R}^{n_{i}}, \]

对于 \(i = 2, \ldots, L+1\)。当 \(i=1\) 时，我们有

\[ \mathcal{O}_{0}(\overline{\mathbf{w}}^{0}) = \bfg_{0}(\mathbf{x}, \mathbf{w}_0). \]

注意到函数 \(\mathcal{O}_{i-1}\) 依赖于输入 \(\mathbf{x}\) —— 在这个设置中我们并不将其视为变量。为了简化符号，我们没有明确表示对 \(\mathbf{x}\) 的依赖。

令 \(\mathbf{w} := \overline{\mathbf{w}}^{L}\)，最终的输出是

\[ \bfh(\mathbf{w}) = \mathcal{O}_{L}(\overline{\mathbf{w}}^{L}). \]

展开组合，这可以写成另一种形式

\[ \bfh(\mathbf{w}) = \bfg_{L}(\bfg_{L-1}(\cdots \bfg_1(\bfg_0(\mathbf{x},\mathbf{w}_0),\mathbf{w}_1), \cdots, \mathbf{w}_{L-1}), \mathbf{w}_{L}). \]

同样，我们没有明确表示对 \(\mathbf{x}\) 的依赖。

作为最后一步，我们有一个损失函数 \(\ell : \mathbb{R}^{n_{L+1}} \to \mathbb{R}\)，它接受最后一层的输出作为输入，并衡量与给定标签 \(\mathbf{y} \in \Delta_K\) 的拟合度。下面我们将看到一些示例。最终的函数是

\[ f(\mathbf{w}) = \ell(\bfh(\mathbf{w})) \in \mathbb{R}. \]

我们寻求计算 \(f(\mathbf{w})\) 关于参数 \(\mathbf{w}\) 的梯度，以便应用梯度下降法。

**示例：** **（继续）** 我们回到上一小节中的运行示例。也就是说，\(\bfg_i(\mathbf{z}_i, \mathbf{w}_i) = \mathcal{W}_{i} \mathbf{z}_i\)，其中 \(\mathcal{W}_{i} \in \mathbb{R}^{n_{i+1} \times n_i}\) 的元素被视为参数，我们令 \(\mathbf{w}_i = \mathrm{vec}(\mathcal{W}_{i}^T)\)。还假设 \(\ell : \mathbb{R}^K \to \mathbb{R}\) 被定义为

\[ \ell(\hat{\mathbf{y}}) = \frac{1}{2} \|\mathbf{y} - \hat{\mathbf{y}}\|², \]

对于一个固定的、已知的向量 \(\mathbf{y} \in \mathbb{R}^{K}\)。

通过递归计算 \(f\) 得到

\[\begin{align*} \mathbf{z}_0 &:= \mathbf{x}\\ \mathbf{z}_1 &:= \mathcal{O}_0(\overline{\mathbf{w}}⁰) = \bfg_0(\mathbf{z}_0, \mathbf{w}_0) = \mathcal{W}_{0} \mathbf{z}_0 = \mathcal{W}_{0} \mathbf{x}\\ \mathbf{z}_2 &:= \mathcal{O}_1(\overline{\mathbf{w}}¹) = \bfg_1(\mathbf{z}_1, \mathbf{w}_1) = \mathcal{W}_{1} \mathbf{z}_1 = \mathcal{W}_{1} \mathcal{W}_{0} \mathbf{x}\\ \vdots\\ \mathbf{z}_L &:= \mathcal{O}_{L-1}(\overline{\mathbf{w}}^{L-1}) = \bfg_{L-1}(\mathbf{z}_{L-1}, \mathbf{w}_{L-1}) = \mathcal{W}_{L-1} \mathbf{z}_{L-1} = \mathcal{W}_{L-1} \cdots \mathcal{W}_{1} \mathcal{W}_{0} \mathbf{x}\\ \hat{\mathbf{y}} := \mathbf{z}_{L+1} &:= \mathcal{O}_{L}(\overline{\mathbf{w}}^{L}) = \bfg_{L}(\mathbf{z}_{L}, \mathbf{w}_{L}) = \mathcal{W}_{L} \mathbf{z}_{L} = \mathcal{W}_{L} \mathcal{W}_{L-1} \cdots \mathcal{W}_{1} \mathcal{W}_{0} \mathbf{x}\\ f(\mathbf{x}) &:= \ell(\hat{\mathbf{y}}) = \frac{1}{2}\|\mathbf{y} - \hat{\mathbf{y}}\|² = \frac{1}{2}\left\|\mathbf{y} - \mathcal{W}_{L} \mathcal{W}_{L-1} \cdots \mathcal{W}_{1} \mathcal{W}_{0} \mathbf{x}\right\|². \end{align*}\]

\(\lhd\)

**应用链式法则** 回想一下，从**链式法则**中得到的关键洞察是，为了计算复合函数如 \(\bfh(\mathbf{w})\) 的梯度——无论其多么复杂——只需要分别计算中间函数的雅可比矩阵，然后取**矩阵乘积**。在本节中，我们计算渐进情况下的必要雅可比矩阵。

将基本复合步骤重新写为将是有用的。

\[ \mathcal{O}_{i}(\overline{\mathbf{w}}^{i}) = \bfg_{i}(\mathcal{O}_{i-1}(\overline{\mathbf{w}}^{i-1}), \mathbf{w}_{i}) = \bfg_{i}(\mathcal{I}_{i}(\overline{\mathbf{w}}^{i})) \in \mathbb{R}^{n_{i+1}}, \]

其中，层 \(i+1\) 的输入（包括层特定参数和前一层输出）是

\[ \mathcal{I}_{i}(\overline{\mathbf{w}}^{i}) = \left( \mathcal{O}_{i-1}(\overline{\mathbf{w}}^{i-1}), \mathbf{w}_{i} \right) \in \mathbb{R}^{n_{i} + r_{i}}, \]

对于 \(i = 1, \ldots, L\)。当 \(i=0\) 时，我们有

\[ \mathcal{I}_{0}(\overline{\mathbf{w}}^{0}) = \left(\mathbf{z}_0, \mathbf{w}_0 \right) = \left(\mathbf{x}, \mathbf{w}_0 \right). \]

应用**链式法则**，我们得到

\[ J_{\mathcal{O}_{i}}(\overline{\mathbf{w}}^{i}) = J_{\bfg_i}(\mathcal{I}_{i}(\overline{\mathbf{w}}^{i})) \,J_{\mathcal{I}_{i}}(\overline{\mathbf{w}}^{i}). \]

首先，求

\[ \mathcal{I}_{i}(\overline{\mathbf{w}}^{i}) = \left( \mathcal{O}_{i-1}(\overline{\mathbf{w}}^{i-1}), \mathbf{w}_i \right) \]

具有简单的分块对角结构

\[\begin{split} J_{\mathcal{I}_{i}}(\overline{\mathbf{w}}^{i}) = \begin{pmatrix} J_{\mathcal{O}_{i-1}}(\overline{\mathbf{w}}^{i-1}) & 0 \\ 0 & I_{r_i \times r_i} \end{pmatrix} \in \mathbb{R}^{(n_{i} + r_{i})\times(r_0 + \cdots + r_i)} \end{split}\]

因为 \(\mathcal{I}_{i}\) 的第一个块组件 \(\mathcal{O}_{i-1}(\overline{\mathbf{w}}^{i-1})\) 不依赖于 \(\mathbf{w}_i\)，而 \(\mathcal{I}_{i}\) 的第二个块组件 \(\mathbf{w}_i\) 不依赖于 \(\overline{\mathbf{w}}^{i-1}\)。注意，这是一个相当大的矩阵，其列数特别是随着 \(i\) 的增加而增长。最后一个公式适用于 \(i \geq 1\)。当 \(i=0\) 时，我们有 \(\mathcal{I}_{0}(\overline{\mathbf{w}}^{0}) = \left(\mathbf{x}, \mathbf{w}_0\right)\)，因此

\[\begin{split} J_{\mathcal{I}_{0}}(\overline{\mathbf{w}}^{0}) = \begin{pmatrix} \mathbf{0}_{d \times r_0} \\ I_{r_0 \times r_0} \end{pmatrix} \in \mathbb{R}^{(d+ r_0) \times r_0}. \end{split}\]

我们同样将 \(\bfg_i(\mathbf{z}_i, \mathbf{w}_i)\) 的雅可比矩阵进行分块，即，将其分为对应于对 \(\mathbf{z}_{i}\) 的偏导数（对应的块用 \(A_i\) 表示）和对应于 \(\mathbf{w}_i\) 的偏导数（对应的块用 \(B_i\) 表示）的列

\[ J_{\bfg_i}(\mathbf{z}_i, \mathbf{w}_i) = \begin{pmatrix} A_i & B_i \end{pmatrix} \in \mathbb{R}^{n_{i+1} \times (n_i + r_i)}, \]

在 \((\mathbf{z}_i, \mathbf{w}_i) = \mathcal{I}_{i}(\overline{\mathbf{w}}^{i}) = (\mathcal{O}_{i-1}(\overline{\mathbf{w}}^{i-1}), \mathbf{w}_i)\) 处进行评估。注意，\(A_i\) 和 \(B_i\) 依赖于函数 \(\bfg_i\) 的细节，通常这个函数相当简单。我们将在下一节给出例子。

将上述内容代入，我们得到

\[\begin{split} J_{\mathcal{O}_{i}}(\overline{\mathbf{w}}^{i}) = \begin{pmatrix} A_i & B_i \end{pmatrix} \,\begin{pmatrix} J_{\mathcal{O}_{i-1}}(\overline{\mathbf{w}}^{i-1}) & 0 \\ 0 & I_{r_i \times r_i} \end{pmatrix}. \end{split}\]

这导致递归

\[ J_{\mathcal{O}_{i}}(\overline{\mathbf{w}}^{i}) = \begin{pmatrix} A_i \, J_{\mathcal{O}_{i-1}}(\overline{\mathbf{w}}^{i-1}) & B_i \end{pmatrix} \in \mathbb{R}^{n_{i+1}\times(r_0 + \cdots + r_i)} \]

从中可以计算出 \(\mathbf{h}(\mathbf{w})\) 的雅可比矩阵。像 \(J_{\mathcal{I}_{i}}\) 一样，\(J_{\mathcal{O}_{i}}\) 也是一个大矩阵。我们将这个矩阵方程称为 *基本递归*。

基本情况 \(i=0\) 是

\[\begin{split} J_{\mathcal{O}_{0}}(\overline{\mathbf{w}}^{0}) = \begin{pmatrix} A_0 & B_0 \end{pmatrix}\begin{pmatrix} \mathbf{0}_{d \times r_0} \\ I_{r_0 \times r_0} \end{pmatrix} = B_0. \end{split}\]

最后，再次使用**链式法则**

\[\begin{align*} \nabla {f(\mathbf{w})} &= J_{f}(\mathbf{w})^T\\ &= [J_{\ell}(\bfh(\mathbf{w})) \,J_{\bfh}(\mathbf{w})]^T\\ &= J_{\bfh}(\mathbf{w})^T \,\nabla {\ell}(\bfh(\mathbf{w}))\\ &= J_{\mathcal{O}_{L}}(\overline{\mathbf{w}}^{L})^T \,\nabla {\ell}(\mathcal{O}_{L}(\overline{\mathbf{w}}^{L})). \end{align*}\]

矩阵 \(J_{\mathcal{O}_{L}}(\overline{\mathbf{w}}^{L})\) 是通过上述递归计算得到的，而 \(\nabla {\ell}\) 取决于函数 \(\ell\)。

**反向传播** \(\idx{backpropagation}\xdi\) 我们利用基本的递归来计算 \(\bfh\) 的梯度。正如我们所见，有两种方法来做这件事。直接应用递归是其中之一，但它需要许多矩阵-矩阵乘法。最初的几步是

\[ J_{\mathcal{O}_{0}}(\overline{\mathbf{w}}^{0}) = B_0, \]\[ J_{\mathcal{O}_{1}}(\overline{\mathbf{w}}^{1}) = \begin{pmatrix} A_1 J_{\mathcal{O}_{0}}(\overline{\mathbf{w}}^{0}) & B_1 \end{pmatrix} \]\[ J_{\mathcal{O}_{2}}(\overline{\mathbf{w}}^{2}) = \begin{pmatrix} A_2 \, J_{\mathcal{O}_{1}}(\overline{\mathbf{w}}^{1}) & B_2 \end{pmatrix}, \]

等等。

相反，正如对输入 \(\mathbf{x}\) 求导的情况一样，也可以反向运行递归。后一种方法可能要快得多，因为，正如我们下面将要详细说明的，它只涉及矩阵-向量乘法。从最后一步开始，即从以下方程开始

\[ \nabla {f}(\mathbf{w}) = J_{\bfh}(\mathbf{w})^T \,\nabla {\ell}(\bfh(\mathbf{w})). \]

注意到 \(\nabla {\ell}(\bfh(\mathbf{w}))\) 是一个向量——不是一个矩阵。然后使用上述递归展开矩阵 \(J_{\bfh}(\mathbf{w})\)

\[\begin{align*} \nabla {f}(\mathbf{w}) &= J_{\bfh}(\mathbf{w})^T \,\nabla {\ell}(\bfh(\mathbf{w}))\\ &= J_{\mathcal{O}_{L}}(\overline{\mathbf{w}}^{L})^T \,\nabla {\ell}(\bfh(\mathbf{w}))\\ &= \begin{pmatrix} A_L \, J_{\mathcal{O}_{L-1}}(\overline{\mathbf{w}}^{L-1}) & B_L \end{pmatrix}^T \,\nabla {\ell}(\bfh(\mathbf{w}))\\ &= \begin{pmatrix} J_{\mathcal{O}_{L-1}}(\overline{\mathbf{w}}^{L-1})^T A_L^T \\ B_L^T \end{pmatrix} \,\nabla {\ell}(\bfh(\mathbf{w}))\\ &= \begin{pmatrix} J_{\mathcal{O}_{L-1}}(\overline{\mathbf{w}}^{L-1})^T \left\{ A_L^T \,\nabla {\ell}(\bfh(\mathbf{w}))\right\} \\ B_L^T \,\nabla {\ell}(\bfh(\mathbf{w})) \end{pmatrix}. \end{align*}\]

关键在于，两个表达式 \(A_L^T \,\nabla {\ell}(\bfh(\mathbf{w}))\) 和 \(B_L^T \,\nabla {\ell}(\bfh(\mathbf{w}))\) 都是**矩阵-向量乘法**。这种模式在递归的下一级也持续存在。注意，这假设我们首先已经预计算了 \(\bfh(\mathbf{w})\)。

在下一级，我们使用基本递归展开矩阵 \(J_{\mathcal{O}_{L-1}}(\overline{\mathbf{w}}^{L-1})^T\)。

\[\begin{align*} \nabla {f}(\mathbf{w}) &= \begin{pmatrix} J_{\mathcal{O}_{L-1}}(\overline{\mathbf{w}}^{L-1})^T \left\{ A_L^T \,\nabla {\ell}(\bfh(\mathbf{w}))\right\} \\ B_L^T \,\nabla {\ell}(\bfh(\mathbf{w})) \end{pmatrix}\\ &= \begin{pmatrix} \begin{pmatrix} A_{L-1} \, J_{\mathcal{O}_{L-2}}(\overline{\mathbf{w}}^{L-2}) & B_{L-1} \end{pmatrix}^T \left\{ A_L^T \,\nabla {\ell}(\bfh(\mathbf{w}))\right\} \\ B_L^T \,\nabla {\ell}(\bfh(\mathbf{w})) \end{pmatrix}\\ &= \begin{pmatrix} \begin{pmatrix} J_{\mathcal{O}_{L-2}}(\overline{\mathbf{w}}^{L-2})\,A_{L-1}^T \\ B_{L-1}^T \end{pmatrix} \left\{ A_L^T \,\nabla {\ell}(\bfh(\mathbf{w}))\right\} \\ B_L^T \,\nabla {\ell}(\bfh(\mathbf{w})) \end{pmatrix}\\ &= \begin{pmatrix} J_{\mathcal{O}_{L-2}}(\overline{\mathbf{w}}^{L-2})\left\{A_{L-1}^T \left\{ A_L^T \,\nabla {\ell}(\bfh(\mathbf{w}))\right\}\right\} \\ B_{L-1}^T \left\{ A_L^T \,\nabla {\ell}(\bfh(\mathbf{w}))\right\}\\ B_L^T \,\nabla {\ell}(\bfh(\mathbf{w})) \end{pmatrix}. \end{align*}\]

通过归纳法继续推导给出了 \(f\) 的梯度的另一种公式。

事实上，下一级给出

\[\begin{align*} \nabla {f}(\mathbf{w}) &= \begin{pmatrix} J_{\mathcal{O}_{L-3}}(\overline{\mathbf{w}}^{L-3})\left\{A_{L-2}^T \left\{A_{L-1}^T \left\{ A_L^T \,\nabla {\ell}(\bfh(\mathbf{w}))\right\}\right\}\right\} \\ B_{L-2}^T \left\{A_{L-1}^T \left\{ A_L^T \,\nabla {\ell}(\bfh(\mathbf{w}))\right\}\right\} \\ B_{L-1}^T \left\{ A_L^T \,\nabla {\ell}(\bfh(\mathbf{w}))\right\}\\ B_L^T \,\nabla {\ell}(\bfh(\mathbf{w})) \end{pmatrix}. \end{align*}\]

以此类推。注意，我们实际上不需要计算大的矩阵 \(J_{\mathcal{O}_{i}}\) - 只需要计算向量序列 \(B_L^T \,\nabla {\ell}(\bfh(\mathbf{w}))\), \(B_{L-1}^T \left\{ A_L^T \,\nabla {\ell}(\bfh(\mathbf{w}))\right\}\), \(B_{L-2}^T \left\{A_{L-1}^T \left\{ A_L^T \,\nabla {\ell}(\bfh(\mathbf{w}))\right\}\right\}\)，等等。

这些公式可能看起来有些繁琐，但它们具有直观的形式。矩阵 \(A_i\) 是雅可比矩阵 \(J_{\bfg_i}\) 的子矩阵，仅对应于对 \(\mathbf{z}_i\) 的偏导数，即来自前一层的输入。矩阵 \(B_i\) 是雅可比矩阵 \(J_{\bfg_i}\) 的子矩阵，仅对应于对 \(\mathbf{w}_i\) 的偏导数，即特定层的参数。为了计算与第 \(i+1\) 层的参数 \(\mathbf{w}_i\) 对应的 \(\nabla f\) 的子向量，我们从最后一个开始，通过乘以相应的 \(A_j^T\) 对前一层的输入进行反复求导，直到达到 \(i+1\) 层，此时我们通过对特定层参数求偏导数（通过乘以 \(B_i^T\)）来停止这个过程。这个过程在这里停止，因为它之前的层不依赖于 \(\mathbf{w}_i\)，因此其完全影响 \(f\) 已经被考虑在内。

换句话说，我们需要计算

\[ \mathbf{p}_{L} := A_L^T \,\nabla {\ell}(\bfh(\mathbf{w})), \]

以及

\[ \mathbf{q}_{L} := B_L^T \,\nabla {\ell}(\bfh(\mathbf{w})), \]

然后

\[ \mathbf{p}_{L-1} := A_{L-1}^T \mathbf{p}_{L} = A_{L-1}^T \left\{ A_L^T \,\nabla {\ell}(\bfh(\mathbf{w}))\right\} \]

以及

\[ \mathbf{q}_{L-1} := B_{L-1}^T \mathbf{p}_{L} = B_{L-1}^T \left\{ A_L^T \,\nabla {\ell}(\bfh(\mathbf{w}))\right\}, \]

然后

\[ \mathbf{p}_{L-2} := A_{L-2}^T \mathbf{p}_{L-1} = A_{L-2}^T \left\{A_{L-1}^T \left\{ A_L^T \,\nabla {\ell}(\bfh(\mathbf{w}))\right\}\right\} \]

以及

\[ \mathbf{q}_{L-2} := B_{L-2}^T \mathbf{p}_{L-1} = B_{L-2}^T \left\{A_{L-1}^T \left\{ A_L^T \,\nabla {\ell}(\bfh(\mathbf{w}))\right\} \right\}, \]

以此类推。\(\mathbf{p}_i\) 被称为伴随项；它们对应于 \(f\) 对 \(\mathbf{z}_i\) 的偏导数向量。

需要注意的一个细节是，矩阵 \(A_i, B_i\) 依赖于层 \(i-1\) 的输出。为了计算它们，我们首先进行正向传播，即我们令 \(\mathbf{z}_0 = \mathbf{x}\) 然后

\[ \mathbf{z}_1 = \mathcal{O}_{0}(\overline{\mathbf{w}}^{0}) = \bfg_0(\mathbf{z}_0, \mathbf{w}_0), \]\[ \mathbf{z}_2 = \mathcal{O}_{1}(\overline{\mathbf{w}}^{1}) = \bfg_1(\mathcal{O}_{0}(\overline{\mathbf{w}}^{0}), \mathbf{w}_1) = \bfg_1(\mathbf{z}_1, \mathbf{w}_1), \]

以此类推。在正向传播过程中，我们还沿途计算 \(A_i, B_i\)。

现在我们给出完整的算法，它涉及两个阶段。在正向传播阶段，或正向传播步骤中，我们计算以下内容。

*初始化:*

\[\mathbf{z}_0 := \mathbf{x}\]

*正向层循环:* 对于 \(i = 0, 1,\ldots,L\),

\[\begin{align*} \mathbf{z}_{i+1} &:= \bfg_i(\mathbf{z}_i, \mathbf{w}_i)\\ \begin{pmatrix} A_i & B_i \end{pmatrix} &:= J_{\bfg_i}(\mathbf{z}_i, \mathbf{w}_i) \end{align*}\]

*损失:*

\[\begin{align*} z_{L+2} &:= \ell(\mathbf{z}_{L+1})\\ \mathbf{p}_{L+1} &:= \nabla {\ell}(\mathbf{z}_{L+1}). \end{align*}\]

在反向传播阶段，或反向传播步骤中，我们计算以下内容。

*反向层循环:* 对于 \(i = L,\ldots,1, 0\)，

\[\begin{align*} \mathbf{p}_{i} &:= A_i^T \mathbf{p}_{i+1}\\ \mathbf{q}_{i} &:= B_i^T \mathbf{p}_{i+1} \end{align*}\]

*输出:*

\[ \nabla f(\mathbf{w}) = (\mathbf{q}_0,\mathbf{q}_1,\ldots,\mathbf{q}_L). \]

注意，实际上我们不需要计算 \(A_0\) 和 \(\mathbf{p}_0\)。

**示例：** **（继续）** 我们将算法应用于我们的运行示例。从前面的计算中，对于 \(i = 0, 1,\ldots,L\)，雅可比矩阵为

\[\begin{align*} J_{\bfg_i}(\mathbf{z}_i, \mathbf{w}_i) &= \begin{pmatrix} \mathbb{A}_{n_{i+1}}[\mathbf{w}_i] & \mathbb{B}_{n_{i+1}}[\mathbf{z}_i] \end{pmatrix}\\ &= \begin{pmatrix} \mathcal{W}_i & I_{n_{i+1} \times n_{i+1}} \otimes \mathbf{z}_i^T \end{pmatrix}\\ &=: \begin{pmatrix} A_i & B_i \end{pmatrix} \end{align*}\]

以及

\[ J_{\ell}(\hat{\mathbf{y}}) = (\hat{\mathbf{y}} - \mathbf{y})^T. \]

利用**克罗内克积的性质**，我们得到

\[ \mathbf{p}_{L} := A_L^T \,\nabla {\ell}(\bfh(\mathbf{w})) = \mathcal{W}_L^T (\hat{\mathbf{y}} - \mathbf{y}) \]\[\begin{align*} \mathbf{q}_{L} &:= B_L^T \,\nabla {\ell}(\bfh(\mathbf{w})) = (I_{n_{L+1} \times n_{L+1}} \otimes \mathbf{z}_L^T)^T (\hat{\mathbf{y}} - \mathbf{y}) = (\hat{\mathbf{y}} - \mathbf{y}) \otimes \mathbf{z}_L\\ &= (\hat{\mathbf{y}} - \mathbf{y}) \otimes \mathcal{W}_{L-1} \cdots \mathcal{W}_{1} \mathcal{W}_{0} \mathbf{x} \end{align*}\]\[ \mathbf{p}_{L-1} := A_{L-1}^T \mathbf{p}_{L} = \mathcal{W}_{L-1}^T \mathcal{W}_L^T (\hat{\mathbf{y}} - \mathbf{y}) \]\[\begin{align*} \mathbf{q}_{L-1} &:= B_{L-1}^T \mathbf{p}_{L} = (I_{n_{L} \times n_{L}} \otimes \mathbf{z}_{L-1}^T)^T \mathcal{W}_L^T (\hat{\mathbf{y}} - \mathbf{y}) = \mathcal{W}_L^T (\hat{\mathbf{y}} - \mathbf{y}) \otimes \mathbf{z}_{L-1}\\ &= \mathcal{W}_L^T (\hat{\mathbf{y}} - \mathbf{y}) \otimes \mathcal{W}_{L-2} \cdots \mathcal{W}_{1} \mathcal{W}_{0} \mathbf{x} \end{align*}\]\[ \mathbf{p}_{L-2} := A_{L-2}^T \mathbf{p}_{L-1} = \mathcal{W}_{L-2}^T \mathcal{W}_{L-1}^T \mathcal{W}_L^T (\hat{\mathbf{y}} - \mathbf{y}) \]\[\begin{align*} \mathbf{q}_{L-2} &:= B_{L-2}^T \mathbf{p}_{L-1} = (I_{n_{L-1} \times n_{L-1}} \otimes \mathbf{z}_{L-2}^T)^T \mathcal{W}_{L-1}^T \mathcal{W}_L^T (\hat{\mathbf{y}} - \mathbf{y}) = \mathcal{W}_{L-1}^T \mathcal{W}_L^T (\hat{\mathbf{y}} - \mathbf{y}) \otimes \mathbf{z}_{L-2}\\ &= \mathcal{W}_{L-1}^T \mathcal{W}_L^T (\hat{\mathbf{y}} - \mathbf{y}) \otimes \mathcal{W}_{L-3} \cdots \mathcal{W}_{1} \mathcal{W}_{0} \mathbf{x} \end{align*}\]

以此类推。按照模式，最后一步是

\[ \mathbf{p}_1 := \mathcal{W}_{1}^T \cdots \mathcal{W}_{L-1}^T \mathcal{W}_L^T (\hat{\mathbf{y}} - \mathbf{y}) \]\[ \mathbf{q}_0 := B_{0}^T \mathbf{p}_{1} = \mathcal{W}_{1}^T \cdots \mathcal{W}_{L-1}^T \mathcal{W}_L^T (\hat{\mathbf{y}} - \mathbf{y}) \otimes \mathbf{x}. \]

这些计算与之前推导的 \(L=1\) 的情况一致（检查一下！）。\(\lhd\)

**CHAT & LEARN** 反向传播的效率是深度学习成功的关键。向你的最爱人工智能聊天机器人询问反向传播的历史及其在现代深度学习发展中的作用。\(\ddagger\)

***自我评估测验*** *(由 Claude, Gemini 和 ChatGPT 协助)*

**1** 在反向传播算法中，“正向传播”计算了什么？

a) 每一层 \(i\) 的伴随 \(\mathbf{p}_i\)。

b) 每一层 \(i\) 参数的梯度 \(\mathbf{q}_i\)。

c) 每一层 \(i\) 的函数值 \(\mathbf{z}_i\) 和雅可比矩阵 \(A_i, B_i\)。

d) 关于所有参数的最终梯度 \(\nabla f(\mathbf{w})\)。

**2** 反向传播算法中“反向传播”的目的是什么？

a) 计算每一层 \(i\) 的函数值 \(\mathbf{z}_i\)，从输入 \(\mathbf{x}\) 开始。

b) 使用基本递归计算每一层 \(i\) 的雅可比矩阵 \(A_i, B_i\)。

c) 使用基本递归计算每个层 \(i\) 的伴随 \(\mathbf{p}_i\) 和梯度 \(\mathbf{q}_i\)。

d) 为了计算渐进函数的最终输出 \(\ell(\mathbf{z}_{L+1})\)。

**3** 从层数 \(L\) 和矩阵维度 \(m\) 的角度来看，反向传播算法的计算复杂度是什么？

a) \(\approx Lm\)

b) \(\approx Lm²\)

c) \(\approx Lm²d\)

d) \(\approx Lm³d\)

**4** 在渐进函数的背景下，矩阵 \(A_i\) 和 \(B_i\) 有什么意义？

a) 它们分别代表层函数相对于输入和参数的雅可比矩阵。

b) 它们是在前向传递过程中计算的中间值。

c) 它们是在反向传播算法中使用的伴随。

d) 它们是每层的参数矩阵。

**5** 在渐进函数的背景下，以下哪项最好地描述了向量 \(\mathbf{w}_i\) 的作用？

a) 第 \(i\) 层的输入。

b) 第 \(i\) 层的输出。

c) 第 \(i\) 层的特定参数。

d) 所有层到 \(i\) 的参数的连接。

1 的答案：c. 证明：本节介绍了前向传播步骤，该步骤计算“以下内容：初始化：\(\mathbf{z}_0 := \mathbf{x}\) 前向层循环：对于 \(i=0,1,\dots,L\)，\(\mathbf{z}_{i+1} := \mathbf{g}_i(\mathbf{z}_i, \mathbf{w}_i)\) \((A_i,B_i) := J_{\mathbf{g}_i}(\mathbf{z}_i, \mathbf{w}_i)\) 损失：\(\mathbf{z}_{L+2} := \ell(\mathbf{z}_{L+1})\)”

2 的答案：c. 证明：反向传递被描述如下：“反向层循环：对于 \(i=L,\dots,1,0\)，\(\mathbf{p}_i := A_i^T \mathbf{p}_{i+1}\) \(\mathbf{q}_i := B_i^T \mathbf{p}_{i+1}\) 输出：\(\nabla f(\mathbf{w}) = (\mathbf{q}_0, \mathbf{q}_1, \dots, \mathbf{q}_L)\)”。

3 的答案：b. 证明：文本推导出反向模式的操作数大约为 \(2Lm²\)，声明“这大约是 \(2Lm²\) – 这可以比 \(2Lm²d\) 小得多！”

4 的答案：a. 证明：文本将 \(A_i\) 和 \(B_i\) 定义为雅可比矩阵 \(J_{\mathbf{g}_i}(\mathbf{z}_i, \mathbf{w}_i)\) 对应于相对于 \(\mathbf{z}_i\) 和 \(\mathbf{w}_i\) 的偏导数的块。

5 的答案：c. 证明：文本解释说：“在机器学习背景下，每个“层”\(\mathbf{g}_i\)都有参数（在我们的运行示例中，是\(\mathcal{W}_i\)的条目），我们试图优化这些参数。”
