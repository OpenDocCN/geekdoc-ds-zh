# 8.4\. 人工智能构建块 2：随机梯度下降

> 原文：[`mmids-textbook.github.io/chap08_nn/04_sgd/roch-mmids-nn-sgd.html`](https://mmids-textbook.github.io/chap08_nn/04_sgd/roch-mmids-nn-sgd.html)

在展示了如何计算梯度之后，我们现在可以应用梯度下降来拟合数据。

为了获得完整的梯度，我们考虑 $n$ 个样本 $(\mathbf{x}_i,y_i)$，$i=1,\ldots,n$。在此点，我们使 $(\mathbf{x}_i, y_i)$ 的依赖关系明确。损失函数可以取为单个样本贡献的平均值，因此梯度通过线性获得

$$ \nabla \left(\frac{1}{n} \sum_{i=1}^n f_{\mathbf{x}_i,y_i}(\mathbf{w})\right) = \frac{1}{n} \sum_{i=1}^n \nabla f_{\mathbf{x}_i,y_i}(\mathbf{w}), $$

其中每个项都可以通过上述程序单独计算。

然后，我们可以应用梯度下降。我们从任意 $\mathbf{w}^{0}$ 开始并按以下方式更新

$$ \mathbf{w}^{t+1} = \mathbf{w}^{t} - \alpha_t \left(\frac{1}{n} \sum_{i=1}^n \nabla f_{\mathbf{x}_i,y_i}(\mathbf{w}^{t})\right). $$

在大型数据集中，对所有样本求和可能过于昂贵。我们提出了一种流行的替代方案。

## 8.4.1\. 算法#

在 [随机梯度下降](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) (SGD)$\idx{随机梯度下降}\xdi$，梯度下降的一种变体中，我们从 $\{1,\ldots,n\}$ 中随机均匀地选择一个样本 $I_t$ 并按以下方式更新

$$ \mathbf{w}^{t+1} = \mathbf{w}^{t} - \alpha_t \nabla f_{\mathbf{x}_{I_t},y_{I_t}}(\mathbf{w}^{t}). $$

更一般地，在所谓的 mini-batch 版本的 SGD 中，我们选择一个大小为 $b$ 的均匀随机子样本 $\mathcal{B}_t \subseteq \{1,\ldots,n\}$ 而不是替换（即，该大小所有子样本被选中的概率相同）

$$ \mathbf{w}^{t+1} = \mathbf{w}^{t} - \alpha_t \frac{1}{b} \sum_{i\in \mathcal{B}_t} \nabla f_{\mathbf{x}_i,y_i}(\mathbf{w}^{t}). $$

关于上述两个随机更新的关键观察是，在期望上，它们执行了一步梯度下降。这证明是足够的，并且具有计算优势。

**引理** 设定一个批大小 $1 \leq b \leq n$ 和一个任意的参数向量 $\mathbf{w}$。设 $\mathcal{B} \subseteq \{1,\ldots,n\}$ 是大小为 $b$ 的均匀随机子样本。那么

$$ \mathbb{E}\left[\frac{1}{b} \sum_{i\in \mathcal{B}} \nabla f_{\mathbf{x}_i,y_i}(\mathbf{w})\right] = \frac{1}{n} \sum_{i=1}^n \nabla f_{\mathbf{x}_i,y_i}(\mathbf{w}). $$

$\flat$

*证明* 因为 $\mathcal{B}$ 是随机均匀选择（不替换），对于任何大小为 $b$ 的不重复子样本 $B \subseteq \{1,\ldots,n\}$

$$ \mathbb{P}[\mathcal{B} = B] = \frac{1}{\binom{n}{b}}. $$

因此，对所有这样的子样本求和，我们得到

$$\begin{align*} \mathbb{E}\left[\frac{1}{b} \sum_{i\in \mathcal{B}} \nabla f_{\mathbf{x}_i,y_i}(\mathbf{w})\right] &= \sum_{B \subseteq \{1,\ldots,n\}} \mathbb{P}[\mathcal{B} = B] \frac{1}{b} \sum_{i\in B} \nabla f_{\mathbf{x}_i,y_i}(\mathbf{w})\\ &= \sum_{B \subseteq \{1,\ldots,n\}} \frac{1}{\binom{n}{b}} \frac{1}{b} \sum_{i=1}^n \mathbf{1}\{i \in B\} \nabla f_{\mathbf{x}_i,y_i}(\mathbf{w})\\ &= \sum_{i=1}^n \nabla f_{\mathbf{x}_i,y_i}(\mathbf{w}) \frac{1}{b \binom{n}{b}} \sum_{B \subseteq \{1,\ldots,n\}} \mathbf{1}\{i \in B\}. \end{align*}$$

计算内部和需要组合论证。实际上，$\sum_{B \subseteq \{1,\ldots,n\}} \mathbf{1}\{i \in B\}$ 计算了 $i$ 在大小为 $b$ 的子样本中不重复被选择的方式数。那就是 $\binom{n-1}{b-1}$，这是从其他 $n-1$ 个可能元素中选择 $B$ 的剩余 $b-1$ 个元素的方式数。根据二项系数的定义和阶乘的性质，

$$ \frac{\binom{n-1}{b-1}}{b \binom{n}{b}} = \frac{\frac{(n-1)!}{(b-1)! (n-b)!}}{b \frac{n!}{b! (n-b)!}} = \frac{(n-1)!}{n!} \frac{b!}{b (b-1)!} = \frac{1}{n}. $$

将其代入给出结论。 $\square$

作为第一个示例，我们回到逻辑回归$\idx{logistic regression}\xdi$。回想一下，输入数据的形式是 $\{(\boldsymbol{\alpha}_i, b_i) : i=1,\ldots, n\}$，其中 $\boldsymbol{\alpha}_i = (\alpha_{i,1}, \ldots, \alpha_{i,d}) \in \mathbb{R}^d$ 是特征，$b_i \in \{0,1\}$ 是标签。和之前一样，我们使用矩阵表示：$A \in \mathbb{R}^{n \times d}$ 的行是 $\boldsymbol{\alpha}_i^T$，$i = 1,\ldots, n$，而 $\mathbf{b} = (b_1, \ldots, b_n) \in \{0,1\}^n$。我们想要解决最小化问题

$$ \min_{\mathbf{x} \in \mathbb{R}^d} \ell(\mathbf{x}; A, \mathbf{b}) $$

其中损失是

$$\begin{align*} \ell(\mathbf{x}; A, \mathbf{b}) &= \frac{1}{n} \sum_{i=1}^n \left\{- b_i \log(\sigma(\boldsymbol{\alpha_i}^T \mathbf{x})) - (1-b_i) \log(1- \sigma(\boldsymbol{\alpha_i}^T \mathbf{x}))\right\}\\ &= \mathrm{mean}\left(-\mathbf{b} \odot \mathbf{log}(\bsigma(A \mathbf{x})) - (\mathbf{1} - \mathbf{b}) \odot \mathbf{log}(\mathbf{1} - \bsigma(A \mathbf{x}))\right). \end{align*}$$

前面已经计算了梯度

$$\begin{align*} \nabla\ell(\mathbf{x}; A, \mathbf{b}) &= - \frac{1}{n} \sum_{i=1}^n ( b_i - \sigma(\boldsymbol{\alpha}_i^T \mathbf{x}) ) \,\boldsymbol{\alpha}_i\\ &= -\frac{1}{n} A^T [\mathbf{b} - \bsigma(A \mathbf{x})]. \end{align*}$$

对于 SGD 的小批量版本，我们选择一个大小为 $B$ 的随机子样本 $\mathcal{B}_t \subseteq \{1,\ldots,n\}$，并采取以下步骤

$$ \mathbf{x}^{t+1} = \mathbf{x}^{t} +\beta \frac{1}{B} \sum_{i\in \mathcal{B}_t} ( b_i - \sigma(\boldsymbol{\alpha}_i^T \mathbf{x}^t) ) \,\boldsymbol{\alpha}_i. $$

我们修改了之前用于逻辑回归的代码。唯一的改变是选择一个随机的小批量，将其作为数据集输入到下降更新子例程中。

```py
def sigmoid(z): 
    return 1/(1+np.exp(-z))

def pred_fn(x, A): 
    return sigmoid(A @ x)

def loss_fn(x, A, b): 
    return np.mean(-b*np.log(pred_fn(x, A)) - (1 - b)*np.log(1 - pred_fn(x, A)))

def grad_fn(x, A, b):
    return -A.T @ (b - pred_fn(x, A))/len(b)

def desc_update_for_logreg(grad_fn, A, b, curr_x, beta):
    gradient = grad_fn(curr_x, A, b)
    return curr_x - beta*gradient

def sgd_for_logreg(rng, loss_fn, grad_fn, A, b, 
                   init_x, beta=1e-3, niters=int(1e5), batch=40):

    curr_x = init_x
    nsamples = len(b)
    for _ in range(niters):
        I = rng.integers(nsamples, size=batch)
        curr_x = desc_update_for_logreg(
            grad_fn, A[I,:], b[I], curr_x, beta)

    return curr_x 
```

**数值角:** 我们分析了来自 [[ESL](https://web.stanford.edu/~hastie/ElemStatLearn/)] 的数据集，该数据集可以在此 [下载](https://web.stanford.edu/~hastie/ElemStatLearn/data.html)。引用 [[ESL](https://web.stanford.edu/~hastie/ElemStatLearn/)]，第 4.4.2 节

> 数据 [...] 是南非开普敦西部三个农村地区的冠状动脉风险因素研究（CORIS）基线调查的一个子集（Rousseauw 等人，1983 年）。该研究的目的是确定该高发病率地区缺血性心脏病风险因素的强度。数据代表 15 至 64 岁的白人男性，应变量是在调查时心肌梗死（MI）的存在或不存在（该地区 MI 的总体患病率为 5.1%）。我们的数据集中有 160 个病例，以及 302 个对照组。这些数据在 Hastie 和 Tibshirani（1987 年）的书中描述得更详细。

我们加载了数据，我们对数据进行了一些格式上的调整，并查看了一个摘要。

```py
data = pd.read_csv('SAHeart.csv')
data.head() 
```

|  | sbp | tobacco | ldl | adiposity | typea | obesity | alcohol | age | chd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 160.0 | 12.00 | 5.73 | 23.11 | 49.0 | 25.30 | 97.20 | 52.0 | 1.0 |
| 1 | 144.0 | 0.01 | 4.41 | 28.61 | 55.0 | 28.87 | 2.06 | 63.0 | 1.0 |
| 2 | 118.0 | 0.08 | 3.48 | 32.28 | 52.0 | 29.14 | 3.81 | 46.0 | 0.0 |
| 3 | 170.0 | 7.50 | 6.41 | 38.03 | 51.0 | 31.99 | 24.26 | 58.0 | 1.0 |
| 4 | 134.0 | 13.60 | 3.50 | 27.78 | 60.0 | 25.99 | 57.34 | 49.0 | 1.0 |

我们的目的是根据其他变量（这些变量简要描述 [在此](https://web.stanford.edu/~hastie/ElemStatLearn/datasets/SAheart.info.txt)）预测 `chd`，即冠状动脉心脏病。我们再次使用逻辑回归。

我们首先构建数据矩阵。我们只使用了三个预测变量。

```py
feature = data[['tobacco', 'ldl', 'age']].to_numpy()
print(feature) 
```

```py
[[1.200e+01 5.730e+00 5.200e+01]
 [1.000e-02 4.410e+00 6.300e+01]
 [8.000e-02 3.480e+00 4.600e+01]
 ...
 [3.000e+00 1.590e+00 5.500e+01]
 [5.400e+00 1.161e+01 4.000e+01]
 [0.000e+00 4.820e+00 4.600e+01]] 
```

```py
label = data['chd'].to_numpy()
A = np.concatenate((np.ones((len(label),1)),feature),axis=1)
b = label 
```

我们尝试了小批量随机梯度下降法（mini-batch SGD）。

```py
seed = 535
rng = np.random.default_rng(seed)
init_x = np.zeros(A.shape[1])
best_x = sgd_for_logreg(rng, loss_fn, grad_fn, A, b, init_x, beta=1e-3, niters=int(1e6))
print(best_x) 
```

```py
[-4.06558071  0.07990955  0.18813635  0.04693118] 
```

结果更难以可视化。为了了解结果的准确性，我们将我们的预测与真实标签进行比较。通过预测，我们指的是当 $\sigma(\boldsymbol{\alpha}^T \mathbf{x}) > 1/2$ 时，我们预测标签 $1$。我们在训练集上尝试了这种方法。（更好的方法是将数据分成训练集和测试集，但在这里我们不会这样做。）

```py
def logis_acc(x, A, b):
    return np.sum((pred_fn(x, A) > 0.5) == b)/len(b) 
```

```py
logis_acc(best_x, A, b) 
```

```py
0.7207792207792207 
```

$\unlhd$

## 8.4.2. 示例：多项逻辑回归#

我们给出了渐进函数的一个具体例子，以及反向传播和随机梯度下降的应用。

回想一下，一个分类器 $h$ 接受 $\mathbb{R}^d$ 中的输入并预测 $K$ 个可能标签中的一个。下面将要变得清楚的原因是，使用 [独热编码](https://en.wikipedia.org/wiki/One-hot)$\idx{one-hot encoding}\xdi$ 的标签将很方便。也就是说，我们将标签 $i$ 编码为 $K$ 维向量 $\mathbf{e}_i$。在这里，像往常一样，$\mathbf{e}_i$ 是 $\mathbb{R}^K$ 的标准基，即一个 $1$ 在条目 $i$ 上，其他地方为 $0$ 的向量。此外，我们允许分类器的输出是标签 $\{1,\ldots,K\}$ 的概率分布，即一个向量在

$$ \Delta_K = \left\{ (p_1,\ldots,p_K) \in [0,1]^K \,:\, \sum_{k=1}^K p_k = 1 \right\}. $$

注意到 $\mathbf{e}_i$ 本身可以被视为一个概率分布，它将概率 $1$ 分配给 $i$。

**多项式逻辑回归的背景** 我们使用 [多项式逻辑回归](https://en.wikipedia.org/wiki/Multinomial_logistic_regression)$\idx{multinomial logistic regression}\xdi$ 来学习 $K$ 个标签的分类器。在多项式逻辑回归中，我们再次使用输入数据的仿射函数。

这次，我们有 $K$ 个函数，每个函数输出与每个标签关联的分数。然后我们将这些分数转换成 $K$ 个标签的概率分布。有多种方法可以做到这一点。一种标准的方法是 [softmax 函数](https://en.wikipedia.org/wiki/Softmax_function)$\idx{softmax}\xdi$ $\bgamma = (\gamma_1,\ldots,\gamma_K)$：对于 $\mathbf{z} \in \mathbb{R}^K$

$$ \gamma_i(\mathbf{z}) = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}, \quad i=1,\ldots,K. $$

为了解释这个名称，注意到较大的输入被映射到较大的概率。

实际上，由于概率分布必须求和为 $1$，它由分配给前 $K-1$ 个标签的概率决定。换句话说，我们可以省略与最后一个标签相关的分数。但为了简化符号，我们在这里不会这样做。

对于每个 $k$，我们有一个回归函数

$$ \sum_{j=1}^d w^{(k)}_{j} x_{j} = \mathbf{x}_1^T \mathbf{w}^{(k)}, \quad k=1,\ldots,K $$

其中 $\mathbf{w} = (\mathbf{w}^{(1)},\ldots,\mathbf{w}^{(K)})$ 是参数，$\mathbf{w}^{(k)} \in \mathbb{R}^{d}$ 且 $\mathbf{x} \in \mathbb{R}^d$ 是输入。可以通过向 $\mathbf{x}$ 添加一个额外的条目 $1$ 来包含一个常数项。正如我们在线性回归案例中所做的那样，我们假设这种预处理已经完成。为了简化符号，我们让 $\mathcal{W} \in \mathbb{R}^{K \times d}$ 作为具有行 $(\mathbf{w}^{(1)})^T,\ldots,(\mathbf{w}^{(K)})^T$ 的矩阵。

分类器的输出是

$$\begin{align*} \bfh(\mathbf{w}) &= \bgamma\left(\mathcal{W} \mathbf{x}\right), \end{align*}$$

对于 $i=1,\ldots,K$，其中 $\bgamma$ 是 softmax 函数。注意，后者没有关联的参数。

剩下需要定义一个损失函数。为了量化拟合度，自然地使用概率测度之间的距离概念，这里是在输出 $\mathbf{h}(\mathbf{w}) \in \Delta_K$ 和正确标签 $\mathbf{y} \in \{\mathbf{e}_1,\ldots,\mathbf{e}_{K}\} \subseteq \Delta_K$ 之间。存在许多这样的度量。在多项式逻辑回归中，我们使用 Kullback-Leibler 散度，我们在最大似然估计的上下文中已经遇到过。回想一下，对于两个概率分布 $\mathbf{p}, \mathbf{q} \in \Delta_K$，它定义为

$$ \mathrm{KL}(\mathbf{p} \| \mathbf{q}) = \sum_{i=1}^K p_i \log \frac{p_i}{q_i} $$

其中我们只需限制在 $\mathbf{q} > \mathbf{0}$ 的情况下，并且我们使用约定 $0 \log 0 = 0$（这样 $p_i = 0$ 的项对总和的贡献为 $0$）。注意，$\mathbf{p} = \mathbf{q}$ 意味着 $\mathrm{KL}(\mathbf{p} \| \mathbf{q}) = 0$。我们之前已经证明了 $\mathrm{KL}(\mathbf{p} \| \mathbf{q}) \geq 0$，这是一个被称为 *吉布斯不等式* 的结果。

回到损失函数，我们使用恒等式 $\log\frac{\alpha}{\beta} = \log \alpha - \log \beta$ 来重新写

$$\begin{align*} \mathrm{KL}(\mathbf{y} \| \bfh(\mathbf{w})) &= \sum_{i=1}^K y_i \log \frac{y_i}{h_{i}(\mathbf{w})}\\ &= \sum_{i=1}^K y_i \log y_i - \sum_{i=1}^K y_i \log h_{i}(\mathbf{w}), \end{align*}$$

其中 $\bfh = (h_{1},\ldots,h_{K})$. 注意到右侧第一个项不依赖于 $\mathbf{w}$. 因此，当优化 $\mathrm{KL}(\mathbf{y} \| \bfh(\mathbf{w}))$ 时，我们可以忽略它。剩余的项是

$$ H(\mathbf{y}, \bfh(\mathbf{w})) = - \sum_{i=1}^K y_i \log h_{i}(\mathbf{w}). $$

我们用它来定义我们的损失函数。也就是说，我们设

$$ \ell(\hat{\mathbf{y}}) = H(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{i=1}^K y_i \log \hat{y}_{i}. $$

最后，

$$\begin{align*} f(\mathbf{w}) &= \ell(\bfh(\mathbf{w}))\\ &= H(\mathbf{y}, \bfh(\mathbf{w}))\\ &= H\left(\mathbf{y}, \bgamma\left(\mathcal{W} \mathbf{x}\right)\right)\\ &= - \sum_{i=1}^K y_i \log\gamma_i\left(\mathcal{W} \mathbf{x}\right). \end{align*}$$

**计算梯度** 我们应用了上一节中的前向和反向传播步骤。然后，我们使用这些递归关系推导出梯度的解析公式。

前向传播从初始化 $\mathbf{z}_0 := \mathbf{x}$ 开始。前向层循环有两个步骤。将 $\mathbf{w}_0 = (\mathbf{w}_0^{(1)},\ldots,\mathbf{w}_0^{(K)})$ 等于 $\mathbf{w} = (\mathbf{w}^{(1)},\ldots,\mathbf{w}^{(K)})$。首先我们计算

$$\begin{align*} \mathbf{z}_{1} &:= \bfg_0(\mathbf{z}_0,\mathbf{w}_0) = \mathcal{W}_0 \mathbf{z}_0\\ J_{\bfg_0}(\mathbf{z}_0,\mathbf{w}_0) &:=\begin{pmatrix} A_0 & B_0 \end{pmatrix} \end{align*}$$

其中我们定义 $\mathcal{W}_0 \in \mathbb{R}^{K \times d}$ 为具有行 $(\mathbf{w}_0^{(1)})^T,\ldots,(\mathbf{w}_0^{(K-1)})^T$ 的矩阵。我们之前已经计算了雅可比矩阵：

$$\begin{split} A_0 = \mathbb{A}_{K}[\mathbf{w}_0] = \mathcal{W}_0 = \begin{pmatrix} (\mathbf{w}^{(1)}_0)^T\\ \vdots\\ (\mathbf{w}^{(K)}_0)^T \end{pmatrix} \end{split}$$

和

$$ B_0 = \mathbb{B}_{K}[\mathbf{z}_0] = I_{K\times K} \otimes \mathbf{z}_0^T = \begin{pmatrix} \mathbf{e}_1 \mathbf{z}_0^T & \cdots & \mathbf{e}_{K}\mathbf{z}_0^T \end{pmatrix}. $$

在前向层循环的第二步中，我们计算

$$\begin{align*} \hat{\mathbf{y}} := \mathbf{z}_2 &:= \bfg_1(\mathbf{z}_1) = \bgamma(\mathbf{z}_1)\\ A_1 &:= J_{\bfg_1}(\mathbf{z}_1) = J_{\bgamma}(\mathbf{z}_1). \end{align*}$$

因此，我们需要计算 $\bgamma$ 的雅可比矩阵。我们将这个计算分为两种情况。当 $1 \leq i = j \leq K$ 时，

$$\begin{align*} (A_1)_{ii} &= \frac{\partial}{\partial z_{1,i}} \left[ \gamma_i(\mathbf{z}_1) \right]\\ &= \frac{\partial}{\partial z_{1,i}} \left[ \frac{e^{z_{1,i}}}{\sum_{k=1}^{K} e^{z_{1,k}}} \right]\\ &= \frac{e^{z_{1,i}}\left(\sum_{k=1}^{K} e^{z_{1,k}}\right) - e^{z_{1,i}}\left(e^{z_{1,i}}\right)} {\left(\sum_{k=1}^{K} e^{z_{1,k}}\right)²}\\ &= \gamma_i(\mathbf{z}_1) - \gamma_i(\mathbf{z}_1)², \end{align*}$$

通过[商规则](https://en.wikipedia.org/wiki/Quotient_rule)。

当 $1 \leq i, j \leq K$ 且 $i \neq j$ 时，

$$\begin{align*} (A_1)_{ij} &= \frac{\partial}{\partial z_{1,j}} \left[ \gamma_i(\mathbf{z}_1) \right]\\ &= \frac{\partial}{\partial z_{1,j}} \left[ \frac{e^{z_{1,i}}}{\sum_{k=1}^{K} e^{z_{1,k}}} \right]\\ &= \frac{- e^{z_{1,i}}\left(e^{z_{1,j}}\right)} {\left(\sum_{k=1}^{K} e^{z_{1,k}}\right)²}\\ &= - \gamma_i(\mathbf{z}_1)\gamma_j(\mathbf{z}_1). \end{align*}$$

在矩阵形式中，

$$ J_{\bgamma}(\mathbf{z}_1) = A_1 = \mathrm{diag}(\bgamma(\mathbf{z}_1)) - \bgamma(\mathbf{z}_1) \, \bgamma(\mathbf{z}_1)^T. $$

损失函数的雅可比矩阵是

$$ J_{\ell}(\hat{\mathbf{y}}) = \nabla \left[ - \sum_{i=1}^K y_i \log \hat{y}_{i} \right]^T = -\left(\frac{y_1}{\hat{y}_{1}}, \ldots, \frac{y_K}{\hat{y}_{K}}\right)^T = - (\mathbf{y}\oslash\hat{\mathbf{y}})^T, $$

其中记住 $\oslash$ 是 Hadamard 除法（即逐元素除法）。

我们接下来总结整个过程。

*初始化：*

$$\mathbf{z}_0 := \mathbf{x}$$

*前向层循环：*

$$\begin{align*} \mathbf{z}_{1} &:= \bfg_0(\mathbf{z}_0, \mathbf{w}_0) = \mathcal{W}_0 \mathbf{z}_0\\ \begin{pmatrix} A_0 & B_0 \end{pmatrix} &:= J_{\bfg_0}(\mathbf{z}_0,\mathbf{w}_0) = \begin{pmatrix} \mathbb{A}_{K}[\mathbf{w}_0] & \mathbb{B}_{K}[\mathbf{z}_0] \end{pmatrix} \end{align*}$$$$\begin{align*} \hat{\mathbf{y}} := \mathbf{z}_2 &:= \bfg_1(\mathbf{z}_1) = \bgamma(\mathbf{z}_1)\\ A_1 &:= J_{\bfg_1}(\mathbf{z}_1) = \mathrm{diag}(\bgamma(\mathbf{z}_1)) - \bgamma(\mathbf{z}_1) \, \bgamma(\mathbf{z}_1)^T \end{align*}$$

*损失：*

$$\begin{align*} z_3 &:= \ell(\mathbf{z}_2) = - \sum_{i=1}^K y_i \log z_{2,i}\\ \mathbf{p}_2 &:= \nabla {\ell_{\mathbf{y}}}(\mathbf{z}_2) = -\left(\frac{y_1}{z_{2,1}}, \ldots, \frac{y_K}{z_{2,K}}\right) = - \mathbf{y} \oslash \mathbf{z}_2. \end{align*}$$

*反向层循环:*

$$\begin{align*} \mathbf{p}_{1} &:= A_1^T \mathbf{p}_{2} \end{align*}$$$$\begin{align*} \mathbf{q}_{0} &:= B_0^T \mathbf{p}_{1} \end{align*}$$

*输出:*

$$ \nabla f(\mathbf{w}) = \mathbf{q}_0, $$

其中记住$\mathbf{w} := \mathbf{w}_0$。

可以从之前的递归中推导出显式公式。

我们首先计算$\mathbf{p}_1$。我们使用了哈达玛积的性质。我们得到

$$\begin{align*} \mathbf{p}_1 &= A_1^T \mathbf{p}_{2}\\ &= [\mathrm{diag}(\bgamma(\mathbf{z}_1)) - \bgamma(\mathbf{z}_1) \, \bgamma(\mathbf{z}_1)^T]^T [- \mathbf{y} \oslash \bgamma(\mathbf{z}_1)]\\ &= - \mathrm{diag}(\bgamma(\mathbf{z}_1)) \, (\mathbf{y} \oslash \bgamma(\mathbf{z}_1)) + \bgamma(\mathbf{z}_1) \, \bgamma(\mathbf{z}_1)^T \, (\mathbf{y} \oslash \bgamma(\mathbf{z}_1))\\ &= - \mathbf{y} + \bgamma(\mathbf{z}_1) \, \mathbf{1}^T\mathbf{y}\\ &= \bgamma(\mathbf{z}_1) - \mathbf{y}, \end{align*}$$

其中我们使用了$\sum_{k=1}^{K} y_k = 1$。

剩下的就是计算$\mathbf{q}_0$。根据克朗内克积的性质的(e)和(f)部分，我们有

$$\begin{align*} \mathbf{q}_{0} = B_0^T \mathbf{p}_{1} &= (I_{K\times K} \otimes \mathbf{z}_0^T)^T (\bgamma(\mathbf{z}_1) - \mathbf{y})\\ &= ( I_{K\times K} \otimes \mathbf{z}_0)\, (\bgamma(\mathbf{z}_1) - \mathbf{y})\\ &= (\bgamma(\mathbf{z}_1) - \mathbf{y}) \otimes \mathbf{z}_0. \end{align*}$$

最后，将$\mathbf{z}_0 = \mathbf{x}$和$\mathbf{z}_1 = \mathcal{W} \mathbf{x}$代入，梯度为

$$ \nabla f(\mathbf{w}) = \mathbf{q}_0 = (\bgamma\left(\mathcal{W} \mathbf{x}\right) - \mathbf{y}) \otimes \mathbf{x}. $$

可以证明目标函数$f(\mathbf{w})$在$\mathbf{w}$上是凸的。

**数值角落:** 我们将使用 Fashion-MNIST 数据集。这个例子受到了[这些](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) [教程](https://www.tensorflow.org/tutorials/keras/classification)的启发。我们首先检查 GPU 的可用性并加载数据。

```py
device = torch.device("cuda" if torch.cuda.is_available() 
                      else ("mps" if torch.backends.mps.is_available() 
                            else "cpu"))
print("Using device:", device) 
```

```py
Using device: mps 
```

```py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

seed = 42
torch.manual_seed(seed)

if device.type == 'cuda': # device-specific seeding and settings
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
elif device.type == 'mps':
    torch.mps.manual_seed(seed)  # MPS-specific seeding

g = torch.Generator()
g.manual_seed(seed)

train_dataset = datasets.FashionMNIST(root='./data', train=True, 
                               download=True, transform=transforms.ToTensor())
test_dataset = datasets.FashionMNIST(root='./data', train=False, 
                              download=True, transform=transforms.ToTensor())

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) 
```

我们使用了[`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)，它提供了加载数据进行训练的实用工具。我们使用了大小为`BATCH_SIZE = 32`的小批量，并在每次遍历训练数据时对样本进行随机排列（使用选项`shuffle=True`）。函数[`torch.manual_seed()`](https://pytorch.org/docs/stable/generated/torch.manual_seed.html)用于设置 PyTorch 操作的全球种子（例如，权重初始化）。`DataLoader`中的洗牌使用它自己的独立随机数生成器，我们使用[`torch.Generator()`](https://pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)和[`manual_seed()`](https://pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator.manual_seed)进行初始化。（你可以从`seed=42`这个事实中看出，克劳德向我解释了这一点……）

**CHAT & LEARN** 请您向您喜欢的 AI 聊天机器人询问以下内容：

```py
if device.type == 'cuda': # device-specific seeding and settings
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
elif device.type == 'mps':
    torch.mps.manual_seed(seed)  # MPS-specific seeding 
```

$\ddagger$

我们实现多项式逻辑回归来学习 Fashion-MNIST 数据的分类器。在 PyTorch 中，可以使用 `torch.nn.Sequential` 实现函数的组合。我们的模型是：

```py
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 10)
).to(device) 
```

`torch.nn.Flatten` 层将每个输入图像转换为大小为 $784$ 的向量（其中 $784 = 28²$ 是每张图像中的像素数）。在展平之后，我们有一个从 $\mathbb{R}^{784}$ 到 $\mathbb{R}^{10}$ 的仿射映射。请注意，不需要通过添加 $1$ 来预先处理输入。PyTorch 会自动添加一个常数项（或“偏置变量”），除非选择选项 `bias=False`（见[这里](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)）。最终输出是 $10$ 维的。

最后，我们准备好在我们的损失函数上运行我们选择的优化方法，这些方法将在下面指定。有许多[优化器](https://pytorch.org/docs/stable/optim.html#algorithms)可供选择。（参见[这篇帖子](https://hackernoon.com/demystifying-different-variants-of-gradient-descent-optimization-algorithm-19ae9ba2e9bc)，了解许多常见优化器的简要说明。）这里我们使用 SGD 作为优化器。快速教程[在这里](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)。损失函数是[交叉熵](https://en.wikipedia.org/wiki/Cross_entropy)，由 `torch.nn.CrossEntropyLoss` 实现，它首先执行 softmax 操作，并期望标签是实际的类别标签而不是它们的 one-hot 编码。

```py
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3) 
```

我们实现了特殊的训练函数。

```py
def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)    
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def training_loop(train_loader, model, loss_fn, optimizer, device, epochs=3):
    for epoch in range(epochs):
        train(train_loader, model, loss_fn, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs}") 
```

一个 epoch 是一次训练迭代，其中所有样本都会被迭代一次（以随机打乱顺序）。为了节省时间，我们只训练了 10 个 epochs。但如果你训练得更久，效果会更好（试试看！）在每次遍历中，我们计算当前模型的输出，使用 `backward()` 获取梯度，然后使用 `step()` 执行下降更新。我们还需要首先重置梯度（否则它们会默认累加）。

```py
training_loop(train_loader, model, loss_fn, optimizer, device, epochs=10) 
```

由于[过拟合](https://en.wikipedia.org/wiki/Overfitting)的问题，我们使用测试图像来评估最终分类器的性能。

```py
def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    correct = 0    
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(dim=1) == y).type(torch.float).sum().item()

    print(f"Test error: {(100*(correct  /  size)):>0.1f}% accuracy") 
```

```py
test(test_loader, model, loss_fn, device) 
```

```py
Test error: 78.7% accuracy 
```

要进行预测，我们对模型的输出执行 `torch.nn.functional.softmax`。回想一下，它隐式地包含在 `torch.nn.CrossEntropyLoss` 中，但不是 `model` 的实际部分。（注意，softmax 本身没有参数。）

作为说明，我们对每个测试图像都这样做。我们使用 `torch.cat` 将一系列张量连接成一个单独的张量。

```py
import torch.nn.functional as F

def predict_softmax(dataloader, model, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    predictions = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            probabilities = F.softmax(pred, dim=1)
            predictions.append(probabilities.cpu())

    return torch.cat(predictions, dim=0)

predictions = predict_softmax(test_loader, model, device).numpy() 
```

第一张测试图像的结果如下。为了做出预测，我们选择概率最高的标签。

```py
print(predictions[0]) 
```

```py
[4.4307188e-04 3.8354204e-04 2.0886613e-03 8.8066678e-04 3.6079765e-03
 1.7791630e-01 1.4651606e-03 2.2466542e-01 4.8245404e-02 5.4030383e-01] 
```

```py
predictions[0].argmax(0) 
```

```py
9 
```

真实情况是：

```py
images, labels = next(iter(test_loader))
images = images.squeeze().numpy()
labels = labels.numpy()

print(f"{labels[0]}: '{mmids.FashionMNIST_get_class_name(labels[0])}'") 
```

```py
9: 'Ankle boot' 
```

上面的`next(iter(test_loader))`加载了第一个测试图像批次。（有关 Python 中迭代器的背景信息，请参阅[这里](https://docs.python.org/3/tutorial/classes.html#iterators)。）

$\unlhd$

***自我评估测验*** *(由 Claude, Gemini 和 ChatGPT 协助)*

**1** 在随机梯度下降（SGD）中，每个迭代中如何估计梯度？

a) 通过在整个数据集上计算梯度。

b) 使用上一迭代的梯度。

c) 通过随机选择样本子集并计算它们的梯度。

d) 通过平均数据集中所有样本的梯度。

**2** 使用小批量 SGD 而不是标准 SGD 的关键优势是什么？

a) 它保证了更快地收敛到最优解。

b) 它在每个迭代中减少了梯度估计的方差。

c) 它消除了计算梯度的需要。

d) 它增加了每个迭代的计算成本。

**3** 关于随机梯度下降中的更新步骤，以下哪个陈述是正确的？

a) 它始终等于完整梯度下降更新。

b) 它始终与完整梯度下降更新的方向相反。

c) 平均而言，它等于完整梯度下降更新。

d) 它与完整梯度下降更新没有关系。

**4** 在多项式逻辑回归中，softmax 函数$\boldsymbol{\gamma}$的作用是什么？

a) 计算损失函数的梯度。

b) 用于归一化输入特征。

c) 将分数转换成标签的概率分布。

d) 在梯度下降过程中更新模型参数。

**5** 在多项式逻辑回归中，Kullback-Leibler（KL）散度有什么用途？

a) 测量预测概率与真实标签之间的距离。

b) 用于归一化输入特征。

c) 在梯度下降过程中更新模型参数。

d) 计算损失函数的梯度。

1 的答案：c. 理由：文本指出在 SGD 中，“我们在$\{1, ..., n\}$中随机选择一个样本，并按以下方式更新：$\mathbf{w}^{t+1} = \mathbf{w}^t - \alpha_t \nabla f_{\mathbf{x}_{I_t}, y_{I_t}}(\mathbf{w}^t).$”

2 的答案：b. 理由：文本暗示与标准 SGD 相比，小批量 SGD 减少了梯度估计的方差，而标准 SGD 只使用单个样本。

3 的答案：c. 理由：文本证明了一个引理，表明“在期望中，它们[随机更新]执行梯度下降的一步。”

4 的答案：c. 理由：文本定义了 softmax 函数，并指出它被用来“将这些分数转换成标签的概率分布。”

5 号问题的答案：a. 证明：文本将 KL 散度引入为“概率测度之间的距离概念”，并使用它来定义多项式逻辑回归中的损失函数。

## 8.4.1\. 算法#

在[随机梯度下降](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)（SGD）中，梯度下降的一个变体，我们在 $\{1,\ldots,n\}$ 中随机均匀地选择一个样本 $I_t$，并按以下方式更新

$$ \mathbf{w}^{t+1} = \mathbf{w}^{t} - \alpha_t \nabla f_{\mathbf{x}_{I_t},y_{I_t}}(\mathbf{w}^{t}). $$

更一般地，在所谓的 SGD 的小批量版本中，我们选择一个大小为 $b$ 的均匀随机子样本 $\mathcal{B}_t \subseteq \{1,\ldots,n\}$ 而不是替换（即，所有该大小的子样本被选中的概率相同）

$$ \mathbf{w}^{t+1} = \mathbf{w}^{t} - \alpha_t \frac{1}{b} \sum_{i\in \mathcal{B}_t} \nabla f_{\mathbf{x}_i,y_i}(\mathbf{w}^{t}). $$

关于上述两个随机更新的关键观察是，在期望上，它们执行了一步梯度下降。这证明是足够的，并且具有计算上的优势。

**引理** 设定一个批大小 $1 \leq b \leq n$ 和一个任意的参数向量 $\mathbf{w}$。令 $\mathcal{B} \subseteq \{1,\ldots,n\}$ 是大小为 $b$ 的均匀随机子样本。那么

$$ \mathbb{E}\left[\frac{1}{b} \sum_{i\in \mathcal{B}} \nabla f_{\mathbf{x}_i,y_i}(\mathbf{w})\right] = \frac{1}{n} \sum_{i=1}^n \nabla f_{\mathbf{x}_i,y_i}(\mathbf{w}). $$

$\flat$

*证明* 因为 $\mathcal{B}$ 是随机均匀选择（不替换），对于任何大小为 $b$ 的不重复子样本 $B \subseteq \{1,\ldots,n\}$

$$ \mathbb{P}[\mathcal{B} = B] = \frac{1}{\binom{n}{b}}. $$

因此，对所有这样的子样本求和，我们得到

$$\begin{align*} \mathbb{E}\left[\frac{1}{b} \sum_{i\in \mathcal{B}} \nabla f_{\mathbf{x}_i,y_i}(\mathbf{w})\right] &= \sum_{B \subseteq \{1,\ldots,n\}} \mathbb{P}[\mathcal{B} = B] \frac{1}{b} \sum_{i\in B} \nabla f_{\mathbf{x}_i,y_i}(\mathbf{w})\\ &= \sum_{B \subseteq \{1,\ldots,n\}} \frac{1}{\binom{n}{b}} \frac{1}{b} \sum_{i=1}^n \mathbf{1}\{i \in B\} \nabla f_{\mathbf{x}_i,y_i}(\mathbf{w})\\ &= \sum_{i=1}^n \nabla f_{\mathbf{x}_i,y_i}(\mathbf{w}) \frac{1}{b \binom{n}{b}} \sum_{B \subseteq \{1,\ldots,n\}} \mathbf{1}\{i \in B\}. \end{align*}$$

计算内部和需要组合论证。实际上，$\sum_{B \subseteq \{1,\ldots,n\}} \mathbf{1}\{i \in B\}$ 计算了 $i$ 在大小为 $b$ 的子样本中不重复被选择的方式数。这就是 $\binom{n-1}{b-1}$，这是从其他 $n-1$ 个可能元素中选择 $B$ 的剩余 $b-1$ 个元素的方式数。根据二项式系数的定义和阶乘的性质，

$$ \frac{\binom{n-1}{b-1}}{b \binom{n}{b}} = \frac{\frac{(n-1)!}{(b-1)! (n-b)!}}{b \frac{n!}{b! (n-b)!}} = \frac{(n-1)!}{n!} \frac{b!}{b (b-1)!} = \frac{1}{n}. $$

将其代入给出结论。 $\square$

作为第一个示例，我们回到逻辑回归$\idx{logistic regression}\xdi$。回想一下，输入数据的形式为$\{(\boldsymbol{\alpha}_i, b_i) : i=1,\ldots, n\}$，其中$\boldsymbol{\alpha}_i = (\alpha_{i,1}, \ldots, \alpha_{i,d}) \in \mathbb{R}^d$是特征，$b_i \in \{0,1\}$是标签。和之前一样，我们使用矩阵表示：$A \in \mathbb{R}^{n \times d}$的行是$\boldsymbol{\alpha}_i^T$，$i = 1,\ldots, n$，$\mathbf{b} = (b_1, \ldots, b_n) \in \{0,1\}^n$。我们想要解决最小化问题

$$ \min_{\mathbf{x} \in \mathbb{R}^d} \ell(\mathbf{x}; A, \mathbf{b}) $$

其中损失为

$$\begin{align*} \ell(\mathbf{x}; A, \mathbf{b}) &= \frac{1}{n} \sum_{i=1}^n \left\{- b_i \log(\sigma(\boldsymbol{\alpha_i}^T \mathbf{x})) - (1-b_i) \log(1- \sigma(\boldsymbol{\alpha_i}^T \mathbf{x}))\right\}\\ &= \mathrm{mean}\left(-\mathbf{b} \odot \mathbf{log}(\bsigma(A \mathbf{x})) - (\mathbf{1} - \mathbf{b}) \odot \mathbf{log}(\mathbf{1} - \bsigma(A \mathbf{x}))\right). \end{align*}$$

前向梯度之前被计算为

$$\begin{align*} \nabla\ell(\mathbf{x}; A, \mathbf{b}) &= - \frac{1}{n} \sum_{i=1}^n ( b_i - \sigma(\boldsymbol{\alpha}_i^T \mathbf{x}) ) \,\boldsymbol{\alpha}_i\\ &= -\frac{1}{n} A^T [\mathbf{b} - \bsigma(A \mathbf{x})]. \end{align*}$$

对于小批量版本的随机梯度下降（SGD），我们随机选择一个大小为$B$的子样本$\mathcal{B}_t \subseteq \{1,\ldots,n\}$，并采取以下步骤

$$ \mathbf{x}^{t+1} = \mathbf{x}^{t} +\beta \frac{1}{B} \sum_{i\in \mathcal{B}_t} ( b_i - \sigma(\boldsymbol{\alpha}_i^T \mathbf{x}^t) ) \,\boldsymbol{\alpha}_i. $$

我们修改了我们之前的逻辑回归代码。唯一的改变是随机选择一个迷你批量，将其作为数据集输入到下降更新子例程中。

```py
def sigmoid(z): 
    return 1/(1+np.exp(-z))

def pred_fn(x, A): 
    return sigmoid(A @ x)

def loss_fn(x, A, b): 
    return np.mean(-b*np.log(pred_fn(x, A)) - (1 - b)*np.log(1 - pred_fn(x, A)))

def grad_fn(x, A, b):
    return -A.T @ (b - pred_fn(x, A))/len(b)

def desc_update_for_logreg(grad_fn, A, b, curr_x, beta):
    gradient = grad_fn(curr_x, A, b)
    return curr_x - beta*gradient

def sgd_for_logreg(rng, loss_fn, grad_fn, A, b, 
                   init_x, beta=1e-3, niters=int(1e5), batch=40):

    curr_x = init_x
    nsamples = len(b)
    for _ in range(niters):
        I = rng.integers(nsamples, size=batch)
        curr_x = desc_update_for_logreg(
            grad_fn, A[I,:], b[I], curr_x, beta)

    return curr_x 
```

**数值角落：** 我们分析了来自[[ESL](https://web.stanford.edu/~hastie/ElemStatLearn/)]的数据集，该数据集可以在此[下载](https://web.stanford.edu/~hastie/ElemStatLearn/data.html)。引用[[ESL](https://web.stanford.edu/~hastie/ElemStatLearn/)]，第 4.4.2 节

> 数据 [...] 是南非开普敦西部三个农村地区的冠状动脉风险因素研究（CORIS）基线调查的一个子集（Rousseauw 等，1983 年）。该研究旨在确定该高发病率地区缺血性心脏病风险因素的强度。数据代表 15 至 64 岁的白人男性，响应变量是在调查时心肌梗死（MI）的存在或不存在（该地区 MI 的整体患病率为 5.1%）。我们的数据集中有 160 个病例，302 个对照组样本。这些数据在 Hastie 和 Tibshirani（1987 年）中描述得更详细。

我们加载数据，我们稍作格式化并查看摘要。

```py
data = pd.read_csv('SAHeart.csv')
data.head() 
```

|  | sbp | tobacco | ldl | adiposity | typea | obesity | alcohol | age | chd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 160.0 | 12.00 | 5.73 | 23.11 | 49.0 | 25.30 | 97.20 | 52.0 | 1.0 |
| 1 | 144.0 | 0.01 | 4.41 | 28.61 | 55.0 | 28.87 | 2.06 | 63.0 | 1.0 |
| 2 | 118.0 | 0.08 | 3.48 | 32.28 | 52.0 | 29.14 | 3.81 | 46.0 | 0.0 |
| 3 | 170.0 | 7.50 | 6.41 | 38.03 | 51.0 | 31.99 | 24.26 | 58.0 | 1.0 |
| 4 | 134.0 | 13.60 | 3.50 | 27.78 | 60.0 | 25.99 | 57.34 | 49.0 | 1.0 |

我们的目标是根据其他变量（简要描述见[此处](https://web.stanford.edu/~hastie/ElemStatLearn/datasets/SAheart.info.txt)）预测 `chd`，即冠心病。我们再次使用逻辑回归。

我们首先构建数据矩阵。我们只使用三个预测变量。

```py
feature = data[['tobacco', 'ldl', 'age']].to_numpy()
print(feature) 
```

```py
[[1.200e+01 5.730e+00 5.200e+01]
 [1.000e-02 4.410e+00 6.300e+01]
 [8.000e-02 3.480e+00 4.600e+01]
 ...
 [3.000e+00 1.590e+00 5.500e+01]
 [5.400e+00 1.161e+01 4.000e+01]
 [0.000e+00 4.820e+00 4.600e+01]] 
```

```py
label = data['chd'].to_numpy()
A = np.concatenate((np.ones((len(label),1)),feature),axis=1)
b = label 
```

我们尝试使用小批量随机梯度下降法。

```py
seed = 535
rng = np.random.default_rng(seed)
init_x = np.zeros(A.shape[1])
best_x = sgd_for_logreg(rng, loss_fn, grad_fn, A, b, init_x, beta=1e-3, niters=int(1e6))
print(best_x) 
```

```py
[-4.06558071  0.07990955  0.18813635  0.04693118] 
```

结果更难以可视化。为了了解结果的准确性，我们将我们的预测与真实标签进行比较。让我们假设，当我们说预测时，我们指的是当 $\sigma(\boldsymbol{\alpha}^T \mathbf{x}) > 1/2$ 时预测标签 $1$。我们尝试在训练集上这样做。（更好的方法是将数据分成训练集和测试集，但在这里我们不会这样做。）

```py
def logis_acc(x, A, b):
    return np.sum((pred_fn(x, A) > 0.5) == b)/len(b) 
```

```py
logis_acc(best_x, A, b) 
```

```py
0.7207792207792207 
```

$\unlhd$

## 8.4.2\. 示例：多元逻辑回归#

我们给出一个关于渐进函数以及反向传播和随机梯度下降法应用的实例。

回想一下，分类器 $h$ 接受 $\mathbb{R}^d$ 中的输入并预测 $K$ 个可能标签中的一个。为了下面的原因，使用 [独热编码](https://en.wikipedia.org/wiki/One-hot)$\idx{one-hot encoding}\xdi$ 的标签将很方便。也就是说，我们将标签 $i$ 编码为 $K$ 维向量 $\mathbf{e}_i$。在这里，像往常一样，$\mathbf{e}_i$ 是 $\mathbb{R}^K$ 的标准基，即在第 $i$ 个位置有 $1$，其他位置为 $0$ 的向量。此外，我们允许分类器的输出是标签 $\{1,\ldots,K\}$ 的概率分布，即一个向量在

$$ \Delta_K = \left\{ (p_1,\ldots,p_K) \in [0,1]^K \,:\, \sum_{k=1}^K p_k = 1 \right\}. $$

注意到 $\mathbf{e}_i$ 本身也可以被视为一个概率分布，它将概率 $1$ 分配给 $i$。

**多元逻辑回归背景** 我们使用 [多元逻辑回归](https://en.wikipedia.org/wiki/Multinomial_logistic_regression)$\idx{multinomial logistic regression}\xdi$ 来学习 $K$ 个标签的分类器。在多元逻辑回归中，我们再次使用输入数据的仿射函数。

这次，我们有 $K$ 个函数，每个函数输出与每个标签相关的分数。然后我们将这些分数转换成 $K$ 个标签上的概率分布。有多种方法可以做到这一点。一种标准的方法是 [softmax 函数](https://en.wikipedia.org/wiki/Softmax_function)$\idx{softmax}\xdi$ $\bgamma = (\gamma_1,\ldots,\gamma_K)$：对于 $\mathbf{z} \in \mathbb{R}^K$

$$ \gamma_i(\mathbf{z}) = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}, \quad i=1,\ldots,K. $$

为了解释名称，观察较大的输入被映射到较大的概率。

事实上，由于概率分布必须求和为 $1$，它由分配给前 $K-1$ 个标签的概率决定。换句话说，我们可以省略与最后一个标签相关的分数。但为了简化符号，我们在这里不会这样做。

对于每个 $k$，我们有一个回归函数

$$ \sum_{j=1}^d w^{(k)}_{j} x_{j} = \mathbf{x}_1^T \mathbf{w}^{(k)}, \quad k=1,\ldots,K $$

其中 $\mathbf{w} = (\mathbf{w}^{(1)},\ldots,\mathbf{w}^{(K)})$ 是参数，其中 $\mathbf{w}^{(k)} \in \mathbb{R}^{d}$ 且 $\mathbf{x} \in \mathbb{R}^d$ 是输入。可以通过向 $\mathbf{x}$ 添加一个额外的条目 $1$ 来包含一个常数项。正如我们在线性回归案例中所做的那样，我们假设这种预处理已经完成。为了简化符号，我们让 $\mathcal{W} \in \mathbb{R}^{K \times d}$ 作为具有行 $(\mathbf{w}^{(1)})^T,\ldots,(\mathbf{w}^{(K)})^T$ 的矩阵。

分类器的输出是

$$\begin{align*} \bfh(\mathbf{w}) &= \bgamma\left(\mathcal{W} \mathbf{x}\right), \end{align*}$$

对于 $i=1,\ldots,K$，其中 $\bgamma$ 是 softmax 函数。注意，后者没有关联的参数。

剩下的任务是定义一个损失函数。为了量化拟合度，自然地使用概率测度之间的距离概念，这里是在输出 $\mathbf{h}(\mathbf{w}) \in \Delta_K$ 和正确标签 $\mathbf{y} \in \{\mathbf{e}_1,\ldots,\mathbf{e}_{K}\} \subseteq \Delta_K$ 之间的距离。存在许多这样的度量。在多项式逻辑回归中，我们使用 Kullback-Leibler 散度，这在最大似然估计的上下文中已经遇到过。回想一下，对于两个概率分布 $\mathbf{p}, \mathbf{q} \in \Delta_K$，它被定义为

$$ \mathrm{KL}(\mathbf{p} \| \mathbf{q}) = \sum_{i=1}^K p_i \log \frac{p_i}{q_i} $$

其中，我们只需将注意力限制在 $\mathbf{q} > \mathbf{0}$ 的情况，并且使用约定 $0 \log 0 = 0$（这样 $p_i = 0$ 的项对总和的贡献为 $0$）。注意，$\mathbf{p} = \mathbf{q}$ 意味着 $\mathrm{KL}(\mathbf{p} \| \mathbf{q}) = 0$。我们之前已经证明了 $\mathrm{KL}(\mathbf{p} \| \mathbf{q}) \geq 0$，这是一个被称为 *吉布斯不等式* 的结果。

回到损失函数，我们使用恒等式 $\log\frac{\alpha}{\beta} = \log \alpha - \log \beta$ 来重新写

$$\begin{align*} \mathrm{KL}(\mathbf{y} \| \bfh(\mathbf{w})) &= \sum_{i=1}^K y_i \log \frac{y_i}{h_{i}(\mathbf{w})}\\ &= \sum_{i=1}^K y_i \log y_i - \sum_{i=1}^K y_i \log h_{i}(\mathbf{w}), \end{align*}$$

其中 $\bfh = (h_{1},\ldots,h_{K})$。注意，右侧的第一个项不依赖于 $\mathbf{w}$。因此，在优化 $\mathrm{KL}(\mathbf{y} \| \bfh(\mathbf{w}))$ 时可以忽略它。剩余的项是

$$ H(\mathbf{y}, \bfh(\mathbf{w})) = - \sum_{i=1}^K y_i \log h_{i}(\mathbf{w}). $$

我们用它来定义我们的损失函数。也就是说，我们设置

$$ \ell(\hat{\mathbf{y}}) = H(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{i=1}^K y_i \log \hat{y}_{i}. $$

最后，

$$\begin{align*} f(\mathbf{w}) &= \ell(\bfh(\mathbf{w}))\\ &= H(\mathbf{y}, \bfh(\mathbf{w}))\\ &= H\left(\mathbf{y}, \bgamma\left(\mathcal{W} \mathbf{x}\right)\right)\\ &= - \sum_{i=1}^K y_i \log\gamma_i\left(\mathcal{W} \mathbf{x}\right). \end{align*}$$

**计算梯度** 我们应用了上一节中的前向和反向传播步骤。然后我们使用这些递归关系推导出梯度的解析公式。

前向传播从初始化 $\mathbf{z}_0 := \mathbf{x}$ 开始。前向层循环有两个步骤。将 $\mathbf{w}_0 = (\mathbf{w}_0^{(1)},\ldots,\mathbf{w}_0^{(K)})$ 等于 $\mathbf{w} = (\mathbf{w}^{(1)},\ldots,\mathbf{w}^{(K)})$。首先我们计算

$$\begin{align*} \mathbf{z}_{1} &:= \bfg_0(\mathbf{z}_0,\mathbf{w}_0) = \mathcal{W}_0 \mathbf{z}_0\\ J_{\bfg_0}(\mathbf{z}_0,\mathbf{w}_0) &:=\begin{pmatrix} A_0 & B_0 \end{pmatrix} \end{align*}$$

其中我们定义 $\mathcal{W}_0 \in \mathbb{R}^{K \times d}$ 为具有行 $(\mathbf{w}_0^{(1)})^T,\ldots,(\mathbf{w}_0^{(K-1)})^T$ 的矩阵。我们之前已经计算了雅可比矩阵：

$$\begin{split} A_0 = \mathbb{A}_{K}[\mathbf{w}_0] = \mathcal{W}_0 = \begin{pmatrix} (\mathbf{w}^{(1)}_0)^T\\ \vdots\\ (\mathbf{w}^{(K)}_0)^T \end{pmatrix} \end{split}$$

和

$$ B_0 = \mathbb{B}_{K}[\mathbf{z}_0] = I_{K\times K} \otimes \mathbf{z}_0^T = \begin{pmatrix} \mathbf{e}_1 \mathbf{z}_0^T & \cdots & \mathbf{e}_{K}\mathbf{z}_0^T \end{pmatrix}. $$

在前向层循环的第二步中，我们计算

$$\begin{align*} \hat{\mathbf{y}} := \mathbf{z}_2 &:= \bfg_1(\mathbf{z}_1) = \bgamma(\mathbf{z}_1)\\ A_1 &:= J_{\bfg_1}(\mathbf{z}_1) = J_{\bgamma}(\mathbf{z}_1). \end{align*}$$

因此，我们需要计算 $\bgamma$ 的雅可比矩阵。我们将这个计算分为两种情况。当 $1 \leq i = j \leq K$，

$$\begin{align*} (A_1)_{ii} &= \frac{\partial}{\partial z_{1,i}} \left[ \gamma_i(\mathbf{z}_1) \right]\\ &= \frac{\partial}{\partial z_{1,i}} \left[ \frac{e^{z_{1,i}}}{\sum_{k=1}^{K} e^{z_{1,k}}} \right]\\ &= \frac{e^{z_{1,i}}\left(\sum_{k=1}^{K} e^{z_{1,k}}\right) - e^{z_{1,i}}\left(e^{z_{1,i}}\right)} {\left(\sum_{k=1}^{K} e^{z_{1,k}}\right)²}\\ &= \gamma_i(\mathbf{z}_1) - \gamma_i(\mathbf{z}_1)², \end{align*}$$

通过[商规则](https://en.wikipedia.org/wiki/Quotient_rule)。

当 $1 \leq i, j \leq K$ 且 $i \neq j$ 时，

$$\begin{align*} (A_1)_{ij} &= \frac{\partial}{\partial z_{1,j}} \left[ \gamma_i(\mathbf{z}_1) \right]\\ &= \frac{\partial}{\partial z_{1,j}} \left[ \frac{e^{z_{1,i}}}{\sum_{k=1}^{K} e^{z_{1,k}}} \right]\\ &= \frac{- e^{z_{1,i}}\left(e^{z_{1,j}}\right)} {\left(\sum_{k=1}^{K} e^{z_{1,k}}\right)²}\\ &= - \gamma_i(\mathbf{z}_1)\gamma_j(\mathbf{z}_1). \end{align*}$$

以矩阵形式，

$$ J_{\bgamma}(\mathbf{z}_1) = A_1 = \mathrm{diag}(\bgamma(\mathbf{z}_1)) - \bgamma(\mathbf{z}_1) \, \bgamma(\mathbf{z}_1)^T. $$

损失函数的雅可比矩阵是

$$ J_{\ell}(\hat{\mathbf{y}}) = \nabla \left[ - \sum_{i=1}^K y_i \log \hat{y}_{i} \right]^T = -\left(\frac{y_1}{\hat{y}_{1}}, \ldots, \frac{y_K}{\hat{y}_{K}}\right)^T = - (\mathbf{y}\oslash\hat{\mathbf{y}})^T, $$

其中回忆起 $\oslash$ 是哈达玛除法（即逐元素除法）。

我们接下来总结整个流程。

*初始化：*

$$\mathbf{z}_0 := \mathbf{x}$$

*正向层循环：*

$$\begin{align*} \mathbf{z}_{1} &:= \bfg_0(\mathbf{z}_0, \mathbf{w}_0) = \mathcal{W}_0 \mathbf{z}_0\\ \begin{pmatrix} A_0 & B_0 \end{pmatrix} &:= J_{\bfg_0}(\mathbf{z}_0,\mathbf{w}_0) = \begin{pmatrix} \mathbb{A}_{K}[\mathbf{w}_0] & \mathbb{B}_{K}[\mathbf{z}_0] \end{pmatrix} \end{pmatrix}$$$$\begin{align*} \hat{\mathbf{y}} := \mathbf{z}_2 &:= \bfg_1(\mathbf{z}_1) = \bgamma(\mathbf{z}_1)\\ A_1 &:= J_{\bfg_1}(\mathbf{z}_1) = \mathrm{diag}(\bgamma(\mathbf{z}_1)) - \bgamma(\mathbf{z}_1) \, \bgamma(\mathbf{z}_1)^T \end{align*}$$

*损失：*

$$\begin{align*} z_3 &:= \ell(\mathbf{z}_2) = - \sum_{i=1}^K y_i \log z_{2,i}\\ \mathbf{p}_2 &:= \nabla {\ell_{\mathbf{y}}}(\mathbf{z}_2) = -\left(\frac{y_1}{z_{2,1}}, \ldots, \frac{y_K}{z_{2,K}}\right) = - \mathbf{y} \oslash \mathbf{z}_2. \end{align*}$$

*反向层循环：*

$$\begin{align*} \mathbf{p}_{1} &:= A_1^T \mathbf{p}_{2} \end{align*}$$$$\begin{align*} \mathbf{q}_{0} &:= B_0^T \mathbf{p}_{1} \end{align*}$$

*输出：*

$$ \nabla f(\mathbf{w}) = \mathbf{q}_0, $$

其中回忆起 $\mathbf{w} := \mathbf{w}_0$.

可以从之前的递归中推导出显式公式。

我们首先计算 $\mathbf{p}_1$。我们使用哈达玛积的性质。我们得到

$$\begin{align*} \mathbf{p}_1 &= A_1^T \mathbf{p}_{2}\\ &= [\mathrm{diag}(\bgamma(\mathbf{z}_1)) - \bgamma(\mathbf{z}_1) \, \bgamma(\mathbf{z}_1)^T]^T [- \mathbf{y} \oslash \bgamma(\mathbf{z}_1)]\\ &= - \mathrm{diag}(\bgamma(\mathbf{z}_1)) \, (\mathbf{y} \oslash \bgamma(\mathbf{z}_1)) + \bgamma(\mathbf{z}_1) \, \bgamma(\mathbf{z}_1)^T \, (\mathbf{y} \oslash \bgamma(\mathbf{z}_1))\\ &= - \mathbf{y} + \bgamma(\mathbf{z}_1) \, \mathbf{1}^T\mathbf{y}\\ &= \bgamma(\mathbf{z}_1) - \mathbf{y}, \end{align*}$$

其中我们使用了 $\sum_{k=1}^{K} y_k = 1$ 的性质。

剩下的任务是计算 $\mathbf{q}_0$。根据克朗内克积的性质（部分 e 和 f），我们有

$$\begin{align*} \mathbf{q}_{0} = B_0^T \mathbf{p}_{1} &= (I_{K\times K} \otimes \mathbf{z}_0^T)^T (\bgamma(\mathbf{z}_1) - \mathbf{y})\\ &= ( I_{K\times K} \otimes \mathbf{z}_0)\, (\bgamma(\mathbf{z}_1) - \mathbf{y})\\ &= (\bgamma(\mathbf{z}_1) - \mathbf{y}) \otimes \mathbf{z}_0. \end{align*}$$

最后，将 $\mathbf{z}_0 = \mathbf{x}$ 和 $\mathbf{z}_1 = \mathcal{W} \mathbf{x}$ 代入，梯度为

$$ \nabla f(\mathbf{w}) = \mathbf{q}_0 = (\bgamma\left(\mathcal{W} \mathbf{x}\right) - \mathbf{y}) \otimes \mathbf{x}. $$

可以证明目标函数 $f(\mathbf{w})$ 在 $\mathbf{w}$ 上是凸的。

**NUMERICAL CORNER:** 我们将使用 Fashion-MNIST 数据集。这个例子受到了[这些](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) [教程](https://www.tensorflow.org/tutorials/keras/classification)的启发。我们首先检查 GPU 的可用性并加载数据。

```py
device = torch.device("cuda" if torch.cuda.is_available() 
                      else ("mps" if torch.backends.mps.is_available() 
                            else "cpu"))
print("Using device:", device) 
```

```py
Using device: mps 
```

```py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

seed = 42
torch.manual_seed(seed)

if device.type == 'cuda': # device-specific seeding and settings
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
elif device.type == 'mps':
    torch.mps.manual_seed(seed)  # MPS-specific seeding

g = torch.Generator()
g.manual_seed(seed)

train_dataset = datasets.FashionMNIST(root='./data', train=True, 
                               download=True, transform=transforms.ToTensor())
test_dataset = datasets.FashionMNIST(root='./data', train=False, 
                              download=True, transform=transforms.ToTensor())

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) 
```

我们使用了 `torch.utils.data.DataLoader`，它提供了加载数据的批量训练工具。我们使用了大小为 `BATCH_SIZE = 32` 的迷你批次，并在每次遍历训练数据时对样本进行随机排列（使用选项 `shuffle=True`）。函数 `torch.manual_seed()` 用于设置 PyTorch 操作的全球种子（例如，权重初始化）。`DataLoader` 中的洗牌使用其自己的独立随机数生成器，我们使用 `torch.Generator()` 和 `manual_seed()` 进行初始化。（您可以从 `seed=42` 这一事实中看出，Claude 向我解释了这一点…）

**CHAT & LEARN** 请您最喜欢的 AI 聊天机器人解释以下行：

```py
if device.type == 'cuda': # device-specific seeding and settings
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
elif device.type == 'mps':
    torch.mps.manual_seed(seed)  # MPS-specific seeding 
```

$\ddagger$

我们实现了多项式逻辑回归来学习 Fashion-MNIST 数据的分类器。在 PyTorch 中，可以使用 `torch.nn.Sequential` 实现函数的组合。我们的模型是：

```py
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 10)
).to(device) 
```

`torch.nn.Flatten` 层将每个输入图像转换为大小为 $784$ 的向量（其中 $784 = 28²$ 是每个图像中的像素数）。在展平之后，我们有一个从 $\mathbb{R}^{784}$ 到 $\mathbb{R}^{10}$ 的仿射映射。请注意，不需要通过添加 $1$ 来预处理输入。PyTorch 会自动添加一个常数项（或“偏置变量”），除非选择选项 `bias=False`。最终的输出是 $10$ 维的。

最后，我们准备好在损失函数上运行我们选择的优化方法，这些方法将在下面指定。有许多[优化器](https://pytorch.org/docs/stable/optim.html#algorithms)可供选择。（有关许多常见优化器的简要解释，请参阅这篇[文章](https://hackernoon.com/demystifying-different-variants-of-gradient-descent-optimization-algorithm-19ae9ba2e9bc)。）这里我们使用 SGD 作为优化器。快速教程[在这里](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)。损失函数是[交叉熵](https://en.wikipedia.org/wiki/Cross_entropy)，由`torch.nn.CrossEntropyLoss`实现，它首先进行 softmax，并期望标签是实际的类别标签而不是它们的 one-hot 编码。

```py
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3) 
```

我们实现了用于训练的特殊函数。

```py
def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)    
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def training_loop(train_loader, model, loss_fn, optimizer, device, epochs=3):
    for epoch in range(epochs):
        train(train_loader, model, loss_fn, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs}") 
```

一个 epoch 是一次训练迭代，其中所有样本都迭代一次（以随机打乱顺序）。为了节省时间，我们只训练了 10 个 epochs。但如果你训练得更久，效果会更好（试试看！）在每次遍历中，我们计算当前模型的输出，使用`backward()`获取梯度，然后使用`step()`执行下降更新。我们还需要首先重置梯度（否则它们会默认累加）。

```py
training_loop(train_loader, model, loss_fn, optimizer, device, epochs=10) 
```

由于[过拟合](https://en.wikipedia.org/wiki/Overfitting)的问题，我们使用*测试*图像来评估最终分类器的性能。

```py
def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    correct = 0    
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(dim=1) == y).type(torch.float).sum().item()

    print(f"Test error: {(100*(correct  /  size)):>0.1f}% accuracy") 
```

```py
test(test_loader, model, loss_fn, device) 
```

```py
Test error: 78.7% accuracy 
```

为了做出预测，我们对模型输出进行`torch.nn.functional.softmax`计算。回想一下，它在`torch.nn.CrossEntropyLoss`中隐式包含，但不是`model`的实际部分。（注意，softmax 本身没有参数。）

作为说明，我们对每个测试图像都这样做。我们使用`torch.cat`将一系列张量连接成一个单一的张量。

```py
import torch.nn.functional as F

def predict_softmax(dataloader, model, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    predictions = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            probabilities = F.softmax(pred, dim=1)
            predictions.append(probabilities.cpu())

    return torch.cat(predictions, dim=0)

predictions = predict_softmax(test_loader, model, device).numpy() 
```

第一张测试图像的结果如下。为了做出预测，我们选择概率最高的标签。

```py
print(predictions[0]) 
```

```py
[4.4307188e-04 3.8354204e-04 2.0886613e-03 8.8066678e-04 3.6079765e-03
 1.7791630e-01 1.4651606e-03 2.2466542e-01 4.8245404e-02 5.4030383e-01] 
```

```py
predictions[0].argmax(0) 
```

```py
9 
```

事实是：

```py
images, labels = next(iter(test_loader))
images = images.squeeze().numpy()
labels = labels.numpy()

print(f"{labels[0]}: '{mmids.FashionMNIST_get_class_name(labels[0])}'") 
```

```py
9: 'Ankle boot' 
```

如上所述，`next(iter(test_loader))`加载了第一批测试图像。（有关 Python 中迭代器的背景信息，请参阅[这里](https://docs.python.org/3/tutorial/classes.html#iterators)。）

$\unlhd$

***自我评估测验*** *(由 Claude，Gemini 和 ChatGPT 协助)*

**1** 在随机梯度下降（SGD）中，每个迭代中是如何估计梯度的？

a) 通过在整个数据集上计算梯度。

b) 通过使用前一次迭代的梯度。

c) 通过随机选择样本子集并计算它们的梯度。

d) 通过平均数据集中所有样本的梯度。

**2** 使用小批量随机梯度下降（mini-batch SGD）相对于标准随机梯度下降（SGD）的关键优势是什么？

a) 它保证了更快地收敛到最优解。

b) 它减少了每个迭代中梯度估计的方差。

c) 完全消除了计算梯度的需要。

d) 它增加了每迭代的计算成本。

**3** 关于随机梯度下降的更新步骤，以下哪个陈述是正确的？

a) 它始终等于完整梯度下降的更新。

b) 它始终与完整梯度下降的更新方向相反。

c) 它平均上等于完整梯度下降的更新。

d) 它与完整梯度下降的更新没有关系。

**4** 在多项式逻辑回归中，softmax 函数 $\boldsymbol{\gamma}$ 的作用是什么？

a) 用于计算损失函数的梯度。

b) 用于归一化输入特征。

c) 将分数转换成标签的概率分布。

d) 在梯度下降过程中更新模型参数。

**5** Kullback-Leibler (KL) 散度在多项式逻辑回归中用于什么？

a) 用于衡量预测概率与真实标签之间的距离。

b) 用于归一化输入特征。

c) 在梯度下降过程中更新模型参数。

d) 用于计算损失函数的梯度。

1 的答案：c. 理由：文本中提到在随机梯度下降（SGD）中，“我们在 $\{1, ..., n\}$ 中随机均匀地选择一个样本，并按以下方式更新：$\mathbf{w}^{t+1} = \mathbf{w}^t - \alpha_t \nabla f_{\mathbf{x}_{I_t}, y_{I_t}}(\mathbf{w}^t).$””

2 的答案：b. 理由：文本暗示，与仅使用单个样本的标准 SGD 相比，小批量 SGD 减少了梯度估计的方差。

3 的答案：c. 理由：文本证明了一个引理，表明“在期望上，它们[随机更新]执行梯度下降的一步。”

4 的答案：c. 理由：文本定义了 softmax 函数，并指出它被用来“将这些分数转换成标签的概率分布。”

5 的答案：a. 理由：文本将 KL 散度介绍为“概率测度之间的距离概念”，并使用它来定义多项式逻辑回归中的损失函数。
