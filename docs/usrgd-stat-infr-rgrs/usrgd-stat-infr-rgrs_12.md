# 7  最小二乘法的统计特性

> 原文：[`mattblackwell.github.io/gov2002-book/ols_properties.html`](https://mattblackwell.github.io/gov2002-book/ols_properties.html)

1.  回归

1.  7  最小二乘法的统计特性

上一章展示了最小二乘估计量及其许多机械性质，这些性质对于 OLS 的实际应用至关重要。但我们仍需要了解其统计特性，正如本书第一部分所讨论的：无偏性、抽样方差、一致性和渐近正态性。正如我们当时所看到的，这些性质分为有限样本（无偏性和抽样方差）和渐近（一致性、渐近正态性）。

在本章中，我们将重点关注 OLS 的渐近性质，因为这些性质在第 5.2 节中引入的线性投影模型的相对温和条件下成立。我们将看到，OLS 始终一致地估计一个感兴趣的量（最佳线性预测器），无论条件期望是否线性。也就是说，对于估计量的渐近性质，我们不需要通常引用的线性假设。稍后，当我们研究有限样本性质时，我们将展示线性假设如何帮助我们建立无偏性，以及误差的正态性如何允许我们进行精确的有限样本推断。但这些假设非常强，因此理解在没有这些假设的情况下我们能说些什么关于 OLS 的至关重要。

## 7.1 OLS 的大样本性质

正如我们在第三章中看到的，进行 OLS 估计量的统计推断需要两个关键要素：(1) $\bhat$ 方差的稳健估计和(2) 大样本中 $\bhat$ 的近似分布。记住，由于 $\bhat$ 是一个向量，该估计量的方差实际上是一个方差-协方差矩阵。为了获得这两个关键要素，我们首先建立 OLS 的一致性，然后使用中心极限定理推导其渐近分布，其中包括其方差。

我们首先阐述建立 OLS（最小二乘法）大样本性质所需的假设，这些假设与确保最佳线性预测器 $\bfbeta = \E[\X_{i}\X_{i}']^{-1}\E[\X_{i}Y_{i}]$ 定义良好且唯一所需的假设相同。

*线性投影假设* 线性投影模型做出以下假设：

1.  $\{(Y_{i}, \X_{i})\}_{i=1}^n$ 是独立同分布的随机向量

1.  $\E[Y^{2}_{i}] < \infty$ (有限的结果方差)

1.  $\E[\Vert \X_{i}\Vert^{2}] < \infty$ (协变量的有限方差和协方差)

1.  $\E[\X_{i}\X_{i}']$是正定的（协变量中没有线性依赖性）*  *回想一下，这些是对$(Y_{i}, \X_{i})$的联合分布的温和条件，特别是我们**不**假设 CEF（$\E[Y_{i} \mid \X_{i}]$）的线性，也不假设数据具有任何特定的分布。

我们可以将 OLS 估计量有益地分解为实际 BLP 系数加上估计误差，即 $$ \bhat = \left( \frac{1}{n} \sum_{i=1}^n \X_i\X_i' \right)^{-1} \left( \frac{1}{n} \sum_{i=1}^n \X_iY_i \right) = \bfbeta + \underbrace{\left( \frac{1}{n} \sum_{i=1}^n \X_i\X_i' \right)^{-1} \left( \frac{1}{n} \sum_{i=1}^n \X_ie_i \right)}_{\text{估计误差}}. $$

这种分解将帮助我们快速建立$\bhat$的一致性。根据大数定律，我们知道样本均值将以概率收敛到总体期望，因此我们有 $$ \frac{1}{n} \sum_{i=1}^n \X_i\X_i' \inprob \E[\X_i\X_i'] \equiv \mb{Q}_{\X\X} \qquad \frac{1}{n} \sum_{i=1}^n \X_ie_i \inprob \E[\X_{i} e_{i}] = \mb{0}, $$ 这意味着根据连续映射定理（逆函数是连续函数），我们有 $$ \bhat \inprob \bfbeta + \mb{Q}_{\X\X}^{-1}\E[\X_ie_i] = \bfbeta, $$ 线性投影假设确保 LLN 适用于这些样本均值，并且$\E[\X_{i}\X_{i}']$是可逆的。

**定理 7.1** 在上述线性投影假设下，OLS 估计量对于最佳线性投影系数$\bhat \inprob \bfbeta$是一致的。

因此，在相对温和的条件下，OLS 在大样本中应该接近总体线性回归。记住，如果 CEF 是非线性的，这可能与条件期望不相等。我们可以说的是，OLS 收敛到 CEF 的最佳*线性*近似。当然，这也意味着，如果 CEF 是线性的，那么 OLS 将一致地估计 CEF 的系数。

为了强调，对因变量的唯一假设是它（1）具有有限的方差，并且（2）是独立同分布的。在这个假设下，结果可以是连续的、分类的、二元的或事件计数。

接下来，我们希望为 OLS 系数建立渐近正态性结果。我们首先回顾一些关于中心极限定理的关键思想。

*CLT 提醒* *假设我们有一个关于独立同分布随机向量 $\X_1, \ldots, \X_n$ 的函数 $g(\X_{i})$，其中 $\E[g(\X_{i})] = 0$，因此 $\V[g(\X_{i})] = \E[g(\X_{i})g(\X_{i})']$. 然后，如果 $\E[\Vert g(\X_{i})\Vert^{2}] < \infty$，CLT 意味着 $$ \sqrt{n}\left(\frac{1}{n} \sum_{i=1}^{n} g(\X_{i}) - \E[g(\X_{i})]\right) = \frac{1}{\sqrt{n}} \sum_{i=1}^{n} g(\X_{i}) \indist \N(0, \E[g(\X_{i})g(\X_{i}')]) \tag{7.1}$$* *我们现在对分解进行操作，得到估计量的*稳定*版本，$$ \sqrt{n}\left( \bhat - \bfbeta\right) = \left( \frac{1}{n} \sum_{i=1}^n \X_i\X_i' \right)^{-1} \left( \frac{1}{\sqrt{n}} \sum_{i=1}^n \X_ie_i \right). $$ 回想一下，我们稳定估计量是为了确保它在样本量增长时具有固定的方差，从而使其具有非退化的渐近分布。稳定是通过渐近地将其中心化（即减去其收敛到的值）并乘以样本大小的平方根来实现的。我们已经证明，右侧的第一个项将以概率收敛到 $\mb{Q}_{\X\X}^{-1}$。注意，$\E[\X_{i}e_{i}] = 0$，因此我们可以将 方程 7.1 应用于第二个项。向量 $\X_ie_{i}$ 的协方差矩阵是 $$ \mb{\Omega} = \V[\X_{i}e_{i}] = \E[\X_{i}e_{i}(\X_{i}e_{i})'] = \E[e_{i}^{2}\X_{i}\X_{i}']. $$ CLT 将意味着 $$ \frac{1}{\sqrt{n}} \sum_{i=1}^n \X_ie_i \indist \N(0, \mb{\Omega}). $$ 结合这些事实与 Slutsky 定理，可以得出以下定理。

**定理 7.2** 假设线性投影假设成立，并且此外我们还有 $\E[Y_{i}^{4}] < \infty$ 和 $\E[\lVert\X_{i}\rVert^{4}] < \infty$。那么，OLS 估计量在渐近上是正态分布的，有 $$ \sqrt{n}\left( \bhat - \bfbeta\right) \indist \N(0, \mb{V}_{\bfbeta}), $$ 其中 $$ \mb{V}_{\bfbeta} = \mb{Q}_{\X\X}^{-1}\mb{\Omega}\mb{Q}_{\X\X}^{-1} = \left( \E[\X_i\X_i'] \right)^{-1}\E[e_i²\X_i\X_i']\left( \E[\X_i\X_i'] \right)^{-1}. $$

因此，在样本量足够大的情况下，我们可以用具有均值 $\bfbeta$ 和协方差矩阵 $\mb{V}_{\bfbeta}/n$ 的多元正态分布来近似 $\bhat$ 的分布。特别是，这个矩阵第 $j$ 个对角线的平方根将是 $\widehat{\beta}_j$ 的标准误差。了解 OLS 估计量的多元分布的形状将允许我们进行假设检验并为单个系数和系数组生成置信区间。但是，首先我们需要协方差矩阵的估计。**  **## 7.2 OLS 的方差估计

上节中 OLS 的渐近正态性在没有某种方法来估计协方差矩阵的情况下价值有限，$$ \mb{V}_{\bfbeta} = \mb{Q}_{\X\X}^{-1}\mb{\Omega}\mb{Q}_{\X\X}^{-1}. $$ 由于这里的每个项都是一个总体均值，这是一个放置插值估计器的理想位置。目前，我们将使用以下估计器：$$ \begin{aligned} \mb{Q}_{\X\X} &= \E[\X_{i}\X_{i}'] & \widehat{\mb{Q}}_{\X\X} &= \frac{1}{n} \sum_{i=1}^{n} \X_{i}\X_{i}' = \frac{1}{n}\Xmat'\Xmat \\ \mb{\Omega} &= \E[e_i²\X_i\X_i'] & \widehat{\mb{\Omega}} & = \frac{1}{n}\sum_{i=1}^n\widehat{e}_i²\X_i\X_i'. \end{aligned} $$ 在 定理 7.2 的假设下，大数定律将意味着这些估计量对于我们所需要的量是一致的，$\widehat{\mb{Q}}_{\X\X} \inprob \mb{Q}_{\X\X}$ 和 $\widehat{\mb{\Omega}} \inprob \mb{\Omega}$。我们可以将这些估计量代入方差公式，得到 $$ \begin{aligned} \widehat{\mb{V}}_{\bfbeta} &= \widehat{\mb{Q}}_{\X\X}^{-1}\widehat{\mb{\Omega}}\widehat{\mb{Q}}_{\X\X}^{-1} \\ &= \left( \frac{1}{n} \Xmat'\Xmat \right)^{-1} \left( \frac{1}{n} \sum_{i=1}^n\widehat{e}_i²\X_i\X_i' \right) \left( \frac{1}{n} \Xmat'\Xmat \right)^{-1}, \end{aligned} $$ 根据连续映射定理，这是一致的，$\widehat{\mb{V}}_{\bfbeta} \inprob \mb{V}_{\bfbeta}$.

这个估计量有时被称为**稳健方差估计量**，或者更准确地说，是**异方差一致性（HC）方差估计量**。为什么它稳健？考虑大多数统计软件包在估计 OLS 方差时所做的标准**同方差性**假设：误差的方差不依赖于协变量，或者说 $\V[e_{i}^{2} \mid \X_{i}] = \V[e_{i}^{2}]$。这个假设比所需的更强，我们可以依赖于一个更弱的假设，即平方误差与协变量的特定函数不相关：$$ \E[e_{i}^{2}\X_{i}\X_{i}'] = \E[e_{i}^{2}]\E[\X_{i}\X_{i}'] = \sigma^{2}\mb{Q}_{\X\X}, $$ 其中 $\sigma²$ 是残差的方差（因为 $\E[e_{i}] = 0$)。同方差性简化了稳定估计量 $\sqrt{n}(\bhat - \bfbeta)$ 的渐近方差为 $$ \mb{V}^{\texttt{lm}}_{\bfbeta} = \mb{Q}_{\X\X}^{-1}\sigma^{2}\mb{Q}_{\X\X}\mb{Q}_{\X\X}^{-1} = \sigma²\mb{Q}_{\X\X}^{-1}. $$ 我们已经有了 $\mb{Q}_{\X\X}$ 的估计量，但我们还需要一个 $\sigma²$ 的估计量。我们可以很容易地使用 SSR，$$ \widehat{\sigma}^{2} = \frac{1}{n-k-1} \sum_{i=1}^{n} \widehat{e}_{i}^{2}, $$ 其中我们在分母中使用 $n-k-1$ 而不是 $n$ 来纠正残差略小于实际误差的变量性（因为 OLS 机械地试图使残差很小）。对于一致的方差估计，$n-k -1$ 或 $n$ 都可以使用，因为两种方式下 $\widehat{\sigma}² \inprob \sigma²$。因此，在同方差性下，我们有 $$ \widehat{\mb{V}}_{\bfbeta}^{\texttt{lm}} = \widehat{\sigma}^{2}\left(\frac{1}{n}\Xmat'\Xmat\right)^{{-1}} = n\widehat{\sigma}^{2}\left(\Xmat'\Xmat\right)^{{-1}}, $$ 这就是 R 中的 `lm()` 和 Stata 中的 `reg` 所使用的标准方差估计量。

这两个估计量，$\widehat{\mb{V}}_{\bfbeta}$ 和 $\widehat{\mb{V}}_{\bfbeta}^{\texttt{lm}}$，如何比较？请注意，当同方差性成立时，HC 方差估计量和同方差性方差估计量都将是一致的。但是，正如“异方差一致性”标签所暗示的，当同方差性不成立时，只有 HC 方差估计量将是一致的。因此，$\widehat{\mb{V}}_{\bfbeta}$ 具有无论同方差性假设是否成立都具有一致性的优势。然而，这种优势是有代价的。当同方差性正确时，$\widehat{\mb{V}}_{\bfbeta}^{\texttt{lm}}$ 将该假设纳入估计量中，而 HC 方差估计量则必须对其进行估计。因此，当实际上确实存在同方差性时，HC 估计量将具有更高的方差（方差估计量将更加多变）。

既然我们已经建立了 OLS 估计量的渐近正态性和其方差的一致估计量，我们就可以继续使用我们在第一部分讨论的所有统计推断工具，包括假设检验和置信区间。

我们首先定义估计的**异方差一致标准误差**为 $$ \widehat{\se}(\widehat{\beta}_{j}) = \sqrt{\frac{[\widehat{\mb{V}}_{\bfbeta}]_{jj}}{n}}, $$ 其中 $[\widehat{\mb{V}}_{\bfbeta}]_{jj}$ 是 HC 方差估计器的第 $j$ 个对角线元素。请注意，我们在这里除以 $\sqrt{n}$，因为 $\widehat{\mb{V}}_{\bfbeta}$ 是稳定估计量 $\sqrt{n}(\bhat - \bfbeta)$ 的一致估计量，而不是估计量本身。

对于单个系数的假设检验和置信区间几乎与第一部分中提出的最一般情况完全相同。对于对 $H_0: \beta_j = b$ 与 $H_1: \beta_j \neq b$ 的双边检验，我们可以构建 t 统计量并得出结论，在零假设下，$$ \frac{\widehat{\beta}_j - b}{\widehat{\se}(\widehat{\beta}_{j})} \indist \N(0, 1). $$ 统计软件通常会提供对零假设（$X_{ij}$ 与 $Y_i$ 之间没有（部分）线性关系的零假设）的 t 统计量，$$ t = \frac{\widehat{\beta}_{j}}{\widehat{\se}(\widehat{\beta}_{j})}, $$ 这衡量了估计系数在标准误差中的大小。当 $\alpha = 0.05$ 时，渐近正态性意味着当 $t > 1.96$ 时，我们将拒绝这个零假设。我们可以使用以下公式形成渐近有效的置信区间 $$ \left[\widehat{\beta}_{j} - z_{\alpha/2}\;\widehat{\se}(\widehat{\beta}_{j}),\;\widehat{\beta}_{j} + z_{\alpha/2}\;\widehat{\se}(\widehat{\beta}_{j})\right]. $$ 由于以下我们将讨论的原因，标准软件通常依赖于 t 分布而不是正态分布来进行假设检验和置信区间。然而，在大样本中，这种差异影响甚微。

## 7.3 多参数的推断

在多个系数的情况下，我们可能会有涉及多个系数的假设。例如，考虑一个包含两个协变量交互作用的回归，$$ Y_i = \beta_0 + X_i\beta_1 + Z_i\beta_2 + X_iZ_i\beta_3 + e_i. $$ 假设我们想要检验 $X_i$ 不影响 $Y_i$ 的最佳线性预测器的假设。这将意味着 $$ H_{0}: \beta_{1} = 0 \text{ and } \beta_{3} = 0\quad\text{vs}\quad H_{1}: \beta_{1} \neq 0 \text{ or } \beta_{3} \neq 0, $$ 其中我们通常将零假设更紧凑地写成 $H_0: \beta_1 = \beta_3 = 0$。

为了检验这个零假设，我们需要一个检验统计量，它能在两个假设之间进行区分：当备择假设为真时，它应该很大；当零假设为真时，它应该足够小。对于单个系数，我们通常使用 $t$ 统计量来检验 $H_0: \beta_j = b_0$ 的零假设，$$ t = \frac{\widehat{\beta}_{j} - b_{0}}{\widehat{\se}(\widehat{\beta}_{j})}, $$ 我们通常取绝对值 $|t|$ 作为我们估计的极端程度的度量，给定零分布。但请注意，我们也可以使用 $t$ 统计量的平方，它是 $$ t^{2} = \frac{\left(\widehat{\beta}_{j} - b_{0}\right)^{2}}{\V[\widehat{\beta}_{j}]} = \frac{n\left(\widehat{\beta}_{j} - b_{0}\right)^{2}}{[\mb{V}_{\bfbeta}]_{[jj]}}. \tag{7.2}$$

当 $|t|$ 是我们用于双尾检验的常用检验统计量时，我们也可以等价地使用 $t²$ 并得出完全相同的结论（只要我们知道在零假设下 $t²$ 的分布）。结果发现，检验统计量的 $t²$ 版本更容易推广到比较多个系数。这种检验统计量的版本提出了区分零假设和备择假设的另一种一般方法：通过计算它们之间的平方距离并除以估计量的方差。

我们能否将这个想法推广到关于多个参数的假设？将零假设每个成分的平方距离之和加起来是直接的。对于我们的交互作用示例，这将是对 $\widehat{\beta}_1² + \widehat{\beta}_3²$，记住，然而，一些估计系数比其他系数更嘈杂，因此我们应该像对 $t$ 统计量那样考虑不确定性。

在多个参数和多个系数的情况下，方差现在需要矩阵代数。我们可以将关于系数的线性函数的任何假设写成 $H_{0}: \mb{L}\bfbeta = \mb{c}$。例如，在交互作用的情况下，我们有 $$ \mb{L} = \begin{pmatrix} 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ \end{pmatrix} \qquad \mb{c} = \begin{pmatrix} 0 \\ 0 \end{pmatrix} $$ 因此，$\mb{L}\bfbeta = \mb{0}$ 等价于 $\beta_1 = 0$ 和 $\beta_3 = 0$。请注意，使用其他 $\mb{L}$ 矩阵，我们可以表示更复杂的假设，如 $2\beta_1 - \beta_2 = 34$，尽管我们主要坚持更简单的函数。令 $\widehat{\bs{\theta}} = \mb{L}\bhat$ 为系数函数的 OLS 估计。通过 delta 方法（在第 3.9 节中讨论），我们有 $$ \sqrt{n}\left(\mb{L}\bhat - \mb{L}\bfbeta\right) \indist \N(0, \mb{L}\mb{V}_{\bfbeta}\mb{L}'). $$ 我们现在可以通过取 $\mb{L}\bhat - \mb{c}$ 加权距离，权重为方差协方差矩阵 $\mb{L}\mb{V}_{\bfbeta}\mb{L}'$ 来推广 方程 7.2 中的平方 $t$ 统计量， $$ W = n(\mb{L}\bhat - \mb{c})'(\mb{L}\mb{V}_{\bfbeta}\mb{L}')^{-1}(\mb{L}\bhat - \mb{c}), $$ 这被称为 **Wald 统计量**。这个统计量将 t 统计量的思想推广到多个参数。使用 t 统计量，我们重新定位以使均值为 0，并通过标准误差进行除法以获得方差为 1。如果我们忽略中间的方差加权，我们就有 $(\mb{L}\bhat - \mb{c})'(\mb{L}\bhat - \mb{c})$，这仅仅是估计值与零假设的平方偏差之和。包括 $(\mb{L}\mb{V}_{\bfbeta}\mb{L}')^{-1}$ 权重的作用是将 $\mb{L}\bhat - \mb{c}$ 的分布重新缩放，使其围绕 0 旋转对称（因此结果维度是不相关的），每个维度具有相等的方差为 1。通过这种方式，Wald 统计量将随机向量转换为均值中心化和方差为 1（即 t 统计量），同时也使向量中的结果随机变量不相关。¹

为什么要以这种方式转换数据？图 7.1 显示了 OLS 回归中两个系数的假设联合分布的等高线图。我们可能想知道分布中不同点与均值之间的距离，在这种情况下是 $(1, 2)$。如果不考虑联合分布，圆显然比三角形更接近均值。然而，观察分布上的两个点，圆在比三角形更低的等高线上，这意味着对于这个特定的分布，圆比三角形更极端。因此，Wald 统计量考虑了 $\mb{L}\bhat$ 在给定 $\mb{L}\bhat$ 分布的情况下到达 $\mb{c}$ 的“爬升”程度。

![图片](img/1786b9fe10a772e336cbed68c1df9692.png)

图 7.1：两个斜率系数的假设联合分布。圆圈通过标准欧几里得距离更接近分布中心，但一旦考虑联合分布，三角形则更接近。

如果 $\mb{L}$ 只有一行，我们的 Wald 统计量与平方 $t$ 统计量相同，$W = t²$。这一事实将帮助我们思考 $W$ 的渐近分布。注意，当 $n\to\infty$ 时，我们知道由于 $\bhat$ 的渐近正态性，$$ t = \frac{\widehat{\beta}_{j} - \beta_{j}}{\widehat{\se}[\widehat{\beta}_{j}]} \indist \N(0,1) $$ 因此 $t²$ 将在分布上收敛到 $\chi²_1$（因为 $\chi²_1$ 分布只是一个标准正态分布的平方）。在通过协方差矩阵重新中心和缩放后，$W$ 收敛到 $q$ 个独立正态变量的平方和，其中 $q$ 是 $\mb{L}$ 的行数，或者等价地，零假设所隐含的限制数量。因此，在 $\mb{L}\bhat = \mb{c}$ 的零假设下，我们有 $W \indist \chi²_{q}$。

我们需要定义拒绝域，以便在假设检验中使用 Wald 统计量。因为我们正在对 $W \geq 0$ 中的每个距离进行平方，所以 $W$ 的较大值表示与零假设在任一方向上的更大不一致。因此，对于 $\alpha$ 水平的联合零假设检验，我们只需要一个单侧拒绝域，形式为 $\P(W > w_{\alpha}) = \alpha$。获取这些值是直接的（参见上面的提示）。对于 $q = 2$ 和 $\alpha = 0.05$，临界值大约为 6。

*卡方临界值* 我们可以使用 R 中的 `qchisq()` 函数来获取 $\chi²_q$ 分布的临界值。例如，如果我们想获取临界值 $w$，使得 $\P(W > w_{\alpha}) = \alpha$，对于我们的双参数交互示例，我们可以使用：

```r
qchisq(p = 0.95, df = 2)
```

*```r
[1] 5.991465
```**  **Wald 统计量不是标准统计软件函数（如 R 中的`lm()`）提供的常见检验，尽管“手动”实现相当直接。或者，像`{aod}`（https://cran.r-project.org/web/packages/aod/index.html）或`{clubSandwich}`（http://jepusto.github.io/clubSandwich/）这样的包提供了该检验的实现。大多数 OLS 软件实现（如 R 中的`lm()`）报告的是 F 统计量，其公式为$$ F = \frac{W}{q}. $$ 这通常在 W 中使用同方差方差估计器$\mb{V}^{\texttt{lm}}_{\bfbeta}$。此类检验报告的 p 值使用$F_{q,n-k-1}$分布，因为这是当误差（a）同方差和（b）正态分布时$F$统计量的确切分布。当这些假设不成立时，在统计理论中，$F$分布没有依据，但它比$\chi²_q$分布略为保守，并且随着$n\to\infty$，从$F$统计量得出的推断将收敛到从$\chi²_q$分布得出的推断。因此，它可以作为一个*临时*的小样本调整 Wald 检验。例如，如果我们使用$F_{q,n-k-1}$与交互作用示例，其中$q=2$，并且样本大小为$n = 100$，那么在这种情况下，F 检验的临界值（$\alpha = 0.05$）为

```r
qf(0.95, df1 = 2, df2 = 100 - 4)
```

*```r
[1] 3.091191
```*  *此结果意味着 Wald 统计量尺度上的临界值为 6.182（乘以$q = 2$）。与基于$\chi²_2$分布的早期临界值 5.991 相比，我们可以看到，即使在中等大小的数据集中，推断也将非常相似。

最后，请注意，R 中的`lm()`函数报告的 F 统计量是对所有系数（除了截距）同时等于 0 的检验。在现代定量社会科学中，这种检验很少具有实质性意义。***  ***## 7.4 线性 CEF 的有限样本性质

所有上述结果都是大样本性质，我们尚未解决有限样本性质，如抽样方差或无偏性。在上述线性投影假设下，OLS 通常在没有更强假设的情况下是有偏的。本节介绍了更强的假设，这将使我们能够为 OLS 建立更强的性质。然而，通常记住，这些更强的假设可能是错误的。

*假设：线性回归模型* *1.  变量$(Y_{i}, \X_{i})$满足线性 CEF 假设。 $$ \begin{aligned} Y_{i} &= \X_{i}'\bfbeta + e_{i} \\ \E[e_{i}\mid \X_{i}] & = 0. \end{aligned} $$

1.  设计矩阵是可逆的 $\E[\X_{i}\X_{i}'] > 0$（正定）。*  *我们在第五章线性模型中广泛讨论了线性 CEF 的概念。然而，回想一下，如果模型是 **饱和的** 或者当模型中的系数与 $\X_i$ 的唯一值一样多时，CEF 可能是线性的。当模型不是饱和的，线性 CEF 假设仅仅是：一个假设。这个假设能做什么？它可以帮助在有限样本中建立一些良好的统计性质。

在继续之前，请注意，当我们关注 OLS 的有限样本推断时，我们关注的是其基于观察协变量的性质 **条件于观察到的协变量**，例如 $\E[\bhat \mid \Xmat]$ 或 $\V[\bhat \mid \Xmat]$。这种历史原因在于研究者通常选择这些自变量，因此它们不是随机的。因此，有时在某些较老的文章中，$\Xmat$ 被视为“固定”的，甚至可能省略显式的条件语句。

**定理 7.3** 在线性回归模型假设下，OLS 对总体回归系数是无偏的，$$ \E[\bhat \mid \Xmat] = \bfbeta, $$ 其条件抽样方差为 $$ \mb{\V}_{\bhat} = \V[\bhat \mid \Xmat] = \left( \Xmat'\Xmat \right)^{-1}\left( \sum_{i=1}^n \sigma²_i \X_i\X_i' \right) \left( \Xmat'\Xmat \right)^{-1}, $$ 其中 $\sigma²_{i} = \E[e_{i}^{2} \mid \Xmat]$。

*证明*。为了证明条件无偏性，回想一下我们可以将 OLS 估计量写成 $$ \bhat = \bfbeta + (\Xmat'\Xmat)^{-1}\Xmat'\mb{e}, $$ 因此取（条件）期望，我们有 $$ \E[\bhat \mid \Xmat] = \bfbeta + \E[(\Xmat'\Xmat)^{-1}\Xmat'\mb{e} \mid \Xmat] = \bfbeta + (\Xmat'\Xmat)^{-1}\Xmat'\E[\mb{e} \mid \Xmat] = \bfbeta, $$ 因为在线性 CEF 假设下 $\E[\mb{e}\mid \Xmat] = 0$。

对于条件抽样方差，我们可以使用我们已有的分解，$$ \V[\bhat \mid \Xmat] = \V[\bfbeta + (\Xmat'\Xmat)^{-1}\Xmat'\mb{e} \mid \Xmat] = (\Xmat'\Xmat)^{-1}\Xmat'\V[\mb{e} \mid \Xmat]\Xmat(\Xmat'\Xmat)^{-1}. $$ 由于 $\E[\mb{e}\mid \Xmat] = 0$，我们知道 $\V[\mb{e}\mid \Xmat] = \E[\mb{ee}' \mid \Xmat]$，这是一个对角线元素为 $\E[e_{i}^{2} \mid \Xmat] = \sigma²_i$ 和非对角线元素 $\E[e_{i}e_{j} \Xmat] = \E[e_{i}\mid \Xmat]\E[e_{j}\mid\Xmat] = 0$ 的矩阵，其中第一个等式来自于误差在单位之间的独立性。因此，$\V[\mb{e} \mid \Xmat]$ 是一个对角矩阵，其对角线元素为 $\sigma²_i$，这意味着 $$ \Xmat'\V[\mb{e} \mid \Xmat]\Xmat = \sum_{i=1}^n \sigma²_i \X_i\X_i', $$ 建立了条件抽样方差。

这意味着，对于协变量的任何实现，$\Xmat$，OLS 对于真实的回归系数 $\bfbeta$ 是无偏的。根据迭代期望定律，我们也知道它无条件地无偏²，因为 $$ \E[\bhat] = \E[\E[\bhat \mid \Xmat]] = \bfbeta. $$ 这两个陈述之间的差异通常并不特别有意义。

有很多方差在飞来飞去，所以回顾它们是有帮助的。上面，我们推导了 $\mb{Z}_{n} = \sqrt{n}(\bhat - \bfbeta)$ 的渐近方差，$$ \mb{V}_{\bfbeta} = \left( \E[\X_i\X_i'] \right)^{-1}\E[e_i²\X_i\X_i']\left( \E[\X_i\X_i'] \right)^{-1}, $$ 这意味着 $\bhat$ 的近似方差将是 $\mb{V}_{\bfbeta} / n$，因为 $$ \bhat = \frac{Z_n}{\sqrt{n}} + \bfbeta \quad\implies\quad \bhat \overset{a}{\sim} \N(\bfbeta, n^{-1}\mb{V}_{\bfbeta}), $$ 其中 $\overset{a}{\sim}$ 表示渐近分布为。在线性 CEF 下，$\bhat$ 的条件抽样方差具有类似的形式，并且将与

$$ \mb{V}_{\bhat} = \left( \Xmat'\Xmat \right)^{-1}\left( \sum_{i=1}^n \sigma²_i \X_i\X_i' \right) \left( \Xmat'\Xmat \right)^{-1} \approx \mb{V}_{\bfbeta} / n. $$ 在实践中，这两个推导导致基本相同的方差估计量。回想一下，异方差一致方差估计量 $$ \widehat{\mb{V}}_{\bfbeta} = \left( \frac{1}{n} \Xmat'\Xmat \right)^{-1} \left( \frac{1}{n} \sum_{i=1}^n\widehat{e}_i²\X_i\X_i' \right) \left( \frac{1}{n} \Xmat'\Xmat \right)^{-1}, $$ 是渐近方差的合法插值估计量，并且 $$ \widehat{\mb{V}}_{\bhat} = n^{-1}\widehat{\mb{V}}_{\bfbeta}. $$ 因此，在实践中，线性 CEF 下的渐近和有限样本结果证明了相同的方差估计量是合理的。

### 7.4.1 同方差性下的线性 CEF 模型

如果我们愿意假设标准误差是同方差的，我们可以为 OLS 推导出更强的结果。更强的假设通常会导致更强的结论，但显然，这些结论可能不会对假设违反具有鲁棒性。但是，误差的同方差性是一个历史上非常重要的假设，因此像 R 中的`lm()`这样的统计软件实现默认假设了它。

*假设：具有线性 CEF 的同方差性* 除了线性 CEF 假设之外，我们进一步假设 $$ \E[e_i² \mid \X_i] = \E[e_i²] = \sigma², $$ 或者说误差的方差不依赖于协变量。***定理 7.4** 在具有同方差误差的线性 CEF 模型下，条件抽样方差是 $$ \mb{V}^{\texttt{lm}}_{\bhat} = \V[\bhat \mid \Xmat] = \sigma² \left( \Xmat'\Xmat \right)^{-1}, $$ 并且方差估计量 $$ \widehat{\mb{V}}^{\texttt{lm}}_{\bhat} = \widehat{\sigma}² \left( \Xmat'\Xmat \right)^{-1} \quad\text{其中，}\quad \widehat{\sigma}² = \frac{1}{n - k - 1} \sum_{i=1}^n \widehat{e}_i² $$ 是无偏的，$\E[\widehat{\mb{V}}^{\texttt{lm}}_{\bhat} \mid \Xmat] = \mb{V}^{\texttt{lm}}_{\bhat}$.

*证明*. 在同方差性 $\sigma²_i = \sigma²$ 对所有 $i$ 成立的情况下。回想一下，$\sum_{i=1}^n \X_i\X_i' = \Xmat'\Xmat$. 因此，从 定理 7.3 的条件抽样方差，$$ \begin{aligned} \V[\bhat \mid \Xmat] &= \left( \Xmat'\Xmat \right)^{-1}\left( \sum_{i=1}^n \sigma² \X_i\X_i' \right) \left( \Xmat'\Xmat \right)^{-1} \\ &= \sigma²\left( \Xmat'\Xmat \right)^{-1}\left( \sum_{i=1}^n \X_i\X_i' \right) \left( \Xmat'\Xmat \right)^{-1} \\&= \sigma²\left( \Xmat'\Xmat \right)^{-1}\left( \Xmat'\Xmat \right) \left( \Xmat'\Xmat \right)^{-1} \\&= \sigma²\left( \Xmat'\Xmat \right)^{-1} = \mb{V}^{\texttt{lm}}_{\bhat}. \end{aligned} $$

为了无偏性，我们只需证明 $\E[\widehat{\sigma}^{2} \mid \Xmat] = \sigma²$. 回想一下，我们定义 $\mb{M}_{\Xmat}$ 为残差生成器，因为 $\mb{M}_{\Xmat}\mb{Y} = \widehat{\mb{e}}$. 我们可以利用这一点将残差与标准误差联系起来，$$ \mb{M}_{\Xmat}\mb{e} = \mb{M}_{\Xmat}\mb{Y} - \mb{M}_{\Xmat}\Xmat\bfbeta = \mb{M}_{\Xmat}\mb{Y} = \widehat{\mb{e}}, $$ 因此 $$ \V[\widehat{\mb{e}} \mid \Xmat] = \mb{M}_{\Xmat}\V[\mb{e} \mid \Xmat] = \mb{M}_{\Xmat}\sigma², $$ 其中第一个等式成立是因为 $\mb{M}_{\Xmat} = \mb{I}_{n} - \Xmat (\Xmat'\Xmat)^{-1} \Xmat'$ 在给定 $\Xmat$ 的条件下是常数。注意，这个矩阵的对角线元素是特定残差 $\widehat{e}_i$ 的方差，而湮灭矩阵的对角线元素是 $1 - h_{ii}$（因为 $h_{ii}$ 是 $\mb{P}_{\Xmat}$ 的对角线元素）。因此，我们有 $$ \V[\widehat{e}_i \mid \Xmat] = \E[\widehat{e}_{i}^{2} \mid \Xmat] = (1 - h_{ii})\sigma^{2}. $$ 在上一章的 第 6.9.1 节 中，我们建立了这些杠杆值的一个性质是 $\sum_{i=1}^n h_{ii} = k+ 1$，所以 $\sum_{i=1}^n 1- h_{ii} = n - k - 1$，并且我们有 $$ \begin{aligned} \E[\widehat{\sigma}^{2} \mid \Xmat] &= \frac{1}{n-k-1} \sum_{i=1}^{n} \E[\widehat{e}_{i}^{2} \mid \Xmat] \\ &= \frac{\sigma^{2}}{n-k-1} \sum_{i=1}^{n} 1 - h_{ii} \\ &= \sigma^{2}. \end{aligned} $$ 这就证明了 $\E[\widehat{\mb{V}}^{\texttt{lm}}_{\bhat} \mid \Xmat] = \mb{V}^{\texttt{lm}}_{\bhat}$.

因此，在线性 CEF 模型和误差的同方差性假设下，我们有一个无偏方差估计量，它是残差平方和与设计矩阵的简单函数。大多数统计软件包使用 $\widehat{\mb{V}}^{\texttt{lm}}_{\bhat}$ 来估计标准误差。

在同方差性假设下，对于线性 CEF 我们可以推导出的最终结果是一个最优性结果。也就是说，我们可能会问是否存在另一个 $\bfbeta$ 的估计量，在降低抽样方差的意义上优于 OLS。也许令人惊讶的是，没有线性估计量对于 $\bfbeta$ 有更低的条件方差，这意味着 OLS 是 **最佳线性无偏估计量**，通常简称为 BLUE。这个结果广为人知，被称为高斯-马尔可夫定理。

**定理 7.5** 设 $\widetilde{\bfbeta} = \mb{AY}$ 是 $\bfbeta$ 的一个线性无偏估计量。在具有同方差误差的线性 CEF 模型下，$$ \V[\widetilde{\bfbeta}\mid \Xmat] \geq \V[\bhat \mid \Xmat]. $$

*证明*。注意，如果$\widetilde{\bfbeta}$是无偏的，那么$\E[\widetilde{\bfbeta} \mid \Xmat] = \bfbeta$，因此 $$ \bfbeta = \E[\mb{AY} \mid \Xmat] = \mb{A}\E[\mb{Y} \mid \Xmat] = \mb{A}\Xmat\bfbeta, $$ 这意味着 $\mb{A}\Xmat = \mb{I}_n$。将竞争者重新写为 $\widetilde{\bfbeta} = \bhat + \mb{BY}$，其中， $$ \mb{B} = \mb{A} - \left(\Xmat'\Xmat\right)^{-1}\Xmat'. $$ 并注意 $\mb{A}\Xmat = \mb{I}_n$ 意味着 $\mb{B}\Xmat = 0$。我们现在有 $$ \begin{aligned} \widetilde{\bfbeta} &= \left( \left(\Xmat'\Xmat\right)^{-1}\Xmat' + \mb{B}\right)\mb{Y} \\ &= \left( \left(\Xmat'\Xmat\right)^{-1}\Xmat' + \mb{B}\right)\Xmat\bfbeta + \left( \left(\Xmat'\Xmat\right)^{-1}\Xmat' + \mb{B}\right)\mb{e} \\ &= \bfbeta + \mb{B}\Xmat\bfbeta + \left( \left(\Xmat'\Xmat\right)^{-1}\Xmat' + \mb{B}\right)\mb{e} \\ &= \bfbeta + \left( \left(\Xmat'\Xmat\right)^{-1}\Xmat' + \mb{B}\right)\mb{e} \end{aligned} $$ 因此，竞争者的方差为 $$ \begin{aligned} \V[\widetilde{\bfbeta} \mid \Xmat] &= \left( \left(\Xmat'\Xmat\right)^{-1}\Xmat' + \mb{B}\right)\V[\mb{e}\mid \Xmat]\left( \left(\Xmat'\Xmat\right)^{-1}\Xmat' + \mb{B}\right)' \\ &= \sigma^{2}\left( \left(\Xmat'\Xmat\right)^{-1}\Xmat' + \mb{B}\right)\left( \Xmat\left(\Xmat'\Xmat\right)^{-1} + \mb{B}'\right) \\ &= \sigma^{2}\left(\left(\Xmat'\Xmat\right)^{-1}\Xmat'\Xmat\left(\Xmat'\Xmat\right)^{-1} + \left(\Xmat'\Xmat\right)^{-1}\Xmat'\mb{B}' + \mb{B}\Xmat\left(\Xmat'\Xmat\right)^{-1} + \mb{BB}'\right)\\ &= \sigma^{2}\left(\left(\Xmat'\Xmat\right)^{-1} + \mb{BB}'\right)\\ &\geq \sigma^{2}\left(\Xmat'\Xmat\right)^{-1} \\ &= \V[\bhat \mid \Xmat] \end{aligned} $$ 第一个等式来自协方差矩阵的性质，第二个等式是由于同方差性假设，第四个等式是由于 $\mb{B}\Xmat = 0$，这同样意味着 $\Xmat'\mb{B}' = 0$。第五个不等式成立，因为如果 $\mb{B}$ 是满秩的（我们假设它是），那么形式为 $\mb{BB}'$ 的矩阵乘积是正定的。

在这个证明中，我们看到了竞争估计量的方差为 $\sigma²\left(\left(\Xmat'\Xmat\right)^{-1} + \mb{BB}'\right)$，我们论证了在矩阵意义上它是“大于 0”的，这也被称为正定。这在实际中意味着什么？记住，任何正定矩阵都必须有严格正的对角线元素，而 $\V[\bhat \mid \Xmat]$ 和 $V[\widetilde{\bfbeta}\mid \Xmat]$ 的对角线元素是单个参数的方差，即 $\V[\widehat{\beta}_{j} \mid \Xmat]$ 和 $\V[\widetilde{\beta}_{j} \mid \Xmat]$。因此，对于 $\widetilde{\bfbeta}$，单个参数的方差将大于 $\bhat$。

许多教科书将高斯-马尔可夫定理作为 OLS 方法相对于其他方法的重大优势，但认识到其局限性是至关重要的。它需要线性性和同方差误差假设，而这些假设在许多应用中可能是错误的。

最后，请注意，虽然我们已经为线性估计量展示了这个结果，但汉森（2022）证明了一个更通用的结果版本，该结果适用于任何无偏估计量。**  **## 7.5 正态线性模型

最后，我们添加了最强大但也是最不受欢迎的经典线性回归假设之一：（条件）误差的正态性。从历史上看，使用这个假设的原因是，在不知道 $\bhat$ 的抽样分布的情况下，有限样本推断会遇到障碍。在线性 CEF 模型下，我们看到了 $\bhat$ 是无偏的，在同方差性下，我们可以产生条件方差的無偏估计量。但对于假设检验或生成置信区间，我们需要对估计量做出概率陈述，为此，我们需要知道其确切分布。当样本量很大时，我们可以依赖中心极限定理，知道 $\bhat$ 大约是正态分布的。但在小样本情况下我们该如何进行呢？从历史上看，我们会假设误差的（条件）正态性，基本上是带着一些我们知道我们可能是错误的，但希望不是太错误的知识进行。

*正态线性回归模型* *除了线性 CEF 假设之外，我们还假设 $$ e_i \mid \Xmat \sim \N(0, \sigma²). $$*  *有几个重要的观点：

+   这里的假设并不是说 $(Y_{i}, \X_{i})$ 是联合正态分布的（尽管这足以使假设成立），而是 $Y_i$ 在给定 $\X_i$ 的条件下是正态分布的。

+   注意到标准回归模型内嵌了同方差性的假设。

**定理 7.6** 在正态线性回归模型下，我们有 $$ \begin{aligned} \bhat \mid \Xmat &\sim \N\left(\bfbeta, \sigma^{2}\left(\Xmat'\Xmat\right)^{-1}\right) \\ \frac{\widehat{\beta}_{j} - \beta_{j}}{[\widehat{\mb{V}}^{\texttt{lm}}_{\bhat}]_{jj}/\sqrt{n}} &\sim t_{n-k-1} \\ W/q &\sim F_{q, n-k-1}. \end{aligned} $$

这个定理说明，在正态线性回归模型中，系数遵循正态分布，t 统计量遵循 t 分布，Wald 统计量的变换遵循 F 分布。这些都是 **精确** 的结果，不依赖于大样本近似。在误差条件正态性的假设下，这些结果对于 $n = 5$ 和 $n = 500,000$ 都是有效的。

很少有人相信误差遵循正态分布，那么为什么还要展示这些结果呢？不幸的是，大多数统计软件在计算测试的 p 值或构建置信区间时，对 OLS 的实现隐式地假设了这一点。例如，在 R 中，`lm()` 报告的 $t$ 统计量相关的 p 值依赖于 $t_{n-k-1}$ 分布，而用于构建置信区间的临界值也使用该分布。当正态性不成立时，没有原则上的理由使用 $t$ 或 $F$ 分布进行这种推断。但我们可以采取这种 *临时* 程序，并给出两个合理的解释：

+   $\bhat$ 是渐近正态的。然而，这种近似在小样本中可能并不好。在这种情况下，$t$ 分布将使推断更加保守（置信区间更宽，检验拒绝区域更小），这可能会帮助抵消其在小样本中对正态分布的较差近似。

+   当 $n\to\infty$ 时，$t_{n-k-1}$ 将收敛到标准正态分布，因此这种 *临时* 调整对中等或大样本的影响不会很大。

这些论点并不很有说服力，因为 $t$ 近似在有限样本中是否比正态分布更好并不清楚。但当我们寻找更多数据时，这可能是我们能做的最好的事情。*  *## 7.6 概述

在本章中，我们讨论了 OLS 的大样本性质，这些性质相当强。在温和的条件下，OLS 对总体线性回归系数是一致的，并且是渐近正态的。OLS 估计量的方差，以及方差估计量，取决于投影误差是否假设与协变量无关（**同方差**）或可能相关（**异方差**）。对于单个 OLS 系数的置信区间和假设检验与本书第一部分讨论的大致相同，如果我们假设条件期望函数是线性的，我们还可以获得 OLS 的有限样本性质，如条件无偏性。如果我们进一步假设误差是正态分布的，我们可以推导出对所有样本大小都有效的置信区间和假设检验。

Hansen, Bruce E. 2022\. “A Modern Gauss–Markov Theorem.” *Econometrica* 90 (3): 1283–94\. [`doi.org/10.3982/ECTA19255`](https://doi.org/10.3982/ECTA19255).

* * *

1.  Wald 统计量形式为加权内积，$\mb{x}'\mb{Ay}$，其中 $\mb{A}$ 是对称正定加权矩阵。↩︎

1.  在处理离散协变量时，我们基本上忽略了某些边缘情况。特别是，我们假设 $\Xmat'\Xmat$ 的概率为 1 是非奇异的。然而，如果我们有一个二元协变量，这个假设可能会失败，因为整个列全部为 1 或全部为 0 的可能性（尽管可能性很小），这将导致 $\Xmat'\Xmat$ 是奇异矩阵。实际上这并不是什么大问题，但这确实意味着我们理论上必须忽略这个问题，或者关注条件无偏性。↩︎********
