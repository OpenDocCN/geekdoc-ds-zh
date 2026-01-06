# 第三章 计算贝叶斯数据分析

> 原文：[`bruno.nicenboim.me/bayescogsci/ch-compbda.html`](https://bruno.nicenboim.me/bayescogsci/ch-compbda.html)

在上一章中，我们学习了如何解析地推导出我们模型中参数的后验分布。然而，在实践中，这仅适用于非常有限的情况。尽管贝叶斯法则中的分子，即未归一化的后验，很容易计算（通过乘以似然和概率密度/质量函数），但分母，即边缘似然，需要我们进行积分；参见方程 (3.1)。

\[\begin{equation} \begin{aligned} p(\boldsymbol{\Theta}|\boldsymbol{y}) &= \cfrac{ p(\boldsymbol{y}|\boldsymbol{\Theta}) \cdot p(\boldsymbol{\Theta}) }{p(\boldsymbol{y})}\\ p(\boldsymbol{\Theta}|\boldsymbol{y}) &= \cfrac{ p(\boldsymbol{y}|\boldsymbol{\Theta}) \cdot p(\boldsymbol{\Theta}) }{\int_{\boldsymbol{\Theta}} p(\boldsymbol{y}|\boldsymbol{\Theta}) \cdot p(\boldsymbol{\Theta}) d\boldsymbol{\Theta} } \end{aligned} \tag{3.1} \end{equation}\]

除非我们处理的是共轭分布，否则求解将极其困难，或者根本不存在解析解。这曾是过去贝叶斯分析的主要瓶颈，需要贝叶斯分析实践者自己编程实现近似方法，才能开始贝叶斯分析。幸运的是，今天许多免费可用的概率编程语言（下一节将列出）允许我们定义模型，而无需获得相关数值技术的专业知识。

## 3.1 通过采样推导后验

假设我们想要从第 2.2 节推导出模型的后验分布，即“*雨伞*”的完形填空概率 \(\theta\) 的后验分布，给定以下数据：一个单词（例如，“*雨伞*”）在 100 次中回答了 80 次，并且假设二项分布作为似然函数，以及 \(\mathit{Beta}(a=4,b=4)\) 作为完形填空概率的先验分布。如果我们能从 \(\theta\) 的后验分布中生成样本⁹，而不是通过解析方法得到后验分布，那么在足够多的样本下，我们将得到后验分布的良好近似。从后验分布中获取样本将是我们将在本书中讨论的模型中唯一可行的选项。当我们说“获取样本”时，我们是在谈论类似于我们使用 `rbinom()` 或 `rnorm()` 从特定分布中获取样本的情况。有关采样算法的更多详细信息，请参阅进一步阅读部分（第 3.10 节）。

多亏了概率编程语言，获取这些样本将会相对简单，我们将在下一节中更详细地讨论如何实现。目前，让我们假设我们使用某种概率编程语言从闭合概率的后验分布中获得了 20000 个样本：0.782，0.722，0.782，0.727，0.839，0.828，0.769，0.806，0.839，0.832，0.728，0.805，0.764，0.756，0.791，0.739，0.774，0.758，0.815，0.808。图 3.1 显示后验的近似看起来与解析得到的后验非常相似。解析计算得到的均值和方差与近似值之间的差异分别是\(-0.0001\)和\(0.000003\)。

![通过采样生成的\(\theta\)后验分布样本的直方图。黑色线表示解析得到的后验密度图。](img/ed657f238dd81d502c9e96e015f32093.png)

图 3.1：通过采样生成的\(\theta\)后验分布样本的直方图。黑色线表示解析得到的后验密度图。

## 3.2 使用 Stan 的贝叶斯回归模型：brms

贝叶斯统计学的流行激增与计算能力的提升以及概率编程语言的出现密切相关，例如 WinBUGS（Lunn 等 2000）、JAGS（Plummer 2016）、PyMC3（Salvatier，Wiecki 和 Fonnesbeck 2016）、Turing（Ge，Xu 和 Ghahramani 2018）和 Stan（Carpenter 等 2017）；有关历史回顾，请参阅 Plummer (2022)。

这些概率编程语言允许用户在不处理（大部分情况下）采样过程的复杂性定义模型。然而，由于用户必须使用特定的语法完全指定统计模型，因此需要学习一门新语言。¹⁰ 此外，为了正确参数化模型并避免收敛问题，需要了解采样过程的一些知识（这些内容将在第九章中详细讨论）。

有一些替代方案允许在 R 中执行贝叶斯推理，而无需完全手动指定模型。R 包`rstanarm`（Goodrich 等人 2018）和`brms`（Bürkner 2024）提供了许多流行 R 模型拟合函数的贝叶斯等价，例如(g)lmer（Bates，Mächler 等人 2015）等；`rstanarm`和`brms`都使用 Stan 作为估计和采样的后端。与`rstanarm`和`brms`相比，R 包 R-INLA（Lindgren 和 Rue 2015）允许拟合有限的选择似然函数和先验，R-INLA 可以拟合可以表示为潜在高斯模型的模型）。此包使用集成嵌套拉普拉斯近似（INLA）方法进行贝叶斯推理，而不是像上述其他概率语言中的采样算法。另一个替代方案是 JASP（JASP Team 2019），它为频率派和贝叶斯建模提供图形用户界面，并旨在成为 SPSS 的开源替代品。

我们在本书的这一部分将重点关注`brms`。这是因为它可以帮助从频率派模型平滑过渡到它们的贝叶斯等价模型。`brms`包不仅足够强大以满足许多认知科学家的统计需求，而且还有额外的优势，即 Stan 代码可以被检查（使用`brms`函数`make_stancode()`和`make_standata()`），使用户能够自定义他们的模型或从`brms`内部生成的代码中学习，最终完全用 Stan 编写模型。在第八章中关于 Stan 的介绍中，我们实现了当前章节和下一章中提出的模型。

### 3.2.1 简单线性模型：单个受试者重复按按钮（手指敲击任务）

为了说明使用`brms`拟合模型的基本步骤，考虑以下关于手指敲击任务的示例（参见 Hubel 等人 2013 的综述）。假设一个受试者首先看到一片空白屏幕。然后，经过一段时间（比如说 200 毫秒），受试者会在屏幕中央看到一个十字，一旦他们看到十字，他们就会尽可能快地按下空格键，直到实验结束（361 次试验）。这里的因变量是按下空格键到下一次按下所需的时间（以毫秒为单位）。因此，每个试验中的数据是手指敲击时间（以毫秒为单位）。假设研究问题是：这个特定受试者按下键需要多长时间？

让我们基于以下假设对数据进行建模：

1.  存在一个真实的（未知的）基础时间，\(\mu\)毫秒，受试者需要按下空格键。

1.  在这个过程中有一些噪声。

1.  噪声是正态分布的（考虑到手指敲击作为响应时间通常也是偏斜的，这个假设是有问题的；我们将在后面修正这个假设）。¹¹

这意味着每个观测值 \(n\) 的可能性将是：

\[\begin{equation} t_n \sim \mathit{Normal}(\mu, \sigma) \tag{3.2} \end{equation}\]

其中 \(n =1, \ldots, N\)，\(t\) 是因变量（以毫秒为单位的手指敲击时间）。变量 \(N\) 表示数据点的总数。符号 \(\mu\) 表示正态分布函数的 *位置*；位置参数将分布沿水平轴左移或右移。对于正态分布，位置也是分布的均值。符号 \(\sigma\) 表示分布的 *尺度*；当尺度减小时，分布变窄。这种压缩随着尺度参数趋近于零而趋近于尖峰（所有概率质量都集中在一点附近）。对于正态分布，尺度也是其标准差。

读者可能已经遇到了方程 (3.2) 中所示的形式，在方程 (3.3) 中呈现：

\[\begin{equation} t_n = \mu + \varepsilon_n \hbox{, 其中 } \varepsilon_n \stackrel{iid}{\sim} \mathit{Normal}(0,\sigma) \tag{3.3} \end{equation}\]

当模型以这种方式编写时，应理解为每个数据点 \(t_n\) 都围绕一个均值 \(\mu\) 有一定的变异性，这种变异性具有标准差 \(\sigma\)。术语“iid”（独立同分布）意味着残差是独立生成的（它们与任何其他残差值都不相关）。模型在方程 (3.3) 中的表述与说观测数据点 \(t_n\) 是 iid 并且来自 \(Normal(\mu,\sigma)\) 分布的表述完全相同。

对于一个频率派模型，它将给出我们按空格键所需时间的最大似然估计（样本均值），这将足够的信息来在 R 中编写公式 `t ~ 1`，并将其与数据一起插入 `lm()` 函数：`lm(t ~ 1, data)`。这里的 `1` 的含义是 `lm()` 将估计模型中的截距，即我们例子中的 \(\mu\) 的估计。如果读者对线性模型完全不熟悉，章节 4.5 中的参考文献将会有所帮助。

对于贝叶斯线性模型，我们还需要为模型中的两个参数定义先验。假设我们确信按下键所需的时间将是正的并且低于一分钟（或 \(60000\) 毫秒），但我们不希望就哪些值更有可能做出承诺。我们在 \(\sigma\) 中编码我们对任务噪声的了解：我们知道这个参数必须是正的，我们假设任何低于 \(2000\) 毫秒的值可能性相同。这些先验通常强烈不建议使用：平坦（或非常宽）的先验几乎永远不会是我们所知道的最佳近似。先验规格将在在线章节 E 中详细讨论。

在这种情况下，即使我们对任务了解很少，我们也知道按下空格键最多只需要几秒钟。为了教学目的，本节将使用平坦先验；下一节将展示更现实的先验使用。

\[\begin{equation} \begin{aligned} \mu &\sim \mathit{Uniform}(0, 60000) \\ \sigma &\sim \mathit{Uniform}(0, 2000) \end{aligned} \tag{3.4} \end{equation}\]

首先，从 `bcogsci` 包中加载数据框 `df_spacebar`：

```r
data("df_spacebar")
df_spacebar
```

```r
## # A tibble: 361 × 2
##       t trial
##   <int> <int>
## 1   141     1
## 2   138     2
## 3   128     3
## # ℹ 358 more rows
```

在做任何事情之前总是绘制数据是一个好主意；参见图 3.2。正如我们所怀疑的，数据看起来有点偏斜，但我们暂时忽略这一点。

```r
 ggplot(df_spacebar, aes(t)) +
 geom_density() +
 xlab("Finger tapping times") +
 ggtitle("Button-press data")
```

![可视化按钮点击数据。](img/b14a590009b985b0df7a58a9b31ed27c.png)

图 3.2：可视化按钮点击数据。

#### 3.2.1.1 在 `brms` 中指定模型

使用以下方式使用 `brms` 拟合由方程 (3.2) 和 (3.4) 定义的模型。

```r
fit_press <-
 brm(t ~  1,
 data = df_spacebar,
 family = gaussian(),
 prior =
 c(prior(uniform(0, 60000), class = Intercept, lb = 0, ub = 60000),
 prior(uniform(0, 2000), class = sigma, lb = 0, ub = 2000)),
 chains = 4,
 iter = 2000,
 warmup = 1000)
```

`brms` 代码与使用 `lm()` 拟合的模型有一些不同。在这个初始阶段，我们将关注以下选项：

1.  术语 `family = gaussian()` 明确指出底层似然函数是正态分布（高斯和正态是同义词）。这个细节在 R 函数 `lm()` 中是隐含的。其他链接函数也是可能的，正如 R 中的 `glm()` 函数一样。对于 `brms`，与 `lm()` 函数对应的默认值是 `family = gaussian()`。

1.  术语 `prior` 将一个先验规格列表作为参数。尽管这种先验规格是可选的，但研究人员应该始终明确指定每个先验。否则，`brms` 将默认定义先验，这些先验可能或可能不适合研究领域。在分布有受限覆盖范围的情况下，也就是说，不是每个值都是有效的（例如，小于 \(0\) 或大于 \(60000\) 的截距不是有效的），我们需要使用 `lb` 和 `ub` 设置上下边界。¹²

1.  术语 `chains` 指的是采样（默认为四个）的独立运行次数。

1.  术语`iter`指的是采样器为从每个参数的后验分布中采样所进行的迭代次数（默认为`2000`）。

1.  术语`warmup`指的是从采样开始到最终被丢弃的迭代次数（默认为`iter`的一半）。

最后三个选项`chains`、`iter`、`warmup`决定了采样算法的行为：汉密尔顿蒙特卡洛（Hamiltonian Monte Carlo）的 No-U-Turn Sampler（NUTS；Hoffman 和 Gelman 2014）扩展。我们将在第八章中更深入地讨论采样，但基本过程将在下面解释。

#### 3.2.1.2 简要概述采样和收敛

以下是对采样过程的极度简化：代码规范默认从四个独立链开始。每个链在多维空间中“搜索”后验分布的样本，其中每个参数对应一个维度。这个空间的大小由先验和似然决定。链从随机位置开始，在每个迭代中为所有参数各自取一个样本。当采样开始时，样本可能属于也可能不属于参数的后验分布。最终，链将结束在后验分布的附近，从那时起，样本将属于后验。

因此，当采样开始时，来自不同链的样本可能彼此相距甚远，但最终它们将“收敛”并开始从后验分布中提供样本。尽管我们无法保证运行链的迭代次数足以从后验分布中获得样本，但`brms`（以及 Stan）的默认值在许多情况下足以实现收敛。当默认的迭代次数不足时，`brms`（实际上是 Stan）将打印出警告，并提供解决收敛问题的建议。如果所有链都收敛到相同的分布，通过移除“预热”样本，我们确保不会从初始路径到后验分布中获得样本。在`brms`中，默认情况下，每个链中总迭代次数的一半（默认为 2000）将计入“预热”。因此，如果使用四个链和默认迭代次数运行模型，我们将从四个链中获得总共 4000 个样本，在丢弃预热迭代后。

图 3.3(a) 显示了从预热阶段开始的链的路径。这样的图称为轨迹图。预热只是为了说明目的；通常，应该在（假设）收敛已经达到的点之后（即虚线之后）检查链。收敛发生后，一个视觉诊断检查是链应该看起来像“胖毛毛虫”。比较图 3.3(a) 中我们模型的轨迹图与图 3.3(b) 中未收敛的模型的轨迹图。

轨迹图并不总是关于收敛的诊断。轨迹图可能看起来很好，但模型可能没有收敛。幸运的是，Stan 会自动使用链的信息运行几个诊断，如果在拟合模型后没有警告信息，并且轨迹图看起来正常，我们可以合理地确信模型已经收敛，并假设我们的样本来自真实的后验分布。然而，为了使诊断工作，有必要运行多个链（最好是四个），并且至少有几千次迭代。

![（a）按钮点击数据的 brms 模型的轨迹图。所有链的起始值都高于 200，并且位于图表之外。（b）未收敛的模型的轨迹图。我们可以通过观察链没有重叠——每个链似乎是从不同的分布中进行采样的——来诊断非收敛。](img/88e39ee2135d22bfbf43d3e82f42d9f6.png)

图 3.3: (a) 按钮点击数据的 `brms` 模型的轨迹图。所有链的起始值都高于 `200`，并且位于图表之外。(b) 未收敛的模型的轨迹图。我们可以通过观察链没有重叠——每个链似乎是从不同的分布中进行采样的——来诊断非收敛。

#### 3.2.1.3 `brms` 的输出

一旦模型已经拟合（并且假设我们没有收到关于收敛问题的警告信息），我们可以使用 `as_draws_df()`（它存储有关链的元数据）或使用 `as.data.frame()` 打印出每个参数的后验分布的样本：

```r
as_draws_df(fit_press) %>%
 head(3)
```

```r
## # A draws_df: 3 iterations, 1 chains, and 5 variables
##   b_Intercept sigma Intercept lprior  lp__
## 1         168    25       168    -19 -1683
## 2         169    24       169    -19 -1683
## 3         169    25       169    -19 -1683
## # ... hidden reserved variables {'.chain', '.iteration', '.draw'}
```

`brms`输出中的`b_Intercept`项对应于我们的\(\mu\)。我们可以忽略最后三列：`Intercept`是一个假设中心预测器的辅助截距，在这里与`b_Intercept`一致，并在在线部分 A.3 中讨论，`lprior`是（联合）先验分布的对数密度，它存在于与`priorsense`包的兼容性中（[`github.com/n-kall/priorsense`](https://github.com/n-kall/priorsense)），而`lp`实际上不是后验的一部分，它是每个迭代的未归一化后验的对数密度（`lp`在在线部分 B.1 中讨论，应在第八章的上下文中阅读）。

使用`plot(fit_press)`绘制预热后的每个参数的密度和迹图（图 3.4）。

![按钮按压数据的`brms`模型的密度和迹图](img/2921e1da57361c328e8e5cbf2e062094.png)

图 3.4：按钮按压数据的`brms`模型的密度和迹图。

使用`brms`拟合打印对象提供了一个相当详尽的总结：

```r
fit_press
# posterior_summary(fit_press) is also useful
```

```r
##  Family: gaussian 
##   Links: mu = identity; sigma = identity 
## Formula: t ~ 1 
##    Data: df_spacebar (Number of observations: 361) 
##   Draws: 4 chains, each with iter = 2000; warmup = 1000; thin = 1;
##          total post-warmup draws = 4000
## 
## Regression Coefficients:
##           Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
## Intercept   168.63      1.29   166.17   171.23 1.00     3482     2624
## 
## Further Distributional Parameters:
##       Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
## sigma    25.02      0.96    23.26    26.98 1.00     2990     2588
## 
## Draws were sampled using sampling(NUTS). For each parameter, Bulk_ESS
## and Tail_ESS are effective sample size measures, and Rhat is the potential
## scale reduction factor on split chains (at convergence, Rhat = 1).
```

`Estimate`只是后验样本的均值，`Est.Error`是后验的标准差，置信区间（CIs）标记了 95%可信区间的下限和上限（为了区分可信区间和频率主义置信区间，前者将缩写为 CrIs）：

```r
as_draws_df(fit_press)$b_Intercept %>%  mean()
```

```r
## [1] 169
```

```r
as_draws_df(fit_press)$b_Intercept %>%  sd()
```

```r
## [1] 1.29
```

```r
as_draws_df(fit_press)$b_Intercept %>%
 quantile(c(0.025, .975))
```

```r
##  2.5% 97.5% 
##   166   171
```

此外，摘要提供了每个参数的`Rhat`、`Bulk_ESS`和`Tail_ESS`。R-hat 比较了每个参数的链间和链内估计。当链没有很好地混合时，R-hat 大于 1，只有当所有参数的 R-hats 都小于 1.05 时，才能依赖模型。（否则将出现 R 警告）。Bulk ESS（bulk effective sample size）是后验分布主体部分的采样效率的度量，即均值和中位数估计的有效样本量，而 tail ESS（tail effective sample size）表示分布尾部的采样效率，即 5%和 95%分位数的最小有效样本量。有效样本量通常小于预热后的样本数量，因为链的样本不是独立的（它们在一定程度上是相关的），并且与独立样本相比，携带有关后验分布的信息较少。然而，在某些情况下，有效样本量实际上可能大于预热后的样本数量。这可能发生在后验分布呈正态分布的参数（在无约束空间中，见在线部分 B.1）并且对其他参数的依赖性较低（Vehtari 等人 2021）。非常低的有效样本量表明采样问题（并伴随 R 警告），通常与未正确混合的链一起出现。作为经验法则，Vehtari 等人(2021)建议统计摘要至少需要有效样本量\(400\)。

我们可以看到我们可以无问题地拟合我们的模型，并得到一些参数的后验分布。然而，在我们能够解释模型的后验分布之前，我们应该问自己以下问题：

1.  先验编码了哪些信息？先验有意义吗？

1.  模型中假设的似然对于数据来说有意义吗？

我们将通过查看*先验和后验预测分布*以及进行敏感性分析来尝试回答这些问题。这将在接下来的章节中解释。

## 3.3 先验预测分布

我们已经为我们的线性模型定义了以下先验：

\[\begin{equation} \begin{aligned} \mu &\sim \mathit{Uniform}(0, 60000) \\ \sigma &\sim \mathit{Uniform}(0, 2000) \end{aligned} \tag{3.5} \end{equation}\]

这些先验编码了我们对于未来研究中预期看到的数据类型的假设。为了理解这些假设，我们将从模型中生成数据；这种完全由先验分布生成的数据分布称为先验预测分布。重复生成先验预测分布有助于我们检查先验是否合理。我们在这里想了解的是，先验是否能够生成看起来真实的数据？

形式上，我们想了解数据点 \(y_{pred_1},\dots,y_{pred_N}\) 的密度 \(p(\cdot)\)，这些数据点来自长度为 \(N\) 的数据集 \(\boldsymbol{y_{pred}}\)，给定一个先验向量 \(\boldsymbol{\Theta}\) 和我们的似然函数 \(p(\cdot|\boldsymbol{\Theta})\)；（在我们的例子中，\(\boldsymbol{\Theta}=\langle\mu,\sigma \rangle\)）。先验预测密度如下所示：

\[\begin{equation} \begin{aligned} p(\boldsymbol{y_{pred}}) &= p(y_{pred_1},\dots,y_{pred_n})\\ &= \int_{\boldsymbol{\Theta}} p(y_{pred_1}|\boldsymbol{\Theta})\cdot p(y_{pred_2}|\boldsymbol{\Theta})\cdots p(y_{pred_N}|\boldsymbol{\Theta}) p(\boldsymbol{\Theta}) \, \mathrm{d}\boldsymbol{\Theta} \end{aligned} \end{equation}\]

从本质上讲，参数向量被积分出去（参见第 1.7 节）。这给出了在考虑任何观察结果之前，给定先验和似然函数的可能数据集的概率分布，*在没有任何观察结果被考虑之前*。

通过从先验分布中生成样本，可以通过计算方法执行积分。

这里是生成先验预测分布的一种方法：

重复以下操作多次：

1.  从每个先验中取一个样本。

1.  将这些样本插入模型中用作似然函数的概率密度/质量函数，以生成数据集 \(y_{pred_1},\ldots,y_{pred_n}\)。

每个样本是一个想象中的或潜在的数据集。

创建一个执行此操作的函数：

```r
normal_predictive_distribution <-
 function(mu_samples, sigma_samples, N_obs) {
 # empty data frame with headers:
 df_pred <-  tibble(trialn = numeric(0),
 t_pred = numeric(0),
 iter = numeric(0))
 # i iterates from 1 to the length of mu_samples,
 # which we assume is identical to
 # the length of the sigma_samples:
 for (i in seq_along(mu_samples)) {
 mu <-  mu_samples[i]
 sigma <-  sigma_samples[i]
 df_pred <-  bind_rows(df_pred,
 tibble(trialn = seq_len(N_obs), 
 # seq_len generates 1, 2,..., N_obs
 t_pred = rnorm(N_obs, mu, sigma),
 iter = i))
 }
 df_pred
 }
```

以下代码生成了我们定义在 3.2.1 节中的模型的 \(1000\) 个先验预测分布样本。这意味着它将生成 \(361000\) 个预测值（每个 \(1000\) 次模拟的 \(361\) 个预测观测值）。尽管这种方法可行，但它相当慢（几秒钟）。请参阅在线部分 A.1，以获取此函数的更有效版本。第 3.7.2 节将展示使用 `brms` 从先验中采样的可能性，通过设置 `sample_prior = "only"` 来忽略数据中的 `t`。然而，由于 `brms` 仍然依赖于 Stan 的采样器，该采样器使用汉密尔顿蒙特卡洛方法，先验采样过程也可能无法收敛，尤其是在使用非常不提供信息的先验时，例如本例中使用的先验。相比之下，我们上面使用的 `rnorm()` 函数不会出现收敛问题，并且总是会生成多组先验预测数据 \(y_{pred_1},\ldots,y_{pred_n}\)。

以下代码使用了 `tic()` 和 `toc()` 函数从 `tictoc` 包中打印出运行代码所需的时间。

```r
N_samples <-  1000
N_obs <-  nrow(df_spacebar)
mu_samples <-  runif(N_samples, 0, 60000)
sigma_samples <-  runif(N_samples, 0, 2000)
tic()
prior_pred <-
 normal_predictive_distribution(mu_samples = mu_samples,
 sigma_samples = sigma_samples,
 N_obs = N_obs)
toc()
```

```r
## 1.39 sec elapsed
```

```r
prior_pred
```

```r
## # A tibble: 361,000 × 3
##   trialn t_pred  iter
##    <dbl>  <dbl> <dbl>
## 1      1 16710\.     1
## 2      2 16686\.     1
## 3      3 17245\.     1
## # ℹ 360,997 more rows
```

```r
## 0.321 sec elapsed
```

图 3.5 展示了先验预测分布的前 18 个样本（即 18 个独立生成的先验预测数据集），下面是相应的代码。

```r
prior_pred %>%
 filter(iter <=  18) %>%
 ggplot(aes(t_pred)) +
 geom_histogram(aes(y = after_stat(density))) +
 xlab("predicted t (ms)") +
 theme(axis.text.x = element_text(angle = 40,
 vjust = 1,
 hjust = 1)) +
 scale_y_continuous(limits = c(0, 0.0005),
 breaks = c(0, 0.00025, 0.0005),
 name = "density") +
 facet_wrap(~iter, ncol = 3)
```

![图 3.2.1 节中定义的模型先验预测分布的 18 个样本。](img/29726df01a7c3d4fd9275f104415874b.png)

图 3.5：3.2.1 节中定义的模型先验预测分布的 18 个样本。

图 3.5 中的先验预测分布显示了由模型生成的不太现实的数据集：除了数据集显示手指敲击时间分布是对称的——我们知道从先前对这类数据的经验来看，它们通常是右偏斜的——一些数据集显示了不切实际的长手指敲击时间。更糟糕的是，如果我们检查足够的先验预测数据样本，将变得明显，有几个数据集的手指敲击时间值为负。

我们还可以查看先验预测数据中汇总统计量的分布。即使我们事先不知道数据应该是什么样子，我们很可能对可能的均值、最小值或最大值有一些预期。例如，在按钮按压示例中，平均手指敲击时间在 \(200\)-\(600\) 毫秒之间似乎是合理的假设；手指敲击时间不太可能低于 \(50\) 毫秒，即使长时间的注意力分散也不会超过几秒钟。¹³ 图 3.6 展示了三个汇总统计量的分布。

![3.2.1 节中定义的按钮按压模型均值、最小值和最大值的先验预测分布](img/6582c96d95048d11a69bd66a87b88252.png)

图 3.6：3.2.1 节中定义的按钮按压模型均值、最小值和最大值的先验预测分布。

图 3.6 显示，我们使用的先验信息比可能有的要少得多：我们的先验编码了任何在\(0\)到\(60000\)毫秒之间的均值都是等可能的信息。显然，接近\(0\)或\(60000\)毫秒的值会非常令人惊讶。这种广泛的均值范围是由于\(\mu\)上的均匀先验造成的。同样，最大值也非常“均匀”，其范围比预期的要宽得多。最后，在最小值的分布中，出现了负的指节敲击时间。这可能会令人惊讶（我们的\(\mu\)先验排除了负值），但负值出现的原因是先验与似然性一起被解释（Gelman, Simpson, and Betancourt 2017)，而似然性是一个正态分布，即使位置参数\(\mu\)被限制为只有正值，它也会允许负样本的存在。

总结上述讨论，示例中使用的先验显然并不非常现实，考虑到我们可能了解的此类按钮按压任务的指节敲击时间。这引发了一个问题：我们应该选择什么样的先验？在下一节中，我们将考虑这个问题。

## 3.4 先验的影响：敏感性分析

在本书中我们将遇到的大多数情况下，我们可以从四种主要的先验类别中进行选择。在贝叶斯统计中，并没有固定的命名法来对不同的先验进行分类。对于本书，我们为每种类型的先验选择了特定的名称，但这只是一种为了保持一致性的惯例。本书中也没有讨论其他类别的先验。一个例子是不恰当的先验，如\(\mathit{Uniform}(-\infty,+\infty)\)，这些不是合适的概率分布，因为曲线下的面积不等于 1。

当思考先验时，读者不应该纠结于特定类型的先验的确切名称；他们更应该关注该先验在研究问题背景下的含义。

### 3.4.1 平坦、无信息的先验

一种选择是尽可能选择无信息的先验。这种方法的理念是让数据“自己说话”，并且不使用“主观”的先验来偏置统计推断。这种方法存在几个问题：首先，先验与似然一样主观，实际上，不同似然的选择可能对后验的影响比不同先验的选择要大得多。其次，无信息先验在一般情况下是不现实的，因为它们在先验分布的支持范围内对所有值给予相同的权重，忽略了通常对感兴趣参数有一些最小信息的事实。通常，至少，我们知道数量级（反应时间或手指敲击时间将是毫秒而不是天，脑电图信号是微伏而不是伏特等）。第三，无信息先验会使采样变慢，并可能导致收敛问题。除非有大量数据，否则明智的做法是避免这样的先验。第四，并不总是清楚给定的分布的哪种参数化应该分配平坦的先验。例如，正态分布有时是根据其标准差（\(\sigma\)）、方差（\(\sigma²\)）或精度（\(1/\sigma²\)）定义的：标准差的平坦先验对分布的精度来说不是平坦的。尽管有时可以找到一个无信息的先验，它在参数变化下是不变的（也称为杰弗里斯先验；Jaynes 2003，第 6.15 节；Jeffreys 1939，第三章），但这并不总是如此。最后，如果需要计算贝叶斯因子，无信息先验可能导致非常误导性的结论（第十三章）。

在本章讨论的按钮按压示例中，一个平坦的无信息先验的例子将是 \(\mu \sim \mathit{Uniform}(-10^{20},10^{20})\)。在毫秒尺度上，这是一个非常奇怪的先验，用于表示平均按钮按压时间的参数：它允许不可能的大正值，同时也允许负的按钮按压时间，这当然是不可行的。技术上可以使用这样的先验，但这没有太多意义。

### 3.4.2 正则化先验

如果没有太多先验信息（并且这些信息不能通过关于问题的推理来得出），并且有足够的数据（“足够”的含义将在我们查看具体例子时变得清晰），使用 *正则化先验* 是可以的。这些先验会降低极端值的权重（即，它们提供正则化），它们通常不是非常有信息量，并且主要让似然在确定后验时起主导作用。这些先验是理论中立的；也就是说，它们通常不会将参数偏向任何先验信念或理论支持的价值。这种先验背后的想法是帮助稳定计算。这些先验有时在贝叶斯文献中被称为 *弱信息先验* 或 *温和信息先验*。对于许多应用，它们表现良好，但如第十三章所述，如果需要计算贝叶斯因子，它们往往会有问题。

在按钮按压的例子中，一个正则化先验的例子是 \(\mu \sim \mathit{Normal}_{+}(0,1000)\)。这是一个截断在 \(0\) 毫秒处的正态分布先验，允许按钮按压时间有相对受限的正值范围（大约，达到 \(2000\) 毫秒左右）。这是一个正则化先验，因为它排除了负的按钮按压时间，并降低了 \(2000\) 毫秒以上的极端值。在这里，可以假设一个非截断的先验，如 \(\mathit{Normal}(0,1000)\)。即使我们不期望 \(\mu\) 为负，这也可以被视为一个正则化先验；数据将确保后验分布具有正值（因为我们不会有负的按钮按压时间）。

### 3.4.3 原则先验

这里提出的想法是拥有先验，这些先验包含了研究者所拥有的所有（或大部分）与理论无关的信息。由于通常人们知道自己的数据是什么样的，以及不是什么样的，因此可以使用先验预测检查来构建真正反映潜在数据集特性的先验。在这本书中，将会出现许多这类先验的例子。

在按钮按压数据中，一个原则先验的例子是 \(\mu \sim \mathit{Normal}_{+}(250,100)\)。这个先验并不过于严格，但代表了对可能的按钮按压时间的猜测。使用原则先验进行的先验预测检查应该产生依赖变量的现实分布。

### 3.4.4 信息先验

存在一些情况，其中存在大量的先验知识。一般来说，除非有很好的理由使用相对信息丰富的先验（参见第十三章），否则让先验对后验有太多影响不是一个好主意。一个信息先验很重要的例子是，当调查一个我们无法获得许多受试者的语言受损人群时，但关于研究主题已经存在大量先前发表的论文。

在按钮点击数据中，一个信息性先验可以基于先前发表或现有数据的荟萃分析，或者基于对研究主题的专家（或多位专家）进行先验提取的结果。一个信息性先验的例子是\(\mu \sim \mathit{Normal}_{+}(200,20)\)。这个先验将对\(\mu\)的后验产生一些影响，尤其是在数据相对稀疏的情况下。

这四个选项构成一个连续体。从最后一个模型（3.2.1 节）中的均匀先验位于平坦、无信息性和正则化先验之间。在实际数据分析情况下，我们大多数情况下会选择介于正则化和原则性之间的先验。按照上述定义的信息性先验将相对较少地使用；但在进行贝叶斯因子分析（第十三章）时，它们变得更为重要考虑。

## 3.5 使用不同先验重新审视按钮点击示例

如果对之前定义的模型（在 3.2.1 节中）使用更宽的先验，会发生什么？假设认为\(-10^{6}\)到\(10^{6}\)毫秒之间的每个值对位置参数\(\mu\)来说都是等可能的。这个先验显然是不现实的，实际上完全没有意义：我们并不期望出现负的指节敲击时间。至于标准差，可以假设\(0\)到\(10^{6}\)之间的任何值都是等可能的。¹⁴似然值保持不变。

\[\begin{equation} \begin{aligned} \mu &\sim \mathit{Uniform}(-10^{6}, 10^{6}) \\ \sigma &\sim \mathit{Uniform}(0, 10^{6}) \end{aligned} \tag{3.6} \end{equation}\]

```r
# The default settings are used when they are not set explicitly:
# 4 chains, with half of the iterations (set as 3000) as warmup.
fit_press_unif <-  brm(t ~  1,
 data = df_spacebar,
 family = gaussian(),
 prior = c(prior(uniform(-10⁶, 10⁶),
 class = Intercept,
 lb = -10⁶,
 ub = 10⁶),
 prior(uniform(0, 10⁶),
 class = sigma,
 lb = 0,
 ub = 10⁶)),
 iter = 3000,
 control = list(adapt_delta = .99,
 max_treedepth = 15))
```

即使使用这些极端不现实的先验，这些先验要求我们改变迭代次数`iter`以及`adapt_delta`和`max_treedepth`的默认值以实现收敛，模型的输出几乎与之前的一个相同（见图 3.7）。

```r
fit_press_unif
```

```r
## ...
## Population-Level Effects: 
##           Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
## Intercept   168.68      1.28   166.18   171.22 1.00     4192     2840
## 
## Family Specific Parameters: 
##       Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
## sigma    25.02      0.94    23.24    27.02 1.01      602      469
## 
## ...
```

接下来，考虑如果使用非常信息性的先验会发生什么。假设最可能的均值非常接近\(400\)毫秒，而指节敲击时间的标准差非常接近\(100\)毫秒。鉴于这是一个按钮点击时间的模型，这样的信息性先验似乎是不正确的——\(200\)毫秒似乎是一个更现实的平均按钮点击时间，而不是\(400\)毫秒。你可以通过亲自进行实验并查看记录的时间来检查这一点；Linger ([`tedlab.mit.edu/~dr/Linger/`](http://tedlab.mit.edu/~dr/Linger/))这样的软件使得设置这样的实验变得容易。

\(\mathit{Normal}_+\) 符号表示截断在零的正态分布，只允许正值（参见在线部分 A.2 详细讨论了这种类型的分布）。尽管标准差先验被限制为正值，但我们不需要在先验中添加 `lb = 0`，`brms` 会自动考虑这一点。

\[\begin{equation} \begin{aligned} \mu &\sim \mathit{Normal}(400, 10) \\ \sigma &\sim \mathit{Normal}_+(100, 10) \end{aligned} \tag{3.7} \end{equation}\]

```r
fit_press_inf <-  brm(t ~  1,
 data = df_spacebar,
 family = gaussian(),
 prior = c(prior(normal(400, 10), class = Intercept),
 # `brms` knows that SDs need to be bounded
 # to exclude values below zero:
 prior(normal(100, 10), class = sigma)))
```

尽管这些先验分布不切实际但富有信息性，但似然函数主要起主导作用，新的后验均值和可信区间仅比之前的估计值多出几毫秒（见图 3.7）：

```r
fit_press_inf
```

```r
## ...
## Population-Level Effects: 
##           Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
## Intercept   172.95      1.42   170.20   175.77 1.00     2639     2367
## 
## Family Specific Parameters: 
##       Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
## sigma    26.11      1.06    24.19    28.26 1.00     2599     2561
## 
## ...
```

作为敏感性分析的最终示例，选择一些原则性的先验。假设我们有一些先前类似实验的经验，假设平均响应时间预计在 \(200\) 毫秒左右，平均值的 95% 置信区间从 \(0\) 到 \(400\) 毫秒。这种不确定性可能过大，但可能希望允许比实际认为合理的更多的不确定性（这种在允许更多不确定性方面的保守性有时在贝叶斯统计中被称为克罗默尔规则；参见 O’Hagan 和 Forster 2004，第 3.19 节）。在这种情况下，可以选择先验 \(\mathit{Normal}(200, 100)\)。鉴于实验只涉及一个受试者且任务非常简单，可能不会期望残差标准差 \(\sigma\) 很大：例如，可以选择截断正态分布的位置为 \(50\) 毫秒，但仍允许相对较大的不确定性：\(Normal_{+}(50,50)\)。先验规格总结如下。

\[\begin{equation} \begin{aligned} \mu &\sim \mathit{Normal}(200, 100) \\ \sigma &\sim \mathit{Normal}_+(50, 50) \end{aligned} \end{equation}\]

为什么这些先验是原则性的？这里的“原则性”在很大程度上取决于我们的领域知识。在线章节 E 讨论了在指定先验时如何使用领域知识。

通过图形化地展示先验分布并进行先验预测检验，可以更好地理解特定一组先验分布的含义。这些步骤在这里被省略了，但这些内容在在线章节 E 和 F 中有详细讨论。这些章节将提供更多关于选择先验和开发贝叶斯数据分析的原理性工作流程的详细信息。

```r
fit_press_prin <-
 brm(t ~  1,
 data = df_spacebar,
 family = gaussian(),
 prior = c(prior(normal(200, 100), class = Intercept),
 prior(normal(50, 50), class = sigma)))
```

新的估计值几乎与之前相同（见图 3.7）：

```r
fit_press_prin
```

```r
## ...
## Population-Level Effects: 
##           Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
## Intercept   168.68      1.30   166.17   171.26 1.00     3775     2676
## 
## Family Specific Parameters: 
##       Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
## sigma    25.01      0.95    23.21    27.07 1.00     4210     2622
## 
## ...
```

上述使用不同先验的例子不应被误解为先验从不重要。当有足够的数据时，似然函数将在确定后验分布中占主导地位。构成“足够”数据的标准也是模型复杂性的函数；一般来说，更复杂的模型需要更多的数据。

![比较具有“现实”上界均匀先验分布（尽管仍不推荐）的模型后验分布 fit_press 与具有极不切实际宽泛先验的模型 fit_press_unif、具有误指定信息先验的模型 fit_press_inf 和具有原则性先验的模型 fit_press_prin。所有后验几乎重叠，除了 fit_press_inf 的后验，它被移动了几毫秒。](img/6b8b133fb6c8ee6ae69b7a0aaea21ef2.png)

图 3.7：比较具有“现实”上界均匀先验分布（尽管仍不推荐）的模型后验分布 `fit_press` 与具有极不切实际宽泛先验的模型 `fit_press_unif`、具有误指定信息先验的模型 `fit_press_inf` 和具有原则性先验的模型 `fit_press_prin`。所有后验几乎重叠，除了 `fit_press_inf` 的后验，它被移动了几毫秒。

即使在数据足够且似然函数在确定后验分布中占主导地位的情况下，正则化、原则性的先验（即与我们关于数据的先验信念更一致的先验）通常可以加速模型收敛。

为了确定后验受到先验影响的程度，进行敏感性分析是一个好习惯：尝试不同的先验，要么验证后验没有发生剧烈变化，要么报告后验如何受到某些特定先验的影响（例如，从心理语言学中，参见 Vasishth 等人 2013；Vasishth 和 Engelmann 2022）。第十三章将演示敏感性分析对于报告贝叶斯因子至关重要；即使在先验的选择不影响后验分布的情况下，它通常也会影响贝叶斯因子。

## 3.6 后验预测分布

后验预测分布是从模型（似然函数和先验）生成的数据集的集合。在考虑数据后获得参数的后验分布后，可以使用后验分布从模型生成未来的数据。换句话说，给定模型参数的后验分布，后验预测分布为我们提供了关于未来数据可能看起来像什么的某些指示。

一旦得到后验分布 \(p(\boldsymbol{\Theta}\mid \boldsymbol{y})\)，可以通过积分参数来生成基于这些分布的预测：

\[\begin{equation} p(\boldsymbol{y_{pred}}\mid \boldsymbol{y} ) = \int_{\boldsymbol{\Theta}} p(\boldsymbol{y_{pred}}, \boldsymbol{\Theta}\mid \boldsymbol{y})\, \mathrm{d}\boldsymbol{\Theta}= \int_{\boldsymbol{\Theta}} p(\boldsymbol{y_{pred}}\mid \boldsymbol{\Theta},\boldsymbol{y})p(\boldsymbol{\Theta}\mid \boldsymbol{y})\, \mathrm{d}\boldsymbol{\Theta} \end{equation}\]

假设给定 \(\boldsymbol{\Theta}\) 的条件下，过去和未来的观测是条件独立的¹⁵，即 \(p(\boldsymbol{y_{pred}}\mid \boldsymbol{\Theta},\boldsymbol{y})= p(\boldsymbol{y_{pred}}\mid \boldsymbol{\Theta})\)，则上述方程可以写为：

\[\begin{equation} p(\boldsymbol{y_{pred}}\mid \boldsymbol{y} )=\int_{\boldsymbol{\Theta}} p(\boldsymbol{y_{pred}}\mid \boldsymbol{\Theta}) p(\boldsymbol{\Theta}\mid \boldsymbol{y})\, \mathrm{d}\boldsymbol{\Theta} \tag{3.8} \end{equation}\]

在方程 (3.8) 中，我们只对 \(\boldsymbol{y_{pred}}\) 进行了关于 \(\boldsymbol{y}\) 的条件化，我们没有对未知的内容（\(\boldsymbol{\Theta}\)）进行条件化；未知参数已经被积分掉了。这个后验预测分布与使用频率主义方法获得的预测有重要区别。频率主义方法给出了每个预测观测的最大似然估计 \(\boldsymbol{\Theta}\)（一个点值）的点估计，而贝叶斯方法给出了每个预测观测的值分布。与先验预测分布一样，积分可以通过从后验预测分布中生成样本来计算。我们之前创建的相同函数 `normal_predictive_distribution()` 可以在这里使用。唯一的区别是，我们不是从先验中采样 `mu` 和 `sigma`，而是从后验中采样。

```r
N_obs <-  nrow(df_spacebar)
mu_samples <-  as_draws_df(fit_press)$b_Intercept
sigma_samples <-  as_draws_df(fit_press)$sigma
normal_predictive_distribution(mu_samples = mu_samples,
 sigma_samples = sigma_samples,
 N_obs = N_obs)
```

```r
## # A tibble: 1,444,000 × 3
##    iter trialn t_pred
##   <dbl>  <int>  <dbl>
## 1     1      1   128.
## 2     1      2   134.
## 3     1      3   167.
## # ℹ 1,443,997 more rows
```

`brms` 函数中的 `posterior_predict()` 是一个方便的函数，它提供了后验预测分布的样本。使用 `posterior_predict(fit_press)` 命令可以得到矩阵形式的预测指关节敲击时间，其中样本作为行，观测值（数据点）作为列。请注意，如果模型使用 `sample_prior = "only"` 进行拟合，则因变量将被忽略，`posterior_predict()` 将从先验预测分布中产生样本）。

后验预测分布可用于检验所考虑模型的“描述性充分性”（Gelman 等人 2014，第六章；Shiffrin 等人 2008）。通过检验后验预测分布以建立描述性充分性被称为后验预测检验。此处的目标是建立后验预测数据与观察数据大致相似。实现描述性充分性意味着当前数据可能是由该模型生成的。尽管通过描述性充分性测试并不是支持模型的强有力证据，但描述性充分性方面的重大失败可以解释为反对该模型的强有力证据（Shiffrin 等人 2008）。因此，比较不同模型的描述性充分性不足以区分它们的相对性能。

在进行模型比较时，考虑 Roberts 和 Pashler (2000) 定义的准则很重要。尽管 Roberts 和 Pashler (2000) 对过程模型更感兴趣，而不一定是贝叶斯模型，但他们的准则对任何类型的模型比较都很重要。他们的主要观点是，一个模型仅仅与数据拟合良好是不够的。应该检查模型做出的预测范围是否合理约束；如果一个模型可以捕捉任何可能的结果，那么该模型对特定数据集的拟合就不那么有信息量了。在贝叶斯建模的背景下，尽管后验预测检验很重要，但它们应被视为仅作为合理性检查来评估模型行为是否合理（关于这一点，请参阅在线章节 F）。

在许多情况下，可以简单地使用 `brms` 提供的绘图函数（这些函数作为 `bayesplot` 函数的包装器）。例如，绘图函数 `pp_check()` 以模型、预测数据集的数量和可视化类型为参数，可以显示后验预测检验的不同可视化。在这些类型的图表中，观察数据被绘制为 \(y\)，预测数据被绘制为 \(y_{rep}\)。下面，我们使用 `pp_check()` 来调查观察到的手指敲击时间分布与基于某些数量（\(11\) 和 \(100\)）的后验预测分布样本（即模拟数据集）的模型拟合情况；请参阅图 3.8 和 3.9。

```r
pp_check(fit_press, ndraws = 11, type = "hist")
```

![模型 fit_press (\(y_{rep}\)) 后验预测分布的十一样本直方图。](img/6950f5f2641ead399d16632ff5700d14.png)

图 3.8：模型 `fit_press` (\(y_{rep}\)) 后验预测分布的十一样本直方图。

```r
pp_check(fit_press, ndraws = 100, type = "dens_overlay")
```

![后验预测检查图，显示了模型 fit_press 与后验预测分布数据集的拟合情况，使用密度图叠加。](img/d76b604646bdfb12fbc2a645d03546f6.png)

图 3.9：后验预测检查图，显示了模型 `fit_press` 与后验预测分布数据集的拟合情况，使用密度图叠加。

数据略微偏斜，且没有小于 \(100\) 毫秒的值，但预测分布是中心对称的；参见图 3.8 和 3.9。这个后验预测检查显示了观察数据和预测数据之间有轻微的不匹配。我们能否构建一个更好的模型？我们将在下一节回到这个问题。

## 3.7 似然的影响

手指敲击时间（以及一般响应时间）通常不是正态分布。更现实的分布是对数正态分布。一个对数正态分布的随机变量（如时间）只取正实数值，并且是右偏的。尽管其他分布也可以产生具有这种特性的数据，但对数正态分布对于手指敲击时间和响应时间来说将是一个非常合理的分布。

### 3.7.1 对数似然

如果 \(\boldsymbol{y}\) 是对数正态分布的，这意味着 \(\log(\boldsymbol{y})\) 是正态分布的。¹⁶ 对数正态分布也是通过位置参数 \(\mu\) 和尺度参数 \(\sigma\) 定义的，但这些参数是在对数毫秒尺度上；它们对应于数据 \(\boldsymbol{y}\) 的对数 \(\log(\boldsymbol{y})\) 的均值和标准差，这将服从正态分布。因此，当我们使用对数正态似然模型来建模一些数据 \(\boldsymbol{y}\) 时，参数 \(\mu\) 和 \(\sigma\) 的尺度与数据 \(\boldsymbol{y}\) 不同。方程 (3.9) 展示了对数正态分布和正态分布之间的关系。

\[\begin{equation} \begin{aligned} \log(\boldsymbol{y}) &\sim \mathit{Normal}( \mu, \sigma)\\ \boldsymbol{y} &\sim \mathit{LogNormal}( \mu, \sigma) \end{aligned} \tag{3.9} \end{equation}\]

我们可以通过首先设置一个辅助变量 \(z\)，使得 \(z = \log(y)\)，然后使用正态分布来从对数正态分布中获取样本。这意味着 \(z \sim \mathit{Normal}(\mu, \sigma)\)。然后我们可以使用 \(exp(z)\) 作为 \(\mathit{LogNormal}(\mu, \sigma)\) 的样本，因为 \(\exp(z) =\exp(\log(y)) = y\)。下面的代码生成了图 3.10。

```r
mu <-  6
sigma <-  0.5
N <-  500000
# Generate N random samples from a log-normal distribution
sl <-  rlnorm(N, mu, sigma)
ggplot(tibble(samples = sl), aes(samples)) +
 geom_histogram(aes(y = after_stat(density)), binwidth = 50) +
 ggtitle("Log-normal distribution\n") +
 coord_cartesian(xlim = c(0, 2000))
# Generate N random samples from a normal distribution,
# and then exponentiate them
sn <-  exp(rnorm(N, mu, sigma))
ggplot(tibble(samples = sn), aes(samples)) +
 geom_histogram(aes(y = after_stat(density)), binwidth = 50) +
 ggtitle("Exponentiated samples from\na normal distribution") +
 coord_cartesian(xlim = c(0, 2000))
```

![由生成来自对数正态分布的样本或对来自正态分布的样本进行指数运算生成的具有相同参数的两个对数正态分布](img/318135851e5c540b9fc30bb9b6f93ee6.png)![由生成来自对数正态分布的样本或对来自正态分布的样本进行指数运算生成的具有相同参数的两个对数正态分布](img/081dffb48f09a996b139aad827733364.png)

图 3.10：由生成来自对数正态分布的样本或对来自正态分布的样本进行指数运算生成的具有相同参数的两个对数正态分布。

### 3.7.2 使用对数正态似然拟合单个受试者重复按按钮的数据

如果我们假设手指敲击时间是按对数正态分布的，似然函数将按以下方式变化：

\[\begin{equation} t_n \sim \mathit{LogNormal}(\mu,\sigma) \end{equation}\]

但现在先验的尺度需要改变！为了便于说明，让我们从均匀先验开始，尽管我们之前提到，这些在这里实际上并不合适。（更现实的先验将在下面讨论。）

\[\begin{equation} \begin{aligned} \mu &\sim \mathit{Uniform}(0, 11) \\ \sigma &\sim \mathit{Uniform}(0, 1) \\ \end{aligned} \tag{3.10} \end{equation}\]

因为参数的尺度与因变量不同，它们的解释方式发生变化，并且比我们处理假设正态似然（位置和尺度与对数正态分布的均值和标准差不一致）的线性模型更为复杂：

+   *位置 \(\mu\)*：在我们之前的线性模型中，\(\mu\) 代表均值（在正态分布中，均值恰好与中位数和众数相同）。但现在，均值需要通过计算 \(\exp(\mu +\sigma ^{2}/2)\) 来得出。换句话说，在对数正态分布中，均值依赖于 \(\mu\) 和 \(\sigma\)。中位数仅仅是 \(\exp(\mu)\)。请注意，\(\mu\) 的先验不是以毫秒为尺度，而是以对数毫秒为尺度。

+   *尺度 \(\sigma\)*：这是 \(\log(\boldsymbol{y})\) 的正态分布的标准差。具有 *位置* \(\mu\) 和 *尺度* \(\sigma\) 的对数正态分布的标准差将是 \(\exp(\mu +\sigma ^{2}/2)\times \sqrt{\exp(\sigma²)- 1}\)。与正态分布不同，对数正态分布的分散程度依赖于 \(\mu\) 和 \(\sigma\)。

为了理解毫秒尺度上先验的意义，需要考虑先验和似然。生成先验预测分布将有助于解释先验。这个分布可以通过仅对 `normal_predictive_distribution()` 生成的样本进行指数运算来生成（或者，也可以通过替换 `rnorm()` 为 `rlnorm()` 来编辑该函数）。

```r
N_samples <-  1000
N_obs <-  nrow(df_spacebar)
mu_samples <-  runif(N_samples, 0, 11)
sigma_samples <-  runif(N_samples, 0, 1)
prior_pred_ln <-
 normal_predictive_distribution(mu_samples = mu_samples,
 sigma_samples = sigma_samples,
 N_obs = N_obs) %>%
 mutate(t_pred = exp(t_pred))
```

接下来，绘制一些代表性统计量的分布；见图 3.11。

![具有在方程 (3.10) 中定义的先验的对数正态模型均值、中位数、最小值和最大值的前验预测分布，即 \(\mu \sim \mathit{Uniform}(0, 11)\) 和 \(\sigma \sim \mathit{Uniform}(0, 1)\)。x 轴显示从对数尺度转换回来的值。](img/50718c6c0e8d62a9daf6747ae835b69d.png)

图 3.11：具有在方程 (3.10) 中定义的先验的对数正态模型均值、中位数、最小值和最大值的前验预测分布，即 \(\mu \sim \mathit{Uniform}(0, 11)\) 和 \(\sigma \sim \mathit{Uniform}(0, 1)\)。x 轴显示从对数尺度转换回来的值。

由于 \(\exp(\)任何有限实数\() > 0\)，我们不能再生成负值。这些先验可能在模型收敛的意义上起作用；但最好为模型提供正则化先验。正则化先验的一个例子：

\[\begin{equation} \begin{aligned} \mu &\sim \mathit{Normal}(6, 1.5) \\ \sigma &\sim \mathit{Normal}_+(0, 1) \\ \end{aligned} \tag{3.11} \end{equation}\]

这里 \(\sigma\) 的先验是一个截断分布，尽管其位置是零，但这不是它的均值。我们可以使用 `extraDistr` 包中的 `rtnorm()` 函数从先验分布的大量随机样本中计算出其近似均值。在这个函数中，我们必须将参数 `a = 0` 设置为表达正态分布从左侧截断在 0 的事实。（在线部分 A.2 详细讨论了这种类型的分布）：

```r
mean(rtnorm(100000, 0, 1, a = 0))
```

```r
## [1] 0.798
```

在生成先验预测分布之前，我们可以计算出我们 95% 确信观察值的期望中位数将落在的范围。我们通过查看先验均值 \(\mu\) 的两个标准差处发生的情况来完成此操作，即 \(6 - 2\times 1.5\) 和 \(6 + 2\times 1.5\)，并对这些值进行指数化：

```r
c(lower = exp(6 -  2 *  1.5),
 higher = exp(6 +  2 *  1.5))
```

```r
##  lower higher 
##   20.1 8103.1
```

这意味着 \(\mu\) 的先验信息仍然不是很丰富（这些是中位数；由对数正态分布生成的实际值可能分布得更广）。接下来，绘制先验预测分布的一些代表性统计量的分布。`brms` 允许通过在 `brm` 函数中设置 `sample_prior = "only"` 来从先验中采样，忽略观察到的数据 `t`。

如果我们想在没有任何数据的情况下，使用 `brms` 以这种方式生成先验预测数据，我们确实需要包含一个包含相关因变量的数据框（在这种情况下是 `y`）。设置 `sample_prior = "only"` 将忽略因变量的值。

```r
fit_prior_press_ln <-
 brm(t ~  1,
 data = df_spacebar,
 family = lognormal(),
 prior = c(prior(normal(6, 1.5), class = Intercept),
 prior(normal(0, 1), class = sigma)),
 sample_prior = "only",
 control = list(adapt_delta = 0.9))
```

要避免警告，将 `adapt_delta` 参数的默认值从 \(0.8\) 增加到 \(0.9\) 来模拟数据。由于 Stan 以与从后验分布采样相同的方式从先验分布中采样，因此不应忽略警告；始终确保模型收敛。在这方面，第 3.3 节中定义的自定义函数 `normal_predictive_distribution()` 有优势，因为它将始终从先验分布中产生独立的样本，并且不会遇到任何收敛问题。这是因为它仅依赖于 R 中的 `rnorm()` 函数。

使用以下代码绘制均值的先验预测分布。在先验预测分布中，我们通常想要忽略数据；这需要将 `prefix = "ppd"` 设置在 `pp_check()` 中。

要绘制最小值和最大值的分布，只需将 `mean` 分别替换为 `min` 和 `max`。这三个统计量的分布显示在图 3.12 中。

```r
 p1 <-  pp_check(fit_prior_press_ln,
 type = "stat",
 stat = "mean",
 prefix = "ppd") +
 coord_cartesian(xlim = c(0.001, 300000)) +
 scale_x_continuous("Finger tapping times (ms)",
 trans = "log",
 breaks = c(0.001, 1, 100, 1000, 10000, 100000),
 labels = c("0.001", "1", "100", "1000", "10000",
 "100000")) +
 ggtitle("Prior predictive distribution of means")
p2 <-  pp_check(fit_prior_press_ln,
 type = "stat",
 stat = "min",
 prefix = "ppd") +
 coord_cartesian(xlim = c(0.001, 300000)) +
 scale_x_continuous("Finger tapping times (ms)",
 trans = "log",
 breaks = c(0.001, 1, 100, 1000, 10000, 100000),
 labels = c("0.001", "1", "100", "1000", "10000",
 "100000")) +
 ggtitle("Prior predictive distribution of minimum values")
p3 <-  pp_check(fit_prior_press_ln,
 type = "stat",
 stat = "max",
 prefix = "ppd") +
 coord_cartesian(xlim = c(0.001, 300000)) +
 scale_x_continuous("Finger tapping times (ms)",
 trans = "log",
 breaks = c(0.001, 1, 100, 1000, 10000, 100000),
 labels = c("0.001", "1", "10", "1000", "10000",
 "100000")) +
 ggtitle("Prior predictive distribution of maximum values")
plot_grid(p1, p2, p3, nrow = 3, ncol =1)
```

![定义在方程 (3.11) 中的对数正态模型的均值、最大值和最小值的先验预测分布。先验预测分布标记为 \(y_{pred}\)。x 轴显示从对数尺度反变换的值。](img/a571b72cfab0ec71b0f7f70982434c37.png)

图 3.12：定义在方程 (3.11) 中的对数正态模型的均值、最大值和最小值的先验预测分布。先验预测分布标记为 \(y_{pred}\)。x 轴显示从对数尺度反变换的值。

图 3.12 显示，这里使用的先验仍然相当不具信息量。与图 3.12 中显示的我们的正态先验对应的先验预测分布的尾部甚至更靠右，达到比由均匀先验生成的先验预测分布（如图 3.11 所示）更极端的值。新的先验仍然远未代表我们的先验知识。我们可以运行更多次选择先验和生成先验预测分布的迭代，直到我们得到可以生成真实数据的先验。然而，鉴于均值、最大值和最小值的分布的大部分大致位于正确的数量级顺序，这些先验将被接受。一般来说，总结统计量（例如，均值、中位数、最小值、最大值）可以用来测试先验是否在合理的范围内。这可以通过定义特定研究问题的极端数据来实现，这些数据在观察中非常不可能出现（例如，一个单词的阅读时间超过一分钟），并选择先验，使得这种极端的手指敲击时间在先验预测分布中只非常罕见地发生。

接下来，拟合模型；回想一下，与之前的例子相比，分布族和先验都发生了变化。

```r
fit_press_ln <-
 brm(t ~  1,
 data = df_spacebar,
 family = lognormal(),
 prior = c(prior(normal(6, 1.5), class = Intercept),
 prior(normal(0, 1), class = sigma)))
```

当我们查看后验的总结时，参数位于对数尺度上：

```r
fit_press_ln
```

```r
## ...
## Population-Level Effects: 
##           Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
## Intercept     5.12      0.01     5.10     5.13 1.00     4098     2796
## 
## Family Specific Parameters: 
##       Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
## sigma     0.13      0.01     0.13     0.15 1.00     2463     1999
## 
## ...
```

如果研究目标是找出按下空格键需要多长时间（以毫秒为单位），我们需要将 \(\mu\)（或模型中的“截距”）转换为毫秒。因为对数正态分布的中位数是 \(\exp(\mu)\)，以下返回的是中位数估计值（记住，对于均值，我们需要计算 \(\exp(\mu +\sigma ^{2}/2)\)）：

```r
estimate_ms <-  exp(as_draws_df(fit_press_ln)$b_Intercept)
```

要显示这些样本的均值和 95% 似然区间，请输入：

```r
c(mean = mean(estimate_ms),
 quantile(estimate_ms, probs = c(.025, .975)))
```

```r
##  mean  2.5% 97.5% 
##   167   165   169
```

接下来，检查预测数据集是否与观测数据集相似。见图 3.13；将其与早期的图 3.9 进行比较。

```r
pp_check(fit_press_ln, ndraws = 100)
```

![拟合 _press_ln 的后验预测分布。](img/1ca36e870d3ba5a0f7a25fa6fcd8f79d.png)

图 3.13：`fit_press_ln` 的后验预测分布。

关键问题是：与具有正态似然的情况相比，后验预测数据现在是否更接近观测数据？根据图 3.13，似乎是这样的，但并不容易判断。

另一种检查预测数据与观测数据相似程度的方法是查看某些汇总统计量的分布。与先验预测分布一样，检查由不同模型生成的数据集的代表性汇总统计量的分布。然而，与先验预测分布不同，此时我们有一个明确的参考，即我们的观测数据，这意味着我们可以将汇总统计量与我们的数据中的观测统计量进行比较。我们怀疑正态分布会生成过快的指关节敲击时间（因为正态分布是对称的），而对数正态分布可能比正态模型更好地捕捉长尾。基于这个假设，计算后验预测分布的最小值和最大值分布，并将它们与数据中的最小值和最大值分别进行比较。函数 `pp_check()` 通过为 `fit_press` 和 `fit_press_ln` 指定 `stat` 为 `"min"` 或 `"max"` 来实现这一点；以下是一个示例。图示见 3.14 和 3.15。

```r
pp_check(fit_press, type = "stat", stat = "min")
```

![后验预测检查中最小值的分布，使用正态和对数正态概率密度函数。数据中的最小值是 \(110\) 毫秒。](img/f10b6153131b979898dabaad91a0a780.png)![后验预测检查中最小值的分布，使用正态和对数正态概率密度函数。数据中的最小值是 \(110\) 毫秒。](img/f0447b734f05ef191a54add024b56cc2.png)

图 3.14：使用正态和对数正态概率密度函数进行后验预测检查的最小值分布。数据的最大值是 \(110\) 毫秒。![使用正态和对数正态进行后验预测检查的最大值分布。数据的最大值是 \(409\) 毫秒。](img/c14cf134aa05b971b7fae86ae1bfb405.png)![使用正态和对数正态进行后验预测检查的最大值分布。数据的最大值是 \(409\) 毫秒。](img/04012bec32a6cee12702b3a0c85a1e01.png)

图 3.15：使用正态和对数正态进行后验预测检查的最大值分布。数据的最大值是 \(409\) 毫秒。

图 3.14 显示，对数正态似然函数做得稍微好一些，因为最小值包含在对数正态分布的大部分和正态分布的尾部。图 3.15 显示，这两个模型都无法捕捉到观测数据的最大值。一种解释是，我们数据中的对数正态似然观测值可能是由尽可能快地按下的任务生成的，而长指节拍时间的观测值可能是由于注意力分散造成的。如果这个假设是正确的，这意味着按钮按压力度的分布是两个分布的混合；对这种混合过程进行建模需要更复杂的工具，我们将在第十七章节中探讨。

这完成了我们对`brms`的介绍。我们现在可以学习更多关于回归模型的知识了。

## 3.8 最重要命令列表

这里是我们在本章中学到的最重要命令的列表。

+   用于拟合模型、生成先验预测和后验预测数据的`brms`核心函数：

```r
fit_press <-
 brm(t ~  1,
 data = df_spacebar,
 family = gaussian(),
 prior = c(prior(uniform(0, 60000),
 class = Intercept,
 lb = 0,
 ub = 60000),
 prior(uniform(0, 2000),
 class = sigma,
 lb = 0,
 ub = 2000)),
 ## uncomment for prior predictive:
 ## sample_prior = "only",
 ## uncomment when dealing with divergent transitions
 ## control = list(adapt_delta = .9)
 ## default values for chains, iterations and warmup:
 chains = 4,
 iter = 2000,
 warmup = 1000)
```

+   从拟合模型中提取样本：

```r
as_draws_df(fit_press)
```

+   后验的基本绘图

```r
plot(fit_press)
```

+   绘制先验预测/后验预测数据

```r
## Posterior predictive check:
pp_check(fit_press, ndraws = 100, type = "dens_overlay")
## Plot posterior predictive distribution of statistical summaries:
pp_check(fit_press, ndraws = 100, type = "stat", stat = "mean")
## Plot prior predictive distribution of statistical summaries:
pp_check(fit_press, ndraws = 100, type = "stat", stat = "mean",
 prefix = "ppd")
```

## 3.9 总结

本章展示了如何使用正态似然函数拟合和解释贝叶斯模型。我们通过调查先验预测分布和进行敏感性分析来研究先验的影响。我们还通过检查后验预测分布（这让我们对模型的描述性充分性有一些了解）来查看后验的拟合情况。我们还展示了如何使用对数正态似然函数拟合贝叶斯模型，以及如何比较不同模型的预测准确性。

## 3.10 进一步阅读

在 Gamerman 和 Lopes (2006) 的著作中详细讨论了采样算法。Bob Carpenter 的开源书籍 *《概率与统计：基于模拟的导论》* ([`github.com/bob-carpenter/prob-stats`](https://github.com/bob-carpenter/prob-stats)) 中的采样部分以及 Lambert (2018) 和 Lynch (2007) 关于采样算法的部分也非常有帮助。贝叶斯推理的概率编程语言的演变在 Štrumbelj 等人 (2024) 的著作中讨论。Dobson 和 Barnett (2011) 覆盖了线性建模理论的入门内容；更高级的处理可以在 Montgomery, Peck, 和 Vining (2012) 以及 Seber 和 Lee (2003) 的著作中找到。广义线性模型在 McCullagh 和 Nelder (2019) 的著作中有详细的介绍。读者还可以从我们免费提供的在线线性建模讲义中受益：[`github.com/vasishth/LM`](https://github.com/vasishth/LM).

### 参考文献

Bates, Douglas M., Martin Mächler, Ben Bolker, 和 Steve Walker. 2015\. “使用 lme4 拟合线性混合效应模型。” *《统计软件杂志》* 67 (1): 1–48\. [`doi.org/10.18637/jss.v067.i01`](https://doi.org/10.18637/jss.v067.i01).

Blitzstein, Joseph K., 和 Jessica Hwang. 2014\. *《概率论导论》*. Chapman; Hall/CRC.

Bürkner, Paul-Christian. 2017\. “brms: 使用 Stan 的贝叶斯多级模型 R 包。” *《统计软件杂志》* 80 (1): 1–28\. [`doi.org/10.18637/jss.v080.i01`](https://doi.org/10.18637/jss.v080.i01).

Bürkner, Paul-Christian. 2024\. *《使用“Stan”的贝叶斯回归模型：brms》*. [`github.com/paul-buerkner/brms`](https://github.com/paul-buerkner/brms).

Carpenter, Bob, Andrew Gelman, Matthew D. Hoffman, Daniel Lee, Ben Goodrich, Michael J. Betancourt, Marcus Brubaker, Jiqiang Guo, Peter Li, 和 Allen Riddell. 2017\. “Stan: 一种概率编程语言。” *《统计软件杂志》* 76 (1).

Dobson, Annette J., 和 Adrian Barnett. 2011\. *《广义线性模型导论》*. CRC Press.

Duane, Simon, A. D. Kennedy, Brian J. Pendleton, 和 Duncan Roweth. 1987\. “混合蒙特卡洛。” *《物理快报 B》* 195 (2): 216–22\. [`doi.org/10.1016/0370-2693(87)91197-X`](https://doi.org/10.1016/0370-2693(87)91197-X).

Gamerman, Dani, 和 Hedibert F. Lopes. 2006\. *《马尔可夫链蒙特卡洛：贝叶斯推理的随机模拟》*. CRC Press.

Ge, Hong, Kai Xu, 和 Zoubin Ghahramani. 2018\. “Turing: 一种用于灵活概率推理的语言。” 在 *《机器学习研究论文集》* 中，由 Amos Storkey 和 Fernando Perez-Cruz 编辑，第 84 卷，第 1682–90 页。兰萨罗特群岛，帕莱亚·布兰卡，加那利群岛：PMLR. [`proceedings.mlr.press/v84/ge18b.html`](http://proceedings.mlr.press/v84/ge18b.html).

Gelman, Andrew, John B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari, 和 Donald B. Rubin. 2014\. *贝叶斯数据分析*. 第三版. 佛罗里达州博卡拉顿, FL: 查普曼; 哈尔/CRC 出版社.

Gelman, Andrew, Daniel P. Simpson, 和 Michael J. Betancourt. 2017\. “先验通常只能在似然性的背景下理解。” *熵* 19 (10): 555\. [`doi.org/10.3390/e19100555`](https://doi.org/10.3390/e19100555).

Goodrich, Ben, Jonah Gabry, Imad Ali, 和 Sam Brilleman. 2018\. “Rstanarm: 通过 Stan 进行贝叶斯应用回归建模。” [`mc-stan.org/`](http://mc-stan.org/).

Hoffman, Matthew D., 和 Andrew Gelman. 2014\. “无转弯采样器：在哈密顿蒙特卡洛中自适应设置路径长度。” *机器学习研究杂志* 15 (1): 1593–1623\. [`dl.acm.org/citation.cfm?id=2627435.2638586`](http://dl.acm.org/citation.cfm?id=2627435.2638586).

Hubel, Kerry A., Bruce Reed, E. William Yund, Timothy J. Herron, 和 David L. Woods. 2013\. “计算机化手指敲击测量：手优势、年龄和性别的影响。” *感知与运动技能* 116 (3): 929–52\. [`doi.org/https://doi.org/10.2466/25.29.PMS.116.3.929-952`](https://doi.org/https://doi.org/10.2466/25.29.PMS.116.3.929-952).

JASP Team. 2019\. “JASP (版本 0.11.1)[计算机软件].” [`jasp-stats.org/`](https://jasp-stats.org/).

Jaynes, Edwin T. 2003\. *概率论：科学的逻辑*. 剑桥大学出版社.

Jeffreys, Harold. 1939\. *概率论理论*. 牛津: 克拉伦登出版社.

Lambert, Ben. 2018\. *贝叶斯统计学生指南*. 英国伦敦: 萨奇出版社.

Lindgren, Finn, 和 Håvard Rue. 2015\. “使用 R-INLA 进行贝叶斯空间建模。” *统计软件杂志* 63 (1): 1–25\. [`doi.org/ 10.18637/jss.v063.i19`](https://doi.org/%2010.18637/jss.v063.i19).

Luce, R. Duncan. 1991\. *反应时间：在推断基本心理组织中的作用*. 牛津大学出版社.

Lunn, David J., Andrew Thomas, Nichola G. Best, 和 David J. Spiegelhalter. 2000\. “WinBUGS-贝叶斯建模框架：概念、结构和可扩展性。” *统计学与计算* 10 (4): 325–37.

Lynch, Scott Michael. 2007\. *应用贝叶斯统计和社会科学家估计导论*. 纽约, NY: 斯普林格.

McCullagh, Peter, 和 J. A. Nelder. 2019\. *广义线性模型*. 第二版. 佛罗里达州博卡拉顿: 查普曼; 哈尔/CRC.

Montgomery, D. C., E. A. Peck, 和 G. G. Vining. 2012\. *线性回归分析导论*. 第 5 版. 新泽西州霍博肯: 威利出版社.

Neal, Radford M. 2011\. “使用哈密顿动力学进行 MCMC。” 在 *马尔可夫链蒙特卡洛手册* 中，由 Steve Brooks, Andrew Gelman, Galin Jones, 和 Xiao-Li Meng 编辑。泰勒弗朗西斯出版社. [`doi.org/10.1201/b10905-10`](https://doi.org/10.1201/b10905-10).

O’Hagan, Anthony, 和 Jonathan J. Forster. 2004\. “肯德尔高级统计学理论，第 2B 卷：贝叶斯推断。” 威廉姆斯出版社.

Plummer, Martin. 2016. “JAGS 版本 4.2.0 用户手册。”

Plummer, Martin. 2022. “基于模拟的贝叶斯分析。” *统计科学年度评论*。

Roberts, Seth, 和 Harold Pashler. 2000. “良好的拟合有多有说服力？对理论测试的评论。” *心理学评论* 107 (2): 358–67\. [`doi.org/https://doi.org/10.1037/0033-295X.107.2.358`](https://doi.org/https://doi.org/10.1037/0033-295X.107.2.358).

Salvatier, John, Thomas V. Wiecki, 和 Christopher Fonnesbeck. 2016. “使用 PyMC3 在 Python 中进行概率编程。” *PeerJ 计算机科学* 2 (四月): e55\. [`doi.org/https://doi.org/10.7717/peerj-cs.55`](https://doi.org/https://doi.org/10.7717/peerj-cs.55).

Seber, George A. F., 和 Allen J. Lee. 2003. *线性回归分析*. 第二版. 新泽西州霍博肯：John Wiley & Sons.

Shiffrin, Richard M, Michael D Lee, Woojae Kim, 和 Eric-Jan Wagenmakers. 2008. “模型评估方法的综述：关于分层贝叶斯方法的教程。” *认知科学：多学科杂志* 32 (8): 1248–84\. [`doi.org/https://doi.org/10.1080/03640210802414826`](https://doi.org/https://doi.org/10.1080/03640210802414826).

Štrumbelj, Erik, Alexandre Bouchard-Côté, Jukka Corander, Andrew Gelman, Håvard Rue, Lawrence Murray, Henri Pesonen, Martin Plummer, 和 Aki Vehtari. 2024. “贝叶斯推理软件的过去、现在和未来。” *统计科学* 39 (1): 46–61.

Vasishth, Shravan, Zhong Chen, Qiang Li, 和 Gueilan Guo. 2013. “处理中文相对从句：支持主语-相对优势的证据。” *PLoS ONE* 8 (10): 1–14\. [`doi.org/https://doi.org/10.1371/journal.pone.0077006`](https://doi.org/https://doi.org/10.1371/journal.pone.0077006).

Vasishth, Shravan, 和 Felix Engelmann. 2022. *句子理解作为一种认知过程：一种计算方法*. 英国剑桥：剑桥大学出版社. [`books.google.de/books?id=6KZKzgEACAAJ`](https://books.google.de/books?id=6KZKzgEACAAJ).

Vehtari, Aki, Andrew Gelman, Daniel P. Simpson, Bob Carpenter, 和 Paul-Christian Bürkner. 2021. “排名归一化、折叠和定位：改进的\(\widehat{R}\)用于评估 MCMC 的收敛性。” *贝叶斯分析* 16 (2): 667–718\. [`doi.org/10.1214/20-BA1221`](https://doi.org/10.1214/20-BA1221).

* * *

1.  在`brms`包（Bürkner 2017）中，后验分布的样本被称为抽取。↩︎

1.  Python 包 PyMC3 和 Julia 库 Turing 是最近的例外，因为它们完全集成到各自的语言中。↩︎

1.  我们将一个受试者对刺激做出反应或反应所需的时间称为反应时间（反应时间和反应时间通常可以互换使用，参看 Luce 1991）。然而，在这种情况下，除了屏幕上的十字之外，没有其他刺激，受试者一看到十字就立即按下空格键。↩︎

1.  这里还有一个额外的问题。尽管截距参数被分配了一个介于 \(0\) 和 \(60000\) 毫秒之间的均匀分布，但采样器可能会从这个范围之外的初始值开始采样，从而产生警告。由于初始值是随机选择的（除非用户明确指定初始值），采样器可以从 \(0\)-\(60000\) 范围之外的初始值开始。 ↩︎

1.  我们将在第 3.7.2 节中看到如何使用 `brms` 和 `pp_check()` 生成先验预测分布，例如均值、最小值或最大值。 ↩︎

1.  尽管在理论上可以使用更宽的先验，但在实践中，这些是使用 `brms` 实现收敛的最宽先验。 ↩︎

1.  如果两个事件 A 和 B 在给定某个事件 E 的条件下是条件独立的，那么 \(P(A\cap B | E) = P(A|E) P(B|E)\)。参见 Blitzstein 和 Hwang 的第二章 (2014) 以获取示例和更多关于条件独立性的讨论。 ↩︎

1.  更确切地说，是 \(\log_e(\boldsymbol{y})\) 或 \(\ln(\boldsymbol{y})\)，但我们将它写作 \(\log(\boldsymbol{y})\)。 ↩︎
