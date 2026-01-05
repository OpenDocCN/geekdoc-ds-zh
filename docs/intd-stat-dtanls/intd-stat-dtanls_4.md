# 4 统计推断 - 案例研究：对政府的满意度

> 原文：[https://bookdown.org/conradziller/introstatistics/statistical-inference---case-study-satisfaction-with-government.html](https://bookdown.org/conradziller/introstatistics/statistical-inference---case-study-satisfaction-with-government.html)

## 4.1 简介

在西方民主国家，公民对政府的满意度与执政党在即将到来的选举中的选举成功密切相关。同时，满意的公民更愿意支持非首选政策，并且不太可能支持反体制或民粹主义政党。

例如，如果我们对德国政府满意度的程度感兴趣，我们首先需要定义我们想要对其做出陈述的总体（例如，居住在德国的18岁以上的人）。对总体中的每个人进行访谈将是徒劳的。相反，我们可以依靠“捷径”随机抽样，即从这个总体中随机选择观测单位（即随机选择的德国人）并对他们进行调查。其想法是，从随机样本中获得的结果（例如，使用调查项目计算的平均政府支持率）代表了潜在的总体的特征——也就是说，样本中估计的参数（例如，平均支持率）与（未观察到的）总体参数之间的匹配非常接近。

![联邦总理府](../Images/9d094f2d7b0391808b604a030ca9f903.png)联邦总理府

来源：[https://www.bundesregierung.de/breg-de/bundesregierung/bundeskanzleramt/dienstsitze-bundeskanzleramt-466834](https://www.bundesregierung.de/breg-de/bundesregierung/bundeskanzleramt/dienstsitze-bundeskanzleramt-466834)

在统计推断中，我们主要感兴趣的是样本估计如何准确地反映真实的总体参数。因此，目标是找到一个合适的精确度（或不确定性，作为另一面）的度量。我们通过将样本中的估计与反映概率上可能或不可能的值的测试分布相匹配来实现这一点。这一案例研究将讨论如何进行。

在此之前，我们简要介绍一下概率论领域。

## 4.2 概率

概率对应于满足一定标准（即“事件”）的观测数量与总观测数量的比率。一个经典的例子是抛硬币和正面与总抛掷次数之间的比率。概率总是正的（\(P(A)\ge0\)）。所有可能事件（例如，抛硬币时的正面和反面）的概率是1（\(P(\Omega)=1\)）。如果事件A和B不能同时发生，那么A或B发生的概率是这两个事件概率的总和（\(P(A or B)=P(A)+P(B)\)）。

在随机实验（例如，抛硬币）中发生的事件的概率可以用所谓的随机变量来表示。随机变量有一个概率分布，对于二元变量（0/1）这是伯努利分布，对于连续变量这是正态分布。（还有其他分布，如 F 分布或卡方分布，我们将在以后的应用中使用）。

这里我们看到的是随机抽取 10 个球（7 个蓝色，3 个红色）的概率分布，总共进行了 10,000 次抽取。相应地，蓝色和红色的概率分布大约为 0.7 和 0.3：

```r
possible_values <- [c](https://rdrr.io/r/base/c.html)(1,0)
Bernoulli <- [sample](https://rdrr.io/r/base/sample.html)(possible_values,
 size=10000,
 replace=TRUE,
 prob=[c](https://rdrr.io/r/base/c.html)(0.3, 0.7))
[prop.table](https://rdrr.io/r/base/proportions.html)([table](https://rdrr.io/r/base/table.html)(Bernoulli))
```

```r
## Bernoulli
##      0      1 
## 0.7034 0.2966
```

```r
h <- [hist](https://rdrr.io/r/graphics/hist.html)(Bernoulli,plot=FALSE)
h$density = h$counts/[sum](https://rdrr.io/r/base/sum.html)(h$counts)*100
[plot](https://rdrr.io/r/graphics/plot.default.html)(h,freq=FALSE, axes=FALSE)
[axis](https://rdrr.io/r/graphics/axis.html)(1, at = [c](https://rdrr.io/r/base/c.html)(0, 1), labels = [c](https://rdrr.io/r/base/c.html)("Blue", "Red"))
[axis](https://rdrr.io/r/graphics/axis.html)(2, at = [c](https://rdrr.io/r/base/c.html)(0, 10, 20, 30, 40, 50, 60, 70))
```

![图片](../Images/503726bbe43a1bb8173b7e34e6f07530.png)

伯努利分布以事件发生的概率 \(p\) 为特征。对立事件的概率自动由此得出：\(1-p\)。**我们在计算比例的测试统计量时需要这个分布**。

随机变量的正态分布是一个钟形概率密度函数。它给出了**连续随机变量**具有无限中间值的概率。将时间视为一个连续的概念。理论上，我们可以用无限小的单位（毫秒、纳秒等）来衡量时间。然而，在应用研究中，我们会将变量时间赋予离散值。因此，连续变量更是一个理论概念。关于连续随机变量的分布：正态分布以均值 \(\mu\) 和标准差 \(\sigma\) 为特征。随机变量的正态分布的表示法是 \(X \sim N(\mu,\sigma)\)。\(\mu\) 和 \(\sigma\) 的每一种组合都会产生不同形状的正态分布。均值表示分布的中心在 x 轴上的位置，而 \(\sigma\) 表示分布的平坦或陡峭程度（值越高，分布越分散，曲线越平坦）。

所说的**标准正态分布**具有均值为 0 和标准差为 1 (\(X \sim N(0,1)\))。与其他所有正态分布一样，（1）曲线的末端向左右两侧扩散，接近 x 轴但永远不会触及它。因此，值可以从负无穷大到正无穷大。 （2）曲线下的总面积等于 1。（3）所谓的“经验法则”指出，对于标准正态分布，68% 的观测值位于 -1 和 +1 个标准差之间。95% 的观测值位于大约 -2 和 +2 个标准差之间（精确值为 -1.96 和 +1.96），而 99.7% 的观测值位于大约 -3 和 +3 个标准差之间。

```r
# Draw a standard normal distribution:
z = [seq](https://rdrr.io/r/base/seq.html)(-4, 4, length.out=1001)
x = [rnorm](https://rdrr.io/r/stats/Normal.html)(z)
[plot](https://rdrr.io/r/graphics/plot.default.html)( x=z, y=[dnorm](https://rdrr.io/r/stats/Normal.html)(z), bty='n', type='l', main="Standard normal distribution", ylab="Probability density", xlab="z", xlim=[c](https://rdrr.io/r/base/c.html)(-3,3))
[axis](https://rdrr.io/r/graphics/axis.html)(1, at = [seq](https://rdrr.io/r/base/seq.html)(-4, 4, by = 1))
 # annotate the density function with the 5% probability mass tails
[polygon](https://rdrr.io/r/graphics/polygon.html)(x=[c](https://rdrr.io/r/base/c.html)(z[z<=[qnorm](https://rdrr.io/r/stats/Normal.html)(0.025)], [qnorm](https://rdrr.io/r/stats/Normal.html)(0.025), [min](https://rdrr.io/r/base/Extremes.html)(z)), y=[c](https://rdrr.io/r/base/c.html)([dnorm](https://rdrr.io/r/stats/Normal.html)(z[z<=[qnorm](https://rdrr.io/r/stats/Normal.html)(0.025)]), 0, 0), col=[grey](https://rdrr.io/r/grDevices/gray.html)(0.8))
[polygon](https://rdrr.io/r/graphics/polygon.html)(x=[c](https://rdrr.io/r/base/c.html)(z[z>=[qnorm](https://rdrr.io/r/stats/Normal.html)(0.975)], [max](https://rdrr.io/r/base/Extremes.html)(z), [qnorm](https://rdrr.io/r/stats/Normal.html)(0.975)), y=[c](https://rdrr.io/r/base/c.html)([dnorm](https://rdrr.io/r/stats/Normal.html)(z[z>=[qnorm](https://rdrr.io/r/stats/Normal.html)(0.975)]), 0, 0), col=[grey](https://rdrr.io/r/grDevices/gray.html)(0.8))
```

![图片](../Images/7e4896d3aa2bb86af5da0464107f5bff.png)

对于标准正态分布等概率分布，我们不是计算特定值的概率，而是计算值范围的概率。例如，一个值大于1.96个标准差的概率，即\(P(Z\ge1.96)\)，是2.5%。并且一个值小于-1.96个标准差的概率，即\(P(Z\le-1.96)\)，也是2.5%（见上图中的灰色区域，两个区域的总和为5%）。

**我们将使用标准正态分布（也称为Z分布）进行各种统计测试。规则是95%的可观察值位于-1.96和1.96之间，5%在边缘将也很重要在零假设检验中。**

## 4.3 统计推断基础

推断意味着我们使用已知的事实来了解未知的事实（尚未知道）。统计推断意味着我们从（随机）样本中获取信息，并从中推断出（尚未知道）的总体属性（样本是从中抽取的）。基本上，只有随机样本适合这个目的，因为它们以最佳可能的方式（加减一些随机变化）代表潜在总体在所有属性上。

虽然随机样本是进行统计推断的最佳方式，但这个过程并不完美。由于随机偏差，我们样本中的单位可能并不完美地匹配总体的所有属性。可能的情况是，抽取的样本中女性比例异常高。这些随机偏差被称为**抽样变异性**。以下章节将说明抽样变异性随着样本中观察值的数量减少（“数据越多越好”）以及如果我们反复进行随机抽样过程（观察值数量足够大），我们会得到一个类似于正态分布的样本估计分布。这在我们进行统计测试时将非常有用，在统计测试中，我们使用正态分布，就像它是我们感兴趣的数量（可以匹配和测试我们的估计）的抽样分布一样。

### 4.3.1 大数定律

通常，案例数量少会导致抽样变异性更高。你可能从抛硬币实验中熟悉这一点。硬币经常连续抛掷。抛掷10次后，抛出7次正面和3次反面的情况是完全可能的。尽管如此，潜在的几率始终是0.5/0.5（或50%/50%），在所有抛硬币的总体中，正面和反面应该同样频繁出现。如果我们现在继续抛掷硬币，抛掷30次、300次、3000次等，观察到的抛硬币将越来越符合0.5/0.5的概率分布。

大数定律表明，随着样本量的增加，样本均值将越来越接近总体均值。（最终，样本将与整个总体相同。）这意味着更多的数据更好，我们的样本越大，我们越有信心它匹配或至少近似总体参数。

![大数定律：抛硬币](../Images/e067d3d1c6e50c1aaeedc4b319f705e7.png)大数定律：抛硬币

来源：[https://link.springer.com/chapter/10.1007/978-3-030-45553-8_5](https://link.springer.com/chapter/10.1007/978-3-030-45553-8_5)

### 4.3.2 中心极限定理

如果我们从总体中抽取许多（甚至无限）样本，并且对例如政府满意度均值感兴趣，那么将每个样本的均值绘制成直方图将导致一个分布，随着样本的增加而越来越接近正态分布。因此，样本均值的平均值对应于我们本质上感兴趣的总体参数（即居住在德国的所有18岁以上成年人对政府的满意度）。

中心极限定理的启示在以下图中展示。100,000个模拟样本的平均值为4.2，对应于政府满意度的“真实”总体均值。请注意，在现实中，我们没有这样的信息，并且我们无法抽取这么多样本，因为这会花费太高。

```r
[set.seed](https://rdrr.io/r/base/Random.html)(123)
data <- [rnorm](https://rdrr.io/r/stats/Normal.html)(100000, 4.2, 1)
[hist](https://rdrr.io/r/graphics/hist.html)(data, freq = FALSE, col = "gray", xlab = "Data Values", main = "Means of government satisfaction")
[curve](https://rdrr.io/r/graphics/curve.html)([dnorm](https://rdrr.io/r/stats/Normal.html)(x, mean = [mean](https://rdrr.io/r/base/mean.html)(data), sd = [sd](https://rdrr.io/r/stats/sd.html)(data)), col = "black", lwd = 2, add = TRUE)
```

![](../Images/78fad6f32fbeef73ffde823b7f7cf4d0.png)

中心极限定理的另一个特性是，每个个体样本的案例数（\(n\)）越多，样本估计值的分布就越接近正态分布。这对于 \(n>30\) 的工作非常好。此外，被检验的特征在总体中的分布（例如，右偏或左偏）并不重要。在任何情况下，随着样本量的增加（且 \(n>30\)），样本估计值的分布将（随着样本的增加）越来越接近正态分布。

### 4.3.3 为什么统计定律很重要

#### 4.3.3.1 只有一个样本

+   我们基于**一个样本**计算一切，其他的一切都会花费太高。

+   我们永远不能完全确信这个特定的样本以及用它估计的统计数据与总体统计数据相对应，但：

    +   **样本越大**，样本估计值的分布越陡峭，我们得到一个接近总体参数的估计参数的可能性就越大，即使只有一个样本（大数定律）。

    +   对于**30个或更多观测值的样本**，我们知道无论总体分布如何，样本估计值的分布都是正态的；这为我们提供了这样的理由：大多数样本都围绕着总体均值混合，而且有一个相当高的概率，即我们单个样本的估计值**接近分布的中心（即，接近总体参数）**；得到远离分布中心的估计值是不太可能的（中心极限定理）。

#### 4.3.3.2 零假设检验入门

+   中心极限定理也为我们量化不确定性提供了理论依据，因为如果重复的样本估计可以表示为**正态分布**，那么我们可以使用正态分布作为参考框架，以判断我们单个样本的结果的可信度。

+   为了做到这一点，我们对我们要测试的样本估计值所对应的分布做出具体假设（例如，一个假设两组之间均值没有差异的分布）。

+   这种程序称为零假设检验（或，NHT）：

1.  我们提出一个研究假设或所谓的**备择假设（例如，女性对政府的满意度高于男性）**。

1.  我们使用随机样本并估计男性和女性对政府满意度的均值差异。

1.  我们将我们的结果与一个（理论上的）分布进行比较，在这个分布下我们假设**零假设（-> 男女之间在政府满意度上没有差异）**是正确的（即，测试分布）。

1.  如果我们从样本中得到的估计值位于假设零假设的分布中心附近，这不会为备择假设提供太多证据；然而：如果估计值位于该分布的边缘，我们就会得出结论，有证据支持备择假设，该假设指出**在总体中，女性对政府的满意度高于男性**。

+   这种检验程序的前提是我们将样本估计值标准化，使其与测试分布可比（例如，这可以通过以下方式实现：\(z = \frac{estimate}{{standard error}}\))。

#### 4.3.3.3 标准误差

+   为了确定围绕总体参数的变异，我们可以使用抽样分布的标准差。

+   然而，由于在应用案例中我们并没有很多样本，只有一个，我们需要从这个样本中估计这种变异。

+   这种估计值的估计变异称为标准误差，其计算公式取决于估计量（通常我们使用涉及**观测数**的度量来标准化方差或标准差的度量；例如，均值的标准误差公式如下：\(\bar x = \frac{s}{\sqrt{n}}\))。

## 4.4 置信区间

### 4.4.1 基本思想

理论上，点估计（例如，均值或相关系数）的置信区间（CIs）显示了一组包含真实总体参数的概率范围。因此，它是估计不确定度（或相反，精度）的度量。通常有三种类型的置信区间：90%、95% 和 99% 的置信区间。这三种类型反映了在大量样本中，真实总体值位于区间内的概率水平。例如，如果一个人对总体均值感兴趣并进行了重复（“无限”）抽样，那么 95% 的相应样本的置信区间会包含真实总体均值，而 5% 不会包含它。

这可以通过以下图表来说明。

![样本均值的置信区间](../Images/f3474453e6d0423693806d67d7090428.png)样本均值的置信区间

来源：[https://seeing-theory.brown.edu/frequentist-inference/index.html](https://seeing-theory.brown.edu/frequentist-inference/index.html)

在图表中，从具有五个观测值的总体中反复抽取样本（橙色点）。每次，都计算平均值（或估计值，因为我们使用的是样本）。每个样本的平均值的点估计由绿色或红色点表示，置信区间是它左右相应的线。对于绿色估计值，总体均值（虚线）在置信区间内，对于红色估计值则不在。

![样本相关性的置信区间](../Images/e8646d9f0396c286662592c32225e9da.png)样本相关性的置信区间

来源：[https://github.com/simonheb/metrics1-public](https://github.com/simonheb/metrics1-public)

在这里，我们展示的是双变量回归的结果（类似于相关性），而不是平均值。在总体中，x 和 y 的双变量关系由黑色斜率线（回归系数 \(\beta_1=1.2\)）表示。灰色点代表总体中的单位。从这个总体中抽取了四个随机样本，样本中的观测值由红色点表示。相关性的点估计由红色斜率线表示。置信区间是围绕估计值的浅红色区域。在样本 1、2 和 4 中，置信区间包含了总体参数；只有样本 3（左下角）的部分置信区间不再包含总体参数。类似于上述关于平均值的思维实验，从重复样本中得到的 95% 的估计值会包含总体均值，而 5% 不会。

### 4.4.2 如何计算和解释置信区间

置信区间的概念指的是抽样分布，即从总体中重复多次（多达无限多次）的随机样本。对于95%置信区间，解释将是：对于重复样本，95%的样本将包含总体参数，5%的样本将不会。

在实践中，我们只有一个样本，因此只能计算一次置信区间。总体参数实际上是否落在我们的一个样本的置信区间值范围内，不能得出结论。可以说的是，这种情况的可能性很高。例如，对于95%置信区间，95个样本中的100个将包含总体参数，我们的一个样本很可能是这95个之一，而不是那5个。（同时，一些教科书中的解释，如“有95%的概率总体参数在区间内”，从严格意义上讲是不正确的）。

因此，我们可以将置信区间主要解释为我们感兴趣的不知名总体参数的可能或合理的值范围（考虑到我们样本的特征）。因此，置信区间在给出点估计的同时，也给出了如果我们更频繁地重复类似我们的研究，可以预期的变化范围。这确实是重要信息，并且对于许多研究人员来说，它也比零假设检验的显著性值更直观。

此外，置信区间与统计显著性之间存在关系，这是一个与零假设检验相关的概念。我们将在本案例研究的最后探讨这种关系。简而言之，置信区间显示了总体参数可能值的范围，并且同时提供了关于统计显著性的信息。

计算置信区间时，我们通常使用以下公式：

\(point estimate \pm z_\alpha \times standard error\)。

\(z_\alpha\) 指的是临界Z值（即对应于一组错误概率的给定z分布值，这将在下文进行更详细的解释）。

#### 4.4.2.1 均值

对于均值的95%置信区间的上下限，我们使用：\(95\% CIs=[\bar x - 1.96 \times {\frac{s}{\sqrt n}}, \bar x + 1.96 \times {\frac{s}{\sqrt n}}]\)。

让我们用欧洲社会调查（德国样本）的实时数据来探讨这个问题。首先，我们读取数据并了解变量对政府满意度 `stfgov` 的分布情况。`stfgov` 使用一个从0“极其不满意”到10“极其满意”的11点量表进行调查。

```r
data_ess <- read_dta("data/ESS9_DE.dta", encoding = "latin1") 
 [hist](https://rdrr.io/r/graphics/hist.html)(data_ess$stfgov, breaks = "FD")
```

![图片](../Images/8f622a96d1c89b332e19274bde795f99.png)

```r
[summary](https://rdrr.io/r/base/summary.html)(data_ess$stfgov)
```

```r
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
##    0.00    3.00    4.00    4.28    6.00   10.00      66
```

接下来，我们分几个步骤计算95%置信区间：

```r
# We store the mean, the standard deviation and the number of observations as objects, because this way we can refer to them later on
n <- 2292
xbar <- [mean](https://rdrr.io/r/base/mean.html)(data_ess$stfgov, na.rm = TRUE)
s <- [sd](https://rdrr.io/r/stats/sd.html)(data_ess$stfgov, na.rm = TRUE)
 # Set confidence level with 1-alpha (alpha = our willingness to be wrong in repeated samples)
conf.level <- 0.95
 # Calculating the critical z-value for a two-sided test 
z <- [qnorm](https://rdrr.io/r/stats/Normal.html)((1 + conf.level) / 2)
 # Calculate the confidence interval
lower.ci <- xbar - z * s / [sqrt](https://rdrr.io/r/base/MathFun.html)(n)
upper.ci <- xbar + z * s / [sqrt](https://rdrr.io/r/base/MathFun.html)(n)
 # Print confidence intervals
[cat](https://rdrr.io/r/base/cat.html)("The", conf.level*100,"% confidence interval for the population mean is (",[round](https://rdrr.io/r/base/Round.html)(lower.ci, 2), ",", [round](https://rdrr.io/r/base/Round.html)(upper.ci, 2),").\n")
```

```r
## The 95 % confidence interval for the population mean is ( 4.19 , 4.37 ).
```

使用“t.test”命令计算置信区间甚至更简单。虽然我们感兴趣的是z分布，但当观测数足够多时（通常已经>30），t分布和z分布是对应的。

让我们试一试：

```r
[t.test](https://rdrr.io/r/stats/t.test.html)(data_ess$stfgov, conf.level = 0.95)
```

```r
## 
##  One Sample t-test
## 
## data:  data_ess$stfgov
## t = 92.783, df = 2291, p-value < 2.2e-16
## alternative hypothesis: true mean is not equal to 0
## 95 percent confidence interval:
##  4.189216 4.370120
## sample estimates:
## mean of x 
##  4.279668
```

* * *

**问题：解释置信区间。**

您的答案：

解答：

> ***解释：**总体均值的可能值在4.2和4.4之间。**

* * *

*也有可能将置信区间以图表的形式显示：

```r
df <- [data.frame](https://rdrr.io/r/base/data.frame.html)(xbar, lower.ci, upper.ci)
 ggplot(df, aes(x = 1, y = xbar)) +
 theme(axis.text.y = element_blank(),
 axis.ticks.x = element_blank()) +
 geom_point(size = 3, shape = 21, fill = "white", colour = "black") +
 geom_errorbar(aes(ymin = lower.ci, ymax = upper.ci), width = 0.2) +
 coord_flip() +
 labs(x = "Value", y = "Mean with 95% CI") +
 scale_x_continuous(breaks = [seq](https://rdrr.io/r/base/seq.html)(0, 10, by = 1), limits = [c](https://rdrr.io/r/base/c.html)(1, 1))
```

![](../Images/9521033d831eee314a1722faaaf52c8e.png)* *#### 4.4.2.2 比例

除了均值之外，我们还可以计算样本比例值的置信区间。公式略有不同，因为标准误差的计算不同。它是方差的平方根除以观测数。方差是通过将事件发生的概率\(p\)乘以对立概率\(1-p\)得到的：

\(95\% CIs=[p - 1.96 \times \sqrt{\frac{p\times(1-p)}{n}}, p + 1.96 \times \sqrt{\frac{p\times(1-p)}{n}}]\).

让我们通过使用政府满意度变量的二分版本，初始为11点评分尺度来尝试这个例子。在这个例子中，5或以下的值被分配给0（“很少或不满意”），而高于5的值被分配给1（“满意”）。现在我们可以关注那些对政府满意或相当满意（变量值=1）的受访者的比例。请注意，在0/1编码中，二元变量的平均值对应于变量值为1的观测值的比例。

```r
# Recode of variable
data_ess <- data_ess %>% 
 mutate(stfgov_di =
 case_when(stfgov <= 5 ~ 0,
 stfgov > 5 ~ 1))
 # Proportion via table command
[prop.table](https://rdrr.io/r/base/proportions.html)([table](https://rdrr.io/r/base/table.html)(data_ess$stfgov_di))
```

```r
## 
##        0        1 
## 0.693281 0.306719
```

```r
# Correspondence with the mean of the summary command
[summary](https://rdrr.io/r/base/summary.html)(data_ess$stfgov_di)
```

```r
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
##  0.0000  0.0000  0.0000  0.3067  1.0000  1.0000      66
```

```r
# Set confidence level with 1-alpha
conf.level <- 0.95
 # Calculating the critical z-value for a two-sided test
z <- [qnorm](https://rdrr.io/r/stats/Normal.html)((1 + conf.level) / 2)
 # Calculation of the confidence interval
lower.ci <- 0.3067 - z * [sqrt](https://rdrr.io/r/base/MathFun.html)((0.3067*(1-0.3067))/n)
upper.ci <- 0.3067 + z * [sqrt](https://rdrr.io/r/base/MathFun.html)((0.3067*(1-0.3067))/n)
 # Print the confidence intervals
[cat](https://rdrr.io/r/base/cat.html)("The", conf.level*100, "% confidence interval for the proportion value is (", [round](https://rdrr.io/r/base/Round.html)(lower.ci, 2), ",", [round](https://rdrr.io/r/base/Round.html)(upper.ci, 2), ").\n")
```

```r
## The 95 % confidence interval for the proportion value is ( 0.29 , 0.33 ).
```

```r
# Test using t.test
[t.test](https://rdrr.io/r/stats/t.test.html)(data_ess$stfgov_di, conf.level = 0.95)
```

```r
## 
##  One Sample t-test
## 
## data:  data_ess$stfgov_di
## t = 31.837, df = 2291, p-value < 2.2e-16
## alternative hypothesis: true mean is not equal to 0
## 95 percent confidence interval:
##  0.2878265 0.3256115
## sample estimates:
## mean of x 
##  0.306719
```

* * *

**问题：解释置信区间。**

您的答案：

解答：

> ***解释：**总体比例的可能值在0.29和0.33之间。**

* * *

#### 4.4.2.3 相关系数

接下来，我们来看如何计算相关性的置信区间。为此，我们使用以下公式：\(95\% CIs=r\pm 1.96 \times SE\)，其中标准误差SE的计算方法如下：\(SE_r=\sqrt {\frac{(1-r^2)}{n-2}}\)。为了说明这一点，我们让软件计算家庭收入与对政府满意度之间的相关性的置信区间。`hinctnta`衡量受访者的净家庭收入（扣除后）在10个分位数（“十分位”1-10）之间 - 这样测量的原因在于，这样收入就会调整到国家的收入分布，因此可以在各国之间进行比较。关于相关性，我们假设 - 与经济投票理论一致 - 收入较高的人比收入较低的人对政府的满意度更高。

```r
cor <- [cor.test](https://rdrr.io/r/stats/cor.test.html)(data_ess$hinctnta, data_ess$stfgov)
cor
```

```r
## 
##  Pearson's product-moment correlation
## 
## data:  data_ess$hinctnta and data_ess$stfgov
## t = 1.805, df = 2047, p-value = 0.07122
## alternative hypothesis: true correlation is not equal to 0
## 95 percent confidence interval:
##  -0.003445968  0.083023781
## sample estimates:
##        cor 
## 0.03986354
```

* * *

**问题：解释置信区间。**

您的答案：

解答：

> ***解释：**在总体中，相关性的可能值在-0.003和0.08的范围内。**

* * *

相关系数的置信区间的图形表示是二维的。置信区间的宽度可以沿着x轴的值变化。上一个测试的值可以说是平均置信区间。

```r
ciplot <- ggplot(data=data_ess, aes(x = hinctnta , y = stfgov)) + 
 geom_smooth(method = lm, se = TRUE, level = 0.95) +
 xlab("Income (Deciles)") +
 ylab("Satisfaction with the government")
ciplot
```

![](../Images/d86f767f96c71877b2d04915a8ab5e50.png)

* * *

**问题：** 所示的图显示了95%的置信区间。90%的置信区间会是更窄还是更宽？

您的回答：

解答：

90%的置信区间会更窄。回到概率世界的解释：如果我们只想在90%的时间内捕获真实的总体参数（而不是95%），这意味着我们可以缩小参数将落在其中的值域范围。相比之下，如果我们降低犯错的准备程度到1%（即99%的置信区间），满足这个假设的值域范围将会增加——置信区间会变宽。

* * **  *4.5 假设检验

### 4.5.1 基本思想和程序

假设检验使我们能够测试样本估计值对应另一个值的可能性有多大，更具体地说，是另一个值的抽样分布。我们通常使用零假设下的分布来进行这项测试——也就是说，例如，假设两个组之间的均值差异或两个变量之间的相关性在总体中为零。在零假设为真的假设下，检验统计量的分布有时被称为“零分布”。我们知道，根据中心极限定理，在重复抽样和足够大的样本量（>30）的情况下，这样的抽样分布将是正态分布，这意味着我们可以使用正态分布进行我们的统计测试。

通常的目标是测试我们获得的样本估计值与零分布的关系。如果估计值非常“极端”，以至于在零假设为真的情况下，获得这样的结果变得非常不可能，那么这将为零假设提供反对证据，并支持我们的备择假设（例如，两个变量在总体中的相关性为正）。因此，零假设检验的程序是一种**反证法**。我们假设我们想要证明的相反情况，然后尽可能收集反对该假设的证据。

首先，我们首先构建所谓的 **零假设 \(H_0\)**（例如，总体中没有任何或零相关性）。接下来，我们构建一个 **备择假设 \(H_A\)**（例如，总体中的相关性是正的或负的），然后我们希望表明我们找到的相关性足够强，以至于 \(H_0\) 为真的可能性变得非常低。这意味着我们试图反驳 \(H_0\)，这反过来又是对 \(H_A\) 的证据。请注意，我们不为 \(H_A\) 为真分配概率，而是为在 \(H_0\) 为真的情况下观察到的给定估计分配概率（即 \(P\)（相关性 | \(H_0\)））。

以下步骤说明了如何进行零假设检验：

+   **步骤 1**：构建零假设和备择假设。

+   **步骤 2**：确定错误概率（\(\alpha\)，即我们在重复抽样中犯错的意愿，通常为5%或更少）。

+   **步骤 3**：计算测试统计量的值（=估计值/标准误差）。

+   **步骤 4a**：确定一个临界值 c（例如，来自在线表）。

+   **步骤 5a**：如果测试统计量的值落在拒绝域内（即超过临界值），则拒绝 \(H_0\)。

或者：

+   **步骤 4b**：计算 p 值（使用统计软件）。

+   **步骤 5b**：如果p值小于错误概率，则拒绝 \(H_0\)。

+   **步骤 6**：用实质性术语解释假设检验的结果。

### 4.5.2 均值差异的假设检验

均值差异通常指的是比较两组之间变量的均值。为了说明这一点，我们再次使用收入和政府满意度作为例子。我们将变量政府满意度 `stfgov` 保持在其原始的度量形式，以便我们可以计算均值。然而，我们将变量收入 `hinctnta` 二分化为两组：低收入和高收入。

```r
data_ess <- data_ess %>% 
 mutate(hinctnta_di =
 case_when(hinctnta <= 6 ~ 0,
 hinctnta > 6 ~ 1))
[summary](https://rdrr.io/r/base/summary.html)(data_ess$hinctnta_di)
```

```r
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
##  0.0000  0.0000  0.0000  0.4794  1.0000  1.0000     270
```

* * *

**问题**：如果我们假设高收入个体对政府的满意度更高，则构建零假设和备择假设。

你的答案：

```r
H_0: H_A:
```

解答：

通常，从构建备择假设开始更容易。

\(H_A\): 高收入个体对政府的满意度高于低收入个体（即收入与对政府满意度之间存在正相关关系）。

正式：\(\mu(satisfaction|highincome)>\mu(satisfaction|lowincome)\) 或（等价）\(\mu(satisfaction|highincome)-\mu(satisfaction|lowincome)>0\)

\(H_0\): 高收入个体对政府的满意度与低收入个体相等或更低（即收入与对政府满意度之间存在零或负相关关系）。

正式：\(\mu(satisfaction|highincome)\le\mu(satisfaction|lowincome)\) 或（等价）\(\mu(satisfaction|highincome)-\mu(satisfaction|lowincome)\le0\)

注意，所有可能的结果都必须由两个假设都涵盖。这意味着，在制定有方向假设时，零假设必须包括没有关系的情况（即，“\(\le\)”而不是仅仅“\(<\)”）。

* * *

让我们假设一个错误概率为5%或0.05（步骤2）。接下来，我们计算检验统计量（步骤3）。通常，这表示为\(\frac{估计值 - 真实值}{标准误差}\)。由于我们是对零假设进行检验，真实值为0。所谓的z检验统计量的公式如下：

\(z=\frac{估计值}{标准误差}\)。

步骤4和5是关于实际检验的。记住，根据备择假设，我们假设了正的均值差异（即，高收入者比低收入者对政府的满意度更高）。接下来，我们想知道均值差异在零分布中的位置。如果它接近分布的中心，我们没有理由拒绝零假设。然而，如果它在分布的边缘，这将提供拒绝零假设（并支持备择假设）的理由。我们使用均值为0和标准差为1的z分布。记住上面提到的这个分布的性质。如果我们把临界值设定在-1.96以下和+1.96以上，如果我们的检验统计量超过这些临界值，我们就会处于分布的5%范围内。如果这种情况发生，我们可以得出结论，即使\(H_0\)成立，也很难抽取这样的值。注意，如果我们制定无方向假设（即，\(H_A\)：均值差异可以是高于或低于零，\(H_0\)：均值差异等于零）并进行双向检验，我们就会设置一个正负拒绝区域。对于有方向假设（和单侧检验），例如在我们的例子中，拒绝区域只位于分布的负或正范围（取决于\(H_A\)期望的是正还是负统计量）。

```r
# Draw a standard normal distribution:
z = [seq](https://rdrr.io/r/base/seq.html)(-4, 4, length.out=1001)
x = [rnorm](https://rdrr.io/r/stats/Normal.html)(z)
 # Null distribution of two-sided test
[plot](https://rdrr.io/r/graphics/plot.default.html)( x=z, y=[dnorm](https://rdrr.io/r/stats/Normal.html)(z), bty='n', type='l', main="Null distribution of two-sided test, error probability 0.05", ylab="Probability density", xlab="z", xlim=[c](https://rdrr.io/r/base/c.html)(-3,3))
[axis](https://rdrr.io/r/graphics/axis.html)(1, at = [seq](https://rdrr.io/r/base/seq.html)(-4, 4, by = 1))
[polygon](https://rdrr.io/r/graphics/polygon.html)(x=[c](https://rdrr.io/r/base/c.html)(z[z<=[qnorm](https://rdrr.io/r/stats/Normal.html)(0.025)], [qnorm](https://rdrr.io/r/stats/Normal.html)(0.025), [min](https://rdrr.io/r/base/Extremes.html)(z)), y=[c](https://rdrr.io/r/base/c.html)([dnorm](https://rdrr.io/r/stats/Normal.html)(z[z<=[qnorm](https://rdrr.io/r/stats/Normal.html)(0.025)]), 0, 0), col="maroon")
[polygon](https://rdrr.io/r/graphics/polygon.html)(x=[c](https://rdrr.io/r/base/c.html)(z[z>=[qnorm](https://rdrr.io/r/stats/Normal.html)(0.975)], [max](https://rdrr.io/r/base/Extremes.html)(z), [qnorm](https://rdrr.io/r/stats/Normal.html)(0.975)), y=[c](https://rdrr.io/r/base/c.html)([dnorm](https://rdrr.io/r/stats/Normal.html)(z[z>=[qnorm](https://rdrr.io/r/stats/Normal.html)(0.975)]), 0, 0), col="maroon")
```

![](../Images/adbfd17d562ccc396d1fe0c9bea7d598.png)

```r
# Null distribution of one-sided test (H_A>0)
[plot](https://rdrr.io/r/graphics/plot.default.html)( x=z, y=[dnorm](https://rdrr.io/r/stats/Normal.html)(z), bty='n', type='l', main="Null distribution of one-sided test (H_A>0), error probability 0.05", ylab="Probability density", xlab="z", xlim=[c](https://rdrr.io/r/base/c.html)(-3,3))
[axis](https://rdrr.io/r/graphics/axis.html)(1, at = [seq](https://rdrr.io/r/base/seq.html)(-4, 4, by = 1))
[polygon](https://rdrr.io/r/graphics/polygon.html)(x=[c](https://rdrr.io/r/base/c.html)(z[z>=[qnorm](https://rdrr.io/r/stats/Normal.html)(0.95)], [max](https://rdrr.io/r/base/Extremes.html)(z), [qnorm](https://rdrr.io/r/stats/Normal.html)(0.95)), y=[c](https://rdrr.io/r/base/c.html)([dnorm](https://rdrr.io/r/stats/Normal.html)(z[z>=[qnorm](https://rdrr.io/r/stats/Normal.html)(0.95)]), 0, 0), col="maroon")
```

![](../Images/03648c89dbdb157687b7943bf457c54f.png)

```r
# Null distribution of one-sided test (H_A<0)
[plot](https://rdrr.io/r/graphics/plot.default.html)( x=z, y=[dnorm](https://rdrr.io/r/stats/Normal.html)(z), bty='n', type='l', main="Null distribution of one-sided test (H_A<0), error probability 0.05", ylab="Probability density", xlab="z", xlim=[c](https://rdrr.io/r/base/c.html)(-3,3))
[axis](https://rdrr.io/r/graphics/axis.html)(1, at = [seq](https://rdrr.io/r/base/seq.html)(-4, 4, by = 1))
[polygon](https://rdrr.io/r/graphics/polygon.html)(x=[c](https://rdrr.io/r/base/c.html)(z[z<=[qnorm](https://rdrr.io/r/stats/Normal.html)(0.05)], [qnorm](https://rdrr.io/r/stats/Normal.html)(0.05), [min](https://rdrr.io/r/base/Extremes.html)(z)), y=[c](https://rdrr.io/r/base/c.html)([dnorm](https://rdrr.io/r/stats/Normal.html)(z[z<=[qnorm](https://rdrr.io/r/stats/Normal.html)(0.05)]), 0, 0), col="maroon")
```

![](../Images/faa33b90ef9f0d91a5d9bbba48e44b09.png)

* * *

**问题：**如果我们进行单侧检验，拒绝区域会如何变化？如果我们设定1%的错误概率（双向检验），会发生什么变化？

你的答案：

**解答：**

假设存在正的均值差异或相关性，拒绝区域将只位于分布的右侧。由于我们仍然使用5%的错误概率，这个区域将比双向情况更大。因此，临界值c因此稍微向分布中心移动。对于z分布，这个值对于正的备择假设是+1.645（对于负假设是-1.645）。

如果我们将错误概率改为1%并执行双尾测试，则与5%的情况相比，拒绝范围更小。我们的估计值和相应的z统计量需要更大，才能与5%的情况相比拒绝零假设。在1%的情况下，临界值是-2.326和+2.326。

可以从这里获得可以拒绝零假设的临界值：[https://www.criticalvaluecalculator.com/](https://www.criticalvaluecalculator.com/)

* * *

#### 4.5.2.1 关于p值的一般说明

P值**表明**在总体中零假设为真的情况下，找到与我们从样本中得到的估计值（例如，均值差异）一样极端（或更极端）的估计值的概率，即使零假设在总体中为真。在正式术语中，这是给定零假设的估计值的概率\(P\)(estimate | \(H_0\))。

p值的一些特性：

+   p值越小，结果“统计学意义”越强（或者我们有更多理由拒绝零假设，这反过来又为备择假设提供了证据）。

+   根据错误概率，如果p值小于\(\alpha\)，则结果具有统计学意义（通常小于0.05或5%）。

+   如果与估计值相关的测试统计量小于或大于临界z值或t值，则p < 0.05。

+   与置信区间的对应关系：

    +   如果置信区间包含零假设值（通常为0），则这与高p值一致，表明对零假设的反对证据较弱。

    +   如果置信区间不包括0，则这与低p值一致，表明对零假设的反对证据较强。

    +   例如：均值为0.5，95%置信区间为[0.35; 0.65]，在\(\alpha\)=0.05水平上具有统计学意义（拒绝\(H_0\)）。然而，均值为-0.1，95%置信区间为[-0.25; 0.05]，在\(\alpha\)=0.05水平上**不具有**统计学意义（置信区间包括0，未能拒绝\(H_0\)）。

+   为了计算精确的p值，我们依赖于表格或统计软件。

+   p值**不**说明\(H_0\)或\(H_A\)为真的概率；记住\(P\)(estimate | \(H_0\))！

+   当我们制定单侧假设但使用双尾测试（这在社会科学研究中是常态）时，我们特别小心，不希望犯第一类错误（\(H_0\)为真，但我们拒绝了它）；我们可以将p值除以2，得到如果我们进行单侧测试时得到的p值。

#### 4.5.2.2 测试实施

为了实际测试高收入者和低收入者对政府满意度均值之间的差异，该公式表示为 \(z =\frac{(\bar x_1 - \bar x_2)}{standard error}\)（注意均值差异的标准误并非两个标准误之差）。

我们现在计算均值差异并执行相应的假设检验：

```r
[mean](https://rdrr.io/r/base/mean.html)(data_ess$stfgov[data_ess$hinctnta_di==1], na.rm = TRUE)-
 [mean](https://rdrr.io/r/base/mean.html)(data_ess$stfgov[data_ess$hinctnta_di==0], na.rm = TRUE)
```

```r
## [1] 0.1543566
```

```r
[t.test](https://rdrr.io/r/stats/t.test.html)(data_ess$stfgov[data_ess$hinctnta_di==1], data_ess$stfgov[data_ess$hinctnta_di==0])
```

```r
## 
##  Welch Two Sample t-test
## 
## data:  data_ess$stfgov[data_ess$hinctnta_di == 1] and data_ess$stfgov[data_ess$hinctnta_di == 0]
## t = 1.5882, df = 2046.8, p-value = 0.1124
## alternative hypothesis: true difference in means is not equal to 0
## 95 percent confidence interval:
##  -0.03624397  0.34495717
## sample estimates:
## mean of x mean of y 
##  4.354545  4.200189
```

* * *

**问题：** 详细解释统计检验的结果，并参考具体的p值。

你的答案：

解答：

从样本中获得这种结果的可能性，尽管在总体中\(H_0\)是正确的，是0.11或11%。由于这超过了我们愿意犯错的阈值5%，我们不能拒绝零假设（因此找不到支持\(H_A\)的证据）。

即使考虑到我们提出了一个有方向的假设并依赖于单侧检验（因此我们可以将p值除以2），p值仍然超过了我们假定的错误概率（p值为0.055 > \(\alpha\)为0.05）。因此，即使在单侧检验中，我们也不能拒绝零假设。

* * *

### 4.5.3 相关性的假设检验

而不是关注均值差异，我们现在关注相关性的假设检验。程序与之前类似。我们首先提出零假设和备择假设，指定错误概率（本例中为5%），计算检验统计量（\(z=\frac{r}{SE}\)），计算p值，并解释结果。

* * *

**问题：** 假设更高的收入要么与对政府的满意度正相关，要么负相关。提出零假设和备择假设（非有方向的）。

你的答案：

```r
H_0: H_A:
```

解答：

\(H_A\): 口头表达：收入要么与对政府的满意度正相关，要么负相关。

正式表达：\(r \ne 0\)

\(H_0\): 口头表达：收入与对政府的满意度无关。

正式表达：\(r = 0\)

* * *

我们现在计算检验统计量和p值。

```r
cor <- [cor.test](https://rdrr.io/r/stats/cor.test.html)(data_ess$hinctnta, data_ess$stfgov)
cor
```

```r
## 
##  Pearson's product-moment correlation
## 
## data:  data_ess$hinctnta and data_ess$stfgov
## t = 1.805, df = 2047, p-value = 0.07122
## alternative hypothesis: true correlation is not equal to 0
## 95 percent confidence interval:
##  -0.003445968  0.083023781
## sample estimates:
##        cor 
## 0.03986354
```

* * *

**问题：** 根据t值和p值解释统计检验的结果（双侧检验）。

你的答案：

解答：

从样本中获得这种结果的可能性，尽管在总体中\(H_0\)是正确的，是0.07或7%。由于这超过了我们愿意犯错的阈值5%，结果不具有统计学意义，我们不能拒绝零假设（因此找不到支持\(H_A\)的证据）。

* * *

## 4.6 展望

本案例研究回顾了概率论的基础、一些相关的统计定律、置信区间和假设检验。在例子中，我们主要使用了z分布，如果我们处理的是大样本或合理大的样本（>30已经足够），这就是我们的选择。对于小样本和其他目的，我们使用其他分布进行统计检验，这些分布在此简要介绍。

### 4.6.1 t分布

t 分布对应于大样本的 z 分布。对于小样本，分布更平坦，因此边缘更宽。因此，临界 t 值（在右侧或左侧可以拒绝零假设）在右侧更大，在左侧更小。这使得拒绝零假设变得更加困难。换句话说：在小样本中，我们需要更极端的结果来拒绝零假设。这解释了为什么一些异常值在小样本中具有更大的权重。这有助于避免所谓的 alpha 错误（或第一类错误），它指的是错误地推断出统计上显著的结果，尽管在总体中零假设是真的。alpha 错误通常被认为比未能拒绝零假设（beta 错误或第二类错误）对科学进步构成更大的威胁。

Df 指的是自由度，通常是案例数减 1（用于估计）。

![t 分布](../Images/c287a1a35548f8f82bf97e42fb7dfcd5.png)t 分布

来源：[https://www.geo.fu-berlin.de/en/v/soga-py/Basics-of-statistics/Continous-Random-Variables/Students-t-Distribution/index.html](https://www.geo.fu-berlin.de/en/v/soga-py/Basics-of-statistics/Continous-Random-Variables/Students-t-Distribution/index.html)

### 4.6.2 卡方分布

卡方分布用于测试方差，例如，在交叉表中应用。卡方值永远不会为负，因此只适合进行单侧检验。

Df 指的是自由度，在交叉表的情况下，这将是表格大小，例如。

![卡方分布](../Images/f7ed5980e4a4bc02da3af38a835ec9cc.png)卡方分布

来源：[https://www.geo.fu-berlin.de/en/v/soga-py/Basics-of-statistics/Continous-Random-Variables/Chi-Square-Distribution/index.html](https://www.geo.fu-berlin.de/en/v/soga-py/Basics-of-statistics/Continous-Random-Variables/Chi-Square-Distribution/index.html)

### 4.6.3 F 分布

F 分布基本上是两个卡方分布的随机变量的比值（例如，两个方差）。F 分布始终为正，其函数形式取决于考虑的两个变量的自由度。例如，F 检验用于比较回归模型的模型拟合度。

![F 分布 1](../Images/2de8e3c2e232988c4e1262fb8cdbbea5.png)![F 分布 2](../Images/8597d82a7302d8203525fc4557db3545.png)

来源：[https://www.geo.fu-berlin.de/en/v/soga-py/Basics-of-statistics/Continous-Random-Variables/F-Distribution/index.html](https://www.geo.fu-berlin.de/en/v/soga-py/Basics-of-statistics/Continous-Random-Variables/F-Distribution/index.html)

### 4.6.4 统计意义与实质性意义

这里讨论的程序与假设检验和统计显著性的概念相关。它们本质上关注的是量化不确定性和从这些不确定性中推断出是否可以假设从随机样本中得到的结果可以推广到样本所抽取的总体。尽管这些是重要的概念，但统计显著性不同于科学或实质显著性，后者指的是结果相对于其他研究发现或替代解释因素的相关程度。因此，一个统计上显著的结果可能并不具有实质意义，反之，一个具有实质意义的结果可能并不具有统计显著性。例如，统计显著性高度依赖于计算标准误差时所包含的观测数量。例如，可以通过使用标准化效应量，如beta或Cohen的D值，来获得回归系数实质性的度量。

通常，我们寻找的是既在统计上又在实质上都有显著性的经验结果.* *[3 双变量统计——案例研究：美国总统选举](bivariate-statistics-case-study-united-states-presidential-election.html)[5 回归分析——案例研究：对正义的态度](regression-analysis---case-study-attitudes-toward-justice.html)*
