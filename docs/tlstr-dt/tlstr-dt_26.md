# 14 预测

> 原文：[`tellingstorieswithdata.com/13-prediction.html`](https://tellingstorieswithdata.com/13-prediction.html)

1.  建模

1.  14 预测

先决条件**

+   阅读《使用 R 进行统计学习的入门》，([James 等人 [2013] 2021](99-references.html#ref-islr))

    +   重点关注第六章“线性模型选择和正则化”，该章节提供了岭回归和 Lasso 回归的概述。

+   阅读《Python 数据分析》，([McKinney [2011] 2022](99-references.html#ref-pythonfordataanalysis))

    +   重点关注第十三章，该章节提供了 Python 数据分析的实例。

+   阅读《使用 R 进行 NFL 分析的入门》，(Congelio 2024)

    +   重点关注第三章“使用 `nflverse` 包族进行 NFL 分析”和第五章“使用 NFL 数据的高级模型创建”。

关键概念和技能**

软件和包**

+   `arrow` (Richardson 等人 2023)

+   `nflverse` (Carl 等人 2023)

+   `poissonreg` (Kuhn 和 Frick 2022)

+   `tidymodels` (Kuhn 和 Wickham 2020)

    +   `parsnip` (Kuhn 和 Vaughan 2022)

    +   `recipes` (Kuhn 和 Wickham 2022)

    +   `rsample` (Frick 等人 2022)

    +   `tune` (Kuhn 2022)

    +   `yarkdstick` (Kuhn, Vaughan, 和 Hvitfeldt 2022)

+   `tidyverse` (Wickham 等人 2019)

+   `tinytable` (Arel-Bundock 2024)

```r
library(arrow)
library(nflverse)
library(poissonreg)
library(tidymodels)
library(tidyverse)
library(tinytable)
```

## 14.1 简介

如第十二章讨论所述，模型往往侧重于推理或预测。一般来说，根据你的关注点，存在不同的文化。一个原因是因果关系的不同强调，这将在第十五章介绍中介绍。我在这里非常笼统地谈论，但通常在推理时，我们非常关注因果关系，而在预测时则不太关注。这意味着当条件与我们的模型预期相当不同时，我们的预测质量将下降——但我们如何知道条件是否足够不同以至于我们应感到担忧？

这种文化差异的另一种方式是因为数据科学和机器学习的兴起，特别是由具有计算机科学或工程背景的人推动的 Python 模型的发展。这意味着存在额外的词汇差异，因为大部分推理都来自统计学。再次强调，这一切都是非常笼统地说的。

在本章中，我首先关注使用 `tidymodels` 的 R 方法进行预测。然后我介绍了一个灰色区域——我之所以试图广泛地讲话的原因——lasso 回归。这是由统计学家开发的，但主要用于预测。最后，我将所有这些内容介绍到 Python 中。

## 14.2 使用 `tidymodels` 进行预测

### 14.2.1 线性模型

当我们专注于预测时，我们通常会想要拟合许多模型。一种实现这一目标的方法是多次复制和粘贴代码。这是可以的，并且这是大多数人开始的方式，但它容易出错，难以发现。更好的方法是：

1.  更容易地进行缩放；

1.  使我们能够仔细思考过拟合问题；并

1.  添加模型评估。

使用 `tidymodels` (Kuhn 和 Wickham 2020) 通过提供一致的语法来满足这些标准，使我们能够轻松地拟合各种模型。像 `tidyverse` 一样，它是一个包的集合。

为了说明这一点，我们想要估计以下模拟跑步数据的模型：

$$ \begin{aligned} y_i | \mu_i &\sim \mbox{Normal}(\mu_i, \sigma) \\ \mu_i &= \beta_0 +\beta_1x_i \end{aligned} $$

其中 $y_i$ 指的是个体 $i$ 的马拉松时间，$x_i$ 指的是他们的五公里时间。在这里，我们说个体 $i$ 的马拉松时间是正态分布的，均值为 $\mu$，标准差为 $\sigma$，其中均值取决于两个参数 $\beta_0$ 和 $\beta_1$ 以及他们的五公里时间。这里“~”表示“分布为”。我们使用这种稍微不同的符号是为了更明确地说明所使用的分布，但这个模型与 $y_i=\beta_0+\beta_1 x_i + \epsilon_i$ 等价，其中 $\epsilon$ 是正态分布的。

由于我们专注于预测，我们担心我们的数据过拟合，这会限制我们关于其他数据集的断言能力。部分解决这个问题的一种方法是将我们的数据集分成两部分使用 `initial_split()`。

```r
sim_run_data <- 
 read_parquet(file = here::here("outputs/data/running_data.parquet"))

set.seed(853)

sim_run_data_split <-
 initial_split(
 data = sim_run_data,
 prop = 0.80
 )

sim_run_data_split
```

```r
<Training/Testing/Total>
<160/40/200>
```
在分割数据后，我们使用 `training()` 和 `testing()` 创建训练集和测试集。

```r
sim_run_data_train <- training(sim_run_data_split)

sim_run_data_test <- testing(sim_run_data_split)
```

我们将 80%的数据集放入训练集中。我们将使用它来估计模型的参数。我们保留了剩余的 20%，并将使用它来评估我们的模型。我们为什么要这样做？我们的担忧是偏差-方差权衡，它困扰着建模的所有方面。我们担心我们的结果可能过于特定于我们所拥有的数据集，以至于它们不适用于其他数据集。为了举一个极端的例子，考虑一个包含十个观察值的数据集。我们可以提出一个模型，它完美地击中这些观察值。但是当我们把那个模型带到其他数据集时，即使是那些由相同的基本过程生成的数据集，它也不会准确。

处理这种担忧的一种方法是以这种方式分割数据。我们使用训练数据来告知我们对系数的估计，然后使用测试数据来评估模型。一个与训练数据中的数据过于吻合的模型在测试数据中表现不佳，因为它对训练数据过于具体。这种测试-训练分割使我们有机会构建一个合适的模型。

比起最初想的，更难恰当地进行这种分离。我们希望避免测试数据集中的某些方面出现在训练数据集中的情况，因为这不适当地泄露了即将发生的事情。这被称为数据泄露。但如果考虑到数据清洗和准备，这很可能涉及整个数据集，那么可能每个数据集的一些特征正在相互影响。Kapoor 和 Narayanan (2023) 在机器学习的应用中发现了广泛的数据泄露，这可能会使许多研究无效。

要使用 `tidymodels`，我们首先指定我们感兴趣的是使用 `linear_reg()` 进行线性回归。然后，我们使用 `set_engine()` 指定线性回归的类型，在这种情况下是多元线性回归。最后，我们使用 `fit()` 指定模型。虽然这种方法比上面详细说明的 base R 方法需要更多的基础设施，但这种方法的优势在于它可以用来拟合许多模型；我们可以说我们创建了一个模型工厂。

```r
sim_run_data_first_model_tidymodels <-
 linear_reg() |>
 set_engine(engine = "lm") |>
 fit(
 marathon_time ~ five_km_time + was_raining,
 data = sim_run_data_train
 )
```

估计的系数总结在 表 12.4 的第一列中。例如，我们发现，在我们的数据集中，平均而言，五公里跑步时间每增加一分钟，马拉松时间就增加大约八分钟。
  
### 14.2.2 逻辑回归

我们还可以使用 `tidymodels` 解决逻辑回归问题。为了实现这一点，我们首先需要将因变量的类别改变为因子，因为这是分类模型所必需的。

```r
week_or_weekday <- 
 read_parquet(file = "outputs/data/week_or_weekday.parquet")

set.seed(853)

week_or_weekday <-
 week_or_weekday |>
 mutate(is_weekday = as_factor(is_weekday))

week_or_weekday_split <- initial_split(week_or_weekday, prop = 0.80)
week_or_weekday_train <- training(week_or_weekday_split)
week_or_weekday_test <- testing(week_or_weekday_split)

week_or_weekday_tidymodels <-
 logistic_reg(mode = "classification") |>
 set_engine("glm") |>
 fit(
 is_weekday ~ number_of_cars,
 data = week_or_weekday_train
 )
```

正如之前一样，我们可以绘制实际结果与我们的估计值的对比图。但这个方面很棒的是，我们可以使用我们的测试数据集更彻底地评估我们模型的预测能力，例如通过混淆矩阵，它指定了每个预测的真实情况。我们发现模型在保留的数据集上表现良好。有 90 个观测值，模型预测它是工作日，实际上也是工作日，还有 95 个观测值，模型预测它是周末，实际上也是周末。有 15 个观测值预测错误，这些观测值分布在七个预测为工作日但实际上是周末的案例中，以及八个预测相反情况的案例中。

```r
week_or_weekday_tidymodels |>
 predict(new_data = week_or_weekday_test) |>
 cbind(week_or_weekday_test) |>
 conf_mat(truth = is_weekday, estimate = .pred_class)
```

```r
 Truth
Prediction  0  1
         0 90  8
         1  7 95
```
#### 14.2.2.1 美国政治支持

一种方法是使用 `tidymodels` 以与之前相同的方式构建一个以预测为重点的逻辑回归模型，即使用验证集方法（[James 等人 [2013] 2021，176](99-references.html#ref-islr)）。在这种情况下，概率将是投票给拜登的概率。

```r
ces2020 <- 
 read_parquet(file = "outputs/data/ces2020.parquet")

set.seed(853)

ces2020_split <- initial_split(ces2020, prop = 0.80)
ces2020_train <- training(ces2020_split)
ces2020_test <- testing(ces2020_split)

ces_tidymodels <-
 logistic_reg(mode = "classification") |>
 set_engine("glm") |>
 fit(
 voted_for ~ gender + education,
 data = ces2020_train
 )

ces_tidymodels
```

```r
parsnip model object

Call:  stats::glm(formula = voted_for ~ gender + education, family = stats::binomial, 
    data = data)

Coefficients:
                  (Intercept)                     genderMale  
                       0.2157                        -0.4697  
educationHigh school graduate          educationSome college  
                      -0.1857                         0.3502  
              education2-year                education4-year  
                       0.2311                         0.6700  
           educationPost-grad  
                       0.9898  

Degrees of Freedom: 34842 Total (i.e. Null);  34836 Residual
Null Deviance:      47000 
Residual Deviance: 45430    AIC: 45440
```
然后评估它在测试集上的表现。看起来模型在识别特朗普支持者方面有困难。

```r
ces_tidymodels |>
 predict(new_data = ces2020_test) |>
 cbind(ces2020_test) |>
 conf_mat(truth = voted_for, estimate = .pred_class)
```

```r
 Truth
Prediction Trump Biden
     Trump   656   519
     Biden  2834  4702
```
当我们介绍 `tidymodels` 时，我们讨论了随机构建训练集和测试集的重要性。我们使用训练集来估计参数，然后评估模型在测试集上的表现。自然地，我们会问为什么我们要受随机性的影响，以及我们是否充分利用了我们的数据。例如，如果由于测试集中的一些随机包含，一个好的模型被错误地评估，那会怎样？进一步地，如果我们没有大量的测试集，那会怎样？

一种常用的重采样方法，部分解决了这个问题，是 $k$ 折交叉验证。在这个方法中，我们从数据集中创建 $k$ 个不同的样本，或称为“折”，而不进行替换。然后我们将模型拟合到前 $k-1$ 个折，并在最后一个折上评估它。我们重复这个过程 $k$ 次，每次针对一个折，这样每个观测值将被用于训练 $k-1$ 次，并用于测试一次。$k$ 折交叉验证的估计值是平均均方误差（[James 等人 [2013] 2021，181](99-references.html#ref-islr)）。例如，可以使用 `tidymodels` 中的 `vfold_cv()` 函数创建，比如说，十个折。

```r
set.seed(853)

ces2020_10_folds <- vfold_cv(ces2020, v = 10)
```

然后可以使用 `fit_resamples()` 在不同折的组合上拟合模型。在这种情况下，模型将拟合十次。

```r
ces2020_cross_validation <-
 fit_resamples(
 object = logistic_reg(mode = "classification") |> set_engine("glm"),
 preprocessor = recipe(voted_for ~ gender + education,
 data = ces2020),
 resamples = ces2020_10_folds,
 metrics = metric_set(accuracy, sens),
 control = control_resamples(save_pred = TRUE)
 )
```

我们可能对了解我们模型的性能感兴趣，我们可以使用 `collect_metrics()` 在折之间聚合它们（?tbl-metricsvoters-1）。这类细节通常在论文的主要内容中一带而过，但在附录中会包含详细说明。我们模型在折之间的平均准确率为 0.61，平均敏感度为 0.19，平均特异度为 0.90。

```r
collect_metrics(ces2020_cross_validation) |>
 select(.metric, mean) |>
 tt() |> 
 style_tt(j = 1:2, align = "lr") |> 
 format_tt(digits = 2, num_mark_big = ",", num_fmt = "decimal") |> 
 setNames(c("Metric", "Mean"))
conf_mat_resampled(ces2020_cross_validation) |>
 mutate(Proportion = Freq / sum(Freq)) |>
 tt() |> 
 style_tt(j = 1:4, align = "llrr") |> 
 format_tt(digits = 2, num_mark_big = ",", num_fmt = "decimal")
```

表 14.1：十个折的逻辑回归预测选民偏好的平均度量

| 度量标准 | 平均 |
| --- | --- |
| 准确率 | 0.61 |
| 敏感度 | 0.19 |
| 预测 | 真实 | 频率 | 比例 |
| --- | --- | --- | --- |
| 特朗普 | 特朗普 | 327.5 | 0.08 |
| 特朗普 | 拜登 | 267.7 | 0.06 |
| 拜登 | 特朗普 | 1,428.3 | 0.33 |
| 拜登 | 拜登 | 2,331.9 | 0.54 |

这意味着什么？准确率是正确分类的观察值的比例。0.61 的结果表明模型的表现比抛硬币要好，但好不了多少。灵敏度是识别为真的真实观察值的比例（[詹姆斯等人 [2013] 2021，145](99-references.html#ref-islr)）。在这种情况下，这意味着模型预测受访者投票给了特朗普，他们确实是这样做的。特异性是识别为假的假观察值的比例（[詹姆斯等人 [2013] 2021，145](99-references.html#ref-islr)）。在这种情况下，它是预测投票给拜登的选民中实际投票给拜登的比例。这证实了我们的初步想法，即模型在识别特朗普支持者方面存在困难。

通过查看混淆矩阵（?tbl-metricsvoters-2）我们可以更详细地了解这一点。当与重采样方法（如交叉验证）结合使用时，混淆矩阵为每个折叠计算一次，然后取平均值。模型预测拜登的次数比我们根据 2020 年选举的接近程度所预期的要多。这表明我们的模型可能需要额外的变量来做得更好。

最后，我们可能对个体层面的结果感兴趣，并且可以使用 `collect_predictions()` 函数将这些结果添加到我们的数据集中。

```r
ces2020_with_predictions <-
 cbind(
 ces2020,
 collect_predictions(ces2020_cross_validation) |>
 arrange(.row) |>
 select(.pred_class)
 ) |>
 as_tibble()
```

例如，我们可以看到，除了没有高中、高中毕业生或两年大学学历的男性之外，模型基本上都在预测所有个体的拜登支持率（表 14.2）。

```r
ces2020_with_predictions |>
 group_by(gender, education, voted_for) |>
 count(.pred_class) |>
 tt() |> 
 style_tt(j = 1:5, align = "llllr") |> 
 format_tt(digits = 0, num_mark_big = ",", num_fmt = "decimal") |> 
 setNames(c(
 "Gender",
 "Education",
 "Voted for",
 "Predicted vote",
 "Number"
 ))
```

表 14.2：模型预测所有女性和许多男性的支持率都是拜登，无论教育程度如何

| 性别 | 教育 | 投票给 | 预测投票 | 数量 |
| --- | --- | --- | --- | --- |
| 女 | 无高中 | 特朗普 | 比德恩 | 206 |
| 女 | 无高中 | 比德恩 | 比德恩 | 228 |
| 女 | 高中毕业生 | 特朗普 | 比德恩 | 3,204 |
| 女 | 高中毕业生 | 比德恩 | 比德恩 | 3,028 |
| 女 | 一些大学学历 | 特朗普 | 比德恩 | 1,842 |
| 女 | 一些大学学历 | 比德恩 | 比德恩 | 3,325 |
| 女 | 两年制 | 特朗普 | 特朗普 | 1,117 |
| 女 | 两年制 | 比德恩 | 比德恩 | 1,739 |
| 女 | 本科 | 特朗普 | 比德恩 | 1,721 |
| 女 | 本科 | 比德恩 | 比德恩 | 4,295 |
| 女 | 研究生 | 特朗普 | 比德恩 | 745 |
| 女 | 研究生 | 比德恩 | 比德恩 | 2,853 |
| 男 | 无高中 | 特朗普 | 特朗普 | 132 |
| 男 | 无高中 | 比德恩 | 特朗普 | 123 |
| 男 | 高中毕业生 | 特朗普 | 特朗普 | 2,054 |
| 男 | 高中毕业生 | 比德恩 | 特朗普 | 1,528 |
| 男 | 一些大学学历 | 特朗普 | 比德恩 | 1,992 |
| 男 | 一些大学学历 | 比德恩 | 比德恩 | 2,131 |
| 男 | 两年制 | 特朗普 | 特朗普 | 1,089 |
| 男 | 两年制 | 比德恩 | 特朗普 | 1,026 |
| 男 | 本科 | 特朗普 | 比德恩 | 2,208 |
| 男 | 本科 | 比德恩 | 比德恩 | 3,294 |
| 男 | 研究生 | 特朗普 | 比德恩 | 1,248 |

| 男 | 研究生 | 特朗普 | 比德恩 | 2,426 |
  
### 14.2.3 泊松回归

我们可以使用 `tidymodels` 使用 `poissonreg` 估计泊松模型（Kuhn 和 Frick 2022）(表 13.4）。

```r
count_of_A <- 
 read_parquet(file = "outputs/data/count_of_A.parquet")

set.seed(853)

count_of_A_split <-
 initial_split(count_of_A, prop = 0.80)
count_of_A_train <- training(count_of_A_split)
count_of_A_test <- testing(count_of_A_split)

grades_tidymodels <-
 poisson_reg(mode = "regression") |>
 set_engine("glm") |>
 fit(
 number_of_As ~ department,
 data = count_of_A_train
 )
```

此估计的结果在表 13.4 的第二列。由于分割，观察的数量较少，它们与 `glm()` 的估计相似。
  
## 14.3 Lasso 回归

巨人的肩膀* *罗伯特·蒂布斯希瑞尼博士是斯坦福大学统计学和生物医学数据科学系的教授。1981 年，他从斯坦福大学获得统计学博士学位后，加入了多伦多大学担任助理教授。1994 年，他被晋升为正教授，并于 1998 年搬到了斯坦福。他做出了包括上述 GAMs 和 lasso 回归在内的基本贡献，lasso 回归是一种自动变量选择的方法。他是 James 等人（[[2013] 2021](99-references.html#ref-islr)）的作者。1996 年，他获得了 COPSS 总统奖，并于 2019 年被任命为皇家学会院士。

## 14.4 使用 Python 进行预测

### 14.4.1 设置

我们将在 VSCode 中使用 Python，这是微软提供的免费 IDE，您可以从[这里](https://code.visualstudio.com)下载。然后安装 Quarto 和 Python 扩展。

### 14.4.2 数据

使用 parquet 读取数据。*

使用 pandas 进行操作。*

### 14.4.3 模型

#### 14.4.3.1 scikit-learn

#### 14.4.3.2 TensorFlow

## 14.5 练习

### 练习

1.  *(Plan)* 考虑以下场景：*一年中每天你和你叔叔玩飞镖。每一轮每人投掷三支飞镖。每一轮结束后，你将三支飞镖击中的点数相加。所以如果你击中 3、5 和 10，那么那一轮的总分是 18 分。你叔叔有点仁慈，如果你击中的数字小于 5，他假装没看见，让你有机会重新投掷那支飞镖。假设你每天玩 15 轮。* 请绘制出这个数据集可能的样子，然后绘制一个图形来展示所有观察结果。

1.  *(Simulate)* 请进一步考虑所描述的场景，并模拟这种情况。比较如果你没有机会重新投掷，你叔叔的总分与你的总分，以及最终的实际总分。请至少基于模拟数据包括十个测试。

1.  *(Acquire)* 请描述一个可能的此类数据集（或一个对你感兴趣的相关运动或情况）的来源。

1.  *(Explore)* 请使用 `ggplot2` 构建你绘制的图形。然后使用 `tidymodels` 构建一个预测模型，预测谁会获胜。

1.  *(Communicate)* 请写两段关于你所做事情的描述。

### 测验

### 课堂活动

### 任务

请使用 `nflverse` 加载 NFL 四分卫在常规赛期间的某些统计数据。[数据字典](https://nflreadr.nflverse.com/articles/dictionary_player_stats.html)将有助于理解数据。

```r
qb_regular_season_stats <- 
 load_player_stats(seasons = TRUE) |> 
 filter(season_type == "REG" & position == "QB")
```

假设您是一名 NFL 分析师，并且现在是 2023 年常规赛的中期，即第 9 轮比赛刚刚结束。我对您能为每个球队在剩余赛季（即第 10-18 周）生成的最佳 `passing_epa` 预测模型感兴趣。

使用 Quarto，并包含适当的标题、作者、日期、GitHub 仓库链接、章节和引用，并为管理层撰写一份 2-3 页的报告。最佳性能可能需要创造性的特征工程。欢迎您使用 R 或 Python，任何模型，但您应该小心指定模型并在高层次上解释其工作原理。注意泄漏问题！

Arel-Bundock, Vincent. 2024\. *tinytable: 简单且可配置的“HTML”、“LaTeX”、“Markdown”、“Word”、“PNG”、“PDF”和“Typst”格式的表格*. [`vincentarelbundock.github.io/tinytable/`](https://vincentarelbundock.github.io/tinytable/).Carl, Sebastian, Ben Baldwin, Lee Sharpe, Tan Ho, and John Edwards. 2023\. *Nflverse: 简易安装和加载的‘Nflverse’*. [`CRAN.R-project.org/package=nflverse`](https://CRAN.R-project.org/package=nflverse).Congelio, Bradley. 2024\. *使用 R 进行 NFL 分析入门*. 1st ed. Chapman; Hall/CRC. [`bradcongelio.com/nfl-analytics-with-r-book/`](https://bradcongelio.com/nfl-analytics-with-r-book/).Frick, Hannah, Fanny Chow, Max Kuhn, Michael Mahoney, Julia Silge, and Hadley Wickham. 2022\. *rsample: 通用重采样基础设施*. [`CRAN.R-project.org/package=rsample`](https://CRAN.R-project.org/package=rsample).James, Gareth, Daniela Witten, Trevor Hastie, and Robert Tibshirani. (2013) 2021\. *使用 R 进行统计学习的入门*. 2nd ed. Springer. [`www.statlearning.com`](https://www.statlearning.com).Kapoor, Sayash, and Arvind Narayanan. 2023\. “基于机器学习的科学中的泄露和可重复性危机.” *Patterns* 4 (9): 1–12\. [`doi.org/10.1016/j.patter.2023.100804`](https://doi.org/10.1016/j.patter.2023.100804).Kuhn, Max. 2022\. *tune: 整洁的调优工具*. [`CRAN.R-project.org/package=tune`](https://CRAN.R-project.org/package=tune).Kuhn, Max, and Hannah Frick. 2022\. *poissonreg: Poisson 回归的模型包装器*. [`CRAN.R-project.org/package=poissonreg`](https://CRAN.R-project.org/package=poissonreg).Kuhn, Max, and Davis Vaughan. 2022\. *parsnip: 模型和分析函数的通用 API*. [`CRAN.R-project.org/package=parsnip`](https://CRAN.R-project.org/package=parsnip).Kuhn, Max, Davis Vaughan, and Emil Hvitfeldt. 2022\. *yardstick: 模型性能的整洁描述*. [`CRAN.R-project.org/package=yardstick`](https://CRAN.R-project.org/package=yardstick).Kuhn, Max, and Hadley Wickham. 2020\. *tidymodels: 使用 tidyverse 原则进行建模和机器学习的包集合*. [`www.tidymodels.org`](https://www.tidymodels.org).———. 2022\. *recipes: 建模的前处理和特征工程步骤*. [`CRAN.R-project.org/package=recipes`](https://CRAN.R-project.org/package=recipes).McKinney, Wes. (2011) 2022\. *Python 数据分析*. 3rd ed. [`wesmckinney.com/book/`](https://wesmckinney.com/book/).Richardson, Neal, Ian Cook, Nic Crane, Dewey Dunnington, Romain François, Jonathan Keane, Dragoș Moldovan-Grünfeld, Jeroen Ooms, and Apache Arrow. 2023\. *arrow: 集成到 Apache Arrow*. [`CRAN.R-project.org/package=arrow`](https://CRAN.R-project.org/package=arrow).Wickham, Hadley, Mara Averick, Jenny Bryan, Winston Chang, Lucy D’Agostino McGowan, Romain François, Garrett Grolemund, et al. 2019\. “欢迎来到 Tidyverse.” *开源软件杂志* 4 (43): 1686\. [`doi.org/10.21105/joss.01686`](https://doi.org/10.21105/joss.01686).

