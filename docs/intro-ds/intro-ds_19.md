地图 > 数据科学 > 预测未来 > 建模 > 分类 > 线性判别分析

# 线性判别分析

线性判别分析（LDA）是一种最初由 R·A·费舍尔于 1936 年开发的分类方法。它简单、数学上健壮，并且通常能产生准确率与更复杂方法一样好的模型。**算法**LDA 基于寻找一组最佳分离两个类别（目标）的变量（预测器）的线性组合的概念。为了捕捉可分离性的概念，费舍尔定义了以下评分函数。

![](img/37aecbc9efcf8175e0bf51fe29adbeb7.jpg)

给定评分函数，问题是估计最大化评分的线性系数，这可以通过以下方程解决。

![](img/3630d0950655fdb2e98a9b2c8ec219a7.jpg)

评估判别效果的一种方法是计算两组之间的**马氏距离**。距离大于 3 意味着两个平均值相差超过 3 个标准差。这意味着重叠（误分类的概率）非常小。

![](img/8b46439cc878333f1cf28e0775cd8af4.jpg)

最后，通过将新点投影到最大分离方向上对其进行分类，并在以下条件下将其分类为*C1*：

![](img/3046b9e244e18c82995c96c08b42d8e0.jpg)

*例子*：假设我们收到一家银行关于其小型企业客户的数据集，其中包括违约（红色方块）和未违约（蓝色圆圈）的客户，以逾期天数（DAYSDELQ）和业务月数（BUSAGE）分隔。我们使用 LDA 来找到最佳线性模型，最佳地分离两个类别（违约和非违约）。

![](img/859d400cc05286876b7e01fdad43ddbd.jpg)

第一步是计算均值（平均）、协方差矩阵和类别概率。

![](img/6a1d31c9f72a6675699438e79c5a1aa2.jpg)

然后，我们计算混合协方差矩阵，最后计算线性模型的系数。

![](img/39e2c5567f811540813d5bcbb65183ec.jpg)

**马氏距离**为 2.32 表明两组之间存在很小的重叠，这意味着线性模型对类别的良好分离。

![](img/914da9b30755cff1e7b2bb5fa253da4b.jpg)

在下表中，我们使用上述 Z 方程计算 Z 得分。然而，仅凭得分本身不能用于预测结果。我们还需要第 5 列中的方程式来选择 N 类或 Y 类。如果计算值大于-1.1，则预测为 N 类，否则为 Y 类。正如下文所示，LDA 模型犯了两个错误。

![](img/1f872e4aa0953f8125a766ee2960c226.jpg)

**预测因子贡献**可以使用模型分数与预测因子之间的简单线性相关性来测试哪些预测因子对判别函数有显著贡献。相关性的变化范围为-1 到 1，-1 和 1 表示最高贡献但方向不同，而 0 表示根本没有贡献。

# 二次判别分析（QDA）

QDA 是具有二次决策边界的通用判别函数，可用于对具有两个或更多类的数据集进行分类。QDA 具有比 LDA 更强的预测能力，但需要估计每个类别的协方差矩阵。

![](img/c8e761fbe717f8ae0dd82f8080aa2651.jpg)

其中**C***[k]*是类*k*的协方差矩阵（-1 表示逆矩阵），|**C***[k]*|是协方差矩阵**C***[k]*的行列式，*P*(*c[k]*)是类*k*的先验概率。分类规则是简单地找到具有最高*Z*值的类。

| ![](img/Lda.txt) | ![](img/dc9f5f2d562c6ce8cb7def0d0596abff.jpg)LDA 互动 |
| --- | --- |

![](img/04c11d11a10b9a2348a1ab8beb8ecdd8.jpg) 尝试发明一个实时 LDA 分类器。您应该能够随时添加或删除数据和变量（预测因子和类）。