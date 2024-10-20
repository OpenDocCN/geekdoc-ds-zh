# 无监督机器学习的简单指南(2023)

> 原文：<https://www.dataquest.io/blog/unsupervised-machine-learning/>

December 19, 2022![Introduction to machine learning](img/1c7e88fd208634ed965a4022a012c6d1.png)

当开始机器学习时，通常要花一些时间来预测值。这些值可能是信用卡交易是否是欺诈性的，基于客户的行为模式，客户赚了多少，等等。在这样的场景中，我们正在使用有监督的机器学习。

在监督机器学习中，数据集包含我们试图预测的目标变量。顾名思义，我们可以监督模型的性能，因为可以客观地验证其输出是否正确。

当使用无监督算法时，我们有一个未标记的数据集，这意味着我们没有试图预测的目标变量。事实上，我们的目标不是预测任何事情，而是在数据中发现模式。

因为没有目标变量，我们不能通过客观地确定输出是否正确来监督算法。因此，由数据科学家来分析输出并理解算法在数据中找到的模式。

下图说明了这种差异:

![image+1.png](img/17cbb3dd1c519e83f0f843190d3c6a9c.png)

最常见的无监督机器学习类型包括以下几种:

*聚类:根据数据中发现的模式将数据集划分成组的过程，例如，用于划分客户和产品。

*   关联:目标是发现变量之间的模式，而不是条目——例如，经常用于购物篮分析。

*   异常检测:这种算法试图识别特定数据点何时完全脱离数据集模式的其余部分，通常用于欺诈检测。

## 使聚集

聚类算法根据每个数据点的特征将数据集分成多个组。

例如，假设我们有一个包含成千上万客户数据的数据库。我们可以使用聚类将这些客户分成不同的类别，以便为每个群体应用不同的营销策略。

### KMeans 算法

K-means 算法是一种迭代算法，设计用于在给定用户设置的多个聚类的情况下，为数据集寻找分裂。集群的数量称为 k。

在 K-means 中，该算法随机选择 K 个点作为聚类的中心。这些点被称为星团的质心。k 由用户设置。然后，迭代过程开始，其中每次迭代是将数据点分配到最近的质心并重新计算质心作为聚类中的点的平均值的过程。这个过程一直持续下去，直到每个质心位于其聚类的平均值。

下图说明了这一过程:

![image+2.gif](img/53f19c54076f410f0d2549faac1063cf.png)

现在让我们来看一个 K-means 算法的例子。我们有一个客户的[数据集，我们将根据他们的年收入和支出分数对这些客户进行细分。下面是代表这两个变量的曲线图:](https://dq-blog.s3.amazonaws.com/introduction-to-unsupervised-machine-learning/mall_customers.csv)

```py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reading the data
sns.set_style('whitegrid')
df = pd.read_csv('mall_customers.csv')

# Plotting the data
fig, ax = plt.subplots(figsize=(12, 6))

sns.scatterplot('Annual Income', 'Spending Score', data=df, ax=ax)

plt.tight_layout()
plt.show()
```

```py
/usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
```

![image+3.gif](img/92801b1228b13bdacc81f5b01dd5a3a2.png)

然后，我们将数据集分成五个集群:

```py
# Using KMeans
from sklearn.cluster import KMeans

model = KMeans(n_clusters=5)
y = model.fit_predict(df[['Annual Income', 'Spending Score']])

df['CLUSTER'] = y
```

现在，我们有了按刚刚创建的集群分组的相同图:

```py
# Plotting the clustered data
fig, ax = plt.subplots(figsize=(12, 6))

sns.scatterplot('Annual Income', 'Spending Score', data=df, hue='CLUSTER', ax=ax, cmap='tab10')

plt.tight_layout()
plt.show()
```

```py
/usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
```

![image+4.png](img/feb695cbed219523d54b075675fcc292.png)

你可以在这里查看 KMeans 文档。

### 分层聚类

分层聚类是一种基于在聚类之间建立层次结构的聚类技术。这种技术细分为两种主要方法:

*   凝聚聚类
*   分裂聚类

聚集或自下而上的方法包括将每个数据点指定为单个聚类，然后通过迭代过程将这些聚类组合在一起。

比方说，我们有一个 100 行的数据集。首先，自下而上的方法将有 100 个集群。然后，每个点被分组到最近的一个点，创建更大的集群。这些新的聚类不断与最近的聚类组合在一起，直到我们有一个包含所有点的聚类。

分裂的，或自上而下的方法以相反的方式工作。首先，所有的数据点被分组到一个聚类中。然后将最远的点从主簇中取出，一遍又一遍，直到每个点都成为自己的簇。

自底向上的方法远比自顶向下的方法更常见，所以让我们来看一个使用它的例子。我们使用 K-means 示例中使用的同一数据集的样本。下图被称为[树状图](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html)，这是我们可视化层次聚类的方式:

```py
# Creating a dendogram with scipy
import scipy.cluster.hierarchy as shc

sample = df.sample(30)
plt.figure(figsize=(12,6))
dend = shc.dendrogram(shc.linkage(sample[['Annual Income', 'Spending Score']], method='ward'))
plt.grid(False)
```

![image+5.png](img/71fb0497eb7570063975035fb4e14ede.png)

请注意，每个点在开始时都是一个集群，然后它们被分组，直到出现一个集群。

此外，我们将方法设置为 **Ward** 。这是执行分层聚类的一种常见方式。该方法通过计算聚类之间的平方距离之和，在聚类过程的每一步进行聚类合并。有多种其他方法可以使用最大、最小和平均距离，等等。你可以在[文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage)中查到。

我们现在需要设置一个截止点。没有正确的方法可以做到这一点，决定聚类的数量可能是任何聚类过程中最棘手的一步。有几个工具可以帮助你确定这个数字，比如[剪影](https://en.wikipedia.org/wiki/Silhouette_(clustering))和[肘](https://en.wikipedia.org/wiki/Elbow_method_(clustering))方法。在我们的例子中，我们创建了两个集群，树状图如下所示:

```py
import scipy.cluster.hierarchy as shc

plt.figure(figsize=(12,6))
dend = shc.dendrogram(shc.linkage(sample[['Annual Income', 'Spending Score']], method='ward'))
plt.axhline(y=150, color='black', linestyle='--')

plt.grid(False)
```

![image+6.png](img/7d19820cd21f5c4908eba477641c9fe6.png)

从下面的分界点开始，我们剩下两个集群。

## 联合

关联是一种无监督的学习技术，用于发现数据中的“隐藏”规则和模式。它的经典用例被称为*市场篮子*分析。

购物篮分析包括发现彼此高度相关的项目。换句话说，我们使用大量购买的数据来确定哪些商品经常一起购买，以便在网上商店向客户提供建议，或者确定在实体店展示产品的最佳方式。

我们有一个[数据集，包含在杂货店](https://dq-blog.s3.amazonaws.com/introduction-to-unsupervised-machine-learning/Groceries+data.csv)购买的数千次商品的信息。在每一行中，我们都有一个属于购买的项目。它看起来是这样的:

```py
# Reading the initial data
df = pd.read_csv('Groceries data.csv')
df.sort_values(['Member_number', 'Date']).head()
```

|  | 成员编号 | 日期 | 项目说明 | 年 | 月 | 天 | 星期几 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Thirteen thousand three hundred and thirty-one | One thousand | 2014-06-24 | 全脂奶 | Two thousand and fourteen | six | Twenty-four | one |
| Twenty-nine thousand four hundred and eighty | One thousand | 2014-06-24 | 面粉糕饼 | Two thousand and fourteen | six | Twenty-four | one |
| Thirty-two thousand eight hundred and fifty-one | One thousand | 2014-06-24 | 咸点心 | Two thousand and fourteen | six | Twenty-four | one |
| Four thousand eight hundred and forty-three | One thousand | 2015-03-15 | 香肠 | Two thousand and fifteen | three | Fifteen | six |
| Eight thousand three hundred and ninety-five | One thousand | 2015-03-15 | 全脂奶 | Two thousand and fifteen | three | Fifteen | six |

前三行是同一客户在同一天购买的商品。我们假设这是一个单一的交易。

然后，我们以这样一种方式操作这个数据集(这不是本文的重点),即每一行都变成一个完整的事务。对于数据集中被视为`True`或`False`的每个唯一项目，我们都有一列，这取决于它是否是该行中表示的事务的一部分:

```py
# Reading data after manipulation
transactions = pd.read_csv('transactions.csv')
transactions = transactions.astype(bool)
transactions.head()
```

|  | 速食食品 | 超高温牛奶 | 磨料清洁剂 | 阿提夫。好处 | 婴儿化妆品 | 包 | 发酵粉 | 浴室清洁剂 | 牛肉 | 浆果 | … | 火鸡 | 醋 | 闲聊 | 生奶油/酸奶油 | 威士忌酒 | 白面包 | 白葡萄酒 | 全脂奶 | 酸奶 | 洋葱卷 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Zero | 真实的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | … | 错误的 | 错误的 | 错误的 | 真实的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 |
| one | 真实的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | … | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 |
| Two | 真实的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | … | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 |
| three | 真实的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | … | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 真实的 | 错误的 | 错误的 |
| four | 真实的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | … | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 | 错误的 |

5 行× 167 列

你可以在这里访问这个[修改过的数据集。](https://dq-blog.s3.amazonaws.com/introduction-to-unsupervised-machine-learning/transactions.csv)

有了这些数据，我们将使用[先验](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/)算法。例如，这是一个著名的算法，用于识别频繁出现在一个购物篮中的多组商品。这些集合从一个项目到我们设定的所有项目。在我们的案例中，我们不希望器械包包含三件以上的物品:

```py
# Using the apriori algorithm
from mlxtend.frequent_patterns import apriori

frequent_itemsets = apriori(transactions, min_support=0.001, use_colnames=True, max_len=3)
frequent_itemsets
```

|  | 支持 | 项目集 |
| --- | --- | --- |
| Zero | 0.028612 | (即食食品) |
| one | 0.016691 | (超高温牛奶) |
| Two | 0.001431 | (磨料清洁剂) |
| three | 0.001431 | (artif。甜味剂) |
| four | 0.008107 | (发酵粉) |
| … | … | … |
| One thousand eight hundred and ninety-nine | 0.001431 | (全脂牛奶、香肠、zwieback) |
| One thousand nine hundred | 0.001431 | (苏打水、酸奶、热带水果) |
| One thousand nine hundred and one | 0.001431 | (特色巧克力、全脂牛奶、酸奶) |
| One thousand nine hundred and two | 0.001431 | (鲜奶油/酸奶油、酸奶、热带水果) |
| One thousand nine hundred and three | 0.001907 | (全脂牛奶、酸奶、热带水果) |

1904 行× 2 列

我们看到的支持度量对应于特定项目集出现的概率。例如，假设我们有一个 100 个篮子的数据集，其中牛奶和咖啡的组合出现在 20 个篮子中。这意味着这种组合发生在 20%的篮子里；所以支撑是 0.20。

我们现在将使用一个函数来查找数据中的规则。该函数将检查 apriori 算法创建的集合，以识别经常一起购买的产品，并确定它们是否在同一个集合中。

```py
# Using association rules
from mlxtend.frequent_patterns import association_rules

rules = association_rules(frequent_itemsets, metric="lift",  min_threshold=1.5)

display(rules.head())
print(f"Number of rules: {len(rules)}")
```

|  | 祖先 | 结果 | 先行支持 | 后续支持 | 支持 | 信心 | 电梯 | 杠杆作用 | 定罪 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Zero | (瓶装啤酒) | (超高温牛奶) | 0.041965 | 0.016691 | 0.001907 | 0.045455 | 2.723377 | 0.001207 | 1.030134 |
| one | (超高温牛奶) | (瓶装啤酒) | 0.016691 | 0.041965 | 0.001907 | 0.114286 | 2.723377 | 0.001207 | 1.081653 |
| Two | (法兰克福) | (超高温牛奶) | 0.031950 | 0.016691 | 0.001431 | 0.044776 | 2.682729 | 0.000897 | 1.029402 |
| three | (超高温牛奶) | (法兰克福) | 0.016691 | 0.031950 | 0.001431 | 0.085714 | 2.682729 | 0.000897 | 1.058804 |
| four | (人造黄油) | (超高温牛奶) | 0.032427 | 0.016691 | 0.001431 | 0.044118 | 2.643277 | 0.000889 | 1.028693 |

```py
Number of rules: 2388
```

除了支持，我们还应该了解另外两个重要指标。

信心是一个完整的交易发生的概率，假设他们的第一个项目已经在篮子里。换句话说，假设一个人既买牛奶又买咖啡的概率(支持度)是 0.5，一个人只买咖啡的概率(支持度)是 0.10。置信度为 0.10/0.05，等于 0.5。从数学角度来说:

$$
信心(牛奶→咖啡)= \ frac {支持(牛奶→咖啡)} {支持(牛奶)}
$$

**Lift** 指标表示规则出现的频率比我们预期的要高多少。如果 lift 等于 1，则意味着这些项目在统计上相互独立，无法从中得出任何规则。所以，升力越高，我们对规则就越有信心。请注意，在代码中，我们选择根据提升对规则进行排序，并将最小阈值设置为 1.5。

从数学上讲，这种提升是对第二项的信心超过支持:

$$
lift(牛奶→咖啡)= \frac{confidence(牛奶→咖啡)} {support(牛奶→咖啡)}
$$

还有一些其他指标。检查[文档](https://http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/)以查看所有内容。

## 异常检测

异常检测是在数据集中发现异常数据点的过程。换句话说，我们正在寻找完全脱离数据集模式的异常值和点。

这种机器学习通常用于检测欺诈性信用卡交易或设备或机器的故障或即将发生的故障。

虽然我们将异常检测作为无监督的机器学习过程来处理，但它也可以作为有监督的算法来执行。然而，要做到这一点，我们需要一个带标签的数据集，我们知道每个数据点是否是异常的，以便训练一个模型来执行分类任务。

但是我们并不总是拥有我们想要的所有数据。如果我们没有那个标记的数据集，那么它就变成了一个无监督的学习问题，异常检测算法可以帮助我们。

当我们谈论异常检测时，我们不是在谈论一种单一的算法。同一任务有多种不同的算法，每种算法使用不同的策略，得到不同的解决方案。

Scikit-learn 为异常检测实现了一些不同的算法。在这里，我们将提供其中三个的快速介绍和比较，但是请随意查看伟大的 [Scikit-learn 的文档](https://scikit-learn.org/stable/modules/outlier_detection.html)以获得更多信息。

### 隔离森林

[隔离林](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)是一种基于树的算法，用于异常检测。该算法通过随机选择一个特征和该特征中的分割值来工作。这个过程一直持续到我们将数据点隔离为树中的节点。

这一过程的随机性使得异常数据点比其他数据点需要更少的分裂来隔离。因为整棵树是多次创建的，所以我们称之为“森林”当同一个观察结果被快速隔离多次时，算法认为这是一个异常是安全的。

为了使用 Scikit-learn 的 Isolation Forest 实现创建一个快速示例，我们将使用与集群部分相同的客户数据集，考虑相同的两个变量:年收入和支出分数。

下面的代码使用算法在数据集中创建一个新列，其中-1 表示异常，1 表示非异常。散点图让我们可以看到这些类是如何分布的:

```py
# using the Isolation Forest algorithm
from sklearn.ensemble import IsolationForest

model = IsolationForest()
y_pred = model.fit_predict(df[['Annual Income', 'Spending Score']])
df['class'] = y_pred

# Plotting the data
fig, ax = plt.subplots(figsize=(12, 6))

sns.scatterplot('Annual Income', 'Spending Score', data=df, hue='class', ax=ax, cmap='tab10', alpha=0.8)

plt.tight_layout()
plt.show()
```

```py
/usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
```

![image+7.png](img/1507848c53fa90dbb3c6431308f1d16d.png)

### DBSCAN

[DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) 是含噪声应用的基于密度的空间聚类的简称。这不仅是一个异常检测算法，也是一个聚类算法。它的工作原理是将观察结果分成三大类。

**基础**类由在确定半径的圆内具有一定数量的邻居的观测值构成。邻居的数量和圆的半径是由用户设置的超参数。

**边界**观测值的邻居数量少于既定数量，但仍至少有一个，而完全没有邻居的观测值被视为**噪声**类。

该算法迭代地选择随机点，并且基于超参数，确定它们属于哪一类。这种情况一直持续到所有的观察结果都被分配到一个类中。

如前所述，DBSCAN 可以产生多个集群，因为我们有可能拥有比半径超参数相距最远的基本观测值。

让我们在运行隔离林的同一数据集中运行 DBSCAN。例如，我们将半径设置为 12，邻居数量设置为 5:

```py
# Using DBSCAN
from sklearn.cluster import DBSCAN

model = DBSCAN(eps=12, min_samples=5) 
y_pred = model.fit_predict(df[['Annual Income', 'Spending Score']])
df['class'] = y_pred

# Plotting the data
fig, ax = plt.subplots(figsize=(12, 6))

sns.scatterplot('Annual Income', 'Spending Score', data=df, hue='class', ax=ax, cmap='tab10')

plt.tight_layout()
plt.show()
```

```py
/usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
```

![image+8.png](img/6b10a8a7a0aee5b0b1e16b5e8d3fd6ed.png)

请注意，与隔离林相比，我们有了更少的异常。

### 局部异常因素

[局部异常值因子- LOF](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html) 的工作方式类似于 DBSCAN 算法。它们都是基于寻找不属于任何局部观察组的数据点。主要区别在于，局部异常值因子发布另一种算法，[K-最近邻–KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors)，以确定局部密度。

的数量 K 由用户设置，并传递给 KNN，以确定每个数据点的相邻组。当我们将一个观测值的局部密度与其相邻观测值的局部密度进行比较时，可以识别出密度相似的区域和密度明显低于其相邻观测值的区域，这些区域被视为异常值或异常值。

使用 Scikit-learn 的实现并创建散点图，我们可以看到异常不仅位于图的边缘，而且分布在整个图中。这是因为我们根据它们的位置和与近邻的距离来决定它们是否是异常。

```py
# Usinsg Local Outlier Factor
from sklearn.neighbors import LocalOutlierFactor
model = LocalOutlierFactor(n_neighbors=2)
y_pred = model.fit_predict(df[['Annual Income', 'Spending Score']])
df['class'] = y_pred

# Plotting the data
fig, ax = plt.subplots(figsize=(12, 6))

sns.scatterplot('Annual Income', 'Spending Score', data=df, hue='class', ax=ax, cmap='tab10')

plt.tight_layout()
plt.show()
```

```py
/usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning
```

![image+9.png](img/e4c215bb8f4f22300db55bcc699cab0b.png)

如我们所料，不同的异常检测方法产生了不同的结果。如果我们测试了更多的算法，同样的事情也会发生。如前所述，每种算法使用不同的方法来识别异常值，理解这些算法对于获得更好的结果至关重要。

## 最后

本文旨在快速介绍无监督机器学习的有用性和强大功能。我们探讨了有监督的和无监督的机器学习之间的区别，并且我们还浏览了一些重要的用例、算法和无监督学习的例子。

如果你有兴趣了解关于这个主题的更多信息， [Dataquest 的数据科学家学习路径](https://www.dataquest.io/path/data-scientist/)是深入监督和非监督机器学习的一个很好的选择。一定要去看看！