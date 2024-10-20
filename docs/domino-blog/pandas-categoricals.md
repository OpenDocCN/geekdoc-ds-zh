# 熊猫种类

> 原文：<https://www.dominodatalab.com/blog/pandas-categoricals>

免责声明:类别是由熊猫开发团队而不是我创建的。

## 速度比并行性更重要

我通常写的是平行。因此，人们问我如何并行化他们缓慢的计算。答案通常是用更好的方式使用熊猫

*   问:如何用并行技术让我的熊猫编码更快？
*   *答:不需要并行，用熊猫更好*

这几乎总是比使用多核或多台机器更简单、更有效。只有在对存储格式、压缩、数据表示等做出明智的选择之后，才应该考虑并行性..

今天我们将讨论熊猫如何用数字表示分类文本数据。这是一个廉价且未被充分利用的技巧，可以使普通查询的速度提高一个数量级。

## 绝对的

我们的数据通常包含具有许多重复元素的文本列。示例:

*   股票代码–谷歌、APPL、MSFT、...
*   性别–女性，男性，...
*   实验结果——健康、患病、无变化，...
*   州——加利福尼亚州、德克萨斯州、纽约州、...

我们通常将这些表示为文本。Pandas 用包含普通 Python 字符串的对象 dtype 表示文本。这是导致代码缓慢的一个常见原因，因为对象数据类型是以 Python 的速度运行的，而不是以熊猫正常的 C 速度。

Pandas categoricals 是一个新的强大的特性，它用数字编码分类数据，这样我们就可以在这种文本数据上利用 Pandas 的快速 C 代码。

```py
>>> # Example dataframe with names, balances, and genders as object dtypes

>>> df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie', 'Danielle'],

... 'balance': [100.0, 200.0, 300.0, 400.0],

... 'gender': ['Female', 'Male', 'Male', 'Female']},

... columns=['name', 'balance', 'gender'])
```

```py
>>> df.dtypes # Oh no! Slow object dtypes!
name object

balance float64

gender object

dtype: object
```

通过使用类别，我们可以更有效地表示具有许多重复的列，比如性别。这将我们的原始数据存储为两部分

*   原始资料

```py
Female, Male, Male, Female
```

**1。**将每个类别的索引映射到一个整数

```py
Female: 0
Male: 1
...
```

**2。**正常的整数数组

```py
0, 1, 1, 0
```

这个整数数组更加紧凑，现在是一个普通的 C 数组。这允许在以前较慢的对象数据类型列上正常的 C 速度。对列进行分类很容易:

```py
df['gender'] = df['gender'].astype('category')

# Categorize!
```

让我们看看结果

```py
df 
# DataFrame looks the same
```

```py
name balance gender
0 Alice 100 Female

1 Bob 200 Male

2 Charlie 300 Male

3 Danielle 400 Female
```

```py
df.dtypes 
# But dtypes have changed
```

```py
name object
balance float64
gender category
dtype: object
```

```py
df.gender # Note Categories at the bottom
```

```py
0 Female

1 Male

2 Male

3 Female

Name: gender, dtype: category

Categories (2, object): [Female, Male]
```

```py
df.gender.cat.categories # Category index
```

```py
Index([u'Female', u'Male'], dtype='object')
```

```py
df.gender.cat.codes # Numerical values
```

```py
0 0

1 1

2 1

3 0

dtype: int8 # Stored in single bytes!
```

请注意，我们可以将我们的性别更紧凑地存储为单个字节。我们可以继续添加性别(不止两个)，熊猫会在必要时使用新的值(2，3，…)。

我们的数据帧看起来和感觉起来都和以前一样。熊猫内部将平滑用户体验，这样你就不会注意到你实际上是在使用一个紧凑的整数数组。

## 表演

让我们看一个稍微大一点的例子来看看性能差异。

我们从纽约市的出租车数据集中抽取一小部分，并根据 medallion ID 进行分组，以找到在某一段时间内行驶距离最长的出租车司机。

```py
import pandas as pd

df = pd.read_csv('trip_data_1_00.csv')

%time 
df.groupby(df.medallion).trip_distance.sum().sort(ascending=False,inplace=False).head()
```

```py
CPU times: user 161 ms, sys: 0 ns, total: 161 ms
Wall time: 175 ms
```

```py
medallion
1E76B5DCA3A19D03B0FB39BCF2A2F534 870.83
6945300E90C69061B463CCDA370DE5D6 832.91

4F4BEA1914E323156BE0B24EF8205B73 811.99

191115180C29B1E2AF8BE0FD0ABD138F 787.33

B83044D63E9421B76011917CE280C137 782.78

Name: trip_distance, dtype: float64
```

大约需要 170 毫秒。我们差不多同时归类。

```py
%time df['medallion'] = df['medallion'].astype('category')
```

```py
CPU times: user 168 ms, sys: 12.1 ms, total: 180 ms
Wall time: 197 ms
```

现在我们有了数字类别，我们的计算运行 20 毫秒，提高了大约一个数量级。

```py
%time

df.groupby(df.medallion).trip_distance.sum().sort(ascending=False,
inplace=False).head()
```

```py
CPU times: user 16.4 ms, sys: 3.89 ms, total: 20.3 ms
Wall time: 20.3 ms
```

```py
medallion

1E76B5DCA3A19D03B0FB39BCF2A2F534 870.83

6945300E90C69061B463CCDA370DE5D6 832.91

4F4BEA1914E323156BE0B24EF8205B73 811.99

191115180C29B1E2AF8BE0FD0ABD138F 787.33

B83044D63E9421B76011917CE280C137 782.78

Name: trip_distance, dtype: float64
```

在我们完成用类别替换对象数据类型的一次性操作后，我们看到速度几乎提高了一个数量级。该列上的大多数其他计算也同样快。我们的内存使用也急剧下降。

## 结论

熊猫分类有效地编码重复的文本数据。类别对于股票代码、性别、实验结果、城市、州等数据非常有用..类别易于使用，并极大地提高了此类数据的性能。

当处理不方便的大数据或慢数据时，我们有几种选择来提高性能。在存储格式、压缩、列布局和数据表示方面的良好选择可以极大地改善查询时间和内存使用。这些选择中的每一个都和并行性一样重要，但是没有被过分夸大，所以经常被忽视。

跟随 [@mrocklin](https://twitter.com/mrocklin) 并访问 [continuum.io](http://continuum.io/)