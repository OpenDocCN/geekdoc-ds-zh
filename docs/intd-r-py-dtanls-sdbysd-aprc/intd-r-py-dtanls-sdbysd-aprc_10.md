# 第七章 分类数据

> 原文：[`randpythonbook.netlify.app/categorical-data`](https://randpythonbook.netlify.app/categorical-data)

虽然统计学家可能将数据描述为分类或数值，但这种分类与按其在程序中的*类型*对数据进行分类不同。所以，严格来说，如果你有分类数据，你并不一定有义务在你的脚本中使用任何特定的类型来表示它。

然而，有一些类型是专门设计用来与分类数据一起使用的，因此如果你有机会，使用它们特别有利。我们将在本章中描述其中的一些。

## 7.1 R 中的`factor`函数

在 R 中，分类数据通常存储在`factor`变量中。`factor`比整数的`vector`更特殊，因为

+   它们有一个`levels`属性，它由每个响应可能的所有可能值组成；

+   它们可能是有序的，也可能不是，这也会控制它们在数学函数中的使用方式；

+   它们可能有一个`contrasts`属性，这将控制它们在统计建模函数中的使用方式。

这里有一个第一个例子。假设我们问三个人他们最喜欢的季节是什么。数据可能看起来像这样。

```py
allSeasons <-  c("spring", "summer", "autumn", "winter")
responses <-  factor(c("autumn", "summer", "summer"), 
 levels = allSeasons)
levels(responses)
## [1] "spring" "summer" "autumn" "winter"
is.factor(responses)
## [1] TRUE
is.ordered(responses)
## [1] FALSE
#contrasts(responses) 
# ^ controls how factor is used in different  functions
```

`factor`总是有级别，这是每个观测值可以采取的所有可能唯一值的集合。

如果你没有直接指定它们，你应该小心。当你使用默认选项并将上述代码中的第二个赋值替换为`responses <- factor(c("autumn", "summer", "summer"))`时会发生什么？`factor()`函数的文档会告诉你，默认情况下，`factor()`将只取数据中找到的唯一值。在这种情况下，没有人更喜欢冬天或春天，因此它们都不会出现在`levels(responses)`中。这可能是你想要的，也可能不是。

`factor`可以是有序的，也可以是无序的。有序的`factor`用于*有序*数据。有序数据是一种特殊的分类数据，它承认类别具有自然顺序（例如，低/中/高，而不是红/绿/蓝）。

作为另一个例子，假设我们问十个人他们有多喜欢统计计算，他们只能回答“爱它”、“还可以”或“讨厌它”。数据可能看起来像这样。

```py
ordFeelOptions <-  c("hate it", "it's okay", "love it")
responses <-  factor(c("love it", "it's okay", "love it", 
 "love it", "it's okay", "love it", 
 "love it", "love it", "it's okay", 
 "it's okay"), 
 levels = ordFeelOptions,
 ordered = TRUE)
levels(responses)
## [1] "hate it"   "it's okay" "love it"
is.factor(responses)
## [1] TRUE
is.ordered(responses)
## [1] TRUE
# contrasts(responses)
```

当使用`factor()`创建有序因子时，请注意，当你将其插入`factor()`时，`levels=`参数被认为是有序的。在上面的例子中，如果你指定了`levels = c("love it", "it's okay", "hate it")`，那么因子将假设`love it < it's okay < hate it`，这可能是你想要的，也可能不是。

最后，`factor`可能有一个`contrast`属性。你可以使用`contrasts()`函数获取或设置这个属性。这将影响你使用的数据估计统计模型的一些函数。

我不会讨论本文中对比的具体细节，但整体动机很重要。简而言之，使用`factor`s 的主要原因是它们被设计成可以控制如何对分类数据进行建模。更具体地说，改变`factor`的属性可以控制你正在估计的模型的参数化。如果你正在使用特定函数进行分类数据的建模，你需要知道它如何处理`factors`。另一方面，如果你正在编写一个执行分类数据建模的函数，你应该知道如何处理`factors`。

在你的学习中可能会遇到以下两个例子。

1.  考虑使用`factor`s 作为执行线性回归的函数的输入。在线性回归模型中，如果你有分类输入，有许多选择来写下模型。在每个模型中，参数集合将意味着不同的事情。在 R 中，你可能会通过以特定方式创建`factor`来选择模型。

1.  假设你对估计分类模型感兴趣。在这种情况下，*因变量*是分类的，而不是自变量。在这些类型的模型中，选择你的`factor`是否有序是至关重要的。这些选项将估计完全不同的模型，所以请明智地选择！

这些例子的数学细节超出了本文的范围。如果你在回归课程中没有学习过虚拟变量，或者如果你没有考虑过多项逻辑回归和有序逻辑回归之间的区别，或者你考虑过但只是有点生疏，那完全没问题。我只是提到这些作为`factor`类型可以触发特殊行为的例子。

除了使用`factor()`创建一个之外，还有两种常见的结束方式可以得到`factors`：

1.  从数值数据创建`factors`，以及

1.  当读取外部数据文件时，某一列被强制转换为`factor`。

这里是一个（1）的例子。我们可以将非分类数据`cut()`成分类数据。

```py
stockReturns <-  rnorm(6) # not categorical here
typeOfDay <-  cut(stockReturns, breaks = c(-Inf, 0, Inf)) 
typeOfDay
## [1] (-Inf,0] (0, Inf] (-Inf,0] (-Inf,0] (0, Inf] (-Inf,0]
## Levels: (-Inf,0] (0, Inf]
levels(typeOfDay)
## [1] "(-Inf,0]" "(0, Inf]"
is.factor(typeOfDay)
## [1] TRUE
is.ordered(typeOfDay)
## [1] FALSE
```

最后，注意不同函数读取外部数据集的方式。当读取外部文件时，如果某个函数遇到包含字符的列，它将需要决定是否将该列存储为字符向量，还是作为`factor`。例如，`read.csv()`和`read.table()`有一个`stringsAsFactors=`参数，你应该注意这一点。

## 7.2 Pandas 中分类数据的两种选项

Pandas 提供了两种存储分类数据的方法。它们都与 R 的`factor`s 非常相似。你可以使用其中任何一个

1.  一个具有特殊`dtype`的 Pandas `Series`，或者

1.  一个 Pandas `Categorical`容器。

Pandas 的`Series`在 3.2 和 3.4 节中已经讨论过。这些是强制每个元素共享相同`dtype`的容器。在这里，我们在`pd.Series()`中指定`dtype="category"`。

```py
import pandas as pd
szn_s = pd.Series(["autumn", "summer", "summer"], dtype = "category") 
szn_s.cat.categories
## Index(['autumn', 'summer'], dtype='object')
szn_s.cat.ordered
## False
szn_s.dtype
## CategoricalDtype(categories=['autumn', 'summer'], ordered=False)
type(szn_s)
## <class 'pandas.core.series.Series'>
```

第二个选项是使用 Pandas 的`Categorical`容器。它们非常相似，所以选择是微妙的。就像`Series`容器一样，它们也强制所有元素共享相同的`dtype`。

```py
szn_c = pd.Categorical(["autumn", "summer", "summer"])
szn_c.categories
## Index(['autumn', 'summer'], dtype='object')
szn_c.ordered
## False
szn_c.dtype
## CategoricalDtype(categories=['autumn', 'summer'], ordered=False)
type(szn_c)
## <class 'pandas.core.arrays.categorical.Categorical'>
```

你可能已经注意到，在使用`Categorical`容器时，方法和数据成员不是通过`.cat`访问器访问的。它也与 R 的`factor`类似，因为你可以指定更多的参数在构造函数中。

```py
all_szns = ["spring","summer", "autumn", "winter"]
szn_c2 = pd.Categorical(["autumn", "summer", "summer"], 
 categories = all_szns, 
 ordered = False)
```

在 Pandas 中，就像在 R 中一样，你需要非常小心地处理`categories`（即`levels`）。如果你使用有序数据，它们需要按照正确的顺序指定。如果你使用小数据集，要注意是否所有类别都出现在数据中——否则它们将不会被正确推断。

在 Pandas 的`Series`中，指定非默认的`dtype`更困难。一个选项是在对象创建后更改它们。

```py
szn_s = szn_s.cat.set_categories(
 ["autumn", "summer","spring","winter"])
szn_s.cat.categories
## Index(['autumn', 'summer', 'spring', 'winter'], dtype='object')
szn_s = szn_s.cat.remove_categories(['spring','winter'])
szn_s.cat.categories
## Index(['autumn', 'summer'], dtype='object')
szn_s = szn_s.cat.add_categories(["fall", "winter"])
szn_s.cat.categories
## Index(['autumn', 'summer', 'fall', 'winter'], dtype='object')
```

另一个选项是在创建`Series`之前创建`dtype`，并将其传递给`pd.Series()`。

```py
cat_type = pd.CategoricalDtype(
 categories=["autumn", "summer", "spring", "winter"],
 ordered=True)
responses = pd.Series(
 ["autumn", "summer", "summer"], 
 dtype = cat_type)
responses
## 0    autumn
## 1    summer
## 2    summer
## dtype: category
## Categories (4, object): ['autumn' < 'summer' < 'spring' < 'winter']
```

就像在 R 中一样，你可以将数值数据转换为分类数据。该函数甚至与 R 中的名称相同：`pd.cut()`([`pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html#series-creation`](https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html#series-creation))。根据输入类型，它将返回一个`Series`或`Categorical`([`pandas.pydata.org/docs/reference/api/pandas.cut.html`](https://pandas.pydata.org/docs/reference/api/pandas.cut.html))。

```py
import numpy as np
stock_returns = np.random.normal(size=10) # not categorical 
# array input means Categorical output
type_of_day = pd.cut(stock_returns, 
 bins = [-np.inf, 0, np.inf], 
 labels = ['bad day', 'good day']) 
type(type_of_day)
# Series in means Series out
## <class 'pandas.core.arrays.categorical.Categorical'>
type_of_day2 = pd.cut(pd.Series(stock_returns), 
 bins = [-np.inf, 0, np.inf], 
 labels = ['bad day', 'good day']) 
type(type_of_day2)
## <class 'pandas.core.series.Series'>
```

最后，当从外部源读取数据时，仔细选择是否要将字符数据存储为字符串类型，还是分类类型。在这里，我们使用`pd.read_csv()`([`pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html))读取 Fisher 的 Iris 数据集（Fisher 1988)，由(Dua 和 Graff 2017)提供。有关 Pandas 的`DataFrames`的更多信息，请参阅下一章。

```py
import numpy as np
# make 5th col categorical
my_data = pd.read_csv("data/iris.csv", header=None, 
 dtype = {4:"category"}) 
my_data.head(1)
##      0    1    2    3            4
## 0  5.1  3.5  1.4  0.2  Iris-setosa
my_data.dtypes
## 0     float64
## 1     float64
## 2     float64
## 3     float64
## 4    category
## dtype: object
np.unique(my_data[4]).tolist()
## ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
```

## 7.3 练习

### 7.3.1 R 问题

使用以下代码读取这个棋盘数据集（“Chess (King-Rook vs. King-Pawn)” 1989)，由(Dua 和 Graff 2017)提供。你可能需要更改工作目录，但如果你这样做，确保在将脚本提交给我之前注释掉那段代码。

```py
d <-  read.csv("kr-vs-kp.data", header=FALSE, stringsAsFactors = TRUE)
head(d)
```

1.  所有列都是`factor`类型吗？将`TRUE`或`FALSE`赋值给`allFactors`。

1.  这些`factor`中的任何一个应该是有序的吗？将`TRUE`或`FALSE`赋值给`ideallyOrdered`。提示：从[`archive.ics.uci.edu/ml/datasets/Chess+%28King-Rook+vs.+King-Pawn%29`](https://archive.ics.uci.edu/ml/datasets/Chess+%28King-Rook+vs.+King-Pawn%29)读取数据集描述。

1.  这些因素目前是否有序？将`TRUE`或`FALSE`赋值给`currentlyOrdered`。

1.  第一列等于 `'f'` 的百分比是多少（介于 $0$ 和 $100$% 之间）？将你的答案分配给 `percentF`。

假设你有一个以下 `vector`。请确保将此代码包含在你的脚本中。

```py
normSamps <-  rnorm(100)
```

1.  从 `normSamps` 中创建一个 `factor`。将每个元素映射到 `"within 1 sd"` 或 `"outside 1 sd"`，具体取决于元素是否在 $0$ 的 $1$ 理论标准差内。将 `factor` 命名为 `withinOrNot`。

### 7.3.2 Python 问题

考虑以下两个学生的模拟成绩数据：

```py
import pandas as pd
import numpy as np
poss_grades = ['A+','A','A-','B+','B','B-',
 'C+','C','C-','D+','D','D-',
 'F']
grade_values = {'A+':4.0,'A':4.0,'A-':3.7,'B+':3.3,'B':3.0,'B-':2.7,
 'C+':2.3,'C':2.0,'C-':1.7,'D+':1.3,'D':1.0,'D-':.67,
 'F':0.0}
student1 = np.random.choice(poss_grades, size = 10, replace = True)
student2 = np.random.choice(poss_grades, size = 12, replace = True)
```

1.  将这两个 Numpy 数组转换为教科书讨论的 Pandas 类型的分类数据。将这两个变量命名为 `s1` 和 `s2`。

1.  这些数据是分类数据。它们是有序的吗？确保相应地调整 `s1` 和 `s2`。

1.  计算两个学生的 GPA。将浮点数分配给名为 `s1_gpa` 和 `s2_gpa` 的变量。使用 `grade_values` 将每个字母成绩转换为数字，然后使用等权重将每个学生的所有数字平均在一起。

1.  每个类别是否等距分布？如果是，那么这些数据被称为 *等距数据*。你对这个问题的回答是否会影响将任何有序数据平均起来的合法性？将你的回答分配给变量 `ave_ord_data_response`。提示：考虑（任何）两个不同的数据集，它们恰好产生了相同的 GPA。这两个 GPA 的相等性是否具有误导性？

1.  计算每个学生的平均成绩。将你的答案作为 `str` 分配给变量 `s1_mode` 和 `s2_mode`。如果有多个平均成绩，则分配字母顺序中第一个的平均成绩。

假设你正在创建一个 *分类器*，其任务是预测标签。考虑以下预测标签与相应实际标签的 `DataFrame`。请确保将此代码包含在你的脚本中。

```py
import pandas as pd
import numpy as np
d = pd.DataFrame({'predicted label' : [1,2,2,1,2,2,1,2,3,2,2,3], 
 'actual label': [1,2,3,1,2,3,1,2,3,1,2,3]}, 
 dtype='category')
d.dtypes[0]
## CategoricalDtype(categories=[1, 2, 3], ordered=False)
d.dtypes[1]
## CategoricalDtype(categories=[1, 2, 3], ordered=False)
```

1.  将预测准确率（介于 $0$ 和 $100$% 之间）分配给变量 `perc_acc`。

1.  创建一个 *混淆矩阵* 以更好地评估分类器在哪些标签上难以处理。这应该是一个 $3 \times 3$ 的 Numpy `ndarray` 百分比。行将对应于预测标签，列将对应于实际标签，而位置 $(0,2)$ 中的数字，例如，将是模型预测标签 `1` 而实际标签为 `3` 的观察值的百分比。将变量命名为 `confusion`。

### 参考文献

“象棋（王后-车对王-兵）。” 1989\. UCI 机器学习库。

Dua, Dheeru 和 Casey Graff. 2017\. “UCI 机器学习库。”加州大学欧文分校，信息与计算机科学学院。[`archive.ics.uci.edu/ml`](http://archive.ics.uci.edu/ml)。

Fisher, Test, R.A. & Creator. 1988\. “Iris.” UCI 机器学习库。
