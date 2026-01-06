# 第八章 数据框

> 原文：[`randpythonbook.netlify.app/data-frames`](https://randpythonbook.netlify.app/data-frames)

当人们听到“数据”这个词时，很多人会想到信息矩形数组（例如 Excel 工作表）。每一列包含共享数据类型的元素，而这些数据类型可以按列变化。

在 R 和 Python 中都有这种类型的对应：数据框。这甚至可能是 R 和 Python 程序中存储数据最常见的方式，因为许多从外部源读取数据的函数返回的对象都是这种类型（例如 R 中的 `read.csv()` 和 Python 中的 `pd.read_csv()`）。

R 和 Python 的数据框有许多共同之处：

1.  每一列的长度必须与其他所有列相同，

1.  每一列的元素都将具有相同的类型。

1.  任何行的元素可以有不同的类型，

1.  列和行可以用不同的方式命名，

1.  有许多方法可以获取和设置数据的不同子集，并且

1.  在读取数据时，两种语言都会遇到类似的问题。

## 8.1 R 中的数据框

让我们以费舍尔的“鸢尾花”数据集为例（Fisher 1988），由（Dua 和 Graff 2017）提供。我们将从这个逗号分隔的文件中读取这个数据集（有关输入/输出的更多信息，请参阅第九章）。此文件可以从以下链接下载：[`archive.ics.uci.edu/ml/datasets/iris`](https://archive.ics.uci.edu/ml/datasets/iris)。

```py
irisData <-  read.csv("data/iris.csv", header = F)
head(irisData, 3)
##    V1  V2  V3  V4          V5
## 1 5.1 3.5 1.4 0.2 Iris-setosa
## 2 4.9 3.0 1.4 0.2 Iris-setosa
## 3 4.7 3.2 1.3 0.2 Iris-setosa
typeof(irisData)
## [1] "list"
class(irisData) # we'll talk more about classes later
## [1] "data.frame"
dim(irisData)
## [1] 150   5
nrow(irisData)
## [1] 150
ncol(irisData)
## [1] 5
```

虽然有一些例外，但大多数数据集都可以存储为 `data.frame`。这类二维数据集相当常见。任何特定的行通常是对一个实验单元（例如人、地点或事物）的观察。查看特定的列会给你一个为所有观察存储的测量值。

不要依赖 `read.csv()` 或 `read.table()` 的默认参数！在读取数据框后，始终检查以确保

1.  列数是正确的，因为使用了正确的列 *分隔符*（例如 `sep=`），

1.  如果原始文本文件中有一些列名，则列名被正确解析，

1.  如果文本文件中没有列名，数据的第一行没有被用作列名序列，

1.  最后几行没有读取空格

1.  字符列被正确读取（例如 `stringsAsFactors=`），并且

1.  表示缺失数据的特殊字符被正确识别（例如 `na.strings=`）。

[`data.frame`是`list`的特殊情况](https://cran.r-project.org/doc/manuals/r-release/R-lang.html#Data-frame-objects)。列表的每个元素都是一列。列可以是`vector`s 或`factor`s，并且它们可以具有不同的类型。这是数据框和`matrix`之间最大的区别之一。它们都是二维的，但`matrix`需要所有元素都是同一类型。与一般的`list`不同，`data.frame`要求所有列具有相同数量的元素。换句话说，`data.frame`不是一个“*杂乱*”的列表。

通常情况下，你需要从`data.frame`中提取信息片段。这可以通过多种方式完成。如果列有名称，你可以使用`$`运算符来访问单个列。访问单个列之后，可能会创建一个新的向量。你也可以使用`[`运算符通过名称访问多个列。

```py
colnames(irisData) <-  c("sepal.length", "sepal.width", 
 "petal.length","petal.width", 
 "species")
firstCol <-  irisData$sepal.length
head(firstCol)
## [1] 5.1 4.9 4.7 4.6 5.0 5.4
firstTwoCols <-  irisData[c("sepal.length", "sepal.width")] 
head(firstTwoCols, 3)
##   sepal.length sepal.width
## 1          5.1         3.5
## 2          4.9         3.0
## 3          4.7         3.2
```

`[`运算符也用于通过索引数字或某些逻辑标准选择行和列。

```py
topLeft <-  irisData[1,1] # first row, first col
topLeft
## [1] 5.1
firstThreeRows <-  irisData[1:3,] # rows 1-3, all cols
firstThreeRows
##   sepal.length sepal.width petal.length petal.width     species
## 1          5.1         3.5          1.4         0.2 Iris-setosa
## 2          4.9         3.0          1.4         0.2 Iris-setosa
## 3          4.7         3.2          1.3         0.2 Iris-setosa
# rows where species column is setosa
setosaOnly <-  irisData[irisData$species == "Iris-setosa",] 
setosaOnly[1:3,-1]
##   sepal.width petal.length petal.width     species
## 1         3.5          1.4         0.2 Iris-setosa
## 2         3.0          1.4         0.2 Iris-setosa
## 3         3.2          1.3         0.2 Iris-setosa
```

在上面的代码中，`irisData$species == "Iris-setosa"`使用向量化`==`运算符创建了一个逻辑向量（试试看！）。“[`运算符选择逻辑向量对应元素为`TRUE`的行。

注意：根据你如何使用方括号，你可能会得到一个`data.frame`或一个`vector`。例如，尝试`class(irisData[,1])`和`class(irisData[,c(1,2)])`。

在 R 中，`data.frame`s 可能有行名。你可以使用`rownames()`函数获取和设置这个字符`vector`。你可以使用方括号运算符通过名称访问行。

```py
head(rownames(irisData))
## [1] "1" "2" "3" "4" "5" "6"
rownames(irisData) <-  as.numeric(rownames(irisData)) +  1000
head(rownames(irisData))
## [1] "1001" "1002" "1003" "1004" "1005" "1006"
irisData["1002",]
##      sepal.length sepal.width petal.length petal.width     species
## 1002          4.9           3          1.4         0.2 Iris-setosa
```

修改数据的代码通常看起来与提取数据的代码非常相似。你会注意到很多相同的符号（例如`$`，`[`等），但`(`<-`)`会指向另一个方向。

```py
irisData$columnOfOnes <-  rep(1, nrow(irisData))
irisData[,1] <-  NULL #delete first col
irisData[1:2,1] <-  rnorm(n = 2, mean = 999)
irisData[,'sepal.width'] <-  rnorm(n = nrow(irisData), mean = -999)
irisData[irisData$species == "Iris-setosa", 'species'] <- "SETOSA!"
head(irisData, 3)
##      sepal.width petal.length petal.width species columnOfOnes
## 1001   -998.9036          1.4         0.2 SETOSA!            1
## 1002   -997.3385          1.4         0.2 SETOSA!            1
## 1003   -999.6752          1.3         0.2 SETOSA!            1
```

## 8.2 Python 中的数据框

Python 中的 Pandas 库中的数据框是模仿 R 的（McKinney 2017）。

```py
import pandas as pd
iris_data = pd.read_csv("data/iris.csv", header = None)
iris_data.head(3)
##      0    1    2    3            4
## 0  5.1  3.5  1.4  0.2  Iris-setosa
## 1  4.9  3.0  1.4  0.2  Iris-setosa
## 2  4.7  3.2  1.3  0.2  Iris-setosa
iris_data.shape
## (150, 5)
len(iris_data) # num rows
## 150
len(iris_data.columns) # num columns
## 5
list(iris_data.dtypes)[:3]
## [dtype('float64'), dtype('float64'), dtype('float64')]
list(iris_data.dtypes)[3:]
## [dtype('float64'), dtype('O')]
```

其结构与 R 的数据框非常相似。它是二维的，你可以通过名称或数字[访问列和行](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html)。每一列都是一个`Series`对象，每一列可以有不同的`dtype`，这与 R 的情况类似。再次强调，因为元素只需要在列中是同一类型，这是 2 维 Numpy `ndarray`s 和`DataFrame`s 之间的一大区别（参看 R 的`matrix`与 R 的`data.frame`）。

再次强调，不要依赖`pd.read_csv()`的默认参数！在读取数据集之后，始终要检查

1.  列数是正确的，因为使用了正确的列*分隔符*（参看`sep=`），

1.  如果原始文本文件中有列名，则列名被正确解析，

1.  如果文本文件中没有列名，则数据的第一行不会被用作列名序列（参看`header=`），并且

1.  最后几行没有读取空格

1.  字符列被正确读取（参看`dtype=`），并且

1.  表示缺失数据的特殊字符被正确识别（例如 `na.values=`）。

在 Python 中，方括号的使用与 R 略有不同。就像在 R 中一样，你可以使用方括号通过名称访问列，你也可以访问行。然而，与 R 不同的是，你不需要每次使用方括号时都指定行和列。

```py
iris_data.columns = ["sepal.length", "sepal.width", "petal.length", 
 "petal.width", "species"]
first_col = iris_data['sepal.length']
first_col.head()
## 0    5.1
## 1    4.9
## 2    4.7
## 3    4.6
## 4    5.0
## Name: sepal.length, dtype: float64
first_two_cols = iris_data[["sepal.length", "sepal.width"]]
first_two_cols.head(3)
##    sepal.length  sepal.width
## 0           5.1          3.5
## 1           4.9          3.0
## 2           4.7          3.2
```

注意，`iris_data['sepal.length']` 返回一个 `Series`，而 `iris_data[["sepal.length", "sepal.width"]]` 返回一个 Pandas `DataFrame`。这种行为与 R 中发生的情况类似。有关更多详细信息，请点击 [这里](https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html#indexing-selection)。

你可以使用 [`.iloc` 方法](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html) 通过数字选择列和行。`iloc` 可能代表“整数位置”。

```py
# specify rows/cols by number
top_left = iris_data.iloc[0,0]
top_left
## 5.1
first_three_rows_without_last_col = iris_data.iloc[:3,:-1]
first_three_rows_without_last_col
##    sepal.length  sepal.width  petal.length  petal.width
## 0           5.1          3.5           1.4          0.2
## 1           4.9          3.0           1.4          0.2
## 2           4.7          3.2           1.3          0.2
```

除了整数编号之外，你可以使用 [`.loc()` 方法](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html) 来选择列。通常，你应该更喜欢这种方法来访问列，因为通过名称而不是编号访问事物更易于阅读。以下是一些示例。

```py
sepal_w_to_pedal_w = iris_data.loc[:,'sepal.width':'petal.width']
sepal_w_to_pedal_w.head()
##    sepal.width  petal.length  petal.width
## 0          3.5           1.4          0.2
## 1          3.0           1.4          0.2
## 2          3.2           1.3          0.2
## 3          3.1           1.5          0.2
## 4          3.6           1.4          0.2
setosa_only = iris_data.loc[iris_data['species'] == "Iris-setosa",]
# don't need the redundant column anymore
del setosa_only['species']
setosa_only.head(3)
##    sepal.length  sepal.width  petal.length  petal.width
## 0           5.1          3.5           1.4          0.2
## 1           4.9          3.0           1.4          0.2
## 2           4.7          3.2           1.3          0.2
```

注意我们使用了一个 `slice`（即 `'sepal.width':'pedal.width'`）来通过只引用最左端和最右端来访问多个列。与数字切片不同，[右端是包含在内的](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#slicing-ranges)。此外，请注意，这不能与常规方括号运算符（即 `iris_data['sepal.width':'pedal.width']`）一起使用。第二个示例过滤掉了 `"species"` 列元素等于 `"Iris-setosa"` 的行。

Pandas 中的每个 `DataFrame` 都有一个 `.index` 属性。这类似于 R 中的行名，但它更加灵活，因为索引可以采用多种类型。这有助于我们突出 `.loc` 和 `.iloc` 之间的区别。回想一下，`.loc` 是基于标签的选择。标签不一定是字符串。考虑以下示例。

```py
iris_data.index
# reverse the index
## RangeIndex(start=0, stop=150, step=1)
iris_data = iris_data.set_index(iris_data.index[::-1]) 
iris_data.iloc[-2:,:3] # top is now bottom
##    sepal.length  sepal.width  petal.length
## 1           6.2          3.4           5.4
## 0           5.9          3.0           5.1
iris_data.loc[0] # last row has 0 index
## sepal.length               5.9
## sepal.width                  3
## petal.length               5.1
## petal.width                1.8
## species         Iris-virginica
## Name: 0, dtype: object
iris_data.iloc[0] # first row with big index 
## sepal.length            5.1
## sepal.width             3.5
## petal.length            1.4
## petal.width             0.2
## species         Iris-setosa
## Name: 149, dtype: object
```

`iris_data.loc[0]` 选择第 `0` 个索引。第二行反转了索引，所以这实际上是最后一行。如果你想选择第一行，请使用 `iris_data.iloc[0]`。

在数据帧内部修改数据看起来与提取数据非常相似。你会认出前面提到的大多数方法。

```py
import numpy as np
n_rows = iris_data.shape[0]
iris_data['col_ones'] = np.repeat(1.0, n_rows)
iris_data.iloc[:2,0] =  np.random.normal(loc=999, size=2)
rand_nums = np.random.normal(loc=-999, size=n_rows)
iris_data.loc[:,'sepal.width'] = rand_nums
setosa_rows = iris_data['species'] == "Iris-setosa"
iris_data.loc[setosa_rows, 'species'] = "SETOSA!"
del iris_data['petal.length']
iris_data.head(3)
##      sepal.length  sepal.width  petal.width  species  col_ones
## 149    999.146739  -998.446586          0.2  SETOSA!       1.0
## 148    998.005821  -999.015224          0.2  SETOSA!       1.0
## 147      4.700000  -999.803985          0.2  SETOSA!       1.0
```

你还可以使用 [`.assign()` 方法](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.assign.html) 来创建新列。此方法不会就地修改数据帧。它返回一个新的包含额外列的 `DataFrame`。

```py
iris_data = iris_data.assign(new_col_name = np.arange(n_rows))
del iris_data['sepal.length']
iris_data.head(3)
##      sepal.width  petal.width  species  col_ones  new_col_name
## 149  -998.446586          0.2  SETOSA!       1.0             0
## 148  -999.015224          0.2  SETOSA!       1.0             1
## 147  -999.803985          0.2  SETOSA!       1.0             2
```

在上面，我们将 Numpy 数组分配给 `DataFrame` 的列。当你分配 `Series` 对象时要小心。你会在文档中看到，“Pandas 在从 `.loc` 和 `.iloc` 设置 Series 和 `DataFrame` 时对所有的轴进行对齐。”

## 8.3 练习

### 8.3.1 R 问题

考虑数据集 `"original_rt_snippets.txt"`（Socher 等人 2013），由 (Dua 和 Graff 2017) 提供。我们将计算这个数据集上的 *词频-逆文档频率统计*（Jones 1972），这是文本挖掘和自然语言处理中常用的数据转换技术。如果你愿意，可以使用 `stringr` 库来回答这个问题。

1.  将这个数据集作为一个 `vector` 读取，并命名为 `corpus`。

1.  创建一个包含以下短语的 `vector`，名为 `dictionary`：“charming”，“fantasy”，“hate”，和“boring”。

1.  构建一个名为 `bagOfWords` 的 `data.frame`，包含四个列，其中包含字典中每个单词出现的次数。匹配精确短语。为了简单起见，无需担心字母的大小写或使用正则表达式（参看第 3.9 节）。用你正在搜索的短语标记这个 `data.frame` 的列。尽量编写易于修改的代码，以防你决定更改字典中的短语集合。

1.  创建一个包含四个列的 `data.frame`，名为 `termFrequency`。每个元素应与上一个 `data.frame` 中的计数相对应。而不是计数，每个元素应该是 $\log(1 + \text{count})$。

1.  创建一个长度为四的 `vector`，名为 `invDocFrequency`。任何术语 $t$ 的逆文档频率公式是 $\log([\text{语料库中的文档数量}])$ 减去 $\log([\text{包含术语 } t \text{ 的文档数量}])$。确保这个 `vector` 的名称与字典中的单词相同。

1.  创建一个名为 `tfidf`（代表“词频-逆文档频率”）的 `data.frame`。对于行/文档 $d$ 和列/术语 $t$，公式是乘积：$[\text{术语 } t \text{ 和文档 } d \text{ 的词频}] \times [\text{术语 } t \text{ 的逆文档频率}]$。

1.  从 `corpus` 中提取具有 `tfidf` 对应行中至少一个非零元素的元素。将这个 `vector` 呼叫 `informativeDocuments`。

1.  你是否看到任何被标记为信息性的文档，但实际上并不包含你搜索的单词？

`mtcars` 是 R 中内置的数据集，因此你不需要读取它。你可以通过输入 `?datasets::mtcars` 来了解更多关于它的信息。

1.  创建一个与 `mtcars` 相同但移除了 `disp` 列的新 `data.frame`，名为 `withoutDisp`。

1.  为 `withoutDisp` 创建一个新列，名为 `coolnessScore`。公式是 $\frac{1}{\text{mpg}} + \text{四分之一英里时间}$。

1.  创建一个名为 `sortedWD` 的新 `data.frame`，它与 `withoutDisp` 相同，但按酷炫分数降序排列。

1.  从 `sortedWD` 创建一个名为 `specialRows` 的新 `data.frame`，只保留满足 `$\text{重量 (1000 磅)} + \text{后轴比} < 7$` 条件的行。

1.  计算从 `sortedWD` 到 `specialRows` 的行数减少百分比，称为 `percReduction`。确保它在 $0$ 到 $100$ 之间。

这个问题调查了**Zillow 房屋价值指数 (ZHVI)**（[`www.zillow.com/research/data/`](https://www.zillow.com/research/data/)）对于单户住宅的情况。

1.  读取 `"Zip_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month.csv"`。将 `data.frame` 命名为 `homeData`。记住要小心文件路径。此外，当使用文本编辑器查看数据集时，请确保“自动换行”未开启。

1.  提取与弗吉尼亚州夏洛茨维尔对应的 `homeData` 的行，并将它们作为 `data.frame` 分配给变量 `cvilleData`。

1.  将所有唯一的邮政编码分配给一个名为 `cvilleZips` 的 `character vector`。

1.  提取 `cvilleData` 中与房屋价格对应的列，并将它们转置，使得结果中的每一行对应不同的月份。将这个新的 `data.frame` 称为 `timeSeriesData`。同时，确保这个新的 `data.frame` 的列名设置为适当的邮政编码。

1.  编写一个名为 `getAveMonthlyChange` 的函数，该函数接受一个数值 `vector` 并返回平均变化。你的函数不应返回 `NA`，因此请确保适当地处理 `NA`。

1.  计算每个邮政编码的平均月价格变化。将你的结果存储在名为 `aveMonthlyChanges` 的 `vector` 中。确保这个 `vector` 有命名元素，以便可以通过邮政编码提取元素。

### 8.3.2 Python 问题

这个问题涉及查看 S&P500 指数的历史价格。这些数据是从 [`finance.yahoo.com`](https://finance.yahoo.com) 下载的（“GSPC 数据” 2021）。它包含从 “2007-01-03” 开始到 “2021-10-01” 的价格。

1.  以 `data.frame` 的形式读取数据文件 `"gspc.csv"` 并将变量命名为 `gspc`。

1.  使用 [`.set_index()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.set_index.html) 将 `gspc` 的索引更改为其 `"Index"` 列。将新的 `DataFrame` 存储为 `gspc_good_idx`。

1.  回忆一下第三章节练习中提供的 *log returns* 公式。在 `gspc_good_idx` 中添加一个名为 `log_returns` 的列。从 `GSPC.Adjusted` 列计算它们。确保将它们按 $100$ 缩放，并在没有回报的第一个元素中放置一个 `np.nan`。

1.  提取 2021 年可用的所有回报，并将它们存储为名为 `this_year_returns` 的 `Series`。

1.  在 `gspc_good_idx` 中添加一个包含 *drawdown* 时间序列的列。称为 `drawdown`。要计算给定日期的 drawdown，从该日期的价格中减去该日期当前的运行最高价格。此计算仅使用调整后的收盘价。

1.  向`gspc_good_idx`添加一个包含百分比回撤时间序列的列。将此列命名为`perc_drawdown`。使用上一列，但将此数字作为对应运行最大值的百分比。

1.  这个时间序列的最大回撤是多少？将其作为百分比存储在值`mdd`中。

在这个问题中，我们将查看[一些关于氡测量的数据](https://www.tensorflow.org/datasets/catalog/radon)（Gelman 和 Hill 2007）。我们不会读取文本文件，而是使用`tensorflow_datasets`模块将数据加载到 Python 中（“TensorFlow Datasets，一组可用的数据集” 2021）。

请在您的提交中包含以下代码。

```py
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
d = tfds.load("radon")
d = pd.DataFrame(tfds.as_dataframe(d['train']))
```

在您能够导入之前，许多人需要安装`tensorflow`和`tensorflow_datasets`。如果是这样，请阅读第 10.2 节以获取有关如何安装包的更多信息。

1.  将与最高记录的氡水平相关的`d`的行分配给`worst_row`。确保它是一个`DataFrame`。

1.  将`nrows`和`ncols`分别分配给`d`的行数和列数。

1.  将最常见的列数据类型分配给`most_common_dtype`。确保变量是`numpy.dtype`类型

1.  在这个数据集中是否有来自弗吉尼亚的观测数据？如果有，将`any_va`分配为`True`。否则分配为`False`。

1.  在`d`中创建一个名为`dist_from_cville`的新列。使用**哈弗辛公式**计算每行与弗吉尼亚大学的距离，单位为公里。

    +   假设弗吉尼亚大学位于北纬 38.0336$^\circ$，西经 78.5080$^\circ$W

    +   假设地球半径 $r = 6378.137$ 公里。

    +   $(\lambda_1, \phi_1)$（以弧度为单位的带符号经度，以弧度为单位的带符号纬度）和$(\lambda_2, \phi_2)$之间的距离公式是$$\begin{equation} 2 \hspace{1mm} r \hspace{1mm} \text{arcsin}\left( \sqrt{ \sin²\left( \frac{\phi_2 - \phi_1}{2}\right) + \cos(\phi_1)\cos(\phi_2) \sin²\left( \frac{\lambda_2 - \lambda_1}{2} \right) } \right) \end{equation}$$

1.  所有在离我们现在最近的地方测量的平均氡测量值是多少？将您的答案作为`float`类型的`close_ave`分配。

### 参考文献

Dua, Dheeru, 和 Casey Graff. 2017\. “UCI 机器学习库.” 加州大学欧文分校，信息学院；计算机科学系。[`archive.ics.uci.edu/ml`](http://archive.ics.uci.edu/ml).

Fisher, Test, R.A. & Creator. 1988\. “Iris.” UCI 机器学习库。

Gelman, Andrew, 和 Jennifer Hill. 2007\. *使用回归和多级/分层模型进行数据分析*. 社会研究分析方法. 剑桥大学出版社。

“GSPC 数据。” 2021\. [`finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC`](https://finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC)。

Jones, Karen Spärck. 1972\. “术语特定性的统计解释及其在检索中的应用.” *文献与信息工作* 28: 11–21.

McKinney, Wes. 2017\. *Python 数据分析：使用 Pandas、Numpy 和 Ipython 进行数据处理*. 2nd ed. O’Reilly Media, Inc.

Socher, Richard, Alex Perelygin, Jean Wu, Jason Chuang, Christopher Manning, Andrew Ng, and Christopher Potts. 2013\. “基于组合向量语法的解析.” In *EMNLP*.

“TensorFlow 数据集，一组可用的数据集.” 2021\. [`www.tensorflow.org/datasets`](https://www.tensorflow.org/datasets).
