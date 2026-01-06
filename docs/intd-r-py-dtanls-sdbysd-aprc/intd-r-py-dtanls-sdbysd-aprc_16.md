# 第十二章 重塑和组合数据集

> 原文：[`randpythonbook.netlify.app/reshaping-and-combining-data-sets`](https://randpythonbook.netlify.app/reshaping-and-combining-data-sets)

## 12.1 数据的排序和排序

按升序排序数据集，例如，是一个常见的任务。你可能需要这样做，因为

1.  排序和排名通常在*非参数统计*中完成，

1.  你想检查数据集中最“极端”的观测值，

1.  它是生成可视化之前的预处理步骤。

在 R 中，一切始于`向量`s。你应该了解两个常见的函数：`sort()`和`order()`。`sort()`返回排序后的*数据*，而`order()`返回*排序索引*。

```py
sillyData <-  rnorm(5)
print(sillyData)
## [1]  0.3903776  0.5796584  1.4929115  0.3704896 -1.3450719
sort(sillyData)
## [1] -1.3450719  0.3704896  0.3903776  0.5796584  1.4929115
order(sillyData)
## [1] 5 4 1 2 3
```

`order()`函数在按特定列对数据框进行排序时非常有用。下面，我们检查了一个示例数据集（“SAS ^ (Viya ^ (Example Data Sets）2021））中前 5 辆最昂贵的汽车。请注意，我们首先需要稍微清理一下`MSRP`（一个`字符向量`）。我们使用`gsub()`函数在文本中查找模式，并用空字符串替换它们。

```py
carData <-  read.csv("data/cars.csv")
noDollarSignMSRP <-  gsub("$", "", carData$MSRP, fixed = TRUE)
carData$cleanMSRP <-  as.numeric(gsub(",", "", noDollarSignMSRP, 
 fixed = TRUE))
rowIndices <-  order(carData$cleanMSRP, decreasing = TRUE)[1:5]
carData[rowIndices,c("Make", "Model", "MSRP", "cleanMSRP")]
```

```py
##              Make                 Model     MSRP cleanMSRP
## 335       Porsche           911 GT2 2dr $192,465    192465
## 263 Mercedes-Benz             CL600 2dr $128,420    128420
## 272 Mercedes-Benz SL600 convertible 2dr $126,670    126670
## 271 Mercedes-Benz          SL55 AMG 2dr $121,770    121770
## 262 Mercedes-Benz             CL500 2dr  $94,820     94820
```

在 Python 中，Numpy 有`np.argsort()`（[`numpy.org/doc/stable/reference/generated/numpy.argsort.html`](https://numpy.org/doc/stable/reference/generated/numpy.argsort.html)）和`np.sort()`（[`numpy.org/doc/stable/reference/generated/numpy.sort.html`](https://numpy.org/doc/stable/reference/generated/numpy.sort.html)）。

```py
import numpy as np
silly_data = np.random.normal(size=5)
print(silly_data)
## [-0.52817175 -1.07296862  0.86540763 -2.3015387   1.74481176]
print( np.sort(silly_data) )
## [-2.3015387  -1.07296862 -0.52817175  0.86540763  1.74481176]
np.argsort(silly_data)
## array([3, 1, 0, 2, 4])
```

对于 Pandas 的`DataFrame`s，我发现大多数有用的函数都是附加到`DataFrame`类的方法。这意味着，只要某物在`DataFrame`内部，你就可以使用点符号。

```py
import pandas as pd
car_data = pd.read_csv("data/cars.csv")
car_data['no_dlr_msrp'] = car_data['MSRP'].str.replace("$", "", 
 regex = False)
no_commas = car_data['no_dlr_msrp'].str.replace(",","") 
car_data['clean_MSRP'] = no_commas.astype(float)
car_data = car_data.sort_values(by='clean_MSRP', ascending = False)
car_data[["Make", "Model", "MSRP", "clean_MSRP"]].head(5)
##               Make                  Model      MSRP  clean_MSRP
## 334        Porsche            911 GT2 2dr  $192,465    192465.0
## 262  Mercedes-Benz              CL600 2dr  $128,420    128420.0
## 271  Mercedes-Benz  SL600 convertible 2dr  $126,670    126670.0
## 270  Mercedes-Benz           SL55 AMG 2dr  $121,770    121770.0
## 261  Mercedes-Benz              CL500 2dr   $94,820     94820.0
```

Pandas 的`DataFrame`s 和`Series`有一个[`.replace()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html)方法。我们使用这个方法从 MSRP 列中删除美元符号和逗号。请注意，在使用它之前，我们必须访问`Series`列的`.str`属性。字符串处理完毕后，我们使用`astype()`方法将其转换为`float`s 类型的`Series`。

最后，使用与我们在 R 中使用的相同方法（即通过行索引进行原始子集选择）对整个数据框进行排序可能已经完成，但有一个内置的方法叫做`sort_values()`（[`pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html)）可以为我们完成这项工作。

## 12.2 数据集堆叠和并排放置

将数据集堆叠在一起是一个常见的任务。你可能需要这样做，如果

1.  你需要向数据框中添加一行（或多行），

1.  你需要重新组合数据集（例如，重新组合训练/测试分割），或者

1.  你正在逐步创建一个矩阵。

在 R 中，这可以通过`rbind()`（即“行绑定”）来完成。考虑以下示例，它使用了从（Albemarle County Geographic Data Services Office 2021）查询的 GIS 数据，并使用（Ford 2016）的代码进行了清理。

```py
realEstate <-  read.csv("data/albemarle_real_estate.csv")
train <-  realEstate[-1,]
test <-  realEstate[1,]
str(rbind(test, train), strict.width = "cut")
## 'data.frame':    30381 obs. of  12 variables:
##  $ YearBuilt    : int  1769 1818 2004 2006 2004 1995 1900 1960 ..
##  $ YearRemodeled: int  1988 1991 NA NA NA NA NA NA NA NA ...
##  $ Condition    : chr  "Average" "Average" "Average" "Average" ..
##  $ NumStories   : num  1.7 2 1 1 1.5 2.3 2 1 1 1 ...
##  $ FinSqFt      : int  5216 5160 1512 2019 1950 2579 1530 800 9..
##  $ Bedroom      : int  4 6 3 3 3 3 4 2 2 2 ...
##  $ FullBath     : int  3 4 2 3 3 2 1 1 1 1 ...
##  $ HalfBath     : int  0 1 1 0 0 1 0 0 0 0 ...
##  $ TotalRooms   : int  8 11 9 10 8 8 6 4 4 4 ...
##  $ LotSize      : num  5.1 453.9 42.6 5 5.5 ...
##  $ TotalValue   : num  1096600 2978600 677800 453200 389200 ...
##  $ City         : chr  "CROZET" "CROZET" "CROZET" "CROZET" ...
sum(rbind(test, train) !=  realEstate)
## [1] NA
```

上述示例是关于`data.frame`s 的。以下`rbind()`的示例是关于`matrix`对象的。

```py
rbind(matrix(1,nrow = 2, ncol = 3), 
 matrix(2,nrow = 2, ncol = 3))
##      [,1] [,2] [,3]
## [1,]    1    1    1
## [2,]    1    1    1
## [3,]    2    2    2
## [4,]    2    2    2
```

在 Python 中，你可以使用`pd.concat()`函数来堆叠数据框（[pd.concat()](https://www.google.com/search?client=safari&rls=en&q=pandas+concat&ie=UTF-8&oe=UTF-8)）。它有很多选项，所以请随意浏览它们。你也可以将下面的`pd.concat()`调用替换为`test.append(train)`（[test.append(train)](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.append.html)）。考虑以下示例，它使用了阿尔伯马尔县房地产数据（阿尔伯马尔县地理数据服务办公室 2021）（福特 2016）。

```py
import pandas as pd
real_estate = pd.read_csv("data/albemarle_real_estate.csv")
train = real_estate.iloc[1:,]
test = real_estate.iloc[[0],] # need the extra brackets!
stacked = pd.concat([test,train], axis=0)
stacked.iloc[:3,:3]
##    YearBuilt  YearRemodeled Condition
## 0       1769         1988.0   Average
## 1       1818         1991.0   Average
## 2       2004            NaN   Average
(stacked != real_estate).sum().sum()
## 28251
```

注意当我们创建`test`时额外的方括号。如果你使用`real_estate.iloc[0,]`代替，它将返回一个包含所有元素强制转换为相同类型的`Series`，并且这不会与剩余的数据`pd.concat()`正确地合并！

## 12.3 合并或连接数据集

如果你有两个不同的数据集，它们提供了关于相同实验单位的信息，你可以使用**`merge`**（也称为**`join`**）操作将这两个数据集合并在一起。在 R 中，你可以使用`merge()`函数（[merge() 函数](https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/merge)）。在 Python 中，你可以使用[`.merge()` 方法](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html#pandas-dataframe-merge)。

合并（或连接）数据集与并排放置它们不同。并排放置数据集不会重新排序你的数据行，并且该操作要求两个输入数据集在开始时具有相同数量的行。另一方面，合并数据会智能地匹配行，并且它可以处理缺失或重复匹配的情况。在这两种情况下，结果数据集更宽，但合并时，输出可能包含更多或更少的行。

这里有一个澄清的例子。假设你有一些关于某些在线平台上个人账户的匿名数据集。

```py
# in R
baby1 <-  read.csv("data/baby1.csv", stringsAsFactors = FALSE)
baby2 <-  read.csv("data/baby2.csv", stringsAsFactors = FALSE)
head(baby1)
##   idnum height.inches.          email_address
## 1     1             74 fakeemail123@gmail.com
## 2     3             66  anotherfake@gmail.com
## 3     4             62      notreal@gmail.com
## 4    23             62      notreal@gmail.com
head(baby2)
##     idnum      phone                   email
## 1 3901283 5051234567       notreal@gmail.com
## 2   41823 5051234568 notrealeither@gmail.com
## 3 7198273 5051234568   anotherfake@gmail.com
```

你需要问自己的第一件事是**“这两个数据集之间共享的唯一标识符是哪一列？”**在我们的例子中，它们都有一个“识别号”列。然而，这两个数据集来自不同的在线平台，这两个地方使用不同的方案来编号他们的用户。

在这种情况下，最好按电子邮件地址进行合并。用户可能在这两个平台上使用不同的电子邮件地址，但匹配的电子邮件地址意味着你匹配了正确的账户。每个数据集中的列名都不同，因此我们必须按名称指定它们。

```py
# in R
merge(baby1, baby2, by.x = "email_address", by.y = "email")
##           email_address idnum.x height.inches. idnum.y      phone
## 1 anotherfake@gmail.com       3             66 7198273 5051234568
## 2     notreal@gmail.com       4             62 3901283 5051234567
## 3     notreal@gmail.com      23             62 3901283 5051234567
```

在 Python 中，`merge()`是每个`DataFrame`实例附加的方法。

```py
# in Python
baby1.merge(baby2, left_on = "email_address", right_on = "email")
##    idnum_x  height(inches)  ...       phone                  email
## 0        3              66  ...  5051234568  anotherfake@gmail.com
## 1        4              62  ...  5051234567      notreal@gmail.com
## 2       23              62  ...  5051234567      notreal@gmail.com
## 
## [3 rows x 6 columns]
```

电子邮件地址`anotherfake@gmail.com`和`notreal@gmail.com`在两个数据集中都存在，所以每个这些电子邮件地址最终都会出现在结果数据框中。结果数据集中的行更宽，并且每个个人都有更多的属性。

注意重复的电子邮件地址。在这种情况下，用户可能使用相同的电子邮件地址注册了两个账户，或者一个人使用另一个人的电子邮件地址注册了账户。在重复的情况下，两行将与另一个数据框中的相同行匹配。

此外，在这种情况下，所有在两个数据集中都没有找到的电子邮件地址都被丢弃了。这不一定需要是预期的行为。例如，如果我们想确保没有行被丢弃，那是可能的。然而，在这种情况下，对于在两个数据集中都没有找到的电子邮件地址，将缺少一些信息。回想一下，Python 和 R 处理缺失数据的方式不同（见 3.8.2）。

```py
# in R
merge(baby1, baby2, 
 by.x = "email_address", by.y = "email", 
 all.x = TRUE, all.y = TRUE)
##             email_address idnum.x height.inches. idnum.y      phone
## 1   anotherfake@gmail.com       3             66 7198273 5051234568
## 2  fakeemail123@gmail.com       1             74      NA         NA
## 3       notreal@gmail.com       4             62 3901283 5051234567
## 4       notreal@gmail.com      23             62 3901283 5051234567
## 5 notrealeither@gmail.com      NA             NA   41823 5051234568
```

```py
# in Python
le_merge = baby1.merge(baby2, 
 left_on = "email_address", right_on = "email", 
 how = "outer")
le_merge.iloc[:5,3:]
##      idnum_y         phone                    email
## 0        NaN           NaN                      NaN
## 1  7198273.0  5.051235e+09    anotherfake@gmail.com
## 2  3901283.0  5.051235e+09        notreal@gmail.com
## 3  3901283.0  5.051235e+09        notreal@gmail.com
## 4    41823.0  5.051235e+09  notrealeither@gmail.com
```

在 Python 中，这会稍微简洁一些。如果你熟悉 SQL，你可能听说过内连接和外连接。这就是 Pandas 从其中获取一些参数名称的地方（见[12.4.1](https://pandas.pydata.org/pandas-docs/version/0.15/merging.html#database-style-dataframe-joining-merging)）。

最后，如果连接的列中两个数据集都有多个值，结果可以比任一表有更多的行。这是因为**所有可能的匹配**都会出现。

```py
# in R
first <-  data.frame(category = c('a','a'), measurement = c(1,2))
merge(first, first, by.x = "category", by.y = "category")
##   category measurement.x measurement.y
## 1        a             1             1
## 2        a             1             2
## 3        a             2             1
## 4        a             2             2
```

```py
# in Python
first = pd.DataFrame({'category' : ['a','a'], 'measurement' : [1,2]})
first.merge(first, left_on = "category", right_on = "category")
##   category  measurement_x  measurement_y
## 0        a              1              1
## 1        a              1              2
## 2        a              2              1
## 3        a              2              2
```

## 12.4 长格式与宽格式数据

### 12.4.1 R 中的长格式与宽格式

许多类型的数据可以存储在**宽格式**或**长格式**中。

经典的例子是来自**纵向研究**的数据。如果一个实验单位（在下面的例子中是一个人）在一段时间内被反复测量，每行将对应于一个实验单位和一个数据集中长格式下的观察时间。

```py
peopleNames <-  c("Taylor","Taylor","Charlie","Charlie")
fakeLongData1 <-  data.frame(person = peopleNames, 
 timeObserved = c(1, 2, 1, 2),
 nums = c(100,101,300,301))
fakeLongData1
##    person timeObserved nums
## 1  Taylor            1  100
## 2  Taylor            2  101
## 3 Charlie            1  300
## 4 Charlie            2  301
```

如果你对实验单位有多个观察（在单个时间点），也可以使用长格式。这里还有一个例子。

```py
myAttrs <-  c("attrA","attrB","attrA","attrB")
fakeLongData2 <-  data.frame(person = peopleNames, 
 attributeName = myAttrs,
 nums = c(100,101,300,301))
fakeLongData2
##    person attributeName nums
## 1  Taylor         attrA  100
## 2  Taylor         attrB  101
## 3 Charlie         attrA  300
## 4 Charlie         attrB  301
```

如果你想要将长数据集重塑为宽格式，可以使用`reshape()`函数。你需要指定哪些列对应于实验单位，以及哪一列是“因子”变量。

```py
fakeWideData1 <-  reshape(fakeLongData1, 
 direction = "wide", 
 timevar = "timeObserved", 
 idvar = "person", 
 varying = c("before","after")) 
# ^ varying= arg becomes col names in new data set
fakeLongData1
##    person timeObserved nums
## 1  Taylor            1  100
## 2  Taylor            2  101
## 3 Charlie            1  300
## 4 Charlie            2  301
fakeWideData1
##    person before after
## 1  Taylor    100   101
## 3 Charlie    300   301
```

```py
# timevar= is a misnomer here
fakeWideData2 <-  reshape(fakeLongData2, 
 direction = "wide", 
 timevar = "attributeName", 
 idvar = "person", 
 varying = c("attribute A","attribute B")) 
fakeLongData2
##    person attributeName nums
## 1  Taylor         attrA  100
## 2  Taylor         attrB  101
## 3 Charlie         attrA  300
## 4 Charlie         attrB  301
fakeWideData2
##    person attribute A attribute B
## 1  Taylor         100         101
## 3 Charlie         300         301
```

`reshape()`也可以进行相反的操作：它可以将宽数据转换为长数据。

```py
reshape(fakeWideData1, 
 direction = "long",
 idvar = "person", 
 varying = list(c("before","after")),
 v.names = "nums")
##            person time nums
## Taylor.1   Taylor    1  100
## Charlie.1 Charlie    1  300
## Taylor.2   Taylor    2  101
## Charlie.2 Charlie    2  301
fakeLongData1
##    person timeObserved nums
## 1  Taylor            1  100
## 2  Taylor            2  101
## 3 Charlie            1  300
## 4 Charlie            2  301
reshape(fakeWideData2, 
 direction = "long",
 idvar = "person", 
 varying = list(c("attribute A","attribute B")),
 v.names = "nums")
##            person time nums
## Taylor.1   Taylor    1  100
## Charlie.1 Charlie    1  300
## Taylor.2   Taylor    2  101
## Charlie.2 Charlie    2  301
fakeLongData2
##    person attributeName nums
## 1  Taylor         attrA  100
## 2  Taylor         attrB  101
## 3 Charlie         attrA  300
## 4 Charlie         attrB  301
```

### 12.4.2 Python 中的长格式与宽格式

使用 Pandas，我们可以使用`pd.DataFrame.pivot()`将长数据转换为宽数据，并且我们可以使用`pd.DataFrame.melt()`进行相反的操作。

从长格式转换为宽格式后，请确保使用`pd.DataFrame.reset_index()`方法重新塑形数据并删除索引。这里有一个与上面类似的示例。

```py
import pandas as pd
fake_long_data1 = pd.DataFrame(
 {'person' : ["Taylor","Taylor","Charlie","Charlie"], 
 'time_observed' : [1, 2, 1, 2],
 'nums' : [100,101,300,301]})
fake_long_data1
##     person  time_observed  nums
## 0   Taylor              1   100
## 1   Taylor              2   101
## 2  Charlie              1   300
## 3  Charlie              2   301
pivot_data1 = fake_long_data1.pivot(index='person', 
 columns='time_observed', 
 values='nums')
fake_wide_data1 = pivot_data1.reset_index()
fake_wide_data1
## time_observed   person    1    2
## 0              Charlie  300  301
## 1               Taylor  100  101
```

这里有一个更多示例，展示了相同的功能——从长格式转换为宽格式。

```py
people_names = ["Taylor","Taylor","Charlie","Charlie"]
attribute_list = ['attrA', 'attrB', 'attrA', 'attrB']
fake_long_data2 = pd.DataFrame({'person' : people_names, 
 'attribute_name' : attribute_list,
 'nums' : [100,101,300,301]})
fake_wide_data2 = fake_long_data2.pivot(index='person', 
 columns='attribute_name', 
 values='nums').reset_index()
fake_wide_data2
## attribute_name   person  attrA  attrB
## 0               Charlie    300    301
## 1                Taylor    100    101
```

这里有一些从宽到长的其他方向的例子：使用[`pd.DataFrame.melt()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.melt.html?highlight=melt)。第一个示例通过整数指定值列。

```py
fake_wide_data1
## time_observed   person    1    2
## 0              Charlie  300  301
## 1               Taylor  100  101
fake_wide_data1.melt(id_vars = "person", value_vars = [1,2])
##     person time_observed  value
## 0  Charlie             1    300
## 1   Taylor             1    100
## 2  Charlie             2    301
## 3   Taylor             2    101
```

第二个示例使用字符串指定值列。

```py
fake_wide_data2
## attribute_name   person  attrA  attrB
## 0               Charlie    300    301
## 1                Taylor    100    101
fake_wide_data2.melt(id_vars = "person", 
 value_vars = ['attrA','attrB'])
##     person attribute_name  value
## 0  Charlie          attrA    300
## 1   Taylor          attrA    100
## 2  Charlie          attrB    301
## 3   Taylor          attrB    101
```

## 12.5 练习

### 12.5.1 R 问题

回忆`car.data`数据集（“汽车评估” 1997），由（Dua 和 Graff 2017）托管。

1.  将数据集读取为`carData`。

1.  将第三列和第四列转换为*有序*的`因子`。

1.  按第三列然后是第四列（同时）对数据进行排序。不要就地更改数据。相反，将其存储在`ordCarData1`下。

1.  按第四列然后是第三列（同时）对数据进行排序。不要就地更改数据。相反，将其存储在`ordCarData2`下。

```py
day1Data <-  data.frame(idNum = 1:10, 
 measure = rnorm(10))
day2Data <-  data.frame(idNum = 11:20, 
 measure = rnorm(10))
```

1.  假设`day1Data`和`day2Data`是两个具有相同类型测量但实验单位不同的独立数据集。将`day1Data`堆叠在`day2Data`之上，并将结果命名为`stackedData`。

1.  假设`day1Data`和`day2Data`是在相同实验单位上的不同测量值。将它们肩并肩放置，并将结果命名为`sideBySide`。将`day1Data`放在第一位，`day2Data`放在第二位。

如果你处理的是随机矩阵，你可能需要**向量化**一个矩阵对象。这不同于编程中的“向量化”。相反，这意味着你将矩阵写成一个大列向量，通过将列堆叠在一起。具体来说，如果你有一个$n \times p$的实值矩阵$\mathbf{X}$，那么

$$\begin{equation} \text{vec}(\mathbf{X}) =\begin{bmatrix} \mathbf{X}_1 \\ \vdots \\ \mathbf{X}_p \end{bmatrix} \end{equation}$$

其中$\mathbf{X}_i$是第$i$列作为一个$n \times 1$列向量。我们还将使用另一个运算符，**克罗内克积**：

$$\begin{equation} \mathbf{A} \otimes \mathbf{B} = \begin{bmatrix} a_{11} \mathbf{B} & \cdots & a_{1n} \mathbf{B} \\ \vdots & \ddots & \vdots \\ a_{m1} \mathbf{B} & \cdots & a_{mn} \mathbf{B} \\ \end{bmatrix}. \end{equation}$$

如果$\mathbf{A}$是$m \times n$且$\mathbf{B}$是$p \times q$，那么$\mathbf{A} \otimes \mathbf{B}$是$pm \times qn$。

1.  编写一个名为`vec(myMatrix)`的函数。它的输入应该是一个`矩阵`对象。它的输出应该是一个`向量`。提示：`矩阵`对象以列主序存储。

1.  编写一个名为`unVec(myVector, nRows)`的函数，该函数接受一个向量化的矩阵作为`向量`，将其分割成具有`nRows`个元素的元素，然后将它们肩并肩地作为一个`矩阵`放置在一起。

1.  编写一个名为`stackUp(m, BMat)`的函数，该函数返回$\mathbf{1}_m \otimes \mathbf{B}$，其中$\mathbf{1}_m$是一个长度为$m$的由一组成的列向量。你可以用`%x%`来检查你的工作，但不要在函数中使用它。

1.  编写一个名为 `shoulderToShoulder(n, BMat)` 的函数，该函数返回 $\mathbf{1}^\intercal_n \otimes \mathbf{B}$，其中 $\mathbf{1}_n^\intercal$ 是长度为 $n$ 的全一列向量。你可以用 `%x%` 来检查你的工作，但不要在函数中使用它。

此问题使用的是来自 [The Correlates of War Project](https://correlatesofwar.org/) 的军事化州际争端（v5.0）数据集（Palmer 等人）。我们为此问题使用了四个 `.csv` 文件。`MIDA 5.0.csv` 包含了从 1/1/1816 到 12/31/2014 的每个军事化州际争端的必要属性。`MIDB 5.0.csv` 描述了这些争端中的参与者。`MIDI 5.0.csv` 包含了每个军事化州际事件的必要要素，而 `MIDIP 5.0.csv` 描述了这些事件中的参与者。

1.  读入四个数据集，并将它们命名为 `mida`、`midb`、`midi` 和 `midp`。注意将所有 `-9` 实例转换为 `NA`。

1.  检查 `midb` 中 `dispnum` 列等于 `2` 的所有行。不要永久更改 `midb`。这两行是否对应同一冲突？如果是，将 `sameConflict` 赋值为 `TRUE`。否则，赋值为 `FALSE`。

1.  将前两个数据集在争端编号列（`dispnum`）上合并。将得到的 `data.frame` 命名为 `join1`。不要处理任何关于重复列的问题。

1.  在上一个问题中，内连接和外连接有什么区别？如果有区别，将 `theyAreNotTheSame` 赋值为 `TRUE`。否则，将其赋值为 `FALSE`。

1.  通过 `incidnum` 将最后两个数据集合并，结果命名为 `join2`。对于这个问题，内连接和外连接有什么区别？为什么？不要处理任何关于重复列的问题。

1.  代码手册提到，最后两个数据集的时间跨度不如前两个数据集长。那么，我们只关心 `join2` 中的事件。以丢弃 `join1` 中所有不需要的行，并保留 `join2` 中所有行的方式合并 `join2` 和 `join1`。将得到的 `data.frame` 命名为 `midData`。不要处理任何关于重复列的问题。

1.  使用散点图显示最大持续时间和结束年份之间的关系。将每个国家用不同的颜色表示。

1.  创建一个名为 `longData` 的 `data.frame`，其中包含 `midp` 的以下三个列：`incidnum`（事件识别号）`stabb`（参与者的州简称）和 `fatalpre`（精确的死亡人数）。将其转换为“宽”格式。将新表命名为 `wideData`。使用事件编号行作为唯一的行标识变量。

1.  奖励问题：识别 `midData` 中包含重复信息的所有列对，删除除了一个之外的所有列，并将列名改回原始名称。

### 12.5.2 Python 问题

再次回想一下 `"car.data"` 数据集（“Car Evaluation” 1997）。

1.  将数据集读入为 `car_data`。

1.  按照第三列和第四列的顺序对数据进行排序。不要在原地进行数据更改。相反，将其存储在名为 `ord_car_data1` 的名称下。

1.  按照第四列和第三列的顺序对数据进行排序。不要在原地进行数据更改。相反，将其存储在名为 `ord_car_data2` 的名称下。

考虑以下随机数据集。

```py
indexes  = np.random.choice(np.arange(20),size=20,replace=False)
d1 = pd.DataFrame({'a' : indexes, 
 'b' : np.random.normal(size=20)})
d2 = pd.DataFrame({'a' : indexes + 20, 
 'b' : np.random.normal(size=20)})
```

1.  假设 `d1` 和 `d2` 是两个具有相同类型测量但位于不同实验单位上的独立数据集。将 `d1` 堆叠在 `d2` 之上，并将结果称为 `stacked_data_sets`。确保结果的 `index` 是数字 $0$ 到 $39$。

1.  假设 `d1` 和 `d2` 是对相同实验单位的两种不同测量。将它们并排放置，并将结果称为 `side_by_side_data_sets`。将 `d1` 放在第一位，`d2` 放在第二位。

考虑以下两个数据集：

```py
import numpy as np
import pandas as pd
dog_names1 = ['Charlie','Gus', 'Stubby', 'Toni','Pearl']
dog_names2 = ['Charlie','Gus', 'Toni','Arya','Shelby']
nicknames = ['Charles','Gus The Bus',np.nan,'Toni Bologna','Porl']
breed_names = ['Black Lab','Beagle','Golden Retriever','Husky',np.nan]
dataset1 = pd.DataFrame({'dog': dog_names1,
 'nickname': nicknames})
dataset2 = pd.DataFrame({'dog':dog_names2,
 'breed':breed_names})
```

1.  以这种方式合并/连接两个数据集，使得每只狗都有一行，无论两个表是否都有关于该狗的信息。将结果称为 `merged1`。

1.  以这种方式合并/连接两个数据集，使得 `dataset1` 中的每只狗都只有一行，无论是否有关于这些狗品种的信息。将结果称为 `merged2`。

1.  以这种方式合并/连接两个数据集，使得 `dataset2` 中的每只狗都只有一行，无论是否有关于狗昵称的信息。将结果称为 `merged3`。

1.  以这种方式合并/连接两个数据集，使得所有行都拥有完整的信息。将结果称为 `merged4`。

让我们再次考虑费舍尔的“鸢尾花”数据集（Fisher 1988）。

1.  读取 `iris.csv` 并将 `DataFrame` 存储为名为 `iris`。让它有列名 `'a'`、`'b'`、`'c'`、`'d'` 和 `'e'`。

1.  创建一个名为 `name_key` 的 `DataFrame`，用于存储长名称和短名称之间的对应关系。它应该有 3 行和 2 列。长名称是 `iris` 第五列的唯一值。短名称可以是 `'s'`、`'vers'` 或 `'virg'`。使用列名 `'long name'` 和 `'short name'`。

1.  将这两个数据集合并/连接在一起，为 `iris` 添加一个包含短名称信息的新列。不要覆盖 `iris`。相反，给 `DataFrame` 起一个新的名字：`iris_with_short_names`。删除任何包含重复信息的列。

1.  将 `iris_with_short_names` 的前四列列名更改为 `s_len`、`s_wid`、`p_len` 和 `p_wid`。使用 Matplotlib 创建一个包含 4 个子图，排列成 $2 \times 2$ 网格的图形。在每个子图上，绘制这四个列的直方图。确保使用 x 轴标签，以便观众可以知道每个子图正在绘制哪个列。

1.  让我们回到 `iris`。将其更改为长格式。将其存储为名为 `long_iris` 的 `DataFrame`。按顺序将列名设置为 `row`、`variable` 和 `value`。最后，确保它按 `row` 和 `variable` 排序（同时/一次）。

### 参考文献

阿尔贝马尔县地理数据服务办公室。2021\. “阿尔贝马尔县 GIS 网站。” [`www.albemarle.org/government/community-development/gis-mapping/gis-data`](https://www.albemarle.org/government/community-development/gis-mapping/gis-data).

“汽车评估。” 1997\. UCI 机器学习库。

杜阿，德鲁，和凯西·格拉夫。2017\. “UCI 机器学习库。” 加州大学欧文分校，信息学院；计算机科学系。[`archive.ics.uci.edu/ml`](http://archive.ics.uci.edu/ml).

费舍尔，测试，R.A. & 创作者。1988\. “鸢尾花。” UCI 机器学习库。

福特，克莱。2016\. “ggplot: UVA StatLab 研讨会，2016 年秋季的文件。” *GitHub 仓库*。[`github.com/clayford/ggplot2`](https://github.com/clayford/ggplot2); GitHub.

帕尔默，格伦，罗珊·W·麦克曼斯，维托·多拉齐奥，迈克尔·R·肯威克，米凯拉·卡尔滕斯，查斯·布洛克，尼克·迪特里希，凯拉·卡恩，凯兰·里特，迈克尔·J·索尔斯“The Mid5 数据集，2011–2014：程序，编码规则和描述。” *冲突管理与和平科学* 0 (0): 0738894221995743\. [`doi.org/10.1177/0738894221995743`](https://doi.org/10.1177/0738894221995743).

“SAS^(Viya^(示例数据集。” 2021\. [`support.sas.com/documentation/onlinedoc/viya/examples.htm`](https://support.sas.com/documentation/onlinedoc/viya/examples.htm).))
