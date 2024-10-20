# 如何用 30 个代码示例在 R 中创建一个数据框架

> 原文：<https://www.dataquest.io/blog/how-to-create-a-dataframe-in-r/>

May 31, 2022![R Tutorial](img/969366f6d5dc89fc68ad7a6eb6cd0246.png)

## 数据帧是 R 编程语言中的基本数据结构。在本教程中，我们将讨论如何在 r 中创建数据帧。

R 中的数据帧是表格(即二维矩形)数据结构，用于存储任何数据类型的值。它是一个基数 R 的数据结构，这意味着我们不需要安装任何特定的包来创建数据帧并使用它。

与任何其他表一样，每个数据帧都由列(表示变量或属性)和行(表示数据条目)组成。就 R 而言，数据帧是一个长度相等的向量列表，同时也是一个二维数据结构，它类似于一个 R 矩阵，不同之处在于:一个矩阵必须只包含一种数据类型，而一个数据帧则更加通用，因为它可以有多种数据类型。然而，虽然数据帧的不同列可以有不同的数据类型，但是每一列都应该是相同的数据类型。

## 从向量创建 R 中的数据帧

为了从一个或多个相同长度的向量创建 R 中的数据帧，我们使用了 [`data.frame()`](https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/data.frame) 函数。其最基本的语法如下:

```py
df <- data.frame(vector_1, vector_2)
```

我们可以传递任意多的向量给这个函数。每个向量将代表一个数据帧列，并且任何向量的长度将对应于新数据帧中的行数。也可以只传递一个向量给`data.frame()`函数，在这种情况下，将创建一个只有一列的 DataFrame。

从向量创建 R 数据帧的一种方法是首先创建每个向量，然后按照必要的顺序将它们传递给`data.frame()`函数:

```py
rating <- 1:4
animal <- c('koala', 'hedgehog', 'sloth', 'panda') 
country <- c('Australia', 'Italy', 'Peru', 'China')
avg_sleep_hours <- c(21, 18, 17, 10)
super_sleepers <- data.frame(rating, animal, country, avg_sleep_hours)
print(super_sleepers)
```

```py
 rating   animal   country avg_sleep_hours
1      1    koala Australia              21
2      2 hedgehog     Italy              18
3      3    sloth      Peru              17
4      4    panda     China              10
```

(*边注:*更准确的说，刺猬和树懒的地理范围更广一点，但是我们不要那么挑剔！)

或者，我们可以在下面的函数中直接提供所有的向量:

```py
super_sleepers <- data.frame(rating=1:4, 
                             animal=c('koala', 'hedgehog', 'sloth', 'panda'), 
                             country=c('Australia', 'Italy', 'Peru', 'China'),
                             avg_sleep_hours=c(21, 18, 17, 10))
print(super_sleepers)
```

```py
 rating   animal   country avg_sleep_hours
1      1    koala Australia              21
2      2 hedgehog     Italy              18
3      3    sloth      Peru              17
4      4    panda     China              10
```

我们得到了与前面代码中相同的数据帧。请注意以下特征:

*   可使用`c()`函数(如`c('koala', 'hedgehog', 'sloth', 'panda')`)或范围(如`1:4`)创建数据帧的向量。
*   在第一种情况下，我们使用赋值操作符`<-`来创建向量。在第二个例子中，我们使用了参数赋值操作符`=`。
*   在这两种情况下，向量的名称变成了结果数据帧的列名。
*   在第二个例子中，我们可以在引号中包含每个向量名(例如`'rating'=1:4`)，但这并不是真正必要的——结果是一样的。

让我们确认我们得到的数据结构确实是一个数据帧:

```py
print(class(super_sleepers))
```

```py
[1] "data.frame"
```

现在，让我们探索它的结构:

```py
print(str(super_sleepers))
```

```py
'data.frame':   4 obs. of  4 variables:
 $ rating         : int  1 2 3 4
 $ animal         : Factor w/ 4 levels "hedgehog","koala",..: 2 1 4 3
 $ country        : Factor w/ 4 levels "Australia","China",..: 1 3 4 2
 $ avg_sleep_hours: num  21 18 17 10
NULL
```

我们看到，尽管`animal`和`country`向量最初是字符向量，但是对应的列具有 factor 数据类型。这种转换是`data.frame()`函数的默认行为。为了抑制它，我们需要添加一个可选参数`stringsAsFactors`，并将其设置为`FALSE`:

```py
super_sleepers <- data.frame(rating=1:4, 
                             animal=c('koala', 'hedgehog', 'sloth', 'panda'), 
                             country=c('Australia', 'Italy', 'Peru', 'China'),
                             avg_sleep_hours=c(21, 18, 17, 10),
                             stringsAsFactors=FALSE)
print(str(super_sleepers))
```

```py
'data.frame':   4 obs. of  4 variables:
 $ rating         : int  1 2 3 4
 $ animal         : chr  "koala" "hedgehog" "sloth" "panda"
 $ country        : chr  "Australia" "Italy" "Peru" "China"
 $ avg_sleep_hours: num  21 18 17 10
NULL
```

现在我们看到从字符向量创建的列也是字符数据类型。

还可以添加数据帧中各行的名称(默认情况下，这些行是从 1 开始的连续整数)。为此，我们使用可选参数`row.names`，如下所示:

```py
super_sleepers <- data.frame(rating=1:4, 
                             animal=c('koala', 'hedgehog', 'sloth', 'panda'), 
                             country=c('Australia', 'Italy', 'Peru', 'China'),
                             avg_sleep_hours=c(21, 18, 17, 10),
                             row.names=c('row_1', 'row_2', 'row_3', 'row_4'))
print(super_sleepers)
```

```py
 rating   animal   country avg_sleep_hours
row_1      1    koala Australia              21
row_2      2 hedgehog     Italy              18
row_3      3    sloth      Peru              17
row_4      4    panda     China              10
```

注意，在 R 数据帧中，列名和行名(如果存在的话)必须是唯一的。如果我们错误地为两列提供了相同的名称，R 会自动为其中的第二列添加一个后缀:

```py
# Adding by mistake 2 columns called 'animal'
super_sleepers <- data.frame(animal=1:4, 
                             animal=c('koala', 'hedgehog', 'sloth', 'panda'), 
                             country=c('Australia', 'Italy', 'Peru', 'China'),
                             avg_sleep_hours=c(21, 18, 17, 10))
print(super_sleepers)
```

```py
 animal animal.1   country avg_sleep_hours
1      1    koala Australia              21
2      2 hedgehog     Italy              18
3      3    sloth      Peru              17
4      4    panda     China              10
```

相反，如果我们在行名上犯了类似的错误，程序将抛出一个错误:

```py
# Naming by mistake 2 rows 'row_1'
super_sleepers <- data.frame(rating=1:4, 
                             animal=c('koala', 'hedgehog', 'sloth', 'panda'), 
                             country=c('Australia', 'Italy', 'Peru', 'China'),
                             avg_sleep_hours=c(21, 18, 17, 10),
                             row.names=c('row_1', 'row_1', 'row_3', 'row_4'))
print(super_sleepers)
```

```py
Error in data.frame(rating = 1:4, animal = c("koala", "hedgehog", "sloth", : duplicate row.names: row_1
Traceback:

1\. data.frame(rating = 1:4, animal = c("koala", "hedgehog", "sloth", 
 .     "panda"), country = c("Australia", "Italy", "Peru", "China"), 
 .     avg_sleep_hours = c(21, 18, 17, 10), row.names = c("row_1", 
 .         "row_1", "row_3", "row_4"))

2\. stop(gettextf("duplicate row.names: %s", paste(unique(row.names[duplicated(row.names)]), 
 .     collapse = ", ")), domain = NA)
```

如有必要，我们可以在创建后使用`names()`函数重命名数据帧的列:

```py
names(super_sleepers) <- c('col_1', 'col_2', 'col_3', 'col_4')
print(super_sleepers)
```

```py
 col_1    col_2     col_3 col_4
1     1    koala Australia    21
2     2 hedgehog     Italy    18
3     3    sloth      Peru    17
4     4    panda     China    10
```

## 从矩阵创建 R 中的数据帧

从矩阵中创建 R 数据帧是可能的。**然而**，在这种情况下，一个新数据帧的所有值将是相同的数据类型。

假设我们有以下矩阵:

```py
my_matrix <- matrix(c(1, 2, 3, 4, 5, 6), nrow=2)
print(my_matrix)
```

```py
 [,1] [,2] [,3]
[1,]    1    3    5
[2,]    2    4    6
```

我们可以使用相同的`data.frame()`函数从它创建一个数据帧:

```py
df_from_matrix <- data.frame(my_matrix)
print(df_from_matrix)
```

```py
 X1 X2 X3
1  1  3  5
2  2  4  6
```

不幸的是，在这种情况下，无法使用一些可选参数直接在函数内部更改列名(然而，具有讽刺意味的是，我们仍然可以使用`row.names`参数重命名行)。由于默认的列名不是描述性的(或者至少是有意义的)，我们必须在创建 DataFrame 之后通过应用`names()`函数来修复它:

```py
names(df_from_matrix) <- c('col_1', 'col_2', 'col_3')
print(df_from_matrix)
```

```py
 col_1 col_2 col_3
1     1     3     5
2     2     4     6
```

## 从向量列表中创建 R 中的数据帧

在 R 中创建数据帧的另一种方法是向`data.frame()`函数提供一个向量列表。事实上，我们可以把 R 数据帧看作是所有向量长度相同的向量列表的一个特例。

假设我们有下面的向量列表:

```py
my_list <- list(rating=1:4,
                animal=c('koala', 'hedgehog', 'sloth', 'panda'), 
                country=c('Australia', 'Italy', 'Peru', 'China'),
                avg_sleep_hours=c(21, 18, 17, 10))
print(my_list)
```

```py
$rating
[1] 1 2 3 4

$animal
[1] "koala"    "hedgehog" "sloth"    "panda"   

$country
[1] "Australia" "Italy"     "Peru"      "China"    

$avg_sleep_hours
[1] 21 18 17 10
```

现在，我们想从它创建一个数据帧:

```py
super_sleepers <- data.frame(my_list)
print(super_sleepers)
```

```py
 rating   animal   country avg_sleep_hours
1      1    koala Australia              21
2      2 hedgehog     Italy              18
3      3    sloth      Peru              17
4      4    panda     China              10
```

需要再次强调的是，为了能够从向量列表中创建数据帧，所提供的列表中的每个向量必须具有相同数量的条目；否则，程序会抛出一个错误。例如，如果我们试图在创建`my_list`时删除“考拉”,我们仍然可以创建向量列表。然而，当我们试图使用该列表创建数据帧时，我们会得到一个错误。

这里需要注意的另一点是，列表的项应该是名为的*(直接在创建向量列表时，就像我们做的那样，或者稍后，在列表上应用`names()`函数)。如果我们不这样做，并将一个包含“无名”向量的列表传递给`data.frame()`函数，我们将得到一个包含毫无意义的列名的数据帧:*

```py
my_list <- list(1:4,
                c('koala', 'hedgehog', 'sloth', 'panda'), 
                c('Australia', 'Italy', 'Peru', 'China'),
                c(21, 18, 17, 10))

super_sleepers <- data.frame(my_list)
print(super_sleepers)
```

```py
 X1.4 c..koala....hedgehog....sloth....panda..
1    1                                    koala
2    2                                 hedgehog
3    3                                    sloth
4    4                                    panda
  c..Australia....Italy....Peru....China.. c.21..18..17..10.
1                                Australia                21
2                                    Italy                18
3                                     Peru                17
4                                    China                10
```

## 从其他数据帧创建 R 中的数据帧

我们可以通过组合两个或多个其他数据帧来创建 R 中的一个数据帧。我们可以水平或垂直地做这件事。

为了水平组合数据帧*(即将一个数据帧的列添加到另一个数据帧的列中)，我们使用了`cbind()`函数，在这里我们传递必要的数据帧。*

 *假设我们有一个只包含`super_sleepers`表的前两列的数据帧:

```py
super_sleepers_1 <- data.frame(rating=1:4, 
                               animal=c('koala', 'hedgehog', 'sloth', 'panda'))
print(super_sleepers_1)
```

```py
 rating   animal
1      1    koala
2      2 hedgehog
3      3    sloth
4      4    panda
```

`super_sleepers`的后两列保存在另一个数据帧中:

```py
super_sleepers_2 <- data.frame(country=c('Australia', 'Italy', 'Peru', 'China'),
                               avg_sleep_hours=c(21, 18, 17, 10))
print(super_sleepers_2)
```

```py
 country avg_sleep_hours
1 Australia              21
2     Italy              18
3      Peru              17
4     China              10
```

现在，我们将应用`cbind()`函数来连接两个数据帧，并获得初始的`super_sleepers`数据帧:

```py
super_sleepers <- cbind(super_sleepers_1, super_sleepers_2)
print(super_sleepers)
```

```py
 rating   animal   country avg_sleep_hours
1      1    koala Australia              21
2      2 hedgehog     Italy              18
3      3    sloth      Peru              17
4      4    panda     China              10
```

注意，为了成功执行上述操作，数据帧必须具有相同的行数；否则，我们会得到一个错误。

类似地，为了垂直组合数据帧*(即，将一个数据帧的行添加到另一个数据帧的行)，我们使用`rbind()`函数，在这里我们传递必要的数据帧。*

 *假设我们有一个只包含前两行`super_sleepers`的数据帧:

```py
super_sleepers_1 <- data.frame(rating=1:2, 
                               animal=c('koala', 'hedgehog'), 
                               country=c('Australia', 'Italy'),
                               avg_sleep_hours=c(21, 18))
print(super_sleepers_1)
```

```py
 rating   animal   country avg_sleep_hours
1      1    koala Australia              21
2      2 hedgehog     Italy              18
```

另一个数据帧包含最后两行`super_sleepers`:

```py
super_sleepers_2 <- data.frame(rating=3:4, 
                               animal=c('sloth', 'panda'), 
                               country=c('Peru', 'China'),
                               avg_sleep_hours=c(17, 10))
print(super_sleepers_2)
```

```py
 rating animal country avg_sleep_hours
1      3  sloth    Peru              17
2      4  panda   China              10
```

让我们使用`rbind()`函数将它们垂直组合起来，得到我们的初始数据帧:

```py
super_sleepers <- rbind(super_sleepers_1, super_sleepers_2)
print(super_sleepers)
```

```py
 rating   animal   country avg_sleep_hours
1      1    koala Australia              21
2      2 hedgehog     Italy              18
3      3    sloth      Peru              17
4      4    panda     China              10
```

注意，为了成功执行该操作，数据帧必须具有相同数量的列和相同顺序的相同列名。否则，我们会得到一个错误。

## 在 R 中创建一个空的数据帧

在某些情况下，我们可能需要创建一个空的 R 数据帧，只有列名和列数据类型，没有行——然后使用 for 循环填充。为此，我们再次应用`data.frame()`函数，如下所示:

```py
super_sleepers_empty <- data.frame(rating=numeric(),
                                   animal=character(),
                                   country=character(),
                                   avg_sleep_hours=numeric())
print(super_sleepers_empty)
```

```py
[1] rating          animal          country         avg_sleep_hours
<0 rows> (or 0-length row.names)
```

让我们检查新的空数据帧的列的数据类型:

```py
print(str(super_sleepers_empty))
```

```py
'data.frame':   0 obs. of  4 variables:
 $ rating         : num 
 $ animal         : Factor w/ 0 levels: 
 $ country        : Factor w/ 0 levels: 
 $ avg_sleep_hours: num 
NULL
```

正如我们前面看到的，由于由`data.frame()`函数进行的默认转换，我们希望成为字符数据类型的列实际上是因子数据类型。如前所述，我们可以通过引入一个可选参数`stringsAsFactors=FALSE`来修复它:

```py
super_sleepers_empty <- data.frame(rating=numeric(),
                                   animal=character(),
                                   country=character(),
                                   avg_sleep_hours=numeric(),
                                   stringsAsFactors=FALSE)
print(str(super_sleepers_empty))
```

```py
'data.frame':   0 obs. of  4 variables:
 $ rating         : num 
 $ animal         : chr 
 $ country        : chr 
 $ avg_sleep_hours: num 
NULL
```

> **注意:**当应用`data.frame()`函数时，添加`stringsAsFactors=FALSE`参数总是一个好的实践。我们在本教程中并没有太多使用它，只是为了避免代码过载，把重点放在主要细节上。但是，对于实际任务，您应该始终考虑添加此参数，以防止包含字符数据类型的数据帧出现不良行为。

在 R 中创建空数据帧的另一种方法是创建另一个数据帧的空“副本”(实际上意味着我们只复制列名和它们的数据类型)。

让我们重新创建原始的`super_sleepers`(这次，使用`stringsAsFactors=FALSE`参数):

```py
super_sleepers <- data.frame(rating=1:4, 
                             animal=c('koala', 'hedgehog', 'sloth', 'panda'), 
                             country=c('Australia', 'Italy', 'Peru', 'China'),
                             avg_sleep_hours=c(21, 18, 17, 10),
                             stringsAsFactors=FALSE)
print(super_sleepers)
```

```py
 rating   animal   country avg_sleep_hours
1      1    koala Australia              21
2      2 hedgehog     Italy              18
3      3    sloth      Peru              17
4      4    panda     China              10
```

现在，使用以下语法创建一个空模板作为新的数据帧:

```py
super_sleepers_empty <- super_sleepers[FALSE, ]
print(super_sleepers_empty)
```

```py
[1] rating          animal          country         avg_sleep_hours
<0 rows> (or 0-length row.names)
```

让我们仔细检查原始数据帧的列的数据类型是否保留在新的空数据帧中:

```py
print(str(super_sleepers_empty))
```

```py
'data.frame':   0 obs. of  4 variables:
 $ rating         : int 
 $ animal         : chr 
 $ country        : chr 
 $ avg_sleep_hours: num 
NULL
```

最后，我们可以从一个没有行和必要列数的矩阵中创建一个空的数据帧，然后给它分配相应的列名:

```py
columns= c('rating', 'animal', 'country', 'avg_sleep_hours') 
super_sleepers_empty = data.frame(matrix(nrow=0, ncol=length(columns))) 
names(super_sleepers_empty) = columns
print(super_sleepers_empty)
```

```py
[1] rating          animal          country         avg_sleep_hours
<0 rows> (or 0-length row.names)
```

最后一种方法的一个潜在缺点是没有从一开始就设置列的数据类型:

```py
print(str(super_sleepers_empty))
```

```py
'data.frame':   0 obs. of  4 variables:
 $ rating         : logi 
 $ animal         : logi 
 $ country        : logi 
 $ avg_sleep_hours: logi 
NULL
```

## 从文件中读取 R 中的数据帧

除了在 R 中从头开始创建数据帧之外，我们还可以以表格形式导入一个已经存在的数据集，并将其保存为数据帧。事实上，这是为现实任务创建 R 数据框架的最常见方式。

为了了解它是如何工作的，让我们在本地机器上下载一个 Kaggle 数据集[橙子对葡萄柚](https://www.kaggle.com/datasets/joshmcadams/oranges-vs-grapefruit)，将其保存在与笔记本相同的文件夹中，将其作为新的数据帧`citrus`读取，并可视化数据帧的前六行。由于原始数据集以 *csv* 文件的形式存在，我们将使用`read.csv()`函数来读取它:

```py
citrus <- read.csv('citrus.csv')
print(head(citrus))
```

```py
 name diameter weight red green blue
1 orange     2.96  86.76 172    85    2
2 orange     3.91  88.05 166    78    3
3 orange     4.42  95.17 156    81    2
4 orange     4.47  95.60 163    81    4
5 orange     4.48  95.76 161    72    9
6 orange     4.59  95.86 142   100    2
```

> **注意:**不强制将下载的文件保存在工作笔记本的同一文件夹中。如果文件保存在另一个地方，我们只需要提供文件的完整路径，而不仅仅是文件名(例如，`'C:/Users/User/Downloads/citrus.csv'`)。然而，将数据集文件保存在与工作笔记本相同的文件中是一个好的做法。

上面的代码将数据集`'citrus.csv'`读入名为`citrus`的数据帧。

也可以阅读其他类型的文件，而不是 *csv* 。在其他情况下，我们可以找到有用的函数 [`read.table()`](https://www.rdocumentation.org/packages/utils/versions/3.6.2/topics/read.table) (用于读取任何类型的表格数据)、 [`read.delim()`](https://www.rdocumentation.org/packages/utils/versions/3.6.2/topics/read.table) (用于制表符分隔的文本文件)，以及 [`read.fwf()`](https://www.rdocumentation.org/packages/utils/versions/3.6.2/topics/read.fwf) (用于固定宽度的格式化文件)。

## 结论

在本教程中，我们探索了在 R 中创建数据帧的不同方法:从一个或多个向量、从一个矩阵、从向量列表、水平或垂直组合其他数据帧、读取可用的表格数据集并将其分配给新的数据帧。此外，我们考虑了在 R 中创建空数据帧的三种不同方法，以及这些方法何时适用。我们特别关注代码的语法及其变体、技术上的细微差别、好的和坏的实践、可能的陷阱以及修复或避免它们的变通方法。**