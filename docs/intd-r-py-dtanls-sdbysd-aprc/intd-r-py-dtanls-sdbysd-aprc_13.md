# 第九章 输入和输出

> [`randpythonbook.netlify.app/input-and-output`](https://randpythonbook.netlify.app/input-and-output)

## 9.1 一般输入考虑

到目前为止，本文档一直倾向于在脚本中创建小块数据 *within our scripts.* 避免从外部文件读取数据主要是出于教学目的。一般来说，可能会

+   从纯文本文件（例如`"my_data.csv"`或`"log_file.txt"`）读取的数据，

+   从数据库（例如 MySQL、PostgreSQL 等）读取的数据，或

+   在脚本中创建的数据（无论是确定性的还是随机的）。

当讨论读取数据时，本文档主要关注第一类。这样做的原因如下：

1.  文本文件比数据库更容易为学生所获取，

1.  教授第二类需要教授 SQL，这将引入概念上的重叠，

1.  第三类是程序上自我解释的。

第三种原因并不意味着由代码创建的数据不重要。例如，它是创建用于 **模拟研究** 中数据最常见的方法。撰写统计论文的作者需要证明他们的技术可以在“良好”的数据上工作：从 *已知* 的数据生成过程中模拟的数据。在模拟研究中，与“现实世界”不同，你可以访问生成你数据的参数，并检查可能无法观察或隐藏的数据。此外，从现实世界的数据中，无法保证你的模型正确地匹配了真实模型。

你的代码/技术/算法，至少，能否获得与你的代码用于模拟数据的参数“一致”的参数估计？通过你的方法获得的预测或预测是否准确？这些问题通常只能通过模拟假数据来回答。程序上，模拟此类数据主要涉及调用我们之前见过的函数（例如 R 中的`rnorm()`或 Python 中的`np.random.choice()`）。这可能或可能不涉及首先设置伪随机数种子，以实现可重复性。

此外，*基准数据集* 通常可以通过专门的函数调用轻松获取。

尽管这一章的目的是教你如何将文件读入 R 和 Python，但你不应期望在阅读这一节后就能知道如何读取 *所有* 数据集。对于 R 和 Python，都有大量的函数，不同的函数有不同的返回类型，不同的函数适用于不同的文件类型，许多函数散布在众多第三方库中，而且许多这些函数有大量的参数。你可能无法记住所有这些。以我非常谦卑的观点来看，我怀疑你是否有这样的愿望。

相反，**专注于提高你识别和诊断数据输入问题的能力。** 正确读取数据集通常是一个试错的过程。在尝试读取数据集后，始终检查以下项目。其中许多点在前面提到的 @(data-frames-in-r) 部分中已经提到。有些点比从数据库读取结构化数据读取文本数据更适用，反之亦然。

1.  检查**是否使用了正确的列分隔符，或者期望了正确的“固定宽度格式”。** 如果出错，数据框的列可能会以奇怪的方式合并或拆分，并且通常会对数据片段使用错误的数据类型（例如，`"2,3"`而不是 `2` 和 `3`）。此外，注意分隔符是否出现在数据元素或列名称中。例如，有时不清楚以“last, first”格式存储的人名是否可以存储在一列或两列中。此外，文本数据可能会让你惊讶于意外的空格或其他空白字符作为分隔符。

1.  检查**列名是否正确解析和存储。** 列名不应作为 R/Python 中的数据存储。读取数据的函数不应期望在文件中不存在列名时存在列名。

1.  检查**空格和元数据是否正确忽略。** 数据描述有时与数据本身存储在同一个文件中，在读取时应跳过。列名和数据之间的空格不应存储。这可能会出现在文件的开始处，甚至出现在文件的末尾。

1.  **检查类型选择和特殊字符的识别是否正确执行。** 字母是以字符串存储还是以其他方式存储，例如 R 的 `factor`？日期和时间是以特殊的日期/时间类型存储还是以字符串存储？缺失数据是否正确识别？有时数据提供者使用像 $-9999$ 这样的极端数字来表示缺失数据——不要将其存储为浮点数或整数！

1.  **准备好提示 R 或 Python 识别特定字符编码，如果你正在读取用其他语言编写的文本数据。** 所有文本数据都有字符编码，这是一种将数字映射到字符的映射。任何特定的编码都将决定程序中可识别的字符。如果你尝试读取用另一种语言编写的文本数据，你使用的函数可能会抱怨无法识别的字符。幸运的是，通过指定非默认参数，如 `encoding=` 或 `fileEncoding=`，这些错误和警告很容易修复。

这不是一项小任务。更糟糕的是：

+   你不能（或不应该）编辑原始数据以满足你的需求，使其更容易阅读。你必须处理你得到的数据。如果你被允许编辑，比如说，你下载到自己的机器上的文本文件，你不应该这样做——这会导致代码在其他任何地方都无法运行。此外，如果你滥用公司数据库的写权限，例如——这也可能非常危险。

+   数据集通常相当大，所以手动检查每个元素通常是不可能的。在这种情况下，你必须接受检查数据集的顶部和底部，或者可能预测一个可能出现问题的特定位置。

## 9.2 使用 R 读取文本文件

你在本书中已经看到了 `read.csv()` 的示例，所以这不应该让你感到惊讶，这是在 R 中读取数据最常见的方法之一。另一个重要的函数是 `read.table()`。

如果你查看 `read.csv()` 的源代码（在控制台中输入函数名称（不带括号）并按 `<Enter>` 键），你会看到它调用了 `read.table()`。这两个函数的主要区别在于默认参数。**注意默认参数**。不要完全反对写一行长的代码来正确读取数据集。或者，如果你这样做，请选择具有最佳默认参数的函数。

考虑一下来自（Dua 和 Graff 2017）的“Challenger USA Space Shuttle O-Ring 数据集”（[“Challenger USA Space Shuttle O-Ring Data Set”](https://archive.ics.uci.edu/ml/datasets/Challenger+USA+Space+Shuttle+O-Ring)）。原始文本文件的前几行¹⁴看起来像这样。

```py
6 0 66  50  1
6 1 70  50  2
6 0 69  50  3
```

它不使用逗号作为分隔符，也没有标题信息，所以使用默认参数的 `read.csv()` 会产生错误的结果。它会将第一行误认为是列名，并将所有内容存储在一个错误的类型的列中。

```py
d <-  read.csv("data/o-ring-erosion-only.data")
dim(d) # one row short, only 1 col
## [1] 22  1
typeof(d[,1])
## [1] "character"
```

指定 `header=FALSE` 解决了列名问题，但 `sep = " "` 并没有解决分隔符问题。

```py
d <-  read.csv("data/o-ring-erosion-only.data", 
 header=FALSE, sep = " ")
str(d)
## 'data.frame':    23 obs. of  7 variables:
##  $ V1: int  6 6 6 6 6 6 6 6 6 6 ...
##  $ V2: int  0 1 0 0 0 0 0 0 1 1 ...
##  $ V3: int  66 70 69 68 67 72 73 70 57 63 ...
##  $ V4: int  NA NA NA NA NA NA 100 100 200 200 ...
##  $ V5: int  50 50 50 50 50 50 NA NA NA 10 ...
##  $ V6: int  NA NA NA NA NA NA 7 8 9 NA ...
##  $ V7: int  1 2 3 4 5 6 NA NA NA NA ...
```

一个空格严格是一个空格。尽管有些行有两个空格。这导致有两个多余的列填充了 `NA`。

在进一步挖掘文档之后，你会注意到 `""` 对“一个或多个空格、制表符、换行符或回车符”有效。这就是为什么 `read.table()`（使用其默认参数）工作得很好。

```py
d <-  read.table("data/o-ring-erosion-only.data")
str(d)
## 'data.frame':    23 obs. of  5 variables:
##  $ V1: int  6 6 6 6 6 6 6 6 6 6 ...
##  $ V2: int  0 1 0 0 0 0 0 0 1 1 ...
##  $ V3: int  66 70 69 68 67 72 73 70 57 63 ...
##  $ V4: int  50 50 50 50 50 50 100 100 200 200 ...
##  $ V5: int  1 2 3 4 5 6 7 8 9 10 ...
```

这个数据集的列宽度也是“固定”的。它处于“固定宽度格式”，因为任何给定列的所有元素都占用固定数量的字符。第三列包含两位或三位数的整数，但无论如何，每一行都有相同数量的字符。

你可以选择利用这一点并使用一个专门的功能来读取固定宽度的数据（例如`read.fwf()`）。然而，这种方法的一个令人沮丧的地方是，你必须指定这些宽度是什么。这可能会相当繁琐，尤其是如果你的数据集有很多列和/或很多行。然而，好处是文件可以稍微小一点，因为数据提供者不需要在分隔符上浪费字符。

在下面的示例中，我们指定了包含数字左侧空格的宽度。另一方面，如果我们指定`widths=c(2,2,4,4,1)`，这包括数字右侧的空格，那么列就会被识别为`字符`s。

```py
d <-  read.fwf("data/o-ring-erosion-only.data", 
 widths = c(1,2,3,4,3)) # or try c(2,2,4,4,1)
str(d)
## 'data.frame':    23 obs. of  5 variables:
##  $ V1: int  6 6 6 6 6 6 6 6 6 6 ...
##  $ V2: int  0 1 0 0 0 0 0 0 1 1 ...
##  $ V3: int  66 70 69 68 67 72 73 70 57 63 ...
##  $ V4: int  50 50 50 50 50 50 100 100 200 200 ...
##  $ V5: int  1 2 3 4 5 6 7 8 9 10 ...
```

如果你需要读取一些不具有表格结构的文本数据，那么你可能需要使用`readLines()`。这个函数将读取所有文本，将每一行分离成一个`字符`向量的元素，并且不会尝试将行解析为列。进一步的处理可以使用第 3.9 节中的技术完成。

```py
html_data <-  readLines("data/Google.html", warn = FALSE)
head(html_data, 1)
## [1] "<!DOCTYPE html>"
```

一些人在读取上面的数据时可能遇到了困难。这可能是因为你的机器的默认字符编码与我的不同。例如，如果你的字符编码是[“GBK”](https://en.wikipedia.org/wiki/GBK_(character_encoding))，那么你可能会收到一条警告信息，例如“在输入连接上找到无效输入。”这条信息意味着你的机器没有识别数据集中的一些字符。

这些错误很容易修复，所以不要担心。只需在读取数据的函数中指定一个编码参数即可。

```py
tmp <-  read.table("data/Google.html", sep = "~", 
 fileEncoding = "UTF-8") # makes errors disappear
```

## 9.3 使用 Pandas 读取文本文件

Pandas 可以读取多种不同的文件格式。[广泛的文件格式可以通过 Pandas 读取](https://pandas.pydata.org/pandas-docs/stable/reference/io.html)。在这里，我将只提及几个函数。

回想 R 中的`read.table()`和`read.csv()`，它们非常相似。在 Pandas 中，`pd.read_csv()`和`pd.read_table()`也有很多共同之处。它们的主要区别在于默认的列分隔符。

回想一下上面的 O-Ring 数据。列之间没有用逗号分隔，所以如果我们将其视为逗号分隔的文件，结果中的 Pandas `DataFrame`将丢失除一列之外的所有列。

```py
import pandas as pd
d = pd.read_csv("data/o-ring-erosion-only.data")
d.shape # one column and missing a row
## (22, 1)
d.columns # column labels are data
## Index(['6 0 66  50  1'], dtype='object')
```

默认情况下，`pd.read_csv()`期望列标签，这也是一个问题。与 R 不同，`header=`参数不需要是一个布尔值。你需要提供一个`None`。分隔符也需要恰到好处。

```py
pd.read_csv("data/o-ring-erosion-only.data", 
 header=None, sep = " ").head(2) # 1 space: no
##    0  1   2   3     4   5    6
## 0  6  0  66 NaN  50.0 NaN  1.0
## 1  6  1  70 NaN  50.0 NaN  2.0
pd.read_csv("data/o-ring-erosion-only.data", 
 header=None, sep = "\t").head(2) # tabs: no
##                0
## 0  6 0 66  50  1
## 1  6 1 70  50  2
pd.read_table("data/o-ring-erosion-only.data", 
 header=None).head(2) # default sep is tabs, so no
##                0
## 0  6 0 66  50  1
## 1  6 1 70  50  2
pd.read_csv("data/o-ring-erosion-only.data", 
 header=None, sep = "\s+").head(2) # 1 or more spaces: yes
##    0  1   2   3  4
## 0  6  0  66  50  1
## 1  6  1  70  50  2
```

以几乎与我们在 R 中做的方式读取固定宽度文件。以下是一个示例。

```py
d = pd.read_fwf("data/o-ring-erosion-only.data", 
 widths = [1,2,3,4,3], header=None) # try [2,2,4,4,1]
d.info()
## <class 'pandas.core.frame.DataFrame'>
## RangeIndex: 23 entries, 0 to 22
## Data columns (total 5 columns):
##  #   Column  Non-Null Count  Dtype
## ---  ------  --------------  -----
##  0   0       23 non-null     int64
##  1   1       23 non-null     int64
##  2   2       23 non-null     int64
##  3   3       23 non-null     int64
##  4   4       23 non-null     int64
## dtypes: int64(5)
## memory usage: 1.0 KB
```

如果你选择了 `widths=[2,2,4,4,1]`，那么尾随空格将导致 Pandas 识别一个 `dtype` 为 `object`。它没有被识别为字符串的原因是字符串可以有不同长度，所有字符串类型都指定了最大长度。如果你想强制最大长度，可能会有一些速度优势。在下面的例子中，我们使用 `d.astype()` 将两列的类型转换为 `pd.StringDtype`。

```py
d = pd.read_fwf("data/o-ring-erosion-only.data", 
 widths = [2,2,4,4,1], header=None)
list(d.dtypes)[:4]
## [dtype('int64'), dtype('int64'), dtype('O'), dtype('O')]
d = d.astype({2:'string', 3:'string'}) 
list(d.dtypes)[:4]
## [dtype('int64'), dtype('int64'), StringDtype, StringDtype]
```

就像在 R 中一样，你可能会遇到文件编码问题。例如，以下代码将无法运行，因为文件包含中文字符。如果你主要处理 UTF-8 文件，当你尝试运行以下代码时，你会收到一个 `UnicodeDecodeError`。

```py
pd.read_csv("data/message.txt")
```

然而，当你指定 `encoding="gbk"` 时，错误信息就会消失。¹⁵

```py
pd.read_csv("data/message.txt", encoding = "gbk")
## Empty DataFrame
## Columns: [恭喜发财]
## Index: []
```

你也可以使用 Python 读取非结构化、非表格数据。使用内置的 `open()` 函数以读取模式打开一个文件，然后使用 `f.readlines()` 返回一个字符串的 `list`。

```py
f = open("data/Google.html", "r")
d = f.readlines()
d[:1]
## ['<!DOCTYPE html>\n']
print(type(d), type(d[0]))
## <class 'list'> <class 'str'>
```

## 9.4 在 R 中保存数据

保存数据对于保存你的进度很重要。例如，有时运行执行数据清洗的脚本可能需要非常长的时间。保存你的进度可能会让你摆脱多次运行该脚本的责任。

在 R 中，有许多存储数据的选择。我将提到两个：将数据写入纯文本文件，以及保存序列化对象。

### 9.4.1 在 R 中写入表格纯文本数据

如果你想要将表格数据写入文本文件，请使用 `write.table()` 或 `write.csv()`。至少有两个参数你必须指定：第一个参数是你的 R 对象（通常是 `matrix` 或 `data.frame`），第二个参数是硬盘上的文件路径。

这里是一个将 `d` 写入名为 `"oring_out.csv"` 的文件的例子。我选择包含列名，但不包括行名。我还使用逗号分隔列。

```py
write.table(d, file = "data/oring_out.csv", 
 col.names = TRUE, row.names = FALSE, sep = ";")
```

上述代码不会在 R 控制台中打印任何内容，但我们可以使用文本编辑器查看硬盘上的原始文本文件。以下是前三行。

```py
"V1";"V2";"V3";"V4";"V5"
6;0;66;50;1
6;1;70;50;2
```

### 9.4.2 R 中的序列化

或者，你也可以选择以**序列化**的形式存储你的数据。采用这种方法，你仍然以更永久的方式将数据保存到硬盘上，但它以通常更节省内存的格式存储。

回想一下，写入数据的一个常见原因是保存你的进度。当你想要保存进度时，重要的是要问自己：“将我的进度保存为序列化对象，还是保存为原始文本文件更好？”

在做出这个决定时，请考虑*多功能性*。一方面，原始文本文件更灵活，可以在更多地方使用。另一方面，多功能性往往容易出错。

例如，假设你想保存一个清理过的`data.frame`。你确定你会记得将那一列字符串存储为`character`而不是`factor`吗？使用此`data.frame`的任何代码是否需要这一列以这种格式存在？

例如，让我们将对象`d`保存到名为`oring.rds`的文件中。

```py
saveRDS(d, file = "data/oring.rds")
rm(d)
exists("d")
## [1] FALSE
```

使用`saveRDS()`保存后，我们可以自由地使用`rm()`删除变量，因为稍后可以再次读取。为此，请调用`readRDS()`。这个文件有一个 R 能识别的特殊格式，因此你不需要担心从纯文本文件读取数据时通常遇到的任何问题。此外，`.rds`文件通常更小——`oring.rds`只有 248 字节，而`"oring_out.csv"`是 332 字节。

```py
d2 <-  readRDS(file = "data/oring.rds")
head(d2, 3)
##   V1 V2 V3 V4 V5
## 1  6  0 66 50  1
## 2  6  1 70 50  2
## 3  6  0 69 50  3
```

你也可以一次序列化多个对象！惯例规定这些文件以`.RData`后缀结尾。使用`save()`或`save.image()`保存整个全局环境，然后使用`load()`或`attach()`恢复。

```py
rm(list=ls()) # remove everything
a <-  1
b <-  2
save.image(file = "data/my-current-workspace.RData")
rm(list=ls()) 
load("data/my-current-workspace.RData")
ls() # print all objects in your workspace
## [1] "a" "b"
```

## 9.5 Python 中的数据保存

### 9.5.1 Python 中写入表格纯文本数据

你可以使用各种名为`to_*()`的`DataFrame`方法写出表格数据。[`to_*()`方法](https://pandas.pydata.org/pandas-docs/stable/reference/io.html#input-output)。`pd.DataFrame.to_csv()`与 R 中的`write.csv()`有很多相似之处。以下我们将`d`写入名为`oring_out2.csv`的文件。

```py
import pandas as pd
d = pd.read_csv("data/o-ring-erosion-only.data", 
 header=None, sep = "\s+")
d.to_csv("data/oring_out2.csv", 
 header=True, index=False, sep = ",")
```

在文本编辑器中，该文件的几行如下所示。

```py
0,1,2,3,4
6,0,66,50,1
6,1,70,50,2
```

### 9.5.2 Python 中的序列化

Python 中序列化功能很容易获得，就像在 R 中一样。在 Python 中，`pickle`和`cPickle`库可能是最常用的。使用这些库序列化对象被称为*序列化*对象。

Pandas 为每个`DataFrame`附加了一个`.to_pickle()`包装方法。[`.to_pickle()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_pickle.html)。一旦保存了序列化对象，就可以使用`pd.read_pickle()`[`.read_pickle()`](https://pandas.pydata.org/docs/reference/api/pandas.read_pickle.html#pandas.read_pickle)将其读回到 Python 中。这些函数非常方便，因为它们调用所有必要的`pickle`代码，并隐藏了大量复杂性。

这里是一个示例，展示了如何先写出`d`，然后读取回序列化的对象。在 Python 3 中，序列化对象的文件后缀通常是`.pickle`，但有许多其他选择。

```py
d.to_pickle("data/oring.pickle")
del d
d_is_back = pd.read_pickle("data/oring.pickle")
d_is_back.head(2)
##    0  1   2   3  4
## 0  6  0  66  50  1
## 1  6  1  70  50  2
```

不幸的是，`"oring.pickle"`比原始文本文件`"o-ring-erosion-only.data"`（322 字节）大得多（1,676 字节）。这是两个原因造成的。首先，原始数据集很小，因此序列化此对象的开销相对较大，其次，我们没有利用任何压缩。如果你使用类似`d_is_back.to_pickle("data/oring.zip")`的方法，它将变得更小。

在 Python 中，与 R 不同，将当前内存中的所有对象序列化更为困难。虽然可能实现，但可能需要使用第三方库。

谈到第三方代码，有许多提供 R 和 Python 中替代序列化解决方案的代码。在此文本中，我没有讨论任何一种。然而，我将提到其中一些可能提供以下组合：提高读写速度、减少所需内存、提高安全性¹⁶、提高人类可读性和多编程语言之间的互操作性。如果其中任何一项听起来可能有益，我鼓励您进行进一步的研究。

## 9.6 练习

### 9.6.1 R 问题

再次考虑名为 `"gspc.csv"` 的数据集，它包含 S&P500 指数的每日开盘价、最高价、最低价和收盘价。

1.  将此数据集作为 `data.frame` 读取，并命名为 `myData`。不要在作业提交中包含实现此功能的代码。

1.  将此对象写入为 `myData.rds`。完成后，从内存中删除 `myData`。不要在作业提交中包含实现此功能的代码。

1.  读取 `myData.rds`，并将变量存储为 `financialData`。*必须*在项目提交中包含实现此功能的代码。确保此代码假设 `myData.rds` 与代码文件 `io_lab.R` 在同一文件夹中。

### 9.6.2 Python 问题

我们将使用章节中提到的 `"Google.html"` 数据集。

1.  使用 `open()` 打开 `"Google.html"` 文件。将函数的输出存储为 `my_file`。

1.  使用文件的 `.readlines()` 方法将文件内容写入名为 `html_data` 的列表。

1.  将 `list` 强制转换为具有一个名为 `html` 的列的 `DataFrame`。

1.  创建一个名为 `nchars_ineach` 的 `Series`，用于存储文本每行的字符数。提示：`Series.str` 属性有很多有用的方法（[Series.str 属性有很多有用的方法](https://pandas.pydata.org/pandas-docs/stable/reference/series.html#api-series-str)）。

1.  创建一个类似于 `int` 的变量 `num_div_tags`，用于存储短语 “`<div>`” 在文件中出现的总次数。

再次考虑名为 `"gspc.csv"` 的数据集，它包含 S&P500 指数的每日开盘价、最高价、最低价和收盘价。

1.  将此数据集作为 `DataFrame` 读取，并命名为 `my_data`。不要在作业提交中包含实现此功能的代码。

1.  将此对象写入为 `"my_data.pickle"`。完成后，从内存中删除 `my_data`。不要在作业提交中包含实现此功能的代码。

1.  读取 `"my_data.pickle"`，并将变量存储为 `financial_data`。*必须*在项目提交中包含实现此功能的代码。确保此代码假设 `"my_data.pickle"` 与代码文件 `io_lab.py` 在同一文件夹中。

### 参考文献

Dua, Dheeru, 和 Casey Graff. 2017. “UCI 机器学习仓库.” 加州大学欧文分校，信息学院；计算机科学系. [`archive.ics.uci.edu/ml`](http://archive.ics.uci.edu/ml).

* * *

1.  使用文本编辑程序打开原始文本文件，而不是使用执行任何类型处理的程序。例如，如果您用 Microsoft Excel 打开它，数据的外观将发生变化，而且重要的信息，帮助您将数据读入 R 或 Python 的信息将无法供您使用。↑

1.  Python 内置的编码选项列表[在此处](https://docs.python.org/3/library/codecs.html#standard-encodings)提供。↑

1.  `pickle`的[文档](https://docs.python.org/2/library/pickle.html)提到，该库“不针对错误或恶意构造的数据安全”，并建议您“[永远不要从不受信任或未经认证的来源反序列化数据。”]↑
