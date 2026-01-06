# 第五章 R 的列表与 Python 的列表和字典

> 原文：[`randpythonbook.netlify.app/rs-lists-versus-pythons-lists-and-dicts`](https://randpythonbook.netlify.app/rs-lists-versus-pythons-lists-and-dicts)

当你需要在一个容器中存储元素，但无法保证这些元素都具有相同的类型，或者无法保证它们都具有相同的尺寸时，那么在 R 中你需要一个 `list`。在 Python 中，你可能需要一个 `list` 或 `dict`（字典的简称）（Lutz 2013）。

## 5.1 R 中的 `list`s

`list`s 是 R 中最灵活的数据类型之一。你可以以许多不同的方式访问单个元素，每个元素可以有不同的尺寸，每个元素也可以是不同类型。

```py
myList <-  list(c(1,2,3), "May 5th, 2021", c(TRUE, TRUE, FALSE))
myList[1] # length-1 list; first element is length 3 vector
## [[1]]
## [1] 1 2 3
myList[[1]] # length-3 vector
## [1] 1 2 3
```

如果你想要提取一个元素，你需要决定是使用单方括号还是双方括号。前者返回一个 `list`，而后者返回单个元素的类型。

你也可以给列表的元素命名。这可以使代码更易读。为了了解为什么，请查看下面的示例，该示例使用了有关汽车的一些数据（“SAS ^ (Viya ^ (示例数据集）2021））。`lm()` 函数估计一个线性回归模型。它返回一个包含许多组件的 `list`。

```py
dataSet <-  read.csv("data/cars.csv")
results <-  lm(log(Horsepower) ~  Type, data = dataSet)
length(results)
## [1] 13
# names(results) # try this <-
results$contrasts
## $Type
## [1] "contr.treatment"
results['rank']
## $rank
## [1] 6
```

`results` 是一个 `list`（因为 `is.list(results)` 返回 `TRUE`），但更具体地说，它是一个类 `lm` 的 S3 对象。如果你不知道这是什么意思，不要担心！S3 类将在后面的章节中详细讨论。为什么这很重要呢？一方面，我提到这一点是为了防止你在输入 `class(results)` 时看到 `lm` 而不是 `list` 而感到困惑。其次，`lm()` 的作者编写了返回“花哨列表”的代码，这表明他们鼓励以另一种方式访问 `results` 的元素：使用专用函数！例如，你可以使用 `residuals(results)`、`coefficients(results)` 和 `fitted.values(results)`。这些函数并不适用于 R 中的所有列表，但当它们适用时（仅适用于 `lm` 和 `glm` 对象），你可以确信你正在编写 `lm()` 作者鼓励的代码。

## 5.2 Python 中的 `list`s

[Python `list`s](https://docs.python.org/3/library/stdtypes.html#lists) 同样非常灵活。在 Python 中访问和修改列表元素的选择较少——你很可能会使用方括号操作符。元素可以是不同的大小和类型，就像 R 的列表一样。

然而，与 R 不同，你无法给列表的元素命名。如果你想一个允许你通过名称访问元素的容器，请查看 Python 的 [字典](https://docs.python.org/3/library/stdtypes.html#mapping-types-dict)（见第 5.3 节）或 Pandas 的 `Series` 对象（见第 3.2 节）。

从下面的示例中，你可以看到我们已经介绍了列表。我们已经从它们构建了 Numpy 数组。

```py
import numpy as np
another_list = [np.array([1,2,3]), "May 5th, 2021", True, [42,42]]
another_list[2]
## True
another_list[2] = 100
another_list
## [array([1, 2, 3]), 'May 5th, 2021', 100, [42, 42]]
```

Python 列表附带了[方法](https://docs.python.org/3/tutorial/datastructures.html#more-on-lists)，这些方法可能非常有用。

```py
another_list
## [array([1, 2, 3]), 'May 5th, 2021', 100, [42, 42]]
another_list.append('new element')
another_list
## [array([1, 2, 3]), 'May 5th, 2021', 100, [42, 42], 'new element']
```

可以像上面那样创建列表，使用方括号运算符。它们也可以使用 `list()` 函数和创建一个 *列表推导式(list comprehension)* 来创建。列表推导式将在 11.2 中进一步讨论。

```py
my_list = list(('a','b','c')) # converting a tuple to a list
your_list = [i**2 for i in range(3)] # list comprehension
my_list
## ['a', 'b', 'c']
your_list
## [0, 1, 4]
```

上面的代码引用了一个在此文本中未广泛讨论的类型：[`元组(tuple)`](https://docs.python.org/3.3/library/stdtypes.html?highlight=tuple#tuple)。

## 5.3 Python 中的字典

[**字典**](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) 在 Python 中提供了一个键值对的容器。键是 *唯一的*，并且它们必须是 *不可变的*。`字符串(string)` 是最常见的键类型，但也可以使用 `整数(int)`。

这是一个使用大括号（即 `{}`）创建 `字典(dict)` 的示例。这个 `字典(dict)` 存储了一些流行加密货币的当前价格。使用键访问单个元素的价值是通过方括号运算符（即 `[]`）完成的，删除元素是通过 `del` 关键字完成的。

```py
crypto_prices = {'BTC': 38657.14, 'ETH': 2386.54, 'DOGE': .308122}
crypto_prices['DOGE'] # get the current price of Dogecoin
## 0.308122
del crypto_prices['BTC'] # remove the current price of Bitcoin
crypto_prices.keys()
## dict_keys(['ETH', 'DOGE'])
crypto_prices.values()
## dict_values([2386.54, 0.308122])
```

你也可以使用 **字典推导式(dictionary comprehensions)** 来创建 `字典(dict)`。就像列表推导式一样，这些将在 11.2 中进一步讨论。

```py
incr_cryptos = {key:val*1.1 for (key,val) in crypto_prices.items()}
incr_cryptos
## {'ETH': 2625.194, 'DOGE': 0.3389342}
```

个人来说，我不像使用列表那样经常使用字典。如果我有一个字典，我通常会将其转换为 Pandas 数据框（关于这些的更多信息请参阅 8.2）。

```py
import pandas as pd
a_dict = { 'col1': [1,2,3], 'col2' : ['a','b','c']}
df_from_dict = pd.DataFrame(a_dict)
df_from_dict
##    col1 col2
## 0     1    a
## 1     2    b
## 2     3    c
```

## 5.4 练习

### 5.4.1 R 问题

考虑数据集 `"adult.data"`、`"car.data"`、`"hungarian.data"`、`"iris.data"`、`"long-beach-va.data"` 和 `"switzerland.data"`（Janosi 等人 1988)、（Fisher 1988)、（“Adult” 1996）和（“Car Evaluation” 1997）由（Dua 和 Graff 2017）托管。读取所有这些并将它们全部存储为一个 `data.frame`s 的 `列表(list)`。将这个列表命名为 `listDfs`。

下面是 R 中的两个列表：

```py
l1 <-  list(first="a", second=1)
l2 <-  list(first=c(1,2,3), second = "statistics")
```

1.  创建一个新的 `列表(list)`，它是上面两个列表“挤压在一起”的结果。它必须有长度 $4$，并且每个元素是 `l1` 和 `l2` 的元素之一。将这个列表命名为 `l3`。确保删除这四个元素的“标签”或“名称”。

1.  将 `l3` 的第三个元素提取为一个长度为一的 `列表(list)` 并将其分配给名称 `l4`。

1.  将 `l3` 的第三个元素提取为一个 `向量(vector)` 并将其分配给名称 `v1`。

### 5.4.2 Python 问题

使用 `pd.read_csv()` 读取 `car.data`，并使用 `DataFrame` 方法将其转换为 `dict`。将你的答案存储为 `car_dict`。

下面是 Python 中的两个 `字典(dict)`：

```py
d1 = { "first" : "a", "second" : 1}
d2 = { "first" : [1,2,3], "second" : "statistics"}
```

1.  创建一个新的 `列表(list)`，它是上面两个 `字典(dict)`“挤压在一起”的结果（为什么不能是另一个 `字典(dict)`？）。它必须有长度 $4$，并且每个值是 $d1$ 和 $d2$ 的值之一。将这个列表命名为 `my_list`。

1.  使用列表推导式创建一个名为 `special_list` 的列表，包含从零开始到（包括）一百万的所有数字，但不包括能被小于七的任何质数整除的数字。

1.  将上述列表中所有元素的平均值赋值给变量 `special_ave`。

### 参考文献

“成人。” 1996\. UCI 机器学习库。

“汽车评估。” 1997\. UCI 机器学习库。

Dua, Dheeru, and Casey Graff. 2017\. “UCI 机器学习库。” 加州大学欧文分校，信息与计算机科学学院。 [`archive.ics.uci.edu/ml`](http://archive.ics.uci.edu/ml).

Fisher, Test, R.A. & Creator. 1988\. “鸢尾花。” UCI 机器学习库。

Janosi, Andras, William Steinbrunn, Matthias Pfisterer, and Robert Detrano. 1988\. “心脏病。” UCI 机器学习库。

Lutz, Mark. 2013\. *《Python 学习》*. 第 5 版. 北京: O’Reilly. [`www.safaribooksonline.com/library/view/learning-python-5th/9781449355722/`](https://www.safaribooksonline.com/library/view/learning-python-5th/9781449355722/).

“SAS^(Viya^(示例数据集)。” 2021\. [`support.sas.com/documentation/onlinedoc/viya/examples.htm`](https://support.sas.com/documentation/onlinedoc/viya/examples.htm).))
