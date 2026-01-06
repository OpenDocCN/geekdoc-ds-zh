# 第二章 基本类型

> 原文：[`randpythonbook.netlify.app/basic-types`](https://randpythonbook.netlify.app/basic-types)

在每种编程语言中，数据以不同的方式存储。编写操作数据的程序需要理解所有这些选择。这就是为什么我们必须关注我们在 R 和 Python 程序中的不同**数据类型**。不同的类型适用于不同的目的。

Python 和 R 的类型系统之间存在相似之处。然而，也存在许多差异。对这些差异要有准备。本章中的差异比上一章要多得多。

如果你不确定一个变量的类型，可以使用`type()`（在 Python 中）或`typeof()`（在 R 中）来查询它。

在这两种语言中，存储单个信息片段都很简单。然而，虽然 Python 有标量类型，但 R 在标量和复合类型之间并没有那么强的区分。

## 2.1 Python 中的基本类型

在 Python 中，我们最常用的简单类型是`str`（代表字符串）、`int`（代表整数）、`float`（代表浮点数）和`bool`（代表布尔值）。这个列表并不全面，但这些都是一个很好的起点。要获取 Python 中内置类型的完整列表，请点击[这里](https://docs.python.org/3/library/stdtypes.html)。

```py
print(type('a'), type(1), type(1.3))
## <class 'str'> <class 'int'> <class 'float'>
```

字符串用于处理文本数据，如人/地点/事物的名称和文本、推文和电子邮件等消息（Beazley 和 Kenneth Jones 2014）。如果你处理的是数字，如果你有一个可能有小数部分的数字，你需要浮点数；否则，你需要整数。布尔值在需要记录某事是真或假的情况下很有用。它们对于理解第十一部分中的控制流也很重要。

在下一节中，我们将讨论 Numpy 库。这个库包含了一个[更广泛的类型集合](https://numpy.org/doc/stable/user/basics.types.html)，它允许对任何你编写的脚本进行更精细的控制。

### 2.1.1 Python 中的类型转换

在 Python 程序中，我们经常需要在类型之间进行转换。这被称为**类型转换**，它可以隐式或显式地进行。

例如，`int`通常会被隐式转换为`float`，这样算术运算才能正常进行。

```py
my_int = 1
my_float = 3.2
my_sum = my_int + my_float
print("my_int's type", type(my_int))
## my_int's type <class 'int'>
print("my_float's type", type(my_float))
## my_float's type <class 'float'>
print(my_sum)
## 4.2
print("my_sum's type", type(my_sum))
## my_sum's type <class 'float'>
```

如果你总是依赖这种行为，你可能会感到失望。例如，在你的机器上尝试以下代码片段。你将收到以下错误：`TypeError: unsupported operand type(s) for +: 'float' and 'str'`。

```py
3.2 + "3.2"
```

当我们作为程序员明确要求 Python 执行转换时，会发生显式转换。我们将使用`int()`、`str()`、`float()`和`bool()`等函数来完成此操作。

```py
my_date = "5/2/2021"
month_day_year = my_date.split('/')
my_year = int(month_day_year[-1]) 
print('my_year is', my_year, 'and its type is', type(my_year))
## my_year is 2021 and its type is <class 'int'>
```

## 2.2 R 中的基本类型

在 R 中，基本类型的名称只有一点不同。它们是 `logical`（而不是 `bool`），`integer`（而不是 `int`），`double` 或 `numeric`（而不是 `float`)⁵，`character`（而不是 `str`），`complex`（用于涉及虚数的计算），和 `raw`（用于处理字节）。

```py
# cat() is kind of like print()
cat(typeof('a'), typeof(1), typeof(1.3))
## character double double
```

在这种情况下，R 自动将 `1` 升级为双精度浮点数。如果你想强制它成为整数，可以在数字的末尾添加一个大写的“L”。

```py
# cat() is kind of like print()
cat(typeof('a'), typeof(1L), typeof(1.3))
## character integer double
```

### 2.2.1 R 中的类型转换

你可以在 R 中显式和隐式地转换类型，就像你在 Python 中做的那样。隐式转换看起来像这样。

```py
myInt =  1
myDouble =  3.2
mySum =  myInt +  myDouble
print(paste0("my_int's type is ", typeof(myInt)))
## [1] "my_int's type is double"
print(paste0("my_float's type is ", typeof(myDouble)))
## [1] "my_float's type is double"
print(mySum)
## [1] 4.2
print(paste0("my_sum's type is ", typeof(mySum)))
## [1] "my_sum's type is double"
```

可以使用 `as.integer`, `as.logical`, `as.double` 等函数实现显式转换。

```py
print(typeof(1))
## [1] "double"
print(typeof(as.logical(1)))
## [1] "logical"
```

### 2.2.2 R 的简化

R 的基本类型与 Python 的基本类型略有不同。一方面，Python 有基本类型用于单个元素，并且它使用不同的类型作为存储多个元素的容器。另一方面，R 使用相同的类型来存储单个元素和存储多个元素。严格来说，R 没有标量类型。

从技术上讲，我们刚才在 R 中做的所有例子都使用了长度为 1 的 **向量**–`logical` `integer` `double`, `character`, `complex`, 和 `raw` 是向量的可能 **模式**。`vector`s 将在下一节 3 中进一步讨论。

考虑你更喜欢哪个选项。使用单独的类型对标量和集合有什么好处？使用相同类型有什么好处？

## 2.3 练习

### 2.3.1 R 问题

哪种 R 基础类型最适合每份数据？将你的答案分配给长度为四的 `character` `vector`，称为 `questionOne`。

1.  个人的 IP 地址

1.  个人是否参加了研究

1.  植物中发现的种子数量

1.  汽车在赛道上比赛所需的时间

浮点数很奇怪。打印出来的值和存储的值可能不同！在 R 中，你可以使用 `options` 函数来控制打印多少位数字。

1.  将 `2/3` 分配给 `a`

1.  `print` `a`，并将你看到的复制粘贴到变量 `aPrint` 中。确保它是一个 `character`。

1.  查看关于 `options` 的文档。将 `options()$digits` 的值分配给 `numDigitsStart`

1.  将数字位数更改为 `22`

1.  再次，`print`，`a`，并将你看到的复制粘贴到变量 `aPrintv2` 中。确保它是一个 `character`。

1.  将 `options()$digits` 的输出分配给 `numDigitsEnd`

浮点数很奇怪。存储的值可能不是你想要的。[“在 R 的数值类型中，唯一可以精确表示的数字是整数和分母为 2 的幂的分数。”](https://cran.r-project.org/doc/FAQ/R-FAQ.html#Why-doesn_0027t-R-think-these-numbers-are-equal_003f) 因此，你永远不应该在两个浮点数之间进行严格的相等性测试（即使用 `==`）。

1.  将 2 的平方根分配给 `mySqrt`

1.  打印这个变量的平方

1.  测试（使用 `==`）这个变量是否等于 `2`。将测试结果赋值给 `isTwoRecoverable`

1.  测试近似相等（使用 `all.equal`）。换句话说，检查这个变量是否非常接近 `2`。将测试结果赋值给 `closeEnough`。确保阅读该函数的文档，因为返回类型可能很复杂！

### 2.3.2 Python 问题

哪种 Python 类型最适合每份数据？将你的答案赋值给名为 `question_one` 的 `str`ings 列表。

1.  个人的 IP 地址

1.  个人是否参加了研究

1.  植物中发现的种子数量

1.  车在赛道上比赛所需的时间

浮点数很奇怪。打印出来的值和存储的值可能不一样！在 Python 中，如果你想控制用户定义的类型/类的打印位数，需要编辑类的 `__str__` 方法，但我们不会这么做。相反，我们将使用 `str.format()`（[`docs.python.org/3/library/stdtypes.html#str.format`](https://docs.python.org/3/library/stdtypes.html#str.format)）直接返回一个字符串（而不是复制粘贴）。

1.  将 `a` 赋值给 `2/3`

1.  打印 `a`，并将你看到的复制粘贴到变量 `a_print` 中

1.  创建一个显示 2/3 的 22 位数字的 `str`，命名为 `a_printv2`

1.  打印上面的字符串

浮点数很奇怪。存储的值可能不是你想要的。Python 文档中对存储行为可能令人惊讶的讨论非常出色。点击[这里](https://docs.python.org/3/tutorial/floatingpoint.html)阅读。

1.  将 2 的平方根赋值给 `my_sqrt`

1.  打印这个变量的平方

1.  测试（使用 `==`）这个变量是否等于 `2`。将测试结果赋值给 `is_two_recoverable`

1.  测试近似相等（使用 `np.isclose`，在运行 `import numpy as np` 后可用）。换句话说，检查这个变量是否接近 `2`。将测试结果赋值给 `close_enough`。

### 参考文献

Beazley, David M. 和 Brian K. (Brian Kenneth) Jones. 2014. *Python 烹饪书：精通 Python 3 的秘籍*。第三版。pub-ora-media:adr: pub-ora-media.

* * *

1.  “double”是“双精度浮点数”的简称。在其他编程语言中，程序员可以选择他或她想要的十进制精度位数。↩
