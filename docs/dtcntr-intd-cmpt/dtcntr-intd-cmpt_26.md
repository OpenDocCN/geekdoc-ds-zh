# 9.1 从 Pyret 到 Python🔗

> 原文：[`dcic-world.org/2025-08-27/intro-python.html`](https://dcic-world.org/2025-08-27/intro-python.html)

|   9.1.1 表达式、函数和类型 |
| --- |
|   9.1.2 从函数返回值 |
|   9.1.3 示例和测试用例 |
|   9.1.4 关于数字的旁白 |
|   9.1.5 条件语句 |
|   9.1.6 创建和处理列表 |
|   9.1.6.1 过滤器、映射和朋友们 |
|   9.1.7 带有组件的数据 |
|   9.1.7.1 访问数据类中的字段 |
|   9.1.8 遍历列表 |
|   9.1.8.1 介绍 `For` 循环 |
|   9.1.8.2 关于处理列表元素顺序的旁白 |
|   9.1.8.3 在生成列表的函数中使用 `For` 循环 |
|   9.1.8.4 总结：Python 的列表处理模板 |
|   9.1.8.5 Pyret 中的 `for each` 循环 |
|   9.1.8.5.1 可变的变量 |
|   9.1.8.5.2 块注释 |
|   9.1.8.5.3 `for each` 如何工作 |
|   9.1.8.5.4 测试和可变的变量 |

通过到目前为止在 Pyret 的工作，我们已经涵盖了几个核心编程技能：如何处理表格，如何设计好的示例，创建数据类型的基础，以及如何使用函数、条件语句和重复（通过 `filter` 和 `map` 以及递归）等基本计算构建块。你已经拥有了一个坚实的初始工具包，以及一个广阔的世界，其中充满了其他可能的程序！

但我们将暂时改变方向，向你展示如何在 Python 中工作。为什么？

观察相同的概念如何在多种语言中体现，可以帮助你区分核心计算思想与特定语言的符号和习惯用法。如果你计划将编程作为你的职业工作的一部分，你不可避免地会在不同的时间使用不同的语言：我们给你一个机会，在一个受控和温和的环境中练习这项技能。

为什么我们称之为温和的？因为 Pyret 中的符号部分是为了这种过渡而设计的。你会在符号层面上发现 Pyret 和 Python 之间的许多相似之处，同时也有一些有趣的不同之处，这些不同之处突出了语言背后的哲学差异。我们接下来想要编写的下一组程序（特别是，数据丰富的程序，其中数据必须随着时间的推移进行更新和维护）与 Python 的某些特性非常契合，这些特性你在 Pyret 中还没有看到。未来的版本将包含对比两种语言优缺点的内容。

我们通过将一些早期的代码示例重做在 Python 中，来强调 Pyret 和 Python 之间的基本符号差异。

#### 9.1.1 表达式、函数和类型🔗 "链接至此")

在函数练习：笔的成本中，我们通过计算一盒笔的成本的例子介绍了函数和类型的符号。一盒笔包括笔的数量和要打印在笔上的信息。每支笔的成本是 25 美分，加上每字符 2 美分的消息费用。以下是原始的 Pyret 代码：

```py
fun pen-cost(num-pens :: Number, message :: String) -> Number:
  doc: ```每支笔的总成本，每支 25 美分

        加上每条消息字符 2 美分```py
  num-pens * (0.25 + (string-length(message) * 0.02))
end
```

下面是相应的 Python 代码：

```py
def pen_cost(num_pens: int, message: str) -> float:
    """total cost for pens, each at 25 cents plus
       2 cents per message character"""
    return num_pens * (0.25 + (len(message) * 0.02))
```

> 现在行动！
> 
> > 你在两个版本之间看到了哪些符号差异？

下面是差异的总结：

+   Python 使用 `def` 而不是 `fun`。

+   Python 在名称中使用下划线（如 `pen_cost`），而不是 Pyret 中的连字符。

+   类型名称的写法不同：Python 使用 `str` 和 `int` 而不是 `String` 和 `Number`。此外，Python 在类型之前只使用一个冒号，而 Pyret 使用两个冒号。

+   Python 为不同类型的数字有不同的类型：`int` 用于整数，而 `float` 用于小数。Pyret 只使用一个类型（`Number`）来表示所有数字。

+   Python 不对文档字符串进行标记（如 Pyret 中的 `doc:` 所做的那样）。

+   Python 中没有 `end` 注解。相反，Python 使用缩进来定位 if/else 语句、函数或其他多行结构的结束。

+   Python 使用 `return` 标记函数的输出。

这些是符号上的细微差异，随着你在 Python 中编写更多的程序，你会习惯这些差异。

除了符号差异之外，还有一些其他差异。这个示例程序中出现的差异与语言如何使用类型有关。在 Pyret 中，如果你对一个参数添加了类型注解，然后传递了一个不同类型的值，你会得到一个错误信息。Python 忽略类型注解（除非你引入了额外的工具来检查类型）。Python 的类型就像程序员的笔记，但在程序运行时并不强制执行。

> 练习
> 
> > 将以下 `moon-weight` 函数从函数练习：月球重量转换为 Python：
> > 
> > ```py
> > fun moon-weight(earth-weight :: Number) -> Number:
> >   doc:" Compute weight on moon from weight on earth"
> >   earth-weight * 1/6
> > end
> > ```

#### 9.1.2 从函数返回值🔗 "链接至此")

在 Pyret 中，函数体由可选的命名中间值的语句组成，后面跟一个单独的表达式。该单个表达式的值是调用函数的结果。在 Pyret 中，每个函数都会产生一个结果，因此不需要标记结果来自何处。

正如我们将看到的，Python 与之不同：并非所有“函数”都会返回结果（注意名称从 `fun` 更改为 `def`）。在数学中，函数的定义就是有结果。程序员有时会在“函数”和“过程”这两个术语之间进行区分：两者都指参数化计算，但只有前者会向周围计算返回结果。然而，一些程序员和语言更宽松地使用“函数”这个术语来涵盖这两种参数化计算。此外，结果不一定是 `def` 的最后一个表达式。在 Python 中，关键字 `return` 明确标记了其值作为函数结果的那个表达式。

> 现在就做！
> 
> > 将这两个定义放入一个 Python 文件中。
> > 
> > ```py
> > def add1v1(x: int) -> int:
> >     return x + 1
> > 
> > def add1v2(x: int) -> int:
> >     x + 1
> > ```
> > 
> > 在 Python 提示符下，依次调用每个函数。你注意到使用每个函数的结果有什么不同吗？

希望你能注意到，使用 `add1v1` 在提示符后显示答案，而使用 `add1v2` 则不会。这种差异对函数的组合有影响。

> 现在就做！
> 
> > 尝试在 Python 提示符下评估以下两个表达式：每种情况下会发生什么？
> > 
> > `3 * add1v1(4)`
> > 
> > `3 * add1v2(4)`

这个例子说明了为什么 `return` 在 Python 中是必不可少的：没有它，就不会返回任何值，这意味着你无法在另一个表达式中使用函数的结果。那么 `add1v2` 又有什么用呢？保留这个问题；我们将在 Mutating Variables 中回到它。

#### 9.1.3 示例和测试用例🔗 "链接至此")

在 Pyret 中，我们为每个函数都提供了使用 `where:` 块的示例。我们还能够编写 `check:` 块来进行更广泛的测试。作为提醒，以下是包含 `where:` 块的 `pen-cost` 代码：

```py
fun pen-cost(num-pens :: Number, message :: String) -> Number:
  doc: ```每支钢笔的总成本，每支 25 美分

    每条消息字符加 2 美分```py
  num-pens * (0.25 + (string-length(message) * 0.02))
where:
  pen-cost(1, "hi") is 0.29
  pen-cost(10, "smile") is 3.50
end
```

Python 没有关于 `where:` 块的概念，也没有示例和测试之间的区别。Python 有几个不同的测试包；在这里，我们将使用 `pytest`，这是一个标准的轻量级框架，类似于我们在 Pyret 中所做的测试形式。如何设置 pytest 和你的测试文件内容将根据你的 Python IDE 而有所不同。我们假设讲师将提供与他们的工具选择一致的单独说明。为了使用 `pytest`，我们将示例和测试放在一个单独的函数中。以下是对 `pen_cost` 函数的示例：

```py
import pytest

def pen_cost(num_pens: int, message: str) -> float:
    """total cost for pens, each at 25 cents plus
       2 cents per message character"""
    return num_pens * (0.25 + (len(message) * 0.02))

def test_pens():
  assert pen_cost(1, "hi") == 0.29
  assert pen_cost(10, "smile") == 3.50
```

关于此代码的注意事项：

+   我们已经导入了 `pytest`，这是一个轻量级的 Python 测试库。

+   这些例子已经移动到一个函数中（这里 `test_pens`），该函数不接受任何输入。请注意，包含测试用例的函数名称必须以 `test_` 开头，以便 `pytest` 能够找到它们。

+   在 Python 中，单个测试的形式是

    ```py
    assert EXPRESSION == EXPECTED_ANS
    ```

    而不是 Pyret 中的 `is` 形式。

> 现在行动起来！
> 
> > 在 Python 代码中添加一个额外的测试，对应于 Pyret 测试
> > 
> > ```py
> > pen-cost(3, "wow") is 0.93
> > ```
> > 
> > 确保运行测试。
> > 
> 现在行动起来！
> 
> > 你真的尝试运行测试了吗？

哇！发生了一些奇怪的事情：测试失败了。停下来想想：在 Pyret 中工作的同一个测试在 Python 中失败了。这是怎么回事？

#### 9.1.4 关于数字的旁白🔗 "链接至此")

结果表明，不同的编程语言在如何表示和管理实数（非整数）方面做出了不同的决策。有时，这些表示方式上的差异会导致计算值中细微的定量差异。作为一个简单的例子，让我们看看两个看似简单的实数 `1/2` 和 `1/3`。以下是我们在 Pyret 提示符中输入这两个数字时得到的结果：

|

&#124;

> ```py
> 1/2
> ```

&#124;

|

|

```py
0.5
```

|

|

&#124;

> ```py
> 1/3
> ```

&#124;

|

|

```py
0.3
```

|

如果我们在 Python 控制台中输入这两个相同的数字，我们得到的结果是：

|

&#124;

> ```py
> 1/2
> ```

&#124;

|

|

```py
0.5
```

|

|

&#124;

> ```py
> 1/3
> ```

&#124;

|

|

```py
0.3333333333333333
```

|

注意，对于 `1/3`，答案看起来不同。你可能（或者可能没有！）从之前的数学课程中回忆起来，`1/3` 是一个非终止、循环小数的例子。简单来说，如果我们尝试用十进制形式写出 `1/3` 的确切值，我们需要写出无限序列的 `3`。数学家通过在 `3` 上放置一个水平线来表示这一点。这就是我们在 Pyret 中看到的表示法。相比之下，Python 写出 `3` 的部分序列。

在这个区别之下，隐藏着一些关于计算机中数字表示的有趣细节。计算机没有无限的空间来存储数字（或者任何其他东西）：当程序需要与非终止小数一起工作时，底层语言可以选择：

+   近似这个数字（通过在某个点切断无限位数的序列），然后只使用近似值继续工作，或者

+   存储有关数字的额外信息，这可能允许以后用更精确的计算来使用它（尽管总有一些数字无法在有限空间中精确表示）。

Python 采用第一种方法。因此，使用近似值进行的计算有时会产生近似的结果。这就是我们新的 `pen_cost` 测试用例发生的情况。从数学上讲，计算应该得到 `0.93`，但近似值却得到 `0.9299999999999999`。

那么，在这种情况下我们如何编写测试？我们需要告诉 Python，答案应该是“接近” `0.93`，在近似的误差范围内。下面是这个样子的：

```py
assert pen_cost(3, "wow") == pytest.approx(0.93)
```

我们将我们想要的精确答案包装在 `pytest.approx` 中，以表示我们将接受任何接近我们指定的值的答案。如果您想控制小数点的精度位数，可以这样做，但默认的 `± 2.3e-06` 通常足够了。

#### 9.1.5 条件语句🔗 "链接到这里")

继续我们的原始 `pen_cost` 示例，以下是计算订单运费的函数的 Python 版本：

```py
def add_shipping(order_amt: float) -> float:
    """increase order price by costs for shipping"""
    if order_amt == 0:
      return 0
    elif order_amt <= 10:
      return order_amt + 4
    elif (order_amt > 10) and (order_amt < 30):
      return order_amt + 8
    else:
      return order_amt + 12
```

这里要注意的主要区别是，Python 中将 `else if` 写作单词 `elif`。我们使用 `return` 来标记每个条件分支中的函数结果。否则，两种语言的条件结构相当相似。

您可能已经注意到 Python 不需要在 `if` 表达式或函数上显式使用 `end` 注解。相反，Python 会查看您的代码缩进来确定何时结束一个结构。例如，在 `pen_cost` 和 `test_pens` 的代码示例中，Python 确定函数 `pen_cost` 已经结束，因为它在程序文本的左侧检测到一个新的定义（对于 `test_pens`）。同样的原则也适用于结束条件。

我们将回到关于缩进的问题，并在我们更多地使用 Python 时看到更多示例。

#### 9.1.6 创建和处理列表🔗 "链接到这里")

作为列表的一个例子，让我们假设我们一直在玩一个涉及从一组字母中制作单词的游戏。在 Pyret 中，我们可以这样编写一个示例单词列表：

```py
words = [list: "banana", "bean", "falafel", "leaf"]
```

在 Python 中，这个定义看起来是这样的：

```py
words = ["banana", "bean", "falafel", "leaf"]
```

这里的唯一区别是 Python 不使用 Pyret 中需要的 `list:` 标签。

##### 9.1.6.1 过滤器、映射和朋友们🔗 "链接到这里")

当我们第一次学习 Pyret 中的列表时，我们从常见的内置函数开始，例如 `filter`、`map`、`member` 和 `length`。我们还看到了使用 `lambda` 来帮助我们简洁地使用这些函数的例子。这些相同的函数，包括 `lambda`，也存在于 Python 中。以下是一些示例（`#` 是 Python 中的注释字符）：

```py
words = ["banana", "bean", "falafel", "leaf"]

# filter and member
words_with_b = list(filter(lambda wd: "b" in wd, words))
# filter and length
short_words = list(filter(lambda wd: len(wd) < 5, words))
# map and length
word_lengths = list(map(len, words))
```

注意，您必须使用 `list()` 的方式来包装对 `filter`（和 `map`）的调用。内部，Python 有这些函数返回我们尚未讨论（也不需要）的数据类型。使用 `list` 将返回的数据转换为列表。如果您省略 `list`，您将无法将某些函数链接在一起。例如，如果我们试图在不首先将其转换为 `list` 的情况下计算 `map` 的结果长度，我们会得到一个错误：

|

&#124;

> ```py
> len(map(len,b))
> ```

&#124;

|

|

```py
TypeError: object of type 'map' has no len()
```

|

如果这个错误信息现在没有意义，请不要担心（我们还没有学习什么是“对象”）。关键是，如果您在使用 `filter` 或 `map` 的结果时看到这样的错误，您可能忘记将结果包装在 `list` 中。

> 练习
> 
> > 通过编写以下问题的表达式来练习 Python 的列表函数。请只使用我们迄今为止向您展示的列表函数。
> > 
> > +   给定一个数字列表，根据每个数字的符号将其转换为`"pos"`、`"neg"`、`"zero"`字符串列表。
> > +   
> > +   给定一个字符串列表，是否有任何字符串的长度等于 5？
> > +   
> > +   给定一个数字列表，从该列表中生成一个包含 10 到 20 之间偶数的列表。

我们故意专注于使用 Python 内置函数处理计算，而不是向您展示如何编写自己的函数（正如我们在 Pyret 中处理递归那样）。虽然您可以在 Pyret 中编写处理列表的递归函数，但用于此目的的另一种程序风格更为传统。我们将在关于可变变量的章节中探讨这一点。

#### 9.1.7 数据组件🔗 "链接到此处")

一个类似于 Pyret 数据定义（没有变体）的称为 Python 中的数据类。那些熟悉 Python 的人可能会想知道为什么我们使用数据类而不是字典或原始类。与字典相比，数据类允许使用类型提示并捕获我们的数据具有固定字段集合的事实。与原始类相比，数据类生成大量的样板代码，这使得它们比原始类更轻量级。以下是一个 Pyret 中的待办事项列表数据类型及其对应的 Python 代码示例：

```py
# a todo item in Pyret
data ToDoItemData:
  | todoItem(descr :: String,
             due :: Date,
             tags :: List<String>
end
```

```py
------------------------------------------
# the same todo item in Python

# to allow use of dataclasses
from dataclasses import dataclass
# to allow dates as a type (in the ToDoItem)
from datetime import date

@dataclass
class ToDoItem:
    descr: str
    due: date
    tags: list

# a sample list of ToDoItem
myTD = [ToDoItem("buy milk", date(2020, 7, 27), ["shopping", "home"]),
        ToDoItem("grade hwk", date(2020, 7, 27), ["teaching"]),
        ToDoItem("meet students", date(2020, 7, 26), ["research"])
       ]
```

注意事项：

+   类型及其构造函数有一个单独的名称，而不是像我们在 Pyret 中那样有单独的名称。

+   字段名之间没有逗号（但在 Python 中每个字段必须单独占一行）

+   在 Python 中，无法指定列表内容的类型（至少，不使用更高级的包来编写类型）

+   在`class`之前需要`@dataclass`注解。

+   数据类不支持创建具有多个变体的数据类型，就像我们在 Pyret 中经常做的那样。这样做需要比我们在这本书中将要涵盖的更高级的概念。

##### 9.1.7.1 在数据类中访问字段🔗 "链接到此处")

在 Pyret 中，我们通过使用点（`.`）来“挖掘”数据并访问字段来从结构化数据中提取字段。相同的符号在 Python 中同样适用：

|

> ```py
> travel = ToDoItem("buy tickets", date(2020, 7, 30), ["vacation"])
> ```

|

|

&#124;

> ```py
> travel.descr
> ```

&#124;

|

|

```py
"buy tickets"
```

|

#### 9.1.8 遍历列表🔗 "链接到此处")

##### 9.1.8.1 介绍 `For` 循环🔗 "链接到此处")

在 Pyret 中，我们通常编写递归函数来计算列表上的汇总值。作为提醒，这里有一个 Pyret 函数，用于计算列表中的数字之和：

```py
fun sum-list(numlist :: List<Number>) -> Number:
  cases (List) numlist:
    | empty => 0
    | link(fst, rst) => fst + sum-list(rst)
  end
end
```

在 Python 中，将列表分解为其第一个和其余部分并递归处理其余部分是不常见的。相反，我们使用一个称为`for`的构造来依次访问列表中的每个元素。以下是一个使用具体（示例）奇数列表的`for`的形式：

```py
for num in [5, 1, 7, 3]:
   # do something with num
```

这里的 `num` 名称是我们自己选择的，就像 Pyret 中函数参数的名称一样。当 `for` 循环评估时，列表中的每个项目依次被称为 `num`。因此，这个 `for` 示例等同于编写以下内容：

```py
# do something with 5
# do something with 1
# do something with 7
# do something with 3
```

`for` 构造节省了我们多次编写常见代码的时间，并且处理了我们正在处理的列表可以是任意长度的事实（因此我们无法预测需要编写多少次常见代码）。

现在我们使用 `for` 循环来计算列表的累加和。我们首先通过我们的具体列表再次确定重复的计算。首先，让我们用散文的形式表达重复的计算。在 Pyret 中，我们的重复计算是这样的：“将第一个元素加到其余元素的和上”。我们已经说过，在 Python 中我们无法轻松访问“其余的元素”，因此我们需要重新表述。这里有一个替代方案：

```py
# set a running total to 0
# add 5 to the running total
# add 1 to the running total
# add 7 to the running total
# add 3 to the running total
```

注意，这个框架指的是“剩余的计算”，而不是到目前为止已经发生的计算（“累加总合”）。如果你恰好处理了关于 `my-running-sum` 的章节（我的累加和：示例和代码），这个框架可能很熟悉。

让我们将这个散文草图转换为代码，通过将草图中的每一行替换为具体的代码来实现。我们通过设置一个名为 `run_total` 的变量并更新其值来做到这一点，对于列表中的每个元素都这样做。

```py
run_total = 0
run_total = run_total + 5
run_total = run_total + 1
run_total = run_total + 7
run_total = run_total + 3
```

这种可以给现有变量名赋予新值的思想是我们之前没有见过的。事实上，当我们第一次看到如何命名值（在程序目录）时，我们明确表示 Pyret 不允许这样做（至少，不是用我们向您展示的构造）。Python 可以。我们将在不久的将来更深入地探讨这种能力的影响（在修改变量）。现在，让我们利用这种能力，这样我们可以学习遍历列表的模式。首先，让我们将重复的代码行合并为单个 `for` 循环的使用：

```py
run_total = 0
for num in [5, 1, 7, 3]:
   run_total = run_total + num
```

这段代码对于特定的列表来说运行良好，但我们的 Pyret 版本将求和的列表作为函数的参数。为了在 Python 中实现这一点，我们像本章前面其他示例中那样将 `for` 循环包裹在函数中。这就是最终版本。

```py
def sum_list(numlist : list) -> float:
    """sum a list of numbers"""
    run_total = 0
    for num in numlist:
        run_total = run_total + num
    return(run_total)
```

> 立即行动！
> 
> > 为 `sum_list`（Python 版本）编写一组测试用例。

现在 Python 版本已经完成，让我们将其与原始的 Pyret 版本进行比较：

```py
fun sum-list(numlist :: List<Number>) -> Number:
  cases (List) numlist:
    | empty => 0
    | link(fst, rst) => fst + sum-list(rst)
  end
end
```

下面是关于这两段代码的一些需要注意的事项：

+   Python 版本需要一个变量（这里 `run_total`）来保存我们在遍历（处理）列表时逐步构建的计算结果。

+   该变量的初始值是我们之前在 Pyret 中的 `empty` 情况下返回的答案。

+   Pyret 函数的 `link` 情况中的计算用于在 `for` 循环体中更新该变量。

+   在`for`完成处理列表中的所有项目后，Python 版本将变量中的值作为函数的结果返回。

##### 9.1.8.2 关于处理列表元素顺序的旁白🔗 "链接至此")

如果考虑两个程序的运行方式，这里还有一个细微之处：Python 版本从左到右求和元素，而 Pyret 版本从右到左求和。具体来说，`run_total`的值序列是这样计算的：

```py
run_total = 0
run_total = 0 + 5
run_total = 5 + 1
run_total = 6 + 7
run_total = 13 + 3
```

相比之下，Pyret 版本展开如下：

```py
sum_list([list: 5, 1, 7, 3])
5 + sum_list([list: 1, 7, 3])
5 + 1 + sum_list([list: 7, 3])
5 + 1 + 7 + sum_list([list: 3])
5 + 1 + 7 + 3 + sum_list([list:])
5 + 1 + 7 + 3 + 0
5 + 1 + 7 + 3
5 + 1 + 10
5 + 11
16
```

作为提醒，Pyret 版本这样做是因为在`link`情况下，`+`只能在一次计算完其余列表的总和后才能减少到答案。即使我们作为人类看到 Pyret 展开的每一行中的`+`操作链，Pyret 也只看到表达式`fst + sum-list(rst)`，这要求函数调用在`+`执行之前完成。

在求和列表的情况下，我们不会注意到两个版本之间的区别，因为无论我们是左到右还是右到左计算，总和都是相同的。在我们编写的其他函数中，这种差异可能开始变得重要。

##### 9.1.8.3 在生成列表的函数中使用`For`循环🔗 "链接至此")

让我们在另一个函数上练习使用`for`循环，这个函数生成一个列表。具体来说，让我们编写一个程序，它接受一个字符串列表，并生成一个包含列表中包含字母`"z"`的单词的列表。

正如我们的`sum_list`函数一样，我们需要一个变量来存储我们在构建过程中得到的列表。以下代码调用这个`zlist`。代码还展示了如何使用`in`来检查一个字符是否在字符串中（它也适用于检查一个项目是否在列表中）以及如何将一个元素添加到列表的末尾（`append`）。

```py
def all_z_words(wordlist : list) -> list:
    """produce list of words from the input that contain z"""
    zlist = [] # start with an empty list
    for wd in wordlist:
        if "z" in wd:
            zlist = [wd] + zlist
    return(zlist)
```

这段代码遵循`sum_list`的结构，即我们使用类似于在 Pyret 中会使用的表达式来更新`zlist`的值。对于那些有先前的 Python 经验，在这里会使用`zlist.append`的人，请记住这个想法。我们将在可变列表中达到那里。

> 练习
> 
> > 为`all_z_words`编写测试。
> > 
> 练习
> 
> > 使用`filter`编写`all_z_words`的第二个版本。务必为其编写测试！
> > 
> 练习
> 
> > 对比这两个版本及其相应的测试。你注意到什么有趣的地方了吗？

##### 9.1.8.4 总结：Python 的列表处理模板🔗 "链接至此")

正如我们在 Pyret 中编写列表处理函数的模板一样，Python 中也有一个基于`for`循环的相应模板。作为提醒，该模式如下：

```py
def func(lst: list):
  result = ...  # what to return if the input list is empty
  for item in lst:
    # combine item with the result so far
    result = ... item ... result
  return result
```

在学习如何在 Python 中编写列表上的函数时，请记住这个模板。

##### 9.1.8.5 Pyret 中的`for each`循环🔗 "链接至此")

本节可以独立阅读，无需阅读本章的其余部分，所以如果你在介绍 Python 之前被引导到这里，不要担心！虽然下面的内容与 Python 中存在的类似结构相似，但它是在其自身中引入的。

前几节介绍了 Python 中的`for`循环，并展示了使用它们处理列表的模板。Pyret 可以使用以下模式做类似的事情：

```py
fun func(lst :: List) block:
  var result = ...  # what to return if the input list is empty
  for each(item from lst):
    # combine item with the result so far
    result := ... item ... result
  end
  result
end
```

在这个例子中使用了几个新的语言特性，这些特性将在以下几节中介绍。

##### 9.1.8.5.1 可以更改的变量🔗 "链接至此")

首先，请注意，我们使用`var result`引入变量`result`——这意味着它可以变化，这对于与`for each`一起使用很重要。

默认情况下，程序目录中的所有变量都无法更改。也就是说，如果我定义了一个变量`x`，我就不能后来重新定义它：

```py
x = 10
# ...
x = 20 # produces shadowing error
```

如果我们确实想在以后更改（或修改）目录中的变量，我们可以这样做，但我们必须声明变量可以更改——也就是说，当我们定义它时，而不是写`x = 10`，我们必须写`var x = 10`。然后，当我们想要更新它时，我们可以使用`:=`运算符，就像上面模板中所做的那样。

```py
var x = 10
# ... x points to 10 in directory
x := 20
# ... x now points to 20 in directory
```

注意，尝试在未使用`var`声明的变量上使用`:=`会产生错误，并且变量仍然只能声明一次（无论是使用`var x = ...`还是`x = ...`）。

##### 9.1.8.5.2 块注释🔗 "链接至此")

在这些示例中展示的另一个新语言特性是，由于 Pyret 函数默认只期望一个（非定义）表达式，我们必须在顶部添加`block`注释，表示函数体包含多个表达式，其中最后一个表达式是函数评估到的结果。

作为另一个例子，如果我们尝试编写：

```py
fun my-function():
  1
  2
end
```

Pyret 会（正确地）报错——因为函数返回其体内的最后一个表达式，所以`1`将被忽略——这很可能是错误！也许目标是编写：

```py
fun my-function():
  1 + 2
end
```

然而，由于`for each`表达式仅存在以修改变量，包含它们的函数将始终有多个表达式，因此我们需要通知 Pyret 这不是错误。在函数开始处的冒号`:`之前添加`block`（或者，一般而言，将任何表达式包裹在`block:`和`end`中）通知 Pyret 我们理解存在多个表达式，我们只想评估到最后一个。所以，如果我们真的想按照第一个示例编写一个函数，我们可以这样做：

```py
fun my-function() block:
  1
  2
end
```

##### 9.1.8.5.3 `for each`的工作原理🔗 "链接至此")

`for each`表达式为输入列表中的每个元素运行其主体一次，在遍历过程中为每个元素添加一个程序目录条目。它不会直接产生任何值，因此更多地依赖于修改变量（如上所述）来产生计算。

考虑对一组数字进行求和。我们可以编写一个函数来完成这个任务，按照我们的模式，如下所示：

```py
fun sum-list(lst :: List) block:
  var run_total = 0
  for each(item from lst):
    run_total := item + run_total
  end
  run_total
where:
  sum-list([list: 5, 1, 7, 3]) is 16
end
```

在具体的测试输入`[list: 5, 1, 7, 3]`上，循环运行了四次，一次将`item`设置为`5`，然后设置为`1`，然后设置为`7`，最后设置为`3`。

`for each`结构使我们免于多次编写通用代码，并处理我们正在处理的列表可以是任意长度的事实（因此我们无法预测需要编写通用代码的次数）。因此，发生的情况是：

```py
run_total = 0
run_total = run_total + 5
run_total = run_total + 1
run_total = run_total + 7
run_total = run_total + 3
```

##### 9.1.8.5.4 测试和可变变量🔗 "链接到此处")

我们故意展示了一种特定的变量使用模式，这些变量可以改变。虽然还有其他用途（部分在可变变量中探讨），但坚持使用这种特定模板的主要原因在于测试的困难性，相应地，理解使用它们的其他方式的代码也很困难。

特别注意，这种模式意味着我们永远不会在函数外部定义可变变量，这意味着它永远不能被不同的函数或多次函数调用使用。每次函数运行时，都会创建一个新的变量，它在`for each`循环中被修改，然后返回值，并且程序目录中的条目被删除。

考虑如果我们不遵循模式会发生什么。假设我们遇到了以下问题：

> 练习
> 
> > 给定一个数字列表，返回列表的前缀（即从开始的所有元素），其和小于 100。

虽然我们已经了解了可变变量，但没有遵循模式，你可能会写出这样的代码：

```py
var count = 0

fun prefix-under-100(l :: List) -> List:
  var output = [list: ]
  for each(elt from l):
    count := count + elt
    when (count < 100):
      output := output + [list: elt]
    end
  end
end
```

现在，这看起来可能很合理——我们使用了一个新的结构`when`，它是一个没有`else`的`if`表达式——这只有在`for each`块内部才有意义，在那里我们不需要一个结果值。它等价于：

```py
if (count < 100):
  output := output + [list: elt]
else:
  nothing
end
```

其中`nothing`是 Pyret 中用来表示没有特定重要值的值。

但当我们使用这个函数时会发生什么呢？

```py
check:
    prefix-under-100([list: 1, 2, 3]) is [list: 1, 2, 3]
    prefix-under-100([list: 20, 30, 40]) is [list: 20, 30, 40]
    prefix-under-100([list: 80, 20, 10]) is [list: 80]
end
```

前两个测试通过了，但最后一个没有通过。为什么？如果我们再次运行第一个测试，事情会变得更加混乱，即如果我们不是运行上面的代码，而是运行这个`check`块：

```py
check:
    prefix-under-100([list: 1, 2, 3]) is [list: 1, 2, 3]
    prefix-under-100([list: 20, 30, 40]) is [list: 20, 30, 40]
    prefix-under-100([list: 80, 20, 10]) is [list: 80]
    prefix-under-100([list: 1, 2, 3]) is [list: 1, 2, 3]
end
```

现在最初通过测试的测试不再通过了！

我们看到的是，由于变量在函数外部，它在函数的不同调用之间是共享的。它只被添加到程序目录中一次，每次我们调用`prefix-under-100`，程序目录条目都会改变，但它永远不会重置。

故意，所有其他对目录条目的修改都是在只为函数体创建的目录条目上进行的，这意味着当函数退出时，它们会被删除。但现在，我们总是在修改单个`count`变量。这意味着每次我们调用`prefix-under-100`时，它的行为都不同，因为我们不仅必须理解函数体内的代码，还必须知道计数变量的当前值，而这不是仅通过查看代码就能弄清楚的事情！

具有这种行为的函数被称为有“副作用”，它们很难测试，也很难理解，因此更容易出现错误！虽然上面的例子错误的方式相对直接，但副作用可能导致非常微妙的错误，这些错误只有在以特定顺序调用函数时才会发生——这些顺序可能只在非常具体的情况下出现，使得它们难以理解或重现。

虽然在某些地方这样做是必要的，但几乎所有代码都可以不使用副作用来编写，这将使代码更加可靠。我们将在修改变量中探讨我们可能想要这样做的一些情况。

#### 9.1.1 表达式、函数和类型🔗 "链接到这里")

在函数练习：笔的成本中，我们通过计算一盒笔的成本的例子介绍了函数和类型的符号。一盒笔包括若干支笔和要打印在笔上的信息。每支笔的成本是 25 美分，加上每字符 2 美分的消息费用。以下是原始的 Pyret 代码：

```py
fun pen-cost(num-pens :: Number, message :: String) -> Number:
  doc: ```每支笔的总成本，每支 25 美分

        加上每条消息字符 2 美分```py
  num-pens * (0.25 + (string-length(message) * 0.02))
end
```

这是相应的 Python 代码：

```py
def pen_cost(num_pens: int, message: str) -> float:
    """total cost for pens, each at 25 cents plus
       2 cents per message character"""
    return num_pens * (0.25 + (len(message) * 0.02))
```

> 现在行动！
> 
> > 你在两个版本之间看到了哪些符号差异？

这里是差异的总结：

+   Python 使用`def`而不是`fun`。

+   Python 在名称中使用下划线（如`pen_cost`），而不是像 Pyret 中使用连字符。

+   类型名称的写法不同：Python 使用`str`和`int`而不是`String`和`Number`。此外，Python 在类型前只使用一个冒号，而 Pyret 使用两个冒号。

+   Python 为不同类型的数字有不同的类型：`int`用于整数，而`float`用于小数。Pyret 对所有数字只使用一个类型（`Number`）。

+   Python 不标记文档字符串（如 Pyret 使用`doc:`）。

+   Python 中没有`end`注释。相反，Python 使用缩进来定位 if/else 语句、函数或其他多行结构的结束。

+   Python 使用`return`标记函数的输出。

这些是符号上的细微差异，随着你在 Python 中编写更多程序，你会习惯这些差异。

除了符号上的差异之外，还有其他差异。这个样本程序中出现的差异与语言如何使用类型有关。在 Pyret 中，如果你对一个参数添加了类型注解，然后传递了一个不同类型的值，你会得到一个错误消息。Python 忽略了类型注解（除非你引入了额外的工具来检查类型）。Python 类型就像是程序员的笔记，但在程序运行时并不强制执行。

> 练习
> 
> > 将以下`moon-weight`函数从函数练习：月球重量转换为 Python：
> > 
> > ```py
> > fun moon-weight(earth-weight :: Number) -> Number:
> >   doc:" Compute weight on moon from weight on earth"
> >   earth-weight * 1/6
> > end
> > ```

#### 9.1.2 从函数返回值🔗 "链接至此")

在 Pyret 中，函数体由可选的语句组成，用于命名中间值，然后是一个单独的表达式。该单个表达式的值是调用函数的结果。在 Pyret 中，每个函数都会产生一个结果，因此不需要标记结果来源。

正如我们将看到的，Python 是不同的：并非所有“函数”都会返回结果（注意名称从`fun`更改为`def`）。在数学中，函数的定义就是有结果。程序员有时会区分“函数”和“过程”这两个术语：两者都指参数化计算，但只有前者会向周围计算返回结果。然而，一些程序员和语言更宽松地使用“函数”这个术语来涵盖这两种参数化计算。此外，结果不一定是`def`中的最后一个表达式。在 Python 中，关键字`return`明确标记了其值作为函数结果的那个表达式。

> 现在就做！
> 
> > 将这两个定义放入 Python 文件中。
> > 
> > ```py
> > def add1v1(x: int) -> int:
> >     return x + 1
> > 
> > def add1v2(x: int) -> int:
> >     x + 1
> > ```
> > 
> > 在 Python 提示符下，依次调用每个函数。你注意到使用每个函数的结果有什么不同吗？

希望你能注意到，使用`add1v1`在提示符后显示答案，而使用`add1v2`则不会。这种差异对函数的组合有影响。

> 现在就做！
> 
> > 在 Python 提示符下尝试评估以下两个表达式：每种情况下会发生什么？
> > 
> > `3 * add1v1(4)`
> > 
> > `3 * add1v2(4)`

这个例子说明了为什么`return`在 Python 中是必不可少的：没有它，就不会返回任何值，这意味着你无法在另一个表达式中使用函数的结果。那么`add1v2`有什么用呢？记住这个问题；我们将在修改变量中回到它。

#### 9.1.3 示例和测试用例🔗 "链接至此")

在 Pyret 中，我们为每个函数都提供了使用`where:`块的示例。我们还有能力编写`check:`块进行更广泛的测试。作为提醒，以下是包含`where:`块的`pen-cost`代码：

```py
fun pen-cost(num-pens :: Number, message :: String) -> Number:
  doc: ```钢笔的总费用，每支 25 美分

    每条消息字符加 2 美分```py
  num-pens * (0.25 + (string-length(message) * 0.02))
where:
  pen-cost(1, "hi") is 0.29
  pen-cost(10, "smile") is 3.50
end
```

Python 没有关于 `where:` 块的概念，也没有示例和测试之间的区别。Python 有几个不同的测试包；在这里，我们将使用 `pytest`，这是一个标准的轻量级框架，类似于我们在 Pyret 中进行的测试形式。如何设置 pytest 和你的测试文件内容将根据你的 Python IDE 而有所不同。我们假设讲师将提供与他们的工具选择一致的单独说明。为了使用 `pytest`，我们将示例和测试放在一个单独的函数中。以下是对 `pen_cost` 函数的示例：

```py
import pytest

def pen_cost(num_pens: int, message: str) -> float:
    """total cost for pens, each at 25 cents plus
       2 cents per message character"""
    return num_pens * (0.25 + (len(message) * 0.02))

def test_pens():
  assert pen_cost(1, "hi") == 0.29
  assert pen_cost(10, "smile") == 3.50
```

关于此代码的注意事项：

+   我们已经导入了 `pytest`，一个轻量级的 Python 测试库。

+   示例已经移动到一个函数中（这里 `test_pens`），该函数不接受任何输入。请注意，包含测试用例的函数名称必须以 `test_` 开头，以便 `pytest` 能够找到它们。

+   在 Python 中，单个测试的形式如下

    ```py
    assert EXPRESSION == EXPECTED_ANS
    ```

    而不是 Pyret 中的 `is` 形式。

> 立刻行动！
> 
> > 向 Python 代码中添加一个额外的测试，对应于 Pyret 测试
> > 
> > ```py
> > pen-cost(3, "wow") is 0.93
> > ```
> > 
> > 确保运行测试。
> > 
> 立刻行动！
> 
> > 你真的尝试运行测试了吗？

哇！发生了一些奇怪的事情：测试失败了。停下来想想：在 Pyret 中运行正常的同一个测试在 Python 中失败了。这是怎么回事？

#### 9.1.4 数字的补充说明🔗 "链接到此处")

结果表明，不同的编程语言在表示和管理实数（非整数）方面做出了不同的决定。有时，这些表示之间的差异会导致计算值中的细微数量差异。作为一个简单的例子，让我们看看两个看似简单的实数 `1/2` 和 `1/3`。以下是我们在 Pyret 提示符中输入这两个数字时得到的结果：

|

&#124;

> ```py
> 1/2
> ```

&#124;

|

|

```py
0.5
```

|

|

&#124;

> ```py
> 1/3
> ```

&#124;

|

|

```py
0.3
```

|

如果我们在 Python 控制台中输入这两个相同的数字，我们得到的结果是：

|

&#124;

> ```py
> 1/2
> ```

&#124;

|

|

```py
0.5
```

|

|

&#124;

> ```py
> 1/3
> ```

&#124;

|

|

```py
0.3333333333333333
```

|

注意，对于 `1/3`，答案看起来不同。你可能（或者可能没有！）从之前的数学课中回忆起来，`1/3` 是一个非终止的循环小数的例子。简单来说，如果我们试图用小数形式写出 `1/3` 的确切值，我们需要写出无限序列的 `3`。数学家通过在 `3` 上放置一条水平线来表示这一点。这就是我们在 Pyret 中看到的表示法。相比之下，Python 会写出 `3` 的部分序列。

在这个区别之下，有一些关于计算机中数字表示的有趣细节。计算机没有无限的空间来存储数字（或者任何其他东西）：当程序需要与非终止小数一起工作时，底层语言可以选择：

+   近似数字（通过在某个点上截断无限序列的数字），然后只使用近似值继续工作，或者

+   存储关于数字的额外信息，这可能允许我们在以后用它进行更精确的计算（尽管总有一些数字在有限空间中无法精确表示）。

Python 采用第一种方法。因此，使用近似值进行的计算有时会产生近似的结果。这就是我们新的 `pen_cost` 测试用例发生的情况。虽然从数学上讲，计算应该得到 `0.93`，但近似值却得到了 `0.9299999999999999`。

那么，我们如何在这种情况写下测试？我们需要告诉 Python，答案应该是“接近” `0.93`，在近似的误差范围内。下面是这个样子的：

```py
assert pen_cost(3, "wow") == pytest.approx(0.93)
```

我们用 `pytest.approx` 将我们想要的精确答案包装起来，以表明我们将接受任何接近我们指定值的答案。如果你想控制小数点的精度位数，你可以这样做，但默认的 `± 2.3e-06` 通常足够了。

#### 9.1.5 条件语句🔗 "链接到此处")

继续使用我们原来的 `pen_cost` 示例，下面是计算订单运费的函数的 Python 版本：

```py
def add_shipping(order_amt: float) -> float:
    """increase order price by costs for shipping"""
    if order_amt == 0:
      return 0
    elif order_amt <= 10:
      return order_amt + 4
    elif (order_amt > 10) and (order_amt < 30):
      return order_amt + 8
    else:
      return order_amt + 12
```

这里要注意的主要区别是，Python 中的 `else if` 写作单词 `elif`。我们使用 `return` 来标记条件语句每个分支的函数结果。否则，两种语言的条件结构相当相似。

你可能已经注意到，Python 不需要在 `if` 表达式或函数上显式地使用 `end` 注解。相反，Python 会查看你代码的缩进来确定何时一个结构结束。例如，在 `pen_cost` 和 `test_pens` 的代码示例中，Python 确定函数 `pen_cost` 已经结束，因为它在程序文本的左侧边缘检测到一个新的定义（对于 `test_pens`）。同样的原则也适用于结束条件。

当我们与 Python 一起工作时，我们会回到关于缩进的这个点，并看到更多的例子。

#### 9.1.6 创建和处理列表🔗 "链接到此处")

作为一个列表的例子，让我们假设我们一直在玩一个需要用一组合字牌来造词的游戏。在 Pyret 中，我们可以这样写一个样本单词列表：

```py
words = [list: "banana", "bean", "falafel", "leaf"]
```

在 Python 中，这个定义看起来是这样的：

```py
words = ["banana", "bean", "falafel", "leaf"]
```

这里的唯一区别是 Python 不使用 Pyret 中需要的 `list:` 标签。

##### 9.1.6.1 过滤器、映射和朋友们🔗 "链接到此处")

当我们最初学习 Pyret 中的列表时，我们从常见的内置函数开始，如 `filter`、`map`、`member` 和 `length`。我们还看到了 `lambda` 的使用，帮助我们简洁地使用这些函数。这些相同的函数，包括 `lambda`，也存在于 Python 中。以下是一些示例（`#` 是 Python 中的注释字符）：

```py
words = ["banana", "bean", "falafel", "leaf"]

# filter and member
words_with_b = list(filter(lambda wd: "b" in wd, words))
# filter and length
short_words = list(filter(lambda wd: len(wd) < 5, words))
# map and length
word_lengths = list(map(len, words))
```

注意，你必须使用`list()`来包装对`filter`（和`map`）的调用。内部，Python 有这些函数返回一种我们尚未讨论（也不需要）的数据类型。使用`list`将返回的数据转换为列表。如果你省略了`list`，你将无法链式调用某些函数。例如，如果我们试图在不首先转换为列表的情况下计算`map`的结果的长度，我们会得到一个错误：

|

&#124;

> ```py
> len(map(len,b))
> ```

&#124;

|

|

```py
TypeError: object of type 'map' has no len()
```

|

不要担心这个错误信息现在没有意义（我们还没有学习什么是“对象”）。关键是，如果你在使用`filter`或`map`的结果时看到这样的错误，你很可能忘记将结果包装在`list`中。

> 练习
> 
> > 通过编写以下问题的表达式来练习 Python 的列表函数。仅使用我们迄今为止向您展示的列表函数。
> > 
> > +   给定一个数字列表，根据每个数字的符号将其转换为字符串列表`"pos"`、`"neg"`、`"zero"`。
> > +   
> > +   给定一个字符串列表，是否有任何字符串的长度等于 5？
> > +   
> > +   给定一个数字列表，从该列表中生成一个包含 10 到 20 之间偶数的列表。

我们有意专注于使用 Python 内置函数处理列表的计算，而不是向您展示如何编写自己的函数（正如我们在 Pyret 中处理递归时所做的）。虽然您可以在 Pyret 中编写递归函数来处理列表，但用于此目的的编程风格更为传统。我们将在关于可变变量的章节中探讨这一点。

##### 9.1.6.1 过滤器、映射和朋友们🔗 "链接到此处")

当我们最初在 Pyret 中学习列表时，我们从常见的内置函数开始，例如`filter`、`map`、`member`和`length`。我们还看到了`lambda`的使用，帮助我们简洁地使用这些函数。这些相同的函数，包括`lambda`，也存在于 Python 中。以下是一些示例（`#`是 Python 中的注释字符）：

```py
words = ["banana", "bean", "falafel", "leaf"]

# filter and member
words_with_b = list(filter(lambda wd: "b" in wd, words))
# filter and length
short_words = list(filter(lambda wd: len(wd) < 5, words))
# map and length
word_lengths = list(map(len, words))
```

注意，你必须使用`list()`来包装对`filter`（和`map`）的调用。Python 内部，这些函数返回一种我们尚未讨论（也不需要）的数据类型。使用`list`将返回的数据转换为列表。如果你省略了`list`，你将无法链式调用某些函数。例如，如果我们试图在不首先转换为列表的情况下计算`map`的结果的长度，我们会得到一个错误：

|

&#124;

> ```py
> len(map(len,b))
> ```

&#124;

|

|

```py
TypeError: object of type 'map' has no len()
```

|

不要担心这个错误信息现在没有意义（我们还没有学习什么是“对象”）。关键是，如果你在使用`filter`或`map`的结果时看到这样的错误，你很可能忘记将结果包装在`list`中。

> 练习
> 
> > 通过编写以下问题的表达式来练习 Python 的列表函数。仅使用我们迄今为止向您展示的列表函数。
> > 
> > +   给定一个数字列表，根据每个数字的符号将其转换为字符串列表`"pos"`、`"neg"`、`"zero"`。
> > +   
> > +   给定一个字符串列表，任何字符串的长度是否等于 5？
> > +   
> > +   给定一个数字列表，从该列表中生成一个介于 10 和 20 之间的偶数列表。

我们有意专注于使用 Python 的内置函数处理列表的计算，而不是向您展示如何编写自己的函数（就像我们在 Pyret 中的递归那样）。虽然您可以在 Pyret 中编写递归函数来处理列表，但用于此目的的编程风格更传统。我们将在关于 Mutating Variables 的章节中探讨这一点。

#### 9.1.7 具有组件的数据🔗 "链接至此")

与 Pyret 中的数据定义（没有变体）相对应的是 Python 中的 dataclass。那些熟悉 Python 的人可能会 wonder 为什么我们使用 dataclass 而不是字典或原始类。与字典相比，dataclass 允许使用类型提示并捕获我们的数据具有固定字段集合的事实。与原始类相比，dataclass 生成大量的样板代码，这使得它们比原始类更轻量级。以下是一个 Pyret 中的 todo-list 数据类型及其对应的 Python 代码示例：

```py
# a todo item in Pyret
data ToDoItemData:
  | todoItem(descr :: String,
             due :: Date,
             tags :: List<String>
end
```

```py
------------------------------------------
# the same todo item in Python

# to allow use of dataclasses
from dataclasses import dataclass
# to allow dates as a type (in the ToDoItem)
from datetime import date

@dataclass
class ToDoItem:
    descr: str
    due: date
    tags: list

# a sample list of ToDoItem
myTD = [ToDoItem("buy milk", date(2020, 7, 27), ["shopping", "home"]),
        ToDoItem("grade hwk", date(2020, 7, 27), ["teaching"]),
        ToDoItem("meet students", date(2020, 7, 26), ["research"])
       ]
```

注意事项：

+   类型及其构造函数只有一个名称，而不是像 Pyret 中那样有单独的名称。

+   字段名称之间没有逗号（但在 Python 中每个字段都必须单独一行）

+   在 Python 中无法指定列表内容的类型（至少，不使用更高级的包来编写类型）

+   在 `class` 前面需要 `@dataclass` 注解。

+   Dataclasses 不支持创建具有多个变体的数据类型，就像我们在 Pyret 中经常做的那样。这样做需要比本书中将要涵盖的更高级的概念。

##### 9.1.7.1 在 Dataclasses 中访问字段🔗 "链接至此")

在 Pyret 中，我们通过使用点（句点）来“深入”数据并访问字段来从结构化数据中提取字段。相同的符号在 Python 中也适用：

|

> ```py
> travel = ToDoItem("buy tickets", date(2020, 7, 30), ["vacation"])
> ```

|

|

&#124;

> ```py
> travel.descr
> ```

&#124;

|

|

```py
"buy tickets"
```

|

##### 9.1.7.1 在 Dataclasses 中访问字段🔗 "链接至此")

在 Pyret 中，我们通过使用点（句点）来“深入”数据并访问字段来从结构化数据中提取字段。相同的符号在 Python 中也适用：

|

> ```py
> travel = ToDoItem("buy tickets", date(2020, 7, 30), ["vacation"])
> ```

|

|

&#124;

> ```py
> travel.descr
> ```

&#124;

|

|

```py
"buy tickets"
```

|

#### 9.1.8 遍历列表🔗 "链接至此")

##### 9.1.8.1 介绍 `For` 循环🔗 "链接至此")

在 Pyret 中，我们通常编写递归函数来计算列表上的汇总值。作为提醒，这里有一个 Pyret 函数，用于计算列表中的数字之和：

```py
fun sum-list(numlist :: List<Number>) -> Number:
  cases (List) numlist:
    | empty => 0
    | link(fst, rst) => fst + sum-list(rst)
  end
end
```

在 Python 中，将列表分解为其首元素和其余部分并递归处理其余部分是不常见的。相反，我们使用一个称为 `for` 的结构来依次访问列表中的每个元素。以下是一个使用具体（示例）奇数列表的 `for` 形式：

```py
for num in [5, 1, 7, 3]:
   # do something with num
```

这里使用的 `num` 名称是我们自己选择的，就像 Pyret 中函数参数的名称一样。当 `for` 循环评估时，列表中的每个项目依次被称为 `num`。因此，这个 `for` 示例等同于编写以下内容：

```py
# do something with 5
# do something with 1
# do something with 7
# do something with 3
```

`for` 构造节省了我们多次编写常见代码的时间，并且处理了我们正在处理的列表可以具有任意长度的事实（因此我们无法预测需要编写多少次常见代码）。

现在，让我们使用 `for` 来计算列表的累计总和。我们首先再次确定我们的具体列表中的重复计算。起初，让我们只用文字表达重复的计算。在 Pyret 中，我们的重复计算是这样的：“将第一个元素添加到其余元素的总和”。我们已经说过，在 Python 中我们无法轻松访问“其余的元素”，因此我们需要重新表述。这里有一个替代方案：

```py
# set a running total to 0
# add 5 to the running total
# add 1 to the running total
# add 7 to the running total
# add 3 to the running total
```

注意，这个框架指的是“剩余的计算”，而不是到目前为止已经发生的计算（即“累计总和”）。如果你恰好已经阅读了关于`my-running-sum`: 示例和代码的章节，这个框架可能很熟悉。

让我们将这个文字草图转换为代码，通过将草图中的每一行替换为具体的代码来实现。我们这样做是通过设置一个名为 `run_total` 的变量，并为其每个元素更新其值。

```py
run_total = 0
run_total = run_total + 5
run_total = run_total + 1
run_total = run_total + 7
run_total = run_total + 3
```

这种给现有变量名赋予新值的思想是我们之前没有见过的。实际上，当我们第一次看到如何命名值（在程序目录中）时，我们明确表示 Pyret 不允许这样做（至少，不是用我们向您展示的构造）。Python 可以。我们将在稍后更深入地探讨这种能力带来的后果（在变量修改中）。现在，让我们只使用这种能力，这样我们可以学习遍历列表的模式。首先，让我们将重复的代码行合并为单个 `for` 的使用：

```py
run_total = 0
for num in [5, 1, 7, 3]:
   run_total = run_total + num
```

这段代码对于特定的列表来说工作得很好，但我们的 Pyret 版本将求和的列表作为函数的参数。要在 Python 中实现这一点，我们需要像本章前面其他示例中那样将 `for` 包裹在函数中。这是最终版本。

```py
def sum_list(numlist : list) -> float:
    """sum a list of numbers"""
    run_total = 0
    for num in numlist:
        run_total = run_total + num
    return(run_total)
```

> 现在行动！
> 
> > 为 `sum_list`（Python 版本）编写一组测试。

现在 Python 版本已完成，让我们将其与原始 Pyret 版本进行比较：

```py
fun sum-list(numlist :: List<Number>) -> Number:
  cases (List) numlist:
    | empty => 0
    | link(fst, rst) => fst + sum-list(rst)
  end
end
```

下面是关于这两段代码需要注意的一些事项：

+   Python 版本需要一个变量（这里为 `run_total`）来存储我们在遍历（处理）列表时逐步构建的计算结果。

+   该变量的初始值是 Pyret 中在 `empty` 情况下返回的答案。

+   Pyret 函数的 `link` 情况中的计算用于在 `for` 的主体中更新该变量。

+   在 `for` 循环处理完列表中的所有项目后，Python 版本将变量中的值作为函数的结果返回。

##### 9.1.8.2 关于处理列表元素顺序的旁白链接 "链接到此处")

如果考虑两个程序如何运行，这里还有一个细微之处：Python 版本从左到右累加元素，而 Pyret 版本则是从右到左累加。具体来说，`run_total` 的值序列是这样计算的：

```py
run_total = 0
run_total = 0 + 5
run_total = 5 + 1
run_total = 6 + 7
run_total = 13 + 3
```

相比之下，Pyret 版本展开如下：

```py
sum_list([list: 5, 1, 7, 3])
5 + sum_list([list: 1, 7, 3])
5 + 1 + sum_list([list: 7, 3])
5 + 1 + 7 + sum_list([list: 3])
5 + 1 + 7 + 3 + sum_list([list:])
5 + 1 + 7 + 3 + 0
5 + 1 + 7 + 3
5 + 1 + 10
5 + 11
16
```

作为提醒，Pyret 版本这样做是因为在 `link` 情况下，`+` 只能在计算完列表剩余部分的和之后才能得到答案。尽管我们人类看到 Pyret 展开中每一行的 `+` 操作链，但 Pyret 只看到表达式 `fst + sum-list(rst)`，这要求函数调用完成后再执行 `+`。

在求和列表的情况下，我们不会注意到两个版本之间的区别，因为无论我们是左到右还是右到左计算，和都是相同的。在我们编写的其他函数中，这种差异可能会开始变得重要。

##### 9.1.8.3 在生成列表的函数中使用 `For` 循环链接 "链接到此处")

让我们在另一个函数上练习使用 `for` 循环，这个函数遍历列表并生成一个列表。具体来说，让我们编写一个程序，它接受一个字符串列表，并生成一个包含列表中包含字母 `"z"` 的单词的列表。

正如我们在 `sum_list` 函数中所做的那样，我们在构建列表时需要一个变量来存储结果列表。以下代码称为 `zlist`。该代码还展示了如何使用 `in` 来检查一个字符是否在字符串中（它也适用于检查一个元素是否在列表中），以及如何将元素添加到列表的末尾（`append`）。

```py
def all_z_words(wordlist : list) -> list:
    """produce list of words from the input that contain z"""
    zlist = [] # start with an empty list
    for wd in wordlist:
        if "z" in wd:
            zlist = [wd] + zlist
    return(zlist)
```

这段代码遵循 `sum_list` 的结构，即我们使用类似于在 Pyret 中会使用的表达式更新 `zlist` 的值。对于那些有先前的 Python 经验并会在此时使用 `zlist.append` 的人来说，请记住这个想法。我们将在 可变列表 中到达那里。

> 练习
> 
> > 为 `all_z_words` 编写测试。
> > 
> 练习
> 
> > 使用 `filter` 编写 `all_z_words` 的第二个版本。务必为其编写测试！
> > 
> 练习
> 
> > 对比这两个版本及其相应的测试。你注意到什么有趣的地方了吗？

##### 9.1.8.4 总结：Python 的列表处理模板链接 "链接到此处")

正如我们在 Pyret 中编写列表处理函数有一个模板一样，Python 中基于 `for` 循环也有相应的模板。作为提醒，该模式如下：

```py
def func(lst: list):
  result = ...  # what to return if the input list is empty
  for item in lst:
    # combine item with the result so far
    result = ... item ... result
  return result
```

在学习如何在 Python 中编写列表函数时，请记住这个模板。

##### 9.1.8.5 `for each` 循环在 Pyret 中的使用链接 "链接到此处")

本节可以独立阅读，无需阅读本章的其余部分，所以如果你在接触 Python 之前被引导到这里，请不要担心！虽然下面的内容反映了 Python 中存在的类似结构，但它是以独立的方式引入的。

前几节介绍了 Python 中的 `for` 循环，并展示了使用它们处理列表的模板。Pyret 可以使用以下模式做类似的事情：

```py
fun func(lst :: List) block:
  var result = ...  # what to return if the input list is empty
  for each(item from lst):
    # combine item with the result so far
    result := ... item ... result
  end
  result
end
```

在本例中使用了几个新的语言特性，这些特性将在以下几节中介绍。

##### 9.1.8.5.1 可变的变量🔗 "链接至此")

首先，请注意，我们使用 `var result` 引入变量 `result` – 这意味着它可以变化，这对于与 `for each` 的使用很重要。

默认情况下，程序目录中的所有变量都不能更改。也就是说，如果我定义了一个变量 `x`，我就不能后来重新定义它：

```py
x = 10
# ...
x = 20 # produces shadowing error
```

如果我们想在以后更改（或修改）目录中的变量，我们可以这样做，但我们必须声明变量可以更改 – 也就是说，当我们定义它时，而不是写 `x = 10`，我们必须写 `var x = 10`。然后，当我们想要更新它时，我们可以使用 `:=` 操作符，就像上面模板中所做的那样。

```py
var x = 10
# ... x points to 10 in directory
x := 20
# ... x now points to 20 in directory
```

注意，尝试在未使用 `var` 声明的变量上使用 `:=` 会产生错误，并且变量仍然只能声明一次（无论是使用 `var x = ...` 还是 `x = ...`）。

##### 9.1.8.5.2 块表示法🔗 "链接至此")

在这些示例中展示的另一个新语言特性是，由于 Pyret 函数默认只期望一个（非定义）表达式，我们必须在顶部添加 `block` 注解，表示函数的主体是多个表达式，其中最后一个表达式是函数评估到的。

作为另一个例子，如果我们尝试编写：

```py
fun my-function():
  1
  2
end
```

Pyret 会（正确地）产生错误 – 因为函数返回其主体中的最后一个表达式，所以 `1` 将被忽略 – 这很可能是错误！也许目标是编写：

```py
fun my-function():
  1 + 2
end
```

然而，由于 `for each` 表达式仅用于修改变量，包含它们的函数将始终有多个表达式，因此我们需要通知 Pyret 这不是一个错误。在函数开始前的 `:` 前添加 `block`（或者，一般而言，将任何表达式包裹在 `block:` 和 `end` 中）通知 Pyret 我们理解存在多个表达式，我们只想评估最后一个。所以，如果我们真的想按照第一个示例编写一个函数，我们可以这样做：

```py
fun my-function() block:
  1
  2
end
```

##### 9.1.8.5.3 `for each` 的工作原理🔗 "链接至此")

`for each` 表达式对输入列表中的每个元素运行其主体一次，在遍历过程中为每个元素在程序目录中添加一个条目。它不会直接产生任何值，因此更多地依赖于修改变量（如上所述）来产生计算。

考虑对数字列表求和。我们可以编写一个函数来完成这个任务，遵循我们的模式，如下所示：

```py
fun sum-list(lst :: List) block:
  var run_total = 0
  for each(item from lst):
    run_total := item + run_total
  end
  run_total
where:
  sum-list([list: 5, 1, 7, 3]) is 16
end
```

在具体的测试输入`[list: 5, 1, 7, 3]`上，循环运行了四次，一次将`item`设置为`5`，然后设置为`1`，接着设置为`7`，最后设置为`3`。

`for each`结构使我们免于多次编写通用代码，并处理我们正在处理的列表可以具有任意长度的事实（因此我们无法预测需要编写通用代码的次数）。因此，发生的情况是：

```py
run_total = 0
run_total = run_total + 5
run_total = run_total + 1
run_total = run_total + 7
run_total = run_total + 3
```

##### 9.1.8.5.4 测试和可变变量🔗 "链接到此处")

我们故意展示了一种非常特定的使用可变变量的模式。虽然还有其他用途（在可变变量部分中部分探索），但坚持这个特定模板的主要原因是在测试和理解使用它们的代码方面的困难。

尤其要注意的是，这个模式意味着我们永远不会定义可以在函数外部改变的变量，这意味着它永远不能被不同的函数或多个函数调用所使用。每次函数运行时，都会创建一个新的变量，它在`for each`循环中被修改，然后返回值，并且程序目录中的条目被移除。

考虑如果我们不遵循我们的模式会发生什么。假设我们遇到了以下问题：

> 练习
> 
> > 给定一个数字列表，返回列表的前缀（即从开始的所有元素），其和小于 100。

虽然我们已经了解了可变变量，但没有遵循这个模式，你可能会写出这样的代码：

```py
var count = 0

fun prefix-under-100(l :: List) -> List:
  var output = [list: ]
  for each(elt from l):
    count := count + elt
    when (count < 100):
      output := output + [list: elt]
    end
  end
end
```

现在，这看起来可能很合理——我们使用了一个新的结构`when`，这是一个没有`else`的`if`表达式——这只有在`for each`块内部才有意义，在那里我们不需要结果值。它等价于：

```py
if (count < 100):
  output := output + [list: elt]
else:
  nothing
end
```

其中`nothing`是 Pyret 中用来表示没有特定重要值的值。

但当我们使用这个函数时会发生什么呢？

```py
check:
    prefix-under-100([list: 1, 2, 3]) is [list: 1, 2, 3]
    prefix-under-100([list: 20, 30, 40]) is [list: 20, 30, 40]
    prefix-under-100([list: 80, 20, 10]) is [list: 80]
end
```

前两个测试通过了，但最后一个没有通过。为什么？如果我们再次运行第一个测试，事情会变得更加混乱，即如果我们不是运行上面的代码，而是运行这个`check`块：

```py
check:
    prefix-under-100([list: 1, 2, 3]) is [list: 1, 2, 3]
    prefix-under-100([list: 20, 30, 40]) is [list: 20, 30, 40]
    prefix-under-100([list: 80, 20, 10]) is [list: 80]
    prefix-under-100([list: 1, 2, 3]) is [list: 1, 2, 3]
end
```

现在最初通过的测试不再通过了！

我们所看到的是，由于变量位于函数外部，它会在函数的不同调用之间共享。它只会在程序目录中添加一次，每次我们调用`prefix-under-100`时，程序目录条目都会改变，但它永远不会重置。

故意，所有其他对修改的用法都仅限于为函数体创建的目录条目，这意味着当函数退出时，它们会被删除。但现在，我们总是在修改单个 `count` 变量。这意味着每次我们调用 `prefix-under-100` 时，它的行为都不同，因为我们不仅必须理解函数体内的代码，还必须知道计数变量的当前值，而这并不是仅通过查看代码就能弄清楚的事情！

具有这种行为的函数被称为有“副作用”，它们更难测试和更难理解，因此更容易出现错误！虽然上述示例在相对直接的方式上是错误的，但副作用可以导致非常微妙的错误，这些错误仅在以特定顺序调用函数时才会发生——这些顺序可能只在非常具体的情况下出现，使得它们难以理解或重现。

虽然在某些地方这样做是必要的，但几乎所有的代码都可以不产生副作用来编写，这将使代码更加可靠。我们将在 修改变量 中探讨我们可能想要这样做的一些情况。

##### 9.1.8.1 介绍 `For` 循环🔗 "链接到此处")

在 Pyret 中，我们通常编写递归函数来计算列表上的汇总值。作为提醒，这里有一个 Pyret 函数，用于计算列表中的数字之和：

```py
fun sum-list(numlist :: List<Number>) -> Number:
  cases (List) numlist:
    | empty => 0
    | link(fst, rst) => fst + sum-list(rst)
  end
end
```

在 Python 中，将列表拆分为其首部和其余部分并递归处理其余部分是不寻常的。相反，我们使用一个称为 `for` 的构造来依次访问列表中的每个元素。以下是一个具体的（示例）奇数列表的 `for` 形式：

```py
for num in [5, 1, 7, 3]:
   # do something with num
```

在这里，`num` 这个名字是我们自己选择的，就像在 Pyret 函数中参数的名字一样。当 `for` 循环执行时，列表中的每个项目依次被称为 `num`。因此，这个 `for` 示例等同于以下写法：

```py
# do something with 5
# do something with 1
# do something with 7
# do something with 3
```

`for` 构造使我们免于多次编写常见代码，并处理我们正在处理的列表可以是任意长度的事实（因此我们无法预测需要编写多少次常见代码）。

现在，让我们使用 `for` 来计算列表的累积和。我们首先将再次使用我们的具体列表来找出重复的计算。首先，让我们用散文的形式表达重复的计算。在 Pyret 中，我们的重复计算是这样的：“将第一个项目添加到其余项目的和中”。我们已经说过，在 Python 中我们无法轻松访问“其余的项目”，因此我们需要重新表述。以下是另一种表述方式：

```py
# set a running total to 0
# add 5 to the running total
# add 1 to the running total
# add 7 to the running total
# add 3 to the running total
```

注意，这个框架指的是“剩余的计算”，而不是到目前为止已经发生的计算（即“累计总和”）。如果您恰好已经完成了关于`my-running-sum`: 示例和代码的章节，这个框架可能很熟悉。

让我们将这个文字草图转换为代码，通过将草图中的每一行替换为具体的代码来实现。我们通过设置一个名为 `run_total` 的变量并更新其值来为每个元素执行此操作。

```py
run_total = 0
run_total = run_total + 5
run_total = run_total + 1
run_total = run_total + 7
run_total = run_total + 3
```

这种给现有变量名赋予新值的思想是我们之前没有见过的。事实上，当我们第一次看到如何命名值（在程序目录中）时，我们明确表示 Pyret 不允许这样做（至少，不是用我们向您展示的构造）。Python 可以。我们将在稍后更深入地探讨这种能力的影响（在修改变量中）。现在，让我们只使用这种能力，这样我们就可以学习遍历列表的模式。首先，让我们将重复的代码行合并为单个 `for` 的使用：

```py
run_total = 0
for num in [5, 1, 7, 3]:
   run_total = run_total + num
```

这段代码对于特定的列表来说工作得很好，但我们的 Pyret 版本将求和的列表作为函数的参数。为了在 Python 中实现这一点，我们像本章前面的一些其他示例那样将 `for` 包裹在函数中。这是最终版本。

```py
def sum_list(numlist : list) -> float:
    """sum a list of numbers"""
    run_total = 0
    for num in numlist:
        run_total = run_total + num
    return(run_total)
```

> 立刻行动！
> 
> > 为 `sum_list`（Python 版本）编写一组测试。

现在，Python 版本已经完成，让我们将其与原始 Pyret 版本进行比较：

```py
fun sum-list(numlist :: List<Number>) -> Number:
  cases (List) numlist:
    | empty => 0
    | link(fst, rst) => fst + sum-list(rst)
  end
end
```

关于这两段代码，有一些需要注意的事项：

+   Python 版本需要一个变量（这里 `run_total`）来保存我们在遍历（处理）列表时构建的计算结果。

+   该变量的初始值是 Pyret 中 `empty` 情况下我们返回的答案。

+   Pyret 函数的 `link` 情况中的计算用于更新 `for` 体中的该变量。

+   在 `for` 完成处理列表中的所有项目后，Python 版本将变量中的值作为函数的结果返回。

##### 9.1.8.2 关于处理列表元素顺序的旁白🔗 "链接至此")

如果我们考虑这两个程序是如何运行的，这里还有一个细微之处：Python 版本是从左到右累加元素，而 Pyret 版本是从右到左累加。具体来说，`run_total` 的值序列是这样计算的：

```py
run_total = 0
run_total = 0 + 5
run_total = 5 + 1
run_total = 6 + 7
run_total = 13 + 3
```

相比之下，Pyret 版本展开如下：

```py
sum_list([list: 5, 1, 7, 3])
5 + sum_list([list: 1, 7, 3])
5 + 1 + sum_list([list: 7, 3])
5 + 1 + 7 + sum_list([list: 3])
5 + 1 + 7 + 3 + sum_list([list:])
5 + 1 + 7 + 3 + 0
5 + 1 + 7 + 3
5 + 1 + 10
5 + 11
16
```

作为提醒，Pyret 版本之所以这样做，是因为 `link` 情况中的 `+` 只能在计算了列表其余部分的和之后才能简化为答案。尽管我们作为人类看到 Pyret 展开中每一行的 `+` 操作链，但 Pyret 只看到表达式 `fst + sum-list(rst)`，这需要在 `+` 执行之前完成函数调用。

在求和列表的情况下，我们不会注意到两个版本之间的区别，因为无论我们是左到右还是右到左计算，总和都是相同的。在我们编写的其他函数中，这种差异可能开始变得重要。

##### 9.1.8.3 在生成列表的函数中使用`For`循环🔗 "链接至此")

让我们在另一个函数上练习使用`for`循环，这次是一个生成列表的函数。具体来说，让我们编写一个程序，它接受一个字符串列表，并生成包含字母`"z"`的单词列表。

正如我们的`sum_list`函数一样，我们将在构建列表的过程中需要一个变量来存储结果。以下代码调用这个`zlist`。代码还展示了如何使用`in`来检查一个字符是否在字符串中（它也适用于检查一个项目是否在列表中）以及如何将一个元素添加到列表的末尾（`append`）。

```py
def all_z_words(wordlist : list) -> list:
    """produce list of words from the input that contain z"""
    zlist = [] # start with an empty list
    for wd in wordlist:
        if "z" in wd:
            zlist = [wd] + zlist
    return(zlist)
```

此代码遵循`sum_list`的结构，即我们使用类似于在 Pyret 中会使用的表达式更新`zlist`的值。对于那些有先前的 Python 经验，在这里会使用`zlist.append`的人，请记住这个想法。我们将在可变列表中达到那里。

> 练习
> 
> > 为`all_z_words`编写测试。
> > 
> 练习
> 
> > 使用`filter`编写`all_z_words`的第二个版本。确保为其编写测试！
> > 
> 练习
> 
> > 对比这两个版本及其相应的测试。你注意到什么有趣的地方了吗？

##### 9.1.8.4 摘要：Python 的列表处理模板🔗 "链接至此")

正如我们在 Pyret 中编写列表处理函数时有一个模板一样，Python 中也有一个基于`for`循环的相应模板。作为提醒，那个模式如下：

```py
def func(lst: list):
  result = ...  # what to return if the input list is empty
  for item in lst:
    # combine item with the result so far
    result = ... item ... result
  return result
```

当你学习在 Python 中编写列表函数时，请记住这个模板。

##### 9.1.8.5 Pyret 中的`for each`循环🔗 "链接至此")

本节可以独立阅读，无需阅读本章的其余部分，所以如果你在介绍 Python 之前被引导到这里，请不要担心！虽然下面的内容反映了 Python 中存在的类似结构，但它是以独立的方式引入的。

前几节介绍了 Python 中的`for`循环，并展示了使用它们处理列表的模板。Pyret 可以使用以下模式做类似的事情：

```py
fun func(lst :: List) block:
  var result = ...  # what to return if the input list is empty
  for each(item from lst):
    # combine item with the result so far
    result := ... item ... result
  end
  result
end
```

在此示例中使用了几个新的语言特性，这些特性将在接下来的几个章节中介绍。

##### 9.1.8.5.1 可变变量🔗 "链接至此")

首先，注意我们使用`var result`引入变量`result`——这意味着它可以变化，这对于与`for each`一起使用很重要。

默认情况下，程序目录中的所有变量永远不能更改。也就是说，如果我定义了一个变量`x`，我以后就不能重新定义它：

```py
x = 10
# ...
x = 20 # produces shadowing error
```

如果我们稍后确实想要更改（或变异）目录中的变量，我们可以这样做，但我们必须声明该变量可以更改——也就是说，当我们定义它时，而不是写`x = 10`，我们必须写`var x = 10`。然后，当我们想要更新它时，我们可以使用`:=`运算符来更新，就像模板上面所做的那样。

```py
var x = 10
# ... x points to 10 in directory
x := 20
# ... x now points to 20 in directory
```

注意，尝试在未使用`var`声明的变量上使用`:=`会产生错误，并且变量仍然只能声明一次（无论是使用`var x = ...`还是`x = ...`）。

##### 9.1.8.5.2 块注释🔗 "链接至此")

在这些示例中展示的另一个新语言特性是，由于 Pyret 函数默认只期望一个（非定义）表达式，我们必须在顶部添加`block`注释，表示函数的主体是多个表达式，最后一个表达式是函数求值的结果。

作为另一个例子，如果我们尝试编写：

```py
fun my-function():
  1
  2
end
```

Pyret 会（正确地）报错——因为函数返回其主体中的最后一个表达式，所以`1`将被忽略——这很可能是错误！也许目标是编写：

```py
fun my-function():
  1 + 2
end
```

然而，由于`for each`表达式仅用于修改变量，包含它们的函数将始终有多个表达式，因此我们需要通知 Pyret 这不是错误。在函数开始的`:`之前添加`block`（或者一般地，将任何表达式包裹在`block:`和`end`中）通知 Pyret 我们理解存在多个表达式，我们只想评估最后一个。因此，如果我们真正想要编写第一个示例中的函数，我们可以这样做：

```py
fun my-function() block:
  1
  2
end
```

##### 9.1.8.5.3 `for each`如何工作🔗 "链接至此")

一个`for each`表达式对其输入列表中的每个元素运行其主体一次，在遍历过程中为每个元素在程序目录中添加一个条目。它不会直接产生任何值，因此更多地依赖于修改变量（如上所述）来产生计算。

考虑求和一组数字。我们可以编写一个函数来完成这个任务，遵循我们的模式，如下所示：

```py
fun sum-list(lst :: List) block:
  var run_total = 0
  for each(item from lst):
    run_total := item + run_total
  end
  run_total
where:
  sum-list([list: 5, 1, 7, 3]) is 16
end
```

在具体的测试输入`[list: 5, 1, 7, 3]`上，循环运行四次，一次将`item`设置为`5`，然后设置为`1`，然后设置为`7`，最后设置为`3`。

`for each`构造避免了我们多次编写常见代码，并且还处理了我们所处理的列表可以是任意长度的事实（因此我们无法预测需要编写多少次常见代码）。因此，发生的情况是：

```py
run_total = 0
run_total = run_total + 5
run_total = run_total + 1
run_total = run_total + 7
run_total = run_total + 3
```

##### 9.1.8.5.4 测试和可能变化的变量🔗 "链接至此")

我们故意展示了使用可变变量的特定模式。虽然还有其他用途（部分在可变变量中探讨），但坚持使用这个特定模板的主要原因在于测试的困难，相应地，理解使用它们的代码也很困难。

特别注意，这个模式意味着我们永远不会在函数外部定义可以改变的变量，这意味着它永远不能被不同的函数或多次函数调用使用。每次函数运行时，都会创建一个新的变量，它在`for each`循环中被修改，然后返回值，并且程序目录中的条目被删除。

考虑如果我们不遵循我们的模式会发生什么。假设我们遇到了以下问题：

> 练习
> 
> > 给定一个数字列表，返回列表的前缀（即从开始的所有元素），其和小于 100。

在了解了可变变量之后，但没有遵循模式，你可能会写出这样的代码：

```py
var count = 0

fun prefix-under-100(l :: List) -> List:
  var output = [list: ]
  for each(elt from l):
    count := count + elt
    when (count < 100):
      output := output + [list: elt]
    end
  end
end
```

现在，这看起来可能合情合理——我们使用了一个新的构造，`when`，这是一个没有`else`的`if`表达式——这只有在`for each`块内部才有意义，因为我们不需要一个结果值。它相当于：

```py
if (count < 100):
  output := output + [list: elt]
else:
  nothing
end
```

其中`nothing`是 Pyret 中用来表示没有特定重要值的值。

但当我们使用这个函数时会发生什么呢？

```py
check:
    prefix-under-100([list: 1, 2, 3]) is [list: 1, 2, 3]
    prefix-under-100([list: 20, 30, 40]) is [list: 20, 30, 40]
    prefix-under-100([list: 80, 20, 10]) is [list: 80]
end
```

前两个测试通过了，但最后一个没有通过。为什么？如果我们再次运行第一个测试，事情会变得更加混乱，即如果我们不是运行上面的`check`块，而是运行这个：

```py
check:
    prefix-under-100([list: 1, 2, 3]) is [list: 1, 2, 3]
    prefix-under-100([list: 20, 30, 40]) is [list: 20, 30, 40]
    prefix-under-100([list: 80, 20, 10]) is [list: 80]
    prefix-under-100([list: 1, 2, 3]) is [list: 1, 2, 3]
end
```

现在最初通过测试的测试不再通过了！

我们看到的是，由于变量在函数外部，它被不同函数调用共享。它被添加到程序目录中一次，每次我们调用`prefix-under-100`时，程序目录条目都会改变，但它永远不会重置。

故意地，所有其他使用变异的方法都只针对为函数主体创建的目录条目，这意味着当函数退出时，它们会被删除。但现在，我们总是在修改单个`count`变量。这意味着每次我们调用`prefix-under-100`时，它的行为都不同，因为我们不仅必须理解函数主体中的代码，还必须知道计数变量的当前值，而这并不是仅仅通过查看代码就能弄清楚的事情！

具有这种行为的函数被称为有“副作用”，它们更难测试，也更难理解，因此更容易出现错误！虽然上面的例子在相对直接的方式上是错误的，但副作用可以导致非常微妙的错误，这些错误只有在以特定顺序调用函数时才会发生——这些顺序可能只在非常具体的情况下出现，使得它们难以理解或重现。

虽然在某些地方这样做是必要的，但几乎所有代码都可以不产生副作用地编写，并且将更加可靠。我们将在修改变量中探讨我们可能想要这样做的一些情况。

##### 9.1.8.5.1 可变变量🔗 "链接至此")

首先，请注意我们使用`var result`引入了变量`result`——这意味着它可以变化，这对于与`for each`一起使用很重要。

默认情况下，程序目录中的所有变量都不能更改。也就是说，如果我定义了一个变量`x`，我以后就不能重新定义它：

```py
x = 10
# ...
x = 20 # produces shadowing error
```

如果我们以后确实想要更改（或修改）目录中的变量，我们可以这样做，但我们必须声明该变量可以更改——也就是说，当我们定义它时，而不是写`x = 10`，我们必须写`var x = 10`。然后，当我们想要更新它时，我们可以使用`:=`运算符来更新，就像模板中所示的那样。

```py
var x = 10
# ... x points to 10 in directory
x := 20
# ... x now points to 20 in directory
```

注意，尝试在未使用`var`声明的变量上使用`:=`会产生错误，并且变量仍然只能声明一次（无论是使用`var x = ...`还是`x = ...`）。

##### 9.1.8.5.2 块注释🔗 "链接至此")

这些示例中展示的另一个新语言特性是，由于 Pyret 函数默认只期望一个（非定义）表达式，我们必须在顶部添加`block`注解，表示函数的主体包含多个表达式，最后一个表达式是函数评估的结果。

作为另一个例子，如果我们尝试编写：

```py
fun my-function():
  1
  2
end
```

Pyret 会（正确地）报错——因为函数返回其主体中的最后一个表达式，所以`1`将被忽略——这很可能是错误！也许目标是编写：

```py
fun my-function():
  1 + 2
end
```

然而，由于`for each`表达式仅用于修改变量，包含它们的函数将始终有多个表达式，因此我们需要通知 Pyret 这并不是一个错误。在函数开始处的冒号`:`之前添加`block`（或者更一般地，将任何表达式包裹在`block:`和`end`中）会通知 Pyret 我们理解存在多个表达式，我们只想评估最后一个。因此，如果我们真正想要按照第一个示例编写一个函数，我们可以这样做：

```py
fun my-function() block:
  1
  2
end
```

##### 9.1.8.5.3 `for each`如何工作🔗 "链接至此")

一个`for each`表达式会对输入列表中的每个元素运行一次其主体，并在遍历过程中为每个元素在程序目录中添加一个条目。它不会直接产生任何值，因此更多地依赖于修改变量（如上所述）来产生计算结果。

考虑对一组数字进行求和。我们可以编写一个函数来完成这个任务，按照我们的模式，如下所示：

```py
fun sum-list(lst :: List) block:
  var run_total = 0
  for each(item from lst):
    run_total := item + run_total
  end
  run_total
where:
  sum-list([list: 5, 1, 7, 3]) is 16
end
```

在具体的测试输入`[list: 5, 1, 7, 3]`上，循环运行了四次，一次将`item`设置为`5`，然后设置为`1`，然后设置为`7`，最后设置为`3`。

`for each` 构造节省了我们多次编写常见代码的时间，并且处理了我们正在处理的列表可以是任意长度的事实（因此我们无法预测需要编写多少次常见代码）。因此，发生的情况是：

```py
run_total = 0
run_total = run_total + 5
run_total = run_total + 1
run_total = run_total + 7
run_total = run_total + 3
```

##### 9.1.8.5.4 测试和可变变量🔗 "链接到此处")

我们故意展示了一种使用可变变量的特定模式。虽然还有其他用途（部分在可变变量中探讨），但坚持这种特定模板的主要原因在于使用其他方式测试和相应地理解这些代码的困难。

特别是，请注意，这种模式意味着我们永远不会在函数外部定义可以改变的变量，这意味着它永远不能被不同的函数或多个函数调用使用。每次函数运行时，都会创建一个新的变量，它在`for each`循环中修改，然后返回其值，并且程序目录中的条目被删除。

考虑如果我们不遵循我们的模式会发生什么。让我们假设我们有一个以下问题：

> 练习
> 
> > 给定一个数字列表，返回列表的前缀（即，从开始的所有元素），其和小于 100。

虽然学习了可变变量，但没有遵循模式，你可能会写出这样的代码：

```py
var count = 0

fun prefix-under-100(l :: List) -> List:
  var output = [list: ]
  for each(elt from l):
    count := count + elt
    when (count < 100):
      output := output + [list: elt]
    end
  end
end
```

现在，这看起来可能是有道理的——我们使用了一个新的构造，`when`，这是一个没有`else`的`if`表达式——这只有在`for each`块内部才有意义，在那里我们不需要结果值。它等价于：

```py
if (count < 100):
  output := output + [list: elt]
else:
  nothing
end
```

其中`nothing`是 Pyret 中用来表示没有特定重要值的值。

但当我们使用这个函数时会发生什么呢？

```py
check:
    prefix-under-100([list: 1, 2, 3]) is [list: 1, 2, 3]
    prefix-under-100([list: 20, 30, 40]) is [list: 20, 30, 40]
    prefix-under-100([list: 80, 20, 10]) is [list: 80]
end
```

前两个测试通过了，但最后一个没有通过。为什么？如果我们再次运行第一个测试，事情会变得更加混乱，即，如果我们不是运行上面的`check`块，而是运行这个：

```py
check:
    prefix-under-100([list: 1, 2, 3]) is [list: 1, 2, 3]
    prefix-under-100([list: 20, 30, 40]) is [list: 20, 30, 40]
    prefix-under-100([list: 80, 20, 10]) is [list: 80]
    prefix-under-100([list: 1, 2, 3]) is [list: 1, 2, 3]
end
```

现在最初通过测试的测试不再通过了！

我们看到的是，由于变量在函数外部，它会在函数的不同调用之间共享。它只被添加到程序目录中一次，每次我们调用`prefix-under-100`时，程序目录条目都会改变，但它永远不会重置。

故意地，所有其他使用变异的方法都只应用于仅用于函数主体的目录条目，这意味着当函数退出时，它们会被移除。但现在，我们总是在修改单个`count`变量。这意味着每次我们调用`prefix-under-100`时，它的行为都不同，因为我们不仅必须理解函数主体中的代码，还必须知道计数变量的当前值，而这并不是仅仅通过查看代码就能弄清楚的事情！

具有这种行为的函数被称为有“副作用”，它们更难测试，也更难理解，因此更容易出现错误！虽然上述例子在相对直接的方式上是错误的，但副作用可能导致非常微妙的错误，这些错误只有在函数以特定顺序被调用时才会发生——这些顺序可能只出现在非常具体的情况下，使得它们难以理解或重现。

虽然在某些地方做这件事是必要的，但几乎所有代码都可以在不产生副作用的情况下编写，这将使代码更加可靠。我们将在修改变量中探讨一些我们可能想要这样做的情况。
