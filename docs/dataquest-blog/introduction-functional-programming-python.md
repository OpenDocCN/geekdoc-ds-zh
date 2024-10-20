# 教程:Python 函数和函数式编程

> 原文：<https://www.dataquest.io/blog/introduction-functional-programming-python/>

January 25, 2018

我们大多数人都是作为一种面向对象的语言被介绍给 Python 的，但是 Python 函数对于数据科学家和程序员来说也是有用的工具。虽然开始使用类和对象很容易，但是还有其他方法来编写 Python 代码。像 Java 这样的语言很难摆脱面向对象的思维，但是 Python 让这变得很容易。

鉴于 Python 为不同的代码编写方法提供了便利，一个合乎逻辑的后续问题是:什么是不同的代码编写方法？虽然这个问题有几个答案，但最常见的替代代码编写风格被称为**函数式编程**。函数式编程因编写函数而得名，函数是程序中逻辑的主要来源。

在本帖中，我们将:

*   通过比较函数式编程和面向对象编程来解释函数式编程的基础。
*   解释为什么你想在你自己的代码中加入函数式编程。
*   向您展示 Python 如何允许您在两者之间切换。

## 比较面向对象和函数式

介绍函数式编程最简单的方法是将其与我们已经知道的东西进行比较:面向对象编程。假设我们想要创建一个行计数器类，它接收一个文件，读取每一行，然后计算文件中的总行数。使用一个*类*，它可能看起来像下面这样:

```py
class LineCounter:
    def __init__(self, filename):
        self.file = open(filename, 'r')
        self.lines = []

        def read(self):
            self.lines = [line for line in self.file]

        def count(self):
            return len(self.lines)
```

虽然不是最好的实现，但它确实提供了对面向对象设计的洞察。在类中，有熟悉的方法和属性的概念。属性设置并检索对象的状态，方法操作该状态。

为了让这两个概念都起作用，对象的状态必须随着时间而改变。在调用了`read()`方法之后，这种状态变化在`lines`属性中很明显。作为一个例子，下面是我们如何使用这个类:

```py
# example_file.txt contains 100 lines.
lc = LineCounter('example_file.txt')
print(lc.lines)
>> []
print(lc.count())
>> 0

# The lc object must read the file to
# set the lines property.
lc.read()
# The `lc.lines` property has been changed.
# This is called changing the state of the lc
# object.
print(lc.lines)
>> [['Hello world!', ...]]
print(lc.count())
>> 100
```

一个物体不断变化的状态既是它的福也是它的祸。为了理解为什么一个变化的状态可以被看作是负面的，我们必须引入一个替代。另一种方法是将行计数器构建为一系列独立的函数。

```py
def read(filename):
    with open(filename, 'r') as f:
        return [line for line in f]

def count(lines):
    return len(lines)

example_lines = read('example_log.txt')
lines_count = count(example_lines)
```

## 使用纯函数

在前面的例子中，我们只能通过使用函数来计算行数。当我们只使用函数时，我们正在应用一种函数式编程方法，毫无疑问，这种方法叫做 [*函数式编程*](https://docs.python.org/3.6/howto/functional.html) 。函数式编程背后的概念要求函数是**无状态的**，并且仅依靠*给定的输入来产生输出。*

满足上述准则的函数称为**纯函数**。这里有一个例子来突出纯函数和非纯函数之间的区别:

```py
# Create a global variable `A`.
A = 5

def impure_sum(b):
    # Adds two numbers, but uses the
    # global `A` variable.
    return b + A

def pure_sum(a, b):
    # Adds two numbers, using
    # ONLY the local function inputs.
    return a + b

print(impure_sum(6))
>> 11

print(pure_sum(4, 6))
>> 10
```

使用纯函数优于不纯(非纯)函数的好处是减少了**副作用**。当在函数的操作中执行了超出其范围的更改时，就会产生副作用。例如，当我们改变一个对象的状态，执行任何 I/O 操作，甚至调用`print()`:

```py
def read_and_print(filename):
    with open(filename) as f:
        # Side effect of opening a
        # file outside of function.
        data = [line for line in f]
    for line in data:
        # Call out to the operating system
        # "println" method (side effect).
        print(line)
```

程序员减少代码中的副作用，使代码更容易跟踪、测试和调试。代码库的副作用越多，就越难逐步完成一个程序并理解它的执行顺序。

虽然尝试和消除所有副作用很方便，但它们通常用于简化编程。如果我们禁止所有的副作用，那么你将不能读入一个文件，调用 print，甚至不能在一个函数中赋值一个变量。函数式编程的倡导者理解这种权衡，并试图在不牺牲开发实现时间的情况下尽可能消除副作用。

## λ表达式

代替函数声明的`def`语法，我们可以使用一个 [`lambda`表达式](https://docs.python.org/3.5/tutorial/controlflow.html#lambda-expressions)来编写 Python 函数。lambda 语法严格遵循`def`语法，但它不是一对一的映射。下面是一个构建两个整数相加的函数的示例:

```py
# Using `def` (old way).
def old_add(a, b):
    return a + b

# Using `lambda` (new way).
new_add = lambda a, b: a + bold_add(10, 5) == new_add(10, 5)
>> True
```

`lambda`表达式接受逗号分隔的输入序列(像`def`)。然后，紧跟在冒号后面，不使用显式 return 语句返回表达式。最后，当将`lambda`表达式赋给一个变量时，它的行为就像一个 Python 函数，并且可以使用函数调用语法:`new_add()`来调用。

如果我们不把`lambda`赋给一个变量名，它将被称为一个**匿名函数**。这些匿名函数非常有用，尤其是在将它们用作另一个函数的输入时。例如，`sorted()` [函数](https://docs.python.org/3/howto/sorting.html#key-functions)接受一个可选的`key`参数(一个函数),描述列表中的项目应该如何排序。

```py
unsorted = [('b', 6), ('a', 10), ('d', 0), ('c', 4)]

# Sort on the second tuple value (the integer).
print(sorted(unsorted, key=lambda x: x[1]))
>> [('d', 0), ('c', 4), ('b', 6), ('a', 10)]
```

## 地图功能

虽然将函数作为参数传递的能力并不是 Python 所独有的，但这是编程语言的一项最新发展。允许这种行为的函数被称为**一级函数**。任何包含一级函数*的语言都可以用函数式风格编写*。

在函数范式中有一组常用的重要的一级函数。这些函数接受一个 [Python iterable](https://docs.python.org/3/glossary.html#term-iterable) ，并且像`sorted()`一样，为列表中的每个元素应用一个函数。在接下来的几节中，我们将检查这些函数中的每一个，但是它们都遵循`function_name(function_to_apply, iterable_of_elements)`的一般形式。

我们将使用的第一个函数是`map()`函数。`map()`函数接受一个 iterable(即。`list`)，并创建一个新的可迭代对象，一个特殊的地图对象。新对象将一级函数应用于每个元素。

```py
# Pseudocode for map.
def map(func, seq):
    # Return `Map` object with
    # the function applied to every
    # element.
    return Map(
        func(x)
        for x in seq
    )
```

下面是我们如何使用 map 将`10`或`20`添加到列表中的每个元素:

```py
values = [1, 2, 3, 4, 5]

# Note: We convert the returned map object to
# a list data structure.
add_10 = list(map(lambda x: x + 10, values))
add_20 = list(map(lambda x: x + 20, values))

print(add_10)
>> [11, 12, 13, 14, 15]

print(add_20)
>> [21, 22, 23, 24, 25]
```

注意，将来自`map()`的返回值转换为一个`list`对象是很重要的。如果你希望返回的`map`对象像`list`一样工作，那么使用它是很困难的。首先，打印它不会显示它的每一项，其次，您只能迭代它一次。

## 过滤功能

我们要使用的第二个函数是`filter()`函数。`filter()`函数接受一个 iterable，创建一个新的 iterable 对象(同样是一个特殊的`map`对象)，以及一个必须返回一个`bool`值的一级函数。新的`map`对象是所有返回`True`的元素的过滤后的 iterable。

```py
# Pseudocode for filter.
def filter(evaluate, seq):
    # Return `Map` object with
    # the evaluate function applied to every
    # element.
    return Map(
        x for x in seq
        if evaluate(x) is True
    )
```

下面是我们如何从列表中过滤奇数或偶数值:

```py
values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Note: We convert the returned filter object to
# a list data structure.
even = list(filter(lambda x: x % 2 == 0, values))
odd = list(filter(lambda x: x % 2 == 1, values))

print(even)
>> [2, 4, 6, 8, 10]

print(odd)
>> [1, 3, 5, 7, 9]
```

## Reduce 函数

我们要看的最后一个函数是 [`functools`包](https://docs.python.org/3/library/functools.html)中的`reduce()`函数。`reduce()`函数接受一个可迭代对象，然后将该可迭代对象简化为一个值。Reduce 不同于`filter()`和`map()`，因为`reduce()`接受一个有两个输入值的函数。

这里有一个例子，说明我们如何使用`reduce()`对列表中的所有元素求和。

```py
from functools import reduce

values = [1, 2, 3, 4]

summed = reduce(lambda a, b: a + b, values)
print(summed)
>> 10
```

![diagram of reduce](img/f7217b82cd69495e0959904d9a374ad3.png)

一个有趣的注意事项是，你不需要**而不是**去操作`lambda`表达式的第二个值。例如，您可以编写一个总是返回 iterable 的第一个值的函数:

```py
from functools import reduce

values = [1, 2, 3, 4, 5]

# By convention, we add `_` as a placeholder for an input
# we do not use.
first_value = reduce(lambda a, _: a, values)
print(first_value)
>> 1
```

## 用列表理解重写

因为我们最终会转换成列表，所以我们应该使用列表理解来重写`map()`和`filter()`函数。这是编写列表的更为*Python 化的*方式，因为我们利用了 Python 的语法来制作列表。以下是你如何将前面的`map()`和`filter()`的例子翻译成列表理解:

```py
values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Map.
add_10 = [x + 10 for x in values]
print(add_10)
>> [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# Filter.
even = [x for x in values if x % 2 == 0]
print(even)
>> [2, 4, 6, 8, 10]
```

从例子中，你可以看到我们不需要添加 lambda 表达式。如果您希望在自己的代码中添加`map()`或`filter()`函数，这通常是推荐的方式。然而，在下一节中，我们将提供一个仍然使用`map()`和`filter()`函数的案例。

## 书写函数分部

有时，我们希望使用函数的行为，但减少它的参数数量。目的是“保存”其中一个输入，并使用保存的输入创建一个默认行为的新函数。假设我们想写一个函数，它总是将 2 加到任何数上:

```py
def add_two(b):
    return 2 + b

print(add_two(4))
>> 6
```

`add_two`函数类似于一般的函数，$f(a，b) = a + b$，只是它默认了其中一个参数($a = 2$)。在 Python 中，我们可以使用`functools`包中的[模块](https://docs.python.org/3.6/library/functools.html#functools.partial)来设置这些参数默认值。`partial`模块接收一个函数，并从第一个参数开始“冻结”任意数量的参数(或 kwargs ),然后返回一个带有默认输入的新函数。

```py
from functools import partialdef add(a, b):
    return a + b

add_two = partial(add, 2)
add_ten = partial(add, 10)

print(add_two(4))
>> 6
print(add_ten(4))
>> 14
```

分部可以接受任何函数，包括标准库中的函数。

```py
# A partial that grabs IP addresses using
# the `map` function from the standard library.
extract_ips = partial(
    map,
    lambda x: x.split(' ')[0]
)
lines = read('example_log.txt')
ip_addresses = list(extract_ip(lines))
```

## 后续步骤

在这篇文章中，我们介绍了函数式编程的范例。我们学习了 Python 中的 lambda 表达式、重要的函数以及偏导数的概念。总之，我们展示了 Python 为程序员提供了在函数式编程和面向对象编程之间轻松切换的工具。

查看其他一些可能对您有帮助的资源:

*   [Python 教程](https://www.dataquest.io/python-tutorials-for-data-science/) —我们不断扩充的数据科学 Python 教程列表。
*   [数据科学课程](https://www.dataquest.io/path/data-scientist/) —通过完全交互式的编程、数据科学和统计课程，让您的学习更上一层楼，就在您的浏览器中。

## 这个教程有帮助吗？

选择你的道路，不断学习有价值的数据技能。

![arrow down left](img/2215dd1efd21629477b52ea871afdd98.png)![arrow right down](img/2e703f405f987a154317ac045ee00a68.png)[Python Tutorials](/python-tutorials-for-data-science/)

在我们的免费教程中练习 Python 编程技能。

[Data science courses](/data-science-courses/)

通过我们的交互式浏览器数据科学课程，投入到 Python、R、SQL 等语言的学习中。