# 教程:为什么函数在 Python 中修改列表和字典

> 原文：<https://www.dataquest.io/blog/tutorial-functions-modify-lists-dictionaries-python/>

March 14, 2019![python-data-science-tutorial-mutable-data-types](img/60f9e5fb67b7196578110b07788a275f.png)

Python 的函数(包括[内置函数](https://docs.python.org/3/library/functions.html)和我们自己编写的自定义函数)是处理数据的重要工具。但是他们对我们的数据所做的事情可能会有点混乱，如果我们不知道发生了什么，这可能会在我们的分析中造成严重的错误。

在本教程中，我们将仔细研究 Python 如何处理在函数内部被操纵的不同数据类型，并学习如何确保我们的数据仅在我们*希望*改变时才被改变。

## 函数中的内存隔离

为了理解 Python 如何处理函数内部的全局变量，让我们做一个小实验。我们将创建两个全局变量 number_1 和 number_2，并将它们赋给整数 5 和 10。然后，我们将使用这些全局变量作为执行一些简单数学运算的函数的参数。我们还将使用变量名作为函数的参数名。然后，我们会看到函数中所有变量的使用是否影响了这些变量的全局值。

```py
number_1 = 5
number_2 = 10

def multiply_and_add(number_1, number_2):
    number_1 = number_1 * 10
    number_2 = number_2 * 10
    return number_1 + number_2

a_sum = multiply_and_add(number_1, number_2)
print(a_sum)
print(number_1)
print(number_2)
```

```py
150
5
10
```

正如我们在上面看到的，函数工作正常，全局变量`number_1`和`number_2`的值*没有*改变，即使我们在函数中使用它们作为自变量*和*参数名。这是因为 Python 将函数中的变量存储在与全局变量不同的内存位置。他们被孤立了。因此，变量`number_1`可以在全局范围内有一个值(5 ),在函数内部有一个不同的值(50 ),在函数内部它是独立的。

(顺便说一句，如果你对*参数*和*参数*、[之间的区别感到困惑，Python 关于这个主题的文档是很有帮助的](https://docs.python.org/3.3/faq/programming.html#faq-argument-vs-parameter)。)

## 列表和字典呢？

### 列表

我们已经看到，我们在函数中对上面的`number_1`这样的变量所做的事情不会影响它的全局值。但是`number_1`是一个整数，这是一个非常基本的数据类型。如果我们用不同的数据类型(比如列表)进行相同的实验，会发生什么呢？下面，我们将创建一个名为`duplicate_last()`的函数，它将复制我们作为参数传递的任何列表中的最后一个条目。

```py
initial_list = [1, 2, 3]

def duplicate_last(a_list):
    last_element = a_list[-1]
    a_list.append(last_element)
    return a_list

new_list = duplicate_last(a_list = initial_list)
print(new_list)
print(initial_list)
```

```py
[1, 2, 3, 3]
[1, 2, 3, 3]
```

正如我们所看到的，这里`initial_list` *的全局值被*更新了，尽管它的值只是在函数内部被改变了！

### 字典

现在，让我们编写一个以字典作为参数的函数，看看在函数内部操作全局字典变量时，它是否会被修改。

为了让这个更真实一点，我们将使用在我们的 [Python 基础课程](https://www.dataquest.io/course/python-for-data-science-fundamentals)中使用的`AppleStore.csv`数据集的数据(数据可以在这里[下载)。](https://www.kaggle.com/ramamet4/app-store-apple-data-set-10k-apps)

在下面的代码片段中，我们从一个字典开始，该字典包含数据集中每个年龄等级的应用程序的数量(因此有 4，433 个应用程序被评为“4+”，987 个应用程序被评为“9+”，等等)。).假设我们想要计算每个年龄分级的百分比，这样我们就可以了解 App Store 中哪些年龄分级是最常见的。

为此，我们将编写一个名为`make_percentages()`的函数，它将一个字典作为参数，并将计数转换为百分比。我们需要从零开始计数，然后遍历字典中的每个值，将它们添加到计数中，这样我们就可以得到评级的总数。然后我们将再次遍历字典，对每个值做一些数学运算来计算百分比。

```py
content_ratings = {'4+': 4433, '9+': 987, '12+': 1155, '17+': 622}

def make_percentages(a_dictionary):
    total = 0
    for key in a_dictionary:
        count = a_dictionary[key]
        total += count

    for key in a_dictionary:
        a_dictionary[key] = (a_dictionary[key] / total) * 100

    return a_dictionary
```

在我们查看输出之前，让我们快速回顾一下上面发生了什么。在将我们的应用年龄分级字典分配给变量`content_ratings`之后，我们创建了一个名为`make_percentages()`的新函数，它接受一个参数:`a_dictionary`。

为了计算属于每个年龄等级的应用程序的百分比，我们需要知道应用程序的总数，所以我们首先将一个名为`total`的新变量设置为`0`，然后遍历`a_dictionary`中的每个键，将其添加到`total`。

一旦完成，我们需要做的就是再次遍历`a_dictionary`，将每个条目除以总数，然后将结果乘以 100。这会给我们一个带有百分比的字典。

但是当我们使用全局变量`content_ratings`作为这个新函数的参数时会发生什么呢？

```py
c_ratings_percentages = make_percentages(content_ratings)
print(c_ratings_percentages)
print(content_ratings)
```

```py
{'4+': 61.595109073224954, '9+': 13.714047519799916, '12+': 16.04835348061692, '17+': 8.642489926358204}
{'4+': 61.595109073224954, '9+': 13.714047519799916, '12+': 16.04835348061692, '17+': 8.642489926358204}
```

正如我们在列表中看到的那样，我们的全局`content_ratings`变量已经被修改，尽管它只是在我们创建的`make_percentages()`函数中被修改。

这里到底发生了什么？我们碰到了**可变**和**不可变**数据类型之间的差异。

## 可变和不可变数据类型

在 Python 中，数据类型可以是可变的(可改变的)或不可变的(不可改变的)。虽然我们在 Python 入门中使用的大多数数据类型都是不可变的(包括整数、浮点、字符串、布尔和元组)，但列表和字典是可变的。这意味着**即使在函数**内部使用，全局列表或字典也可以被更改，就像我们在上面的例子中看到的一样。

为了理解可变(可改变的)和不可变(不可改变的)之间的区别，看看 Python 实际上是如何处理这些变量的是有帮助的。

让我们从考虑一个简单的变量赋值开始:

```py
a = 5
```

变量名`a`就像一个指向`5`的指针，它帮助我们在任何需要的时候检索`5`。

![img](img/f5d0af6b418e0c68225d96950261c807.png) ![img](img/f5d0af6b418e0c68225d96950261c807.png)

`5`是整数，整数是不可变的数据类型。如果一个数据类型是不可变的，这意味着它一旦被创建**就不能被更新**。如果我们做了`a += 1`，我们实际上并没有将`5`更新为`6`。在下面的动画中，我们可以看到:

*   `a`最初指向`5`。
*   运行`a += 1`，这将指针从`5`移动到`6`，它实际上并没有改变数字`5`。

![py1m6_imm_correct](img/f6e9e945177efd9867021e992bfdbec5.png) ![py1m6_imm_correct](img/f6e9e945177efd9867021e992bfdbec5.png)

列表和字典等可变数据类型的行为不同。他们 ***可以*更新**。例如，让我们列一个简单的清单:

```py
list_1 = [1, 2]
```

如果我们将一个`3`追加到这个列表的末尾，我们不只是简单地将`list_1`指向一个不同的列表，我们是直接更新现有的列表:

![img](img/24fc6c7d3e5f254896dc13777e23b966.png) ![img](img/24fc6c7d3e5f254896dc13777e23b966.png)

即使我们创建了多个列表变量，只要它们指向同一个列表，当列表改变时，它们都会被更新，如下面的代码所示:

```py
list_1 = [1, 2]
list_2 = list_1
list_1.append(3)
print(list_1)
print(list_2)
```

```py
[1, 2, 3]
[1, 2, 3]
```

下面是上面代码中实际发生的事情的动画可视化:

![img](img/fa36d4b05208c8afa0368ef53363fe01.png) ![img](img/fa36d4b05208c8afa0368ef53363fe01.png)

这就解释了为什么我们在早期试验列表和字典时，我们的全局变量被改变了。因为列表和字典是可变的，所以改变它们(即使在函数内部)也会改变列表或字典本身，这对于不可变的数据类型来说不是这样。

## 保持可变数据类型不变

一般来说，我们不希望函数改变全局变量，即使它们包含可变的数据类型，比如列表或字典。这是因为在更复杂的分析和程序中，我们可能会频繁地使用许多不同的函数。如果他们都在改变他们正在工作的列表和字典，那么跟踪什么在改变什么会变得相当困难。

谢天谢地，有一个简单的方法可以解决这个问题:我们可以使用一个名为`.copy()`的内置 Python 方法来复制列表或字典。

如果你还没有学过方法，不要着急。它们包含在[我们的中级 Python 课程](https://www.dataquest.io/course/python-for-data-science-intermediate)中，但是对于本教程，你需要知道的是`.copy()`的工作方式类似于`.append()`:

```py
list.append() # adds something to a list
list.copy() # makes a copy of a list
```

让我们再来看看我们为列表编写的函数，并更新它，这样函数*内部发生的事情*不会改变`initial_list`。我们需要做的就是将传递给函数的参数从`initial_list`改为`initial_list.copy()`。

```py
initial_list = [1, 2, 3]

def duplicate_last(a_list):
    last_element = a_list[-1]
    a_list.append(last_element)
    return a_list

new_list = duplicate_last(a_list = initial_list.copy()) # making a copy of the list
print(new_list)
print(initial_list)
```

```py
[1, 2, 3, 3]
[1, 2, 3]
```

如我们所见，这已经解决了我们的问题。原因如下:使用`.copy()`创建了一个单独的列表副本，因此`a_list`不是指向`initial_list`本身，而是指向一个以`initial_list`的副本开始的新列表。在那之后对`a_list`所做的任何改变都是针对那个单独的列表，而不是`initial_list`本身，因此`initial_list`的全局值是不变的。

![img](img/57767c9a88409aed632510c04a869b24.png) ![img](img/57767c9a88409aed632510c04a869b24.png)

然而，这个解决方案仍然不是完美的，因为我们必须记住在每次传递一个参数给我们的函数时添加`.copy()`，否则会有意外改变`initial_list`的全局值的风险。如果我们不想为此担心，我们实际上可以在函数本身内部创建列表副本:

```py
initial_list = [1, 2, 3]

def duplicate_last(a_list):
    copy_list = a_list.copy() # making a copy of the list
    last_element = copy_list[-1]
    copy_list.append(last_element)
    return copy_list

new_list = duplicate_last(a_list = initial_list)
print(new_list)
print(initial_list)
```

```py
[1, 2, 3, 3]
[1, 2, 3]
```

使用这种方法，我们可以安全地将像`initial_list`这样的可变全局变量传递给我们的函数，并且全局值不会被改变，因为函数本身制作了一个副本，然后在该副本上执行它的操作。

`.copy()`方法也适用于字典。与列表一样，我们可以简单地将`.copy()`添加到参数中，我们传递函数来创建一个将用于该函数的副本，而不改变原始变量:

```py
content_ratings = {'4+': 4433, '9+': 987, '12+': 1155, '17+': 622}

def make_percentages(a_dictionary):
    total = 0
    for key in a_dictionary:
        count = a_dictionary[key]
        total += count

    for key in a_dictionary:
        a_dictionary[key] = (a_dictionary[key] / total) * 100

    return a_dictionary

c_ratings_percentages = make_percentages(content_ratings.copy()) # making a copy of the dictionary
print(c_ratings_percentages)
print(content_ratings)
```

```py
{'4+': 61.595109073224954, '9+': 13.714047519799916, '12+': 16.04835348061692, '17+': 8.642489926358204}
{'4+': 4433, '9+': 987, '12+': 1155, '17+': 622}
```

但是同样，使用这种方法意味着我们需要记住在每次将字典传递给函数`make_percentages()`时添加`.copy()` *。如果我们要经常使用这个函数，那么最好将复制操作实现到函数本身中，这样我们就不必记住这样做了。*

下面，我们将在函数内部使用`.copy()`。这将确保我们可以在不改变作为参数传递给它的全局变量的情况下使用它，并且我们不需要记住给我们传递的每个参数添加`.copy()`。

```py
content_ratings = {'4+': 4433, '9+': 987, '12+': 1155, '17+': 622}

def make_percentages(a_dictionary):
    copy_dict = a_dictionary.copy() # create a copy of the dictionary
    total = 0
    for key in a_dictionary:
        count = a_dictionary[key]
        total += count

    for key in copy_dict: #use the copied table so original isn't changed
        copy_dict[key] = (copy_dict[key] / total) * 100

    return copy_dict

c_ratings_percentages = make_percentages(content_ratings)
print(c_ratings_percentages)
print(content_ratings)
```

```py
{'4+': 61.595109073224954, '9+': 13.714047519799916, '12+': 16.04835348061692, '17+': 8.642489926358204}
{'4+': 4433, '9+': 987, '12+': 1155, '17+': 622}
```

正如我们所看到的，修改我们的函数来创建我们的字典的副本，然后只在副本中将计数更改为百分比*，这允许我们执行我们想要的操作，而不实际更改`content_ratings`。*

## 结论

在本教程中，我们研究了可变数据类型(可以改变)和不可变数据类型(不能改变)之间的区别。我们学习了如何使用方法`.copy()`来复制可变数据类型，比如列表和字典，这样我们就可以在函数*中使用它们，而不需要*改变它们的全局值。

## 这个教程有帮助吗？

选择你的道路，不断学习有价值的数据技能。

![arrow down left](img/2215dd1efd21629477b52ea871afdd98.png)![arrow right down](img/2e703f405f987a154317ac045ee00a68.png)[Python Tutorials](/python-tutorials-for-data-science/)

在我们的免费教程中练习 Python 编程技能。

[Data science courses](/data-science-courses/)

通过我们的交互式浏览器数据科学课程，投入到 Python、R、SQL 等语言的学习中。