# Python For 循环的基础:教程

> 原文：<https://www.dataquest.io/blog/python-for-loop-tutorial/>

May 30, 2019![python-for-loop-tutorial](img/3437e4d94a4e0cb58da101ae5df7cb84.png)

当您在 Python 中处理数据时， **for loops** 可能是一个强大的工具。但是当你刚开始的时候，它们也可能有点令人困惑。在本教程中，我们将一头扎进 for 循环，并了解当您在 Python 中进行数据清理或数据分析时，如何使用它们来做各种有趣的事情。

本教程是为 Python 初学者编写的，但是如果你以前从未写过一行代码，你可能想从我们免费的 Python 基础课程的开头开始，因为我们在这里不会涉及基本语法。

## 什么是循环？

在大多数数据科学工作的上下文中，Python for loops 用于遍历一个**可迭代对象**(如列表、元组、集合等)。)并对每个条目执行相同的操作。例如，For 循环允许我们遍历列表，对列表中的每一项执行相同的操作。

(顺便说一下，可交互对象是我们可以迭代或“循环”遍历的任何 Python 对象，每次返回一个元素。例如，列表是可迭代的，并且按照条目被列出的顺序，一次返回一个列表条目。字符串是可迭代的，按照字符出现的顺序，一次返回一个字符。等等。)

创建 for 循环的方法是，首先定义要循环的 iterable 对象，然后定义要对 iterable 对象中的每一项执行的操作。例如，当遍历一个列表时，首先指定想要遍历的列表，然后用*和*指定想要对每个列表项执行什么操作。

让我们看一个简单的例子:如果我们有一个用 Python 存储的名字列表，我们可以使用一个 for 循环来遍历这个列表，打印每个名字，直到它到达末尾。下面，我们将创建我们的名字列表，然后编写一个 for 循环来遍历它，按顺序打印列表中的每个条目。

```py
our_list = ['Lily', 'Brad', 'Fatima', 'Zining']

for name in our_list:
    print(name) 
```

```py
Lily
Brad
Fatima
Zining 
```

这个简单循环中的代码提出了一个问题:**变量`name`来自哪里？**我们之前没有在代码中定义它！但是因为 for 循环遍历列表、元组等。按顺序，这个变量实际上可以被称为几乎任何东西。当循环执行时，Python 会将我们放在那个位置的任何变量名解释为按顺序引用每个列表条目。

所以，在上面的代码中:

*   `name`在循环的第一次迭代时指向`'Lily'`…
*   …然后`'Brad'`在循环的第二次迭代中…
*   …等等。

不管我们怎么称呼这个变量，情况都会如此。因此，举例来说，如果我们重写代码，用`x`替换`name`，我们将得到完全相同的结果:

```py
for x in our_list:
    print(x) 
```

```py
Lily
Brad
Fatima
Zining 
```

注意，这种技术适用于任何可迭代对象。例如，字符串是可迭代的，我们可以使用相同类型的 For 循环来迭代字符串中的每个字符:

```py
for letter in 'Lily':
    print(letter) 
```

```py
L
i
l
y 
```

## 对列表的列表使用 For 循环

然而，在实际的数据分析工作中，我们不太可能使用如上所述的简短列表。一般来说，我们必须处理表格格式的数据集，有多行和多列。这种数据可以在 Python 中存储为列表的列表，其中表的每一行都存储为列表列表中的列表，我们也可以使用 for 循环来遍历这些列表。

为了了解如何做到这一点，让我们看看一个更现实的场景，并探索这个小数据表，其中包含一些美国价格和几款电动汽车的[美国环保局范围估计值](https://www.epa.gov/greenvehicles/explaining-electric-plug-hybrid-electric-vehicles)。

| 车辆 | 范围 | 价格 |
| --- | --- | --- |
| 特斯拉 Model 3 LR | Three hundred and ten | Forty-nine thousand nine hundred |
| 现代离子 EV | One hundred and twenty-four | Thirty thousand three hundred and fifteen |
| 雪佛兰博尔特 | Two hundred and thirty-eight | Thirty-six thousand six hundred and twenty |

我们可以将相同的数据集表示为列表的列表，如下所示:

```py
ev_data = [['vehicle', 'range', 'price'], 
           ['Tesla Model 3 LR', '310', '49900'], 
           ['Hyundai Ioniq EV', '124', '30315'],
           ['Chevy Bolt', '238', '36620']] 
```

您可能已经注意到，在上面的列表中，我们的产品系列和价格数字实际上是以字符串而不是整数的形式存储的。以这种方式存储数据并不少见，但是为了便于分析，我们希望将这些字符串转换成整数，以便用它们进行一些计算。让我们使用一个 for 循环来遍历我们的列表，选择每个列表中的`price`条目，并将其从字符串更改为整数。

为此，我们需要做几件事。首先，我们需要跳过表中的第一行，因为它们是列名，如果我们试图将非数字字符串如`'range'`转换成整数，就会出现错误。我们可以使用列表切片来选择第一行之后的每一行*。(如果你需要温习这一点，或者列表的任何其他方面，请查看[我们关于 Python 编程基础的互动课程](https://www.dataquest.io/course/python-for-data-science-fundamentals/))。*

然后，我们将遍历列表列表，对于每次迭代，我们将选择`range`列中的元素，这是表中的第二列。我们将把在该列中找到的值赋给一个名为`'range'`的变量。为此，我们将使用索引号`1`(在 Python 中，iterable 中的第一个条目位于索引`0`，第二个条目位于索引`1`，依此类推。).

最后，我们将使用 Python 内置的`'int()`函数将范围数转换成整数，并在我们的数据集中用这些整数替换原始字符串。

```py
for row in ev_data[1:]:         # loop through each row in ev_data starting with row 2 (index 1)
    ev_range = row[1]           # each car's range is found in column 2 (index 1)
    ev_range = int(ev_range)    # convert each range number from a string to an integer
    row[1] = ev_range           # assign range, which is now an integer, back to index 1 in each row

print(ev_data) 
```

```py
[['vehicle', 'range', 'price'], ['Tesla Model 3 LR', 310, '49900'], ['Hyundai Ioniq EV', 124, '30315'], ['Chevy Bolt', 238, '36620']] 
```

既然我们已经将这些值存储为整数，我们还可以使用 for 循环来进行一些计算。比方说，我们想算出这个列表中一辆电动汽车的平均里程。我们需要将范围数加在一起，然后除以列表中的汽车总数。

同样，我们可以使用 for 循环来选择数据集中需要的特定列。我们将从创建一个名为`total_range`的变量开始，在这里我们可以存储范围的总和。然后，我们将编写另一个 for 循环，再次跳过标题行，再次将第二列(索引 1)标识为范围值。

之后，我们需要做的就是在我们的 for 循环中把这个值加到`total_range`上，然后在循环完成后用`total_range`除以汽车数量来计算这个值。

(注意，在下面的代码中，我们将通过计算列表的长度减去标题行来计算汽车的数量。对于像我们这样短的列表，我们也可以简单地除以 3，因为汽车的数量很容易计算，但是如果列表中添加了额外的汽车数据，就会破坏我们的计算。出于这个原因，最好使用`len()`来计算代码中汽车列表的长度，这样如果将来有额外的条目添加到我们的数据集中，我们可以重新运行这个代码，它仍然会产生正确的答案。)

```py
total_range = 0                     # create a variable to store the total range number

for row in ev_data[1:]:             # loop through each row in ev_data starting with row 2 (index 1)
    ev_range = row[1]               # each car's range is found in column 2 (index 1)
    total_range += ev_range         # add this number to the number stored in total_range

number_of_cars = len(ev_data[1:])   # calculate the length of our list, minus the header row

print(total_range / number_of_cars) # print the average range 
```

```py
224.0 
```

Python for 循环功能强大，可以在其中嵌套更复杂的指令。为了演示这一点，让我们对我们的`'price'`列重复上面的两个步骤，这次是在一个 for 循环中。

```py
total_price = 0                     # create a variable to store the total range number

for row in ev_data[1:]:             # loop through each row in ev_data starting with row 2 (index 1)
    price = row[2]                  # each car's price is found in column 3 (index 2)
    price = int(price)              # convert each price number from a string to an integer
    row[2] = price                  # assign price, which is now an integer, back to index 2 in each row
    total_price += price            # add each car's price to total_price

number_of_cars = len(ev_data[1:])   # calculate the length of our list, minus the header row

print(total_price / number_of_cars) # print the average price 
```

```py
38945.0 
```

我们还可以在 for 循环中嵌套其他元素，比如 If Else 语句，甚至其他 for 循环。

例如，假设我们希望在列表中找到每辆行驶里程超过 200 英里的汽车。我们可以从创建一个新的空列表来保存我们的长期汽车数据开始。然后，我们将使用 for 循环遍历前面创建的包含汽车数据的列表列表`ev_data`，仅当其范围值大于 200:

```py
long_range_car_list = []       # creating a new list to store our long range car data

for row in ev_data[1:]:        # iterate through ev_data, skipping the header row
    ev_range = row[1]          # assign the range number, which is at index 1 in the row, to the range variable
    if ev_range > 200:         # append the whole row to long-range list if range is higher than 200
        long_range_car_list.append(row)

print(long_range_car_list) 
```

```py
[['Tesla Model 3 LR', 310, 49900], ['Chevy Bolt', 238, 36620]] 
```

当然，对于如此小的数据集，手工执行这些操作也很简单。但是这些相同的技术将在具有成千上万行的数据集上工作，这可以使清理、排序和分析巨大的数据集成为非常快速的工作。

## 其他有用的技巧:范围、中断和继续

仅仅通过掌握上面描述的技术，您就可以从 for 循环中获得惊人的收益，但是让我们更深入地了解一些其他可能有帮助的东西，即使您在数据科学工作的上下文中不太经常使用它们。

#### 范围

For 循环可以与 Python 的`range()`函数一起使用，遍历指定范围内的每个数字。例如:

```py
for x in range(5, 9):
    print(x) 
```

```py
5
6
7
8 
```

请注意，Python 在范围计数中不包括范围的最大值，这就是为什么数字 9 没有出现在上面。如果我们想让这段代码从 5 数到 9，包括 9，我们需要将`range(5, 9)`改为`range(5, 10)`:

```py
for x in range(5, 10):
    print(x) 
```

```py
5
6
7
8
9 
```

如果在`range()`函数中只指定了一个数字，Python 会把它作为最大值，并指定一个缺省的最小值零:

```py
for x in range(3):
    print(x) 
```

```py
0
1
2 
```

您甚至可以向`range()`函数添加第三个参数，以指定您希望以特定数字的增量进行计数。正如您在上面看到的，默认值是 1，但是如果您添加第三个参数 3，例如，您可以将`range()`与 for 循环一起使用，以三为单位向上计数:

```py
for x in range(0, 9, 3):
    print(x) 
```

```py
0
3
6 
```

#### 破裂

默认情况下，Python for 循环将遍历分配给它的可交互对象的每个可能的迭代。通常，当我们使用 for 循环时，这很好，因为我们希望对列表中的每一项执行相同的操作(例如)。

但是，有时，如果满足某个条件，我们可能希望停止循环。在这种情况下，`break`语句是有用的。当在 for 循环中与 if 语句一起使用时，`break`允许我们在循环结束前中断它。

让我们先来看一个简单的例子，使用我们之前创建的名为`our_list`的列表:

```py
for name in our_list:
    break
    print(name) 
```

当我们运行这段代码时，什么也没有打印出来。这是因为在我们的 for 循环中，`break`语句在`print(name)`之前。当 Python 看到`break`时，它会停止执行 for 循环，循环中 `break`后出现的*代码不会运行。*

让我们在这个循环中添加一个 if 语句，这样当 Python 到达紫凝这个名字时，我们就可以跳出这个循环了:

```py
for name in our_list:
    if name == 'Zining':
        break
    print(name) 
```

```py
Lily
Brad
Fatima 
```

在这里，我们可以看到紫凝这个名字没有被印刷出来。下面是每次循环迭代发生的情况:

1.  Python 检查名字是否是“紫凝”。它*不是*，所以它继续执行 if 语句下面的代码，并输出名字。
2.  Python 检查第二个名字是否是“紫凝”。它*不是*，所以它继续执行 if 语句下面的代码，并打印第二个名字。
3.  Python 检查第三个名字是否是“紫凝”。它*不是*，所以它继续执行 if 语句下面的代码，并打印第三个名字。
4.  Python 检查第四个名字是否是“紫凝”。如果*为*，则`break`被执行，for 循环结束。

让我们回到我们编写的用于收集电动汽车远程数据的代码，并再看一个例子。我们将插入一个 break 语句，一旦遇到字符串`'Tesla'`就停止查找:

```py
long_range_car_list = []    # creating our empty long-range car list again

for row in ev_data[1:]:     # iterate through ev_data as before looking for cars with a range > 200
    ev_range = row[1]          
    if ev_range > 200:         
        long_range_car_list.append(row)
    if 'Tesla' in row[0]:   # but if 'Tesla' appears in the vehicle column, end the loop
            break

print(long_range_car_list) 
```

```py
[['Tesla Model 3 LR', 310, 49900]] 
```

在上面的代码中，我们可以看到特斯拉仍然被添加到了`long_range_car_list`，因为我们将它添加到了列表*中，在使用`break`的 if 语句*之前。Chevy Bolt 没有被添加到我们的列表中，因为尽管它的行驶里程超过 200 英里，`break`在 Python 到达 Chevy Bolt 行之前就结束了循环。

(记住，for 循环是按顺序执行的。如果在我们的原始数据集中，Bolt 列在 Tesla 之前，它将被包含在`long_range_car_list`中。

#### 继续

当我们循环遍历一个像列表这样的可迭代对象时，我们也可能会遇到想要跳过特定的一行或多行的情况。对于像跳过标题行这样的简单情况，我们可以使用列表切片，但是如果我们想要基于更复杂的条件跳过行，这很快就变得不切实际了。相反，我们可以使用`continue`语句跳过 for 循环的一次迭代(“循环”)，并进入下一次循环。

例如，当 Python 在列表上执行 for 循环时看到`continue`时，它将在该点停止，并移动到列表上的下一项。任何低于`continue`的代码都不会被执行。

让我们回到我们的名字列表(`our_names`)中，如果名字是“Brad”，那么在打印之前使用带有 if 语句的`continue`来结束循环迭代:

```py
for name in our_list:
    if name == 'Brad':
        continue
    print(name) 
```

```py
Lily
Fatima
Zining 
```

上面，我们可以看到布拉德的名字被跳过了，我们列表中的其余名字是按顺序打印的。简而言之，这说明了`break`和`continue`的区别:

*   `break`完全结束循环*。Python 执行`break`时，for 循环结束。*
**   `continue`结束循环的一个*特定迭代*并移动到列表中的下一项。当 Python 执行`continue`时，它会立即进入下一个循环迭代，但不会完全结束循环。*

 *为了对`continue`进行更多的练习，让我们列出一个*短程*电动车的列表，使用`continue`采取稍微不同的方法。我们将编写一个 for 循环，将每辆电动汽车的*添加到我们的短程列表中，而不是识别行驶里程小于 200 英里的电动汽车，但是在*之前有一个`continue`语句*，如果行驶里程大于 200 英里，我们将添加到新列表中:*

```py
short_range_car_list = []               # creating our empty short-range car list 

for row in ev_data[1:]:                 # iterate through ev_data as before 
    ev_range = row[1]          
    if ev_range > 200:                  #  if the car has a range of > 200
        continue                        # end the loop here; do not execute the code below, continue to the next row
    short_range_car_list.append(row)    # append the row to our short-range car list

print(short_range_car_list) 
```

```py
[['Hyundai Ioniq EV', 124, 30315]] 
```

这可能不是创建我们的短程汽车列表的最有效和可读的方式，但它确实演示了`continue`如何工作，所以让我们来看看这里到底发生了什么。

在第一次循环中，Python 正在查看特斯拉行。那辆车*和*确实有超过 200 英里的续航里程，所以 Python 看到 if 语句为真，并执行嵌套在 if 语句中的`continue`，这使得*立即*跳到下一行`ev_data`开始下一个循环。

在第二个循环中，Python 正在查看下一行，这是 Hyundai 行。那辆车的行驶里程不到 200 英里，因此 Python 发现条件 if 语句是*而不是* met，并执行 for 循环中的其余代码，将 Hyundai 行追加到`short_range_car_list`。

在第三个也是最后一个循环中，Python 正在查看 Chevy 行。那辆车的行驶里程超过 200 英里，这意味着条件 if 语句为真。因此，Python 再次执行嵌套的`continue`，这结束了循环，并且由于我们的数据集中没有更多的数据行，因此完全结束了 for 循环。

## 额外资源

希望此时，您已经熟悉了 Python 中的 for 循环，并且了解了它们对于常见的数据科学任务(如数据清理、数据准备和数据分析)的用处。

准备好迈出下一步了吗？以下是一些可供查看的附加资源:

*   [**高级 Python For Loops 教程**](https://www.dataquest.io/blog/python-for-loop-tutorial/)——在本教程的“续集”中，学习使用 For Loops 和 NumPy、Pandas 以及其他更高级的技术。
*   [Python 教程](https://www.dataquest.io/python-tutorials-for-data-science/) —我们不断扩充的数据科学 Python 教程列表。
*   [数据科学课程](https://www.dataquest.io/path/data-scientist/) —通过完全交互式的编程、数据科学和统计课程，让您的学习更上一层楼，就在您的浏览器中。
*   Python 关于 For 循环的官方文档–官方文档没有本教程那么深入，但它回顾了 For 循环的基础知识，解释了一些相关概念，如 While 循环。
*   [Dataquest 的数据科学 Python 基础课程](https://www.dataquest.io/course/python-for-data-science-fundamentals/)–我们的 Python 基础课程从头开始介绍数据科学的 Python 编码。它涵盖了列表、循环等等，你可以直接在浏览器中进行交互式编码。
*   [Dataquest 面向数据科学的中级 Python 课程](https://www.dataquest.io/course/python-for-data-science-intermediate/)——当你觉得自己已经掌握了 for 循环和其他核心 Python 概念时，这是另一个交互式课程，它将帮助你将 Python 技能提升到一个新的水平。
*   [免费数据集练习](https://www.dataquest.io/blog/free-datasets-for-projects/)–通过从这些来源中的一个获取免费数据集，并将您的新技能应用于大型真实数据集，自己练习循环。第一部分中的数据集(用于数据可视化)应该特别适合实践项目，因为它们应该已经相对干净了。

祝你好运，快乐循环！

## 这个教程有帮助吗？

选择你的道路，不断学习有价值的数据技能。

![arrow down left](img/2215dd1efd21629477b52ea871afdd98.png)![arrow right down](img/2e703f405f987a154317ac045ee00a68.png)[Python Tutorials](/python-tutorials-for-data-science/)

在我们的免费教程中练习 Python 编程技能。

[Data science courses](/data-science-courses/)

通过我们的交互式浏览器数据科学课程，投入到 Python、R、SQL 等语言的学习中。*