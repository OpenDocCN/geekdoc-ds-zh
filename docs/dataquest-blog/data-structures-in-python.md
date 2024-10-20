# Python 数据结构:列表、字典、集合、元组(2022)

> 原文：<https://www.dataquest.io/blog/data-structures-in-python/>

September 19, 2022![Data structures](img/fa8f96730b0a998385f69a6e9505c827.png)

阅读完本教程后，您将了解 Python 中存在哪些数据结构，何时应用它们，以及它们的优缺点。我们将从总体上讨论数据结构，然后深入 Python 数据结构:列表、字典、集合和元组。

## 什么是数据结构？

数据结构是在计算机内存中组织数据的一种方式，用编程语言实现。这种组织是高效存储、检索和修改数据所必需的。这是一个基本概念，因为数据结构是任何现代软件的主要构件之一。学习存在什么样的数据结构以及如何在不同的情况下有效地使用它们是学习任何编程语言的第一步。

## Python 中的数据结构

Python 中的内置数据结构可以分为两大类:**可变**和**不可变**。可变的(来自拉丁语 *mutabilis* ，“可变的”)数据结构是我们可以修改的——例如，通过添加、删除或改变它们的元素。Python 有三种可变的数据结构:**列表**、**字典**和**集合**。另一方面，不可变的数据结构是那些我们在创建后不能修改的数据结构。Python 中唯一基本的内置不可变数据结构是一个**元组**。

Python 还有一些高级的数据结构，比如[栈](https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-stacks)或者[队列](https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-queues)，可以用基本的数据结构实现。但是，这些在数据科学中很少使用，在软件工程和复杂算法的实现领域更常见，所以在本教程中我们不讨论它们。

不同的 Python 第三方包实现了自己的数据结构，比如`pandas`中的[数据帧](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)和[系列](https://pandas.pydata.org/docs/reference/api/pandas.Series.html?highlight=series#pandas.Series)或者`NumPy`中的[数组](https://numpy.org/doc/stable/reference/generated/numpy.array.html)。然而，我们也不会在这里谈论它们，因为这些是更具体的教程的主题(例如[如何创建和使用 Pandas DataFrame](https://www.dataquest.io/blog/tutorial-how-to-create-and-use-a-pandas-dataframe/) 或 [NumPy 教程:用 Python 进行数据分析](https://www.dataquest.io/blog/numpy-tutorial-python/))。

让我们从可变数据结构开始:列表、字典和集合。

### 列表

Python 中的列表被实现为**动态可变数组**，它保存了一个**有序的**条目集合。

首先，在许多编程语言中，数组是包含相同数据类型的元素集合的数据结构(例如，所有元素都是整数)。然而，在 Python 中，列表可以包含不同的数据类型和对象。例如，整数、字符串甚至函数都可以存储在同一个列表中。列表的不同元素可以通过整数索引来访问，其中列表的第一个元素的索引为 0。这个属性来源于这样一个事实，即在 Python 中，列表是有序的，这意味着它们保留了将元素插入列表的顺序。

接下来，我们可以任意添加、删除和更改列表中的元素。例如，`.append()`方法向列表中添加新元素，而`.remove()`方法从列表中删除元素。此外，通过索引访问列表的元素，我们可以将它更改为另一个元素。有关不同列表方法的更多详细信息，请参考[文档](https://docs.python.org/dev/tutorial/datastructures.html#more-on-lists)。

最后，当创建一个列表时，我们不必预先指定它将包含的元素数量；所以可以随心所欲的扩展，让它充满活力。

当我们想要存储不同数据类型的集合，并随后添加、移除或对列表的每个元素执行操作(通过循环遍历它们)时，列表是有用的。此外，通过创建例如字典列表、元组列表或列表，列表对于存储其他数据结构(甚至其他列表)是有用的。将表存储为列表的列表(其中每个内部列表代表一个表的列)以供后续数据分析是非常常见的。

因此，列表的**优点是:**

*   它们代表了存储相关对象集合的最简单方法。
*   它们很容易通过删除、添加和更改元素来修改。
*   它们对于创建嵌套数据结构很有用，比如列表/字典列表。

然而，它们也有缺点:

*   在对元素执行算术运算时，它们可能会非常慢。(为了提高速度，请使用 NumPy 的数组。)
*   由于其隐蔽的实现，它们使用更多的磁盘空间。

#### 例子

最后，我们来看几个例子。

我们可以使用方括号(`[]`)创建一个列表，用逗号分隔零个或多个元素，或者使用[构造函数`list()`](https://docs.python.org/3/library/stdtypes.html#list) 。后者也可以用于将某些其他数据结构转换成列表。

```py
# Create an empty list using square brackets
l1 = []

# Create a four-element list using square brackets
l2 = [1, 2, "3", 4]  # Note that this lists contains two different data types: integers and strings

# Create an empty list using the list() constructor
l3 = list()

# Create a three-element list from a tuple using the list() constructor
# We will talk about tuples later in the tutorial
l4 = list((1, 2, 3))

# Print out lists
print(f"List l1: {l1}")
print(f"List l2: {l2}")
print(f"List l3: {l3}")
print(f"List l4: {l4}")
```

```py
List l1: []
List l2: [1, 2, '3', 4]
List l3: []
List l4: [1, 2, 3]
```

我们可以使用索引来访问列表的元素，其中列表的第一个元素的索引为 0:

```py
# Print out the first element of list l2
print(f"The first element of the list l2 is {l2[0]}.")
print()

# Print out the third element of list l4
print(f"The third element of the list l4 is {l4[2]}.")
```

```py
 The first element of the list l2 is 1.

    The third element of the list l4 is 3.
```

我们还可以**分割**列表并同时访问多个元素:

```py
# Assign the third and the fourth elements of l2 to a new list
l5 = l2[2:]

# Print out the resulting list
print(l5)
```

```py
 ['3', 4]
```

注意，如果我们想要从索引 2(包括)到列表末尾的所有元素，我们不必指定我们想要访问的最后一个元素的索引。一般来说，列表分片的工作方式如下:

1.  左方括号。
2.  写入我们要访问的第一个元素的第一个索引。该元素将包含在输出中。将冒号放在该索引后面。
3.  写索引，加上我们想要访问的最后一个元素。这里需要加 1，因为我们写的索引**下的元素不会包含在输出**中。

让我们用一个例子来展示这种行为:

```py
print(f"List l2: {l2}")

# Access the second and the third elements of list l2 (these are the indices 1 and 2)
print(f"Second and third elements of list l2: {l2[1:3]}")
```

```py
 List l2: [1, 2, '3', 4]
    Second and third elements of list l2: [2, '3']
```

注意，我们指定的最后一个索引是 3，而不是 2，尽管我们想访问索引 2 下的元素。因此，我们写的最后一个索引不包括在内。

您可以尝试不同的索引和更大的列表，以了解索引是如何工作的。

现在让我们证明列表是可变的。例如，我们可以将一个新元素添加到列表中，或者从列表中添加一个特定的元素:

```py
# Append a new element to the list l1
l1.append(5)

# Print the modified list
print("Appended 5 to the list l1:")
print(l1)

# Remove element 5 from the list l1
l1.remove(5)

# Print the modified list
print("Removed element 5 from the list l1:")
print(l1)
```

```py
 Appended 5 to list l1:
    [5]
    Removed element 5 from the list l1:
    []
```

此外，我们可以通过访问所需的索引并为该索引分配一个新值来修改列表中已经存在的元素:

```py
# Print the original list l2
print("Original l2:")
print(l2)

# Change value at index 2 (third element) in l2
l2[2] = 5

# Print the modified list l2
print("Modified l2:")
print(l2)
```

```py
 Original l2:
    [1, 2, '3', 4]
    Modified l2:
    [1, 2, 5, 4]
```

当然，我们只是触及了 Python 列表的皮毛。你可以从[这门课](https://app.dataquest.io/c/114/m/608/python-lists/1/storing-row-elements-into-variables)中学到更多，或者看看[的 Python 文档](https://docs.python.org/3/tutorial/datastructures.html#more-on-lists)。

### 字典

Python 中的字典与现实世界中的字典非常相似。这些是**可变的**数据结构，包含一组**键**和与之相关的**值**。这种结构使它们非常类似于单词定义词典。例如，单词 *dictionary* (我们的关键字)与其在[牛津在线词典](https://www.oxfordlearnersdictionaries.com/definition/english/dictionary?q=dictionary)中的定义(值)相关联:*按字母顺序给出一种语言的单词列表并解释其含义的书或电子资源，或者给出一种外语的单词*。

字典用于快速访问与**唯一**键相关的某些数据。唯一性是必不可少的，因为我们只需要访问特定的信息，不要与其他条目混淆。想象一下，我们想要阅读*数据科学*的定义，但是一本字典将我们重定向到两个不同的页面:哪一个是正确的？请注意，从技术上讲，我们可以创建一个具有两个或更多相同键的字典，尽管由于字典的性质，这是不可取的。

```py
# Create dictionary with duplicate keys
d1 = {"1": 1, "1": 2}
print(d1)

# It will only print one key, although no error was thrown
# If we  try to access this key, then it'll return 2, so the value of the second key
print(d1["1"])

# It is technically possible to create a dictionary, although this dictionary will not support them,
# and will contain only one of the key
```

当我们能够关联(用技术术语来说，**映射**)特定数据的唯一键，并且我们希望非常快速地**访问该数据**(在恒定时间内，不管字典大小如何)时，我们就使用字典。此外，字典值可能相当复杂。例如，我们的关键字可以是客户的名字，他们的个人数据(值)可以是带有关键字的字典，如“年龄”、“家乡”等。

因此，字典的**优点是:**

*   如果我们需要生成`key:value`对，它们使得代码更容易阅读。我们也可以对列表的列表做同样的事情(其中内部列表是成对的“键”和“值”)，但是这看起来更加复杂和混乱。
*   我们可以很快地在字典中查找某个值。相反，对于一个列表，我们必须在命中所需元素之前读取列表。如果我们增加元素的数量，这种差异会急剧增加。

然而，**他们的缺点**是:

*   它们占据了很大的空间。如果我们需要处理大量的数据，这不是最合适的数据结构。
*   在 Python 3.6.0 和更高版本中，字典[记住元素插入的顺序](https://docs.python.org/3/whatsnew/3.6.html#pep-520-preserving-class-attribute-definition-order)。请记住这一点，以避免在不同版本的 Python 中使用相同代码时出现兼容性问题。

#### 例子

现在让我们来看几个例子。首先，我们可以用花括号(`{}`)或`dict()`构造函数创建一个字典:

```py
# Create an empty dictionary using curly brackets
d1 = {}

# Create a two-element dictionary using curly brackets
d2 = {"John": {"Age": 27, "Hometown": "Boston"}, "Rebecca": {"Age": 31, "Hometown": "Chicago"}}
# Note that the above dictionary has a more complex structure as its values are dictionaries themselves!

# Create an empty dictionary using the dict() constructor
d3 = dict()

# Create a two-element dictionary using the dict() constructor
d4 = dict([["one", 1], ["two", 2]])  # Note that we created the dictionary from a list of lists

# Print out dictionaries
print(f"Dictionary d1: {d1}")
print(f"Dictionary d2: {d2}")
print(f"Dictionary d3: {d3}")
print(f"Dictionary d4: {d4}")
```

```py
 Dictionary d1: {}
    Dictionary d2: {'John': {'Age': 27, 'Hometown': 'Boston'}, 'Rebecca': {'Age': 31, 'Hometown': 'Chicago'}}
    Dictionary d3: {}
    Dictionary d4: {'one': 1, 'two': 2}
```

现在让我们访问字典中的一个元素。我们可以用与列表相同的方法做到这一点:

```py
# Access the value associated with the key 'John'
print("John's personal data is:")
print(d2["John"])
```

```py
 John's personal data is:
    {'Age': 27, 'Hometown': 'Boston'}
```

接下来，我们还可以修改字典—例如，通过添加新的`key:value`对:

```py
# Add another name to the dictionary d2
d2["Violet"] = {"Age": 34, "Hometown": "Los Angeles"}

# Print out the modified dictionary
print(d2)
```

```py
 {'John': {'Age': 27, 'Hometown': 'Boston'}, 'Rebecca': {'Age': 31, 'Hometown': 'Chicago'}, 'Violet': {'Age': 34, 'Hometown': 'Los Angeles'}}
```

我们可以看到，一个新的关键，“紫罗兰”，已被添加。

从字典中删除元素也是可能的，所以通过阅读[文档](https://docs.python.org/3/library/stdtypes.html?highlight=update#mapping-types-dict)来寻找这样做的方法。此外，你可以阅读关于 Python 字典的更深入的教程(有大量的例子)或者看看 [DataQuest 的字典课程](https://app.dataquest.io/c/126/m/643/python-dictionaries/1/storing-data)。

### 设置

Python 中的集合可以定义为不可变的唯一元素的可变动态集合。集合中包含的元素必须是不可变的。集合可能看起来非常类似于列表，但实际上，它们是非常不同的。

首先，它们可能**只包含唯一的元素**，所以不允许重复。因此，集合可以用来从列表中删除重复项。接下来，就像数学中的[集合](https://en.wikipedia.org/wiki/Set_(mathematics))一样，它们有独特的运算可以应用于它们，比如集合并、交集等。最后，它们在检查特定元素是否包含在集合中时非常有效。

因此，器械包的优点是:

*   我们可以对它们执行独特的(但相似的)操作。
*   如果我们想检查某个元素是否包含在一个集合中，它们比列表要快得多。

但是他们的缺点是:

*   集合本质上是无序的。如果我们关心保持插入顺序，它们不是我们的最佳选择。
*   我们不能像处理列表那样通过索引来改变集合元素。

#### 例子

为了创建一个集合，我们可以使用花括号(`{}`)或者`set()`构造函数。不要把集合和字典混淆(字典也使用花括号)，因为集合不包含`key:value`对。但是请注意，与字典键一样，只有不可变的数据结构或类型才允许作为集合元素。这一次，让我们直接创建填充集:

```py
# Create a set using curly brackets
s1 = {1, 2, 3}

# Create a set using the set() constructor
s2 = set([1, 2, 3, 4])

# Print out sets
print(f"Set s1: {s1}")
print(f"Set s2: {s2}")
```

```py
 Set s1: {1, 2, 3}
    Set s2: {1, 2, 3, 4}
```

在第二个例子中，我们使用了一个 **iterable** (比如一个列表)来创建一个集合。然而，如果我们使用列表作为集合元素，Python 会抛出一个错误。你认为为什么会这样？**提示**:阅读集合的定义。

为了练习，您可以尝试使用其他数据结构来创建一个集合。

与它们的数学对应物一样，我们可以在集合上执行某些操作。例如，我们可以创建集合的**联合**，这基本上意味着将两个集合合并在一起。但是，如果两个集合有两个或更多相同的值，则得到的集合将只包含其中一个值。创建并集有两种方法:要么用`union()`方法，要么用竖线(`|`)操作符。我们来举个例子:

```py
# Create two new sets
names1 = set(["Glory", "Tony", "Joel", "Dennis"])
names2 = set(["Morgan", "Joel", "Tony", "Emmanuel", "Diego"])

# Create a union of two sets using the union() method
names_union = names1.union(names2)

# Create a union of two sets using the | operator
names_union = names1 | names2

# Print out the resulting union
print(names_union)
```

```py
 {'Glory', 'Dennis', 'Diego', 'Joel', 'Emmanuel', 'Tony', 'Morgan'}
```

在上面的并集中，我们可以看到`Tony`和`Joel`只出现了一次，尽管我们合并了两个集合。

接下来，我们可能还想找出哪些名字同时出现在两个集合中。这可以通过`intersection()`方法或与(`&`)运算符来完成。

```py
# Intersection of two sets using the intersection() method
names_intersection = names1.intersection(names2)

# Intersection of two sets using the & operator
names_intersection = names1 & names2

# Print out the resulting intersection
print(names_intersection)
```

```py
 {'Joel', 'Tony'}
```

`Joel`和`Tony`出现在两组中；因此，它们由集合交集返回。

集合运算的最后一个例子是两个集合之间的差。换句话说，该操作将返回第一个集合中的所有元素，而不是第二个集合中的所有元素。我们可以使用`difference()`方法或减号(`-`):

```py
# Create a set of all the names present in names1 but absent in names2 with the difference() method
names_difference = names1.difference(names2)

# Create a set of all the names present in names1 but absent in names2 with the - operator
names_difference = names1 - names2

# Print out the resulting difference
print(names_difference)
```

```py
 {'Dennis', 'Glory'}
```

如果你交换集合的位置会发生什么？试着在尝试之前预测结果。

还有其他操作可以在集合中使用。更多信息，请参考[本教程](https://www.dataquest.io/blog/tutorial-everything-you-need-to-know-about-python-sets/)，或 [Python 文档](https://docs.python.org/3/library/stdtypes.html#set)。

最后，作为奖励，让我们比较一下使用集合与使用列表相比，在检查集合中元素的存在时有多快。

```py
import time

def find_element(iterable):
    """Find an element in range 0-4999 (included) in an iterable and pass."""
    for i in range(5000):
        if i in iterable:
            pass

# Create a list and a set
s = set(range(10000000))

l = list(range(10000000))

start_time = time.time()
find_element(s) # Find elements in a set
print(f"Finding an element in a set took {time.time() - start_time} seconds.")

start_time = time.time()
find_element(l) # Find elements in a list
print(f"Finding an element in a list took {time.time() - start_time} seconds.")
```

```py
 Finding an element in a set took 0.00016832351684570312 seconds.
    Finding an element in a list took 0.04723954200744629 seconds.
```

显然，使用集合比使用列表要快得多。对于较大的集合和列表，这种差异会增大。

### 元组

元组几乎与列表相同，因此它们包含元素的有序集合，除了一个属性:它们是**不可变的**。如果我们需要一个一旦创建就不能再修改的数据结构，我们会使用元组。此外，如果所有元素都是不可变的，元组可以用作字典键。

除此之外，元组具有与列表相同的属性。为了创建一个元组，我们可以使用圆括号(`()`)或者`tuple()`构造函数。我们可以很容易地将列表转换成元组，反之亦然(回想一下，我们从元组创建了列表`l4`)。

元组的**优点是:**

*   它们是不可变的，所以一旦被创建，我们可以确定我们不会错误地改变它们的内容。
*   如果它们的所有元素都是不可变的，那么它们可以用作字典键。

元组的**缺点是:**

*   当我们必须处理可修改的对象时，我们不能使用它们；我们不得不求助于列表。
*   无法复制元组。
*   它们比列表占用更多的内存。

#### 例子

让我们来看一些例子:

```py
# Create a tuple using round brackets
t1 = (1, 2, 3, 4)

# Create a tuple from a list the tuple() constructor
t2 = tuple([1, 2, 3, 4, 5])

# Create a tuple using the tuple() constructor
t3 = tuple([1, 2, 3, 4, 5, 6])

# Print out tuples
print(f"Tuple t1: {t1}")
print(f"Tuple t2: {t2}")
print(f"Tuple t3: {t3}")
```

```py
 Tuple t1: (1, 2, 3, 4)
    Tuple t2: (1, 2, 3, 4, 5)
    Tuple t3: (1, 2, 3, 4, 5, 6)
```

是否有可能从其他数据结构(即集合或字典)创建元组？试着练习一下。

元组是不可变的；因此，一旦它们被创建，我们就不能改变它们的元素。让我们看看如果我们尝试这样做会发生什么:

```py
# Try to change the value at index 0 in tuple t1
t1[0] = 1
```

```py
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'tuple' object does not support item assignment
```

是一个`TypeError`！元组不支持项赋值，因为它们是不可变的。为了解决这个问题，我们可以将这个元组转换成一个列表。

然而，我们可以通过索引来访问元组中的元素，就像在列表中一样:

```py
# Print out the value at index 1 in the tuple t2
print(f"The value at index 1 in t2 is {t2[1]}.")
```

```py
 The value at index 1 in t2 is 2.
```

元组也可以用作字典键。例如，我们可以将某些元素及其连续索引存储在一个元组中，并为它们赋值:

```py
# Use tuples as dictionary keys
working_hours = {("Rebecca", 1): 38, ("Thomas", 2): 40}
```

如果使用元组作为字典键，那么元组必须包含不可变的对象:

```py
# Use tuples containing mutable objects as dictionary keys
working_hours = {(["Rebecca", 1]): 38, (["Thomas", 2]): 40}
```

```py
---------------------------------------------------------------------------

TypeError                                 Traceback (most recent call last)

Input In [20], in <cell line:="">()
      1 # Use tuples containing mutable objects as dictionary keys
----> 2 working_hours = {(["Rebecca", 1]): 38, (["Thomas", 2]): 40}

TypeError: unhashable type: 'list'
```

如果我们的元组/键包含可变对象(在这种情况下是列表),我们得到一个`TypeError`。

## 结论

让我们总结一下从本教程中学到的内容:

*   数据结构是编程中的一个基本概念，是轻松存储和检索数据所必需的。
*   Python 有四种主要的数据结构，分为可变(列表、字典和集合)和不可变(元组)类型。
*   列表对于保存相关对象的异构集合非常有用。
*   每当我们需要将一个键链接到一个值并通过一个键快速访问一些数据时，我们就需要字典，就像在现实世界的字典中一样。
*   集合允许我们对它们进行运算，如求交或求差；因此，它们对于比较两组数据很有用。
*   元组类似于列表，但不可变；它们可以用作数据容器，我们不希望被错误地修改。