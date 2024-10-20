# 教程:关于 Python 集合你需要知道的一切

> 原文：<https://www.dataquest.io/blog/tutorial-everything-you-need-to-know-about-python-sets/>

January 21, 2022![Python Set Tutorial](img/1fa210e32870f579d5a0670e77c690fe.png)

在本教程中，我们将详细探讨 Python 集合:什么是 Python 集合，何时以及为何使用它，如何创建它，如何修改它，以及我们可以对 Python 集合执行哪些操作。

### Python 中的集合是什么？

集合是一种内置的 Python 数据结构，用于在单个变量中存储唯一项的集合，可能是混合数据类型。Python 集包括:

*   **无序**–器械包中的物品没有任何已定义的顺序
*   **未编入索引的**–我们无法像访问列表一样访问带有`[i]`的项目
*   **可变**–集合可以被修改为整数或元组
*   可迭代的(Iterable)——我们可以对集合中的项目进行循环

请注意，虽然 Python 集合本身是可变的(我们可以从中删除项目或添加新项目)，但它的项目必须是不可变的数据类型，如整数、浮点、元组或字符串。

Python 集合的主要应用包括:

*   删除重复项
*   正在检查集合成员资格
*   执行数学集合运算，如并、交、差和对称差

### 创建 Python 集

我们可以通过两种方式创建 Python 集合:

1.  通过使用内置的`set()`函数和传入的 iterable 对象(比如列表、元组或字符串)
2.  通过将逗号分隔的所有项目放在一对花括号内`{}`

在这两种情况下，重要的是要记住 Python 集合的未来项(即，可迭代对象的单个元素或放在花括号内的项)本身可以是可迭代的(例如，元组)，但它们不能是可变类型，例如列表、字典或另一个集合。

让我们看看它是如何工作的:

```py
# First way: using the set() function on an iterable object
set1 = set([1, 1, 1, 2, 2, 3])          # from a list
set2 = set(('a', 'a', 'b', 'b', 'c'))   # from a tuple
set3 = set('anaconda')                  # from a string

# Second way: using curly braces
set4 = {1, 1, 'anaconda', 'anaconda', 8.6, (1, 2, 3), None}

print('Set1:', set1)
print('Set2:', set2)
print('Set3:', set3)
print('Set4:', set4)

# Incorrect way: trying to create a set with mutable items (a list and a set)
set5 = {1, 1, 'anaconda', 'anaconda', 8.6, [1, 2, 3], {1, 2, 3}}
print('Set5:', set5)
```

```py
Set1: {1, 2, 3}
Set2: {'b', 'c', 'a'}
Set3: {'n', 'o', 'd', 'c', 'a'}
Set4: {1, 8.6, 'anaconda', (1, 2, 3), None}

---------------------------------------------------------------------------

TypeError                                 Traceback (most recent call last)

~\AppData\Local\Temp/ipykernel_6832/1057027593.py in <module>
     13 
     14 # Incorrect way: trying to create a set with mutable items (a list and a set)
---> 15 set5 = {1, 1, 'anaconda', 'anaconda', 8.6, [1, 2, 3], {1, 2, 3}}
     16 print('Set5:', set5)

TypeError: unhashable type: 'list'
```

我们可以在这里作出以下观察:

*   在每种情况下，重复序列都已被删除
*   项目的初始顺序已更改
*   `set4`包含不同数据类型的元素
*   试图用可变项目(一个列表和一个集合)创建一个 Python 集合导致了一个`TypeError`

当我们需要创建一个空的 Python 集合时，会出现一种特殊的情况。因为空花括号`{}`创建了一个空的 Python 字典，所以我们不能用这种方法在 Python 中创建一个空集。在这种情况下，使用`set()`功能仍然有效:

```py
empty1 = {}
empty2 = set()

print(type(empty1))
print(type(empty2))
```

```py
 <class><class>
```

### 正在检查集合成员资格

为了检查某个项目在 Python 集合中是否存在，我们使用操作符关键字`in`或关键字组合`not in`:

```py
myset = {1, 2, 3}
print(1 in myset)
print(1 not in myset)
```

```py
True
False
```

### 访问 Python 集中的值

由于 Python 集合是无序和无索引的，我们不能通过索引或切片来访问它的项目。一种方法是遍历集合:

```py
myset = {'a', 'b', 'c', 'd'}

for item in myset:
    print(item)
```

```py
b
d
a
c
```

输出值的顺序可能与原始集合中显示的顺序不同。

### 修改 Python 集

#### 向 Python 集添加项目

我们可以使用`add()`方法向 Python 集合添加单个不可变项，或者使用`update()`方法添加几个不可变项。后者以元组、列表、字符串或其他*不可变项目*的集合作为其参数，然后将它们中的每个单个*唯一项目*(或在字符串的情况下，每个单个*唯一字符*)添加到集合中:

```py
# Initial set
myset = set()

# Adding a single immutable item
myset.add('a')
print(myset)

# Adding several items
myset.update({'b', 'c'})        # a set of immutable items
print(myset)
myset.update(['d', 'd', 'd'])   # a list of immutable items
print(myset)
myset.update(['e'], ['f'])      # several lists of immutable items
print(myset)
myset.update('fgh')             # a string
print(myset)
myset.update([[1, 2], [3, 4]])  # an attempt to add a list of mutable items (lists)
print(myset)
```

```py
{'a'}
{'b', 'c', 'a'}
{'b', 'd', 'c', 'a'}
{'e', 'f', 'b', 'd', 'c', 'a'}
{'e', 'f', 'b', 'h', 'd', 'c', 'a', 'g'}

---------------------------------------------------------------------------

TypeError                                 Traceback (most recent call last)

~\AppData\Local\Temp/ipykernel_6832/2286239840.py in <module>
     15 myset.update('fgh')             # a string
     16 print(myset)
---> 17 myset.update([[1, 2], [3, 4]])  # an attempt to add a list of mutable items (lists)
     18 print(myset)

TypeError: unhashable type: 'list'
```

#### 从 Python 集中移除项目

要从 Python 集合中移除一个或多个项目，我们可以选择以下四种方法之一:

1.  `discard()`–删除特定的项目，如果该项目不在集合中，则不做任何操作
2.  `remove()`–删除特定的项目，如果该项目不在集合中，则引发`KeyError`
3.  `pop()`–移除并返回一个随机项目，或者如果集合为空，则引发`KeyError`
4.  `clear()`–清除集合(删除所有项目)

让我们看一些例子:

```py
# Initial set
myset = {1, 2, 3, 4}
print(myset)

# Removing a particular item using the discard() method
myset.discard(1)  # the item was present in the set
print(myset)
myset.discard(5)  # the item was absent in the set
print(myset)

# Removing a particular item using the remove() method
myset.remove(4)   # the item was present in the set
print(myset)
myset.remove(5)   # the item was absent in the set
print(myset)
```

```py
{1, 2, 3, 4}
{2, 3, 4}
{2, 3, 4}
{2, 3}

---------------------------------------------------------------------------

KeyError                                  Traceback (most recent call last)

~\AppData\Local\Temp/ipykernel_6832/2462592674.py in <module>
     12 myset.remove(4)   # the item was present in the set
     13 print(myset)
---> 14 myset.remove(5)   # the item was absent in the set
     15 print(myset)

KeyError: 5
```

```py
# Taking the set from the code above
myset = {2, 3}

# Removing and returning a random item
print(myset.pop())  # the removed and returned item
print(myset)        # the updated set

# Removing all the items
myset.clear()
print(myset)

# An attempt to remove and return a random item from an empty set
myset.pop()
print(myset)
```

```py
2
{3}
set()

---------------------------------------------------------------------------

KeyError                                  Traceback (most recent call last)

~\AppData\Local\Temp/ipykernel_6832/901310087.py in <module>
     11 
     12 # An attempt to remove and return a random item from an empty set
---> 13 myset.pop()
     14 print(myset)

KeyError: 'pop from an empty set'
```

### 用于集合的内置 Python 函数

一些适用于列表和其他类似集合的数据结构的内置 Python 函数对于不同用途的 Python 集合也很有用。让我们考虑一下最有用的功能:

*   `len()`–返回集合大小(集合中的项目数)
*   `min()` and `max()`–return the small/large item in the set and are most used for the sets with numerical values

    > **Note:** In the case of tuples or strings as set items, this becomes a bit complicated. For strings, the comparison follows the principle of lexicography (that is, the ASCII values of characters of two or more strings are compared from left to right). Instead, tuples are compared according to items with the same index, also from left to right. Using `min()` and `max()` functions on Python collections with mixed data type items will raise `TypeError`.

*   `sum()`–返回仅包含数值的集合中所有项目的总和

```py
# A set with numeric items
myset = {5, 10, 15}
print('Set:', myset)
print('Size:', len(myset))
print('Min:', min(myset))
print('Max:', max(myset))
print('Sum:', sum(myset))
print('\n')

# A set with string items
myset = {'a', 'A', 'b', 'Bb'}
print('Set:', myset)
print('Min:', min(myset))
print('Max:', max(myset))
print('\n')

# A set with tuple items
myset = {(1, 2), (1, 0), (2, 3)}
print('Set:', myset)
print('Min:', min(myset))
print('Max:', max(myset))
```

```py
Set: {10, 5, 15}
Size: 3
Min: 5
Max: 15
Sum: 30

Set: {'A', 'b', 'a', 'Bb'}
Min: A
Max: b

Set: {(1, 0), (1, 2), (2, 3)}
Min: (1, 0)
Max: (2, 3)
```

注意在集合`{'b', 'a', 'A', 'Bb'}`中，最小值是`A`，而不是`a`。发生这种情况是因为，从字典上看，所有的大写字母都比所有的小写字母低。

*   `all()`–如果集合中的所有项目都评估为`True`，或者集合为空，则返回`True`
*   `any()`–如果集合中至少有一个项目评估为`True`，则返回`True`(对于空集，返回`False` )

    > **注:**评估为`True`的值是不评估为`False`的值。在计算机编程语言集合项的上下文中,评估为`False`的值是`0``0.0``''``False``None`和`()`.
    > 
    > 

```py
print(all({1, 2}))
print(all({1, False}))
print(any({1, False}))
print(any({False, False}))
```

```py
True
False
True
False
```

*   `sorted()`–返回集合中项目的排序列表

```py
myset = {4, 2, 5, 1, 3}
print(sorted(myset))

myset = {'c', 'b', 'e', 'a', 'd'}
print(sorted(myset))
```

```py
[1, 2, 3, 4, 5]
['a', 'b', 'c', 'd', 'e']
```

### 对 Python 集合执行数学集合运算

这组运算包括并、交、差和对称差。它们中的每一个都可以由操作者或方法来执行。

让我们在以下两个 Python 集合上练习数学集合运算:

```py
a = {1, 2, 3, 4, 5}
b = {4, 5, 6, 7}
```

#### 集合联合

两个(或更多)Python 集合的并集返回两个(所有)集合中所有唯一项目的新集合。可以使用`|`操作符或`union()`方法执行:

```py
print(a | b)
print(b | a)
print(a.union(b))
print(b.union(a))
```

```py
{1, 2, 3, 4, 5, 6, 7}
{1, 2, 3, 4, 5, 6, 7}
{1, 2, 3, 4, 5, 6, 7}
{1, 2, 3, 4, 5, 6, 7}
```

正如我们所看到的，对于 union 操作，集合的顺序并不重要:我们可以用相同的结果编写`a | b`或`b | a`，对于方法的使用也是如此。

两个以上 Python 集合上 union 操作的语法如下:`a | b | c`或`a.union(b, c)`。

请注意，在上面的示例(以及所有后续的示例)中，在操作符前后添加空格只是为了提高可读性。

#### 设置交集

两个(或更多)Python 集合的交集返回两个(所有)集合共有的项目的新集合。可以使用`&`操作符或`intersection()`方法执行:

```py
print(a & b)
print(b & a)
print(a.intersection(b))
print(b.intersection(a))
```

```py
{4, 5}
{4, 5}
{4, 5}
{4, 5}
```

同样，在这种情况下，集合的顺序无关紧要:`a & b`或`b & a`将产生相同的结果，使用方法时也是如此。

两个以上 Python 集合的交集操作的语法如下:`a & b & c`或`a.intersection(b, c)`。

#### 集合差异

两个(或更多)Python 集合的差返回一个新的集合，其中包含第一个(左)集合中第二个(右)集合中不存在的所有项目。在两组以上的情况下，操作从左向右进行。对于这个集合操作，我们可以使用`-`操作符或`difference()`方法:

```py
print(a - b)
print(b - a)
print(a.difference(b))
print(b.difference(a))
```

```py
{1, 2, 3}
{6, 7}
{1, 2, 3}
{6, 7}
```

这里集合的顺序很重要:`a - b`(或`a.difference(b)`)返回所有在`a`但不在`b`的项目，而`b - a`(或`b.difference(a)`)返回所有在`b`但不在`a`的项目。

对两个以上的 Python 集合进行差运算的语法如下:`a - b - c`或`a.difference(b, c)`。在这种情况下，我们首先计算`a - b`，然后找出得到的集合和右边下一个集合的差，也就是`c`，以此类推。

#### 设置对称差

两个 Python 集合的对称差返回第一个或第二个集合中存在的一组新项目，但不是两个都存在。换句话说，两个集合的对称差是集合并和集合交之间的差，这对于多个集合的对称差也有意义。我们可以使用`^`操作符或`symmetric_difference()`方法来执行这个操作:

```py
print(a ^ b)
print(b ^ a)
print(a.symmetric_difference(b))
print(b.symmetric_difference(a))
```

```py
{1, 2, 3, 6, 7}
{1, 2, 3, 6, 7}
{1, 2, 3, 6, 7}
{1, 2, 3, 6, 7}
```

对于对称差分运算，集合的顺序无关紧要:`a ^ b`或`b ^ a`将产生相同的结果，我们可以说使用该方法也是如此。

对两个以上 Python 集合进行对称差分运算的语法如下:`a ^ b ^ c`。然而，这一次，我们不能使用`symmetric_difference()`方法，因为它只接受一个参数，否则会引发`TypeError`:

```py
a = {1, 2, 3, 4, 5}
b = {4, 5, 6, 7}
c = {7, 8, 9}

a.symmetric_difference(b, c)
```

```py
---------------------------------------------------------------------------

TypeError                                 Traceback (most recent call last)

~\AppData\Local\Temp/ipykernel_6832/4105073859.py in <module>
      3 c = {7, 8, 9}
      4 
----> 5 a.symmetric_difference(b, c)

TypeError: set.symmetric_difference() takes exactly one argument (2 given)
```

### Python 集合上的其他集合操作

还有一些其他有用的方法和运算符可用于处理两个或多个 Python 集:

*   `intersection_update()`(或`&=`运算符)—用当前集合与另一个(或多个)集合的交集重写当前集合
*   `difference_update()`(或`-=`运算符)—用当前集合与另一个(或多个)集合的差异重写当前集合
*   `symmetric_difference_update()`(或`^=`运算符)—用当前集合与另一个(或多个)集合的对称差重写当前集合
*   `isdisjoint()`(没有对应的运算符)—如果两个集合没有任何公共项，则返回`True`，这意味着这些集合的交集是一个空集
*   `issubset()`(或`<=`运算符)—如果另一个集合包含当前集合的每个项目，则返回`True`，包括两个集合相同的情况。如果要排除后一种情况，就不能用这种方法；相反，我们需要使用`<`(严格来说更小)操作符
*   `issuperset()`(或`>=`运算符)—如果当前集合包含另一个集合的每个项目，包括两个集合相同的情况，则返回`True`。如果要排除后一种情况，就不能用这种方法；相反，我们需要使用`>`(严格更大)操作符

### 结论

最后，让我们回顾一下我们在本教程中学到的 Python 集合:

*   Python 集的主要特征
*   Python 集合的主要应用
*   创建 Python 集的两种方法
*   如何创建空 Python 集
*   如何检查某个项目在 Python 集合中是否存在
*   如何访问 Python 集中的值
*   向 Python 集合添加新项的两种方法
*   从 Python 集合中移除项目的四种方法
*   哪些内置 Python 函数适用于 Python 集合
*   如何通过方法或运算符对两个或更多 Python 集合执行主要数学集合运算
*   我们还可以在两个或更多 Python 集合上执行哪些操作

现在，您应该熟悉在 Python 中创建和使用集合的所有细微差别。