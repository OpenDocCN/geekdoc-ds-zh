# 9.3 数组

> 原文：[`dcic-world.org/2025-08-27/arrays.html`](https://dcic-world.org/2025-08-27/arrays.html)

| |           9.3.1 两种有序项的内存布局 |
| --- | --- |
| |           9.3.2 部分迭代有序数据 |

我们在上一个章节结束时提出了一个问题，即如何快速访问列表中的特定元素。具体来说，如果你有一个名为 Runners 的列表（这是我们上次提到的例子），并且你写下：

```py
finishers[9]
```

定位第 10 名跑者的时间需要多久（记住，索引从 0 开始）？

这取决于列表在内存中的布局。

#### 9.3.1 两种有序项的内存布局 "链接至此")

当我们说“列表”时，通常意味着：一个有序项的集合。有序项的集合在内存中是如何排列的？以下有两个例子，使用课程名称列表：

```py
courses = ["CS111", "ENGN90", "VISA100"]
```

在第一种版本中，元素按连续的内存位置排列（这是我们到目前为止展示列表的方式）：

| Prog Directory           Memory |
| --- |
| -------------------------------------------------------------------- |
| courses --> loc 1001      loc 1001 --> [loc 1002, loc1003, loc 1004] |
| |                         loc 1002 --> "CS111" |
| |                         loc 1003 --> "ENGN90" |
| |                         loc 1004 --> "VISA100" |

在第二种版本中，每个元素都被捕获为一个包含元素和下一个列表位置的 dataype。当我们处于 Pyret 时，这个 dataype 被称为`link`。

| Prog Directory           Memory |
| --- |
| -------------------------------------------------------------------- |
| courses --> loc 1001      loc 1001 --> 链接("CS111", loc 1002) |
| |                         loc 1002 --> 链接("ENGN90", loc 1003) |
| |                         loc 1003 --> 链接("VISA100", loc 1004) |
| |                         loc 1004 --> empty |

两种版本之间的权衡是什么？在第一种版本中，我们可以以常数时间通过索引访问元素，就像在散列表中一样，但更改内容（添加或删除）需要移动内存中的数据。在第二种版本中，集合的大小可以任意增长或缩小，但查找特定值需要与索引成比例的时间。每种组织方式在某些程序中都有其合适的位置。

在数据结构术语中，第一种组织方式被称为数组。第二种被称为链表。Pyret 实现了链表，而数组是一个单独的数据类型（与列表的表示不同）。Python 将列表实现为数组。当你接触一种新的编程语言时，如果你关心底层操作的运行时性能，你需要查找它的列表是链表还是数组。

回到上一章中我们关于跑步者的讨论，我们可以简单地使用 Python 列表（数组）而不是散列表，并且能够访问完成特定位置的跑步者的名字。但让我们提出一个不同的问题。

我们将如何报告每个年龄组的顶尖完成者？特别是，我们想要编写一个如下所示的功能：

```py
def top_5_range(runners: list, lo: int, high: int) -> list:
    """get list of top 5 finishers with ages in
       the range given by lo to high, inclusive
    """
```

想想你会如何编写这段代码。

这里是我们的解决方案：

```py
def top_5_range(runners: list, lo: int, high: int) -> list:
    """get list of top 5 finishers with ages in
       the range given by lo to high, inclusive
    """

    # count of runners seen who are in age range
    in_range: int = 0
    # the list of finishers
    result: list = []

    for r in runners:
        if lo <= r.age and r.age <= high:
            in_range += 1
            result.append(r)
        if in_range == 5:
            return result
    print("Fewer than five in category")
    return result
```

这里，我们不是在到达列表的末尾时才返回，而是希望在列表中有五个跑步者时返回。因此，我们设置了一个额外的变量（`in_range`）来帮助我们跟踪计算的进度。一旦我们有了 5 个跑步者，我们就返回列表。如果我们从未达到 5 个跑步者，我们将向用户打印一条警告信息，然后返回我们已有的结果。

我们不能只查看列表的长度，而不是维护 `in_range` 变量吗？是的，我们可以这样做，尽管这个版本为我们的下一个例子提供了一个对比。

#### 9.3.2 通过有序数据部分迭代 "链接到此处")

如果我们只想打印出前五名完成者，而不是收集一个列表呢？虽然通常将计算和显示数据分开是更好的做法，但在实践中，我们有时会合并它们，或者执行其他操作（如将一些数据写入文件）而不会返回任何内容。我们如何修改代码来打印名字而不是构建跑步者的列表？

这里的挑战是如何停止计算。当我们构建列表时，我们使用返回来停止计算。但如果我们没有返回，或者需要在到达数据末尾之前停止循环，我们该怎么办？

我们使用一个名为 `break` 的命令，它的作用是终止循环并继续执行剩余的计算。在这里，`break` 代替了内部的返回语句：

```py
def print_top_5_range(runners: list, lo: int, high: int):
    """print top 5 finishers with ages in
       the range given by lo to high, inclusive
    """

    # count of runners seen who are in age range
    in_range: int = 0

    for r in runners:
        if lo <= r.age and r.age <= high:
            in_range += 1
            print(r.name)
        if in_range == 5:
            break
    print("End of results") 
```

如果 Python 遇到 `break` 语句，它将终止 for 循环并跳转到下一个语句，即函数末尾的打印语句。

#### 9.3.1 有序项的两种内存布局 "链接到此处")

当我们说“列表”时，我们通常是指：具有顺序的项的集合。有序项的集合在内存中是如何排列的？这里有两个例子，使用课程名称列表：

```py
courses = ["CS111", "ENGN90", "VISA100"]
```

在第一个版本中，元素被放置在连续的内存位置中（这是我们迄今为止展示列表的方式）：

| 程序目录           内存 |
| --- |
| -------------------------------------------------------------------- |
| courses --> loc 1001      loc 1001 --> [loc 1002, loc1003, loc 1004] |
|                           loc 1002 --> "CS111" |
|                           loc 1003 --> "ENGN90" |
|                           loc 1004 --> "VISA100" |

在第二个版本中，每个元素都被捕获为一个包含元素和下一个列表位置的类型。当我们使用 Pyret 时，这种类型被称为 `link`。

| 程序目录           内存 |
| --- |
| -------------------------------------------------------------------- |
| courses --> loc 1001     loc 1001 --> 链接("CS111", loc 1002) |
| |                         loc 1002 --> 链接("ENGN90", loc 1003) |
| |                         loc 1003 --> 链接("VISA100", loc 1004) |
| |                         loc 1004 --> 空值 |

两种版本之间的权衡是什么？在第一种中，我们可以以常数时间通过索引访问项目，就像散列表一样，但更改内容（添加或删除）需要移动内存中的内容。在第二种中，集合的大小可以任意增长或缩小，但查找特定值需要与索引成比例的时间。每个组织在其某些程序中都有其位置。

在数据结构术语中，第一种组织称为数组。第二种称为链表。Pyret 实现了链表，而数组是一个不同的数据类型（与列表有不同的表示法）。Python 实现列表为数组。当你接触一种新的编程语言时，如果你关心底层操作的运行时性能，你需要查找其列表是链表还是数组。

回到上一章中我们关于跑者的讨论，我们可以简单地使用 Python 列表（数组）而不是散列表，并且能够访问在特定位置完成比赛的跑者的名字。但让我们提出一个不同的问题。

我们如何报告每个年龄组的顶尖完成者？特别是，我们想要编写一个如下所示的函数：

```py
def top_5_range(runners: list, lo: int, high: int) -> list:
    """get list of top 5 finishers with ages in
       the range given by lo to high, inclusive
    """
```

思考一下你会如何编写这段代码。

这是我们的解决方案：

```py
def top_5_range(runners: list, lo: int, high: int) -> list:
    """get list of top 5 finishers with ages in
       the range given by lo to high, inclusive
    """

    # count of runners seen who are in age range
    in_range: int = 0
    # the list of finishers
    result: list = []

    for r in runners:
        if lo <= r.age and r.age <= high:
            in_range += 1
            result.append(r)
        if in_range == 5:
            return result
    print("Fewer than five in category")
    return result
```

在这里，我们不想只在到达列表末尾时才返回，我们希望在列表中有五个跑者时返回。因此，我们设置了一个额外的变量（`in_range`）来帮助我们跟踪计算的进度。一旦我们有了 5 个跑者，我们就返回列表。如果我们从未达到 5 个跑者，我们向用户打印一个警告然后返回我们已有的结果。

我们不能只查看列表的长度，而不是维护`in_range`变量吗？是的，我们可以这样做，尽管这个版本与我们的下一个例子形成对比。

#### 9.3.2 部分遍历有序数据 "链接到此处")

如果我们只想打印出前 5 名完成者，而不是收集一个列表，会怎样？虽然通常将计算和显示数据分开是更好的做法，但在实践中，我们有时会合并它们，或者执行其他操作（如将一些数据写入文件）这些操作不会返回任何内容。我们如何修改代码来打印名字而不是构建跑者列表？

这里的挑战是如何停止计算。当我们构建一个列表时，我们使用 return 来停止计算。但如果我们代码没有返回，或者需要在到达数据末尾之前停止循环，我们该怎么办？

我们使用一个名为 `break` 的命令，该命令指示终止循环并继续剩余的计算。在这里，`break` 代替了内部的返回语句：

```py
def print_top_5_range(runners: list, lo: int, high: int):
    """print top 5 finishers with ages in
       the range given by lo to high, inclusive
    """

    # count of runners seen who are in age range
    in_range: int = 0

    for r in runners:
        if lo <= r.age and r.age <= high:
            in_range += 1
            print(r.name)
        if in_range == 5:
            break
    print("End of results") 
```

如果 Python 遇到 `break` 语句，它将终止 for 循环并跳转到下一个语句，即函数末尾的打印语句。
