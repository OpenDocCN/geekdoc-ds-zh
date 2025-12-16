# 7.1 树🔗

> 原文：[`dcic-world.org/2025-08-27/trees.html`](https://dcic-world.org/2025-08-27/trees.html)

| |   7.1.1 数据设计问题 – 家谱数据 |
| --- | --- |
| |     7.1.1.1 从家谱表中计算遗传父母 |
| |     7.1.1.2 从家谱表中计算祖父母 |
| |     7.1.1.3 为家谱树创建数据类型 |
| |   7.1.2 处理家谱树的程序 |
| |   7.1.3 总结如何处理树问题 |
| |   7.1.4 研究问题 |

#### 7.1.1 数据设计问题 – 家谱数据🔗 "链接到这里")

想象一下，如果我们想管理用于医学研究目的的家谱信息。具体来说，我们想记录人们的出生年份、眼睛颜色和遗传父母。以下是这样数据的样本表，每行代表一个人：

```py
ancestors = table: name, birthyear, eyecolor, female-parent, male-parent
  row: "Anna", 1997, "blue", "Susan", "Charlie"
  row: "Susan", 1971, "blue", "Ellen", "Bill"
  row: "Charlie", 1972, "green", "", ""
  row: "Ellen", 1945, "brown", "Laura", "John"
  row: "John", 1922, "brown", "", "Robert"
  row: "Laura", 1922, "brown", "", ""
  row: "Robert", 1895, "blue", "", ""
end
```

为了我们的研究，我们希望能够回答以下问题：

+   某个特定人的遗传祖父母是谁？

+   每种眼睛颜色的频率是多少？

+   一个特定的人是否是另一个特定人的祖先？

+   我们有多少代的信息？

+   一个人出生时，他们的眼睛颜色与他们的遗传父母那时的年龄相关吗？

让我们从第一个问题开始：

> 现在行动！
> 
> > 你会如何计算给定人的已知祖父母的列表？为了本章的目的，你可以假设每个人都有一个独特的名字（虽然这在实践中并不现实，但为了简化我们的计算，我们暂时这样做；我们将在本章的后面重新讨论这个问题）。
> > 
> > （提示：制定一个任务计划。它是否建议任何特定的辅助函数？）

我们的任务计划包含两个关键步骤：找到指定人的遗传父母的姓名，然后找到这些人的父母的姓名。这两个步骤都需要从姓名计算已知的父母，因此我们应该为这个目的创建一个辅助函数（我们将称之为`parents-of`）。由于这听起来像是一个常规的表格程序，我们可以用它来复习一下：

##### 7.1.1.1 从家谱表中计算遗传父母🔗 "链接到这里")

我们如何计算某人的遗传父母列表？让我们为这个任务草拟一个计划：

+   过滤表格以找到该人

+   提取女性父母的姓名

+   提取男性父母的姓名

+   列出那些姓名

这些是我们之前见过的任务，因此我们可以直接将这个计划转换为代码：

```py
fun parents-of(anc-table :: Table, who :: String) -> List<String>:
  doc: "Return list of names of known parents of given name"
  matches = filter-with(anc-table, lam(r): r["name"] == who end)
  if matches.length() > 0:
    person-row = matches.row-n(0)
    [list:
      person-row["female-parent"],
      person-row["male-parent"]]
  else:
    empty
  end
where:
  parents-of(ancestors, "Anna")
    is [list: "Susan", "Charlie"]
  parents-of(ancestors, "Kathi") is empty
end
```

> 现在行动！
> 
> > 你对这个程序满意吗？包括在`where`块中的示例？写下你所有的批评意见。

这里可能有一些问题。你发现了多少？

+   例子很弱：它们都没有考虑至少有一位父母信息缺失的人。

+   在未知父母的情况下返回的姓名列表中包括空字符串，这实际上并不是一个名字。如果我们使用这个姓名列表进行后续计算（例如计算某人的祖父母的名字），这可能会引起问题。

+   如果空字符串不是输出列表的一部分，那么从请求`"Robert"`（他在表中）的父母和请求`"Kathi"`（不在表中）的父母将得到相同的结果。这些是根本不同的案例，可以说需要不同的输出以便我们可以区分它们。

为了修复这些问题，我们需要从产生的父母列表中删除空字符串，并在名字不在表中时返回除`empty`列表之外的内容。由于这个函数的输出是一个字符串列表，很难看出可以返回什么不会与有效的名字列表混淆。我们目前的解决方案是让 Pyret 抛出一个错误（就像你在 Pyret 无法完成运行你的程序时得到的那样）。下面是一个处理这两个问题的解决方案：

```py
fun parents-of(anc-table :: Table, who :: String) -> List<String>:
  doc: "Return list of names of known parents of given name"
  matches = filter-with(anc-table, lam(r): r["name"] == who end)
  if matches.length() > 0:
    person-row = matches.row-n(0)
    names =
     [list: person-row["female-parent"],
       person-row["male-parent"]]
    L.filter(lam(n): not(n == "") end, names)
  else:
    raise("No such person " + who)
  end
where:
  parents-of(ancestors, "Anna") is [list: "Susan", "Charlie"]
  parents-of(ancestors, "John") is [list: "Robert"]
  parents-of(ancestors, "Robert") is empty
  parents-of(ancestors, "Kathi") raises "No such person"
end
```

`raise`构造函数告诉 Pyret 停止程序并产生一个错误信息。错误信息不必与程序预期的输出类型相匹配。如果你用不在表中的名字运行这个函数，你将在交互面板中看到一个错误出现，并且没有返回结果。

在`where`块中，我们看到如何检查一个表达式是否会产生错误：我们不是使用`is`来检查值的相等性，而是使用`raises`来检查提供的字符串是否是程序实际产生的错误的一个子字符串。

##### 7.1.1.2 从家谱表中计算祖父母🔗 "链接至此")

一旦我们有了`parents-of`函数，我们应该能够通过计算父母的父母来计算祖父母，如下所示：

```py
fun grandparents-of(anc-table :: Table, who :: String) -> List<String>:
  doc: "compute list of known grandparents in the table"
  # glue together lists of mother's parents and father's parents
  plist = parents-of(anc-table, who) # gives a list of two names
  parents-of(anc-table, plist.first) +
    parents-of(anc-table, plist.rest.first)
where:
  grandparents-of(ancestors, "Anna") is [list: "Ellen", "Bill"]
  grandparents-of(ancestors, "Laura") is [list:]
  grandparents-of(ancestors, "John") is [list: ]
  grandparents-of(ancestors, "Kathi") is [list:]
end
```

> 现在行动起来！
> 
> > 回顾我们的样本家谱树：对于哪些人，这个计算将正确地计算出祖父母的列表？

当表中同时有两位父母时，这段计算祖父母的代码运行良好。然而，对于没有两位父母的人来说，`plist`将包含少于两个名字，因此`plist.rest.first`（如果不是`plist.first`）的表达式将产生错误。

这里是一个在计算祖父母集合之前检查父母数量的版本：

```py
fun grandparents-of(anc-table :: Table, who :: String) -> List<String>:
  doc: "compute list of known grandparents in the table"
  # glue together lists of mother's parents and father's parents
  plist = parents-of(anc-table, who) # gives a list of two names
  if plist.length() == 2:
    parents-of(anc-table, plist.first) + parents-of(anc-table, plist.rest.first)
  else if plist.length() == 1:
    parents-of(anc-table, plist.first)
  else: empty
  end
where:
  grandparents-of(ancestors, "Anna") is [list: "Ellen", "Bill"]
  grandparents-of(ancestors, "Laura") is [list:]
  grandparents-of(ancestors, "John") is [list: ]
  grandparents-of(ancestors, "Kathi") raises "No such person"
end
```

如果我们现在想要收集某人的所有祖先呢？由于我们不知道有多少代，我们需要使用递归。这种方法也会很昂贵，因为我们最终会在每次使用`filter`时多次过滤表，这会检查表的每一行。

回顾一下家谱树图片。在那里我们没有进行任何复杂的过滤——我们只是从一个人直接跟随到他们的母亲或父亲。我们能否在代码中实现这个想法？是的，通过数据类型。

##### 7.1.1.3 为祖先树创建数据类型🔗 "链接到这里")

对于这种方法，我们希望为祖先树创建一个具有设置个人变体（构造函数）的数据类型。回顾我们的图片——什么信息构成了一个人？他们的名字、他们的母亲和他们的父亲（以及出生年份和眼睛颜色，这些在图片中没有显示）。这表明以下数据类型，它基本上将一行转换为个人值：

```py
data AncTree:
  | person(
      name :: String,
      birthyear :: Number,
      eye :: String,
      mother :: ________,
      father :: ________
      )
end
```

例如，anna 的行可能看起来像：

```py
anna-row = person("Anna", 1997, "blue", ???, ???)
```

我们应该填什么类型？快速头脑风暴产生了几个想法：

+   `person`

+   `List<person>`

+   一些新的数据类型

+   `AncTree`

+   `String`

它应该是哪一个？

如果我们使用`String`，我们就回到了表格行，并且我们无法轻松地从一个人转到另一个人。因此，我们应该将其命名为`AncTree`。

```py
data AncTree:
  | person(
      name :: String,
      birthyear :: Number,
      eye :: String,
      mother :: AncTree,
      father :: AncTree
      )
end
```

> 现在行动！
> 
> > 使用这个定义从`Anna`开始编写`AncTree`。

你卡住了吗？当我们用完已知的人时，我们该怎么办？为了处理这种情况，我们必须在`AncTree`定义中添加一个选项来捕获我们不知道任何信息的人。

```py
data AncTree:
  | noInfo
  | person(
      name :: String,
      birthyear :: Number,
      eye :: String,
      mother :: AncTree,
      father :: AncTree
      )
end
```

下面是使用这种数据类型编写的 Anna 的树：

```py
anna-tree =
  person("Anna", 1997, "blue",
    person("Susan", 1971, "blue",
      person("Ellen", 1945, "brown",
        person("Laura", 1920, "blue", noInfo, noInfo),
        person("John", 1920, "green",
          noInfo,
          person("Robert", 1893, "brown", noInfo, noInfo))),
      person("Bill", 1946, "blue", noInfo, noInfo)),
    person("Charlie", 1972, "green", noInfo, noInfo))
```

我们也可以为每个人的数据单独命名。

```py
robert-tree = person("Robert", 1893, "brown", noInfo, noInfo)
laura-tree = person("Laura", 1920, "blue", noInfo, noInfo)
john-tree = person("John", 1920, "green", noInfo, robert-tree)
ellen-tree = person("Ellen", 1945, "brown", laura-tree, john-tree)
bill-tree = person("Bill", 1946, "blue", noInfo, noInfo)
susan-tree = person("Susan", 1971, "blue", ellen-tree, bill-tree)
charlie-tree = person("Charlie", 1972, "green", noInfo, noInfo)
anna-tree2 = person("Anna", 1997, "blue", susan-tree, charlie-tree)
```

后者为你提供了可以用于其他示例的树的片段，但失去了第一个版本中可见的结构。你可以通过挖掘数据来获得第一个版本的片段，例如编写`anna-tree.mother.mother`来获取从“Ellen”开始的树。

下面是针对`AncTree`编写的`parents-of`函数：

```py
fun parents-of-tree(tr :: AncTree) -> List<String>:
  cases (AncTree) tr:
    | noInfo => empty
    | person(n, y, e, m, f) => [list: m.name, f.name]
      # person bit more complicated if parent is missing
  end
end
```

#### 7.1.2 处理祖先树的程序🔗 "链接到这里")

我们如何编写一个函数来确定树中是否有人有特定的名字？为了清楚起见，我们正在尝试填写以下代码：

```py
fun in-tree(at :: AncTree, name :: String) -> Boolean:
  doc: "determine whether name is in the tree"
  ...
```

我们如何开始？添加一些示例，记得检查`AncTree`定义的两种情况：

```py
fun in-tree(at :: AncTree, name :: String) -> Boolean:
  doc: "determine whether name is in the tree"
  ...
where:
  in-tree(anna-tree, "Anna") is true
  in-tree(anna-tree, "Ellen") is true
  in-tree(ellen-tree, "Anna") is false
  in-tree(noInfo, "Ellen") is false
end
```

接下来是什么？当我们处理列表时，我们讨论了模板，这是我们根据数据结构知道我们可以编写的代码框架。模板命名了每种数据类型的各个部分，并对具有相同类型的部分进行递归调用。下面是填充在`AncTree`上的模板：

```py
fun in-tree(at :: AncTree, name :: String) -> Boolean:
  doc: "determine whether name is in the tree"
  cases (AncTree) at:     # comes from AncTree being data with cases
    | noInfo => ...
    | person(n, y, e, m, f) => ... in-tree(m, name) ... in-tree(f, name)
  end
where:
  in-tree(anna-tree, "Anna") is true
  in-tree(anna-tree, "Ellen") is true
  in-tree(ellen-tree, "Anna") is false
  in-tree(noInfo, "Ellen") is false
end
```

完成代码，我们需要思考如何填充省略号。

+   当树是`noInfo`时，它没有更多的人，所以答案应该是 false（如示例中所示）。

+   当树是一个人时，有三种可能性：我们可能在一个具有我们正在寻找的名字的人那里，或者名字可能在母亲的树中，或者名字可能在父亲的树中。

    我们知道如何检查一个人的名字是否与我们正在寻找的名字匹配。递归调用已经询问了名字是否在母亲的树或父亲的树中。我们只需要将这些部分组合成一个布尔答案。由于有三种可能性，我们应该使用`or`来组合它们。

下面是最终的代码：

```py
fun in-tree(at :: AncTree, name :: String) -> Boolean:
  doc: "determine whether name is in the tree"
  cases (AncTree) at:     # comes from AncTree being data with cases
    | noInfo => false
    | person(n, y, e, m, f) => (name == n) or in-tree(m, name) or in-tree(f, name)
      # n is the same as at.name
      # m is the same as at.mother
  end
where:
  in-tree(anna-tree, "Anna") is true
  in-tree(anna-tree, "Ellen") is true
  in-tree(ellen-tree, "Anna") is false
  in-tree(noInfo, "Ellen") is false
end
```

#### 7.1.3 总结如何处理树问题🔗 "链接至此")

我们使用与我们在列表上覆盖的设计配方设计树程序：

> 策略：在树上编写程序
> 
> > +   为你的树编写数据类型，包括基本/叶节点情况
> > +   
> > +   为测试编写你的树示例
> > +   
> > +   编写函数名、参数和类型（`fun`行）
> > +   
> > +   编写`where`检查你的代码
> > +   
> > +   编写模板，包括情况和递归调用。以下是祖先树的模板，对于任意函数称为 treeF：
> > +   
> >     ```py
> >     fun treeF(name :: String, t :: AncTree) -> Boolean:
> >       cases (AncTree) anct:
> >         | unknown => ...
> >         | person(n, y, e, m, f) =>
> >          ... treeF(name, m) ... treeF(name, f)
> >       end
> >     end
> >     ```
> >     
> > +   用具体问题的细节填写模板
> > +   
> > +   使用你的示例测试你的代码

#### 7.1.4 学习问题🔗 "链接至此")

+   想象在表格（使用按过滤）上和在树上编写。每种方法可能需要多少次将所需名称与表格/树中的名称进行比较？

+   为什么我们需要使用递归函数来处理树？

+   我们将按什么顺序检查树版本中的名称？

为了练习，尝试以下问题

+   树中有多少蓝眼睛的人？

+   树中有多少人？

+   树中有多少代？

+   在树中有多少人有一个特定的名字？

+   有多少人的名字以"A"开头？

+   ...等等

#### 7.1.1 数据设计问题 – 家谱数据🔗 "链接至此")

想象一下，如果我们想为了医学研究的目的管理家谱信息。具体来说，我们想记录人们的出生年份、眼睛颜色和遗传父母。以下是这样数据的样本表，每人一行：

```py
ancestors = table: name, birthyear, eyecolor, female-parent, male-parent
  row: "Anna", 1997, "blue", "Susan", "Charlie"
  row: "Susan", 1971, "blue", "Ellen", "Bill"
  row: "Charlie", 1972, "green", "", ""
  row: "Ellen", 1945, "brown", "Laura", "John"
  row: "John", 1922, "brown", "", "Robert"
  row: "Laura", 1922, "brown", "", ""
  row: "Robert", 1895, "blue", "", ""
end
```

为了我们的研究，我们希望能够回答以下问题：

+   特定人员的遗传祖父母是谁？

+   每种眼睛颜色的频率是多少？

+   是否有特定的人是另一个特定人的祖先？

+   我们有多少代的信息？

+   一个人出生时，他们的眼睛颜色是否与遗传父母的年龄相关？

让我们从第一个问题开始：

> 立即行动！
> 
> > 你会如何计算给定人员的已知祖父母列表？在本章中，你可以假设每个人都有一个独特的名字（虽然这在实践中并不现实，但这将简化我们目前的计算；我们将在本章稍后重新讨论这个问题）。
> > 
> > （提示：制定任务计划。它是否建议任何特定的辅助函数？）

我们的任务计划有两个关键步骤：找到指定人员的遗传父母的姓名，然后找到这些人员的父母的姓名。这两个步骤都需要从姓名计算已知父母，因此我们应该为这个创建一个辅助函数（我们将它称为`parents-of`）。由于这听起来像是一个常规的表格程序，我们可以用它来复习一下：

##### 7.1.1.1 从家谱表中计算遗传父母🔗 "链接至此")

我们如何计算某人的遗传父母列表？让我们为这个任务草拟一个计划：

+   过滤表格以找到这个人

+   提取女性父母的姓名

+   提取男性父母的姓名

+   列出这些名字

这些是我们之前见过的任务，所以我们可以直接将这个计划转换为代码：

```py
fun parents-of(anc-table :: Table, who :: String) -> List<String>:
  doc: "Return list of names of known parents of given name"
  matches = filter-with(anc-table, lam(r): r["name"] == who end)
  if matches.length() > 0:
    person-row = matches.row-n(0)
    [list:
      person-row["female-parent"],
      person-row["male-parent"]]
  else:
    empty
  end
where:
  parents-of(ancestors, "Anna")
    is [list: "Susan", "Charlie"]
  parents-of(ancestors, "Kathi") is empty
end
```

> 立刻行动！
> 
> > 你对这个程序满意吗？包括在`where`块中的示例？写下你所有的批评意见。

这里可能有一些问题。你发现了多少？

+   示例很弱：它们都没有考虑至少一个父母信息缺失的人。

+   在未知父母的情况下返回的姓名列表中包括空字符串，这实际上并不是一个名字。如果我们使用这个姓名列表进行后续计算（例如计算某人的祖父母姓名），这可能会引起问题。

+   如果空字符串不是输出列表的一部分，那么我们询问`"Robert"`（他在表中）的父母和询问`"Kathi"`（他不在表中）的父母将得到相同的结果。这些是根本不同的案例，可以说需要不同的输出以便我们可以区分它们。

为了修复这些问题，我们需要从产生的父母列表中删除空字符串，并在姓名不在表中时返回除`empty`列表之外的内容。由于这个函数的输出是一个字符串列表，很难看出可以返回什么不会与有效的姓名列表混淆。我们现在的解决方案是让 Pyret 抛出一个错误（就像你在 Pyret 无法完成运行你的程序时得到的那样）。这是一个处理这两个问题的解决方案：

```py
fun parents-of(anc-table :: Table, who :: String) -> List<String>:
  doc: "Return list of names of known parents of given name"
  matches = filter-with(anc-table, lam(r): r["name"] == who end)
  if matches.length() > 0:
    person-row = matches.row-n(0)
    names =
     [list: person-row["female-parent"],
       person-row["male-parent"]]
    L.filter(lam(n): not(n == "") end, names)
  else:
    raise("No such person " + who)
  end
where:
  parents-of(ancestors, "Anna") is [list: "Susan", "Charlie"]
  parents-of(ancestors, "John") is [list: "Robert"]
  parents-of(ancestors, "Robert") is empty
  parents-of(ancestors, "Kathi") raises "No such person"
end
```

`raise`构造函数告诉 Pyret 停止程序并产生一个错误信息。错误信息不必与程序预期的输出类型相匹配。如果你用不在表格中的名字运行这个函数，你会在交互面板中看到一个错误出现，并且没有返回结果。

在`where`块中，我们看到如何检查表达式是否会产生错误：我们不是使用`is`来检查值的相等性，而是使用`raises`来检查提供的字符串是否是程序实际产生的错误的一个子字符串。

##### 7.1.1.2 从家谱表中计算祖父母🔗 "链接至此")

一旦我们有了`parents-of`函数，我们应该能够通过计算父母的父母来计算祖父母，如下所示：

```py
fun grandparents-of(anc-table :: Table, who :: String) -> List<String>:
  doc: "compute list of known grandparents in the table"
  # glue together lists of mother's parents and father's parents
  plist = parents-of(anc-table, who) # gives a list of two names
  parents-of(anc-table, plist.first) +
    parents-of(anc-table, plist.rest.first)
where:
  grandparents-of(ancestors, "Anna") is [list: "Ellen", "Bill"]
  grandparents-of(ancestors, "Laura") is [list:]
  grandparents-of(ancestors, "John") is [list: ]
  grandparents-of(ancestors, "Kathi") is [list:]
end
```

> 立刻行动！
> 
> > 回顾我们的样本家谱树：对于哪些人，这个计算可以正确地计算出祖父母列表？

这段计算祖父母的代码对于表格中同时有双亲的人来说是有效的。然而，对于没有双亲的人来说，`plist`将少于两个名字，所以表达式`plist.rest.first`（如果不是`plist.first`）将产生错误。

这是一个在计算祖父母集合之前检查父母数量的版本：

```py
fun grandparents-of(anc-table :: Table, who :: String) -> List<String>:
  doc: "compute list of known grandparents in the table"
  # glue together lists of mother's parents and father's parents
  plist = parents-of(anc-table, who) # gives a list of two names
  if plist.length() == 2:
    parents-of(anc-table, plist.first) + parents-of(anc-table, plist.rest.first)
  else if plist.length() == 1:
    parents-of(anc-table, plist.first)
  else: empty
  end
where:
  grandparents-of(ancestors, "Anna") is [list: "Ellen", "Bill"]
  grandparents-of(ancestors, "Laura") is [list:]
  grandparents-of(ancestors, "John") is [list: ]
  grandparents-of(ancestors, "Kathi") raises "No such person"
end
```

如果我们现在想收集某人的所有祖先呢？由于我们不知道有多少代，我们需要使用递归。这种方法也会很昂贵，因为我们最终会多次过滤表，每次使用`filter`都会检查表中的每一行。

回顾祖先树图片。我们那里没有做任何复杂的过滤——我们只是从一个人直接跟随图片中的线条到他们的母亲或父亲。我们能否在代码中实现这个想法？是的，通过数据类型。

##### 7.1.1.3 创建祖先树的数据类型🔗 "链接到这里")

对于这种方法，我们希望为祖先树创建一个数据类型，它有一个用于设置个人的变体（构造函数）。回顾我们的图片——什么信息构成了一个人？他们的名字、他们的母亲和他们的父亲（以及出生年份和眼睛颜色，这些在图片中没有显示）。这表明以下数据类型，它基本上将一行转换成个人值：

```py
data AncTree:
  | person(
      name :: String,
      birthyear :: Number,
      eye :: String,
      mother :: ________,
      father :: ________
      )
end
```

例如，安娜的行可能看起来像：

```py
anna-row = person("Anna", 1997, "blue", ???, ???)
```

我们应该把什么类型填入空白处？快速头脑风暴产生了几个想法：

+   `person`

+   `List<person>`

+   一些新的数据类型

+   `AncTree`

+   `String`

它应该是这样的？

如果我们使用一个`String`，我们就会回到表格行，并且无法轻松地从一个人跳转到另一个人。因此，我们应该将其做成一个`AncTree`。

```py
data AncTree:
  | person(
      name :: String,
      birthyear :: Number,
      eye :: String,
      mother :: AncTree,
      father :: AncTree
      )
end
```

> 现在就做！
> 
> > 使用这个定义从`Anna`开始编写`AncTree`。

你卡住了吗？当我们用完已知的人时，我们该怎么办？为了处理这个问题，我们必须在`AncTree`定义中添加一个选项来捕捉我们一无所知的人。

```py
data AncTree:
  | noInfo
  | person(
      name :: String,
      birthyear :: Number,
      eye :: String,
      mother :: AncTree,
      father :: AncTree
      )
end
```

这是用这种数据类型编写的安娜的树：

```py
anna-tree =
  person("Anna", 1997, "blue",
    person("Susan", 1971, "blue",
      person("Ellen", 1945, "brown",
        person("Laura", 1920, "blue", noInfo, noInfo),
        person("John", 1920, "green",
          noInfo,
          person("Robert", 1893, "brown", noInfo, noInfo))),
      person("Bill", 1946, "blue", noInfo, noInfo)),
    person("Charlie", 1972, "green", noInfo, noInfo))
```

我们也可以为每个个人数据单独命名。

```py
robert-tree = person("Robert", 1893, "brown", noInfo, noInfo)
laura-tree = person("Laura", 1920, "blue", noInfo, noInfo)
john-tree = person("John", 1920, "green", noInfo, robert-tree)
ellen-tree = person("Ellen", 1945, "brown", laura-tree, john-tree)
bill-tree = person("Bill", 1946, "blue", noInfo, noInfo)
susan-tree = person("Susan", 1971, "blue", ellen-tree, bill-tree)
charlie-tree = person("Charlie", 1972, "green", noInfo, noInfo)
anna-tree2 = person("Anna", 1997, "blue", susan-tree, charlie-tree)
```

后者提供了树的部分以用作其他示例，但失去了第一版中可见的结构。你可以通过深入数据来获得第一版的部分，例如，通过编写`anna-tree.mother.mother`来从“Ellen”开始获取树。

这是针对`AncTree`编写的`parents-of`函数：

```py
fun parents-of-tree(tr :: AncTree) -> List<String>:
  cases (AncTree) tr:
    | noInfo => empty
    | person(n, y, e, m, f) => [list: m.name, f.name]
      # person bit more complicated if parent is missing
  end
end
```

##### 7.1.1.1 从家谱表中计算遗传父母🔗 "链接到这里")

我们如何计算某人的遗传父母列表？让我们为这个任务草拟一个计划：

+   过滤表以找到个人

+   提取女性父母的姓名

+   提取男性父母的姓名

+   制作这些名字的列表

这些是我们之前见过的任务，因此我们可以直接将此计划转换为代码：

```py
fun parents-of(anc-table :: Table, who :: String) -> List<String>:
  doc: "Return list of names of known parents of given name"
  matches = filter-with(anc-table, lam(r): r["name"] == who end)
  if matches.length() > 0:
    person-row = matches.row-n(0)
    [list:
      person-row["female-parent"],
      person-row["male-parent"]]
  else:
    empty
  end
where:
  parents-of(ancestors, "Anna")
    is [list: "Susan", "Charlie"]
  parents-of(ancestors, "Kathi") is empty
end
```

> 现在就做！
> 
> > 你对这个程序满意吗？包括在`where`块中的示例吗？写下你所有的批评。

这里可能有一些问题。你抓住了多少？

+   例子不够强大：它们中没有一个考虑至少缺失一个父母信息的个人。

+   在未知父母的情况下返回的姓名列表中包含空字符串，这实际上并不是一个姓名。如果我们使用这个姓名列表进行后续计算（例如计算某人的祖父母姓名），这可能会引起问题。

+   如果空字符串不是输出列表的一部分，那么从请求 `"Robert"`（他在表中）的父母和请求 `"Kathi"`（她不在表中）的父母将得到相同的结果。这些是根本不同的案例，可以说需要不同的输出以便我们可以区分它们。

为了解决这些问题，我们需要从生成的父母列表中移除空字符串，并在姓名不在表中时返回除空列表之外的内容。由于此函数的输出是一个字符串列表，很难看出返回的内容不会与有效的姓名列表混淆。我们目前的解决方案是让 Pyret 抛出错误（就像 Pyret 无法完成运行你的程序时得到的错误一样）。以下是一个解决这两个问题的解决方案：

```py
fun parents-of(anc-table :: Table, who :: String) -> List<String>:
  doc: "Return list of names of known parents of given name"
  matches = filter-with(anc-table, lam(r): r["name"] == who end)
  if matches.length() > 0:
    person-row = matches.row-n(0)
    names =
     [list: person-row["female-parent"],
       person-row["male-parent"]]
    L.filter(lam(n): not(n == "") end, names)
  else:
    raise("No such person " + who)
  end
where:
  parents-of(ancestors, "Anna") is [list: "Susan", "Charlie"]
  parents-of(ancestors, "John") is [list: "Robert"]
  parents-of(ancestors, "Robert") is empty
  parents-of(ancestors, "Kathi") raises "No such person"
end
```

`raise` 构造函数告诉 Pyret 停止程序并生成一个错误消息。错误消息不需要与程序的预期输出类型匹配。如果你用不在表中的姓名运行此函数，你将在交互式面板中看到一个错误出现，并且没有返回结果。

在 `where` 块中，我们看到如何检查表达式是否会生成错误：我们不是使用 `is` 来检查值的相等性，而是使用 `raises` 来检查提供的字符串是否是程序实际产生的错误消息的子字符串。

##### 7.1.1.2 从家谱表中计算祖父母🔗 "链接到此处")

一旦我们有了 `parents-of` 函数，我们就应该能够通过计算父母的父母来计算祖父母，如下所示：

```py
fun grandparents-of(anc-table :: Table, who :: String) -> List<String>:
  doc: "compute list of known grandparents in the table"
  # glue together lists of mother's parents and father's parents
  plist = parents-of(anc-table, who) # gives a list of two names
  parents-of(anc-table, plist.first) +
    parents-of(anc-table, plist.rest.first)
where:
  grandparents-of(ancestors, "Anna") is [list: "Ellen", "Bill"]
  grandparents-of(ancestors, "Laura") is [list:]
  grandparents-of(ancestors, "John") is [list: ]
  grandparents-of(ancestors, "Kathi") is [list:]
end
```

> 立刻行动！
> 
> > 回顾我们的示例家谱树：对于哪些人，这将正确计算出祖父母的列表？

这段关于祖父母的代码对于在表中都有父母的个人来说是有效的。然而，对于没有两个父母的个人，`plist` 将包含少于两个姓名，因此表达式 `plist.rest.first`（如果不是 `plist.first`）将产生错误。

这是一个在计算祖父母的集合之前检查父母数量的版本：

```py
fun grandparents-of(anc-table :: Table, who :: String) -> List<String>:
  doc: "compute list of known grandparents in the table"
  # glue together lists of mother's parents and father's parents
  plist = parents-of(anc-table, who) # gives a list of two names
  if plist.length() == 2:
    parents-of(anc-table, plist.first) + parents-of(anc-table, plist.rest.first)
  else if plist.length() == 1:
    parents-of(anc-table, plist.first)
  else: empty
  end
where:
  grandparents-of(ancestors, "Anna") is [list: "Ellen", "Bill"]
  grandparents-of(ancestors, "Laura") is [list:]
  grandparents-of(ancestors, "John") is [list: ]
  grandparents-of(ancestors, "Kathi") raises "No such person"
end
```

如果我们现在想收集某人的所有祖先呢？由于我们不知道有多少代，我们需要使用递归。这种方法也会很昂贵，因为我们最终会多次过滤表，每次使用 `filter` 都会检查表的每一行。

回顾家谱树图片。我们那里没有进行任何复杂的过滤——我们只是从一个人直接跟随到他们的母亲或父亲。我们能否在代码中实现这个想法？是的，通过数据类型。

##### 7.1.1.3 创建祖先树的数据类型🔗 "链接到此处")

对于这种方法，我们希望为祖先树创建一个数据类型，它有一个用于设置个人的变体（构造函数）。回顾我们的图片——什么信息构成了一个人？他们的名字、他们的母亲和他们的父亲（以及出生年份和眼睛颜色，这些在图片中没有显示）。这建议以下数据类型，它基本上将一行转换为个人值：

```py
data AncTree:
  | person(
      name :: String,
      birthyear :: Number,
      eye :: String,
      mother :: ________,
      father :: ________
      )
end
```

例如，anna 的行可能看起来像这样：

```py
anna-row = person("Anna", 1997, "blue", ???, ???)
```

我们应该填入什么类型？快速头脑风暴产生了几个想法：

+   `person`

+   `List<person>`

+   一些新的数据类型

+   `AncTree`

+   `String`

应该选择哪一个？

如果我们使用`String`，我们就回到了表行，并且无法轻松地从一个人转到另一个人。因此，我们应该将其作为`AncTree`。

```py
data AncTree:
  | person(
      name :: String,
      birthyear :: Number,
      eye :: String,
      mother :: AncTree,
      father :: AncTree
      )
end
```

> 现在做什么？
> 
> > 使用这个定义从`Anna`开始编写`AncTree`。

你卡住了吗？当我们用尽已知的人时，我们该怎么办？为了处理这种情况，我们必须在`AncTree`定义中添加一个选项来捕获我们一无所知的人。

```py
data AncTree:
  | noInfo
  | person(
      name :: String,
      birthyear :: Number,
      eye :: String,
      mother :: AncTree,
      father :: AncTree
      )
end
```

这里是使用这种数据类型编写的 Anna 的树：

```py
anna-tree =
  person("Anna", 1997, "blue",
    person("Susan", 1971, "blue",
      person("Ellen", 1945, "brown",
        person("Laura", 1920, "blue", noInfo, noInfo),
        person("John", 1920, "green",
          noInfo,
          person("Robert", 1893, "brown", noInfo, noInfo))),
      person("Bill", 1946, "blue", noInfo, noInfo)),
    person("Charlie", 1972, "green", noInfo, noInfo))
```

我们也可以为每个人数据单独命名。

```py
robert-tree = person("Robert", 1893, "brown", noInfo, noInfo)
laura-tree = person("Laura", 1920, "blue", noInfo, noInfo)
john-tree = person("John", 1920, "green", noInfo, robert-tree)
ellen-tree = person("Ellen", 1945, "brown", laura-tree, john-tree)
bill-tree = person("Bill", 1946, "blue", noInfo, noInfo)
susan-tree = person("Susan", 1971, "blue", ellen-tree, bill-tree)
charlie-tree = person("Charlie", 1972, "green", noInfo, noInfo)
anna-tree2 = person("Anna", 1997, "blue", susan-tree, charlie-tree)
```

后者提供了可以用于其他示例的树的片段，但失去了第一个版本中可见的缩进结构。你可以通过深入数据来获得第一个版本的片段，例如，通过编写`anna-tree.mother.mother`来从"Ellen"开始获取树。

这里是针对`AncTree`编写的`parents-of`函数：

```py
fun parents-of-tree(tr :: AncTree) -> List<String>:
  cases (AncTree) tr:
    | noInfo => empty
    | person(n, y, e, m, f) => [list: m.name, f.name]
      # person bit more complicated if parent is missing
  end
end
```

#### 7.1.2 处理祖先树的程序🔗 "链接至此")

我们如何编写一个函数来确定树中是否有人有特定的名字？为了清楚起见，我们正在尝试填写以下代码：

```py
fun in-tree(at :: AncTree, name :: String) -> Boolean:
  doc: "determine whether name is in the tree"
  ...
```

我们如何开始？添加一些示例，记得检查`AncTree`定义的两种情况：

```py
fun in-tree(at :: AncTree, name :: String) -> Boolean:
  doc: "determine whether name is in the tree"
  ...
where:
  in-tree(anna-tree, "Anna") is true
  in-tree(anna-tree, "Ellen") is true
  in-tree(ellen-tree, "Anna") is false
  in-tree(noInfo, "Ellen") is false
end
```

接下来是什么？当我们处理列表时，我们谈论了模板，这是我们根据数据结构知道我们可以编写的代码的骨架。模板命名了每种数据类型的各个部分，并在具有相同类型的部分上执行递归调用。这里是填充了`AncTree`的模板：

```py
fun in-tree(at :: AncTree, name :: String) -> Boolean:
  doc: "determine whether name is in the tree"
  cases (AncTree) at:     # comes from AncTree being data with cases
    | noInfo => ...
    | person(n, y, e, m, f) => ... in-tree(m, name) ... in-tree(f, name)
  end
where:
  in-tree(anna-tree, "Anna") is true
  in-tree(anna-tree, "Ellen") is true
  in-tree(ellen-tree, "Anna") is false
  in-tree(noInfo, "Ellen") is false
end
```

为了完成代码，我们需要考虑如何填充省略号。

+   当树是`noInfo`时，它没有更多的人，所以答案应该是 false（如示例中所示）。

+   当树代表一个人时，有三种可能性：我们可能找到了我们要找的名字对应的人，或者名字可能在母亲的树中，或者名字可能在父亲的树中。

    我们知道如何检查人的名字是否与我们寻找的名字匹配。递归调用已经询问了名字是否在母亲的树或父亲的树中。我们只需要将这些部分组合成一个布尔答案。由于有三种可能性，我们应该用`or`组合它们。

这里是最终的代码：

```py
fun in-tree(at :: AncTree, name :: String) -> Boolean:
  doc: "determine whether name is in the tree"
  cases (AncTree) at:     # comes from AncTree being data with cases
    | noInfo => false
    | person(n, y, e, m, f) => (name == n) or in-tree(m, name) or in-tree(f, name)
      # n is the same as at.name
      # m is the same as at.mother
  end
where:
  in-tree(anna-tree, "Anna") is true
  in-tree(anna-tree, "Ellen") is true
  in-tree(ellen-tree, "Anna") is false
  in-tree(noInfo, "Ellen") is false
end
```

#### 7.1.3 总结如何处理树问题🔗 "链接至此")

我们使用在列表中介绍过的相同设计方法来设计树程序：

> 策略：在树上编写程序
> 
> > +   为你的树编写数据类型，包括基本/叶节点情况
> > +   
> > +   编写你的树示例，用于测试
> > +   
> > +   写出函数名、参数和类型（`fun`行）
> > +   
> > +   为你的代码编写`where`检查
> > +   
> > +   编写模板，包括情况和递归调用。以下是祖先树模板的再次呈现，用于一个任意函数称为 treeF：
> > +   
> >     ```py
> >     fun treeF(name :: String, t :: AncTree) -> Boolean:
> >       cases (AncTree) anct:
> >         | unknown => ...
> >         | person(n, y, e, m, f) =>
> >          ... treeF(name, m) ... treeF(name, f)
> >       end
> >     end
> >     ```
> >     
> > +   使用详细信息填写模板以针对问题
> > +   
> > +   使用你的示例测试你的代码

#### 7.1.4 研究问题🔗 "链接到此处")

+   想象在表格（使用按条件筛选）上编写-in-tree，与在树上编写相比，每种方法可能多少次将搜索的名称与表/树中的名称进行比较？

+   为什么我们需要使用递归函数来处理树？

+   我们将按什么顺序检查树版本中的名称？

为了练习，尝试以下问题

+   树中有多少人名字是蓝色的？

+   树中有多少人？

+   树中有多少代？

+   有多少人名字在树中？

+   有多少人名字以"A"开头？

+   ...等等
