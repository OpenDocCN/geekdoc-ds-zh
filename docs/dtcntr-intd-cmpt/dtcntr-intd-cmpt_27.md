# 9.2 字典

> 原文：[`dcic-world.org/2025-08-27/dictionaries.html`](https://dcic-world.org/2025-08-27/dictionaries.html)

|     9.2.1 创建和使用字典 |
| --- |
|     9.2.2 在字典中搜索值 |
|     9.2.3 具有更复杂值的字典 |
|     9.2.4 字典与数据类 |
|       摘要 |

到目前为止，我们已经看到了几种处理顺序数据（如列表）的方法。在 Pyret 和 Python 中，我们可以使用 `filter` 和 `map` 来执行产生列表的某些操作。在 Pyret 中，我们使用递归将列表数据聚合为单个值。在 Python 中，我们使用 for 循环来完成这项任务。虽然我们也可以使用递归或 for 循环来处理 `filter` 和 `map` 任务，但使用这些命名的操作符可以使其他人更快地阅读你的代码并理解它执行的操作类型。

这个观察结果引发了一个问题：是否有其他常见的代码模式是用递归或 for 循环编写的，这些模式会从专门的处理中受益？

例如，假设我们有一个用于航班的数据类。每个航班都有其出发地和目的地城市、航班代码（包括航空公司名称和航班号），以及航班上的座位数。想象一下，我们还拥有查找单个航班的目的地和座位容量的函数：

```py
@dataclass
class Flight:
    from_city: str
    to_city: str
    code: str
    seats: int

schedule = [Flight('NYC','PVD','CSA-342',50),
            Flight('PVD','ORD','CSA-590',50),
            Flight('NYC','ORD','CSA-723',120),
            Flight('ORD','DEN','CSA-145',175),
            Flight('BOS','ORD','CSA-647',80)]

def destination1(for_code: str, flights: list):
   '''get the destination of the flight with the given code'''
   for fl in flights:
      if fl.code == for_code:
          return fl.to_city

def capacity1(for_code: str, flights: list):
   '''get the seating capacity of the flight with the given code'''
   for fl in flights:
      if fl.code == for_code:
          return fl.seats
```

> 现在行动起来！
> 
> > 看看 `destination1` 和 `capacity1` 之间的相似性。我们如何在这两个函数之间共享通用代码？

`destination1` 和 `capacity1` 都遍历航班列表，寻找具有给定航班代码的航班，然后从该航班中提取一些信息。for 循环除了寻找所需的航班数据外，没有做其他任何事情。这表明在这里使用一个 `find_flight` 辅助函数可能会有用：

```py
def find_flight(for_code: str, flights: list):
   '''return the flight with the given code'''
   for fl in flights:
      if fl.code == for_code:
          return fl

def destination2(for_code: str, flights: list):
    return find_flight(for_code, flights).to_city

def capacity2(for_code: str, flights: list):
    return find_flight(for_code, flights).seats
```

在基于特定信息从列表中搜索单个元素在许多程序中很常见。事实上，这种情况如此普遍，以至于语言提供了特殊的数据结构和操作来帮助完成这项任务。在 Python 中，这种数据结构称为字典（在其他语言中，类似的数据结构被称为 hashmap、hashtable 和关联数组，尽管所有这些变体都有一些关键的区别）。

#### 9.2.1 创建和使用字典 "链接到这里")

字典将唯一的值（称为键）映射到每个键对应的数据片段（称为值）。以下是我们将航班示例改写为字典而不是列表的形式：

```py
sched_dict = {'CSA-342': Flight('NYC','PVD','CSA-342',50),
              'CSA-590': Flight('PVD','ORD','CSA-590',50),
              'CSA-723': Flight('NYC','ORD','CSA-723',120),
              'CSA-145': Flight('ORD','DEN','CSA-145',175),
              'CSA-647': Flight('BOS','ORD','CSA-647',80)
             }
```

字典的一般形式是：

```py
{key1: value1,
 key2: value2,
 key3: value3,
 ...}
```

字典被设计成能够通过键值轻松查找值。为了获取

```py
Flight
```

关联到键 `'CSA-145'` 的，我们可以简单地写：

```py
sched_dict['CSA-145']
```

要获取航班`'CSA-145'`的座位数，我们可以简单地写下：

```py
sched_dict['CSA-145'].seats
```

换句话说，字典数据结构消除了遍历列表以找到具有特定键的`Flight`的需要。字典查找操作为我们完成了这项工作。实际上，字典甚至更加微妙：根据它们的设计方式，字典可以在不遍历所有值（甚至任何其他值）的情况下检索键的值。一般来说，你可以假设基于字典的查找比基于列表的查找要快得多。这是更高级的话题；其中一些内容在[SECREF]中有所解释。

字典的一个局限性是它们只允许每个键一个值。让我们考虑一个不同的例子，这次使用建筑中的房间作为键，占用者作为值：

```py
office_dict = {410: 'Farhan',
               411: 'Pauline',
               412: 'Marisol',
               413: 'Saleh'}
```

如果有人搬进了办公室 412？在 Python 中，我们可以这样更改该键的值：

```py
office_dict[412] = 'Zeynep'
```

现在，任何对`office_dict[412]`的使用都将评估为`'Zeynep'`而不是`'Marisol'`。

#### 9.2.2 在字典中搜索值 "链接到此处")

如果我们想找到所有座位数超过 100 的航班呢？为此，我们必须遍历所有的键值对并检查它们的余额。这听起来又像是我们需要一个 for 循环。但在字典上这看起来会是什么样子呢？

结果表明，这看起来就像在列表上编写 for 循环（至少在 Python 中是这样）。以下是一个创建座位数超过 100 个座位的航班列表的程序：

```py
above_100 = []

# the room variable takes on each key in the dictionary
for flight_code in sched_dict:
    if sched_dict[flight_code].seats > 100:
        above_100.append(sched_dict[flight_code])
```

在这里，for 循环遍历键。在循环内部，我们使用每个键来检索其对应的`Flight`，对`Flight`进行余额检查，然后如果它符合我们的标准，就将`Flight`放入我们的运行列表中。

> 练习
> 
> > 创建一个将教室或会议室的名称映射到它们座位数的字典。编写表达式来：
> > 
> > 1.  查询特定房间有多少座位
> > 1.  
> > 1.  将特定房间的容量改为比最初多 10 个座位
> > 1.  
> > 1.  找出可以容纳至少 50 名学生的所有房间

#### 9.2.3 具有更复杂值的字典 "链接到此处")

> 现在行动！
> 
> > 田径锦标赛需要管理将要参加比赛的各个队伍的运动员名单。例如，“红队”有“Shaoming”和“Lijin”，“绿队”包含“Obi”和“Chinara”，而“蓝队”有“Mateo”和“Sophia”。想出一个组织数据的方法，使得组织者可以轻松访问每个队伍的运动员名单，同时考虑到可能比这里列出的三个队伍还要多。

这看起来像是一个字典情况，因为我们有一个有意义的键（队伍名称），我们想要通过它来访问值（运动员的姓名）。然而，我们之前已经说过，字典只允许每个键一个值。考虑以下代码：

```py
players = {}
players["Team Red"] = "Shaoming"
players["Team Red"] = "Lijin"
```

> 现在行动！
> 
> > 运行此代码后字典中会有什么内容？如果你不确定，可以试一试！

我们如何将多个玩家名称存储在同一个键下？这里的见解是，我们想要与队名关联的是玩家集合，而不是单个玩家。因此，我们应该在每个键下存储玩家列表，如下所示：

```py
players = {}
players["Team Red"] = ["Shaoming", "Lijin"]
players["Team Green"] = ["Obi", "Chinara"]
players["Team Blue"] = ["Mateo", "Sophia"]
```

字典中的值不仅限于基本值。它们可以是任意复杂的，包括列表、表格，甚至是其他字典（等等！）！每个键仍然只有一个值，这是字典的要求。

#### 9.2.4 字典与数据类 "链接至此")

之前，我们学习了使用数据类（dataclasses）在 Python 中创建复合数据的方法。这里再次介绍我们之前引入的 `ToDoItem` 数据类，以及该类的示例数据：

```py
class ToDoItem:
    descr: str
    due: date
    tags: list

milk = ToDoItem("buy milk", date(2020, 7, 27), ["shopping", "home"]
```

可以将数据类中的字段名视为类似于字典中的键。如果我们这样做，我们也可以通过以下方式通过字典捕获 `milk` 数据：

```py
milk_dict = {"descr": "buy milk",
             "due": date(2020, 7, 27),
             "tags": ["shopping", "home"]
             }
```

> 现在行动！
> 
> > 创建一个字典来捕获复合数据
> > 
> > ```py
> > ToDoItem("grade hwk", date(2020, 7, 27), ["teaching"])
> > ```
> > 
> 现在行动！
> 
> > 创建一个名为 `myTD_D` 的待办事项列表，其中包含字典列表，而不是数据类列表。

将这两种方法并排放置，以下是对比：

```py
myTD_L = [ToDoItem("buy milk", date(2020, 7, 27), ["shopping", "home"]),
          ToDoItem("grade hwk", date(2020, 7, 27), ["teaching"]),
          ToDoItem("meet students", date(2020, 7, 26), ["research"])
         ]

myTD_D = [milk_dict,
          {"descr": "grade hwk",
           "due": date(2020, 7, 27),
           "tags": ["teaching"]
          },
          {"descr": "meet students",
           "due": date(2020, 7, 26),
           "tags": ["research"]
          }
         ]
```

> 现在行动！
> 
> > 你认为使用数据类和字典来表示复合数据各有什么优缺点？

数据类有固定数量的字段，而目录允许任意数量的键。数据类字段可以用类型注解（大多数语言在创建新数据时会检查这些类型）；字典可以为每个键和值使用固定类型，尽管当使用字典捕获具有不同类型字段的字典时，这会变得限制性。数据类为你提供了一个创建新数据的函数名，而使用字典时，你必须自己创建这样的函数。

总体而言，数据类在错误检查方面提供了更多的语言支持：你不能为错误数量的字段或字段值提供数据。字典更灵活：你可以更容易地支持可选字段，包括在程序运行时添加新字段/键。这些在某种编程情况下都更有意义。

> 现在行动！
> 
> > 编写一个函数 `ToDoItem_D`，它接受一个描述、截止日期和标签列表，并返回一个包含待办事项每个字段键的字典。

##### 摘要 "链接至此")

Python 程序员倾向于大量使用字典。在本章中，我们看到了字典在两种不同环境中的应用：

+   其中键唯一标识更大集合中的不同实体或个人；值代表关于每个个体的某种一致类型的信息。该字典总体上捕获了大量个人的信息，每个个体都有自己的键。

+   其中键命名了复合数据字段；与每个字段相关联的值可以与其他字段的值具有不同的类型。这种设置对应于数据类的使用，其中字典捕获有关一个人的信息；需要其他结构（如列表或另一个字典）来保存每个个人的字典。

作为一般规则，当你有一组固定的字段时，最好使用数据类。在 Python 社区中，字典的使用与数据类的使用有些关联（在其他语言中则不那么明显）。然而，第一个设置是字典在几乎所有语言中的常见用法，尤其是因为字典通常被构建为提供对特定键相关数据的快速访问。

#### 9.2.1 创建和使用字典 "链接到此处")

字典将唯一的值（称为键）映射到每个键对应的数据片段（称为值）。以下是我们将飞行示例改写为字典而不是列表的形式：

```py
sched_dict = {'CSA-342': Flight('NYC','PVD','CSA-342',50),
              'CSA-590': Flight('PVD','ORD','CSA-590',50),
              'CSA-723': Flight('NYC','ORD','CSA-723',120),
              'CSA-145': Flight('ORD','DEN','CSA-145',175),
              'CSA-647': Flight('BOS','ORD','CSA-647',80)
             }
```

字典的一般形式是：

```py
{key1: value1,
 key2: value2,
 key3: value3,
 ...}
```

字典被设计成能够根据键轻松查找值。为了获取

```py
Flight
```

与键 `'CSA-145'` 相关，我们可以简单地写：

```py
sched_dict['CSA-145']
```

要获取 `'CSA-145'` 飞行的座位数，我们可以简单地写：

```py
sched_dict['CSA-145'].seats
```

换句话说，字典数据结构消除了遍历列表以找到具有特定键的 `Flight` 的需要。字典查找操作为我们完成了这项工作。实际上，字典甚至更加精细：根据它们的设计方式，字典可以在不遍历所有值（甚至任何其他值）的情况下检索键的值。一般来说，你可以假设基于字典的查找比基于列表的查找要快得多。这是如何工作的是一个更高级的话题；其中一些内容在 [SECREF] 中有解释。

字典的一个局限性是它们只允许每个键一个值。让我们考虑一个不同的例子，这次是使用建筑中的房间作为键，居住者作为值：

```py
office_dict = {410: 'Farhan',
               411: 'Pauline',
               412: 'Marisol',
               413: 'Saleh'}
```

如果有人搬进了办公室 412，在 Python 中，我们可以这样设置该键的值：

```py
office_dict[412] = 'Zeynep'
```

现在，任何对 `office_dict[412]` 的使用都将评估为 `'Zeynep'` 而不是 `'Marisol'`。

#### 9.2.2 在字典中搜索值 "链接到此处")

如果我们想找到所有座位数超过 100 的航班，我们必须遍历所有的键值对并检查它们的余额。这听起来又像我们需要一个 for 循环。但在字典上这看起来是什么样子呢？

结果表明，它看起来就像在列表上写一个 for 循环（至少在 Python 中是这样）。以下是一个创建座位数超过 100 的航班列表的程序：

```py
above_100 = []

# the room variable takes on each key in the dictionary
for flight_code in sched_dict:
    if sched_dict[flight_code].seats > 100:
        above_100.append(sched_dict[flight_code])
```

在这里，for 循环遍历键。在循环内部，我们使用每个键来检索其对应的`Flight`，对`Flight`进行平衡检查，然后如果它符合我们的标准，就将`Flight`放入我们的运行列表中。

> 练习
> 
> > 创建一个将教室或会议室的名称映射到它们拥有的座位数的字典。编写表达式：
> > 
> > 1.  查询特定房间有多少座位
> > 1.  
> > 1.  将特定房间的容量改为比最初多 10 个座位
> > 1.  
> > 1.  找出可以容纳至少 50 名学生的所有房间

#### 9.2.3 具有更复杂值的字典 "链接到此处")

> 现在就做！
> 
> > 田径锦标赛需要管理将要参赛的各个队伍的玩家名称。例如，“红队”有“Shaoming”和“Lijin”，“绿队”包含“Obi”和“Chinara”，“蓝队”有“Mateo”和“Sophia”。想出一个组织数据的方法，使得组织者可以轻松访问每个队伍的玩家名称，同时考虑到可能比这里列出的三个队伍还要多很多队伍。

这感觉像是一个字典情况，因为我们有一个有意义的键（团队名称），我们想要通过它访问值（玩家的名称）。然而，我们之前已经说过，字典每个键只能有一个值。考虑以下代码：

```py
players = {}
players["Team Red"] = "Shaoming"
players["Team Red"] = "Lijin"
```

> 现在就做！
> 
> > 运行此代码后字典中会有什么？如果你不确定，试着试一下！

我们如何将多个玩家名称存储在同一个键下？这里的洞察力在于，我们想要与团队名称关联的是玩家集合，而不是单个玩家。因此，我们应该在每个键下存储玩家列表，如下所示：

```py
players = {}
players["Team Red"] = ["Shaoming", "Lijin"]
players["Team Green"] = ["Obi", "Chinara"]
players["Team Blue"] = ["Mateo", "Sophia"]
```

字典中的值不仅限于基本值。它们可以是任意复杂的，包括列表、表格，甚至是其他字典（等等！）！每个键仍然只有一个值，这是字典的要求。

#### 9.2.4 字典与 dataclasses 的比较 "链接到此处")

之前，我们学习了 dataclass 作为在 Python 中创建复合数据的一种方法。这里再次介绍我们之前引入的`ToDoItem` dataclass 以及该类的示例数据：

```py
class ToDoItem:
    descr: str
    due: date
    tags: list

milk = ToDoItem("buy milk", date(2020, 7, 27), ["shopping", "home"]
```

可以将 dataclass 中的字段名称视为类似于字典中的键。如果我们这样做，我们也可以通过以下方式通过字典捕获`milk`数据：

```py
milk_dict = {"descr": "buy milk",
             "due": date(2020, 7, 27),
             "tags": ["shopping", "home"]
             }
```

> 现在就做！
> 
> > 创建一个字典来捕获复合数据
> > 
> > ```py
> > ToDoItem("grade hwk", date(2020, 7, 27), ["teaching"])
> > ```
> > 
> 现在就做！
> 
> > 创建一个名为`myTD_D`的待办事项列表，其中包含字典列表，而不是 dataclasses 列表。

将这两种方法并排放置，以下是它们的对比：

```py
myTD_L = [ToDoItem("buy milk", date(2020, 7, 27), ["shopping", "home"]),
          ToDoItem("grade hwk", date(2020, 7, 27), ["teaching"]),
          ToDoItem("meet students", date(2020, 7, 26), ["research"])
         ]

myTD_D = [milk_dict,
          {"descr": "grade hwk",
           "due": date(2020, 7, 27),
           "tags": ["teaching"]
          },
          {"descr": "meet students",
           "due": date(2020, 7, 26),
           "tags": ["research"]
          }
         ]
```

> 现在就做！
> 
> > 你认为使用 dataclasses 和字典来表示复合数据有哪些优点和缺点？

数据类具有固定数量的字段，而目录允许任意数量的键。数据类的字段可以用类型进行注解（大多数语言在创建新数据时会检查类型）；字典可以为每个键和值使用固定类型，但当使用字典捕获具有不同类型字段的数据类时，这会变得限制性。数据类为你提供了一个创建新数据的函数名，而使用字典时，你必须自己创建这样的函数。

总体而言，数据类提供了更多的语言支持以进行错误检查：你不能为错误数量的字段或字段值提供数据。字典更灵活：你可以更容易地支持可选字段，包括在程序运行时添加新字段/键。这些在某种编程场景中更有意义。

> 现在行动！
> 
> > 编写一个函数`ToDoItem_D`，它接受一个描述、截止日期和标签列表，并返回一个包含待办事项每个字段键的字典。

##### 摘要 "链接至此")

Python 程序员倾向于大量使用字典。在本章中，我们看到了字典在两种不同场景下的应用：

+   一种情况是键唯一标识更大集合中的不同实体或个人；值代表关于每个个体的某种一致类型的信息。字典整体上捕捉了关于大量个体的信息，每个个体都有自己的键。

+   一种情况是键命名复合数据中的字段；与每个字段关联的值可以与其他字段的值具有不同的类型。这种设置对应于数据类的使用，其中字典捕捉了关于一个个体的信息；需要其他结构（如列表或另一个字典）来保存每个个体的字典。

作为一般规则，当你有一个固定字段集时，最好使用数据类。在 Python 社区中，字典用于数据类的使用与编程实践有关（在其他语言中则不那么相关）。然而，第一种设置是几乎所有语言中字典的常见用法，尤其是在字典通常被构建为提供快速访问与特定键关联的数据的情况下。

##### 摘要 "链接至此")

Python 程序员倾向于大量使用字典。在本章中，我们看到了字典在两种不同场景下的应用：

+   一种情况是键唯一标识更大集合中的不同实体或个人；值代表关于每个个体的某种一致类型的信息。字典整体上捕捉了关于大量个体的信息，每个个体都有自己的键。

+   其中键命名了复合数据字段；与每个字段相关联的值可以与其他字段的值具有不同的类型。这种设置对应于数据类的使用，其中字典捕获有关一个个体的信息；需要其他结构（如列表或另一个字典）来保存每个个体的字典。

作为一般规则，当你有一组固定的字段时，最好在第二个设置中使用数据类。在数据类中使用字典与 Python 社区的编程实践有一定的关联（在其他语言中则不那么明显）。然而，第一个设置是几乎所有语言中字典的常见用法，尤其是在字典通常被构建以提供对与特定键相关联的数据的快速访问的情况下。
