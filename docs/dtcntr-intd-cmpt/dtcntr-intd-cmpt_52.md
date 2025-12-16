# 17.2 基本图遍历

> 原文：[`dcic-world.org/2025-08-27/basic-graph-trav.html`](https://dcic-world.org/2025-08-27/basic-graph-trav.html)

| |   17.2.1 可达性 |
| --- | --- |
| |   17.2.1.1 简单递归 |
| |   17.2.1.2 清理循环 |
| |   17.2.1.3 带记忆的遍历 |
| |   17.2.1.4 更好的接口 |
| |   17.2.2 深度优先遍历和广度优先遍历 |

就像我们迄今为止看到的所有数据一样，为了处理一个数据项，我们必须遍历它——即，访问构成数据的数据。对于图来说，这可以非常有趣！

#### 17.2.1 可达性 "链接到这里")

图的许多用途都需要解决可达性问题：我们是否可以使用图中的边从节点 A 到节点 B。例如，一个社交网络可能会建议所有可以从现有联系人到达的人作为联系人。在互联网上，交通工程师关心数据包是否可以从一台机器到达另一台机器。在网络上，我们关心网站上的所有公共页面是否可以从主页到达。我们将以我们的旅行图作为运行示例来研究如何计算可达性。

##### 17.2.1.1 简单递归 "链接到这里"

在最简单的情况下，可达性是容易的。我们想知道是否存在一条路径，这条路径是一个或多个链接边的序列。从一个节点对，即源节点和目标节点之间。（可达性的更复杂版本可能会计算实际的路径，但我们现在将忽略这一点。）有两种可能性：源节点和目标节点相同，或者它们不同。

+   如果它们相同，那么显然可达性是显而易见的。

+   如果它们不是，我们必须遍历源节点的邻居并询问目的地是否可以从这些邻居中的每一个到达。

这转化为以下函数：

<graph-reach-1-main> ::=

```py
fun reach-1(src :: Key, dst :: Key, g :: Graph) -> Boolean:
  if src == dst:
    true
  else:
    <graph-reach-1-loop>
    loop(neighbors(src, g))
  end
end
```

其中，通过 `src` 的邻居的循环是：

<graph-reach-1-loop> ::=

```py
fun loop(ns):
  cases (List) ns:
    | empty => false
    | link(f, r) =>
      if reach-1(f, dst, g): true else: loop(r) end
  end
end
```

我们可以这样测试：

<graph-reach-tests> ::=

```py
check:
  reach = reach-1
  reach("was", "was", kn-cities) is true
  reach("was", "chi", kn-cities) is true
  reach("was", "bmg", kn-cities) is false
  reach("was", "hou", kn-cities) is true
  reach("was", "den", kn-cities) is true
  reach("was", "saf", kn-cities) is true
end
```

不幸的是，我们无法得知这些测试的结果，因为其中一些根本就没有完成。这是因为我们有一个无限循环，这是由于图的循环性质造成的！

> 练习
> 
> > 哪个上面的例子会导致循环？为什么？

##### 17.2.1.2 清理循环 "链接到这里")

在我们继续之前，让我们尝试改进循环的表达式。虽然上面的嵌套函数是一个完全合理的定义，但我们可以使用 Pyret 的 `for` 来提高其可读性。

上面的循环的本质是遍历一个布尔值的列表；如果其中之一为真，则整个循环评估为真；如果它们都是假的，那么我们没有找到到达目标节点的路径，所以循环评估为假。因此：

```py
fun ormap(fun-body, l):
  cases (List) l:
    | empty => false
    | link(f, r) =>
      if fun-body(f): true else: ormap(fun-body, r) end
  end
end
```

使用这种方法，我们可以替换循环定义和使用：

```py
for ormap(n from neighbors(src, g)):
  reach-1(n, dst, g)
end
```

##### 17.2.1.3 带记忆的遍历 "链接到这里")

因为我们有循环数据，我们必须记住我们已经访问过的节点，以避免再次遍历它们。然后，每次我们开始遍历一个新的节点，我们就将它添加到我们已经开始访问的节点集合中，以便。如果我们返回到该节点，因为我们假设图在此期间没有改变，我们知道从这个节点进行的额外遍历不会对结果产生任何影响。这个特性被称为☛ 幂等性。

因此，我们定义了第二次尝试可达性，它接受一个额外的参数：我们已经开始访问的节点集合（集合由图表示）。与<graph-reach-1-main>的关键区别在于，在我们开始遍历边之前，我们应该检查我们是否已经开始处理该节点。这导致了以下定义：

<graph-reach-2> ::=

```py
fun reach-2(src :: Key, dst :: Key, g :: Graph, visited :: List<Key>) -> Boolean:
  if visited.member(src):
    false
  else if src == dst:
    true
  else:
    new-visited = link(src, visited)
    for ormap(n from neighbors(src, g)):
      reach-2(n, dst, g, new-visited)
    end
  end
end
```

特别是，注意额外的新的条件：如果可达性检查之前已经访问过这个节点，就没有必要从这里进一步遍历，所以它返回`false`。（可能还有其他图的部分需要探索，其他递归调用将完成这些工作。）

> 练习
> 
> > 如果前两个条件被交换了，即`reach-2`的开始是以
> > 
> > ```py
> > if src == dst:
> >   true
> > else if visited.member(src):
> >   false
> > ```
> > 
> > ? 请具体举例说明。
> > 
> 练习
> 
> > 我们反复谈论记住我们已经开始访问的节点，而不是我们已经完成访问的节点。这种区别重要吗？为什么重要？

##### 17.2.1.4 更好的接口 "链接到这里")

正如测试`reach-2`的过程所显示的，我们可能有一个更好的实现，但我们已经改变了函数的接口；现在它有一个不必要的额外参数，这不仅是一个麻烦，如果我们不小心误用它，还可能导致错误。因此，我们应该通过将核心代码移动到内部函数来清理我们的定义：

```py
fun reach-3(s :: Key, d :: Key, g :: Graph) -> Boolean:
  fun reacher(src :: Key, dst :: Key, visited :: List<Key>) -> Boolean:
    if visited.member(src):
      false
    else if src == dst:
      true
    else:
      new-visited = link(src, visited)
      for ormap(n from neighbors(src, g)):
        reacher(n, dst, new-visited)
      end
    end
  end
  reacher(s, d, empty)
end
```

我们现在已经恢复了原始接口，并正确实现了可达性。

> 练习
> 
> > 这真的给我们提供了一个正确的实现吗？特别是，这解决了上面`size`函数解决的问题吗？创建一个测试用例来展示这个问题，然后修复它。

#### 17.2.2 深度优先和广度优先遍历 "链接到这里")

计算机科学文本中通常将这些称为深度优先和广度优先搜索。然而，搜索只是一个特定目的；遍历是一个通用任务，可以用于许多目的。

我们上面看到的可达性算法有一个特殊属性。在它访问的每个节点上，通常有一个相邻节点的集合，它可以在那里继续遍历。它至少有两个选择：它要么首先访问每个直接邻居，然后访问所有邻居的邻居；要么选择一个邻居，递归，然后在访问完成之后只访问下一个直接邻居。前者被称为广度优先遍历，而后者是深度优先遍历。

我们设计的算法使用深度优先策略：在 <graph-reach-1-loop> 内，我们在访问第二个邻居之前，先递归访问邻居列表的第一个元素，依此类推。另一种选择是有一个数据结构，我们将所有邻居插入其中，然后一次取出一个元素，这样我们首先访问所有邻居的邻居，依此类推。这自然对应于一个队列 [一个例子：从列表中得到的队列]。

> 练习
> 
> > 使用队列实现广度优先遍历。

如果我们正确检查以确保我们不重复访问节点，那么广度和深度优先遍历都将正确地遍历整个可达图，而不重复（因此不会进入无限循环）。每个遍历只从一个节点开始，从该节点考虑每一条边。因此，如果一个图有 \(N\) 个节点和 \(E\) 条边，那么遍历的复杂度下限是 \(O([N, E \rightarrow N + E])\)。我们还必须考虑检查我们是否已经访问过节点之前的花费（这是一个集合成员问题，我们将在其他地方解决：集合的几种变体）。最后，我们必须考虑维护跟踪我们遍历的数据结构的成本。在深度优先遍历的情况下，递归——<wbr>它使用机器的堆栈——<wbr>自动以恒定开销完成。在广度优先遍历的情况下，程序必须管理队列，这可能会增加超过恒定开销。在实践中，堆栈通常会比队列表现得更好，因为它是受机器硬件支持的。

这可能会表明深度优先遍历总是优于广度优先遍历。然而，广度优先遍历有一个非常重要且宝贵的属性。从一个节点 \(N\) 开始，当它访问一个节点 \(P\) 时，计算到达 \(P\) 所经过的边的数量。广度优先遍历保证不可能有更短的路径到达 \(P\)：也就是说，它找到了到达 \(P\) 的最短路径。

> 练习
> 
> > 为什么是“a”而不是“the”最短路径？
> > 
> 练习
> 
> > 证明广度优先遍历可以找到最短路径。

#### 17.2.1 可达性 "链接到此处")

许多使用图的应用都需要处理可达性：我们是否可以使用图中的边从一个节点到达另一个节点。例如，一个社交网络可能会建议所有可以从现有联系人到达的人作为联系人。在互联网上，交通工程师关心数据包是否可以从一台机器到达另一台机器。在网络上，我们关心网站上的所有公共页面是否可以从主页到达。我们将以我们的旅行图作为运行示例来研究如何计算可达性。

##### 17.2.1.1 简单递归 "链接至此")

在最简单的情况下，可达性是容易的。我们想知道是否存在一条路径，一条由零个或多个链接边组成的路径，从一个节点对，一个源节点和一个目标节点之间。 (更复杂的可达性版本可能会计算实际的路径，但我们现在将忽略这一点。)有两种可能性：源节点和目标节点相同，或者它们不同。

+   如果它们相同，那么显然可达性是显而易见的。

+   如果它们不同，我们必须遍历源节点的邻居并询问目标节点是否可以从这些邻居中的每一个到达。

这转化为以下函数：

<graph-reach-1-main> ::=

```py
fun reach-1(src :: Key, dst :: Key, g :: Graph) -> Boolean:
  if src == dst:
    true
  else:
    <graph-reach-1-loop>
    loop(neighbors(src, g))
  end
end
```

其中 `src` 的邻居遍历循环是：

<graph-reach-1-loop> ::=

```py
fun loop(ns):
  cases (List) ns:
    | empty => false
    | link(f, r) =>
      if reach-1(f, dst, g): true else: loop(r) end
  end
end
```

我们可以这样测试：

<graph-reach-tests> ::=

```py
check:
  reach = reach-1
  reach("was", "was", kn-cities) is true
  reach("was", "chi", kn-cities) is true
  reach("was", "bmg", kn-cities) is false
  reach("was", "hou", kn-cities) is true
  reach("was", "den", kn-cities) is true
  reach("was", "saf", kn-cities) is true
end
```

不幸的是，我们不知道这些测试的结果如何，因为其中一些根本就没有完成。这是因为我们有一个无限循环，这是由于图的循环性质造成的！

> 练习
> 
> > 上述哪个例子会导致循环？为什么？

##### 17.2.1.2 清理循环 "链接至此")

在我们继续之前，让我们尝试改进循环的表达式。虽然上面的嵌套函数是一个完全合理的定义，但我们可以使用 Pyret 的 `for` 来提高其可读性。

上面的循环的本质是遍历一个布尔值列表；如果其中之一为真，则整个循环评估为真；如果它们都为假，那么我们没有找到到达目标节点的路径，所以循环评估为假。因此：

```py
fun ormap(fun-body, l):
  cases (List) l:
    | empty => false
    | link(f, r) =>
      if fun-body(f): true else: ormap(fun-body, r) end
  end
end
```

使用这个，我们可以替换循环定义并使用：

```py
for ormap(n from neighbors(src, g)):
  reach-1(n, dst, g)
end
```

##### 17.2.1.3 使用记忆进行遍历 "链接至此")

由于我们具有循环数据，我们必须记住我们已经访问过的节点并避免再次遍历它们。然后，每次我们开始遍历一个新的节点时，我们将其添加到我们已经开始访问的节点集合中，以便。如果我们返回到该节点，因为我们可以假设在此期间图没有改变，我们知道从该节点进行的额外遍历不会对结果产生任何影响。这个特性被称为☛ 幂等性。

因此，我们定义了第二次尝试的可达性，它接受一个额外的参数：我们已经开始访问的节点集（其中集合由图表示）。与<graph-reach-1-main>的关键区别在于，在我们开始遍历边之前，我们应该检查我们是否已经开始处理该节点。这导致了以下定义：

<graph-reach-2> ::=

```py
fun reach-2(src :: Key, dst :: Key, g :: Graph, visited :: List<Key>) -> Boolean:
  if visited.member(src):
    false
  else if src == dst:
    true
  else:
    new-visited = link(src, visited)
    for ormap(n from neighbors(src, g)):
      reach-2(n, dst, g, new-visited)
    end
  end
end
```

特别注意额外的条件：如果可达性检查之前已经访问过这个节点，就没有必要从这里进一步遍历，因此它返回`false`。（可能还有其他图的部分需要探索，这将由其他递归调用完成。）

> 练习
> 
> > 如果前两个条件被交换了，即`reach-2`的开始是
> > 
> > ```py
> > if src == dst:
> >   true
> > else if visited.member(src):
> >   false
> > ```
> > 
> > 请用具体的例子具体说明。
> > 
> 练习
> 
> > 我们反复讨论记住我们已经开始访问的节点，而不是我们已经完成访问的节点。这种区别重要吗？如何重要？

##### 17.2.1.4 更好的接口 "链接到这里")

如测试`reach-2`的过程所示，我们可能有一个更好的实现，但我们已经改变了函数的接口；现在它有一个不必要的额外参数，这不仅是一个麻烦，如果我们不小心误用它，还可能导致错误。因此，我们应该通过将核心代码移动到内部函数来清理我们的定义：

```py
fun reach-3(s :: Key, d :: Key, g :: Graph) -> Boolean:
  fun reacher(src :: Key, dst :: Key, visited :: List<Key>) -> Boolean:
    if visited.member(src):
      false
    else if src == dst:
      true
    else:
      new-visited = link(src, visited)
      for ormap(n from neighbors(src, g)):
        reacher(n, dst, new-visited)
      end
    end
  end
  reacher(s, d, empty)
end
```

我们现在已经恢复了原始接口，同时正确实现了可达性。

> 练习
> 
> > 这真的给我们提供了一个正确的实现吗？特别是，这解决了上面`size`函数所解决的问题吗？创建一个测试用例来展示这个问题，然后修复它。

##### 17.2.1.1 简单递归 "链接到这里")

在最简单的情况下，可达性很容易。我们想知道是否存在一条路径，一条路径是由零个或多个链接边组成的序列。从一个节点对，一个源节点和一个目标节点。 (一个更复杂的可达性版本可能会计算实际的路径，但我们现在忽略这一点。)有两种可能性：源节点和目标节点是相同的，或者它们不是。

+   如果它们相同，那么显然可达性是显而易见的。

+   如果它们不相同，我们必须遍历源节点的邻居，并询问从每个邻居是否可以到达目标。

这转化为以下函数：

<graph-reach-1-main> ::=

```py
fun reach-1(src :: Key, dst :: Key, g :: Graph) -> Boolean:
  if src == dst:
    true
  else:
    <graph-reach-1-loop>
    loop(neighbors(src, g))
  end
end
```

其中通过`src`的邻居的循环是：

<graph-reach-1-loop> ::=

```py
fun loop(ns):
  cases (List) ns:
    | empty => false
    | link(f, r) =>
      if reach-1(f, dst, g): true else: loop(r) end
  end
end
```

我们可以这样测试：

<graph-reach-tests> ::=

```py
check:
  reach = reach-1
  reach("was", "was", kn-cities) is true
  reach("was", "chi", kn-cities) is true
  reach("was", "bmg", kn-cities) is false
  reach("was", "hou", kn-cities) is true
  reach("was", "den", kn-cities) is true
  reach("was", "saf", kn-cities) is true
end
```

很遗憾，我们无法得知这些测试的结果，因为其中一些根本无法完成。这是因为我们有一个无限循环，这是由于图的循环性质造成的！

> 练习
> 
> > 上述哪个示例会导致循环？为什么？

##### 17.2.1.2 清理循环 "链接到这里")

在我们继续之前，让我们尝试改进循环的表达式。虽然上面的嵌套函数是一个完全合理的定义，但我们可以使用 Pyret 的 `for` 来提高其可读性。

上面的循环的本质是遍历一个布尔值列表；如果其中之一为真，则整个循环评估为真；如果它们都是假的，那么我们没有找到到达目标节点的路径，所以循环评估为假。因此：

```py
fun ormap(fun-body, l):
  cases (List) l:
    | empty => false
    | link(f, r) =>
      if fun-body(f): true else: ormap(fun-body, r) end
  end
end
```

使用这个方法，我们可以替换循环定义和用法：

```py
for ormap(n from neighbors(src, g)):
  reach-1(n, dst, g)
end
```

##### 17.2.1.3 使用记忆进行遍历 "链接到这里")

由于我们有循环数据，我们必须记住我们已经访问过的节点，以避免再次遍历它们。然后，每次我们开始遍历一个新的节点时，我们将其添加到我们已经开始访问的节点集合中。如果我们回到那个节点，因为我们可以假设在此期间图没有改变，我们知道从这个节点进行的额外遍历不会对结果产生影响。这个特性被称为☛ 幂等性。

因此，我们定义了一个新的可达性尝试，它接受一个额外的参数：我们已经开始访问的节点集合（其中集合以图的形式表示）。与<graph-reach-1-main>的关键区别在于，在我们开始遍历边之前，我们应该检查是否已经开始处理该节点。这导致了以下定义：

<graph-reach-2> ::=

```py
fun reach-2(src :: Key, dst :: Key, g :: Graph, visited :: List<Key>) -> Boolean:
  if visited.member(src):
    false
  else if src == dst:
    true
  else:
    new-visited = link(src, visited)
    for ormap(n from neighbors(src, g)):
      reach-2(n, dst, g, new-visited)
    end
  end
end
```

特别注意额外的条件：如果可达性检查之前已经访问过这个节点，那么从这里进一步遍历就没有意义了，所以它返回`false`。（可能还有其他图的部分需要探索，这将由其他递归调用完成。）

> 练习
> 
> > 如果前两个条件被交换了，即`reach-2`的开始是
> > 
> > ```py
> > if src == dst:
> >   true
> > else if visited.member(src):
> >   false
> > ```
> > 
> > 请具体举例说明。
> > 
> 练习
> 
> > 我们反复谈论记住我们已经开始访问的节点，而不是我们已经完成访问的节点。这种区别重要吗？如何重要？

##### 17.2.1.4 更好的接口 "链接到这里")

如测试`reach-2`的过程所示，我们可能有一个更好的实现，但我们改变了函数的接口；现在它有一个不必要的额外参数，这不仅是一个麻烦，如果我们不小心误用它，还可能导致错误。因此，我们应该通过将核心代码移动到内部函数来清理我们的定义：

```py
fun reach-3(s :: Key, d :: Key, g :: Graph) -> Boolean:
  fun reacher(src :: Key, dst :: Key, visited :: List<Key>) -> Boolean:
    if visited.member(src):
      false
    else if src == dst:
      true
    else:
      new-visited = link(src, visited)
      for ormap(n from neighbors(src, g)):
        reacher(n, dst, new-visited)
      end
    end
  end
  reacher(s, d, empty)
end
```

我们现在已经恢复了原始接口，同时正确实现了可达性。

> 练习
> 
> > 这真的给了我们一个正确的实现吗？特别是，这解决了上面`size`函数解决的问题吗？创建一个测试用例来展示这个问题，然后修复它。

#### 17.2.2 深度优先和广度优先遍历 "链接到这里")

计算机科学文本中通常将这些称为深度优先搜索和广度优先搜索。然而，搜索只是特定目的；遍历是一个通用任务，可以用于许多目的。

我们上面看到的可达性算法有一个特殊性质。在它访问的每个节点上，通常有一组相邻节点，它可以在这些节点上继续遍历。它至少有两个选择：它可以先访问每个直接邻居，然后访问所有邻居的邻居；或者它可以选择一个邻居，递归，然后在访问完成之后才访问下一个直接邻居。前者被称为广度优先遍历，而后者被称为深度优先遍历。

我们设计的算法采用深度优先策略：在 <graph-reach-1-loop> 内，我们在访问第二个邻居之前，先递归访问邻居列表的第一个元素，依此类推。另一种选择是创建一个数据结构，将所有邻居插入其中，然后每次取出一个元素，这样我们首先访问所有邻居，然后再访问他们的邻居，依此类推。这自然对应于队列 [一个例子：从列表中得到的队列]。

> 练习
> 
> > 使用队列实现广度优先遍历。

如果我们正确地检查以确保我们不重复访问节点，那么广度和深度优先遍历都将正确地遍历整个可达图，而不重复（因此不会陷入无限循环）。每个遍历只从一个节点开始，考虑每一条边。因此，如果一个图有 \(N\) 个节点和 \(E\) 条边，那么遍历的复杂度下限是 \(O([N, E \rightarrow N + E])\)。我们还必须考虑检查我们是否已经访问过节点的成本（这是一个集合成员问题，我们将在其他地方解决：集合的几种变体）。最后，我们必须考虑维护跟踪我们遍历的数据结构的成本。在深度优先遍历的情况下，递归——<wbr>它使用机器的栈——<wbr>自动以恒定开销完成。在广度优先遍历的情况下，程序必须管理队列，这可能会增加超过恒定开销。在实践中，堆栈通常会比队列表现得更好，因为它是受机器硬件支持的。

这可能会让人认为深度优先遍历总是优于广度优先遍历。然而，广度优先遍历有一个非常重要的特性。从一个节点 \(N\) 开始，当它访问一个节点 \(P\) 时，计算到达 \(P\) 所经过的边的数量。广度优先遍历保证不存在到达 \(P\) 的更短路径：也就是说，它找到了到达 \(P\) 的最短路径。

> 练习
> 
> > 为什么是“a”而不是“the”最短路径？
> > 
> 练习
> 
> > 证明广度优先遍历找到最短路径。
