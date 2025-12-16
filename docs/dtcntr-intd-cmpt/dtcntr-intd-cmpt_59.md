# 18.3 联合查找🔗

> 原文：[`dcic-world.org/2025-08-27/union-find.html`](https://dcic-world.org/2025-08-27/union-find.html)

| |   18.3.1 使用状态实现) |
| --- | --- |
| |   18.3.2 优化) |
| |   18.3.3 分析) |

我们之前 [检查组件连通性)] 看过如何检查组件的连通性，但发现那个解决方案不满意。回想一下，这归结为两个集合操作：我们想要构造集合的并集，然后确定两个元素是否属于同一个集合。

我们现在将看到如何使用状态来实现这一点。我们将尽量使它与上一个版本尽可能相似，以增强比较。

#### 18.3.1 使用状态实现🔗 "链接到这里")

首先，我们必须更新元素的定义，使`parent`字段可变：

```py
data Element:
  | elt(val, ref parent :: Option<Element>)
end
```

要确定两个元素是否属于同一个集合，我们仍然会依赖`fynd`。然而，正如我们很快将看到的，`fynd`不再需要给出整个元素集合。因为`is-in-same-set`消耗那个集合的唯一原因是为了将其传递给`fynd`，所以我们可以从这里移除它。其他什么都没有改变：

```py
fun is-in-same-set(e1 :: Element, e2 :: Element) -> Boolean:
  s1 = fynd(e1)
  s2 = fynd(e2)
  identical(s1, s2)
end
```

更新现在是关键的区别：我们使用变异来改变父代的价值：

```py
fun update-set-with(child :: Element, parent :: Element):
  child!{parent: some(parent)}
end
```

在`parent: some(parent)`中，第一个`parent`是字段名称，而第二个是参数名称。此外，我们必须使用`some`来满足可选类型。自然地，它不是`none`，因为这次变异的全部目的就是将父代更改为另一个元素，而不管之前是什么。

给定这个定义，`union`也基本保持不变，除了返回类型的变化。之前，它需要返回更新后的元素集合；现在，因为更新是通过变异来执行的，所以不再需要返回任何内容：

```py
fun union(e1 :: Element, e2 :: Element):
  s1 = fynd(e1)
  s2 = fynd(e2)
  if identical(s1, s2):
    s1
  else:
    update-set-with(s1, s2)
  end
end
```

最后，是`fynd`。它的实现现在非常简单。不再需要搜索集合。之前，我们必须搜索，因为联合操作发生后，父引用可能不再有效。现在，任何此类更改都会通过变异自动反映。因此：

```py
fun fynd(e :: Element) -> Element:
  cases (Option) e!parent:
    | none => e
    | some(p) => fynd(p)
  end
end
```

#### 18.3.2 优化🔗 "链接到这里")

再次看看`fynd`。在`some`情况下，绑定到`e`的元素不是集合名称；这是通过递归遍历`parent`引用获得的。然而，当这个值返回时，我们并没有做任何事情来反映这种新的知识！相反，下次我们尝试找到这个元素的父代时，我们将再次执行相同的递归遍历。

使用变异可以帮助解决这个问题。这个想法简单到不能再简单：计算父代的价值，并更新它。

```py
fun fynd(e :: Element) -> Element:
  cases (Option) e!parent block:
    | none => e
    | some(p) =>
      new-parent = fynd(p)
      e!{parent: some(new-parent)}
      new-parent
  end
end
```

注意，这个更新将应用于寻找集合名称的递归链中的每个元素。因此，下次应用 `fynd` 到这些元素中的任何一个时，将受益于这次更新。这个想法被称为路径压缩。

我们还可以应用一个有趣的想法。这是维护每个元素的秩，这大致是元素树（其中该元素是它们的集合名称）的深度。当我们合并两个元素时，我们让秩较大的元素成为秩较小的元素的父元素。这有助于避免生成非常长的路径到集合名称元素，而是趋向于“灌木丛”状的树。这也减少了必须遍历以找到代表者的父元素数量。

#### 18.3.3 分析🔗 "链接至此")

这个优化的并查集数据结构有一个令人瞩目的分析。当然，在最坏的情况下，我们必须遍历整个父元素链来找到名称元素，这需要与集合中元素数量成比例的时间。然而，一旦我们应用上述优化，我们就再也不需要遍历相同的链了！特别是，如果我们对一系列集合等价测试进行摊销分析，在一系列并操作之后，我们发现后续检查的成本非常小——确实，几乎是一个可以得到的非常小的函数，而不需要是常数。实际的[分析](http://en.wikipedia.org/wiki/Disjoint-set_data_structure)相当复杂；它也是计算机科学中最显著的算法分析之一。这里有 Robert Tarjan 关于他分析历史的[简要演讲](https://www.youtube.com/watch?v=Hhk8ANKWGJA)。

#### 18.3.1 使用状态实现🔗 "链接至此")

首先，我们必须更新元素的定义，使 `parent` 字段可变：

```py
data Element:
  | elt(val, ref parent :: Option<Element>)
end
```

要确定两个元素是否属于同一集合，我们仍然会依赖 `fynd`。然而，正如我们很快就会看到的，`fynd` 不再需要给出整个元素集。因为 `is-in-same-set` 消耗该集合的唯一原因是为了将其传递给 `fynd`，所以我们可以从这里删除它。其他什么都没有改变：

```py
fun is-in-same-set(e1 :: Element, e2 :: Element) -> Boolean:
  s1 = fynd(e1)
  s2 = fynd(e2)
  identical(s1, s2)
end
```

更新现在是关键的区别：我们使用变异来更改父元素的值：

```py
fun update-set-with(child :: Element, parent :: Element):
  child!{parent: some(parent)}
end
```

在 `parent: some(parent)` 中，第一个 `parent` 是字段名，而第二个是参数名。此外，我们必须使用 `some` 来满足选项类型。自然地，它不是 `none`，因为整个变异的目的就是将父元素更改为其他元素，而不管之前是什么。

根据这个定义，`union` 也基本保持不变，除了返回类型的变化。之前，它需要返回更新后的元素集；现在，因为更新是通过变异来执行的，所以不再需要返回任何内容：

```py
fun union(e1 :: Element, e2 :: Element):
  s1 = fynd(e1)
  s2 = fynd(e2)
  if identical(s1, s2):
    s1
  else:
    update-set-with(s1, s2)
  end
end
```

最后，`fynd`。它的实现现在非常简单。不再需要搜索集合。以前，我们必须搜索，因为联合操作发生后，父引用可能不再有效。现在，任何此类更改都会通过变异自动反映出来。因此：

```py
fun fynd(e :: Element) -> Element:
  cases (Option) e!parent:
    | none => e
    | some(p) => fynd(p)
  end
end
```

#### 18.3.2 优化🔗 "链接至此")

再次看看 `fynd`。在 `some` 情况下，绑定到 `e` 的元素不是集合名称；这是通过递归遍历 `parent` 引用获得的。然而，当这个值返回时，我们并没有做任何事情来反映这种新的知识！相反，下次我们尝试找到这个元素的父节点时，我们将再次执行相同的递归遍历。

使用变异可以帮助解决这个问题。这个想法非常简单：计算父节点的值，并更新它。

```py
fun fynd(e :: Element) -> Element:
  cases (Option) e!parent block:
    | none => e
    | some(p) =>
      new-parent = fynd(p)
      e!{parent: some(new-parent)}
      new-parent
  end
end
```

注意，这次更新将适用于找到集合名称的递归链中的每个元素。因此，下次应用 `fynd` 到这些元素中的任何一个时，将受益于这次更新。这个想法被称为路径压缩。

我们还可以应用另一个有趣的想法。这是维护每个元素的等级，这大致是元素树（其中该元素是它们的集合名称）的深度。当我们合并两个元素时，我们让等级较高的元素成为等级较低的元素的父节点。这有助于避免非常长的路径生长到集合名称元素，而是趋向于“灌木丛”状的树。这也减少了必须遍历以找到代表者的父节点数量。

#### 18.3.3 分析🔗 "链接至此")

这个优化的并查集数据结构有一个非常出色的分析。当然，在最坏的情况下，我们必须遍历整个父节点链来找到名称元素，这需要与集合中元素数量成比例的时间。然而，一旦我们应用上述优化，我们就永远不需要再次遍历相同的链！特别是，如果我们对一系列集合等价测试进行摊销分析，这些测试是在一系列联合操作之后进行的，我们发现后续检查的成本非常小——确实，几乎是一个尽可能小的函数，而不需要是常数。实际的[分析](http://en.wikipedia.org/wiki/Disjoint-set_data_structure)非常复杂；它也是计算机科学中最出色的算法分析之一。这里有 Robert Tarjan 关于他分析历史的[简要演讲](https://www.youtube.com/watch?v=Hhk8ANKWGJA)。
