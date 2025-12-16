# 8.2 从列表创建队列🔗

> 原文：[`dcic-world.org/2025-08-27/queues-from-lists.html`](https://dcic-world.org/2025-08-27/queues-from-lists.html)

| 8.2.1 使用包装数据类型 |
| --- |
| 8.2.2 结合答案 |
| 8.2.3 使用选择器 |
| 8.2.4 使用元组 |
| 8.2.5 选择器方法 |

假设你有一个列表。当你取它的第一个元素时，你得到的是最近被`link`创建的元素。下一个元素是第二个最近被`link`的元素，以此类推。也就是说，最后进入的是第一个出来的。这被称为 LIFO，即“后进先出”的数据结构。列表是 LIFO；我们有时也将其称为栈。

但在许多情况下，你希望第一个进入的是第一个出来。当你站在超市的队伍中，试图购买音乐会门票，提交工作请求，或执行任何其他任务时，你希望因为你是第一个而得到奖励，而不是惩罚。也就是说，你希望有一个 FIFO。这被称为队列。

我们在这里玩的游戏是，我们想要一个数据类型，但我们的语言给了我们另一个（在这种情况下，是列表），我们必须找出如何在另一个中编码它。我们将在其他地方看到如何使用列表来编码集合 [将集合表示为列表]。在这里，让我们看看我们如何可以使用列表来编码队列。

对于集合，我们允许集合类型是列表的别名；也就是说，两者是相同的。在编码时，我们还有一个选择，即创建一个完全新的类型，它除了包装编码类型的值之外不做任何事情。我们将使用这个原则来展示这可能如何工作。

#### 8.2.1 使用包装数据类型🔗 "链接到此处")

具体来说，我们将这样表示队列。对于下面的所有代码，使用 Pyret 类型检查器来确保我们正确地组合代码是有帮助的：

```py
data Queue<T>:
  | queue(l :: List<T>)
end
```

使用这种编码，我们可以开始定义一些辅助函数：例如，构建一个空队列和检查空性的方法：

```py
fun mk-mtq<T>() -> Queue<T>:
  queue(empty)
end

fun is-mtq<T>(q :: Queue<T>) -> Boolean:
  is-empty(q.l)
end
```

向队列中添加元素通常称为“入队”。它具有以下类型：

```py
enqueue :: <T> Queue<T>, T -> Queue<T>
```

这里是对应的实现：

```py
fun enqueue(q, e):
  queue(link(e, q.l))
end
```

> 现在行动！
> 
> > 我们有选择吗？

是的，我们确实做到了！我们本可以将新元素作为第一个元素或最后一个元素。在这里要小心：我们指的是表示队列的列表的第一个或最后一个元素，而不是队列本身。在那里，FIFO（先进先出）给了我们没有选择。我们只是碰巧选择了这种表示方式。另一种表示方式同样有效；我们只需要一致地实现所有其他操作。现在我们先坚持使用这种表示方式。

现在我们遇到了一个问题。什么是“出队”的含义？我们需要获取一个元素，但同时也需要获取其余的元素。让我们首先将其写成两个函数，这两个函数与列表中的 first 和 rest 非常相似：

```py
qpeek :: <T> Queue<T> -> T
qrest :: <T> Queue<T> -> Queue<T>
```

让我们写出几个例子来确保我们知道这些应该如何工作：

```py
q_ = mk-mtq()
q3 = enqueue(q_, 3)
m43 = enqueue(q3, 4)
m543 = enqueue(m43, 5)

check:
  qpeek(q3) is 3
  qpeek(m43) is 3
  qpeek(m543) is 3
end

check:
  qrest(q3) is mk-mtq()
  qrest(m43) is enqueue(mk-mtq(), 4)
  qrest(m543) is enqueue(enqueue(mk-mtq(), 4), 5)
end
```

现在，让我们来实现这些：

```py
fun qpeek(q):
  if is-mtq(q):
    raise("can't peek an empty queue")
  else:
    q.l.get(q.l.length() - 1)
  end
end

fun qrest(q):
  fun safe-rest(l :: List<T>) -> List<T>:
    cases (List) l:
      | empty => raise("can't dequeue an empty queue")
      | link(f, r) => r
    end
  end
  queue(safe-rest(q.l.reverse()).reverse())
end
```

#### 8.2.2 结合答案🔗 "链接到这里")

然而，如果我们想同时获得最老的元素和队列的其余部分，那会很好。这意味着单个函数需要返回两个值；由于函数一次只能返回一个值，它需要使用数据结构来保存这两个值。此外，注意上面提到的`qpeek`和`qrest`都有可能没有更多的元素！我们不妨在类型中也反映这一点。因此，我们最终得到一个看起来像这样的类型

```py
data Dequeued<T>:
  | none-left
  | elt-and-q(e :: T, q :: Queue<T>)
end
```

> 练习
> 
> > 编写使用此返回类型的函数。

注意，这也遵循了我们的原则，即在返回类型中体现异常行为：选项类型，特别是在摘要中。

> 练习
> 
> > 使用这种返回类型编写函数。

#### 8.2.3 使用选择器🔗 "链接到这里")

`Dequeued`看起来熟悉吗？当然应该熟悉！它基本上与 Pyret 中用于集合的选择器相同：从集合中选择元素。如果我们让队列提供相同的操作，我们可以重用语言中已经构建的`Pick`库，并重用任何期望选择器接口编写的代码。

要这样做，首先我们需要导入选择器库：

```py
include pick
```

然后，我们可以编写：

```py
dequeue :: <T> Queue<T> -> Pick<T, Queue<T>>
```

这里有一些示例，说明它是如何工作的：

```py
check:
  dequeue(q_) is pick-none
  dequeue(q3) is pick-some(3, mk-mtq())
  dequeue(m43) is pick-some(3, enqueue(mk-mtq(), 4))
  dequeue(m543) is pick-some(3, enqueue(enqueue(mk-mtq(), 4), 5))
end
```

这里是对应的代码：

```py
fun dequeue<T>(q):
  rev = q.l.reverse()
  cases (List) rev:
    | empty => pick-none
    | link(f, r) =>
      pick-some(f, queue(r.reverse()))
  end
end
```

在大 O 复杂度方面，这是一个效率极低实现，每次在`qrest`或`dequeue`时都会导致两次反转。要了解如何做得更好，以及进行更深入的分析，请参阅示例：从列表中创建队列。

有一个需要注意的事情是，通过仅提供选择器接口，我们略微改变了队列的含义。Pyret 中的选择器接口是为集合设计的，集合没有顺序的概念。但队列当然是一个非常有序的数据类型；顺序是它们存在的原因。因此，通过仅提供选择器接口，我们并没有提供队列设计所保证的保证。因此，我们应在有序接口之外提供选择器，而不是代替它。

到目前为止，我们已经完成了基本内容，但这里还有两个你可能感兴趣的部分。

#### 8.2.4 使用元组🔗 "链接到这里")

之前，我们创建了`Dequeued`数据类型来表示从`dequeue`返回的值。实际上，创建这种类型的数据类型通常很有用，可以记录函数并确保即使它们的值在代码中远离创建位置也能有意义地解释类型。

然而，有时我们想在特殊情况下创建一个复合数据：它代表函数的返回值，而这个返回值不会存活很长时间，即，它会在返回后立即被拆分，之后只使用组成部分。在这种情况下，为这样一个短暂的目的创建一个新的数据类型可能会感觉像是一种负担。对于这种情况，Pyret 有一个内置的通用数据类型，称为元组。

这里有一些元组的示例，它们说明了它们的语法；请注意，每个位置（由`;`分隔）都接受一个表达式，而不仅仅是常量值：

```py
{1; 2}
{3; 4; 5}
{1 + 2; 3}
{6}
{}
```

我们还可以按照以下方式从元组中提取值：

```py
{a; b} = {1; 2}
```

评估`a`和`b`并查看它们绑定的是什么。

```py
{c; d; e} = {1 + 2; 6 - 2; 5}
```

类似地，查看`c`、`d`和`e`绑定的是什么。

> 练习
> 
> > 如果我们使用太少或太多的变量会发生什么？在 Pyret 中尝试以下操作并查看结果：
> > 
> > ```py
> > {p; q} = {1}
> > {p} = {1; 2}
> > {p} = 1
> > ```
> > 
> 立即行动！
> 
> > 如果我们写成这样会怎样？
> > 
> > ```py
> > p = {1; 2}
> > ```

这将`p`绑定到整个元组上。

> 练习
> 
> > 我们如何拆分`p`的组成部分？

现在我们有了元组，我们可以这样编写 dequeue：

```py
fun dequeue-tuple<T>(q :: Queue<T>) -> {T; Queue<T>}:
  rev = q.l.reverse()
  cases (List) rev:
    | empty => raise("can't dequeue an empty queue")
    | link(f, r) =>
      {f; queue(r.reverse())}
  end
end

check:
  dequeue-tuple(q3) is {3; mk-mtq()}
  dequeue-tuple(m43) is {3; enqueue(mk-mtq(), 4)}
  dequeue-tuple(m543) is {3; enqueue(enqueue(mk-mtq(), 4), 5)}
end
```

我们可以这样更普遍地使用它：

```py
fun q2l<T>(q :: Queue<T>) -> List<T>:
  if is-mtq(q):
    empty
  else:
    {e; rq} = dequeue-tuple(q)
    link(e, q2l(rq))
  end
end

check:
  q2l(mk-mtq()) is empty
  q2l(q3) is [list: 3]
  q2l(m43) is [list: 3, 4]
  q2l(m543) is [list: 3, 4, 5]
end
```

你可以在程序中使用元组，只要你遵循上述规则，确保元组适用。一般来说，元组可能会导致可读性降低，并增加出错的可能性（因为来自一个来源的元组与来自另一个来源的元组无法区分）。请谨慎使用！

#### 8.2.5 选择器方法🔗 "链接到此处")

第二点，这实际上是可选的：你可能已经注意到，`Set`有一个内置的`pick`方法。我们有一个选择函数，但没有选择方法。现在我们将看到我们如何将其编写为方法：

```py
data Queue<T>:
  | queue(l :: List<T>) with:
    method pick(self):
      rev = self.l.reverse()
      cases (List) rev:
        | empty => pick-none
        | link(f, r) =>
          pick-some(f, queue(r.reverse()))
      end
    end
end
```

这是我们之前对`Queue`定义的替代品，因为我们增加了一个方法，但保留了通用的数据类型结构，所以所有现有的代码仍然有效。此外，我们可以用 picker 接口来重写`q2l`：

```py
fun q2lm<T>(c :: Queue<T>) -> List<T>:
  cases (Pick) c.pick():
    | pick-none => empty
    | pick-some(e, r) => link(e, q2lm(r))
  end
end

check:
  q2lm(m543)
end
```

我们还可以在支持 Pick 接口的数据上编写通用程序。例如，这里有一个函数，它可以将满足该接口的任何内容转换为列表：

```py
fun pick2l<T>(c) -> List<T>:
  cases (Pick) c.pick():
    | pick-none => empty
    | pick-some(e, r) => link(e, pick2l(r))
  end
end
```

例如，它适用于集合和我们的新`Queue`：

```py
import sets as S    # put this at the top of the file

check:
  pick2l([S.set: 3, 4, 5]).sort() is [list: 3, 4, 5]
  pick2l(m543) is [list: 3, 4, 5]
end
```

> 练习
> 
> > 你看懂为什么在上述测试中我们调用了`sort`了吗？

这里唯一的弱点是，对于最后一部分（使函数通用），我们必须退出类型检查器，因为`pick2l`不能由当前的 Pyret 类型检查器进行类型检查。它需要一个类型检查器目前还没有的功能。

#### 8.2.1 使用包装数据类型🔗 "链接到此处")

具体来说，我们将这样表示队列。对于以下所有代码，使用 Pyret 类型检查器来确保我们正确地编写代码是有帮助的：

```py
data Queue<T>:
  | queue(l :: List<T>)
end
```

使用这种编码，我们可以开始定义一些辅助函数：例如，构建一个空队列和检查空性的方法：

```py
fun mk-mtq<T>() -> Queue<T>:
  queue(empty)
end

fun is-mtq<T>(q :: Queue<T>) -> Boolean:
  is-empty(q.l)
end
```

向队列中添加元素通常称为“入队”。它具有以下类型：

```py
enqueue :: <T> Queue<T>, T -> Queue<T>
```

这里是对应的实现：

```py
fun enqueue(q, e):
  queue(link(e, q.l))
end
```

> 现在行动起来！
> 
> > 我们有选择吗？

是的，我们做到了！我们可以将新元素作为第一个元素或最后一个元素。在这里要小心：我们指的是表示队列的列表的第一个或最后一个元素，而不是队列本身。在那里，FIFO 没有给我们选择。我们只是碰巧选择了这种表示。另一种表示同样有效；我们只需要一致地实现所有其他操作。现在让我们坚持使用这种表示。

现在我们遇到了一个问题。什么是“出队”的含义？我们需要获取一个元素，但还需要获取其余部分。让我们首先将其编写为两个函数，非常类似于列表上的 first 和 rest：

```py
qpeek :: <T> Queue<T> -> T
qrest :: <T> Queue<T> -> Queue<T>
```

让我们编写一些示例，以确保我们知道这些应该如何工作：

```py
q_ = mk-mtq()
q3 = enqueue(q_, 3)
m43 = enqueue(q3, 4)
m543 = enqueue(m43, 5)

check:
  qpeek(q3) is 3
  qpeek(m43) is 3
  qpeek(m543) is 3
end

check:
  qrest(q3) is mk-mtq()
  qrest(m43) is enqueue(mk-mtq(), 4)
  qrest(m543) is enqueue(enqueue(mk-mtq(), 4), 5)
end
```

现在让我们实现这些：

```py
fun qpeek(q):
  if is-mtq(q):
    raise("can't peek an empty queue")
  else:
    q.l.get(q.l.length() - 1)
  end
end

fun qrest(q):
  fun safe-rest(l :: List<T>) -> List<T>:
    cases (List) l:
      | empty => raise("can't dequeue an empty queue")
      | link(f, r) => r
    end
  end
  queue(safe-rest(q.l.reverse()).reverse())
end
```

#### 8.2.2 结合答案🔗 "链接至此")

然而，如果我们想同时获得最老的元素和队列的其余部分，那会更好。这意味着单个函数需要返回两个值；由于函数一次只能返回一个值，它需要使用一个数据结构来保存这两个值。此外，请注意，上面的`qpeek`和`qrest`都有可能没有更多的元素！我们最好也在类型中反映这一点。因此，我们最终得到一个看起来像这样的类型

```py
data Dequeued<T>:
  | none-left
  | elt-and-q(e :: T, q :: Queue<T>)
end
```

> 练习
> 
> > 编写使用这种返回类型的函数。

注意，这也遵循了我们的原则，即在返回类型中体现异常行为：选项类型，尤其是在总结中。

> 练习
> 
> > 使用这种返回类型编写函数。

#### 8.2.3 使用选择器🔗 "链接至此")

`Dequeued`看起来熟悉吗？当然应该熟悉！它基本上与 Pyret 中用于集合的选择器相同：从集合中选取元素。如果我们让队列提供相同的操作，我们可以重用语言中已经构建的`Pick`库，并重用任何期望选择器接口编写的代码。

要做到这一点，首先我们需要导入选择器库：

```py
include pick
```

然后，我们可以编写：

```py
dequeue :: <T> Queue<T> -> Pick<T, Queue<T>>
```

这里有一些示例，展示了它是如何工作的：

```py
check:
  dequeue(q_) is pick-none
  dequeue(q3) is pick-some(3, mk-mtq())
  dequeue(m43) is pick-some(3, enqueue(mk-mtq(), 4))
  dequeue(m543) is pick-some(3, enqueue(enqueue(mk-mtq(), 4), 5))
end
```

下面是对应的代码：

```py
fun dequeue<T>(q):
  rev = q.l.reverse()
  cases (List) rev:
    | empty => pick-none
    | link(f, r) =>
      pick-some(f, queue(r.reverse()))
  end
end
```

在大 O 复杂度方面，这是一个效率极低实现，每次`qrest`或`dequeue`都会导致两次反转。要了解如何做得更好，以及进行更复杂分析，请参阅示例：从列表中构建队列。

有一个需要注意的事情是，通过只提供选择器界面，我们略微改变了队列的含义。Pyret 中的选择器界面是为集合设计的，集合没有顺序的概念。但队列当然是一个非常有序的数据类型；顺序是它们存在的原因。所以，通过只提供选择器界面，我们没有提供队列设计时所保证的保证。因此，我们应在有序界面之外提供选择器，而不是取代它。

到这一点，我们已经完成了基本内容，但这里还有两个你可能感兴趣的部分。

#### 8.2.4 使用元组🔗 "链接至此")

之前，我们创建了`Dequeued`数据类型来表示出队操作的返回值。确实，创建这种类型的数据类型通常很有用，可以记录函数并确保即使它们的值在代码中远离创建位置也能有意义地解释类型。

然而，有时在特殊情况下，我们想要创建一个复合数据：它代表函数的返回值，而这个返回值不会存活很长时间，即，它会在返回后立即被拆分，之后只使用其组成部分。在这种情况下，为这样一个短暂的目的创建一个新的数据类型可能会感觉像是一种负担。对于这种情况，Pyret 有一个内置的通用数据类型，称为元组。

这里有一些元组的例子，它们说明了它们的语法；请注意，每个位置（由`;`分隔）都包含一个表达式，而不仅仅是常量值：

```py
{1; 2}
{3; 4; 5}
{1 + 2; 3}
{6}
{}
```

我们还可以这样从元组中提取值：

```py
{a; b} = {1; 2}
```

评估 `a` 和 `b` 并看看它们被绑定到了什么。

```py
{c; d; e} = {1 + 2; 6 - 2; 5}
```

类似地，看看`c`、`d`和`e`被绑定到了什么。

> 练习
> 
> > 如果我们使用太少或太多的变量会发生什么？在 Pyret 中尝试以下操作并看看会发生什么：
> > 
> > ```py
> > {p; q} = {1}
> > {p} = {1; 2}
> > {p} = 1
> > ```
> > 
> 现在就做！
> 
> > 如果我们换成写这个会怎样呢？
> > 
> > ```py
> > p = {1; 2}
> > ```

这将`p`绑定到了整个元组。

> 练习
> 
> > 我们如何拆分`p`的组成部分？

现在我们有了元组，我们可以这样写出队操作：

```py
fun dequeue-tuple<T>(q :: Queue<T>) -> {T; Queue<T>}:
  rev = q.l.reverse()
  cases (List) rev:
    | empty => raise("can't dequeue an empty queue")
    | link(f, r) =>
      {f; queue(r.reverse())}
  end
end

check:
  dequeue-tuple(q3) is {3; mk-mtq()}
  dequeue-tuple(m43) is {3; enqueue(mk-mtq(), 4)}
  dequeue-tuple(m543) is {3; enqueue(enqueue(mk-mtq(), 4), 5)}
end
```

这就是我们可以更普遍地使用它的方法：

```py
fun q2l<T>(q :: Queue<T>) -> List<T>:
  if is-mtq(q):
    empty
  else:
    {e; rq} = dequeue-tuple(q)
    link(e, q2l(rq))
  end
end

check:
  q2l(mk-mtq()) is empty
  q2l(q3) is [list: 3]
  q2l(m43) is [list: 3, 4]
  q2l(m543) is [list: 3, 4, 5]
end
```

只要遵循上述适用于元组的情况的规则，你就可以在程序中使用元组。一般来说，元组可能会降低可读性，并增加出错的可能性（因为来自一个源头的元组与来自另一个源头的元组是无法区分的）。使用时要谨慎！

#### 8.2.5 选择器方法🔗 "链接至此")

第二点，这实际上是可选的：你可能已经注意到了，`Set`有一个内置的`pick`方法。我们有一个选择函数，但没有选择方法。现在我们将看到如何将其编写为方法：

```py
data Queue<T>:
  | queue(l :: List<T>) with:
    method pick(self):
      rev = self.l.reverse()
      cases (List) rev:
        | empty => pick-none
        | link(f, r) =>
          pick-some(f, queue(r.reverse()))
      end
    end
end
```

这是我们之前定义的`Queue`的替代品，因为我们增加了一个方法，但保留了通用数据类型结构，所以所有现有的代码仍然可以工作。此外，我们可以用选择器界面重写`q2l`：

```py
fun q2lm<T>(c :: Queue<T>) -> List<T>:
  cases (Pick) c.pick():
    | pick-none => empty
    | pick-some(e, r) => link(e, q2lm(r))
  end
end

check:
  q2lm(m543)
end
```

我们还可以编写支持 Pick 接口的数据的泛型程序。例如，这里有一个函数可以将满足该接口的任何内容转换为列表：

```py
fun pick2l<T>(c) -> List<T>:
  cases (Pick) c.pick():
    | pick-none => empty
    | pick-some(e, r) => link(e, pick2l(r))
  end
end
```

例如，它适用于集合和我们的新`Queue`：

```py
import sets as S    # put this at the top of the file

check:
  pick2l([S.set: 3, 4, 5]).sort() is [list: 3, 4, 5]
  pick2l(m543) is [list: 3, 4, 5]
end
```

> 练习
> 
> > 你看到为什么我们在上面的测试中调用了`sort`了吗？

这里的唯一弱点是，对于最后一部分（使函数通用），我们必须脱离类型检查器，因为`pick2l`无法通过当前的 Pyret 类型检查器进行类型化。它需要一个类型检查器目前还没有的功能。
