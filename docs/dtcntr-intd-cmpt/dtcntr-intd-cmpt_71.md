# 26 循环的解构🔗

> 原文：[`dcic-world.org/2025-08-27/deconstructing-loops.html`](https://dcic-world.org/2025-08-27/deconstructing-loops.html)

| |   26.1 设置：两个函数 |
| --- | --- |
| |   26.2 抽象化循环 |
| |   26.3 它真的是一个循环吗？ |
| |   26.4 重新审视`for` |
| |   26.5 重写 Pollard-Rho |
| |   26.6 嵌套循环 |
| |   26.7 循环、值和定制 |

### 26.1 设置：两个函数🔗 "链接到此处")

让我们看看我们在因数分解数字中写过的两个函数：

```py
fun gcd(a, b):
  if b == 0:
    a
  else:
    gcd(b, num-modulo(a, b))
  end
end

fun pr(n):
  fun g(x): num-modulo((x * x) + 1, n) end
  fun iter(x, y, d):
    new-x = g(x)
    new-y = g(g(y))
    new-d = gcd(num-abs(new-x - new-y), n)
    ask:
      | new-d == 1 then:
        iter(new-x, new-y, new-d)
      | new-d == n then:
        none
      | otherwise:
        some(new-d)
    end
  end
  iter(2, 2, 1)
end
```

我们都通过递归来编写：`gcd`通过调用自身，`pr`通过对其内部函数的递归。但如果你之前编程过，你可能已经用循环编写过类似的程序。

> 练习
> 
> > 因为在 Pyret 中没有循环，我们能做的最好的事情就是使用一个高阶函数；你们会使用哪些？

但让我们看看我们是否可以做到“更好”，即更接近传统程序的外观。

在我们开始更改任何代码之前，让我们确保我们对`gcd`有一些测试：

```py
check:
  gcd(4, 5) is 1
  gcd(5, 7) is 1
  gcd(21, 21) is 21
  gcd(12, 24) is 12
  gcd(12, 9) is 3
end
```

### 26.2 抽象化循环🔗 "链接到此处")

现在我们来考虑如何创建一个循环。在每次迭代中，循环都有一个状态：它是否完成，或者它是否应该继续。由于我们这里有两个参数，让我们记录两个参数以继续：

```py
data LoopStatus:
  | done(final-value)
  | next-2(new-arg-1, new-arg-2)
end
```

现在我们可以编写一个执行实际迭代的函数：

```py
fun loop-2(f, arg-1, arg-2):
  r = f(arg-1, arg-2)
  cases (LoopStatus) r:
    | done(v) => v
    | next-2(new-arg-1, new-arg-2) => loop-2(f, new-arg-1, new-arg-2)
  end
end
```

注意，这完全是通用的：它与`gcd`无关。（它以与高阶函数如`map`和`filter`相同的方式是通用的。）它只是如果`f`指示重复就重复，如果`f`指示停止就停止。这就是循环的本质。

> 练习
> 
> > 注意，如果我们愿意，我们也可以将[Staging] `loop-2`分阶段进行，因为`f`永远不会改变。以这种方式重写它。

使用`loop-2`，我们可以重写`gcd`：

```py
fun gcd(p, q):
  loop-2(
    {(a, b):
      if b == 0:
        done(a)
      else:
        next-2(b, num-modulo(a, b))
      end},
    p,
    q)
end
```

现在你可能觉得我们一点有用的东西都没做。实际上，这看起来像是一个重大的倒退。至少在我们之前，我们只有简单、干净的递归，这正是欧几里得所期望的。现在我们有一个高阶函数，我们把它作为函数传递了原来的`gcd`代码，还有一个`LoopStatus`数据类型，……一切变得复杂多了。

但，实际上并不是这样。我们之所以以这种形式编写它，是因为我们即将利用 Pyret 的一个特性。Pyret 中的`for`构造实际上被重写为以下形式：

```py
for F(a from a_i, b from b_i, …): BODY end
```

被重写为

```py
F({(a, b, …): BODY}, a_i, b_i, …)
```

例如，如果我们写

```py
for map(i from range(0, 10)): i + 1 end
```

这变成

```py
map({(i): i + 1}, range(0, 10))
```

现在你可能明白为什么我们重写了`gcd`。从后往前看，我们可以重写

```py
F({(a, b, …): BODY}, a_i, b_i, …)
```

然后

```py
for F(a from a_i, b from b_i, …): BODY end
```

那么函数就变成了

```py
fun gcd(p, q):
  for loop-2(a from p, b from q):
    if b == 0:
      done(a)
    else:
      next-2(b, num-modulo(a, b))
    end
  end
end
```

现在它非常接近传统的“循环”程序。

### 26.3 它真的是一个循环吗？🔗 "链接至此")

整个这一节应该被视为对具有更高级计算知识的人的旁白。

如果你了解一些语言实现的知识，你可能知道循环具有迭代不消耗额外空间（除程序本身需要的空间外）的性质，并且重复执行非常快（一个“跳转指令”）。原则上，我们的 `loop-2` 函数没有这个特性：每个迭代都是一个函数调用，这更昂贵，并构建了额外的栈上下文。然而，实际上这些并不一定发生。

在空间方面，对 `loop-2` 的递归调用是 `loop-2` 调用的最后一件事。此外，`loop-2` 中的任何内容都不会消耗和操作那个递归调用的返回值。因此，这被称为尾递归。Pyret—<wbr>像一些其他语言—<wbr>导致尾递归不占用任何额外的栈空间。原则上，Pyret 也可以将一些尾递归转换为跳转。因此，这个版本的性能几乎与传统循环相同。

### 26.4 重新审视 `for`🔗 "链接至此")

上文给出的 `for` 的定义应该让你怀疑：循环在哪里？！？事实上，Pyret 的 `for` 完全不做任何循环：它只是 `lam` 的一个花哨的写法。任何“循环”行为都在 `for` 后面的函数中。为了证明这一点，让我们使用一个非循环函数来使用 `for`。

回想一下

```py
for F(a from a_i, b from b_i, …): BODY end
```

被重写为

```py
F({(a, b, …): BODY}, a_i, b_i, …)
```

因此，假设我们有一个这样的函数（来自 函数作为数据）：

```py
delta-x = 0.0001
fun d-dx-at(f, x):
  (f(x + delta-x) - f(x)) / delta-x
end
```

我们可以这样称呼它以得到大约 20：

```py
d-dx-at({(n): n * n}, 10)
```

这意味着我们也可以这样称呼它：

```py
for d-dx-at(n from 10): n * n end
```

事实上：

```py
check:
  for d-dx-at(n from 10): n * n end
  is
  d-dx-at({(n): n * n}, 10)
end
```

由于 `d-dx-at` 没有迭代行为，没有发生迭代。循环行为完全由 `for` 后指定的函数给出，例如 `map`、`filter` 或上面的 `loop-2`。

### 26.5 重新编写 Pollard-Rho🔗 "链接至此")

现在让我们来处理 Pollard-rho。注意，它是一个三参数函数，所以我们不能使用之前有的 `loop-2`：当每个迭代有两个参数改变时（通常是迭代变量和累加器），这个循环才适用。我们可以很容易地设计一个 3-参数版本的循环，比如 `loop-3`，但我们也可以有一个更通用的解决方案，使用一个元组：

```py
data LoopStatus:
  | done(v)
  | next–2(new-x, new-y)
  | next-n(new-t)
end

fun loop-n(f, t):
  r = f(t)
  cases (LoopStatus) r:
    | done(v) => v
    | next-n(new-t) => loop-n(f, new-t)
  end
end
```

其中 `t` 是一个元组。

因此，现在我们可以重写 `pr`。首先，我们将旧的 `pr` 函数重命名为 `pr-old`，这样我们就可以保留它以供测试。现在我们可以定义一个基于“循环”的 `pr`：

```py
fun pr(n):
  fun g(x): num-modulo((x * x) + 1, n) end
  for loop-n({x; y; d} from {2; 2; 1}):
    new-x = g(x)
    new-y = g(g(y))
    new-d = gcd(num-abs(new-x - new-y), n)
    ask:
      | new-d == 1 then:
        next-n({new-x; new-y; new-d})
      | new-d == n then:
        done(none)
      | otherwise:
        done(some(new-d))
    end
  end
end
```

事实上，我们可以测试这两个函数的行为完全相同：

```py
check:
  ns = range(2, 100)
  l1 = map(pr-old, ns)
  l2 = map(pr, ns)
  l1 is l2
end
```

### 26.6 嵌套循环🔗 "链接至此")

我们也可以这样写一个嵌套循环。假设我们有一个列表如下

```py
lol = [list: [list: 1, 2], [list: 3], [list:], [list: 4, 5, 6]]
```

我们想通过求和每个子列表来求和整个列表。这里就是：

```py
for loop-2(ll from lol, sum from 0):
  cases (List) ll:
    | empty => done(sum)
    | link(l, rl) =>
      l-sum =
        for loop-2(es from l, sub-sum from 0):
          cases (List) es:
            | empty => done(sub-sum)
            | link(e, r) => next-2(r, e + sub-sum)
          end
        end
      next-2(rl, sum + l-sum)
  end
end
```

我们可以通过将其写成两个函数来简化它：

```py
fun sum-a-lon(lon :: List<Number>):
  for loop-2(es from lon, sum from 0):
    cases (List) es:
      | empty => done(sum)
      | link(e, r) =>
        next-2(r, e + sum)
    end
  end
end

fun sum-a-lolon(lolon :: List<List<Number>>):
  for loop-2(l from lolon, sum from 0):
    cases (List) l:
      | empty => done(sum)
      | link(lon, r) =>
        next-2(r, sum-a-lon(lon) + sum)
    end
  end
end

check:
  sum-a-lolon(lol) is 21
end
```

注意到这两个函数非常相似。这表明一个抽象：

```py
fun sum-a-list(f, L):
  for loop-2(e from L, sum from 0):
    cases (List) e:
      | empty => done(sum)
      | link(elt, r) =>
        next-2(r, f(elt) + sum)
    end
  end
end
```

使用这个，我们可以将前面两个函数重写为：

```py
fun sum-a-lon(lon :: List<Number>):
  sum-a-list({(e): e}, lon)
end

fun sum-a-lolon(lolon :: List<List<Number>>):
  sum-a-list(sum-a-lon, lolon)
end

check:
  sum-a-lolon(lol) is 21
end
```

通过注释，可以清楚地了解每个函数的作用。在`sum-a-lon`中，每个元素都是一个数字，所以它“贡献了自己”到总和中。在`sum-a-lolon`中，每个元素是一个数字列表，所以它“贡献了其`sum-a-lon`”到总和中。

最后，为了使这个循环完整，我们可以将上述函数重写如下：

```py
fun sum-a-lon(lon :: List<Number>):
  for sum-a-list(e :: Number from lon): e end
end

fun sum-a-lolon(lolon :: List<List<Number>>):
  for sum-a-list(l :: List<Number> from lolon): sum-a-lon(l) end
end
```

这可能使每个元素贡献的更清晰。在`sum-a-lon`中，每个元素是一个数字，所以它只贡献那个数字。在`sum-a-lolon`中，每个元素是一个数字列表，所以它必须贡献该列表的`sum-a-lon`。

### 26.7 循环、值和定制🔗 "链接到这里")

观察上述循环与传统循环在以下两个重要方面的不同：

1.  每个循环都产生一个值。这与语言的其余部分一致，其中——尽可能——计算试图产生答案。我们不必产生一个值；例如，以下程序，类似于许多其他语言中的循环程序，在 Pyret 中运行得很好：

    ```py
    for each(i from range(0, 10)): print(i) end
    ```

    然而，这是一个不寻常的情况。通常，我们希望表达式产生值，以便我们可以将它们组合在一起。

1.  许多语言对应该有多少个循环结构有强烈的看法：两个？三个？四个？在 Pyret 中，根本没有内置的循环结构；只有一个语法（`for`），它作为创建特定`lam`的代理。有了它，我们可以重用现有的迭代函数（如`map`和`filter`），也可以定义新的函数。有些可以非常通用，如`loop-2`或`loop-n`，但有些可以非常具体，如`sum-a-list`。语言设计者不会阻止你编写对你有用的循环，有时循环可以非常具有表现力，正如我们从重写`sum-a-lon`和`sum-a-lolon`在`for`和`sum-a-list`之上所看到的那样。

### 26.1 设置：两个函数🔗 "链接到这里")

让我们看看我们在因式分解数字中编写的两个函数：

```py
fun gcd(a, b):
  if b == 0:
    a
  else:
    gcd(b, num-modulo(a, b))
  end
end

fun pr(n):
  fun g(x): num-modulo((x * x) + 1, n) end
  fun iter(x, y, d):
    new-x = g(x)
    new-y = g(g(y))
    new-d = gcd(num-abs(new-x - new-y), n)
    ask:
      | new-d == 1 then:
        iter(new-x, new-y, new-d)
      | new-d == n then:
        none
      | otherwise:
        some(new-d)
    end
  end
  iter(2, 2, 1)
end
```

我们已经递归地编写了两个函数：`gcd`通过调用自身，`pr`通过对其内部函数的递归。但如果你之前编程过，你可能已经用循环编写过类似的程序。

> 练习
> 
> > 由于 Pyret 中没有循环，我们能做的最好的是使用一个高阶函数；你会使用哪些？

但让我们看看我们是否可以做一些“更好的”，即更接近传统外观的程序。

在我们开始更改任何代码之前，让我们确保我们有针对`gcd`的测试：

```py
check:
  gcd(4, 5) is 1
  gcd(5, 7) is 1
  gcd(21, 21) is 21
  gcd(12, 24) is 12
  gcd(12, 9) is 3
end
```

### 26.2 抽象循环🔗 "链接到这里")

现在让我们思考一下如何创建一个循环。在每次迭代中，循环都有一个状态：是否完成或者是否应该继续。由于我们这里有两个参数，让我们记录两个参数以继续：

```py
data LoopStatus:
  | done(final-value)
  | next-2(new-arg-1, new-arg-2)
end
```

现在我们可以编写一个执行实际迭代的函数：

```py
fun loop-2(f, arg-1, arg-2):
  r = f(arg-1, arg-2)
  cases (LoopStatus) r:
    | done(v) => v
    | next-2(new-arg-1, new-arg-2) => loop-2(f, new-arg-1, new-arg-2)
  end
end
```

注意，这完全是通用的：它与 `gcd` 没有关系。（它以与 `map` 和 `filter` 等高阶函数相同的方式是通用的。）它只是如果 `f` 指示重复就重复，如果 `f` 指示停止就停止。这就是循环的本质。

> 练习
> 
> > 注意，如果我们愿意，可以安排 [Staging] `loop-2`，因为 `f` 从未改变。以这种方式重写它。

使用 `loop-2`，我们可以重新编写 `gcd`：

```py
fun gcd(p, q):
  loop-2(
    {(a, b):
      if b == 0:
        done(a)
      else:
        next-2(b, num-modulo(a, b))
      end},
    p,
    q)
end
```

现在你可能觉得我们根本没做任何有用的事情。事实上，这看起来像是一个重大的倒退。至少在我们之前，我们只有简单、干净的递归，这正是欧几里得所期望的。现在我们有一个高阶函数，我们将其作为函数传递了之前的 `gcd` 代码，还有一个 `LoopStatus` 数据类型，……一切变得更加复杂。

但，并非如此。我们之所以以这种形式呈现，是因为我们即将利用 Pyret 的一个特性。Pyret 中的 `for` 构造实际上被重写为以下形式：

```py
for F(a from a_i, b from b_i, …): BODY end
```

被重新编写为

```py
F({(a, b, …): BODY}, a_i, b_i, …)
```

例如，如果我们写

```py
for map(i from range(0, 10)): i + 1 end
```

这变成了

```py
map({(i): i + 1}, range(0, 10))
```

现在你可能明白我们为什么要重新编写 `gcd`。从后往前，我们可以这样重新编写

```py
F({(a, b, …): BODY}, a_i, b_i, …)
```

作为

```py
for F(a from a_i, b from b_i, …): BODY end
```

因此，函数就变成了这样

```py
fun gcd(p, q):
  for loop-2(a from p, b from q):
    if b == 0:
      done(a)
    else:
      next-2(b, num-modulo(a, b))
    end
  end
end
```

现在它非常类似于传统的“循环”程序。

### 26.3 真的是循环吗？🔗 "链接到此处")

整个这一部分应被视为对具有更高级计算知识的人的旁白。

如果你了解一些语言实现的知识，你可能知道循环具有这样的属性：迭代不会消耗额外的空间（除了程序已经需要的空间），重复进行得非常快（一个“跳转指令”）。原则上，我们的 `loop-2` 函数不具有这个属性：每次迭代都是一个函数调用，这更昂贵，并且构建了额外的堆栈上下文。然而，实际上这些并没有真正发生。

在空间方面，对 `loop-2` 的递归调用是 `loop-2` 调用的最后一件事。此外，`loop-2` 中没有任何东西消耗并操作那个递归调用的返回值。因此，这被称为尾调用。Pyret（就像一些其他语言一样）导致尾调用不占用任何额外的堆栈空间。原则上，Pyret 也可以将一些尾调用转换为跳转。因此，这个版本的性能几乎与传统循环相同。

### 26.4 重新审视 `for`🔗 "链接到此处")

上文给出的 `for` 的定义可能会让你怀疑：循环在哪里？！？事实上，Pyret 的 `for` 实际上根本不做任何循环：它只是以更花哨的方式编写 `lam`。任何“循环”行为都在 `for` 后面编写的函数中。为了证明这一点，让我们使用一个非循环函数来使用 `for`。

回想一下

```py
for F(a from a_i, b from b_i, …): BODY end
```

被重新编写为

```py
F({(a, b, …): BODY}, a_i, b_i, …)
```

因此，假设我们有一个这样的函数（来自 函数作为数据）：

```py
delta-x = 0.0001
fun d-dx-at(f, x):
  (f(x + delta-x) - f(x)) / delta-x
end
```

我们可以这样调用它来得到大约 20：

```py
d-dx-at({(n): n * n}, 10)
```

这意味着我们也可以这样调用它：

```py
for d-dx-at(n from 10): n * n end
```

事实上：

```py
check:
  for d-dx-at(n from 10): n * n end
  is
  d-dx-at({(n): n * n}, 10)
end
```

由于 `d-dx-at` 没有迭代行为，因此没有发生迭代。循环行为完全由 `for` 后指定的函数给出，例如上面的 `map`、`filter` 或 `loop-2`。

### 26.5 重写 Pollard-Rho🔗 "链接到此处")

现在，让我们来处理 Pollard-rho。注意，它是一个三参数函数，所以我们不能使用之前使用的 `loop-2`：当每个迭代中都有两个参数变化时（通常是迭代变量和累加器），这个循环才适用。我们可以轻松设计一个 3 参数版本的循环，比如 `loop-3`，但我们也可以有一个更通用的解决方案，使用元组：

```py
data LoopStatus:
  | done(v)
  | next–2(new-x, new-y)
  | next-n(new-t)
end

fun loop-n(f, t):
  r = f(t)
  cases (LoopStatus) r:
    | done(v) => v
    | next-n(new-t) => loop-n(f, new-t)
  end
end
```

其中 `t` 是一个元组。

因此现在我们可以重写 `pr`。首先，我们将旧的 `pr` 函数重命名为 `pr-old`，这样我们就可以保留它以便于测试。现在我们可以定义一个基于“循环”的 `pr`：

```py
fun pr(n):
  fun g(x): num-modulo((x * x) + 1, n) end
  for loop-n({x; y; d} from {2; 2; 1}):
    new-x = g(x)
    new-y = g(g(y))
    new-d = gcd(num-abs(new-x - new-y), n)
    ask:
      | new-d == 1 then:
        next-n({new-x; new-y; new-d})
      | new-d == n then:
        done(none)
      | otherwise:
        done(some(new-d))
    end
  end
end
```

事实上，我们可以测试这两个函数的行为完全相同：

```py
check:
  ns = range(2, 100)
  l1 = map(pr-old, ns)
  l2 = map(pr, ns)
  l1 is l2
end
```

### 26.6 嵌套循环🔗 "链接到此处")

我们也可以用这种方式写嵌套循环。假设我们有一个列表如下

```py
lol = [list: [list: 1, 2], [list: 3], [list:], [list: 4, 5, 6]]
```

我们希望通过求和每个子列表来求和整个内容。如下所示：

```py
for loop-2(ll from lol, sum from 0):
  cases (List) ll:
    | empty => done(sum)
    | link(l, rl) =>
      l-sum =
        for loop-2(es from l, sub-sum from 0):
          cases (List) es:
            | empty => done(sub-sum)
            | link(e, r) => next-2(r, e + sub-sum)
          end
        end
      next-2(rl, sum + l-sum)
  end
end
```

我们可以通过将其写成两个函数来简化这一点：

```py
fun sum-a-lon(lon :: List<Number>):
  for loop-2(es from lon, sum from 0):
    cases (List) es:
      | empty => done(sum)
      | link(e, r) =>
        next-2(r, e + sum)
    end
  end
end

fun sum-a-lolon(lolon :: List<List<Number>>):
  for loop-2(l from lolon, sum from 0):
    cases (List) l:
      | empty => done(sum)
      | link(lon, r) =>
        next-2(r, sum-a-lon(lon) + sum)
    end
  end
end

check:
  sum-a-lolon(lol) is 21
end
```

注意到这两个函数非常相似。这表明了一种抽象：

```py
fun sum-a-list(f, L):
  for loop-2(e from L, sum from 0):
    cases (List) e:
      | empty => done(sum)
      | link(elt, r) =>
        next-2(r, f(elt) + sum)
    end
  end
end
```

使用这个，我们可以将之前的两个函数重写为：

```py
fun sum-a-lon(lon :: List<Number>):
  sum-a-list({(e): e}, lon)
end

fun sum-a-lolon(lolon :: List<List<Number>>):
  sum-a-list(sum-a-lon, lolon)
end

check:
  sum-a-lolon(lol) is 21
end
```

通过注释，我们可以清楚地了解每个函数的作用。在 `sum-a-lon` 中，每个元素是一个数字，因此它“贡献自身”到总和中。在 `sum-a-lolon` 中，每个元素是一组数字的列表，因此它“贡献其 `sum-a-lon`”到总和中。

最后，为了使这个循环完整，我们可以将上述函数重写如下：

```py
fun sum-a-lon(lon :: List<Number>):
  for sum-a-list(e :: Number from lon): e end
end

fun sum-a-lolon(lolon :: List<List<Number>>):
  for sum-a-list(l :: List<Number> from lolon): sum-a-lon(l) end
end
```

这使得每个元素贡献的内容更加清晰。在 `sum-a-lon` 中，每个元素是一个数字，因此它只贡献那个数字。在 `sum-a-lolon` 中，每个元素是一组数字的列表，因此它必须贡献该列表的 `sum-a-lon`。

### 26.7 循环、值和定制🔗 "链接到此处")

观察上述循环与传统循环在两个重要方面的不同：

1.  每个循环都会产生一个值。这与语言的其他部分一致，其中——尽可能——计算试图产生答案。我们不必产生一个值；例如，以下程序，类似于许多其他语言中的循环程序，将运行得很好：

    ```py
    for each(i from range(0, 10)): print(i) end
    ```

    然而，这是一个特殊情况。通常，我们希望表达式产生值，这样我们就可以将它们组合在一起。

1.  许多语言对应该有多少个循环结构有着强烈的看法：两个？三个？四个？在 Pyret 中，根本没有任何内置的循环结构；只有一个语法（`for`），它作为创建特定`lam`的代理。有了它，我们可以重用现有的迭代函数（如`map`和`filter`），也可以定义新的函数。有些可以非常通用，比如`loop-2`或`loop-n`，但有些则非常具体，比如`sum-a-list`。语言设计者不会阻止你编写适合你情况的循环，有时循环可以非常具有表现力，正如我们从在`for`和`sum-a-list`之上重写`sum-a-lon`和`sum-a-lolon`中看到的那样。
