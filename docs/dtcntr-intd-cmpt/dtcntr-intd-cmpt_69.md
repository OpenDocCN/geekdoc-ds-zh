# 24 预演

> 原文：[`dcic-world.org/2025-08-27/staging.html`](https://dcic-world.org/2025-08-27/staging.html)

| |   24.1 问题定义 |
| --- | --- |
| |   24.2 初始解决方案 |
| |   24.3 重构 |
| |   24.4 分离参数 |
| |   24.5 上下文 |

### 24.1 问题定义 "链接到这里")

之前，我们看到了表示祖先的二叉树的详细发展 [创建祖先树的数据类型]。在接下来的内容中，我们不需要太多细节，因此我们将给出一个本质上相同的数据定义的简化版本：

```py
data ABT:
  | unknown
  | person(name :: String, bm :: ABT, bf :: ABT)
end
```

然后，我们可以编写如下函数：

```py
fun abt-size(p :: ABT):
  doc: "Compute the number of known people in the ancestor tree"
  cases (ABT) p:
    | unknown => 0
    | person(n, p1, p2) => 1 + abt-size(p1) + abt-size(p2)
  end
end
```

现在，让我们考虑一个稍微不同的函数：`how-many-named`，它告诉我们一个家庭中有多少人有一个特定的名字。不仅可能有多个人有相同的名字，在某些文化中，跨代使用相同的名字并不罕见，无论是在连续的几代人之间还是在跳过一代。

> 现在行动！
> 
> > `how-many-named`的契约是什么？这个函数的契约将至关重要，所以请确保你完成这一步！

这里有一个有意义的契约：

```py
how-many-named :: ABT, String -> Number
```

它需要一个搜索的树，一个要搜索的名字，并返回一个计数。

> 现在行动！
> 
> > 定义`how-many-named`。

### 24.2 初始解决方案 "链接到这里")

很可能你得到了类似这样的结果：

```py
fun how-many-named(p, looking-for):
  cases (ABT) p:
    | unknown => 0
    | person(n, p1, p2) =>
      if n == looking-for:
        1 + how-many-named(p1, looking-for) + how-many-named(p2, looking-for)
      else:
        how-many-named(p1, looking-for) + how-many-named(p2, looking-for)
      end
  end
end
```

假设你已经定义了这个人：

```py
p =
  person("A",
    person("B",
      person("A", unknown, unknown),
      person("C",
        person("A", unknown, unknown),
        person("B", unknown, unknown))),
    person("C", unknown, unknown))
```

这样一来，我们可以编写一个测试，例如

```py
check:
  how-many-named(p, "A") is 3
end
```

### 24.3 重构 "链接到这里")

现在，让我们对这个函数应用一些转换，有时称为代码重构。

首先，注意重复的表达式。整个条件实际上是在说，我们想知道这个人对整体计数贡献了多少；其余的计数保持不变。

一种使这更明确的方法是（可能令人惊讶地）将`else`重写为明确指出具有不同名字的人对计数贡献`0`：

```py
fun how-many-named(p, looking-for):
  cases (ABT) p:
    | unknown => 0
    | person(n, p1, p2) =>
      if n == looking-for:
        1 + how-many-named(p1, looking-for) + how-many-named(p2, looking-for)
      else:
        0 + how-many-named(p1, looking-for) + how-many-named(p2, looking-for)
      end
  end
end
```

这种有些奇怪的重新编写的原因是它清楚地说明了什么是共同的，什么是不同的。共同的是在两个父节点中查找。变化的是这个人贡献了多少，而且只有这一点取决于条件。因此，我们可以更简洁地（如果我们知道如何阅读这样的代码，则更有意义）用以下方式表达：

```py
fun how-many-named(p, looking-for):
  cases (ABT) p:
    | unknown => 0
    | person(n, p1, p2) =>
      (if n == looking-for: 1 else: 0 end)
      +
      how-many-named(p1, looking-for) +
      how-many-named(p2, looking-for)
  end
end
```

如果你有过编程经验，这可能会让你觉得有点奇怪，但`if`实际上是一个表达式，它有一个值；在这种情况下，值是`0`或`1`。然后这个值可以用于加法。

现在，让我们更仔细地看看这段代码。注意一些有趣的事情。我们一直在向 `how-many-named` 传递两个参数；然而，其中只有一个参数（`p`）实际上在变化。我们正在寻找的名称没有改变，正如我们所期望的那样：我们在整个树中寻找相同的名称。我们如何在代码中反映这一点？

首先，我们将做一些看似无用的操作，但这也是一个无辜的改变，所以它不应该让我们太烦恼：我们将改变参数的顺序。也就是说，我们的契约从

```py
how-many-named :: ABT, String -> Number
```

to

```py
how-many-named :: String, ABT -> Number
```

因此，函数相应地变为

```py
fun how-many-named(looking-for, p):
  cases (ABT) p:
    | unknown => 0
    | person(n, p1, p2) =>
      (if n == looking-for: 1 else: 0 end)
      +
      how-many-named(p1, looking-for) +
      how-many-named(p2, looking-for)
  end
end
```

我们现在所做的是将“常数”参数放在第一位，将“变化”参数放在第二位。

> 现在行动！
> 
> > 尝试这个方法并确保它工作！

它没有！我们必须改变不仅仅是函数头：我们还需要改变它的调用方式。记住，它在函数体内部被调用了两次，并且也从示例中调用。因此，整个函数看起来是这样的：

```py
fun how-many-named(looking-for, p):
  cases (ABT) p:
    | unknown => 0
    | person(n, p1, p2) =>
      (if n == looking-for: 1 else: 0 end)
      +
      how-many-named(looking-for, p1) +
      how-many-named(looking-for, p2)
  end
end
```

并且示例中读取的是 `how-many-named("A", p)` 而不是。

### 24.4 分离参数 "链接至此")

这为我们设置了下一阶段。函数的参数旨在表示函数中可能发生变化的内容。因为一旦我们最初得到要寻找的名称，该名称就是一个常数，我们希望实际的搜索函数只接受一个参数：我们在树中的搜索位置。

即，我们希望搜索函数的契约是 `(ABT -> Number)`。为了实现这一点，我们需要另一个函数来接受 `String` 部分。因此，契约必须变为

```py
how-many-named :: String -> (ABT -> Number)
```

其中 `how-many-named` 消耗一个名称并返回一个函数，该函数将消耗实际的树以进行检查。

这表明以下函数体：

```py
fun how-many-named(looking-for):
  lam(p :: ABT) -> Number:
    cases (ABT) p:
      | unknown => 0
      | person(n, p1, p2) =>
        (if n == looking-for: 1 else: 0 end)
        +
        how-many-named(looking-for, p1) +
        how-many-named(looking-for, p2)
    end
  end
end
```

然而，这个函数体是不正确的：Pyret 类型检查器会给我们类型错误。这是因为 `how-many-named` 只接受一个参数，而不是两个，就像在两个递归调用中那样。

我们如何解决这个问题？记住，这个改变的整个目的是我们不想改变名称，只想改变树。这意味着我们想要在内部函数上递归。我们目前无法这样做，因为它没有名字！所以我们必须给它一个名字，并在它上面递归：

```py
fun how-many-named(looking-for):
  fun search-in(p :: ABT) -> Number:
    cases (ABT) p:
      | unknown => 0
      | person(n, p1, p2) =>
        (if n == looking-for: 1 else: 0 end)
        +
        search-in(p1) +
        search-in(p2)
    end
  end
end
```

这现在让我们只对应该变化的部分进行递归，同时保持我们要寻找的名称不变（因此，在搜索期间保持固定）。

> 现在行动！
> 
> > 尝试上面的方法并确保它工作。

仍然不行：上面的函数体有一个语法错误！这是因为 `how-many-named` 实际上没有返回任何类型的值。

它应该返回什么？一旦我们向函数提供一个名称，我们应该得到一个在树中搜索该名称的函数。但我们已经有了这样一个函数：`search-in`。因此，`how-many-named` 应该只返回 … `search-in`。

```py
fun how-many-named(looking-for):
  fun search-in(p :: ABT) -> Number:
    cases (ABT) p:
      | unknown => 0
      | person(n, p1, p2) =>
        (if n == looking-for: 1 else: 0 end)
        +
        search-in(p1) +
        search-in(p2)
    end
  end

  search-in
end
```

这仍然不起作用，因为我们还没有更改示例。让我们更新一下：我们如何使用 `how-many-named`？我们必须用名字（例如 `"A"`）来调用它；这会返回一个函数——<wbr>绑定到 `search-in` 的函数——<wbr>它期望一个祖先树。这样做应该会返回一个计数。因此，示例应该被重写为

```py
how-many-As = how-many-named("A")
how-many-As(p) is 3
```

这是一种编写示例的有教育意义的途径。然而，我们也可以更简洁地编写它。注意，`how-many-named("A")` 返回一个函数，我们应用函数到参数的方式是 `(…)`。因此，我们也可以这样写：

```py
how-many-named("A")(p) is 3
```

### 24.5 上下文 "链接到此处")

我们刚才应用的那种转换通常被称为柯里化，以纪念 Haskell Curry，他是最早描述它的人之一，尽管它是由 Moses Schönfinkel 更早发现的，甚至更早由 Gottlob Frege 发现。这里柯里化的特定用途，即我们更早地将“静态”参数移动，更晚地将“动态”参数移动，并在静态-动态划分上拆分，被称为分阶段。这是一种非常有用的编程技术，而且更重要的是，它使一些编译器能够生成更高效的程序。

更加微妙但重要的是，分阶段计算与未分阶段计算讲述的故事不同，我们可以仅从契约中读出这一点：

```py
how-many-named :: String, ABT -> Number
how-many-named :: String -> (ABT -> Number)
```

第一个说字符串可能与个人共同变化。第二个排除了这种解释。

> 现在行动！
> 
> > 前者有用吗？我们什么时候会有名字也变化的情况？

想象一个稍微不同的问题：我们想知道一个孩子有多少次与父母有相同的名字。然后，当我们遍历树时，随着人的名字（可能）不断变化，我们在父母中寻找的名字也发生了变化。

> 练习
> 
> > 编写这个函数。

相反，分阶段类型排除了这种解释和这种行为。通过这种方式，它向读者发出信号，说明计算可能的行为仅从类型中就可以看出。同样，未分阶段类型可以被解读为给读者一个暗示，说明行为可能取决于两个参数的变化，因此可以容纳更广泛的行为范围（例如，检查父母-孩子或祖父母-孩子名字的重用）。

这里还有一个分阶段的好例子：一点微积分。

最后，值得注意的是，一些语言，如 Haskell 和 OCaml，会自动进行这种转换。事实上，它们甚至没有多参数函数：看起来像多个参数实际上是一系列分阶段函数。这可以极端地导致一种非常优雅且强大的编程风格。Pyret 选择不这样做，因为虽然这对于高级程序员来说是一个强大的工具，但对于经验较少的程序员来说，发现参数数量和参数之间的不匹配是非常有用的。

### 24.1 问题定义 "链接到此处")

之前，我们看到了表示祖先树的二叉树的详细开发 [创建祖先树的数据类型]。在接下来的内容中，我们不需要很多细节，所以我们将给出一个本质上相同的数据定义的简化版本：

```py
data ABT:
  | unknown
  | person(name :: String, bm :: ABT, bf :: ABT)
end
```

然后，我们可以编写这样的函数：

```py
fun abt-size(p :: ABT):
  doc: "Compute the number of known people in the ancestor tree"
  cases (ABT) p:
    | unknown => 0
    | person(n, p1, p2) => 1 + abt-size(p1) + abt-size(p2)
  end
end
```

现在，让我们考虑一个稍微不同的函数：`how-many-named`，它告诉我们一个家庭中有多少人有一个特定的名字。不仅可能有多个人有相同的名字，在某些文化中，跨代使用相同的名字也不罕见，无论是连续几代还是跳过一代。

> 现在就做！
> 
> > `how-many-named`的合约是什么？这个函数的合约将至关重要，所以请确保你完成这一步！

这里有一个有意义的合约：

```py
how-many-named :: ABT, String -> Number
```

它接受一个要搜索的树，一个要搜索的名称，并返回一个计数。

> 现在就做！
> 
> > 定义`how-many-named`。

### 24.2 初始解决方案 "链接到这里")

很可能你得到了类似这样的结果：

```py
fun how-many-named(p, looking-for):
  cases (ABT) p:
    | unknown => 0
    | person(n, p1, p2) =>
      if n == looking-for:
        1 + how-many-named(p1, looking-for) + how-many-named(p2, looking-for)
      else:
        how-many-named(p1, looking-for) + how-many-named(p2, looking-for)
      end
  end
end
```

假设你定义了这个人物：

```py
p =
  person("A",
    person("B",
      person("A", unknown, unknown),
      person("C",
        person("A", unknown, unknown),
        person("B", unknown, unknown))),
    person("C", unknown, unknown))
```

有了这个，我们可以编写一个测试，例如

```py
check:
  how-many-named(p, "A") is 3
end
```

### 24.3 重构 "链接到这里")

现在让我们应用一些变换，有时称为代码重构，对这个函数进行变换。

首先，注意重复的表达式。整个条件实际上是在说，我们想知道这个人对整体计数的贡献有多少；其余的计数保持不变。

一种使这更明确的方法是（可能令人惊讶地）将`else`重写为明确指出具有不同名称的人对计数贡献`0`：

```py
fun how-many-named(p, looking-for):
  cases (ABT) p:
    | unknown => 0
    | person(n, p1, p2) =>
      if n == looking-for:
        1 + how-many-named(p1, looking-for) + how-many-named(p2, looking-for)
      else:
        0 + how-many-named(p1, looking-for) + how-many-named(p2, looking-for)
      end
  end
end
```

这种有些奇怪的重新编写的原因是它清楚地说明了什么是共同的，什么是不同的。共同的是查看两个父节点。变化的是这个人对贡献的多少，而且只有这一点取决于条件。因此，我们可以更简洁地（如果我们知道如何阅读这样的代码，更有意义地）用以下方式表达：

```py
fun how-many-named(p, looking-for):
  cases (ABT) p:
    | unknown => 0
    | person(n, p1, p2) =>
      (if n == looking-for: 1 else: 0 end)
      +
      how-many-named(p1, looking-for) +
      how-many-named(p2, looking-for)
  end
end
```

如果你有过编程经验，这可能会让你觉得有点奇怪，但`if`实际上是一个表达式，它有一个值；在这种情况下，这个值是`0`或`1`。这个值然后可以用于加法。

现在，让我们更仔细地看看这段代码。注意一些有趣的事情。我们一直在向`how-many-named`传递两个参数；然而，其中只有一个参数（`p`）实际上在变化。我们正在寻找的名称没有变化，正如我们所期望的：我们在整个树中寻找相同的名称。我们如何在代码中反映这一点？

首先，我们将做一些看似无用的东西，但这也是一个无辜的改变，所以它不应该让我们太烦恼：我们将改变参数的顺序。也就是说，我们的合约从

```py
how-many-named :: ABT, String -> Number
```

变为

```py
how-many-named :: String, ABT -> Number
```

因此，函数相应地变为

```py
fun how-many-named(looking-for, p):
  cases (ABT) p:
    | unknown => 0
    | person(n, p1, p2) =>
      (if n == looking-for: 1 else: 0 end)
      +
      how-many-named(p1, looking-for) +
      how-many-named(p2, looking-for)
  end
end
```

我们现在所做的是将“常数”参数放在第一位，将“变化”参数放在第二位。

> 现在就做！
> 
> > 尝试这个，确保它工作！

它仍然不行！我们必须改变不仅仅是函数头：我们还要改变它的调用方式。记住，它在函数体内部被调用了两次，也从例子中被调用。因此，整个函数的读法如下：

```py
fun how-many-named(looking-for, p):
  cases (ABT) p:
    | unknown => 0
    | person(n, p1, p2) =>
      (if n == looking-for: 1 else: 0 end)
      +
      how-many-named(looking-for, p1) +
      how-many-named(looking-for, p2)
  end
end
```

并且例子读作`how-many-named("A", p)`。

### 24.4 分离参数 "链接到此处")

这为我们设置了下一阶段。函数的参数是用来指示函数中可能变化的内容。因为我们正在寻找的名称一旦我们最初得到它就是一个常数，我们希望实际的搜索函数只接受一个参数：我们在树中的搜索位置。

即，我们希望搜索函数的合约是`(ABT -> Number)`。为了实现这一点，我们需要另一个函数，它将接受`String`部分。因此，合约必须变为

```py
how-many-named :: String -> (ABT -> Number)
```

其中`how-many-named`消耗一个名称并返回一个将消耗实际树来检查的函数。

这表明以下函数体：

```py
fun how-many-named(looking-for):
  lam(p :: ABT) -> Number:
    cases (ABT) p:
      | unknown => 0
      | person(n, p1, p2) =>
        (if n == looking-for: 1 else: 0 end)
        +
        how-many-named(looking-for, p1) +
        how-many-named(looking-for, p2)
    end
  end
end
```

然而，这个函数体是不正确的：Pyret 类型检查器会给我们类型错误。这是因为`how-many-named`只接受一个参数，而不是两个，就像在两个递归调用中那样。

我们该如何修复这个问题？记住，这个改变的整个目的是我们不想改变名称，只想改变树。这意味着我们想要递归调用内部函数。我们目前无法这样做，因为它没有名称！所以我们必须给它一个名称，并递归调用它：

```py
fun how-many-named(looking-for):
  fun search-in(p :: ABT) -> Number:
    cases (ABT) p:
      | unknown => 0
      | person(n, p1, p2) =>
        (if n == looking-for: 1 else: 0 end)
        +
        search-in(p1) +
        search-in(p2)
    end
  end
end
```

现在它让我们可以递归地处理应该变化的部分，同时保持我们正在寻找的名称不变（因此，在搜索期间是固定的）。

> 现在就做！
> 
> > 尝试上面的方法并确保它工作。

仍然不行：上面的主体有一个语法错误！这是因为`how-many-named`实际上没有返回任何类型的值。

它应该返回什么？一旦我们给函数提供了一个名称，我们应该得到一个在树中搜索该名称的函数。但我们已经有了这样一个函数：`search-in`。因此，`how-many-named`应该只返回… `search-in`。

```py
fun how-many-named(looking-for):
  fun search-in(p :: ABT) -> Number:
    cases (ABT) p:
      | unknown => 0
      | person(n, p1, p2) =>
        (if n == looking-for: 1 else: 0 end)
        +
        search-in(p1) +
        search-in(p2)
    end
  end

  search-in
end
```

这仍然不会工作，因为我们没有改变例子。让我们更新一下：我们如何使用`how-many-named`？我们必须用名称（如`"A"`）调用它；这返回一个函数——<wbr>绑定到`search-in`的函数——<wbr>它期望一个祖先树。这样做应该返回一个计数。因此，例子应该被重写为

```py
how-many-As = how-many-named("A")
how-many-As(p) is 3
```

这是一种很有教育意义的写例子方式。然而，我们也可以更简洁地写。注意，`how-many-named("A")`返回一个函数，而我们应用函数到参数的方式是`(…)`。因此，我们也可以这样写：

```py
how-many-named("A")(p) is 3
```

### 24.5 上下文 "链接到此处")

我们刚才应用的那种转换通常被称为柯里化，以纪念哈斯克尔·柯里，他是最早描述这种转换的人之一，尽管它是由摩西·申芬克尔发现的，甚至更早的是戈特洛布·弗雷格。这里柯里化的特定用途，即我们更早地将“静态”参数移动，而将“动态”参数移动得更晚，并在静态-动态划分上拆分，被称为分阶段。这是一个非常有用的编程技术，而且它还使一些编译器能够生成更高效的程序。

更加微妙但重要的是，分阶段计算讲述的故事与未分阶段的不同，我们可以仅从契约中读出这一点：

```py
how-many-named :: String, ABT -> Number
how-many-named :: String -> (ABT -> Number)
```

第一个说字符串可能与人物共同变化。第二个排除了那种解释。

> 现在行动起来！
> 
> > 前者有用吗？我们什么时候会有名字也变化的情况？

想象一个稍微不同的问题：我们想知道一个孩子与父母同名的情况有多常见。然后，当我们遍历树时，随着人物（可能）的名字不断变化，我们在父母那里寻找的名字也会随之改变。

> 练习
> 
> > 编写这个函数。

相比之下，分阶段类型规则排除了那种解释和行为。这样，它向读者传递了一个信号，即计算可能的行为仅从类型中就可以得知。同样，未分阶段类型可以被解读为给读者一个暗示，即行为可能取决于两个参数的变化，因此可以容纳更广泛的行为范围（例如，检查父母与子女或祖父母与孙子女的名字重复）。

这里还有一个关于分阶段的非常棒的例子：一点微积分。

最后，值得注意的是，一些语言，如 Haskell 和 OCaml，会自动进行这种转换。事实上，它们甚至没有多参数函数：看起来像多个参数实际上是一系列分阶段函数。这可以极端地导致一种非常优雅且强大的编程风格。Pyret 选择不这样做，因为虽然这对于高级程序员来说是一个强大的工具，但对于经验较少的程序员来说，发现参数和参数数量不匹配是非常有用的。
