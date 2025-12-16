# 28 Pyret for Racketeers and Schemers

> 原文：[`dcic-world.org/2025-08-27/p4rs.html`](https://dcic-world.org/2025-08-27/p4rs.html)

| 28.1 数字、字符串和布尔值](#%28part._.Numbers__.Strings__and_.Booleans%29) |
| --- |
| 28.2 中缀表达式](#%28part._.Infix_.Expressions%29) |
| 28.3 函数定义和应用](#%28part._.Function_.Definition_and_.Application%29) |
| 28.4 测试](#%28part._.Tests%29) |
| 28.5 变量名](#%28part._.Variable_.Names%29) |
| 28.6 数据定义](#%28part._.Data_.Definitions%29) |
| 28.7 条件语句](#%28part._.Conditionals%29) |
| 28.8 列表](#%28part._.Lists%29) |
| 28.9 首类函数](#%28part._.First-.Class_.Functions%29) |
| 28.10 注释](#%28part._.Annotations%29) |
| 28.11 还有其他什么？](#%28part._.What_.Else_%29) |

如果你之前在像 Scheme 或 Racket 的学生级别（或 WeScheme 编程环境）这样的语言中编程过，或者甚至在某些部分的 OCaml、Haskell、Scala、Erlang、Clojure 或其他语言中编程过，你会发现 Pyret 的许多部分非常熟悉。本章专门编写来帮助你通过展示如何转换语法，从（学生）Racket/Scheme/WeScheme（缩写为“RSW”）过渡到 Pyret。我们所说的内容大多适用于所有这些语言，尽管在某些情况下，我们将具体提到 Racket（和 WeScheme）中未在 Scheme 中找到的功能。

在下面的每个例子中，两个程序将产生相同的结果。

### 28.1 数字、字符串和布尔值 "链接到此处")

两个语言之间的数字非常相似。像 Scheme 一样，Pyret 实现了任意精度数字和有理数。一些 Scheme 中更奇特的数字系统（如复数）不在 Pyret 中；Pyret 还对不精确数字的处理略有不同。

| RSW | Pyret |
| --- | --- |
| 1 | `1` |
| RSW | Pyret |
| 1/2 | `1/2` |
| RSW | Pyret |
| #i3.14 | `~3.14` |

字符串也非常相似，尽管 Pyret 允许你使用单引号。

| RSW | Pyret |
| --- | --- |
| "Hello, world!" | `"Hello, world!"` |
| RSW | Pyret |
| “Hello”，他说 | `"“Hello”，他说"` |
| RSW | Pyret |
| “Hello”，他说 | `'"Hello"，他说'` |

布尔值具有相同的名称：

| RSW | Pyret |
| --- | --- |
| true | `true` |
| RSW | Pyret |
| false | `false` |

### 28.2 中缀表达式 "链接到此处")

Pyret 使用中缀语法，类似于许多其他文本编程语言：

| RSW | Pyret |
| --- | --- |
| (+ 1 2) | `1 + 2` |
| RSW | Pyret |
| (* (- 4 2) 5) | `(4 - 2) * 5` |

注意，Pyret 没有关于运算符优先级顺序的规则，所以当你混合运算符时，你必须括号表达式以使你的意图明确。当你链式使用相同的运算符时，你不需要括号；链式在两种语言中都是左结合的：

| RSW | Pyret |
| --- | --- |
| (/ 1 2 3 4) | `1 / 2 / 3 / 4` |

这两个都计算为 1/24。

### 28.3 函数定义和应用 "链接至此")

在 Pyret 中，函数定义和应用具有中缀语法，更类似于许多其他文本编程语言。应用使用从传统代数书籍中熟悉的语法：

| RSW | Pyret |
| --- | --- |
| (dist 3 4) | `dist(3, 4)` |

应用在函数头中使用类似的语法，在主体中使用中缀：

| RSW | Pyret |
| --- | --- |

|

> &#124; (define (dist x y) &#124;
> 
> &#124; (sqrt (+ (* x x) &#124;
> 
> &#124; (* y y))) &#124;

|

```py
fun dist(x, y):
  num-sqrt((x * x) +
           (y * y))
end
```

|

### 28.4 测试 "链接至此")

实际上，有三种不同的方式来编写 Racket 的 check-expect 测试的等效代码。它们可以翻译成 check 块：

| RSW | Pyret |
| --- | --- |
| (check-expect 1 1) |

```py
check:
  1 is 1
end
```

|

注意，可以将多个测试放入一个块中：

| RSW | Pyret |
| --- | --- |

|

> &#124; (check-expect 1 1) &#124;
> 
> &#124; (check-expect 2 2) &#124;

|

```py
check:
  1 is 1
  2 is 2
end
```

|

第二种方式是：作为 `check` 的别名，我们也可以写 `examples`。这两个在功能上是相同的，但它们捕捉了例子（探索问题，在尝试解决方案之前编写）和测试（试图找到解决方案中的错误，并编写来探测其设计）之间的人类差异。

第三种方式是编写一个 `where` 块来伴随函数定义。例如：

```py
fun double(n):
  n + n
where:
  double(0) is 0
  double(10) is 20
  double(-1) is -2
end
```

这些甚至可以用于内部函数（即包含在其他函数中的函数），这对于 check-expect 来说并不成立。

与 Racket 不同，Pyret 中的测试块可以包含文档字符串。当 Pyret 报告测试成功和失败时，会使用它。例如，尝试运行并查看您得到的结果：

```py
check "squaring always produces non-negatives":
  (0 * 0) is 0
  (-2 * -2) is 4
  (3 * 3) is 9
end
```

这对于记录测试块的目的很有用。

就像在 Racket 中一样，Pyret 中有许多测试运算符（除了 `is`）。参见 [文档](https://www.pyret.org/docs/latest/testing.html)。

### 28.5 变量名 "链接至此")

两种语言都有相当宽容的变量命名系统。虽然你可以在两者中使用驼峰式和下划线，但传统上使用所谓的 [kebab-case](http://c2.com/cgi/wiki?KebabCase)。这个名称不准确。单词“kebab”只是意味着“肉”。串是“shish”。因此，至少应该被称为“shish kebab case”。因此：

| RSW | Pyret |
| --- | --- |
| this-is-a-name | `this-is-a-name` |

即使 Pyret 有中缀减法，该语言也能明确区分 `this-name`（一个变量）和 `this - name`（一个减法表达式），因为后者中的 `-` 必须被空格包围。

尽管存在这种缩进约定，Pyret 并不允许 Scheme 允许的一些更奇特的名字。例如，可以这样写

> | (define e^i*pi -1) |
> | --- |

在 Scheme 中是有效的，但在 Pyret 中不是有效的变量名。

### 28.6 数据定义 "链接至此")

Pyret 在处理数据定义方面与 Racket（甚至与 Scheme）有所不同。首先，我们将看到如何定义一个结构：

| RSW | Pyret |
| --- | --- |
| (define-struct pt (x y)) |

```py
data Point:
  &#124; pt(x, y)
end
```

|

这可能看起来有些过度，但我们将很快看到为什么它是有用的。同时，值得注意的是，当数据定义中只有一种数据类型时，占用这么多行来编写它感觉很不方便。将其写在一行是有效的，但现在中间的 `|` 让代码看起来很丑：

```py
data Point: | pt(x, y) end
```

因此，Pyret 允许你省略初始的 `|`，从而使得代码更易读。

```py
data Point: pt(x, y) end
```

现在假设我们有两种类型的点。在 Racket 的学生语言中，我们会用注释来描述这一点：

> | ;; 一个点可以是 |
> | --- |
> | ;; - (pt number number), or |
> | ;; - (pt3d number number number) |

在 Pyret 中，我们可以直接表达这一点：

```py
data Point:
  | pt(x, y)
  | pt3d(x, y, z)
end
```

简而言之，Racket 优化了单变体情况，而 Pyret 优化了多变体情况。因此，在 Racket 中清楚地表达多变体情况很困难，而在 Pyret 中表达单变体情况则显得笨拙。

对于结构，Racket 和 Pyret 都暴露了构造函数、选择器和谓词。构造函数只是函数：

| RSW | Pyret |
| --- | --- |
| (pt 1 2) | `pt(1, 2)` |

谓词也是遵循特定命名方案的函数：

| RSW | Pyret |
| --- | --- |
| (pt? x) | is-pt(x) |

它们的行为方式相同（如果参数是由该构造函数构造的，则返回 true，否则返回 false）。相比之下，选择在两种语言中是不同的（我们将在下面关于 `cases` 的部分中看到更多关于选择的内容）：

| RSW | Pyret |
| --- | --- |
| (pt-x v) | `v.x` |

注意，在 Racket 的情况下，pt-x 在提取 x 字段的值之前会检查参数是否由 pt 构造。因此，pt-x 和 pt3d-x 是两个不同的函数，不能互相替代。相比之下，在 Pyret 中，`.x` 从任何具有该字段的值中提取 `x` 字段，而不考虑它是如何构造的。因此，无论值是由 `pt` 或 `pt3d`（或任何其他具有该字段的对象）构造的，我们都可以在值上使用 `.x`。相比之下，`cases` 会注意这种区别。

### 28.7 条件 "链接到此处")

Pyret 中有几种条件类型，比 Racket 的学生语言多一种。

通用条件可以使用 `if` 表达，对应于 Racket 的 `if`，但语法更复杂。

| RSW | Pyret |
| --- | --- |

|

> &#124; (if full-moon &#124;
> 
> &#124;     "howl" &#124;
> 
> &#124;     "meow") &#124;

|

```py
if full-moon:
  "howl"
else:
  "meow"
end
```

|

| RSW | Pyret |
| --- | --- |

|

> &#124; (if full-moon &#124;
> 
> &#124;     "howl" &#124;
> 
> &#124;     (if new-moon &#124;
> 
> &#124;         "bark" &#124;
> 
> &#124;         "meow")) &#124;

|

```py
if full-moon:
  "howl"
else if new-moon:
  "bark"
else:
  "meow"
end
```

|

注意，`if` 包含 `else if`，这使得可以在同一缩进级别上列出一系列问题，这在 Racket 中是不存在的。相应的 Racket 代码将写成

> | (cond |
> | --- |
> |   [full-moon "howl"] |
> |   [new-moon "bark"] |
> |   [else "meow"]) |

为了恢复缩进。Pyret 中有一个类似的构造，称为 `ask`，旨在与 `cond` 平行：

```py
ask:
  | full-moon then: "howl"
  | new-moon then:  "bark"
  | otherwise:      "meow"
end
```

在 Racket 中，我们也使用 `cond` 来根据数据类型分发：

> | (cond |
> | --- |
> | |   [(pt? v)   (+ (pt-x v) (pt-y v))] |
> | |   [(pt3d? v) (+ (pt-x v) (pt-z v))]) |

我们可以在 Pyret 中以非常相似的方式编写：

```py
ask:
  | is-pt(v)   then: v.x + v.y
  | is-pt3d(v) then: v.x + v.z
end
```

或者甚至可以写成：

```py
if is-pt(v):
  v.x + v.y
else if is-pt3d(v):
  v.x + v.z
end
```

(与 Racket 学生语言一样，如果条件分支没有匹配，Pyret 版本将发出错误信号。)

然而，Pyret 提供了专门用于数据定义的特殊语法：

```py
cases (Point) v:
  | pt(x, y)      => x + y
  | pt3d(x, y, z) => x + z
end
```

这检查 `v` 是否为 `Point`，提供了一种干净的语法方式来识别不同的分支，并使得为每个字段位置提供一个简洁的局部名称成为可能，而不是必须使用选择器如 `.x`。一般来说，在 Pyret 中，我们更喜欢使用 `cases` 来处理数据定义。然而，有时，例如，数据有多种变体，但函数只处理其中的一小部分。在这种情况下，显式使用谓词和选择器更有意义。

### 28.8 列表 "链接到此处")

在 Racket 中，根据语言级别，列表可以使用 cons 或 list 创建，空列表使用 empty。Pyret 中的对应概念分别称为 `link`、`list` 和 `empty`。`link` 是一个接受两个参数的函数，就像在 Racket 中一样：

| RSW | Pyret |
| --- | --- |
| (cons 1 empty) | `link(1, empty)` |
| RSW | Pyret |
| (list 1 2 3) | `[list: 1, 2, 3]` |

注意，表示列表的语法 `[1, 2, 3]` 在 Pyret 中是非法的：列表没有自己的专用语法。相反，我们必须使用显式的构造函数：正如 `[list: 1, 2, 3]` 构造列表一样，`[set: 1, 2, 3]` 构造集合而不是列表。实际上，我们可以 [创建自己的构造函数](https://www.pyret.org/docs/latest/Expressions.html#%28part._s~3aconstruct-expr%29) 并使用这种语法。

> 练习
> 
> > 尝试输入 `[1, 2, 3]` 并查看错误信息。

这展示了如何构造列表。要分解它们，我们使用 `cases`。有两种变体，`empty` 和 `link`（我们用来构造列表）：

| RSW | Pyret |
| --- | --- |

|

> &#124; (cond &#124;
> 
> &#124;   [(empty? l) 0] &#124;
> 
> &#124;   [(cons? l) &#124;
> 
> &#124;   (+ (first l) &#124;
> 
> &#124;     (g (rest l)))] &#124;

|

```py
cases (List) l:
  &#124; empty      => 0
  &#124; link(f, r) => f + g(r)
end
```

|

通常将字段命名为 `f` 和 `r`（分别代表“first”和“rest”）。当然，如果存在其他同名的事物，这种约定就不适用；特别是当编写列表的嵌套分解时，我们通常写成 `fr` 和 `rr`（分别代表“first of the rest”和“rest of the rest”）。

### 28.9 一等函数 "链接到此处")

Racket 的 lambda 对应于 Pyret 的 `lam`：

| RSW | Pyret |
| --- | --- |
| (lambda (x y) (+ x y)) | `lam(x, y): x + y end` |

### 28.10 注解 "链接到此处")

在学生 Racket 语言中，注解通常以注释的形式书写：

> | ; square: Number -> Number |
> | --- |
> | ; sort-nums: List<Number> -> List<Number> |
> | ; sort: List<T> * (T * T -> Boolean) -> List<T> |

在 Pyret 中，我们直接在参数和返回值上写注释。Pyret 将在有限程度上动态检查它们，并且可以使用其类型检查器静态检查它们。上述对应的注释将写成

```py
fun square(n :: Number) -> Number: ...

fun sort-nums(l :: List<Number>) -> List<Number>: ...

fun sort<T>(l :: List<T>, cmp :: (T, T -> Boolean)) -> List<T>: ...
```

### 28.11 其他内容？ "链接至此")

如果您想看到 Scheme 或 Racket 语法的其他部分被翻译，请[告知我们](http://cs.brown.edu/~sk/Contact/)。

### 28.1 数字、字符串和布尔值 "链接至此")

两个语言中的数字非常相似。像 Scheme 一样，Pyret 实现了任意精度数字和有理数。一些 Scheme 中更奇特的数字系统（如复数）不在 Pyret 中；Pyret 还对不精确数字的处理略有不同。

| RSW | Pyret |
| --- | --- |
| 1 | `1` |
| RSW | Pyret |
| 1/2 | `1/2` |
| RSW | Pyret |
| #i3.14 | `~3.14` |

字符串也非常相似，尽管 Pyret 允许您使用单引号。

| RSW | Pyret |
| --- | --- |
| "Hello, world!" | `"Hello, world!"` |
| RSW | Pyret |
| "\"Hello\", he said" | `"\"Hello\", he said"` |
| RSW | Pyret |
| "\"Hello\", he said" | `'"Hello", he said'` |

布尔值具有相同的名称：

| RSW | Pyret |
| --- | --- |
| true | `true` |
| RSW | Pyret |
| false | `false` |

### 28.2 中缀表达式 "链接至此")

Pyret 使用中缀语法，这让人联想到许多其他文本编程语言：

| RSW | Pyret |
| --- | --- |
| (+ 1 2) | `1 + 2` |
| RSW | Pyret |
| (* (- 4 2) 5) | `(4 - 2) * 5` |

注意，Pyret 没有关于运算符优先级顺序的规则，所以当您混合运算符时，您必须使用括号来明确您的意图。当您链式使用相同的运算符时，您不需要使用括号；链式在两种语言中都是左结合的：

| RSW | Pyret |
| --- | --- |
| (/ 1 2 3 4) | `1 / 2 / 3 / 4` |

这两个都等于 1/24。

### 28.3 函数定义和应用 "链接至此")

Pyret 中的函数定义和应用使用中缀语法，这让人联想到许多其他文本编程语言。应用使用从传统代数书籍中熟悉的语法：

| RSW | Pyret |
| --- | --- |
| (dist 3 4) | `dist(3, 4)` |

函数头中相应地使用类似的语法，而在函数体中使用中缀语法：

| RSW | Pyret |
| --- | --- |

|

> | (define (dist x y) | 
> 
> | | (sqrt (+ (* x x) | 
> 
> | | (* y y)))) | 

|

```py
fun dist(x, y):
  num-sqrt((x * x) +
           (y * y))
end
```

|

### 28.4 测试 "链接至此")

实际上，有三种不同的方式来编写 Racket 的 check-expect 测试的等价物。它们可以翻译成检查块：

| RSW | Pyret |
| --- | --- |
| (check-expect 1 1) |  |

```py
check:
  1 is 1
end
```

|

注意，多个测试可以放入一个单独的块中：

| RSW | Pyret |
| --- | --- |

|

> | (check-expect 1 1) | 
> 
> | (check-expect 2 2) | 

|

```py
check:
  1 is 1
  2 is 2
end
```

|

第二种方法是：作为`check`的别名，我们也可以写成`examples`。这两个在功能上是相同的，但它们捕捉了例子（探索问题，在尝试解决方案之前编写）和测试（试图在解决方案中找到错误，并编写来探测其设计）之间的人类差异。

第三种方法是编写一个`where`块来伴随函数定义。例如：

```py
fun double(n):
  n + n
where:
  double(0) is 0
  double(10) is 20
  double(-1) is -2
end
```

这些甚至可以是为内部函数（即包含在其他函数中的函数）编写的，这对于 check-expect 来说并不成立。

在 Pyret 中，与 Racket 不同，测试块可以包含文档字符串。当 Pyret 报告测试成功和失败时，会使用这个字符串。例如，尝试运行并查看你得到的结果：

```py
check "squaring always produces non-negatives":
  (0 * 0) is 0
  (-2 * -2) is 4
  (3 * 3) is 9
end
```

这对于记录测试块的目的很有用。

就像在 Racket 中一样，Pyret 中有许多测试运算符（除了`is`之外）。参见[文档](https://www.pyret.org/docs/latest/testing.html)。

### 28.5 变量名 "链接到此处")

两种语言都有相当宽容的变量命名系统。虽然你可以在两者中使用驼峰式和下划线，但传统上使用的是所谓的[kebab-case](http://c2.com/cgi/wiki?KebabCase)。这个名称不准确。单词“kebab”只是意味着“肉”。串烤架是“shish”。因此，至少应该被称为“shish kebab case”。因此：

| RSW | Pyret |
| --- | --- |
| this-is-a-name | `this-is-a-name` |

尽管 Pyret 有中缀减法，但语言可以明确地区分`this-name`（一个变量）和`this - name`（一个减法表达式），因为后者的`-`必须被空格包围。

尽管有这种间距约定，Pyret 不允许 Scheme 允许的一些更奇特的名字。例如，可以这样写

> | (define e^i*pi -1) |
> | --- |

在 Scheme 中是有效的，但在 Pyret 中不是有效的变量名。

### 28.6 数据定义 "链接到此处")

Pyret 在处理数据定义方面与 Racket（甚至更不用说 Scheme）有所不同。首先，我们将看到如何定义一个结构：

| RSW | Pyret |
| --- | --- |
| (define-struct pt (x y)) |

```py
data Point:
  &#124; pt(x, y)
end
```

|

这可能看起来有些过度，但我们将很快看到为什么它是有用的。同时，值得注意的是，当数据定义中只有一个数据类型时，占用这么多行会感觉难以管理。在一行中写它是有效的，但现在中间的`|`看起来很丑陋：

```py
data Point: | pt(x, y) end
```

因此，Pyret 允许你省略开头的`|`，从而使其更易于阅读

```py
data Point: pt(x, y) end
```

现在假设我们有两种类型的点。在 Racket 的学生语言中，我们会用注释来描述这一点：

> | ;; A Point is either |
> | --- |
> | ;; - (pt number number), or |
> | ;; - (pt3d number number number) |

在 Pyret 中，我们可以直接表达这一点：

```py
data Point:
  | pt(x, y)
  | pt3d(x, y, z)
end
```

简而言之，Racket 优化了单变体情况，而 Pyret 优化了多变体情况。因此，在 Racket 中清楚地表达多变体情况比较困难，而在 Pyret 中表达单变体情况则显得笨拙。

对于结构体，Racket 和 Pyret 都暴露了构造函数、选择器和断言。构造函数只是函数：

| RSW | Pyret |
| --- | --- |
| |   (pt 1 2) |   `pt(1, 2)` |

断言也是遵循特定命名约定的函数：

| RSW | Pyret |
| --- | --- |
| |   (pt? x) | is-pt(x) |

它们的行为方式相同（如果参数是由该构造函数构造的，则返回 true，否则返回 false）。相比之下，选择在两种语言中是不同的（我们将在下面关于`cases`的讨论中看到更多关于选择的内容）：

| RSW | Pyret |
| --- | --- |
| |   (pt-x v) | `v.x` |

注意，在 Racket 的情况下，pt-x 在提取 x 字段的值之前会检查参数是否是由 pt 构造的。因此，pt-x 和 pt3d-x 是两个不同的函数，不能互相替代。相比之下，在 Pyret 中，`.x`会从任何具有该字段的值中提取`x`字段，而不考虑它是如何构造的。因此，我们可以在由`pt`或`pt3d`（或任何其他具有该字段的对象）构造的值上使用`.x`。相比之下，`cases`则注意这种区别。

### 28.7 条件语句 "链接到此处")

Pyret 中有几种条件语句，比 Racket 的学生语言多一种。

通用条件语句可以使用 `if` 来编写，对应于 Racket 的 `if`，但语法更丰富。

| RSW | Pyret |
| --- | --- |

|

> |   (if 满月 |
> 
> |   "howl" |
> 
> |   "meow") |   "喵喵" |

|

```py
if full-moon:
  "howl"
else:
  "meow"
end
```

|

| RSW | Pyret |
| --- | --- |

|

> |   (if 满月 |
> 
> |   "howl" |
> 
> |   (if new-moon |
> 
> |   "吠叫" |
> 
> |   "meow")) |   "喵喵" |

|

```py
if full-moon:
  "howl"
else if new-moon:
  "bark"
else:
  "meow"
end
```

|

注意，`if` 包含 `else if`，这使得可以在同一缩进级别上列出一系列问题，这在 Racket 中是不具备的。在 Racket 中对应的代码将写成

> | (cond |
> | --- |
> | |   [满月 "嗥叫"] |
> | |   [新月 "吠叫"] |
> | |   [else "meow"]) |

为了恢复缩进。Pyret 中有一个类似的构造，称为`ask`，旨在与`cond`平行：

```py
ask:
  | full-moon then: "howl"
  | new-moon then:  "bark"
  | otherwise:      "meow"
end
```

在 Racket 中，我们同样使用 `cond` 来根据数据类型进行分发：

> | (cond |
> | --- |
> | |   [(pt? v)   (+ (pt-x v) (pt-y v))] |
> | |   [(pt3d? v) (+ (pt-x v) (pt-z v))] |

我们可以在 Pyret 中非常相似地写出：

```py
ask:
  | is-pt(v)   then: v.x + v.y
  | is-pt3d(v) then: v.x + v.z
end
```

或者甚至可以写成：

```py
if is-pt(v):
  v.x + v.y
else if is-pt3d(v):
  v.x + v.z
end
```

（与 Racket 的学生语言一样，如果条件语句的任何分支都没有匹配，Pyret 版本将发出错误信号。）

然而，Pyret 提供了一种特殊的语法，专门用于数据定义：

```py
cases (Point) v:
  | pt(x, y)      => x + y
  | pt3d(x, y, z) => x + z
end
```

这检查 `v` 是否为 `Point`，提供了一种干净的语法方式来识别不同的分支，并使得为每个字段位置提供一个简洁的局部名称成为可能，而不是必须使用选择器如 `.x`。一般来说，在 Pyret 中，我们更喜欢使用 `cases` 来处理数据定义。然而，有时，例如，数据有多种变体，但函数只处理其中的一小部分。在这种情况下，显式使用谓词和选择器更有意义。

### 28.8 列表 "链接至此")

在 Racket 中，根据语言级别，列表的创建使用 cons 或 list，空列表使用 empty。Pyret 中的对应概念分别称为 `link`、`list` 和 `empty`。`link` 是一个接受两个参数的函数，就像在 Racket 中一样：

| RSW | Pyret |
| --- | --- |
| (cons 1 empty) | `link(1, empty)` |
| RSW | Pyret |
| (list 1 2 3) | `[list: 1, 2, 3]` |

注意，在许多语言中表示列表的语法 `[1, 2, 3]` 在 Pyret 中是非法的：列表没有自己的特权语法。相反，我们必须使用显式的构造器：正如 `[list: 1, 2, 3]` 构建一个列表一样，`[set: 1, 2, 3]` 构建一个集合而不是列表。实际上，我们可以[创建自己的构造器](https://www.pyret.org/docs/latest/Expressions.html#%28part._s~3aconstruct-expr%29)并使用这种语法。

> 练习
> 
> > 尝试输入 `[1, 2, 3]` 并查看错误信息。

这展示了我们如何构建列表。要分解它们，我们使用 `cases`。有两种变体，`empty` 和 `link`（我们用来构建列表）：

| RSW | Pyret |
| --- | --- |

|

> &#124; (cond &#124;
> 
> &#124; [(empty? l) 0] &#124;
> 
> &#124; [(cons? l) &#124;
> 
> &#124; (+ (first l) &#124;
> 
> &#124; (g (rest l)))]) &#124;

|

```py
cases (List) l:
  &#124; empty      => 0
  &#124; link(f, r) => f + g(r)
end
```

|

通常将字段称为 `f` 和 `r`（分别代表“第一个”和“其余”）。当然，如果存在其他同名的事物，这种约定就不适用；特别是在编写列表的嵌套解构时，我们通常写作 `fr` 和 `rr`（分别代表“其余的第一个”和“其余的其余”）。

### 28.9 首类函数 "链接至此")

Racket 的 lambda 的等价物是 Pyret 的 `lam`：

| RSW | Pyret |
| --- | --- |
| (lambda (x y) (+ x y)) | `lam(x, y): x + y end` |

### 28.10 注释 "链接至此")

在学生 Racket 语言中，注释通常以注释的形式编写：

> | ; square: Number -> Number |
> | --- |
> | ; sort-nums: List<Number> -> List<Number> |
> | ; sort: List<T> * (T * T -> Boolean) -> List<T> |

在 Pyret 中，我们直接在参数和返回值上写注释。Pyret 将在有限范围内动态检查它们，并且可以使用其类型检查器静态检查它们。上述对应的注释将写作

```py
fun square(n :: Number) -> Number: ...

fun sort-nums(l :: List<Number>) -> List<Number>: ...

fun sort<T>(l :: List<T>, cmp :: (T, T -> Boolean)) -> List<T>: ...
```

### 28.11 其他内容？ "链接至此")

如果您希望看到 Scheme 或 Racket 语法中其他部分的翻译，请[告知我们](http://cs.brown.edu/~sk/Contact/)。
