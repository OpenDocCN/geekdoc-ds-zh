# 3.3 从重复表达式到函数

> 原文：[`dcic-world.org/2025-08-27/From_Repeated_Expressions_to_Functions.html`](https://dcic-world.org/2025-08-27/From_Repeated_Expressions_to_Functions.html)

| |   3.3.1 示例：相似的旗帜 |
| --- | --- |
| |   3.3.2 定义函数 |
| |     3.3.2.1 函数如何评估 |
| |     3.3.2.2 类型注解 |
| |     3.3.2.3 文档 |
| |   3.3.3 函数练习：月球重量 |
| |   3.3.4 使用示例文档化函数 |
| |   3.3.5 函数练习：钢笔成本 |
| |   3.3.6 回顾：定义函数 |

#### 3.3.1 示例：相似的旗帜 "链接到这里")

考虑以下两个表达式来绘制亚美尼亚和奥地利（分别）的旗帜。这两个国家拥有相同的旗帜，只是颜色不同。`frame`操作符在图像周围绘制一个小黑框。

```py
# Lines starting with # are comments for human readers.
# Pyret ignores everything on a line after #.

# armenia
frame(
  above(rectangle(120, 30, "solid", "red"),
    above(rectangle(120, 30, "solid", "blue"),
      rectangle(120, 30, "solid", "orange"))))

# austria
frame(
  above(rectangle(120, 30, "solid", "red"),
    above(rectangle(120, 30, "solid", "white"),
      rectangle(120, 30, "solid", "red"))))
```

而不是写两次这个程序，我们最好只写一次公共表达式，然后只需更改颜色来生成每个旗帜。具体来说，我们希望有一个自定义操作符，比如`three-stripe-flag`，我们可以用它如下：

```py
# armenia
three-stripe-flag("red", "blue", "orange")

# austria
three-stripe-flag("red", "white", "red")
```

在这个程序中，我们只提供`three-stripe-flag`，它带有自定义图像创建到特定标志的信息。操作本身将负责创建和对齐矩形。我们希望最终得到的亚美尼亚和奥地利国旗与我们的原始程序得到的结果相同。Pyret 中不存在这样的操作符：它仅限于我们创建国旗图像的应用。因此，为了让这个程序工作，我们需要能够在 Pyret 中添加自己的操作符（即函数）。

#### 3.3.2 定义函数 "链接到这里")

在编程中，函数接受一个或多个（配置）参数，并使用它们来产生结果。

> 策略：从表达式创建函数
> 
> > 如果我们有多个具体的表达式，除了几个特定的数据值外完全相同，我们可以创建一个函数，其共同代码如下：
> > 
> > +   至少写下两个表示所需计算的表达式（在这种情况下，生成亚美尼亚和奥地利旗帜的表达式）。
> > +   
> > +   确定哪些部分是固定的（即创建尺寸为`120`和`30`的矩形，使用`above`堆叠矩形），哪些是变化的（即条纹颜色）。
> > +   
> > +   对于每个变化的部分，给它起一个名字（比如`top`、`middle`和`bottom`），这将代表该部分的参数。
> > +   
> > +   将示例重写为这些参数的形式。例如：
> > +   
> >     ```py
> >     frame(
> >       above(rectangle(120, 30, "solid", top),
> >         above(rectangle(120, 30, "solid", middle),
> >           rectangle(120, 30, "solid", bottom))))
> >     ```
> >     
> > +   给函数起一个有暗示性的名字：例如，`three-stripe-flag`。
> > +   
> > +   将函数的语法围绕表达式写出来：
> > +   
> >     ```py
> >     fun <function name>(<parameters>):
> >       <the expression goes here>
> >     end
> >     ```
> >     
> >     其中表达式被称为函数体。（程序员经常使用尖括号来说“用适当的东西替换”，括号本身不是符号的一部分。）

这是最终产品：

```py
fun three-stripe-flag(top, middle, bottom):
  frame(
    above(rectangle(120, 30, "solid", top),
      above(rectangle(120, 30, "solid", middle),
        rectangle(120, 30, "solid", bottom))))
end
```

虽然现在看起来工作量很大，但一旦习惯了，就不会这样了。我们会一遍又一遍地走相同的步骤，最终它们会变得如此直观，以至于你不需要从多个类似的表达式开始。

> 现在行动起来！
> 
> > 为什么函数体只有一个表达式，而之前我们为每个标志都有一个单独的表达式？

我们只有一个表达式，因为整个目的就是要消除所有变化的部分，并用参数替换它们。

拥有这个函数后，我们可以编写以下两个表达式来生成我们原始的旗帜图像：

```py
three-stripe-flag("red", "blue", "orange")
three-stripe-flag("red", "white", "red")
```

当我们为函数的参数提供值以获得结果时，我们说我们在调用函数。我们使用术语“调用”来表示这种形式的表达式。

如果我们想要命名生成的图像，可以按照以下方式操作：

```py
armenia = three-stripe-flag("red", "blue", "orange")
austria = three-stripe-flag("red", "white", "red")
```

（旁注：Pyret 只允许目录中每个名称一个值。如果你的文件已经对 `armenia` 或 `austria` 有定义，Pyret 在这一点上会给你一个错误。你可以使用不同的名称（如 `austria2`）或使用 `#` 注释掉原始定义。）

##### 3.3.2.1 函数评估 "链接到此处")

到目前为止，我们已经学习了 Pyret 处理你的程序的三条规则：

+   如果你写了一个表达式，Pyret 会评估它以产生其值。

+   如果你写了一个定义名称的语句，Pyret 会评估该表达式（`=` 的右侧），然后在目录中创建一个条目，将名称与值关联起来。

+   如果你写了一个使用目录中名称的表达式，Pyret 会用相应的值替换该名称。

现在我们能够定义自己的函数了，我们必须考虑两个额外的案例：当你定义一个函数（使用 `fun`）时 Pyret 会做什么，以及当你调用一个函数（提供参数值）时 Pyret 会做什么？

+   当 Pyret 在你的文件中遇到函数定义时，它会在目录中创建一个条目，将函数的名称与代码关联起来。此时不会评估函数体。

+   当 Pyret 在评估表达式时遇到函数调用，它会用函数体替换调用，但将体中的参数名称替换为参数值。然后 Pyret 继续使用替换后的值评估体。

作为函数调用规则的例子，如果你评估

```py
three-stripe-flag("red", "blue", "orange")
```

Pyret 从函数体开始

```py
frame(
  above(rectangle(120, 30, "solid", top),
    above(rectangle(120, 30, "solid", middle),
      rectangle(120, 30, "solid", bottom))))
```

替换参数值

```py
frame(
  above(rectangle(120, 30, "solid", "red"),
    above(rectangle(120, 30, "solid", "blue"),
      rectangle(120, 30, "solid", "orange"))))
```

然后评估表达式，生成旗帜图像。

注意，第二个表达式（替换后的值）与我们最初用于亚美尼亚国旗的表达式相同。替换恢复了该表达式，同时仍然允许程序员用 `three-stripe-flag` 的简写形式来编写。

##### 3.3.2.2 类型注解 "链接到此处")

如果我们犯了一个错误，并尝试如下调用该函数：

```py
three-stripe-flag(50, "blue", "red")
```

> 立刻行动！
> 
> > 你认为 Pyret 会为这个表达式产生什么结果？

`three-stripe-flag` 的第一个参数应该是顶部条纹的颜色。值 `50` 不是一个字符串（更不用说是一个命名颜色的字符串）。Pyret 将在第一次调用 `rectangle` 时将 `50` 替换为 `top`，产生以下结果：

```py
frame(
  above(rectangle(120, 30, "solid", 50),
    above(rectangle(120, 30, "solid", "blue"),
      rectangle(120, 30, "solid", "red"))))
```

当 Pyret 尝试评估 `rectangle` 表达式以创建顶部条纹时，它生成一个错误，该错误引用了那次对 `rectangle` 的调用。

如果别人正在使用你的函数，这个错误可能没有意义：他们没有写一个关于矩形的表达式。难道不是更好让 Pyret 报告 `three-stripe-flag` 本身的使用有问题吗？

作为 `three-stripe-flag` 的作者，你可以通过为每个参数注解提供有关预期值类型的详细信息来实现这一点。以下是函数定义的再次呈现，这次要求三个参数必须是字符串：

```py
fun three-stripe-flag(top :: String,
      middle :: String,
      bottom :: String):
  frame(
    above(rectangle(120, 30, "solid", top),
      above(rectangle(120, 30, "solid", middle),
        rectangle(120, 30, "solid", bottom))))
end
```

注意，这里的符号与我们在文档中的合约中看到的类似：参数名称后面跟着一个双冒号（`::`）和一个类型名称（到目前为止，是 `Number`、`String` 或 `Image` 之一）。将每个参数放在单独一行不是必需的，但有时有助于可读性。

使用这个新定义运行你的文件，并再次尝试错误的调用。你应该得到一个不同的错误消息，该消息仅涉及 `three-stripe-flag`。

在函数的输出类型上添加类型注解以捕获函数输出类型也是常见的做法。该注解位于参数列表之后：

```py
fun three-stripe-flag(top :: String,
      middle :: String,
      bottom :: String) -> Image:
  frame(
    above(rectangle(120, 30, "solid", top),
      above(rectangle(120, 30, "solid", middle),
        rectangle(120, 30, "solid", bottom))))
end
```

注意，所有这些类型注解都是可选的。Pyret 不论你是否包含它们都会运行你的程序。你可以对某些参数添加类型注解，而对其他参数不添加；你可以包含输出类型，但不包含任何参数类型。不同的编程语言对类型的规则不同。

我们将把类型视为扮演两个角色：为 Pyret 提供信息，使其能够更准确地聚焦错误信息，并指导程序的人类读者正确使用用户定义的函数。

##### 3.3.2.3 文档 "链接到此处")

想象一下，几个月后你打开了本章的程序文件。你会记得 `three-stripe-flag` 执行的计算是什么吗？名字确实很有暗示性，但它遗漏了条纹是垂直堆叠（而不是水平堆叠）以及条纹高度相等等细节。函数名称并不是为了携带这么多信息而设计的。

程序员还使用文档字符串注释函数，这是一个简短的、用人类语言描述函数做什么的描述。以下是一个 Pyret 文档字符串的例子，用于 `three-stripe-flag`：

```py
fun three-stripe-flag(top :: String,
      middle :: String,
      bottom :: String) -> Image:
  doc: "produce image of flag with three equal-height horizontal stripes"
  frame(
    above(rectangle(120, 30, "solid", top),
      above(rectangle(120, 30, "solid", middle),
        rectangle(120, 30, "solid", bottom))))
end
```

虽然从 Pyret 的角度来看，文档字符串也是可选的，但当你编写函数时，你应该始终提供一个。对于任何必须阅读你的程序的人来说，它们都非常有帮助，无论是同事、评分者……还是几周后的你自己。

#### 3.3.3 函数练习：月球重量 "链接到此处")

假设我们负责为月球探险队配备宇航员装备。我们必须确定每位宇航员在月球表面的体重。在月球上，物体的重量只有地球上重量的六分之一。以下是几位宇航员的体重表达式（以磅为单位）：

```py
100 * 1/6
150 * 1/6
90 * 1/6
```

就像我们的亚美尼亚和奥地利国旗的例子一样，我们正在多次写相同的表达式。这是我们应该创建一个函数的另一个情况，该函数将变化的数据作为参数，但只捕获一次固定的计算。

在旗帜的情况下，我们注意到我们实际上写了一个相同的表达式多次。这里，我们有一个我们预期会多次执行的计算（每次为每位宇航员）。一遍又一遍地写相同的表达式很无聊。此外，如果我们多次复制或重新输入一个表达式，迟早会犯转录错误。这是一个 [DRY 原则](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself) 的例子，其中 DRY 意味着“不要重复自己”。

让我们回顾一下创建函数的步骤：

+   记录一些所需计算的示例。我们上面已经做了。

+   确定哪些部分是固定的（上面，`* 1/6`）以及哪些是变化的（上面，`100`，`150`，`90`……）。

+   对于每个变化的部分，给它一个名字（比如 `earth-weight`），这将代表它的参数。

+   将示例重写为该参数的形式：

    ```py
    earth-weight * 1/6
    ```

    这将是主体，即函数内的表达式。

+   为函数想出一个有暗示性的名字：例如，`moon-weight`。

+   将函数的语法围绕主体表达式编写：

    ```py
    fun moon-weight(earth-weight):
      earth-weight * 1/6
    end
    ```

+   记得包括参数和输出的类型，以及文档字符串。这会产生最终的函数：

    ```py
    fun moon-weight(earth-weight :: Number) -> Number:
      doc: "Compute weight on moon from weight on earth"
      earth-weight * 1/6
    end
    ```

#### 3.3.4 使用示例记录函数 "链接到此处")

在上述每个函数中，我们首先从一些我们想要计算的示例开始，从那里推广到通用公式，将其转化为函数，然后使用该函数代替原始表达式。

现在我们已经完成了，最初的示例有什么用呢？看起来很诱人，想要把它们扔掉。然而，关于软件的一个重要规则你应该学习：软件会进化。随着时间的推移，任何有实际用途的程序都会发生变化和增长，因此最终可能会产生与最初不同的值。有时这是故意的，但有时这是错误的结果（包括在打字时意外添加或删除文本这样的愚蠢但不可避免的错误）。因此，保留这些示例以供将来参考总是有用的，这样你就可以立即得知函数是否偏离了它应该概括的示例。

Pyret 使这变得很容易做。每个函数都可以伴随一个记录示例的`where`子句。例如，我们的`moon-weight`函数可以被修改为读取：

```py
fun moon-weight(earth-weight :: Number) -> Number:
  doc: "Compute weight on moon from weight on earth"
  earth-weight * 1/6
where:
  moon-weight(100) is 100 * 1/6
  moon-weight(150) is 150 * 1/6
  moon-weight(90) is 90 * 1/6
end
```

当以这种方式编写时，Pyret 实际上会在每次运行程序时检查答案，并在你更改函数使其与这些示例不一致时通知你。

> 现在就做！
> 
> > 检查这个！更改公式——例如，将函数体替换为
> > 
> > ```py
> > earth-weight * 1/3
> > ```
> > 
> > ——并查看会发生什么。注意 CPO 的输出：你应该习惯于识别这种类型的输出。
> > 
> 现在就做！
> 
> > 现在，修复函数体，然后更改其中一个答案——例如，写入
> > 
> > ```py
> > moon-weight(90) is 90 * 1/3
> > ```
> > 
> > ——并查看会发生什么。将这种情况的输出与上面的输出进行对比。

当然，对于这样一个简单的函数来说，你犯错的可能性很小（除非是打字错误）。毕竟，示例与函数本身的主体非常相似。然而，稍后我们将看到，示例可以比主体简单得多，而且确实有可能出现不一致的情况。到那时，示例在确保我们没有在程序中犯错方面变得非常有价值。事实上，这在专业软件开发中非常有价值，优秀的程序员总是写下大量的示例——称为测试——以确保他们的程序按预期运行。

对于我们的目的，我们编写示例作为确保我们理解问题的过程的一部分。在开始编写代码解决问题之前确保你理解问题总是一个好主意。示例是一个很好的中间点：你可以先在具体值上草拟相关的计算，然后考虑将其转换为函数。如果你无法编写示例，那么你可能也无法编写函数。示例将编程过程分解成更小、更易管理的步骤。

#### 3.3.5 函数练习：钢笔的成本 "链接到此处")

让我们再创建一个函数，这次是一个更复杂的例子。想象一下，你正在尝试计算带有标语（或信息）的笔订单的总成本。每支笔的成本是 25 美分，加上每条信息中每个字符额外的 2 美分（我们将单词之间的空格也计为字符）。

再次按照我们的步骤创建一个函数，让我们先写出两个具体的表达式来完成这个计算。

```py
# ordering 3 pens that say "wow"
3 * (0.25 + (string-length("wow") * 0.02))

# ordering 10 pens that say "smile"
10 * (0.25 + (string-length("smile") * 0.02))
```

这些例子介绍了一个新的内置函数，称为 `string-length`。它接受一个字符串作为输入，并产生字符串中的字符数（包括空格和标点符号）。这些例子还展示了处理非整数数字的示例。Pyret 要求小数点前面的数字，所以如果“整数部分”为零，你需要在小数点前写 `0`。此外，请注意 Pyret 使用小数点；它不支持[“0,02”](https://en.wikipedia.org/wiki/Decimal_separator)之类的约定。

编写函数的第二步是确定我们两个例子中哪些信息不同。在这种情况下，我们有两个：笔的数量和要放在笔上的信息。这意味着我们的函数将有两个参数，而不仅仅是一个。

```py
fun pen-cost(num-pens :: Number, message :: String) -> Number:
  num-pens * (0.25 + (string-length(message) * 0.02))
end
```

当然，当内容过长时，使用多行可能会有所帮助：

```py
fun pen-cost(num-pens :: Number, message :: String)
  -> Number:
  num-pens * (0.25 + (string-length(message) * 0.02))
end
```

如果你想要编写一个多行文档字符串，你需要使用 ```py` ``` ```py` rather than `"` to begin and end it, like so:

```

fun pen-cost(num-pens :: Number, message :: String)

-> Number:

doc: ```pytotal cost for pens, each 25 cents
       plus 2 cents per message character```

num-pens * (0.25 + (字符串长度(message) * 0.02))

end

```py

We should also document the examples that we used when creating the function:

```

fun pen-cost(num-pens :: Number, message :: String)

-> Number:

doc: ```pytotal cost for pens, each 25 cents
       plus 2 cents per message character```

num-pens * (0.25 + (字符串长度(message) * 0.02))

其中：

pen-cost(3, "wow")

    is 3 * (0.25 + (字符串长度("wow") * 0.02))

pen-cost(10, "smile")

    is 10 * (0.25 + (字符串长度("smile") * 0.02))

end

```py

When writing `where` examples, we also want to include special yet valid cases that the function might have to handle, such as an empty message.

```

pen-cost(5, "") is 5 * 0.25

```py

Note that our empty-message example has a simpler expression on the right side of `is`. The expression for what the function returns doesn’t have to match the body expression; it simply has to evaluate to the same value as you expect the example to produce. Sometimes, we’ll find it easier to just write the expected value directly. For the case of someone ordering no pens, for example, we’d include:

```

pen-cost(0, "bears") is 0

```py

The point of the examples is to document how a function behaves on a variety of inputs. What goes to the right of the `is` should summarize the computation or the answer in some meaningful way. Most important? Do not write the function, run it to determine the answer, then put that answer on the right side of the `is`! Why not? Because the examples are meant to give some redundancy to the design process, so that you catch errors you might have made. If your function body is incorrect, and you use the function to generate the example, you won’t get the benefit of using the example to check for errors.

We’ll keep returning to this idea of writing good examples. Don’t worry if you still have questions for now. Also, for the time being, we won’t worry about nonsensical situations like negative numbers of pens. We’ll get to those after we’ve learned additional coding techniques that will help us handle such situations properly.

> Do Now!
> 
> > We could have combined our two special cases into one example, such as
> > 
> > ```

> > pen-cost(0, "") is 0
> > 
> > ```py
> > 
> > Does doing this seem like a good idea? Why or why not?

#### 3.3.6 Recap: Defining Functions "Link to here")

This chapter has introduced the idea of a function. Functions play a key role in programming: they let us configure computations with different concrete values at different times. The first time we compute the cost of pens, we might be asking about `10` pens that say `"Welcome"`. The next time, we might be asking about `100` pens that say `"Go Bears!"`. The core computation is the same in both cases, so we want to write it out once, configuring it with different concrete values each time we use it.

We’ve covered several specific ideas about functions:

*   We showed the `fun` notation for writing functions. You learned that a function has a name (that we can use to refer to it), one or more parameters (names for the values we want to configure), as well as a body, which is the computation that we want to perform once we have concrete values for the parameters.

*   We showed that we should include examples with our functions, to illustrate what the function computes on various specific values. Examples go in a `where` block within the function.

*   We showed that we can use a function by providing concrete values to configure its parameters. To do this, we write the name of the function we want to use, followed by a pair of parenthesis around comma-separated values for the parameters. For example, writing the following expression (at the interactions prompt) will compute the cost of a specific order of pens:

    ```

    pen-cost(10, "Welcome")

    ```py

*   We discussed that if we define a function in the definitions pane then press Run, Pyret will make an entry in the directory with the name of the function. If we later use the function, Pyret will look up the code that goes with that name, substitute the concrete values we provided for the parameters, and return the result of evaluating the resulting expression. Pyret will NOT produce anything in the interactions pane for a function definition (other than a report about whether the examples hold).

There’s much more to learn about functions, including different reasons for creating them. We’ll get to those in due course.

#### 3.3.1 Example: Similar Flags "Link to here")

Consider the following two expressions to draw the flags of Armenia and Austria (respectively). These two countries have the same flag, just with different colors. The `frame` operator draws a small black frame around the image.

```

# 以#开头的行是供人类阅读的注释。

# Pyret 忽略每行#号之后的内容。

# armenia

frame(

above(rectangle(120, 30, "solid", "red"),

    above(rectangle(120, 30, "solid", "blue"),

    rectangle(120, 30, "solid", "orange"))))

# austria

frame(

above(rectangle(120, 30, "solid", "red"),

    above(rectangle(120, 30, "solid", "white"),

    rectangle(120, 30, "solid", "red"))))

```py

Rather than write this program twice, it would be nice to write the common expression only once, then just change the colors to generate each flag. Concretely, we’d like to have a custom operator such as `three-stripe-flag` that we could use as follows:

```

# armenia

three-stripe-flag("red", "blue", "orange")

# austria

three-stripe-flag("red", "white", "red")

```py

In this program, we provide `three-stripe-flag` only with the information that customizes the image creation to a specific flag. The operation itself would take care of creating and aligning the rectangles. We want to end up with the same images for the Armenian and Austrian flags as we would have gotten with our original program. Such an operator doesn’t exist in Pyret: it is specific only to our application of creating flag images. To make this program work, then, we need the ability to add our own operators (henceforth called functions) to Pyret.

#### 3.3.2 Defining Functions "Link to here")

In programming, a function takes one or more (configuration) parameters and uses them to produce a result.

> Strategy: Creating Functions From Expressions
> 
> > If we have multiple concrete expressions that are identical except for a couple of specific data values, we create a function with the common code as follows:
> > 
> > *   Write down at least two expressions showing the desired computation (in this case, the expressions that produce the Armenian and Austrian flags).
> >     
> >     
> > *   Identify which parts are fixed (i.e., the creation of rectangles with dimensions `120` and `30`, the use of `above` to stack the rectangles) and which are changing (i.e., the stripe colors).
> >     
> >     
> > *   For each changing part, give it a name (say `top`, `middle`, and `bottom`), which will be the parameter that stands for that part.
> >     
> >     
> > *   Rewrite the examples to be in terms of these parameters. For example:
> >     
> >     
> >     
> >     ```

> >     frame(
> >     
> >     above(rectangle(120, 30, "solid", top),
> >     
> >         above(rectangle(120, 30, "solid", middle),
> >         
> >         rectangle(120, 30, "solid", bottom))))
> >         
> >     ```py
> >     
> >     
> > *   Name the function something suggestive: e.g., `three-stripe-flag`.
> >     
> >     
> > *   Write the syntax for functions around the expression:
> >     
> >     
> >     
> >     ```
> >     
> >     fun <function name>(<parameters>):
> >     
> >     <表达式在这里>
> >     
> >     end
> >     
> >     ```py
> >     
> >     
> >     
> >     where the expression is called the body of the function. (Programmers often use angle brackets to say “replace with something appropriate”; the brackets themselves aren’t part of the notation.)

Here’s the end product:

```

fun three-stripe-flag(top, middle, bottom):

frame(

    above(rectangle(120, 30, "solid", top),

    above(rectangle(120, 30, "solid", middle),

        rectangle(120, 30, "solid", bottom))))

end

```py

While this looks like a lot of work now, it won’t once you get used to it. We will go through the same steps over and over, and eventually they’ll become so intuitive that you won’t need to start from multiple similar expressions.

> Do Now!
> 
> > Why does the function body have only one expression, when before we had a separate one for each flag?

We have only one expression because the whole point was to get rid of all the changing parts and replace them with parameters.

With this function in hand, we can write the following two expressions to generate our original flag images:

```

three-stripe-flag("red", "blue", "orange")

three-stripe-flag("red", "white", "red")

```py

When we provide values for the parameters of a function to get a result, we say that we are calling the function. We use the term call for expressions of this form.

If we want to name the resulting images, we can do so as follows:

```

armenia = three-stripe-flag("red", "blue", "orange")

austria = three-stripe-flag("red", "white", "red")

```py

(Side note: Pyret only allows one value per name in the directory. If your file already had definitions for the names `armenia` or `austria`, Pyret will give you an error at this point. You can use a different name (like `austria2`) or comment out the original definition using `#`.)

##### 3.3.2.1 How Functions Evaluate "Link to here")

So far, we have learned three rules for how Pyret processes your program:

*   If you write an expression, Pyret evaluates it to produce its value.

*   If you write a statement that defines a name, Pyret evaluates the expression (right side of `=`), then makes an entry in the directory to associate the name with the value.

*   If you write an expression that uses a name from the directory, Pyret substitutes the name with the corresponding value.

Now that we can define our own functions, we have to consider two more cases: what does Pyret do when you define a function (using `fun`), and what does Pyret do when you call a function (with values for the parameters)?

*   When Pyret encounters a function definition in your file, it makes an entry in the directory to associate the name of the function with its code. The body of the function does not get evaluated at this time.

*   When Pyret encounters a function call while evaluating an expression, it replaces the call with the body of the function, but with the parameter values substituted for the parameter names in the body. Pyret then continues to evaluate the body with the substituted values.

As an example of the function-call rule, if you evaluate

```

three-stripe-flag("red", "blue", "orange")

```py

Pyret starts from the function body

```

frame(

above(rectangle(120, 30, "solid", top),

    above(rectangle(120, 30, "solid", middle),

    rectangle(120, 30, "solid", bottom))))

```py

substitutes the parameter values

```

frame(

above(rectangle(120, 30, "solid", "red"),

    above(rectangle(120, 30, "solid", "blue"),

    rectangle(120, 30, "solid", "orange"))))

```py

then evaluates the expression, producing the flag image.

Note that the second expression (with the substituted values) is the same expression we started from for the Armenian flag. Substitution restores that expression, while still allowing the programmer to write the shorthand in terms of `three-stripe-flag`.

##### 3.3.2.2 Type Annotations "Link to here")

What if we made a mistake, and tried to call the function as follows:

```

three-stripe-flag(50, "blue", "red")

```py

> Do Now!
> 
> > What do you think Pyret will produce for this expression?

The first parameter to `three-stripe-flag` is supposed to be the color of the top stripe. The value `50` is not a string (much less a string naming a color). Pyret will substitute `50` for `top` in the first call to `rectangle`, yielding the following:

```

frame(

above(rectangle(120, 30, "solid", 50),

    above(rectangle(120, 30, "solid", "blue"),

    rectangle(120, 30, "solid", "red"))))

```py

When Pyret tries to evaluate the `rectangle` expression to create the top stripe, it generates an error that refers to that call to `rectangle`.

If someone else were using your function, this error might not make sense: they didn’t write an expression about rectangles. Wouldn’t it be better to have Pyret report that there was a problem in the use of `three-stripe-flag` itself?

As the author of `three-stripe-flag`, you can make that happen by annotating the parameters with information about the expected type of value for each parameter. Here’s the function definition again, this time requiring the three parameters to be strings:

```

fun three-stripe-flag(top :: String,

    middle :: String,

    bottom :: String):

frame(

    above(rectangle(120, 30, "solid", top),

    above(rectangle(120, 30, "solid", middle),

        rectangle(120, 30, "solid", bottom))))

end

```py

Notice that the notation here is similar to what we saw in contracts within the documentation: the parameter name is followed by a double-colon (`::`) and a type name (so far, one of `Number`, `String`, or `Image`).Putting each parameter on its own line is not required, but it sometimes helps with readability.

Run your file with this new definition and try the erroneous call again. You should get a different error message that is just in terms of `three-stripe-flag`.

It is also common practice to add a type annotation that captures the type of the function’s output. That annotation goes after the list of parameters:

```

fun three-stripe-flag(top :: String,

    middle :: String,

    bottom :: String) -> Image:

frame(

    above(rectangle(120, 30, "solid", top),

    above(rectangle(120, 30, "solid", middle),

        rectangle(120, 30, "solid", bottom))))

end

```py

Note that all of these type annotations are optional. Pyret will run your program whether or not you include them. You can put type annotations on some parameters and not others; you can include the output type but not any of the parameter types. Different programming languages have different rules about types.

We will think of types as playing two roles: giving Pyret information that it can use to focus error messages more accurately, and guiding human readers of programs as to the proper use of user-defined functions.

##### 3.3.2.3 Documentation "Link to here")

Imagine that you opened your program file from this chapter a couple of months from now. Would you remember what computation `three-stripe-flag` does? The name is certainly suggestive, but it misses details such as that the stripes are stacked vertically (rather than horizontally) and that the stripes are equal height. Function names aren’t designed to carry this much information.

Programmers also annotate a function with a docstring, a short, human-language description of what the function does. Here’s what the Pyret docstring might look like for `three-stripe-flag`:

```

fun three-stripe-flag(top :: String,

    middle :: String,

    bottom :: String) -> Image:

doc: "生成三条等高水平条纹的旗帜图像"

frame(

    above(rectangle(120, 30, "solid", top),

    above(rectangle(120, 30, "solid", middle),

        rectangle(120, 30, "solid", bottom))))

end

```py

While docstrings are also optional from Pyret’s perspective, you should always provide one when you write a function. They are extremely helpful to anyone who has to read your program, whether that is a co-worker, grader…or yourself, a couple of weeks from now.

##### 3.3.2.1 How Functions Evaluate "Link to here")

So far, we have learned three rules for how Pyret processes your program:

*   If you write an expression, Pyret evaluates it to produce its value.

*   If you write a statement that defines a name, Pyret evaluates the expression (right side of `=`), then makes an entry in the directory to associate the name with the value.

*   If you write an expression that uses a name from the directory, Pyret substitutes the name with the corresponding value.

Now that we can define our own functions, we have to consider two more cases: what does Pyret do when you define a function (using `fun`), and what does Pyret do when you call a function (with values for the parameters)?

*   When Pyret encounters a function definition in your file, it makes an entry in the directory to associate the name of the function with its code. The body of the function does not get evaluated at this time.

*   When Pyret encounters a function call while evaluating an expression, it replaces the call with the body of the function, but with the parameter values substituted for the parameter names in the body. Pyret then continues to evaluate the body with the substituted values.

As an example of the function-call rule, if you evaluate

```

three-stripe-flag("red", "blue", "orange")

```py

Pyret starts from the function body

```

frame(

above(rectangle(120, 30, "solid", top),

    above(rectangle(120, 30, "solid", middle),

    rectangle(120, 30, "solid", bottom))))

```py

substitutes the parameter values

```

frame(

above(rectangle(120, 30, "solid", "red"),

    above(rectangle(120, 30, "solid", "blue"),

    rectangle(120, 30, "solid", "orange"))))

```py

then evaluates the expression, producing the flag image.

Note that the second expression (with the substituted values) is the same expression we started from for the Armenian flag. Substitution restores that expression, while still allowing the programmer to write the shorthand in terms of `three-stripe-flag`.

##### 3.3.2.2 Type Annotations "Link to here")

What if we made a mistake, and tried to call the function as follows:

```

three-stripe-flag(50, "blue", "red")

```py

> Do Now!
> 
> > What do you think Pyret will produce for this expression?

The first parameter to `three-stripe-flag` is supposed to be the color of the top stripe. The value `50` is not a string (much less a string naming a color). Pyret will substitute `50` for `top` in the first call to `rectangle`, yielding the following:

```

frame(

above(rectangle(120, 30, "solid", 50),

    above(rectangle(120, 30, "solid", "blue"),

    rectangle(120, 30, "solid", "red"))))

```py

When Pyret tries to evaluate the `rectangle` expression to create the top stripe, it generates an error that refers to that call to `rectangle`.

If someone else were using your function, this error might not make sense: they didn’t write an expression about rectangles. Wouldn’t it be better to have Pyret report that there was a problem in the use of `three-stripe-flag` itself?

As the author of `three-stripe-flag`, you can make that happen by annotating the parameters with information about the expected type of value for each parameter. Here’s the function definition again, this time requiring the three parameters to be strings:

```

fun three-stripe-flag(top :: String,

    middle :: String,

    bottom :: String):

frame(

    above(rectangle(120, 30, "solid", top),

    above(rectangle(120, 30, "solid", middle),

        rectangle(120, 30, "solid", bottom))))

end

```py

Notice that the notation here is similar to what we saw in contracts within the documentation: the parameter name is followed by a double-colon (`::`) and a type name (so far, one of `Number`, `String`, or `Image`).Putting each parameter on its own line is not required, but it sometimes helps with readability.

Run your file with this new definition and try the erroneous call again. You should get a different error message that is just in terms of `three-stripe-flag`.

It is also common practice to add a type annotation that captures the type of the function’s output. That annotation goes after the list of parameters:

```

fun three-stripe-flag(top :: String,

    middle :: String,

    bottom :: String) -> Image:

frame(

    above(rectangle(120, 30, "solid", top),

    above(rectangle(120, 30, "solid", middle),

        rectangle(120, 30, "solid", bottom))))

end

```py

Note that all of these type annotations are optional. Pyret will run your program whether or not you include them. You can put type annotations on some parameters and not others; you can include the output type but not any of the parameter types. Different programming languages have different rules about types.

We will think of types as playing two roles: giving Pyret information that it can use to focus error messages more accurately, and guiding human readers of programs as to the proper use of user-defined functions.

##### 3.3.2.3 Documentation "Link to here")

Imagine that you opened your program file from this chapter a couple of months from now. Would you remember what computation `three-stripe-flag` does? The name is certainly suggestive, but it misses details such as that the stripes are stacked vertically (rather than horizontally) and that the stripes are equal height. Function names aren’t designed to carry this much information.

Programmers also annotate a function with a docstring, a short, human-language description of what the function does. Here’s what the Pyret docstring might look like for `three-stripe-flag`:

```

fun three-stripe-flag(top :: String,

    middle :: String,

    bottom :: String) -> Image:

doc: "生成三条等高水平条纹的旗帜图像"

frame(

    above(rectangle(120, 30, "solid", top),

    above(rectangle(120, 30, "solid", middle),

        rectangle(120, 30, "solid", bottom))))

end

```py

While docstrings are also optional from Pyret’s perspective, you should always provide one when you write a function. They are extremely helpful to anyone who has to read your program, whether that is a co-worker, grader…or yourself, a couple of weeks from now.

#### 3.3.3 Functions Practice: Moon Weight "Link to here")

Suppose we’re responsible for outfitting a team of astronauts for lunar exploration. We have to determine how much each of them will weigh on the Moon’s surface. On the Moon, objects weigh only one-sixth their weight on earth. Here are the expressions for several astronauts (whose weights are expressed in pounds):

```

100 * 1/6

150 * 1/6

90 * 1/6

```py

As with our examples of the Armenian and Austrian flags, we are writing the same expression multiple times. This is another situation in which we should create a function that takes the changing data as a parameter but captures the fixed computation only once.

In the case of the flags, we noticed we had written essentially the same expression more than once. Here, we have a computation that we expect to do multiple times (once for each astronaut). It’s boring to write the same expression over and over again. Besides, if we copy or re-type an expression multiple times, sooner or later we’re bound to make a transcription error.This is an instance of the [DRY principle](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself), where DRY means "don’t repeat yourself".

Let’s remind ourselves of the steps for creating a function:

*   Write down some examples of the desired calculation. We did that above.

*   Identify which parts are fixed (above, `* 1/6`) and which are changing (above, `100`, `150`, `90`...).

*   For each changing part, give it a name (say `earth-weight`), which will be the parameter that stands for it.

*   Rewrite the examples to be in terms of this parameter:

    ```

    地球重量 * 1/6

    ```py

    This will be the body, i.e., the expression inside the function.

*   Come up with a suggestive name for the function: e.g., `moon-weight`.

*   Write the syntax for functions around the body expression:

    ```

    fun moon-weight(earth-weight):

    earth-weight * 1/6

    end

    ```py

*   Remember to include the types of the parameter and output, as well as the documentation string. This yields the final function:

    ```

    fun moon-weight(earth-weight :: Number) -> Number:

    doc: "从地球重量计算月球重量"

    earth-weight * 1/6

    end

    ```py

#### 3.3.4 Documenting Functions with Examples "Link to here")

In each of the functions above, we’ve started with some examples of what we wanted to compute, generalized from there to a generic formula, turned this into a function, and then used the function in place of the original expressions.

Now that we’re done, what use are the initial examples? It seems tempting to toss them away. However, there’s an important rule about software that you should learn: Software Evolves. Over time, any program that has any use will change and grow, and as a result may end up producing different values than it did initially. Sometimes these are intended, but sometimes these are a result of mistakes (including such silly but inevitable mistakes like accidentally adding or deleting text while typing). Therefore, it’s always useful to keep those examples around for future reference, so you can immediately be alerted if the function deviates from the examples it was supposed to generalize.

Pyret makes this easy to do. Every function can be accompanied by a `where` clause that records the examples. For instance, our `moon-weight` function can be modified to read:

```

fun moon-weight(earth-weight :: Number) -> Number:

doc: "从地球重量计算月球重量"

earth-weight * 1/6

where:

moon-weight(100) is 100 * 1/6

moon-weight(150) is 150 * 1/6

moon-weight(90) is 90 * 1/6

end

```py

When written this way, Pyret will actually check the answers every time you run the program, and notify you if you have changed the function to be inconsistent with these examples.

> Do Now!
> 
> > Check this! Change the formula—<wbr>for instance, replace the body of the function with
> > 
> > ```

> > earth-weight * 1/3
> > 
> > ```py
> > 
> > —<wbr>and see what happens. Pay attention to the output from CPO: you should get used to recognizing this kind of output.

> Do Now!
> 
> > Now, fix the function body, and instead change one of the answers—<wbr>e.g., write
> > 
> > ```
> > 
> > moon-weight(90) is 90 * 1/3
> > 
> > ```py
> > 
> > —<wbr>and see what happens. Contrast the output in this case with the output above.

Of course, it’s pretty unlikely you will make a mistake with a function this simple (except through a typo). After all, the examples are so similar to the function’s own body. Later, however, we will see that the examples can be much simpler than the body, and there is a real chance for things to get inconsistent. At that point, the examples become invaluable in making sure we haven’t made a mistake in our program. In fact, this is so valuable in professional software development that good programmers always write down large collections of examples—<wbr>called tests—<wbr>to make sure their programs are behaving as they expect.

For our purposes, we are writing examples as part of the process of making sure we understand the problem. It’s always a good idea to make sure you understand the question before you start writing code to solve a problem. Examples are a nice intermediate point: you can sketch out the relevant computation on concrete values first, then worry about turning it into a function. If you can’t write the examples, chances are you won’t be able to write the function either. Examples break down the programming process into smaller, manageable steps.

#### 3.3.5 Functions Practice: Cost of pens "Link to here")

Let’s create one more function, this time for a more complicated example. Imagine that you are trying to compute the total cost of an order of pens with slogans (or messages) printed on them. Each pen costs 25 cents plus an additional 2 cents per character in the message (we’ll count spaces between words as characters).

Following our steps to create a function once again, let’s start by writing two concrete expressions that do this computation.

```

# ordering 3 pens that say "wow"

3 * (0.25 + (string-length("wow") * 0.02))

# ordering 10 pens that say "smile"

10 * (0.25 + (string-length("smile") * 0.02))

```py

These examples introduce a new built-in function called `string-length`. It takes a string as input and produces the number of characters (including spaces and punctuation) in the string. These examples also show an example of working with numbers other than integers.Pyret requires a number before the decimal point, so if the “whole number” part is zero, you need to write `0` before the decimal. Also observe that Pyret uses a decimal point; it doesn’t support conventions such as [“0,02”](https://en.wikipedia.org/wiki/Decimal_separator).

The second step to writing a function was to identify which information differs across our two examples. In this case, we have two: the number of pens and the message to put on the pens. This means our function will have two parameters rather than just one.

```

fun pen-cost(num-pens :: Number, message :: String) -> Number:

num-pens * (0.25 + (string-length(message) * 0.02))

end

```py

Of course, as things get too long, it may be helpful to use multiple lines:

```

fun pen-cost(num-pens :: Number, message :: String)

-> Number:

num-pens * (0.25 + (string-length(message) * 0.02))

end

```py

If you want to write a multi-line docstring, you need to use ```` ```py ````而不是`"`来开始和结束，如下所示：

```py
fun pen-cost(num-pens :: Number, message :: String)
  -> Number:
  doc: ```笔的总成本，每支 25 美分

    每个消息字符额外加 2 美分```py
  num-pens * (0.25 + (string-length(message) * 0.02))
end
```

我们还应该记录我们创建函数时使用的示例：

```py
fun pen-cost(num-pens :: Number, message :: String)
  -> Number:
  doc: ```笔的总成本，每支 25 美分

    每个消息字符额外加 2 美分```py
  num-pens * (0.25 + (string-length(message) * 0.02))
where:
  pen-cost(3, "wow")
    is 3 * (0.25 + (string-length("wow") * 0.02))
  pen-cost(10, "smile")
    is 10 * (0.25 + (string-length("smile") * 0.02))
end
```

当编写`where`示例时，我们还想包括函数可能需要处理的一些特殊但有效的案例，例如空消息。

```py
pen-cost(5, "") is 5 * 0.25
```

注意，我们的空消息示例在`is`右侧有一个更简单的表达式。函数返回的表达式不必与主体表达式匹配；它只需评估为与示例预期产生的值相同。有时，我们会发现直接写出预期值更容易。例如，对于某人订购零支笔的情况，我们会包括：

```py
pen-cost(0, "bears") is 0
```

这些例子的目的是记录函数在多种输入下的行为。`is`右侧的内容应该以某种有意义的方式总结计算或答案。最重要的是？不要编写函数，运行它以确定答案，然后将该答案放在`is`的右侧！为什么不呢？因为示例旨在为设计过程提供一些冗余，以便你能够捕捉到你可能犯的错误。如果你的函数体不正确，并且你使用该函数生成示例，你将无法从使用示例来检查错误中获得好处。

我们将不断回到编写良好示例的想法。现在如果你还有问题，不用担心。此外，目前我们不会担心像负数笔这样的无意义情况。在我们学习了可以帮助我们正确处理这些情况的额外编码技术之后，我们将处理这些问题。

> 现在行动！
> 
> > 我们本可以将两个特殊情况合并成一个例子，例如
> > 
> > ```py
> > pen-cost(0, "") is 0
> > ```
> > 
> > 做这件事看起来是个好主意吗？为什么是或不是？

#### 3.3.6 回顾：定义函数 "链接到这里")

本章介绍了函数的概念。函数在编程中扮演着关键角色：它们让我们能够在不同时间使用不同的具体值来配置计算。第一次计算笔的成本时，我们可能是在询问关于`10`支写着“欢迎”的笔。下一次，我们可能是在询问关于`100`支写着“加油熊！”的笔。这两种情况的核心计算是相同的，因此我们希望只写一次，每次使用时都配置不同的具体值。

我们已经介绍了关于函数的几个具体想法：

+   我们展示了用于编写函数的`fun`符号。你了解到一个函数有一个名称（我们可以用它来引用它），一个或多个参数（我们想要配置的值的名称），以及一个主体，即一旦我们为参数提供了具体值，我们想要执行的计算。

+   我们展示了我们应该在我们的函数中包含示例，以说明函数在各个特定值上的计算结果。示例位于函数内的`where`块中。

+   我们展示了我们可以通过为参数提供具体值来使用函数。为此，我们写下我们想要使用的函数名称，然后是参数的逗号分隔值对，用一对括号括起来。例如，在交互式提示符中写下以下表达式将计算特定笔订单的成本：

    ```py
    pen-cost(10, "Welcome")
    ```

+   我们讨论了，如果我们定义一个函数在定义面板中，然后按运行，Pyret 将在目录中为该函数创建一个条目。如果我们稍后使用该函数，Pyret 将查找与该名称相关的代码，用我们提供的具体值替换参数，并返回评估结果的值。Pyret 在交互式面板中不会为函数定义产生任何内容（除了报告示例是否成立）。

关于函数，还有很多东西要学习，包括创建它们的不同原因。我们将在适当的时候讨论这些内容。
