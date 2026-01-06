# 第六章 函数

> 原文：[`randpythonbook.netlify.app/functions`](https://randpythonbook.netlify.app/functions)

此文本已经介绍了如何*使用*预制的函数。至少我们已经讨论了如何一次性使用它们——只需写出函数名，在该名称后写一些括号，然后通过在两个括号之间以逗号分隔的方式写入任何必需的参数来插入。这在 R 和 Python 中都是这样工作的。

在本节中，我们将探讨如何*定义*我们自己的函数。这不仅有助于我们理解预制的函数，而且如果我们需要一些尚未提供的额外功能，这也会很有用。

编写我们自己的函数也有助于“打包”计算。这种实用性很快就会变得明显。考虑估计回归模型的任务。如果你有一个执行所有必需计算的功能，那么

+   你可以估算模型，无需考虑底层细节或自己编写任何代码，并且

+   你可以在任何项目中对任何数据集上的任何模型进行拟合时重复使用此函数。

## 6.1 定义 R 函数

要在 R 中创建一个函数，我们需要另一个名为 `function()` 的函数。我们以与在 R 中给任何其他变量命名相同的方式给 `function()` 的输出命名，即使用赋值运算符 `<-` 。以下是一个名为 `addOne()` 的玩具函数的示例。在这里，`myInput` 是一个占位符，它指的是函数用户最终插入的任何内容。

```py
addOne <-  function(myInput){  # define the function
 myOutput <-  myInput +  1
 return(myOutput)
}
addOne(41) # call/invoke/use the function 
## [1] 42
```

在定义下方，函数使用 `41` 作为输入被调用。当发生这种情况时，以下事件序列发生

+   值 `41` 被分配给 `myInput`

+   `myOutput` 被赋予值 `42`

+   `myOutput`，其值为 `42`，将从函数返回

+   临时变量 `myInput` 和 `myOutput` 被销毁。

我们得到所需的答案，并且所有不必要的中间变量在不再需要后都会被清理并丢弃。

如果你对编写函数感兴趣，我建议你首先在函数外部编写逻辑。由于你的临时变量在最终结果获得后不会被销毁，因此初始代码将更容易调试。一旦你对工作代码满意，你就可以将逻辑复制并粘贴到函数定义中，并用函数输入（如上面的 `myInput`）替换永久变量。

## 6.2 定义 Python 函数

要在 Python 中创建一个函数，我们使用 `def` 语句（而不是 R 中的 `function()` 函数）。函数的期望名称接下来。然后是形式参数，以逗号分隔的方式放在括号内，就像在 R 中一样。

在 Python 中定义函数稍微简洁一些。与 R 不同，没有赋值运算符，没有花括号，`return`也不是像 R 中的函数那样，因此不需要在其后使用括号。尽管如此，有一个语法上的添加——在定义的第一行末尾需要有一个冒号（`:`）。

下面是一个名为`add_one()`的玩具函数的例子。

```py
def add_one(my_input):  # define the function
 my_output = my_input + 1
 return my_output
add_one(41) # call/invoke/use the function 
## 42
```

在定义下方，函数使用`41`作为输入被调用。当发生这种情况时，以下一系列事件发生

+   将值`41`赋给`my_input`

+   `my_output`被赋予值`42`

+   `my_output`，其值为`42`，从函数中返回

+   临时变量`my_input`和`my_output`被销毁。

我们得到了期望的答案，并且所有不必要的中间变量在不再需要后都被清理并丢弃。

## 6.3 R 的自定义函数的更多细节

在技术上，在 R 中，函数被[定义为三个捆绑在一起的东西](https://cran.r-project.org/doc/manuals/r-release/R-lang.html#Function-objects)：

1.  一个**形式参数列表**（也称为*形式*），

1.  一个**主体**，和

1.  一个**父环境**。

*形式参数列表*正如其名。它是函数接受的参数列表。你可以使用`formals()`函数访问函数的形式参数列表。请注意，它不是用户将插入的*实际*参数——在函数最初创建时这是不可知的。

下面是另一个接受一个名为`whichNumber`的参数的函数，该参数有一个**默认参数**为`1`。如果函数的调用者没有指定她想要添加到`myInput`的内容，`addNumber()`将使用`1`作为默认值。这个默认值出现在`formals(addNumber)`的输出中。

```py
addNumber <-  function(myInput, whichNumber = 1){ 
 myOutput <-  myInput +  whichNumber
 return(myOutput)
}
addNumber(3) # no second argument being provided by the user here
## [1] 4
formals(addNumber)
## $myInput
## 
## 
## $whichNumber
## [1] 1
```

函数的*主体*正如其名。它是函数执行的工作。你可以使用`body()`函数访问函数的主体。

```py
addNumber <-  function(myInput, whichNumber = 1){ 
 myOutput <-  myInput +  whichNumber
 return(myOutput)
}
body(addNumber)
## {
##     myOutput <- myInput + whichNumber
##     return(myOutput)
## }
```

你创建的每个函数也都有一个*父环境*¹⁰。你可以使用`environment()`函数获取/设置它。环境帮助函数知道它可以使用哪些变量以及如何使用它们。函数的父环境是函数被*创建*的地方，它包含函数也可以使用的变量。函数可以使用哪些变量的规则称为*作用域*。当你使用 R 创建函数时，你主要使用**词法作用域**。这将在第 6.5 节（/functions#function-scope-in-r）中更详细地讨论。

关于环境的信息还有很多，这里没有提供。例如，用户定义的函数也有[绑定、执行和调用环境与之相关](http://adv-r.had.co.nz/Environments.html#function-envs)，并且环境用于创建包命名空间，这在两个包各自有一个同名函数时非常重要。

## 6.4 Python 用户定义函数的更多细节

大概来说，Python 函数与 R 函数有相同的东西。它们有**形式参数列表**、主体，并且创建了[命名空间](https://docs.python.org/3/tutorial/classes.html#python-scopes-and-namespaces)，这有助于组织函数可以访问的变量，以及哪些代码可以调用这个新函数。命名空间只是“从名称到对象的映射。”

这三个概念与 R 中的类似，名称有时略有不同，并且组织方式也不尽相同。要访问这些信息片段，您需要访问函数的*特殊属性*。Python 中的用户定义函数附带了大量的信息。如果您想查看所有这些信息，可以访问[这个文档页面](https://docs.python.org/3/reference/datamodel.html#objects-values-and-types)。

例如，让我们尝试找到用户定义函数的*形式参数列表*。这又是函数接受的输入集合。就像在 R 中一样，这并不是用户将插入的*实际*参数——在函数创建时这是不可知的。¹¹ 这里我们还有一个名为`add_number()`的函数，它接受一个名为`which_number`的参数，该参数有一个默认参数`1`。

```py
def add_number(my_input, which_number = 1): # define a function
 my_output = my_input + which_number
 return my_output
add_number(3) # no second argument being provided by the user here
## 4
add_number.__code__.co_varnames # note this also contains *my_output*
## ('my_input', 'which_number', 'my_output')
add_number.__defaults__
## (1,)
```

`__code__`属性提供了更多功能。要查看其内容的所有名称列表，可以使用`dir(add_number.__code__)`。

如果`add_number.__code__`的表示法看起来很奇怪，请不要担心。点(`.`)操作符将在未来的面向对象编程章节中变得更加清晰。现在，只需将`__code__`视为属于`add_number`的对象。属于其他对象的对象在 Python 中被称为**属性**。点操作符帮助我们访问其他对象内部的属性。它还帮助我们访问我们`导入`到脚本中的模块所属的对象。

## 6.5 R 中的函数作用域

R 使用**词法作用域**。这意味着，在 R 中，

1.  函数可以使用在其内部定义的*局部*变量，

1.  函数可以使用在函数自身被*定义*的环境中定义的*全局*变量，并且

1.  函数不一定可以使用在函数被*调用*的环境中定义的*全局*变量，并且

1.  如果存在名称冲突，函数将优先使用*局部*变量。

第一个特征很明显。第二个和第三个特征很重要，需要区分。考虑以下代码。`sillyFunction()`可以访问`a`，因为`sillyFunction()`和`a`是在同一个地方定义的。

```py
a <-  3
sillyFunction <-  function(){
 return(a +  20) 
}
environment(sillyFunction) # the env. it was defined in contains a
## <environment: R_GlobalEnv>
sillyFunction()
## [1] 23
```

另一方面，以下示例将无法工作，因为`a`和`anotherSillyFunc()`不是在同一个地方定义的。调用函数与定义函数并不相同。

```py
anotherSillyFunc <-  function(){
 return(a +  20) 
}
highLevelFunc <-  function(){
 a <-  99
 # this isn't the global environment anotherSillyFunc() was defined in
 cat("environment inside highLevelFunc(): ", environment())
 anotherSillyFunc()
}
```

最后，这里有一个函数优先选择一个`a`而不是另一个的演示。当`sillyFunction()`尝试访问`a`时，它首先在其自身体内查找，因此最内层的被使用。另一方面，`print(a)`显示的是全局变量`3`。

```py
a <-  3
sillyFunction <-  function(){
 a <-  20
 return(a +  20) 
}
sillyFunction()
## [1] 40
print(a)
## [1] 3
```

如果你在函数内部创建函数，同样的概念也适用。内部函数`innerFunc()`会“由内向外”寻找变量，但只在其定义的地方寻找。

下面我们调用`outerFunc()`，然后它调用`innerFunc()`。`innerFunc()`可以引用变量`b`，因为它位于与`innerFunc()`创建时相同的环境中。有趣的是，`innerFunc()`也可以引用变量`a`，因为该变量被`outerFunc()`捕获，这为`innerFunc()`提供了访问权限。

```py
a <- "outside both"
outerFunc <-  function(){
 b <- "inside one"
 innerFunc <-  function(){
 print(a) 
 print(b)
 }
 return(innerFunc())
}
outerFunc()
## [1] "outside both"
## [1] "inside one"
```

这里有一个有趣的例子。如果我们要求`outerFunc()`返回函数`innerFunc()`（而不是`innerFunct()`的返回对象……函数也是对象！），我们可能会惊讶地看到`innerFunc()`仍然可以成功引用`b`，即使它不存在于**调用环境**中。但不要感到惊讶！重要的是函数**创建**时可用的情况。

```py
outerFuncV2 <-  function(){
 b <- "inside one"
 innerFunc <-  function(){
 print(b)
 }
 return(innerFunc) # note the missing inner parentheses!
}
myFunc <-  outerFuncV2() # get a new function
ls(environment(myFunc)) # list all data attached to this function
## [1] "b"         "innerFunc"
myFunc()
## [1] "inside one"
```

我们在创建返回其他函数的函数时经常使用这个特性。这将在第十五章中更详细地讨论。在上面的例子中，返回另一个函数的函数`outerFuncV2()`被称为**函数工厂**。

人们有时会将 R 的函数称为**闭包**，以强调它们从创建它们的父环境中捕获变量，以强调它们捆绑的数据。

## 6.6 Python 中的函数作用域

Python 使用与 R 相同的**词法作用域**。这意味着，在 Python 中，

1.  函数可以使用定义在其内部的*局部*变量。

1.  函数在名称冲突的情况下有一个优先级顺序来决定选择哪个变量，并且

1.  函数有时可以使用自身外部定义的变量，但这种能力取决于函数和变量在哪里**定义**，而不是函数在哪里**调用**。

关于特性（2）和（3），有一个著名的缩写词描述了 Python 在查找和选择变量时遵循的规则：**LEGB**。

+   L: 局部，

+   E: 封闭的，

+   G: 全局，并且

+   B: 内置的。

Python 函数将按照以下顺序在这些命名空间中搜索变量。¹²。

“*局部*”指的是在函数的代码块内部定义的变量。下面的函数使用局部变量`a`而不是全局变量。

```py
a = 3
def silly_function():
 a = 22 # local a
 print("local variables are ", locals())
 return a + 20
silly_function()
## local variables are  {'a': 22}
## 42
silly_function.__code__.co_nlocals # number of local variables
## 1
silly_function.__code__.co_varnames # names of local variables
## ('a',)
```

“*封装*”指的是在封装命名空间中定义的变量，而不是全局命名空间中的变量。这些变量有时被称为**自由变量**。在下面的例子中，`inner_func()`没有局部`a`变量，但有一个全局变量和一个封装命名空间中的变量。`inner_func()`选择了封装命名空间中的那个。此外，`inner_func()`还有一个自己的`a`副本来使用，即使`a`在`outer_func()`调用完成后最初被销毁。

```py
a = "outside both"
def outer_func():
 a = "inside one"
 def inner_func():
 print(a)
 return inner_func
my_new_func = outer_func()
my_new_func()
## inside one
my_new_func.__code__.co_freevars
## ('a',)
```

“*全局*”作用域包含在模块级命名空间中定义的变量。如果下面示例中的代码是整个脚本的全部，那么`a`将是一个全局变量。

```py
a = "outside both"
def outer_func():
 b = "inside one"
 def inner_func():
 print(a) 
 inner_func()
outer_func()
## outside both
```

就像在 R 中一样，Python 函数**不一定**能找到函数被**调用**时的变量。例如，以下是一些模仿上述 R 示例的代码。`a`和`b`都可以在`inner_func()`内部访问。这是由于 LEGB 规则。

然而，如果我们开始在另一个函数内部使用`outer_func()`，在另一个函数中**调用**它，而它是在别处**定义**的，那么它就无法访问调用点的变量。你可能会对以下代码的功能感到惊讶。这是否会打印出正确的字符串：“我现在想使用的 a！”？不！

```py
a = "outside both"
def outer_func():
 b = "inside one"
 def inner_func():
 print(a) 
 print(b)
 return inner_func() 
def third_func():
 a = "this is the a I want to use now!"
 outer_func()
third_func() 
```

```py
## outside both
## inside one
```

如果你觉得自己已经理解了词法作用域，那就太好了！那么你应该准备好去学习第十五章了。如果没有，那就继续玩转这些例子。如果不理解 R 和 Python 共享的作用域规则，编写你自己的函数将始终感觉比实际要困难得多。

## 6.7 修改函数的参数

我们能否/应该修改函数的参数？这样做带来的灵活性听起来很有力量；然而，不这样做是推荐的，因为它使得程序更容易推理。

### 6.7.1 R 中的按值传递

在 R 中，函数修改其参数之一是**困难的**。¹³考虑以下代码。

```py
a <-  1
f <-  function(arg){
 arg <-  2 # modifying a temporary variable, not a
 return(arg)
}
print(f(a))
## [1] 2
print(a)
## [1] 1
```

函数`f`有一个名为`arg`的参数。当执行`f(a)`时，会修改`a`的一个**副本**。当函数在其体内构造所有输入变量的副本时，这被称为**按值传递**的语义。这个副本是一个临时中间值，它只为函数提供一个起点，以产生返回值`2`。

`arg`原本可以被称为`a`，并且将发生相同的行为。然而，给这两者不同的名字有助于提醒你和他人 R 是复制其参数的。

仍然可以修改`a`，但我也不推荐这样做。我将在子节 6.7 中进一步讨论这个问题。

### 6.7.2 Python 中的按赋值传递

在 Python 中，情况更为复杂。Python 函数具有**按赋值传递**的语义。这是 Python 非常独特的地方。这意味着你修改函数参数的能力取决于

+   争论的类型是什么，以及

+   你试图对其做什么。

我们将首先通过一些例子，然后解释为什么它会以这种方式工作。这里有一些与上面例子相似的代码。

```py
a = 1
def f(arg):
 arg = 2
 return arg
print(f(a))
## 2
print(a)
## 1
```

在这种情况下，`a` 没有被修改。这是因为 `a` 是一个 `int`。在 Python 中，`int` 是 **不可变** 的，这意味着一旦创建，其 [值](https://docs.python.org/3/reference/datamodel.html#objects-values-and-types) 就不能被更改，无论是在函数的作用域内还是外。然而，考虑 `a` 是一个 `list` 的情况，这是一个 **可变** 类型。可变类型是指在创建后可以更改其值的类型。

```py
a = [999]
def f(arg):
 arg[0] = 2
 return arg

print(f(a))
## [2]
print(a) # not [999] anymore!
## [2]
```

在这种情况下 `a` *被修改了*。在函数内部更改参数的值会影响函数外部的那个变量。

准备感到困惑了吗？这里有一个棘手的第三个例子。如果我们接受一个列表，但试图对它做其他的事情会发生什么。

```py
a = [999]
def f(arg):
 arg = [2]
 return arg

print(f(a))
## [2]
print(a) # didn't change this time :(
## [999]
```

那次 `a` 在全局作用域中并没有永久改变。为什么会发生这种情况？我以为 `list` 是可变的！

所有这一切背后的原因甚至与函数本身无关。相反，它与 Python 如何管理 [对象、值和类型](https://docs.python.org/3/reference/datamodel.html#objects-values-and-types) 有关。它还与 [赋值](https://docs.python.org/3/reference/executionmodel.html#naming-and-binding) 期间发生的事情有关。

让我们重新审视上面的代码，但将所有内容都从函数中提取出来。Python 是按赋值传递的，所以我们只需要理解赋值是如何工作的。从不可变的 `int` 示例开始，我们有以下内容。

```py
# old code: 
# a = 1
# def f(arg):
#     arg = 2
#     return arg
a = 1    # still done in global scope
arg = a  # arg is a name that is bound to the object a refers to
arg = 2  # arg is a name that is bound to the object 2
print(arg is a)
## False
print(id(a), id(arg)) # different!`
## 139774560007520 139774560007552
print(a)
## 1
```

`[`id()`](https://docs.python.org/3/library/functions.html#id) 函数返回对象的 **标识**，这有点像它的内存地址。对象的标识是唯一的且恒定的。如果两个变量，比如 `a` 和 `b`，具有相同的标识，`a is b` 将评估为 `True`。否则，它将评估为 `False`。

在第一行，名称 `a` 被绑定到对象 `1`。在第二行，名称 `arg` 被绑定到由名称 `a` 所引用的对象。在第二行完成后，`arg` 和 `a` 是同一对象的两个名称（你可以通过在此行后立即插入 `arg is a` 来确认这一事实）。

在第三行，`arg` 被绑定到 `2`。变量 `arg` 可以被更改，但只能通过将其重新绑定到另一个对象。重新绑定 `arg` 不会更改 `a` 所引用的值，因为 `a` 仍然引用 `1`，一个与 `2` 分离的对象。没有必要重新绑定 `a`，因为它在第三行根本就没有被提及。

如果我们回到第一个函数示例，基本上是同样的概念。然而，唯一的区别是 `arg` 在它自己的作用域中。让我们看看我们第二个代码块的一个简化版本，它使用了一个可变列表。

```py
a = [999]
# old code:
# def f(arg):
#     arg[0] = 2
#     return arg
arg = a
arg[0] = 2
print(arg)
## [2]
print(a)
## [2]
print(arg is a)
## True
```

在这个例子中，当我们运行 `arg = a` 时，名称 `arg` 绑定到与 `a` 绑定到同一个对象上。这一点是相同的。然而，这里唯一的区别是，由于列表是可变的，改变 `arg` 的第一个元素是“就地”完成的，并且所有变量都可以访问被修改的对象。

为什么第三个例子产生了意外的结果？区别在于这一行 `arg = [2]`。这会将名称 `arg` 绑定到不同的变量上。`list`s 仍然是可变的，但这与重新绑定无关——重新绑定名称无论绑定到什么类型的对象上都会生效。在这种情况下，我们正在将 `arg` 重新绑定到一个完全不同的列表上。

## 6.8 访问和修改捕获的变量

在上一节中，我们讨论了作为函数参数传递的变量。现在我们讨论的是**捕获**的变量。它们不是作为变量传递的，但它们仍然在函数内部使用。一般来说，尽管在两种语言中都可以访问和修改非局部捕获的变量，但这不是一个好主意。

### 6.8.1 在 R 中访问捕获的变量

如 Hadley Wickham 在他的书中[所写](https://adv-r.hadley.nz/functions#dynamic-lookup)，“词法作用域决定了在哪里查找值，但不是何时查找值。” R 有**动态查找**，这意味着函数内部的代码只有在函数**运行**时才会尝试访问引用的变量，而不是在定义时。

考虑下面的 R 代码。`dataReadyForModeling()` 函数是在全局环境中创建的，全局环境包含一个名为 `dataAreClean` 的布尔变量。

```py
# R
dataAreClean <-  TRUE
dataReadyForModeling <-  function(){
 return(dataAreClean)
}
dataAreClean <-  FALSE
# readyToDoSecondPart() # what happens if we call it now?
```

现在想象一下与一个合作者共享一些代码。进一步想象，你的合作者是领域专家，对 R 编程了解不多。假设他在完成工作后更改了脚本中的全局变量 `dataAreClean`。这不应该导致对整体程序相对微小的改变吗？

让我们进一步探讨这个假设。考虑以下（非常典型）的任何条件为真时可能发生的情况：

+   你或你的合作者不确定 `dataReadyForModeling()` 会返回什么，因为你不理解动态查找，或者

+   很难直观地跟踪所有对 `dataAreClean` 的赋值（例如，你的脚本相当长或者它经常变化），或者

+   你不是按顺序运行代码（例如，你一次又一次地测试代码块，而不是清除你的内存并从头开始 `source()`，一次又一次地）。

在这些情况中，对程序的理解将会受损。然而，如果你遵循上述原则，即函数代码中永远不要引用非局部变量，那么小组的每个成员都可以单独完成自己的工作，最小化相互之间的依赖。

违反这个规则可能造成麻烦的另一个原因是如果你定义了一个引用了不存在变量的函数。*定义*这个函数永远不会抛出错误，因为 R 会假设该变量定义在全局环境中。*调用*这个函数可能会抛出错误，除非你意外地定义了该变量，或者如果你忘记删除不再使用的变量。使用以下代码定义`myFunc()`不会抛出错误，即使你认为它应该会！

```py
# R
myFunc <-  function(){
 return(varigbleNameWithTypo) #varigble?
}
```

### 6.8.2 Python 中访问捕获的变量

在 Python 中也是同样的情况。考虑`everything_is_safe()`，一个与`dataReadyForModeling()`类似的功能。

```py
# python
missile_launch_codes_set = True
def everything_is_safe():
 return not missile_launch_codes_set

missile_launch_codes_set = False
everything_is_safe()
## True
```

我们还可以定义`my_func()`，它与`myFunc()`类似。定义这个函数也不会抛出错误。

```py
# python
def my_func():
 return varigble_name_with_typo
```

所以请远离在函数体外部引用变量！

### 6.8.3 R 中修改捕获的变量

现在假设我们想要做得更过分一些，除了*访问*全局变量之外，还想*修改*它们呢？

```py
a <-  1
makeATwo <-  function(arg){
 arg <-  2
 a <<-  arg
}
print(makeATwo(a))
## [1] 2
print(a)
## [1] 2
```

在上面的程序中，`makeATwo()`将`a`复制到`arg`中。然后它将`2`赋值给这个副本。**然后它将这个`2`写入父环境中的全局`a`变量。**它是通过使用 R 的超级赋值运算符`<<-`来做到这一点的。无论传递给这个函数的输入是什么，它都会将`2`精确地赋值给`a`，无论是什么。

这是有问题的，因为你正忙于关注一个函数：`makeATwo()`。每次你编写依赖于`a`（或依赖于`a`的东西，或依赖于依赖于`a`的东西，或……）的代码时，你将不得不反复打断你的思路来*尝试*记住你正在做的事情是否与当前的`makeATwo()`调用位置兼容。

### 6.8.4 Python 中修改捕获的变量

Python 中有一个与 R 的超级赋值运算符（`<<-`）类似的东西。它是`global`关键字。这个关键字将允许你在函数内部修改全局变量。

使用`global`关键字的好处是它使得查找**副作用**相对容易（函数的副作用是指它对非局部变量所做的更改）。是的，这个关键字应该少用，甚至比仅仅引用全局变量还要少用，但如果你在调试时，想要追踪变量意外被更改的地方，你可以按`Ctrl-F`并搜索短语`global`。

```py
a = 1
def increment_a():
 global a
 a += 1
[increment_a() for _ in range(10)]
## [None, None, None, None, None, None, None, None, None, None]
print(a)
## 11
```

## 6.9 练习

### 6.9.1 R 问题

假设你有一个矩阵 $\mathbf{X} \in \mathbb{R}^{n \times p}$ 和一个列向量 $\mathbf{y} \in \mathbb{R}^{n}$。为了估计线性回归模型 $$\begin{equation} \mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \epsilon, \end{equation}$$ 其中 $\boldsymbol{\beta} \in \mathbb{R}^p$ 是一个误差的列向量，你可以使用微积分而不是数值优化。$\boldsymbol{\beta}$ 的最小二乘估计公式为 $$\begin{equation} \hat{\boldsymbol{\beta}} = (\mathbf{X}^\intercal \mathbf{X})^{-1} \mathbf{X}^\intercal \mathbf{y}. \end{equation}$$

一旦找到这个 $p$-维向量，你还可以获得 *预测（或拟合）值*

$$\begin{equation} \hat{\mathbf{y}} := \mathbf{X}\hat{\boldsymbol{\beta}}, \end{equation}$$ 以及 *残差（或误差）*

$$\begin{equation} \mathbf{y} - \hat{\mathbf{y}} \end{equation}$$

编写一个名为 `getLinModEstimates()` 的函数，该函数按以下顺序接受两个参数：

+   响应数据 `vector` $\mathbf{y}$

+   预测变量矩阵 $\mathbf{X}$.

让它返回一个包含三个输出的命名 `list`：

+   系数估计作为 `vector`，

+   一个拟合值 `vector`，以及

+   一个残差 `vector`。

返回列表中的三个元素应具有名称 `coefficients`、`fitVals` 和 `residuals`。

编写一个名为 `monteCarlo` 的函数，

+   接收一个名为 `sim(n)` 的函数作为输入，该函数模拟 `n` 个标量变量，

+   接收一个函数作为输入，该函数在每个随机变量样本上评估 $f(x)$，并且理想情况下接受所有随机变量作为 `vector`，

+   返回一个函数，该函数接受一个整数值参数（`num_sims`）并输出一个长度为 `vector`。

假设 `sim(n)` 只有一个参数：`n`，这是所需的模拟次数。`sim(n)` 的输出应是一个长度为 `n` 的 `vector`。

此返回函数的输出应该是期望的蒙特卡洛估计：$\mathbb{E}[f(X)] \approx \frac{1}{n}\sum_{i=1}^n f(X^i)$。

编写一个名为 `myDFT()` 的函数，该函数计算 `vector` 的 **离散傅里叶变换** 并返回另一个 `vector`。您可以自由地检查您的结果与 `spec.pgram()`、`fft()` 或 `astsa::mvspec()` 的结果，但不要在您的提交中包含对这些函数的调用。您还应该意识到，不同的函数以不同的方式转换和缩放答案，因此在使用任何函数进行测试之前，请务必阅读其文档。

给定数据 $x_1,x_2,\ldots,x_n$, $i = \sqrt{-1}$, 以及 **傅里叶/基本频率** $\omega_j= j/n$ 对于 $j=0,1,\ldots,n-1$, 我们定义离散傅里叶变换（DFT）为：

$$\begin{equation} \label{eq:DFT} d(\omega_j)= n^{-1/2} \sum_{t=1}^n x_t e^{-2 \pi i \omega_j t} \end{equation}$$

### 6.9.2 Python 问题

估计统计模型通常涉及某种形式的优化，并且通常情况下，优化是数值执行的。最著名的优化算法之一是 **牛顿法**。

假设你有一个函数 $f(x)$，它接受一个标量值输入并返回一个标量。还假设你有函数的导数 $f'(x)$、二阶导数 $f''(x)$，以及 $f(x)$ 的最小化输入的起始点猜测：$x_0$。

算法反复应用以下递归：

$$\begin{equation} x_{n+1} = x_{n} - \frac{f'(x_n)}{f''(x_{n})}. \end{equation}$$ 在 $f$ 的适当正则性条件下，经过多次上述递归迭代后，当 $\tilde{n}$ 非常大时，$x_{\tilde{n}}$ 将几乎与 $x_{\tilde{n}-1}$ 相同，且 $x_{\tilde{n}}$ 非常接近 $\text{argmin}_x f(x)$。换句话说，$x_{\tilde{n}}$ 是 $f$ 的最小值，也是 $f'$ 的根。

1.  编写一个名为 `f` 的函数，它接受一个 `float` 类型的 `x` 并返回 $(x-42)² - 33$。

1.  编写一个名为 `f_prime` 的函数，它接受一个 `float` 并返回上述函数的导数。

1.  编写一个名为 `f_dub_prime` 的函数，它接受一个 `float` 并返回 $f$ 的二阶导数的评估。

1.  从理论上讲，$f$ 的最小值是什么？将你的答案分配给变量 `best_x`。

1.  编写一个名为 `minimize()` 的函数，它接受三个参数，并在执行 **十次迭代** 的牛顿算法后返回 $x_{10}$。不要害怕复制粘贴大约十行代码。我们还没有学习循环，所以这没关系。有序参数如下：

    +   评估你感兴趣的函数导数的函数，

    +   评估目标函数二阶导数的函数，

    +   最小化器的初始猜测。

1.  通过将上述函数插入测试，并使用起始点 $10$ 来测试你的函数。将输出分配给名为 `x_ten` 的变量。

编写一个名为 `smw_inverse(A,U,C,V)` 的函数，该函数使用 **Sherman-Morrison-Woodbury 公式**（Guttman 1946）来求矩阵的逆。它按顺序接受参数 $A$、$U$、$C$ 和 $V$，并且作为 Numpy `ndarray`s。假设 `A` 是对角矩阵。

$$\begin{equation} (A + UCV)^{-1} = A^{-1} - A^{-1}U(C^{-1} + VA^{-1}U)^{-1}V A^{-1} \end{equation}$$ 尽管这个公式难以记忆，但当 $A$ 和 $C$ 更容易求逆时（例如，如果 $A$ 是对角矩阵且 $C$ 是标量），它可以非常方便地加快矩阵求逆的速度。这个公式在矩阵相乘的应用中经常出现（有很多这样的例子）。

为了检查你的工作，选择某些输入，并确保你的公式与直观的左侧方法相对应。

### 参考文献

Guttman, Louis. 1946\. “矩阵逆的计算扩展方法。” *《数学统计年刊》* 17 (3): 336–43\. [`doi.org/10.1214/aoms/1177730946`](https://doi.org/10.1214/aoms/1177730946).

* * *

1.  原始函数是不包含 R 代码的函数，它们在内部用 C 语言实现。这是 R 中唯一没有父环境的函数类型。↩

1.  你可能已经注意到，Python 使用两个不同的词来避免混淆。与 R 不同，Python 使用“参数”（而不是“参数”）一词来指代函数接受的输入，而“参数”则指用户插入的具体值。↩

1.  函数并不是唯一拥有自己的命名空间的事物。[类也是如此](https://docs.python.org/3/tutorial/classes.html#a-first-look-at-classes)。关于类的更多信息请参阅第十四章。↩

1.  虽然有一些例外，但这通常是正确的。↩
