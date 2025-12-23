# 2. 基础

> 原文：[`leanprover-community.github.io/mathematics_in_lean/C02_Basics.html`](https://leanprover-community.github.io/mathematics_in_lean/C02_Basics.html)

*数学在 Lean 中* **   2. 基础

+   查看页面源代码

* * *

本章旨在向您介绍 Lean 中数学推理的精髓：计算、应用引理和定理，以及关于通用结构的推理。

## 2.1. 计算

我们通常学会进行数学计算，而不将其视为证明。但当我们像 Lean 所要求的那样为计算的每一步进行辩护时，最终结果是证明计算的左侧等于右侧。

在 Lean 中，陈述一个定理等同于陈述一个目标，即证明该定理的目标。Lean 提供了重写策略`rw`，用于在目标中将等式的左侧替换为右侧。如果`a`、`b`和`c`是实数，`mul_assoc a b c`是等式`a * b * c = a * (b * c)`，而`mul_comm a b`是等式`a * b = b * a`。Lean 提供了自动化，通常可以消除显式引用此类事实的需要，但它们对于说明目的很有用。在 Lean 中，乘法结合律适用于左侧，因此`mul_assoc`的左侧也可以写成`(a * b) * c`。然而，通常好的风格是注意 Lean 的符号约定，并在 Lean 也这样做的情况下省略括号。

让我们尝试`rw`。

```py
example  (a  b  c  :  ℝ)  :  a  *  b  *  c  =  b  *  (a  *  c)  :=  by
  rw  [mul_comm  a  b]
  rw  [mul_assoc  b  a  c] 
```

在相关示例文件的开头，`import` 行导入来自 Mathlib 的实数理论以及有用的自动化。为了简洁起见，我们在教科书中通常省略此类信息。

您可以尝试修改以查看会发生什么。您可以在 VS Code 中键入`ℝ`字符为`\R`或`\real`。符号只有在您按下空格键或制表键时才会出现。如果您在阅读 Lean 文件时将鼠标悬停在符号上，VS Code 将显示可以用来输入它的语法。如果您想查看所有可用的缩写，您可以按 Ctrl-Shift-P，然后键入缩写以获取访问`Lean 4: 显示 Unicode 输入缩写`命令的权限。如果您的键盘没有容易访问的反斜杠，您可以通过更改`lean4.input.leader`设置来更改前导字符。

当光标位于策略证明的中间时，Lean 会在*Lean Infoview*窗口中报告当前的*证明状态*。随着您将光标移过证明的每一步，您可以看到状态的变化。Lean 中的一个典型证明状态可能如下所示：

```py
1  goal
x  y  :  ℕ,
h₁  :  Prime  x,
h₂  :  ¬Even  x,
h₃  :  y  >  x
⊢  y  ≥  4 
```

在以 `⊢` 开头的行之前表示的是*上下文*：它们是当前正在使用的对象和假设。在这个例子中，这些包括两个对象，`x` 和 `y`，每个都是自然数。它们还包括三个假设，分别标记为 `h₁`、`h₂` 和 `h₃`。在 Lean 中，上下文中的所有内容都用标识符标记。你可以将这些下标标签键入为 `h\1`、`h\2` 和 `h\3`，但任何合法的标识符都可以：你可以使用 `h1`、`h2`、`h3`，或者 `foo`、`bar` 和 `baz`。最后一行代表*目标*，即要证明的事实。有时人们用*目标*来表示要证明的事实，用*目标*来表示上下文和目标的组合。在实践中，通常可以清楚地理解意图。

尝试证明这些恒等式，在每种情况下用策略证明替换 `sorry`。使用 `rw` 策略，你可以使用左箭头 (`\l`) 来反转一个恒等式。例如，`rw [← mul_assoc a b c]` 将当前目标中的 `a * (b * c)` 替换为 `a * b * c`。注意，指向左边的箭头指的是从右到左在 `mul_assoc` 提供的恒等式中移动，它与目标左右两侧无关。

```py
example  (a  b  c  :  ℝ)  :  c  *  b  *  a  =  b  *  (a  *  c)  :=  by
  sorry

example  (a  b  c  :  ℝ)  :  a  *  (b  *  c)  =  b  *  (a  *  c)  :=  by
  sorry 
```

你也可以不带参数使用像 `mul_assoc` 和 `mul_comm` 这样的恒等式。在这种情况下，重写策略尝试将左侧与目标中的表达式匹配，使用它找到的第一个模式。

```py
example  (a  b  c  :  ℝ)  :  a  *  b  *  c  =  b  *  c  *  a  :=  by
  rw  [mul_assoc]
  rw  [mul_comm] 
```

你也可以提供*部分*信息。例如，`mul_comm a` 匹配任何形式为 `a * ?` 的模式，并将其重写为 `? * a`。尝试在不提供任何参数的情况下完成这些示例的第一个，以及只提供一个参数的第二个。

```py
example  (a  b  c  :  ℝ)  :  a  *  (b  *  c)  =  b  *  (c  *  a)  :=  by
  sorry

example  (a  b  c  :  ℝ)  :  a  *  (b  *  c)  =  b  *  (a  *  c)  :=  by
  sorry 
```

你也可以使用来自局部上下文的事实 `rw`。

```py
example  (a  b  c  d  e  f  :  ℝ)  (h  :  a  *  b  =  c  *  d)  (h'  :  e  =  f)  :  a  *  (b  *  e)  =  c  *  (d  *  f)  :=  by
  rw  [h']
  rw  [←  mul_assoc]
  rw  [h]
  rw  [mul_assoc] 
```

尝试这些，使用定理 `sub_self` 对第二个进行操作：

```py
example  (a  b  c  d  e  f  :  ℝ)  (h  :  b  *  c  =  e  *  f)  :  a  *  b  *  c  *  d  =  a  *  e  *  f  *  d  :=  by
  sorry

example  (a  b  c  d  :  ℝ)  (hyp  :  c  =  b  *  a  -  d)  (hyp'  :  d  =  a  *  b)  :  c  =  0  :=  by
  sorry 
```

可以通过在方括号内用逗号分隔相关恒等式来使用单个命令执行多个重写命令。

```py
example  (a  b  c  d  e  f  :  ℝ)  (h  :  a  *  b  =  c  *  d)  (h'  :  e  =  f)  :  a  *  (b  *  e)  =  c  *  (d  *  f)  :=  by
  rw  [h',  ←  mul_assoc,  h,  mul_assoc] 
```

在任何重写列表中，将光标放在逗号后面，你仍然可以看到增量进度。

另一个技巧是，我们可以在示例或定理之外一次性声明变量。然后 Lean 会自动包含它们。

```py
variable  (a  b  c  d  e  f  :  ℝ)

example  (h  :  a  *  b  =  c  *  d)  (h'  :  e  =  f)  :  a  *  (b  *  e)  =  c  *  (d  *  f)  :=  by
  rw  [h',  ←  mul_assoc,  h,  mul_assoc] 
```

检查上述证明开始时的策略状态，可以发现 Lean 确实包含了所有变量。我们可以通过将其放在 `section ... end` 块中来限定声明的范围。最后，从引言中回忆起 Lean 为我们提供了一个命令来确定表达式的类型：

```py
section
variable  (a  b  c  :  ℝ)

#check  a
#check  a  +  b
#check  (a  :  ℝ)
#check  mul_comm  a  b
#check  (mul_comm  a  b  :  a  *  b  =  b  *  a)
#check  mul_assoc  c  a  b
#check  mul_comm  a
#check  mul_comm

end 
```

`#check`命令适用于对象和事实。对于命令`#check a`，Lean 报告`a`的类型为`ℝ`。对于命令`#check mul_comm a b`，Lean 报告`mul_comm a b`是事实`a * b = b * a`的证明。命令`#check (a : ℝ)`表示我们期望`a`的类型是`ℝ`，如果这不是情况，Lean 将引发错误。我们将在稍后解释最后三个`#check`命令的输出，但在此同时，您可以查看它们，并尝试一些自己的`#check`命令。

让我们尝试一些更复杂的例子。定理`two_mul a`表明`2 * a = a + a`。定理`add_mul`和`mul_add`表达了乘法在加法上的分配性，而定理`add_assoc`表达了加法的结合性。使用`#check`命令查看精确的陈述。

```py
example  :  (a  +  b)  *  (a  +  b)  =  a  *  a  +  2  *  (a  *  b)  +  b  *  b  :=  by
  rw  [mul_add,  add_mul,  add_mul]
  rw  [←  add_assoc,  add_assoc  (a  *  a)]
  rw  [mul_comm  b  a,  ←  two_mul] 
```

虽然可以通过在编辑器中逐步执行来弄清楚这个证明的情况，但单独阅读时很难理解。Lean 提供了使用`calc`关键字编写这种证明的更结构化方式。

```py
example  :  (a  +  b)  *  (a  +  b)  =  a  *  a  +  2  *  (a  *  b)  +  b  *  b  :=
  calc
  (a  +  b)  *  (a  +  b)  =  a  *  a  +  b  *  a  +  (a  *  b  +  b  *  b)  :=  by
  rw  [mul_add,  add_mul,  add_mul]
  _  =  a  *  a  +  (b  *  a  +  a  *  b)  +  b  *  b  :=  by
  rw  [←  add_assoc,  add_assoc  (a  *  a)]
  _  =  a  *  a  +  2  *  (a  *  b)  +  b  *  b  :=  by
  rw  [mul_comm  b  a,  ←  two_mul] 
```

注意，证明并不是从`by`开始的：以`calc`开头的表达式是一个*证明项*。`calc`表达式也可以用在策略证明中，但 Lean 将其解释为使用生成的证明项来解决目标的指令。`calc`语法很挑剔：下划线和论证必须采用上述格式。Lean 使用缩进来确定策略块或`calc`块的开始和结束位置；尝试更改上述证明中的缩进来看看会发生什么。

编写`calc`证明的一种方法首先使用`sorry`策略进行论证，确保 Lean 接受这些表达式，然后使用策略论证各个步骤。

```py
example  :  (a  +  b)  *  (a  +  b)  =  a  *  a  +  2  *  (a  *  b)  +  b  *  b  :=
  calc
  (a  +  b)  *  (a  +  b)  =  a  *  a  +  b  *  a  +  (a  *  b  +  b  *  b)  :=  by
  sorry
  _  =  a  *  a  +  (b  *  a  +  a  *  b)  +  b  *  b  :=  by
  sorry
  _  =  a  *  a  +  2  *  (a  *  b)  +  b  *  b  :=  by
  sorry 
```

尝试使用纯`rw`证明和更结构化的`calc`证明来证明以下恒等式：

```py
example  :  (a  +  b)  *  (c  +  d)  =  a  *  c  +  a  *  d  +  b  *  c  +  b  *  d  :=  by
  sorry 
```

以下练习稍微有些挑战性。你可以使用下面列出的定理。

```py
example  (a  b  :  ℝ)  :  (a  +  b)  *  (a  -  b)  =  a  ^  2  -  b  ^  2  :=  by
  sorry

#check  pow_two  a
#check  mul_sub  a  b  c
#check  add_mul  a  b  c
#check  add_sub  a  b  c
#check  sub_sub  a  b  c
#check  add_zero  a 
```

我们还可以在假设的上下文中进行重写。例如，`rw [mul_comm a b] at hyp`将假设`hyp`中的`a * b`替换为`b * a`。

```py
example  (a  b  c  d  :  ℝ)  (hyp  :  c  =  d  *  a  +  b)  (hyp'  :  b  =  a  *  d)  :  c  =  2  *  a  *  d  :=  by
  rw  [hyp']  at  hyp
  rw  [mul_comm  d  a]  at  hyp
  rw  [←  two_mul  (a  *  d)]  at  hyp
  rw  [←  mul_assoc  2  a  d]  at  hyp
  exact  hyp 
```

在最后一步，`exact`策略可以使用`hyp`来解决目标，因为那时`hyp`与目标完全匹配。

我们通过指出 Mathlib 提供了一个有用的自动化功能，即`ring`策略来结束本节，该策略旨在证明任何交换环中的恒等式，只要它们纯粹地来自环公理，而不使用任何局部假设。

```py
example  :  c  *  b  *  a  =  b  *  (a  *  c)  :=  by
  ring

example  :  (a  +  b)  *  (a  +  b)  =  a  *  a  +  2  *  (a  *  b)  +  b  *  b  :=  by
  ring

example  :  (a  +  b)  *  (a  -  b)  =  a  ^  2  -  b  ^  2  :=  by
  ring

example  (hyp  :  c  =  d  *  a  +  b)  (hyp'  :  b  =  a  *  d)  :  c  =  2  *  a  *  d  :=  by
  rw  [hyp,  hyp']
  ring 
```

当我们导入`Mathlib.Data.Real.Basic`时，`ring`策略是间接导入的，但在下一节中我们将看到它可以用在除了实数以外的结构上。可以使用命令`import Mathlib.Tactic`显式导入。我们将在那里看到其他常见代数结构的类似策略。

有一种 `rw` 的变体称为 `nth_rw`，它允许你只替换目标中特定实例的表达式。可能的匹配从 1 开始枚举，所以在这个例子中，`nth_rw 2 [h]` 将 `a + b` 的第二个出现替换为 `c`。

```py
example  (a  b  c  :  ℕ)  (h  :  a  +  b  =  c)  :  (a  +  b)  *  (a  +  b)  =  a  *  c  +  b  *  c  :=  by
  nth_rw  2  [h]
  rw  [add_mul] 
```

## 2.2\. 在代数结构中证明恒等式

从数学上讲，一个环由一个对象集合 $R$、运算 $+$ 和 $\times$、常数 $0$ 和 $1$ 以及一个运算 $x \mapsto -x$ 组成，该运算满足以下条件：

+   $R$ 与 $+$ 是一个 *阿贝尔群*，其中 $0$ 是加法单位元，否定是逆元。

+   乘法与单位元 $1$ 结合，并且乘法对加法分配。

在 Lean 中，对象的集合被表示为一个 *类型*，`R`。环的公理如下：

```py
variable  (R  :  Type*)  [Ring  R]

#check  (add_assoc  :  ∀  a  b  c  :  R,  a  +  b  +  c  =  a  +  (b  +  c))
#check  (add_comm  :  ∀  a  b  :  R,  a  +  b  =  b  +  a)
#check  (zero_add  :  ∀  a  :  R,  0  +  a  =  a)
#check  (neg_add_cancel  :  ∀  a  :  R,  -a  +  a  =  0)
#check  (mul_assoc  :  ∀  a  b  c  :  R,  a  *  b  *  c  =  a  *  (b  *  c))
#check  (mul_one  :  ∀  a  :  R,  a  *  1  =  a)
#check  (one_mul  :  ∀  a  :  R,  1  *  a  =  a)
#check  (mul_add  :  ∀  a  b  c  :  R,  a  *  (b  +  c)  =  a  *  b  +  a  *  c)
#check  (add_mul  :  ∀  a  b  c  :  R,  (a  +  b)  *  c  =  a  *  c  +  b  *  c) 
```

你将在以后了解第一行中的方括号，但就目前而言，只需说这个声明给我们一个类型，`R`，以及 `R` 上的环结构。然后 Lean 允许我们使用 `R` 的元素进行通用的环符号，并利用关于环的定理库。

一些定理的名称应该看起来很熟悉：它们正是我们在上一节中使用实数进行计算时使用的那些。Lean 不仅擅长证明关于自然数和整数等具体数学结构的性质，而且也擅长证明关于抽象结构（如环）的性质，这些结构是公理化定义的。此外，Lean 支持 *通用推理* 关于抽象和具体结构，并且可以训练它识别适当的实例。因此，任何关于环的定理都可以应用于具体的环，如整数环 `ℤ`、有理数环 `ℚ` 和复数环 `ℂ`。它也可以应用于任何扩展环的抽象结构的实例，例如任何有序环或任何域。

然而，并非所有实数的性质在任意环中都成立。例如，实数上的乘法是交换的，但通常不成立。如果你已经学过线性代数课程，你会认识到，对于每个 $n$，实数的 $n$ 阶矩阵形成一个环，其中交换性通常不成立。如果我们声明 `R` 为 *交换环*，实际上，当我们将 `ℝ` 替换为 `R` 时，上一节中的所有定理仍然成立。

```py
variable  (R  :  Type*)  [CommRing  R]
variable  (a  b  c  d  :  R)

example  :  c  *  b  *  a  =  b  *  (a  *  c)  :=  by  ring

example  :  (a  +  b)  *  (a  +  b)  =  a  *  a  +  2  *  (a  *  b)  +  b  *  b  :=  by  ring

example  :  (a  +  b)  *  (a  -  b)  =  a  ^  2  -  b  ^  2  :=  by  ring

example  (hyp  :  c  =  d  *  a  +  b)  (hyp'  :  b  =  a  *  d)  :  c  =  2  *  a  *  d  :=  by
  rw  [hyp,  hyp']
  ring 
```

我们留给你们去验证所有其他证明都保持不变。注意，当证明很短，如 `by ring` 或 `by linarith` 或 `by sorry` 时，将其放在 `by` 的同一行上是常见（并且允许）的。良好的证明写作风格应在简洁性和可读性之间取得平衡。

本节的目标是加强你在上一节中发展的技能，并将它们应用于对环进行公理推理。我们将从上面列出的公理开始，并使用它们推导出其他事实。我们证明的大多数事实已经在 Mathlib 中。我们将给出我们证明的版本相同的名称，以帮助你学习库的内容以及命名约定。

Lean 提供了一种类似于编程语言中使用的组织机制：当在 *命名空间* `bar` 中引入定义或定理 `foo` 时，其全名是 `bar.foo`。命令 `open bar` 会在稍后 *打开* 命名空间，这允许我们使用较短的名称 `foo`。为了避免由于名称冲突而引起的错误，在下一个示例中，我们将我们库定理的版本放在一个新的命名空间 `MyRing` 中。

以下示例表明，我们不需要将 `add_zero` 或 `add_neg_cancel` 作为环公理，因为它们可以从其他公理中得出。

```py
namespace  MyRing
variable  {R  :  Type*}  [Ring  R]

theorem  add_zero  (a  :  R)  :  a  +  0  =  a  :=  by  rw  [add_comm,  zero_add]

theorem  add_neg_cancel  (a  :  R)  :  a  +  -a  =  0  :=  by  rw  [add_comm,  neg_add_cancel]

#check  MyRing.add_zero
#check  add_zero

end  MyRing 
```

这种效果是，我们可以暂时重新证明库中的一个定理，然后继续使用库版本。但不要作弊！在接下来的练习中，请务必只使用我们在本节中之前已经证明的关于环的一般事实。

（如果你仔细观察，你可能已经注意到我们在 `(R : Type*)` 中将圆括号改为了花括号 `{R : Type*}`。这表明 `R` 是一个 *隐式参数*。我们将在稍后解释这意味着什么，但在此期间不必担心。）

这里有一个有用的定理：

```py
theorem  neg_add_cancel_left  (a  b  :  R)  :  -a  +  (a  +  b)  =  b  :=  by
  rw  [←  add_assoc,  neg_add_cancel,  zero_add] 
```

证明伴随版本：

```py
theorem  add_neg_cancel_right  (a  b  :  R)  :  a  +  b  +  -b  =  a  :=  by
  sorry 
```

使用这些来证明以下内容：

```py
theorem  add_left_cancel  {a  b  c  :  R}  (h  :  a  +  b  =  a  +  c)  :  b  =  c  :=  by
  sorry

theorem  add_right_cancel  {a  b  c  :  R}  (h  :  a  +  b  =  c  +  b)  :  a  =  c  :=  by
  sorry 
```

足够的规划下，你可以用三次重写来完成每一个。

现在，我们将解释花括号的使用。想象一下，你处于一个情境，其中你的上下文中有 `a`、`b` 和 `c`，以及一个假设 `h : a + b = a + c`，你想要得出结论 `b = c`。在 Lean 中，你可以像对对象一样对假设和事实应用定理，所以你可能认为 `add_left_cancel a b c h` 是 `b = c` 这一事实的证明。但请注意，明确写出 `a`、`b` 和 `c` 是多余的，因为假设 `h` 清楚地表明了这些是我们心中的对象。在这种情况下，输入几个额外的字符并不麻烦，但如果我们想要将 `add_left_cancel` 应用到更复杂的表达式中，编写它们将会很繁琐。在这些情况下，Lean 允许我们将参数标记为 *隐式*，这意味着它们应该被省略，并通过其他方式（如后续参数和假设）推断出来。`{a b c : R}` 中的花括号正是这样做的。因此，给定上述定理的陈述，正确的表达式仅仅是 `add_left_cancel h`。

为了说明，让我们展示 `a * 0 = 0` 可以从环公理中得出。

```py
theorem  mul_zero  (a  :  R)  :  a  *  0  =  0  :=  by
  have  h  :  a  *  0  +  a  *  0  =  a  *  0  +  0  :=  by
  rw  [←  mul_add,  add_zero,  add_zero]
  rw  [add_left_cancel  h] 
```

我们使用了一个新的技巧！如果你逐步通过证明，你可以看到发生了什么。`have` 策略引入了一个新的目标，`a * 0 + a * 0 = a * 0 + 0`，与原始目标具有相同的环境。下一行缩进的事实表明 Lean 预期一个策略块，该块用于证明这个新目标。因此，缩进促进了模块化证明风格：缩进的子证明建立了由 `have` 引入的目标。之后，我们回到证明原始目标，除了增加了一个新的假设 `h`：证明它之后，我们现在可以自由地使用它。此时，目标正好是 `add_left_cancel h` 的结果。

我们同样可以用 `apply add_left_cancel h` 或 `exact add_left_cancel h` 来结束证明。`exact` 策略的参数是一个完全证明当前目标的证明项，而不创建任何新的目标。`apply` 策略是一个变体，其参数不一定是完整的证明。缺失的部分要么由 Lean 自动推断，要么成为需要证明的新目标。虽然 `exact` 策略在技术上可能是多余的，因为它严格不如 `apply` 强大，但它使证明脚本对人类读者来说更清晰，并且在库演变时更容易维护。

记住，乘法不一定假设是交换的，因此下面的定理也需要一些工作。

```py
theorem  zero_mul  (a  :  R)  :  0  *  a  =  0  :=  by
  sorry 
```

到现在为止，你也应该能够将下一个练习中的每个 `sorry` 替换为一个证明，仍然只使用本节中我们建立的关于环的事实。

```py
theorem  neg_eq_of_add_eq_zero  {a  b  :  R}  (h  :  a  +  b  =  0)  :  -a  =  b  :=  by
  sorry

theorem  eq_neg_of_add_eq_zero  {a  b  :  R}  (h  :  a  +  b  =  0)  :  a  =  -b  :=  by
  sorry

theorem  neg_zero  :  (-0  :  R)  =  0  :=  by
  apply  neg_eq_of_add_eq_zero
  rw  [add_zero]

theorem  neg_neg  (a  :  R)  :  -  -a  =  a  :=  by
  sorry 
```

我们在第三个定理中必须使用注释 `(-0 : R)` 而不是 `0`，因为没有指定 `R`，Lean 就无法推断我们心中所想的 `0` 是什么，默认情况下它会被解释为自然数。

在 Lean 中，环中的减法可以证明等于加法上的加法逆元。

```py
example  (a  b  :  R)  :  a  -  b  =  a  +  -b  :=
  sub_eq_add_neg  a  b 
```

在实数上，它是这样定义的：

```py
example  (a  b  :  ℝ)  :  a  -  b  =  a  +  -b  :=
  rfl

example  (a  b  :  ℝ)  :  a  -  b  =  a  +  -b  :=  by
  rfl 
```

证明项 `rfl` 是“自反性”的简称。将其作为 `a - b = a + -b` 的证明迫使 Lean 展开定义并认识到两边是相同的。`rfl` 策略做的是同样的事情。这是 Lean 内在逻辑中所谓的 *定义性等价* 的一个例子。这意味着不仅可以用 `sub_eq_add_neg` 来重写 `a - b = a + -b`，而且在某些上下文中，当处理实数时，你可以交换方程的两边。例如，你现在有足够的信息来证明上一节中的定理 `self_sub`：

```py
theorem  self_sub  (a  :  R)  :  a  -  a  =  0  :=  by
  sorry 
```

证明你可以使用 `rw` 来证明这一点，但如果将任意的环 `R` 替换为实数，你也可以使用 `apply` 或 `exact` 中的任何一个来证明。

Lean 知道 `1 + 1 = 2` 在任何环中都成立。通过一点努力，你可以用这个来证明上一节中的定理 `two_mul`：

```py
theorem  one_add_one_eq_two  :  1  +  1  =  (2  :  R)  :=  by
  norm_num

theorem  two_mul  (a  :  R)  :  2  *  a  =  a  +  a  :=  by
  sorry 
```

我们在本节结束时指出，我们上面建立的一些关于加法和负的事实不需要环公理的全部强度，甚至不需要加法的交换性。弱化的 *群* 概念可以如下公理化：

```py
variable  (A  :  Type*)  [AddGroup  A]

#check  (add_assoc  :  ∀  a  b  c  :  A,  a  +  b  +  c  =  a  +  (b  +  c))
#check  (zero_add  :  ∀  a  :  A,  0  +  a  =  a)
#check  (neg_add_cancel  :  ∀  a  :  A,  -a  +  a  =  0) 
```

当群运算交换时，使用加法符号是惯例，否则使用乘法符号。因此，Lean 定义了乘法版本以及加法版本（以及它们的阿贝尔变体 `AddCommGroup` 和 `CommGroup`）。

```py
variable  {G  :  Type*}  [Group  G]

#check  (mul_assoc  :  ∀  a  b  c  :  G,  a  *  b  *  c  =  a  *  (b  *  c))
#check  (one_mul  :  ∀  a  :  G,  1  *  a  =  a)
#check  (inv_mul_cancel  :  ∀  a  :  G,  a⁻¹  *  a  =  1) 
```

如果你感到自信，尝试仅使用这些公理来证明以下关于群的事实。在这个过程中，你需要证明一些辅助引理。本节中我们已经完成的证明提供了一些提示。

```py
theorem  mul_inv_cancel  (a  :  G)  :  a  *  a⁻¹  =  1  :=  by
  sorry

theorem  mul_one  (a  :  G)  :  a  *  1  =  a  :=  by
  sorry

theorem  mul_inv_rev  (a  b  :  G)  :  (a  *  b)⁻¹  =  b⁻¹  *  a⁻¹  :=  by
  sorry 
```

明确调用这些引理是繁琐的，因此 Mathlib 提供了类似于环的策略来覆盖大多数用法：`group` 用于非交换乘法群，`abel` 用于阿贝尔加法群，而 `noncomm_ring` 用于非交换环。看起来很奇怪，代数结构被称为 Ring 和 CommRing，而策略被命名为 noncomm_ring 和 ring。这部分是历史原因，但也为了方便使用更短的名称来处理交换环的策略，因为它使用得更频繁。## 2.3. 使用定理和引理

重写对于证明方程很有用，但对于其他类型的定理呢？例如，我们如何证明一个不等式，比如 $a + e^b \le a + e^c$ 在 $b \le c$ 时总是成立？我们已经看到定理可以应用于论点和假设，并且可以使用 `apply` 和 `exact` 策略来解决目标。在本节中，我们将充分利用这些工具。

考虑库中的定理 `le_refl` 和 `le_trans`：

```py
#check  (le_refl  :  ∀  a  :  ℝ,  a  ≤  a)
#check  (le_trans  :  a  ≤  b  →  b  ≤  c  →  a  ≤  c) 
```

正如我们在第 3.1 节中更详细地解释的那样，`le_trans` 的陈述中的隐式括号与右侧关联，因此它应该被解释为 `a ≤ b → (b ≤ c → a ≤ c)`。库设计者已经将 `a`、`b` 和 `c` 设置为 `le_trans` 的隐式参数，这样 Lean 就不会让你明确地提供它们（除非你真的坚持，我们稍后会讨论）。相反，它期望从它们被使用的上下文中推断它们。例如，当假设 `h : a ≤ b` 和 `h' : b ≤ c` 在上下文中时，以下所有内容都有效：

```py
variable  (h  :  a  ≤  b)  (h'  :  b  ≤  c)

#check  (le_refl  :  ∀  a  :  Real,  a  ≤  a)
#check  (le_refl  a  :  a  ≤  a)
#check  (le_trans  :  a  ≤  b  →  b  ≤  c  →  a  ≤  c)
#check  (le_trans  h  :  b  ≤  c  →  a  ≤  c)
#check  (le_trans  h  h'  :  a  ≤  c) 
```

`apply` 策略接受一个一般陈述或蕴涵的证明，尝试将结论与当前目标匹配，并将假设（如果有的话）作为新的目标留下。如果给定的证明与目标完全匹配（模定义等价），则可以使用 `exact` 策略而不是 `apply`。所以，所有这些都有效：

```py
example  (x  y  z  :  ℝ)  (h₀  :  x  ≤  y)  (h₁  :  y  ≤  z)  :  x  ≤  z  :=  by
  apply  le_trans
  ·  apply  h₀
  ·  apply  h₁

example  (x  y  z  :  ℝ)  (h₀  :  x  ≤  y)  (h₁  :  y  ≤  z)  :  x  ≤  z  :=  by
  apply  le_trans  h₀
  apply  h₁

example  (x  y  z  :  ℝ)  (h₀  :  x  ≤  y)  (h₁  :  y  ≤  z)  :  x  ≤  z  :=
  le_trans  h₀  h₁

example  (x  :  ℝ)  :  x  ≤  x  :=  by
  apply  le_refl

example  (x  :  ℝ)  :  x  ≤  x  :=
  le_refl  x 
```

在第一个例子中，应用 `le_trans` 创建了两个目标，我们使用点来表示每个证明的开始位置。点是可以省略的，但它们有助于 *聚焦* 目标：在点引入的块内，只有一个目标可见，并且必须在块的末尾之前完成。在这里，我们通过用另一个点开始一个新的块来结束第一个块。我们也可以减少缩进。在第三个例子和最后一个例子中，我们完全避免了进入策略模式：`le_trans h₀ h₁` 和 `le_refl x` 是我们需要证明项。

这里还有一些图书馆定理：

```py
#check  (le_refl  :  ∀  a,  a  ≤  a)
#check  (le_trans  :  a  ≤  b  →  b  ≤  c  →  a  ≤  c)
#check  (lt_of_le_of_lt  :  a  ≤  b  →  b  <  c  →  a  <  c)
#check  (lt_of_lt_of_le  :  a  <  b  →  b  ≤  c  →  a  <  c)
#check  (lt_trans  :  a  <  b  →  b  <  c  →  a  <  c) 
```

将它们与 `apply` 和 `exact` 结合起来证明以下内容：

```py
example  (h₀  :  a  ≤  b)  (h₁  :  b  <  c)  (h₂  :  c  ≤  d)  (h₃  :  d  <  e)  :  a  <  e  :=  by
  sorry 
```

实际上，Lean 有一个策略可以自动完成这类事情：

```py
example  (h₀  :  a  ≤  b)  (h₁  :  b  <  c)  (h₂  :  c  ≤  d)  (h₃  :  d  <  e)  :  a  <  e  :=  by
  linarith 
```

`linarith` 策略被设计用来处理 *线性算术*。

```py
example  (h  :  2  *  a  ≤  3  *  b)  (h'  :  1  ≤  a)  (h''  :  d  =  2)  :  d  +  a  ≤  5  *  b  :=  by
  linarith 
```

除了上下文中的方程和不等式，`linarith` 还会使用您作为参数传递的额外不等式。在下一个例子中，`exp_le_exp.mpr h'` 是 `exp b ≤ exp c` 的证明，我们将在下面解释。请注意，在 Lean 中，我们用 `f x` 来表示函数 `f` 对参数 `x` 的应用，这与我们用 `h x` 来表示事实或定理 `h` 对参数 `x` 的应用完全相同。括号仅用于复合参数，例如 `f (x + y)`。如果没有括号，`f x + y` 将被解析为 `(f x) + y`。

```py
example  (h  :  1  ≤  a)  (h'  :  b  ≤  c)  :  2  +  a  +  exp  b  ≤  3  *  a  +  exp  c  :=  by
  linarith  [exp_le_exp.mpr  h'] 
```

图书馆中还有一些定理可以用来在实数上建立不等式。

```py
#check  (exp_le_exp  :  exp  a  ≤  exp  b  ↔  a  ≤  b)
#check  (exp_lt_exp  :  exp  a  <  exp  b  ↔  a  <  b)
#check  (log_le_log  :  0  <  a  →  a  ≤  b  →  log  a  ≤  log  b)
#check  (log_lt_log  :  0  <  a  →  a  <  b  →  log  a  <  log  b)
#check  (add_le_add  :  a  ≤  b  →  c  ≤  d  →  a  +  c  ≤  b  +  d)
#check  (add_le_add_left  :  a  ≤  b  →  ∀  c,  c  +  a  ≤  c  +  b)
#check  (add_le_add_right  :  a  ≤  b  →  ∀  c,  a  +  c  ≤  b  +  c)
#check  (add_lt_add_of_le_of_lt  :  a  ≤  b  →  c  <  d  →  a  +  c  <  b  +  d)
#check  (add_lt_add_of_lt_of_le  :  a  <  b  →  c  ≤  d  →  a  +  c  <  b  +  d)
#check  (add_lt_add_left  :  a  <  b  →  ∀  c,  c  +  a  <  c  +  b)
#check  (add_lt_add_right  :  a  <  b  →  ∀  c,  a  +  c  <  b  +  c)
#check  (add_nonneg  :  0  ≤  a  →  0  ≤  b  →  0  ≤  a  +  b)
#check  (add_pos  :  0  <  a  →  0  <  b  →  0  <  a  +  b)
#check  (add_pos_of_pos_of_nonneg  :  0  <  a  →  0  ≤  b  →  0  <  a  +  b)
#check  (exp_pos  :  ∀  a,  0  <  exp  a)
#check  add_le_add_left 
```

一些定理，如 `exp_le_exp`、`exp_lt_exp` 使用了 *双向蕴涵*，它表示“当且仅当”这个短语。（您可以在 VS Code 中使用 `\lr` 或 `\iff` 来输入它）。我们将在下一章中更详细地讨论这个连接词。这样的定理可以用 `rw` 来将目标重写为等价的目标：

```py
example  (h  :  a  ≤  b)  :  exp  a  ≤  exp  b  :=  by
  rw  [exp_le_exp]
  exact  h 
```

然而，在本节中，我们将使用以下事实：如果 `h : A ↔ B` 是这样的等价关系，那么 `h.mp` 建立了正向方向，`A → B`，而 `h.mpr` 建立了反向方向，`B → A`。在这里，`mp` 代表“modus ponens”，而 `mpr` 代表“modus ponens reverse”。如果您愿意，也可以使用 `h.1` 和 `h.2` 来代替 `h.mp` 和 `h.mpr`。因此，以下证明是有效的：

```py
example  (h₀  :  a  ≤  b)  (h₁  :  c  <  d)  :  a  +  exp  c  +  e  <  b  +  exp  d  +  e  :=  by
  apply  add_lt_add_of_lt_of_le
  ·  apply  add_lt_add_of_le_of_lt  h₀
  apply  exp_lt_exp.mpr  h₁
  apply  le_refl 
```

第一行，`apply add_lt_add_of_lt_of_le` 创建了两个目标，并且再次使用点来区分第一个和第二个证明。

尝试以下示例。中间的示例向您展示了 `norm_num` 策略可以用来解决具体的数值目标。

```py
example  (h₀  :  d  ≤  e)  :  c  +  exp  (a  +  d)  ≤  c  +  exp  (a  +  e)  :=  by  sorry

example  :  (0  :  ℝ)  <  1  :=  by  norm_num

example  (h  :  a  ≤  b)  :  log  (1  +  exp  a)  ≤  log  (1  +  exp  b)  :=  by
  have  h₀  :  0  <  1  +  exp  a  :=  by  sorry
  apply  log_le_log  h₀
  sorry 
```

从这些例子中，应该很明显，能够找到所需的库定理是形式化的重要部分。您可以使用多种策略：

+   您可以在其 [GitHub 仓库](https://github.com/leanprover-community/mathlib4) 中浏览 Mathlib。

+   您可以使用 Mathlib [网页](https://leanprover-community.github.io/mathlib4_docs/) 上的 API 文档。

+   你可以使用 Loogle <https://loogle.lean-lang.org> 通过模式搜索 Lean 和 Mathlib 的定义和定理。

+   你可以依靠 Mathlib 的命名约定和在编辑器中的 Ctrl-space 完成功能来猜测定理名称（或在 Mac 键盘上按 Cmd-space）。在 Lean 中，一个名为 `A_of_B_of_C` 的定理从形式为 `B` 和 `C` 的假设中建立 `A`，其中 `A`、`B` 和 `C` 大致表示我们可能大声读出的目标。因此，一个类似于 `x + y ≤ ...` 的定理可能以 `add_le` 开头。键入 `add_le` 并按 Ctrl-space 将提供一些有用的选择。请注意，按 Ctrl-space 两次将显示有关可用完成信息的更多信息。

+   如果你右键单击 VS Code 中的一个现有定理名称，编辑器将显示一个菜单，其中包含跳转到定理定义文件并找到附近类似定理的选项。

+   你可以使用 `apply?` 策略，它试图在库中找到相关的定理。

```py
example  :  0  ≤  a  ^  2  :=  by
  -- apply?
  exact  sq_nonneg  a 
```

要尝试在这个例子中使用 `apply?`，请删除 `exact` 命令并取消注释上一行。使用这些技巧，看看你是否能找到完成下一个例子所需的内容：

```py
example  (h  :  a  ≤  b)  :  c  -  exp  b  ≤  c  -  exp  a  :=  by
  sorry 
```

使用相同的技巧，确认 `linarith` 而不是 `apply?` 也可以完成工作。

这里是不等式的一个例子：

```py
example  :  2*a*b  ≤  a²  +  b²  :=  by
  have  h  :  0  ≤  a²  -  2*a*b  +  b²
  calc
  a²  -  2*a*b  +  b²  =  (a  -  b)²  :=  by  ring
  _  ≥  0  :=  by  apply  pow_two_nonneg

  calc
  2*a*b  =  2*a*b  +  0  :=  by  ring
  _  ≤  2*a*b  +  (a²  -  2*a*b  +  b²)  :=  add_le_add  (le_refl  _)  h
  _  =  a²  +  b²  :=  by  ring 
```

Mathlib 倾向于在二进制运算符（如 `*` 和 `^`）周围放置空格，但在本例中，更紧凑的格式提高了可读性。有几个值得注意的地方。首先，表达式 `s ≥ t` 在定义上是等价于 `t ≤ s` 的。原则上，这意味着应该能够互换使用它们。但是 Lean 的一些自动化工具没有识别出这种等价性，因此 Mathlib 倾向于更喜欢 `≤` 而不是 `≥`。其次，我们广泛使用了 `ring` 策略。这是一个真正的节省时间！最后，请注意，在第二个 `calc` 证明的第二行中，我们不需要写 `by exact add_le_add (le_refl _) h`，我们可以简单地写出证明项 `add_le_add (le_refl _) h`。

实际上，上述证明中唯一的巧妙之处在于找出假设 `h`。一旦我们有了它，第二个计算只涉及线性代数，而 `linarith` 可以处理它：

```py
example  :  2*a*b  ≤  a²  +  b²  :=  by
  have  h  :  0  ≤  a²  -  2*a*b  +  b²
  calc
  a²  -  2*a*b  +  b²  =  (a  -  b)²  :=  by  ring
  _  ≥  0  :=  by  apply  pow_two_nonneg
  linarith 
```

多么好！我们挑战你使用这些想法来证明以下定理。你可以使用定理 `abs_le'.mpr`。你还需要使用 `constructor` 策略将合取式拆分为两个目标；参见第 3.4 节。

```py
example  :  |a*b|  ≤  (a²  +  b²)/2  :=  by
  sorry

#check  abs_le'.mpr 
```

如果你设法解决了这个问题，恭喜你！你正朝着成为一位大师级形式化专家的道路上迈进。  ## 2.4\. 使用 apply 和 rw 的更多示例

实数上的 `min` 函数由以下三个事实唯一确定：

```py
#check  (min_le_left  a  b  :  min  a  b  ≤  a)
#check  (min_le_right  a  b  :  min  a  b  ≤  b)
#check  (le_min  :  c  ≤  a  →  c  ≤  b  →  c  ≤  min  a  b) 
```

你能猜出以类似方式描述 `max` 的定理的名称吗？

注意，我们必须通过编写 `min a b` 而不是 `min (a, b)` 来将 `min` 应用到一对参数 `a` 和 `b`。形式上，`min` 是一个类型为 `ℝ → ℝ → ℝ` 的函数。当我们用多个箭头写这样的类型时，约定是隐式括号向右结合，所以类型被解释为 `ℝ → (ℝ → ℝ)`。最终效果是，如果 `a` 和 `b` 的类型是 `ℝ`，那么 `min a` 的类型是 `ℝ → ℝ`，`min a b` 的类型是 `ℝ`，所以 `min` 的行为就像一个有两个参数的函数，正如我们所期望的那样。以这种方式处理多个参数被称为 *currying*，这是逻辑学家 Haskell Curry 的名字。

Lean 中的运算顺序可能需要一些时间来习惯。函数应用比中缀操作绑定得更紧密，所以表达式 `min a b + c` 被解释为 `(min a b) + c`。随着时间的推移，这些约定将变得自然而然。

使用定理 `le_antisymm`，我们可以证明如果每个数都小于或等于另一个数，那么两个实数是相等的。使用这个和上述事实，我们可以证明 `min` 是可交换的：

```py
example  :  min  a  b  =  min  b  a  :=  by
  apply  le_antisymm
  ·  show  min  a  b  ≤  min  b  a
  apply  le_min
  ·  apply  min_le_right
  apply  min_le_left
  ·  show  min  b  a  ≤  min  a  b
  apply  le_min
  ·  apply  min_le_right
  apply  min_le_left 
```

在这里，我们使用点来分隔不同目标证明。我们的用法不一致：在外层，我们使用点和缩进来表示两个目标，而对于嵌套证明，我们只使用点直到只剩下一个目标。这两种约定都是合理且有用的。我们还使用 `show` 策略来结构化证明并指示每个块中正在证明的内容。即使没有 `show` 命令，证明仍然有效，但使用它们可以使证明更容易阅读和维护。

你可能会觉得证明过程很重复。为了预示你以后将学习到的技能，我们注意到避免重复的一种方法就是陈述一个局部引理然后使用它：

```py
example  :  min  a  b  =  min  b  a  :=  by
  have  h  :  ∀  x  y  :  ℝ,  min  x  y  ≤  min  y  x  :=  by
  intro  x  y
  apply  le_min
  apply  min_le_right
  apply  min_le_left
  apply  le_antisymm
  apply  h
  apply  h 
```

我们将在 第 3.1 节 中更多地讨论全称量词，但在这里只需说，假设 `h` 表示对于任何 `x` 和 `y`，所期望的不等式都成立，而 `intro` 策略引入任意的 `x` 和 `y` 来建立结论。在 `le_antisymm` 之后的第一个 `apply` 隐式地使用了 `h a b`，而第二个则使用了 `h b a`。

另一种解决方案是使用 `repeat` 策略，它可以尽可能多次地应用策略（或一个块）。

```py
example  :  min  a  b  =  min  b  a  :=  by
  apply  le_antisymm
  repeat
  apply  le_min
  apply  min_le_right
  apply  min_le_left 
```

我们鼓励你将以下内容作为练习来证明。你可以使用上面描述的任何一种技巧来缩短第一个证明。

```py
example  :  max  a  b  =  max  b  a  :=  by
  sorry
example  :  min  (min  a  b)  c  =  min  a  (min  b  c)  :=  by
  sorry 
```

当然，你也可以尝试证明 `max` 的结合律。

一个有趣的事实是 `min` 在 `max` 上分配的方式与乘法在加法上分配的方式相同，反之亦然。换句话说，在实数上，我们有恒等式 `min a (max b c) = max (min a b) (min a c)`，以及将 `max` 和 `min` 交换的对应版本。但在下一节中，我们将看到这并不从 `≤` 的传递性和自反性以及上面列出的 `min` 和 `max` 的特征属性中得出。我们需要使用事实，即实数上的 `≤` 是一个 *全序*，也就是说，它满足 `∀ x y, x ≤ y ∨ y ≤ x`。这里的析取符号 `∨` 代表“或”。在第一种情况下，我们有 `min x y = x`，在第二种情况下，我们有 `min x y = y`。我们将在 第 3.5 节 中学习如何进行情况推理，但到目前为止，我们将坚持不需要情况分解的例子。

这里有一个这样的例子：

```py
theorem  aux  :  min  a  b  +  c  ≤  min  (a  +  c)  (b  +  c)  :=  by
  sorry
example  :  min  a  b  +  c  =  min  (a  +  c)  (b  +  c)  :=  by
  sorry 
```

显然，`aux` 提供了证明等式所需的两个不等式之一，但将其应用于合适的值也会得到另一个方向。作为一个提示，你可以使用定理 `add_neg_cancel_right` 和 `linarith` 策略。

Lean 的命名约定在库中三角不等式的名称中得到了体现：

```py
#check  (abs_add  :  ∀  a  b  :  ℝ,  |a  +  b|  ≤  |a|  +  |b|) 
```

使用它来证明以下变体，同时使用 `add_sub_cancel_right`：

```py
example  :  |a|  -  |b|  ≤  |a  -  b|  :=
  sorry
end 
```

看看你是否能在三行或更少的代码中完成这个任务。你可以使用定理 `sub_add_cancel`。

在接下来的章节中，我们还将使用的一个重要关系是自然数上的可除性关系，`x ∣ y`。请注意：可除性符号 *不是* 你键盘上的普通横线。相反，它是一个通过在 VS Code 中输入 `\|` 获得的 unicode 字符。按照惯例，Mathlib 在定理名称中使用 `dvd` 来指代它。

```py
example  (h₀  :  x  ∣  y)  (h₁  :  y  ∣  z)  :  x  ∣  z  :=
  dvd_trans  h₀  h₁

example  :  x  ∣  y  *  x  *  z  :=  by
  apply  dvd_mul_of_dvd_left
  apply  dvd_mul_left

example  :  x  ∣  x  ^  2  :=  by
  apply  dvd_mul_left 
```

在最后一个例子中，指数是一个自然数，应用 `dvd_mul_left` 强制 Lean 展开定义 `x²` 为 `x¹ * x`。看看你是否能猜出你需要证明以下定理的名称：

```py
example  (h  :  x  ∣  w)  :  x  ∣  y  *  (x  *  z)  +  x  ^  2  +  w  ^  2  :=  by
  sorry
end 
```

关于可除性，*最大公约数* `gcd` 和最小公倍数 `lcm` 与 `min` 和 `max` 类似。由于每个数都能整除 `0`，所以 `0` 真的是可除性关系中的最大元素：

```py
variable  (m  n  :  ℕ)

#check  (Nat.gcd_zero_right  n  :  Nat.gcd  n  0  =  n)
#check  (Nat.gcd_zero_left  n  :  Nat.gcd  0  n  =  n)
#check  (Nat.lcm_zero_right  n  :  Nat.lcm  n  0  =  0)
#check  (Nat.lcm_zero_left  n  :  Nat.lcm  0  n  =  0) 
```

看看你是否能猜出你需要证明以下定理的名称：

```py
example  :  Nat.gcd  m  n  =  Nat.gcd  n  m  :=  by
  sorry 
```

提示：你可以使用 `dvd_antisymm`，但如果你这样做，Lean 会抱怨表达式在通用定理和版本 `Nat.dvd_antisymm`（专门针对自然数的版本）之间是模糊的。你可以使用 `_root_.dvd_antisymm` 来指定通用版本；两者都可以工作。

在 第 2.2 节 中，我们看到了许多常见的关于实数的恒等式在更一般的代数结构中成立，例如交换环。我们可以使用任何我们想要的公理来描述代数结构，而不仅仅是方程。例如，一个 *偏序* 由一个集合和一个二元关系组成，该关系是自反的、传递的和反对称的，类似于实数上的 `≤`。Lean 了解偏序：

```py
variable  {α  :  Type*}  [PartialOrder  α]
variable  (x  y  z  :  α)

#check  x  ≤  y
#check  (le_refl  x  :  x  ≤  x)
#check  (le_trans  :  x  ≤  y  →  y  ≤  z  →  x  ≤  z)
#check  (le_antisymm  :  x  ≤  y  →  y  ≤  x  →  x  =  y) 
```

在这里，我们采用 Mathlib 的约定，使用像 `α`、`β` 和 `γ`（输入为 `\a`、`\b` 和 `\g`）这样的字母表示任意类型。该库通常使用像 `R` 和 `G` 这样的字母表示代数结构如环和群的载体，但通常在几乎没有与之关联的结构时使用希腊字母表示类型。

与任何偏序 `≤` 相关联的，还有一个 *严格偏序* `<`，它在某种程度上类似于实数中的 `<`。在这个顺序中说 `x` 小于 `y` 等价于说它小于或等于 `y` 但不等于 `y`。

```py
#check  x  <  y
#check  (lt_irrefl  x  :  ¬  (x  <  x))
#check  (lt_trans  :  x  <  y  →  y  <  z  →  x  <  z)
#check  (lt_of_le_of_lt  :  x  ≤  y  →  y  <  z  →  x  <  z)
#check  (lt_of_lt_of_le  :  x  <  y  →  y  ≤  z  →  x  <  z)

example  :  x  <  y  ↔  x  ≤  y  ∧  x  ≠  y  :=
  lt_iff_le_and_ne 
```

在这个例子中，符号 `∧` 代表“和”，符号 `¬` 代表“非”，`x ≠ y` 简写为 `¬ (x = y)`。在 第三章 中，您将学习如何使用这些逻辑连接词来 *证明* `<` 具有指示的性质。

一个 *拉丁格* 是一个扩展了偏序并带有 `⊓` 和 `⊔` 操作的结构，这些操作类似于实数上的 `min` 和 `max`：

```py
variable  {α  :  Type*}  [Lattice  α]
variable  (x  y  z  :  α)

#check  x  ⊓  y
#check  (inf_le_left  :  x  ⊓  y  ≤  x)
#check  (inf_le_right  :  x  ⊓  y  ≤  y)
#check  (le_inf  :  z  ≤  x  →  z  ≤  y  →  z  ≤  x  ⊓  y)
#check  x  ⊔  y
#check  (le_sup_left  :  x  ≤  x  ⊔  y)
#check  (le_sup_right  :  y  ≤  x  ⊔  y)
#check  (sup_le  :  x  ≤  z  →  y  ≤  z  →  x  ⊔  y  ≤  z) 
```

`⊓` 和 `⊔` 的特征使得可以称它们分别为 *最大下界* 和 *最小上界*。您可以在 VS code 中使用 `\glb` 和 `\lub` 来输入它们。这些符号也常被称为 *下确界* 和 *上确界*，Mathlib 在定理名称中称它们为 `inf` 和 `sup`。为了进一步复杂化问题，它们也常被称为 *交* 和 *并*。因此，如果您与拉丁格一起工作，您必须记住以下字典：

+   `⊓` 是 *最大下界*、*下确界* 或 *交*。

+   `⊔` 是 *最小上界*、*上确界* 或 *并*。

拉丁格的一些实例包括：

+   任何全序，如整数或实数上的 `≤`

+   在某个域的子集集合上的 `∩` 和 `∪`，具有排序 `⊆`

+   在布尔真值上的 `∧` 和 `∨`，其中排序 `x ≤ y` 当且仅当 `x` 为假或 `y` 为真

+   `gcd` 和 `lcm` 在自然数（或正自然数）上，具有可除性排序 `∣`

+   向量空间线性子空间的集合，其中最大下界由交集给出，最小上界由两个空间的和给出，排序是包含关系

+   在一个集合（或在 Lean 中，一个类型）上的拓扑集合，其中两个拓扑的最小下界是由它们的并集生成的拓扑，最小上界是它们的交集，且排序是反向包含

你可以检查，就像`min`/`max`和`gcd`/`lcm`一样，你可以仅使用它们的特征公理以及`le_refl`和`le_trans`来证明下确界和上确界的交换性和结合性。

当看到目标`x ≤ z`时使用`apply le_trans`不是一个好主意。实际上，Lean 没有方法猜测我们想要使用哪个中间元素`y`。因此，`apply le_trans`会产生三个看起来像`x ≤ ?a`、`?a ≤ z`和`α`的目标，其中`?a`（可能有一个更复杂的自动生成的名称）代表神秘的`y`。最后一个目标，类型为`α`，是提供`y`的值。它最后出现，因为 Lean 希望从第一个目标`x ≤ ?a`的证明中自动推断它。为了避免这种不吸引人的情况，你可以使用`calc`策略显式地提供`y`。或者，你可以使用`trans`策略，它将`y`作为参数，并产生预期的目标`x ≤ y`和`y ≤ z`。当然，你也可以通过直接提供一个完整的证明来避免这个问题，例如`exact le_trans inf_le_left inf_le_right`，但这需要更多的计划。

```py
example  :  x  ⊓  y  =  y  ⊓  x  :=  by
  sorry

example  :  x  ⊓  y  ⊓  z  =  x  ⊓  (y  ⊓  z)  :=  by
  sorry

example  :  x  ⊔  y  =  y  ⊔  x  :=  by
  sorry

example  :  x  ⊔  y  ⊔  z  =  x  ⊔  (y  ⊔  z)  :=  by
  sorry 
```

你可以在 Mathlib 中找到这些定理，分别命名为`inf_comm`、`inf_assoc`、`sup_comm`和`sup_assoc`。

另一个很好的练习是仅使用那些公理来证明**吸收律**：

```py
theorem  absorb1  :  x  ⊓  (x  ⊔  y)  =  x  :=  by
  sorry

theorem  absorb2  :  x  ⊔  x  ⊓  y  =  x  :=  by
  sorry 
```

这些可以在 Mathlib 中以`inf_sup_self`和`sup_inf_self`的名称找到。

满足额外的恒等式`x ⊓ (y ⊔ z) = (x ⊓ y) ⊔ (x ⊓ z)`和`x ⊔ (y ⊓ z) = (x ⊔ y) ⊓ (x ⊔ z)`的格称为**分配格**。Lean 也知道这些：

```py
variable  {α  :  Type*}  [DistribLattice  α]
variable  (x  y  z  :  α)

#check  (inf_sup_left  x  y  z  :  x  ⊓  (y  ⊔  z)  =  x  ⊓  y  ⊔  x  ⊓  z)
#check  (inf_sup_right  x  y  z  :  (x  ⊔  y)  ⊓  z  =  x  ⊓  z  ⊔  y  ⊓  z)
#check  (sup_inf_left  x  y  z  :  x  ⊔  y  ⊓  z  =  (x  ⊔  y)  ⊓  (x  ⊔  z))
#check  (sup_inf_right  x  y  z  :  x  ⊓  y  ⊔  z  =  (x  ⊔  z)  ⊓  (y  ⊔  z)) 
```

在`⊓`和`⊔`的交换性给定的情况下，左右版本很容易被证明是等价的。通过提供一个非分配格的显式描述，其中包含有限多个元素，来证明并非每个格都是分配的，这是一个很好的练习。同样，证明在任何格中，分配律中的一个蕴含另一个也是一个很好的练习：

```py
variable  {α  :  Type*}  [Lattice  α]
variable  (a  b  c  :  α)

example  (h  :  ∀  x  y  z  :  α,  x  ⊓  (y  ⊔  z)  =  x  ⊓  y  ⊔  x  ⊓  z)  :  a  ⊔  b  ⊓  c  =  (a  ⊔  b)  ⊓  (a  ⊔  c)  :=  by
  sorry

example  (h  :  ∀  x  y  z  :  α,  x  ⊔  y  ⊓  z  =  (x  ⊔  y)  ⊓  (x  ⊔  z))  :  a  ⊓  (b  ⊔  c)  =  a  ⊓  b  ⊔  a  ⊓  c  :=  by
  sorry 
```

可以将公理结构组合成更大的结构。例如，一个**严格有序环**由一个环及其载体上的偏序组成，该偏序满足额外的公理，这些公理说明环运算与顺序是兼容的：

```py
variable  {R  :  Type*}  [Ring  R]  [PartialOrder  R]  [IsStrictOrderedRing  R]
variable  (a  b  c  :  R)

#check  (add_le_add_left  :  a  ≤  b  →  ∀  c,  c  +  a  ≤  c  +  b)
#check  (mul_pos  :  0  <  a  →  0  <  b  →  0  <  a  *  b) 
```

第三章将提供从`mul_pos`和`<`的定义中推导以下内容的方法：

```py
#check  (mul_nonneg  :  0  ≤  a  →  0  ≤  b  →  0  ≤  a  *  b) 
```

然后是一个扩展练习，即证明许多用于推理算术和实数顺序的常见事实对于任何有序环都是通用的。这里有一些你可以尝试的例子，仅使用环、偏序以及最后两个例子中列出的事实（注意，这些环不假设是交换的，因此没有环策略）：

```py
example  (h  :  a  ≤  b)  :  0  ≤  b  -  a  :=  by
  sorry

example  (h:  0  ≤  b  -  a)  :  a  ≤  b  :=  by
  sorry

example  (h  :  a  ≤  b)  (h'  :  0  ≤  c)  :  a  *  c  ≤  b  *  c  :=  by
  sorry 
```

最后，这里有一个最后的例子。一个**度量空间**由一个集合和一个距离概念`dist x y`组成，该距离将任何一对元素映射到一个实数。距离函数假设满足以下公理：

```py
variable  {X  :  Type*}  [MetricSpace  X]
variable  (x  y  z  :  X)

#check  (dist_self  x  :  dist  x  x  =  0)
#check  (dist_comm  x  y  :  dist  x  y  =  dist  y  x)
#check  (dist_triangle  x  y  z  :  dist  x  z  ≤  dist  x  y  +  dist  y  z) 
```

在掌握这一部分后，你可以证明以下公理意味着距离总是非负的：

```py
example  (x  y  :  X)  :  0  ≤  dist  x  y  :=  by
  sorry 
```

我们推荐使用定理 `nonneg_of_mul_nonneg_left`。正如你可能猜到的，这个定理在 Mathlib 中被称为 `dist_nonneg`。上一节 下一节

* * *

© 版权 2020-2025，Jeremy Avigad，Patrick Massot。文本许可协议为 CC BY 4.0。

使用 [Sphinx](https://www.sphinx-doc.org/) 和由 [Read the Docs](https://readthedocs.org) 提供的 [主题](https://github.com/readthedocs/sphinx_rtd_theme) 构建。本章旨在向您介绍 Lean 中数学推理的精髓：计算、应用引理和定理，以及关于通用结构的推理。

## 2.1. 计算

我们通常学习进行数学计算，而不把它们当作证明。但当我们像 Lean 所要求的那样证明计算的每一步，最终结果是证明计算的左边等于右边。

在 Lean 中，陈述一个定理等同于陈述一个目标，即证明该定理的目标。Lean 提供了重写策略 `rw`，在目标中将等式的左边替换为右边。如果 `a`、`b` 和 `c` 是实数，`mul_assoc a b c` 是等式 `a * b * c = a * (b * c)`，而 `mul_comm a b` 是等式 `a * b = b * a`。Lean 提供了自动化，通常可以消除明确引用此类事实的需要，但它们对于说明目的很有用。在 Lean 中，乘法从左结合，因此 `mul_assoc` 的左边也可以写成 `(a * b) * c`。然而，通常来说，注意 Lean 的符号约定并省略括号是好的风格。

让我们尝试一下 `rw`。

```py
example  (a  b  c  :  ℝ)  :  a  *  b  *  c  =  b  *  (a  *  c)  :=  by
  rw  [mul_comm  a  b]
  rw  [mul_assoc  b  a  c] 
```

在相关示例文件的开始部分，`import` 行从 Mathlib 导入实数理论以及有用的自动化。为了简洁起见，我们在教科书中通常省略此类信息。

你可以随意修改以查看会发生什么。在 VS Code 中，你可以将 `ℝ` 字符键入为 `\R` 或 `\real`。符号只有在按下空格键或制表键后才会出现。当你阅读 Lean 文件时，将鼠标悬停在符号上，VS Code 将显示可以用来输入它的语法。如果你好奇想查看所有可用的缩写，你可以按 Ctrl-Shift-P，然后键入缩写以访问 `Lean 4: 显示 Unicode 输入缩写` 命令。如果你的键盘上没有容易访问的反斜杠，你可以通过更改 `lean4.input.leader` 设置来更改前导字符。

当光标在策略证明的中间时，Lean 在*Lean Infoview*窗口中报告当前的*证明状态*。当你将光标移过证明的每一步时，你可以看到状态的变化。Lean 中的一个典型证明状态可能如下所示：

```py
1  goal
x  y  :  ℕ,
h₁  :  Prime  x,
h₂  :  ¬Even  x,
h₃  :  y  >  x
⊢  y  ≥  4 
```

在以`⊢`开始的行之前的行表示*上下文*：它们是当前正在使用的对象和假设。在这个例子中，这些包括两个对象，`x`和`y`，每个都是自然数。它们还包括三个假设，标记为`h₁`、`h₂`和`h₃`。在 Lean 中，上下文中的所有内容都用标识符标记。你可以输入这些带下标的标签作为`h\1`、`h\2`和`h\3`，但任何合法的标识符都可以：你可以使用`h1`、`h2`、`h3`代替，或者`foo`、`bar`和`baz`。最后一行代表*目标*，即要证明的事实。有时人们用*目标*来表示要证明的事实，用*目标*来表示上下文和目标的组合。在实践中，通常可以清楚地理解意图的含义。

尝试证明这些恒等式，在每种情况下用策略证明替换`sorry`。使用`rw`策略，你可以使用左箭头（`\l`）来反转一个恒等式。例如，`rw [← mul_assoc a b c]`将当前目标中的`a * (b * c)`替换为`a * b * c`。注意，指向左边的箭头指的是从右到左在`mul_assoc`提供的恒等式中移动，它与目标左右两侧无关。

```py
example  (a  b  c  :  ℝ)  :  c  *  b  *  a  =  b  *  (a  *  c)  :=  by
  sorry

example  (a  b  c  :  ℝ)  :  a  *  (b  *  c)  =  b  *  (a  *  c)  :=  by
  sorry 
```

你也可以在不提供参数的情况下使用像`mul_assoc`和`mul_comm`这样的恒等式。在这种情况下，重写策略尝试将目标左边的表达式与目标中的表达式匹配，使用它找到的第一个模式。

```py
example  (a  b  c  :  ℝ)  :  a  *  b  *  c  =  b  *  c  *  a  :=  by
  rw  [mul_assoc]
  rw  [mul_comm] 
```

你还可以提供*部分*信息。例如，`mul_comm a`与形式为`a * ?`的任何模式匹配，并将其重写为`? * a`。尝试在不提供任何参数的情况下执行这些示例中的第一个，以及在只有一个参数的情况下执行第二个。

```py
example  (a  b  c  :  ℝ)  :  a  *  (b  *  c)  =  b  *  (c  *  a)  :=  by
  sorry

example  (a  b  c  :  ℝ)  :  a  *  (b  *  c)  =  b  *  (a  *  c)  :=  by
  sorry 
```

你还可以使用来自本地上下文的事实与`rw`结合。

```py
example  (a  b  c  d  e  f  :  ℝ)  (h  :  a  *  b  =  c  *  d)  (h'  :  e  =  f)  :  a  *  (b  *  e)  =  c  *  (d  *  f)  :=  by
  rw  [h']
  rw  [←  mul_assoc]
  rw  [h]
  rw  [mul_assoc] 
```

尝试这些，对于第二个使用定理`sub_self`：

```py
example  (a  b  c  d  e  f  :  ℝ)  (h  :  b  *  c  =  e  *  f)  :  a  *  b  *  c  *  d  =  a  *  e  *  f  *  d  :=  by
  sorry

example  (a  b  c  d  :  ℝ)  (hyp  :  c  =  b  *  a  -  d)  (hyp'  :  d  =  a  *  b)  :  c  =  0  :=  by
  sorry 
```

可以通过在方括号内用逗号分隔的相关身份来执行多个重写命令。

```py
example  (a  b  c  d  e  f  :  ℝ)  (h  :  a  *  b  =  c  *  d)  (h'  :  e  =  f)  :  a  *  (b  *  e)  =  c  *  (d  *  f)  :=  by
  rw  [h',  ←  mul_assoc,  h,  mul_assoc] 
```

你仍然可以通过在任何重写列表中将光标放在逗号之后来看到增量进步。

另一个技巧是，我们可以在示例或定理之外一次性声明变量。然后 Lean 会自动包含它们。

```py
variable  (a  b  c  d  e  f  :  ℝ)

example  (h  :  a  *  b  =  c  *  d)  (h'  :  e  =  f)  :  a  *  (b  *  e)  =  c  *  (d  *  f)  :=  by
  rw  [h',  ←  mul_assoc,  h,  mul_assoc] 
```

检查上述证明开始时的策略状态，可以发现 Lean 确实包含了所有变量。我们可以通过将其放在`section ... end`块中来限定声明的范围。最后，从介绍中回忆起 Lean 为我们提供了一个命令来确定表达式的类型：

```py
section
variable  (a  b  c  :  ℝ)

#check  a
#check  a  +  b
#check  (a  :  ℝ)
#check  mul_comm  a  b
#check  (mul_comm  a  b  :  a  *  b  =  b  *  a)
#check  mul_assoc  c  a  b
#check  mul_comm  a
#check  mul_comm

end 
```

`#check` 命令适用于对象和事实。对于 `#check a` 命令，Lean 报告 `a` 的类型为 `ℝ`。对于 `#check mul_comm a b` 命令，Lean 报告 `mul_comm a b` 是事实 `a * b = b * a` 的证明。命令 `#check (a : ℝ)` 表明我们期望 `a` 的类型是 `ℝ`，如果这不是情况，Lean 将引发错误。我们将在稍后解释最后三个 `#check` 命令的输出，但在此同时，您可以查看它们，并尝试一些自己的 `#check` 命令。

让我们尝试一些更多的例子。定理 `two_mul a` 表明 `2 * a = a + a`。定理 `add_mul` 和 `mul_add` 表达了乘法对加法的分配律，而定理 `add_assoc` 表达了加法的结合律。使用 `#check` 命令来查看精确的陈述。

```py
example  :  (a  +  b)  *  (a  +  b)  =  a  *  a  +  2  *  (a  *  b)  +  b  *  b  :=  by
  rw  [mul_add,  add_mul,  add_mul]
  rw  [←  add_assoc,  add_assoc  (a  *  a)]
  rw  [mul_comm  b  a,  ←  two_mul] 
```

虽然可以通过在编辑器中逐步执行来弄清楚这个证明的情况，但单独阅读起来很困难。Lean 提供了一种更结构化的方式来编写这样的证明，使用 `calc` 关键字。

```py
example  :  (a  +  b)  *  (a  +  b)  =  a  *  a  +  2  *  (a  *  b)  +  b  *  b  :=
  calc
  (a  +  b)  *  (a  +  b)  =  a  *  a  +  b  *  a  +  (a  *  b  +  b  *  b)  :=  by
  rw  [mul_add,  add_mul,  add_mul]
  _  =  a  *  a  +  (b  *  a  +  a  *  b)  +  b  *  b  :=  by
  rw  [←  add_assoc,  add_assoc  (a  *  a)]
  _  =  a  *  a  +  2  *  (a  *  b)  +  b  *  b  :=  by
  rw  [mul_comm  b  a,  ←  two_mul] 
```

注意，证明并不以 `by` 开头：以 `calc` 开头的表达式是一个 *证明项*。`calc` 表达式也可以用在策略证明中，但 Lean 将其解释为使用生成的证明项来解决目标的指令。`calc` 语法很挑剔：下划线和论证必须采用上述格式。Lean 使用缩进来确定诸如策略块或 `calc` 块的开始和结束位置；尝试更改上述证明中的缩进以查看会发生什么。

写一个 `calc` 证明的一种方法首先是用 `sorry` 策略进行论证，确保 Lean 接受这些表达式模这些，然后使用策略来论证各个步骤。

```py
example  :  (a  +  b)  *  (a  +  b)  =  a  *  a  +  2  *  (a  *  b)  +  b  *  b  :=
  calc
  (a  +  b)  *  (a  +  b)  =  a  *  a  +  b  *  a  +  (a  *  b  +  b  *  b)  :=  by
  sorry
  _  =  a  *  a  +  (b  *  a  +  a  *  b)  +  b  *  b  :=  by
  sorry
  _  =  a  *  a  +  2  *  (a  *  b)  +  b  *  b  :=  by
  sorry 
```

尝试使用纯 `rw` 证明和更结构化的 `calc` 证明来证明以下恒等式：

```py
example  :  (a  +  b)  *  (c  +  d)  =  a  *  c  +  a  *  d  +  b  *  c  +  b  *  d  :=  by
  sorry 
```

以下练习稍微有些挑战性。您可以使用下面列出的定理。

```py
example  (a  b  :  ℝ)  :  (a  +  b)  *  (a  -  b)  =  a  ^  2  -  b  ^  2  :=  by
  sorry

#check  pow_two  a
#check  mul_sub  a  b  c
#check  add_mul  a  b  c
#check  add_sub  a  b  c
#check  sub_sub  a  b  c
#check  add_zero  a 
```

我们也可以在假设的上下文中进行重写。例如，`rw [mul_comm a b] at hyp` 将假设 `hyp` 中的 `a * b` 替换为 `b * a`。

```py
example  (a  b  c  d  :  ℝ)  (hyp  :  c  =  d  *  a  +  b)  (hyp'  :  b  =  a  *  d)  :  c  =  2  *  a  *  d  :=  by
  rw  [hyp']  at  hyp
  rw  [mul_comm  d  a]  at  hyp
  rw  [←  two_mul  (a  *  d)]  at  hyp
  rw  [←  mul_assoc  2  a  d]  at  hyp
  exact  hyp 
```

在最后一步，`exact` 策略可以使用 `hyp` 来解决目标，因为此时 `hyp` 与目标完全匹配。

我们通过指出 Mathlib 提供了一个有用的自动化功能，即 `ring` 策略，该策略旨在证明任何交换环中的恒等式，只要它们纯粹从环公理中得出，而不使用任何局部假设，来结束本节。

```py
example  :  c  *  b  *  a  =  b  *  (a  *  c)  :=  by
  ring

example  :  (a  +  b)  *  (a  +  b)  =  a  *  a  +  2  *  (a  *  b)  +  b  *  b  :=  by
  ring

example  :  (a  +  b)  *  (a  -  b)  =  a  ^  2  -  b  ^  2  :=  by
  ring

example  (hyp  :  c  =  d  *  a  +  b)  (hyp'  :  b  =  a  *  d)  :  c  =  2  *  a  *  d  :=  by
  rw  [hyp,  hyp']
  ring 
```

当我们导入 `Mathlib.Data.Real.Basic` 时，`ring` 策略是间接导入的，但我们在下一节将看到它可以用作对除了实数以外的结构进行计算。可以使用命令 `import Mathlib.Tactic` 明确导入。我们将看到有类似策略用于其他常见的代数结构。

有一种 `rw` 的变体称为 `nth_rw`，它允许你只替换目标中特定实例的表达式。可能的匹配从 1 开始枚举，所以在这个例子中，`nth_rw 2 [h]` 将 `a + b` 的第二个出现替换为 `c`。

```py
example  (a  b  c  :  ℕ)  (h  :  a  +  b  =  c)  :  (a  +  b)  *  (a  +  b)  =  a  *  c  +  b  *  c  :=  by
  nth_rw  2  [h]
  rw  [add_mul] 
```

## 2.2\. 在代数结构中证明恒等式

从数学上讲，一个环由一组对象 $R$、运算 $+$ 和 $\times$、常数 $0$ 和 $1$ 以及一个运算 $x \mapsto -x$ 组成，该运算满足以下条件：

+   $R$ 与 $+$ 是一个 *阿贝尔群*，其中 $0$ 是加法单位元，负号是逆元。

+   乘法与单位元 $1$ 结合，并且乘法对加法分配。

在 Lean 中，对象的集合被表示为一个 *类型*，`R`。环的公理如下：

```py
variable  (R  :  Type*)  [Ring  R]

#check  (add_assoc  :  ∀  a  b  c  :  R,  a  +  b  +  c  =  a  +  (b  +  c))
#check  (add_comm  :  ∀  a  b  :  R,  a  +  b  =  b  +  a)
#check  (zero_add  :  ∀  a  :  R,  0  +  a  =  a)
#check  (neg_add_cancel  :  ∀  a  :  R,  -a  +  a  =  0)
#check  (mul_assoc  :  ∀  a  b  c  :  R,  a  *  b  *  c  =  a  *  (b  *  c))
#check  (mul_one  :  ∀  a  :  R,  a  *  1  =  a)
#check  (one_mul  :  ∀  a  :  R,  1  *  a  =  a)
#check  (mul_add  :  ∀  a  b  c  :  R,  a  *  (b  +  c)  =  a  *  b  +  a  *  c)
#check  (add_mul  :  ∀  a  b  c  :  R,  (a  +  b)  *  c  =  a  *  c  +  b  *  c) 
```

你将在以后学习到第一行中方括号的作用，但在此阶段，只需说这个声明给我们一个类型 `R` 和 `R` 上的环结构即可。Lean 然后允许我们使用 `R` 的元素进行通用的环符号表示，并利用关于环的定理库。

一些定理的名称应该看起来很熟悉：它们正是我们在上一节中使用实数进行计算时用到的那些。Lean 不仅擅长证明关于自然数和整数等具体数学结构的性质，而且擅长证明关于抽象结构（如环）的性质。此外，Lean 支持关于抽象和具体结构的 *通用推理*，并且可以训练它识别适当的实例。因此，任何关于环的定理都可以应用于具体的环，如整数环 `ℤ`、有理数环 `ℚ` 和复数环 `ℂ`。它也可以应用于任何扩展环的抽象结构的实例，例如任何有序环或任何域。

然而，并非所有实数的性质在任意环中都成立。例如，实数上的乘法是交换的，但这一点在一般情况下并不成立。如果你已经学过线性代数课程，你会认识到，对于每一个 $n$，实数的 $n$ 阶矩阵形成一个环，其中交换律通常不成立。如果我们声明 `R` 为一个 *交换* 环，实际上，当我们将 `ℝ` 替换为 `R` 时，上一节中所有定理仍然成立。

```py
variable  (R  :  Type*)  [CommRing  R]
variable  (a  b  c  d  :  R)

example  :  c  *  b  *  a  =  b  *  (a  *  c)  :=  by  ring

example  :  (a  +  b)  *  (a  +  b)  =  a  *  a  +  2  *  (a  *  b)  +  b  *  b  :=  by  ring

example  :  (a  +  b)  *  (a  -  b)  =  a  ^  2  -  b  ^  2  :=  by  ring

example  (hyp  :  c  =  d  *  a  +  b)  (hyp'  :  b  =  a  *  d)  :  c  =  2  *  a  *  d  :=  by
  rw  [hyp,  hyp']
  ring 
```

我们留给你们去检查其他所有证明是否保持不变。注意，当证明很短，如 `by ring` 或 `by linarith` 或 `by sorry` 时，将其放在 `by` 的同一行上是常见（且允许）的。良好的证明风格应该在简洁性和可读性之间取得平衡。

本节的目标是加强你在上一节中发展的技能，并将它们应用于对环进行公理推理。我们将从上面列出的公理开始，并使用它们来推导其他事实。我们证明的大多数事实已经在 Mathlib 中。我们将给出我们证明的版本相同的名称，以帮助你学习库的内容以及命名约定。

Lean 提供了一种类似于编程语言中使用的组织机制：当一个定义或定理`foo`在*命名空间*`bar`中引入时，它的全名是`bar.foo`。命令`open bar`稍后*打开*命名空间，这允许我们使用较短的名称`foo`。为了避免由于名称冲突而引起的错误，在下一个例子中，我们将我们的库定理版本放在一个新的命名空间`MyRing`中。

下一例子表明，我们不需要`add_zero`或`add_neg_cancel`作为环公理，因为它们可以从其他公理中得出。

```py
namespace  MyRing
variable  {R  :  Type*}  [Ring  R]

theorem  add_zero  (a  :  R)  :  a  +  0  =  a  :=  by  rw  [add_comm,  zero_add]

theorem  add_neg_cancel  (a  :  R)  :  a  +  -a  =  0  :=  by  rw  [add_comm,  neg_add_cancel]

#check  MyRing.add_zero
#check  add_zero

end  MyRing 
```

最终的效果是，我们可以暂时重新证明库中的一个定理，然后继续使用库版本。但不要作弊！在接下来的练习中，请务必只使用我们在本节中之前已经证明过的关于环的一般事实。

（如果你仔细观察，你可能已经注意到我们在`(R : Type*)`中使用了圆括号，而在`{R : Type*}`中使用了花括号。这表明`R`是一个*隐式参数*。我们稍后会解释这意味着什么，但在此期间请不要担心。）

这里有一个有用的定理：

```py
theorem  neg_add_cancel_left  (a  b  :  R)  :  -a  +  (a  +  b)  =  b  :=  by
  rw  [←  add_assoc,  neg_add_cancel,  zero_add] 
```

证明伴随版本：

```py
theorem  add_neg_cancel_right  (a  b  :  R)  :  a  +  b  +  -b  =  a  :=  by
  sorry 
```

使用这些来证明以下内容：

```py
theorem  add_left_cancel  {a  b  c  :  R}  (h  :  a  +  b  =  a  +  c)  :  b  =  c  :=  by
  sorry

theorem  add_right_cancel  {a  b  c  :  R}  (h  :  a  +  b  =  c  +  b)  :  a  =  c  :=  by
  sorry 
```

足够的计划下，你可以用三次重写来完成每一个。

我们现在将解释花括号的使用。想象一下，你处于一个情境，其中你的上下文中有`a`、`b`和`c`，以及一个假设`h : a + b = a + c`，你想要得出结论`b = c`。在 Lean 中，你可以像对对象一样对假设和事实应用定理，所以你可能认为`add_left_cancel a b c h`是事实`b = c`的证明。但请注意，明确写出`a`、`b`和`c`是多余的，因为假设`h`清楚地表明了这些是我们心中的对象。在这种情况下，输入几个额外的字符并不麻烦，但如果我们要将`add_left_cancel`应用于更复杂的表达式，编写它们将会很繁琐。在这些情况下，Lean 允许我们将参数标记为*隐式*，这意味着它们应该被省略，并通过其他方式推断，例如后续的参数和假设。`{a b c : R}`中的花括号正是这样做的。所以，给定上述定理的陈述，正确的表达式仅仅是`add_left_cancel h`。

为了说明，让我们展示`a * 0 = 0`可以从环公理中得出。

```py
theorem  mul_zero  (a  :  R)  :  a  *  0  =  0  :=  by
  have  h  :  a  *  0  +  a  *  0  =  a  *  0  +  0  :=  by
  rw  [←  mul_add,  add_zero,  add_zero]
  rw  [add_left_cancel  h] 
```

我们使用了一个新技巧！如果你逐步查看证明，你可以看到发生了什么。`have` 策略引入了一个新的目标，`a * 0 + a * 0 = a * 0 + 0`，与原始目标具有相同的环境。下一行缩进的事实表明 Lean 预期一个策略块，该块用于证明这个新目标。因此，缩进促进了模块化证明风格：缩进的子证明建立了由 `have` 引入的目标。之后，我们回到证明原始目标，除了增加了一个新的假设 `h`：证明它之后，我们现在可以自由地使用它。此时，目标正好是 `add_left_cancel h` 的结果。

我们同样可以用 `apply add_left_cancel h` 或 `exact add_left_cancel h` 来结束证明。`exact` 策略的参数是一个证明项，它完全证明了当前目标，而不创建任何新的目标。`apply` 策略是一个变体，其参数不一定是完整的证明。缺失的部分要么由 Lean 自动推断，要么成为需要证明的新目标。虽然 `exact` 策略在技术上可能是多余的，因为它严格不如 `apply` 强大，但它使证明脚本对人类读者来说更清晰，并且在库演变时更容易维护。

记住乘法不一定假设是交换的，所以下面的定理也需要一些工作。

```py
theorem  zero_mul  (a  :  R)  :  0  *  a  =  0  :=  by
  sorry 
```

到现在为止，你也应该能够用证明来替换下一个练习中的每个 `sorry`，仍然只使用本节中我们建立的关于环的事实。

```py
theorem  neg_eq_of_add_eq_zero  {a  b  :  R}  (h  :  a  +  b  =  0)  :  -a  =  b  :=  by
  sorry

theorem  eq_neg_of_add_eq_zero  {a  b  :  R}  (h  :  a  +  b  =  0)  :  a  =  -b  :=  by
  sorry

theorem  neg_zero  :  (-0  :  R)  =  0  :=  by
  apply  neg_eq_of_add_eq_zero
  rw  [add_zero]

theorem  neg_neg  (a  :  R)  :  -  -a  =  a  :=  by
  sorry 
```

在第三个定理中，我们不得不使用注释 `(-0 : R)` 而不是 `0`，因为没有指定 `R`，Lean 就无法推断我们指的是哪个 `0`，默认情况下它会被解释为自然数。

在 Lean 中，环中的减法可以证明等于加法上的相反数。

```py
example  (a  b  :  R)  :  a  -  b  =  a  +  -b  :=
  sub_eq_add_neg  a  b 
```

在实数上，它是这样 *定义* 的：

```py
example  (a  b  :  ℝ)  :  a  -  b  =  a  +  -b  :=
  rfl

example  (a  b  :  ℝ)  :  a  -  b  =  a  +  -b  :=  by
  rfl 
```

证明项 `rfl` 是“自反性”的缩写。将其作为 `a - b = a + -b` 的证明，迫使 Lean 展开定义并识别出两边是相同的。`rfl` 策略做的是同样的事情。这是 Lean 内在逻辑中所谓的 *定义等价* 的一个例子。这意味着不仅可以用 `sub_eq_add_neg` 重新编写以替换 `a - b = a + -b`，而且在某些上下文中，当处理实数时，你可以互换地使用方程的两边。例如，你现在有足够的信息来证明上一节中的定理 `self_sub`：

```py
theorem  self_sub  (a  :  R)  :  a  -  a  =  0  :=  by
  sorry 
```

展示你可以使用 `rw` 来证明这一点，但如果将任意的环 `R` 替换为实数，你也可以使用 `apply` 或 `exact` 来证明它。

Lean 知道 `1 + 1 = 2` 在任何环中都成立。通过一点努力，你可以用这个来证明上一节中的定理 `two_mul`：

```py
theorem  one_add_one_eq_two  :  1  +  1  =  (2  :  R)  :=  by
  norm_num

theorem  two_mul  (a  :  R)  :  2  *  a  =  a  +  a  :=  by
  sorry 
```

我们通过指出，我们上面建立的一些关于加法和否定的事实不需要环公理的全部力量，甚至不需要加法的交换性。一个更弱的概念的*群*可以如下公理化：

```py
variable  (A  :  Type*)  [AddGroup  A]

#check  (add_assoc  :  ∀  a  b  c  :  A,  a  +  b  +  c  =  a  +  (b  +  c))
#check  (zero_add  :  ∀  a  :  A,  0  +  a  =  a)
#check  (neg_add_cancel  :  ∀  a  :  A,  -a  +  a  =  0) 
```

当群运算交换时，使用加法符号是惯例，否则使用乘法符号。因此，Lean 定义了乘法版本以及加法版本（以及它们的阿贝尔变体，`AddCommGroup` 和 `CommGroup`）。

```py
variable  {G  :  Type*}  [Group  G]

#check  (mul_assoc  :  ∀  a  b  c  :  G,  a  *  b  *  c  =  a  *  (b  *  c))
#check  (one_mul  :  ∀  a  :  G,  1  *  a  =  a)
#check  (inv_mul_cancel  :  ∀  a  :  G,  a⁻¹  *  a  =  1) 
```

如果你感到自信，尝试只使用这些公理证明以下关于群的事实。在过程中，你需要证明许多辅助引理。本节中我们进行的证明提供了一些提示。

```py
theorem  mul_inv_cancel  (a  :  G)  :  a  *  a⁻¹  =  1  :=  by
  sorry

theorem  mul_one  (a  :  G)  :  a  *  1  =  a  :=  by
  sorry

theorem  mul_inv_rev  (a  b  :  G)  :  (a  *  b)⁻¹  =  b⁻¹  *  a⁻¹  :=  by
  sorry 
```

明确调用这些引理是繁琐的，所以 Mathlib 提供了类似于环的策略，以覆盖大多数用法：group 用于非交换乘法群，abel 用于阿贝尔加法群，noncomm_ring 用于非交换环。看起来很奇怪，代数结构被称为 Ring 和 CommRing，而策略被命名为 noncomm_ring 和 ring。这部分是历史原因，但也是为了方便使用更短的名称来处理交换环的策略，因为它使用得更频繁。

重写对于证明方程很有用，但其他类型的定理呢？例如，我们如何证明一个不等式，比如 $a + e^b \le a + e^c$ 在 $b \le c$ 时成立？我们已经看到定理可以应用于论点和假设，并且可以使用 `apply` 和 `exact` 策略来解决目标。在本节中，我们将充分利用这些工具。

考虑库定理 `le_refl` 和 `le_trans`：

```py
#check  (le_refl  :  ∀  a  :  ℝ,  a  ≤  a)
#check  (le_trans  :  a  ≤  b  →  b  ≤  c  →  a  ≤  c) 
```

如我们在第 3.1 节中更详细地解释的那样，`le_trans` 表述中的隐含括号是向右结合的，因此它应该被解释为 `a ≤ b → (b ≤ c → a ≤ c)`。库设计者已经将 `a`、`b` 和 `c` 设置为 `le_trans` 的隐含参数，这样 Lean 就不会让你明确地提供它们（除非你真的坚持，我们稍后会讨论）。相反，它期望从它们被使用的上下文中推断它们。例如，当假设 `h : a ≤ b` 和 `h' : b ≤ c` 在上下文中时，以下所有内容都有效：

```py
variable  (h  :  a  ≤  b)  (h'  :  b  ≤  c)

#check  (le_refl  :  ∀  a  :  Real,  a  ≤  a)
#check  (le_refl  a  :  a  ≤  a)
#check  (le_trans  :  a  ≤  b  →  b  ≤  c  →  a  ≤  c)
#check  (le_trans  h  :  b  ≤  c  →  a  ≤  c)
#check  (le_trans  h  h'  :  a  ≤  c) 
```

`apply` 策略接受一个一般陈述或蕴涵的证明，尝试将结论与当前目标匹配，并将假设（如果有的话）作为新目标留下。如果给定的证明与目标完全匹配（模定义等价），则可以使用 `exact` 策略而不是 `apply`。所以，所有这些都有效：

```py
example  (x  y  z  :  ℝ)  (h₀  :  x  ≤  y)  (h₁  :  y  ≤  z)  :  x  ≤  z  :=  by
  apply  le_trans
  ·  apply  h₀
  ·  apply  h₁

example  (x  y  z  :  ℝ)  (h₀  :  x  ≤  y)  (h₁  :  y  ≤  z)  :  x  ≤  z  :=  by
  apply  le_trans  h₀
  apply  h₁

example  (x  y  z  :  ℝ)  (h₀  :  x  ≤  y)  (h₁  :  y  ≤  z)  :  x  ≤  z  :=
  le_trans  h₀  h₁

example  (x  :  ℝ)  :  x  ≤  x  :=  by
  apply  le_refl

example  (x  :  ℝ)  :  x  ≤  x  :=
  le_refl  x 
```

在第一个示例中，应用 `le_trans` 产生了两个目标，我们使用点来指示每个证明的开始位置。点是可以省略的，但它们有助于 *聚焦* 目标：在点引入的块内，只有一个目标可见，并且必须在块的末尾之前完成。在这里，我们通过用另一个点开始一个新的块来结束第一个块。我们也可以减少缩进。在第三个示例和最后一个示例中，我们完全避免了进入策略模式：`le_trans h₀ h₁` 和 `le_refl x` 是我们需要证明项。

这里还有一些库定理：

```py
#check  (le_refl  :  ∀  a,  a  ≤  a)
#check  (le_trans  :  a  ≤  b  →  b  ≤  c  →  a  ≤  c)
#check  (lt_of_le_of_lt  :  a  ≤  b  →  b  <  c  →  a  <  c)
#check  (lt_of_lt_of_le  :  a  <  b  →  b  ≤  c  →  a  <  c)
#check  (lt_trans  :  a  <  b  →  b  <  c  →  a  <  c) 
```

将它们与 `apply` 和 `exact` 结合起来证明以下内容：

```py
example  (h₀  :  a  ≤  b)  (h₁  :  b  <  c)  (h₂  :  c  ≤  d)  (h₃  :  d  <  e)  :  a  <  e  :=  by
  sorry 
```

实际上，Lean 有一个策略可以自动完成这类事情：

```py
example  (h₀  :  a  ≤  b)  (h₁  :  b  <  c)  (h₂  :  c  ≤  d)  (h₃  :  d  <  e)  :  a  <  e  :=  by
  linarith 
```

`linarith` 策略被设计用来处理 *线性算术*。

```py
example  (h  :  2  *  a  ≤  3  *  b)  (h'  :  1  ≤  a)  (h''  :  d  =  2)  :  d  +  a  ≤  5  *  b  :=  by
  linarith 
```

除了上下文中的等式和不等式之外，`linarith` 还会使用你作为参数传递的额外不等式。在下一个示例中，`exp_le_exp.mpr h'` 是 `exp b ≤ exp c` 的证明，我们将在下一刻解释。请注意，在 Lean 中，我们用 `f x` 来表示函数 `f` 对参数 `x` 的应用，这与我们用 `h x` 来表示事实或定理 `h` 对参数 `x` 的应用的方式完全相同。括号仅用于复合参数，例如 `f (x + y)`。如果没有括号，`f x + y` 将被解析为 `(f x) + y`。

```py
example  (h  :  1  ≤  a)  (h'  :  b  ≤  c)  :  2  +  a  +  exp  b  ≤  3  *  a  +  exp  c  :=  by
  linarith  [exp_le_exp.mpr  h'] 
```

这里有一些库中的定理，可以用来在实数上建立不等式。

```py
#check  (exp_le_exp  :  exp  a  ≤  exp  b  ↔  a  ≤  b)
#check  (exp_lt_exp  :  exp  a  <  exp  b  ↔  a  <  b)
#check  (log_le_log  :  0  <  a  →  a  ≤  b  →  log  a  ≤  log  b)
#check  (log_lt_log  :  0  <  a  →  a  <  b  →  log  a  <  log  b)
#check  (add_le_add  :  a  ≤  b  →  c  ≤  d  →  a  +  c  ≤  b  +  d)
#check  (add_le_add_left  :  a  ≤  b  →  ∀  c,  c  +  a  ≤  c  +  b)
#check  (add_le_add_right  :  a  ≤  b  →  ∀  c,  a  +  c  ≤  b  +  c)
#check  (add_lt_add_of_le_of_lt  :  a  ≤  b  →  c  <  d  →  a  +  c  <  b  +  d)
#check  (add_lt_add_of_lt_of_le  :  a  <  b  →  c  ≤  d  →  a  +  c  <  b  +  d)
#check  (add_lt_add_left  :  a  <  b  →  ∀  c,  c  +  a  <  c  +  b)
#check  (add_lt_add_right  :  a  <  b  →  ∀  c,  a  +  c  <  b  +  c)
#check  (add_nonneg  :  0  ≤  a  →  0  ≤  b  →  0  ≤  a  +  b)
#check  (add_pos  :  0  <  a  →  0  <  b  →  0  <  a  +  b)
#check  (add_pos_of_pos_of_nonneg  :  0  <  a  →  0  ≤  b  →  0  <  a  +  b)
#check  (exp_pos  :  ∀  a,  0  <  exp  a)
#check  add_le_add_left 
```

一些定理，如 `exp_le_exp` 和 `exp_lt_exp`，使用 *双向蕴涵*，它表示短语“当且仅当”。（你可以在 VS Code 中使用 `\lr` 或 `\iff` 来输入它）。我们将在下一章中更详细地讨论这个连接词。这样的定理可以用 `rw` 来重写一个目标为等价的目标：

```py
example  (h  :  a  ≤  b)  :  exp  a  ≤  exp  b  :=  by
  rw  [exp_le_exp]
  exact  h 
```

在本节中，然而，我们将使用以下事实：如果 `h : A ↔ B` 是这样一个等价关系，那么 `h.mp` 建立了正向方向，`A → B`，而 `h.mpr` 建立了反向方向，`B → A`。在这里，`mp` 代表“modus ponens”，而 `mpr` 代表“modus ponens reverse”。如果你更喜欢，你也可以使用 `h.1` 和 `h.2` 分别代表 `h.mp` 和 `h.mpr`。因此，以下证明是有效的：

```py
example  (h₀  :  a  ≤  b)  (h₁  :  c  <  d)  :  a  +  exp  c  +  e  <  b  +  exp  d  +  e  :=  by
  apply  add_lt_add_of_lt_of_le
  ·  apply  add_lt_add_of_le_of_lt  h₀
  apply  exp_lt_exp.mpr  h₁
  apply  le_refl 
```

第一行，`apply add_lt_add_of_lt_of_le` 产生了两个目标，并且我们再次使用一个点来区分第一个证明和第二个证明。

尝试以下示例。中间的示例显示你可以使用 `norm_num` 策略来解决具体的数值目标。

```py
example  (h₀  :  d  ≤  e)  :  c  +  exp  (a  +  d)  ≤  c  +  exp  (a  +  e)  :=  by  sorry

example  :  (0  :  ℝ)  <  1  :=  by  norm_num

example  (h  :  a  ≤  b)  :  log  (1  +  exp  a)  ≤  log  (1  +  exp  b)  :=  by
  have  h₀  :  0  <  1  +  exp  a  :=  by  sorry
  apply  log_le_log  h₀
  sorry 
```

从这些示例中，应该很明显，能够找到你需要的库定理是形式化的重要部分。你可以使用以下几种策略：

+   你可以在其 [GitHub 仓库](https://github.com/leanprover-community/mathlib4) 中浏览 Mathlib。

+   你可以使用 Mathlib 的 [网页](https://leanprover-community.github.io/mathlib4_docs/) 上的 API 文档。

+   你可以使用 Loogle <https://loogle.lean-lang.org>通过模式搜索 Lean 和 Mathlib 的定义和定理。

+   你可以依靠 Mathlib 的命名约定和编辑器的 Ctrl-space 自动完成来猜测定理名称（或在 Mac 键盘上按 Cmd-space）。在 Lean 中，一个名为`A_of_B_of_C`的定理从形式为`B`和`C`的假设中建立`A`，其中`A`、`B`和`C`近似于我们大声读出目标的方式。所以一个建立类似`x + y ≤ ...`的定理可能以`add_le`开头。键入`add_le`并按 Ctrl-space 将给出一些有用的选择。请注意，按 Ctrl-space 两次将显示有关可用完成信息的更多信息。

+   如果你右键单击 VS Code 中的现有定理名称，编辑器将显示一个菜单，其中包含跳转到定理定义文件的选项，你可以在附近找到类似定理。

+   你可以使用`apply?`策略，它试图在库中找到相关的定理。

```py
example  :  0  ≤  a  ^  2  :=  by
  -- apply?
  exact  sq_nonneg  a 
```

在这个例子中尝试`apply?`，请删除`exact`命令并取消上一行的注释。使用这些技巧，看看你是否能找到进行下一个例子所需的内容：

```py
example  (h  :  a  ≤  b)  :  c  -  exp  b  ≤  c  -  exp  a  :=  by
  sorry 
```

使用相同的技巧，确认`linarith`而不是`apply?`也可以完成这项工作。

这里还有一个不等式的例子：

```py
example  :  2*a*b  ≤  a²  +  b²  :=  by
  have  h  :  0  ≤  a²  -  2*a*b  +  b²
  calc
  a²  -  2*a*b  +  b²  =  (a  -  b)²  :=  by  ring
  _  ≥  0  :=  by  apply  pow_two_nonneg

  calc
  2*a*b  =  2*a*b  +  0  :=  by  ring
  _  ≤  2*a*b  +  (a²  -  2*a*b  +  b²)  :=  add_le_add  (le_refl  _)  h
  _  =  a²  +  b²  :=  by  ring 
```

Mathlib 倾向于在二元运算如`*`和`^`周围放置空格，但在这个例子中，更紧凑的格式提高了可读性。有一些值得注意的事情。首先，表达式`s ≥ t`在定义上是等价于`t ≤ s`的。原则上，这意味着人们应该能够互换使用它们。但是，Lean 的一些自动化没有识别出这种等价性，所以 Mathlib 倾向于更喜欢`≤`而不是`≥`。其次，我们广泛使用了`ring`策略。它真的节省了时间！最后，注意在第二个`calc`证明的第二行中，我们不需要写`by exact add_le_add (le_refl _) h`，我们可以简单地写出证明项`add_le_add (le_refl _) h`。

实际上，上述证明中唯一的巧妙之处在于找出假设`h`。一旦我们有了它，第二个计算只涉及线性代数，而`linarith`可以处理它：

```py
example  :  2*a*b  ≤  a²  +  b²  :=  by
  have  h  :  0  ≤  a²  -  2*a*b  +  b²
  calc
  a²  -  2*a*b  +  b²  =  (a  -  b)²  :=  by  ring
  _  ≥  0  :=  by  apply  pow_two_nonneg
  linarith 
```

真是太好了！我们挑战你使用这些想法来证明以下定理。你可以使用定理`abs_le'.mpr`。你还需要使用`constructor`策略将合取分解为两个目标；参见第 3.4 节。

```py
example  :  |a*b|  ≤  (a²  +  b²)/2  :=  by
  sorry

#check  abs_le'.mpr 
```

如果你设法解决了这个问题，恭喜你！你正朝着成为一位大师级形式化专家的道路上迈进。

实数上的`min`函数由以下三个事实唯一表征：

```py
#check  (min_le_left  a  b  :  min  a  b  ≤  a)
#check  (min_le_right  a  b  :  min  a  b  ≤  b)
#check  (le_min  :  c  ≤  a  →  c  ≤  b  →  c  ≤  min  a  b) 
```

你能猜出表征`max`的定理的名称吗？

注意，我们必须通过编写 `min a b` 而不是 `min (a, b)` 来将 `min` 应用到一对参数 `a` 和 `b` 上。形式上，`min` 是一个类型为 `ℝ → ℝ → ℝ` 的函数。当我们用多个箭头写这样的类型时，约定是隐式括号向右结合，所以类型被解释为 `ℝ → (ℝ → ℝ)`。最终的效果是，如果 `a` 和 `b` 的类型是 `ℝ`，那么 `min a` 的类型是 `ℝ → ℝ`，`min a b` 的类型是 `ℝ`，所以 `min` 的行为就像一个有两个参数的函数，正如我们所期望的那样。以这种方式处理多个参数被称为 *currying*，这是逻辑学家 Haskell Curry 的名字。

Lean 中的运算顺序也需要一些时间来适应。函数应用比中缀操作绑定得更紧密，所以表达式 `min a b + c` 被解释为 `(min a b) + c`。随着时间的推移，这些约定将变得自然而然。

使用定理 `le_antisymm`，我们可以证明两个实数相等当且仅当每个数都小于或等于另一个数。利用这一点和上述事实，我们可以证明 `min` 是可交换的：

```py
example  :  min  a  b  =  min  b  a  :=  by
  apply  le_antisymm
  ·  show  min  a  b  ≤  min  b  a
  apply  le_min
  ·  apply  min_le_right
  apply  min_le_left
  ·  show  min  b  a  ≤  min  a  b
  apply  le_min
  ·  apply  min_le_right
  apply  min_le_left 
```

在这里，我们使用了点来分隔不同目标证明。我们的用法不一致：在外层，我们使用点和缩进来表示两个目标，而对于嵌套证明，我们只使用点，直到只剩下一个目标。这两种约定都是合理且有用的。我们还使用 `show` 策略来结构化证明并指示每个块中正在证明的内容。即使没有 `show` 命令，证明仍然有效，但使用它们可以使证明更容易阅读和维护。

可能会让你感到烦恼的是，证明是重复的。为了预示你以后将学习到的技能，我们注意到避免重复的一种方法是可以声明一个局部引理然后使用它：

```py
example  :  min  a  b  =  min  b  a  :=  by
  have  h  :  ∀  x  y  :  ℝ,  min  x  y  ≤  min  y  x  :=  by
  intro  x  y
  apply  le_min
  apply  min_le_right
  apply  min_le_left
  apply  le_antisymm
  apply  h
  apply  h 
```

我们将在 第 3.1 节 中详细介绍全称量词，但在这里只需说明，假设 `h` 表示对于任何 `x` 和 `y`，所期望的不等式都成立，而 `intro` 策略引入任意的 `x` 和 `y` 来建立结论。在 `le_antisymm` 之后的第一 `apply` 隐式地使用了 `h a b`，而第二个 `apply` 使用了 `h b a`。

另一种解决方案是使用 `repeat` 策略，它可以尽可能多次地应用策略（或一个块）。

```py
example  :  min  a  b  =  min  b  a  :=  by
  apply  le_antisymm
  repeat
  apply  le_min
  apply  min_le_right
  apply  min_le_left 
```

我们鼓励你将以下内容作为练习来证明。你可以使用上面描述的任何一种技巧来缩短第一个证明。

```py
example  :  max  a  b  =  max  b  a  :=  by
  sorry
example  :  min  (min  a  b)  c  =  min  a  (min  b  c)  :=  by
  sorry 
```

当然，你也可以证明 `max` 的结合律。

这是一个有趣的事实，即`min`在`max`上的分配方式与乘法在加法上的分配方式相同，反之亦然。换句话说，在实数上，我们有恒等式`min a (max b c) = max (min a b) (min a c)`，以及将`max`和`min`互换的对应版本。但在下一节中，我们将看到这一点并不从`≤`的可传递性和自反性以及上面列出的`min`和`max`的特征属性中得出。我们需要使用`≤`在实数上是全序的事实，也就是说，它满足`∀ x y, x ≤ y ∨ y ≤ x`。在这里，析取符号`∨`代表“或”。在第一种情况下，我们有`min x y = x`，在第二种情况下，我们有`min x y = y`。我们将学习如何在第 3.5 节中通过情况推理，但就目前而言，我们将坚持使用不需要情况分解的例子。

这里有一个这样的例子：

```py
theorem  aux  :  min  a  b  +  c  ≤  min  (a  +  c)  (b  +  c)  :=  by
  sorry
example  :  min  a  b  +  c  =  min  (a  +  c)  (b  +  c)  :=  by
  sorry 
```

很明显，`aux`提供了证明等式所需的两个不等式之一，但将其应用于适当的值也会得到另一个方向。作为一个提示，你可以使用定理`add_neg_cancel_right`和`linarith`策略。

Lean 的命名约定在库的名称中得到了体现，即三角不等式的名称：

```py
#check  (abs_add  :  ∀  a  b  :  ℝ,  |a  +  b|  ≤  |a|  +  |b|) 
```

使用它来证明以下变体，同时使用`add_sub_cancel_right`：

```py
example  :  |a|  -  |b|  ≤  |a  -  b|  :=
  sorry
end 
```

看看你是否能在三行或更少的文字中做到这一点。你可以使用定理`sub_add_cancel`。

在接下来的章节中，我们将使用的一个重要关系是自然数上的除法关系 `x ∣ y`。请注意：除法符号 *不是* 你键盘上的普通竖线。相反，它是一个通过在 VS Code 中输入 `\|` 获得的 Unicode 字符。按照惯例，Mathlib 在定理名称中使用 `dvd` 来指代它。

```py
example  (h₀  :  x  ∣  y)  (h₁  :  y  ∣  z)  :  x  ∣  z  :=
  dvd_trans  h₀  h₁

example  :  x  ∣  y  *  x  *  z  :=  by
  apply  dvd_mul_of_dvd_left
  apply  dvd_mul_left

example  :  x  ∣  x  ^  2  :=  by
  apply  dvd_mul_left 
```

在最后一个例子中，指数是一个自然数，应用`dvd_mul_left`迫使 Lean 展开`x²`的定义为`x¹ * x`。看看你是否能猜出你需要证明以下内容所需的定理名称：

```py
example  (h  :  x  ∣  w)  :  x  ∣  y  *  (x  *  z)  +  x  ^  2  +  w  ^  2  :=  by
  sorry
end 
```

在除法方面，*最大公约数* `gcd` 和最小公倍数 `lcm` 与 `min` 和 `max` 类似。由于每个数都能整除 `0`，所以 `0` 真的是除法关系中的最大元素：

```py
variable  (m  n  :  ℕ)

#check  (Nat.gcd_zero_right  n  :  Nat.gcd  n  0  =  n)
#check  (Nat.gcd_zero_left  n  :  Nat.gcd  0  n  =  n)
#check  (Nat.lcm_zero_right  n  :  Nat.lcm  n  0  =  0)
#check  (Nat.lcm_zero_left  n  :  Nat.lcm  0  n  =  0) 
```

看看你是否能猜出你需要证明以下内容所需的定理名称：

```py
example  :  Nat.gcd  m  n  =  Nat.gcd  n  m  :=  by
  sorry 
```

提示：你可以使用`dvd_antisymm`，但如果你这样做，Lean 会抱怨表达式在通用定理和版本`Nat.dvd_antisymm`（专门针对自然数的版本）之间是模糊的。你可以使用`_root_.dvd_antisymm`来指定通用版本；任何一个都可以工作。

在第 2.2 节中，我们看到了许多常见的关于实数的恒等式在更一般的代数结构中也是成立的，例如交换环。我们可以使用任何我们想要的公理来描述代数结构，而不仅仅是方程。例如，**部分序**由一个集合和二元关系组成，该关系是自反的、传递的和反对称的，类似于实数上的 `≤`。Lean 了解部分序：

```py
variable  {α  :  Type*}  [PartialOrder  α]
variable  (x  y  z  :  α)

#check  x  ≤  y
#check  (le_refl  x  :  x  ≤  x)
#check  (le_trans  :  x  ≤  y  →  y  ≤  z  →  x  ≤  z)
#check  (le_antisymm  :  x  ≤  y  →  y  ≤  x  →  x  =  y) 
```

在这里，我们采用了 Mathlib 的约定，使用像 `α`、`β` 和 `γ`（输入为 `\a`、`\b` 和 `\g`）这样的字母来表示任意类型。该库通常使用像 `R` 和 `G` 这样的字母来表示环和群等代数结构的载体，但在一般情况下，希腊字母用于类型，特别是在与它们关联的结构很少或没有结构时。 

与任何部分序 `≤` 相关联的，还有一个**严格部分序**`<`，它在某种程度上类似于实数上的 `<`。在这个顺序中，说 `x` 小于 `y` 等同于说它小于或等于 `y` 但不等于 `y`。

```py
#check  x  <  y
#check  (lt_irrefl  x  :  ¬  (x  <  x))
#check  (lt_trans  :  x  <  y  →  y  <  z  →  x  <  z)
#check  (lt_of_le_of_lt  :  x  ≤  y  →  y  <  z  →  x  <  z)
#check  (lt_of_lt_of_le  :  x  <  y  →  y  ≤  z  →  x  <  z)

example  :  x  <  y  ↔  x  ≤  y  ∧  x  ≠  y  :=
  lt_iff_le_and_ne 
```

在这个例子中，符号 `∧` 表示“和”，符号 `¬` 表示“非”，而 `x ≠ y` 简写为 `¬ (x = y)`。在第三章中，您将学习如何使用这些逻辑连接词来**证明** `<` 具有指示的性质。

**格**是一种扩展了部分序（`≤`）并带有 `⊓` 和 `⊔` 操作的结构，这些操作与实数上的 `min` 和 `max` 类似：

```py
variable  {α  :  Type*}  [Lattice  α]
variable  (x  y  z  :  α)

#check  x  ⊓  y
#check  (inf_le_left  :  x  ⊓  y  ≤  x)
#check  (inf_le_right  :  x  ⊓  y  ≤  y)
#check  (le_inf  :  z  ≤  x  →  z  ≤  y  →  z  ≤  x  ⊓  y)
#check  x  ⊔  y
#check  (le_sup_left  :  x  ≤  x  ⊔  y)
#check  (le_sup_right  :  y  ≤  x  ⊔  y)
#check  (sup_le  :  x  ≤  z  →  y  ≤  z  →  x  ⊔  y  ≤  z) 
```

`⊓` 和 `⊔` 的定义使得它们分别被称为**最大下界**和**最小上界**。您可以在 VS code 中使用 `\glb` 和 `\lub` 来输入这些符号。这些符号也常被称为**下确界**和**上确界**，Mathlib 在定理名称中用 `inf` 和 `sup` 来指代它们。为了进一步复杂化问题，它们也常被称为**交**和**并**。因此，如果您与格（lattices）打交道，您必须记住以下字典：

+   `⊓` 是**最大下界**、**下确界**或**交**。

+   `⊔` 是**最小上界**、**上确界**或**并**。

一些格的实例包括：

+   在任何全序（如整数或实数与 `≤`）上的 `min` 和 `max`。

+   在某个域的子集集合上，使用 `⊆` 排序，`∩` 和 `∪` 表示。

+   在布尔真值上的 `∧` 和 `∨`，如果 `x` 为假或 `y` 为真，则 `x ≤ y`。

+   在自然数（或正自然数）上的 `gcd` 和 `lcm`，使用可除性排序 `∣`。

+   向量空间线性子空间的集合，其中最大下界由交集给出，最小上界由两个空间的和给出，排序是包含关系。

+   集合上的拓扑集合（或在 Lean 中，类型），其中两个拓扑的最大下界是由它们的并生成的拓扑，最小上界是它们的交集，排序是反向包含关系。

你可以检查，与`min`/`max`和`gcd`/`lcm`一样，你可以仅使用它们的特征公理以及`le_refl`和`le_trans`来证明下确界和上确界的交换性和结合性。

当看到目标`x ≤ z`时使用`apply le_trans`不是一个好主意。实际上，Lean 没有方法猜测我们想要使用哪个中间元素`y`。因此，`apply le_trans`会产生三个看起来像`x ≤ ?a`、`?a ≤ z`和`α`的目标，其中`?a`（可能有一个更复杂的自动生成的名称）代表神秘的`y`。最后一个目标，类型为`α`，是提供`y`的值。它最后出现，因为 Lean 希望从第一个目标`x ≤ ?a`的证明中自动推断它。为了避免这种不吸引人的情况，你可以使用`calc`策略显式地提供`y`。或者，你可以使用接受`y`作为参数的`trans`策略，它会产生预期的目标`x ≤ y`和`y ≤ z`。当然，你也可以通过直接提供一个完整的证明来避免这个问题，例如`exact le_trans inf_le_left inf_le_right`，但这需要更多的计划。

```py
example  :  x  ⊓  y  =  y  ⊓  x  :=  by
  sorry

example  :  x  ⊓  y  ⊓  z  =  x  ⊓  (y  ⊓  z)  :=  by
  sorry

example  :  x  ⊔  y  =  y  ⊔  x  :=  by
  sorry

example  :  x  ⊔  y  ⊔  z  =  x  ⊔  (y  ⊔  z)  :=  by
  sorry 
```

你可以在 Mathlib 中找到这些定理，分别命名为`inf_comm`、`inf_assoc`、`sup_comm`和`sup_assoc`。

另一个很好的练习是仅使用那些公理来证明**吸收律**：

```py
theorem  absorb1  :  x  ⊓  (x  ⊔  y)  =  x  :=  by
  sorry

theorem  absorb2  :  x  ⊔  x  ⊓  y  =  x  :=  by
  sorry 
```

这些可以在 Mathlib 中找到，名称分别为`inf_sup_self`和`sup_inf_self`。

满足额外的恒等式`x ⊓ (y ⊔ z) = (x ⊓ y) ⊔ (x ⊓ z)`和`x ⊔ (y ⊓ z) = (x ⊔ y) ⊓ (x ⊔ z)`的格称为**分配格**。Lean 也了解这些：

```py
variable  {α  :  Type*}  [DistribLattice  α]
variable  (x  y  z  :  α)

#check  (inf_sup_left  x  y  z  :  x  ⊓  (y  ⊔  z)  =  x  ⊓  y  ⊔  x  ⊓  z)
#check  (inf_sup_right  x  y  z  :  (x  ⊔  y)  ⊓  z  =  x  ⊓  z  ⊔  y  ⊓  z)
#check  (sup_inf_left  x  y  z  :  x  ⊔  y  ⊓  z  =  (x  ⊔  y)  ⊓  (x  ⊔  z))
#check  (sup_inf_right  x  y  z  :  x  ⊓  y  ⊔  z  =  (x  ⊔  z)  ⊓  (y  ⊔  z)) 
```

左侧和右侧版本很容易证明是等价的，考虑到`⊓`和`⊔`的交换性。证明不是每个格都是分配格是一个很好的练习，可以通过提供一个具有有限个元素的非分配格的显式描述来完成。证明在任何格中，分配律中的任何一个都蕴含另一个也是一个很好的练习：

```py
variable  {α  :  Type*}  [Lattice  α]
variable  (a  b  c  :  α)

example  (h  :  ∀  x  y  z  :  α,  x  ⊓  (y  ⊔  z)  =  x  ⊓  y  ⊔  x  ⊓  z)  :  a  ⊔  b  ⊓  c  =  (a  ⊔  b)  ⊓  (a  ⊔  c)  :=  by
  sorry

example  (h  :  ∀  x  y  z  :  α,  x  ⊔  y  ⊓  z  =  (x  ⊔  y)  ⊓  (x  ⊔  z))  :  a  ⊓  (b  ⊔  c)  =  a  ⊓  b  ⊔  a  ⊓  c  :=  by
  sorry 
```

有可能将公理结构组合成更大的结构。例如，一个**严格有序环**由一个环及其载体上的偏序组成，该偏序满足额外的公理，这些公理说明环运算与顺序是兼容的：

```py
variable  {R  :  Type*}  [Ring  R]  [PartialOrder  R]  [IsStrictOrderedRing  R]
variable  (a  b  c  :  R)

#check  (add_le_add_left  :  a  ≤  b  →  ∀  c,  c  +  a  ≤  c  +  b)
#check  (mul_pos  :  0  <  a  →  0  <  b  →  0  <  a  *  b) 
```

第三章将提供从`mul_pos`和`<`的定义中推导出以下内容的方法：

```py
#check  (mul_nonneg  :  0  ≤  a  →  0  ≤  b  →  0  ≤  a  *  b) 
```

然后是一个扩展练习，证明许多用于推理算术和实数排序的常见事实对于任何有序环都是通用的。这里有一些你可以尝试的例子，仅使用环的性质、偏序以及最后两个例子中列出的事实（注意，这些环不一定假设是交换的，因此没有提供环策略）：

```py
example  (h  :  a  ≤  b)  :  0  ≤  b  -  a  :=  by
  sorry

example  (h:  0  ≤  b  -  a)  :  a  ≤  b  :=  by
  sorry

example  (h  :  a  ≤  b)  (h'  :  0  ≤  c)  :  a  *  c  ≤  b  *  c  :=  by
  sorry 
```

最后，这里有一个最后的例子。一个**度量空间**由一个集合组成，该集合配备了距离的概念，`dist x y`，将任何一对元素映射到一个实数。距离函数假设满足以下公理：

```py
variable  {X  :  Type*}  [MetricSpace  X]
variable  (x  y  z  :  X)

#check  (dist_self  x  :  dist  x  x  =  0)
#check  (dist_comm  x  y  :  dist  x  y  =  dist  y  x)
#check  (dist_triangle  x  y  z  :  dist  x  z  ≤  dist  x  y  +  dist  y  z) 
```

掌握了这个部分后，您可以证明距离总是非负的，这是由这些公理得出的：

```py
example  (x  y  :  X)  :  0  ≤  dist  x  y  :=  by
  sorry 
```

我们建议使用定理 `nonneg_of_mul_nonneg_left`。正如您可能已经猜到的，这个定理在 Mathlib 中被称为 `dist_nonneg`。

## 2.1\. 计算

我们通常学习进行数学计算，而不把它们当作证明。但当我们像 Lean 所要求的那样证明计算的每一步，最终结果是证明计算的左侧等于右侧。

在 Lean 中，陈述一个定理相当于陈述一个目标，即证明该定理的目标。Lean 提供了重写策略 `rw`，在目标中将等式的左侧替换为右侧。如果 `a`、`b` 和 `c` 是实数，`mul_assoc a b c` 是等式 `a * b * c = a * (b * c)`，而 `mul_comm a b` 是等式 `a * b = b * a`。Lean 提供了自动化，通常可以消除明确引用此类事实的需要，但它们对于说明目的很有用。在 Lean 中，乘法从左结合，因此 `mul_assoc` 的左侧也可以写成 `(a * b) * c`。然而，通常良好的风格是注意 Lean 的符号约定，并在 Lean 也这样做时省略括号。

让我们试试 `rw`。

```py
example  (a  b  c  :  ℝ)  :  a  *  b  *  c  =  b  *  (a  *  c)  :=  by
  rw  [mul_comm  a  b]
  rw  [mul_assoc  b  a  c] 
```

在相关示例文件的开始部分，`import` 行导入了 Mathlib 中的实数理论以及有用的自动化。为了简洁起见，我们在教科书中通常抑制此类信息。

您可以随意修改以查看会发生什么。在 VS Code 中，您可以将 `ℝ` 字符键入为 `\R` 或 `\real`。符号只有在您按下空格键或制表键后才会显示。当您在阅读 Lean 文件时悬停在符号上，VS Code 将显示可以用来输入它的语法。如果您想查看所有可用的缩写，您可以按 Ctrl-Shift-P，然后输入缩写以访问 `Lean 4: 显示 Unicode 输入缩写` 命令。如果您的键盘上没有容易访问的反斜杠，您可以通过更改 `lean4.input.leader` 设置来更改前导字符。

当光标位于策略证明的中间时，Lean 在 *Lean Infoview* 窗口中报告当前的 *证明状态*。当您将光标移过证明的每一步时，您可以看到状态的变化。Lean 中的一个典型证明状态可能如下所示：

```py
1  goal
x  y  :  ℕ,
h₁  :  Prime  x,
h₂  :  ¬Even  x,
h₃  :  y  >  x
⊢  y  ≥  4 
```

以`⊢`开头的行表示*上下文*：它们是当前正在使用的对象和假设。在这个例子中，这些包括两个对象，`x`和`y`，每个都是自然数。它们还包括三个假设，分别标记为`h₁`、`h₂`和`h₃`。在 Lean 中，上下文中的所有内容都用标识符标记。您可以将这些下标标签键入为`h\1`、`h\2`和`h\3`，但任何合法的标识符都可以：您可以使用`h1`、`h2`、`h3`代替，或者使用`foo`、`bar`和`baz`。最后一行代表*目标*，即要证明的事实。有时人们用*目标*来表示要证明的事实，用*目标*来表示上下文和目标的组合。在实践中，通常可以清楚地理解意图。

尝试证明这些恒等式，在每种情况下用策略证明替换`sorry`。使用`rw`策略，您可以使用左箭头（`\l`）来反转一个恒等式。例如，`rw [← mul_assoc a b c]`将当前目标中的`a * (b * c)`替换为`a * b * c`。请注意，指向左边的箭头指的是从右到左在`mul_assoc`提供的恒等式中移动，它与目标左侧或右侧无关。

```py
example  (a  b  c  :  ℝ)  :  c  *  b  *  a  =  b  *  (a  *  c)  :=  by
  sorry

example  (a  b  c  :  ℝ)  :  a  *  (b  *  c)  =  b  *  (a  *  c)  :=  by
  sorry 
```

您也可以在不提供参数的情况下使用像`mul_assoc`和`mul_comm`这样的恒等式。在这种情况下，重写策略会尝试使用它找到的第一个模式将目标左侧与目标中的表达式匹配。

```py
example  (a  b  c  :  ℝ)  :  a  *  b  *  c  =  b  *  c  *  a  :=  by
  rw  [mul_assoc]
  rw  [mul_comm] 
```

您也可以提供*部分*信息。例如，`mul_comm a`与形式为`a * ?`的任何模式匹配，并将其重写为`? * a`。尝试在不提供任何参数的情况下执行这些示例的第一个，以及只提供一个参数的情况下执行第二个。

```py
example  (a  b  c  :  ℝ)  :  a  *  (b  *  c)  =  b  *  (c  *  a)  :=  by
  sorry

example  (a  b  c  :  ℝ)  :  a  *  (b  *  c)  =  b  *  (a  *  c)  :=  by
  sorry 
```

您也可以使用`rw`与局部上下文中的事实。

```py
example  (a  b  c  d  e  f  :  ℝ)  (h  :  a  *  b  =  c  *  d)  (h'  :  e  =  f)  :  a  *  (b  *  e)  =  c  *  (d  *  f)  :=  by
  rw  [h']
  rw  [←  mul_assoc]
  rw  [h]
  rw  [mul_assoc] 
```

尝试这些，对于第二个使用定理`sub_self`：

```py
example  (a  b  c  d  e  f  :  ℝ)  (h  :  b  *  c  =  e  *  f)  :  a  *  b  *  c  *  d  =  a  *  e  *  f  *  d  :=  by
  sorry

example  (a  b  c  d  :  ℝ)  (hyp  :  c  =  b  *  a  -  d)  (hyp'  :  d  =  a  *  b)  :  c  =  0  :=  by
  sorry 
```

可以通过在方括号内用逗号分隔相关恒等式来执行多个重写命令。

```py
example  (a  b  c  d  e  f  :  ℝ)  (h  :  a  *  b  =  c  *  d)  (h'  :  e  =  f)  :  a  *  (b  *  e)  =  c  *  (d  *  f)  :=  by
  rw  [h',  ←  mul_assoc,  h,  mul_assoc] 
```

您仍然可以通过在任何重写列表的逗号后放置光标来看到增量进度。

另一个技巧是，我们可以在示例或定理之外一次性声明变量。然后 Lean 会自动包含它们。

```py
variable  (a  b  c  d  e  f  :  ℝ)

example  (h  :  a  *  b  =  c  *  d)  (h'  :  e  =  f)  :  a  *  (b  *  e)  =  c  *  (d  *  f)  :=  by
  rw  [h',  ←  mul_assoc,  h,  mul_assoc] 
```

检查上述证明开始时的策略状态，可以发现 Lean 确实包含了所有变量。我们可以通过将其放在`section ... end`块中来限定声明的范围。最后，从引言中回忆起，Lean 为我们提供了一个命令来确定表达式的类型：

```py
section
variable  (a  b  c  :  ℝ)

#check  a
#check  a  +  b
#check  (a  :  ℝ)
#check  mul_comm  a  b
#check  (mul_comm  a  b  :  a  *  b  =  b  *  a)
#check  mul_assoc  c  a  b
#check  mul_comm  a
#check  mul_comm

end 
```

`#check` 命令适用于对象和事实。对于命令 `#check a`，Lean 报告 `a` 的类型为 `ℝ`。对于命令 `#check mul_comm a b`，Lean 报告 `mul_comm a b` 是事实 `a * b = b * a` 的证明。命令 `#check (a : ℝ)` 表明我们期望 `a` 的类型是 `ℝ`，如果这不是情况，Lean 将引发错误。我们将在稍后解释最后三个 `#check` 命令的输出，但在此同时，您可以查看它们，并尝试一些自己的 `#check` 命令。

让我们尝试一些更多的例子。定理 `two_mul a` 表示 `2 * a = a + a`。定理 `add_mul` 和 `mul_add` 表达了乘法在加法上的分配律，而定理 `add_assoc` 表达了加法的结合律。使用 `#check` 命令来查看精确的陈述。

```py
example  :  (a  +  b)  *  (a  +  b)  =  a  *  a  +  2  *  (a  *  b)  +  b  *  b  :=  by
  rw  [mul_add,  add_mul,  add_mul]
  rw  [←  add_assoc,  add_assoc  (a  *  a)]
  rw  [mul_comm  b  a,  ←  two_mul] 
```

虽然可以通过在编辑器中逐步执行这个证明来弄清楚发生了什么，但单独阅读起来很困难。Lean 提供了一种使用 `calc` 关键字来编写这种证明的更结构化的方式。

```py
example  :  (a  +  b)  *  (a  +  b)  =  a  *  a  +  2  *  (a  *  b)  +  b  *  b  :=
  calc
  (a  +  b)  *  (a  +  b)  =  a  *  a  +  b  *  a  +  (a  *  b  +  b  *  b)  :=  by
  rw  [mul_add,  add_mul,  add_mul]
  _  =  a  *  a  +  (b  *  a  +  a  *  b)  +  b  *  b  :=  by
  rw  [←  add_assoc,  add_assoc  (a  *  a)]
  _  =  a  *  a  +  2  *  (a  *  b)  +  b  *  b  :=  by
  rw  [mul_comm  b  a,  ←  two_mul] 
```

注意，证明并不以 `by` 开头：以 `calc` 开头的表达式是一个 *证明项*。`calc` 表达式也可以用在策略证明中，但 Lean 将其解释为使用生成的证明项来解决目标的指令。`calc` 语法很挑剔：下划线和证明必须采用上述格式。Lean 使用缩进来确定策略块或 `calc` 块的开始和结束位置；尝试更改上述证明中的缩进来看看会发生什么。

写一个 `calc` 证明的一种方法是首先使用 `sorry` 策略来概述它，确保 Lean 接受这些表达式，然后使用策略来证明各个步骤。

```py
example  :  (a  +  b)  *  (a  +  b)  =  a  *  a  +  2  *  (a  *  b)  +  b  *  b  :=
  calc
  (a  +  b)  *  (a  +  b)  =  a  *  a  +  b  *  a  +  (a  *  b  +  b  *  b)  :=  by
  sorry
  _  =  a  *  a  +  (b  *  a  +  a  *  b)  +  b  *  b  :=  by
  sorry
  _  =  a  *  a  +  2  *  (a  *  b)  +  b  *  b  :=  by
  sorry 
```

尝试使用纯 `rw` 证明和更结构化的 `calc` 证明来证明以下恒等式：

```py
example  :  (a  +  b)  *  (c  +  d)  =  a  *  c  +  a  *  d  +  b  *  c  +  b  *  d  :=  by
  sorry 
```

以下练习稍微有些挑战性。你可以使用下面列出的定理。

```py
example  (a  b  :  ℝ)  :  (a  +  b)  *  (a  -  b)  =  a  ^  2  -  b  ^  2  :=  by
  sorry

#check  pow_two  a
#check  mul_sub  a  b  c
#check  add_mul  a  b  c
#check  add_sub  a  b  c
#check  sub_sub  a  b  c
#check  add_zero  a 
```

我们也可以在上下文中的假设中执行重写。例如，`rw [mul_comm a b] at hyp` 将假设 `hyp` 中的 `a * b` 替换为 `b * a`。

```py
example  (a  b  c  d  :  ℝ)  (hyp  :  c  =  d  *  a  +  b)  (hyp'  :  b  =  a  *  d)  :  c  =  2  *  a  *  d  :=  by
  rw  [hyp']  at  hyp
  rw  [mul_comm  d  a]  at  hyp
  rw  [←  two_mul  (a  *  d)]  at  hyp
  rw  [←  mul_assoc  2  a  d]  at  hyp
  exact  hyp 
```

在最后一步，`exact` 策略可以使用 `hyp` 来解决目标，因为那时 `hyp` 与目标完全匹配。

我们通过指出 Mathlib 提供了一个有用的自动化功能，即 `ring` 策略，该策略旨在证明任何交换环中的恒等式，只要它们纯粹地来自环公理，而不使用任何局部假设，来结束本节。

```py
example  :  c  *  b  *  a  =  b  *  (a  *  c)  :=  by
  ring

example  :  (a  +  b)  *  (a  +  b)  =  a  *  a  +  2  *  (a  *  b)  +  b  *  b  :=  by
  ring

example  :  (a  +  b)  *  (a  -  b)  =  a  ^  2  -  b  ^  2  :=  by
  ring

example  (hyp  :  c  =  d  *  a  +  b)  (hyp'  :  b  =  a  *  d)  :  c  =  2  *  a  *  d  :=  by
  rw  [hyp,  hyp']
  ring 
```

当我们导入 `Mathlib.Data.Real.Basic` 时，`ring` 策略是间接导入的，但我们将看到在下一节中它可以用作对除了实数以外的结构进行计算。可以使用命令 `import Mathlib.Tactic` 明确导入。我们将看到还有其他类似策略用于其他常见的代数结构。

有一种 `rw` 的变体称为 `nth_rw`，它允许你只替换目标中特定实例的表达式。可能的匹配从 1 开始枚举，所以在这个例子中，`nth_rw 2 [h]` 将 `a + b` 的第二个出现替换为 `c`。

```py
example  (a  b  c  :  ℕ)  (h  :  a  +  b  =  c)  :  (a  +  b)  *  (a  +  b)  =  a  *  c  +  b  *  c  :=  by
  nth_rw  2  [h]
  rw  [add_mul] 
```

## 2.2\. 在代数结构中证明恒等式

从数学上讲，一个环由一组对象 $R$、运算 $+$ 和 $\times$、常数 $0$ 和 $1$ 以及一个运算 $x \mapsto -x$ 组成，使得：

+   $R$ 加号表示的是一个 *阿贝尔群*，其中 $0$ 是加法单位元，否定是逆元。

+   乘法与单位元 $1$ 结合，并且乘法对加法分配。

在 Lean 中，对象的集合被表示为一个 *类型*，`R`。环公理如下：

```py
variable  (R  :  Type*)  [Ring  R]

#check  (add_assoc  :  ∀  a  b  c  :  R,  a  +  b  +  c  =  a  +  (b  +  c))
#check  (add_comm  :  ∀  a  b  :  R,  a  +  b  =  b  +  a)
#check  (zero_add  :  ∀  a  :  R,  0  +  a  =  a)
#check  (neg_add_cancel  :  ∀  a  :  R,  -a  +  a  =  0)
#check  (mul_assoc  :  ∀  a  b  c  :  R,  a  *  b  *  c  =  a  *  (b  *  c))
#check  (mul_one  :  ∀  a  :  R,  a  *  1  =  a)
#check  (one_mul  :  ∀  a  :  R,  1  *  a  =  a)
#check  (mul_add  :  ∀  a  b  c  :  R,  a  *  (b  +  c)  =  a  *  b  +  a  *  c)
#check  (add_mul  :  ∀  a  b  c  :  R,  (a  +  b)  *  c  =  a  *  c  +  b  *  c) 
```

你将在以后了解第一行中的方括号，但暂时只需知道这个声明给我们一个类型，`R`，以及 `R` 上的环结构。Lean 然后允许我们使用 `R` 的元素进行通用的环符号，并利用关于环的定理库。

一些定理的名称应该看起来很熟悉：它们正是我们在上一节中使用实数进行计算时用到的。Lean 不仅擅长证明关于自然数和整数等具体数学结构的性质，而且也擅长证明关于抽象结构（如环）的性质。此外，Lean 支持关于抽象和具体结构的 *通用推理*，并且可以被训练以识别适当的实例。因此，任何关于环的定理都可以应用于具体的环，如整数环 `ℤ`、有理数环 `ℚ` 和复数环 `ℂ`。它也可以应用于任何扩展环的抽象结构的实例，例如任何有序环或任何域。

然而，并非所有实数的性质在任意环中都成立。例如，实数上的乘法是交换的，但这一点在一般情况下并不成立。如果你已经学过线性代数课程，你会认识到，对于每一个 $n$，实数的 $n$ 阶矩阵形成一个环，其中交换律通常不成立。如果我们声明 `R` 是一个 *交换环*，实际上，当我们将 `ℝ` 替换为 `R` 时，上一节中所有定理仍然成立。

```py
variable  (R  :  Type*)  [CommRing  R]
variable  (a  b  c  d  :  R)

example  :  c  *  b  *  a  =  b  *  (a  *  c)  :=  by  ring

example  :  (a  +  b)  *  (a  +  b)  =  a  *  a  +  2  *  (a  *  b)  +  b  *  b  :=  by  ring

example  :  (a  +  b)  *  (a  -  b)  =  a  ^  2  -  b  ^  2  :=  by  ring

example  (hyp  :  c  =  d  *  a  +  b)  (hyp'  :  b  =  a  *  d)  :  c  =  2  *  a  *  d  :=  by
  rw  [hyp,  hyp']
  ring 
```

我们留给你们去检查其他所有证明是否保持不变。注意，当证明很短，如 `by ring` 或 `by linarith` 或 `by sorry` 时，将其放在 `by` 的同一行上是常见（并且是允许的）。好的证明风格应该在简洁性和可读性之间取得平衡。

本节的目标是加强你在上一节中发展的技能，并将它们应用于关于环的公理推理。我们将从上面列出的公理开始，并使用它们推导出其他事实。我们证明的大多数事实已经在 Mathlib 中。我们将给出我们证明的版本相同的名称，以帮助你学习库的内容以及命名约定。

Lean 提供了一个类似于编程语言中使用的组织机制：当在*命名空间*`bar`中引入定义或定理`foo`时，它的全名是`bar.foo`。命令`open bar`稍后*打开*命名空间，这允许我们使用较短的名称`foo`。为了避免由于名称冲突而引起的错误，在下一个例子中，我们将我们的库定理版本放在一个新的命名空间`MyRing`中。

以下例子表明，我们不需要`add_zero`或`add_neg_cancel`作为环公理，因为它们可以从其他公理中得出。

```py
namespace  MyRing
variable  {R  :  Type*}  [Ring  R]

theorem  add_zero  (a  :  R)  :  a  +  0  =  a  :=  by  rw  [add_comm,  zero_add]

theorem  add_neg_cancel  (a  :  R)  :  a  +  -a  =  0  :=  by  rw  [add_comm,  neg_add_cancel]

#check  MyRing.add_zero
#check  add_zero

end  MyRing 
```

这种净效应是，我们可以暂时重新证明库中的一个定理，然后继续使用库版本。但不要作弊！在接下来的练习中，请务必只使用我们在本节前面已经证明的关于环的一般事实。

（如果你仔细观察，你可能已经注意到我们在`(R : Type*)`中将圆括号改为了花括号`{R : Type*}`。这表示`R`是一个*隐式参数*。我们稍后会解释这意味着什么，但在此期间不必担心。）

这里有一个有用的定理：

```py
theorem  neg_add_cancel_left  (a  b  :  R)  :  -a  +  (a  +  b)  =  b  :=  by
  rw  [←  add_assoc,  neg_add_cancel,  zero_add] 
```

证明其伴随版本：

```py
theorem  add_neg_cancel_right  (a  b  :  R)  :  a  +  b  +  -b  =  a  :=  by
  sorry 
```

使用这些来证明以下内容：

```py
theorem  add_left_cancel  {a  b  c  :  R}  (h  :  a  +  b  =  a  +  c)  :  b  =  c  :=  by
  sorry

theorem  add_right_cancel  {a  b  c  :  R}  (h  :  a  +  b  =  c  +  b)  :  a  =  c  :=  by
  sorry 
```

通过足够的规划，你可以用三次重写来完成每一个。

现在我们将解释花括号的使用。想象一下，你处于一个情境，其中你的上下文中有`a`、`b`和`c`，以及一个假设`h : a + b = a + c`，你想要得出结论`b = c`。在 Lean 中，你可以像对对象一样对假设和事实应用定理，所以你可能认为`add_left_cancel a b c h`是`b = c`这个事实的证明。但请注意，明确写出`a`、`b`和`c`是多余的，因为假设`h`清楚地表明了这些是我们心中的对象。在这种情况下，输入一些额外的字符并不麻烦，但如果我们要将`add_left_cancel`应用于更复杂的表达式，编写它们将会很繁琐。在这些情况下，Lean 允许我们将参数标记为*隐式*，这意味着它们应该被省略，并通过其他方式推断，例如后续的参数和假设。`{a b c : R}`中的花括号正是如此。因此，给定上述定理的陈述，正确的表达式仅仅是`add_left_cancel h`。

为了说明，让我们展示`a * 0 = 0`可以从环公理中得出。

```py
theorem  mul_zero  (a  :  R)  :  a  *  0  =  0  :=  by
  have  h  :  a  *  0  +  a  *  0  =  a  *  0  +  0  :=  by
  rw  [←  mul_add,  add_zero,  add_zero]
  rw  [add_left_cancel  h] 
```

我们使用了一个新技巧！如果你逐步查看证明过程，你可以看到正在发生什么。`have` 策略引入了一个新的目标，`a * 0 + a * 0 = a * 0 + 0`，与原始目标具有相同的上下文。下一行缩进的事实表明 Lean 预期一个用于证明这个新目标的策略块。因此，缩进促进了模块化证明风格：缩进的子证明建立了由 `have` 引入的目标。之后，我们回到证明原始目标，除了增加了一个新的假设 `h`：证明它之后，我们现在可以自由使用它。此时，目标正好是 `add_left_cancel h` 的结果。

我们同样可以用 `apply add_left_cancel h` 或 `exact add_left_cancel h` 来结束证明。`exact` 策略的参数是一个完全证明当前目标的证明项，而不创建任何新的目标。`apply` 策略是一个变体，其参数不一定是完整的证明。缺失的部分要么由 Lean 自动推断，要么成为需要证明的新目标。虽然 `exact` 策略在技术上可能是多余的，因为它严格不如 `apply` 强大，但它使证明脚本对人类读者来说更清晰，并且在库演变时更容易维护。

记住，乘法不一定假设是交换的，所以下面的定理也需要一些工作。

```py
theorem  zero_mul  (a  :  R)  :  0  *  a  =  0  :=  by
  sorry 
```

到现在为止，你也应该能够用证明替换下一个练习中的每个 `sorry`，仍然只使用我们在本节中建立的关于环的事实。

```py
theorem  neg_eq_of_add_eq_zero  {a  b  :  R}  (h  :  a  +  b  =  0)  :  -a  =  b  :=  by
  sorry

theorem  eq_neg_of_add_eq_zero  {a  b  :  R}  (h  :  a  +  b  =  0)  :  a  =  -b  :=  by
  sorry

theorem  neg_zero  :  (-0  :  R)  =  0  :=  by
  apply  neg_eq_of_add_eq_zero
  rw  [add_zero]

theorem  neg_neg  (a  :  R)  :  -  -a  =  a  :=  by
  sorry 
```

在第三个定理中，我们不得不使用注释 `(-0 : R)` 而不是 `0`，因为没有指定 `R`，Lean 无法推断我们心中所想的 `0` 是什么，默认情况下它会被解释为自然数。

在 Lean 中，环中的减法可以证明等于加法上的加法逆元。

```py
example  (a  b  :  R)  :  a  -  b  =  a  +  -b  :=
  sub_eq_add_neg  a  b 
```

在实数上，它是这样定义的：

```py
example  (a  b  :  ℝ)  :  a  -  b  =  a  +  -b  :=
  rfl

example  (a  b  :  ℝ)  :  a  -  b  =  a  +  -b  :=  by
  rfl 
```

证明项 `rfl` 是“自反性”的缩写。将其作为 `a - b = a + -b` 的证明迫使 Lean 展开定义并识别两边的相同。`rfl` 策略做的是同样的事情。这是 Lean 内在逻辑中所谓的 *定义性等价* 的一个例子。这意味着不仅可以用 `sub_eq_add_neg` 来重写 `a - b = a + -b`，而且在某些上下文中，当处理实数时，你可以将方程的两边互换使用。例如，你现在有足够的信息来证明上一节中的定理 `self_sub`：

```py
theorem  self_sub  (a  :  R)  :  a  -  a  =  0  :=  by
  sorry 
```

展示你可以使用 `rw` 来证明这一点，但如果将任意的环 `R` 替换为实数，你也可以使用 `apply` 或 `exact` 来证明它。

Lean 知道 `1 + 1 = 2` 在任何环中都成立。通过一点努力，你可以用它来证明上一节中的定理 `two_mul`：

```py
theorem  one_add_one_eq_two  :  1  +  1  =  (2  :  R)  :=  by
  norm_num

theorem  two_mul  (a  :  R)  :  2  *  a  =  a  +  a  :=  by
  sorry 
```

我们在本节的结尾指出，我们上面建立的一些关于加法和否定的事实不需要环公理的全部力量，甚至不需要加法的交换性。一个 *群* 的较弱概念可以如下公理化：

```py
variable  (A  :  Type*)  [AddGroup  A]

#check  (add_assoc  :  ∀  a  b  c  :  A,  a  +  b  +  c  =  a  +  (b  +  c))
#check  (zero_add  :  ∀  a  :  A,  0  +  a  =  a)
#check  (neg_add_cancel  :  ∀  a  :  A,  -a  +  a  =  0) 
```

当群运算交换时，使用加法符号是惯例，否则使用乘法符号。因此，Lean 定义了乘法版本以及加法版本（以及它们的阿贝尔变体，`AddCommGroup` 和 `CommGroup`）。

```py
variable  {G  :  Type*}  [Group  G]

#check  (mul_assoc  :  ∀  a  b  c  :  G,  a  *  b  *  c  =  a  *  (b  *  c))
#check  (one_mul  :  ∀  a  :  G,  1  *  a  =  a)
#check  (inv_mul_cancel  :  ∀  a  :  G,  a⁻¹  *  a  =  1) 
```

如果你感到自信，尝试使用这些公理证明以下关于群的事实。在过程中，你需要证明许多辅助引理。本节中我们进行的证明提供了一些提示。

```py
theorem  mul_inv_cancel  (a  :  G)  :  a  *  a⁻¹  =  1  :=  by
  sorry

theorem  mul_one  (a  :  G)  :  a  *  1  =  a  :=  by
  sorry

theorem  mul_inv_rev  (a  b  :  G)  :  (a  *  b)⁻¹  =  b⁻¹  *  a⁻¹  :=  by
  sorry 
```

明确调用这些引理是繁琐的，因此 Mathlib 提供了类似于环的策略来覆盖大多数用法：`group` 用于非交换乘法群，`abel` 用于阿贝尔加法群，`noncomm_ring` 用于非交换环。看起来很奇怪，代数结构被称为 Ring 和 CommRing，而策略被命名为 noncomm_ring 和 ring。这部分是历史原因，但也为了方便使用更短的名称来处理交换环的策略，因为它使用得更频繁。

## 2.3\. 使用定理和引理

重写对于证明方程非常有用，但对于其他类型的定理呢？例如，我们如何证明一个不等式，比如 $a + e^b \le a + e^c$ 在 $b \le c$ 时总是成立？我们已经看到定理可以应用于论点和假设，并且可以使用 `apply` 和 `exact` 策略来解决目标。在本节中，我们将充分利用这些工具。

考虑库定理 `le_refl` 和 `le_trans`：

```py
#check  (le_refl  :  ∀  a  :  ℝ,  a  ≤  a)
#check  (le_trans  :  a  ≤  b  →  b  ≤  c  →  a  ≤  c) 
```

如我们在第 3.1 节中更详细地解释的那样，`le_trans` 声明中的隐式括号与右侧关联，因此它应该被解释为 `a ≤ b → (b ≤ c → a ≤ c)`。库设计者已经将 `a`、`b` 和 `c` 设置为 `le_trans` 的隐式参数，这样 Lean 就不会让你明确地提供它们（除非你真的坚持，我们稍后会讨论）。相反，它期望从它们被使用的上下文中推断它们。例如，当假设 `h : a ≤ b` 和 `h' : b ≤ c` 在上下文中时，以下所有内容都有效：

```py
variable  (h  :  a  ≤  b)  (h'  :  b  ≤  c)

#check  (le_refl  :  ∀  a  :  Real,  a  ≤  a)
#check  (le_refl  a  :  a  ≤  a)
#check  (le_trans  :  a  ≤  b  →  b  ≤  c  →  a  ≤  c)
#check  (le_trans  h  :  b  ≤  c  →  a  ≤  c)
#check  (le_trans  h  h'  :  a  ≤  c) 
```

`apply` 策略接受一个一般陈述或蕴涵的证明，尝试将结论与当前目标匹配，并将假设（如果有的话）作为新目标留下。如果给定的证明与目标完全匹配（模 *定义性* 等价），则可以使用 `exact` 策略代替 `apply`。所以，所有这些都有效：

```py
example  (x  y  z  :  ℝ)  (h₀  :  x  ≤  y)  (h₁  :  y  ≤  z)  :  x  ≤  z  :=  by
  apply  le_trans
  ·  apply  h₀
  ·  apply  h₁

example  (x  y  z  :  ℝ)  (h₀  :  x  ≤  y)  (h₁  :  y  ≤  z)  :  x  ≤  z  :=  by
  apply  le_trans  h₀
  apply  h₁

example  (x  y  z  :  ℝ)  (h₀  :  x  ≤  y)  (h₁  :  y  ≤  z)  :  x  ≤  z  :=
  le_trans  h₀  h₁

example  (x  :  ℝ)  :  x  ≤  x  :=  by
  apply  le_refl

example  (x  :  ℝ)  :  x  ≤  x  :=
  le_refl  x 
```

在第一个例子中，应用 `le_trans` 创建了两个目标，我们使用点来表示每个证明的开始位置。点是可以省略的，但它们有助于 *聚焦* 目标：在点引入的块内，只有一个目标可见，并且必须在块的末尾之前完成。在这里，我们通过用另一个点开始一个新的块来结束第一个块。我们也可以减少缩进。在第三个例子和最后一个例子中，我们完全避免了进入策略模式：`le_trans h₀ h₁` 和 `le_refl x` 是我们需要证明的项。

这里有一些更多的库定理：

```py
#check  (le_refl  :  ∀  a,  a  ≤  a)
#check  (le_trans  :  a  ≤  b  →  b  ≤  c  →  a  ≤  c)
#check  (lt_of_le_of_lt  :  a  ≤  b  →  b  <  c  →  a  <  c)
#check  (lt_of_lt_of_le  :  a  <  b  →  b  ≤  c  →  a  <  c)
#check  (lt_trans  :  a  <  b  →  b  <  c  →  a  <  c) 
```

将它们与 `apply` 和 `exact` 结合起来，以证明以下内容：

```py
example  (h₀  :  a  ≤  b)  (h₁  :  b  <  c)  (h₂  :  c  ≤  d)  (h₃  :  d  <  e)  :  a  <  e  :=  by
  sorry 
```

事实上，Lean 有一种策略可以自动完成这类事情：

```py
example  (h₀  :  a  ≤  b)  (h₁  :  b  <  c)  (h₂  :  c  ≤  d)  (h₃  :  d  <  e)  :  a  <  e  :=  by
  linarith 
```

`linarith` 策略被设计用来处理 *线性代数*。

```py
example  (h  :  2  *  a  ≤  3  *  b)  (h'  :  1  ≤  a)  (h''  :  d  =  2)  :  d  +  a  ≤  5  *  b  :=  by
  linarith 
```

除了上下文中的方程和不等式之外，`linarith` 还会使用你作为参数传递的额外不等式。在下一个例子中，`exp_le_exp.mpr h'` 是 `exp b ≤ exp c` 的证明，我们将在下一刻解释。请注意，在 Lean 中，我们用 `f x` 来表示函数 `f` 对参数 `x` 的应用，这与我们用 `h x` 来表示事实或定理 `h` 对参数 `x` 的应用完全相同。括号仅用于复合参数，例如 `f (x + y)`。如果没有括号，`f x + y` 将被解析为 `(f x) + y`。

```py
example  (h  :  1  ≤  a)  (h'  :  b  ≤  c)  :  2  +  a  +  exp  b  ≤  3  *  a  +  exp  c  :=  by
  linarith  [exp_le_exp.mpr  h'] 
```

这里有一些库中的更多定理，可以用来在实数上建立不等式。

```py
#check  (exp_le_exp  :  exp  a  ≤  exp  b  ↔  a  ≤  b)
#check  (exp_lt_exp  :  exp  a  <  exp  b  ↔  a  <  b)
#check  (log_le_log  :  0  <  a  →  a  ≤  b  →  log  a  ≤  log  b)
#check  (log_lt_log  :  0  <  a  →  a  <  b  →  log  a  <  log  b)
#check  (add_le_add  :  a  ≤  b  →  c  ≤  d  →  a  +  c  ≤  b  +  d)
#check  (add_le_add_left  :  a  ≤  b  →  ∀  c,  c  +  a  ≤  c  +  b)
#check  (add_le_add_right  :  a  ≤  b  →  ∀  c,  a  +  c  ≤  b  +  c)
#check  (add_lt_add_of_le_of_lt  :  a  ≤  b  →  c  <  d  →  a  +  c  <  b  +  d)
#check  (add_lt_add_of_lt_of_le  :  a  <  b  →  c  ≤  d  →  a  +  c  <  b  +  d)
#check  (add_lt_add_left  :  a  <  b  →  ∀  c,  c  +  a  <  c  +  b)
#check  (add_lt_add_right  :  a  <  b  →  ∀  c,  a  +  c  <  b  +  c)
#check  (add_nonneg  :  0  ≤  a  →  0  ≤  b  →  0  ≤  a  +  b)
#check  (add_pos  :  0  <  a  →  0  <  b  →  0  <  a  +  b)
#check  (add_pos_of_pos_of_nonneg  :  0  <  a  →  0  ≤  b  →  0  <  a  +  b)
#check  (exp_pos  :  ∀  a,  0  <  exp  a)
#check  add_le_add_left 
```

一些定理，如 `exp_le_exp` 和 `exp_lt_exp`，使用 *双向蕴涵*，表示“当且仅当”的短语。（你可以在 VS Code 中使用 `\lr` 或 `\iff` 来输入它）。我们将在下一章更详细地讨论这个连接词。这样的定理可以用 `rw` 将目标重写为等价的目标：

```py
example  (h  :  a  ≤  b)  :  exp  a  ≤  exp  b  :=  by
  rw  [exp_le_exp]
  exact  h 
```

然而，在本节中，我们将使用以下事实：如果 `h : A ↔ B` 是这样的等价关系，那么 `h.mp` 建立了正向方向，`A → B`，而 `h.mpr` 建立了反向方向，`B → A`。在这里，`mp` 代表“肯定前件”（modus ponens），而 `mpr` 代表“否定后件”（modus ponens reverse）。如果你愿意，你也可以使用 `h.1` 和 `h.2` 来代替 `h.mp` 和 `h.mpr`。因此，以下证明是有效的：

```py
example  (h₀  :  a  ≤  b)  (h₁  :  c  <  d)  :  a  +  exp  c  +  e  <  b  +  exp  d  +  e  :=  by
  apply  add_lt_add_of_lt_of_le
  ·  apply  add_lt_add_of_le_of_lt  h₀
  apply  exp_lt_exp.mpr  h₁
  apply  le_refl 
```

第一行，`apply add_lt_add_of_lt_of_le` 创建了两个目标，并且再次我们使用点来区分第一个证明和第二个证明。

尝试以下示例。中间的示例显示你可以使用 `norm_num` 策略来解决具体的数值目标。

```py
example  (h₀  :  d  ≤  e)  :  c  +  exp  (a  +  d)  ≤  c  +  exp  (a  +  e)  :=  by  sorry

example  :  (0  :  ℝ)  <  1  :=  by  norm_num

example  (h  :  a  ≤  b)  :  log  (1  +  exp  a)  ≤  log  (1  +  exp  b)  :=  by
  have  h₀  :  0  <  1  +  exp  a  :=  by  sorry
  apply  log_le_log  h₀
  sorry 
```

从这些例子中，应该很明显，能够找到你需要的库定理是形式化的重要组成部分。你可以使用多种策略：

+   你可以在 [GitHub 仓库](https://github.com/leanprover-community/mathlib4) 中浏览 Mathlib。

+   你可以使用 Mathlib [网页](https://leanprover-community.github.io/mathlib4_docs/) 上的 API 文档。

+   你可以使用 Loogle <https://loogle.lean-lang.org> 通过模式搜索 Lean 和 Mathlib 的定义和定理。

+   你可以依靠 Mathlib 的命名约定和编辑器中的 Ctrl-space 完成功能来猜测定理名称（或在 Mac 键盘上按 Cmd-space）。在 Lean 中，名为 `A_of_B_of_C` 的定理从形式为 `B` 和 `C` 的假设中建立 `A`，其中 `A`、`B` 和 `C` 大致对应于我们大声读出目标的方式。因此，建立类似 `x + y ≤ ...` 的定理可能以 `add_le` 开头。键入 `add_le` 并按 Ctrl-space 将提供一些有用的选择。请注意，按 Ctrl-space 两次将显示有关可用完成信息的更多信息。

+   如果你右键单击 VS Code 中的一个现有定理名称，编辑器将显示一个菜单，其中包含跳转到定理定义文件的选项，你可以在附近找到类似定理。

+   你可以使用 `apply?` 策略，它会尝试在库中找到相关的定理。

```py
example  :  0  ≤  a  ^  2  :=  by
  -- apply?
  exact  sq_nonneg  a 
```

要尝试在这个例子中应用 `apply?`，请删除 `exact` 命令并取消注释上一行。使用这些技巧，看看你是否能找到完成下一个例子所需的内容：

```py
example  (h  :  a  ≤  b)  :  c  -  exp  b  ≤  c  -  exp  a  :=  by
  sorry 
```

使用相同的技巧，确认 `linarith` 而不是 `apply?` 也可以完成这项工作。

这里是另一个不等式的例子：

```py
example  :  2*a*b  ≤  a²  +  b²  :=  by
  have  h  :  0  ≤  a²  -  2*a*b  +  b²
  calc
  a²  -  2*a*b  +  b²  =  (a  -  b)²  :=  by  ring
  _  ≥  0  :=  by  apply  pow_two_nonneg

  calc
  2*a*b  =  2*a*b  +  0  :=  by  ring
  _  ≤  2*a*b  +  (a²  -  2*a*b  +  b²)  :=  add_le_add  (le_refl  _)  h
  _  =  a²  +  b²  :=  by  ring 
```

Mathlib 倾向于在二进制运算符如 `*` 和 `^` 的周围放置空格，但在这个例子中，更紧凑的格式提高了可读性。有几个值得注意的地方。首先，表达式 `s ≥ t` 在定义上是等价于 `t ≤ s` 的。原则上，这意味着应该能够互换使用它们。但是 Lean 的一些自动化工具没有识别出这种等价性，因此 Mathlib 倾向于更喜欢 `≤` 而不是 `≥`。其次，我们广泛使用了 `ring` 策略。这是一个真正的省时工具！最后，请注意，在第二个 `calc` 证明的第二行中，我们不需要写 `by exact add_le_add (le_refl _) h`，我们可以简单地写证明项 `add_le_add (le_refl _) h`。

实际上，上述证明中唯一的巧妙之处在于找出假设 `h`。一旦我们找到了它，第二个计算就只涉及线性算术，而 `linarith` 可以处理它：

```py
example  :  2*a*b  ≤  a²  +  b²  :=  by
  have  h  :  0  ≤  a²  -  2*a*b  +  b²
  calc
  a²  -  2*a*b  +  b²  =  (a  -  b)²  :=  by  ring
  _  ≥  0  :=  by  apply  pow_two_nonneg
  linarith 
```

多么好！我们挑战你使用这些想法来证明以下定理。你可以使用定理 `abs_le'.mpr`。你还需要使用 `constructor` 策略将合取分解为两个目标；参见第 3.4 节。

```py
example  :  |a*b|  ≤  (a²  +  b²)/2  :=  by
  sorry

#check  abs_le'.mpr 
```

如果你设法解决了这个问题，恭喜你！你正朝着成为一位大师级形式化专家的道路上迈进。

## 2.4\. 使用 apply 和 rw 的更多示例

实数上的 `min` 函数由以下三个事实唯一确定：

```py
#check  (min_le_left  a  b  :  min  a  b  ≤  a)
#check  (min_le_right  a  b  :  min  a  b  ≤  b)
#check  (le_min  :  c  ≤  a  →  c  ≤  b  →  c  ≤  min  a  b) 
```

你能猜出那些以类似方式描述 `max` 的定理的名字吗？

注意，我们必须通过编写 `min a b` 而不是 `min (a, b)` 来将 `min` 应用到一对参数 `a` 和 `b` 上。形式上，`min` 是一个类型为 `ℝ → ℝ → ℝ` 的函数。当我们用多个箭头写出这样的类型时，惯例是隐式括号向右结合，因此类型被解释为 `ℝ → (ℝ → ℝ)`。最终的效果是，如果 `a` 和 `b` 的类型为 `ℝ`，那么 `min a` 的类型为 `ℝ → ℝ`，而 `min a b` 的类型为 `ℝ`，所以 `min` 的行为就像一个接受两个参数的函数，正如我们所期望的那样。以这种方式处理多个参数的方法被称为 *currying*，这是以逻辑学家 Haskell Curry 命名的。

Lean 中运算的顺序也需要一些时间来习惯。函数应用比中缀操作绑定得更紧，所以表达式 `min a b + c` 被解释为 `(min a b) + c`。随着时间的推移，这些惯例将变得习以为常。

使用定理 `le_antisymm`，我们可以证明如果每个数都小于或等于另一个数，那么两个实数是相等的。利用这一点和上述事实，我们可以证明 `min` 是可交换的：

```py
example  :  min  a  b  =  min  b  a  :=  by
  apply  le_antisymm
  ·  show  min  a  b  ≤  min  b  a
  apply  le_min
  ·  apply  min_le_right
  apply  min_le_left
  ·  show  min  b  a  ≤  min  a  b
  apply  le_min
  ·  apply  min_le_right
  apply  min_le_left 
```

在这里，我们使用了点来分隔不同目标证明。我们的用法不一致：在外层，我们使用点和缩进来表示两个目标，而对于嵌套证明，我们只使用点，直到只剩下一个目标。这两种惯例都是合理且有用的。我们还使用 `show` 策略来结构化证明并指示每个块中正在证明的内容。即使没有 `show` 命令，证明仍然有效，但使用它们可以使证明更容易阅读和维护。

可能会让你感到烦恼的是，证明是重复的。为了预示你以后将学习到的技能，我们注意到避免重复的一种方法就是陈述一个局部引理然后使用它：

```py
example  :  min  a  b  =  min  b  a  :=  by
  have  h  :  ∀  x  y  :  ℝ,  min  x  y  ≤  min  y  x  :=  by
  intro  x  y
  apply  le_min
  apply  min_le_right
  apply  min_le_left
  apply  le_antisymm
  apply  h
  apply  h 
```

我们将在 第 3.1 节 中详细介绍全称量词，但在这里只需说，假设 `h` 表示对于任何 `x` 和 `y`，所期望的不等式都成立，而 `intro` 策略通过引入任意的 `x` 和 `y` 来建立结论。在 `le_antisymm` 之后的第一个 `apply` 隐式地使用了 `h a b`，而第二个则使用了 `h b a`。

另一种解决方案是使用 `repeat` 策略，它可以尽可能多次地应用策略（或一个块）。

```py
example  :  min  a  b  =  min  b  a  :=  by
  apply  le_antisymm
  repeat
  apply  le_min
  apply  min_le_right
  apply  min_le_left 
```

我们鼓励您将以下内容作为练习来证明。您可以使用刚刚描述的任何一种技巧来简化第一个。

```py
example  :  max  a  b  =  max  b  a  :=  by
  sorry
example  :  min  (min  a  b)  c  =  min  a  (min  b  c)  :=  by
  sorry 
```

当然，你也可以证明 `max` 的结合性。

一个有趣的事实是 `min` 在 `max` 上的分配方式与乘法在加法上的分配方式相同，反之亦然。换句话说，在实数上，我们有恒等式 `min a (max b c) = max (min a b) (min a c)`，以及相应地交换 `max` 和 `min` 的版本。但在下一节中，我们将看到这并不从 `≤` 的传递性和自反性以及上述 `min` 和 `max` 的特征属性中得出。我们需要使用实数上的 `≤` 是一个 *全序* 的这一事实，也就是说，它满足 `∀ x y, x ≤ y ∨ y ≤ x`。这里的析取符号，`∨`，代表“或”。在第一种情况下，我们有 `min x y = x`，在第二种情况下，我们有 `min x y = y`。我们将在 第 3.5 节中学习如何进行分情况推理，但就目前而言，我们将坚持使用不需要分情况分析的例子。

这里有一个这样的例子：

```py
theorem  aux  :  min  a  b  +  c  ≤  min  (a  +  c)  (b  +  c)  :=  by
  sorry
example  :  min  a  b  +  c  =  min  (a  +  c)  (b  +  c)  :=  by
  sorry 
```

显然，`aux` 提供了证明等式所需两个不等式之一，但将其应用于合适的值也会得到另一个方向。作为一个提示，你可以使用定理 `add_neg_cancel_right` 和 `linarith` 策略。

Lean 的命名约定在库中三角不等式的名称中得到了体现：

```py
#check  (abs_add  :  ∀  a  b  :  ℝ,  |a  +  b|  ≤  |a|  +  |b|) 
```

使用它来证明以下变体，同时使用 `add_sub_cancel_right`：

```py
example  :  |a|  -  |b|  ≤  |a  -  b|  :=
  sorry
end 
```

看看你是否能在三行或更少的文字中完成这个任务。你可以使用定理 `sub_add_cancel`。

在接下来的章节中，我们将使用的一个重要关系是自然数上的可除性关系，`x ∣ y`。请注意：可除性符号 *不是* 你键盘上的普通竖线。而是通过在 VS Code 中输入 `\|` 获得的 unicode 字符。按照惯例，Mathlib 在定理名称中使用 `dvd` 来指代它。

```py
example  (h₀  :  x  ∣  y)  (h₁  :  y  ∣  z)  :  x  ∣  z  :=
  dvd_trans  h₀  h₁

example  :  x  ∣  y  *  x  *  z  :=  by
  apply  dvd_mul_of_dvd_left
  apply  dvd_mul_left

example  :  x  ∣  x  ^  2  :=  by
  apply  dvd_mul_left 
```

在上一个例子中，指数是一个自然数，应用 `dvd_mul_left` 会迫使 Lean 展开定义 `x²` 为 `x¹ * x`。看看你是否能猜出你需要证明以下定理的名称：

```py
example  (h  :  x  ∣  w)  :  x  ∣  y  *  (x  *  z)  +  x  ^  2  +  w  ^  2  :=  by
  sorry
end 
```

在除法方面，*最大公约数*，`gcd`，和最小公倍数，`lcm`，类似于 `min` 和 `max`。由于每个数都能整除 `0`，所以 `0` 真的是除法意义上的最大元素：

```py
variable  (m  n  :  ℕ)

#check  (Nat.gcd_zero_right  n  :  Nat.gcd  n  0  =  n)
#check  (Nat.gcd_zero_left  n  :  Nat.gcd  0  n  =  n)
#check  (Nat.lcm_zero_right  n  :  Nat.lcm  n  0  =  0)
#check  (Nat.lcm_zero_left  n  :  Nat.lcm  0  n  =  0) 
```

看看你是否能猜出你需要证明以下定理的名称：

```py
example  :  Nat.gcd  m  n  =  Nat.gcd  n  m  :=  by
  sorry 
```

提示：你可以使用 `dvd_antisymm`，但如果你这样做，Lean 会抱怨表达式在通用定理和版本 `Nat.dvd_antisymm`（专门针对自然数的版本）之间是模糊的。你可以使用 `_root_.dvd_antisymm` 来指定通用版本；两者都适用。

## 2.5\. 证明关于代数结构的事实

在 第 2.2 节 中，我们看到了许多常见的关于实数的恒等式在更一般的代数结构中同样成立，例如交换环。我们可以使用任何我们想要的公理来描述代数结构，而不仅仅是方程。例如，*偏序* 由一个集合和二元关系组成，该关系是自反的、传递的和反对称的，就像实数上的 `≤`。Lean 了解偏序：

```py
variable  {α  :  Type*}  [PartialOrder  α]
variable  (x  y  z  :  α)

#check  x  ≤  y
#check  (le_refl  x  :  x  ≤  x)
#check  (le_trans  :  x  ≤  y  →  y  ≤  z  →  x  ≤  z)
#check  (le_antisymm  :  x  ≤  y  →  y  ≤  x  →  x  =  y) 
```

在这里，我们采用了 Mathlib 的约定，使用像 `α`、`β` 和 `γ`（输入为 `\a`、`\b` 和 `\g`）这样的字母表示任意类型。该库通常使用像 `R` 和 `G` 这样的字母表示代数结构如环和群的载体，但通常希腊字母用于表示类型，尤其是在它们与很少或没有结构相关联时。

与任何偏序 `≤` 相关，也存在一个 *严格偏序* `<`，它在某种程度上类似于实数上的 `<`。在这个顺序中，说 `x` 小于 `y` 等同于说它小于或等于 `y` 但不等于 `y`。

```py
#check  x  <  y
#check  (lt_irrefl  x  :  ¬  (x  <  x))
#check  (lt_trans  :  x  <  y  →  y  <  z  →  x  <  z)
#check  (lt_of_le_of_lt  :  x  ≤  y  →  y  <  z  →  x  <  z)
#check  (lt_of_lt_of_le  :  x  <  y  →  y  ≤  z  →  x  <  z)

example  :  x  <  y  ↔  x  ≤  y  ∧  x  ≠  y  :=
  lt_iff_le_and_ne 
```

在这个例子中，符号 `∧` 表示“和”，符号 `¬` 表示“非”，`x ≠ y` 简写为 `¬ (x = y)`。在 第三章 中，您将学习如何使用这些逻辑连接词来 *证明* `<` 具有指示的性质。

*格* 是一个扩展了偏序并带有类似于实数上的 `min` 和 `max` 的操作 `⊓` 和 `⊔` 的结构：

```py
variable  {α  :  Type*}  [Lattice  α]
variable  (x  y  z  :  α)

#check  x  ⊓  y
#check  (inf_le_left  :  x  ⊓  y  ≤  x)
#check  (inf_le_right  :  x  ⊓  y  ≤  y)
#check  (le_inf  :  z  ≤  x  →  z  ≤  y  →  z  ≤  x  ⊓  y)
#check  x  ⊔  y
#check  (le_sup_left  :  x  ≤  x  ⊔  y)
#check  (le_sup_right  :  y  ≤  x  ⊔  y)
#check  (sup_le  :  x  ≤  z  →  y  ≤  z  →  x  ⊔  y  ≤  z) 
```

`⊓` 和 `⊔` 的特征使得它们分别被称为 *最大下界* 和 *最小上界*。您可以在 VS code 中使用 `\glb` 和 `\lub` 来输入它们。这些符号也常被称为 *下确界* 和 *上确界*，Mathlib 在定理名称中称它们为 `inf` 和 `sup`。为了进一步复杂化问题，它们也常被称为 *交* 和 *并*。因此，如果您与格（lattices）一起工作，您必须记住以下字典：

+   `⊓` 是 *最大下界*、*下确界* 或 *交*。

+   `⊔` 是 *最小上界*、*上确界* 或 *并*。

一些格的实例包括：

+   在任何全序，如整数或实数上的 `≤`，的 `min` 和 `max`

+   集合的子集上的 `∩` 和 `∪`，具有顺序 `⊆`

+   布尔真值上的 `∧` 和 `∨`，如果 `x` 为假或 `y` 为真，则顺序 `x ≤ y`

+   自然数（或正自然数）上的 `gcd` 和 `lcm`，具有除法顺序 `∣`

+   向量空间的线性子空间的集合，其中最大下界由交集给出，最小上界由两个空间的和给出，顺序是包含

+   集合上的拓扑（或在 Lean 中，类型）的集合，其中两个拓扑的最大下界是由它们的并集生成的拓扑，最小上界是它们的交集，顺序是反向包含

你可以检查，与`min`/`max`和`gcd`/`lcm`一样，你可以仅使用它们的特征公理以及`le_refl`和`le_trans`证明下确界和上确界的交换性和结合性：

当看到目标`x ≤ z`时使用`apply le_trans`不是一个好主意。实际上，Lean 没有方法猜测我们想要使用哪个中间元素`y`。因此，`apply le_trans`会产生三个看起来像`x ≤ ?a`、`?a ≤ z`和`α`的目标，其中`?a`（可能有一个更复杂的自动生成的名称）代表神秘的`y`。最后一个目标，类型为`α`，是提供`y`的值。它最后出现，因为 Lean 希望从第一个目标`x ≤ ?a`的证明中自动推断它。为了避免这种不吸引人的情况，你可以使用`calc`策略显式提供`y`。或者，你可以使用接受`y`作为参数的`trans`策略，它会产生预期的目标`x ≤ y`和`y ≤ z`。当然，你也可以通过直接提供一个完整的证明来避免这个问题，例如`exact le_trans inf_le_left inf_le_right`，但这需要更多的计划。

```py
example  :  x  ⊓  y  =  y  ⊓  x  :=  by
  sorry

example  :  x  ⊓  y  ⊓  z  =  x  ⊓  (y  ⊓  z)  :=  by
  sorry

example  :  x  ⊔  y  =  y  ⊔  x  :=  by
  sorry

example  :  x  ⊔  y  ⊔  z  =  x  ⊔  (y  ⊔  z)  :=  by
  sorry 
```

你可以在 Mathlib 中找到这些定理，分别命名为`inf_comm`、`inf_assoc`、`sup_comm`和`sup_assoc`。

另一个很好的练习是仅使用这些公理证明**吸收律**：

```py
theorem  absorb1  :  x  ⊓  (x  ⊔  y)  =  x  :=  by
  sorry

theorem  absorb2  :  x  ⊔  x  ⊓  y  =  x  :=  by
  sorry 
```

这些可以在 Mathlib 中找到，名称分别为`inf_sup_self`和`sup_inf_self`。

满足额外恒等式`x ⊓ (y ⊔ z) = (x ⊓ y) ⊔ (x ⊓ z)`和`x ⊔ (y ⊓ z) = (x ⊔ y) ⊓ (x ⊔ z)`的格称为**分配格**。Lean 也了解这些：

```py
variable  {α  :  Type*}  [DistribLattice  α]
variable  (x  y  z  :  α)

#check  (inf_sup_left  x  y  z  :  x  ⊓  (y  ⊔  z)  =  x  ⊓  y  ⊔  x  ⊓  z)
#check  (inf_sup_right  x  y  z  :  (x  ⊔  y)  ⊓  z  =  x  ⊓  z  ⊔  y  ⊓  z)
#check  (sup_inf_left  x  y  z  :  x  ⊔  y  ⊓  z  =  (x  ⊔  y)  ⊓  (x  ⊔  z))
#check  (sup_inf_right  x  y  z  :  x  ⊓  y  ⊔  z  =  (x  ⊔  z)  ⊓  (y  ⊔  z)) 
```

左侧和右侧版本很容易证明是等价的，考虑到`⊓`和`⊔`的交换性。通过提供一个非分配格的显式描述，证明不是每个格都是分配的，这是一个很好的练习。同样，证明在任何格中，分配律中的任意一个都蕴含另一个，也是一个很好的练习：

```py
variable  {α  :  Type*}  [Lattice  α]
variable  (a  b  c  :  α)

example  (h  :  ∀  x  y  z  :  α,  x  ⊓  (y  ⊔  z)  =  x  ⊓  y  ⊔  x  ⊓  z)  :  a  ⊔  b  ⊓  c  =  (a  ⊔  b)  ⊓  (a  ⊔  c)  :=  by
  sorry

example  (h  :  ∀  x  y  z  :  α,  x  ⊔  y  ⊓  z  =  (x  ⊔  y)  ⊓  (x  ⊔  z))  :  a  ⊓  (b  ⊔  c)  =  a  ⊓  b  ⊔  a  ⊓  c  :=  by
  sorry 
```

可以将公理结构组合成更大的结构。例如，一个**严格有序环**由一个环以及一个在载体上的偏序组成，该偏序满足额外的公理，这些公理说明环运算与顺序是兼容的：

```py
variable  {R  :  Type*}  [Ring  R]  [PartialOrder  R]  [IsStrictOrderedRing  R]
variable  (a  b  c  :  R)

#check  (add_le_add_left  :  a  ≤  b  →  ∀  c,  c  +  a  ≤  c  +  b)
#check  (mul_pos  :  0  <  a  →  0  <  b  →  0  <  a  *  b) 
```

第三章将提供从`mul_pos`和`<`的定义推导以下内容的方法：

```py
#check  (mul_nonneg  :  0  ≤  a  →  0  ≤  b  →  0  ≤  a  *  b) 
```

然后是一个扩展练习，用以证明许多用于对算术和实数排序进行推理的常见事实在任意有序环中普遍成立。这里有一些你可以尝试的例子，仅使用环的性质、偏序以及前两个例子中列出的事实（请注意，这些环并不假设是交换的，因此环策略不可用）：

```py
example  (h  :  a  ≤  b)  :  0  ≤  b  -  a  :=  by
  sorry

example  (h:  0  ≤  b  -  a)  :  a  ≤  b  :=  by
  sorry

example  (h  :  a  ≤  b)  (h'  :  0  ≤  c)  :  a  *  c  ≤  b  *  c  :=  by
  sorry 
```

最后，这里有一个最后的例子。一个**度量空间**由一个集合组成，该集合具有距离的概念，`dist x y`，将任何一对元素映射到一个实数。距离函数假设满足以下公理：

```py
variable  {X  :  Type*}  [MetricSpace  X]
variable  (x  y  z  :  X)

#check  (dist_self  x  :  dist  x  x  =  0)
#check  (dist_comm  x  y  :  dist  x  y  =  dist  y  x)
#check  (dist_triangle  x  y  z  :  dist  x  z  ≤  dist  x  y  +  dist  y  z) 
```

在掌握这一章节后，你可以证明以下公理意味着距离总是非负的：

```py
example  (x  y  :  X)  :  0  ≤  dist  x  y  :=  by
  sorry 
```

我们推荐使用定理 `nonneg_of_mul_nonneg_left`。正如你可能猜到的，这个定理在 Mathlib 中被称为 `dist_nonneg`*。
