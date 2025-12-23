# 3. 逻辑

> 原文：[`leanprover-community.github.io/mathematics_in_lean/C03_Logic.html`](https://leanprover-community.github.io/mathematics_in_lean/C03_Logic.html)

*数学在 Lean 中* **   3. 逻辑

+   查看页面源代码

* * *

在上一章中，我们处理了方程、不等式和像“$x$ 整除 $y$”这样的基本数学陈述。复杂的数学陈述是通过使用逻辑术语“和”、“或”、“非”、“如果…那么”、“每个”和“一些”从这些简单的陈述构建而成的。在本章中，我们将向您展示如何处理以这种方式构建的陈述。

## 3.1. 蕴含和全称量词

考虑 `#check` 之后的陈述：

```py
#check  ∀  x  :  ℝ,  0  ≤  x  →  |x|  =  x 
```

用文字来说，我们会说“对于每一个实数 `x`，如果 `0 ≤ x`，那么 `x` 的绝对值等于 `x`”。我们也可以有更复杂的陈述，例如：

```py
#check  ∀  x  y  ε  :  ℝ,  0  <  ε  →  ε  ≤  1  →  |x|  <  ε  →  |y|  <  ε  →  |x  *  y|  <  ε 
```

用文字来说，我们会说“对于每一个 `x`、`y` 和 `ε`，如果 `0 < ε ≤ 1`，则 `x` 的绝对值小于 `ε`，`y` 的绝对值小于 `ε`，那么 `x * y` 的绝对值小于 `ε`。”在 Lean 中，在一系列蕴含中，有隐式的括号分组到右边。所以上面的表达式意味着“如果 `0 < ε`，那么如果 `ε ≤ 1`，那么如果 `|x| < ε`…” 因此，这个表达式表明所有的假设共同蕴含了结论。

您已经看到，尽管这个陈述中的全称量词遍历对象，而蕴含箭头引入假设，但 Lean 以非常相似的方式处理这两个概念。特别是，如果您已经证明了这种形式的定理，您可以用相同的方式将其应用于对象和假设。我们将以下陈述作为例子，我们将在稍后帮助您证明：

```py
theorem  my_lemma  :  ∀  x  y  ε  :  ℝ,  0  <  ε  →  ε  ≤  1  →  |x|  <  ε  →  |y|  <  ε  →  |x  *  y|  <  ε  :=
  sorry

section
variable  (a  b  δ  :  ℝ)
variable  (h₀  :  0  <  δ)  (h₁  :  δ  ≤  1)
variable  (ha  :  |a|  <  δ)  (hb  :  |b|  <  δ)

#check  my_lemma  a  b  δ
#check  my_lemma  a  b  δ  h₀  h₁
#check  my_lemma  a  b  δ  h₀  h₁  ha  hb

end 
```

您也已经看到，在 Lean 中，当量词可以从后续的假设中推断出来时，通常使用花括号来使量词变量隐式。当我们这样做时，我们只需将引理应用于假设，而不必提及对象。

```py
theorem  my_lemma2  :  ∀  {x  y  ε  :  ℝ},  0  <  ε  →  ε  ≤  1  →  |x|  <  ε  →  |y|  <  ε  →  |x  *  y|  <  ε  :=
  sorry

section
variable  (a  b  δ  :  ℝ)
variable  (h₀  :  0  <  δ)  (h₁  :  δ  ≤  1)
variable  (ha  :  |a|  <  δ)  (hb  :  |b|  <  δ)

#check  my_lemma2  h₀  h₁  ha  hb

end 
```

在这个阶段，您也知道，如果您使用 `apply` 策略将 `my_lemma` 应用到形式为 `|a * b| < δ` 的目标上，您将剩下需要证明每个假设的新目标。

要证明这样的陈述，使用 `intro` 策略。看看它在以下示例中的表现：

```py
theorem  my_lemma3  :
  ∀  {x  y  ε  :  ℝ},  0  <  ε  →  ε  ≤  1  →  |x|  <  ε  →  |y|  <  ε  →  |x  *  y|  <  ε  :=  by
  intro  x  y  ε  epos  ele1  xlt  ylt
  sorry 
```

我们可以为全称量词变量使用任何我们想要的名称；它们不必是 `x`、`y` 和 `ε`。注意，即使它们被标记为隐式，我们也必须引入这些变量：使它们隐式意味着在编写使用 `my_lemma` 的表达式时我们可以省略它们，但它们仍然是我们正在证明的陈述的一个基本部分。在 `intro` 命令之后，目标是如果在冒号之前列出了所有变量和假设，它将是什么，就像我们在上一节中所做的那样。一会儿我们将看到为什么有时需要在证明开始后引入变量和假设。

为了帮助你证明引理，我们将从以下步骤开始：

```py
theorem  my_lemma4  :
  ∀  {x  y  ε  :  ℝ},  0  <  ε  →  ε  ≤  1  →  |x|  <  ε  →  |y|  <  ε  →  |x  *  y|  <  ε  :=  by
  intro  x  y  ε  epos  ele1  xlt  ylt
  calc
  |x  *  y|  =  |x|  *  |y|  :=  sorry
  _  ≤  |x|  *  ε  :=  sorry
  _  <  1  *  ε  :=  sorry
  _  =  ε  :=  sorry 
```

使用定理 `abs_mul`、`mul_le_mul`、`abs_nonneg`、`mul_lt_mul_right` 和 `one_mul` 完成证明。记住，你可以使用 Ctrl-space 完成提示（或在 Mac 上使用 Cmd-space 完成提示）找到这样的定理。还要记住，你可以使用 `.mp` 和 `.mpr` 或 `.1` 和 `.2` 来提取双向条件语句的两个方向。

全称量词通常隐藏在定义中，当需要时 Lean 会展开定义来暴露它们。例如，让我们定义两个谓词，`FnUb f a` 和 `FnLb f a`，其中 `f` 是从实数到实数的函数，而 `a` 是一个实数。第一个谓词表示 `a` 是 `f` 的值的上界，第二个谓词表示 `a` 是 `f` 的值的下界。

```py
def  FnUb  (f  :  ℝ  →  ℝ)  (a  :  ℝ)  :  Prop  :=
  ∀  x,  f  x  ≤  a

def  FnLb  (f  :  ℝ  →  ℝ)  (a  :  ℝ)  :  Prop  :=
  ∀  x,  a  ≤  f  x 
```

在下一个例子中，`fun x ↦ f x + g x` 是将 `x` 映射到 `f x + g x` 的函数。从表达式 `f x + g x` 到这个函数的过程在类型理论中被称为 lambda 抽象。

```py
example  (hfa  :  FnUb  f  a)  (hgb  :  FnUb  g  b)  :  FnUb  (fun  x  ↦  f  x  +  g  x)  (a  +  b)  :=  by
  intro  x
  dsimp
  apply  add_le_add
  apply  hfa
  apply  hgb 
```

将 `intro` 应用到目标 `FnUb (fun x ↦ f x + g x) (a + b)` 迫使 Lean 展开定义 `FnUb` 并引入 `x` 作为全称量词。目标变为 `(fun (x : ℝ) ↦ f x + g x) x ≤ a + b`。但是将 `(fun x ↦ f x + g x)` 应用到 `x` 应该得到 `f x + g x`，而 `dsimp` 命令执行了这种简化。（“d”代表“定义性的。”）你可以删除那个命令，证明仍然有效；Lean 无论如何都必须执行那个收缩来理解下一个 `apply`。`dsimp` 命令只是使目标更易于阅读，并帮助我们确定下一步该做什么。另一个选择是使用 `change` 策略，通过编写 `change f x + g x ≤ a + b` 来实现。这有助于使证明更易于阅读，并让你对目标如何转换有更多的控制。

证明的其余部分是常规的。最后两个 `apply` 命令迫使 Lean 在假设中展开 `FnUb` 的定义。尝试进行类似的证明：

```py
example  (hfa  :  FnLb  f  a)  (hgb  :  FnLb  g  b)  :  FnLb  (fun  x  ↦  f  x  +  g  x)  (a  +  b)  :=
  sorry

example  (nnf  :  FnLb  f  0)  (nng  :  FnLb  g  0)  :  FnLb  (fun  x  ↦  f  x  *  g  x)  0  :=
  sorry

example  (hfa  :  FnUb  f  a)  (hgb  :  FnUb  g  b)  (nng  :  FnLb  g  0)  (nna  :  0  ≤  a)  :
  FnUb  (fun  x  ↦  f  x  *  g  x)  (a  *  b)  :=
  sorry 
```

尽管我们已经为从实数到实数的函数定义了 `FnUb` 和 `FnLb`，但你应该认识到定义和证明要更通用。这些定义对于任何在值域上有序概念的类型之间的函数都是有意义的。检查定理 `add_le_add` 的类型显示，它适用于任何“有序加法交换幺半群”的结构；现在不需要知道这意味着什么，但值得知道自然数、整数、有理数和实数都是实例。因此，如果我们在这个普遍性级别上证明定理 `fnUb_add`，它将适用于所有这些实例。

```py
variable  {α  :  Type*}  {R  :  Type*}  [AddCommMonoid  R]  [PartialOrder  R]  [IsOrderedCancelAddMonoid  R]

#check  add_le_add

def  FnUb'  (f  :  α  →  R)  (a  :  R)  :  Prop  :=
  ∀  x,  f  x  ≤  a

theorem  fnUb_add  {f  g  :  α  →  R}  {a  b  :  R}  (hfa  :  FnUb'  f  a)  (hgb  :  FnUb'  g  b)  :
  FnUb'  (fun  x  ↦  f  x  +  g  x)  (a  +  b)  :=  fun  x  ↦  add_le_add  (hfa  x)  (hgb  x) 
```

你已经在 第 2.2 节 中看到了这样的方括号，尽管我们还没有解释它们的意义。为了具体化，我们将在大多数例子中坚持使用实数，但值得知道 Mathlib 包含定义和定理，它们在高度普遍的层面上工作。

作为另一个隐藏的全称量词的例子，Mathlib 定义了一个谓词 `Monotone`，它表示函数在其参数上是非递减的：

```py
example  (f  :  ℝ  →  ℝ)  (h  :  Monotone  f)  :  ∀  {a  b},  a  ≤  b  →  f  a  ≤  f  b  :=
  @h 
```

谓词 `Monotone f` 被定义为冒号后面的确切表达式。我们需要在 `h` 前面放置 `@` 符号，因为如果不这样做，Lean 会扩展 `h` 的隐含参数并插入占位符。

证明关于单调性的陈述涉及使用 `intro` 引入两个变量，例如，`a` 和 `b`，以及假设 `a ≤ b`。要 *使用* 单调性假设，你可以将其应用于合适的论点和假设，然后将结果表达式应用于目标。或者，你可以将其应用于目标，让 Lean 通过显示剩余的假设作为新的子目标来帮助你反向工作。

```py
example  (mf  :  Monotone  f)  (mg  :  Monotone  g)  :  Monotone  fun  x  ↦  f  x  +  g  x  :=  by
  intro  a  b  aleb
  apply  add_le_add
  apply  mf  aleb
  apply  mg  aleb 
```

当证明如此简短时，通常更方便给出一个证明项。为了描述一个临时引入对象 `a` 和 `b` 以及假设 `aleb` 的证明，Lean 使用记号 `fun a b aleb ↦ ...`。这与像 `fun x ↦ x²` 这样的表达式通过临时命名一个对象 `x` 然后用它来描述一个值的方式来描述一个函数类似。因此，前一个证明中的 `intro` 命令对应于下一个证明项中的 lambda 抽象。然后的 `apply` 命令对应于构建定理对其参数的应用。

```py
example  (mf  :  Monotone  f)  (mg  :  Monotone  g)  :  Monotone  fun  x  ↦  f  x  +  g  x  :=
  fun  a  b  aleb  ↦  add_le_add  (mf  aleb)  (mg  aleb) 
```

这里有一个有用的技巧：如果你开始使用下划线 `_` 来编写证明项 `fun a b aleb ↦ _`，在表达式应该放置的地方，Lean 会标记一个错误，表明它无法猜测该表达式的值。如果你在 VS Code 中的 Lean Goal 窗口检查或悬停在波浪形错误标记上，Lean 会显示剩余表达式必须解决的目標。

尝试使用策略或证明项来证明这些：

```py
example  {c  :  ℝ}  (mf  :  Monotone  f)  (nnc  :  0  ≤  c)  :  Monotone  fun  x  ↦  c  *  f  x  :=
  sorry

example  (mf  :  Monotone  f)  (mg  :  Monotone  g)  :  Monotone  fun  x  ↦  f  (g  x)  :=
  sorry 
```

这里有一些更多的例子。从 $\Bbb R$ 到 $\Bbb R$ 的函数 $f$ 被称为 *偶函数*，如果对于每个 $x$，$f(-x) = f(x)$；如果对于每个 $x$，$f(-x) = -f(x)$，则称为 *奇函数*。以下示例正式定义了这两个概念，并建立了一个关于它们的命题。你可以完成其他命题的证明。

```py
def  FnEven  (f  :  ℝ  →  ℝ)  :  Prop  :=
  ∀  x,  f  x  =  f  (-x)

def  FnOdd  (f  :  ℝ  →  ℝ)  :  Prop  :=
  ∀  x,  f  x  =  -f  (-x)

example  (ef  :  FnEven  f)  (eg  :  FnEven  g)  :  FnEven  fun  x  ↦  f  x  +  g  x  :=  by
  intro  x
  calc
  (fun  x  ↦  f  x  +  g  x)  x  =  f  x  +  g  x  :=  rfl
  _  =  f  (-x)  +  g  (-x)  :=  by  rw  [ef,  eg]

example  (of  :  FnOdd  f)  (og  :  FnOdd  g)  :  FnEven  fun  x  ↦  f  x  *  g  x  :=  by
  sorry

example  (ef  :  FnEven  f)  (og  :  FnOdd  g)  :  FnOdd  fun  x  ↦  f  x  *  g  x  :=  by
  sorry

example  (ef  :  FnEven  f)  (og  :  FnOdd  g)  :  FnEven  fun  x  ↦  f  (g  x)  :=  by
  sorry 
```

第一个证明可以使用 `dsimp` 或 `change` 来缩短，以消除 lambda 抽象。但你可以检查，除非我们明确地消除 lambda 抽象，否则随后的 `rw` 不会工作，因为否则它无法在表达式中找到 `f x` 和 `g x` 的模式。与一些其他策略相反，`rw` 在句法层面上操作，它不会为你展开定义或应用简化（它有一个名为 `erw` 的变体，在这个方向上尝试得稍微努力一些，但不是很多）。

一旦你知道如何找到它们，你可以在任何地方找到隐含的全称量词。

Mathlib 包含了一个用于操作集合的良好库。回想一下，Lean 不使用基于集合论的公理系统，因此这里的“集合”一词具有其平凡的意义，即某些给定类型 `α` 的数学对象的集合。如果 `x` 的类型是 `α`，而 `s` 的类型是 `Set α`，那么 `x ∈ s` 是一个命题，它断言 `x` 是 `s` 的一个元素。如果 `y` 具有某种不同的类型 `β`，那么表达式 `y ∈ s` 就没有意义。这里的“没有意义”是指“没有类型，因此 Lean 不接受它作为一个有效的语句”。这与例如 Zermelo-Fraenkel 集合论形成对比，在 Zermelo-Fraenkel 集合论中，对于每一个数学对象 `a` 和 `b`，`a ∈ b` 都是一个有效的语句。例如，在 ZF 中，“sin ∈ cos”是一个有效的语句。集合论基础的这一缺陷是重要的动机，即不在旨在通过检测无意义表达式来协助我们的证明辅助工具中使用它。在 Lean 中，`sin`的类型是 `ℝ → ℝ`，而`cos`的类型是 `ℝ → ℝ`，这并不等于 `Set (ℝ → ℝ)`，即使展开定义之后也是如此，因此语句“sin ∈ cos”没有意义。人们也可以使用 Lean 来研究集合论本身。例如，连续假设与 Zermelo-Fraenkel 公理的独立性已经在 Lean 中形式化。但这样的集合论元理论完全超出了本书的范围。

如果 `s` 和 `t` 的类型都是 `Set α`，那么子集关系 `s ⊆ t` 被定义为意味着 `∀ {x : α}, x ∈ s → x ∈ t`。量词中的变量被标记为隐式，这样给定 `h : s ⊆ t` 和 `h' : x ∈ s`，我们可以将 `h h'` 写作 `x ∈ t` 的理由。以下示例提供了一个策略证明和一个证明项，以证明子集关系的自反性，并要求你为传递性做同样的工作。

```py
variable  {α  :  Type*}  (r  s  t  :  Set  α)

example  :  s  ⊆  s  :=  by
  intro  x  xs
  exact  xs

theorem  Subset.refl  :  s  ⊆  s  :=  fun  x  xs  ↦  xs

theorem  Subset.trans  :  r  ⊆  s  →  s  ⊆  t  →  r  ⊆  t  :=  by
  sorry 
```

正如我们为函数定义了 `FnUb` 一样，我们可以定义 `SetUb s a` 来表示 `a` 是集合 `s` 的上界，假设 `s` 是具有某种类型元素的集合，并且与它相关联一个顺序。在下一个例子中，我们要求你证明如果 `a` 是 `s` 的一个界且 `a ≤ b`，那么 `b` 也是 `s` 的一个界。

```py
variable  {α  :  Type*}  [PartialOrder  α]
variable  (s  :  Set  α)  (a  b  :  α)

def  SetUb  (s  :  Set  α)  (a  :  α)  :=
  ∀  x,  x  ∈  s  →  x  ≤  a

example  (h  :  SetUb  s  a)  (h'  :  a  ≤  b)  :  SetUb  s  b  :=
  sorry 
```

我们以一个最后的、非常重要的例子结束本节。如果一个函数 $f$ 对于每一个 $x_1$ 和 $x_2$，如果 $f(x_1) = f(x_2)$ 则 $x_1 = x_2$，那么称这个函数为 *单射*。Mathlib 使用 `Function.Injective f` 并隐含地定义 `x₁` 和 `x₂`。下一个例子展示了在实数上，任何加常数的函数都是单射的。然后我们要求你使用例子中的引理名称作为灵感来源，证明乘以非零常数也是单射的。回忆一下，在猜测引理名称的开始后，你应该使用 Ctrl-space 完成提示。

```py
open  Function

example  (c  :  ℝ)  :  Injective  fun  x  ↦  x  +  c  :=  by
  intro  x₁  x₂  h'
  exact  (add_left_inj  c).mp  h'

example  {c  :  ℝ}  (h  :  c  ≠  0)  :  Injective  fun  x  ↦  c  *  x  :=  by
  sorry 
```

最后，证明两个单射函数的复合也是单射：

```py
variable  {α  :  Type*}  {β  :  Type*}  {γ  :  Type*}
variable  {g  :  β  →  γ}  {f  :  α  →  β}

example  (injg  :  Injective  g)  (injf  :  Injective  f)  :  Injective  fun  x  ↦  g  (f  x)  :=  by
  sorry 
```  ## 3.2\. 存在量词

存在量词，在 VS Code 中可以输入为 `\ex`，用来表示“存在”这个短语。在 Lean 中的形式表达式 `∃ x : ℝ, 2 < x ∧ x < 3` 表示存在一个实数在 2 和 3 之间。（我们将在 第 3.4 节 中讨论合取符号 `∧`。）证明此类陈述的规范方法是展示一个实数并证明它具有所述的性质。数字 2.5，当我们不能从上下文中推断出我们指的是实数时，可以输入为 `5 / 2` 或 `(5 : ℝ) / 2`，它具有所需性质，而 `norm_num` 战术可以证明它符合描述。

我们有几种方法可以将信息组合起来。给定一个以存在量词开始的目標，`use` 战术用于提供对象，留下证明性质的目標。

```py
example  :  ∃  x  :  ℝ,  2  <  x  ∧  x  <  3  :=  by
  use  5  /  2
  norm_num 
```

你可以给出 `use` 战术的证明以及数据：

```py
example  :  ∃  x  :  ℝ,  2  <  x  ∧  x  <  3  :=  by
  have  h1  :  2  <  (5  :  ℝ)  /  2  :=  by  norm_num
  have  h2  :  (5  :  ℝ)  /  2  <  3  :=  by  norm_num
  use  5  /  2,  h1,  h2 
```

实际上，`use` 战术会自动尝试使用可用的假设。

```py
example  :  ∃  x  :  ℝ,  2  <  x  ∧  x  <  3  :=  by
  have  h  :  2  <  (5  :  ℝ)  /  2  ∧  (5  :  ℝ)  /  2  <  3  :=  by  norm_num
  use  5  /  2 
```

或者，我们可以使用 Lean 的 *匿名构造函数* 符号来构造存在量词的证明。

```py
example  :  ∃  x  :  ℝ,  2  <  x  ∧  x  <  3  :=
  have  h  :  2  <  (5  :  ℝ)  /  2  ∧  (5  :  ℝ)  /  2  <  3  :=  by  norm_num
  ⟨5  /  2,  h⟩ 
```

注意，这里没有 `by`；我们在这里给出一个明确的证明项。左右尖括号，分别可以用 `\<` 和 `\>` 输入，告诉 Lean 使用适合当前目标的任何构造来组合给定的数据。我们可以在不首先进入战术模式的情况下使用这种符号：

```py
example  :  ∃  x  :  ℝ,  2  <  x  ∧  x  <  3  :=
  ⟨5  /  2,  by  norm_num⟩ 
```

因此，我们现在知道了如何 *证明* 存在语句。但我们如何 *使用* 它呢？如果我们知道存在具有某种性质的物体，我们应该能够给它一个任意的名字并对它进行推理。例如，记住上一节中的谓词 `FnUb f a` 和 `FnLb f a`，它们分别表示 `a` 是 `f` 的上界或下界。我们可以使用存在量词来说明“`f` 有界”，而不必指定界限：

```py
def  FnUb  (f  :  ℝ  →  ℝ)  (a  :  ℝ)  :  Prop  :=
  ∀  x,  f  x  ≤  a

def  FnLb  (f  :  ℝ  →  ℝ)  (a  :  ℝ)  :  Prop  :=
  ∀  x,  a  ≤  f  x

def  FnHasUb  (f  :  ℝ  →  ℝ)  :=
  ∃  a,  FnUb  f  a

def  FnHasLb  (f  :  ℝ  →  ℝ)  :=
  ∃  a,  FnLb  f  a 
```

我们可以使用上一节中的定理 `FnUb_add` 来证明，如果 `f` 和 `g` 有上界，那么 `fun x ↦ f x + g x` 也有上界。

```py
variable  {f  g  :  ℝ  →  ℝ}

example  (ubf  :  FnHasUb  f)  (ubg  :  FnHasUb  g)  :  FnHasUb  fun  x  ↦  f  x  +  g  x  :=  by
  rcases  ubf  with  ⟨a,  ubfa⟩
  rcases  ubg  with  ⟨b,  ubgb⟩
  use  a  +  b
  apply  fnUb_add  ubfa  ubgb 
```

`rcases` 策略解包存在量词中的信息。像 `⟨a, ubfa⟩` 这样的注释，用与匿名构造函数相同的尖括号书写，被称为 *模式*，它们描述了我们解包主参数时预期找到的信息。给定存在上界的假设 `ubf`，`rcases ubf with ⟨a, ubfa⟩` 将一个新的变量 `a`（上界）添加到上下文中，以及具有给定属性的假设 `ubfa`。目标保持不变；*改变*的是，我们现在可以使用新的对象和新的假设来证明目标。这是数学中常见的推理方法：我们解包由某些假设断言或暗示存在的对象，然后使用它来建立其他事物的存在。

尝试使用这种方法来建立以下内容。你可能发现将上一节中的某些示例转换为命名定理很有用，就像我们对待 `fn_ub_add` 一样，或者你可以直接将参数插入到证明中。

```py
example  (lbf  :  FnHasLb  f)  (lbg  :  FnHasLb  g)  :  FnHasLb  fun  x  ↦  f  x  +  g  x  :=  by
  sorry

example  {c  :  ℝ}  (ubf  :  FnHasUb  f)  (h  :  c  ≥  0)  :  FnHasUb  fun  x  ↦  c  *  f  x  :=  by
  sorry 
```

`rcases` 中的“r”代表“递归”，因为它允许我们使用任意复杂的模式来解包嵌套数据。`rintro` 策略是 `intro` 和 `rcases` 的组合：

```py
example  :  FnHasUb  f  →  FnHasUb  g  →  FnHasUb  fun  x  ↦  f  x  +  g  x  :=  by
  rintro  ⟨a,  ubfa⟩  ⟨b,  ubgb⟩
  exact  ⟨a  +  b,  fnUb_add  ubfa  ubgb⟩ 
```

事实上，Lean 还支持在表达式和证明项中使用模式匹配函数：

```py
example  :  FnHasUb  f  →  FnHasUb  g  →  FnHasUb  fun  x  ↦  f  x  +  g  x  :=
  fun  ⟨a,  ubfa⟩  ⟨b,  ubgb⟩  ↦  ⟨a  +  b,  fnUb_add  ubfa  ubgb⟩ 
```

在假设中解包信息这项任务非常重要，以至于 Lean 和 Mathlib 提供了多种方法来完成它。例如，`obtain` 策略提供了提示性语法：

```py
example  (ubf  :  FnHasUb  f)  (ubg  :  FnHasUb  g)  :  FnHasUb  fun  x  ↦  f  x  +  g  x  :=  by
  obtain  ⟨a,  ubfa⟩  :=  ubf
  obtain  ⟨b,  ubgb⟩  :=  ubg
  exact  ⟨a  +  b,  fnUb_add  ubfa  ubgb⟩ 
```

将第一个 `obtain` 指令视为将 `ubf` 的“内容”与给定的模式进行匹配，并将组件分配给命名变量。`rcases` 和 `obtain` 被说成是“破坏”它们的参数。

Lean 还支持与其它函数式编程语言中使用的语法类似的语法：

```py
example  (ubf  :  FnHasUb  f)  (ubg  :  FnHasUb  g)  :  FnHasUb  fun  x  ↦  f  x  +  g  x  :=  by
  cases  ubf
  case  intro  a  ubfa  =>
  cases  ubg
  case  intro  b  ubgb  =>
  exact  ⟨a  +  b,  fnUb_add  ubfa  ubgb⟩

example  (ubf  :  FnHasUb  f)  (ubg  :  FnHasUb  g)  :  FnHasUb  fun  x  ↦  f  x  +  g  x  :=  by
  cases  ubf
  next  a  ubfa  =>
  cases  ubg
  next  b  ubgb  =>
  exact  ⟨a  +  b,  fnUb_add  ubfa  ubgb⟩

example  (ubf  :  FnHasUb  f)  (ubg  :  FnHasUb  g)  :  FnHasUb  fun  x  ↦  f  x  +  g  x  :=  by
  match  ubf,  ubg  with
  |  ⟨a,  ubfa⟩,  ⟨b,  ubgb⟩  =>
  exact  ⟨a  +  b,  fnUb_add  ubfa  ubgb⟩

example  (ubf  :  FnHasUb  f)  (ubg  :  FnHasUb  g)  :  FnHasUb  fun  x  ↦  f  x  +  g  x  :=
  match  ubf,  ubg  with
  |  ⟨a,  ubfa⟩,  ⟨b,  ubgb⟩  =>
  ⟨a  +  b,  fnUb_add  ubfa  ubgb⟩ 
```

在第一个例子中，如果你将光标放在 `cases ubf` 之后，你会看到这个策略产生了一个单一的目标，Lean 将其标记为 `intro`。（所选择的特定名称来自构建存在性陈述证明的公理原语的内部名称。）然后 `case` 策略命名了组件。第二个例子与第一个类似，只是使用 `next` 而不是 `case` 意味着你不必提到 `intro`。在最后两个例子中，单词 `match` 突出了我们在这里所做的是计算机科学家所说的“模式匹配”。请注意，第三个证明以 `by` 开头，之后 `match` 的策略版本期望箭头右侧有一个策略证明。最后一个例子是一个证明项：没有看到任何策略。

在本书的其余部分，我们将坚持使用 `rcases`、`rintro` 和 `obtain` 作为使用存在量词的首选方式。但看看替代语法也无妨，尤其是如果你有可能与计算机科学家为伍的话。

为了说明 `rcases` 可以使用的一种方式，我们证明了一个古老的数学难题：如果两个整数 `x` 和 `y` 可以分别表示为两个平方的和，那么它们的乘积 `x * y` 也可以。实际上，这个陈述对于任何交换环都成立，而不仅仅是整数。在下一个例子中，`rcases` 一次展开两个存在量词。然后我们提供一个列表，其中包含将 `x * y` 表示为平方和所需的神奇值，并将其作为 `use` 语句的参数，并使用 `ring` 来验证它们是否有效。

```py
variable  {α  :  Type*}  [CommRing  α]

def  SumOfSquares  (x  :  α)  :=
  ∃  a  b,  x  =  a  ^  2  +  b  ^  2

theorem  sumOfSquares_mul  {x  y  :  α}  (sosx  :  SumOfSquares  x)  (sosy  :  SumOfSquares  y)  :
  SumOfSquares  (x  *  y)  :=  by
  rcases  sosx  with  ⟨a,  b,  xeq⟩
  rcases  sosy  with  ⟨c,  d,  yeq⟩
  rw  [xeq,  yeq]
  use  a  *  c  -  b  *  d,  a  *  d  +  b  *  c
  ring 
```

这个证明并没有提供很多洞见，但这里有一种激发它的方法。一个**高斯整数**是形如 $a + bi$ 的数，其中 $a$ 和 $b$ 是整数，$i = \sqrt{-1}$。根据定义，高斯整数 $a + bi$ 的**范数**是 $a² + b²$。因此，高斯整数的范数是平方和，任何平方和都可以用这种方式表示。上述定理反映了高斯整数乘积的范数是它们范数的乘积这一事实：如果 $x$ 是 $a + bi$ 的范数，$y$ 是 $c + di$ 的范数，那么 $xy$ 就是 $(a + bi) (c + di)$ 的范数。我们神秘的证明说明了这样一个事实：最容易形式化的证明并不总是最清晰的。在第 7.3 节中，我们将提供定义高斯整数并使用它们提供另一种证明的方法。

在存在量词内部展开方程并使用它来重写目标表达式的模式经常出现，以至于 `rcases` 策略提供了一个缩写：如果你用关键字 `rfl` 代替新标识符，`rcases` 会自动进行重写（这个技巧与模式匹配的 lambda 表达式不兼容）。

```py
theorem  sumOfSquares_mul'  {x  y  :  α}  (sosx  :  SumOfSquares  x)  (sosy  :  SumOfSquares  y)  :
  SumOfSquares  (x  *  y)  :=  by
  rcases  sosx  with  ⟨a,  b,  rfl⟩
  rcases  sosy  with  ⟨c,  d,  rfl⟩
  use  a  *  c  -  b  *  d,  a  *  d  +  b  *  c
  ring 
```

就像全称量词一样，如果你知道如何识别，你可以在任何地方找到隐藏的存在量词。例如，可除性隐含地是一个“存在”陈述。

```py
example  (divab  :  a  ∣  b)  (divbc  :  b  ∣  c)  :  a  ∣  c  :=  by
  rcases  divab  with  ⟨d,  beq⟩
  rcases  divbc  with  ⟨e,  ceq⟩
  rw  [ceq,  beq]
  use  d  *  e;  ring 
```

再次强调，这为使用 `rcases` 和 `rfl` 提供了一个很好的环境。在上面的证明中尝试一下。感觉非常好！

然后尝试证明以下内容：

```py
example  (divab  :  a  ∣  b)  (divac  :  a  ∣  c)  :  a  ∣  b  +  c  :=  by
  sorry 
```

对于另一个重要的例子，一个函数 $f : \alpha \to \beta$ 被称为 *外射*，如果对于域 $\alpha$ 中的每个 $y$，在陪域 $\beta$ 中都有一个 $x$，使得 $f(x) = y$。注意，这个陈述包括全称量词和存在量词，这也解释了为什么下一个例子会同时使用 `intro` 和 `use`。

```py
example  {c  :  ℝ}  :  Surjective  fun  x  ↦  x  +  c  :=  by
  intro  x
  use  x  -  c
  dsimp;  ring 
```

尝试使用定理 `mul_div_cancel₀` 自己做这个例子。

```py
example  {c  :  ℝ}  (h  :  c  ≠  0)  :  Surjective  fun  x  ↦  c  *  x  :=  by
  sorry 
```

到目前为止，值得提到的是，有一个策略 `field_simp`，它通常会以有用的方式消除分母。它可以与 `ring` 策略一起使用。

```py
example  (x  y  :  ℝ)  (h  :  x  -  y  ≠  0)  :  (x  ^  2  -  y  ^  2)  /  (x  -  y)  =  x  +  y  :=  by
  field_simp  [h]
  ring 
```

下一个例子通过应用到一个合适的值来使用外射性假设。请注意，你可以用 `rcases` 与任何表达式一起使用，而不仅仅是假设。

```py
example  {f  :  ℝ  →  ℝ}  (h  :  Surjective  f)  :  ∃  x,  f  x  ^  2  =  4  :=  by
  rcases  h  2  with  ⟨x,  hx⟩
  use  x
  rw  [hx]
  norm_num 
```

尝试使用这些方法来证明外射函数的合成仍然是外射的。

```py
variable  {α  :  Type*}  {β  :  Type*}  {γ  :  Type*}
variable  {g  :  β  →  γ}  {f  :  α  →  β}

example  (surjg  :  Surjective  g)  (surjf  :  Surjective  f)  :  Surjective  fun  x  ↦  g  (f  x)  :=  by
  sorry 
```  ## 3.3\. 否定

符号 `¬` 用于表示否定，所以 `¬ x < y` 表示 `x` 不小于 `y`，`¬ x = y`（或等价地，`x ≠ y`）表示 `x` 不等于 `y`，而 `¬ ∃ z, x < z ∧ z < y` 表示不存在一个 `z` 在 `x` 和 `y` 之间。在 Lean 中，符号 `¬ A` 缩写为 `A → False`，你可以将其视为 `A` 导致矛盾。实际上，这意味着你已经知道如何处理否定：你可以通过引入一个假设 `h : A` 并证明 `False` 来证明 `¬ A`，如果你有 `h : ¬ A` 和 `h' : A`，那么将 `h` 应用到 `h'` 上会产生 `False`。

为了说明，考虑严格顺序的不自反原理 `lt_irrefl`，它表示对于每个 `a`，我们有 `¬ a < a`。不对称原理 `lt_asymm` 表示 `a < b → ¬ b < a`。让我们证明 `lt_asymm` 可以从 `lt_irrefl` 推导出来。

```py
example  (h  :  a  <  b)  :  ¬b  <  a  :=  by
  intro  h'
  have  :  a  <  a  :=  lt_trans  h  h'
  apply  lt_irrefl  a  this 
```

这个例子介绍了一些新的技巧。首先，当你使用 `have` 而不提供标签时，Lean 使用名称 `this`，提供了一种方便的回指方式。因为证明非常简短，所以我们提供了一个显式的证明项。但在这个证明中，你应该真正关注的是 `intro` 策略的结果，它留下一个 `False` 的目标，以及我们最终通过将 `lt_irrefl` 应用到 `a < a` 的证明上来证明 `False` 的事实。

这里是另一个例子，它使用了上一节中定义的谓词 `FnHasUb`，它表示一个函数有一个上界。

```py
example  (h  :  ∀  a,  ∃  x,  f  x  >  a)  :  ¬FnHasUb  f  :=  by
  intro  fnub
  rcases  fnub  with  ⟨a,  fnuba⟩
  rcases  h  a  with  ⟨x,  hx⟩
  have  :  f  x  ≤  a  :=  fnuba  x
  linarith 
```

记住，当目标可以从上下文中的线性方程和不等式推导出来时，使用 `linarith` 通常很方便。

尝试以类似的方式证明这些：

```py
example  (h  :  ∀  a,  ∃  x,  f  x  <  a)  :  ¬FnHasLb  f  :=
  sorry

example  :  ¬FnHasUb  fun  x  ↦  x  :=
  sorry 
```

Mathlib 提供了一些有用的定理，用于关联顺序和否定：

```py
#check  (not_le_of_gt  :  a  >  b  →  ¬a  ≤  b)
#check  (not_lt_of_ge  :  a  ≥  b  →  ¬a  <  b)
#check  (lt_of_not_ge  :  ¬a  ≥  b  →  a  <  b)
#check  (le_of_not_gt  :  ¬a  >  b  →  a  ≤  b) 
```

回忆一下谓词`Monotone f`，它表示`f`是非递减的。使用刚刚列举的一些定理来证明以下结论：

```py
example  (h  :  Monotone  f)  (h'  :  f  a  <  f  b)  :  a  <  b  :=  by
  sorry

example  (h  :  a  ≤  b)  (h'  :  f  b  <  f  a)  :  ¬Monotone  f  :=  by
  sorry 
```

我们可以证明，如果我们将最后一段代码中的`<`替换为`≤`，那么第一个例子无法被证明。注意，我们可以通过给出反例来证明全称量化语句的否定。完成证明。

```py
example  :  ¬∀  {f  :  ℝ  →  ℝ},  Monotone  f  →  ∀  {a  b},  f  a  ≤  f  b  →  a  ≤  b  :=  by
  intro  h
  let  f  :=  fun  x  :  ℝ  ↦  (0  :  ℝ)
  have  monof  :  Monotone  f  :=  by  sorry
  have  h'  :  f  1  ≤  f  0  :=  le_refl  _
  sorry 
```

本例介绍了`let`策略，该策略向上下文中添加一个*局部定义*。如果你将光标放在`let`命令之后，在目标窗口中你会看到定义`f : ℝ → ℝ := fun x ↦ 0`已经被添加到上下文中。当 Lean 需要时，它会展开`f`的定义。特别是，当我们使用`le_refl`证明`f 1 ≤ f 0`时，Lean 会将`f 1`和`f 0`简化为`0`。

使用`le_of_not_gt`来证明以下结论：

```py
example  (x  :  ℝ)  (h  :  ∀  ε  >  0,  x  <  ε)  :  x  ≤  0  :=  by
  sorry 
```

在我们刚刚做的许多证明中隐含的事实是，如果`P`是任何属性，说没有任何具有属性`P`的东西等同于说所有东西都没有属性`P`，而说不是所有东西都有属性`P`等同于说有些东西没有属性`P`。换句话说，以下四个蕴含都是有效的（但其中之一不能使用我们之前解释的方法来证明）：

```py
variable  {α  :  Type*}  (P  :  α  →  Prop)  (Q  :  Prop)

example  (h  :  ¬∃  x,  P  x)  :  ∀  x,  ¬P  x  :=  by
  sorry

example  (h  :  ∀  x,  ¬P  x)  :  ¬∃  x,  P  x  :=  by
  sorry

example  (h  :  ¬∀  x,  P  x)  :  ∃  x,  ¬P  x  :=  by
  sorry

example  (h  :  ∃  x,  ¬P  x)  :  ¬∀  x,  P  x  :=  by
  sorry 
```

第一、第二和第四个结论使用你已见过的方法证明起来很简单。我们鼓励你尝试一下。然而，第三个结论比较困难，因为它从其不存在是矛盾的这一事实中得出一个对象存在的结论。这是一个*经典*数学推理的例子。我们可以通过反证法证明第三个蕴含如下。

```py
example  (h  :  ¬∀  x,  P  x)  :  ∃  x,  ¬P  x  :=  by
  by_contra  h'
  apply  h
  intro  x
  show  P  x
  by_contra  h''
  exact  h'  ⟨x,  h''⟩ 
```

确保你理解这是如何工作的。`by_contra`策略允许我们通过假设`¬ Q`并推导出矛盾来证明目标`Q`。实际上，它等价于使用等价性`not_not : ¬ ¬ Q ↔ Q`。确认你可以使用`by_contra`证明这个等价性的正向方向，而反向方向则遵循普通的否定规则。

```py
example  (h  :  ¬¬Q)  :  Q  :=  by
  sorry

example  (h  :  Q)  :  ¬¬Q  :=  by
  sorry 
```

使用反证法来证明以下结论，这是我们之前证明的一个蕴含的逆命题。（提示：首先使用`intro`。）

```py
example  (h  :  ¬FnHasUb  f)  :  ∀  a,  ∃  x,  f  x  >  a  :=  by
  sorry 
```

处理带有否定前缀的复合语句通常很繁琐，将这样的语句替换为等价形式，其中否定已经被推向内部，是一种常见的数学模式。为了方便这一操作，Mathlib 提供了一个`push_neg`策略，它以这种方式重新表述目标。命令`push_neg at h`重新表述了假设`h`。

```py
example  (h  :  ¬∀  a,  ∃  x,  f  x  >  a)  :  FnHasUb  f  :=  by
  push_neg  at  h
  exact  h

example  (h  :  ¬FnHasUb  f)  :  ∀  a,  ∃  x,  f  x  >  a  :=  by
  dsimp  only  [FnHasUb,  FnUb]  at  h
  push_neg  at  h
  exact  h 
```

在第二个例子中，我们使用 `dsimp` 来展开 `FnHasUb` 和 `FnUb` 的定义。（我们需要使用 `dsimp` 而不是 `rw` 来展开 `FnUb`，因为它出现在量词的作用域内。）你可以验证在上述示例中，使用 `¬∃ x, P x` 和 `¬∀ x, P x`，`push_neg` 策略做了预期的事情。即使不知道如何使用合取符号，你也应该能够使用 `push_neg` 来证明以下内容：

```py
example  (h  :  ¬Monotone  f)  :  ∃  x  y,  x  ≤  y  ∧  f  y  <  f  x  :=  by
  sorry 
```

Mathlib 还有一个策略，`contrapose`，它将目标 `A → B` 转换为 `¬B → ¬A`。同样，给定一个从假设 `h : A` 证明 `B` 的目标，`contrapose h` 使你留下一个从假设 `¬B` 证明 `¬A` 的目标。使用 `contrapose!` 而不是 `contrapose` 将 `push_neg` 应用到目标和相关假设上。

```py
example  (h  :  ¬FnHasUb  f)  :  ∀  a,  ∃  x,  f  x  >  a  :=  by
  contrapose!  h
  exact  h

example  (x  :  ℝ)  (h  :  ∀  ε  >  0,  x  ≤  ε)  :  x  ≤  0  :=  by
  contrapose!  h
  use  x  /  2
  constructor  <;>  linarith 
```

我们还没有解释 `constructor` 命令或其后分号的用法，但我们将在这下一节中解释。

我们以 *ex falso* 原则结束本节，该原则表明任何东西都可以从矛盾中得出。在 Lean 中，这表示为 `False.elim`，它为任何命题 `P` 建立了 `False → P`。这看起来可能像是一个奇怪的原则，但它相当常见。我们经常通过分情况证明定理，有时我们可以表明其中一种情况是矛盾的。在这种情况下，我们需要断言矛盾建立了目标，这样我们就可以继续下一个目标。（我们将在 第 3.5 节 中看到推理的例子。）

Lean 提供了多种在达到矛盾后关闭目标的方法。

```py
example  (h  :  0  <  0)  :  a  >  37  :=  by
  exfalso
  apply  lt_irrefl  0  h

example  (h  :  0  <  0)  :  a  >  37  :=
  absurd  h  (lt_irrefl  0)

example  (h  :  0  <  0)  :  a  >  37  :=  by
  have  h'  :  ¬0  <  0  :=  lt_irrefl  0
  contradiction 
```

`exfalso` 策略将当前目标替换为证明 `False` 的目标。给定 `h : P` 和 `h' : ¬ P`，项 `absurd h h'` 建立了任何命题。最后，`contradiction` 策略试图通过在假设中找到一个矛盾来关闭目标，例如形式为 `h : P` 和 `h' : ¬ P` 的一对。当然，在这个例子中，`linarith` 也同样有效。## 3.4. 合取与双条件

你已经看到合取符号 `∧` 用于表示“和”。`constructor` 策略允许你通过先证明 `A` 然后证明 `B` 来证明形式为 `A ∧ B` 的陈述。

```py
example  {x  y  :  ℝ}  (h₀  :  x  ≤  y)  (h₁  :  ¬y  ≤  x)  :  x  ≤  y  ∧  x  ≠  y  :=  by
  constructor
  ·  assumption
  intro  h
  apply  h₁
  rw  [h] 
```

在这个例子中，`assumption` 策略告诉 Lean 寻找一个假设来解决问题。注意，最后的 `rw` 通过应用 `≤` 的自反性来完成目标。以下是通过匿名构造器尖括号执行先前示例的替代方法。第一种是先前证明的简洁证明术语版本，它在 `by` 关键字处进入策略模式。

```py
example  {x  y  :  ℝ}  (h₀  :  x  ≤  y)  (h₁  :  ¬y  ≤  x)  :  x  ≤  y  ∧  x  ≠  y  :=
  ⟨h₀,  fun  h  ↦  h₁  (by  rw  [h])⟩

example  {x  y  :  ℝ}  (h₀  :  x  ≤  y)  (h₁  :  ¬y  ≤  x)  :  x  ≤  y  ∧  x  ≠  y  :=
  have  h  :  x  ≠  y  :=  by
  contrapose!  h₁
  rw  [h₁]
  ⟨h₀,  h⟩ 
```

*使用* 合取而不是证明一个合取涉及展开两个部分的证明。你可以使用 `rcases` 策略，以及 `rintro` 或模式匹配的 `fun`，所有这些都在与存在量词使用类似的方式下进行。

```py
example  {x  y  :  ℝ}  (h  :  x  ≤  y  ∧  x  ≠  y)  :  ¬y  ≤  x  :=  by
  rcases  h  with  ⟨h₀,  h₁⟩
  contrapose!  h₁
  exact  le_antisymm  h₀  h₁

example  {x  y  :  ℝ}  :  x  ≤  y  ∧  x  ≠  y  →  ¬y  ≤  x  :=  by
  rintro  ⟨h₀,  h₁⟩  h'
  exact  h₁  (le_antisymm  h₀  h')

example  {x  y  :  ℝ}  :  x  ≤  y  ∧  x  ≠  y  →  ¬y  ≤  x  :=
  fun  ⟨h₀,  h₁⟩  h'  ↦  h₁  (le_antisymm  h₀  h') 
```

类似于 `obtain` 策略，还有一个模式匹配的 `have`：

```py
example  {x  y  :  ℝ}  (h  :  x  ≤  y  ∧  x  ≠  y)  :  ¬y  ≤  x  :=  by
  have  ⟨h₀,  h₁⟩  :=  h
  contrapose!  h₁
  exact  le_antisymm  h₀  h₁ 
```

与 `rcases` 不同，这里 `have` 策略将 `h` 留在上下文中。即使我们不会使用它们，我们再次有了计算机科学家的模式匹配语法：

```py
example  {x  y  :  ℝ}  (h  :  x  ≤  y  ∧  x  ≠  y)  :  ¬y  ≤  x  :=  by
  cases  h
  case  intro  h₀  h₁  =>
  contrapose!  h₁
  exact  le_antisymm  h₀  h₁

example  {x  y  :  ℝ}  (h  :  x  ≤  y  ∧  x  ≠  y)  :  ¬y  ≤  x  :=  by
  cases  h
  next  h₀  h₁  =>
  contrapose!  h₁
  exact  le_antisymm  h₀  h₁

example  {x  y  :  ℝ}  (h  :  x  ≤  y  ∧  x  ≠  y)  :  ¬y  ≤  x  :=  by
  match  h  with
  |  ⟨h₀,  h₁⟩  =>
  contrapose!  h₁
  exact  le_antisymm  h₀  h₁ 
```

与使用存在量词相比，你还可以通过编写 `h.left` 和 `h.right`，或者等价地，`h.1` 和 `h.2` 来提取假设 `h : A ∧ B` 的两个组成部分的证明。

```py
example  {x  y  :  ℝ}  (h  :  x  ≤  y  ∧  x  ≠  y)  :  ¬y  ≤  x  :=  by
  intro  h'
  apply  h.right
  exact  le_antisymm  h.left  h'

example  {x  y  :  ℝ}  (h  :  x  ≤  y  ∧  x  ≠  y)  :  ¬y  ≤  x  :=
  fun  h'  ↦  h.right  (le_antisymm  h.left  h') 
```

尝试使用这些技术来提出证明以下内容的各种方法：

```py
example  {m  n  :  ℕ}  (h  :  m  ∣  n  ∧  m  ≠  n)  :  m  ∣  n  ∧  ¬n  ∣  m  :=
  sorry 
```

你可以使用匿名构造函数、`rintro` 和 `rcases` 嵌套使用 `∃` 和 `∧`。

```py
example  :  ∃  x  :  ℝ,  2  <  x  ∧  x  <  4  :=
  ⟨5  /  2,  by  norm_num,  by  norm_num⟩

example  (x  y  :  ℝ)  :  (∃  z  :  ℝ,  x  <  z  ∧  z  <  y)  →  x  <  y  :=  by
  rintro  ⟨z,  xltz,  zlty⟩
  exact  lt_trans  xltz  zlty

example  (x  y  :  ℝ)  :  (∃  z  :  ℝ,  x  <  z  ∧  z  <  y)  →  x  <  y  :=
  fun  ⟨z,  xltz,  zlty⟩  ↦  lt_trans  xltz  zlty 
```

你也可以使用 `use` 策略：

```py
example  :  ∃  x  :  ℝ,  2  <  x  ∧  x  <  4  :=  by
  use  5  /  2
  constructor  <;>  norm_num

example  :  ∃  m  n  :  ℕ,  4  <  m  ∧  m  <  n  ∧  n  <  10  ∧  Nat.Prime  m  ∧  Nat.Prime  n  :=  by
  use  5
  use  7
  norm_num

example  {x  y  :  ℝ}  :  x  ≤  y  ∧  x  ≠  y  →  x  ≤  y  ∧  ¬y  ≤  x  :=  by
  rintro  ⟨h₀,  h₁⟩
  use  h₀
  exact  fun  h'  ↦  h₁  (le_antisymm  h₀  h') 
```

在第一个例子中，`constructor` 命令后面的分号告诉 Lean 对产生的两个目标都使用 `norm_num` 策略。

在 Lean 中，`A ↔ B` 并不是定义为 `(A → B) ∧ (B → A)`，但它本来可以是，并且其行为大致相同。你已经看到你可以为 `h : A ↔ B` 的两个方向编写 `h.mp` 和 `h.mpr` 或 `h.1` 和 `h.2`。你也可以使用 `cases` 和相关工具。为了证明一个“如果且仅如果”的陈述，你可以使用 `constructor` 或尖括号，就像你证明合取一样。

```py
example  {x  y  :  ℝ}  (h  :  x  ≤  y)  :  ¬y  ≤  x  ↔  x  ≠  y  :=  by
  constructor
  ·  contrapose!
  rintro  rfl
  rfl
  contrapose!
  exact  le_antisymm  h

example  {x  y  :  ℝ}  (h  :  x  ≤  y)  :  ¬y  ≤  x  ↔  x  ≠  y  :=
  ⟨fun  h₀  h₁  ↦  h₀  (by  rw  [h₁]),  fun  h₀  h₁  ↦  h₀  (le_antisymm  h  h₁)⟩ 
```

最后一个证明项难以理解。记住，在编写这样的表达式时，你可以使用下划线来查看 Lean 期望的内容。

尝试使用你刚刚看到的各种技术和工具来证明以下内容：

```py
example  {x  y  :  ℝ}  :  x  ≤  y  ∧  ¬y  ≤  x  ↔  x  ≤  y  ∧  x  ≠  y  :=
  sorry 
```

对于一个更有趣的练习，证明对于任何两个实数 `x` 和 `y`，`x² + y² = 0` 当且仅当 `x = 0` 和 `y = 0`。我们建议使用 `linarith`、`pow_two_nonneg` 和 `pow_eq_zero` 证明一个辅助引理。

```py
theorem  aux  {x  y  :  ℝ}  (h  :  x  ^  2  +  y  ^  2  =  0)  :  x  =  0  :=
  have  h'  :  x  ^  2  =  0  :=  by  sorry
  pow_eq_zero  h'

example  (x  y  :  ℝ)  :  x  ^  2  +  y  ^  2  =  0  ↔  x  =  0  ∧  y  =  0  :=
  sorry 
```

在 Lean 中，双条件具有双重生命。你可以将其视为合取，并分别使用其两部分。但 Lean 也知道它是一个命题之间的自反、对称和传递关系，你也可以使用它与 `calc` 和 `rw` 一起。将一个陈述重写为等价陈述通常很方便。在下一个例子中，我们使用 `abs_lt` 将形式为 `|x| < y` 的表达式替换为等价表达式 `- y < x ∧ x < y`，在下一个例子中，我们使用 `Nat.dvd_gcd_iff` 将形式为 `m ∣ Nat.gcd n k` 的表达式替换为等价表达式 `m ∣ n ∧ m ∣ k`。

```py
example  (x  :  ℝ)  :  |x  +  3|  <  5  →  -8  <  x  ∧  x  <  2  :=  by
  rw  [abs_lt]
  intro  h
  constructor  <;>  linarith

example  :  3  ∣  Nat.gcd  6  15  :=  by
  rw  [Nat.dvd_gcd_iff]
  constructor  <;>  norm_num 
```

看看你是否可以使用下面的定理中的 `rw` 来提供一个简短的证明，证明否定不是一个非递减函数。（注意，`push_neg` 不会为你展开定义，所以定理证明中的 `rw [Monotone]` 是必需的。）

```py
theorem  not_monotone_iff  {f  :  ℝ  →  ℝ}  :  ¬Monotone  f  ↔  ∃  x  y,  x  ≤  y  ∧  f  x  >  f  y  :=  by
  rw  [Monotone]
  push_neg
  rfl

example  :  ¬Monotone  fun  x  :  ℝ  ↦  -x  :=  by
  sorry 
```

本节剩余的练习旨在让你在合取和双条件方面获得更多实践。记住，一个**偏序**是一个传递的、自反的和反对称的二进制关系。有时会出现一个更弱的概念：**预序**只是一个自反的、传递的关系。对于任何预序 `≤`，Lean 通过 `a < b ↔ a ≤ b ∧ ¬ b ≤ a` 来公理化相关的严格预序。证明如果 `≤` 是一个偏序，那么 `a < b` 等价于 `a ≤ b ∧ a ≠ b`：

```py
variable  {α  :  Type*}  [PartialOrder  α]
variable  (a  b  :  α)

example  :  a  <  b  ↔  a  ≤  b  ∧  a  ≠  b  :=  by
  rw  [lt_iff_le_not_ge]
  sorry 
```

除了逻辑运算之外，你不需要比 `le_refl` 和 `le_trans` 更多的东西。证明即使 `≤` 只被假设为偏序，我们也可以证明严格顺序是不可自反的和传递的。在第二个例子中，为了方便起见，我们使用简化器而不是 `rw` 来用 `≤` 和 `¬` 表达 `<`。我们稍后会回到简化器，但在这里我们只依赖于这样一个事实，即它将反复使用指定的引理，即使它需要实例化为不同的值。

```py
variable  {α  :  Type*}  [Preorder  α]
variable  (a  b  c  :  α)

example  :  ¬a  <  a  :=  by
  rw  [lt_iff_le_not_ge]
  sorry

example  :  a  <  b  →  b  <  c  →  a  <  c  :=  by
  simp  only  [lt_iff_le_not_ge]
  sorry 
```  ## 3.5\. 析取

证明析取 `A ∨ B` 的标准方法是证明 `A` 或证明 `B`。`left` 策略选择 `A`，而 `right` 策略选择 `B`。

```py
variable  {x  y  :  ℝ}

example  (h  :  y  >  x  ^  2)  :  y  >  0  ∨  y  <  -1  :=  by
  left
  linarith  [pow_two_nonneg  x]

example  (h  :  -y  >  x  ^  2  +  1)  :  y  >  0  ∨  y  <  -1  :=  by
  right
  linarith  [pow_two_nonneg  x] 
```

我们不能使用匿名构造函数来构造“或”的证明，因为 Lean 必须猜测我们正在尝试证明哪个析取项。当我们编写证明项时，我们可以使用 `Or.inl` 和 `Or.inr` 来代替，以明确做出选择。在这里，`inl` 是“引入左”的简称，而 `inr` 是“引入右”的简称。

```py
example  (h  :  y  >  0)  :  y  >  0  ∨  y  <  -1  :=
  Or.inl  h

example  (h  :  y  <  -1)  :  y  >  0  ∨  y  <  -1  :=
  Or.inr  h 
```

通过证明析取的一侧或另一侧来证明析取可能看起来很奇怪。在实践中，哪个情况成立通常取决于假设和数据中隐含或显含的情况区分。`rcases` 策略允许我们利用形式为 `A ∨ B` 的假设。与 `rcases` 与合取或存在量词的使用相比，这里 `rcases` 策略产生 *两个* 目标。这两个目标有相同的结论，但在第一种情况下，假设 `A` 为真，在第二种情况下，假设 `B` 为真。换句话说，正如其名称所暗示的，`rcases` 策略通过分情况证明。像往常一样，我们可以告诉 Lean 使用哪些名称来命名假设。在下一个例子中，我们告诉 Lean 在每个分支上使用名称 `h`。

```py
example  :  x  <  |y|  →  x  <  y  ∨  x  <  -y  :=  by
  rcases  le_or_gt  0  y  with  h  |  h
  ·  rw  [abs_of_nonneg  h]
  intro  h;  left;  exact  h
  ·  rw  [abs_of_neg  h]
  intro  h;  right;  exact  h 
```

注意，模式从合取的情况中的 `⟨h₀, h₁⟩` 变为析取的情况中的 `h₀ | h₁`。将第一个模式视为与包含 `h₀` 和 `h₁` 的数据匹配，而第二个模式，带有竖线，与包含 `h₀` 或 `h₁` 的数据匹配。在这种情况下，因为两个目标是独立的，所以我们选择在每个情况下使用相同的名称，即 `h`。

绝对值函数被定义为一种方式，我们可以立即证明 `x ≥ 0` 蕴含 `|x| = x`（这是定理 `abs_of_nonneg`），以及 `x < 0` 蕴含 `|x| = -x`（这是 `abs_of_neg`）。表达式 `le_or_gt 0 x` 建立了 `0 ≤ x ∨ x < 0`，允许我们对这两种情况分别进行分割。

Lean 也支持计算机科学家用于析取的匹配语法。现在 `cases` 策略更具吸引力，因为它允许我们为每个 `case` 命名，并为引入的假设命名，使其更接近使用位置。

```py
example  :  x  <  |y|  →  x  <  y  ∨  x  <  -y  :=  by
  cases  le_or_gt  0  y
  case  inl  h  =>
  rw  [abs_of_nonneg  h]
  intro  h;  left;  exact  h
  case  inr  h  =>
  rw  [abs_of_neg  h]
  intro  h;  right;  exact  h 
```

`inl`和`inr`的名字分别代表“引入左”和“引入右”。使用`case`的优点是你可以在任意顺序证明情况；Lean 使用标签来找到相关的目标。如果你不在乎这一点，你可以使用`next`，或者`match`，甚至是一个模式匹配的`have`。

```py
example  :  x  <  |y|  →  x  <  y  ∨  x  <  -y  :=  by
  cases  le_or_gt  0  y
  next  h  =>
  rw  [abs_of_nonneg  h]
  intro  h;  left;  exact  h
  next  h  =>
  rw  [abs_of_neg  h]
  intro  h;  right;  exact  h

example  :  x  <  |y|  →  x  <  y  ∨  x  <  -y  :=  by
  match  le_or_gt  0  y  with
  |  Or.inl  h  =>
  rw  [abs_of_nonneg  h]
  intro  h;  left;  exact  h
  |  Or.inr  h  =>
  rw  [abs_of_neg  h]
  intro  h;  right;  exact  h 
```

在`match`的情况下，我们需要使用证明析取的规范方式`Or.inl`和`Or.inr`的全称。在这本教科书中，我们将一般使用`rcases`来分情况讨论析取。

尝试使用下一个片段中的前两个定理来证明三角不等式。它们在 Mathlib 中具有相同的名称。

```py
namespace  MyAbs

theorem  le_abs_self  (x  :  ℝ)  :  x  ≤  |x|  :=  by
  sorry

theorem  neg_le_abs_self  (x  :  ℝ)  :  -x  ≤  |x|  :=  by
  sorry

theorem  abs_add  (x  y  :  ℝ)  :  |x  +  y|  ≤  |x|  +  |y|  :=  by
  sorry 
```

如果你喜欢这些（当然，是字面意义上的喜欢）并且想要更多关于析取的练习，试试这些。

```py
theorem  lt_abs  :  x  <  |y|  ↔  x  <  y  ∨  x  <  -y  :=  by
  sorry

theorem  abs_lt  :  |x|  <  y  ↔  -y  <  x  ∧  x  <  y  :=  by
  sorry 
```

你也可以使用嵌套析取的`rcases`和`rintro`。当这些导致具有多个目标的真正情况分割时，每个新目标的模式由一个竖线分隔。

```py
example  {x  :  ℝ}  (h  :  x  ≠  0)  :  x  <  0  ∨  x  >  0  :=  by
  rcases  lt_trichotomy  x  0  with  xlt  |  xeq  |  xgt
  ·  left
  exact  xlt
  ·  contradiction
  ·  right;  exact  xgt 
```

你仍然可以嵌套模式并使用`rfl`关键字替换方程：

```py
example  {m  n  k  :  ℕ}  (h  :  m  ∣  n  ∨  m  ∣  k)  :  m  ∣  n  *  k  :=  by
  rcases  h  with  ⟨a,  rfl⟩  |  ⟨b,  rfl⟩
  ·  rw  [mul_assoc]
  apply  dvd_mul_right
  ·  rw  [mul_comm,  mul_assoc]
  apply  dvd_mul_right 
```

看看你是否可以用一行（长）来证明以下内容。使用`rcases`来展开假设并分情况讨论，并使用分号和`linarith`来解决每个分支。

```py
example  {z  :  ℝ}  (h  :  ∃  x  y,  z  =  x  ^  2  +  y  ^  2  ∨  z  =  x  ^  2  +  y  ^  2  +  1)  :  z  ≥  0  :=  by
  sorry 
```

在实数上，一个方程`x * y = 0`告诉我们`x = 0`或`y = 0`。在 Mathlib 中，这个事实被称为`eq_zero_or_eq_zero_of_mul_eq_zero`，它是析取如何出现的另一个很好的例子。看看你是否可以用它来证明以下内容：

```py
example  {x  :  ℝ}  (h  :  x  ^  2  =  1)  :  x  =  1  ∨  x  =  -1  :=  by
  sorry

example  {x  y  :  ℝ}  (h  :  x  ^  2  =  y  ^  2)  :  x  =  y  ∨  x  =  -y  :=  by
  sorry 
```

记住你可以使用`ring`策略来帮助计算。

在任意环$R$中，一个元素$x$，如果存在某个非零元素$y$使得$x y = 0$，则称为**左零因子**；一个元素$x$，如果存在某个非零元素$y$使得$y x = 0$，则称为**右零因子**；如果一个元素是左零因子或右零因子，则简单地称为**零因子**。定理`eq_zero_or_eq_zero_of_mul_eq_zero`表明实数没有非平凡零因子。具有这种性质的交换环称为**整环**。你上面两个定理的证明在任意整环中同样适用：

```py
variable  {R  :  Type*}  [CommRing  R]  [IsDomain  R]
variable  (x  y  :  R)

example  (h  :  x  ^  2  =  1)  :  x  =  1  ∨  x  =  -1  :=  by
  sorry

example  (h  :  x  ^  2  =  y  ^  2)  :  x  =  y  ∨  x  =  -y  :=  by
  sorry 
```

实际上，如果你小心的话，可以不用乘法的交换性来证明第一个定理。在这种情况下，只需假设$R$是一个`Ring`而不是`CommRing`。

有时在证明中，我们想要根据某个陈述是否为真来分情况讨论。对于任何命题$P$，我们可以使用`em P : P ∨ ¬ P`。`em`这个名字是“排中律”的缩写。

```py
example  (P  :  Prop)  :  ¬¬P  →  P  :=  by
  intro  h
  cases  em  P
  ·  assumption
  ·  contradiction 
```

或者，你可以使用`by_cases`策略。

```py
example  (P  :  Prop)  :  ¬¬P  →  P  :=  by
  intro  h
  by_cases  h'  :  P
  ·  assumption
  contradiction 
```

注意到`by_cases`策略允许你为每个分支中引入的假设指定一个标签，在这种情况下，`h' : P`在一个分支中，`h' : ¬ P`在另一个分支中。如果你省略了标签，Lean 默认使用`h`。尝试使用`by_cases`来证明以下等价性，以建立其中一个方向。

```py
example  (P  Q  :  Prop)  :  P  →  Q  ↔  ¬P  ∨  Q  :=  by
  sorry 
```  ## 3.6\. 序列与收敛

现在我们已经掌握了足够的技能来做一些真正的数学。在 Lean 中，我们可以将一个实数序列 $s_0, s_1, s_2, \ldots$ 表示为一个函数 `s : ℕ → ℝ`。这样的序列被称为 *收敛* 到一个数 $a$，如果对于每一个 $\varepsilon > 0$，都有一个点，在此点之后序列始终保持在 $a$ 的 $\varepsilon$ 范围内，也就是说，存在一个数 $N$，使得对于每一个 $n \ge N$，$| s_n - a | < \varepsilon$。在 Lean 中，我们可以这样表示：

```py
def  ConvergesTo  (s  :  ℕ  →  ℝ)  (a  :  ℝ)  :=
  ∀  ε  >  0,  ∃  N,  ∀  n  ≥  N,  |s  n  -  a|  <  ε 
```

符号 `∀ ε > 0, ...` 是 `∀ ε, ε > 0 → ...` 的方便缩写，同样地，`∀ n ≥ N, ...` 缩写为 `∀ n, n ≥ N → ...`。并且记住，`ε > 0`，反过来，定义为 `0 < ε`，而 `n ≥ N` 定义为 `N ≤ n`。

在本节中，我们将建立一些关于收敛的性质。但首先，我们将讨论三种用于处理等式的策略，这些策略将非常有用。第一个，`ext` 策略，为我们提供了一种证明两个函数相等的方法。设 $f(x) = x + 1$ 和 $g(x) = 1 + x$ 是从实数到实数的函数。当然，$f = g$，因为它们对于每一个 $x$ 都返回相同的值。`ext` 策略使我们能够通过证明在所有参数值上它们的值都相同来证明函数之间的等式。

```py
example  :  (fun  x  y  :  ℝ  ↦  (x  +  y)  ^  2)  =  fun  x  y  :  ℝ  ↦  x  ^  2  +  2  *  x  *  y  +  y  ^  2  :=  by
  ext
  ring 
```

我们稍后会看到，`ext` 实际上更通用，并且还可以指定出现的变量的名称。例如，你可以在上面的证明中尝试用 `ext u v` 替换 `ext`。第二个策略，`congr` 策略，允许我们通过协调不同的部分来证明两个表达式之间的等式：

```py
example  (a  b  :  ℝ)  :  |a|  =  |a  -  b  +  b|  :=  by
  congr
  ring 
```

在这里，`congr` 策略从每一边剥去 `abs`，留下我们证明 `a = a - b + b`。

最后，`convert` 策略用于在定理的结论不完全匹配目标时应用定理。例如，假设我们想从 `1 < a` 证明 `a < a * a`。库中的一个定理 `mul_lt_mul_right` 将允许我们证明 `1 * a < a * a`。一种可能的方法是反向工作并重写目标，使其具有那种形式。相反，`convert` 策略允许我们直接应用定理，并留下证明目标匹配所需方程的任务。

```py
example  {a  :  ℝ}  (h  :  1  <  a)  :  a  <  a  *  a  :=  by
  convert  (mul_lt_mul_right  _).2  h
  ·  rw  [one_mul]
  exact  lt_trans  zero_lt_one  h 
```

这个例子说明了另一个有用的技巧：当我们应用一个带有下划线的表达式，而 Lean 无法自动为我们填充它时，它就简单地将其留给我们作为另一个目标。

下面的例子表明，任何常数序列 $a, a, a, \ldots$ 都会收敛。

```py
theorem  convergesTo_const  (a  :  ℝ)  :  ConvergesTo  (fun  x  :  ℕ  ↦  a)  a  :=  by
  intro  ε  εpos
  use  0
  intro  n  nge
  rw  [sub_self,  abs_zero]
  apply  εpos 
```

Lean 有一个策略，`simp`，它经常可以帮你省去手动执行 `rw [sub_self, abs_zero]` 等步骤的麻烦。我们很快就会告诉你更多关于它的信息。

对于一个更有趣的定理，让我们证明如果 `s` 收敛到 `a` 且 `t` 收敛到 `b`，那么 `fun n ↦ s n + t n` 收敛到 `a + b`。在开始编写正式证明之前，有一个清晰的笔和纸证明是有帮助的。给定大于 `0` 的 `ε`，想法是使用假设来获得一个 `Ns`，使得在那个点之后，`s` 在 `a` 的 `ε / 2` 范围内，以及一个 `Nt`，使得在那个点之后，`t` 在 `b` 的 `ε / 2` 范围内。然后，每当 `n` 大于或等于 `Ns` 和 `Nt` 的最大值时，序列 `fun n ↦ s n + t n` 应该在 `a + b` 的 `ε` 范围内。以下示例开始实施这一策略。看看你是否可以完成它。

```py
theorem  convergesTo_add  {s  t  :  ℕ  →  ℝ}  {a  b  :  ℝ}
  (cs  :  ConvergesTo  s  a)  (ct  :  ConvergesTo  t  b)  :
  ConvergesTo  (fun  n  ↦  s  n  +  t  n)  (a  +  b)  :=  by
  intro  ε  εpos
  dsimp  -- this line is not needed but cleans up the goal a bit.
  have  ε2pos  :  0  <  ε  /  2  :=  by  linarith
  rcases  cs  (ε  /  2)  ε2pos  with  ⟨Ns,  hs⟩
  rcases  ct  (ε  /  2)  ε2pos  with  ⟨Nt,  ht⟩
  use  max  Ns  Nt
  sorry 
```

作为提示，你可以使用 `le_of_max_le_left` 和 `le_of_max_le_right`，并且 `norm_num` 可以证明 `ε / 2 + ε / 2 = ε`。此外，使用 `congr` 策略来证明 `|s n + t n - (a + b)|` 等于 `|(s n - a) + (t n - b)|` 是有帮助的，因为这样你就可以使用三角不等式。注意，我们标记了所有变量 `s`、`t`、`a` 和 `b` 为隐含的，因为它们可以从假设中推断出来。

用乘法代替加法来证明相同的定理是棘手的。我们将通过首先证明一些辅助陈述来达到这一点。看看你是否也可以完成下一个证明，该证明表明如果 `s` 收敛到 `a`，那么 `fun n ↦ c * s n` 收敛到 `c * a`。根据 `c` 是否为零来分情况考虑是有帮助的。我们已经处理了零的情况，并留下了额外的假设 `c` 不为零来证明结果。

```py
theorem  convergesTo_mul_const  {s  :  ℕ  →  ℝ}  {a  :  ℝ}  (c  :  ℝ)  (cs  :  ConvergesTo  s  a)  :
  ConvergesTo  (fun  n  ↦  c  *  s  n)  (c  *  a)  :=  by
  by_cases  h  :  c  =  0
  ·  convert  convergesTo_const  0
  ·  rw  [h]
  ring
  rw  [h]
  ring
  have  acpos  :  0  <  |c|  :=  abs_pos.mpr  h
  sorry 
```

下一个定理也是独立有趣的：它表明收敛序列最终在绝对值上是有限的。我们已经为你开始了；看看你是否可以完成它。

```py
theorem  exists_abs_le_of_convergesTo  {s  :  ℕ  →  ℝ}  {a  :  ℝ}  (cs  :  ConvergesTo  s  a)  :
  ∃  N  b,  ∀  n,  N  ≤  n  →  |s  n|  <  b  :=  by
  rcases  cs  1  zero_lt_one  with  ⟨N,  h⟩
  use  N,  |a|  +  1
  sorry 
```

事实上，该定理可以加强到断言存在一个对所有 `n` 的值都成立的界限 `b`。但这个版本对我们来说已经足够强大，我们将在本节末尾看到它具有更一般的适用性。

下一个引理是辅助性的：我们证明如果 `s` 收敛到 `a` 且 `t` 收敛到 `0`，那么 `fun n ↦ s n * t n` 收敛到 `0`。为此，我们使用前面的定理找到一个 `B` 来限制 `s` 在某个点 `N₀` 之后。看看你是否能理解我们所概述的策略并完成证明。

```py
theorem  aux  {s  t  :  ℕ  →  ℝ}  {a  :  ℝ}  (cs  :  ConvergesTo  s  a)  (ct  :  ConvergesTo  t  0)  :
  ConvergesTo  (fun  n  ↦  s  n  *  t  n)  0  :=  by
  intro  ε  εpos
  dsimp
  rcases  exists_abs_le_of_convergesTo  cs  with  ⟨N₀,  B,  h₀⟩
  have  Bpos  :  0  <  B  :=  lt_of_le_of_lt  (abs_nonneg  _)  (h₀  N₀  (le_refl  _))
  have  pos₀  :  ε  /  B  >  0  :=  div_pos  εpos  Bpos
  rcases  ct  _  pos₀  with  ⟨N₁,  h₁⟩
  sorry 
```

如果你已经走到这一步，恭喜你！我们现在已经接近我们的定理了。以下证明完成了它。

```py
theorem  convergesTo_mul  {s  t  :  ℕ  →  ℝ}  {a  b  :  ℝ}
  (cs  :  ConvergesTo  s  a)  (ct  :  ConvergesTo  t  b)  :
  ConvergesTo  (fun  n  ↦  s  n  *  t  n)  (a  *  b)  :=  by
  have  h₁  :  ConvergesTo  (fun  n  ↦  s  n  *  (t  n  +  -b))  0  :=  by
  apply  aux  cs
  convert  convergesTo_add  ct  (convergesTo_const  (-b))
  ring
  have  :=  convergesTo_add  h₁  (convergesTo_mul_const  b  cs)
  convert  convergesTo_add  h₁  (convergesTo_mul_const  b  cs)  using  1
  ·  ext;  ring
  ring 
```

对于另一个具有挑战性的练习，尝试完成以下关于极限唯一的证明草稿。（如果你感到大胆，你可以删除证明草稿并尝试从头开始证明。）

```py
theorem  convergesTo_unique  {s  :  ℕ  →  ℝ}  {a  b  :  ℝ}
  (sa  :  ConvergesTo  s  a)  (sb  :  ConvergesTo  s  b)  :
  a  =  b  :=  by
  by_contra  abne
  have  :  |a  -  b|  >  0  :=  by  sorry
  let  ε  :=  |a  -  b|  /  2
  have  εpos  :  ε  >  0  :=  by
  change  |a  -  b|  /  2  >  0
  linarith
  rcases  sa  ε  εpos  with  ⟨Na,  hNa⟩
  rcases  sb  ε  εpos  with  ⟨Nb,  hNb⟩
  let  N  :=  max  Na  Nb
  have  absa  :  |s  N  -  a|  <  ε  :=  by  sorry
  have  absb  :  |s  N  -  b|  <  ε  :=  by  sorry
  have  :  |a  -  b|  <  |a  -  b|  :=  by  sorry
  exact  lt_irrefl  _  this 
```

我们以观察结束本节，即我们的证明可以推广。例如，我们使用的自然数的唯一性质是它们的结构携带一个具有 `min` 和 `max` 的偏序。你可以检查，如果你将 `ℕ` 在任何地方都替换为任何线性序 `α`，一切仍然有效：

```py
variable  {α  :  Type*}  [LinearOrder  α]

def  ConvergesTo'  (s  :  α  →  ℝ)  (a  :  ℝ)  :=
  ∀  ε  >  0,  ∃  N,  ∀  n  ≥  N,  |s  n  -  a|  <  ε 
```

在第 11.1 节中，我们将看到 Mathlib 有处理收敛性的机制，这些机制在更广泛的范围内，不仅抽象掉了定义域和值域的特定特征，而且还抽象了不同类型的收敛。 上一节 下一节

* * *

© 版权所有 2020-2025，Jeremy Avigad，Patrick Massot。文本许可协议为 CC BY 4.0。

使用[Sphinx](https://www.sphinx-doc.org/)构建，使用[主题](https://github.com/readthedocs/sphinx_rtd_theme)由[Read the Docs](https://readthedocs.org)提供。在上一章中，我们处理了方程、不等式和像“$x$ divides $y$”这样的基本数学陈述。复杂的数学陈述是通过使用像“and”、“or”、“not”、“if … then”、“every”和“some”这样的逻辑术语从这些简单的陈述构建而成的。在本章中，我们向你展示如何处理以这种方式构建的陈述。

## 3.1\. 蕴涵和全称量词

考虑`#check`之后的陈述：

```py
#check  ∀  x  :  ℝ,  0  ≤  x  →  |x|  =  x 
```

用话来说，我们会说“对于每一个实数`x`，如果`0 ≤ x`则`x`的绝对值等于`x`”。我们也可以有更复杂的陈述，例如：

```py
#check  ∀  x  y  ε  :  ℝ,  0  <  ε  →  ε  ≤  1  →  |x|  <  ε  →  |y|  <  ε  →  |x  *  y|  <  ε 
```

用话来说，我们会说“对于每一个`x`，`y`和`ε`，如果`0 < ε ≤ 1`，则`x`的绝对值小于`ε`，`y`的绝对值也小于`ε`，那么`x * y`的绝对值也小于`ε`。”在 Lean 中，在一系列蕴涵中，有隐式的括号分组到右边。所以上面的表达式意味着“如果`0 < ε`那么如果`ε ≤ 1`那么如果`|x| < ε`…”因此，这个表达式表明所有假设共同蕴涵结论。

你已经看到，尽管这个陈述中的全称量词在对象上取值，而蕴涵箭头引入了假设，但 Lean 以非常相似的方式处理这两者。特别是，如果你已经证明了这种形式的定理，你可以以相同的方式将其应用于对象和假设。我们将以下陈述作为例子，稍后我们将帮助你证明它：

```py
theorem  my_lemma  :  ∀  x  y  ε  :  ℝ,  0  <  ε  →  ε  ≤  1  →  |x|  <  ε  →  |y|  <  ε  →  |x  *  y|  <  ε  :=
  sorry

section
variable  (a  b  δ  :  ℝ)
variable  (h₀  :  0  <  δ)  (h₁  :  δ  ≤  1)
variable  (ha  :  |a|  <  δ)  (hb  :  |b|  <  δ)

#check  my_lemma  a  b  δ
#check  my_lemma  a  b  δ  h₀  h₁
#check  my_lemma  a  b  δ  h₀  h₁  ha  hb

end 
```

你也已经看到，在 Lean 中，当量词可以从后续假设中推断出来时，通常使用花括号来使量词隐式化。当我们这样做时，我们只需将引理应用于假设，而不必提及对象。

```py
theorem  my_lemma2  :  ∀  {x  y  ε  :  ℝ},  0  <  ε  →  ε  ≤  1  →  |x|  <  ε  →  |y|  <  ε  →  |x  *  y|  <  ε  :=
  sorry

section
variable  (a  b  δ  :  ℝ)
variable  (h₀  :  0  <  δ)  (h₁  :  δ  ≤  1)
variable  (ha  :  |a|  <  δ)  (hb  :  |b|  <  δ)

#check  my_lemma2  h₀  h₁  ha  hb

end 
```

在这个阶段，你也知道，如果你使用`apply`策略将`my_lemma`应用于形式为`|a * b| < δ`的目标，你将留下新的目标，需要你证明每个假设。

要证明这样的陈述，请使用`intro`策略。看看它在以下示例中的表现：

```py
theorem  my_lemma3  :
  ∀  {x  y  ε  :  ℝ},  0  <  ε  →  ε  ≤  1  →  |x|  <  ε  →  |y|  <  ε  →  |x  *  y|  <  ε  :=  by
  intro  x  y  ε  epos  ele1  xlt  ylt
  sorry 
```

我们可以为全称量化的变量使用任何我们想要的名称；它们不必是 `x`、`y` 和 `ε`。注意，即使它们被标记为隐式，我们也必须引入这些变量：使它们成为隐式意味着在写一个使用 `my_lemma` 的表达式时，我们可以省略它们，但它们仍然是我们要证明的陈述的一个基本部分。在 `intro` 命令之后，目标是如果在冒号之前列出了所有变量和假设，就像我们在上一节中做的那样，那么它将是什么。在不久的将来，我们将看到为什么有时在证明开始之后引入变量和假设是必要的。

为了帮助你证明这个引理，我们将从以下步骤开始：

```py
theorem  my_lemma4  :
  ∀  {x  y  ε  :  ℝ},  0  <  ε  →  ε  ≤  1  →  |x|  <  ε  →  |y|  <  ε  →  |x  *  y|  <  ε  :=  by
  intro  x  y  ε  epos  ele1  xlt  ylt
  calc
  |x  *  y|  =  |x|  *  |y|  :=  sorry
  _  ≤  |x|  *  ε  :=  sorry
  _  <  1  *  ε  :=  sorry
  _  =  ε  :=  sorry 
```

使用定理 `abs_mul`、`mul_le_mul`、`abs_nonneg`、`mul_lt_mul_right` 和 `one_mul` 完成证明。记住，你可以使用 Ctrl-space 完成提示（或在 Mac 上使用 Cmd-space 完成提示）来找到这样的定理。还要记住，你可以使用 `.mp` 和 `.mpr` 或 `.1` 和 `.2` 来提取双向条件语句的两个方向。

全称量词通常隐藏在定义中，当需要时 Lean 会展开定义来揭示它们。例如，让我们定义两个谓词，`FnUb f a` 和 `FnLb f a`，其中 `f` 是从实数到实数的函数，而 `a` 是一个实数。第一个谓词表示 `a` 是 `f` 的值的上界，第二个谓词表示 `a` 是 `f` 的值的下界。

```py
def  FnUb  (f  :  ℝ  →  ℝ)  (a  :  ℝ)  :  Prop  :=
  ∀  x,  f  x  ≤  a

def  FnLb  (f  :  ℝ  →  ℝ)  (a  :  ℝ)  :  Prop  :=
  ∀  x,  a  ≤  f  x 
```

在下一个例子中，`fun x ↦ f x + g x` 是将 `x` 映射到 `f x + g x` 的函数。从表达式 `f x + g x` 到这个函数的转换在类型理论中被称为 lambda 抽象。

```py
example  (hfa  :  FnUb  f  a)  (hgb  :  FnUb  g  b)  :  FnUb  (fun  x  ↦  f  x  +  g  x)  (a  +  b)  :=  by
  intro  x
  dsimp
  apply  add_le_add
  apply  hfa
  apply  hgb 
```

将 `intro` 应用到目标 `FnUb (fun x ↦ f x + g x) (a + b)` 迫使 Lean 展开定义 `FnUb` 并引入 `x` 作为全称量词。目标随后变为 `(fun (x : ℝ) ↦ f x + g x) x ≤ a + b`。但是将 `(fun x ↦ f x + g x)` 应用到 `x` 应该得到 `f x + g x`，而 `dsimp` 命令执行了这种简化。（“d”代表“定义性的。”）你可以删除那个命令，证明仍然有效；Lean 无论如何都必须执行那个收缩，以便理解下一个 `apply`。`dsimp` 命令只是使目标更易于阅读，并帮助我们确定下一步要做什么。另一个选择是使用 `change` 策略，通过编写 `change f x + g x ≤ a + b` 来实现。这有助于使证明更易于阅读，并让你能够更好地控制目标是如何转换的。

证明的其余部分是常规的。最后的两个 `apply` 命令迫使 Lean 在假设中展开 `FnUb` 的定义。尝试执行类似的证明：

```py
example  (hfa  :  FnLb  f  a)  (hgb  :  FnLb  g  b)  :  FnLb  (fun  x  ↦  f  x  +  g  x)  (a  +  b)  :=
  sorry

example  (nnf  :  FnLb  f  0)  (nng  :  FnLb  g  0)  :  FnLb  (fun  x  ↦  f  x  *  g  x)  0  :=
  sorry

example  (hfa  :  FnUb  f  a)  (hgb  :  FnUb  g  b)  (nng  :  FnLb  g  0)  (nna  :  0  ≤  a)  :
  FnUb  (fun  x  ↦  f  x  *  g  x)  (a  *  b)  :=
  sorry 
```

尽管我们已经为从实数到实数的函数定义了 `FnUb` 和 `FnLb`，但你应该认识到定义和证明要更通用。这些定义适用于任何两个类型之间的函数，其中在值域上有有序的概念。检查定理 `add_le_add` 的类型显示，它适用于任何“有序加法交换幺半群”的结构；现在不需要详细说明这意味着什么，但值得知道自然数、整数、有理数和实数都是实例。因此，如果我们在这个普遍性级别上证明定理 `fnUb_add`，它将适用于所有这些实例。

```py
variable  {α  :  Type*}  {R  :  Type*}  [AddCommMonoid  R]  [PartialOrder  R]  [IsOrderedCancelAddMonoid  R]

#check  add_le_add

def  FnUb'  (f  :  α  →  R)  (a  :  R)  :  Prop  :=
  ∀  x,  f  x  ≤  a

theorem  fnUb_add  {f  g  :  α  →  R}  {a  b  :  R}  (hfa  :  FnUb'  f  a)  (hgb  :  FnUb'  g  b)  :
  FnUb'  (fun  x  ↦  f  x  +  g  x)  (a  +  b)  :=  fun  x  ↦  add_le_add  (hfa  x)  (hgb  x) 
```

你已经在 第 2.2 节 中看到了这样的方括号，尽管我们还没有解释它们的含义。为了具体化，我们将在大多数示例中坚持使用实数，但值得知道 Mathlib 包含在高度普遍性级别上工作的定义和定理。

以下是一个隐藏的全称量词的例子：Mathlib 定义了一个谓词 `Monotone`，它表示一个函数在其参数上是单调递增的：

```py
example  (f  :  ℝ  →  ℝ)  (h  :  Monotone  f)  :  ∀  {a  b},  a  ≤  b  →  f  a  ≤  f  b  :=
  @h 
```

属性 `Monotone f` 被定义为冒号后面的确切表达式。我们需要在 `h` 前面放置 `@` 符号，因为如果不这样做，Lean 会扩展 `h` 的隐含参数并插入占位符。

证明关于单调性的陈述涉及使用 `intro` 引入两个变量，例如 `a` 和 `b`，以及假设 `a ≤ b`。要使用单调性假设，你可以将其应用于合适的参数和假设，然后将结果表达式应用于目标。或者，你可以将其应用于目标，让 Lean 通过显示剩余假设作为新的子目标来帮助你反向工作。

```py
example  (mf  :  Monotone  f)  (mg  :  Monotone  g)  :  Monotone  fun  x  ↦  f  x  +  g  x  :=  by
  intro  a  b  aleb
  apply  add_le_add
  apply  mf  aleb
  apply  mg  aleb 
```

当证明非常简短时，通常方便给出一个证明项。为了描述一个临时引入对象 `a` 和 `b` 以及假设 `aleb` 的证明，Lean 使用了 `fun a b aleb ↦ ...` 的符号。这与像 `fun x ↦ x²` 这样的表达式通过临时命名一个对象 `x` 并使用它来描述一个值的方式来描述函数类似。因此，前一个证明中的 `intro` 命令对应于下一个证明项中的 lambda 抽象。然后 `apply` 命令对应于构建定理对其参数的应用。

```py
example  (mf  :  Monotone  f)  (mg  :  Monotone  g)  :  Monotone  fun  x  ↦  f  x  +  g  x  :=
  fun  a  b  aleb  ↦  add_le_add  (mf  aleb)  (mg  aleb) 
```

这里有一个有用的技巧：如果你开始编写证明项 `fun a b aleb ↦ _`，在表达式其余部分应该放置的地方使用下划线，Lean 会标记一个错误，表明它无法猜测该表达式的值。如果你在 VS Code 中的 Lean 目标窗口中检查或悬停在波浪形错误标记上，Lean 会显示剩余表达式需要解决的目標。

尝试使用策略或证明项证明以下内容：

```py
example  {c  :  ℝ}  (mf  :  Monotone  f)  (nnc  :  0  ≤  c)  :  Monotone  fun  x  ↦  c  *  f  x  :=
  sorry

example  (mf  :  Monotone  f)  (mg  :  Monotone  g)  :  Monotone  fun  x  ↦  f  (g  x)  :=
  sorry 
```

这里有一些更多的例子。从 $\Bbb R$ 到 $\Bbb R$ 的函数 $f$ 被称为 *偶函数*，如果对于每个 $x$，有 $f(-x) = f(x)$；如果对于每个 $x$，有 $f(-x) = -f(x)$，则称为 *奇函数*。以下示例正式定义了这两个概念，并确立了一个关于它们的定理。你可以完成其他定理的证明。

```py
def  FnEven  (f  :  ℝ  →  ℝ)  :  Prop  :=
  ∀  x,  f  x  =  f  (-x)

def  FnOdd  (f  :  ℝ  →  ℝ)  :  Prop  :=
  ∀  x,  f  x  =  -f  (-x)

example  (ef  :  FnEven  f)  (eg  :  FnEven  g)  :  FnEven  fun  x  ↦  f  x  +  g  x  :=  by
  intro  x
  calc
  (fun  x  ↦  f  x  +  g  x)  x  =  f  x  +  g  x  :=  rfl
  _  =  f  (-x)  +  g  (-x)  :=  by  rw  [ef,  eg]

example  (of  :  FnOdd  f)  (og  :  FnOdd  g)  :  FnEven  fun  x  ↦  f  x  *  g  x  :=  by
  sorry

example  (ef  :  FnEven  f)  (og  :  FnOdd  g)  :  FnOdd  fun  x  ↦  f  x  *  g  x  :=  by
  sorry

example  (ef  :  FnEven  f)  (og  :  FnOdd  g)  :  FnEven  fun  x  ↦  f  (g  x)  :=  by
  sorry 
```

首个证明可以使用 `dsimp` 或 `change` 来缩短，从而去除 lambda 抽象。但你可以检查，除非我们明确去除 lambda 抽象，否则后续的 `rw` 不会起作用，因为否则它无法在表达式中找到 `f x` 和 `g x` 这两个模式。与一些其他策略相反，`rw` 在句法层面上操作，它不会展开定义或为你应用简化（它有一个名为 `erw` 的变体，在这个方向上尝试得稍微努力一些，但并不太多）。

一旦你知道如何找到它们，你可以在任何地方找到隐含的全称量词。

Mathlib 包含了一个用于操作集合的良好库。回想一下，Lean 不使用基于集合论的公理系统，因此这里的“集合”一词具有其平凡的数学对象集合的含义，这些对象属于某个给定的类型 `α`。如果 `x` 的类型是 `α`，而 `s` 的类型是 `Set α`，那么 `x ∈ s` 是一个命题，它断言 `x` 是 `s` 的一个元素。如果 `y` 有某种不同的类型 `β`，那么表达式 `y ∈ s` 就没有意义。这里的“没有意义”是指“没有类型，因此 Lean 不接受它作为一个有效的语句”。这与例如 Zermelo-Fraenkel 集合论不同，在 ZF 中，`a ∈ b` 对于每个数学对象 `a` 和 `b` 都是一个有效的语句。例如，`sin ∈ cos` 在 ZF 中是一个有效的语句。集合论基础的这个缺陷是重要的动机，即不在旨在通过检测无意义表达式来帮助我们证明辅助工具中使用它。在 Lean 中，`sin` 的类型是 `ℝ → ℝ`，而 `cos` 的类型是 `ℝ → ℝ`，这并不等于 `Set (ℝ → ℝ)`，即使展开定义后也是如此，因此语句 `sin ∈ cos` 没有意义。人们也可以使用 Lean 来处理集合论本身。例如，连续假设与 Zermelo-Fraenkel 公理的独立性已经在 Lean 中形式化。但这样的集合论元理论完全超出了本书的范围。

如果 `s` 和 `t` 的类型都是 `Set α`，那么子集关系 `s ⊆ t` 被定义为 `∀ {x : α}, x ∈ s → x ∈ t`。量词中的变量被标记为隐式，这样给定 `h : s ⊆ t` 和 `h' : x ∈ s`，我们可以将 `h h'` 写作 `x ∈ t` 的理由。以下示例提供了一个策略证明和一个证明项，以证明子集关系的自反性，并要求你为传递性做同样的工作。

```py
variable  {α  :  Type*}  (r  s  t  :  Set  α)

example  :  s  ⊆  s  :=  by
  intro  x  xs
  exact  xs

theorem  Subset.refl  :  s  ⊆  s  :=  fun  x  xs  ↦  xs

theorem  Subset.trans  :  r  ⊆  s  →  s  ⊆  t  →  r  ⊆  t  :=  by
  sorry 
```

正如我们为函数定义了`FnUb`，我们也可以定义`SetUb s a`，表示`a`是集合`s`的上界，假设`s`是某种类型元素的集合，并且与它相关联一个顺序。在下一个例子中，我们要求你证明如果`a`是`s`的界限且`a ≤ b`，那么`b`也是`s`的界限。

```py
variable  {α  :  Type*}  [PartialOrder  α]
variable  (s  :  Set  α)  (a  b  :  α)

def  SetUb  (s  :  Set  α)  (a  :  α)  :=
  ∀  x,  x  ∈  s  →  x  ≤  a

example  (h  :  SetUb  s  a)  (h'  :  a  ≤  b)  :  SetUb  s  b  :=
  sorry 
```

我们以一个最后的例子来结束本节。一个函数 $f$ 被称为**单射**，如果对于每一个 $x_1$ 和 $x_2$，如果 $f(x_1) = f(x_2)$ 则 $x_1 = x_2$。Mathlib 使用 `Function.Injective f` 定义 `x₁` 和 `x₂` 是隐式的。下一个例子显示，在实数上，任何加常数的函数都是单射。然后我们要求你证明乘以一个非零常数也是单射，可以使用例子中的引理名称作为灵感来源。回想一下，在猜测引理名称的开头后，你应该使用 Ctrl-space 完成提示。

```py
open  Function

example  (c  :  ℝ)  :  Injective  fun  x  ↦  x  +  c  :=  by
  intro  x₁  x₂  h'
  exact  (add_left_inj  c).mp  h'

example  {c  :  ℝ}  (h  :  c  ≠  0)  :  Injective  fun  x  ↦  c  *  x  :=  by
  sorry 
```

最后，证明两个单射函数的复合也是单射：

```py
variable  {α  :  Type*}  {β  :  Type*}  {γ  :  Type*}
variable  {g  :  β  →  γ}  {f  :  α  →  β}

example  (injg  :  Injective  g)  (injf  :  Injective  f)  :  Injective  fun  x  ↦  g  (f  x)  :=  by
  sorry 
```  ## 3.2\. 存在量词

存在量词，在 VS Code 中可以输入为`\ex`，用于表示“存在”这个短语。在 Lean 中的形式表达式 `∃ x : ℝ, 2 < x ∧ x < 3` 表示存在一个实数在 2 和 3 之间。（我们将在第 3.4 节中讨论合取符号`∧`。）证明此类陈述的规范方法是展示一个实数并证明它具有所述属性。数字 2.5，我们可以输入为 `5 / 2` 或 `(5 : ℝ) / 2` 当 Lean 无法从上下文中推断出我们指的是实数时，具有所需的属性，并且`norm_num`策略可以证明它符合描述。

我们有几种方法可以将信息组合起来。给定一个以存在量词开始的目標，使用`use`策略来提供对象，留下证明属性的目標。

```py
example  :  ∃  x  :  ℝ,  2  <  x  ∧  x  <  3  :=  by
  use  5  /  2
  norm_num 
```

你可以给出`use`策略的证明以及数据：

```py
example  :  ∃  x  :  ℝ,  2  <  x  ∧  x  <  3  :=  by
  have  h1  :  2  <  (5  :  ℝ)  /  2  :=  by  norm_num
  have  h2  :  (5  :  ℝ)  /  2  <  3  :=  by  norm_num
  use  5  /  2,  h1,  h2 
```

事实上，`use`策略会自动尝试使用可用的假设。

```py
example  :  ∃  x  :  ℝ,  2  <  x  ∧  x  <  3  :=  by
  have  h  :  2  <  (5  :  ℝ)  /  2  ∧  (5  :  ℝ)  /  2  <  3  :=  by  norm_num
  use  5  /  2 
```

或者，我们可以使用 Lean 的**匿名构造函数**符号来构造存在量词的证明。

```py
example  :  ∃  x  :  ℝ,  2  <  x  ∧  x  <  3  :=
  have  h  :  2  <  (5  :  ℝ)  /  2  ∧  (5  :  ℝ)  /  2  <  3  :=  by  norm_num
  ⟨5  /  2,  h⟩ 
```

注意，这里没有`by`；我们在这里给出一个明确的证明项。左和右尖括号，分别可以输入为`\<`和`\>`，告诉 Lean 使用适合当前目標的任何构造来组合给定数据。我们可以使用这种符号而不必首先进入策略模式：

```py
example  :  ∃  x  :  ℝ,  2  <  x  ∧  x  <  3  :=
  ⟨5  /  2,  by  norm_num⟩ 
```

因此，现在我们知道了如何 *证明* 一个存在语句。但如何 *使用* 它呢？如果我们知道存在具有某种属性的对象，我们应该能够为任意一个对象命名并对其进行推理。例如，记住上一节中的谓词 `FnUb f a` 和 `FnLb f a`，它们分别表示 `a` 是 `f` 的上界或下界。我们可以使用存在量词来说明“`f` 有界”，而不指定界限：

```py
def  FnUb  (f  :  ℝ  →  ℝ)  (a  :  ℝ)  :  Prop  :=
  ∀  x,  f  x  ≤  a

def  FnLb  (f  :  ℝ  →  ℝ)  (a  :  ℝ)  :  Prop  :=
  ∀  x,  a  ≤  f  x

def  FnHasUb  (f  :  ℝ  →  ℝ)  :=
  ∃  a,  FnUb  f  a

def  FnHasLb  (f  :  ℝ  →  ℝ)  :=
  ∃  a,  FnLb  f  a 
```

我们可以使用上一节中的定理 `FnUb_add` 来证明，如果 `f` 和 `g` 有上界，那么 `fun x ↦ f x + g x` 也有上界。

```py
variable  {f  g  :  ℝ  →  ℝ}

example  (ubf  :  FnHasUb  f)  (ubg  :  FnHasUb  g)  :  FnHasUb  fun  x  ↦  f  x  +  g  x  :=  by
  rcases  ubf  with  ⟨a,  ubfa⟩
  rcases  ubg  with  ⟨b,  ubgb⟩
  use  a  +  b
  apply  fnUb_add  ubfa  ubgb 
```

`rcases` 策略解包存在量词中的信息。像 `⟨a, ubfa⟩` 这样的注释，使用与匿名构造函数相同的尖括号书写，被称为 *模式*，它们描述了我们解包主论点时预期找到的信息。给定存在上界 `f` 的假设 `ubf`，`rcases ubf with ⟨a, ubfa⟩` 将一个新的变量 `a` 添加到上下文中，以及具有给定属性的假设 `ubfa`。目标保持不变；*改变*的是，我们现在可以使用新的对象和新的假设来证明目标。这是数学中常见的推理方法：我们解包由某些假设断言或暗示存在的对象，然后使用它来建立其他事物的存在。

尝试使用这种方法来建立以下内容。你可能发现将上一节的一些示例转换为命名定理很有用，就像我们对 `fn_ub_add` 所做的那样，或者你可以直接将参数插入到证明中。

```py
example  (lbf  :  FnHasLb  f)  (lbg  :  FnHasLb  g)  :  FnHasLb  fun  x  ↦  f  x  +  g  x  :=  by
  sorry

example  {c  :  ℝ}  (ubf  :  FnHasUb  f)  (h  :  c  ≥  0)  :  FnHasUb  fun  x  ↦  c  *  f  x  :=  by
  sorry 
```

`rcases` 中的“r”代表“递归”，因为它允许我们使用任意复杂的模式来解包嵌套数据。`rintro` 策略是 `intro` 和 `rcases` 的组合：

```py
example  :  FnHasUb  f  →  FnHasUb  g  →  FnHasUb  fun  x  ↦  f  x  +  g  x  :=  by
  rintro  ⟨a,  ubfa⟩  ⟨b,  ubgb⟩
  exact  ⟨a  +  b,  fnUb_add  ubfa  ubgb⟩ 
```

实际上，Lean 还支持在表达式和证明项中使用模式匹配函数：

```py
example  :  FnHasUb  f  →  FnHasUb  g  →  FnHasUb  fun  x  ↦  f  x  +  g  x  :=
  fun  ⟨a,  ubfa⟩  ⟨b,  ubgb⟩  ↦  ⟨a  +  b,  fnUb_add  ubfa  ubgb⟩ 
```

在假设中解包信息是一项如此重要的任务，以至于 Lean 和 Mathlib 提供了多种方法来完成它。例如，`obtain` 策略提供了提示性语法：

```py
example  (ubf  :  FnHasUb  f)  (ubg  :  FnHasUb  g)  :  FnHasUb  fun  x  ↦  f  x  +  g  x  :=  by
  obtain  ⟨a,  ubfa⟩  :=  ubf
  obtain  ⟨b,  ubgb⟩  :=  ubg
  exact  ⟨a  +  b,  fnUb_add  ubfa  ubgb⟩ 
```

将第一个 `obtain` 指令视为将 `ubf` 的“内容”与给定的模式匹配，并将组件分配给命名变量。`rcases` 和 `obtain` 被说成是 `destruct` 它们的参数。

Lean 还支持与其它函数式编程语言类似的语法：

```py
example  (ubf  :  FnHasUb  f)  (ubg  :  FnHasUb  g)  :  FnHasUb  fun  x  ↦  f  x  +  g  x  :=  by
  cases  ubf
  case  intro  a  ubfa  =>
  cases  ubg
  case  intro  b  ubgb  =>
  exact  ⟨a  +  b,  fnUb_add  ubfa  ubgb⟩

example  (ubf  :  FnHasUb  f)  (ubg  :  FnHasUb  g)  :  FnHasUb  fun  x  ↦  f  x  +  g  x  :=  by
  cases  ubf
  next  a  ubfa  =>
  cases  ubg
  next  b  ubgb  =>
  exact  ⟨a  +  b,  fnUb_add  ubfa  ubgb⟩

example  (ubf  :  FnHasUb  f)  (ubg  :  FnHasUb  g)  :  FnHasUb  fun  x  ↦  f  x  +  g  x  :=  by
  match  ubf,  ubg  with
  |  ⟨a,  ubfa⟩,  ⟨b,  ubgb⟩  =>
  exact  ⟨a  +  b,  fnUb_add  ubfa  ubgb⟩

example  (ubf  :  FnHasUb  f)  (ubg  :  FnHasUb  g)  :  FnHasUb  fun  x  ↦  f  x  +  g  x  :=
  match  ubf,  ubg  with
  |  ⟨a,  ubfa⟩,  ⟨b,  ubgb⟩  =>
  ⟨a  +  b,  fnUb_add  ubfa  ubgb⟩ 
```

在第一个例子中，如果你将光标放在`cases ubf`之后，你会看到这个策略产生了一个单一的目标，Lean 将其标记为`intro`。（所选择的特定名称来自构建存在性陈述证明的公理原语的内部名称。）然后`case`策略命名了组件。第二个例子与第一个类似，只是使用`next`代替`case`意味着你可以避免提及`intro`。在最后两个例子中，单词`match`突出了我们在这里所做的是计算机科学家所说的“模式匹配”。请注意，第三个证明以`by`开始，之后`match`的策略版本期望箭头右侧有一个策略证明。最后一个例子是一个证明项：没有可见的策略。

在本书的剩余部分，我们将坚持使用`rcases`、`rintro`和`obtain`作为使用存在量词的首选方式。但看看备选语法也无妨，尤其是如果你有可能与计算机科学家为伍的话。

为了说明`rcases`可以用来证明的一种方式，我们证明了一个古老的数学难题：如果两个整数`x`和`y`都可以写成两个平方的和，那么它们的乘积`x * y`也可以。实际上，这个陈述对于任何交换环都成立，而不仅仅是整数。在下一个例子中，`rcases`同时展开两个存在量词。然后我们提供一个列表，包含将`x * y`表示为平方和所需的神奇值，并将其作为`use`语句的参数，并使用`ring`来验证它们是否有效。

```py
variable  {α  :  Type*}  [CommRing  α]

def  SumOfSquares  (x  :  α)  :=
  ∃  a  b,  x  =  a  ^  2  +  b  ^  2

theorem  sumOfSquares_mul  {x  y  :  α}  (sosx  :  SumOfSquares  x)  (sosy  :  SumOfSquares  y)  :
  SumOfSquares  (x  *  y)  :=  by
  rcases  sosx  with  ⟨a,  b,  xeq⟩
  rcases  sosy  with  ⟨c,  d,  yeq⟩
  rw  [xeq,  yeq]
  use  a  *  c  -  b  *  d,  a  *  d  +  b  *  c
  ring 
```

这个证明并没有提供很多洞见，但这里有一种激发它的方法。一个**高斯整数**是形如 $a + bi$ 的数，其中 $a$ 和 $b$ 是整数，且 $i = \sqrt{-1}$。根据定义，高斯整数 $a + bi$ 的**范数**是 $a² + b²$。因此，高斯整数的范数是平方和，任何平方和都可以用这种方式表示。上述定理反映了这样一个事实：高斯整数乘积的范数是它们范数的乘积：如果 $x$ 是 $a + bi$ 的范数，$y$ 是 $c + di$ 的范数，那么 $xy$ 就是 $(a + bi) (c + di)$ 的范数。我们晦涩的证明说明了这样一个事实：最易于形式化的证明并不总是最清晰的。在第 7.3 节中，我们将提供定义高斯整数并使用它们提供另一种证明的方法。

在存在量词内部展开方程并使用它来重写目标表达式的模式经常出现，以至于`rcases`策略提供了一个缩写：如果你用关键字`rfl`代替一个新标识符，`rcases`会自动进行重写（这个技巧与模式匹配 lambda 不兼容）。

```py
theorem  sumOfSquares_mul'  {x  y  :  α}  (sosx  :  SumOfSquares  x)  (sosy  :  SumOfSquares  y)  :
  SumOfSquares  (x  *  y)  :=  by
  rcases  sosx  with  ⟨a,  b,  rfl⟩
  rcases  sosy  with  ⟨c,  d,  rfl⟩
  use  a  *  c  -  b  *  d,  a  *  d  +  b  *  c
  ring 
```

就像全称量词一样，如果你知道如何找到它们，你可以在任何地方找到隐藏的存在量词。例如，可除性隐含地是一个“存在”陈述。

```py
example  (divab  :  a  ∣  b)  (divbc  :  b  ∣  c)  :  a  ∣  c  :=  by
  rcases  divab  with  ⟨d,  beq⟩
  rcases  divbc  with  ⟨e,  ceq⟩
  rw  [ceq,  beq]
  use  d  *  e;  ring 
```

再次强调，这为使用 `rcases` 和 `rfl` 提供了一个很好的环境。在上面的证明中试一试。感觉相当不错！

然后尝试证明以下内容：

```py
example  (divab  :  a  ∣  b)  (divac  :  a  ∣  c)  :  a  ∣  b  +  c  :=  by
  sorry 
```

对于另一个重要的例子，如果函数 $f : \alpha \to \beta$ 被称为 *外射*，那么对于域 $\alpha$ 中的每个 $y$，在陪域 $\beta$ 中都有一个 $x$，使得 $f(x) = y$。注意，这个陈述包括全称量词和存在量词，这就是为什么下一个例子同时使用了 `intro` 和 `use` 的原因。

```py
example  {c  :  ℝ}  :  Surjective  fun  x  ↦  x  +  c  :=  by
  intro  x
  use  x  -  c
  dsimp;  ring 
```

尝试使用定理 `mul_div_cancel₀` 自己做这个例子。

```py
example  {c  :  ℝ}  (h  :  c  ≠  0)  :  Surjective  fun  x  ↦  c  *  x  :=  by
  sorry 
```

在这一点上，值得提到的是，有一个策略 `field_simp`，它通常会以有用的方式消除分母。它可以与 `ring` 策略一起使用。

```py
example  (x  y  :  ℝ)  (h  :  x  -  y  ≠  0)  :  (x  ^  2  -  y  ^  2)  /  (x  -  y)  =  x  +  y  :=  by
  field_simp  [h]
  ring 
```

下一个例子通过应用到一个合适的值来使用外射假设。注意，你可以用 `rcases` 对任何表达式进行操作，而不仅仅是假设。

```py
example  {f  :  ℝ  →  ℝ}  (h  :  Surjective  f)  :  ∃  x,  f  x  ^  2  =  4  :=  by
  rcases  h  2  with  ⟨x,  hx⟩
  use  x
  rw  [hx]
  norm_num 
```

看看你是否可以使用这些方法来证明外射函数的复合仍然是外射的。

```py
variable  {α  :  Type*}  {β  :  Type*}  {γ  :  Type*}
variable  {g  :  β  →  γ}  {f  :  α  →  β}

example  (surjg  :  Surjective  g)  (surjf  :  Surjective  f)  :  Surjective  fun  x  ↦  g  (f  x)  :=  by
  sorry 
```  ## 3.3. 否定

符号 `¬` 用于表示否定，所以 `¬ x < y` 表示 `x` 不小于 `y`，`¬ x = y`（或等价地，`x ≠ y`）表示 `x` 不等于 `y`，而 `¬ ∃ z, x < z ∧ z < y` 表示不存在一个 `z` 在 `x` 和 `y` 之间。在 Lean 中，记法 `¬ A` 简写为 `A → False`，你可以将其视为说 `A` 导致矛盾。实际上，这意味着你已经知道如何处理否定：你可以通过引入一个假设 `h : A` 并证明 `False` 来证明 `¬ A`，如果你有 `h : ¬ A` 和 `h' : A`，那么将 `h` 应用到 `h'` 上会产生 `False`。

为了说明，考虑严格顺序的不可自反原理 `lt_irrefl`，它表示对于每个 `a`，我们有 `¬ a < a`。不对称原理 `lt_asymm` 表示我们有 `a < b → ¬ b < a`。让我们证明 `lt_asymm` 可以从 `lt_irrefl` 推出。

```py
example  (h  :  a  <  b)  :  ¬b  <  a  :=  by
  intro  h'
  have  :  a  <  a  :=  lt_trans  h  h'
  apply  lt_irrefl  a  this 
```

这个例子介绍了一些新技巧。首先，当你使用 `have` 而不提供标签时，Lean 使用名称 `this`，这提供了一个方便的方式来引用它。因为证明非常简短，所以我们提供了一个显式的证明项。但在这个证明中，你应该真正关注的是 `intro` 策略的结果，它留下一个 `False` 的目标，以及我们最终通过将 `lt_irrefl` 应用到一个 `a < a` 的证明上证明 `False` 的事实。

这里是另一个例子，它使用了上一节中定义的谓词 `FnHasUb`，该谓词表示一个函数有一个上界。

```py
example  (h  :  ∀  a,  ∃  x,  f  x  >  a)  :  ¬FnHasUb  f  :=  by
  intro  fnub
  rcases  fnub  with  ⟨a,  fnuba⟩
  rcases  h  a  with  ⟨x,  hx⟩
  have  :  f  x  ≤  a  :=  fnuba  x
  linarith 
```

记住，当目标从上下文中的线性方程和不等式得出时，使用 `linarith` 通常很方便。

看看你是否可以用类似的方法证明以下内容：

```py
example  (h  :  ∀  a,  ∃  x,  f  x  <  a)  :  ¬FnHasLb  f  :=
  sorry

example  :  ¬FnHasUb  fun  x  ↦  x  :=
  sorry 
```

Mathlib 提供了多个有用的定理，用于关联顺序和否定：

```py
#check  (not_le_of_gt  :  a  >  b  →  ¬a  ≤  b)
#check  (not_lt_of_ge  :  a  ≥  b  →  ¬a  <  b)
#check  (lt_of_not_ge  :  ¬a  ≥  b  →  a  <  b)
#check  (le_of_not_gt  :  ¬a  >  b  →  a  ≤  b) 
```

回忆谓词 `Monotone f`，它表示 `f` 是非递减的。使用你刚刚列举的一些定理来证明以下内容：

```py
example  (h  :  Monotone  f)  (h'  :  f  a  <  f  b)  :  a  <  b  :=  by
  sorry

example  (h  :  a  ≤  b)  (h'  :  f  b  <  f  a)  :  ¬Monotone  f  :=  by
  sorry 
```

如果我们将 `<` 替换为 `≤`，我们可以证明最后一段代码中的第一个例子是无法证明的。注意，我们可以通过给出一个反例来证明全称量化语句的否定。完成证明。

```py
example  :  ¬∀  {f  :  ℝ  →  ℝ},  Monotone  f  →  ∀  {a  b},  f  a  ≤  f  b  →  a  ≤  b  :=  by
  intro  h
  let  f  :=  fun  x  :  ℝ  ↦  (0  :  ℝ)
  have  monof  :  Monotone  f  :=  by  sorry
  have  h'  :  f  1  ≤  f  0  :=  le_refl  _
  sorry 
```

这个例子介绍了 `let` 策略，它向上下文中添加一个 *局部定义*。如果你在 `let` 命令后放置光标，在目标窗口中你会看到已经添加到上下文中的定义 `f : ℝ → ℝ := fun x ↦ 0`。当 Lean 需要时，它会展开 `f` 的定义。特别是，当我们使用 `le_refl` 证明 `f 1 ≤ f 0` 时，Lean 会将 `f 1` 和 `f 0` 简化为 `0`。

使用 `le_of_not_gt` 来证明以下内容：

```py
example  (x  :  ℝ)  (h  :  ∀  ε  >  0,  x  <  ε)  :  x  ≤  0  :=  by
  sorry 
```

在我们刚刚证明的许多证明中隐含的事实是，如果 `P` 是任何属性，说没有任何具有属性 `P` 的东西等同于说所有东西都没有属性 `P`，而说并非所有东西都有属性 `P` 等价于说有些东西没有属性 `P`。换句话说，以下四个蕴含都是有效的（但其中之一不能使用我们迄今为止所解释的方法来证明）：

```py
variable  {α  :  Type*}  (P  :  α  →  Prop)  (Q  :  Prop)

example  (h  :  ¬∃  x,  P  x)  :  ∀  x,  ¬P  x  :=  by
  sorry

example  (h  :  ∀  x,  ¬P  x)  :  ¬∃  x,  P  x  :=  by
  sorry

example  (h  :  ¬∀  x,  P  x)  :  ∃  x,  ¬P  x  :=  by
  sorry

example  (h  :  ∃  x,  ¬P  x)  :  ¬∀  x,  P  x  :=  by
  sorry 
```

第一个、第二个和第四个定理使用你已经看到的方法很容易证明。我们鼓励你尝试一下。然而，第三个定理更难证明，因为它从其不存在是矛盾的这一事实得出存在一个对象。这是一个 *经典* 数学推理的例子。我们可以通过以下方式使用反证法来证明第三个蕴含：

```py
example  (h  :  ¬∀  x,  P  x)  :  ∃  x,  ¬P  x  :=  by
  by_contra  h'
  apply  h
  intro  x
  show  P  x
  by_contra  h''
  exact  h'  ⟨x,  h''⟩ 
```

确保你理解这是如何工作的。`by_contra` 策略允许我们通过假设 `¬ Q` 并推导出矛盾来证明目标 `Q`。实际上，它等价于使用等价性 `not_not : ¬ ¬ Q ↔ Q`。确认你可以使用 `by_contra` 来证明这个等价的正向方向，而反向方向则遵循否定规则的普通规则。

```py
example  (h  :  ¬¬Q)  :  Q  :=  by
  sorry

example  (h  :  Q)  :  ¬¬Q  :=  by
  sorry 
```

使用反证法来证明以下内容，这是我们上面证明的一个蕴含的逆命题。（提示：首先使用 `intro`。）

```py
example  (h  :  ¬FnHasUb  f)  :  ∀  a,  ∃  x,  f  x  >  a  :=  by
  sorry 
```

处理带有前面否定词的复合语句通常很繁琐，并且将否定词推入等价形式的数学模式是一种常见的数学模式。为了方便这样做，Mathlib 提供了一个 `push_neg` 策略，它以这种方式重新表述目标。命令 `push_neg at h` 重新表述假设 `h`。

```py
example  (h  :  ¬∀  a,  ∃  x,  f  x  >  a)  :  FnHasUb  f  :=  by
  push_neg  at  h
  exact  h

example  (h  :  ¬FnHasUb  f)  :  ∀  a,  ∃  x,  f  x  >  a  :=  by
  dsimp  only  [FnHasUb,  FnUb]  at  h
  push_neg  at  h
  exact  h 
```

在第二个例子中，我们使用`dsimp`展开`FnHasUb`和`FnUb`的定义。（我们需要使用`dsimp`而不是`rw`来展开`FnUb`，因为它出现在量词的作用域中。）你可以验证在上述例子中，使用`¬∃ x, P x`和`¬∀ x, P x`时，`push_neg`策略做了预期的事情。即使不知道如何使用合取符号，你也应该能够使用`push_neg`来证明以下内容：

```py
example  (h  :  ¬Monotone  f)  :  ∃  x  y,  x  ≤  y  ∧  f  y  <  f  x  :=  by
  sorry 
```

Mathlib 还有一个策略，`contrapose`，它将目标`A → B`转换为`¬B → ¬A`。同样，给定一个从假设`h : A`证明`B`的目标，`contrapose h`让你有一个从假设`¬B`证明`¬A`的目标。使用`contrapose!`而不是`contrapose`将`push_neg`应用于目标和相关假设。

```py
example  (h  :  ¬FnHasUb  f)  :  ∀  a,  ∃  x,  f  x  >  a  :=  by
  contrapose!  h
  exact  h

example  (x  :  ℝ)  (h  :  ∀  ε  >  0,  x  ≤  ε)  :  x  ≤  0  :=  by
  contrapose!  h
  use  x  /  2
  constructor  <;>  linarith 
```

我们还没有解释`constructor`命令及其后的分号的使用，但将在下一节中解释。

我们以*ex falso*原则结束本节，该原则指出，任何东西都可以从矛盾中得出。在 Lean 中，这由`False.elim`表示，它为任何命题`P`建立了`False → P`。这看起来可能像一条奇怪的原则，但它出现的频率相当高。我们经常通过分情况证明定理，有时我们可以证明其中一种情况是矛盾的。在这种情况下，我们需要断言矛盾建立了目标，这样我们就可以继续下一个目标。（我们将在第 3.5 节中看到通过分情况推理的例子。）

Lean 提供了一些方法来关闭目标，一旦达到矛盾。

```py
example  (h  :  0  <  0)  :  a  >  37  :=  by
  exfalso
  apply  lt_irrefl  0  h

example  (h  :  0  <  0)  :  a  >  37  :=
  absurd  h  (lt_irrefl  0)

example  (h  :  0  <  0)  :  a  >  37  :=  by
  have  h'  :  ¬0  <  0  :=  lt_irrefl  0
  contradiction 
```

`exfalso`策略将当前目标替换为证明`False`的目标。给定`h : P`和`h' : ¬ P`，项`absurd h h'`建立了任何命题。最后，`contradiction`策略试图通过在假设中找到矛盾来关闭目标，例如形式为`h : P`和`h' : ¬ P`的一对。当然，在这个例子中，`linarith`也有效。## 3.4. 合取与双条件

你已经看到合取符号`∧`用于表达“和”。`constructor`策略允许你通过先证明`A`然后证明`B`来证明形式为`A ∧ B`的陈述。

```py
example  {x  y  :  ℝ}  (h₀  :  x  ≤  y)  (h₁  :  ¬y  ≤  x)  :  x  ≤  y  ∧  x  ≠  y  :=  by
  constructor
  ·  assumption
  intro  h
  apply  h₁
  rw  [h] 
```

在这个例子中，`assumption`策略告诉 Lean 找到一个假设来解决目标。注意，最后的`rw`通过应用`≤`的反射性来完成目标。以下是以匿名构造函数尖括号的形式执行先前例子的替代方法。第一种是先前证明的整洁证明项版本，它在`by`关键字处进入策略模式。

```py
example  {x  y  :  ℝ}  (h₀  :  x  ≤  y)  (h₁  :  ¬y  ≤  x)  :  x  ≤  y  ∧  x  ≠  y  :=
  ⟨h₀,  fun  h  ↦  h₁  (by  rw  [h])⟩

example  {x  y  :  ℝ}  (h₀  :  x  ≤  y)  (h₁  :  ¬y  ≤  x)  :  x  ≤  y  ∧  x  ≠  y  :=
  have  h  :  x  ≠  y  :=  by
  contrapose!  h₁
  rw  [h₁]
  ⟨h₀,  h⟩ 
```

*使用*合取而不是证明一个合取涉及展开两个部分的证明。你可以使用`rcases`策略来做到这一点，以及`rintro`或模式匹配的`fun`，所有这些都在与存在量词使用类似的方式中使用。

```py
example  {x  y  :  ℝ}  (h  :  x  ≤  y  ∧  x  ≠  y)  :  ¬y  ≤  x  :=  by
  rcases  h  with  ⟨h₀,  h₁⟩
  contrapose!  h₁
  exact  le_antisymm  h₀  h₁

example  {x  y  :  ℝ}  :  x  ≤  y  ∧  x  ≠  y  →  ¬y  ≤  x  :=  by
  rintro  ⟨h₀,  h₁⟩  h'
  exact  h₁  (le_antisymm  h₀  h')

example  {x  y  :  ℝ}  :  x  ≤  y  ∧  x  ≠  y  →  ¬y  ≤  x  :=
  fun  ⟨h₀,  h₁⟩  h'  ↦  h₁  (le_antisymm  h₀  h') 
```

与`obtain`策略类似，还有一个模式匹配的`have`：

```py
example  {x  y  :  ℝ}  (h  :  x  ≤  y  ∧  x  ≠  y)  :  ¬y  ≤  x  :=  by
  have  ⟨h₀,  h₁⟩  :=  h
  contrapose!  h₁
  exact  le_antisymm  h₀  h₁ 
```

与`rcases`不同，这里`have`策略将`h`留在上下文中。即使我们不会使用它们，我们再次有了计算机科学家的模式匹配语法：

```py
example  {x  y  :  ℝ}  (h  :  x  ≤  y  ∧  x  ≠  y)  :  ¬y  ≤  x  :=  by
  cases  h
  case  intro  h₀  h₁  =>
  contrapose!  h₁
  exact  le_antisymm  h₀  h₁

example  {x  y  :  ℝ}  (h  :  x  ≤  y  ∧  x  ≠  y)  :  ¬y  ≤  x  :=  by
  cases  h
  next  h₀  h₁  =>
  contrapose!  h₁
  exact  le_antisymm  h₀  h₁

example  {x  y  :  ℝ}  (h  :  x  ≤  y  ∧  x  ≠  y)  :  ¬y  ≤  x  :=  by
  match  h  with
  |  ⟨h₀,  h₁⟩  =>
  contrapose!  h₁
  exact  le_antisymm  h₀  h₁ 
```

与使用存在量词相比，你也可以通过编写`h.left`和`h.right`，或者等价地，`h.1`和`h.2`来提取假设`h : A ∧ B`的两个组成部分的证明。

```py
example  {x  y  :  ℝ}  (h  :  x  ≤  y  ∧  x  ≠  y)  :  ¬y  ≤  x  :=  by
  intro  h'
  apply  h.right
  exact  le_antisymm  h.left  h'

example  {x  y  :  ℝ}  (h  :  x  ≤  y  ∧  x  ≠  y)  :  ¬y  ≤  x  :=
  fun  h'  ↦  h.right  (le_antisymm  h.left  h') 
```

尝试使用这些技术来提出证明以下命题的各种方法：

```py
example  {m  n  :  ℕ}  (h  :  m  ∣  n  ∧  m  ≠  n)  :  m  ∣  n  ∧  ¬n  ∣  m  :=
  sorry 
```

你可以使用匿名构造函数、`rintro`和`rcases`来嵌套使用`∃`和`∧`。

```py
example  :  ∃  x  :  ℝ,  2  <  x  ∧  x  <  4  :=
  ⟨5  /  2,  by  norm_num,  by  norm_num⟩

example  (x  y  :  ℝ)  :  (∃  z  :  ℝ,  x  <  z  ∧  z  <  y)  →  x  <  y  :=  by
  rintro  ⟨z,  xltz,  zlty⟩
  exact  lt_trans  xltz  zlty

example  (x  y  :  ℝ)  :  (∃  z  :  ℝ,  x  <  z  ∧  z  <  y)  →  x  <  y  :=
  fun  ⟨z,  xltz,  zlty⟩  ↦  lt_trans  xltz  zlty 
```

你还可以使用`use`策略：

```py
example  :  ∃  x  :  ℝ,  2  <  x  ∧  x  <  4  :=  by
  use  5  /  2
  constructor  <;>  norm_num

example  :  ∃  m  n  :  ℕ,  4  <  m  ∧  m  <  n  ∧  n  <  10  ∧  Nat.Prime  m  ∧  Nat.Prime  n  :=  by
  use  5
  use  7
  norm_num

example  {x  y  :  ℝ}  :  x  ≤  y  ∧  x  ≠  y  →  x  ≤  y  ∧  ¬y  ≤  x  :=  by
  rintro  ⟨h₀,  h₁⟩
  use  h₀
  exact  fun  h'  ↦  h₁  (le_antisymm  h₀  h') 
```

在第一个例子中，`constructor`命令后面的分号告诉 Lean 对产生的两个目标都使用`norm_num`策略。

在 Lean 中，`A ↔ B`不是定义为`(A → B) ∧ (B → A)`，但它本来可以是，并且它的行为大致相同。你已经看到，你可以为`h : A ↔ B`的两个方向编写`h.mp`和`h.mpr`或`h.1`和`h.2`。你也可以使用`cases`和相关工具。为了证明一个如果且仅如果的陈述，你可以使用`constructor`或尖括号，就像你证明合取一样。

```py
example  {x  y  :  ℝ}  (h  :  x  ≤  y)  :  ¬y  ≤  x  ↔  x  ≠  y  :=  by
  constructor
  ·  contrapose!
  rintro  rfl
  rfl
  contrapose!
  exact  le_antisymm  h

example  {x  y  :  ℝ}  (h  :  x  ≤  y)  :  ¬y  ≤  x  ↔  x  ≠  y  :=
  ⟨fun  h₀  h₁  ↦  h₀  (by  rw  [h₁]),  fun  h₀  h₁  ↦  h₀  (le_antisymm  h  h₁)⟩ 
```

最后的证明项难以理解。记住，在编写类似的表达式时，你可以使用下划线来查看 Lean 期望的内容。

尝试使用你刚刚看到的各种技术和工具来证明以下命题：

```py
example  {x  y  :  ℝ}  :  x  ≤  y  ∧  ¬y  ≤  x  ↔  x  ≤  y  ∧  x  ≠  y  :=
  sorry 
```

对于一个更有趣的练习，证明对于任何两个实数`x`和`y`，如果`x² + y² = 0`，则`x = 0`且`y = 0`。我们建议使用`linarith`、`pow_two_nonneg`和`pow_eq_zero`证明一个辅助引理。

```py
theorem  aux  {x  y  :  ℝ}  (h  :  x  ^  2  +  y  ^  2  =  0)  :  x  =  0  :=
  have  h'  :  x  ^  2  =  0  :=  by  sorry
  pow_eq_zero  h'

example  (x  y  :  ℝ)  :  x  ^  2  +  y  ^  2  =  0  ↔  x  =  0  ∧  y  =  0  :=
  sorry 
```

在 Lean 中，双条件有两种生活。你可以将其视为合取并分别使用其两部分。但 Lean 也知道它是一个命题之间的自反、对称和传递关系，你也可以使用`calc`和`rw`。将一个陈述重写为等价陈述通常很方便。在下一个例子中，我们使用`abs_lt`将形式为`|x| < y`的表达式替换为等价表达式`- y < x ∧ x < y`，在下一个例子中，我们使用`Nat.dvd_gcd_iff`将形式为`m ∣ Nat.gcd n k`的表达式替换为等价表达式`m ∣ n ∧ m ∣ k`。

```py
example  (x  :  ℝ)  :  |x  +  3|  <  5  →  -8  <  x  ∧  x  <  2  :=  by
  rw  [abs_lt]
  intro  h
  constructor  <;>  linarith

example  :  3  ∣  Nat.gcd  6  15  :=  by
  rw  [Nat.dvd_gcd_iff]
  constructor  <;>  norm_num 
```

看看你是否可以使用`rw`与下面的定理一起提供否定不是非递增函数的简短证明。（注意，`push_neg`不会为你展开定义，所以定理证明中的`rw [Monotone]`是必需的。）

```py
theorem  not_monotone_iff  {f  :  ℝ  →  ℝ}  :  ¬Monotone  f  ↔  ∃  x  y,  x  ≤  y  ∧  f  x  >  f  y  :=  by
  rw  [Monotone]
  push_neg
  rfl

example  :  ¬Monotone  fun  x  :  ℝ  ↦  -x  :=  by
  sorry 
```

本节剩余的练习旨在让你在合取和双条件方面获得更多实践。记住，*偏序*是一个传递的、自反的和反对称的二元关系。有时会出现一个更弱的概念：*预序*只是一个自反的和传递的关系。对于任何预序`≤`，Lean 通过`a < b ↔ a ≤ b ∧ ¬ b ≤ a`来公理化相关的严格预序。证明如果`≤`是一个偏序，那么`a < b`等价于`a ≤ b ∧ a ≠ b`：

```py
variable  {α  :  Type*}  [PartialOrder  α]
variable  (a  b  :  α)

example  :  a  <  b  ↔  a  ≤  b  ∧  a  ≠  b  :=  by
  rw  [lt_iff_le_not_ge]
  sorry 
```

除了逻辑运算之外，你不需要比 `le_refl` 和 `le_trans` 更多的东西。证明即使 `≤` 只被假设为偏序，我们也可以证明严格顺序是不可自反的和传递的。在第二个例子中，为了方便起见，我们使用简化器而不是 `rw` 来用 `≤` 和 `¬` 表达 `<`。我们稍后会回到简化器，但在这里我们只依赖于这样一个事实，即它将反复使用指定的引理，即使它需要实例化为不同的值。

```py
variable  {α  :  Type*}  [Preorder  α]
variable  (a  b  c  :  α)

example  :  ¬a  <  a  :=  by
  rw  [lt_iff_le_not_ge]
  sorry

example  :  a  <  b  →  b  <  c  →  a  <  c  :=  by
  simp  only  [lt_iff_le_not_ge]
  sorry 
```  ## 3.5\. 析取

证明析取 `A ∨ B` 的规范方法是证明 `A` 或证明 `B`。`left` 策略选择 `A`，而 `right` 策略选择 `B`。

```py
variable  {x  y  :  ℝ}

example  (h  :  y  >  x  ^  2)  :  y  >  0  ∨  y  <  -1  :=  by
  left
  linarith  [pow_two_nonneg  x]

example  (h  :  -y  >  x  ^  2  +  1)  :  y  >  0  ∨  y  <  -1  :=  by
  right
  linarith  [pow_two_nonneg  x] 
```

我们不能使用匿名构造函数来构造一个“或”的证明，因为 Lean 需要猜测我们正在尝试证明哪个析取支。当我们编写证明项时，我们可以使用 `Or.inl` 和 `Or.inr` 来明确地做出选择。在这里，`inl` 是“引入左”的缩写，而 `inr` 是“引入右”的缩写。

```py
example  (h  :  y  >  0)  :  y  >  0  ∨  y  <  -1  :=
  Or.inl  h

example  (h  :  y  <  -1)  :  y  >  0  ∨  y  <  -1  :=
  Or.inr  h 
```

通过证明析取的一侧或另一侧来证明析取可能看起来很奇怪。在实践中，哪个情况成立通常取决于假设和数据中隐含或显含的情况区分。`rcases` 策略允许我们利用形式为 `A ∨ B` 的假设。与 `rcases` 与合取或存在量词的使用相比，这里 `rcases` 策略产生 *两个* 目标。这两个目标有相同的结论，但在第一种情况下，假设 `A` 为真，在第二种情况下，假设 `B` 为真。换句话说，正如其名称所暗示的，`rcases` 策略通过分情况进行证明。像往常一样，我们可以告诉 Lean 使用哪些名称来命名假设。在下一个例子中，我们告诉 Lean 在每个分支上使用名称 `h`。

```py
example  :  x  <  |y|  →  x  <  y  ∨  x  <  -y  :=  by
  rcases  le_or_gt  0  y  with  h  |  h
  ·  rw  [abs_of_nonneg  h]
  intro  h;  left;  exact  h
  ·  rw  [abs_of_neg  h]
  intro  h;  right;  exact  h 
```

注意，模式从合取的情况中的 `⟨h₀, h₁⟩` 变为析取的情况中的 `h₀ | h₁`。将第一个模式视为与包含 *两个* `h₀` 和 `h₁` 的数据匹配，而第二个模式，带有竖线，与包含 *一个* `h₀` 或 `h₁` 的数据匹配。在这种情况下，因为两个目标是分开的，所以我们选择在每个情况下使用相同的名称，即 `h`。

绝对值函数被定义为这样的方式，我们可以立即证明 `x ≥ 0` 蕴含 `|x| = x`（这是定理 `abs_of_nonneg`），以及 `x < 0` 蕴含 `|x| = -x`（这是 `abs_of_neg`）。表达式 `le_or_gt 0 x` 建立了 `0 ≤ x ∨ x < 0`，允许我们在这两种情况下进行拆分。

Lean 还支持计算机科学家的析取模式匹配语法。现在 `cases` 策略更具吸引力，因为它允许我们为每个 `case` 命名，并为引入的假设命名，使其更接近使用位置。

```py
example  :  x  <  |y|  →  x  <  y  ∨  x  <  -y  :=  by
  cases  le_or_gt  0  y
  case  inl  h  =>
  rw  [abs_of_nonneg  h]
  intro  h;  left;  exact  h
  case  inr  h  =>
  rw  [abs_of_neg  h]
  intro  h;  right;  exact  h 
```

`inl` 和 `inr` 的名字是“intro left”和“intro right”的简称。使用 `case` 的优点是你可以按任意顺序证明情况；Lean 使用标签来找到相关的目标。如果你不在乎这一点，你可以使用 `next`，或者 `match`，甚至是一个模式匹配的 `have`。

```py
example  :  x  <  |y|  →  x  <  y  ∨  x  <  -y  :=  by
  cases  le_or_gt  0  y
  next  h  =>
  rw  [abs_of_nonneg  h]
  intro  h;  left;  exact  h
  next  h  =>
  rw  [abs_of_neg  h]
  intro  h;  right;  exact  h

example  :  x  <  |y|  →  x  <  y  ∨  x  <  -y  :=  by
  match  le_or_gt  0  y  with
  |  Or.inl  h  =>
  rw  [abs_of_nonneg  h]
  intro  h;  left;  exact  h
  |  Or.inr  h  =>
  rw  [abs_of_neg  h]
  intro  h;  right;  exact  h 
```

在 `match` 的情况下，我们需要使用证明析取的规范方式 `Or.inl` 和 `Or.inr` 的全名。在这本教科书中，我们将通常使用 `rcases` 来分割析取的情况。

尝试使用下一个片段中的前两个定理来证明三角不等式。它们在 Mathlib 中的名称与数学中的名称相同。

```py
namespace  MyAbs

theorem  le_abs_self  (x  :  ℝ)  :  x  ≤  |x|  :=  by
  sorry

theorem  neg_le_abs_self  (x  :  ℝ)  :  -x  ≤  |x|  :=  by
  sorry

theorem  abs_add  (x  y  :  ℝ)  :  |x  +  y|  ≤  |x|  +  |y|  :=  by
  sorry 
```

如果你喜欢这些（故意为之）并且想要更多关于析取的练习，试试这些。

```py
theorem  lt_abs  :  x  <  |y|  ↔  x  <  y  ∨  x  <  -y  :=  by
  sorry

theorem  abs_lt  :  |x|  <  y  ↔  -y  <  x  ∧  x  <  y  :=  by
  sorry 
```

你也可以使用 `rcases` 和 `rintro` 与嵌套析取一起使用。当这些导致具有多个目标的真正情况分割时，每个新目标的模式由一个垂直线分隔。

```py
example  {x  :  ℝ}  (h  :  x  ≠  0)  :  x  <  0  ∨  x  >  0  :=  by
  rcases  lt_trichotomy  x  0  with  xlt  |  xeq  |  xgt
  ·  left
  exact  xlt
  ·  contradiction
  ·  right;  exact  xgt 
```

你仍然可以嵌套模式并使用 `rfl` 关键字来替换等式：

```py
example  {m  n  k  :  ℕ}  (h  :  m  ∣  n  ∨  m  ∣  k)  :  m  ∣  n  *  k  :=  by
  rcases  h  with  ⟨a,  rfl⟩  |  ⟨b,  rfl⟩
  ·  rw  [mul_assoc]
  apply  dvd_mul_right
  ·  rw  [mul_comm,  mul_assoc]
  apply  dvd_mul_right 
```

看看你是否能用一行（长）代码证明以下内容。使用 `rcases` 来展开假设并按情况分割，并使用分号和 `linarith` 来解决每个分支。

```py
example  {z  :  ℝ}  (h  :  ∃  x  y,  z  =  x  ^  2  +  y  ^  2  ∨  z  =  x  ^  2  +  y  ^  2  +  1)  :  z  ≥  0  :=  by
  sorry 
```

在实数上，方程 `x * y = 0` 告诉我们 `x = 0` 或 `y = 0`。在 Mathlib 中，这个事实被称为 `eq_zero_or_eq_zero_of_mul_eq_zero`，它是如何从析取中产生的一个很好的例子。看看你是否可以用它来证明以下内容：

```py
example  {x  :  ℝ}  (h  :  x  ^  2  =  1)  :  x  =  1  ∨  x  =  -1  :=  by
  sorry

example  {x  y  :  ℝ}  (h  :  x  ^  2  =  y  ^  2)  :  x  =  y  ∨  x  =  -y  :=  by
  sorry 
```

记住，你可以使用 `ring` 策略来帮助计算。

在任意环 $R$ 中，一个元素 $x$，使得对于某个非零的 $y$ 有 $x y = 0$，被称为 *左零因子*，一个元素 $x$，使得对于某个非零的 $y$ 有 $y x = 0$，被称为 *右零因子*，而一个既是左零因子又是右零因子的元素被称为简单的 *零因子*。定理 `eq_zero_or_eq_zero_of_mul_eq_zero` 表明实数没有非平凡零因子。具有这种性质的交换环被称为 *整环*。你上面两个定理的证明在任意整环中同样有效：

```py
variable  {R  :  Type*}  [CommRing  R]  [IsDomain  R]
variable  (x  y  :  R)

example  (h  :  x  ^  2  =  1)  :  x  =  1  ∨  x  =  -1  :=  by
  sorry

example  (h  :  x  ^  2  =  y  ^  2)  :  x  =  y  ∨  x  =  -y  :=  by
  sorry 
```

事实上，如果你小心，你可以不使用乘法的交换性来证明第一个定理。在这种情况下，只需假设 `R` 是一个 `Ring` 而不是 `CommRing`。

有时在证明中，我们想要根据某个陈述是否为真来分割情况。对于任何命题 `P`，我们可以使用 `em P : P ∨ ¬ P`。`em` 的名字是“excluded middle”的简称。

```py
example  (P  :  Prop)  :  ¬¬P  →  P  :=  by
  intro  h
  cases  em  P
  ·  assumption
  ·  contradiction 
```

或者，你也可以使用 `by_cases` 策略。

```py
example  (P  :  Prop)  :  ¬¬P  →  P  :=  by
  intro  h
  by_cases  h'  :  P
  ·  assumption
  contradiction 
```

注意到 `by_cases` 策略允许你为每个分支中引入的假设指定一个标签，在这种情况下，一个分支是 `h' : P`，另一个分支是 `h' : ¬ P`。如果你省略了标签，Lean 默认使用 `h`。尝试使用 `by_cases` 来证明以下等价性，以建立其中一个方向。

```py
example  (P  Q  :  Prop)  :  P  →  Q  ↔  ¬P  ∨  Q  :=  by
  sorry 
```  ## 3.6\. 序列与收敛

现在，我们已经有足够的技能来做一些真正的数学了。在 Lean 中，我们可以将实数序列 $s_0, s_1, s_2, \ldots$ 表示为一个函数 `s : ℕ → ℝ`。这样的序列被称为*收敛*到数字 $a$，如果对于每一个 $\varepsilon > 0$，都有一个点，在此点之后序列始终保持在 $a$ 的 $\varepsilon$ 范围内，也就是说，存在一个数字 $N$，使得对于每一个 $n \ge N$，$| s_n - a | < \varepsilon$。在 Lean 中，我们可以这样表示：

```py
def  ConvergesTo  (s  :  ℕ  →  ℝ)  (a  :  ℝ)  :=
  ∀  ε  >  0,  ∃  N,  ∀  n  ≥  N,  |s  n  -  a|  <  ε 
```

符号 `∀ ε > 0, ...` 是 `∀ ε, ε > 0 → ...` 的方便缩写，同样，`∀ n ≥ N, ...` 缩写为 `∀ n, n ≥ N → ...`。并且记住，`ε > 0`，反过来，定义为 `0 < ε`，而 `n ≥ N` 定义为 `N ≤ n`。

在本节中，我们将建立一些收敛的性质。但首先，我们将讨论三个用于处理等式的策略，这些策略将非常有用。第一个，`ext` 策略，为我们提供了一种证明两个函数相等的方法。设 $f(x) = x + 1$ 和 $g(x) = 1 + x$ 是从实数到实数的函数。当然，$f = g$，因为它们对每个 $x$ 返回相同的值。`ext` 策略使我们能够通过证明在所有参数值上它们的值都相同来证明函数之间的等式。

```py
example  :  (fun  x  y  :  ℝ  ↦  (x  +  y)  ^  2)  =  fun  x  y  :  ℝ  ↦  x  ^  2  +  2  *  x  *  y  +  y  ^  2  :=  by
  ext
  ring 
```

我们稍后会看到，`ext` 实际上更加通用，并且也可以指定出现变量的名称。例如，你可以在上面的证明中将 `ext` 替换为 `ext u v`。第二个策略，`congr` 策略，允许我们通过协调不同的部分来证明两个表达式之间的等式：

```py
example  (a  b  :  ℝ)  :  |a|  =  |a  -  b  +  b|  :=  by
  congr
  ring 
```

在这里，`congr` 策略剥去了每一侧的 `abs`，留下我们证明 `a = a - b + b`。

最后，`convert` 策略用于在定理的结论与目标不完全匹配时将定理应用于目标。例如，假设我们想从 `1 < a` 证明 `a < a * a`。库中的一个定理 `mul_lt_mul_right` 将允许我们证明 `1 * a < a * a`。一种可能性是反向工作并重写目标，使其具有那种形式。相反，`convert` 策略允许我们按照原样应用定理，并留下证明目标匹配所需方程的任务。

```py
example  {a  :  ℝ}  (h  :  1  <  a)  :  a  <  a  *  a  :=  by
  convert  (mul_lt_mul_right  _).2  h
  ·  rw  [one_mul]
  exact  lt_trans  zero_lt_one  h 
```

这个例子说明了另一个有用的技巧：当我们应用一个带有下划线的表达式，而 Lean 无法自动填充它时，它就简单地将其留给我们作为另一个目标。

下面的例子表明，任何常数序列 $a, a, a, \ldots$ 都会收敛。

```py
theorem  convergesTo_const  (a  :  ℝ)  :  ConvergesTo  (fun  x  :  ℕ  ↦  a)  a  :=  by
  intro  ε  εpos
  use  0
  intro  n  nge
  rw  [sub_self,  abs_zero]
  apply  εpos 
```

Lean 有一个策略，`simp`，它经常可以节省我们手动执行 `rw [sub_self, abs_zero]` 等步骤的麻烦。我们很快就会告诉你更多关于它的信息。

对于一个更有趣的定理，让我们证明如果`s`收敛到`a`且`t`收敛到`b`，那么`fun n ↦ s n + t n`收敛到`a + b`。在开始写正式证明之前，有一个清晰的笔和纸证明是有帮助的。给定大于`0`的`ε`，想法是使用假设来获得一个`Ns`，这样在这一点之后，`s`就在`a`的`ε / 2`范围内，以及一个`Nt`，这样在这一点之后，`t`就在`b`的`ε / 2`范围内。然后，每当`n`大于或等于`Ns`和`Nt`的最大值时，序列`fun n ↦ s n + t n`应该在`a + b`的`ε`范围内。以下示例开始实施这一策略。看看你是否能完成它。

```py
theorem  convergesTo_add  {s  t  :  ℕ  →  ℝ}  {a  b  :  ℝ}
  (cs  :  ConvergesTo  s  a)  (ct  :  ConvergesTo  t  b)  :
  ConvergesTo  (fun  n  ↦  s  n  +  t  n)  (a  +  b)  :=  by
  intro  ε  εpos
  dsimp  -- this line is not needed but cleans up the goal a bit.
  have  ε2pos  :  0  <  ε  /  2  :=  by  linarith
  rcases  cs  (ε  /  2)  ε2pos  with  ⟨Ns,  hs⟩
  rcases  ct  (ε  /  2)  ε2pos  with  ⟨Nt,  ht⟩
  use  max  Ns  Nt
  sorry 
```

作为提示，你可以使用`le_of_max_le_left`和`le_of_max_le_right`，以及`norm_num`可以证明`ε / 2 + ε / 2 = ε`。此外，使用`congr`策略来证明`|s n + t n - (a + b)|`等于`|(s n - a) + (t n - b)|`是有帮助的，因为这样你就可以使用三角不等式。注意，我们标记了所有变量`s`、`t`、`a`和`b`为隐含的，因为它们可以从假设中推断出来。

用乘法代替加法来证明同一个定理是棘手的。我们将通过首先证明一些辅助命题来达到这个目标。看看你是否也能完成下一个证明，它表明如果`s`收敛到`a`，那么`fun n ↦ c * s n`收敛到`c * a`。根据`c`是否等于零来分情况讨论是有帮助的。我们已经处理了零的情况，并留下你来证明在`c`非零的额外假设下得到的结果。

```py
theorem  convergesTo_mul_const  {s  :  ℕ  →  ℝ}  {a  :  ℝ}  (c  :  ℝ)  (cs  :  ConvergesTo  s  a)  :
  ConvergesTo  (fun  n  ↦  c  *  s  n)  (c  *  a)  :=  by
  by_cases  h  :  c  =  0
  ·  convert  convergesTo_const  0
  ·  rw  [h]
  ring
  rw  [h]
  ring
  have  acpos  :  0  <  |c|  :=  abs_pos.mpr  h
  sorry 
```

下一个定理也是独立有趣的：它表明收敛序列最终在绝对值上是有界的。我们已经为你奠定了基础；看看你是否能完成它。

```py
theorem  exists_abs_le_of_convergesTo  {s  :  ℕ  →  ℝ}  {a  :  ℝ}  (cs  :  ConvergesTo  s  a)  :
  ∃  N  b,  ∀  n,  N  ≤  n  →  |s  n|  <  b  :=  by
  rcases  cs  1  zero_lt_one  with  ⟨N,  h⟩
  use  N,  |a|  +  1
  sorry 
```

事实上，这个定理可以被加强，断言存在一个对所有`n`值都成立的界限`b`。但这个版本对我们来说已经足够强大，我们将在本节末尾看到它更普遍地成立。

下一个引理是辅助性的：我们证明如果`s`收敛到`a`且`t`收敛到`0`，那么`fun n ↦ s n * t n`收敛到`0`。为此，我们使用前面的定理找到一个`B`，它从某个点`N₀`开始限制`s`。看看你是否能理解我们所概述的策略并完成证明。

```py
theorem  aux  {s  t  :  ℕ  →  ℝ}  {a  :  ℝ}  (cs  :  ConvergesTo  s  a)  (ct  :  ConvergesTo  t  0)  :
  ConvergesTo  (fun  n  ↦  s  n  *  t  n)  0  :=  by
  intro  ε  εpos
  dsimp
  rcases  exists_abs_le_of_convergesTo  cs  with  ⟨N₀,  B,  h₀⟩
  have  Bpos  :  0  <  B  :=  lt_of_le_of_lt  (abs_nonneg  _)  (h₀  N₀  (le_refl  _))
  have  pos₀  :  ε  /  B  >  0  :=  div_pos  εpos  Bpos
  rcases  ct  _  pos₀  with  ⟨N₁,  h₁⟩
  sorry 
```

如果你已经走到这一步，恭喜你！我们现在已经接近我们的定理了。以下证明完成了它。

```py
theorem  convergesTo_mul  {s  t  :  ℕ  →  ℝ}  {a  b  :  ℝ}
  (cs  :  ConvergesTo  s  a)  (ct  :  ConvergesTo  t  b)  :
  ConvergesTo  (fun  n  ↦  s  n  *  t  n)  (a  *  b)  :=  by
  have  h₁  :  ConvergesTo  (fun  n  ↦  s  n  *  (t  n  +  -b))  0  :=  by
  apply  aux  cs
  convert  convergesTo_add  ct  (convergesTo_const  (-b))
  ring
  have  :=  convergesTo_add  h₁  (convergesTo_mul_const  b  cs)
  convert  convergesTo_add  h₁  (convergesTo_mul_const  b  cs)  using  1
  ·  ext;  ring
  ring 
```

对于另一个具有挑战性的练习，尝试填写以下关于极限唯一性的证明草稿。（如果你感到自信，你可以删除证明草稿，并尝试从头开始证明。）

```py
theorem  convergesTo_unique  {s  :  ℕ  →  ℝ}  {a  b  :  ℝ}
  (sa  :  ConvergesTo  s  a)  (sb  :  ConvergesTo  s  b)  :
  a  =  b  :=  by
  by_contra  abne
  have  :  |a  -  b|  >  0  :=  by  sorry
  let  ε  :=  |a  -  b|  /  2
  have  εpos  :  ε  >  0  :=  by
  change  |a  -  b|  /  2  >  0
  linarith
  rcases  sa  ε  εpos  with  ⟨Na,  hNa⟩
  rcases  sb  ε  εpos  with  ⟨Nb,  hNb⟩
  let  N  :=  max  Na  Nb
  have  absa  :  |s  N  -  a|  <  ε  :=  by  sorry
  have  absb  :  |s  N  -  b|  <  ε  :=  by  sorry
  have  :  |a  -  b|  <  |a  -  b|  :=  by  sorry
  exact  lt_irrefl  _  this 
```

我们在本节结束时注意到我们的证明可以推广。例如，我们使用的自然数的唯一属性是它们的结构携带一个具有`min`和`max`的偏序。你可以检查，如果你在所有地方用任何线性序`α`替换`ℕ`，一切仍然有效。

```py
variable  {α  :  Type*}  [LinearOrder  α]

def  ConvergesTo'  (s  :  α  →  ℝ)  (a  :  ℝ)  :=
  ∀  ε  >  0,  ∃  N,  ∀  n  ≥  N,  |s  n  -  a|  <  ε 
```

在第 11.1 节中，我们将看到 Mathlib 有处理收敛性的机制，这些机制在更广泛的范围内，不仅抽象了定义域和值域的特定特征，而且还抽象了不同类型的收敛。## 3.1. 蕴含和全称量词

考虑`#check`之后的陈述：

```py
#check  ∀  x  :  ℝ,  0  ≤  x  →  |x|  =  x 
```

用文字来说，我们会说“对于每一个实数`x`，如果`0 ≤ x`，那么`x`的绝对值等于`x`”。我们也可以有更复杂的陈述，例如：

```py
#check  ∀  x  y  ε  :  ℝ,  0  <  ε  →  ε  ≤  1  →  |x|  <  ε  →  |y|  <  ε  →  |x  *  y|  <  ε 
```

用文字来说，我们会说“对于每一个`x`、`y`和`ε`，如果`0 < ε ≤ 1`，`x`的绝对值小于`ε`，`y`的绝对值小于`ε`，那么`x * y`的绝对值小于`ε`。”在 Lean 中，在一系列蕴含中，有隐式的括号分组到右边。所以上面的表达式意味着“如果`0 < ε`，那么如果`ε ≤ 1`，那么如果`|x| < ε`……”因此，这个表达式表明所有假设共同蕴含结论。

你已经看到，尽管这个陈述中的全称量词在对象上取值，而蕴含箭头引入了假设，但 Lean 以非常相似的方式处理这两者。特别是，如果你已经证明了这种形式的定理，你可以以相同的方式将其应用于对象和假设。我们将以下陈述作为例子，稍后我们将帮助你证明它：

```py
theorem  my_lemma  :  ∀  x  y  ε  :  ℝ,  0  <  ε  →  ε  ≤  1  →  |x|  <  ε  →  |y|  <  ε  →  |x  *  y|  <  ε  :=
  sorry

section
variable  (a  b  δ  :  ℝ)
variable  (h₀  :  0  <  δ)  (h₁  :  δ  ≤  1)
variable  (ha  :  |a|  <  δ)  (hb  :  |b|  <  δ)

#check  my_lemma  a  b  δ
#check  my_lemma  a  b  δ  h₀  h₁
#check  my_lemma  a  b  δ  h₀  h₁  ha  hb

end 
```

你也已经看到，在 Lean 中，当量化的变量可以从后续的假设中推断出来时，通常使用花括号使量化的变量隐式。当我们这样做时，我们只需将引理应用于假设，而无需提及对象。

```py
theorem  my_lemma2  :  ∀  {x  y  ε  :  ℝ},  0  <  ε  →  ε  ≤  1  →  |x|  <  ε  →  |y|  <  ε  →  |x  *  y|  <  ε  :=
  sorry

section
variable  (a  b  δ  :  ℝ)
variable  (h₀  :  0  <  δ)  (h₁  :  δ  ≤  1)
variable  (ha  :  |a|  <  δ)  (hb  :  |b|  <  δ)

#check  my_lemma2  h₀  h₁  ha  hb

end 
```

在这个阶段，你也知道，如果你使用`apply`策略将`my_lemma`应用于形式为`|a * b| < δ`的目标，你将面临新的目标，需要你证明每个假设。

要证明这样的陈述，使用`intro`策略。看看它在下面的例子中做了什么：

```py
theorem  my_lemma3  :
  ∀  {x  y  ε  :  ℝ},  0  <  ε  →  ε  ≤  1  →  |x|  <  ε  →  |y|  <  ε  →  |x  *  y|  <  ε  :=  by
  intro  x  y  ε  epos  ele1  xlt  ylt
  sorry 
```

我们可以为全称量化的变量使用任何我们想要的名称；它们不必是`x`、`y`和`ε`。请注意，即使它们被标记为隐式，我们也必须引入这些变量：使它们隐式意味着在写一个表达式*使用*`my_lemma`时我们可以省略它们，但它们仍然是我们正在证明的陈述的一个基本部分。在`intro`命令之后，目标是如果我们在冒号之前列出了所有变量和假设，它将是什么，就像我们在上一节中做的那样。一会儿我们将看到为什么有时在证明开始后引入变量和假设是必要的。

为了帮助你证明引理，我们将从以下内容开始：

```py
theorem  my_lemma4  :
  ∀  {x  y  ε  :  ℝ},  0  <  ε  →  ε  ≤  1  →  |x|  <  ε  →  |y|  <  ε  →  |x  *  y|  <  ε  :=  by
  intro  x  y  ε  epos  ele1  xlt  ylt
  calc
  |x  *  y|  =  |x|  *  |y|  :=  sorry
  _  ≤  |x|  *  ε  :=  sorry
  _  <  1  *  ε  :=  sorry
  _  =  ε  :=  sorry 
```

使用定理 `abs_mul`、`mul_le_mul`、`abs_nonneg`、`mul_lt_mul_right` 和 `one_mul` 完成证明。记住，你可以使用 Ctrl-space 完成提示（或在 Mac 上使用 Cmd-space 完成提示）来找到这样的定理。还要记住，你可以使用 `.mp` 和 `.mpr` 或 `.1` 和 `.2` 来提取双向条件语句的两个方向。

全称量词通常隐藏在定义中，当需要时 Lean 会展开定义来揭示它们。例如，让我们定义两个谓词，`FnUb f a` 和 `FnLb f a`，其中 `f` 是从实数到实数的函数，而 `a` 是一个实数。第一个谓词表示 `a` 是 `f` 的值的上界，第二个谓词表示 `a` 是 `f` 的值的下界。

```py
def  FnUb  (f  :  ℝ  →  ℝ)  (a  :  ℝ)  :  Prop  :=
  ∀  x,  f  x  ≤  a

def  FnLb  (f  :  ℝ  →  ℝ)  (a  :  ℝ)  :  Prop  :=
  ∀  x,  a  ≤  f  x 
```

在下一个例子中，`fun x ↦ f x + g x` 是将 `x` 映射到 `f x + g x` 的函数。从表达式 `f x + g x` 到这个函数的转换在类型理论中称为 lambda 抽象。

```py
example  (hfa  :  FnUb  f  a)  (hgb  :  FnUb  g  b)  :  FnUb  (fun  x  ↦  f  x  +  g  x)  (a  +  b)  :=  by
  intro  x
  dsimp
  apply  add_le_add
  apply  hfa
  apply  hgb 
```

将 `intro` 应用于目标 `FnUb (fun x ↦ f x + g x) (a + b)` 迫使 Lean 展开对 `FnUb` 的定义并引入 `x` 作为全称量词。目标变为 `(fun (x : ℝ) ↦ f x + g x) x ≤ a + b`。但是将 `(fun x ↦ f x + g x)` 应用于 `x` 应该得到 `f x + g x`，而 `dsimp` 命令执行了这种简化。（“d”代表“定义性的。”）你可以删除该命令，证明仍然有效；Lean 无论如何都必须执行这种收缩以使下一个 `apply` 有意义。`dsimp` 命令只是使目标更易于阅读，并帮助我们确定下一步该做什么。另一个选择是使用 `change` 策略，通过编写 `change f x + g x ≤ a + b` 来实现。这有助于使证明更易于阅读，并让你对目标如何转换有更多的控制。

证明的其余部分是常规的。最后的两个 `apply` 命令迫使 Lean 在假设中展开 `FnUb` 的定义。尝试执行类似的证明：

```py
example  (hfa  :  FnLb  f  a)  (hgb  :  FnLb  g  b)  :  FnLb  (fun  x  ↦  f  x  +  g  x)  (a  +  b)  :=
  sorry

example  (nnf  :  FnLb  f  0)  (nng  :  FnLb  g  0)  :  FnLb  (fun  x  ↦  f  x  *  g  x)  0  :=
  sorry

example  (hfa  :  FnUb  f  a)  (hgb  :  FnUb  g  b)  (nng  :  FnLb  g  0)  (nna  :  0  ≤  a)  :
  FnUb  (fun  x  ↦  f  x  *  g  x)  (a  *  b)  :=
  sorry 
```

尽管我们已经为从实数到实数的函数定义了 `FnUb` 和 `FnLb`，但你应该认识到定义和证明要普遍得多。这些定义适用于任何两个类型之间的函数，其中目标域上有序的概念。检查定理 `add_le_add` 的类型显示，它适用于任何“有序加法交换幺半群”的结构；现在不需要详细说明这意味着什么，但值得知道自然数、整数、有理数和实数都是实例。因此，如果我们在这个普遍性级别上证明定理 `fnUb_add`，它将适用于所有这些实例。

```py
variable  {α  :  Type*}  {R  :  Type*}  [AddCommMonoid  R]  [PartialOrder  R]  [IsOrderedCancelAddMonoid  R]

#check  add_le_add

def  FnUb'  (f  :  α  →  R)  (a  :  R)  :  Prop  :=
  ∀  x,  f  x  ≤  a

theorem  fnUb_add  {f  g  :  α  →  R}  {a  b  :  R}  (hfa  :  FnUb'  f  a)  (hgb  :  FnUb'  g  b)  :
  FnUb'  (fun  x  ↦  f  x  +  g  x)  (a  +  b)  :=  fun  x  ↦  add_le_add  (hfa  x)  (hgb  x) 
```

你已经在 第 2.2 节 中看到过这样的方括号，尽管我们还没有解释它们的意义。为了具体化，我们将大部分例子都坚持使用实数，但值得知道 Mathlib 包含在高度普遍性级别上工作的定义和定理。

对于另一个隐藏的全称量词的例子，Mathlib 定义了一个谓词 `Monotone`，它表示函数在其参数上是非递减的：

```py
example  (f  :  ℝ  →  ℝ)  (h  :  Monotone  f)  :  ∀  {a  b},  a  ≤  b  →  f  a  ≤  f  b  :=
  @h 
```

属性 `Monotone f` 被定义为冒号后面的确切表达式。我们需要在 `h` 前面放置 `@` 符号，因为我们不这样做的话，Lean 会将隐含的参数扩展到 `h` 并插入占位符。

证明关于单调性的命题涉及使用 `intro` 引入两个变量，例如 `a` 和 `b`，以及假设 `a ≤ b`。要 *使用* 单调性假设，你可以将其应用于合适的参数和假设，然后将结果表达式应用于目標。或者，你可以将其应用于目標，让 Lean 通过显示剩余假设作为新的子目標来帮助你反向工作。

```py
example  (mf  :  Monotone  f)  (mg  :  Monotone  g)  :  Monotone  fun  x  ↦  f  x  +  g  x  :=  by
  intro  a  b  aleb
  apply  add_le_add
  apply  mf  aleb
  apply  mg  aleb 
```

当证明如此简短时，通常方便给出一个证明术语。为了描述一个临时引入对象 `a` 和 `b` 以及假设 `aleb` 的证明，Lean 使用了 `fun a b aleb ↦ ...` 的符号。这与像 `fun x ↦ x²` 这样的表达式通过临时命名一个对象 `x` 并用它来描述一个值的方式来描述一个函数是相似的。因此，前一个证明中的 `intro` 命令对应于下一个证明术语中的 lambda 抽象。然后，`apply` 命令对应于将定理应用于其参数的构造。

```py
example  (mf  :  Monotone  f)  (mg  :  Monotone  g)  :  Monotone  fun  x  ↦  f  x  +  g  x  :=
  fun  a  b  aleb  ↦  add_le_add  (mf  aleb)  (mg  aleb) 
```

这里有一个有用的技巧：如果你开始编写证明术语 `fun a b aleb ↦ _` 并在表达式其余部分应该放置的地方使用下划线，Lean 将会标记一个错误，表明它无法猜测该表达式的值。如果你在 VS Code 中的 Lean Goal 窗口或悬停在波浪形错误标记上，Lean 将会显示剩余表达式需要解决的目標。

尝试使用策略或证明术语来证明这些命题：

```py
example  {c  :  ℝ}  (mf  :  Monotone  f)  (nnc  :  0  ≤  c)  :  Monotone  fun  x  ↦  c  *  f  x  :=
  sorry

example  (mf  :  Monotone  f)  (mg  :  Monotone  g)  :  Monotone  fun  x  ↦  f  (g  x)  :=
  sorry 
```

这里有一些更多的例子。从 $\Bbb R$ 到 $\Bbb R$ 的函数 $f$ 被称为 *偶函数*，如果对于每个 $x$，$f(-x) = f(x)$；如果对于每个 $x$，$f(-x) = -f(x)$，则称为 *奇函数*。以下示例正式定义了这两个概念并建立了一个关于它们的命题。你可以完成其他命题的证明。

```py
def  FnEven  (f  :  ℝ  →  ℝ)  :  Prop  :=
  ∀  x,  f  x  =  f  (-x)

def  FnOdd  (f  :  ℝ  →  ℝ)  :  Prop  :=
  ∀  x,  f  x  =  -f  (-x)

example  (ef  :  FnEven  f)  (eg  :  FnEven  g)  :  FnEven  fun  x  ↦  f  x  +  g  x  :=  by
  intro  x
  calc
  (fun  x  ↦  f  x  +  g  x)  x  =  f  x  +  g  x  :=  rfl
  _  =  f  (-x)  +  g  (-x)  :=  by  rw  [ef,  eg]

example  (of  :  FnOdd  f)  (og  :  FnOdd  g)  :  FnEven  fun  x  ↦  f  x  *  g  x  :=  by
  sorry

example  (ef  :  FnEven  f)  (og  :  FnOdd  g)  :  FnOdd  fun  x  ↦  f  x  *  g  x  :=  by
  sorry

example  (ef  :  FnEven  f)  (og  :  FnOdd  g)  :  FnEven  fun  x  ↦  f  (g  x)  :=  by
  sorry 
```

第一个证明可以使用 `dsimp` 或 `change` 来缩短，以消除 lambda 抽象。但你可以检查，除非我们明确地消除 lambda 抽象，否则后续的 `rw` 不会工作，因为否则它无法在表达式中找到 `f x` 和 `g x` 的模式。与一些其他策略不同，`rw` 在句法级别上操作，它不会展开定义或为你应用简化（它有一个名为 `erw` 的变体，在这个方向上尝试得稍微努力一些，但不是很多）。

一旦你知道如何找到它们，你可以在任何地方找到隐含的全称量词。

Mathlib 包含了一个用于操作集合的良好库。回想一下，Lean 不使用基于集合理论的公理系统，因此在这里，“集合”一词具有平凡的意义，即某种给定类型 `α` 的数学对象的集合。如果 `x` 具有类型 `α` 且 `s` 具有类型 `Set α`，那么 `x ∈ s` 是一个命题，它断言 `x` 是 `s` 的一个元素。如果 `y` 具有某种不同的类型 `β`，那么表达式 `y ∈ s` 就没有意义。这里的“没有意义”意味着“没有类型，因此 Lean 不接受它作为一个有效的语句”。这与例如 Zermelo-Fraenkel 集合理论形成对比，在 ZF 中，对于每个数学对象 `a` 和 `b`，`a ∈ b` 都是一个有效的语句。例如，在 ZF 中 `sin ∈ cos` 是一个有效的语句。集合理论基础的这种缺陷是重要的动机，即不在旨在通过检测无意义表达式来帮助我们证明辅助工具中使用它。在 Lean 中，`sin` 具有类型 `ℝ → ℝ`，`cos` 也具有类型 `ℝ → ℝ`，这并不等于 `Set (ℝ → ℝ)`，即使展开定义之后也是如此，因此语句 `sin ∈ cos` 没有意义。人们也可以使用 Lean 来研究集合理论本身。例如，连续性假设与 Zermelo-Fraenkel 公理的独立性已经在 Lean 中形式化。但这样的集合理论元理论完全超出了本书的范围。

如果 `s` 和 `t` 都是类型 `Set α`，那么子集关系 `s ⊆ t` 被定义为 `∀ {x : α}, x ∈ s → x ∈ t`。量词中的变量被标记为隐含的，这样给定 `h : s ⊆ t` 和 `h' : x ∈ s`，我们可以写出 `h h'` 作为 `x ∈ t` 的理由。以下例子提供了一个策略证明和一个证明项，以证明子集关系的自反性，并要求你为传递性做同样的证明。

```py
variable  {α  :  Type*}  (r  s  t  :  Set  α)

example  :  s  ⊆  s  :=  by
  intro  x  xs
  exact  xs

theorem  Subset.refl  :  s  ⊆  s  :=  fun  x  xs  ↦  xs

theorem  Subset.trans  :  r  ⊆  s  →  s  ⊆  t  →  r  ⊆  t  :=  by
  sorry 
```

正如我们为函数定义了 `FnUb`，我们也可以定义 `SetUb s a` 来表示 `a` 是集合 `s` 的上界，假设 `s` 是某种具有关联顺序的元素集合。在下一个例子中，我们要求你证明如果 `a` 是 `s` 的一个界且 `a ≤ b`，那么 `b` 也是 `s` 的一个界。

```py
variable  {α  :  Type*}  [PartialOrder  α]
variable  (s  :  Set  α)  (a  b  :  α)

def  SetUb  (s  :  Set  α)  (a  :  α)  :=
  ∀  x,  x  ∈  s  →  x  ≤  a

example  (h  :  SetUb  s  a)  (h'  :  a  ≤  b)  :  SetUb  s  b  :=
  sorry 
```

我们以一个最后的、重要的例子结束本节。一个函数 $f$ 被称为 *注入的*，如果对于每一个 $x_1$ 和 $x_2$，如果 $f(x_1) = f(x_2)$，那么 $x_1 = x_2$。Mathlib 使用隐含的 `x₁` 和 `x₂` 定义 `Function.Injective f`。下一个例子展示了在实数上，任何添加常数的函数都是注入的。然后我们要求你使用例子中的引理名称作为灵感来源，证明乘以非零常数也是注入的。回想一下，你应该在猜测引理名称的开头后使用 Ctrl-space 完成功能。

```py
open  Function

example  (c  :  ℝ)  :  Injective  fun  x  ↦  x  +  c  :=  by
  intro  x₁  x₂  h'
  exact  (add_left_inj  c).mp  h'

example  {c  :  ℝ}  (h  :  c  ≠  0)  :  Injective  fun  x  ↦  c  *  x  :=  by
  sorry 
```

最后，证明两个注入函数的复合是注入的：

```py
variable  {α  :  Type*}  {β  :  Type*}  {γ  :  Type*}
variable  {g  :  β  →  γ}  {f  :  α  →  β}

example  (injg  :  Injective  g)  (injf  :  Injective  f)  :  Injective  fun  x  ↦  g  (f  x)  :=  by
  sorry 
```

## 3.2\. 存在量词

存在量词，可以在 VS Code 中输入为 `\ex`，用来表示“存在”这个短语。在 Lean 中，形式表达式 `∃ x : ℝ, 2 < x ∧ x < 3` 表示存在一个实数在 2 和 3 之间。（我们将在 第 3.4 节 讨论合取符号 `∧`。）证明此类陈述的规范方法是展示一个实数并证明它具有所述的性质。数字 2.5，我们可以在 Lean 中输入为 `5 / 2` 或 `(5 : ℝ) / 2`（当 Lean 无法从上下文中推断出我们指的是实数时），具有所需的性质，而 `norm_num` 策略可以证明它符合描述。

我们有几种方法可以将信息组合起来。给定一个以存在量词开始的目標，使用 `use` 策略来提供对象，留下证明该属性的目标。

```py
example  :  ∃  x  :  ℝ,  2  <  x  ∧  x  <  3  :=  by
  use  5  /  2
  norm_num 
```

您可以提供 `use` 策略的证明以及数据：

```py
example  :  ∃  x  :  ℝ,  2  <  x  ∧  x  <  3  :=  by
  have  h1  :  2  <  (5  :  ℝ)  /  2  :=  by  norm_num
  have  h2  :  (5  :  ℝ)  /  2  <  3  :=  by  norm_num
  use  5  /  2,  h1,  h2 
```

实际上，`use` 策略会自动尝试使用可用的假设。

```py
example  :  ∃  x  :  ℝ,  2  <  x  ∧  x  <  3  :=  by
  have  h  :  2  <  (5  :  ℝ)  /  2  ∧  (5  :  ℝ)  /  2  <  3  :=  by  norm_num
  use  5  /  2 
```

或者，我们可以使用 Lean 的 *匿名构造函数* 符号来构造一个存在量词的证明。

```py
example  :  ∃  x  :  ℝ,  2  <  x  ∧  x  <  3  :=
  have  h  :  2  <  (5  :  ℝ)  /  2  ∧  (5  :  ℝ)  /  2  <  3  :=  by  norm_num
  ⟨5  /  2,  h⟩ 
```

注意，这里没有 `by`；我们在这里提供了一个明确的证明项。左右尖括号，分别可以输入为 `\<` 和 `\>`，告诉 Lean 使用适当的构造将给定数据组合起来。我们可以使用这种符号而不必首先进入策略模式：

```py
example  :  ∃  x  :  ℝ,  2  <  x  ∧  x  <  3  :=
  ⟨5  /  2,  by  norm_num⟩ 
```

因此，现在我们知道了如何 *证明* 一个存在陈述。但如何 *使用* 它呢？如果我们知道存在具有某种性质的某个对象，我们应该能够给它命名并对其进行分析。例如，记住上一节中的谓词 `FnUb f a` 和 `FnLb f a`，它们分别表示 `a` 是 `f` 的上界或下界。我们可以使用存在量词来说明“`f` 有界”，而不必指定界限：

```py
def  FnUb  (f  :  ℝ  →  ℝ)  (a  :  ℝ)  :  Prop  :=
  ∀  x,  f  x  ≤  a

def  FnLb  (f  :  ℝ  →  ℝ)  (a  :  ℝ)  :  Prop  :=
  ∀  x,  a  ≤  f  x

def  FnHasUb  (f  :  ℝ  →  ℝ)  :=
  ∃  a,  FnUb  f  a

def  FnHasLb  (f  :  ℝ  →  ℝ)  :=
  ∃  a,  FnLb  f  a 
```

我们可以使用上一节中的定理 `FnUb_add` 来证明，如果 `f` 和 `g` 有上界，那么 `fun x ↦ f x + g x` 也有上界。

```py
variable  {f  g  :  ℝ  →  ℝ}

example  (ubf  :  FnHasUb  f)  (ubg  :  FnHasUb  g)  :  FnHasUb  fun  x  ↦  f  x  +  g  x  :=  by
  rcases  ubf  with  ⟨a,  ubfa⟩
  rcases  ubg  with  ⟨b,  ubgb⟩
  use  a  +  b
  apply  fnUb_add  ubfa  ubgb 
```

`rcases` 策略解包存在量词中的信息。像 `⟨a, ubfa⟩` 这样的注释，使用与匿名构造函数相同的尖括号，被称为 *模式*，它们描述了我们解包主论点时预期找到的信息。给定存在 `f` 的上界的假设 `ubf`，`rcases ubf with ⟨a, ubfa⟩` 将一个新的变量 `a`（上界）添加到上下文中，以及具有给定性质的假设 `ubfa`。目标保持不变；真正改变的是，我们现在可以使用新的对象和新的假设来证明目标。这是数学中常见的推理方法：我们解包那些被某些假设断言或暗示存在性的对象，然后使用它来建立其他事物的存在。

尝试使用这种方法来建立以下内容。你可能发现将上一节的一些示例转换为命名定理很有用，就像我们对 `fn_ub_add` 所做的那样，或者你可以直接将参数插入到证明中。

```py
example  (lbf  :  FnHasLb  f)  (lbg  :  FnHasLb  g)  :  FnHasLb  fun  x  ↦  f  x  +  g  x  :=  by
  sorry

example  {c  :  ℝ}  (ubf  :  FnHasUb  f)  (h  :  c  ≥  0)  :  FnHasUb  fun  x  ↦  c  *  f  x  :=  by
  sorry 
```

`rcases` 中的 “r” 代表 “recursive”，因为它允许我们使用任意复杂的模式来解包嵌套数据。`rintro` 策略是 `intro` 和 `rcases` 的结合：

```py
example  :  FnHasUb  f  →  FnHasUb  g  →  FnHasUb  fun  x  ↦  f  x  +  g  x  :=  by
  rintro  ⟨a,  ubfa⟩  ⟨b,  ubgb⟩
  exact  ⟨a  +  b,  fnUb_add  ubfa  ubgb⟩ 
```

实际上，Lean 也支持在表达式和证明项中使用模式匹配函数：

```py
example  :  FnHasUb  f  →  FnHasUb  g  →  FnHasUb  fun  x  ↦  f  x  +  g  x  :=
  fun  ⟨a,  ubfa⟩  ⟨b,  ubgb⟩  ↦  ⟨a  +  b,  fnUb_add  ubfa  ubgb⟩ 
```

在假设中解包信息这项任务非常重要，以至于 Lean 和 Mathlib 提供了多种方法来完成它。例如，`obtain` 策略提供了提示性语法：

```py
example  (ubf  :  FnHasUb  f)  (ubg  :  FnHasUb  g)  :  FnHasUb  fun  x  ↦  f  x  +  g  x  :=  by
  obtain  ⟨a,  ubfa⟩  :=  ubf
  obtain  ⟨b,  ubgb⟩  :=  ubg
  exact  ⟨a  +  b,  fnUb_add  ubfa  ubgb⟩ 
```

将第一个 `obtain` 指令视为将 `ubf` 的“内容”与给定的模式匹配，并将组件分配给命名变量的过程。`rcases` 和 `obtain` 被说成是“破坏”它们的参数。

Lean 还支持与其他函数式编程语言中使用的语法相似的语法：

```py
example  (ubf  :  FnHasUb  f)  (ubg  :  FnHasUb  g)  :  FnHasUb  fun  x  ↦  f  x  +  g  x  :=  by
  cases  ubf
  case  intro  a  ubfa  =>
  cases  ubg
  case  intro  b  ubgb  =>
  exact  ⟨a  +  b,  fnUb_add  ubfa  ubgb⟩

example  (ubf  :  FnHasUb  f)  (ubg  :  FnHasUb  g)  :  FnHasUb  fun  x  ↦  f  x  +  g  x  :=  by
  cases  ubf
  next  a  ubfa  =>
  cases  ubg
  next  b  ubgb  =>
  exact  ⟨a  +  b,  fnUb_add  ubfa  ubgb⟩

example  (ubf  :  FnHasUb  f)  (ubg  :  FnHasUb  g)  :  FnHasUb  fun  x  ↦  f  x  +  g  x  :=  by
  match  ubf,  ubg  with
  |  ⟨a,  ubfa⟩,  ⟨b,  ubgb⟩  =>
  exact  ⟨a  +  b,  fnUb_add  ubfa  ubgb⟩

example  (ubf  :  FnHasUb  f)  (ubg  :  FnHasUb  g)  :  FnHasUb  fun  x  ↦  f  x  +  g  x  :=
  match  ubf,  ubg  with
  |  ⟨a,  ubfa⟩,  ⟨b,  ubgb⟩  =>
  ⟨a  +  b,  fnUb_add  ubfa  ubgb⟩ 
```

在第一个例子中，如果你将光标放在 `cases ubf` 之后，你会看到该策略产生了一个单一的目标，Lean 已经将其标记为 `intro`。（所选名称来自构建存在性陈述证明的公理原语的内部名称。）然后 `case` 策略命名了组件。第二个例子与第一个类似，只是使用 `next` 而不是 `case` 意味着你不必提到 `intro`。在最后两个例子中，`match`一词突出了我们在这里所做的是计算机科学家所说的“模式匹配”。注意，第三个证明从 `by` 开始，之后 `match` 的策略版本期望箭头右侧有一个策略证明。最后一个例子是一个证明项：没有看到任何策略。

在本书的其余部分，我们将坚持使用 `rcases`、`rintro` 和 `obtain`，作为使用存在量词的首选方式。但看看替代语法也无妨，尤其是如果你有可能发现自己与计算机科学家在一起。

为了说明 `rcases` 可以用的一种方式，我们证明了一个古老的数学难题：如果两个整数 `x` 和 `y` 都可以写成两个平方数的和，那么它们的乘积 `x * y` 也可以。实际上，这个陈述对于任何交换环都成立，而不仅仅是整数。在下一个例子中，`rcases` 一次解包了两个存在量词。然后我们提供一个列表，其中包含将 `x * y` 表示为平方数和所需的魔法值，并将其作为参数传递给 `use` 语句，并使用 `ring` 来验证它们是否有效。

```py
variable  {α  :  Type*}  [CommRing  α]

def  SumOfSquares  (x  :  α)  :=
  ∃  a  b,  x  =  a  ^  2  +  b  ^  2

theorem  sumOfSquares_mul  {x  y  :  α}  (sosx  :  SumOfSquares  x)  (sosy  :  SumOfSquares  y)  :
  SumOfSquares  (x  *  y)  :=  by
  rcases  sosx  with  ⟨a,  b,  xeq⟩
  rcases  sosy  with  ⟨c,  d,  yeq⟩
  rw  [xeq,  yeq]
  use  a  *  c  -  b  *  d,  a  *  d  +  b  *  c
  ring 
```

这个证明并没有提供很多见解，但这里有一种激发它的方法。高斯整数是一种形式为 $a + bi$ 的数，其中 $a$ 和 $b$ 是整数，$i = \sqrt{-1}$。高斯整数 $a + bi$ 的**范数**，根据定义，是 $a² + b²$。因此，高斯整数的范数是平方和，任何平方和都可以用这种方式表示。上述定理反映了高斯整数乘积的范数是它们范数的乘积这一事实：如果 $x$ 是 $a + bi$ 的范数，$y$ 是 $c + di$ 的范数，那么 $xy$ 是 $(a + bi) (c + di)$ 的范数。我们晦涩的证明说明了这样一个事实：最容易形式化的证明并不总是最清晰的。在第 7.3 节中，我们将提供定义高斯整数并使用它们提供另一种证明的方法。

在存在量词内部展开方程的模式，然后使用它来重写目标表达式，这种模式经常出现，以至于 `rcases` 策略提供了一个缩写：如果你用关键字 `rfl` 代替一个新标识符，`rcases` 会自动进行重写（这个技巧与模式匹配的 lambda 不兼容）。

```py
theorem  sumOfSquares_mul'  {x  y  :  α}  (sosx  :  SumOfSquares  x)  (sosy  :  SumOfSquares  y)  :
  SumOfSquares  (x  *  y)  :=  by
  rcases  sosx  with  ⟨a,  b,  rfl⟩
  rcases  sosy  with  ⟨c,  d,  rfl⟩
  use  a  *  c  -  b  *  d,  a  *  d  +  b  *  c
  ring 
```

正如全称量词一样，如果你知道如何识别它们，你可以在任何地方找到隐藏的存在量词。例如，可除性隐含地是一个“存在”陈述。

```py
example  (divab  :  a  ∣  b)  (divbc  :  b  ∣  c)  :  a  ∣  c  :=  by
  rcases  divab  with  ⟨d,  beq⟩
  rcases  divbc  with  ⟨e,  ceq⟩
  rw  [ceq,  beq]
  use  d  *  e;  ring 
```

再次，这为使用 `rcases` 和 `rfl` 提供了一个很好的环境。在上面的证明中试一试。感觉相当不错！

然后尝试证明以下内容：

```py
example  (divab  :  a  ∣  b)  (divac  :  a  ∣  c)  :  a  ∣  b  +  c  :=  by
  sorry 
```

对于另一个重要的例子，一个函数 $f : \alpha \to \beta$ 被称为**满射**，如果对于域 $\alpha$ 中的每一个 $x$，在值域 $\beta$ 中都存在一个 $y$，使得 $f(x) = y$。注意，这个陈述包含了全称量词和存在量词，这也解释了为什么下一个例子会同时使用 `intro` 和 `use`。

```py
example  {c  :  ℝ}  :  Surjective  fun  x  ↦  x  +  c  :=  by
  intro  x
  use  x  -  c
  dsimp;  ring 
```

尝试使用定理 `mul_div_cancel₀` 自己做这个例子。

```py
example  {c  :  ℝ}  (h  :  c  ≠  0)  :  Surjective  fun  x  ↦  c  *  x  :=  by
  sorry 
```

在这一点上，值得提到的是，有一个策略 `field_simp`，它通常会以有用的方式消除分母。它可以与 `ring` 策略一起使用。

```py
example  (x  y  :  ℝ)  (h  :  x  -  y  ≠  0)  :  (x  ^  2  -  y  ^  2)  /  (x  -  y)  =  x  +  y  :=  by
  field_simp  [h]
  ring 
```

下一个例子通过应用适当的值来使用满射假设。请注意，你可以用 `rcases` 与任何表达式一起使用，而不仅仅是假设。

```py
example  {f  :  ℝ  →  ℝ}  (h  :  Surjective  f)  :  ∃  x,  f  x  ^  2  =  4  :=  by
  rcases  h  2  with  ⟨x,  hx⟩
  use  x
  rw  [hx]
  norm_num 
```

看看你是否可以使用这些方法来证明满射函数的复合是满射的。

```py
variable  {α  :  Type*}  {β  :  Type*}  {γ  :  Type*}
variable  {g  :  β  →  γ}  {f  :  α  →  β}

example  (surjg  :  Surjective  g)  (surjf  :  Surjective  f)  :  Surjective  fun  x  ↦  g  (f  x)  :=  by
  sorry 
```

## 3.3. 否定

符号 `¬` 用于表示否定，因此 `¬ x < y` 表示 `x` 不小于 `y`，`¬ x = y`（或等价地，`x ≠ y`）表示 `x` 不等于 `y`，而 `¬ ∃ z, x < z ∧ z < y` 表示不存在一个 `z` 在 `x` 和 `y` 之间。在 Lean 中，符号 `¬ A` 简写为 `A → False`，你可以将其理解为 `A` 导致矛盾。实际上，这意味着你已经了解了一些关于如何处理否定的方法：你可以通过引入假设 `h : A` 并证明 `False` 来证明 `¬ A`，如果你有 `h : ¬ A` 和 `h' : A`，那么将 `h` 应用到 `h'` 上会得到 `False`。

为了说明，考虑严格顺序的不自反性原理 `lt_irrefl`，它表示对于每个 `a`，我们有 `¬ a < a`。不对称性原理 `lt_asymm` 表示 `a < b → ¬ b < a`。让我们证明 `lt_asymm` 可以从 `lt_irrefl` 推导出来。

```py
example  (h  :  a  <  b)  :  ¬b  <  a  :=  by
  intro  h'
  have  :  a  <  a  :=  lt_trans  h  h'
  apply  lt_irrefl  a  this 
```

这个例子介绍了一些新的技巧。首先，当你使用 `have` 而不提供标签时，Lean 使用名称 `this`，提供了一种方便的回指方式。因为证明非常简短，所以我们提供了一个显式的证明项。但你应该真正关注这个证明的是 `intro` 策略的结果，它留下一个 `False` 的目标，以及我们最终通过将 `lt_irrefl` 应用到一个 `a < a` 的证明上证明 `False` 的事实。

这里是另一个例子，它使用了上一节中定义的谓词 `FnHasUb`，它表示一个函数有一个上界。

```py
example  (h  :  ∀  a,  ∃  x,  f  x  >  a)  :  ¬FnHasUb  f  :=  by
  intro  fnub
  rcases  fnub  with  ⟨a,  fnuba⟩
  rcases  h  a  with  ⟨x,  hx⟩
  have  :  f  x  ≤  a  :=  fnuba  x
  linarith 
```

记住，当目标由上下文中的线性方程和不等式推导出来时，使用 `linarith` 通常很方便。

看看你是否可以用类似的方式证明这些定理：

```py
example  (h  :  ∀  a,  ∃  x,  f  x  <  a)  :  ¬FnHasLb  f  :=
  sorry

example  :  ¬FnHasUb  fun  x  ↦  x  :=
  sorry 
```

Mathlib 提供了多个有用的定理，用于关联顺序和否定：

```py
#check  (not_le_of_gt  :  a  >  b  →  ¬a  ≤  b)
#check  (not_lt_of_ge  :  a  ≥  b  →  ¬a  <  b)
#check  (lt_of_not_ge  :  ¬a  ≥  b  →  a  <  b)
#check  (le_of_not_gt  :  ¬a  >  b  →  a  ≤  b) 
```

回忆一下谓词 `Monotone f`，它表示 `f` 是非递减的。使用刚刚列举的一些定理来证明以下内容：

```py
example  (h  :  Monotone  f)  (h'  :  f  a  <  f  b)  :  a  <  b  :=  by
  sorry

example  (h  :  a  ≤  b)  (h'  :  f  b  <  f  a)  :  ¬Monotone  f  :=  by
  sorry 
```

我们可以证明，如果我们将 `<` 替换为 `≤`，则上一段代码中的第一个例子无法被证明。注意，我们可以通过给出反例来证明全称量化语句的否定。完成证明。

```py
example  :  ¬∀  {f  :  ℝ  →  ℝ},  Monotone  f  →  ∀  {a  b},  f  a  ≤  f  b  →  a  ≤  b  :=  by
  intro  h
  let  f  :=  fun  x  :  ℝ  ↦  (0  :  ℝ)
  have  monof  :  Monotone  f  :=  by  sorry
  have  h'  :  f  1  ≤  f  0  :=  le_refl  _
  sorry 
```

这个例子引入了 `let` 策略，它向上下文中添加一个 *局部定义*。如果你将光标放在 `let` 命令之后，在目标窗口中你会看到已经添加到上下文中的定义 `f : ℝ → ℝ := fun x ↦ 0`。当 Lean 需要时，它会展开 `f` 的定义。特别是，当我们使用 `le_refl` 证明 `f 1 ≤ f 0` 时，Lean 会将 `f 1` 和 `f 0` 简化为 `0`。

使用 `le_of_not_gt` 来证明以下内容：

```py
example  (x  :  ℝ)  (h  :  ∀  ε  >  0,  x  <  ε)  :  x  ≤  0  :=  by
  sorry 
```

在我们刚刚做的许多证明中隐含的事实是，如果 `P` 是任何属性，说没有任何具有属性 `P` 的东西等同于说所有东西都未能具有属性 `P`，而说并非所有东西都具有属性 `P` 等价于说有些东西未能具有属性 `P`。换句话说，以下四个蕴涵都是有效的（但其中之一不能使用我们之前解释的方法来证明）：

```py
variable  {α  :  Type*}  (P  :  α  →  Prop)  (Q  :  Prop)

example  (h  :  ¬∃  x,  P  x)  :  ∀  x,  ¬P  x  :=  by
  sorry

example  (h  :  ∀  x,  ¬P  x)  :  ¬∃  x,  P  x  :=  by
  sorry

example  (h  :  ¬∀  x,  P  x)  :  ∃  x,  ¬P  x  :=  by
  sorry

example  (h  :  ∃  x,  ¬P  x)  :  ¬∀  x,  P  x  :=  by
  sorry 
```

第一、第二和第四个可以通过你已经看到的方法直接证明。我们鼓励你尝试一下。然而，第三个比较困难，因为它从其不存在是矛盾的这一事实中得出一个对象存在的结论。这是一个 *经典* 数学推理的例子。我们可以通过以下方式使用反证法来证明第三个蕴涵：

```py
example  (h  :  ¬∀  x,  P  x)  :  ∃  x,  ¬P  x  :=  by
  by_contra  h'
  apply  h
  intro  x
  show  P  x
  by_contra  h''
  exact  h'  ⟨x,  h''⟩ 
```

确保你理解这是如何工作的。`by_contra` 策略允许我们通过假设 `¬ Q` 并推导出矛盾来证明目标 `Q`。实际上，这等价于使用等价性 `not_not : ¬ ¬ Q ↔ Q`。确认你可以使用 `by_contra` 证明这个等价的正向方向，而反向方向则遵循否定规则的普通规则。

```py
example  (h  :  ¬¬Q)  :  Q  :=  by
  sorry

example  (h  :  Q)  :  ¬¬Q  :=  by
  sorry 
```

使用反证法来建立以下内容，这是我们之前证明的一个蕴涵的逆命题。（提示：首先使用 `intro`。）

```py
example  (h  :  ¬FnHasUb  f)  :  ∀  a,  ∃  x,  f  x  >  a  :=  by
  sorry 
```

处理带有否定前缀的复合语句通常很繁琐，将否定推入等价形式是常见的数学模式。为了方便这一点，Mathlib 提供了一个 `push_neg` 策略，它以这种方式重新表述目标。命令 `push_neg at h` 重新表述假设 `h`。

```py
example  (h  :  ¬∀  a,  ∃  x,  f  x  >  a)  :  FnHasUb  f  :=  by
  push_neg  at  h
  exact  h

example  (h  :  ¬FnHasUb  f)  :  ∀  a,  ∃  x,  f  x  >  a  :=  by
  dsimp  only  [FnHasUb,  FnUb]  at  h
  push_neg  at  h
  exact  h 
```

在第二个例子中，我们使用 `dsimp` 来展开 `FnHasUb` 和 `FnUb` 的定义。（我们需要使用 `dsimp` 而不是 `rw` 来展开 `FnUb`，因为它出现在量词的作用域内。）你可以验证在上面的例子中，使用 `¬∃ x, P x` 和 `¬∀ x, P x` 时，`push_neg` 策略做了预期的事情。即使不知道如何使用合取符号，你也应该能够使用 `push_neg` 来证明以下内容：

```py
example  (h  :  ¬Monotone  f)  :  ∃  x  y,  x  ≤  y  ∧  f  y  <  f  x  :=  by
  sorry 
```

Mathlib 还有一个策略，`contrapose`，它将目标 `A → B` 转换为 `¬B → ¬A`。同样，给定一个从假设 `h : A` 证明 `B` 的目标，`contrapose h` 会让你留下一个从假设 `¬B` 证明 `¬A` 的目标。使用 `contrapose!` 而不是 `contrapose` 将 `push_neg` 应用到目标和相关假设上。

```py
example  (h  :  ¬FnHasUb  f)  :  ∀  a,  ∃  x,  f  x  >  a  :=  by
  contrapose!  h
  exact  h

example  (x  :  ℝ)  (h  :  ∀  ε  >  0,  x  ≤  ε)  :  x  ≤  0  :=  by
  contrapose!  h
  use  x  /  2
  constructor  <;>  linarith 
```

我们还没有解释 `constructor` 命令及其后的分号的使用，但将在下一节中解释。

我们以 *ex falso* 原则结束本节，该原则表明任何东西都从矛盾中得出。在 Lean 中，这表示为 `False.elim`，它为任何命题 `P` 建立 `False → P`。这看起来可能像是一个奇怪的原则，但它相当常见。我们经常通过分情况证明定理，有时我们可以表明其中一种情况是矛盾的。在这种情况下，我们需要断言矛盾建立了目标，这样我们就可以继续进行下一个。（我们将在 第 3.5 节 中看到推理的例子。）

Lean 提供了多种在达到矛盾后关闭目标的方法。

```py
example  (h  :  0  <  0)  :  a  >  37  :=  by
  exfalso
  apply  lt_irrefl  0  h

example  (h  :  0  <  0)  :  a  >  37  :=
  absurd  h  (lt_irrefl  0)

example  (h  :  0  <  0)  :  a  >  37  :=  by
  have  h'  :  ¬0  <  0  :=  lt_irrefl  0
  contradiction 
```

`exfalso` 策略将当前目标替换为证明 `False` 的目标。给定 `h : P` 和 `h' : ¬ P`，术语 `absurd h h'` 建立任何命题。最后，`contradiction` 策略试图通过在假设中找到矛盾来关闭目标，例如形式为 `h : P` 和 `h' : ¬ P` 的成对。当然，在这个例子中，`linarith` 也有效。

## 3.4. 联合与 Iff

您已经看到，合取符号 `∧` 用于表示“和”。`constructor` 策略允许您通过先证明 `A` 然后证明 `B` 来证明形式为 `A ∧ B` 的陈述。

```py
example  {x  y  :  ℝ}  (h₀  :  x  ≤  y)  (h₁  :  ¬y  ≤  x)  :  x  ≤  y  ∧  x  ≠  y  :=  by
  constructor
  ·  assumption
  intro  h
  apply  h₁
  rw  [h] 
```

在本例中，`assumption` 策略告诉 Lean 寻找一个假设来解决问题。注意，最后的 `rw` 通过应用 `≤` 的自反性来完成目标。以下是通过匿名构造器尖括号执行先前示例的替代方法。第一种是先前证明的简洁证明术语版本，它在 `by` 关键字处进入策略模式。

```py
example  {x  y  :  ℝ}  (h₀  :  x  ≤  y)  (h₁  :  ¬y  ≤  x)  :  x  ≤  y  ∧  x  ≠  y  :=
  ⟨h₀,  fun  h  ↦  h₁  (by  rw  [h])⟩

example  {x  y  :  ℝ}  (h₀  :  x  ≤  y)  (h₁  :  ¬y  ≤  x)  :  x  ≤  y  ∧  x  ≠  y  :=
  have  h  :  x  ≠  y  :=  by
  contrapose!  h₁
  rw  [h₁]
  ⟨h₀,  h⟩ 
```

*使用* 联合而不是证明一个部分涉及展开两个部分的证明。您可以使用 `rcases` 策略来完成此操作，以及 `rintro` 或模式匹配 `fun`，所有这些都与存在量词的使用方式相似。

```py
example  {x  y  :  ℝ}  (h  :  x  ≤  y  ∧  x  ≠  y)  :  ¬y  ≤  x  :=  by
  rcases  h  with  ⟨h₀,  h₁⟩
  contrapose!  h₁
  exact  le_antisymm  h₀  h₁

example  {x  y  :  ℝ}  :  x  ≤  y  ∧  x  ≠  y  →  ¬y  ≤  x  :=  by
  rintro  ⟨h₀,  h₁⟩  h'
  exact  h₁  (le_antisymm  h₀  h')

example  {x  y  :  ℝ}  :  x  ≤  y  ∧  x  ≠  y  →  ¬y  ≤  x  :=
  fun  ⟨h₀,  h₁⟩  h'  ↦  h₁  (le_antisymm  h₀  h') 
```

类似于 `obtain` 策略，还有一个模式匹配的 `have`：

```py
example  {x  y  :  ℝ}  (h  :  x  ≤  y  ∧  x  ≠  y)  :  ¬y  ≤  x  :=  by
  have  ⟨h₀,  h₁⟩  :=  h
  contrapose!  h₁
  exact  le_antisymm  h₀  h₁ 
```

与 `rcases` 相比，这里的 `have` 策略将 `h` 留在上下文中。即使我们不会使用它们，我们再次有了计算机科学家的模式匹配语法：

```py
example  {x  y  :  ℝ}  (h  :  x  ≤  y  ∧  x  ≠  y)  :  ¬y  ≤  x  :=  by
  cases  h
  case  intro  h₀  h₁  =>
  contrapose!  h₁
  exact  le_antisymm  h₀  h₁

example  {x  y  :  ℝ}  (h  :  x  ≤  y  ∧  x  ≠  y)  :  ¬y  ≤  x  :=  by
  cases  h
  next  h₀  h₁  =>
  contrapose!  h₁
  exact  le_antisymm  h₀  h₁

example  {x  y  :  ℝ}  (h  :  x  ≤  y  ∧  x  ≠  y)  :  ¬y  ≤  x  :=  by
  match  h  with
  |  ⟨h₀,  h₁⟩  =>
  contrapose!  h₁
  exact  le_antisymm  h₀  h₁ 
```

与使用存在量词相比，您还可以通过编写 `h.left` 和 `h.right`，或等价地，`h.1` 和 `h.2`，来提取假设 `h : A ∧ B` 的两个组成部分的证明。

```py
example  {x  y  :  ℝ}  (h  :  x  ≤  y  ∧  x  ≠  y)  :  ¬y  ≤  x  :=  by
  intro  h'
  apply  h.right
  exact  le_antisymm  h.left  h'

example  {x  y  :  ℝ}  (h  :  x  ≤  y  ∧  x  ≠  y)  :  ¬y  ≤  x  :=
  fun  h'  ↦  h.right  (le_antisymm  h.left  h') 
```

尝试使用这些技术来想出证明以下内容的各种方法：

```py
example  {m  n  :  ℕ}  (h  :  m  ∣  n  ∧  m  ≠  n)  :  m  ∣  n  ∧  ¬n  ∣  m  :=
  sorry 
```

您可以使用匿名构造器、`rintro` 和 `rcases` 来嵌套使用 `∃` 和 `∧`。

```py
example  :  ∃  x  :  ℝ,  2  <  x  ∧  x  <  4  :=
  ⟨5  /  2,  by  norm_num,  by  norm_num⟩

example  (x  y  :  ℝ)  :  (∃  z  :  ℝ,  x  <  z  ∧  z  <  y)  →  x  <  y  :=  by
  rintro  ⟨z,  xltz,  zlty⟩
  exact  lt_trans  xltz  zlty

example  (x  y  :  ℝ)  :  (∃  z  :  ℝ,  x  <  z  ∧  z  <  y)  →  x  <  y  :=
  fun  ⟨z,  xltz,  zlty⟩  ↦  lt_trans  xltz  zlty 
```

您还可以使用 `use` 策略：

```py
example  :  ∃  x  :  ℝ,  2  <  x  ∧  x  <  4  :=  by
  use  5  /  2
  constructor  <;>  norm_num

example  :  ∃  m  n  :  ℕ,  4  <  m  ∧  m  <  n  ∧  n  <  10  ∧  Nat.Prime  m  ∧  Nat.Prime  n  :=  by
  use  5
  use  7
  norm_num

example  {x  y  :  ℝ}  :  x  ≤  y  ∧  x  ≠  y  →  x  ≤  y  ∧  ¬y  ≤  x  :=  by
  rintro  ⟨h₀,  h₁⟩
  use  h₀
  exact  fun  h'  ↦  h₁  (le_antisymm  h₀  h') 
```

在第一个例子中，`constructor` 命令后面的分号告诉 Lean 对产生的两个目标都使用 `norm_num` 策略。

在 Lean 中，`A ↔ B` 并不是定义为 `(A → B) ∧ (B → A)`，但它本来可以是这样的，并且它的行为大致相同。你已经看到你可以为 `h : A ↔ B` 的两个方向写 `h.mp` 和 `h.mpr` 或 `h.1` 和 `h.2`。你也可以使用 `cases` 和相关工具。为了证明一个“如果且仅如果”的命题，你可以使用 `constructor` 或尖括号，就像你证明合取命题一样。

```py
example  {x  y  :  ℝ}  (h  :  x  ≤  y)  :  ¬y  ≤  x  ↔  x  ≠  y  :=  by
  constructor
  ·  contrapose!
  rintro  rfl
  rfl
  contrapose!
  exact  le_antisymm  h

example  {x  y  :  ℝ}  (h  :  x  ≤  y)  :  ¬y  ≤  x  ↔  x  ≠  y  :=
  ⟨fun  h₀  h₁  ↦  h₀  (by  rw  [h₁]),  fun  h₀  h₁  ↦  h₀  (le_antisymm  h  h₁)⟩ 
```

最后的证明项难以理解。记住，在编写这样的表达式时，你可以使用下划线来查看 Lean 期望的内容。

尝试使用你刚刚看到的各种技术和工具来证明以下内容：

```py
example  {x  y  :  ℝ}  :  x  ≤  y  ∧  ¬y  ≤  x  ↔  x  ≤  y  ∧  x  ≠  y  :=
  sorry 
```

为了进行更有趣的练习，证明对于任何两个实数 `x` 和 `y`，如果 `x² + y² = 0`，则 `x = 0` 且 `y = 0`。我们建议使用 `linarith`、`pow_two_nonneg` 和 `pow_eq_zero` 来证明一个辅助引理。

```py
theorem  aux  {x  y  :  ℝ}  (h  :  x  ^  2  +  y  ^  2  =  0)  :  x  =  0  :=
  have  h'  :  x  ^  2  =  0  :=  by  sorry
  pow_eq_zero  h'

example  (x  y  :  ℝ)  :  x  ^  2  +  y  ^  2  =  0  ↔  x  =  0  ∧  y  =  0  :=
  sorry 
```

在 Lean 中，双条件命题过着双重生活。你可以将其视为合取并分别使用其两部分。但 Lean 也知道它是一个命题之间的自反、对称和传递关系，你也可以使用 `calc` 和 `rw`。将一个命题重写为等价命题通常很方便。在下一个例子中，我们使用 `abs_lt` 将形式为 `|x| < y` 的表达式替换为等价表达式 `- y < x ∧ x < y`，在之后的例子中，我们使用 `Nat.dvd_gcd_iff` 将形式为 `m ∣ Nat.gcd n k` 的表达式替换为等价表达式 `m ∣ n ∧ m ∣ k`。

```py
example  (x  :  ℝ)  :  |x  +  3|  <  5  →  -8  <  x  ∧  x  <  2  :=  by
  rw  [abs_lt]
  intro  h
  constructor  <;>  linarith

example  :  3  ∣  Nat.gcd  6  15  :=  by
  rw  [Nat.dvd_gcd_iff]
  constructor  <;>  norm_num 
```

看看你是否可以使用下面的定理与 `rw` 结合来提供一个简短的证明，证明否定不是一个不减函数。（注意，`push_neg` 不会为你展开定义，因此定理证明中的 `rw [Monotone]` 是必需的。）

```py
theorem  not_monotone_iff  {f  :  ℝ  →  ℝ}  :  ¬Monotone  f  ↔  ∃  x  y,  x  ≤  y  ∧  f  x  >  f  y  :=  by
  rw  [Monotone]
  push_neg
  rfl

example  :  ¬Monotone  fun  x  :  ℝ  ↦  -x  :=  by
  sorry 
```

本节剩余的练习旨在让你在合取和双条件命题方面获得更多实践。记住，*偏序* 是一个传递、自反且反对称的二进制关系。有时会出现一个更弱的概念：*偏序* 只是一个自反且传递的关系。对于任何偏序 `≤`，Lean 通过 `a < b ↔ a ≤ b ∧ ¬ b ≤ a` 公理化相关的严格偏序。证明如果 `≤` 是一个偏序，那么 `a < b` 等价于 `a ≤ b ∧ a ≠ b`：

```py
variable  {α  :  Type*}  [PartialOrder  α]
variable  (a  b  :  α)

example  :  a  <  b  ↔  a  ≤  b  ∧  a  ≠  b  :=  by
  rw  [lt_iff_le_not_ge]
  sorry 
```

逻辑运算之外，你不需要比 `le_refl` 和 `le_trans` 更多的东西。证明即使在 `≤` 只被假设为偏序的情况下，我们也可以证明严格序是不可自反的且传递的。在第二个例子中，为了方便起见，我们使用简化器而不是 `rw` 来用 `≤` 和 `¬` 表达 `<`。我们稍后会回到简化器，但在这里我们只依赖于这样一个事实，即它将反复使用所指示的引理，即使它需要实例化为不同的值。

```py
variable  {α  :  Type*}  [Preorder  α]
variable  (a  b  c  :  α)

example  :  ¬a  <  a  :=  by
  rw  [lt_iff_le_not_ge]
  sorry

example  :  a  <  b  →  b  <  c  →  a  <  c  :=  by
  simp  only  [lt_iff_le_not_ge]
  sorry 
```

## 3.5. 析取

证明析取 `A ∨ B` 的规范方法是证明 `A` 或证明 `B`。`left` 策略选择 `A`，而 `right` 策略选择 `B`。

```py
variable  {x  y  :  ℝ}

example  (h  :  y  >  x  ^  2)  :  y  >  0  ∨  y  <  -1  :=  by
  left
  linarith  [pow_two_nonneg  x]

example  (h  :  -y  >  x  ^  2  +  1)  :  y  >  0  ∨  y  <  -1  :=  by
  right
  linarith  [pow_two_nonneg  x] 
```

我们不能使用匿名构造函数来构造“或”的证明，因为 Lean 必须猜测我们正在尝试证明哪个析取项。当我们编写证明项时，我们可以使用`Or.inl`和`Or.inr`来明确选择。在这里，`inl`代表“引入左”，`inr`代表“引入右”。

```py
example  (h  :  y  >  0)  :  y  >  0  ∨  y  <  -1  :=
  Or.inl  h

example  (h  :  y  <  -1)  :  y  >  0  ∨  y  <  -1  :=
  Or.inr  h 
```

通过证明一个析取项的任一边来证明析取似乎很奇怪。在实践中，哪个情况成立通常取决于假设和数据中的隐式或显式的情况区分。`rcases`策略允许我们利用形式为`A ∨ B`的假设。与使用`rcases`与合取或存在量词相比，这里的`rcases`策略产生*两个*目标。这两个目标有相同的结论，但在第一种情况下，假设`A`为真，在第二种情况下，假设`B`为真。换句话说，正如其名称所暗示的，`rcases`策略通过分情况进行证明。像往常一样，我们可以告诉 Lean 使用哪些名称作为假设。在下一个例子中，我们告诉 Lean 在每个分支上使用名称`h`。

```py
example  :  x  <  |y|  →  x  <  y  ∨  x  <  -y  :=  by
  rcases  le_or_gt  0  y  with  h  |  h
  ·  rw  [abs_of_nonneg  h]
  intro  h;  left;  exact  h
  ·  rw  [abs_of_neg  h]
  intro  h;  right;  exact  h 
```

注意，模式从合取的情况中的`⟨h₀, h₁⟩`变为析取的情况中的`h₀ | h₁`。将第一个模式视为与包含`h₀`和`h₁`的*两者*的数据匹配，而第二个模式，带有竖线，与包含*任一*`h₀`或`h₁`的数据匹配。在这种情况下，因为两个目标是独立的，所以我们选择在每个情况下使用相同的名称，即`h`。

绝对值函数被定义为一种方式，我们可以立即证明`x ≥ 0`意味着`|x| = x`（这是定理`abs_of_nonneg`），以及`x < 0`意味着`|x| = -x`（这是`abs_of_neg`）。表达式`le_or_gt 0 x`建立了`0 ≤ x ∨ x < 0`，允许我们在这两种情况下进行拆分。

Lean 还支持计算机科学家的析取模式匹配语法。现在`cases`策略更有吸引力，因为它允许我们为每个`case`命名，并为引入的假设命名，使其更接近使用位置。

```py
example  :  x  <  |y|  →  x  <  y  ∨  x  <  -y  :=  by
  cases  le_or_gt  0  y
  case  inl  h  =>
  rw  [abs_of_nonneg  h]
  intro  h;  left;  exact  h
  case  inr  h  =>
  rw  [abs_of_neg  h]
  intro  h;  right;  exact  h 
```

`inl`和`inr`的名称分别代表“intro left”和“intro right”。使用`case`的优点是你可以按任意顺序证明情况；Lean 使用标签来找到相关的目标。如果你不关心这一点，你可以使用`next`，或者`match`，甚至是一个模式匹配的`have`。

```py
example  :  x  <  |y|  →  x  <  y  ∨  x  <  -y  :=  by
  cases  le_or_gt  0  y
  next  h  =>
  rw  [abs_of_nonneg  h]
  intro  h;  left;  exact  h
  next  h  =>
  rw  [abs_of_neg  h]
  intro  h;  right;  exact  h

example  :  x  <  |y|  →  x  <  y  ∨  x  <  -y  :=  by
  match  le_or_gt  0  y  with
  |  Or.inl  h  =>
  rw  [abs_of_nonneg  h]
  intro  h;  left;  exact  h
  |  Or.inr  h  =>
  rw  [abs_of_neg  h]
  intro  h;  right;  exact  h 
```

在`match`的情况下，我们需要使用规范方式证明析取的完整名称`Or.inl`和`Or.inr`。在这本教科书中，我们将通常使用`rcases`来根据析取的情况进行拆分。

尝试使用下一片段中的前两个定理来证明三角不等式。它们在 Mathlib 中的名称与它们相同。

```py
namespace  MyAbs

theorem  le_abs_self  (x  :  ℝ)  :  x  ≤  |x|  :=  by
  sorry

theorem  neg_le_abs_self  (x  :  ℝ)  :  -x  ≤  |x|  :=  by
  sorry

theorem  abs_add  (x  y  :  ℝ)  :  |x  +  y|  ≤  |x|  +  |y|  :=  by
  sorry 
```

如果你喜欢这些（有意为之）并且想要更多关于析取的练习，请尝试这些。

```py
theorem  lt_abs  :  x  <  |y|  ↔  x  <  y  ∨  x  <  -y  :=  by
  sorry

theorem  abs_lt  :  |x|  <  y  ↔  -y  <  x  ∧  x  <  y  :=  by
  sorry 
```

你也可以使用 `rcases` 和 `rintro` 与嵌套析取一起使用。当这些导致具有多个目标的真正情况分割时，每个新目标的模式由一个垂直线分隔。

```py
example  {x  :  ℝ}  (h  :  x  ≠  0)  :  x  <  0  ∨  x  >  0  :=  by
  rcases  lt_trichotomy  x  0  with  xlt  |  xeq  |  xgt
  ·  left
  exact  xlt
  ·  contradiction
  ·  right;  exact  xgt 
```

你仍然可以嵌套模式并使用 `rfl` 关键字来替换等式：

```py
example  {m  n  k  :  ℕ}  (h  :  m  ∣  n  ∨  m  ∣  k)  :  m  ∣  n  *  k  :=  by
  rcases  h  with  ⟨a,  rfl⟩  |  ⟨b,  rfl⟩
  ·  rw  [mul_assoc]
  apply  dvd_mul_right
  ·  rw  [mul_comm,  mul_assoc]
  apply  dvd_mul_right 
```

看看你是否可以用一行（长）来证明以下内容。使用 `rcases` 来展开假设并分情况讨论，并使用分号和 `linarith` 来解决每个分支。

```py
example  {z  :  ℝ}  (h  :  ∃  x  y,  z  =  x  ^  2  +  y  ^  2  ∨  z  =  x  ^  2  +  y  ^  2  +  1)  :  z  ≥  0  :=  by
  sorry 
```

在实数上，等式 `x * y = 0` 告诉我们 `x = 0` 或 `y = 0`。在 Mathlib 中，这个事实被称为 `eq_zero_or_eq_zero_of_mul_eq_zero`，它是另一个很好的例子，说明了析取是如何产生的。看看你是否可以用它来证明以下内容：

```py
example  {x  :  ℝ}  (h  :  x  ^  2  =  1)  :  x  =  1  ∨  x  =  -1  :=  by
  sorry

example  {x  y  :  ℝ}  (h  :  x  ^  2  =  y  ^  2)  :  x  =  y  ∨  x  =  -y  :=  by
  sorry 
```

记住，你可以使用 `ring` 策略来帮助计算。

在一个任意的环 $R$ 中，一个元素 $x$，使得对于某个非零的 $y$ 有 $x y = 0$，被称为 *左零因子*，一个元素 $x$，使得对于某个非零的 $y$ 有 $y x = 0$，被称为 *右零因子*，而一个既是左零因子又是右零因子的元素被称为简单的 *零因子*。定理 `eq_zero_or_eq_zero_of_mul_eq_zero` 表明实数没有非平凡的零因子。具有这种性质的交换环被称为 *整环*。你上面两个定理的证明在任意整环中同样有效：

```py
variable  {R  :  Type*}  [CommRing  R]  [IsDomain  R]
variable  (x  y  :  R)

example  (h  :  x  ^  2  =  1)  :  x  =  1  ∨  x  =  -1  :=  by
  sorry

example  (h  :  x  ^  2  =  y  ^  2)  :  x  =  y  ∨  x  =  -y  :=  by
  sorry 
```

事实上，如果你小心的话，你可以不用乘法的交换律就能证明第一个定理。在这种情况下，只需假设 `R` 是一个 `Ring` 而不是 `CommRing` 即可。

有时在证明中，我们希望根据某个陈述是否为真来分情况讨论。对于任何命题 `P`，我们可以使用 `em P : P ∨ ¬ P`。名称 `em` 是“排中律”的缩写。

```py
example  (P  :  Prop)  :  ¬¬P  →  P  :=  by
  intro  h
  cases  em  P
  ·  assumption
  ·  contradiction 
```

或者，你可以使用 `by_cases` 策略。

```py
example  (P  :  Prop)  :  ¬¬P  →  P  :=  by
  intro  h
  by_cases  h'  :  P
  ·  assumption
  contradiction 
```

注意，`by_cases` 策略允许你为每个分支中引入的假设指定一个标签，在这种情况下，`h' : P` 在一个分支中，而在另一个分支中是 `h' : ¬ P`。如果你省略了标签，Lean 默认使用 `h`。尝试使用 `by_cases` 来证明以下等价性，以建立其中一个方向。

```py
example  (P  Q  :  Prop)  :  P  →  Q  ↔  ¬P  ∨  Q  :=  by
  sorry 
```

## 3.6\. 序列和收敛

现在我们已经掌握了足够的技能来进行一些真正的数学。在 Lean 中，我们可以将实数序列 $s_0, s_1, s_2, \ldots$ 表示为一个函数 `s : ℕ → ℝ`。这样的序列被称为 *收敛到* 一个数 $a$，如果对于每一个 $\varepsilon > 0$，存在一个点，在此点之后序列始终保持在 $a$ 的 $\varepsilon$ 范围内，也就是说，存在一个数 $N$，使得对于每一个 $n \ge N$，$| s_n - a | < \varepsilon$。在 Lean 中，我们可以这样表示：

```py
def  ConvergesTo  (s  :  ℕ  →  ℝ)  (a  :  ℝ)  :=
  ∀  ε  >  0,  ∃  N,  ∀  n  ≥  N,  |s  n  -  a|  <  ε 
```

符号 `∀ ε > 0, ...` 是 `∀ ε, ε > 0 → ...` 的方便缩写，同样地，`∀ n ≥ N, ...` 缩写为 `∀ n, n ≥ N → ...`。并且记住，`ε > 0`，反过来，定义为 `0 < ε`，而 `n ≥ N` 定义为 `N ≤ n`。

在本节中，我们将建立一些收敛性的性质。但首先，我们将讨论三种处理等式的方法，这些方法将非常有用。第一种，`ext` 方法，为我们提供了一种证明两个函数相等的方法。设 $f(x) = x + 1$ 和 $g(x) = 1 + x$ 是从实数到实数的函数。那么，当然，$f = g$，因为它们对每个 $x$ 都返回相同的值。`ext` 方法使我们能够通过证明在所有参数值上它们的值都相同来证明函数之间的等式。

```py
example  :  (fun  x  y  :  ℝ  ↦  (x  +  y)  ^  2)  =  fun  x  y  :  ℝ  ↦  x  ^  2  +  2  *  x  *  y  +  y  ^  2  :=  by
  ext
  ring 
```

我们将在后面看到，`ext` 实际上更通用，并且还可以指定出现变量的名称。例如，你可以在上面的证明中尝试将 `ext` 替换为 `ext u v`。第二种方法，`congr` 方法，允许我们通过协调不同的部分来证明两个表达式之间的等式：

```py
example  (a  b  :  ℝ)  :  |a|  =  |a  -  b  +  b|  :=  by
  congr
  ring 
```

在这里，`congr` 方法剥去了每边的 `abs`，留下我们证明 `a = a - b + b`。

最后，`convert` 方法用于在定理的结论与目标不完全匹配时将定理应用于目标。例如，假设我们想从 `1 < a` 证明 `a < a * a`。库中的一个定理 `mul_lt_mul_right` 将使我们能够证明 `1 * a < a * a`。一种可能性是反向工作并重写目标，使其具有那种形式。相反，`convert` 方法允许我们按原样应用定理，并留下证明所需方程的任务。

```py
example  {a  :  ℝ}  (h  :  1  <  a)  :  a  <  a  *  a  :=  by
  convert  (mul_lt_mul_right  _).2  h
  ·  rw  [one_mul]
  exact  lt_trans  zero_lt_one  h 
```

这个例子说明了另一个有用的技巧：当我们应用一个带有下划线的表达式，而 Lean 不能自动为我们填充它时，它简单地将其留给我们作为另一个目标。

下面的例子表明，任何常数序列 $a, a, a, \ldots$ 都会收敛。

```py
theorem  convergesTo_const  (a  :  ℝ)  :  ConvergesTo  (fun  x  :  ℕ  ↦  a)  a  :=  by
  intro  ε  εpos
  use  0
  intro  n  nge
  rw  [sub_self,  abs_zero]
  apply  εpos 
```

Lean 有一个名为 `simp` 的方法，它经常可以节省你手动执行 `rw [sub_self, abs_zero]` 等步骤的麻烦。我们很快就会告诉你更多关于它的信息。

对于一个更有趣的定理，让我们证明如果 `s` 收敛到 `a` 且 `t` 收敛到 `b`，那么 `fun n ↦ s n + t n` 收敛到 `a + b`。在开始编写正式证明之前，有一个清晰的笔和纸证明是有帮助的。给定大于 `0` 的 `ε`，想法是使用假设来获得一个 `Ns`，这样在这一点之后，`s` 就在 `a` 的 `ε / 2` 范围内，以及一个 `Nt`，这样在这一点之后，`t` 就在 `b` 的 `ε / 2` 范围内。然后，每当 `n` 大于或等于 `Ns` 和 `Nt` 的最大值时，序列 `fun n ↦ s n + t n` 应该在 `a + b` 的 `ε` 范围内。以下示例开始实施这种策略。看看你是否能完成它。

```py
theorem  convergesTo_add  {s  t  :  ℕ  →  ℝ}  {a  b  :  ℝ}
  (cs  :  ConvergesTo  s  a)  (ct  :  ConvergesTo  t  b)  :
  ConvergesTo  (fun  n  ↦  s  n  +  t  n)  (a  +  b)  :=  by
  intro  ε  εpos
  dsimp  -- this line is not needed but cleans up the goal a bit.
  have  ε2pos  :  0  <  ε  /  2  :=  by  linarith
  rcases  cs  (ε  /  2)  ε2pos  with  ⟨Ns,  hs⟩
  rcases  ct  (ε  /  2)  ε2pos  with  ⟨Nt,  ht⟩
  use  max  Ns  Nt
  sorry 
```

作为提示，你可以使用 `le_of_max_le_left` 和 `le_of_max_le_right`，以及 `norm_num` 可以证明 `ε / 2 + ε / 2 = ε`。此外，使用 `congr` 方法来证明 `|s n + t n - (a + b)|` 等于 `|(s n - a) + (t n - b)|` 是有帮助的，因为这样你就可以使用三角不等式。注意，我们标记了所有变量 `s`、`t`、`a` 和 `b` 为隐式，因为它们可以从假设中推断出来。

用乘法代替加法来证明相同的定理是棘手的。我们将通过首先证明一些辅助命题来达到这一点。看看你是否也能完成下一个证明，该证明表明如果`s`收敛到`a`，那么`fun n ↦ c * s n`收敛到`c * a`。根据`c`是否为零来分情况考虑是有帮助的。我们已经处理了零的情况，并留下了额外的假设`c`不为零来证明结果。

```py
theorem  convergesTo_mul_const  {s  :  ℕ  →  ℝ}  {a  :  ℝ}  (c  :  ℝ)  (cs  :  ConvergesTo  s  a)  :
  ConvergesTo  (fun  n  ↦  c  *  s  n)  (c  *  a)  :=  by
  by_cases  h  :  c  =  0
  ·  convert  convergesTo_const  0
  ·  rw  [h]
  ring
  rw  [h]
  ring
  have  acpos  :  0  <  |c|  :=  abs_pos.mpr  h
  sorry 
```

下一个定理也是独立有趣的：它表明收敛序列最终在绝对值上是有限的。我们已经给你开了一个头；看看你是否能完成它。

```py
theorem  exists_abs_le_of_convergesTo  {s  :  ℕ  →  ℝ}  {a  :  ℝ}  (cs  :  ConvergesTo  s  a)  :
  ∃  N  b,  ∀  n,  N  ≤  n  →  |s  n|  <  b  :=  by
  rcases  cs  1  zero_lt_one  with  ⟨N,  h⟩
  use  N,  |a|  +  1
  sorry 
```

事实上，这个定理可以被加强，以断言存在一个对所有`n`值都成立的界限`b`。但这个版本对我们来说已经足够强大，我们将在本节的末尾看到它具有更一般的适用性。

下一个引理是辅助性的：我们证明如果`s`收敛到`a`且`t`收敛到`0`，那么`fun n ↦ s n * t n`收敛到`0`。为此，我们使用前面的定理找到一个`B`，它从某个点`N₀`开始限制`s`。看看你是否能理解我们所概述的策略并完成证明。

```py
theorem  aux  {s  t  :  ℕ  →  ℝ}  {a  :  ℝ}  (cs  :  ConvergesTo  s  a)  (ct  :  ConvergesTo  t  0)  :
  ConvergesTo  (fun  n  ↦  s  n  *  t  n)  0  :=  by
  intro  ε  εpos
  dsimp
  rcases  exists_abs_le_of_convergesTo  cs  with  ⟨N₀,  B,  h₀⟩
  have  Bpos  :  0  <  B  :=  lt_of_le_of_lt  (abs_nonneg  _)  (h₀  N₀  (le_refl  _))
  have  pos₀  :  ε  /  B  >  0  :=  div_pos  εpos  Bpos
  rcases  ct  _  pos₀  with  ⟨N₁,  h₁⟩
  sorry 
```

如果你已经走到这一步，恭喜你！我们现在已经接近我们的定理了。下面的证明将把它完成。

```py
theorem  convergesTo_mul  {s  t  :  ℕ  →  ℝ}  {a  b  :  ℝ}
  (cs  :  ConvergesTo  s  a)  (ct  :  ConvergesTo  t  b)  :
  ConvergesTo  (fun  n  ↦  s  n  *  t  n)  (a  *  b)  :=  by
  have  h₁  :  ConvergesTo  (fun  n  ↦  s  n  *  (t  n  +  -b))  0  :=  by
  apply  aux  cs
  convert  convergesTo_add  ct  (convergesTo_const  (-b))
  ring
  have  :=  convergesTo_add  h₁  (convergesTo_mul_const  b  cs)
  convert  convergesTo_add  h₁  (convergesTo_mul_const  b  cs)  using  1
  ·  ext;  ring
  ring 
```

对于另一个挑战性的练习，尝试填写以下极限唯一的证明草稿。（如果你有勇气，你可以删除证明草稿，并尝试从头开始证明。）

```py
theorem  convergesTo_unique  {s  :  ℕ  →  ℝ}  {a  b  :  ℝ}
  (sa  :  ConvergesTo  s  a)  (sb  :  ConvergesTo  s  b)  :
  a  =  b  :=  by
  by_contra  abne
  have  :  |a  -  b|  >  0  :=  by  sorry
  let  ε  :=  |a  -  b|  /  2
  have  εpos  :  ε  >  0  :=  by
  change  |a  -  b|  /  2  >  0
  linarith
  rcases  sa  ε  εpos  with  ⟨Na,  hNa⟩
  rcases  sb  ε  εpos  with  ⟨Nb,  hNb⟩
  let  N  :=  max  Na  Nb
  have  absa  :  |s  N  -  a|  <  ε  :=  by  sorry
  have  absb  :  |s  N  -  b|  <  ε  :=  by  sorry
  have  :  |a  -  b|  <  |a  -  b|  :=  by  sorry
  exact  lt_irrefl  _  this 
```

我们在本节结束时注意到，我们的证明可以推广。例如，我们使用的自然数的唯一属性是它们的结构携带一个具有`min`和`max`的偏序。你可以检查，如果你在所有地方用任何线性序`α`替换`ℕ`，一切仍然有效：

```py
variable  {α  :  Type*}  [LinearOrder  α]

def  ConvergesTo'  (s  :  α  →  ℝ)  (a  :  ℝ)  :=
  ∀  ε  >  0,  ∃  N,  ∀  n  ≥  N,  |s  n  -  a|  <  ε 
```

在第 11.1 节中，我们将看到 Mathlib 有处理收敛的机制，这些机制在极其一般化的术语下进行，不仅抽象掉了定义域和值域的特定特征，而且还抽象了不同类型的收敛。
