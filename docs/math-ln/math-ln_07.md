# 7. 结构

> 原文：[`leanprover-community.github.io/mathematics_in_lean/C07_Structures.html`](https://leanprover-community.github.io/mathematics_in_lean/C07_Structures.html)

*Lean 中的数学* **   7. 结构

+   查看页面源代码

* * *

现代数学在本质上是使用代数结构，这些结构封装了可以在多个设置中实例化的模式。该主题提供了定义此类结构及其特定实例的多种方法。

因此，Lean 提供了定义结构及其操作的形式化方法。您已经看到了 Lean 中代数结构的例子，例如环和格，这些在第二章中讨论过。本章将解释您在那里看到的神秘方括号注解，`[Ring α]`和`[Lattice α]`。它还将向您展示如何定义和使用您自己的代数结构。

对于更详细的技术信息，您可以查阅[Lean 中的定理证明](https://leanprover.github.io/theorem_proving_in_lean/)，以及安妮·巴恩的一篇论文，[在 Lean 数学库中使用和滥用实例参数](https://arxiv.org/abs/2202.01629)。

## 7.1. 定义结构

在术语的最广泛意义上，*结构*是一组数据的规范，可能包含数据必须满足的约束。结构的*实例*是满足这些约束的特定数据包。例如，我们可以指定一个点是一组三个实数的元组：

```py
@[ext]
structure  Point  where
  x  :  ℝ
  y  :  ℝ
  z  :  ℝ 
```

`@[ext]`注解告诉 Lean 自动生成定理，这些定理可以用来证明当结构的组件相等时，两个结构实例是相等的，这种性质称为*扩展性*。

```py
#check  Point.ext

example  (a  b  :  Point)  (hx  :  a.x  =  b.x)  (hy  :  a.y  =  b.y)  (hz  :  a.z  =  b.z)  :  a  =  b  :=  by
  ext
  repeat'  assumption 
```

然后，我们可以定义`Point`结构的特定实例。Lean 提供了多种实现方式。

```py
def  myPoint1  :  Point  where
  x  :=  2
  y  :=  -1
  z  :=  4

def  myPoint2  :  Point  :=
  ⟨2,  -1,  4⟩

def  myPoint3  :=
  Point.mk  2  (-1)  4 
```

在第一个例子中，结构的字段被明确命名。在`myPoint3`的定义中提到的`Point.mk`函数被称为`Point`结构的*构造函数*，因为它用于构建元素。如果您想，也可以指定不同的名称，比如`build`。

```py
structure  Point'  where  build  ::
  x  :  ℝ
  y  :  ℝ
  z  :  ℝ

#check  Point'.build  2  (-1)  4 
```

下面的两个例子展示了如何在结构上定义函数。第二个例子使`Point.mk`构造函数显式化，而第一个例子为了简洁使用了匿名构造函数。Lean 可以从`add`指示的类型推断出相关的构造函数。将定义和与`Point`这样的结构相关的定理放在具有相同名称的命名空间中是一种惯例。在下面的例子中，因为我们已经打开了`Point`命名空间，所以`add`的全称是`Point.add`。当命名空间未打开时，我们必须使用全称。但请记住，使用匿名投影符号通常很方便，它允许我们写出`a.add b`而不是`Point.add a b`。因为`a`具有`Point`类型，Lean 将前者解释为后者。

```py
namespace  Point

def  add  (a  b  :  Point)  :  Point  :=
  ⟨a.x  +  b.x,  a.y  +  b.y,  a.z  +  b.z⟩

def  add'  (a  b  :  Point)  :  Point  where
  x  :=  a.x  +  b.x
  y  :=  a.y  +  b.y
  z  :=  a.z  +  b.z

#check  add  myPoint1  myPoint2
#check  myPoint1.add  myPoint2

end  Point

#check  Point.add  myPoint1  myPoint2
#check  myPoint1.add  myPoint2 
```

下面我们将继续在相关命名空间中放置定义，但我们将省略引用片段中的命名空间命令。为了证明加法函数的性质，我们可以使用`rw`来展开定义，并使用`ext`将结构中两个元素之间的等式简化为分量之间的等式。下面我们使用`protected`关键字，即使命名空间已打开，定理的名称也是`Point.add_comm`。这在我们想要避免与像`add_comm`这样的通用定理产生歧义时很有帮助。

```py
protected  theorem  add_comm  (a  b  :  Point)  :  add  a  b  =  add  b  a  :=  by
  rw  [add,  add]
  ext  <;>  dsimp
  repeat'  apply  add_comm

example  (a  b  :  Point)  :  add  a  b  =  add  b  a  :=  by  simp  [add,  add_comm] 
```

因为 Lean 可以内部展开定义和简化投影，有时我们想要的等式在语义上是成立的。

```py
theorem  add_x  (a  b  :  Point)  :  (a.add  b).x  =  a.x  +  b.x  :=
  rfl 
```

还可以使用模式匹配在结构上定义函数，方式类似于我们在第 5.2 节中定义递归函数的方式。下面的`addAlt`和`addAlt'`定义基本上是相同的；唯一的区别是我们第二个定义中使用了匿名构造函数符号。虽然有时以这种方式定义函数很方便，结构 eta-归约使这种替代定义在语义上等价，但它可能在后续的证明中使事情变得不那么方便。特别是，`rw [addAlt]`会给我们留下一个包含`match`语句的更混乱的目标视图。

```py
def  addAlt  :  Point  →  Point  →  Point
  |  Point.mk  x₁  y₁  z₁,  Point.mk  x₂  y₂  z₂  =>  ⟨x₁  +  x₂,  y₁  +  y₂,  z₁  +  z₂⟩

def  addAlt'  :  Point  →  Point  →  Point
  |  ⟨x₁,  y₁,  z₁⟩,  ⟨x₂,  y₂,  z₂⟩  =>  ⟨x₁  +  x₂,  y₁  +  y₂,  z₁  +  z₂⟩

theorem  addAlt_x  (a  b  :  Point)  :  (a.addAlt  b).x  =  a.x  +  b.x  :=  by
  rfl

theorem  addAlt_comm  (a  b  :  Point)  :  addAlt  a  b  =  addAlt  b  a  :=  by
  rw  [addAlt,  addAlt]
  -- the same proof still works, but the goal view here is harder to read
  ext  <;>  dsimp
  repeat'  apply  add_comm 
```

数学构造通常涉及将捆绑的信息拆分开来，并以不同的方式重新组合。因此，Lean 和 Mathlib 提供如此多的方法来高效地完成这项任务是有意义的。作为练习，尝试证明`Point.add`是结合的。然后定义点的标量乘法，并证明它对加法分配。

```py
protected  theorem  add_assoc  (a  b  c  :  Point)  :  (a.add  b).add  c  =  a.add  (b.add  c)  :=  by
  sorry

def  smul  (r  :  ℝ)  (a  :  Point)  :  Point  :=
  sorry

theorem  smul_distrib  (r  :  ℝ)  (a  b  :  Point)  :
  (smul  r  a).add  (smul  r  b)  =  smul  r  (a.add  b)  :=  by
  sorry 
```

使用结构只是通往代数抽象道路上的第一步。我们还没有将`Point.add`链接到通用`+`符号的方法，或者将`Point.add_comm`和`Point.add_assoc`连接到通用的`add_comm`和`add_assoc`定理。这些任务属于使用结构的*代数*方面，我们将在下一节中解释如何执行它们。现在，只需将结构视为一种将对象和信息捆绑在一起的方式。

特别有用的是，结构不仅可以指定数据类型，还可以指定数据必须满足的约束。在 Lean 中，后者表示为类型`Prop`的字段。例如，*标准 2-单纯形*被定义为满足$x ≥ 0$、$y ≥ 0$、$z ≥ 0$和$x + y + z = 1$的点集$(x, y, z)$。如果你不熟悉这个概念，你应该画一个图，并说服自己这个集合是三维空间中的等边三角形，其顶点为$(1, 0, 0)$、$(0, 1, 0)$和$(0, 0, 1)$，以及其内部。我们可以在 Lean 中表示如下：

```py
structure  StandardTwoSimplex  where
  x  :  ℝ
  y  :  ℝ
  z  :  ℝ
  x_nonneg  :  0  ≤  x
  y_nonneg  :  0  ≤  y
  z_nonneg  :  0  ≤  z
  sum_eq  :  x  +  y  +  z  =  1 
```

注意到最后四个字段指的是`x`、`y`和`z`，即前三个字段。我们可以定义一个从二单纯形到自身的映射，该映射交换`x`和`y`：

```py
def  swapXy  (a  :  StandardTwoSimplex)  :  StandardTwoSimplex
  where
  x  :=  a.y
  y  :=  a.x
  z  :=  a.z
  x_nonneg  :=  a.y_nonneg
  y_nonneg  :=  a.x_nonneg
  z_nonneg  :=  a.z_nonneg
  sum_eq  :=  by  rw  [add_comm  a.y  a.x,  a.sum_eq] 
```

更有趣的是，我们可以计算单纯形上两点之间的中点。我们在文件开头添加了`noncomputable section`短语，以便在实数上使用除法。

```py
noncomputable  section

def  midpoint  (a  b  :  StandardTwoSimplex)  :  StandardTwoSimplex
  where
  x  :=  (a.x  +  b.x)  /  2
  y  :=  (a.y  +  b.y)  /  2
  z  :=  (a.z  +  b.z)  /  2
  x_nonneg  :=  div_nonneg  (add_nonneg  a.x_nonneg  b.x_nonneg)  (by  norm_num)
  y_nonneg  :=  div_nonneg  (add_nonneg  a.y_nonneg  b.y_nonneg)  (by  norm_num)
  z_nonneg  :=  div_nonneg  (add_nonneg  a.z_nonneg  b.z_nonneg)  (by  norm_num)
  sum_eq  :=  by  field_simp;  linarith  [a.sum_eq,  b.sum_eq] 
```

在这里，我们用简洁的证明术语建立了`x_nonneg`、`y_nonneg`和`z_nonneg`，但在战术模式下使用`by`来建立`sum_eq`。

给定一个满足$0 \le \lambda \le 1$的参数$\lambda$，我们可以取标准 2-单纯形上两点$a$和$b$的加权平均值$\lambda a + (1 - \lambda) b$。我们挑战你定义这个函数，类似于上面的`midpoint`函数。

```py
def  weightedAverage  (lambda  :  Real)  (lambda_nonneg  :  0  ≤  lambda)  (lambda_le  :  lambda  ≤  1)
  (a  b  :  StandardTwoSimplex)  :  StandardTwoSimplex  :=
  sorry 
```

结构可以依赖于参数。例如，我们可以将标准的 2-单纯形推广到任何$n$的标准$n$-单纯形。在这个阶段，你不必了解类型`Fin n`的任何东西，除了它有$n$个元素，以及 Lean 知道如何对其求和。

```py
open  BigOperators

structure  StandardSimplex  (n  :  ℕ)  where
  V  :  Fin  n  →  ℝ
  NonNeg  :  ∀  i  :  Fin  n,  0  ≤  V  i
  sum_eq_one  :  (∑  i,  V  i)  =  1

namespace  StandardSimplex

def  midpoint  (n  :  ℕ)  (a  b  :  StandardSimplex  n)  :  StandardSimplex  n
  where
  V  i  :=  (a.V  i  +  b.V  i)  /  2
  NonNeg  :=  by
  intro  i
  apply  div_nonneg
  ·  linarith  [a.NonNeg  i,  b.NonNeg  i]
  norm_num
  sum_eq_one  :=  by
  simp  [div_eq_mul_inv,  ←  Finset.sum_mul,  Finset.sum_add_distrib,
  a.sum_eq_one,  b.sum_eq_one]
  field_simp

end  StandardSimplex 
```

作为练习，看看你是否可以定义标准$n$-单纯形上两点的加权平均值。你可以使用`Finset.sum_add_distrib`和`Finset.mul_sum`来操作相关的和。

我们已经看到，结构可以被用来捆绑数据和属性。有趣的是，它们也可以用来捆绑属性而不包含数据。例如，下一个结构`IsLinear`捆绑了线性的两个组成部分。

```py
structure  IsLinear  (f  :  ℝ  →  ℝ)  where
  is_additive  :  ∀  x  y,  f  (x  +  y)  =  f  x  +  f  y
  preserves_mul  :  ∀  x  c,  f  (c  *  x)  =  c  *  f  x

section
variable  (f  :  ℝ  →  ℝ)  (linf  :  IsLinear  f)

#check  linf.is_additive
#check  linf.preserves_mul

end 
```

值得指出的是，结构并不是捆绑数据的唯一方式。`Point`数据结构可以使用通用类型积来定义，而`IsLinear`可以用简单的`and`来定义。

```py
def  Point''  :=
  ℝ  ×  ℝ  ×  ℝ

def  IsLinear'  (f  :  ℝ  →  ℝ)  :=
  (∀  x  y,  f  (x  +  y)  =  f  x  +  f  y)  ∧  ∀  x  c,  f  (c  *  x)  =  c  *  f  x 
```

通用类型构造甚至可以用作具有组件之间依赖的结构。例如，*子类型*构造将数据与属性结合起来。你可以将下一个示例中的类型`PReal`视为正实数的类型。任何`x : PReal`都有两个组成部分：值和正属性。你可以通过`x.val`访问这些组件，它具有类型`ℝ`，以及`x.property`，它代表`0 < x.val`这一事实。

```py
def  PReal  :=
  {  y  :  ℝ  //  0  <  y  }

section
variable  (x  :  PReal)

#check  x.val
#check  x.property
#check  x.1
#check  x.2

end 
```

我们本可以使用子类型来定义标准的 2-单纯形，以及任意$n$的标准的$n$-单纯形。

```py
def  StandardTwoSimplex'  :=
  {  p  :  ℝ  ×  ℝ  ×  ℝ  //  0  ≤  p.1  ∧  0  ≤  p.2.1  ∧  0  ≤  p.2.2  ∧  p.1  +  p.2.1  +  p.2.2  =  1  }

def  StandardSimplex'  (n  :  ℕ)  :=
  {  v  :  Fin  n  →  ℝ  //  (∀  i  :  Fin  n,  0  ≤  v  i)  ∧  (∑  i,  v  i)  =  1  } 
```

类似地，*Sigma 类型* 是有序对的推广，其中第二个组件的类型取决于第一个组件的类型。

```py
def  StdSimplex  :=  Σ  n  :  ℕ,  StandardSimplex  n

section
variable  (s  :  StdSimplex)

#check  s.fst
#check  s.snd

#check  s.1
#check  s.2

end 
```

给定 `s : StdSimplex`，第一个组件 `s.fst` 是一个自然数，第二个组件是相应单纯形 `StandardSimplex s.fst` 的一个元素。Sigma 类型与子类型之间的区别在于，Sigma 类型的第二个组件是数据而不是命题。

尽管我们可以使用产品、子类型和 Sigma 类型来代替结构体，但使用结构体仍然具有许多优点。定义一个结构体会抽象出底层表示，并为访问组件的函数提供自定义名称。这使得证明更加健壮：仅依赖于结构体接口的证明，在定义更改时通常仍然有效，只要我们用新的定义重新定义旧的访问器。此外，正如我们即将看到的，Lean 提供了将结构体编织成丰富、相互关联的层次结构以及管理它们之间交互的支持。## 7.2\. 代数结构

为了阐明我们所说的“代数结构”的含义，考虑一些例子会有所帮助。

1.  一个 *部分有序集* 由一个集合 $P$ 和一个在 $P$ 上的二元关系 $\le$ 组成，该关系是传递的和自反的。

1.  一个 *群* 由一个集合 $G$ 和一个结合的二元运算、一个单位元素 $1$ 以及一个函数 $g \mapsto g^{-1}$，该函数为 $G$ 中的每个 $g$ 返回一个逆元素组成。如果一个群的操作是交换的，则该群是 *阿贝尔群* 或 *交换群*。

1.  一个 *格* 是一个具有交集和并集的部分有序集。

1.  一个 *环* 由一个（加法表示的）阿贝尔群 $(R, +, 0, x \mapsto -x)$ 和一个结合乘法运算 $\cdot$ 以及一个单位 $1$ 组成，使得乘法对加法分配。如果一个环的乘法是交换的，则该环是 *交换环*。

1.  一个 *有序环* $(R, +, 0, -, \cdot, 1, \le)$ 由一个环及其元素上的部分序组成，使得对于 $R$ 中的每个 $a$、$b$ 和 $c$，$a \le b$ 蕴含 $a + c \le b + c$，并且 $0 \le a$ 和 $0 \le b$ 蕴含 $0 \le a b$ 对于 $R$ 中的每个 $a$ 和 $b$ 成立。

1.  一个 *度量空间* 由一个集合 $X$ 和一个函数 $d : X \times X \to \mathbb{R}$ 组成，使得以下条件成立：

    +   对于 $X$ 中的每个 $x$ 和 $y$，有 $d(x, y) \ge 0$。

    +   如果且仅当 $x = y$ 时，$d(x, y) = 0$。

    +   对于 $X$ 中的每个 $x$ 和 $y$，有 $d(x, y) = d(y, x)$。

    +   对于 $X$ 中的每个 $x$、$y$ 和 $z$，有 $d(x, z) \le d(x, y) + d(y, z)$。

1.  一个 *拓扑空间* 由一个集合 $X$ 和一个集合 $\mathcal T$ 的子集组成，称为 $X$ 的 *开子集*，使得以下条件成立：

    +   空集和 $X$ 是开集。

    +   两个开集的交集是开集。

    +   开集的任意并集是开集。

在这些例子中，结构的元素属于一个集合，称为*载体集*，有时它代表整个结构。例如，当我们说“设$G$为一个群”然后“设$g \in G$”时，我们使用$G$来代表结构和它的载体。并非每个代数结构都以这种方式与单个载体集相关联。例如，一个*二部图*涉及两个集合之间的关系，正如*伽罗瓦连接*一样，一个*范畴*也涉及两个感兴趣的集合，通常称为*对象*和*态射*。

这些例子表明了证明辅助工具为了支持代数推理必须执行的一些事情。首先，它需要识别结构的具体实例。数系$\mathbb{Z}$、$\mathbb{Q}$和$\mathbb{R}$都是有序环，我们应该能够在这些实例中的任何一个上应用关于有序环的通用定理。有时一个具体集合可能以多种方式成为结构的实例。例如，除了$\mathbb{R}$上的常规拓扑，它是实分析的基础之外，我们还可以考虑$\mathbb{R}$上的*离散*拓扑，其中每个集合都是开集。

其次，证明辅助工具需要支持结构上的通用符号。在 Lean 中，符号`*`用于所有常规数系中的乘法，以及用于通用群和环中的乘法。当我们使用表达式`f x * y`时，Lean 必须使用关于`f`、`x`和`y`类型的知识来确定我们心中的乘法是哪一种。

第三，它需要处理结构可以通过各种方式从其他结构继承定义、定理和符号的事实。一些结构通过添加更多的公理来扩展其他结构。交换环仍然是环，所以任何在环中有意义的定义在交换环中也有意义，任何在环中成立的定理在交换环中也成立。一些结构通过添加更多数据来扩展其他结构。例如，任何环的加法部分是一个加法群。环结构添加了乘法和单位元，以及规范这些乘法和将它们与加法部分相关联的公理。有时我们可以用另一个结构来定义一个结构。任何度量空间都有一个与之相关的典型拓扑，即*度量空间拓扑*，还有可以与任何线性序相关联的各种拓扑。

最后，重要的是要记住，数学允许我们使用函数和运算来定义结构，就像我们使用函数和运算来定义数字一样。群的积和幂仍然是群。对于每一个 $n$，模 $n$ 的整数构成一个环，而对于每一个 $k > 0$，具有该环系数的多项式 $k \times k$ 矩阵再次构成一个环。因此，我们可以像计算它们的元素一样容易地计算结构。这意味着代数结构在数学中具有双重生活，既是对象集合的容器，也是它们自身的对象。证明辅助工具必须适应这种双重角色。

当处理与具有代数结构关联的类型元素时，证明辅助工具需要识别该结构并找到相关的定义、定理和符号。所有这些听起来像是一大堆工作，确实如此。但 Lean 使用一组基本的机制来完成这些任务。本节的目标是解释这些机制并展示如何使用它们。

第一个要素几乎是显而易见的：从形式上讲，代数结构是 第 7.1 节 中的结构。一个代数结构是一组满足某些公理假设的数据的规范，我们在 第 7.1 节 中看到，这正是 `structure` 命令设计用来适应的。这是一场天作之合！

给定一个数据类型 `α`，我们可以如下定义 `α` 上的群结构。

```py
structure  Group₁  (α  :  Type*)  where
  mul  :  α  →  α  →  α
  one  :  α
  inv  :  α  →  α
  mul_assoc  :  ∀  x  y  z  :  α,  mul  (mul  x  y)  z  =  mul  x  (mul  y  z)
  mul_one  :  ∀  x  :  α,  mul  x  one  =  x
  one_mul  :  ∀  x  :  α,  mul  one  x  =  x
  inv_mul_cancel  :  ∀  x  :  α,  mul  (inv  x)  x  =  one 
```

注意到类型 `α` 是 `Group₁` 定义中的一个 *参数*。因此，你应该将对象 `struc : Group₁ α` 视为 `α` 上的群结构。我们在 第 2.2 节 中看到，`inv_mul_cancel` 的对应物 `mul_inv_cancel` 是从其他群公理中得出的，因此没有必要将其添加到定义中。

这个群的定义与 Mathlib 中 `Group` 的定义相似，我们选择了 `Group₁` 这个名字来区分我们的版本。如果你输入 `#check Group` 并使用 Ctrl 点击定义，你会看到 Mathlib 版本的 `Group` 被定义为扩展另一个结构；我们将在稍后解释如何做到这一点。如果你输入 `#print Group`，你也会看到 Mathlib 版本的 `Group` 有许多额外的字段。由于我们将解释的原因，有时在结构中添加冗余信息是有用的，这样就可以为从核心数据定义的对象和函数提供额外的字段。现在不用担心这一点。请放心，我们的简化版本 `Group₁` 在道德上是与 Mathlib 使用的群定义相同的。

有时将类型与其结构捆绑在一起是有用的，Mathlib 也包含了一个与以下内容等价的 `Grp` 结构定义：

```py
structure  Grp₁  where
  α  :  Type*
  str  :  Group₁  α 
```

Mathlib 版本位于 `Mathlib.Algebra.Category.Grp.Basic` 中，如果您将此添加到示例文件开头的导入中，则可以 `#check` 它。

由于以下原因将变得更为清晰，通常更有用将类型 `α` 与结构 `Group α` 分开。我们将这两个对象合称为 *部分捆绑结构*，因为表示结合了大多数但不是所有组件到一个结构中。在 Mathlib 中，当它用作群的同构类型时，通常使用大写罗马字母如 `G` 表示类型。

让我们构建一个群，也就是说，`Group₁` 类型的元素。对于任何一对类型 `α` 和 `β`，Mathlib 定义了 `α` 和 `β` 之间的 *等价* 类型 `Equiv α β`。Mathlib 还为这个类型定义了有启发性的符号 `α ≃ β`。元素 `f : α ≃ β` 是 `α` 和 `β` 之间的双射，由四个组件表示：从 `α` 到 `β` 的函数 `f.toFun`，从 `β` 到 `α` 的逆函数 `f.invFun`，以及两个指定这些函数确实是彼此的逆的性质。

```py
variable  (α  β  γ  :  Type*)
variable  (f  :  α  ≃  β)  (g  :  β  ≃  γ)

#check  Equiv  α  β
#check  (f.toFun  :  α  →  β)
#check  (f.invFun  :  β  →  α)
#check  (f.right_inv  :  ∀  x  :  β,  f  (f.invFun  x)  =  x)
#check  (f.left_inv  :  ∀  x  :  α,  f.invFun  (f  x)  =  x)
#check  (Equiv.refl  α  :  α  ≃  α)
#check  (f.symm  :  β  ≃  α)
#check  (f.trans  g  :  α  ≃  γ) 
```

注意最后三个构造的命名非常具有创意。我们认为恒等函数 `Equiv.refl`、逆操作 `Equiv.symm` 和组合操作 `Equiv.trans` 是显式证据，表明属于双射对应关系的性质是一个等价关系。

还要注意，`f.trans g` 需要按逆序组合前向函数。Mathlib 已经声明了一个从 `Equiv α β` 到函数类型 `α → β` 的 *强制转换*，因此我们可以省略写入 `.toFun`，由 Lean 为我们插入。

```py
example  (x  :  α)  :  (f.trans  g).toFun  x  =  g.toFun  (f.toFun  x)  :=
  rfl

example  (x  :  α)  :  (f.trans  g)  x  =  g  (f  x)  :=
  rfl

example  :  (f.trans  g  :  α  →  γ)  =  g  ∘  f  :=
  rfl 
```

Mathlib 还定义了 `perm α` 类型，它是 `α` 与自身之间的等价关系类型。

```py
example  (α  :  Type*)  :  Equiv.Perm  α  =  (α  ≃  α)  :=
  rfl 
```

应该很明显，`Equiv.Perm α` 在等价关系的组合下形成一个群。我们这样安排，使得 `mul f g` 等于 `g.trans f`，其前向函数是 `f ∘ g`。换句话说，乘法就是我们通常认为的双射的组合。我们在这里定义这个群：

```py
def  permGroup  {α  :  Type*}  :  Group₁  (Equiv.Perm  α)
  where
  mul  f  g  :=  Equiv.trans  g  f
  one  :=  Equiv.refl  α
  inv  :=  Equiv.symm
  mul_assoc  f  g  h  :=  (Equiv.trans_assoc  _  _  _).symm
  one_mul  :=  Equiv.trans_refl
  mul_one  :=  Equiv.refl_trans
  inv_mul_cancel  :=  Equiv.self_trans_symm 
```

事实上，Mathlib 在 `Algebra.Group.End` 文件中精确地定义了 `Equiv.Perm α` 上的这个 `Group` 结构。一如既往，您可以在 `permGroup` 定义中悬停查看使用的定理以查看它们的陈述，并且您可以跳转到原始文件中的定义以了解更多关于它们是如何实现的。

在普通数学中，我们通常认为符号与结构是独立的。例如，我们可以考虑群 $(G_1, \cdot, 1, \cdot^{-1})$、$(G_2, \circ, e, i(\cdot))$ 和 $(G_3, +, 0, -)$。在第一种情况下，我们用 $\cdot$ 写二元运算，用 $1$ 写单位元，用 $x \mapsto x^{-1}$ 写逆函数。在第二种和第三种情况下，我们使用显示的符号替代。然而，当我们形式化 Lean 中的群概念时，符号与结构联系得更紧密。在 Lean 中，任何 `Group` 的组件被命名为 `mul`、`one` 和 `inv`，我们很快就会看到乘法符号是如何设置来引用它们的。如果我们想使用加法符号，我们则使用同构结构 `AddGroup`（加法群的底层结构）。它的组件被命名为 `add`、`zero` 和 `neg`，相关的符号正是你所期望的。

回想一下我们在 第 7.1 节 中定义的 `Point` 类型以及我们那里定义的加法函数。这些定义在伴随本节的示例文件中重现。作为一个练习，定义一个类似于我们上面定义的 `Group₁` 结构的 `AddGroup₁` 结构，除了它使用前面描述的加法命名方案。在 `Point` 数据类型上定义否定和零，并在 `Point` 上定义 `AddGroup₁` 结构。

```py
structure  AddGroup₁  (α  :  Type*)  where
  (add  :  α  →  α  →  α)
  -- fill in the rest
@[ext]
structure  Point  where
  x  :  ℝ
  y  :  ℝ
  z  :  ℝ

namespace  Point

def  add  (a  b  :  Point)  :  Point  :=
  ⟨a.x  +  b.x,  a.y  +  b.y,  a.z  +  b.z⟩

def  neg  (a  :  Point)  :  Point  :=  sorry

def  zero  :  Point  :=  sorry

def  addGroupPoint  :  AddGroup₁  Point  :=  sorry

end  Point 
```

我们正在取得进展。现在我们知道了如何在 Lean 中定义代数结构，也知道如何定义这些结构的实例。但我们还希望将符号与结构关联起来，以便我们可以与每个实例一起使用它。此外，我们希望安排好，可以定义结构上的一个操作并使用任何特定的实例，我们还想安排好，可以证明关于结构的一个定理并使用任何实例。

实际上，Mathlib 已经配置好了使用 `Equiv.Perm α` 的泛型群符号、定义和定理。

```py
variable  {α  :  Type*}  (f  g  :  Equiv.Perm  α)  (n  :  ℕ)

#check  f  *  g
#check  mul_assoc  f  g  g⁻¹

-- group power, defined for any group
#check  g  ^  n

example  :  f  *  g  *  g⁻¹  =  f  :=  by  rw  [mul_assoc,  mul_inv_cancel,  mul_one]

example  :  f  *  g  *  g⁻¹  =  f  :=
  mul_inv_cancel_right  f  g

example  {α  :  Type*}  (f  g  :  Equiv.Perm  α)  :  g.symm.trans  (g.trans  f)  =  f  :=
  mul_inv_cancel_right  f  g 
```

你可以检查一下，我们上面要求你定义的 `Point` 上的加法群结构并不是这样。我们的任务现在是要理解那些使 `Equiv.Perm α` 的例子按预期工作的底层魔法。

问题是 Lean 需要能够 *找到* 相关的符号和隐含的群结构，使用我们在输入的表达式中找到的信息。同样，当我们用类型为 `ℝ` 的表达式 `x` 和 `y` 写出 `x + y` 时，Lean 需要解释 `+` 符号为实数上的相关加法函数。它还必须识别类型 `ℝ` 为交换环的一个实例，这样就可以使用所有关于交换环的定义和定理。例如，连续性在 Lean 中是在任何两个拓扑空间上定义的。当我们有 `f : ℝ → ℂ` 并写出 `Continuous f` 时，Lean 必须找到 `ℝ` 和 `ℂ` 上的相关拓扑。

魔法是通过三个事物的结合实现的。

1.  *逻辑*。任何组中应该被解释的定义，将组类型和组结构作为参数。同样，关于任意组元素的定理以组类型和组结构的全称量词开始。

1.  *隐式参数*。类型和结构的参数通常被省略，这样我们就不必写它们或在 Lean 信息窗口中看到它们。Lean 会默默地为我们填写信息。

1.  *类型类推断*，也称为*类推断*，这是一个简单但强大的机制，使我们能够为 Lean 注册信息以供以后使用。当 Lean 被调用以填充定义、定理或符号中的隐式参数时，它可以利用已注册的信息。

而一个注解`(grp : Group G)`告诉 Lean 它应该明确地给出该参数，注解`{grp : Group G}`告诉 Lean 它应该从表达式的上下文中推断出它，注解`[grp : Group G]`告诉 Lean 应该使用类型类推断来合成相应的参数。由于使用此类参数的整个目的是我们通常不需要明确地引用它们，Lean 允许我们写出`[Group G]`并保留名称匿名。你可能已经注意到 Lean 会自动选择像`_inst_1`这样的名称。当我们使用匿名方括号注解与`variables`命令一起使用时，只要变量仍然在作用域内，Lean 就会自动将参数`[Group G]`添加到任何提及`G`的定义或定理中。

我们如何注册 Lean 需要用来执行搜索的信息？回到我们的组示例，我们只需要做两个更改。首先，我们不再使用`structure`命令来定义组结构，而是使用关键字`class`来表示它是一个类推断的候选。其次，我们不再使用`def`来定义特定的实例，而是使用关键字`instance`来在 Lean 中注册特定的实例。与类变量的名称一样，我们可以保留实例定义的名称为匿名，因为在一般情况下，我们希望 Lean 找到它并使用它，而不必让我们烦恼于细节。

```py
class  Group₂  (α  :  Type*)  where
  mul  :  α  →  α  →  α
  one  :  α
  inv  :  α  →  α
  mul_assoc  :  ∀  x  y  z  :  α,  mul  (mul  x  y)  z  =  mul  x  (mul  y  z)
  mul_one  :  ∀  x  :  α,  mul  x  one  =  x
  one_mul  :  ∀  x  :  α,  mul  one  x  =  x
  inv_mul_cancel  :  ∀  x  :  α,  mul  (inv  x)  x  =  one

instance  {α  :  Type*}  :  Group₂  (Equiv.Perm  α)  where
  mul  f  g  :=  Equiv.trans  g  f
  one  :=  Equiv.refl  α
  inv  :=  Equiv.symm
  mul_assoc  f  g  h  :=  (Equiv.trans_assoc  _  _  _).symm
  one_mul  :=  Equiv.trans_refl
  mul_one  :=  Equiv.refl_trans
  inv_mul_cancel  :=  Equiv.self_trans_symm 
```

以下说明了它们的使用。

```py
#check  Group₂.mul

def  mySquare  {α  :  Type*}  [Group₂  α]  (x  :  α)  :=
  Group₂.mul  x  x

#check  mySquare

section
variable  {β  :  Type*}  (f  g  :  Equiv.Perm  β)

example  :  Group₂.mul  f  g  =  g.trans  f  :=
  rfl

example  :  mySquare  f  =  f.trans  f  :=
  rfl

end 
```

`#check` 命令显示 `Group₂.mul` 有一个隐含的参数 `[Group₂ α]`，我们期望通过类推断找到它，其中 `α` 是 `Group₂.mul` 参数的类型。换句话说，`{α : Type*}` 是群元素类型的隐含参数，`[Group₂ α]` 是 `α` 上的群结构的隐含参数。同样，当我们为 `Group₂` 定义一个泛型平方函数 `my_square` 时，我们使用一个隐含参数 `{α : Type*}` 用于元素类型和一个隐含参数 `[Group₂ α]` 用于 `Group₂` 结构。

在第一个例子中，当我们编写 `Group₂.mul f g` 时，`f` 和 `g` 的类型告诉 Lean 在 `Group₂.mul` 的参数 `α` 必须实例化为 `Equiv.Perm β`。这意味着 Lean 必须找到一个 `Group₂ (Equiv.Perm β)` 的元素。之前的 `instance` 声明告诉 Lean 如何做到这一点。问题解决了！

这种简单的机制，用于注册信息以便 Lean 在需要时能够找到它，非常有用。这里有一种出现的方式。在 Lean 的基础上，数据类型 `α` 可能是空的。然而，在许多应用中，知道一个类型至少有一个元素是有用的。例如，函数 `List.headI`，它返回列表的第一个元素，当列表为空时可以返回默认值。为了使这一点工作，Lean 库定义了一个类 `Inhabited α`，它所做的只是存储一个默认值。我们可以证明 `Point` 类型是一个实例：

```py
instance  :  Inhabited  Point  where  default  :=  ⟨0,  0,  0⟩

#check  (default  :  Point)

example  :  ([]  :  List  Point).headI  =  default  :=
  rfl 
```

类推断机制也用于泛型符号。表达式 `x + y` 是 `Add.add x y` 的缩写，其中——正如你所猜到的——`Add α` 是一个存储在 `α` 上的二元函数的类。编写 `x + y` 告诉 Lean 寻找一个已注册的 `[Add.add α]` 实例并使用相应的函数。下面，我们注册了 `Point` 的加法函数。

```py
instance  :  Add  Point  where  add  :=  Point.add

section
variable  (x  y  :  Point)

#check  x  +  y

example  :  x  +  y  =  Point.add  x  y  :=
  rfl

end 
```

以这种方式，我们还可以将符号 `+` 分配给其他类型的二元运算。

但我们还能做得更好。我们已经看到 `*` 可以在任何群中使用，`+` 可以在任何加法群中使用，并且两者都可以在任何环中使用。当我们定义 Lean 中环的新实例时，我们不需要为该实例定义 `+` 和 `*`，因为 Lean 知道这些为每个环都定义了。我们可以使用这种方法为我们的 `Group₂` 类指定符号：

```py
instance  {α  :  Type*}  [Group₂  α]  :  Mul  α  :=
  ⟨Group₂.mul⟩

instance  {α  :  Type*}  [Group₂  α]  :  One  α  :=
  ⟨Group₂.one⟩

instance  {α  :  Type*}  [Group₂  α]  :  Inv  α  :=
  ⟨Group₂.inv⟩

section
variable  {α  :  Type*}  (f  g  :  Equiv.Perm  α)

#check  f  *  1  *  g⁻¹

def  foo  :  f  *  1  *  g⁻¹  =  g.symm.trans  ((Equiv.refl  α).trans  f)  :=
  rfl

end 
```

使这种方法有效的是 Lean 执行递归搜索。根据我们声明的实例，Lean 可以通过找到一个 `Group₂ (Equiv.Perm α)` 的实例来找到一个 `Mul (Equiv.Perm α)` 的实例，并且它可以通过我们提供的一个实例来找到一个 `Group₂ (Equiv.Perm α)` 的实例。Lean 能够找到这两个事实并将它们链接在一起。

我们刚才给出的例子是危险的，因为 Lean 的库中也有一个 `Group (Equiv.Perm α)` 的实例，并且乘法在任意群上都是定义好的。所以，哪个实例被找到是不确定的。实际上，Lean 优先选择较新的声明，除非你明确指定了不同的优先级。此外，还有另一种方法告诉 Lean 一个结构是另一个结构的实例，即使用 `extends` 关键字。这就是 Mathlib 如何指定，例如，每个交换环都是一个环。你可以在第八部分和*《Lean 中的定理证明》*中的一个关于类推断的[章节](https://leanprover.github.io/theorem_proving_in_lean4/type_classes.html#managing-type-class-inference)中找到更多信息。

通常，为已经定义了表示法的代数结构的实例指定 `*` 的值是一个坏主意。在 Lean 中重新定义 `Group` 的概念是一个人为的例子。然而，在这种情况下，两种对群表示法的解释都展开为 `Equiv.trans`、`Equiv.refl` 和 `Equiv.symm`，方式相同。

作为一项类似的练习，定义一个类似于 `Group₂` 的类 `AddGroup₂`。在任意的 `AddGroup₂` 上使用类 `Add`、`Neg` 和 `Zero` 定义加法、取反和零的常规表示法。然后证明 `Point` 是 `AddGroup₂` 的一个实例。尝试一下，确保加法群表示法适用于 `Point` 的元素。

```py
class  AddGroup₂  (α  :  Type*)  where
  add  :  α  →  α  →  α
  -- fill in the rest 
```

我们已经在上文中为 `Point` 声明了 `Add`、`Neg` 和 `Zero` 的实例，这并不是一个大问题。再次强调，两种合成表示法应该得出相同的答案。

类推断是微妙的，使用时必须小心，因为它配置了无形地控制我们输入的表达式解释的自动化。然而，当明智地使用时，类推断是一个强大的工具。它是 Lean 中代数推理成为可能的原因。## 7.3. 构建高斯整数

我们将通过构建一个重要的数学对象——*高斯整数*，并展示它是一个欧几里得域，来展示 Lean 中代数层次结构的用法。换句话说，根据我们一直在使用的术语，我们将定义高斯整数，并展示它们是欧几里得域结构的实例。

用普通的数学术语来说，高斯整数集 $\Bbb{Z}[i]$ 是复数集 $\{ a + b i \mid a, b \in \Bbb{Z}\}$ 的集合。但与其将它们定义为复数的一个子集，我们的目标是在这里将它们定义为一个独立的数据类型。我们通过将高斯整数表示为一对整数来实现这一点，我们将这对整数视为*实部*和*虚部*。

```py
@[ext]
structure  GaussInt  where
  re  :  ℤ
  im  :  ℤ 
```

我们首先证明高斯整数具有环的结构，其中 `0` 定义为 `⟨0, 0⟩`，`1` 定义为 `⟨1, 0⟩`，加法定义为逐点。为了确定乘法定义，记住我们希望元素 $i$，由 `⟨0, 1⟩` 表示，是 $-1$ 的一个平方根。因此我们希望

$$\begin{split}(a + bi) (c + di) & = ac + bci + adi + bd i² \\ & = (ac - bd) + (bc + ad)i.\end{split}$$

这解释了下面 `Mul` 的定义。

```py
instance  :  Zero  GaussInt  :=
  ⟨⟨0,  0⟩⟩

instance  :  One  GaussInt  :=
  ⟨⟨1,  0⟩⟩

instance  :  Add  GaussInt  :=
  ⟨fun  x  y  ↦  ⟨x.re  +  y.re,  x.im  +  y.im⟩⟩

instance  :  Neg  GaussInt  :=
  ⟨fun  x  ↦  ⟨-x.re,  -x.im⟩⟩

instance  :  Mul  GaussInt  :=
  ⟨fun  x  y  ↦  ⟨x.re  *  y.re  -  x.im  *  y.im,  x.re  *  y.im  +  x.im  *  y.re⟩⟩ 
```

如 第 7.1 节 所述，将所有与数据类型相关的定义放在具有相同名称的命名空间中是一个好主意。因此，在本章相关的 Lean 文件中，这些定义是在 `GaussInt` 命名空间中进行的。

注意，在这里我们直接定义了符号 `0`、`1`、`+`、`-` 和 `*` 的解释，而不是将它们命名为 `GaussInt.zero` 等等，并将符号分配给它们。对于使用 `simp` 和 `rw` 等操作来说，有一个明确的名称对于定义来说通常是很有用的。

```py
theorem  zero_def  :  (0  :  GaussInt)  =  ⟨0,  0⟩  :=
  rfl

theorem  one_def  :  (1  :  GaussInt)  =  ⟨1,  0⟩  :=
  rfl

theorem  add_def  (x  y  :  GaussInt)  :  x  +  y  =  ⟨x.re  +  y.re,  x.im  +  y.im⟩  :=
  rfl

theorem  neg_def  (x  :  GaussInt)  :  -x  =  ⟨-x.re,  -x.im⟩  :=
  rfl

theorem  mul_def  (x  y  :  GaussInt)  :
  x  *  y  =  ⟨x.re  *  y.re  -  x.im  *  y.im,  x.re  *  y.im  +  x.im  *  y.re⟩  :=
  rfl 
```

给出计算实部和虚部的规则并声明给简化器也是有用的。

```py
@[simp]
theorem  zero_re  :  (0  :  GaussInt).re  =  0  :=
  rfl

@[simp]
theorem  zero_im  :  (0  :  GaussInt).im  =  0  :=
  rfl

@[simp]
theorem  one_re  :  (1  :  GaussInt).re  =  1  :=
  rfl

@[simp]
theorem  one_im  :  (1  :  GaussInt).im  =  0  :=
  rfl

@[simp]
theorem  add_re  (x  y  :  GaussInt)  :  (x  +  y).re  =  x.re  +  y.re  :=
  rfl

@[simp]
theorem  add_im  (x  y  :  GaussInt)  :  (x  +  y).im  =  x.im  +  y.im  :=
  rfl

@[simp]
theorem  neg_re  (x  :  GaussInt)  :  (-x).re  =  -x.re  :=
  rfl

@[simp]
theorem  neg_im  (x  :  GaussInt)  :  (-x).im  =  -x.im  :=
  rfl

@[simp]
theorem  mul_re  (x  y  :  GaussInt)  :  (x  *  y).re  =  x.re  *  y.re  -  x.im  *  y.im  :=
  rfl

@[simp]
theorem  mul_im  (x  y  :  GaussInt)  :  (x  *  y).im  =  x.re  *  y.im  +  x.im  *  y.re  :=
  rfl 
```

现在出人意料地容易证明高斯整数是交换环的一个实例。我们正在充分利用结构概念。每个特定的高斯整数是 `GaussInt` 结构的一个实例，而类型 `GaussInt` 本身以及相关的操作是 `CommRing` 结构的一个实例。`CommRing` 结构反过来又扩展了 `Zero`、`One`、`Add`、`Neg` 和 `Mul` 等符号结构。

如果你输入 `instance : CommRing GaussInt := _`，然后在 VS Code 中点击出现的灯泡图标，并让 Lean 填写结构定义的骨架，你会看到大量的条目。然而，跳转到结构的定义中，你会发现许多字段都有默认定义，Lean 会自动为你填写。下面列出了关键的定义。一个特殊情况是 `nsmul` 和 `zsmul`，目前可以忽略，将在下一章中解释。在每种情况下，相关恒等式都是通过展开定义、使用 `ext` 策略将恒等式简化到其实部和虚部、简化，并在必要时在整数中执行相关环运算来证明的。请注意，我们很容易避免重复所有这些代码，但这不是当前讨论的主题。

```py
instance  instCommRing  :  CommRing  GaussInt  where
  zero  :=  0
  one  :=  1
  add  :=  (·  +  ·)
  neg  x  :=  -x
  mul  :=  (·  *  ·)
  nsmul  :=  nsmulRec
  zsmul  :=  zsmulRec
  add_assoc  :=  by
  intros
  ext  <;>  simp  <;>  ring
  zero_add  :=  by
  intro
  ext  <;>  simp
  add_zero  :=  by
  intro
  ext  <;>  simp
  neg_add_cancel  :=  by
  intro
  ext  <;>  simp
  add_comm  :=  by
  intros
  ext  <;>  simp  <;>  ring
  mul_assoc  :=  by
  intros
  ext  <;>  simp  <;>  ring
  one_mul  :=  by
  intro
  ext  <;>  simp
  mul_one  :=  by
  intro
  ext  <;>  simp
  left_distrib  :=  by
  intros
  ext  <;>  simp  <;>  ring
  right_distrib  :=  by
  intros
  ext  <;>  simp  <;>  ring
  mul_comm  :=  by
  intros
  ext  <;>  simp  <;>  ring
  zero_mul  :=  by
  intros
  ext  <;>  simp
  mul_zero  :=  by
  intros
  ext  <;>  simp 
```

Lean 的库定义了 *非平凡* 类型为至少有两个不同元素的类型。在环的上下文中，这相当于说零不等于一。由于一些常见的定理依赖于这个事实，我们不妨现在就建立它。

```py
instance  :  Nontrivial  GaussInt  :=  by
  use  0,  1
  rw  [Ne,  GaussInt.ext_iff]
  simp 
```

我们现在将展示高斯整数具有一个重要的附加属性。一个 *欧几里得域* 是一个带有 *范数* 函数 $N : R \to \mathbb{N}$ 的环 $R$，它具有以下两个性质：

+   对于 $R$ 中的每一个 $a$ 和 $b \ne 0$，存在 $R$ 中的 $q$ 和 $r$，使得 $a = bq + r$，且 $r = 0$ 或 $N(r) < N(b)$。

+   对于每一个 $a$ 和 $b \ne 0$，$N(a) \le N(ab)$。

带有 $N(a) = |a|$ 的整数环 $\Bbb{Z}$ 是欧几里得域的一个典型例子。在这种情况下，我们可以取 $q$ 为 $a$ 除以 $b$ 的整数除法的结果，$r$ 为余数。这些函数在 Lean 中定义为满足以下条件：

```py
example  (a  b  :  ℤ)  :  a  =  b  *  (a  /  b)  +  a  %  b  :=
  Eq.symm  (Int.ediv_add_emod  a  b)

example  (a  b  :  ℤ)  :  b  ≠  0  →  0  ≤  a  %  b  :=
  Int.emod_nonneg  a

example  (a  b  :  ℤ)  :  b  ≠  0  →  a  %  b  <  |b|  :=
  Int.emod_lt_abs  a 
```

在一个任意的环中，一个元素 $a$ 被称为*单位*，如果它能整除 $1$。一个非零元素 $a$ 被称为*不可约*，如果它不能写成 $a = bc$ 的形式，其中 $b$ 和 $c$ 都不是单位。在整数中，每个不可约元素 $a$ 都是*素数*，也就是说，每当 $a$ 整除一个乘积 $bc$ 时，它要么整除 $b$，要么整除 $c$。但在其他环中，这个性质可能不成立。在环 $\Bbb{Z}[\sqrt{-5}]$ 中，我们有

$$6 = 2 \cdot 3 = (1 + \sqrt{-5})(1 - \sqrt{-5}),$$

并且元素 $2$、$3$、$1 + \sqrt{-5}$ 和 $1 - \sqrt{-5}$ 都是不可约的，但它们不是素数。例如，$2$ 能整除乘积 $(1 + \sqrt{-5})(1 - \sqrt{-5})$，但它不能整除任何一个因子。特别是，我们不再有唯一的分解：数字 $6$ 可以以多种方式分解为不可约元素。

相反，每个欧几里得域都是一个唯一分解域，这意味着每个不可约元素都是素数。欧几里得域的公理意味着可以写出任何非零元素为不可约元素的有限乘积。它们还意味着可以使用欧几里得算法找到任何两个非零元素 `a` 和 `b` 的最大公约数，即任何其他公约数都能整除的元素。这反过来又意味着不可约元素的分解是唯一的，直到乘以单位元素。

我们现在证明高斯整数是一个具有由 $N(a + bi) = (a + bi)(a - bi) = a² + b²$ 定义的范数的欧几里得域。高斯整数 $a - bi$ 被称为 $a + bi$ 的*共轭*。不难验证，对于任何复数 $x$ 和 $y$，我们有 $N(xy) = N(x)N(y)$。

为了看到这种范数的定义使得高斯整数成为一个欧几里得域，只有第一个性质是具有挑战性的。假设我们想要将 $a + bi = (c + di) q + r$ 写成合适的 $q$ 和 $r$。将 $a + bi$ 和 $c + di$ 作为复数处理，进行除法

$$\frac{a + bi}{c + di} = \frac{(a + bi)(c - di)}{(c + di)(c-di)} = \frac{ac + bd}{c² + d²} + \frac{bc -ad}{c²+d²} i.$$

实部和虚部可能不是整数，但我们可以将它们四舍五入到最近的整数 $u$ 和 $v$。然后我们可以将右侧表示为 $(u + vi) + (u' + v'i)$，其中 $u' + v'i$ 是剩余的部分。注意，我们有 $|u'| \le 1/2$ 和 $|v'| \le 1/2$，因此

$$N(u' + v' i) = (u')² + (v')² \le 1/4 + 1/4 \le 1/2.$$

乘以 $c + di$，我们得到

$$a + bi = (c + di) (u + vi) + (c + di) (u' + v'i).$$

将 $q = u + vi$ 和 $r = (c + di) (u' + v'i)$ 代入，我们得到 $a + bi = (c + di) q + r$，我们只需要对 $N(r)$ 进行界限定：

$$N(r) = N(c + di)N(u' + v'i) \le N(c + di) \cdot 1/2 < N(c + di).$$

我们刚才进行的论证需要将高斯整数视为复数集的子集。因此，在 Lean 中形式化它的一个选项是将高斯整数嵌入到复数中，将整数嵌入到高斯整数中，定义从实数到整数的舍入函数，并非常小心地在这些数系之间来回传递。事实上，这正是 Mathlib 中采用的方法，其中高斯整数本身被构造为二次整数环的特殊情况。参见文件 [GaussianInt.lean](https://github.com/leanprover-community/mathlib4/blob/master/Mathlib/NumberTheory/Zsqrtd/GaussianInt.lean)。

在这里，我们将进行一个保持在整数范围内的论证。这说明了在形式化数学时人们通常面临的选择。给定一个需要库中尚未存在的概念或工具的论证，你有两个选择：要么形式化所需的概念和工具，要么调整论证以利用你已有的概念和工具。当结果可以在其他上下文中使用时，第一个选择通常是时间的好投资。然而，从实用主义的角度来看，有时寻找更基础的证明可能更有效。

整数除法的商余定理指出，对于每一个 $a$ 和非零 $b$，存在 $q$ 和 $r$ 使得 $a = b q + r$ 且 $0 \le r < b$。在这里，我们将使用以下变体，它指出存在 $q'$ 和 $r'$ 使得 $a = b q' + r'$ 且 $|r'| \le b/2$。你可以检查，如果第一个陈述中的 $r$ 的值满足 $r \le b/2$，我们可以取 $q' = q$ 和 $r' = r$，否则我们可以取 $q' = q + 1$ 和 $r' = r - b$。我们感谢 Heather Macbeth 提出以下更优雅的方法，它避免了分情况定义。我们只需在除法之前将 `b / 2` 加到 `a` 上，然后从余数中减去它。

```py
def  div'  (a  b  :  ℤ)  :=
  (a  +  b  /  2)  /  b

def  mod'  (a  b  :  ℤ)  :=
  (a  +  b  /  2)  %  b  -  b  /  2

theorem  div'_add_mod'  (a  b  :  ℤ)  :  b  *  div'  a  b  +  mod'  a  b  =  a  :=  by
  rw  [div',  mod']
  linarith  [Int.ediv_add_emod  (a  +  b  /  2)  b]

theorem  abs_mod'_le  (a  b  :  ℤ)  (h  :  0  <  b)  :  |mod'  a  b|  ≤  b  /  2  :=  by
  rw  [mod',  abs_le]
  constructor
  ·  linarith  [Int.emod_nonneg  (a  +  b  /  2)  h.ne']
  have  :=  Int.emod_lt_of_pos  (a  +  b  /  2)  h
  have  :=  Int.ediv_add_emod  b  2
  have  :=  Int.emod_lt_of_pos  b  zero_lt_two
  linarith 
```

注意到我们老朋友 `linarith` 的使用。我们还需要将 `mod'` 用 `div'` 来表示。

```py
theorem  mod'_eq  (a  b  :  ℤ)  :  mod'  a  b  =  a  -  b  *  div'  a  b  :=  by  linarith  [div'_add_mod'  a  b] 
```

我们将使用 $x² + y²$ 仅当 $x$ 和 $y$ 都为零时等于零的事实。作为练习，我们要求你证明这在任何有序环中都成立。

```py
theorem  sq_add_sq_eq_zero  {α  :  Type*}  [Ring  α]  [LinearOrder  α]  [IsStrictOrderedRing  α]
  (x  y  :  α)  :  x  ^  2  +  y  ^  2  =  0  ↔  x  =  0  ∧  y  =  0  :=  by
  sorry 
```

我们将在这个部分的所有剩余定义和定理放入 `GaussInt` 命名空间。首先，我们定义 `norm` 函数并要求你建立其一些性质。证明都很简短。

```py
def  norm  (x  :  GaussInt)  :=
  x.re  ^  2  +  x.im  ^  2

@[simp]
theorem  norm_nonneg  (x  :  GaussInt)  :  0  ≤  norm  x  :=  by
  sorry
theorem  norm_eq_zero  (x  :  GaussInt)  :  norm  x  =  0  ↔  x  =  0  :=  by
  sorry
theorem  norm_pos  (x  :  GaussInt)  :  0  <  norm  x  ↔  x  ≠  0  :=  by
  sorry
theorem  norm_mul  (x  y  :  GaussInt)  :  norm  (x  *  y)  =  norm  x  *  norm  y  :=  by
  sorry 
```

接下来我们定义共轭函数：

```py
def  conj  (x  :  GaussInt)  :  GaussInt  :=
  ⟨x.re,  -x.im⟩

@[simp]
theorem  conj_re  (x  :  GaussInt)  :  (conj  x).re  =  x.re  :=
  rfl

@[simp]
theorem  conj_im  (x  :  GaussInt)  :  (conj  x).im  =  -x.im  :=
  rfl

theorem  norm_conj  (x  :  GaussInt)  :  norm  (conj  x)  =  norm  x  :=  by  simp  [norm] 
```

最后，我们用 `x / y` 的记号定义高斯整数的除法，它将复数商四舍五入到最接近的高斯整数。我们使用定制的 `Int.div'` 来实现这一点。正如我们上面所计算的，如果 `x` 是 $a + bi$ 且 `y` 是 $c + di$，那么 `x / y` 的实部和虚部是到 $c + di$ 的范数最近的整数。

$$\frac{ac + bd}{c² + d²} \quad \text{和} \quad \frac{bc -ad}{c²+d²},$$

分别。在这里，分子是 $(a + bi) (c - di)$ 的实部和虚部，分母都是 $c + di$ 的范数。

```py
instance  :  Div  GaussInt  :=
  ⟨fun  x  y  ↦  ⟨Int.div'  (x  *  conj  y).re  (norm  y),  Int.div'  (x  *  conj  y).im  (norm  y)⟩⟩ 
```

定义了 `x / y` 之后，我们定义 `x % y` 为余数，`x - (x / y) * y`。如上所述，我们在定理 `div_def` 和 `mod_def` 中记录这些定义，以便我们可以使用它们与 `simp` 和 `rw` 一起使用。

```py
instance  :  Mod  GaussInt  :=
  ⟨fun  x  y  ↦  x  -  y  *  (x  /  y)⟩

theorem  div_def  (x  y  :  GaussInt)  :
  x  /  y  =  ⟨Int.div'  (x  *  conj  y).re  (norm  y),  Int.div'  (x  *  conj  y).im  (norm  y)⟩  :=
  rfl

theorem  mod_def  (x  y  :  GaussInt)  :  x  %  y  =  x  -  y  *  (x  /  y)  :=
  rfl 
```

这些定义立即给出了对于每一个 `x` 和 `y`，`x = y * (x / y) + x % y`，因此我们只需要证明当 `y` 不为零时，`x % y` 的范数小于 `y` 的范数。

我们刚刚定义了 `x / y` 的实部和虚部分别为 `div' (x * conj y).re (norm y)` 和 `div' (x * conj y).im (norm y)`。计算后，我们有

> `(x % y) * conj y = (x - x / y * y) * conj y = x * conj y - x / y * (y * conj y)`

右侧的实部和虚部正好是 `mod' (x * conj y).re (norm y)` 和 `mod' (x * conj y).im (norm y)`。根据 `div'` 和 `mod'` 的性质，这些保证小于或等于 `norm y / 2`。因此我们有

> `norm ((x % y) * conj y) ≤ (norm y / 2)² + (norm y / 2)² ≤ (norm y / 2) * norm y`。

另一方面，我们有

> `norm ((x % y) * conj y) = norm (x % y) * norm (conj y) = norm (x % y) * norm y`。

除以 `norm y`，我们得到 `norm (x % y) ≤ (norm y) / 2 < norm y`，正如所要求的。

这项繁杂的计算将在下一个证明中进行。我们鼓励你逐步查看细节，看看你是否能找到一个更简洁的论证。

```py
theorem  norm_mod_lt  (x  :  GaussInt)  {y  :  GaussInt}  (hy  :  y  ≠  0)  :
  (x  %  y).norm  <  y.norm  :=  by
  have  norm_y_pos  :  0  <  norm  y  :=  by  rwa  [norm_pos]
  have  H1  :  x  %  y  *  conj  y  =  ⟨Int.mod'  (x  *  conj  y).re  (norm  y),  Int.mod'  (x  *  conj  y).im  (norm  y)⟩
  ·  ext  <;>  simp  [Int.mod'_eq,  mod_def,  div_def,  norm]  <;>  ring
  have  H2  :  norm  (x  %  y)  *  norm  y  ≤  norm  y  /  2  *  norm  y
  ·  calc
  norm  (x  %  y)  *  norm  y  =  norm  (x  %  y  *  conj  y)  :=  by  simp  only  [norm_mul,  norm_conj]
  _  =  |Int.mod'  (x.re  *  y.re  +  x.im  *  y.im)  (norm  y)|  ^  2
  +  |Int.mod'  (-(x.re  *  y.im)  +  x.im  *  y.re)  (norm  y)|  ^  2  :=  by  simp  [H1,  norm,  sq_abs]
  _  ≤  (y.norm  /  2)  ^  2  +  (y.norm  /  2)  ^  2  :=  by  gcongr  <;>  apply  Int.abs_mod'_le  _  _  norm_y_pos
  _  =  norm  y  /  2  *  (norm  y  /  2  *  2)  :=  by  ring
  _  ≤  norm  y  /  2  *  norm  y  :=  by  gcongr;  apply  Int.ediv_mul_le;  norm_num
  calc  norm  (x  %  y)  ≤  norm  y  /  2  :=  le_of_mul_le_mul_right  H2  norm_y_pos
  _  <  norm  y  :=  by
  apply  Int.ediv_lt_of_lt_mul
  ·  norm_num
  ·  linarith 
```

我们已经接近终点。我们的 `norm` 函数将高斯整数映射到非负整数。我们需要一个将高斯整数映射到自然数的函数，我们通过将 `norm` 与将整数映射到自然数的函数 `Int.natAbs` 组合来实现这一点。下一个两个引理中的第一个建立了将范数映射到自然数再映射回整数不会改变值的性质。第二个重新表述了范数是递减的事实。

```py
theorem  coe_natAbs_norm  (x  :  GaussInt)  :  (x.norm.natAbs  :  ℤ)  =  x.norm  :=
  Int.natAbs_of_nonneg  (norm_nonneg  _)

theorem  natAbs_norm_mod_lt  (x  y  :  GaussInt)  (hy  :  y  ≠  0)  :
  (x  %  y).norm.natAbs  <  y.norm.natAbs  :=  by
  apply  Int.ofNat_lt.1
  simp  only  [Int.natCast_natAbs,  abs_of_nonneg,  norm_nonneg]
  exact  norm_mod_lt  x  hy 
```

我们还需要在欧几里得域上建立范数函数的第二个关键性质。

```py
theorem  not_norm_mul_left_lt_norm  (x  :  GaussInt)  {y  :  GaussInt}  (hy  :  y  ≠  0)  :
  ¬(norm  (x  *  y)).natAbs  <  (norm  x).natAbs  :=  by
  apply  not_lt_of_ge
  rw  [norm_mul,  Int.natAbs_mul]
  apply  le_mul_of_one_le_right  (Nat.zero_le  _)
  apply  Int.ofNat_le.1
  rw  [coe_natAbs_norm]
  exact  Int.add_one_le_of_lt  ((norm_pos  _).mpr  hy) 
```

我们现在可以将这些放在一起来证明高斯整数是欧几里得域的一个实例。我们使用我们定义的商和余数函数。Mathlib 中欧几里得域的定义比上面的定义更通用，因为它允许我们证明余数与任何良基测度相关减少。比较返回自然数的范数函数的值只是这种测度的一个实例，在这种情况下，所需性质是定理`natAbs_norm_mod_lt`和`not_norm_mul_left_lt_norm`。

```py
instance  :  EuclideanDomain  GaussInt  :=
  {  GaussInt.instCommRing  with
  quotient  :=  (·  /  ·)
  remainder  :=  (·  %  ·)
  quotient_mul_add_remainder_eq  :=
  fun  x  y  ↦  by  rw  [mod_def,  add_comm]  ;  ring
  quotient_zero  :=  fun  x  ↦  by
  simp  [div_def,  norm,  Int.div']
  rfl
  r  :=  (measure  (Int.natAbs  ∘  norm)).1
  r_wellFounded  :=  (measure  (Int.natAbs  ∘  norm)).2
  remainder_lt  :=  natAbs_norm_mod_lt
  mul_left_not_lt  :=  not_norm_mul_left_lt_norm  } 
```

立即的回报是，我们现在知道，在高斯整数中，素数和不可约的概念是一致的。

```py
example  (x  :  GaussInt)  :  Irreducible  x  ↔  Prime  x  :=
  irreducible_iff_prime 
``` 上一节 下一节

* * *

© 版权所有 2020-2025，Jeremy Avigad，Patrick Massot。文本许可协议为 CC BY 4.0。

使用[Sphinx](https://www.sphinx-doc.org/)构建，使用[主题](https://github.com/readthedocs/sphinx_rtd_theme)由[Read the Docs](https://readthedocs.org)提供。现代数学在本质上是使用代数结构，这些结构封装了可以在多个设置中实例化的模式。该主题提供了定义此类结构及其构建特定实例的各种方法。

因此，Lean 提供了定义结构及其形式化操作的方法。您已经看到了 Lean 中代数结构的例子，例如环和格，这些在第二章中有所讨论。本章将解释您在那里看到的神秘方括号注释，`[Ring α]`和`[Lattice α]`。它还将向您展示如何定义和使用您自己的代数结构。

对于更详细的技术信息，您可以查阅[Lean 中的定理证明](https://leanprover.github.io/theorem_proving_in_lean/)，以及 Anne Baanen 的一篇论文，[在 Lean 数学库中使用和滥用实例参数](https://arxiv.org/abs/2202.01629)。

## 7.1\. 定义结构

在最广泛的意义上，一个*结构*是一组数据的指定，可能包含数据需要满足的约束。该结构的*实例*是满足这些约束的特定数据包。例如，我们可以指定一个点是三个实数的元组：

```py
@[ext]
structure  Point  where
  x  :  ℝ
  y  :  ℝ
  z  :  ℝ 
```

`@[ext]`注释告诉 Lean 自动生成定理，这些定理可以用来证明当结构实例的组件相等时，两个结构实例是相等的，这种性质称为*扩展性*。

```py
#check  Point.ext

example  (a  b  :  Point)  (hx  :  a.x  =  b.x)  (hy  :  a.y  =  b.y)  (hz  :  a.z  =  b.z)  :  a  =  b  :=  by
  ext
  repeat'  assumption 
```

然后，我们可以定义`Point`结构的特定实例。Lean 提供了多种方法来实现这一点。

```py
def  myPoint1  :  Point  where
  x  :=  2
  y  :=  -1
  z  :=  4

def  myPoint2  :  Point  :=
  ⟨2,  -1,  4⟩

def  myPoint3  :=
  Point.mk  2  (-1)  4 
```

在第一个例子中，结构的字段被明确命名。在 `myPoint3` 的定义中提到的 `Point.mk` 函数被称为 `Point` 结构的 *构造函数*，因为它用于构建元素。如果你想，可以指定不同的名称，比如 `build`。

```py
structure  Point'  where  build  ::
  x  :  ℝ
  y  :  ℝ
  z  :  ℝ

#check  Point'.build  2  (-1)  4 
```

下面的两个例子展示了如何在结构上定义函数。第二个例子使 `Point.mk` 构造函数显式化，而第一个例子为了简洁使用了匿名构造函数。Lean 可以从 `add` 的指示类型推断出相关的构造函数。将定义和定理与像 `Point` 这样的结构关联的命名空间与相同名称的命名空间放在一起是一种惯例。在下面的例子中，因为我们已经打开了 `Point` 命名空间，所以 `add` 的全名是 `Point.add`。当命名空间未打开时，我们必须使用全名。但请记住，使用匿名投影符号通常很方便，它允许我们写出 `a.add b` 而不是 `Point.add a b`。因为 `a` 的类型是 `Point`，所以 Lean 将前者解释为后者。

```py
namespace  Point

def  add  (a  b  :  Point)  :  Point  :=
  ⟨a.x  +  b.x,  a.y  +  b.y,  a.z  +  b.z⟩

def  add'  (a  b  :  Point)  :  Point  where
  x  :=  a.x  +  b.x
  y  :=  a.y  +  b.y
  z  :=  a.z  +  b.z

#check  add  myPoint1  myPoint2
#check  myPoint1.add  myPoint2

end  Point

#check  Point.add  myPoint1  myPoint2
#check  myPoint1.add  myPoint2 
```

下面我们将继续在相关命名空间中放置定义，但我们将省略引用片段中的命名空间命令。为了证明加法函数的性质，我们可以使用 `rw` 来展开定义，并使用 `ext` 将结构中两个元素之间的等式简化为组件之间的等式。我们使用 `protected` 关键字，即使命名空间是开放的，定理的名称也是 `Point.add_comm`。这在我们想要避免与像 `add_comm` 这样的通用定理产生歧义时很有帮助。

```py
protected  theorem  add_comm  (a  b  :  Point)  :  add  a  b  =  add  b  a  :=  by
  rw  [add,  add]
  ext  <;>  dsimp
  repeat'  apply  add_comm

example  (a  b  :  Point)  :  add  a  b  =  add  b  a  :=  by  simp  [add,  add_comm] 
```

因为 Lean 可以在内部展开定义并简化投影，有时我们想要的方程在定义上是成立的。

```py
theorem  add_x  (a  b  :  Point)  :  (a.add  b).x  =  a.x  +  b.x  :=
  rfl 
```

还可以使用模式匹配在结构上定义函数，这与我们在 第 5.2 节 中定义递归函数的方式类似。下面的 `addAlt` 和 `addAlt'` 定义基本上是相同的；唯一的区别是我们第二个使用了匿名构造函数符号。虽然以这种方式定义函数有时很方便，结构性的 eta-归约使这种替代定义上等价，但它可能会在后续的证明中使事情变得不那么方便。特别是，`rw [addAlt]` 会留下一个包含 `match` 语句的更混乱的目标视图。

```py
def  addAlt  :  Point  →  Point  →  Point
  |  Point.mk  x₁  y₁  z₁,  Point.mk  x₂  y₂  z₂  =>  ⟨x₁  +  x₂,  y₁  +  y₂,  z₁  +  z₂⟩

def  addAlt'  :  Point  →  Point  →  Point
  |  ⟨x₁,  y₁,  z₁⟩,  ⟨x₂,  y₂,  z₂⟩  =>  ⟨x₁  +  x₂,  y₁  +  y₂,  z₁  +  z₂⟩

theorem  addAlt_x  (a  b  :  Point)  :  (a.addAlt  b).x  =  a.x  +  b.x  :=  by
  rfl

theorem  addAlt_comm  (a  b  :  Point)  :  addAlt  a  b  =  addAlt  b  a  :=  by
  rw  [addAlt,  addAlt]
  -- the same proof still works, but the goal view here is harder to read
  ext  <;>  dsimp
  repeat'  apply  add_comm 
```

数学构造通常涉及拆分捆绑的信息，并以不同的方式重新组合。因此，Lean 和 Mathlib 提供了如此多的方法来高效地完成这项工作是有意义的。作为练习，尝试证明 `Point.add` 是结合的。然后定义点的标量乘法，并证明它对加法分配。

```py
protected  theorem  add_assoc  (a  b  c  :  Point)  :  (a.add  b).add  c  =  a.add  (b.add  c)  :=  by
  sorry

def  smul  (r  :  ℝ)  (a  :  Point)  :  Point  :=
  sorry

theorem  smul_distrib  (r  :  ℝ)  (a  b  :  Point)  :
  (smul  r  a).add  (smul  r  b)  =  smul  r  (a.add  b)  :=  by
  sorry 
```

使用结构只是通往代数抽象道路上的第一步。我们还没有一种方法将 `Point.add` 与泛型 `+` 符号链接起来，或者将 `Point.add_comm` 和 `Point.add_assoc` 与泛型 `add_comm` 和 `add_assoc` 公理连接起来。这些任务属于使用结构的 *代数* 方面，我们将在下一节中解释如何执行它们。现在，只需将结构视为捆绑对象和信息的方式即可。

特别有用的是，一个结构不仅可以指定数据类型，还可以指定数据必须满足的约束。在 Lean 中，后者表示为类型 `Prop` 的字段。例如，*标准 2-单纯形*定义为满足 $x ≥ 0$、$y ≥ 0$、$z ≥ 0$ 以及 $x + y + z = 1$ 的点集 $(x, y, z)$。如果你不熟悉这个概念，你应该画一个图，并说服自己这个集合是三维空间中的等边三角形，其顶点为 $(1, 0, 0)$、$(0, 1, 0)$ 和 $(0, 0, 1)$，以及其内部。我们可以在 Lean 中如下表示它：

```py
structure  StandardTwoSimplex  where
  x  :  ℝ
  y  :  ℝ
  z  :  ℝ
  x_nonneg  :  0  ≤  x
  y_nonneg  :  0  ≤  y
  z_nonneg  :  0  ≤  z
  sum_eq  :  x  +  y  +  z  =  1 
```

注意到最后四个字段指的是 `x`、`y` 和 `z`，即前三个字段。我们可以定义一个从二单纯形到自身的映射，该映射交换 `x` 和 `y`：

```py
def  swapXy  (a  :  StandardTwoSimplex)  :  StandardTwoSimplex
  where
  x  :=  a.y
  y  :=  a.x
  z  :=  a.z
  x_nonneg  :=  a.y_nonneg
  y_nonneg  :=  a.x_nonneg
  z_nonneg  :=  a.z_nonneg
  sum_eq  :=  by  rw  [add_comm  a.y  a.x,  a.sum_eq] 
```

更有趣的是，我们可以在单纯形上计算两点之间的中点。我们在文件开头添加了“不可计算部分”这个短语，以便在实数上使用除法。

```py
noncomputable  section

def  midpoint  (a  b  :  StandardTwoSimplex)  :  StandardTwoSimplex
  where
  x  :=  (a.x  +  b.x)  /  2
  y  :=  (a.y  +  b.y)  /  2
  z  :=  (a.z  +  b.z)  /  2
  x_nonneg  :=  div_nonneg  (add_nonneg  a.x_nonneg  b.x_nonneg)  (by  norm_num)
  y_nonneg  :=  div_nonneg  (add_nonneg  a.y_nonneg  b.y_nonneg)  (by  norm_num)
  z_nonneg  :=  div_nonneg  (add_nonneg  a.z_nonneg  b.z_nonneg)  (by  norm_num)
  sum_eq  :=  by  field_simp;  linarith  [a.sum_eq,  b.sum_eq] 
```

在这里，我们已经用简洁的证明项建立了 `x_nonneg`、`y_nonneg` 和 `z_nonneg`，但在战术模式中使用 `by` 建立了 `sum_eq`。

给定一个满足 $0 \le \lambda \le 1$ 的参数 $\lambda$，我们可以在标准 2-单纯形中取两个点 $a$ 和 $b$ 的加权平均 $\lambda a + (1 - \lambda) b$。我们挑战你定义这个函数，类似于上面的 `midpoint` 函数。

```py
def  weightedAverage  (lambda  :  Real)  (lambda_nonneg  :  0  ≤  lambda)  (lambda_le  :  lambda  ≤  1)
  (a  b  :  StandardTwoSimplex)  :  StandardTwoSimplex  :=
  sorry 
```

结构可以依赖于参数。例如，我们可以将标准 2-单纯形推广到标准 $n$-单纯形，对于任何 $n$ 都可以这样做。在这个阶段，你不必了解类型 `Fin n` 的任何东西，除了它有 $n$ 个元素，以及 Lean 知道如何对其求和。

```py
open  BigOperators

structure  StandardSimplex  (n  :  ℕ)  where
  V  :  Fin  n  →  ℝ
  NonNeg  :  ∀  i  :  Fin  n,  0  ≤  V  i
  sum_eq_one  :  (∑  i,  V  i)  =  1

namespace  StandardSimplex

def  midpoint  (n  :  ℕ)  (a  b  :  StandardSimplex  n)  :  StandardSimplex  n
  where
  V  i  :=  (a.V  i  +  b.V  i)  /  2
  NonNeg  :=  by
  intro  i
  apply  div_nonneg
  ·  linarith  [a.NonNeg  i,  b.NonNeg  i]
  norm_num
  sum_eq_one  :=  by
  simp  [div_eq_mul_inv,  ←  Finset.sum_mul,  Finset.sum_add_distrib,
  a.sum_eq_one,  b.sum_eq_one]
  field_simp

end  StandardSimplex 
```

作为练习，看看你是否可以定义标准 $n$-单纯形中两个点的加权平均。你可以使用 `Finset.sum_add_distrib` 和 `Finset.mul_sum` 来操作相关的和。

我们已经看到，结构可以用来捆绑数据和属性。有趣的是，它们也可以用来捆绑属性而不包含数据。例如，下一个结构 `IsLinear` 捆绑了线性性的两个组成部分。

```py
structure  IsLinear  (f  :  ℝ  →  ℝ)  where
  is_additive  :  ∀  x  y,  f  (x  +  y)  =  f  x  +  f  y
  preserves_mul  :  ∀  x  c,  f  (c  *  x)  =  c  *  f  x

section
variable  (f  :  ℝ  →  ℝ)  (linf  :  IsLinear  f)

#check  linf.is_additive
#check  linf.preserves_mul

end 
```

值得指出的是，结构并不是捆绑数据的唯一方式。可以使用泛型类型积定义 `Point` 数据结构，并且可以使用简单的 `and` 定义 `IsLinear`。

```py
def  Point''  :=
  ℝ  ×  ℝ  ×  ℝ

def  IsLinear'  (f  :  ℝ  →  ℝ)  :=
  (∀  x  y,  f  (x  +  y)  =  f  x  +  f  y)  ∧  ∀  x  c,  f  (c  *  x)  =  c  *  f  x 
```

泛型类型构造甚至可以用作具有组件之间依赖关系的结构。例如，*子类型*构造将数据与属性结合起来。你可以将下一个示例中的类型`PReal`视为正实数的类型。任何`x : PReal`都有两个组件：值和正属性。你可以通过`x.val`访问这些组件，它具有类型`ℝ`，以及`x.property`，它表示`0 < x.val`这一事实。

```py
def  PReal  :=
  {  y  :  ℝ  //  0  <  y  }

section
variable  (x  :  PReal)

#check  x.val
#check  x.property
#check  x.1
#check  x.2

end 
```

我们可以使用子类型来定义标准的 2-单纯形，以及任意$n$的标准$n$-单纯形。

```py
def  StandardTwoSimplex'  :=
  {  p  :  ℝ  ×  ℝ  ×  ℝ  //  0  ≤  p.1  ∧  0  ≤  p.2.1  ∧  0  ≤  p.2.2  ∧  p.1  +  p.2.1  +  p.2.2  =  1  }

def  StandardSimplex'  (n  :  ℕ)  :=
  {  v  :  Fin  n  →  ℝ  //  (∀  i  :  Fin  n,  0  ≤  v  i)  ∧  (∑  i,  v  i)  =  1  } 
```

类似地，*Sigma 类型*是有序对的推广，其中第二个组件的类型取决于第一个组件的类型。

```py
def  StdSimplex  :=  Σ  n  :  ℕ,  StandardSimplex  n

section
variable  (s  :  StdSimplex)

#check  s.fst
#check  s.snd

#check  s.1
#check  s.2

end 
```

给定`s : StdSimplex`，第一个组件`s.fst`是一个自然数，第二个组件是对应单纯形`StandardSimplex s.fst`的一个元素。Sigma 类型与子类型之间的区别在于，Sigma 类型的第二个组件是数据而不是命题。

尽管我们可以使用积、子类型和 Sigma 类型来代替结构，但使用结构有许多优点。定义结构抽象出底层表示，并为访问组件的函数提供自定义名称。这使得证明更加健壮：仅依赖于结构接口的证明，在定义更改时通常仍然有效，只要我们用新定义重新定义旧的访问器。此外，正如我们即将看到的，Lean 提供了将结构编织成丰富、相互关联的层次结构以及管理它们之间交互的支持。  

为了阐明我们所说的*代数结构*的含义，考虑一些例子会有所帮助。

1.  一个*偏序集*由一个集合$P$和一个在$P$上的二元关系$\le$组成，该关系是传递的和自反的。

1.  一个*群*由一个具有结合二元运算、单位元素 $1$ 和一个函数 $g \mapsto g^{-1}$ 的集合 $G$ 组成，该函数为 $G$ 中的每个 $g$ 返回一个逆元素。如果运算是对称的，则群是*阿贝尔的*或*交换的*。

1.  一个*格*是一个具有交集和并集的偏序集。

1.  一个*环*由一个（加法表示的）阿贝尔群 $(R, +, 0, x \mapsto -x)$ 组成，以及一个结合乘法运算 $\cdot$ 和一个单位 $1$，使得乘法对加法分配。如果乘法是交换的，则环是*交换的*。

1.  一个*有序环* $(R, +, 0, -, \cdot, 1, \le)$ 由一个环及其元素上的偏序组成，使得对于$R$中的每个$a$、$b$和$c$，$a \le b$意味着$a + c \le b + c$，并且$0 \le a$和$0 \le b$意味着对于$R$中的每个$a$和$b$，$0 \le a b$。

1.  一个 *度量空间* 由一个集合 $X$ 和一个函数 $d : X \times X \to \mathbb{R}$ 组成，使得以下条件成立：

    +   对于 $X$ 中的每一个 $x$ 和 $y$，有 $d(x, y) \ge 0$。

    +   $d(x, y) = 0$ 当且仅当 $x = y$。

    +   对于 $X$ 中的每一个 $x$ 和 $y$，有 $d(x, y) = d(y, x)$。

    +   对于 $X$ 中的每一个 $x$、$y$ 和 $z$，有 $d(x, z) \le d(x, y) + d(y, z)$。

1.  一个 *拓扑空间* 由一个集合 $X$ 和一个 $X$ 的子集的集合 $\mathcal T$ 组成，称为 $X$ 的 *开子集*，使得以下条件成立：

    +   空集和 $X$ 是开集。

    +   两个开集的交集是开集。

    +   开集的任意并集是开集。

在这些例子中，结构的元素属于一个集合，称为 *载体集*，有时它代表整个结构。例如，当我们说“设 $G$ 为一个群”然后“设 $g \in G$”时，我们使用 $G$ 来代表结构和它的载体。并非每个代数结构都以这种方式与单个载体集相关联。例如，一个 *二部图* 涉及两个集合之间的关系，正如 *高斯连接* 一样，一个 *范畴* 也涉及两个感兴趣的集合，通常称为 *对象* 和 *态射*。

这些例子表明，证明辅助工具为了支持代数推理必须完成的一些事情。首先，它需要识别结构的具体实例。数系 $\mathbb{Z}$、$\mathbb{Q}$ 和 $\mathbb{R}$ 都是序环，我们应该能够在这些实例中的任何一个上应用关于序环的通用定理。有时一个具体集合可能以多种方式成为某个结构的一个实例。例如，除了 $\mathbb{R}$ 上的通常拓扑，它是实分析的基础之外，我们还可以考虑 $\mathbb{R}$ 上的 *离散* 拓扑，其中每个集合都是开集。

其次，证明辅助工具需要支持结构上的通用符号。在 Lean 中，符号 `*` 用于所有常规数系中的乘法，以及用于通用群和环中的乘法。当我们使用像 `f x * y` 这样的表达式时，Lean 必须使用关于 `f`、`x` 和 `y` 的类型信息来确定我们心中的乘法是哪一种。

第三，它需要处理结构可以从其他结构以各种方式继承定义、定理和记号的事实。一些结构通过添加更多的公理来扩展其他结构。交换环仍然是一个环，所以任何在环中有意义的定义在交换环中也有意义，任何在环中成立的定理在交换环中也成立。一些结构通过添加更多的数据来扩展其他结构。例如，任何环的加法部分是一个加法群。环结构添加了乘法和单位元，以及规范这些乘法和单位元以及它们与加法部分关系的公理。有时我们可以用另一个结构来定义一个结构。任何度量空间都有一个与之相关的典型拓扑，即*度量空间拓扑*，还有可以与任何线性顺序相关联的各种拓扑。

最后，重要的是要记住，数学允许我们使用函数和运算以定义结构的方式，就像我们使用函数和运算来定义数字一样。群的乘积和幂仍然是群。对于每一个$n$，模$n$的整数构成一个环，对于每一个$k > 0$，系数在该环中的$k \times k$多项式矩阵再次构成一个环。因此，我们可以像计算它们的元素一样容易地计算结构。这意味着代数结构在数学中过着双重生活，既是对象集合的容器，也是它们自身的对象。证明辅助工具必须适应这种双重角色。

当处理与具有代数结构相关联的类型元素时，证明辅助工具需要识别该结构并找到相关的定义、定理和记号。所有这些听起来像是一大堆工作，确实如此。但 Lean 使用一小套基本机制来完成这些任务。本节的目标是解释这些机制并展示如何使用它们。

第一个要素几乎是显而易见的：从严格意义上讲，代数结构是第 7.1 节意义上的结构。代数结构是一组满足某些公理假设的数据的指定，我们在第 7.1 节中看到，这正是`structure`命令设计用来适应的。这是一场天作之合！

给定一个数据类型`α`，我们可以如下定义`α`上的群结构。

```py
structure  Group₁  (α  :  Type*)  where
  mul  :  α  →  α  →  α
  one  :  α
  inv  :  α  →  α
  mul_assoc  :  ∀  x  y  z  :  α,  mul  (mul  x  y)  z  =  mul  x  (mul  y  z)
  mul_one  :  ∀  x  :  α,  mul  x  one  =  x
  one_mul  :  ∀  x  :  α,  mul  one  x  =  x
  inv_mul_cancel  :  ∀  x  :  α,  mul  (inv  x)  x  =  one 
```

注意，类型`α`是`Group₁`定义中的一个*参数*。因此，你应该将对象`struc : Group₁ α`视为`α`上的群结构。我们在第 2.2 节中看到，`inv_mul_cancel`的对应项`mul_inv_cancel`可以从其他群公理中得出，因此没有必要将其添加到定义中。

这个群的定义与 Mathlib 中 `Group` 的定义相似，我们选择 `Group₁` 这个名字来区分我们的版本。如果你写 `#check Group` 并在定义上 ctrl-click，你会看到 Mathlib 版本的 `Group` 被定义为扩展另一个结构；我们将在稍后解释如何做到这一点。如果你输入 `#print Group`，你也会看到 Mathlib 版本的 `Group` 有许多额外的字段。由于我们将解释的原因，有时在结构中添加冗余信息是有用的，这样就可以为从核心数据定义的对象和函数添加额外的字段。现在不用担心这个问题。请放心，我们的简化版本 `Group₁` 在道德上是与 Mathlib 使用的群定义相同的。

有时将类型与结构捆绑在一起是有用的，Mathlib 也包含了一个与以下等价的 `Grp` 结构定义：

```py
structure  Grp₁  where
  α  :  Type*
  str  :  Group₁  α 
```

Mathlib 版本可以在 `Mathlib.Algebra.Category.Grp.Basic` 中找到，如果你在示例文件的开始处添加这个导入，你可以使用 `#check` 来检查它。

由于以下原因将变得更为清晰，通常更有用的是将类型 `α` 与结构 `Group α` 分开。我们将这两个对象一起称为 *部分捆绑结构*，因为表示结合了大多数但不是所有组件到一个结构中。在 Mathlib 中，当用作群的载体类型时，通常使用大写罗马字母如 `G` 来表示类型。

让我们构造一个群，也就是说，`Group₁` 类型的元素。对于任何一对类型 `α` 和 `β`，Mathlib 定义了 `α` 和 `β` 之间 *等价* 的类型 `Equiv α β`。Mathlib 还为这个类型定义了有启发性的符号 `α ≃ β`。一个元素 `f : α ≃ β` 是 `α` 和 `β` 之间的双射，由四个组件表示：一个从 `α` 到 `β` 的函数 `f.toFun`，一个从 `β` 到 `α` 的逆函数 `f.invFun`，以及两个指定这些函数确实是彼此的逆的性质。

```py
variable  (α  β  γ  :  Type*)
variable  (f  :  α  ≃  β)  (g  :  β  ≃  γ)

#check  Equiv  α  β
#check  (f.toFun  :  α  →  β)
#check  (f.invFun  :  β  →  α)
#check  (f.right_inv  :  ∀  x  :  β,  f  (f.invFun  x)  =  x)
#check  (f.left_inv  :  ∀  x  :  α,  f.invFun  (f  x)  =  x)
#check  (Equiv.refl  α  :  α  ≃  α)
#check  (f.symm  :  β  ≃  α)
#check  (f.trans  g  :  α  ≃  γ) 
```

注意最后三个构造的命名非常具有创意。我们将恒等函数 `Equiv.refl`、逆运算 `Equiv.symm` 和复合运算 `Equiv.trans` 视为明确的证据，表明双射对应性是一个等价关系。

注意，`f.trans g` 需要以前向函数的逆序进行组合。Mathlib 声明了一个从 `Equiv α β` 到函数类型 `α → β` 的 *强制转换*，因此我们可以省略写 `.toFun`，让 Lean 为我们插入它。

```py
example  (x  :  α)  :  (f.trans  g).toFun  x  =  g.toFun  (f.toFun  x)  :=
  rfl

example  (x  :  α)  :  (f.trans  g)  x  =  g  (f  x)  :=
  rfl

example  :  (f.trans  g  :  α  →  γ)  =  g  ∘  f  :=
  rfl 
```

Mathlib 还定义了类型 `perm α`，它是 `α` 与自身之间的等价类型。

```py
example  (α  :  Type*)  :  Equiv.Perm  α  =  (α  ≃  α)  :=
  rfl 
```

应该清楚，`Equiv.Perm α` 在等价关系的组合下形成一个群。我们这样安排，使得 `mul f g` 等于 `g.trans f`，其前向函数是 `f ∘ g`。换句话说，乘法就是我们通常认为的双射的复合。在这里，我们定义了这个群：

```py
def  permGroup  {α  :  Type*}  :  Group₁  (Equiv.Perm  α)
  where
  mul  f  g  :=  Equiv.trans  g  f
  one  :=  Equiv.refl  α
  inv  :=  Equiv.symm
  mul_assoc  f  g  h  :=  (Equiv.trans_assoc  _  _  _).symm
  one_mul  :=  Equiv.trans_refl
  mul_one  :=  Equiv.refl_trans
  inv_mul_cancel  :=  Equiv.self_trans_symm 
```

事实上，Mathlib 在 `Algebra.Group.End` 文件中在 `Equiv.Perm α` 上定义了确切的这个 `Group` 结构。一如既往，你可以悬停在 `permGroup` 定义中使用的定理上以查看它们的陈述，你也可以跳转到原始文件中的定义以了解更多关于它们是如何实现的。

在常规数学中，我们通常认为符号与结构是独立的。例如，我们可以考虑群 $(G_1, \cdot, 1, \cdot^{-1})$、$(G_2, \circ, e, i(\cdot))$ 和 $(G_3, +, 0, -)$。在第一种情况下，我们用 $\cdot$ 表示二元运算，用 $1$ 表示单位元，用 $x \mapsto x^{-1}$ 表示逆函数。在第二种和第三种情况下，我们使用显示的符号替代。然而，当我们用 Lean 正式化群的观念时，符号与结构的关系更为紧密。在 Lean 中，任何 `Group` 的组成部分被命名为 `mul`、`one` 和 `inv`，我们很快就会看到如何设置乘法符号来引用它们。如果我们想使用加法符号，我们则使用同构结构 `AddGroup`（加法群的底层结构）。它的组成部分被命名为 `add`、`zero` 和 `neg`，相关的符号正如你所期望的那样。

回想一下我们在 第 7.1 节 中定义的 `Point` 类型以及我们那里定义的加法函数。这些定义在伴随本节的示例文件中重现。作为一个练习，定义一个类似于我们上面定义的 `Group₁` 结构的 `AddGroup₁` 结构，但它使用前面描述的加法命名方案。在 `Point` 数据类型上定义否定和零，并在 `Point` 上定义 `AddGroup₁` 结构。

```py
structure  AddGroup₁  (α  :  Type*)  where
  (add  :  α  →  α  →  α)
  -- fill in the rest
@[ext]
structure  Point  where
  x  :  ℝ
  y  :  ℝ
  z  :  ℝ

namespace  Point

def  add  (a  b  :  Point)  :  Point  :=
  ⟨a.x  +  b.x,  a.y  +  b.y,  a.z  +  b.z⟩

def  neg  (a  :  Point)  :  Point  :=  sorry

def  zero  :  Point  :=  sorry

def  addGroupPoint  :  AddGroup₁  Point  :=  sorry

end  Point 
```

我们正在取得进展。现在我们知道了如何在 Lean 中定义代数结构，也知道如何定义这些结构的实例。但我们还希望将符号与结构关联起来，以便我们可以与每个实例一起使用它。此外，我们希望将其组织得可以定义结构上的一个运算并使用任何特定实例，我们还想组织得可以证明关于结构的一个定理并使用任何实例。

事实上，Mathlib 已经设置好使用通用的群符号、定义和定理为 `Equiv.Perm α`。

```py
variable  {α  :  Type*}  (f  g  :  Equiv.Perm  α)  (n  :  ℕ)

#check  f  *  g
#check  mul_assoc  f  g  g⁻¹

-- group power, defined for any group
#check  g  ^  n

example  :  f  *  g  *  g⁻¹  =  f  :=  by  rw  [mul_assoc,  mul_inv_cancel,  mul_one]

example  :  f  *  g  *  g⁻¹  =  f  :=
  mul_inv_cancel_right  f  g

example  {α  :  Type*}  (f  g  :  Equiv.Perm  α)  :  g.symm.trans  (g.trans  f)  =  f  :=
  mul_inv_cancel_right  f  g 
```

你可以检查我们上面要求你定义的 `Point` 上的加法群结构并非如此。我们的任务现在是要理解在 `Equiv.Perm α` 的例子背后发生的魔法，以便它们能以这种方式工作。

问题在于 Lean 需要能够*找到*相关的符号和隐含的群结构，使用我们在输入的表达式中找到的信息。同样，当我们用具有类型`ℝ`的表达式`x`和`y`来写`x + y`时，Lean 需要将`+`符号解释为实数上的相关加法函数。它还必须识别类型`ℝ`作为交换环的一个实例，这样所有交换环的定义和定理都可用。例如，连续性在 Lean 中相对于任何两个拓扑空间定义。当我们有`f : ℝ → ℂ`并写`Continuous f`时，Lean 必须找到`ℝ`和`ℂ`上的相关拓扑。

这个魔法是通过三个事物的结合实现的。

1.  *逻辑*。任何群中应被解释的定义，其参数是群类型和群结构。同样，关于任意群元素的定理以对群类型和群结构的全称量词开始。

1.  *隐含参数*。类型和结构的参数通常被省略，这样我们就不必写它们或在 Lean 信息窗口中看到它们。Lean 会默默地为我们填写信息。

1.  *类型类推断*。也称为*类推断*，这是一个简单但强大的机制，使我们能够为 Lean 注册信息以便以后使用。当 Lean 被调用以填充定义、定理或符号中的隐含参数时，它可以利用已注册的信息。

而一个注解`(grp : Group G)`告诉 Lean 它应该期望显式地给出该参数，注解`{grp : Group G}`告诉 Lean 它应该尝试从表达式的上下文中推断出来，注解`[grp : Group G]`告诉 Lean 相应的参数应该通过类型类推断来合成。由于使用此类参数的整个目的通常是我们不需要显式地引用它们，Lean 允许我们写`[Group G]`并使名称匿名。你可能已经注意到 Lean 会自动选择像`_inst_1`这样的名称。当我们使用匿名方括号注解与`variables`命令一起使用时，只要变量仍然在作用域内，Lean 就会自动将参数`[Group G]`添加到任何提及`G`的定义或定理中。

我们如何注册 Lean 执行搜索所需的信息？回到我们的群组例子，我们只需要做两个更改。首先，我们不是使用`structure`命令来定义群结构，而是使用关键字`class`来表示它是类推断的候选。其次，我们不是使用`def`定义特定的实例，而是使用关键字`instance`将特定的实例注册到 Lean 中。与类变量名一样，我们可以省略实例定义的名称，因为在一般情况下，我们希望 Lean 找到它并投入使用，而不必麻烦我们处理细节。

```py
class  Group₂  (α  :  Type*)  where
  mul  :  α  →  α  →  α
  one  :  α
  inv  :  α  →  α
  mul_assoc  :  ∀  x  y  z  :  α,  mul  (mul  x  y)  z  =  mul  x  (mul  y  z)
  mul_one  :  ∀  x  :  α,  mul  x  one  =  x
  one_mul  :  ∀  x  :  α,  mul  one  x  =  x
  inv_mul_cancel  :  ∀  x  :  α,  mul  (inv  x)  x  =  one

instance  {α  :  Type*}  :  Group₂  (Equiv.Perm  α)  where
  mul  f  g  :=  Equiv.trans  g  f
  one  :=  Equiv.refl  α
  inv  :=  Equiv.symm
  mul_assoc  f  g  h  :=  (Equiv.trans_assoc  _  _  _).symm
  one_mul  :=  Equiv.trans_refl
  mul_one  :=  Equiv.refl_trans
  inv_mul_cancel  :=  Equiv.self_trans_symm 
```

以下说明了它们的使用。

```py
#check  Group₂.mul

def  mySquare  {α  :  Type*}  [Group₂  α]  (x  :  α)  :=
  Group₂.mul  x  x

#check  mySquare

section
variable  {β  :  Type*}  (f  g  :  Equiv.Perm  β)

example  :  Group₂.mul  f  g  =  g.trans  f  :=
  rfl

example  :  mySquare  f  =  f.trans  f  :=
  rfl

end 
```

`#check`命令显示`Group₂.mul`有一个隐含的参数`[Group₂ α]`，我们期望通过类推断找到它，其中`α`是`Group₂.mul`参数的类型。换句话说，`{α : Type*}`是群元素类型的隐含参数，`[Group₂ α]`是`α`上的群结构的隐含参数。同样，当我们为`Group₂`定义一个泛型平方函数`my_square`时，我们使用一个隐含参数`{α : Type*}`作为元素类型，以及一个隐含参数`[Group₂ α]`作为`Group₂`结构。

在第一个例子中，当我们写`Group₂.mul f g`时，`f`和`g`的类型告诉 Lean，在`Group₂.mul`的参数`α`中必须实例化为`Equiv.Perm β`。这意味着 Lean 必须找到一个`Group₂ (Equiv.Perm β)`的元素。之前的`instance`声明告诉 Lean 如何做到这一点。问题解决！

这种简单的信息注册机制，使得 Lean 在需要时能够找到它，非常有用。这里有一种它出现的方式。在 Lean 的基础中，数据类型`α`可能是空的。然而，在许多应用中，知道一个类型至少有一个元素是有用的。例如，函数`List.headI`，它返回列表的第一个元素，当列表为空时可以返回默认值。为了实现这一点，Lean 库定义了一个类`Inhabited α`，它所做的只是存储一个默认值。我们可以证明`Point`类型是一个实例：

```py
instance  :  Inhabited  Point  where  default  :=  ⟨0,  0,  0⟩

#check  (default  :  Point)

example  :  ([]  :  List  Point).headI  =  default  :=
  rfl 
```

类推断机制也用于泛型表示。表达式`x + y`是`Add.add x y`的缩写，其中——你可能已经猜到了——`Add α`是一个存储在`α`上的二元函数的类。写`x + y`告诉 Lean 找到已注册的`[Add.add α]`实例，并使用相应的函数。下面，我们注册了`Point`的加法函数。

```py
instance  :  Add  Point  where  add  :=  Point.add

section
variable  (x  y  :  Point)

#check  x  +  y

example  :  x  +  y  =  Point.add  x  y  :=
  rfl

end 
```

以这种方式，我们还可以将符号`+`分配给其他类型上的二元运算。

但我们可以做得更好。我们已经看到 `*` 可以在任何群中使用，`+` 可以在任何加法群中使用，并且两者都可以在任何环中使用。当我们定义 Lean 中的新环实例时，我们不需要为该实例定义 `+` 和 `*`，因为 Lean 知道这些在每一个环中都是定义好的。我们可以使用这种方法来指定 `Group₂` 类的符号：

```py
instance  {α  :  Type*}  [Group₂  α]  :  Mul  α  :=
  ⟨Group₂.mul⟩

instance  {α  :  Type*}  [Group₂  α]  :  One  α  :=
  ⟨Group₂.one⟩

instance  {α  :  Type*}  [Group₂  α]  :  Inv  α  :=
  ⟨Group₂.inv⟩

section
variable  {α  :  Type*}  (f  g  :  Equiv.Perm  α)

#check  f  *  1  *  g⁻¹

def  foo  :  f  *  1  *  g⁻¹  =  g.symm.trans  ((Equiv.refl  α).trans  f)  :=
  rfl

end 
```

这种方法之所以有效，是因为 Lean 进行了递归搜索。根据我们声明的实例，Lean 可以通过找到一个 `Group₂ (Equiv.Perm α)` 的实例来找到一个 `Mul (Equiv.Perm α)` 的实例，并且它可以通过我们提供的一个实例来找到一个 `Group₂ (Equiv.Perm α)` 的实例。Lean 能够找到这两个事实并将它们连接起来。

我们刚才给出的例子是危险的，因为 Lean 的库中也有一个 `Group (Equiv.Perm α)` 的实例，并且乘法在任意群上都是定义好的。所以，哪个实例被找到是不确定的。实际上，Lean 优先考虑较新的声明，除非你明确指定了不同的优先级。此外，还有另一种方法告诉 Lean 一个结构是另一个结构的实例，即使用 `extends` 关键字。这就是 Mathlib 如何指定，例如，每个交换环都是一个环。你可以在第八部分和*《Lean 中的定理证明》*中的一个关于类推断的[章节](https://leanprover.github.io/theorem_proving_in_lean4/type_classes.html#managing-type-class-inference)中找到更多信息。

通常，为已经定义了符号的代数结构实例指定 `*` 的值是一个坏主意。在 Lean 中重新定义 `Group` 的概念是一个人为的例子。然而，在这种情况下，群符号的两种解释都展开为 `Equiv.trans`、`Equiv.refl` 和 `Equiv.symm`，方式相同。

作为类似的人为练习，定义一个类似于 `Group₂` 的类 `AddGroup₂`。使用类 `Add`、`Neg` 和 `Zero` 在任何 `AddGroup₂` 上定义加法、负数和零的常规符号。然后证明 `Point` 是 `AddGroup₂` 的一个实例。尝试一下，确保加法群符号对 `Point` 的元素有效。

```py
class  AddGroup₂  (α  :  Type*)  where
  add  :  α  →  α  →  α
  -- fill in the rest 
```

我们已经在上面的 `Point` 中声明了实例 `Add`、`Neg` 和 `Zero` 并不是什么大问题。再次强调，两种合成符号的方法应该得出相同的答案。

类推断是微妙的，使用时必须小心，因为它配置了自动化的设置，这些设置无形中控制着我们输入的表达式的解释。然而，如果使用得当，类推断是一个强大的工具。它是使 Lean 中的代数推理成为可能的原因。## 7.3\. 构建高斯整数

我们现在将通过构建一个重要的数学对象，即高斯整数，并展示它是一个欧几里得域，来展示在 Lean 中使用代数层次结构的用法。换句话说，根据我们一直在使用的术语，我们将定义高斯整数并展示它们是欧几里得域结构的实例。

在普通数学术语中，高斯整数集 $\Bbb{Z}[i]$ 是复数集 $\{ a + b i \mid a, b \in \Bbb{Z}\}$。但与其将它们定义为复数的子集，我们的目标在这里是将它们定义为它们自己的数据类型。我们通过将高斯整数表示为整数对来实现这一点，我们将这些整数视为*实部*和*虚部*。

```py
@[ext]
structure  GaussInt  where
  re  :  ℤ
  im  :  ℤ 
```

我们首先展示高斯整数具有环的结构，其中`0`被定义为`⟨0, 0⟩`，`1`被定义为`⟨1, 0⟩`，加法被定义为逐点加法。为了确定乘法定义，记住我们希望元素 $i$，由`⟨0, 1⟩`表示，是 $-1$ 的一个平方根。因此我们希望

$$\begin{split}(a + bi) (c + di) & = ac + bci + adi + bd i² \\ & = (ac - bd) + (bc + ad)i.\end{split}$$

这解释了下面`Mul`的定义。

```py
instance  :  Zero  GaussInt  :=
  ⟨⟨0,  0⟩⟩

instance  :  One  GaussInt  :=
  ⟨⟨1,  0⟩⟩

instance  :  Add  GaussInt  :=
  ⟨fun  x  y  ↦  ⟨x.re  +  y.re,  x.im  +  y.im⟩⟩

instance  :  Neg  GaussInt  :=
  ⟨fun  x  ↦  ⟨-x.re,  -x.im⟩⟩

instance  :  Mul  GaussInt  :=
  ⟨fun  x  y  ↦  ⟨x.re  *  y.re  -  x.im  *  y.im,  x.re  *  y.im  +  x.im  *  y.re⟩⟩ 
```

如 第 7.1 节 所述，将所有与数据类型相关的定义放在具有相同名称的命名空间中是一个好主意。因此，在本章相关的 Lean 文件中，这些定义是在`GaussInt`命名空间中进行的。

注意，在这里我们直接定义了符号 `0`、`1`、`+`、`-` 和 `*` 的解释，而不是将它们命名为 `GaussInt.zero` 等等，并将符号分配给它们。对于与 `simp` 和 `rw` 一起使用，有一个明确的名称对于定义来说通常是有用的。

```py
theorem  zero_def  :  (0  :  GaussInt)  =  ⟨0,  0⟩  :=
  rfl

theorem  one_def  :  (1  :  GaussInt)  =  ⟨1,  0⟩  :=
  rfl

theorem  add_def  (x  y  :  GaussInt)  :  x  +  y  =  ⟨x.re  +  y.re,  x.im  +  y.im⟩  :=
  rfl

theorem  neg_def  (x  :  GaussInt)  :  -x  =  ⟨-x.re,  -x.im⟩  :=
  rfl

theorem  mul_def  (x  y  :  GaussInt)  :
  x  *  y  =  ⟨x.re  *  y.re  -  x.im  *  y.im,  x.re  *  y.im  +  x.im  *  y.re⟩  :=
  rfl 
```

也有必要命名计算实部和虚部的规则，并将它们声明给简化器。

```py
@[simp]
theorem  zero_re  :  (0  :  GaussInt).re  =  0  :=
  rfl

@[simp]
theorem  zero_im  :  (0  :  GaussInt).im  =  0  :=
  rfl

@[simp]
theorem  one_re  :  (1  :  GaussInt).re  =  1  :=
  rfl

@[simp]
theorem  one_im  :  (1  :  GaussInt).im  =  0  :=
  rfl

@[simp]
theorem  add_re  (x  y  :  GaussInt)  :  (x  +  y).re  =  x.re  +  y.re  :=
  rfl

@[simp]
theorem  add_im  (x  y  :  GaussInt)  :  (x  +  y).im  =  x.im  +  y.im  :=
  rfl

@[simp]
theorem  neg_re  (x  :  GaussInt)  :  (-x).re  =  -x.re  :=
  rfl

@[simp]
theorem  neg_im  (x  :  GaussInt)  :  (-x).im  =  -x.im  :=
  rfl

@[simp]
theorem  mul_re  (x  y  :  GaussInt)  :  (x  *  y).re  =  x.re  *  y.re  -  x.im  *  y.im  :=
  rfl

@[simp]
theorem  mul_im  (x  y  :  GaussInt)  :  (x  *  y).im  =  x.re  *  y.im  +  x.im  *  y.re  :=
  rfl 
```

现在证明高斯整数是交换环的实例变得出奇地简单。我们正在充分利用结构概念。每个特定的高斯整数是`GaussInt`结构的实例，而类型`GaussInt`本身，连同相关操作，是`CommRing`结构的实例。`CommRing`结构反过来又扩展了`Zero`、`One`、`Add`、`Neg`和`Mul`的符号结构。

如果你输入 `instance : CommRing GaussInt := _`，然后在 VS Code 中点击出现的灯泡，然后让 Lean 填写结构定义的骨架，你会看到很多条目。然而，跳转到结构的定义，你会发现许多字段都有默认定义，Lean 会自动为你填写。下面定义的是基本的部分。一个特殊情况是 `nsmul` 和 `zsmul`，现在可以忽略它们，将在下一章中解释。在每种情况下，相关恒等式都是通过展开定义、使用 `ext` 策略将恒等式简化到其实部和虚部、简化，并在必要时在整数中进行相关环计算来证明的。请注意，我们可以轻松避免重复所有这些代码，但这不是当前讨论的主题。

```py
instance  instCommRing  :  CommRing  GaussInt  where
  zero  :=  0
  one  :=  1
  add  :=  (·  +  ·)
  neg  x  :=  -x
  mul  :=  (·  *  ·)
  nsmul  :=  nsmulRec
  zsmul  :=  zsmulRec
  add_assoc  :=  by
  intros
  ext  <;>  simp  <;>  ring
  zero_add  :=  by
  intro
  ext  <;>  simp
  add_zero  :=  by
  intro
  ext  <;>  simp
  neg_add_cancel  :=  by
  intro
  ext  <;>  simp
  add_comm  :=  by
  intros
  ext  <;>  simp  <;>  ring
  mul_assoc  :=  by
  intros
  ext  <;>  simp  <;>  ring
  one_mul  :=  by
  intro
  ext  <;>  simp
  mul_one  :=  by
  intro
  ext  <;>  simp
  left_distrib  :=  by
  intros
  ext  <;>  simp  <;>  ring
  right_distrib  :=  by
  intros
  ext  <;>  simp  <;>  ring
  mul_comm  :=  by
  intros
  ext  <;>  simp  <;>  ring
  zero_mul  :=  by
  intros
  ext  <;>  simp
  mul_zero  :=  by
  intros
  ext  <;>  simp 
```

Lean 的库定义**非平凡**类型为至少有两个不同元素的类型。在环的上下文中，这相当于说零不等于一。由于一些常见的定理依赖于这个事实，我们不妨现在就建立它。

```py
instance  :  Nontrivial  GaussInt  :=  by
  use  0,  1
  rw  [Ne,  GaussInt.ext_iff]
  simp 
```

我们现在将展示高斯整数具有一个重要的附加属性。一个**欧几里得域**是一个带有**范数**函数 $N : R \to \mathbb{N}$ 的环 $R$，它具有以下两个性质：

+   对于 $R$ 中的每个 $a$ 和 $b \ne 0$，存在 $q$ 和 $r$ 在 $R$ 中，使得 $a = bq + r$，并且要么 $r = 0$，要么 $N(r) < N(b)$。

+   对于每个 $a$ 和 $b \ne 0$，$N(a) \le N(ab)$。

整数环 $\Bbb{Z}$ 与 $N(a) = |a|$ 是欧几里得域的一个典型例子。在这种情况下，我们可以取 $q$ 为 $a$ 除以 $b$ 的整数除法的结果，$r$ 为余数。这些函数在 Lean 中定义，以满足以下条件：

```py
example  (a  b  :  ℤ)  :  a  =  b  *  (a  /  b)  +  a  %  b  :=
  Eq.symm  (Int.ediv_add_emod  a  b)

example  (a  b  :  ℤ)  :  b  ≠  0  →  0  ≤  a  %  b  :=
  Int.emod_nonneg  a

example  (a  b  :  ℤ)  :  b  ≠  0  →  a  %  b  <  |b|  :=
  Int.emod_lt_abs  a 
```

在一个任意的环中，一个元素 $a$ 被称为**单位**，如果它能整除 $1$。一个非零元素 $a$ 被称为**不可约**的，如果它不能写成形式 $a = bc$，其中 $b$ 和 $c$ 都不是单位。在整数中，每个不可约元素 $a$ 都是**素数**，也就是说，每当 $a$ 整除一个乘积 $bc$ 时，它要么整除 $b$，要么整除 $c$。但在其他环中，这个性质可能不成立。在环 $\Bbb{Z}[\sqrt{-5}]$ 中，我们有

$$6 = 2 \cdot 3 = (1 + \sqrt{-5})(1 - \sqrt{-5}),$$

并且元素 $2$、$3$、$1 + \sqrt{-5}$ 和 $1 - \sqrt{-5}$ 都是不可约的，但它们不是素数。例如，$2$ 整除乘积 $(1 + \sqrt{-5})(1 - \sqrt{-5})$，但它不整除任何一个因子。特别是，我们不再有唯一的分解：数字 $6$ 可以以多种方式分解为不可约元素。

相比之下，每个欧几里得整环都是一个唯一分解整环，这意味着每个不可约元素都是素数。欧几里得整环的公理意味着任何一个非零元素都可以表示为不可约元素的有限乘积。它们还意味着可以使用欧几里得算法找到任意两个非零元素 $a$ 和 $b$ 的最大公约数，即可以被任何其他公约数整除的元素。这反过来又意味着不可约元素的分解是唯一的，直到乘以单位元素。

现在我们证明高斯整数是具有由 $N(a + bi) = (a + bi)(a - bi) = a² + b²$ 定义的范数的欧几里得整环。高斯整数 $a - bi$ 被称为 $a + bi$ 的**共轭**。对于任何复数 $x$ 和 $y$，我们不难验证 $N(xy) = N(x)N(y)$。

要证明这种范数的定义使得高斯整数成为一个欧几里得整环，只有第一个性质是具有挑战性的。假设我们想要将 $a + bi$ 写成 $a + bi = (c + di) q + r$ 的形式，其中 $q$ 和 $r$ 是合适的。将 $a + bi$ 和 $c + di$ 作为复数处理，进行除法

$$\frac{a + bi}{c + di} = \frac{(a + bi)(c - di)}{(c + di)(c-di)} = \frac{ac + bd}{c² + d²} + \frac{bc -ad}{c²+d²} i.$$

实部和虚部可能不是整数，但我们可以将它们四舍五入到最近的整数 $u$ 和 $v$。然后我们可以将右侧表示为 $(u + vi) + (u' + v'i)$，其中 $u' + v'i$ 是剩余的部分。注意，我们有 $|u'| \le 1/2$ 和 $|v'| \le 1/2$，因此

$$N(u' + v' i) = (u')² + (v')² \le 1/4 + 1/4 \le 1/2.$$

乘以 $c + di$，我们得到

$$a + bi = (c + di) (u + vi) + (c + di) (u' + v'i).$$

令 $q = u + vi$ 和 $r = (c + di) (u' + v'i)$，我们有 $a + bi = (c + di) q + r$，我们只需要界定 $N(r)$：

$$N(r) = N(c + di)N(u' + v'i) \le N(c + di) \cdot 1/2 < N(c + di).$$

我们刚才进行的论证需要将高斯整数视为复数集的一个子集。因此，在 Lean 中形式化它的一个选项是将高斯整数嵌入到复数中，将整数嵌入到高斯整数中，定义从实数到整数的舍入函数，并非常小心地在这些数系之间进行适当的转换。实际上，这正是 Mathlib 中所采用的方法，其中高斯整数本身被构造为**二次整数环**的一个特例。参见文件 [GaussianInt.lean](https://github.com/leanprover-community/mathlib4/blob/master/Mathlib/NumberTheory/Zsqrtd/GaussianInt.lean)。

在这里，我们将执行一个始终保持在整数范围内的论证。这说明了在形式化数学时人们常常面临的一个选择。给定一个需要或机器不在库中的概念或工具的论证，你有两个选择：要么形式化所需的概念和工具，要么调整论证以利用你已有的概念和工具。当结果可以在其他上下文中使用时，第一个选择通常是值得投入时间的。然而，从实用主义的角度来看，有时寻找一个更基础的证明可能更有效。

整数的通常的商余定理表明，对于每一个 $a$ 和非零的 $b$，存在 $q$ 和 $r$ 使得 $a = b q + r$ 且 $0 \le r < b$。在这里，我们将利用以下变体，它表明存在 $q'$ 和 $r'$ 使得 $a = b q' + r'$ 且 $|r'| \le b/2$。你可以检查，如果第一个陈述中的 $r$ 的值满足 $r \le b/2$，我们可以取 $q' = q$ 和 $r' = r$，否则我们可以取 $q' = q + 1$ 和 $r' = r - b$。我们感谢 Heather Macbeth 提出以下更优雅的方法，它避免了分情况定义。我们只是在除法之前将 `b / 2` 加到 `a` 上，然后从余数中减去它。

```py
def  div'  (a  b  :  ℤ)  :=
  (a  +  b  /  2)  /  b

def  mod'  (a  b  :  ℤ)  :=
  (a  +  b  /  2)  %  b  -  b  /  2

theorem  div'_add_mod'  (a  b  :  ℤ)  :  b  *  div'  a  b  +  mod'  a  b  =  a  :=  by
  rw  [div',  mod']
  linarith  [Int.ediv_add_emod  (a  +  b  /  2)  b]

theorem  abs_mod'_le  (a  b  :  ℤ)  (h  :  0  <  b)  :  |mod'  a  b|  ≤  b  /  2  :=  by
  rw  [mod',  abs_le]
  constructor
  ·  linarith  [Int.emod_nonneg  (a  +  b  /  2)  h.ne']
  have  :=  Int.emod_lt_of_pos  (a  +  b  /  2)  h
  have  :=  Int.ediv_add_emod  b  2
  have  :=  Int.emod_lt_of_pos  b  zero_lt_two
  linarith 
```

注意我们老朋友 `linarith` 的使用。我们还将需要用 `div'` 表达 `mod'`。

```py
theorem  mod'_eq  (a  b  :  ℤ)  :  mod'  a  b  =  a  -  b  *  div'  a  b  :=  by  linarith  [div'_add_mod'  a  b] 
```

我们将使用 $x² + y²$ 仅当 $x$ 和 $y$ 都为零时等于零的事实。作为一个练习，我们要求你证明这在任何有序环中都成立。

```py
theorem  sq_add_sq_eq_zero  {α  :  Type*}  [Ring  α]  [LinearOrder  α]  [IsStrictOrderedRing  α]
  (x  y  :  α)  :  x  ^  2  +  y  ^  2  =  0  ↔  x  =  0  ∧  y  =  0  :=  by
  sorry 
```

我们将在这个部分将所有剩余的定义和定理放入`GaussInt`命名空间中。首先，我们定义`norm`函数并要求你建立其一些性质。证明都是简短的。

```py
def  norm  (x  :  GaussInt)  :=
  x.re  ^  2  +  x.im  ^  2

@[simp]
theorem  norm_nonneg  (x  :  GaussInt)  :  0  ≤  norm  x  :=  by
  sorry
theorem  norm_eq_zero  (x  :  GaussInt)  :  norm  x  =  0  ↔  x  =  0  :=  by
  sorry
theorem  norm_pos  (x  :  GaussInt)  :  0  <  norm  x  ↔  x  ≠  0  :=  by
  sorry
theorem  norm_mul  (x  y  :  GaussInt)  :  norm  (x  *  y)  =  norm  x  *  norm  y  :=  by
  sorry 
```

接下来，我们定义共轭函数：

```py
def  conj  (x  :  GaussInt)  :  GaussInt  :=
  ⟨x.re,  -x.im⟩

@[simp]
theorem  conj_re  (x  :  GaussInt)  :  (conj  x).re  =  x.re  :=
  rfl

@[simp]
theorem  conj_im  (x  :  GaussInt)  :  (conj  x).im  =  -x.im  :=
  rfl

theorem  norm_conj  (x  :  GaussInt)  :  norm  (conj  x)  =  norm  x  :=  by  simp  [norm] 
```

最后，我们用 `x / y` 的符号定义高斯整数的除法，它将复数商四舍五入到最接近的高斯整数。我们使用我们定制的 `Int.div'` 来实现这一点。正如我们上面计算的，如果 `x` 是 $a + bi$ 而 `y` 是 $c + di$，那么 `x / y` 的实部和虚部是

$$\frac{ac + bd}{c² + d²} \quad \text{和} \quad \frac{bc -ad}{c²+d²},$$

分别。在这里，分子是 $(a + bi) (c - di)$ 的实部和虚部，分母都是 $c + di$ 的范数。

```py
instance  :  Div  GaussInt  :=
  ⟨fun  x  y  ↦  ⟨Int.div'  (x  *  conj  y).re  (norm  y),  Int.div'  (x  *  conj  y).im  (norm  y)⟩⟩ 
```

在定义了 `x / y` 之后，我们定义 `x % y` 为余数，`x - (x / y) * y`。如上所述，我们在定理 `div_def` 和 `mod_def` 中记录这些定义，以便我们可以使用它们与 `simp` 和 `rw` 一起。

```py
instance  :  Mod  GaussInt  :=
  ⟨fun  x  y  ↦  x  -  y  *  (x  /  y)⟩

theorem  div_def  (x  y  :  GaussInt)  :
  x  /  y  =  ⟨Int.div'  (x  *  conj  y).re  (norm  y),  Int.div'  (x  *  conj  y).im  (norm  y)⟩  :=
  rfl

theorem  mod_def  (x  y  :  GaussInt)  :  x  %  y  =  x  -  y  *  (x  /  y)  :=
  rfl 
```

这些定义立即给出了对于每个 `x` 和 `y` 的 `x = y * (x / y) + x % y`，所以我们只需要证明当 `y` 不为零时，`x % y` 的范数小于 `y` 的范数。

我们刚刚定义了 `x / y` 的实部和虚部为 `div' (x * conj y).re (norm y)` 和 `div' (x * conj y).im (norm y)`，分别。计算后，我们有

> `(x % y) * conj y = (x - x / y * y) * conj y = x * conj y - x / y * (y * conj y)`

右侧的实部和虚部正好是 `mod' (x * conj y).re (norm y)` 和 `mod' (x * conj y).im (norm y)`。根据 `div'` 和 `mod'` 的性质，这些值保证小于或等于 `norm y / 2`。因此我们有

> `norm ((x % y) * conj y) ≤ (norm y / 2)² + (norm y / 2)² ≤ (norm y / 2) * norm y`。

另一方面，我们有

> `norm ((x % y) * conj y) = norm (x % y) * norm (conj y) = norm (x % y) * norm y`。

通过除以 `norm y`，我们得到 `norm (x % y) ≤ (norm y) / 2 < norm y`，正如所需的那样。

这个复杂的计算将在下一个证明中进行。我们鼓励你逐步查看细节，看看你是否能找到一个更好的论点。

```py
theorem  norm_mod_lt  (x  :  GaussInt)  {y  :  GaussInt}  (hy  :  y  ≠  0)  :
  (x  %  y).norm  <  y.norm  :=  by
  have  norm_y_pos  :  0  <  norm  y  :=  by  rwa  [norm_pos]
  have  H1  :  x  %  y  *  conj  y  =  ⟨Int.mod'  (x  *  conj  y).re  (norm  y),  Int.mod'  (x  *  conj  y).im  (norm  y)⟩
  ·  ext  <;>  simp  [Int.mod'_eq,  mod_def,  div_def,  norm]  <;>  ring
  have  H2  :  norm  (x  %  y)  *  norm  y  ≤  norm  y  /  2  *  norm  y
  ·  calc
  norm  (x  %  y)  *  norm  y  =  norm  (x  %  y  *  conj  y)  :=  by  simp  only  [norm_mul,  norm_conj]
  _  =  |Int.mod'  (x.re  *  y.re  +  x.im  *  y.im)  (norm  y)|  ^  2
  +  |Int.mod'  (-(x.re  *  y.im)  +  x.im  *  y.re)  (norm  y)|  ^  2  :=  by  simp  [H1,  norm,  sq_abs]
  _  ≤  (y.norm  /  2)  ^  2  +  (y.norm  /  2)  ^  2  :=  by  gcongr  <;>  apply  Int.abs_mod'_le  _  _  norm_y_pos
  _  =  norm  y  /  2  *  (norm  y  /  2  *  2)  :=  by  ring
  _  ≤  norm  y  /  2  *  norm  y  :=  by  gcongr;  apply  Int.ediv_mul_le;  norm_num
  calc  norm  (x  %  y)  ≤  norm  y  /  2  :=  le_of_mul_le_mul_right  H2  norm_y_pos
  _  <  norm  y  :=  by
  apply  Int.ediv_lt_of_lt_mul
  ·  norm_num
  ·  linarith 
```

我们已经接近终点。我们的 `norm` 函数将高斯整数映射到非负整数。我们需要一个将高斯整数映射到自然数的函数，我们通过将 `norm` 与函数 `Int.natAbs`（将整数映射到自然数）组合来实现这一点。接下来的两个引理中的第一个建立了将范数映射到自然数再映射回整数不会改变值的性质。第二个引理重新表述了范数是递减的事实。

```py
theorem  coe_natAbs_norm  (x  :  GaussInt)  :  (x.norm.natAbs  :  ℤ)  =  x.norm  :=
  Int.natAbs_of_nonneg  (norm_nonneg  _)

theorem  natAbs_norm_mod_lt  (x  y  :  GaussInt)  (hy  :  y  ≠  0)  :
  (x  %  y).norm.natAbs  <  y.norm.natAbs  :=  by
  apply  Int.ofNat_lt.1
  simp  only  [Int.natCast_natAbs,  abs_of_nonneg,  norm_nonneg]
  exact  norm_mod_lt  x  hy 
```

我们还需要建立欧几里得域上范数函数的第二个关键性质。

```py
theorem  not_norm_mul_left_lt_norm  (x  :  GaussInt)  {y  :  GaussInt}  (hy  :  y  ≠  0)  :
  ¬(norm  (x  *  y)).natAbs  <  (norm  x).natAbs  :=  by
  apply  not_lt_of_ge
  rw  [norm_mul,  Int.natAbs_mul]
  apply  le_mul_of_one_le_right  (Nat.zero_le  _)
  apply  Int.ofNat_le.1
  rw  [coe_natAbs_norm]
  exact  Int.add_one_le_of_lt  ((norm_pos  _).mpr  hy) 
```

现在我们可以将其组合起来，以证明高斯整数是欧几里得域的一个实例。我们使用我们定义的商和余数函数。Mathlib 对欧几里得域的定义比上面的定义更通用，因为它允许我们证明余数与任何良基测度相关。比较返回自然数的范数函数的值只是这种测度的一个实例，在这种情况下，所需性质是定理 `natAbs_norm_mod_lt` 和 `not_norm_mul_left_lt_norm`。

```py
instance  :  EuclideanDomain  GaussInt  :=
  {  GaussInt.instCommRing  with
  quotient  :=  (·  /  ·)
  remainder  :=  (·  %  ·)
  quotient_mul_add_remainder_eq  :=
  fun  x  y  ↦  by  rw  [mod_def,  add_comm]  ;  ring
  quotient_zero  :=  fun  x  ↦  by
  simp  [div_def,  norm,  Int.div']
  rfl
  r  :=  (measure  (Int.natAbs  ∘  norm)).1
  r_wellFounded  :=  (measure  (Int.natAbs  ∘  norm)).2
  remainder_lt  :=  natAbs_norm_mod_lt
  mul_left_not_lt  :=  not_norm_mul_left_lt_norm  } 
```

一个直接的好处是，我们现在知道在高斯整数中，素数和不可约的概念是一致的。

```py
example  (x  :  GaussInt)  :  Irreducible  x  ↔  Prime  x  :=
  irreducible_iff_prime 
```  ## 7.1\. 定义结构

在这个术语的最广泛意义上，一个 *结构* 是对一组数据的规范，可能包含数据必须满足的约束。结构的 *实例* 是满足这些约束的特定数据包。例如，我们可以指定一个点是三个实数的元组：

```py
@[ext]
structure  Point  where
  x  :  ℝ
  y  :  ℝ
  z  :  ℝ 
```

`@[ext]` 注解告诉 Lean 自动生成定理，这些定理可以用来证明当结构的不同组件相等时，两个结构实例是相等的，这种性质称为 *外延性*。

```py
#check  Point.ext

example  (a  b  :  Point)  (hx  :  a.x  =  b.x)  (hy  :  a.y  =  b.y)  (hz  :  a.z  =  b.z)  :  a  =  b  :=  by
  ext
  repeat'  assumption 
```

然后，我们可以定义 `Point` 结构的特定实例。Lean 提供了多种实现方式。

```py
def  myPoint1  :  Point  where
  x  :=  2
  y  :=  -1
  z  :=  4

def  myPoint2  :  Point  :=
  ⟨2,  -1,  4⟩

def  myPoint3  :=
  Point.mk  2  (-1)  4 
```

在第一个例子中，结构的字段被明确命名。在`myPoint3`的定义中提到的函数`Point.mk`被称为`Point`结构的*构造器*，因为它用于构建元素。如果你想，可以指定不同的名称，比如`build`。

```py
structure  Point'  where  build  ::
  x  :  ℝ
  y  :  ℝ
  z  :  ℝ

#check  Point'.build  2  (-1)  4 
```

下面的两个例子展示了如何在结构上定义函数。第二个例子使`Point.mk`构造器显式化，而第一个例子为了简洁使用了匿名构造器。Lean 可以从`add`的指示类型推断出相关的构造器。将定义和与结构如`Point`相关的定理放在具有相同名称的命名空间中是一种惯例。在下面的例子中，因为我们已经打开了`Point`命名空间，所以`add`的全称是`Point.add`。当命名空间未打开时，我们必须使用全称。但请记住，使用匿名投影符号通常很方便，它允许我们写出`a.add b`而不是`Point.add a b`。Lean 将前者解释为后者，因为`a`具有`Point`类型。

```py
namespace  Point

def  add  (a  b  :  Point)  :  Point  :=
  ⟨a.x  +  b.x,  a.y  +  b.y,  a.z  +  b.z⟩

def  add'  (a  b  :  Point)  :  Point  where
  x  :=  a.x  +  b.x
  y  :=  a.y  +  b.y
  z  :=  a.z  +  b.z

#check  add  myPoint1  myPoint2
#check  myPoint1.add  myPoint2

end  Point

#check  Point.add  myPoint1  myPoint2
#check  myPoint1.add  myPoint2 
```

下面我们将继续在相关命名空间中放置定义，但我们将省略引用片段中的命名空间命令。为了证明加法函数的性质，我们可以使用`rw`来展开定义，并使用`ext`将结构中两个元素之间的等式简化为组件之间的等式。下面我们使用`protected`关键字，即使命名空间是开放的，定理的名称也是`Point.add_comm`。这在我们想要避免与通用定理如`add_comm`产生歧义时很有帮助。

```py
protected  theorem  add_comm  (a  b  :  Point)  :  add  a  b  =  add  b  a  :=  by
  rw  [add,  add]
  ext  <;>  dsimp
  repeat'  apply  add_comm

example  (a  b  :  Point)  :  add  a  b  =  add  b  a  :=  by  simp  [add,  add_comm] 
```

由于 Lean 可以内部展开定义并简化投影，有时我们想要的等式在定义上是成立的。

```py
theorem  add_x  (a  b  :  Point)  :  (a.add  b).x  =  a.x  +  b.x  :=
  rfl 
```

使用模式匹配在结构上定义函数也是可能的，其方式与我们定义递归函数的方式类似，如第 5.2 节中所述。下面的`addAlt`和`addAlt'`定义本质上相同；唯一的区别在于我们在第二个中使用了匿名构造器符号。尽管有时以这种方式定义函数很方便，结构η-归约使得这种替代定义等价，但它可能会在后续的证明中使事情变得不那么方便。特别是，`rw [addAlt]`会给我们留下一个包含`match`语句的更混乱的目标视图。

```py
def  addAlt  :  Point  →  Point  →  Point
  |  Point.mk  x₁  y₁  z₁,  Point.mk  x₂  y₂  z₂  =>  ⟨x₁  +  x₂,  y₁  +  y₂,  z₁  +  z₂⟩

def  addAlt'  :  Point  →  Point  →  Point
  |  ⟨x₁,  y₁,  z₁⟩,  ⟨x₂,  y₂,  z₂⟩  =>  ⟨x₁  +  x₂,  y₁  +  y₂,  z₁  +  z₂⟩

theorem  addAlt_x  (a  b  :  Point)  :  (a.addAlt  b).x  =  a.x  +  b.x  :=  by
  rfl

theorem  addAlt_comm  (a  b  :  Point)  :  addAlt  a  b  =  addAlt  b  a  :=  by
  rw  [addAlt,  addAlt]
  -- the same proof still works, but the goal view here is harder to read
  ext  <;>  dsimp
  repeat'  apply  add_comm 
```

数学构造通常涉及将捆绑的信息拆分开来，以不同的方式重新组合。因此，Lean 和 Mathlib 提供多种高效执行此操作的方法是有意义的。作为练习，尝试证明`Point.add`是结合的。然后定义点的标量乘法，并证明它对加法是分配的。

```py
protected  theorem  add_assoc  (a  b  c  :  Point)  :  (a.add  b).add  c  =  a.add  (b.add  c)  :=  by
  sorry

def  smul  (r  :  ℝ)  (a  :  Point)  :  Point  :=
  sorry

theorem  smul_distrib  (r  :  ℝ)  (a  b  :  Point)  :
  (smul  r  a).add  (smul  r  b)  =  smul  r  (a.add  b)  :=  by
  sorry 
```

使用结构只是通往代数抽象道路上的第一步。我们还没有一种方法将 `Point.add` 与通用的 `+` 符号链接起来，或者将 `Point.add_comm` 和 `Point.add_assoc` 与通用的 `add_comm` 和 `add_assoc` 公理联系起来。这些任务属于使用结构的 *代数* 方面，我们将在下一节中解释如何执行它们。现在，只需将结构视为将对象和信息捆绑在一起的方式。

特别有用的是，结构不仅可以指定数据类型，还可以指定数据必须满足的约束。在 Lean 中，后者表示为 `Prop` 类型的字段。例如，*标准 2-单纯形* 定义为满足 $x ≥ 0$、$y ≥ 0$、$z ≥ 0$ 和 $x + y + z = 1$ 的点集 $(x, y, z)$。如果你不熟悉这个概念，你应该画一个图，并说服自己这个集合是三维空间中的等边三角形，其顶点为 $(1, 0, 0)$、$(0, 1, 0)$ 和 $(0, 0, 1)$，以及其内部。我们可以在 Lean 中如下表示它：

```py
structure  StandardTwoSimplex  where
  x  :  ℝ
  y  :  ℝ
  z  :  ℝ
  x_nonneg  :  0  ≤  x
  y_nonneg  :  0  ≤  y
  z_nonneg  :  0  ≤  z
  sum_eq  :  x  +  y  +  z  =  1 
```

注意到最后四个字段指的是 `x`、`y` 和 `z`，即前三个字段。我们可以定义一个从二单纯形到自身的映射，该映射交换 `x` 和 `y`：

```py
def  swapXy  (a  :  StandardTwoSimplex)  :  StandardTwoSimplex
  where
  x  :=  a.y
  y  :=  a.x
  z  :=  a.z
  x_nonneg  :=  a.y_nonneg
  y_nonneg  :=  a.x_nonneg
  z_nonneg  :=  a.z_nonneg
  sum_eq  :=  by  rw  [add_comm  a.y  a.x,  a.sum_eq] 
```

更有趣的是，我们可以在单纯形上计算两点之间的中点。我们在文件开头添加了“不可计算部分”这个短语，以便在实数上使用除法。

```py
noncomputable  section

def  midpoint  (a  b  :  StandardTwoSimplex)  :  StandardTwoSimplex
  where
  x  :=  (a.x  +  b.x)  /  2
  y  :=  (a.y  +  b.y)  /  2
  z  :=  (a.z  +  b.z)  /  2
  x_nonneg  :=  div_nonneg  (add_nonneg  a.x_nonneg  b.x_nonneg)  (by  norm_num)
  y_nonneg  :=  div_nonneg  (add_nonneg  a.y_nonneg  b.y_nonneg)  (by  norm_num)
  z_nonneg  :=  div_nonneg  (add_nonneg  a.z_nonneg  b.z_nonneg)  (by  norm_num)
  sum_eq  :=  by  field_simp;  linarith  [a.sum_eq,  b.sum_eq] 
```

在这里，我们用简洁的证明项建立了 `x_nonneg`、`y_nonneg` 和 `z_nonneg`，但使用 `by` 在战术模式中建立 `sum_eq`。

给定一个满足 $0 \le \lambda \le 1$ 的参数 $\lambda$，我们可以在标准 2-单纯形上取两个点 $a$ 和 $b$ 的加权平均 $\lambda a + (1 - \lambda) b$。我们挑战你定义这个函数，类似于上面的 `midpoint` 函数。

```py
def  weightedAverage  (lambda  :  Real)  (lambda_nonneg  :  0  ≤  lambda)  (lambda_le  :  lambda  ≤  1)
  (a  b  :  StandardTwoSimplex)  :  StandardTwoSimplex  :=
  sorry 
```

结构可以依赖于参数。例如，我们可以将标准 2-单纯形推广到任何 $n$ 的标准 $n$-单纯形。在这个阶段，你不必了解 `Fin n` 类型，除了它有 $n$ 个元素，以及 Lean 知道如何对它求和。

```py
open  BigOperators

structure  StandardSimplex  (n  :  ℕ)  where
  V  :  Fin  n  →  ℝ
  NonNeg  :  ∀  i  :  Fin  n,  0  ≤  V  i
  sum_eq_one  :  (∑  i,  V  i)  =  1

namespace  StandardSimplex

def  midpoint  (n  :  ℕ)  (a  b  :  StandardSimplex  n)  :  StandardSimplex  n
  where
  V  i  :=  (a.V  i  +  b.V  i)  /  2
  NonNeg  :=  by
  intro  i
  apply  div_nonneg
  ·  linarith  [a.NonNeg  i,  b.NonNeg  i]
  norm_num
  sum_eq_one  :=  by
  simp  [div_eq_mul_inv,  ←  Finset.sum_mul,  Finset.sum_add_distrib,
  a.sum_eq_one,  b.sum_eq_one]
  field_simp

end  StandardSimplex 
```

作为练习，看看你是否可以定义标准 $n$-单纯形上两个点的加权平均。你可以使用 `Finset.sum_add_distrib` 和 `Finset.mul_sum` 来操作相关的和。

我们已经看到，结构可以用来捆绑数据和属性。有趣的是，它们也可以用来捆绑属性而不包含数据。例如，下一个结构 `IsLinear` 捆绑了线性的两个组成部分。

```py
structure  IsLinear  (f  :  ℝ  →  ℝ)  where
  is_additive  :  ∀  x  y,  f  (x  +  y)  =  f  x  +  f  y
  preserves_mul  :  ∀  x  c,  f  (c  *  x)  =  c  *  f  x

section
variable  (f  :  ℝ  →  ℝ)  (linf  :  IsLinear  f)

#check  linf.is_additive
#check  linf.preserves_mul

end 
```

值得指出的是，结构并不是捆绑数据的唯一方式。`Point` 数据结构可以使用通用类型积来定义，而 `IsLinear` 可以用简单的 `and` 定义。

```py
def  Point''  :=
  ℝ  ×  ℝ  ×  ℝ

def  IsLinear'  (f  :  ℝ  →  ℝ)  :=
  (∀  x  y,  f  (x  +  y)  =  f  x  +  f  y)  ∧  ∀  x  c,  f  (c  *  x)  =  c  *  f  x 
```

通用类型构造甚至可以用来代替其组件之间有依赖关系的结构。例如，**子类型**构造将数据与属性结合起来。你可以在下一个示例中将类型`PReal`视为正实数的类型。任何`x : PReal`都有两个组件：值和正属性。你可以通过`x.val`访问这些组件，它具有类型`ℝ`，以及`x.property`，它代表`0 < x.val`的事实。

```py
def  PReal  :=
  {  y  :  ℝ  //  0  <  y  }

section
variable  (x  :  PReal)

#check  x.val
#check  x.property
#check  x.1
#check  x.2

end 
```

我们本可以使用子类型来定义标准的 2-单纯形，以及任意$n$的标准$n$-单纯形。

```py
def  StandardTwoSimplex'  :=
  {  p  :  ℝ  ×  ℝ  ×  ℝ  //  0  ≤  p.1  ∧  0  ≤  p.2.1  ∧  0  ≤  p.2.2  ∧  p.1  +  p.2.1  +  p.2.2  =  1  }

def  StandardSimplex'  (n  :  ℕ)  :=
  {  v  :  Fin  n  →  ℝ  //  (∀  i  :  Fin  n,  0  ≤  v  i)  ∧  (∑  i,  v  i)  =  1  } 
```

同样，**Sigma 类型**是有序对的推广，其中第二个组件的类型取决于第一个组件的类型。

```py
def  StdSimplex  :=  Σ  n  :  ℕ,  StandardSimplex  n

section
variable  (s  :  StdSimplex)

#check  s.fst
#check  s.snd

#check  s.1
#check  s.2

end 
```

给定`s : StdSimplex`，第一个组件`s.fst`是一个自然数，第二个组件是相应单纯形`StandardSimplex s.fst`的一个元素。Sigma 类型与子类型之间的区别在于，Sigma 类型的第二个组件是数据而不是命题。

尽管我们可以使用产品、子类型和 Sigma 类型来代替结构体，但使用结构体仍然具有许多优点。定义一个结构体会抽象出底层的表示，并为访问组件的函数提供自定义名称。这使得证明更加健壮：仅依赖于结构体接口的证明，在定义更改时通常仍然有效，只要我们用新的定义重新定义旧的访问器。此外，正如我们即将看到的，Lean 提供了将结构体编织成丰富、相互关联的层次结构以及管理它们之间交互的支持。

## 7.2\. 代数结构

为了阐明我们所说的“代数结构”的含义，考虑一些例子会有所帮助。

1.  一个**偏序集**由一个集合$P$和一个在$P$上的传递且自反的二元关系$\le$组成。

1.  一个**群**由一个具有结合二元运算、单位元素$1$和函数$g \mapsto g^{-1}$的集合$G$组成，该函数为$G$中的每个$g$返回一个逆元素。如果运算是对称的，则群是**阿贝尔的**或**交换的**。

1.  一个**格**是具有交集和并集的偏序集。

1.  一个**环**由一个（加法表示的）阿贝尔群$(R, +, 0, x \mapsto -x)$以及一个结合乘法运算$\cdot$和一个单位$1$组成，使得乘法对加法分配。如果乘法是交换的，则环是**交换的**。

1.  一个**有序环**$(R, +, 0, -, \cdot, 1, \le)$由一个环及其元素上的偏序组成，使得对于$R$中的每个$a$、$b$和$c$，$a \le b$意味着$a + c \le b + c$，并且$0 \le a$和$0 \le b$意味着对于$R$中的每个$a$和$b$，$0 \le a b$。

1.  一个**度量空间**由一个集合 $X$ 和一个函数 $d : X \times X \to \mathbb{R}$ 组成，使得以下条件成立：

    +   对于 $X$ 中的每个 $x$ 和 $y$，有 $d(x, y) \ge 0$。

    +   $d(x, y) = 0$ 当且仅当 $x = y$。

    +   对于 $X$ 中的每个 $x$ 和 $y$，有 $d(x, y) = d(y, x)$。

    +   对于 $X$ 中的每个 $x$，$y$ 和 $z$，有 $d(x, z) \le d(x, y) + d(y, z)$。

1.  一个**拓扑空间**由一个集合 $X$ 和一个 $X$ 的子集集合 $\mathcal T$ 组成，称为 $X$ 的**开子集**，使得以下条件成立：

    +   空集和 $X$ 是开集。

    +   两个开集的交集是开集。

    +   开集的任意并集是开集。

在这些例子中，结构的元素属于一个集合，称为**载体集**，有时它代表整个结构。例如，当我们说“设 $G$ 为一个群”然后“设 $g \in G$”时，我们使用 $G$ 来代表结构和它的载体。并非每个代数结构都以这种方式与单个载体集相关联。例如，一个**二部图**涉及两个集合之间的关系，正如**高斯连接**一样，一个**范畴**也涉及两个感兴趣的集合，通常称为**对象**和**态射**。

例子表明，证明辅助工具必须执行一些事情以支持代数推理。首先，它需要识别结构的具体实例。数系 $\mathbb{Z}$，$\mathbb{Q}$，和 $\mathbb{R}$ 都是序环，我们应该能够在这些实例中的任何一个上应用关于序环的通用定理。有时一个具体集合可能以多种方式成为某个结构的一个实例。例如，除了 $\mathbb{R}$ 上的通常拓扑，它是实分析的基础之外，我们还可以考虑 $\mathbb{R}$ 上的**离散**拓扑，在这种拓扑中，每个集合都是开集。

其次，证明辅助工具需要支持结构上的通用符号。在 Lean 中，符号 `*` 用于所有常规数系中的乘法，以及用于通用群和环中的乘法。当我们使用像 `f x * y` 这样的表达式时，Lean 必须使用有关 `f`，`x` 和 `y` 的类型的信息来确定我们心中所想的乘法是什么。

第三，需要处理这样一个事实：结构可以通过各种方式从其他结构继承定义、定理和符号。一些结构通过添加更多的公理来扩展其他结构。交换环仍然是一个环，所以任何在环中具有意义的定义在交换环中也具有意义，任何在环中成立的定理在交换环中也成立。一些结构通过添加更多的数据来扩展其他结构。例如，任何环的加法部分是一个加法群。环结构添加了乘法和单位元，以及规范这些乘法和单位元并使其与加法部分相关联的公理。有时我们可以用另一个结构来定义一个结构。任何度量空间都有一个与之相关的典型拓扑，即*度量空间拓扑*，还有各种可以与任何线性顺序相关联的拓扑。

最后，重要的是要记住，数学允许我们使用函数和运算以定义结构的方式，就像我们使用函数和运算来定义数字一样。群的乘积和幂仍然是群。对于每一个$n$，模$n$的整数构成一个环，对于每一个$k > 0$，系数在该环中的$k \times k$多项式矩阵再次构成一个环。因此，我们可以像计算它们的元素一样容易地计算结构。这意味着代数结构在数学中过着双重生活，既是对象集合的容器，也是它们自身的对象。证明辅助工具必须适应这种双重角色。

当处理与代数结构相关联的类型元素时，证明辅助工具需要识别该结构并找到相关的定义、定理和符号。所有这些听起来像是一大堆工作，确实如此。但 Lean 使用一小套基本机制来完成这些任务。本节的目标是解释这些机制并展示如何使用它们。

第一个要素几乎是显而易见的：从形式上讲，代数结构是第 7.1 节意义上的结构。代数结构是一组满足某些公理假设的数据的规范，我们在第 7.1 节中看到，这正是`structure`命令设计用来适应的。这是一场天作之合！

给定一个数据类型`α`，我们可以如下定义`α`上的群结构。

```py
structure  Group₁  (α  :  Type*)  where
  mul  :  α  →  α  →  α
  one  :  α
  inv  :  α  →  α
  mul_assoc  :  ∀  x  y  z  :  α,  mul  (mul  x  y)  z  =  mul  x  (mul  y  z)
  mul_one  :  ∀  x  :  α,  mul  x  one  =  x
  one_mul  :  ∀  x  :  α,  mul  one  x  =  x
  inv_mul_cancel  :  ∀  x  :  α,  mul  (inv  x)  x  =  one 
```

注意，类型`α`是`Group₁`定义中的一个*参数*。因此，你应该将对象`struc : Group₁ α`视为`α`上的群结构。我们在第 2.2 节中看到，`inv_mul_cancel`的对应项`mul_inv_cancel`可以从其他群公理中得出，因此没有必要将其添加到定义中。

这个群的定义与 Mathlib 中 `Group` 的定义相似，我们选择了 `Group₁` 这个名字来区分我们的版本。如果你输入 `#check Group` 并在定义上按住 Ctrl 点击，你会看到 Mathlib 版本的 `Group` 是定义为扩展另一个结构的；我们将在稍后解释如何做到这一点。如果你输入 `#print Group`，你也会看到 Mathlib 版本的 `Group` 有许多额外的字段。由于我们将在稍后解释的原因，有时在结构中添加冗余信息是有用的，这样就可以为从核心数据定义的对象和函数提供额外的字段。现在不用担心这个。放心，我们的简化版本 `Group₁` 在道德上是与 Mathlib 使用的群定义相同的。

有时将类型与结构捆绑在一起是有用的，Mathlib 也包含了一个与以下等价的 `Grp` 结构定义：

```py
structure  Grp₁  where
  α  :  Type*
  str  :  Group₁  α 
```

Mathlib 版本可以在 `Mathlib.Algebra.Category.Grp.Basic` 中找到，如果你在示例文件的开始处添加这个导入，你可以使用 `#check` 来检查它。

由于以下原因将在下面变得更为清晰，通常更有用的是将类型 `α` 与结构 `Group α` 分开。我们将这两个对象一起称为 *部分捆绑结构*，因为表示结合了大多数但不是所有组件到一个结构中。在 Mathlib 中，当用作群的载体类型时，通常使用大写罗马字母如 `G` 来表示类型。

让我们构建一个群，也就是说，`Group₁` 类型的元素。对于任何一对类型 `α` 和 `β`，Mathlib 定义了 `α` 和 `β` 之间等价关系的类型 `Equiv α β`。Mathlib 还为这个类型定义了有启发性的符号 `α ≃ β`。一个元素 `f : α ≃ β` 是 `α` 和 `β` 之间的双射，由四个组成部分表示：一个从 `α` 到 `β` 的函数 `f.toFun`，一个从 `β` 到 `α` 的逆函数 `f.invFun`，以及两个指定这些函数确实是彼此的逆的性质。

```py
variable  (α  β  γ  :  Type*)
variable  (f  :  α  ≃  β)  (g  :  β  ≃  γ)

#check  Equiv  α  β
#check  (f.toFun  :  α  →  β)
#check  (f.invFun  :  β  →  α)
#check  (f.right_inv  :  ∀  x  :  β,  f  (f.invFun  x)  =  x)
#check  (f.left_inv  :  ∀  x  :  α,  f.invFun  (f  x)  =  x)
#check  (Equiv.refl  α  :  α  ≃  α)
#check  (f.symm  :  β  ≃  α)
#check  (f.trans  g  :  α  ≃  γ) 
```

注意最后三个构造的创造性命名。我们认为恒等函数 `Equiv.refl`、逆操作 `Equiv.symm` 和复合操作 `Equiv.trans` 是双射对应性质是等价关系的明确证据。

注意到 `f.trans g` 需要按逆序组合前向函数。Mathlib 声明了一个从 `Equiv α β` 到函数类型 `α → β` 的 *强制转换*，因此我们可以省略写 `.toFun`，让 Lean 为我们插入它。

```py
example  (x  :  α)  :  (f.trans  g).toFun  x  =  g.toFun  (f.toFun  x)  :=
  rfl

example  (x  :  α)  :  (f.trans  g)  x  =  g  (f  x)  :=
  rfl

example  :  (f.trans  g  :  α  →  γ)  =  g  ∘  f  :=
  rfl 
```

Mathlib 还定义了 `α` 和自身之间等价关系的类型 `perm α`。

```py
example  (α  :  Type*)  :  Equiv.Perm  α  =  (α  ≃  α)  :=
  rfl 
```

应该很明显，`Equiv.Perm α` 在等价关系的复合下形成一个群。我们这样安排，使得 `mul f g` 等于 `g.trans f`，其前向函数是 `f ∘ g`。换句话说，乘法就是我们通常认为的双射的复合。在这里，我们定义这个群：

```py
def  permGroup  {α  :  Type*}  :  Group₁  (Equiv.Perm  α)
  where
  mul  f  g  :=  Equiv.trans  g  f
  one  :=  Equiv.refl  α
  inv  :=  Equiv.symm
  mul_assoc  f  g  h  :=  (Equiv.trans_assoc  _  _  _).symm
  one_mul  :=  Equiv.trans_refl
  mul_one  :=  Equiv.refl_trans
  inv_mul_cancel  :=  Equiv.self_trans_symm 
```

事实上，Mathlib 在 `Algebra.Group.End` 文件中精确地定义了 `Equiv.Perm α` 上的这个 `Group` 结构。像往常一样，你可以悬停在 `permGroup` 定义中使用的定理上，以查看它们的陈述，并且你可以跳转到原始文件中的定义，以了解更多关于它们是如何实现的。

在常规数学中，我们通常认为符号与结构是独立的。例如，我们可以考虑群 $(G_1, \cdot, 1, \cdot^{-1})$、$(G_2, \circ, e, i(\cdot))$ 和 $(G_3, +, 0, -)$。在第一种情况下，我们用 $\cdot$ 表示二元运算，用 $1$ 表示单位元，用 $x \mapsto x^{-1}$ 表示逆函数。在第二种和第三种情况下，我们使用显示的符号替代。然而，当我们用 Lean 正式化群的观念时，符号与结构之间的联系更为紧密。在 Lean 中，任何 `Group` 的组成部分被命名为 `mul`、`one` 和 `inv`，我们很快就会看到乘法符号是如何设置来引用它们的。如果我们想使用加法符号，我们则使用同构结构 `AddGroup`（加法群的底层结构）。它的组成部分被命名为 `add`、`zero` 和 `neg`，相关的符号正如你所期望的那样。

回想一下我们在 第 7.1 节 中定义的 `Point` 类型以及我们那里定义的加法函数。这些定义在伴随本节的示例文件中重现。作为一个练习，定义一个类似于我们上面定义的 `Group₁` 结构的 `AddGroup₁` 结构，但使用前面描述的加法命名方案。在 `Point` 数据类型上定义否定和零，并在 `Point` 上定义 `AddGroup₁` 结构。

```py
structure  AddGroup₁  (α  :  Type*)  where
  (add  :  α  →  α  →  α)
  -- fill in the rest
@[ext]
structure  Point  where
  x  :  ℝ
  y  :  ℝ
  z  :  ℝ

namespace  Point

def  add  (a  b  :  Point)  :  Point  :=
  ⟨a.x  +  b.x,  a.y  +  b.y,  a.z  +  b.z⟩

def  neg  (a  :  Point)  :  Point  :=  sorry

def  zero  :  Point  :=  sorry

def  addGroupPoint  :  AddGroup₁  Point  :=  sorry

end  Point 
```

我们正在取得进展。现在我们知道如何在 Lean 中定义代数结构，也知道如何定义这些结构的实例。但我们还希望将符号与结构关联起来，以便我们可以与每个实例一起使用它。此外，我们希望安排好，以便我们可以在结构上定义一个运算并使用任何特定的实例，我们还想安排好，以便我们可以在结构上证明一个定理并使用任何实例。

事实上，Mathlib 已经设置好，为 `Equiv.Perm α` 使用通用的群符号、定义和定理。

```py
variable  {α  :  Type*}  (f  g  :  Equiv.Perm  α)  (n  :  ℕ)

#check  f  *  g
#check  mul_assoc  f  g  g⁻¹

-- group power, defined for any group
#check  g  ^  n

example  :  f  *  g  *  g⁻¹  =  f  :=  by  rw  [mul_assoc,  mul_inv_cancel,  mul_one]

example  :  f  *  g  *  g⁻¹  =  f  :=
  mul_inv_cancel_right  f  g

example  {α  :  Type*}  (f  g  :  Equiv.Perm  α)  :  g.symm.trans  (g.trans  f)  =  f  :=
  mul_inv_cancel_right  f  g 
```

你可以检查我们上面要求你定义的 `Point` 上的加法群结构并不是这样。我们的任务现在是要理解在 `Equiv.Perm α` 的例子背后发生的魔法，以便它们能按预期工作。

问题在于 Lean 需要能够 *找到* 相关的符号和隐含的群结构，使用我们在输入的表达式中找到的信息。同样，当我们用具有类型 `ℝ` 的表达式 `x` 和 `y` 写 `x + y` 时，Lean 需要解释 `+` 符号为实数上的相关加法函数。它还必须识别类型 `ℝ` 为交换环的实例，这样所有交换环的定义和定理都可用。例如，连续性在 Lean 中相对于任何两个拓扑空间定义。当我们有 `f : ℝ → ℂ` 并写 `Continuous f` 时，Lean 必须找到 `ℝ` 和 `ℂ` 上的相关拓扑。

这个魔法是通过三件事的结合实现的。

1.  *逻辑。* 任何一组中应被解释的定义，将组类型和组结构作为论证。同样，关于任意组元素的定理，也是从组类型和组结构的全称量词开始的。

1.  *隐含参数。* 类型结构和结构的参数通常被省略，这样我们就不必写它们或在 Lean 信息窗口中看到它们。Lean 会默默地为我们填写信息。

1.  *类型类推断。* 也称为 *类推断*，这是一个简单但强大的机制，使我们能够为 Lean 注册信息以供以后使用。当 Lean 被调用以填充定义、定理或符号中的隐含参数时，它可以利用已注册的信息。

而一个注释 `(grp : Group G)` 告诉 Lean 应该明确地期望得到这个论证，注释 `{grp : Group G}` 告诉 Lean 应该尝试从表达式的上下文中推断它，注释 `[grp : Group G]` 告诉 Lean 相应的论证应该通过类型类推断来合成。由于使用此类论证的整个目的是我们通常不需要明确地引用它们，Lean 允许我们写出 `[Group G]` 并使名称匿名。你可能已经注意到 Lean 会自动选择像 `_inst_1` 这样的名称。当我们使用匿名方括号注释与 `variables` 命令一起使用时，只要变量仍然在作用域内，Lean 就会自动将 `[Group G]` 参数添加到任何提及 `G` 的定义或定理中。

我们如何注册 Lean 需要用于执行搜索的信息？回到我们的群例子，我们只需要做两个更改。首先，我们不是使用 `structure` 命令来定义群结构，而是使用关键字 `class` 来表示它是类推断的候选。其次，我们不是使用 `def` 定义特定的实例，而是使用关键字 `instance` 将特定的实例注册到 Lean 中。与类变量名称一样，我们可以省略实例定义的名称，因为在一般情况下，我们希望 Lean 找到它并投入使用，而不必麻烦我们处理细节。

```py
class  Group₂  (α  :  Type*)  where
  mul  :  α  →  α  →  α
  one  :  α
  inv  :  α  →  α
  mul_assoc  :  ∀  x  y  z  :  α,  mul  (mul  x  y)  z  =  mul  x  (mul  y  z)
  mul_one  :  ∀  x  :  α,  mul  x  one  =  x
  one_mul  :  ∀  x  :  α,  mul  one  x  =  x
  inv_mul_cancel  :  ∀  x  :  α,  mul  (inv  x)  x  =  one

instance  {α  :  Type*}  :  Group₂  (Equiv.Perm  α)  where
  mul  f  g  :=  Equiv.trans  g  f
  one  :=  Equiv.refl  α
  inv  :=  Equiv.symm
  mul_assoc  f  g  h  :=  (Equiv.trans_assoc  _  _  _).symm
  one_mul  :=  Equiv.trans_refl
  mul_one  :=  Equiv.refl_trans
  inv_mul_cancel  :=  Equiv.self_trans_symm 
```

下面的例子说明了它们的使用。

```py
#check  Group₂.mul

def  mySquare  {α  :  Type*}  [Group₂  α]  (x  :  α)  :=
  Group₂.mul  x  x

#check  mySquare

section
variable  {β  :  Type*}  (f  g  :  Equiv.Perm  β)

example  :  Group₂.mul  f  g  =  g.trans  f  :=
  rfl

example  :  mySquare  f  =  f.trans  f  :=
  rfl

end 
```

`#check` 命令显示 `Group₂.mul` 有一个隐含的参数 `[Group₂ α]`，我们期望它通过类推断找到，其中 `α` 是 `Group₂.mul` 参数的类型。换句话说，`{α : Type*}` 是群元素类型的隐含参数，而 `[Group₂ α]` 是 `α` 上群结构的隐含参数。同样，当我们为 `Group₂` 定义一个通用的平方函数 `my_square` 时，我们使用一个隐含参数 `{α : Type*}` 用于元素类型，以及一个隐含参数 `[Group₂ α]` 用于 `Group₂` 结构。

在第一个例子中，当我们写下 `Group₂.mul f g` 时，`f` 和 `g` 的类型告诉 Lean 在 `Group₂.mul` 的参数 `α` 必须实例化为 `Equiv.Perm β`。这意味着 Lean 必须找到一个 `Group₂ (Equiv.Perm β)` 的元素。之前的 `instance` 声明告诉 Lean 如何做到这一点。问题解决！

这种简单的机制，用于注册信息以便 Lean 在需要时可以找到它，是非常有用的。这里有一种它出现的方式。在 Lean 的基础上，一个数据类型 `α` 可能是空的。然而，在许多应用中，知道一个类型至少有一个元素是有用的。例如，函数 `List.headI`，它返回列表的第一个元素，当列表为空时可以返回默认值。为了使它工作，Lean 库定义了一个类 `Inhabited α`，它所做的只是存储一个默认值。我们可以证明 `Point` 类型是一个实例：

```py
instance  :  Inhabited  Point  where  default  :=  ⟨0,  0,  0⟩

#check  (default  :  Point)

example  :  ([]  :  List  Point).headI  =  default  :=
  rfl 
```

类推断机制也用于泛型符号。表达式 `x + y` 是 `Add.add x y` 的缩写，其中——正如你所猜的——`Add α` 是一个存储在 `α` 上的二元函数的类。写下 `x + y` 告诉 Lean 找到一个注册的 `[Add.add α]` 实例，并使用相应的函数。下面，我们注册了 `Point` 的加法函数。

```py
instance  :  Add  Point  where  add  :=  Point.add

section
variable  (x  y  :  Point)

#check  x  +  y

example  :  x  +  y  =  Point.add  x  y  :=
  rfl

end 
```

以这种方式，我们也可以将符号 `+` 分配给其他类型的二元运算。

但我们可以做得更好。我们已经看到`*`可以在任何群中使用，`+`可以在任何加法群中使用，并且两者都可以在任何环中使用。当我们定义 Lean 中的新环实例时，我们不需要为该实例定义`+`和`*`，因为 Lean 知道这些在每一个环中都是定义好的。我们可以使用这种方法为我们的`Group₂`类指定符号：

```py
instance  {α  :  Type*}  [Group₂  α]  :  Mul  α  :=
  ⟨Group₂.mul⟩

instance  {α  :  Type*}  [Group₂  α]  :  One  α  :=
  ⟨Group₂.one⟩

instance  {α  :  Type*}  [Group₂  α]  :  Inv  α  :=
  ⟨Group₂.inv⟩

section
variable  {α  :  Type*}  (f  g  :  Equiv.Perm  α)

#check  f  *  1  *  g⁻¹

def  foo  :  f  *  1  *  g⁻¹  =  g.symm.trans  ((Equiv.refl  α).trans  f)  :=
  rfl

end 
```

使这种方法有效的是 Lean 执行递归搜索。根据我们声明的实例，Lean 可以通过找到`Group₂ (Equiv.Perm α)`的实例来找到`Mul (Equiv.Perm α)`的实例，并且它可以通过我们提供的一个实例来找到`Group₂ (Equiv.Perm α)`的实例。Lean 能够找到这两个事实并将它们串联起来。

我们刚刚给出的例子是危险的，因为 Lean 的库也有一个`Group (Equiv.Perm α)`的实例，并且乘法在任意群上都是定义好的。所以找到哪个实例是不确定的。实际上，Lean 倾向于优先考虑较新的声明，除非你明确指定了不同的优先级。此外，还有另一种方法告诉 Lean 一个结构是另一个结构的实例，使用`extends`关键字。这就是 Mathlib 指定例如每个交换环都是环的方式。你可以在第八部分和*《Lean 中的定理证明》*中的一个关于类推断的[章节](https://leanprover.github.io/theorem_proving_in_lean4/type_classes.html#managing-type-class-inference)中找到更多信息。

通常来说，为一个已经定义了符号的代数结构实例指定`*`的值是一个坏主意。在 Lean 中重新定义`Group`的概念是一个人为的例子。然而，在这种情况下，群符号的两种解释都展开为`Equiv.trans`、`Equiv.refl`和`Equiv.symm`，方式相同。

作为一项类似的练习，定义一个与`Group₂`类似的类`AddGroup₂`。在任意的`AddGroup₂`上使用类`Add`、`Neg`和`Zero`定义加法、减法和零的常规符号。然后证明`Point`是`AddGroup₂`的一个实例。尝试一下，确保加法群符号对`Point`的元素有效。

```py
class  AddGroup₂  (α  :  Type*)  where
  add  :  α  →  α  →  α
  -- fill in the rest 
```

我们已经为`Point`声明了实例`Add`、`Neg`和`Zero`，这并不是一个大问题。再次强调，两种合成符号的方式应该得出相同的答案。

类推断很微妙，使用时必须小心，因为它配置了自动化，这种自动化无形中控制着我们输入表达式的解释。然而，如果使用得当，类推断是一个强大的工具。它是使在 Lean 中进行代数推理成为可能的原因。

## 7.3\. 构建高斯整数

现在我们将通过构建一个重要的数学对象，高斯整数（*Gaussian integers*），并展示它是一个欧几里得域，来展示在 Lean 中使用代数层次结构的应用。换句话说，根据我们一直在使用的术语，我们将定义高斯整数并展示它们是欧几里得域结构的一个实例。

在普通数学术语中，高斯整数集 $\Bbb{Z}[i]$ 是复数集 $\{ a + b i \mid a, b \in \Bbb{Z}\}$。但我们的目标不是将它们定义为复数集的子集，而是将它们定义为一个独立的数据类型。我们通过将高斯整数表示为整数对来实现这一点，我们将这些整数视为 *实部* 和 *虚部*。

```py
@[ext]
structure  GaussInt  where
  re  :  ℤ
  im  :  ℤ 
```

我们首先展示高斯整数具有环的结构，其中 `0` 定义为 `⟨0, 0⟩`，`1` 定义为 `⟨1, 0⟩`，加法定义为逐点。为了确定乘法定义，记住我们希望元素 $i$，由 `⟨0, 1⟩` 表示，是 $-1$ 的一个平方根。因此我们希望

$$\begin{split}(a + bi) (c + di) & = ac + bci + adi + bd i² \\ & = (ac - bd) + (bc + ad)i.\end{split}$$

这解释了下面 `Mul` 的定义。

```py
instance  :  Zero  GaussInt  :=
  ⟨⟨0,  0⟩⟩

instance  :  One  GaussInt  :=
  ⟨⟨1,  0⟩⟩

instance  :  Add  GaussInt  :=
  ⟨fun  x  y  ↦  ⟨x.re  +  y.re,  x.im  +  y.im⟩⟩

instance  :  Neg  GaussInt  :=
  ⟨fun  x  ↦  ⟨-x.re,  -x.im⟩⟩

instance  :  Mul  GaussInt  :=
  ⟨fun  x  y  ↦  ⟨x.re  *  y.re  -  x.im  *  y.im,  x.re  *  y.im  +  x.im  *  y.re⟩⟩ 
```

如 第 7.1 节 所述，将一个数据类型相关的所有定义放在具有相同名称的空间中是一个好主意。因此，在本章相关的 Lean 文件中，这些定义是在 `GaussInt` 空间中进行的。

注意，在这里我们直接定义了记号 `0`、`1`、`+`、`-` 和 `*` 的解释，而不是将它们命名为 `GaussInt.zero` 等并分配给这些记号。为定义提供显式的名称通常很有用，例如与 `simp` 和 `rw` 一起使用。

```py
theorem  zero_def  :  (0  :  GaussInt)  =  ⟨0,  0⟩  :=
  rfl

theorem  one_def  :  (1  :  GaussInt)  =  ⟨1,  0⟩  :=
  rfl

theorem  add_def  (x  y  :  GaussInt)  :  x  +  y  =  ⟨x.re  +  y.re,  x.im  +  y.im⟩  :=
  rfl

theorem  neg_def  (x  :  GaussInt)  :  -x  =  ⟨-x.re,  -x.im⟩  :=
  rfl

theorem  mul_def  (x  y  :  GaussInt)  :
  x  *  y  =  ⟨x.re  *  y.re  -  x.im  *  y.im,  x.re  *  y.im  +  x.im  *  y.re⟩  :=
  rfl 
```

给出计算实部和虚部的规则并将其声明给简化器也是有用的。

```py
@[simp]
theorem  zero_re  :  (0  :  GaussInt).re  =  0  :=
  rfl

@[simp]
theorem  zero_im  :  (0  :  GaussInt).im  =  0  :=
  rfl

@[simp]
theorem  one_re  :  (1  :  GaussInt).re  =  1  :=
  rfl

@[simp]
theorem  one_im  :  (1  :  GaussInt).im  =  0  :=
  rfl

@[simp]
theorem  add_re  (x  y  :  GaussInt)  :  (x  +  y).re  =  x.re  +  y.re  :=
  rfl

@[simp]
theorem  add_im  (x  y  :  GaussInt)  :  (x  +  y).im  =  x.im  +  y.im  :=
  rfl

@[simp]
theorem  neg_re  (x  :  GaussInt)  :  (-x).re  =  -x.re  :=
  rfl

@[simp]
theorem  neg_im  (x  :  GaussInt)  :  (-x).im  =  -x.im  :=
  rfl

@[simp]
theorem  mul_re  (x  y  :  GaussInt)  :  (x  *  y).re  =  x.re  *  y.re  -  x.im  *  y.im  :=
  rfl

@[simp]
theorem  mul_im  (x  y  :  GaussInt)  :  (x  *  y).im  =  x.re  *  y.im  +  x.im  *  y.re  :=
  rfl 
```

现在出人意料地容易证明高斯整数是一个交换环的实例。我们正在很好地利用结构概念。每个特定的高斯整数是 `GaussInt` 结构的一个实例，而 `GaussInt` 类型本身，连同相关操作，是 `CommRing` 结构的一个实例。`CommRing` 结构反过来又扩展了 `Zero`、`One`、`Add`、`Neg` 和 `Mul` 等记法结构。

如果你输入 `instance : CommRing GaussInt := _`，然后在 VS Code 中点击出现的灯泡图标，并让 Lean 填写结构定义的骨架，你会看到很多条目。然而，跳转到结构的定义，你会发现许多字段都有默认定义，Lean 会自动为你填写。下面列出了关键的定义。一个特殊情况是 `nsmul` 和 `zsmul`，目前可以忽略，将在下一章中解释。在每种情况下，相关恒等式都是通过展开定义、使用 `ext` 策略将恒等式简化为其实部和虚部、简化，并在必要时在整数中进行相关环计算来证明的。请注意，我们可以轻松避免重复所有这些代码，但这不是当前讨论的主题。

```py
instance  instCommRing  :  CommRing  GaussInt  where
  zero  :=  0
  one  :=  1
  add  :=  (·  +  ·)
  neg  x  :=  -x
  mul  :=  (·  *  ·)
  nsmul  :=  nsmulRec
  zsmul  :=  zsmulRec
  add_assoc  :=  by
  intros
  ext  <;>  simp  <;>  ring
  zero_add  :=  by
  intro
  ext  <;>  simp
  add_zero  :=  by
  intro
  ext  <;>  simp
  neg_add_cancel  :=  by
  intro
  ext  <;>  simp
  add_comm  :=  by
  intros
  ext  <;>  simp  <;>  ring
  mul_assoc  :=  by
  intros
  ext  <;>  simp  <;>  ring
  one_mul  :=  by
  intro
  ext  <;>  simp
  mul_one  :=  by
  intro
  ext  <;>  simp
  left_distrib  :=  by
  intros
  ext  <;>  simp  <;>  ring
  right_distrib  :=  by
  intros
  ext  <;>  simp  <;>  ring
  mul_comm  :=  by
  intros
  ext  <;>  simp  <;>  ring
  zero_mul  :=  by
  intros
  ext  <;>  simp
  mul_zero  :=  by
  intros
  ext  <;>  simp 
```

Lean 的库定义 *非平凡* 类型为至少有两个不同元素的类型。在环的上下文中，这相当于说零不等于一。由于一些常见的定理依赖于这个事实，我们不妨现在就建立它。

```py
instance  :  Nontrivial  GaussInt  :=  by
  use  0,  1
  rw  [Ne,  GaussInt.ext_iff]
  simp 
```

现在我们将展示高斯整数具有一个重要的附加属性。一个 *欧几里得域* 是一个带有 *范数* 函数 $N : R \to \mathbb{N}$ 的环 $R$，它具有以下两个属性：

+   对于 $R$ 中的每个 $a$ 和 $b \ne 0$，存在 $q$ 和 $r$ 在 $R$ 中，使得 $a = bq + r$，并且要么 $r = 0$，要么 $N(r) < N(b)$。

+   对于每个 $a$ 和 $b \ne 0$，$N(a) \le N(ab)$。

整数环 $\Bbb{Z}$ 中，$N(a) = |a|$ 是欧几里得域的一个典型例子。在这种情况下，我们可以取 $q$ 为 $a$ 除以 $b$ 的整数除法结果，$r$ 为余数。这些函数在 Lean 中定义，以便它们满足以下条件：

```py
example  (a  b  :  ℤ)  :  a  =  b  *  (a  /  b)  +  a  %  b  :=
  Eq.symm  (Int.ediv_add_emod  a  b)

example  (a  b  :  ℤ)  :  b  ≠  0  →  0  ≤  a  %  b  :=
  Int.emod_nonneg  a

example  (a  b  :  ℤ)  :  b  ≠  0  →  a  %  b  <  |b|  :=
  Int.emod_lt_abs  a 
```

在一个任意的环中，一个元素 $a$ 被称为 *单位*，如果它可以整除 $1$。一个非零元素 $a$ 被称为 *不可约*，如果它不能写成 $a = bc$ 的形式，其中 $b$ 和 $c$ 都不是单位。在整数中，每个不可约元素 $a$ 都是 *素数*，也就是说，每当 $a$ 整除一个乘积 $bc$ 时，它要么整除 $b$，要么整除 $c$。但在其他环中，这个属性可能不成立。在环 $\Bbb{Z}[\sqrt{-5}]$ 中，我们有

$$6 = 2 \cdot 3 = (1 + \sqrt{-5})(1 - \sqrt{-5}),$$

元素 $2$、$3$、$1 + \sqrt{-5}$ 和 $1 - \sqrt{-5}$ 都是不可约的，但它们不是素数。例如，$2$ 可以整除乘积 $(1 + \sqrt{-5})(1 - \sqrt{-5})$，但它不能整除任何一个因子。特别是，我们不再有唯一的分解：数字 $6$ 可以以多种方式分解成不可约元素。

相比之下，每个欧几里得域都是一个唯一分解域，这意味着每个不可约元素都是素数。欧几里得域的公理意味着可以写出任何非零元素为不可约元素的有限乘积。它们还意味着可以使用欧几里得算法找到任何两个非零元素 `a` 和 `b` 的最大公约数，即任何其他公约数都能整除的元素。这反过来又意味着不可约元素的分解是唯一的，直到乘以单位元素。

我们现在证明高斯整数是一个具有由 $N(a + bi) = (a + bi)(a - bi) = a² + b²$ 定义的范数的欧几里得域。高斯整数 $a - bi$ 被称为 $a + bi$ 的**共轭**。对于任何复数 $x$ 和 $y$，检查并不困难，我们有 $N(xy) = N(x)N(y)$。

为了看到这个范数的定义使得高斯整数成为一个欧几里得域，只需考虑第一个性质就具有挑战性。假设我们想要将 $a + bi = (c + di) q + r$ 写成合适的 $q$ 和 $r$。将 $a + bi$ 和 $c + di$ 作为复数来处理，进行除法运算

$$\frac{a + bi}{c + di} = \frac{(a + bi)(c - di)}{(c + di)(c-di)} = \frac{ac + bd}{c² + d²} + \frac{bc -ad}{c²+d²} i.$$

实部和虚部可能不是整数，但我们可以将它们四舍五入到最近的整数 $u$ 和 $v$。然后我们可以将右侧表示为 $(u + vi) + (u' + v'i)$，其中 $u' + v'i$ 是剩余的部分。注意，我们有 $|u'| \le 1/2$ 和 $|v'| \le 1/2$，因此

$$N(u' + v' i) = (u')² + (v')² \le 1/4 + 1/4 \le 1/2.$$

乘以 $c + di$，我们得到

$$a + bi = (c + di) (u + vi) + (c + di) (u' + v'i).$$

设 $q = u + vi$ 和 $r = (c + di) (u' + v'i)$，则 $a + bi = (c + di) q + r$，我们只需要界定 $N(r)$：

$$N(r) = N(c + di)N(u' + v'i) \le N(c + di) \cdot 1/2 < N(c + di).$$

我们刚才进行的论证需要将高斯整数视为复数的一个子集。因此，在 Lean 中形式化它的一个选项是将高斯整数嵌入到复数中，将整数嵌入到高斯整数中，定义从实数到整数的舍入函数，并且要非常小心地在这些数系之间适当地来回传递。实际上，这正是 Mathlib 中所采用的方法，其中高斯整数本身被构造为**二次整数环**的一个特例。参见文件 [GaussianInt.lean](https://github.com/leanprover-community/mathlib4/blob/master/Mathlib/NumberTheory/Zsqrtd/GaussianInt.lean)。

在这里，我们将进行一个保持在整数范围内的论证。这说明了在形式化数学时人们通常面临的选择。给定一个需要或使用库中尚未存在的概念或工具的论证，你有两个选择：要么形式化所需的概念和工具，要么调整论证以利用你已有的概念和工具。当结果可以在其他上下文中使用时，第一个选择通常是时间的好投资。然而，从实用主义的角度来看，有时寻找更基础的证明可能更有效。

整数的常规商余定理表明，对于每一个$a$和非零的$b$，存在$q$和$r$使得$a = b q + r$且$0 \le r < b$。在这里，我们将利用以下变体，它表明存在$q'$和$r'$使得$a = b q' + r'$且$|r'| \le b/2$。你可以检查，如果第一个陈述中的$r$的值满足$r \le b/2$，我们可以取$q' = q$和$r' = r$，否则我们可以取$q' = q + 1$和$r' = r - b$。我们感谢 Heather Macbeth 提出了以下更优雅的方法，它避免了分情况定义。我们只是在除法之前将`b / 2`加到`a`上，然后从余数中减去。

```py
def  div'  (a  b  :  ℤ)  :=
  (a  +  b  /  2)  /  b

def  mod'  (a  b  :  ℤ)  :=
  (a  +  b  /  2)  %  b  -  b  /  2

theorem  div'_add_mod'  (a  b  :  ℤ)  :  b  *  div'  a  b  +  mod'  a  b  =  a  :=  by
  rw  [div',  mod']
  linarith  [Int.ediv_add_emod  (a  +  b  /  2)  b]

theorem  abs_mod'_le  (a  b  :  ℤ)  (h  :  0  <  b)  :  |mod'  a  b|  ≤  b  /  2  :=  by
  rw  [mod',  abs_le]
  constructor
  ·  linarith  [Int.emod_nonneg  (a  +  b  /  2)  h.ne']
  have  :=  Int.emod_lt_of_pos  (a  +  b  /  2)  h
  have  :=  Int.ediv_add_emod  b  2
  have  :=  Int.emod_lt_of_pos  b  zero_lt_two
  linarith 
```

注意到我们使用了老朋友`linarith`。我们还需要将`mod'`用`div'`来表示。

```py
theorem  mod'_eq  (a  b  :  ℤ)  :  mod'  a  b  =  a  -  b  *  div'  a  b  :=  by  linarith  [div'_add_mod'  a  b] 
```

我们将使用$x² + y²$等于零当且仅当$x$和$y$都为零的事实。作为一个练习，我们要求你证明这在任何有序环中都成立。

```py
theorem  sq_add_sq_eq_zero  {α  :  Type*}  [Ring  α]  [LinearOrder  α]  [IsStrictOrderedRing  α]
  (x  y  :  α)  :  x  ^  2  +  y  ^  2  =  0  ↔  x  =  0  ∧  y  =  0  :=  by
  sorry 
```

我们将在这个部分的剩余定义和定理中放入`GaussInt`命名空间。首先，我们定义`norm`函数并要求你证明其一些性质。证明都很简短。

```py
def  norm  (x  :  GaussInt)  :=
  x.re  ^  2  +  x.im  ^  2

@[simp]
theorem  norm_nonneg  (x  :  GaussInt)  :  0  ≤  norm  x  :=  by
  sorry
theorem  norm_eq_zero  (x  :  GaussInt)  :  norm  x  =  0  ↔  x  =  0  :=  by
  sorry
theorem  norm_pos  (x  :  GaussInt)  :  0  <  norm  x  ↔  x  ≠  0  :=  by
  sorry
theorem  norm_mul  (x  y  :  GaussInt)  :  norm  (x  *  y)  =  norm  x  *  norm  y  :=  by
  sorry 
```

接下来，我们定义共轭函数：

```py
def  conj  (x  :  GaussInt)  :  GaussInt  :=
  ⟨x.re,  -x.im⟩

@[simp]
theorem  conj_re  (x  :  GaussInt)  :  (conj  x).re  =  x.re  :=
  rfl

@[simp]
theorem  conj_im  (x  :  GaussInt)  :  (conj  x).im  =  -x.im  :=
  rfl

theorem  norm_conj  (x  :  GaussInt)  :  norm  (conj  x)  =  norm  x  :=  by  simp  [norm] 
```

最后，我们用`x / y`的符号定义高斯整数的除法，它将复数商四舍五入到最接近的高斯整数。我们使用定制的`Int.div'`来完成这个目的。正如我们上面计算的，如果`x`是$a + bi$且`y`是$c + di$，那么`x / y`的实部和虚部是接近于

$$\frac{ac + bd}{c² + d²} \quad \text{和} \quad \frac{bc -ad}{c²+d²},$$

分别。在这里，分子是$(a + bi) (c - di)$的实部和虚部，分母都是$c + di$的范数。

```py
instance  :  Div  GaussInt  :=
  ⟨fun  x  y  ↦  ⟨Int.div'  (x  *  conj  y).re  (norm  y),  Int.div'  (x  *  conj  y).im  (norm  y)⟩⟩ 
```

在定义了`x / y`之后，我们定义`x % y`为余数，即`x - (x / y) * y`。与上面类似，我们在定理`div_def`和`mod_def`中记录这些定义，以便我们可以使用它们进行`simp`和`rw`。

```py
instance  :  Mod  GaussInt  :=
  ⟨fun  x  y  ↦  x  -  y  *  (x  /  y)⟩

theorem  div_def  (x  y  :  GaussInt)  :
  x  /  y  =  ⟨Int.div'  (x  *  conj  y).re  (norm  y),  Int.div'  (x  *  conj  y).im  (norm  y)⟩  :=
  rfl

theorem  mod_def  (x  y  :  GaussInt)  :  x  %  y  =  x  -  y  *  (x  /  y)  :=
  rfl 
```

这些定义立即给出了对于每个`x`和`y`的`x = y * (x / y) + x % y`，所以我们只需要证明当`y`不为零时，`x % y`的范数小于`y`的范数。

我们刚刚定义了`x / y`的实部和虚部分别为`div' (x * conj y).re (norm y)`和`div' (x * conj y).im (norm y)`。计算后，我们得到

> `(x % y) * conj y = (x - x / y * y) * conj y = x * conj y - x / y * (y * conj y)`

右侧的实部和虚部正好是 `mod' (x * conj y).re (norm y)` 和 `mod' (x * conj y).im (norm y)`。根据 `div'` 和 `mod'` 的性质，这些值保证小于或等于 `norm y / 2`。因此我们有

> `norm ((x % y) * conj y) ≤ (norm y / 2)² + (norm y / 2)² ≤ (norm y / 2) * norm y`.

另一方面，我们有

> `norm ((x % y) * conj y) = norm (x % y) * norm (conj y) = norm (x % y) * norm y`.

通过除以 `norm y`，我们得到 `norm (x % y) ≤ (norm y) / 2 < norm y`，正如所需的那样。

这个混乱的计算将在下一个证明中进行。我们鼓励您逐步查看细节，看看您是否能找到一个更合理的论点。

```py
theorem  norm_mod_lt  (x  :  GaussInt)  {y  :  GaussInt}  (hy  :  y  ≠  0)  :
  (x  %  y).norm  <  y.norm  :=  by
  have  norm_y_pos  :  0  <  norm  y  :=  by  rwa  [norm_pos]
  have  H1  :  x  %  y  *  conj  y  =  ⟨Int.mod'  (x  *  conj  y).re  (norm  y),  Int.mod'  (x  *  conj  y).im  (norm  y)⟩
  ·  ext  <;>  simp  [Int.mod'_eq,  mod_def,  div_def,  norm]  <;>  ring
  have  H2  :  norm  (x  %  y)  *  norm  y  ≤  norm  y  /  2  *  norm  y
  ·  calc
  norm  (x  %  y)  *  norm  y  =  norm  (x  %  y  *  conj  y)  :=  by  simp  only  [norm_mul,  norm_conj]
  _  =  |Int.mod'  (x.re  *  y.re  +  x.im  *  y.im)  (norm  y)|  ^  2
  +  |Int.mod'  (-(x.re  *  y.im)  +  x.im  *  y.re)  (norm  y)|  ^  2  :=  by  simp  [H1,  norm,  sq_abs]
  _  ≤  (y.norm  /  2)  ^  2  +  (y.norm  /  2)  ^  2  :=  by  gcongr  <;>  apply  Int.abs_mod'_le  _  _  norm_y_pos
  _  =  norm  y  /  2  *  (norm  y  /  2  *  2)  :=  by  ring
  _  ≤  norm  y  /  2  *  norm  y  :=  by  gcongr;  apply  Int.ediv_mul_le;  norm_num
  calc  norm  (x  %  y)  ≤  norm  y  /  2  :=  le_of_mul_le_mul_right  H2  norm_y_pos
  _  <  norm  y  :=  by
  apply  Int.ediv_lt_of_lt_mul
  ·  norm_num
  ·  linarith 
```

我们已经接近终点。我们的 `norm` 函数将高斯整数映射到非负整数。我们需要一个将高斯整数映射到自然数的函数，我们通过将 `norm` 与函数 `Int.natAbs` 组合来获得，该函数将整数映射到自然数。下两个引理中的第一个建立了将范数映射到自然数再映射回整数不会改变值。第二个重新表述了范数是递减的事实。

```py
theorem  coe_natAbs_norm  (x  :  GaussInt)  :  (x.norm.natAbs  :  ℤ)  =  x.norm  :=
  Int.natAbs_of_nonneg  (norm_nonneg  _)

theorem  natAbs_norm_mod_lt  (x  y  :  GaussInt)  (hy  :  y  ≠  0)  :
  (x  %  y).norm.natAbs  <  y.norm.natAbs  :=  by
  apply  Int.ofNat_lt.1
  simp  only  [Int.natCast_natAbs,  abs_of_nonneg,  norm_nonneg]
  exact  norm_mod_lt  x  hy 
```

我们还需要建立欧几里得域上范数函数的第二个关键性质。

```py
theorem  not_norm_mul_left_lt_norm  (x  :  GaussInt)  {y  :  GaussInt}  (hy  :  y  ≠  0)  :
  ¬(norm  (x  *  y)).natAbs  <  (norm  x).natAbs  :=  by
  apply  not_lt_of_ge
  rw  [norm_mul,  Int.natAbs_mul]
  apply  le_mul_of_one_le_right  (Nat.zero_le  _)
  apply  Int.ofNat_le.1
  rw  [coe_natAbs_norm]
  exact  Int.add_one_le_of_lt  ((norm_pos  _).mpr  hy) 
```

现在我们可以将其组合起来，以表明高斯整数是欧几里得域的一个实例。我们使用我们定义的商和余数函数。Mathlib 对欧几里得域的定义比上面的定义更通用，因为它允许我们证明余数与任何良基测度相关时是递减的。比较返回自然数的范数函数的值只是这种测度的一个实例，在这种情况下，所需性质是定理 `natAbs_norm_mod_lt` 和 `not_norm_mul_left_lt_norm`。

```py
instance  :  EuclideanDomain  GaussInt  :=
  {  GaussInt.instCommRing  with
  quotient  :=  (·  /  ·)
  remainder  :=  (·  %  ·)
  quotient_mul_add_remainder_eq  :=
  fun  x  y  ↦  by  rw  [mod_def,  add_comm]  ;  ring
  quotient_zero  :=  fun  x  ↦  by
  simp  [div_def,  norm,  Int.div']
  rfl
  r  :=  (measure  (Int.natAbs  ∘  norm)).1
  r_wellFounded  :=  (measure  (Int.natAbs  ∘  norm)).2
  remainder_lt  :=  natAbs_norm_mod_lt
  mul_left_not_lt  :=  not_norm_mul_left_lt_norm  } 
```

一个直接的好处是，我们现在知道在高斯整数中，素数和不可约的概念是一致的。

```py
example  (x  :  GaussInt)  :  Irreducible  x  ↔  Prime  x  :=
  irreducible_iff_prime 
```*
