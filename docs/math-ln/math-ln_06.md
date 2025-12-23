# 6. 离散数学

> [`leanprover-community.github.io/mathematics_in_lean/C06_Discrete_Mathematics.html`](https://leanprover-community.github.io/mathematics_in_lean/C06_Discrete_Mathematics.html)

*Lean 中的数学* **   6. 离散数学

+   查看页面源代码

* * *

*离散数学*是研究有限集合、对象和结构的学科。我们可以计算有限集合的元素，我们可以对其元素计算有限和或积，我们可以计算最大值和最小值，等等。我们还可以研究由有限次应用某些生成函数生成的对象，我们可以通过结构递归定义函数，并通过结构归纳证明定理。本章描述了 Mathlib 支持这些活动的部分。

## 6.1. Finsets 和 Fintypes

在 Mathlib 中处理有限集合和类型可能会令人困惑，因为该库提供了多种处理它们的方式。在本节中，我们将讨论其中最常见的一些。

我们已经在第 5.2 节和第 5.3 节中遇到了类型`Finset`。正如其名所示，类型`Finset α`的元素是类型`α`的有限集合。我们将称这些为“finsets”。`Finset`数据类型被设计成具有计算解释，并且许多对`Finset α`的基本操作都假设`α`具有可判定的等价性，这保证了存在一个算法来测试`a : α`是否是 finset `s`的元素。

```py
section
variable  {α  :  Type*}  [DecidableEq  α]  (a  :  α)  (s  t  :  Finset  α)

#check  a  ∈  s
#check  s  ∩  t

end 
```

如果你移除了声明`[DecidableEq α]`，Lean 将在`#check s ∩ t`这一行上抱怨，因为它无法计算交集。然而，你应该期望能够计算的所有数据类型都具有可判定的等价性，如果你通过打开`Classical`命名空间并声明`noncomputable section`进行经典工作，你就可以对任何类型的元素进行 finsets 的推理。

Finsets 支持集合所支持的大多数集合论操作：

```py
open  Finset

variable  (a  b  c  :  Finset  ℕ)
variable  (n  :  ℕ)

#check  a  ∩  b
#check  a  ∪  b
#check  a  \  b
#check  (∅  :  Finset  ℕ)

example  :  a  ∩  (b  ∪  c)  =  (a  ∩  b)  ∪  (a  ∩  c)  :=  by
  ext  x;  simp  only  [mem_inter,  mem_union];  tauto

example  :  a  ∩  (b  ∪  c)  =  (a  ∩  b)  ∪  (a  ∩  c)  :=  by  rw  [inter_union_distrib_left] 
```

注意，我们已经打开了`Finset`命名空间，其中包含特定于 finsets 的定理。如果你浏览下面的最后一个例子，你会看到应用`ext`然后是`simp`将恒等式简化为一个命题逻辑问题。作为一个练习，你可以尝试证明一些来自第四章的集合恒等式，并将其转移到 finsets 中。

你已经看到了表示自然数有限集合$\{ 0, 1, \ldots, n-1 \}$的符号`Finset.range n`。`Finset`还允许你通过枚举元素来定义有限集合：

```py
#check  ({0,  2,  5}  :  Finset  Nat)

def  example1  :  Finset  ℕ  :=  {0,  1,  2} 
```

有多种方法可以让 Lean 认识到，在这种方式呈现的集合中，元素的顺序和重复项并不重要。

```py
example  :  ({0,  1,  2}  :  Finset  ℕ)  =  {1,  2,  0}  :=  by  decide

example  :  ({0,  1,  2}  :  Finset  ℕ)  =  {0,  1,  1,  2}  :=  by  decide

example  :  ({0,  1}  :  Finset  ℕ)  =  {1,  0}  :=  by  rw  [Finset.pair_comm]

example  (x  :  Nat)  :  ({x,  x}  :  Finset  ℕ)  =  {x}  :=  by  simp

example  (x  y  z  :  Nat)  :  ({x,  y,  z,  y,  z,  x}  :  Finset  ℕ)  =  {x,  y,  z}  :=  by
  ext  i;  simp  [or_comm,  or_assoc,  or_left_comm]

example  (x  y  z  :  Nat)  :  ({x,  y,  z,  y,  z,  x}  :  Finset  ℕ)  =  {x,  y,  z}  :=  by
  ext  i;  simp;  tauto 
```

你可以使用 `insert` 向 Finset 添加单个元素，并使用 `Finset.erase` 删除单个元素。请注意，`erase` 在 `Finset` 命名空间中，但 `insert` 在根命名空间中。

```py
example  (s  :  Finset  ℕ)  (a  :  ℕ)  (h  :  a  ∉  s)  :  (insert  a  s  |>.erase  a)  =  s  :=
  Finset.erase_insert  h

example  (s  :  Finset  ℕ)  (a  :  ℕ)  (h  :  a  ∈  s)  :  insert  a  (s.erase  a)  =  s  :=
  Finset.insert_erase  h 
```

事实上，`{0, 1, 2}` 只是 `insert 0 (insert 1 (singleton 2))` 的符号。

```py
set_option  pp.notation  false  in
#check  ({0,  1,  2}  :  Finset  ℕ) 
```

给定一个 finset `s` 和一个谓词 `P`，我们可以使用集合构造器符号 `{x ∈ s | P x}` 来定义满足 `P` 的 `s` 的元素集合。这是 `Finset.filter P s` 的符号，也可以写成 `s.filter P`。

```py
example  :  {m  ∈  range  n  |  Even  m}  =  (range  n).filter  Even  :=  rfl
example  :  {m  ∈  range  n  |  Even  m  ∧  m  ≠  3}  =  (range  n).filter  (fun  m  ↦  Even  m  ∧  m  ≠  3)  :=  rfl

example  :  {m  ∈  range  10  |  Even  m}  =  {0,  2,  4,  6,  8}  :=  by  decide 
```

Mathlib 知道在函数下的 finset 的像是一个 finset。

```py
#check  (range  5).image  (fun  x  ↦  x  *  2)

example  :  (range  5).image  (fun  x  ↦  x  *  2)  =  {x  ∈  range  10  |  Even  x}  :=  by  decide 
```

Lean 也知道两个 finset 的笛卡尔积 `s ×ˢ t` 是一个 finset，以及一个 finset 的幂集也是一个 finset。（注意，符号 `s ×ˢ t` 也可以用于集合。）

```py
#check  s  ×ˢ  t
#check  s.powerset 
```

在 finset 的元素上定义操作是棘手的，因为任何这样的定义都必须独立于元素呈现的顺序。当然，你可以通过组合现有操作来定义函数。你还可以使用 `Finset.fold` 来对元素上的二元操作进行折叠，前提是操作是结合的和交换的，因为这些性质保证了结果与操作应用的顺序无关。有限和、积和并集就是这样定义的。在下面的最后一个例子中，`biUnion` 代表“有界索引并”。用传统的数学符号，这个表达式将被写成 $\bigcup_{i ∈ s} g(i)$。

```py
#check  Finset.fold

def  f  (n  :  ℕ)  :  Int  :=  (↑n)²

#check  (range  5).fold  (fun  x  y  :  Int  ↦  x  +  y)  0  f
#eval  (range  5).fold  (fun  x  y  :  Int  ↦  x  +  y)  0  f

#check  ∑  i  ∈  range  5,  i²
#check  ∏  i  ∈  range  5,  i  +  1

variable  (g  :  Nat  →  Finset  Int)

#check  (range  5).biUnion  g 
```

对于 finset 有一个自然的归纳原则：要证明每个 finset 都有一个属性，只需证明空集具有该属性，并且当我们向 finset 添加一个新元素时，该属性得到保持。（在下一个例子的归纳步骤中，`@` 符号在 `@insert` 中是必需的，因为它为参数 `a` 和 `s` 提供了名称，因为它们已被标记为隐式。）

```py
#check  Finset.induction

example  {α  :  Type*}  [DecidableEq  α]  (f  :  α  →  ℕ)  (s  :  Finset  α)  (h  :  ∀  x  ∈  s,  f  x  ≠  0)  :
  ∏  x  ∈  s,  f  x  ≠  0  :=  by
  induction  s  using  Finset.induction_on  with
  |  empty  =>  simp
  |  @insert  a  s  anins  ih  =>
  rw  [prod_insert  anins]
  apply  mul_ne_zero
  ·  apply  h;  apply  mem_insert_self
  apply  ih
  intros  x  xs
  exact  h  x  (mem_insert_of_mem  xs) 
```

如果 `s` 是一个 finset，`Finset.Nonempty s` 被定义为 `∃ x, x ∈ s`。你可以使用经典选择来选择非空 finset 的一个元素。同样，库定义了 `Finset.toList s`，它使用选择以某种顺序选择 `s` 的元素。

```py
noncomputable  example  (s  :  Finset  ℕ)  (h  :  s.Nonempty)  :  ℕ  :=  Classical.choose  h

example  (s  :  Finset  ℕ)  (h  :  s.Nonempty)  :  Classical.choose  h  ∈  s  :=  Classical.choose_spec  h

noncomputable  example  (s  :  Finset  ℕ)  :  List  ℕ  :=  s.toList

example  (s  :  Finset  ℕ)  (a  :  ℕ)  :  a  ∈  s.toList  ↔  a  ∈  s  :=  mem_toList 
```

你可以使用 `Finset.min` 和 `Finset.max` 来选择线性顺序元素的 finset 的最小或最大元素，同样你也可以使用 `Finset.inf` 和 `Finset.sup` 与格元素的 finset 一起使用，但有一个问题。空 finset 的最小元素应该是什么？你可以检查下面函数的带撇版本添加了一个先决条件，即 finset 不能为空。不带撇版本的 `Finset.min` 和 `Finset.max` 分别向输出类型添加一个顶或底元素，以处理 finset 为空的情况。不带撇版本的 `Finset.inf` 和 `Finset.sup` 假设格已经配备了顶或底元素。

```py
#check  Finset.min
#check  Finset.min'
#check  Finset.max
#check  Finset.max'
#check  Finset.inf
#check  Finset.inf'
#check  Finset.sup
#check  Finset.sup'

example  :  Finset.Nonempty  {2,  6,  7}  :=  ⟨6,  by  trivial⟩
example  :  Finset.min'  {2,  6,  7}  ⟨6,  by  trivial⟩  =  2  :=  by  trivial 
```

每个 finset `s` 都有一个有限的基数，`Finset.card s`，当 `Finset` 命名空间打开时可以写成 `#s`。

```py
#check  Finset.card

#eval  (range  5).card

example  (s  :  Finset  ℕ)  :  s.card  =  #s  :=  by  rfl

example  (s  :  Finset  ℕ)  :  s.card  =  ∑  i  ∈  s,  1  :=  by  rw  [card_eq_sum_ones]

example  (s  :  Finset  ℕ)  :  s.card  =  ∑  i  ∈  s,  1  :=  by  simp 
```

下一节将全部关于推理基数。

在形式化数学时，人们常常需要决定是否用集合或类型来表述自己的定义和定理。使用类型通常可以简化符号和证明，但处理类型的子集可能更加灵活。finset 的类型对应物是 *fintype*，即某个 `α` 的类型 `Fintype α`。根据定义，fintype 只是一个带有包含所有其元素的 finset `univ` 的数据类型。

```py
variable  {α  :  Type*}  [Fintype  α]

example  :  ∀  x  :  α,  x  ∈  Finset.univ  :=  by
  intro  x;  exact  mem_univ  x 
```

`Fintype.card α` 等于相应 finset 的基数。

```py
example  :  Fintype.card  α  =  (Finset.univ  :  Finset  α).card  :=  rfl 
```

我们已经看到了一个 fintype 的典型例子，即每个 `n` 的类型 `Fin n`。Lean 认识到 fintype 在乘积运算等操作下是封闭的。

```py
example  :  Fintype.card  (Fin  5)  =  5  :=  by  simp
example  :  Fintype.card  ((Fin  5)  ×  (Fin  3))  =  15  :=  by  simp 
```

`Finset α` 的任何元素 `s` 都可以强制转换为类型 `(↑s : Finset α)`，即包含在 `s` 中的 `α` 的元素的子类型。

```py
variable  (s  :  Finset  ℕ)

example  :  (↑s  :  Type)  =  {x  :  ℕ  //  x  ∈  s}  :=  rfl
example  :  Fintype.card  ↑s  =  s.card  :=  by  simp 
```

Lean 和 Mathlib 使用 *类型类推理* 来追踪 fintype 上的额外结构，即包含所有元素的通用 finset。换句话说，你可以将 fintype 视为一个带有额外数据的代数结构。第七章 解释了这是如何工作的。  ## 6.2\. 计数论证

计数事物的艺术是组合数学的一个核心部分。Mathlib 包含了几个用于计数 finset 元素的基本恒等式：

```py
open  Finset

variable  {α  β  :  Type*}  [DecidableEq  α]  [DecidableEq  β]  (s  t  :  Finset  α)  (f  :  α  →  β)

example  :  #(s  ×ˢ  t)  =  #s  *  #t  :=  by  rw  [card_product]
example  :  #(s  ×ˢ  t)  =  #s  *  #t  :=  by  simp

example  :  #(s  ∪  t)  =  #s  +  #t  -  #(s  ∩  t)  :=  by  rw  [card_union]

example  (h  :  Disjoint  s  t)  :  #(s  ∪  t)  =  #s  +  #t  :=  by  rw  [card_union_of_disjoint  h]
example  (h  :  Disjoint  s  t)  :  #(s  ∪  t)  =  #s  +  #t  :=  by  simp  [h]

example  (h  :  Function.Injective  f)  :  #(s.image  f)  =  #s  :=  by  rw  [card_image_of_injective  _  h]

example  (h  :  Set.InjOn  f  s)  :  #(s.image  f)  =  #s  :=  by  rw  [card_image_of_injOn  h] 
```

打开 `Finset` 命名空间允许我们使用 `#s` 表示 `s.card` 的符号，以及使用缩短的名称 card_union 等等。

Mathlib 也可以计数 fintype 的元素：

```py
open  Fintype

variable  {α  β  :  Type*}  [Fintype  α]  [Fintype  β]

example  :  card  (α  ×  β)  =  card  α  *  card  β  :=  by  simp

example  :  card  (α  ⊕  β)  =  card  α  +  card  β  :=  by  simp

example  (n  :  ℕ)  :  card  (Fin  n  →  α)  =  (card  α)^n  :=  by  simp

variable  {n  :  ℕ}  {γ  :  Fin  n  →  Type*}  [∀  i,  Fintype  (γ  i)]

example  :  card  ((i  :  Fin  n)  →  γ  i)  =  ∏  i,  card  (γ  i)  :=  by  simp

example  :  card  (Σ  i,  γ  i)  =  ∑  i,  card  (γ  i)  :=  by  simp 
```

当 `Fintype` 命名空间未开放时，我们必须使用 `Fintype.card` 而不是 card。

以下是一个计算 finset 基数的例子，即范围 n 与移位超过 n 的范围 n 的副本的并集。这个计算需要证明联合中的两个集合是互斥的；证明的第一行产生了侧条件 `Disjoint (range n) (image (fun i ↦ m + i) (range n))`，该条件在证明的末尾得到证实。`Disjoint` 谓词过于通用，直接对我们没有直接用处，但定理 `disjoint_iff_ne` 将其置于我们可以使用的形式。

```py
#check  Disjoint

example  (m  n  :  ℕ)  (h  :  m  ≥  n)  :
  card  (range  n  ∪  (range  n).image  (fun  i  ↦  m  +  i))  =  2  *  n  :=  by
  rw  [card_union_of_disjoint,  card_range,  card_image_of_injective,  card_range];  omega
  .  apply  add_right_injective
  .  simp  [disjoint_iff_ne];  omega 
```

在本节中，`omega` 将是我们处理算术计算和不等式的一个主要工具。

这里有一个更有趣的例子。考虑由 $\{0, \ldots, n\} \times \{0, \ldots, n\}$ 的子集组成的对 $(i, j)$，其中 $i < j$。如果你将它们视为坐标平面上的格点，它们构成了一个以 $(0, 0)$ 和 $(n, n)$ 为顶点的上三角形，不包括对角线。整个正方形的基数是 $(n + 1)²$，减去对角线的长度并将结果除以二，我们得到三角形的基数是 $n (n + 1) / 2$。

或者，我们注意到三角形的行的大小为 $0, 1, \ldots, n$，因此基数是前 $n$ 个自然数的和。下面证明的第一部分将三角形描述为行的并集，其中行 $j$ 由数字 $0, 1, ..., j - 1$ 与 $j$ 配对组成。在下面的证明中，符号 `(., j)` 简化了函数 `fun i ↦ (i, j)`。其余的证明只是对有限集基数进行计算。

```py
def  triangle  (n  :  ℕ)  :  Finset  (ℕ  ×  ℕ)  :=  {p  ∈  range  (n+1)  ×ˢ  range  (n+1)  |  p.1  <  p.2}

example  (n  :  ℕ)  :  #(triangle  n)  =  (n  +  1)  *  n  /  2  :=  by
  have  :  triangle  n  =  (range  (n+1)).biUnion  (fun  j  ↦  (range  j).image  (.,  j))  :=  by
  ext  p
  simp  only  [triangle,  mem_filter,  mem_product,  mem_range,  mem_biUnion,  mem_image]
  constructor
  .  rintro  ⟨⟨hp1,  hp2⟩,  hp3⟩
  use  p.2,  hp2,  p.1,  hp3
  .  rintro  ⟨p1,  hp1,  p2,  hp2,  rfl⟩
  omega
  rw  [this,  card_biUnion];  swap
  ·  -- take care of disjointness first
  intro  x  _  y  _  xney
  simp  [disjoint_iff_ne,  xney]
  -- continue the calculation
  transitivity  (∑  i  ∈  range  (n  +  1),  i)
  ·  congr;  ext  i
  rw  [card_image_of_injective,  card_range]
  intros  i1  i2;  simp
  rw  [sum_range_id];  rfl 
```

以下是对证明的变体，它使用有限类型而不是有限集进行计算。类型 `α ≃ β` 是 `α` 和 `β` 之间等价关系的类型，由正向映射、反向映射以及证明这两个映射是彼此的逆映射组成。证明中的第一个 `have` 显示 `triangle n` 与 `Fin i` 的不相交并等价，其中 `i` 在 `Fin (n + 1)` 上取值。有趣的是，正向函数和反向函数是用策略构建的，而不是明确写出。由于它们所做的只是移动数据和信息，`rfl` 证明了它们是逆映射。

之后，`rw [←Fintype.card_coe]` 将 `#(triangle n)` 重写为子类型 `{ x // x ∈ triangle n }` 的基数，其余的证明都是计算。

```py
example  (n  :  ℕ)  :  #(triangle  n)  =  (n  +  1)  *  n  /  2  :=  by
  have  :  triangle  n  ≃  Σ  i  :  Fin  (n  +  1),  Fin  i.val  :=
  {  toFun  :=  by
  rintro  ⟨⟨i,  j⟩,  hp⟩
  simp  [triangle]  at  hp
  exact  ⟨⟨j,  hp.1.2⟩,  ⟨i,  hp.2⟩⟩
  invFun  :=  by
  rintro  ⟨i,  j⟩
  use  ⟨j,  i⟩
  simp  [triangle]
  exact  j.isLt.trans  i.isLt
  left_inv  :=  by  intro  i;  rfl
  right_inv  :=  by  intro  i;  rfl  }
  rw  [←Fintype.card_coe]
  trans;  apply  (Fintype.card_congr  this)
  rw  [Fintype.card_sigma,  sum_fin_eq_sum_range]
  convert  Finset.sum_range_id  (n  +  1)
  simp_all 
```

这里是另一种方法。下面证明的第一行将问题简化为证明 `2 * #(triangle n) = (n + 1) * n`。我们可以通过证明三角形的两个副本正好填满矩形 `range n ×ˢ range (n + 1)` 来做到这一点。作为一个练习，看看你是否能填补计算步骤。在解答中，我们在倒数第二步大量使用了 `omega`，但不幸的是，我们不得不手动做相当多的工作。

```py
example  (n  :  ℕ)  :  #(triangle  n)  =  (n  +  1)  *  n  /  2  :=  by
  apply  Nat.eq_div_of_mul_eq_right  (by  norm_num)
  let  turn  (p  :  ℕ  ×  ℕ)  :  ℕ  ×  ℕ  :=  (n  -  1  -  p.1,  n  -  p.2)
  calc  2  *  #(triangle  n)
  =  #(triangle  n)  +  #(triangle  n)  :=  by
  sorry
  _  =  #(triangle  n)  +  #(triangle  n  |>.image  turn)  :=  by
  sorry
  _  =  #(range  n  ×ˢ  range  (n  +  1))  :=  by
  sorry
  _  =  (n  +  1)  *  n  :=  by
  sorry 
```

你可以自己验证，如果我们用 `n + 1` 替换 `n` 并在 `triangle` 的定义中将 `<` 替换为 `≤`，我们会得到相同的三角形，只是向下移动了。下面的练习要求你使用这个事实来证明两个三角形具有相同的大小。

```py
def  triangle'  (n  :  ℕ)  :  Finset  (ℕ  ×  ℕ)  :=  {p  ∈  range  n  ×ˢ  range  n  |  p.1  ≤  p.2}

example  (n  :  ℕ)  :  #(triangle'  n)  =  #(triangle  n)  :=  by  sorry 
```

让我们以一个例子和一个练习来结束这一节，这个例子和练习来自 Bhavik Mehta 在 2023 年于 *Lean for the Curious Mathematician* 上给出的组合学教程 [教程](https://www.youtube.com/watch?v=_cJctOIXWE4&list=PLlF-CfQhukNn7xEbfL38eLgkveyk9_myQ&index=8&t=2737s&ab_channel=leanprovercommunity)。假设我们有一个二分图，其顶点集为 `s` 和 `t`，对于 `s` 中的每个 `a`，至少有三条离开 `a` 的边，而对于 `t` 中的每个 `b`，最多有一条进入 `b` 的边。那么图中边的总数至少是 `s` 的基数的三倍，最多是 `t` 的基数，由此可以推出三倍的 `s` 的基数最多是 `t` 的基数。以下定理实现了这个论证，其中我们使用关系 `r` 来表示图的边。证明是一个优雅的计算。

```py
open  Classical
variable  (s  t  :  Finset  Nat)  (a  b  :  Nat)

theorem  doubleCounting  {α  β  :  Type*}  (s  :  Finset  α)  (t  :  Finset  β)
  (r  :  α  →  β  →  Prop)
  (h_left  :  ∀  a  ∈  s,  3  ≤  #{b  ∈  t  |  r  a  b})
  (h_right  :  ∀  b  ∈  t,  #{a  ∈  s  |  r  a  b}  ≤  1)  :
  3  *  #(s)  ≤  #(t)  :=  by
  calc  3  *  #(s)
  =  ∑  a  ∈  s,  3  :=  by  simp  [sum_const_nat,  mul_comm]
  _  ≤  ∑  a  ∈  s,  #({b  ∈  t  |  r  a  b})  :=  sum_le_sum  h_left
  _  =  ∑  a  ∈  s,  ∑  b  ∈  t,  if  r  a  b  then  1  else  0  :=  by  simp
  _  =  ∑  b  ∈  t,  ∑  a  ∈  s,  if  r  a  b  then  1  else  0  :=  sum_comm
  _  =  ∑  b  ∈  t,  #({a  ∈  s  |  r  a  b})  :=  by  simp
  _  ≤  ∑  b  ∈  t,  1  :=  sum_le_sum  h_right
  _  ≤  #(t)  :=  by  simp 
```

以下练习也来自 Mehta 的教程。假设 `A` 是 `range (2 * n)` 的一个包含 `n + 1` 个元素的子集。很容易看出 `A` 必须包含两个连续的整数，因此包含两个互质的元素。如果你观看教程，你会看到大量的努力用于建立以下事实，现在这个事实已经被 `omega` 自动证明。

```py
example  (m  k  :  ℕ)  (h  :  m  ≠  k)  (h'  :  m  /  2  =  k  /  2)  :  m  =  k  +  1  ∨  k  =  m  +  1  :=  by  omega 
```

Mehta 练习的解决方案使用了抽屉原理，形式为 `exists_lt_card_fiber_of_mul_lt_card_of_maps_to`，以表明在 `A` 中存在两个不同的元素 `m` 和 `k`，使得 `m / 2 = k / 2`。看看你是否能完成该事实的证明，然后使用它来完成证明。

```py
example  {n  :  ℕ}  (A  :  Finset  ℕ)
  (hA  :  #(A)  =  n  +  1)
  (hA'  :  A  ⊆  range  (2  *  n))  :
  ∃  m  ∈  A,  ∃  k  ∈  A,  Nat.Coprime  m  k  :=  by
  have  :  ∃  t  ∈  range  n,  1  <  #({u  ∈  A  |  u  /  2  =  t})  :=  by
  apply  exists_lt_card_fiber_of_mul_lt_card_of_maps_to
  ·  sorry
  ·  sorry
  rcases  this  with  ⟨t,  ht,  ht'⟩
  simp  only  [one_lt_card,  mem_filter]  at  ht'
  sorry 
```  ## 6.3\. 归纳定义的类型

Lean 的基础允许我们定义归纳类型，即从底部向上生成实例的数据类型。例如，由空列表 `nil` 开始，并逐个添加元素到列表前端的 `List α` 列表类型。下面我们将定义一个二叉树类型 `BinTree`，其元素从空树开始，通过将新节点附加到两个现有树来构建新树。

在 Lean 中，可以定义对象无限、如可数分支良好基础树的归纳类型。在离散数学中，有限归纳定义通常被使用，尤其是在与计算机科学相关的离散数学分支中。Lean 不仅提供了定义此类类型的手段，还提供了归纳和递归定义的原则。例如，数据类型 `List α` 是通过归纳定义的：

```py
namespace  MyListSpace

inductive  List  (α  :  Type*)  where
  |  nil  :  List  α
  |  cons  :  α  →  List  α  →  List  α

end  MyListSpace 
```

归纳定义说明 `List α` 的每个元素要么是 `nil`，即空列表，要么是 `cons a as`，其中 `a` 是 `α` 的一个元素，`as` 是 `α` 的元素列表。构造函数被适当地命名为 `List.nil` 和 `List.cons`，但你可以使用 `List` 命名空间开放的简短表示。当 `List` 命名空间 *不* 开放时，你可以在 Lean 需要列表的任何地方写 `.nil` 和 `.cons a as`，Lean 将自动插入 `List` 限定符。在本节中，我们将临时定义放在单独的命名空间，如 `MyListSpace`，以避免与标准库冲突。在临时命名空间之外，我们将恢复使用标准库定义。

Lean 定义了 `[]` 表示 `nil` 和 `::` 表示 `cons` 的符号，你可以用 `[a, b, c]` 表示 `a :: b :: c :: []`。append 和 map 函数如下递归定义：

```py
def  append  {α  :  Type*}  :  List  α  →  List  α  →  List  α
  |  [],  bs  =>  bs
  |  a  ::  as,  bs  =>  a  ::  (append  as  bs)

def  map  {α  β  :  Type*}  (f  :  α  →  β)  :  List  α  →  List  β
  |  []  =>  []
  |  a  ::  as  =>  f  a  ::  map  f  as

#eval  append  [1,  2,  3]  [4,  5,  6]
#eval  map  (fun  n  =>  n²)  [1,  2,  3,  4,  5] 
```

注意，存在一个基本情况和一个递归情况。在每种情况下，两个定义子句都定义性地成立：

```py
theorem  nil_append  {α  :  Type*}  (as  :  List  α)  :  append  []  as  =  as  :=  rfl

theorem  cons_append  {α  :  Type*}  (a  :  α)  (as  :  List  α)  (bs  :  List  α)  :
  append  (a  ::  as)  bs  =  a  ::  (append  as  bs)  :=  rfl

theorem  map_nil  {α  β  :  Type*}  (f  :  α  →  β)  :  map  f  []  =  []  :=  rfl

theorem  map_cons  {α  β  :  Type*}  (f  :  α  →  β)  (a  :  α)  (as  :  List  α)  :
  map  f  (a  ::  as)  =  f  a  ::  map  f  as  :=  rfl 
```

函数 `append` 和 `map` 定义在标准库中，`append as bs` 可以写成 `as ++ bs`。

Lean 允许你按照定义的结构通过归纳来编写证明。

```py
variable  {α  β  γ  :  Type*}
variable  (as  bs  cs  :  List  α)
variable  (a  b  c  :  α)

open  List

theorem  append_nil  :  ∀  as  :  List  α,  as  ++  []  =  as
  |  []  =>  rfl
  |  a  ::  as  =>  by  rw  [cons_append,  append_nil  as]

theorem  map_map  (f  :  α  →  β)  (g  :  β  →  γ)  :
  ∀  as  :  List  α,  map  g  (map  f  as)  =  map  (g  ∘  f)  as
  |  []  =>  rfl
  |  a  ::  as  =>  by  rw  [map_cons,  map_cons,  map_cons,  map_map  f  g  as];  rfl 
```

你也可以使用 `induction'` 策略。

当然，这些定理已经在标准库中。作为一个练习，尝试在 `MyListSpace3` 命名空间中定义一个函数 `reverse`（为了避免与标准 `List.reverse` 冲突），该函数反转一个列表。你可以使用 `#eval reverse [1, 2, 3, 4, 5]` 来测试它。`reverse` 的最直接定义需要二次时间，但不用担心这一点。你可以跳转到标准库中 `List.reverse` 的定义，以查看线性时间实现。尝试证明 `reverse (as ++ bs) = reverse bs ++ reverse as` 和 `reverse (reverse as) = as`。你可以使用 `cons_append` 和 `append_assoc`，但你可能需要提出辅助引理并证明它们。

```py
def  reverse  :  List  α  →  List  α  :=  sorry

theorem  reverse_append  (as  bs  :  List  α)  :  reverse  (as  ++  bs)  =  reverse  bs  ++  reverse  as  :=  by
  sorry

theorem  reverse_reverse  (as  :  List  α)  :  reverse  (reverse  as)  =  as  :=  by  sorry 
```

作为另一个例子，考虑以下二叉树的归纳定义以及计算二叉树大小和深度的函数。

```py
inductive  BinTree  where
  |  empty  :  BinTree
  |  node  :  BinTree  →  BinTree  →  BinTree

namespace  BinTree

def  size  :  BinTree  →  ℕ
  |  empty  =>  0
  |  node  l  r  =>  size  l  +  size  r  +  1

def  depth  :  BinTree  →  ℕ
  |  empty  =>  0
  |  node  l  r  =>  max  (depth  l)  (depth  r)  +  1 
```

将空二叉树计为大小为 0 和深度为 0 的二叉树是很方便的。在文献中，这种数据类型有时被称为 *扩展二叉树*。包括空树意味着，例如，我们可以定义由根节点、空左子树和由单个节点组成的右子树构成的树 `node empty (node empty empty)`。

这里是一个重要的不等式，它关联了大小和深度：

```py
theorem  size_le  :  ∀  t  :  BinTree,  size  t  ≤  2^depth  t  -  1
  |  empty  =>  Nat.zero_le  _
  |  node  l  r  =>  by
  simp  only  [depth,  size]
  calc  l.size  +  r.size  +  1
  ≤  (2^l.depth  -  1)  +  (2^r.depth  -  1)  +  1  :=  by
  gcongr  <;>  apply  size_le
  _  ≤  (2  ^  max  l.depth  r.depth  -  1)  +  (2  ^  max  l.depth  r.depth  -  1)  +  1  :=  by
  gcongr  <;>  simp
  _  ≤  2  ^  (max  l.depth  r.depth  +  1)  -  1  :=  by
  have  :  0  <  2  ^  max  l.depth  r.depth  :=  by  simp
  omega 
```

尝试证明以下不等式，这稍微容易一些。记住，如果你像前一个定理那样进行归纳证明，你必须删除 `:= by`。

```py
theorem  depth_le_size  :  ∀  t  :  BinTree,  depth  t  ≤  size  t  :=  by  sorry 
```

还定义二叉树上的 `flip` 操作，它递归地交换左右子树。

```py
def  flip  :  BinTree  →  BinTree  :=  sorry 
```

如果你做对了，以下证明应该是 rfl。

```py
example:  flip  (node  (node  empty  (node  empty  empty))  (node  empty  empty))  =
  node  (node  empty  empty)  (node  (node  empty  empty)  empty)  :=  sorry 
```

证明以下内容：

```py
theorem  size_flip  :  ∀  t,  size  (flip  t)  =  size  t  :=  by  sorry 
```

我们以一些形式逻辑结束本节。以下是对命题公式的归纳定义。

```py
inductive  PropForm  :  Type  where
  |  var  (n  :  ℕ)  :  PropForm
  |  fls  :  PropForm
  |  conj  (A  B  :  PropForm)  :  PropForm
  |  disj  (A  B  :  PropForm)  :  PropForm
  |  impl  (A  B  :  PropForm)  :  PropForm 
```

每个命题公式要么是一个变量 `var n`，要么是常量假 `fls`，要么是形如 `conj A B`、`disj A B` 或 `impl A B` 的复合公式。用常规数学符号，这些通常分别写成 $p_n$、$\bot$、$A \wedge B$、$A \vee B$ 和 $A \to B$。其他命题连接词可以用这些来定义；例如，我们可以定义 $\neg A$ 为 $A \to \bot$ 和 $A \leftrightarrow B$ 为 $(A \to B) \wedge (B \to A)$。

定义了命题公式的数据类型之后，我们定义了相对于布尔真值赋值 `v` 的变量赋值来评估命题公式的含义。

```py
def  eval  :  PropForm  →  (ℕ  →  Bool)  →  Bool
  |  var  n,  v  =>  v  n
  |  fls,  _  =>  false
  |  conj  A  B,  v  =>  A.eval  v  &&  B.eval  v
  |  disj  A  B,  v  =>  A.eval  v  ||  B.eval  v
  |  impl  A  B,  v  =>  !  A.eval  v  ||  B.eval  v 
```

下一个定义指定了公式中出现的变量集合，接下来的定理表明，在两个变量上达成一致的真值赋值上评估公式会产生相同的值。

```py
def  vars  :  PropForm  →  Finset  ℕ
  |  var  n  =>  {n}
  |  fls  =>  ∅
  |  conj  A  B  =>  A.vars  ∪  B.vars
  |  disj  A  B  =>  A.vars  ∪  B.vars
  |  impl  A  B  =>  A.vars  ∪  B.vars

theorem  eval_eq_eval  :  ∀  (A  :  PropForm)  (v1  v2  :  ℕ  →  Bool),
  (∀  n  ∈  A.vars,  v1  n  =  v2  n)  →  A.eval  v1  =  A.eval  v2
  |  var  n,  v1,  v2,  h  =>  by  simp_all  [vars,  eval,  h]
  |  fls,  v1,  v2,  h  =>  by  simp_all  [eval]
  |  conj  A  B,  v1,  v2,  h  =>  by
  simp_all  [vars,  eval,  eval_eq_eval  A  v1  v2,  eval_eq_eval  B  v1  v2]
  |  disj  A  B,  v1,  v2,  h  =>  by
  simp_all  [vars,  eval,  eval_eq_eval  A  v1  v2,  eval_eq_eval  B  v1  v2]
  |  impl  A  B,  v1,  v2,  h  =>  by
  simp_all  [vars,  eval,  eval_eq_eval  A  v1  v2,  eval_eq_eval  B  v1  v2] 
```

注意到重复，我们可以巧妙地使用自动化。

```py
theorem  eval_eq_eval'  (A  :  PropForm)  (v1  v2  :  ℕ  →  Bool)  (h  :  ∀  n  ∈  A.vars,  v1  n  =  v2  n)  :
  A.eval  v1  =  A.eval  v2  :=  by
  cases  A  <;>  simp_all  [eval,  vars,  fun  A  =>  eval_eq_eval'  A  v1  v2] 
```

函数 `subst A m C` 描述了将公式 `C` 替换到公式 `A` 中每个 `var m` 出现的结果。

```py
def  subst  :  PropForm  →  ℕ  →  PropForm  →  PropForm
  |  var  n,  m,  C  =>  if  n  =  m  then  C  else  var  n
  |  fls,  _,  _  =>  fls
  |  conj  A  B,  m,  C  =>  conj  (A.subst  m  C)  (B.subst  m  C)
  |  disj  A  B,  m,  C  =>  disj  (A.subst  m  C)  (B.subst  m  C)
  |  impl  A  B,  m,  C  =>  impl  (A.subst  m  C)  (B.subst  m  C) 
```

例如，展示替换一个在公式中不出现的变量没有任何影响：

```py
theorem  subst_eq_of_not_mem_vars  :
  ∀  (A  :  PropForm)  (n  :  ℕ)  (C  :  PropForm),  n  ∉  A.vars  →  A.subst  n  C  =  A  :=  sorry 
```

以下定理提出了更微妙且有趣的内容：在真值赋值`v`上评估`A.subst n C`与在将`C`的值赋给变量`n`的真值赋值上评估`A`是相同的。看看你是否能证明它。

```py
theorem  subst_eval_eq  :  ∀  (A  :  PropForm)  (n  :  ℕ)  (C  :  PropForm)  (v  :  ℕ  →  Bool),
  (A.subst  n  C).eval  v  =  A.eval  (fun  m  =>  if  m  =  n  then  C.eval  v  else  v  m)  :=  sorry 
``` 上一节 下一节

* * *

© 版权所有 2020-2025，Jeremy Avigad，Patrick Massot。文本许可协议为 CC BY 4.0。

使用[Sphinx](https://www.sphinx-doc.org/)和[Read the Docs](https://readthedocs.org)提供的[主题](https://github.com/readthedocs/sphinx_rtd_theme)构建。*离散数学*是研究有限集合、对象和结构的学科。我们可以计算有限集合的元素数量，也可以对其元素进行有限求和或乘积计算，还可以计算最大值和最小值等。我们还可以研究由有限次应用某些生成函数生成的对象，可以通过结构递归定义函数，并通过结构归纳证明定理。本章描述了 Mathlib 支持这些活动的部分。

## 6.1. Finsets 和 Fintypes

在 Mathlib 中处理有限集合和类型可能会令人困惑，因为该库提供了多种处理它们的方式。在本节中，我们将讨论最常见的方法。

我们已经在第 5.2 节和第 5.3 节中遇到了`Finset`类型。正如其名所示，`Finset α`类型的一个元素是类型`α`的有限集合。我们将称这些为“finsets”。`Finset`数据类型旨在具有计算解释，并且`Finset α`上的许多基本操作都假设`α`具有可判定的等价性，这保证了存在一个算法来测试`a : α`是否是 finset `s`的元素。

```py
section
variable  {α  :  Type*}  [DecidableEq  α]  (a  :  α)  (s  t  :  Finset  α)

#check  a  ∈  s
#check  s  ∩  t

end 
```

如果你移除声明`[DecidableEq α]`，Lean 将在`#check s ∩ t`行上抱怨，因为它无法计算交集。然而，你应该期望能够计算的所有数据类型都具有可判定的等价性，如果你通过打开`Classical`命名空间并声明`noncomputable section`进行经典工作，你就可以对任何类型的元素进行 finsets 的推理。

Finsets 支持集合所具有的大多数集合论操作：

```py
open  Finset

variable  (a  b  c  :  Finset  ℕ)
variable  (n  :  ℕ)

#check  a  ∩  b
#check  a  ∪  b
#check  a  \  b
#check  (∅  :  Finset  ℕ)

example  :  a  ∩  (b  ∪  c)  =  (a  ∩  b)  ∪  (a  ∩  c)  :=  by
  ext  x;  simp  only  [mem_inter,  mem_union];  tauto

example  :  a  ∩  (b  ∪  c)  =  (a  ∩  b)  ∪  (a  ∩  c)  :=  by  rw  [inter_union_distrib_left] 
```

注意，我们已经打开了`Finset`命名空间，其中包含特定于 finsets 的定理。如果你浏览下面的最后一个例子，你会看到应用`ext`后跟`simp`将恒等式简化为命题逻辑中的问题。作为练习，你可以尝试证明一些来自第四章的集合恒等式，并将其转移到 finsets 中。

你已经看到了 `Finset.range n` 的表示法，它表示自然数有限集合 $\{ 0, 1, \ldots, n-1 \}$。`Finset` 还允许你通过枚举元素来定义有限集合：

```py
#check  ({0,  2,  5}  :  Finset  Nat)

def  example1  :  Finset  ℕ  :=  {0,  1,  2} 
```

有多种方法可以让 Lean 认识到以这种方式呈现的集合中元素的顺序和重复是不重要的。

```py
example  :  ({0,  1,  2}  :  Finset  ℕ)  =  {1,  2,  0}  :=  by  decide

example  :  ({0,  1,  2}  :  Finset  ℕ)  =  {0,  1,  1,  2}  :=  by  decide

example  :  ({0,  1}  :  Finset  ℕ)  =  {1,  0}  :=  by  rw  [Finset.pair_comm]

example  (x  :  Nat)  :  ({x,  x}  :  Finset  ℕ)  =  {x}  :=  by  simp

example  (x  y  z  :  Nat)  :  ({x,  y,  z,  y,  z,  x}  :  Finset  ℕ)  =  {x,  y,  z}  :=  by
  ext  i;  simp  [or_comm,  or_assoc,  or_left_comm]

example  (x  y  z  :  Nat)  :  ({x,  y,  z,  y,  z,  x}  :  Finset  ℕ)  =  {x,  y,  z}  :=  by
  ext  i;  simp;  tauto 
```

你可以使用 `insert` 向有限集合添加单个元素，并使用 `Finset.erase` 删除单个元素。请注意，`erase` 在 `Finset` 命名空间中，但 `insert` 在根命名空间中。

```py
example  (s  :  Finset  ℕ)  (a  :  ℕ)  (h  :  a  ∉  s)  :  (insert  a  s  |>.erase  a)  =  s  :=
  Finset.erase_insert  h

example  (s  :  Finset  ℕ)  (a  :  ℕ)  (h  :  a  ∈  s)  :  insert  a  (s.erase  a)  =  s  :=
  Finset.insert_erase  h 
```

事实上，`{0, 1, 2}` 只是对 `insert 0 (insert 1 (singleton 2))` 的表示。

```py
set_option  pp.notation  false  in
#check  ({0,  1,  2}  :  Finset  ℕ) 
```

给定一个有限集合 `s` 和一个谓词 `P`，我们可以使用集合构造符号 `{x ∈ s | P x}` 来定义满足 `P` 的 `s` 的元素集合。这是 `Finset.filter P s` 的表示法，也可以写作 `s.filter P`。

```py
example  :  {m  ∈  range  n  |  Even  m}  =  (range  n).filter  Even  :=  rfl
example  :  {m  ∈  range  n  |  Even  m  ∧  m  ≠  3}  =  (range  n).filter  (fun  m  ↦  Even  m  ∧  m  ≠  3)  :=  rfl

example  :  {m  ∈  range  10  |  Even  m}  =  {0,  2,  4,  6,  8}  :=  by  decide 
```

Mathlib 知道有限集合在函数下的像也是一个有限集合。

```py
#check  (range  5).image  (fun  x  ↦  x  *  2)

example  :  (range  5).image  (fun  x  ↦  x  *  2)  =  {x  ∈  range  10  |  Even  x}  :=  by  decide 
```

Lean 还知道两个有限集合的笛卡尔积 `s ×ˢ t` 是一个有限集合，以及有限集合的幂集也是一个有限集合。（注意，`s ×ˢ t` 的表示法也适用于集合。）

```py
#check  s  ×ˢ  t
#check  s.powerset 
```

在有限集合的元素上定义操作是棘手的，因为任何这样的定义都必须独立于元素呈现的顺序。当然，你可以通过组合现有操作来定义函数。你还可以使用 `Finset.fold` 来对元素执行二元操作，前提是操作是结合的和交换的，因为这些性质保证了结果与操作应用的顺序无关。有限和、积和并集就是这样定义的。在下面的最后一个例子中，`biUnion` 表示“有界索引并”。按照传统的数学符号，这个表达式将被写作 $\bigcup_{i ∈ s} g(i)$。

```py
#check  Finset.fold

def  f  (n  :  ℕ)  :  Int  :=  (↑n)²

#check  (range  5).fold  (fun  x  y  :  Int  ↦  x  +  y)  0  f
#eval  (range  5).fold  (fun  x  y  :  Int  ↦  x  +  y)  0  f

#check  ∑  i  ∈  range  5,  i²
#check  ∏  i  ∈  range  5,  i  +  1

variable  (g  :  Nat  →  Finset  Int)

#check  (range  5).biUnion  g 
```

有限集合有一个自然的归纳原理：要证明每个有限集合都具有某个性质，需要证明空集具有该性质，并且当我们向有限集合添加一个新元素时，该性质得到保持。（在 `@insert` 的下一个例子的归纳步骤中，`@` 符号是必需的，因为它为参数 `a` 和 `s` 提供了名称，因为它们已被标记为隐式。）

```py
#check  Finset.induction

example  {α  :  Type*}  [DecidableEq  α]  (f  :  α  →  ℕ)  (s  :  Finset  α)  (h  :  ∀  x  ∈  s,  f  x  ≠  0)  :
  ∏  x  ∈  s,  f  x  ≠  0  :=  by
  induction  s  using  Finset.induction_on  with
  |  empty  =>  simp
  |  @insert  a  s  anins  ih  =>
  rw  [prod_insert  anins]
  apply  mul_ne_zero
  ·  apply  h;  apply  mem_insert_self
  apply  ih
  intros  x  xs
  exact  h  x  (mem_insert_of_mem  xs) 
```

如果 `s` 是一个有限集合，`Finset.Nonempty s` 被定义为 `∃ x, x ∈ s`。你可以使用经典选择来选择一个非空有限集合的元素。同样，库定义了 `Finset.toList s`，它使用选择来以某种顺序选择 `s` 的元素。

```py
noncomputable  example  (s  :  Finset  ℕ)  (h  :  s.Nonempty)  :  ℕ  :=  Classical.choose  h

example  (s  :  Finset  ℕ)  (h  :  s.Nonempty)  :  Classical.choose  h  ∈  s  :=  Classical.choose_spec  h

noncomputable  example  (s  :  Finset  ℕ)  :  List  ℕ  :=  s.toList

example  (s  :  Finset  ℕ)  (a  :  ℕ)  :  a  ∈  s.toList  ↔  a  ∈  s  :=  mem_toList 
```

你可以使用 `Finset.min` 和 `Finset.max` 来选择线性顺序元素的 finset 的最小或最大元素，同样你也可以使用 `Finset.inf` 和 `Finset.sup` 与格元素的 finset，但有一个问题。空 finset 的最小元素应该是什么？你可以检查以下函数的带撇版本添加了一个先决条件，即 finset 不能为空。不带撇版本的 `Finset.min` 和 `Finset.max` 分别向输出类型添加一个顶或底元素，以处理 finset 为空的情况。不带撇版本的 `Finset.inf` 和 `Finset.sup` 假设格已经配备了顶或底元素。

```py
#check  Finset.min
#check  Finset.min'
#check  Finset.max
#check  Finset.max'
#check  Finset.inf
#check  Finset.inf'
#check  Finset.sup
#check  Finset.sup'

example  :  Finset.Nonempty  {2,  6,  7}  :=  ⟨6,  by  trivial⟩
example  :  Finset.min'  {2,  6,  7}  ⟨6,  by  trivial⟩  =  2  :=  by  trivial 
```

每个 finset `s` 都有一个有限的基数，`Finset.card s`，当 `Finset` 命名空间打开时可以写成 `#s`。

```py
#check  Finset.card

#eval  (range  5).card

example  (s  :  Finset  ℕ)  :  s.card  =  #s  :=  by  rfl

example  (s  :  Finset  ℕ)  :  s.card  =  ∑  i  ∈  s,  1  :=  by  rw  [card_eq_sum_ones]

example  (s  :  Finset  ℕ)  :  s.card  =  ∑  i  ∈  s,  1  :=  by  simp 
```

下一个部分完全是关于推理基数。

在形式化数学时，人们常常需要决定是否用集合或类型来表述自己的定义和定理。使用类型通常可以简化符号和证明，但处理类型的子集可能更加灵活。基于类型的 finset 类似物是 *fintype*，即某个 `α` 的类型 `Fintype α`。根据定义，fintype 只是一个带有包含所有元素的 finset `univ` 的数据类型。

```py
variable  {α  :  Type*}  [Fintype  α]

example  :  ∀  x  :  α,  x  ∈  Finset.univ  :=  by
  intro  x;  exact  mem_univ  x 
```

`Fintype.card α` 等于相应的 finset 的基数。

```py
example  :  Fintype.card  α  =  (Finset.univ  :  Finset  α).card  :=  rfl 
```

我们已经看到了 fintype 的一个典型例子，即每个 `n` 的类型 `Fin n`。Lean 认识到 fintype 在诸如积运算之类的操作下是封闭的。

```py
example  :  Fintype.card  (Fin  5)  =  5  :=  by  simp
example  :  Fintype.card  ((Fin  5)  ×  (Fin  3))  =  15  :=  by  simp 
```

`Finset α` 的任何元素 `s` 都可以强制转换为类型 `(↑s : Finset α)`，即包含在 `s` 中的 `α` 的元素的子类型。

```py
variable  (s  :  Finset  ℕ)

example  :  (↑s  :  Type)  =  {x  :  ℕ  //  x  ∈  s}  :=  rfl
example  :  Fintype.card  ↑s  =  s.card  :=  by  simp 
```

Lean 和 Mathlib 使用 *类型类推断* 来跟踪 fintype 上的附加结构，即包含所有元素的通用 finset。换句话说，你可以将 fintype 视为一个带有额外数据的代数结构。第七章 解释了这是如何工作的。

计数事物的艺术是组合数学的核心部分。Mathlib 包含了几个用于计数 finset 元素的基本恒等式：

```py
open  Finset

variable  {α  β  :  Type*}  [DecidableEq  α]  [DecidableEq  β]  (s  t  :  Finset  α)  (f  :  α  →  β)

example  :  #(s  ×ˢ  t)  =  #s  *  #t  :=  by  rw  [card_product]
example  :  #(s  ×ˢ  t)  =  #s  *  #t  :=  by  simp

example  :  #(s  ∪  t)  =  #s  +  #t  -  #(s  ∩  t)  :=  by  rw  [card_union]

example  (h  :  Disjoint  s  t)  :  #(s  ∪  t)  =  #s  +  #t  :=  by  rw  [card_union_of_disjoint  h]
example  (h  :  Disjoint  s  t)  :  #(s  ∪  t)  =  #s  +  #t  :=  by  simp  [h]

example  (h  :  Function.Injective  f)  :  #(s.image  f)  =  #s  :=  by  rw  [card_image_of_injective  _  h]

example  (h  :  Set.InjOn  f  s)  :  #(s.image  f)  =  #s  :=  by  rw  [card_image_of_injOn  h] 
```

打开 `Finset` 命名空间允许我们使用 `#s` 表示 `s.card`，以及使用缩短的名称 card_union 等等。

Mathlib 也可以计数 fintype 的元素：

```py
open  Fintype

variable  {α  β  :  Type*}  [Fintype  α]  [Fintype  β]

example  :  card  (α  ×  β)  =  card  α  *  card  β  :=  by  simp

example  :  card  (α  ⊕  β)  =  card  α  +  card  β  :=  by  simp

example  (n  :  ℕ)  :  card  (Fin  n  →  α)  =  (card  α)^n  :=  by  simp

variable  {n  :  ℕ}  {γ  :  Fin  n  →  Type*}  [∀  i,  Fintype  (γ  i)]

example  :  card  ((i  :  Fin  n)  →  γ  i)  =  ∏  i,  card  (γ  i)  :=  by  simp

example  :  card  (Σ  i,  γ  i)  =  ∑  i,  card  (γ  i)  :=  by  simp 
```

当 `Fintype` 命名空间没有打开时，我们必须使用 `Fintype.card` 而不是 card。

以下是一个计算有限集基数（finset）的例子，即范围 n 与一个比范围 n 移动超过 n 的范围 n 的副本的并集。计算需要证明联合中的两个集合是互斥的；证明的第一行给出了条件 `Disjoint (range n) (image (fun i ↦ m + i) (range n))`，这个条件在证明的末尾得到证实。`Disjoint` 谓词过于通用，对我们直接有用，但定理 `disjoint_iff_ne` 将其转换成我们可以使用的形式。

```py
#check  Disjoint

example  (m  n  :  ℕ)  (h  :  m  ≥  n)  :
  card  (range  n  ∪  (range  n).image  (fun  i  ↦  m  +  i))  =  2  *  n  :=  by
  rw  [card_union_of_disjoint,  card_range,  card_image_of_injective,  card_range];  omega
  .  apply  add_right_injective
  .  simp  [disjoint_iff_ne];  omega 
```

在本节中，`omega` 将成为我们的得力助手，用于处理算术计算和不等式。

这里有一个更有趣的例子。考虑由 $\{0, \ldots, n\} \times \{0, \ldots, n\}$ 的子集组成的对 $(i, j)$，其中 $i < j$。如果你把它们看作坐标平面上的格点，它们构成了一个以 $(0, 0)$ 和 $(n, n)$ 为顶点的正方形的上三角形，不包括对角线。整个正方形的基数是 $(n + 1)²$，减去对角线的长度并将结果除以二，我们得到三角形的基数是 $n (n + 1) / 2$。

或者，我们注意到三角形的行的大小是 $0, 1, \ldots, n$，因此基数是前 $n$ 个自然数的和。下面证明中的第一个 `have` 将三角形描述为行的并集，其中行 $j$ 由数字 $0, 1, ..., j - 1$ 与 $j$ 配对组成。在下面的证明中，符号 `(., j)` 简化了函数 `fun i ↦ (i, j)`。其余的证明只是对有限集基数进行计算。

```py
def  triangle  (n  :  ℕ)  :  Finset  (ℕ  ×  ℕ)  :=  {p  ∈  range  (n+1)  ×ˢ  range  (n+1)  |  p.1  <  p.2}

example  (n  :  ℕ)  :  #(triangle  n)  =  (n  +  1)  *  n  /  2  :=  by
  have  :  triangle  n  =  (range  (n+1)).biUnion  (fun  j  ↦  (range  j).image  (.,  j))  :=  by
  ext  p
  simp  only  [triangle,  mem_filter,  mem_product,  mem_range,  mem_biUnion,  mem_image]
  constructor
  .  rintro  ⟨⟨hp1,  hp2⟩,  hp3⟩
  use  p.2,  hp2,  p.1,  hp3
  .  rintro  ⟨p1,  hp1,  p2,  hp2,  rfl⟩
  omega
  rw  [this,  card_biUnion];  swap
  ·  -- take care of disjointness first
  intro  x  _  y  _  xney
  simp  [disjoint_iff_ne,  xney]
  -- continue the calculation
  transitivity  (∑  i  ∈  range  (n  +  1),  i)
  ·  congr;  ext  i
  rw  [card_image_of_injective,  card_range]
  intros  i1  i2;  simp
  rw  [sum_range_id];  rfl 
```

以下是对证明的变体，它使用有限类型（fintype）而不是有限集（finset）进行计算。类型 `α ≃ β` 是 `α` 和 `β` 之间等价类的类型，由正向映射、反向映射以及证明这两个映射是彼此的逆映射组成。证明中的第一个 `have` 显示 `triangle n` 与 `Fin i` 的不相交并等价，其中 `i` 在 `Fin (n + 1)` 上取值。有趣的是，正向函数和反向函数是用策略构建的，而不是明确写出。由于它们所做的只是移动数据和信息，`rfl` 证明了它们是逆映射。

之后，`rw [←Fintype.card_coe]` 将 `#(triangle n)` 重写为子类型 `{ x // x ∈ triangle n }` 的基数，其余的证明都是计算。

```py
example  (n  :  ℕ)  :  #(triangle  n)  =  (n  +  1)  *  n  /  2  :=  by
  have  :  triangle  n  ≃  Σ  i  :  Fin  (n  +  1),  Fin  i.val  :=
  {  toFun  :=  by
  rintro  ⟨⟨i,  j⟩,  hp⟩
  simp  [triangle]  at  hp
  exact  ⟨⟨j,  hp.1.2⟩,  ⟨i,  hp.2⟩⟩
  invFun  :=  by
  rintro  ⟨i,  j⟩
  use  ⟨j,  i⟩
  simp  [triangle]
  exact  j.isLt.trans  i.isLt
  left_inv  :=  by  intro  i;  rfl
  right_inv  :=  by  intro  i;  rfl  }
  rw  [←Fintype.card_coe]
  trans;  apply  (Fintype.card_congr  this)
  rw  [Fintype.card_sigma,  sum_fin_eq_sum_range]
  convert  Finset.sum_range_id  (n  +  1)
  simp_all 
```

这里还有另一种方法。下面证明的第一行将问题简化为证明 `2 * #(triangle n) = (n + 1) * n`。我们可以通过证明两个三角形的副本正好填满矩形 `range n ×ˢ range (n + 1)` 来做到这一点。作为一个练习，看看你是否能填补计算步骤。在解决方案中，我们在倒数第二步中广泛地使用了 `omega`，但不幸的是，我们不得不手动做相当多的工作。

```py
example  (n  :  ℕ)  :  #(triangle  n)  =  (n  +  1)  *  n  /  2  :=  by
  apply  Nat.eq_div_of_mul_eq_right  (by  norm_num)
  let  turn  (p  :  ℕ  ×  ℕ)  :  ℕ  ×  ℕ  :=  (n  -  1  -  p.1,  n  -  p.2)
  calc  2  *  #(triangle  n)
  =  #(triangle  n)  +  #(triangle  n)  :=  by
  sorry
  _  =  #(triangle  n)  +  #(triangle  n  |>.image  turn)  :=  by
  sorry
  _  =  #(range  n  ×ˢ  range  (n  +  1))  :=  by
  sorry
  _  =  (n  +  1)  *  n  :=  by
  sorry 
```

你可以自己验证，如果我们将`n`替换为`n + 1`，并在`triangle`的定义中将`<`替换为`≤`，我们会得到相同的三角形，只是向下移动了。下面的练习要求你使用这个事实来证明两个三角形具有相同的大小。

```py
def  triangle'  (n  :  ℕ)  :  Finset  (ℕ  ×  ℕ)  :=  {p  ∈  range  n  ×ˢ  range  n  |  p.1  ≤  p.2}

example  (n  :  ℕ)  :  #(triangle'  n)  =  #(triangle  n)  :=  by  sorry 
```

让我们以一个例子和一个练习来结束本节，这个例子和练习来自 Bhavik Mehta 在 2023 年于*Lean for the Curious Mathematician*上给出的[组合学教程](https://www.youtube.com/watch?v=_cJctOIXWE4&list=PLlF-CfQhukNn7xEbfL38eLgkveyk9_myQ&index=8&t=2737s&ab_channel=leanprovercommunity)。假设我们有一个由顶点集`s`和`t`组成的二分图，对于`s`中的每个`a`，至少有三条边离开`a`，而对于`t`中的每个`b`，最多只有一条边进入`b`。那么图中边的总数至少是`s`的基数的三倍，最多是`t`的基数，由此可以推出`s`的基数的三倍最多是`t`的基数。以下定理实现了这个论证，其中我们使用关系`r`来表示图的边。证明是一个优雅的计算。

```py
open  Classical
variable  (s  t  :  Finset  Nat)  (a  b  :  Nat)

theorem  doubleCounting  {α  β  :  Type*}  (s  :  Finset  α)  (t  :  Finset  β)
  (r  :  α  →  β  →  Prop)
  (h_left  :  ∀  a  ∈  s,  3  ≤  #{b  ∈  t  |  r  a  b})
  (h_right  :  ∀  b  ∈  t,  #{a  ∈  s  |  r  a  b}  ≤  1)  :
  3  *  #(s)  ≤  #(t)  :=  by
  calc  3  *  #(s)
  =  ∑  a  ∈  s,  3  :=  by  simp  [sum_const_nat,  mul_comm]
  _  ≤  ∑  a  ∈  s,  #({b  ∈  t  |  r  a  b})  :=  sum_le_sum  h_left
  _  =  ∑  a  ∈  s,  ∑  b  ∈  t,  if  r  a  b  then  1  else  0  :=  by  simp
  _  =  ∑  b  ∈  t,  ∑  a  ∈  s,  if  r  a  b  then  1  else  0  :=  sum_comm
  _  =  ∑  b  ∈  t,  #({a  ∈  s  |  r  a  b})  :=  by  simp
  _  ≤  ∑  b  ∈  t,  1  :=  sum_le_sum  h_right
  _  ≤  #(t)  :=  by  simp 
```

以下练习也来自 Mehta 的教程。假设`A`是`range (2 * n)`的子集，包含`n + 1`个元素。很容易看出`A`必须包含两个连续的整数，因此包含两个互质的元素。如果你观看教程，你会看到大量的努力被用于建立以下事实，现在这个事实被`omega`自动证明。

```py
example  (m  k  :  ℕ)  (h  :  m  ≠  k)  (h'  :  m  /  2  =  k  /  2)  :  m  =  k  +  1  ∨  k  =  m  +  1  :=  by  omega 
```

Mehta 练习的解决方案使用了鸽巢原理，形式为`exists_lt_card_fiber_of_mul_lt_card_of_maps_to`，来证明在`A`中有两个不同的元素`m`和`k`，使得`m / 2 = k / 2`。看看你是否能完成这个事实的证明，然后使用它来完成证明。

```py
example  {n  :  ℕ}  (A  :  Finset  ℕ)
  (hA  :  #(A)  =  n  +  1)
  (hA'  :  A  ⊆  range  (2  *  n))  :
  ∃  m  ∈  A,  ∃  k  ∈  A,  Nat.Coprime  m  k  :=  by
  have  :  ∃  t  ∈  range  n,  1  <  #({u  ∈  A  |  u  /  2  =  t})  :=  by
  apply  exists_lt_card_fiber_of_mul_lt_card_of_maps_to
  ·  sorry
  ·  sorry
  rcases  this  with  ⟨t,  ht,  ht'⟩
  simp  only  [one_lt_card,  mem_filter]  at  ht'
  sorry 
```  ## 6.3\. 归纳定义的类型

Lean 的基础允许我们定义归纳类型，即从底部向上生成实例的数据类型。例如，由`α`的元素组成的列表数据类型`List α`是通过从空列表`nil`开始，并依次向列表前添加元素来生成的。下面我们将定义一个二叉树类型`BinTree`，其元素是通过从空树开始，并通过将新节点附加到两个现有树来构建新树来生成的。

在 Lean 中，可以定义对象无限归纳类型，例如可数分支良基树。在离散数学中，有限归纳定义被广泛使用，尤其是在与计算机科学相关的离散数学分支中。Lean 不仅提供了定义此类类型的手段，还提供了归纳和递归定义的原则。例如，数据类型`List α`是通过归纳定义的：

```py
namespace  MyListSpace

inductive  List  (α  :  Type*)  where
  |  nil  :  List  α
  |  cons  :  α  →  List  α  →  List  α

end  MyListSpace 
```

归纳定义说明`List α`的每个元素要么是`nil`，即空列表，要么是`cons a as`，其中`a`是`α`的一个元素，`as`是`α`元素列表。构造函数被适当地命名为`List.nil`和`List.cons`，但如果你使用`List`命名空间是开放的，你可以使用更短的符号。当`List`命名空间*不是*开放时，你可以在`Lean`期望列表的任何地方写`.nil`和`.cons a as`，`Lean`将自动插入`List`限定符。在本节中，我们将临时定义放在单独的命名空间，如`MyListSpace`，以避免与标准库冲突。在临时命名空间之外，我们将恢复使用标准库定义。

Lean 定义了`[]`表示`nil`和`::`表示`cons`的符号，你可以用`[a, b, c]`表示`a :: b :: c :: []`。append 和 map 函数递归地定义为以下：

```py
def  append  {α  :  Type*}  :  List  α  →  List  α  →  List  α
  |  [],  bs  =>  bs
  |  a  ::  as,  bs  =>  a  ::  (append  as  bs)

def  map  {α  β  :  Type*}  (f  :  α  →  β)  :  List  α  →  List  β
  |  []  =>  []
  |  a  ::  as  =>  f  a  ::  map  f  as

#eval  append  [1,  2,  3]  [4,  5,  6]
#eval  map  (fun  n  =>  n²)  [1,  2,  3,  4,  5] 
```

注意，这里有一个基本情况和一个递归情况。在每种情况下，两个定义子句都定义性地成立：

```py
theorem  nil_append  {α  :  Type*}  (as  :  List  α)  :  append  []  as  =  as  :=  rfl

theorem  cons_append  {α  :  Type*}  (a  :  α)  (as  :  List  α)  (bs  :  List  α)  :
  append  (a  ::  as)  bs  =  a  ::  (append  as  bs)  :=  rfl

theorem  map_nil  {α  β  :  Type*}  (f  :  α  →  β)  :  map  f  []  =  []  :=  rfl

theorem  map_cons  {α  β  :  Type*}  (f  :  α  →  β)  (a  :  α)  (as  :  List  α)  :
  map  f  (a  ::  as)  =  f  a  ::  map  f  as  :=  rfl 
```

函数`append`和`map`在标准库中定义，`append as bs`可以写成`as ++ bs`。

Lean 允许你按照定义的结构通过归纳来编写证明。

```py
variable  {α  β  γ  :  Type*}
variable  (as  bs  cs  :  List  α)
variable  (a  b  c  :  α)

open  List

theorem  append_nil  :  ∀  as  :  List  α,  as  ++  []  =  as
  |  []  =>  rfl
  |  a  ::  as  =>  by  rw  [cons_append,  append_nil  as]

theorem  map_map  (f  :  α  →  β)  (g  :  β  →  γ)  :
  ∀  as  :  List  α,  map  g  (map  f  as)  =  map  (g  ∘  f)  as
  |  []  =>  rfl
  |  a  ::  as  =>  by  rw  [map_cons,  map_cons,  map_cons,  map_map  f  g  as];  rfl 
```

你还可以使用`induction'`策略。

当然，这些定理已经在标准库中。作为一个练习，尝试在`MyListSpace3`命名空间中定义一个函数`reverse`（以避免与标准`List.reverse`冲突），该函数可以反转一个列表。你可以使用`#eval reverse [1, 2, 3, 4, 5]`来测试它。`reverse`的最直接定义需要二次时间，但不用担心这一点。你可以跳转到标准库中`List.reverse`的定义，以查看线性时间实现。尝试证明`reverse (as ++ bs) = reverse bs ++ reverse as`和`reverse (reverse as) = as`。你可以使用`cons_append`和`append_assoc`，但你可能需要提出辅助引理并证明它们。

```py
def  reverse  :  List  α  →  List  α  :=  sorry

theorem  reverse_append  (as  bs  :  List  α)  :  reverse  (as  ++  bs)  =  reverse  bs  ++  reverse  as  :=  by
  sorry

theorem  reverse_reverse  (as  :  List  α)  :  reverse  (reverse  as)  =  as  :=  by  sorry 
```

对于另一个例子，考虑以下关于二叉树的归纳定义以及计算二叉树大小和深度的函数。

```py
inductive  BinTree  where
  |  empty  :  BinTree
  |  node  :  BinTree  →  BinTree  →  BinTree

namespace  BinTree

def  size  :  BinTree  →  ℕ
  |  empty  =>  0
  |  node  l  r  =>  size  l  +  size  r  +  1

def  depth  :  BinTree  →  ℕ
  |  empty  =>  0
  |  node  l  r  =>  max  (depth  l)  (depth  r)  +  1 
```

将空二叉树视为大小为 0、深度为 0 的二叉树是很方便的。在文献中，这种数据类型有时被称为*扩展二叉树*。包括空树意味着，例如，我们可以定义一个由根节点、空左子树和由单个节点组成的右子树构成的树`node empty (node empty empty)`。

这里有一个重要的不等式，它关联了大小和深度：

```py
theorem  size_le  :  ∀  t  :  BinTree,  size  t  ≤  2^depth  t  -  1
  |  empty  =>  Nat.zero_le  _
  |  node  l  r  =>  by
  simp  only  [depth,  size]
  calc  l.size  +  r.size  +  1
  ≤  (2^l.depth  -  1)  +  (2^r.depth  -  1)  +  1  :=  by
  gcongr  <;>  apply  size_le
  _  ≤  (2  ^  max  l.depth  r.depth  -  1)  +  (2  ^  max  l.depth  r.depth  -  1)  +  1  :=  by
  gcongr  <;>  simp
  _  ≤  2  ^  (max  l.depth  r.depth  +  1)  -  1  :=  by
  have  :  0  <  2  ^  max  l.depth  r.depth  :=  by  simp
  omega 
```

尝试证明以下不等式，这相对容易一些。记住，如果你像上一个定理那样进行归纳证明，你必须删除`:= by`。

```py
theorem  depth_le_size  :  ∀  t  :  BinTree,  depth  t  ≤  size  t  :=  by  sorry 
```

还定义二叉树上的`flip`操作，该操作递归地交换左右子树。

```py
def  flip  :  BinTree  →  BinTree  :=  sorry 
```

如果你做得正确，以下证明应该是`rfl`。

```py
example:  flip  (node  (node  empty  (node  empty  empty))  (node  empty  empty))  =
  node  (node  empty  empty)  (node  (node  empty  empty)  empty)  :=  sorry 
```

证明以下内容：

```py
theorem  size_flip  :  ∀  t,  size  (flip  t)  =  size  t  :=  by  sorry 
```

我们以一些形式逻辑结束本节。以下是对命题公式的归纳定义。

```py
inductive  PropForm  :  Type  where
  |  var  (n  :  ℕ)  :  PropForm
  |  fls  :  PropForm
  |  conj  (A  B  :  PropForm)  :  PropForm
  |  disj  (A  B  :  PropForm)  :  PropForm
  |  impl  (A  B  :  PropForm)  :  PropForm 
```

每个命题公式要么是一个变量 `var n`，要么是常假 `fls`，要么是形式为 `conj A B`、`disj A B` 或 `impl A B` 的复合公式。使用常规数学符号，这些通常分别写作 $p_n$、$\bot$、$A \wedge B$、$A \vee B$ 和 $A \to B$。其他命题连接词可以用这些来定义；例如，我们可以将 $\neg A$ 定义为 $A \to \bot$，将 $A \leftrightarrow B$ 定义为 $(A \to B) \wedge (B \to A)$。

定义了命题公式的数据类型后，我们定义了相对于将布尔真值分配给变量的赋值 `v` 来评估命题公式意味着什么。

```py
def  eval  :  PropForm  →  (ℕ  →  Bool)  →  Bool
  |  var  n,  v  =>  v  n
  |  fls,  _  =>  false
  |  conj  A  B,  v  =>  A.eval  v  &&  B.eval  v
  |  disj  A  B,  v  =>  A.eval  v  ||  B.eval  v
  |  impl  A  B,  v  =>  !  A.eval  v  ||  B.eval  v 
```

下一个定义指定了公式中出现的变量的集合，接下来的定理表明，在两个在变量上达成一致的真值赋值上评估公式会产生相同的值。

```py
def  vars  :  PropForm  →  Finset  ℕ
  |  var  n  =>  {n}
  |  fls  =>  ∅
  |  conj  A  B  =>  A.vars  ∪  B.vars
  |  disj  A  B  =>  A.vars  ∪  B.vars
  |  impl  A  B  =>  A.vars  ∪  B.vars

theorem  eval_eq_eval  :  ∀  (A  :  PropForm)  (v1  v2  :  ℕ  →  Bool),
  (∀  n  ∈  A.vars,  v1  n  =  v2  n)  →  A.eval  v1  =  A.eval  v2
  |  var  n,  v1,  v2,  h  =>  by  simp_all  [vars,  eval,  h]
  |  fls,  v1,  v2,  h  =>  by  simp_all  [eval]
  |  conj  A  B,  v1,  v2,  h  =>  by
  simp_all  [vars,  eval,  eval_eq_eval  A  v1  v2,  eval_eq_eval  B  v1  v2]
  |  disj  A  B,  v1,  v2,  h  =>  by
  simp_all  [vars,  eval,  eval_eq_eval  A  v1  v2,  eval_eq_eval  B  v1  v2]
  |  impl  A  B,  v1,  v2,  h  =>  by
  simp_all  [vars,  eval,  eval_eq_eval  A  v1  v2,  eval_eq_eval  B  v1  v2] 
```

注意到重复，我们可以巧妙地使用自动化。

```py
theorem  eval_eq_eval'  (A  :  PropForm)  (v1  v2  :  ℕ  →  Bool)  (h  :  ∀  n  ∈  A.vars,  v1  n  =  v2  n)  :
  A.eval  v1  =  A.eval  v2  :=  by
  cases  A  <;>  simp_all  [eval,  vars,  fun  A  =>  eval_eq_eval'  A  v1  v2] 
```

函数 `subst A m C` 描述了将公式 `C` 替换为公式 `A` 中每个 `var m` 出现的结果。

```py
def  subst  :  PropForm  →  ℕ  →  PropForm  →  PropForm
  |  var  n,  m,  C  =>  if  n  =  m  then  C  else  var  n
  |  fls,  _,  _  =>  fls
  |  conj  A  B,  m,  C  =>  conj  (A.subst  m  C)  (B.subst  m  C)
  |  disj  A  B,  m,  C  =>  disj  (A.subst  m  C)  (B.subst  m  C)
  |  impl  A  B,  m,  C  =>  impl  (A.subst  m  C)  (B.subst  m  C) 
```

例如，展示替换一个在公式中未出现的变量没有任何影响：

```py
theorem  subst_eq_of_not_mem_vars  :
  ∀  (A  :  PropForm)  (n  :  ℕ)  (C  :  PropForm),  n  ∉  A.vars  →  A.subst  n  C  =  A  :=  sorry 
```

以下定理提出了更微妙和有趣的内容：在真值赋值 `v` 上评估 `A.subst n C` 与在将 `C` 的值分配给 `var n` 的真值赋值上评估 `A` 是相同的。看看你是否能证明它。

```py
theorem  subst_eval_eq  :  ∀  (A  :  PropForm)  (n  :  ℕ)  (C  :  PropForm)  (v  :  ℕ  →  Bool),
  (A.subst  n  C).eval  v  =  A.eval  (fun  m  =>  if  m  =  n  then  C.eval  v  else  v  m)  :=  sorry 
```  ## 6.1\. Finsets 和 Fintypes

在 Mathlib 中处理有限集和类型可能会令人困惑，因为该库提供了多种处理它们的方式。在本节中，我们将讨论最常见的方法。

我们已经在 第 5.2 节 和 第 5.3 节 中遇到了类型 `Finset`。正如其名所示，类型 `Finset α` 的元素是类型 `α` 的有限集合。我们将称这些为“有限集”。`Finset` 数据类型旨在具有计算解释，并且 `Finset α` 上的许多基本操作都假设 `α` 具有可判定的等价性，这保证了存在一个算法来测试 `a : α` 是否是有限集 `s` 的元素。

```py
section
variable  {α  :  Type*}  [DecidableEq  α]  (a  :  α)  (s  t  :  Finset  α)

#check  a  ∈  s
#check  s  ∩  t

end 
```

如果你移除声明 `[DecidableEq α]`，Lean 将在第 `#check s ∩ t` 行上抱怨，因为它无法计算交集。然而，你应该期望能够计算的所有数据类型都具有可判定的等价性，并且如果你通过打开 `Classical` 命名空间并声明 `noncomputable section` 来进行经典工作，你就可以对任何类型的元素的所有有限集进行推理。

有限集支持大多数集合论操作，这些操作集合也支持：

```py
open  Finset

variable  (a  b  c  :  Finset  ℕ)
variable  (n  :  ℕ)

#check  a  ∩  b
#check  a  ∪  b
#check  a  \  b
#check  (∅  :  Finset  ℕ)

example  :  a  ∩  (b  ∪  c)  =  (a  ∩  b)  ∪  (a  ∩  c)  :=  by
  ext  x;  simp  only  [mem_inter,  mem_union];  tauto

example  :  a  ∩  (b  ∪  c)  =  (a  ∩  b)  ∪  (a  ∩  c)  :=  by  rw  [inter_union_distrib_left] 
```

注意，我们已经打开了 `Finset` 命名空间，其中包含特定于有限集合的定理。如果你浏览下面的最后一个例子，你会看到应用 `ext` 后跟 `simp` 将恒等式简化为一个命题逻辑问题。作为练习，你可以尝试证明一些来自 第四章 的集合恒等式，这些恒等式被转移到有限集合中。

你已经看到了 `Finset.range n` 的表示法，表示自然数有限集合 $\{ 0, 1, \ldots, n-1 \}$。`Finset` 还允许你通过枚举元素来定义有限集合：

```py
#check  ({0,  2,  5}  :  Finset  Nat)

def  example1  :  Finset  ℕ  :=  {0,  1,  2} 
```

有多种方法可以让 Lean 识别出在这种方式呈现的集合中元素顺序和重复元素不重要。

```py
example  :  ({0,  1,  2}  :  Finset  ℕ)  =  {1,  2,  0}  :=  by  decide

example  :  ({0,  1,  2}  :  Finset  ℕ)  =  {0,  1,  1,  2}  :=  by  decide

example  :  ({0,  1}  :  Finset  ℕ)  =  {1,  0}  :=  by  rw  [Finset.pair_comm]

example  (x  :  Nat)  :  ({x,  x}  :  Finset  ℕ)  =  {x}  :=  by  simp

example  (x  y  z  :  Nat)  :  ({x,  y,  z,  y,  z,  x}  :  Finset  ℕ)  =  {x,  y,  z}  :=  by
  ext  i;  simp  [or_comm,  or_assoc,  or_left_comm]

example  (x  y  z  :  Nat)  :  ({x,  y,  z,  y,  z,  x}  :  Finset  ℕ)  =  {x,  y,  z}  :=  by
  ext  i;  simp;  tauto 
```

你可以使用 `insert` 向有限集合添加单个元素，并使用 `Finset.erase` 删除单个元素。请注意，`erase` 在 `Finset` 命名空间中，而 `insert` 在根命名空间中。

```py
example  (s  :  Finset  ℕ)  (a  :  ℕ)  (h  :  a  ∉  s)  :  (insert  a  s  |>.erase  a)  =  s  :=
  Finset.erase_insert  h

example  (s  :  Finset  ℕ)  (a  :  ℕ)  (h  :  a  ∈  s)  :  insert  a  (s.erase  a)  =  s  :=
  Finset.insert_erase  h 
```

事实上，`{0, 1, 2}` 只是 `insert 0 (insert 1 (singleton 2))` 的表示法。

```py
set_option  pp.notation  false  in
#check  ({0,  1,  2}  :  Finset  ℕ) 
```

给定一个有限集合 `s` 和一个谓词 `P`，我们可以使用集合构造器符号 `{x ∈ s | P x}` 来定义满足 `P` 的 `s` 的元素集合。这是 `Finset.filter P s` 的表示法，也可以写成 `s.filter P`。

```py
example  :  {m  ∈  range  n  |  Even  m}  =  (range  n).filter  Even  :=  rfl
example  :  {m  ∈  range  n  |  Even  m  ∧  m  ≠  3}  =  (range  n).filter  (fun  m  ↦  Even  m  ∧  m  ≠  3)  :=  rfl

example  :  {m  ∈  range  10  |  Even  m}  =  {0,  2,  4,  6,  8}  :=  by  decide 
```

Mathlib 知道在函数下的有限集合的像是有限集合。

```py
#check  (range  5).image  (fun  x  ↦  x  *  2)

example  :  (range  5).image  (fun  x  ↦  x  *  2)  =  {x  ∈  range  10  |  Even  x}  :=  by  decide 
```

Lean 还知道，两个有限集合的笛卡尔积 `s ×ˢ t` 也是一个有限集合，并且有限集合的幂集也是一个有限集合。（注意，符号 `s ×ˢ t` 也可以用于集合。）

```py
#check  s  ×ˢ  t
#check  s.powerset 
```

在有限集合的元素上定义操作是棘手的，因为任何这样的定义都必须独立于元素呈现的顺序。当然，你可以通过组合现有操作来定义函数。你还可以使用 `Finset.fold` 对元素应用二元运算，前提是该运算是结合的和交换的，因为这些性质保证了结果与运算应用顺序无关。有限和、积和并集就是这样定义的。在下面的最后一个例子中，`biUnion` 代表“有界索引并”。用传统的数学符号，该表达式将被写成 $\bigcup_{i ∈ s} g(i)$。

```py
#check  Finset.fold

def  f  (n  :  ℕ)  :  Int  :=  (↑n)²

#check  (range  5).fold  (fun  x  y  :  Int  ↦  x  +  y)  0  f
#eval  (range  5).fold  (fun  x  y  :  Int  ↦  x  +  y)  0  f

#check  ∑  i  ∈  range  5,  i²
#check  ∏  i  ∈  range  5,  i  +  1

variable  (g  :  Nat  →  Finset  Int)

#check  (range  5).biUnion  g 
```

对于有限集合的归纳原理是自然的：要证明每个有限集合都具有某个属性，需要证明空集具有该属性，并且当我们向有限集合添加一个新元素时，该属性得到保持。（在 `@insert` 的下一个例子的归纳步骤中，`@` 符号是必需的，因为它为参数 `a` 和 `s` 提供了名称，因为它们已被标记为隐式。）

```py
#check  Finset.induction

example  {α  :  Type*}  [DecidableEq  α]  (f  :  α  →  ℕ)  (s  :  Finset  α)  (h  :  ∀  x  ∈  s,  f  x  ≠  0)  :
  ∏  x  ∈  s,  f  x  ≠  0  :=  by
  induction  s  using  Finset.induction_on  with
  |  empty  =>  simp
  |  @insert  a  s  anins  ih  =>
  rw  [prod_insert  anins]
  apply  mul_ne_zero
  ·  apply  h;  apply  mem_insert_self
  apply  ih
  intros  x  xs
  exact  h  x  (mem_insert_of_mem  xs) 
```

如果 `s` 是一个有限集合，则 `Finset.Nonempty s` 被定义为 `∃ x, x ∈ s`。你可以使用经典选择来选择一个非空有限集合的元素。同样，库定义了 `Finset.toList s`，它使用选择以某种顺序选择 `s` 的元素。

```py
noncomputable  example  (s  :  Finset  ℕ)  (h  :  s.Nonempty)  :  ℕ  :=  Classical.choose  h

example  (s  :  Finset  ℕ)  (h  :  s.Nonempty)  :  Classical.choose  h  ∈  s  :=  Classical.choose_spec  h

noncomputable  example  (s  :  Finset  ℕ)  :  List  ℕ  :=  s.toList

example  (s  :  Finset  ℕ)  (a  :  ℕ)  :  a  ∈  s.toList  ↔  a  ∈  s  :=  mem_toList 
```

你可以使用`Finset.min`和`Finset.max`来选择线性顺序元素的有限集的最小或最大元素，同样你也可以使用`Finset.inf`和`Finset.sup`与格元素的有限集，但有一个问题。空有限集的最小元素应该是什么？你可以检查以下函数的带撇版本添加了一个先决条件，即有限集非空。不带撇版本的`Finset.min`和`Finset.max`分别向输出类型添加一个上界或下界元素，以处理有限集为空的情况。不带撇版本的`Finset.inf`和`Finset.sup`假设格已经配备了上界或下界元素。

```py
#check  Finset.min
#check  Finset.min'
#check  Finset.max
#check  Finset.max'
#check  Finset.inf
#check  Finset.inf'
#check  Finset.sup
#check  Finset.sup'

example  :  Finset.Nonempty  {2,  6,  7}  :=  ⟨6,  by  trivial⟩
example  :  Finset.min'  {2,  6,  7}  ⟨6,  by  trivial⟩  =  2  :=  by  trivial 
```

每个有限集`s`都有一个有限的基数，`Finset.card s`，当`Finset`命名空间打开时，可以写成`#s`。

```py
#check  Finset.card

#eval  (range  5).card

example  (s  :  Finset  ℕ)  :  s.card  =  #s  :=  by  rfl

example  (s  :  Finset  ℕ)  :  s.card  =  ∑  i  ∈  s,  1  :=  by  rw  [card_eq_sum_ones]

example  (s  :  Finset  ℕ)  :  s.card  =  ∑  i  ∈  s,  1  :=  by  simp 
```

下一节全部关于推理基数。

在形式化数学时，人们常常需要决定是否用集合或类型来表述自己的定义和定理。使用类型通常可以简化符号和证明，但处理类型的子集可能更加灵活。基于类型的有限集（finset）的类似物是**fintype**，即某个`α`的`Fintype α`类型。根据定义，fintype 仅仅是一个带有包含所有元素的有限集`univ`的数据类型。

```py
variable  {α  :  Type*}  [Fintype  α]

example  :  ∀  x  :  α,  x  ∈  Finset.univ  :=  by
  intro  x;  exact  mem_univ  x 
```

`Fintype.card α`等于相应有限集的基数。

```py
example  :  Fintype.card  α  =  (Finset.univ  :  Finset  α).card  :=  rfl 
```

我们已经看到了一个 fintype 的原型示例，即每个`n`的`Fin n`类型。Lean 认识到 fintype 在乘积运算等操作下是封闭的。

```py
example  :  Fintype.card  (Fin  5)  =  5  :=  by  simp
example  :  Fintype.card  ((Fin  5)  ×  (Fin  3))  =  15  :=  by  simp 
```

`Finset α`的任何元素`s`都可以强制转换为类型`(↑s : Finset α)`，即包含在`s`中的`α`的元素的子类型。

```py
variable  (s  :  Finset  ℕ)

example  :  (↑s  :  Type)  =  {x  :  ℕ  //  x  ∈  s}  :=  rfl
example  :  Fintype.card  ↑s  =  s.card  :=  by  simp 
```

Lean 和 Mathlib 使用**类型类推断**来跟踪 fintype 上的附加结构，即包含所有元素的通用有限集。换句话说，你可以将 fintype 视为一个带有额外数据的代数结构。第七章解释了这是如何工作的。

## 6.2\. 计数论证

计数事物的艺术是组合学的核心部分。Mathlib 包含几个用于计数有限集元素的基数的基本恒等式：

```py
open  Finset

variable  {α  β  :  Type*}  [DecidableEq  α]  [DecidableEq  β]  (s  t  :  Finset  α)  (f  :  α  →  β)

example  :  #(s  ×ˢ  t)  =  #s  *  #t  :=  by  rw  [card_product]
example  :  #(s  ×ˢ  t)  =  #s  *  #t  :=  by  simp

example  :  #(s  ∪  t)  =  #s  +  #t  -  #(s  ∩  t)  :=  by  rw  [card_union]

example  (h  :  Disjoint  s  t)  :  #(s  ∪  t)  =  #s  +  #t  :=  by  rw  [card_union_of_disjoint  h]
example  (h  :  Disjoint  s  t)  :  #(s  ∪  t)  =  #s  +  #t  :=  by  simp  [h]

example  (h  :  Function.Injective  f)  :  #(s.image  f)  =  #s  :=  by  rw  [card_image_of_injective  _  h]

example  (h  :  Set.InjOn  f  s)  :  #(s.image  f)  =  #s  :=  by  rw  [card_image_of_injOn  h] 
```

打开`Finset`命名空间允许我们使用`#s`表示`s.card`，以及使用缩短的名称如 card_union 等。

Mathlib 也可以计数 fintype 的元素：

```py
open  Fintype

variable  {α  β  :  Type*}  [Fintype  α]  [Fintype  β]

example  :  card  (α  ×  β)  =  card  α  *  card  β  :=  by  simp

example  :  card  (α  ⊕  β)  =  card  α  +  card  β  :=  by  simp

example  (n  :  ℕ)  :  card  (Fin  n  →  α)  =  (card  α)^n  :=  by  simp

variable  {n  :  ℕ}  {γ  :  Fin  n  →  Type*}  [∀  i,  Fintype  (γ  i)]

example  :  card  ((i  :  Fin  n)  →  γ  i)  =  ∏  i,  card  (γ  i)  :=  by  simp

example  :  card  (Σ  i,  γ  i)  =  ∑  i,  card  (γ  i)  :=  by  simp 
```

当`Fintype`命名空间未打开时，我们必须使用`Fintype.card`而不是 card。

以下是一个计算有限集合（finset）基数（即 n 范围与 n 范围的副本的并集，该副本已通过 n 进行偏移）的例子。计算需要证明并集中的两个集合是互斥的；证明的第一行产生了侧条件`Disjoint (range n) (image (fun i ↦ m + i) (range n))`，该条件在证明的末尾得到证实。`Disjoint`谓词过于通用，对我们直接有用，但定理`disjoint_iff_ne`将其置于我们可以使用的形式。

```py
#check  Disjoint

example  (m  n  :  ℕ)  (h  :  m  ≥  n)  :
  card  (range  n  ∪  (range  n).image  (fun  i  ↦  m  +  i))  =  2  *  n  :=  by
  rw  [card_union_of_disjoint,  card_range,  card_image_of_injective,  card_range];  omega
  .  apply  add_right_injective
  .  simp  [disjoint_iff_ne];  omega 
```

在本节中，`omega`将是我们处理算术计算和不等式的工作马。

这里有一个更有趣的例子。考虑由$\{0, \ldots, n\} \times \{0, \ldots, n\}$组成的子集，其中包含满足$i < j$的成对$(i, j)$。如果你把它们看作坐标平面中的格点，它们构成了一个以$(0, 0)$和$(n, n)$为顶点的正方形上方的三角形，不包括对角线。整个正方形的基数为$(n + 1)²$，减去对角线的大小并将结果除以二，我们得到三角形的基数为$n (n + 1) / 2$。

或者，我们注意到三角形的行的大小为$0, 1, \ldots, n$，因此基数是前$n$个自然数的和。下面证明的第一个`have`描述了三角形作为行的并集，其中行$j$由数字$0, 1, ..., j - 1$与$j$配对组成。在下面的证明中，符号`(., j)`是函数`fun i ↦ (i, j)`的缩写。其余的证明只是对有限集合基数进行计算。

```py
def  triangle  (n  :  ℕ)  :  Finset  (ℕ  ×  ℕ)  :=  {p  ∈  range  (n+1)  ×ˢ  range  (n+1)  |  p.1  <  p.2}

example  (n  :  ℕ)  :  #(triangle  n)  =  (n  +  1)  *  n  /  2  :=  by
  have  :  triangle  n  =  (range  (n+1)).biUnion  (fun  j  ↦  (range  j).image  (.,  j))  :=  by
  ext  p
  simp  only  [triangle,  mem_filter,  mem_product,  mem_range,  mem_biUnion,  mem_image]
  constructor
  .  rintro  ⟨⟨hp1,  hp2⟩,  hp3⟩
  use  p.2,  hp2,  p.1,  hp3
  .  rintro  ⟨p1,  hp1,  p2,  hp2,  rfl⟩
  omega
  rw  [this,  card_biUnion];  swap
  ·  -- take care of disjointness first
  intro  x  _  y  _  xney
  simp  [disjoint_iff_ne,  xney]
  -- continue the calculation
  transitivity  (∑  i  ∈  range  (n  +  1),  i)
  ·  congr;  ext  i
  rw  [card_image_of_injective,  card_range]
  intros  i1  i2;  simp
  rw  [sum_range_id];  rfl 
```

以下是对证明的变体，它使用有限类型（fintype）而不是有限集合（finset）进行计算。类型`α ≃ β`是`α`和`β`之间等价类的类型，由正向映射、反向映射以及证明这两个映射互为逆映射组成。证明中的第一个`have`表明`triangle n`与`Fin i`的并集等价，其中`i`在`Fin (n + 1)`范围内。有趣的是，正向函数和反向函数是用策略构建的，而不是明确写出。由于它们所做的只是移动数据和信息，`rfl`建立了它们是逆映射。

之后，`rw [←Fintype.card_coe]`将`#(triangle n)`重写为子类型`{ x // x ∈ triangle n }`的基数，其余的证明是一个计算过程。

```py
example  (n  :  ℕ)  :  #(triangle  n)  =  (n  +  1)  *  n  /  2  :=  by
  have  :  triangle  n  ≃  Σ  i  :  Fin  (n  +  1),  Fin  i.val  :=
  {  toFun  :=  by
  rintro  ⟨⟨i,  j⟩,  hp⟩
  simp  [triangle]  at  hp
  exact  ⟨⟨j,  hp.1.2⟩,  ⟨i,  hp.2⟩⟩
  invFun  :=  by
  rintro  ⟨i,  j⟩
  use  ⟨j,  i⟩
  simp  [triangle]
  exact  j.isLt.trans  i.isLt
  left_inv  :=  by  intro  i;  rfl
  right_inv  :=  by  intro  i;  rfl  }
  rw  [←Fintype.card_coe]
  trans;  apply  (Fintype.card_congr  this)
  rw  [Fintype.card_sigma,  sum_fin_eq_sum_range]
  convert  Finset.sum_range_id  (n  +  1)
  simp_all 
```

这里还有另一种方法。下面证明的第一行将问题简化为证明`2 * #(triangle n) = (n + 1) * n`。我们可以通过证明三角形的两个副本正好填满矩形`range n ×ˢ range (n + 1)`来实现这一点。作为一个练习，看看你是否能填补计算步骤。在解决方案中，我们在倒数第二步中广泛使用`omega`，但不幸的是，我们必须手动做相当多的工作。

```py
example  (n  :  ℕ)  :  #(triangle  n)  =  (n  +  1)  *  n  /  2  :=  by
  apply  Nat.eq_div_of_mul_eq_right  (by  norm_num)
  let  turn  (p  :  ℕ  ×  ℕ)  :  ℕ  ×  ℕ  :=  (n  -  1  -  p.1,  n  -  p.2)
  calc  2  *  #(triangle  n)
  =  #(triangle  n)  +  #(triangle  n)  :=  by
  sorry
  _  =  #(triangle  n)  +  #(triangle  n  |>.image  turn)  :=  by
  sorry
  _  =  #(range  n  ×ˢ  range  (n  +  1))  :=  by
  sorry
  _  =  (n  +  1)  *  n  :=  by
  sorry 
```

你可以自己验证，如果我们将 `n` 替换为 `n + 1` 并在 `triangle` 的定义中将 `<` 替换为 `≤`，我们会得到相同的三角形，只是向下移动了。下面的练习要求你使用这个事实来证明两个三角形具有相同的大小。

```py
def  triangle'  (n  :  ℕ)  :  Finset  (ℕ  ×  ℕ)  :=  {p  ∈  range  n  ×ˢ  range  n  |  p.1  ≤  p.2}

example  (n  :  ℕ)  :  #(triangle'  n)  =  #(triangle  n)  :=  by  sorry 
```

让我们以一个例子和一个练习来结束本节，这个例子和练习来自 Bhavik Mehta 在 2023 年于 *Lean for the Curious Mathematician* 上提供的组合学教程[教程](https://www.youtube.com/watch?v=_cJctOIXWE4&list=PLlF-CfQhukNn7xEbfL38eLgkveyk9_myQ&index=8&t=2737s&ab_channel=leanprovercommunity)。假设我们有一个二分图，其顶点集为 `s` 和 `t`，对于 `s` 中的每个 `a`，至少有三条边离开 `a`，而对于 `t` 中的每个 `b`，最多只有一条边进入 `b`。那么图中边的总数至少是 `s` 的三倍，最多是 `t` 的数量，从而得出三倍的 `s` 的数量最多是 `t` 的数量。以下定理实现了这个论证，其中我们使用关系 `r` 来表示图的边。证明是一个优雅的计算。

```py
open  Classical
variable  (s  t  :  Finset  Nat)  (a  b  :  Nat)

theorem  doubleCounting  {α  β  :  Type*}  (s  :  Finset  α)  (t  :  Finset  β)
  (r  :  α  →  β  →  Prop)
  (h_left  :  ∀  a  ∈  s,  3  ≤  #{b  ∈  t  |  r  a  b})
  (h_right  :  ∀  b  ∈  t,  #{a  ∈  s  |  r  a  b}  ≤  1)  :
  3  *  #(s)  ≤  #(t)  :=  by
  calc  3  *  #(s)
  =  ∑  a  ∈  s,  3  :=  by  simp  [sum_const_nat,  mul_comm]
  _  ≤  ∑  a  ∈  s,  #({b  ∈  t  |  r  a  b})  :=  sum_le_sum  h_left
  _  =  ∑  a  ∈  s,  ∑  b  ∈  t,  if  r  a  b  then  1  else  0  :=  by  simp
  _  =  ∑  b  ∈  t,  ∑  a  ∈  s,  if  r  a  b  then  1  else  0  :=  sum_comm
  _  =  ∑  b  ∈  t,  #({a  ∈  s  |  r  a  b})  :=  by  simp
  _  ≤  ∑  b  ∈  t,  1  :=  sum_le_sum  h_right
  _  ≤  #(t)  :=  by  simp 
```

以下练习也来自 Mehta 的教程。假设 `A` 是 `range (2 * n)` 的一个子集，包含 `n + 1` 个元素。很容易看出 `A` 必须包含两个连续的整数，因此包含两个互质的元素。如果你观看教程，你会看到在建立以下事实上花费了大量精力，现在这个事实由 `omega` 自动证明。这个事实是，`A` 必须包含两个连续的整数，因此包含两个互质的元素。

```py
example  (m  k  :  ℕ)  (h  :  m  ≠  k)  (h'  :  m  /  2  =  k  /  2)  :  m  =  k  +  1  ∨  k  =  m  +  1  :=  by  omega 
```

Mehta 练习的解决方案使用了鸽巢原理，形式为 `exists_lt_card_fiber_of_mul_lt_card_of_maps_to`，来证明在 `A` 中存在两个不同的元素 `m` 和 `k`，使得 `m / 2 = k / 2`。看看你是否能完成这个事实的证明，然后使用它来完成证明。

```py
example  {n  :  ℕ}  (A  :  Finset  ℕ)
  (hA  :  #(A)  =  n  +  1)
  (hA'  :  A  ⊆  range  (2  *  n))  :
  ∃  m  ∈  A,  ∃  k  ∈  A,  Nat.Coprime  m  k  :=  by
  have  :  ∃  t  ∈  range  n,  1  <  #({u  ∈  A  |  u  /  2  =  t})  :=  by
  apply  exists_lt_card_fiber_of_mul_lt_card_of_maps_to
  ·  sorry
  ·  sorry
  rcases  this  with  ⟨t,  ht,  ht'⟩
  simp  only  [one_lt_card,  mem_filter]  at  ht'
  sorry 
```

## 6.3. 递归定义的类型

Lean 的基础允许我们定义递归类型，即从底部向上生成实例的数据类型。例如，由 `α` 的元素组成的列表数据类型 `List α` 是从空列表 `nil` 开始，并依次向列表前添加元素生成的。下面我们将定义一个二叉树类型 `BinTree`，其元素是从空树开始，通过将新节点附加到两个现有树来构建新树。

在 Lean 中，可以定义对象无限递归类型，例如可数分支良好基础树。在离散数学中，有限递归定义通常被使用，尤其是在与计算机科学相关的离散数学分支中。Lean 不仅提供了定义此类类型的手段，还提供了归纳原理和递归定义的原则。例如，数据类型 `List α` 是通过归纳定义的：

```py
namespace  MyListSpace

inductive  List  (α  :  Type*)  where
  |  nil  :  List  α
  |  cons  :  α  →  List  α  →  List  α

end  MyListSpace 
```

归纳定义说明 `List α` 的每个元素要么是 `nil`，即空列表，要么是 `cons a as`，其中 `a` 是 `α` 的一个元素，`as` 是 `α` 元素列表。构造函数被适当地命名为 `List.nil` 和 `List.cons`，但如果你在 `List` 命名空间是开放的，你可以使用简短的表示法。当 `List` 命名空间不是开放的，你可以在 Lean 需要列表的地方写 `.nil` 和 `.cons a as`，Lean 会自动插入 `List` 限定符。在本节中，我们将临时定义放在单独的命名空间中，如 `MyListSpace`，以避免与标准库冲突。在临时命名空间之外，我们将恢复使用标准库定义。

Lean 定义了 `[]` 符号表示 `nil` 和 `::` 符号表示 `cons`，你可以用 `[a, b, c]` 来表示 `a :: b :: c :: []`。`append` 和 `map` 函数被递归地定义为如下：

```py
def  append  {α  :  Type*}  :  List  α  →  List  α  →  List  α
  |  [],  bs  =>  bs
  |  a  ::  as,  bs  =>  a  ::  (append  as  bs)

def  map  {α  β  :  Type*}  (f  :  α  →  β)  :  List  α  →  List  β
  |  []  =>  []
  |  a  ::  as  =>  f  a  ::  map  f  as

#eval  append  [1,  2,  3]  [4,  5,  6]
#eval  map  (fun  n  =>  n²)  [1,  2,  3,  4,  5] 
```

注意，这里有一个基本情况和一个递归情况。在每种情况下，两个定义子句都是定义上成立的：

```py
theorem  nil_append  {α  :  Type*}  (as  :  List  α)  :  append  []  as  =  as  :=  rfl

theorem  cons_append  {α  :  Type*}  (a  :  α)  (as  :  List  α)  (bs  :  List  α)  :
  append  (a  ::  as)  bs  =  a  ::  (append  as  bs)  :=  rfl

theorem  map_nil  {α  β  :  Type*}  (f  :  α  →  β)  :  map  f  []  =  []  :=  rfl

theorem  map_cons  {α  β  :  Type*}  (f  :  α  →  β)  (a  :  α)  (as  :  List  α)  :
  map  f  (a  ::  as)  =  f  a  ::  map  f  as  :=  rfl 
```

`append` 和 `map` 函数定义在标准库中，`append as bs` 可以写成 `as ++ bs`。

Lean 允许你按照定义的结构来编写归纳证明。

```py
variable  {α  β  γ  :  Type*}
variable  (as  bs  cs  :  List  α)
variable  (a  b  c  :  α)

open  List

theorem  append_nil  :  ∀  as  :  List  α,  as  ++  []  =  as
  |  []  =>  rfl
  |  a  ::  as  =>  by  rw  [cons_append,  append_nil  as]

theorem  map_map  (f  :  α  →  β)  (g  :  β  →  γ)  :
  ∀  as  :  List  α,  map  g  (map  f  as)  =  map  (g  ∘  f)  as
  |  []  =>  rfl
  |  a  ::  as  =>  by  rw  [map_cons,  map_cons,  map_cons,  map_map  f  g  as];  rfl 
```

你也可以使用 `induction'` 策略。

当然，这些定理已经包含在标准库中。作为一个练习，尝试在 `MyListSpace3` 命名空间中定义一个 `reverse` 函数（以避免与标准库中的 `List.reverse` 冲突），该函数可以反转一个列表。你可以使用 `#eval reverse [1, 2, 3, 4, 5]` 来测试它。`reverse` 的最直接定义需要二次时间复杂度，但不用担心这一点。你可以跳转到标准库中 `List.reverse` 的定义，以查看线性时间复杂度的实现。尝试证明 `reverse (as ++ bs) = reverse bs ++ reverse as` 和 `reverse (reverse as) = as`。你可以使用 `cons_append` 和 `append_assoc`，但你可能需要提出辅助引理并证明它们。

```py
def  reverse  :  List  α  →  List  α  :=  sorry

theorem  reverse_append  (as  bs  :  List  α)  :  reverse  (as  ++  bs)  =  reverse  bs  ++  reverse  as  :=  by
  sorry

theorem  reverse_reverse  (as  :  List  α)  :  reverse  (reverse  as)  =  as  :=  by  sorry 
```

作为另一个例子，考虑以下二叉树的归纳定义及其计算二叉树大小和深度的函数。

```py
inductive  BinTree  where
  |  empty  :  BinTree
  |  node  :  BinTree  →  BinTree  →  BinTree

namespace  BinTree

def  size  :  BinTree  →  ℕ
  |  empty  =>  0
  |  node  l  r  =>  size  l  +  size  r  +  1

def  depth  :  BinTree  →  ℕ
  |  empty  =>  0
  |  node  l  r  =>  max  (depth  l)  (depth  r)  +  1 
```

计算空二叉树的大小为 0 和深度为 0 是方便的。在文献中，这种数据类型有时被称为 *扩展二叉树*。包括空树意味着，例如，我们可以定义由根节点、空左子树和由单个节点组成的右子树构成的树 `node empty (node empty empty)`。

这里是一个重要的关于大小和深度的不等式：

```py
theorem  size_le  :  ∀  t  :  BinTree,  size  t  ≤  2^depth  t  -  1
  |  empty  =>  Nat.zero_le  _
  |  node  l  r  =>  by
  simp  only  [depth,  size]
  calc  l.size  +  r.size  +  1
  ≤  (2^l.depth  -  1)  +  (2^r.depth  -  1)  +  1  :=  by
  gcongr  <;>  apply  size_le
  _  ≤  (2  ^  max  l.depth  r.depth  -  1)  +  (2  ^  max  l.depth  r.depth  -  1)  +  1  :=  by
  gcongr  <;>  simp
  _  ≤  2  ^  (max  l.depth  r.depth  +  1)  -  1  :=  by
  have  :  0  <  2  ^  max  l.depth  r.depth  :=  by  simp
  omega 
```

尝试证明以下不等式，这相对容易一些。记住，如果你像上一个定理那样进行归纳证明，你必须删除 `:= by`。

```py
theorem  depth_le_size  :  ∀  t  :  BinTree,  depth  t  ≤  size  t  :=  by  sorry 
```

还定义了二叉树上的 `flip` 操作，该操作递归地交换左右子树。

```py
def  flip  :  BinTree  →  BinTree  :=  sorry 
```

如果你做得正确，以下证明应该是 rfl。

```py
example:  flip  (node  (node  empty  (node  empty  empty))  (node  empty  empty))  =
  node  (node  empty  empty)  (node  (node  empty  empty)  empty)  :=  sorry 
```

证明以下：

```py
theorem  size_flip  :  ∀  t,  size  (flip  t)  =  size  t  :=  by  sorry 
```

我们以一些形式逻辑结束本节。以下是对命题公式的归纳定义。

```py
inductive  PropForm  :  Type  where
  |  var  (n  :  ℕ)  :  PropForm
  |  fls  :  PropForm
  |  conj  (A  B  :  PropForm)  :  PropForm
  |  disj  (A  B  :  PropForm)  :  PropForm
  |  impl  (A  B  :  PropForm)  :  PropForm 
```

每个命题公式要么是一个变量 `var n`，要么是常假 `fls`，要么是形式为 `conj A B`、`disj A B` 或 `impl A B` 的复合公式。用常规数学符号，这些通常分别写成 $p_n$、$\bot$、$A \wedge B$、$A \vee B$ 和 $A \to B$。其他命题连接词可以用这些来定义；例如，我们可以将 $\neg A$ 定义为 $A \to \bot$，将 $A \leftrightarrow B$ 定义为 $(A \to B) \wedge (B \to A)$。

定义了命题公式的数据类型后，我们定义了相对于将布尔真值赋给变量的赋值 `v` 评估命题公式意味着什么。

```py
def  eval  :  PropForm  →  (ℕ  →  Bool)  →  Bool
  |  var  n,  v  =>  v  n
  |  fls,  _  =>  false
  |  conj  A  B,  v  =>  A.eval  v  &&  B.eval  v
  |  disj  A  B,  v  =>  A.eval  v  ||  B.eval  v
  |  impl  A  B,  v  =>  !  A.eval  v  ||  B.eval  v 
```

下一个定义指定了公式中出现的变量集合，接下来的定理表明，在两个在变量上达成一致的真值赋值上评估公式会产生相同的值。

```py
def  vars  :  PropForm  →  Finset  ℕ
  |  var  n  =>  {n}
  |  fls  =>  ∅
  |  conj  A  B  =>  A.vars  ∪  B.vars
  |  disj  A  B  =>  A.vars  ∪  B.vars
  |  impl  A  B  =>  A.vars  ∪  B.vars

theorem  eval_eq_eval  :  ∀  (A  :  PropForm)  (v1  v2  :  ℕ  →  Bool),
  (∀  n  ∈  A.vars,  v1  n  =  v2  n)  →  A.eval  v1  =  A.eval  v2
  |  var  n,  v1,  v2,  h  =>  by  simp_all  [vars,  eval,  h]
  |  fls,  v1,  v2,  h  =>  by  simp_all  [eval]
  |  conj  A  B,  v1,  v2,  h  =>  by
  simp_all  [vars,  eval,  eval_eq_eval  A  v1  v2,  eval_eq_eval  B  v1  v2]
  |  disj  A  B,  v1,  v2,  h  =>  by
  simp_all  [vars,  eval,  eval_eq_eval  A  v1  v2,  eval_eq_eval  B  v1  v2]
  |  impl  A  B,  v1,  v2,  h  =>  by
  simp_all  [vars,  eval,  eval_eq_eval  A  v1  v2,  eval_eq_eval  B  v1  v2] 
```

注意到重复，我们可以巧妙地使用自动化。

```py
theorem  eval_eq_eval'  (A  :  PropForm)  (v1  v2  :  ℕ  →  Bool)  (h  :  ∀  n  ∈  A.vars,  v1  n  =  v2  n)  :
  A.eval  v1  =  A.eval  v2  :=  by
  cases  A  <;>  simp_all  [eval,  vars,  fun  A  =>  eval_eq_eval'  A  v1  v2] 
```

函数 `subst A m C` 描述了将公式 `C` 替换到公式 `A` 中每个 `var m` 出现的结果。

```py
def  subst  :  PropForm  →  ℕ  →  PropForm  →  PropForm
  |  var  n,  m,  C  =>  if  n  =  m  then  C  else  var  n
  |  fls,  _,  _  =>  fls
  |  conj  A  B,  m,  C  =>  conj  (A.subst  m  C)  (B.subst  m  C)
  |  disj  A  B,  m,  C  =>  disj  (A.subst  m  C)  (B.subst  m  C)
  |  impl  A  B,  m,  C  =>  impl  (A.subst  m  C)  (B.subst  m  C) 
```

例如，展示替换一个在公式中未出现的变量没有任何效果：

```py
theorem  subst_eq_of_not_mem_vars  :
  ∀  (A  :  PropForm)  (n  :  ℕ)  (C  :  PropForm),  n  ∉  A.vars  →  A.subst  n  C  =  A  :=  sorry 
```

以下定理提出了更微妙和有趣的观点：在真值赋值 `v` 上评估 `A.subst n C` 与在将 `C` 的值赋给变量 `var n` 的真值赋值上评估 `A` 是相同的。看看你是否能证明它。

```py
theorem  subst_eval_eq  :  ∀  (A  :  PropForm)  (n  :  ℕ)  (C  :  PropForm)  (v  :  ℕ  →  Bool),
  (A.subst  n  C).eval  v  =  A.eval  (fun  m  =>  if  m  =  n  then  C.eval  v  else  v  m)  :=  sorry 
```*
