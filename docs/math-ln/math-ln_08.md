# 8. 层次结构

> 原文：[`leanprover-community.github.io/mathematics_in_lean/C08_Hierarchies.html`](https://leanprover-community.github.io/mathematics_in_lean/C08_Hierarchies.html)

*Lean 中的数学* **   8. 层次结构

+   查看页面源代码

* * *

我们在第七章中看到了如何定义群类并构建这个类的实例，然后是如何构建交换环类的实例。但当然这里有一个层次结构：交换环特别是一个加法群。在本章中，我们将研究如何构建这样的层次结构。它们出现在数学的所有分支中，但本章的重点将放在代数示例上。

在更多关于使用现有层次结构的讨论之前讨论如何构建层次结构可能显得有些过早。但是，为了使用这些层次结构，需要对支撑这些层次结构的技术有一定的了解。因此，你可能仍然应该阅读这一章，但不要在第一次阅读时过于努力地记住所有内容，然后阅读接下来的章节，再回来这里进行第二次阅读。

在本章中，我们将重新定义（Mathlib 中出现的许多事物的简化版本），因此我们将使用索引来区分我们的版本。例如，我们将有 `Ring₁` 作为我们的 `Ring` 版本。由于我们将逐渐解释更强大的形式化结构的方法，这些索引有时会超过一个。

## 8.1. 基础

在所有层次结构的底层，我们找到了携带数据的类。以下类记录了给定的类型 `α` 被赋予了一个称为 `one` 的特殊元素。在这个阶段，它没有任何属性。

```py
class  One₁  (α  :  Type)  where
  /-- The element one -/
  one  :  α 
```

由于在本章中我们将更频繁地使用类，我们需要了解一些关于 `class` 命令所做事情的更多细节。首先，上面的 `class` 命令定义了一个带有参数 `α : Type` 和单个字段 `one` 的结构 `One₁`。它还标记这个结构为一个类，以便对于某些类型 `α` 的 `One₁ α` 参数，只要它们被标记为实例隐式，即出现在方括号之间，就可以使用实例解析过程进行推断。这两个效果也可以通过带有 `class` 属性的 `structure` 命令实现，即写作 `@[class] structure` 实例的 `class`。但是，类命令还确保 `One₁ α` 在其自己的字段中作为实例隐式参数出现。比较：

```py
#check  One₁.one  -- One₁.one {α : Type} [self : One₁ α] : α

@[class]  structure  One₂  (α  :  Type)  where
  /-- The element one -/
  one  :  α

#check  One₂.one 
```

在第二次检查中，我们可以看到 `self : One₂ α` 是一个显式参数。让我们确保第一个版本确实可以在没有任何显式参数的情况下使用。

```py
example  (α  :  Type)  [One₁  α]  :  α  :=  One₁.one 
```

备注：在上面的例子中，参数`One₁ α`被标记为实例隐式，这有点愚蠢，因为这只影响声明的*使用*和由`example`命令创建的声明不能使用。然而，它允许我们避免给这个参数命名，更重要的是，它开始养成标记`One₁ α`参数为实例隐式的良好习惯。

另一个要注意的是，所有这些只有在 Lean 知道`α`是什么的时候才会起作用。在上面的例子中，省略类型注解`: α`会产生一个错误信息，例如：`typeclass instance problem is stuck, it is often due to metavariables One₁ (?m.263 α)`，其中`?m.263 α`意味着“依赖于`α`的某些类型”（而 263 是一个自动生成的索引，可以用来区分几个未知的事物）。避免这个问题的另一种方法就是使用类型注解，如下所示：

```py
example  (α  :  Type)  [One₁  α]  :=  (One₁.one  :  α) 
```

你可能在第 3.6 节中玩序列的极限时已经遇到过这个问题，如果你试图声明例如`0 < 1`，但没有告诉 Lean 你是指自然数还是实数的这个不等式。

我们下一个任务是给`One₁.one`分配一个符号。由于我们不希望与内置的`1`的符号冲突，我们将使用`𝟙`。这是通过以下命令实现的，其中第一行告诉 Lean 使用`One₁.one`的文档作为符号`𝟙`的文档。

```py
@[inherit_doc]
notation  "𝟙"  =>  One₁.one

example  {α  :  Type}  [One₁  α]  :  α  :=  𝟙

example  {α  :  Type}  [One₁  α]  :  (𝟙  :  α)  =  𝟙  :=  rfl 
```

我们现在想要一个携带数据的类来记录二元操作。我们目前不想在加法和乘法之间做出选择，所以我们将使用菱形。

```py
class  Dia₁  (α  :  Type)  where
  dia  :  α  →  α  →  α

infixl:70  " ⋄ "  =>  Dia₁.dia 
```

就像`One₁`的例子一样，在这个阶段操作没有任何属性。现在让我们定义一个半群结构类，其中操作用`⋄`表示。现在我们手动定义它为一个有两个字段的结构，一个`Dia₁`实例和一些`Prop`类型的字段`dia_assoc`，断言`⋄`的结合性。

```py
class  Semigroup₀  (α  :  Type)  where
  toDia₁  :  Dia₁  α
  /-- Diamond is associative -/
  dia_assoc  :  ∀  a  b  c  :  α,  a  ⋄  b  ⋄  c  =  a  ⋄  (b  ⋄  c) 
```

注意，在声明 dia_assoc 时，之前定义的字段 toDia₁在局部上下文中，因此可以在 Lean 搜索 Dia₁ α的实例以理解⋄ b 时使用。然而，这个 toDia₁字段不会成为类型类实例数据库的一部分。因此，执行`example {α : Type} [Semigroup₁ α] (a b : α) : α := a ⋄ b`会失败，错误信息为`failed to synthesize instance Dia₁ α`。

我们可以通过稍后添加`instance`属性来解决这个问题。

```py
attribute  [instance]  Semigroup₀.toDia₁

example  {α  :  Type}  [Semigroup₀  α]  (a  b  :  α)  :  α  :=  a  ⋄  b 
```

在构建之前，我们需要使用不同的语法来添加这个 toDia₁字段，告诉 Lean 将 Dia₁ α视为如果它的字段是 Semigroup₁本身的字段。这也方便地自动添加了 toDia₁实例。`class`命令通过使用`extends`语法支持这一点，如下所示：

```py
class  Semigroup₁  (α  :  Type)  extends  toDia₁  :  Dia₁  α  where
  /-- Diamond is associative -/
  dia_assoc  :  ∀  a  b  c  :  α,  a  ⋄  b  ⋄  c  =  a  ⋄  (b  ⋄  c)

example  {α  :  Type}  [Semigroup₁  α]  (a  b  :  α)  :  α  :=  a  ⋄  b 
```

注意，这种语法在`structure`命令中也是可用的，尽管在这种情况下，它只解决了编写像 toDia₁这样的字段的问题，因为在这种情况下没有实例要定义。

在`extends`语法中，字段名`toDia₁`是可选的。默认情况下，它采用被扩展的类的名称，并在其前面加上“to”。

```py
class  Semigroup₂  (α  :  Type)  extends  Dia₁  α  where
  /-- Diamond is associative -/
  dia_assoc  :  ∀  a  b  c  :  α,  a  ⋄  b  ⋄  c  =  a  ⋄  (b  ⋄  c) 
```

让我们现在尝试组合一个菱形操作和一个特殊元素，并使用断言这个元素在两侧都是中性的公理。

```py
class  DiaOneClass₁  (α  :  Type)  extends  One₁  α,  Dia₁  α  where
  /-- One is a left neutral element for diamond. -/
  one_dia  :  ∀  a  :  α,  𝟙  ⋄  a  =  a
  /-- One is a right neutral element for diamond -/
  dia_one  :  ∀  a  :  α,  a  ⋄  𝟙  =  a 
```

在下一个例子中，我们告诉 Lean `α`有一个`DiaOneClass₁`结构，并陈述了一个使用 Dia₁实例和 One₁实例的属性。为了看到 Lean 如何找到这些实例，我们设置了一个跟踪选项，其结果可以在 Infoview 中查看。默认情况下，这个结果相当简洁，但可以通过单击以黑色箭头结尾的行来展开。它包括 Lean 在成功之前尝试找到实例的失败尝试。成功的尝试确实涉及由`extends`语法生成的实例。

```py
set_option  trace.Meta.synthInstance  true  in
example  {α  :  Type}  [DiaOneClass₁  α]  (a  b  :  α)  :  Prop  :=  a  ⋄  b  =  𝟙 
```

注意，我们不需要在组合现有类时包含额外的字段。因此，我们可以定义幺半群如下：

```py
class  Monoid₁  (α  :  Type)  extends  Semigroup₁  α,  DiaOneClass₁  α 
```

虽然上述定义看起来很直接，但它隐藏了一个重要的微妙之处。`Semigroup₁ α`和`DiaOneClass₁ α`都扩展了`Dia₁ α`，因此人们可能会担心，有一个`Monoid₁ α`实例会在`α`上给出两个无关的菱形操作，一个来自字段`Monoid₁.toSemigroup₁`，另一个来自字段`Monoid₁.toDiaOneClass₁`。

事实上，如果我们尝试手动使用以下方式构建一个幺半群类：

```py
class  Monoid₂  (α  :  Type)  where
  toSemigroup₁  :  Semigroup₁  α
  toDiaOneClass₁  :  DiaOneClass₁  α 
```

那么，我们会得到两个完全无关的菱形操作`Monoid₂.toSemigroup₁.toDia₁.dia`和`Monoid₂.toDiaOneClass₁.toDia₁.dia`。

使用`extends`语法生成的版本没有这个缺陷。

```py
example  {α  :  Type}  [Monoid₁  α]  :
  (Monoid₁.toSemigroup₁.toDia₁.dia  :  α  →  α  →  α)  =  Monoid₁.toDiaOneClass₁.toDia₁.dia  :=  rfl 
```

因此，`class`命令为我们做了一些魔法（`structure`命令也会这样做）。查看我们类字段的一个简单方法就是检查它们的构造函数。比较：

```py
/- Monoid₂.mk {α : Type} (toSemigroup₁ : Semigroup₁ α) (toDiaOneClass₁ : DiaOneClass₁ α) : Monoid₂ α -/
#check  Monoid₂.mk

/- Monoid₁.mk {α : Type} [toSemigroup₁ : Semigroup₁ α] [toOne₁ : One₁ α] (one_dia : ∀ (a : α), 𝟙 ⋄ a = a) (dia_one : ∀ (a : α), a ⋄ 𝟙 = a) : Monoid₁ α -/
#check  Monoid₁.mk 
```

因此，我们看到`Monoid₁`期望接受`Semigroup₁ α`参数，但然后它不会接受一个潜在的冲突`DiaOneClass₁ α`参数，而是将其拆分并仅包含非重叠部分。它还自动生成了一个实例`Monoid₁.toDiaOneClass₁`，这不是一个字段，但它具有预期的签名，从最终用户的角度来看，它恢复了两个扩展类`Semigroup₁`和`DiaOneClass₁`之间的对称性。

```py
#check  Monoid₁.toSemigroup₁
#check  Monoid₁.toDiaOneClass₁ 
```

我们现在非常接近定义群。我们可以在幺半群结构中添加一个字段，断言每个元素都存在逆元。但那样的话，我们就需要努力访问这些逆元。在实践中，将其作为数据添加更为方便。为了优化可重用性，我们定义了一个新的数据承载类，然后给它一些符号。

```py
class  Inv₁  (α  :  Type)  where
  /-- The inversion function -/
  inv  :  α  →  α

@[inherit_doc]
postfix:max  "⁻¹"  =>  Inv₁.inv

class  Group₁  (G  :  Type)  extends  Monoid₁  G,  Inv₁  G  where
  inv_dia  :  ∀  a  :  G,  a⁻¹  ⋄  a  =  𝟙 
```

上述定义可能看起来太弱了，我们只要求`a⁻¹`是`a`的左逆。但另一方面是自动的。为了证明这一点，我们需要一个初步的引理。

```py
lemma  left_inv_eq_right_inv₁  {M  :  Type}  [Monoid₁  M]  {a  b  c  :  M}  (hba  :  b  ⋄  a  =  𝟙)  (hac  :  a  ⋄  c  =  𝟙)  :  b  =  c  :=  by
  rw  [←  DiaOneClass₁.one_dia  c,  ←  hba,  Semigroup₁.dia_assoc,  hac,  DiaOneClass₁.dia_one  b] 
```

在这个引理中，给出全名相当令人烦恼，尤其是当它需要知道哪个层次结构提供了这些事实时。解决这个问题的一种方法是通过使用`export`命令将那些事实作为根命名空间中的引理来复制。

```py
export  DiaOneClass₁  (one_dia  dia_one)
export  Semigroup₁  (dia_assoc)
export  Group₁  (inv_dia) 
```

然后，我们可以将上面的证明重写为：

```py
example  {M  :  Type}  [Monoid₁  M]  {a  b  c  :  M}  (hba  :  b  ⋄  a  =  𝟙)  (hac  :  a  ⋄  c  =  𝟙)  :  b  =  c  :=  by
  rw  [←  one_dia  c,  ←  hba,  dia_assoc,  hac,  dia_one  b] 
```

现在轮到你来证明关于我们的代数结构的事情了。

```py
lemma  inv_eq_of_dia  [Group₁  G]  {a  b  :  G}  (h  :  a  ⋄  b  =  𝟙)  :  a⁻¹  =  b  :=
  sorry

lemma  dia_inv  [Group₁  G]  (a  :  G)  :  a  ⋄  a⁻¹  =  𝟙  :=
  sorry 
```

在这个阶段，我们希望继续定义环，但有一个严重的问题。一个类型上的环结构包含一个加法群结构和乘法幺半群结构，以及关于它们之间相互作用的一些性质。但到目前为止，我们为所有操作硬编码了一个符号 `⋄`。更基本的是，类型类系统假设每个类型只有一个类型类的实例。有各种方法可以解决这个问题。令人惊讶的是，Mathlib 使用了复制加法和乘法理论的原始想法，这得益于一些代码生成属性。结构和类都在加法和乘法符号中定义，并通过 `to_additive` 属性将它们链接起来。在像半群这样的多重继承的情况下，自动生成的“对称恢复”实例也需要标记。这有点技术性；你不需要理解细节。重要的是，引理只以乘法符号表示，并标记为 `to_additive` 以生成加法版本 `left_inv_eq_right_inv'` 及其自动生成的加法版本 `left_neg_eq_right_neg'`。为了检查这个加法版本的名字，我们在 `left_inv_eq_right_inv'` 的顶部使用了 `whatsnew in` 命令。

```py
class  AddSemigroup₃  (α  :  Type)  extends  Add  α  where
  /-- Addition is associative -/
  add_assoc₃  :  ∀  a  b  c  :  α,  a  +  b  +  c  =  a  +  (b  +  c)

@[to_additive  AddSemigroup₃]
class  Semigroup₃  (α  :  Type)  extends  Mul  α  where
  /-- Multiplication is associative -/
  mul_assoc₃  :  ∀  a  b  c  :  α,  a  *  b  *  c  =  a  *  (b  *  c)

class  AddMonoid₃  (α  :  Type)  extends  AddSemigroup₃  α,  AddZeroClass  α

@[to_additive  AddMonoid₃]
class  Monoid₃  (α  :  Type)  extends  Semigroup₃  α,  MulOneClass  α

export  Semigroup₃  (mul_assoc₃)
export  AddSemigroup₃  (add_assoc₃)

whatsnew  in
@[to_additive]
lemma  left_inv_eq_right_inv'  {M  :  Type}  [Monoid₃  M]  {a  b  c  :  M}  (hba  :  b  *  a  =  1)  (hac  :  a  *  c  =  1)  :  b  =  c  :=  by
  rw  [←  one_mul  c,  ←  hba,  mul_assoc₃,  hac,  mul_one  b]

#check  left_neg_eq_right_neg' 
```

配备了这项技术，我们也可以轻松地定义交换半群、幺半群和群，然后定义环。

```py
class  AddCommSemigroup₃  (α  :  Type)  extends  AddSemigroup₃  α  where
  add_comm  :  ∀  a  b  :  α,  a  +  b  =  b  +  a

@[to_additive  AddCommSemigroup₃]
class  CommSemigroup₃  (α  :  Type)  extends  Semigroup₃  α  where
  mul_comm  :  ∀  a  b  :  α,  a  *  b  =  b  *  a

class  AddCommMonoid₃  (α  :  Type)  extends  AddMonoid₃  α,  AddCommSemigroup₃  α

@[to_additive  AddCommMonoid₃]
class  CommMonoid₃  (α  :  Type)  extends  Monoid₃  α,  CommSemigroup₃  α

class  AddGroup₃  (G  :  Type)  extends  AddMonoid₃  G,  Neg  G  where
  neg_add  :  ∀  a  :  G,  -a  +  a  =  0

@[to_additive  AddGroup₃]
class  Group₃  (G  :  Type)  extends  Monoid₃  G,  Inv  G  where
  inv_mul  :  ∀  a  :  G,  a⁻¹  *  a  =  1 
```

我们应该记住在适当的时候用 `simp` 标记引理。

```py
attribute  [simp]  Group₃.inv_mul  AddGroup₃.neg_add 
```

然后，由于我们转向了标准符号，我们需要重复说明一点，但至少 `to_additive` 执行了将乘法符号转换为加法符号的转换工作。

```py
@[to_additive]
lemma  inv_eq_of_mul  [Group₃  G]  {a  b  :  G}  (h  :  a  *  b  =  1)  :  a⁻¹  =  b  :=
  sorry 
```

注意，`to_additive` 可以被要求为引理标记 `simp` 并将此属性传播到加法版本，如下所示。

```py
@[to_additive  (attr  :=  simp)]
lemma  Group₃.mul_inv  {G  :  Type}  [Group₃  G]  {a  :  G}  :  a  *  a⁻¹  =  1  :=  by
  sorry

@[to_additive]
lemma  mul_left_cancel₃  {G  :  Type}  [Group₃  G]  {a  b  c  :  G}  (h  :  a  *  b  =  a  *  c)  :  b  =  c  :=  by
  sorry

@[to_additive]
lemma  mul_right_cancel₃  {G  :  Type}  [Group₃  G]  {a  b  c  :  G}  (h  :  b*a  =  c*a)  :  b  =  c  :=  by
  sorry

class  AddCommGroup₃  (G  :  Type)  extends  AddGroup₃  G,  AddCommMonoid₃  G

@[to_additive  AddCommGroup₃]
class  CommGroup₃  (G  :  Type)  extends  Group₃  G,  CommMonoid₃  G 
```

现在，我们已经准备好定义环了。为了演示目的，我们不会假设加法是交换的，然后立即提供一个 `AddCommGroup₃` 的实例。Mathlib 不玩这种游戏，首先是因为在实践中这并不会使任何环实例更容易，而且 Mathlib 的代数层次结构通过半环，这些半环类似于环但没有相反元素，因此下面的证明对它们不适用。我们在这里获得的好处，除了如果你从未见过它，它是一个很好的练习之外，还有一个使用允许将父结构作为实例参数提供的语法构建实例的例子，然后提供额外的字段。这里 Ring₃ R 参数为 AddCommGroup₃ R 提供了所有需要的东西，除了 add_comm。

```py
class  Ring₃  (R  :  Type)  extends  AddGroup₃  R,  Monoid₃  R,  MulZeroClass  R  where
  /-- Multiplication is left distributive over addition -/
  left_distrib  :  ∀  a  b  c  :  R,  a  *  (b  +  c)  =  a  *  b  +  a  *  c
  /-- Multiplication is right distributive over addition -/
  right_distrib  :  ∀  a  b  c  :  R,  (a  +  b)  *  c  =  a  *  c  +  b  *  c

instance  {R  :  Type}  [Ring₃  R]  :  AddCommGroup₃  R  :=
{  add_comm  :=  by
  sorry  } 
```

当然，我们也可以构建具体的实例，例如整数上的环结构（当然下面的实例使用的是 Mathlib 中已经完成的所有工作）。

```py
instance  :  Ring₃  ℤ  where
  add  :=  (·  +  ·)
  add_assoc₃  :=  add_assoc
  zero  :=  0
  zero_add  :=  by  simp
  add_zero  :=  by  simp
  neg  :=  (-  ·)
  neg_add  :=  by  simp
  mul  :=  (·  *  ·)
  mul_assoc₃  :=  mul_assoc
  one  :=  1
  one_mul  :=  by  simp
  mul_one  :=  by  simp
  zero_mul  :=  by  simp
  mul_zero  :=  by  simp
  left_distrib  :=  Int.mul_add
  right_distrib  :=  Int.add_mul 
```

作为练习，你现在可以设置一个简单的层次结构，包括一个有序交换幺半群类，它具有部分序和交换幺半群结构，使得`∀ a b : α, a ≤ b → ∀ c : α, c * a ≤ c * b`。当然，你需要为以下类添加字段和可能`extends`子句。

```py
class  LE₁  (α  :  Type)  where
  /-- The Less-or-Equal relation. -/
  le  :  α  →  α  →  Prop

@[inherit_doc]  infix:50  " ≤₁ "  =>  LE₁.le

class  Preorder₁  (α  :  Type)

class  PartialOrder₁  (α  :  Type)

class  OrderedCommMonoid₁  (α  :  Type)

instance  :  OrderedCommMonoid₁  ℕ  where 
```

现在，我们想要讨论涉及多个类型的代数结构。一个典型的例子是环上的模块。如果你不知道什么是模块，你可以假装它意味着向量空间，并认为我们所有的环都是域。这些结构是带有某些环元素的标量乘法的交换加法群。

我们首先定义由某些类型`α`在某个类型`β`上的一些类型`α`定义的标量乘法的携带数据类型类，并给它一个右结合的符号。

```py
class  SMul₃  (α  :  Type)  (β  :  Type)  where
  /-- Scalar multiplication -/
  smul  :  α  →  β  →  β

infixr:73  " • "  =>  SMul₃.smul 
```

然后，我们可以定义模块（如果你不知道什么是模块，可以再次考虑向量空间）。

```py
class  Module₁  (R  :  Type)  [Ring₃  R]  (M  :  Type)  [AddCommGroup₃  M]  extends  SMul₃  R  M  where
  zero_smul  :  ∀  m  :  M,  (0  :  R)  •  m  =  0
  one_smul  :  ∀  m  :  M,  (1  :  R)  •  m  =  m
  mul_smul  :  ∀  (a  b  :  R)  (m  :  M),  (a  *  b)  •  m  =  a  •  b  •  m
  add_smul  :  ∀  (a  b  :  R)  (m  :  M),  (a  +  b)  •  m  =  a  •  m  +  b  •  m
  smul_add  :  ∀  (a  :  R)  (m  n  :  M),  a  •  (m  +  n)  =  a  •  m  +  a  •  n 
```

这里有一些有趣的事情正在发生。虽然在这个定义中，`R`上的环结构是一个参数并不太令人惊讶，但你可能预计`AddCommGroup₃ M`将像`SMul₃ R M`一样成为`extends`子句的一部分。尝试这样做会导致一个听起来很神秘的错误信息：“无法为实例 Module₁.toAddCommGroup₃找到合成顺序，类型为(R : Type) → [inst : Ring₃ R] → {M : Type} → [self : Module₁ R M] → AddCommGroup₃ M，所有剩余的参数都有元变量：Ring₃ ?R @Module₁ ?R ?inst✝ M”。为了理解这条信息，你需要记住，这样的`extends`子句会导致一个标记为实例的字段`Module₃.toAddCommGroup₃`。这个实例将具有错误信息中出现的签名：(R : Type) → [inst : Ring₃ R] → {M : Type} → [self : Module₁ R M] → AddCommGroup₃ M。在类型类数据库中拥有这样的实例后，每次 Lean 寻找某个`M`的`AddCommGroup₃ M`实例时，它都需要在开始寻找`Module₁ R M`实例的主要任务之前，去寻找一个完全未指定的类型`R`和一个`Ring₃ R`实例。这两个辅助任务是错误信息中提到的元变量所代表的，分别用`?R`和`?inst✝`表示。这样的`Module₃.toAddCommGroup₃`实例将是对实例解析过程的巨大陷阱，然后`class`命令拒绝设置它。

那么`extends SMul₃ R M`又是怎么回事呢？它创建了一个域`Module₁.toSMul₃ : {R : Type} → [inst : Ring₃ R] → {M : Type} → [inst_1 : AddCommGroup₃ M] → [self : Module₁ R M] → SMul₃ R M`，其最终结果`SMul₃ R M`提到了`R`和`M`，因此这个域可以安全地用作实例。规则很容易记住：`extends`子句中出现的每个类都应该提到参数中出现的每个类型。

让我们创建第一个模块实例：一个环是其自身上的一个模块，使用其乘法作为标量乘法。

```py
instance  selfModule  (R  :  Type)  [Ring₃  R]  :  Module₁  R  R  where
  smul  :=  fun  r  s  ↦  r*s
  zero_smul  :=  zero_mul
  one_smul  :=  one_mul
  mul_smul  :=  mul_assoc₃
  add_smul  :=  Ring₃.right_distrib
  smul_add  :=  Ring₃.left_distrib 
```

作为第二个例子，每个阿贝尔群都是 `ℤ` 上的一个模块（这也是通过允许非可逆标量来推广向量空间理论的原因之一）。首先，可以为任何带有零和加法的类型定义自然数的标量乘法：`n • a` 被定义为 `a + ⋯ + a`，其中 `a` 出现 `n` 次。然后，通过确保 `(-1) • a = -a` 来将其扩展到整数的标量乘法。

```py
def  nsmul₁  {M  :  Type*}  [Zero  M]  [Add  M]  :  ℕ  →  M  →  M
  |  0,  _  =>  0
  |  n  +  1,  a  =>  a  +  nsmul₁  n  a

def  zsmul₁  {M  :  Type*}  [Zero  M]  [Add  M]  [Neg  M]  :  ℤ  →  M  →  M
  |  Int.ofNat  n,  a  =>  nsmul₁  n  a
  |  Int.negSucc  n,  a  =>  -nsmul₁  n.succ  a 
```

证明这一点并得出一个模块结构有点繁琐，并且对于当前的讨论来说并不有趣，所以我们将对所有公理表示歉意。你**不需要**用证明来替换这些歉意。如果你坚持这样做，那么你可能需要陈述并证明关于 `nsmul₁` 和 `zsmul₁` 的几个中间引理。

```py
instance  abGrpModule  (A  :  Type)  [AddCommGroup₃  A]  :  Module₁  ℤ  A  where
  smul  :=  zsmul₁
  zero_smul  :=  sorry
  one_smul  :=  sorry
  mul_smul  :=  sorry
  add_smul  :=  sorry
  smul_add  :=  sorry 
```

一个更加重要的问题是，我们现在在 `ℤ` 环上对 `ℤ` 本身有两个模块结构：`abGrpModule ℤ`，因为 `ℤ` 是一个阿贝尔群，以及 `selfModule ℤ`，因为 `ℤ` 是一个环。这两个模块结构对应于相同的阿贝尔群结构，但它们是否具有相同的标量乘法并不明显。实际上，它们确实如此，但这并不是定义上的，需要证明。这对类型类实例解析过程来说是个坏消息，并将导致用户在使用这个层次结构时遇到非常令人沮丧的失败。当直接要求找到一个实例时，Lean 会选择一个，我们可以使用以下方式来查看它：

```py
#synth  Module₁  ℤ  ℤ  -- abGrpModule ℤ 
```

但在更间接的上下文中，可能会发生 Lean 推断出另一个，然后变得困惑的情况。这种情况被称为“坏钻石”。这与我们上面使用的钻石操作无关，它指的是从 `ℤ` 到其 `Module₁ ℤ` 的路径绘制方式，无论是通过 `AddCommGroup₃ ℤ` 还是 `Ring₃ ℤ`。

重要的是要理解并非所有钻石都是坏的。事实上，Mathlib 中到处都是钻石，本章也是如此。在非常开始的时候，我们就看到了可以从 `Monoid₁ α` 通过 `Semigroup₁ α` 或 `DiaOneClass₁ α` 到 `Dia₁ α` 的转换，并且由于 `class` 命令的工作，得到的两个 `Dia₁ α` 实例在定义上是相等的。特别是，底部有 `Prop` 值类的一个钻石不能是坏的，因为任何两个相同陈述的证明在定义上是相等的。

但我们用模块创建的钻石肯定是不好的。问题在于 `smul` 字段，它是数据，而不是证明，我们有两种不是定义上相等的构造。修复这个问题的稳健方式是确保从丰富结构到贫弱结构的转换总是通过忘记数据来完成的，而不是通过定义数据。这个众所周知的设计模式被命名为“遗忘继承”，并在[`inria.hal.science/hal-02463336v2`](https://inria.hal.science/hal-02463336v2)中广泛讨论。

在我们的具体情况下，我们可以修改`AddMonoid₃`的定义，包括一个`nsmul`数据字段和一些`Prop`类型的字段，以确保这个操作是可证明的，正如我们上面构造的那样。这些字段在下面的定义中使用`:=`在它们的类型之后给出默认值。多亏了这些默认值，大多数实例将与我们之前的定义完全一样构建。但在`ℤ`的特殊情况下，我们将能够提供特定的值。

```py
class  AddMonoid₄  (M  :  Type)  extends  AddSemigroup₃  M,  AddZeroClass  M  where
  /-- Multiplication by a natural number. -/
  nsmul  :  ℕ  →  M  →  M  :=  nsmul₁
  /-- Multiplication by `(0 : ℕ)` gives `0`. -/
  nsmul_zero  :  ∀  x,  nsmul  0  x  =  0  :=  by  intros;  rfl
  /-- Multiplication by `(n + 1 : ℕ)` behaves as expected. -/
  nsmul_succ  :  ∀  (n  :  ℕ)  (x),  nsmul  (n  +  1)  x  =  x  +  nsmul  n  x  :=  by  intros;  rfl

instance  mySMul  {M  :  Type}  [AddMonoid₄  M]  :  SMul  ℕ  M  :=  ⟨AddMonoid₄.nsmul⟩ 
```

让我们检查我们是否仍然可以构建一个不提供`nsmul`相关字段的乘积单群实例。

```py
instance  (M  N  :  Type)  [AddMonoid₄  M]  [AddMonoid₄  N]  :  AddMonoid₄  (M  ×  N)  where
  add  :=  fun  p  q  ↦  (p.1  +  q.1,  p.2  +  q.2)
  add_assoc₃  :=  fun  a  b  c  ↦  by  ext  <;>  apply  add_assoc₃
  zero  :=  (0,  0)
  zero_add  :=  fun  a  ↦  by  ext  <;>  apply  zero_add
  add_zero  :=  fun  a  ↦  by  ext  <;>  apply  add_zero 
```

现在我们来处理`ℤ`的特殊情况，我们想要使用`ℕ`到`ℤ`的强制转换和`ℤ`上的乘法来构建`nsmul`。特别注意的是，证明字段比上面的默认值包含更多的工作。

```py
instance  :  AddMonoid₄  ℤ  where
  add  :=  (·  +  ·)
  add_assoc₃  :=  Int.add_assoc
  zero  :=  0
  zero_add  :=  Int.zero_add
  add_zero  :=  Int.add_zero
  nsmul  :=  fun  n  m  ↦  (n  :  ℤ)  *  m
  nsmul_zero  :=  Int.zero_mul
  nsmul_succ  :=  fun  n  m  ↦  show  (n  +  1  :  ℤ)  *  m  =  m  +  n  *  m
  by  rw  [Int.add_mul,  Int.add_comm,  Int.one_mul] 
```

让我们检查我们是否解决了我们的问题。因为 Lean 已经有一个自然数和整数标量乘法的定义，我们想要确保我们的实例被使用，所以我们不会使用`•`符号，而是调用`SMul.mul`并明确提供我们上面定义的实例。

```py
example  (n  :  ℕ)  (m  :  ℤ)  :  SMul.smul  (self  :=  mySMul)  n  m  =  n  *  m  :=  rfl 
```

这个故事继续讲述将`zsmul`字段纳入群的定义以及类似的技巧。你现在可以阅读 Mathlib 中单群、群、环和模的定义了。它们比我们在这里看到的要复杂，因为它们是巨大层次结构的一部分，但所有原则都已在上面解释过。

作为练习，你可以回到你上面构建的序关系层次结构，尝试引入一个带有`<₁`小于符号的类型类`LT₁`，并确保每个偏序都带有`<₁`，它有一个从`≤₁`构建的默认值，以及一个`Prop`类型的字段，断言这两个比较运算符之间的自然关系。## 8.2. 态射

到目前为止，在本章中，我们讨论了如何创建数学结构的层次结构。但定义结构并不是真正完成，直到我们有态射。这里有两种主要的方法。最明显的一个是定义一个关于函数的谓词。

```py
def  isMonoidHom₁  [Monoid  G]  [Monoid  H]  (f  :  G  →  H)  :  Prop  :=
  f  1  =  1  ∧  ∀  g  g',  f  (g  *  g')  =  f  g  *  f  g' 
```

在这个定义中，使用连词有点令人不快。特别是当用户想要访问两个条件时，他们需要记住我们选择的顺序。因此，我们可以使用一个结构来代替。

```py
structure  isMonoidHom₂  [Monoid  G]  [Monoid  H]  (f  :  G  →  H)  :  Prop  where
  map_one  :  f  1  =  1
  map_mul  :  ∀  g  g',  f  (g  *  g')  =  f  g  *  f  g' 
```

一旦我们到了这里，甚至有将其做成一个类并使用类型类实例解析过程自动推断出复杂函数的`isMonoidHom₂`的诱惑。例如，单群同态的复合是一个单群同态，这似乎是一个有用的实例。然而，这样的实例对于解析过程来说会非常棘手，因为它需要到处寻找`g ∘ f`。在`g (f x)`中看到它失败会非常沮丧。更普遍地说，我们必须始终记住，识别给定表达式中的哪个函数被应用是一个非常困难的问题，被称为“高阶统一问题”。因此，Mathlib 不使用这种类方法。

一个更基本的问题是，我们是使用上述的谓词（使用`def`或`structure`）还是使用捆绑函数和谓词的结构。这在很大程度上是一个心理问题。考虑一个不是同态的从单群到单群之间的函数是非常罕见的。这真的感觉像是“单群同态”不是一个你可以赋予裸函数的形容词，它是一个名词。另一方面，有人可以争辩说，在拓扑空间之间的连续函数实际上是一个恰好是连续的函数。这就是 Mathlib 有`Continuous`谓词的一个原因。例如，你可以写：

```py
example  :  Continuous  (id  :  ℝ  →  ℝ)  :=  continuous_id 
```

我们仍然有连续函数的捆绑，这在例如给连续函数的空间赋予拓扑时很方便，但它们不是处理连续性的主要工具。

相比之下，单群（或其他代数结构）之间的同态是捆绑在一起的，如下所示：

```py
@[ext]
structure  MonoidHom₁  (G  H  :  Type)  [Monoid  G]  [Monoid  H]  where
  toFun  :  G  →  H
  map_one  :  toFun  1  =  1
  map_mul  :  ∀  g  g',  toFun  (g  *  g')  =  toFun  g  *  toFun  g' 
```

当然，我们不想在所有地方都输入`toFun`，所以我们使用`CoeFun`类型类注册了一个强制转换。它的第一个参数是我们想要强制转换为函数的类型。第二个参数描述了目标函数类型。在我们的情况下，对于每个`f : MonoidHom₁ G H`，它总是`G → H`。我们还用`coe`属性标记了`MonoidHom₁.toFun`，以确保它在战术状态中几乎不可见，只需一个`↑`前缀即可。

```py
instance  [Monoid  G]  [Monoid  H]  :  CoeFun  (MonoidHom₁  G  H)  (fun  _  ↦  G  →  H)  where
  coe  :=  MonoidHom₁.toFun

attribute  [coe]  MonoidHom₁.toFun 
```

让我们检查我们是否真的可以将捆绑的单群同态应用于一个元素。

```py
example  [Monoid  G]  [Monoid  H]  (f  :  MonoidHom₁  G  H)  :  f  1  =  1  :=  f.map_one 
```

我们可以用其他类型的同态做同样的事情，直到我们达到环同态。

```py
@[ext]
structure  AddMonoidHom₁  (G  H  :  Type)  [AddMonoid  G]  [AddMonoid  H]  where
  toFun  :  G  →  H
  map_zero  :  toFun  0  =  0
  map_add  :  ∀  g  g',  toFun  (g  +  g')  =  toFun  g  +  toFun  g'

instance  [AddMonoid  G]  [AddMonoid  H]  :  CoeFun  (AddMonoidHom₁  G  H)  (fun  _  ↦  G  →  H)  where
  coe  :=  AddMonoidHom₁.toFun

attribute  [coe]  AddMonoidHom₁.toFun

@[ext]
structure  RingHom₁  (R  S  :  Type)  [Ring  R]  [Ring  S]  extends  MonoidHom₁  R  S,  AddMonoidHom₁  R  S 
```

关于这种方法有几个问题。一个较小的问题是，由于`RingHom₁.toFun`不存在，我们不太清楚`coe`属性应该放在哪里，相关的函数是`MonoidHom₁.toFun ∘ RingHom₁.toMonoidHom₁`，这不是一个可以标记属性的声明（但我们可以定义一个`CoeFun  (RingHom₁ R S) (fun _ ↦ R → S)`实例）。一个更重要的问题是，关于单群同态的引理不会直接适用于环同态。这留下了两种选择：要么每次想要应用单群同态引理时都玩弄`RingHom₁.toMonoidHom₁`，要么为环同态重新陈述每一个这样的引理。这两种选择都不吸引人，因此 Mathlib 在这里使用了一个新的层次技巧。想法是为至少是单群同态的对象定义一个类型类，用单群同态和环同态实例化这个类，并使用它来陈述每一个引理。在下面的定义中，`F`可以是`MonoidHom₁ M N`，或者如果`M`和`N`有环结构，可以是`RingHom₁ M N`。

```py
class  MonoidHomClass₁  (F  :  Type)  (M  N  :  Type)  [Monoid  M]  [Monoid  N]  where
  toFun  :  F  →  M  →  N
  map_one  :  ∀  f  :  F,  toFun  f  1  =  1
  map_mul  :  ∀  f  g  g',  toFun  f  (g  *  g')  =  toFun  f  g  *  toFun  f  g' 
```

然而，上述实现存在一个问题。我们还没有注册一个转换到函数实例的强制。现在让我们尝试做这件事。

```py
def  badInst  [Monoid  M]  [Monoid  N]  [MonoidHomClass₁  F  M  N]  :  CoeFun  F  (fun  _  ↦  M  →  N)  where
  coe  :=  MonoidHomClass₁.toFun 
```

将其作为一个实例是错误的。当面对类似`f x`的东西，其中`f`的类型不是函数类型时，Lean 会尝试找到一个`CoeFun`实例来将`f`转换成一个函数。上述函数的类型是：`{M N F : Type} → [Monoid M] → [Monoid N] → [MonoidHomClass₁ F M N] → CoeFun F (fun x ↦ M → N)`，因此，当它尝试应用它时，Lean 在先验上不清楚未知类型`M`、`N`和`F`应该以什么顺序推断。这是一种与我们已经看到的略有不同的不良实例，但归结为同一个问题：不知道`M`，Lean 将不得不在一个未知类型上搜索单群实例，因此绝望地尝试数据库中的每一个单群实例。如果你对这种实例的效果感到好奇，你可以在上述声明顶部键入`set_option synthInstance.checkSynthOrder false in`，将`def badInst`替换为`instance`，并在这个文件中寻找随机的失败。

在这里，解决方案很简单，我们需要告诉 Lean 首先搜索`F`是什么，然后推断出`M`和`N`。这是通过使用`outParam`函数来完成的。这个函数定义为恒等函数，但仍然被类型类机制识别并触发所需的行为。因此，我们可以重新定义我们的类，注意`outParam`函数：

```py
class  MonoidHomClass₂  (F  :  Type)  (M  N  :  outParam  Type)  [Monoid  M]  [Monoid  N]  where
  toFun  :  F  →  M  →  N
  map_one  :  ∀  f  :  F,  toFun  f  1  =  1
  map_mul  :  ∀  f  g  g',  toFun  f  (g  *  g')  =  toFun  f  g  *  toFun  f  g'

instance  [Monoid  M]  [Monoid  N]  [MonoidHomClass₂  F  M  N]  :  CoeFun  F  (fun  _  ↦  M  →  N)  where
  coe  :=  MonoidHomClass₂.toFun

attribute  [coe]  MonoidHomClass₂.toFun 
```

现在我们可以继续执行我们的计划，实例化这个类。

```py
instance  (M  N  :  Type)  [Monoid  M]  [Monoid  N]  :  MonoidHomClass₂  (MonoidHom₁  M  N)  M  N  where
  toFun  :=  MonoidHom₁.toFun
  map_one  :=  fun  f  ↦  f.map_one
  map_mul  :=  fun  f  ↦  f.map_mul

instance  (R  S  :  Type)  [Ring  R]  [Ring  S]  :  MonoidHomClass₂  (RingHom₁  R  S)  R  S  where
  toFun  :=  fun  f  ↦  f.toMonoidHom₁.toFun
  map_one  :=  fun  f  ↦  f.toMonoidHom₁.map_one
  map_mul  :=  fun  f  ↦  f.toMonoidHom₁.map_mul 
```

如承诺的那样，我们关于`f : F`的每一个引理，假设`MonoidHomClass₁ F`的实例，都将适用于单群同态和环同态。让我们看一个示例引理并检查它是否适用于这两种情况。

```py
lemma  map_inv_of_inv  [Monoid  M]  [Monoid  N]  [MonoidHomClass₂  F  M  N]  (f  :  F)  {m  m'  :  M}  (h  :  m*m'  =  1)  :
  f  m  *  f  m'  =  1  :=  by
  rw  [←  MonoidHomClass₂.map_mul,  h,  MonoidHomClass₂.map_one]

example  [Monoid  M]  [Monoid  N]  (f  :  MonoidHom₁  M  N)  {m  m'  :  M}  (h  :  m*m'  =  1)  :  f  m  *  f  m'  =  1  :=
map_inv_of_inv  f  h

example  [Ring  R]  [Ring  S]  (f  :  RingHom₁  R  S)  {r  r'  :  R}  (h  :  r*r'  =  1)  :  f  r  *  f  r'  =  1  :=
map_inv_of_inv  f  h 
```

初看起来，可能看起来我们回到了我们以前的老坏主意，将 `MonoidHom₁` 作为类。但我们并没有。一切都被提升了一个抽象层次。类型类解析过程不会寻找函数，它将寻找 `MonoidHom₁` 或 `RingHom₁`。

我们的方法中存在的一个问题是围绕 `toFun` 字段及其对应的 `CoeFun` 实例和 `coe` 属性的重复代码。最好也记录下这种模式仅用于具有额外属性的功能，这意味着函数的强制转换应该是单射的。因此，Mathlib 通过添加一个基于类 `DFunLike`（其中“DFun”代表依赖函数）的抽象层。让我们在基础层之上重新定义我们的 `MonoidHomClass`。

```py
class  MonoidHomClass₃  (F  :  Type)  (M  N  :  outParam  Type)  [Monoid  M]  [Monoid  N]  extends
  DFunLike  F  M  (fun  _  ↦  N)  where
  map_one  :  ∀  f  :  F,  f  1  =  1
  map_mul  :  ∀  (f  :  F)  g  g',  f  (g  *  g')  =  f  g  *  f  g'

instance  (M  N  :  Type)  [Monoid  M]  [Monoid  N]  :  MonoidHomClass₃  (MonoidHom₁  M  N)  M  N  where
  coe  :=  MonoidHom₁.toFun
  coe_injective'  _  _  :=  MonoidHom₁.ext
  map_one  :=  MonoidHom₁.map_one
  map_mul  :=  MonoidHom₁.map_mul 
```

当然，同态的层次结构并没有在这里停止。我们可以继续定义一个扩展 `MonoidHomClass₃` 的类 `RingHomClass₃`，并在 `RingHom` 上实例化它，然后稍后在其上实例化 `AlgebraHom`（代数是具有额外结构的环）。但我们已经涵盖了 Mathlib 中用于同态的主要形式化思想，你应该准备好理解 Mathlib 中同态是如何定义的。

作为练习，你应该尝试定义你的有序类型之间打包的保持顺序的函数类，然后定义保持顺序的幺半群同态。这只是为了训练目的。像连续函数一样，保持顺序的函数在 Mathlib 中主要是未打包的，它们由 `Monotone` 谓词定义。当然，你需要完成下面的类定义。

```py
@[ext]
structure  OrderPresHom  (α  β  :  Type)  [LE  α]  [LE  β]  where
  toFun  :  α  →  β
  le_of_le  :  ∀  a  a',  a  ≤  a'  →  toFun  a  ≤  toFun  a'

@[ext]
structure  OrderPresMonoidHom  (M  N  :  Type)  [Monoid  M]  [LE  M]  [Monoid  N]  [LE  N]  extends
MonoidHom₁  M  N,  OrderPresHom  M  N

class  OrderPresHomClass  (F  :  Type)  (α  β  :  outParam  Type)  [LE  α]  [LE  β]

instance  (α  β  :  Type)  [LE  α]  [LE  β]  :  OrderPresHomClass  (OrderPresHom  α  β)  α  β  where

instance  (α  β  :  Type)  [LE  α]  [Monoid  α]  [LE  β]  [Monoid  β]  :
  OrderPresHomClass  (OrderPresMonoidHom  α  β)  α  β  where

instance  (α  β  :  Type)  [LE  α]  [Monoid  α]  [LE  β]  [Monoid  β]  :
  MonoidHomClass₃  (OrderPresMonoidHom  α  β)  α  β
  :=  sorry 
```  ## 8.3\. 子对象

在定义了一些代数结构和其同态之后，下一步是考虑继承这种代数结构的集合，例如子群或子环。这很大程度上与我们的前一个主题重叠。实际上，`X` 中的集合被实现为一个从 `X` 到 `Prop` 的函数，因此子对象是满足一定谓词的函数。因此，我们可以重用导致 `DFunLike` 类及其后代的许多想法。我们不会重用 `DFunLike` 本身，因为这会打破从 `Set X` 到 `X → Prop` 的抽象障碍。相反，有一个 `SetLike` 类。该类不是将注入包装到函数类型中，而是将注入包装到 `Set` 类型中，并定义相应的强制转换和 `Membership` 实例。

```py
@[ext]
structure  Submonoid₁  (M  :  Type)  [Monoid  M]  where
  /-- The carrier of a submonoid. -/
  carrier  :  Set  M
  /-- The product of two elements of a submonoid belongs to the submonoid. -/
  mul_mem  {a  b}  :  a  ∈  carrier  →  b  ∈  carrier  →  a  *  b  ∈  carrier
  /-- The unit element belongs to the submonoid. -/
  one_mem  :  1  ∈  carrier

/-- Submonoids in `M` can be seen as sets in `M`. -/
instance  [Monoid  M]  :  SetLike  (Submonoid₁  M)  M  where
  coe  :=  Submonoid₁.carrier
  coe_injective'  _  _  :=  Submonoid₁.ext 
```

配备了上述 `SetLike` 实例，我们已经在不使用 `N.carrier` 的情况下自然地陈述了一个子幺半群 `N` 包含 `1`。我们还可以在 `M` 中默默地将 `N` 作为集合处理，或者在其映射下取其直接像。

```py
example  [Monoid  M]  (N  :  Submonoid₁  M)  :  1  ∈  N  :=  N.one_mem

example  [Monoid  M]  (N  :  Submonoid₁  M)  (α  :  Type)  (f  :  M  →  α)  :=  f  ''  N 
```

我们还有一个强制转换到 `Type` 的操作，它使用 `Subtype`，因此，给定一个子幺半群 `N`，我们可以写一个参数 `(x : N)`，它可以被强制转换到属于 `M` 且属于 `N` 的一个元素。

```py
example  [Monoid  M]  (N  :  Submonoid₁  M)  (x  :  N)  :  (x  :  M)  ∈  N  :=  x.property 
```

使用这种强制转换为`Type`，我们也可以处理给子幺半群赋予幺半群结构的问题。我们将使用上面提到的与`N`关联的类型强制转换，以及断言这种强制转换是单射的`SetCoe.ext`引理。这两个都是由`SetLike`实例提供的。

```py
instance  SubMonoid₁Monoid  [Monoid  M]  (N  :  Submonoid₁  M)  :  Monoid  N  where
  mul  :=  fun  x  y  ↦  ⟨x*y,  N.mul_mem  x.property  y.property⟩
  mul_assoc  :=  fun  x  y  z  ↦  SetCoe.ext  (mul_assoc  (x  :  M)  y  z)
  one  :=  ⟨1,  N.one_mem⟩
  one_mul  :=  fun  x  ↦  SetCoe.ext  (one_mul  (x  :  M))
  mul_one  :=  fun  x  ↦  SetCoe.ext  (mul_one  (x  :  M)) 
```

注意，在上面的实例中，我们不是使用对`M`的强制转换并调用`property`字段，而是可以使用解构绑定符如下所示。

```py
example  [Monoid  M]  (N  :  Submonoid₁  M)  :  Monoid  N  where
  mul  :=  fun  ⟨x,  hx⟩  ⟨y,  hy⟩  ↦  ⟨x*y,  N.mul_mem  hx  hy⟩
  mul_assoc  :=  fun  ⟨x,  _⟩  ⟨y,  _⟩  ⟨z,  _⟩  ↦  SetCoe.ext  (mul_assoc  x  y  z)
  one  :=  ⟨1,  N.one_mem⟩
  one_mul  :=  fun  ⟨x,  _⟩  ↦  SetCoe.ext  (one_mul  x)
  mul_one  :=  fun  ⟨x,  _⟩  ↦  SetCoe.ext  (mul_one  x) 
```

为了将关于子幺半群的定义应用于子群或子环，我们需要一个类，就像对于态射那样。注意这个类接受一个`SetLike`实例作为参数，因此它不需要载体域并且可以在其字段中使用成员符号。

```py
class  SubmonoidClass₁  (S  :  Type)  (M  :  Type)  [Monoid  M]  [SetLike  S  M]  :  Prop  where
  mul_mem  :  ∀  (s  :  S)  {a  b  :  M},  a  ∈  s  →  b  ∈  s  →  a  *  b  ∈  s
  one_mem  :  ∀  s  :  S,  1  ∈  s

instance  [Monoid  M]  :  SubmonoidClass₁  (Submonoid₁  M)  M  where
  mul_mem  :=  Submonoid₁.mul_mem
  one_mem  :=  Submonoid₁.one_mem 
```

作为练习，你应该定义一个`Subgroup₁`结构，给它赋予一个`SetLike`实例和一个`SubmonoidClass₁`实例，在`Subgroup₁`关联的子类型上放置一个`Group`实例，并定义一个`SubgroupClass₁`类。

关于 Mathlib 中给定代数对象的子对象总是形成一个完备格，这一点非常重要，并且这种结构被大量使用。例如，你可能想要查找一个引理，说明子幺半群的交集是一个子幺半群。但这不会是一个引理，而是一个下确界构造。让我们来看两个子幺半群的情况。

```py
instance  [Monoid  M]  :  Min  (Submonoid₁  M)  :=
  ⟨fun  S₁  S₂  ↦
  {  carrier  :=  S₁  ∩  S₂
  one_mem  :=  ⟨S₁.one_mem,  S₂.one_mem⟩
  mul_mem  :=  fun  ⟨hx,  hx'⟩  ⟨hy,  hy'⟩  ↦  ⟨S₁.mul_mem  hx  hy,  S₂.mul_mem  hx'  hy'⟩  }⟩ 
```

这允许我们得到两个子幺半群的交集作为一个子幺半群。

```py
example  [Monoid  M]  (N  P  :  Submonoid₁  M)  :  Submonoid₁  M  :=  N  ⊓  P 
```

你可能会觉得我们不得不在上面的例子中使用下确界符号`⊓`而不是交集符号`∩`是一种遗憾。但想想上确界。两个子幺半群的并集不是一个子幺半群。然而，子幺半群仍然形成一个格（甚至是一个完备格）。实际上，`N ⊔ P`是由`N`和`P`的并集生成的子幺半群，当然用`N ∪ P`来表示它会非常令人困惑。所以，你可以看到使用`N ⊓ P`的用法要一致得多。它在各种类型的代数结构中也非常一致。一开始看到两个向量子空间`E`和`F`的和用`E ⊔ F`表示而不是`E + F`可能会觉得有点奇怪。但你会习惯的。很快，你将把`E + F`的表示看作是一种干扰，强调`E ⊔ F`的元素可以写成`E`的一个元素和`F`的一个元素的和，而不是强调`E ⊔ F`是包含`E`和`F`的最小子向量子空间这一基本事实。

我们本章的最后一个主题是商。同样，我们想要解释在 Mathlib 中如何构建方便的表示法并避免代码重复。在这里，主要的工具是`HasQuotient`类，它允许使用像`M ⧸ N`这样的表示法。注意，商符号`⧸`是一个特殊的 Unicode 字符，不是一个常规的 ASCII 除法符号。

作为例子，我们将构建一个交换幺半群除以子幺半群的商，证明留给你们。在上一个例子中，你可以使用`Setoid.refl`，但它不会自动获取相关的`Setoid`结构。你可以通过使用`@`语法提供所有参数来解决这个问题，就像在`@Setoid.refl M N.Setoid`中一样。

```py
def  Submonoid.Setoid  [CommMonoid  M]  (N  :  Submonoid  M)  :  Setoid  M  where
  r  :=  fun  x  y  ↦  ∃  w  ∈  N,  ∃  z  ∈  N,  x*w  =  y*z
  iseqv  :=  {
  refl  :=  fun  x  ↦  ⟨1,  N.one_mem,  1,  N.one_mem,  rfl⟩
  symm  :=  fun  ⟨w,  hw,  z,  hz,  h⟩  ↦  ⟨z,  hz,  w,  hw,  h.symm⟩
  trans  :=  by
  sorry
  }

instance  [CommMonoid  M]  :  HasQuotient  M  (Submonoid  M)  where
  quotient'  :=  fun  N  ↦  Quotient  N.Setoid

def  QuotientMonoid.mk  [CommMonoid  M]  (N  :  Submonoid  M)  :  M  →  M  ⧸  N  :=  Quotient.mk  N.Setoid

instance  [CommMonoid  M]  (N  :  Submonoid  M)  :  Monoid  (M  ⧸  N)  where
  mul  :=  Quotient.map₂  (·  *  ·)  (by
  sorry
  )
  mul_assoc  :=  by
  sorry
  one  :=  QuotientMonoid.mk  N  1
  one_mul  :=  by
  sorry
  mul_one  :=  by
  sorry 
``` 上一节 下一节

* * *

© 版权所有 2020-2025，Jeremy Avigad，Patrick Massot。文本许可协议为 CC BY 4.0。

使用[Sphinx](https://www.sphinx-doc.org/)和由[Read the Docs](https://readthedocs.org)提供的[主题](https://github.com/readthedocs/sphinx_rtd_theme)构建。我们在第七章中看到了如何定义群类并构建该类的实例，然后是如何构建交换环类的实例。但当然这里也有一个层次结构：交换环特别是一个加法群。在本章中，我们将研究如何构建这样的层次结构。它们出现在数学的所有分支中，但本章的重点将放在代数示例上。

在更多关于如何使用现有层次结构的讨论之前讨论如何构建层次结构可能显得有些过早。但为了使用这些层次结构，对支撑这些层次结构的技术有一些了解是必要的。因此，你可能仍然应该阅读这一章，但不要在第一次阅读时过于努力地记住所有内容，然后阅读接下来的章节，再回来这里进行第二次阅读。

在本章中，我们将重新定义（Mathlib 中出现的许多事物的简化版本）许多东西，因此我们将使用索引来区分我们的版本。例如，我们将有`Ring₁`作为我们的`Ring`版本。由于我们将逐步解释更强大的形式化结构的方法，这些索引有时会超过一个。

## 8.1. 基础

在 Lean 的所有层次结构的底部，我们找到了携带数据的类。以下类记录了给定的类型`α`被赋予了一个称为`one`的特称元素。在这一阶段，它没有任何属性。

```py
class  One₁  (α  :  Type)  where
  /-- The element one -/
  one  :  α 
```

由于我们将在本章中更频繁地使用类，我们需要了解一些关于`class`命令做了什么的更多细节。首先，上面的`class`命令定义了一个带有参数`α : Type`和单个字段`one`的结构`One₁`。它还标记这个结构为一个类，以便某些类型`α`的`One₁ α`参数可以通过实例解析过程推断出来，只要它们被标记为实例隐式，即出现在方括号之间。这两个效果也可以通过带有`class`属性的`structure`命令实现，即写作`@[class] structure`实例。但类命令还确保`One₁ α`出现在其自己的字段中作为实例隐式参数。比较：

```py
#check  One₁.one  -- One₁.one {α : Type} [self : One₁ α] : α

@[class]  structure  One₂  (α  :  Type)  where
  /-- The element one -/
  one  :  α

#check  One₂.one 
```

在第二个检查中，我们可以看到`self : One₂ α`是一个显式参数。让我们确保第一个版本确实可以在没有任何显式参数的情况下使用。

```py
example  (α  :  Type)  [One₁  α]  :  α  :=  One₁.one 
```

备注：在上面的例子中，参数`One₁ α`被标记为实例隐式，这有点愚蠢，因为这只影响声明的*使用*以及由`example`命令创建的声明不能被使用。然而，它允许我们避免给这个参数命名，更重要的是，它开始培养标记`One₁ α`参数为实例隐式的良好习惯。

另一个备注是，所有这些只有在 Lean 知道`α`是什么时才会工作。在上面的例子中，省略类型注解`: α`将生成一个错误信息，例如：`typeclass instance problem is stuck, it is often due to metavariables One₁ (?m.263 α)`，其中`?m.263 α`意味着“依赖于`α`的某些类型”（而 263 是一个自动生成的索引，可以用来区分几个未知的事物）。避免这个问题的另一种方法是在类型声明中使用类型注解，如下所示：

```py
example  (α  :  Type)  [One₁  α]  :=  (One₁.one  :  α) 
```

你可能已经在第 3.6 节中玩序列的极限时遇到过这个问题，如果你尝试声明例如`0 < 1`而没有告诉 Lean 你指的是自然数还是实数的不等式。

我们下一个任务是给`One₁.one`分配一个符号。由于我们不希望与内置的`1`的符号冲突，我们将使用`𝟙`。这可以通过以下命令实现，其中第一行告诉 Lean 使用`One₁.one`的文档作为符号`𝟙`的文档。

```py
@[inherit_doc]
notation  "𝟙"  =>  One₁.one

example  {α  :  Type}  [One₁  α]  :  α  :=  𝟙

example  {α  :  Type}  [One₁  α]  :  (𝟙  :  α)  =  𝟙  :=  rfl 
```

我们现在想要一个携带数据的类来记录二元运算。目前我们不想在加法和乘法之间做出选择，所以我们将使用菱形。

```py
class  Dia₁  (α  :  Type)  where
  dia  :  α  →  α  →  α

infixl:70  " ⋄ "  =>  Dia₁.dia 
```

正如在`One₁`的例子中，这个操作在这个阶段没有任何属性。现在让我们定义一个半群结构类，其中操作用`⋄`表示。现在，我们通过手动定义一个具有两个字段的结构来定义它，一个`Dia₁`实例和一些`Prop`类型的字段`dia_assoc`，该字段断言`⋄`的结合性。

```py
class  Semigroup₀  (α  :  Type)  where
  toDia₁  :  Dia₁  α
  /-- Diamond is associative -/
  dia_assoc  :  ∀  a  b  c  :  α,  a  ⋄  b  ⋄  c  =  a  ⋄  (b  ⋄  c) 
```

注意，在声明 dia_assoc 时，之前定义的字段 toDia₁处于局部上下文中，因此当 Lean 搜索 Dia₁ α的实例以理解⋄ b 时可以使用它。然而，这个 toDia₁字段并不成为类型类实例数据库的一部分。因此，执行`example {α : Type} [Semigroup₁ α] (a b : α) : α := a ⋄ b`将会失败，并显示错误信息`failed to synthesize instance Dia₁ α`。

我们可以通过稍后添加`instance`属性来修复这个问题。

```py
attribute  [instance]  Semigroup₀.toDia₁

example  {α  :  Type}  [Semigroup₀  α]  (a  b  :  α)  :  α  :=  a  ⋄  b 
```

在构建之前，我们需要使用不同的语法来添加这个 toDia₁字段，告诉 Lean 将 Dia₁ α视为如果它的字段是 Semigroup₁本身的字段。这也方便地自动添加了 toDia₁实例。`class`命令使用`extends`语法支持这一点，如下所示：

```py
class  Semigroup₁  (α  :  Type)  extends  toDia₁  :  Dia₁  α  where
  /-- Diamond is associative -/
  dia_assoc  :  ∀  a  b  c  :  α,  a  ⋄  b  ⋄  c  =  a  ⋄  (b  ⋄  c)

example  {α  :  Type}  [Semigroup₁  α]  (a  b  :  α)  :  α  :=  a  ⋄  b 
```

注意这个语法在 `structure` 命令中也是可用的，尽管在那个情况下，它只解决了编写诸如 toDia₁ 这样的字段的问题，因为在这种情况下没有实例需要定义。

在 `extends` 语法中，字段名 toDia₁ 是可选的。默认情况下，它采用被扩展的类的名称，并在其前面加上“to”。

```py
class  Semigroup₂  (α  :  Type)  extends  Dia₁  α  where
  /-- Diamond is associative -/
  dia_assoc  :  ∀  a  b  c  :  α,  a  ⋄  b  ⋄  c  =  a  ⋄  (b  ⋄  c) 
```

现在我们尝试将一个菱形操作和一个特殊元素结合，通过公理说明这个元素在两边都是中性的。

```py
class  DiaOneClass₁  (α  :  Type)  extends  One₁  α,  Dia₁  α  where
  /-- One is a left neutral element for diamond. -/
  one_dia  :  ∀  a  :  α,  𝟙  ⋄  a  =  a
  /-- One is a right neutral element for diamond -/
  dia_one  :  ∀  a  :  α,  a  ⋄  𝟙  =  a 
```

在下一个例子中，我们告诉 Lean `α` 有一个 `DiaOneClass₁` 结构，并陈述一个使用 Dia₁ 实例和 One₁ 实例的属性。为了看到 Lean 如何找到这些实例，我们设置了一个跟踪选项，其结果可以在 Infoview 中看到。默认情况下，这个结果相当简短，但可以通过单击以黑色箭头结尾的行来扩展。它包括 Lean 在成功之前尝试找到实例的失败尝试。成功的尝试确实涉及由 `extends` 语法生成的实例。

```py
set_option  trace.Meta.synthInstance  true  in
example  {α  :  Type}  [DiaOneClass₁  α]  (a  b  :  α)  :  Prop  :=  a  ⋄  b  =  𝟙 
```

注意，在组合现有类时，我们不需要包含额外的字段。因此，我们可以定义幺半群如下：

```py
class  Monoid₁  (α  :  Type)  extends  Semigroup₁  α,  DiaOneClass₁  α 
```

虽然上面的定义看起来很简单，但它隐藏了一个重要的微妙之处。`Semigroup₁ α` 和 `DiaOneClass₁ α` 都扩展了 `Dia₁ α`，因此人们可能会担心拥有一个 `Monoid₁ α` 实例会在 `α` 上产生两个无关的菱形操作，一个来自域 `Monoid₁.toSemigroup₁`，另一个来自域 `Monoid₁.toDiaOneClass₁`。

事实上，如果我们尝试手动使用以下方式构建一个幺半群类：

```py
class  Monoid₂  (α  :  Type)  where
  toSemigroup₁  :  Semigroup₁  α
  toDiaOneClass₁  :  DiaOneClass₁  α 
```

然后，我们得到两个完全无关的菱形操作 `Monoid₂.toSemigroup₁.toDia₁.dia` 和 `Monoid₂.toDiaOneClass₁.toDia₁.dia`。

使用 `extends` 语法生成的版本没有这个缺陷。

```py
example  {α  :  Type}  [Monoid₁  α]  :
  (Monoid₁.toSemigroup₁.toDia₁.dia  :  α  →  α  →  α)  =  Monoid₁.toDiaOneClass₁.toDia₁.dia  :=  rfl 
```

因此，`class` 命令为我们做了一些魔法（`structure` 命令也会这样做）。一个简单的方法来查看我们类的字段是检查它们的构造函数。比较：

```py
/- Monoid₂.mk {α : Type} (toSemigroup₁ : Semigroup₁ α) (toDiaOneClass₁ : DiaOneClass₁ α) : Monoid₂ α -/
#check  Monoid₂.mk

/- Monoid₁.mk {α : Type} [toSemigroup₁ : Semigroup₁ α] [toOne₁ : One₁ α] (one_dia : ∀ (a : α), 𝟙 ⋄ a = a) (dia_one : ∀ (a : α), a ⋄ 𝟙 = a) : Monoid₁ α -/
#check  Monoid₁.mk 
```

因此，我们看到 `Monoid₁` 按预期接受 `Semigroup₁ α` 参数，但它不会接受一个潜在的重复的 `DiaOneClass₁ α` 参数，而是将其拆分并只包含非重叠的部分。它还自动生成了一个实例 `Monoid₁.toDiaOneClass₁`，这不是一个域，但它具有预期的签名，从最终用户的角度来看，恢复了两个扩展类 `Semigroup₁` 和 `DiaOneClass₁` 之间的对称性。

```py
#check  Monoid₁.toSemigroup₁
#check  Monoid₁.toDiaOneClass₁ 
```

我们现在非常接近定义群的概念。我们可以在幺半群结构中添加一个域，断言每个元素都存在逆元。但那样我们就需要工作来访问这些逆元。在实践中，将其作为数据添加更为方便。为了优化可重用性，我们定义一个新的数据承载类，然后给它一些符号。

```py
class  Inv₁  (α  :  Type)  where
  /-- The inversion function -/
  inv  :  α  →  α

@[inherit_doc]
postfix:max  "⁻¹"  =>  Inv₁.inv

class  Group₁  (G  :  Type)  extends  Monoid₁  G,  Inv₁  G  where
  inv_dia  :  ∀  a  :  G,  a⁻¹  ⋄  a  =  𝟙 
```

上述定义可能看起来太弱，我们只要求 `a⁻¹` 是 `a` 的左逆。但另一方面是自动的。为了证明这一点，我们需要一个初步的引理。

```py
lemma  left_inv_eq_right_inv₁  {M  :  Type}  [Monoid₁  M]  {a  b  c  :  M}  (hba  :  b  ⋄  a  =  𝟙)  (hac  :  a  ⋄  c  =  𝟙)  :  b  =  c  :=  by
  rw  [←  DiaOneClass₁.one_dia  c,  ←  hba,  Semigroup₁.dia_assoc,  hac,  DiaOneClass₁.dia_one  b] 
```

在这个引理中，给出全名相当令人烦恼，尤其是因为它需要知道哪个层次结构提供了这些事实。解决这个问题的一种方法是通过使用`export`命令将那些事实作为根命名空间中的引理来复制。

```py
export  DiaOneClass₁  (one_dia  dia_one)
export  Semigroup₁  (dia_assoc)
export  Group₁  (inv_dia) 
```

我们可以将上述证明重写为：

```py
example  {M  :  Type}  [Monoid₁  M]  {a  b  c  :  M}  (hba  :  b  ⋄  a  =  𝟙)  (hac  :  a  ⋄  c  =  𝟙)  :  b  =  c  :=  by
  rw  [←  one_dia  c,  ←  hba,  dia_assoc,  hac,  dia_one  b] 
```

现在轮到你来证明关于我们的代数结构的事情了。

```py
lemma  inv_eq_of_dia  [Group₁  G]  {a  b  :  G}  (h  :  a  ⋄  b  =  𝟙)  :  a⁻¹  =  b  :=
  sorry

lemma  dia_inv  [Group₁  G]  (a  :  G)  :  a  ⋄  a⁻¹  =  𝟙  :=
  sorry 
```

在这个阶段，我们希望继续定义环，但存在一个严重的问题。一个类型上的环结构包含一个加法群结构和乘法幺半群结构，以及关于它们之间相互作用的某些性质。但到目前为止，我们为所有操作硬编码了一个符号`⋄`。更基本的是，类型类系统假设每个类型只有一个类型类的实例。有各种方法可以解决这个问题。令人惊讶的是，Mathlib 使用了一种原始的想法，通过一些代码生成属性来复制所有加法和乘法理论的内容。结构和类都在加法和乘法符号下定义，并通过属性`to_additive`将它们链接起来。在类似半群的多重继承情况下，自动生成的“对称恢复”实例也需要标记。这有点技术性；你不需要理解细节。重要的是，引理只以乘法符号表示，并标记为`to_additive`以生成加法版本`left_inv_eq_right_inv'`及其自动生成的加法版本`left_neg_eq_right_neg'`。为了检查这个加法版本的名称，我们在`left_inv_eq_right_inv'`的顶部使用了`whatsnew in`命令。

```py
class  AddSemigroup₃  (α  :  Type)  extends  Add  α  where
  /-- Addition is associative -/
  add_assoc₃  :  ∀  a  b  c  :  α,  a  +  b  +  c  =  a  +  (b  +  c)

@[to_additive  AddSemigroup₃]
class  Semigroup₃  (α  :  Type)  extends  Mul  α  where
  /-- Multiplication is associative -/
  mul_assoc₃  :  ∀  a  b  c  :  α,  a  *  b  *  c  =  a  *  (b  *  c)

class  AddMonoid₃  (α  :  Type)  extends  AddSemigroup₃  α,  AddZeroClass  α

@[to_additive  AddMonoid₃]
class  Monoid₃  (α  :  Type)  extends  Semigroup₃  α,  MulOneClass  α

export  Semigroup₃  (mul_assoc₃)
export  AddSemigroup₃  (add_assoc₃)

whatsnew  in
@[to_additive]
lemma  left_inv_eq_right_inv'  {M  :  Type}  [Monoid₃  M]  {a  b  c  :  M}  (hba  :  b  *  a  =  1)  (hac  :  a  *  c  =  1)  :  b  =  c  :=  by
  rw  [←  one_mul  c,  ←  hba,  mul_assoc₃,  hac,  mul_one  b]

#check  left_neg_eq_right_neg' 
```

有了这项技术，我们也可以轻松地定义交换半群、幺半群和群，然后定义环。

```py
class  AddCommSemigroup₃  (α  :  Type)  extends  AddSemigroup₃  α  where
  add_comm  :  ∀  a  b  :  α,  a  +  b  =  b  +  a

@[to_additive  AddCommSemigroup₃]
class  CommSemigroup₃  (α  :  Type)  extends  Semigroup₃  α  where
  mul_comm  :  ∀  a  b  :  α,  a  *  b  =  b  *  a

class  AddCommMonoid₃  (α  :  Type)  extends  AddMonoid₃  α,  AddCommSemigroup₃  α

@[to_additive  AddCommMonoid₃]
class  CommMonoid₃  (α  :  Type)  extends  Monoid₃  α,  CommSemigroup₃  α

class  AddGroup₃  (G  :  Type)  extends  AddMonoid₃  G,  Neg  G  where
  neg_add  :  ∀  a  :  G,  -a  +  a  =  0

@[to_additive  AddGroup₃]
class  Group₃  (G  :  Type)  extends  Monoid₃  G,  Inv  G  where
  inv_mul  :  ∀  a  :  G,  a⁻¹  *  a  =  1 
```

我们应该记得在适当的时候用`simp`标记引理。

```py
attribute  [simp]  Group₃.inv_mul  AddGroup₃.neg_add 
```

然后，由于我们转向标准符号，我们需要重复一下，但至少`to_additive`完成了将乘法符号翻译为加法符号的工作。

```py
@[to_additive]
lemma  inv_eq_of_mul  [Group₃  G]  {a  b  :  G}  (h  :  a  *  b  =  1)  :  a⁻¹  =  b  :=
  sorry 
```

注意，`to_additive`可以要求标记一个引理为`simp`并将该属性传播到加法版本，如下所示。

```py
@[to_additive  (attr  :=  simp)]
lemma  Group₃.mul_inv  {G  :  Type}  [Group₃  G]  {a  :  G}  :  a  *  a⁻¹  =  1  :=  by
  sorry

@[to_additive]
lemma  mul_left_cancel₃  {G  :  Type}  [Group₃  G]  {a  b  c  :  G}  (h  :  a  *  b  =  a  *  c)  :  b  =  c  :=  by
  sorry

@[to_additive]
lemma  mul_right_cancel₃  {G  :  Type}  [Group₃  G]  {a  b  c  :  G}  (h  :  b*a  =  c*a)  :  b  =  c  :=  by
  sorry

class  AddCommGroup₃  (G  :  Type)  extends  AddGroup₃  G,  AddCommMonoid₃  G

@[to_additive  AddCommGroup₃]
class  CommGroup₃  (G  :  Type)  extends  Group₃  G,  CommMonoid₃  G 
```

现在，我们已准备好定义环。为了演示目的，我们不会假设加法是交换的，然后立即提供一个`AddCommGroup₃`的实例。Mathlib 不玩这个游戏，首先是因为在实践中这不会使任何环实例更容易，而且 Mathlib 的代数层次结构通过半环，它们像环但没有相反元素，所以下面的证明对它们不适用。我们在这里获得的好处，除了如果你从未见过它，这是一个很好的练习之外，还有一个使用允许提供父结构作为实例参数的语法来构建实例的例子，然后提供额外的字段。这里，Ring₃ R 参数提供了 AddCommGroup₃ R 想要的任何东西，除了 add_comm。

```py
class  Ring₃  (R  :  Type)  extends  AddGroup₃  R,  Monoid₃  R,  MulZeroClass  R  where
  /-- Multiplication is left distributive over addition -/
  left_distrib  :  ∀  a  b  c  :  R,  a  *  (b  +  c)  =  a  *  b  +  a  *  c
  /-- Multiplication is right distributive over addition -/
  right_distrib  :  ∀  a  b  c  :  R,  (a  +  b)  *  c  =  a  *  c  +  b  *  c

instance  {R  :  Type}  [Ring₃  R]  :  AddCommGroup₃  R  :=
{  add_comm  :=  by
  sorry  } 
```

当然，我们也可以构建具体的实例，例如整数上的环结构（当然下面的实例使用 Mathlib 中已经完成的所有工作）。

```py
instance  :  Ring₃  ℤ  where
  add  :=  (·  +  ·)
  add_assoc₃  :=  add_assoc
  zero  :=  0
  zero_add  :=  by  simp
  add_zero  :=  by  simp
  neg  :=  (-  ·)
  neg_add  :=  by  simp
  mul  :=  (·  *  ·)
  mul_assoc₃  :=  mul_assoc
  one  :=  1
  one_mul  :=  by  simp
  mul_one  :=  by  simp
  zero_mul  :=  by  simp
  mul_zero  :=  by  simp
  left_distrib  :=  Int.mul_add
  right_distrib  :=  Int.add_mul 
```

作为练习，你现在可以设置一个简单的层次结构，包括一个有序交换幺半群类，它具有部分序和交换幺半群结构，使得`∀ a b : α, a ≤ b → ∀ c : α, c * a ≤ c * b`。当然，你需要向以下类中添加域和可能`extends`子句。

```py
class  LE₁  (α  :  Type)  where
  /-- The Less-or-Equal relation. -/
  le  :  α  →  α  →  Prop

@[inherit_doc]  infix:50  " ≤₁ "  =>  LE₁.le

class  Preorder₁  (α  :  Type)

class  PartialOrder₁  (α  :  Type)

class  OrderedCommMonoid₁  (α  :  Type)

instance  :  OrderedCommMonoid₁  ℕ  where 
```

我们现在想要讨论涉及多种类型的代数结构。一个典型的例子是环上的模。如果你不知道什么是模，你可以假设它意味着向量空间，并认为我们所有的环都是域。这些结构是带有由某些环的元素进行的标量乘法的交换加法群。

我们首先定义由类型`α`在类型`β`上携带的数据类型类，并给它一个右结合的符号。

```py
class  SMul₃  (α  :  Type)  (β  :  Type)  where
  /-- Scalar multiplication -/
  smul  :  α  →  β  →  β

infixr:73  " • "  =>  SMul₃.smul 
```

然后，我们可以定义模（如果你不知道什么是模，可以再次考虑向量空间）。

```py
class  Module₁  (R  :  Type)  [Ring₃  R]  (M  :  Type)  [AddCommGroup₃  M]  extends  SMul₃  R  M  where
  zero_smul  :  ∀  m  :  M,  (0  :  R)  •  m  =  0
  one_smul  :  ∀  m  :  M,  (1  :  R)  •  m  =  m
  mul_smul  :  ∀  (a  b  :  R)  (m  :  M),  (a  *  b)  •  m  =  a  •  b  •  m
  add_smul  :  ∀  (a  b  :  R)  (m  :  M),  (a  +  b)  •  m  =  a  •  m  +  b  •  m
  smul_add  :  ∀  (a  :  R)  (m  n  :  M),  a  •  (m  +  n)  =  a  •  m  +  a  •  n 
```

这里有一些有趣的事情正在发生。虽然`R`上的环结构在这个定义中是参数并不太令人惊讶，但你可能预计`AddCommGroup₃ M`将像`SMul₃ R M`一样成为`extends`子句的一部分。尝试这样做会导致一个听起来很神秘的错误信息：“无法为实例 Module₁.toAddCommGroup₃找到合成顺序，类型为(R : Type) → [inst : Ring₃ R] → {M : Type} → [self : Module₁ R M] → AddCommGroup₃ M，所有剩余的参数都有元变量：Ring₃ ?R @Module₁ ?R ?inst✝ M”。为了理解这条信息，你需要记住这样的`extends`子句会导致一个标记为实例的`Module₃.toAddCommGroup₃`字段。这个实例将具有错误信息中出现的签名：`(R : Type) → [inst : Ring₃ R] → {M : Type} → [self : Module₁ R M] → AddCommGroup₃ M`。在类型类数据库中有了这样的实例后，每次 Lean 寻找某个`M`的`AddCommGroup₃ M`实例时，它都需要在开始寻找`Module₁ R M`实例的主要任务之前，去寻找一个完全未指定的类型`R`和`Ring₃ R`实例。这两个副任务由错误信息中提到的元变量表示，并在这里用`?R`和`?inst✝`表示。这样的`Module₃.toAddCommGroup₃`实例将成为实例解析过程的一个巨大陷阱，然后`class`命令拒绝设置它。

那么`extends SMul₃ R M`又是怎么回事呢？它创建了一个字段`Module₁.toSMul₃ : {R : Type} → [inst : Ring₃ R] → {M : Type} → [inst_1 : AddCommGroup₃ M] → [self : Module₁ R M] → SMul₃ R M`，其最终结果`SMul₃ R M`提到了`R`和`M`，因此这个字段可以安全地用作实例。规则很容易记住：`extends`子句中出现的每个类都应该提到参数中出现的每个类型。

让我们创建我们的第一个模块实例：一个环是其自身的模块，使用其乘法作为标量乘法。

```py
instance  selfModule  (R  :  Type)  [Ring₃  R]  :  Module₁  R  R  where
  smul  :=  fun  r  s  ↦  r*s
  zero_smul  :=  zero_mul
  one_smul  :=  one_mul
  mul_smul  :=  mul_assoc₃
  add_smul  :=  Ring₃.right_distrib
  smul_add  :=  Ring₃.left_distrib 
```

作为第二个例子，每个阿贝尔群都是`ℤ`的模块（这是通过允许非可逆标量来泛化向量空间理论的原因之一）。首先，可以为任何带有零和加法的类型定义自然数的标量乘法：`n • a`定义为`a + ⋯ + a`，其中`a`出现`n`次。然后，通过确保`(-1) • a = -a`将其扩展到整数的标量乘法。

```py
def  nsmul₁  {M  :  Type*}  [Zero  M]  [Add  M]  :  ℕ  →  M  →  M
  |  0,  _  =>  0
  |  n  +  1,  a  =>  a  +  nsmul₁  n  a

def  zsmul₁  {M  :  Type*}  [Zero  M]  [Add  M]  [Neg  M]  :  ℤ  →  M  →  M
  |  Int.ofNat  n,  a  =>  nsmul₁  n  a
  |  Int.negSucc  n,  a  =>  -nsmul₁  n.succ  a 
```

证明这一点会产生一个模块结构，对于当前的讨论来说有点繁琐且不有趣，所以我们将对所有公理表示歉意。你**不是**被要求用证明来替换这些歉意。如果你坚持这样做，你可能需要陈述并证明关于`nsmul₁`和`zsmul₁`的几个中间引理。

```py
instance  abGrpModule  (A  :  Type)  [AddCommGroup₃  A]  :  Module₁  ℤ  A  where
  smul  :=  zsmul₁
  zero_smul  :=  sorry
  one_smul  :=  sorry
  mul_smul  :=  sorry
  add_smul  :=  sorry
  smul_add  :=  sorry 
```

一个更加重要的问题是，我们现在在环`ℤ`上对`ℤ`本身有两个模块结构：`abGrpModule ℤ`，因为`ℤ`是一个阿贝尔群，以及`selfModule ℤ`，因为`ℤ`是一个环。这两个模块结构对应于相同的阿贝尔群结构，但它们是否具有相同的标量乘法并不明显。实际上，它们确实如此，但这并不是定义上的，它需要一个证明。这对类型类实例解析过程来说是非常糟糕的消息，并将导致用户在使用这个层次结构时遇到非常令人沮丧的失败。当直接要求找到一个实例时，Lean 会选择一个，我们可以使用以下方式查看：

```py
#synth  Module₁  ℤ  ℤ  -- abGrpModule ℤ 
```

但在更间接的上下文中，可能会发生 Lean 推断出另一个，然后变得困惑的情况。这种情况被称为坏钻石。这与我们上面使用的钻石操作无关，它指的是从`ℤ`到其`Module₁ ℤ`通过`AddCommGroup₃ ℤ`或`Ring₃ ℤ`绘制路径的方式。

重要的是要理解并非所有钻石都是坏的。实际上，在 Mathlib 中到处都是钻石，也包括本章。在非常开始的时候，我们就看到了可以从`Monoid₁ α`通过`Semigroup₁ α`或`DiaOneClass₁ α`转换到`Dia₁ α`，多亏了`class`命令的工作，得到的两个`Dia₁ α`实例在定义上是相等的。特别是，底部有`Prop`值类的钻石不能是坏的，因为任何两个相同陈述的证明在定义上都是相等的。

但我们创建的模块钻石肯定是不好的。问题出在`smul`字段上，它不是证明，而是数据，我们有两组不是定义上相等的构造。修复这个问题的稳健方式是确保从丰富结构到贫弱结构的转换总是通过忘记数据来完成的，而不是通过定义数据。这个众所周知的模式被命名为“遗忘继承”，并在[`inria.hal.science/hal-02463336v2`](https://inria.hal.science/hal-02463336v2)中广泛讨论。

在我们的具体情况下，我们可以修改 `AddMonoid₃` 的定义，包括一个 `nsmul` 数据字段和一些确保此操作可证明是我们上面构造的 `Prop` 值字段。这些字段在下面的定义中使用 `:=` 在它们的类型之后提供默认值。多亏了这些默认值，大多数实例将与我们之前的定义完全一样构建。但在 `ℤ` 的特殊情况下，我们将能够提供特定的值。

```py
class  AddMonoid₄  (M  :  Type)  extends  AddSemigroup₃  M,  AddZeroClass  M  where
  /-- Multiplication by a natural number. -/
  nsmul  :  ℕ  →  M  →  M  :=  nsmul₁
  /-- Multiplication by `(0 : ℕ)` gives `0`. -/
  nsmul_zero  :  ∀  x,  nsmul  0  x  =  0  :=  by  intros;  rfl
  /-- Multiplication by `(n + 1 : ℕ)` behaves as expected. -/
  nsmul_succ  :  ∀  (n  :  ℕ)  (x),  nsmul  (n  +  1)  x  =  x  +  nsmul  n  x  :=  by  intros;  rfl

instance  mySMul  {M  :  Type}  [AddMonoid₄  M]  :  SMul  ℕ  M  :=  ⟨AddMonoid₄.nsmul⟩ 
```

让我们检查我们是否仍然可以构建一个不提供 `nsmul` 相关字段的乘积幺半群实例。

```py
instance  (M  N  :  Type)  [AddMonoid₄  M]  [AddMonoid₄  N]  :  AddMonoid₄  (M  ×  N)  where
  add  :=  fun  p  q  ↦  (p.1  +  q.1,  p.2  +  q.2)
  add_assoc₃  :=  fun  a  b  c  ↦  by  ext  <;>  apply  add_assoc₃
  zero  :=  (0,  0)
  zero_add  :=  fun  a  ↦  by  ext  <;>  apply  zero_add
  add_zero  :=  fun  a  ↦  by  ext  <;>  apply  add_zero 
```

现在，让我们处理 `ℤ` 的特殊情况，我们想使用 `ℕ` 到 `ℤ` 的强制转换和 `ℤ` 上的乘法来构建 `nsmul`。特别要注意证明字段比上面的默认值包含更多的工作。

```py
instance  :  AddMonoid₄  ℤ  where
  add  :=  (·  +  ·)
  add_assoc₃  :=  Int.add_assoc
  zero  :=  0
  zero_add  :=  Int.zero_add
  add_zero  :=  Int.add_zero
  nsmul  :=  fun  n  m  ↦  (n  :  ℤ)  *  m
  nsmul_zero  :=  Int.zero_mul
  nsmul_succ  :=  fun  n  m  ↦  show  (n  +  1  :  ℤ)  *  m  =  m  +  n  *  m
  by  rw  [Int.add_mul,  Int.add_comm,  Int.one_mul] 
```

让我们检查我们是否解决了问题。因为 Lean 已经定义了自然数和整数的标量乘法，我们想确保我们的实例被使用，所以我们不会使用 `•` 符号，而是调用 `SMul.mul` 并显式提供上面定义的实例。

```py
example  (n  :  ℕ)  (m  :  ℤ)  :  SMul.smul  (self  :=  mySMul)  n  m  =  n  *  m  :=  rfl 
```

这个故事随后继续，将 `zsmul` 字段纳入群的定义以及类似的技巧。你现在可以阅读 Mathlib 中单群、群、环和模块的定义了。它们比我们这里看到的更复杂，因为它们是巨大层次结构的一部分，但所有原则都已在上面解释。

作为练习，你可以回到你上面构建的序关系层次，尝试纳入一个携带 `<₁` 小于号注记的 `LT₁` 类型类，并确保每个偏序都带有 `<₁`，它有一个从 `≤₁` 构建的默认值，以及一个 `Prop` 值字段，断言这两个比较运算符之间的自然关系。## 8.2\. 态射

到目前为止，在本章中，我们讨论了如何创建数学结构的层次。但直到我们有了态射，定义结构才算真正完成。这里有两种主要的方法。最明显的一种是定义一个关于函数的谓词。

```py
def  isMonoidHom₁  [Monoid  G]  [Monoid  H]  (f  :  G  →  H)  :  Prop  :=
  f  1  =  1  ∧  ∀  g  g',  f  (g  *  g')  =  f  g  *  f  g' 
```

在这个定义中，使用合取有点不愉快。特别是当用户想要访问两个条件时，他们需要记住我们选择的排序。因此，我们可以使用一个结构。

```py
structure  isMonoidHom₂  [Monoid  G]  [Monoid  H]  (f  :  G  →  H)  :  Prop  where
  map_one  :  f  1  =  1
  map_mul  :  ∀  g  g',  f  (g  *  g')  =  f  g  *  f  g' 
```

一旦我们到了这里，甚至有将其作为一个类并使用类型类实例解析过程来自动推断复杂函数的`isMonoidHom₂`的诱惑。例如，单群同态的复合仍然是一个单群同态，这似乎是一个有用的实例。然而，对于解析过程来说，这样的实例会非常棘手，因为它需要到处寻找`g ∘ f`。在`g (f x)`中看到它失败会非常令人沮丧。更普遍地说，我们必须始终牢记，识别给定表达式中的哪个函数被应用是一个非常困难的问题，被称为“高阶统一问题”。因此，Mathlib 不使用这种类方法。

一个更基本的问题是，我们是使用上述谓词（使用`def`或`structure`）还是使用捆绑函数和谓词的结构。这部分是一个心理问题。考虑一个不是同态的群之间的函数是非常罕见的。这真的感觉像“单群同态”不是一个你可以分配给裸函数的形容词，它是一个名词。另一方面，可以争辩说，拓扑空间之间的连续函数实际上是一个恰好是连续的函数。这是 Mathlib 有`Continuous`谓词的一个原因。例如，你可以写：

```py
example  :  Continuous  (id  :  ℝ  →  ℝ)  :=  continuous_id 
```

我们仍然有一大堆连续函数，例如，它们方便地将拓扑结构置于连续函数的空间中，但它们并不是处理连续性的主要工具。

相比之下，群（或其他代数结构）之间的形态捆绑如下：

```py
@[ext]
structure  MonoidHom₁  (G  H  :  Type)  [Monoid  G]  [Monoid  H]  where
  toFun  :  G  →  H
  map_one  :  toFun  1  =  1
  map_mul  :  ∀  g  g',  toFun  (g  *  g')  =  toFun  g  *  toFun  g' 
```

当然，我们不想在所有地方都输入`toFun`，所以我们使用`CoeFun`类型类注册了一个强制转换。它的第一个参数是我们想要强制转换为函数的类型。第二个参数描述了目标函数类型。在我们的情况下，对于每个`f : MonoidHom₁ G H`，它总是`G → H`。我们还使用`coe`属性标记`MonoidHom₁.toFun`，以确保它在战术状态中几乎不可见，只需一个`↑`前缀即可。

```py
instance  [Monoid  G]  [Monoid  H]  :  CoeFun  (MonoidHom₁  G  H)  (fun  _  ↦  G  →  H)  where
  coe  :=  MonoidHom₁.toFun

attribute  [coe]  MonoidHom₁.toFun 
```

让我们检查我们是否真的可以将一个捆绑的单群同态应用于一个元素。

```py
example  [Monoid  G]  [Monoid  H]  (f  :  MonoidHom₁  G  H)  :  f  1  =  1  :=  f.map_one 
```

我们可以用其他类型的形态做同样的事情，直到我们达到环同态。

```py
@[ext]
structure  AddMonoidHom₁  (G  H  :  Type)  [AddMonoid  G]  [AddMonoid  H]  where
  toFun  :  G  →  H
  map_zero  :  toFun  0  =  0
  map_add  :  ∀  g  g',  toFun  (g  +  g')  =  toFun  g  +  toFun  g'

instance  [AddMonoid  G]  [AddMonoid  H]  :  CoeFun  (AddMonoidHom₁  G  H)  (fun  _  ↦  G  →  H)  where
  coe  :=  AddMonoidHom₁.toFun

attribute  [coe]  AddMonoidHom₁.toFun

@[ext]
structure  RingHom₁  (R  S  :  Type)  [Ring  R]  [Ring  S]  extends  MonoidHom₁  R  S,  AddMonoidHom₁  R  S 
```

这种方法有几个问题。一个较小的问题是，我们不知道将 `coe` 属性放在哪里，因为 `RingHom₁.toFun` 不存在，相关的函数是 `MonoidHom₁.toFun ∘ RingHom₁.toMonoidHom₁`，这不是一个可以标记属性的声明（但我们仍然可以定义一个 `CoeFun  (RingHom₁ R S) (fun _ ↦ R → S)` 实例）。一个更重要的问题是，关于单群同态的引理不会直接适用于环同态。这留下了两种选择：每次想要应用单群同态引理时都玩弄 `RingHom₁.toMonoidHom₁`，或者为环同态重新陈述每个这样的引理。这两种选择都不吸引人，因此 Mathlib 在这里使用了一个新的层次技巧。想法是为至少是单群同态的对象定义一个类型类，用单群同态和环同态实例化这个类，并使用它来陈述每个引理。在下面的定义中，`F` 可以是 `MonoidHom₁ M N`，或者如果 `M` 和 `N` 有环结构，则是 `RingHom₁ M N`。

```py
class  MonoidHomClass₁  (F  :  Type)  (M  N  :  Type)  [Monoid  M]  [Monoid  N]  where
  toFun  :  F  →  M  →  N
  map_one  :  ∀  f  :  F,  toFun  f  1  =  1
  map_mul  :  ∀  f  g  g',  toFun  f  (g  *  g')  =  toFun  f  g  *  toFun  f  g' 
```

然而，上述实现有一个问题。我们还没有注册一个转换到函数实例的转换。让我们现在尝试做这件事。

```py
def  badInst  [Monoid  M]  [Monoid  N]  [MonoidHomClass₁  F  M  N]  :  CoeFun  F  (fun  _  ↦  M  →  N)  where
  coe  :=  MonoidHomClass₁.toFun 
```

将其作为一个实例是糟糕的。当面对像 `f x` 这样的东西，其中 `f` 的类型不是函数类型时，Lean 会尝试找到一个 `CoeFun` 实例来将 `f` 转换为函数。上述函数的类型是：`{M N F : Type} → [Monoid M] → [Monoid N] → [MonoidHomClass₁ F M N] → CoeFun F (fun x ↦ M → N)`，因此，当它尝试应用它时，Lean 并不清楚未知类型 `M`、`N` 和 `F` 应该以什么顺序推断。这是一种与我们已经看到的略有不同的不良实例，但归结为同一个问题：不知道 `M`，Lean 就必须在一个未知类型上搜索单群实例，因此绝望地尝试数据库中的每个单群实例。如果你好奇想看到这种实例的效果，你可以在上述声明顶部键入 `set_option synthInstance.checkSynthOrder false in`，将 `def badInst` 替换为 `instance`，并在这个文件中寻找随机的失败。

在这种情况下，解决方案很简单，我们需要告诉 Lean 首先搜索 `F` 是什么，然后推断 `M` 和 `N`。这是通过使用 `outParam` 函数来完成的。这个函数定义为恒等函数，但仍然被类型类机制识别并触发所需的行为。因此，我们可以重新定义我们的类，注意 `outParam` 函数：

```py
class  MonoidHomClass₂  (F  :  Type)  (M  N  :  outParam  Type)  [Monoid  M]  [Monoid  N]  where
  toFun  :  F  →  M  →  N
  map_one  :  ∀  f  :  F,  toFun  f  1  =  1
  map_mul  :  ∀  f  g  g',  toFun  f  (g  *  g')  =  toFun  f  g  *  toFun  f  g'

instance  [Monoid  M]  [Monoid  N]  [MonoidHomClass₂  F  M  N]  :  CoeFun  F  (fun  _  ↦  M  →  N)  where
  coe  :=  MonoidHomClass₂.toFun

attribute  [coe]  MonoidHomClass₂.toFun 
```

现在我们可以继续我们的计划来实例化这个类。

```py
instance  (M  N  :  Type)  [Monoid  M]  [Monoid  N]  :  MonoidHomClass₂  (MonoidHom₁  M  N)  M  N  where
  toFun  :=  MonoidHom₁.toFun
  map_one  :=  fun  f  ↦  f.map_one
  map_mul  :=  fun  f  ↦  f.map_mul

instance  (R  S  :  Type)  [Ring  R]  [Ring  S]  :  MonoidHomClass₂  (RingHom₁  R  S)  R  S  where
  toFun  :=  fun  f  ↦  f.toMonoidHom₁.toFun
  map_one  :=  fun  f  ↦  f.toMonoidHom₁.map_one
  map_mul  :=  fun  f  ↦  f.toMonoidHom₁.map_mul 
```

如同承诺，我们关于 `f : F` 的每个证明，假设 `MonoidHomClass₁ F` 的一个实例，都将适用于单群同态和环同态。让我们看一个示例引理并检查它是否适用于这两种情况。

```py
lemma  map_inv_of_inv  [Monoid  M]  [Monoid  N]  [MonoidHomClass₂  F  M  N]  (f  :  F)  {m  m'  :  M}  (h  :  m*m'  =  1)  :
  f  m  *  f  m'  =  1  :=  by
  rw  [←  MonoidHomClass₂.map_mul,  h,  MonoidHomClass₂.map_one]

example  [Monoid  M]  [Monoid  N]  (f  :  MonoidHom₁  M  N)  {m  m'  :  M}  (h  :  m*m'  =  1)  :  f  m  *  f  m'  =  1  :=
map_inv_of_inv  f  h

example  [Ring  R]  [Ring  S]  (f  :  RingHom₁  R  S)  {r  r'  :  R}  (h  :  r*r'  =  1)  :  f  r  *  f  r'  =  1  :=
map_inv_of_inv  f  h 
```

初看起来，我们可能觉得我们又回到了将 `MonoidHom₁` 作为类的老想法。但我们并没有。一切都被提升了一个抽象层次。类型类解析过程不会寻找函数，它将寻找 `MonoidHom₁` 或 `RingHom₁`。

我们的方法中还有一个问题，那就是 `toFun` 字段周围的重复代码以及相应的 `CoeFun` 实例和 `coe` 属性。最好也记录这种模式仅用于具有额外属性的功能，这意味着函数的强制转换应该是单射的。因此，Mathlib 通过基类 `DFunLike`（其中“DFun”代表依赖函数）添加了一个抽象层。让我们在基层之上重新定义我们的 `MonoidHomClass`。

```py
class  MonoidHomClass₃  (F  :  Type)  (M  N  :  outParam  Type)  [Monoid  M]  [Monoid  N]  extends
  DFunLike  F  M  (fun  _  ↦  N)  where
  map_one  :  ∀  f  :  F,  f  1  =  1
  map_mul  :  ∀  (f  :  F)  g  g',  f  (g  *  g')  =  f  g  *  f  g'

instance  (M  N  :  Type)  [Monoid  M]  [Monoid  N]  :  MonoidHomClass₃  (MonoidHom₁  M  N)  M  N  where
  coe  :=  MonoidHom₁.toFun
  coe_injective'  _  _  :=  MonoidHom₁.ext
  map_one  :=  MonoidHom₁.map_one
  map_mul  :=  MonoidHom₁.map_mul 
```

当然，形态的层次结构并没有在这里停止。我们可以继续定义一个扩展 `MonoidHomClass₃` 的类 `RingHomClass₃`，并在 `RingHom` 上实例化它，然后稍后在其上实例化 `AlgebraHom`（代数是具有额外结构的环）。但我们已经涵盖了 Mathlib 中用于形态的主要形式化思想，你应该准备好理解 Mathlib 中形态是如何定义的。

作为练习，你应该尝试定义你的有序类型之间打包的保持顺序的函数类，然后定义保持顺序的幺半群同态。这仅用于训练目的。像连续函数一样，保持顺序的函数在 Mathlib 中主要是未打包的，它们由 `Monotone` 谓词定义。当然，你需要完成下面的类定义。

```py
@[ext]
structure  OrderPresHom  (α  β  :  Type)  [LE  α]  [LE  β]  where
  toFun  :  α  →  β
  le_of_le  :  ∀  a  a',  a  ≤  a'  →  toFun  a  ≤  toFun  a'

@[ext]
structure  OrderPresMonoidHom  (M  N  :  Type)  [Monoid  M]  [LE  M]  [Monoid  N]  [LE  N]  extends
MonoidHom₁  M  N,  OrderPresHom  M  N

class  OrderPresHomClass  (F  :  Type)  (α  β  :  outParam  Type)  [LE  α]  [LE  β]

instance  (α  β  :  Type)  [LE  α]  [LE  β]  :  OrderPresHomClass  (OrderPresHom  α  β)  α  β  where

instance  (α  β  :  Type)  [LE  α]  [Monoid  α]  [LE  β]  [Monoid  β]  :
  OrderPresHomClass  (OrderPresMonoidHom  α  β)  α  β  where

instance  (α  β  :  Type)  [LE  α]  [Monoid  α]  [LE  β]  [Monoid  β]  :
  MonoidHomClass₃  (OrderPresMonoidHom  α  β)  α  β
  :=  sorry 
```  ## 8.3\. 子对象

在定义了一些代数结构和其形态之后，下一步是考虑继承这种代数结构的集合，例如子群或子环。这很大程度上与我们的前一个主题重叠。实际上，`X` 中的集合被实现为一个从 `X` 到 `Prop` 的函数，因此子对象是满足一定谓词的函数。因此，我们可以重用导致 `DFunLike` 类及其后代的许多想法。我们不会重用 `DFunLike` 本身，因为这会打破从 `Set X` 到 `X → Prop` 的抽象障碍。取而代之的是，有一个 `SetLike` 类。该类不是将注入包装到函数类型中，而是将注入包装到 `Set` 类型中，并定义相应的强制转换和 `Membership` 实例。

```py
@[ext]
structure  Submonoid₁  (M  :  Type)  [Monoid  M]  where
  /-- The carrier of a submonoid. -/
  carrier  :  Set  M
  /-- The product of two elements of a submonoid belongs to the submonoid. -/
  mul_mem  {a  b}  :  a  ∈  carrier  →  b  ∈  carrier  →  a  *  b  ∈  carrier
  /-- The unit element belongs to the submonoid. -/
  one_mem  :  1  ∈  carrier

/-- Submonoids in `M` can be seen as sets in `M`. -/
instance  [Monoid  M]  :  SetLike  (Submonoid₁  M)  M  where
  coe  :=  Submonoid₁.carrier
  coe_injective'  _  _  :=  Submonoid₁.ext 
```

配备了上述 `SetLike` 实例，我们无需使用 `N.carrier` 就可以自然地声明子幺半群 `N` 包含 `1`。我们还可以在 `M` 中默默地将 `N` 视为一个集合，并取其在映射下的直接像。

```py
example  [Monoid  M]  (N  :  Submonoid₁  M)  :  1  ∈  N  :=  N.one_mem

example  [Monoid  M]  (N  :  Submonoid₁  M)  (α  :  Type)  (f  :  M  →  α)  :=  f  ''  N 
```

我们还有一个使用 `Subtype` 的 `Type` 强制转换，因此，给定一个子幺半群 `N`，我们可以写一个参数 `(x : N)`，它可以被强制转换为属于 `M` 且属于 `N` 的一个元素。

```py
example  [Monoid  M]  (N  :  Submonoid₁  M)  (x  :  N)  :  (x  :  M)  ∈  N  :=  x.property 
```

使用这种到 `Type` 的强制转换，我们还可以处理给子群配备幺半群结构的问题。我们将使用与上面 `N` 关联的类型相关的强制转换，以及断言这种强制转换是单射的引理 `SetCoe.ext`。这两个都由 `SetLike` 实例提供。

```py
instance  SubMonoid₁Monoid  [Monoid  M]  (N  :  Submonoid₁  M)  :  Monoid  N  where
  mul  :=  fun  x  y  ↦  ⟨x*y,  N.mul_mem  x.property  y.property⟩
  mul_assoc  :=  fun  x  y  z  ↦  SetCoe.ext  (mul_assoc  (x  :  M)  y  z)
  one  :=  ⟨1,  N.one_mem⟩
  one_mul  :=  fun  x  ↦  SetCoe.ext  (one_mul  (x  :  M))
  mul_one  :=  fun  x  ↦  SetCoe.ext  (mul_one  (x  :  M)) 
```

注意，在上面的实例中，我们不是使用到 `M` 的强制转换并调用 `property` 字段，而是可以使用如下结构化绑定。

```py
example  [Monoid  M]  (N  :  Submonoid₁  M)  :  Monoid  N  where
  mul  :=  fun  ⟨x,  hx⟩  ⟨y,  hy⟩  ↦  ⟨x*y,  N.mul_mem  hx  hy⟩
  mul_assoc  :=  fun  ⟨x,  _⟩  ⟨y,  _⟩  ⟨z,  _⟩  ↦  SetCoe.ext  (mul_assoc  x  y  z)
  one  :=  ⟨1,  N.one_mem⟩
  one_mul  :=  fun  ⟨x,  _⟩  ↦  SetCoe.ext  (one_mul  x)
  mul_one  :=  fun  ⟨x,  _⟩  ↦  SetCoe.ext  (mul_one  x) 
```

为了将关于子群或子环的引理应用于子群，我们需要一个类，就像对于态射一样。注意这个类接受一个 `SetLike` 实例作为参数，因此它不需要载体域并且可以在其字段中使用成员符号。

```py
class  SubmonoidClass₁  (S  :  Type)  (M  :  Type)  [Monoid  M]  [SetLike  S  M]  :  Prop  where
  mul_mem  :  ∀  (s  :  S)  {a  b  :  M},  a  ∈  s  →  b  ∈  s  →  a  *  b  ∈  s
  one_mem  :  ∀  s  :  S,  1  ∈  s

instance  [Monoid  M]  :  SubmonoidClass₁  (Submonoid₁  M)  M  where
  mul_mem  :=  Submonoid₁.mul_mem
  one_mem  :=  Submonoid₁.one_mem 
```

作为练习，你应该定义一个 `Subgroup₁` 结构，给它配备一个 `SetLike` 实例和一个 `SubmonoidClass₁` 实例，在 `Subgroup₁` 关联的子类型上放置一个 `Group` 实例，并定义一个 `SubgroupClass₁` 类。

在 Mathlib 中，关于给定代数对象的子对象总是形成一个完备格，并且这个结构被大量使用。例如，你可能想要寻找一个说子群的交集是一个子群的引理。但这不会是一个引理，而是一个下确界构造。让我们来看两个子群的情况。

```py
instance  [Monoid  M]  :  Min  (Submonoid₁  M)  :=
  ⟨fun  S₁  S₂  ↦
  {  carrier  :=  S₁  ∩  S₂
  one_mem  :=  ⟨S₁.one_mem,  S₂.one_mem⟩
  mul_mem  :=  fun  ⟨hx,  hx'⟩  ⟨hy,  hy'⟩  ↦  ⟨S₁.mul_mem  hx  hy,  S₂.mul_mem  hx'  hy'⟩  }⟩ 
```

这允许我们得到两个子群的交集作为一个子群。

```py
example  [Monoid  M]  (N  P  :  Submonoid₁  M)  :  Submonoid₁  M  :=  N  ⊓  P 
```

你可能会认为我们不得不在上面的例子中使用下确界符号 `⊓` 而不是交集符号 `∩` 是一种遗憾。但想想上确界。两个子群的并集不是一个子群。然而，子群仍然形成一个格（甚至是一个完备格）。实际上，`N ⊔ P` 是由 `N` 和 `P` 的并集生成的子群，当然用 `N ∪ P` 来表示它会非常令人困惑。所以，你可以看到使用 `N ⊓ P` 的做法更加一致。它在各种类型的代数结构中也非常一致。一开始看到两个向量子空间 `E` 和 `F` 的和用 `E ⊔ F` 而不是 `E + F` 表示可能会觉得有点奇怪。但你会习惯的。很快，你将把 `E + F` 符号看作是一个干扰，强调 `E ⊔ F` 的元素可以写成 `E` 的一个元素和 `F` 的一个元素的和中，而不是强调 `E ⊔ F` 是包含 `E` 和 `F` 的最小向量子空间这一基本事实。

本章的最后一个主题是商的概念。我们再次想要解释在 Mathlib 中如何构建方便的符号并避免代码重复。在这里，主要的工具是 `HasQuotient` 类，它允许使用像 `M ⧸ N` 这样的符号。请注意，商符号 `⧸` 是一个特殊的 Unicode 字符，而不是常规的 ASCII 除法符号。

作为例子，我们将通过子幺半群构建一个交换幺半群的商，证明留给你们。在上一个例子中，你可以使用`Setoid.refl`，但它不会自动选择相关的`Setoid`结构。你可以通过使用`@`语法提供所有参数来解决这个问题，例如`@Setoid.refl M N.Setoid`。

```py
def  Submonoid.Setoid  [CommMonoid  M]  (N  :  Submonoid  M)  :  Setoid  M  where
  r  :=  fun  x  y  ↦  ∃  w  ∈  N,  ∃  z  ∈  N,  x*w  =  y*z
  iseqv  :=  {
  refl  :=  fun  x  ↦  ⟨1,  N.one_mem,  1,  N.one_mem,  rfl⟩
  symm  :=  fun  ⟨w,  hw,  z,  hz,  h⟩  ↦  ⟨z,  hz,  w,  hw,  h.symm⟩
  trans  :=  by
  sorry
  }

instance  [CommMonoid  M]  :  HasQuotient  M  (Submonoid  M)  where
  quotient'  :=  fun  N  ↦  Quotient  N.Setoid

def  QuotientMonoid.mk  [CommMonoid  M]  (N  :  Submonoid  M)  :  M  →  M  ⧸  N  :=  Quotient.mk  N.Setoid

instance  [CommMonoid  M]  (N  :  Submonoid  M)  :  Monoid  (M  ⧸  N)  where
  mul  :=  Quotient.map₂  (·  *  ·)  (by
  sorry
  )
  mul_assoc  :=  by
  sorry
  one  :=  QuotientMonoid.mk  N  1
  one_mul  :=  by
  sorry
  mul_one  :=  by
  sorry 
```  ## 8.1\. 基础

在 Lean 的所有层次结构的底部，我们找到了携带数据的类。以下类记录了给定的类型`α`被赋予了一个称为`one`的特称元素。在这个阶段，它没有任何属性。

```py
class  One₁  (α  :  Type)  where
  /-- The element one -/
  one  :  α 
```

由于我们将在本章中大量使用类，我们需要了解一些关于`class`命令做了什么的更多细节。首先，上面的`class`命令定义了一个带有参数`α : Type`和单个字段`one`的结构`One₁`。它还标记这个结构为类，以便某些类型`α`的`One₁ α`的参数可以通过实例解析过程推断出来，只要它们被标记为实例隐式，即出现在方括号之间。这两个效果也可以通过带有`class`属性的`structure`命令来实现，即写作`@[class] structure`实例`class`。但是，类命令还确保`One₁ α`在其自己的字段中作为实例隐式参数出现。比较：

```py
#check  One₁.one  -- One₁.one {α : Type} [self : One₁ α] : α

@[class]  structure  One₂  (α  :  Type)  where
  /-- The element one -/
  one  :  α

#check  One₂.one 
```

在第二个检查中，我们可以看到`self : One₂ α`是一个显式参数。让我们确保第一个版本确实可以在没有任何显式参数的情况下使用。

```py
example  (α  :  Type)  [One₁  α]  :  α  :=  One₁.one 
```

备注：在上面的例子中，参数`One₁ α`被标记为实例隐式，这有点愚蠢，因为这只影响声明的*使用*以及由`example`命令创建的声明不能被使用。然而，它允许我们避免给这个参数命名，更重要的是，它开始养成将`One₁ α`参数标记为实例隐式的良好习惯。

另一个需要注意的是，所有这些只有在 Lean 知道`α`是什么的情况下才会工作。在上面的例子中，省略类型注解`: α`将生成一个错误信息，例如：`typeclass instance problem is stuck, it is often due to metavariables One₁ (?m.263 α)`，其中`?m.263 α`表示“依赖于`α`的某些类型”（而 263 只是一个自动生成的索引，可以用来区分几个未知的事物）。避免这个问题的另一种方法是在使用类型注解，如下所示：

```py
example  (α  :  Type)  [One₁  α]  :=  (One₁.one  :  α) 
```

你可能在第 3.6 节中玩序列的极限时已经遇到过这个问题，如果你试图声明例如`0 < 1`，但没有告诉 Lean 你是指自然数还是实数的这个不等式。

我们接下来的任务是给 `One₁.one` 分配一个符号。由于我们不希望与内置的 `1` 的符号发生冲突，我们将使用 `𝟙`。这是通过以下命令实现的，其中第一行告诉 Lean 使用 `One₁.one` 的文档作为符号 `𝟙` 的文档。

```py
@[inherit_doc]
notation  "𝟙"  =>  One₁.one

example  {α  :  Type}  [One₁  α]  :  α  :=  𝟙

example  {α  :  Type}  [One₁  α]  :  (𝟙  :  α)  =  𝟙  :=  rfl 
```

我们现在想要一个携带数据的类来记录二元操作。目前我们不想在加法和乘法之间做出选择，所以我们将使用菱形。

```py
class  Dia₁  (α  :  Type)  where
  dia  :  α  →  α  →  α

infixl:70  " ⋄ "  =>  Dia₁.dia 
```

与 `One₁` 例子一样，这个操作在这个阶段没有任何属性。现在让我们定义一个半群结构的类，其中操作用 `⋄` 表示。目前，我们通过手动定义一个具有两个字段的结构来定义它，一个 `Dia₁` 实例和一些 `Prop` 类型的字段 `dia_assoc`，断言 `⋄` 的结合性。

```py
class  Semigroup₀  (α  :  Type)  where
  toDia₁  :  Dia₁  α
  /-- Diamond is associative -/
  dia_assoc  :  ∀  a  b  c  :  α,  a  ⋄  b  ⋄  c  =  a  ⋄  (b  ⋄  c) 
```

注意，在陈述 dia_assoc 时，之前定义的字段 toDia₁ 位于局部上下文中，因此当 Lean 搜索 Dia₁ α 的实例以理解 ⋄ b 时可以使用。然而，这个 toDia₁ 字段不会成为类型类实例数据库的一部分。因此，执行 `example {α : Type} [Semigroup₁ α] (a b : α) : α := a ⋄ b` 会失败，错误信息为 `failed to synthesize instance Dia₁ α`。

我们可以通过稍后添加 `instance` 属性来修复这个问题。

```py
attribute  [instance]  Semigroup₀.toDia₁

example  {α  :  Type}  [Semigroup₀  α]  (a  b  :  α)  :  α  :=  a  ⋄  b 
```

在构建之前，我们需要使用不同的语法来添加这个 toDia₁ 字段，告诉 Lean Dia₁ α 应该被视为其字段是 Semigroup₁ 自身的字段。这也方便地自动添加了 toDia₁ 实例。`class` 命令使用 `extends` 语法支持这一点，如下所示：

```py
class  Semigroup₁  (α  :  Type)  extends  toDia₁  :  Dia₁  α  where
  /-- Diamond is associative -/
  dia_assoc  :  ∀  a  b  c  :  α,  a  ⋄  b  ⋄  c  =  a  ⋄  (b  ⋄  c)

example  {α  :  Type}  [Semigroup₁  α]  (a  b  :  α)  :  α  :=  a  ⋄  b 
```

注意，此语法也适用于 `structure` 命令，尽管在这种情况下，它仅解决了编写 toDia₁ 等字段的问题，因为在那种情况下没有实例要定义。

在 `extends` 语法中，字段名 toDia₁ 是可选的。默认情况下，它采用被扩展的类的名称，并在其前面加上“to”。

```py
class  Semigroup₂  (α  :  Type)  extends  Dia₁  α  where
  /-- Diamond is associative -/
  dia_assoc  :  ∀  a  b  c  :  α,  a  ⋄  b  ⋄  c  =  a  ⋄  (b  ⋄  c) 
```

现在我们尝试将一个菱形操作和一个特殊元素结合起来，通过公理说明这个元素在两边都是中性的。

```py
class  DiaOneClass₁  (α  :  Type)  extends  One₁  α,  Dia₁  α  where
  /-- One is a left neutral element for diamond. -/
  one_dia  :  ∀  a  :  α,  𝟙  ⋄  a  =  a
  /-- One is a right neutral element for diamond -/
  dia_one  :  ∀  a  :  α,  a  ⋄  𝟙  =  a 
```

在下一个例子中，我们告诉 Lean `α` 具有结构 `DiaOneClass₁`，并陈述了一个使用 Dia₁ 实例和 One₁ 实例的属性。为了了解 Lean 如何找到这些实例，我们设置了一个跟踪选项，其结果可以在 Infoview 中查看。默认情况下，这个结果相当简短，但可以通过点击以黑色箭头结束的行来展开。它包括 Lean 在成功之前尝试找到实例的失败尝试。成功的尝试确实涉及由 `extends` 语法生成的实例。

```py
set_option  trace.Meta.synthInstance  true  in
example  {α  :  Type}  [DiaOneClass₁  α]  (a  b  :  α)  :  Prop  :=  a  ⋄  b  =  𝟙 
```

注意，在组合现有类时，我们不需要包含额外的字段。因此，我们可以定义幺半群如下：

```py
class  Monoid₁  (α  :  Type)  extends  Semigroup₁  α,  DiaOneClass₁  α 
```

虽然上述定义看起来很简单，但它隐藏了一个重要的微妙之处。`Semigroup₁ α` 和 `DiaOneClass₁ α` 都扩展了 `Dia₁ α`，因此人们可能会担心拥有一个 `Monoid₁ α` 实例会给出两个与 `α` 无关的菱形操作，一个来自字段 `Monoid₁.toSemigroup₁`，另一个来自字段 `Monoid₁.toDiaOneClass₁`。

事实上，如果我们尝试手动构建一个幺半群类，使用：

```py
class  Monoid₂  (α  :  Type)  where
  toSemigroup₁  :  Semigroup₁  α
  toDiaOneClass₁  :  DiaOneClass₁  α 
```

那么，我们得到两个完全无关的菱形操作 `Monoid₂.toSemigroup₁.toDia₁.dia` 和 `Monoid₂.toDiaOneClass₁.toDia₁.dia`。

使用 `extends` 语法生成的版本没有这个缺陷。

```py
example  {α  :  Type}  [Monoid₁  α]  :
  (Monoid₁.toSemigroup₁.toDia₁.dia  :  α  →  α  →  α)  =  Monoid₁.toDiaOneClass₁.toDia₁.dia  :=  rfl 
```

因此，`class` 命令为我们做了一些魔法（`structure` 命令也会这样做）。查看我们类的字段的一个简单方法是比较它们的构造函数。比较：

```py
/- Monoid₂.mk {α : Type} (toSemigroup₁ : Semigroup₁ α) (toDiaOneClass₁ : DiaOneClass₁ α) : Monoid₂ α -/
#check  Monoid₂.mk

/- Monoid₁.mk {α : Type} [toSemigroup₁ : Semigroup₁ α] [toOne₁ : One₁ α] (one_dia : ∀ (a : α), 𝟙 ⋄ a = a) (dia_one : ∀ (a : α), a ⋄ 𝟙 = a) : Monoid₁ α -/
#check  Monoid₁.mk 
```

因此我们看到 `Monoid₁` 按预期接受 `Semigroup₁ α` 参数，但它不会接受一个潜在的重复的 `DiaOneClass₁ α` 参数，而是将其拆分并只包含非重叠的部分。它还自动生成了一个实例 `Monoid₁.toDiaOneClass₁`，这不是一个字段，但它具有预期的签名，从最终用户的角度来看，它恢复了两个扩展类 `Semigroup₁` 和 `DiaOneClass₁` 之间的对称性。

```py
#check  Monoid₁.toSemigroup₁
#check  Monoid₁.toDiaOneClass₁ 
```

我们现在非常接近定义群了。我们可以在幺半群结构中添加一个字段，断言每个元素的存在性。但那时我们需要努力来访问这些逆元。在实践中，将其作为数据添加更方便。为了优化可重用性，我们定义了一个新的数据承载类，然后给它一些符号。

```py
class  Inv₁  (α  :  Type)  where
  /-- The inversion function -/
  inv  :  α  →  α

@[inherit_doc]
postfix:max  "⁻¹"  =>  Inv₁.inv

class  Group₁  (G  :  Type)  extends  Monoid₁  G,  Inv₁  G  where
  inv_dia  :  ∀  a  :  G,  a⁻¹  ⋄  a  =  𝟙 
```

上述定义可能看起来太弱了，我们只要求 `a⁻¹` 是 `a` 的左逆。但另一方面是自动的。为了证明这一点，我们需要一个初步的引理。

```py
lemma  left_inv_eq_right_inv₁  {M  :  Type}  [Monoid₁  M]  {a  b  c  :  M}  (hba  :  b  ⋄  a  =  𝟙)  (hac  :  a  ⋄  c  =  𝟙)  :  b  =  c  :=  by
  rw  [←  DiaOneClass₁.one_dia  c,  ←  hba,  Semigroup₁.dia_assoc,  hac,  DiaOneClass₁.dia_one  b] 
```

在这个引理中，给出全名相当令人烦恼，尤其是因为它需要知道哪个部分提供了这些事实。一种修复方法是使用 `export` 命令将这些事实作为根名称空间中的引理复制。

```py
export  DiaOneClass₁  (one_dia  dia_one)
export  Semigroup₁  (dia_assoc)
export  Group₁  (inv_dia) 
```

然后，我们可以将上述证明重写为：

```py
example  {M  :  Type}  [Monoid₁  M]  {a  b  c  :  M}  (hba  :  b  ⋄  a  =  𝟙)  (hac  :  a  ⋄  c  =  𝟙)  :  b  =  c  :=  by
  rw  [←  one_dia  c,  ←  hba,  dia_assoc,  hac,  dia_one  b] 
```

现在轮到你来证明关于我们的代数结构的事实了。

```py
lemma  inv_eq_of_dia  [Group₁  G]  {a  b  :  G}  (h  :  a  ⋄  b  =  𝟙)  :  a⁻¹  =  b  :=
  sorry

lemma  dia_inv  [Group₁  G]  (a  :  G)  :  a  ⋄  a⁻¹  =  𝟙  :=
  sorry 
```

在这个阶段，我们希望继续定义环，但有一个严重的问题。一个类型上的环结构包含一个加法群结构和乘法幺半群结构，以及关于它们之间相互作用的某些性质。但到目前为止，我们为所有操作硬编码了一个符号`⋄`。更基本的是，类型类系统假设每个类型只有一个类型类的实例。有各种方法可以解决这个问题。令人惊讶的是，Mathlib 使用一种原始的想法，通过一些代码生成属性来为加法和乘法理论复制一切。结构和类都在加法和乘法符号下定义，并通过属性`to_additive`将它们链接起来。在像半群这样的多重继承的情况下，自动生成的“对称恢复”实例也需要标记。这有点技术性；你不需要理解细节。重要的是，引理只以乘法符号的形式陈述，并标记了`to_additive`属性以生成加法版本`left_inv_eq_right_inv'`及其自动生成的加法版本`left_neg_eq_right_neg'`。为了检查这个加法版本的名字，我们在`left_inv_eq_right_inv'`的顶部使用了`whatsnew in`命令。

```py
class  AddSemigroup₃  (α  :  Type)  extends  Add  α  where
  /-- Addition is associative -/
  add_assoc₃  :  ∀  a  b  c  :  α,  a  +  b  +  c  =  a  +  (b  +  c)

@[to_additive  AddSemigroup₃]
class  Semigroup₃  (α  :  Type)  extends  Mul  α  where
  /-- Multiplication is associative -/
  mul_assoc₃  :  ∀  a  b  c  :  α,  a  *  b  *  c  =  a  *  (b  *  c)

class  AddMonoid₃  (α  :  Type)  extends  AddSemigroup₃  α,  AddZeroClass  α

@[to_additive  AddMonoid₃]
class  Monoid₃  (α  :  Type)  extends  Semigroup₃  α,  MulOneClass  α

export  Semigroup₃  (mul_assoc₃)
export  AddSemigroup₃  (add_assoc₃)

whatsnew  in
@[to_additive]
lemma  left_inv_eq_right_inv'  {M  :  Type}  [Monoid₃  M]  {a  b  c  :  M}  (hba  :  b  *  a  =  1)  (hac  :  a  *  c  =  1)  :  b  =  c  :=  by
  rw  [←  one_mul  c,  ←  hba,  mul_assoc₃,  hac,  mul_one  b]

#check  left_neg_eq_right_neg' 
```

有了这项技术，我们也可以轻松地定义交换半群、幺半群和群，然后定义环。

```py
class  AddCommSemigroup₃  (α  :  Type)  extends  AddSemigroup₃  α  where
  add_comm  :  ∀  a  b  :  α,  a  +  b  =  b  +  a

@[to_additive  AddCommSemigroup₃]
class  CommSemigroup₃  (α  :  Type)  extends  Semigroup₃  α  where
  mul_comm  :  ∀  a  b  :  α,  a  *  b  =  b  *  a

class  AddCommMonoid₃  (α  :  Type)  extends  AddMonoid₃  α,  AddCommSemigroup₃  α

@[to_additive  AddCommMonoid₃]
class  CommMonoid₃  (α  :  Type)  extends  Monoid₃  α,  CommSemigroup₃  α

class  AddGroup₃  (G  :  Type)  extends  AddMonoid₃  G,  Neg  G  where
  neg_add  :  ∀  a  :  G,  -a  +  a  =  0

@[to_additive  AddGroup₃]
class  Group₃  (G  :  Type)  extends  Monoid₃  G,  Inv  G  where
  inv_mul  :  ∀  a  :  G,  a⁻¹  *  a  =  1 
```

我们应该记得在适当的时候给引理标记`simp`。

```py
attribute  [simp]  Group₃.inv_mul  AddGroup₃.neg_add 
```

然后我们需要重复一下，因为我们切换到标准符号，但至少`to_additive`完成了从乘法符号到加法符号的翻译工作。

```py
@[to_additive]
lemma  inv_eq_of_mul  [Group₃  G]  {a  b  :  G}  (h  :  a  *  b  =  1)  :  a⁻¹  =  b  :=
  sorry 
```

注意，`to_additive`可以要求给一个引理标记`simp`并将该属性传播到加法版本，如下所示。

```py
@[to_additive  (attr  :=  simp)]
lemma  Group₃.mul_inv  {G  :  Type}  [Group₃  G]  {a  :  G}  :  a  *  a⁻¹  =  1  :=  by
  sorry

@[to_additive]
lemma  mul_left_cancel₃  {G  :  Type}  [Group₃  G]  {a  b  c  :  G}  (h  :  a  *  b  =  a  *  c)  :  b  =  c  :=  by
  sorry

@[to_additive]
lemma  mul_right_cancel₃  {G  :  Type}  [Group₃  G]  {a  b  c  :  G}  (h  :  b*a  =  c*a)  :  b  =  c  :=  by
  sorry

class  AddCommGroup₃  (G  :  Type)  extends  AddGroup₃  G,  AddCommMonoid₃  G

@[to_additive  AddCommGroup₃]
class  CommGroup₃  (G  :  Type)  extends  Group₃  G,  CommMonoid₃  G 
```

我们现在准备好处理环了。为了演示目的，我们不会假设加法是交换的，然后立即提供一个`AddCommGroup₃`的实例。Mathlib 不玩这种游戏，首先是因为在实际上这并不会让任何环实例更容易，而且 Mathlib 的代数层次结构通过半环，这些半环像环一样但没有相反数，所以下面的证明对它们不适用。我们在这里获得的好处，除了如果你从未见过它，是一个很好的练习之外，还有一个使用允许提供父结构作为实例参数然后提供额外字段的语法的构建实例的例子。这里 Ring₃ R 参数提供了 AddCommGroup₃ R 想要的任何东西，除了 add_comm。

```py
class  Ring₃  (R  :  Type)  extends  AddGroup₃  R,  Monoid₃  R,  MulZeroClass  R  where
  /-- Multiplication is left distributive over addition -/
  left_distrib  :  ∀  a  b  c  :  R,  a  *  (b  +  c)  =  a  *  b  +  a  *  c
  /-- Multiplication is right distributive over addition -/
  right_distrib  :  ∀  a  b  c  :  R,  (a  +  b)  *  c  =  a  *  c  +  b  *  c

instance  {R  :  Type}  [Ring₃  R]  :  AddCommGroup₃  R  :=
{  add_comm  :=  by
  sorry  } 
```

当然，我们也可以构建具体的实例，例如整数上的环结构（当然下面的实例使用的是 Mathlib 中所有的工作都已经完成）。

```py
instance  :  Ring₃  ℤ  where
  add  :=  (·  +  ·)
  add_assoc₃  :=  add_assoc
  zero  :=  0
  zero_add  :=  by  simp
  add_zero  :=  by  simp
  neg  :=  (-  ·)
  neg_add  :=  by  simp
  mul  :=  (·  *  ·)
  mul_assoc₃  :=  mul_assoc
  one  :=  1
  one_mul  :=  by  simp
  mul_one  :=  by  simp
  zero_mul  :=  by  simp
  mul_zero  :=  by  simp
  left_distrib  :=  Int.mul_add
  right_distrib  :=  Int.add_mul 
```

作为练习，你现在可以设置一个简单的层次结构来表示顺序关系，包括一个用于有序交换幺半群的类，它既有偏序结构，也有交换幺半群结构，使得`∀ a b : α, a ≤ b → ∀ c : α, c * a ≤ c * b`。当然，你需要为以下类添加字段和可能性的`extends`子句。

```py
class  LE₁  (α  :  Type)  where
  /-- The Less-or-Equal relation. -/
  le  :  α  →  α  →  Prop

@[inherit_doc]  infix:50  " ≤₁ "  =>  LE₁.le

class  Preorder₁  (α  :  Type)

class  PartialOrder₁  (α  :  Type)

class  OrderedCommMonoid₁  (α  :  Type)

instance  :  OrderedCommMonoid₁  ℕ  where 
```

我们现在想讨论涉及多个类型的代数结构。一个典型的例子是环上的模块。如果你不知道什么是模块，你可以假装它意味着向量空间，并认为我们所有的环都是域。这些结构是带有某些环元素的标量乘法的交换加法群。

我们首先定义由类型`α`在类型`β`上通过某种类型`α`进行标量乘法的携带数据的类型类，并给它一个右结合的符号。

```py
class  SMul₃  (α  :  Type)  (β  :  Type)  where
  /-- Scalar multiplication -/
  smul  :  α  →  β  →  β

infixr:73  " • "  =>  SMul₃.smul 
```

然后，我们可以定义模块（如果你不知道什么是模块，再想想向量空间）。

```py
class  Module₁  (R  :  Type)  [Ring₃  R]  (M  :  Type)  [AddCommGroup₃  M]  extends  SMul₃  R  M  where
  zero_smul  :  ∀  m  :  M,  (0  :  R)  •  m  =  0
  one_smul  :  ∀  m  :  M,  (1  :  R)  •  m  =  m
  mul_smul  :  ∀  (a  b  :  R)  (m  :  M),  (a  *  b)  •  m  =  a  •  b  •  m
  add_smul  :  ∀  (a  b  :  R)  (m  :  M),  (a  +  b)  •  m  =  a  •  m  +  b  •  m
  smul_add  :  ∀  (a  :  R)  (m  n  :  M),  a  •  (m  +  n)  =  a  •  m  +  a  •  n 
```

这里有一些有趣的事情正在发生。虽然`R`上的环结构在这个定义中作为参数并不太令人惊讶，但你可能预计`AddCommGroup₃ M`将像`SMul₃ R M`一样成为`extends`子句的一部分。尝试这样做会导致一个听起来很神秘的错误信息：“无法为实例 Module₁.toAddCommGroup₃找到合成顺序，类型为(R : Type) → [inst : Ring₃ R] → {M : Type} → [self : Module₁ R M] → AddCommGroup₃ M，所有剩余的参数都有元变量：Ring₃ ?R @Module₁ ?R ?inst✝ M”。为了理解这条信息，你需要记住，这样的`extends`子句会导致一个标记为实例的字段`Module₃.toAddCommGroup₃`。这个实例将具有错误信息中出现的签名：`(R : Type) → [inst : Ring₃ R] → {M : Type} → [self : Module₁ R M] → AddCommGroup₃ M`。在类型类数据库中，每次 Lean 寻找某个`M`的`AddCommGroup₃ M`实例时，它都需要在开始寻找`Module₁ R M`实例的主要任务之前，去寻找一个完全未指定的类型`R`和一个`Ring₃ R`实例。这两个辅助任务是错误信息中提到的元变量所代表的，分别用`?R`和`?inst✝`表示。这样的`Module₃.toAddCommGroup₃`实例将成为实例解析过程的一个巨大陷阱，然后`class`命令拒绝设置它。

那么`extends SMul₃ R M`又是怎么回事呢？它创建了一个字段`Module₁.toSMul₃ : {R : Type} → [inst : Ring₃ R] → {M : Type} → [inst_1 : AddCommGroup₃ M] → [self : Module₁ R M] → SMul₃ R M`，其最终结果`SMul₃ R M`提到了`R`和`M`，因此这个字段可以安全地用作实例。规则很容易记住：`extends`子句中出现的每个类都应该提到参数中出现的每个类型。

让我们创建我们的第一个模块实例：一个环是其自身的模块，使用其乘法作为标量乘法。

```py
instance  selfModule  (R  :  Type)  [Ring₃  R]  :  Module₁  R  R  where
  smul  :=  fun  r  s  ↦  r*s
  zero_smul  :=  zero_mul
  one_smul  :=  one_mul
  mul_smul  :=  mul_assoc₃
  add_smul  :=  Ring₃.right_distrib
  smul_add  :=  Ring₃.left_distrib 
```

作为第二个例子，每个阿贝尔群都是 `ℤ` 上的模块（这是通过允许非可逆标量来推广向量空间理论的原因之一）。首先，可以为任何带有零和加法的类型定义自然数的标量乘法：`n • a` 被定义为 `a + ⋯ + a`，其中 `a` 出现 `n` 次。然后，通过确保 `(-1) • a = -a` 来将标量乘法扩展到整数。

```py
def  nsmul₁  {M  :  Type*}  [Zero  M]  [Add  M]  :  ℕ  →  M  →  M
  |  0,  _  =>  0
  |  n  +  1,  a  =>  a  +  nsmul₁  n  a

def  zsmul₁  {M  :  Type*}  [Zero  M]  [Add  M]  [Neg  M]  :  ℤ  →  M  →  M
  |  Int.ofNat  n,  a  =>  nsmul₁  n  a
  |  Int.negSucc  n,  a  =>  -nsmul₁  n.succ  a 
```

证明这一点以产生一个模块结构有点繁琐，并且对于当前的讨论来说并不有趣，所以我们将对所有公理表示歉意。你**不**需要用证明来替换这些歉意。如果你坚持这样做，你可能需要陈述并证明关于 `nsmul₁` 和 `zsmul₁` 的几个中间引理。

```py
instance  abGrpModule  (A  :  Type)  [AddCommGroup₃  A]  :  Module₁  ℤ  A  where
  smul  :=  zsmul₁
  zero_smul  :=  sorry
  one_smul  :=  sorry
  mul_smul  :=  sorry
  add_smul  :=  sorry
  smul_add  :=  sorry 
```

一个更加重要的问题是，我们现在在环 `ℤ` 上对 `ℤ` 本身有两个模块结构：`abGrpModule ℤ`，因为 `ℤ` 是一个阿贝尔群，以及 `selfModule ℤ`，因为 `ℤ` 是一个环。这两个模块结构对应于相同的阿贝尔群结构，但它们是否具有相同的标量乘法并不明显。实际上，它们确实如此，但这并不是定义上的，需要证明。这对类型类实例解析过程来说是个坏消息，并将导致用户在使用这个层次结构时遇到非常令人沮丧的失败。当直接要求找到一个实例时，Lean 会选择一个，我们可以通过以下方式看到它：

```py
#synth  Module₁  ℤ  ℤ  -- abGrpModule ℤ 
```

但在更间接的上下文中，可能会发生 Lean 推断出另一个，然后变得困惑的情况。这种情况被称为“坏钻石”。这与我们上面使用的钻石操作无关，它指的是从 `ℤ` 到其 `Module₁ ℤ` 的路径，可以通过 `AddCommGroup₃ ℤ` 或 `Ring₃ ℤ` 来绘制。

重要的是要理解并非所有钻石都是坏的。事实上，在 Mathlib 中到处都有钻石，在本章中也是如此。在非常开始的时候，我们就看到了可以从 `Monoid₁ α` 通过 `Semigroup₁ α` 或 `DiaOneClass₁ α` 到 `Dia₁ α` 的转换，并且由于 `class` 命令的工作，得到的两个 `Dia₁ α` 实例在定义上是相等的。特别是，底部有 `Prop` 值类的钻石不能是坏的，因为任何两个相同陈述的证明在定义上是相等的。

但我们用模块创建的钻石肯定是不好的。问题在于 `smul` 字段，它是数据，而不是证明，并且我们有两种不是定义上相等的构造。修复这个问题的稳健方法是确保从丰富结构到贫弱结构的转换总是通过忘记数据来完成，而不是通过定义数据。这个众所周知的模式被命名为“遗忘继承”，并在 [`inria.hal.science/hal-02463336v2`](https://inria.hal.science/hal-02463336v2) 中广泛讨论。

在我们的具体情况下，我们可以修改`AddMonoid₃`的定义，包括一个`nsmul`数据字段和一些`Prop`类型的字段，以确保这个操作是可证明的，与我们上面构造的是同一个。这些字段在下面的定义中使用`:=`在它们的类型后面赋予默认值。多亏了这些默认值，大多数实例将与我们之前的定义完全一样构建。但在`ℤ`的特殊情况下，我们将能够提供特定的值。

```py
class  AddMonoid₄  (M  :  Type)  extends  AddSemigroup₃  M,  AddZeroClass  M  where
  /-- Multiplication by a natural number. -/
  nsmul  :  ℕ  →  M  →  M  :=  nsmul₁
  /-- Multiplication by `(0 : ℕ)` gives `0`. -/
  nsmul_zero  :  ∀  x,  nsmul  0  x  =  0  :=  by  intros;  rfl
  /-- Multiplication by `(n + 1 : ℕ)` behaves as expected. -/
  nsmul_succ  :  ∀  (n  :  ℕ)  (x),  nsmul  (n  +  1)  x  =  x  +  nsmul  n  x  :=  by  intros;  rfl

instance  mySMul  {M  :  Type}  [AddMonoid₄  M]  :  SMul  ℕ  M  :=  ⟨AddMonoid₄.nsmul⟩ 
```

让我们检查我们是否仍然可以构建一个乘积幺半群实例，而不提供`nsmul`相关字段。

```py
instance  (M  N  :  Type)  [AddMonoid₄  M]  [AddMonoid₄  N]  :  AddMonoid₄  (M  ×  N)  where
  add  :=  fun  p  q  ↦  (p.1  +  q.1,  p.2  +  q.2)
  add_assoc₃  :=  fun  a  b  c  ↦  by  ext  <;>  apply  add_assoc₃
  zero  :=  (0,  0)
  zero_add  :=  fun  a  ↦  by  ext  <;>  apply  zero_add
  add_zero  :=  fun  a  ↦  by  ext  <;>  apply  add_zero 
```

现在让我们处理`ℤ`的特殊情况，我们想要使用`ℕ`到`ℤ`的强制转换和`ℤ`上的乘法来构建`nsmul`。特别注意的是，证明字段比上面的默认值包含更多的工作。

```py
instance  :  AddMonoid₄  ℤ  where
  add  :=  (·  +  ·)
  add_assoc₃  :=  Int.add_assoc
  zero  :=  0
  zero_add  :=  Int.zero_add
  add_zero  :=  Int.add_zero
  nsmul  :=  fun  n  m  ↦  (n  :  ℤ)  *  m
  nsmul_zero  :=  Int.zero_mul
  nsmul_succ  :=  fun  n  m  ↦  show  (n  +  1  :  ℤ)  *  m  =  m  +  n  *  m
  by  rw  [Int.add_mul,  Int.add_comm,  Int.one_mul] 
```

让我们检查我们是否解决了我们的问题。因为 Lean 已经有一个自然数和整数标量乘法的定义，我们想要确保我们的实例被使用，所以我们不会使用`•`符号，而是调用`SMul.mul`并明确提供我们上面定义的实例。

```py
example  (n  :  ℕ)  (m  :  ℤ)  :  SMul.smul  (self  :=  mySMul)  n  m  =  n  *  m  :=  rfl 
```

这个故事继续通过将`zsmul`字段纳入群的定义中，并使用类似的技巧。你现在可以阅读 Mathlib 中单群、群、环和模块的定义了。它们比我们这里看到的更复杂，因为它们是巨大层次结构的一部分，但所有原则都已经在上面解释过了。

作为练习，你可以回到你上面构建的顺序关系层次，尝试结合一个携带小于符号`<₁`的类型类`LT₁`，并确保每个偏序都带有`<₁`，它有一个从`≤₁`构建的默认值，以及一个`Prop`类型的字段，断言这两个比较运算符之间的自然关系。

## 8.2. 摩尔型

到目前为止，在本章中，我们讨论了如何创建数学结构的层次。但定义结构并不是真正完成，直到我们有摩尔型。这里有两种主要的方法。最明显的一个是定义一个关于函数的谓词。

```py
def  isMonoidHom₁  [Monoid  G]  [Monoid  H]  (f  :  G  →  H)  :  Prop  :=
  f  1  =  1  ∧  ∀  g  g',  f  (g  *  g')  =  f  g  *  f  g' 
```

在这个定义中，使用合取有点不愉快。特别是当用户想要访问两个条件时，他们需要记住我们选择的排序。因此，我们可以使用一个结构。

```py
structure  isMonoidHom₂  [Monoid  G]  [Monoid  H]  (f  :  G  →  H)  :  Prop  where
  map_one  :  f  1  =  1
  map_mul  :  ∀  g  g',  f  (g  *  g')  =  f  g  *  f  g' 
```

一旦我们到了这里，甚至有将其做成一个类并使用类型类实例解析过程自动推断出复杂函数的`isMonoidHom₂`的诱惑。例如，幺半群同态的复合是一个幺半群同态，这似乎是一个有用的实例。然而，这样的实例对于解析过程来说会非常棘手，因为它需要到处寻找`g ∘ f`。在`g (f x)`中看到它失败会非常沮丧。更普遍地说，我们必须始终记住，识别给定表达式中应用了哪个函数是一个非常困难的问题，称为“高阶统一问题”。因此，Mathlib 不使用这种类方法。

一个更基本的问题是，我们是否使用上述谓词（使用`def`或`structure`）或者使用捆绑函数和谓词的结构。这部分是一个心理问题。考虑一个不是同态的幺半群之间的函数是非常罕见的。这真的感觉像“幺半群同态”不是一个你可以赋予一个裸函数的形容词，它是一个名词。另一方面，有人可以争辩说，拓扑空间之间的连续函数实际上是一个恰好是连续的函数。这是 Mathlib 有`Continuous`谓词的一个原因。例如，你可以写：

```py
example  :  Continuous  (id  :  ℝ  →  ℝ)  :=  continuous_id 
```

我们仍然有连续函数的捆绑，这在例如给连续函数的空间赋予拓扑时很方便，但它们不是处理连续性的主要工具。

相比之下，幺半群（或其他代数结构）之间的同态捆绑如下：

```py
@[ext]
structure  MonoidHom₁  (G  H  :  Type)  [Monoid  G]  [Monoid  H]  where
  toFun  :  G  →  H
  map_one  :  toFun  1  =  1
  map_mul  :  ∀  g  g',  toFun  (g  *  g')  =  toFun  g  *  toFun  g' 
```

当然，我们不想在所有地方都输入`toFun`，所以我们使用`CoeFun`类型类注册了一个强制转换。它的第一个参数是我们想要强制转换为函数的类型。第二个参数描述了目标函数类型。在我们的例子中，对于每个`f : MonoidHom₁ G H`，它总是`G → H`。我们还使用`coe`属性标记`MonoidHom₁.toFun`，以确保它在战术状态中几乎不可见，只需一个`↑`前缀即可。

```py
instance  [Monoid  G]  [Monoid  H]  :  CoeFun  (MonoidHom₁  G  H)  (fun  _  ↦  G  →  H)  where
  coe  :=  MonoidHom₁.toFun

attribute  [coe]  MonoidHom₁.toFun 
```

让我们检查我们是否真的可以将一个捆绑的幺半群同态应用于一个元素。

```py
example  [Monoid  G]  [Monoid  H]  (f  :  MonoidHom₁  G  H)  :  f  1  =  1  :=  f.map_one 
```

我们可以用同样的方法处理其他类型的同态，直到我们达到环同态。

```py
@[ext]
structure  AddMonoidHom₁  (G  H  :  Type)  [AddMonoid  G]  [AddMonoid  H]  where
  toFun  :  G  →  H
  map_zero  :  toFun  0  =  0
  map_add  :  ∀  g  g',  toFun  (g  +  g')  =  toFun  g  +  toFun  g'

instance  [AddMonoid  G]  [AddMonoid  H]  :  CoeFun  (AddMonoidHom₁  G  H)  (fun  _  ↦  G  →  H)  where
  coe  :=  AddMonoidHom₁.toFun

attribute  [coe]  AddMonoidHom₁.toFun

@[ext]
structure  RingHom₁  (R  S  :  Type)  [Ring  R]  [Ring  S]  extends  MonoidHom₁  R  S,  AddMonoidHom₁  R  S 
```

这种方法有几个问题。一个较小的问题是，我们不知道在哪里放置 `coe` 属性，因为 `RingHom₁.toFun` 不存在，相关的函数是 `MonoidHom₁.toFun ∘ RingHom₁.toMonoidHom₁`，这不是一个可以标记属性的声明（但我们仍然可以定义一个 `CoeFun (RingHom₁ R S) (fun _ ↦ R → S)` 实例）。一个更重要的一个是，关于代数同态的引理不会直接适用于环同态。这留下了两种选择：每次想要应用代数同态引理时都玩弄 `RingHom₁.toMonoidHom₁`，或者为环同态重新陈述每个这样的引理。这两种选择都不吸引人，因此 Mathlib 在这里使用了一个新的层次结构技巧。想法是为至少是代数同态的对象定义一个类型类，用代数同态和环同态实例化这个类，并使用它来陈述每个引理。在下面的定义中，`F` 可以是 `MonoidHom₁ M N`，或者如果 `M` 和 `N` 有环结构，则是 `RingHom₁ M N`。

```py
class  MonoidHomClass₁  (F  :  Type)  (M  N  :  Type)  [Monoid  M]  [Monoid  N]  where
  toFun  :  F  →  M  →  N
  map_one  :  ∀  f  :  F,  toFun  f  1  =  1
  map_mul  :  ∀  f  g  g',  toFun  f  (g  *  g')  =  toFun  f  g  *  toFun  f  g' 
```

然而，上述实现存在一个问题。我们还没有注册一个强制转换为函数实例。现在让我们尝试一下。

```py
def  badInst  [Monoid  M]  [Monoid  N]  [MonoidHomClass₁  F  M  N]  :  CoeFun  F  (fun  _  ↦  M  →  N)  where
  coe  :=  MonoidHomClass₁.toFun 
```

将其作为一个实例会不好。当面对类似 `f x` 这样的情况，其中 `f` 的类型不是函数类型时，Lean 将尝试找到一个 `CoeFun` 实例来将 `f` 转换为函数。上述函数的类型是：`{M N F : Type} → [Monoid M] → [Monoid N] → [MonoidHomClass₁ F M N] → CoeFun F (fun x ↦ M → N)`，因此，当它尝试应用它时，Lean 并不清楚未知类型 `M`、`N` 和 `F` 应该以何种顺序进行推断。这是一种与之前所见略有不同但本质上相同的问题：不知道 `M` 时，Lean 将不得不在未知类型上搜索一个代数实例，因此会无望地尝试数据库中的 *每一个* 代数实例。如果你对这种实例的效果感到好奇，可以在上述声明上方输入 `set_option synthInstance.checkSynthOrder false in`，将 `def badInst` 替换为 `instance`，并在此文件中查找随机失败。

在这里，解决方案很简单，我们需要告诉 Lean 首先搜索 `F` 是什么，然后推断 `M` 和 `N`。这是通过使用 `outParam` 函数实现的。这个函数定义为恒等函数，但仍然被类型类机制识别并触发所需的行为。因此，我们可以重新定义我们的类，注意使用 `outParam` 函数：

```py
class  MonoidHomClass₂  (F  :  Type)  (M  N  :  outParam  Type)  [Monoid  M]  [Monoid  N]  where
  toFun  :  F  →  M  →  N
  map_one  :  ∀  f  :  F,  toFun  f  1  =  1
  map_mul  :  ∀  f  g  g',  toFun  f  (g  *  g')  =  toFun  f  g  *  toFun  f  g'

instance  [Monoid  M]  [Monoid  N]  [MonoidHomClass₂  F  M  N]  :  CoeFun  F  (fun  _  ↦  M  →  N)  where
  coe  :=  MonoidHomClass₂.toFun

attribute  [coe]  MonoidHomClass₂.toFun 
```

现在，我们可以继续我们的计划来实例化这个类。

```py
instance  (M  N  :  Type)  [Monoid  M]  [Monoid  N]  :  MonoidHomClass₂  (MonoidHom₁  M  N)  M  N  where
  toFun  :=  MonoidHom₁.toFun
  map_one  :=  fun  f  ↦  f.map_one
  map_mul  :=  fun  f  ↦  f.map_mul

instance  (R  S  :  Type)  [Ring  R]  [Ring  S]  :  MonoidHomClass₂  (RingHom₁  R  S)  R  S  where
  toFun  :=  fun  f  ↦  f.toMonoidHom₁.toFun
  map_one  :=  fun  f  ↦  f.toMonoidHom₁.map_one
  map_mul  :=  fun  f  ↦  f.toMonoidHom₁.map_mul 
```

如承诺的那样，我们关于 `f : F` 的每个引理，假设 `MonoidHomClass₁ F` 的一个实例，都将适用于代数同态和环同态。让我们看一个示例引理并检查它是否适用于这两种情况。

```py
lemma  map_inv_of_inv  [Monoid  M]  [Monoid  N]  [MonoidHomClass₂  F  M  N]  (f  :  F)  {m  m'  :  M}  (h  :  m*m'  =  1)  :
  f  m  *  f  m'  =  1  :=  by
  rw  [←  MonoidHomClass₂.map_mul,  h,  MonoidHomClass₂.map_one]

example  [Monoid  M]  [Monoid  N]  (f  :  MonoidHom₁  M  N)  {m  m'  :  M}  (h  :  m*m'  =  1)  :  f  m  *  f  m'  =  1  :=
map_inv_of_inv  f  h

example  [Ring  R]  [Ring  S]  (f  :  RingHom₁  R  S)  {r  r'  :  R}  (h  :  r*r'  =  1)  :  f  r  *  f  r'  =  1  :=
map_inv_of_inv  f  h 
```

初看起来，可能看起来我们回到了我们以前的老坏主意，将`MonoidHom₁`做成一个类。但我们并没有。一切都被提升了一个抽象层次。类型类解析过程不会寻找函数，它将寻找`MonoidHom₁`或`RingHom₁`。

我们的方法中存在的一个问题是围绕`toFun`字段及其对应的`CoeFun`实例和`coe`属性周围的重复代码。最好也记录下这种模式仅用于具有额外属性的功能，这意味着函数到函数的强制转换应该是单射的。因此，Mathlib 通过添加一个基于类`DFunLike`（其中“DFun”代表依赖函数）的额外抽象层。让我们在基础层之上重新定义我们的`MonoidHomClass`。

```py
class  MonoidHomClass₃  (F  :  Type)  (M  N  :  outParam  Type)  [Monoid  M]  [Monoid  N]  extends
  DFunLike  F  M  (fun  _  ↦  N)  where
  map_one  :  ∀  f  :  F,  f  1  =  1
  map_mul  :  ∀  (f  :  F)  g  g',  f  (g  *  g')  =  f  g  *  f  g'

instance  (M  N  :  Type)  [Monoid  M]  [Monoid  N]  :  MonoidHomClass₃  (MonoidHom₁  M  N)  M  N  where
  coe  :=  MonoidHom₁.toFun
  coe_injective'  _  _  :=  MonoidHom₁.ext
  map_one  :=  MonoidHom₁.map_one
  map_mul  :=  MonoidHom₁.map_mul 
```

当然，形态的层次结构并没有在这里停止。我们可以继续定义一个扩展`MonoidHomClass₃`的类`RingHomClass₃`，并在`RingHom`上实例化它，然后稍后在其上实例化`AlgebraHom`（代数是具有一些额外结构的环）。但我们已经涵盖了 Mathlib 中用于形态的主要形式化思想，你应该准备好理解在 Mathlib 中如何定义形态。

作为练习，你应该尝试定义你的有序类型之间打包的保持顺序的函数的类，然后定义保持顺序的幺半群形态。这仅用于训练目的。像连续函数一样，保持顺序的函数在 Mathlib 中主要是未打包的，它们由`Monotone`谓词定义。当然，你需要完成下面的类定义。

```py
@[ext]
structure  OrderPresHom  (α  β  :  Type)  [LE  α]  [LE  β]  where
  toFun  :  α  →  β
  le_of_le  :  ∀  a  a',  a  ≤  a'  →  toFun  a  ≤  toFun  a'

@[ext]
structure  OrderPresMonoidHom  (M  N  :  Type)  [Monoid  M]  [LE  M]  [Monoid  N]  [LE  N]  extends
MonoidHom₁  M  N,  OrderPresHom  M  N

class  OrderPresHomClass  (F  :  Type)  (α  β  :  outParam  Type)  [LE  α]  [LE  β]

instance  (α  β  :  Type)  [LE  α]  [LE  β]  :  OrderPresHomClass  (OrderPresHom  α  β)  α  β  where

instance  (α  β  :  Type)  [LE  α]  [Monoid  α]  [LE  β]  [Monoid  β]  :
  OrderPresHomClass  (OrderPresMonoidHom  α  β)  α  β  where

instance  (α  β  :  Type)  [LE  α]  [Monoid  α]  [LE  β]  [Monoid  β]  :
  MonoidHomClass₃  (OrderPresMonoidHom  α  β)  α  β
  :=  sorry 
```

## 8.3. 子对象

在定义了一些代数结构和其形态之后，下一步是考虑继承这种代数结构的集合，例如子群或子环。这很大程度上与我们的前一个主题重叠。实际上，`X`中的集合被实现为一个从`X`到`Prop`的函数，因此子对象是满足一定谓词的函数。因此，我们可以重用导致`DFunLike`类及其后代的许多想法。我们不会重用`DFunLike`本身，因为这会打破从`Set X`到`X → Prop`的抽象障碍。相反，有一个`SetLike`类。该类不是将注入包装到函数类型中，而是将注入包装到`Set`类型中，并定义相应的强制转换和`Membership`实例。

```py
@[ext]
structure  Submonoid₁  (M  :  Type)  [Monoid  M]  where
  /-- The carrier of a submonoid. -/
  carrier  :  Set  M
  /-- The product of two elements of a submonoid belongs to the submonoid. -/
  mul_mem  {a  b}  :  a  ∈  carrier  →  b  ∈  carrier  →  a  *  b  ∈  carrier
  /-- The unit element belongs to the submonoid. -/
  one_mem  :  1  ∈  carrier

/-- Submonoids in `M` can be seen as sets in `M`. -/
instance  [Monoid  M]  :  SetLike  (Submonoid₁  M)  M  where
  coe  :=  Submonoid₁.carrier
  coe_injective'  _  _  :=  Submonoid₁.ext 
```

配备了上述`SetLike`实例，我们目前已经可以自然地陈述一个子幺半群`N`包含`1`，而无需使用`N.carrier`。我们还可以在`M`中将`N`视为一个集合，并对其直接像进行映射。

```py
example  [Monoid  M]  (N  :  Submonoid₁  M)  :  1  ∈  N  :=  N.one_mem

example  [Monoid  M]  (N  :  Submonoid₁  M)  (α  :  Type)  (f  :  M  →  α)  :=  f  ''  N 
```

我们还有一个到`Type`的强制转换，它使用`Subtype`，因此，给定一个子幺半群`N`，我们可以写一个参数`(x : N)`，它可以被强制转换为属于`M`且属于`N`的元素。

```py
example  [Monoid  M]  (N  :  Submonoid₁  M)  (x  :  N)  :  (x  :  M)  ∈  N  :=  x.property 
```

使用这种到 `Type` 的强制转换，我们也可以处理给子群族赋予幺半群结构的问题。我们将使用上面提到的与 `N` 关联的类型强制转换，以及断言这种强制转换是单射的 `SetCoe.ext` 引理。这两个都是由 `SetLike` 实例提供的。

```py
instance  SubMonoid₁Monoid  [Monoid  M]  (N  :  Submonoid₁  M)  :  Monoid  N  where
  mul  :=  fun  x  y  ↦  ⟨x*y,  N.mul_mem  x.property  y.property⟩
  mul_assoc  :=  fun  x  y  z  ↦  SetCoe.ext  (mul_assoc  (x  :  M)  y  z)
  one  :=  ⟨1,  N.one_mem⟩
  one_mul  :=  fun  x  ↦  SetCoe.ext  (one_mul  (x  :  M))
  mul_one  :=  fun  x  ↦  SetCoe.ext  (mul_one  (x  :  M)) 
```

注意，在上面的实例中，我们除了使用到 `M` 的强制转换并调用 `property` 字段外，还可以使用解构绑定符，如下所示。

```py
example  [Monoid  M]  (N  :  Submonoid₁  M)  :  Monoid  N  where
  mul  :=  fun  ⟨x,  hx⟩  ⟨y,  hy⟩  ↦  ⟨x*y,  N.mul_mem  hx  hy⟩
  mul_assoc  :=  fun  ⟨x,  _⟩  ⟨y,  _⟩  ⟨z,  _⟩  ↦  SetCoe.ext  (mul_assoc  x  y  z)
  one  :=  ⟨1,  N.one_mem⟩
  one_mul  :=  fun  ⟨x,  _⟩  ↦  SetCoe.ext  (one_mul  x)
  mul_one  :=  fun  ⟨x,  _⟩  ↦  SetCoe.ext  (mul_one  x) 
```

为了将关于子群族的引理应用于子群或子环，我们需要一个类，就像对于态射一样。注意这个类接受一个 `SetLike` 实例作为参数，因此它不需要载体域，可以在其字段中使用成员符号。

```py
class  SubmonoidClass₁  (S  :  Type)  (M  :  Type)  [Monoid  M]  [SetLike  S  M]  :  Prop  where
  mul_mem  :  ∀  (s  :  S)  {a  b  :  M},  a  ∈  s  →  b  ∈  s  →  a  *  b  ∈  s
  one_mem  :  ∀  s  :  S,  1  ∈  s

instance  [Monoid  M]  :  SubmonoidClass₁  (Submonoid₁  M)  M  where
  mul_mem  :=  Submonoid₁.mul_mem
  one_mem  :=  Submonoid₁.one_mem 
```

作为练习，你应该定义一个 `Subgroup₁` 结构，给它赋予一个 `SetLike` 实例和一个 `SubmonoidClass₁` 实例，在关联于 `Subgroup₁` 的子类型上放置一个 `Group` 实例，并定义一个 `SubgroupClass₁` 类。

关于 Mathlib 中给定代数对象的子对象，总是形成一个完备格，并且这个结构被大量使用。例如，你可能想要查找一个说子群族的交集是一个子群族的引理。但这不会是一个引理，这将是一个下确界构造。让我们看看两个子群族的例子。

```py
instance  [Monoid  M]  :  Min  (Submonoid₁  M)  :=
  ⟨fun  S₁  S₂  ↦
  {  carrier  :=  S₁  ∩  S₂
  one_mem  :=  ⟨S₁.one_mem,  S₂.one_mem⟩
  mul_mem  :=  fun  ⟨hx,  hx'⟩  ⟨hy,  hy'⟩  ↦  ⟨S₁.mul_mem  hx  hy,  S₂.mul_mem  hx'  hy'⟩  }⟩ 
```

这允许我们得到两个子群族的交集作为一个子群族。

```py
example  [Monoid  M]  (N  P  :  Submonoid₁  M)  :  Submonoid₁  M  :=  N  ⊓  P 
```

你可能觉得我们不得不在上面的例子中使用符号 `⊓` 而不是交集符号 `∩` 是一种遗憾。但想想上确界。两个子群族的并集不是一个子群族。然而，子群族仍然形成一个格（甚至是一个完备格）。实际上，`N ⊔ P` 是由 `N` 和 `P` 的并集生成的子群族，当然用 `N ∪ P` 来表示它会非常令人困惑。所以，你可以看到使用 `N ⊓ P` 的方式要一致得多。它在各种代数结构中的一致性也更强。一开始看到两个向量子空间 `E` 和 `F` 的和用 `E ⊔ F` 表示而不是 `E + F` 可能会显得有些奇怪。但你会习惯的。很快，你就会认为 `E + F` 的表示是一个干扰，强调 `E ⊔ F` 的元素可以写成 `E` 的一个元素和 `F` 的一个元素的和，而不是强调 `E ⊔ F` 是包含 `E` 和 `F` 的最小向量子空间这一基本事实。

本章的最后一个主题是商。同样，我们想要解释在 Mathlib 中如何构建方便的记号并避免代码重复。这里的主要工具是 `HasQuotient` 类，它允许使用像 `M ⧸ N` 这样的记号。注意，商符号 `⧸` 是一个特殊的 Unicode 字符，不是一个常规的 ASCII 除法符号。

例如，我们将通过子幺半群构建一个交换幺半群的商，证明留给你们。在上一个例子中，你可以使用 `Setoid.refl`，但它不会自动获取相关的 `Setoid` 结构。你可以通过使用 `@` 语法提供所有参数来解决这个问题，就像在 `@Setoid.refl M N.Setoid` 中一样。

```py
def  Submonoid.Setoid  [CommMonoid  M]  (N  :  Submonoid  M)  :  Setoid  M  where
  r  :=  fun  x  y  ↦  ∃  w  ∈  N,  ∃  z  ∈  N,  x*w  =  y*z
  iseqv  :=  {
  refl  :=  fun  x  ↦  ⟨1,  N.one_mem,  1,  N.one_mem,  rfl⟩
  symm  :=  fun  ⟨w,  hw,  z,  hz,  h⟩  ↦  ⟨z,  hz,  w,  hw,  h.symm⟩
  trans  :=  by
  sorry
  }

instance  [CommMonoid  M]  :  HasQuotient  M  (Submonoid  M)  where
  quotient'  :=  fun  N  ↦  Quotient  N.Setoid

def  QuotientMonoid.mk  [CommMonoid  M]  (N  :  Submonoid  M)  :  M  →  M  ⧸  N  :=  Quotient.mk  N.Setoid

instance  [CommMonoid  M]  (N  :  Submonoid  M)  :  Monoid  (M  ⧸  N)  where
  mul  :=  Quotient.map₂  (·  *  ·)  (by
  sorry
  )
  mul_assoc  :=  by
  sorry
  one  :=  QuotientMonoid.mk  N  1
  one_mul  :=  by
  sorry
  mul_one  :=  by
  sorry 
```*
