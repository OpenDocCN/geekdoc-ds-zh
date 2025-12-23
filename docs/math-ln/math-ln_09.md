# 9. 群与环

> 原文：[`leanprover-community.github.io/mathematics_in_lean/C09_Groups_and_Rings.html`](https://leanprover-community.github.io/mathematics_in_lean/C09_Groups_and_Rings.html)

*Mathematics in Lean* **   9. 群与环

+   查看页面源代码

* * *

我们在第 2.2 节（[C02_Basics.html#proving-identities-in-algebraic-structures](https://leanprover-community.github.io/mathematics_in_lean/C02_Basics.html#proving-identities-in-algebraic-structures)）中看到了如何在群和环中推理操作。后来，在第 7.2 节（[C07_Structures.html#section-algebraic-structures](https://leanprover-community.github.io/mathematics_in_lean/C07_Structures.html#section-algebraic-structures)）中，我们看到了如何定义抽象代数结构，例如群结构，以及具体的实例，如高斯整数上的环结构。[第八章](https://leanprover-community.github.io/mathematics_in_lean/C08_Hierarchies.html#hierarchies)解释了在 Mathlib 中如何处理抽象结构的层次。

在本章中，我们将更详细地研究群与环。由于 Mathlib 库不断增长，我们无法涵盖这些主题在 Mathlib 中处理的各个方面。但我们将提供进入库的入口点，并展示基本概念是如何被使用的。与第八章（[C08_Hierarchies.html#hierarchies](https://leanprover-community.github.io/mathematics_in_lean/C08_Hierarchies.html#hierarchies)）的讨论有一些重叠，但在这里我们将专注于如何使用 Mathlib，而不是处理这些主题的设计决策。因此，理解一些示例可能需要回顾第八章的内容。

## 9.1. 群与群同态

### 9.1.1. 群及其同态

抽象代数课程通常从群开始，然后过渡到环、域和向量空间。在讨论环上的乘法时，这涉及到一些扭曲，因为乘法操作并不来自群结构，但许多证明可以直接从群理论转移到这个新环境。在用笔和纸做数学时，最常见的解决方案是将这些证明作为练习。一种不太高效但更安全、更符合形式化要求的做法是使用幺半群。在类型 M 上的*幺半群*结构是一个内部结合律，它是结合的并有一个单位元。幺半群主要用于适应群和环的乘性结构。但也有一些自然例子；例如，自然数集加上加法运算形成一个幺半群。

从实际应用的角度来看，在使用 Mathlib 时，你可以基本上忽略群。但是，当你通过浏览 Mathlib 文件寻找引理时，你需要知道它们的存在。否则，你可能会在群论文件中寻找一个陈述，而实际上它是在群中找到的，因为群不需要元素可逆。

类型 `M` 上的单群结构类型写成 `Monoid M`。函数 `Monoid` 是一个类型类，所以它几乎总是作为一个隐式参数实例出现（换句话说，在方括号中）。默认情况下，`Monoid` 使用乘法表示法进行操作；对于加法表示法，请使用 `AddMonoid`。这些结构的交换版本在 `Monoid` 前添加 `Comm` 前缀。

```py
example  {M  :  Type*}  [Monoid  M]  (x  :  M)  :  x  *  1  =  x  :=  mul_one  x

example  {M  :  Type*}  [AddCommMonoid  M]  (x  y  :  M)  :  x  +  y  =  y  +  x  :=  add_comm  x  y 
```

注意，尽管 `AddMonoid` 在库中可以找到，但使用非交换操作的加法表示法通常很令人困惑。

两个单群 `M` 和 `N` 之间的同态类型称为 `MonoidHom M N`，并写成 `M →* N`。当我们将其应用于 `M` 的元素时，Lean 会自动将其视为从 `M` 到 `N` 的函数。加法版本称为 `AddMonoidHom`，并写成 `M →+ N`。

```py
example  {M  N  :  Type*}  [Monoid  M]  [Monoid  N]  (x  y  :  M)  (f  :  M  →*  N)  :  f  (x  *  y)  =  f  x  *  f  y  :=
  f.map_mul  x  y

example  {M  N  :  Type*}  [AddMonoid  M]  [AddMonoid  N]  (f  :  M  →+  N)  :  f  0  =  0  :=
  f.map_zero 
```

这些同态是打包映射，即它们将映射及其一些属性打包在一起。记住，第 8.2 节 解释了打包映射；这里我们只是简单地指出一个不幸的后果，即我们不能使用普通函数组合来组合映射。相反，我们需要使用 `MonoidHom.comp` 和 `AddMonoidHom.comp`。

```py
example  {M  N  P  :  Type*}  [AddMonoid  M]  [AddMonoid  N]  [AddMonoid  P]
  (f  :  M  →+  N)  (g  :  N  →+  P)  :  M  →+  P  :=  g.comp  f 
```

### 9.1.2. 群及其同态

我们将有很多关于群的话要说，群是具有每个元素都有逆元的额外属性的单群。

```py
example  {G  :  Type*}  [Group  G]  (x  :  G)  :  x  *  x⁻¹  =  1  :=  mul_inv_cancel  x 
```

与我们之前看到的 `ring` 策略类似，有一个 `group` 策略可以证明任何在任何群中成立的恒等式。（等价地，它证明了在自由群中成立的恒等式。）

```py
example  {G  :  Type*}  [Group  G]  (x  y  z  :  G)  :  x  *  (y  *  z)  *  (x  *  z)⁻¹  *  (x  *  y  *  x⁻¹)⁻¹  =  1  :=  by
  group 
```

此外，还有一个用于交换加法群恒等式的策略，称为 `abel`。

```py
example  {G  :  Type*}  [AddCommGroup  G]  (x  y  z  :  G)  :  z  +  x  +  (y  -  z  -  x)  =  y  :=  by
  abel 
```

有趣的是，群同态不过是在群之间的单群同态。因此，我们可以复制并粘贴我们之前的一个例子，将 `Monoid` 替换为 `Group`。

```py
example  {G  H  :  Type*}  [Group  G]  [Group  H]  (x  y  :  G)  (f  :  G  →*  H)  :  f  (x  *  y)  =  f  x  *  f  y  :=
  f.map_mul  x  y 
```

当然，我们确实得到了一些新的属性，例如这个：

```py
example  {G  H  :  Type*}  [Group  G]  [Group  H]  (x  :  G)  (f  :  G  →*  H)  :  f  (x⁻¹)  =  (f  x)⁻¹  :=
  f.map_inv  x 
```

你可能担心构造群同态会让我们做不必要的额外工作，因为单群同态的定义强制要求中性元素被发送到中性元素，而在群同态的情况下这是自动的。在实践中，额外的努力并不困难，但为了避免它，有一个函数可以从群之间的兼容于组合律的函数构建群同态。

```py
example  {G  H  :  Type*}  [Group  G]  [Group  H]  (f  :  G  →  H)  (h  :  ∀  x  y,  f  (x  *  y)  =  f  x  *  f  y)  :
  G  →*  H  :=
  MonoidHom.mk'  f  h 
```

此外，还有一种类型 `MulEquiv` 的群（或单群）同构，用 `≃*` 表示（在加法表示法中用 `≃+` 表示）。`f : G ≃* H` 的逆是 `MulEquiv.symm f : H ≃* G`，`f` 和 `g` 的组合是 `MulEquiv.trans f g`，而 `G` 的恒等同构是 `M̀ulEquiv.refl G`。使用匿名投影符号，前两个可以分别写成 `f.symm` 和 `f.trans g`。当需要时，此类元素会自动转换为同态和函数。

```py
example  {G  H  :  Type*}  [Group  G]  [Group  H]  (f  :  G  ≃*  H)  :
  f.trans  f.symm  =  MulEquiv.refl  G  :=
  f.self_trans_symm 
```

可以使用 `MulEquiv.ofBijective` 从双射同态构建同构。这样做会使逆函数不可计算。

```py
noncomputable  example  {G  H  :  Type*}  [Group  G]  [Group  H]
  (f  :  G  →*  H)  (h  :  Function.Bijective  f)  :
  G  ≃*  H  :=
  MulEquiv.ofBijective  f  h 
```

### 9.1.3\. 子群

正如群同态是捆绑在一起的，`G` 的一个子群也是一个捆绑结构，由 `G` 中的一个集合及其相关的闭包性质组成。

```py
example  {G  :  Type*}  [Group  G]  (H  :  Subgroup  G)  {x  y  :  G}  (hx  :  x  ∈  H)  (hy  :  y  ∈  H)  :
  x  *  y  ∈  H  :=
  H.mul_mem  hx  hy

example  {G  :  Type*}  [Group  G]  (H  :  Subgroup  G)  {x  :  G}  (hx  :  x  ∈  H)  :
  x⁻¹  ∈  H  :=
  H.inv_mem  hx 
```

在上面的例子中，重要的是要理解 `Subgroup G` 是 `G` 的子群的类型，而不是一个谓词 `IsSubgroup H`，其中 `H` 是 `Set G` 的一个元素。`Subgroup G` 被赋予了到 `Set G` 的强制转换和一个关于 `G` 的成员谓词。参见 第 8.3 节 了解如何以及为什么这样做。

当然，如果两个子群具有相同的元素，则它们是相同的。这个事实被注册用于与 `ext` 策略一起使用，该策略可以用来证明两个子群相等，就像它被用来证明两个集合相等一样。

例如，为了陈述和证明 `ℤ` 是 `ℚ` 的加法子群，我们真正想要的是构造一个类型为 `AddSubgroup ℚ` 的项，其投影到 `Set ℚ` 是 `ℤ`，或者更精确地说，是 `ℤ` 在 `ℚ` 中的像。

```py
example  :  AddSubgroup  ℚ  where
  carrier  :=  Set.range  ((↑)  :  ℤ  →  ℚ)
  add_mem'  :=  by
  rintro  _  _  ⟨n,  rfl⟩  ⟨m,  rfl⟩
  use  n  +  m
  simp
  zero_mem'  :=  by
  use  0
  simp
  neg_mem'  :=  by
  rintro  _  ⟨n,  rfl⟩
  use  -n
  simp 
```

使用类型类，Mathlib 知道群的一个子群继承了群结构。

```py
example  {G  :  Type*}  [Group  G]  (H  :  Subgroup  G)  :  Group  H  :=  inferInstance 
```

这个例子很微妙。对象 `H` 不是一个类型，但 Lean 会自动将其强制转换为类型，通过将其解释为 `G` 的一个子类型。因此，上面的例子可以更明确地重述为：

```py
example  {G  :  Type*}  [Group  G]  (H  :  Subgroup  G)  :  Group  {x  :  G  //  x  ∈  H}  :=  inferInstance 
```

拥有类型 `Subgroup G` 而不是谓词 `IsSubgroup : Set G → Prop` 的重要好处是，可以很容易地为 `Subgroup G` 赋予额外的结构。重要的是，它具有关于包含的完备格结构。例如，我们不是有一个断言两个 `G` 的子群的交集仍然是子群的引理，而是使用了格运算 `⊓` 来构造交集。然后我们可以将关于格的任意引理应用于构造。

让我们检查两个子群的交集的底层集合确实，按照定义，是它们的交集。

```py
example  {G  :  Type*}  [Group  G]  (H  H'  :  Subgroup  G)  :
  ((H  ⊓  H'  :  Subgroup  G)  :  Set  G)  =  (H  :  Set  G)  ∩  (H'  :  Set  G)  :=  rfl 
```

对于底层集合的交集使用不同的符号可能看起来很奇怪，但这种对应关系并不适用于上确界运算和集合的并集，因为子群的并集在一般情况下不是一个子群。相反，需要使用由并集生成的子群，这是通过 `Subgroup.closure` 来实现的。

```py
example  {G  :  Type*}  [Group  G]  (H  H'  :  Subgroup  G)  :
  ((H  ⊔  H'  :  Subgroup  G)  :  Set  G)  =  Subgroup.closure  ((H  :  Set  G)  ∪  (H'  :  Set  G))  :=  by
  rw  [Subgroup.sup_eq_closure] 
```

另一个微妙之处在于 `G` 本身没有类型 `Subgroup G`，因此我们需要一种方法来谈论 `G` 作为 `G` 的子群。这也由格结构提供：全子群是这个格的顶元素。

```py
example  {G  :  Type*}  [Group  G]  (x  :  G)  :  x  ∈  (⊤  :  Subgroup  G)  :=  trivial 
```

类似地，这个格的底元素是只有一个元素的子群，即中性元素。

```py
example  {G  :  Type*}  [Group  G]  (x  :  G)  :  x  ∈  (⊥  :  Subgroup  G)  ↔  x  =  1  :=  Subgroup.mem_bot 
```

作为操作群和子群的一个练习，你可以通过环境群中的一个元素来定义子群的共轭。

```py
def  conjugate  {G  :  Type*}  [Group  G]  (x  :  G)  (H  :  Subgroup  G)  :  Subgroup  G  where
  carrier  :=  {a  :  G  |  ∃  h,  h  ∈  H  ∧  a  =  x  *  h  *  x⁻¹}
  one_mem'  :=  by
  dsimp
  sorry
  inv_mem'  :=  by
  dsimp
  sorry
  mul_mem'  :=  by
  dsimp
  sorry 
```

将前两个主题结合起来，可以使用群同态来推进和拉回子群。在 Mathlib 中的命名约定是将这些操作称为`map`和`comap`。这些不是常见的数学术语，但它们的优势是比“推进”和“直接像”更短。

```py
example  {G  H  :  Type*}  [Group  G]  [Group  H]  (G'  :  Subgroup  G)  (f  :  G  →*  H)  :  Subgroup  H  :=
  Subgroup.map  f  G'

example  {G  H  :  Type*}  [Group  G]  [Group  H]  (H'  :  Subgroup  H)  (f  :  G  →*  H)  :  Subgroup  G  :=
  Subgroup.comap  f  H'

#check  Subgroup.mem_map
#check  Subgroup.mem_comap 
```

特别地，在映射`f`下的底子群的逆像是称为`f`的*核*的子群，而`f`的值域也是一个子群。

```py
example  {G  H  :  Type*}  [Group  G]  [Group  H]  (f  :  G  →*  H)  (g  :  G)  :
  g  ∈  MonoidHom.ker  f  ↔  f  g  =  1  :=
  f.mem_ker

example  {G  H  :  Type*}  [Group  G]  [Group  H]  (f  :  G  →*  H)  (h  :  H)  :
  h  ∈  MonoidHom.range  f  ↔  ∃  g  :  G,  f  g  =  h  :=
  f.mem_range 
```

作为操作群同态和子群的训练，让我们证明一些基本性质。它们已经在 Mathlib 中得到了证明，所以如果你想从这些练习中受益，不要急于使用`exact?`。

```py
section  exercises
variable  {G  H  :  Type*}  [Group  G]  [Group  H]

open  Subgroup

example  (φ  :  G  →*  H)  (S  T  :  Subgroup  H)  (hST  :  S  ≤  T)  :  comap  φ  S  ≤  comap  φ  T  :=  by
  sorry

example  (φ  :  G  →*  H)  (S  T  :  Subgroup  G)  (hST  :  S  ≤  T)  :  map  φ  S  ≤  map  φ  T  :=  by
  sorry

variable  {K  :  Type*}  [Group  K]

-- Remember you can use the `ext` tactic to prove an equality of subgroups.
example  (φ  :  G  →*  H)  (ψ  :  H  →*  K)  (U  :  Subgroup  K)  :
  comap  (ψ.comp  φ)  U  =  comap  φ  (comap  ψ  U)  :=  by
  sorry

-- Pushing a subgroup along one homomorphism and then another is equal to
-- pushing it forward along the composite of the homomorphisms.
example  (φ  :  G  →*  H)  (ψ  :  H  →*  K)  (S  :  Subgroup  G)  :
  map  (ψ.comp  φ)  S  =  map  ψ  (S.map  φ)  :=  by
  sorry

end  exercises 
```

让我们用两个非常经典的结果来结束对 Mathlib 中子群的介绍。拉格朗日定理表明有限群子群的基数是群基数的因子。西罗第一定理是拉格朗日定理的一个著名的部分逆定理。

虽然 Mathlib 的这个角落部分是为了允许计算而设置的，但我们可以使用以下`open scoped`命令告诉 Lean 使用非构造性逻辑。

```py
open  scoped  Classical

example  {G  :  Type*}  [Group  G]  (G'  :  Subgroup  G)  :  Nat.card  G'  ∣  Nat.card  G  :=
  ⟨G'.index,  mul_comm  G'.index  _  ▸  G'.index_mul_card.symm⟩

open  Subgroup

example  {G  :  Type*}  [Group  G]  [Finite  G]  (p  :  ℕ)  {n  :  ℕ}  [Fact  p.Prime]
  (hdvd  :  p  ^  n  ∣  Nat.card  G)  :  ∃  K  :  Subgroup  G,  Nat.card  K  =  p  ^  n  :=
  Sylow.exists_subgroup_card_pow_prime  p  hdvd 
```

接下来的两个练习推导出拉格朗日引理的一个推论。（这也在 Mathlib 中已经有了，所以不要急于使用`exact?`。）

```py
lemma  eq_bot_iff_card  {G  :  Type*}  [Group  G]  {H  :  Subgroup  G}  :
  H  =  ⊥  ↔  Nat.card  H  =  1  :=  by
  suffices  (∀  x  ∈  H,  x  =  1)  ↔  ∃  x  ∈  H,  ∀  a  ∈  H,  a  =  x  by
  simpa  [eq_bot_iff_forall,  Nat.card_eq_one_iff_exists]
  sorry

#check  card_dvd_of_le

lemma  inf_bot_of_coprime  {G  :  Type*}  [Group  G]  (H  K  :  Subgroup  G)
  (h  :  (Nat.card  H).Coprime  (Nat.card  K))  :  H  ⊓  K  =  ⊥  :=  by
  sorry 
```

### 9.1.4\. 具体群

在 Mathlib 中也可以操作具体群，尽管这通常比处理抽象理论更复杂。例如，给定任何类型`X`，`X`的排列群是`Equiv.Perm X`。特别是对称群$\mathfrak{S}_n$是`Equiv.Perm (Fin n)`。可以对这个群陈述抽象结果，例如说如果`X`是有限的，那么`Equiv.Perm X`由循环生成。

```py
open  Equiv

example  {X  :  Type*}  [Finite  X]  :  Subgroup.closure  {σ  :  Perm  X  |  Perm.IsCycle  σ}  =  ⊤  :=
  Perm.closure_isCycle 
```

可以完全具体地计算循环的乘积。下面我们使用`#simp`命令，它在一个给定的表达式中调用`simp`策略。符号`c[]`用于定义循环排列。在例子中，结果是`ℕ`的排列。可以通过在第一个数字上使用类型注解，例如`(1 : Fin 5)`，使其成为`Perm (Fin 5)`中的计算。

```py
#simp  [mul_assoc]  c[1,  2,  3]  *  c[2,  3,  4] 
```

与此同时，使用自由群和群表示也是处理具体群的一种方法。类型`α`上的自由群是`FreeGroup α`，包含映射是`FreeGroup.of : α → FreeGroup α`。例如，让我们定义一个有三个元素`a`、`b`和`c`的类型`S`，以及相应自由群中的元素`ab⁻¹`。

```py
section  FreeGroup

inductive  S  |  a  |  b  |  c

open  S

def  myElement  :  FreeGroup  S  :=  (.of  a)  *  (.of  b)⁻¹ 
```

注意，我们给出了定义的预期类型，这样 Lean 就知道`.of`意味着`FreeGroup.of`。

自由群的通用性质体现在`FreeGroup.lift`等价性中。例如，让我们定义从`FreeGroup S`到`Perm (Fin 5)`的群同态，将`a`映射到`c[1, 2, 3]`，将`b`映射到`c[2, 3, 1]`，将`c`映射到`c[2, 3]`，

```py
def  myMorphism  :  FreeGroup  S  →*  Perm  (Fin  5)  :=
  FreeGroup.lift  fun  |  .a  =>  c[1,  2,  3]
  |  .b  =>  c[2,  3,  1]
  |  .c  =>  c[2,  3] 
```

作为最后一个具体的例子，让我们看看如何定义一个由单个元素生成且其立方为 1 的群（因此该群将与 $\mathbb{Z}/3$ 同构），并构建从该群到 `Perm (Fin 5)` 的态射。

作为只有一个元素的类型，我们将使用 `Unit`，其唯一元素由 `()` 表示。`PresentedGroup` 函数接受一个关系集，即某个自由群的一组元素，并返回一个群，该群是自由群除以由关系生成的正规子群。（我们将在第 9.1.6 节中看到如何处理更一般的商。）由于我们以某种方式将此隐藏在定义之后，我们使用 `deriving Group` 来强制在 `myGroup` 上创建一个群实例。

```py
def  myGroup  :=  PresentedGroup  {.of  ()  ^  3}  deriving  Group 
```

呈现群的全称性质确保可以从将关系映射到目标群中性元的函数构建出从这个群出去的态射。因此，我们需要这样的函数和一个证明该条件成立。然后我们可以将这个证明输入到 `PresentedGroup.toGroup` 中，以获得所需的群态射。

```py
def  myMap  :  Unit  →  Perm  (Fin  5)
|  ()  =>  c[1,  2,  3]

lemma  compat_myMap  :
  ∀  r  ∈  ({.of  ()  ^  3}  :  Set  (FreeGroup  Unit)),  FreeGroup.lift  myMap  r  =  1  :=  by
  rintro  _  rfl
  simp
  decide

def  myNewMorphism  :  myGroup  →*  Perm  (Fin  5)  :=  PresentedGroup.toGroup  compat_myMap

end  FreeGroup 
```

### 9.1.5\. 群作用

群论与数学其他部分的一个重要交互方式是通过群作用的使用。一个群 `G` 在某种类型 `X` 上的作用不过是从 `G` 到 `Equiv.Perm X` 的一个态射。所以在某种意义上，群作用已经被之前的讨论所涵盖。但我们不希望携带这个态射；相反，我们希望尽可能由 Lean 自动推断出来。因此，我们有一个类型类来表示这一点，即 `MulAction G X`。这种设置的缺点是，在同一个类型上有多个同一群的作用需要一些扭曲，例如定义类型同义词，每个同义词都携带不同的类型类实例。

这特别允许我们使用 `g • x` 来表示群元素 `g` 对点 `x` 的作用。

```py
noncomputable  section  GroupActions

example  {G  X  :  Type*}  [Group  G]  [MulAction  G  X]  (g  g':  G)  (x  :  X)  :
  g  •  (g'  •  x)  =  (g  *  g')  •  x  :=
  (mul_smul  g  g'  x).symm 
```

还有一个加法群的版本，称为 `AddAction`，其中作用由 `+ᵥ` 表示。这被用于例如仿射空间的定义中。

```py
example  {G  X  :  Type*}  [AddGroup  G]  [AddAction  G  X]  (g  g'  :  G)  (x  :  X)  :
  g  +ᵥ  (g'  +ᵥ  x)  =  (g  +  g')  +ᵥ  x  :=
  (add_vadd  g  g'  x).symm 
```

基础群态射被称为 `MulAction.toPermHom`。

```py
open  MulAction

example  {G  X  :  Type*}  [Group  G]  [MulAction  G  X]  :  G  →*  Equiv.Perm  X  :=
  toPermHom  G  X 
```

作为说明，让我们看看如何定义任何群 `G` 到置换群 `Perm G` 的凯莱同构嵌入。

```py
def  CayleyIsoMorphism  (G  :  Type*)  [Group  G]  :  G  ≃*  (toPermHom  G  G).range  :=
  Equiv.Perm.subgroupOfMulAction  G  G 
```

注意，在上述定义之前，并没有要求必须有群而不是幺半群（或任何带有乘法运算的类型）。

当我们想要将 `X` 划分为轨道时，群条件真正进入画面。`X` 上的对应等价关系称为 `MulAction.orbitRel`。它没有被声明为一个全局实例。

```py
example  {G  X  :  Type*}  [Group  G]  [MulAction  G  X]  :  Setoid  X  :=  orbitRel  G  X 
```

使用这个结果，我们可以陈述`X`在`G`的作用下被划分为轨道。更精确地说，我们得到一个从`X`到依赖积`(ω : orbitRel.Quotient G X) × (orbit G (Quotient.out' ω))`的双射，其中`Quotient.out' ω`简单地选择一个投影到`ω`的元素。回想一下，这个依赖积的元素是`⟨ω, x⟩`对，其中`x`的类型`orbit G (Quotient.out' ω)`依赖于`ω`。

```py
example  {G  X  :  Type*}  [Group  G]  [MulAction  G  X]  :
  X  ≃  (ω  :  orbitRel.Quotient  G  X)  ×  (orbit  G  (Quotient.out  ω))  :=
  MulAction.selfEquivSigmaOrbits  G  X 
```

特别地，当 X 是有限集时，这可以与`Fintype.card_congr`和`Fintype.card_sigma`结合，推导出`X`的基数是轨道基数的和。此外，轨道与通过左平移的稳定子群作用下的`G`的商一一对应。这种通过左平移的子群作用被用来定义通过子群进行商的群，用符号/表示，因此我们可以使用以下简洁的陈述。

```py
example  {G  X  :  Type*}  [Group  G]  [MulAction  G  X]  (x  :  X)  :
  orbit  G  x  ≃  G  ⧸  stabilizer  G  x  :=
  MulAction.orbitEquivQuotientStabilizer  G  x 
```

将上述两个结果结合起来，一个重要的特殊情况是当`X`是一个带有子群`H`通过平移作用的群`G`。在这种情况下，所有稳定子群都是平凡的，因此每个轨道都与`H`一一对应，我们得到：

```py
example  {G  :  Type*}  [Group  G]  (H  :  Subgroup  G)  :  G  ≃  (G  ⧸  H)  ×  H  :=
  groupEquivQuotientProdSubgroup 
```

这是上面我们看到的拉格朗日定理的概念变体。注意这个版本没有有限性的假设。

作为本节的一个练习，让我们通过共轭作用来构建一个群对其子群的作用，使用我们在前一个练习中定义的`共轭`。

```py
variable  {G  :  Type*}  [Group  G]

lemma  conjugate_one  (H  :  Subgroup  G)  :  conjugate  1  H  =  H  :=  by
  sorry

instance  :  MulAction  G  (Subgroup  G)  where
  smul  :=  conjugate
  one_smul  :=  by
  sorry
  mul_smul  :=  by
  sorry

end  GroupActions 
```

### 9.1.6. 商群

在上面关于子群在群上作用的讨论中，我们看到了商`G ⧸ H`的出现。在一般情况下，这只是一个类型。它可以赋予一个群结构，使得商映射是一个群同态当且仅当`H`是一个正规子群（并且这种群结构是唯一的）。

正规性假设是一个类型类`Subgroup.Normal`，这样类型类推理就可以用它来推导商上的群结构。

```py
noncomputable  section  QuotientGroup

example  {G  :  Type*}  [Group  G]  (H  :  Subgroup  G)  [H.Normal]  :  Group  (G  ⧸  H)  :=  inferInstance

example  {G  :  Type*}  [Group  G]  (H  :  Subgroup  G)  [H.Normal]  :  G  →*  G  ⧸  H  :=
  QuotientGroup.mk'  H 
```

通过`QuotientGroup.lift`可以访问商群的全称性质：一旦一个群同态`φ`的核包含`N`，它就会下降到`G ⧸ N`。

```py
example  {G  :  Type*}  [Group  G]  (N  :  Subgroup  G)  [N.Normal]  {M  :  Type*}
  [Group  M]  (φ  :  G  →*  M)  (h  :  N  ≤  MonoidHom.ker  φ)  :  G  ⧸  N  →*  M  :=
  QuotientGroup.lift  N  φ  h 
```

在上面的代码片段中，目标群被称为`M`是一个线索，表明在`M`上有一个幺半群结构就足够了。

当`N = ker φ`时，这是一个重要的特殊情况。在这种情况下，下降的同态是单射的，我们得到一个到其像上的群同构。这个结果通常被称为第一同构定理。

```py
example  {G  :  Type*}  [Group  G]  {M  :  Type*}  [Group  M]  (φ  :  G  →*  M)  :
  G  ⧸  MonoidHom.ker  φ  →*  MonoidHom.range  φ  :=
  QuotientGroup.quotientKerEquivRange  φ 
```

将全称性质应用于一个同态`φ : G →* G'`与商群投影`Quotient.mk' N'`的组合，我们也可以寻求从`G ⧸ N`到`G' ⧸ N'`的映射。对`φ`的要求通常表述为“`φ`应该将`N`放入`N'`中。”但这等同于要求`φ`将`N'`拉回到`N`上，而后一种条件更容易处理，因为拉回的定义不涉及存在量词。

```py
example  {G  G':  Type*}  [Group  G]  [Group  G']
  {N  :  Subgroup  G}  [N.Normal]  {N'  :  Subgroup  G'}  [N'.Normal]
  {φ  :  G  →*  G'}  (h  :  N  ≤  Subgroup.comap  φ  N')  :  G  ⧸  N  →*  G'  ⧸  N':=
  QuotientGroup.map  N  N'  φ  h 
```

需要记住的一个微妙之处是，类型`G ⧸ N`实际上依赖于`N`（直到定义等价），因此证明两个正规子群`N`和`M`相等不足以使相应的商相等。然而，普遍性质在这种情况下确实给出了一个同构。

```py
example  {G  :  Type*}  [Group  G]  {M  N  :  Subgroup  G}  [M.Normal]
  [N.Normal]  (h  :  M  =  N)  :  G  ⧸  M  ≃*  G  ⧸  N  :=  QuotientGroup.quotientMulEquivOfEq  h 
```

作为本节最后的练习系列，我们将证明如果`H`和`K`是有限群`G`的互斥正规子群，且它们的基数乘积等于`G`的基数，那么`G`同构于`H × K`。请记住，在这个上下文中，“互斥”意味着`H ⊓ K = ⊥`。

我们从稍微玩一下拉格朗日引理开始，而不假设子群是正规的或互斥的。

```py
section
variable  {G  :  Type*}  [Group  G]  {H  K  :  Subgroup  G}

open  MonoidHom

#check  Nat.card_pos  -- The nonempty argument will be automatically inferred for subgroups
#check  Subgroup.index_eq_card
#check  Subgroup.index_mul_card
#check  Nat.eq_of_mul_eq_mul_right

lemma  aux_card_eq  [Finite  G]  (h'  :  Nat.card  G  =  Nat.card  H  *  Nat.card  K)  :
  Nat.card  (G  ⧸  H)  =  Nat.card  K  :=  by
  sorry 
```

从现在起，我们假设我们的子群是正规的且互斥的，并假设基数条件。现在我们构建所需同构的第一个构建块。

```py
variable  [H.Normal]  [K.Normal]  [Fintype  G]  (h  :  Disjoint  H  K)
  (h'  :  Nat.card  G  =  Nat.card  H  *  Nat.card  K)

#check  Nat.bijective_iff_injective_and_card
#check  ker_eq_bot_iff
#check  restrict
#check  ker_restrict

def  iso₁  :  K  ≃*  G  ⧸  H  :=  by
  sorry 
```

现在我们可以定义我们的第二个构建块。我们需要`MonoidHom.prod`，它从`G₀`到`G₁ × G₂`构建一个形态，其中形态是从`G₀`到`G₁`和`G₂`的。

```py
def  iso₂  :  G  ≃*  (G  ⧸  K)  ×  (G  ⧸  H)  :=  by
  sorry 
```

我们准备将所有部件组合在一起。

```py
#check  MulEquiv.prodCongr

def  finalIso  :  G  ≃*  H  ×  K  :=
  sorry 
```  ## 9.2. 环

### 9.2.1. 环，它们的单位，形态和子环

类型`R`上的环结构类型是`Ring R`。假设乘法是交换的变体是`CommRing R`。我们已经看到`ring`策略将证明任何从交换环公理得出的等式。

```py
example  {R  :  Type*}  [CommRing  R]  (x  y  :  R)  :  (x  +  y)  ^  2  =  x  ^  2  +  y  ^  2  +  2  *  x  *  y  :=  by  ring 
```

更为奇特的变体不需要`R`上的加法构成一个群，而只需要一个加法幺半群。相应的类型类是`Semiring R`和`CommSemiring R`。自然数的类型是`CommSemiring R`的一个重要实例，任何取自然数值的函数的类型也是如此。另一个重要例子是环中的理想类型，将在下面讨论。`ring`策略的名称是双倍误导性的，因为它假设了交换性，但在半环中也能工作。换句话说，它适用于任何`CommSemiring`。

```py
example  (x  y  :  ℕ)  :  (x  +  y)  ^  2  =  x  ^  2  +  y  ^  2  +  2  *  x  *  y  :=  by  ring 
```

环和半环类的版本也有不假设存在乘法单位元或乘法结合律的情况。我们在这里不会讨论这些情况。

一些在环论导论中传统上教授的概念实际上是关于底乘法幺半群的。一个突出的例子是环的单位定义。每个（乘法）幺半群 `M` 都有一个谓词 `IsUnit : M → Prop`，断言存在一个两边的逆元，一个单位类型 `Units M`，记作 `Mˣ`，以及一个到 `M` 的强制转换。类型 `Units M` 将可逆元素与其逆元以及确保每个确实是另一个的逆元的属性捆绑在一起。这个实现细节主要在定义可计算函数时相关。在大多数情况下，可以使用 `IsUnit.unit {x : M} : IsUnit x → Mˣ` 来构建一个单位。在交换情况下，还有一个 `Units.mkOfMulEqOne (x y : M) : x * y = 1 → Mˣ`，它构建了被视为单位的 `x`。

```py
example  (x  :  ℤˣ)  :  x  =  1  ∨  x  =  -1  :=  Int.units_eq_one_or  x

example  {M  :  Type*}  [Monoid  M]  (x  :  Mˣ)  :  (x  :  M)  *  x⁻¹  =  1  :=  Units.mul_inv  x

example  {M  :  Type*}  [Monoid  M]  :  Group  Mˣ  :=  inferInstance 
```

两个（半）环 `R` 和 `S` 之间的环同态类型是 `RingHom R S`，记作 `R →+* S`。

```py
example  {R  S  :  Type*}  [Ring  R]  [Ring  S]  (f  :  R  →+*  S)  (x  y  :  R)  :
  f  (x  +  y)  =  f  x  +  f  y  :=  f.map_add  x  y

example  {R  S  :  Type*}  [Ring  R]  [Ring  S]  (f  :  R  →+*  S)  :  Rˣ  →*  Sˣ  :=
  Units.map  f 
```

同构变体是 `RingEquiv`，记作 `≃+*`。

与子幺半群和子群一样，存在一个 `Subring R` 类型，用于表示环 `R` 的子环，但这个类型比子群类型要少用得多，因为不能通过子环来商环。

```py
example  {R  :  Type*}  [Ring  R]  (S  :  Subring  R)  :  Ring  S  :=  inferInstance 
```

还要注意，`RingHom.range` 产生一个子环。

### 9.2.2\. 理想和商

由于历史原因，Mathlib 只为交换环提供了理想理论。（环库最初是为了快速推进现代代数几何的基础而开发的。）因此，在本节中，我们将使用交换（半）环。`R` 的理想被定义为将 `R` 视为 `R`-模的子模。模将在线性代数章节中稍后介绍，但这个实现细节可以大部分安全忽略，因为大多数（但不是所有）相关引理都在理想的特殊上下文中重新表述。但是匿名投影符号并不总是按预期工作。例如，在下面的片段中不能将 `Ideal.Quotient.mk I` 替换为 `I.Quotient.mk`，因为有两个 `.`，所以它将被解析为 `(Ideal.Quotient I).mk`；但 `Ideal.Quotient` 本身并不存在。

```py
example  {R  :  Type*}  [CommRing  R]  (I  :  Ideal  R)  :  R  →+*  R  ⧸  I  :=
  Ideal.Quotient.mk  I

example  {R  :  Type*}  [CommRing  R]  {a  :  R}  {I  :  Ideal  R}  :
  Ideal.Quotient.mk  I  a  =  0  ↔  a  ∈  I  :=
  Ideal.Quotient.eq_zero_iff_mem 
```

商环的泛性质是 `Ideal.Quotient.lift`。

```py
example  {R  S  :  Type*}  [CommRing  R]  [CommRing  S]  (I  :  Ideal  R)  (f  :  R  →+*  S)
  (H  :  I  ≤  RingHom.ker  f)  :  R  ⧸  I  →+*  S  :=
  Ideal.Quotient.lift  I  f  H 
```

尤其是导致环的第一同构定理。

```py
example  {R  S  :  Type*}  [CommRing  R]  CommRing  S  :
  R  ⧸  RingHom.ker  f  ≃+*  f.range  :=
  RingHom.quotientKerEquivRange  f 
```

理想在包含关系下形成一个完全格结构，以及一个半环结构。这两个结构相互作用得很好。

```py
variable  {R  :  Type*}  [CommRing  R]  {I  J  :  Ideal  R}

example  :  I  +  J  =  I  ⊔  J  :=  rfl

example  {x  :  R}  :  x  ∈  I  +  J  ↔  ∃  a  ∈  I,  ∃  b  ∈  J,  a  +  b  =  x  :=  by
  simp  [Submodule.mem_sup]

example  :  I  *  J  ≤  J  :=  Ideal.mul_le_left

example  :  I  *  J  ≤  I  :=  Ideal.mul_le_right

example  :  I  *  J  ≤  I  ⊓  J  :=  Ideal.mul_le_inf 
```

可以使用环同态通过 `Ideal.map` 推理想前移，通过 `Ideal.comap` 拉回理想。通常，后者更方便使用，因为它不涉及存在量词。这解释了为什么它被用来陈述允许我们在商环之间构建同态的条件。

```py
example  {R  S  :  Type*}  [CommRing  R]  [CommRing  S]  (I  :  Ideal  R)  (J  :  Ideal  S)  (f  :  R  →+*  S)
  (H  :  I  ≤  Ideal.comap  f  J)  :  R  ⧸  I  →+*  S  ⧸  J  :=
  Ideal.quotientMap  J  f  H 
```

一个微妙之处在于类型`R ⧸ I`实际上依赖于`I`（直到定义等价），因此，两个理想`I`和`J`相等的证明不足以使相应的商相等。然而，普遍性质在这种情况下确实提供了一个同构。

```py
example  {R  :  Type*}  [CommRing  R]  {I  J  :  Ideal  R}  (h  :  I  =  J)  :  R  ⧸  I  ≃+*  R  ⧸  J  :=
  Ideal.quotEquivOfEq  h 
```

我们现在可以将中国剩余同构作为一个例子来展示。请注意，索引下确界符号`⨅`和类型的大乘积符号`Π`之间的区别。根据你的字体，这些可能很难区分。

```py
example  {R  :  Type*}  [CommRing  R]  {ι  :  Type*}  [Fintype  ι]  (f  :  ι  →  Ideal  R)
  (hf  :  ∀  i  j,  i  ≠  j  →  IsCoprime  (f  i)  (f  j))  :  (R  ⧸  ⨅  i,  f  i)  ≃+*  Π  i,  R  ⧸  f  i  :=
  Ideal.quotientInfRingEquivPiQuotient  f  hf 
```

中国剩余定理的初等版本，一个关于`ZMod`的陈述，可以很容易地从先前的定理中推导出来：

```py
open  BigOperators  PiNotation

example  {ι  :  Type*}  [Fintype  ι]  (a  :  ι  →  ℕ)  (coprime  :  ∀  i  j,  i  ≠  j  →  (a  i).Coprime  (a  j))  :
  ZMod  (∏  i,  a  i)  ≃+*  Π  i,  ZMod  (a  i)  :=
  ZMod.prodEquivPi  a  coprime 
```

作为一系列练习，我们将重新证明一般情况下的中国剩余定理。

我们首先需要定义定理中出现的映射，作为一个环同态，使用商环的普遍性质。

```py
variable  {ι  R  :  Type*}  [CommRing  R]
open  Ideal  Quotient  Function

#check  Pi.ringHom
#check  ker_Pi_Quotient_mk

/-- The homomorphism from ``R ⧸ ⨅ i, I i`` to ``Π i, R ⧸ I i`` featured in the Chinese
 Remainder Theorem. -/
def  chineseMap  (I  :  ι  →  Ideal  R)  :  (R  ⧸  ⨅  i,  I  i)  →+*  Π  i,  R  ⧸  I  i  :=
  sorry 
```

确保以下两个引理可以通过`rfl`证明。

```py
lemma  chineseMap_mk  (I  :  ι  →  Ideal  R)  (x  :  R)  :
  chineseMap  I  (Quotient.mk  _  x)  =  fun  i  :  ι  ↦  Ideal.Quotient.mk  (I  i)  x  :=
  sorry

lemma  chineseMap_mk'  (I  :  ι  →  Ideal  R)  (x  :  R)  (i  :  ι)  :
  chineseMap  I  (mk  _  x)  i  =  mk  (I  i)  x  :=
  sorry 
```

下一个引理证明了中国剩余定理简单一半的证明，没有任何关于理想族的前提。证明不到一行长。

```py
#check  injective_lift_iff

lemma  chineseMap_inj  (I  :  ι  →  Ideal  R)  :  Injective  (chineseMap  I)  :=  by
  sorry 
```

我们现在准备好证明定理的核心，这将展示我们的`chineseMap`的满射性。首先，我们需要知道表达互质（也称为共最大性假设）的不同方式。下面只需要前两种。

```py
#check  IsCoprime
#check  isCoprime_iff_add
#check  isCoprime_iff_exists
#check  isCoprime_iff_sup_eq
#check  isCoprime_iff_codisjoint 
```

我们利用归纳法于`Finset`。以下给出了`Finset`的相关引理。记住，`ring`策略适用于半环，并且环的理想形成一个半环。

```py
#check  Finset.mem_insert_of_mem
#check  Finset.mem_insert_self

theorem  isCoprime_Inf  {I  :  Ideal  R}  {J  :  ι  →  Ideal  R}  {s  :  Finset  ι}
  (hf  :  ∀  j  ∈  s,  IsCoprime  I  (J  j))  :  IsCoprime  I  (⨅  j  ∈  s,  J  j)  :=  by
  classical
  simp_rw  [isCoprime_iff_add]  at  *
  induction  s  using  Finset.induction  with
  |  empty  =>
  simp
  |  @insert  i  s  _  hs  =>
  rw  [Finset.iInf_insert,  inf_comm,  one_eq_top,  eq_top_iff,  ←  one_eq_top]
  set  K  :=  ⨅  j  ∈  s,  J  j
  calc
  1  =  I  +  K  :=  sorry
  _  =  I  +  K  *  (I  +  J  i)  :=  sorry
  _  =  (1  +  K)  *  I  +  K  *  J  i  :=  sorry
  _  ≤  I  +  K  ⊓  J  i  :=  sorry 
```

我们现在可以证明中国剩余定理中出现的映射的满射性。

```py
lemma  chineseMap_surj  [Fintype  ι]  {I  :  ι  →  Ideal  R}
  (hI  :  ∀  i  j,  i  ≠  j  →  IsCoprime  (I  i)  (I  j))  :  Surjective  (chineseMap  I)  :=  by
  classical
  intro  g
  choose  f  hf  using  fun  i  ↦  Ideal.Quotient.mk_surjective  (g  i)
  have  key  :  ∀  i,  ∃  e  :  R,  mk  (I  i)  e  =  1  ∧  ∀  j,  j  ≠  i  →  mk  (I  j)  e  =  0  :=  by
  intro  i
  have  hI'  :  ∀  j  ∈  ({i}  :  Finset  ι)ᶜ,  IsCoprime  (I  i)  (I  j)  :=  by
  sorry
  sorry
  choose  e  he  using  key
  use  mk  _  (∑  i,  f  i  *  e  i)
  sorry 
```

现在所有的部分都在以下内容中汇集在一起：

```py
noncomputable  def  chineseIso  [Fintype  ι]  (f  :  ι  →  Ideal  R)
  (hf  :  ∀  i  j,  i  ≠  j  →  IsCoprime  (f  i)  (f  j))  :  (R  ⧸  ⨅  i,  f  i)  ≃+*  Π  i,  R  ⧸  f  i  :=
  {  Equiv.ofBijective  _  ⟨chineseMap_inj  f,  chineseMap_surj  hf⟩,
  chineseMap  f  with  } 
```

### 9.2.3\. 代数和多项式

给定一个交换（半）环`R`，一个在`R`上的代数是一个半环`A`，它配备了一个环同态，其像与`A`的每个元素交换。这被编码为类型类`Algebra R A`。从`R`到`A`的态射被称为结构映射，在 Lean 中表示为`algebraMap R A : R →+* A`。对于某个`r : R`，`a : A`通过`algebraMap R A r`的乘法称为`a`通过`r`的标量乘法，表示为`r • a`。请注意，这种代数的概念有时被称为*结合有单位代数*，以强调存在更一般的代数概念。

`algebraMap R A`是环同态的事实将标量乘法的许多性质打包在一起，如下所示：

```py
example  {R  A  :  Type*}  [CommRing  R]  [Ring  A]  [Algebra  R  A]  (r  r'  :  R)  (a  :  A)  :
  (r  +  r')  •  a  =  r  •  a  +  r'  •  a  :=
  add_smul  r  r'  a

example  {R  A  :  Type*}  [CommRing  R]  [Ring  A]  [Algebra  R  A]  (r  r'  :  R)  (a  :  A)  :
  (r  *  r')  •  a  =  r  •  r'  •  a  :=
  mul_smul  r  r'  a 
```

两个`R`-代数`A`和`B`之间的态射是环同态，并且与`R`中元素的标量乘法交换。它们是类型为`AlgHom R A B`的打包态射，表示为`A →ₐ[R] B`。

非交换代数的重要例子包括自同态代数和方阵代数，这两者都将在线性代数章节中介绍。在本章中，我们将讨论一个最重要的交换代数例子，即多项式代数。

在 `R` 中系数的单变量多项式代数称为 `Polynomial R`，一旦打开 `Polynomial` 命名空间，就可以写成 `R[X]`。从 `R` 到 `R[X]` 的代数结构映射用 `C` 表示，代表“常数”，因为相应的多项式函数始终是常数。不定元用 `X` 表示。

```py
open  Polynomial

example  {R  :  Type*}  [CommRing  R]  :  R[X]  :=  X

example  {R  :  Type*}  [CommRing  R]  (r  :  R)  :=  X  -  C  r 
```

在上述第一个例子中，我们向 Lean 提供预期的类型至关重要，因为它不能从定义的主体中确定。在第二个例子中，目标多项式代数可以通过我们对 `C r` 的使用来推断，因为 `r` 的类型是已知的。

因为 `C` 是从 `R` 到 `R[X]` 的环同态，所以我们可以在环 `R[X]` 中计算之前，使用所有环同态引理，如 `map_zero`、`map_one`、`map_mul` 和 `map_pow`。例如：

```py
example  {R  :  Type*}  [CommRing  R]  (r  :  R)  :  (X  +  C  r)  *  (X  -  C  r)  =  X  ^  2  -  C  (r  ^  2)  :=  by
  rw  [C.map_pow]
  ring 
```

您可以使用 `Polynomial.coeff` 访问系数。

```py
example  {R  :  Type*}  [CommRing  R]  (r:R)  :  (C  r).coeff  0  =  r  :=  by  simp

example  {R  :  Type*}  [CommRing  R]  :  (X  ^  2  +  2  *  X  +  C  3  :  R[X]).coeff  1  =  2  :=  by  simp 
```

定义多项式的次数总是很棘手，因为零多项式的特殊情况。Mathlib 有两种变体：`Polynomial.natDegree : R[X] → ℕ` 将零多项式的次数赋值为 `0`，而 `Polynomial.degree : R[X] → WithBot ℕ` 赋值为 `⊥`。在后一种情况中，`WithBot ℕ` 可以看作是 `ℕ ∪ {-∞}`，除了 `-∞` 用 `⊥` 表示，这与完备格中的底元素符号相同。这个特殊值被用作零多项式的次数，并且对加法具有吸收性。（对于乘法几乎具有吸收性，除了 `⊥ * 0 = 0`。）

从道德上讲，`degree` 版本是正确的。例如，它允许我们陈述乘积次数的预期公式（假设基环没有零因子）。

```py
example  {R  :  Type*}  [Semiring  R]  [NoZeroDivisors  R]  {p  q  :  R[X]}  :
  degree  (p  *  q)  =  degree  p  +  degree  q  :=
  Polynomial.degree_mul 
```

而 `natDegree` 版本需要假设非零多项式。

```py
example  {R  :  Type*}  [Semiring  R]  [NoZeroDivisors  R]  {p  q  :  R[X]}  (hp  :  p  ≠  0)  (hq  :  q  ≠  0)  :
  natDegree  (p  *  q)  =  natDegree  p  +  natDegree  q  :=
  Polynomial.natDegree_mul  hp  hq 
```

然而，`ℕ` 比使用 `WithBot ℕ` 更方便，所以 Mathlib 提供了这两种版本，并提供引理在它们之间进行转换。此外，`natDegree` 是在计算复合次数时更方便的定义。多项式的复合是 `Polynomial.comp`，我们有：

```py
example  {R  :  Type*}  [Semiring  R]  [NoZeroDivisors  R]  {p  q  :  R[X]}  :
  natDegree  (comp  p  q)  =  natDegree  p  *  natDegree  q  :=
  Polynomial.natDegree_comp 
```

多项式产生了多项式函数：任何多项式都可以使用 `Polynomial.eval` 在 `R` 上进行评估。

```py
example  {R  :  Type*}  [CommRing  R]  (P:  R[X])  (x  :  R)  :=  P.eval  x

example  {R  :  Type*}  [CommRing  R]  (r  :  R)  :  (X  -  C  r).eval  r  =  0  :=  by  simp 
```

特别是，存在一个谓词 `IsRoot`，它对多项式在 `R` 中的根元素 `r` 成立。

```py
example  {R  :  Type*}  [CommRing  R]  (P  :  R[X])  (r  :  R)  :  IsRoot  P  r  ↔  P.eval  r  =  0  :=  Iff.rfl 
```

我们想说的是，假设 `R` 没有零因子，一个多项式的根的数量最多与其次数相同，这里的根是按重数计算的。但一旦再次遇到零多项式的情况就令人痛苦。因此，Mathlib 定义 `Polynomial.roots` 将多项式 `P` 映射到一个多重集，即如果 `P` 为零，则定义为空集，否则为 `P` 的根及其重数。这个定义仅在基础环是整环时才有效，因为否则定义不具有好的性质。

```py
example  {R  :  Type*}  [CommRing  R]  [IsDomain  R]  (r  :  R)  :  (X  -  C  r).roots  =  {r}  :=
  roots_X_sub_C  r

example  {R  :  Type*}  [CommRing  R]  [IsDomain  R]  (r  :  R)  (n  :  ℕ):
  ((X  -  C  r)  ^  n).roots  =  n  •  {r}  :=
  by  simp 
```

`Polynomial.eval` 和 `Polynomial.roots` 都只考虑系数环。它们不允许我们说 `X ^ 2 - 2 : ℚ[X]` 在 `ℝ` 中有一个根，或者 `X ^ 2 + 1 : ℝ[X]` 在 `ℂ` 中有一个根。为此，我们需要 `Polynomial.aeval`，它将在任何 `R`-代数中评估 `P : R[X]`。更确切地说，给定一个半环 `A` 和一个 `Algebra R A` 实例，`Polynomial.aeval` 将 `a` 的每个元素通过评估在 `a` 上的 `R`-代数同态发送。由于 `AlgHom` 有一个强制到函数的转换，因此可以将它应用于多项式。但 `aeval` 没有以多项式作为参数，因此不能使用像上面 `P.eval` 中的点号表示法。

```py
example  :  aeval  Complex.I  (X  ^  2  +  1  :  ℝ[X])  =  0  :=  by  simp 
```

在这个上下文中，对应于 `roots` 的函数是 `aroots`，它接受一个多项式和一个代数，然后输出一个多重集（关于零多项式的警告与 `roots` 相同）。

```py
open  Complex  Polynomial

example  :  aroots  (X  ^  2  +  1  :  ℝ[X])  ℂ  =  {Complex.I,  -I}  :=  by
  suffices  roots  (X  ^  2  +  1  :  ℂ[X])  =  {I,  -I}  by  simpa  [aroots_def]
  have  factored  :  (X  ^  2  +  1  :  ℂ[X])  =  (X  -  C  I)  *  (X  -  C  (-I))  :=  by
  have  key  :  (C  I  *  C  I  :  ℂ[X])  =  -1  :=  by  simp  [←  C_mul]
  rw  [C_neg]
  linear_combination  key
  have  p_ne_zero  :  (X  -  C  I)  *  (X  -  C  (-I))  ≠  0  :=  by
  intro  H
  apply_fun  eval  0  at  H
  simp  [eval]  at  H
  simp  only  [factored,  roots_mul  p_ne_zero,  roots_X_sub_C]
  rfl

-- Mathlib knows about D'Alembert-Gauss theorem: ``ℂ`` is algebraically closed.
example  :  IsAlgClosed  ℂ  :=  inferInstance 
```

更一般地，给定一个环同态 `f : R →+* S`，可以使用 `Polynomial.eval₂` 在 `S` 中的某个点评估 `P : R[X]`。这个操作产生了一个从 `R[X]` 到 `S` 的实际函数，因为它不假设存在一个 `Algebra R S` 实例，所以点号表示法按预期工作。

```py
#check  (Complex.ofRealHom  :  ℝ  →+*  ℂ)

example  :  (X  ^  2  +  1  :  ℝ[X]).eval₂  Complex.ofRealHom  Complex.I  =  0  :=  by  simp 
```

让我们简要地提一下多元多项式。给定一个交换半环 `R`，系数在 `R` 中且不定元由类型 `σ` 索引的 `R`-代数多项式是 `MVPolynomial σ R`。给定 `i : σ`，相应的多项式是 `MvPolynomial.X i`。（像往常一样，可以打开 `MVPolynomial` 命名空间以缩短为 `X i`。）例如，如果我们想要两个不定元，我们可以使用 `Fin 2` 作为 `σ`，并将定义单位圆的 $\mathbb{R}²$ 中的多项式写为：

```py
open  MvPolynomial

def  circleEquation  :  MvPolynomial  (Fin  2)  ℝ  :=  X  0  ^  2  +  X  1  ^  2  -  1 
```

回想一下，函数应用具有非常高的优先级，因此上面的表达式读作 `(X 0) ^ 2 + (X 1) ^ 2 - 1`。我们可以评估它来确保坐标为 $(1, 0)$ 的点在圆上。回想一下，`![...]` 符号表示由自然数 `n` 确定的 `Fin n → X` 的元素，其中 `n` 由参数的数量决定，而 `X` 由参数的类型决定。

```py
example  :  MvPolynomial.eval  ![1,  0]  circleEquation  =  0  :=  by  simp  [circleEquation] 
``` 上一节 下一节

* * *

© 版权所有 2020-2025，Jeremy Avigad，Patrick Massot。文本许可协议为 CC BY 4.0。

使用[Sphinx](https://www.sphinx-doc.org/)和[Read the Docs](https://readthedocs.org)提供的[主题](https://github.com/readthedocs/sphinx_rtd_theme)构建。我们在第 2.2 节中看到了如何在群和环中推理运算。后来，在第 7.2 节第 7.2 节中，我们看到了如何定义抽象代数结构，例如群结构，以及具体的实例，如高斯整数上的环结构。第八章解释了在 Mathlib 中如何处理抽象结构的层次。

在本章中，我们将更详细地处理群和环。我们无法涵盖 Mathlib 中这些主题处理的各个方面，特别是由于 Mathlib 一直在不断增长。但我们将提供库的入口点，并展示基本概念是如何使用的。与第八章的讨论有一些重叠，但在这里我们将专注于如何使用 Mathlib，而不是处理这些主题的设计决策。因此，理解一些示例可能需要回顾第八章中的背景知识。

## 9.1\. 幺半群和群

### 9.1.1\. 幺半群及其同态

抽象代数课程通常从群开始，然后逐步过渡到环、域和向量空间。在讨论环上的乘法时，这涉及到一些扭曲，因为乘法运算并不来自群结构，但许多证明可以直接从群论转移到这个新环境。在用笔和纸做数学时，最常见的解决方案是将这些证明作为练习。一种不太高效但更安全、更符合形式化要求的做法是使用幺半群。一个类型 M 上的*幺半群*结构是一个内部组合律，它是结合的并有一个中性元素。幺半群主要用于适应群和环的乘法结构。但也有一些自然例子；例如，自然数集加上加法运算形成一个幺半群。

从实际的角度来看，在使用 Mathlib 时，你可以基本上忽略幺半群。但当你通过浏览 Mathlib 文件寻找引理时，你需要知道它们的存在。否则，你可能会在群论文件中寻找一个陈述，而实际上它是在与幺半群一起找到的，因为它不需要元素是可逆的。

类型 `M` 上的单群结构类型写成 `Monoid M`。函数 `Monoid` 是一个类型类，所以它几乎总是作为隐式参数实例（换句话说，在方括号中）出现。默认情况下，`Monoid` 使用乘法表示法进行操作；对于加法表示法，请使用 `AddMonoid`。这些结构的交换版本在 `Monoid` 前添加前缀 `Comm`。

```py
example  {M  :  Type*}  [Monoid  M]  (x  :  M)  :  x  *  1  =  x  :=  mul_one  x

example  {M  :  Type*}  [AddCommMonoid  M]  (x  y  :  M)  :  x  +  y  =  y  +  x  :=  add_comm  x  y 
```

注意，尽管 `AddMonoid` 在库中可以找到，但使用非交换操作的加法表示法通常很令人困惑。

两个单群 `M` 和 `N` 之间的同态类型称为 `MonoidHom M N`，并写成 `M →* N`。当我们将其应用于 `M` 的元素时，Lean 会自动将此类同态视为从 `M` 到 `N` 的函数。加法版本称为 `AddMonoidHom`，并写成 `M →+ N`。

```py
example  {M  N  :  Type*}  [Monoid  M]  [Monoid  N]  (x  y  :  M)  (f  :  M  →*  N)  :  f  (x  *  y)  =  f  x  *  f  y  :=
  f.map_mul  x  y

example  {M  N  :  Type*}  [AddMonoid  M]  [AddMonoid  N]  (f  :  M  →+  N)  :  f  0  =  0  :=
  f.map_zero 
```

这些同态是打包映射，即它们将映射及其一些属性打包在一起。记住，第 8.2 节 解释了打包映射；这里我们只是简单地指出一个不幸的后果，即我们不能使用普通函数组合来组合映射。相反，我们需要使用 `MonoidHom.comp` 和 `AddMonoidHom.comp`。

```py
example  {M  N  P  :  Type*}  [AddMonoid  M]  [AddMonoid  N]  [AddMonoid  P]
  (f  :  M  →+  N)  (g  :  N  →+  P)  :  M  →+  P  :=  g.comp  f 
```

### 9.1.2\. 群及其同态

我们将有很多关于群的话要说，群是具有额外属性的单群，即每个元素都有一个逆元。

```py
example  {G  :  Type*}  [Group  G]  (x  :  G)  :  x  *  x⁻¹  =  1  :=  mul_inv_cancel  x 
```

与我们之前看到的 `ring` 策略类似，存在一个 `group` 策略，它可以证明在任意群中成立的任何恒等式。（等价地，它证明了在自由群中成立的恒等式。）

```py
example  {G  :  Type*}  [Group  G]  (x  y  z  :  G)  :  x  *  (y  *  z)  *  (x  *  z)⁻¹  *  (x  *  y  *  x⁻¹)⁻¹  =  1  :=  by
  group 
```

此外，还有一个用于交换加法群恒等式的策略，称为 `abel`。

```py
example  {G  :  Type*}  [AddCommGroup  G]  (x  y  z  :  G)  :  z  +  x  +  (y  -  z  -  x)  =  y  :=  by
  abel 
```

有趣的是，群同态不过是在群之间的单群同态。因此，我们可以复制并粘贴我们之前的一个例子，将 `Monoid` 替换为 `Group`。

```py
example  {G  H  :  Type*}  [Group  G]  [Group  H]  (x  y  :  G)  (f  :  G  →*  H)  :  f  (x  *  y)  =  f  x  *  f  y  :=
  f.map_mul  x  y 
```

当然，我们确实得到了一些新的属性，例如这个：

```py
example  {G  H  :  Type*}  [Group  G]  [Group  H]  (x  :  G)  (f  :  G  →*  H)  :  f  (x⁻¹)  =  (f  x)⁻¹  :=
  f.map_inv  x 
```

你可能担心构造群同态会让我们做不必要的额外工作，因为单群同态的定义强制要求中性元素被发送到中性元素，而在群同态的情况下这是自动的。在实践中，额外的努力并不困难，但为了避免它，有一个函数可以从群之间的兼容于组合律的函数构建群同态。

```py
example  {G  H  :  Type*}  [Group  G]  [Group  H]  (f  :  G  →  H)  (h  :  ∀  x  y,  f  (x  *  y)  =  f  x  *  f  y)  :
  G  →*  H  :=
  MonoidHom.mk'  f  h 
```

此外，还有一个表示群（或单群）同构的类型 `MulEquiv`，用 `≃*` 表示（在加法表示法中用 `≃+` 表示 `AddEquiv`）。`f : G ≃* H` 的逆是 `MulEquiv.symm f : H ≃* G`，`f` 和 `g` 的组合是 `MulEquiv.trans f g`，恒等同构是 `MulEquiv.refl G`。使用匿名投影符号，前两个可以分别写成 `f.symm` 和 `f.trans g`。当需要时，此类型中的元素会自动转换为同态和函数。

```py
example  {G  H  :  Type*}  [Group  G]  [Group  H]  (f  :  G  ≃*  H)  :
  f.trans  f.symm  =  MulEquiv.refl  G  :=
  f.self_trans_symm 
```

可以使用 `MulEquiv.ofBijective` 从双射同态构建同构。这样做会使逆函数不可计算。

```py
noncomputable  example  {G  H  :  Type*}  [Group  G]  [Group  H]
  (f  :  G  →*  H)  (h  :  Function.Bijective  f)  :
  G  ≃*  H  :=
  MulEquiv.ofBijective  f  h 
```

### 9.1.3\. 子群

正如群同态是捆绑在一起一样，`G` 的一个子群也是一个由 `G` 中的集合及其相关闭包性质组成的捆绑结构。

```py
example  {G  :  Type*}  [Group  G]  (H  :  Subgroup  G)  {x  y  :  G}  (hx  :  x  ∈  H)  (hy  :  y  ∈  H)  :
  x  *  y  ∈  H  :=
  H.mul_mem  hx  hy

example  {G  :  Type*}  [Group  G]  (H  :  Subgroup  G)  {x  :  G}  (hx  :  x  ∈  H)  :
  x⁻¹  ∈  H  :=
  H.inv_mem  hx 
```

在上面的例子中，重要的是要理解 `Subgroup G` 是 `G` 的子群类型，而不是一个谓词 `IsSubgroup H`，其中 `H` 是 `Set G` 的一个元素。`Subgroup G` 被赋予了到 `Set G` 的强制转换和 `G` 上的成员谓词。参见 第 8.3 节 了解如何以及为什么这样做。

当然，如果两个子群具有相同的元素，则它们是相同的。这个事实被注册用于与 `ext` 策略一起使用，该策略可以用来证明两个子群相等，就像它被用来证明两个集合相等一样。

例如，为了陈述和证明 `ℤ` 是 `ℚ` 的加法子群，我们真正想要的是构造一个类型 `AddSubgroup ℚ` 的项，其投影到 `Set ℚ` 是 `ℤ`，或者更精确地说，是 `ℤ` 在 `ℚ` 中的像。

```py
example  :  AddSubgroup  ℚ  where
  carrier  :=  Set.range  ((↑)  :  ℤ  →  ℚ)
  add_mem'  :=  by
  rintro  _  _  ⟨n,  rfl⟩  ⟨m,  rfl⟩
  use  n  +  m
  simp
  zero_mem'  :=  by
  use  0
  simp
  neg_mem'  :=  by
  rintro  _  ⟨n,  rfl⟩
  use  -n
  simp 
```

使用类型类，Mathlib 知道群的一个子群继承了群结构。

```py
example  {G  :  Type*}  [Group  G]  (H  :  Subgroup  G)  :  Group  H  :=  inferInstance 
```

这个例子很微妙。对象 `H` 不是一个类型，但 Lean 会自动将其解释为 `G` 的子类型，将其强制转换为类型。因此，上面的例子可以更明确地重述为：

```py
example  {G  :  Type*}  [Group  G]  (H  :  Subgroup  G)  :  Group  {x  :  G  //  x  ∈  H}  :=  inferInstance 
```

拥有类型 `Subgroup G` 而不是谓词 `IsSubgroup : Set G → Prop` 的重要好处是，可以轻松地为 `Subgroup G` 赋予额外的结构。重要的是，它具有关于包含的完备格结构。例如，我们不是通过一个引理来声明 `G` 的两个子群的交集仍然是子群，而是使用了格运算 `⊓` 来构造交集。然后我们可以将关于格的任意引理应用于构造过程。

让我们验证两个子群的交集的底层集合确实，根据定义，是它们的交集。

```py
example  {G  :  Type*}  [Group  G]  (H  H'  :  Subgroup  G)  :
  ((H  ⊓  H'  :  Subgroup  G)  :  Set  G)  =  (H  :  Set  G)  ∩  (H'  :  Set  G)  :=  rfl 
```

对于底层集合的交集使用不同的符号可能看起来很奇怪，但这种对应关系并不适用于上确界运算和集合的并集，因为子群的并集在一般情况下不是子群。相反，需要使用由并集生成的子群，这可以通过 `Subgroup.closure` 来完成。

```py
example  {G  :  Type*}  [Group  G]  (H  H'  :  Subgroup  G)  :
  ((H  ⊔  H'  :  Subgroup  G)  :  Set  G)  =  Subgroup.closure  ((H  :  Set  G)  ∪  (H'  :  Set  G))  :=  by
  rw  [Subgroup.sup_eq_closure] 
```

另一个微妙之处在于，`G` 本身没有类型 `Subgroup G`，因此我们需要一种方式来谈论 `G` 作为 `G` 的子群。这也由格结构提供：整个子群是这个格的顶元素。

```py
example  {G  :  Type*}  [Group  G]  (x  :  G)  :  x  ∈  (⊤  :  Subgroup  G)  :=  trivial 
```

类似地，这个格的底元素是只有一个中性元素的子群。

```py
example  {G  :  Type*}  [Group  G]  (x  :  G)  :  x  ∈  (⊥  :  Subgroup  G)  ↔  x  =  1  :=  Subgroup.mem_bot 
```

作为操作群和子群的一个练习，你可以定义由环境群中的元素生成的子群的共轭。

```py
def  conjugate  {G  :  Type*}  [Group  G]  (x  :  G)  (H  :  Subgroup  G)  :  Subgroup  G  where
  carrier  :=  {a  :  G  |  ∃  h,  h  ∈  H  ∧  a  =  x  *  h  *  x⁻¹}
  one_mem'  :=  by
  dsimp
  sorry
  inv_mem'  :=  by
  dsimp
  sorry
  mul_mem'  :=  by
  dsimp
  sorry 
```

将前两个主题结合起来，可以使用群同态来推前和拉回子群。Mathlib 中的命名约定是将这些操作称为 `map` 和 `comap`。这些不是常见的数学术语，但它们的优势是比“推前”和“直接像”更短。

```py
example  {G  H  :  Type*}  [Group  G]  [Group  H]  (G'  :  Subgroup  G)  (f  :  G  →*  H)  :  Subgroup  H  :=
  Subgroup.map  f  G'

example  {G  H  :  Type*}  [Group  G]  [Group  H]  (H'  :  Subgroup  H)  (f  :  G  →*  H)  :  Subgroup  G  :=
  Subgroup.comap  f  H'

#check  Subgroup.mem_map
#check  Subgroup.mem_comap 
```

特别是，映射 `f` 下底子群的逆像是称为 `f` 的 *核* 的子群，而 `f` 的值域也是一个子群。

```py
example  {G  H  :  Type*}  [Group  G]  [Group  H]  (f  :  G  →*  H)  (g  :  G)  :
  g  ∈  MonoidHom.ker  f  ↔  f  g  =  1  :=
  f.mem_ker

example  {G  H  :  Type*}  [Group  G]  [Group  H]  (f  :  G  →*  H)  (h  :  H)  :
  h  ∈  MonoidHom.range  f  ↔  ∃  g  :  G,  f  g  =  h  :=
  f.mem_range 
```

作为操作群同态和子群的计算练习，让我们证明一些基本性质。它们已经在 Mathlib 中得到了证明，所以如果你想从这些练习中受益，不要急于使用 `exact?`。

```py
section  exercises
variable  {G  H  :  Type*}  [Group  G]  [Group  H]

open  Subgroup

example  (φ  :  G  →*  H)  (S  T  :  Subgroup  H)  (hST  :  S  ≤  T)  :  comap  φ  S  ≤  comap  φ  T  :=  by
  sorry

example  (φ  :  G  →*  H)  (S  T  :  Subgroup  G)  (hST  :  S  ≤  T)  :  map  φ  S  ≤  map  φ  T  :=  by
  sorry

variable  {K  :  Type*}  [Group  K]

-- Remember you can use the `ext` tactic to prove an equality of subgroups.
example  (φ  :  G  →*  H)  (ψ  :  H  →*  K)  (U  :  Subgroup  K)  :
  comap  (ψ.comp  φ)  U  =  comap  φ  (comap  ψ  U)  :=  by
  sorry

-- Pushing a subgroup along one homomorphism and then another is equal to
-- pushing it forward along the composite of the homomorphisms.
example  (φ  :  G  →*  H)  (ψ  :  H  →*  K)  (S  :  Subgroup  G)  :
  map  (ψ.comp  φ)  S  =  map  ψ  (S.map  φ)  :=  by
  sorry

end  exercises 
```

让我们用两个非常经典的结果来完成对 Mathlib 中子群的介绍。拉格朗日定理表明有限群子群的基数是群基数的约数。西罗第一定理是拉格朗日定理的一个著名的部分逆定理。

虽然 Mathlib 的这个角落部分是为了允许计算而设置的，但我们可以使用以下 `open scoped` 命令告诉 Lean 使用非构造性逻辑。

```py
open  scoped  Classical

example  {G  :  Type*}  [Group  G]  (G'  :  Subgroup  G)  :  Nat.card  G'  ∣  Nat.card  G  :=
  ⟨G'.index,  mul_comm  G'.index  _  ▸  G'.index_mul_card.symm⟩

open  Subgroup

example  {G  :  Type*}  [Group  G]  [Finite  G]  (p  :  ℕ)  {n  :  ℕ}  [Fact  p.Prime]
  (hdvd  :  p  ^  n  ∣  Nat.card  G)  :  ∃  K  :  Subgroup  G,  Nat.card  K  =  p  ^  n  :=
  Sylow.exists_subgroup_card_pow_prime  p  hdvd 
```

接下来的两个练习推导出拉格朗日引理的一个推论。（这已经在 Mathlib 中了，所以不要急于使用 `exact?`。）

```py
lemma  eq_bot_iff_card  {G  :  Type*}  [Group  G]  {H  :  Subgroup  G}  :
  H  =  ⊥  ↔  Nat.card  H  =  1  :=  by
  suffices  (∀  x  ∈  H,  x  =  1)  ↔  ∃  x  ∈  H,  ∀  a  ∈  H,  a  =  x  by
  simpa  [eq_bot_iff_forall,  Nat.card_eq_one_iff_exists]
  sorry

#check  card_dvd_of_le

lemma  inf_bot_of_coprime  {G  :  Type*}  [Group  G]  (H  K  :  Subgroup  G)
  (h  :  (Nat.card  H).Coprime  (Nat.card  K))  :  H  ⊓  K  =  ⊥  :=  by
  sorry 
```

### 9.1.4\. 具体群

在 Mathlib 中也可以操作具体群，尽管这通常比操作抽象理论更复杂。例如，给定任何类型 `X`，`X` 的排列群是 `Equiv.Perm X`。特别是，对称群 $\mathfrak{S}_n$ 是 `Equiv.Perm (Fin n)`。可以对这个群陈述抽象结果，例如，如果 `X` 是有限的，则 `Equiv.Perm X` 由循环生成。

```py
open  Equiv

example  {X  :  Type*}  [Finite  X]  :  Subgroup.closure  {σ  :  Perm  X  |  Perm.IsCycle  σ}  =  ⊤  :=
  Perm.closure_isCycle 
```

可以完全具体并计算循环的实际乘积。以下我们使用 `#simp` 命令，它在一个给定的表达式中调用 `simp` 策略。符号 `c[]` 用于定义循环排列。在示例中，结果是 `ℕ` 的排列。可以在第一个数字上使用类型注解，如 `(1 : Fin 5)`，使其成为 `Perm (Fin 5)` 中的计算。

```py
#simp  [mul_assoc]  c[1,  2,  3]  *  c[2,  3,  4] 
```

另一种与具体群一起工作的方法是使用自由群和群表示。类型 `α` 上的自由群是 `FreeGroup α`，包含映射是 `FreeGroup.of : α → FreeGroup α`。例如，让我们定义一个有三个元素 `a`、`b` 和 `c` 的类型 `S`，以及对应自由群中的元素 `ab⁻¹`。

```py
section  FreeGroup

inductive  S  |  a  |  b  |  c

open  S

def  myElement  :  FreeGroup  S  :=  (.of  a)  *  (.of  b)⁻¹ 
```

注意，我们给出了定义的预期类型，这样 Lean 就知道 `.of` 的意思是 `FreeGroup.of`。

自由群的通用性质体现在 `FreeGroup.lift` 的等价性中。例如，让我们定义从 `FreeGroup S` 到 `Perm (Fin 5)` 的群同态，将 `a` 映射到 `c[1, 2, 3]`，`b` 映射到 `c[2, 3, 1]`，`c` 映射到 `c[2, 3]`，

```py
def  myMorphism  :  FreeGroup  S  →*  Perm  (Fin  5)  :=
  FreeGroup.lift  fun  |  .a  =>  c[1,  2,  3]
  |  .b  =>  c[2,  3,  1]
  |  .c  =>  c[2,  3] 
```

作为最后一个具体的例子，让我们看看如何定义一个由单个元素生成且其立方为 1 的群（因此该群将与$\mathbb{Z}/3$同构），并构建从该群到`Perm (Fin 5)`的态射。

作为只有一个元素的类型，我们将使用`Unit`，其唯一元素用`()`表示。函数`PresentedGroup`接受一组关系，即某个自由群的一组元素，并返回一个通过由关系生成的正规子群商化的自由群。 (我们将在第 9.1.6 节中看到如何处理更一般化的商。) 由于我们以某种方式将此隐藏在定义之后，我们使用`deriving Group`来强制在`myGroup`上创建一个群实例。

```py
def  myGroup  :=  PresentedGroup  {.of  ()  ^  3}  deriving  Group 
```

呈现群的全称性质确保了可以从将关系映射到目标群中性元素的函数构建出从这个群的外部态射。因此，我们需要这样的函数和一个证明该条件成立。然后我们可以将这个证明输入到`PresentedGroup.toGroup`中，以获得所需的群态射。

```py
def  myMap  :  Unit  →  Perm  (Fin  5)
|  ()  =>  c[1,  2,  3]

lemma  compat_myMap  :
  ∀  r  ∈  ({.of  ()  ^  3}  :  Set  (FreeGroup  Unit)),  FreeGroup.lift  myMap  r  =  1  :=  by
  rintro  _  rfl
  simp
  decide

def  myNewMorphism  :  myGroup  →*  Perm  (Fin  5)  :=  PresentedGroup.toGroup  compat_myMap

end  FreeGroup 
```

### 9.1.5\. 群作用

群论与数学其他部分的一个重要交互方式是通过群作用的使用。一个群`G`在某个类型`X`上的作用不过是`G`到`Equiv.Perm X`的态射。所以在某种意义上，群作用已经被之前的讨论所涵盖。但我们不希望携带这个态射；相反，我们希望尽可能由 Lean 自动推断出它。因此，我们有一个类型类来处理这种情况，即`MulAction G X`。这种设置的缺点是，在同一个类型上有多个同一群的作用需要一些扭曲，例如定义类型同义词，每个同义词都携带不同的类型类实例。

这使我们能够特别地使用`g • x`来表示群元素`g`对点`x`的作用。

```py
noncomputable  section  GroupActions

example  {G  X  :  Type*}  [Group  G]  [MulAction  G  X]  (g  g':  G)  (x  :  X)  :
  g  •  (g'  •  x)  =  (g  *  g')  •  x  :=
  (mul_smul  g  g'  x).symm 
```

对于加法群也有一个版本，称为`AddAction`，其中作用用`+ᵥ`表示。这被用于例如仿射空间的定义中。

```py
example  {G  X  :  Type*}  [AddGroup  G]  [AddAction  G  X]  (g  g'  :  G)  (x  :  X)  :
  g  +ᵥ  (g'  +ᵥ  x)  =  (g  +  g')  +ᵥ  x  :=
  (add_vadd  g  g'  x).symm 
```

基础群态射被称为`MulAction.toPermHom`。

```py
open  MulAction

example  {G  X  :  Type*}  [Group  G]  [MulAction  G  X]  :  G  →*  Equiv.Perm  X  :=
  toPermHom  G  X 
```

作为说明，让我们看看如何定义任何群`G`的凯莱同构嵌入到排列群中，即`Perm G`。

```py
def  CayleyIsoMorphism  (G  :  Type*)  [Group  G]  :  G  ≃*  (toPermHom  G  G).range  :=
  Equiv.Perm.subgroupOfMulAction  G  G 
```

注意，在上述定义之前，并没有要求必须有群而不是幺半群（或任何带有乘法运算的类型）。

当我们想要将`X`划分为轨道时，群条件真正进入画面。`X`上的对应等价关系被称为`MulAction.orbitRel`。它没有被声明为全局实例。

```py
example  {G  X  :  Type*}  [Group  G]  [MulAction  G  X]  :  Setoid  X  :=  orbitRel  G  X 
```

使用这个，我们可以陈述 `X` 在 `G` 的作用下被划分为轨道。更精确地说，我们得到 `X` 与依赖积 `(ω : orbitRel.Quotient G X) × (orbit G (Quotient.out' ω))` 之间的双射，其中 `Quotient.out' ω` 简单地选择一个投影到 `ω` 的元素。回想一下，这个依赖积的元素是 `⟨ω, x⟩` 这样的对，其中 `x` 的类型 `orbit G (Quotient.out' ω)` 依赖于 `ω`。

```py
example  {G  X  :  Type*}  [Group  G]  [MulAction  G  X]  :
  X  ≃  (ω  :  orbitRel.Quotient  G  X)  ×  (orbit  G  (Quotient.out  ω))  :=
  MulAction.selfEquivSigmaOrbits  G  X 
```

特别地，当 `X` 是有限的，这可以与 `Fintype.card_congr` 和 `Fintype.card_sigma` 结合，推导出 `X` 的基数是轨道基数的和。此外，轨道与通过稳定子群左平移作用下的 `G` 的商之间存在双射。这种通过左平移作用的子群作用被用来定义通过子群进行商的群，记作 /，因此我们可以使用以下简洁的陈述。

```py
example  {G  X  :  Type*}  [Group  G]  [MulAction  G  X]  (x  :  X)  :
  orbit  G  x  ≃  G  ⧸  stabilizer  G  x  :=
  MulAction.orbitEquivQuotientStabilizer  G  x 
```

上面的两个结果的结合的一个重要特殊情况是当 `X` 是一个带有子群 `H` 通过平移作用的群 `G`。在这种情况下，所有稳定子群都是平凡的，因此每个轨道都与 `H` 之间存在双射，我们得到：

```py
example  {G  :  Type*}  [Group  G]  (H  :  Subgroup  G)  :  G  ≃  (G  ⧸  H)  ×  H  :=
  groupEquivQuotientProdSubgroup 
```

这是上面我们看到的拉格朗日定理的概念变体。注意这个版本没有有限性的假设。

作为本节的练习，让我们通过共轭使用我们之前练习中定义的 `共轭` 来构建一个群对其子群的作用。

```py
variable  {G  :  Type*}  [Group  G]

lemma  conjugate_one  (H  :  Subgroup  G)  :  conjugate  1  H  =  H  :=  by
  sorry

instance  :  MulAction  G  (Subgroup  G)  where
  smul  :=  conjugate
  one_smul  :=  by
  sorry
  mul_smul  :=  by
  sorry

end  GroupActions 
```

### 9.1.6. 商群

在上述关于子群在群上作用的讨论中，我们看到了 `G ⧸ H` 的商出现。一般来说，这只是一个类型。它可以赋予一个群结构，使得商映射是一个群同态当且仅当 `H` 是正规子群（并且这种群结构是唯一的）。

正规性假设是一个类型类 `Subgroup.Normal`，这样类型类推理就可以用它来推导商上的群结构。

```py
noncomputable  section  QuotientGroup

example  {G  :  Type*}  [Group  G]  (H  :  Subgroup  G)  [H.Normal]  :  Group  (G  ⧸  H)  :=  inferInstance

example  {G  :  Type*}  [Group  G]  (H  :  Subgroup  G)  [H.Normal]  :  G  →*  G  ⧸  H  :=
  QuotientGroup.mk'  H 
```

商群的全称性质通过 `QuotientGroup.lift` 访问：一个群形态 `φ` 只要其核包含 `N`，就会下降到 `G ⧸ N`。

```py
example  {G  :  Type*}  [Group  G]  (N  :  Subgroup  G)  [N.Normal]  {M  :  Type*}
  [Group  M]  (φ  :  G  →*  M)  (h  :  N  ≤  MonoidHom.ker  φ)  :  G  ⧸  N  →*  M  :=
  QuotientGroup.lift  N  φ  h 
```

在上面的代码片段中，目标群被称为 `M` 是一个线索，表明在 `M` 上有一个幺半群结构就足够了。

一个重要的特殊情况是当 `N = ker φ`。在这种情况下，下降的形态是单射的，我们得到一个到其像上的群同构。这个结果通常被称为第一同构定理。

```py
example  {G  :  Type*}  [Group  G]  {M  :  Type*}  [Group  M]  (φ  :  G  →*  M)  :
  G  ⧸  MonoidHom.ker  φ  →*  MonoidHom.range  φ  :=
  QuotientGroup.quotientKerEquivRange  φ 
```

将通用性质应用于一个形态 `φ : G →* G'` 与商群投影 `Quotient.mk' N'` 的组合，我们也可以寻求从 `G ⧸ N` 到 `G' ⧸ N'` 的形态。对 `φ` 的要求通常表述为“`φ` 应该将 `N` 发送到 `N'` 内部。”但这等同于要求 `φ` 将 `N'` 拉回到 `N` 上，而后一种条件更容易处理，因为拉回的定义不涉及存在量词。

```py
example  {G  G':  Type*}  [Group  G]  [Group  G']
  {N  :  Subgroup  G}  [N.Normal]  {N'  :  Subgroup  G'}  [N'.Normal]
  {φ  :  G  →*  G'}  (h  :  N  ≤  Subgroup.comap  φ  N')  :  G  ⧸  N  →*  G'  ⧸  N':=
  QuotientGroup.map  N  N'  φ  h 
```

需要记住的一个微妙之处是，类型 `G ⧸ N` 实际上取决于 `N`（直到定义上的等价性），因此仅仅证明两个正规子群 `N` 和 `M` 相等并不足以使相应的商相等。然而，普遍性质在这种情况下确实给出了一个同构。

```py
example  {G  :  Type*}  [Group  G]  {M  N  :  Subgroup  G}  [M.Normal]
  [N.Normal]  (h  :  M  =  N)  :  G  ⧸  M  ≃*  G  ⧸  N  :=  QuotientGroup.quotientMulEquivOfEq  h 
```

作为本节的最后一系列练习，我们将证明如果 `H` 和 `K` 是有限群 `G` 的不相交正规子群，且它们的基数乘积等于 `G` 的基数，那么 `G` 与 `H × K` 同构。回忆一下，这里的“不相交”意味着 `H ⊓ K = ⊥`。

我们首先玩一点拉格朗日引理，而不假设子群是正规的或不相交的。

```py
section
variable  {G  :  Type*}  [Group  G]  {H  K  :  Subgroup  G}

open  MonoidHom

#check  Nat.card_pos  -- The nonempty argument will be automatically inferred for subgroups
#check  Subgroup.index_eq_card
#check  Subgroup.index_mul_card
#check  Nat.eq_of_mul_eq_mul_right

lemma  aux_card_eq  [Finite  G]  (h'  :  Nat.card  G  =  Nat.card  H  *  Nat.card  K)  :
  Nat.card  (G  ⧸  H)  =  Nat.card  K  :=  by
  sorry 
```

从现在开始，我们假设我们的子群是正规的且不相交的，并且假设基数条件。现在我们构建所需同构的第一个构建块。

```py
variable  [H.Normal]  [K.Normal]  [Fintype  G]  (h  :  Disjoint  H  K)
  (h'  :  Nat.card  G  =  Nat.card  H  *  Nat.card  K)

#check  Nat.bijective_iff_injective_and_card
#check  ker_eq_bot_iff
#check  restrict
#check  ker_restrict

def  iso₁  :  K  ≃*  G  ⧸  H  :=  by
  sorry 
```

现在我们可以定义我们的第二个构建块。我们需要 `MonoidHom.prod`，它从 `G₀` 到 `G₁ × G₂` 构建一个形态，其中包含从 `G₀` 到 `G₁` 和 `G₂` 的形态。

```py
def  iso₂  :  G  ≃*  (G  ⧸  K)  ×  (G  ⧸  H)  :=  by
  sorry 
```

我们现在准备将所有这些部分组合在一起。

```py
#check  MulEquiv.prodCongr

def  finalIso  :  G  ≃*  H  ×  K  :=
  sorry 
```  ## 9.2. 环

### 9.2.1. 环、它们的单位、形态和子环

类型 `R` 上的环结构类型是 `Ring R`。假设乘法是交换的变体是 `CommRing R`。我们已经看到，`ring` 策略将证明任何从交换环公理中得出的等价性。

```py
example  {R  :  Type*}  [CommRing  R]  (x  y  :  R)  :  (x  +  y)  ^  2  =  x  ^  2  +  y  ^  2  +  2  *  x  *  y  :=  by  ring 
```

更奇特的变体不需要 `R` 上的加法形成一个群，而只是一个加法幺半群。相应的类型类是 `Semiring R` 和 `CommSemiring R`。自然数的类型是 `CommSemiring R` 的重要实例，任何取自然数值的函数类型也是如此。另一个重要例子是环中的理想类型，将在下面讨论。`ring` 策略的名称具有双重误导性，因为它假设交换性，但在半环中也能工作。换句话说，它适用于任何 `CommSemiring`。

```py
example  (x  y  :  ℕ)  :  (x  +  y)  ^  2  =  x  ^  2  +  y  ^  2  +  2  *  x  *  y  :=  by  ring 
```

环和半环类也有不假设乘法存在乘法单位或结合律的版本。我们在这里不会讨论那些。

一些在环论导论中传统上教授的概念实际上是关于底层的乘法幺半群。一个突出的例子是环的单位定义。每个（乘法）幺半群 `M` 都有一个谓词 `IsUnit : M → Prop`，断言存在一个两边的逆元，一个单位类型 `Units M`，符号为 `Mˣ`，以及一个到 `M` 的强制转换。类型 `Units M` 将可逆元素与其逆元捆绑在一起，以及确保每个确实是另一个的逆元的属性。这个实现细节主要在定义可计算函数时相关。在大多数情况下，可以使用 `IsUnit.unit {x : M} : IsUnit x → Mˣ` 来构建一个单位。在交换情况下，还有一个 `Units.mkOfMulEqOne (x y : M) : x * y = 1 → Mˣ`，它构建了被视为单位的 `x`。

```py
example  (x  :  ℤˣ)  :  x  =  1  ∨  x  =  -1  :=  Int.units_eq_one_or  x

example  {M  :  Type*}  [Monoid  M]  (x  :  Mˣ)  :  (x  :  M)  *  x⁻¹  =  1  :=  Units.mul_inv  x

example  {M  :  Type*}  [Monoid  M]  :  Group  Mˣ  :=  inferInstance 
```

两个（半）环 `R` 和 `S` 之间的环同态类型是 `RingHom R S`，符号为 `R →+* S`。

```py
example  {R  S  :  Type*}  [Ring  R]  [Ring  S]  (f  :  R  →+*  S)  (x  y  :  R)  :
  f  (x  +  y)  =  f  x  +  f  y  :=  f.map_add  x  y

example  {R  S  :  Type*}  [Ring  R]  [Ring  S]  (f  :  R  →+*  S)  :  Rˣ  →*  Sˣ  :=
  Units.map  f 
```

同构变体是 `RingEquiv`，符号为 `≃+*`。

与子幺半群和子群一样，存在一个 `Subring R` 类型，用于表示环 `R` 的子环，但这个类型比子群的类型要少用得多，因为不能通过子环来商环。

```py
example  {R  :  Type*}  [Ring  R]  (S  :  Subring  R)  :  Ring  S  :=  inferInstance 
```

还要注意，`RingHom.range` 产生一个子环。

### 9.2.2\. 理想与商

由于历史原因，Mathlib 只为交换环提供了一个理想理论。（环库最初是为了快速推进现代代数几何的基础而开发的。）因此，在本节中，我们将使用交换（半）环。`R` 的理想被定义为将 `R` 视为 `R`-模的子模。模块将在线性代数章节中稍后讨论，但这个实现细节可以大部分安全忽略，因为大多数（但不是所有）相关引理都在理想的特殊上下文中重新表述。但匿名投影符号并不总是按预期工作。例如，在下面的片段中不能将 `Ideal.Quotient.mk I` 替换为 `I.Quotient.mk`，因为有两个 `.`，所以它将被解析为 `(Ideal.Quotient I).mk`；但 `Ideal.Quotient` 本身并不存在。

```py
example  {R  :  Type*}  [CommRing  R]  (I  :  Ideal  R)  :  R  →+*  R  ⧸  I  :=
  Ideal.Quotient.mk  I

example  {R  :  Type*}  [CommRing  R]  {a  :  R}  {I  :  Ideal  R}  :
  Ideal.Quotient.mk  I  a  =  0  ↔  a  ∈  I  :=
  Ideal.Quotient.eq_zero_iff_mem 
```

商环的通用性质是 `Ideal.Quotient.lift`。

```py
example  {R  S  :  Type*}  [CommRing  R]  [CommRing  S]  (I  :  Ideal  R)  (f  :  R  →+*  S)
  (H  :  I  ≤  RingHom.ker  f)  :  R  ⧸  I  →+*  S  :=
  Ideal.Quotient.lift  I  f  H 
```

尤其是它导致了环的第一同构定理。

```py
example  {R  S  :  Type*}  [CommRing  R]  CommRing  S  :
  R  ⧸  RingHom.ker  f  ≃+*  f.range  :=
  RingHom.quotientKerEquivRange  f 
```

理想在包含关系下形成一个完备格结构，以及一个半环结构。这两个结构相互作用得很好。

```py
variable  {R  :  Type*}  [CommRing  R]  {I  J  :  Ideal  R}

example  :  I  +  J  =  I  ⊔  J  :=  rfl

example  {x  :  R}  :  x  ∈  I  +  J  ↔  ∃  a  ∈  I,  ∃  b  ∈  J,  a  +  b  =  x  :=  by
  simp  [Submodule.mem_sup]

example  :  I  *  J  ≤  J  :=  Ideal.mul_le_left

example  :  I  *  J  ≤  I  :=  Ideal.mul_le_right

example  :  I  *  J  ≤  I  ⊓  J  :=  Ideal.mul_le_inf 
```

可以使用环同态通过 `Ideal.map` 推理想前移，并通过 `Ideal.comap` 拉回理想。通常，后者更方便使用，因为它不涉及存在量词。这解释了为什么它被用来陈述允许我们在商环之间构建同态的条件。

```py
example  {R  S  :  Type*}  [CommRing  R]  [CommRing  S]  (I  :  Ideal  R)  (J  :  Ideal  S)  (f  :  R  →+*  S)
  (H  :  I  ≤  Ideal.comap  f  J)  :  R  ⧸  I  →+*  S  ⧸  J  :=
  Ideal.quotientMap  J  f  H 
```

一个微妙之处在于类型 `R ⧸ I` 实际上依赖于 `I`（直到定义等价），因此仅仅证明两个理想 `I` 和 `J` 相等并不足以使相应的商相等。然而，普遍性质在这种情况下确实提供了一个同构。

```py
example  {R  :  Type*}  [CommRing  R]  {I  J  :  Ideal  R}  (h  :  I  =  J)  :  R  ⧸  I  ≃+*  R  ⧸  J  :=
  Ideal.quotEquivOfEq  h 
```

我们现在可以以中国剩余同构为例来展示。请注意，索引下确界符号 `⨅` 和类型的大乘积符号 `Π` 之间的区别。根据你的字体，这些可能很难区分。

```py
example  {R  :  Type*}  [CommRing  R]  {ι  :  Type*}  [Fintype  ι]  (f  :  ι  →  Ideal  R)
  (hf  :  ∀  i  j,  i  ≠  j  →  IsCoprime  (f  i)  (f  j))  :  (R  ⧸  ⨅  i,  f  i)  ≃+*  Π  i,  R  ⧸  f  i  :=
  Ideal.quotientInfRingEquivPiQuotient  f  hf 
```

中国剩余定理的初等版本，一个关于 `ZMod` 的陈述，可以很容易地从先前的定理中推导出来：

```py
open  BigOperators  PiNotation

example  {ι  :  Type*}  [Fintype  ι]  (a  :  ι  →  ℕ)  (coprime  :  ∀  i  j,  i  ≠  j  →  (a  i).Coprime  (a  j))  :
  ZMod  (∏  i,  a  i)  ≃+*  Π  i,  ZMod  (a  i)  :=
  ZMod.prodEquivPi  a  coprime 
```

作为一系列练习，我们将重新证明一般情况下的中国剩余定理。

我们首先需要定义定理中出现的映射，作为一个环同态，使用商环的普遍性质。

```py
variable  {ι  R  :  Type*}  [CommRing  R]
open  Ideal  Quotient  Function

#check  Pi.ringHom
#check  ker_Pi_Quotient_mk

/-- The homomorphism from ``R ⧸ ⨅ i, I i`` to ``Π i, R ⧸ I i`` featured in the Chinese
 Remainder Theorem. -/
def  chineseMap  (I  :  ι  →  Ideal  R)  :  (R  ⧸  ⨅  i,  I  i)  →+*  Π  i,  R  ⧸  I  i  :=
  sorry 
```

确保以下两个引理可以通过 `rfl` 证明。

```py
lemma  chineseMap_mk  (I  :  ι  →  Ideal  R)  (x  :  R)  :
  chineseMap  I  (Quotient.mk  _  x)  =  fun  i  :  ι  ↦  Ideal.Quotient.mk  (I  i)  x  :=
  sorry

lemma  chineseMap_mk'  (I  :  ι  →  Ideal  R)  (x  :  R)  (i  :  ι)  :
  chineseMap  I  (mk  _  x)  i  =  mk  (I  i)  x  :=
  sorry 
```

下一个引理证明了中国剩余定理的简单一半，对理想的集合没有任何假设。证明不到一行长。

```py
#check  injective_lift_iff

lemma  chineseMap_inj  (I  :  ι  →  Ideal  R)  :  Injective  (chineseMap  I)  :=  by
  sorry 
```

我们现在准备好展示定理的核心，这将证明我们的 `chineseMap` 的满射性。首先我们需要知道表达互质（也称为共最大性假设）的不同方式。下面只需要前两种。

```py
#check  IsCoprime
#check  isCoprime_iff_add
#check  isCoprime_iff_exists
#check  isCoprime_iff_sup_eq
#check  isCoprime_iff_codisjoint 
```

我们利用归纳法于 `Finset`。下面给出了 `Finset` 的相关引理。记住 `ring` 策略适用于半环，并且环的理想形成一个半环。

```py
#check  Finset.mem_insert_of_mem
#check  Finset.mem_insert_self

theorem  isCoprime_Inf  {I  :  Ideal  R}  {J  :  ι  →  Ideal  R}  {s  :  Finset  ι}
  (hf  :  ∀  j  ∈  s,  IsCoprime  I  (J  j))  :  IsCoprime  I  (⨅  j  ∈  s,  J  j)  :=  by
  classical
  simp_rw  [isCoprime_iff_add]  at  *
  induction  s  using  Finset.induction  with
  |  empty  =>
  simp
  |  @insert  i  s  _  hs  =>
  rw  [Finset.iInf_insert,  inf_comm,  one_eq_top,  eq_top_iff,  ←  one_eq_top]
  set  K  :=  ⨅  j  ∈  s,  J  j
  calc
  1  =  I  +  K  :=  sorry
  _  =  I  +  K  *  (I  +  J  i)  :=  sorry
  _  =  (1  +  K)  *  I  +  K  *  J  i  :=  sorry
  _  ≤  I  +  K  ⊓  J  i  :=  sorry 
```

我们现在可以证明中国剩余定理中出现的映射的满射性。

```py
lemma  chineseMap_surj  [Fintype  ι]  {I  :  ι  →  Ideal  R}
  (hI  :  ∀  i  j,  i  ≠  j  →  IsCoprime  (I  i)  (I  j))  :  Surjective  (chineseMap  I)  :=  by
  classical
  intro  g
  choose  f  hf  using  fun  i  ↦  Ideal.Quotient.mk_surjective  (g  i)
  have  key  :  ∀  i,  ∃  e  :  R,  mk  (I  i)  e  =  1  ∧  ∀  j,  j  ≠  i  →  mk  (I  j)  e  =  0  :=  by
  intro  i
  have  hI'  :  ∀  j  ∈  ({i}  :  Finset  ι)ᶜ,  IsCoprime  (I  i)  (I  j)  :=  by
  sorry
  sorry
  choose  e  he  using  key
  use  mk  _  (∑  i,  f  i  *  e  i)
  sorry 
```

现在所有的部分都在以下内容中汇集在一起：

```py
noncomputable  def  chineseIso  [Fintype  ι]  (f  :  ι  →  Ideal  R)
  (hf  :  ∀  i  j,  i  ≠  j  →  IsCoprime  (f  i)  (f  j))  :  (R  ⧸  ⨅  i,  f  i)  ≃+*  Π  i,  R  ⧸  f  i  :=
  {  Equiv.ofBijective  _  ⟨chineseMap_inj  f,  chineseMap_surj  hf⟩,
  chineseMap  f  with  } 
```

### 9.2.3. 代数和多项式

给定一个交换（半）环 `R`，一个 `R` 上的代数是一个半环 `A`，它配备了一个环同态，其像与 `A` 的每个元素交换。这被编码为类型类 `Algebra R A`。从 `R` 到 `A` 的映射被称为结构映射，在 Lean 中表示为 `algebraMap R A : R →+* A`。对于某个 `r : R`，`a : A` 通过 `algebraMap R A r` 的乘法称为 `a` 通过 `r` 的标量乘法，表示为 `r • a`。请注意，这种代数的概念有时被称为 *结合有单位代数*，以强调存在更一般的代数概念。

`algebraMap R A` 是环同态的事实将标量乘法的许多性质打包在一起，如下所示：

```py
example  {R  A  :  Type*}  [CommRing  R]  [Ring  A]  [Algebra  R  A]  (r  r'  :  R)  (a  :  A)  :
  (r  +  r')  •  a  =  r  •  a  +  r'  •  a  :=
  add_smul  r  r'  a

example  {R  A  :  Type*}  [CommRing  R]  [Ring  A]  [Algebra  R  A]  (r  r'  :  R)  (a  :  A)  :
  (r  *  r')  •  a  =  r  •  r'  •  a  :=
  mul_smul  r  r'  a 
```

两个 `R`-代数 `A` 和 `B` 之间的映射是与 `R` 的元素进行标量乘法交换的环同态。它们被捆绑为具有类型 `AlgHom R A B` 的映射，表示为 `A →ₐ[R] B`。

非交换代数的重要例子包括自同态代数和方阵代数，这两者都将在线性代数章节中介绍。在本章中，我们将讨论一个最重要的交换代数例子，即多项式代数。

单变量多项式代数，其系数在 `R` 中，被称为 `Polynomial R`，一旦打开 `Polynomial` 命名空间，就可以写成 `R[X]`。从 `R` 到 `R[X]` 的代数结构映射用 `C` 表示，它代表“常数”，因为相应的多项式函数始终是常数。不定元用 `X` 表示。

```py
open  Polynomial

example  {R  :  Type*}  [CommRing  R]  :  R[X]  :=  X

example  {R  :  Type*}  [CommRing  R]  (r  :  R)  :=  X  -  C  r 
```

在上面的第一个例子中，我们给 Lean 提供预期的类型是至关重要的，因为类型不能从定义的主体中确定。在第二个例子中，目标多项式代数可以通过我们使用 `C r` 来推断，因为 `r` 的类型是已知的。

因为 `C` 是从 `R` 到 `R[X]` 的环同态，我们可以在环 `R[X]` 中计算之前使用所有环同态引理，例如 `map_zero`、`map_one`、`map_mul` 和 `map_pow`。例如：

```py
example  {R  :  Type*}  [CommRing  R]  (r  :  R)  :  (X  +  C  r)  *  (X  -  C  r)  =  X  ^  2  -  C  (r  ^  2)  :=  by
  rw  [C.map_pow]
  ring 
```

您可以使用 `Polynomial.coeff` 访问系数。

```py
example  {R  :  Type*}  [CommRing  R]  (r:R)  :  (C  r).coeff  0  =  r  :=  by  simp

example  {R  :  Type*}  [CommRing  R]  :  (X  ^  2  +  2  *  X  +  C  3  :  R[X]).coeff  1  =  2  :=  by  simp 
```

定义多项式的度数总是很棘手，因为零多项式的特殊情况。Mathlib 有两种变体：`Polynomial.natDegree : R[X] → ℕ` 将零多项式的度数分配为 `0`，而 `Polynomial.degree : R[X] → WithBot ℕ` 分配 `⊥`。在后一种情况下，`WithBot ℕ` 可以看作是 `ℕ ∪ {-∞}`，除了 `-∞` 用 `⊥` 表示，它与完备格中的底元素具有相同的符号。这个特殊值用作零多项式的度数，并且对于加法是吸收的。（对于乘法几乎也是吸收的，除了 `⊥ * 0 = 0`。）

从道德上讲，`degree` 版本是正确的。例如，它允许我们陈述乘积度数的预期公式（假设基环没有零因子）。

```py
example  {R  :  Type*}  [Semiring  R]  [NoZeroDivisors  R]  {p  q  :  R[X]}  :
  degree  (p  *  q)  =  degree  p  +  degree  q  :=
  Polynomial.degree_mul 
```

而 `natDegree` 的版本需要假设非零多项式。

```py
example  {R  :  Type*}  [Semiring  R]  [NoZeroDivisors  R]  {p  q  :  R[X]}  (hp  :  p  ≠  0)  (hq  :  q  ≠  0)  :
  natDegree  (p  *  q)  =  natDegree  p  +  natDegree  q  :=
  Polynomial.natDegree_mul  hp  hq 
```

然而，`ℕ` 比使用 `WithBot ℕ` 更方便，因此 Mathlib 提供了这两种版本，并提供引理在它们之间进行转换。此外，`natDegree` 是在计算复合多项式的度数时更方便的定义。多项式的复合是 `Polynomial.comp`，我们有：

```py
example  {R  :  Type*}  [Semiring  R]  [NoZeroDivisors  R]  {p  q  :  R[X]}  :
  natDegree  (comp  p  q)  =  natDegree  p  *  natDegree  q  :=
  Polynomial.natDegree_comp 
```

多项式产生多项式函数：任何多项式都可以使用 `Polynomial.eval` 在 `R` 上进行评估。

```py
example  {R  :  Type*}  [CommRing  R]  (P:  R[X])  (x  :  R)  :=  P.eval  x

example  {R  :  Type*}  [CommRing  R]  (r  :  R)  :  (X  -  C  r).eval  r  =  0  :=  by  simp 
```

特别地，有一个谓词 `IsRoot`，它对 `R` 中的元素 `r` 成立，其中多项式为零。

```py
example  {R  :  Type*}  [CommRing  R]  (P  :  R[X])  (r  :  R)  :  IsRoot  P  r  ↔  P.eval  r  =  0  :=  Iff.rfl 
```

我们希望说，假设 `R` 没有零因子，一个多项式的根的数量最多与其次数相同，其中根是按重数计算的。但又一次，零多项式的情况很痛苦。因此，Mathlib 定义 `Polynomial.roots` 将多项式 `P` 映射到一个多重集，即如果 `P` 是零，则定义为空集，否则是 `P` 的根，带有重数。这个定义仅在基础环是域时才有效，因为否则定义没有好的性质。

```py
example  {R  :  Type*}  [CommRing  R]  [IsDomain  R]  (r  :  R)  :  (X  -  C  r).roots  =  {r}  :=
  roots_X_sub_C  r

example  {R  :  Type*}  [CommRing  R]  [IsDomain  R]  (r  :  R)  (n  :  ℕ):
  ((X  -  C  r)  ^  n).roots  =  n  •  {r}  :=
  by  simp 
```

`Polynomial.eval` 和 `Polynomial.roots` 只考虑系数环。它们不允许我们说 `X ^ 2 - 2 : ℚ[X]` 在 `ℝ` 中有一个根，或者 `X ^ 2 + 1 : ℝ[X]` 在 `ℂ` 中有一个根。为此，我们需要 `Polynomial.aeval`，它将在任何 `R`-代数中评估 `P : R[X]`。更精确地说，给定一个半环 `A` 和一个 `Algebra R A` 实例，`Polynomial.aeval` 将 `a` 的每个元素发送到评估在 `a` 上的 `R`-代数同态。由于 `AlgHom` 有一个到函数的强制转换，因此可以将它应用于多项式。但是 `aeval` 没有以多项式作为参数，因此不能使用如上 `P.eval` 中的点符号。

```py
example  :  aeval  Complex.I  (X  ^  2  +  1  :  ℝ[X])  =  0  :=  by  simp 
```

在这个上下文中对应于 `roots` 的函数是 `aroots`，它接受一个多项式然后是一个代数，并输出一个多重集（与 `roots` 相同的关于零多项式的警告）。

```py
open  Complex  Polynomial

example  :  aroots  (X  ^  2  +  1  :  ℝ[X])  ℂ  =  {Complex.I,  -I}  :=  by
  suffices  roots  (X  ^  2  +  1  :  ℂ[X])  =  {I,  -I}  by  simpa  [aroots_def]
  have  factored  :  (X  ^  2  +  1  :  ℂ[X])  =  (X  -  C  I)  *  (X  -  C  (-I))  :=  by
  have  key  :  (C  I  *  C  I  :  ℂ[X])  =  -1  :=  by  simp  [←  C_mul]
  rw  [C_neg]
  linear_combination  key
  have  p_ne_zero  :  (X  -  C  I)  *  (X  -  C  (-I))  ≠  0  :=  by
  intro  H
  apply_fun  eval  0  at  H
  simp  [eval]  at  H
  simp  only  [factored,  roots_mul  p_ne_zero,  roots_X_sub_C]
  rfl

-- Mathlib knows about D'Alembert-Gauss theorem: ``ℂ`` is algebraically closed.
example  :  IsAlgClosed  ℂ  :=  inferInstance 
```

更一般地，给定一个环同态 `f : R →+* S`，可以使用 `Polynomial.eval₂` 在 `S` 中的某一点评估 `P : R[X]`。这个操作产生了一个从 `R[X]` 到 `S` 的实际函数，因为它不假设存在一个 `Algebra R S` 实例，所以点符号的使用方式正如你所期望的那样。

```py
#check  (Complex.ofRealHom  :  ℝ  →+*  ℂ)

example  :  (X  ^  2  +  1  :  ℝ[X]).eval₂  Complex.ofRealHom  Complex.I  =  0  :=  by  simp 
```

最后，让我们简要地提一下多元多项式。给定一个交换半环 `R`，系数在 `R` 中且变量由类型 `σ` 索引的 `R`-代数多项式是 `MVPolynomial σ R`。给定 `i : σ`，相应的多项式是 `MvPolynomial.X i`。（如通常一样，可以打开 `MVPolynomial` 命名空间以缩短为 `X i`。）例如，如果我们想要两个变量，我们可以使用 `Fin 2` 作为 `σ`，并将定义单位圆的 $\mathbb{R}²$ 中的多项式写为：

```py
open  MvPolynomial

def  circleEquation  :  MvPolynomial  (Fin  2)  ℝ  :=  X  0  ^  2  +  X  1  ^  2  -  1 
```

回想一下，函数应用有很高的优先级，所以上面的表达式读作 `(X 0) ^ 2 + (X 1) ^ 2 - 1`。我们可以评估它以确保坐标为 $(1, 0)$ 的点在圆上。回想一下，`![...]` 符号表示 `Fin n → X` 的元素，其中 `n` 是由参数的数量决定的某个自然数，`X` 是由参数的类型决定的某个类型。

```py
example  :  MvPolynomial.eval  ![1,  0]  circleEquation  =  0  :=  by  simp  [circleEquation] 
```  ## 9.1. 单群与群

### 9.1.1. 单群及其同态

抽象代数课程通常从群开始，然后逐步过渡到环、域和向量空间。在讨论环上的乘法时，这涉及到一些扭曲，因为乘法运算并不来自群结构，但许多证明可以直接从群论转移到这个新环境。在用笔和纸做数学时，最常见的解决办法是将这些证明作为练习。一种不太高效但更安全、更符合形式化要求的做法是使用幺半群。类型 M 上的幺半群结构是一个内部结合律，它具有单位元。幺半群主要用于适应群和环的乘性结构。但也有一些自然例子；例如，自然数集加上加法运算形成一个幺半群。

从实际观点来看，在使用 Mathlib 时，你可以基本上忽略幺半群。但当你浏览 Mathlib 文件寻找引理时，你需要知道它们的存在。否则，你可能会在群理论文件中寻找一个实际上在幺半群中找到的陈述，因为它们不需要元素是可逆的。

类型 `M` 上的幺半群结构写作 `Monoid M`。函数 `Monoid` 是一个类型类，所以它几乎总是作为一个隐式参数实例（换句话说，在方括号中）出现。默认情况下，`Monoid` 使用乘法符号表示操作；对于加法符号，请使用 `AddMonoid`。这些结构的交换版本在 `Monoid` 前添加前缀 `Comm`。

```py
example  {M  :  Type*}  [Monoid  M]  (x  :  M)  :  x  *  1  =  x  :=  mul_one  x

example  {M  :  Type*}  [AddCommMonoid  M]  (x  y  :  M)  :  x  +  y  =  y  +  x  :=  add_comm  x  y 
```

注意，尽管 `AddMonoid` 在库中可以找到，但使用非交换操作的加法符号通常很令人困惑。

么半群 `M` 和 `N` 之间的同态类型称为 `MonoidHom M N`，并写作 `M →* N`。当我们将其应用于 `M` 的元素时，Lean 会自动将此类同态视为从 `M` 到 `N` 的函数。加法版本称为 `AddMonoidHom`，并写作 `M →+ N`。

```py
example  {M  N  :  Type*}  [Monoid  M]  [Monoid  N]  (x  y  :  M)  (f  :  M  →*  N)  :  f  (x  *  y)  =  f  x  *  f  y  :=
  f.map_mul  x  y

example  {M  N  :  Type*}  [AddMonoid  M]  [AddMonoid  N]  (f  :  M  →+  N)  :  f  0  =  0  :=
  f.map_zero 
```

这些同态是打包映射，即它们将一个映射及其一些属性打包在一起。记住，第 8.2 节 解释了打包映射；这里我们只是简单地指出一个不太幸运的后果，即我们不能使用普通函数复合来复合映射。相反，我们需要使用 `MonoidHom.comp` 和 `AddMonoidHom.comp`。

```py
example  {M  N  P  :  Type*}  [AddMonoid  M]  [AddMonoid  N]  [AddMonoid  P]
  (f  :  M  →+  N)  (g  :  N  →+  P)  :  M  →+  P  :=  g.comp  f 
```

### 9.1.2\. 群及其同态

我们将有很多关于群的内容要讲，群是具有额外性质（每个元素都有一个逆元）的幺半群。

```py
example  {G  :  Type*}  [Group  G]  (x  :  G)  :  x  *  x⁻¹  =  1  :=  mul_inv_cancel  x 
```

与我们之前看到的 `ring` 策略类似，存在一个 `group` 策略可以证明在任意群中成立的任何恒等式。（等价地，它也证明了在自由群中成立的恒等式。）

```py
example  {G  :  Type*}  [Group  G]  (x  y  z  :  G)  :  x  *  (y  *  z)  *  (x  *  z)⁻¹  *  (x  *  y  *  x⁻¹)⁻¹  =  1  :=  by
  group 
```

对于交换加法群中的恒等式，也存在一个名为 `abel` 的策略。

```py
example  {G  :  Type*}  [AddCommGroup  G]  (x  y  z  :  G)  :  z  +  x  +  (y  -  z  -  x)  =  y  :=  by
  abel 
```

有趣的是，群同态不过是一个群之间的单群同态。因此，我们可以复制并粘贴我们之前的一个例子，将`Monoid`替换为`Group`。

```py
example  {G  H  :  Type*}  [Group  G]  [Group  H]  (x  y  :  G)  (f  :  G  →*  H)  :  f  (x  *  y)  =  f  x  *  f  y  :=
  f.map_mul  x  y 
```

当然，我们确实得到了一些新的属性，例如这个：

```py
example  {G  H  :  Type*}  [Group  G]  [Group  H]  (x  :  G)  (f  :  G  →*  H)  :  f  (x⁻¹)  =  (f  x)⁻¹  :=
  f.map_inv  x 
```

你可能担心构建群同态会让我们做不必要的额外工作，因为单群同态的定义强制要求单位元被映射到单位元，而在群同态的情况下这是自动的。实际上，额外的努力并不困难，但为了避免这种额外的工作，有一个函数可以从群之间的兼容组合律的函数构建群同态。

```py
example  {G  H  :  Type*}  [Group  G]  [Group  H]  (f  :  G  →  H)  (h  :  ∀  x  y,  f  (x  *  y)  =  f  x  *  f  y)  :
  G  →*  H  :=
  MonoidHom.mk'  f  h 
```

此外，还有一个表示群（或单群）同构的类型`MulEquiv`，用`≃*`表示（在加法表示法中用`≃+`表示`AddEquiv`）。`f : G ≃* H`的逆是`MulEquiv.symm f : H ≃* G`，`f`和`g`的组合是`MulEquiv.trans f g`，单位同构是`MulEquiv.refl G`。使用匿名投影符号，前两个可以写成`f.symm`和`f.trans g`。当需要时，此类型中的元素会自动强制转换为形态和函数。

```py
example  {G  H  :  Type*}  [Group  G]  [Group  H]  (f  :  G  ≃*  H)  :
  f.trans  f.symm  =  MulEquiv.refl  G  :=
  f.self_trans_symm 
```

可以使用`MulEquiv.ofBijective`从双射同态构建同构。这样做会使逆函数不可计算。

```py
noncomputable  example  {G  H  :  Type*}  [Group  G]  [Group  H]
  (f  :  G  →*  H)  (h  :  Function.Bijective  f)  :
  G  ≃*  H  :=
  MulEquiv.ofBijective  f  h 
```

### 9.1.3. 子群

正如群同态是打包的，`G`的子群也是一个由`G`中的集合及其相关闭包性质组成的打包结构。

```py
example  {G  :  Type*}  [Group  G]  (H  :  Subgroup  G)  {x  y  :  G}  (hx  :  x  ∈  H)  (hy  :  y  ∈  H)  :
  x  *  y  ∈  H  :=
  H.mul_mem  hx  hy

example  {G  :  Type*}  [Group  G]  (H  :  Subgroup  G)  {x  :  G}  (hx  :  x  ∈  H)  :
  x⁻¹  ∈  H  :=
  H.inv_mem  hx 
```

在上面的例子中，重要的是要理解`Subgroup G`是`G`的子群类型，而不是一个谓词`IsSubgroup H`，其中`H`是`Set G`的一个元素。`Subgroup G`被赋予了到`Set G`的强制转换和一个在`G`上的成员谓词。参见第 8.3 节了解如何以及为什么这样做。

当然，如果两个子群具有相同的元素，那么它们就是相同的。这个事实被注册用于与`ext`策略一起使用，它可以用来证明两个子群是相同的，就像它被用来证明两个集合是相同的一样。

例如，为了陈述和证明`ℤ`是`ℚ`的加法子群，我们真正想要的是构建一个类型为`AddSubgroup ℚ`的项，其投影到`Set ℚ`是`ℤ`，或者更精确地说，是`ℤ`在`ℚ`中的像。

```py
example  :  AddSubgroup  ℚ  where
  carrier  :=  Set.range  ((↑)  :  ℤ  →  ℚ)
  add_mem'  :=  by
  rintro  _  _  ⟨n,  rfl⟩  ⟨m,  rfl⟩
  use  n  +  m
  simp
  zero_mem'  :=  by
  use  0
  simp
  neg_mem'  :=  by
  rintro  _  ⟨n,  rfl⟩
  use  -n
  simp 
```

使用类型类，Mathlib 知道群的一个子群继承了群结构。

```py
example  {G  :  Type*}  [Group  G]  (H  :  Subgroup  G)  :  Group  H  :=  inferInstance 
```

这个例子很微妙。对象`H`不是一个类型，但 Lean 会自动将其解释为`G`的子类型，并将其强制转换为类型。因此，上述例子可以更明确地重述为：

```py
example  {G  :  Type*}  [Group  G]  (H  :  Subgroup  G)  :  Group  {x  :  G  //  x  ∈  H}  :=  inferInstance 
```

拥有 `Subgroup G` 类型而不是谓词 `IsSubgroup : Set G → Prop` 的重要好处是，可以轻松地为 `Subgroup G` 赋予额外的结构。重要的是，它具有关于包含的完备格结构。例如，我们不是通过一个断言两个 `G` 的子群的交集仍然是子群的引理，而是使用了格运算 `⊓` 来构造交集。然后我们可以将关于格的任意引理应用于构造。

让我们验证两个子群的下确界所对应的集合确实，按照定义，是它们的交集。

```py
example  {G  :  Type*}  [Group  G]  (H  H'  :  Subgroup  G)  :
  ((H  ⊓  H'  :  Subgroup  G)  :  Set  G)  =  (H  :  Set  G)  ∩  (H'  :  Set  G)  :=  rfl 
```

对于底层集合交集的不同表示可能看起来很奇怪，但这个对应关系并不适用于上确界运算和集合的并集，因为子群的并集在一般情况下不是子群。相反，需要使用由并集生成的子群，这是通过 `Subgroup.closure` 实现的。

```py
example  {G  :  Type*}  [Group  G]  (H  H'  :  Subgroup  G)  :
  ((H  ⊔  H'  :  Subgroup  G)  :  Set  G)  =  Subgroup.closure  ((H  :  Set  G)  ∪  (H'  :  Set  G))  :=  by
  rw  [Subgroup.sup_eq_closure] 
```

另一个细微之处在于，`G` 本身并没有 `Subgroup G` 类型，因此我们需要一种方式来谈论 `G` 作为 `G` 的子群。这也由格结构提供：全子群是这个格的最高元素。

```py
example  {G  :  Type*}  [Group  G]  (x  :  G)  :  x  ∈  (⊤  :  Subgroup  G)  :=  trivial 
```

同样，这个格的底元素是只包含中性元素的子群。

```py
example  {G  :  Type*}  [Group  G]  (x  :  G)  :  x  ∈  (⊥  :  Subgroup  G)  ↔  x  =  1  :=  Subgroup.mem_bot 
```

作为操作群和子群的练习，你可以定义由环境群中的元素对子群进行共轭。

```py
def  conjugate  {G  :  Type*}  [Group  G]  (x  :  G)  (H  :  Subgroup  G)  :  Subgroup  G  where
  carrier  :=  {a  :  G  |  ∃  h,  h  ∈  H  ∧  a  =  x  *  h  *  x⁻¹}
  one_mem'  :=  by
  dsimp
  sorry
  inv_mem'  :=  by
  dsimp
  sorry
  mul_mem'  :=  by
  dsimp
  sorry 
```

将前两个主题结合起来，可以使用群同态来推进和拉回子群。在 Mathlib 中的命名约定是将这些操作称为 `map` 和 `comap`。这些不是常见的数学术语，但它们的优势是比“推进”和“直接像”更短。

```py
example  {G  H  :  Type*}  [Group  G]  [Group  H]  (G'  :  Subgroup  G)  (f  :  G  →*  H)  :  Subgroup  H  :=
  Subgroup.map  f  G'

example  {G  H  :  Type*}  [Group  G]  [Group  H]  (H'  :  Subgroup  H)  (f  :  G  →*  H)  :  Subgroup  G  :=
  Subgroup.comap  f  H'

#check  Subgroup.mem_map
#check  Subgroup.mem_comap 
```

特别地，在映射 `f` 下底子群的逆像是一个称为 `f` 的**核**的子群，而 `f` 的值域也是一个子群。

```py
example  {G  H  :  Type*}  [Group  G]  [Group  H]  (f  :  G  →*  H)  (g  :  G)  :
  g  ∈  MonoidHom.ker  f  ↔  f  g  =  1  :=
  f.mem_ker

example  {G  H  :  Type*}  [Group  G]  [Group  H]  (f  :  G  →*  H)  (h  :  H)  :
  h  ∈  MonoidHom.range  f  ↔  ∃  g  :  G,  f  g  =  h  :=
  f.mem_range 
```

作为操作群同态和子群练习，让我们证明一些基本性质。它们已经在 Mathlib 中得到证明，所以如果你想从这些练习中受益，不要太快地使用 `exact?`。

```py
section  exercises
variable  {G  H  :  Type*}  [Group  G]  [Group  H]

open  Subgroup

example  (φ  :  G  →*  H)  (S  T  :  Subgroup  H)  (hST  :  S  ≤  T)  :  comap  φ  S  ≤  comap  φ  T  :=  by
  sorry

example  (φ  :  G  →*  H)  (S  T  :  Subgroup  G)  (hST  :  S  ≤  T)  :  map  φ  S  ≤  map  φ  T  :=  by
  sorry

variable  {K  :  Type*}  [Group  K]

-- Remember you can use the `ext` tactic to prove an equality of subgroups.
example  (φ  :  G  →*  H)  (ψ  :  H  →*  K)  (U  :  Subgroup  K)  :
  comap  (ψ.comp  φ)  U  =  comap  φ  (comap  ψ  U)  :=  by
  sorry

-- Pushing a subgroup along one homomorphism and then another is equal to
-- pushing it forward along the composite of the homomorphisms.
example  (φ  :  G  →*  H)  (ψ  :  H  →*  K)  (S  :  Subgroup  G)  :
  map  (ψ.comp  φ)  S  =  map  ψ  (S.map  φ)  :=  by
  sorry

end  exercises 
```

让我们用两个非常经典的结果来完成对 Mathlib 中子群介绍的介绍。拉格朗日定理表明有限群子群的基数是群基数的约数。西罗第一定理是拉格朗日定理的一个著名的部分逆定理。

虽然 Mathlib 的这个角落部分是为了允许计算而设置的，但我们仍然可以使用以下 `open scoped` 命令告诉 Lean 使用非构造性逻辑。

```py
open  scoped  Classical

example  {G  :  Type*}  [Group  G]  (G'  :  Subgroup  G)  :  Nat.card  G'  ∣  Nat.card  G  :=
  ⟨G'.index,  mul_comm  G'.index  _  ▸  G'.index_mul_card.symm⟩

open  Subgroup

example  {G  :  Type*}  [Group  G]  [Finite  G]  (p  :  ℕ)  {n  :  ℕ}  [Fact  p.Prime]
  (hdvd  :  p  ^  n  ∣  Nat.card  G)  :  ∃  K  :  Subgroup  G,  Nat.card  K  =  p  ^  n  :=
  Sylow.exists_subgroup_card_pow_prime  p  hdvd 
```

接下来的两个练习推导出拉格朗日引理的一个推论。（这已经在 Mathlib 中，所以如果你想快速使用 `exact?`，不要这么做。）

```py
lemma  eq_bot_iff_card  {G  :  Type*}  [Group  G]  {H  :  Subgroup  G}  :
  H  =  ⊥  ↔  Nat.card  H  =  1  :=  by
  suffices  (∀  x  ∈  H,  x  =  1)  ↔  ∃  x  ∈  H,  ∀  a  ∈  H,  a  =  x  by
  simpa  [eq_bot_iff_forall,  Nat.card_eq_one_iff_exists]
  sorry

#check  card_dvd_of_le

lemma  inf_bot_of_coprime  {G  :  Type*}  [Group  G]  (H  K  :  Subgroup  G)
  (h  :  (Nat.card  H).Coprime  (Nat.card  K))  :  H  ⊓  K  =  ⊥  :=  by
  sorry 
```

### 9.1.4. 具体群

在 Mathlib 中也可以操作具体的群，尽管这通常比操作抽象理论更复杂。例如，给定任何类型 `X`，`X` 的排列群是 `Equiv.Perm X`。特别是对称群 $\mathfrak{S}_n$ 是 `Equiv.Perm (Fin n)`。可以对此群陈述抽象结果，例如，如果 `X` 是有限的，则 `Equiv.Perm X` 由循环生成。

```py
open  Equiv

example  {X  :  Type*}  [Finite  X]  :  Subgroup.closure  {σ  :  Perm  X  |  Perm.IsCycle  σ}  =  ⊤  :=
  Perm.closure_isCycle 
```

可以完全具体并计算循环的实际乘积。以下我们使用 `#simp` 命令，该命令在给定的表达式中调用 `simp` 策略。符号 `c[]` 用于定义循环排列。在示例中，结果是 `ℕ` 的排列。可以在第一个数字上使用类型注解，如 `(1 : Fin 5)`，使其成为 `Perm (Fin 5)` 中的计算。

```py
#simp  [mul_assoc]  c[1,  2,  3]  *  c[2,  3,  4] 
```

另一种处理具体群的方法是使用自由群和群表示。类型 `α` 上的自由群是 `FreeGroup α`，包含映射是 `FreeGroup.of : α → FreeGroup α`。例如，让我们定义一个有三个元素 `a`、`b` 和 `c` 的类型 `S`，以及相应自由群中的元素 `ab⁻¹`。

```py
section  FreeGroup

inductive  S  |  a  |  b  |  c

open  S

def  myElement  :  FreeGroup  S  :=  (.of  a)  *  (.of  b)⁻¹ 
```

注意，我们给出了定义的预期类型，这样 Lean 就知道 `.of` 表示 `FreeGroup.of`。

自由群的全称性质体现在 `FreeGroup.lift` 的等价性中。例如，让我们定义从 `FreeGroup S` 到 `Perm (Fin 5)` 的群同态，该同态将 `a` 映射到 `c[1, 2, 3]`，将 `b` 映射到 `c[2, 3, 1]`，将 `c` 映射到 `c[2, 3]`，

```py
def  myMorphism  :  FreeGroup  S  →*  Perm  (Fin  5)  :=
  FreeGroup.lift  fun  |  .a  =>  c[1,  2,  3]
  |  .b  =>  c[2,  3,  1]
  |  .c  =>  c[2,  3] 
```

作为最后一个具体的例子，让我们看看如何定义一个由单个元素生成且该元素的立方为 1 的群（因此该群将与 $\mathbb{Z}/3$ 同构）以及如何从该群到 `Perm (Fin 5)` 构建一个同态。

作为只有一个元素的类型，我们将使用 `Unit`，其唯一元素用 `()` 表示。`PresentedGroup` 函数接受一个关系集，即某个自由群的元素集，并返回一个群，该群是自由群除以由关系生成的正规子群的商。（我们将在 第 9.1.6 节 中看到如何处理更一般的商。）由于我们以某种方式将此隐藏在定义之后，我们使用 `deriving Group` 来强制在 `myGroup` 上创建群实例。

```py
def  myGroup  :=  PresentedGroup  {.of  ()  ^  3}  deriving  Group 
```

呈现群的全称性质确保可以从将关系映射到目标群中性元素的函数构建出该群的外部同态。因此我们需要这样的函数以及一个证明该条件成立的证明。然后我们可以将这个证明输入到 `PresentedGroup.toGroup` 中，以获得所需的群同态。

```py
def  myMap  :  Unit  →  Perm  (Fin  5)
|  ()  =>  c[1,  2,  3]

lemma  compat_myMap  :
  ∀  r  ∈  ({.of  ()  ^  3}  :  Set  (FreeGroup  Unit)),  FreeGroup.lift  myMap  r  =  1  :=  by
  rintro  _  rfl
  simp
  decide

def  myNewMorphism  :  myGroup  →*  Perm  (Fin  5)  :=  PresentedGroup.toGroup  compat_myMap

end  FreeGroup 
```

### 9.1.5. 群作用

群论与数学其他部分的一个重要交互方式是通过群作用的使用。一个群 `G` 对某些类型 `X` 的作用不过是 `G` 到 `Equiv.Perm X` 的一个同态。所以在某种意义上，群作用已经被之前的讨论所涵盖。但我们不希望携带这个同态；相反，我们希望尽可能由 Lean 自动推断出来。因此，我们有一个类型类来表示这一点，即 `MulAction G X`。这种设置的缺点是，在同一个类型上有多个相同群的多个作用需要一些扭曲，例如定义类型同义词，每个同义词都携带不同的类型类实例。

这使得我们特别可以使用 `g • x` 来表示群元素 `g` 对点 `x` 的作用。

```py
noncomputable  section  GroupActions

example  {G  X  :  Type*}  [Group  G]  [MulAction  G  X]  (g  g':  G)  (x  :  X)  :
  g  •  (g'  •  x)  =  (g  *  g')  •  x  :=
  (mul_smul  g  g'  x).symm 
```

还有另一种加法群版本，称为 `AddAction`，其中作用由 `+ᵥ` 表示。这在仿射空间的定义中得到了应用。

```py
example  {G  X  :  Type*}  [AddGroup  G]  [AddAction  G  X]  (g  g'  :  G)  (x  :  X)  :
  g  +ᵥ  (g'  +ᵥ  x)  =  (g  +  g')  +ᵥ  x  :=
  (add_vadd  g  g'  x).symm 
```

基础群同态被称为 `MulAction.toPermHom`。

```py
open  MulAction

example  {G  X  :  Type*}  [Group  G]  [MulAction  G  X]  :  G  →*  Equiv.Perm  X  :=
  toPermHom  G  X 
```

作为本节的练习，让我们看看如何定义任何群 `G` 到排列群 `Perm G` 的凯莱同构嵌入。

```py
def  CayleyIsoMorphism  (G  :  Type*)  [Group  G]  :  G  ≃*  (toPermHom  G  G).range  :=
  Equiv.Perm.subgroupOfMulAction  G  G 
```

注意，上述定义之前没有任何东西要求必须有群而不是幺半群（或任何带有乘法运算的类型）。

当我们想要将 `X` 划分为轨道时，群条件真正进入画面。`X` 上的对应等价关系被称为 `MulAction.orbitRel`。它没有被声明为全局实例。

```py
example  {G  X  :  Type*}  [Group  G]  [MulAction  G  X]  :  Setoid  X  :=  orbitRel  G  X 
```

使用这种方法，我们可以陈述 `X` 在 `G` 的作用下被划分为轨道。更精确地说，我们得到 `X` 与依赖积 `(ω : orbitRel.Quotient G X) × (orbit G (Quotient.out' ω))` 之间的双射，其中 `Quotient.out' ω` 简单地选择一个投影到 `ω` 的元素。回想一下，这个依赖积的元素是成对出现的 `⟨ω, x⟩`，其中 `x` 的类型 `orbit G (Quotient.out' ω)` 依赖于 `ω`。

```py
example  {G  X  :  Type*}  [Group  G]  [MulAction  G  X]  :
  X  ≃  (ω  :  orbitRel.Quotient  G  X)  ×  (orbit  G  (Quotient.out  ω))  :=
  MulAction.selfEquivSigmaOrbits  G  X 
```

特别地，当 `X` 是有限集时，这可以与 `Fintype.card_congr` 和 `Fintype.card_sigma` 结合，推导出 `X` 的基数是轨道基数的和。此外，轨道与通过左平移的稳定子群作用下的 `G` 的商一一对应。这种通过左平移的子群作用被用来定义子群下的群商，记作 /，因此我们可以使用以下简洁的陈述。

```py
example  {G  X  :  Type*}  [Group  G]  [MulAction  G  X]  (x  :  X)  :
  orbit  G  x  ≃  G  ⧸  stabilizer  G  x  :=
  MulAction.orbitEquivQuotientStabilizer  G  x 
```

将上述两个结果结合的重要特殊情况是当 `X` 是一个带有子群 `H` 通过平移作用的群 `G`。在这种情况下，所有稳定子群都是平凡的，因此每个轨道都与 `H` 一一对应，我们得到：

```py
example  {G  :  Type*}  [Group  G]  (H  :  Subgroup  G)  :  G  ≃  (G  ⧸  H)  ×  H  :=
  groupEquivQuotientProdSubgroup 
```

这是上述拉格朗日定理版本的概念变体。注意这个版本没有有限性的假设。

作为本节的练习，让我们通过使用之前练习中定义的 `conjugate` 来构建群在其子群上的共轭作用。

```py
variable  {G  :  Type*}  [Group  G]

lemma  conjugate_one  (H  :  Subgroup  G)  :  conjugate  1  H  =  H  :=  by
  sorry

instance  :  MulAction  G  (Subgroup  G)  where
  smul  :=  conjugate
  one_smul  :=  by
  sorry
  mul_smul  :=  by
  sorry

end  GroupActions 
```

### 9.1.6\. 商群

在上述关于子群在群上作用的讨论中，我们看到了商`G ⧸ H`的出现。一般来说，这只是一个类型。它可以赋予一个群结构，使得商映射是一个群同态，当且仅当`H`是一个正规子群（在这种情况下，这种群结构是唯一的）。

正规性假设是一个类型类`Subgroup.Normal`，这样类型类推理就可以用它来推导商群上的群结构。

```py
noncomputable  section  QuotientGroup

example  {G  :  Type*}  [Group  G]  (H  :  Subgroup  G)  [H.Normal]  :  Group  (G  ⧸  H)  :=  inferInstance

example  {G  :  Type*}  [Group  G]  (H  :  Subgroup  G)  [H.Normal]  :  G  →*  G  ⧸  H  :=
  QuotientGroup.mk'  H 
```

商群的全称性质通过`QuotientGroup.lift`访问：一个群同态`φ`一旦其核包含`N`，就会下降到`G ⧸ N`。

```py
example  {G  :  Type*}  [Group  G]  (N  :  Subgroup  G)  [N.Normal]  {M  :  Type*}
  [Group  M]  (φ  :  G  →*  M)  (h  :  N  ≤  MonoidHom.ker  φ)  :  G  ⧸  N  →*  M  :=
  QuotientGroup.lift  N  φ  h 
```

目标群被称为`M`的事实是上述代码片段的一个线索，表明在`M`上有一个单群结构就足够了。

一个重要的特殊情况是当`N = ker φ`时。在这种情况下，下降的同态是单射的，我们得到一个到其像上的群同构。这个结果通常被称为第一同构定理。

```py
example  {G  :  Type*}  [Group  G]  {M  :  Type*}  [Group  M]  (φ  :  G  →*  M)  :
  G  ⧸  MonoidHom.ker  φ  →*  MonoidHom.range  φ  :=
  QuotientGroup.quotientKerEquivRange  φ 
```

将全称性质应用于一个同态`φ : G →* G'`与商群投影`Quotient.mk' N'`的组合，我们也可以寻求一个从`G ⧸ N`到`G' ⧸ N'`的同态。对`φ`的要求通常表述为“`φ`应该将`N`放入`N'`内部。”但这等同于要求`φ`将`N'`拉回到`N`上，而后一种条件更容易处理，因为拉回的定义不涉及存在量词。

```py
example  {G  G':  Type*}  [Group  G]  [Group  G']
  {N  :  Subgroup  G}  [N.Normal]  {N'  :  Subgroup  G'}  [N'.Normal]
  {φ  :  G  →*  G'}  (h  :  N  ≤  Subgroup.comap  φ  N')  :  G  ⧸  N  →*  G'  ⧸  N':=
  QuotientGroup.map  N  N'  φ  h 
```

需要记住的一个微妙之处是，类型`G ⧸ N`实际上依赖于`N`（直到定义等价），所以有两个正规子群`N`和`M`相等的证明并不足以使相应的商相等。然而，全称性质在这种情况下确实给出了一个同构。

```py
example  {G  :  Type*}  [Group  G]  {M  N  :  Subgroup  G}  [M.Normal]
  [N.Normal]  (h  :  M  =  N)  :  G  ⧸  M  ≃*  G  ⧸  N  :=  QuotientGroup.quotientMulEquivOfEq  h 
```

作为本节最后的练习，我们将证明如果`H`和`K`是有限群`G`的互斥正规子群，并且它们的基数乘积等于`G`的基数，那么`G`同构于`H × K`。请记住，在这个上下文中，“互斥”意味着`H ⊓ K = ⊥`。

我们首先对拉格朗日引理进行一些探索，而不假设子群是正规的或互斥的。

```py
section
variable  {G  :  Type*}  [Group  G]  {H  K  :  Subgroup  G}

open  MonoidHom

#check  Nat.card_pos  -- The nonempty argument will be automatically inferred for subgroups
#check  Subgroup.index_eq_card
#check  Subgroup.index_mul_card
#check  Nat.eq_of_mul_eq_mul_right

lemma  aux_card_eq  [Finite  G]  (h'  :  Nat.card  G  =  Nat.card  H  *  Nat.card  K)  :
  Nat.card  (G  ⧸  H)  =  Nat.card  K  :=  by
  sorry 
```

从现在开始，我们假设我们的子群是正规的且互斥的，并且假设基数条件。现在我们构建所需同构的第一个构建块。

```py
variable  [H.Normal]  [K.Normal]  [Fintype  G]  (h  :  Disjoint  H  K)
  (h'  :  Nat.card  G  =  Nat.card  H  *  Nat.card  K)

#check  Nat.bijective_iff_injective_and_card
#check  ker_eq_bot_iff
#check  restrict
#check  ker_restrict

def  iso₁  :  K  ≃*  G  ⧸  H  :=  by
  sorry 
```

现在我们可以定义我们的第二个构建块。我们需要`MonoidHom.prod`，它从`G₀`到`G₁ × G₂`的映射中构建一个同态，这些映射是从`G₀`到`G₁`和`G₂`的映射。

```py
def  iso₂  :  G  ≃*  (G  ⧸  K)  ×  (G  ⧸  H)  :=  by
  sorry 
```

我们已经准备好将所有这些部分组合在一起。

```py
#check  MulEquiv.prodCongr

def  finalIso  :  G  ≃*  H  ×  K  :=
  sorry 
```  ### 9.1.1\. 单群及其同态

抽象代数课程通常从群开始，然后逐步过渡到环、域和向量空间。在讨论环上的乘法时，这涉及到一些扭曲，因为乘法运算并不来自群结构，但许多证明可以直接从群论转移到这个新环境。在用笔和纸做数学时，最常见的解决办法是将这些证明作为练习题。一种不太高效但更安全、更符合形式化要求的做法是使用幺半群。在类型 M 上的幺半群结构是一个内部组合律，它是结合的并且有一个中性元素。幺半群主要用于适应群和环的乘性结构。但也有一些自然例子；例如，自然数集加上加法运算形成一个幺半群。

从实际观点来看，在使用 Mathlib 时，你可以基本上忽略幺半群。但当你浏览 Mathlib 文件寻找引理时，你需要知道它们的存在。否则，你可能会在群理论文件中寻找一个陈述，而实际上它是在与幺半群一起找到的，因为它不需要元素是可逆的。

类型 `M` 上的幺半群结构类型写作 `Monoid M`。函数 `Monoid` 是一个类型类，所以它几乎总是作为一个隐式参数实例（换句话说，在方括号中）出现。默认情况下，`Monoid` 使用乘法符号进行操作；对于加法符号，请使用 `AddMonoid`。这些结构的交换版本在 `Monoid` 前添加前缀 `Comm`。

```py
example  {M  :  Type*}  [Monoid  M]  (x  :  M)  :  x  *  1  =  x  :=  mul_one  x

example  {M  :  Type*}  [AddCommMonoid  M]  (x  y  :  M)  :  x  +  y  =  y  +  x  :=  add_comm  x  y 
```

注意，尽管 `AddMonoid` 在库中可以找到，但使用非交换操作的加法符号通常很令人困惑。

么半群 `M` 和 `N` 之间的形态类型称为 `MonoidHom M N`，并写作 `M →* N`。当我们将其应用于 `M` 的元素时，Lean 会自动将这样的形态视为从 `M` 到 `N` 的函数。加法版本的形态类型称为 `AddMonoidHom`，并写作 `M →+ N`。

```py
example  {M  N  :  Type*}  [Monoid  M]  [Monoid  N]  (x  y  :  M)  (f  :  M  →*  N)  :  f  (x  *  y)  =  f  x  *  f  y  :=
  f.map_mul  x  y

example  {M  N  :  Type*}  [AddMonoid  M]  [AddMonoid  N]  (f  :  M  →+  N)  :  f  0  =  0  :=
  f.map_zero 
```

这些形态是打包映射，即它们将一个映射及其一些属性打包在一起。记住，第 8.2 节 解释了打包映射；这里我们只是简单地指出一个不太幸运的后果，即我们不能使用普通函数复合来复合映射。相反，我们需要使用 `MonoidHom.comp` 和 `AddMonoidHom.comp`。

```py
example  {M  N  P  :  Type*}  [AddMonoid  M]  [AddMonoid  N]  [AddMonoid  P]
  (f  :  M  →+  N)  (g  :  N  →+  P)  :  M  →+  P  :=  g.comp  f 
```

### 9.1.2. 群及其形态

我们将有很多关于群的内容要讲，群是具有每个元素都有逆元的额外属性的幺半群。

```py
example  {G  :  Type*}  [Group  G]  (x  :  G)  :  x  *  x⁻¹  =  1  :=  mul_inv_cancel  x 
```

与我们之前看到的 `ring` 策略类似，存在一个 `group` 策略可以证明任何在任意群中成立的恒等式。（等价地，它证明了在自由群中成立的恒等式。）

```py
example  {G  :  Type*}  [Group  G]  (x  y  z  :  G)  :  x  *  (y  *  z)  *  (x  *  z)⁻¹  *  (x  *  y  *  x⁻¹)⁻¹  =  1  :=  by
  group 
```

此外，还有一个用于交换加法群恒等式的策略，称为 `abel`。

```py
example  {G  :  Type*}  [AddCommGroup  G]  (x  y  z  :  G)  :  z  +  x  +  (y  -  z  -  x)  =  y  :=  by
  abel 
```

有趣的是，群同态不过是一个群之间的幺半群同态。因此，我们可以复制并粘贴我们早期的一个例子，将 `Monoid` 替换为 `Group`。

```py
example  {G  H  :  Type*}  [Group  G]  [Group  H]  (x  y  :  G)  (f  :  G  →*  H)  :  f  (x  *  y)  =  f  x  *  f  y  :=
  f.map_mul  x  y 
```

当然，我们确实得到了一些新的属性，例如这个：

```py
example  {G  H  :  Type*}  [Group  G]  [Group  H]  (x  :  G)  (f  :  G  →*  H)  :  f  (x⁻¹)  =  (f  x)⁻¹  :=
  f.map_inv  x 
```

你可能担心构建群同态将需要我们做不必要的额外工作，因为幺半群同态的定义强制要求中性元素被发送到中性元素，而在群同态的情况下这是自动的。在实践中，额外的努力并不困难，但为了避免这种情况，有一个函数可以从群之间的函数构建群同态，该函数与组合法则兼容。

```py
example  {G  H  :  Type*}  [Group  G]  [Group  H]  (f  :  G  →  H)  (h  :  ∀  x  y,  f  (x  *  y)  =  f  x  *  f  y)  :
  G  →*  H  :=
  MonoidHom.mk'  f  h 
```

此外，还有一个表示群（或幺半群）同构的类型 `MulEquiv`，用 `≃*` 表示（在加法表示法中用 `≃+` 表示 `AddEquiv`）。`f : G ≃* H` 的逆是 `MulEquiv.symm f : H ≃* G`，`f` 和 `g` 的复合是 `MulEquiv.trans f g`，而 `G` 的恒等同构是 `MulEquiv.refl G`。使用匿名投影符号，前两个可以分别写成 `f.symm` 和 `f.trans g`。当需要时，此类型中的元素会自动转换为形态和函数。

```py
example  {G  H  :  Type*}  [Group  G]  [Group  H]  (f  :  G  ≃*  H)  :
  f.trans  f.symm  =  MulEquiv.refl  G  :=
  f.self_trans_symm 
```

可以使用 `MulEquiv.ofBijective` 从双射形态构建同构。这样做会使逆函数不可计算。

```py
noncomputable  example  {G  H  :  Type*}  [Group  G]  [Group  H]
  (f  :  G  →*  H)  (h  :  Function.Bijective  f)  :
  G  ≃*  H  :=
  MulEquiv.ofBijective  f  h 
```

### 9.1.3. 子群

正如群同态是打包的，`G` 的一个子群也是一个打包的结构，由 `G` 中的一个集合及其相关的闭包性质组成。

```py
example  {G  :  Type*}  [Group  G]  (H  :  Subgroup  G)  {x  y  :  G}  (hx  :  x  ∈  H)  (hy  :  y  ∈  H)  :
  x  *  y  ∈  H  :=
  H.mul_mem  hx  hy

example  {G  :  Type*}  [Group  G]  (H  :  Subgroup  G)  {x  :  G}  (hx  :  x  ∈  H)  :
  x⁻¹  ∈  H  :=
  H.inv_mem  hx 
```

在上面的例子中，重要的是要理解 `Subgroup G` 是 `G` 的子群类型，而不是一个谓词 `IsSubgroup H`，其中 `H` 是 `Set G` 的一个元素。`Subgroup G` 被赋予了到 `Set G` 的强制转换和 `G` 上的成员谓词。参见 第 8.3 节 了解如何以及为什么这样做。

当然，如果两个子群具有相同的元素，则它们是相同的。这一事实被注册用于 `ext` 策略，该策略可以用来证明两个子群相等，就像它被用来证明两个集合相等一样。

要陈述和证明，例如，`ℤ` 是 `ℚ` 的加法子群，我们真正想要的是构造一个类型为 `AddSubgroup ℚ` 的项，其投影到 `Set ℚ` 是 `ℤ`，或者更精确地说，是 `ℤ` 在 `ℚ` 中的像。

```py
example  :  AddSubgroup  ℚ  where
  carrier  :=  Set.range  ((↑)  :  ℤ  →  ℚ)
  add_mem'  :=  by
  rintro  _  _  ⟨n,  rfl⟩  ⟨m,  rfl⟩
  use  n  +  m
  simp
  zero_mem'  :=  by
  use  0
  simp
  neg_mem'  :=  by
  rintro  _  ⟨n,  rfl⟩
  use  -n
  simp 
```

使用类型类，Mathlib 知道群的一个子群继承了群结构。

```py
example  {G  :  Type*}  [Group  G]  (H  :  Subgroup  G)  :  Group  H  :=  inferInstance 
```

这个例子很微妙。对象 `H` 不是一个类型，但 Lean 会自动将其解释为 `G` 的子类型，从而将其强制转换为类型。因此，上述例子可以更明确地重述为：

```py
example  {G  :  Type*}  [Group  G]  (H  :  Subgroup  G)  :  Group  {x  :  G  //  x  ∈  H}  :=  inferInstance 
```

拥有类型 `Subgroup G` 而不是谓词 `IsSubgroup : Set G → Prop` 的重要好处是，可以轻松地为 `Subgroup G` 赋予额外的结构。重要的是，它具有关于包含的完备格结构。例如，我们不是用命题说明 `G` 的两个子群的交集仍然是子群，而是使用了格运算 `⊓` 来构造交集。然后我们可以将关于格的任意引理应用于构造。

让我们检查两个子群下确界的底层集合确实，按照定义，是它们的交集。

```py
example  {G  :  Type*}  [Group  G]  (H  H'  :  Subgroup  G)  :
  ((H  ⊓  H'  :  Subgroup  G)  :  Set  G)  =  (H  :  Set  G)  ∩  (H'  :  Set  G)  :=  rfl 
```

对于底层集合交集的表示使用不同的符号可能看起来很奇怪，但这个对应关系并不适用于上确界运算和集合的并集，因为子群的并集在一般情况下不是子群。相反，需要使用由并集生成的子群，这可以通过 `Subgroup.closure` 实现。

```py
example  {G  :  Type*}  [Group  G]  (H  H'  :  Subgroup  G)  :
  ((H  ⊔  H'  :  Subgroup  G)  :  Set  G)  =  Subgroup.closure  ((H  :  Set  G)  ∪  (H'  :  Set  G))  :=  by
  rw  [Subgroup.sup_eq_closure] 
```

另一个细微之处在于，`G` 本身并没有类型 `Subgroup G`，因此我们需要一种方式来谈论 `G` 作为 `G` 的子群。这一点也由格结构提供：全子群是这个格的最高元素。

```py
example  {G  :  Type*}  [Group  G]  (x  :  G)  :  x  ∈  (⊤  :  Subgroup  G)  :=  trivial 
```

类似地，这个格的底部元素是只包含中性元素的子群。

```py
example  {G  :  Type*}  [Group  G]  (x  :  G)  :  x  ∈  (⊥  :  Subgroup  G)  ↔  x  =  1  :=  Subgroup.mem_bot 
```

作为操作群和子群的一个练习，你可以通过环境群中的一个元素定义子群的共轭。

```py
def  conjugate  {G  :  Type*}  [Group  G]  (x  :  G)  (H  :  Subgroup  G)  :  Subgroup  G  where
  carrier  :=  {a  :  G  |  ∃  h,  h  ∈  H  ∧  a  =  x  *  h  *  x⁻¹}
  one_mem'  :=  by
  dsimp
  sorry
  inv_mem'  :=  by
  dsimp
  sorry
  mul_mem'  :=  by
  dsimp
  sorry 
```

将前两个主题结合起来，可以使用群同态来推进和拉回子群。在 Mathlib 中的命名约定是将这些操作称为 `map` 和 `comap`。这些不是常见的数学术语，但它们的优势是比“推进”和“直接像”更短。

```py
example  {G  H  :  Type*}  [Group  G]  [Group  H]  (G'  :  Subgroup  G)  (f  :  G  →*  H)  :  Subgroup  H  :=
  Subgroup.map  f  G'

example  {G  H  :  Type*}  [Group  G]  [Group  H]  (H'  :  Subgroup  H)  (f  :  G  →*  H)  :  Subgroup  G  :=
  Subgroup.comap  f  H'

#check  Subgroup.mem_map
#check  Subgroup.mem_comap 
```

特别地，在映射 `f` 下底子群的逆像是称为 `f` 的 *核* 的子群，而 `f` 的值域也是一个子群。

```py
example  {G  H  :  Type*}  [Group  G]  [Group  H]  (f  :  G  →*  H)  (g  :  G)  :
  g  ∈  MonoidHom.ker  f  ↔  f  g  =  1  :=
  f.mem_ker

example  {G  H  :  Type*}  [Group  G]  [Group  H]  (f  :  G  →*  H)  (h  :  H)  :
  h  ∈  MonoidHom.range  f  ↔  ∃  g  :  G,  f  g  =  h  :=
  f.mem_range 
```

作为操作群同态和子群的练习，让我们证明一些基本性质。它们已经在 Mathlib 中得到证明，所以如果你想从这些练习中受益，不要太快地使用 `exact?`。

```py
section  exercises
variable  {G  H  :  Type*}  [Group  G]  [Group  H]

open  Subgroup

example  (φ  :  G  →*  H)  (S  T  :  Subgroup  H)  (hST  :  S  ≤  T)  :  comap  φ  S  ≤  comap  φ  T  :=  by
  sorry

example  (φ  :  G  →*  H)  (S  T  :  Subgroup  G)  (hST  :  S  ≤  T)  :  map  φ  S  ≤  map  φ  T  :=  by
  sorry

variable  {K  :  Type*}  [Group  K]

-- Remember you can use the `ext` tactic to prove an equality of subgroups.
example  (φ  :  G  →*  H)  (ψ  :  H  →*  K)  (U  :  Subgroup  K)  :
  comap  (ψ.comp  φ)  U  =  comap  φ  (comap  ψ  U)  :=  by
  sorry

-- Pushing a subgroup along one homomorphism and then another is equal to
-- pushing it forward along the composite of the homomorphisms.
example  (φ  :  G  →*  H)  (ψ  :  H  →*  K)  (S  :  Subgroup  G)  :
  map  (ψ.comp  φ)  S  =  map  ψ  (S.map  φ)  :=  by
  sorry

end  exercises 
```

让我们用两个非常经典的结果来完成对 Mathlib 中子群的介绍。拉格朗日定理表明有限群子群的基数是群基数的约数。西罗第一定理是拉格朗日定理的一个著名的部分逆定理。

虽然 Mathlib 的这个角落部分是为了允许计算而设置的，但我们可以使用以下 `open scoped` 命令告诉 Lean 使用非构造性逻辑。

```py
open  scoped  Classical

example  {G  :  Type*}  [Group  G]  (G'  :  Subgroup  G)  :  Nat.card  G'  ∣  Nat.card  G  :=
  ⟨G'.index,  mul_comm  G'.index  _  ▸  G'.index_mul_card.symm⟩

open  Subgroup

example  {G  :  Type*}  [Group  G]  [Finite  G]  (p  :  ℕ)  {n  :  ℕ}  [Fact  p.Prime]
  (hdvd  :  p  ^  n  ∣  Nat.card  G)  :  ∃  K  :  Subgroup  G,  Nat.card  K  =  p  ^  n  :=
  Sylow.exists_subgroup_card_pow_prime  p  hdvd 
```

接下来的两个练习推导出拉格朗日引理的一个推论。（这也在 Mathlib 中，所以如果你想从这些练习中受益，不要太快地使用 `exact?`。）

```py
lemma  eq_bot_iff_card  {G  :  Type*}  [Group  G]  {H  :  Subgroup  G}  :
  H  =  ⊥  ↔  Nat.card  H  =  1  :=  by
  suffices  (∀  x  ∈  H,  x  =  1)  ↔  ∃  x  ∈  H,  ∀  a  ∈  H,  a  =  x  by
  simpa  [eq_bot_iff_forall,  Nat.card_eq_one_iff_exists]
  sorry

#check  card_dvd_of_le

lemma  inf_bot_of_coprime  {G  :  Type*}  [Group  G]  (H  K  :  Subgroup  G)
  (h  :  (Nat.card  H).Coprime  (Nat.card  K))  :  H  ⊓  K  =  ⊥  :=  by
  sorry 
```

### 9.1.4. 具体群

也可以在 Mathlib 中操作具体群，尽管这通常比处理抽象理论更复杂。例如，给定任何类型 `X`，`X` 的排列群是 `Equiv.Perm X`。特别是，对称群 $\mathfrak{S}_n$ 是 `Equiv.Perm (Fin n)`。可以对此群陈述抽象结果，例如，如果 `X` 是有限的，则 `Equiv.Perm X` 由循环生成。

```py
open  Equiv

example  {X  :  Type*}  [Finite  X]  :  Subgroup.closure  {σ  :  Perm  X  |  Perm.IsCycle  σ}  =  ⊤  :=
  Perm.closure_isCycle 
```

可以完全具体，并计算循环的实际乘积。下面我们使用 `#simp` 命令，该命令在给定表达式中调用 `simp` 策略。符号 `c[]` 用于定义循环排列。在示例中，结果是 `ℕ` 的排列。可以在第一个数字上使用类型注解，如 `(1 : Fin 5)`，使其成为 `Perm (Fin 5)` 中的计算。

```py
#simp  [mul_assoc]  c[1,  2,  3]  *  c[2,  3,  4] 
```

另一种处理具体群的方法是使用自由群和群表示。类型 `α` 上的自由群是 `FreeGroup α`，其包含映射是 `FreeGroup.of : α → FreeGroup α`。例如，让我们定义一个有三个元素 `a`、`b` 和 `c` 的类型 `S`，以及对应自由群中的元素 `ab⁻¹`。

```py
section  FreeGroup

inductive  S  |  a  |  b  |  c

open  S

def  myElement  :  FreeGroup  S  :=  (.of  a)  *  (.of  b)⁻¹ 
```

注意，我们给出了定义的预期类型，这样 Lean 就知道 `.of` 表示 `FreeGroup.of`。

自由群的泛性质体现在等价 `FreeGroup.lift` 中。例如，让我们定义从 `FreeGroup S` 到 `Perm (Fin 5)` 的群同态，该同态将 `a` 映射到 `c[1, 2, 3]`，将 `b` 映射到 `c[2, 3, 1]`，将 `c` 映射到 `c[2, 3]`，

```py
def  myMorphism  :  FreeGroup  S  →*  Perm  (Fin  5)  :=
  FreeGroup.lift  fun  |  .a  =>  c[1,  2,  3]
  |  .b  =>  c[2,  3,  1]
  |  .c  =>  c[2,  3] 
```

作为最后一个具体的例子，让我们看看如何定义一个由单个元素生成且该元素的立方为 1 的群（因此该群将与 $\mathbb{Z}/3$ 同构）以及如何从该群到 `Perm (Fin 5)` 构建一个同态。

作为只有一个元素的类型，我们将使用 `Unit`，其唯一元素用 `()` 表示。函数 `PresentedGroup` 接受一组关系，即某个自由群的一组元素，并返回一个群，该群是自由群除以由关系生成的正规子群。（我们将在 第 9.1.6 节 中看到如何处理更一般的商。）由于我们以某种方式将此隐藏在定义之后，我们使用 `deriving Group` 来强制在 `myGroup` 上创建群实例。

```py
def  myGroup  :=  PresentedGroup  {.of  ()  ^  3}  deriving  Group 
```

呈现群的全称性质确保可以从将关系映射到目标群中性元素的函数构建出该群的外部同态。因此我们需要这样的函数和一个证明该条件成立。然后我们可以将这个证明输入到 `PresentedGroup.toGroup` 中，以获得所需的群同态。

```py
def  myMap  :  Unit  →  Perm  (Fin  5)
|  ()  =>  c[1,  2,  3]

lemma  compat_myMap  :
  ∀  r  ∈  ({.of  ()  ^  3}  :  Set  (FreeGroup  Unit)),  FreeGroup.lift  myMap  r  =  1  :=  by
  rintro  _  rfl
  simp
  decide

def  myNewMorphism  :  myGroup  →*  Perm  (Fin  5)  :=  PresentedGroup.toGroup  compat_myMap

end  FreeGroup 
```

### 9.1.5. 群作用

群论与数学其他部分的一个重要交互方式是通过群作用的使用。一个群 `G` 在某些类型 `X` 上的作用不过是 `G` 到 `Equiv.Perm X` 的一个同态。所以在某种意义上，群作用已经被之前的讨论所涵盖。但我们不希望携带这个同态；相反，我们希望尽可能多地由 Lean 自动推断它。因此，我们有一个类型类来表示这一点，即 `MulAction G X`。这种设置的缺点是，在同一个类型上有多个同一群的作用需要一些扭曲，例如定义类型同义词，每个同义词都携带不同的类型类实例。

这使我们能够特别地使用 `g • x` 来表示群元素 `g` 对点 `x` 的作用。

```py
noncomputable  section  GroupActions

example  {G  X  :  Type*}  [Group  G]  [MulAction  G  X]  (g  g':  G)  (x  :  X)  :
  g  •  (g'  •  x)  =  (g  *  g')  •  x  :=
  (mul_smul  g  g'  x).symm 
```

也有一个名为 `AddAction` 的加法群版本，其中动作由 `+ᵥ` 表示。这例如用于仿射空间的定义中。

```py
example  {G  X  :  Type*}  [AddGroup  G]  [AddAction  G  X]  (g  g'  :  G)  (x  :  X)  :
  g  +ᵥ  (g'  +ᵥ  x)  =  (g  +  g')  +ᵥ  x  :=
  (add_vadd  g  g'  x).symm 
```

基础群同态被称为 `MulAction.toPermHom`。

```py
open  MulAction

example  {G  X  :  Type*}  [Group  G]  [MulAction  G  X]  :  G  →*  Equiv.Perm  X  :=
  toPermHom  G  X 
```

作为说明，让我们看看如何定义任何群 `G` 到排列群 `Perm G` 的凯莱同构嵌入。

```py
def  CayleyIsoMorphism  (G  :  Type*)  [Group  G]  :  G  ≃*  (toPermHom  G  G).range  :=
  Equiv.Perm.subgroupOfMulAction  G  G 
```

注意，在上述定义之前，并不需要有一个群而不是幺半群（或任何带有乘法运算的类型）。

当我们想要将 `X` 划分为轨道时，群条件真正进入画面。`X` 上的对应等价关系被称为 `MulAction.orbitRel`。它没有被声明为一个全局实例。

```py
example  {G  X  :  Type*}  [Group  G]  [MulAction  G  X]  :  Setoid  X  :=  orbitRel  G  X 
```

使用这个结果，我们可以陈述 `X` 在 `G` 的作用下被划分为轨道。更精确地说，我们得到 `X` 与依赖积 `(ω : orbitRel.Quotient G X) × (orbit G (Quotient.out' ω))` 之间的双射，其中 `Quotient.out' ω` 简单地选择一个投影到 `ω` 的元素。回想一下，这个依赖积的元素是 `⟨ω, x⟩` 对，其中 `x` 的类型 `orbit G (Quotient.out' ω)` 依赖于 `ω`。

```py
example  {G  X  :  Type*}  [Group  G]  [MulAction  G  X]  :
  X  ≃  (ω  :  orbitRel.Quotient  G  X)  ×  (orbit  G  (Quotient.out  ω))  :=
  MulAction.selfEquivSigmaOrbits  G  X 
```

特别地，当 `X` 是有限集时，这可以与 `Fintype.card_congr` 和 `Fintype.card_sigma` 结合，推导出 `X` 的基数是轨道基数的和。此外，轨道与通过左平移稳定子群作用下的 `G` 的商一一对应。这种子群通过左平移的作用被用来定义群通过子群的商，记作 /，因此我们可以使用以下简洁的陈述。

```py
example  {G  X  :  Type*}  [Group  G]  [MulAction  G  X]  (x  :  X)  :
  orbit  G  x  ≃  G  ⧸  stabilizer  G  x  :=
  MulAction.orbitEquivQuotientStabilizer  G  x 
```

将上述两个结果结合的重要特殊情况是，当 `X` 是一个带有子群 `H` 通过平移作用的群 `G` 时。在这种情况下，所有稳定子群都是平凡的，因此每个轨道都与 `H` 一一对应，我们得到：

```py
example  {G  :  Type*}  [Group  G]  (H  :  Subgroup  G)  :  G  ≃  (G  ⧸  H)  ×  H  :=
  groupEquivQuotientProdSubgroup 
```

这是上述拉格朗日定理版本的概念变体。注意这个版本没有有限性的假设。

作为本节的练习，让我们根据之前练习中定义的 `conjugate`，构建群对其子群通过共轭的作用。

```py
variable  {G  :  Type*}  [Group  G]

lemma  conjugate_one  (H  :  Subgroup  G)  :  conjugate  1  H  =  H  :=  by
  sorry

instance  :  MulAction  G  (Subgroup  G)  where
  smul  :=  conjugate
  one_smul  :=  by
  sorry
  mul_smul  :=  by
  sorry

end  GroupActions 
```

### 9.1.6. 商群

在上述关于子群在群上作用的讨论中，我们看到了商 `G ⧸ H` 出现。在一般情况下，这只是一个类型。它可以赋予一个群结构，使得商映射是一个群同态，当且仅当 `H` 是正规子群（在这种情况下，这种群结构是唯一的）。

正规性假设是一个类型类 `Subgroup.Normal`，这样类型类推理就可以用它来推导商上的群结构。

```py
noncomputable  section  QuotientGroup

example  {G  :  Type*}  [Group  G]  (H  :  Subgroup  G)  [H.Normal]  :  Group  (G  ⧸  H)  :=  inferInstance

example  {G  :  Type*}  [Group  G]  (H  :  Subgroup  G)  [H.Normal]  :  G  →*  G  ⧸  H  :=
  QuotientGroup.mk'  H 
```

商群的泛性质通过 `QuotientGroup.lift` 访问：一个群同态 `φ` 只要其核包含 `N`，就会下降到 `G ⧸ N`。

```py
example  {G  :  Type*}  [Group  G]  (N  :  Subgroup  G)  [N.Normal]  {M  :  Type*}
  [Group  M]  (φ  :  G  →*  M)  (h  :  N  ≤  MonoidHom.ker  φ)  :  G  ⧸  N  →*  M  :=
  QuotientGroup.lift  N  φ  h 
```

目标群被称为 `M` 的这个事实是上述片段的一个线索，表明在 `M` 上有一个幺半群结构就足够了。

一个重要的特殊情况是当 `N = ker φ`。在这种情况下，下降的态射是单射的，并且我们得到一个到其像上的群同构。这个结果通常被称为第一同构定理。

```py
example  {G  :  Type*}  [Group  G]  {M  :  Type*}  [Group  M]  (φ  :  G  →*  M)  :
  G  ⧸  MonoidHom.ker  φ  →*  MonoidHom.range  φ  :=
  QuotientGroup.quotientKerEquivRange  φ 
```

将泛性质应用于一个态射 `φ : G →* G'` 与商群投影 `Quotient.mk' N'` 的组合，我们也可以试图得到一个从 `G ⧸ N` 到 `G' ⧸ N'` 的态射。对 `φ` 的要求通常表述为“`φ` 应该将 `N` 发送到 `N'` 内部。”但这等价于要求 `φ` 将 `N'` 拉回到 `N` 上，而后一种条件更容易处理，因为拉回的定义不涉及存在量词。

```py
example  {G  G':  Type*}  [Group  G]  [Group  G']
  {N  :  Subgroup  G}  [N.Normal]  {N'  :  Subgroup  G'}  [N'.Normal]
  {φ  :  G  →*  G'}  (h  :  N  ≤  Subgroup.comap  φ  N')  :  G  ⧸  N  →*  G'  ⧸  N':=
  QuotientGroup.map  N  N'  φ  h 
```

需要记住的一个微妙之处是，类型 `G ⧸ N` 实际上依赖于 `N`（直到定义等价），因此，两个正规子群 `N` 和 `M` 相等的证明不足以使相应的商相等。然而，泛性质在这种情况下确实给出了一个同构。

```py
example  {G  :  Type*}  [Group  G]  {M  N  :  Subgroup  G}  [M.Normal]
  [N.Normal]  (h  :  M  =  N)  :  G  ⧸  M  ≃*  G  ⧸  N  :=  QuotientGroup.quotientMulEquivOfEq  h 
```

作为本节最后的练习系列，我们将证明如果 `H` 和 `K` 是有限群 `G` 的不相交正规子群，且它们的基数乘积等于 `G` 的基数，那么 `G` 同构于 `H × K`。回忆一下，在这个上下文中，“不相交”意味着 `H ⊓ K = ⊥`。

我们首先通过不假设子群是正规或不相交的，来玩一点拉格朗日引理。

```py
section
variable  {G  :  Type*}  [Group  G]  {H  K  :  Subgroup  G}

open  MonoidHom

#check  Nat.card_pos  -- The nonempty argument will be automatically inferred for subgroups
#check  Subgroup.index_eq_card
#check  Subgroup.index_mul_card
#check  Nat.eq_of_mul_eq_mul_right

lemma  aux_card_eq  [Finite  G]  (h'  :  Nat.card  G  =  Nat.card  H  *  Nat.card  K)  :
  Nat.card  (G  ⧸  H)  =  Nat.card  K  :=  by
  sorry 
```

从现在开始，我们假设我们的子群是正规且不相交的，并假设基数条件。现在我们构建所需同构的第一个构建块。

```py
variable  [H.Normal]  [K.Normal]  [Fintype  G]  (h  :  Disjoint  H  K)
  (h'  :  Nat.card  G  =  Nat.card  H  *  Nat.card  K)

#check  Nat.bijective_iff_injective_and_card
#check  ker_eq_bot_iff
#check  restrict
#check  ker_restrict

def  iso₁  :  K  ≃*  G  ⧸  H  :=  by
  sorry 
```

现在我们可以定义我们的第二个构建块。我们需要 `MonoidHom.prod`，它从 `G₀` 到 `G₁ × G₂` 构建一个态射，其中包含从 `G₀` 到 `G₁` 和 `G₂` 的态射。

```py
def  iso₂  :  G  ≃*  (G  ⧸  K)  ×  (G  ⧸  H)  :=  by
  sorry 
```

我们已经准备好将所有部件组合在一起。

```py
#check  MulEquiv.prodCongr

def  finalIso  :  G  ≃*  H  ×  K  :=
  sorry 
```

## 9.2. 环

### 9.2.1. 环、它们的单位、态射和子环

在类型 `R` 上的环结构类型是 `Ring R`。假设乘法是交换的变体是 `CommRing R`。我们已经看到，`ring` 策略将证明任何从交换环公理中得出的等式。

```py
example  {R  :  Type*}  [CommRing  R]  (x  y  :  R)  :  (x  +  y)  ^  2  =  x  ^  2  +  y  ^  2  +  2  *  x  *  y  :=  by  ring 
```

更为奇特的变体不需要 `R` 上的加法形成一个群，而只是一个加法幺半群。相应的类型类是 `Semiring R` 和 `CommSemiring R`。自然数的类型是 `CommSemiring R` 的重要实例，任何取自然数为值的函数类型也是如此。另一个重要例子是环中的理想类型，将在下面讨论。`ring` 策略的名称具有双重误导性，因为它假设了交换性，但在半环中也能工作。换句话说，它适用于任何 `CommSemiring`。

```py
example  (x  y  :  ℕ)  :  (x  +  y)  ^  2  =  x  ^  2  +  y  ^  2  +  2  *  x  *  y  :=  by  ring 
```

环和半环类的版本也有不假设存在乘法单位元或乘法结合律的情况。我们这里不会讨论这些。

在环论导论中传统上教授的一些概念实际上与底层的乘法幺半群有关。一个突出的例子是环的单位定义。每个（乘法）幺半群 `M` 都有一个谓词 `IsUnit : M → Prop`，断言存在一个两边的逆元，一个单位类型 `Units M`，记作 `Mˣ`，以及一个到 `M` 的强制转换。类型 `Units M` 将可逆元素与其逆元以及确保每个确实是另一个的逆元的属性捆绑在一起。这个实现细节主要在定义可计算函数时相关。在大多数情况下，可以使用 `IsUnit.unit {x : M} : IsUnit x → Mˣ` 来构建一个单位。在交换情况下，还有一个 `Units.mkOfMulEqOne (x y : M) : x * y = 1 → Mˣ`，它构建了被视为单位的 `x`。

```py
example  (x  :  ℤˣ)  :  x  =  1  ∨  x  =  -1  :=  Int.units_eq_one_or  x

example  {M  :  Type*}  [Monoid  M]  (x  :  Mˣ)  :  (x  :  M)  *  x⁻¹  =  1  :=  Units.mul_inv  x

example  {M  :  Type*}  [Monoid  M]  :  Group  Mˣ  :=  inferInstance 
```

两个（半）环 `R` 和 `S` 之间的环同态类型是 `RingHom R S`，记作 `R →+* S`。

```py
example  {R  S  :  Type*}  [Ring  R]  [Ring  S]  (f  :  R  →+*  S)  (x  y  :  R)  :
  f  (x  +  y)  =  f  x  +  f  y  :=  f.map_add  x  y

example  {R  S  :  Type*}  [Ring  R]  [Ring  S]  (f  :  R  →+*  S)  :  Rˣ  →*  Sˣ  :=
  Units.map  f 
```

同构变体是 `RingEquiv`，记作 `≃+*`。

与子幺半群和子群一样，存在一个 `Subring R` 类型，用于表示环 `R` 的子环，但这个类型比子群类型要少用得多，因为不能通过子环来商环。

```py
example  {R  :  Type*}  [Ring  R]  (S  :  Subring  R)  :  Ring  S  :=  inferInstance 
```

还要注意，`RingHom.range` 产生一个子环。

### 9.2.2\. 理想与商

由于历史原因，Mathlib 只为交换环提供了理想理论。（环库最初是为了快速推进现代代数几何的基础而开发的。）因此，在本节中，我们将使用交换（半）环。`R`的理想被定义为将`R`视为`R`-模的子模。模块将在线性代数章节中稍后讨论，但这个实现细节可以大部分安全忽略，因为大多数（但不是所有）相关的引理都在理想的特殊上下文中重新表述。但匿名投影符号并不总是按预期工作。例如，在下面的片段中不能将`Ideal.Quotient.mk I`替换为`I.Quotient.mk`，因为有两个`.`，所以它将被解析为`(Ideal.Quotient I).mk`；但`Ideal.Quotient`本身并不存在。

```py
example  {R  :  Type*}  [CommRing  R]  (I  :  Ideal  R)  :  R  →+*  R  ⧸  I  :=
  Ideal.Quotient.mk  I

example  {R  :  Type*}  [CommRing  R]  {a  :  R}  {I  :  Ideal  R}  :
  Ideal.Quotient.mk  I  a  =  0  ↔  a  ∈  I  :=
  Ideal.Quotient.eq_zero_iff_mem 
```

商环的泛性性质是`Ideal.Quotient.lift`。

```py
example  {R  S  :  Type*}  [CommRing  R]  [CommRing  S]  (I  :  Ideal  R)  (f  :  R  →+*  S)
  (H  :  I  ≤  RingHom.ker  f)  :  R  ⧸  I  →+*  S  :=
  Ideal.Quotient.lift  I  f  H 
```

尤其是它导致了环的第一同构定理。

```py
example  {R  S  :  Type*}  [CommRing  R]  CommRing  S  :
  R  ⧸  RingHom.ker  f  ≃+*  f.range  :=
  RingHom.quotientKerEquivRange  f 
```

理想在包含关系下形成一个完全格结构，以及一个半环结构。这两个结构相互作用得很好。

```py
variable  {R  :  Type*}  [CommRing  R]  {I  J  :  Ideal  R}

example  :  I  +  J  =  I  ⊔  J  :=  rfl

example  {x  :  R}  :  x  ∈  I  +  J  ↔  ∃  a  ∈  I,  ∃  b  ∈  J,  a  +  b  =  x  :=  by
  simp  [Submodule.mem_sup]

example  :  I  *  J  ≤  J  :=  Ideal.mul_le_left

example  :  I  *  J  ≤  I  :=  Ideal.mul_le_right

example  :  I  *  J  ≤  I  ⊓  J  :=  Ideal.mul_le_inf 
```

可以使用环同态来分别使用`Ideal.map`和`Ideal.comap`将理想前推和后拉。通常，后者更方便使用，因为它不涉及存在量词。这解释了为什么它被用来表述允许我们在商环之间构建同态的条件。

```py
example  {R  S  :  Type*}  [CommRing  R]  [CommRing  S]  (I  :  Ideal  R)  (J  :  Ideal  S)  (f  :  R  →+*  S)
  (H  :  I  ≤  Ideal.comap  f  J)  :  R  ⧸  I  →+*  S  ⧸  J  :=
  Ideal.quotientMap  J  f  H 
```

一个微妙之处在于类型`R ⧸ I`实际上依赖于`I`（直到定义等价），因此，两个理想`I`和`J`相等的证明不足以使相应的商相等。然而，泛性性质在这种情况下确实提供了一个同构。

```py
example  {R  :  Type*}  [CommRing  R]  {I  J  :  Ideal  R}  (h  :  I  =  J)  :  R  ⧸  I  ≃+*  R  ⧸  J  :=
  Ideal.quotEquivOfEq  h 
```

我们现在可以以一个例子来展示中国剩余同构。请注意，索引下确界符号`⨅`和类型的大积符号`Π`之间的区别。根据你的字体，这些可能很难区分。

```py
example  {R  :  Type*}  [CommRing  R]  {ι  :  Type*}  [Fintype  ι]  (f  :  ι  →  Ideal  R)
  (hf  :  ∀  i  j,  i  ≠  j  →  IsCoprime  (f  i)  (f  j))  :  (R  ⧸  ⨅  i,  f  i)  ≃+*  Π  i,  R  ⧸  f  i  :=
  Ideal.quotientInfRingEquivPiQuotient  f  hf 
```

中国剩余定理的初等版本，关于`ZMod`的陈述，可以很容易地从先前的定理中推导出来：

```py
open  BigOperators  PiNotation

example  {ι  :  Type*}  [Fintype  ι]  (a  :  ι  →  ℕ)  (coprime  :  ∀  i  j,  i  ≠  j  →  (a  i).Coprime  (a  j))  :
  ZMod  (∏  i,  a  i)  ≃+*  Π  i,  ZMod  (a  i)  :=
  ZMod.prodEquivPi  a  coprime 
```

作为一系列练习，我们将重新证明在一般情况下的中国剩余定理。

我们首先需要定义定理中出现的映射，作为一个环同态，利用商环的泛性性质。

```py
variable  {ι  R  :  Type*}  [CommRing  R]
open  Ideal  Quotient  Function

#check  Pi.ringHom
#check  ker_Pi_Quotient_mk

/-- The homomorphism from ``R ⧸ ⨅ i, I i`` to ``Π i, R ⧸ I i`` featured in the Chinese
 Remainder Theorem. -/
def  chineseMap  (I  :  ι  →  Ideal  R)  :  (R  ⧸  ⨅  i,  I  i)  →+*  Π  i,  R  ⧸  I  i  :=
  sorry 
```

确保以下两个引理可以通过`rfl`证明。

```py
lemma  chineseMap_mk  (I  :  ι  →  Ideal  R)  (x  :  R)  :
  chineseMap  I  (Quotient.mk  _  x)  =  fun  i  :  ι  ↦  Ideal.Quotient.mk  (I  i)  x  :=
  sorry

lemma  chineseMap_mk'  (I  :  ι  →  Ideal  R)  (x  :  R)  (i  :  ι)  :
  chineseMap  I  (mk  _  x)  i  =  mk  (I  i)  x  :=
  sorry 
```

下一个引理证明了中国剩余定理的简单一半，对理想的族没有任何假设。证明不到一行长。

```py
#check  injective_lift_iff

lemma  chineseMap_inj  (I  :  ι  →  Ideal  R)  :  Injective  (chineseMap  I)  :=  by
  sorry 
```

现在我们已经准备好进入定理的核心部分，这将展示我们的`chineseMap`的满射性。首先我们需要知道表达互质（也称为共最大性假设）的不同方式。下面只需要前两种。

```py
#check  IsCoprime
#check  isCoprime_iff_add
#check  isCoprime_iff_exists
#check  isCoprime_iff_sup_eq
#check  isCoprime_iff_codisjoint 
```

我们利用这个机会，通过`Finset`进行归纳。以下给出了与`Finset`相关的引理。记住，`ring`策略适用于半环，并且环的理想形成一个半环。

```py
#check  Finset.mem_insert_of_mem
#check  Finset.mem_insert_self

theorem  isCoprime_Inf  {I  :  Ideal  R}  {J  :  ι  →  Ideal  R}  {s  :  Finset  ι}
  (hf  :  ∀  j  ∈  s,  IsCoprime  I  (J  j))  :  IsCoprime  I  (⨅  j  ∈  s,  J  j)  :=  by
  classical
  simp_rw  [isCoprime_iff_add]  at  *
  induction  s  using  Finset.induction  with
  |  empty  =>
  simp
  |  @insert  i  s  _  hs  =>
  rw  [Finset.iInf_insert,  inf_comm,  one_eq_top,  eq_top_iff,  ←  one_eq_top]
  set  K  :=  ⨅  j  ∈  s,  J  j
  calc
  1  =  I  +  K  :=  sorry
  _  =  I  +  K  *  (I  +  J  i)  :=  sorry
  _  =  (1  +  K)  *  I  +  K  *  J  i  :=  sorry
  _  ≤  I  +  K  ⊓  J  i  :=  sorry 
```

我们现在可以证明中国剩余定理中出现的映射的满射性。

```py
lemma  chineseMap_surj  [Fintype  ι]  {I  :  ι  →  Ideal  R}
  (hI  :  ∀  i  j,  i  ≠  j  →  IsCoprime  (I  i)  (I  j))  :  Surjective  (chineseMap  I)  :=  by
  classical
  intro  g
  choose  f  hf  using  fun  i  ↦  Ideal.Quotient.mk_surjective  (g  i)
  have  key  :  ∀  i,  ∃  e  :  R,  mk  (I  i)  e  =  1  ∧  ∀  j,  j  ≠  i  →  mk  (I  j)  e  =  0  :=  by
  intro  i
  have  hI'  :  ∀  j  ∈  ({i}  :  Finset  ι)ᶜ,  IsCoprime  (I  i)  (I  j)  :=  by
  sorry
  sorry
  choose  e  he  using  key
  use  mk  _  (∑  i,  f  i  *  e  i)
  sorry 
```

现在所有这些部分都在以下内容中汇集在一起：

```py
noncomputable  def  chineseIso  [Fintype  ι]  (f  :  ι  →  Ideal  R)
  (hf  :  ∀  i  j,  i  ≠  j  →  IsCoprime  (f  i)  (f  j))  :  (R  ⧸  ⨅  i,  f  i)  ≃+*  Π  i,  R  ⧸  f  i  :=
  {  Equiv.ofBijective  _  ⟨chineseMap_inj  f,  chineseMap_surj  hf⟩,
  chineseMap  f  with  } 
```

### 9.2.3\. 代数和多项式

给定一个交换（半）环 `R`，一个在 `R` 上的代数是一个半环 `A`，它配备了一个环同态，其像与 `A` 的每个元素交换。这被编码为类型类 `Algebra R A`。从 `R` 到 `A` 的同态被称为结构映射，在 Lean 中表示为 `algebraMap R A : R →+* A`。对于某个 `r : R`，`a : A` 通过 `algebraMap R A r` 的乘法称为 `a` 通过 `r` 的标量乘法，表示为 `r • a`。请注意，这种代数的概念有时被称为*结合有单位代数*，以强调存在更一般的代数概念。

`algebraMap R A` 是环同态的事实将标量乘法的许多属性打包在一起，例如以下内容：

```py
example  {R  A  :  Type*}  [CommRing  R]  [Ring  A]  [Algebra  R  A]  (r  r'  :  R)  (a  :  A)  :
  (r  +  r')  •  a  =  r  •  a  +  r'  •  a  :=
  add_smul  r  r'  a

example  {R  A  :  Type*}  [CommRing  R]  [Ring  A]  [Algebra  R  A]  (r  r'  :  R)  (a  :  A)  :
  (r  *  r')  •  a  =  r  •  r'  •  a  :=
  mul_smul  r  r'  a 
```

两个 `R`-代数 `A` 和 `B` 之间的形态是环同态，它们与 `R` 的元素进行标量乘法时保持交换。它们是带有类型 `AlgHom R A B` 的打包形态，表示为 `A →ₐ[R] B`。

非交换代数的重要例子包括自同态代数和方阵代数，这两者都将在线性代数章节中介绍。在本章中，我们将讨论一个最重要的交换代数例子，即多项式代数。

具有系数在 `R` 中的单变量多项式代数称为 `Polynomial R`，一旦打开 `Polynomial` 命名空间，就可以写成 `R[X]`。从 `R` 到 `R[X]` 的代数结构映射表示为 `C`，代表“常数”，因为相应的多项式函数始终是常数。不定元表示为 `X`。

```py
open  Polynomial

example  {R  :  Type*}  [CommRing  R]  :  R[X]  :=  X

example  {R  :  Type*}  [CommRing  R]  (r  :  R)  :=  X  -  C  r 
```

在上述第一个例子中，我们给 Lean 指定预期的类型至关重要，因为类型不能从定义体中确定。在第二个例子中，由于 `r` 的类型已知，我们可以从我们对 `C r` 的使用中推断出目标多项式代数。

因为 `C` 是从 `R` 到 `R[X]` 的环同态，所以我们可以在计算环 `R[X]` 之前使用所有环同态引理，例如 `map_zero`、`map_one`、`map_mul` 和 `map_pow`。例如：

```py
example  {R  :  Type*}  [CommRing  R]  (r  :  R)  :  (X  +  C  r)  *  (X  -  C  r)  =  X  ^  2  -  C  (r  ^  2)  :=  by
  rw  [C.map_pow]
  ring 
```

您可以使用 `Polynomial.coeff` 访问系数。

```py
example  {R  :  Type*}  [CommRing  R]  (r:R)  :  (C  r).coeff  0  =  r  :=  by  simp

example  {R  :  Type*}  [CommRing  R]  :  (X  ^  2  +  2  *  X  +  C  3  :  R[X]).coeff  1  =  2  :=  by  simp 
```

定义多项式的次数总是很棘手，因为零多项式的特殊情况。Mathlib 有两种变体：`Polynomial.natDegree : R[X] → ℕ` 将零多项式的次数分配为 `0`，而 `Polynomial.degree : R[X] → WithBot ℕ` 分配 `⊥`。在后一种情况下，`WithBot ℕ` 可以看作是 `ℕ ∪ {-∞}`，除了 `-∞` 被表示为 `⊥`，与完备格中的底元素相同的符号。这个特殊值被用作零多项式的次数，并且对于加法是吸收的。（对于乘法几乎也是吸收的，除了 `⊥ * 0 = 0`。）

从道德上讲，`degree` 版本是正确的。例如，它允许我们陈述乘积的预期公式（假设基环没有零因子）。

```py
example  {R  :  Type*}  [Semiring  R]  [NoZeroDivisors  R]  {p  q  :  R[X]}  :
  degree  (p  *  q)  =  degree  p  +  degree  q  :=
  Polynomial.degree_mul 
```

然而，对于 `natDegree` 版本需要假设非零多项式。

```py
example  {R  :  Type*}  [Semiring  R]  [NoZeroDivisors  R]  {p  q  :  R[X]}  (hp  :  p  ≠  0)  (hq  :  q  ≠  0)  :
  natDegree  (p  *  q)  =  natDegree  p  +  natDegree  q  :=
  Polynomial.natDegree_mul  hp  hq 
```

然而，`ℕ` 比使用 `WithBot ℕ` 更方便，所以 Mathlib 提供了两种版本，并提供了一些引理来在它们之间进行转换。此外，`natDegree` 是在计算复合的次数时更方便的定义。多项式的复合是 `Polynomial.comp`，我们有：

```py
example  {R  :  Type*}  [Semiring  R]  [NoZeroDivisors  R]  {p  q  :  R[X]}  :
  natDegree  (comp  p  q)  =  natDegree  p  *  natDegree  q  :=
  Polynomial.natDegree_comp 
```

多项式产生了多项式函数：任何多项式都可以使用 `Polynomial.eval` 在 `R` 上进行评估。

```py
example  {R  :  Type*}  [CommRing  R]  (P:  R[X])  (x  :  R)  :=  P.eval  x

example  {R  :  Type*}  [CommRing  R]  (r  :  R)  :  (X  -  C  r).eval  r  =  0  :=  by  simp 
```

特别是，有一个谓词 `IsRoot`，它对于在 `R` 中使多项式为零的元素 `r` 成立。

```py
example  {R  :  Type*}  [CommRing  R]  (P  :  R[X])  (r  :  R)  :  IsRoot  P  r  ↔  P.eval  r  =  0  :=  Iff.rfl 
```

我们希望说明，假设 `R` 没有零因子，一个多项式的根的数量最多与其次数相同，这里的根是按重数计算的。但一旦再次遇到零多项式的情况就令人痛苦。因此，Mathlib 定义了 `Polynomial.roots` 来将一个多项式 `P` 映射到一个多重集，即如果 `P` 是零多项式，则定义为空集，否则是 `P` 的根及其重数。这个定义仅在基础环是整环时才有效，因为否则定义不具有好的性质。

```py
example  {R  :  Type*}  [CommRing  R]  [IsDomain  R]  (r  :  R)  :  (X  -  C  r).roots  =  {r}  :=
  roots_X_sub_C  r

example  {R  :  Type*}  [CommRing  R]  [IsDomain  R]  (r  :  R)  (n  :  ℕ):
  ((X  -  C  r)  ^  n).roots  =  n  •  {r}  :=
  by  simp 
```

`Polynomial.eval` 和 `Polynomial.roots` 只考虑系数环。它们不允许我们说 `X ^ 2 - 2 : ℚ[X]` 在 `ℝ` 中有一个根，或者 `X ^ 2 + 1 : ℝ[X]` 在 `ℂ` 中有一个根。为此，我们需要 `Polynomial.aeval`，它将在任何 `R`-代数中评估 `P : R[X]`。更精确地说，给定一个半环 `A` 和一个 `Algebra R A` 实例，`Polynomial.aeval` 将 `a` 的每个元素沿着在 `a` 处的评估的 `R`-代数同态发送。由于 `AlgHom` 有一个到函数的强制转换，可以将其应用于多项式。但 `aeval` 没有以多项式作为参数，因此不能使用像上面 `P.eval` 中的点符号。

```py
example  :  aeval  Complex.I  (X  ^  2  +  1  :  ℝ[X])  =  0  :=  by  simp 
```

在这个上下文中对应于 `roots` 的函数是 `aroots`，它接受一个多项式然后是一个代数，并输出一个多重集（与 `roots` 相同，关于零多项式的注意事项）。

```py
open  Complex  Polynomial

example  :  aroots  (X  ^  2  +  1  :  ℝ[X])  ℂ  =  {Complex.I,  -I}  :=  by
  suffices  roots  (X  ^  2  +  1  :  ℂ[X])  =  {I,  -I}  by  simpa  [aroots_def]
  have  factored  :  (X  ^  2  +  1  :  ℂ[X])  =  (X  -  C  I)  *  (X  -  C  (-I))  :=  by
  have  key  :  (C  I  *  C  I  :  ℂ[X])  =  -1  :=  by  simp  [←  C_mul]
  rw  [C_neg]
  linear_combination  key
  have  p_ne_zero  :  (X  -  C  I)  *  (X  -  C  (-I))  ≠  0  :=  by
  intro  H
  apply_fun  eval  0  at  H
  simp  [eval]  at  H
  simp  only  [factored,  roots_mul  p_ne_zero,  roots_X_sub_C]
  rfl

-- Mathlib knows about D'Alembert-Gauss theorem: ``ℂ`` is algebraically closed.
example  :  IsAlgClosed  ℂ  :=  inferInstance 
```

更普遍地，给定一个环同态 `f : R →+* S`，可以使用 `Polynomial.eval₂` 在 `S` 中的某一点评估 `P : R[X]`。这个操作产生了一个从 `R[X]` 到 `S` 的实际函数，因为它不假设存在一个 `Algebra R S` 实例，所以点符号的使用方式正如你所期望的那样。

```py
#check  (Complex.ofRealHom  :  ℝ  →+*  ℂ)

example  :  (X  ^  2  +  1  :  ℝ[X]).eval₂  Complex.ofRealHom  Complex.I  =  0  :=  by  simp 
```

让我们简要地提一下多元多项式。给定一个交换半环 `R`，系数在 `R` 中且变量由类型 `σ` 索引的多项式 `R`-代数是 `MVPolynomial σ R`。给定 `i : σ`，相应的多项式是 `MvPolynomial.X i`。（像往常一样，可以打开 `MVPolynomial` 命名空间来缩短表示为 `X i`。）例如，如果我们想要两个变量，我们可以使用 `Fin 2` 作为 `σ`，并将定义单位圆的 $\mathbb{R}²$ 中的多项式写为：

```py
open  MvPolynomial

def  circleEquation  :  MvPolynomial  (Fin  2)  ℝ  :=  X  0  ^  2  +  X  1  ^  2  -  1 
```

回想一下，函数应用有很高的优先级，所以上面的表达式读作 `(X 0) ^ 2 + (X 1) ^ 2 - 1`。我们可以评估它以确保坐标为 $(1, 0)$ 的点在圆上。回想一下，`![...]` 符号表示 `Fin n → X` 的元素，其中 `n` 是由参数的数量决定的某个自然数，`X` 是由参数的类型决定的某个类型。

```py
example  :  MvPolynomial.eval  ![1,  0]  circleEquation  =  0  :=  by  simp  [circleEquation] 
```

### 9.2.1\. 环、它们的单位、同态和子环

类型 `R` 上的环结构类型是 `Ring R`。假设乘法是交换的变体是 `CommRing R`。我们已经看到，`ring` 策略将证明任何从交换环公理中得出的等式。

```py
example  {R  :  Type*}  [CommRing  R]  (x  y  :  R)  :  (x  +  y)  ^  2  =  x  ^  2  +  y  ^  2  +  2  *  x  *  y  :=  by  ring 
```

更奇特的变体不需要 `R` 上的加法形成一个群，而只需要一个加法幺半群。相应的类型类是 `Semiring R` 和 `CommSemiring R`。自然数的类型是 `CommSemiring R` 的重要实例，任何取自然数值的函数类型也是如此。另一个重要例子是环中的理想类型，将在下面讨论。`ring` 策略的名称具有双重误导性，因为它假设了交换性，但在半环中也能工作。换句话说，它适用于任何 `CommSemiring`。

```py
example  (x  y  :  ℕ)  :  (x  +  y)  ^  2  =  x  ^  2  +  y  ^  2  +  2  *  x  *  y  :=  by  ring 
```

环和半环类的版本也有不假设存在乘法单位或乘法结合律的情况。我们这里不会讨论这些。

一些在环论导论中传统上教授的概念实际上是关于底层乘法幺半群的。一个突出的例子是环的单位定义。每个（乘法）幺半群 `M` 都有一个谓词 `IsUnit : M → Prop`，它断言存在一个两边的逆元，一个单位类型 `Units M`，记作 `Mˣ`，以及一个到 `M` 的强制转换。类型 `Units M` 将可逆元素及其逆元以及确保每个确实是另一个的逆元的属性捆绑在一起。这个实现细节主要在定义可计算函数时相关。在大多数情况下，可以使用 `IsUnit.unit {x : M} : IsUnit x → Mˣ` 来构建一个单位。在交换情况下，还有一个 `Units.mkOfMulEqOne (x y : M) : x * y = 1 → Mˣ`，它构建了被视为单位的 `x`。

```py
example  (x  :  ℤˣ)  :  x  =  1  ∨  x  =  -1  :=  Int.units_eq_one_or  x

example  {M  :  Type*}  [Monoid  M]  (x  :  Mˣ)  :  (x  :  M)  *  x⁻¹  =  1  :=  Units.mul_inv  x

example  {M  :  Type*}  [Monoid  M]  :  Group  Mˣ  :=  inferInstance 
```

两个（半）环 `R` 和 `S` 之间环同态的类型是 `RingHom R S`，记作 `R →+* S`。

```py
example  {R  S  :  Type*}  [Ring  R]  [Ring  S]  (f  :  R  →+*  S)  (x  y  :  R)  :
  f  (x  +  y)  =  f  x  +  f  y  :=  f.map_add  x  y

example  {R  S  :  Type*}  [Ring  R]  [Ring  S]  (f  :  R  →+*  S)  :  Rˣ  →*  Sˣ  :=
  Units.map  f 
```

同构变体是 `RingEquiv`，记作 `≃+*`。

与子幺半群和子群一样，存在一个 `Subring R` 类型，用于表示环 `R` 的子环，但这个类型比子群类型要少用得多，因为不能通过子环来商环。

```py
example  {R  :  Type*}  [Ring  R]  (S  :  Subring  R)  :  Ring  S  :=  inferInstance 
```

还要注意，`RingHom.range` 产生一个子环。

### 9.2.2\. 理想与商

由于历史原因，Mathlib 只有一种关于交换环的理想理论。（环库最初是为了快速推进现代代数几何的基础而开发的。）因此，在本节中，我们将使用交换（半）环。`R` 的理想被定义为将 `R` 视为 `R`-模块的子模块。模块将在线性代数章节中稍后介绍，但这个实现细节可以大部分安全忽略，因为大多数（但不是所有）相关引理都在理想的特殊上下文中重新表述。但匿名投影符号并不总是按预期工作。例如，在下面的片段中不能将 `Ideal.Quotient.mk I` 替换为 `I.Quotient.mk`，因为有两个 `.`，所以它将被解析为 `(Ideal.Quotient I).mk`；但 `Ideal.Quotient` 本身并不存在。

```py
example  {R  :  Type*}  [CommRing  R]  (I  :  Ideal  R)  :  R  →+*  R  ⧸  I  :=
  Ideal.Quotient.mk  I

example  {R  :  Type*}  [CommRing  R]  {a  :  R}  {I  :  Ideal  R}  :
  Ideal.Quotient.mk  I  a  =  0  ↔  a  ∈  I  :=
  Ideal.Quotient.eq_zero_iff_mem 
```

商环的泛性性质是 `Ideal.Quotient.lift`。

```py
example  {R  S  :  Type*}  [CommRing  R]  [CommRing  S]  (I  :  Ideal  R)  (f  :  R  →+*  S)
  (H  :  I  ≤  RingHom.ker  f)  :  R  ⧸  I  →+*  S  :=
  Ideal.Quotient.lift  I  f  H 
```

尤其是它导致了环的第一同构定理。

```py
example  {R  S  :  Type*}  [CommRing  R]  CommRing  S  :
  R  ⧸  RingHom.ker  f  ≃+*  f.range  :=
  RingHom.quotientKerEquivRange  f 
```

理想在包含关系下形成一个完全格结构，以及半环结构。这两个结构相互作用得很好。

```py
variable  {R  :  Type*}  [CommRing  R]  {I  J  :  Ideal  R}

example  :  I  +  J  =  I  ⊔  J  :=  rfl

example  {x  :  R}  :  x  ∈  I  +  J  ↔  ∃  a  ∈  I,  ∃  b  ∈  J,  a  +  b  =  x  :=  by
  simp  [Submodule.mem_sup]

example  :  I  *  J  ≤  J  :=  Ideal.mul_le_left

example  :  I  *  J  ≤  I  :=  Ideal.mul_le_right

example  :  I  *  J  ≤  I  ⊓  J  :=  Ideal.mul_le_inf 
```

可以使用环同态将理想向前推进并使用 `Ideal.map` 和 `Ideal.comap` 分别将其拉回。通常，后者更方便使用，因为它不涉及存在量词。这解释了为什么它被用来表述允许我们在商环之间构建同态的条件。

```py
example  {R  S  :  Type*}  [CommRing  R]  [CommRing  S]  (I  :  Ideal  R)  (J  :  Ideal  S)  (f  :  R  →+*  S)
  (H  :  I  ≤  Ideal.comap  f  J)  :  R  ⧸  I  →+*  S  ⧸  J  :=
  Ideal.quotientMap  J  f  H 
```

一个微妙之处在于类型 `R ⧸ I` 实际上依赖于 `I`（直到定义等价），因此，两个理想 `I` 和 `J` 相等的证明不足以使相应的商相等。然而，泛性性质在这种情况下确实提供了一个同构。

```py
example  {R  :  Type*}  [CommRing  R]  {I  J  :  Ideal  R}  (h  :  I  =  J)  :  R  ⧸  I  ≃+*  R  ⧸  J  :=
  Ideal.quotEquivOfEq  h 
```

我们现在可以以一个例子来展示中国剩余同构。请注意，索引下确界符号 `⨅` 和类型的大积符号 `Π` 之间的区别。根据你的字体，这些可能很难区分。

```py
example  {R  :  Type*}  [CommRing  R]  {ι  :  Type*}  [Fintype  ι]  (f  :  ι  →  Ideal  R)
  (hf  :  ∀  i  j,  i  ≠  j  →  IsCoprime  (f  i)  (f  j))  :  (R  ⧸  ⨅  i,  f  i)  ≃+*  Π  i,  R  ⧸  f  i  :=
  Ideal.quotientInfRingEquivPiQuotient  f  hf 
```

中国剩余定理的初等版本，一个关于 `ZMod` 的陈述，可以很容易地从先前的定理中推导出来：

```py
open  BigOperators  PiNotation

example  {ι  :  Type*}  [Fintype  ι]  (a  :  ι  →  ℕ)  (coprime  :  ∀  i  j,  i  ≠  j  →  (a  i).Coprime  (a  j))  :
  ZMod  (∏  i,  a  i)  ≃+*  Π  i,  ZMod  (a  i)  :=
  ZMod.prodEquivPi  a  coprime 
```

作为一系列练习，我们将重新证明一般情况下的中国剩余定理。

我们首先需要定义定理中出现的映射，作为一个环同态，使用商环的泛性性质。

```py
variable  {ι  R  :  Type*}  [CommRing  R]
open  Ideal  Quotient  Function

#check  Pi.ringHom
#check  ker_Pi_Quotient_mk

/-- The homomorphism from ``R ⧸ ⨅ i, I i`` to ``Π i, R ⧸ I i`` featured in the Chinese
 Remainder Theorem. -/
def  chineseMap  (I  :  ι  →  Ideal  R)  :  (R  ⧸  ⨅  i,  I  i)  →+*  Π  i,  R  ⧸  I  i  :=
  sorry 
```

确保以下两个引理可以通过 `rfl` 证明。

```py
lemma  chineseMap_mk  (I  :  ι  →  Ideal  R)  (x  :  R)  :
  chineseMap  I  (Quotient.mk  _  x)  =  fun  i  :  ι  ↦  Ideal.Quotient.mk  (I  i)  x  :=
  sorry

lemma  chineseMap_mk'  (I  :  ι  →  Ideal  R)  (x  :  R)  (i  :  ι)  :
  chineseMap  I  (mk  _  x)  i  =  mk  (I  i)  x  :=
  sorry 
```

下一个引理证明了中国剩余定理的简单一半，对理想的族没有任何假设。证明不到一行长。

```py
#check  injective_lift_iff

lemma  chineseMap_inj  (I  :  ι  →  Ideal  R)  :  Injective  (chineseMap  I)  :=  by
  sorry 
```

我们现在可以准备定理的核心部分，它将展示我们的 `chineseMap` 的满射性。首先我们需要知道表达互质（也称为共最大性假设）的不同方式。下面只需要前两种。

```py
#check  IsCoprime
#check  isCoprime_iff_add
#check  isCoprime_iff_exists
#check  isCoprime_iff_sup_eq
#check  isCoprime_iff_codisjoint 
```

我们利用这个机会使用对 `Finset` 的归纳。以下给出了 `Finset` 上的相关引理。记住，`ring` 策略适用于半环，并且环的理想形成一个半环。

```py
#check  Finset.mem_insert_of_mem
#check  Finset.mem_insert_self

theorem  isCoprime_Inf  {I  :  Ideal  R}  {J  :  ι  →  Ideal  R}  {s  :  Finset  ι}
  (hf  :  ∀  j  ∈  s,  IsCoprime  I  (J  j))  :  IsCoprime  I  (⨅  j  ∈  s,  J  j)  :=  by
  classical
  simp_rw  [isCoprime_iff_add]  at  *
  induction  s  using  Finset.induction  with
  |  empty  =>
  simp
  |  @insert  i  s  _  hs  =>
  rw  [Finset.iInf_insert,  inf_comm,  one_eq_top,  eq_top_iff,  ←  one_eq_top]
  set  K  :=  ⨅  j  ∈  s,  J  j
  calc
  1  =  I  +  K  :=  sorry
  _  =  I  +  K  *  (I  +  J  i)  :=  sorry
  _  =  (1  +  K)  *  I  +  K  *  J  i  :=  sorry
  _  ≤  I  +  K  ⊓  J  i  :=  sorry 
```

我们现在可以证明中国剩余定理中出现的映射的满射性。

```py
lemma  chineseMap_surj  [Fintype  ι]  {I  :  ι  →  Ideal  R}
  (hI  :  ∀  i  j,  i  ≠  j  →  IsCoprime  (I  i)  (I  j))  :  Surjective  (chineseMap  I)  :=  by
  classical
  intro  g
  choose  f  hf  using  fun  i  ↦  Ideal.Quotient.mk_surjective  (g  i)
  have  key  :  ∀  i,  ∃  e  :  R,  mk  (I  i)  e  =  1  ∧  ∀  j,  j  ≠  i  →  mk  (I  j)  e  =  0  :=  by
  intro  i
  have  hI'  :  ∀  j  ∈  ({i}  :  Finset  ι)ᶜ,  IsCoprime  (I  i)  (I  j)  :=  by
  sorry
  sorry
  choose  e  he  using  key
  use  mk  _  (∑  i,  f  i  *  e  i)
  sorry 
```

现在所有这些部分都在以下内容中汇集在一起：

```py
noncomputable  def  chineseIso  [Fintype  ι]  (f  :  ι  →  Ideal  R)
  (hf  :  ∀  i  j,  i  ≠  j  →  IsCoprime  (f  i)  (f  j))  :  (R  ⧸  ⨅  i,  f  i)  ≃+*  Π  i,  R  ⧸  f  i  :=
  {  Equiv.ofBijective  _  ⟨chineseMap_inj  f,  chineseMap_surj  hf⟩,
  chineseMap  f  with  } 
```

### 9.2.3\. 代数与多项式

给定一个交换（半）环 `R`，`R` 上的代数是一个半环 `A`，它配备了一个环同态，其像与 `A` 的每个元素都交换。这被编码为类型类 `Algebra R A`。从 `R` 到 `A` 的映射称为结构映射，在 Lean 中表示为 `algebraMap R A : R →+* A`。对于某个 `r : R`，`a : A` 通过 `algebraMap R A r` 的乘法称为 `a` 通过 `r` 的标量乘法，并用 `r • a` 表示。请注意，这种代数的概念有时被称为 *结合有单位代数*，以强调存在更一般的代数概念。

事实是 `algebraMap R A` 是一个环同态，它将标量乘法的许多性质打包在一起，例如以下内容：

```py
example  {R  A  :  Type*}  [CommRing  R]  [Ring  A]  [Algebra  R  A]  (r  r'  :  R)  (a  :  A)  :
  (r  +  r')  •  a  =  r  •  a  +  r'  •  a  :=
  add_smul  r  r'  a

example  {R  A  :  Type*}  [CommRing  R]  [Ring  A]  [Algebra  R  A]  (r  r'  :  R)  (a  :  A)  :
  (r  *  r')  •  a  =  r  •  r'  •  a  :=
  mul_smul  r  r'  a 
```

两个 `R`-代数 `A` 和 `B` 之间的映射是环同态，它们与 `R` 的元素进行标量乘法时保持交换。它们是带有类型 `AlgHom R A B` 的打包映射，用 `A →ₐ[R] B` 表示。

非交换代数的重要例子包括自同态代数和方阵代数，这两者都将在线性代数章节中介绍。在本章中，我们将讨论一个最重要的交换代数例子，即多项式代数。

单变量多项式代数，其系数在 `R` 中，被称为 `多项式 R`，一旦打开 `多项式` 命名空间，就可以写成 `R[X]`。从 `R` 到 `R[X]` 的代数结构映射用 `C` 表示，因为相应的多项式函数始终是常数。不定元用 `X` 表示。

```py
open  Polynomial

example  {R  :  Type*}  [CommRing  R]  :  R[X]  :=  X

example  {R  :  Type*}  [CommRing  R]  (r  :  R)  :=  X  -  C  r 
```

在上述第一个例子中，我们向 Lean 提供预期的类型是至关重要的，因为它不能从定义的主体中确定。在第二个例子中，目标多项式代数可以通过我们对 `C r` 的使用来推断，因为 `r` 的类型是已知的。

因为 `C` 是从 `R` 到 `R[X]` 的环同态，所以我们可以在环 `R[X]` 中计算之前，使用所有环同态引理，如 `map_zero`、`map_one`、`map_mul` 和 `map_pow`。例如：

```py
example  {R  :  Type*}  [CommRing  R]  (r  :  R)  :  (X  +  C  r)  *  (X  -  C  r)  =  X  ^  2  -  C  (r  ^  2)  :=  by
  rw  [C.map_pow]
  ring 
```

您可以使用 `Polynomial.coeff` 访问系数。

```py
example  {R  :  Type*}  [CommRing  R]  (r:R)  :  (C  r).coeff  0  =  r  :=  by  simp

example  {R  :  Type*}  [CommRing  R]  :  (X  ^  2  +  2  *  X  +  C  3  :  R[X]).coeff  1  =  2  :=  by  simp 
```

定义多项式的次数总是很棘手，因为零多项式的特殊情况。Mathlib 有两种变体：`Polynomial.natDegree : R[X] → ℕ` 将零多项式的次数赋值为 `0`，而 `Polynomial.degree : R[X] → WithBot ℕ` 赋值为 `⊥`。在后一种情况下，`WithBot ℕ` 可以看作是 `ℕ ∪ {-∞}`，除了 `-∞` 用 `⊥` 表示，这与完备格中的底元素符号相同。这个特殊值被用作零多项式的次数，并且对加法是吸收的。（对于乘法几乎也是吸收的，除了 `⊥ * 0 = 0`。）

从道德上讲，`degree`版本是正确的。例如，它允许我们陈述乘积的度数的预期公式（假设基环没有零因子）。

```py
example  {R  :  Type*}  [Semiring  R]  [NoZeroDivisors  R]  {p  q  :  R[X]}  :
  degree  (p  *  q)  =  degree  p  +  degree  q  :=
  Polynomial.degree_mul 
```

与`natDegree`版本不同，需要假设多项式不为零。

```py
example  {R  :  Type*}  [Semiring  R]  [NoZeroDivisors  R]  {p  q  :  R[X]}  (hp  :  p  ≠  0)  (hq  :  q  ≠  0)  :
  natDegree  (p  *  q)  =  natDegree  p  +  natDegree  q  :=
  Polynomial.natDegree_mul  hp  hq 
```

然而，`ℕ`比`WithBot ℕ`更易于使用，因此 Mathlib 提供了这两种版本，并提供了将它们之间转换的引理。此外，当计算复合的度数时，`natDegree`是更方便的定义。多项式的复合是`Polynomial.comp`，我们有：

```py
example  {R  :  Type*}  [Semiring  R]  [NoZeroDivisors  R]  {p  q  :  R[X]}  :
  natDegree  (comp  p  q)  =  natDegree  p  *  natDegree  q  :=
  Polynomial.natDegree_comp 
```

多项式产生多项式函数：任何多项式都可以使用`Polynomial.eval`在`R`上评估。

```py
example  {R  :  Type*}  [CommRing  R]  (P:  R[X])  (x  :  R)  :=  P.eval  x

example  {R  :  Type*}  [CommRing  R]  (r  :  R)  :  (X  -  C  r).eval  r  =  0  :=  by  simp 
```

特别是，存在一个谓词`IsRoot`，它对`R`中的元素`r`成立，其中多项式为零。

```py
example  {R  :  Type*}  [CommRing  R]  (P  :  R[X])  (r  :  R)  :  IsRoot  P  r  ↔  P.eval  r  =  0  :=  Iff.rfl 
```

我们希望说，假设`R`没有零因子，多项式的根的数量最多与其度数相同，其中根是按重数计算的。但又一次，零多项式的情况很痛苦。因此，Mathlib 定义`Polynomial.roots`将多项式`P`发送到多重集，即如果`P`为零则定义为空集，否则是`P`的根，带有重数。这个定义仅在基础环是域时才成立，因为否则定义没有良好的性质。

```py
example  {R  :  Type*}  [CommRing  R]  [IsDomain  R]  (r  :  R)  :  (X  -  C  r).roots  =  {r}  :=
  roots_X_sub_C  r

example  {R  :  Type*}  [CommRing  R]  [IsDomain  R]  (r  :  R)  (n  :  ℕ):
  ((X  -  C  r)  ^  n).roots  =  n  •  {r}  :=
  by  simp 
```

`Polynomial.eval`和`Polynomial.roots`只考虑系数环。它们不允许我们说`X ^ 2 - 2 : ℚ[X]`在`ℝ`中有根，或者`X ^ 2 + 1 : ℝ[X]`在`ℂ`中有根。为此，我们需要`Polynomial.aeval`，它将在任何`R`-代数中评估`P : R[X]`。更精确地说，给定一个半环`A`和一个`Algebra R A`实例，`Polynomial.aeval`将`a`中的每个元素沿着在`a`上的评估的`R`-代数同态发送。由于`AlgHom`可以强制转换为函数，因此可以将它应用于多项式。但是`aeval`没有多项式作为参数，因此不能像上面的`P.eval`那样使用点符号。

```py
example  :  aeval  Complex.I  (X  ^  2  +  1  :  ℝ[X])  =  0  :=  by  simp 
```

在这个上下文中对应于`roots`的函数是`aroots`，它接受一个多项式然后是一个代数，并输出一个多重集（与`roots`相同的关于零多项式的警告）。

```py
open  Complex  Polynomial

example  :  aroots  (X  ^  2  +  1  :  ℝ[X])  ℂ  =  {Complex.I,  -I}  :=  by
  suffices  roots  (X  ^  2  +  1  :  ℂ[X])  =  {I,  -I}  by  simpa  [aroots_def]
  have  factored  :  (X  ^  2  +  1  :  ℂ[X])  =  (X  -  C  I)  *  (X  -  C  (-I))  :=  by
  have  key  :  (C  I  *  C  I  :  ℂ[X])  =  -1  :=  by  simp  [←  C_mul]
  rw  [C_neg]
  linear_combination  key
  have  p_ne_zero  :  (X  -  C  I)  *  (X  -  C  (-I))  ≠  0  :=  by
  intro  H
  apply_fun  eval  0  at  H
  simp  [eval]  at  H
  simp  only  [factored,  roots_mul  p_ne_zero,  roots_X_sub_C]
  rfl

-- Mathlib knows about D'Alembert-Gauss theorem: ``ℂ`` is algebraically closed.
example  :  IsAlgClosed  ℂ  :=  inferInstance 
```

更一般地，给定一个环同态`f : R →+* S`，可以使用`Polynomial.eval₂`在`S`中的某个点评估`P : R[X]`。这个操作实际上产生了一个从`R[X]`到`S`的函数，因为它不假设存在一个`Algebra R S`实例，所以点符号的使用方式符合预期。

```py
#check  (Complex.ofRealHom  :  ℝ  →+*  ℂ)

example  :  (X  ^  2  +  1  :  ℝ[X]).eval₂  Complex.ofRealHom  Complex.I  =  0  :=  by  simp 
```

让我们简要地提及多元多项式。给定一个交换半环`R`，系数在`R`中且变量由类型`σ`索引的多项式`R`-代数是`MVPolynomial σ R`。给定`i : σ`，相应的多项式是`MvPolynomial.X i`。（像往常一样，可以打开`MVPolynomial`命名空间以缩短为`X i`。）例如，如果我们想要两个变量，我们可以使用`Fin 2`作为`σ`，并将定义单位圆的`R²`中的多项式写为：

```py
open  MvPolynomial

def  circleEquation  :  MvPolynomial  (Fin  2)  ℝ  :=  X  0  ^  2  +  X  1  ^  2  -  1 
```

回想一下，函数应用具有非常高的优先级，因此上述表达式被读取为 `(X 0) ^ 2 + (X 1) ^ 2 - 1`。我们可以对其进行评估以确保坐标为 $(1, 0)$ 的点位于圆上。回想一下，`![...]` 符号表示由自然数 `n` 确定的 `Fin n → X` 的元素，其中 `n` 由参数的数量决定，而 `X` 由参数的类型决定。

```py
example  :  MvPolynomial.eval  ![1,  0]  circleEquation  =  0  :=  by  simp  [circleEquation] 
```*
