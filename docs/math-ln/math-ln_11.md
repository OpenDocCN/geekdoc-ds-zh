# 11. 拓扑学

> 原文：[`leanprover-community.github.io/mathematics_in_lean/C11_Topology.html`](https://leanprover-community.github.io/mathematics_in_lean/C11_Topology.html)

*Mathematics in Lean* **   11. 拓扑学

+   查看页面源代码

* * *

微积分基于函数的概念，用于模拟相互依赖的量。例如，研究随时间变化的量是很常见的。**极限**的概念也是基本的。我们可以说，当`x`趋近于值`a`时，函数`f(x)`的极限是一个值`b`，或者说`f(x)`当`x`趋近于`a`时收敛到`b`。等价地，我们可以说当`x`趋近于值`a`时，`f(x)`趋近于`b`，或者说它当`x`趋向于`a`时趋于`b`。我们已经在第 3.6 节中开始考虑这样的概念了。

**拓扑学**是极限和连续性的抽象研究。在第二章到第七章中，我们已经涵盖了形式化的基本要素，本章将解释 Mathlib 中拓扑概念的正式化。不仅拓扑抽象在更广泛的范围内适用，而且它们在某种程度上具有矛盾性，使得在具体实例中推理极限和连续性变得更加容易。

拓扑概念建立在许多数学结构层之上。第一层是朴素集合论，如第四章所述。下一层是**过滤器**理论，我们将在第 11.1 节中描述。在此基础上，我们叠加了**拓扑空间**、**度量空间**以及一个稍微更神秘的中间概念，称为**均匀空间**。

虽然前几章依赖于你可能熟悉的数学概念，但过滤器这一概念对许多在职数学家来说并不那么熟悉。然而，这一概念对于有效地形式化数学是至关重要的。让我们解释一下原因。设`f : ℝ → ℝ`为任何函数。我们可以考虑当`x`趋近于某个值`x₀`时`f x`的极限，但也可以考虑当`x`趋近于正无穷或负无穷时`f x`的极限。此外，我们还可以考虑当`x`从右侧（通常写作`x₀⁺`）或从左侧（写作`x₀⁻`）趋近于`x₀`时`f x`的极限。存在一些变体，其中`x`趋近于`x₀`、`x₀⁺`或`x₀⁻`，但不允许`x`取`x₀`本身的值。这至少导致了`x`趋近于某物的八种方式。我们还可以将`x`限制为有理值或对定义域施加其他约束，但让我们坚持这 8 种情况。

我们在值域上也有类似的多种选择：我们可以指定`f x`是从左边还是右边接近一个值，或者它接近正无穷或负无穷，等等。例如，我们可能希望说，当`x`从右边趋向于`x₀`而不等于`x₀`时，`f x`趋向于`+∞`。这导致了 64 种不同的极限陈述，而我们甚至还没有开始处理与我们在第 3.6 节中处理序列极限类似的情况。

当涉及到支持引理时，问题变得更加复杂。例如，极限可以组合：如果`f x`在`x`趋向于`x₀`时趋向于`y₀`，而`g y`在`y`趋向于`y₀`时趋向于`z₀`，那么`g ∘ f x`在`x`趋向于`x₀`时也趋向于`z₀`。这里涉及到三种“趋向于”的概念，每种都可以用前一段描述的八种方式中的任何一种来实例化。这导致了 512 个引理，要添加到库中确实很多！非正式地说，数学家通常只证明其中两三个，并简单地注明其余的可以“以相同的方式”证明。形式化数学需要使相关的“相同”概念完全明确，这正是 Bourbaki 的滤波器理论所做到的。

## 11.1\. 滤波器

在类型`X`上的*滤波器*是一组满足以下三个条件的`X`的集合。这个概念支持两个相关想法：

+   *极限*，包括上面讨论的所有类型的极限：序列的有限和无限极限，函数在一点或无穷远处的有限和无限极限，等等。

+   *最终发生的事情*，包括对于足够大的`n : ℕ`，或者在足够接近的点`x`，或者在足够接近的点对，或者在测度理论意义上的几乎处处。类似地，滤波器也可以表达*经常发生的事情*的概念：对于任意大的`n`，在任意给定点的任意邻域中等。

对应这些描述的滤波器将在本节后面定义，但我们已经可以命名它们：

+   `(atTop : Filter ℕ)`, 由包含`{n | n ≥ N}`（对于某个`N`）的`ℕ`的集合组成

+   `𝓝 x`, 由拓扑空间中`x`的邻域组成

+   `𝓤 X`，由均匀空间（均匀空间是度量空间和拓扑群的推广）的邻域组成。

+   `μ.ae`，由补集相对于测度`μ`的测度为零的集合组成。

一般定义如下：一个`Filter X`的滤波器是一个满足以下条件的集合`F.sets : Set (Set X)`：

+   `F.univ_sets : univ ∈ F.sets`

+   `F.sets_of_superset : ∀ {U V}, U ∈ F.sets → U ⊆ V → V ∈ F.sets`

+   `F.inter_sets : ∀ {U V}, U ∈ F.sets → V ∈ F.sets → U ∩ V ∈ F.sets`。

第一个条件说明 `X` 的所有元素集合属于 `F.sets`。第二个条件说明如果 `U` 属于 `F.sets`，那么包含 `U` 的任何东西也属于 `F.sets`。第三个条件说明 `F.sets` 对有限交集是封闭的。在 Mathlib 中，过滤器 `F` 被定义为捆绑 `F.sets` 和其三个属性的构造，但属性不携带任何额外数据，并且将 `F` 和 `F.sets` 之间的区别模糊化是方便的。因此，我们定义 `U ∈ F` 为 `U ∈ F.sets`。这解释了为什么在提及 `U ∈ F` 的某些引理名称中出现了单词 `sets`。

将过滤器视为定义“足够大”的集合的概念可能有所帮助。第一个条件说明 `univ` 是足够大的，第二个条件说明包含足够大集合的集合也是足够大的，第三个条件说明两个足够大集合的交集也是足够大的。

将类型 `X` 上的过滤器视为 `Set X` 的广义元素可能更有用。例如，`atTop` 是“非常大的数集”，而 `𝓝 x₀` 是“非常接近 `x₀` 的点集”。这种观点的一个表现是，我们可以将所谓的“主过滤器”与任何 `s : Set X` 关联起来，该过滤器包含所有包含 `s` 的集合。这个定义已经在 Mathlib 中，并且有一个符号 `𝓟`（在 `Filter` 命名空间中局部化）。为了演示目的，我们要求您利用这个机会在这里推导出这个定义。

```py
def  principal  {α  :  Type*}  (s  :  Set  α)  :  Filter  α
  where
  sets  :=  {  t  |  s  ⊆  t  }
  univ_sets  :=  sorry
  sets_of_superset  :=  sorry
  inter_sets  :=  sorry 
```

对于我们的第二个例子，我们要求您定义过滤器 `atTop : Filter ℕ`。（我们可以用任何具有偏序的 `ℕ` 替代。）

```py
example  :  Filter  ℕ  :=
  {  sets  :=  {  s  |  ∃  a,  ∀  b,  a  ≤  b  →  b  ∈  s  }
  univ_sets  :=  sorry
  sets_of_superset  :=  sorry
  inter_sets  :=  sorry  } 
```

我们还可以直接定义任何 `x : ℝ` 的邻域过滤器 `𝓝 x`。在实数中，`x` 的邻域是一个包含开区间 $(x_0 - \varepsilon, x_0 + \varepsilon)$ 的集合，在 Mathlib 中定义为 `Ioo (x₀ - ε) (x₀ + ε)`。（这种邻域的概念只是 Mathlib 中更一般构造的一个特例。）

通过这些例子，我们已经在以下方面定义了函数 `f : X → Y` 沿着某个 `F : Filter X` 收敛到某个 `G : Filter Y` 的含义：

```py
def  Tendsto₁  {X  Y  :  Type*}  (f  :  X  →  Y)  (F  :  Filter  X)  (G  :  Filter  Y)  :=
  ∀  V  ∈  G,  f  ⁻¹'  V  ∈  F 
```

当 `X` 是 `ℕ` 且 `Y` 是 `ℝ` 时，`Tendsto₁ u atTop (𝓝 x)` 等价于说序列 `u : ℕ → ℝ` 收敛到实数 `x`。当 `X` 和 `Y` 都是 `ℝ` 时，`Tendsto f (𝓝 x₀) (𝓝 y₀)` 等价于熟悉的极限概念 $\lim_{x \to x₀} f(x) = y₀$。介绍中提到的所有其他类型的极限也等价于在源和目标上选择合适的过滤器时的 `Tendsto₁` 的实例。

上述的 `Tendsto₁` 概念在定义上等同于 Mathlib 中定义的 `Tendsto` 概念，但后者定义得更为抽象。`Tendsto₁` 定义的缺点是它暴露了量词和 `G` 的元素，并隐藏了我们通过将过滤器视为广义集合所获得的直观感受。我们可以通过使用更多的代数和集合论工具来隐藏量词 `∀ V` 并使直观感受更加明显。第一个要素是与任何映射 `f : X → Y` 相关的 *前推* 操作 $f_*$，在 Mathlib 中表示为 `Filter.map f`。给定 `X` 上的过滤器 `F`，`Filter.map f F : Filter Y` 被定义为使得 `V ∈ Filter.map f F ↔ f ⁻¹' V ∈ F` 成立。在示例文件中，我们已经打开了 `Filter` 命名空间，以便可以将 `Filter.map` 写作 `map`。这意味着我们可以使用 `Filter Y` 上的顺序关系来重写 `Tendsto` 的定义，这种顺序关系是成员集的逆包含。换句话说，给定 `G H : Filter Y`，我们有 `G ≤ H ↔ ∀ V : Set Y, V ∈ H → V ∈ G`。

```py
def  Tendsto₂  {X  Y  :  Type*}  (f  :  X  →  Y)  (F  :  Filter  X)  (G  :  Filter  Y)  :=
  map  f  F  ≤  G

example  {X  Y  :  Type*}  (f  :  X  →  Y)  (F  :  Filter  X)  (G  :  Filter  Y)  :
  Tendsto₂  f  F  G  ↔  Tendsto₁  f  F  G  :=
  Iff.rfl 
```

可能看起来过滤器上的顺序关系是相反的。但回想一下，我们可以通过 `𝓟 : Set X → Filter X` 的包含来将 `X` 上的过滤器视为 `Set X` 的广义元素，该映射将任何集合 `s` 映射到相应的本原过滤器。这个包含是顺序保持的，因此 `Filter` 上的顺序关系确实可以被视为广义集合之间的自然包含关系。在这个类比中，前推类似于直接像。实际上，`map f (𝓟 s) = 𝓟 (f '' s)`。

现在，我们可以直观地理解为什么序列 `u : ℕ → ℝ` 收敛到点 `x₀` 当且仅当我们有 `map u atTop ≤ 𝓝 x₀`。不等式意味着“在 `u` 下的直接像”的“非常大的自然数集”包含在“非常接近 `x₀` 的点集”中。

如承诺的那样，`Tendsto₂` 的定义没有展示任何量词或集合。它还利用了前推操作代数性质。首先，每个 `Filter.map f` 是单调的。其次，`Filter.map` 与组合兼容。

```py
#check  (@Filter.map_mono  :  ∀  {α  β}  {m  :  α  →  β},  Monotone  (map  m))

#check
  (@Filter.map_map  :
  ∀  {α  β  γ}  {f  :  Filter  α}  {m  :  α  →  β}  {m'  :  β  →  γ},  map  m'  (map  m  f)  =  map  (m'  ∘  m)  f) 
```

这两个性质共同允许我们证明极限可以组合，一次产生介绍中描述的 512 种组合公理变体，以及更多。你可以使用 `Tendsto₁` 的关于全称量词的定义或代数定义，以及上述两个引理来练习证明以下陈述。

```py
example  {X  Y  Z  :  Type*}  {F  :  Filter  X}  {G  :  Filter  Y}  {H  :  Filter  Z}  {f  :  X  →  Y}  {g  :  Y  →  Z}
  (hf  :  Tendsto₁  f  F  G)  (hg  :  Tendsto₁  g  G  H)  :  Tendsto₁  (g  ∘  f)  F  H  :=
  sorry 
```

前推构造使用映射将过滤器从映射源推送到映射目标。还有一个 *后推* 操作，`Filter.comap`，朝相反的方向进行。这推广了集合上的前像操作。对于任何映射 `f`，`Filter.map f` 和 `Filter.comap f` 形成了所谓的 *Galois 连接*，也就是说，它们满足

> `Filter.map_le_iff_le_comap : Filter.map f F ≤ G ↔ F ≤ Filter.comap f G`

对于每个 `F` 和 `G`。这个运算可以用来提供 `Tendsto` 的另一种表述，这将证明上是（但不是定义上）等价于 Mathlib 中的表述。

`comap` 运算可以用来将过滤器限制为子类型。例如，假设我们有 `f : ℝ → ℝ`、`x₀ : ℝ` 和 `y₀ : ℝ`，并且假设我们想要说明当 `x` 在有理数中接近 `x₀` 时，`f x` 接近 `y₀`。我们可以使用强制映射 `(↑) : ℚ → ℝ` 将过滤器 `𝓝 x₀` 反向映射到 `ℚ`，并声明 `Tendsto (f ∘ (↑) : ℚ → ℝ) (comap (↑) (𝓝 x₀)) (𝓝 y₀)`。

```py
variable  (f  :  ℝ  →  ℝ)  (x₀  y₀  :  ℝ)

#check  comap  ((↑)  :  ℚ  →  ℝ)  (𝓝  x₀)

#check  Tendsto  (f  ∘  (↑))  (comap  ((↑)  :  ℚ  →  ℝ)  (𝓝  x₀))  (𝓝  y₀) 
```

反函数运算也与组合兼容，但它具有反变性质，也就是说，它反转了参数的顺序。

```py
section
variable  {α  β  γ  :  Type*}  (F  :  Filter  α)  {m  :  γ  →  β}  {n  :  β  →  α}

#check  (comap_comap  :  comap  m  (comap  n  F)  =  comap  (n  ∘  m)  F)

end 
```

现在我们将注意力转向平面 `ℝ × ℝ`，并尝试理解点 `(x₀, y₀)` 的邻域如何与 `𝓝 x₀` 和 `𝓝 y₀` 相关。存在一个乘法运算 `Filter.prod : Filter X → Filter Y → Filter (X × Y)`，表示为 `×ˢ`，它回答了这个问题：

```py
example  :  𝓝  (x₀,  y₀)  =  𝓝  x₀  ×ˢ  𝓝  y₀  :=
  nhds_prod_eq 
```

乘法运算定义为反函数运算和 `inf` 运算的术语：

> `F ×ˢ G = (comap Prod.fst F) ⊓ (comap Prod.snd G)`。

这里 `inf` 运算指的是任何类型 `X` 上 `Filter X` 的格结构，其中 `F ⊓ G` 是小于 `F` 和 `G` 的最大过滤器。因此，`inf` 运算推广了集合交集的概念。

Mathlib 中许多证明都使用了上述所有结构（`map`、`comap`、`inf`、`sup` 和 `prod`）来给出关于收敛性的代数证明，而从未引用过滤器的成员。你可以在以下引理的证明中练习这样做，如果需要，可以展开 `Tendsto` 和 `Filter.prod` 的定义。

```py
#check  le_inf_iff

example  (f  :  ℕ  →  ℝ  ×  ℝ)  (x₀  y₀  :  ℝ)  :
  Tendsto  f  atTop  (𝓝  (x₀,  y₀))  ↔
  Tendsto  (Prod.fst  ∘  f)  atTop  (𝓝  x₀)  ∧  Tendsto  (Prod.snd  ∘  f)  atTop  (𝓝  y₀)  :=
  sorry 
```

有序类型 `Filter X` 实际上是一个 *完备* 拓扑，也就是说，存在一个底元素，存在一个顶元素，并且 `X` 上的每个过滤器集合都有一个 `Inf` 和一个 `Sup`。

注意，根据过滤器定义的第二性质（如果 `U` 属于 `F`，则任何大于 `U` 的东西也属于 `F`），第一个性质（`X` 的所有居民集合属于 `F`）等价于 `F` 不是空集合的性质。这不应与更微妙的问题混淆，即空集是否是 `F` 的 *元素*。过滤器的定义并不禁止 `∅ ∈ F`，但如果空集在 `F` 中，则每个集合都在 `F` 中，也就是说，`∀ U : Set X, U ∈ F`。在这种情况下，`F` 是一个非常平凡的过滤器，这正是完备格 `Filter X` 的底元素。这与布尔巴基关于过滤器定义不同，布尔巴基的定义不允许包含空集的过滤器。

由于我们在定义中包含了平凡滤波器，我们有时需要明确假设某些引理的非平凡性。然而，作为回报，理论具有更好的全局性质。我们已经看到，包括平凡滤波器给我们提供了一个底元素。它还允许我们定义`principal : Set X → Filter X`，它将`∅`映射到`⊥`，而不需要添加一个先决条件来排除空集。它还允许我们定义没有先决条件的拉回操作。实际上，可能会发生`comap f F = ⊥`，尽管`F ≠ ⊥`。例如，给定`x₀ : ℝ`和`s : Set ℝ`，如果`x₀`属于`s`的闭包，那么从对应于`s`的子类型强制转换的`𝓝 x₀`的拉回是非平凡的。

为了管理需要假设某些滤波器非平凡的引理，Mathlib 有一个类型类`Filter.NeBot`，库中有假设`(F : Filter X) [F.NeBot]`的引理。实例数据库知道，例如，`(atTop : Filter ℕ).NeBot`，它知道前推一个非平凡滤波器会得到一个非平凡滤波器。因此，假设`[F.NeBot]`的引理将自动适用于任何序列`u`的`map u atTop`。

我们对滤波器的代数性质及其与极限的关系的考察基本上已经完成，但我们还没有证明我们重新获得了通常的极限概念。表面上，`Tendsto u atTop (𝓝 x₀)`似乎比第 3.6 节中定义的收敛概念更强，因为我们要求`x₀`的每个邻域都有一个属于`atTop`的逆像，而通常的定义只要求对于标准邻域`Ioo (x₀ - ε) (x₀ + ε)`是这样的。关键是，根据定义，每个邻域都包含这样的标准邻域。这个观察导致了一个**滤波基**的概念。

给定`F : Filter X`，一个集合族`s : ι → Set X`是`F`的基，如果对于每个集合`U`，当且仅当它包含某个`s i`时，`U ∈ F`。换句话说，从形式上讲，如果`s`满足`∀ U : Set X, U ∈ F ↔ ∃ i, s i ⊆ U`，则`s`是一个基。考虑一个在`ι`上的谓词，它只选择索引类型中的某些值`i`，这甚至更加灵活。在`𝓝 x₀`的情况下，我们希望`ι`是`ℝ`，我们用`ε`表示`i`，谓词应该选择`ε`的正值。因此，集合`Ioo  (x₀ - ε) (x₀ + ε)`形成`ℝ`上的邻域拓扑的基，可以这样表述：

```py
example  (x₀  :  ℝ)  :  HasBasis  (𝓝  x₀)  (fun  ε  :  ℝ  ↦  0  <  ε)  fun  ε  ↦  Ioo  (x₀  -  ε)  (x₀  +  ε)  :=
  nhds_basis_Ioo_pos  x₀ 
```

对于`atTop`滤波器也有一个很好的基。引理`Filter.HasBasis.tendsto_iff`允许我们用`F`和`G`的基重新表述形式为`Tendsto f F G`的陈述。将这些部分放在一起，我们基本上得到了第 3.6 节中使用的收敛概念。

```py
example  (u  :  ℕ  →  ℝ)  (x₀  :  ℝ)  :
  Tendsto  u  atTop  (𝓝  x₀)  ↔  ∀  ε  >  0,  ∃  N,  ∀  n  ≥  N,  u  n  ∈  Ioo  (x₀  -  ε)  (x₀  +  ε)  :=  by
  have  :  atTop.HasBasis  (fun  _  :  ℕ  ↦  True)  Ici  :=  atTop_basis
  rw  [this.tendsto_iff  (nhds_basis_Ioo_pos  x₀)]
  simp 
```

我们现在展示如何过滤器有助于处理对于足够大的数字或对于足够接近给定点的点所持有的属性。在第 3.6 节中，我们经常遇到这样的情况：我们知道某些属性 `P n` 对于足够大的 `n` 成立，而某些其他属性 `Q n` 对于足够大的 `n` 成立。使用 `cases` 两次给出了满足 `∀ n ≥ N_P, P n` 和 `∀ n ≥ N_Q, Q n` 的 `N_P` 和 `N_Q`。使用 `set N := max N_P N_Q`，我们最终可以证明 `∀ n ≥ N, P n ∧ Q n`。这样做反复进行会变得令人厌烦。

我们可以通过注意到陈述“`P n` 和 `Q n` 对于足够大的 `n` 成立”意味着我们拥有 `{n | P n} ∈ atTop` 和 `{n | Q n} ∈ atTop` 来做得更好。`atTop` 是一个过滤器的性质意味着 `atTop` 中两个元素的交集再次在 `atTop` 中，因此我们得到 `{n | P n ∧ Q n} ∈ atTop`。写作 `{n | P n} ∈ atTop` 是不愉快的，但我们可以使用更具说明性的符号 `∀ᶠ n in atTop, P n`。这里上标的 `f` 代表“过滤器”。你可以将这个符号理解为对于所有在“非常大的数字集合”中的 `n`，`P n` 成立。`∀ᶠ` 符号代表 `Filter.Eventually`，而 `Filter.Eventually.and` 引理使用过滤器的交集性质来完成我们刚才描述的事情：

```py
example  (P  Q  :  ℕ  →  Prop)  (hP  :  ∀ᶠ  n  in  atTop,  P  n)  (hQ  :  ∀ᶠ  n  in  atTop,  Q  n)  :
  ∀ᶠ  n  in  atTop,  P  n  ∧  Q  n  :=
  hP.and  hQ 
```

这个符号如此方便且直观，以至于当 `P` 是一个等式或不等式陈述时，我们也有专门的实现。例如，设 `u` 和 `v` 是两个实数序列，我们将证明如果 `u n` 和 `v n` 对于足够大的 `n` 一致，那么 `u` 趋于 `x₀` 当且仅当 `v` 趋于 `x₀`。首先我们将使用通用的 `Eventually`，然后是专门针对等价谓词的 `EventuallyEq`。这两个陈述在定义上是等价的，所以两种情况下的证明工作相同。

```py
example  (u  v  :  ℕ  →  ℝ)  (h  :  ∀ᶠ  n  in  atTop,  u  n  =  v  n)  (x₀  :  ℝ)  :
  Tendsto  u  atTop  (𝓝  x₀)  ↔  Tendsto  v  atTop  (𝓝  x₀)  :=
  tendsto_congr'  h

example  (u  v  :  ℕ  →  ℝ)  (h  :  u  =ᶠ[atTop]  v)  (x₀  :  ℝ)  :
  Tendsto  u  atTop  (𝓝  x₀)  ↔  Tendsto  v  atTop  (𝓝  x₀)  :=
  tendsto_congr'  h 
```

通过 `Eventually` 来回顾过滤器的定义是有教育意义的。给定 `F : Filter X`，对于 `X` 上的任何谓词 `P` 和 `Q`，

+   条件 `univ ∈ F` 确保了 `(∀ x, P x) → ∀ᶠ x in F, P x`。

+   条件 `U ∈ F → U ⊆ V → V ∈ F` 确保了 `(∀ᶠ x in F, P x) → (∀ x, P x → Q x) → ∀ᶠ x in F, Q x`，并且

+   条件 `U ∈ F → V ∈ F → U ∩ V ∈ F` 确保了 `(∀ᶠ x in F, P x) → (∀ᶠ x in F, Q x) → ∀ᶠ x in F, P x ∧ Q x`.

```py
#check  Eventually.of_forall
#check  Eventually.mono
#check  Eventually.and 
```

第二项，对应于 `Eventually.mono`，支持使用过滤器的优雅方式，特别是当与 `Eventually.and` 结合使用时。`filter_upwards` 策略允许我们组合它们。比较：

```py
example  (P  Q  R  :  ℕ  →  Prop)  (hP  :  ∀ᶠ  n  in  atTop,  P  n)  (hQ  :  ∀ᶠ  n  in  atTop,  Q  n)
  (hR  :  ∀ᶠ  n  in  atTop,  P  n  ∧  Q  n  →  R  n)  :  ∀ᶠ  n  in  atTop,  R  n  :=  by
  apply  (hP.and  (hQ.and  hR)).mono
  rintro  n  ⟨h,  h',  h''⟩
  exact  h''  ⟨h,  h'⟩

example  (P  Q  R  :  ℕ  →  Prop)  (hP  :  ∀ᶠ  n  in  atTop,  P  n)  (hQ  :  ∀ᶠ  n  in  atTop,  Q  n)
  (hR  :  ∀ᶠ  n  in  atTop,  P  n  ∧  Q  n  →  R  n)  :  ∀ᶠ  n  in  atTop,  R  n  :=  by
  filter_upwards  [hP,  hQ,  hR]  with  n  h  h'  h''
  exact  h''  ⟨h,  h'⟩ 
```

了解测度理论的读者会注意到，集合的补集具有零测度（即“几乎每个点的集合”）的过滤器 `μ.ae` 作为 `Tendsto` 的源或目标并不是非常有用，但它可以方便地与 `Eventually` 结合使用，以表明一个属性对于几乎每个点都成立。

存在一个 `∀ᶠ x in F, P x` 的对偶版本，这在某些情况下是有用的：`∃ᶠ x in F, P x` 意味着 `{x | ¬P x} ∉ F`。例如，`∃ᶠ n in atTop, P n` 意味着存在任意大的 `n` 使得 `P n` 成立。`∃ᶠ` 符号代表 `Filter.Frequently`。

对于一个更复杂的例子，考虑以下关于序列 `u`、集合 `M` 和值 `x` 的陈述：

> 如果 `u` 收敛到 `x` 并且对于足够大的 `n`，`u n` 属于 `M`，那么 `x` 在 `M` 的闭集中。

这可以形式化为以下内容：

> `Tendsto u atTop (𝓝 x) → (∀ᶠ n in atTop, u n ∈ M) → x ∈ closure M`。

这是拓扑库中 `mem_closure_of_tendsto` 定理的一个特例。看看你是否可以使用引用的引理来证明它，利用 `ClusterPt x F` 表示 `(𝓝 x ⊓ F).NeBot` 以及根据定义，假设 `∀ᶠ n in atTop, u n ∈ M` 意味着 `M ∈ map u atTop`。

```py
#check  mem_closure_iff_clusterPt
#check  le_principal_iff
#check  neBot_of_le

example  (u  :  ℕ  →  ℝ)  (M  :  Set  ℝ)  (x  :  ℝ)  (hux  :  Tendsto  u  atTop  (𝓝  x))
  (huM  :  ∀ᶠ  n  in  atTop,  u  n  ∈  M)  :  x  ∈  closure  M  :=
  sorry 
```  ## 11.2\. 距离空间

上一节中的例子主要关注实数序列。在本节中，我们将提高一点普遍性，并关注度量空间。度量空间是一个带有距离函数 `dist : X → X → ℝ` 的类型 `X`，这是从 `X = ℝ` 的情况下的函数 `fun x y ↦ |x - y|` 的一般化。

引入这样的空间很容易，我们将检查从距离函数所需的所有属性。

```py
variable  {X  :  Type*}  [MetricSpace  X]  (a  b  c  :  X)

#check  (dist  a  b  :  ℝ)
#check  (dist_nonneg  :  0  ≤  dist  a  b)
#check  (dist_eq_zero  :  dist  a  b  =  0  ↔  a  =  b)
#check  (dist_comm  a  b  :  dist  a  b  =  dist  b  a)
#check  (dist_triangle  a  b  c  :  dist  a  c  ≤  dist  a  b  +  dist  b  c) 
```

注意，我们还有距离可以无限或 `dist a b` 可以是零而没有 `a = b` 或两者都为零的变体。它们分别称为 `EMetricSpace`、`PseudoMetricSpace` 和 `PseudoEMetricSpace`（这里的“e”代表“扩展”）。

注意，我们的从 `ℝ` 到度量空间之旅跳过了需要线性代数的特殊情况的范空间，这将在微积分章节中解释。

### 11.2.1\. 收敛性和连续性

使用距离函数，我们可以在度量空间之间定义收敛序列和连续函数。实际上，它们在下一节中更一般的设置中定义，但我们有将定义重新表述为距离的引理。

```py
example  {u  :  ℕ  →  X}  {a  :  X}  :
  Tendsto  u  atTop  (𝓝  a)  ↔  ∀  ε  >  0,  ∃  N,  ∀  n  ≥  N,  dist  (u  n)  a  <  ε  :=
  Metric.tendsto_atTop

example  {X  Y  :  Type*}  [MetricSpace  X]  [MetricSpace  Y]  {f  :  X  →  Y}  :
  Continuous  f  ↔
  ∀  x  :  X,  ∀  ε  >  0,  ∃  δ  >  0,  ∀  x',  dist  x'  x  <  δ  →  dist  (f  x')  (f  x)  <  ε  :=
  Metric.continuous_iff 
```

许多引理都有一些连续性的假设，所以我们最终证明了许多连续性结果，并且有一个专门用于这个任务的 `continuity` 策略。让我们证明一个将在下面的练习中需要的连续性陈述。注意，Lean 知道如何将两个度量空间的乘积视为度量空间，因此考虑从 `X × X` 到 `ℝ` 的连续函数是有意义的。特别是（未展开的）距离函数是这样的函数。

```py
example  {X  Y  :  Type*}  [MetricSpace  X]  [MetricSpace  Y]  {f  :  X  →  Y}  (hf  :  Continuous  f)  :
  Continuous  fun  p  :  X  ×  X  ↦  dist  (f  p.1)  (f  p.2)  :=  by  continuity 
```

这个策略有点慢，因此了解如何手动完成它也很有用。我们首先需要使用 `fun p : X × X ↦ f p.1` 是连续的，因为它是由 `f` 组成的，`f` 是根据假设 `hf` 连续的，以及投影 `prod.fst` 的连续性，这是引理 `continuous_fst` 的内容。组合属性是 `Continuous.comp`，它在 `Continuous` 命名空间中，因此我们可以使用点符号将 `Continuous.comp hf continuous_fst` 压缩为 `hf.comp continuous_fst`，这实际上更易于阅读，因为它真正地读作组合我们的假设和我们的引理。我们可以对第二个分量做同样的处理，以得到 `fun p : X × X ↦ f p.2` 的连续性。然后我们使用 `Continuous.prod_mk` 将这两个连续性组装起来，得到 `(hf.comp continuous_fst).prod_mk (hf.comp continuous_snd) : Continuous (fun p : X × X ↦ (f p.1, f p.2))`，然后再进行一次组合，以得到完整的证明。

```py
example  {X  Y  :  Type*}  [MetricSpace  X]  [MetricSpace  Y]  {f  :  X  →  Y}  (hf  :  Continuous  f)  :
  Continuous  fun  p  :  X  ×  X  ↦  dist  (f  p.1)  (f  p.2)  :=
  continuous_dist.comp  ((hf.comp  continuous_fst).prodMk  (hf.comp  continuous_snd)) 
```

通过 `Continuous.comp` 将 `Continuous.prod_mk` 和 `continuous_dist` 结合起来感觉有些笨拙，即使像上面那样大量使用点符号也是如此。一个更严重的问题是，这个漂亮的证明需要大量的规划。Lean 接受上述证明项，因为它是一个完整的项，证明了与我们的目标定义等价的说法，关键的定义展开是函数的组合。确实，我们的目标函数 `fun p : X × X ↦ dist (f p.1) (f p.2)` 并没有以组合的形式呈现。我们提供的证明项证明了 `dist ∘ (fun p : X × X ↦ (f p.1, f p.2))` 的连续性，这恰好与我们的目标函数定义等价。但如果我们尝试从 `apply continuous_dist.comp` 开始逐步构建这个证明，Lean 的展开器将无法识别组合并拒绝应用这个引理。当涉及到类型乘积时，这个问题尤其严重。

在这里应用更好的引理是 `Continuous.dist {f g : X → Y} : Continuous f → Continuous g → Continuous (fun x ↦ dist (f x) (g x))`，这对 Lean 的展开器来说更友好，并且在直接提供完整证明项时也提供了更短的证明，如下面的两个新证明所示：

```py
example  {X  Y  :  Type*}  [MetricSpace  X]  [MetricSpace  Y]  {f  :  X  →  Y}  (hf  :  Continuous  f)  :
  Continuous  fun  p  :  X  ×  X  ↦  dist  (f  p.1)  (f  p.2)  :=  by
  apply  Continuous.dist
  exact  hf.comp  continuous_fst
  exact  hf.comp  continuous_snd

example  {X  Y  :  Type*}  [MetricSpace  X]  [MetricSpace  Y]  {f  :  X  →  Y}  (hf  :  Continuous  f)  :
  Continuous  fun  p  :  X  ×  X  ↦  dist  (f  p.1)  (f  p.2)  :=
  (hf.comp  continuous_fst).dist  (hf.comp  continuous_snd) 
```

注意，如果没有来自函数组合的详细阐述问题，另一种压缩我们证明的方法是使用 `Continuous.prod_map`，这在某些情况下很有用，并给出了另一个证明项 `continuous_dist.comp (hf.prod_map hf)`，这甚至更短于手动输入。

由于在详细阐述和简短输入之间做出选择都很令人沮丧，让我们用 `Continuous.fst'` 提供的最后一点压缩来结束这次讨论，它允许我们将 `hf.comp continuous_fst` 压缩为 `hf.fst'`（以及 `snd` 的相同处理），并得到我们的最终证明，现在几乎接近晦涩难懂了。

```py
example  {X  Y  :  Type*}  [MetricSpace  X]  [MetricSpace  Y]  {f  :  X  →  Y}  (hf  :  Continuous  f)  :
  Continuous  fun  p  :  X  ×  X  ↦  dist  (f  p.1)  (f  p.2)  :=
  hf.fst'.dist  hf.snd' 
```

现在轮到你来证明一些连续性引理了。在尝试了连续性策略之后，你需要使用 `Continuous.add`、`continuous_pow` 和 `continuous_id` 来手动完成它。

```py
example  {f  :  ℝ  →  X}  (hf  :  Continuous  f)  :  Continuous  fun  x  :  ℝ  ↦  f  (x  ^  2  +  x)  :=
  sorry 
```

到目前为止，我们看到了连续性作为一个全局概念，但也可以定义在一点上的连续性。

```py
example  {X  Y  :  Type*}  [MetricSpace  X]  [MetricSpace  Y]  (f  :  X  →  Y)  (a  :  X)  :
  ContinuousAt  f  a  ↔  ∀  ε  >  0,  ∃  δ  >  0,  ∀  {x},  dist  x  a  <  δ  →  dist  (f  x)  (f  a)  <  ε  :=
  Metric.continuousAt_iff 
```

### 11.2.2\. 球、开集和闭集

一旦我们有了距离函数，最重要的几何定义就是（开）球和闭球。

```py
variable  (r  :  ℝ)

example  :  Metric.ball  a  r  =  {  b  |  dist  b  a  <  r  }  :=
  rfl

example  :  Metric.closedBall  a  r  =  {  b  |  dist  b  a  ≤  r  }  :=
  rfl 
```

注意，这里的 r 是任何实数，没有符号限制。当然，有些陈述确实需要半径条件。

```py
example  (hr  :  0  <  r)  :  a  ∈  Metric.ball  a  r  :=
  Metric.mem_ball_self  hr

example  (hr  :  0  ≤  r)  :  a  ∈  Metric.closedBall  a  r  :=
  Metric.mem_closedBall_self  hr 
```

一旦我们有了球，我们就可以定义开集。实际上，它们在下一节中更一般的设置中被定义，但我们有引理将定义重新表述为球的形式。

```py
example  (s  :  Set  X)  :  IsOpen  s  ↔  ∀  x  ∈  s,  ∃  ε  >  0,  Metric.ball  x  ε  ⊆  s  :=
  Metric.isOpen_iff 
```

然后闭集是补集为开集的集合。它们的重要性质是它们在极限下是封闭的。集合的闭包是包含它的最小闭集。

```py
example  {s  :  Set  X}  :  IsClosed  s  ↔  IsOpen  (sᶜ)  :=
  isOpen_compl_iff.symm

example  {s  :  Set  X}  (hs  :  IsClosed  s)  {u  :  ℕ  →  X}  (hu  :  Tendsto  u  atTop  (𝓝  a))
  (hus  :  ∀  n,  u  n  ∈  s)  :  a  ∈  s  :=
  hs.mem_of_tendsto  hu  (Eventually.of_forall  hus)

example  {s  :  Set  X}  :  a  ∈  closure  s  ↔  ∀  ε  >  0,  ∃  b  ∈  s,  a  ∈  Metric.ball  b  ε  :=
  Metric.mem_closure_iff 
```

在不使用`mem_closure_iff_seq_limit`的情况下完成下一个练习

```py
example  {u  :  ℕ  →  X}  (hu  :  Tendsto  u  atTop  (𝓝  a))  {s  :  Set  X}  (hs  :  ∀  n,  u  n  ∈  s)  :
  a  ∈  closure  s  :=  by
  sorry 
```

记住从过滤器部分，邻域过滤器在 Mathlib 中起着重要作用。在度量空间上下文中，关键点是球为这些过滤器提供基。这里的主要引理是`Metric.nhds_basis_ball`和`Metric.nhds_basis_closedBall`，它们为正半径的开球和闭球断言这一点。中心点是隐含的参数，因此我们可以像以下示例中那样调用`Filter.HasBasis.mem_iff`。

```py
example  {x  :  X}  {s  :  Set  X}  :  s  ∈  𝓝  x  ↔  ∃  ε  >  0,  Metric.ball  x  ε  ⊆  s  :=
  Metric.nhds_basis_ball.mem_iff

example  {x  :  X}  {s  :  Set  X}  :  s  ∈  𝓝  x  ↔  ∃  ε  >  0,  Metric.closedBall  x  ε  ⊆  s  :=
  Metric.nhds_basis_closedBall.mem_iff 
```

### 11.2.3\. 紧致性

紧致性是一个重要的拓扑概念。它区分了度量空间中具有与实数线段相同性质的子集与其他区间：

+   任何在紧集内取值的序列都有一个子序列在这个集合中收敛。

+   在非空紧集上定义的任何连续函数，其值在实数中都是有界的，并且在其界限的某处达到（这被称为极值定理）。

+   紧集是闭集。

让我们先检查实数单位区间确实是一个紧集，然后检查一般度量空间中紧集的上述命题。在第二个命题中，我们只需要给定集合上的连续性，所以我们将使用`ContinuousOn`而不是`Continuous`，并且我们将为最小值和最大值给出单独的陈述。当然，所有这些结果都是从更一般的版本中推导出来的，其中一些将在后面的章节中讨论。

```py
example  :  IsCompact  (Set.Icc  0  1  :  Set  ℝ)  :=
  isCompact_Icc

example  {s  :  Set  X}  (hs  :  IsCompact  s)  {u  :  ℕ  →  X}  (hu  :  ∀  n,  u  n  ∈  s)  :
  ∃  a  ∈  s,  ∃  φ  :  ℕ  →  ℕ,  StrictMono  φ  ∧  Tendsto  (u  ∘  φ)  atTop  (𝓝  a)  :=
  hs.tendsto_subseq  hu

example  {s  :  Set  X}  (hs  :  IsCompact  s)  (hs'  :  s.Nonempty)  {f  :  X  →  ℝ}
  (hfs  :  ContinuousOn  f  s)  :
  ∃  x  ∈  s,  ∀  y  ∈  s,  f  x  ≤  f  y  :=
  hs.exists_isMinOn  hs'  hfs

example  {s  :  Set  X}  (hs  :  IsCompact  s)  (hs'  :  s.Nonempty)  {f  :  X  →  ℝ}
  (hfs  :  ContinuousOn  f  s)  :
  ∃  x  ∈  s,  ∀  y  ∈  s,  f  y  ≤  f  x  :=
  hs.exists_isMaxOn  hs'  hfs

example  {s  :  Set  X}  (hs  :  IsCompact  s)  :  IsClosed  s  :=
  hs.isClosed 
```

我们还可以指定度量空间是全局紧致的，使用额外的`Prop`值类型类：

```py
example  {X  :  Type*}  [MetricSpace  X]  [CompactSpace  X]  :  IsCompact  (univ  :  Set  X)  :=
  isCompact_univ 
```

在紧致度量空间中，任何闭集都是紧致的，这是`IsClosed.isCompact`。

### 11.2.4\. 一致连续函数

现在我们转向度量空间上的一致性概念：一致连续函数、柯西序列和完备性。同样，这些都是在更一般的环境中定义的，但我们有关于度量名称空间的引理来访问它们的元素定义。我们首先从一致连续性开始。

```py
example  {X  :  Type*}  [MetricSpace  X]  {Y  :  Type*}  [MetricSpace  Y]  {f  :  X  →  Y}  :
  UniformContinuous  f  ↔
  ∀  ε  >  0,  ∃  δ  >  0,  ∀  {a  b  :  X},  dist  a  b  <  δ  →  dist  (f  a)  (f  b)  <  ε  :=
  Metric.uniformContinuous_iff 
```

为了练习操作所有这些定义，我们将证明从紧致度量空间到度量空间的连续函数是一致连续的（我们将在后面的章节中看到一个更一般的形式）。 

我们首先给出一个非正式的草图。设 `f : X → Y` 是从一个紧致度量空间到度量空间的连续函数。我们固定 `ε > 0` 并开始寻找某个 `δ`。

设 `φ : X × X → ℝ := fun p ↦ dist (f p.1) (f p.2)` 并设 `K := { p : X × X | ε ≤ φ p }`。观察 `φ` 是连续的，因为 `f` 和距离都是连续的。并且 `K` 显然是闭集（使用 `isClosed_le`），因此由于 `X` 是紧致的，`K` 也是紧致的。

然后，我们讨论使用 `eq_empty_or_nonempty` 的两种可能性。如果 `K` 是空的，那么我们显然已经完成了（例如，我们可以将 `δ` 设置为 `1`）。所以让我们假设 `K` 不是空的，并使用极值定理来选择 `(x₀, x₁)`，它在 `K` 上的距离函数上达到下确界。然后我们可以将 `δ` 设置为 `dist x₀ x₁` 并检查一切是否正常工作。

```py
example  {X  :  Type*}  [MetricSpace  X]  [CompactSpace  X]
  {Y  :  Type*}  [MetricSpace  Y]  {f  :  X  →  Y}
  (hf  :  Continuous  f)  :  UniformContinuous  f  :=  by
  sorry 
```

### 11.2.5\. 完备性

在度量空间中的一个柯西序列是一个其项逐渐接近彼此的序列。有几种等价的方式来表述这个想法。特别是收敛序列是柯西序列。逆命题仅在所谓的 *完备* 空间中成立。

```py
example  (u  :  ℕ  →  X)  :
  CauchySeq  u  ↔  ∀  ε  >  0,  ∃  N  :  ℕ,  ∀  m  ≥  N,  ∀  n  ≥  N,  dist  (u  m)  (u  n)  <  ε  :=
  Metric.cauchySeq_iff

example  (u  :  ℕ  →  X)  :
  CauchySeq  u  ↔  ∀  ε  >  0,  ∃  N  :  ℕ,  ∀  n  ≥  N,  dist  (u  n)  (u  N)  <  ε  :=
  Metric.cauchySeq_iff'

example  [CompleteSpace  X]  (u  :  ℕ  →  X)  (hu  :  CauchySeq  u)  :
  ∃  x,  Tendsto  u  atTop  (𝓝  x)  :=
  cauchySeq_tendsto_of_complete  hu 
```

我们将通过证明一个方便的判据来练习使用这个定义，这是 Mathlib 中出现的一个判据的特殊情况。这也是练习在几何环境中使用大和的好机会。除了过滤器部分的解释外，你可能还需要 `tendsto_pow_atTop_nhds_zero_of_lt_one`，`Tendsto.mul` 和 `dist_le_range_sum_dist`。

```py
theorem  cauchySeq_of_le_geometric_two'  {u  :  ℕ  →  X}
  (hu  :  ∀  n  :  ℕ,  dist  (u  n)  (u  (n  +  1))  ≤  (1  /  2)  ^  n)  :  CauchySeq  u  :=  by
  rw  [Metric.cauchySeq_iff']
  intro  ε  ε_pos
  obtain  ⟨N,  hN⟩  :  ∃  N  :  ℕ,  1  /  2  ^  N  *  2  <  ε  :=  by  sorry
  use  N
  intro  n  hn
  obtain  ⟨k,  rfl  :  n  =  N  +  k⟩  :=  le_iff_exists_add.mp  hn
  calc
  dist  (u  (N  +  k))  (u  N)  =  dist  (u  (N  +  0))  (u  (N  +  k))  :=  sorry
  _  ≤  ∑  i  ∈  range  k,  dist  (u  (N  +  i))  (u  (N  +  (i  +  1)))  :=  sorry
  _  ≤  ∑  i  ∈  range  k,  (1  /  2  :  ℝ)  ^  (N  +  i)  :=  sorry
  _  =  1  /  2  ^  N  *  ∑  i  ∈  range  k,  (1  /  2  :  ℝ)  ^  i  :=  sorry
  _  ≤  1  /  2  ^  N  *  2  :=  sorry
  _  <  ε  :=  sorry 
```

我们已经准备好本节的最终挑战：完备度量空间的 Baire 定理！下面的证明框架显示了有趣的技术。它使用感叹号变体的 `choose` 策略（你应该尝试移除这个感叹号）并展示了如何在证明中使用 `Nat.rec_on` 递归地定义某物。

```py
open  Metric

example  [CompleteSpace  X]  (f  :  ℕ  →  Set  X)  (ho  :  ∀  n,  IsOpen  (f  n))  (hd  :  ∀  n,  Dense  (f  n))  :
  Dense  (⋂  n,  f  n)  :=  by
  let  B  :  ℕ  →  ℝ  :=  fun  n  ↦  (1  /  2)  ^  n
  have  Bpos  :  ∀  n,  0  <  B  n
  sorry
  /- Translate the density assumption into two functions `center` and `radius` associating
 to any n, x, δ, δpos a center and a positive radius such that
 `closedBall center radius` is included both in `f n` and in `closedBall x δ`.
 We can also require `radius ≤ (1/2)^(n+1)`, to ensure we get a Cauchy sequence later. -/
  have  :
  ∀  (n  :  ℕ)  (x  :  X),
  ∀  δ  >  0,  ∃  y  :  X,  ∃  r  >  0,  r  ≤  B  (n  +  1)  ∧  closedBall  y  r  ⊆  closedBall  x  δ  ∩  f  n  :=
  by  sorry
  choose!  center  radius  Hpos  HB  Hball  using  this
  intro  x
  rw  [mem_closure_iff_nhds_basis  nhds_basis_closedBall]
  intro  ε  εpos
  /- `ε` is positive. We have to find a point in the ball of radius `ε` around `x`
 belonging to all `f n`. For this, we construct inductively a sequence
 `F n = (c n, r n)` such that the closed ball `closedBall (c n) (r n)` is included
 in the previous ball and in `f n`, and such that `r n` is small enough to ensure
 that `c n` is a Cauchy sequence. Then `c n` converges to a limit which belongs
 to all the `f n`. -/
  let  F  :  ℕ  →  X  ×  ℝ  :=  fun  n  ↦
  Nat.recOn  n  (Prod.mk  x  (min  ε  (B  0)))
  fun  n  p  ↦  Prod.mk  (center  n  p.1  p.2)  (radius  n  p.1  p.2)
  let  c  :  ℕ  →  X  :=  fun  n  ↦  (F  n).1
  let  r  :  ℕ  →  ℝ  :=  fun  n  ↦  (F  n).2
  have  rpos  :  ∀  n,  0  <  r  n  :=  by  sorry
  have  rB  :  ∀  n,  r  n  ≤  B  n  :=  by  sorry
  have  incl  :  ∀  n,  closedBall  (c  (n  +  1))  (r  (n  +  1))  ⊆  closedBall  (c  n)  (r  n)  ∩  f  n  :=  by
  sorry
  have  cdist  :  ∀  n,  dist  (c  n)  (c  (n  +  1))  ≤  B  n  :=  by  sorry
  have  :  CauchySeq  c  :=  cauchySeq_of_le_geometric_two'  cdist
  -- as the sequence `c n` is Cauchy in a complete space, it converges to a limit `y`.
  rcases  cauchySeq_tendsto_of_complete  this  with  ⟨y,  ylim⟩
  -- this point `y` will be the desired point. We will check that it belongs to all
  -- `f n` and to `ball x ε`.
  use  y
  have  I  :  ∀  n,  ∀  m  ≥  n,  closedBall  (c  m)  (r  m)  ⊆  closedBall  (c  n)  (r  n)  :=  by  sorry
  have  yball  :  ∀  n,  y  ∈  closedBall  (c  n)  (r  n)  :=  by  sorry
  sorry 
```  ## 11.3\. 拓扑空间

### 11.3.1\. 基础

现在我们提高一般性，引入拓扑空间。我们将回顾定义拓扑空间的两种主要方式，然后解释拓扑空间范畴比度量空间范畴表现得更好。请注意，我们在这里不会使用 Mathlib 的范畴论，而只是有一个某种范畴的观点。

思考从度量空间到拓扑空间的过渡的第一种方式是，我们只记住开集的概念（或者等价地，闭集的概念）。从这一观点来看，拓扑空间是一个类型，它配备了一组称为开集的集合。这个集合必须满足下面提出的几个公理（这个集合稍微有些冗余，但我们将忽略这一点）。

```py
section
variable  {X  :  Type*}  [TopologicalSpace  X]

example  :  IsOpen  (univ  :  Set  X)  :=
  isOpen_univ

example  :  IsOpen  (∅  :  Set  X)  :=
  isOpen_empty

example  {ι  :  Type*}  {s  :  ι  →  Set  X}  (hs  :  ∀  i,  IsOpen  (s  i))  :  IsOpen  (⋃  i,  s  i)  :=
  isOpen_iUnion  hs

example  {ι  :  Type*}  [Fintype  ι]  {s  :  ι  →  Set  X}  (hs  :  ∀  i,  IsOpen  (s  i))  :
  IsOpen  (⋂  i,  s  i)  :=
  isOpen_iInter_of_finite  hs 
```

闭集被定义为补集是开集的集合。在拓扑空间之间的函数（全局）连续，如果所有开集的前像都是开集。

```py
variable  {Y  :  Type*}  [TopologicalSpace  Y]

example  {f  :  X  →  Y}  :  Continuous  f  ↔  ∀  s,  IsOpen  s  →  IsOpen  (f  ⁻¹'  s)  :=
  continuous_def 
```

使用这个定义，我们已经开始看到，与度量空间相比，拓扑空间只记得足够的信息来谈论连续函数：一个类型上的两个拓扑结构相同，当且仅当它们有相同的连续函数（实际上，如果两个结构有相同的开集，恒等函数将在两个方向上都是连续的）。

然而，当我们转向点的连续性时，我们看到基于开集的方法的局限性。在 Mathlib 中，我们经常将拓扑空间视为带有附加在每个点 `x` 上的邻域过滤器 `𝓝 x` 的类型（相应的函数 `X → Filter X` 满足某些条件，这些条件将在下面进一步解释）。记住从过滤器部分，这些小玩意儿扮演两个相关的角色。首先，`𝓝 x` 被视为 `X` 中靠近 `x` 的点的广义集合。然后，它被看作提供了一种方式，对于任何谓词 `P : X → Prop`，可以说明这个谓词对于足够靠近 `x` 的点成立。让我们声明 `f : X → Y` 在 `x` 处是连续的。纯粹基于过滤器的说法是，`f` 的直接像包含足够靠近 `x` 的点的广义集合包含在足够靠近 `f x` 的点的广义集合中。回想一下，这可以表示为 `map f (𝓝 x) ≤ 𝓝 (f x)` 或 `Tendsto f (𝓝 x) (𝓝 (f x))`。

```py
example  {f  :  X  →  Y}  {x  :  X}  :  ContinuousAt  f  x  ↔  map  f  (𝓝  x)  ≤  𝓝  (f  x)  :=
  Iff.rfl 
```

也可以使用两种邻域（视为普通集合）和一种邻域过滤器（视为广义集合）来表述：“对于任何 `f x` 的邻域 `U`，所有靠近 `x` 的点都被发送到 `U`”。请注意，证明仍然是 `Iff.rfl`，这个观点在定义上是等同于之前的。

```py
example  {f  :  X  →  Y}  {x  :  X}  :  ContinuousAt  f  x  ↔  ∀  U  ∈  𝓝  (f  x),  ∀ᶠ  x  in  𝓝  x,  f  x  ∈  U  :=
  Iff.rfl 
```

我们现在解释如何从一个观点转换到另一个观点。在开集的术语下，我们可以简单地定义 `𝓝 x` 的成员为包含 `x` 的开集的集合。

```py
example  {x  :  X}  {s  :  Set  X}  :  s  ∈  𝓝  x  ↔  ∃  t,  t  ⊆  s  ∧  IsOpen  t  ∧  x  ∈  t  :=
  mem_nhds_iff 
```

要向相反的方向进行，我们需要讨论 `𝓝 : X → Filter X` 必须满足的条件，以便成为拓扑的邻域函数。

第一个约束是，将 `𝓝 x` 视为一个广义集合时，它包含将 `{x}` 视为一个广义集合 `pure x` 的集合（解释这个奇怪的名字会过于离题，所以我们现在简单地接受它）。另一种说法是，如果一个谓词在 `x` 附近的点成立，那么它在 `x` 处也成立。

```py
example  (x  :  X)  :  pure  x  ≤  𝓝  x  :=
  pure_le_nhds  x

example  (x  :  X)  (P  :  X  →  Prop)  (h  :  ∀ᶠ  y  in  𝓝  x,  P  y)  :  P  x  :=
  h.self_of_nhds 
```

然后，一个更微妙的要求是，对于任何谓词 `P : X → Prop` 和任何 `x`，如果 `P y` 对于靠近 `x` 的 `y` 成立，那么对于靠近 `x` 和 `y` 的 `z`，`P z` 也成立。更精确地说，我们有：

```py
example  {P  :  X  →  Prop}  {x  :  X}  (h  :  ∀ᶠ  y  in  𝓝  x,  P  y)  :  ∀ᶠ  y  in  𝓝  x,  ∀ᶠ  z  in  𝓝  y,  P  z  :=
  eventually_eventually_nhds.mpr  h 
```

这两个结果描述了函数 `X → Filter X`，这些函数是 `X` 上拓扑空间结构的邻域函数。仍然存在一个函数 `TopologicalSpace.mkOfNhds : (X → Filter X) → TopologicalSpace X`，但它只有在满足上述两个约束的情况下才会将其输入作为邻域函数返回。更精确地说，我们有一个引理 `TopologicalSpace.nhds_mkOfNhds`，它以不同的方式表述，而我们的下一个练习将从我们上述的表述中推导出这种不同方式。

```py
example  {α  :  Type*}  (n  :  α  →  Filter  α)  (H₀  :  ∀  a,  pure  a  ≤  n  a)
  (H  :  ∀  a  :  α,  ∀  p  :  α  →  Prop,  (∀ᶠ  x  in  n  a,  p  x)  →  ∀ᶠ  y  in  n  a,  ∀ᶠ  x  in  n  y,  p  x)  :
  ∀  a,  ∀  s  ∈  n  a,  ∃  t  ∈  n  a,  t  ⊆  s  ∧  ∀  a'  ∈  t,  s  ∈  n  a'  :=  by
  sorry
end 
```

注意，`TopologicalSpace.mkOfNhds` 并不经常使用，但了解在拓扑空间结构中邻域过滤器究竟意味着什么仍然是很好的。

为了有效地在 Mathlib 中使用拓扑空间，我们需要知道的是，我们使用了 `TopologicalSpace : Type u → Type u` 的许多形式性质。从纯粹数学的角度来看，这些形式性质是解释拓扑空间如何解决度量空间问题的非常干净的方法。从这个角度来看，拓扑空间解决的问题在于度量空间几乎不具有函子性，并且在一般上具有很差的范畴性质。这还基于已经讨论的事实，即度量空间包含大量与拓扑无关的几何信息。

让我们先关注函子性。度量空间结构可以诱导在子集上，或者等价地，它可以由一个注入映射拉回。但这基本上就是全部了。它们不能由一般映射或推前，甚至不能由满射映射推前。

特别地，在度量空间的商或不可数度量空间的积上，没有合理的距离可以放置。例如，考虑类型 `ℝ → ℝ`，它被视为由 `ℝ` 索引的 `ℝ` 的复制品的积。我们希望说，函数序列逐点收敛是一个值得尊重的收敛概念。但在 `ℝ → ℝ` 上没有距离可以给出这种收敛概念。相关地，没有距离可以保证映射 `f : X → (ℝ → ℝ)` 连续当且仅当对于每个 `t : ℝ`，`fun x ↦ f x t` 是连续的。

我们现在回顾用于解决所有这些问题的数据。首先，我们可以使用任何映射 `f : X → Y` 来从一个方向推动或拉动拓扑到另一个方向。这两个操作形成一个伽罗瓦连接。

```py
variable  {X  Y  :  Type*}

example  (f  :  X  →  Y)  :  TopologicalSpace  X  →  TopologicalSpace  Y  :=
  TopologicalSpace.coinduced  f

example  (f  :  X  →  Y)  :  TopologicalSpace  Y  →  TopologicalSpace  X  :=
  TopologicalSpace.induced  f

example  (f  :  X  →  Y)  (T_X  :  TopologicalSpace  X)  (T_Y  :  TopologicalSpace  Y)  :
  TopologicalSpace.coinduced  f  T_X  ≤  T_Y  ↔  T_X  ≤  TopologicalSpace.induced  f  T_Y  :=
  coinduced_le_iff_le_induced 
```

这些操作与函数的复合是兼容的。通常，推前是协变的，拉回是反对变的，参见 `coinduced_compose` 和 `induced_compose`。在纸上，我们将使用 $f_*T$ 表示 `TopologicalSpace.coinduced f T` 和 $f^*T$ 表示 `TopologicalSpace.induced f T`。

接下来的一大块是对于任何给定的结构在 `TopologicalSpace X` 上的完整格结构。如果你认为拓扑主要是开集的数据，那么你期望 `TopologicalSpace X` 上的顺序关系来自 `Set (Set X)`，即你期望如果集合 `u` 对于 `t'` 是开集，那么它对于 `t` 也是开集，你就期望 `t ≤ t'`。然而，我们已经知道 Mathlib 更关注邻域而不是开集，所以对于任何 `x : X`，我们希望从拓扑空间到邻域的映射 `fun T : TopologicalSpace X ↦ @nhds X T x` 是顺序保持的。我们还知道 `Filter X` 上的顺序关系是为了确保顺序保持的 `principal : Set X → Filter X`，允许将过滤器视为广义集合。因此，我们在 `TopologicalSpace X` 上使用的顺序关系与来自 `Set (Set X)` 的顺序关系相反。

```py
example  {T  T'  :  TopologicalSpace  X}  :  T  ≤  T'  ↔  ∀  s,  T'.IsOpen  s  →  T.IsOpen  s  :=
  Iff.rfl 
```

现在我们可以通过结合推前（或拉回）操作和顺序关系来恢复连续性。

```py
example  (T_X  :  TopologicalSpace  X)  (T_Y  :  TopologicalSpace  Y)  (f  :  X  →  Y)  :
  Continuous  f  ↔  TopologicalSpace.coinduced  f  T_X  ≤  T_Y  :=
  continuous_iff_coinduced_le 
```

通过这个定义和推前和复合的兼容性，我们免费获得了这样一个普遍性质：对于任何拓扑空间 $Z$，一个函数 $g : Y → Z$ 在拓扑 $f_*T_X$ 下是连续的，当且仅当 $g ∘ f$ 是连续的。

$$\begin{split}g \text{ 连续 } &⇔ g_*(f_*T_X) ≤ T_Z \\ &⇔ (g ∘ f)_* T_X ≤ T_Z \\ &⇔ g ∘ f \text{ 连续}\end{split}$$

```py
example  {Z  :  Type*}  (f  :  X  →  Y)  (T_X  :  TopologicalSpace  X)  (T_Z  :  TopologicalSpace  Z)
  (g  :  Y  →  Z)  :
  @Continuous  Y  Z  (TopologicalSpace.coinduced  f  T_X)  T_Z  g  ↔
  @Continuous  X  Z  T_X  T_Z  (g  ∘  f)  :=  by
  rw  [continuous_iff_coinduced_le,  coinduced_compose,  continuous_iff_coinduced_le] 
```

因此，我们已经得到了商拓扑（使用投影映射作为 `f`）。这并不是因为 `TopologicalSpace X` 对于所有 `X` 都是完整的格。现在让我们看看所有这些结构是如何通过抽象的胡言乱语来证明乘积拓扑的存在性的。我们上面考虑了 `ℝ → ℝ` 的情况，但现在让我们考虑一般情况 `Π i, X i` 对于某个 `ι : Type*` 和 `X : ι → Type*`。我们希望对于任何拓扑空间 `Z` 和任何函数 `f : Z → Π i, X i`，如果对于所有 `i`，`(fun x ↦ x i) ∘ f` 是连续的，那么 `f` 是连续的。让我们使用表示投影 `(fun (x : Π i, X i) ↦ x i)` 的符号 $p_i$ 来在“纸上”探索这个约束：

$$\begin{split}(∀ i, p_i ∘ f \text{ 连续}) &⇔ ∀ i, (p_i ∘ f)_* T_Z ≤ T_{X_i} \\ &⇔ ∀ i, (p_i)_* f_* T_Z ≤ T_{X_i}\\ &⇔ ∀ i, f_* T_Z ≤ (p_i)^*T_{X_i}\\ &⇔ f_* T_Z ≤ \inf \left[(p_i)^*T_{X_i}\right]\end{split}$$

因此，我们看到我们希望在 `Π i, X i` 上的是什么拓扑：

```py
example  (ι  :  Type*)  (X  :  ι  →  Type*)  (T_X  :  ∀  i,  TopologicalSpace  (X  i))  :
  (Pi.topologicalSpace  :  TopologicalSpace  (∀  i,  X  i))  =
  ⨅  i,  TopologicalSpace.induced  (fun  x  ↦  x  i)  (T_X  i)  :=
  rfl 
```

这就结束了我们对 Mathlib 如何认为拓扑空间通过成为一个更函数化的理论和为任何固定类型提供完整的格结构来修复度量空间理论的缺陷的考察。

### 11.3.2\. 分离与可数性

我们看到拓扑空间的范畴具有非常不错的性质。为此付出的代价是存在相当病态的拓扑空间。你可以对拓扑空间做出一些假设，以确保其行为更接近度量空间。其中最重要的是`T2Space`，也称为“豪斯多夫”，这将确保极限是唯一的。更强的分离性质是`T3Space`，它还确保了正则空间性质：每个点都有一个由闭邻域组成的基。

```py
example  [TopologicalSpace  X]  [T2Space  X]  {u  :  ℕ  →  X}  {a  b  :  X}  (ha  :  Tendsto  u  atTop  (𝓝  a))
  (hb  :  Tendsto  u  atTop  (𝓝  b))  :  a  =  b  :=
  tendsto_nhds_unique  ha  hb

example  [TopologicalSpace  X]  [RegularSpace  X]  (a  :  X)  :
  (𝓝  a).HasBasis  (fun  s  :  Set  X  ↦  s  ∈  𝓝  a  ∧  IsClosed  s)  id  :=
  closed_nhds_basis  a 
```

注意，在每一个拓扑空间中，每个点都有一个由开邻域组成的基，这是定义所要求的。

```py
example  [TopologicalSpace  X]  {x  :  X}  :
  (𝓝  x).HasBasis  (fun  t  :  Set  X  ↦  t  ∈  𝓝  x  ∧  IsOpen  t)  id  :=
  nhds_basis_opens'  x 
```

我们现在的目标是证明允许通过连续性扩展的基本定理。从 Bourbaki 的普通拓扑学书籍，I.8.5，定理 1（只取非平凡蕴含）：

设$X$是一个拓扑空间，$A$是$X$的稠密子集，$f : A → Y$是$A$到$T_3$空间$Y$的连续映射。如果对于$X$中的每个$x$，当$y$在$A$中趋向于$x$时，$f(y)$在$Y$中趋向于一个极限，那么存在一个将$f$扩展到$X$的连续扩展$φ$。

实际上，Mathlib 包含上述引理的一个更通用版本，`IsDenseInducing.continuousAt_extend`，但我们将坚持使用 Bourbaki 的版本。

记住，给定`A : Set X`，`↥A`是与`A`关联的子类型，并且当需要时 Lean 会自动插入那个古怪的向上箭头。并且（包含）强制映射是`(↑) : A → X`。假设“在$A$中趋向于$x$”对应于拉回过滤器`comap (↑) (𝓝 x)`。

让我们先证明一个辅助引理，将其提取出来以简化上下文（特别是这里我们不需要 Y 是一个拓扑空间）。

```py
theorem  aux  {X  Y  A  :  Type*}  [TopologicalSpace  X]  {c  :  A  →  X}
  {f  :  A  →  Y}  {x  :  X}  {F  :  Filter  Y}
  (h  :  Tendsto  f  (comap  c  (𝓝  x))  F)  {V'  :  Set  Y}  (V'_in  :  V'  ∈  F)  :
  ∃  V  ∈  𝓝  x,  IsOpen  V  ∧  c  ⁻¹'  V  ⊆  f  ⁻¹'  V'  :=  by
  sorry 
```

现在我们转向连续扩展定理的主要证明。

当 Lean 需要`↥A`上的拓扑时，它将自动使用诱导拓扑。唯一相关的引理是`nhds_induced (↑) : ∀ a : ↥A, 𝓝 a = comap (↑) (𝓝 ↑a)`（这实际上是一个关于诱导拓扑的通用引理）。

证明概要是：

主要假设和选择公理给出一个函数`φ`，使得`∀ x, Tendsto f (comap (↑) (𝓝 x)) (𝓝 (φ x))`（因为`Y`是豪斯多夫的，`φ`完全确定，但我们不会用到这一点，直到我们试图证明`φ`确实扩展了`f`）。

首先，我们来证明 `φ` 是连续的。固定任意的 `x : X`。由于 `Y` 是正则的，我们只需检查对于 `φ x` 的每一个 *闭* 邻域 `V'`，`φ ⁻¹' V' ∈ 𝓝 x`。极限假设给出了（通过上面的辅助引理）一些 `V ∈ 𝓝 x`，使得 `IsOpen V ∧ (↑) ⁻¹' V ⊆ f ⁻¹' V'`。由于 `V ∈ 𝓝 x`，我们只需证明 `V ⊆ φ ⁻¹' V'`，即 `∀ y ∈ V, φ y ∈ V'`。让我们在 `V` 中固定 `y`。因为 `V` 是 *开* 的，它是 `y` 的邻域。特别是 `(↑) ⁻¹' V ∈ comap (↑) (𝓝 y)`，并且更明显 `f ⁻¹' V' ∈ comap (↑) (𝓝 y)`。此外，`comap (↑) (𝓝 y) ≠ ⊥` 因为 `A` 是稠密的。因为我们知道 `Tendsto f (comap (↑) (𝓝 y)) (𝓝 (φ y))`，这表明 `φ y ∈ closure V'`，并且由于 `V'` 是闭的，我们证明了 `φ y ∈ V'`。

剩下的工作是证明 `φ` 扩展了 `f`。这是 `f` 的连续性和 `Y` 是豪斯多夫空间的事实进入讨论的地方。

```py
example  [TopologicalSpace  X]  [TopologicalSpace  Y]  [T3Space  Y]  {A  :  Set  X}
  (hA  :  ∀  x,  x  ∈  closure  A)  {f  :  A  →  Y}  (f_cont  :  Continuous  f)
  (hf  :  ∀  x  :  X,  ∃  c  :  Y,  Tendsto  f  (comap  (↑)  (𝓝  x))  (𝓝  c))  :
  ∃  φ  :  X  →  Y,  Continuous  φ  ∧  ∀  a  :  A,  φ  a  =  f  a  :=  by
  sorry

#check  HasBasis.tendsto_right_iff 
```

除了分离性质，你可以对拓扑空间做的最主要的一种假设是可数性假设。主要的一个是首先可数性，要求每个点都有一个可数邻域基。特别是这保证了可以使用序列来理解集合的闭包。

```py
example  [TopologicalSpace  X]  [FirstCountableTopology  X]
  {s  :  Set  X}  {a  :  X}  :
  a  ∈  closure  s  ↔  ∃  u  :  ℕ  →  X,  (∀  n,  u  n  ∈  s)  ∧  Tendsto  u  atTop  (𝓝  a)  :=
  mem_closure_iff_seq_limit 
```

### 11.3.3\. 紧性

现在，让我们讨论拓扑空间中紧性的定义。通常有几种思考方式，Mathlib 采用的是滤子版本。

我们首先需要定义拓扑空间上滤子的聚点。给定一个拓扑空间 `X` 上的滤子 `F`，一个点 `x : X` 是 `F` 的聚点，如果 `F` 作为广义集合，与接近 `x` 的点的广义集合的非空交集。

然后，我们可以这样说，一个集合 `s` 是紧的，如果包含在 `s` 中的每一个非空广义集合 `F`，即 `F ≤ 𝓟 s`，在 `s` 中都有一个聚点。

```py
variable  [TopologicalSpace  X]

example  {F  :  Filter  X}  {x  :  X}  :  ClusterPt  x  F  ↔  NeBot  (𝓝  x  ⊓  F)  :=
  Iff.rfl

example  {s  :  Set  X}  :
  IsCompact  s  ↔  ∀  (F  :  Filter  X)  [NeBot  F],  F  ≤  𝓟  s  →  ∃  a  ∈  s,  ClusterPt  a  F  :=
  Iff.rfl 
```

例如，如果 `F` 是 `map u atTop`，即 `u : ℕ → X` 的 `atTop` 的像，`atTop` 是非常大的自然数的广义集合，那么 `F ≤ 𝓟 s` 的假设意味着对于足够大的 `n`，`u n` 属于 `s`。说 `x` 是 `map u atTop` 的聚点意味着非常大的数的像与接近 `x` 的点的集合相交。如果 `𝓝 x` 有可数基，我们可以将其解释为 `u` 有一个子序列收敛到 `x`，从而得到紧性在度量空间中的样子。

```py
example  [FirstCountableTopology  X]  {s  :  Set  X}  {u  :  ℕ  →  X}  (hs  :  IsCompact  s)
  (hu  :  ∀  n,  u  n  ∈  s)  :  ∃  a  ∈  s,  ∃  φ  :  ℕ  →  ℕ,  StrictMono  φ  ∧  Tendsto  (u  ∘  φ)  atTop  (𝓝  a)  :=
  hs.tendsto_subseq  hu 
```

聚点与连续函数的行为良好。

```py
variable  [TopologicalSpace  Y]

example  {x  :  X}  {F  :  Filter  X}  {G  :  Filter  Y}  (H  :  ClusterPt  x  F)  {f  :  X  →  Y}
  (hfx  :  ContinuousAt  f  x)  (hf  :  Tendsto  f  F  G)  :  ClusterPt  (f  x)  G  :=
  ClusterPt.map  H  hfx  hf 
```

作为练习，我们将证明在连续映射下紧集的像也是紧集的。除了我们已经看到的，你应该使用 `Filter.push_pull` 和 `NeBot.of_map`。

```py
example  [TopologicalSpace  Y]  {f  :  X  →  Y}  (hf  :  Continuous  f)  {s  :  Set  X}  (hs  :  IsCompact  s)  :
  IsCompact  (f  ''  s)  :=  by
  intro  F  F_ne  F_le
  have  map_eq  :  map  f  (𝓟  s  ⊓  comap  f  F)  =  𝓟  (f  ''  s)  ⊓  F  :=  by  sorry
  have  Hne  :  (𝓟  s  ⊓  comap  f  F).NeBot  :=  by  sorry
  have  Hle  :  𝓟  s  ⊓  comap  f  F  ≤  𝓟  s  :=  inf_le_left
  sorry 
```

也可以用开覆盖来表示紧性：如果 `s` 是紧的，那么覆盖 `s` 的每一个开集族都有一个有限覆盖子族。

```py
example  {ι  :  Type*}  {s  :  Set  X}  (hs  :  IsCompact  s)  (U  :  ι  →  Set  X)  (hUo  :  ∀  i,  IsOpen  (U  i))
  (hsU  :  s  ⊆  ⋃  i,  U  i)  :  ∃  t  :  Finset  ι,  s  ⊆  ⋃  i  ∈  t,  U  i  :=
  hs.elim_finite_subcover  U  hUo  hsU 
``` 上一节 下一节

* * *

© 版权所有 2020-2025，Jeremy Avigad，Patrick Massot。文本许可协议为 CC BY 4.0。

使用[Sphinx](https://www.sphinx-doc.org/)构建，并采用了[Read the Docs](https://readthedocs.org)提供的[主题](https://github.com/readthedocs/sphinx_rtd_theme)。微积分基于函数的概念，该概念用于模拟相互依赖的量。例如，研究随时间变化的量是很常见的。极限的概念也是基本的。我们可以说，当$x$趋近于值$a$时，函数$f(x)$的极限是一个值$b$，或者说$f(x)$当$x$趋近于$a$时*收敛于*$b$。等价地，我们可以说当$x$趋近于值$a$时，$f(x)$趋近于$b$，或者说它当$x$趋近于$a$时*趋向于*$b$。我们已经在第 3.6 节开始考虑这样的概念了。

*拓扑学*是极限和连续性的抽象研究。在涵盖了第二章到 7 章中形式化的基本要素之后，在本章中，我们将解释在 Mathlib 中如何形式化拓扑概念。不仅拓扑抽象在更广泛的范围内适用，而且它们在某种程度上具有矛盾性，使得在具体实例中推理极限和连续性变得更加容易。

拓扑概念建立在许多数学结构层之上。第一层是描述在第四章中的朴素集合论。下一层是*过滤器*理论，我们将在第 11.1 节中描述。在此基础上，我们叠加了*拓扑空间*、*度量空间*以及一个稍微更奇特的中介概念，称为*均匀空间*。

虽然前几章依赖于你可能已经熟悉的数学概念，但过滤器这个概念对许多工作数学家来说并不那么熟悉。然而，这个概念对于有效地形式化数学是必不可少的。让我们解释一下原因。设$f : ℝ → ℝ$为任意函数。我们可以考虑当$x$趋近于某个值$x₀$时$f x$的极限，但也可以考虑当$x$趋近于正无穷或负无穷时$f x$的极限。此外，我们还可以考虑当$x$从右侧（通常写作$x₀⁺$）或从左侧（写作$x₀⁻$）趋近于$x₀$时$f x$的极限。存在一些变体，其中$x$趋近于$x₀$、$x₀⁺$或$x₀⁻$，但不允许取值$x₀$本身。这至少产生了八种$x$趋近于某物的途径。我们还可以将$x$限制为有理数或对定义域施加其他约束，但让我们坚持这 8 种情况。

在值域上，我们有类似的各种选择：我们可以指定`f x`是从左侧还是右侧趋近于一个值，或者它趋近于正无穷或负无穷，等等。例如，我们可能希望说，当`x`从右侧趋近于`x₀`而不等于`x₀`时，`f x`趋于`+∞`。这导致了 64 种不同的极限陈述，而我们甚至还没有开始处理与数列的极限，正如我们在第 3.6 节中所做的那样。

当涉及到支持引理时，问题变得更加复杂。例如，极限可以组合：如果`f x`在`x`趋近于`x₀`时趋于`y₀`，而`g y`在`y`趋近于`y₀`时趋于`z₀`，那么`g ∘ f x`在`x`趋近于`x₀`时也趋于`z₀`。这里涉及到三种“趋于”的概念，每种都可以用前一段中描述的八种方式中的任何一种来实例化。这导致了 512 个引理，需要添加到库中！非正式地说，数学家通常证明其中两到三个，并简单地指出其余的可以“以相同的方式”证明。形式化数学需要使相关的“相同”概念完全明确，这正是 Bourbaki 的过滤器理论所成功做到的。

## 11.1\. 过滤器

在类型`X`上的*过滤器*是一组`X`的集合，它满足以下三个条件，我们将在下面详细说明。这个概念支持两个相关想法：

+   *极限*，包括上述讨论的所有类型的极限：数列的有限和无限极限，函数在一点或无穷远处的有限和无限极限，等等。

+   *最终发生的事情*，包括对于足够大的`n : ℕ`，或者足够接近一个点`x`，或者对于足够接近的点对，或者在测度理论意义上的几乎处处。类似地，过滤器也可以表达*经常发生的事情*的概念：对于任意大的`n`，在任意给定点的任意邻域内，等等。

对应于这些描述的过滤器将在本节的后面定义，但我们可以先命名它们：

+   `(atTop : Filter ℕ)`，由包含`{n | n ≥ N}`（对于某个`N`）的`ℕ`的集合组成。

+   `𝓝 x`，由拓扑空间中`x`的邻域组成。

+   `𝓤 X`，由均匀空间（均匀空间是度量空间和拓扑群的推广）的邻域组成。

+   `μ.ae`，由其补集相对于测度`μ`的测度为零的集合组成。

一般定义如下：一个过滤器`F : Filter X`是一个集合`F.sets : Set (Set X)`，它满足以下条件：

+   `F.univ_sets : univ ∈ F.sets`

+   `F.sets_of_superset : ∀ {U V}, U ∈ F.sets → U ⊆ V → V ∈ F.sets`

+   `F.inter_sets : ∀ {U V}, U ∈ F.sets → V ∈ F.sets → U ∩ V ∈ F.sets`.

第一个条件说明，`X`的所有元素集合属于`F.sets`。第二个条件说明，如果`U`属于`F.sets`，那么包含`U`的任何东西也属于`F.sets`。第三个条件说明，`F.sets`在有限交集下是封闭的。在 Mathlib 中，一个过滤器`F`被定义为将`F.sets`及其三个属性捆绑在一起的结构，但属性不携带任何额外数据，并且模糊`F`和`F.sets`之间的区别是方便的。因此，我们定义`U ∈ F`表示`U ∈ F.sets`。这解释了为什么在提及`U ∈ F`的一些引理名称中出现了单词`sets`。

可以将过滤器视为定义一个“足够大”的集合的概念。第一个条件说明`univ`足够大，第二个条件说明包含足够大的集合的集合足够大，第三个条件说明两个足够大的集合的交集足够大。

将类型`X`上的过滤器视为`Set X`的广义元素可能更有用。例如，`atTop`是“非常大的数”的集合，而`𝓝 x₀`是“非常接近`x₀`的点”的集合。这种观点的一个表现是，我们可以将任何`s : Set X`与所谓的*主过滤器*关联起来，该过滤器包含所有包含`s`的集合。这个定义已经在 Mathlib 中，并且有`𝓟`（在`Filter`命名空间中局部化）的符号。为了演示目的，我们要求你利用这个机会在这里推导出定义。

```py
def  principal  {α  :  Type*}  (s  :  Set  α)  :  Filter  α
  where
  sets  :=  {  t  |  s  ⊆  t  }
  univ_sets  :=  sorry
  sets_of_superset  :=  sorry
  inter_sets  :=  sorry 
```

对于我们的第二个例子，我们要求你定义过滤器`atTop : Filter ℕ`。（我们可以用任何具有偏序的`ℕ`代替。）

```py
example  :  Filter  ℕ  :=
  {  sets  :=  {  s  |  ∃  a,  ∀  b,  a  ≤  b  →  b  ∈  s  }
  univ_sets  :=  sorry
  sets_of_superset  :=  sorry
  inter_sets  :=  sorry  } 
```

我们还可以直接定义任何`x : ℝ`的邻域过滤器`𝓝 x`。在实数中，`x`的邻域是包含一个开区间`(x_0 - \varepsilon, x_0 + \varepsilon)`的集合，在 Mathlib 中定义为`Ioo (x₀ - ε) (x₀ + ε)`。（这种邻域的概念只是 Mathlib 中更一般构造的一个特例。）

通过这些例子，我们已经可以定义函数`f : X → Y`在某个`F : Filter X`上沿某个`G : Filter Y`收敛的含义，如下所示：

```py
def  Tendsto₁  {X  Y  :  Type*}  (f  :  X  →  Y)  (F  :  Filter  X)  (G  :  Filter  Y)  :=
  ∀  V  ∈  G,  f  ⁻¹'  V  ∈  F 
```

当`X`是`ℕ`且`Y`是`ℝ`时，`Tendsto₁ u atTop (𝓝 x)`等价于说序列`u : ℕ → ℝ`收敛到实数`x`。当`X`和`Y`都是`ℝ`时，`Tendsto f (𝓝 x₀) (𝓝 y₀)`等价于熟悉的极限概念 $\lim_{x \to x₀} f(x) = y₀$。介绍中提到的所有其他类型的极限也等价于在源和目标上选择合适的过滤器时的`Tendsto₁`实例。

上面定义的`Tendsto₁`在定义上等同于在 Mathlib 中定义的`Tendsto`的概念，但后者定义得更抽象。`Tendsto₁`定义的问题在于它暴露了一个量词和`G`的元素，并隐藏了通过将过滤器视为广义集合而获得的直觉。我们可以通过使用更多的代数和集合论工具来隐藏量词`∀ V`并使直觉更加明显。第一个成分是与任何映射`f : X → Y`相关联的*推前*操作`f_*`，在 Mathlib 中用`Filter.map f`表示。给定`X`上的过滤器`F`，`Filter.map f F : Filter Y`被定义为`V ∈ Filter.map f F ↔ f ⁻¹' V ∈ F`在定义上成立。在打开的示例文件中，我们已打开`Filter`命名空间，以便可以将`Filter.map`写成`map`。这意味着我们可以使用`Filter Y`上的顺序关系重写`Tendsto`的定义，该顺序关系是成员集合的反向包含。换句话说，给定`G H : Filter Y`，我们有`G ≤ H ↔ ∀ V : Set Y, V ∈ H → V ∈ G`。

```py
def  Tendsto₂  {X  Y  :  Type*}  (f  :  X  →  Y)  (F  :  Filter  X)  (G  :  Filter  Y)  :=
  map  f  F  ≤  G

example  {X  Y  :  Type*}  (f  :  X  →  Y)  (F  :  Filter  X)  (G  :  Filter  Y)  :
  Tendsto₂  f  F  G  ↔  Tendsto₁  f  F  G  :=
  Iff.rfl 
```

可能看起来过滤器上的顺序关系是相反的。但回想一下，我们可以通过`𝓟 : Set X → Filter X`的包含来将`X`上的过滤器视为`Set X`的广义元素，该映射将任何集合`s`映射到相应的主过滤器。这个包含是顺序保持的，因此`Filter`上的顺序关系确实可以看作是广义集合之间的自然包含关系。在这个类比中，推前相当于直接像。实际上，`map f (𝓟 s) = 𝓟 (f '' s)`。

现在，我们可以直观地理解为什么一个序列`u : ℕ → ℝ`收敛到点`x₀`当且仅当我们有`map u atTop ≤ 𝓝 x₀`。这个不等式意味着“在`u`下的直接像”的“非常大的自然数集合”包含在“非常接近`x₀`的点集合”中。

如承诺的那样，`Tendsto₂`的定义没有展示任何量词或集合。它还利用了推前操作的代数性质。首先，每个`Filter.map f`是单调的。其次，`Filter.map`与组合兼容。

```py
#check  (@Filter.map_mono  :  ∀  {α  β}  {m  :  α  →  β},  Monotone  (map  m))

#check
  (@Filter.map_map  :
  ∀  {α  β  γ}  {f  :  Filter  α}  {m  :  α  →  β}  {m'  :  β  →  γ},  map  m'  (map  m  f)  =  map  (m'  ∘  m)  f) 
```

这两个性质共同使我们能够证明极限可以组合，一次就得到介绍中描述的组成引理的所有 512 个变体，以及更多。你可以通过使用关于`Tendsto₁`的通用量词定义或代数定义，以及上述两个引理来练习证明以下陈述。

```py
example  {X  Y  Z  :  Type*}  {F  :  Filter  X}  {G  :  Filter  Y}  {H  :  Filter  Z}  {f  :  X  →  Y}  {g  :  Y  →  Z}
  (hf  :  Tendsto₁  f  F  G)  (hg  :  Tendsto₁  g  G  H)  :  Tendsto₁  (g  ∘  f)  F  H  :=
  sorry 
```

推前构造使用一个映射将过滤器从映射源推送到映射目标。还有一个相反方向的*拉回*操作，`Filter.comap`。这推广了集合上的前像操作。对于任何映射`f`，`Filter.map f`和`Filter.comap f`形成所谓的*伽罗瓦连接*，也就是说，它们满足

> `Filter.map_le_iff_le_comap : Filter.map f F ≤ G ↔ F ≤ Filter.comap f G`

对于每一个`F`和`G`。这个操作可以用来提供`Tendsto`的另一种公理化表述，这将与 Mathlib 中的表述（虽然不是定义上）等价。

`comap`操作可以用来将过滤器限制为子类型。例如，假设我们有`f : ℝ → ℝ`，`x₀ : ℝ`和`y₀ : ℝ`，并且假设我们想要表述当`x`在有理数中接近`x₀`时，`f x`接近`y₀`。我们可以使用强制映射`(↑) : ℚ → ℝ`将过滤器`𝓝 x₀`拉回到`ℚ`，并表述`Tendsto (f ∘ (↑) : ℚ → ℝ) (comap (↑) (𝓝 x₀)) (𝓝 y₀)`。

```py
variable  (f  :  ℝ  →  ℝ)  (x₀  y₀  :  ℝ)

#check  comap  ((↑)  :  ℚ  →  ℝ)  (𝓝  x₀)

#check  Tendsto  (f  ∘  (↑))  (comap  ((↑)  :  ℚ  →  ℝ)  (𝓝  x₀))  (𝓝  y₀) 
```

拉回操作也与组合兼容，但它是*逆变*的，也就是说，它反转了参数的顺序。

```py
section
variable  {α  β  γ  :  Type*}  (F  :  Filter  α)  {m  :  γ  →  β}  {n  :  β  →  α}

#check  (comap_comap  :  comap  m  (comap  n  F)  =  comap  (n  ∘  m)  F)

end 
```

现在，让我们将注意力转向平面`ℝ × ℝ`，并尝试理解点`(x₀, y₀)`的邻域如何与`𝓝 x₀`和`𝓝 y₀`相关。存在一个乘积操作`Filter.prod : Filter X → Filter Y → Filter (X × Y)`，表示为`×ˢ`，它回答了这个问题：

```py
example  :  𝓝  (x₀,  y₀)  =  𝓝  x₀  ×ˢ  𝓝  y₀  :=
  nhds_prod_eq 
```

产品操作定义为拉回操作和`inf`操作：

> `F ×ˢ G = (comap Prod.fst F) ⊓ (comap Prod.snd G)`。

这里`inf`操作指的是任何类型`X`上`Filter X`的格结构，其中`F ⊓ G`是小于`F`和`G`的最大过滤器。因此，`inf`操作推广了集合交集的概念。

Mathlib 中的许多证明都使用了上述所有结构（`map`、`comap`、`inf`、`sup`和`prod`）来给出关于收敛性的代数证明，而从未引用过滤器的成员。你可以在以下引理的证明中练习这样做，如果需要，可以展开`Tendsto`和`Filter.prod`的定义。

```py
#check  le_inf_iff

example  (f  :  ℕ  →  ℝ  ×  ℝ)  (x₀  y₀  :  ℝ)  :
  Tendsto  f  atTop  (𝓝  (x₀,  y₀))  ↔
  Tendsto  (Prod.fst  ∘  f)  atTop  (𝓝  x₀)  ∧  Tendsto  (Prod.snd  ∘  f)  atTop  (𝓝  y₀)  :=
  sorry 
```

有序类型`Filter X`实际上是一个*完备*格，这意味着存在一个底元素，存在一个顶元素，并且`X`上的每个过滤器集合都有一个`Inf`和`Sup`。

注意，根据过滤器定义中的第二个性质（如果`U`属于`F`，则任何大于`U`的集合也属于`F`），第一个性质（`X`的所有居民集合属于`F`）等价于`F`不是空集合的性质。这不应与更微妙的问题混淆，即空集是否是`F`的*元素*。过滤器的定义并不禁止`∅ ∈ F`，但如果空集在`F`中，则每个集合都在`F`中，也就是说，`∀ U : Set X, U ∈ F`。在这种情况下，`F`是一个相当平凡的过滤器，这正是完全格`Filter X`的底元素。这与 Bourbaki 中过滤器的定义形成对比，Bourbaki 不允许包含空集的过滤器。

由于我们在定义中包括了平凡过滤器，我们有时需要明确假设某些引理的非平凡性。然而，作为回报，理论具有更佳的全局性质。我们已经看到，包括平凡过滤器给我们提供了一个底元素。它还允许我们定义`principal : Set X → Filter X`，该映射将`∅`映射到`⊥`，而不需要添加一个先决条件来排除空集。它还允许我们定义没有先决条件的拉回操作。实际上，可能会发生`comap f F = ⊥`的情况，尽管`F ≠ ⊥`。例如，给定`x₀ : ℝ`和`s : Set ℝ`，从对应于`s`的子类型强制转换下的`𝓝 x₀`的拉回是非平凡的，当且仅当`x₀`属于`s`的闭包。

为了管理需要假设某些过滤器非平凡的引理，Mathlib 有一个类型类`Filter.NeBot`，库中有假设`(F : Filter X) [F.NeBot]`的引理。实例数据库知道，例如，`(atTop : Filter ℕ).NeBot`，它知道前推一个非平凡过滤器会得到一个非平凡过滤器。因此，假设`[F.NeBot]`的引理将自动适用于任何序列`u`的`map u atTop`。

我们对过滤器代数性质及其与极限关系的游览基本上已经完成，但我们还没有证明我们重新捕获了通常的极限概念。表面上，它可能看起来`Tendsto u atTop (𝓝 x₀)`比第 3.6 节中定义的收敛概念更强，因为我们要求`x₀`的每个邻域都有一个属于`atTop`的逆像，而通常的定义只要求对于标准邻域`Ioo (x₀ - ε) (x₀ + ε)`。关键是，根据定义，每个邻域都包含这样的标准邻域。这个观察导致了一个*过滤器基*的概念。

给定`F : Filter X`，一个集合族`s : ι → Set X`是`F`的基，如果对于每个集合`U`，我们有`U ∈ F`当且仅当它包含某个`s i`。换句话说，从形式上讲，如果`s`满足`∀ U : Set X, U ∈ F ↔ ∃ i, s i ⊆ U`，则`s`是一个基。考虑在`ι`上的谓词以仅选择索引类型中的某些值`i`，这甚至更加灵活。在`𝓝 x₀`的情况下，我们希望`ι`是`ℝ`，我们用`ε`表示`i`，谓词应该选择`ε`的正值。因此，集合`Ioo  (x₀ - ε) (x₀ + ε)`形成`ℝ`上的邻域拓扑的基，可以这样表述：

```py
example  (x₀  :  ℝ)  :  HasBasis  (𝓝  x₀)  (fun  ε  :  ℝ  ↦  0  <  ε)  fun  ε  ↦  Ioo  (x₀  -  ε)  (x₀  +  ε)  :=
  nhds_basis_Ioo_pos  x₀ 
```

对于过滤器`atTop`也有一个很好的基础。引理`Filter.HasBasis.tendsto_iff`允许我们根据`F`和`G`的基重新表述形式为`Tendsto f F G`的陈述。将这些部分组合起来，我们基本上得到了我们在第 3.6 节中使用的收敛的概念。

```py
example  (u  :  ℕ  →  ℝ)  (x₀  :  ℝ)  :
  Tendsto  u  atTop  (𝓝  x₀)  ↔  ∀  ε  >  0,  ∃  N,  ∀  n  ≥  N,  u  n  ∈  Ioo  (x₀  -  ε)  (x₀  +  ε)  :=  by
  have  :  atTop.HasBasis  (fun  _  :  ℕ  ↦  True)  Ici  :=  atTop_basis
  rw  [this.tendsto_iff  (nhds_basis_Ioo_pos  x₀)]
  simp 
```

现在我们展示如何使用过滤器简化处理对于足够大的数字或足够接近给定点的点所持有的性质。在 第 3.6 节 中，我们经常遇到这样的情况：我们知道某些性质 `P n` 对于足够大的 `n` 成立，而某些其他性质 `Q n` 对于足够大的 `n` 成立。使用 `cases` 两次给出了满足 `∀ n ≥ N_P, P n` 和 `∀ n ≥ N_Q, Q n` 的 `N_P` 和 `N_Q`。使用 `set N := max N_P N_Q`，我们最终可以证明 `∀ n ≥ N, P n ∧ Q n`。重复这样做会变得令人厌烦。

通过注意到陈述“`P n` 和 `Q n` 对于足够大的 `n` 成立”意味着我们拥有 `{n | P n} ∈ atTop` 和 `{n | Q n} ∈ atTop`，我们可以做得更好。`atTop` 是一个过滤器的事实意味着 `atTop` 中两个元素的交集再次在 `atTop` 中，因此我们得到 `{n | P n ∧ Q n} ∈ atTop`。写作 `{n | P n} ∈ atTop` 是不愉快的，但我们可以使用更具说明性的记法 `∀ᶠ n in atTop, P n`。这里的上标 `f` 代表“过滤器”。你可以将这种记法视为对于所有在“非常大的数字集合”中的 `n`，`P n` 成立。`∀ᶠ` 记号代表 `Filter.Eventually`，而 `Filter.Eventually.and` 引理使用过滤器的交集属性来完成我们刚才描述的工作：

```py
example  (P  Q  :  ℕ  →  Prop)  (hP  :  ∀ᶠ  n  in  atTop,  P  n)  (hQ  :  ∀ᶠ  n  in  atTop,  Q  n)  :
  ∀ᶠ  n  in  atTop,  P  n  ∧  Q  n  :=
  hP.and  hQ 
```

这种记法既方便又直观，因此当 `P` 是一个等式或不等式陈述时，我们也有特殊的处理。例如，设 `u` 和 `v` 是两个实数序列，让我们证明如果 `u n` 和 `v n` 在足够大的 `n` 上相同，那么 `u` 趋于 `x₀` 当且仅当 `v` 趋于 `x₀`。首先我们将使用通用的 `Eventually`，然后使用专门针对等价谓词的 `EventuallyEq`。这两个陈述在定义上是等价的，所以两种情况下都使用相同的证明工作。

```py
example  (u  v  :  ℕ  →  ℝ)  (h  :  ∀ᶠ  n  in  atTop,  u  n  =  v  n)  (x₀  :  ℝ)  :
  Tendsto  u  atTop  (𝓝  x₀)  ↔  Tendsto  v  atTop  (𝓝  x₀)  :=
  tendsto_congr'  h

example  (u  v  :  ℕ  →  ℝ)  (h  :  u  =ᶠ[atTop]  v)  (x₀  :  ℝ)  :
  Tendsto  u  atTop  (𝓝  x₀)  ↔  Tendsto  v  atTop  (𝓝  x₀)  :=
  tendsto_congr'  h 
```

通过 `Eventually` 来回顾过滤器的定义是有教育意义的。给定 `F : Filter X`，对于 `X` 上的任何谓词 `P` 和 `Q`，

+   条件 `univ ∈ F` 确保了 `(∀ x, P x) → ∀ᶠ x in F, P x`，

+   条件 `U ∈ F → U ⊆ V → V ∈ F` 确保了 `(∀ᶠ x in F, P x) → (∀ x, P x → Q x) → ∀ᶠ x in F, Q x`，并且

+   条件 `U ∈ F → V ∈ F → U ∩ V ∈ F` 确保了 `(∀ᶠ x in F, P x) → (∀ᶠ x in F, Q x) → ∀ᶠ x in F, P x ∧ Q x`。

```py
#check  Eventually.of_forall
#check  Eventually.mono
#check  Eventually.and 
```

第二项，对应于 `Eventually.mono`，支持了使用过滤器的优雅方式，尤其是在与 `Eventually.and` 结合使用时。`filter_upwards` 策略允许我们组合它们。比较：

```py
example  (P  Q  R  :  ℕ  →  Prop)  (hP  :  ∀ᶠ  n  in  atTop,  P  n)  (hQ  :  ∀ᶠ  n  in  atTop,  Q  n)
  (hR  :  ∀ᶠ  n  in  atTop,  P  n  ∧  Q  n  →  R  n)  :  ∀ᶠ  n  in  atTop,  R  n  :=  by
  apply  (hP.and  (hQ.and  hR)).mono
  rintro  n  ⟨h,  h',  h''⟩
  exact  h''  ⟨h,  h'⟩

example  (P  Q  R  :  ℕ  →  Prop)  (hP  :  ∀ᶠ  n  in  atTop,  P  n)  (hQ  :  ∀ᶠ  n  in  atTop,  Q  n)
  (hR  :  ∀ᶠ  n  in  atTop,  P  n  ∧  Q  n  →  R  n)  :  ∀ᶠ  n  in  atTop,  R  n  :=  by
  filter_upwards  [hP,  hQ,  hR]  with  n  h  h'  h''
  exact  h''  ⟨h,  h'⟩ 
```

了解测度理论的读者会注意到，集合的补集测度为零（即“几乎每个点的集合”）的过滤器 `μ.ae` 作为 `Tendsto` 的源或目标并不十分有用，但它可以方便地与 `Eventually` 结合使用，以表明某个性质对几乎所有点都成立。

`∀x ∈ F, P x`有一个对偶版本，偶尔很有用：`∃x ∈ F, P x`意味着`{x | ¬P x} ∉ F`。例如，`∃n ∈ atTop, P n`意味着存在任意大的`n`使得`P n`成立。`∃x`符号代表`Filter.Frequently`。

对于一个更复杂的例子，考虑以下关于序列`u`、集合`M`和值`x`的陈述：

> 如果`u`收敛到`x`且对于足够大的`n`，`u n`属于`M`，那么`x`在`M`的闭集中。

这可以形式化为以下内容：

> `Tendsto u atTop (𝓝 x) → (∀x ∈ atTop, u n ∈ M) → x ∈ closure M`。

这是拓扑库中定理`mem_closure_of_tendsto`的一个特例。看看你是否可以使用引用的引理，使用`ClusterPt x F`表示`(𝓝 x ⊓ F).NeBot`以及根据定义，假设`∀x ∈ atTop, u n ∈ M`意味着`M ∈ map u atTop`来证明它。

```py
#check  mem_closure_iff_clusterPt
#check  le_principal_iff
#check  neBot_of_le

example  (u  :  ℕ  →  ℝ)  (M  :  Set  ℝ)  (x  :  ℝ)  (hux  :  Tendsto  u  atTop  (𝓝  x))
  (huM  :  ∀ᶠ  n  in  atTop,  u  n  ∈  M)  :  x  ∈  closure  M  :=
  sorry 
```  ## 11.2\. 度量空间

上一节中的例子主要关注实数序列。在本节中，我们将提高一点一般性，并关注度量空间。度量空间是一个类型`X`，它配备了一个距离函数`dist : X → X → ℝ`，这是从`X = ℝ`的情况下的函数`fun x y ↦ |x - y|`的推广。

引入这样的空间很容易，我们将检查从距离函数所需的所有属性。

```py
variable  {X  :  Type*}  [MetricSpace  X]  (a  b  c  :  X)

#check  (dist  a  b  :  ℝ)
#check  (dist_nonneg  :  0  ≤  dist  a  b)
#check  (dist_eq_zero  :  dist  a  b  =  0  ↔  a  =  b)
#check  (dist_comm  a  b  :  dist  a  b  =  dist  b  a)
#check  (dist_triangle  a  b  c  :  dist  a  c  ≤  dist  a  b  +  dist  b  c) 
```

注意，我们还有变体，其中距离可以是无限的，或者`dist a b`可以是零，而`a ≠ b`或两者都不成立。它们分别称为`EMetricSpace`、`PseudoMetricSpace`和`PseudoEMetricSpace`（这里的“e”代表“扩展”）。

注意，我们从`ℝ`到度量空间的过程跳过了也需要线性代数的特殊情况的范空间，这将在微积分章节中解释。

### 11.2.1\. 收敛性和连续性

使用距离函数，我们已经在度量空间之间定义了收敛序列和连续函数。实际上，它们在下一节中更一般的设置中定义，但我们有引理将定义重新表述为距离的术语。

```py
example  {u  :  ℕ  →  X}  {a  :  X}  :
  Tendsto  u  atTop  (𝓝  a)  ↔  ∀  ε  >  0,  ∃  N,  ∀  n  ≥  N,  dist  (u  n)  a  <  ε  :=
  Metric.tendsto_atTop

example  {X  Y  :  Type*}  [MetricSpace  X]  [MetricSpace  Y]  {f  :  X  →  Y}  :
  Continuous  f  ↔
  ∀  x  :  X,  ∀  ε  >  0,  ∃  δ  >  0,  ∀  x',  dist  x'  x  <  δ  →  dist  (f  x')  (f  x)  <  ε  :=
  Metric.continuous_iff 
```

许多引理都有一些连续性假设，所以我们最终证明了许多连续性结果，并且有一个专门用于这个任务的`连续性`策略。让我们证明一个在下面的练习中需要的连续性陈述。注意，Lean 知道如何将两个度量空间的产品视为度量空间，因此考虑从`X × X`到`ℝ`的连续函数是有意义的。特别是（未展开的）距离函数是这样的函数。

```py
example  {X  Y  :  Type*}  [MetricSpace  X]  [MetricSpace  Y]  {f  :  X  →  Y}  (hf  :  Continuous  f)  :
  Continuous  fun  p  :  X  ×  X  ↦  dist  (f  p.1)  (f  p.2)  :=  by  continuity 
```

这个策略有点慢，因此了解如何手动完成它也很有用。我们首先需要使用`fun p : X × X ↦ f p.1`是连续的，因为它是由`f`（根据假设`hf`是连续的）和投影`prod.fst`（其连续性是引理`continuous_fst`的内容）组成的组合。组合属性是`Continuous.comp`，它在`Continuous`命名空间中，因此我们可以使用点符号将`Continuous.comp hf continuous_fst`压缩为`hf.comp continuous_fst`，这实际上更易读，因为它真正地读作组合我们的假设和我们的引理。我们可以对第二个组件做同样的处理，以得到`fun p : X × X ↦ f p.2`的连续性。然后我们使用`Continuous.prod_mk`将这些连续性组装起来，得到`(hf.comp continuous_fst).prod_mk (hf.comp continuous_snd) : Continuous (fun p : X × X ↦ (f p.1, f p.2))`，然后再进行一次组合，以得到我们的完整证明。

```py
example  {X  Y  :  Type*}  [MetricSpace  X]  [MetricSpace  Y]  {f  :  X  →  Y}  (hf  :  Continuous  f)  :
  Continuous  fun  p  :  X  ×  X  ↦  dist  (f  p.1)  (f  p.2)  :=
  continuous_dist.comp  ((hf.comp  continuous_fst).prodMk  (hf.comp  continuous_snd)) 
```

通过`Continuous.comp`将`Continuous.prod_mk`和`continuous_dist`结合起来感觉有点笨拙，即使像上面那样大量使用点符号也是如此。一个更严重的问题是，这个漂亮的证明需要大量的规划。Lean 接受上述证明项，因为它是一个完整的项，证明了与我们的目标定义等价的说法，关键的定义展开是函数的组合。确实，我们的目标函数`fun p : X × X ↦ dist (f p.1) (f p.2)`并没有以组合的形式呈现。我们提供的证明项证明了`dist ∘ (fun p : X × X ↦ (f p.1, f p.2))`的连续性，这恰好与我们的目标函数定义等价。但如果我们尝试使用从`apply continuous_dist.comp`开始的策略逐步构建这个证明，Lean 的详细说明器将无法识别组合，并拒绝应用这个引理。当涉及到类型乘积时，这个问题尤其严重。

这里应用更好的引理是`Continuous.dist {f g : X → Y} : Continuous f → Continuous g → Continuous (fun x ↦ dist (f x) (g x))`，这对于 Lean 的详细说明器来说更友好，并且在直接提供完整的证明项时也提供了更短的证明，如下面的两个新证明所示：

```py
example  {X  Y  :  Type*}  [MetricSpace  X]  [MetricSpace  Y]  {f  :  X  →  Y}  (hf  :  Continuous  f)  :
  Continuous  fun  p  :  X  ×  X  ↦  dist  (f  p.1)  (f  p.2)  :=  by
  apply  Continuous.dist
  exact  hf.comp  continuous_fst
  exact  hf.comp  continuous_snd

example  {X  Y  :  Type*}  [MetricSpace  X]  [MetricSpace  Y]  {f  :  X  →  Y}  (hf  :  Continuous  f)  :
  Continuous  fun  p  :  X  ×  X  ↦  dist  (f  p.1)  (f  p.2)  :=
  (hf.comp  continuous_fst).dist  (hf.comp  continuous_snd) 
```

注意，如果没有来自组合的详细说明问题，另一种压缩我们证明的方法是使用`Continuous.prod_map`，这在某些情况下很有用，并提供了一个替代的证明项`continuous_dist.comp (hf.prod_map hf)`，这甚至更短，更容易输入。

由于在更好的详细说明版本和更短的输入版本之间做出选择是令人悲伤的，让我们用`Continuous.fst'`提供的最后一点压缩来结束这次讨论，它允许将`hf.comp continuous_fst`压缩为`hf.fst'`（以及`snd`相同），并得到我们的最终证明，现在几乎到了难以理解的程度。

```py
example  {X  Y  :  Type*}  [MetricSpace  X]  [MetricSpace  Y]  {f  :  X  →  Y}  (hf  :  Continuous  f)  :
  Continuous  fun  p  :  X  ×  X  ↦  dist  (f  p.1)  (f  p.2)  :=
  hf.fst'.dist  hf.snd' 
```

现在轮到你了，去证明一些连续性引理。在尝试了连续性策略之后，你需要`Continuous.add`、`continuous_pow`和`continuous_id`来手动完成。

```py
example  {f  :  ℝ  →  X}  (hf  :  Continuous  f)  :  Continuous  fun  x  :  ℝ  ↦  f  (x  ^  2  +  x)  :=
  sorry 
```

到目前为止，我们看到了连续性作为一个全局概念，但也可以定义在一点的连续性。

```py
example  {X  Y  :  Type*}  [MetricSpace  X]  [MetricSpace  Y]  (f  :  X  →  Y)  (a  :  X)  :
  ContinuousAt  f  a  ↔  ∀  ε  >  0,  ∃  δ  >  0,  ∀  {x},  dist  x  a  <  δ  →  dist  (f  x)  (f  a)  <  ε  :=
  Metric.continuousAt_iff 
```

### 11.2.2\. 球、开集和闭集

一旦我们有了距离函数，最重要的几何定义就是（开）球和闭球。

```py
variable  (r  :  ℝ)

example  :  Metric.ball  a  r  =  {  b  |  dist  b  a  <  r  }  :=
  rfl

example  :  Metric.closedBall  a  r  =  {  b  |  dist  b  a  ≤  r  }  :=
  rfl 
```

注意，这里的 r 是任何实数，没有符号限制。当然，一些陈述确实需要半径条件。

```py
example  (hr  :  0  <  r)  :  a  ∈  Metric.ball  a  r  :=
  Metric.mem_ball_self  hr

example  (hr  :  0  ≤  r)  :  a  ∈  Metric.closedBall  a  r  :=
  Metric.mem_closedBall_self  hr 
```

一旦我们有了球，我们就可以定义开集。实际上，它们是在下一节中讨论的更一般设置中定义的，但我们有引理将定义重新表述为球的形式。

```py
example  (s  :  Set  X)  :  IsOpen  s  ↔  ∀  x  ∈  s,  ∃  ε  >  0,  Metric.ball  x  ε  ⊆  s  :=
  Metric.isOpen_iff 
```

然后，闭集是补集为开的集合。它们的重要性质是它们在极限下是封闭的。集合的闭包是包含它的最小闭集。

```py
example  {s  :  Set  X}  :  IsClosed  s  ↔  IsOpen  (sᶜ)  :=
  isOpen_compl_iff.symm

example  {s  :  Set  X}  (hs  :  IsClosed  s)  {u  :  ℕ  →  X}  (hu  :  Tendsto  u  atTop  (𝓝  a))
  (hus  :  ∀  n,  u  n  ∈  s)  :  a  ∈  s  :=
  hs.mem_of_tendsto  hu  (Eventually.of_forall  hus)

example  {s  :  Set  X}  :  a  ∈  closure  s  ↔  ∀  ε  >  0,  ∃  b  ∈  s,  a  ∈  Metric.ball  b  ε  :=
  Metric.mem_closure_iff 
```

在不使用 mem_closure_iff_seq_limit 的情况下完成下一项练习

```py
example  {u  :  ℕ  →  X}  (hu  :  Tendsto  u  atTop  (𝓝  a))  {s  :  Set  X}  (hs  :  ∀  n,  u  n  ∈  s)  :
  a  ∈  closure  s  :=  by
  sorry 
```

记住从过滤器部分，邻域过滤器在 Mathlib 中起着重要作用。在度量空间的情况下，关键点是球为这些过滤器提供了基。这里的主要引理是`Metric.nhds_basis_ball`和`Metric.nhds_basis_closedBall`，它们为正半径的开球和闭球声明了这一点。中心点是隐含的参数，因此我们可以像以下示例中那样调用`Filter.HasBasis.mem_iff`。

```py
example  {x  :  X}  {s  :  Set  X}  :  s  ∈  𝓝  x  ↔  ∃  ε  >  0,  Metric.ball  x  ε  ⊆  s  :=
  Metric.nhds_basis_ball.mem_iff

example  {x  :  X}  {s  :  Set  X}  :  s  ∈  𝓝  x  ↔  ∃  ε  >  0,  Metric.closedBall  x  ε  ⊆  s  :=
  Metric.nhds_basis_closedBall.mem_iff 
```

### 11.2.3\. 紧致性

紧致性是一个重要的拓扑概念。它区分了度量空间中具有与实数线段相同性质的真子集与其他区间：

+   任何在紧集上取值的序列都有一个子序列在这个集合中收敛。

+   任何在非空紧集上定义且取值为实数的连续函数都是有界的，并且在其某个地方达到其界限（这被称为极值定理）。

+   紧集是闭集。

让我们先检查实数单位区间确实是一个紧集，然后检查一般度量空间中紧集的上述命题。在第二个命题中，我们只需要给定集合上的连续性，因此我们将使用`ContinuousOn`而不是`Continuous`，并且我们将分别给出最小值和最大值的单独陈述。当然，所有这些结果都是从更一般的版本中推导出来的，其中一些将在后面的章节中讨论。

```py
example  :  IsCompact  (Set.Icc  0  1  :  Set  ℝ)  :=
  isCompact_Icc

example  {s  :  Set  X}  (hs  :  IsCompact  s)  {u  :  ℕ  →  X}  (hu  :  ∀  n,  u  n  ∈  s)  :
  ∃  a  ∈  s,  ∃  φ  :  ℕ  →  ℕ,  StrictMono  φ  ∧  Tendsto  (u  ∘  φ)  atTop  (𝓝  a)  :=
  hs.tendsto_subseq  hu

example  {s  :  Set  X}  (hs  :  IsCompact  s)  (hs'  :  s.Nonempty)  {f  :  X  →  ℝ}
  (hfs  :  ContinuousOn  f  s)  :
  ∃  x  ∈  s,  ∀  y  ∈  s,  f  x  ≤  f  y  :=
  hs.exists_isMinOn  hs'  hfs

example  {s  :  Set  X}  (hs  :  IsCompact  s)  (hs'  :  s.Nonempty)  {f  :  X  →  ℝ}
  (hfs  :  ContinuousOn  f  s)  :
  ∃  x  ∈  s,  ∀  y  ∈  s,  f  y  ≤  f  x  :=
  hs.exists_isMaxOn  hs'  hfs

example  {s  :  Set  X}  (hs  :  IsCompact  s)  :  IsClosed  s  :=
  hs.isClosed 
```

我们还可以指定度量空间是全局紧致的，使用一个额外的`Prop`值类型类：

```py
example  {X  :  Type*}  [MetricSpace  X]  [CompactSpace  X]  :  IsCompact  (univ  :  Set  X)  :=
  isCompact_univ 
```

在紧致度量空间中，任何闭集都是紧集，这是`IsClosed.isCompact`。

### 11.2.4\. 一致连续函数

现在我们转向度量空间上的均匀性概念：一致连续函数、柯西序列和完备性。这些都是在更一般的环境中定义的，但我们有在度量名称空间中的引理来访问它们的元素定义。我们首先从均匀连续性开始。

```py
example  {X  :  Type*}  [MetricSpace  X]  {Y  :  Type*}  [MetricSpace  Y]  {f  :  X  →  Y}  :
  UniformContinuous  f  ↔
  ∀  ε  >  0,  ∃  δ  >  0,  ∀  {a  b  :  X},  dist  a  b  <  δ  →  dist  (f  a)  (f  b)  <  ε  :=
  Metric.uniformContinuous_iff 
```

为了练习操作所有这些定义，我们将证明从紧致度量空间到度量空间的连续函数是均匀连续的（我们将在后面的章节中看到更一般的形式）。

我们首先给出一个非正式的草图。设 `f : X → Y` 是从紧致度量空间到度量空间的连续函数。我们固定 `ε > 0` 并开始寻找某个 `δ`。

设 `φ : X × X → ℝ := fun p ↦ dist (f p.1) (f p.2)` 和 `K := { p : X × X | ε ≤ φ p }`。观察 `φ` 是连续的，因为 `f` 和距离都是连续的。而且 `K` 显然是闭集（使用 `isClosed_le`），因此 `X` 是紧致的。

然后，我们使用 `eq_empty_or_nonempty` 讨论两种可能性。如果 `K` 是空的，那么显然我们已经完成了（例如，我们可以将 `δ` 设置为 `1`）。所以让我们假设 `K` 不是空的，并使用极值定理来选择 `(x₀, x₁)`，它达到距离函数在 `K` 上的下确界。然后我们可以将 `δ` 设置为 `dist x₀ x₁` 并检查一切是否正常工作。

```py
example  {X  :  Type*}  [MetricSpace  X]  [CompactSpace  X]
  {Y  :  Type*}  [MetricSpace  Y]  {f  :  X  →  Y}
  (hf  :  Continuous  f)  :  UniformContinuous  f  :=  by
  sorry 
```

### 11.2.5\. 完备性

在度量空间中，柯西序列是一个其项彼此越来越接近的序列。有几种等价的方式来表述这个想法。特别是收敛序列是柯西序列。逆命题只在所谓的 *完备* 空间中成立。

```py
example  (u  :  ℕ  →  X)  :
  CauchySeq  u  ↔  ∀  ε  >  0,  ∃  N  :  ℕ,  ∀  m  ≥  N,  ∀  n  ≥  N,  dist  (u  m)  (u  n)  <  ε  :=
  Metric.cauchySeq_iff

example  (u  :  ℕ  →  X)  :
  CauchySeq  u  ↔  ∀  ε  >  0,  ∃  N  :  ℕ,  ∀  n  ≥  N,  dist  (u  n)  (u  N)  <  ε  :=
  Metric.cauchySeq_iff'

example  [CompleteSpace  X]  (u  :  ℕ  →  X)  (hu  :  CauchySeq  u)  :
  ∃  x,  Tendsto  u  atTop  (𝓝  x)  :=
  cauchySeq_tendsto_of_complete  hu 
```

我们将通过证明一个方便的判据来练习使用这个定义，这个判据是 Mathlib 中出现的一个判据的特殊情况。这也是练习在几何环境中使用大和的好机会。除了过滤器部分的解释外，你可能还需要 `tendsto_pow_atTop_nhds_zero_of_lt_one`、`Tendsto.mul` 和 `dist_le_range_sum_dist`。

```py
theorem  cauchySeq_of_le_geometric_two'  {u  :  ℕ  →  X}
  (hu  :  ∀  n  :  ℕ,  dist  (u  n)  (u  (n  +  1))  ≤  (1  /  2)  ^  n)  :  CauchySeq  u  :=  by
  rw  [Metric.cauchySeq_iff']
  intro  ε  ε_pos
  obtain  ⟨N,  hN⟩  :  ∃  N  :  ℕ,  1  /  2  ^  N  *  2  <  ε  :=  by  sorry
  use  N
  intro  n  hn
  obtain  ⟨k,  rfl  :  n  =  N  +  k⟩  :=  le_iff_exists_add.mp  hn
  calc
  dist  (u  (N  +  k))  (u  N)  =  dist  (u  (N  +  0))  (u  (N  +  k))  :=  sorry
  _  ≤  ∑  i  ∈  range  k,  dist  (u  (N  +  i))  (u  (N  +  (i  +  1)))  :=  sorry
  _  ≤  ∑  i  ∈  range  k,  (1  /  2  :  ℝ)  ^  (N  +  i)  :=  sorry
  _  =  1  /  2  ^  N  *  ∑  i  ∈  range  k,  (1  /  2  :  ℝ)  ^  i  :=  sorry
  _  ≤  1  /  2  ^  N  *  2  :=  sorry
  _  <  ε  :=  sorry 
```

我们已经准备好本节的最终挑战：完备度量空间的 Baire 定理！下面的证明框架展示了有趣的技术。它使用了感叹号变体的 `choose` 策略（你应该尝试移除这个感叹号）并展示了如何在证明中使用 `Nat.rec_on` 递归地定义某物。

```py
open  Metric

example  [CompleteSpace  X]  (f  :  ℕ  →  Set  X)  (ho  :  ∀  n,  IsOpen  (f  n))  (hd  :  ∀  n,  Dense  (f  n))  :
  Dense  (⋂  n,  f  n)  :=  by
  let  B  :  ℕ  →  ℝ  :=  fun  n  ↦  (1  /  2)  ^  n
  have  Bpos  :  ∀  n,  0  <  B  n
  sorry
  /- Translate the density assumption into two functions `center` and `radius` associating
 to any n, x, δ, δpos a center and a positive radius such that
 `closedBall center radius` is included both in `f n` and in `closedBall x δ`.
 We can also require `radius ≤ (1/2)^(n+1)`, to ensure we get a Cauchy sequence later. -/
  have  :
  ∀  (n  :  ℕ)  (x  :  X),
  ∀  δ  >  0,  ∃  y  :  X,  ∃  r  >  0,  r  ≤  B  (n  +  1)  ∧  closedBall  y  r  ⊆  closedBall  x  δ  ∩  f  n  :=
  by  sorry
  choose!  center  radius  Hpos  HB  Hball  using  this
  intro  x
  rw  [mem_closure_iff_nhds_basis  nhds_basis_closedBall]
  intro  ε  εpos
  /- `ε` is positive. We have to find a point in the ball of radius `ε` around `x`
 belonging to all `f n`. For this, we construct inductively a sequence
 `F n = (c n, r n)` such that the closed ball `closedBall (c n) (r n)` is included
 in the previous ball and in `f n`, and such that `r n` is small enough to ensure
 that `c n` is a Cauchy sequence. Then `c n` converges to a limit which belongs
 to all the `f n`. -/
  let  F  :  ℕ  →  X  ×  ℝ  :=  fun  n  ↦
  Nat.recOn  n  (Prod.mk  x  (min  ε  (B  0)))
  fun  n  p  ↦  Prod.mk  (center  n  p.1  p.2)  (radius  n  p.1  p.2)
  let  c  :  ℕ  →  X  :=  fun  n  ↦  (F  n).1
  let  r  :  ℕ  →  ℝ  :=  fun  n  ↦  (F  n).2
  have  rpos  :  ∀  n,  0  <  r  n  :=  by  sorry
  have  rB  :  ∀  n,  r  n  ≤  B  n  :=  by  sorry
  have  incl  :  ∀  n,  closedBall  (c  (n  +  1))  (r  (n  +  1))  ⊆  closedBall  (c  n)  (r  n)  ∩  f  n  :=  by
  sorry
  have  cdist  :  ∀  n,  dist  (c  n)  (c  (n  +  1))  ≤  B  n  :=  by  sorry
  have  :  CauchySeq  c  :=  cauchySeq_of_le_geometric_two'  cdist
  -- as the sequence `c n` is Cauchy in a complete space, it converges to a limit `y`.
  rcases  cauchySeq_tendsto_of_complete  this  with  ⟨y,  ylim⟩
  -- this point `y` will be the desired point. We will check that it belongs to all
  -- `f n` and to `ball x ε`.
  use  y
  have  I  :  ∀  n,  ∀  m  ≥  n,  closedBall  (c  m)  (r  m)  ⊆  closedBall  (c  n)  (r  n)  :=  by  sorry
  have  yball  :  ∀  n,  y  ∈  closedBall  (c  n)  (r  n)  :=  by  sorry
  sorry 
```  ## 11.3\. 拓扑空间

### 11.3.1\. 基础

现在我们提高一般性，引入拓扑空间。我们将回顾定义拓扑空间的两种主要方法，然后解释拓扑空间范畴比度量空间范畴表现得更好。请注意，我们在这里不会使用 Mathlib 范畴论，而只是拥有某种范畴性的观点。

考虑从度量空间到拓扑空间的过渡的第一种方法是，我们只记住开集（或等价地闭集）的概念。从这个角度来看，拓扑空间是一种类型，它配备了一组称为开集的集合。这个集合必须满足下面提出的若干公理（这个集合略微冗余，但我们将忽略这一点）。

```py
section
variable  {X  :  Type*}  [TopologicalSpace  X]

example  :  IsOpen  (univ  :  Set  X)  :=
  isOpen_univ

example  :  IsOpen  (∅  :  Set  X)  :=
  isOpen_empty

example  {ι  :  Type*}  {s  :  ι  →  Set  X}  (hs  :  ∀  i,  IsOpen  (s  i))  :  IsOpen  (⋃  i,  s  i)  :=
  isOpen_iUnion  hs

example  {ι  :  Type*}  [Fintype  ι]  {s  :  ι  →  Set  X}  (hs  :  ∀  i,  IsOpen  (s  i))  :
  IsOpen  (⋂  i,  s  i)  :=
  isOpen_iInter_of_finite  hs 
```

闭集定义为补集是开集的集合。在拓扑空间之间的函数（全局）连续，如果所有开集的前像都是开集。

```py
variable  {Y  :  Type*}  [TopologicalSpace  Y]

example  {f  :  X  →  Y}  :  Continuous  f  ↔  ∀  s,  IsOpen  s  →  IsOpen  (f  ⁻¹'  s)  :=
  continuous_def 
```

通过这个定义，我们已看到，与度量空间相比，拓扑空间只保留足够的信息来讨论连续函数：一个类型上的两个拓扑结构相同，当且仅当它们具有相同的连续函数（实际上，如果两个结构具有相同的开集，恒等函数在两个方向上都将连续）。 

然而，一旦我们转向点的连续性，我们就看到了基于开集的方法的局限性。在 Mathlib 中，我们经常将拓扑空间视为带有附加在每个点`x`上的邻域滤波器`𝓝 x`的类型（相应的函数`X → Filter X`满足某些条件，这些条件将在下面进一步解释）。记住从滤波器部分，这些小玩意儿扮演两个相关的角色。首先，`𝓝 x`被视为接近`x`的`X`中点的广义集合。然后，它被视为为任何谓词`P : X → Prop`提供一种方式，即这个谓词对足够接近`x`的点成立。让我们声明`f : X → Y`在`x`处是连续的。纯粹基于滤波器的方式是说，在`f`的直接像下，接近`x`的点的广义集合包含在接近`f x`的点的广义集合中。回想一下，这可以表示为`map f (𝓝 x) ≤ 𝓝 (f x)`或`Tendsto f (𝓝 x) (𝓝 (f x))`。

```py
example  {f  :  X  →  Y}  {x  :  X}  :  ContinuousAt  f  x  ↔  map  f  (𝓝  x)  ≤  𝓝  (f  x)  :=
  Iff.rfl 
```

也可以使用两种邻域（视为普通集合）和一种邻域滤波器（视为广义集合）来表述：对于`f x`的任何邻域`U`，所有接近`x`的点都被发送到`U`。请注意，证明仍然是`Iff.rfl`，这种观点在定义上是等同于前一种的。

```py
example  {f  :  X  →  Y}  {x  :  X}  :  ContinuousAt  f  x  ↔  ∀  U  ∈  𝓝  (f  x),  ∀ᶠ  x  in  𝓝  x,  f  x  ∈  U  :=
  Iff.rfl 
```

现在我们解释如何从一个观点转到另一个观点。从开集的角度来看，我们可以简单地定义`𝓝 x`的成员为包含`x`的某个开集的集合。

```py
example  {x  :  X}  {s  :  Set  X}  :  s  ∈  𝓝  x  ↔  ∃  t,  t  ⊆  s  ∧  IsOpen  t  ∧  x  ∈  t  :=
  mem_nhds_iff 
```

要朝相反的方向进行，我们需要讨论`𝓝 : X → Filter X`必须满足的条件，以便成为拓扑的邻域函数。

第一个约束是，将`𝓝 x`视为一个广义集合时，它包含将`{x}`视为广义集合`纯 x`的集合（解释这个奇怪的名字会太分散注意力，所以我们现在简单地接受它）。另一种说法是，如果一个谓词在`x`附近的点成立，那么它在`x`处也成立。

```py
example  (x  :  X)  :  pure  x  ≤  𝓝  x  :=
  pure_le_nhds  x

example  (x  :  X)  (P  :  X  →  Prop)  (h  :  ∀ᶠ  y  in  𝓝  x,  P  y)  :  P  x  :=
  h.self_of_nhds 
```

然后，一个更微妙的要求是，对于任何谓词`P : X → Prop`和任何`x`，如果`P y`在`y`接近`x`时成立，那么对于接近`x`和`y`的`z`，`P z`也成立。更精确地说，我们有：

```py
example  {P  :  X  →  Prop}  {x  :  X}  (h  :  ∀ᶠ  y  in  𝓝  x,  P  y)  :  ∀ᶠ  y  in  𝓝  x,  ∀ᶠ  z  in  𝓝  y,  P  z  :=
  eventually_eventually_nhds.mpr  h 
```

这两个结果描述了函数 `X → Filter X`，它们是 `X` 上拓扑空间结构的邻域函数。仍然存在一个函数 `TopologicalSpace.mkOfNhds : (X → Filter X) → TopologicalSpace X`，但它只有在满足上述两个约束的情况下才会将其输入作为邻域函数返回。更精确地说，我们有一个引理 `TopologicalSpace.nhds_mkOfNhds`，它以不同的方式表述，而我们的下一个练习将从我们上述的表述中推导出这种不同方式。

```py
example  {α  :  Type*}  (n  :  α  →  Filter  α)  (H₀  :  ∀  a,  pure  a  ≤  n  a)
  (H  :  ∀  a  :  α,  ∀  p  :  α  →  Prop,  (∀ᶠ  x  in  n  a,  p  x)  →  ∀ᶠ  y  in  n  a,  ∀ᶠ  x  in  n  y,  p  x)  :
  ∀  a,  ∀  s  ∈  n  a,  ∃  t  ∈  n  a,  t  ⊆  s  ∧  ∀  a'  ∈  t,  s  ∈  n  a'  :=  by
  sorry
end 
```

注意，`TopologicalSpace.mkOfNhds` 并不经常使用，但了解在拓扑空间结构中邻域过滤器究竟意味着什么仍然是很好的。

为了有效地在 Mathlib 中使用拓扑空间，我们需要知道的是，我们使用了 `TopologicalSpace : Type u → Type u` 的许多形式性质。从纯粹数学的角度来看，这些形式性质是解释拓扑空间如何解决度量空间问题的非常干净的方法。从这个角度来看，拓扑空间解决的问题在于度量空间几乎不具有函子性，并且在一般上具有很差的范畴性质。这还基于已经讨论的事实，即度量空间包含大量与拓扑无关的几何信息。

让我们先关注函子性。可以在子集上诱导度量空间结构，或者等价地，可以通过一个注入映射将其后拉。但这几乎就是全部了。它们不能通过一般映射或前推，甚至不能通过满射映射进行后拉或前推。

特别地，在度量空间的商或不可数度量空间的积上放置一个有意义的距离是没有意义的。例如，考虑类型 `ℝ → ℝ`，它被视为由 `ℝ` 索引的 `ℝ` 的复制品的积。我们希望说，函数序列逐点收敛是一个值得尊重的收敛概念。但在 `ℝ → ℝ` 上没有距离可以给出这种收敛概念。相关地，没有距离可以保证映射 `f : X → (ℝ → ℝ)` 在且仅当对于每个 `t : ℝ`，`fun x ↦ f x t` 是连续的时是连续的。

我们现在回顾用于解决所有这些问题的数据。首先，我们可以使用任何映射 `f : X → Y` 来从一个方向推动或拉动拓扑到另一个方向。这两个操作形成一个伽罗瓦连接。

```py
variable  {X  Y  :  Type*}

example  (f  :  X  →  Y)  :  TopologicalSpace  X  →  TopologicalSpace  Y  :=
  TopologicalSpace.coinduced  f

example  (f  :  X  →  Y)  :  TopologicalSpace  Y  →  TopologicalSpace  X  :=
  TopologicalSpace.induced  f

example  (f  :  X  →  Y)  (T_X  :  TopologicalSpace  X)  (T_Y  :  TopologicalSpace  Y)  :
  TopologicalSpace.coinduced  f  T_X  ≤  T_Y  ↔  T_X  ≤  TopologicalSpace.induced  f  T_Y  :=
  coinduced_le_iff_le_induced 
```

这些操作与函数的复合是兼容的。像往常一样，前推是协变的，后拉是反对变的，参见 `coinduced_compose` 和 `induced_compose`。在纸上，我们将使用符号 $f_*T$ 表示 `TopologicalSpace.coinduced f T`，以及 $f^*T$ 表示 `TopologicalSpace.induced f T`。

接下来的一大块内容是针对任何给定的结构在`TopologicalSpace X`上的完整格结构。如果你认为拓扑主要是开集的数据，那么你期望`TopologicalSpace X`上的顺序关系来自`Set (Set X)`，即你期望如果集合`u`对于`t'`是开集，那么它对于`t`也是开集，你就期望`t ≤ t'`。然而，我们已经知道 Mathlib 更关注邻域而不是开集，所以对于任何`x : X`，我们希望从拓扑空间到邻域的映射`fun T : TopologicalSpace X ↦ @nhds X T x`是顺序保持的。我们还知道`Filter X`上的顺序关系被设计用来确保顺序保持的`principal : Set X → Filter X`，允许将过滤器视为广义集合。因此，我们在`TopologicalSpace X`上使用的顺序关系与来自`Set (Set X)`的顺序关系相反。

```py
example  {T  T'  :  TopologicalSpace  X}  :  T  ≤  T'  ↔  ∀  s,  T'.IsOpen  s  →  T.IsOpen  s  :=
  Iff.rfl 
```

现在，我们可以通过结合推前（或拉回）操作与顺序关系来恢复连续性。

```py
example  (T_X  :  TopologicalSpace  X)  (T_Y  :  TopologicalSpace  Y)  (f  :  X  →  Y)  :
  Continuous  f  ↔  TopologicalSpace.coinduced  f  T_X  ≤  T_Y  :=
  continuous_iff_coinduced_le 
```

通过这个定义和推前与复合的兼容性，我们免费获得了这样的泛性质：对于任何拓扑空间$Z$，如果函数$g : Y → Z$在拓扑$f_*T_X$下是连续的，当且仅当$g ∘ f$是连续的。

$$\begin{split}g \text{ 连续 } &⇔ g_*(f_*T_X) ≤ T_Z \\ &⇔ (g ∘ f)_* T_X ≤ T_Z \\ &⇔ g ∘ f \text{ 连续}\end{split}$$

```py
example  {Z  :  Type*}  (f  :  X  →  Y)  (T_X  :  TopologicalSpace  X)  (T_Z  :  TopologicalSpace  Z)
  (g  :  Y  →  Z)  :
  @Continuous  Y  Z  (TopologicalSpace.coinduced  f  T_X)  T_Z  g  ↔
  @Continuous  X  Z  T_X  T_Z  (g  ∘  f)  :=  by
  rw  [continuous_iff_coinduced_le,  coinduced_compose,  continuous_iff_coinduced_le] 
```

因此，我们已经得到了商拓扑（使用投影映射作为`f`）。这并不是因为`TopologicalSpace X`对于所有`X`都是完整的格。现在让我们看看所有这些结构如何通过抽象的胡言乱语证明乘积拓扑的存在。我们上面考虑了`ℝ → ℝ`的情况，但现在让我们考虑一般情况`Π i, X i`对于某个`ι : Type*`和`X : ι → Type*`。我们希望对于任何拓扑空间`Z`和任何函数`f : Z → Π i, X i`，如果对于所有`i`，`(fun x ↦ x i) ∘ f`是连续的，那么`f`是连续的。让我们使用表示投影`(fun (x : Π i, X i) ↦ x i)`的符号$p_i$来“在纸上”探索这个约束：

$$\begin{split}(∀ i, p_i ∘ f \text{ 连续}) &⇔ ∀ i, (p_i ∘ f)_* T_Z ≤ T_{X_i} \\ &⇔ ∀ i, (p_i)_* f_* T_Z ≤ T_{X_i}\\ &⇔ ∀ i, f_* T_Z ≤ (p_i)^*T_{X_i}\\ &⇔ f_* T_Z ≤ \inf \left[(p_i)^*T_{X_i}\right]\end{split}$$

因此，我们看到我们希望在`Π i, X i`上的拓扑是什么：

```py
example  (ι  :  Type*)  (X  :  ι  →  Type*)  (T_X  :  ∀  i,  TopologicalSpace  (X  i))  :
  (Pi.topologicalSpace  :  TopologicalSpace  (∀  i,  X  i))  =
  ⨅  i,  TopologicalSpace.induced  (fun  x  ↦  x  i)  (T_X  i)  :=
  rfl 
```

这就结束了我们对 Mathlib 如何认为拓扑空间通过成为一个更函数化的理论和为任何固定类型提供完整的格结构来修复度量空间理论的缺陷的考察。

### 11.3.2\. 分离与可数性

我们看到拓扑空间范畴具有非常好的性质。为此所付出的代价是存在相当病态的拓扑空间。你可以对拓扑空间做出一系列假设以确保其行为更接近度量空间。最重要的是 `T2Space`，也称为“豪斯多夫”，这将确保极限是唯一的。更强的分离性质是 `T3Space`，它还确保了正则空间性质：每个点都有一个闭邻域基。

```py
example  [TopologicalSpace  X]  [T2Space  X]  {u  :  ℕ  →  X}  {a  b  :  X}  (ha  :  Tendsto  u  atTop  (𝓝  a))
  (hb  :  Tendsto  u  atTop  (𝓝  b))  :  a  =  b  :=
  tendsto_nhds_unique  ha  hb

example  [TopologicalSpace  X]  [RegularSpace  X]  (a  :  X)  :
  (𝓝  a).HasBasis  (fun  s  :  Set  X  ↦  s  ∈  𝓝  a  ∧  IsClosed  s)  id  :=
  closed_nhds_basis  a 
```

注意，在每一个拓扑空间中，每个点都有一个开邻域基，这是定义所要求的。

```py
example  [TopologicalSpace  X]  {x  :  X}  :
  (𝓝  x).HasBasis  (fun  t  :  Set  X  ↦  t  ∈  𝓝  x  ∧  IsOpen  t)  id  :=
  nhds_basis_opens'  x 
```

我们现在的目标是证明一个基本定理，该定理允许通过连续性进行扩展。从布尔巴基的《一般拓扑学》一书，I.8.5，定理 1（仅考虑非平凡蕴含）：

设 $X$ 是一个拓扑空间，$A$ 是 $X$ 的一个稠密子集，$f : A → Y$ 是将 $A$ 映射到 $T_3$ 空间 $Y$ 的连续映射。如果对于 $X$ 中的每个 $x$，当 $y$ 在 $A$ 中趋向于 $x$ 时，$f(y)$ 在 $Y$ 中趋向于一个极限，那么存在一个将 $f$ 扩展到 $X$ 的连续扩展 $φ$。

实际上，Mathlib 包含了上述引理的一个更一般版本，`IsDenseInducing.continuousAt_extend`，但我们将坚持使用布尔巴基的版本。

记住，给定 `A : Set X`，`↥A` 是与 `A` 关联的子类型，并且 Lean 会自动在需要时插入那个有趣的向上箭头。而（包含）强制映射是 `(↑) : A → X`。假设“在 $A$ 中趋向于 $x$”对应于拉回过滤器 `comap (↑) (𝓝 x)`。

让我们先证明一个辅助引理，将其提取出来以简化上下文（特别是我们在这里不需要 $Y$ 是一个拓扑空间）。

```py
theorem  aux  {X  Y  A  :  Type*}  [TopologicalSpace  X]  {c  :  A  →  X}
  {f  :  A  →  Y}  {x  :  X}  {F  :  Filter  Y}
  (h  :  Tendsto  f  (comap  c  (𝓝  x))  F)  {V'  :  Set  Y}  (V'_in  :  V'  ∈  F)  :
  ∃  V  ∈  𝓝  x,  IsOpen  V  ∧  c  ⁻¹'  V  ⊆  f  ⁻¹'  V'  :=  by
  sorry 
```

让我们现在转向连续扩展定理的主要证明。

当 Lean 需要在 `↥A` 上使用拓扑时，它将自动使用诱导拓扑。唯一相关的引理是 `nhds_induced (↑) : ∀ a : ↥A, 𝓝 a = comap (↑) (𝓝 ↑a)`（这实际上是一个关于诱导拓扑的通用引理）。

证明概要是：

主要假设和选择公理给出一个函数 `φ`，使得 `∀ x, Tendsto f (comap (↑) (𝓝 x)) (𝓝 (φ x))`（因为 `Y` 是豪斯多夫的，`φ` 是完全确定的，但我们不会在尝试证明 `φ` 确实扩展 `f` 之前需要它）。

让我们先证明`φ`是连续的。固定任意的`x : X`。由于`Y`是正则的，我们只需检查对于`φ x`的每一个*闭*邻域`V'`，`φ ⁻¹' V' ∈ 𝓝 x`。极限假设给出了（通过上面的辅助引理）一些`V ∈ 𝓝 x`，使得`IsOpen V ∧ (↑) ⁻¹' V ⊆ f ⁻¹' V'`。由于`V ∈ 𝓝 x`，我们只需证明`V ⊆ φ ⁻¹' V'`，即`∀ y ∈ V, φ y ∈ V'`。让我们固定`V`中的`y`。因为`V`是*开*的，所以它是`y`的一个邻域。特别是`(↑) ⁻¹' V ∈ comap (↑) (𝓝 y)`，并且更明显`f ⁻¹' V' ∈ comap (↑) (𝓝 y)`。此外，`comap (↑) (𝓝 y) ≠ ⊥`，因为`A`是稠密的。因为我们知道`Tendsto f (comap (↑) (𝓝 y)) (𝓝 (φ y))`，这表明`φ y ∈ closure V'`，并且由于`V'`是闭的，我们证明了`φ y ∈ V'`。

剩下的工作是证明`φ`扩展了`f`。这是`f`的连续性进入讨论的地方，以及`Y`是豪斯多夫的事实。

```py
example  [TopologicalSpace  X]  [TopologicalSpace  Y]  [T3Space  Y]  {A  :  Set  X}
  (hA  :  ∀  x,  x  ∈  closure  A)  {f  :  A  →  Y}  (f_cont  :  Continuous  f)
  (hf  :  ∀  x  :  X,  ∃  c  :  Y,  Tendsto  f  (comap  (↑)  (𝓝  x))  (𝓝  c))  :
  ∃  φ  :  X  →  Y,  Continuous  φ  ∧  ∀  a  :  A,  φ  a  =  f  a  :=  by
  sorry

#check  HasBasis.tendsto_right_iff 
```

除了分离性质，你可以在拓扑空间上做出的主要假设是可数性假设，以使其更接近度量空间。主要的一个是第一可数性，要求每个点都有一个可数的邻域基。特别是这保证了集合的闭包可以用序列来理解。

```py
example  [TopologicalSpace  X]  [FirstCountableTopology  X]
  {s  :  Set  X}  {a  :  X}  :
  a  ∈  closure  s  ↔  ∃  u  :  ℕ  →  X,  (∀  n,  u  n  ∈  s)  ∧  Tendsto  u  atTop  (𝓝  a)  :=
  mem_closure_iff_seq_limit 
```

### 11.3.3\. 紧致性

现在我们来讨论拓扑空间中紧致性的定义。通常有几种思考方式，Mathlib 选择了过滤器版本。

我们首先需要定义过滤器的聚点。给定一个拓扑空间`X`上的过滤器`F`，如果`F`作为一个广义集与接近`x`的点的广义集非空交集，那么点`x : X`是`F`的聚点。

然后，我们可以这样说，如果`s`中的每一个非空广义集`F`（即`F ≤ 𝓟 s`），在`s`中都有一个聚点，那么`s`是紧致的。

```py
variable  [TopologicalSpace  X]

example  {F  :  Filter  X}  {x  :  X}  :  ClusterPt  x  F  ↔  NeBot  (𝓝  x  ⊓  F)  :=
  Iff.rfl

example  {s  :  Set  X}  :
  IsCompact  s  ↔  ∀  (F  :  Filter  X)  [NeBot  F],  F  ≤  𝓟  s  →  ∃  a  ∈  s,  ClusterPt  a  F  :=
  Iff.rfl 
```

例如，如果`F`是`map u atTop`，即`u : ℕ → X`的`atTop`的像，`atTop`是极大的自然数的广义集，那么`F ≤ 𝓟 s`的假设意味着对于足够大的`n`，`u n`属于`s`。说`x`是`map u atTop`的聚点意味着非常大的数的像与接近`x`的点的集合相交。如果`𝓝 x`有一个可数基，我们可以将其解释为说`u`有一个子序列收敛到`x`，这样我们就得到了度量空间中紧致性的样子。

```py
example  [FirstCountableTopology  X]  {s  :  Set  X}  {u  :  ℕ  →  X}  (hs  :  IsCompact  s)
  (hu  :  ∀  n,  u  n  ∈  s)  :  ∃  a  ∈  s,  ∃  φ  :  ℕ  →  ℕ,  StrictMono  φ  ∧  Tendsto  (u  ∘  φ)  atTop  (𝓝  a)  :=
  hs.tendsto_subseq  hu 
```

聚点与连续函数的行为良好。

```py
variable  [TopologicalSpace  Y]

example  {x  :  X}  {F  :  Filter  X}  {G  :  Filter  Y}  (H  :  ClusterPt  x  F)  {f  :  X  →  Y}
  (hfx  :  ContinuousAt  f  x)  (hf  :  Tendsto  f  F  G)  :  ClusterPt  (f  x)  G  :=
  ClusterPt.map  H  hfx  hf 
```

作为练习，我们将证明紧集在连续映射下的像也是紧致的。除了我们已经看到的，你还应该使用`Filter.push_pull`和`NeBot.of_map`。

```py
example  [TopologicalSpace  Y]  {f  :  X  →  Y}  (hf  :  Continuous  f)  {s  :  Set  X}  (hs  :  IsCompact  s)  :
  IsCompact  (f  ''  s)  :=  by
  intro  F  F_ne  F_le
  have  map_eq  :  map  f  (𝓟  s  ⊓  comap  f  F)  =  𝓟  (f  ''  s)  ⊓  F  :=  by  sorry
  have  Hne  :  (𝓟  s  ⊓  comap  f  F).NeBot  :=  by  sorry
  have  Hle  :  𝓟  s  ⊓  comap  f  F  ≤  𝓟  s  :=  inf_le_left
  sorry 
```

一个人也可以用开覆盖来表示紧致性：如果覆盖`s`的每一个开集族都有一个有限覆盖子族，那么`s`是紧致的。

```py
example  {ι  :  Type*}  {s  :  Set  X}  (hs  :  IsCompact  s)  (U  :  ι  →  Set  X)  (hUo  :  ∀  i,  IsOpen  (U  i))
  (hsU  :  s  ⊆  ⋃  i,  U  i)  :  ∃  t  :  Finset  ι,  s  ⊆  ⋃  i  ∈  t,  U  i  :=
  hs.elim_finite_subcover  U  hUo  hsU 
```  ## 11.1\. 过滤器

在类型`X`上的一个*过滤器*是`X`的集合的集合，它满足以下三个条件，我们将在下面详细说明。这个概念支持两个相关想法：

+   *极限*，包括上面讨论的所有类型的极限：序列的有限和无限极限，函数在一点或无穷远处的有限和无限极限等。

+   *最终发生的事情*，包括对于足够大的 `n : ℕ`，或在点 `x` 足够近的地方，或对于足够接近的点对，或在测度理论意义上的几乎处处发生的事情。类似地，过滤器也可以表达 *经常发生的事情* 的想法：对于任意大的 `n`，在任意给定点的任意邻域中等。

对应这些描述的过滤器将在本节后面定义，但我们可以先命名它们：

+   `(atTop : Filter ℕ)`，由包含 `{n | n ≥ N}` 的 `ℕ` 集合组成的过滤器（其中 `N` 是某个数）

+   `𝓝 x`，由拓扑空间中 `x` 的邻域组成

+   `𝓤 X`，由均匀空间（均匀空间是度量空间和拓扑群的推广）的邻域组成

+   `μ.ae`，由相对于测度 `μ` 补集测度为零的集合组成

一般定义如下：过滤器 `F : Filter X` 是一个集合 `F.sets : Set (Set X)`，满足以下条件：

+   `F.univ_sets`：`univ` 属于 `F.sets`

+   `F.sets_of_superset`：对所有 `{U V}`，如果 `U ∈ F.sets` 且 `U ⊆ V`，则 `V ∈ F.sets`

+   `F.inter_sets`：对所有 `{U V}`，如果 `U ∈ F.sets` 且 `V ∈ F.sets`，则 `U ∩ V ∈ F.sets`。

第一个条件说明 `X` 的所有元素集合属于 `F.sets`。第二个条件说明如果 `U` 属于 `F.sets`，那么包含 `U` 的任何集合也属于 `F.sets`。第三个条件说明 `F.sets` 对有限交集是封闭的。在 Mathlib 中，过滤器 `F` 被定义为捆绑 `F.sets` 和其三个属性的结构的定义，但这些属性不携带额外的数据，并且将 `F` 和 `F.sets` 之间的区别模糊化是方便的。因此，我们定义 `U ∈ F` 表示 `U ∈ F.sets`。这解释了为什么在提及 `U ∈ F` 的某些引理名称中出现了单词 `sets`。

可以将过滤器视为定义“足够大”的集合的概念。第一个条件说明 `univ` 是足够大的，第二个条件说明包含足够大集合的集合也是足够大的，第三个条件说明两个足够大集合的交集也是足够大的。

将类型 `X` 上的过滤器视为 `Set X` 的广义元素可能更有用。例如，`atTop` 是“非常大的数”的集合，而 `𝓝 x₀` 是“非常接近 `x₀` 的点”的集合。这种观点的一个表现是，我们可以将任何 `s : Set X` 与所谓的 *主过滤器* 关联起来，该过滤器由包含 `s` 的所有集合组成。这个定义已经在 Mathlib 中，并且有 `𝓟`（在 `Filter` 命名空间中局部化）的符号。为了演示目的，我们要求你利用这个机会在这里推导出定义。

```py
def  principal  {α  :  Type*}  (s  :  Set  α)  :  Filter  α
  where
  sets  :=  {  t  |  s  ⊆  t  }
  univ_sets  :=  sorry
  sets_of_superset  :=  sorry
  inter_sets  :=  sorry 
```

对于我们的第二个例子，我们要求你定义过滤器 `atTop : Filter ℕ`。（我们可以用任何具有偏序的类型代替 `ℕ`。）

```py
example  :  Filter  ℕ  :=
  {  sets  :=  {  s  |  ∃  a,  ∀  b,  a  ≤  b  →  b  ∈  s  }
  univ_sets  :=  sorry
  sets_of_superset  :=  sorry
  inter_sets  :=  sorry  } 
```

我们还可以直接定义任何 `x : ℝ` 的邻域过滤器 `𝓝 x`。在实数中，`x` 的邻域是一个包含开区间 $(x_0 - \varepsilon, x_0 + \varepsilon)$ 的集合，在 Mathlib 中定义为 `Ioo (x₀ - ε) (x₀ + ε)`。（这种邻域的概念只是 Mathlib 中更一般构造的一个特例。）

通过这些例子，我们已可以定义函数 `f : X → Y` 沿着某个 `F : Filter X` 收敛到某个 `G : Filter Y` 的含义，如下所示：

```py
def  Tendsto₁  {X  Y  :  Type*}  (f  :  X  →  Y)  (F  :  Filter  X)  (G  :  Filter  Y)  :=
  ∀  V  ∈  G,  f  ⁻¹'  V  ∈  F 
```

当 `X` 是 `ℕ` 且 `Y` 是 `ℝ` 时，`Tendsto₁ u atTop (𝓝 x)` 等价于说序列 `u : ℕ → ℝ` 收敛到实数 `x`。当 `X` 和 `Y` 都是 `ℝ` 时，`Tendsto f (𝓝 x₀) (𝓝 y₀)` 等价于熟悉的极限概念 $\lim_{x \to x₀} f(x) = y₀$。介绍中提到的所有其他类型的极限也都是通过在源和目标上选择合适的过滤器，等价于 `Tendsto₁` 的实例。

上述 `Tendsto₁` 的概念在定义上是等价于 Mathlib 中定义的 `Tendsto` 概念，但后者定义得更抽象。`Tendsto₁` 的定义问题在于它暴露了一个量词和 `G` 的元素，并隐藏了通过将过滤器视为广义集合而获得的直觉。我们可以通过使用更多的代数和集合论工具来隐藏量词 `∀ V` 并使直觉更加明显。第一个成分是与任何映射 `f : X → Y` 关联的 *推前* 操作 $f_*$，在 Mathlib 中表示为 `Filter.map f`。给定 `X` 上的过滤器 `F`，`Filter.map f F : Filter Y` 被定义为 `V ∈ Filter.map f F ↔ f ⁻¹' V ∈ F` 定义上成立。在打开的示例文件中，我们已经打开了 `Filter` 命名空间，以便可以将 `Filter.map` 写作 `map`。这意味着我们可以使用 `Filter Y` 上的顺序关系重写 `Tendsto` 的定义，这是成员集合的反向包含。换句话说，给定 `G H : Filter Y`，我们有 `G ≤ H ↔ ∀ V : Set Y, V ∈ H → V ∈ G`。

```py
def  Tendsto₂  {X  Y  :  Type*}  (f  :  X  →  Y)  (F  :  Filter  X)  (G  :  Filter  Y)  :=
  map  f  F  ≤  G

example  {X  Y  :  Type*}  (f  :  X  →  Y)  (F  :  Filter  X)  (G  :  Filter  Y)  :
  Tendsto₂  f  F  G  ↔  Tendsto₁  f  F  G  :=
  Iff.rfl 
```

可能看起来过滤器上的顺序关系是反的。但回想一下，我们可以通过 `𝓟 : Set X → Filter X` 的包含来将 `X` 上的过滤器视为 `Set X` 的广义元素，该映射将任何集合 `s` 映射到相应的素过滤器。这个包含是顺序保持的，所以 `Filter` 上的顺序关系确实可以看作是广义集合之间的自然包含关系。在这个类比中，推前像是直接像。实际上，`map f (𝓟 s) = 𝓟 (f '' s)`。

现在，我们可以直观地理解为什么序列 `u : ℕ → ℝ` 收敛到点 `x₀` 当且仅当 `map u atTop ≤ 𝓝 x₀`。不等式意味着“在 `u` 下的直接像”的“非常大的自然数集合”包含在“非常接近 `x₀` 的点集合”中。

如承诺的那样，`Tendsto₂`的定义没有出现任何量词或集合。它还利用了推进操作的代数性质。首先，每个`Filter.map f`都是单调的。其次，`Filter.map`与复合操作兼容。

```py
#check  (@Filter.map_mono  :  ∀  {α  β}  {m  :  α  →  β},  Monotone  (map  m))

#check
  (@Filter.map_map  :
  ∀  {α  β  γ}  {f  :  Filter  α}  {m  :  α  →  β}  {m'  :  β  →  γ},  map  m'  (map  m  f)  =  map  (m'  ∘  m)  f) 
```

这两个性质共同允许我们证明极限可以组合，一次产生介绍中描述的 512 个组合引理变体，以及更多。你可以使用`Tendsto₁`的关于全称量词的定义或代数定义，以及上述两个引理来证明以下陈述。

```py
example  {X  Y  Z  :  Type*}  {F  :  Filter  X}  {G  :  Filter  Y}  {H  :  Filter  Z}  {f  :  X  →  Y}  {g  :  Y  →  Z}
  (hf  :  Tendsto₁  f  F  G)  (hg  :  Tendsto₁  g  G  H)  :  Tendsto₁  (g  ∘  f)  F  H  :=
  sorry 
```

推进构造使用一个映射将过滤器从映射源推进到映射目标。还有一个相反方向的*拉回*操作，`Filter.comap`。这推广了集合上的前像操作。对于任何映射`f`，`Filter.map f`和`Filter.comap f`形成了一个称为*伽罗瓦连接*的结构，也就是说，它们满足

> `Filter.map_le_iff_le_comap : Filter.map f F ≤ G ↔ F ≤ Filter.comap f G`

对于每个`F`和`G`。这个操作可以用来提供`Tendsto`的另一种表述，这将证明上是（但不是定义上）等价于 Mathlib 中的表述。

`comap`操作可以用来将过滤器限制为子类型。例如，假设我们有`f : ℝ → ℝ`，`x₀ : ℝ`和`y₀ : ℝ`，并且假设我们想要说明当`x`在有理数内接近`x₀`时，`f x`接近`y₀`。我们可以使用强制映射`(↑) : ℚ → ℝ`将过滤器`𝓝 x₀`拉回到`ℚ`，并声明`Tendsto (f ∘ (↑) : ℚ → ℝ) (comap (↑) (𝓝 x₀)) (𝓝 y₀)`。

```py
variable  (f  :  ℝ  →  ℝ)  (x₀  y₀  :  ℝ)

#check  comap  ((↑)  :  ℚ  →  ℝ)  (𝓝  x₀)

#check  Tendsto  (f  ∘  (↑))  (comap  ((↑)  :  ℚ  →  ℝ)  (𝓝  x₀))  (𝓝  y₀) 
```

拉回操作也与复合操作兼容，但它具有*逆变异性*，也就是说，它反转了参数的顺序。

```py
section
variable  {α  β  γ  :  Type*}  (F  :  Filter  α)  {m  :  γ  →  β}  {n  :  β  →  α}

#check  (comap_comap  :  comap  m  (comap  n  F)  =  comap  (n  ∘  m)  F)

end 
```

现在，让我们将注意力转向平面`ℝ × ℝ`，并尝试理解点`(x₀, y₀)`的邻域如何与`𝓝 x₀`和`𝓝 y₀`相关。存在一个乘法操作`Filter.prod : Filter X → Filter Y → Filter (X × Y)`，表示为`×ˢ`，它回答了这个问题：

```py
example  :  𝓝  (x₀,  y₀)  =  𝓝  x₀  ×ˢ  𝓝  y₀  :=
  nhds_prod_eq 
```

乘法操作是在拉回操作和`inf`操作的基础上定义的：

> `F ×ˢ G = (comap Prod.fst F) ⊓ (comap Prod.snd G)`。

这里，`inf`操作指的是任何类型`X`上`Filter X`的格结构，其中`F ⊓ G`是小于`F`和`G`两者的最大过滤器。因此，`inf`操作推广了集合交集的概念。

Mathlib 中的许多证明都使用了上述所有结构（`map`、`comap`、`inf`、`sup`和`prod`）来给出关于收敛的代数证明，而从未引用过滤器的成员。你可以在以下引理的证明中练习这样做，如果需要，可以展开`Tendsto`和`Filter.prod`的定义。

```py
#check  le_inf_iff

example  (f  :  ℕ  →  ℝ  ×  ℝ)  (x₀  y₀  :  ℝ)  :
  Tendsto  f  atTop  (𝓝  (x₀,  y₀))  ↔
  Tendsto  (Prod.fst  ∘  f)  atTop  (𝓝  x₀)  ∧  Tendsto  (Prod.snd  ∘  f)  atTop  (𝓝  y₀)  :=
  sorry 
```

有序类型`Filter X`实际上是一个*完备*格，也就是说，存在一个底元素，存在一个顶元素，并且`X`上的每个过滤器集合都有一个`Inf`和`Sup`。

注意，根据滤波器定义中的第二个性质（如果`U`属于`F`，那么任何大于`U`的集合也属于`F`），第一个性质（所有`X`的居民集合属于`F`）等价于`F`不是空集合集合的性质。这不应该与更微妙的问题混淆，即空集是否是`F`的**元素**。滤波器的定义并不禁止`∅ ∈ F`，但如果空集在`F`中，那么每一个集合都在`F`中，也就是说，`∀ U : Set X, U ∈ F`。在这种情况下，`F`是一个相当平凡的滤波器，这正是完备格`Filter X`的底元素。这与 Bourbaki 中滤波器的定义形成对比，后者不允许包含空集的滤波器。

由于我们在定义中包括了平凡滤波器，我们有时需要在某些引理中明确假设非平凡性。然而，作为回报，理论具有更漂亮的全局性质。我们已经看到，包括平凡滤波器给我们提供了一个底元素。它还允许我们定义`principal : Set X → Filter X`，它将`∅`映射到`⊥`，而不需要添加一个先决条件来排除空集。它还允许我们定义没有先决条件的拉回操作。实际上，可能会发生`comap f F = ⊥`尽管`F ≠ ⊥`。例如，给定`x₀ : ℝ`和`s : Set ℝ`，从对应于`s`的子类型强制转换下的`𝓝 x₀`的拉回是非平凡的，当且仅当`x₀`属于`s`的闭包。

为了管理需要假设某些滤波器是非平凡的引理，Mathlib 有一个类型类`Filter.NeBot`，库中有假设`(F : Filter X) [F.NeBot]`的引理。实例数据库知道，例如，`(atTop : Filter ℕ).NeBot`，它知道向前传递一个非平凡滤波器会得到一个非平凡滤波器。因此，假设`[F.NeBot]`的引理将自动适用于任何序列`u`的`map u atTop`。

我们对滤波器的代数性质及其与极限的关系的考察基本上已经完成，但我们还没有证明我们重新获得了通常的极限概念。表面上，`Tendsto u atTop (𝓝 x₀)`似乎比第 3.6 节中定义的收敛性概念更强，因为我们要求`x₀`的每一个邻域都有一个属于`atTop`的逆像，而通常的定义只要求对于标准邻域`Ioo (x₀ - ε) (x₀ + ε)`是这样的。关键是，根据定义，每一个邻域都包含这样的标准邻域。这个观察导致了一个**滤波基**的概念。

给定 `F : Filter X`，一个集合族 `s : ι → Set X` 是 `F` 的基当且仅当对于每一个集合 `U`，如果且仅当它包含某些 `s i`，则 `U ∈ F`。换句话说，从形式上讲，如果 `s` 满足 `∀ U : Set X, U ∈ F ↔ ∃ i, s i ⊆ U`，则 `s` 是一个基。考虑在 `ι` 上的一个谓词，它只选择索引类型中的某些值 `i`，这甚至更加灵活。在 `𝓝 x₀` 的情况下，我们希望 `ι` 是 `ℝ`，我们用 `ε` 表示 `i`，谓词应该选择 `ε` 的正值。因此，集合 `Ioo  (x₀ - ε) (x₀ + ε)` 形成实数邻域拓扑的基的事实表述如下：

```py
example  (x₀  :  ℝ)  :  HasBasis  (𝓝  x₀)  (fun  ε  :  ℝ  ↦  0  <  ε)  fun  ε  ↦  Ioo  (x₀  -  ε)  (x₀  +  ε)  :=
  nhds_basis_Ioo_pos  x₀ 
```

对于 `atTop` 过滤器也有一个很好的基。引理 `Filter.HasBasis.tendsto_iff` 允许我们用 `F` 和 `G` 的基重新表述形式为 `Tendsto f F G` 的陈述。将这些部分放在一起，我们基本上得到了我们在 第 3.6 节 中使用的收敛的概念。

```py
example  (u  :  ℕ  →  ℝ)  (x₀  :  ℝ)  :
  Tendsto  u  atTop  (𝓝  x₀)  ↔  ∀  ε  >  0,  ∃  N,  ∀  n  ≥  N,  u  n  ∈  Ioo  (x₀  -  ε)  (x₀  +  ε)  :=  by
  have  :  atTop.HasBasis  (fun  _  :  ℕ  ↦  True)  Ici  :=  atTop_basis
  rw  [this.tendsto_iff  (nhds_basis_Ioo_pos  x₀)]
  simp 
```

现在我们将展示过滤器如何帮助处理对于足够大的数字或对于足够接近给定点的点所持有的属性。在 第 3.6 节 中，我们经常面临这样的情况：我们知道某些属性 `P n` 对于足够大的 `n` 成立，而其他属性 `Q n` 对于足够大的 `n` 成立。使用 `cases` 两次给出了满足 `∀ n ≥ N_P, P n` 和 `∀ n ≥ N_Q, Q n` 的 `N_P` 和 `N_Q`。使用 `set N := max N_P N_Q`，我们最终可以证明 `∀ n ≥ N, P n ∧ Q n`。重复这样做会变得令人厌烦。

通过注意到陈述“`P n` 和 `Q n` 对于足够大的 `n` 成立”意味着我们有一个 `{n | P n} ∈ atTop` 和 `{n | Q n} ∈ atTop`，我们可以做得更好。`atTop` 是一个过滤器的事实意味着 `atTop` 中两个元素的交集再次在 `atTop` 中，因此我们得到 `{n | P n ∧ Q n} ∈ atTop`。写作 `{n | P n} ∈ atTop` 是不愉快的，但我们可以使用更具说明性的符号 `∀ᶠ n in atTop, P n`。这里上标的 `f` 代表“Filter”。你可以将这个符号理解为对于所有在“非常大的数字集合”中的 `n`，`P n` 成立。`∀ᶠ` 符号代表 `Filter.Eventually`，引理 `Filter.Eventually.and` 使用过滤器的交集属性来完成我们刚才描述的工作：

```py
example  (P  Q  :  ℕ  →  Prop)  (hP  :  ∀ᶠ  n  in  atTop,  P  n)  (hQ  :  ∀ᶠ  n  in  atTop,  Q  n)  :
  ∀ᶠ  n  in  atTop,  P  n  ∧  Q  n  :=
  hP.and  hQ 
```

这种记法既方便又直观，以至于当 `P` 是一个等式或不等式陈述时，我们也有专门的定义。例如，设 `u` 和 `v` 是两个实数序列，我们将证明如果 `u n` 和 `v n` 在足够大的 `n` 上相同，则 `u` 趋于 `x₀` 当且仅当 `v` 趋于 `x₀`。首先我们将使用通用的 `Eventually`，然后是专门针对等式谓词的 `EventuallyEq`。这两个陈述在定义上是等价的，所以两种情况下的证明工作相同。

```py
example  (u  v  :  ℕ  →  ℝ)  (h  :  ∀ᶠ  n  in  atTop,  u  n  =  v  n)  (x₀  :  ℝ)  :
  Tendsto  u  atTop  (𝓝  x₀)  ↔  Tendsto  v  atTop  (𝓝  x₀)  :=
  tendsto_congr'  h

example  (u  v  :  ℕ  →  ℝ)  (h  :  u  =ᶠ[atTop]  v)  (x₀  :  ℝ)  :
  Tendsto  u  atTop  (𝓝  x₀)  ↔  Tendsto  v  atTop  (𝓝  x₀)  :=
  tendsto_congr'  h 
```

回顾 `Eventually` 术语下的过滤器定义是有益的。给定 `F : Filter X`，对于 `X` 上的任何谓词 `P` 和 `Q`，

+   条件`univ ∈ F`确保`(∀ x, P x) → ∀ᶠ x in F, P x`，

+   条件`U ∈ F → U ⊆ V → V ∈ F`确保`(∀ᶠ x in F, P x) → (∀ x, P x → Q x) → ∀ᶠ x in F, Q x`，

+   条件`U ∈ F → V ∈ F → U ∩ V ∈ F`确保`(∀ᶠ x in F, P x) → (∀ᶠ x in F, Q x) → ∀ᶠ x in F, P x ∧ Q x`。

```py
#check  Eventually.of_forall
#check  Eventually.mono
#check  Eventually.and 
```

第二项，对应于`Eventually.mono`，支持使用过滤器的优雅方式，特别是当与`Eventually.and`结合使用时。`filter_upwards`策略允许我们组合它们。比较：

```py
example  (P  Q  R  :  ℕ  →  Prop)  (hP  :  ∀ᶠ  n  in  atTop,  P  n)  (hQ  :  ∀ᶠ  n  in  atTop,  Q  n)
  (hR  :  ∀ᶠ  n  in  atTop,  P  n  ∧  Q  n  →  R  n)  :  ∀ᶠ  n  in  atTop,  R  n  :=  by
  apply  (hP.and  (hQ.and  hR)).mono
  rintro  n  ⟨h,  h',  h''⟩
  exact  h''  ⟨h,  h'⟩

example  (P  Q  R  :  ℕ  →  Prop)  (hP  :  ∀ᶠ  n  in  atTop,  P  n)  (hQ  :  ∀ᶠ  n  in  atTop,  Q  n)
  (hR  :  ∀ᶠ  n  in  atTop,  P  n  ∧  Q  n  →  R  n)  :  ∀ᶠ  n  in  atTop,  R  n  :=  by
  filter_upwards  [hP,  hQ,  hR]  with  n  h  h'  h''
  exact  h''  ⟨h,  h'⟩ 
```

了解测度理论的读者会注意到，集合的补集测度为零（即“几乎每个点的集合”）的过滤器`μ.ae`作为`Tendsto`的源或目标并不十分有用，但它可以方便地与`Eventually`结合使用，以表明某个属性对几乎所有点都成立。

存在一个与`∀ᶠ x in F, P x`相对应的对偶版本，偶尔很有用：`∃ᶠ x in F, P x`意味着`{x | ¬P x} ∉ F`。例如，`∃ᶠ n in atTop, P n`意味着存在任意大的`n`使得`P n`成立。`∃ᶠ`符号代表`Filter.Frequently`。

对于一个更复杂的例子，考虑以下关于序列`u`、集合`M`和值`x`的陈述：

> 如果`u`收敛到`x`且对于足够大的`n`，`u n`属于`M`，那么`x`在`M`的闭集中。

这可以形式化为以下内容：

> `Tendsto u atTop (𝓝 x) → (∀ᶠ n in atTop, u n ∈ M) → x ∈ closure M`。

这是拓扑库中定理`mem_closure_of_tendsto`的一个特例。看看你是否可以用引用的引理来证明它，利用`ClusterPt x F`表示`(𝓝 x ⊓ F).NeBot`，以及根据定义，假设`∀ᶠ n in atTop, u n ∈ M`意味着`M ∈ map u atTop`。

```py
#check  mem_closure_iff_clusterPt
#check  le_principal_iff
#check  neBot_of_le

example  (u  :  ℕ  →  ℝ)  (M  :  Set  ℝ)  (x  :  ℝ)  (hux  :  Tendsto  u  atTop  (𝓝  x))
  (huM  :  ∀ᶠ  n  in  atTop,  u  n  ∈  M)  :  x  ∈  closure  M  :=
  sorry 
```

## 11.2. 度量空间

上一节中的例子主要关注实数序列。在本节中，我们将提高一点普遍性，关注度量空间。度量空间是一个类型`X`，它配备了一个距离函数`dist : X → X → ℝ`，这是从`X = ℝ`的情况下的函数`fun x y ↦ |x - y|`的推广。

引入这样的空间很容易，我们将检查距离函数所需的所有属性。

```py
variable  {X  :  Type*}  [MetricSpace  X]  (a  b  c  :  X)

#check  (dist  a  b  :  ℝ)
#check  (dist_nonneg  :  0  ≤  dist  a  b)
#check  (dist_eq_zero  :  dist  a  b  =  0  ↔  a  =  b)
#check  (dist_comm  a  b  :  dist  a  b  =  dist  b  a)
#check  (dist_triangle  a  b  c  :  dist  a  c  ≤  dist  a  b  +  dist  b  c) 
```

注意我们还有变体，其中距离可以是无限的，或者`dist a b`可以是零，而`a ≠ b`或两者都不成立。它们分别称为`EMetricSpace`、`PseudoMetricSpace`和`PseudoEMetricSpace`（这里的“e”代表“扩展”）。

注意，我们从`ℝ`到度量空间的过程中跳过了需要线性代数的特殊情况的范数空间，这将在微积分章节中解释。

### 11.2.1. 收敛与连续性

使用距离函数，我们可以在度量空间之间定义收敛序列和连续函数。实际上，它们在下一节中更一般的设置中定义，但我们有将定义重新表述为距离的引理。

```py
example  {u  :  ℕ  →  X}  {a  :  X}  :
  Tendsto  u  atTop  (𝓝  a)  ↔  ∀  ε  >  0,  ∃  N,  ∀  n  ≥  N,  dist  (u  n)  a  <  ε  :=
  Metric.tendsto_atTop

example  {X  Y  :  Type*}  [MetricSpace  X]  [MetricSpace  Y]  {f  :  X  →  Y}  :
  Continuous  f  ↔
  ∀  x  :  X,  ∀  ε  >  0,  ∃  δ  >  0,  ∀  x',  dist  x'  x  <  δ  →  dist  (f  x')  (f  x)  <  ε  :=
  Metric.continuous_iff 
```

许多引理都有一些连续性假设，所以我们最终证明了许多连续性结果，并且有一个专门用于这个任务的`连续性`策略。让我们证明一个在下面的练习中需要的连续性陈述。注意，Lean 知道如何将两个度量空间的产品视为度量空间，因此考虑从`X × X`到`ℝ`的连续函数是有意义的。特别是，距离函数（未展开版本）就是这样一种函数。

```py
example  {X  Y  :  Type*}  [MetricSpace  X]  [MetricSpace  Y]  {f  :  X  →  Y}  (hf  :  Continuous  f)  :
  Continuous  fun  p  :  X  ×  X  ↦  dist  (f  p.1)  (f  p.2)  :=  by  continuity 
```

这个策略有点慢，所以了解如何手动完成它也很有用。我们首先需要使用`fun p : X × X ↦ f p.1`是连续的，因为它是由`f`（根据假设`hf`是连续的）和投影`prod.fst`（其连续性是引理`continuous_fst`的内容）组成的复合。复合属性是`Continuous.comp`，它在`Continuous`命名空间中，因此我们可以使用点符号将`Continuous.comp hf continuous_fst`压缩为`hf.comp continuous_fst`，这实际上更易于阅读，因为它真正地读作组合我们的假设和我们的引理。我们可以对第二个分量做同样的处理，以获得`fun p : X × X ↦ f p.2`的连续性。然后我们使用`Continuous.prod_mk`将这些两个连续性组装起来，得到`(hf.comp continuous_fst).prod_mk (hf.comp continuous_snd) : Continuous (fun p : X × X ↦ (f p.1, f p.2))`，并再次组合以得到完整的证明。

```py
example  {X  Y  :  Type*}  [MetricSpace  X]  [MetricSpace  Y]  {f  :  X  →  Y}  (hf  :  Continuous  f)  :
  Continuous  fun  p  :  X  ×  X  ↦  dist  (f  p.1)  (f  p.2)  :=
  continuous_dist.comp  ((hf.comp  continuous_fst).prodMk  (hf.comp  continuous_snd)) 
```

通过`Continuous.prod_mk`和`continuous_dist`通过`Continuous.comp`的组合感觉有点笨拙，即使像上面那样大量使用点符号也是如此。一个更严重的问题是，这个漂亮的证明需要大量的规划。Lean 接受上述证明项，因为它是一个完整的项，它证明了与我们的目标定义等价的一个陈述，关键的定义是函数的复合。确实，我们的目标函数`fun p : X × X ↦ dist (f p.1) (f p.2)`并没有以复合的形式呈现。我们提供的证明项证明了`dist ∘ (fun p : X × X ↦ (f p.1, f p.2))`的连续性，这恰好与我们的目标函数定义等价。但如果我们尝试从`apply continuous_dist.comp`开始的策略逐步构建这个证明，那么 Lean 的展开器将无法识别复合，并拒绝应用这个引理。当涉及到类型的产品时，这个问题尤其严重。

在这里应用更好的引理是`Continuous.dist {f g : X → Y} : Continuous f → Continuous g → Continuous (fun x ↦ dist (f x) (g x))`，这对于 Lean 的展开器来说更易于处理，并且在直接提供完整的证明项时也提供了更短的证明，如下面的两个新证明所示：

```py
example  {X  Y  :  Type*}  [MetricSpace  X]  [MetricSpace  Y]  {f  :  X  →  Y}  (hf  :  Continuous  f)  :
  Continuous  fun  p  :  X  ×  X  ↦  dist  (f  p.1)  (f  p.2)  :=  by
  apply  Continuous.dist
  exact  hf.comp  continuous_fst
  exact  hf.comp  continuous_snd

example  {X  Y  :  Type*}  [MetricSpace  X]  [MetricSpace  Y]  {f  :  X  →  Y}  (hf  :  Continuous  f)  :
  Continuous  fun  p  :  X  ×  X  ↦  dist  (f  p.1)  (f  p.2)  :=
  (hf.comp  continuous_fst).dist  (hf.comp  continuous_snd) 
```

注意，如果没有由于复合引起的展开问题，另一种压缩我们证明的方法是使用`Continuous.prod_map`，这在某些情况下很有用，并给出了一个替代的证明项`continuous_dist.comp (hf.prod_map hf)`，这甚至更短于输入。

由于在详细阐述和简短输入之间做出选择都很令人难过，让我们用`Continuous.fst'`提供的最后一点压缩来结束这次讨论，它允许将`hf.comp continuous_fst`压缩为`hf.fst'`（以及`snd`），并得到我们的最终证明，现在正接近晦涩。

```py
example  {X  Y  :  Type*}  [MetricSpace  X]  [MetricSpace  Y]  {f  :  X  →  Y}  (hf  :  Continuous  f)  :
  Continuous  fun  p  :  X  ×  X  ↦  dist  (f  p.1)  (f  p.2)  :=
  hf.fst'.dist  hf.snd' 
```

现在轮到你了，去证明一些连续性引理。在尝试连续性策略之后，你需要`Continuous.add`、`continuous_pow`和`continuous_id`来手动完成。

```py
example  {f  :  ℝ  →  X}  (hf  :  Continuous  f)  :  Continuous  fun  x  :  ℝ  ↦  f  (x  ^  2  +  x)  :=
  sorry 
```

到目前为止，我们将连续性视为一个全局概念，但也可以定义在一点上的连续性。

```py
example  {X  Y  :  Type*}  [MetricSpace  X]  [MetricSpace  Y]  (f  :  X  →  Y)  (a  :  X)  :
  ContinuousAt  f  a  ↔  ∀  ε  >  0,  ∃  δ  >  0,  ∀  {x},  dist  x  a  <  δ  →  dist  (f  x)  (f  a)  <  ε  :=
  Metric.continuousAt_iff 
```

### 11.2.2\. 球、开集和闭集

一旦我们有一个距离函数，最重要的几何定义就是（开）球和闭球。

```py
variable  (r  :  ℝ)

example  :  Metric.ball  a  r  =  {  b  |  dist  b  a  <  r  }  :=
  rfl

example  :  Metric.closedBall  a  r  =  {  b  |  dist  b  a  ≤  r  }  :=
  rfl 
```

注意，这里的 r 是任何实数，没有符号限制。当然，有些陈述确实需要半径条件。

```py
example  (hr  :  0  <  r)  :  a  ∈  Metric.ball  a  r  :=
  Metric.mem_ball_self  hr

example  (hr  :  0  ≤  r)  :  a  ∈  Metric.closedBall  a  r  :=
  Metric.mem_closedBall_self  hr 
```

一旦我们有球，我们就可以定义开集。实际上，它们是在下一节中讨论的更一般设置中定义的，但我们有引理将定义重新表述为球的形式。

```py
example  (s  :  Set  X)  :  IsOpen  s  ↔  ∀  x  ∈  s,  ∃  ε  >  0,  Metric.ball  x  ε  ⊆  s  :=
  Metric.isOpen_iff 
```

然后闭集是补集为开的集合。它们的重要性质是它们在极限下是封闭的。集合的闭包是包含它的最小闭集。

```py
example  {s  :  Set  X}  :  IsClosed  s  ↔  IsOpen  (sᶜ)  :=
  isOpen_compl_iff.symm

example  {s  :  Set  X}  (hs  :  IsClosed  s)  {u  :  ℕ  →  X}  (hu  :  Tendsto  u  atTop  (𝓝  a))
  (hus  :  ∀  n,  u  n  ∈  s)  :  a  ∈  s  :=
  hs.mem_of_tendsto  hu  (Eventually.of_forall  hus)

example  {s  :  Set  X}  :  a  ∈  closure  s  ↔  ∀  ε  >  0,  ∃  b  ∈  s,  a  ∈  Metric.ball  b  ε  :=
  Metric.mem_closure_iff 
```

在不使用`mem_closure_iff_seq_limit`的情况下完成下一个练习。

```py
example  {u  :  ℕ  →  X}  (hu  :  Tendsto  u  atTop  (𝓝  a))  {s  :  Set  X}  (hs  :  ∀  n,  u  n  ∈  s)  :
  a  ∈  closure  s  :=  by
  sorry 
```

记住从过滤器部分，邻域过滤器在 Mathlib 中起着重要作用。在度量空间上下文中，关键点是球为这些过滤器提供基。这里的主要引理是`Metric.nhds_basis_ball`和`Metric.nhds_basis_closedBall`，它们为正半径的开球和闭球断言这一点。中心点是隐含的参数，因此我们可以像以下示例中那样调用`Filter.HasBasis.mem_iff`。

```py
example  {x  :  X}  {s  :  Set  X}  :  s  ∈  𝓝  x  ↔  ∃  ε  >  0,  Metric.ball  x  ε  ⊆  s  :=
  Metric.nhds_basis_ball.mem_iff

example  {x  :  X}  {s  :  Set  X}  :  s  ∈  𝓝  x  ↔  ∃  ε  >  0,  Metric.closedBall  x  ε  ⊆  s  :=
  Metric.nhds_basis_closedBall.mem_iff 
```

### 11.2.3\. 紧性

紧性是一个重要的拓扑概念。它区分了度量空间的子集，这些子集与其他区间相比，与实数中的线段具有相同的性质：

+   在紧集上取值的任何序列都有一个子序列在这个集合中收敛。

+   在非空紧集上取实数值的任何连续函数都是有界的，并且在其某处达到其界限（这被称为极值定理）。

+   紧集是闭集。

让我们先检查实数单位区间确实是一个紧集，然后检查一般度量空间中紧集的上述命题。在第二个命题中，我们只需要给定集合上的连续性，因此我们将使用`ContinuousOn`而不是`Continuous`，并且我们将分别给出最小值和最大值的单独陈述。当然，所有这些结果都是从更一般的版本中推导出来的，其中一些将在后面的章节中讨论。

```py
example  :  IsCompact  (Set.Icc  0  1  :  Set  ℝ)  :=
  isCompact_Icc

example  {s  :  Set  X}  (hs  :  IsCompact  s)  {u  :  ℕ  →  X}  (hu  :  ∀  n,  u  n  ∈  s)  :
  ∃  a  ∈  s,  ∃  φ  :  ℕ  →  ℕ,  StrictMono  φ  ∧  Tendsto  (u  ∘  φ)  atTop  (𝓝  a)  :=
  hs.tendsto_subseq  hu

example  {s  :  Set  X}  (hs  :  IsCompact  s)  (hs'  :  s.Nonempty)  {f  :  X  →  ℝ}
  (hfs  :  ContinuousOn  f  s)  :
  ∃  x  ∈  s,  ∀  y  ∈  s,  f  x  ≤  f  y  :=
  hs.exists_isMinOn  hs'  hfs

example  {s  :  Set  X}  (hs  :  IsCompact  s)  (hs'  :  s.Nonempty)  {f  :  X  →  ℝ}
  (hfs  :  ContinuousOn  f  s)  :
  ∃  x  ∈  s,  ∀  y  ∈  s,  f  y  ≤  f  x  :=
  hs.exists_isMaxOn  hs'  hfs

example  {s  :  Set  X}  (hs  :  IsCompact  s)  :  IsClosed  s  :=
  hs.isClosed 
```

我们还可以指定一个度量空间是全局紧的，使用一个额外的`Prop`值类型类：

```py
example  {X  :  Type*}  [MetricSpace  X]  [CompactSpace  X]  :  IsCompact  (univ  :  Set  X)  :=
  isCompact_univ 
```

在紧度量空间中，任何闭集都是紧集，这是`IsClosed.isCompact`。

### 11.2.4\. 均匀连续函数

现在，我们将转向度量空间上的均匀性概念：均匀连续函数、柯西序列和完备性。同样，这些在更一般的环境中定义，但我们有在度量名称空间中的引理来访问它们的元素定义。我们首先从均匀连续性开始。

```py
example  {X  :  Type*}  [MetricSpace  X]  {Y  :  Type*}  [MetricSpace  Y]  {f  :  X  →  Y}  :
  UniformContinuous  f  ↔
  ∀  ε  >  0,  ∃  δ  >  0,  ∀  {a  b  :  X},  dist  a  b  <  δ  →  dist  (f  a)  (f  b)  <  ε  :=
  Metric.uniformContinuous_iff 
```

为了练习操作所有这些定义，我们将证明从紧致度量空间到度量空间的连续函数是均匀连续的（我们将在后面的章节中看到更一般的形式）。

我们将首先给出一个非正式的草图。设 `f : X → Y` 是从紧致度量空间到度量空间的连续函数。我们固定 `ε > 0` 并开始寻找某个 `δ`。

设 `φ : X × X → ℝ := fun p ↦ dist (f p.1) (f p.2)` 和 `K := { p : X × X | ε ≤ φ p }`。观察 `φ` 是连续的，因为 `f` 和距离都是连续的。而且 `K` 显然是闭集（使用 `isClosed_le`），因此由于 `X` 是紧致的，`K` 也是紧致的。

然后我们讨论使用 `eq_empty_or_nonempty` 的两种可能性。如果 `K` 是空的，那么显然我们已经完成了（例如，我们可以将 `δ` 设置为 `1`）。所以让我们假设 `K` 不是空的，并使用极值定理来选择 `(x₀, x₁)`，它在 `K` 上的距离函数上达到下确界。然后我们可以将 `δ` 设置为 `dist x₀ x₁` 并检查一切是否正常工作。

```py
example  {X  :  Type*}  [MetricSpace  X]  [CompactSpace  X]
  {Y  :  Type*}  [MetricSpace  Y]  {f  :  X  →  Y}
  (hf  :  Continuous  f)  :  UniformContinuous  f  :=  by
  sorry 
```

### 11.2.5\. 完备性

在度量空间中的柯西序列是一个项与项之间的距离越来越近的序列。有几种等价的方式来表述这个想法。特别是收敛序列是柯西序列。逆命题只在所谓的 *完备* 空间中成立。

```py
example  (u  :  ℕ  →  X)  :
  CauchySeq  u  ↔  ∀  ε  >  0,  ∃  N  :  ℕ,  ∀  m  ≥  N,  ∀  n  ≥  N,  dist  (u  m)  (u  n)  <  ε  :=
  Metric.cauchySeq_iff

example  (u  :  ℕ  →  X)  :
  CauchySeq  u  ↔  ∀  ε  >  0,  ∃  N  :  ℕ,  ∀  n  ≥  N,  dist  (u  n)  (u  N)  <  ε  :=
  Metric.cauchySeq_iff'

example  [CompleteSpace  X]  (u  :  ℕ  →  X)  (hu  :  CauchySeq  u)  :
  ∃  x,  Tendsto  u  atTop  (𝓝  x)  :=
  cauchySeq_tendsto_of_complete  hu 
```

我们将通过证明一个方便的判据来练习使用这个定义，这是 Mathlib 中出现的一个判据的特殊情况。这也是练习在几何环境中使用大和的好机会。除了过滤器部分的解释外，你可能还需要 `tendsto_pow_atTop_nhds_zero_of_lt_one`、`Tendsto.mul` 和 `dist_le_range_sum_dist`。

```py
theorem  cauchySeq_of_le_geometric_two'  {u  :  ℕ  →  X}
  (hu  :  ∀  n  :  ℕ,  dist  (u  n)  (u  (n  +  1))  ≤  (1  /  2)  ^  n)  :  CauchySeq  u  :=  by
  rw  [Metric.cauchySeq_iff']
  intro  ε  ε_pos
  obtain  ⟨N,  hN⟩  :  ∃  N  :  ℕ,  1  /  2  ^  N  *  2  <  ε  :=  by  sorry
  use  N
  intro  n  hn
  obtain  ⟨k,  rfl  :  n  =  N  +  k⟩  :=  le_iff_exists_add.mp  hn
  calc
  dist  (u  (N  +  k))  (u  N)  =  dist  (u  (N  +  0))  (u  (N  +  k))  :=  sorry
  _  ≤  ∑  i  ∈  range  k,  dist  (u  (N  +  i))  (u  (N  +  (i  +  1)))  :=  sorry
  _  ≤  ∑  i  ∈  range  k,  (1  /  2  :  ℝ)  ^  (N  +  i)  :=  sorry
  _  =  1  /  2  ^  N  *  ∑  i  ∈  range  k,  (1  /  2  :  ℝ)  ^  i  :=  sorry
  _  ≤  1  /  2  ^  N  *  2  :=  sorry
  _  <  ε  :=  sorry 
```

我们已经准备好本节的最终挑战：完备度量空间的 Baire 定理！下面的证明框架展示了有趣的技术。它使用了感叹号变体的 `choose` 策略（你应该尝试移除这个感叹号），并展示了如何在证明中使用 `Nat.rec_on` 递归地定义某物。

```py
open  Metric

example  [CompleteSpace  X]  (f  :  ℕ  →  Set  X)  (ho  :  ∀  n,  IsOpen  (f  n))  (hd  :  ∀  n,  Dense  (f  n))  :
  Dense  (⋂  n,  f  n)  :=  by
  let  B  :  ℕ  →  ℝ  :=  fun  n  ↦  (1  /  2)  ^  n
  have  Bpos  :  ∀  n,  0  <  B  n
  sorry
  /- Translate the density assumption into two functions `center` and `radius` associating
 to any n, x, δ, δpos a center and a positive radius such that
 `closedBall center radius` is included both in `f n` and in `closedBall x δ`.
 We can also require `radius ≤ (1/2)^(n+1)`, to ensure we get a Cauchy sequence later. -/
  have  :
  ∀  (n  :  ℕ)  (x  :  X),
  ∀  δ  >  0,  ∃  y  :  X,  ∃  r  >  0,  r  ≤  B  (n  +  1)  ∧  closedBall  y  r  ⊆  closedBall  x  δ  ∩  f  n  :=
  by  sorry
  choose!  center  radius  Hpos  HB  Hball  using  this
  intro  x
  rw  [mem_closure_iff_nhds_basis  nhds_basis_closedBall]
  intro  ε  εpos
  /- `ε` is positive. We have to find a point in the ball of radius `ε` around `x`
 belonging to all `f n`. For this, we construct inductively a sequence
 `F n = (c n, r n)` such that the closed ball `closedBall (c n) (r n)` is included
 in the previous ball and in `f n`, and such that `r n` is small enough to ensure
 that `c n` is a Cauchy sequence. Then `c n` converges to a limit which belongs
 to all the `f n`. -/
  let  F  :  ℕ  →  X  ×  ℝ  :=  fun  n  ↦
  Nat.recOn  n  (Prod.mk  x  (min  ε  (B  0)))
  fun  n  p  ↦  Prod.mk  (center  n  p.1  p.2)  (radius  n  p.1  p.2)
  let  c  :  ℕ  →  X  :=  fun  n  ↦  (F  n).1
  let  r  :  ℕ  →  ℝ  :=  fun  n  ↦  (F  n).2
  have  rpos  :  ∀  n,  0  <  r  n  :=  by  sorry
  have  rB  :  ∀  n,  r  n  ≤  B  n  :=  by  sorry
  have  incl  :  ∀  n,  closedBall  (c  (n  +  1))  (r  (n  +  1))  ⊆  closedBall  (c  n)  (r  n)  ∩  f  n  :=  by
  sorry
  have  cdist  :  ∀  n,  dist  (c  n)  (c  (n  +  1))  ≤  B  n  :=  by  sorry
  have  :  CauchySeq  c  :=  cauchySeq_of_le_geometric_two'  cdist
  -- as the sequence `c n` is Cauchy in a complete space, it converges to a limit `y`.
  rcases  cauchySeq_tendsto_of_complete  this  with  ⟨y,  ylim⟩
  -- this point `y` will be the desired point. We will check that it belongs to all
  -- `f n` and to `ball x ε`.
  use  y
  have  I  :  ∀  n,  ∀  m  ≥  n,  closedBall  (c  m)  (r  m)  ⊆  closedBall  (c  n)  (r  n)  :=  by  sorry
  have  yball  :  ∀  n,  y  ∈  closedBall  (c  n)  (r  n)  :=  by  sorry
  sorry 
```

### 11.2.1\. 收敛与连续

使用距离函数，我们已经在度量空间之间定义了收敛序列和连续函数。实际上，它们在下一节中讨论的更一般的环境中定义，但我们有引理将定义重新表述为距离的形式。

```py
example  {u  :  ℕ  →  X}  {a  :  X}  :
  Tendsto  u  atTop  (𝓝  a)  ↔  ∀  ε  >  0,  ∃  N,  ∀  n  ≥  N,  dist  (u  n)  a  <  ε  :=
  Metric.tendsto_atTop

example  {X  Y  :  Type*}  [MetricSpace  X]  [MetricSpace  Y]  {f  :  X  →  Y}  :
  Continuous  f  ↔
  ∀  x  :  X,  ∀  ε  >  0,  ∃  δ  >  0,  ∀  x',  dist  x'  x  <  δ  →  dist  (f  x')  (f  x)  <  ε  :=
  Metric.continuous_iff 
```

许多引理都有一些连续性的假设，所以我们最终证明了很多连续性结果，并且有一个 `continuity` 策略专门用于这个任务。让我们证明一个在下面的练习中需要的连续性陈述。注意，Lean 知道如何将两个度量空间的产品视为度量空间，因此考虑从 `X × X` 到 `ℝ` 的连续函数是有意义的。特别是，距离函数（未展开版本）是这样的函数。

```py
example  {X  Y  :  Type*}  [MetricSpace  X]  [MetricSpace  Y]  {f  :  X  →  Y}  (hf  :  Continuous  f)  :
  Continuous  fun  p  :  X  ×  X  ↦  dist  (f  p.1)  (f  p.2)  :=  by  continuity 
```

这个策略有点慢，因此了解如何手动完成它也是有用的。我们首先需要使用 `fun p : X × X ↦ f p.1` 是连续的，因为它是由连续的 `f`（由假设 `hf` 确定）和投影 `prod.fst`（其连续性是引理 `continuous_fst` 的内容）组成的复合。复合属性是 `Continuous.comp`，它在 `Continuous` 命名空间中，因此我们可以使用点符号将 `Continuous.comp hf continuous_fst` 压缩为 `hf.comp continuous_fst`，这实际上更易于阅读，因为它真正地读作将我们的假设和引理组合在一起。我们可以对第二个组件做同样的处理，以获得 `fun p : X × X ↦ f p.2` 的连续性。然后我们使用 `Continuous.prod_mk` 将这两个连续性组装起来，得到 `(hf.comp continuous_fst).prod_mk (hf.comp continuous_snd) : Continuous (fun p : X × X ↦ (f p.1, f p.2))`，然后再进行一次复合，以得到完整的证明。

```py
example  {X  Y  :  Type*}  [MetricSpace  X]  [MetricSpace  Y]  {f  :  X  →  Y}  (hf  :  Continuous  f)  :
  Continuous  fun  p  :  X  ×  X  ↦  dist  (f  p.1)  (f  p.2)  :=
  continuous_dist.comp  ((hf.comp  continuous_fst).prodMk  (hf.comp  continuous_snd)) 
```

通过 `Continuous.comp` 将 `Continuous.prod_mk` 和 `continuous_dist` 结合在一起感觉有些笨拙，即使像上面那样大量使用点符号也是如此。一个更严重的问题是，这个漂亮的证明需要大量的规划。Lean 接受上述证明项，因为它是一个完整的项，证明了与我们的目标定义等价的一个陈述，关键的定义展开是函数的复合。确实，我们的目标函数 `fun p : X × X ↦ dist (f p.1) (f p.2)` 并没有以复合的形式呈现。我们提供的证明项证明了 `dist ∘ (fun p : X × X ↦ (f p.1, f p.2))` 的连续性，这恰好与我们的目标函数在定义上是等价的。但如果我们尝试从 `apply continuous_dist.comp` 开始逐步构建这个证明，Lean 的展开器将无法识别复合，并拒绝应用这个引理。当涉及到类型乘积时，这个问题尤其严重。

在这里应用更好的引理是 `Continuous.dist {f g : X → Y} : Continuous f → Continuous g → Continuous (fun x ↦ dist (f x) (g x))`，这对于 Lean 的展开器来说更易于处理，并且在直接提供完整的证明项时也提供了更短的证明，如下面的两个新证明所示：

```py
example  {X  Y  :  Type*}  [MetricSpace  X]  [MetricSpace  Y]  {f  :  X  →  Y}  (hf  :  Continuous  f)  :
  Continuous  fun  p  :  X  ×  X  ↦  dist  (f  p.1)  (f  p.2)  :=  by
  apply  Continuous.dist
  exact  hf.comp  continuous_fst
  exact  hf.comp  continuous_snd

example  {X  Y  :  Type*}  [MetricSpace  X]  [MetricSpace  Y]  {f  :  X  →  Y}  (hf  :  Continuous  f)  :
  Continuous  fun  p  :  X  ×  X  ↦  dist  (f  p.1)  (f  p.2)  :=
  (hf.comp  continuous_fst).dist  (hf.comp  continuous_snd) 
```

注意，如果没有由于复合引起的展开问题，另一种压缩我们证明的方法是使用 `Continuous.prod_map`，这在某些情况下很有用，并给出了一个替代的证明项 `continuous_dist.comp (hf.prod_map hf)`，这甚至更短于输入类型。

由于决定一个版本对详细说明更好，而另一个版本则更短，所以我们用`Continuous.fst'`提供的最后一点压缩来结束这次讨论，它允许将`hf.comp continuous_fst`压缩为`hf.fst'`（以及`snd`），并得到我们的最终证明，现在接近模糊。

```py
example  {X  Y  :  Type*}  [MetricSpace  X]  [MetricSpace  Y]  {f  :  X  →  Y}  (hf  :  Continuous  f)  :
  Continuous  fun  p  :  X  ×  X  ↦  dist  (f  p.1)  (f  p.2)  :=
  hf.fst'.dist  hf.snd' 
```

现在轮到你来证明一些连续性引理了。在尝试连续性策略之后，你需要`Continuous.add`、`continuous_pow`和`continuous_id`来手动完成。

```py
example  {f  :  ℝ  →  X}  (hf  :  Continuous  f)  :  Continuous  fun  x  :  ℝ  ↦  f  (x  ^  2  +  x)  :=
  sorry 
```

到目前为止，我们看到了连续性作为一个全局概念，但也可以定义在一点上的连续性。

```py
example  {X  Y  :  Type*}  [MetricSpace  X]  [MetricSpace  Y]  (f  :  X  →  Y)  (a  :  X)  :
  ContinuousAt  f  a  ↔  ∀  ε  >  0,  ∃  δ  >  0,  ∀  {x},  dist  x  a  <  δ  →  dist  (f  x)  (f  a)  <  ε  :=
  Metric.continuousAt_iff 
```

### 11.2.2\. 球、开集和闭集

一旦我们有了距离函数，最重要的几何定义就是（开）球和闭球。

```py
variable  (r  :  ℝ)

example  :  Metric.ball  a  r  =  {  b  |  dist  b  a  <  r  }  :=
  rfl

example  :  Metric.closedBall  a  r  =  {  b  |  dist  b  a  ≤  r  }  :=
  rfl 
```

注意，这里的 r 是任何实数，没有符号限制。当然，一些陈述确实需要半径条件。

```py
example  (hr  :  0  <  r)  :  a  ∈  Metric.ball  a  r  :=
  Metric.mem_ball_self  hr

example  (hr  :  0  ≤  r)  :  a  ∈  Metric.closedBall  a  r  :=
  Metric.mem_closedBall_self  hr 
```

一旦我们有了球，我们就可以定义开集。实际上，它们是在更一般的设置中定义的，这将在下一节中介绍，但我们有引理将定义重新表述为球的形式。

```py
example  (s  :  Set  X)  :  IsOpen  s  ↔  ∀  x  ∈  s,  ∃  ε  >  0,  Metric.ball  x  ε  ⊆  s  :=
  Metric.isOpen_iff 
```

然后闭集是补集为开集的集合。它们的重要性质是它们在极限下是封闭的。一个集合的闭包是包含它的最小闭集。

```py
example  {s  :  Set  X}  :  IsClosed  s  ↔  IsOpen  (sᶜ)  :=
  isOpen_compl_iff.symm

example  {s  :  Set  X}  (hs  :  IsClosed  s)  {u  :  ℕ  →  X}  (hu  :  Tendsto  u  atTop  (𝓝  a))
  (hus  :  ∀  n,  u  n  ∈  s)  :  a  ∈  s  :=
  hs.mem_of_tendsto  hu  (Eventually.of_forall  hus)

example  {s  :  Set  X}  :  a  ∈  closure  s  ↔  ∀  ε  >  0,  ∃  b  ∈  s,  a  ∈  Metric.ball  b  ε  :=
  Metric.mem_closure_iff 
```

在不使用`mem_closure_iff_seq_limit`的情况下完成下一个练习

```py
example  {u  :  ℕ  →  X}  (hu  :  Tendsto  u  atTop  (𝓝  a))  {s  :  Set  X}  (hs  :  ∀  n,  u  n  ∈  s)  :
  a  ∈  closure  s  :=  by
  sorry 
```

记住从过滤器部分，邻域过滤器在 Mathlib 中起着重要作用。在度量空间的情况下，关键点是球提供了这些过滤器的基。这里的主要引理是`Metric.nhds_basis_ball`和`Metric.nhds_basis_closedBall`，它们声称对于正半径的开球和闭球。中心点是隐含的参数，因此我们可以像以下示例中那样调用`Filter.HasBasis.mem_iff`。

```py
example  {x  :  X}  {s  :  Set  X}  :  s  ∈  𝓝  x  ↔  ∃  ε  >  0,  Metric.ball  x  ε  ⊆  s  :=
  Metric.nhds_basis_ball.mem_iff

example  {x  :  X}  {s  :  Set  X}  :  s  ∈  𝓝  x  ↔  ∃  ε  >  0,  Metric.closedBall  x  ε  ⊆  s  :=
  Metric.nhds_basis_closedBall.mem_iff 
```

### 11.2.3\. 紧致性

紧致性是一个重要的拓扑概念。它区分了度量空间中的子集，这些子集与其他区间相比，与实数中的线段具有相同的性质：

+   任何在紧致集合中取值的序列都有一个子序列在这个集合中收敛。

+   任何在非空紧致集合上定义的连续函数，其值在实数中都是有界的，并且在其某个地方达到其界限（这被称为极值定理）。

+   紧致集是闭集。

让我们先检查实数中的单位区间确实是一个紧致集合，然后检查一般度量空间中紧致集合的上述断言。在第二个陈述中，我们只需要给定集合上的连续性，所以我们将使用`ContinuousOn`而不是`Continuous`，我们将为最小值和最大值给出单独的陈述。当然，所有这些结果都是从更一般的版本中推导出来的，其中一些将在后面的章节中讨论。

```py
example  :  IsCompact  (Set.Icc  0  1  :  Set  ℝ)  :=
  isCompact_Icc

example  {s  :  Set  X}  (hs  :  IsCompact  s)  {u  :  ℕ  →  X}  (hu  :  ∀  n,  u  n  ∈  s)  :
  ∃  a  ∈  s,  ∃  φ  :  ℕ  →  ℕ,  StrictMono  φ  ∧  Tendsto  (u  ∘  φ)  atTop  (𝓝  a)  :=
  hs.tendsto_subseq  hu

example  {s  :  Set  X}  (hs  :  IsCompact  s)  (hs'  :  s.Nonempty)  {f  :  X  →  ℝ}
  (hfs  :  ContinuousOn  f  s)  :
  ∃  x  ∈  s,  ∀  y  ∈  s,  f  x  ≤  f  y  :=
  hs.exists_isMinOn  hs'  hfs

example  {s  :  Set  X}  (hs  :  IsCompact  s)  (hs'  :  s.Nonempty)  {f  :  X  →  ℝ}
  (hfs  :  ContinuousOn  f  s)  :
  ∃  x  ∈  s,  ∀  y  ∈  s,  f  y  ≤  f  x  :=
  hs.exists_isMaxOn  hs'  hfs

example  {s  :  Set  X}  (hs  :  IsCompact  s)  :  IsClosed  s  :=
  hs.isClosed 
```

我们还可以指定一个度量空间是全局紧致的，使用一个额外的`Prop`值类型类：

```py
example  {X  :  Type*}  [MetricSpace  X]  [CompactSpace  X]  :  IsCompact  (univ  :  Set  X)  :=
  isCompact_univ 
```

在紧致度量空间中，任何闭集都是紧致的，这是`IsClosed.isCompact`。

### 11.2.4\. 均匀连续函数

我们现在转向度量空间上的均匀性概念：均匀连续函数、柯西序列和完备性。同样，这些都是在更一般的环境中定义的，但我们有在度量名称空间中的引理来访问它们的元素定义。我们首先从均匀连续性开始。

```py
example  {X  :  Type*}  [MetricSpace  X]  {Y  :  Type*}  [MetricSpace  Y]  {f  :  X  →  Y}  :
  UniformContinuous  f  ↔
  ∀  ε  >  0,  ∃  δ  >  0,  ∀  {a  b  :  X},  dist  a  b  <  δ  →  dist  (f  a)  (f  b)  <  ε  :=
  Metric.uniformContinuous_iff 
```

为了练习操作所有这些定义，我们将证明从紧致度量空间到度量空间的连续函数是均匀连续的（我们将在稍后的章节中看到更一般的形式）。

我们首先给出一个非正式的草图。设 `f : X → Y` 是从一个紧致度量空间到度量空间的连续函数。我们固定 `ε > 0` 并开始寻找某个 `δ`。

设 `φ : X × X → ℝ := fun p ↦ dist (f p.1) (f p.2)` 和 `K := { p : X × X | ε ≤ φ p }`。观察 `φ` 是连续的，因为 `f` 和距离都是连续的。并且 `K` 显然是闭集（使用 `isClosed_le`），因此由于 `X` 是紧致的，`K` 也是紧致的。

然后我们讨论两种可能性，使用 `eq_empty_or_nonempty`。如果 `K` 是空的，那么我们显然已经完成了（例如，我们可以将 `δ` 设置为 `1`）。所以让我们假设 `K` 不是空的，并使用极值定理来选择 `(x₀, x₁)`，它达到距离函数在 `K` 上的下确界。然后我们可以将 `δ` 设置为 `dist x₀ x₁` 并检查一切是否正常工作。

```py
example  {X  :  Type*}  [MetricSpace  X]  [CompactSpace  X]
  {Y  :  Type*}  [MetricSpace  Y]  {f  :  X  →  Y}
  (hf  :  Continuous  f)  :  UniformContinuous  f  :=  by
  sorry 
```

### 11.2.5\. 完备性

在度量空间中的柯西序列是一个其项彼此越来越接近的序列。有几种等价的方式来表述这个想法。特别是收敛序列是柯西序列。逆命题仅在所谓的 *完备* 空间中成立。

```py
example  (u  :  ℕ  →  X)  :
  CauchySeq  u  ↔  ∀  ε  >  0,  ∃  N  :  ℕ,  ∀  m  ≥  N,  ∀  n  ≥  N,  dist  (u  m)  (u  n)  <  ε  :=
  Metric.cauchySeq_iff

example  (u  :  ℕ  →  X)  :
  CauchySeq  u  ↔  ∀  ε  >  0,  ∃  N  :  ℕ,  ∀  n  ≥  N,  dist  (u  n)  (u  N)  <  ε  :=
  Metric.cauchySeq_iff'

example  [CompleteSpace  X]  (u  :  ℕ  →  X)  (hu  :  CauchySeq  u)  :
  ∃  x,  Tendsto  u  atTop  (𝓝  x)  :=
  cauchySeq_tendsto_of_complete  hu 
```

我们将通过证明一个方便的判据来练习使用这个定义，这个判据是 Mathlib 中出现的一个判据的特殊情况。这也是练习在几何环境中使用大和的好机会。除了过滤器部分的解释外，你可能还需要 `tendsto_pow_atTop_nhds_zero_of_lt_one`、`Tendsto.mul` 和 `dist_le_range_sum_dist`。

```py
theorem  cauchySeq_of_le_geometric_two'  {u  :  ℕ  →  X}
  (hu  :  ∀  n  :  ℕ,  dist  (u  n)  (u  (n  +  1))  ≤  (1  /  2)  ^  n)  :  CauchySeq  u  :=  by
  rw  [Metric.cauchySeq_iff']
  intro  ε  ε_pos
  obtain  ⟨N,  hN⟩  :  ∃  N  :  ℕ,  1  /  2  ^  N  *  2  <  ε  :=  by  sorry
  use  N
  intro  n  hn
  obtain  ⟨k,  rfl  :  n  =  N  +  k⟩  :=  le_iff_exists_add.mp  hn
  calc
  dist  (u  (N  +  k))  (u  N)  =  dist  (u  (N  +  0))  (u  (N  +  k))  :=  sorry
  _  ≤  ∑  i  ∈  range  k,  dist  (u  (N  +  i))  (u  (N  +  (i  +  1)))  :=  sorry
  _  ≤  ∑  i  ∈  range  k,  (1  /  2  :  ℝ)  ^  (N  +  i)  :=  sorry
  _  =  1  /  2  ^  N  *  ∑  i  ∈  range  k,  (1  /  2  :  ℝ)  ^  i  :=  sorry
  _  ≤  1  /  2  ^  N  *  2  :=  sorry
  _  <  ε  :=  sorry 
```

我们已经准备好本节的最终大 BOSS：完备度量空间的 Baire 定理！下面的证明框架展示了有趣的技巧。它使用了感叹号变体的 `choose` 策略（你应该尝试移除这个感叹号）并展示了如何在证明中使用 `Nat.rec_on` 递归地定义某物。

```py
open  Metric

example  [CompleteSpace  X]  (f  :  ℕ  →  Set  X)  (ho  :  ∀  n,  IsOpen  (f  n))  (hd  :  ∀  n,  Dense  (f  n))  :
  Dense  (⋂  n,  f  n)  :=  by
  let  B  :  ℕ  →  ℝ  :=  fun  n  ↦  (1  /  2)  ^  n
  have  Bpos  :  ∀  n,  0  <  B  n
  sorry
  /- Translate the density assumption into two functions `center` and `radius` associating
 to any n, x, δ, δpos a center and a positive radius such that
 `closedBall center radius` is included both in `f n` and in `closedBall x δ`.
 We can also require `radius ≤ (1/2)^(n+1)`, to ensure we get a Cauchy sequence later. -/
  have  :
  ∀  (n  :  ℕ)  (x  :  X),
  ∀  δ  >  0,  ∃  y  :  X,  ∃  r  >  0,  r  ≤  B  (n  +  1)  ∧  closedBall  y  r  ⊆  closedBall  x  δ  ∩  f  n  :=
  by  sorry
  choose!  center  radius  Hpos  HB  Hball  using  this
  intro  x
  rw  [mem_closure_iff_nhds_basis  nhds_basis_closedBall]
  intro  ε  εpos
  /- `ε` is positive. We have to find a point in the ball of radius `ε` around `x`
 belonging to all `f n`. For this, we construct inductively a sequence
 `F n = (c n, r n)` such that the closed ball `closedBall (c n) (r n)` is included
 in the previous ball and in `f n`, and such that `r n` is small enough to ensure
 that `c n` is a Cauchy sequence. Then `c n` converges to a limit which belongs
 to all the `f n`. -/
  let  F  :  ℕ  →  X  ×  ℝ  :=  fun  n  ↦
  Nat.recOn  n  (Prod.mk  x  (min  ε  (B  0)))
  fun  n  p  ↦  Prod.mk  (center  n  p.1  p.2)  (radius  n  p.1  p.2)
  let  c  :  ℕ  →  X  :=  fun  n  ↦  (F  n).1
  let  r  :  ℕ  →  ℝ  :=  fun  n  ↦  (F  n).2
  have  rpos  :  ∀  n,  0  <  r  n  :=  by  sorry
  have  rB  :  ∀  n,  r  n  ≤  B  n  :=  by  sorry
  have  incl  :  ∀  n,  closedBall  (c  (n  +  1))  (r  (n  +  1))  ⊆  closedBall  (c  n)  (r  n)  ∩  f  n  :=  by
  sorry
  have  cdist  :  ∀  n,  dist  (c  n)  (c  (n  +  1))  ≤  B  n  :=  by  sorry
  have  :  CauchySeq  c  :=  cauchySeq_of_le_geometric_two'  cdist
  -- as the sequence `c n` is Cauchy in a complete space, it converges to a limit `y`.
  rcases  cauchySeq_tendsto_of_complete  this  with  ⟨y,  ylim⟩
  -- this point `y` will be the desired point. We will check that it belongs to all
  -- `f n` and to `ball x ε`.
  use  y
  have  I  :  ∀  n,  ∀  m  ≥  n,  closedBall  (c  m)  (r  m)  ⊆  closedBall  (c  n)  (r  n)  :=  by  sorry
  have  yball  :  ∀  n,  y  ∈  closedBall  (c  n)  (r  n)  :=  by  sorry
  sorry 
```

## 11.3\. 拓扑空间

### 11.3.1\. 基础

我们现在提高一般性，引入拓扑空间。我们将回顾定义拓扑空间的两种主要方法，然后解释拓扑空间范畴比度量空间范畴表现得更好。请注意，我们在这里不会使用 Mathlib 的范畴论，而只是有一个某种范畴的观点。

考虑从度量空间到拓扑空间的过渡的第一种方法是，我们只记住开集的概念（或等价地，闭集的概念）。从这个角度来看，拓扑空间是一种类型，它配备了一组称为开集的集合。这个集合必须满足以下（下面）提出的若干公理（这个集合略微冗余，但我们将忽略这一点）。

```py
section
variable  {X  :  Type*}  [TopologicalSpace  X]

example  :  IsOpen  (univ  :  Set  X)  :=
  isOpen_univ

example  :  IsOpen  (∅  :  Set  X)  :=
  isOpen_empty

example  {ι  :  Type*}  {s  :  ι  →  Set  X}  (hs  :  ∀  i,  IsOpen  (s  i))  :  IsOpen  (⋃  i,  s  i)  :=
  isOpen_iUnion  hs

example  {ι  :  Type*}  [Fintype  ι]  {s  :  ι  →  Set  X}  (hs  :  ∀  i,  IsOpen  (s  i))  :
  IsOpen  (⋂  i,  s  i)  :=
  isOpen_iInter_of_finite  hs 
```

闭集定义为补集是开集的集合。在拓扑空间之间的函数（全局）连续，如果所有开集的前像都是开集。

```py
variable  {Y  :  Type*}  [TopologicalSpace  Y]

example  {f  :  X  →  Y}  :  Continuous  f  ↔  ∀  s,  IsOpen  s  →  IsOpen  (f  ⁻¹'  s)  :=
  continuous_def 
```

根据这个定义，我们已看到，与度量空间相比，拓扑空间只记住足够的信息来讨论连续函数：一个类型上的两个拓扑结构相同，当且仅当它们有相同的连续函数（实际上，如果两个结构有相同的开集，那么恒等函数在这两个方向上都将连续）。

然而，一旦我们转向点的连续性，我们就看到了基于开集的方法的局限性。在 Mathlib 中，我们经常将拓扑空间视为类型，每个点`x`都附有一个邻域滤波器`𝓝 x`（相应的函数`X → Filter X`满足以下条件，将在下面进一步解释）。记住从滤波器部分，这些小玩意儿扮演两个相关的角色。首先，`𝓝 x`被视为接近`x`的`X`点的广义集合。然后，它被视为提供了一种方式，对于任何谓词`P : X → Prop`，可以说明这个谓词对于足够接近`x`的点成立。让我们声明`f : X → Y`在`x`处是连续的。纯粹基于滤波器的方式是说，`f`的直接像包含接近`x`的点的广义集合，包含在接近`f x`的点的广义集合中。回忆一下，这可以表示为`map f (𝓝 x) ≤ 𝓝 (f x)`或`Tendsto f (𝓝 x) (𝓝 (f x))`。

```py
example  {f  :  X  →  Y}  {x  :  X}  :  ContinuousAt  f  x  ↔  map  f  (𝓝  x)  ≤  𝓝  (f  x)  :=
  Iff.rfl 
```

也可以使用两种邻域（视为普通集合）和一种邻域滤波器（视为广义集合）来表述：对于`f x`的任何邻域`U`，所有接近`x`的点都被发送到`U`。请注意，证明仍然是`Iff.rfl`，这个观点在定义上等同于前一个观点。

```py
example  {f  :  X  →  Y}  {x  :  X}  :  ContinuousAt  f  x  ↔  ∀  U  ∈  𝓝  (f  x),  ∀ᶠ  x  in  𝓝  x,  f  x  ∈  U  :=
  Iff.rfl 
```

现在我们解释如何从一个观点过渡到另一个观点。从开集的角度来看，我们可以简单地定义`𝓝 x`的成员为包含`x`的开集的集合。

```py
example  {x  :  X}  {s  :  Set  X}  :  s  ∈  𝓝  x  ↔  ∃  t,  t  ⊆  s  ∧  IsOpen  t  ∧  x  ∈  t  :=
  mem_nhds_iff 
```

要朝相反的方向前进，我们需要讨论`𝓝 : X → Filter X`必须满足的条件，以便成为拓扑的邻域函数。

第一个约束是，`𝓝 x`（视为广义集合）包含被视为广义集合`pure x`的集合`{x}`（解释这个奇怪的名字会太分散注意力，所以我们现在简单地接受它）。另一种说法是，如果一个谓词对于接近`x`的点成立，那么它在`x`处也成立。

```py
example  (x  :  X)  :  pure  x  ≤  𝓝  x  :=
  pure_le_nhds  x

example  (x  :  X)  (P  :  X  →  Prop)  (h  :  ∀ᶠ  y  in  𝓝  x,  P  y)  :  P  x  :=
  h.self_of_nhds 
```

然后一个更微妙的要求是，对于任何谓词`P : X → Prop`和任何`x`，如果`P y`对接近`x`的`y`成立，那么对于接近`x`和接近`y`的`z`，`P z`也成立。更精确地说，我们有：

```py
example  {P  :  X  →  Prop}  {x  :  X}  (h  :  ∀ᶠ  y  in  𝓝  x,  P  y)  :  ∀ᶠ  y  in  𝓝  x,  ∀ᶠ  z  in  𝓝  y,  P  z  :=
  eventually_eventually_nhds.mpr  h 
```

这两个结果描述了`X → Filter X`的函数，这些函数是`X`上的拓扑空间结构的邻域函数。仍然有一个函数`TopologicalSpace.mkOfNhds : (X → Filter X) → TopologicalSpace X`，但它只有在满足上述两个约束时才会将其输入作为邻域函数返回。更精确地说，我们有一个引理`TopologicalSpace.nhds_mkOfNhds`，它以不同的方式表达，我们的下一个练习将从上述方式推导出这一点。

```py
example  {α  :  Type*}  (n  :  α  →  Filter  α)  (H₀  :  ∀  a,  pure  a  ≤  n  a)
  (H  :  ∀  a  :  α,  ∀  p  :  α  →  Prop,  (∀ᶠ  x  in  n  a,  p  x)  →  ∀ᶠ  y  in  n  a,  ∀ᶠ  x  in  n  y,  p  x)  :
  ∀  a,  ∀  s  ∈  n  a,  ∃  t  ∈  n  a,  t  ⊆  s  ∧  ∀  a'  ∈  t,  s  ∈  n  a'  :=  by
  sorry
end 
```

注意，`TopologicalSpace.mkOfNhds`并不经常使用，但了解在拓扑空间结构中邻域滤波器究竟意味着什么仍然是有益的。

为了在 Mathlib 中高效地使用拓扑空间，需要知道的是，我们使用了`TopologicalSpace : Type u → Type u`的许多形式属性。从纯粹数学的角度来看，这些形式属性是解释拓扑空间如何解决度量空间存在的问题的一种非常干净的方式。从这个角度来看，拓扑空间解决的问题在于度量空间几乎不具有函子性，并且在一般意义上具有非常差的范畴性质。这还基于已经讨论的事实，即度量空间包含大量与拓扑无关的几何信息。

让我们先关注函子性。度量空间结构可以诱导在子集上，或者等价地，它可以由一个注入映射拉回。但这就是全部了。它们不能由一般映射或推进，甚至不能由满射映射拉回。

特别是，在度量空间的商或不可数度量空间的积上无法放置一个有意义的距离。例如，考虑类型`ℝ → ℝ`，它被视为由`ℝ`索引的`ℝ`的复制品的积。我们希望说，函数序列逐点收敛是一个值得尊重的收敛概念。但在`ℝ → ℝ`上没有距离可以给出这种收敛概念。相关地，没有距离可以保证映射`f : X → (ℝ → ℝ)`是连续的当且仅当对于每个`t : ℝ`，`fun x ↦ f x t`是连续的。

我们现在回顾用于解决所有这些问题的数据。首先，我们可以使用任何映射`f : X → Y`来从一个方向推进或拉回拓扑。这两个操作形成一个 Galois 连接。

```py
variable  {X  Y  :  Type*}

example  (f  :  X  →  Y)  :  TopologicalSpace  X  →  TopologicalSpace  Y  :=
  TopologicalSpace.coinduced  f

example  (f  :  X  →  Y)  :  TopologicalSpace  Y  →  TopologicalSpace  X  :=
  TopologicalSpace.induced  f

example  (f  :  X  →  Y)  (T_X  :  TopologicalSpace  X)  (T_Y  :  TopologicalSpace  Y)  :
  TopologicalSpace.coinduced  f  T_X  ≤  T_Y  ↔  T_X  ≤  TopologicalSpace.induced  f  T_Y  :=
  coinduced_le_iff_le_induced 
```

这些操作与函数的组合是兼容的。通常，推进是协变的，而拉回是反对称的，参见`coinduced_compose`和`induced_compose`。在纸上，我们将使用`f_*T`表示`TopologicalSpace.coinduced f T`，使用`f^*T`表示`TopologicalSpace.induced f T`。

接下来的重要部分是对于任何给定的结构在 `TopologicalSpace X` 上的完备格结构。如果你认为拓扑主要是开集的数据，那么你期望 `TopologicalSpace X` 上的序关系来自 `Set (Set X)`，即你期望如果集合 `u` 对于 `t'` 是开集，那么它对于 `t` 也是开集，你就期望 `t ≤ t'`。然而，我们已经知道 Mathlib 更关注邻域而不是开集，所以对于任何 `x : X`，我们希望从拓扑空间到邻域的映射 `fun T : TopologicalSpace X ↦ @nhds X T x` 是序保持的。我们还知道 `Filter X` 上的序关系被设计成确保序保持的 `principal : Set X → Filter X`，允许将过滤器视为广义集合。因此，我们在 `TopologicalSpace X` 上使用的序关系与来自 `Set (Set X)` 的序关系相反。

```py
example  {T  T'  :  TopologicalSpace  X}  :  T  ≤  T'  ↔  ∀  s,  T'.IsOpen  s  →  T.IsOpen  s  :=
  Iff.rfl 
```

现在我们可以通过结合推前（或拉回）操作与序关系来恢复连续性。

```py
example  (T_X  :  TopologicalSpace  X)  (T_Y  :  TopologicalSpace  Y)  (f  :  X  →  Y)  :
  Continuous  f  ↔  TopologicalSpace.coinduced  f  T_X  ≤  T_Y  :=
  continuous_iff_coinduced_le 
```

通过这个定义和推前与复合的兼容性，我们免费得到了这样的泛性质：对于任何拓扑空间 $Z$，函数 $g : Y → Z$ 在拓扑 $f_*T_X$ 上是连续的，当且仅当 $g ∘ f$ 是连续的。

$$\begin{split}g \text{ continuous } &⇔ g_*(f_*T_X) ≤ T_Z \\ &⇔ (g ∘ f)_* T_X ≤ T_Z \\ &⇔ g ∘ f \text{ continuous}\end{split}$$

```py
example  {Z  :  Type*}  (f  :  X  →  Y)  (T_X  :  TopologicalSpace  X)  (T_Z  :  TopologicalSpace  Z)
  (g  :  Y  →  Z)  :
  @Continuous  Y  Z  (TopologicalSpace.coinduced  f  T_X)  T_Z  g  ↔
  @Continuous  X  Z  T_X  T_Z  (g  ∘  f)  :=  by
  rw  [continuous_iff_coinduced_le,  coinduced_compose,  continuous_iff_coinduced_le] 
```

因此，我们已得到商拓扑（使用投影映射作为 `f`）。这并不是使用 `TopologicalSpace X` 是所有 `X` 的完备格。现在让我们看看所有这些结构是如何通过抽象的胡言乱语来证明乘积拓扑的存在性的。我们上面考虑了 `ℝ → ℝ` 的情况，但现在让我们考虑一般情况 `Π i, X i` 对于某些 `ι : Type*` 和 `X : ι → Type*`。我们希望对于任何拓扑空间 `Z` 和任何函数 `f : Z → Π i, X i`，如果 `f` 是连续的，当且仅当对于所有 `i`，`(fun x ↦ x i) ∘ f` 是连续的。让我们使用表示投影 `(fun (x : Π i, X i) ↦ x i)` 的符号 $p_i$ 在“纸上”探索这个约束：

$$\begin{split}(∀ i, p_i ∘ f \text{ continuous}) &⇔ ∀ i, (p_i ∘ f)_* T_Z ≤ T_{X_i} \\ &⇔ ∀ i, (p_i)_* f_* T_Z ≤ T_{X_i}\\ &⇔ ∀ i, f_* T_Z ≤ (p_i)^*T_{X_i}\\ &⇔ f_* T_Z ≤ \inf \left[(p_i)^*T_{X_i}\right]\end{split}$$

因此，我们看到我们希望在 `Π i, X i` 上具有的拓扑结构：

```py
example  (ι  :  Type*)  (X  :  ι  →  Type*)  (T_X  :  ∀  i,  TopologicalSpace  (X  i))  :
  (Pi.topologicalSpace  :  TopologicalSpace  (∀  i,  X  i))  =
  ⨅  i,  TopologicalSpace.induced  (fun  x  ↦  x  i)  (T_X  i)  :=
  rfl 
```

这就结束了我们对 Mathlib 怎样认为拓扑空间通过成为一个更函子化的理论和为任何固定类型具有完备格结构来修复度量空间理论的缺陷的考察。

### 11.3.2. 分离与可数性

我们看到拓扑空间的范畴具有非常美好的性质。为此付出的代价是存在相当病态的拓扑空间。你可以对拓扑空间做出一些假设，以确保其行为更接近度量空间。其中最重要的是 `T2Space`，也称为“豪斯多夫”，这将确保极限是唯一的。更强的分离性质是 `T3Space`，它还确保了正则空间性质：每个点都有一个闭邻域基。

```py
example  [TopologicalSpace  X]  [T2Space  X]  {u  :  ℕ  →  X}  {a  b  :  X}  (ha  :  Tendsto  u  atTop  (𝓝  a))
  (hb  :  Tendsto  u  atTop  (𝓝  b))  :  a  =  b  :=
  tendsto_nhds_unique  ha  hb

example  [TopologicalSpace  X]  [RegularSpace  X]  (a  :  X)  :
  (𝓝  a).HasBasis  (fun  s  :  Set  X  ↦  s  ∈  𝓝  a  ∧  IsClosed  s)  id  :=
  closed_nhds_basis  a 
```

注意，在每一个拓扑空间中，每个点都有一个开邻域基，这是定义所要求的。

```py
example  [TopologicalSpace  X]  {x  :  X}  :
  (𝓝  x).HasBasis  (fun  t  :  Set  X  ↦  t  ∈  𝓝  x  ∧  IsOpen  t)  id  :=
  nhds_basis_opens'  x 
```

我们现在的目标是证明一个基本定理，该定理允许通过连续性进行扩展。从布尔巴基的《一般拓扑学》第 I 卷第八章，定理 1（仅考虑非平凡蕴含）：

设 $X$ 为一个拓扑空间，$A$ 为 $X$ 的一个稠密子集，$f : A → Y$ 为将 $A$ 映射到 $T_3$ 空间 $Y$ 的连续映射。如果对于 $X$ 中的每个 $x$，当 $y$ 在 $A$ 中趋向于 $x$ 时，$f(y)$ 在 $Y$ 中趋向于一个极限，那么存在一个 $f$ 在 $X$ 上的连续扩展 $φ$。

实际上，Mathlib 包含了上述引理的一个更一般的版本，即 `IsDenseInducing.continuousAt_extend`，但在这里我们将坚持使用布尔巴基的版本。

记住，给定 `A : Set X`，`↥A` 是与 `A` 关联的子类型，Lean 将在需要时自动插入那个古怪的向上箭头。并且（包含）强制映射是 `(↑) : A → X`。假设“在 $A$ 中趋向于 $x$”对应于拉回过滤器 `comap (↑) (𝓝 x)`。

让我们先证明一个辅助引理，将其提取出来以简化上下文（特别是我们在这里不需要 $Y$ 是一个拓扑空间）。

```py
theorem  aux  {X  Y  A  :  Type*}  [TopologicalSpace  X]  {c  :  A  →  X}
  {f  :  A  →  Y}  {x  :  X}  {F  :  Filter  Y}
  (h  :  Tendsto  f  (comap  c  (𝓝  x))  F)  {V'  :  Set  Y}  (V'_in  :  V'  ∈  F)  :
  ∃  V  ∈  𝓝  x,  IsOpen  V  ∧  c  ⁻¹'  V  ⊆  f  ⁻¹'  V'  :=  by
  sorry 
```

现在我们转向连续扩展定理的主要证明。

当 Lean 需要一个 `↥A` 上的拓扑时，它将自动使用诱导拓扑。唯一相关的引理是 `nhds_induced (↑) : ∀ a : ↥A, 𝓝 a = comap (↑) (𝓝 ↑a)`（这实际上是一个关于诱导拓扑的一般引理）。

证明的大致思路是：

主要假设和选择公理给出一个函数 `φ`，使得 `∀ x, Tendsto f (comap (↑) (𝓝 x)) (𝓝 (φ x))`（因为 `Y` 是豪斯多夫的，`φ` 是完全确定的，但我们不会在尝试证明 `φ` 确实扩展 `f` 之前需要这一点）。

让我们先证明 `φ` 是连续的。固定任意的 `x : X`。由于 `Y` 是正则的，我们只需要检查对于 `φ x` 的每个 *闭* 邻域 `V'`，`φ ⁻¹' V' ∈ 𝓝 x`。极限假设给出了（通过上面的辅助引理）一些 `V ∈ 𝓝 x`，使得 `IsOpen V ∧ (↑) ⁻¹' V ⊆ f ⁻¹' V'`。由于 `V ∈ 𝓝 x`，我们只需要证明 `V ⊆ φ ⁻¹' V'`，即 `∀ y ∈ V, φ y ∈ V'`。让我们固定 `V` 中的 `y`。因为 `V` 是 *开* 的，它是 `y` 的邻域。特别是 `(↑) ⁻¹' V ∈ comap (↑) (𝓝 y)`，并且更明显 `f ⁻¹' V' ∈ comap (↑) (𝓝 y)`。此外，`comap (↑) (𝓝 y) ≠ ⊥` 因为 `A` 是稠密的。因为我们知道 `Tendsto f (comap (↑) (𝓝 y)) (𝓝 (φ y))`，这表明 `φ y ∈ closure V'`，并且由于 `V'` 是闭的，我们证明了 `φ y ∈ V'`。

剩下的工作是证明 `φ` 扩展了 `f`。这是 `f` 的连续性进入讨论的地方，以及 `Y` 是豪斯多夫空间的事实。

```py
example  [TopologicalSpace  X]  [TopologicalSpace  Y]  [T3Space  Y]  {A  :  Set  X}
  (hA  :  ∀  x,  x  ∈  closure  A)  {f  :  A  →  Y}  (f_cont  :  Continuous  f)
  (hf  :  ∀  x  :  X,  ∃  c  :  Y,  Tendsto  f  (comap  (↑)  (𝓝  x))  (𝓝  c))  :
  ∃  φ  :  X  →  Y,  Continuous  φ  ∧  ∀  a  :  A,  φ  a  =  f  a  :=  by
  sorry

#check  HasBasis.tendsto_right_iff 
```

除了分离性质外，你可以在拓扑空间上做出的主要假设，以使其更接近度量空间，是可数性假设。主要的一个是第一可数性，要求每个点都有一个可数的邻域基。特别是这确保了集合的闭包可以用序列来理解。

```py
example  [TopologicalSpace  X]  [FirstCountableTopology  X]
  {s  :  Set  X}  {a  :  X}  :
  a  ∈  closure  s  ↔  ∃  u  :  ℕ  →  X,  (∀  n,  u  n  ∈  s)  ∧  Tendsto  u  atTop  (𝓝  a)  :=
  mem_closure_iff_seq_limit 
```

### 11.3.3\. 紧致性

让我们现在讨论拓扑空间中紧致性的定义。通常有几种思考方式，Mathlib 采用的是过滤器版本。

我们首先需要定义过滤器的聚点。给定一个拓扑空间 `X` 上的过滤器 `F`，一个点 `x : X` 是 `F` 的聚点，如果 `F` 作为广义集合，与接近 `x` 的点的广义集合有非空交集。

然后，我们可以说一个集合 `s` 是紧致的，如果每个包含在 `s` 中的非空广义集合 `F`，即 `F ≤ 𝓟 s`，在 `s` 中都有一个聚点。

```py
variable  [TopologicalSpace  X]

example  {F  :  Filter  X}  {x  :  X}  :  ClusterPt  x  F  ↔  NeBot  (𝓝  x  ⊓  F)  :=
  Iff.rfl

example  {s  :  Set  X}  :
  IsCompact  s  ↔  ∀  (F  :  Filter  X)  [NeBot  F],  F  ≤  𝓟  s  →  ∃  a  ∈  s,  ClusterPt  a  F  :=
  Iff.rfl 
```

例如，如果 `F` 是 `map u atTop`，即 `u : ℕ → X` 在 `atTop`（一个非常大的自然数的广义集合）下的像，那么假设 `F ≤ 𝓟 s` 意味着对于足够大的 `n`，`u n` 属于 `s`。说 `x` 是 `map u atTop` 的聚点意味着非常大的数的像与接近 `x` 的点的集合相交。如果 `𝓝 x` 有一个可数基，我们可以将其解释为 `u` 有一个收敛到 `x` 的子序列，这样我们就得到了度量空间中紧致性的样子。

```py
example  [FirstCountableTopology  X]  {s  :  Set  X}  {u  :  ℕ  →  X}  (hs  :  IsCompact  s)
  (hu  :  ∀  n,  u  n  ∈  s)  :  ∃  a  ∈  s,  ∃  φ  :  ℕ  →  ℕ,  StrictMono  φ  ∧  Tendsto  (u  ∘  φ)  atTop  (𝓝  a)  :=
  hs.tendsto_subseq  hu 
```

聚点与连续函数的行为良好。

```py
variable  [TopologicalSpace  Y]

example  {x  :  X}  {F  :  Filter  X}  {G  :  Filter  Y}  (H  :  ClusterPt  x  F)  {f  :  X  →  Y}
  (hfx  :  ContinuousAt  f  x)  (hf  :  Tendsto  f  F  G)  :  ClusterPt  (f  x)  G  :=
  ClusterPt.map  H  hfx  hf 
```

作为练习，我们将证明紧致集合在连续映射下的像是紧致的。除了我们已经看到的内容外，你还应该使用 `Filter.push_pull` 和 `NeBot.of_map`。

```py
example  [TopologicalSpace  Y]  {f  :  X  →  Y}  (hf  :  Continuous  f)  {s  :  Set  X}  (hs  :  IsCompact  s)  :
  IsCompact  (f  ''  s)  :=  by
  intro  F  F_ne  F_le
  have  map_eq  :  map  f  (𝓟  s  ⊓  comap  f  F)  =  𝓟  (f  ''  s)  ⊓  F  :=  by  sorry
  have  Hne  :  (𝓟  s  ⊓  comap  f  F).NeBot  :=  by  sorry
  have  Hle  :  𝓟  s  ⊓  comap  f  F  ≤  𝓟  s  :=  inf_le_left
  sorry 
```

也可以用开覆盖来表示紧致性：如果 `s` 是紧致的，那么覆盖 `s` 的每个开集族都有一个有限覆盖子族。

```py
example  {ι  :  Type*}  {s  :  Set  X}  (hs  :  IsCompact  s)  (U  :  ι  →  Set  X)  (hUo  :  ∀  i,  IsOpen  (U  i))
  (hsU  :  s  ⊆  ⋃  i,  U  i)  :  ∃  t  :  Finset  ι,  s  ⊆  ⋃  i  ∈  t,  U  i  :=
  hs.elim_finite_subcover  U  hUo  hsU 
```

### 11.3.1\. 基础

现在我们提高一般性，引入拓扑空间。我们将回顾定义拓扑空间的两种主要方法，然后解释拓扑空间范畴比度量空间范畴表现得更好。注意，我们在这里不会使用 Mathlib 范畴论，而只是有一个某种范畴的观点。

思考从度量空间到拓扑空间过渡的第一种方法是，我们只保留了开集的概念（或等价地，闭集的概念）。从这种观点来看，拓扑空间是一个类型，它配备了一个称为开集的集合。这个集合必须满足以下列出的若干公理（这个集合略微冗余，但我们将忽略这一点）。

```py
section
variable  {X  :  Type*}  [TopologicalSpace  X]

example  :  IsOpen  (univ  :  Set  X)  :=
  isOpen_univ

example  :  IsOpen  (∅  :  Set  X)  :=
  isOpen_empty

example  {ι  :  Type*}  {s  :  ι  →  Set  X}  (hs  :  ∀  i,  IsOpen  (s  i))  :  IsOpen  (⋃  i,  s  i)  :=
  isOpen_iUnion  hs

example  {ι  :  Type*}  [Fintype  ι]  {s  :  ι  →  Set  X}  (hs  :  ∀  i,  IsOpen  (s  i))  :
  IsOpen  (⋂  i,  s  i)  :=
  isOpen_iInter_of_finite  hs 
```

闭集定义为补集是开集的集合。在拓扑空间之间的函数（全局）连续，如果所有开集的前像都是开集。

```py
variable  {Y  :  Type*}  [TopologicalSpace  Y]

example  {f  :  X  →  Y}  :  Continuous  f  ↔  ∀  s,  IsOpen  s  →  IsOpen  (f  ⁻¹'  s)  :=
  continuous_def 
```

通过这个定义，我们已看到，与度量空间相比，拓扑空间只保留了足够的信息来讨论连续函数：一个类型上的两个拓扑结构相同，当且仅当它们具有相同的连续函数（实际上，如果两个结构具有相同的开集，那么恒等函数在这两个方向上将是连续的）。

然而，当我们转向点的连续性时，我们看到了基于开集的方法的局限性。在 Mathlib 中，我们经常将拓扑空间视为每个点`x`都附有一个邻域滤波器`𝓝 x`的类型（相应的函数`X → Filter X`满足以下条件，将在下面进一步解释）。记住从滤波器部分，这些小玩意儿扮演两个相关的角色。首先，`𝓝 x`被视为`X`中靠近`x`的点的广义集合。然后，它被视为为任何谓词`P : X → Prop`提供一种方式，即这个谓词对足够靠近`x`的点成立。让我们声明`f : X → Y`在`x`处是连续的。纯粹基于滤波器的方式是说，`f`的直接像包含靠近`x`的点的广义集合包含在靠近`f x`的点的广义集合中。回想一下，这可以表示为`map f (𝓝 x) ≤ 𝓝 (f x)`或`Tendsto f (𝓝 x) (𝓝 (f x))`。

```py
example  {f  :  X  →  Y}  {x  :  X}  :  ContinuousAt  f  x  ↔  map  f  (𝓝  x)  ≤  𝓝  (f  x)  :=
  Iff.rfl 
```

也可以使用两种邻域（视为普通集合）和邻域滤波器（视为广义集合）来表述：“对于`f x`的任何邻域`U`，所有靠近`x`的点都被发送到`U`”。请注意，证明仍然是`Iff.rfl`，这个观点在定义上是等价于前一个的。

```py
example  {f  :  X  →  Y}  {x  :  X}  :  ContinuousAt  f  x  ↔  ∀  U  ∈  𝓝  (f  x),  ∀ᶠ  x  in  𝓝  x,  f  x  ∈  U  :=
  Iff.rfl 
```

现在我们解释如何从一个观点转到另一个观点。从开集的角度来看，我们可以简单地定义`𝓝 x`的成员为包含`x`的开集的集合。

```py
example  {x  :  X}  {s  :  Set  X}  :  s  ∈  𝓝  x  ↔  ∃  t,  t  ⊆  s  ∧  IsOpen  t  ∧  x  ∈  t  :=
  mem_nhds_iff 
```

要朝相反的方向前进，我们需要讨论`𝓝 : X → Filter X`必须满足的条件，以便成为拓扑的邻域函数。

第一个约束是，`𝓝 x`（作为一个广义集合）包含集合 `{x}`（作为一个广义集合 `pure x`），另一种说法是，如果一个谓词对于接近 `x` 的点成立，那么它在 `x` 上也成立。

```py
example  (x  :  X)  :  pure  x  ≤  𝓝  x  :=
  pure_le_nhds  x

example  (x  :  X)  (P  :  X  →  Prop)  (h  :  ∀ᶠ  y  in  𝓝  x,  P  y)  :  P  x  :=
  h.self_of_nhds 
```

然后一个更微妙的要求是，对于任何谓词 `P : X → Prop` 和任何 `x`，如果 `P y` 对于接近 `x` 的 `y` 成立，那么对于接近 `x` 和 `y` 的 `z`，`P z` 也成立。更精确地说，我们有：

```py
example  {P  :  X  →  Prop}  {x  :  X}  (h  :  ∀ᶠ  y  in  𝓝  x,  P  y)  :  ∀ᶠ  y  in  𝓝  x,  ∀ᶠ  z  in  𝓝  y,  P  z  :=
  eventually_eventually_nhds.mpr  h 
```

这两个结果描述了 `X → Filter X` 的函数，这些函数是 `X` 上的拓扑空间结构的邻域函数。仍然有一个函数 `TopologicalSpace.mkOfNhds : (X → Filter X) → TopologicalSpace X`，但它只会将其输入作为邻域函数返回，如果它满足上述两个约束。更精确地说，我们有一个引理 `TopologicalSpace.nhds_mkOfNhds`，它以不同的方式表达，我们的下一个练习将从我们上述的方式推导出这一点。

```py
example  {α  :  Type*}  (n  :  α  →  Filter  α)  (H₀  :  ∀  a,  pure  a  ≤  n  a)
  (H  :  ∀  a  :  α,  ∀  p  :  α  →  Prop,  (∀ᶠ  x  in  n  a,  p  x)  →  ∀ᶠ  y  in  n  a,  ∀ᶠ  x  in  n  y,  p  x)  :
  ∀  a,  ∀  s  ∈  n  a,  ∃  t  ∈  n  a,  t  ⊆  s  ∧  ∀  a'  ∈  t,  s  ∈  n  a'  :=  by
  sorry
end 
```

注意，`TopologicalSpace.mkOfNhds` 并不经常使用，但了解它在拓扑空间结构中精确意义上意味着什么仍然是有益的。

为了有效地在 Mathlib 中使用拓扑空间，我们需要知道的是，我们使用了大量的 `TopologicalSpace : Type u → Type u` 的形式性质。从纯粹数学的角度来看，这些形式性质是解释拓扑空间如何解决度量空间存在的问题的一种非常干净的方式。从这个角度来看，拓扑空间解决的问题在于度量空间几乎不具有函子性，并且在一般上具有很差的范畴性质。这还基于已经讨论的事实，即度量空间包含大量与拓扑无关的几何信息。

让我们先关注函子性。度量空间结构可以诱导在子集上，或者等价地，可以通过一个注入映射将其拉回。但这几乎就是全部了。它们不能通过一般映射或通过全射映射推前。

尤其是在度量空间的商或不可数度量空间的积上，没有合理的距离可以放置。例如，考虑类型 `ℝ → ℝ`，它被视为由 `ℝ` 的副本组成的积，这些副本由 `ℝ` 索引。我们希望说函数序列逐点收敛是一个值得尊重的收敛概念。但在 `ℝ → ℝ` 上没有距离可以给出这种收敛概念。相关地，没有距离可以保证映射 `f : X → (ℝ → ℝ)` 连续当且仅当对于每个 `t : ℝ`，函数 `fun x ↦ f x t` 是连续的。

现在我们回顾一下解决所有这些问题的数据。首先，我们可以使用任何映射 `f : X → Y` 将拓扑从一个方面推到另一个方面。这两个操作形成一个高卢联系。

```py
variable  {X  Y  :  Type*}

example  (f  :  X  →  Y)  :  TopologicalSpace  X  →  TopologicalSpace  Y  :=
  TopologicalSpace.coinduced  f

example  (f  :  X  →  Y)  :  TopologicalSpace  Y  →  TopologicalSpace  X  :=
  TopologicalSpace.induced  f

example  (f  :  X  →  Y)  (T_X  :  TopologicalSpace  X)  (T_Y  :  TopologicalSpace  Y)  :
  TopologicalSpace.coinduced  f  T_X  ≤  T_Y  ↔  T_X  ≤  TopologicalSpace.induced  f  T_Y  :=
  coinduced_le_iff_le_induced 
```

这些操作与函数的复合是兼容的。像往常一样，推前是协变的，拉回是反对称的，参见`coinduced_compose`和`induced_compose`。在纸上我们将使用符号$f_*T$表示`TopologicalSpace.coinduced f T`和$f^*T$表示`TopologicalSpace.induced f T`。

然后下一个重要部分是对于任何给定的结构在`TopologicalSpace X`上的完备格结构。如果你认为拓扑主要是开集的数据，那么你期望`TopologicalSpace X`上的序关系来自`Set (Set X)`，即你期望如果集合`u`对于`t'`是开集，那么它对于`t`也是开集，那么`t ≤ t'`。然而，我们已经知道 Mathlib 更关注邻域而不是开集，所以对于任何`x : X`，我们希望从拓扑空间到邻域的映射`fun T : TopologicalSpace X ↦ @nhds X T x`是序保持的。我们知道`Filter X`上的序关系是为了确保序保持的`principal : Set X → Filter X`，允许将过滤器视为广义集合。所以我们使用的`TopologicalSpace X`上的序关系与来自`Set (Set X)`的序关系相反。

```py
example  {T  T'  :  TopologicalSpace  X}  :  T  ≤  T'  ↔  ∀  s,  T'.IsOpen  s  →  T.IsOpen  s  :=
  Iff.rfl 
```

现在我们可以通过结合推前（或拉回）操作与序关系来恢复连续性。

```py
example  (T_X  :  TopologicalSpace  X)  (T_Y  :  TopologicalSpace  Y)  (f  :  X  →  Y)  :
  Continuous  f  ↔  TopologicalSpace.coinduced  f  T_X  ≤  T_Y  :=
  continuous_iff_coinduced_le 
```

通过这个定义和推前与复合的兼容性，我们免费获得了一个普遍性质，即对于任何拓扑空间$Z$，如果函数$g : Y → Z$在拓扑$f_*T_X$上是连续的，当且仅当$g ∘ f$是连续的。

$$\begin{split}g \text{ 连续 } &⇔ g_*(f_*T_X) ≤ T_Z \\ &⇔ (g ∘ f)_* T_X ≤ T_Z \\ &⇔ g ∘ f \text{ 连续}\end{split}$$

```py
example  {Z  :  Type*}  (f  :  X  →  Y)  (T_X  :  TopologicalSpace  X)  (T_Z  :  TopologicalSpace  Z)
  (g  :  Y  →  Z)  :
  @Continuous  Y  Z  (TopologicalSpace.coinduced  f  T_X)  T_Z  g  ↔
  @Continuous  X  Z  T_X  T_Z  (g  ∘  f)  :=  by
  rw  [continuous_iff_coinduced_le,  coinduced_compose,  continuous_iff_coinduced_le] 
```

因此，我们已得到商拓扑（使用投影映射作为`f`）。这并不是使用`TopologicalSpace X`对所有`X`都是完备格的事实。现在让我们看看所有这些结构如何通过抽象的胡言乱语证明乘积拓扑的存在。我们上面考虑了`ℝ → ℝ`的情况，但现在让我们考虑一般情况`Π i, X i`对于某些`ι : Type*`和`X : ι → Type*`。我们希望对于任何拓扑空间`Z`和任何函数`f : Z → Π i, X i`，如果`f`是连续的，当且仅当对于所有`i`，`(fun x ↦ x i) ∘ f`是连续的。让我们使用表示投影`(fun (x : Π i, X i) ↦ x i)`的符号$p_i$来“在纸上”探索这个约束：

$$\begin{split}(∀ i, p_i ∘ f \text{ 连续}) &⇔ ∀ i, (p_i ∘ f)_* T_Z ≤ T_{X_i} \\ &⇔ ∀ i, (p_i)_* f_* T_Z ≤ T_{X_i}\\ &⇔ ∀ i, f_* T_Z ≤ (p_i)^*T_{X_i}\\ &⇔ f_* T_Z ≤ \inf \left[(p_i)^*T_{X_i}\right]\end{split}$$

因此，我们看到我们希望在`Π i, X i`上的拓扑是什么：

```py
example  (ι  :  Type*)  (X  :  ι  →  Type*)  (T_X  :  ∀  i,  TopologicalSpace  (X  i))  :
  (Pi.topologicalSpace  :  TopologicalSpace  (∀  i,  X  i))  =
  ⨅  i,  TopologicalSpace.induced  (fun  x  ↦  x  i)  (T_X  i)  :=
  rfl 
```

这就结束了我们对 Mathlib 如何认为拓扑空间通过成为一个更函子化的理论和为任何固定类型具有完备格结构来修复度量空间理论的缺陷的考察。

### 11.3.2\. 分隔与可数性

我们看到，拓扑空间类的性质非常好。为此付出的代价是存在相当病态的拓扑空间。你可以在拓扑空间上做出一些假设，以确保其行为更接近度量空间。其中最重要的是`T2Space`，也称为“Hausdorff”，这将确保极限是唯一的。更强的分离性质是`T3Space`，它还确保了正则空间性质：每个点都有一个闭邻域基。

```py
example  [TopologicalSpace  X]  [T2Space  X]  {u  :  ℕ  →  X}  {a  b  :  X}  (ha  :  Tendsto  u  atTop  (𝓝  a))
  (hb  :  Tendsto  u  atTop  (𝓝  b))  :  a  =  b  :=
  tendsto_nhds_unique  ha  hb

example  [TopologicalSpace  X]  [RegularSpace  X]  (a  :  X)  :
  (𝓝  a).HasBasis  (fun  s  :  Set  X  ↦  s  ∈  𝓝  a  ∧  IsClosed  s)  id  :=
  closed_nhds_basis  a 
```

注意，在每一个拓扑空间中，每个点都有一个开邻域基，这是定义所要求的。

```py
example  [TopologicalSpace  X]  {x  :  X}  :
  (𝓝  x).HasBasis  (fun  t  :  Set  X  ↦  t  ∈  𝓝  x  ∧  IsOpen  t)  id  :=
  nhds_basis_opens'  x 
```

我们现在的目标是证明一个基本定理，该定理允许通过连续性进行扩展。从 Bourbaki 的普通拓扑学书籍，I.8.5，定理 1（仅考虑非平凡蕴含）：

设$X$是一个拓扑空间，$A$是$X$的稠密子集，$f : A → Y$是$A$到$T_3$空间$Y$的连续映射。如果对于$X$中的每个$x$，当$y$在$A$中趋向于$x$时，$f(y)$在$Y$中趋向于一个极限，那么存在一个将$f$扩展到$X$的连续扩展$φ$。

实际上，Mathlib 包含上述引理的一个更一般的版本，即`IsDenseInducing.continuousAt_extend`，但在这里我们将坚持使用 Bourbaki 的版本。

记住，给定`A : Set X`，`↥A`是与`A`关联的子类型，并且当需要时 Lean 会自动插入那个有趣的向上箭头。而（包含）强制映射是`(↑) : A → X`。假设“在`A`中趋向于`x`的同时保持不变”对应于拉回过滤器`comap (↑) (𝓝 x)`。

让我们先证明一个辅助引理，将其提取出来以简化上下文（特别是我们在这里不需要 Y 是拓扑空间）。

```py
theorem  aux  {X  Y  A  :  Type*}  [TopologicalSpace  X]  {c  :  A  →  X}
  {f  :  A  →  Y}  {x  :  X}  {F  :  Filter  Y}
  (h  :  Tendsto  f  (comap  c  (𝓝  x))  F)  {V'  :  Set  Y}  (V'_in  :  V'  ∈  F)  :
  ∃  V  ∈  𝓝  x,  IsOpen  V  ∧  c  ⁻¹'  V  ⊆  f  ⁻¹'  V'  :=  by
  sorry 
```

现在我们转向连续扩展定理的主要证明。

当 Lean 需要一个在`↥A`上的拓扑时，它将自动使用诱导拓扑。唯一相关的引理是`nhds_induced (↑) : ∀ a : ↥A, 𝓝 a = comap (↑) (𝓝 ↑a)`（这实际上是一个关于诱导拓扑的通用引理）。

证明概要是：

主要假设和选择公理给出一个函数`φ`，使得`∀ x, Tendsto f (comap (↑) (𝓝 x)) (𝓝 (φ x))`（因为`Y`是 Hausdorff 的，`φ`完全确定，但我们不会在尝试证明`φ`确实扩展`f`之前需要它）。

让我们先证明`φ`是连续的。固定任意的`x : X`。由于`Y`是正则的，我们只需检查对于`φ x`的每一个闭邻域`V'`，`φ ⁻¹' V' ∈ 𝓝 x`。极限假设给出了（通过上面的辅助引理）一些`V ∈ 𝓝 x`，使得`IsOpen V ∧ (↑) ⁻¹' V ⊆ f ⁻¹' V'`。由于`V ∈ 𝓝 x`，我们只需证明`V ⊆ φ ⁻¹' V'`，即`∀ y ∈ V, φ y ∈ V'`。让我们固定`V`中的`y`。因为`V`是开集，它是`y`的一个邻域。特别是`(↑) ⁻¹' V ∈ comap (↑) (𝓝 y)`，并且更明显地`f ⁻¹' V' ∈ comap (↑) (𝓝 y)`。此外，由于`A`是稠密的，`comap (↑) (𝓝 y) ≠ ⊥`。因为我们知道`Tendsto f (comap (↑) (𝓝 y)) (𝓝 (φ y))`，这表明`φ y ∈ closure V'`，并且由于`V'`是闭集，我们证明了`φ y ∈ V'`。

剩下的工作是证明`φ`扩展了`f`。这是`f`的连续性和`Y`是豪斯多夫的事实进入讨论的地方。

```py
example  [TopologicalSpace  X]  [TopologicalSpace  Y]  [T3Space  Y]  {A  :  Set  X}
  (hA  :  ∀  x,  x  ∈  closure  A)  {f  :  A  →  Y}  (f_cont  :  Continuous  f)
  (hf  :  ∀  x  :  X,  ∃  c  :  Y,  Tendsto  f  (comap  (↑)  (𝓝  x))  (𝓝  c))  :
  ∃  φ  :  X  →  Y,  Continuous  φ  ∧  ∀  a  :  A,  φ  a  =  f  a  :=  by
  sorry

#check  HasBasis.tendsto_right_iff 
```

除了分离性质外，你可以对拓扑空间做出的主要假设是可数性假设，这有助于将其与度量空间联系起来。主要的一个是第一可数性，要求每个点都有一个可数的邻域基。特别是这确保了集合的闭包可以用序列来理解。

```py
example  [TopologicalSpace  X]  [FirstCountableTopology  X]
  {s  :  Set  X}  {a  :  X}  :
  a  ∈  closure  s  ↔  ∃  u  :  ℕ  →  X,  (∀  n,  u  n  ∈  s)  ∧  Tendsto  u  atTop  (𝓝  a)  :=
  mem_closure_iff_seq_limit 
```

### 11.3.3\. 紧致性

现在我们来讨论拓扑空间中紧致性的定义。通常有几种思考方式，Mathlib 选择了滤波版本。

我们首先需要定义滤波的聚点。给定一个拓扑空间`X`上的滤波`F`，如果点`x : X`是`F`（作为一个广义集合）与接近`x`的点的广义集合非空交集的点，那么`x`是`F`的聚点。

然后，我们可以说，如果`s`中的每一个非空广义集合`F`（即`F ≤ 𝓟 s`），在`s`中都有一个聚点，那么`s`是紧致的。

```py
variable  [TopologicalSpace  X]

example  {F  :  Filter  X}  {x  :  X}  :  ClusterPt  x  F  ↔  NeBot  (𝓝  x  ⊓  F)  :=
  Iff.rfl

example  {s  :  Set  X}  :
  IsCompact  s  ↔  ∀  (F  :  Filter  X)  [NeBot  F],  F  ≤  𝓟  s  →  ∃  a  ∈  s,  ClusterPt  a  F  :=
  Iff.rfl 
```

例如，如果`F`是`map u atTop`，即`u : ℕ → X`映射下的`atTop`（一个非常大的自然数的广义集合）的像，那么假设`F ≤ 𝓟 s`意味着对于足够大的`n`，`u n`属于`s`。说`x`是`map u atTop`的聚点意味着非常大的数的像与接近`x`的点的集合相交。如果`𝓝 x`有一个可数基，我们可以将其解释为`u`有一个子序列收敛到`x`，从而我们得到了紧致性在度量空间中的样子。

```py
example  [FirstCountableTopology  X]  {s  :  Set  X}  {u  :  ℕ  →  X}  (hs  :  IsCompact  s)
  (hu  :  ∀  n,  u  n  ∈  s)  :  ∃  a  ∈  s,  ∃  φ  :  ℕ  →  ℕ,  StrictMono  φ  ∧  Tendsto  (u  ∘  φ)  atTop  (𝓝  a)  :=
  hs.tendsto_subseq  hu 
```

聚点与连续函数的行为良好。

```py
variable  [TopologicalSpace  Y]

example  {x  :  X}  {F  :  Filter  X}  {G  :  Filter  Y}  (H  :  ClusterPt  x  F)  {f  :  X  →  Y}
  (hfx  :  ContinuousAt  f  x)  (hf  :  Tendsto  f  F  G)  :  ClusterPt  (f  x)  G  :=
  ClusterPt.map  H  hfx  hf 
```

作为练习，我们将证明紧集在连续映射下的像是紧致的。除了我们已经看到的，你应该使用`Filter.push_pull`和`NeBot.of_map`。

```py
example  [TopologicalSpace  Y]  {f  :  X  →  Y}  (hf  :  Continuous  f)  {s  :  Set  X}  (hs  :  IsCompact  s)  :
  IsCompact  (f  ''  s)  :=  by
  intro  F  F_ne  F_le
  have  map_eq  :  map  f  (𝓟  s  ⊓  comap  f  F)  =  𝓟  (f  ''  s)  ⊓  F  :=  by  sorry
  have  Hne  :  (𝓟  s  ⊓  comap  f  F).NeBot  :=  by  sorry
  have  Hle  :  𝓟  s  ⊓  comap  f  F  ≤  𝓟  s  :=  inf_le_left
  sorry 
```

一个人也可以用开覆盖来表示紧致性：如果覆盖`s`的每一个开集族都有一个有限覆盖子族，那么`s`是紧致的。

```py
example  {ι  :  Type*}  {s  :  Set  X}  (hs  :  IsCompact  s)  (U  :  ι  →  Set  X)  (hUo  :  ∀  i,  IsOpen  (U  i))
  (hsU  :  s  ⊆  ⋃  i,  U  i)  :  ∃  t  :  Finset  ι,  s  ⊆  ⋃  i  ∈  t,  U  i  :=
  hs.elim_finite_subcover  U  hUo  hsU 
```*
