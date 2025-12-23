# 10. 线性代数

> 原文：[`leanprover-community.github.io/mathematics_in_lean/C10_Linear_Algebra.html`](https://leanprover-community.github.io/mathematics_in_lean/C10_Linear_Algebra.html)

*Lean 中的数学* **   10. 线性代数

+   查看页面源代码

* * *

## 10.1. 向量空间和线性映射

### 10.1.1. 向量空间

我们将直接从任何域上的向量空间开始抽象线性代数。然而，你可以在第 10.4.1 节中找到有关矩阵的信息，这部分内容与抽象理论无关。Mathlib 实际上处理的是涉及“模块”一词的更一般版本的线性代数，但到目前为止，我们将假装这只是一个古怪的拼写习惯。

表达“设 $K$ 为一个域，设 $V$ 为 $K$ 上的向量空间”（并将它们作为后续结果的隐含参数）的方式是：

```py
variable  {K  :  Type*}  [Field  K]  {V  :  Type*}  [AddCommGroup  V]  [Module  K  V] 
```

我们在第八章中解释了为什么我们需要两个独立的类型类 `[AddCommGroup V] [Module K V]`。简而言之，从数学上讲，我们想要表达拥有 $K$ 向量空间结构意味着拥有加法交换群结构。我们可以将这一点告诉 Lean。但这样一来，每当 Lean 需要在类型 $V$ 上找到这样的群结构时，它就会使用一个*完全未指定的*域 $K$ 来寻找向量空间结构，而这个 $K$ 无法从 $V$ 中推断出来。这对类型类综合系统来说是非常糟糕的。

向量 $v$ 与标量 $a$ 的乘积表示为 a • v。以下列出了关于此操作与加法交互的几个代数规则。当然，simp 或 apply? 会找到这些证明。还有一个模块策略，它可以解决从向量空间和域公理得出的目标，就像在交换环中使用 ring 策略或在群中使用 group 策略一样。但记住标量乘法在引理名称中缩写为 smul 仍然是有用的。

```py
example  (a  :  K)  (u  v  :  V)  :  a  •  (u  +  v)  =  a  •  u  +  a  •  v  :=
  smul_add  a  u  v

example  (a  b  :  K)  (u  :  V)  :  (a  +  b)  •  u  =  a  •  u  +  b  •  u  :=
  add_smul  a  b  u

example  (a  b  :  K)  (u  :  V)  :  a  •  b  •  u  =  b  •  a  •  u  :=
  smul_comm  a  b  u 
```

作为对更高级读者的快速提示，让我们指出，正如术语所暗示的，Mathlib 的线性代数还涵盖了（不一定交换的）环上的模块。事实上，它甚至涵盖了半环上的半模。如果你认为你不需要这种程度的普遍性，你可以冥想以下示例，它很好地捕捉了关于理想在子模块上作用的许多代数规则：

```py
example  {R  M  :  Type*}  [CommSemiring  R]  [AddCommMonoid  M]  [Module  R  M]  :
  Module  (Ideal  R)  (Submodule  R  M)  :=
  inferInstance 
```

### 10.1.2. 线性映射

接下来我们需要线性映射。与群同态一样，Mathlib 中的线性映射是捆绑映射，即由映射及其线性性质证明组成的包。当应用这些捆绑映射时，它们会被转换为普通函数。有关此设计的更多信息，请参阅第八章。

两个`K`-向量空间`V`和`W`之间的线性映射类型表示为`V →ₗ[K] W`。下标 l 代表线性。一开始，指定`K`在这个符号中可能感觉有点奇怪。但这是几个字段同时出现时的关键。例如，从$ℂ$到$ℂ$的实线性映射是每个映射$z ↦ az + b\bar{z}$，而只有映射$z ↦ az$是复线性，这种差异在复分析中是至关重要的。

```py
variable  {W  :  Type*}  [AddCommGroup  W]  [Module  K  W]

variable  (φ  :  V  →ₗ[K]  W)

example  (a  :  K)  (v  :  V)  :  φ  (a  •  v)  =  a  •  φ  v  :=
  map_smul  φ  a  v

example  (v  w  :  V)  :  φ  (v  +  w)  =  φ  v  +  φ  w  :=
  map_add  φ  v  w 
```

注意，`V →ₗ[K] W`本身携带有趣的代数结构（这是捆绑那些映射的部分动机）。它是一个`K`-向量空间，因此我们可以对线性映射进行加法，并用标量乘以它们。

```py
variable  (ψ  :  V  →ₗ[K]  W)

#check  (2  •  φ  +  ψ  :  V  →ₗ[K]  W) 
```

使用捆绑映射的一个缺点是我们不能使用普通函数复合。我们需要使用`LinearMap.comp`或符号`∘ₗ`。

```py
variable  (θ  :  W  →ₗ[K]  V)

#check  (φ.comp  θ  :  W  →ₗ[K]  W)
#check  (φ  ∘ₗ  θ  :  W  →ₗ[K]  W) 
```

构造线性映射有两种主要方法。首先，我们可以通过提供函数和线性证明来构建结构。像往常一样，这是通过结构代码操作来实现的：你可以输入`example : V →ₗ[K] V := _`并使用附加在下划线上的代码操作“生成骨架”。

```py
example  :  V  →ₗ[K]  V  where
  toFun  v  :=  3  •  v
  map_add'  _  _  :=  smul_add  ..
  map_smul'  _  _  :=  smul_comm  .. 
```

你可能会好奇为什么`LinearMap`的证明域以一个撇号结尾。这是因为它们是在定义函数转换之前定义的，因此它们是以`LinearMap.toFun`为术语来表述的。然后，它们被重新表述为`LinearMap.map_add`和`LinearMap.map_smul`，以函数转换的术语来表述。但这还不是故事的结束。人们还希望有一个适用于任何（捆绑的）保持加法的映射的`map_add`版本，例如加法群同态、线性映射、连续线性映射、`K`-代数映射等。这个版本是根命名空间中的`map_add`。中间版本`LinearMap.map_add`有点冗余，但允许使用点符号，这在某些时候可能很方便。对于`map_smul`也有类似的故事，而通用框架在第八章中得到了解释。

```py
#check  (φ.map_add'  :  ∀  x  y  :  V,  φ.toFun  (x  +  y)  =  φ.toFun  x  +  φ.toFun  y)
#check  (φ.map_add  :  ∀  x  y  :  V,  φ  (x  +  y)  =  φ  x  +  φ  y)
#check  (map_add  φ  :  ∀  x  y  :  V,  φ  (x  +  y)  =  φ  x  +  φ  y) 
```

也可以使用 Mathlib 中已定义的各种组合子从已定义的线性映射构建线性映射。例如，上面的例子已经知道是`LinearMap.lsmul K V 3`。为什么`K`和`V`在这里是显式参数有几个原因。最紧迫的一个原因是，从裸`LinearMap.lsmul 3`中，Lean 无法推断出`V`甚至`K`。但`LinearMap.lsmul K V`本身也是一个有趣的对象：它具有类型`K →ₗ[K] V →ₗ[K] V`，这意味着它是一个从`K`（被视为其自身的向量空间）到从`V`到`V`的`K`-线性映射的`K`-线性映射。

```py
#check  (LinearMap.lsmul  K  V  3  :  V  →ₗ[K]  V)
#check  (LinearMap.lsmul  K  V  :  K  →ₗ[K]  V  →ₗ[K]  V) 
```

此外，还有一个表示为`V ≃ₗ[K] W`的线性同构类型`LinearEquiv`。`f : V ≃ₗ[K] W`的逆是`f.symm : W ≃ₗ[K] V`，`f`和`g`的复合是`f.trans g`，也记作`f ≪≫ₗ g`，而`V`的恒等同构是`LinearEquiv.refl K V`。当需要时，此类型中的元素会自动转换为形态和函数。

```py
example  (f  :  V  ≃ₗ[K]  W)  :  f  ≪≫ₗ  f.symm  =  LinearEquiv.refl  K  V  :=
  f.self_trans_symm 
```

可以使用 `LinearEquiv.ofBijective` 从双射态射构建同构。这样做会使逆函数不可计算。

```py
noncomputable  example  (f  :  V  →ₗ[K]  W)  (h  :  Function.Bijective  f)  :  V  ≃ₗ[K]  W  :=
  .ofBijective  f  h 
```

注意，在上面的例子中，Lean 使用宣布的类型来理解 `.ofBijective` 指的是 `LinearEquiv.ofBijective`（无需打开任何命名空间）。

### 10.1.3\. 向量空间的和与积

我们可以使用直接和与直接积从旧向量空间构建新的向量空间。让我们从两个向量空间开始。在这种情况下，和与积之间没有区别，我们可以简单地使用积类型。在下面的代码片段中，我们简单地展示了如何将所有结构映射（包含和投射）作为线性映射来获取，以及构建线性映射到积和从和的通用性质（如果您不熟悉关于和与积的范畴论区别，您可以简单地忽略通用性质词汇，并关注以下示例的类型）。

```py
section  binary_product

variable  {W  :  Type*}  [AddCommGroup  W]  [Module  K  W]
variable  {U  :  Type*}  [AddCommGroup  U]  [Module  K  U]
variable  {T  :  Type*}  [AddCommGroup  T]  [Module  K  T]

-- First projection map
example  :  V  ×  W  →ₗ[K]  V  :=  LinearMap.fst  K  V  W

-- Second projection map
example  :  V  ×  W  →ₗ[K]  W  :=  LinearMap.snd  K  V  W

-- Universal property of the product
example  (φ  :  U  →ₗ[K]  V)  (ψ  :  U  →ₗ[K]  W)  :  U  →ₗ[K]  V  ×  W  :=  LinearMap.prod  φ  ψ

-- The product map does the expected thing, first component
example  (φ  :  U  →ₗ[K]  V)  (ψ  :  U  →ₗ[K]  W)  :  LinearMap.fst  K  V  W  ∘ₗ  LinearMap.prod  φ  ψ  =  φ  :=  rfl

-- The product map does the expected thing, second component
example  (φ  :  U  →ₗ[K]  V)  (ψ  :  U  →ₗ[K]  W)  :  LinearMap.snd  K  V  W  ∘ₗ  LinearMap.prod  φ  ψ  =  ψ  :=  rfl

-- We can also combine maps in parallel
example  (φ  :  V  →ₗ[K]  U)  (ψ  :  W  →ₗ[K]  T)  :  (V  ×  W)  →ₗ[K]  (U  ×  T)  :=  φ.prodMap  ψ

-- This is simply done by combining the projections with the universal property
example  (φ  :  V  →ₗ[K]  U)  (ψ  :  W  →ₗ[K]  T)  :
  φ.prodMap  ψ  =  (φ  ∘ₗ  .fst  K  V  W).prod  (ψ  ∘ₗ  .snd  K  V  W)  :=  rfl

-- First inclusion map
example  :  V  →ₗ[K]  V  ×  W  :=  LinearMap.inl  K  V  W

-- Second inclusion map
example  :  W  →ₗ[K]  V  ×  W  :=  LinearMap.inr  K  V  W

-- Universal property of the sum (aka coproduct)
example  (φ  :  V  →ₗ[K]  U)  (ψ  :  W  →ₗ[K]  U)  :  V  ×  W  →ₗ[K]  U  :=  φ.coprod  ψ

-- The coproduct map does the expected thing, first component
example  (φ  :  V  →ₗ[K]  U)  (ψ  :  W  →ₗ[K]  U)  :  φ.coprod  ψ  ∘ₗ  LinearMap.inl  K  V  W  =  φ  :=
  LinearMap.coprod_inl  φ  ψ

-- The coproduct map does the expected thing, second component
example  (φ  :  V  →ₗ[K]  U)  (ψ  :  W  →ₗ[K]  U)  :  φ.coprod  ψ  ∘ₗ  LinearMap.inr  K  V  W  =  ψ  :=
  LinearMap.coprod_inr  φ  ψ

-- The coproduct map is defined in the expected way
example  (φ  :  V  →ₗ[K]  U)  (ψ  :  W  →ₗ[K]  U)  (v  :  V)  (w  :  W)  :
  φ.coprod  ψ  (v,  w)  =  φ  v  +  ψ  w  :=
  rfl

end  binary_product 
```

现在我们转向任意向量空间族的和与积。在这里，我们将简单地看看如何定义一个向量空间族并访问和与积的通用性质。请注意，直接和的符号范围限定在 `DirectSum` 命名空间内，并且直接和的通用性质要求索引类型的可判定等价性（这某种程度上是一个实现上的意外）。

```py
section  families
open  DirectSum

variable  {ι  :  Type*}  [DecidableEq  ι]
  (V  :  ι  →  Type*)  [∀  i,  AddCommGroup  (V  i)]  [∀  i,  Module  K  (V  i)]

-- The universal property of the direct sum assembles maps from the summands to build
-- a map from the direct sum
example  (φ  :  Π  i,  (V  i  →ₗ[K]  W))  :  (⨁  i,  V  i)  →ₗ[K]  W  :=
  DirectSum.toModule  K  ι  W  φ

-- The universal property of the direct product assembles maps into the factors
-- to build a map into the direct product
example  (φ  :  Π  i,  (W  →ₗ[K]  V  i))  :  W  →ₗ[K]  (Π  i,  V  i)  :=
  LinearMap.pi  φ

-- The projection maps from the product
example  (i  :  ι)  :  (Π  j,  V  j)  →ₗ[K]  V  i  :=  LinearMap.proj  i

-- The inclusion maps into the sum
example  (i  :  ι)  :  V  i  →ₗ[K]  (⨁  i,  V  i)  :=  DirectSum.lof  K  ι  V  i

-- The inclusion maps into the product
example  (i  :  ι)  :  V  i  →ₗ[K]  (Π  i,  V  i)  :=  LinearMap.single  K  V  i

-- In case `ι` is a finite type, there is an isomorphism between the sum and product.
example  [Fintype  ι]  :  (⨁  i,  V  i)  ≃ₗ[K]  (Π  i,  V  i)  :=
  linearEquivFunOnFintype  K  ι  V

end  families 
```

## 10.2\. 子空间与商

### 10.2.1\. 子空间

正如线性映射是打包的，`V` 的线性子空间也是一个打包的结构，由 `V` 中的一个集合组成，称为子空间的载体，具有相关的闭包性质。由于 Mathlib 实际上用于线性代数的更一般上下文，所以这里出现的是“模块”一词而不是向量空间。 

```py
variable  {K  :  Type*}  [Field  K]  {V  :  Type*}  [AddCommGroup  V]  [Module  K  V]

example  (U  :  Submodule  K  V)  {x  y  :  V}  (hx  :  x  ∈  U)  (hy  :  y  ∈  U)  :
  x  +  y  ∈  U  :=
  U.add_mem  hx  hy

example  (U  :  Submodule  K  V)  {x  :  V}  (hx  :  x  ∈  U)  (a  :  K)  :
  a  •  x  ∈  U  :=
  U.smul_mem  a  hx 
```

在上面的例子中，重要的是要理解 `Submodule K V` 是 `V` 的 `K`-线性子空间的类型，而不是一个 `IsSubmodule U` 的谓词，其中 `U` 是 `Set V` 的一个元素。`Submodule K V` 被赋予了到 `Set V` 的强制转换和 `V` 上的成员谓词。有关如何以及为什么这样做，请参见 第 8.3 节 的解释。

当然，如果两个子空间具有相同的元素，则它们是相同的。这一事实被注册用于与 `ext` 策略一起使用，该策略可以用来证明两个子空间相等，就像它被用来证明两个集合相等一样。

例如，为了陈述和证明 `ℝ` 是 `ℂ` 的 `ℝ`-线性子空间，我们真正想要的是构造一个类型为 `Submodule ℝ ℂ` 的项，其投影到 `Set ℂ` 是 `ℝ`，或者更精确地说，是 `ℝ` 在 `ℂ` 中的像。

```py
noncomputable  example  :  Submodule  ℝ  ℂ  where
  carrier  :=  Set.range  ((↑)  :  ℝ  →  ℂ)
  add_mem'  :=  by
  rintro  _  _  ⟨n,  rfl⟩  ⟨m,  rfl⟩
  use  n  +  m
  simp
  zero_mem'  :=  by
  use  0
  simp
  smul_mem'  :=  by
  rintro  c  -  ⟨a,  rfl⟩
  use  c*a
  simp 
```

`Submodule` 中证明字段末尾的素数与 `LinearMap` 中的素数类似。这些字段是以 `carrier` 字段为依据声明的，因为它们是在 `MemberShip` 实例之前定义的。然后，它们被上面提到的 `Submodule.add_mem`、`Submodule.zero_mem` 和 `Submodule.smul_mem` 所取代。

作为操作子空间和线性映射的练习，你将定义由线性映射（当然，我们将在下面看到 Mathlib 已经知道这一点）生成的子空间的逆像。记住，可以使用 `Set.mem_preimage` 重写涉及成员资格和逆像的陈述。这是除了上面讨论的关于 `LinearMap` 和 `Submodule` 的引理之外，你需要的唯一引理。

```py
def  preimage  {W  :  Type*}  [AddCommGroup  W]  [Module  K  W]  (φ  :  V  →ₗ[K]  W)  (H  :  Submodule  K  W)  :
  Submodule  K  V  where
  carrier  :=  φ  ⁻¹'  H
  zero_mem'  :=  by
  sorry
  add_mem'  :=  by
  sorry
  smul_mem'  :=  by
  sorry 
```

使用类型类，Mathlib 知道向量空间的子空间继承了向量空间的结构。

```py
example  (U  :  Submodule  K  V)  :  Module  K  U  :=  inferInstance 
```

这个例子很微妙。对象 `U` 不是一个类型，但 Lean 会自动将其解释为 `V` 的子类型，从而将其转换为类型。因此，上述例子可以更明确地表述为：

```py
example  (U  :  Submodule  K  V)  :  Module  K  {x  :  V  //  x  ∈  U}  :=  inferInstance 
```

### 10.2.2\. 完备格结构和内部直和

拥有类型 `Submodule K V` 而不是谓词 `IsSubmodule : Set V → Prop` 的重要好处是，可以轻松地为 `Submodule K V` 赋予额外的结构。重要的是，它具有关于包含的完备格结构。例如，我们不是用一条定理来声明 `V` 的两个子空间的交集仍然是子空间，而是使用格运算 `⊓` 来构造交集。然后我们可以将关于格的任意引理应用于构造。

让我们验证两个子空间下确界的底层集合确实按照定义是它们的交集。

```py
example  (H  H'  :  Submodule  K  V)  :
  ((H  ⊓  H'  :  Submodule  K  V)  :  Set  V)  =  (H  :  Set  V)  ∩  (H'  :  Set  V)  :=  rfl 
```

对于底层集合的交集使用不同的符号可能看起来很奇怪，但这种对应关系并不适用于上确界运算和集合的并集，因为子空间的并集在一般情况下不是子空间。相反，需要使用由并集生成的子空间，这是通过 `Submodule.span` 实现的。

```py
example  (H  H'  :  Submodule  K  V)  :
  ((H  ⊔  H'  :  Submodule  K  V)  :  Set  V)  =  Submodule.span  K  ((H  :  Set  V)  ∪  (H'  :  Set  V))  :=  by
  simp  [Submodule.span_union] 
```

另一个微妙之处在于，`V` 本身没有类型 `Submodule K V`，因此我们需要一种方式来谈论 `V` 作为 `V` 的子空间。这也由格结构提供：全子空间是这个格的顶元素。

```py
example  (x  :  V)  :  x  ∈  (⊤  :  Submodule  K  V)  :=  trivial 
```

同样，这个格的底元素是只包含零元素的子空间。

```py
example  (x  :  V)  :  x  ∈  (⊥  :  Submodule  K  V)  ↔  x  =  0  :=  Submodule.mem_bot  K 
```

尤其是我们可以讨论（内部）直和的子空间的情况。在两个子空间的情况下，我们使用适用于任何有界偏序类型的通用谓词 `IsCompl`。在一般子空间族的情况下，我们使用 `DirectSum.IsInternal`。

```py
-- If two subspaces are in direct sum then they span the whole space.
example  (U  V  :  Submodule  K  V)  (h  :  IsCompl  U  V)  :
  U  ⊔  V  =  ⊤  :=  h.sup_eq_top

-- If two subspaces are in direct sum then they intersect only at zero.
example  (U  V  :  Submodule  K  V)  (h  :  IsCompl  U  V)  :
  U  ⊓  V  =  ⊥  :=  h.inf_eq_bot

section
open  DirectSum
variable  {ι  :  Type*}  [DecidableEq  ι]

-- If subspaces are in direct sum then they span the whole space.
example  (U  :  ι  →  Submodule  K  V)  (h  :  DirectSum.IsInternal  U)  :
  ⨆  i,  U  i  =  ⊤  :=  h.submodule_iSup_eq_top

-- If subspaces are in direct sum then they pairwise intersect only at zero.
example  {ι  :  Type*}  [DecidableEq  ι]  (U  :  ι  →  Submodule  K  V)  (h  :  DirectSum.IsInternal  U)
  {i  j  :  ι}  (hij  :  i  ≠  j)  :  U  i  ⊓  U  j  =  ⊥  :=
  (h.submodule_iSupIndep.pairwiseDisjoint  hij).eq_bot

-- Those conditions characterize direct sums.
#check  DirectSum.isInternal_submodule_iff_independent_and_iSup_eq_top

-- The relation with external direct sums: if a family of subspaces is
-- in internal direct sum then the map from their external direct sum into `V`
-- is a linear isomorphism.
noncomputable  example  {ι  :  Type*}  [DecidableEq  ι]  (U  :  ι  →  Submodule  K  V)
  (h  :  DirectSum.IsInternal  U)  :  (⨁  i,  U  i)  ≃ₗ[K]  V  :=
  LinearEquiv.ofBijective  (coeLinearMap  U)  h
end 
```

### 10.2.3\. 由一组生成的子空间

除了从现有子空间构建子空间之外，我们还可以使用 `Submodule.span K s` 从任何集合 `s` 中构建它们，该函数构建包含 `s` 的最小子空间。在纸上，通常使用这种空间由 `s` 的所有线性组合元素构成。但通常更有效的是使用其通过 `Submodule.span_le` 表达的泛性质，以及伽罗瓦连接的整个理论。

```py
example  {s  :  Set  V}  (E  :  Submodule  K  V)  :  Submodule.span  K  s  ≤  E  ↔  s  ⊆  E  :=
  Submodule.span_le

example  :  GaloisInsertion  (Submodule.span  K)  ((↑)  :  Submodule  K  V  →  Set  V)  :=
  Submodule.gi  K  V 
```

当这些还不够时，可以使用相关的归纳原理 `Submodule.span_induction`，该原理确保只要在 `zero` 和 `s` 的元素上成立，并且对加法和标量乘法稳定，那么 `s` 的张量中的每个元素都保持该性质。

作为练习，让我们重新证明 `Submodule.mem_sup` 的一个蕴含。记住，你可以使用模块策略来关闭由与 `V` 上的各种代数操作相关的公理得出的目标。

```py
example  {S  T  :  Submodule  K  V}  {x  :  V}  (h  :  x  ∈  S  ⊔  T)  :
  ∃  s  ∈  S,  ∃  t  ∈  T,  x  =  s  +  t  :=  by
  rw  [←  S.span_eq,  ←  T.span_eq,  ←  Submodule.span_union]  at  h
  induction  h  using  Submodule.span_induction  with
  |  mem  y  h  =>
  sorry
  |  zero  =>
  sorry
  |  add  x  y  hx  hy  hx'  hy'  =>
  sorry
  |  smul  a  x  hx  hx'  =>
  sorry 
```

### 10.2.4. 推拉子空间

如前所述，我们现在描述如何通过线性映射推拉子空间。在 Mathlib 中，第一个操作称为 `map`，第二个操作称为 `comap`。

```py
section

variable  {W  :  Type*}  [AddCommGroup  W]  [Module  K  W]  (φ  :  V  →ₗ[K]  W)

variable  (E  :  Submodule  K  V)  in
#check  (Submodule.map  φ  E  :  Submodule  K  W)

variable  (F  :  Submodule  K  W)  in
#check  (Submodule.comap  φ  F  :  Submodule  K  V) 
```

注意这些位于 `Submodule` 命名空间中，因此可以使用点符号，并编写 `E.map φ` 而不是 `Submodule.map φ E`，但这很不容易阅读（尽管一些 Mathlib 贡献者使用这种拼写）。

特别地，线性映射的范围和核是子空间。这些特殊情况很重要，足以得到声明。

```py
example  :  LinearMap.range  φ  =  .map  φ  ⊤  :=  LinearMap.range_eq_map  φ

example  :  LinearMap.ker  φ  =  .comap  φ  ⊥  :=  Submodule.comap_bot  φ  -- or `rfl` 
```

注意，我们不能写 `φ.ker` 而不是 `LinearMap.ker φ`，因为 `LinearMap.ker` 也适用于保持更多结构的映射类，因此它不期望一个以 `LinearMap` 开头的类型的参数，因此点符号在这里不起作用。然而，我们能够在右侧使用另一种点符号。因为 Lean 在展开左侧后期望一个类型为 `Submodule K V` 的项，它将 `.comap` 解释为 `Submodule.comap`。

以下引理给出了这些子模块与 `φ` 的性质之间的关键关系。

```py
open  Function  LinearMap

example  :  Injective  φ  ↔  ker  φ  =  ⊥  :=  ker_eq_bot.symm

example  :  Surjective  φ  ↔  range  φ  =  ⊤  :=  range_eq_top.symm 
```

作为练习，让我们证明 `map` 和 `comap` 的伽罗瓦连接性质。可以使用以下引理，但这不是必需的，因为它们根据定义是正确的。

```py
#check  Submodule.mem_map_of_mem
#check  Submodule.mem_map
#check  Submodule.mem_comap

example  (E  :  Submodule  K  V)  (F  :  Submodule  K  W)  :
  Submodule.map  φ  E  ≤  F  ↔  E  ≤  Submodule.comap  φ  F  :=  by
  sorry 
```

### 10.2.5. 商空间

商向量空间使用通用商符号（使用 `\quot` 打印，而不是普通 `/`）。商空间的投影是 `Submodule.mkQ`，其泛性质是 `Submodule.liftQ`。

```py
variable  (E  :  Submodule  K  V)

example  :  Module  K  (V  ⧸  E)  :=  inferInstance

example  :  V  →ₗ[K]  V  ⧸  E  :=  E.mkQ

example  :  ker  E.mkQ  =  E  :=  E.ker_mkQ

example  :  range  E.mkQ  =  ⊤  :=  E.range_mkQ

example  (hφ  :  E  ≤  ker  φ)  :  V  ⧸  E  →ₗ[K]  W  :=  E.liftQ  φ  hφ

example  (F  :  Submodule  K  W)  (hφ  :  E  ≤  .comap  φ  F)  :  V  ⧸  E  →ₗ[K]  W  ⧸  F  :=  E.mapQ  F  φ  hφ

noncomputable  example  :  (V  ⧸  LinearMap.ker  φ)  ≃ₗ[K]  range  φ  :=  φ.quotKerEquivRange 
```

作为练习，让我们证明商空间子空间的对应定理。Mathlib 知道一个稍微更精确的版本，称为 `Submodule.comapMkQRelIso`。

```py
open  Submodule

#check  Submodule.map_comap_eq
#check  Submodule.comap_map_eq

example  :  Submodule  K  (V  ⧸  E)  ≃  {  F  :  Submodule  K  V  //  E  ≤  F  }  where
  toFun  :=  sorry
  invFun  :=  sorry
  left_inv  :=  sorry
  right_inv  :=  sorry 
```

## 10.3. 内射

线性映射的一个重要特殊情况是内射映射：从向量空间到自身的线性映射。它们很有趣，因为它们构成一个`K`-代数。特别是，我们可以对它们上的系数在`K`中的多项式进行评估，并且它们可以有特征值和特征向量。

Mathlib 使用缩写`Module.End K V := V →ₗ[K] V`，这在使用大量此类映射时很方便（尤其是在打开`Module`命名空间之后）。

```py
variable  {K  :  Type*}  [Field  K]  {V  :  Type*}  [AddCommGroup  V]  [Module  K  V]

variable  {W  :  Type*}  [AddCommGroup  W]  [Module  K  W]

open  Polynomial  Module  LinearMap  End

example  (φ  ψ  :  End  K  V)  :  φ  *  ψ  =  φ  ∘ₗ  ψ  :=
  End.mul_eq_comp  φ  ψ  -- `rfl` would also work

-- evaluating `P` on `φ`
example  (P  :  K[X])  (φ  :  End  K  V)  :  V  →ₗ[K]  V  :=
  aeval  φ  P

-- evaluating `X` on `φ` gives back `φ`
example  (φ  :  End  K  V)  :  aeval  φ  (X  :  K[X])  =  φ  :=
  aeval_X  φ 
```

作为操作内射映射、子空间和多项式的练习，让我们证明（二元的）核引理：对于任何内射映射$φ$和任何两个互质的二次多项式$P$和$Q$，我们有$\ker P(φ) ⊕ \ker Q(φ) = \ker \big(PQ(φ)\big)$。

注意，`IsCoprime x y`被定义为`∃ a b, a * x + b * y = 1`。

```py
#check  Submodule.eq_bot_iff
#check  Submodule.mem_inf
#check  LinearMap.mem_ker

example  (P  Q  :  K[X])  (h  :  IsCoprime  P  Q)  (φ  :  End  K  V)  :  ker  (aeval  φ  P)  ⊓  ker  (aeval  φ  Q)  =  ⊥  :=  by
  sorry

#check  Submodule.add_mem_sup
#check  map_mul
#check  End.mul_apply
#check  LinearMap.ker_le_ker_comp

example  (P  Q  :  K[X])  (h  :  IsCoprime  P  Q)  (φ  :  End  K  V)  :
  ker  (aeval  φ  P)  ⊔  ker  (aeval  φ  Q)  =  ker  (aeval  φ  (P*Q))  :=  by
  sorry 
```

我们现在转向对特征空间和特征值的讨论。与内射映射$φ$和标量$a$相关的特征空间是$φ - aId$的核。特征空间对所有`a`的值都有定义，尽管它们只有在非零时才有趣。然而，根据定义，特征向量是特征空间中的非零元素。相应的谓词是`End.HasEigenvector`。

```py
example  (φ  :  End  K  V)  (a  :  K)  :  φ.eigenspace  a  =  LinearMap.ker  (φ  -  a  •  1)  :=
  End.eigenspace_def 
```

然后有一个谓词`End.HasEigenvalue`和相应的子类型`End.Eigenvalues`。

```py
example  (φ  :  End  K  V)  (a  :  K)  :  φ.HasEigenvalue  a  ↔  φ.eigenspace  a  ≠  ⊥  :=
  Iff.rfl

example  (φ  :  End  K  V)  (a  :  K)  :  φ.HasEigenvalue  a  ↔  ∃  v,  φ.HasEigenvector  a  v  :=
  ⟨End.HasEigenvalue.exists_hasEigenvector,  fun  ⟨_,  hv⟩  ↦  φ.hasEigenvalue_of_hasEigenvector  hv⟩

example  (φ  :  End  K  V)  :  φ.Eigenvalues  =  {a  //  φ.HasEigenvalue  a}  :=
  rfl

-- Eigenvalue are roots of the minimal polynomial
example  (φ  :  End  K  V)  (a  :  K)  :  φ.HasEigenvalue  a  →  (minpoly  K  φ).IsRoot  a  :=
  φ.isRoot_of_hasEigenvalue

-- In finite dimension, the converse is also true (we will discuss dimension below)
example  [FiniteDimensional  K  V]  (φ  :  End  K  V)  (a  :  K)  :
  φ.HasEigenvalue  a  ↔  (minpoly  K  φ).IsRoot  a  :=
  φ.hasEigenvalue_iff_isRoot

-- Cayley-Hamilton
example  [FiniteDimensional  K  V]  (φ  :  End  K  V)  :  aeval  φ  φ.charpoly  =  0  :=
  φ.aeval_self_charpoly 
```

## 10.4. 矩阵、基和维度

### 10.4.1. 矩阵

在介绍抽象向量空间的基之前，我们回到更基础的线性代数设置，即对于某个域$K$的$K^n$。在这里，主要对象是向量和矩阵。对于具体的向量，可以使用`![…]`表示法，其中分量由逗号分隔。对于具体的矩阵，我们可以使用`!![…]`表示法，行由分号分隔，行的分量由冒号分隔。当条目具有可计算的类型，如`ℕ`或`ℚ`时，我们可以使用`eval`命令进行基本操作。

```py
section  matrices

-- Adding vectors
#eval  ![1,  2]  +  ![3,  4]  -- ![4, 6]

-- Adding matrices
#eval  !![1,  2;  3,  4]  +  !![3,  4;  5,  6]  -- !![4, 6; 8, 10]

-- Multiplying matrices
#eval  !![1,  2;  3,  4]  *  !![3,  4;  5,  6]  -- !![13, 16; 29, 36] 
```

重要的是要理解，这种`#eval`的使用仅对探索有意义，它并不打算取代 Sage 这样的计算机代数系统。这里用于矩阵的数据表示在计算上并不高效。它使用函数而不是数组，并且优化用于证明而不是计算。`#eval`使用的虚拟机也没有为此用途进行优化。

当心矩阵表示法列表行，但向量表示法既不是行向量也不是列向量。从左（分别）乘以矩阵的向量将向量解释为行（分别）向量。这对应于`Matrix.vecMul`操作，符号为`ᵥ*`，以及`Matrix.mulVec`操作，符号为` *ᵥ`。这些符号在`Matrix`命名空间中定义，因此我们需要打开该命名空间。

```py
open  Matrix

-- matrices acting on vectors on the left
#eval  !![1,  2;  3,  4]  *ᵥ  ![1,  1]  -- ![3, 7]

-- matrices acting on vectors on the left, resulting in a size one matrix
#eval  !![1,  2]  *ᵥ  ![1,  1]  -- ![3]

-- matrices acting on vectors on the right
#eval  ![1,  1,  1]  ᵥ*  !![1,  2;  3,  4;  5,  6]  -- ![9, 12] 
```

为了生成具有由向量指定的相同行或列的矩阵，我们使用 `Matrix.replicateRow` 和 `Matrix.replicateCol`，其中参数是索引行或列的类型和向量。例如，可以得到单行或单列矩阵（更精确地说，行或列由 `Fin 1` 索引的矩阵）。

```py
#eval  replicateRow  (Fin  1)  ![1,  2]  -- !![1, 2]

#eval  replicateCol  (Fin  1)  ![1,  2]  -- !![1; 2] 
```

其他熟悉的操作包括向量点积、矩阵转置，对于方阵，还有行列式和迹。

```py
-- vector dot product
#eval  ![1,  2]  ⬝ᵥ  ![3,  4]  -- `11`

-- matrix transpose
#eval  !![1,  2;  3,  4]ᵀ  -- `!![1, 3; 2, 4]`

-- determinant
#eval  !![(1  :  ℤ),  2;  3,  4].det  -- `-2`

-- trace
#eval  !![(1  :  ℤ),  2;  3,  4].trace  -- `5` 
```

当条目没有可计算的类型时，例如如果它们是实数，我们无法期望 `#eval` 能有所帮助。此外，这种评估不能在没有显著扩大可信代码库（即检查证明时需要信任的 Lean 的部分）的情况下用于证明。

因此，在证明中也使用 `simp` 和 `norm_num` 策略，或者它们的命令对应物进行快速探索是很好的。

```py
#simp  !![(1  :  ℝ),  2;  3,  4].det  -- `4 - 2*3`

#norm_num  !![(1  :  ℝ),  2;  3,  4].det  -- `-2`

#norm_num  !![(1  :  ℝ),  2;  3,  4].trace  -- `5`

variable  (a  b  c  d  :  ℝ)  in
#simp  !![a,  b;  c,  d].det  -- `a * d – b * c` 
```

方阵上的下一个重要操作是求逆。与数的除法总是定义并返回除以零的人工值零一样，求逆操作在所有矩阵上定义，并在不可逆矩阵上返回零矩阵。

更精确地说，存在一个通用的函数 `Ring.inverse`，它在任何环中执行此操作，并且对于任何矩阵 `A`，`A⁻¹` 被定义为 `Ring.inverse A.det • A.adjugate`。根据克莱姆法则，当 `A` 的行列式不为零时，这确实是 `A` 的逆。

```py
#norm_num  [Matrix.inv_def]  !![(1  :  ℝ),  2;  3,  4]⁻¹  -- !![-2, 1; 3 / 2, -(1 / 2)] 
```

当然，这个定义对于可逆矩阵来说非常有用。存在一个通用的类型类 `Invertible`，它有助于记录这一点。例如，下一个示例中的 `simp` 调用将使用具有 `Invertible` 类型类假设的 `inv_mul_of_invertible` 公理，因此只有在类型类合成系统可以找到它时才会触发。在这里，我们使用 `have` 语句使这一事实可用。

```py
example  :  !![(1  :  ℝ),  2;  3,  4]⁻¹  *  !![(1  :  ℝ),  2;  3,  4]  =  1  :=  by
  have  :  Invertible  !![(1  :  ℝ),  2;  3,  4]  :=  by
  apply  Matrix.invertibleOfIsUnitDet
  norm_num
  simp 
```

在这个完全具体的情况下，我们也可以使用 `norm_num` 机制和 `apply?` 来找到最终的行：

```py
example  :  !![(1  :  ℝ),  2;  3,  4]⁻¹  *  !![(1  :  ℝ),  2;  3,  4]  =  1  :=  by
  norm_num  [Matrix.inv_def]
  exact  one_fin_two.symm 
```

所有的具体矩阵都有它们的行和列由 `Fin n` 索引，对于某个 `n`（行和列不一定相同）。但有时使用任意有限类型索引矩阵更方便。例如，有限图的邻接矩阵的行和列自然由图的顶点索引。

事实上，当仅仅想要定义矩阵而不在它们上定义任何操作时，索引类型的有限性甚至都不需要，系数可以有任何类型，而不需要任何代数结构。因此，Mathlib 简单地将 `Matrix m n α` 定义为 `m → n → α`，对于任何类型 `m`、`n` 和 `α`，而我们迄今为止所使用的矩阵类型如 `Matrix (Fin 2) (Fin 2) ℝ`。当然，代数操作需要对 `m`、`n` 和 `α` 有更多的假设。

注意我们不直接使用`m → n → α`的主要原因是为了让类型类系统理解我们的意图。例如，对于一个环`R`，类型`n → R`被赋予了点乘操作，同样`m → n → R`也有这种操作，但这不是我们在矩阵上想要的乘法。

在下面的第一个例子中，我们迫使 Lean 看穿`Matrix`的定义，并接受这个陈述是有意义的，然后通过检查所有项来证明它。

但接下来的两个例子揭示了 Lean 在`Fin 2 → Fin 2 → ℤ`上使用点乘，而在`Matrix (Fin 2) (Fin 2) ℤ`上使用矩阵乘法。

```py
section

example  :  (fun  _  ↦  1  :  Fin  2  →  Fin  2  →  ℤ)  =  !![1,  1;  1,  1]  :=  by
  ext  i  j
  fin_cases  i  <;>  fin_cases  j  <;>  rfl

example  :  (fun  _  ↦  1  :  Fin  2  →  Fin  2  →  ℤ)  *  (fun  _  ↦  1  :  Fin  2  →  Fin  2  →  ℤ)  =  !![1,  1;  1,  1]  :=  by
  ext  i  j
  fin_cases  i  <;>  fin_cases  j  <;>  rfl

example  :  !![1,  1;  1,  1]  *  !![1,  1;  1,  1]  =  !![2,  2;  2,  2]  :=  by
  norm_num 
```

为了将矩阵定义为函数而不失去`Matrix`在类型类合成中的优势，我们可以使用函数和矩阵之间的等价性`Matrix.of`。这种等价性实际上是通过`Equiv.refl`秘密定义的。

例如，我们可以定义与向量`v`对应的 Vandermonde 矩阵。

```py
example  {n  :  ℕ}  (v  :  Fin  n  →  ℝ)  :
  Matrix.vandermonde  v  =  Matrix.of  (fun  i  j  :  Fin  n  ↦  v  i  ^  (j  :  ℕ))  :=
  rfl
end
end  matrices 
```

### 10.4.2. 基

现在我们想讨论向量空间的基。非正式地说，有许多定义这种概念的方法。可以使用一个泛性质。可以说基是一组线性无关且张成的向量。或者可以将这些性质结合起来，直接说基是一组向量，每个向量都可以唯一地表示为基向量的线性组合。另一种说法是，基提供了一个与基域`K`的幂的线性同构，其中`K`被视为`K`上的向量空间。

这个同构版本实际上是 Mathlib 在底层用作定义的版本，其他特征化都是从这个证明的。在无限基的情况下，必须对“`K`的幂”这个想法稍加小心。确实，在这个代数背景下，只有有限线性组合才有意义。因此，作为参考向量空间，我们需要的不是`K`的副本的直接积，而是一个直接和。我们可以使用`⨁ i : ι, K`对于某些索引基的类型`ι`，但我们更倾向于使用更专业的拼写`ι →₀ K`，这意味着“从`ι`到`K`的有限支持函数”，即函数在`ι`的有限集合外为零（这个有限集合不是固定的，它依赖于函数）。将来自基`B`的这样一个函数在向量`v`和`ι`上求值，返回`v`在`ι`-th 基向量上的分量（或坐标）。

由`V`中类型`ι`的`K`向量空间基的索引类型是`Basis ι K V`。这种同构称为`Basis.repr`。

```py
variable  {K  :  Type*}  [Field  K]  {V  :  Type*}  [AddCommGroup  V]  [Module  K  V]

section

variable  {ι  :  Type*}  (B  :  Basis  ι  K  V)  (v  :  V)  (i  :  ι)

-- The basis vector with index ``i``
#check  (B  i  :  V)

-- the linear isomorphism with the model space given by ``B``
#check  (B.repr  :  V  ≃ₗ[K]  ι  →₀  K)

-- the component function of ``v``
#check  (B.repr  v  :  ι  →₀  K)

-- the component of ``v`` with index ``i``
#check  (B.repr  v  i  :  K) 
```

而不是从一个这样的同构开始，可以开始于一个线性无关且张成的向量族`b`，这是`Basis.mk`。

假设家庭是跨越的，这一假设被表述为 `⊤ ≤ Submodule.span K (Set.range b)`。这里的 `⊤` 是 `V` 的上子模块，即 `V` 作为自身的子模块来看待。这种表述看起来有点复杂，但下面我们将看到，它几乎在定义上等同于更易读的 `∀ v, v ∈ Submodule.span K (Set.range b)`（下面片段中的下划线指的是无用的信息 `v ∈ ⊤`）。

```py
noncomputable  example  (b  :  ι  →  V)  (b_indep  :  LinearIndependent  K  b)
  (b_spans  :  ∀  v,  v  ∈  Submodule.span  K  (Set.range  b))  :  Basis  ι  K  V  :=
  Basis.mk  b_indep  (fun  v  _  ↦  b_spans  v)

-- The family of vectors underlying the above basis is indeed ``b``.
example  (b  :  ι  →  V)  (b_indep  :  LinearIndependent  K  b)
  (b_spans  :  ∀  v,  v  ∈  Submodule.span  K  (Set.range  b))  (i  :  ι)  :
  Basis.mk  b_indep  (fun  v  _  ↦  b_spans  v)  i  =  b  i  :=
  Basis.mk_apply  b_indep  (fun  v  _  ↦  b_spans  v)  i 
```

尤其是模型向量空间 `ι →₀ K` 有一个所谓的规范基，其 `repr` 函数在任意向量上的评估是恒等同构。它被称为 `Finsupp.basisSingleOne`，其中 `Finsupp` 表示有限支撑的函数，而 `basisSingleOne` 指的是基向量是除了单个输入值外其他地方都为零的函数。更精确地说，由 `i : ι` 索引的基向量是 `Finsupp.single i 1`，这是一个有限支撑的函数，在 `i` 处取值为 `1`，在其他地方取值为 `0`。

```py
variable  [DecidableEq  ι] 
```

```py
example  :  Finsupp.basisSingleOne.repr  =  LinearEquiv.refl  K  (ι  →₀  K)  :=
  rfl

example  (i  :  ι)  :  Finsupp.basisSingleOne  i  =  Finsupp.single  i  1  :=
  rfl 
```

当索引类型是有限的时候，有限支撑函数的故事是不必要的。在这种情况下，我们可以使用更简单的 `Pi.basisFun`，它给出了整个 `ι → K` 的基。

```py
example  [Finite  ι]  (x  :  ι  →  K)  (i  :  ι)  :  (Pi.basisFun  K  ι).repr  x  i  =  x  i  :=  by
  simp 
```

回到抽象向量空间基的一般情况，我们可以将任何向量表示为基向量的线性组合。让我们先看看有限基的简单情况。

```py
example  [Fintype  ι]  :  ∑  i  :  ι,  B.repr  v  i  •  (B  i)  =  v  :=
  B.sum_repr  v 
```

当 `ι` 不是有限时，上述陈述在先验上没有意义：我们不能在 `ι` 上取和。然而，被求和的函数的支撑是有限的（它是 `B.repr v` 的支撑）。但是我们需要应用一个考虑到这一点的构造。在这里，Mathlib 使用了一个特殊目的的函数，需要一些时间来习惯：`Finsupp.linearCombination`（它是建立在更一般的 `Finsupp.sum` 之上的）。给定一个有限支撑的函数 `c` 从类型 `ι` 到基域 `K` 以及任何从 `ι` 到 `V` 的函数 `f`，`Finsupp.linearCombination K f c` 是 `c` 的支撑上的标量乘积 `c • f` 的和。特别是，我们可以将其替换为包含 `c` 的支撑的任何有限集上的和。

```py
example  (c  :  ι  →₀  K)  (f  :  ι  →  V)  (s  :  Finset  ι)  (h  :  c.support  ⊆  s)  :
  Finsupp.linearCombination  K  f  c  =  ∑  i  ∈  s,  c  i  •  f  i  :=
  Finsupp.linearCombination_apply_of_mem_supported  K  h 
```

也可以假设 `f` 是有限支撑的，并且仍然可以得到一个定义良好的和。但是 `Finsupp.linearCombination` 所做的选择与我们的基讨论相关，因为它允许我们陈述 `Basis.sum_repr` 的一般化。

```py
example  :  Finsupp.linearCombination  K  B  (B.repr  v)  =  v  :=
  B.linearCombination_repr  v 
```

一个人可能会想知道为什么 `K` 在这里是一个显式的参数，尽管它可以从 `c` 的类型中推断出来。关键是部分应用 `Finsupp.linearCombination K f` 本身是有趣的。它不是一个从 `ι →₀ K` 到 `V` 的裸函数，而是一个 `K`-线性映射。

```py
variable  (f  :  ι  →  V)  in
#check  (Finsupp.linearCombination  K  f  :  (ι  →₀  K)  →ₗ[K]  V) 
```

回到数学讨论，重要的是要理解，在形式化数学中，基中向量的表示可能没有你想象的那么有用。事实上，直接使用基的更抽象性质通常更有效。特别是，基的普遍性质将它们与其他代数中的自由对象连接起来，允许通过指定基向量的像来构造线性映射。这是 `Basis.constr`。对于任何 `K`-向量空间 `W`，我们的基 `B` 给出一个线性同构 `Basis.constr B K`，从 `ι → W` 到 `V →ₗ[K] W`。这个同构的特点是它将任何函数 `u : ι → W` 映射到一个线性映射，该映射将基向量 `B i` 映射到 `u i`，对于每个 `i : ι`。

```py
section

variable  {W  :  Type*}  [AddCommGroup  W]  [Module  K  W]
  (φ  :  V  →ₗ[K]  W)  (u  :  ι  →  W)

#check  (B.constr  K  :  (ι  →  W)  ≃ₗ[K]  (V  →ₗ[K]  W))

#check  (B.constr  K  u  :  V  →ₗ[K]  W)

example  (i  :  ι)  :  B.constr  K  u  (B  i)  =  u  i  :=
  B.constr_basis  K  u  i 
```

这个属性确实是特征性的，因为线性映射由它们在基上的值确定：

```py
example  (φ  ψ  :  V  →ₗ[K]  W)  (h  :  ∀  i,  φ  (B  i)  =  ψ  (B  i))  :  φ  =  ψ  :=
  B.ext  h 
```

如果我们在目标空间上也有一个基 `B'`，那么我们可以将线性映射与矩阵相识别。这种识别是一个 `K`-线性同构。

```py
variable  {ι'  :  Type*}  (B'  :  Basis  ι'  K  W)  [Fintype  ι]  [DecidableEq  ι]  [Fintype  ι']  [DecidableEq  ι']

open  LinearMap

#check  (toMatrix  B  B'  :  (V  →ₗ[K]  W)  ≃ₗ[K]  Matrix  ι'  ι  K)

open  Matrix  -- get access to the ``*ᵥ`` notation for multiplication between matrices and vectors.

example  (φ  :  V  →ₗ[K]  W)  (v  :  V)  :  (toMatrix  B  B'  φ)  *ᵥ  (B.repr  v)  =  B'.repr  (φ  v)  :=
  toMatrix_mulVec_repr  B  B'  φ  v

variable  {ι''  :  Type*}  (B''  :  Basis  ι''  K  W)  [Fintype  ι'']  [DecidableEq  ι'']

example  (φ  :  V  →ₗ[K]  W)  :  (toMatrix  B  B''  φ)  =  (toMatrix  B'  B''  .id)  *  (toMatrix  B  B'  φ)  :=  by
  simp

end 
```

作为这个主题的练习，我们将证明定理的一部分，该定理保证了内射有明确定义的行列式。具体来说，我们想要证明当两个基由相同的类型索引时，它们附加到任何内射上的矩阵具有相同的行列式。然后需要使用这些基都具有同构的索引类型来补充，以得到完整的结果。

当然，Mathlib 已经知道这一点，`simp` 可以立即关闭目标，所以你不应该太早使用它，而应该使用提供的引理。

```py
open  Module  LinearMap  Matrix

-- Some lemmas coming from the fact that `LinearMap.toMatrix` is an algebra morphism.
#check  toMatrix_comp
#check  id_comp
#check  comp_id
#check  toMatrix_id

-- Some lemmas coming from the fact that ``Matrix.det`` is a multiplicative monoid morphism.
#check  Matrix.det_mul
#check  Matrix.det_one

example  [Fintype  ι]  (B'  :  Basis  ι  K  V)  (φ  :  End  K  V)  :
  (toMatrix  B  B  φ).det  =  (toMatrix  B'  B'  φ).det  :=  by
  set  M  :=  toMatrix  B  B  φ
  set  M'  :=  toMatrix  B'  B'  φ
  set  P  :=  (toMatrix  B  B')  LinearMap.id
  set  P'  :=  (toMatrix  B'  B)  LinearMap.id
  sorry
end 
```

### 10.4.3\. 维度

回到单个向量空间的情况，基也有助于定义维度的概念。在这里，也有有限维向量空间的基本情况。对于这样的空间，我们期望维度是一个自然数。这是 `Module.finrank`。它将基域作为显式参数，因为给定的阿贝尔群可以是在不同域上的向量空间。

```py
section
#check  (Module.finrank  K  V  :  ℕ)

-- `Fin n → K` is the archetypical space with dimension `n` over `K`.
example  (n  :  ℕ)  :  Module.finrank  K  (Fin  n  →  K)  =  n  :=
  Module.finrank_fin_fun  K

-- Seen as a vector space over itself, `ℂ` has dimension one.
example  :  Module.finrank  ℂ  ℂ  =  1  :=
  Module.finrank_self  ℂ

-- But as a real vector space it has dimension two.
example  :  Module.finrank  ℝ  ℂ  =  2  :=
  Complex.finrank_real_complex 
```

注意，`Module.finrank` 定义适用于任何向量空间。对于无限维向量空间，它返回零，就像除以零返回零一样。

当然，许多引理需要有限维度的假设。这就是 `FiniteDimensional` 类型类的作用。例如，考虑如果没有这个假设，下一个例子会如何失败。

```py
example  [FiniteDimensional  K  V]  :  0  <  Module.finrank  K  V  ↔  Nontrivial  V  :=
  Module.finrank_pos_iff 
```

在上述陈述中，`Nontrivial V` 表示 `V` 至少有两个不同的元素。注意，`Module.finrank_pos_iff` 没有显式的参数。当从左到右使用它时这是可以的，但不是从右到左使用时，因为 Lean 没有从陈述 `Nontrivial V` 中猜测 `K` 的方法。在这种情况下，使用名称参数语法是有用的，在检查引理是在名为 `R` 的环上陈述之后。因此，我们可以写出：

```py
example  [FiniteDimensional  K  V]  (h  :  0  <  Module.finrank  K  V)  :  Nontrivial  V  :=  by
  apply  (Module.finrank_pos_iff  (R  :=  K)).1
  exact  h 
```

上述拼写很奇怪，因为我们已经有了 `h` 作为假设，所以我们完全可以给出完整的证明 `Module.finrank_pos_iff.1 h`，但对于更复杂的情况来说，了解这一点是好的。

根据定义，`有限维 K V` 可以从任何基中读取。

```py
variable  {ι  :  Type*}  (B  :  Basis  ι  K  V)

example  [Finite  ι]  :  FiniteDimensional  K  V  :=  FiniteDimensional.of_fintype_basis  B

example  [FiniteDimensional  K  V]  :  Finite  ι  :=
  (FiniteDimensional.fintypeBasisIndex  B).finite
end 
```

利用对应于线性子空间的子类型具有向量空间结构，我们可以讨论子空间的维度。

```py
section
variable  (E  F  :  Submodule  K  V)  [FiniteDimensional  K  V]

open  Module

example  :  finrank  K  (E  ⊔  F  :  Submodule  K  V)  +  finrank  K  (E  ⊓  F  :  Submodule  K  V)  =
  finrank  K  E  +  finrank  K  F  :=
  Submodule.finrank_sup_add_finrank_inf_eq  E  F

example  :  finrank  K  E  ≤  finrank  K  V  :=  Submodule.finrank_le  E 
```

在上述第一个陈述中，类型赋值的目的是确保将 `Type*` 的强制转换不会触发得太早。

我们现在可以准备一个关于 `finrank` 和子空间的练习。

```py
example  (h  :  finrank  K  V  <  finrank  K  E  +  finrank  K  F)  :
  Nontrivial  (E  ⊓  F  :  Submodule  K  V)  :=  by
  sorry
end 
```

让我们现在转向维度理论的通用情况。在这种情况下，`finrank` 是无用的，但我们仍然有，对于任何两个相同向量空间的基，它们之间的类型存在双射。因此，我们仍然可以希望将秩定义为基数，即“存在双射等价关系下类型集合的商”中的一个元素。

当讨论基数时，就像在这本书的其他地方一样，很难忽视围绕罗素悖论的基础性问题。没有所有类型的类型，因为这会导致逻辑上的不一致。这个问题通过我们通常试图忽略的宇宙层次结构得到解决。

每个类型都有一个宇宙级别，这些级别的行为类似于自然数。特别是存在零级，相应的宇宙 `Type 0` 简单地表示为 `Type`。这个宇宙足以容纳几乎所有经典数学。例如 `ℕ` 和 `ℝ` 有类型 `Type`。每个级别 `u` 有一个后继者，表示为 `u + 1`，并且 `Type u` 有类型 `Type (u+1)`。

但宇宙级别不是自然数，它们具有非常不同的性质，并且没有类型。特别是你无法在 Lean 中陈述类似于 `u ≠ u + 1` 的内容。根本不存在这样的类型。甚至陈述 `Type u ≠ Type (u+1)` 也没有任何意义，因为 `Type u` 和 `Type (u+1)` 具有不同的类型。

每当我们书写 `Type*` 时，Lean 会插入一个名为 `u_n` 的宇宙级别变量，其中 `n` 是一个数字。这允许定义和陈述存在于所有宇宙中。

给定一个宇宙级别 `u`，我们可以在 `Type u` 上定义一个等价关系，即如果两个类型 `α` 和 `β` 之间存在双射，则这两个类型 `α` 和 `β` 是等价的。商类型 `Cardinal.{u}` 位于 `Type (u+1)` 中。大括号表示一个宇宙变量。在这个商类型中，`α : Type u` 的像为 `Cardinal.mk α : Cardinal.{u}`。

但我们无法直接比较不同宇宙中的基数。因此，技术上我们无法将向量空间 `V` 的秩定义为索引 `V` 的基的所有类型的基数。因此，它被定义为 `V` 中所有线性无关集的基数 `Module.rank K V` 的上确界。如果 `V` 的宇宙级别为 `u`，则其秩的类型为 `Cardinal.{u}`。

```py
#check  V  -- Type u_2
#check  Module.rank  K  V  -- Cardinal.{u_2} 
```

尽管如此，我们仍然可以将这个定义与基联系起来。事实上，在宇宙级别上也有一个交换的 `max` 操作，并且对于两个宇宙级别 `u` 和 `v`，存在一个操作 `Cardinal.lift.{u, v} : Cardinal.{v} → Cardinal.{max v u}`，它允许将基数放入一个共同的宇宙，并陈述维度定理。

```py
universe  u  v  -- `u` and `v` will denote universe levels

variable  {ι  :  Type  u}  (B  :  Basis  ι  K  V)
  {ι'  :  Type  v}  (B'  :  Basis  ι'  K  V)

example  :  Cardinal.lift.{v,  u}  (.mk  ι)  =  Cardinal.lift.{u,  v}  (.mk  ι')  :=
  mk_eq_mk_of_basis  B  B' 
```

我们可以使用自然数到有限基数（或更精确地说，生活在 `Cardinal.{v}` 中的有限基数，其中 `v` 是 `V` 的宇宙级别）的强制转换将有限维情况与这次讨论联系起来。

```py
example  [FiniteDimensional  K  V]  :
  (Module.finrank  K  V  :  Cardinal)  =  Module.rank  K  V  :=
  Module.finrank_eq_rank  K  V 
``` 上一节 下一节

* * *

© 版权所有 2020-2025，Jeremy Avigad，Patrick Massot。文本许可协议为 CC BY 4.0。

使用 [Sphinx](https://www.sphinx-doc.org/) 和由 [Read the Docs](https://readthedocs.org) 提供的 [主题](https://github.com/readthedocs/sphinx_rtd_theme) 构建。## 10.1. 向量空间和线性映射

### 10.1.1. 向量空间

我们将直接从抽象线性代数开始，它发生在任何域上的向量空间中。然而，你可以在第 10.4.1 节中找到有关矩阵的信息，这部分内容与抽象理论没有逻辑上的依赖。Mathlib 实际上处理的是一个更通用的线性代数版本，涉及“模块”一词，但到目前为止，我们将假装这只是一个古怪的拼写习惯。

表达“设 $K$ 为一个域，$V$ 为 $K$ 上的向量空间”（并将它们作为后续结果的隐含参数）的方式是：

```py
variable  {K  :  Type*}  [Field  K]  {V  :  Type*}  [AddCommGroup  V]  [Module  K  V] 
```

我们在第八章中解释了为什么我们需要两个独立的类型类 `[AddCommGroup V] [Module K V]`。简而言之，从数学上讲，我们想要表达拥有 $K$ 向量空间结构意味着拥有加法交换群结构。我们可以告诉 Lean。但这样一来，每当 Lean 需要在类型 $V$ 上找到这样的群结构时，它就会使用一个*完全未指定的*域 $K$ 来寻找向量空间结构，而这个 $K$ 无法从 $V$ 中推断出来。这对类型类综合系统来说是非常糟糕的。

向量 $v$ 与标量 $a$ 的乘积表示为 a • v。我们在以下示例中列出了一些关于此操作与加法交互的代数规则。当然，simp 或 apply? 会找到这些证明。还有一个模块策略，它解决从向量空间和域的公理中得出的目标，就像在交换环中使用 ring 策略或在群中使用 group 策略一样。但仍然有用的是记住标量乘法在引理名称中简写为 smul。

```py
example  (a  :  K)  (u  v  :  V)  :  a  •  (u  +  v)  =  a  •  u  +  a  •  v  :=
  smul_add  a  u  v

example  (a  b  :  K)  (u  :  V)  :  (a  +  b)  •  u  =  a  •  u  +  b  •  u  :=
  add_smul  a  b  u

example  (a  b  :  K)  (u  :  V)  :  a  •  b  •  u  =  b  •  a  •  u  :=
  smul_comm  a  b  u 
```

作为对更高级读者的快速提示，让我们指出，正如术语所暗示的，Mathlib 的线性代数也涵盖了（不一定交换的）环上的模。实际上，它甚至涵盖了半环上的半模。如果你认为你不需要这种程度的普遍性，你可以冥想以下示例，它很好地捕捉了关于理想在子模上作用的许多代数规则：

```py
example  {R  M  :  Type*}  [CommSemiring  R]  [AddCommMonoid  M]  [Module  R  M]  :
  Module  (Ideal  R)  (Submodule  R  M)  :=
  inferInstance 
```

### 10.1.2\. 线性映射

接下来我们需要线性映射。像群同态一样，Mathlib 中的线性映射是捆绑映射，即由映射及其线性性质证明组成的包。这些捆绑映射在应用时被转换为普通函数。有关此设计的更多信息，请参阅 第八章。

两个 `K`-向量空间 `V` 和 `W` 之间线性映射的类型表示为 `V →ₗ[K] W`。下标 l 代表线性。一开始可能觉得在这个符号中指定 `K` 很奇怪。但这是几个领域同时存在时的关键。例如，从 $ℂ$ 到 $ℂ$ 的实线性映射是每个映射 $z ↦ az + b\bar{z}$，而只有映射 $z ↦ az$ 是复线性，这种差异在复分析中至关重要。

```py
variable  {W  :  Type*}  [AddCommGroup  W]  [Module  K  W]

variable  (φ  :  V  →ₗ[K]  W)

example  (a  :  K)  (v  :  V)  :  φ  (a  •  v)  =  a  •  φ  v  :=
  map_smul  φ  a  v

example  (v  w  :  V)  :  φ  (v  +  w)  =  φ  v  +  φ  w  :=
  map_add  φ  v  w 
```

注意，`V →ₗ[K] W` 本身就带有有趣的代数结构（这是捆绑那些映射的部分动机）。它是一个 `K`-向量空间，因此我们可以添加线性映射并乘以标量。

```py
variable  (ψ  :  V  →ₗ[K]  W)

#check  (2  •  φ  +  ψ  :  V  →ₗ[K]  W) 
```

使用捆绑映射的一个缺点是我们不能使用普通函数组合。我们需要使用 `LinearMap.comp` 或 `∘ₗ` 的符号。

```py
variable  (θ  :  W  →ₗ[K]  V)

#check  (φ.comp  θ  :  W  →ₗ[K]  W)
#check  (φ  ∘ₗ  θ  :  W  →ₗ[K]  W) 
```

构造线性映射有两种主要方式。首先，我们可以通过提供函数和线性证明来构建结构。像往常一样，这是通过结构代码操作实现的：你可以输入 `example : V →ₗ[K] V := _` 并使用附加到下划线的“生成骨架”代码操作。

```py
example  :  V  →ₗ[K]  V  where
  toFun  v  :=  3  •  v
  map_add'  _  _  :=  smul_add  ..
  map_smul'  _  _  :=  smul_comm  .. 
```

你可能会想知道为什么 `LinearMap` 的证明字段以撇号结尾。这是因为它们在定义函数强制转换之前就已经定义了，因此它们是用 `LinearMap.toFun` 表述的。然后它们被重新表述为 `LinearMap.map_add` 和 `LinearMap.map_smul`，以函数强制转换的形式。但这还不是故事的结尾。人们还想要一个适用于任何（捆绑的）保持加法的映射的 `map_add` 版本，例如加法群同态、线性映射、连续线性映射、`K`-代数映射等。这个版本是 `map_add`（在根命名空间中）。中间版本 `LinearMap.map_add` 有点冗余，但允许使用点符号，这在某些时候可能很方便。对于 `map_smul` 也有类似的故事，一般框架在 第八章 中解释。

```py
#check  (φ.map_add'  :  ∀  x  y  :  V,  φ.toFun  (x  +  y)  =  φ.toFun  x  +  φ.toFun  y)
#check  (φ.map_add  :  ∀  x  y  :  V,  φ  (x  +  y)  =  φ  x  +  φ  y)
#check  (map_add  φ  :  ∀  x  y  :  V,  φ  (x  +  y)  =  φ  x  +  φ  y) 
```

也可以使用各种组合子从 Mathlib 中已定义的线性映射中构建线性映射。例如，上面的例子已经知道为 `LinearMap.lsmul K V 3`。`K` 和 `V` 在这里作为显式参数有几个原因。最紧迫的一个原因是，从裸的 `LinearMap.lsmul 3` 中，Lean 无法推断出 `V` 或甚至 `K`。但 `LinearMap.lsmul K V` 本身也是一个有趣的对象：它具有类型 `K →ₗ[K] V →ₗ[K] V`，这意味着它是一个从 `K`（被视为自身的向量空间）到 `V` 到 `V` 的 `K`-线性映射。

```py
#check  (LinearMap.lsmul  K  V  3  :  V  →ₗ[K]  V)
#check  (LinearMap.lsmul  K  V  :  K  →ₗ[K]  V  →ₗ[K]  V) 
```

还有表示为 `V ≃ₗ[K] W` 的线性同构类型 `LinearEquiv`。`f : V ≃ₗ[K] W` 的逆是 `f.symm : W ≃ₗ[K] V`，`f` 和 `g` 的组合是 `f.trans g`，也记作 `f ≪≫ₗ g`，恒等同构是 `V` 的 `LinearEquiv.refl K V`。此类型中的元素在必要时自动转换为形态和函数。

```py
example  (f  :  V  ≃ₗ[K]  W)  :  f  ≪≫ₗ  f.symm  =  LinearEquiv.refl  K  V  :=
  f.self_trans_symm 
```

可以使用 `LinearEquiv.ofBijective` 从双射形态构建同构。这样做会使逆函数不可计算。

```py
noncomputable  example  (f  :  V  →ₗ[K]  W)  (h  :  Function.Bijective  f)  :  V  ≃ₗ[K]  W  :=
  .ofBijective  f  h 
```

注意，在上面的例子中，Lean 使用宣布的类型来理解 `.ofBijective` 指的是 `LinearEquiv.ofBijective`（无需打开任何命名空间）。

### 10.1.3. 向量空间的和与积

我们可以使用直接和和直接积从旧的向量空间中构建新的向量空间。让我们从两个向量空间开始。在这种情况下，和与积之间没有区别，我们可以简单地使用积类型。在下面的代码片段中，我们简单地展示了如何将所有结构映射（包含和投射）作为线性映射来获取，以及构建线性映射到积和从和中构造的通用性质（如果你不熟悉加法和积之间的范畴论区别，你可以简单地忽略通用性质词汇，并关注以下示例的类型）。

```py
section  binary_product

variable  {W  :  Type*}  [AddCommGroup  W]  [Module  K  W]
variable  {U  :  Type*}  [AddCommGroup  U]  [Module  K  U]
variable  {T  :  Type*}  [AddCommGroup  T]  [Module  K  T]

-- First projection map
example  :  V  ×  W  →ₗ[K]  V  :=  LinearMap.fst  K  V  W

-- Second projection map
example  :  V  ×  W  →ₗ[K]  W  :=  LinearMap.snd  K  V  W

-- Universal property of the product
example  (φ  :  U  →ₗ[K]  V)  (ψ  :  U  →ₗ[K]  W)  :  U  →ₗ[K]  V  ×  W  :=  LinearMap.prod  φ  ψ

-- The product map does the expected thing, first component
example  (φ  :  U  →ₗ[K]  V)  (ψ  :  U  →ₗ[K]  W)  :  LinearMap.fst  K  V  W  ∘ₗ  LinearMap.prod  φ  ψ  =  φ  :=  rfl

-- The product map does the expected thing, second component
example  (φ  :  U  →ₗ[K]  V)  (ψ  :  U  →ₗ[K]  W)  :  LinearMap.snd  K  V  W  ∘ₗ  LinearMap.prod  φ  ψ  =  ψ  :=  rfl

-- We can also combine maps in parallel
example  (φ  :  V  →ₗ[K]  U)  (ψ  :  W  →ₗ[K]  T)  :  (V  ×  W)  →ₗ[K]  (U  ×  T)  :=  φ.prodMap  ψ

-- This is simply done by combining the projections with the universal property
example  (φ  :  V  →ₗ[K]  U)  (ψ  :  W  →ₗ[K]  T)  :
  φ.prodMap  ψ  =  (φ  ∘ₗ  .fst  K  V  W).prod  (ψ  ∘ₗ  .snd  K  V  W)  :=  rfl

-- First inclusion map
example  :  V  →ₗ[K]  V  ×  W  :=  LinearMap.inl  K  V  W

-- Second inclusion map
example  :  W  →ₗ[K]  V  ×  W  :=  LinearMap.inr  K  V  W

-- Universal property of the sum (aka coproduct)
example  (φ  :  V  →ₗ[K]  U)  (ψ  :  W  →ₗ[K]  U)  :  V  ×  W  →ₗ[K]  U  :=  φ.coprod  ψ

-- The coproduct map does the expected thing, first component
example  (φ  :  V  →ₗ[K]  U)  (ψ  :  W  →ₗ[K]  U)  :  φ.coprod  ψ  ∘ₗ  LinearMap.inl  K  V  W  =  φ  :=
  LinearMap.coprod_inl  φ  ψ

-- The coproduct map does the expected thing, second component
example  (φ  :  V  →ₗ[K]  U)  (ψ  :  W  →ₗ[K]  U)  :  φ.coprod  ψ  ∘ₗ  LinearMap.inr  K  V  W  =  ψ  :=
  LinearMap.coprod_inr  φ  ψ

-- The coproduct map is defined in the expected way
example  (φ  :  V  →ₗ[K]  U)  (ψ  :  W  →ₗ[K]  U)  (v  :  V)  (w  :  W)  :
  φ.coprod  ψ  (v,  w)  =  φ  v  +  ψ  w  :=
  rfl

end  binary_product 
```

现在我们转向任意向量空间族的和与积。在这里，我们将简单地看到如何定义一个向量空间族并访问和与积的通用性质。请注意，直接和的符号范围限定在 `DirectSum` 命名空间中，并且直接和的通用性质要求索引类型上的可判定等价性（这某种程度上是一个实现上的意外）。

```py
section  families
open  DirectSum

variable  {ι  :  Type*}  [DecidableEq  ι]
  (V  :  ι  →  Type*)  [∀  i,  AddCommGroup  (V  i)]  [∀  i,  Module  K  (V  i)]

-- The universal property of the direct sum assembles maps from the summands to build
-- a map from the direct sum
example  (φ  :  Π  i,  (V  i  →ₗ[K]  W))  :  (⨁  i,  V  i)  →ₗ[K]  W  :=
  DirectSum.toModule  K  ι  W  φ

-- The universal property of the direct product assembles maps into the factors
-- to build a map into the direct product
example  (φ  :  Π  i,  (W  →ₗ[K]  V  i))  :  W  →ₗ[K]  (Π  i,  V  i)  :=
  LinearMap.pi  φ

-- The projection maps from the product
example  (i  :  ι)  :  (Π  j,  V  j)  →ₗ[K]  V  i  :=  LinearMap.proj  i

-- The inclusion maps into the sum
example  (i  :  ι)  :  V  i  →ₗ[K]  (⨁  i,  V  i)  :=  DirectSum.lof  K  ι  V  i

-- The inclusion maps into the product
example  (i  :  ι)  :  V  i  →ₗ[K]  (Π  i,  V  i)  :=  LinearMap.single  K  V  i

-- In case `ι` is a finite type, there is an isomorphism between the sum and product.
example  [Fintype  ι]  :  (⨁  i,  V  i)  ≃ₗ[K]  (Π  i,  V  i)  :=
  linearEquivFunOnFintype  K  ι  V

end  families 
```

## 10.2. 子空间和商

### 10.2.1. 子空间

正如线性映射是捆绑在一起的，`V` 的线性子空间也是一个由 `V` 中的集合组成的捆绑结构，称为子空间的载体，并具有相关的闭包性质。由于 Mathlib 实际上用于线性代数的更一般上下文，所以这里出现的是“模块”一词而不是向量空间。

```py
variable  {K  :  Type*}  [Field  K]  {V  :  Type*}  [AddCommGroup  V]  [Module  K  V]

example  (U  :  Submodule  K  V)  {x  y  :  V}  (hx  :  x  ∈  U)  (hy  :  y  ∈  U)  :
  x  +  y  ∈  U  :=
  U.add_mem  hx  hy

example  (U  :  Submodule  K  V)  {x  :  V}  (hx  :  x  ∈  U)  (a  :  K)  :
  a  •  x  ∈  U  :=
  U.smul_mem  a  hx 
```

在上面的例子中，重要的是要理解`Submodule K V`是`V`的`K`-线性子空间类型，而不是一个`IsSubmodule U`谓词，其中`U`是`Set V`的元素。`Submodule K V`被赋予了到`Set V`的转换和`V`上的成员谓词。参见第 8.3 节以了解如何以及为什么这样做。

当然，如果两个子空间具有相同的元素，则它们是相同的。这一事实被注册用于与`ext`策略一起使用，该策略可以用来证明两个子空间相等，就像它被用来证明两个集合相等一样。

例如，为了表述和证明`ℝ`是`ℂ`的`ℝ`-线性子空间，我们真正想要的是构造一个类型为`Submodule ℝ ℂ`的项，其投影到`Set ℂ`是`ℝ`，或者更精确地说，是`ℝ`在`ℂ`中的像。

```py
noncomputable  example  :  Submodule  ℝ  ℂ  where
  carrier  :=  Set.range  ((↑)  :  ℝ  →  ℂ)
  add_mem'  :=  by
  rintro  _  _  ⟨n,  rfl⟩  ⟨m,  rfl⟩
  use  n  +  m
  simp
  zero_mem'  :=  by
  use  0
  simp
  smul_mem'  :=  by
  rintro  c  -  ⟨a,  rfl⟩
  use  c*a
  simp 
```

在`Submodule`中证明字段末尾的素数与`LinearMap`中的素数类似。这些字段是用`carrier`字段来表述的，因为它们是在`MemberShip`实例之前定义的。然后，它们被上面看到的`Submodule.add_mem`、`Submodule.zero_mem`和`Submodule.smul_mem`所取代。

作为操作子空间和线性映射的练习，你将通过一个线性映射定义子空间的逆映射（当然，我们将在下面看到 Mathlib 已经知道这一点）。记住，可以使用`Set.mem_preimage`来重写涉及成员和逆映射的陈述。这是除了上面讨论的关于`LinearMap`和`Submodule`的引理之外，你将需要的唯一引理。

```py
def  preimage  {W  :  Type*}  [AddCommGroup  W]  [Module  K  W]  (φ  :  V  →ₗ[K]  W)  (H  :  Submodule  K  W)  :
  Submodule  K  V  where
  carrier  :=  φ  ⁻¹'  H
  zero_mem'  :=  by
  sorry
  add_mem'  :=  by
  sorry
  smul_mem'  :=  by
  sorry 
```

使用类型类，Mathlib 知道向量空间的子空间继承了向量空间结构。

```py
example  (U  :  Submodule  K  V)  :  Module  K  U  :=  inferInstance 
```

这个例子很微妙。对象`U`不是一个类型，但 Lean 会自动将其解释为`V`的子类型，从而将其转换为类型。因此，上面的例子可以更明确地表述为：

```py
example  (U  :  Submodule  K  V)  :  Module  K  {x  :  V  //  x  ∈  U}  :=  inferInstance 
```

### 10.2.2. 完全格结构和内部直和

拥有类型`Submodule K V`而不是谓词`IsSubmodule : Set V → Prop`的一个重要好处是，可以很容易地为`Submodule K V`赋予额外的结构。重要的是，它具有关于包含的完全格结构。例如，我们不是用说明两个`V`的子空间交集仍然是子空间的引理，而是使用格运算`⊓`来构造交集。然后，我们可以将任意关于格的引理应用于构造。

让我们检查两个子空间下确界所对应的集合确实，按照定义，是它们的交集。

```py
example  (H  H'  :  Submodule  K  V)  :
  ((H  ⊓  H'  :  Submodule  K  V)  :  Set  V)  =  (H  :  Set  V)  ∩  (H'  :  Set  V)  :=  rfl 
```

对于相当于底层集合交集的符号可能看起来很奇怪，但对应关系并不适用于上确界操作和集合并集，因为子空间的并集在一般情况下不是子空间。相反，需要使用由并集生成的子空间，这可以通过`Submodule.span`来完成。

```py
example  (H  H'  :  Submodule  K  V)  :
  ((H  ⊔  H'  :  Submodule  K  V)  :  Set  V)  =  Submodule.span  K  ((H  :  Set  V)  ∪  (H'  :  Set  V))  :=  by
  simp  [Submodule.span_union] 
```

另一个微妙之处在于，`V`本身不具有类型`Submodule K V`，因此我们需要一种方式来谈论`V`作为`V`的子空间。这也由格结构提供：整个子空间是这个格的最高元素。

```py
example  (x  :  V)  :  x  ∈  (⊤  :  Submodule  K  V)  :=  trivial 
```

同样，这个格的最低元素是只包含零元素的子空间。

```py
example  (x  :  V)  :  x  ∈  (⊥  :  Submodule  K  V)  ↔  x  =  0  :=  Submodule.mem_bot  K 
```

特别地，我们可以讨论那些在（内部）直和中的子空间的情况。在两个子空间的情况下，我们使用通用谓词`IsCompl`，它对任何有界偏序类型都有意义。在一般子空间族的情况下，我们使用`DirectSum.IsInternal`。

```py
-- If two subspaces are in direct sum then they span the whole space.
example  (U  V  :  Submodule  K  V)  (h  :  IsCompl  U  V)  :
  U  ⊔  V  =  ⊤  :=  h.sup_eq_top

-- If two subspaces are in direct sum then they intersect only at zero.
example  (U  V  :  Submodule  K  V)  (h  :  IsCompl  U  V)  :
  U  ⊓  V  =  ⊥  :=  h.inf_eq_bot

section
open  DirectSum
variable  {ι  :  Type*}  [DecidableEq  ι]

-- If subspaces are in direct sum then they span the whole space.
example  (U  :  ι  →  Submodule  K  V)  (h  :  DirectSum.IsInternal  U)  :
  ⨆  i,  U  i  =  ⊤  :=  h.submodule_iSup_eq_top

-- If subspaces are in direct sum then they pairwise intersect only at zero.
example  {ι  :  Type*}  [DecidableEq  ι]  (U  :  ι  →  Submodule  K  V)  (h  :  DirectSum.IsInternal  U)
  {i  j  :  ι}  (hij  :  i  ≠  j)  :  U  i  ⊓  U  j  =  ⊥  :=
  (h.submodule_iSupIndep.pairwiseDisjoint  hij).eq_bot

-- Those conditions characterize direct sums.
#check  DirectSum.isInternal_submodule_iff_independent_and_iSup_eq_top

-- The relation with external direct sums: if a family of subspaces is
-- in internal direct sum then the map from their external direct sum into `V`
-- is a linear isomorphism.
noncomputable  example  {ι  :  Type*}  [DecidableEq  ι]  (U  :  ι  →  Submodule  K  V)
  (h  :  DirectSum.IsInternal  U)  :  (⨁  i,  U  i)  ≃ₗ[K]  V  :=
  LinearEquiv.ofBijective  (coeLinearMap  U)  h
end 
```

### 10.2.3\. 由集合生成的子空间

除了从现有子空间构建子空间之外，我们还可以使用`Submodule.span K s`从任何集合`s`构建子空间，它构建包含`s`的最小子空间。在纸上，通常使用这个空间由`s`的元素的所有线性组合构成。但通常更有效的是使用其由`Submodule.span_le`表达的通用性质，以及整个 Galois 连接理论。

```py
example  {s  :  Set  V}  (E  :  Submodule  K  V)  :  Submodule.span  K  s  ≤  E  ↔  s  ⊆  E  :=
  Submodule.span_le

example  :  GaloisInsertion  (Submodule.span  K)  ((↑)  :  Submodule  K  V  →  Set  V)  :=
  Submodule.gi  K  V 
```

当这些还不够时，可以使用相关的归纳原理`Submodule.span_induction`，该原理确保只要在`zero`和`s`的元素上成立，并且对加法和数乘稳定，那么`s`的生成空间中的每个元素都满足该性质。

作为练习，让我们重新证明`Submodule.mem_sup`的一个蕴含。记住，你可以使用模块策略来关闭由与`V`上各种代数操作相关的公理得出的目标。

```py
example  {S  T  :  Submodule  K  V}  {x  :  V}  (h  :  x  ∈  S  ⊔  T)  :
  ∃  s  ∈  S,  ∃  t  ∈  T,  x  =  s  +  t  :=  by
  rw  [←  S.span_eq,  ←  T.span_eq,  ←  Submodule.span_union]  at  h
  induction  h  using  Submodule.span_induction  with
  |  mem  y  h  =>
  sorry
  |  zero  =>
  sorry
  |  add  x  y  hx  hy  hx'  hy'  =>
  sorry
  |  smul  a  x  hx  hx'  =>
  sorry 
```

### 10.2.4\. 推拉子空间

如前所述，我们现在描述如何通过线性映射推拉子空间。在 Mathlib 中，通常第一个操作称为`map`，第二个操作称为`comap`。

```py
section

variable  {W  :  Type*}  [AddCommGroup  W]  [Module  K  W]  (φ  :  V  →ₗ[K]  W)

variable  (E  :  Submodule  K  V)  in
#check  (Submodule.map  φ  E  :  Submodule  K  W)

variable  (F  :  Submodule  K  W)  in
#check  (Submodule.comap  φ  F  :  Submodule  K  V) 
```

注意那些位于`Submodule`命名空间中的元素，因此可以使用点符号并写作`E.map φ`而不是`Submodule.map φ E`，但这在阅读上相当尴尬（尽管一些 Mathlib 贡献者使用这种拼写）。

特别是，线性映射的范围和核是子空间。这些特殊情况很重要，足以得到声明。

```py
example  :  LinearMap.range  φ  =  .map  φ  ⊤  :=  LinearMap.range_eq_map  φ

example  :  LinearMap.ker  φ  =  .comap  φ  ⊥  :=  Submodule.comap_bot  φ  -- or `rfl` 
```

注意，我们不能写`φ.ker`代替`LinearMap.ker φ`，因为`LinearMap.ker`也适用于保持更多结构的映射类，因此它不期望一个以`LinearMap`开头的类型作为参数，因此点符号在这里不起作用。然而，我们能够在右侧使用另一种点符号。因为 Lean 在展开左侧后期望一个类型为`Submodule K V`的项，它将`.comap`解释为`Submodule.comap`。

以下引理给出了这些子模块与`φ`的性质之间的关键关系。

```py
open  Function  LinearMap

example  :  Injective  φ  ↔  ker  φ  =  ⊥  :=  ker_eq_bot.symm

example  :  Surjective  φ  ↔  range  φ  =  ⊤  :=  range_eq_top.symm 
```

作为练习，让我们证明`map`和`comap`的伽罗瓦连接性质。可以使用以下引理，但这不是必需的，因为它们是按定义成立的。

```py
#check  Submodule.mem_map_of_mem
#check  Submodule.mem_map
#check  Submodule.mem_comap

example  (E  :  Submodule  K  V)  (F  :  Submodule  K  W)  :
  Submodule.map  φ  E  ≤  F  ↔  E  ≤  Submodule.comap  φ  F  :=  by
  sorry 
```

### 10.2.5\. 商空间

商向量空间使用通用的商记号（使用`\quot`输入，而不是普通`/`）。商空间的投影是`Submodule.mkQ`，其通用性质是`Submodule.liftQ`。

```py
variable  (E  :  Submodule  K  V)

example  :  Module  K  (V  ⧸  E)  :=  inferInstance

example  :  V  →ₗ[K]  V  ⧸  E  :=  E.mkQ

example  :  ker  E.mkQ  =  E  :=  E.ker_mkQ

example  :  range  E.mkQ  =  ⊤  :=  E.range_mkQ

example  (hφ  :  E  ≤  ker  φ)  :  V  ⧸  E  →ₗ[K]  W  :=  E.liftQ  φ  hφ

example  (F  :  Submodule  K  W)  (hφ  :  E  ≤  .comap  φ  F)  :  V  ⧸  E  →ₗ[K]  W  ⧸  F  :=  E.mapQ  F  φ  hφ

noncomputable  example  :  (V  ⧸  LinearMap.ker  φ)  ≃ₗ[K]  range  φ  :=  φ.quotKerEquivRange 
```

作为练习，让我们证明商空间子空间的对应定理。Mathlib 知道一个稍微更精确的版本，即`Submodule.comapMkQRelIso`。

```py
open  Submodule

#check  Submodule.map_comap_eq
#check  Submodule.comap_map_eq

example  :  Submodule  K  (V  ⧸  E)  ≃  {  F  :  Submodule  K  V  //  E  ≤  F  }  where
  toFun  :=  sorry
  invFun  :=  sorry
  left_inv  :=  sorry
  right_inv  :=  sorry 
```

## 10.3\. 内射

线性映射的一个重要特殊情况是内射：从向量空间到自身的线性映射。它们很有趣，因为它们形成一个`K`-代数。特别是，我们可以在它们上评估系数在`K`中的多项式，并且它们可以有特征值和特征向量。

Mathlib 使用缩写`Module.End K V := V →ₗ[K] V`，这在使用很多这些（尤其是在打开`Module`命名空间之后）时很方便。

```py
variable  {K  :  Type*}  [Field  K]  {V  :  Type*}  [AddCommGroup  V]  [Module  K  V]

variable  {W  :  Type*}  [AddCommGroup  W]  [Module  K  W]

open  Polynomial  Module  LinearMap  End

example  (φ  ψ  :  End  K  V)  :  φ  *  ψ  =  φ  ∘ₗ  ψ  :=
  End.mul_eq_comp  φ  ψ  -- `rfl` would also work

-- evaluating `P` on `φ`
example  (P  :  K[X])  (φ  :  End  K  V)  :  V  →ₗ[K]  V  :=
  aeval  φ  P

-- evaluating `X` on `φ` gives back `φ`
example  (φ  :  End  K  V)  :  aeval  φ  (X  :  K[X])  =  φ  :=
  aeval_X  φ 
```

作为操作内射、子空间和多项式的练习，让我们证明（二元的）核引理：对于任何内射$φ$和任意两个互质的聚合物$P$和$Q$，我们有$\ker P(φ) ⊕ \ker Q(φ) = \ker \big(PQ(φ)\big)$。

注意，`IsCoprime x y`被定义为`∃ a b, a * x + b * y = 1`。

```py
#check  Submodule.eq_bot_iff
#check  Submodule.mem_inf
#check  LinearMap.mem_ker

example  (P  Q  :  K[X])  (h  :  IsCoprime  P  Q)  (φ  :  End  K  V)  :  ker  (aeval  φ  P)  ⊓  ker  (aeval  φ  Q)  =  ⊥  :=  by
  sorry

#check  Submodule.add_mem_sup
#check  map_mul
#check  End.mul_apply
#check  LinearMap.ker_le_ker_comp

example  (P  Q  :  K[X])  (h  :  IsCoprime  P  Q)  (φ  :  End  K  V)  :
  ker  (aeval  φ  P)  ⊔  ker  (aeval  φ  Q)  =  ker  (aeval  φ  (P*Q))  :=  by
  sorry 
```

现在我们转向对特征空间和特征值的讨论。与内射$φ$和标量$a$相关的特征空间是$φ - aId$的核。特征空间对所有`a`的值都定义，尽管它们只有在非零时才有趣。然而，根据定义，特征向量是特征空间中的非零元素。相应的谓词是`End.HasEigenvector`。

```py
example  (φ  :  End  K  V)  (a  :  K)  :  φ.eigenspace  a  =  LinearMap.ker  (φ  -  a  •  1)  :=
  End.eigenspace_def 
```

然后存在一个谓词`End.HasEigenvalue`和相应的子类型`End.Eigenvalues`。

```py
example  (φ  :  End  K  V)  (a  :  K)  :  φ.HasEigenvalue  a  ↔  φ.eigenspace  a  ≠  ⊥  :=
  Iff.rfl

example  (φ  :  End  K  V)  (a  :  K)  :  φ.HasEigenvalue  a  ↔  ∃  v,  φ.HasEigenvector  a  v  :=
  ⟨End.HasEigenvalue.exists_hasEigenvector,  fun  ⟨_,  hv⟩  ↦  φ.hasEigenvalue_of_hasEigenvector  hv⟩

example  (φ  :  End  K  V)  :  φ.Eigenvalues  =  {a  //  φ.HasEigenvalue  a}  :=
  rfl

-- Eigenvalue are roots of the minimal polynomial
example  (φ  :  End  K  V)  (a  :  K)  :  φ.HasEigenvalue  a  →  (minpoly  K  φ).IsRoot  a  :=
  φ.isRoot_of_hasEigenvalue

-- In finite dimension, the converse is also true (we will discuss dimension below)
example  [FiniteDimensional  K  V]  (φ  :  End  K  V)  (a  :  K)  :
  φ.HasEigenvalue  a  ↔  (minpoly  K  φ).IsRoot  a  :=
  φ.hasEigenvalue_iff_isRoot

-- Cayley-Hamilton
example  [FiniteDimensional  K  V]  (φ  :  End  K  V)  :  aeval  φ  φ.charpoly  =  0  :=
  φ.aeval_self_charpoly 
```

## 10.4\. 矩阵、基和维度

### 10.4.1\. 矩阵

在介绍抽象向量空间的基之前，我们回到更基础的线性代数设置，即某个域 $K$ 的 $K^n$。在这里，主要对象是向量和矩阵。对于具体向量，可以使用 `![…]` 符号，其中分量由逗号分隔。对于具体矩阵，我们可以使用 `!![…]` 符号，行由分号分隔，行的分量由冒号分隔。当条目具有可计算类型，如 `ℕ` 或 `ℚ` 时，我们可以使用 `eval` 命令进行基本操作。

```py
section  matrices

-- Adding vectors
#eval  ![1,  2]  +  ![3,  4]  -- ![4, 6]

-- Adding matrices
#eval  !![1,  2;  3,  4]  +  !![3,  4;  5,  6]  -- !![4, 6; 8, 10]

-- Multiplying matrices
#eval  !![1,  2;  3,  4]  *  !![3,  4;  5,  6]  -- !![13, 16; 29, 36] 
```

重要的是要理解，这种 `#eval` 的使用仅对探索有趣，它不是用来取代计算机代数系统，如 Sage。这里用于矩阵的数据表示在计算效率方面没有任何优势。它使用函数而不是数组，并且优化用于证明，而不是计算。`#eval` 所使用的虚拟机也不是为此用途优化的。

注意矩阵表示法列出行，但向量表示法既不是行向量也不是列向量。从左（分别）乘以矩阵的向量解释为行（分别）向量。这对应于 `Matrix.vecMul` 操作，符号为 `ᵥ*`，以及 `Matrix.mulVec` 操作，符号为 ` *ᵥ`。这些符号在 `Matrix` 命名空间中定义，因此我们需要打开它。

```py
open  Matrix

-- matrices acting on vectors on the left
#eval  !![1,  2;  3,  4]  *ᵥ  ![1,  1]  -- ![3, 7]

-- matrices acting on vectors on the left, resulting in a size one matrix
#eval  !![1,  2]  *ᵥ  ![1,  1]  -- ![3]

-- matrices acting on vectors on the right
#eval  ![1,  1,  1]  ᵥ*  !![1,  2;  3,  4;  5,  6]  -- ![9, 12] 
```

为了生成具有由向量指定的相同行或列的矩阵，我们使用 `Matrix.replicateRow` 和 `Matrix.replicateCol`，其中参数是索引行或列的类型和向量。例如，可以得到单行或单列矩阵（更精确地说，行或列由 `Fin 1` 索引的矩阵）。

```py
#eval  replicateRow  (Fin  1)  ![1,  2]  -- !![1, 2]

#eval  replicateCol  (Fin  1)  ![1,  2]  -- !![1; 2] 
```

其他熟悉的操作包括向量点积、矩阵转置，以及对于方阵，行列式和迹。

```py
-- vector dot product
#eval  ![1,  2]  ⬝ᵥ  ![3,  4]  -- `11`

-- matrix transpose
#eval  !![1,  2;  3,  4]ᵀ  -- `!![1, 3; 2, 4]`

-- determinant
#eval  !![(1  :  ℤ),  2;  3,  4].det  -- `-2`

-- trace
#eval  !![(1  :  ℤ),  2;  3,  4].trace  -- `5` 
```

当条目没有可计算的类型时，例如如果它们是实数，我们无法期望 `#eval` 能有所帮助。此外，这种评估在没有显著扩大可信代码库（即检查证明时需要信任的 Lean 的部分）的情况下不能用于证明。

因此，在证明中也使用 `simp` 和 `norm_num` 策略，或者它们的命令对应物进行快速探索是很好的。

```py
#simp  !![(1  :  ℝ),  2;  3,  4].det  -- `4 - 2*3`

#norm_num  !![(1  :  ℝ),  2;  3,  4].det  -- `-2`

#norm_num  !![(1  :  ℝ),  2;  3,  4].trace  -- `5`

variable  (a  b  c  d  :  ℝ)  in
#simp  !![a,  b;  c,  d].det  -- `a * d – b * c` 
```

对于方阵来说，下一个重要的操作是求逆。与数的除法总是定义且在除以零时返回人工值零一样，求逆操作在所有矩阵上定义，并在不可逆矩阵上返回零矩阵。

更精确地说，存在一个通用函数 `Ring.inverse`，它在任何环中执行此操作，并且对于任何矩阵 `A`，`A⁻¹` 被定义为 `Ring.inverse A.det • A.adjugate`。根据克莱默法则，当 `A` 的行列式不为零时，这确实是 `A` 的逆。

```py
#norm_num  [Matrix.inv_def]  !![(1  :  ℝ),  2;  3,  4]⁻¹  -- !![-2, 1; 3 / 2, -(1 / 2)] 
```

当然，这个定义对于可逆矩阵来说非常有用。有一个通用的类型类`Invertible`有助于记录这一点。例如，下一个例子中的`simp`调用将使用具有`Invertible`类型类假设的`inv_mul_of_invertible`引理，因此只有在类型类合成系统可以找到它的情况下才会触发。在这里，我们使用`have`语句使这一事实可用。

```py
example  :  !![(1  :  ℝ),  2;  3,  4]⁻¹  *  !![(1  :  ℝ),  2;  3,  4]  =  1  :=  by
  have  :  Invertible  !![(1  :  ℝ),  2;  3,  4]  :=  by
  apply  Matrix.invertibleOfIsUnitDet
  norm_num
  simp 
```

在这个完全具体的例子中，我们也可以使用`norm_num`机制和`apply?`来找到最终的行：

```py
example  :  !![(1  :  ℝ),  2;  3,  4]⁻¹  *  !![(1  :  ℝ),  2;  3,  4]  =  1  :=  by
  norm_num  [Matrix.inv_def]
  exact  one_fin_two.symm 
```

所有的具体矩阵都有它们的行和列由`Fin n`索引，其中`n`是某个值（行和列不一定相同）。但有时使用任意有限类型索引矩阵更方便。例如，有限图的邻接矩阵的行和列自然由图的顶点索引。

事实上，当仅仅想要定义矩阵而不在它们上定义任何操作时，索引类型的有限性甚至都不需要，系数可以有任何类型，而不需要任何代数结构。因此，Mathlib 简单地定义`Matrix m n α`为`m → n → α`，对于任何类型`m`、`n`和`α`，我们迄今为止所使用的矩阵类型如`Matrix (Fin 2) (Fin 2) ℝ`。当然，代数运算需要对`m`、`n`和`α`有更多的假设。

注意，我们不直接使用`m → n → α`的主要原因是为了让类型类系统理解我们的意图。例如，对于一个环`R`，类型`n → R`被赋予了点乘操作，同样`m → n → R`也有这种操作，但这并不是我们在矩阵上想要的乘法。

在下面的第一个例子中，我们迫使 Lean 理解`Matrix`的定义，并接受这个陈述是有意义的，然后通过检查所有项来证明它。

但接下来的两个例子揭示，Lean 在`Fin 2 → Fin 2 → ℤ`上使用点乘，但在`Matrix (Fin 2) (Fin 2) ℤ`上使用矩阵乘法。

```py
section

example  :  (fun  _  ↦  1  :  Fin  2  →  Fin  2  →  ℤ)  =  !![1,  1;  1,  1]  :=  by
  ext  i  j
  fin_cases  i  <;>  fin_cases  j  <;>  rfl

example  :  (fun  _  ↦  1  :  Fin  2  →  Fin  2  →  ℤ)  *  (fun  _  ↦  1  :  Fin  2  →  Fin  2  →  ℤ)  =  !![1,  1;  1,  1]  :=  by
  ext  i  j
  fin_cases  i  <;>  fin_cases  j  <;>  rfl

example  :  !![1,  1;  1,  1]  *  !![1,  1;  1,  1]  =  !![2,  2;  2,  2]  :=  by
  norm_num 
```

为了将矩阵定义为函数而不失去`Matrix`在类型类合成中的优势，我们可以使用函数和矩阵之间的等价性`Matrix.of`。这个等价性是秘密地使用`Equiv.refl`定义的。

例如，我们可以定义与向量`v`对应的 Vandermonde 矩阵。

```py
example  {n  :  ℕ}  (v  :  Fin  n  →  ℝ)  :
  Matrix.vandermonde  v  =  Matrix.of  (fun  i  j  :  Fin  n  ↦  v  i  ^  (j  :  ℕ))  :=
  rfl
end
end  matrices 
```

### 10.4.2. 基

现在，我们想要讨论向量空间的基。非正式地说，有许多定义这个概念的方法。可以使用一个通用性质。可以说基是一组线性无关且张成的向量族。或者可以将这些性质结合起来，直接说基是一组向量，每个向量都可以唯一地表示为基向量的线性组合。还有一种说法是，基提供了一个与基域`K`的幂的线性同构，将`K`视为`K`上的向量空间。

这个同构版本实际上是 Mathlib 在底层用作定义的版本，其他特征化都是从这个版本证明出来的。在无限基的情况下，必须对“`K` 的幂”这个概念稍加小心。确实，在这个代数背景下，只有有限线性组合才有意义。因此，我们需要作为参考向量空间的不应该是 `K` 的直接积，而是一个直接和。我们可以用 `⨁ i : ι, K` 表示某个索引基的 `ι` 类型，但我们更倾向于使用更专业的表示 `ι →₀ K`，这意味着“从 `ι` 到 `K` 的有限支撑函数”，即函数在 `ι` 的有限集合之外为零（这个有限集合不是固定的，它取决于函数）。将来自基 `B` 的这样一个函数在向量 `v` 和 `i : ι` 上进行评估，返回 `v` 在第 `i` 个基向量上的分量（或坐标）。

作为 `V` 作为 `K` 向量空间的基的索引类型 `ι` 的类型是 `Basis ι K V`。这个同构被称为 `Basis.repr`。

```py
variable  {K  :  Type*}  [Field  K]  {V  :  Type*}  [AddCommGroup  V]  [Module  K  V]

section

variable  {ι  :  Type*}  (B  :  Basis  ι  K  V)  (v  :  V)  (i  :  ι)

-- The basis vector with index ``i``
#check  (B  i  :  V)

-- the linear isomorphism with the model space given by ``B``
#check  (B.repr  :  V  ≃ₗ[K]  ι  →₀  K)

-- the component function of ``v``
#check  (B.repr  v  :  ι  →₀  K)

-- the component of ``v`` with index ``i``
#check  (B.repr  v  i  :  K) 
```

可以从这样一个同构开始，也可以从一个线性无关且生成的一组向量 `b` 开始，这就是 `Basis.mk`。

将族是生成性的假设表述为 `⊤ ≤ Submodule.span K (Set.range b)`。这里 `⊤` 是 `V` 的上子模，即 `V` 作为自身的子模来看待。这种表述看起来有点复杂，但下面我们将看到它几乎等同于更易读的 `∀ v, v ∈ Submodule.span K (Set.range b)`（下面片段中的下划线指的是无用的信息 `v ∈ ⊤`）。

```py
noncomputable  example  (b  :  ι  →  V)  (b_indep  :  LinearIndependent  K  b)
  (b_spans  :  ∀  v,  v  ∈  Submodule.span  K  (Set.range  b))  :  Basis  ι  K  V  :=
  Basis.mk  b_indep  (fun  v  _  ↦  b_spans  v)

-- The family of vectors underlying the above basis is indeed ``b``.
example  (b  :  ι  →  V)  (b_indep  :  LinearIndependent  K  b)
  (b_spans  :  ∀  v,  v  ∈  Submodule.span  K  (Set.range  b))  (i  :  ι)  :
  Basis.mk  b_indep  (fun  v  _  ↦  b_spans  v)  i  =  b  i  :=
  Basis.mk_apply  b_indep  (fun  v  _  ↦  b_spans  v)  i 
```

特别地，模型向量空间 `ι →₀ K` 有一个所谓的规范基，其 `repr` 函数在任意向量上的评估是恒等同构。它被称为 `Finsupp.basisSingleOne`，其中 `Finsupp` 表示具有有限支撑的函数，而 `basisSingleOne` 指的是基向量是除了单个输入值外都为零的函数。更精确地说，由 `i : ι` 索引的基向量是 `Finsupp.single i 1`，这是一个在 `i` 处取值为 `1` 而在其他所有地方取值为 `0` 的有限支撑函数。

```py
variable  [DecidableEq  ι] 
```

```py
example  :  Finsupp.basisSingleOne.repr  =  LinearEquiv.refl  K  (ι  →₀  K)  :=
  rfl

example  (i  :  ι)  :  Finsupp.basisSingleOne  i  =  Finsupp.single  i  1  :=
  rfl 
```

当索引类型是有限的时候，有限支撑函数的故事是不必要的。在这种情况下，我们可以使用更简单的 `Pi.basisFun`，它给出整个 `ι → K` 的基。

```py
example  [Finite  ι]  (x  :  ι  →  K)  (i  :  ι)  :  (Pi.basisFun  K  ι).repr  x  i  =  x  i  :=  by
  simp 
```

回到抽象向量空间基的一般情况，我们可以将任何向量表示为基向量的线性组合。让我们先看看有限基的简单情况。

```py
example  [Fintype  ι]  :  ∑  i  :  ι,  B.repr  v  i  •  (B  i)  =  v  :=
  B.sum_repr  v 
```

当 `ι` 不是有限时，上述陈述在先验上没有意义：我们不能对 `ι` 进行求和。然而，被求和函数的支持是有限的（它是 `B.repr v` 的支持）。但是我们需要应用一个考虑这一点的构造。在这里，Mathlib 使用了一个特殊目的的函数，需要一些时间来习惯：`Finsupp.linearCombination`（它建立在更一般的 `Finsupp.sum` 之上）。给定一个有限支持的函数 `c` 从类型 `ι` 到基域 `K` 以及任何从 `ι` 到 `V` 的函数 `f`，`Finsupp.linearCombination K f c` 是对 `c` 的支持上的标量乘积 `c • f` 的求和。特别是，我们可以将其替换为包含 `c` 的支持中的任何有限集上的求和。

```py
example  (c  :  ι  →₀  K)  (f  :  ι  →  V)  (s  :  Finset  ι)  (h  :  c.support  ⊆  s)  :
  Finsupp.linearCombination  K  f  c  =  ∑  i  ∈  s,  c  i  •  f  i  :=
  Finsupp.linearCombination_apply_of_mem_supported  K  h 
```

也可以假设 `f` 是有限支持的，并且仍然得到一个定义良好的和。但是 `Finsupp.linearCombination` 所做的选择与我们的基讨论相关，因为它允许陈述 `Basis.sum_repr` 的一般化。

```py
example  :  Finsupp.linearCombination  K  B  (B.repr  v)  =  v  :=
  B.linearCombination_repr  v 
```

也有人可能会想知道为什么 `K` 是一个显式的参数，尽管它可以从 `c` 的类型中推断出来。关键是部分应用 `Finsupp.linearCombination K f` 本身就很有趣。它不是一个从 `ι →₀ K` 到 `V` 的裸函数，而是一个 `K`-线性映射。

```py
variable  (f  :  ι  →  V)  in
#check  (Finsupp.linearCombination  K  f  :  (ι  →₀  K)  →ₗ[K]  V) 
```

回到数学讨论，重要的是要理解，在形式化数学中，基中向量的表示不如你可能想象的那么有用。事实上，直接使用基的更抽象性质通常更有效。特别是，基的普遍性质将它们与其他代数中的自由对象连接起来，允许通过指定基向量的像来构造线性映射。这是 `Basis.constr`。对于任何 `K`-向量空间 `W`，我们的基 `B` 给出一个线性同构 `Basis.constr B K` 从 `ι → W` 到 `V →ₗ[K] W`。这个同构的特点是它将任何函数 `u : ι → W` 映射到一个线性映射，该映射将基向量 `B i` 映射到 `u i`，对于每个 `i : ι`。

```py
section

variable  {W  :  Type*}  [AddCommGroup  W]  [Module  K  W]
  (φ  :  V  →ₗ[K]  W)  (u  :  ι  →  W)

#check  (B.constr  K  :  (ι  →  W)  ≃ₗ[K]  (V  →ₗ[K]  W))

#check  (B.constr  K  u  :  V  →ₗ[K]  W)

example  (i  :  ι)  :  B.constr  K  u  (B  i)  =  u  i  :=
  B.constr_basis  K  u  i 
```

这个属性确实是特征性的，因为线性映射由其在基上的值决定：

```py
example  (φ  ψ  :  V  →ₗ[K]  W)  (h  :  ∀  i,  φ  (B  i)  =  ψ  (B  i))  :  φ  =  ψ  :=
  B.ext  h 
```

如果我们还在目标空间上有基 `B'`，那么我们可以将线性映射与矩阵相识别。这种识别是一个 `K`-线性同构。

```py
variable  {ι'  :  Type*}  (B'  :  Basis  ι'  K  W)  [Fintype  ι]  [DecidableEq  ι]  [Fintype  ι']  [DecidableEq  ι']

open  LinearMap

#check  (toMatrix  B  B'  :  (V  →ₗ[K]  W)  ≃ₗ[K]  Matrix  ι'  ι  K)

open  Matrix  -- get access to the ``*ᵥ`` notation for multiplication between matrices and vectors.

example  (φ  :  V  →ₗ[K]  W)  (v  :  V)  :  (toMatrix  B  B'  φ)  *ᵥ  (B.repr  v)  =  B'.repr  (φ  v)  :=
  toMatrix_mulVec_repr  B  B'  φ  v

variable  {ι''  :  Type*}  (B''  :  Basis  ι''  K  W)  [Fintype  ι'']  [DecidableEq  ι'']

example  (φ  :  V  →ₗ[K]  W)  :  (toMatrix  B  B''  φ)  =  (toMatrix  B'  B''  .id)  *  (toMatrix  B  B'  φ)  :=  by
  simp

end 
```

作为这个主题的练习，我们将证明定理的一部分，该定理保证了内射映射有一个定义良好的行列式。具体来说，我们想要证明当两个基由相同的类型索引时，它们附加到任何内射映射上的矩阵具有相同的行列式。然后需要使用所有基都具有同构索引类型的事实来补充这一结果。

当然，Mathlib 已经知道这一点，`simp` 可以立即关闭目标，所以你不应该过早地使用它，而应该使用提供的引理。

```py
open  Module  LinearMap  Matrix

-- Some lemmas coming from the fact that `LinearMap.toMatrix` is an algebra morphism.
#check  toMatrix_comp
#check  id_comp
#check  comp_id
#check  toMatrix_id

-- Some lemmas coming from the fact that ``Matrix.det`` is a multiplicative monoid morphism.
#check  Matrix.det_mul
#check  Matrix.det_one

example  [Fintype  ι]  (B'  :  Basis  ι  K  V)  (φ  :  End  K  V)  :
  (toMatrix  B  B  φ).det  =  (toMatrix  B'  B'  φ).det  :=  by
  set  M  :=  toMatrix  B  B  φ
  set  M'  :=  toMatrix  B'  B'  φ
  set  P  :=  (toMatrix  B  B')  LinearMap.id
  set  P'  :=  (toMatrix  B'  B)  LinearMap.id
  sorry
end 
```

### 10.4.3\. 维度

回到单个向量空间的情况，基也用于定义维度概念。在这里，有限维向量空间是基本的情况。对于这样的空间，我们期望维度是一个自然数。这是 `Module.finrank`。它将基域作为显式参数，因为一个给定的阿贝尔群可以是在不同域上的向量空间。

```py
section
#check  (Module.finrank  K  V  :  ℕ)

-- `Fin n → K` is the archetypical space with dimension `n` over `K`.
example  (n  :  ℕ)  :  Module.finrank  K  (Fin  n  →  K)  =  n  :=
  Module.finrank_fin_fun  K

-- Seen as a vector space over itself, `ℂ` has dimension one.
example  :  Module.finrank  ℂ  ℂ  =  1  :=
  Module.finrank_self  ℂ

-- But as a real vector space it has dimension two.
example  :  Module.finrank  ℝ  ℂ  =  2  :=
  Complex.finrank_real_complex 
```

注意，`Module.finrank` 定义了任何向量空间。对于无限维向量空间，它返回零，就像除以零返回零一样。

当然，许多引理需要有限维度的假设。这就是 `FiniteDimensional` 类型类的作用。例如，考虑下一个例子在没有这个假设的情况下是如何失败的。

```py
example  [FiniteDimensional  K  V]  :  0  <  Module.finrank  K  V  ↔  Nontrivial  V  :=
  Module.finrank_pos_iff 
```

在上述陈述中，`Nontrivial V` 表示 `V` 至少有两个不同的元素。请注意，`Module.finrank_pos_iff` 没有显式的参数。当从左到右使用它时这是可以的，但不是从右到左，因为 Lean 没有从 `Nontrivial V` 这个陈述中猜测 `K` 的方法。在这种情况下，使用名称参数语法是有用的，前提是已经确认引理是在一个名为 `R` 的环上声明的。因此，我们可以写出：

```py
example  [FiniteDimensional  K  V]  (h  :  0  <  Module.finrank  K  V)  :  Nontrivial  V  :=  by
  apply  (Module.finrank_pos_iff  (R  :=  K)).1
  exact  h 
```

上述拼写很奇怪，因为我们已经有了 `h` 作为假设，所以我们完全可以给出完整的证明 `Module.finrank_pos_iff.1 h`，但在更复杂的情况下了解这一点是好的。

根据定义，`FiniteDimensional K V` 可以从任何基中读取。

```py
variable  {ι  :  Type*}  (B  :  Basis  ι  K  V)

example  [Finite  ι]  :  FiniteDimensional  K  V  :=  FiniteDimensional.of_fintype_basis  B

example  [FiniteDimensional  K  V]  :  Finite  ι  :=
  (FiniteDimensional.fintypeBasisIndex  B).finite
end 
```

利用对应于线性子空间的子类型具有向量空间结构，我们可以讨论子空间的维度。

```py
section
variable  (E  F  :  Submodule  K  V)  [FiniteDimensional  K  V]

open  Module

example  :  finrank  K  (E  ⊔  F  :  Submodule  K  V)  +  finrank  K  (E  ⊓  F  :  Submodule  K  V)  =
  finrank  K  E  +  finrank  K  F  :=
  Submodule.finrank_sup_add_finrank_inf_eq  E  F

example  :  finrank  K  E  ≤  finrank  K  V  :=  Submodule.finrank_le  E 
```

在上述第一个陈述中，类型赋值的目的是确保将强制转换为 `Type*` 不会太早触发。

我们现在可以准备一个关于 `finrank` 和子空间的练习。

```py
example  (h  :  finrank  K  V  <  finrank  K  E  +  finrank  K  F)  :
  Nontrivial  (E  ⊓  F  :  Submodule  K  V)  :=  by
  sorry
end 
```

现在我们转向维度理论的一般情况。在这种情况下，`finrank` 是无用的，但我们仍然有，对于同一向量空间的两个基，它们之间的类型索引存在一个双射。因此，我们仍然可以希望将秩定义为基数，即“在存在双射等价关系下的类型集合的商”中的一个元素。

当讨论基数时，就像在这本书的其他地方一样，很难忽视围绕 Russell 的悖论的基础性问题。没有所有类型的类型，因为这会导致逻辑不一致。这个问题通过我们通常试图忽略的宇宙层次结构得到解决。

每个类型都有一个宇宙级别，这些级别的行为类似于自然数。特别是有一个零级，相应的宇宙 `Type 0` 简单地表示为 `Type`。这个宇宙足以容纳几乎所有经典数学。例如 `ℕ` 和 `ℝ` 有 `Type` 类型。每个级别 `u` 有一个后继，表示为 `u + 1`，而 `Type u` 有 `Type (u+1)` 类型。

但宇宙级别不是自然数，它们具有非常不同的性质，并且没有类型。特别是，你无法在 Lean 中陈述类似于`u ≠ u + 1`的东西。根本不存在这样的类型。甚至陈述`Type u ≠ Type (u+1)`也没有意义，因为`Type u`和`Type (u+1)`具有不同的类型。

每当我们写`Type*`时，Lean 会插入一个名为`u_n`的宇宙级别变量，其中`n`是一个数字。这允许定义和陈述存在于所有宇宙中。

给定一个宇宙级别`u`，我们可以在`Type u`上定义一个等价关系，如果两个类型`α`和`β`之间存在双射，则称这两个类型`α`和`β`是等价的。商类型`Cardinal.{u}`生活在`Type (u+1)`中。大括号表示一个宇宙变量。在这个商中，`α : Type u`的像是`Cardinal.mk α : Cardinal.{u}`。

但我们无法直接比较不同宇宙中的基数。因此，技术上我们无法将向量空间`V`的秩定义为所有索引`V`的基的类型集合的基数。因此，它被定义为`V`中所有线性无关集合的基数`Module.rank K V`的上确界。如果`V`的宇宙级别为`u`，则其秩具有类型`Cardinal.{u}`。

```py
#check  V  -- Type u_2
#check  Module.rank  K  V  -- Cardinal.{u_2} 
```

尽管如此，我们仍然可以将这个定义与基联系起来。确实，在宇宙级别上也有一个交换的`max`操作，并且对于两个宇宙级别`u`和`v`，存在一个操作`Cardinal.lift.{u, v} : Cardinal.{v} → Cardinal.{max v u}`，它允许将基数放入一个共同的宇宙中并陈述维度定理。

```py
universe  u  v  -- `u` and `v` will denote universe levels

variable  {ι  :  Type  u}  (B  :  Basis  ι  K  V)
  {ι'  :  Type  v}  (B'  :  Basis  ι'  K  V)

example  :  Cardinal.lift.{v,  u}  (.mk  ι)  =  Cardinal.lift.{u,  v}  (.mk  ι')  :=
  mk_eq_mk_of_basis  B  B' 
```

我们可以使用从自然数到有限基数的强制转换（或者更精确地说，是生活在`Cardinal.{v}`中的有限基数，其中`v`是`V`的宇宙级别）将有限维情况与这次讨论联系起来。

```py
example  [FiniteDimensional  K  V]  :
  (Module.finrank  K  V  :  Cardinal)  =  Module.rank  K  V  :=
  Module.finrank_eq_rank  K  V 
```

## 10.1\. 向量空间和线性映射

### 10.1.1\. 向量空间

我们将直接从抽象线性代数开始，它发生在任何域上的向量空间中。然而，你可以在第 10.4.1 节中找到有关矩阵的信息，这部分内容与抽象理论无关。Mathlib 实际上处理的是涉及模块一词的更一般版本的线性代数，但就目前而言，我们将假装这仅仅是一种古怪的拼写习惯。

要说“设$K$为一个域，设$V$为$K$上的向量空间”（并将它们作为后续结果的隐含参数）的方式是：

```py
variable  {K  :  Type*}  [Field  K]  {V  :  Type*}  [AddCommGroup  V]  [Module  K  V] 
```

我们在第八章中解释了为什么我们需要两个单独的类型类 `[AddCommGroup V] [Module K V]`。简而言之，数学上我们想要表达的是，具有 $K$ 向量空间结构意味着具有加法交换群结构。我们可以告诉 Lean。但然后每当 Lean 需要在类型 $V$ 上找到这样的群结构时，它就会使用一个*完全未指定的*字段 $K$ 来寻找向量空间结构，而这个字段无法从 $V$ 推断出来。这对类型类综合系统来说是非常糟糕的。

向量 $v$ 与标量 $a$ 的乘积表示为 a • v。我们在以下示例中列出了一些关于此操作与加法交互的代数规则。当然，simp 或 apply? 会找到这些证明。还有一个模块策略，它解决从向量空间和域的公理中得出的目标，就像在交换环中使用环策略或在群中使用群策略一样。但仍然有用的是记住标量乘法在引理名称中简写为 smul。

```py
example  (a  :  K)  (u  v  :  V)  :  a  •  (u  +  v)  =  a  •  u  +  a  •  v  :=
  smul_add  a  u  v

example  (a  b  :  K)  (u  :  V)  :  (a  +  b)  •  u  =  a  •  u  +  b  •  u  :=
  add_smul  a  b  u

example  (a  b  :  K)  (u  :  V)  :  a  •  b  •  u  =  b  •  a  •  u  :=
  smul_comm  a  b  u 
```

作为对更高级读者的快速提示，让我们指出，正如术语所暗示的，Mathlib 的线性代数也涵盖了（不一定交换的）环上的模。实际上，它甚至涵盖了半环上的半模。如果你认为你不需要这种程度的普遍性，你可以冥想以下示例，它很好地捕捉了许多关于理想在子模上作用的代数规则：

```py
example  {R  M  :  Type*}  [CommSemiring  R]  [AddCommMonoid  M]  [Module  R  M]  :
  Module  (Ideal  R)  (Submodule  R  M)  :=
  inferInstance 
```

### 10.1.2\. 线性映射

接下来我们需要线性映射。像群同态一样，Mathlib 中的线性映射是打包映射，即由映射及其线性性质证明组成的包。这些打包映射在应用时被转换为普通函数。有关此设计的更多信息，请参阅第八章。

两个 `K` 向量空间 `V` 和 `W` 之间线性映射的类型表示为 `V →ₗ[K] W`。下标 l 代表线性。一开始可能觉得在这个符号中指定 `K` 很奇怪。但这是几个领域同时出现时的关键。例如，从 $ℂ$ 到 $ℂ$ 的实线性映射是每个映射 $z ↦ az + b\bar{z}$，而只有映射 $z ↦ az$ 是复线性，这种差异在复分析中是至关重要的。

```py
variable  {W  :  Type*}  [AddCommGroup  W]  [Module  K  W]

variable  (φ  :  V  →ₗ[K]  W)

example  (a  :  K)  (v  :  V)  :  φ  (a  •  v)  =  a  •  φ  v  :=
  map_smul  φ  a  v

example  (v  w  :  V)  :  φ  (v  +  w)  =  φ  v  +  φ  w  :=
  map_add  φ  v  w 
```

注意，`V →ₗ[K] W` 本身承载着有趣的代数结构（这是打包这些映射的部分动机）。它是一个 $K$ 向量空间，因此我们可以添加线性映射并乘以标量。

```py
variable  (ψ  :  V  →ₗ[K]  W)

#check  (2  •  φ  +  ψ  :  V  →ₗ[K]  W) 
```

使用打包映射的一个缺点是我们不能使用普通函数组合。我们需要使用 `LinearMap.comp` 或 `∘ₗ` 的符号。

```py
variable  (θ  :  W  →ₗ[K]  V)

#check  (φ.comp  θ  :  W  →ₗ[K]  W)
#check  (φ  ∘ₗ  θ  :  W  →ₗ[K]  W) 
```

构建线性映射主要有两种方法。首先，我们可以通过提供函数和线性证明来构建结构。通常，这可以通过结构代码操作来实现：你可以输入 `example : V →ₗ[K] V := _` 并使用附加在下划线上的“生成骨架”代码操作。

```py
example  :  V  →ₗ[K]  V  where
  toFun  v  :=  3  •  v
  map_add'  _  _  :=  smul_add  ..
  map_smul'  _  _  :=  smul_comm  .. 
```

你可能会想知道为什么 `LinearMap` 的证明字段以撇号结尾。这是因为它们在定义函数强制转换之前就已经定义了，因此它们是以 `LinearMap.toFun` 为术语表述的。然后，它们被重新表述为 `LinearMap.map_add` 和 `LinearMap.map_smul`，以函数强制转换的术语表述。但这还不是故事的结尾。人们还希望有一个适用于任何（捆绑的）保持加法的映射的 `map_add` 版本，例如加法群同态、线性映射、连续线性映射、`K`-代数映射等。这个版本是 `map_add`（在根命名空间中）。中间版本 `LinearMap.map_add` 有点冗余，但允许使用点符号，这在某些时候可能很方便。对于 `map_smul` 也有类似的故事，并且通用框架在第八章中解释。

```py
#check  (φ.map_add'  :  ∀  x  y  :  V,  φ.toFun  (x  +  y)  =  φ.toFun  x  +  φ.toFun  y)
#check  (φ.map_add  :  ∀  x  y  :  V,  φ  (x  +  y)  =  φ  x  +  φ  y)
#check  (map_add  φ  :  ∀  x  y  :  V,  φ  (x  +  y)  =  φ  x  +  φ  y) 
```

还可以从 Mathlib 中已定义的映射使用各种组合器构建线性映射。例如，上面的例子已经知道是 `LinearMap.lsmul K V 3`。`K` 和 `V` 在这里作为显式参数有几个原因。最紧迫的一个原因是，从裸 `LinearMap.lsmul 3` 中，Lean 无法推断出 `V` 或甚至 `K`。但 `LinearMap.lsmul K V` 本身也是一个有趣的对象：它具有类型 `K →ₗ[K] V →ₗ[K] V`，这意味着它是一个从 `K`（被视为自身的向量空间）到 `V` 到 `V` 的 `K`-线性映射。

```py
#check  (LinearMap.lsmul  K  V  3  :  V  →ₗ[K]  V)
#check  (LinearMap.lsmul  K  V  :  K  →ₗ[K]  V  →ₗ[K]  V) 
```

还有一个表示线性同构的 `LinearEquiv` 类型，表示为 `V ≃ₗ[K] W`。`f : V ≃ₗ[K] W` 的逆是 `f.symm : W ≃ₗ[K] V`，`f` 和 `g` 的组合是 `f.trans g`，也称为 `f ≪≫ₗ g`，恒等同构是 `V` 的 `LinearEquiv.refl K V`。当需要时，此类型的元素会自动转换为形态和函数。

```py
example  (f  :  V  ≃ₗ[K]  W)  :  f  ≪≫ₗ  f.symm  =  LinearEquiv.refl  K  V  :=
  f.self_trans_symm 
```

可以使用 `LinearEquiv.ofBijective` 从双射映射构建同构。这样做会使逆函数不可计算。

```py
noncomputable  example  (f  :  V  →ₗ[K]  W)  (h  :  Function.Bijective  f)  :  V  ≃ₗ[K]  W  :=
  .ofBijective  f  h 
```

注意，在上面的例子中，Lean 使用宣布的类型来理解 `.ofBijective` 指的是 `LinearEquiv.ofBijective`（无需打开任何命名空间）。

### 10.1.3. 向量空间的和与积

我们可以使用直接和和直接积从旧向量空间构建新的向量空间。让我们从两个向量空间开始。在这种情况下，和与积之间没有区别，我们可以简单地使用积类型。在下面的代码片段中，我们简单地展示了如何将所有结构映射（包含和投射）作为线性映射来获取，以及构建线性映射到积和从和的通用性质（如果你不熟悉加和与积之间的范畴论区别，你可以简单地忽略通用性质词汇，并关注以下示例的类型）。

```py
section  binary_product

variable  {W  :  Type*}  [AddCommGroup  W]  [Module  K  W]
variable  {U  :  Type*}  [AddCommGroup  U]  [Module  K  U]
variable  {T  :  Type*}  [AddCommGroup  T]  [Module  K  T]

-- First projection map
example  :  V  ×  W  →ₗ[K]  V  :=  LinearMap.fst  K  V  W

-- Second projection map
example  :  V  ×  W  →ₗ[K]  W  :=  LinearMap.snd  K  V  W

-- Universal property of the product
example  (φ  :  U  →ₗ[K]  V)  (ψ  :  U  →ₗ[K]  W)  :  U  →ₗ[K]  V  ×  W  :=  LinearMap.prod  φ  ψ

-- The product map does the expected thing, first component
example  (φ  :  U  →ₗ[K]  V)  (ψ  :  U  →ₗ[K]  W)  :  LinearMap.fst  K  V  W  ∘ₗ  LinearMap.prod  φ  ψ  =  φ  :=  rfl

-- The product map does the expected thing, second component
example  (φ  :  U  →ₗ[K]  V)  (ψ  :  U  →ₗ[K]  W)  :  LinearMap.snd  K  V  W  ∘ₗ  LinearMap.prod  φ  ψ  =  ψ  :=  rfl

-- We can also combine maps in parallel
example  (φ  :  V  →ₗ[K]  U)  (ψ  :  W  →ₗ[K]  T)  :  (V  ×  W)  →ₗ[K]  (U  ×  T)  :=  φ.prodMap  ψ

-- This is simply done by combining the projections with the universal property
example  (φ  :  V  →ₗ[K]  U)  (ψ  :  W  →ₗ[K]  T)  :
  φ.prodMap  ψ  =  (φ  ∘ₗ  .fst  K  V  W).prod  (ψ  ∘ₗ  .snd  K  V  W)  :=  rfl

-- First inclusion map
example  :  V  →ₗ[K]  V  ×  W  :=  LinearMap.inl  K  V  W

-- Second inclusion map
example  :  W  →ₗ[K]  V  ×  W  :=  LinearMap.inr  K  V  W

-- Universal property of the sum (aka coproduct)
example  (φ  :  V  →ₗ[K]  U)  (ψ  :  W  →ₗ[K]  U)  :  V  ×  W  →ₗ[K]  U  :=  φ.coprod  ψ

-- The coproduct map does the expected thing, first component
example  (φ  :  V  →ₗ[K]  U)  (ψ  :  W  →ₗ[K]  U)  :  φ.coprod  ψ  ∘ₗ  LinearMap.inl  K  V  W  =  φ  :=
  LinearMap.coprod_inl  φ  ψ

-- The coproduct map does the expected thing, second component
example  (φ  :  V  →ₗ[K]  U)  (ψ  :  W  →ₗ[K]  U)  :  φ.coprod  ψ  ∘ₗ  LinearMap.inr  K  V  W  =  ψ  :=
  LinearMap.coprod_inr  φ  ψ

-- The coproduct map is defined in the expected way
example  (φ  :  V  →ₗ[K]  U)  (ψ  :  W  →ₗ[K]  U)  (v  :  V)  (w  :  W)  :
  φ.coprod  ψ  (v,  w)  =  φ  v  +  ψ  w  :=
  rfl

end  binary_product 
```

现在，让我们转向任意向量空间族的和与积。在这里，我们将简单地看到如何定义一个向量空间族，并访问和与积的通用性质。请注意，直接和的表示法是作用域在`DirectSum`命名空间中，并且直接和的通用性质要求索引类型的可判定等价性（这某种程度上是一个实现上的意外）。

```py
section  families
open  DirectSum

variable  {ι  :  Type*}  [DecidableEq  ι]
  (V  :  ι  →  Type*)  [∀  i,  AddCommGroup  (V  i)]  [∀  i,  Module  K  (V  i)]

-- The universal property of the direct sum assembles maps from the summands to build
-- a map from the direct sum
example  (φ  :  Π  i,  (V  i  →ₗ[K]  W))  :  (⨁  i,  V  i)  →ₗ[K]  W  :=
  DirectSum.toModule  K  ι  W  φ

-- The universal property of the direct product assembles maps into the factors
-- to build a map into the direct product
example  (φ  :  Π  i,  (W  →ₗ[K]  V  i))  :  W  →ₗ[K]  (Π  i,  V  i)  :=
  LinearMap.pi  φ

-- The projection maps from the product
example  (i  :  ι)  :  (Π  j,  V  j)  →ₗ[K]  V  i  :=  LinearMap.proj  i

-- The inclusion maps into the sum
example  (i  :  ι)  :  V  i  →ₗ[K]  (⨁  i,  V  i)  :=  DirectSum.lof  K  ι  V  i

-- The inclusion maps into the product
example  (i  :  ι)  :  V  i  →ₗ[K]  (Π  i,  V  i)  :=  LinearMap.single  K  V  i

-- In case `ι` is a finite type, there is an isomorphism between the sum and product.
example  [Fintype  ι]  :  (⨁  i,  V  i)  ≃ₗ[K]  (Π  i,  V  i)  :=
  linearEquivFunOnFintype  K  ι  V

end  families 
```

### 10.1.1\. 向量空间

我们将直接开始抽象线性代数，它发生在任何域上的向量空间中。然而，你可以在第 10.4.1 节中找到有关矩阵的信息，这部分内容逻辑上不依赖于这个抽象理论。Mathlib 实际上处理的是一个更通用的线性代数版本，涉及到“module”这个词，但到目前为止，我们将假装这仅仅是一个古怪的拼写习惯。

要说“让$K$成为一个域，让$V$成为$K$上的向量空间”（并将它们作为后续结果的隐含参数）的方式是：

```py
variable  {K  :  Type*}  [Field  K]  {V  :  Type*}  [AddCommGroup  V]  [Module  K  V] 
```

我们在第八章中解释了为什么我们需要两个独立的类型类`[AddCommGroup V] [Module K V]`。简而言之，从数学上讲，我们想要表达拥有$K$-向量空间结构意味着拥有加法交换群结构。我们可以告诉 Lean。但然后每当 Lean 需要在一个类型$V$上找到这样的群结构时，它就会使用一个*完全未指定*的域$K$来寻找向量空间结构，而这个域$K$无法从$V$中推断出来。这对类型类综合系统来说是非常糟糕的。

向量 v 乘以标量 a 表示为 a • v。我们在以下示例中列出了一些关于此操作与加法交互的代数规则。当然，simp 或 apply?会找到那些证明。还有一个模块策略，它解决从向量空间和域的公理得出的目标，就像在交换环中使用 ring 策略或在群中使用 group 策略一样。但记住标量乘法在引理名称中缩写为 smul 仍然是有用的。

```py
example  (a  :  K)  (u  v  :  V)  :  a  •  (u  +  v)  =  a  •  u  +  a  •  v  :=
  smul_add  a  u  v

example  (a  b  :  K)  (u  :  V)  :  (a  +  b)  •  u  =  a  •  u  +  b  •  u  :=
  add_smul  a  b  u

example  (a  b  :  K)  (u  :  V)  :  a  •  b  •  u  =  b  •  a  •  u  :=
  smul_comm  a  b  u 
```

作为对更高级读者的快速提示，让我们指出，正如术语所暗示的，Mathlib 的线性代数也涵盖了（不一定交换的）环上的模块。事实上，它甚至涵盖了半环上的半模。如果你认为你不需要这种泛化程度，你可以思考以下示例，它很好地捕捉了关于理想在子模上作用的许多代数规则：

```py
example  {R  M  :  Type*}  [CommSemiring  R]  [AddCommMonoid  M]  [Module  R  M]  :
  Module  (Ideal  R)  (Submodule  R  M)  :=
  inferInstance 
```

### 10.1.2\. 线性映射

接下来我们需要线性映射。与群同态类似，Mathlib 中的线性映射是打包映射，即由映射及其线性性质证明组成的包。当应用时，这些打包映射会被转换为普通函数。有关此设计的更多信息，请参阅第八章。

两个 `K`-向量空间 `V` 和 `W` 之间的线性映射类型表示为 `V →ₗ[K] W`。下标 l 代表线性。一开始可能觉得在这个符号中指定 `K` 很奇怪。但这是至关重要的，当涉及多个域时。例如，从 $ℂ$ 到 $ℂ$ 的实线性映射是所有形式为 $z ↦ az + b\bar{z}$ 的映射，而只有形式为 $z ↦ az$ 的映射是复线性，这种差异在复分析中至关重要。

```py
variable  {W  :  Type*}  [AddCommGroup  W]  [Module  K  W]

variable  (φ  :  V  →ₗ[K]  W)

example  (a  :  K)  (v  :  V)  :  φ  (a  •  v)  =  a  •  φ  v  :=
  map_smul  φ  a  v

example  (v  w  :  V)  :  φ  (v  +  w)  =  φ  v  +  φ  w  :=
  map_add  φ  v  w 
```

注意，`V →ₗ[K] W` 本身承载着有趣的代数结构（这是打包这些映射的部分动机）。它是一个 `K`-向量空间，因此我们可以添加线性映射并乘以标量。

```py
variable  (ψ  :  V  →ₗ[K]  W)

#check  (2  •  φ  +  ψ  :  V  →ₗ[K]  W) 
```

使用打包映射的一个缺点是我们不能使用普通函数组合。我们需要使用 `LinearMap.comp` 或 `∘ₗ` 符号。

```py
variable  (θ  :  W  →ₗ[K]  V)

#check  (φ.comp  θ  :  W  →ₗ[K]  W)
#check  (φ  ∘ₗ  θ  :  W  →ₗ[K]  W) 
```

构建线性映射主要有两种方式。首先，我们可以通过提供函数和线性证明来构建结构。像往常一样，这可以通过结构代码操作来实现：你可以输入 `example : V →ₗ[K] V := _` 并使用附加在下划线上的“生成骨架”代码操作。

```py
example  :  V  →ₗ[K]  V  where
  toFun  v  :=  3  •  v
  map_add'  _  _  :=  smul_add  ..
  map_smul'  _  _  :=  smul_comm  .. 
```

你可能会想知道为什么 `LinearMap` 的证明字段以撇号结尾。这是因为它们在定义函数强制转换之前就已经定义了，因此它们是以 `LinearMap.toFun` 为术语表述的。然后，它们被重新表述为 `LinearMap.map_add` 和 `LinearMap.map_smul`，以函数强制转换的术语表述。但这还不是故事的结束。人们还希望有一个适用于任何（打包）保持加法的映射的 `map_add` 版本，例如加法群同态、线性映射、连续线性映射、`K`-代数映射等。这个版本是 `map_add`（在根命名空间中）。中间版本 `LinearMap.map_add` 有点冗余，但允许使用点符号，这在某些时候可能很方便。对于 `map_smul` 也有类似的故事，而通用框架在第八章中得到了解释。

```py
#check  (φ.map_add'  :  ∀  x  y  :  V,  φ.toFun  (x  +  y)  =  φ.toFun  x  +  φ.toFun  y)
#check  (φ.map_add  :  ∀  x  y  :  V,  φ  (x  +  y)  =  φ  x  +  φ  y)
#check  (map_add  φ  :  ∀  x  y  :  V,  φ  (x  +  y)  =  φ  x  +  φ  y) 
```

还可以从 Mathlib 中已定义的线性映射中使用各种组合子构建线性映射。例如，上面的例子已经众所周知为 `LinearMap.lsmul K V 3`。`K` 和 `V` 在这里作为显式参数有几个原因。最紧迫的一个原因是，从裸的 `LinearMap.lsmul 3` 中，Lean 无法推断出 `V` 或甚至 `K`。但 `LinearMap.lsmul K V` 本身也是一个有趣的对象：它具有类型 `K →ₗ[K] V →ₗ[K] V`，这意味着它是一个从 `K`（被视为自身的向量空间）到 `V` 到 `V` 的 `K`-线性映射。

```py
#check  (LinearMap.lsmul  K  V  3  :  V  →ₗ[K]  V)
#check  (LinearMap.lsmul  K  V  :  K  →ₗ[K]  V  →ₗ[K]  V) 
```

还有一个表示为 `V ≃ₗ[K] W` 的线性同构类型 `LinearEquiv`。`f : V ≃ₗ[K] W` 的逆是 `f.symm : W ≃ₗ[K] V`，`f` 和 `g` 的组合是 `f.trans g`，也记作 `f ≪≫ₗ g`，恒等同构是 `V` 的 `LinearEquiv.refl K V`。当需要时，此类型元素会自动转换为形态和函数。

```py
example  (f  :  V  ≃ₗ[K]  W)  :  f  ≪≫ₗ  f.symm  =  LinearEquiv.refl  K  V  :=
  f.self_trans_symm 
```

可以使用 `LinearEquiv.ofBijective` 从双射形态构建同构。这样做会使逆函数不可计算。

```py
noncomputable  example  (f  :  V  →ₗ[K]  W)  (h  :  Function.Bijective  f)  :  V  ≃ₗ[K]  W  :=
  .ofBijective  f  h 
```

注意，在上面的例子中，Lean 使用宣布的类型来理解 `.ofBijective` 指的是 `LinearEquiv.ofBijective`（无需打开任何命名空间）。

### 10.1.3\. 向量空间的和与积

我们可以使用直接和与直接积从旧向量空间构建新的向量空间。让我们从两个向量空间开始。在这种情况下，和与积之间没有区别，我们可以简单地使用积类型。在下面的代码片段中，我们简单地展示了如何将所有结构映射（包含和投射）作为线性映射来获取，以及构建线性映射到积和从和的通用性质（如果你不熟悉和与积之间的范畴论区别，你可以简单地忽略通用性质词汇，并关注以下示例的类型）。

```py
section  binary_product

variable  {W  :  Type*}  [AddCommGroup  W]  [Module  K  W]
variable  {U  :  Type*}  [AddCommGroup  U]  [Module  K  U]
variable  {T  :  Type*}  [AddCommGroup  T]  [Module  K  T]

-- First projection map
example  :  V  ×  W  →ₗ[K]  V  :=  LinearMap.fst  K  V  W

-- Second projection map
example  :  V  ×  W  →ₗ[K]  W  :=  LinearMap.snd  K  V  W

-- Universal property of the product
example  (φ  :  U  →ₗ[K]  V)  (ψ  :  U  →ₗ[K]  W)  :  U  →ₗ[K]  V  ×  W  :=  LinearMap.prod  φ  ψ

-- The product map does the expected thing, first component
example  (φ  :  U  →ₗ[K]  V)  (ψ  :  U  →ₗ[K]  W)  :  LinearMap.fst  K  V  W  ∘ₗ  LinearMap.prod  φ  ψ  =  φ  :=  rfl

-- The product map does the expected thing, second component
example  (φ  :  U  →ₗ[K]  V)  (ψ  :  U  →ₗ[K]  W)  :  LinearMap.snd  K  V  W  ∘ₗ  LinearMap.prod  φ  ψ  =  ψ  :=  rfl

-- We can also combine maps in parallel
example  (φ  :  V  →ₗ[K]  U)  (ψ  :  W  →ₗ[K]  T)  :  (V  ×  W)  →ₗ[K]  (U  ×  T)  :=  φ.prodMap  ψ

-- This is simply done by combining the projections with the universal property
example  (φ  :  V  →ₗ[K]  U)  (ψ  :  W  →ₗ[K]  T)  :
  φ.prodMap  ψ  =  (φ  ∘ₗ  .fst  K  V  W).prod  (ψ  ∘ₗ  .snd  K  V  W)  :=  rfl

-- First inclusion map
example  :  V  →ₗ[K]  V  ×  W  :=  LinearMap.inl  K  V  W

-- Second inclusion map
example  :  W  →ₗ[K]  V  ×  W  :=  LinearMap.inr  K  V  W

-- Universal property of the sum (aka coproduct)
example  (φ  :  V  →ₗ[K]  U)  (ψ  :  W  →ₗ[K]  U)  :  V  ×  W  →ₗ[K]  U  :=  φ.coprod  ψ

-- The coproduct map does the expected thing, first component
example  (φ  :  V  →ₗ[K]  U)  (ψ  :  W  →ₗ[K]  U)  :  φ.coprod  ψ  ∘ₗ  LinearMap.inl  K  V  W  =  φ  :=
  LinearMap.coprod_inl  φ  ψ

-- The coproduct map does the expected thing, second component
example  (φ  :  V  →ₗ[K]  U)  (ψ  :  W  →ₗ[K]  U)  :  φ.coprod  ψ  ∘ₗ  LinearMap.inr  K  V  W  =  ψ  :=
  LinearMap.coprod_inr  φ  ψ

-- The coproduct map is defined in the expected way
example  (φ  :  V  →ₗ[K]  U)  (ψ  :  W  →ₗ[K]  U)  (v  :  V)  (w  :  W)  :
  φ.coprod  ψ  (v,  w)  =  φ  v  +  ψ  w  :=
  rfl

end  binary_product 
```

让我们现在转向任意向量空间族的和与积。在这里，我们将简单地看到如何定义一个向量空间族，并访问和与积的通用性质。请注意，直接和的符号范围限定在 `DirectSum` 命名空间内，并且直接和的通用性质要求索引类型的可判定等价（这某种程度上是一个实现上的意外）。

```py
section  families
open  DirectSum

variable  {ι  :  Type*}  [DecidableEq  ι]
  (V  :  ι  →  Type*)  [∀  i,  AddCommGroup  (V  i)]  [∀  i,  Module  K  (V  i)]

-- The universal property of the direct sum assembles maps from the summands to build
-- a map from the direct sum
example  (φ  :  Π  i,  (V  i  →ₗ[K]  W))  :  (⨁  i,  V  i)  →ₗ[K]  W  :=
  DirectSum.toModule  K  ι  W  φ

-- The universal property of the direct product assembles maps into the factors
-- to build a map into the direct product
example  (φ  :  Π  i,  (W  →ₗ[K]  V  i))  :  W  →ₗ[K]  (Π  i,  V  i)  :=
  LinearMap.pi  φ

-- The projection maps from the product
example  (i  :  ι)  :  (Π  j,  V  j)  →ₗ[K]  V  i  :=  LinearMap.proj  i

-- The inclusion maps into the sum
example  (i  :  ι)  :  V  i  →ₗ[K]  (⨁  i,  V  i)  :=  DirectSum.lof  K  ι  V  i

-- The inclusion maps into the product
example  (i  :  ι)  :  V  i  →ₗ[K]  (Π  i,  V  i)  :=  LinearMap.single  K  V  i

-- In case `ι` is a finite type, there is an isomorphism between the sum and product.
example  [Fintype  ι]  :  (⨁  i,  V  i)  ≃ₗ[K]  (Π  i,  V  i)  :=
  linearEquivFunOnFintype  K  ι  V

end  families 
```

## 10.2\. 子空间和商

### 10.2.1\. 子空间

正如线性映射是打包的，`V` 的线性子空间也是一个打包结构，由 `V` 中的一个集合组成，称为子空间的载体，并具有相关的闭包性质。再次，由于 Mathlib 实际使用的线性代数更一般的环境，这里出现的是“模块”一词而不是向量空间。

```py
variable  {K  :  Type*}  [Field  K]  {V  :  Type*}  [AddCommGroup  V]  [Module  K  V]

example  (U  :  Submodule  K  V)  {x  y  :  V}  (hx  :  x  ∈  U)  (hy  :  y  ∈  U)  :
  x  +  y  ∈  U  :=
  U.add_mem  hx  hy

example  (U  :  Submodule  K  V)  {x  :  V}  (hx  :  x  ∈  U)  (a  :  K)  :
  a  •  x  ∈  U  :=
  U.smul_mem  a  hx 
```

在上面的例子中，重要的是要理解`Submodule K V`是`V`的`K`-线性子空间类型，而不是一个谓词`IsSubmodule U`，其中`U`是`Set V`的一个元素。`Submodule K V`被赋予了到`Set V`的强制转换和`V`上的成员谓词。参见第 8.3 节以了解如何以及为什么这样做。

当然，如果两个子空间具有相同的元素，则它们是相同的。这个事实被注册用于与`ext`策略一起使用，该策略可以用来证明两个子空间相等，就像它被用来证明两个集合相等一样。

要陈述和证明，例如，`ℝ`是`ℂ`的`ℝ`-线性子空间，我们真正想要的是构造一个类型为`Submodule ℝ ℂ`的项，其投影到`Set ℂ`是`ℝ`，或者更精确地说，是`ℝ`在`ℂ`中的像。

```py
noncomputable  example  :  Submodule  ℝ  ℂ  where
  carrier  :=  Set.range  ((↑)  :  ℝ  →  ℂ)
  add_mem'  :=  by
  rintro  _  _  ⟨n,  rfl⟩  ⟨m,  rfl⟩
  use  n  +  m
  simp
  zero_mem'  :=  by
  use  0
  simp
  smul_mem'  :=  by
  rintro  c  -  ⟨a,  rfl⟩
  use  c*a
  simp 
```

`Submodule`中证明字段末尾的素数与`LinearMap`中的素数类似。这些字段是用`carrier`字段来表述的，因为它们是在`MemberShip`实例之前定义的。然后，它们被上面看到的`Submodule.add_mem`、`Submodule.zero_mem`和`Submodule.smul_mem`所取代。

作为操作子空间和线性映射的练习，你将通过线性映射定义子空间的逆像（当然，我们将在下面看到 Mathlib 已经知道这一点）。记住，`Set.mem_preimage`可以用来重写涉及成员和逆像的陈述。除了上面讨论的关于`LinearMap`和`Submodule`的引理之外，你只需要这个引理。

```py
def  preimage  {W  :  Type*}  [AddCommGroup  W]  [Module  K  W]  (φ  :  V  →ₗ[K]  W)  (H  :  Submodule  K  W)  :
  Submodule  K  V  where
  carrier  :=  φ  ⁻¹'  H
  zero_mem'  :=  by
  sorry
  add_mem'  :=  by
  sorry
  smul_mem'  :=  by
  sorry 
```

使用类型类，Mathlib 知道向量空间的子空间继承了向量空间结构。

```py
example  (U  :  Submodule  K  V)  :  Module  K  U  :=  inferInstance 
```

这个例子很微妙。对象`U`不是一个类型，但 Lean 会自动将其解释为`V`的子类型，从而将其强制转换为类型。因此，上面的例子可以更明确地重述如下：

```py
example  (U  :  Submodule  K  V)  :  Module  K  {x  :  V  //  x  ∈  U}  :=  inferInstance 
```

### 10.2.2. 完备格结构和内部直和

使用类型`Submodule K V`而不是谓词`IsSubmodule : Set V → Prop`的一个重要好处是，可以轻松地为`Submodule K V`赋予额外的结构。重要的是，它具有关于包含的完备格结构。例如，我们不是通过一个引理来声明`V`的两个子空间交集仍然是子空间，而是使用格运算`⊓`来构造交集。然后我们可以将关于格的任意引理应用于构造过程。

让我们验证两个子空间下确界所对应的集合确实，按照定义，是它们的交集。

```py
example  (H  H'  :  Submodule  K  V)  :
  ((H  ⊓  H'  :  Submodule  K  V)  :  Set  V)  =  (H  :  Set  V)  ∩  (H'  :  Set  V)  :=  rfl 
```

对于底层集合的交集，使用不同的符号可能看起来有些奇怪，但这种对应关系并不适用于上确界操作和集合的并集，因为子空间的并集在一般情况下不是子空间。相反，需要使用由并集生成的子空间，这可以通过`Submodule.span`来实现。

```py
example  (H  H'  :  Submodule  K  V)  :
  ((H  ⊔  H'  :  Submodule  K  V)  :  Set  V)  =  Submodule.span  K  ((H  :  Set  V)  ∪  (H'  :  Set  V))  :=  by
  simp  [Submodule.span_union] 
```

另一个微妙之处在于`V`本身不具有类型`Submodule K V`，因此我们需要一种方式来谈论`V`作为`V`的子空间。这也由格结构提供：整个子空间是这个格的顶部元素。

```py
example  (x  :  V)  :  x  ∈  (⊤  :  Submodule  K  V)  :=  trivial 
```

同样，这个格的底部元素是只包含零元素的子空间。

```py
example  (x  :  V)  :  x  ∈  (⊥  :  Submodule  K  V)  ↔  x  =  0  :=  Submodule.mem_bot  K 
```

特别地，我们可以讨论那些在（内部）直和中的子空间的情况。在两个子空间的情况下，我们使用通用谓词`IsCompl`，它对任何有界偏序类型都有意义。在一般子空间族的情况下，我们使用`DirectSum.IsInternal`。

```py
-- If two subspaces are in direct sum then they span the whole space.
example  (U  V  :  Submodule  K  V)  (h  :  IsCompl  U  V)  :
  U  ⊔  V  =  ⊤  :=  h.sup_eq_top

-- If two subspaces are in direct sum then they intersect only at zero.
example  (U  V  :  Submodule  K  V)  (h  :  IsCompl  U  V)  :
  U  ⊓  V  =  ⊥  :=  h.inf_eq_bot

section
open  DirectSum
variable  {ι  :  Type*}  [DecidableEq  ι]

-- If subspaces are in direct sum then they span the whole space.
example  (U  :  ι  →  Submodule  K  V)  (h  :  DirectSum.IsInternal  U)  :
  ⨆  i,  U  i  =  ⊤  :=  h.submodule_iSup_eq_top

-- If subspaces are in direct sum then they pairwise intersect only at zero.
example  {ι  :  Type*}  [DecidableEq  ι]  (U  :  ι  →  Submodule  K  V)  (h  :  DirectSum.IsInternal  U)
  {i  j  :  ι}  (hij  :  i  ≠  j)  :  U  i  ⊓  U  j  =  ⊥  :=
  (h.submodule_iSupIndep.pairwiseDisjoint  hij).eq_bot

-- Those conditions characterize direct sums.
#check  DirectSum.isInternal_submodule_iff_independent_and_iSup_eq_top

-- The relation with external direct sums: if a family of subspaces is
-- in internal direct sum then the map from their external direct sum into `V`
-- is a linear isomorphism.
noncomputable  example  {ι  :  Type*}  [DecidableEq  ι]  (U  :  ι  →  Submodule  K  V)
  (h  :  DirectSum.IsInternal  U)  :  (⨁  i,  U  i)  ≃ₗ[K]  V  :=
  LinearEquiv.ofBijective  (coeLinearMap  U)  h
end 
```

### 10.2.3\. 由一组基生成的子空间

除了从现有的子空间构建子空间之外，我们还可以使用`Submodule.span K s`从任何集合`s`构建子空间，它构建包含`s`的最小子空间。在纸上，通常使用这个空间由`s`的元素的所有线性组合构成。但通常更有效的是使用其由`Submodule.span_le`表达的通用性质，以及整个 Galois 连接理论。

```py
example  {s  :  Set  V}  (E  :  Submodule  K  V)  :  Submodule.span  K  s  ≤  E  ↔  s  ⊆  E  :=
  Submodule.span_le

example  :  GaloisInsertion  (Submodule.span  K)  ((↑)  :  Submodule  K  V  →  Set  V)  :=
  Submodule.gi  K  V 
```

当这些还不够时，可以使用相关的归纳原理`Submodule.span_induction`，它确保只要在`zero`和`s`的元素上成立，并且对加法和数乘稳定，那么对于`s`的生成空间中的每个元素都成立该性质。

作为练习，让我们重新证明`Submodule.mem_sup`的一个蕴含。记住，你可以使用模块策略来关闭由与`V`上各种代数操作相关的公理得出的目标。

```py
example  {S  T  :  Submodule  K  V}  {x  :  V}  (h  :  x  ∈  S  ⊔  T)  :
  ∃  s  ∈  S,  ∃  t  ∈  T,  x  =  s  +  t  :=  by
  rw  [←  S.span_eq,  ←  T.span_eq,  ←  Submodule.span_union]  at  h
  induction  h  using  Submodule.span_induction  with
  |  mem  y  h  =>
  sorry
  |  zero  =>
  sorry
  |  add  x  y  hx  hy  hx'  hy'  =>
  sorry
  |  smul  a  x  hx  hx'  =>
  sorry 
```

### 10.2.4\. 推拉子空间

如前所述，我们现在描述如何通过线性映射来推拉子空间。在 Mathlib 中，通常第一个操作称为`map`，第二个操作称为`comap`。

```py
section

variable  {W  :  Type*}  [AddCommGroup  W]  [Module  K  W]  (φ  :  V  →ₗ[K]  W)

variable  (E  :  Submodule  K  V)  in
#check  (Submodule.map  φ  E  :  Submodule  K  W)

variable  (F  :  Submodule  K  W)  in
#check  (Submodule.comap  φ  F  :  Submodule  K  V) 
```

注意这些操作位于`Submodule`命名空间中，因此可以使用点符号，并可以写作`E.map φ`而不是`Submodule.map φ E`，但这读起来相当笨拙（尽管一些 Mathlib 贡献者使用这种拼写）。

特别地，线性映射的范围和核是子空间。这些特殊情况很重要，足以得到声明。

```py
example  :  LinearMap.range  φ  =  .map  φ  ⊤  :=  LinearMap.range_eq_map  φ

example  :  LinearMap.ker  φ  =  .comap  φ  ⊥  :=  Submodule.comap_bot  φ  -- or `rfl` 
```

注意，我们不能用 `φ.ker` 代替 `LinearMap.ker φ`，因为 `LinearMap.ker` 也适用于保持更多结构的映射类，因此它不期望一个以 `LinearMap` 开头的类型的参数，因此点符号在这里不适用。然而，我们能够在右侧使用另一种点符号。因为 Lean 在展开左侧之后期望一个类型为 `Submodule K V` 的项，它将 `.comap` 解释为 `Submodule.comap`。

以下引理给出了这些子模块与 `φ` 的性质之间的关键关系。

```py
open  Function  LinearMap

example  :  Injective  φ  ↔  ker  φ  =  ⊥  :=  ker_eq_bot.symm

example  :  Surjective  φ  ↔  range  φ  =  ⊤  :=  range_eq_top.symm 
```

作为练习，让我们证明 `map` 和 `comap` 的伽罗瓦连接性质。可以使用以下引理，但这不是必需的，因为它们是按定义成立的。

```py
#check  Submodule.mem_map_of_mem
#check  Submodule.mem_map
#check  Submodule.mem_comap

example  (E  :  Submodule  K  V)  (F  :  Submodule  K  W)  :
  Submodule.map  φ  E  ≤  F  ↔  E  ≤  Submodule.comap  φ  F  :=  by
  sorry 
```

### 10.2.5\. 商空间

商向量空间使用通用的商符号（使用 `\quot` 输入，而不是普通的 `/`）。商空间的投影是 `Submodule.mkQ`，而其泛性质是 `Submodule.liftQ`。

```py
variable  (E  :  Submodule  K  V)

example  :  Module  K  (V  ⧸  E)  :=  inferInstance

example  :  V  →ₗ[K]  V  ⧸  E  :=  E.mkQ

example  :  ker  E.mkQ  =  E  :=  E.ker_mkQ

example  :  range  E.mkQ  =  ⊤  :=  E.range_mkQ

example  (hφ  :  E  ≤  ker  φ)  :  V  ⧸  E  →ₗ[K]  W  :=  E.liftQ  φ  hφ

example  (F  :  Submodule  K  W)  (hφ  :  E  ≤  .comap  φ  F)  :  V  ⧸  E  →ₗ[K]  W  ⧸  F  :=  E.mapQ  F  φ  hφ

noncomputable  example  :  (V  ⧸  LinearMap.ker  φ)  ≃ₗ[K]  range  φ  :=  φ.quotKerEquivRange 
```

作为练习，让我们证明商空间的子空间的对应定理。Mathlib 知道一个稍微更精确的版本，作为 `Submodule.comapMkQRelIso`。

```py
open  Submodule

#check  Submodule.map_comap_eq
#check  Submodule.comap_map_eq

example  :  Submodule  K  (V  ⧸  E)  ≃  {  F  :  Submodule  K  V  //  E  ≤  F  }  where
  toFun  :=  sorry
  invFun  :=  sorry
  left_inv  :=  sorry
  right_inv  :=  sorry 
```

### 10.2.1\. 子空间

正如线性映射被封装一样，`V` 的线性子空间也是一个封装的结构，由 `V` 中的一个集合组成，称为子空间的载体，并具有相关的闭包性质。再次使用“模块”这个词而不是向量空间，是因为 Mathlib 实际上用于线性代数的更一般上下文。

```py
variable  {K  :  Type*}  [Field  K]  {V  :  Type*}  [AddCommGroup  V]  [Module  K  V]

example  (U  :  Submodule  K  V)  {x  y  :  V}  (hx  :  x  ∈  U)  (hy  :  y  ∈  U)  :
  x  +  y  ∈  U  :=
  U.add_mem  hx  hy

example  (U  :  Submodule  K  V)  {x  :  V}  (hx  :  x  ∈  U)  (a  :  K)  :
  a  •  x  ∈  U  :=
  U.smul_mem  a  hx 
```

在上面的例子中，重要的是要理解 `Submodule K V` 是 `V` 的 `K`-线性子空间的类型，而不是一个 `IsSubmodule U` 的谓词，其中 `U` 是 `Set V` 的一个元素。`Submodule K V` 被赋予了到 `Set V` 的强制转换和 `V` 上的成员谓词。参见 第 8.3 节 了解如何以及为什么这样做。

当然，如果两个子空间具有相同的元素，则它们是相同的。这个事实被注册用于 `ext` 策略，它可以用来证明两个子空间相等，就像它被用来证明两个集合相等一样。

例如，为了陈述和证明 `ℝ` 是 `ℂ` 的 `ℝ`-线性子空间，我们真正想要的是构造一个类型为 `Submodule ℝ ℂ` 的项，其投影到 `Set ℂ` 是 `ℝ`，或者更精确地说，是 `ℝ` 在 `ℂ` 中的像。

```py
noncomputable  example  :  Submodule  ℝ  ℂ  where
  carrier  :=  Set.range  ((↑)  :  ℝ  →  ℂ)
  add_mem'  :=  by
  rintro  _  _  ⟨n,  rfl⟩  ⟨m,  rfl⟩
  use  n  +  m
  simp
  zero_mem'  :=  by
  use  0
  simp
  smul_mem'  :=  by
  rintro  c  -  ⟨a,  rfl⟩
  use  c*a
  simp 
```

`Submodule` 中证明字段末尾的素数与 `LinearMap` 中的类似。这些字段是用 `carrier` 字段来表述的，因为它们是在 `MemberShip` 实例之前定义的。然后，它们被我们上面看到的 `Submodule.add_mem`、`Submodule.zero_mem` 和 `Submodule.smul_mem` 所取代。

作为操作子空间和线性映射的练习，你将定义由线性映射（当然，我们将在下面看到 Mathlib 已经知道这一点）生成的子空间的逆像。记住，可以使用`Set.mem_preimage`来重写涉及成员和逆像的陈述。这是除了上面讨论的关于`LinearMap`和`Submodule`的引理之外，你将需要的唯一引理。

```py
def  preimage  {W  :  Type*}  [AddCommGroup  W]  [Module  K  W]  (φ  :  V  →ₗ[K]  W)  (H  :  Submodule  K  W)  :
  Submodule  K  V  where
  carrier  :=  φ  ⁻¹'  H
  zero_mem'  :=  by
  sorry
  add_mem'  :=  by
  sorry
  smul_mem'  :=  by
  sorry 
```

使用类型类，Mathlib 知道向量空间的子空间继承了向量空间的结构。

```py
example  (U  :  Submodule  K  V)  :  Module  K  U  :=  inferInstance 
```

这个例子很微妙。对象`U`不是一个类型，但 Lean 会自动将其解释为`V`的子类型，将其转换为类型。因此，上面的例子可以更明确地表述为：

```py
example  (U  :  Submodule  K  V)  :  Module  K  {x  :  V  //  x  ∈  U}  :=  inferInstance 
```

### 10.2.2\. 完全格结构和内部直接和

使用类型`Submodule K V`而不是谓词`IsSubmodule : Set V → Prop`的一个重要好处是，可以轻松地为`Submodule K V`赋予额外的结构。重要的是，它具有关于包含的完全格结构。例如，我们不是用一条断言两个`V`的子空间的交集仍然是子空间的引理，而是使用格运算`⊓`来构造交集。然后我们可以将关于格的任意引理应用于构造过程。

让我们验证两个子空间下确界的底层集合确实按照定义是它们的交集。

```py
example  (H  H'  :  Submodule  K  V)  :
  ((H  ⊓  H'  :  Submodule  K  V)  :  Set  V)  =  (H  :  Set  V)  ∩  (H'  :  Set  V)  :=  rfl 
```

对于底层集合的交集使用不同的符号可能看起来有些奇怪，但这种对应关系并不适用于上确界运算和集合的并集，因为子空间的并集在一般情况下不是子空间。相反，需要使用由并集生成的子空间，这可以通过使用`Submodule.span`来完成。

```py
example  (H  H'  :  Submodule  K  V)  :
  ((H  ⊔  H'  :  Submodule  K  V)  :  Set  V)  =  Submodule.span  K  ((H  :  Set  V)  ∪  (H'  :  Set  V))  :=  by
  simp  [Submodule.span_union] 
```

另一个微妙之处在于`V`本身不具有类型`Submodule K V`，因此我们需要一种方式来谈论`V`作为`V`的子空间。这也由格结构提供：整个子空间是这个格的顶元素。

```py
example  (x  :  V)  :  x  ∈  (⊤  :  Submodule  K  V)  :=  trivial 
```

类似地，这个格的底元素是只包含零元素的子空间。

```py
example  (x  :  V)  :  x  ∈  (⊥  :  Submodule  K  V)  ↔  x  =  0  :=  Submodule.mem_bot  K 
```

尤其是我们可以讨论（内部）直接和的子空间的情况。在两个子空间的情况下，我们使用适用于任何有界偏序类型的通用谓词`IsCompl`。在一般子空间族的情况下，我们使用`DirectSum.IsInternal`。

```py
-- If two subspaces are in direct sum then they span the whole space.
example  (U  V  :  Submodule  K  V)  (h  :  IsCompl  U  V)  :
  U  ⊔  V  =  ⊤  :=  h.sup_eq_top

-- If two subspaces are in direct sum then they intersect only at zero.
example  (U  V  :  Submodule  K  V)  (h  :  IsCompl  U  V)  :
  U  ⊓  V  =  ⊥  :=  h.inf_eq_bot

section
open  DirectSum
variable  {ι  :  Type*}  [DecidableEq  ι]

-- If subspaces are in direct sum then they span the whole space.
example  (U  :  ι  →  Submodule  K  V)  (h  :  DirectSum.IsInternal  U)  :
  ⨆  i,  U  i  =  ⊤  :=  h.submodule_iSup_eq_top

-- If subspaces are in direct sum then they pairwise intersect only at zero.
example  {ι  :  Type*}  [DecidableEq  ι]  (U  :  ι  →  Submodule  K  V)  (h  :  DirectSum.IsInternal  U)
  {i  j  :  ι}  (hij  :  i  ≠  j)  :  U  i  ⊓  U  j  =  ⊥  :=
  (h.submodule_iSupIndep.pairwiseDisjoint  hij).eq_bot

-- Those conditions characterize direct sums.
#check  DirectSum.isInternal_submodule_iff_independent_and_iSup_eq_top

-- The relation with external direct sums: if a family of subspaces is
-- in internal direct sum then the map from their external direct sum into `V`
-- is a linear isomorphism.
noncomputable  example  {ι  :  Type*}  [DecidableEq  ι]  (U  :  ι  →  Submodule  K  V)
  (h  :  DirectSum.IsInternal  U)  :  (⨁  i,  U  i)  ≃ₗ[K]  V  :=
  LinearEquiv.ofBijective  (coeLinearMap  U)  h
end 
```

### 10.2.3\. 由集合生成的子空间

除了从现有的子空间构建子空间之外，我们还可以使用`Submodule.span K s`从任何集合`s`构建子空间，它构建包含`s`的最小子空间。在纸上，通常使用这个空间由`s`的元素的所有线性组合构成。但通常更有效的是使用其通过`Submodule.span_le`表达的通用性质，以及 Galois 连接的全套理论。

```py
example  {s  :  Set  V}  (E  :  Submodule  K  V)  :  Submodule.span  K  s  ≤  E  ↔  s  ⊆  E  :=
  Submodule.span_le

example  :  GaloisInsertion  (Submodule.span  K)  ((↑)  :  Submodule  K  V  →  Set  V)  :=
  Submodule.gi  K  V 
```

当这些不够用时，可以使用相关的归纳原理 `Submodule.span_induction`，该原理确保只要在 `zero` 和 `s` 的元素上成立，并且对和以及标量乘法稳定，那么该性质就适用于 `s` 的张量积的每个元素。

作为练习，让我们重新证明 `Submodule.mem_sup` 的一个推论。记住，你可以使用模块策略来关闭由与 `V` 上的各种代数操作相关的公理得出的目标。

```py
example  {S  T  :  Submodule  K  V}  {x  :  V}  (h  :  x  ∈  S  ⊔  T)  :
  ∃  s  ∈  S,  ∃  t  ∈  T,  x  =  s  +  t  :=  by
  rw  [←  S.span_eq,  ←  T.span_eq,  ←  Submodule.span_union]  at  h
  induction  h  using  Submodule.span_induction  with
  |  mem  y  h  =>
  sorry
  |  zero  =>
  sorry
  |  add  x  y  hx  hy  hx'  hy'  =>
  sorry
  |  smul  a  x  hx  hx'  =>
  sorry 
```

### 10.2.4\. 推进和拉回子空间

如前所述，我们现在描述如何通过线性映射推进和拉回子空间。在 Mathlib 中，通常第一个操作称为 `map`，第二个操作称为 `comap`。

```py
section

variable  {W  :  Type*}  [AddCommGroup  W]  [Module  K  W]  (φ  :  V  →ₗ[K]  W)

variable  (E  :  Submodule  K  V)  in
#check  (Submodule.map  φ  E  :  Submodule  K  W)

variable  (F  :  Submodule  K  W)  in
#check  (Submodule.comap  φ  F  :  Submodule  K  V) 
```

注意，这些位于 `Submodule` 命名空间中，因此可以使用点符号，并可以写 `E.map φ` 而不是 `Submodule.map φ E`，但这读起来相当笨拙（尽管一些 Mathlib 贡献者使用这种拼写）。

特别是，线性映射的值域和核是子空间。这些特殊情况很重要，足以获得声明。

```py
example  :  LinearMap.range  φ  =  .map  φ  ⊤  :=  LinearMap.range_eq_map  φ

example  :  LinearMap.ker  φ  =  .comap  φ  ⊥  :=  Submodule.comap_bot  φ  -- or `rfl` 
```

注意，我们不能写 `φ.ker` 而不是 `LinearMap.ker φ`，因为 `LinearMap.ker` 也适用于保持更多结构的映射类，因此它不期望一个以 `LinearMap` 开头的类型作为参数，因此点符号在这里不适用。然而，我们能够在右侧使用另一种点符号。因为 Lean 预期在展开左侧之后，有一个类型为 `Submodule K V` 的项，它将 `.comap` 解释为 `Submodule.comap`。

以下引理给出了这些子模块与 `φ` 的性质之间的关键关系。

```py
open  Function  LinearMap

example  :  Injective  φ  ↔  ker  φ  =  ⊥  :=  ker_eq_bot.symm

example  :  Surjective  φ  ↔  range  φ  =  ⊤  :=  range_eq_top.symm 
```

作为练习，让我们证明 `map` 和 `comap` 的 Galois 关联性质。可以使用以下引理，但这不是必需的，因为它们是按定义成立的。

```py
#check  Submodule.mem_map_of_mem
#check  Submodule.mem_map
#check  Submodule.mem_comap

example  (E  :  Submodule  K  V)  (F  :  Submodule  K  W)  :
  Submodule.map  φ  E  ≤  F  ↔  E  ≤  Submodule.comap  φ  F  :=  by
  sorry 
```

### 10.2.5\. 商空间

商向量空间使用通用的商符号（使用 `\quot` 打印，而不是普通 `/`）。商空间的投影是 `Submodule.mkQ`，其通用性质是 `Submodule.liftQ`。

```py
variable  (E  :  Submodule  K  V)

example  :  Module  K  (V  ⧸  E)  :=  inferInstance

example  :  V  →ₗ[K]  V  ⧸  E  :=  E.mkQ

example  :  ker  E.mkQ  =  E  :=  E.ker_mkQ

example  :  range  E.mkQ  =  ⊤  :=  E.range_mkQ

example  (hφ  :  E  ≤  ker  φ)  :  V  ⧸  E  →ₗ[K]  W  :=  E.liftQ  φ  hφ

example  (F  :  Submodule  K  W)  (hφ  :  E  ≤  .comap  φ  F)  :  V  ⧸  E  →ₗ[K]  W  ⧸  F  :=  E.mapQ  F  φ  hφ

noncomputable  example  :  (V  ⧸  LinearMap.ker  φ)  ≃ₗ[K]  range  φ  :=  φ.quotKerEquivRange 
```

作为练习，让我们证明商空间子空间的对应定理。Mathlib 知道一个稍微更精确的版本，即 `Submodule.comapMkQRelIso`。

```py
open  Submodule

#check  Submodule.map_comap_eq
#check  Submodule.comap_map_eq

example  :  Submodule  K  (V  ⧸  E)  ≃  {  F  :  Submodule  K  V  //  E  ≤  F  }  where
  toFun  :=  sorry
  invFun  :=  sorry
  left_inv  :=  sorry
  right_inv  :=  sorry 
```

## 10.3\. 内射

线性映射的一个重要特殊情况是内射：从向量空间到自身的线性映射。它们很有趣，因为它们形成一个 `K`-代数。特别是，我们可以在它们上评估系数在 `K` 中的多项式，并且它们可以有特征值和特征向量。

Mathlib 使用缩写 `Module.End K V := V →ₗ[K] V`，这在使用很多这些（特别是在打开 `Module` 命名空间之后）时很方便。

```py
variable  {K  :  Type*}  [Field  K]  {V  :  Type*}  [AddCommGroup  V]  [Module  K  V]

variable  {W  :  Type*}  [AddCommGroup  W]  [Module  K  W]

open  Polynomial  Module  LinearMap  End

example  (φ  ψ  :  End  K  V)  :  φ  *  ψ  =  φ  ∘ₗ  ψ  :=
  End.mul_eq_comp  φ  ψ  -- `rfl` would also work

-- evaluating `P` on `φ`
example  (P  :  K[X])  (φ  :  End  K  V)  :  V  →ₗ[K]  V  :=
  aeval  φ  P

-- evaluating `X` on `φ` gives back `φ`
example  (φ  :  End  K  V)  :  aeval  φ  (X  :  K[X])  =  φ  :=
  aeval_X  φ 
```

作为操作内射、子空间和多项式的练习，让我们证明（二元的）核引理：对于任何内射 $φ$ 和任意两个互质的代数多项式 $P$ 和 $Q$，我们有 $\ker P(φ) ⊕ \ker Q(φ) = \ker \big(PQ(φ)\big)$。

注意，`IsCoprime x y` 被定义为 `∃ a b, a * x + b * y = 1`。

```py
#check  Submodule.eq_bot_iff
#check  Submodule.mem_inf
#check  LinearMap.mem_ker

example  (P  Q  :  K[X])  (h  :  IsCoprime  P  Q)  (φ  :  End  K  V)  :  ker  (aeval  φ  P)  ⊓  ker  (aeval  φ  Q)  =  ⊥  :=  by
  sorry

#check  Submodule.add_mem_sup
#check  map_mul
#check  End.mul_apply
#check  LinearMap.ker_le_ker_comp

example  (P  Q  :  K[X])  (h  :  IsCoprime  P  Q)  (φ  :  End  K  V)  :
  ker  (aeval  φ  P)  ⊔  ker  (aeval  φ  Q)  =  ker  (aeval  φ  (P*Q))  :=  by
  sorry 
```

我们现在转向对特征空间和特征值的讨论。与内射 $φ$ 和标量 $a$ 相关的特征空间是 $φ - aId$ 的核。特征空间对所有 `a` 的值都定义，尽管它们只有在非零时才有意义。然而，根据定义，特征向量是特征空间中的非零元素。相应的谓词是 `End.HasEigenvector`。

```py
example  (φ  :  End  K  V)  (a  :  K)  :  φ.eigenspace  a  =  LinearMap.ker  (φ  -  a  •  1)  :=
  End.eigenspace_def 
```

然后有一个谓词 `End.HasEigenvalue` 和相应的子类型 `End.Eigenvalues`。

```py
example  (φ  :  End  K  V)  (a  :  K)  :  φ.HasEigenvalue  a  ↔  φ.eigenspace  a  ≠  ⊥  :=
  Iff.rfl

example  (φ  :  End  K  V)  (a  :  K)  :  φ.HasEigenvalue  a  ↔  ∃  v,  φ.HasEigenvector  a  v  :=
  ⟨End.HasEigenvalue.exists_hasEigenvector,  fun  ⟨_,  hv⟩  ↦  φ.hasEigenvalue_of_hasEigenvector  hv⟩

example  (φ  :  End  K  V)  :  φ.Eigenvalues  =  {a  //  φ.HasEigenvalue  a}  :=
  rfl

-- Eigenvalue are roots of the minimal polynomial
example  (φ  :  End  K  V)  (a  :  K)  :  φ.HasEigenvalue  a  →  (minpoly  K  φ).IsRoot  a  :=
  φ.isRoot_of_hasEigenvalue

-- In finite dimension, the converse is also true (we will discuss dimension below)
example  [FiniteDimensional  K  V]  (φ  :  End  K  V)  (a  :  K)  :
  φ.HasEigenvalue  a  ↔  (minpoly  K  φ).IsRoot  a  :=
  φ.hasEigenvalue_iff_isRoot

-- Cayley-Hamilton
example  [FiniteDimensional  K  V]  (φ  :  End  K  V)  :  aeval  φ  φ.charpoly  =  0  :=
  φ.aeval_self_charpoly 
```

## 10.4\. 矩阵、基和维度

### 10.4.1\. 矩阵

在介绍抽象向量空间的基之前，我们回顾一下对于某个域 $K$ 的 $K^n$ 中的线性代数的基础设置。在这里，主要对象是向量和矩阵。对于具体的向量，可以使用 `![…]` 符号，其中分量由逗号分隔。对于具体的矩阵，我们可以使用 `!![…]` 符号，行由分号分隔，而行内的分量由冒号分隔。当项具有可计算类型，如 `ℕ` 或 `ℚ` 时，我们可以使用 `eval` 命令来进行基本操作。

```py
section  matrices

-- Adding vectors
#eval  ![1,  2]  +  ![3,  4]  -- ![4, 6]

-- Adding matrices
#eval  !![1,  2;  3,  4]  +  !![3,  4;  5,  6]  -- !![4, 6; 8, 10]

-- Multiplying matrices
#eval  !![1,  2;  3,  4]  *  !![3,  4;  5,  6]  -- !![13, 16; 29, 36] 
```

重要的是要理解这种 `#eval` 的使用仅对探索有趣，它不是用来取代像 Sage 这样的计算机代数系统的。这里用于矩阵的数据表示在任何方面都不是计算高效的。它使用函数而不是数组，并且优化于证明，而不是计算。`#eval` 所使用的虚拟机也不是为此用途优化的。

警惕矩阵表示法列表行，但向量表示法既不是行向量也不是列向量。从左（分别）乘以矩阵的向量将向量解释为行（分别）向量。这对应于 `Matrix.vecMul` 操作，符号为 `ᵥ*`，以及 `Matrix.mulVec` 操作，符号为 ` *ᵥ`。这些符号在 `Matrix` 命名空间中定义，因此我们需要打开它。

```py
open  Matrix

-- matrices acting on vectors on the left
#eval  !![1,  2;  3,  4]  *ᵥ  ![1,  1]  -- ![3, 7]

-- matrices acting on vectors on the left, resulting in a size one matrix
#eval  !![1,  2]  *ᵥ  ![1,  1]  -- ![3]

-- matrices acting on vectors on the right
#eval  ![1,  1,  1]  ᵥ*  !![1,  2;  3,  4;  5,  6]  -- ![9, 12] 
```

为了生成具有由向量指定的相同行或列的矩阵，我们使用 `Matrix.replicateRow` 和 `Matrix.replicateCol`，其中参数是索引行或列的类型和向量。例如，可以得到单行或单列矩阵（更精确地说，行或列由 `Fin 1` 索引的矩阵）。

```py
#eval  replicateRow  (Fin  1)  ![1,  2]  -- !![1, 2]

#eval  replicateCol  (Fin  1)  ![1,  2]  -- !![1; 2] 
```

其他熟悉的操作包括向量点积、矩阵转置，对于方阵，还有行列式和迹。

```py
-- vector dot product
#eval  ![1,  2]  ⬝ᵥ  ![3,  4]  -- `11`

-- matrix transpose
#eval  !![1,  2;  3,  4]ᵀ  -- `!![1, 3; 2, 4]`

-- determinant
#eval  !![(1  :  ℤ),  2;  3,  4].det  -- `-2`

-- trace
#eval  !![(1  :  ℤ),  2;  3,  4].trace  -- `5` 
```

当条目没有可计算的类型时，例如如果它们是实数，我们无法期望 `#eval` 能有所帮助。此外，这种评估不能在没有显著扩大可信代码库（即检查证明时需要信任的 Lean 的部分）的情况下用于证明。

因此，在证明中也使用 `simp` 和 `norm_num` 策略，或者它们的命令对应物进行快速探索是很好的。

```py
#simp  !![(1  :  ℝ),  2;  3,  4].det  -- `4 - 2*3`

#norm_num  !![(1  :  ℝ),  2;  3,  4].det  -- `-2`

#norm_num  !![(1  :  ℝ),  2;  3,  4].trace  -- `5`

variable  (a  b  c  d  :  ℝ)  in
#simp  !![a,  b;  c,  d].det  -- `a * d – b * c` 
```

平方矩阵上的下一个重要操作是求逆。与数的除法总是定义并返回除以零的人工值零一样，求逆操作在所有矩阵上定义，对于不可逆矩阵返回零矩阵。

更精确地说，有一个通用的函数 `Ring.inverse` 在任何环中执行此操作，并且对于任何矩阵 `A`，`A⁻¹` 被定义为 `Ring.inverse A.det • A.adjugate`。根据克莱姆法则，当 `A` 的行列式不为零时，这确实是 `A` 的逆。

```py
#norm_num  [Matrix.inv_def]  !![(1  :  ℝ),  2;  3,  4]⁻¹  -- !![-2, 1; 3 / 2, -(1 / 2)] 
```

当然，这种定义真正有用的只有对于可逆矩阵。有一个通用的类型类 `Invertible` 帮助记录这一点。例如，下一个例子中的 `simp` 调用将使用具有 `Invertible` 类型类假设的 `inv_mul_of_invertible` 引理，因此只有在类型类合成系统可以找到它的情况下才会触发。在这里，我们使用 `have` 陈述使这一事实可用。

```py
example  :  !![(1  :  ℝ),  2;  3,  4]⁻¹  *  !![(1  :  ℝ),  2;  3,  4]  =  1  :=  by
  have  :  Invertible  !![(1  :  ℝ),  2;  3,  4]  :=  by
  apply  Matrix.invertibleOfIsUnitDet
  norm_num
  simp 
```

在这个完全具体的情况下，我们也可以使用 `norm_num` 机制和 `apply?` 来找到最终行：

```py
example  :  !![(1  :  ℝ),  2;  3,  4]⁻¹  *  !![(1  :  ℝ),  2;  3,  4]  =  1  :=  by
  norm_num  [Matrix.inv_def]
  exact  one_fin_two.symm 
```

上面的所有具体矩阵都有它们的行和列由 `Fin n`（对于某个 `n`，行和列不一定相同）索引。但有时使用任意有限类型索引矩阵更为方便。例如，有限图的邻接矩阵的行和列自然由图的顶点索引。

事实上，当仅仅想要定义矩阵而不在它们上定义任何操作时，索引类型的有限性甚至都不需要，系数可以有任何类型，而不需要任何代数结构。因此，Mathlib 简单地将 `Matrix m n α` 定义为 `m → n → α`，对于任何类型 `m`、`n` 和 `α`，而我们迄今为止所使用的矩阵类型如 `Matrix (Fin 2) (Fin 2) ℝ`。当然，代数操作需要对 `m`、`n` 和 `α` 有更多的假设。

注意我们不直接使用 `m → n → α` 的主要原因是为了让类型类系统理解我们的意图。例如，对于环 `R`，类型 `n → R` 被赋予了点积乘法操作，同样 `m → n → R` 也有这种操作，但这不是我们在矩阵上想要的乘法。

在下面的第一个例子中，我们迫使 Lean 看透 `Matrix` 的定义，并接受该陈述是有意义的，然后通过检查所有条目来证明它。

但接下来的两个例子揭示了 Lean 在 `Fin 2 → Fin 2 → ℤ` 上使用点积乘法，但在 `Matrix (Fin 2) (Fin 2) ℤ` 上使用矩阵乘法。

```py
section

example  :  (fun  _  ↦  1  :  Fin  2  →  Fin  2  →  ℤ)  =  !![1,  1;  1,  1]  :=  by
  ext  i  j
  fin_cases  i  <;>  fin_cases  j  <;>  rfl

example  :  (fun  _  ↦  1  :  Fin  2  →  Fin  2  →  ℤ)  *  (fun  _  ↦  1  :  Fin  2  →  Fin  2  →  ℤ)  =  !![1,  1;  1,  1]  :=  by
  ext  i  j
  fin_cases  i  <;>  fin_cases  j  <;>  rfl

example  :  !![1,  1;  1,  1]  *  !![1,  1;  1,  1]  =  !![2,  2;  2,  2]  :=  by
  norm_num 
```

为了定义矩阵作为函数而不失去 `Matrix` 类型类综合的好处，我们可以使用函数和矩阵之间的等价性 `Matrix.of`。这个等价性实际上是通过使用 `Equiv.refl` 来秘密定义的。

例如，我们可以定义与向量 `v` 对应的范德蒙德矩阵。

```py
example  {n  :  ℕ}  (v  :  Fin  n  →  ℝ)  :
  Matrix.vandermonde  v  =  Matrix.of  (fun  i  j  :  Fin  n  ↦  v  i  ^  (j  :  ℕ))  :=
  rfl
end
end  matrices 
```

### 10.4.2\. 基

现在我们想讨论向量空间的基。非正式地说，有定义这个概念的好几种方法。人们可以使用一个通用属性。人们可以说一个基是一组线性无关且张成的向量。或者人们可以结合这些属性，直接说一个基是一组向量，每个向量都可以唯一地表示为基向量的线性组合。还有一种说法是，基提供了一个与基域 `K` 的幂的线性同构，`K` 被视为 `K` 上的向量空间。

这个同构版本实际上是 Mathlib 在底层用作定义的版本，其他特征化都是从这个版本证明出来的。在无限基的情况下，必须对“`K` 的幂”这个概念稍加小心。确实，在这个代数背景下，只有有限线性组合才有意义。因此，我们需要作为参考的向量空间不是 `K` 的副本的直接积，而是一个直接和。我们可以使用 `⨁ i : ι, K` 对于某些类型 `ι` 来索引基，但我们更倾向于使用更专业的表示 `ι →₀ K`，这意味着“从 `ι` 到 `K` 的具有有限支撑的函数”，即除了在 `ι` 中的有限集合外都为零的函数（这个有限集合不是固定的，它依赖于函数）。在一个基 `B` 中评估来自基的这样一个函数在向量 `v` 和 `i : ι` 上的值，返回 `v` 在第 `i` 个基向量上的分量（或坐标）。

作为 `K` 向量空间 `V` 的类型 `ι` 的基的类型是 `Basis ι K V`。这个同构被称为 `Basis.repr`。

```py
variable  {K  :  Type*}  [Field  K]  {V  :  Type*}  [AddCommGroup  V]  [Module  K  V]

section

variable  {ι  :  Type*}  (B  :  Basis  ι  K  V)  (v  :  V)  (i  :  ι)

-- The basis vector with index ``i``
#check  (B  i  :  V)

-- the linear isomorphism with the model space given by ``B``
#check  (B.repr  :  V  ≃ₗ[K]  ι  →₀  K)

-- the component function of ``v``
#check  (B.repr  v  :  ι  →₀  K)

-- the component of ``v`` with index ``i``
#check  (B.repr  v  i  :  K) 
```

而不是从一个这样的同构开始，人们可以从一组线性无关且张成的向量 `b` 开始，这就是 `Basis.mk`。

假设这个族是张成的，表示为 `⊤ ≤ Submodule.span K (Set.range b)`。这里 `⊤` 是 `V` 的上子模，即 `V` 作为自身的子模来看待。这种表示看起来有点扭曲，但下面我们将看到它几乎等同于更易读的 `∀ v, v ∈ Submodule.span K (Set.range b)`（下面片段中的下划线指的是无用的信息 `v ∈ ⊤`）。

```py
noncomputable  example  (b  :  ι  →  V)  (b_indep  :  LinearIndependent  K  b)
  (b_spans  :  ∀  v,  v  ∈  Submodule.span  K  (Set.range  b))  :  Basis  ι  K  V  :=
  Basis.mk  b_indep  (fun  v  _  ↦  b_spans  v)

-- The family of vectors underlying the above basis is indeed ``b``.
example  (b  :  ι  →  V)  (b_indep  :  LinearIndependent  K  b)
  (b_spans  :  ∀  v,  v  ∈  Submodule.span  K  (Set.range  b))  (i  :  ι)  :
  Basis.mk  b_indep  (fun  v  _  ↦  b_spans  v)  i  =  b  i  :=
  Basis.mk_apply  b_indep  (fun  v  _  ↦  b_spans  v)  i 
```

特别地，模型向量空间 `ι →₀ K` 有一个所谓的规范基，其 `repr` 函数在任意向量上的评估是恒等同构。它被称为 `Finsupp.basisSingleOne`，其中 `Finsupp` 表示具有有限支撑的函数，而 `basisSingleOne` 指的是基向量是除了单个输入值外都为零的函数。更精确地说，由 `i : ι` 索引的基向量是 `Finsupp.single i 1`，这是一个在 `i` 处取值为 `1` 而在其他所有地方取值为 `0` 的有限支撑函数。

```py
variable  [DecidableEq  ι] 
```

```py
example  :  Finsupp.basisSingleOne.repr  =  LinearEquiv.refl  K  (ι  →₀  K)  :=
  rfl

example  (i  :  ι)  :  Finsupp.basisSingleOne  i  =  Finsupp.single  i  1  :=
  rfl 
```

当索引类型是有限的时，有限支撑函数的故事是不必要的。在这种情况下，我们可以使用更简单的`Pi.basisFun`，它给出整个`ι → K`的基。

```py
example  [Finite  ι]  (x  :  ι  →  K)  (i  :  ι)  :  (Pi.basisFun  K  ι).repr  x  i  =  x  i  :=  by
  simp 
```

回到抽象向量空间基的一般情况，我们可以将任何向量表示为基向量的线性组合。让我们首先看看有限基的简单情况。

```py
example  [Fintype  ι]  :  ∑  i  :  ι,  B.repr  v  i  •  (B  i)  =  v  :=
  B.sum_repr  v 
```

当`ι`不是有限的时，上述陈述在先验上没有意义：我们不能在`ι`上取和。然而，被求和函数的支持是有限的（它是`B.repr v`的支持）。但是我们需要应用一个考虑到这一点的构造。在这里，Mathlib 使用了一个特殊目的的函数，需要一些时间来习惯：`Finsupp.linearCombination`（它建立在更一般的`Finsupp.sum`之上）。给定一个从类型`ι`到基域`K`的有限支撑函数`c`和任何从`ι`到`V`的函数`f`，`Finsupp.linearCombination K f c`是`c`的支持上的标量乘积`c • f`的和。特别是，我们可以用包含`c`的支持的任何有限集上的和来替换它。

```py
example  (c  :  ι  →₀  K)  (f  :  ι  →  V)  (s  :  Finset  ι)  (h  :  c.support  ⊆  s)  :
  Finsupp.linearCombination  K  f  c  =  ∑  i  ∈  s,  c  i  •  f  i  :=
  Finsupp.linearCombination_apply_of_mem_supported  K  h 
```

也可以假设`f`是有限支撑的，并且仍然可以得到一个定义良好的和。但`Finsupp.linearCombination`所做的选择与我们的基讨论相关，因为它允许我们陈述`Basis.sum_repr`的推广。

```py
example  :  Finsupp.linearCombination  K  B  (B.repr  v)  =  v  :=
  B.linearCombination_repr  v 
```

有些人可能会想知道为什么`K`在这里是一个显式的参数，尽管它可以从`c`的类型中推断出来。关键是部分应用`Finsupp.linearCombination K f`本身是有趣的。它不是一个从`ι →₀ K`到`V`的裸函数，而是一个`K`-线性映射。

```py
variable  (f  :  ι  →  V)  in
#check  (Finsupp.linearCombination  K  f  :  (ι  →₀  K)  →ₗ[K]  V) 
```

回到数学讨论，重要的是要理解，在形式化的数学中，基中向量的表示可能没有你想象的那么有用。事实上，直接使用基的更抽象性质通常更有效。特别是，基的普遍性质将它们与代数中的其他自由对象联系起来，允许通过指定基向量的像来构造线性映射。这是`Basis.constr`。对于任何`K`-向量空间`W`，我们的基`B`给出了一个线性同构`Basis.constr B K`，从`ι → W`到`V →ₗ[K] W`。这个同构的特点是它将任何函数`u : ι → W`映射到一个线性映射，该映射将基向量`B i`映射到`u i`，对于每个`i : ι`。

```py
section

variable  {W  :  Type*}  [AddCommGroup  W]  [Module  K  W]
  (φ  :  V  →ₗ[K]  W)  (u  :  ι  →  W)

#check  (B.constr  K  :  (ι  →  W)  ≃ₗ[K]  (V  →ₗ[K]  W))

#check  (B.constr  K  u  :  V  →ₗ[K]  W)

example  (i  :  ι)  :  B.constr  K  u  (B  i)  =  u  i  :=
  B.constr_basis  K  u  i 
```

这个性质确实是特征性的，因为线性映射由其在基上的值确定：

```py
example  (φ  ψ  :  V  →ₗ[K]  W)  (h  :  ∀  i,  φ  (B  i)  =  ψ  (B  i))  :  φ  =  ψ  :=
  B.ext  h 
```

如果我们在目标空间上也有一个基`B'`，那么我们可以将线性映射与矩阵相识别。这种识别是一个`K`-线性同构。

```py
variable  {ι'  :  Type*}  (B'  :  Basis  ι'  K  W)  [Fintype  ι]  [DecidableEq  ι]  [Fintype  ι']  [DecidableEq  ι']

open  LinearMap

#check  (toMatrix  B  B'  :  (V  →ₗ[K]  W)  ≃ₗ[K]  Matrix  ι'  ι  K)

open  Matrix  -- get access to the ``*ᵥ`` notation for multiplication between matrices and vectors.

example  (φ  :  V  →ₗ[K]  W)  (v  :  V)  :  (toMatrix  B  B'  φ)  *ᵥ  (B.repr  v)  =  B'.repr  (φ  v)  :=
  toMatrix_mulVec_repr  B  B'  φ  v

variable  {ι''  :  Type*}  (B''  :  Basis  ι''  K  W)  [Fintype  ι'']  [DecidableEq  ι'']

example  (φ  :  V  →ₗ[K]  W)  :  (toMatrix  B  B''  φ)  =  (toMatrix  B'  B''  .id)  *  (toMatrix  B  B'  φ)  :=  by
  simp

end 
```

作为这个主题的练习，我们将证明保证内射有良好定义的行列式的定理的一部分。也就是说，我们想要证明当两个基由相同的类型索引时，它们附加到任何内射上的矩阵具有相同的行列式。然后需要使用基都具有同构索引类型来补充这一点，以得到完整的结果。

当然，Mathlib 已经知道这一点，`simp`可以立即关闭目标，所以你不应该太早使用它，而应该使用提供的引理。

```py
open  Module  LinearMap  Matrix

-- Some lemmas coming from the fact that `LinearMap.toMatrix` is an algebra morphism.
#check  toMatrix_comp
#check  id_comp
#check  comp_id
#check  toMatrix_id

-- Some lemmas coming from the fact that ``Matrix.det`` is a multiplicative monoid morphism.
#check  Matrix.det_mul
#check  Matrix.det_one

example  [Fintype  ι]  (B'  :  Basis  ι  K  V)  (φ  :  End  K  V)  :
  (toMatrix  B  B  φ).det  =  (toMatrix  B'  B'  φ).det  :=  by
  set  M  :=  toMatrix  B  B  φ
  set  M'  :=  toMatrix  B'  B'  φ
  set  P  :=  (toMatrix  B  B')  LinearMap.id
  set  P'  :=  (toMatrix  B'  B)  LinearMap.id
  sorry
end 
```

### 10.4.3. 维度

返回到单个向量空间的情况，基对于定义维度概念也是有用的。在这里，有限维向量空间是基本的情况。对于这样的空间，我们期望维度是一个自然数。这是`Module.finrank`。它将基域作为显式参数，因为给定的阿贝尔群可以是在不同域上的向量空间。

```py
section
#check  (Module.finrank  K  V  :  ℕ)

-- `Fin n → K` is the archetypical space with dimension `n` over `K`.
example  (n  :  ℕ)  :  Module.finrank  K  (Fin  n  →  K)  =  n  :=
  Module.finrank_fin_fun  K

-- Seen as a vector space over itself, `ℂ` has dimension one.
example  :  Module.finrank  ℂ  ℂ  =  1  :=
  Module.finrank_self  ℂ

-- But as a real vector space it has dimension two.
example  :  Module.finrank  ℝ  ℂ  =  2  :=
  Complex.finrank_real_complex 
```

注意，`Module.finrank`为任何向量空间定义。对于无限维向量空间，它返回零，就像除以零返回零一样。

当然，许多引理需要有限维度的假设。这就是`FiniteDimensional`类型类的作用。例如，考虑如果没有这个假设，下一个例子会如何失败。

```py
example  [FiniteDimensional  K  V]  :  0  <  Module.finrank  K  V  ↔  Nontrivial  V  :=
  Module.finrank_pos_iff 
```

在上述陈述中，`Nontrivial V`表示`V`至少有两个不同的元素。请注意，`Module.finrank_pos_iff`没有显式的参数。当从左到右使用它时这是可以的，但当你从右到左使用它时就不行了，因为 Lean 无法从陈述`Nontrivial V`中猜测`K`。在这种情况下，使用名称参数语法是有用的，在检查引理是在名为`R`的环上陈述之后。因此，我们可以写出：

```py
example  [FiniteDimensional  K  V]  (h  :  0  <  Module.finrank  K  V)  :  Nontrivial  V  :=  by
  apply  (Module.finrank_pos_iff  (R  :=  K)).1
  exact  h 
```

上述拼写很奇怪，因为我们已经有了`h`作为假设，所以我们完全可以给出完整的证明`Module.finrank_pos_iff.1 h`，但对于更复杂的情况来说，了解这一点是好的。

根据定义，`FiniteDimensional K V`可以从任何基中读取。

```py
variable  {ι  :  Type*}  (B  :  Basis  ι  K  V)

example  [Finite  ι]  :  FiniteDimensional  K  V  :=  FiniteDimensional.of_fintype_basis  B

example  [FiniteDimensional  K  V]  :  Finite  ι  :=
  (FiniteDimensional.fintypeBasisIndex  B).finite
end 
```

利用对应于线性子空间的子类型具有向量空间结构，我们可以谈论子空间的维度。

```py
section
variable  (E  F  :  Submodule  K  V)  [FiniteDimensional  K  V]

open  Module

example  :  finrank  K  (E  ⊔  F  :  Submodule  K  V)  +  finrank  K  (E  ⊓  F  :  Submodule  K  V)  =
  finrank  K  E  +  finrank  K  F  :=
  Submodule.finrank_sup_add_finrank_inf_eq  E  F

example  :  finrank  K  E  ≤  finrank  K  V  :=  Submodule.finrank_le  E 
```

在上述第一个陈述中，类型注解的目的是确保将强制转换为`Type*`不会太早触发。

我们现在可以准备一个关于`finrank`和子空间的练习。

```py
example  (h  :  finrank  K  V  <  finrank  K  E  +  finrank  K  F)  :
  Nontrivial  (E  ⊓  F  :  Submodule  K  V)  :=  by
  sorry
end 
```

现在我们转向维度理论的一般情况。在这种情况下，`finrank`是无用的，但我们仍然有，对于任何相同向量空间的两个基，它们索引的类型之间存在一个双射。因此，我们仍然可以希望将秩定义为基数，即“在存在双射等价关系下的类型集合的商”中的一个元素。

当讨论基数时，就像我们在本书的其他地方所做的那样，很难忽视围绕罗素悖论的基础性问题。没有所有类型的类型，因为这会导致逻辑上的不一致。这个问题通过我们通常试图忽略的宇宙层次结构得到解决。

每种类型都有一个宇宙级别，这些级别的行为类似于自然数。特别是存在零级，相应的宇宙`Type 0`简单地表示为`Type`。这个宇宙足以容纳几乎所有经典数学。例如`ℕ`和`ℝ`具有`Type`类型。每个级别`u`都有一个后继，表示为`u + 1`，并且`Type u`具有类型`Type (u+1)`。

但宇宙级别不是自然数，它们具有非常不同的性质，并且没有类型。特别是你无法在 Lean 中声明类似于`u ≠ u + 1`的东西。根本不存在这样的类型。甚至声明`Type u ≠ Type (u+1)`也没有任何意义，因为`Type u`和`Type (u+1)`具有不同的类型。

无论何时我们写`Type*`，Lean 都会插入一个名为`u_n`的宇宙级别变量，其中`n`是一个数字。这允许定义和陈述存在于所有宇宙中。

给定一个宇宙级别`u`，我们可以在`Type u`上定义一个等价关系，即如果存在两个类型`α`和`β`之间的双射，则这两个类型`α`和`β`是等价的。商类型`Cardinal.{u}`生活在`Type (u+1)`中。大括号表示一个宇宙变量。在这个商中`α : Type u`的像是`Cardinal.mk α : Cardinal.{u}`。

但我们无法直接比较不同宇宙中的基数。因此，技术上我们无法将向量空间`V`的秩定义为所有类型索引`V`的基的基数。因此，它被定义为所有线性无关集合的基数`Module.rank K V`的上确界。如果`V`的宇宙级别为`u`，则其秩具有类型`Cardinal.{u}`。

```py
#check  V  -- Type u_2
#check  Module.rank  K  V  -- Cardinal.{u_2} 
```

仍然可以将这个定义与基数相关联。确实，在宇宙级别上也有一个交换的`max`操作，并且给定两个宇宙级别`u`和`v`，存在一个操作`Cardinal.lift.{u, v} : Cardinal.{v} → Cardinal.{max v u}`，它允许将基数放入一个共同的宇宙并陈述维度定理。

```py
universe  u  v  -- `u` and `v` will denote universe levels

variable  {ι  :  Type  u}  (B  :  Basis  ι  K  V)
  {ι'  :  Type  v}  (B'  :  Basis  ι'  K  V)

example  :  Cardinal.lift.{v,  u}  (.mk  ι)  =  Cardinal.lift.{u,  v}  (.mk  ι')  :=
  mk_eq_mk_of_basis  B  B' 
```

我们可以使用从自然数到有限基数的强制转换（或者更精确地说，是生活在`Cardinal.{v}`中的有限基数，其中`v`是`V`的宇宙级别）将有限维情况与这个讨论联系起来。

```py
example  [FiniteDimensional  K  V]  :
  (Module.finrank  K  V  :  Cardinal)  =  Module.rank  K  V  :=
  Module.finrank_eq_rank  K  V 
```

### 10.4.1\. 矩阵

在介绍抽象向量空间的基之前，我们回到更基础的线性代数设置，即对于某个域$K$的$K^n$。在这里，主要对象是向量和矩阵。对于具体的向量，可以使用`![…]`表示法，其中分量由逗号分隔。对于具体的矩阵，我们可以使用`!![…]`表示法，行由分号分隔，行的分量由冒号分隔。当条目具有可计算类型，如`ℕ`或`ℚ`时，我们可以使用`eval`命令进行基本操作。

```py
section  matrices

-- Adding vectors
#eval  ![1,  2]  +  ![3,  4]  -- ![4, 6]

-- Adding matrices
#eval  !![1,  2;  3,  4]  +  !![3,  4;  5,  6]  -- !![4, 6; 8, 10]

-- Multiplying matrices
#eval  !![1,  2;  3,  4]  *  !![3,  4;  5,  6]  -- !![13, 16; 29, 36] 
```

重要的是要理解这种`#eval`的使用仅对探索有趣，它不是用来替换计算机代数系统，如 Sage。这里用于矩阵的数据表示在计算上没有任何效率。它使用函数而不是数组，并且优化用于证明，而不是计算。`#eval`使用的虚拟机也不是为此用途优化的。

注意矩阵表示法列表行，但向量表示法既不是行向量也不是列向量。从左（分别）乘以矩阵的向量将向量解释为行（分别）向量。这对应于`Matrix.vecMul`操作，表示为`ᵥ*`，以及`Matrix.mulVec`操作，表示为` *ᵥ`。这些表示法在`Matrix`命名空间中定义，因此我们需要打开它。

```py
open  Matrix

-- matrices acting on vectors on the left
#eval  !![1,  2;  3,  4]  *ᵥ  ![1,  1]  -- ![3, 7]

-- matrices acting on vectors on the left, resulting in a size one matrix
#eval  !![1,  2]  *ᵥ  ![1,  1]  -- ![3]

-- matrices acting on vectors on the right
#eval  ![1,  1,  1]  ᵥ*  !![1,  2;  3,  4;  5,  6]  -- ![9, 12] 
```

为了生成具有由向量指定的相同行或列的矩阵，我们使用`Matrix.replicateRow`和`Matrix.replicateCol`，其中参数是索引行或列的类型和向量。例如，可以得到单行或单列矩阵（更精确地说，行或列由`Fin 1`索引的矩阵）。

```py
#eval  replicateRow  (Fin  1)  ![1,  2]  -- !![1, 2]

#eval  replicateCol  (Fin  1)  ![1,  2]  -- !![1; 2] 
```

其他熟悉的操作包括向量点积、矩阵转置，对于方阵，还有行列式和迹。

```py
-- vector dot product
#eval  ![1,  2]  ⬝ᵥ  ![3,  4]  -- `11`

-- matrix transpose
#eval  !![1,  2;  3,  4]ᵀ  -- `!![1, 3; 2, 4]`

-- determinant
#eval  !![(1  :  ℤ),  2;  3,  4].det  -- `-2`

-- trace
#eval  !![(1  :  ℤ),  2;  3,  4].trace  -- `5` 
```

当条目没有可计算类型时，例如如果它们是实数，我们无法期望`#eval`能有所帮助。此外，这种评估不能在没有显著扩大可信代码库的情况下用于证明（即检查证明时需要信任的 Lean 的部分）。

因此，在证明中也很好使用`simp`和`norm_num`策略，或者它们的命令对应部分进行快速探索。

```py
#simp  !![(1  :  ℝ),  2;  3,  4].det  -- `4 - 2*3`

#norm_num  !![(1  :  ℝ),  2;  3,  4].det  -- `-2`

#norm_num  !![(1  :  ℝ),  2;  3,  4].trace  -- `5`

variable  (a  b  c  d  :  ℝ)  in
#simp  !![a,  b;  c,  d].det  -- `a * d – b * c` 
```

方阵上的下一个重要操作是求逆。与数的除法总是定义并返回除以零的人工值零一样，求逆操作在所有矩阵上定义，对于不可逆矩阵返回零矩阵。

更确切地说，有一个通用的函数`Ring.inverse`在任意环中执行此操作，并且对于任何矩阵`A`，`A⁻¹`定义为`Ring.inverse A.det • A.adjugate`。根据克莱姆法则，当`A`的行列式不为零时，这确实是`A`的逆。

```py
#norm_num  [Matrix.inv_def]  !![(1  :  ℝ),  2;  3,  4]⁻¹  -- !![-2, 1; 3 / 2, -(1 / 2)] 
```

当然，这个定义对于可逆矩阵来说非常有用。有一个通用的类型类 `Invertible` 帮助记录这一点。例如，在下一个例子中的 `simp` 调用将使用具有 `Invertible` 类型类假设的 `inv_mul_of_invertible` 引理，因此只有在类型类合成系统可以找到它的情况下才会触发。在这里，我们使用 `have` 语句使这一事实可用。

```py
example  :  !![(1  :  ℝ),  2;  3,  4]⁻¹  *  !![(1  :  ℝ),  2;  3,  4]  =  1  :=  by
  have  :  Invertible  !![(1  :  ℝ),  2;  3,  4]  :=  by
  apply  Matrix.invertibleOfIsUnitDet
  norm_num
  simp 
```

在这个完全具体的例子中，我们也可以使用 `norm_num` 机制和 `apply?` 来找到最终的行：

```py
example  :  !![(1  :  ℝ),  2;  3,  4]⁻¹  *  !![(1  :  ℝ),  2;  3,  4]  =  1  :=  by
  norm_num  [Matrix.inv_def]
  exact  one_fin_two.symm 
```

所有上述的矩阵都具有它们的行和列通过 `Fin n`（对于某个 `n`）进行索引，对于行和列不一定是相同的。但有时使用任意的有限类型来索引矩阵会更方便。例如，有限图的邻接矩阵的行和列自然地通过图的顶点进行索引。

事实上，当仅仅想要定义矩阵而不在它们上定义任何操作时，索引类型的有限性甚至不是必需的，系数可以有任何类型，而不需要任何代数结构。因此，Mathlib 简单地将 `Matrix m n α` 定义为 `m → n → α`，对于任何类型 `m`、`n` 和 `α`，而我们迄今为止使用的矩阵具有如 `Matrix (Fin 2) (Fin 2) ℝ` 这样的类型。当然，代数运算需要对 `m`、`n` 和 `α` 有更多的假设。

注意我们不直接使用 `m → n → α` 的主要原因是为了让类型类系统理解我们的意图。例如，对于一个环 `R`，类型 `n → R` 被赋予了点乘运算，同样地 `m → n → R` 也有这种运算，但这并不是我们在矩阵上想要的乘法。

在下面的第一个例子中，我们迫使 Lean 看透 `Matrix` 的定义，并接受这个陈述是有意义的，然后通过检查所有项来证明它。

但接下来的两个例子揭示了 Lean 使用 `Fin 2 → Fin 2 → ℤ` 上的点乘，但使用 `Matrix (Fin 2) (Fin 2) ℤ` 上的矩阵乘法。

```py
section

example  :  (fun  _  ↦  1  :  Fin  2  →  Fin  2  →  ℤ)  =  !![1,  1;  1,  1]  :=  by
  ext  i  j
  fin_cases  i  <;>  fin_cases  j  <;>  rfl

example  :  (fun  _  ↦  1  :  Fin  2  →  Fin  2  →  ℤ)  *  (fun  _  ↦  1  :  Fin  2  →  Fin  2  →  ℤ)  =  !![1,  1;  1,  1]  :=  by
  ext  i  j
  fin_cases  i  <;>  fin_cases  j  <;>  rfl

example  :  !![1,  1;  1,  1]  *  !![1,  1;  1,  1]  =  !![2,  2;  2,  2]  :=  by
  norm_num 
```

为了定义矩阵作为函数而不失去 `Matrix` 在类型类合成中的好处，我们可以使用函数和矩阵之间的等价性 `Matrix.of`。这个等价性是秘密地使用 `Equiv.refl` 定义的。

例如，我们可以定义与向量 `v` 对应的范德蒙德矩阵。

```py
example  {n  :  ℕ}  (v  :  Fin  n  →  ℝ)  :
  Matrix.vandermonde  v  =  Matrix.of  (fun  i  j  :  Fin  n  ↦  v  i  ^  (j  :  ℕ))  :=
  rfl
end
end  matrices 
```

### 10.4.2\. 基

我们现在想要讨论向量空间的基。非正式地说，有许多定义这个概念的方法。可以使用一个通用性质。可以说基是一组线性无关且张成的向量族。或者可以将这些性质结合起来，直接说基是一组向量，使得每个向量都可以唯一地表示为基向量的线性组合。还有一种说法是，基提供了一个与基域 `K` 的幂的线性同构，将 `K` 视为 `K` 上的向量空间。

这种同构版本实际上是 Mathlib 在底层用作定义的版本，其他特征化都是从这个版本证明出来的。在无限基的情况下，必须对“`K`的幂”这一概念稍加小心。实际上，在这个代数背景下，只有有限线性组合才有意义。因此，我们需要作为参考的向量空间不是`K`的副本的直接积，而是一个直接和。我们可以用`⨁ i : ι, K`来表示某个类型`ι`索引基，但我们更倾向于使用更专业的拼写`ι →₀ K`，这意味着“从`ι`到`K`的有限支撑函数”，即函数在`ι`的有限集合外为零（这个有限集合不是固定的，它依赖于函数）。将来自基`B`的这样一个函数在向量`v`和`i : ι`上求值，返回`v`在`i`-th 基向量上的分量（或坐标）。

`V`作为`K`向量空间的类型`ι`的基的类型是`Basis ι K V`。这个同构被称为`Basis.repr`。

```py
variable  {K  :  Type*}  [Field  K]  {V  :  Type*}  [AddCommGroup  V]  [Module  K  V]

section

variable  {ι  :  Type*}  (B  :  Basis  ι  K  V)  (v  :  V)  (i  :  ι)

-- The basis vector with index ``i``
#check  (B  i  :  V)

-- the linear isomorphism with the model space given by ``B``
#check  (B.repr  :  V  ≃ₗ[K]  ι  →₀  K)

-- the component function of ``v``
#check  (B.repr  v  :  ι  →₀  K)

-- the component of ``v`` with index ``i``
#check  (B.repr  v  i  :  K) 
```

一个人可以从一个线性无关且生成向量的集合`b`开始，而不是从这样的同构开始，这就是`Basis.mk`。

家庭是生成集的假设被表述为`⊤ ≤ Submodule.span K (Set.range b)`。这里`⊤`是`V`的顶子模，即`V`作为自身的子模。这种表述看起来有点复杂，但下面我们将看到它几乎等同于更易读的`∀ v, v ∈ Submodule.span K (Set.range b)`（下面片段中的下划线指的是无用的信息`v ∈ ⊤`）。

```py
noncomputable  example  (b  :  ι  →  V)  (b_indep  :  LinearIndependent  K  b)
  (b_spans  :  ∀  v,  v  ∈  Submodule.span  K  (Set.range  b))  :  Basis  ι  K  V  :=
  Basis.mk  b_indep  (fun  v  _  ↦  b_spans  v)

-- The family of vectors underlying the above basis is indeed ``b``.
example  (b  :  ι  →  V)  (b_indep  :  LinearIndependent  K  b)
  (b_spans  :  ∀  v,  v  ∈  Submodule.span  K  (Set.range  b))  (i  :  ι)  :
  Basis.mk  b_indep  (fun  v  _  ↦  b_spans  v)  i  =  b  i  :=
  Basis.mk_apply  b_indep  (fun  v  _  ↦  b_spans  v)  i 
```

特别是模型向量空间`ι →₀ K`有一个所谓的规范基，其`repr`函数在任意向量上的求值是恒等同构。它被称为`Finsupp.basisSingleOne`，其中`Finsupp`表示有限支撑函数，`basisSingleOne`指的是基向量是除了单个输入值外其他地方为零的函数。更精确地说，由`i : ι`索引的基向量是`Finsupp.single i 1`，这是一个有限支撑函数，在`i`处取值为`1`，在其他所有地方取值为`0`。

```py
variable  [DecidableEq  ι] 
```

```py
example  :  Finsupp.basisSingleOne.repr  =  LinearEquiv.refl  K  (ι  →₀  K)  :=
  rfl

example  (i  :  ι)  :  Finsupp.basisSingleOne  i  =  Finsupp.single  i  1  :=
  rfl 
```

当索引类型是有限的时候，有限支撑函数的故事是不必要的。在这种情况下，我们可以使用更简单的`Pi.basisFun`，它给出整个`ι → K`的基。

```py
example  [Finite  ι]  (x  :  ι  →  K)  (i  :  ι)  :  (Pi.basisFun  K  ι).repr  x  i  =  x  i  :=  by
  simp 
```

回到抽象向量空间基的一般情况，我们可以将任何向量表示为基向量的线性组合。让我们先看看有限基的简单情况。

```py
example  [Fintype  ι]  :  ∑  i  :  ι,  B.repr  v  i  •  (B  i)  =  v  :=
  B.sum_repr  v 
```

当`ι`不是有限时，上述陈述在先验上没有意义：我们无法对`ι`进行求和。然而，被求和函数的支持集是有限的（它是`B.repr v`的支持集）。但是，我们需要应用一个考虑到这一点的构造。在这里，Mathlib 使用了一个特殊目的的函数，需要一些时间来习惯：`Finsupp.linearCombination`（它建立在更一般的`Finsupp.sum`之上）。给定一个从类型`ι`到基域`K`的有限支持函数`c`以及从`ι`到`V`的任何函数`f`，`Finsupp.linearCombination K f c`是对`c`的支持集上标量乘积`c • f`的求和。特别是，我们可以用包含`c`的支持集的任何有限集上的求和来替换它。

```py
example  (c  :  ι  →₀  K)  (f  :  ι  →  V)  (s  :  Finset  ι)  (h  :  c.support  ⊆  s)  :
  Finsupp.linearCombination  K  f  c  =  ∑  i  ∈  s,  c  i  •  f  i  :=
  Finsupp.linearCombination_apply_of_mem_supported  K  h 
```

也可以假设`f`是有限支持的，并且仍然可以得到一个定义良好的求和。但是，`Finsupp.linearCombination`所做的选择与我们的基讨论相关，因为它允许我们陈述`Basis.sum_repr`的推广。

```py
example  :  Finsupp.linearCombination  K  B  (B.repr  v)  =  v  :=
  B.linearCombination_repr  v 
```

也可以想知道为什么`K`在这里是一个显式的参数，尽管它可以从`c`的类型中推断出来。关键是部分应用`Finsupp.linearCombination K f`本身是有趣的。它不是一个从`ι →₀ K`到`V`的裸函数，而是一个`K`-线性映射。

```py
variable  (f  :  ι  →  V)  in
#check  (Finsupp.linearCombination  K  f  :  (ι  →₀  K)  →ₗ[K]  V) 
```

返回到数学讨论，重要的是要理解，在形式化数学中，基中向量的表示可能没有你想象的那么有用。事实上，直接使用基的更抽象性质通常更有效。特别是，基的普遍性质将它们与其他代数中的自由对象联系起来，允许通过指定基向量的像来构造线性映射。这是`Basis.constr`。对于任何`K`-向量空间`W`，我们的基`B`给出一个线性同构`Basis.constr B K`，从`ι → W`到`V →ₗ[K] W`。这个同构的特点是它将任何函数`u : ι → W`映射到一个线性映射，该映射将基向量`B i`映射到`u i`，对于每个`i : ι`。

```py
section

variable  {W  :  Type*}  [AddCommGroup  W]  [Module  K  W]
  (φ  :  V  →ₗ[K]  W)  (u  :  ι  →  W)

#check  (B.constr  K  :  (ι  →  W)  ≃ₗ[K]  (V  →ₗ[K]  W))

#check  (B.constr  K  u  :  V  →ₗ[K]  W)

example  (i  :  ι)  :  B.constr  K  u  (B  i)  =  u  i  :=
  B.constr_basis  K  u  i 
```

这个性质确实是特征性的，因为线性映射由其在基上的值确定：

```py
example  (φ  ψ  :  V  →ₗ[K]  W)  (h  :  ∀  i,  φ  (B  i)  =  ψ  (B  i))  :  φ  =  ψ  :=
  B.ext  h 
```

如果我们在目标空间上也有一个基`B'`，那么我们可以将线性映射与矩阵相识别。这种识别是一个`K`-线性同构。

```py
variable  {ι'  :  Type*}  (B'  :  Basis  ι'  K  W)  [Fintype  ι]  [DecidableEq  ι]  [Fintype  ι']  [DecidableEq  ι']

open  LinearMap

#check  (toMatrix  B  B'  :  (V  →ₗ[K]  W)  ≃ₗ[K]  Matrix  ι'  ι  K)

open  Matrix  -- get access to the ``*ᵥ`` notation for multiplication between matrices and vectors.

example  (φ  :  V  →ₗ[K]  W)  (v  :  V)  :  (toMatrix  B  B'  φ)  *ᵥ  (B.repr  v)  =  B'.repr  (φ  v)  :=
  toMatrix_mulVec_repr  B  B'  φ  v

variable  {ι''  :  Type*}  (B''  :  Basis  ι''  K  W)  [Fintype  ι'']  [DecidableEq  ι'']

example  (φ  :  V  →ₗ[K]  W)  :  (toMatrix  B  B''  φ)  =  (toMatrix  B'  B''  .id)  *  (toMatrix  B  B'  φ)  :=  by
  simp

end 
```

作为这个主题的练习，我们将证明定理的一部分，该定理保证了内射有定义良好的行列式。具体来说，我们想要证明当两个基由相同的类型索引时，它们附加到任何内射上的矩阵有相同的行列式。这需要使用所有基都有同构索引类型来补充，以得到完整的结果。

当然，Mathlib 已经知道这一点，`simp`可以立即关闭目标，所以你不应该太早使用它，而应该使用提供的引理。

```py
open  Module  LinearMap  Matrix

-- Some lemmas coming from the fact that `LinearMap.toMatrix` is an algebra morphism.
#check  toMatrix_comp
#check  id_comp
#check  comp_id
#check  toMatrix_id

-- Some lemmas coming from the fact that ``Matrix.det`` is a multiplicative monoid morphism.
#check  Matrix.det_mul
#check  Matrix.det_one

example  [Fintype  ι]  (B'  :  Basis  ι  K  V)  (φ  :  End  K  V)  :
  (toMatrix  B  B  φ).det  =  (toMatrix  B'  B'  φ).det  :=  by
  set  M  :=  toMatrix  B  B  φ
  set  M'  :=  toMatrix  B'  B'  φ
  set  P  :=  (toMatrix  B  B')  LinearMap.id
  set  P'  :=  (toMatrix  B'  B)  LinearMap.id
  sorry
end 
```

### 10.4.3\. 维度

回到单个向量空间的情况，基对于定义维度概念也是有用的。在这里，也存在有限维向量空间的基本情况。对于这样的空间，我们期望维度是一个自然数。这是`Module.finrank`。它将基域作为显式参数，因为一个给定的阿贝尔群可以是在不同域上的向量空间。

```py
section
#check  (Module.finrank  K  V  :  ℕ)

-- `Fin n → K` is the archetypical space with dimension `n` over `K`.
example  (n  :  ℕ)  :  Module.finrank  K  (Fin  n  →  K)  =  n  :=
  Module.finrank_fin_fun  K

-- Seen as a vector space over itself, `ℂ` has dimension one.
example  :  Module.finrank  ℂ  ℂ  =  1  :=
  Module.finrank_self  ℂ

-- But as a real vector space it has dimension two.
example  :  Module.finrank  ℝ  ℂ  =  2  :=
  Complex.finrank_real_complex 
```

注意，`Module.finrank`为任何向量空间定义。对于无限维向量空间，它返回零，就像除以零返回零一样。

当然，许多引理需要有限维度的假设。这就是`FiniteDimensional`类型类的作用。例如，想想如果没有这个假设，下一个例子会如何失败。

```py
example  [FiniteDimensional  K  V]  :  0  <  Module.finrank  K  V  ↔  Nontrivial  V  :=
  Module.finrank_pos_iff 
```

在上述陈述中，`Nontrivial V`意味着`V`至少有两个不同的元素。请注意，`Module.finrank_pos_iff`没有显式参数。当从左到右使用它时这是可以的，但不是从右到左使用时，因为 Lean 无法从陈述`Nontrivial V`中猜测`K`。在这种情况下，在确认引理是在一个名为`R`的环上陈述之后，使用名称参数语法是有用的。因此，我们可以写出：

```py
example  [FiniteDimensional  K  V]  (h  :  0  <  Module.finrank  K  V)  :  Nontrivial  V  :=  by
  apply  (Module.finrank_pos_iff  (R  :=  K)).1
  exact  h 
```

上述拼写很奇怪，因为我们已经有了`h`作为假设，所以我们完全可以给出完整的证明`Module.finrank_pos_iff.1 h`，但在更复杂的情况下了解这一点是好的。

根据定义，`FiniteDimensional K V`可以从任何基中读取。

```py
variable  {ι  :  Type*}  (B  :  Basis  ι  K  V)

example  [Finite  ι]  :  FiniteDimensional  K  V  :=  FiniteDimensional.of_fintype_basis  B

example  [FiniteDimensional  K  V]  :  Finite  ι  :=
  (FiniteDimensional.fintypeBasisIndex  B).finite
end 
```

利用对应于线性子空间的子类型具有向量空间结构，我们可以谈论子空间的维度。

```py
section
variable  (E  F  :  Submodule  K  V)  [FiniteDimensional  K  V]

open  Module

example  :  finrank  K  (E  ⊔  F  :  Submodule  K  V)  +  finrank  K  (E  ⊓  F  :  Submodule  K  V)  =
  finrank  K  E  +  finrank  K  F  :=
  Submodule.finrank_sup_add_finrank_inf_eq  E  F

example  :  finrank  K  E  ≤  finrank  K  V  :=  Submodule.finrank_le  E 
```

在上述第一个陈述中，类型注解的目的是确保将`Type*`的强制转换不会太早触发。

现在我们已经准备好进行关于`finrank`和子空间的练习了。

```py
example  (h  :  finrank  K  V  <  finrank  K  E  +  finrank  K  F)  :
  Nontrivial  (E  ⊓  F  :  Submodule  K  V)  :=  by
  sorry
end 
```

现在让我们转向维度理论的通用情况。在这种情况下，`finrank`是无用的，但我们仍然有这样一个事实：对于同一向量空间中的任意两个基，这些基的类型之间存在一一对应关系。因此，我们仍然可以希望将秩定义为基数，即“在存在一一对应等价关系下的类型集合的商”中的一个元素。

当讨论基数时，就像在这本书的其他地方一样，很难忽视围绕 Russel 悖论的基础性问题。没有所有类型的类型，因为这会导致逻辑不一致。这个问题通过我们通常试图忽略的宇宙层次结构得到解决。

每个类型都有一个宇宙级别，这些级别与自然数的行为相似。特别是有一个零级，相应的宇宙`Type 0`简单地表示为`Type`。这个宇宙足以容纳几乎所有经典数学。例如`ℕ`和`ℝ`的类型是`Type`。每个级别`u`都有一个后继，表示为`u + 1`，而`Type u`的类型是`Type (u+1)`。

但是宇宙级别不是自然数，它们具有非常不同的性质，并且没有类型。特别是你无法在 Lean 中声明类似于 `u ≠ u + 1` 的内容。根本不存在这样的类型。甚至声明 `Type u ≠ Type (u+1)` 也没有意义，因为 `Type u` 和 `Type (u+1)` 具有不同的类型。

每当我们写 `Type*` 时，Lean 会插入一个名为 `u_n` 的宇宙级别变量，其中 `n` 是一个数字。这允许定义和陈述存在于所有宇宙中。

给定一个宇宙级别 `u`，我们可以在 `Type u` 上定义一个等价关系，即如果存在两个类型 `α` 和 `β` 之间的双射，则这两个类型 `α` 和 `β` 是等价的。商类型 `Cardinal.{u}` 居于 `Type (u+1)` 中。大括号表示一个宇宙变量。在这个商中，`α : Type u` 的像为 `Cardinal.mk α : Cardinal.{u}`。

但是我们无法直接比较不同宇宙中的基数。因此，技术上我们无法将向量空间 `V` 的秩定义为所有基索引类型的基数。因此，它被定义为 `V` 中所有线性无关集的基数 `Module.rank K V` 的上确界。如果 `V` 的宇宙级别为 `u`，则其秩具有类型 `Cardinal.{u}`。

```py
#check  V  -- Type u_2
#check  Module.rank  K  V  -- Cardinal.{u_2} 
```

尽管如此，我们仍然可以将这个定义与基联系起来。确实，在宇宙级别上也有一个交换的 `max` 操作，并且对于两个宇宙级别 `u` 和 `v`，存在一个操作 `Cardinal.lift.{u, v} : Cardinal.{v} → Cardinal.{max v u}`，它允许将基数放入一个共同的宇宙中并陈述维度定理。

```py
universe  u  v  -- `u` and `v` will denote universe levels

variable  {ι  :  Type  u}  (B  :  Basis  ι  K  V)
  {ι'  :  Type  v}  (B'  :  Basis  ι'  K  V)

example  :  Cardinal.lift.{v,  u}  (.mk  ι)  =  Cardinal.lift.{u,  v}  (.mk  ι')  :=
  mk_eq_mk_of_basis  B  B' 
```

我们可以使用从自然数到有限基数（或者更精确地说，居住在 `Cardinal.{v}` 中的有限基数，其中 `v` 是 `V` 的宇宙级别）的强制转换将有限维情况与这次讨论联系起来。

```py
example  [FiniteDimensional  K  V]  :
  (Module.finrank  K  V  :  Cardinal)  =  Module.rank  K  V  :=
  Module.finrank_eq_rank  K  V 
```*
