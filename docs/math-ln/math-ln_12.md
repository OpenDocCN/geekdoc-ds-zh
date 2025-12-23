# 12\. 微分学

> 原文：[`leanprover-community.github.io/mathematics_in_lean/C12_Differential_Calculus.html`](https://leanprover-community.github.io/mathematics_in_lean/C12_Differential_Calculus.html)

*Lean 中的数学* **   12. 微分学

+   查看页面源代码

* * *

我们现在考虑从*分析*中的概念的形式化，从本章的微分开始，并在下一章转向积分和测度理论。在第 12.1 节，我们坚持使用从实数到实数的函数的设置，这在任何初等微积分课程中都很熟悉。在第 12.2 节，我们随后考虑在更广泛的设置中的导数概念。

## 12.1\. 基础微分学

设`f`是从实数到实数的函数。在谈论`f`在单一点的导数和谈论导数函数之间有一个区别。在 Mathlib 中，第一个概念如下表示。

```py
open  Real

/-- The sin function has derivative 1 at 0\. -/
example  :  HasDerivAt  sin  1  0  :=  by  simpa  using  hasDerivAt_sin  0 
```

我们也可以通过写作`DifferentiableAt ℝ`来表示函数在一点可导，而不必指定其在该点的导数。我们明确指定`ℝ`是因为在稍微更一般的情况下，当我们谈论从`ℂ`到`ℂ`的函数时，我们希望能够区分在实意义下可导和在复导数意义下可导。

```py
example  (x  :  ℝ)  :  DifferentiableAt  ℝ  sin  x  :=
  (hasDerivAt_sin  x).differentiableAt 
```

每次我们想要引用导数时都必须提供可导性的证明将会很麻烦。因此，Mathlib 提供了一个函数`deriv f : ℝ → ℝ`，它对任何函数`f : ℝ → ℝ`都定义，但在`f`不可导的任何点上定义为`0`。

```py
example  {f  :  ℝ  →  ℝ}  {x  a  :  ℝ}  (h  :  HasDerivAt  f  a  x)  :  deriv  f  x  =  a  :=
  h.deriv

example  {f  :  ℝ  →  ℝ}  {x  :  ℝ}  (h  :  ¬DifferentiableAt  ℝ  f  x)  :  deriv  f  x  =  0  :=
  deriv_zero_of_not_differentiableAt  h 
```

当然，关于`deriv`的引理有很多确实需要可导性假设。例如，你应该考虑在没有可导性假设的情况下，下一个引理的反例。

```py
example  {f  g  :  ℝ  →  ℝ}  {x  :  ℝ}  (hf  :  DifferentiableAt  ℝ  f  x)  (hg  :  DifferentiableAt  ℝ  g  x)  :
  deriv  (f  +  g)  x  =  deriv  f  x  +  deriv  g  x  :=
  deriv_add  hf  hg 
```

然而，有趣的是，有一些陈述可以通过利用`deriv`的值在函数不可导时默认为零的事实来避免可导性假设。因此，理解以下陈述需要知道`deriv`的确切定义。

```py
example  {f  :  ℝ  →  ℝ}  {a  :  ℝ}  (h  :  IsLocalMin  f  a)  :  deriv  f  a  =  0  :=
  h.deriv_eq_zero 
```

我们甚至可以在没有任何可导性假设的情况下陈述罗尔定理，这看起来甚至更奇怪。

```py
open  Set

example  {f  :  ℝ  →  ℝ}  {a  b  :  ℝ}  (hab  :  a  <  b)  (hfc  :  ContinuousOn  f  (Icc  a  b))  (hfI  :  f  a  =  f  b)  :
  ∃  c  ∈  Ioo  a  b,  deriv  f  c  =  0  :=
  exists_deriv_eq_zero  hab  hfc  hfI 
```

当然，这个技巧对一般平均值定理不适用。

```py
example  (f  :  ℝ  →  ℝ)  {a  b  :  ℝ}  (hab  :  a  <  b)  (hf  :  ContinuousOn  f  (Icc  a  b))
  (hf'  :  DifferentiableOn  ℝ  f  (Ioo  a  b))  :  ∃  c  ∈  Ioo  a  b,  deriv  f  c  =  (f  b  -  f  a)  /  (b  -  a)  :=
  exists_deriv_eq_slope  f  hab  hf  hf' 
```

Lean 可以使用`simp`策略自动计算一些简单的导数。

```py
example  :  deriv  (fun  x  :  ℝ  ↦  x  ^  5)  6  =  5  *  6  ^  4  :=  by  simp

example  :  deriv  sin  π  =  -1  :=  by  simp 
```  ## 12.2\. 范数空间中的微分学

### 12.2.1\. 范数空间

可以使用**范数向量空间**的概念将微分推广到`ℝ`之外，这个概念包含了方向和距离。我们首先从**范数群**的概念开始，这是一个加法交换群，它配备了一个满足以下条件的实值范数函数。

```py
variable  {E  :  Type*}  [NormedAddCommGroup  E]

example  (x  :  E)  :  0  ≤  ‖x‖  :=
  norm_nonneg  x

example  {x  :  E}  :  ‖x‖  =  0  ↔  x  =  0  :=
  norm_eq_zero

example  (x  y  :  E)  :  ‖x  +  y‖  ≤  ‖x‖  +  ‖y‖  :=
  norm_add_le  x  y 
```

每个范数空间都是一个带有距离函数 $d(x, y) = \| x - y \|$ 的度量空间，因此它也是一个拓扑空间。Lean 和 Mathlib 都知道这一点。

```py
example  :  MetricSpace  E  :=  by  infer_instance

example  {X  :  Type*}  [TopologicalSpace  X]  {f  :  X  →  E}  (hf  :  Continuous  f)  :
  Continuous  fun  x  ↦  ‖f  x‖  :=
  hf.norm 
```

为了使用线性代数中的范数概念，我们在`NormedAddGroup E`之上添加了假设`NormedSpace ℝ E`。这规定`E`是`ℝ`上的一个向量空间，并且标量乘法满足以下条件。

```py
variable  [NormedSpace  ℝ  E]

example  (a  :  ℝ)  (x  :  E)  :  ‖a  •  x‖  =  |a|  *  ‖x‖  :=
  norm_smul  a  x 
```

完全范数空间被称为**Banach 空间**。每个有限维向量空间都是完备的。

```py
example  [FiniteDimensional  ℝ  E]  :  CompleteSpace  E  :=  by  infer_instance 
```

在所有之前的例子中，我们使用了实数作为基域。更一般地，我们可以对任何**非平凡范数域**上的向量空间进行微积分。这些是配备了实值范数且范数乘法具有性质（即并非每个元素都有范数零或一，或者说存在一个范数大于一的元素）的域。

```py
example  (𝕜  :  Type*)  [NontriviallyNormedField  𝕜]  (x  y  :  𝕜)  :  ‖x  *  y‖  =  ‖x‖  *  ‖y‖  :=
  norm_mul  x  y

example  (𝕜  :  Type*)  [NontriviallyNormedField  𝕜]  :  ∃  x  :  𝕜,  1  <  ‖x‖  :=
  NormedField.exists_one_lt_norm  𝕜 
```

在非平凡范数域上的有限维向量空间只要该域本身是完备的，就是完备的。

```py
example  (𝕜  :  Type*)  [NontriviallyNormedField  𝕜]  (E  :  Type*)  [NormedAddCommGroup  E]
  [NormedSpace  𝕜  E]  [CompleteSpace  𝕜]  [FiniteDimensional  𝕜  E]  :  CompleteSpace  E  :=
  FiniteDimensional.complete  𝕜  E 
```

### 12.2.2\. 连续线性映射

现在我们转向范数空间范畴中的态射，即连续线性映射。在 Mathlib 中，范数空间`E`和`F`之间`𝕜`-线性连续映射的类型被写成`E →L[𝕜] F`。它们被实现为**捆绑映射**，这意味着这个类型的元素包含函数本身以及线性性和连续性的属性。Lean 将插入一个强制转换，以便连续线性映射可以被视为一个函数。

```py
variable  {𝕜  :  Type*}  [NontriviallyNormedField  𝕜]  {E  :  Type*}  [NormedAddCommGroup  E]
  [NormedSpace  𝕜  E]  {F  :  Type*}  [NormedAddCommGroup  F]  [NormedSpace  𝕜  F]

example  :  E  →L[𝕜]  E  :=
  ContinuousLinearMap.id  𝕜  E

example  (f  :  E  →L[𝕜]  F)  :  E  →  F  :=
  f

example  (f  :  E  →L[𝕜]  F)  :  Continuous  f  :=
  f.cont

example  (f  :  E  →L[𝕜]  F)  (x  y  :  E)  :  f  (x  +  y)  =  f  x  +  f  y  :=
  f.map_add  x  y

example  (f  :  E  →L[𝕜]  F)  (a  :  𝕜)  (x  :  E)  :  f  (a  •  x)  =  a  •  f  x  :=
  f.map_smul  a  x 
```

连续线性映射有一个由以下性质表征的算子范数。

```py
variable  (f  :  E  →L[𝕜]  F)

example  (x  :  E)  :  ‖f  x‖  ≤  ‖f‖  *  ‖x‖  :=
  f.le_opNorm  x

example  {M  :  ℝ}  (hMp  :  0  ≤  M)  (hM  :  ∀  x,  ‖f  x‖  ≤  M  *  ‖x‖)  :  ‖f‖  ≤  M  :=
  f.opNorm_le_bound  hMp  hM 
```

还有一个关于捆绑连续线性**同构**的概念。这类同构的类型是`E ≃L[𝕜] F`。

作为一项具有挑战性的练习，你可以证明 Banach-Steinhaus 定理，也称为一致有界性原理。该原理表明，从 Banach 空间到范数空间的一族连续线性映射在每一点上是有界的，那么这些线性映射的范数是一致有界的。主要成分是 Baire 定理`nonempty_interior_of_iUnion_of_closed`。（你在拓扑章节中证明了这一版本。）次要成分包括`continuous_linear_map.opNorm_le_of_shell`、`interior_subset`和`interior_iInter_subset`以及`isClosed_le`。

```py
variable  {𝕜  :  Type*}  [NontriviallyNormedField  𝕜]  {E  :  Type*}  [NormedAddCommGroup  E]
  [NormedSpace  𝕜  E]  {F  :  Type*}  [NormedAddCommGroup  F]  [NormedSpace  𝕜  F]

open  Metric

example  {ι  :  Type*}  [CompleteSpace  E]  {g  :  ι  →  E  →L[𝕜]  F}  (h  :  ∀  x,  ∃  C,  ∀  i,  ‖g  i  x‖  ≤  C)  :
  ∃  C',  ∀  i,  ‖g  i‖  ≤  C'  :=  by
  -- sequence of subsets consisting of those `x : E` with norms `‖g i x‖` bounded by `n`
  let  e  :  ℕ  →  Set  E  :=  fun  n  ↦  ⋂  i  :  ι,  {  x  :  E  |  ‖g  i  x‖  ≤  n  }
  -- each of these sets is closed
  have  hc  :  ∀  n  :  ℕ,  IsClosed  (e  n)
  sorry
  -- the union is the entire space; this is where we use `h`
  have  hU  :  (⋃  n  :  ℕ,  e  n)  =  univ
  sorry
  /- apply the Baire category theorem to conclude that for some `m : ℕ`,
 `e m` contains some `x` -/
  obtain  ⟨m,  x,  hx⟩  :  ∃  m,  ∃  x,  x  ∈  interior  (e  m)  :=  sorry
  obtain  ⟨ε,  ε_pos,  hε⟩  :  ∃  ε  >  0,  ball  x  ε  ⊆  interior  (e  m)  :=  sorry
  obtain  ⟨k,  hk⟩  :  ∃  k  :  𝕜,  1  <  ‖k‖  :=  sorry
  -- show all elements in the ball have norm bounded by `m` after applying any `g i`
  have  real_norm_le  :  ∀  z  ∈  ball  x  ε,  ∀  (i  :  ι),  ‖g  i  z‖  ≤  m
  sorry
  have  εk_pos  :  0  <  ε  /  ‖k‖  :=  sorry
  refine  ⟨(m  +  m  :  ℕ)  /  (ε  /  ‖k‖),  fun  i  ↦  ContinuousLinearMap.opNorm_le_of_shell  ε_pos  ?_  hk  ?_⟩
  sorry
  sorry 
```

### 12.2.3\. 渐近比较

定义可微性还需要进行渐近比较。Mathlib 有一个广泛的库，涵盖了大的 O 和小的 o 关系，其定义如下所示。打开 `asymptotics` 局域允许我们使用相应的符号。这里我们只使用小的 o 来定义可微性。

```py
open  Asymptotics

example  {α  :  Type*}  {E  :  Type*}  [NormedGroup  E]  {F  :  Type*}  [NormedGroup  F]  (c  :  ℝ)
  (l  :  Filter  α)  (f  :  α  →  E)  (g  :  α  →  F)  :  IsBigOWith  c  l  f  g  ↔  ∀ᶠ  x  in  l,  ‖f  x‖  ≤  c  *  ‖g  x‖  :=
  isBigOWith_iff

example  {α  :  Type*}  {E  :  Type*}  [NormedGroup  E]  {F  :  Type*}  [NormedGroup  F]
  (l  :  Filter  α)  (f  :  α  →  E)  (g  :  α  →  F)  :  f  =O[l]  g  ↔  ∃  C,  IsBigOWith  C  l  f  g  :=
  isBigO_iff_isBigOWith

example  {α  :  Type*}  {E  :  Type*}  [NormedGroup  E]  {F  :  Type*}  [NormedGroup  F]
  (l  :  Filter  α)  (f  :  α  →  E)  (g  :  α  →  F)  :  f  =o[l]  g  ↔  ∀  C  >  0,  IsBigOWith  C  l  f  g  :=
  isLittleO_iff_forall_isBigOWith

example  {α  :  Type*}  {E  :  Type*}  [NormedAddCommGroup  E]  (l  :  Filter  α)  (f  g  :  α  →  E)  :
  f  ~[l]  g  ↔  (f  -  g)  =o[l]  g  :=
  Iff.rfl 
```

### 12.2.4\. 可微性

我们现在可以讨论范数空间之间的可微函数。类似于一维的初等情况，Mathlib 定义了一个谓词 `HasFDerivAt` 和一个函数 `fderiv`。这里的字母“f”代表 *Fréchet*。

```py
open  Topology

variable  {𝕜  :  Type*}  [NontriviallyNormedField  𝕜]  {E  :  Type*}  [NormedAddCommGroup  E]
  [NormedSpace  𝕜  E]  {F  :  Type*}  [NormedAddCommGroup  F]  [NormedSpace  𝕜  F]

example  (f  :  E  →  F)  (f'  :  E  →L[𝕜]  F)  (x₀  :  E)  :
  HasFDerivAt  f  f'  x₀  ↔  (fun  x  ↦  f  x  -  f  x₀  -  f'  (x  -  x₀))  =o[𝓝  x₀]  fun  x  ↦  x  -  x₀  :=
  hasFDerivAtFilter_iff_isLittleO  ..

example  (f  :  E  →  F)  (f'  :  E  →L[𝕜]  F)  (x₀  :  E)  (hff'  :  HasFDerivAt  f  f'  x₀)  :  fderiv  𝕜  f  x₀  =  f'  :=
  hff'.fderiv 
```

我们还有迭代导数，其值在多线性映射类型 `E [×n]→L[𝕜] F` 中，并且我们有连续可微函数。类型 `ℕ∞` 是 `ℕ` 加上一个额外的元素 `∞`，它比任何自然数都大。因此，$\mathcal{C}^\infty$ 函数是满足 `ContDiff 𝕜 ⊤ f` 的函数 `f`。

```py
example  (n  :  ℕ)  (f  :  E  →  F)  :  E  →  E[×n]→L[𝕜]  F  :=
  iteratedFDeriv  𝕜  n  f

example  (n  :  ℕ∞)  {f  :  E  →  F}  :
  ContDiff  𝕜  n  f  ↔
  (∀  m  :  ℕ,  (m  :  WithTop  ℕ)  ≤  n  →  Continuous  fun  x  ↦  iteratedFDeriv  𝕜  m  f  x)  ∧
  ∀  m  :  ℕ,  (m  :  WithTop  ℕ)  <  n  →  Differentiable  𝕜  fun  x  ↦  iteratedFDeriv  𝕜  m  f  x  :=
  contDiff_iff_continuous_differentiable 
```

`ContDiff` 中的可微参数也可以取值 `ω : WithTop ℕ∞` 来表示解析函数。

有一个更严格的可微性概念称为 `HasStrictFDerivAt`，它在逆函数定理和隐函数定理的陈述中使用，这两个定理都在 Mathlib 中。在 `ℝ` 或 `ℂ` 上，连续可微函数是严格可微的。

```py
example  {𝕂  :  Type*}  [RCLike  𝕂]  {E  :  Type*}  [NormedAddCommGroup  E]  [NormedSpace  𝕂  E]  {F  :  Type*}
  [NormedAddCommGroup  F]  [NormedSpace  𝕂  F]  {f  :  E  →  F}  {x  :  E}  {n  :  WithTop  ℕ∞}
  (hf  :  ContDiffAt  𝕂  n  f  x)  (hn  :  1  ≤  n)  :  HasStrictFDerivAt  f  (fderiv  𝕂  f  x)  x  :=
  hf.hasStrictFDerivAt  hn 
```

局部逆定理是通过一个操作来陈述的，该操作从一个函数和假设函数在点 `a` 处严格可微以及其导数是一个同构来生成一个逆函数。

下面的第一个例子得到了这个局部逆。下一个例子陈述了它确实是左和右的局部逆，并且它是严格可微的。

```py
section  LocalInverse
variable  [CompleteSpace  E]  {f  :  E  →  F}  {f'  :  E  ≃L[𝕜]  F}  {a  :  E}

example  (hf  :  HasStrictFDerivAt  f  (f'  :  E  →L[𝕜]  F)  a)  :  F  →  E  :=
  HasStrictFDerivAt.localInverse  f  f'  a  hf

example  (hf  :  HasStrictFDerivAt  f  (f'  :  E  →L[𝕜]  F)  a)  :
  ∀ᶠ  x  in  𝓝  a,  hf.localInverse  f  f'  a  (f  x)  =  x  :=
  hf.eventually_left_inverse

example  (hf  :  HasStrictFDerivAt  f  (f'  :  E  →L[𝕜]  F)  a)  :
  ∀ᶠ  x  in  𝓝  (f  a),  f  (hf.localInverse  f  f'  a  x)  =  x  :=
  hf.eventually_right_inverse

example  {f  :  E  →  F}  {f'  :  E  ≃L[𝕜]  F}  {a  :  E}
  (hf  :  HasStrictFDerivAt  f  (f'  :  E  →L[𝕜]  F)  a)  :
  HasStrictFDerivAt  (HasStrictFDerivAt.localInverse  f  f'  a  hf)  (f'.symm  :  F  →L[𝕜]  E)  (f  a)  :=
  HasStrictFDerivAt.to_localInverse  hf

end  LocalInverse 
```

这只是对 Mathlib 中的微分学的快速浏览。库中包含了许多我们没有讨论的变体。例如，你可能想在单变量设置中使用单侧导数。这样做的方法在 Mathlib 的更一般背景下可以找到；参见 `HasFDerivWithinAt` 或更通用的 `HasFDerivAtFilter`。上一节 下一节

* * *

© 版权 2020-2025，Jeremy Avigad，Patrick Massot。文本许可协议为 CC BY 4.0。

使用 [Sphinx](https://www.sphinx-doc.org/) 和由 [Read the Docs](https://readthedocs.org) 提供的 [主题](https://github.com/readthedocs/sphinx_rtd_theme) 构建。我们现在考虑从 *分析* 的概念形式化，从本章的微分开始，并在下一章转向积分和测度理论。在 第 12.1 节 中，我们坚持使用从实数到实数的函数的设置，这在任何初等微积分课程中都很熟悉。在 第 12.2 节 中，我们随后考虑在更广泛的设置中的导数概念。

## 12.1\. 初等微分学

设 `f` 是从实数到实数的函数。在谈论 `f` 在某一点的导数和谈论导数函数之间有一个区别。在 Mathlib 中，第一个概念表示如下。

```py
open  Real

/-- The sin function has derivative 1 at 0\. -/
example  :  HasDerivAt  sin  1  0  :=  by  simpa  using  hasDerivAt_sin  0 
```

我们也可以通过写 `DifferentiableAt ℝ` 来表达 `f` 在某一点可微，而不必指定其在该点的导数。我们明确指定 `ℝ`，因为在稍微更一般的环境中，当我们谈论从 `ℂ` 到 `ℂ` 的函数时，我们希望能够区分在实意义下可微和在复导数意义下可微。

```py
example  (x  :  ℝ)  :  DifferentiableAt  ℝ  sin  x  :=
  (hasDerivAt_sin  x).differentiableAt 
```

每次我们想要引用导数时都必须提供可微性的证明将会很麻烦。因此，Mathlib 提供了一个函数 `deriv f : ℝ → ℝ`，它对任何函数 `f : ℝ → ℝ` 都有定义，但在 `f` 不可微的点被定义为取值 `0`。

```py
example  {f  :  ℝ  →  ℝ}  {x  a  :  ℝ}  (h  :  HasDerivAt  f  a  x)  :  deriv  f  x  =  a  :=
  h.deriv

example  {f  :  ℝ  →  ℝ}  {x  :  ℝ}  (h  :  ¬DifferentiableAt  ℝ  f  x)  :  deriv  f  x  =  0  :=
  deriv_zero_of_not_differentiableAt  h 
```

当然，关于 `deriv` 的引理中有很多是要求可微性假设的。例如，你应该考虑在没有可微性假设的情况下，下一个引理的反例。

```py
example  {f  g  :  ℝ  →  ℝ}  {x  :  ℝ}  (hf  :  DifferentiableAt  ℝ  f  x)  (hg  :  DifferentiableAt  ℝ  g  x)  :
  deriv  (f  +  g)  x  =  deriv  f  x  +  deriv  g  x  :=
  deriv_add  hf  hg 
```

然而，有趣的是，有一些陈述可以通过利用 `deriv` 在函数不可导时默认为零的事实来避免可微性的假设。因此，理解以下陈述需要知道 `deriv` 的精确定义。

```py
example  {f  :  ℝ  →  ℝ}  {a  :  ℝ}  (h  :  IsLocalMin  f  a)  :  deriv  f  a  =  0  :=
  h.deriv_eq_zero 
```

我们甚至可以在没有任何可微性假设的情况下陈述罗尔定理，这看起来甚至更奇怪。

```py
open  Set

example  {f  :  ℝ  →  ℝ}  {a  b  :  ℝ}  (hab  :  a  <  b)  (hfc  :  ContinuousOn  f  (Icc  a  b))  (hfI  :  f  a  =  f  b)  :
  ∃  c  ∈  Ioo  a  b,  deriv  f  c  =  0  :=
  exists_deriv_eq_zero  hab  hfc  hfI 
```

当然，这个技巧对一般平均值定理不适用。

```py
example  (f  :  ℝ  →  ℝ)  {a  b  :  ℝ}  (hab  :  a  <  b)  (hf  :  ContinuousOn  f  (Icc  a  b))
  (hf'  :  DifferentiableOn  ℝ  f  (Ioo  a  b))  :  ∃  c  ∈  Ioo  a  b,  deriv  f  c  =  (f  b  -  f  a)  /  (b  -  a)  :=
  exists_deriv_eq_slope  f  hab  hf  hf' 
```

Lean 可以使用 `simp` 策略自动计算一些简单的导数。

```py
example  :  deriv  (fun  x  :  ℝ  ↦  x  ^  5)  6  =  5  *  6  ^  4  :=  by  simp

example  :  deriv  sin  π  =  -1  :=  by  simp 
```  ## 12.2\. 赋范空间中的微分学

### 12.2.1\. 赋范空间

可以使用 *赋范向量空间* 的概念将微分推广到 `ℝ` 之外，它封装了方向和距离。我们首先从 *赋范群* 的概念开始，它是一个加法交换群，并配备了满足以下条件的实值范数函数。

```py
variable  {E  :  Type*}  [NormedAddCommGroup  E]

example  (x  :  E)  :  0  ≤  ‖x‖  :=
  norm_nonneg  x

example  {x  :  E}  :  ‖x‖  =  0  ↔  x  =  0  :=
  norm_eq_zero

example  (x  y  :  E)  :  ‖x  +  y‖  ≤  ‖x‖  +  ‖y‖  :=
  norm_add_le  x  y 
```

每个赋范空间都是一个具有距离函数 $d(x, y) = \| x - y \|$ 的度量空间，因此它也是一个拓扑空间。Lean 和 Mathlib 都知道这一点。

```py
example  :  MetricSpace  E  :=  by  infer_instance

example  {X  :  Type*}  [TopologicalSpace  X]  {f  :  X  →  E}  (hf  :  Continuous  f)  :
  Continuous  fun  x  ↦  ‖f  x‖  :=
  hf.norm 
```

为了使用来自线性代数的范数概念，我们在 `NormedAddGroup E` 上添加了 `NormedSpace ℝ E` 的假设。这规定 `E` 是一个实向量空间，并且标量乘法满足以下条件。

```py
variable  [NormedSpace  ℝ  E]

example  (a  :  ℝ)  (x  :  E)  :  ‖a  •  x‖  =  |a|  *  ‖x‖  :=
  norm_smul  a  x 
```

完全赋范空间被称为 *Banach 空间*。每个有限维向量空间都是完备的。

```py
example  [FiniteDimensional  ℝ  E]  :  CompleteSpace  E  :=  by  infer_instance 
```

在所有之前的例子中，我们使用了实数作为基域。更一般地，我们可以在任何*非平凡范数域*上的向量空间中进行微积分。这些是配备了实值范数且范数乘法且具有不是每个元素范数为零或一的特性的域（等价地，存在一个范数大于一的元素）。

```py
example  (𝕜  :  Type*)  [NontriviallyNormedField  𝕜]  (x  y  :  𝕜)  :  ‖x  *  y‖  =  ‖x‖  *  ‖y‖  :=
  norm_mul  x  y

example  (𝕜  :  Type*)  [NontriviallyNormedField  𝕜]  :  ∃  x  :  𝕜,  1  <  ‖x‖  :=
  NormedField.exists_one_lt_norm  𝕜 
```

在非平凡范数域上的有限维向量空间只要域本身是完备的，就是完备的。

```py
example  (𝕜  :  Type*)  [NontriviallyNormedField  𝕜]  (E  :  Type*)  [NormedAddCommGroup  E]
  [NormedSpace  𝕜  E]  [CompleteSpace  𝕜]  [FiniteDimensional  𝕜  E]  :  CompleteSpace  E  :=
  FiniteDimensional.complete  𝕜  E 
```

### 12.2.2\. 连续线性映射

现在我们转向范数空间范畴中的态射，即连续线性映射。在 Mathlib 中，范数空间`E`和`F`之间的`𝕜`-线性连续映射的类型被写成`E →L[𝕜] F`。它们被实现为*捆绑映射*，这意味着该类型的元素包含函数本身以及线性性和连续性的属性。Lean 将插入一个强制转换，以便连续线性映射可以被视为函数。

```py
variable  {𝕜  :  Type*}  [NontriviallyNormedField  𝕜]  {E  :  Type*}  [NormedAddCommGroup  E]
  [NormedSpace  𝕜  E]  {F  :  Type*}  [NormedAddCommGroup  F]  [NormedSpace  𝕜  F]

example  :  E  →L[𝕜]  E  :=
  ContinuousLinearMap.id  𝕜  E

example  (f  :  E  →L[𝕜]  F)  :  E  →  F  :=
  f

example  (f  :  E  →L[𝕜]  F)  :  Continuous  f  :=
  f.cont

example  (f  :  E  →L[𝕜]  F)  (x  y  :  E)  :  f  (x  +  y)  =  f  x  +  f  y  :=
  f.map_add  x  y

example  (f  :  E  →L[𝕜]  F)  (a  :  𝕜)  (x  :  E)  :  f  (a  •  x)  =  a  •  f  x  :=
  f.map_smul  a  x 
```

连续线性映射有一个由以下性质表征的算子范数。

```py
variable  (f  :  E  →L[𝕜]  F)

example  (x  :  E)  :  ‖f  x‖  ≤  ‖f‖  *  ‖x‖  :=
  f.le_opNorm  x

example  {M  :  ℝ}  (hMp  :  0  ≤  M)  (hM  :  ∀  x,  ‖f  x‖  ≤  M  *  ‖x‖)  :  ‖f‖  ≤  M  :=
  f.opNorm_le_bound  hMp  hM 
```

还有一个捆绑连续线性*同构*的概念。这种同构的类型是`E ≃L[𝕜] F`。

作为一项挑战性的练习，你可以证明 Banach-Steinhaus 定理，也称为一致有界性原理。该原理表明，从 Banach 空间到范数空间的一族连续线性映射是逐点有界的，那么这些线性映射的范数是一致有界的。主要成分是 Baire 定理`nonempty_interior_of_iUnion_of_closed`。（你在拓扑章节中证明了这一版本。）次要成分包括`continuous_linear_map.opNorm_le_of_shell`、`interior_subset`和`interior_iInter_subset`以及`isClosed_le`。

```py
variable  {𝕜  :  Type*}  [NontriviallyNormedField  𝕜]  {E  :  Type*}  [NormedAddCommGroup  E]
  [NormedSpace  𝕜  E]  {F  :  Type*}  [NormedAddCommGroup  F]  [NormedSpace  𝕜  F]

open  Metric

example  {ι  :  Type*}  [CompleteSpace  E]  {g  :  ι  →  E  →L[𝕜]  F}  (h  :  ∀  x,  ∃  C,  ∀  i,  ‖g  i  x‖  ≤  C)  :
  ∃  C',  ∀  i,  ‖g  i‖  ≤  C'  :=  by
  -- sequence of subsets consisting of those `x : E` with norms `‖g i x‖` bounded by `n`
  let  e  :  ℕ  →  Set  E  :=  fun  n  ↦  ⋂  i  :  ι,  {  x  :  E  |  ‖g  i  x‖  ≤  n  }
  -- each of these sets is closed
  have  hc  :  ∀  n  :  ℕ,  IsClosed  (e  n)
  sorry
  -- the union is the entire space; this is where we use `h`
  have  hU  :  (⋃  n  :  ℕ,  e  n)  =  univ
  sorry
  /- apply the Baire category theorem to conclude that for some `m : ℕ`,
 `e m` contains some `x` -/
  obtain  ⟨m,  x,  hx⟩  :  ∃  m,  ∃  x,  x  ∈  interior  (e  m)  :=  sorry
  obtain  ⟨ε,  ε_pos,  hε⟩  :  ∃  ε  >  0,  ball  x  ε  ⊆  interior  (e  m)  :=  sorry
  obtain  ⟨k,  hk⟩  :  ∃  k  :  𝕜,  1  <  ‖k‖  :=  sorry
  -- show all elements in the ball have norm bounded by `m` after applying any `g i`
  have  real_norm_le  :  ∀  z  ∈  ball  x  ε,  ∀  (i  :  ι),  ‖g  i  z‖  ≤  m
  sorry
  have  εk_pos  :  0  <  ε  /  ‖k‖  :=  sorry
  refine  ⟨(m  +  m  :  ℕ)  /  (ε  /  ‖k‖),  fun  i  ↦  ContinuousLinearMap.opNorm_le_of_shell  ε_pos  ?_  hk  ?_⟩
  sorry
  sorry 
```

### 12.2.3\. 作为渐近比较

定义可微性也需要渐近比较。Mathlib 有一个广泛的库，涵盖了大的 O 和小的小 o 关系，其定义如下所示。打开`asymptotics`区域允许我们使用相应的符号。在这里，我们只使用小 o 来定义可微性。

```py
open  Asymptotics

example  {α  :  Type*}  {E  :  Type*}  [NormedGroup  E]  {F  :  Type*}  [NormedGroup  F]  (c  :  ℝ)
  (l  :  Filter  α)  (f  :  α  →  E)  (g  :  α  →  F)  :  IsBigOWith  c  l  f  g  ↔  ∀ᶠ  x  in  l,  ‖f  x‖  ≤  c  *  ‖g  x‖  :=
  isBigOWith_iff

example  {α  :  Type*}  {E  :  Type*}  [NormedGroup  E]  {F  :  Type*}  [NormedGroup  F]
  (l  :  Filter  α)  (f  :  α  →  E)  (g  :  α  →  F)  :  f  =O[l]  g  ↔  ∃  C,  IsBigOWith  C  l  f  g  :=
  isBigO_iff_isBigOWith

example  {α  :  Type*}  {E  :  Type*}  [NormedGroup  E]  {F  :  Type*}  [NormedGroup  F]
  (l  :  Filter  α)  (f  :  α  →  E)  (g  :  α  →  F)  :  f  =o[l]  g  ↔  ∀  C  >  0,  IsBigOWith  C  l  f  g  :=
  isLittleO_iff_forall_isBigOWith

example  {α  :  Type*}  {E  :  Type*}  [NormedAddCommGroup  E]  (l  :  Filter  α)  (f  g  :  α  →  E)  :
  f  ~[l]  g  ↔  (f  -  g)  =o[l]  g  :=
  Iff.rfl 
```

### 12.2.4\. 可微性

我们现在可以讨论范数空间之间的可微函数。类比于一维的初等函数，Mathlib 定义了一个谓词`HasFDerivAt`和一个函数`fderiv`。在这里，“f”代表*Fréchet*。

```py
open  Topology

variable  {𝕜  :  Type*}  [NontriviallyNormedField  𝕜]  {E  :  Type*}  [NormedAddCommGroup  E]
  [NormedSpace  𝕜  E]  {F  :  Type*}  [NormedAddCommGroup  F]  [NormedSpace  𝕜  F]

example  (f  :  E  →  F)  (f'  :  E  →L[𝕜]  F)  (x₀  :  E)  :
  HasFDerivAt  f  f'  x₀  ↔  (fun  x  ↦  f  x  -  f  x₀  -  f'  (x  -  x₀))  =o[𝓝  x₀]  fun  x  ↦  x  -  x₀  :=
  hasFDerivAtFilter_iff_isLittleO  ..

example  (f  :  E  →  F)  (f'  :  E  →L[𝕜]  F)  (x₀  :  E)  (hff'  :  HasFDerivAt  f  f'  x₀)  :  fderiv  𝕜  f  x₀  =  f'  :=
  hff'.fderiv 
```

我们还有迭代导数，其值在多线性映射类型`E [×n]→L[𝕜] F`中，并且我们有连续微分函数。类型`ℕ∞`是`ℕ`加上一个额外的元素`∞`，它比每一个自然数都要大。所以$\mathcal{C}^\infty$函数是满足`ContDiff 𝕜 ⊤ f`的函数`f`。

```py
example  (n  :  ℕ)  (f  :  E  →  F)  :  E  →  E[×n]→L[𝕜]  F  :=
  iteratedFDeriv  𝕜  n  f

example  (n  :  ℕ∞)  {f  :  E  →  F}  :
  ContDiff  𝕜  n  f  ↔
  (∀  m  :  ℕ,  (m  :  WithTop  ℕ)  ≤  n  →  Continuous  fun  x  ↦  iteratedFDeriv  𝕜  m  f  x)  ∧
  ∀  m  :  ℕ,  (m  :  WithTop  ℕ)  <  n  →  Differentiable  𝕜  fun  x  ↦  iteratedFDeriv  𝕜  m  f  x  :=
  contDiff_iff_continuous_differentiable 
```

`ContDiff`中的可微参数也可以取值`ω : WithTop ℕ∞`来表示解析函数。

有一个更严格的可微性概念称为 `HasStrictFDerivAt`，它在逆函数定理和隐函数定理的陈述中使用，这两个定理都在 Mathlib 中。在 `ℝ` 或 `ℂ` 上，连续可微的函数是严格可微的。

```py
example  {𝕂  :  Type*}  [RCLike  𝕂]  {E  :  Type*}  [NormedAddCommGroup  E]  [NormedSpace  𝕂  E]  {F  :  Type*}
  [NormedAddCommGroup  F]  [NormedSpace  𝕂  F]  {f  :  E  →  F}  {x  :  E}  {n  :  WithTop  ℕ∞}
  (hf  :  ContDiffAt  𝕂  n  f  x)  (hn  :  1  ≤  n)  :  HasStrictFDerivAt  f  (fderiv  𝕂  f  x)  x  :=
  hf.hasStrictFDerivAt  hn 
```

局部逆定理是通过一个操作来陈述的，该操作从一个函数和假设函数在点 `a` 处严格可微以及其导数是一个同构的假设中产生一个逆函数。

下面的第一个例子得到了这个局部逆。下一个例子陈述了它确实是左逆和右逆，并且它是严格可微的。

```py
section  LocalInverse
variable  [CompleteSpace  E]  {f  :  E  →  F}  {f'  :  E  ≃L[𝕜]  F}  {a  :  E}

example  (hf  :  HasStrictFDerivAt  f  (f'  :  E  →L[𝕜]  F)  a)  :  F  →  E  :=
  HasStrictFDerivAt.localInverse  f  f'  a  hf

example  (hf  :  HasStrictFDerivAt  f  (f'  :  E  →L[𝕜]  F)  a)  :
  ∀ᶠ  x  in  𝓝  a,  hf.localInverse  f  f'  a  (f  x)  =  x  :=
  hf.eventually_left_inverse

example  (hf  :  HasStrictFDerivAt  f  (f'  :  E  →L[𝕜]  F)  a)  :
  ∀ᶠ  x  in  𝓝  (f  a),  f  (hf.localInverse  f  f'  a  x)  =  x  :=
  hf.eventually_right_inverse

example  {f  :  E  →  F}  {f'  :  E  ≃L[𝕜]  F}  {a  :  E}
  (hf  :  HasStrictFDerivAt  f  (f'  :  E  →L[𝕜]  F)  a)  :
  HasStrictFDerivAt  (HasStrictFDerivAt.localInverse  f  f'  a  hf)  (f'.symm  :  F  →L[𝕜]  E)  (f  a)  :=
  HasStrictFDerivAt.to_localInverse  hf

end  LocalInverse 
```

这只是对 Mathlib 中的微分学的快速浏览。该库包含了许多我们没有讨论的变体。例如，你可能想在单变量设置中使用单侧导数。这样做的方法在 Mathlib 的更一般背景下可以找到；参见 `HasFDerivWithinAt` 或更一般的 `HasFDerivAtFilter`。  ## 12.1\. 基础微分学

设 `f` 是从实数到实数的函数。在谈论 `f` 在某一点的导数和谈论导数函数之间有一个区别。在 Mathlib 中，第一个概念表示如下。

```py
open  Real

/-- The sin function has derivative 1 at 0\. -/
example  :  HasDerivAt  sin  1  0  :=  by  simpa  using  hasDerivAt_sin  0 
```

我们还可以通过写 `DifferentiableAt ℝ` 来表达 `f` 在某一点可微，而不指定其在该点的导数。我们明确指定 `ℝ`，因为在稍微更一般的情况下，当我们谈论从 `ℂ` 到 `ℂ` 的函数时，我们想要能够区分在实数意义上的可微和在复数导数意义上的可微。

```py
example  (x  :  ℝ)  :  DifferentiableAt  ℝ  sin  x  :=
  (hasDerivAt_sin  x).differentiableAt 
```

每次我们想要引用导数时都必须提供可微性的证明将会很麻烦。因此，Mathlib 提供了一个函数 `deriv f : ℝ → ℝ`，它对任何函数 `f : ℝ → ℝ` 都有定义，但在 `f` 不可微的点处定义为 `0`。

```py
example  {f  :  ℝ  →  ℝ}  {x  a  :  ℝ}  (h  :  HasDerivAt  f  a  x)  :  deriv  f  x  =  a  :=
  h.deriv

example  {f  :  ℝ  →  ℝ}  {x  :  ℝ}  (h  :  ¬DifferentiableAt  ℝ  f  x)  :  deriv  f  x  =  0  :=
  deriv_zero_of_not_differentiableAt  h 
```

当然，关于 `deriv` 的引理中有很多确实需要可微性的假设。例如，你应该在没有可微性假设的情况下考虑下一个引理的反例。

```py
example  {f  g  :  ℝ  →  ℝ}  {x  :  ℝ}  (hf  :  DifferentiableAt  ℝ  f  x)  (hg  :  DifferentiableAt  ℝ  g  x)  :
  deriv  (f  +  g)  x  =  deriv  f  x  +  deriv  g  x  :=
  deriv_add  hf  hg 
```

然而，有趣的是，有一些陈述可以通过利用 `deriv` 在函数不可微时默认为零的事实来避免可微性的假设。因此，理解以下陈述需要知道 `deriv` 的精确定义。

```py
example  {f  :  ℝ  →  ℝ}  {a  :  ℝ}  (h  :  IsLocalMin  f  a)  :  deriv  f  a  =  0  :=
  h.deriv_eq_zero 
```

我们甚至可以在没有任何可微性假设的情况下陈述罗尔定理，这似乎更加奇怪。

```py
open  Set

example  {f  :  ℝ  →  ℝ}  {a  b  :  ℝ}  (hab  :  a  <  b)  (hfc  :  ContinuousOn  f  (Icc  a  b))  (hfI  :  f  a  =  f  b)  :
  ∃  c  ∈  Ioo  a  b,  deriv  f  c  =  0  :=
  exists_deriv_eq_zero  hab  hfc  hfI 
```

当然，这个技巧对一般平均值定理不适用。

```py
example  (f  :  ℝ  →  ℝ)  {a  b  :  ℝ}  (hab  :  a  <  b)  (hf  :  ContinuousOn  f  (Icc  a  b))
  (hf'  :  DifferentiableOn  ℝ  f  (Ioo  a  b))  :  ∃  c  ∈  Ioo  a  b,  deriv  f  c  =  (f  b  -  f  a)  /  (b  -  a)  :=
  exists_deriv_eq_slope  f  hab  hf  hf' 
```

Lean 可以使用 `simp` 策略自动计算一些简单的导数。

```py
example  :  deriv  (fun  x  :  ℝ  ↦  x  ^  5)  6  =  5  *  6  ^  4  :=  by  simp

example  :  deriv  sin  π  =  -1  :=  by  simp 
```

## 12.2\. 范数空间中的微分学

### 12.2.1. 范数空间

使用*范数向量空间*的概念，可以将微分推广到`ℝ`之外，该概念封装了方向和距离。我们首先从*范数群*的概念开始，这是一个加法交换群，它配备了满足以下条件的实值范数函数。

```py
variable  {E  :  Type*}  [NormedAddCommGroup  E]

example  (x  :  E)  :  0  ≤  ‖x‖  :=
  norm_nonneg  x

example  {x  :  E}  :  ‖x‖  =  0  ↔  x  =  0  :=
  norm_eq_zero

example  (x  y  :  E)  :  ‖x  +  y‖  ≤  ‖x‖  +  ‖y‖  :=
  norm_add_le  x  y 
```

每个范数空间都是一个具有距离函数 $d(x, y) = \| x - y \|$ 的度量空间，因此它也是一个拓扑空间。Lean 和 Mathlib 都知道这一点。

```py
example  :  MetricSpace  E  :=  by  infer_instance

example  {X  :  Type*}  [TopologicalSpace  X]  {f  :  X  →  E}  (hf  :  Continuous  f)  :
  Continuous  fun  x  ↦  ‖f  x‖  :=
  hf.norm 
```

为了使用范数的概念与线性代数的概念相结合，我们在`NormedAddGroup E`之上添加了假设`NormedSpace ℝ E`。这规定`E`是`ℝ`上的向量空间，并且标量乘法满足以下条件。

```py
variable  [NormedSpace  ℝ  E]

example  (a  :  ℝ)  (x  :  E)  :  ‖a  •  x‖  =  |a|  *  ‖x‖  :=
  norm_smul  a  x 
```

完全范数空间被称为*Banach 空间*。每个有限维向量空间都是完备的。

```py
example  [FiniteDimensional  ℝ  E]  :  CompleteSpace  E  :=  by  infer_instance 
```

在所有之前的例子中，我们使用了实数作为基域。更一般地，我们可以在任何*非平凡范数域*上的向量空间中进行微积分。这些是配备了实值范数且范数乘法且具有以下性质的域：并非每个元素都有范数为零或一（等价地，存在一个范数大于一的元素）。

```py
example  (𝕜  :  Type*)  [NontriviallyNormedField  𝕜]  (x  y  :  𝕜)  :  ‖x  *  y‖  =  ‖x‖  *  ‖y‖  :=
  norm_mul  x  y

example  (𝕜  :  Type*)  [NontriviallyNormedField  𝕜]  :  ∃  x  :  𝕜,  1  <  ‖x‖  :=
  NormedField.exists_one_lt_norm  𝕜 
```

在非平凡范数域上的有限维向量空间只要该域本身是完备的，就是完备的。

```py
example  (𝕜  :  Type*)  [NontriviallyNormedField  𝕜]  (E  :  Type*)  [NormedAddCommGroup  E]
  [NormedSpace  𝕜  E]  [CompleteSpace  𝕜]  [FiniteDimensional  𝕜  E]  :  CompleteSpace  E  :=
  FiniteDimensional.complete  𝕜  E 
```

### 12.2.2. 连续线性映射

我们现在转向范数空间范畴中的形态，即连续线性映射。在 Mathlib 中，范数空间`E`和`F`之间`𝕜`-线性连续映射的类型被写成`E →L[𝕜] F`。它们被实现为*捆绑映射*，这意味着该类型的元素包含函数本身以及线性性和连续性的属性。Lean 将插入一个强制转换，以便连续线性映射可以被视为函数。

```py
variable  {𝕜  :  Type*}  [NontriviallyNormedField  𝕜]  {E  :  Type*}  [NormedAddCommGroup  E]
  [NormedSpace  𝕜  E]  {F  :  Type*}  [NormedAddCommGroup  F]  [NormedSpace  𝕜  F]

example  :  E  →L[𝕜]  E  :=
  ContinuousLinearMap.id  𝕜  E

example  (f  :  E  →L[𝕜]  F)  :  E  →  F  :=
  f

example  (f  :  E  →L[𝕜]  F)  :  Continuous  f  :=
  f.cont

example  (f  :  E  →L[𝕜]  F)  (x  y  :  E)  :  f  (x  +  y)  =  f  x  +  f  y  :=
  f.map_add  x  y

example  (f  :  E  →L[𝕜]  F)  (a  :  𝕜)  (x  :  E)  :  f  (a  •  x)  =  a  •  f  x  :=
  f.map_smul  a  x 
```

连续线性映射有一个算子范数，其特征如下。

```py
variable  (f  :  E  →L[𝕜]  F)

example  (x  :  E)  :  ‖f  x‖  ≤  ‖f‖  *  ‖x‖  :=
  f.le_opNorm  x

example  {M  :  ℝ}  (hMp  :  0  ≤  M)  (hM  :  ∀  x,  ‖f  x‖  ≤  M  *  ‖x‖)  :  ‖f‖  ≤  M  :=
  f.opNorm_le_bound  hMp  hM 
```

还有一个捆绑连续线性*同构*的概念。这类同构的类型是`E ≃L[𝕜] F`。

作为一项挑战性的练习，你可以证明 Banach-Steinhaus 定理，也称为一致有界性原理。该原理表明，从 Banach 空间到范数空间的连续线性映射族是逐点有界的，那么这些线性映射的范数是一致有界的。主要成分是 Baire 定理`nonempty_interior_of_iUnion_of_closed`。（你在拓扑章节中证明了这一版本。）次要成分包括`continuous_linear_map.opNorm_le_of_shell`、`interior_subset`和`interior_iInter_subset`以及`isClosed_le`。

```py
variable  {𝕜  :  Type*}  [NontriviallyNormedField  𝕜]  {E  :  Type*}  [NormedAddCommGroup  E]
  [NormedSpace  𝕜  E]  {F  :  Type*}  [NormedAddCommGroup  F]  [NormedSpace  𝕜  F]

open  Metric

example  {ι  :  Type*}  [CompleteSpace  E]  {g  :  ι  →  E  →L[𝕜]  F}  (h  :  ∀  x,  ∃  C,  ∀  i,  ‖g  i  x‖  ≤  C)  :
  ∃  C',  ∀  i,  ‖g  i‖  ≤  C'  :=  by
  -- sequence of subsets consisting of those `x : E` with norms `‖g i x‖` bounded by `n`
  let  e  :  ℕ  →  Set  E  :=  fun  n  ↦  ⋂  i  :  ι,  {  x  :  E  |  ‖g  i  x‖  ≤  n  }
  -- each of these sets is closed
  have  hc  :  ∀  n  :  ℕ,  IsClosed  (e  n)
  sorry
  -- the union is the entire space; this is where we use `h`
  have  hU  :  (⋃  n  :  ℕ,  e  n)  =  univ
  sorry
  /- apply the Baire category theorem to conclude that for some `m : ℕ`,
 `e m` contains some `x` -/
  obtain  ⟨m,  x,  hx⟩  :  ∃  m,  ∃  x,  x  ∈  interior  (e  m)  :=  sorry
  obtain  ⟨ε,  ε_pos,  hε⟩  :  ∃  ε  >  0,  ball  x  ε  ⊆  interior  (e  m)  :=  sorry
  obtain  ⟨k,  hk⟩  :  ∃  k  :  𝕜,  1  <  ‖k‖  :=  sorry
  -- show all elements in the ball have norm bounded by `m` after applying any `g i`
  have  real_norm_le  :  ∀  z  ∈  ball  x  ε,  ∀  (i  :  ι),  ‖g  i  z‖  ≤  m
  sorry
  have  εk_pos  :  0  <  ε  /  ‖k‖  :=  sorry
  refine  ⟨(m  +  m  :  ℕ)  /  (ε  /  ‖k‖),  fun  i  ↦  ContinuousLinearMap.opNorm_le_of_shell  ε_pos  ?_  hk  ?_⟩
  sorry
  sorry 
```

### 12.2.3. 渐近比较

定义可微性还需要进行渐近比较。Mathlib 有一个广泛的库，涵盖了大的 O 和小的 o 关系，其定义如下。打开 `asymptotics` 局部允许我们使用相应的符号。在这里，我们只使用小的 o 来定义可微性。

```py
open  Asymptotics

example  {α  :  Type*}  {E  :  Type*}  [NormedGroup  E]  {F  :  Type*}  [NormedGroup  F]  (c  :  ℝ)
  (l  :  Filter  α)  (f  :  α  →  E)  (g  :  α  →  F)  :  IsBigOWith  c  l  f  g  ↔  ∀ᶠ  x  in  l,  ‖f  x‖  ≤  c  *  ‖g  x‖  :=
  isBigOWith_iff

example  {α  :  Type*}  {E  :  Type*}  [NormedGroup  E]  {F  :  Type*}  [NormedGroup  F]
  (l  :  Filter  α)  (f  :  α  →  E)  (g  :  α  →  F)  :  f  =O[l]  g  ↔  ∃  C,  IsBigOWith  C  l  f  g  :=
  isBigO_iff_isBigOWith

example  {α  :  Type*}  {E  :  Type*}  [NormedGroup  E]  {F  :  Type*}  [NormedGroup  F]
  (l  :  Filter  α)  (f  :  α  →  E)  (g  :  α  →  F)  :  f  =o[l]  g  ↔  ∀  C  >  0,  IsBigOWith  C  l  f  g  :=
  isLittleO_iff_forall_isBigOWith

example  {α  :  Type*}  {E  :  Type*}  [NormedAddCommGroup  E]  (l  :  Filter  α)  (f  g  :  α  →  E)  :
  f  ~[l]  g  ↔  (f  -  g)  =o[l]  g  :=
  Iff.rfl 
```

### 12.2.4\. 可微性

现在我们已经准备好讨论范数空间之间的可微函数。类比于一维的初等情况，Mathlib 定义了一个谓词 `HasFDerivAt` 和一个函数 `fderiv`。这里的字母“f”代表 *Fréchet*。

```py
open  Topology

variable  {𝕜  :  Type*}  [NontriviallyNormedField  𝕜]  {E  :  Type*}  [NormedAddCommGroup  E]
  [NormedSpace  𝕜  E]  {F  :  Type*}  [NormedAddCommGroup  F]  [NormedSpace  𝕜  F]

example  (f  :  E  →  F)  (f'  :  E  →L[𝕜]  F)  (x₀  :  E)  :
  HasFDerivAt  f  f'  x₀  ↔  (fun  x  ↦  f  x  -  f  x₀  -  f'  (x  -  x₀))  =o[𝓝  x₀]  fun  x  ↦  x  -  x₀  :=
  hasFDerivAtFilter_iff_isLittleO  ..

example  (f  :  E  →  F)  (f'  :  E  →L[𝕜]  F)  (x₀  :  E)  (hff'  :  HasFDerivAt  f  f'  x₀)  :  fderiv  𝕜  f  x₀  =  f'  :=
  hff'.fderiv 
```

我们还有迭代导数，其值在多线性映射类型 `E [×n]→L[𝕜] F` 中，我们还有连续微分函数。类型 `ℕ∞` 是 `ℕ` 加上一个额外的元素 `∞`，它比每一个自然数都要大。因此 $\mathcal{C}^\infty$ 函数是满足 `ContDiff 𝕜 ⊤ f` 的函数 `f`。

```py
example  (n  :  ℕ)  (f  :  E  →  F)  :  E  →  E[×n]→L[𝕜]  F  :=
  iteratedFDeriv  𝕜  n  f

example  (n  :  ℕ∞)  {f  :  E  →  F}  :
  ContDiff  𝕜  n  f  ↔
  (∀  m  :  ℕ,  (m  :  WithTop  ℕ)  ≤  n  →  Continuous  fun  x  ↦  iteratedFDeriv  𝕜  m  f  x)  ∧
  ∀  m  :  ℕ,  (m  :  WithTop  ℕ)  <  n  →  Differentiable  𝕜  fun  x  ↦  iteratedFDeriv  𝕜  m  f  x  :=
  contDiff_iff_continuous_differentiable 
```

`ContDiff` 中的可微性参数也可以取值 `ω : WithTop ℕ∞` 来表示解析函数。

有一个更严格的可微性概念称为 `HasStrictFDerivAt`，它在逆函数定理和隐函数定理的表述中使用，这两个定理都在 Mathlib 中。在 `ℝ` 或 `ℂ` 上，连续可微的函数是严格可微的。

```py
example  {𝕂  :  Type*}  [RCLike  𝕂]  {E  :  Type*}  [NormedAddCommGroup  E]  [NormedSpace  𝕂  E]  {F  :  Type*}
  [NormedAddCommGroup  F]  [NormedSpace  𝕂  F]  {f  :  E  →  F}  {x  :  E}  {n  :  WithTop  ℕ∞}
  (hf  :  ContDiffAt  𝕂  n  f  x)  (hn  :  1  ≤  n)  :  HasStrictFDerivAt  f  (fderiv  𝕂  f  x)  x  :=
  hf.hasStrictFDerivAt  hn 
```

本地逆定理是通过一个操作来表述的，该操作从一个函数及其在点 `a` 处严格可微的假设以及其导数是一个同构的假设中产生一个逆函数。

下面的第一个例子得到了这个局部逆。下一个例子说明它确实是左逆和右逆，并且它是严格可微的。

```py
section  LocalInverse
variable  [CompleteSpace  E]  {f  :  E  →  F}  {f'  :  E  ≃L[𝕜]  F}  {a  :  E}

example  (hf  :  HasStrictFDerivAt  f  (f'  :  E  →L[𝕜]  F)  a)  :  F  →  E  :=
  HasStrictFDerivAt.localInverse  f  f'  a  hf

example  (hf  :  HasStrictFDerivAt  f  (f'  :  E  →L[𝕜]  F)  a)  :
  ∀ᶠ  x  in  𝓝  a,  hf.localInverse  f  f'  a  (f  x)  =  x  :=
  hf.eventually_left_inverse

example  (hf  :  HasStrictFDerivAt  f  (f'  :  E  →L[𝕜]  F)  a)  :
  ∀ᶠ  x  in  𝓝  (f  a),  f  (hf.localInverse  f  f'  a  x)  =  x  :=
  hf.eventually_right_inverse

example  {f  :  E  →  F}  {f'  :  E  ≃L[𝕜]  F}  {a  :  E}
  (hf  :  HasStrictFDerivAt  f  (f'  :  E  →L[𝕜]  F)  a)  :
  HasStrictFDerivAt  (HasStrictFDerivAt.localInverse  f  f'  a  hf)  (f'.symm  :  F  →L[𝕜]  E)  (f  a)  :=
  HasStrictFDerivAt.to_localInverse  hf

end  LocalInverse 
```

这只是对 Mathlib 中的微分学的快速浏览。该库包含了许多我们没有讨论的变体。例如，你可能在单变量设置中使用单侧导数。这样做的方法在 Mathlib 的更一般背景下可以找到；参见 `HasFDerivWithinAt` 或更一般的 `HasFDerivAtFilter`。

### 12.2.1\. 范数空间

使用 *范数向量空间* 的概念可以将微分推广到 `ℝ` 之外，该概念封装了方向和距离。我们首先从 *范数群* 的概念开始，它是一个加法交换群，并配备了一个满足以下条件的实值范数函数。

```py
variable  {E  :  Type*}  [NormedAddCommGroup  E]

example  (x  :  E)  :  0  ≤  ‖x‖  :=
  norm_nonneg  x

example  {x  :  E}  :  ‖x‖  =  0  ↔  x  =  0  :=
  norm_eq_zero

example  (x  y  :  E)  :  ‖x  +  y‖  ≤  ‖x‖  +  ‖y‖  :=
  norm_add_le  x  y 
```

每个范数空间都是一个带有距离函数 $d(x, y) = \| x - y \|$ 的度量空间，因此它也是一个拓扑空间。Lean 和 Mathlib 都知道这一点。

```py
example  :  MetricSpace  E  :=  by  infer_instance

example  {X  :  Type*}  [TopologicalSpace  X]  {f  :  X  →  E}  (hf  :  Continuous  f)  :
  Continuous  fun  x  ↦  ‖f  x‖  :=
  hf.norm 
```

为了使用来自线性代数的范数概念，我们在 `NormedAddGroup E` 上添加了假设 `NormedSpace ℝ E`。这规定 `E` 是一个实数域上的向量空间，并且标量乘法满足以下条件。

```py
variable  [NormedSpace  ℝ  E]

example  (a  :  ℝ)  (x  :  E)  :  ‖a  •  x‖  =  |a|  *  ‖x‖  :=
  norm_smul  a  x 
```

完全范数空间被称为 *Banach 空间*。每个有限维向量空间都是完备的。

```py
example  [FiniteDimensional  ℝ  E]  :  CompleteSpace  E  :=  by  infer_instance 
```

在所有之前的例子中，我们使用了实数作为基域。更一般地说，我们可以对任何 *非平凡范数域* 上的向量空间进行微积分。这些是配备了实值范数且该范数是乘法的，并且具有不是每个元素都有范数零或一的属性（等价地，存在一个范数大于一的元素）。

```py
example  (𝕜  :  Type*)  [NontriviallyNormedField  𝕜]  (x  y  :  𝕜)  :  ‖x  *  y‖  =  ‖x‖  *  ‖y‖  :=
  norm_mul  x  y

example  (𝕜  :  Type*)  [NontriviallyNormedField  𝕜]  :  ∃  x  :  𝕜,  1  <  ‖x‖  :=
  NormedField.exists_one_lt_norm  𝕜 
```

在一个非平凡范数域上的有限维向量空间只要该域本身是完备的，就是完备的。

```py
example  (𝕜  :  Type*)  [NontriviallyNormedField  𝕜]  (E  :  Type*)  [NormedAddCommGroup  E]
  [NormedSpace  𝕜  E]  [CompleteSpace  𝕜]  [FiniteDimensional  𝕜  E]  :  CompleteSpace  E  :=
  FiniteDimensional.complete  𝕜  E 
```

### 12.2.2\. 连续线性映射

现在，我们转向范数空间范畴中的态射，即连续线性映射。在 Mathlib 中，范数空间 `E` 和 `F` 之间 `𝕜`-线性连续映射的类型被写成 `E →L[𝕜] F`。它们被实现为 *捆绑映射*，这意味着这个类型的元素包含函数本身以及线性性和连续性的属性。Lean 将插入一个强制转换，以便连续线性映射可以被视为一个函数。

```py
variable  {𝕜  :  Type*}  [NontriviallyNormedField  𝕜]  {E  :  Type*}  [NormedAddCommGroup  E]
  [NormedSpace  𝕜  E]  {F  :  Type*}  [NormedAddCommGroup  F]  [NormedSpace  𝕜  F]

example  :  E  →L[𝕜]  E  :=
  ContinuousLinearMap.id  𝕜  E

example  (f  :  E  →L[𝕜]  F)  :  E  →  F  :=
  f

example  (f  :  E  →L[𝕜]  F)  :  Continuous  f  :=
  f.cont

example  (f  :  E  →L[𝕜]  F)  (x  y  :  E)  :  f  (x  +  y)  =  f  x  +  f  y  :=
  f.map_add  x  y

example  (f  :  E  →L[𝕜]  F)  (a  :  𝕜)  (x  :  E)  :  f  (a  •  x)  =  a  •  f  x  :=
  f.map_smul  a  x 
```

连续线性映射有一个由以下性质表征的算子范数。

```py
variable  (f  :  E  →L[𝕜]  F)

example  (x  :  E)  :  ‖f  x‖  ≤  ‖f‖  *  ‖x‖  :=
  f.le_opNorm  x

example  {M  :  ℝ}  (hMp  :  0  ≤  M)  (hM  :  ∀  x,  ‖f  x‖  ≤  M  *  ‖x‖)  :  ‖f‖  ≤  M  :=
  f.opNorm_le_bound  hMp  hM 
```

还有一个捆绑连续线性 *同构* 的概念。这种同构的类型是 `E ≃L[𝕜] F`。

作为一项具有挑战性的练习，你可以证明 Banach-Steinhaus 定理，也称为一致有界性原理。该原理表明，从 Banach 空间到范数空间的连续线性映射族在每一点上是有界的，那么这些线性映射的范数是一致有界的。主要成分是 Baire 定理 `nonempty_interior_of_iUnion_of_closed`。（你在拓扑章节中证明了该定理的一个版本。）次要成分包括 `continuous_linear_map.opNorm_le_of_shell`、`interior_subset` 和 `interior_iInter_subset` 以及 `isClosed_le`。

```py
variable  {𝕜  :  Type*}  [NontriviallyNormedField  𝕜]  {E  :  Type*}  [NormedAddCommGroup  E]
  [NormedSpace  𝕜  E]  {F  :  Type*}  [NormedAddCommGroup  F]  [NormedSpace  𝕜  F]

open  Metric

example  {ι  :  Type*}  [CompleteSpace  E]  {g  :  ι  →  E  →L[𝕜]  F}  (h  :  ∀  x,  ∃  C,  ∀  i,  ‖g  i  x‖  ≤  C)  :
  ∃  C',  ∀  i,  ‖g  i‖  ≤  C'  :=  by
  -- sequence of subsets consisting of those `x : E` with norms `‖g i x‖` bounded by `n`
  let  e  :  ℕ  →  Set  E  :=  fun  n  ↦  ⋂  i  :  ι,  {  x  :  E  |  ‖g  i  x‖  ≤  n  }
  -- each of these sets is closed
  have  hc  :  ∀  n  :  ℕ,  IsClosed  (e  n)
  sorry
  -- the union is the entire space; this is where we use `h`
  have  hU  :  (⋃  n  :  ℕ,  e  n)  =  univ
  sorry
  /- apply the Baire category theorem to conclude that for some `m : ℕ`,
 `e m` contains some `x` -/
  obtain  ⟨m,  x,  hx⟩  :  ∃  m,  ∃  x,  x  ∈  interior  (e  m)  :=  sorry
  obtain  ⟨ε,  ε_pos,  hε⟩  :  ∃  ε  >  0,  ball  x  ε  ⊆  interior  (e  m)  :=  sorry
  obtain  ⟨k,  hk⟩  :  ∃  k  :  𝕜,  1  <  ‖k‖  :=  sorry
  -- show all elements in the ball have norm bounded by `m` after applying any `g i`
  have  real_norm_le  :  ∀  z  ∈  ball  x  ε,  ∀  (i  :  ι),  ‖g  i  z‖  ≤  m
  sorry
  have  εk_pos  :  0  <  ε  /  ‖k‖  :=  sorry
  refine  ⟨(m  +  m  :  ℕ)  /  (ε  /  ‖k‖),  fun  i  ↦  ContinuousLinearMap.opNorm_le_of_shell  ε_pos  ?_  hk  ?_⟩
  sorry
  sorry 
```

### 12.2.3\. 渐近比较

定义可微性也需要进行渐近比较。Mathlib 有一个涵盖大 O 和小 o 关系的广泛库，其定义如下所示。打开 `asymptotics` 局域允许我们使用相应的符号。在这里，我们只使用小 o 来定义可微性。

```py
open  Asymptotics

example  {α  :  Type*}  {E  :  Type*}  [NormedGroup  E]  {F  :  Type*}  [NormedGroup  F]  (c  :  ℝ)
  (l  :  Filter  α)  (f  :  α  →  E)  (g  :  α  →  F)  :  IsBigOWith  c  l  f  g  ↔  ∀ᶠ  x  in  l,  ‖f  x‖  ≤  c  *  ‖g  x‖  :=
  isBigOWith_iff

example  {α  :  Type*}  {E  :  Type*}  [NormedGroup  E]  {F  :  Type*}  [NormedGroup  F]
  (l  :  Filter  α)  (f  :  α  →  E)  (g  :  α  →  F)  :  f  =O[l]  g  ↔  ∃  C,  IsBigOWith  C  l  f  g  :=
  isBigO_iff_isBigOWith

example  {α  :  Type*}  {E  :  Type*}  [NormedGroup  E]  {F  :  Type*}  [NormedGroup  F]
  (l  :  Filter  α)  (f  :  α  →  E)  (g  :  α  →  F)  :  f  =o[l]  g  ↔  ∀  C  >  0,  IsBigOWith  C  l  f  g  :=
  isLittleO_iff_forall_isBigOWith

example  {α  :  Type*}  {E  :  Type*}  [NormedAddCommGroup  E]  (l  :  Filter  α)  (f  g  :  α  →  E)  :
  f  ~[l]  g  ↔  (f  -  g)  =o[l]  g  :=
  Iff.rfl 
```

### 12.2.4\. 可微性

现在，我们准备讨论范数空间之间的可微函数。类比于一维的初等情况，Mathlib 定义了一个谓词 `HasFDerivAt` 和一个函数 `fderiv`。在这里，“f”代表 *Fréchet*。

```py
open  Topology

variable  {𝕜  :  Type*}  [NontriviallyNormedField  𝕜]  {E  :  Type*}  [NormedAddCommGroup  E]
  [NormedSpace  𝕜  E]  {F  :  Type*}  [NormedAddCommGroup  F]  [NormedSpace  𝕜  F]

example  (f  :  E  →  F)  (f'  :  E  →L[𝕜]  F)  (x₀  :  E)  :
  HasFDerivAt  f  f'  x₀  ↔  (fun  x  ↦  f  x  -  f  x₀  -  f'  (x  -  x₀))  =o[𝓝  x₀]  fun  x  ↦  x  -  x₀  :=
  hasFDerivAtFilter_iff_isLittleO  ..

example  (f  :  E  →  F)  (f'  :  E  →L[𝕜]  F)  (x₀  :  E)  (hff'  :  HasFDerivAt  f  f'  x₀)  :  fderiv  𝕜  f  x₀  =  f'  :=
  hff'.fderiv 
```

我们还有迭代导数，其值在多线性映射类型 `E [×n]→L[𝕜] F` 中，并且我们有连续微分函数。类型 `ℕ∞` 是在 `ℕ` 的基础上增加了一个元素 `∞`，这个元素比任何自然数都要大。因此，$\mathcal{C}^\infty$ 函数是满足 `ContDiff 𝕜 ⊤ f` 的函数 `f`。

```py
example  (n  :  ℕ)  (f  :  E  →  F)  :  E  →  E[×n]→L[𝕜]  F  :=
  iteratedFDeriv  𝕜  n  f

example  (n  :  ℕ∞)  {f  :  E  →  F}  :
  ContDiff  𝕜  n  f  ↔
  (∀  m  :  ℕ,  (m  :  WithTop  ℕ)  ≤  n  →  Continuous  fun  x  ↦  iteratedFDeriv  𝕜  m  f  x)  ∧
  ∀  m  :  ℕ,  (m  :  WithTop  ℕ)  <  n  →  Differentiable  𝕜  fun  x  ↦  iteratedFDeriv  𝕜  m  f  x  :=
  contDiff_iff_continuous_differentiable 
```

`ContDiff` 中的可微性参数也可以取值 `ω : WithTop ℕ∞` 来表示解析函数。

存在一个更严格的可微性概念，称为 `HasStrictFDerivAt`，它在逆函数定理和隐函数定理的陈述中使用，这两个定理都在 Mathlib 中。在 `ℝ` 或 `ℂ` 上，连续可微的函数是严格可微的。

```py
example  {𝕂  :  Type*}  [RCLike  𝕂]  {E  :  Type*}  [NormedAddCommGroup  E]  [NormedSpace  𝕂  E]  {F  :  Type*}
  [NormedAddCommGroup  F]  [NormedSpace  𝕂  F]  {f  :  E  →  F}  {x  :  E}  {n  :  WithTop  ℕ∞}
  (hf  :  ContDiffAt  𝕂  n  f  x)  (hn  :  1  ≤  n)  :  HasStrictFDerivAt  f  (fderiv  𝕂  f  x)  x  :=
  hf.hasStrictFDerivAt  hn 
```

局部逆定理是通过一个操作来陈述的，该操作从一个函数和假设函数在点 `a` 处严格可微以及其导数是一个同构来生成逆函数。

下面的第一个例子得到了这个局部逆。下一个例子指出，这确实是一个从左到右的局部逆，并且它是严格可微的。

```py
section  LocalInverse
variable  [CompleteSpace  E]  {f  :  E  →  F}  {f'  :  E  ≃L[𝕜]  F}  {a  :  E}

example  (hf  :  HasStrictFDerivAt  f  (f'  :  E  →L[𝕜]  F)  a)  :  F  →  E  :=
  HasStrictFDerivAt.localInverse  f  f'  a  hf

example  (hf  :  HasStrictFDerivAt  f  (f'  :  E  →L[𝕜]  F)  a)  :
  ∀ᶠ  x  in  𝓝  a,  hf.localInverse  f  f'  a  (f  x)  =  x  :=
  hf.eventually_left_inverse

example  (hf  :  HasStrictFDerivAt  f  (f'  :  E  →L[𝕜]  F)  a)  :
  ∀ᶠ  x  in  𝓝  (f  a),  f  (hf.localInverse  f  f'  a  x)  =  x  :=
  hf.eventually_right_inverse

example  {f  :  E  →  F}  {f'  :  E  ≃L[𝕜]  F}  {a  :  E}
  (hf  :  HasStrictFDerivAt  f  (f'  :  E  →L[𝕜]  F)  a)  :
  HasStrictFDerivAt  (HasStrictFDerivAt.localInverse  f  f'  a  hf)  (f'.symm  :  F  →L[𝕜]  E)  (f  a)  :=
  HasStrictFDerivAt.to_localInverse  hf

end  LocalInverse 
```

这只是对 Mathlib 中的微分学的快速浏览。该库包含了许多我们没有讨论的变体。例如，你可能想在单变量设置中使用单侧导数。这样做的方法在 Mathlib 的更一般背景下可以找到；参见 `HasFDerivWithinAt` 或更通用的 `HasFDerivAtFilter`*。
