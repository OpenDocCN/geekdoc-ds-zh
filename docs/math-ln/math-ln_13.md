# 13. 积分与测度理论

> 原文：[`leanprover-community.github.io/mathematics_in_lean/C13_Integration_and_Measure_Theory.html`](https://leanprover-community.github.io/mathematics_in_lean/C13_Integration_and_Measure_Theory.html)

*Lean 中的数学* **   13. 积分与测度理论

+   查看页面源代码

* * *

## 13.1. 基本积分

我们首先关注在 `ℝ` 的有限区间上函数的积分。我们可以积分基本函数。

```py
open  MeasureTheory  intervalIntegral

open  Interval
-- this introduces the notation `[[a, b]]` for the segment from `min a b` to `max a b`

example  (a  b  :  ℝ)  :  (∫  x  in  a..b,  x)  =  (b  ^  2  -  a  ^  2)  /  2  :=
  integral_id

example  {a  b  :  ℝ}  (h  :  (0  :  ℝ)  ∉  [[a,  b]])  :  (∫  x  in  a..b,  1  /  x)  =  Real.log  (b  /  a)  :=
  integral_one_div  h 
```

微积分的基本定理将积分和微分联系起来。以下给出该定理两部分的简化陈述。第一部分说明积分提供了对微分的逆运算，第二部分指定了如何计算导数的积分。（这两部分非常密切相关，但它们的最佳版本，此处未展示，并不等价。）

```py
example  (f  :  ℝ  →  ℝ)  (hf  :  Continuous  f)  (a  b  :  ℝ)  :  deriv  (fun  u  ↦  ∫  x  :  ℝ  in  a..u,  f  x)  b  =  f  b  :=
  (integral_hasStrictDerivAt_right  (hf.intervalIntegrable  _  _)  (hf.stronglyMeasurableAtFilter  _  _)
  hf.continuousAt).hasDerivAt.deriv

example  {f  :  ℝ  →  ℝ}  {a  b  :  ℝ}  {f'  :  ℝ  →  ℝ}  (h  :  ∀  x  ∈  [[a,  b]],  HasDerivAt  f  (f'  x)  x)
  (h'  :  IntervalIntegrable  f'  volume  a  b)  :  (∫  y  in  a..b,  f'  y)  =  f  b  -  f  a  :=
  integral_eq_sub_of_hasDerivAt  h  h' 
```

卷积也在 Mathlib 中定义，并且其基本性质得到了证明。

```py
open  Convolution

example  (f  :  ℝ  →  ℝ)  (g  :  ℝ  →  ℝ)  :  f  ⋆  g  =  fun  x  ↦  ∫  t,  f  t  *  g  (x  -  t)  :=
  rfl 
```  ## 13.2. 测度理论

Mathlib 中积分的一般背景是测度理论。甚至上一节中的基本积分实际上也是博赫纳积分。博赫纳积分是勒贝格积分的推广，其中目标空间可以是任何 Banach 空间，不一定是有限维的。

测度理论发展的第一个组成部分是集合的 $\sigma$-代数概念，这些集合被称为 *可测集*。类型类 `MeasurableSpace` 用于为一个类型提供这种结构。集合 `empty` 和 `univ` 是可测的，可测集合的补集是可测的，可测集合的可数并或交集是可测的。请注意，这些公理是多余的；如果你 `#print MeasurableSpace`，你会看到 Mathlib 使用的那些。如下面的例子所示，可数性假设可以使用 `Encodable` 类型类来表示。

```py
variable  {α  :  Type*}  [MeasurableSpace  α]

example  :  MeasurableSet  (∅  :  Set  α)  :=
  MeasurableSet.empty

example  :  MeasurableSet  (univ  :  Set  α)  :=
  MeasurableSet.univ

example  {s  :  Set  α}  (hs  :  MeasurableSet  s)  :  MeasurableSet  (sᶜ)  :=
  hs.compl

example  :  Encodable  ℕ  :=  by  infer_instance

example  (n  :  ℕ)  :  Encodable  (Fin  n)  :=  by  infer_instance

variable  {ι  :  Type*}  [Encodable  ι]

example  {f  :  ι  →  Set  α}  (h  :  ∀  b,  MeasurableSet  (f  b))  :  MeasurableSet  (⋃  b,  f  b)  :=
  MeasurableSet.iUnion  h

example  {f  :  ι  →  Set  α}  (h  :  ∀  b,  MeasurableSet  (f  b))  :  MeasurableSet  (⋂  b,  f  b)  :=
  MeasurableSet.iInter  h 
```

一旦一个类型是可测的，我们就可以对其进行测量。在纸上，一个集合（或类型）上的测度是测度集到扩展非负实数的函数，在可数可分并集中是可加的。在 Mathlib 中，我们不想每次将测度应用于集合时都携带可测性假设。因此，我们将测度扩展到任何集合 `s`，作为包含 `s` 的可测集合测度的下确界。当然，许多引理仍然需要可测性假设，但并非所有。

```py
open  MeasureTheory  Function
variable  {μ  :  Measure  α}

example  (s  :  Set  α)  :  μ  s  =  ⨅  (t  :  Set  α)  (_  :  s  ⊆  t)  (_  :  MeasurableSet  t),  μ  t  :=
  measure_eq_iInf  s

example  (s  :  ι  →  Set  α)  :  μ  (⋃  i,  s  i)  ≤  ∑'  i,  μ  (s  i)  :=
  measure_iUnion_le  s

example  {f  :  ℕ  →  Set  α}  (hmeas  :  ∀  i,  MeasurableSet  (f  i))  (hdis  :  Pairwise  (Disjoint  on  f))  :
  μ  (⋃  i,  f  i)  =  ∑'  i,  μ  (f  i)  :=
  μ.m_iUnion  hmeas  hdis 
```

一旦一个类型与一个测度相关联，我们就说一个性质 `P` 在几乎处处成立，如果该性质失败元素的集合的测度为 0。几乎处处成立的性质集合形成一个过滤器，但 Mathlib 引入了特殊的符号来说明一个性质在几乎处处成立。

```py
example  {P  :  α  →  Prop}  :  (∀ᵐ  x  ∂μ,  P  x)  ↔  ∀ᶠ  x  in  ae  μ,  P  x  :=
  Iff.rfl 
```  ## 13.3. 积分

现在我们有了可测空间和测度，我们可以考虑积分。如上所述，Mathlib 使用一个非常通用的积分概念，允许任何 Banach 空间作为目标。通常，我们不希望我们的符号携带任何假设，因此我们定义积分的方式是，如果所讨论的函数不可积，则积分等于零。与积分有关的大多数引理都有可积性假设。

```py
section
variable  {E  :  Type*}  [NormedAddCommGroup  E]  [NormedSpace  ℝ  E]  [CompleteSpace  E]  {f  :  α  →  E}

example  {f  g  :  α  →  E}  (hf  :  Integrable  f  μ)  (hg  :  Integrable  g  μ)  :
  ∫  a,  f  a  +  g  a  ∂μ  =  ∫  a,  f  a  ∂μ  +  ∫  a,  g  a  ∂μ  :=
  integral_add  hf  hg 
```

作为我们各种约定之间复杂交互的一个例子，让我们看看如何积分常数函数。回想一下，测度 `μ` 在 `ℝ≥0∞` 上取值，这是扩展非负实数的类型。存在一个函数 `ENNReal.toReal : ℝ≥0∞ → ℝ`，它将无穷大点 `⊤` 映射到零。对于任何 `s : Set α`，如果 `μ s = ⊤`，则非零常数函数在 `s` 上不可积。在这种情况下，它们的积分根据定义等于零，正如 `(μ s).toReal` 一样。因此，在所有情况下，我们都有以下引理。

```py
example  {s  :  Set  α}  (c  :  E)  :  ∫  x  in  s,  c  ∂μ  =  (μ  s).toReal  •  c  :=
  setIntegral_const  c 
```

现在我们简要说明如何访问积分理论中最重要的定理，从支配收敛定理开始。Mathlib 中有几个版本，这里我们只展示最基本的一个。

```py
open  Filter

example  {F  :  ℕ  →  α  →  E}  {f  :  α  →  E}  (bound  :  α  →  ℝ)  (hmeas  :  ∀  n,  AEStronglyMeasurable  (F  n)  μ)
  (hint  :  Integrable  bound  μ)  (hbound  :  ∀  n,  ∀ᵐ  a  ∂μ,  ‖F  n  a‖  ≤  bound  a)
  (hlim  :  ∀ᵐ  a  ∂μ,  Tendsto  (fun  n  :  ℕ  ↦  F  n  a)  atTop  (𝓝  (f  a)))  :
  Tendsto  (fun  n  ↦  ∫  a,  F  n  a  ∂μ)  atTop  (𝓝  (∫  a,  f  a  ∂μ))  :=
  tendsto_integral_of_dominated_convergence  bound  hmeas  hint  hbound  hlim 
```

然后，我们有了关于乘积类型积分的 Fubini 定理。

```py
example  {α  :  Type*}  [MeasurableSpace  α]  {μ  :  Measure  α}  [SigmaFinite  μ]  {β  :  Type*}
  [MeasurableSpace  β]  {ν  :  Measure  β}  [SigmaFinite  ν]  (f  :  α  ×  β  →  E)
  (hf  :  Integrable  f  (μ.prod  ν))  :  ∫  z,  f  z  ∂  μ.prod  ν  =  ∫  x,  ∫  y,  f  (x,  y)  ∂ν  ∂μ  :=
  integral_prod  f  hf 
```

存在一个非常通用的卷积版本，适用于任何连续的双线性形式。

```py
open  Convolution

variable  {𝕜  :  Type*}  {G  :  Type*}  {E  :  Type*}  {E'  :  Type*}  {F  :  Type*}  [NormedAddCommGroup  E]
  [NormedAddCommGroup  E']  [NormedAddCommGroup  F]  [NontriviallyNormedField  𝕜]  [NormedSpace  𝕜  E]
  [NormedSpace  𝕜  E']  [NormedSpace  𝕜  F]  [MeasurableSpace  G]  [NormedSpace  ℝ  F]  [CompleteSpace  F]
  [Sub  G]

example  (f  :  G  →  E)  (g  :  G  →  E')  (L  :  E  →L[𝕜]  E'  →L[𝕜]  F)  (μ  :  Measure  G)  :
  f  ⋆[L,  μ]  g  =  fun  x  ↦  ∫  t,  L  (f  t)  (g  (x  -  t))  ∂μ  :=
  rfl 
```

最后，Mathlib 有一个非常通用的变量替换公式的版本。在下面的陈述中，`BorelSpace E` 表示 `E` 上的 $\sigma$-代数是由 `E` 的开集生成的，而 `IsAddHaarMeasure μ` 表示测度 `μ` 是左不变的，对紧集给出有限质量，并对开集给出正质量。

```py
example  {E  :  Type*}  [NormedAddCommGroup  E]  [NormedSpace  ℝ  E]  [FiniteDimensional  ℝ  E]
  [MeasurableSpace  E]  [BorelSpace  E]  (μ  :  Measure  E)  [μ.IsAddHaarMeasure]  {F  :  Type*}
  [NormedAddCommGroup  F]  [NormedSpace  ℝ  F]  [CompleteSpace  F]  {s  :  Set  E}  {f  :  E  →  E}
  {f'  :  E  →  E  →L[ℝ]  E}  (hs  :  MeasurableSet  s)
  (hf  :  ∀  x  :  E,  x  ∈  s  →  HasFDerivWithinAt  f  (f'  x)  s  x)  (h_inj  :  InjOn  f  s)  (g  :  E  →  F)  :
  ∫  x  in  f  ''  s,  g  x  ∂μ  =  ∫  x  in  s,  |(f'  x).det|  •  g  (f  x)  ∂μ  :=
  integral_image_eq_integral_abs_det_fderiv_smul  μ  hs  hf  h_inj  g 
``` 上一节 下一节

* * *

© 版权所有 2020-2025，Jeremy Avigad，Patrick Massot。文本许可协议为 CC BY 4.0。

使用 [Sphinx](https://www.sphinx-doc.org/) 构建，使用 [主题](https://github.com/readthedocs/sphinx_rtd_theme) 由 [Read the Docs](https://readthedocs.org) 提供。## 13.1. 初等积分

我们首先关注在 `ℝ` 上的有限区间上函数的积分。我们可以积分初等函数。

```py
open  MeasureTheory  intervalIntegral

open  Interval
-- this introduces the notation `[[a, b]]` for the segment from `min a b` to `max a b`

example  (a  b  :  ℝ)  :  (∫  x  in  a..b,  x)  =  (b  ^  2  -  a  ^  2)  /  2  :=
  integral_id

example  {a  b  :  ℝ}  (h  :  (0  :  ℝ)  ∉  [[a,  b]])  :  (∫  x  in  a..b,  1  /  x)  =  Real.log  (b  /  a)  :=
  integral_one_div  h 
```

微积分基本定理将积分和微分联系起来。下面我们给出该定理两部分的简化陈述。第一部分说明积分提供了对微分的逆运算，第二部分指定了如何计算导数的积分。（这两部分非常紧密相关，但它们的最佳版本（此处未展示）并不等价。）

```py
example  (f  :  ℝ  →  ℝ)  (hf  :  Continuous  f)  (a  b  :  ℝ)  :  deriv  (fun  u  ↦  ∫  x  :  ℝ  in  a..u,  f  x)  b  =  f  b  :=
  (integral_hasStrictDerivAt_right  (hf.intervalIntegrable  _  _)  (hf.stronglyMeasurableAtFilter  _  _)
  hf.continuousAt).hasDerivAt.deriv

example  {f  :  ℝ  →  ℝ}  {a  b  :  ℝ}  {f'  :  ℝ  →  ℝ}  (h  :  ∀  x  ∈  [[a,  b]],  HasDerivAt  f  (f'  x)  x)
  (h'  :  IntervalIntegrable  f'  volume  a  b)  :  (∫  y  in  a..b,  f'  y)  =  f  b  -  f  a  :=
  integral_eq_sub_of_hasDerivAt  h  h' 
```

卷积也在 Mathlib 中定义，并且其基本性质得到了证明。

```py
open  Convolution

example  (f  :  ℝ  →  ℝ)  (g  :  ℝ  →  ℝ)  :  f  ⋆  g  =  fun  x  ↦  ∫  t,  f  t  *  g  (x  -  t)  :=
  rfl 
```

Mathlib 中积分的一般背景是测度理论。甚至上一节中的基本积分实际上也是 Bochner 积分。Bochner 积分是 Lebesgue 积分的推广，其中目标空间可以是任何 Banach 空间，不一定是有限维的。

测度理论发展中的第一个组成部分是集合的 $\sigma$-代数概念，这些集合被称为**可测**集合。类型类 `MeasurableSpace` 用于为一个类型提供这种结构。集合 `empty` 和 `univ` 是可测的，可测集合的补集也是可测的，可测集合的可数并或交也是可测的。请注意，这些公理是多余的；如果你 `#print MeasurableSpace`，你会看到 Mathlib 使用的那些。如下面的例子所示，可数性假设可以使用 `Encodable` 类型类来表示。

```py
variable  {α  :  Type*}  [MeasurableSpace  α]

example  :  MeasurableSet  (∅  :  Set  α)  :=
  MeasurableSet.empty

example  :  MeasurableSet  (univ  :  Set  α)  :=
  MeasurableSet.univ

example  {s  :  Set  α}  (hs  :  MeasurableSet  s)  :  MeasurableSet  (sᶜ)  :=
  hs.compl

example  :  Encodable  ℕ  :=  by  infer_instance

example  (n  :  ℕ)  :  Encodable  (Fin  n)  :=  by  infer_instance

variable  {ι  :  Type*}  [Encodable  ι]

example  {f  :  ι  →  Set  α}  (h  :  ∀  b,  MeasurableSet  (f  b))  :  MeasurableSet  (⋃  b,  f  b)  :=
  MeasurableSet.iUnion  h

example  {f  :  ι  →  Set  α}  (h  :  ∀  b,  MeasurableSet  (f  b))  :  MeasurableSet  (⋂  b,  f  b)  :=
  MeasurableSet.iInter  h 
```

一旦一个类型是可测的，我们就可以对其进行测量。在纸上，一个在带有 $\sigma$-代数的集合（或类型）上的测度是一个从可测集合到扩展非负实数的函数，它在可数不相交的并集上是可加的。在 Mathlib 中，我们不希望在每次将测度应用于集合时都携带可测性假设。因此，我们将测度扩展到任何集合 `s`，作为包含 `s` 的可测集合测度的下确界。当然，许多引理仍然需要可测性假设，但并非所有。

```py
open  MeasureTheory  Function
variable  {μ  :  Measure  α}

example  (s  :  Set  α)  :  μ  s  =  ⨅  (t  :  Set  α)  (_  :  s  ⊆  t)  (_  :  MeasurableSet  t),  μ  t  :=
  measure_eq_iInf  s

example  (s  :  ι  →  Set  α)  :  μ  (⋃  i,  s  i)  ≤  ∑'  i,  μ  (s  i)  :=
  measure_iUnion_le  s

example  {f  :  ℕ  →  Set  α}  (hmeas  :  ∀  i,  MeasurableSet  (f  i))  (hdis  :  Pairwise  (Disjoint  on  f))  :
  μ  (⋃  i,  f  i)  =  ∑'  i,  μ  (f  i)  :=
  μ.m_iUnion  hmeas  hdis 
```

一旦一个类型与其相关联的测度，我们说一个属性 `P` 在**几乎处处**成立，如果属性失败的元素集合的测度为 0。几乎处处成立的属性集合形成一个过滤器，但 Mathlib 引入了特殊的符号来说明一个属性几乎处处成立。

```py
example  {P  :  α  →  Prop}  :  (∀ᵐ  x  ∂μ,  P  x)  ↔  ∀ᶠ  x  in  ae  μ,  P  x  :=
  Iff.rfl 
```  ## 13.3\. 积分

现在我们有了可测空间和测度，我们可以考虑积分。如上所述，Mathlib 使用一个非常一般的积分概念，允许任何 Banach 空间作为目标。像往常一样，我们不希望我们的符号携带假设，所以我们定义积分的方式是，如果相关函数不可积，则积分等于零。与积分有关的大多数引理都有可积性假设。

```py
section
variable  {E  :  Type*}  [NormedAddCommGroup  E]  [NormedSpace  ℝ  E]  [CompleteSpace  E]  {f  :  α  →  E}

example  {f  g  :  α  →  E}  (hf  :  Integrable  f  μ)  (hg  :  Integrable  g  μ)  :
  ∫  a,  f  a  +  g  a  ∂μ  =  ∫  a,  f  a  ∂μ  +  ∫  a,  g  a  ∂μ  :=
  integral_add  hf  hg 
```

作为我们各种约定之间复杂交互的一个例子，让我们看看如何积分常数函数。回想一下，测度 `μ` 在 `ℝ≥0∞` 类型中取值，即扩展非负实数类型。存在一个函数 `ENNReal.toReal : ℝ≥0∞ → ℝ`，它将无穷大点 `⊤` 映射到零。对于任何 `s : Set α`，如果 `μ s = ⊤`，则非零常数函数在 `s` 上不可积。在这种情况下，它们的积分根据定义等于零，就像 `(μ s).toReal` 一样。所以，在所有情况下，我们都有以下引理。

```py
example  {s  :  Set  α}  (c  :  E)  :  ∫  x  in  s,  c  ∂μ  =  (μ  s).toReal  •  c  :=
  setIntegral_const  c 
```

我们现在简要解释如何访问积分理论中最重要的定理，从支配收敛定理开始。Mathlib 中有几个版本，这里我们只展示最基本的一个。

```py
open  Filter

example  {F  :  ℕ  →  α  →  E}  {f  :  α  →  E}  (bound  :  α  →  ℝ)  (hmeas  :  ∀  n,  AEStronglyMeasurable  (F  n)  μ)
  (hint  :  Integrable  bound  μ)  (hbound  :  ∀  n,  ∀ᵐ  a  ∂μ,  ‖F  n  a‖  ≤  bound  a)
  (hlim  :  ∀ᵐ  a  ∂μ,  Tendsto  (fun  n  :  ℕ  ↦  F  n  a)  atTop  (𝓝  (f  a)))  :
  Tendsto  (fun  n  ↦  ∫  a,  F  n  a  ∂μ)  atTop  (𝓝  (∫  a,  f  a  ∂μ))  :=
  tendsto_integral_of_dominated_convergence  bound  hmeas  hint  hbound  hlim 
```

然后，我们有了乘积类型上的积分的傅里叶定理。

```py
example  {α  :  Type*}  [MeasurableSpace  α]  {μ  :  Measure  α}  [SigmaFinite  μ]  {β  :  Type*}
  [MeasurableSpace  β]  {ν  :  Measure  β}  [SigmaFinite  ν]  (f  :  α  ×  β  →  E)
  (hf  :  Integrable  f  (μ.prod  ν))  :  ∫  z,  f  z  ∂  μ.prod  ν  =  ∫  x,  ∫  y,  f  (x,  y)  ∂ν  ∂μ  :=
  integral_prod  f  hf 
```

存在一个非常通用的卷积版本，适用于任何连续的双线性形式。

```py
open  Convolution

variable  {𝕜  :  Type*}  {G  :  Type*}  {E  :  Type*}  {E'  :  Type*}  {F  :  Type*}  [NormedAddCommGroup  E]
  [NormedAddCommGroup  E']  [NormedAddCommGroup  F]  [NontriviallyNormedField  𝕜]  [NormedSpace  𝕜  E]
  [NormedSpace  𝕜  E']  [NormedSpace  𝕜  F]  [MeasurableSpace  G]  [NormedSpace  ℝ  F]  [CompleteSpace  F]
  [Sub  G]

example  (f  :  G  →  E)  (g  :  G  →  E')  (L  :  E  →L[𝕜]  E'  →L[𝕜]  F)  (μ  :  Measure  G)  :
  f  ⋆[L,  μ]  g  =  fun  x  ↦  ∫  t,  L  (f  t)  (g  (x  -  t))  ∂μ  :=
  rfl 
```

最后，Mathlib 有一个非常通用的变量替换公式。在下面的陈述中，`BorelSpace E` 表示 `E` 上的 $\sigma$-代数是由 `E` 的开集生成的，而 `IsAddHaarMeasure μ` 表示测度 `μ` 是左不变的，对紧集赋予有限质量，对开集赋予正质量。

```py
example  {E  :  Type*}  [NormedAddCommGroup  E]  [NormedSpace  ℝ  E]  [FiniteDimensional  ℝ  E]
  [MeasurableSpace  E]  [BorelSpace  E]  (μ  :  Measure  E)  [μ.IsAddHaarMeasure]  {F  :  Type*}
  [NormedAddCommGroup  F]  [NormedSpace  ℝ  F]  [CompleteSpace  F]  {s  :  Set  E}  {f  :  E  →  E}
  {f'  :  E  →  E  →L[ℝ]  E}  (hs  :  MeasurableSet  s)
  (hf  :  ∀  x  :  E,  x  ∈  s  →  HasFDerivWithinAt  f  (f'  x)  s  x)  (h_inj  :  InjOn  f  s)  (g  :  E  →  F)  :
  ∫  x  in  f  ''  s,  g  x  ∂μ  =  ∫  x  in  s,  |(f'  x).det|  •  g  (f  x)  ∂μ  :=
  integral_image_eq_integral_abs_det_fderiv_smul  μ  hs  hf  h_inj  g 
```  ## 13.1\. 基本积分

我们首先关注在有限区间 $\mathbb{R}$ 上的函数积分。我们可以积分基本函数。

```py
open  MeasureTheory  intervalIntegral

open  Interval
-- this introduces the notation `[[a, b]]` for the segment from `min a b` to `max a b`

example  (a  b  :  ℝ)  :  (∫  x  in  a..b,  x)  =  (b  ^  2  -  a  ^  2)  /  2  :=
  integral_id

example  {a  b  :  ℝ}  (h  :  (0  :  ℝ)  ∉  [[a,  b]])  :  (∫  x  in  a..b,  1  /  x)  =  Real.log  (b  /  a)  :=
  integral_one_div  h 
```

微积分的基本定理将积分和微分联系起来。以下我们给出该定理两部分的简化陈述。第一部分说明积分是微分的逆运算，第二部分则指定了如何计算导数的积分。（这两部分非常紧密相关，但它们的最佳版本（此处未展示）并不等价。）

```py
example  (f  :  ℝ  →  ℝ)  (hf  :  Continuous  f)  (a  b  :  ℝ)  :  deriv  (fun  u  ↦  ∫  x  :  ℝ  in  a..u,  f  x)  b  =  f  b  :=
  (integral_hasStrictDerivAt_right  (hf.intervalIntegrable  _  _)  (hf.stronglyMeasurableAtFilter  _  _)
  hf.continuousAt).hasDerivAt.deriv

example  {f  :  ℝ  →  ℝ}  {a  b  :  ℝ}  {f'  :  ℝ  →  ℝ}  (h  :  ∀  x  ∈  [[a,  b]],  HasDerivAt  f  (f'  x)  x)
  (h'  :  IntervalIntegrable  f'  volume  a  b)  :  (∫  y  in  a..b,  f'  y)  =  f  b  -  f  a  :=
  integral_eq_sub_of_hasDerivAt  h  h' 
```

Mathlib 中也定义了卷积，并证明了其基本性质。

```py
open  Convolution

example  (f  :  ℝ  →  ℝ)  (g  :  ℝ  →  ℝ)  :  f  ⋆  g  =  fun  x  ↦  ∫  t,  f  t  *  g  (x  -  t)  :=
  rfl 
```

## 13.2\. 测度论

在 Mathlib 中，积分的一般背景是测度论。甚至上一节中的基本积分实际上也是博赫纳积分。博赫纳积分是勒贝格积分的推广，其中目标空间可以是任何 Banach 空间，不一定是有限维的。

测度论发展中的第一个组成部分是集合的 $\sigma$-代数概念，这些集合被称为 *可测* 集合。类型类 `MeasurableSpace` 用于为一个类型提供这种结构。空集和全集是可测的，可测集合的补集是可测的，可测集合的可数并集或交集也是可测的。请注意，这些公理是多余的；如果你 `#print MeasurableSpace`，你会看到 Mathlib 使用的那些。如下面的例子所示，可数性假设可以使用 `Encodable` 类型类来表示。

```py
variable  {α  :  Type*}  [MeasurableSpace  α]

example  :  MeasurableSet  (∅  :  Set  α)  :=
  MeasurableSet.empty

example  :  MeasurableSet  (univ  :  Set  α)  :=
  MeasurableSet.univ

example  {s  :  Set  α}  (hs  :  MeasurableSet  s)  :  MeasurableSet  (sᶜ)  :=
  hs.compl

example  :  Encodable  ℕ  :=  by  infer_instance

example  (n  :  ℕ)  :  Encodable  (Fin  n)  :=  by  infer_instance

variable  {ι  :  Type*}  [Encodable  ι]

example  {f  :  ι  →  Set  α}  (h  :  ∀  b,  MeasurableSet  (f  b))  :  MeasurableSet  (⋃  b,  f  b)  :=
  MeasurableSet.iUnion  h

example  {f  :  ι  →  Set  α}  (h  :  ∀  b,  MeasurableSet  (f  b))  :  MeasurableSet  (⋂  b,  f  b)  :=
  MeasurableSet.iInter  h 
```

一旦一个类型是可测的，我们就可以对其进行测量。在纸上，一个带有 $\sigma$-代数的集合（或类型）上的测度是从可测集合到扩展非负实数的函数，在可数可分并集中是可加的。在 Mathlib 中，我们不希望在每次将测度应用于集合时都携带可测性假设。因此，我们将测度扩展到任何集合 `s`，作为包含 `s` 的可测集合测度的下确界。当然，许多引理仍然需要可测性假设，但并非所有。

```py
open  MeasureTheory  Function
variable  {μ  :  Measure  α}

example  (s  :  Set  α)  :  μ  s  =  ⨅  (t  :  Set  α)  (_  :  s  ⊆  t)  (_  :  MeasurableSet  t),  μ  t  :=
  measure_eq_iInf  s

example  (s  :  ι  →  Set  α)  :  μ  (⋃  i,  s  i)  ≤  ∑'  i,  μ  (s  i)  :=
  measure_iUnion_le  s

example  {f  :  ℕ  →  Set  α}  (hmeas  :  ∀  i,  MeasurableSet  (f  i))  (hdis  :  Pairwise  (Disjoint  on  f))  :
  μ  (⋃  i,  f  i)  =  ∑'  i,  μ  (f  i)  :=
  μ.m_iUnion  hmeas  hdis 
```

一旦一个类型与一个测度相关联，我们就说一个性质 `P` 在几乎处处成立，如果该性质失败元素的集合的测度为 0。几乎处处成立的性质集合形成一个过滤器，但 Mathlib 引入了特殊的符号来说明一个性质在几乎处处成立。

```py
example  {P  :  α  →  Prop}  :  (∀ᵐ  x  ∂μ,  P  x)  ↔  ∀ᶠ  x  in  ae  μ,  P  x  :=
  Iff.rfl 
```

## 13.3\. 积分

现在我们有了可测空间和测度，我们可以考虑积分。如上所述，Mathlib 使用一个非常通用的积分概念，允许任何 Banach 空间作为目标。通常，我们不希望我们的符号携带任何假设，所以我们定义积分的方式是，如果所讨论的函数不可积，则积分等于零。与积分有关的大多数引理都有可积性的假设。

```py
section
variable  {E  :  Type*}  [NormedAddCommGroup  E]  [NormedSpace  ℝ  E]  [CompleteSpace  E]  {f  :  α  →  E}

example  {f  g  :  α  →  E}  (hf  :  Integrable  f  μ)  (hg  :  Integrable  g  μ)  :
  ∫  a,  f  a  +  g  a  ∂μ  =  ∫  a,  f  a  ∂μ  +  ∫  a,  g  a  ∂μ  :=
  integral_add  hf  hg 
```

作为我们各种约定之间复杂交互的一个例子，让我们看看如何积分常数函数。回忆一下，测度 `μ` 取值在 `ℝ≥0∞`，即扩展非负实数的类型。有一个函数 `ENNReal.toReal : ℝ≥0∞ → ℝ`，它将无穷大的点 `⊤` 映射到零。对于任何 `s : Set α`，如果 `μ s = ⊤`，则非零常数函数在 `s` 上不可积。在这种情况下，它们的积分根据定义等于零，就像 `(μ s).toReal` 一样。所以，在所有情况下，我们都有以下引理。

```py
example  {s  :  Set  α}  (c  :  E)  :  ∫  x  in  s,  c  ∂μ  =  (μ  s).toReal  •  c  :=
  setIntegral_const  c 
```

我们现在快速解释如何访问积分理论中最重要的一些定理，从支配收敛定理开始。Mathlib 中有几个版本，这里我们只展示最基本的一个。

```py
open  Filter

example  {F  :  ℕ  →  α  →  E}  {f  :  α  →  E}  (bound  :  α  →  ℝ)  (hmeas  :  ∀  n,  AEStronglyMeasurable  (F  n)  μ)
  (hint  :  Integrable  bound  μ)  (hbound  :  ∀  n,  ∀ᵐ  a  ∂μ,  ‖F  n  a‖  ≤  bound  a)
  (hlim  :  ∀ᵐ  a  ∂μ,  Tendsto  (fun  n  :  ℕ  ↦  F  n  a)  atTop  (𝓝  (f  a)))  :
  Tendsto  (fun  n  ↦  ∫  a,  F  n  a  ∂μ)  atTop  (𝓝  (∫  a,  f  a  ∂μ))  :=
  tendsto_integral_of_dominated_convergence  bound  hmeas  hint  hbound  hlim 
```

然后我们有乘积类型上的积分的 Fubini 定理。

```py
example  {α  :  Type*}  [MeasurableSpace  α]  {μ  :  Measure  α}  [SigmaFinite  μ]  {β  :  Type*}
  [MeasurableSpace  β]  {ν  :  Measure  β}  [SigmaFinite  ν]  (f  :  α  ×  β  →  E)
  (hf  :  Integrable  f  (μ.prod  ν))  :  ∫  z,  f  z  ∂  μ.prod  ν  =  ∫  x,  ∫  y,  f  (x,  y)  ∂ν  ∂μ  :=
  integral_prod  f  hf 
```

存在一个非常通用的卷积版本，适用于任何连续的双线性形式。

```py
open  Convolution

variable  {𝕜  :  Type*}  {G  :  Type*}  {E  :  Type*}  {E'  :  Type*}  {F  :  Type*}  [NormedAddCommGroup  E]
  [NormedAddCommGroup  E']  [NormedAddCommGroup  F]  [NontriviallyNormedField  𝕜]  [NormedSpace  𝕜  E]
  [NormedSpace  𝕜  E']  [NormedSpace  𝕜  F]  [MeasurableSpace  G]  [NormedSpace  ℝ  F]  [CompleteSpace  F]
  [Sub  G]

example  (f  :  G  →  E)  (g  :  G  →  E')  (L  :  E  →L[𝕜]  E'  →L[𝕜]  F)  (μ  :  Measure  G)  :
  f  ⋆[L,  μ]  g  =  fun  x  ↦  ∫  t,  L  (f  t)  (g  (x  -  t))  ∂μ  :=
  rfl 
```

最后，Mathlib 有一个非常通用的变量替换公式的版本。在下面的陈述中，`BorelSpace E` 表示 `E` 上的 $\sigma$-代数是由 `E` 的开集生成的，而 `IsAddHaarMeasure μ` 表示测度 `μ` 是左不变的，对紧集给出有限质量，并对开集给出正质量。

```py
example  {E  :  Type*}  [NormedAddCommGroup  E]  [NormedSpace  ℝ  E]  [FiniteDimensional  ℝ  E]
  [MeasurableSpace  E]  [BorelSpace  E]  (μ  :  Measure  E)  [μ.IsAddHaarMeasure]  {F  :  Type*}
  [NormedAddCommGroup  F]  [NormedSpace  ℝ  F]  [CompleteSpace  F]  {s  :  Set  E}  {f  :  E  →  E}
  {f'  :  E  →  E  →L[ℝ]  E}  (hs  :  MeasurableSet  s)
  (hf  :  ∀  x  :  E,  x  ∈  s  →  HasFDerivWithinAt  f  (f'  x)  s  x)  (h_inj  :  InjOn  f  s)  (g  :  E  →  F)  :
  ∫  x  in  f  ''  s,  g  x  ∂μ  =  ∫  x  in  s,  |(f'  x).det|  •  g  (f  x)  ∂μ  :=
  integral_image_eq_integral_abs_det_fderiv_smul  μ  hs  hf  h_inj  g 
```*
