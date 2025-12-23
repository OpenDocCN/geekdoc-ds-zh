# 5\. 初等数论

> 原文：[`leanprover-community.github.io/mathematics_in_lean/C05_Elementary_Number_Theory.html`](https://leanprover-community.github.io/mathematics_in_lean/C05_Elementary_Number_Theory.html)

*Mathematics in Lean* **   5\. 初等数论

+   查看页面源代码

* * *

在本章中，我们将向您展示如何形式化一些初等数论的结果。随着我们处理更实质性的数学内容，证明将变得更长、更复杂，建立在您已经掌握的技能之上。

## 5.1\. 无理根

让我们从古希腊人已知的一个事实开始，即 2 的平方根是无理数。如果我们假设相反的情况，我们可以将 $\sqrt{2} = a / b$ 写成最简分数。平方两边得到 $a² = 2 b²$，这意味着 $a$ 是偶数。如果我们写成 $a = 2c$，那么我们得到 $4c² = 2 b²$，从而 $b² = 2 c²$。这表明 $b$ 也是偶数，这与我们假设的 $a / b$ 已经被化简到最简形式的事实相矛盾。

说 $a / b$ 是最简分数，意味着 $a$ 和 $b$ 没有任何公因数，也就是说，它们是 *互质*。Mathlib 定义谓词 `Nat.Coprime m n` 为 `Nat.gcd m n = 1`。使用 Lean 的匿名投影符号，如果 `s` 和 `t` 是类型为 `Nat` 的表达式，我们可以写 `s.Coprime t` 而不是 `Nat.Coprime s t`，对于 `Nat.gcd` 也是如此。通常，当需要时，Lean 会自动展开 `Nat.Coprime` 的定义，但我们可以通过重写或使用标识符 `Nat.Coprime` 来手动进行。`norm_num` 策略足够智能，可以计算具体值。

```py
#print  Nat.Coprime

example  (m  n  :  Nat)  (h  :  m.Coprime  n)  :  m.gcd  n  =  1  :=
  h

example  (m  n  :  Nat)  (h  :  m.Coprime  n)  :  m.gcd  n  =  1  :=  by
  rw  [Nat.Coprime]  at  h
  exact  h

example  :  Nat.Coprime  12  7  :=  by  norm_num

example  :  Nat.gcd  12  8  =  4  :=  by  norm_num 
```

我们已经在 第 2.4 节 遇到了 `gcd` 函数。对于整数也有 `gcd` 的版本；我们将在下面讨论不同数系之间的关系。甚至还有一个通用的 `gcd` 函数和通用的 `Prime` 和 `Coprime` 概念，它们在一般的代数结构中是有意义的。我们将在下一章了解 Lean 如何管理这种通用性。同时，在本节中，我们将关注自然数。

我们还需要理解素数的概念，`Nat.Prime`。定理 `Nat.prime_def_lt` 提供了一种熟悉的描述，而 `Nat.Prime.eq_one_or_self_of_dvd` 提供了另一种描述。

```py
#check  Nat.prime_def_lt

example  (p  :  ℕ)  (prime_p  :  Nat.Prime  p)  :  2  ≤  p  ∧  ∀  m  :  ℕ,  m  <  p  →  m  ∣  p  →  m  =  1  :=  by
  rwa  [Nat.prime_def_lt]  at  prime_p

#check  Nat.Prime.eq_one_or_self_of_dvd

example  (p  :  ℕ)  (prime_p  :  Nat.Prime  p)  :  ∀  m  :  ℕ,  m  ∣  p  →  m  =  1  ∨  m  =  p  :=
  prime_p.eq_one_or_self_of_dvd

example  :  Nat.Prime  17  :=  by  norm_num

-- commonly used
example  :  Nat.Prime  2  :=
  Nat.prime_two

example  :  Nat.Prime  3  :=
  Nat.prime_three 
```

在自然数中，一个素数具有这样的性质，即它不能写成非平凡因子的乘积。在更广泛的数学背景下，具有这种性质的环的元素被称为 *不可约的*。如果一个元素在除以一个乘积时总是除以其中一个因子，那么这个元素被称为 *素数*。自然数的一个重要性质是，在那个设置中，这两个概念是一致的，从而产生了定理 `Nat.Prime.dvd_mul`。

我们可以使用这个事实来在上述论证中建立一条关键性质：如果一个数的平方是偶数，那么那个数也是偶数。Mathlib 在 `Algebra.Group.Even` 中定义了谓词 `Even`，但以下面的原因将变得清晰，我们将简单地使用 `2 ∣ m` 来表达 `m` 是偶数。

```py
#check  Nat.Prime.dvd_mul
#check  Nat.Prime.dvd_mul  Nat.prime_two
#check  Nat.prime_two.dvd_mul

theorem  even_of_even_sqr  {m  :  ℕ}  (h  :  2  ∣  m  ^  2)  :  2  ∣  m  :=  by
  rw  [pow_two,  Nat.prime_two.dvd_mul]  at  h
  cases  h  <;>  assumption

example  {m  :  ℕ}  (h  :  2  ∣  m  ^  2)  :  2  ∣  m  :=
  Nat.Prime.dvd_of_dvd_pow  Nat.prime_two  h 
```

随着我们继续前进，你需要熟练地找到所需的 facts。记住，如果你能猜出名称的前缀并且已经导入了相关库，你可以使用 tab 完成功能（有时需要按 `ctrl-tab`）来找到你想要的内容。你可以通过在任何标识符上按 `ctrl-click` 来跳转到定义它的文件，这使你能够浏览附近的定义和定理。你还可以使用 [Lean 社区网页](https://leanprover-community.github.io/)上的搜索引擎，如果所有其他方法都失败了，不要犹豫，在 [Zulip](https://leanprover.zulipchat.com/) 上提问。

```py
example  (a  b  c  :  Nat)  (h  :  a  *  b  =  a  *  c)  (h'  :  a  ≠  0)  :  b  =  c  :=
  -- apply? suggests the following:
  (mul_right_inj'  h').mp  h 
```

我们证明平方根无理性的核心包含在以下定理中。看看你是否能完成证明草图，使用 `even_of_even_sqr` 和定理 `Nat.dvd_gcd`。

```py
example  {m  n  :  ℕ}  (coprime_mn  :  m.Coprime  n)  :  m  ^  2  ≠  2  *  n  ^  2  :=  by
  intro  sqr_eq
  have  :  2  ∣  m  :=  by
  sorry
  obtain  ⟨k,  meq⟩  :=  dvd_iff_exists_eq_mul_left.mp  this
  have  :  2  *  (2  *  k  ^  2)  =  2  *  n  ^  2  :=  by
  rw  [←  sqr_eq,  meq]
  ring
  have  :  2  *  k  ^  2  =  n  ^  2  :=
  sorry
  have  :  2  ∣  n  :=  by
  sorry
  have  :  2  ∣  m.gcd  n  :=  by
  sorry
  have  :  2  ∣  1  :=  by
  sorry
  norm_num  at  this 
```

实际上，经过非常少的修改，我们可以将 `2` 替换为任意素数。在下一个例子中尝试一下。在证明的末尾，你需要从 `p ∣ 1` 推导出一个矛盾。你可以使用 `Nat.Prime.two_le`，它表明任何素数都大于或等于 `2`，以及 `Nat.le_of_dvd`。

```py
example  {m  n  p  :  ℕ}  (coprime_mn  :  m.Coprime  n)  (prime_p  :  p.Prime)  :  m  ^  2  ≠  p  *  n  ^  2  :=  by
  sorry 
```

让我们考虑另一种方法。以下是一个快速证明：如果 $p$ 是素数，那么 $m² \ne p n²$：如果我们假设 $m² = p n²$ 并考虑 $m$ 和 $n$ 的素数分解，那么 $p$ 在等式的左边出现偶数次，在右边出现奇数次，这是一个矛盾。请注意，这个论证要求 $n$ 以及因此 $m$ 不为零。下面的形式化确认了这个假设是充分的。

唯一分解定理表明，除了零以外的任何自然数都可以以唯一的方式写成素数的乘积。Mathlib 包含了这个定理的正式版本，它用函数 `Nat.primeFactorsList` 来表达，该函数返回一个数在非递减顺序中的素数因子列表。该库证明了 `Nat.primeFactorsList n` 的所有元素都是素数，任何大于零的 `n` 都等于其因子的乘积，并且如果 `n` 等于另一个素数列表的乘积，那么那个列表是 `Nat.primeFactorsList n` 的一个排列。

```py
#check  Nat.primeFactorsList
#check  Nat.prime_of_mem_primeFactorsList
#check  Nat.prod_primeFactorsList
#check  Nat.primeFactorsList_unique 
```

你可以浏览这些定理以及附近的其他定理，尽管我们还没有讨论列表成员、乘积或排列。对于手头的任务，我们不需要这些。相反，我们将使用 Mathlib 有一个函数 `Nat.factorization` 的事实，它表示与函数相同的数据。具体来说，`Nat.factorization n p`，我们也可以写成 `n.factorization p`，返回 `p` 在 `n` 的素数分解中的次数。我们将使用以下三个事实。

```py
theorem  factorization_mul'  {m  n  :  ℕ}  (mnez  :  m  ≠  0)  (nnez  :  n  ≠  0)  (p  :  ℕ)  :
  (m  *  n).factorization  p  =  m.factorization  p  +  n.factorization  p  :=  by
  rw  [Nat.factorization_mul  mnez  nnez]
  rfl

theorem  factorization_pow'  (n  k  p  :  ℕ)  :
  (n  ^  k).factorization  p  =  k  *  n.factorization  p  :=  by
  rw  [Nat.factorization_pow]
  rfl

theorem  Nat.Prime.factorization'  {p  :  ℕ}  (prime_p  :  p.Prime)  :
  p.factorization  p  =  1  :=  by
  rw  [prime_p.factorization]
  simp 
```

实际上，`n.factorization` 在 Lean 中被定义为有限支持函数，这解释了你将在上述证明中看到的奇怪符号。现在不用担心这个。就我们的目的而言，我们可以将上述三个定理作为一个黑盒使用。

下一个例子表明简化器足够智能，可以将 `n² ≠ 0` 替换为 `n ≠ 0`。`simpa` 策略只是调用 `simp` 后跟 `assumption`。

检查你是否可以使用上面的恒等式来填补证明中的空白部分。

```py
example  {m  n  p  :  ℕ}  (nnz  :  n  ≠  0)  (prime_p  :  p.Prime)  :  m  ^  2  ≠  p  *  n  ^  2  :=  by
  intro  sqr_eq
  have  nsqr_nez  :  n  ^  2  ≠  0  :=  by  simpa
  have  eq1  :  Nat.factorization  (m  ^  2)  p  =  2  *  m.factorization  p  :=  by
  sorry
  have  eq2  :  (p  *  n  ^  2).factorization  p  =  2  *  n.factorization  p  +  1  :=  by
  sorry
  have  :  2  *  m.factorization  p  %  2  =  (2  *  n.factorization  p  +  1)  %  2  :=  by
  rw  [←  eq1,  sqr_eq,  eq2]
  rw  [add_comm,  Nat.add_mul_mod_self_left,  Nat.mul_mod_right]  at  this
  norm_num  at  this 
```

这个证明的一个好处是它也具有普遍性。`2` 没有什么特别之处；经过一些小的修改，这个证明表明，每当我们将 `m^k = r * n^k` 写出来时，任何素数 `p` 在 `r` 中的次数必须是 `k` 的倍数。

要使用 `Nat.count_factors_mul_of_pos` 与 `r * n^k`，我们需要知道 `r` 是正数。但当 `r` 为零时，下面的定理是显而易见的，并且可以通过简化器轻易证明。因此，证明是分情况进行的。行 `rcases r with _ | r` 将目标替换为两个版本：一个是将 `r` 替换为 `0` 的版本，另一个是将 `r` 替换为 `r + 1` 的版本。在第二种情况下，我们可以使用定理 `r.succ_ne_zero`，它建立了 `r + 1 ≠ 0`（`succ` 表示后继）。

注意，以 `have : npow_nz` 开头的行提供了一个简短的证明项证明 `n^k ≠ 0`。要理解它是如何工作的，试着用策略证明来替换它，然后思考策略是如何描述证明项的。

检查你是否可以填补下面证明中的空白部分。在最后，你可以使用 `Nat.dvd_sub'` 和 `Nat.dvd_mul_right` 来完成它。

注意，这个例子并没有假设 `p` 是素数，但当 `p` 不是素数时，结论是显而易见的，因为根据定义，`r.factorization p` 是零，而且证明在所有情况下都适用。

```py
example  {m  n  k  r  :  ℕ}  (nnz  :  n  ≠  0)  (pow_eq  :  m  ^  k  =  r  *  n  ^  k)  {p  :  ℕ}  :
  k  ∣  r.factorization  p  :=  by
  rcases  r  with  _  |  r
  ·  simp
  have  npow_nz  :  n  ^  k  ≠  0  :=  fun  npowz  ↦  nnz  (pow_eq_zero  npowz)
  have  eq1  :  (m  ^  k).factorization  p  =  k  *  m.factorization  p  :=  by
  sorry
  have  eq2  :  ((r  +  1)  *  n  ^  k).factorization  p  =
  k  *  n.factorization  p  +  (r  +  1).factorization  p  :=  by
  sorry
  have  :  r.succ.factorization  p  =  k  *  m.factorization  p  -  k  *  n.factorization  p  :=  by
  rw  [←  eq1,  pow_eq,  eq2,  add_comm,  Nat.add_sub_cancel]
  rw  [this]
  sorry 
```

我们可能希望以多种方式改进这些结果。首先，一个证明说两个平方根是无理数的证明应该对两个平方根有所说明，这可以理解为实数或复数中的一个元素。并且说它是无理数应该对有理数有所说明，即没有有理数等于它。此外，我们还应该将本节中的定理扩展到整数。虽然从数学上明显，如果我们能够将两个平方根写成两个整数的商，那么我们也可以将其写成两个自然数的商，但正式证明这一点需要一些努力。

在 Mathlib 中，自然数、整数、有理数、实数和复数分别由不同的数据类型表示。将注意力限制在单独的域中通常是有帮助的：我们将看到对自然数进行归纳是很容易的，而当实数不是图景的一部分时，对整数的可除性进行推理是最容易的。但是，在不同域之间进行调解是一个头疼的问题，我们将不得不应对。我们将在本章的后面回到这个问题。

我们也应该期待能够加强最后一个定理的结论，使其表明数字 `r` 是一个 `k`-次幂，因为它的 `k`-次根正是 `r` 中每个质因子的乘积，每个质因子的幂次是 `r` 中该质因子的幂次除以 `k`。为了能够做到这一点，我们需要更好的推理手段来处理有限集合上的乘积和求和，这也是我们将再次探讨的主题。

事实上，本节中的结果在 Mathlib 中以更大的普遍性得到确立，在 `Data.Real.Irrational` 中。`multiplicity` 的概念被定义为任意交换幺半群，并且它取值在扩展自然数 `enat` 中，它将无穷大值添加到自然数中。在下一章中，我们将开始发展欣赏 Lean 支持这种普遍性的手段。  

自然数集 $\mathbb{N} = \{ 0, 1, 2, \ldots \}$ 不仅在其自身的基础上具有根本的重要性，而且在构建新的数学对象中也起着核心作用。Lean 的基础允许我们声明 *归纳类型*，这些类型是通过给定的构造函数列表归纳生成的。在 Lean 中，自然数被声明如下。

```py
inductive  Nat  where
  |  zero  :  Nat
  |  succ  (n  :  Nat)  :  Nat 
```

你可以在图书馆中通过输入 `#check Nat` 并在标识符 `Nat` 上使用 `ctrl-click` 来找到这个。该命令指定 `Nat` 是由两个构造函数 `zero : Nat` 和 `succ : Nat → Nat` 自由和归纳生成的数据类型。当然，库引入了 `ℕ` 和 `0` 的符号来分别表示 `nat` 和 `zero`。 (数字被转换为二进制表示，但我们现在不必担心这个细节。)

对于工作的数学家来说，“自由”意味着类型`Nat`有一个元素`zero`和一个注入的后续函数`succ`，其像不包括`zero`。

```py
example  (n  :  Nat)  :  n.succ  ≠  Nat.zero  :=
  Nat.succ_ne_zero  n

example  (m  n  :  Nat)  (h  :  m.succ  =  n.succ)  :  m  =  n  :=
  Nat.succ.inj  h 
```

对于工作的数学家来说，“归纳”这个词意味着自然数伴随着归纳证明的原则和递归定义的原则。本节将向您展示如何使用这些原则。

下面是阶乘函数的递归定义示例。

```py
def  fac  :  ℕ  →  ℕ
  |  0  =>  1
  |  n  +  1  =>  (n  +  1)  *  fac  n 
```

语法需要一些时间来适应。请注意，第一行没有`:=`。接下来的两行提供了递归定义的基础情况和归纳步骤。这些等式在定义上是成立的，但也可以通过将名称`fac`赋予`simp`或`rw`来手动使用。

```py
example  :  fac  0  =  1  :=
  rfl

example  :  fac  0  =  1  :=  by
  rw  [fac]

example  :  fac  0  =  1  :=  by
  simp  [fac]

example  (n  :  ℕ)  :  fac  (n  +  1)  =  (n  +  1)  *  fac  n  :=
  rfl

example  (n  :  ℕ)  :  fac  (n  +  1)  =  (n  +  1)  *  fac  n  :=  by
  rw  [fac]

example  (n  :  ℕ)  :  fac  (n  +  1)  =  (n  +  1)  *  fac  n  :=  by
  simp  [fac] 
```

阶乘函数实际上已经在 Mathlib 中定义为`Nat.factorial`。您可以通过输入`#check Nat.factorial`并使用`ctrl-click`来跳转到它。为了说明目的，我们将在示例中继续使用`fac`。定义`Nat.factorial`之前的注释`@[simp]`指定了定义方程应该被添加到简化器默认使用的恒等式数据库中。

归纳原理表明，我们可以通过证明该命题对 0 成立，并且每当它对自然数$n$成立时，它也对$n + 1$成立，来证明关于自然数的一般命题。因此，证明中`induction' n with n ih`这一行产生了两个目标：在第一个目标中，我们需要证明`0 < fac 0`，而在第二个目标中，我们有额外的假设`ih : 0 < fac n`和需要证明的`0 < fac (n + 1)`。短语`with n ih`用于命名归纳假设的变量和假设，您可以为它们选择任何名称。

```py
theorem  fac_pos  (n  :  ℕ)  :  0  <  fac  n  :=  by
  induction'  n  with  n  ih
  ·  rw  [fac]
  exact  zero_lt_one
  rw  [fac]
  exact  mul_pos  n.succ_pos  ih 
```

`归纳`策略足够智能，可以将依赖于归纳变量的假设作为归纳假设的一部分。通过逐步分析下一个示例，您可以看到发生了什么。

```py
theorem  dvd_fac  {i  n  :  ℕ}  (ipos  :  0  <  i)  (ile  :  i  ≤  n)  :  i  ∣  fac  n  :=  by
  induction'  n  with  n  ih
  ·  exact  absurd  ipos  (not_lt_of_ge  ile)
  rw  [fac]
  rcases  Nat.of_le_succ  ile  with  h  |  h
  ·  apply  dvd_mul_of_dvd_right  (ih  h)
  rw  [h]
  apply  dvd_mul_right 
```

下面的示例提供了一个阶乘函数的粗糙下界。实际上，从分情况证明开始更容易，这样证明的其余部分就从`n = 1`的情况开始。看看您是否能通过使用`pow_succ`或`pow_succ'`进行归纳证明来完成论证。

```py
theorem  pow_two_le_fac  (n  :  ℕ)  :  2  ^  (n  -  1)  ≤  fac  n  :=  by
  rcases  n  with  _  |  n
  ·  simp  [fac]
  sorry 
```

归纳通常用于证明涉及有限和积的恒等式。Mathlib 定义了表达式`Finset.sum s f`，其中`s : Finset α`是类型`α`的元素有限集，`f`是在`α`上定义的函数。`f`的陪域可以是支持交换、结合加法运算和零元素的任何类型。如果您导入`Algebra.BigOperators.Ring`并发出`open BigOperators`命令，您可以使用更具说明性的符号`∑ x ∈ s, f x`。当然，对于有限积也有类似的操作和符号。

我们将在下一节中讨论`Finset`类型及其支持的操作，并在稍后的章节中再次讨论。现在，我们只将使用`Finset.range n`，这是小于`n`的自然数的有限集合。

```py
variable  {α  :  Type*}  (s  :  Finset  ℕ)  (f  :  ℕ  →  ℕ)  (n  :  ℕ)

#check  Finset.sum  s  f
#check  Finset.prod  s  f

open  BigOperators
open  Finset

example  :  s.sum  f  =  ∑  x  ∈  s,  f  x  :=
  rfl

example  :  s.prod  f  =  ∏  x  ∈  s,  f  x  :=
  rfl

example  :  (range  n).sum  f  =  ∑  x  ∈  range  n,  f  x  :=
  rfl

example  :  (range  n).prod  f  =  ∏  x  ∈  range  n,  f  x  :=
  rfl 
```

事实`Finset.sum_range_zero`和`Finset.sum_range_succ`提供了求和到$n$的递归描述，对于乘积也是如此。

```py
example  (f  :  ℕ  →  ℕ)  :  ∑  x  ∈  range  0,  f  x  =  0  :=
  Finset.sum_range_zero  f

example  (f  :  ℕ  →  ℕ)  (n  :  ℕ)  :  ∑  x  ∈  range  n.succ,  f  x  =  ∑  x  ∈  range  n,  f  x  +  f  n  :=
  Finset.sum_range_succ  f  n

example  (f  :  ℕ  →  ℕ)  :  ∏  x  ∈  range  0,  f  x  =  1  :=
  Finset.prod_range_zero  f

example  (f  :  ℕ  →  ℕ)  (n  :  ℕ)  :  ∏  x  ∈  range  n.succ,  f  x  =  (∏  x  ∈  range  n,  f  x)  *  f  n  :=
  Finset.prod_range_succ  f  n 
```

每对中的第一个恒等式是定义性的，也就是说，你可以用`rfl`替换证明。

以下表示我们定义的阶乘函数作为乘积。

```py
example  (n  :  ℕ)  :  fac  n  =  ∏  i  ∈  range  n,  (i  +  1)  :=  by
  induction'  n  with  n  ih
  ·  simp  [fac,  prod_range_zero]
  simp  [fac,  ih,  prod_range_succ,  mul_comm] 
```

我们将`mul_comm`作为简化规则包括在内，这一点值得注意。使用恒等式`x * y = y * x`进行简化似乎很危险，因为这通常会无限循环。Lean 的简化器足够聪明，能够识别这一点，并且只在结果项在某些固定但任意的项的排序中具有较小值的情况下应用该规则。以下示例表明，使用三个规则`mul_assoc`、`mul_comm`和`mul_left_comm`可以成功地识别出括号位置和变量排序相同的产品。

```py
example  (a  b  c  d  e  f  :  ℕ)  :  a  *  (b  *  c  *  f  *  (d  *  e))  =  d  *  (a  *  f  *  e)  *  (c  *  b)  :=  by
  simp  [mul_assoc,  mul_comm,  mul_left_comm] 
```

大概来说，这些规则通过将括号推向右边，然后重新排列两侧的表达式，直到它们都遵循相同的规范顺序来工作。使用这些规则以及相应的加法规则进行简化是一个实用的技巧。

返回到求和恒等式，我们建议逐步通过以下证明，即自然数从 1 加到$n$（包括$n$）的和是$n (n + 1) / 2$。证明的第一步消除了分母。在形式化恒等式时，这通常是有用的，因为除法运算通常有附带条件。（同样，在可能的情况下避免在自然数上使用减法也是有用的。）

```py
theorem  sum_id  (n  :  ℕ)  :  ∑  i  ∈  range  (n  +  1),  i  =  n  *  (n  +  1)  /  2  :=  by
  symm;  apply  Nat.div_eq_of_eq_mul_right  (by  norm_num  :  0  <  2)
  induction'  n  with  n  ih
  ·  simp
  rw  [Finset.sum_range_succ,  mul_add  2,  ←  ih]
  ring 
```

我们鼓励你证明平方和的类似恒等式，以及你在网上可以找到的其他恒等式。

```py
theorem  sum_sqr  (n  :  ℕ)  :  ∑  i  ∈  range  (n  +  1),  i  ^  2  =  n  *  (n  +  1)  *  (2  *  n  +  1)  /  6  :=  by
  sorry 
```

在 Lean 的核心库中，加法和乘法本身是通过递归定义的，它们的基本性质是通过归纳建立的。如果你喜欢思考像那样的基础主题，你可能会喜欢通过乘法和加法的交换律和结合律以及乘法对加法的分配律的证明。你可以在以下概述的自然数副本上这样做。注意，我们可以使用`induction`策略与`MyNat`；Lean 足够聪明，知道要使用相关的归纳原理（当然，这与`Nat`的相同）。

我们从加法的交换律开始。一个很好的经验法则是，由于加法和乘法是通过第二个参数的递归定义的，因此通常通过在该位置出现的变量进行归纳证明是有利的。在证明结合律时，决定使用哪个变量有点棘手。

没有使用零、一、加法和乘法的常规符号来写东西可能会让人感到困惑。我们将在后面学习如何定义这样的符号。在命名空间 `MyNat` 中工作意味着我们可以直接写 `zero` 和 `succ`，而不是 `MyNat.zero` 和 `MyNat.succ`，并且这些名称的解释优先于其他解释。在命名空间外部，下面定义的 `add` 的全名是 `MyNat.add`。

如果你发现你真的喜欢这类东西，可以尝试定义截断减法和指数运算，并证明它们的一些性质。记住，截断减法在零处截断。为了定义这一点，定义一个前驱函数 `pred`，它从任何非零数中减去一，并固定零。函数 `pred` 可以通过一个简单的递归实例来定义。

```py
inductive  MyNat  where
  |  zero  :  MyNat
  |  succ  :  MyNat  →  MyNat

namespace  MyNat

def  add  :  MyNat  →  MyNat  →  MyNat
  |  x,  zero  =>  x
  |  x,  succ  y  =>  succ  (add  x  y)

def  mul  :  MyNat  →  MyNat  →  MyNat
  |  x,  zero  =>  zero
  |  x,  succ  y  =>  add  (mul  x  y)  x

theorem  zero_add  (n  :  MyNat)  :  add  zero  n  =  n  :=  by
  induction'  n  with  n  ih
  ·  rfl
  rw  [add,  ih]

theorem  succ_add  (m  n  :  MyNat)  :  add  (succ  m)  n  =  succ  (add  m  n)  :=  by
  induction'  n  with  n  ih
  ·  rfl
  rw  [add,  ih]
  rfl

theorem  add_comm  (m  n  :  MyNat)  :  add  m  n  =  add  n  m  :=  by
  induction'  n  with  n  ih
  ·  rw  [zero_add]
  rfl
  rw  [add,  succ_add,  ih]

theorem  add_assoc  (m  n  k  :  MyNat)  :  add  (add  m  n)  k  =  add  m  (add  n  k)  :=  by
  sorry
theorem  mul_add  (m  n  k  :  MyNat)  :  mul  m  (add  n  k)  =  add  (mul  m  n)  (mul  m  k)  :=  by
  sorry
theorem  zero_mul  (n  :  MyNat)  :  mul  zero  n  =  zero  :=  by
  sorry
theorem  succ_mul  (m  n  :  MyNat)  :  mul  (succ  m)  n  =  add  (mul  m  n)  n  :=  by
  sorry
theorem  mul_comm  (m  n  :  MyNat)  :  mul  m  n  =  mul  n  m  :=  by
  sorry
end  MyNat 
```  ## 5.3\. 无限多个素数

让我们继续用另一个数学标准来探索归纳和递归：证明存在无限多个素数。一种表述方式是，对于每个自然数 $n$，都存在一个大于 $n$ 的素数。为了证明这一点，设 $p$ 为 $n! + 1$ 的任何素数因子。如果 $p$ 小于或等于 $n$，它就整除 $n!$。由于它也整除 $n! + 1$，它就整除 1，这是矛盾的。因此 $p$ 大于 $n$。

为了形式化这个证明，我们需要证明任何大于或等于 2 的数都有一个素数因子。为了做到这一点，我们需要证明任何不等于 0 或 1 的自然数都大于或等于 2。这把我们带到了形式化的一个奇特特性：通常像这样的简单陈述往往是最令人烦恼的形式化。在这里，我们考虑了几种实现方法。

首先，我们可以使用 `cases` 策略和后继函数尊重自然数上的顺序这一事实。

```py
theorem  two_le  {m  :  ℕ}  (h0  :  m  ≠  0)  (h1  :  m  ≠  1)  :  2  ≤  m  :=  by
  cases  m;  contradiction
  case  succ  m  =>
  cases  m;  contradiction
  repeat  apply  Nat.succ_le_succ
  apply  zero_le 
```

另一种策略是使用 `interval_cases` 策略，该策略在变量包含在自然数或整数的区间内时自动将目标分解为情况。记住，你可以悬停在它上面以查看其文档。

```py
example  {m  :  ℕ}  (h0  :  m  ≠  0)  (h1  :  m  ≠  1)  :  2  ≤  m  :=  by
  by_contra  h
  push_neg  at  h
  interval_cases  m  <;>  contradiction 
```

回想一下，`interval_cases m` 后面的分号意味着下一个策略将应用于它生成的每个情况。另一个选择是使用 `decide` 策略，它试图找到一个决策过程来解决问题。Lean 知道你可以通过决定每个有限实例来决定以有界量词 `∀ x, x < n → ...` 或 `∃ x, x < n ∧ ...` 开头的语句的真值。

```py
example  {m  :  ℕ}  (h0  :  m  ≠  0)  (h1  :  m  ≠  1)  :  2  ≤  m  :=  by
  by_contra  h
  push_neg  at  h
  revert  h0  h1
  revert  h  m
  decide 
```

在手头有定理 `two_le` 的情况下，让我们首先证明每个大于两的自然数都有一个素数因子。Mathlib 包含一个函数 `Nat.minFac`，它返回最小的素数因子，但为了学习库的新部分，我们将避免使用它，并直接证明这个定理。

在这里，普通的归纳法是不够的。我们想要使用 **强归纳法**，这允许我们通过表明对于每一个自然数 $n$，如果 $P$ 对所有小于 $n$ 的值成立，那么它对 $n$ 也成立，来证明每一个自然数 $n$ 都具有属性 $P$。在 Lean 中，这个原则被称为 `Nat.strong_induction_on`，我们可以使用 `using` 关键字告诉归纳策略使用它。注意，当我们这样做时，没有基本情况；它被归纳步骤所包含。

论证如下。假设 $n ≥ 2$，如果 $n$ 是素数，那么我们就完成了。如果不是，那么根据素数含义的一个特征，它有一个非平凡因子 $m$，我们可以将归纳假设应用于它。通过下一个证明步骤来查看这是如何实现的。

```py
theorem  exists_prime_factor  {n  :  Nat}  (h  :  2  ≤  n)  :  ∃  p  :  Nat,  p.Prime  ∧  p  ∣  n  :=  by
  by_cases  np  :  n.Prime
  ·  use  n,  np
  induction'  n  using  Nat.strong_induction_on  with  n  ih
  rw  [Nat.prime_def_lt]  at  np
  push_neg  at  np
  rcases  np  h  with  ⟨m,  mltn,  mdvdn,  mne1⟩
  have  :  m  ≠  0  :=  by
  intro  mz
  rw  [mz,  zero_dvd_iff]  at  mdvdn
  linarith
  have  mgt2  :  2  ≤  m  :=  two_le  this  mne1
  by_cases  mp  :  m.Prime
  ·  use  m,  mp
  ·  rcases  ih  m  mltn  mgt2  mp  with  ⟨p,  pp,  pdvd⟩
  use  p,  pp
  apply  pdvd.trans  mdvdn 
```

我们现在可以证明我们定理的以下表述。看看你是否能完成草图。你可以使用 `Nat.factorial_pos`、`Nat.dvd_factorial` 和 `Nat.dvd_sub'`。

```py
theorem  primes_infinite  :  ∀  n,  ∃  p  >  n,  Nat.Prime  p  :=  by
  intro  n
  have  :  2  ≤  Nat.factorial  n  +  1  :=  by
  sorry
  rcases  exists_prime_factor  this  with  ⟨p,  pp,  pdvd⟩
  refine  ⟨p,  ?_,  pp⟩
  show  p  >  n
  by_contra  ple
  push_neg  at  ple
  have  :  p  ∣  Nat.factorial  n  :=  by
  sorry
  have  :  p  ∣  1  :=  by
  sorry
  show  False
  sorry 
```

让我们考虑上述证明的一个变体，其中我们不是使用阶乘函数，而是假设我们被一个有限集 $\{ p_1, \ldots, p_n \}$ 给定，并考虑 $\prod_{i = 1}^n p_i + 1$ 的一个素数因子。这个素数因子必须与每个 $p_i$ 不同，这表明不存在包含所有素数的有限集。

将此论证形式化需要我们关于有限集进行推理。在 Lean 中，对于任何类型 `α`，类型 `Finset α` 表示类型 `α` 的元素构成的有限集。关于有限集的计算机推理需要有一个在 `α` 上测试等价的程序，这就是为什么下面的代码片段包含了假设 `[DecidableEq α]`。对于像 `ℕ`、`ℤ` 和 `ℚ` 这样的具体数据类型，这个假设会自动满足。当推理实数时，可以使用经典逻辑并放弃计算解释来满足这个假设。

我们使用命令 `open Finset` 来使用相关定理的简短名称。与集合的情况不同，涉及有限集的大多数等价关系在定义上并不成立，因此需要手动使用等价关系如 `Finset.subset_iff`、`Finset.mem_union`、`Finset.mem_inter` 和 `Finset.mem_sdiff` 来展开。`ext` 策略仍然可以用来通过表明一个有限集的每个元素都是另一个集合的元素来证明两个有限集相等。

```py
open  Finset

section
variable  {α  :  Type*}  [DecidableEq  α]  (r  s  t  :  Finset  α)

example  :  r  ∩  (s  ∪  t)  ⊆  r  ∩  s  ∪  r  ∩  t  :=  by
  rw  [subset_iff]
  intro  x
  rw  [mem_inter,  mem_union,  mem_union,  mem_inter,  mem_inter]
  tauto

example  :  r  ∩  (s  ∪  t)  ⊆  r  ∩  s  ∪  r  ∩  t  :=  by
  simp  [subset_iff]
  intro  x
  tauto

example  :  r  ∩  s  ∪  r  ∩  t  ⊆  r  ∩  (s  ∪  t)  :=  by
  simp  [subset_iff]
  intro  x
  tauto

example  :  r  ∩  s  ∪  r  ∩  t  =  r  ∩  (s  ∪  t)  :=  by
  ext  x
  simp
  tauto

end 
```

我们使用了一个新的技巧：`tauto` 策略（以及一个增强版本 `tauto!`，它使用经典逻辑）可以用来处理命题恒真。看看你是否能使用这些方法来证明下面的两个例子。

```py
example  :  (r  ∪  s)  ∩  (r  ∪  t)  =  r  ∪  s  ∩  t  :=  by
  sorry
example  :  (r  \  s)  \  t  =  r  \  (s  ∪  t)  :=  by
  sorry 
```

定理 `Finset.dvd_prod_of_mem` 告诉我们，如果 `n` 是有限集 `s` 的一个元素，那么 `n` 可以整除 `∏ i ∈ s, i`。

```py
example  (s  :  Finset  ℕ)  (n  :  ℕ)  (h  :  n  ∈  s)  :  n  ∣  ∏  i  ∈  s,  i  :=
  Finset.dvd_prod_of_mem  _  h 
```

我们还需要知道，在 `n` 是素数且 `s` 是素数集合的情况下，逆命题也成立。为了证明这一点，我们需要以下引理，你应该能够使用定理 `Nat.Prime.eq_one_or_self_of_dvd` 来证明它。

```py
theorem  _root_.Nat.Prime.eq_of_dvd_of_prime  {p  q  :  ℕ}
  (prime_p  :  Nat.Prime  p)  (prime_q  :  Nat.Prime  q)  (h  :  p  ∣  q)  :
  p  =  q  :=  by
  sorry 
```

我们可以使用这个引理来证明，如果一个素数 `p` 整除有限个素数的乘积，那么它等于其中的一个。Mathlib 提供了一个关于有限集的有用归纳原理：为了证明一个性质对任意有限集 `s` 成立，需要证明它对空集成立，并且在我们添加一个新元素 `a ∉ s` 时它仍然成立。这个原理被称为 `Finset.induction_on`。当我们告诉归纳策略使用它时，我们还可以指定 `a` 和 `s` 的名称，以及在归纳步骤中 `a ∉ s` 的假设的名称，以及归纳假设的名称。表达式 `Finset.insert a s` 表示 `s` 与单元素 `a` 的并集。然后，`Finset.prod_empty` 和 `Finset.prod_insert` 提供了与乘积相关的重写规则。在下面的证明中，第一个 `simp` 应用了 `Finset.prod_empty`。逐步查看证明的开始，以查看归纳展开，然后完成它。

```py
theorem  mem_of_dvd_prod_primes  {s  :  Finset  ℕ}  {p  :  ℕ}  (prime_p  :  p.Prime)  :
  (∀  n  ∈  s,  Nat.Prime  n)  →  (p  ∣  ∏  n  ∈  s,  n)  →  p  ∈  s  :=  by
  intro  h₀  h₁
  induction'  s  using  Finset.induction_on  with  a  s  ans  ih
  ·  simp  at  h₁
  linarith  [prime_p.two_le]
  simp  [Finset.prod_insert  ans,  prime_p.dvd_mul]  at  h₀  h₁
  rw  [mem_insert]
  sorry 
```

我们还需要有限集的一个最后性质。给定一个元素 `s : Set α` 和一个关于 `α` 的谓词 `P`，在 第四章 中，我们用 `{ x ∈ s | P x }` 表示满足 `P` 的 `s` 的元素集合。给定 `s : Finset α`，类似的概念写成 `s.filter P`。

```py
example  (s  :  Finset  ℕ)  (x  :  ℕ)  :  x  ∈  s.filter  Nat.Prime  ↔  x  ∈  s  ∧  x.Prime  :=
  mem_filter 
```

现在我们证明一个关于无穷多个素数的陈述的另一种表述，即给定任何 `s : Finset ℕ`，存在一个素数 `p` 它不是 `s` 的元素。为了达到矛盾，我们假设所有素数都在 `s` 中，然后缩减到一个只包含所有素数的集合 `s'`。取该集合的乘积，加一，并找到结果的素数因子，导致我们寻找的矛盾。看看你是否能完成下面的草图。你可以在第一个 `have` 的证明中使用 `Finset.prod_pos`。

```py
theorem  primes_infinite'  :  ∀  s  :  Finset  Nat,  ∃  p,  Nat.Prime  p  ∧  p  ∉  s  :=  by
  intro  s
  by_contra  h
  push_neg  at  h
  set  s'  :=  s.filter  Nat.Prime  with  s'_def
  have  mem_s'  :  ∀  {n  :  ℕ},  n  ∈  s'  ↔  n.Prime  :=  by
  intro  n
  simp  [s'_def]
  apply  h
  have  :  2  ≤  (∏  i  ∈  s',  i)  +  1  :=  by
  sorry
  rcases  exists_prime_factor  this  with  ⟨p,  pp,  pdvd⟩
  have  :  p  ∣  ∏  i  ∈  s',  i  :=  by
  sorry
  have  :  p  ∣  1  :=  by
  convert  Nat.dvd_sub  pdvd  this
  simp
  show  False
  sorry 
```

因此，我们已经看到了两种表达无穷多个素数的方法：说它们不受任何 `n` 的限制，以及说它们不包含在任何有限集 `s` 中。下面的两个证明表明这些表述是等价的。在第二个中，为了形成 `s.filter Q`，我们必须假设存在一个判断 `Q` 是否成立的程序。Lean 知道 `Nat.Prime` 有一个程序。一般来说，如果我们通过编写 `open Classical` 使用经典逻辑，我们可以省略这个假设。

在 Mathlib 中，`Finset.sup s f` 表示 `f x` 在 `x` 在 `s` 上取值时的上确界，当 `s` 为空且 `f` 的陪域为 `ℕ` 时返回 `0`。在第一个证明中，我们使用 `s.sup id`，其中 `id` 是恒等函数，来引用 `s` 中的最大值。

```py
theorem  bounded_of_ex_finset  (Q  :  ℕ  →  Prop)  :
  (∃  s  :  Finset  ℕ,  ∀  k,  Q  k  →  k  ∈  s)  →  ∃  n,  ∀  k,  Q  k  →  k  <  n  :=  by
  rintro  ⟨s,  hs⟩
  use  s.sup  id  +  1
  intro  k  Qk
  apply  Nat.lt_succ_of_le
  show  id  k  ≤  s.sup  id
  apply  le_sup  (hs  k  Qk)

theorem  ex_finset_of_bounded  (Q  :  ℕ  →  Prop)  [DecidablePred  Q]  :
  (∃  n,  ∀  k,  Q  k  →  k  ≤  n)  →  ∃  s  :  Finset  ℕ,  ∀  k,  Q  k  ↔  k  ∈  s  :=  by
  rintro  ⟨n,  hn⟩
  use  (range  (n  +  1)).filter  Q
  intro  k
  simp  [Nat.lt_succ_iff]
  exact  hn  k 
```

我们第二个证明中关于存在无限多个质数的微小变化表明，存在无限多个与 4 同余 3 的质数。论证如下。首先，注意如果两个数$m$和$n$的乘积等于 4 模 3，那么这两个数中至少有一个与 4 同余 3。毕竟，它们都必须是奇数，如果它们都等于 4 模 1，那么它们的乘积也是。我们可以利用这个观察结果来证明如果某个大于 2 的数与 4 同余 3，那么这个数有一个与 4 同余 3 的质数因子。

现在假设只有有限多个与 4 同余 3 的质数，比如说，$p_1, \ldots, p_k$。不失一般性，我们可以假设$p_1 = 3$。考虑乘积$4 \prod_{i = 2}^k p_i + 3$。很容易看出这个数与 4 同余 3，所以它有一个与 4 同余 3 的质数因子$p$。不可能$p = 3$；因为$p$能整除$4 \prod_{i = 2}^k p_i + 3$，如果$p$等于 3，那么它也会整除$\prod_{i = 2}^k p_i$，这意味着$p$等于$p_i$中的一个，对于$i = 2, \ldots, k$；但我们已经排除了 3。所以$p$必须是其他元素$p_i$中的一个。但在那种情况下，$p$能整除$4 \prod_{i = 2}^k p_i$和 3，这与它不是 3 的事实相矛盾。

在 Lean 中，表示`n % m`（读作“`n`模`m`”），表示`n`除以`m`的余数。

```py
example  :  27  %  4  =  3  :=  by  norm_num 
```

然后，我们可以将陈述“`n`与 4 同余 3”表示为`n % 4 = 3`。以下示例和定理总结了我们将需要使用的事实。第一个命名的定理是通过对少数几个情况推理的另一个示例。在第二个命名的定理中，记住分号意味着后续的策略块应用于由前面的策略创建的所有目标。

```py
example  (n  :  ℕ)  :  (4  *  n  +  3)  %  4  =  3  :=  by
  rw  [add_comm,  Nat.add_mul_mod_self_left]

theorem  mod_4_eq_3_or_mod_4_eq_3  {m  n  :  ℕ}  (h  :  m  *  n  %  4  =  3)  :  m  %  4  =  3  ∨  n  %  4  =  3  :=  by
  revert  h
  rw  [Nat.mul_mod]
  have  :  m  %  4  <  4  :=  Nat.mod_lt  m  (by  norm_num)
  interval_cases  m  %  4  <;>  simp  [-Nat.mul_mod_mod]
  have  :  n  %  4  <  4  :=  Nat.mod_lt  n  (by  norm_num)
  interval_cases  n  %  4  <;>  simp

theorem  two_le_of_mod_4_eq_3  {n  :  ℕ}  (h  :  n  %  4  =  3)  :  2  ≤  n  :=  by
  apply  two_le  <;>
  ·  intro  neq
  rw  [neq]  at  h
  norm_num  at  h 
```

我们还需要以下事实，即如果`m`是`n`的非平凡因子，那么`n / m`也是。尝试使用`Nat.div_dvd_of_dvd`和`Nat.div_lt_self`来完成证明。

```py
theorem  aux  {m  n  :  ℕ}  (h₀  :  m  ∣  n)  (h₁  :  2  ≤  m)  (h₂  :  m  <  n)  :  n  /  m  ∣  n  ∧  n  /  m  <  n  :=  by
  sorry 
```

现在将所有这些部分组合起来，以证明任何与 4 同余 3 的数都有一个具有相同性质的质数因子。

```py
theorem  exists_prime_factor_mod_4_eq_3  {n  :  Nat}  (h  :  n  %  4  =  3)  :
  ∃  p  :  Nat,  p.Prime  ∧  p  ∣  n  ∧  p  %  4  =  3  :=  by
  by_cases  np  :  n.Prime
  ·  use  n
  induction'  n  using  Nat.strong_induction_on  with  n  ih
  rw  [Nat.prime_def_lt]  at  np
  push_neg  at  np
  rcases  np  (two_le_of_mod_4_eq_3  h)  with  ⟨m,  mltn,  mdvdn,  mne1⟩
  have  mge2  :  2  ≤  m  :=  by
  apply  two_le  _  mne1
  intro  mz
  rw  [mz,  zero_dvd_iff]  at  mdvdn
  linarith
  have  neq  :  m  *  (n  /  m)  =  n  :=  Nat.mul_div_cancel'  mdvdn
  have  :  m  %  4  =  3  ∨  n  /  m  %  4  =  3  :=  by
  apply  mod_4_eq_3_or_mod_4_eq_3
  rw  [neq,  h]
  rcases  this  with  h1  |  h1
  .  sorry
  .  sorry 
```

我们已经接近终点。给定一个由质数组成的集合`s`，如果集合中存在 3，我们需要讨论从该集合中移除 3 的结果。函数`Finset.erase`可以处理这种情况。

```py
example  (m  n  :  ℕ)  (s  :  Finset  ℕ)  (h  :  m  ∈  erase  s  n)  :  m  ≠  n  ∧  m  ∈  s  :=  by
  rwa  [mem_erase]  at  h

example  (m  n  :  ℕ)  (s  :  Finset  ℕ)  (h  :  m  ∈  erase  s  n)  :  m  ≠  n  ∧  m  ∈  s  :=  by
  simp  at  h
  assumption 
```

现在，我们已经准备好证明存在无限多个与 4 同余 3 的质数。在下面填入缺失的部分。我们的解决方案在过程中使用了`Nat.dvd_add_iff_left`和`Nat.dvd_sub'`。

```py
theorem  primes_mod_4_eq_3_infinite  :  ∀  n,  ∃  p  >  n,  Nat.Prime  p  ∧  p  %  4  =  3  :=  by
  by_contra  h
  push_neg  at  h
  rcases  h  with  ⟨n,  hn⟩
  have  :  ∃  s  :  Finset  Nat,  ∀  p  :  ℕ,  p.Prime  ∧  p  %  4  =  3  ↔  p  ∈  s  :=  by
  apply  ex_finset_of_bounded
  use  n
  contrapose!  hn
  rcases  hn  with  ⟨p,  ⟨pp,  p4⟩,  pltn⟩
  exact  ⟨p,  pltn,  pp,  p4⟩
  rcases  this  with  ⟨s,  hs⟩
  have  h₁  :  ((4  *  ∏  i  ∈  erase  s  3,  i)  +  3)  %  4  =  3  :=  by
  sorry
  rcases  exists_prime_factor_mod_4_eq_3  h₁  with  ⟨p,  pp,  pdvd,  p4eq⟩
  have  ps  :  p  ∈  s  :=  by
  sorry
  have  pne3  :  p  ≠  3  :=  by
  sorry
  have  :  p  ∣  4  *  ∏  i  ∈  erase  s  3,  i  :=  by
  sorry
  have  :  p  ∣  3  :=  by
  sorry
  have  :  p  =  3  :=  by
  sorry
  contradiction 
```

如果你设法完成了证明，恭喜你！这已经是一项正式化的重大成就。## 5.4. 更多归纳

在第 5.2 节中，我们看到了如何通过自然数的递归定义阶乘函数。

```py
def  fac  :  ℕ  →  ℕ
  |  0  =>  1
  |  n  +  1  =>  (n  +  1)  *  fac  n 
```

我们还看到了如何使用`induction'`策略证明定理。

```py
theorem  fac_pos  (n  :  ℕ)  :  0  <  fac  n  :=  by
  induction'  n  with  n  ih
  ·  rw  [fac]
  exact  zero_lt_one
  rw  [fac]
  exact  mul_pos  n.succ_pos  ih 
```

`induction` 策略（不带撇号）允许更结构化的语法。

```py
example  (n  :  ℕ)  :  0  <  fac  n  :=  by
  induction  n
  case  zero  =>
  rw  [fac]
  exact  zero_lt_one
  case  succ  n  ih  =>
  rw  [fac]
  exact  mul_pos  n.succ_pos  ih

example  (n  :  ℕ)  :  0  <  fac  n  :=  by
  induction  n  with
  |  zero  =>
  rw  [fac]
  exact  zero_lt_one
  |  succ  n  ih  =>
  rw  [fac]
  exact  mul_pos  n.succ_pos  ih 
```

如同往常，你可以悬停在 `induction` 关键字上以阅读文档。案例的名称 `zero` 和 `succ` 来自于类型 ℕ 的定义。注意，`succ` 案例允许你为归纳变量和归纳假设选择任何你想要的名称，这里为 `n` 和 `ih`。你甚至可以使用定义递归函数的相同符号来证明定理。

```py
theorem  fac_pos'  :  ∀  n,  0  <  fac  n
  |  0  =>  by
  rw  [fac]
  exact  zero_lt_one
  |  n  +  1  =>  by
  rw  [fac]
  exact  mul_pos  n.succ_pos  (fac_pos'  n) 
```

注意也缺少了 `:=`，冒号后面的 `∀ n`，每个案例中的 `by` 关键字，以及归纳调用 `fac_pos' n`。这就像定理是 `n` 的递归函数，在归纳步骤中我们进行递归调用。

这种定义风格非常灵活。Lean 的设计者内置了定义递归函数的复杂手段，这些手段也扩展到了归纳证明。例如，我们可以用多个基本情况定义斐波那契函数。

```py
@[simp]  def  fib  :  ℕ  →  ℕ
  |  0  =>  0
  |  1  =>  1
  |  n  +  2  =>  fib  n  +  fib  (n  +  1) 
```

`@[simp]` 注释表示简化器将使用定义方程。你也可以通过编写 `rw [fib]` 来应用它们。下面将有助于为 `n + 2` 情况命名。

```py
theorem  fib_add_two  (n  :  ℕ)  :  fib  (n  +  2)  =  fib  n  +  fib  (n  +  1)  :=  rfl

example  (n  :  ℕ)  :  fib  (n  +  2)  =  fib  n  +  fib  (n  +  1)  :=  by  rw  [fib] 
```

使用 Lean 的递归函数表示法，你可以通过自然数上的归纳来执行与 `fib` 的递归定义相对应的证明。以下示例提供了第 n 个斐波那契数的显式公式，该公式以黄金分割数 `φ` 和其共轭 `φ'` 为基础。我们必须告诉 Lean，我们不期望我们的定义生成代码，因为实数上的算术运算是不可计算的。

```py
noncomputable  section

def  phi  :  ℝ  :=  (1  +  √5)  /  2
def  phi'  :  ℝ  :=  (1  -  √5)  /  2

theorem  phi_sq  :  phi²  =  phi  +  1  :=  by
  field_simp  [phi,  add_sq];  ring

theorem  phi'_sq  :  phi'²  =  phi'  +  1  :=  by
  field_simp  [phi',  sub_sq];  ring

theorem  fib_eq  :  ∀  n,  fib  n  =  (phi^n  -  phi'^n)  /  √5
  |  0  =>  by  simp
  |  1  =>  by  field_simp  [phi,  phi']
  |  n+2  =>  by  field_simp  [fib_eq,  pow_add,  phi_sq,  phi'_sq];  ring

end 
```

涉及斐波那契函数的归纳证明不必是那种形式。以下我们重现了 `Mathlib` 证明连续的斐波那契数是互质的。

```py
theorem  fib_coprime_fib_succ  (n  :  ℕ)  :  Nat.Coprime  (fib  n)  (fib  (n  +  1))  :=  by
  induction  n  with
  |  zero  =>  simp
  |  succ  n  ih  =>
  simp  only  [fib,  Nat.coprime_add_self_right]
  exact  ih.symm 
```

使用 Lean 的计算解释，我们可以评估斐波那契数。

```py
#eval  fib  6
#eval  List.range  20  |>.map  fib 
```

`fib` 的直接实现计算效率低下。实际上，它的运行时间与其参数呈指数关系。（你应该思考一下为什么。）在 Lean 中，我们可以实现以下尾递归版本，其运行时间与 `n` 线性相关，并证明它计算的是相同的函数。

```py
def  fib'  (n  :  Nat)  :  Nat  :=
  aux  n  0  1
where  aux
  |  0,  x,  _  =>  x
  |  n+1,  x,  y  =>  aux  n  y  (x  +  y)

theorem  fib'.aux_eq  (m  n  :  ℕ)  :  fib'.aux  n  (fib  m)  (fib  (m  +  1))  =  fib  (n  +  m)  :=  by
  induction  n  generalizing  m  with
  |  zero  =>  simp  [fib'.aux]
  |  succ  n  ih  =>  rw  [fib'.aux,  ←fib_add_two,  ih,  add_assoc,  add_comm  1]

theorem  fib'_eq_fib  :  fib'  =  fib  :=  by
  ext  n
  erw  [fib',  fib'.aux_eq  0  n];  rfl

#eval  fib'  10000 
```

注意 `fib'.aux_eq` 证明中的 `generalizing` 关键字。它用于在归纳假设前插入 `∀ m`，这样在归纳步骤中，`m` 可以取不同的值。你可以逐步检查证明，并确认在这种情况下，量词需要在归纳步骤中实例化为 `m + 1`。

注意也使用了 `erw`（表示“扩展重写”）而不是 `rw`。这是因为为了重写目标 `fib'.aux_eq`，`fib 0` 和 `fib 1` 必须分别简化为 `0` 和 `1`。`erw` 策略在展开定义以匹配参数方面比 `rw` 更激进。这并不总是好主意；在某些情况下，它可能会浪费很多时间，所以请谨慎使用 `erw`。

这是`generalizing`关键字在证明`Mathlib`中发现的另一个恒等式时的另一个例子。该恒等式的不正式证明可以在[这里](https://proofwiki.org/wiki/Fibonacci_Number_in_terms_of_Smaller_Fibonacci_Numbers)找到。我们提供了正式证明的两个变体。

```py
theorem  fib_add  (m  n  :  ℕ)  :  fib  (m  +  n  +  1)  =  fib  m  *  fib  n  +  fib  (m  +  1)  *  fib  (n  +  1)  :=  by
  induction  n  generalizing  m  with
  |  zero  =>  simp
  |  succ  n  ih  =>
  specialize  ih  (m  +  1)
  rw  [add_assoc  m  1  n,  add_comm  1  n]  at  ih
  simp  only  [fib_add_two,  Nat.succ_eq_add_one,  ih]
  ring

theorem  fib_add'  :  ∀  m  n,  fib  (m  +  n  +  1)  =  fib  m  *  fib  n  +  fib  (m  +  1)  *  fib  (n  +  1)
  |  _,  0  =>  by  simp
  |  m,  n  +  1  =>  by
  have  :=  fib_add'  (m  +  1)  n
  rw  [add_assoc  m  1  n,  add_comm  1  n]  at  this
  simp  only  [fib_add_two,  Nat.succ_eq_add_one,  this]
  ring 
```

作为练习，使用`fib_add`来证明以下内容。

```py
example  (n  :  ℕ):  (fib  n)  ^  2  +  (fib  (n  +  1))  ^  2  =  fib  (2  *  n  +  1)  :=  by  sorry 
```

Lean 定义递归函数的机制足够灵活，允许任意递归调用，只要参数的复杂性根据某个良基度量递减。在下一个例子中，我们展示了每个自然数`n ≠ 1`都有一个素数因子，利用了如果`n`非零且不是素数，它有一个较小的因子的这一事实。（你可以检查 Mathlib 在`Nat`命名空间中是否有同名定理，尽管它的证明与这里给出的不同。）

```py
#check  (@Nat.not_prime_iff_exists_dvd_lt  :
  ∀  {n  :  ℕ},  2  ≤  n  →  (¬Nat.Prime  n  ↔  ∃  m,  m  ∣  n  ∧  2  ≤  m  ∧  m  <  n))

theorem  ne_one_iff_exists_prime_dvd  :  ∀  {n},  n  ≠  1  ↔  ∃  p  :  ℕ,  p.Prime  ∧  p  ∣  n
  |  0  =>  by  simpa  using  Exists.intro  2  Nat.prime_two
  |  1  =>  by  simp  [Nat.not_prime_one]
  |  n  +  2  =>  by
  have  hn  :  n+2  ≠  1  :=  by  omega
  simp  only  [Ne,  not_false_iff,  true_iff,  hn]
  by_cases  h  :  Nat.Prime  (n  +  2)
  ·  use  n+2,  h
  ·  have  :  2  ≤  n  +  2  :=  by  omega
  rw  [Nat.not_prime_iff_exists_dvd_lt  this]  at  h
  rcases  h  with  ⟨m,  mdvdn,  mge2,  -⟩
  have  :  m  ≠  1  :=  by  omega
  rw  [ne_one_iff_exists_prime_dvd]  at  this
  rcases  this  with  ⟨p,  primep,  pdvdm⟩
  use  p,  primep
  exact  pdvdm.trans  mdvdn 
```

这行代码`rw [ne_one_iff_exists_prime_dvd] at this`就像一个魔术：我们在自己的证明中使用了我们正在证明的定理。使其工作的是归纳调用在`m`处实例化，当前情况是`n + 2`，并且上下文有`m < n + 2`。Lean 能够找到假设并使用它来证明归纳是良基的。Lean 在找出什么是递减的方面相当出色；在这种情况下，定理陈述中的`n`的选择和小于关系是显而易见的。在更复杂的情况下，Lean 提供了提供此信息的机制。请参阅 Lean 参考手册中关于[良基递归](https://lean-lang.org/doc/reference/latest//Definitions/Recursive-Definitions/#well-founded-recursion)的部分。

有时，在证明中，你需要根据自然数`n`是零还是后继来分情况讨论，而不需要在后继情况下要求归纳假设。为此，你可以使用`cases`和`rcases`策略。

```py
theorem  zero_lt_of_mul_eq_one  (m  n  :  ℕ)  :  n  *  m  =  1  →  0  <  n  ∧  0  <  m  :=  by
  cases  n  <;>  cases  m  <;>  simp

example  (m  n  :  ℕ)  :  n*m  =  1  →  0  <  n  ∧  0  <  m  :=  by
  rcases  m  with  (_  |  m);  simp
  rcases  n  with  (_  |  n)  <;>  simp 
```

这是一个有用的技巧。通常，你有一个关于自然数`n`的定理，其中零的情况很容易处理。如果你对`n`进行情况分析并快速处理零的情况，你将剩下原始目标，只是将`n`替换为`n + 1`。上一节 下一节

* * *

© 版权所有 2020-2025，Jeremy Avigad，Patrick Massot。文本许可协议为 CC BY 4.0。

使用[Sphinx](https://www.sphinx-doc.org/)和[Read the Docs](https://readthedocs.org)提供的[主题](https://github.com/readthedocs/sphinx_rtd_theme)构建。在本章中，我们向您展示如何将数论中的某些基本结果形式化。随着我们处理更实质性的数学内容，证明将变得更长、更复杂，建立在您已经掌握的技能之上。

## 5.1. 无理根

让我们从古希腊人已知的一个事实开始，即 2 的平方根是无理数。如果我们假设相反的情况，我们可以将$\sqrt{2} = a / b$写成最简分数。平方两边得到$a² = 2 b²$，这意味着`a`是偶数。如果我们写`a = 2c`，那么我们得到$4c² = 2 b²$，从而$b² = 2 c²$。这表明`b`也是偶数，这与我们假设的`a / b`已经被化简到最简形式的事实相矛盾。

说`a / b`是最简分数意味着`a`和`b`没有公共因子，也就是说，它们是*互质*。Mathlib 定义谓词`Nat.Coprime m n`为`Nat.gcd m n = 1`。使用 Lean 的匿名投影符号，如果`s`和`t`是类型为`Nat`的表达式，我们可以写`s.Coprime t`而不是`Nat.Coprime s t`，对于`Nat.gcd`也是如此。通常，当需要时，Lean 会自动展开`Nat.Coprime`的定义，但我们可以通过重写或使用标识符`Nat.Coprime`进行简化来手动完成。`norm_num`策略足够智能，可以计算具体值。

```py
#print  Nat.Coprime

example  (m  n  :  Nat)  (h  :  m.Coprime  n)  :  m.gcd  n  =  1  :=
  h

example  (m  n  :  Nat)  (h  :  m.Coprime  n)  :  m.gcd  n  =  1  :=  by
  rw  [Nat.Coprime]  at  h
  exact  h

example  :  Nat.Coprime  12  7  :=  by  norm_num

example  :  Nat.gcd  12  8  =  4  :=  by  norm_num 
```

我们已经在第 2.4 节中遇到了`gcd`函数。对于整数也有`gcd`的版本；我们将在下面讨论不同数系之间的关系。甚至还有一个通用的`gcd`函数和通用的`Prime`和`Coprime`概念，这些在一般的代数结构中是有意义的。我们将在下一章了解 Lean 如何管理这种通用性。同时，在本节中，我们将关注自然数。

我们还需要一个素数的概念，即`Nat.Prime`。定理`Nat.prime_def_lt`提供了一个熟悉的特征描述，而`Nat.Prime.eq_one_or_self_of_dvd`提供了另一个。

```py
#check  Nat.prime_def_lt

example  (p  :  ℕ)  (prime_p  :  Nat.Prime  p)  :  2  ≤  p  ∧  ∀  m  :  ℕ,  m  <  p  →  m  ∣  p  →  m  =  1  :=  by
  rwa  [Nat.prime_def_lt]  at  prime_p

#check  Nat.Prime.eq_one_or_self_of_dvd

example  (p  :  ℕ)  (prime_p  :  Nat.Prime  p)  :  ∀  m  :  ℕ,  m  ∣  p  →  m  =  1  ∨  m  =  p  :=
  prime_p.eq_one_or_self_of_dvd

example  :  Nat.Prime  17  :=  by  norm_num

-- commonly used
example  :  Nat.Prime  2  :=
  Nat.prime_two

example  :  Nat.Prime  3  :=
  Nat.prime_three 
```

在自然数中，素数有一个属性，即它不能写成非平凡因子的乘积。在更广泛的数学背景下，具有这种属性的环元素被称为*不可约*。如果一个环元素在乘积中除以，它就除以其中一个因子，那么这个环元素被称为*素数*。自然数的一个重要属性是，在这种设置中，这两个概念是一致的，从而产生了定理`Nat.Prime.dvd_mul`。

我们可以使用这个事实来建立上述论证中的一个关键属性：如果一个数的平方是偶数，那么这个数也是偶数。Mathlib 在`Algebra.Group.Even`中定义谓词`Even`，但如下文所述，我们将简单地使用`2 ∣ m`来表示`m`是偶数。

```py
#check  Nat.Prime.dvd_mul
#check  Nat.Prime.dvd_mul  Nat.prime_two
#check  Nat.prime_two.dvd_mul

theorem  even_of_even_sqr  {m  :  ℕ}  (h  :  2  ∣  m  ^  2)  :  2  ∣  m  :=  by
  rw  [pow_two,  Nat.prime_two.dvd_mul]  at  h
  cases  h  <;>  assumption

example  {m  :  ℕ}  (h  :  2  ∣  m  ^  2)  :  2  ∣  m  :=
  Nat.Prime.dvd_of_dvd_pow  Nat.prime_two  h 
```

随着我们继续前进，你需要熟练地找到你需要的事实。记住，如果你能猜出名称的前缀并且你已经导入了相关的库，你可以使用制表符补全（有时需要 `ctrl-tab`）来找到你想要的内容。你可以在任何标识符上使用 `ctrl-click` 跳转到其定义的文件，这使你能够浏览附近的定义和定理。你还可以使用 [Lean 社区网页](https://leanprover-community.github.io/) 上的搜索引擎，如果所有其他方法都失败了，不要犹豫在 [Zulip](https://leanprover.zulipchat.com/) 上提问。

```py
example  (a  b  c  :  Nat)  (h  :  a  *  b  =  a  *  c)  (h'  :  a  ≠  0)  :  b  =  c  :=
  -- apply? suggests the following:
  (mul_right_inj'  h').mp  h 
```

我们证明二平方根无理性的核心包含在以下定理中。看看你是否能填充证明草图，使用 `even_of_even_sqr` 和定理 `Nat.dvd_gcd`。

```py
example  {m  n  :  ℕ}  (coprime_mn  :  m.Coprime  n)  :  m  ^  2  ≠  2  *  n  ^  2  :=  by
  intro  sqr_eq
  have  :  2  ∣  m  :=  by
  sorry
  obtain  ⟨k,  meq⟩  :=  dvd_iff_exists_eq_mul_left.mp  this
  have  :  2  *  (2  *  k  ^  2)  =  2  *  n  ^  2  :=  by
  rw  [←  sqr_eq,  meq]
  ring
  have  :  2  *  k  ^  2  =  n  ^  2  :=
  sorry
  have  :  2  ∣  n  :=  by
  sorry
  have  :  2  ∣  m.gcd  n  :=  by
  sorry
  have  :  2  ∣  1  :=  by
  sorry
  norm_num  at  this 
```

实际上，只需做很少的修改，我们就可以用任意素数替换 `2`。在下一个例子中尝试一下。在证明的末尾，你需要从 `p ∣ 1` 推导出矛盾。你可以使用 `Nat.Prime.two_le`，它表明任何素数都大于或等于二，以及 `Nat.le_of_dvd`。

```py
example  {m  n  p  :  ℕ}  (coprime_mn  :  m.Coprime  n)  (prime_p  :  p.Prime)  :  m  ^  2  ≠  p  *  n  ^  2  :=  by
  sorry 
```

让我们考虑另一种方法。以下是一个快速证明：如果 `p` 是素数，那么 `m² ≠ p n²`：如果我们假设 `m² = p n²` 并考虑 `m` 和 `n` 的素数分解，那么 `p` 在等式的左边出现偶数次，在右边出现奇数次，这是矛盾的。请注意，这个论证要求 `n` 以及因此 `m` 不为零。下面的形式化确认了这个假设是充分的。

唯一分解定理指出，除了零以外的任何自然数都可以以唯一的方式写成素数的乘积。Mathlib 包含了这个定理的正式版本，它通过一个函数 `Nat.primeFactorsList` 来表达，该函数返回一个数素因子的非递减顺序列表。该库证明了 `Nat.primeFactorsList n` 的所有元素都是素数，任何大于零的 `n` 都等于其因子的乘积，并且如果 `n` 等于另一个素数列表的乘积，那么那个列表是 `Nat.primeFactorsList n` 的一个排列。

```py
#check  Nat.primeFactorsList
#check  Nat.prime_of_mem_primeFactorsList
#check  Nat.prod_primeFactorsList
#check  Nat.primeFactorsList_unique 
```

尽管我们还没有讨论列表成员、乘积或排列，但你仍然可以浏览这些定理以及其他附近的定理。对于当前的任务，我们不需要这些内容。相反，我们将使用 Mathlib 有一个函数 `Nat.factorization` 的这一事实，该函数以函数的形式表示相同的数据。具体来说，`Nat.factorization n p`，我们也可以写作 `n.factorization p`，返回 `p` 在 `n` 的素因子分解中的次数。我们将使用以下三个事实。

```py
theorem  factorization_mul'  {m  n  :  ℕ}  (mnez  :  m  ≠  0)  (nnez  :  n  ≠  0)  (p  :  ℕ)  :
  (m  *  n).factorization  p  =  m.factorization  p  +  n.factorization  p  :=  by
  rw  [Nat.factorization_mul  mnez  nnez]
  rfl

theorem  factorization_pow'  (n  k  p  :  ℕ)  :
  (n  ^  k).factorization  p  =  k  *  n.factorization  p  :=  by
  rw  [Nat.factorization_pow]
  rfl

theorem  Nat.Prime.factorization'  {p  :  ℕ}  (prime_p  :  p.Prime)  :
  p.factorization  p  =  1  :=  by
  rw  [prime_p.factorization]
  simp 
```

实际上，`n.factorization` 在 Lean 中定义为有限支持函数，这解释了你将在上述证明中看到的奇怪符号。现在不用担心这个。对于我们的目的，我们可以将上述三个定理作为一个黑盒使用。

下一个例子表明，简化器足够聪明，可以将 `n² ≠ 0` 替换为 `n ≠ 0`。`simpa` 策略只是调用 `simp` 后跟 `assumption`。

看看你是否能使用上面的恒等式来填补证明中的缺失部分。

```py
example  {m  n  p  :  ℕ}  (nnz  :  n  ≠  0)  (prime_p  :  p.Prime)  :  m  ^  2  ≠  p  *  n  ^  2  :=  by
  intro  sqr_eq
  have  nsqr_nez  :  n  ^  2  ≠  0  :=  by  simpa
  have  eq1  :  Nat.factorization  (m  ^  2)  p  =  2  *  m.factorization  p  :=  by
  sorry
  have  eq2  :  (p  *  n  ^  2).factorization  p  =  2  *  n.factorization  p  +  1  :=  by
  sorry
  have  :  2  *  m.factorization  p  %  2  =  (2  *  n.factorization  p  +  1)  %  2  :=  by
  rw  [←  eq1,  sqr_eq,  eq2]
  rw  [add_comm,  Nat.add_mul_mod_self_left,  Nat.mul_mod_right]  at  this
  norm_num  at  this 
```

这个证明的另一个优点是它也具有普遍性。`2` 没有什么特别之处；经过一些小的修改，这个证明表明，每当我们将 `m^k = r * n^k` 写出来时，任何素数 `p` 在 `r` 中的次数必须是 `k` 的倍数。

要使用 `Nat.count_factors_mul_of_pos` 与 `r * n^k`，我们需要知道 `r` 是正数。但当 `r` 为零时，下面的定理是显而易见的，并且可以通过简化器轻松证明。因此，证明是分情况进行的。行 `rcases r with _ | r` 将目标替换为两个版本：一个是将 `r` 替换为 `0` 的版本，另一个是将 `r` 替换为 `r + 1` 的版本。在第二种情况下，我们可以使用定理 `r.succ_ne_zero`，它建立了 `r + 1 ≠ 0`（`succ` 表示后继）。

注意，以 `have : npow_nz` 开头的行提供了一个简短的证明项证明 `n^k ≠ 0`。要理解它是如何工作的，试着用策略证明来替换它，然后思考策略是如何描述证明项的。

看看你是否能填补下面证明中的缺失部分。在最后，你可以使用 `Nat.dvd_sub'` 和 `Nat.dvd_mul_right` 来完成它。

注意，这个例子并没有假设 `p` 是素数，但当 `p` 不是素数时，结论是显而易见的，因为根据定义，`r.factorization p` 是零，而且证明在所有情况下都适用。

```py
example  {m  n  k  r  :  ℕ}  (nnz  :  n  ≠  0)  (pow_eq  :  m  ^  k  =  r  *  n  ^  k)  {p  :  ℕ}  :
  k  ∣  r.factorization  p  :=  by
  rcases  r  with  _  |  r
  ·  simp
  have  npow_nz  :  n  ^  k  ≠  0  :=  fun  npowz  ↦  nnz  (pow_eq_zero  npowz)
  have  eq1  :  (m  ^  k).factorization  p  =  k  *  m.factorization  p  :=  by
  sorry
  have  eq2  :  ((r  +  1)  *  n  ^  k).factorization  p  =
  k  *  n.factorization  p  +  (r  +  1).factorization  p  :=  by
  sorry
  have  :  r.succ.factorization  p  =  k  *  m.factorization  p  -  k  *  n.factorization  p  :=  by
  rw  [←  eq1,  pow_eq,  eq2,  add_comm,  Nat.add_sub_cancel]
  rw  [this]
  sorry 
```

我们可能希望以多种方式改进这些结果。首先，一个证明说平方根是无理数应该对平方根说些什么，这可以理解为实数或复数中的一个元素。并且说它是无理数应该对有理数说些什么，即没有有理数等于它。此外，我们还应该将本节中的定理扩展到整数。虽然从数学上明显，如果我们能将平方根写成两个整数的商，那么我们也能将其写成两个自然数的商，但正式证明这一点需要一些努力。

在 Mathlib 中，自然数、整数、有理数、实数和复数由不同的数据类型表示。将注意力限制在单独的域中通常是有帮助的：我们将看到对自然数进行归纳是很容易的，而且当实数不在图中时，对整数除法的推理是最容易的。但需要在不同的域之间进行调解，这是一个头疼的问题，我们将在本章的后面回到这个问题。

我们也应该期待能够加强最后一个定理的结论，使其表明数字 `r` 是一个 `k`-次幂，因为它的 `k`-次方根仅仅是每个除 `r` 的质数乘以其在 `r` 中的重数除以 `k` 的乘积。为了做到这一点，我们需要更好的推理手段来处理有限集合上的乘积和求和，这也是我们将再次讨论的话题。

事实上，本节中的结果在 Mathlib 中以更大的普遍性得到确立，在 `Data.Real.Irrational` 中。`multiplicity` 的概念被定义为任意交换幺半群，并且它取值于扩展的自然数 `enat`，这为自然数添加了无穷大的值。在下一章中，我们将开始开发欣赏 Lean 支持这种普遍性的方法。## 5.2. 归纳与递归

自然数集合 $\mathbb{N} = \{ 0, 1, 2, \ldots \}$ 不仅在其自身的基础上具有根本的重要性，而且在构建新的数学对象中也起着核心作用。Lean 的基础允许我们声明 *归纳类型*，这些类型是通过给定的构造函数列表归纳生成的。在 Lean 中，自然数被声明如下。

```py
inductive  Nat  where
  |  zero  :  Nat
  |  succ  (n  :  Nat)  :  Nat 
```

您可以通过编写 `#check Nat` 并在标识符 `Nat` 上使用 `ctrl-click` 来在库中找到它。该命令指定 `Nat` 是由两个构造函数 `zero : Nat` 和 `succ : Nat → Nat` 自由且归纳生成的数据类型。当然，库引入了 `ℕ` 和 `0` 的符号分别代表 `nat` 和 `zero`。 (数字被转换为二进制表示，但我们现在不必担心这个细节。)

对于工作的数学家来说，“自由”一词的含义是类型 `Nat` 有一个元素 `zero` 和一个注入的后续函数 `succ`，其像不包括 `zero`。

```py
example  (n  :  Nat)  :  n.succ  ≠  Nat.zero  :=
  Nat.succ_ne_zero  n

example  (m  n  :  Nat)  (h  :  m.succ  =  n.succ)  :  m  =  n  :=
  Nat.succ.inj  h 
```

对于工作的数学家来说，“归纳”一词的含义是自然数伴随着归纳证明的原则和递归定义的原则。本节将向您展示如何使用这些。

这里是阶乘函数的递归定义的一个例子。

```py
def  fac  :  ℕ  →  ℕ
  |  0  =>  1
  |  n  +  1  =>  (n  +  1)  *  fac  n 
```

语法需要一些时间来适应。注意，第一行没有 `:=`。接下来的两行提供了递归定义的基础情况和归纳步骤。这些等式在定义上是成立的，但也可以通过将 `fac` 命名为 `simp` 或 `rw` 来手动使用。

```py
example  :  fac  0  =  1  :=
  rfl

example  :  fac  0  =  1  :=  by
  rw  [fac]

example  :  fac  0  =  1  :=  by
  simp  [fac]

example  (n  :  ℕ)  :  fac  (n  +  1)  =  (n  +  1)  *  fac  n  :=
  rfl

example  (n  :  ℕ)  :  fac  (n  +  1)  =  (n  +  1)  *  fac  n  :=  by
  rw  [fac]

example  (n  :  ℕ)  :  fac  (n  +  1)  =  (n  +  1)  *  fac  n  :=  by
  simp  [fac] 
```

实际上，阶乘函数已经在 Mathlib 中定义为 `Nat.factorial`。再次，您可以通过键入 `#check Nat.factorial` 并使用 `ctrl-click` 来跳转到它。为了说明目的，我们将在示例中继续使用 `fac`。在 `Nat.factorial` 的定义之前标注 `@[simp]` 指定定义方程应该被添加到简化器默认使用的恒等式数据库中。

归纳原理表明，我们可以通过证明该命题对 0 成立，并且每当它对自然数 $n$ 成立时，它也对 $n + 1$ 成立，来证明关于自然数的一般命题。以下证明中的 `induction' n with n ih` 行因此产生两个目标：在第一个目标中，我们需要证明 `0 < fac 0`，在第二个目标中，我们有额外的假设 `ih : 0 < fac n` 并需要证明 `0 < fac (n + 1)`。短语 `with n ih` 用于命名归纳假设的变量和假设，你可以为它们选择任何你想要的名称。

```py
theorem  fac_pos  (n  :  ℕ)  :  0  <  fac  n  :=  by
  induction'  n  with  n  ih
  ·  rw  [fac]
  exact  zero_lt_one
  rw  [fac]
  exact  mul_pos  n.succ_pos  ih 
```

`induction'` 策略足够智能，可以将依赖于归纳变量的假设作为归纳假设的一部分包括在内。逐步查看下一个示例，看看发生了什么。

```py
theorem  dvd_fac  {i  n  :  ℕ}  (ipos  :  0  <  i)  (ile  :  i  ≤  n)  :  i  ∣  fac  n  :=  by
  induction'  n  with  n  ih
  ·  exact  absurd  ipos  (not_lt_of_ge  ile)
  rw  [fac]
  rcases  Nat.of_le_succ  ile  with  h  |  h
  ·  apply  dvd_mul_of_dvd_right  (ih  h)
  rw  [h]
  apply  dvd_mul_right 
```

以下示例提供了一个关于阶乘函数的粗糙下界。实际上，从证明的案例开始似乎更容易，因此证明的其余部分从 $n = 1$ 的案例开始。看看你是否可以用 `pow_succ` 或 `pow_succ'` 通过归纳证明来完成论证。

```py
theorem  pow_two_le_fac  (n  :  ℕ)  :  2  ^  (n  -  1)  ≤  fac  n  :=  by
  rcases  n  with  _  |  n
  ·  simp  [fac]
  sorry 
```

归纳法常用于证明涉及有限和积的恒等式。Mathlib 定义了表达式 `Finset.sum s f`，其中 `s : Finset α` 是类型 `α` 的元素有限集合，`f` 是定义在 `α` 上的函数。函数 `f` 的陪域可以是支持具有零元素的交换、结合加法运算的任何类型。如果你导入 `Algebra.BigOperators.Ring` 并执行 `open BigOperators` 命令，你可以使用更具提示性的符号 `∑ x ∈ s, f x`。当然，也存在类似的操作和符号用于有限积。

我们将在下一节中讨论 `Finset` 类型及其支持的运算，并在稍后的章节中再次讨论。现在，我们只会使用 `Finset.range n`，这是小于 `n` 的自然数的有限集合。

```py
variable  {α  :  Type*}  (s  :  Finset  ℕ)  (f  :  ℕ  →  ℕ)  (n  :  ℕ)

#check  Finset.sum  s  f
#check  Finset.prod  s  f

open  BigOperators
open  Finset

example  :  s.sum  f  =  ∑  x  ∈  s,  f  x  :=
  rfl

example  :  s.prod  f  =  ∏  x  ∈  s,  f  x  :=
  rfl

example  :  (range  n).sum  f  =  ∑  x  ∈  range  n,  f  x  :=
  rfl

example  :  (range  n).prod  f  =  ∏  x  ∈  range  n,  f  x  :=
  rfl 
```

事实 `Finset.sum_range_zero` 和 `Finset.sum_range_succ` 为 $n$ 的求和提供了一个递归描述，对于积也是如此。

```py
example  (f  :  ℕ  →  ℕ)  :  ∑  x  ∈  range  0,  f  x  =  0  :=
  Finset.sum_range_zero  f

example  (f  :  ℕ  →  ℕ)  (n  :  ℕ)  :  ∑  x  ∈  range  n.succ,  f  x  =  ∑  x  ∈  range  n,  f  x  +  f  n  :=
  Finset.sum_range_succ  f  n

example  (f  :  ℕ  →  ℕ)  :  ∏  x  ∈  range  0,  f  x  =  1  :=
  Finset.prod_range_zero  f

example  (f  :  ℕ  →  ℕ)  (n  :  ℕ)  :  ∏  x  ∈  range  n.succ,  f  x  =  (∏  x  ∈  range  n,  f  x)  *  f  n  :=
  Finset.prod_range_succ  f  n 
```

每对中的第一个恒等式是定义性的，也就是说，你可以用 `rfl` 替换证明。

以下表示我们定义的阶乘函数作为积。

```py
example  (n  :  ℕ)  :  fac  n  =  ∏  i  ∈  range  n,  (i  +  1)  :=  by
  induction'  n  with  n  ih
  ·  simp  [fac,  prod_range_zero]
  simp  [fac,  ih,  prod_range_succ,  mul_comm] 
```

我们将 `mul_comm` 作为简化规则包括在内的事实值得注意。使用恒等式 `x * y = y * x` 进行简化似乎很危险，因为这通常会无限循环。Lean 的简化器足够智能，能够识别这一点，并且仅在结果项在某些固定但任意的项的排序中具有更小值的情况下应用该规则。以下示例表明，使用 `mul_assoc`、`mul_comm` 和 `mul_left_comm` 这三个规则进行简化能够识别出相同但括号位置和变量排序不同的积。

```py
example  (a  b  c  d  e  f  :  ℕ)  :  a  *  (b  *  c  *  f  *  (d  *  e))  =  d  *  (a  *  f  *  e)  *  (c  *  b)  :=  by
  simp  [mul_assoc,  mul_comm,  mul_left_comm] 
```

大概来说，这些规则通过将括号推向右边，然后重新排列两侧的表达式，直到它们都遵循相同的规范顺序来工作。使用这些规则简化，以及相应的加法规则，是一个实用的技巧。

回到求和恒等式，我们建议逐步通过以下证明，即自然数从 1 加到$n$（包括$n$）的和是$n (n + 1) / 2$。证明的第一步消除了分母。这在形式化恒等式时通常很有用，因为涉及除法的计算通常有附加条件。（同样，在可能的情况下避免在自然数上使用减法也是有用的。）

```py
theorem  sum_id  (n  :  ℕ)  :  ∑  i  ∈  range  (n  +  1),  i  =  n  *  (n  +  1)  /  2  :=  by
  symm;  apply  Nat.div_eq_of_eq_mul_right  (by  norm_num  :  0  <  2)
  induction'  n  with  n  ih
  ·  simp
  rw  [Finset.sum_range_succ,  mul_add  2,  ←  ih]
  ring 
```

我们鼓励你证明平方和的类似恒等式，以及你在网上能找到的其他恒等式。

```py
theorem  sum_sqr  (n  :  ℕ)  :  ∑  i  ∈  range  (n  +  1),  i  ^  2  =  n  *  (n  +  1)  *  (2  *  n  +  1)  /  6  :=  by
  sorry 
```

在 Lean 的核心库中，加法和乘法本身是通过递归定义的，它们的基本性质是通过归纳法建立的。如果你喜欢思考像那样的基础主题，你可能喜欢通过乘法和加法的交换律和结合律以及乘法对加法的分配律的证明来工作。你可以按照以下概述在自然数的副本上这样做。注意，我们可以使用`induction`策略与`MyNat`；Lean 足够智能，知道要使用相关的归纳原理（当然，这与`Nat`的相同）。

我们从加法的交换律开始。一个很好的经验法则是，由于加法和乘法是通过第二个参数的递归定义的，因此通常在出现在该位置的变量上通过归纳法进行证明是有利的。在证明结合律时决定使用哪个变量有点棘手。

没有使用零、一、加法和乘法的常规符号来写东西可能会让人困惑。我们将在以后学习如何定义这样的符号。在`MyNat`命名空间中工作意味着我们可以写`zero`和`succ`而不是`MyNat.zero`和`MyNat.succ`，并且这些名称的解释优先于其他解释。在命名空间外部，下面定义的`add`的完整名称是`MyNat.add`。

如果你发现你真的喜欢这类东西，尝试定义截断减法和指数运算，并证明它们的一些性质。记住，截断减法在零处截断。为了定义这一点，定义一个前驱函数`pred`，它从任何非零数中减去一，并固定零。函数`pred`可以通过一个简单的递归实例来定义。

```py
inductive  MyNat  where
  |  zero  :  MyNat
  |  succ  :  MyNat  →  MyNat

namespace  MyNat

def  add  :  MyNat  →  MyNat  →  MyNat
  |  x,  zero  =>  x
  |  x,  succ  y  =>  succ  (add  x  y)

def  mul  :  MyNat  →  MyNat  →  MyNat
  |  x,  zero  =>  zero
  |  x,  succ  y  =>  add  (mul  x  y)  x

theorem  zero_add  (n  :  MyNat)  :  add  zero  n  =  n  :=  by
  induction'  n  with  n  ih
  ·  rfl
  rw  [add,  ih]

theorem  succ_add  (m  n  :  MyNat)  :  add  (succ  m)  n  =  succ  (add  m  n)  :=  by
  induction'  n  with  n  ih
  ·  rfl
  rw  [add,  ih]
  rfl

theorem  add_comm  (m  n  :  MyNat)  :  add  m  n  =  add  n  m  :=  by
  induction'  n  with  n  ih
  ·  rw  [zero_add]
  rfl
  rw  [add,  succ_add,  ih]

theorem  add_assoc  (m  n  k  :  MyNat)  :  add  (add  m  n)  k  =  add  m  (add  n  k)  :=  by
  sorry
theorem  mul_add  (m  n  k  :  MyNat)  :  mul  m  (add  n  k)  =  add  (mul  m  n)  (mul  m  k)  :=  by
  sorry
theorem  zero_mul  (n  :  MyNat)  :  mul  zero  n  =  zero  :=  by
  sorry
theorem  succ_mul  (m  n  :  MyNat)  :  mul  (succ  m)  n  =  add  (mul  m  n)  n  :=  by
  sorry
theorem  mul_comm  (m  n  :  MyNat)  :  mul  m  n  =  mul  n  m  :=  by
  sorry
end  MyNat 
```  ## 5.3\. 无限多个素数

让我们继续用另一个数学标准来探索归纳和递归：证明存在无限多个质数。一种表述方式是，对于每一个自然数 $n$，都存在一个大于 $n$ 的质数。为了证明这一点，设 $p$ 是 $n! + 1$ 的任意一个质因数。如果 $p$ 小于或等于 $n$，它就会整除 $n!$。由于它也整除 $n! + 1$，它就会整除 1，这是矛盾的。因此 $p$ 必须大于 $n$。

为了使这个证明形式化，我们需要证明任何大于或等于 2 的数都有一个质因数。要做到这一点，我们需要证明任何不等于 0 或 1 的自然数都大于或等于 2。这引出了形式化中的一个奇特特性：这类简单陈述往往是最令人烦恼的。在这里，我们考虑了几种实现方法。

首先，我们可以使用 `cases` 策略和后继函数尊重自然数上的顺序这一事实。

```py
theorem  two_le  {m  :  ℕ}  (h0  :  m  ≠  0)  (h1  :  m  ≠  1)  :  2  ≤  m  :=  by
  cases  m;  contradiction
  case  succ  m  =>
  cases  m;  contradiction
  repeat  apply  Nat.succ_le_succ
  apply  zero_le 
```

另一种策略是使用 `interval_cases` 策略，它会自动将目标分解为当变量位于自然数或整数的区间内时的几种情况。记住，你可以悬停在它上面以查看其文档。

```py
example  {m  :  ℕ}  (h0  :  m  ≠  0)  (h1  :  m  ≠  1)  :  2  ≤  m  :=  by
  by_contra  h
  push_neg  at  h
  interval_cases  m  <;>  contradiction 
```

回想一下，`interval_cases m` 后面的分号意味着下一个策略将应用于它生成的每个情况。另一个选项是使用 `decide` 策略，它试图找到一个决策过程来解决问题。Lean 知道你可以通过决定每个有限实例来决定以有界量词 `∀ x, x < n → ...` 或 `∃ x, x < n ∧ ...` 开头的语句的真值。

```py
example  {m  :  ℕ}  (h0  :  m  ≠  0)  (h1  :  m  ≠  1)  :  2  ≤  m  :=  by
  by_contra  h
  push_neg  at  h
  revert  h0  h1
  revert  h  m
  decide 
```

拥有定理 `two_le` 后，让我们首先证明每个大于两的自然数都有一个质数因子。Mathlib 包含一个函数 `Nat.minFac`，它返回最小的质数因子，但为了学习库的新部分，我们将避免使用它，并直接证明这个定理。

在这里，普通的归纳不足以解决问题。我们想要使用 **强归纳**，这允许我们通过证明对于每个数 $n$，如果 $P$ 对所有小于 $n$ 的值成立，那么它对 $n$ 也成立，来证明每个自然数 $n$ 都有一个性质 $P$。在 Lean 中，这个原则被称为 `Nat.strong_induction_on`，我们可以使用 `using` 关键字告诉归纳策略使用它。注意，当我们这样做时，没有基础情况；它被归纳步骤所包含。

论证如下。假设 $n ≥ 2$，如果 $n$ 是质数，那么我们就完成了。如果不是，那么根据质数的定义之一，它有一个非平凡因子 $m$，我们可以对它应用归纳假设。通过查看下一个证明，我们可以看到这是如何实现的。

```py
theorem  exists_prime_factor  {n  :  Nat}  (h  :  2  ≤  n)  :  ∃  p  :  Nat,  p.Prime  ∧  p  ∣  n  :=  by
  by_cases  np  :  n.Prime
  ·  use  n,  np
  induction'  n  using  Nat.strong_induction_on  with  n  ih
  rw  [Nat.prime_def_lt]  at  np
  push_neg  at  np
  rcases  np  h  with  ⟨m,  mltn,  mdvdn,  mne1⟩
  have  :  m  ≠  0  :=  by
  intro  mz
  rw  [mz,  zero_dvd_iff]  at  mdvdn
  linarith
  have  mgt2  :  2  ≤  m  :=  two_le  this  mne1
  by_cases  mp  :  m.Prime
  ·  use  m,  mp
  ·  rcases  ih  m  mltn  mgt2  mp  with  ⟨p,  pp,  pdvd⟩
  use  p,  pp
  apply  pdvd.trans  mdvdn 
```

我们现在可以证明我们定理的以下表述。看看你是否可以完成这个草图。你可以使用 `Nat.factorial_pos`、`Nat.dvd_factorial` 和 `Nat.dvd_sub'`。

```py
theorem  primes_infinite  :  ∀  n,  ∃  p  >  n,  Nat.Prime  p  :=  by
  intro  n
  have  :  2  ≤  Nat.factorial  n  +  1  :=  by
  sorry
  rcases  exists_prime_factor  this  with  ⟨p,  pp,  pdvd⟩
  refine  ⟨p,  ?_,  pp⟩
  show  p  >  n
  by_contra  ple
  push_neg  at  ple
  have  :  p  ∣  Nat.factorial  n  :=  by
  sorry
  have  :  p  ∣  1  :=  by
  sorry
  show  False
  sorry 
```

让我们考虑上述证明的一个变体，其中我们不是使用阶乘函数，而是假设我们被一个有限集合 $\{ p_1, \ldots, p_n \}$ 给定，并考虑 $\prod_{i = 1}^n p_i + 1$ 的一个素数因子。这个素数因子必须与每个 $p_i$ 不同，这表明不存在包含所有素数的有限集合。

将这个论证形式化需要我们对有限集进行推理。在 Lean 中，对于任何类型 `α`，类型 `Finset α` 表示类型 `α` 的元素构成的有限集。对有限集进行计算推理需要有一个测试 `α` 上等性的过程，这就是为什么下面的代码片段包括假设 `[DecidableEq α]`。对于像 `ℕ`、`ℤ` 和 `ℚ` 这样的具体数据类型，这个假设会自动满足。当推理实数时，可以使用经典逻辑并放弃计算解释来满足这个假设。

我们使用命令 `open Finset` 来使用相关定理的较短名称。与集合的情况不同，大多数涉及有限集的等价关系在定义上并不成立，因此需要手动使用等价关系如 `Finset.subset_iff`、`Finset.mem_union`、`Finset.mem_inter` 和 `Finset.mem_sdiff` 来展开。`ext` 策略仍然可以用来通过显示一个有限集的每个元素都是另一个集合的元素来证明两个有限集相等。

```py
open  Finset

section
variable  {α  :  Type*}  [DecidableEq  α]  (r  s  t  :  Finset  α)

example  :  r  ∩  (s  ∪  t)  ⊆  r  ∩  s  ∪  r  ∩  t  :=  by
  rw  [subset_iff]
  intro  x
  rw  [mem_inter,  mem_union,  mem_union,  mem_inter,  mem_inter]
  tauto

example  :  r  ∩  (s  ∪  t)  ⊆  r  ∩  s  ∪  r  ∩  t  :=  by
  simp  [subset_iff]
  intro  x
  tauto

example  :  r  ∩  s  ∪  r  ∩  t  ⊆  r  ∩  (s  ∪  t)  :=  by
  simp  [subset_iff]
  intro  x
  tauto

example  :  r  ∩  s  ∪  r  ∩  t  =  r  ∩  (s  ∪  t)  :=  by
  ext  x
  simp
  tauto

end 
```

我们使用了一个新的技巧：`tauto` 策略（以及一个增强版本 `tauto!`，它使用经典逻辑）可以用来省略命题恒真式。看看你是否可以使用这些方法来证明下面的两个例子。

```py
example  :  (r  ∪  s)  ∩  (r  ∪  t)  =  r  ∪  s  ∩  t  :=  by
  sorry
example  :  (r  \  s)  \  t  =  r  \  (s  ∪  t)  :=  by
  sorry 
```

定理 `Finset.dvd_prod_of_mem` 告诉我们，如果 `n` 是有限集合 `s` 的一个元素，那么 `n` 能整除 `∏ i ∈ s, i`。

```py
example  (s  :  Finset  ℕ)  (n  :  ℕ)  (h  :  n  ∈  s)  :  n  ∣  ∏  i  ∈  s,  i  :=
  Finset.dvd_prod_of_mem  _  h 
```

我们还需要知道，在 `n` 是素数且 `s` 是素数集合的情况下，逆命题也成立。为了证明这一点，我们需要以下引理，你应该能够使用定理 `Nat.Prime.eq_one_or_self_of_dvd` 来证明。

```py
theorem  _root_.Nat.Prime.eq_of_dvd_of_prime  {p  q  :  ℕ}
  (prime_p  :  Nat.Prime  p)  (prime_q  :  Nat.Prime  q)  (h  :  p  ∣  q)  :
  p  =  q  :=  by
  sorry 
```

我们可以使用这个引理来证明如果一个素数 `p` 整除有限个素数的乘积，那么它等于其中的一个。Mathlib 提供了一个有用的有限集归纳原理：为了证明一个性质对任意有限集 `s` 成立，需要证明它对空集成立，并且在我们添加一个新元素 `a ∉ s` 时它被保留。这个原理被称为 `Finset.induction_on`。当我们告诉归纳策略使用它时，我们还可以指定 `a` 和 `s` 的名称，以及在归纳步骤中 `a ∉ s` 的假设的名称，以及归纳假设的名称。表达式 `Finset.insert a s` 表示 `s` 与单元素集合 `a` 的并集。`Finset.prod_empty` 和 `Finset.prod_insert` 然后提供了乘积的相关重写规则。在下面的证明中，第一个 `simp` 应用了 `Finset.prod_empty`。逐步通过证明的开始，看看归纳展开，然后完成它。

```py
theorem  mem_of_dvd_prod_primes  {s  :  Finset  ℕ}  {p  :  ℕ}  (prime_p  :  p.Prime)  :
  (∀  n  ∈  s,  Nat.Prime  n)  →  (p  ∣  ∏  n  ∈  s,  n)  →  p  ∈  s  :=  by
  intro  h₀  h₁
  induction'  s  using  Finset.induction_on  with  a  s  ans  ih
  ·  simp  at  h₁
  linarith  [prime_p.two_le]
  simp  [Finset.prod_insert  ans,  prime_p.dvd_mul]  at  h₀  h₁
  rw  [mem_insert]
  sorry 
```

我们需要有限集的一个最后属性。给定一个元素 `s : Set α` 和一个在 `α` 上的谓词 `P`，在 第四章 中，我们用 `{ x ∈ s | P x }` 表示满足 `P` 的 `s` 的元素集合。给定 `s : Finset α`，类似的概念表示为 `s.filter P`。

```py
example  (s  :  Finset  ℕ)  (x  :  ℕ)  :  x  ∈  s.filter  Nat.Prime  ↔  x  ∈  s  ∧  x.Prime  :=
  mem_filter 
```

现在我们证明关于存在无限多个素数的陈述的另一种表述，即给定任何 `s : Finset ℕ`，存在一个素数 `p`，它不是 `s` 的元素。为了达到矛盾，我们假设所有素数都在 `s` 中，然后缩减到一个只包含所有素数的集合 `s'`。计算该集合的乘积，加一，并找到结果的素数因子，将导致我们寻找的矛盾。看看你是否能完成下面的草图。你可以在第一个 `have` 的证明中使用 `Finset.prod_pos`。 

```py
theorem  primes_infinite'  :  ∀  s  :  Finset  Nat,  ∃  p,  Nat.Prime  p  ∧  p  ∉  s  :=  by
  intro  s
  by_contra  h
  push_neg  at  h
  set  s'  :=  s.filter  Nat.Prime  with  s'_def
  have  mem_s'  :  ∀  {n  :  ℕ},  n  ∈  s'  ↔  n.Prime  :=  by
  intro  n
  simp  [s'_def]
  apply  h
  have  :  2  ≤  (∏  i  ∈  s',  i)  +  1  :=  by
  sorry
  rcases  exists_prime_factor  this  with  ⟨p,  pp,  pdvd⟩
  have  :  p  ∣  ∏  i  ∈  s',  i  :=  by
  sorry
  have  :  p  ∣  1  :=  by
  convert  Nat.dvd_sub  pdvd  this
  simp
  show  False
  sorry 
```

因此，我们已经看到了两种表达存在无限多个素数的方法：说它们不受任何 `n` 的限制，以及说它们不包含在任何有限集 `s` 中。下面的两个证明表明这些表述是等价的。在第二个证明中，为了形成 `s.filter Q`，我们必须假设存在一个判断 `Q` 是否成立的程序。Lean 知道存在一个用于 `Nat.Prime` 的程序。一般来说，如果我们通过编写 `open Classical` 使用经典逻辑，我们可以省去这个假设。

在 Mathlib 中，`Finset.sup s f` 表示 `f x` 的值在 `x` 在 `s` 上取值时的上确界，当 `s` 为空且 `f` 的陪域为 `ℕ` 时返回 `0`。在第一个证明中，我们使用 `s.sup id`，其中 `id` 是恒等函数，来指代 `s` 中的最大值。

```py
theorem  bounded_of_ex_finset  (Q  :  ℕ  →  Prop)  :
  (∃  s  :  Finset  ℕ,  ∀  k,  Q  k  →  k  ∈  s)  →  ∃  n,  ∀  k,  Q  k  →  k  <  n  :=  by
  rintro  ⟨s,  hs⟩
  use  s.sup  id  +  1
  intro  k  Qk
  apply  Nat.lt_succ_of_le
  show  id  k  ≤  s.sup  id
  apply  le_sup  (hs  k  Qk)

theorem  ex_finset_of_bounded  (Q  :  ℕ  →  Prop)  [DecidablePred  Q]  :
  (∃  n,  ∀  k,  Q  k  →  k  ≤  n)  →  ∃  s  :  Finset  ℕ,  ∀  k,  Q  k  ↔  k  ∈  s  :=  by
  rintro  ⟨n,  hn⟩
  use  (range  (n  +  1)).filter  Q
  intro  k
  simp  [Nat.lt_succ_iff]
  exact  hn  k 
```

对我们第二个证明无限多个素数的微小变化表明，存在无限多个模 4 同余于 3 的素数。论证如下。首先，注意如果两个数 $m$ 和 $n$ 的乘积模 4 等于 3，那么这两个数中至少有一个模 4 同余于 3。毕竟，它们都必须是奇数，如果它们都模 4 同余于 1，那么它们的乘积也是。我们可以利用这个观察结果来证明如果某个大于 2 的数模 4 同余于 3，那么这个数有一个模 4 同余于 3 的素数约数。

现在假设只有有限多个模 4 同余于 3 的素数，比如说，$p_1, \ldots, p_k$。不失一般性，我们可以假设 $p_1 = 3$。考虑乘积 $4 \prod_{i = 2}^k p_i + 3$。很容易看出这个数模 4 同余于 3，所以它有一个模 4 同余于 3 的素数因子 $p$。不可能 $p = 3$；因为 $p$ 能整除 $4 \prod_{i = 2}^k p_i + 3$，如果 $p$ 等于 3，那么它也会整除 $\prod_{i = 2}^k p_i$，这意味着 $p$ 等于 $p_i$ 中的一个，对于 $i = 2, \ldots, k$；但我们已经排除了 3。所以 $p$ 必须是其他元素 $p_i$ 中的一个。但在那种情况下，$p$ 能整除 $4 \prod_{i = 2}^k p_i$ 和 3，这与它不是 3 的事实相矛盾。

在 Lean 中，表示法 `n % m` 读作“`n` 模 `m`”，表示 `n` 除以 `m` 的余数。

```py
example  :  27  %  4  =  3  :=  by  norm_num 
```

然后，我们可以将陈述“`n` 模 4 同余于 3”表示为 `n % 4 = 3`。以下示例和定理总结了我们将需要使用的事实。第一个命名的定理是通过对少数几个情况进行推理的另一个例子。在第二个命名的定理中，记住分号意味着后续的策略块应用于由前面的策略块创建的所有目标。

```py
example  (n  :  ℕ)  :  (4  *  n  +  3)  %  4  =  3  :=  by
  rw  [add_comm,  Nat.add_mul_mod_self_left]

theorem  mod_4_eq_3_or_mod_4_eq_3  {m  n  :  ℕ}  (h  :  m  *  n  %  4  =  3)  :  m  %  4  =  3  ∨  n  %  4  =  3  :=  by
  revert  h
  rw  [Nat.mul_mod]
  have  :  m  %  4  <  4  :=  Nat.mod_lt  m  (by  norm_num)
  interval_cases  m  %  4  <;>  simp  [-Nat.mul_mod_mod]
  have  :  n  %  4  <  4  :=  Nat.mod_lt  n  (by  norm_num)
  interval_cases  n  %  4  <;>  simp

theorem  two_le_of_mod_4_eq_3  {n  :  ℕ}  (h  :  n  %  4  =  3)  :  2  ≤  n  :=  by
  apply  two_le  <;>
  ·  intro  neq
  rw  [neq]  at  h
  norm_num  at  h 
```

我们还需要以下事实，即如果 `m` 是 `n` 的非平凡约数，那么 `n / m` 也是。看看你是否能使用 `Nat.div_dvd_of_dvd` 和 `Nat.div_lt_self` 完成证明。

```py
theorem  aux  {m  n  :  ℕ}  (h₀  :  m  ∣  n)  (h₁  :  2  ≤  m)  (h₂  :  m  <  n)  :  n  /  m  ∣  n  ∧  n  /  m  <  n  :=  by
  sorry 
```

现在将所有这些部分组合起来，证明任何模 4 同余于 3 的数都有一个具有相同性质的素数约数。

```py
theorem  exists_prime_factor_mod_4_eq_3  {n  :  Nat}  (h  :  n  %  4  =  3)  :
  ∃  p  :  Nat,  p.Prime  ∧  p  ∣  n  ∧  p  %  4  =  3  :=  by
  by_cases  np  :  n.Prime
  ·  use  n
  induction'  n  using  Nat.strong_induction_on  with  n  ih
  rw  [Nat.prime_def_lt]  at  np
  push_neg  at  np
  rcases  np  (two_le_of_mod_4_eq_3  h)  with  ⟨m,  mltn,  mdvdn,  mne1⟩
  have  mge2  :  2  ≤  m  :=  by
  apply  two_le  _  mne1
  intro  mz
  rw  [mz,  zero_dvd_iff]  at  mdvdn
  linarith
  have  neq  :  m  *  (n  /  m)  =  n  :=  Nat.mul_div_cancel'  mdvdn
  have  :  m  %  4  =  3  ∨  n  /  m  %  4  =  3  :=  by
  apply  mod_4_eq_3_or_mod_4_eq_3
  rw  [neq,  h]
  rcases  this  with  h1  |  h1
  .  sorry
  .  sorry 
```

我们已经接近终点。给定一个素数集合 `s`，如果其中包含 3，我们需要讨论从该集合中移除 3 的结果。函数 `Finset.erase` 处理这个问题。

```py
example  (m  n  :  ℕ)  (s  :  Finset  ℕ)  (h  :  m  ∈  erase  s  n)  :  m  ≠  n  ∧  m  ∈  s  :=  by
  rwa  [mem_erase]  at  h

example  (m  n  :  ℕ)  (s  :  Finset  ℕ)  (h  :  m  ∈  erase  s  n)  :  m  ≠  n  ∧  m  ∈  s  :=  by
  simp  at  h
  assumption 
```

我们现在准备证明存在无限多个模 4 同余于 3 的素数。在下面的空白处填写缺失的部分。我们的解决方案在过程中使用了 `Nat.dvd_add_iff_left` 和 `Nat.dvd_sub'`。

```py
theorem  primes_mod_4_eq_3_infinite  :  ∀  n,  ∃  p  >  n,  Nat.Prime  p  ∧  p  %  4  =  3  :=  by
  by_contra  h
  push_neg  at  h
  rcases  h  with  ⟨n,  hn⟩
  have  :  ∃  s  :  Finset  Nat,  ∀  p  :  ℕ,  p.Prime  ∧  p  %  4  =  3  ↔  p  ∈  s  :=  by
  apply  ex_finset_of_bounded
  use  n
  contrapose!  hn
  rcases  hn  with  ⟨p,  ⟨pp,  p4⟩,  pltn⟩
  exact  ⟨p,  pltn,  pp,  p4⟩
  rcases  this  with  ⟨s,  hs⟩
  have  h₁  :  ((4  *  ∏  i  ∈  erase  s  3,  i)  +  3)  %  4  =  3  :=  by
  sorry
  rcases  exists_prime_factor_mod_4_eq_3  h₁  with  ⟨p,  pp,  pdvd,  p4eq⟩
  have  ps  :  p  ∈  s  :=  by
  sorry
  have  pne3  :  p  ≠  3  :=  by
  sorry
  have  :  p  ∣  4  *  ∏  i  ∈  erase  s  3,  i  :=  by
  sorry
  have  :  p  ∣  3  :=  by
  sorry
  have  :  p  =  3  :=  by
  sorry
  contradiction 
```

如果你设法完成了证明，恭喜你！这已经是一项正式化的重大成就。## 5.4. 更多归纳

在 第 5.2 节 中，我们看到了如何通过自然数的递归定义阶乘函数。

```py
def  fac  :  ℕ  →  ℕ
  |  0  =>  1
  |  n  +  1  =>  (n  +  1)  *  fac  n 
```

我们还看到了如何使用 `induction'` 策略证明定理。

```py
theorem  fac_pos  (n  :  ℕ)  :  0  <  fac  n  :=  by
  induction'  n  with  n  ih
  ·  rw  [fac]
  exact  zero_lt_one
  rw  [fac]
  exact  mul_pos  n.succ_pos  ih 
```

`induction` 策略（不带撇号）允许更结构化的语法。

```py
example  (n  :  ℕ)  :  0  <  fac  n  :=  by
  induction  n
  case  zero  =>
  rw  [fac]
  exact  zero_lt_one
  case  succ  n  ih  =>
  rw  [fac]
  exact  mul_pos  n.succ_pos  ih

example  (n  :  ℕ)  :  0  <  fac  n  :=  by
  induction  n  with
  |  zero  =>
  rw  [fac]
  exact  zero_lt_one
  |  succ  n  ih  =>
  rw  [fac]
  exact  mul_pos  n.succ_pos  ih 
```

如同往常，你可以悬停在 `induction` 关键字上以阅读文档。案例的名称 `zero` 和 `succ` 来自于类型 ℕ 的定义。注意，`succ` 案例允许你为归纳变量和归纳假设选择任何你想要的名称，这里为 `n` 和 `ih`。你甚至可以使用定义递归函数的相同符号来证明定理。

```py
theorem  fac_pos'  :  ∀  n,  0  <  fac  n
  |  0  =>  by
  rw  [fac]
  exact  zero_lt_one
  |  n  +  1  =>  by
  rw  [fac]
  exact  mul_pos  n.succ_pos  (fac_pos'  n) 
```

注意到没有 `:=`，冒号后面的 `∀ n`，每个情况中的 `by` 关键字，以及归纳调用 `fac_pos' n`。这就像定理是 `n` 的递归函数，在归纳步骤中我们进行递归调用。

这种定义风格非常灵活。Lean 的设计者内置了定义递归函数的复杂手段，这些手段扩展到了通过归纳进行证明。例如，我们可以使用多个基本情况来定义斐波那契函数。

```py
@[simp]  def  fib  :  ℕ  →  ℕ
  |  0  =>  0
  |  1  =>  1
  |  n  +  2  =>  fib  n  +  fib  (n  +  1) 
```

`@[simp]` 注解表示简化器将使用定义方程。你也可以通过编写 `rw [fib]` 来应用它们。下面将有助于为 `n + 2` 的情况命名。

```py
theorem  fib_add_two  (n  :  ℕ)  :  fib  (n  +  2)  =  fib  n  +  fib  (n  +  1)  :=  rfl

example  (n  :  ℕ)  :  fib  (n  +  2)  =  fib  n  +  fib  (n  +  1)  :=  by  rw  [fib] 
```

使用 Lean 的递归函数表示法，你可以通过归纳自然数来执行与 `fib` 的递归定义相对应的证明。以下示例提供了第 n 个斐波那契数的显式公式，该公式以黄金分割数 `φ` 及其共轭 `φ'` 为基础。我们必须告诉 Lean，我们预期我们的定义不会生成代码，因为实数上的算术运算是不可计算的。

```py
noncomputable  section

def  phi  :  ℝ  :=  (1  +  √5)  /  2
def  phi'  :  ℝ  :=  (1  -  √5)  /  2

theorem  phi_sq  :  phi²  =  phi  +  1  :=  by
  field_simp  [phi,  add_sq];  ring

theorem  phi'_sq  :  phi'²  =  phi'  +  1  :=  by
  field_simp  [phi',  sub_sq];  ring

theorem  fib_eq  :  ∀  n,  fib  n  =  (phi^n  -  phi'^n)  /  √5
  |  0  =>  by  simp
  |  1  =>  by  field_simp  [phi,  phi']
  |  n+2  =>  by  field_simp  [fib_eq,  pow_add,  phi_sq,  phi'_sq];  ring

end 
```

涉及斐波那契函数的归纳证明不必是那种形式。以下我们重现了 `Mathlib` 证明连续斐波那契数是互质的。

```py
theorem  fib_coprime_fib_succ  (n  :  ℕ)  :  Nat.Coprime  (fib  n)  (fib  (n  +  1))  :=  by
  induction  n  with
  |  zero  =>  simp
  |  succ  n  ih  =>
  simp  only  [fib,  Nat.coprime_add_self_right]
  exact  ih.symm 
```

使用 Lean 的计算解释，我们可以评估斐波那契数。

```py
#eval  fib  6
#eval  List.range  20  |>.map  fib 
```

`fib` 的直接实现计算效率低下。实际上，它的运行时间与其参数呈指数关系。（你应该思考一下为什么。）在 Lean 中，我们可以实现以下尾递归版本，其运行时间与 `n` 线性相关，并证明它计算的是相同的函数。

```py
def  fib'  (n  :  Nat)  :  Nat  :=
  aux  n  0  1
where  aux
  |  0,  x,  _  =>  x
  |  n+1,  x,  y  =>  aux  n  y  (x  +  y)

theorem  fib'.aux_eq  (m  n  :  ℕ)  :  fib'.aux  n  (fib  m)  (fib  (m  +  1))  =  fib  (n  +  m)  :=  by
  induction  n  generalizing  m  with
  |  zero  =>  simp  [fib'.aux]
  |  succ  n  ih  =>  rw  [fib'.aux,  ←fib_add_two,  ih,  add_assoc,  add_comm  1]

theorem  fib'_eq_fib  :  fib'  =  fib  :=  by
  ext  n
  erw  [fib',  fib'.aux_eq  0  n];  rfl

#eval  fib'  10000 
```

注意 `fib'.aux_eq` 证明中的 `generalizing` 关键字。它用于在归纳假设前插入 `∀ m`，这样在归纳步骤中，`m` 可以取不同的值。你可以逐步检查证明，并确认在这种情况下，量词需要在归纳步骤中实例化为 `m + 1`。

注意也使用了 `erw`（表示“扩展重写”）而不是 `rw`。这是因为为了重写目标 `fib'.aux_eq`，`fib 0` 和 `fib 1` 必须分别简化为 `0` 和 `1`。与 `rw` 相比，`erw` 在展开定义以匹配参数方面更为激进。这并不总是好主意；在某些情况下，它可能会浪费大量时间，因此请谨慎使用 `erw`。

这里是 `generalizing` 关键字在另一个在 `Mathlib` 中找到的恒等式证明中使用的另一个例子。该恒等式的非正式证明可以在[这里](https://proofwiki.org/wiki/Fibonacci_Number_in_terms_of_Smaller_Fibonacci_Numbers)找到。我们提供了两种正式证明的变体。

```py
theorem  fib_add  (m  n  :  ℕ)  :  fib  (m  +  n  +  1)  =  fib  m  *  fib  n  +  fib  (m  +  1)  *  fib  (n  +  1)  :=  by
  induction  n  generalizing  m  with
  |  zero  =>  simp
  |  succ  n  ih  =>
  specialize  ih  (m  +  1)
  rw  [add_assoc  m  1  n,  add_comm  1  n]  at  ih
  simp  only  [fib_add_two,  Nat.succ_eq_add_one,  ih]
  ring

theorem  fib_add'  :  ∀  m  n,  fib  (m  +  n  +  1)  =  fib  m  *  fib  n  +  fib  (m  +  1)  *  fib  (n  +  1)
  |  _,  0  =>  by  simp
  |  m,  n  +  1  =>  by
  have  :=  fib_add'  (m  +  1)  n
  rw  [add_assoc  m  1  n,  add_comm  1  n]  at  this
  simp  only  [fib_add_two,  Nat.succ_eq_add_one,  this]
  ring 
```

作为练习，使用 `fib_add` 证明以下内容。

```py
example  (n  :  ℕ):  (fib  n)  ^  2  +  (fib  (n  +  1))  ^  2  =  fib  (2  *  n  +  1)  :=  by  sorry 
```

Lean 定义递归函数的机制足够灵活，允许任意递归调用，只要参数的复杂度根据某个良基度量递减。在下一个例子中，我们展示了每个自然数 `n ≠ 1` 都有一个素数因子，利用了如果 `n` 非零且不是素数，它有一个更小的因子的这一事实。（你可以检查 Mathlib 在 `Nat` 命名空间中有一个同名的定理，尽管它的证明与这里给出的不同。）

```py
#check  (@Nat.not_prime_iff_exists_dvd_lt  :
  ∀  {n  :  ℕ},  2  ≤  n  →  (¬Nat.Prime  n  ↔  ∃  m,  m  ∣  n  ∧  2  ≤  m  ∧  m  <  n))

theorem  ne_one_iff_exists_prime_dvd  :  ∀  {n},  n  ≠  1  ↔  ∃  p  :  ℕ,  p.Prime  ∧  p  ∣  n
  |  0  =>  by  simpa  using  Exists.intro  2  Nat.prime_two
  |  1  =>  by  simp  [Nat.not_prime_one]
  |  n  +  2  =>  by
  have  hn  :  n+2  ≠  1  :=  by  omega
  simp  only  [Ne,  not_false_iff,  true_iff,  hn]
  by_cases  h  :  Nat.Prime  (n  +  2)
  ·  use  n+2,  h
  ·  have  :  2  ≤  n  +  2  :=  by  omega
  rw  [Nat.not_prime_iff_exists_dvd_lt  this]  at  h
  rcases  h  with  ⟨m,  mdvdn,  mge2,  -⟩
  have  :  m  ≠  1  :=  by  omega
  rw  [ne_one_iff_exists_prime_dvd]  at  this
  rcases  this  with  ⟨p,  primep,  pdvdm⟩
  use  p,  primep
  exact  pdvdm.trans  mdvdn 
```

这行代码 `rw [ne_one_iff_exists_prime_dvd] at this` 就像是一个魔术：我们正在使用我们正在证明的定理本身来证明它。使其工作的是归纳调用在 `m` 上实例化，当前情况是 `n + 2`，并且上下文有 `m < n + 2`。Lean 可以找到假设并使用它来证明归纳是良基的。Lean 在找出什么是递减的方面相当出色；在这种情况下，定理陈述中的 `n` 的选择和小于关系是显而易见的。在更复杂的情况下，Lean 提供了提供此信息的机制。请参阅 Lean 参考手册中关于[良基递归](https://lean-lang.org/doc/reference/latest//Definitions/Recursive-Definitions/#well-founded-recursion)的部分。

有时，在证明中，你可能需要根据自然数 `n` 是零还是后继来分情况讨论，而不需要在后继情况下使用归纳假设。为此，你可以使用 `cases` 和 `rcases` 策略。

```py
theorem  zero_lt_of_mul_eq_one  (m  n  :  ℕ)  :  n  *  m  =  1  →  0  <  n  ∧  0  <  m  :=  by
  cases  n  <;>  cases  m  <;>  simp

example  (m  n  :  ℕ)  :  n*m  =  1  →  0  <  n  ∧  0  <  m  :=  by
  rcases  m  with  (_  |  m);  simp
  rcases  n  with  (_  |  n)  <;>  simp 
```

这是一个有用的技巧。通常，你有一个关于自然数 `n` 的定理，其中零的情况很容易处理。如果你对 `n` 进行情况分析并快速处理零的情况，你将剩下用 `n + 1` 替换后的原始目标。

让我们从古希腊人已知的一个事实开始，即 2 的平方根是无理数。如果我们假设相反，我们可以将 $\sqrt{2} = a / b$ 写成最简分数。平方两边得到 $a² = 2 b²$，这意味着 $a$ 是偶数。如果我们写成 $a = 2c$，那么我们得到 $4c² = 2 b²$，从而 $b² = 2 c²$。这表明 $b$ 也是偶数，这与我们假设的 $a / b$ 已经被化简到最简形式的事实相矛盾。

说$a / b$是最简分数意味着$a$和$b$没有公共因子，也就是说，它们是*互质的*。Mathlib 定义谓词`Nat.Coprime m n`为`Nat.gcd m n = 1`。使用 Lean 的匿名投影符号，如果`s`和`t`是类型为`Nat`的表达式，我们可以写`s.Coprime t`而不是`Nat.Coprime s t`，对于`Nat.gcd`也是如此。通常，当需要时，Lean 会自动展开`Nat.Coprime`的定义，但我们也可以通过重写或使用标识符`Nat.Coprime`进行简化来手动完成。`norm_num`策略足够智能，可以计算具体值。

```py
#print  Nat.Coprime

example  (m  n  :  Nat)  (h  :  m.Coprime  n)  :  m.gcd  n  =  1  :=
  h

example  (m  n  :  Nat)  (h  :  m.Coprime  n)  :  m.gcd  n  =  1  :=  by
  rw  [Nat.Coprime]  at  h
  exact  h

example  :  Nat.Coprime  12  7  :=  by  norm_num

example  :  Nat.gcd  12  8  =  4  :=  by  norm_num 
```

我们已经在第 2.4 节中遇到了`gcd`函数。还有为整数提供的`gcd`版本；我们将在下面讨论不同数系之间的关系。甚至还有一个通用的`gcd`函数和通用的`Prime`和`Coprime`概念，它们在一般的代数结构中是有意义的。我们将在下一章中了解 Lean 如何管理这种通用性。同时，在本节中，我们将把注意力限制在自然数上。

我们还需要素数的概念，即`Nat.Prime`。定理`Nat.prime_def_lt`提供了一种熟悉的特征描述，而`Nat.Prime.eq_one_or_self_of_dvd`提供了另一种描述。

```py
#check  Nat.prime_def_lt

example  (p  :  ℕ)  (prime_p  :  Nat.Prime  p)  :  2  ≤  p  ∧  ∀  m  :  ℕ,  m  <  p  →  m  ∣  p  →  m  =  1  :=  by
  rwa  [Nat.prime_def_lt]  at  prime_p

#check  Nat.Prime.eq_one_or_self_of_dvd

example  (p  :  ℕ)  (prime_p  :  Nat.Prime  p)  :  ∀  m  :  ℕ,  m  ∣  p  →  m  =  1  ∨  m  =  p  :=
  prime_p.eq_one_or_self_of_dvd

example  :  Nat.Prime  17  :=  by  norm_num

-- commonly used
example  :  Nat.Prime  2  :=
  Nat.prime_two

example  :  Nat.Prime  3  :=
  Nat.prime_three 
```

在自然数中，一个素数具有这样的性质：它不能被写成非平凡因子的乘积。在更广泛的数学背景下，具有这种性质的环中的元素被称为*不可约的*。如果一个元素除以一个乘积时，它能够除以其中一个因子，则称该元素为*素数*。自然数的一个重要性质是，在这种设置下，这两个概念是一致的，从而产生了定理`Nat.Prime.dvd_mul`。

我们可以使用这个事实来建立上述论证中的一个关键性质：如果一个数的平方是偶数，那么这个数也是偶数。Mathlib 在`Algebra.Group.Even`中定义谓词`Even`，但以下面的原因将变得清晰，我们将简单地使用`2 ∣ m`来表示`m`是偶数。

```py
#check  Nat.Prime.dvd_mul
#check  Nat.Prime.dvd_mul  Nat.prime_two
#check  Nat.prime_two.dvd_mul

theorem  even_of_even_sqr  {m  :  ℕ}  (h  :  2  ∣  m  ^  2)  :  2  ∣  m  :=  by
  rw  [pow_two,  Nat.prime_two.dvd_mul]  at  h
  cases  h  <;>  assumption

example  {m  :  ℕ}  (h  :  2  ∣  m  ^  2)  :  2  ∣  m  :=
  Nat.Prime.dvd_of_dvd_pow  Nat.prime_two  h 
```

随着我们继续前进，你需要熟练地找到你需要的事实。记住，如果你能猜出名称的前缀并且已经导入了相关库，你可以使用制表符补全（有时需要按`ctrl-tab`）来找到你想要的内容。你可以在任何标识符上按`ctrl-click`跳转到其定义的文件，这使你能够浏览附近的定义和定理。你还可以使用[Lean 社区网页](https://leanprover-community.github.io/)上的搜索引擎，如果所有其他方法都失败了，不要犹豫，在[Zulip](https://leanprover.zulipchat.com/)上提问。

```py
example  (a  b  c  :  Nat)  (h  :  a  *  b  =  a  *  c)  (h'  :  a  ≠  0)  :  b  =  c  :=
  -- apply? suggests the following:
  (mul_right_inj'  h').mp  h 
```

我们证明 2 的平方根是无理数的核心包含在以下定理中。看看你是否能完成证明草图，使用 `even_of_even_sqr` 和定理 `Nat.dvd_gcd`。

```py
example  {m  n  :  ℕ}  (coprime_mn  :  m.Coprime  n)  :  m  ^  2  ≠  2  *  n  ^  2  :=  by
  intro  sqr_eq
  have  :  2  ∣  m  :=  by
  sorry
  obtain  ⟨k,  meq⟩  :=  dvd_iff_exists_eq_mul_left.mp  this
  have  :  2  *  (2  *  k  ^  2)  =  2  *  n  ^  2  :=  by
  rw  [←  sqr_eq,  meq]
  ring
  have  :  2  *  k  ^  2  =  n  ^  2  :=
  sorry
  have  :  2  ∣  n  :=  by
  sorry
  have  :  2  ∣  m.gcd  n  :=  by
  sorry
  have  :  2  ∣  1  :=  by
  sorry
  norm_num  at  this 
```

事实上，经过非常少的修改，我们可以将 `2` 替换为任意素数。在下一个例子中尝试一下。在证明的末尾，你需要从 `p ∣ 1` 推导出矛盾。你可以使用 `Nat.Prime.two_le`，它表明任何素数都大于或等于 2，以及 `Nat.le_of_dvd`。

```py
example  {m  n  p  :  ℕ}  (coprime_mn  :  m.Coprime  n)  (prime_p  :  p.Prime)  :  m  ^  2  ≠  p  *  n  ^  2  :=  by
  sorry 
```

让我们考虑另一种方法。以下是一个快速证明：如果 $p$ 是素数，那么 $m² \ne p n²$：如果我们假设 $m² = p n²$ 并考虑 $m$ 和 $n$ 的素数分解，那么 $p$ 在等式左边出现偶数次，在右边出现奇数次，这是矛盾的。请注意，这个论证要求 $n$ 以及因此 $m$ 不等于零。下面的形式化确认了这个假设是充分的。

唯一分解定理表明，除了零以外的任何自然数都可以以唯一的方式写成素数的乘积。Mathlib 包含了这个定理的正式版本，它用函数 `Nat.primeFactorsList` 来表达，该函数返回一个数在非递减顺序中的素数因子列表。库证明了 `Nat.primeFactorsList n` 的所有元素都是素数，任何大于零的 `n` 都等于其因子的乘积，并且如果 `n` 等于另一个素数列表的乘积，那么该列表是 `Nat.primeFactorsList n` 的一个排列。

```py
#check  Nat.primeFactorsList
#check  Nat.prime_of_mem_primeFactorsList
#check  Nat.prod_primeFactorsList
#check  Nat.primeFactorsList_unique 
```

尽管我们还没有讨论列表成员、乘积或排列，你仍然可以浏览这些定理以及其他附近的定理。对于当前的任务，我们不需要任何这些。我们将使用 Mathlib 有一个函数 `Nat.factorization` 的事实，它代表与函数相同的数据。具体来说，`Nat.factorization n p`，我们也可以写成 `n.factorization p`，返回 `p` 在 `n` 的素数分解中的次数。我们将使用以下三个事实。

```py
theorem  factorization_mul'  {m  n  :  ℕ}  (mnez  :  m  ≠  0)  (nnez  :  n  ≠  0)  (p  :  ℕ)  :
  (m  *  n).factorization  p  =  m.factorization  p  +  n.factorization  p  :=  by
  rw  [Nat.factorization_mul  mnez  nnez]
  rfl

theorem  factorization_pow'  (n  k  p  :  ℕ)  :
  (n  ^  k).factorization  p  =  k  *  n.factorization  p  :=  by
  rw  [Nat.factorization_pow]
  rfl

theorem  Nat.Prime.factorization'  {p  :  ℕ}  (prime_p  :  p.Prime)  :
  p.factorization  p  =  1  :=  by
  rw  [prime_p.factorization]
  simp 
```

事实上，`n.factorization` 在 Lean 中被定义为有限支撑的函数，这解释了你将在上述证明中看到的奇怪符号。现在不用担心这个。就我们的目的而言，我们可以将上述三个定理视为黑盒使用。

以下示例表明，简化器足够智能，可以将 `n² ≠ 0` 替换为 `n ≠ 0`。`simpa` 策略只是调用 `simp` 后跟 `assumption`。

看看你是否能使用上述恒等式来填补证明中的空白部分。

```py
example  {m  n  p  :  ℕ}  (nnz  :  n  ≠  0)  (prime_p  :  p.Prime)  :  m  ^  2  ≠  p  *  n  ^  2  :=  by
  intro  sqr_eq
  have  nsqr_nez  :  n  ^  2  ≠  0  :=  by  simpa
  have  eq1  :  Nat.factorization  (m  ^  2)  p  =  2  *  m.factorization  p  :=  by
  sorry
  have  eq2  :  (p  *  n  ^  2).factorization  p  =  2  *  n.factorization  p  +  1  :=  by
  sorry
  have  :  2  *  m.factorization  p  %  2  =  (2  *  n.factorization  p  +  1)  %  2  :=  by
  rw  [←  eq1,  sqr_eq,  eq2]
  rw  [add_comm,  Nat.add_mul_mod_self_left,  Nat.mul_mod_right]  at  this
  norm_num  at  this 
```

这个证明的一个好处是它也具有普遍性。关于 `2` 没有什么特别之处；经过一些小的修改，这个证明表明，每当我们将 `m^k = r * n^k` 写出来时，`r` 中任何素数 `p` 的次数必须是 `k` 的倍数。

要使用 `Nat.count_factors_mul_of_pos` 与 `r * n^k`，我们需要知道 `r` 是正数。但当 `r` 为零时，下面的定理是显而易见的，并且可以通过简化器轻松证明。因此，证明是分情况进行的。`rcases r with _ | r` 这一行将目标替换为两个版本：一个是将 `r` 替换为 `0` 的版本，另一个是将 `r` 替换为 `r + 1` 的版本。在第二种情况下，我们可以使用定理 `r.succ_ne_zero`，它建立了 `r + 1 ≠ 0`（`succ` 表示后继）。

还要注意，以 `have : npow_nz` 开头的行提供了一个简短的证明项证明 `n^k ≠ 0`。要理解它是如何工作的，试着用战术证明来替换它，然后思考战术是如何描述证明项的。

看看你是否能填补下面证明的空白部分。在最后，你可以使用 `Nat.dvd_sub'` 和 `Nat.dvd_mul_right` 来完成它。

注意，这个例子并没有假设 `p` 是素数，但当 `p` 不是素数时，结论是显而易见的，因为根据定义，`r.factorization p` 将为零，而且证明在所有情况下都适用。

```py
example  {m  n  k  r  :  ℕ}  (nnz  :  n  ≠  0)  (pow_eq  :  m  ^  k  =  r  *  n  ^  k)  {p  :  ℕ}  :
  k  ∣  r.factorization  p  :=  by
  rcases  r  with  _  |  r
  ·  simp
  have  npow_nz  :  n  ^  k  ≠  0  :=  fun  npowz  ↦  nnz  (pow_eq_zero  npowz)
  have  eq1  :  (m  ^  k).factorization  p  =  k  *  m.factorization  p  :=  by
  sorry
  have  eq2  :  ((r  +  1)  *  n  ^  k).factorization  p  =
  k  *  n.factorization  p  +  (r  +  1).factorization  p  :=  by
  sorry
  have  :  r.succ.factorization  p  =  k  *  m.factorization  p  -  k  *  n.factorization  p  :=  by
  rw  [←  eq1,  pow_eq,  eq2,  add_comm,  Nat.add_sub_cancel]
  rw  [this]
  sorry 
```

我们可能希望以多种方式改进这些结果。首先，证明 2 的平方根是无理数应该对 2 的平方根有所说明，这可以理解为实数或复数中的一个元素。并且说它是无理数应该对有理数有所说明，即没有有理数等于它。此外，我们还应该将本节中的定理扩展到整数。虽然从数学上明显，如果我们能将 2 的平方根写成两个整数的商，那么我们也可以将其写成两个自然数的商，但正式证明这一点需要一些努力。

在 Mathlib 中，自然数、整数、有理数、实数和复数由不同的数据类型表示。将注意力限制在单独的域中通常是有帮助的：我们将看到对自然数进行归纳是很容易的，而且当实数不是图景的一部分时，关于整数可除性的推理是最容易的。但需要在不同的域之间进行调解，这是一个头疼的问题，我们将不得不应对。我们将在本章的后面回到这个问题。

我们还应该期望能够加强最后一个定理的结论，使其表明数 `r` 是一个 `k` 次幂，因为它的 `k` 次根只是每个除 `r` 的质数乘以其在 `r` 中的幂次，然后除以 `k` 的乘积。要能够做到这一点，我们需要更好的推理手段来处理有限集上的乘积和求和，这也是我们将回到的话题。

实际上，本节中的结果在 Mathlib 的 `Data.Real.Irrational` 中以更大的普遍性得到确立。对于任意交换幺半群，定义了 `multiplicity` 的概念，并且它在扩展自然数 `enat` 中取值，它将无穷大值添加到自然数中。在下一章中，我们将开始开发欣赏 Lean 支持这种普遍性的方法。

## 5.2. 归纳与递归

自然数集合 $\mathbb{N} = \{ 0, 1, 2, \ldots \}$ 不仅在其自身的基础上具有根本重要性，而且在构建新的数学对象中也起着核心作用。Lean 的基础允许我们声明 *归纳类型*，这些类型是通过给定的构造函数列表归纳生成的。在 Lean 中，自然数被声明如下。

```py
inductive  Nat  where
  |  zero  :  Nat
  |  succ  (n  :  Nat)  :  Nat 
```

您可以在图书馆中通过输入 `#check Nat` 然后点击 `ctrl-click` 在标识符 `Nat` 上找到它。该命令指定 `Nat` 是由两个构造函数 `zero : Nat` 和 `succ : Nat → Nat` 自由且归纳地生成的数据类型。当然，库为 `nat` 和 `zero` 分别引入了符号 `ℕ` 和 `0`。（数值被转换为二进制表示，但我们现在不必担心这个细节。）

对于工作的数学家来说，“自由”意味着类型 `Nat` 有一个元素 `zero` 和一个不包含 `zero` 的像的注入后继函数 `succ`。

```py
example  (n  :  Nat)  :  n.succ  ≠  Nat.zero  :=
  Nat.succ_ne_zero  n

example  (m  n  :  Nat)  (h  :  m.succ  =  n.succ)  :  m  =  n  :=
  Nat.succ.inj  h 
```

对于工作的数学家来说，“归纳”意味着自然数带有归纳证明原理和递归定义原理。本节将向您展示如何使用这些原理。

下面是阶乘函数的递归定义的一个例子。

```py
def  fac  :  ℕ  →  ℕ
  |  0  =>  1
  |  n  +  1  =>  (n  +  1)  *  fac  n 
```

语法需要一些时间来习惯。请注意，第一行没有 `:=`。接下来的两行提供了递归定义的基础情况和归纳步骤。这些等式在定义上是成立的，但也可以通过将 `fac` 命名给 `simp` 或 `rw` 来手动使用。

```py
example  :  fac  0  =  1  :=
  rfl

example  :  fac  0  =  1  :=  by
  rw  [fac]

example  :  fac  0  =  1  :=  by
  simp  [fac]

example  (n  :  ℕ)  :  fac  (n  +  1)  =  (n  +  1)  *  fac  n  :=
  rfl

example  (n  :  ℕ)  :  fac  (n  +  1)  =  (n  +  1)  *  fac  n  :=  by
  rw  [fac]

example  (n  :  ℕ)  :  fac  (n  +  1)  =  (n  +  1)  *  fac  n  :=  by
  simp  [fac] 
```

实际上，阶乘函数已经在 Mathlib 中定义为 `Nat.factorial`。您可以通过输入 `#check Nat.factorial` 并使用 `ctrl-click` 来跳转到它。为了说明目的，我们将在示例中继续使用 `fac`。定义 `Nat.factorial` 前的注释 `@[simp]` 指定定义方程应该被添加到简化器默认使用的恒等式数据库中。

归纳原理表明，我们可以通过证明该命题对 0 成立，并且每当它对自然数$n$成立时，它也对$n + 1$成立，来证明关于自然数的普遍命题。因此，以下证明中的“归纳' n with n ih”这一行产生了两个目标：在第一个目标中，我们需要证明`0 < fac 0`，而在第二个目标中，我们有附加的假设`ih : 0 < fac n`和需要证明的`0 < fac (n + 1)`。短语`with n ih`用于命名归纳假设中的变量和假设，你可以为它们选择任何你想要的名称。

```py
theorem  fac_pos  (n  :  ℕ)  :  0  <  fac  n  :=  by
  induction'  n  with  n  ih
  ·  rw  [fac]
  exact  zero_lt_one
  rw  [fac]
  exact  mul_pos  n.succ_pos  ih 
```

`induction'`策略足够智能，可以将依赖于归纳变量的假设作为归纳假设的一部分。通过下一个示例逐步查看发生了什么。

```py
theorem  dvd_fac  {i  n  :  ℕ}  (ipos  :  0  <  i)  (ile  :  i  ≤  n)  :  i  ∣  fac  n  :=  by
  induction'  n  with  n  ih
  ·  exact  absurd  ipos  (not_lt_of_ge  ile)
  rw  [fac]
  rcases  Nat.of_le_succ  ile  with  h  |  h
  ·  apply  dvd_mul_of_dvd_right  (ih  h)
  rw  [h]
  apply  dvd_mul_right 
```

以下示例提供了一个关于阶乘函数的粗糙下界。实际上，从证明的案例开始更容易，因此证明的其余部分从$n = 1$的情况开始。看看你是否可以用`pow_succ`或`pow_succ'`通过归纳证明来完成论证。

```py
theorem  pow_two_le_fac  (n  :  ℕ)  :  2  ^  (n  -  1)  ≤  fac  n  :=  by
  rcases  n  with  _  |  n
  ·  simp  [fac]
  sorry 
```

归纳通常用于证明涉及有限和与积的恒等式。Mathlib 定义了表达式`Finset.sum s f`，其中`s : Finset α`是类型`α`的元素有限集合，`f`是在`α`上定义的函数。`f`的陪域可以是支持具有零元素的交换、结合加法运算的任何类型。如果你导入`Algebra.BigOperators.Ring`并发出`open BigOperators`命令，你可以使用更具说明性的符号`∑ x ∈ s, f x`。当然，也存在类似的操作和符号用于有限积。

我们将在下一节中讨论`Finset`类型及其支持的运算，并在稍后的章节中再次讨论。现在，我们只会使用`Finset.range n`，这是小于`n`的自然数的有限集合。

```py
variable  {α  :  Type*}  (s  :  Finset  ℕ)  (f  :  ℕ  →  ℕ)  (n  :  ℕ)

#check  Finset.sum  s  f
#check  Finset.prod  s  f

open  BigOperators
open  Finset

example  :  s.sum  f  =  ∑  x  ∈  s,  f  x  :=
  rfl

example  :  s.prod  f  =  ∏  x  ∈  s,  f  x  :=
  rfl

example  :  (range  n).sum  f  =  ∑  x  ∈  range  n,  f  x  :=
  rfl

example  :  (range  n).prod  f  =  ∏  x  ∈  range  n,  f  x  :=
  rfl 
```

事实`Finset.sum_range_zero`和`Finset.sum_range_succ`提供了对$n$求和的递归描述，同样也适用于积。

```py
example  (f  :  ℕ  →  ℕ)  :  ∑  x  ∈  range  0,  f  x  =  0  :=
  Finset.sum_range_zero  f

example  (f  :  ℕ  →  ℕ)  (n  :  ℕ)  :  ∑  x  ∈  range  n.succ,  f  x  =  ∑  x  ∈  range  n,  f  x  +  f  n  :=
  Finset.sum_range_succ  f  n

example  (f  :  ℕ  →  ℕ)  :  ∏  x  ∈  range  0,  f  x  =  1  :=
  Finset.prod_range_zero  f

example  (f  :  ℕ  →  ℕ)  (n  :  ℕ)  :  ∏  x  ∈  range  n.succ,  f  x  =  (∏  x  ∈  range  n,  f  x)  *  f  n  :=
  Finset.prod_range_succ  f  n 
```

每一对中的第一个恒等式是定义上成立的，也就是说，你可以用`rfl`替换证明。

以下表示我们定义的阶乘函数，作为一个乘积。

```py
example  (n  :  ℕ)  :  fac  n  =  ∏  i  ∈  range  n,  (i  +  1)  :=  by
  induction'  n  with  n  ih
  ·  simp  [fac,  prod_range_zero]
  simp  [fac,  ih,  prod_range_succ,  mul_comm] 
```

我们将`mul_comm`作为简化规则包括在内的事实值得注意。使用恒等式`x * y = y * x`进行简化似乎很危险，因为这通常会无限循环。Lean 的简化器足够智能，能够识别这一点，并且仅在结果项在某种固定但任意的项的排序中具有更小值的情况下应用该规则。以下示例表明，使用`mul_assoc`、`mul_comm`和`mul_left_comm`这三个规则进行简化，可以成功地识别出相同但括号位置和变量排序不同的乘积。

```py
example  (a  b  c  d  e  f  :  ℕ)  :  a  *  (b  *  c  *  f  *  (d  *  e))  =  d  *  (a  *  f  *  e)  *  (c  *  b)  :=  by
  simp  [mul_assoc,  mul_comm,  mul_left_comm] 
```

大概来说，规则是通过将括号推向右边，然后重新排列两边的表达式，直到它们都遵循相同的规范顺序。使用这些规则和相应的加法规则进行简化是一个实用的技巧。

返回到求和恒等式，我们建议逐步通过以下证明，即自然数从 1 加到$n$的和是$n (n + 1) / 2$。证明的第一步消除了分母。这在形式化恒等式时通常很有用，因为除法运算通常有副作用条件。（同样，在可能的情况下避免在自然数上使用减法也是有用的。）

```py
theorem  sum_id  (n  :  ℕ)  :  ∑  i  ∈  range  (n  +  1),  i  =  n  *  (n  +  1)  /  2  :=  by
  symm;  apply  Nat.div_eq_of_eq_mul_right  (by  norm_num  :  0  <  2)
  induction'  n  with  n  ih
  ·  simp
  rw  [Finset.sum_range_succ,  mul_add  2,  ←  ih]
  ring 
```

我们鼓励你证明平方和的类似恒等式，以及你在网上能找到的其他恒等式。

```py
theorem  sum_sqr  (n  :  ℕ)  :  ∑  i  ∈  range  (n  +  1),  i  ^  2  =  n  *  (n  +  1)  *  (2  *  n  +  1)  /  6  :=  by
  sorry 
```

在 Lean 的核心库中，加法和乘法本身是通过递归定义的，它们的基本性质是通过归纳法建立的。如果你喜欢思考像那样的基础主题，你可能会喜欢通过乘法和加法的交换律、结合律以及乘法对加法的分配律的证明来工作。你可以在以下概述的自然数副本上这样做。注意，我们可以使用`induction`策略与`MyNat`一起使用；Lean 足够智能，知道要使用相关的归纳原理（当然，这与`Nat`的相同）。

我们从加法的交换律开始。一个很好的经验法则是，由于加法和乘法是通过递归定义在第二个参数上的，因此通常在出现该位置的变量上通过归纳法进行证明是有利的。在证明结合律时决定使用哪个变量有点棘手。

没有使用零、一、加法和乘法的常规符号来写东西可能会让人困惑。我们将在后面学习如何定义这样的符号。在命名空间`MyNat`中工作意味着我们可以写`zero`和`succ`而不是`MyNat.zero`和`MyNat.succ`，并且这些名称的解释优先于其他解释。在命名空间外部，下面定义的`add`的全名是`MyNat.add`。

如果你发现你真的喜欢这类东西，尝试定义截断减法和指数运算，并证明它们的一些性质。记住，截断减法在零处截断。为了定义这一点，定义一个前驱函数`pred`，它从任何非零数中减去一，并固定零。函数`pred`可以通过一个简单的递归实例来定义。

```py
inductive  MyNat  where
  |  zero  :  MyNat
  |  succ  :  MyNat  →  MyNat

namespace  MyNat

def  add  :  MyNat  →  MyNat  →  MyNat
  |  x,  zero  =>  x
  |  x,  succ  y  =>  succ  (add  x  y)

def  mul  :  MyNat  →  MyNat  →  MyNat
  |  x,  zero  =>  zero
  |  x,  succ  y  =>  add  (mul  x  y)  x

theorem  zero_add  (n  :  MyNat)  :  add  zero  n  =  n  :=  by
  induction'  n  with  n  ih
  ·  rfl
  rw  [add,  ih]

theorem  succ_add  (m  n  :  MyNat)  :  add  (succ  m)  n  =  succ  (add  m  n)  :=  by
  induction'  n  with  n  ih
  ·  rfl
  rw  [add,  ih]
  rfl

theorem  add_comm  (m  n  :  MyNat)  :  add  m  n  =  add  n  m  :=  by
  induction'  n  with  n  ih
  ·  rw  [zero_add]
  rfl
  rw  [add,  succ_add,  ih]

theorem  add_assoc  (m  n  k  :  MyNat)  :  add  (add  m  n)  k  =  add  m  (add  n  k)  :=  by
  sorry
theorem  mul_add  (m  n  k  :  MyNat)  :  mul  m  (add  n  k)  =  add  (mul  m  n)  (mul  m  k)  :=  by
  sorry
theorem  zero_mul  (n  :  MyNat)  :  mul  zero  n  =  zero  :=  by
  sorry
theorem  succ_mul  (m  n  :  MyNat)  :  mul  (succ  m)  n  =  add  (mul  m  n)  n  :=  by
  sorry
theorem  mul_comm  (m  n  :  MyNat)  :  mul  m  n  =  mul  n  m  :=  by
  sorry
end  MyNat 
```

## 5.3. 无穷多个素数

让我们继续用另一个数学标准来探索归纳和递归：证明存在无限多个质数。一种表述方式是，对于每一个自然数 $n$，都存在一个大于 $n$ 的质数。为了证明这一点，设 $p$ 是 $n! + 1$ 的任意一个质因数。如果 $p$ 小于或等于 $n$，它就会整除 $n!$。由于它也整除 $n! + 1$，它就会整除 1，这产生了矛盾。因此 $p$ 必须大于 $n$。

为了形式化这个证明，我们需要证明任何大于或等于 2 的数都有一个质因数。要做到这一点，我们需要证明任何不等于 0 或 1 的自然数都大于或等于 2。这引出了形式化中的一个奇特特性：通常像这样的简单陈述是最令人烦恼的。在这里，我们考虑了几种实现方法。

首先，我们可以使用 `cases` 策略和后继函数尊重自然数上的顺序这一事实。

```py
theorem  two_le  {m  :  ℕ}  (h0  :  m  ≠  0)  (h1  :  m  ≠  1)  :  2  ≤  m  :=  by
  cases  m;  contradiction
  case  succ  m  =>
  cases  m;  contradiction
  repeat  apply  Nat.succ_le_succ
  apply  zero_le 
```

另一种策略是使用 `interval_cases` 策略，它会自动将目标分解为当变量包含在自然数或整数的区间内时的各种情况。记住，你可以悬停在它上面来查看其文档。

```py
example  {m  :  ℕ}  (h0  :  m  ≠  0)  (h1  :  m  ≠  1)  :  2  ≤  m  :=  by
  by_contra  h
  push_neg  at  h
  interval_cases  m  <;>  contradiction 
```

回想一下，`interval_cases m` 后面的分号意味着下一个策略会被应用到它生成的每一个情况中。另一个选择是使用 `decide` 策略，它试图找到一个决策过程来解决问题。Lean 知道你可以通过决定每个有限实例来决定以有界量词 `∀ x, x < n → ...` 或 `∃ x, x < n ∧ ...` 开头的语句的真值。

```py
example  {m  :  ℕ}  (h0  :  m  ≠  0)  (h1  :  m  ≠  1)  :  2  ≤  m  :=  by
  by_contra  h
  push_neg  at  h
  revert  h0  h1
  revert  h  m
  decide 
```

在手头有 `two_le` 定理的情况下，让我们首先证明每个大于两的自然数都有一个质数因子。Mathlib 包含一个函数 `Nat.minFac`，它返回最小的质数因子，但为了学习库的新部分，我们将避免使用它，并直接证明这个定理。

在这里，普通的归纳不足以。我们想要使用 *强归纳*，它允许我们通过证明对于每个数 $n$，如果 $P$ 对所有小于 $n$ 的值成立，那么它对 $n$ 也成立，来证明每个自然数 $n$ 都有一个性质 $P$。在 Lean 中，这个原则被称为 `Nat.strong_induction_on`，我们可以使用 `using` 关键字告诉归纳策略使用它。注意，当我们这样做时，没有基础情况；它被归纳步骤所包含。

论证如下。假设 $n ≥ 2$，如果 $n$ 是质数，那么我们就完成了。如果不是，那么根据质数的定义之一，它有一个非平凡因子 $m$，我们可以对它应用归纳假设。通过查看下一个证明，我们可以看到这是如何实现的。

```py
theorem  exists_prime_factor  {n  :  Nat}  (h  :  2  ≤  n)  :  ∃  p  :  Nat,  p.Prime  ∧  p  ∣  n  :=  by
  by_cases  np  :  n.Prime
  ·  use  n,  np
  induction'  n  using  Nat.strong_induction_on  with  n  ih
  rw  [Nat.prime_def_lt]  at  np
  push_neg  at  np
  rcases  np  h  with  ⟨m,  mltn,  mdvdn,  mne1⟩
  have  :  m  ≠  0  :=  by
  intro  mz
  rw  [mz,  zero_dvd_iff]  at  mdvdn
  linarith
  have  mgt2  :  2  ≤  m  :=  two_le  this  mne1
  by_cases  mp  :  m.Prime
  ·  use  m,  mp
  ·  rcases  ih  m  mltn  mgt2  mp  with  ⟨p,  pp,  pdvd⟩
  use  p,  pp
  apply  pdvd.trans  mdvdn 
```

我们现在可以证明我们定理的以下表述。看看你是否能够完成草图。你可以使用 `Nat.factorial_pos`、`Nat.dvd_factorial` 和 `Nat.dvd_sub'`。

```py
theorem  primes_infinite  :  ∀  n,  ∃  p  >  n,  Nat.Prime  p  :=  by
  intro  n
  have  :  2  ≤  Nat.factorial  n  +  1  :=  by
  sorry
  rcases  exists_prime_factor  this  with  ⟨p,  pp,  pdvd⟩
  refine  ⟨p,  ?_,  pp⟩
  show  p  >  n
  by_contra  ple
  push_neg  at  ple
  have  :  p  ∣  Nat.factorial  n  :=  by
  sorry
  have  :  p  ∣  1  :=  by
  sorry
  show  False
  sorry 
```

让我们考虑上述证明的一个变体，其中我们不是使用阶乘函数，而是假设我们被赋予一个有限集 $\{ p_1, \ldots, p_n \}$，并考虑 $\prod_{i = 1}^n p_i + 1$ 的一个素数因子。这个素数因子必须与每个 $p_i$ 不同，这表明不存在包含所有素数的有限集。

将这个论证形式化需要我们对有限集进行推理。在 Lean 中，对于任何类型 `α`，类型 `Finset α` 表示类型 `α` 的元素构成的有限集。对有限集进行计算推理需要有一个在 `α` 上测试等价的程序，这就是为什么下面的代码片段包括了假设 `[DecidableEq α]`。对于像 `ℕ`、`ℤ` 和 `ℚ` 这样的具体数据类型，这个假设会自动满足。当推理实数时，可以使用经典逻辑并放弃计算解释来满足这个假设。

我们使用命令 `open Finset` 来使用相关定理的简短名称。与集合的情况不同，涉及有限集的多数等价关系在定义上并不成立，因此需要手动使用等价关系如 `Finset.subset_iff`、`Finset.mem_union`、`Finset.mem_inter` 和 `Finset.mem_sdiff` 来扩展。`ext` 策略仍然可以用来通过显示一个有限集的每个元素都是另一个集合的元素来证明两个有限集相等。

```py
open  Finset

section
variable  {α  :  Type*}  [DecidableEq  α]  (r  s  t  :  Finset  α)

example  :  r  ∩  (s  ∪  t)  ⊆  r  ∩  s  ∪  r  ∩  t  :=  by
  rw  [subset_iff]
  intro  x
  rw  [mem_inter,  mem_union,  mem_union,  mem_inter,  mem_inter]
  tauto

example  :  r  ∩  (s  ∪  t)  ⊆  r  ∩  s  ∪  r  ∩  t  :=  by
  simp  [subset_iff]
  intro  x
  tauto

example  :  r  ∩  s  ∪  r  ∩  t  ⊆  r  ∩  (s  ∪  t)  :=  by
  simp  [subset_iff]
  intro  x
  tauto

example  :  r  ∩  s  ∪  r  ∩  t  =  r  ∩  (s  ∪  t)  :=  by
  ext  x
  simp
  tauto

end 
```

我们使用了一个新技巧：`tauto` 策略（以及一个增强版本 `tauto!`，它使用经典逻辑）可以用来处理命题恒真。看看你是否可以使用这些方法来证明下面的两个例子。

```py
example  :  (r  ∪  s)  ∩  (r  ∪  t)  =  r  ∪  s  ∩  t  :=  by
  sorry
example  :  (r  \  s)  \  t  =  r  \  (s  ∪  t)  :=  by
  sorry 
```

定理 `Finset.dvd_prod_of_mem` 告诉我们，如果 `n` 是有限集 `s` 的一个元素，那么 `n` 可以整除 `∏ i ∈ s, i`。

```py
example  (s  :  Finset  ℕ)  (n  :  ℕ)  (h  :  n  ∈  s)  :  n  ∣  ∏  i  ∈  s,  i  :=
  Finset.dvd_prod_of_mem  _  h 
```

我们还需要知道，当 `n` 是素数且 `s` 是素数集合时，逆命题也成立。为了证明这一点，我们需要以下引理，你应该能够使用定理 `Nat.Prime.eq_one_or_self_of_dvd` 来证明。

```py
theorem  _root_.Nat.Prime.eq_of_dvd_of_prime  {p  q  :  ℕ}
  (prime_p  :  Nat.Prime  p)  (prime_q  :  Nat.Prime  q)  (h  :  p  ∣  q)  :
  p  =  q  :=  by
  sorry 
```

我们可以使用这个引理来证明如果一个素数 `p` 整除有限个素数的乘积，那么它等于其中的一个。Mathlib 提供了一个有用的有限集归纳原理：为了证明一个性质对任意有限集 `s` 成立，需要证明它对空集成立，并且当我们在其中添加一个新元素 `a ∉ s` 时，它仍然成立。这个原理被称为 `Finset.induction_on`。当我们告诉归纳策略使用它时，我们还可以指定 `a` 和 `s` 的名称，以及在归纳步骤中的假设 `a ∉ s` 的名称，以及归纳假设的名称。表达式 `Finset.insert a s` 表示 `s` 与单元素集合 `a` 的并集。`Finset.prod_empty` 和 `Finset.prod_insert` 然后提供了与乘积相关的重写规则。在下面的证明中，第一个 `simp` 应用了 `Finset.prod_empty`。逐步查看证明的开始部分以查看归纳展开，然后完成它。

```py
theorem  mem_of_dvd_prod_primes  {s  :  Finset  ℕ}  {p  :  ℕ}  (prime_p  :  p.Prime)  :
  (∀  n  ∈  s,  Nat.Prime  n)  →  (p  ∣  ∏  n  ∈  s,  n)  →  p  ∈  s  :=  by
  intro  h₀  h₁
  induction'  s  using  Finset.induction_on  with  a  s  ans  ih
  ·  simp  at  h₁
  linarith  [prime_p.two_le]
  simp  [Finset.prod_insert  ans,  prime_p.dvd_mul]  at  h₀  h₁
  rw  [mem_insert]
  sorry 
```

我们需要有限集的一个最后属性。给定一个元素 `s : Set α` 和一个在 `α` 上的谓词 `P`，在 第四章 中，我们用 `{ x ∈ s | P x }` 表示满足 `P` 的 `s` 的元素集合。给定 `s : Finset α`，类似的概念表示为 `s.filter P`。

```py
example  (s  :  Finset  ℕ)  (x  :  ℕ)  :  x  ∈  s.filter  Nat.Prime  ↔  x  ∈  s  ∧  x.Prime  :=
  mem_filter 
```

现在我们证明关于存在无限多个素数的另一种表述，即给定任何 `s : Finset ℕ`，存在一个素数 `p`，它不是 `s` 的元素。为了达到矛盾，我们假设所有素数都在 `s` 中，然后削减到一个只包含所有素数的集合 `s'`。取该集合的乘积，加一，并找到结果的一个素数因子，导致我们寻找的矛盾。看看你是否能完成下面的草图。你可以在第一个 `have` 的证明中使用 `Finset.prod_pos`。

```py
theorem  primes_infinite'  :  ∀  s  :  Finset  Nat,  ∃  p,  Nat.Prime  p  ∧  p  ∉  s  :=  by
  intro  s
  by_contra  h
  push_neg  at  h
  set  s'  :=  s.filter  Nat.Prime  with  s'_def
  have  mem_s'  :  ∀  {n  :  ℕ},  n  ∈  s'  ↔  n.Prime  :=  by
  intro  n
  simp  [s'_def]
  apply  h
  have  :  2  ≤  (∏  i  ∈  s',  i)  +  1  :=  by
  sorry
  rcases  exists_prime_factor  this  with  ⟨p,  pp,  pdvd⟩
  have  :  p  ∣  ∏  i  ∈  s',  i  :=  by
  sorry
  have  :  p  ∣  1  :=  by
  convert  Nat.dvd_sub  pdvd  this
  simp
  show  False
  sorry 
```

因此，我们已经看到了两种表达存在无限多个素数的方法：说它们不被任何 `n` 所界定，以及说它们不包含在任何有限集 `s` 中。下面的两个证明表明这些表述是等价的。在第二个证明中，为了形成 `s.filter Q`，我们必须假设存在一个判断 `Q` 是否成立的程序。Lean 知道存在一个用于 `Nat.Prime` 的程序。一般来说，如果我们通过编写 `open Classical` 使用经典逻辑，我们可以省去这个假设。

在 Mathlib 中，`Finset.sup s f` 表示 `f x` 的值在 `x` 在 `s` 上取值时的上确界，当 `s` 为空且 `f` 的陪域为 `ℕ` 时返回 `0`。在第一个证明中，我们使用 `s.sup id`，其中 `id` 是恒等函数，来指代 `s` 中的最大值。

```py
theorem  bounded_of_ex_finset  (Q  :  ℕ  →  Prop)  :
  (∃  s  :  Finset  ℕ,  ∀  k,  Q  k  →  k  ∈  s)  →  ∃  n,  ∀  k,  Q  k  →  k  <  n  :=  by
  rintro  ⟨s,  hs⟩
  use  s.sup  id  +  1
  intro  k  Qk
  apply  Nat.lt_succ_of_le
  show  id  k  ≤  s.sup  id
  apply  le_sup  (hs  k  Qk)

theorem  ex_finset_of_bounded  (Q  :  ℕ  →  Prop)  [DecidablePred  Q]  :
  (∃  n,  ∀  k,  Q  k  →  k  ≤  n)  →  ∃  s  :  Finset  ℕ,  ∀  k,  Q  k  ↔  k  ∈  s  :=  by
  rintro  ⟨n,  hn⟩
  use  (range  (n  +  1)).filter  Q
  intro  k
  simp  [Nat.lt_succ_iff]
  exact  hn  k 
```

在我们的第二个证明中，即存在无限多个素数，有一个小的变化表明存在无限多个与 4 模 3 同余的素数。论证如下。首先，注意如果两个数 $m$ 和 $n$ 的乘积等于 4 模 3，那么这两个数中有一个与 4 模 3 同余。毕竟，它们都必须是奇数，如果它们都等于 4 模 3 的 1，那么它们的乘积也是。我们可以利用这个观察结果来证明如果一个大于 2 的数与 4 模 3 同余，那么这个数有一个与 4 模 3 同余的素数因子。

现在假设只有有限多个与 4 模 3 同余的素数，比如说，$p_1, \ldots, p_k$。不失一般性，我们可以假设 $p_1 = 3$。考虑乘积 $4 \prod_{i = 2}^k p_i + 3$。很容易看出这个数与 4 模 3 同余，所以它有一个与 4 模 3 同余的素数因子 $p$。不可能 $p = 3$；因为 $p$ 整除 $4 \prod_{i = 2}^k p_i + 3$，如果 $p$ 等于 3，那么它也会整除 $\prod_{i = 2}^k p_i$，这意味着 $p$ 等于 $p_i$ 中的一个，对于 $i = 2, \ldots, k$；但我们已经排除了 3。所以 $p$ 必须是其他元素 $p_i$ 中的一个。但在那种情况下，$p$ 整除 $4 \prod_{i = 2}^k p_i$ 和 3，这与它不是 3 的事实相矛盾。

在精益（Lean）中，表示法 `n % m`，读作“`n` 模 `m`”，表示 `n` 除以 `m` 的余数。

```py
example  :  27  %  4  =  3  :=  by  norm_num 
```

然后，我们可以将陈述“`n` 与 4 模 3 同余”表示为 `n % 4 = 3`。以下示例和定理总结了我们将需要使用的事实。第一个命名的定理是通过对少数几个情况进行推理的另一个例子。在第二个命名的定理中，记住分号意味着后续的策略块应用于由前面的策略块创建的所有目标。

```py
example  (n  :  ℕ)  :  (4  *  n  +  3)  %  4  =  3  :=  by
  rw  [add_comm,  Nat.add_mul_mod_self_left]

theorem  mod_4_eq_3_or_mod_4_eq_3  {m  n  :  ℕ}  (h  :  m  *  n  %  4  =  3)  :  m  %  4  =  3  ∨  n  %  4  =  3  :=  by
  revert  h
  rw  [Nat.mul_mod]
  have  :  m  %  4  <  4  :=  Nat.mod_lt  m  (by  norm_num)
  interval_cases  m  %  4  <;>  simp  [-Nat.mul_mod_mod]
  have  :  n  %  4  <  4  :=  Nat.mod_lt  n  (by  norm_num)
  interval_cases  n  %  4  <;>  simp

theorem  two_le_of_mod_4_eq_3  {n  :  ℕ}  (h  :  n  %  4  =  3)  :  2  ≤  n  :=  by
  apply  two_le  <;>
  ·  intro  neq
  rw  [neq]  at  h
  norm_num  at  h 
```

我们还需要以下事实，即如果 `m` 是 `n` 的非平凡因子，那么 `n / m` 也是。看看你是否可以使用 `Nat.div_dvd_of_dvd` 和 `Nat.div_lt_self` 来完成证明。

```py
theorem  aux  {m  n  :  ℕ}  (h₀  :  m  ∣  n)  (h₁  :  2  ≤  m)  (h₂  :  m  <  n)  :  n  /  m  ∣  n  ∧  n  /  m  <  n  :=  by
  sorry 
```

现在将所有这些部分组合起来，证明任何与 4 模 3 同余的数都有一个具有相同性质的素数因子。

```py
theorem  exists_prime_factor_mod_4_eq_3  {n  :  Nat}  (h  :  n  %  4  =  3)  :
  ∃  p  :  Nat,  p.Prime  ∧  p  ∣  n  ∧  p  %  4  =  3  :=  by
  by_cases  np  :  n.Prime
  ·  use  n
  induction'  n  using  Nat.strong_induction_on  with  n  ih
  rw  [Nat.prime_def_lt]  at  np
  push_neg  at  np
  rcases  np  (two_le_of_mod_4_eq_3  h)  with  ⟨m,  mltn,  mdvdn,  mne1⟩
  have  mge2  :  2  ≤  m  :=  by
  apply  two_le  _  mne1
  intro  mz
  rw  [mz,  zero_dvd_iff]  at  mdvdn
  linarith
  have  neq  :  m  *  (n  /  m)  =  n  :=  Nat.mul_div_cancel'  mdvdn
  have  :  m  %  4  =  3  ∨  n  /  m  %  4  =  3  :=  by
  apply  mod_4_eq_3_or_mod_4_eq_3
  rw  [neq,  h]
  rcases  this  with  h1  |  h1
  .  sorry
  .  sorry 
```

我们已经接近终点。给定一个素数集合 `s`，如果其中包含 3，我们需要讨论从该集合中移除 3 的结果。函数 `Finset.erase` 处理这个问题。

```py
example  (m  n  :  ℕ)  (s  :  Finset  ℕ)  (h  :  m  ∈  erase  s  n)  :  m  ≠  n  ∧  m  ∈  s  :=  by
  rwa  [mem_erase]  at  h

example  (m  n  :  ℕ)  (s  :  Finset  ℕ)  (h  :  m  ∈  erase  s  n)  :  m  ≠  n  ∧  m  ∈  s  :=  by
  simp  at  h
  assumption 
```

我们现在准备证明存在无限多个与 4 模 3 同余的素数。在下面填入缺失的部分。我们的解决方案在过程中使用了 `Nat.dvd_add_iff_left` 和 `Nat.dvd_sub'`。

```py
theorem  primes_mod_4_eq_3_infinite  :  ∀  n,  ∃  p  >  n,  Nat.Prime  p  ∧  p  %  4  =  3  :=  by
  by_contra  h
  push_neg  at  h
  rcases  h  with  ⟨n,  hn⟩
  have  :  ∃  s  :  Finset  Nat,  ∀  p  :  ℕ,  p.Prime  ∧  p  %  4  =  3  ↔  p  ∈  s  :=  by
  apply  ex_finset_of_bounded
  use  n
  contrapose!  hn
  rcases  hn  with  ⟨p,  ⟨pp,  p4⟩,  pltn⟩
  exact  ⟨p,  pltn,  pp,  p4⟩
  rcases  this  with  ⟨s,  hs⟩
  have  h₁  :  ((4  *  ∏  i  ∈  erase  s  3,  i)  +  3)  %  4  =  3  :=  by
  sorry
  rcases  exists_prime_factor_mod_4_eq_3  h₁  with  ⟨p,  pp,  pdvd,  p4eq⟩
  have  ps  :  p  ∈  s  :=  by
  sorry
  have  pne3  :  p  ≠  3  :=  by
  sorry
  have  :  p  ∣  4  *  ∏  i  ∈  erase  s  3,  i  :=  by
  sorry
  have  :  p  ∣  3  :=  by
  sorry
  have  :  p  =  3  :=  by
  sorry
  contradiction 
```

如果你设法完成了证明，恭喜你！这已经是一项正式化的重大成就。

## 5.4\. 更多归纳

在 第 5.2 节 中，我们看到了如何通过自然数的递归来定义阶乘函数。

```py
def  fac  :  ℕ  →  ℕ
  |  0  =>  1
  |  n  +  1  =>  (n  +  1)  *  fac  n 
```

我们还看到了如何使用 `induction'` 策略来证明定理。

```py
theorem  fac_pos  (n  :  ℕ)  :  0  <  fac  n  :=  by
  induction'  n  with  n  ih
  ·  rw  [fac]
  exact  zero_lt_one
  rw  [fac]
  exact  mul_pos  n.succ_pos  ih 
```

`induction` 策略（不带撇号）允许更结构化的语法。

```py
example  (n  :  ℕ)  :  0  <  fac  n  :=  by
  induction  n
  case  zero  =>
  rw  [fac]
  exact  zero_lt_one
  case  succ  n  ih  =>
  rw  [fac]
  exact  mul_pos  n.succ_pos  ih

example  (n  :  ℕ)  :  0  <  fac  n  :=  by
  induction  n  with
  |  zero  =>
  rw  [fac]
  exact  zero_lt_one
  |  succ  n  ih  =>
  rw  [fac]
  exact  mul_pos  n.succ_pos  ih 
```

如同往常，你可以悬停在 `induction` 关键字上以阅读文档。`zero` 和 `succ` 这两个案例的名称来自类型 ℕ 的定义。请注意，`succ` 案例允许你为归纳变量和归纳假设选择任何你想要的名称，这里分别是 `n` 和 `ih`。你甚至可以使用定义递归函数的相同符号来证明一个定理。

```py
theorem  fac_pos'  :  ∀  n,  0  <  fac  n
  |  0  =>  by
  rw  [fac]
  exact  zero_lt_one
  |  n  +  1  =>  by
  rw  [fac]
  exact  mul_pos  n.succ_pos  (fac_pos'  n) 
```

注意到缺少了 `:=`，冒号后面的 `∀ n`，每个案例中的 `by` 关键字，以及归纳调用 `fac_pos' n`。这就像定理是 `n` 的递归函数，在归纳步骤中我们进行递归调用。

这种定义风格非常灵活。Lean 的设计者内置了定义递归函数的复杂手段，这些手段也扩展到了归纳证明。例如，我们可以定义具有多个基准情况的斐波那契函数。

```py
@[simp]  def  fib  :  ℕ  →  ℕ
  |  0  =>  0
  |  1  =>  1
  |  n  +  2  =>  fib  n  +  fib  (n  +  1) 
```

`@[simp]` 注解意味着简化器将使用定义方程。你也可以通过编写 `rw [fib]` 来应用它们。下面将有助于为 `n + 2` 情况命名。

```py
theorem  fib_add_two  (n  :  ℕ)  :  fib  (n  +  2)  =  fib  n  +  fib  (n  +  1)  :=  rfl

example  (n  :  ℕ)  :  fib  (n  +  2)  =  fib  n  +  fib  (n  +  1)  :=  by  rw  [fib] 
```

使用 Lean 的递归函数表示法，你可以通过自然数上的归纳法进行证明，这些证明与 `fib` 的递归定义相呼应。以下示例提供了一个关于第 n 个斐波那契数的显式公式，该公式以黄金分割数 `φ` 及其共轭 `φ'` 为基础。我们必须告诉 Lean，我们预期我们的定义不会生成代码，因为实数上的算术运算是不可计算的。

```py
noncomputable  section

def  phi  :  ℝ  :=  (1  +  √5)  /  2
def  phi'  :  ℝ  :=  (1  -  √5)  /  2

theorem  phi_sq  :  phi²  =  phi  +  1  :=  by
  field_simp  [phi,  add_sq];  ring

theorem  phi'_sq  :  phi'²  =  phi'  +  1  :=  by
  field_simp  [phi',  sub_sq];  ring

theorem  fib_eq  :  ∀  n,  fib  n  =  (phi^n  -  phi'^n)  /  √5
  |  0  =>  by  simp
  |  1  =>  by  field_simp  [phi,  phi']
  |  n+2  =>  by  field_simp  [fib_eq,  pow_add,  phi_sq,  phi'_sq];  ring

end 
```

涉及斐波那契函数的归纳证明不必采用那种形式。以下我们重现了 `Mathlib` 中证明连续斐波那契数互质的证明。

```py
theorem  fib_coprime_fib_succ  (n  :  ℕ)  :  Nat.Coprime  (fib  n)  (fib  (n  +  1))  :=  by
  induction  n  with
  |  zero  =>  simp
  |  succ  n  ih  =>
  simp  only  [fib,  Nat.coprime_add_self_right]
  exact  ih.symm 
```

使用 Lean 的计算解释，我们可以评估斐波那契数列。

```py
#eval  fib  6
#eval  List.range  20  |>.map  fib 
```

`fib` 的直接实现计算效率低下。实际上，它的运行时间与其参数呈指数关系。（你应该思考一下为什么。）在 Lean 中，我们可以实现以下尾递归版本，其运行时间与 `n` 线性相关，并证明它计算的是相同的函数。

```py
def  fib'  (n  :  Nat)  :  Nat  :=
  aux  n  0  1
where  aux
  |  0,  x,  _  =>  x
  |  n+1,  x,  y  =>  aux  n  y  (x  +  y)

theorem  fib'.aux_eq  (m  n  :  ℕ)  :  fib'.aux  n  (fib  m)  (fib  (m  +  1))  =  fib  (n  +  m)  :=  by
  induction  n  generalizing  m  with
  |  zero  =>  simp  [fib'.aux]
  |  succ  n  ih  =>  rw  [fib'.aux,  ←fib_add_two,  ih,  add_assoc,  add_comm  1]

theorem  fib'_eq_fib  :  fib'  =  fib  :=  by
  ext  n
  erw  [fib',  fib'.aux_eq  0  n];  rfl

#eval  fib'  10000 
```

注意在 `fib'.aux_eq` 的证明中使用的 `generalizing` 关键字。它用于在归纳假设前插入 `∀ m`，这样在归纳步骤中，`m` 可以取不同的值。你可以逐步通过证明并检查在这种情况下，量词需要在归纳步骤中实例化为 `m + 1`。

注意也使用了 `erw`（表示“扩展重写”）而不是 `rw`。这是因为为了重写目标 `fib'.aux_eq`，`fib 0` 和 `fib 1` 必须分别简化为 `0` 和 `1`。`erw` 策略在展开定义以匹配参数方面比 `rw` 更激进。这并不总是好主意；在某些情况下，它可能会浪费大量时间，因此请谨慎使用 `erw`。

这里是 `generalizing` 关键字在 `Mathlib` 中另一个恒等式证明中使用的另一个例子。该恒等式的不正式证明可以在[这里](https://proofwiki.org/wiki/Fibonacci_Number_in_terms_of_Smaller_Fibonacci_Numbers)找到。我们提供了两种正式证明的变体。

```py
theorem  fib_add  (m  n  :  ℕ)  :  fib  (m  +  n  +  1)  =  fib  m  *  fib  n  +  fib  (m  +  1)  *  fib  (n  +  1)  :=  by
  induction  n  generalizing  m  with
  |  zero  =>  simp
  |  succ  n  ih  =>
  specialize  ih  (m  +  1)
  rw  [add_assoc  m  1  n,  add_comm  1  n]  at  ih
  simp  only  [fib_add_two,  Nat.succ_eq_add_one,  ih]
  ring

theorem  fib_add'  :  ∀  m  n,  fib  (m  +  n  +  1)  =  fib  m  *  fib  n  +  fib  (m  +  1)  *  fib  (n  +  1)
  |  _,  0  =>  by  simp
  |  m,  n  +  1  =>  by
  have  :=  fib_add'  (m  +  1)  n
  rw  [add_assoc  m  1  n,  add_comm  1  n]  at  this
  simp  only  [fib_add_two,  Nat.succ_eq_add_one,  this]
  ring 
```

作为练习，使用 `fib_add` 证明以下内容。

```py
example  (n  :  ℕ):  (fib  n)  ^  2  +  (fib  (n  +  1))  ^  2  =  fib  (2  *  n  +  1)  :=  by  sorry 
```

Lean 定义递归函数的机制足够灵活，允许任意递归调用，只要参数的复杂度根据某种已建立的度量减少。在下一个例子中，我们展示了每个自然数 `n ≠ 1` 都有一个素数因子，利用了以下事实：如果 `n` 非零且不是素数，它有一个更小的因子。（你可以检查 Mathlib 在 `Nat` 命名空间中有一个同名定理，尽管它的证明与这里给出的不同。）

```py
#check  (@Nat.not_prime_iff_exists_dvd_lt  :
  ∀  {n  :  ℕ},  2  ≤  n  →  (¬Nat.Prime  n  ↔  ∃  m,  m  ∣  n  ∧  2  ≤  m  ∧  m  <  n))

theorem  ne_one_iff_exists_prime_dvd  :  ∀  {n},  n  ≠  1  ↔  ∃  p  :  ℕ,  p.Prime  ∧  p  ∣  n
  |  0  =>  by  simpa  using  Exists.intro  2  Nat.prime_two
  |  1  =>  by  simp  [Nat.not_prime_one]
  |  n  +  2  =>  by
  have  hn  :  n+2  ≠  1  :=  by  omega
  simp  only  [Ne,  not_false_iff,  true_iff,  hn]
  by_cases  h  :  Nat.Prime  (n  +  2)
  ·  use  n+2,  h
  ·  have  :  2  ≤  n  +  2  :=  by  omega
  rw  [Nat.not_prime_iff_exists_dvd_lt  this]  at  h
  rcases  h  with  ⟨m,  mdvdn,  mge2,  -⟩
  have  :  m  ≠  1  :=  by  omega
  rw  [ne_one_iff_exists_prime_dvd]  at  this
  rcases  this  with  ⟨p,  primep,  pdvdm⟩
  use  p,  primep
  exact  pdvdm.trans  mdvdn 
```

这行 `rw [ne_one_iff_exists_prime_dvd] at this` 就像是一个魔术：我们正在使用我们正在证明的定理在其自己的证明中。使其工作的是归纳调用在 `m` 上实例化，当前情况是 `n + 2`，并且上下文有 `m < n + 2`。Lean 可以找到假设并使用它来证明归纳是良基的。Lean 在确定什么是递减方面相当擅长；在这种情况下，定理陈述中的 `n` 选择和小于关系是显然的。在更复杂的情况下，Lean 提供了提供此信息的机制。请参阅 Lean 参考手册中关于[良基递归](https://lean-lang.org/doc/reference/latest//Definitions/Recursive-Definitions/#well-founded-recursion)的部分。

有时，在证明中，你需要根据自然数 `n` 是否为零或后继来分情况讨论，在后继情况下不需要归纳假设。为此，你可以使用 `cases` 和 `rcases` 策略。

```py
theorem  zero_lt_of_mul_eq_one  (m  n  :  ℕ)  :  n  *  m  =  1  →  0  <  n  ∧  0  <  m  :=  by
  cases  n  <;>  cases  m  <;>  simp

example  (m  n  :  ℕ)  :  n*m  =  1  →  0  <  n  ∧  0  <  m  :=  by
  rcases  m  with  (_  |  m);  simp
  rcases  n  with  (_  |  n)  <;>  simp 
```

这是一个有用的技巧。通常，你有一个关于自然数 `n` 的定理，其中零的情况很容易处理。如果你对 `n` 进行情况分析并快速处理零的情况，你将剩下原始目标，只是将 `n` 替换为 `n + 1`*。
