# 1. 简介

> 原文：[`leanprover-community.github.io/mathematics_in_lean/C01_Introduction.html`](https://leanprover-community.github.io/mathematics_in_lean/C01_Introduction.html)

*Lean 中的数学* **   1. 简介

+   查看页面源代码

* * *

## 1.1. 开始

本书的目标是教你使用 Lean 4 交互式证明辅助工具形式化数学。它假设你了解一些数学知识，但不需要太多。虽然我们将涵盖从数论到测度理论和分析的例子，但我们将专注于这些领域的初阶方面，希望如果你不熟悉它们，你可以在学习过程中掌握它们。我们也不预设任何关于形式方法的背景知识。形式化可以看作是一种计算机编程：我们将用 Lean 可以理解的有组织的语言（类似于编程语言）编写数学定义、定理和证明。作为回报，Lean 提供反馈和信息，解释表达式并保证它们是良好形成的，并最终证明我们证明的正确性。

你可以从 [Lean 项目页面](https://leanprover.github.io) 和 [Lean 社区网页](https://leanprover-community.github.io/) 了解更多关于 Lean 的信息。本教程基于 Lean 的庞大且不断增长的库 *Mathlib*。我们还强烈建议如果你还没有的话，加入 [Lean Zulip 在线聊天组](https://leanprover.zulipchat.com/)。在那里，你会发现一个充满活力和欢迎的 Lean 爱好者社区，他们乐于回答问题和提供精神支持。

虽然你可以在网上阅读这本书的 pdf 或 html 版本，但它旨在进行交互式阅读，在 VS Code 编辑器中运行 Lean。要开始：

1.  按照以下[安装说明](https://leanprover-community.github.io/get_started.html)安装 Lean 4 和 VS Code。

1.  确保你已经安装了 [git](https://git-scm.com/)。

1.  按照以下[说明](https://leanprover-community.github.io/install/project.html#working-on-an-existing-project)获取 `mathematics_in_lean` 仓库并在 VS Code 中打开它。

1.  本书中的每一节都有一个相关的 Lean 文件，其中包含示例和练习。你可以在 `MIL` 文件夹中找到它们，按章节组织。我们强烈建议复制该文件夹并在这个副本中进行实验和练习。这样，原始文件保持不变，也使得在更改时更新仓库更容易（见下文）。你可以将副本命名为 `my_files` 或你想要的任何名称，并使用它来创建你自己的 Lean 文件。

在那个阶段，你可以按照以下方式在 VS Code 的侧面板中打开教科书：

1.  输入 `ctrl-shift-P` (`command-shift-P` 在 macOS 上)。

1.  在出现的栏中输入 `Lean 4: Docs: Show Documentation Resources`，然后按回车键。（你可以在它被菜单中的高亮显示后按回车键来选择它。）

1.  在打开的窗口中，点击 `Mathematics in Lean`。

或者，你可以在云中使用 Lean 和 VS Code，使用 [Gitpod](https://gitpod.io/)。你可以在 Github 上的 Mathematics in Lean [项目页面](https://github.com/leanprover-community/mathematics_in_lean)上找到如何做到这一点的说明。我们仍然建议使用上面描述的 MIL 文件夹的副本进行工作。

这本教科书和相关的仓库仍在进行中。你可以在 `mathematics_in_lean` 文件夹内通过输入 `git pull` 后跟 `lake exe cache get` 来更新仓库。（这假设你没有更改 `MIL` 文件夹的内容，这就是我们建议制作副本的原因。）

我们希望你在阅读包含解释、说明和提示的教科书的同时，在 `MIL` 文件夹中完成练习。文本通常会包括例子，就像这个例子一样：

```py
#eval  "Hello, World!" 
```

你应该能够在相关的 Lean 文件中找到相应的例子。如果你点击该行，VS Code 将在 `Lean Goal` 窗口中显示 Lean 的反馈，如果你将鼠标悬停在 `#eval` 命令上，VS Code 将在弹出窗口中显示 Lean 对此命令的响应。你被鼓励编辑文件并尝试自己的例子。

此外，这本书还提供了许多具有挑战性的练习供你尝试。不要匆匆略过这些！Lean 是关于 *交互式* 做数学，而不仅仅是阅读它。完成练习是体验的核心。你不必全部完成；当你觉得你已经掌握了相关技能时，你可以自由地继续前进。你总是可以比较你的解决方案与每个部分相关的 `solutions` 文件夹中的解决方案。

## 1.2. 概述

简而言之，Lean 是一个用于在称为 *依赖类型理论* 的形式语言中构建复杂表达式的工具。

每个表达式都有一个 *类型*，你可以使用 #check 命令来打印它。一些表达式的类型像 ℕ 或 ℕ → ℕ。这些都是数学对象。

```py
#check  2  +  2

def  f  (x  :  ℕ)  :=
  x  +  3

#check  f 
```

一些表达式的类型是 Prop。这些都是数学陈述。

```py
#check  2  +  2  =  4

def  FermatLastTheorem  :=
  ∀  x  y  z  n  :  ℕ,  n  >  2  ∧  x  *  y  *  z  ≠  0  →  x  ^  n  +  y  ^  n  ≠  z  ^  n

#check  FermatLastTheorem 
```

一些表达式的类型是 P，其中 P 本身有类型 Prop。这样的表达式是命题 P 的证明。

```py
theorem  easy  :  2  +  2  =  4  :=
  rfl

#check  easy

theorem  hard  :  FermatLastTheorem  :=
  sorry

#check  hard 
```

如果你设法构造了一个类型为 `FermatLastTheorem` 的表达式，并且 Lean 接受它作为该类型的项，那么你已经做了非常令人印象深刻的事情。（使用 `sorry` 是作弊，Lean 知道这一点。）所以现在你知道游戏规则了。剩下要学的是规则。

本书与配套教程 [Theorem Proving in Lean](https://leanprover.github.io/theorem_proving_in_lean4/) 相辅相成，后者提供了对 Lean 的底层逻辑框架和核心语法的更全面介绍。*Theorem Proving in Lean* 适合那些喜欢在使用新洗碗机之前从头到尾阅读用户手册的人。如果你是那种喜欢按下 *启动* 按钮，稍后再找出如何激活去污功能的人，那么从这里开始，并在需要时参考 *Theorem Proving in Lean* 会更有意义。

另一个区分 *Mathematics in Lean* 和 *Theorem Proving in Lean* 的特点是，在这里我们更加重视 *tactics* 的使用。鉴于我们试图构建复杂的表达式，Lean 提供了两种方法来实现这一点：我们可以写下表达式本身（即其合适的文本描述），或者我们可以向 Lean 提供如何构建它们的 *指令*。例如，以下表达式代表了一个证明，即如果 `n` 是偶数，那么 `m * n` 也是偶数：

```py
example  :  ∀  m  n  :  Nat,  Even  n  →  Even  (m  *  n)  :=  fun  m  n  ⟨k,  (hk  :  n  =  k  +  k)⟩  ↦
  have  hmn  :  m  *  n  =  m  *  k  +  m  *  k  :=  by  rw  [hk,  mul_add]
  show  ∃  l,  m  *  n  =  l  +  l  from  ⟨_,  hmn⟩ 
```

*证明项* 可以压缩成一行：

```py
example  :  ∀  m  n  :  Nat,  Even  n  →  Even  (m  *  n)  :=
fun  m  n  ⟨k,  hk⟩  ↦  ⟨m  *  k,  by  rw  [hk,  mul_add]⟩ 
```

下面是同一定理的 *策略风格* 证明，其中以 `--` 开头的行是注释，因此 Lean 会忽略它们：

```py
example  :  ∀  m  n  :  Nat,  Even  n  →  Even  (m  *  n)  :=  by
  -- Say `m` and `n` are natural numbers, and assume `n = 2 * k`.
  rintro  m  n  ⟨k,  hk⟩
  -- We need to prove `m * n` is twice a natural number. Let's show it's twice `m * k`.
  use  m  *  k
  -- Substitute for `n`,
  rw  [hk]
  -- and now it's obvious.
  ring 
```

当你在 VS Code 中输入这样证明的每一行时，Lean 会在一个单独的窗口中显示 *证明状态*，告诉你你已经建立了哪些事实，以及还需要证明定理的任务。你可以通过逐行执行来重放证明，因为 Lean 会继续显示光标所在点的证明状态。在这个例子中，你会发现证明的第一行引入了 `m` 和 `n`（如果我们想的话，我们可以在那时将它们重命名），并且还将假设 `Even n` 分解为 `k` 和 `n = 2 * k`。第二行 `use m * k` 声明我们将通过展示 `m * n = 2 * (m * k)` 来证明 `m * n` 是偶数。下一行使用 `rw` 策略将目标中的 `n` 替换为 `2 * k`（`rw` 代表“重写”），而 `ring` 策略解决了由此产生的目标 `m * (2 * k) = 2 * (m * k)`。

以小步骤构建证明并逐步提供反馈的能力非常强大。因此，策略证明通常比证明项更容易、更快地编写。两者之间没有明显的区别：策略证明可以插入到证明项中，就像我们在上面的例子中使用短语 `by rw [hk, mul_add]` 一样。我们还将看到，相反，在策略证明的中间插入一个简短的证明项通常很有用。尽管如此，在这本书中，我们将重点介绍策略的使用。

在我们的例子中，策略证明也可以简化为一行：

```py
example  :  ∀  m  n  :  Nat,  Even  n  →  Even  (m  *  n)  :=  by
  rintro  m  n  ⟨k,  hk⟩;  use  m  *  k;  rw  [hk];  ring 
```

在这里，我们使用了策略来执行小的证明步骤。但它们也可以提供实质性的自动化，并证明更长的计算和更大的推理步骤是合理的。例如，我们可以使用 Lean 的简化器，并应用特定的规则来简化关于偶性的陈述，从而自动证明我们的定理。

```py
example  :  ∀  m  n  :  Nat,  Even  n  →  Even  (m  *  n)  :=  by
  intros;  simp  [*,  parity_simps] 
```

两个介绍之间的另一个重大区别是，*Lean 中的定理证明*仅依赖于核心 Lean 及其内置策略，而*Lean 中的数学*建立在 Lean 强大且不断增长的库*Mathlib*之上。因此，我们可以向您展示如何使用库中的一些数学对象和定理，以及一些非常有用的策略。这本书的目的不是用作库的完整概述；[社区](https://leanprover-community.github.io/)网页包含了广泛的文档。相反，我们的目标是向您介绍支撑这种形式化的思维方式，并指出基本的入门点，以便您能够舒适地浏览库并自行查找内容。

交互式定理证明可能会令人沮丧，学习曲线很陡峭。但 Lean 社区对新来者非常欢迎，人们全天候在[Lean Zulip 聊天组](https://leanprover.zulipchat.com/)上提供帮助，回答问题。我们希望在那里见到您，并确信不久您也将能够回答这样的问题并为*Mathlib*的发展做出贡献。

因此，这是您的任务，如果您选择接受它：深入其中，尝试练习，带着问题来 Zulip，享受乐趣。但请提前警告：交互式定理证明将挑战您以全新的方式思考数学和数学推理。您的生活可能永远不再一样。

*致谢。*我们感谢 Gabriel Ebner 为在 VS Code 中运行此教程的基础设施，以及 Kim Morrison 和 Mario Carneiro 将其从 Lean 4 迁移过来提供的帮助。我们还感谢 Takeshi Abe、Julian Berman、Alex Best、Thomas Browning、Bulwi Cha、Hanson Char、Bryan Gin-ge Chen、Steven Clontz、Mauricio Collaris、Johan Commelin、Mark Czubin、Alexandru Duca、Pierpaolo Frasa、Denis Gorbachev、Winston de Greef、Mathieu Guay-Paquet、Marc Huisinga、Benjamin Jones、Julian Külshammer、Victor Liu、Jimmy Lu、Martin C. Martin、Giovanni Mascellani、John McDowell、Joseph McKinsey、Bhavik Mehta、Isaiah Mindich、Kabelo Moiloa、Hunter Monroe、Pietro Monticone、Oliver Nash、Emanuelle Natale、Filippo A. E. Nuccio、Pim Otte、Bartosz Piotrowski、Nicolas Rolland、Keith Rush、Yannick Seurin、Guilherme Silva、Bernardo Subercaseaux、Pedro Sánchez Terraf、Matthew Toohey、Alistair Tucker、Floris van Doorn、Eric Wieser 和其他人的帮助。我们的工作部分得到了 Hoskinson 形式数学中心的支持。上一页 下一页

* * *

© 版权所有 2020-2025，Jeremy Avigad，Patrick Massot。文本许可协议为 CC BY 4.0。

使用[Sphinx](https://www.sphinx-doc.org/)和[Read the Docs](https://readthedocs.org)提供的[主题](https://github.com/readthedocs/sphinx_rtd_theme)构建。## 1.1. 开始使用

本书的目标是教会您使用 Lean 4 交互式证明辅助工具形式化数学。它假设您了解一些数学知识，但不需要太多。尽管我们将涵盖从数论到测度理论和分析的例子，但我们将重点介绍这些领域的初等方面，希望如果您不熟悉，您可以在学习过程中掌握它们。我们也不预设任何关于形式方法的背景知识。形式化可以看作是一种计算机编程：我们将使用 Lean 可以理解的有组织的语言（类似于编程语言）编写数学定义、定理和证明。作为回报，Lean 提供反馈和信息，解释表达式并保证它们是良好形成的，并最终证明我们的证明是正确的。

您可以从[Lean 项目页面](https://leanprover.github.io)和[Lean 社区网页](https://leanprover-community.github.io/)了解更多关于 Lean 的信息。本教程基于 Lean 的大型且不断增长的库*Mathlib*。我们还强烈建议如果您还没有，加入[Lean Zulip 在线聊天组](https://leanprover.zulipchat.com/)。在那里，您会发现一个充满活力和欢迎的 Lean 爱好者社区，他们乐于回答问题和提供道德支持。

虽然您可以在网上阅读这本书的 pdf 或 html 版本，但它旨在交互式阅读，在 VS Code 编辑器中运行 Lean。要开始：

1.  按照这些[安装说明](https://leanprover-community.github.io/get_started.html)安装 Lean 4 和 VS Code。

1.  确保您已安装[git](https://git-scm.com/)。

1.  按照这些[说明](https://leanprover-community.github.io/install/project.html#working-on-an-existing-project)获取`mathematics_in_lean`存储库，并在 VS Code 中打开它。

1.  本书中的每一节都有一个相关的 Lean 文件，包含示例和练习。您可以在`MIL`文件夹中找到它们，按章节组织。我们强烈建议复制该文件夹，并在其中进行实验和练习。这样，原始文件保持不变，也使得在更改时更新存储库更容易（见下文）。您可以将其命名为`my_files`或您想要的任何名称，并使用它来创建您自己的 Lean 文件。

到那时，您可以在 VS Code 的侧面板中打开教科书，如下所示：

1.  输入`ctrl-shift-P`（在 macOS 中为`command-shift-P`）。

1.  在出现的栏中输入`Lean 4: Docs: Show Documentation Resources`，然后按回车键。（您可以在它被菜单中的高亮显示后立即按回车键选择它。）

1.  在打开的窗口中，点击 `Mathematics in Lean`。

或者，你可以在云中使用 Lean 和 VS Code，使用 [Gitpod](https://gitpod.io/)。你可以在 Github 上的 Mathematics in Lean [项目页面](https://github.com/leanprover-community/mathematics_in_lean)上找到如何做到这一点的说明。我们仍然建议按照上述方法在 `MIL` 文件夹的副本中工作。

这本教科书和相关的仓库仍在进行中。你可以在 `mathematics_in_lean` 文件夹内通过输入 `git pull` 然后跟 `lake exe cache get` 来更新仓库。（这假设你没有更改 `MIL` 文件夹的内容，这就是我们建议制作副本的原因。）

我们希望你在阅读包含解释、指令和提示的教科书的同时，在 `MIL` 文件夹中完成练习。文本通常会包括示例，就像这个例子一样：

```py
#eval  "Hello, World!" 
```

你应该能在相关的 Lean 文件中找到相应的示例。如果你点击该行，VS Code 将在 `Lean Goal` 窗口中显示 Lean 的反馈，如果你将鼠标悬停在 `#eval` 命令上，VS Code 将在弹出窗口中显示 Lean 对此命令的响应。鼓励你编辑文件并尝试自己的示例。

此外，本书还提供了许多具有挑战性的练习供你尝试。不要匆匆而过！Lean 是关于 *交互式* 做数学，而不仅仅是阅读它。完成练习是体验的核心。你不必全部完成；当你觉得你已经掌握了相关技能时，可以自由地继续前进。你总是可以比较你的解决方案与每个部分相关的 `solutions` 文件夹中的解决方案。

## 1.2. 概述

简而言之，Lean 是一个用于在名为 *依赖类型理论* 的形式语言中构建复杂表达式的工具。

每个表达式都有一个 *类型*，你可以使用 #check 命令来打印它。一些表达式具有类型如 ℕ 或 ℕ → ℕ。这些是数学对象。

```py
#check  2  +  2

def  f  (x  :  ℕ)  :=
  x  +  3

#check  f 
```

一些表达式具有类型 Prop。这些是数学陈述。

```py
#check  2  +  2  =  4

def  FermatLastTheorem  :=
  ∀  x  y  z  n  :  ℕ,  n  >  2  ∧  x  *  y  *  z  ≠  0  →  x  ^  n  +  y  ^  n  ≠  z  ^  n

#check  FermatLastTheorem 
```

一些表达式具有类型 P，其中 P 本身具有类型 Prop。这样的表达式是命题 P 的证明。

```py
theorem  easy  :  2  +  2  =  4  :=
  rfl

#check  easy

theorem  hard  :  FermatLastTheorem  :=
  sorry

#check  hard 
```

如果你成功构造了一个类型为 `FermatLastTheorem` 的表达式，并且 Lean 接受它作为该类型的项，那么你已经做了非常令人印象深刻的事情。（使用 `sorry` 是作弊，而 Lean 知道这一点。）所以现在你知道游戏规则了。剩下要学的就是规则了。

这本书与一个配套教程 [Lean 中的定理证明](https://leanprover.github.io/theorem_proving_in_lean4/) 相辅相成，该教程提供了对 Lean 的底层逻辑框架和核心语法的更全面介绍。*Lean 中的定理证明* 是为那些喜欢在使用新洗碗机之前从头到尾阅读用户手册的人准备的。如果你是那种喜欢按下 *开始* 按钮，稍后再找出如何激活去污功能的人，那么从这里开始，并在需要时参考 *Lean 中的定理证明* 会更有意义。

另一个将 *Lean 中的数学* 与 *Lean 中的定理证明* 区分开来的特点是，在这里我们更加重视 *策略* 的使用。鉴于我们试图构建复杂的表达式，Lean 提供了两种方法：我们可以写下表达式本身（即其合适的文本描述），或者我们可以向 Lean 提供如何构建它们的 *指令*。例如，以下表达式代表了一个证明，即如果 `n` 是偶数，那么 `m * n` 也是偶数：

```py
example  :  ∀  m  n  :  Nat,  Even  n  →  Even  (m  *  n)  :=  fun  m  n  ⟨k,  (hk  :  n  =  k  +  k)⟩  ↦
  have  hmn  :  m  *  n  =  m  *  k  +  m  *  k  :=  by  rw  [hk,  mul_add]
  show  ∃  l,  m  *  n  =  l  +  l  from  ⟨_,  hmn⟩ 
```

*证明项*可以被压缩成一行：

```py
example  :  ∀  m  n  :  Nat,  Even  n  →  Even  (m  *  n)  :=
fun  m  n  ⟨k,  hk⟩  ↦  ⟨m  *  k,  by  rw  [hk,  mul_add]⟩ 
```

下面是一个相反的 *策略风格* 的同一定理证明，其中以 `--` 开头的行是注释，因此 Lean 会忽略它们：

```py
example  :  ∀  m  n  :  Nat,  Even  n  →  Even  (m  *  n)  :=  by
  -- Say `m` and `n` are natural numbers, and assume `n = 2 * k`.
  rintro  m  n  ⟨k,  hk⟩
  -- We need to prove `m * n` is twice a natural number. Let's show it's twice `m * k`.
  use  m  *  k
  -- Substitute for `n`,
  rw  [hk]
  -- and now it's obvious.
  ring 
```

当你在 VS Code 中输入这样一条证明的每一行时，Lean 会在一个单独的窗口中显示 *证明状态*，告诉你你已经建立了哪些事实，以及还需要证明定理的任务。你可以通过逐行回放证明，因为 Lean 会继续显示光标所在点的证明状态。在这个例子中，你会看到证明的第一行引入了 `m` 和 `n`（如果我们想的话，我们可以在那时给它们重新命名），并且将假设 `Even n` 分解为 `k` 和 `n = 2 * k` 的假设。第二行 `use m * k` 声明我们将通过展示 `m * n = 2 * (m * k)` 来证明 `m * n` 是偶数。下一行使用 `rw` 策略将目标中的 `n` 替换为 `2 * k`（`rw` 代表“重写”），而 `ring` 策略解决了由此产生的目标 `m * (2 * k) = 2 * (m * k)`。

能够以小步骤构建证明并得到增量反馈的能力非常强大。因此，策略证明通常比证明项更容易和更快地编写。两者之间没有明显的区别：策略证明可以插入到证明项中，就像我们在上面的例子中用短语 `by rw [hk, mul_add]` 所做的那样。我们还将看到，相反地，在策略证明的中间插入一个简短的证明项通常是有用的。尽管如此，在这本书中，我们将侧重于策略的使用。

在我们的例子中，策略证明也可以简化为一行：

```py
example  :  ∀  m  n  :  Nat,  Even  n  →  Even  (m  *  n)  :=  by
  rintro  m  n  ⟨k,  hk⟩;  use  m  *  k;  rw  [hk];  ring 
```

在这里，我们使用了策略来执行小的证明步骤。但它们也可以提供实质性的自动化，并证明更长的计算和更大的推理步骤。例如，我们可以使用 Lean 的简化器，并应用特定的规则来简化关于偶数的陈述，从而自动证明我们的定理。

```py
example  :  ∀  m  n  :  Nat,  Even  n  →  Even  (m  *  n)  :=  by
  intros;  simp  [*,  parity_simps] 
```

两个介绍之间的另一个重大区别是，*Lean 中的定理证明*仅依赖于核心 Lean 及其内置策略，而*Lean 中的数学*建立在 Lean 强大且不断增长的库*Mathlib*之上。因此，我们可以向你展示如何使用库中的一些数学对象和定理，以及一些非常有用的策略。这本书的目的不是用作库的完整概述；[社区](https://leanprover-community.github.io/)网页包含了广泛的文档。相反，我们的目标是介绍支撑这种形式化的思维方式，并指出基本的入门点，以便你能够舒适地浏览库并自己找到所需的内容。

交互式定理证明可能会让人感到沮丧，学习曲线也很陡峭。但 Lean 社区对新来者非常欢迎，人们全天候在[Lean Zulip 聊天组](https://leanprover.zulipchat.com/)上提供帮助，解答问题。我们希望在那里见到你，并且确信不久之后你也能回答这样的问题，并为*Mathlib*的发展做出贡献。

所以，这里是你可以选择接受的任务：深入其中，尝试练习，带着问题来 Zulip，享受乐趣。但请提前警告：交互式定理证明将挑战你以全新的方式思考数学和数学推理。你的生活可能永远不再一样。

*致谢。*我们感谢 Gabriel Ebner 为在 VS Code 中运行此教程的基础设施，以及 Kim Morrison 和 Mario Carneiro 将其从 Lean 4 迁移过来提供的帮助。我们还感谢 Takeshi Abe、Julian Berman、Alex Best、Thomas Browning、Bulwi Cha、Hanson Char、Bryan Gin-ge Chen、Steven Clontz、Mauricio Collaris、Johan Commelin、Mark Czubin、Alexandru Duca、Pierpaolo Frasa、Denis Gorbachev、Winston de Greef、Mathieu Guay-Paquet、Marc Huisinga、Benjamin Jones、Julian Külshammer、Victor Liu、Jimmy Lu、Martin C. Martin、Giovanni Mascellani、John McDowell、Joseph McKinsey、Bhavik Mehta、Isaiah Mindich、Kabelo Moiloa、Hunter Monroe、Pietro Monticone、Oliver Nash、Emanuelle Natale、Filippo A. E. Nuccio、Pim Otte、Bartosz Piotrowski、Nicolas Rolland、Keith Rush、Yannick Seurin、Guilherme Silva、Bernardo Subercaseaux、Pedro Sánchez Terraf、Matthew Toohey、Alistair Tucker、Floris van Doorn、Eric Wieser 和其他人的帮助。我们的工作得到了 Hoskinson 形式数学中心的部分支持。

## 1.1\. 开始

本书的目标是教会你使用 Lean 4 交互式证明辅助工具来形式化数学。它假设你了解一些数学知识，但不需要太多。尽管我们将涵盖从数论到测度论和分析的例子，但我们将专注于这些领域的初阶方面，希望如果你不熟悉它们，你可以在学习过程中掌握它们。我们也不预设任何关于形式方法的背景知识。形式化可以看作是一种计算机编程：我们将用 Lean 可以理解的有组织语言（类似于编程语言）来编写数学定义、定理和证明。作为回报，Lean 提供反馈和信息，解释表达式并保证它们是良好形成的，并最终证明我们证明的正确性。

你可以从[Lean 项目页面](https://leanprover.github.io)和[Lean 社区网页](https://leanprover-community.github.io/)了解更多关于 Lean 的信息。本教程基于 Lean 的庞大且不断增长的库 *Mathlib*。我们还强烈建议如果你还没有的话，加入[Lean Zulip 在线聊天组](https://leanprover.zulipchat.com/)。在那里，你会发现一个充满活力和欢迎的 Lean 爱好者社区，乐于回答问题和提供道德支持。

虽然你可以在网上阅读本书的 pdf 或 html 版本，但它旨在以交互式方式阅读，在 VS Code 编辑器中运行 Lean。要开始：

1.  按照这些[安装说明](https://leanprover-community.github.io/get_started.html)安装 Lean 4 和 VS Code。

1.  确保你已经安装了 [git](https://git-scm.com/)。

1.  按照这些[说明](https://leanprover-community.github.io/install/project.html#working-on-an-existing-project)获取 `mathematics_in_lean` 仓库并在 VS Code 中打开它。

1.  本书中的每一节都有一个相关的 Lean 文件，包含示例和练习。你可以在 `MIL` 文件夹中找到它们，按章节组织。我们强烈建议复制该文件夹，并在该副本中进行实验和练习。这样，原始文件保持不变，也使得在它发生变化时更新仓库更容易（见下文）。你可以将副本命名为 `my_files` 或你想要的任何名称，并使用它来创建你自己的 Lean 文件。

在这一点上，你可以按照以下方式在 VS Code 的侧面板中打开教科书：

1.  输入 `ctrl-shift-P`（在 macOS 上为 `command-shift-P`）。

1.  在出现的栏中输入 `Lean 4: Docs: Show Documentation Resources`，然后按回车键。（你可以按回车键选择它，一旦它在菜单中高亮显示。）

1.  在打开的窗口中，点击 `Mathematics in Lean`。

或者，您可以在云中运行 Lean 和 VS Code，使用[Gitpod](https://gitpod.io/)。您可以在 GitHub 上的 Mathematics in Lean [项目页面](https://github.com/leanprover-community/mathematics_in_lean)上找到如何做到这一点的说明。我们仍然建议按照上述方法在 MIL 文件夹的副本中工作。

这本教科书和相关的仓库仍在不断完善中。您可以通过在`mathematics_in_lean`文件夹内输入`git pull`命令后跟`lake exe cache get`来更新仓库。（这假设您没有更改`MIL`文件夹的内容，这就是我们建议制作副本的原因。）

我们希望您在阅读包含解释、说明和提示的教科书的同时，在`MIL`文件夹中完成练习。文本通常会包含示例，例如这个：

```py
#eval  "Hello, World!" 
```

您应该能够在相关的 Lean 文件中找到相应的示例。如果您点击该行，VS Code 将在`Lean Goal`窗口中显示 Lean 的反馈，如果您将鼠标悬停在`#eval`命令上，VS Code 将在弹出窗口中显示 Lean 对此命令的响应。您被鼓励编辑文件并尝试自己的示例。

此外，本书还提供了许多具有挑战性的练习供您尝试。不要匆匆而过！Lean 是关于*交互式*做数学的，而不仅仅是阅读它。完成练习是体验的核心。您不必全部完成；当您觉得您已经掌握了相关技能时，您可以自由地继续前进。您总是可以比较您在每个部分的关联`solutions`文件夹中的解决方案。

## 1.2. 概述

简单来说，Lean 是一个在称为*依赖类型理论*的正式语言中构建复杂表达式的工具。

每个表达式都有一个*类型*，您可以使用#check 命令来打印它。一些表达式具有类型如ℕ或ℕ → ℕ。这些是数学对象。

```py
#check  2  +  2

def  f  (x  :  ℕ)  :=
  x  +  3

#check  f 
```

一些表达式具有类型 Prop。这些都是数学陈述。

```py
#check  2  +  2  =  4

def  FermatLastTheorem  :=
  ∀  x  y  z  n  :  ℕ,  n  >  2  ∧  x  *  y  *  z  ≠  0  →  x  ^  n  +  y  ^  n  ≠  z  ^  n

#check  FermatLastTheorem 
```

一些表达式具有类型 P，其中 P 本身具有类型 Prop。这样的表达式是命题 P 的证明。

```py
theorem  easy  :  2  +  2  =  4  :=
  rfl

#check  easy

theorem  hard  :  FermatLastTheorem  :=
  sorry

#check  hard 
```

如果您设法构造了一个类型为`FermatLastTheorem`的表达式，并且 Lean 接受它作为该类型的项，那么您已经做了非常了不起的事情。（使用`sorry`是作弊，Lean 也知道。）所以现在您知道了游戏规则。剩下要学习的就是规则了。

这本书是配套教程 [Lean 中的定理证明](https://leanprover.github.io/theorem_proving_in_lean4/) 的补充，后者提供了对 Lean 的底层逻辑框架和核心语法的更全面介绍。*Lean 中的定理证明* 是为那些喜欢在使用新洗碗机之前从头到尾阅读用户手册的人准备的。如果你是那种喜欢按下 *开始* 按钮，稍后再找出如何激活去污功能的人，那么从这里开始，并在需要时参考 *Lean 中的定理证明* 会更有意义。

另一个将 *Lean 中的数学* 与 *Lean 中的定理证明* 区分开来的因素是，在这里我们更加重视 *策略* 的使用。鉴于我们试图构建复杂的表达式，Lean 提供了两种方法：我们可以写下表达式本身（即其合适的文本描述），或者我们可以向 Lean 提供如何构建它们的 *指令*。例如，以下表达式代表了一个证明，即如果 `n` 是偶数，那么 `m * n` 也是偶数：

```py
example  :  ∀  m  n  :  Nat,  Even  n  →  Even  (m  *  n)  :=  fun  m  n  ⟨k,  (hk  :  n  =  k  +  k)⟩  ↦
  have  hmn  :  m  *  n  =  m  *  k  +  m  *  k  :=  by  rw  [hk,  mul_add]
  show  ∃  l,  m  *  n  =  l  +  l  from  ⟨_,  hmn⟩ 
```

*证明项* 可以压缩为单行：

```py
example  :  ∀  m  n  :  Nat,  Even  n  →  Even  (m  *  n)  :=
fun  m  n  ⟨k,  hk⟩  ↦  ⟨m  *  k,  by  rw  [hk,  mul_add]⟩ 
```

以下是对同一定理的 *策略风格* 证明，其中以 `--` 开头的行是注释，因此被 Lean 忽略：

```py
example  :  ∀  m  n  :  Nat,  Even  n  →  Even  (m  *  n)  :=  by
  -- Say `m` and `n` are natural numbers, and assume `n = 2 * k`.
  rintro  m  n  ⟨k,  hk⟩
  -- We need to prove `m * n` is twice a natural number. Let's show it's twice `m * k`.
  use  m  *  k
  -- Substitute for `n`,
  rw  [hk]
  -- and now it's obvious.
  ring 
```

当你在 VS Code 中输入这样一条证明的每一行时，Lean 会在一个单独的窗口中显示 *证明状态*，告诉你你已经建立了哪些事实，以及还需要证明定理的任务。你可以通过逐行执行来重放证明，因为 Lean 会继续显示光标所在点的证明状态。在这个例子中，你会发现证明的第一行引入了 `m` 和 `n`（如果我们想的话，我们可以在那时将它们重命名），并且将假设 `Even n` 分解为 `k` 和 `n = 2 * k`。第二行 `use m * k` 声明我们将通过展示 `m * n = 2 * (m * k)` 来证明 `m * n` 是偶数。下一行使用 `rw` 策略将目标中的 `n` 替换为 `2 * k`（`rw` 代表“重写”），而 `ring` 策略解决了由此产生的目标 `m * (2 * k) = 2 * (m * k)`。

使用小步骤构建证明并得到增量反馈的能力非常强大。因此，策略证明通常比证明项更容易、更快地编写。两者之间没有明显的区别：策略证明可以插入到证明项中，就像我们在上面的例子中用短语 `by rw [hk, mul_add]` 所做的那样。我们还将看到，相反地，在策略证明的中间插入一个简短的证明项通常是有用的。尽管如此，在这本书中，我们将重点介绍策略的使用。

在我们的例子中，策略证明也可以简化为单行：

```py
example  :  ∀  m  n  :  Nat,  Even  n  →  Even  (m  *  n)  :=  by
  rintro  m  n  ⟨k,  hk⟩;  use  m  *  k;  rw  [hk];  ring 
```

在这里，我们使用了策略来执行小的证明步骤。但它们也可以提供实质性的自动化，并证明更长的计算和更大的推理步骤。例如，我们可以使用 Lean 的简化器，并应用特定的规则来简化关于偶数的陈述，从而自动证明我们的定理。

```py
example  :  ∀  m  n  :  Nat,  Even  n  →  Even  (m  *  n)  :=  by
  intros;  simp  [*,  parity_simps] 
```

两个介绍之间的另一个重大区别是，*Lean 中的定理证明*仅依赖于核心 Lean 及其内置策略，而*Lean 中的数学*建立在 Lean 强大且不断增长的库*Mathlib*之上。因此，我们可以向您展示如何使用库中的某些数学对象和定理，以及一些非常有用的策略。这本书的目的不是用作库的完整概述；[社区](https://leanprover-community.github.io/)网页包含广泛的文档。我们的目标是通过介绍支撑形式化的思维方式，指出基本的入门点，让您能够舒适地浏览库并自行查找内容。

交互式定理证明可能会让人感到沮丧，学习曲线很陡峭。但 Lean 社区对新来者非常欢迎，人们全天候在[Lean Zulip 聊天组](https://leanprover.zulipchat.com/)上提供帮助。我们希望在那里见到你，并且毫无疑问，不久你也将能够回答这样的问题，并为*Mathlib*的发展做出贡献。

所以，如果你选择接受，这就是你的任务：深入其中，尝试练习，带着问题来 Zulip，享受乐趣。但请提前警告：交互式定理证明将挑战你以全新的方式思考数学和数学推理。你的生活可能永远都不会一样。

*致谢。* 我们感谢 Gabriel Ebner 为在 VS Code 中运行此教程的基础设施建设，感谢 Kim Morrison 和 Mario Carneiro 帮助将其从 Lean 4 迁移过来。我们还感谢 Takeshi Abe、Julian Berman、Alex Best、Thomas Browning、Bulwi Cha、Hanson Char、Bryan Gin-ge Chen、Steven Clontz、Mauricio Collaris、Johan Commelin、Mark Czubin、Alexandru Duca、Pierpaolo Frasa、Denis Gorbachev、Winston de Greef、Mathieu Guay-Paquet、Marc Huisinga、Benjamin Jones、Julian Külshammer、Victor Liu、Jimmy Lu、Martin C. Martin、Giovanni Mascellani、John McDowell、Joseph McKinsey、Bhavik Mehta、Isaiah Mindich、Kabelo Moiloa、Hunter Monroe、Pietro Monticone、Oliver Nash、Emanuelle Natale、Filippo A. E. Nuccio、Pim Otte、Bartosz Piotrowski、Nicolas Rolland、Keith Rush、Yannick Seurin、Guilherme Silva、Bernardo Subercaseaux、Pedro Sánchez Terraf、Matthew Toohey、Alistair Tucker、Floris van Doorn、Eric Wieser 以及其他人提供的帮助和纠正。我们的工作部分得到了 Hoskinson 形式数学中心的支持。
