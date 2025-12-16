# 29 Pyret vs. Python

> 原文：[`dcic-world.org/2025-08-27/pyret-vs-python.html`](https://dcic-world.org/2025-08-27/pyret-vs-python.html)

对于好奇者，我们在此提供一些示例，以证明我们对 Python 早期编程的挫败感。

| Python | Pyret |
| --- | --- |
| Python 默认暴露机器算术。因此，默认情况下，`0.1 + 0.2`不等于`0.3`。（我们希望你没有对此感到惊讶。）这种情况的原因是一个迷人的研究主题，但我们发现它对于编写算术程序的初学者来说是一个干扰。如果我们忽略浮点数的细节，我们是否认真对待程序可靠性的声明？ | Pyret 默认实现精确算术，包括有理数，`0.1 + 0.2`在 Pyret 中确实等于`0.3`。当计算必须返回一个不精确的数字时，Pyret 会明确地这样做：这是建立在可靠性基础上的课程的一个关键要求。 |
| Python | Pyret |
| 理解创建变量和更新其值之间的区别是一个关键的学习成果，以及理解变量的作用域。Python 明确地将声明与更新混淆，并且[与作用域有着复杂的历史](https://cs.brown.edu/~sk/Publications/Papers/Published/pmmwplck-python-full-monty/)。 | Pyret 是静态作用域的，并且付出了巨大的努力——例如，在表格查询语言的设计中——来保持这一点。在 Pyret 中处理变量的语法没有歧义。 |
| Python | Pyret |
| Python 有一个在语言设计后期添加的弱定义、可选的注解机制，[它混淆了值和类型](https://twitter.com/joepolitz/status/1357751800795832321?s=20)。 | 借鉴我们从[我们的](https://cs.brown.edu/~sk/Publications/Papers/Published/ffkwf-mrspidey/) [几个](https://cs.brown.edu/~sk/Publications/Papers/Published/gsk-flow-typing-theory/) [先前](https://cs.brown.edu/~sk/Publications/Papers/Published/pqk-progressive-types/) [研究](https://cs.brown.edu/~sk/Publications/Papers/Published/pgk-sem-type-fc-member-name/) [项目](https://cs.brown.edu/~sk/Publications/Papers/Published/lpgk-tejas-type-sys-js/)中学到的经验，Pyret 从一开始就被设计为具有可类型化，并采用了几个微妙的设计选择来实现这一点。Pyret 还支持（目前是动态的）对精炼类型注解的支持。 |
| Python | Pyret |
| Python 的注解机制没有精炼的概念。 | 为了让学生为具有丰富类型系统的现代编程语言做好准备，Pyret 的注解语法支持精炼。然而，这些是动态检查的，因此学生不需要满足任何特定证明辅助器的任意性。 |
| Python | Pyret |
| Python 对测试的支持较弱。虽然它有广泛的测试软件的专业库，但这些给学习者带来了非同小可的负担，因此大多数入门课程都不使用它们。 | 首先，一个宣称可靠性的课程必须将测试放在核心位置。其次，我们的教学法高度重视使用示例，特别是从具体实例中构建抽象。出于这两个原因，Pyret 在语言本身中提供了广泛的支持——不是通过可选的外部库——用于编写示例和测试，并为在这样做时出现的许多有趣且棘手的问题提供了直接的语言支持。 |
| Python | Pyret |
| 现代测试远不止单元测试。此外，基于属性的测试是思考形式属性的一个非常有用的[入口](https://cs.brown.edu/~sk/Publications/Papers/Published/wnk-use-rel-prob-pbt/)。在 Python 中，这只能通过库来实现。 | Pyret 具有方便的语言特性——例如在测试中使用`satisfies`而不是`is`——以轻量级的方式让学生接触这些想法。 |
| Python | Pyret |
| 状态在库中无处不在。 | 状态是编程的一个重要但也很复杂的部分。Pyret 引导学生在不允许状态编程的完整范围内编程，同时提供语言和输出方面的保障。例如，变量除非另有声明，否则是不可变的；可变字段会显示出来，以提醒学生该值可能已更改，甚至可能已经更改。 |
| Python | Pyret |
| 等价比较是简单的，并且与其他大多数专业语言一致。 | 等价实际上很微妙，并且作为教学工具很有用。因此，Pyret 精心设计了一系列等价运算符，它们不仅具有实际价值，而且具有教学用途。 |
| Python | Pyret |
| 图像不是语言中的值。你可以编写一个程序来生成图像，但你不能只是在你编程环境中查看它。 | 图像是值。Pyret 可以打印图像，就像它可以打印字符串或数字一样（为什么不呢？）。图像是有趣的价值，但它们并不轻浮：它们特别有助于阐明和解释像函数组合这样重要但抽象的问题。 |
| Python | Pyret |
| 语言没有内置的响应式程序概念。 | 响应性是语言的核心概念，也是设计和实现研究的话题。 |
| Python | Pyret |
| Python 的错误信息并非以新手为主要受众。 | 新手会犯很多错误。他们可能会特别害怕错误报告，并可能因为造成错误而感到沮丧。因此，Pyret 的错误信息是近十年[研究](https://cs.brown.edu/~sk/Publications/Papers/Published/mfk-measur-effect-error-msg-novice-sigcse/) [成果](https://cs.brown.edu/~sk/Publications/Papers/Published/mfk-mind-lang-novice-inter-error-msg/) [的结晶](https://cs.brown.edu/~sk/Publications/Papers/Published/wk-error-msg-classifier/)。事实上，一些教育工作者已经创建了依赖于 Pyret 错误信息性质和呈现方式的教学技巧。 |
| Python | Pyret |
| Python 已经开始遭受复杂性蔓延的困扰，我们认为这虽然对专业人士有利，但损害了新手。例如，Python 中 map 的结果实际上是一个特殊的生成器值。这可能导致需要额外解释的结果，比如 map(str, [1, 2, 3])产生<map object at 0x1045f4940>。类型提示（如上所述）是另一个例子。 | 由于 Pyret 的目标受众是使用本书风格编程的新手程序员，我们在添加任何功能时的主要目标是保留早期体验并避免惊喜。 |
| Python | Pyret |
| 数据定义是计算机科学的核心，但 Python 过度依赖内置数据结构（尤其是字典）并使得用户自定义的数据结构难以创建。 | Pyret 借鉴了像 Standard ML、OCaml 和 Haskell 这样的语言丰富的传统，提供了代数数据类型，其缺失往往迫使程序员进行难以处理（且效率低下）的编码技巧。 |
| Python | Pyret |
| Python 有更多粗糙的角落，可能导致意外和不受欢迎的结果。例如，`=`有时引入新变量，有时重新绑定它们。一个学生忘记返回值的函数不会导致错误，而是静默地返回`None`。Python 有一个复杂的[表](https://papl.cs.brown.edu/2020/growing-lang.html#%28part._design-space-cond%29)，描述了哪些值是真实的，哪些是假的。等等。 | Pyret 从一开始就被设计用来避免所有这些问题。 |
