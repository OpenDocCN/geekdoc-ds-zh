# 18  结论

> 原文：[`tellingstorieswithdata.com/17-concluding.html`](https://tellingstorieswithdata.com/17-concluding.html)

1.  应用

1.  18  结论

**先决条件**

+   阅读 *五种修复统计方法* (Leek 等人 2017)

    +   对如何更好地进行数据科学的反思。

+   阅读 *十种改变科学的计算机代码* (Perkel 2021)

    +   讨论支撑数据科学的计算创新。

+   阅读 *从数据之旅中学习* (Leonelli 2020)

    +   广泛讨论数据在数据科学中的作用。

+   阅读 *真理、证明和可重复性：无代码的反击无计可施* (Gray 和 Marwick 2019)

    +   强调数据科学中可重复性的重要性。

+   观看 *作为业余软件开发者的科学* (McElreath 2020)

    +   详细介绍软件开发中适用于数据科学的经验教训。

## 18.1 结论

有句老话，大意是“愿你在有趣的时代生活”。也许每一代人都有这种感觉，但我们确实生活在有趣的时代。在这本书中，我们介绍了一些用数据讲故事的基本技能。这只是开始。

在不到一代人的时间里，数据科学已经从几乎不存在的东西，变成了学术界和工业界的一个定义性部分。这种变化的速度和程度对学习数据科学的人产生了许多影响。例如，这可能意味着人们不应该只做出优化当前数据科学外观的决定，还应该考虑可能发生的事情。虽然这有点困难，但这也是数据科学如此激动人心的原因之一。这可能意味着以下选择：

+   学习基础课程，而不仅仅是时尚的应用；

+   阅读核心文本，而不仅仅是流行的东西；并且

+   努力成为至少几个不同领域的交汇点，而不是过度专业化。

学习数据科学时最激动人心的时刻之一是意识到你只是喜欢玩数据。十年前，这并不适合任何特定的部门或公司。如今，它几乎适合它们中的任何一个。

数据科学需要在方法和应用上都坚持多样性。它正日益成为世界上最重要的工作，霸权方法没有立足之地。这正是对数据充满热情并能够构建事物的激动人心的时候。

这本书的核心论点是数据科学需要一场革命，我们提出了一种可能的样子。这场革命建立在统计学悠久的历史之上，大量借鉴了计算机科学，并在需要时借鉴其他学科，但核心是可重复性、工作流程和尊重。当数据科学开始时，它是模糊不清且定义不明确的。随着其成熟，我们现在将其视为能够独立存在。

这本书是对数据科学是什么以及它可能是什么的重新构想。在第一章中，我们提供了一个非正式的数据科学定义。我们现在重新审视它。我们认为数据科学是开发和应用一个原则性、经过测试、可重复、端到端工作流程的过程，该流程专注于定量措施本身，并作为探索问题的基础。我们长期以来就知道数学和统计理论中的严谨性是什么样子：定理伴随着证明(Horton 等人 2022)。我们越来越多地知道数据科学中的严谨性是什么样子：伴随着经过验证、测试、可重复的代码和数据的主张。严谨的数据科学创造了关于世界的持久理解。

## 18.2 一些突出的问题

当我们思考数据科学时，存在许多悬而未决的问题。这些问题并不是有明确答案的类型。相反，它们是需要探索和玩耍的问题。这项工作将推动数据科学的发展，更重要的是，帮助我们更好地讲述世界的故事。在这里，我们详细说明其中的一些。

**1. 我们如何编写有效的测试？**

计算机科学在测试及其单元和功能测试的重要性方面建立了坚实的基础，这一点被广泛接受。这本书的一个创新之处在于将测试整合到整个数据科学工作流程中，但就像任何事物的第一版一样，这需要相当程度的改进和发展。

我们需要彻底将测试与数据科学相结合。但尚不清楚这应该是什么样子，我们应该如何去做，以及最终状态是什么。在数据科学中，拥有经过良好测试的代码意味着什么？代码覆盖率，即有测试的代码行数的百分比，在数据科学中并不特别有意义，但我们应该使用什么来代替？数据科学中的测试看起来是什么样子？它们是如何编写的？统计学中广泛使用的模拟，数据科学已经采纳，提供了基础，但还需要大量的工作和投资。

**2. 数据清洗和准备阶段发生了什么？**

我们对数据清洗和准备对估计的影响程度没有很好的理解。Huntington-Klein 等人（2021），以及 Breznau 等人（2022）等人都开始了这项工作。他们表明，隐藏的研究决策对后续估计有重大影响，有时甚至大于标准误差。统计学为我们提供了对建模如何影响估计的良好理解，但我们还需要更多关于数据科学工作流程早期阶段影响的调查。更具体地说，我们需要寻找关键故障点并了解失败可能发生的方式。

随着我们扩展到更大的数据集，这一点尤其令人担忧。例如，ImageNet 是一个包含 1400 万张图片的数据集，这些图片都是手工标注的。在时间和金钱上的成本使得逐个检查每张图片以确保标签符合每个数据集用户的需要变得难以承受。然而，如果不进行这项工作，就很难对后续模型预测有信心，尤其是在不明显的情况下。

**3. 我们如何创建有效的名称？**

生物学的一项重大成就就是双名法。这是由 18 世纪的医生 Carolus Linnaeus 建立的正式系统命名方法（Morange 2016, 81）。每个物种都通过两个具有拉丁语法形式的单词来指代：第一个是它的属名，第二个是描述该物种的形容词。在生物学中，确保标准化命名法得到积极考虑。例如，研究人员推荐使用命名委员会（McCarthy et al. 2023）。正如在第九章讨论的，名称是数据科学中摩擦的一个大来源，因此在数据科学中也需要一个标准化的方法。

这之所以如此紧迫，是因为它影响了对理解的理解，进而影响效率。双名法提供的是诊断信息，而不仅仅是随意参考（Koerner 2000, 45）。这在数据科学由团队进行而不是单个个体进行时尤其如此。深入了解有效名称的构成以及鼓励这些名称的基础设施将带来显著收益。

**4. 数据科学与组成部分之间应有的适当关系是什么？**

我们将数据科学的起源描述为多种学科的融合。展望未来，我们需要考虑这些组成部分，尤其是统计学和计算机科学，应扮演什么角色。更普遍地说，我们还需要建立数据科学与计量经济学、应用数学和计算社会科学之间的关系及其相互作用。这些学科利用数据科学来回答自己领域的问题，但像统计学和计算机科学一样，它们也回馈数据科学。例如，计算社会科学中机器学习应用需要关注透明度、可解释性、不确定性和伦理问题，这一切都推动了在其他学科中进行的更理论化的机器学习研究（Wallach 2018）。

我们必须小心，继续从统计学家那里学习统计学，从计算机科学家那里学习计算机科学等。一个不这样做危险的例子是 p 值，我们在本书中并没有过多讨论，但它却主导了定量分析，尽管统计学家已经警告了其误用几十年。不向统计学家学习统计学的一个问题是，统计实践可能会变成一种天真遵循的食谱，因为这是教授它的最简单方式，尽管这并不是统计学家进行统计的方式。

数据科学必须与这些学科保持紧密的联系。如何继续确保数据科学拥有最佳方面，同时避免不良实践，这是一个特别重大的挑战。这不仅是一个技术问题，也是一个文化问题（Meng 2021）。特别重要的是确保数据科学保持一个包容性的卓越文化。

**5. 我们如何教授数据科学？**

我们开始就数据科学的基础达成共识。这包括发展对以下内容的舒适度：计算思维、抽样、统计学、图表、Git 和 GitHub、SQL、命令行、清理杂乱数据、包括 R 和 Python 在内的几种语言、伦理和写作。但在如何最好地教授它的问题上，我们几乎没有共识。部分原因是因为数据科学教师通常来自不同的领域，但这也是资源优先级差异的一部分。

问题的复杂性在于，鉴于对数据科学技能的需求，我们不能将数据科学教育仅限于研究生，因为本科生在进入职场时也需要这些技能。如果要在本科水平上教授数据科学，那么它需要足够强大，能够在大型班级中教授。开发可扩展的教学工具至关重要。例如，GitHub Actions 可以用来运行学生代码的检查，并提出改进建议，而不需要教师的参与。然而，将案例研究风格的课程扩展到规模特别困难，而学生通常发现这种课程非常有用。需要大量的创新。

**6. 行业与学术界之间的关系是什么样的？**

数据科学领域在行业中的创新相当多，但有时这种知识无法共享，即使可以共享，也往往进展缓慢。自 20 世纪 60 年代以来，学术界一直在使用“数据科学”这个术语，但正是由于行业的原因，它在过去十年左右的时间里变得流行（Irizarry 2020）。

将学术界和行业结合起来，既是数据科学的一个关键挑战，也是最容易忽视的一个。例如，行业面临的问题的性质，如确定客户的需求和大规模运营，与典型的学术关注点相去甚远。存在一种危险，即除非学者在行业中保持一只脚，并使行业能够积极参与学术界，否则学术研究可能会变得毫无意义。如果没有立即的回报，确保行业经验在学术招聘和拨款评估中得到重视可能会有所帮助，同样，鼓励学术界创业也会有所帮助。

## 18.3 下一步

这本书已经覆盖了很多内容，尽管我们快到结尾了，正如小说《日瓦戈医生》中的管家史蒂文斯所说，由石黑一雄所著：

> 今晚是一天中最好的部分。你已经完成了你一天的工作。现在你可以把脚放起来，享受它。
> 
> 伊什 iguro (1989)

很可能有一些方面你想进一步探索，基于你已建立的基础。如果是这样，那么这本书已经实现了它的目标。

如果你在本书开始时对数据科学是新手，那么下一步就是回顾我们跳过的内容。从 *数据科学：首次介绍* (Timbers, Campbell, and Lee 2022) 开始。之后，通过 *R 语言数据科学* ([Wickham, Çetinkaya-Rundel, and Grolemund [2016] 2023](99-references.html#ref-r4ds))。我们在本书中使用了 R 语言，并且只是简要提到了 SQL 和 Python，但发展对这些语言的舒适度很重要。从 *SQL 数据科学家指南* (Teate 2022)、*Python 数据分析指南* ([McKinney [2011] 2022](99-references.html#ref-pythonfordataanalysis)) 和免费的 Replit “100 天 Python 编程” [课程](https://replit.com/learn/100-days-of-python) 开始。

抽样是数据科学中的一个关键但容易被忽视的方面。阅读 *抽样：设计和分析* ([Lohr [1999] 2022](99-references.html#ref-lohr)) 是一个明智的选择。为了加深你对调查和实验的理解，接下来可以阅读 *实地实验：设计、分析和解释* (Gerber and Green 2012) 和 *值得信赖的在线受控实验* (Kohavi, Tang, and Xu 2020)。

为了培养更好的数据可视化技能，首先转向阅读 *数据草图* (Bremer and Wu 2021) 和 *数据可视化* (Healy 2018)。之后，建立强大的基础，例如 *图形语法* (Wilkinson 2005)。

如果你想要学习更多关于建模的知识，接下来的步骤是 *统计重思：R 和 Stan 中的贝叶斯课程实例* ([McElreath [2015] 2020](99-references.html#ref-citemcelreath))，它还附带了一系列优秀的配套视频，*贝叶斯规则！使用 R 进行贝叶斯建模的入门* (Johnson, Ott, and Dogucu 2022)，以及 *回归及其他故事* (Gelman, Hill, and Vehtari 2020)。建立概率基础也很重要，可以通过 *所有统计* (Wasserman 2005) 来实现。

如果你感兴趣的是机器学习，那么下一步自然就是 *统计学习导论* ([James et al. [2013] 2021](99-references.html#ref-islr))，接着是 *统计学习要素* (Friedman, Tibshirani, and Hastie 2009)。

要了解更多关于因果关系的知识，可以从通过 *《因果推断：混音带》* (Cunningham 2021) 和 *《效果：研究设计和因果关系简介》* (Huntington-Klein 2021) 的经济学视角开始。然后转向通过 *《如果》* (Hernán 和 Robins 2023) 的健康科学视角。

对于文本数据，从 *《文本作为数据》* (Grimmer, Roberts, 和 Stewart 2022) 开始。然后转向 *《R 中的文本分析监督机器学习》* (Hvitfeldt 和 Silge 2021)。

在伦理方面，有许多书籍。在这本书中，我们已经涵盖了它的许多章节，但从头到尾阅读 *《数据女性主义》* (D’Ignazio 和 Klein 2020) 会很有用，同样 *《人工智能地图》* (Crawford 2021) 也是如此。

最后，对于写作，最好是向内看。强迫自己每天写一个月。然后重复做。你会变得更好。话虽如此，有一些有用的书籍，包括 *《工作》* (Caro 2019) 和 *《写作：工艺回忆录》* (King 2000)。

我们经常听到“让数据说话”的短语。希望这很清楚，这种情况永远不会发生。我们所能做的就是承认我们是那些使用数据讲故事的人，并努力使它们值得。

> 是她的声音创造了
> 
> 天空在消失处最为尖锐。
> 
> 她以小时为单位测量它的孤独。
> 
> 她是世界的唯一创造者
> 
> 在其中她唱歌。当她唱歌时，大海，
> 
> 无论它有什么自我，都变成了自我
> 
> 那就是她的歌，因为她就是创造者。
> 
> 提取自“基韦斯特的秩序观念”，(Stevens 1934)

## 18.4 练习

### 问题

1.  数据科学是什么？

1.  数据影响谁，什么影响数据？

1.  讨论在模型中包含“种族”和/或“性取向”的问题。

1.  什么让一个故事更有或更少的说服力？

1.  处理数据时，伦理的作用是什么？

### 课堂活动

+   清理你的 GitHub：删除不必要的仓库，固定你最好的那些，更新你的个人资料，添加一个[个人 README](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-github-profile/customizing-your-profile/managing-your-profile-readme)。

Bremer, Nadieh, and Shirley Wu. 2021\. *数据草图*. A K Peters/CRC Press. [`doi.org/10.1201/9780429445019`](https://doi.org/10.1201/9780429445019).Breznau, Nate, Eike Mark Rinke, Alexander Wuttke, Hung HV Nguyen, Muna Adem, Jule Adriaans, Amalia Alvarez-Benjumea, et al. 2022\. “观察许多研究人员使用相同的数据和假设揭示了一个隐藏的不确定性宇宙。” *美国国家科学院院刊* 119 (44): e2203150119\. [`doi.org/10.1073/pnas.2203150119`](https://doi.org/10.1073/pnas.2203150119).Caro, Robert. 2019\. *工作*. 1st ed. New York: Knopf.Crawford, Kate. 2021\. *人工智能图谱*. 1st ed. New Haven: Yale University Press.Cunningham, Scott. 2021\. *因果推断：混音*. 1st ed. New Haven: Yale Press. [`mixtape.scunning.com`](https://mixtape.scunning.com).D’Ignazio, Catherine, and Lauren Klein. 2020\. *数据女性主义*. Massachusetts: The MIT Press. [`data-feminism.mitpress.mit.edu`](https://data-feminism.mitpress.mit.edu).Friedman, Jerome, Robert Tibshirani, and Trevor Hastie. 2009\. *统计学习的要素*. 2nd ed. Springer. [`hastie.su.domains/ElemStatLearn/`](https://hastie.su.domains/ElemStatLearn/).Gelman, Andrew, Jennifer Hill, and Aki Vehtari. 2020\. *回归及其他故事*. Cambridge University Press. [`avehtari.github.io/ROS-Examples/`](https://avehtari.github.io/ROS-Examples/).Gerber, Alan, and Donald Green. 2012\. *实地实验：设计、分析和解释*. New York: WW Norton.Gray, Charles T., and Ben Marwick. 2019\. “真理、证据和可重复性：无代码的代码无反击。” In *计算机与信息科学通讯*, 111–29\. Springer Singapore. [`doi.org/10.1007/978-981-15-1960-4_8`](https://doi.org/10.1007/978-981-15-1960-4_8).Grimmer, Justin, Margaret Roberts, and Brandon Stewart. 2022\. *文本作为数据：机器学习和社会科学的新框架*. New Jersey: Princeton University Press.Healy, Kieran. 2018\. *数据可视化*. New Jersey: Princeton University Press. [`socviz.co`](https://socviz.co).Hernán, Miguel, and James Robins. 2023\. *如果*. 1st ed. Boca Raton: Chapman & Hall/CRC. [`www.hsph.harvard.edu/miguel-hernan/causal-inference-book/`](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/).Horton, Nicholas, Rohan Alexander, Micaela Parker, Aneta Piekut, and Colin Rundel. 2022\. “可重复性和负责任的工作流程在数据科学和统计学课程中的日益重要性。” *统计学和数据科学教育杂志* 30 (3): 207–8\. [`doi.org/10.1080/26939169.2022.2141001`](https://doi.org/10.1080/26939169.2022.2141001).Huntington-Klein, Nick. 2021\. *效应：研究设计和因果关系的介绍*. 1st ed. Chapman & Hall. [`theeffectbook.net`](https://theeffectbook.net).Huntington-Klein, Nick, Andreu Arenas, Emily Beam, Marco Bertoni, Jeffrey Bloem, Pralhad Burli, Naibin Chen, et al. 2021\. “隐藏的研究者决策在应用微观经济学中的影响。” *经济调查* 59: 944–60\. [`doi.org/10.1111/ecin.12992`](https://doi.org/10.1111/ecin.12992).Hvitfeldt, Emil, and Julia Silge. 2021\. *R 中的文本分析监督机器学习*. 1st ed. Chapman; Hall/CRC. [`doi.org/10.1201/9781003093459`](https://doi.org/10.1201/9781003093459).Irizarry, Rafael. 2020\. “学术界在数据科学教育中的作用。” *哈佛数据科学评论* 2 (1). [`doi.org/10.1162/99608f92.dd363929`](https://doi.org/10.1162/99608f92.dd363929).Ishiguro, Kazuo. 1989\. *日之残骸*. 1st ed. Faber; Faber.James, Gareth, Daniela Witten, Trevor Hastie, and Robert Tibshirani. (2013) 2021\. *统计学习及其在 R 中的应用介绍*. 2nd ed. Springer. [`www.statlearning.com`](https://www.statlearning.com).Johnson, Alicia, Miles Ott, and Mine Dogucu. 2022\. *贝叶斯规则！使用 R 进行贝叶斯建模的介绍*. 1st ed. Chapman; Hall/CRC. [`www.bayesrulesbook.com`](https://www.bayesrulesbook.com).King, Stephen. 2000\. *写作：工艺回忆录*. 1st ed. Scribner.Koerner, Lisbet. 2000\. *林奈：自然与国家*. Cambridge: Harvard University Press.Kohavi, Ron, Diane Tang, and Ya Xu. 2020\. *可信在线受控实验：A/B 测试实用指南*. Cambridge University Press.Leek, Jeff, Blakeley McShane, Andrew Gelman, David Colquhoun, Michèle Nuijten, and Steven Goodman. 2017\. “修复统计学的五种方法。” *自然* 551 (7682): 557–59\. [`doi.org/10.1038/d41586-017-07522-z`](https://doi.org/10.1038/d41586-017-07522-z).Leonelli, Sabina. 2020\. “从数据之旅中学习。” In *科学中的数据之旅*, 1–24\. Springer International Publishing. [`doi.org/10.1007/978-3-030-37177-7_1`](https://doi.org/10.1007/978-3-030-37177-7_1).Lohr, Sharon. (1999) 2022\. *抽样：设计和分析*. 3rd ed. Chapman; Hall/CRC.McCarthy, Fiona M., Tamsin E. M. Jones, Anne E. Kwitek, Cynthia L. Smith, Peter D. Vize, Monte Westerfield, and Elspeth A. Bruford. 2023\. “在脊椎动物中标准化基因命名的案例。” *自然* 614 (7948): E31–32\. [`doi.org/10.1038/s41586-022-05633-w`](https://doi.org/10.1038/s41586-022-05633-w).McElreath, Richard. (2015) 2020\. *统计重思：R 和 Stan 中的贝叶斯课程实例*. 2nd ed. Chapman; Hall/CRC.———. 2020\. “科学作为业余软件开发的软件。” *YouTube*，九月。 [`youtu.be/zwRdO9%5FGGhY`](https://youtu.be/zwRdO9%5FGGhY).McKinney, Wes. (2011) 2022\. *Python 数据分析*. 3rd ed. [`wesmckinney.com/book/`](https://wesmckinney.com/book/).Meng, Xiao-Li. 2021\. “数据、数据科学或数据科学家有什么价值？” *哈佛数据科学评论* 3 (1). [`doi.org/10.1162/99608f92.ee717cf7`](https://doi.org/10.1162/99608f92.ee717cf7).Morange, Michel. 2016\. *生物学史*. New Jersey: Princeton University Press.Perkel, Jeffrey. 2021\. “十个改变科学的计算机代码。” *自然* 589 (7842): 344–48\. [`doi.org/10.1038/d41586-021-00075-2`](https://doi.org/10.1038/d41586-021-00075-2).Stevens, Wallace. 1934\. *基韦斯特的秩序观念*. [`www.poetryfoundation.org/poems/43431/the-idea-of-order-at-key-west`](https://www.poetryfoundation.org/poems/43431/the-idea-of-order-at-key-west).Teate, Renée. 2022\. *数据科学家用 SQL*. Wiley.Timbers, Tiffany, Trevor Campbell, and Melissa Lee. 2022\. *数据科学：第一次介绍*. Chapman; Hall/CRC. [`datasciencebook.ca`](https://datasciencebook.ca).Wallach, Hanna. 2018\. “计算社会科学 $\ne$ 计算机科学 + 社会数据。” *ACM 通讯* 61 (3): 42–44\. [`doi.org/10.1145/3132698`](https://doi.org/10.1145/3132698).Wasserman, Larry. 2005\. *所有统计*. Springer.Wickham, Hadley, Mine Çetinkaya-Rundel, and Garrett Grolemund. (2016) 2023\. *R 数据科学*. 2nd ed. O’Reilly Media. [`r4ds.hadley.nz`](https://r4ds.hadley.nz).Wilkinson, Leland. 2005\. *图形语法*. 2nd ed. Springer.
