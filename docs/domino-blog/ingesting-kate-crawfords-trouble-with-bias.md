# 解读凯特·克劳福德的《偏见的烦恼》

> 原文：<https://www.dominodatalab.com/blog/ingesting-kate-crawfords-trouble-with-bias>

[*凯特·克劳福德*](https://twitter.com/katecrawford) *在最近的[旧金山城市艺术和演讲讲座](https://www.cityarts.net/event/artificial-intelligence-with-kate-crawford/)上讨论了偏见，讨论的[录音将于 5 月 6 日](https://www.cityarts.net/radio-broadcasts/)在 [KQED](https://www.kqed.org/) 和[当地分支机构](https://www.cityarts.net/radio-broadcasts/public-radio-affiliates/)播出。多米诺的成员是城市艺术讲座的现场观众。这篇 Domino 数据科学领域笔记提供了从 Crawford 的[城市艺术演讲](https://www.cityarts.net/event/artificial-intelligence-with-kate-crawford)和她的 [NIPS 主题演讲](https://www.youtube.com/watch?v=fMym_BKWQzk)中摘录的见解，为我们的数据科学博客读者提供了额外的广度、深度和背景。这篇博客文章涵盖了 Crawford 的研究，其中包括作为社会技术挑战的偏见、系统在接受有偏见的数据时的含义、模型的可解释性以及解决偏见的建议。*

## 凯特·克劳福德的《偏见的烦恼》

去年，[达美乐的首席执行官](//blog.dominodatalab.com/managing-data-science-as-a-capability/)在全公司的行业新闻讨论帖中发布了一个链接，链接到凯特·克劳福德的“[偏见带来的麻烦](https://www.youtube.com/watch?v=fMym_BKWQzk)”NIPS 主题演讲。[整个公司的多米诺骨牌](https://www.dominodatalab.com/careers/?utm_source=blog&utm_medium=post&utm_campaign=)每天都在讨论行业动向，一些团队成员还参加了 Kate Crawford 最近在三月份与 Indre Viskontas 的 [AI 讨论](https://www.cityarts.net/event/artificial-intelligence-with-kate-crawford/)。5 月 6 日将在 [KQED](https://www.kqed.org/) 和[当地分支机构](https://www.cityarts.net/radio-broadcathsts/public-radio-affiliates/)播出城市艺术与讲座讨论[的录音。这篇博客文章涵盖了演讲的主要亮点，包括当系统训练和接受有偏见的数据时的影响，偏见作为一种社会技术挑战，模型可解释性(或缺乏可解释性)，以及如何认识到数据不是中立的，作为解决偏见的关键步骤。这篇文章还引用了凯特·克劳福德(Kate Crawford)在 NIPS 上的主题演讲](https://www.cityarts.net/radio-broadcasts/)，以了解更多背景信息，并提出了三条解决偏见的建议。这些额外的建议包括使用公平辩论，与机器学习以外的人合作创建跨学科系统，以及“更努力地思考分类的伦理”。

## 接受过去偏见的系统培训:意义

在城市艺术讲座的早期，Indre Viskontas 要求 Kate Crawford 提供关于算法如何影响人们的额外背景。克劳福德提供了[红线](https://en.wikipedia.org/wiki/Redlining)作为一个例子，说明当任何系统被训练并接受基于过去偏见的数据时会发生什么。她讨论了 20 世纪 30 年代和 40 年代美国抵押贷款行业的红线是如何不给居住在以非洲裔美国人为主的社区的人贷款的，以及如何

“我们看到财富从黑人人口向白人人口的巨大转移……非常深远……这是民权运动的关键部分。最终通过了《公平住房法案》。

然而，当有系统可能在数据集上训练，但仍然有过去的偏见，那么就有可能[在今天](https://www.washingtonpost.com/news/wonk/wp/2018/03/28/redlining-was-banned-50-years-ago-its-still-hurting-minorities-today/)看到过去偏见的回声。Crawford 还提到了[关于亚马逊当天送达服务的彭博报告](https://www.bloomberg.com/graphics/2016-amazon-same-day/)如何表明“低收入黑人社区，即使在市中心，他们也无法获得当天送达……这几乎就像是这些旧红线地图的复制”。克劳福德表示，她过去住在波士顿，当时在她的社区里，有许多服务，但后来

“如果你在岩石街那边，那就没有服务。这是一个非凡的时刻，你意识到这些长期的种族歧视历史成为我们的算法系统摄取的数据模式，然后成为现在和未来的决策者。因此，这是一种方式，我们开始看到这些基本形式的歧视变成了这些存在于今天人工智能基础设施中的幽灵。”

这促使讨论转向围绕中性数据以及数据是否有可能是中性的辩论。

## 中性数据？一个社会技术问题。

维斯康塔斯指出,“计算机不可能有偏见”。这是客观的。”存在。维斯康塔斯还提到了 Twitter 机器人，这些机器人已经从他们的 Twitter 环境中学会了种族主义，并问克劳福德，“在训练这些算法方面，我们需要考虑什么？”Crawford 回应说,“该做什么”正在像 [FATML、](https://www.fatml.org/resources/relevant-scholarship)这样的会议上进行辩论，人们在那里讨论“我们是如何开始看到这些偏见模式的？以及如何实际纠正它们”。然而，克劳福德也指出，即使定义什么是“中立”也不容易，特别是如果偏见背后有“几十年，如果不是几百年”的历史，然后“你在重置什么？底线在哪里？”克劳福德主张将这些偏见和系统视为社会技术。克劳福德还在她的“[偏见的麻烦](https://www.youtube.com/watch?v=fMym_BKWQzk)”主题演讲中讨论了作为社会技术问题的偏见，其中偏见是“与分类更一致的问题”。

在“[偏见带来的麻烦](https://www.youtube.com/watch?v=fMym_BKWQzk)”主题演讲中，她还举例说明了分类是如何与特定时代或文化的社会规范联系在一起的。克劳福德引用了一位 17 世纪的思想家是如何指出有十一种性别的，包括女人、男人、神、女神、无生命的物体等等。这在今天看来可能有些武断和可笑。然而，

“分类工作总是文化的反映，因此它总是有点武断，它的时代和现代机器做出的决定从根本上把世界分成几部分，我们对这些部分的去向所做的选择将会产生后果”。

作为性别分类的现代对比，Crawford 还在 [NIPS 主题演讲](https://www.youtube.com/watch?v=fMym_BKWQzk)中指出，与四年前提到两种性别相比，脸书在 2017 年提到了五十六种性别。然而，在城市艺术讲座中，克劳福德指出，认识到数据不是中性的，是解决偏见的一大步。

## 模型可解释性或缺乏可解释性

在 [City Arts talk](https://www.cityarts.net/event/artificial-intelligence-with-kate-crawford/) 中，Viskontas 提出了一些后续问题，这些问题涉及根据有偏见的数据训练的算法可能会犯代价高昂的错误，以及是否有机会重新思考输入算法的数据。Crawford 讨论了模型的可解释性，引用了一个丰富的 Caruana 研究，他们创建了两个系统，一个是不可解释的，另一个是更可解释的。一个更容易解释的方法让研究小组看到他们是如何达到这一点的。例如，Crawford 表示，他们能够发现为什么算法会将一个本应立即被送入重症监护的高风险群体送回家。因为该团队有一个可解释的系统，他们能够看到，从历史上看，有一个极高风险的群体总是“被分成重症监护病房，因此没有数据可供[算法]学习”，因此，该算法表明高风险的人应该被送回家。

根据 Rich Caruana 等人的论文，“[可理解的医疗保健模型:预测肺炎风险和住院 30 天再入院](http://people.dbmi.columbia.edu/noemie/papers/15kdd.pdf)”

在机器学习中，通常必须在准确性和可理解性之间进行权衡。更精确的模型(如提升树、随机森林和神经网络)通常是不可理解的，但更可理解的模型(如逻辑回归、朴素贝叶斯和单决策树)通常具有明显更差的准确性。这种权衡有时会限制模型的准确性，而这些模型可以应用于医疗保健等任务关键型应用中，在这些应用中，理解、验证、编辑和信任学习到的模型非常重要。在肺炎风险预测案例研究中，可理解模型揭示了数据中令人惊讶的模式，这些模式以前阻止了复杂的学习模型在该领域中的应用，但因为它是可理解的和模块化的，所以允许这些模式被识别和删除。”

克劳福德引用卡鲁纳的研究作为人们应该“对数据代表的东西更加挑剔”的原因，以及这是“真正的一大步”的原因。“要意识到数据从来都不是中立的。数据总是来自一段人类历史。”

## 解决偏见的三个建议:“偏见的麻烦”NIPS 主题演讲

虽然克劳福德讨论了数据在城市艺术和 NIPS 会谈中如何从来都不是中立的，但在 [NIPS 会谈](https://www.youtube.com/watch?v=fMym_BKWQzk)中，她提出了三个关键建议供从事机器学习的人考虑:使用公平辩论，创建跨学科系统，以及“更加努力地思考分类的道德问题”。克劳福德将公平取证定义为测试我们的系统，“从建立预发布试验开始，在这些试验中，你可以看到一个系统如何在不同的人群中工作”，以及考虑“我们如何跟踪训练数据的生命周期，以实际了解谁建立了它以及人口统计偏差可能是什么。”Crawford 还讨论了如何需要与机器学习行业之外的领域专家合作，以创建测试和评估高风险决策系统的跨学科系统。克劳福德提倡的第三条建议是“更加努力地思考分类的伦理”。克劳福德指出，研究人员应该考虑是谁要求将人类分为特定的群体，以及他们为什么要求进行分类。克劳福德列举了历史上的人们，像[雷内·卡米尔](https://en.wikipedia.org/wiki/Ren%C3%A9_Carmille)，如何面对这些问题。卡米尔决定破坏二战中从死亡集中营拯救人们的人口普查系统。

[https://www.youtube.com/embed/fMym_BKWQzk?feature=oembed](https://www.youtube.com/embed/fMym_BKWQzk?feature=oembed)

## 摘要

Domino 团队成员每天都审查、吸收和讨论行业研究。Kate Crawford 对偏见的研究包括“[偏见带来的麻烦](https://www.youtube.com/watch?v=fMym_BKWQzk)”NIPS keynote 和[城市艺术和讲座讲座](https://www.cityarts.net/event/artificial-intelligence-with-kate-crawford)深入探讨了偏见如何影响开发、训练、解释和评估机器学习模型。虽然城市艺术和讲座是一个“面对面”的现场活动，但[的录音将于 5 月 6 日周日](https://www.cityarts.net/radio-broadcasts/)在 [KQED](https://www.kqed.org/) 和[当地分支机构](https://www.cityarts.net/radio-broadcasts/public-radio-affiliates/)播出。

*^(Domino 数据科学领域笔记提供数据科学研究、趋势、技术等亮点，支持数据科学家和数据科学领导者加快工作或职业发展。如果您对本博客系列中涉及的数据科学工作感兴趣，请发送电子邮件至 writeforus(at)dominodatalab(dot)com。)*