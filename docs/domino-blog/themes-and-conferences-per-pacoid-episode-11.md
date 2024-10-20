# 每个 Pacoid 的主题和会议，第 11 集

> 原文：<https://www.dominodatalab.com/blog/themes-and-conferences-per-pacoid-episode-11>

*[Paco Nathan](https://twitter.com/pacoid) 的最新文章涵盖了程序合成、AutoPandas、模型驱动的数据查询等等。*

## 介绍

欢迎回到我们每月一次的主题发现和会议总结。顺便说一句，第二版的视频出来了:【https://rev.dominodatalab.com/rev-2019/ 

甲板上这次的‘绕月:*节目综合*。换句话说，使用元数据来生成代码。在这种情况下，会生成用于数据准备的代码，数据科学工作中的大量“时间和劳动”都集中在这里。SQL 优化提供了有用的类比，给出了 SQL 查询如何在内部被转换成查询图**，然后真正聪明的 SQL 引擎处理该图。我们在[气流](https://airbnb.io/projects/airflow/)等方面看到的一个长期趋势是**将基于图形的元数据外部化**，并在单个 SQL 查询的生命周期之外利用它，使我们的工作流更加智能和健壮。让我们把细节留到下一次，但简单来说，这项工作会引出关于数据集的知识图表和 [Jupyter](https://jupyter.org/) 支持的新特性。**

 **## 程序合成

如果你还没看过 AutoPandas，去看看吧。要点是这样的:给一只[熊猫。DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) ，假设您需要通过对其中一列执行``groupby()``来聚合数据，然后计算分组值的平均值。您可以将数据帧的“之前”和“之后”版本作为输入输出示例对提供给 AutoPandas。然后点击 AutoPandas 中的*合成*按钮，获得 Python 中的一段代码。完了，完了。

作为程序合成的背景，我们通过编写详细的指令来给计算机编程似乎有点奇怪。一大群软件工程师，每天 24 小时埋头苦干——真的必须这样吗？Don Knuth 在他的“[识字编程](http://www.literateprogramming.com/knuthweb.pdf)”文章中提出了一个著名的相关问题:

让我们改变对程序构造的传统态度:与其想象我们的主要任务是指导计算机做什么，不如让我们集中精力向人类解释我们希望计算机做什么

顺便说一句，Knuth 1983 年的那篇文章可能是我第一次看到“Web”这个词被用作与计算机相关的意思。不知道蒂姆·伯纳斯·李在创建万维网时是否重复了这一点。无论如何，Knuth 对*文化编程*的描述指导了很多后续工作，包括【Jupyter 项目。最终目标是让人们解释他们需要做什么，然后让计算机生成或合成代码来执行所需的操作。这难道不像是机器学习的一个有价值的目标吗——让机器学会更有效地工作？这就是*节目合成*的主旨。

程序综合的许多早期工作倾向于集中在特定领域语言(DSL)上，因为它们是相对受限的问题空间。比如你可以写一个基本的编程语言解释器作为[Scala](https://github.com/tpolecat/basic-dsl)中的一个小 DSL。实际上，Scala 因为实现了各种各样的 DSL 而广受欢迎。

然而，AutoPandas 有一个更大胆的计划。他们开始实现“宽”API 的程序合成，而不是“窄”受限的 DSL。这暗示了数据科学的工作，在这些工作中，许多繁重的数据准备工作都是在像 pandas、NumPy 等这样的库中完成的。，这似乎符合 AutoPandas 研究的意图。

## 自动潘达斯:起源

AutoPandas 是在加州大学伯克利分校 rise lab 创建的，其总体思想在 NeurIPS 2018 年的论文“[从输入-输出示例进行 API 函数的神经推理](https://www.carolemieux.com/autopandas_mlforsys18.pdf)”中有所描述，作者包括 Rohan Bavishi、Caroline Lemieux、Neel Kant、Roy Fox、Koushik Sen 和 Ion 斯托伊察。另见:卡罗琳·雷米尔关于 NeurIPS 演讲的[幻灯片](https://www.carolemieux.com/autopandas_mlforsys18_slides.pdf)，以及罗汉·巴维希来自 2019 年夏季度假胜地的[视频](https://youtu.be/QD4xBW1JVQI)。

AutoPandas 的作者观察到:

*   流行数据科学包的 API 往往有相对陡峭的学习曲线。
*   当文档中找不到合适的例子时，人们会求助于在线资源，如 StackOverflow，以了解如何使用 API。
*   这些在线资源可能不太适合回答问题:版本问题、陈旧的答案、粗鲁/钓鱼等。

相反， *[程序合成](https://en.wikipedia.org/wiki/Program_synthesis)* 可以解决这些问题。换句话说，人们可以通过展示例子来描述他们想如何使用 API 的意图，然后收到合成的代码。在论文和视频中，作者描述了他们希望如何“自动化 API 的 StackOverflow ”,并提出了具体目标:

*   为现实的、广泛的 API(而不是狭隘的 DSL)开发一个程序合成引擎。
*   扩展问题以处理复杂的数据结构。
*   扩展问题以合成数百个函数和数千个潜在参数。

他们的方法是基于使用深度学习的组件，特别是使用[图嵌入](https://towardsdatascience.com/overview-of-deep-learning-on-graph-embeddings-4305c10ad4a4):

*“在高层次上，我们的方法通过以下方式工作:(1)将 I/O 示例预处理到图中，(2)将这些示例馈送到可训练的神经网络，该神经网络学习图中每个节点的高维表示，以及(3)汇集到神经网络的输出，并应用 softmax 来选择熊猫函数。在此之后，我们使用穷举搜索来找到正确的论点；这些细节超出了本文的范围。”*

对于训练和评估数据，他们使用 StackOverflow 中的[示例，Wes McKinney](https://stackoverflow.com/questions/tagged/pandas) (熊猫的创造者)的 [Python 用于数据分析，以及](https://www.goodreads.com/book/show/36313494-python-for-data-analysis)[数据学校视频系列](https://www.dataschool.io/easier-data-analysis-with-pandas/)。在问题的第一部分，用一个图来表示数据帧的输入输出例子。在问题的第二部分，涉及到一些搜索，潜在的搜索空间可能很大且很复杂。即使对于简单的“深度 1”解析树，搜索空间也具有 10^17 分支因子。早期，RISElab 研究人员的蛮力尝试能够将搜索空间缩小到 10^4—reduced 的 13 个数量级。令人印象深刻。使用最大似然模型进行更有效的搜索，将搜索空间缩小到 10^2—which，可以在适度的硬件上运行。那就更厉害了。

## 自动潘达斯:结果

到目前为止，AutoPandas 已经支持了 119 只熊猫。DataFrame 转换函数及其合成代码的结果引人注目。回到 StackOverflow，作者发现他们可以解决 65%的*熊猫。DataFrame* 问题。这是基于对结果的“top-1”评估，这意味着合成代码的最佳预测输出必须与 StackOverflow 上选择的答案相匹配。或者，就结果的“成功率”评估而言，AutoPandas 为 82%的 StackOverflow 答案预测了正确的功能等价。

在架构方面，AutoPandas 的前端运行在无服务器计算上，而其后端基于 [Ray](https://rise.cs.berkeley.edu/projects/ray/) 。后端处理的一部分需要深度学习(图形嵌入)，而其他部分利用[强化学习](https://www.dominodatalab.com/blog/what-is-reinforcement-learning)。

展望未来，AutoPandas 团队打算扩展该系统以合成更长的程序，并可能将其移入开源。他们使用的方法适用于其他流行的数据科学 API，如 [NumPy](https://www.numpy.org/) 、 [Tensorflow](https://www.tensorflow.org/) 等等。这个领域肯定会变得有趣。

## 程序合成的一个范例

有关程序合成的更多背景信息，请查看詹姆斯·博恩霍尔特(James born Holt)2015 年的“[程序合成解释](https://homes.cs.washington.edu/~bornholt/post/synthesis-explained.html)”，以及亚历克斯·波洛佐夫(Alex Polozov)2018 年更近的“[2017-18 年程序合成](https://alexpolozov.com/blog/program-synthesis-2018/)”。

如果你想深入了解，这里有相关论文和文章的样本:

*   "[用深度学习合成程序](https://alexpolozov.com/blog/program-synthesis-2018/)"-尼尚特·辛哈(2017-03-25)
*   "[一个程序合成引物](https://barghouthi.github.io/2017/04/24/synthesis-primer/)"–Aws Albarghouthi(2017-04-24)
*   "[从输入-输出示例中进行交互式查询综合](https://scythe.cs.washington.edu/media/scythe-demo.pdf)"——王成龙，阿尔文·张，拉斯蒂斯拉夫·博迪克(2017-05-14)
*   "[ICLR 2018](https://scythe.cs.washington.edu/media/scythe-demo.pdf)项目综合论文"——伊利亚·波洛舒欣(2018-05-01)
*   "[程序合成是可能的](https://www.cs.cornell.edu/~asampson/blog/minisynth.html)"–阿德里安·桑普森(2018-05-09)
*   "[利用语法和强化学习进行神经程序合成](https://arxiv.org/pdf/1805.04276.pdf)"–鲁迪·布内尔，马修·豪斯克内希特，雅各布·德夫林，里沙布·辛格，普什梅特·柯利(2018-05-28)
*   “[执行引导的神经程序综合](https://openreview.net/pdf?id=H1gfOiAqYm)”——陈，，宋晓(2018-09-27)
*   "[软件写软件——程序合成 101](https://openreview.net/pdf?id=H1gfOiAqYm)"——亚历山大·维迪博尔斯基(2019-01-20)
*   "[使用有经验的垃圾收集器自动合成长程序](https://arxiv.org/pdf/1809.04682.pdf)"–Amit Zohar，Lior Wolf (2019-01-22)

总之，这些链接应该提供一些轻松的夏日阅读乐趣。

顺便说一句，为了减少可能的混淆，请注意，在野外还有其他名为“autopandas”的 Python 项目，它们与 RISElab 的工作无关:

*   [https://pypi.org/project/autopandas/](https://pypi.org/project/autopandas/)
*   [https://github.com/Didayolo/autopandas](https://github.com/Didayolo/autopandas)

## 我们来谈谈 SQL

说到 DSL，最容易识别的 DSL 之一就是 SQL。SQL 的内部与程序合成有相似之处，并有助于指出用于数据科学工作的开源的新兴主题。

SQL 是一种声明性语言。SQL 侧重于描述必须执行“什么”——结果集应该是什么样子，而不是应该“如何”执行该工作的低级细节。要构建 SQL 查询，必须描述所涉及的数据源和高级操作(SELECT、JOIN、WHERE 等。)请求。然后，查询引擎开始忙碌起来，在幕后发生了很多事情:查询历史和统计信息通知各种优化策略；索引被自动使用；中间结果得到缓存；算法变体被替换，等等。

为了实现所有这些神奇的转换，首先 SQL 引擎解析一个查询，生成一个查询图，它是一个*有向无环图* (DAG)。然后，查询引擎识别要使用的表(如果您回到 Edgar Codd 的[原始论文](https://www.seas.upenn.edu/~zives/03f/cis550/codd.pdf)，也称为“关系”)。查询图提供了在关系数据库栈的多个层进行优化所利用的元数据。最后，生成一些代码来操作表和索引中的数据，如果一切顺利，结果集将作为输出。这并不完全不同于程序合成，但更像是 [OG](https://www.urbandictionary.com/define.php?term=OG) 。

一个快乐、高效的 SQL 开发人员通常看不到这种神奇的事情发生，而是关注于如何声明*结果集*应该是什么样的，而不是太多隐藏的细节。这说明了 SQL 非凡的学习曲线方面，如此多的数据如何能够被执行*而不需要*费心处理细节。显然，对于那些正在学习数据管理和分析的人来说，SQL 有助于减少[认知负荷](https://en.wikipedia.org/wiki/Cognitive_load)。SQL 的悠久历史和普及性使得数据驱动的工作更容易被更广泛的受众所接受。

## SQL 和 Spark

要全面了解 SQL 优化器是如何工作的，请看一下 Spark SQL 中的 *[Catalyst 优化器](https://databricks.com/blog/2015/04/13/deep-dive-into-spark-sqls-catalyst-optimizer.html)* 。详情请见他们的 [SIGMOD 2015 论文](http://people.csail.mit.edu/matei/papers/2015/sigmod_spark_sql.pdf)，其中迈克尔阿姆布鲁斯特&公司惊人地打出了一个公园:

“Spark SQL 的核心是 Catalyst optimizer，它以一种新颖的方式利用高级编程语言特性(例如 Scala 的模式匹配和准引号)来构建可扩展的查询优化器。”

根据定义，Spark 程序是 Dag，旨在使用各种不同类型的数据源。Spark 项目也有相对可互换的“数据框架”概念(不完全是*熊猫。DataFrame* ，但是关闭)和 SQL 查询结果集。正如 Catalyst 论文所述，它是“为现代数据分析的复杂需求量身定制的”，非常方便。
例如，假设您有(1)一堆通过 ETL 过程加载的 JSON 中的客户档案记录，以及(2)您需要将这些记录与存储在 [Parquet](http://parquet.apache.org/) 中的日志文件连接起来。这些日志文件的列数很可能比大多数查询所需的要多。Catalyst 足够聪明，可以通过 DAG 图推断出它可以对 Parquet 数据使用 *[下推谓词](https://jaceklaskowski.gitbooks.io/mastering-spark-sql/spark-sql-Optimizer-PushDownPredicate.html)* ，只反序列化查询所需的确切列。较少的数据被解压缩、反序列化、加载到内存中、在处理过程中运行等。要点提示:**对特定数据集的图表和元数据进行智能处理可以在多个层面上实现优化。那是好东西。**

## 模型驱动的数据查询

正如蒂姆·菲利普·克拉斯卡等人在“[学习索引结构的案例](https://dl.acm.org/citation.cfm?id=3196909)”(见[视频](https://www.youtube.com/watch?v=NaqJO7rrXy0&feature=youtu.be))中指出的那样，内部智能(B 树等。)代表了机器学习的早期形式。此外，它们可以被机器学习模型取代，以大幅提高性能:

“我们已经证明，机器学习模型有可能提供超越最先进指数的显著优势，我们相信这是未来研究的一个富有成效的方向。”

学习指数的研究强调了另一个有趣的点。关于 SQL 和它的声明性本质(换句话说，它对于如此广泛的应用程序的易用性)的一个权衡是，系统中的许多“智能”是在幕后发生的。按照理论，如果您*将所有数据处理集中*在同一个关系数据库管理系统中，那么该系统可以收集关于您的用例的元数据(例如，查询统计数据),并利用它作为 SQL 优化器的反馈。

当然，如果你在你的数据科学工作流程中使用几个不同的数据管理框架——就像现在每个人都做的那样——那么 RDBMS 的大部分魔力就会烟消云散。有些人可能会问:“难道我们不能回到商业智能、OLAP 和企业数据仓库的辉煌时代吗？”不，妖怪已经跑出瓶子了。21 世纪初的一个明显教训是:依赖于[集权](https://www.mckinsey.com/capabilities/people-and-organizational-performance/our-insights/to-centralize-or-not-to-centralize)的大规模战略通常是有风险的(约翰·罗布在我刚刚读过的 *[【勇敢的新战争】](https://www.goodreads.com/book/show/679469.Brave_New_War)* 中详细探讨了这一点——好东西)。一刀切(OSFA)是数据科学工作中的一个有害神话，没有一个数据框架会适合您的所有用例和需求。

Apache Spark 在集成各种不同的数据源和数据汇方面进行了创新，特别是对于非结构化数据，并将“应用程序代码”结构化为 SQL 语句，其结果集成为数据帧。即便如此，还是有一个问题:Spark 和其他 SQL 引擎一样，每次运行程序时都会重新创建一个查询图。表示运行时开销的。查询图用于在并行化、计算时间成本、内存资源等方面优化生成的应用程序代码。之后，图形消失。这种方法的好处(降低认知负荷等。)抵消增加的运行时开销和消失的元数据。如果更正式地外部化，后者可能特别有价值。

## 回路

这就是我们兜了一圈的地方。查看上面提到的“[镰刀](https://scythe.cs.washington.edu/)”演示和来自华盛顿大学的王成龙、阿尔文·张和拉斯·博迪克的相关[论文](https://scythe.cs.washington.edu/media/scythe-demo.pdf)。他们为 SQL 查询创建了一个程序合成系统，使用了类似于 AutoPandas 的输入输出示例，还利用了 StackOverflow 问题。虽然 AutoPandas 项目没有具体引用(如果我错过了参考文献，请原谅？)早期 U Washington 项目的几个方面似乎非常相似，包括实验设计、训练/测试数据源，甚至幻灯片。华盛顿大学的团队开发了一个 SQL 示例编程系统，这是一种有趣的方式，可以将更多的机器学习引入 SQL 查询。它解决了将一些关于 SQL 查询使用的元数据外部化的问题——为机器学习重用这些元数据。相比之下，RISElab 团队专注于*熊猫。DataFrame* 和杠杆射线。它们解决了一个更具挑战性的问题(“宽”API，而不是“窄”DSL，如 SQL)，此外，它们还利用了更多的 SOTA 工具。我期待着看到 RISElab 研究的展开并取得许多进展。

此时，您可能已经开始将 Catalyst、Scythe、Learned Indexes、AutoPandas 等联系起来。机器学习在这部电影中扮演了很大的角色。图形也发挥了很大的作用，除了 SQL 使用一个内在化的图形(阅读:大部分是凡人看不见的)；然而，越来越多的情况是利用一个*外化的*图来代替。

## 关于数据集的外部化图形

我们看到的一个长期趋势是[气流](https://airbnb.io/projects/airflow/)等。，就是将基于图的元数据具体化。有没有办法利用关于数据集的元数据来使我们的工作流更智能、更健壮？换句话说，在单个 SQL 查询的生命周期之外还有其他用途吗？鉴于目前处于“数据目录和谱系”与“数据集知识图”交汇处的开源项目的[风暴](https://docs.google.com/spreadsheets/d/1K_nA805YXXsez-S83zk6gi1bXoYTjNF06kCBHWKLpow/edit#gid=661099478)，答案是响亮的“YAAASSS！”

在这里我想分享一些关于 Jupyter 项目的相关新闻。开发中有一组新的“丰富的上下文”功能，以支持作为顶级实体的项目、实时协作(à la GDocs)和评论、数据注册、元数据处理、注释和使用跟踪。

假设，假设您有一群数据科学家在 Jupyter 工作，并且您的组织正在认真对待数据治理。假设您想要跟踪关于数据集使用的元数据，例如，以便您可以随后[构建关于您组织中使用的数据集的知识图](https://www.akbc.ws/2019/)，然后利用机器学习进行数据治理应用。跟我到目前为止？通过这个[非常受欢迎的开源](https://twitter.com/parente/status/1111346560582017028)数据基础设施将知识图实践注入主流 It 现实是有意义的。

更好的是，Jupyter 是开放标准的典范，所以请随意加入对话:

*   [数据浏览器](https://github.com/jupyterlab/jupyterlab-data-explorer/blob/master/press_release.md)
*   [元数据浏览器](https://github.com/jupyterlab/jupyterlab-metadata-service/blob/master/press_release.md)
*   [评论](https://github.com/jupyterlab/jupyterlab-commenting/blob/master/press_release.md)

另外，GitHub 上有一个[问题/主题提供了更多的背景故事。](https://github.com/jupyterlab/jupyterlab/issues/5548)

总的来说，你对熊猫的使用。DataFrame 和其他流行的用于数据科学工作的开源 API 可能会变得更加智能和自动化。工作流正在不断发展，以在更长的生命周期内捕获数据集的元数据，这样数据科学工作流本身就可以变得更加模型驱动。**