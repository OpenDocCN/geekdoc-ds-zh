# 模型可解释性:对话继续

> 原文：<https://www.dominodatalab.com/blog/model-interpretability-the-conversation-continues>

这个 Domino 数据科学领域的笔记涵盖了可解释性的定义和 PDR 框架的概述。见解来自郁彬、w .詹姆斯·默多克、钱丹·辛格、卡尔·库姆伯和雷扎·阿巴西-阿西最近的论文《可解释机器学习的定义、方法和应用》。

## 介绍

模型的可解释性继续在工业界引发公众讨论。我们之前已经讨论过[模型可解释性](https://blog.dominodatalab.com/tag/model-interpretability/)，包括[提出的机器学习(ML)可解释性的定义](https://blog.dominodatalab.com/make-machine-learning-interpretability-rigorous/)。然而，郁彬、詹姆斯·默多克、钱丹·辛格、卡尔·库姆伯和礼萨·阿巴西-阿西在他们最近的论文《[可解释机器学习的定义、方法和应用](https://bids.berkeley.edu/news/berkeley-team-unveils-new-unified-framework-selecting-and-evaluating-interpretable-machine)》中指出，早期的定义还不够。Yu 等人主张“在机器学习的背景下定义可解释性”，并使用预测性、描述性和相关性(PDR)框架，因为“关于可解释性的概念存在相当大的混乱”。

数据科学工作是实验性的、反复的，有时令人困惑。然而，尽管复杂(或因为复杂)，数据科学家和研究人员策划并使用不同的语言、工具、包、技术和框架来解决他们试图解决的问题。业界一直在评估潜在的组件，以确定集成组件是否会有助于或妨碍他们的工作流程。虽然我们提供了一个[平台即服务](https://www.dominodatalab.com/)，行业可以使用他们选择的语言、工具和基础设施来支持模型驱动的工作流，但我们在这个博客中涵盖了[实用技术](https://blog.dominodatalab.com/category/practical-data-science-techniques/)和研究，以帮助行业做出自己的评估。这篇博客文章提供了对提议的 PDR 框架的简要概述，以及一些供业界考虑的额外资源。

## 为什么要迭代可解释 ML 的定义？

虽然包括[Finale Doshi-维勒兹](https://finale.seas.harvard.edu/)和 [Been Kim](https://beenkim.github.io/) 在内的研究人员已经提出了可解释性的定义，但于等人在[最近的论文](https://www.pnas.org/content/early/2019/10/15/1900654116)中认为先前的定义还不够深入，而且

这导致了对可解释性概念的相当大的混淆。特别是，不清楚解释某个东西意味着什么，不同的方法之间存在什么共同点，以及如何为特定的问题/受众选择一种解释方法。”

和[主](https://www.pnas.org/content/116/44/22071)

“作为更大的数据科学生命周期的一部分，我们专注于使用解释从 ML 模型中产生洞察力，而不是一般的可解释性。我们将可解释的机器学习定义为从机器学习模型中提取相关知识，涉及包含在数据中或由模型学习的关系。在这里，我们认为知识是相关的，如果它为特定的受众提供了对所选问题的洞察力。这些见解通常用于指导沟通、行动和发现。它们可以以可视化、自然语言或数学方程式等形式生成，具体取决于上下文和受众。”

Yu 等人[还认为](https://www.pnas.org/content/116/44/22071)以前的定义关注 ML 可解释性的子集，而不是整体性的，并且说明性的、描述性的、相关的(PDR)框架，加上词汇表，旨在“完全捕捉可解释的机器学习、其益处及其对具体数据问题的应用。”

## 规范性、描述性、相关性(PDR)框架

Yu 等人指出，在“如何针对特定问题和受众选择和评估解释方法”以及 PDR 框架如何应对这一挑战方面缺乏明确性。PDR 框架包括

为特定问题选择解释方法的三个要素:预测准确性、描述准确性和相关性。

于等人还认为，要使一个解释可信，实践者应该寻求预测和描述准确性的最大化。然而，在选择模型时需要考虑权衡。举个例子，

*“基于模型的解释方法的简单性产生了一致的高描述准确性，但有时会导致复杂数据集的预测准确性较低。另一方面，在图像分析等复杂设置中，复杂模型可以提供较高的预测准确性，但更难分析，导致描述准确性较低。”*

### 预测准确性

Yu 等人[将](https://www.pnas.org/content/116/44/22071)预测精度定义为在解释的上下文中，关于与模型的基础数据关系的近似值。如果近似性很差，那么提取的洞察力也会受到影响。在构建模型时，可能会出现类似这样的错误。

“在标准的监督 ML 框架中，通过测试集精度等措施，评估模型拟合的质量已经得到了很好的研究。在解释的上下文中，我们将这种误差描述为预测准确性。请注意，在涉及可解释性的问题中，人们必须适当地衡量预测的准确性。特别是，用于检查预测准确性的数据必须与感兴趣的人群相似。例如，对一家医院的病人进行评估可能不能推广到其他医院。此外，问题通常需要超出平均准确度的预测准确度的概念。预测的分布很重要。例如，如果某个特定类别的预测误差非常高，就会出现问题。”

### 描述准确性

Yu 等人定义了描述准确性，

*“在解释的上下文中，解释方法客观地捕捉由机器学习模型学习的关系的程度。*

Yu 等人指出，当关系不明显时，描述准确性对于复杂的黑盒模型或神经网络是一个挑战。

### 关联

于等人认为，在口译的语境中，相关性的定义是“如果它为特定受众提供了对所选领域问题的洞察力。”岳等人还指出，相关性有助于关于准确性的权衡决策，并强调观众是人类观众。

根据手头问题的背景，从业者可能会选择关注其中一个而不是另一个。例如，当可解释性被用于审计模型的预测时，比如为了加强公平性，描述的准确性可能更重要。相比之下，可解释性也可以单独用作提高模型预测准确性的工具，例如，通过改进特征工程。”

## 结论和资源

这篇 Domino 数据科学领域笔记提供了 Yu 等人对 ML 可解释性和 PDR 框架的定义的简要概述，以帮助研究人员和数据科学家评估是否将特定的技术或框架集成到他们现有的工作流程中。有关可解释性的更多信息，请查阅以下资源

*   郁彬等人。[《可解释的定义、方法和应用》](https://www.pnas.org/content/116/44/22071)【门控】
*   郁彬等人。[可解释的机器学习:定义、方法和应用](https://arxiv.org/abs/1901.04592)
*   多米诺。[“数据伦理:质疑真理，重新安排权力](https://blog.dominodatalab.com/data-ethics-contesting-truth-and-rearranging-power/)”
*   多米诺。"[让机器学习更严谨](https://blog.dominodatalab.com/make-machine-learning-interpretability-rigorous/)"
*   多米诺。"[TCAV 模型的可解释性](https://blog.dominodatalab.com/model-interpretability-tcav-testing-concept-activation-vectors/)"
*   多米诺。"凯特·克劳福德的《偏见的烦恼》"
*   多米诺。[“对石灰的信任:是，不是，也许是？”](https://blog.dominodatalab.com/trust-in-lime-local-interpretable-model-agnostic-explanations/)
*   乔希·波杜斯卡。 [SHAP 系列。](https://blog.dominodatalab.com/tag/shap/)
*   乔希·波杜斯卡。可解释的 ML/AI 的数据科学剧本
*   帕科·内森。[临近模型可解释性的新兴线索](https://blog.dominodatalab.com/themes-and-conferences-per-pacoid-episode-9/)

*^([多米诺数据科学领域笔记](https://blog.dominodatalab.com/tag/domino-data-science-field-note/)提供数据科学研究、趋势、技术等亮点，支持数据科学家和数据科学领导者加快工作。如果您对本博客系列中涉及的数据科学工作感兴趣，请发送电子邮件至 writeforus(at)dominodatalab(dot)com。)*