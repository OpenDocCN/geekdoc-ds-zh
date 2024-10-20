# 对莱姆的信任:是，不是，也许是？

> 原文：<https://www.dominodatalab.com/blog/trust-in-lime-local-interpretable-model-agnostic-explanations>

*在本 Domino 数据科学领域笔记中，我们简要讨论了一种用于生成解释的算法和框架， [LIME(本地可解释模型不可知解释)](https://github.com/marcotcr/lime)，这可能有助于数据科学家、机器学习研究人员和工程师决定是否信任任何模型中任何分类器的预测，包括看似“黑盒”的模型。*

## 你信任你的模型吗？

信任是复杂的。

韦氏词典的[对信任的定义](https://www.merriam-webster.com/dictionary/trust)包括多个组成部分，传达了与信任相关的潜在细微差别。在构建和部署模型时，数据科学家、机器学习研究人员和工程师决定是否信任模型或分类器的预测。他们还权衡了可解释性。关于机器学习可解释性的[严格程度的讨论，以及关于](/blog/make-machine-learning-interpretability-rigorous/)[缺乏机器学习可解释性](/blog/ingesting-kate-crawfords-trouble-with-bias/)的影响，之前已经在 Domino 数据科学博客中报道过。在这篇博客文章中，我们简要讨论了一种用于生成解释的算法和框架， [LIME](https://github.com/marcotcr/lime) (本地可解释模型不可知解释)，这可能有助于人们决定是否信任任何模型中任何分类器的预测，包括看似“黑盒”的模型。

## 什么是石灰？(本地可解释的模型不可知的解释)

2016 年，马尔科·图利奥·里贝里奥、萨梅尔·辛格和卡洛斯·盖斯特林发表了一篇论文，“[我为什么要相信你？:解释任何分类器](https://arxiv.org/pdf/1602.04938.pdf)的预测，这在 [KDD 2016](https://www.youtube.com/watch?v=KP7-JtFMLo4) 以及[博客帖子](https://homes.cs.washington.edu/~marcotcr/blog/lime/)中讨论过。在[的论文](https://arxiv.org/pdf/1602.04938.pdf)中，Riberio、Singh 和 Guestrin 建议将 LIME 作为一种手段，“为单个预测提供解释，作为‘信任预测问题’的解决方案，并选择多个这样的预测(和解释)作为‘信任模型’问题的解决方案。”Riberio、Singh 和 Guestrin 还将 LIME 定义为“一种算法，它可以通过用可解释的模型在局部对其进行近似，以忠实的方式解释任何分类器或回归器的预测”。在后来的[数据怀疑论者访谈中，里贝罗](https://dataskeptic.com/blog/transcripts/2016/trusting-machine-learning-models-with-lime)还指出

“我们把它作为一个产生解释的框架。我们做了一些特别的解释，特别是线性模型，但是 LIME 基本上是一个试图平衡可解释性和忠实性的等式。所以我会说这是一个框架，你可以用 LIME 框架得出许多不同种类的解释。”

## 为什么信任是机器学习中的问题？

Riberio、Singh 和 Guestrin “认为解释预测是让人类信任和有效使用机器学习的一个重要方面，如果解释是可信和可理解的”。他们还认为，“机器学习实践者经常必须从许多备选方案中选择一个模型，这要求他们评估两个或更多模型之间的相对信任度”以及

*“每个机器学习应用程序也需要对模型的整体信任度进行一定的测量。分类模型的开发和评估通常包括收集* *带注释的数据，其中一个保留的子集用于自动评估。虽然这对于许多应用来说是一个有用的管道，但是对验证数据的评估可能并不对应于“在野外”的性能，因为从业者经常高估他们的模型的准确性[20]，因此信任不能仅仅依赖于它。查看示例提供了一种评估模型真实性的替代方法，尤其是在示例得到解释的情况下。”*

过度拟合、数据泄漏、数据集偏移(训练数据不同于测试数据)是“模型或其评估可能出错”的几种方式。这些类型的挑战导致人们在模型开发和部署期间评估是否信任模型。

## LIME 如何解决信任问题？

为了了解模型的潜在行为，LIME 使人类能够以对人类有意义的不同方式干扰输入，观察预测可能如何变化，然后人类评估是否信任特定任务的模型。Riberio、Singh 和 Guestrin 在许多论文和演讲中使用图像分类作为例子。[在树蛙示例](https://www.oreilly.com/learning/introduction-to-local-interpretable-model-agnostic-explanations-lime)中，Riberio、Singh 和 Guestrin 评估分类器是否能够“预测图像包含树蛙的可能性”。他们将树蛙图像分割成不同的可解释部分，然后扰乱、“屏蔽”或隐藏各种可解释部分。然后

*“对于每一个被扰乱的实例，我们根据模型得到一只树蛙在图像中的概率。然后，我们在这个数据集上学习一个简单的(线性)模型，这个模型是局部加权的——也就是说，我们更关心在与原始图像更相似的扰动实例中出错。最后，我们给出了具有最高正权重的超级像素作为解释，将其他所有东西都变灰。”*

然后，人类能够观察模型预测，将其与他们自己对图像的观察进行比较，并评估是否信任该模型。里贝里奥在数据怀疑论者访谈中指出,“我们人类有直觉...我们知道在很多情况下一个好的解释是什么样的，如果我们看到模型是否以合理的方式运作，我们就会更倾向于相信它。”

## 要考虑的资源

这篇博客文章通过摘录石灰研究的重点，提供了一个蒸馏石灰的概述。如果您有兴趣了解更多关于 LIME 的知识，并进一步评估 LIME 作为一种可行的工具来帮助您决定是否信任您的模型，请考虑阅读这篇博文中回顾和引用的资源:

### Python/R 代码

*   [使用属性重要性、PDP 和时间解释黑盒模型](/blog/explaining-black-box-models-using-attribute-importance-pdps-and-lime)
*   [LIME 开源项目](https://github.com/marcotcr/lime)
*   [Python 石灰包的 R 口](https://github.com/thomasp85/lime)

### 报纸

*   里贝里奥、辛格和盖斯特林的“'[为什么](https://arxiv.org/pdf/1602.04938.pdf) [我应该相信你”:解释任何分类器的预测](https://arxiv.org/pdf/1602.04938.pdf)
*   Riberio、Singh 和 Guestrin 的"[机器学习的模型不可知可解释性](http://sameersingh.org/files/papers/lime-whi16.pdf)"
*   里贝里奥、辛格和盖斯特林的“[锚:高精度模型不可知论解释](https://homes.cs.washington.edu/~marcotcr/aaai18.pdf)

### 视频和采访

*   数据怀疑论者的[“用石灰信任机器学习模型”](https://dataskeptic.com/blog/transcripts/2016/trusting-machine-learning-models-with-lime)
*   KDD2016 "' [我为什么要相信你':解释任何分类器的预测](https://www.youtube.com/watch?v=KP7-JtFMLo4)"

### 博客帖子

*   Riberio、Singh 和 Guestrin 的“[对局部可解释模型不可知解释的介绍(LIME)](https://www.oreilly.com/learning/introduction-to-local-interpretable-model-agnostic-explanations-lime)
*   托马斯·林·彼得森和迈克尔·贝尼斯特的《理解石灰》

*^(Domino 数据实验室提供数据科学研究的亮点、趋势、技术等，支持数据科学家和数据科学领导者加快他们的工作或职业发展。如果你对我们博客中报道的你的数据科学工作感兴趣，请发邮件到 writeforus@dominodatalab.com 给我们。)*