# 使用贝叶斯方法清理人类标签

> 原文：<https://www.dominodatalab.com/blog/using-bayesian-methods-to-clean-up-human-labels>

## 会话摘要

Derrick Higgins 在最近的 [Data Science Popup](https://popup.dominodatalab.com/?utm_source=blog&utm_medium=post&utm_campaign=using-bayesian-methods-to-clean-up-human-labels) 会议上，深入研究了在收集和创建数据集时如何使用贝叶斯方法来提高注释质量。希金斯从覆盖模型开始，这些模型是“称为概率图形模型的建模家族”的一部分。概率图形模型常用于贝叶斯统计。然后，Higgins 介绍了“一个基于高斯混合模型的示例，展示了如何开始使用 Stan 并在没有太多该框架的经验的情况下构建模型。”

会议的主要亮点包括

*   需要带有 y 变量的标注或注释数据，以使用监督模型进行预测
*   不是每个人都能得到干净的数据集，比如学术界使用的 MNIST 和虹膜数据集
*   覆盖模型，如 IRT，梅斯，贝叶斯网络(贝叶斯网)，潜在的狄利克雷分配(LDA)，斯坦(使用 PyStan)和更多

希金斯在会议结束时提出了两点:“一是思考你的数据。考虑如何创建数据，以便对建模有用。不要认为任何事情都是理所当然的，马上开始建模。第二，图形模型:不像以前那么难了。”

要了解更多关于本次会议的见解，请观看视频或通读文字记录。

[https://fast.wistia.net/embed/iframe/7xiha29s07](https://fast.wistia.net/embed/iframe/7xiha29s07)