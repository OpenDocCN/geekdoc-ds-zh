# 提供数字出处:从建模到生产

> 原文：<https://www.dominodatalab.com/blog/providing-digital-provenance-modeling-production>

在上周的*用户！在 R User* 大会上，我谈到了数字起源、可重复研究的重要性，以及 Domino 如何解决数据科学家在尝试这种最佳实践时面临的许多挑战。关于这个话题的更多信息，请听下面的录音。

你正在做什么来确保你正在减轻与出处(或缺少出处)相关的许多风险？

[https://player.vimeo.com/video/175955786?title=0&byline=0&portrait=0](https://player.vimeo.com/video/175955786?title=0&byline=0&portrait=0)

## 谈话摘要

在整个数据科学过程中，再现性非常重要。最近的研究表明，项目探索性分析阶段的潜意识偏见会对最终结论产生巨大影响。管理生产中模型的部署和生命周期的问题是巨大的和多样的，并且再现性通常停留在单个分析师的水平上。虽然 R 对可重复研究有最好的支持，有像 KnitR to packrat 这样的工具，但是它们的范围有限。

在本次演讲中，我们将展示一个我们在 Domino 开发的解决方案，它允许生产中的每个模型从 EDA 到训练运行和用于生成的精确数据集都具有完全的再现性。我们讨论如何利用 Docker 作为我们的再现引擎，以及这如何允许我们提供无可辩驳的模型出处。