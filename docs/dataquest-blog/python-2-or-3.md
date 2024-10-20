# 我应该学 Python 2 还是 3？(以及为什么它很重要)

> 原文：<https://www.dataquest.io/blog/python-2-or-3/>

September 28, 2022

如果您想知道应该学习 Python 2 还是 Python 3，您并不孤单。

这是我们在 [Dataquest](https://www.dataquest.io/) 最常听到的问题之一，在那里，我们将 Python 作为数据科学课程的一部分来教授。

让我们解决这个问题。

剧透:在 Dataquest，我们只教 Python 3。如果您准备好现在就开始学习，请免费注册[Python 简介](https://www.dataquest.io/course/introduction-to-python/)课程！

如果你仍然犹豫不决，或者只是想了解更多信息，请继续阅读。这篇文章给出了问题背后的一些背景，解释了我们的立场，并告诉你应该学习哪个版本。

**如果你想要简单的答案，这里就是:**你应该学习 Python 3，因为它是与当今数据科学项目最相关的版本。另外，它很容易学习，很少需要担心兼容性问题。

需要更彻底的解释吗？让我们先简要回顾一下 Python 2 对 3 之争背后的历史。

## 2008 年:Python 3.0 的诞生

不，那不是印刷错误。Python 3 发布于 2008 年。

如果您是 Python 2-3 争论的新手，请注意这场争论已经酝酿了将近 15 年了！仅此一点就应该告诉你这是多么大的一件事。

### **2008 年的向后不兼容版本**

Python 于 2008 年 12 月 3 日发布了 3.0 版本。这个版本的特别之处在于它是一个[向后不兼容的版本](https://snarky.ca/why-python-3-exists/)。

作为这种向后不兼容的结果，将项目从 Python 2.x 迁移到 3.x 将需要很大的改变。这不仅包括单个项目和应用程序，还包括构成 Python 生态系统一部分的所有库。

### **Python 3 反冲**

当时，这一变化被视为极具争议。因此，许多项目抵制了迁移的痛苦，尤其是在科学 Python 社区中。例如，主数字库 NumPy 花了整整两年时间才发布了它的第一个 3.x 版本！

在接下来的几年里，其他项目开始发布 3.x 兼容版本。到 2012 年，许多库已经支持 3.x，但大多数仍然是用 2.x 编写的。随着时间的推移，工具的发布使得移植代码更加容易，但仍然有很大的阻力。

在接下来的几年里，发布了几个工具来帮助旧代码库从 Python 2 过渡到 Python 3。

最初，Python 将 Python 2.x 的“生命终结”日期定在 2015 年。然而，在 2014 年，他们宣布将终止日期延长至 2020 年。这样做的部分原因是为了减轻那些还不能迁移到 Python 3 的用户的担忧。

然而，Python 2 的日子显然是有限的。2017 年，流行的网络框架 Django [宣布他们的新 2.0 版本将不支持 Python 2.x](https://news.ycombinator.com/item?id=13433927) 。

此外，许多软件包开始宣布停止支持 2.x。甚至科学图书馆[也承诺](https://www.python3statement.org/)在 2020 年或更早之前停止支持 2.x。

## 快进到今天:为什么这还是一个问题？

今天，很少有库不支持 Python 3。

但是，如果 Python 2.x 不再受支持，那么为什么围绕 Python 2 对 3 的问题仍然存在混乱呢？

答案是双重的。

首先，网上有很多基于 Python 2 的旧的免费学习 Python 的资源。这包括来自 Coursera、Udemy 和 edX 等平台的大多数 MOOC 课程。

由于数据科学专业的学生总是希望省钱，这些免费资源很有吸引力。

另外，Zed Shaw 非常受欢迎的*Learn Python Hard Way*是用 Python 2.x 编写的。他直到 2017 年才写了一本关于 Python 3 的书[——几乎是在它发布整整十年之后！](https://www.amazon.com/Learn-Python-Hard-Way-Introduction/dp/0134692888/ref=pd_lpo_1?pd_rd_i=0134692888&psc=1)

直到最近，我还以为这只是因为泽德这些年来太懒了，没有更新他的课程。但是后来我发现了他那篇有争议的文章:[针对 Python 3 的案例](https://learnpythonthehardway.org/book/nopython3.html)。

尽管 Eevee 对 Python 3 进行了精彩的反驳，Zed 的抨击还是造成了损失。当然，今天赞同 Zed 的人数是极少数。但是整个争议减缓了从 Python 2 到 3 的过渡。这也给这个领域的许多新来者搅浑了水。

## **那么我应该学习哪种 Python 呢？**

关于 Python 2 和 3 的所有争论，您可能会认为学习其中一个会是一个困难的决定。然而实际上，这很简单。

### **Python 3 是明显的赢家**

Python 3.x 是未来，随着 Python 2.x 支持的减少，您应该花时间学习将会持续的版本。

如果您担心兼容性问题，不必担心。我只使用 Python 3.x，很少遇到兼容性问题。

偶尔(可能每 3-4 个月一次)，我会发现我正在尝试运行一些需要 Python 2 支持的东西。在这些罕见的情况下，Python 的 [virtualenv](https://virtualenv.pypa.io/en/stable/) 允许我立即在我的机器上创建一个 2.x 环境来运行那部分遗留软件。

### **不要在 Python 2 上浪费时间**

让我们明确一点:Python 2 已经过时了。Python 2.x 不会有未来的安全或 bug 修复，你的时间最好花在学习 3.x 上。

万一你最终使用的是一个遗留的 Python 2 代码库，像 [Python-Future](https://python-future.org/) 这样的工具会让你在只学过 Python 3 的情况下轻松使用。

Dataquest 是学习成为使用 Python 的数据科学家的最佳在线平台(当然是 3.x！).我们有毕业生在 SpaceX、亚马逊等公司工作。如果您对此感兴趣，您可以在 [Dataquest.io](https://www.dataquest.io/) *注册并免费完成我们的第一门课程。*