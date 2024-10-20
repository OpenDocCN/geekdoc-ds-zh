# 用 Plotly 和 Domino 可视化机器学习

> 原文：<https://www.dominodatalab.com/blog/visualizing-data-with-plotly-and-domino>

我最近有机会与 Domino Data Lab 合作制作了一个网上研讨会，演示了如何使用 Plotly 在 Domino 笔记本中创建数据可视化。在这篇文章中，我将分享我在一起使用 Plotly 和 Domino 时发现的一些好处。

[Plotly](https://www.dominodatalab.com/data-science-dictionary/plotly) 是一个面向数据科学家和工程师的基于网络的数据可视化平台。我们的平台背后的引擎是 plotly.js，这是一个基于 D3.js 和 stack.gl 构建的开源图表库。我们有 R、Python 和 MATLAB 的 API，使数据科学家可以轻松地使用他们选择的编程语言，并且可以访问使用多种语言的团队。这在 Domino 平台上运行得非常好，在那里您可以用各种语言创建笔记本。下面我将展示两个例子，一个用 R 语言，一个用 Python 语言。

* * *

## 普罗特利河

首先，让我们看看 R 中的一个例子。我将使用 Plotly 的 twitter 帐户中的一些 twitter 数据来展示如何使用 Plotly 的 R 库为 K-Means 聚类创建一个交互式的二维可视化。这个例子是基于之前的 [*r-bloggers* 的帖子](http://rstudio-pubs-static.s3.amazonaws.com/5983_af66eca6775f4528a72b8e243a6ecf2d.html)。可以说，我最喜欢的 Domino 笔记本特性是在浏览器选项卡中打开 RStudio 会话的能力。我强烈建议在 Domino 中启动一个 RStudio 会话，并亲自测试一下！

## 设置

在 R 中，我们将首先安装必要的包，加载库，然后使用一个 twitter 应用程序启动身份验证并从 twitter 获取数据。