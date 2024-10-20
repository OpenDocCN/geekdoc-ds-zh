# 你应该知道的 15 个数据科学 Python 库

> 原文：<https://www.dataquest.io/blog/15-python-libraries-for-data-science/>

February 5, 2020![python-libraries-for-data-science](img/b0655c3e4cc55778e5d2a0125ba93448.png)

Python 是数据科学家和软件开发人员用于数据科学任务的最流行的语言之一。它可用于预测结果、自动化任务、简化流程以及提供商业智能见解。

在普通 Python 中处理数据是可能的，但是有相当多的开源库使得 Python 数据任务变得非常非常容易。

你肯定听说过其中的一些，但是有没有你可能错过的有用的库呢？这里列出了 Python 生态系统中最重要的数据科学任务库，涵盖了数据处理、建模和可视化等领域。

## 数据挖掘

#### 1.[刺儿头](https://github.com/scrapy/scrapy)

Scrapy 是最受欢迎的 Python 数据科学库之一，它帮助构建爬行程序(蜘蛛机器人),可以从网络上检索结构化数据，例如 URL 或联系信息。这是一个很好的收集数据的工具，例如，Python 机器学习模型。

开发人员使用它从 API 收集数据。这个成熟的框架在其界面设计中遵循了“不要重复自己”的原则。因此，该工具鼓励用户编写通用代码，这些代码可以重用来构建和扩展大型爬虫。

#### 2.[美丽的风景](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)

BeautifulSoup 是另一个非常受欢迎的网页抓取和数据抓取库。如果你想收集一些网站上的数据，但不是通过适当的 CSV 或 API，BeautifulSoup 可以帮助你收集数据，并将其整理成你需要的格式。

## 数据处理和建模

#### 3.[NumPy](https://github.com/numpy/numpy)

NumPy(数值 Python)是科学计算和执行基本和高级数组操作的完美工具。

这个库提供了许多在 Python 中对 n 数组和矩阵执行操作的便利特性。它有助于处理存储相同数据类型值的数组，并使对数组执行数学运算(及其矢量化)更加容易。事实上，NumPy 数组类型上数学运算的向量化提高了性能并加快了执行时间。

#### 4.[轨道](https://github.com/scipy/scipy)

这个有用的库包括线性代数、积分、优化和统计模块。它的主要功能是建立在 NumPy 之上的，所以它的数组使用这个库。SciPy 非常适合各种科学编程项目(科学、数学和工程)。它在子模块中提供了高效的数值例程，如数值优化、积分等。大量的文档使得使用这个库变得非常容易。

#### 5.[熊猫](https://github.com/pandas-dev/pandas)

Pandas 是一个库，用来帮助开发人员直观地处理“标签”和“关系”数据。它基于两种主要的数据结构:“系列”(一维，像一个项目列表)和“数据框”(二维，像一个有多个列的表)。Pandas 允许将数据结构转换为 DataFrame 对象，处理丢失的数据，从 DataFrame 中添加/删除列，输入丢失的文件，用直方图或绘图框绘制数据。这是数据争论、操作和可视化的必备工具。

*(想学熊猫？查看 [Dataquest 的 NumPy 和熊猫基础课程](https://www.dataquest.io/course/pandas-fundamentals/)，或者我们众多[免费熊猫教程](https://www.dataquest.io/blog/tag/tutorial+pandas/)中的一个。)*

#### 6\. [Keras](https://github.com/keras-team/keras)

Keras 是一个用于构建神经网络和建模的伟大的库。它使用起来非常简单，并为开发人员提供了很好的可扩展性。该库利用其他包(Theano 或 TensorFlow)作为其后端。此外，微软整合了 CNTK(微软认知工具包)作为另一个后端。如果你想使用紧凑的系统快速进行实验，这是一个很好的选择——极简主义的设计方法真的有回报！

#### 7\. [SciKit-Learn](https://github.com/scikit-learn/scikit-learn)

这是基于 Python 的数据科学项目的行业标准。Scikits 是 SciPy 堆栈中的一组包，它们是为特定功能(例如图像处理)而创建的。Scikit-learn 使用 SciPy 的数学运算向最常见的机器学习算法展示了一个简洁的接口。

数据科学家使用它来处理标准的机器学习和数据挖掘任务，如聚类、回归、模型选择、维度减少和分类。另一个优势？它配有高质量的文档，并提供高性能。

#### 8.[指针](https://github.com/pytorch/pytorch)

PyTorch 是一个非常适合希望轻松执行深度学习任务的数据科学家的框架。该工具允许使用 GPU 加速执行张量计算。它还用于其他任务，例如，创建动态计算图和自动计算梯度。PyTorch 基于 Torch，这是一个用 C 实现的开源深度学习库，封装器在 Lua 中。

#### 9\. [TensorFlow](https://github.com/tensorflow/tensorflow)

TensorFlow 是一个流行的用于机器学习和深度学习的 Python 框架，由 Google Brain 开发。它是对象识别、语音识别和许多其他任务的最佳工具。它有助于处理需要处理多个数据集的人工神经网络。该库包括各种层助手(tflearn、tf-slim、skflow)，这使得它的功能更加强大。TensorFlow 不断推出新版本，包括潜在安全漏洞的修复或 TensorFlow 和 GPU 集成的改进。

#### 10. [XGBoost](https://github.com/dmlc/xgboost)

使用这个库在梯度提升框架下实现机器学习算法。XGBoost 具有可移植性、灵活性和高效性。它提供并行树提升，帮助团队解决许多数据科学问题。另一个优势是，开发人员可以在 Hadoop、SGE 和 MPI 等主流分布式环境中运行相同的代码。

## 数据可视化

#### 11. [Matplotlib](https://github.com/matplotlib/matplotlib)

这是一个标准的数据科学库，有助于生成数据可视化，如二维图表和图形(直方图、散点图、非笛卡尔坐标图)。Matplotlib 是那些在数据科学项目中非常有用的绘图库之一，它提供了一个面向对象的 API，用于将绘图嵌入到应用程序中。

正是由于这个库，Python 可以与 MatLab 或 Mathematica 这样的科学工具竞争。然而，开发人员在使用这个库生成高级可视化时，需要编写比平时更多的代码。请注意，流行的绘图库可以与 Matplotlib 无缝协作。

#### 12. [Seaborn](https://github.com/mwaskom/seaborn)

Seaborn 基于 Matplotlib，是一个有用的 Python 机器学习工具，用于可视化统计模型——热图和其他类型的可视化，这些模型汇总数据并描述整体分布。使用这个库时，您可以从大量的可视化图库中获益(包括复杂的图形，如时间序列、联合图和小提琴图)。

#### 13.[散景](https://github.com/bokeh/bokeh)

这个库是一个很好的工具，可以使用 JavaScript 小部件在浏览器中创建交互式和可伸缩的可视化。Bokeh 完全独立于 Matplotlib。它侧重于交互性，并通过现代浏览器呈现可视化——类似于数据驱动文档(d3.js)。它提供了一组图形、交互能力(比如链接图或添加 JavaScript 小部件)和样式。

#### 14.[阴谋地](https://github.com/plotly/plotly.py)

这款基于网络的数据可视化工具提供了许多有用的现成图形，您可以在 [Plot.ly 网站](https://plot.ly/)上找到它们。该库在交互式 web 应用程序中工作得非常好。它的创建者正忙于用新的图形和特性来扩展库，以支持多链接视图、动画和相声集成。

#### 15.[pydot](https://github.com/pydot/pydot)

这个库有助于生成有向图和无向图。它充当 Graphviz(用纯 Python 编写)的接口。在这个库的帮助下，你可以很容易地显示图形的结构。当你开发基于神经网络和决策树的算法时，这是很方便的。

## 结论

这个列表并不完整，因为 Python 生态系统提供了许多其他工具来帮助完成机器学习任务和构建算法。参与使用 Python 的数据科学项目的数据科学家和软件工程师将使用许多这些工具，因为它们对于用 Python 构建高性能 ML 模型是必不可少的。

你知道其他对 ML 项目有用的库吗？请告诉我们您认为对 Python 数据生态系统至关重要的其他工具！

____

*这篇文章是一家专注于 Python 的软件开发公司 [Sunscrapers](https://sunscrapers.com/python-development-services/) 的客座贡献。Sunscrapers 主办并赞助了许多 Python 活动和会议，鼓励其工程师分享他们的知识并参与[开源项目](https://github.com/sunscrapers)。*