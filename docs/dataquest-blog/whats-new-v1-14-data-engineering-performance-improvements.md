# 1.14 版的新功能:数据工程路径和性能改进！

> 原文：<https://www.dataquest.io/blog/whats-new-v1-14-data-engineering-performance-improvements/>

March 2, 2017Our latest Dataquest release has over 20 new features, including many major performance improvements and the launch of our much-anticipated data engineering path.

## 新路径:数据工程

我们的第一道菜

[数据工程路径](https://www.dataquest.io/path/data-engineer/)到了！![](img/d24ff601759f440d78431c36ff1f89be.png)数据工程是一个广阔的领域，包括:

*   使用大数据
*   构建分布式系统
*   创建可靠的管道
*   组合数据源
*   与数据科学团队合作，为他们构建合适的解决方案。

如果您想了解更多关于数据工程的信息，您可以阅读我们的指南:

[什么是数据工程师？](https://www.dataquest.io/blog/what-is-a-data-engineer/)这条新路线的第一门课程是[处理 pandas](https://www.dataquest.io/course/pandas-large-datasets) 中的大型数据集，它包括各种技术，扩展了我们现有的 [pandas 课程](https://www.dataquest.io/course/python-for-data-science-intermediate/)，并向您展示如何使用 pandas 处理更大的数据集。

### 课程大纲

本课程包含五课，包括两个指导项目:

*   [优化数据帧内存占用](https://app.dataquest.io/m/163) **空闲**
*   [分块处理数据帧](https://app.dataquest.io/m/164)
*   [指导性项目:练习优化数据帧和分块处理](https://app.dataquest.io/m/165)
*   [用 SQLite 扩充熊猫](https://app.dataquest.io/m/166)
*   [引导项目:分析来自 Crunchbase 的创业融资交易](https://app.dataquest.io/m/167)

像我们所有的路径一样，数据工程路径将是一个持续的迭代展示——我们将在这一年中不断增加更多的经验和课程！

[免费开始我们新的数据工程之路！](https://app.dataquest.io/m/163)

## 性能改进

我们的技术团队一直在努力工作，他们继续让代码运行得更快，并提高我们各种课程类型的稳定性，让您的学习体验更好。

### 减少代码运行时间

基于我们在 9 月份所做的更改，我们已经将所有代码运行转移到基于 websockets，这使得代码运行时间减少了 21%。在过去的六个月中，我们的代码运行时间中位数从 4.2 秒减少到了 1.7 秒，减少了 59%。

![](img/37f4e5e4c44f553fb96dd0ab833c587c.png)过去六个月代码运行次数。

### 控制台和命令行课程的稳定性改进

此外，我们重新设计了命令行课程和控制台的通信，这将提高这些课程的可靠性和稳定性。我们还引入了一个网络和容器状态面板，它将让您了解运行容器和连接的代码的状态。

![](img/2920616f05f3be65916df653e26aa0aa.png)网络和容器状态面板。

## 1.14 版本中的新功能

今天版本中的完整功能列表如下:

*   启动我们新的[数据工程途径](https://www.dataquest.io/path/data-engineer/)的第一门课程。
*   改进了代码运行时间。
*   为运行容器的代码启用了池，以减少由于容器初始化而导致的等待时间。
*   改进了控制台和命令行课程的稳定性。
*   一个新的通知源，当有人回复或支持你的问答帖子时，它会显示给你。
*   将内部通信消息系统移到标题，使界面更整洁。
*   改进了我们支付页面的设计和功能。
*   添加了网络和容器状态面板，以帮助排除故障。
*   在命令行课程中添加了一个重置按钮，可以帮助你在卡住的时候将步骤恢复到正确的状态。
*   当您的付款由于错误的邮政编码而被拒绝时，我们现在会显示更详细的信息。
*   重写了我们的 [Python 基础](https://www.dataquest.io/course/python-for-data-science-fundamentals/)课程，使其更加清晰。
*   使我们的第一门课程——[Python 编程:初学者](https://www.dataquest.io/course/python-for-data-science-fundamentals/)——完全免费，并增加了两个指导项目。
*   添加了一个新的引导项目:[使用下载的数据](https://app.dataquest.io/m/220)。
*   删除了旧的数据可视化课程，该课程在 11 月被两个新课程所取代。
*   修正了不同课程中的其他小错误。
*   修正了一个错误，一些用户不能生成证书。
*   修正了一个用户不能更新电子邮件地址的错误。
*   为通过 facebook 注册的用户添加了一个输入电子邮件地址的提示。
*   修正了防火墙后的某些用户不能运行代码的错误。
*   修复了 Microsoft Edge v14 阻止用户登录和创建帐户的问题。
*   其他小的错误修复和稳定性改进。

## 即将推出

如果您对未来几周的内容感兴趣:

*   数据工程路径中的第二个课程
*   一门新的“机器学习基础”课程
*   能够在每节课中下载源数据集。
*   一个新的课程结束屏幕，让你更容易给我们反馈。
*   一个新的课程界面，等等！

一如既往，我们希望听到您对我们下一步应该构建什么的反馈。如果你有一个好主意，

给我们发电子邮件！