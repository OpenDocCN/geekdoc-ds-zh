# 理解模型-视图-控制器

模型-视图-控制器（MVC）可能是用户界面设计和 Web 编程中最常用的架构解决方案；MVC 首次在 70 年代引入，逐渐被适应和改变为各种子类型和变体，以至于普通术语“MVC”没有额外的限定词已经失去了特定性。作为一种一般性解释，它是一个相当宽松的指导原则，用于在应用程序的数据和可视化部分需要互动但尽可能保持松散耦合时组织代码。在实践中如何实现这一点取决于特定的 MVC 化身。

MVC [](GLOSSARY.html) 可以 [](GLOSSARY.html) 被视为 [](GLOSSARY.html) 更 [](GLOSSARY.html) 基本的 [](GLOSSARY.html) 设计 [](GLOSSARY.html) 模式的 [](GLOSSARY.html) 聚合，比如 [](GLOSSARY.html) Composite，Mediator [](GLOSSARY.html) 和 [](GLOSSARY.html) Observer/Notifier。 [](GLOSSARY.html) MVC [](GLOSSARY.html) 的 [](GLOSSARY.html) 复杂性 [](GLOSSARY.html) 和 [](GLOSSARY.html) 风格 [](GLOSSARY.html) 的 [](GLOSSARY.html) 变化 [](GLOSSARY.html) 来自于 [](GLOSSARY.html) 这些 [](GLOSSARY.html) 独立的 [](GLOSSARY.html) 模式 [](GLOSSARY.html) 的 [](GLOSSARY.html) 所有 [](GLOSSARY.html) 可能 [](GLOSSARY.html) 的 [](GLOSSARY.html) 用法 [](GLOSSARY.html) 和 [](GLOSSARY.html) 变化，以 [](GLOSSARY.html) 满足 [](GLOSSARY.html) GUI [](GLOSSARY.html) 的 [](GLOSSARY.html) 可能 [](GLOSSARY.html) 复杂的 [](GLOSSARY.html) 需求。

本 [](GLOSSARY.html) 书 [](GLOSSARY.html) 的 [](GLOSSARY.html) 目标 [](GLOSSARY.html) 是 [](GLOSSARY.html) 探索 [](GLOSSARY.html) MVC [](GLOSSARY.html) 的 [](GLOSSARY.html) 变化 [](GLOSSARY.html) 和 [](GLOSSARY.html) 细微之处，对 [](GLOSSARY.html) 它们 [](GLOSSARY.html) 进行 [](GLOSSARY.html) 比较 [](GLOSSARY.html) 和 [](GLOSSARY.html) 分析。 其中 [](GLOSSARY.html) 的 [](GLOSSARY.html) 区别 [](GLOSSARY.html) 特征 [](GLOSSARY.html) 在于 [](GLOSSARY.html) 给 [](GLOSSARY.html) 主要 [](GLOSSARY.html) 角色 [](GLOSSARY.html) 分配 [](GLOSSARY.html) 职责，具体地说是 [](GLOSSARY.html) 在 [](GLOSSARY.html) 用户 [](GLOSSARY.html) 和 [](GLOSSARY.html) 应用 [](GLOSSARY.html) 状态 [](GLOSSARY.html) 之间 [](GLOSSARY.html) 的 [](GLOSSARY.html) 交互 [](GLOSSARY.html) 中 “谁 [](GLOSSARY.html) 负责 [](GLOSSARY.html) 什么” 和 “谁 [](GLOSSARY.html) 知道 [](GLOSSARY.html) 谁”的 [](GLOSSARY.html) 分配。 [](GLOSSARY.html) MVC [](GLOSSARY.html) 的 [](GLOSSARY.html) 变化 [](GLOSSARY.html) 以不同的 [](GLOSSARY.html) 方式 [](GLOSSARY.html) 分配 [](GLOSSARY.html) 新的 [](GLOSSARY.html) 和 [](GLOSSARY.html) 旧的 [](GLOSSARY.html) 职责，连接 [](GLOSSARY.html) 或 [](GLOSSARY.html) 组织 [](GLOSSARY.html) 主要 [](GLOSSARY.html) 角色，或者 [](GLOSSARY.html) 添加 [](GLOSSARY.html) 中间 [](GLOSSARY.html) 对象 [](GLOSSARY.html) 以 [](GLOSSARY.html) 获得 [](GLOSSARY.html) 更多的 [](GLOSSARY.html) 灵活性 [](GLOSSARY.html) 和 [](GLOSSARY.html) 满足 [](GLOSSARY.html) 特殊的 [](GLOSSARY.html) 使用 [](GLOSSARY.html) 情况。

本 [](GLOSSARY.html) 书 [](GLOSSARY.html) 的 [](GLOSSARY.html) 结构 [](GLOSSARY.html) 如 [](GLOSSARY.html) 下：

+   第一章将通过代码介绍一个简单的从头开始的 MVC 应用程序，目标是部署一个通用的词汇表。该章将定义组件、角色和通信模式，并以一句话结束，说明由此得出的公式在现代软件开发中已经过时且过于简单。

+   一旦装备了术语表，第二章将介绍 MVC 变体，以解决特定的 UI 约束和实际需求，或者提高开发效率。

+   第三章将扩展 MVC 的概念到分层 MVC 方案。

+   第四章将专注于从复杂的现代 GUI 中产生的特殊技术。

+   在第五和最后一章中，我们将专注于 Web MVC 及其实现。

在整本书中，将会展示示例代码或实际实现来阐明设计思想。GUI 渲染将利用优秀的 Qt 工具包。Qt 提供了预制机制来满足一些 MVC 需求，但在即将展示的代码中，这些机制将被故意跳过，以展示所呈现的概念。

### 致谢和动机

我[](GLOSSARY.html)开始[](GLOSSARY.html)写[](GLOSSARY.html)这本[](GLOSSARY.html)书[](GLOSSARY.html)是[](GLOSSARY.html)一种[](GLOSSARY.html)偶然。起初，我[](GLOSSARY.html)想要[](GLOSSARY.html)写一系列博客文章来描述模型-视图-控制器及一些相关模式。随着我从网络和个人经验中收集了越来越多的信息，我突然发现我所写的内容的数量和结构超出了博客的范围，因此决定将其重新定义为一本书。我对这个决定感到满意，因为它给了我添加我本来不会添加的材料的自由。

这个[](GLOSSARY.html)作品[](GLOSSARY.html)展示了[](GLOSSARY.html)和[](GLOSSARY.html)丰富了[](GLOSSARY.html)设计[](GLOSSARY.html)解决方案，最佳[](GLOSSARY.html)实践和[](GLOSSARY.html)实验[](GLOSSARY.html)由[](GLOSSARY.html)无数[](GLOSSARY.html)博客[](GLOSSARY.html)文章[](GLOSSARY.html)和[](GLOSSARY.html)评论[](GLOSSARY.html)提供。对[](GLOSSARY.html)这些作者[](GLOSSARY.html)表示[](GLOSSARY.html)我的[](GLOSSARY.html)感谢和[](GLOSSARY.html)感激。作为[](GLOSSARY.html)一个[](GLOSSARY.html)正在[](GLOSSARY.html)进行中的[](GLOSSARY.html)工作，还有[](GLOSSARY.html)很多[](GLOSSARY.html)工作[](GLOSSARY.html)要[](GLOSSARY.html)做。请[](GLOSSARY.html)耐心等待，但[](GLOSSARY.html)请[](GLOSSARY.html)随时[](GLOSSARY.html)给我[](GLOSSARY.html)反馈，提出[](GLOSSARY.html)请求，并[](GLOSSARY.html)利用[](GLOSSARY.html)已经[](GLOSSARY.html)存在的[](GLOSSARY.html)材料。

这本[](GLOSSARY.html)书是[](GLOSSARY.html)根据[](GLOSSARY.html)GFDL[](GLOSSARY.html)许可证发布的，免费（免费），主要是因为三个原因

+   如[](GLOSSARY.html)所述，这里[](GLOSSARY.html)呈现的大部分[](GLOSSARY.html)材料是[](GLOSSARY.html)从[](GLOSSARY.html)网络上[](GLOSSARY.html)收集的。这是[](GLOSSARY.html)我[](GLOSSARY.html)个人的[](GLOSSARY.html)努力[](GLOSSARY.html)来[](GLOSSARY.html)组织这些[](GLOSSARY.html)知识，但[](GLOSSARY.html)我[](GLOSSARY.html)有一个[](GLOSSARY.html)较低的[](GLOSSARY.html)起步[](GLOSSARY.html)门槛。

+   我[](GLOSSARY.html)已经[与商业出版商合作出版了一本书](http://www.amazon.com/Computing-Comparative-Microbial-Genomics-Microbiologists/dp/1849967636)，从[](GLOSSARY.html)我的[](GLOSSARY.html)经验和[](GLOSSARY.html)数学来看，我[](GLOSSARY.html)认为如果[](GLOSSARY.html)我[](GLOSSARY.html)在网上[](GLOSSARY.html)出版一本书，并[](GLOSSARY.html)接受[](GLOSSARY.html)捐赠，我[](GLOSSARY.html)可能[](GLOSSARY.html)会[](GLOSSARY.html)获得更多的[](GLOSSARY.html)钱和[](GLOSSARY.html)反馈，而不是通过[](GLOSSARY.html)出版商。

+   本书是我作为软件开发和设计专业人士组合的一部分，我很自豪地专注于一个个人项目以增强我的能力。

话虽如此，我很乐意接受捐赠：

+   在[GratiPay](https://gratipay.com/StefanoBorini/)上

+   在比特币上

+   在 PayPal 上（使用我的电子邮件地址 stefano.borini 在 ferrara.linux.it）

本书的来源可作为一个 github 仓库在以下 URL 处找到：

[`github.com/stefanoborini/modelviewcontroller-src`](https://github.com/stefanoborini/modelviewcontroller-src)

我还有一个个人网站[`forthescience.org`](http://forthescience.org)，您可以在那里找到更多关于我、我的简历和活动的信息。
