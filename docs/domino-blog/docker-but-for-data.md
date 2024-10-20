# Docker，但是对于数据

> 原文：<https://www.dominodatalab.com/blog/docker-but-for-data>

*[棉被](https://quiltdata.com/)的联合创始人兼首席技术官 Aneesh Karve 访问了 Domino MeetUp，讨论数据基础设施的发展。这篇博文提供了会议摘要、视频和演示文稿。Karve 也是[“用 Jupyter 和 Quilt 进行可复制的机器学习”](//blog.dominodatalab.com/reproducible-machine-learning-with-jupyter-and-quilt/)的作者。*

Session Summary

被子数据的联合创始人兼首席技术官 Aneesh Karve 讨论了数据像源代码一样被管理的必然性。版本、包和编译是软件工程的基本要素。他问道，如果没有这些元素，我们为什么要做数据工程呢？Karve 深入研究了 Quilt 的开源工作，以创建一个跨平台的数据注册表，该注册表可以存储、跟踪和整理进出内存的数据。他进一步阐述了生态系统技术，如 Apache Arrow、Presto DB、Jupyter 和 Hive。

本次会议的一些亮点包括:

*   数据包代表“可部署、版本化、可重用的数据元素”
*   *“我们观察了过去三年的计算轨迹，发现容器的垂直增长不仅带来了标准化，而且成本大大降低，生产率大大提高。从虚拟化的角度来看，您接下来要考虑的是数据。我们能为数据创建一个类似 Docker 的东西吗？”*
*   如果你有代码依赖，你` pip install -r requirements.txt `，然后你` import foo `，` import bar `然后开始工作。数据依赖注入要混乱得多。您通常有许多数据准备代码，从网络获取文件，将这些文件写入磁盘，然后有自定义的解析例程，然后运行数据清理脚本。
*   了解了脸书通量模式后，看看 React，不变性提供了很多可预测性。
*   计算和存储的分离，如 Presto DB 等技术所示。与其衡量你收集了多少数据，不如衡量你能多快将数据转化为代码。*那叫数据回报。*
*   鉴于代码的 linter 和语法检查器无处不在，提倡为数据使用 linter，以确保数据符合特定的配置文件。
*   DDL，数据描述语言，是棉被编译器和蜂巢生态系统之间的桥梁。
*   如果你有 Drill，pandas，Cassandra，Kudu，HBase，Parquet，所有这些节点都得到了很好的优化，它们是高性能的。然而，你实际上在哪里度过你的时间呢？Karve 指出，您花费 80%的时间来移动数据，您失去了所有这些性能优化，并且您还有许多非规范化的、重复的逻辑。

[https://fast.wistia.net/embed/iframe/at7vn34fbo](https://fast.wistia.net/embed/iframe/at7vn34fbo)