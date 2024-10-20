# 容器对于数据科学的真正价值

> 原文：<https://www.dominodatalab.com/blog/the-real-value-of-containers-for-data-science>

每年，你交的税中有 50 美元被投入到无法复制的研究中。
气候公司科学副总裁埃里克·安德列科在 2016 年圣荷西 Strata+Hadoop World 大会上发言

本周在圣何塞的 Strata+Hadoop 世界博览会上，很明显，供应商们已经抓住了容器的漏洞。去年毕竟是容器的[年。令我印象深刻的是，Docker 等容器技术主要用于计算基础设施。然而，数据科学家的真正价值不在这里。](http://www.infoworld.com/article/3016800/virtualization/2015-the-year-containers-upended-it.html)

供应商正在提供[可扩展的计算和存储]，发言者正在讨论在容器上部署[Hadoop，或者在 Hadoop 上运行容器。但是，计算基础设施只会间接影响数据科学家开展研究的能力。虽然有人努力将容器化的](http://conferences.oreilly.com/strata/hadoop-big-data-ca/public/schedule/detail/47489)[科学研究和 Python 堆栈](https://github.com/jupyter/docker-stacks/tree/master/datascience-notebook)带到桌面上，但对于许多最终用户来说，这是一项不小的工作。

在昨天的会议上，气候公司的科学副总裁埃里克·安德列科谈到了“把科学放在数据科学中”。他强调再现性原则是科学方法的核心。然而，科学正面临着一场危机:无法重现研究成果。一个相当令人吃惊的例子:最近的[研究](http://www.nature.com/nature/journal/v483/n7391/full/483531a.html)发现，药物发现项目中近 90%的研究是不可重复的(关于这一点的更多信息，请参见我们最近[对埃里克](//blog.dominodatalab.com/building-a-high-throughput-data-science-machine/)的采访)。

这就是容器在数据科学中的真正价值:在某个时间点捕获实验状态(数据、代码、结果、包版本、参数等)的能力，使得在研究过程的任何阶段重现实验成为可能。

可重复性对于监管环境下的定量研究至关重要，例如，记录贷款模型的出处以证明其避免了种族偏见。然而，即使在不受监管的环境中，再现性对于数据科学项目的成功也是至关重要的，因为它有助于捕捉稍后可能会浮出水面的假设和偏见。

科学过程是建立在不断积累的洞察力上的。这种积累可能是由一组数据科学家在一段时间内合作推动的，也可能是由一名数据科学家根据过去的经验构建的。只有过去的结果是可重复的，这个过程才有效。

容器是支持数据科学的科学方法的理想技术。我们从一开始就在 Domino 中大量使用 Docker，我们很高兴在接下来的博客文章中分享更多关于使用容器的经验和教训。