# 评估生成对抗网络

> 原文：<https://www.dominodatalab.com/blog/evaluating-generative-adversarial-networks-gans>

*本文提供了对 GANs 的简明见解，以帮助数据科学家和研究人员评估是否要进一步调查 GANs。如果你对教程以及 Domino 项目中的实践代码示例[感兴趣，那么考虑参加即将到来的网络研讨会，“](https://try.dominodatalab.com/u/nmanchev/GAN/overview)[生成对抗网络:精华教程](https://www.brighttalk.com/webcast/17563/379668)”。*

## 介绍

随着主流对 deepfakes 越来越多的关注，生成敌对网络(GANs)也进入了主流聚光灯下。毫不奇怪，这种主流关注也可能导致数据科学家和研究人员提出问题，或评估是否在自己的工作流中利用 GANs。虽然 Domino 提供了一个平台，让行业能够利用他们选择的语言、工具和基础设施来支持模型驱动的工作流，但我们也致力于支持数据科学家和研究人员评估像 GANs 这样的框架是否有助于加快他们的工作。这篇博客文章提供了对 GANs 的高级见解。如果你对一个 [Domino 项目](https://try.dominodatalab.com/u/nmanchev/GAN/overview)中的教程和实践代码示例感兴趣，那么考虑参加即将到来的网络研讨会，[“生成对抗网络:一个精选教程”](https://www.brighttalk.com/webcast/17563/379668)，由 EMEA Domino 首席数据科学家尼古拉·曼切夫主持。这两个[多米诺项目](https://try.dominodatalab.com/u/nmanchev/GAN/overview)

![Domino Data Lab Enterprise MLOPs workspace](img/6b1e1dd9d48df10435ae75f1f1e5bdfa.png) *[Domino Project: https://try.dominodatalab.com/workspaces/nmanchev/GAN](https://try.dominodatalab.com/workspaces/nmanchev/GAN)*

![JupyterLab space in Domino](img/e143e8fb167e937b850a47e38b22fa7c.png)

[网络研讨会](https://www.brighttalk.com/webcast/17563/379668)比这篇博文更有深度，因为它们涵盖了一个基本 GAN 模型的实现，并展示了如何使用敌对网络来生成训练样本。

## 为什么是甘斯？考虑区别性与生成性

关于区分性量词和生成性量词的利弊问题已经争论了很多年。判别模型利用观察到的数据并捕捉给定观察的条件概率。从统计学的角度来看，逻辑回归是区别对待的一个例子。[反向传播算法](https://www.deeplearningbook.org/contents/mlp.html#pf25)或 backprop，从深度学习的角度来看，是一种成功的判别方法的例子。

生成式不同，因为它利用了[联合概率分布](https://en.wikipedia.org/wiki/Joint_probability_distribution)。生成模型学习数据的分布，并洞察给定示例的可能性。然而，伊恩·古德菲勒、让·普盖-阿巴迪、迈赫迪·米尔扎、徐炳、戴维·沃德-法利、谢尔吉尔·奥泽尔、亚伦·库维尔和约舒阿·本吉奥在论文《[生成对抗网络](https://papers.nips.cc/paper/5423-generative-adversarial-nets)》中指出

*“由于在最大似然估计和相关策略中出现的许多难以处理的概率计算的近似困难，以及由于在生成上下文中利用分段线性单元的优势的困难，深度生成模型的影响[比鉴别性的]要小。我们提出了一种新的生成模型估计过程，避开了这些困难*

Goodfellow 等人提出了 GANs 并解释道:

*“在提出的对抗性网络框架中，生成模型与对手对抗:一个学习确定样本是来自模型分布还是数据分布的判别模型。生成模型可以被认为类似于一队伪造者，试图生产假币并在不被发现的情况下使用，而辨别模型类似于警察，试图检测假币。这场游戏中的竞争促使两个团队改进他们的方法，直到无法区分假冒品和真品。”*

## GANs:可能对半监督学习和多模型设置有用

gan 对于半监督学习以及包含未标记数据或部分数据样本被标记的数据的情况是有用的。当寻找对应于单个输入的多个正确答案时(即，多模态设置)，生成模型也是有用的。在后续教程中， [Goodfellow 引用](https://arxiv.org/pdf/1701.00160.pdf)

*“生成模型可以用缺失数据进行训练，并可以对缺失数据的输入进行预测。缺失数据的一个特别有趣的例子是半监督学习，其中许多甚至大多数训练样本的标签都是缺失的。现代深度学习算法通常需要极其多的标记样本才能很好地概括。半监督学习是减少标签数量的一种策略。该学习算法可以通过学习大量通常更容易获得的未标记样本来提高其泛化能力。生成模型，尤其是 GANs，能够相当好地执行半监督学习。”*

和

*“生成模型，尤其是 GANs，使机器学习能够处理多模态输出。对于许多任务，单个输入可能对应许多不同的正确答案，每个答案都是可接受的。一些训练机器学习模型的传统手段，如最小化期望输出和模型预测输出之间的均方误差，无法训练出可以产生多个不同正确答案的模型。”*

## 结论

这篇博文集中于对 GANs 的简明高层次见解，以帮助研究人员评估是否要进一步调查 GANs。如果对教程感兴趣，那么请查看即将举行的网络研讨会[“生成式对抗网络:精选教程”](https://www.brighttalk.com/webcast/17563/379668)，由 Domino EMEA 首席数据科学家尼古拉·曼切夫主持。网上研讨会和[补充 Domino 项目](https://try.dominodatalab.com/u/nmanchev/GAN/overview)比这篇博文提供了更多的深度，因为它们涵盖了一个基本 GAN 模型的实现，并演示了如何使用敌对网络来生成训练样本。