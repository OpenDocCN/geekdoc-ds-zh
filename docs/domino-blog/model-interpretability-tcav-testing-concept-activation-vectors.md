# 用 TCAV 的模型可解释性(用概念激活向量测试)

> 原文：<https://www.dominodatalab.com/blog/model-interpretability-tcav-testing-concept-activation-vectors>

*这份多米诺数据科学领域笔记提供了来自 [Been Kim](https://twitter.com/_beenkim) 最近的 [MLConf 2018 talk](https://youtu.be/Ff-Dx79QEEY) 和 [research](http://proceedings.mlr.press/v80/kim18d/kim18d.pdf) 关于使用概念激活向量(TCAV)进行测试的非常精炼的见解和摘录，这是一种可解释性方法，允许研究人员理解和定量测量他们的神经网络模型用于预测的高级概念，“[即使概念不是训练的一部分](https://www.slideshare.net/SessionsEvents/interpretability-beyond-feature-attribution-quantitative-testing-with-concept-activation-vectors-tcav-123005780)。如果对这篇博文中没有提供的其他见解感兴趣，请参考 [MLConf 2018 视频](https://youtu.be/Ff-Dx79QEEY)、 [ICML 2018 视频](https://www.youtube.com/watch?v=DNk-hcSV1pY)和[论文](http://proceedings.mlr.press/v80/kim18d/kim18d.pdf)。*

## 介绍

如果有一种方法可以定量地衡量你的机器学习(ML)模型是否反映了特定领域的专业知识或潜在的偏见，会怎么样？有培训后的讲解？在全球层面上而不是在当地层面上？业内人士会感兴趣吗？这是谷歌大脑高级研究科学家[金](https://beenkim.github.io/)在 [MLConf 2018 演讲](https://youtu.be/Ff-Dx79QEEY)、*“特征归因之外的可解释性:用概念激活向量(TCAV)进行测试”中提出的问题。*ml conf 演讲基于 Kim 合著的[论文](http://proceedings.mlr.press/v80/kim18d/kim18d.pdf)并且[代码可用](https://github.com/tensorflow/tcav)。这份 Domino 数据科学领域笔记提供了一些关于 TCAV 的精华见解，这是一种可解释性方法，允许研究人员理解和定量测量他们的神经网络模型用于预测的高级概念，“即使该概念不是训练的一部分”( [Kim Slide 33](https://www.slideshare.net/SessionsEvents/interpretability-beyond-feature-attribution-quantitative-testing-with-concept-activation-vectors-tcav-123005780) )。TCAV“使用方向导数来量化用户定义的概念对分类结果的重要程度”( [Kim et al 2018](http://proceedings.mlr.press/v80/kim18d/kim18d.pdf) )。

## 引入概念激活向量(CAV)

所以。什么是概念激活向量或 CAV？CAVs =研究人员创造的名称，用于将高层概念表示为向量，“激活”是“价值观的方向”。

在研究中， [Kim 等人](http://proceedings.mlr.press/v80/kim18d/kim18d.pdf)指出

*“一个概念的 CAV 只是该概念的示例集的值(例如，激活)方向上的一个向量…我们通过在概念的示例和随机反例之间训练一个线性分类器，然后取垂直于决策边界的向量，来导出 CAV。”*

在 [MLConf 演讲](https://youtu.be/Ff-Dx79QEEY)中，Kim 转述道,“嗯，这个向量只是意味着一个方向指向远离随机的方向，更像是这个概念。拥有这样一个载体的想法并不新鲜，因为性别方向在 [word2vec 的一篇论文](https://arxiv.org/pdf/1607.06520.pdf)中讨论过。[金还指出](https://youtu.be/Ff-Dx79QEEY)骑士队“只是找回那个载体的另一种方式”。

## 用概念激活向量测试(TCAV):斑马

在[研究](http://proceedings.mlr.press/v80/kim18d/kim18d.pdf)中用来展示 TCAV 的一个突出的图像分类例子是“斑马的预测对条纹的存在有多敏感”。在 [MLConf 会谈](https://www.youtube.com/watch?v=Ff-Dx79QEEY&feature=youtu.be)中，Kim 解开获得 TCAV 分数的量化解释

“TCAV 的核心思想是对 logit 层进行方向求导，即斑马纹相对于我们刚刚得到的向量的概率……直观地说，这意味着如果我让这幅图更像这个概念，或者更不像这个概念，斑马的概率会有多大变化？如果变化很大，这是一个重要的概念，如果变化不大，这不是一个重要的概念。我们对很多很多的斑马图片都是这样做的，以获得更有力和可靠的解释。最后的分数，TCAV 分数，仅仅是斑马图片的比率，其中条纹积极地增加了斑马的概率。换句话说，如果你有 100 张斑马的图片，其中有多少张返回了正方向导数？就是这样。”

如 MLConf 讲座的[第 38 张幻灯片所示，同样需要注意的是](https://www.slideshare.net/SessionsEvents/interpretability-beyond-feature-attribution-quantitative-testing-with-concept-activation-vectors-tcav-123005780)

“当且仅当你的人际网络了解了一个概念，TCAV 才提供了这个概念的数量重要性”。

## 为什么考虑 TCAV？

在 MLConf 的演讲中，Kim 主张将可解释性作为一种工具，致力于“更负责、更安全地”使用 ML。在[的论文内，](http://proceedings.mlr.press/v80/kim18d/kim18d.pdf)金等人指出

“TCAV 是朝着创建对深度学习模型内部状态的人类友好的线性解释迈出的一步，以便关于模型决策的问题可以根据自然的高级概念来回答”。

当向不是 ML 专家的涉众翻译看似黑箱的模型时，使用高层概念可能是有用的。例如，当构建一个模型来帮助医生进行诊断时，使用医生友好的高级概念来测试您的模型以消除偏见或领域专业知识水平可能会有所帮助。这篇博客文章提供了最近关于 TCAV 研究的精华和摘录。如果有兴趣了解更多信息，请参考以下在撰写本文时参考和回顾的资源。

## 密码

*   [TCAV Github](https://github.com/tensorflow/tcav)

## MLConf 2018

*   [幻灯片](https://www.slideshare.net/SessionsEvents/interpretability-beyond-feature-attribution-quantitative-testing-with-concept-activation-vectors-tcav-123005780)
*   [视频](https://youtu.be/Ff-Dx79QEEY)

## ICML 2018: TCAV 相关链接

*   [在 ICML 发表的论文](http://proceedings.mlr.press/v80/kim18d/kim18d.pdf)
*   [幻灯片](https://beenkim.github.io/slides/TCAV_ICML_pdf.pdf)
*   [视频](https://www.youtube.com/watch?v=DNk-hcSV1pY)

## 其他相关文件和幻灯片

*   朱利叶斯·阿德巴约等人。[显著图的健全性检查](https://arxiv.org/abs/1810.03292)
*   Been Kim，"[可解释机器学习简介](https://beenkim.github.io/slides/DLSS2018Vector_Been.pdf)"
*   多米诺，“[让机器学习更严谨](https://blog.dominodatalab.com/make-machine-learning-interpretability-rigorous/)”

*^(Domino 数据科学领域笔记提供数据科学研究、趋势、技术等亮点，支持数据科学家和数据科学领导者加快工作或职业发展。如果您对本博客系列中涉及的数据科学工作感兴趣，请发送电子邮件至 writeforus(at)dominodatalab(dot)com。)*