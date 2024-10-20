# 模型评估

> 原文：<https://www.dominodatalab.com/blog/model-evaluation>

*这份多米诺数据科学领域笔记提供了 [Alice Zheng](http://alicezheng.org/) 的报告“[评估机器学习模型](https://www.oreilly.com/data/free/evaluating-machine-learning-models.csp)”的一些亮点，包括监督学习模型的评估指标和离线评估机制。[全面深入的报告](https://www.oreilly.com/data/free/evaluating-machine-learning-models.csp)还涵盖了离线与在线评估机制、超参数调整和潜在的 A/B 测试缺陷[可供下载](https://www.oreilly.com/data/free/evaluating-machine-learning-models.csp)。作为报告补充的[精选幻灯片](https://www.slideshare.net/AliceZheng3/evaluating-machine-learning-models-a-beginners-guide)也可供使用。*

## 为什么模型评估很重要

数据科学家做模型。我们经常会听到数据科学家讨论他们如何负责建立一个模型作为产品或者建立一系列相互依赖的影响商业战略的 T2 模型。机器学习模型开发的一个基本且具有挑战性的方面是评估其性能。与假设数据分布保持不变的统计模型不同，机器学习模型中的数据分布可能会随时间漂移。评估模型和检测分布漂移使人们能够识别何时需要重新训练机器学习模型。在 [Alice Zheng](http://alicezheng.org/) 的[“评估机器学习模型”](https://www.oreilly.com/data/free/evaluating-machine-learning-models.csp)报告中，Zheng 主张在任何项目开始时考虑模型评估，因为它将有助于回答诸如“我如何衡量这个项目的成功？”并避免“在好的度量是模糊的或不可行的情况下，在不完善的项目上工作”。

## 监督学习模型的评价指标

郑指出“开发一个机器学习模型有多个阶段…..因此，我们需要在多个地方对模型进行评估”。郑主张在原型阶段，或“我们尝试不同的模型以找到最好的一个(模型选择)”时考虑模型评估。郑还指出，“评估指标与机器学习任务相关联”，“任务有不同的指标”。郑在报告中涉及的一些评估指标包括监督学习的分类、回归和排名。郑还提到要考虑的两个包包括 [R 的度量包](https://cran.r-project.org/web/packages/Metrics/Metrics.pdf)和 [scikit-learn 的模型评估。](http://scikit-learn.org/stable/modules/model_evaluation.html)

### 分类

关于分类，郑提到，用于测量分类性能的最流行的度量包括准确度、[混淆矩阵](https://www.coursera.org/learn/big-data-machine-learning/lecture/o4hXx/confusion-matrix)、对数损失和 [AUC(曲线下面积)](http://www.dataschool.io/roc-curves-and-auc-explained/)。虽然准确性“衡量分类器做出正确预测的频率”，因为它是“正确预测的数量与预测的总数(测试集中的数据点数量)之间的比率”，但[混淆矩阵](http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/)(或混淆表)显示了每个类别的正确和错误分类的更详细的细分。郑指出，当想要了解类之间的区别时，使用混淆矩阵是有用的，特别是当“两个类的错误分类的成本可能不同，或者一个类的测试数据可能比另一个多得多”时。“例如，在癌症诊断中做出假阳性或假阴性的后果是不同的。

至于对数损失(对数损失)，郑指出，“如果分类器的原始输出是数字概率，而不是 0 或 1 的类别标签，则可以使用对数损失。概率可以被理解为信心的衡量标准…..因为它是一个“软”的准确性度量，包含了概率置信度的思想。至于 AUC，郑将其描述为“一种将 ROC 曲线总结成一个数字的方法，这样它就可以容易地和自动地进行比较”。ROC 曲线是一条完整的曲线，并且“提供了分类器的细微细节”。关于 AUC 和 ROC 的更多解释，郑推荐[本教程](http://www.dataschool.io/roc-curves-and-auc-explained/)。

### 等级

郑指出，“一个主要的排名指标，精确召回，也是分类任务的流行”。虽然这是两个指标，但它们通常一起使用。郑指出“在数学上，精度和召回率可以定义如下:

*   precision = #快乐的正确答案/# ranker 返回的项目总数
*   回忆= #快乐正确答案/ #相关项目总数。"

此外，“在底层实现中，分类器可以向每个项目分配数字分数，而不是分类类标签，并且排序器可以简单地通过原始分数对项目进行排序。”郑还指出，个人推荐可能是排名问题或回归模型的另一个例子。郑注意到“推荐者既可以作为排名者，也可以作为分数预测者。在第一种情况下，输出是每个用户的排序项目列表。在分数预测的情况下，推荐器需要返回每个用户-项目对的预测分数——这是回归模型的一个例子”。

### 回归

对于回归，郑在报告中指出，“在回归任务中，模型学习预测数字分数。”如前所述，个性化推荐是指我们“试图预测用户对某个项目的评分”。郑还指出，“回归任务最常用的指标之一是(均方根误差)，也称为(均方根偏差)。然而，郑警告说，虽然 RSME 是普遍使用的，但也有一些挑战。RSMEs 对大的异常值特别敏感。如果回归器在单个数据点上表现很差，平均误差可能很大”或者“平均值不是稳健的(对于大的异常值)”。郑指出，真实数据中总会有“异常值”，“模型对它们的表现可能不会很好”。因此，寻找不受较大异常值影响的稳健估计值非常重要。”郑提议查看[中值绝对百分比](https://arxiv.org/pdf/1605.02541.pdf)是有用的，因为它“为我们提供了典型误差的相对度量”

## 离线评估机制

郑在文中主张

模型必须在一个数据集上进行评估，该数据集在统计上独立于它被训练的数据集。为什么？因为它在训练集上的性能是对其在新数据上的真实性能的过于乐观的估计。训练模型的过程已经适应了训练数据。一个更公平的评估将衡量模型在它尚未看到的数据上的表现。从统计学的角度来说，这给出了泛化误差的估计值，它衡量了模型对新数据的泛化能力。

郑还指出，研究人员可以使用保留验证作为生成新数据的一种方式。保留验证，“假设所有数据点都是独立同分布的(独立同分布)，我们只是随机保留部分数据进行验证。我们在较大部分的数据上训练模型，并在较小的保留集上评估验证指标。”郑还指出，当需要一种生成额外数据集的机制时，也可以使用重采样技术，如自举或交叉验证。“自举”通过从单个原始数据集中采样来生成多个数据集。每个“新”数据集可用于估计感兴趣的数量。由于有多个数据集，因此有多个估计值，人们也可以计算估计值的方差或置信区间等内容。”郑指出，交叉验证“在训练数据集如此之小，以至于人们无法承受仅出于验证目的而保留部分数据时是有用的。”虽然交叉验证有许多变体，但最常用的一种是 [k 倍交叉验证](https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation),它

*“将训练数据集分成 k 倍…k 个折叠中的每一个都轮流作为拒绝验证集；在剩余的 k -1 个褶皱上训练一个模型，并在保持褶皱上测量。整体性能是所有 k 倍性能的平均值。对所有需要评估的超参数设置重复此过程，然后选择产生最高 k 倍平均值的超参数。”*

郑还指出 [sckit-learn 交叉验证模块](http://scikit-learn.org/stable/modules/cross_validation.html)可能是有用的。

## 摘要

由于数据科学家在制作模型上花费了大量时间，因此尽早考虑评估指标可能有助于数据科学家加快工作速度，并为项目的成功做好准备。然而，评估机器学习模型是一个已知的挑战。这篇 Domino 数据科学领域笔记提供了摘自郑报告的一些见解。完整的深度报告可从[下载](https://www.oreilly.com/data/free/evaluating-machine-learning-models.csp)。

[![The Practical Guide to  Accelerating the Data Science Lifecycle  Lessons from the field on becoming a model-driven businesses.   Read the Guide](img/733c37e12c2c7c37295fb3198e3a226a.png)](https://cta-redirect.hubspot.com/cta/redirect/6816846/c77ca351-ae85-425a-9ee3-c264b3bc4a69) 

*^(Domino 数据科学领域笔记提供数据科学研究、趋势、技术等亮点，支持数据科学家和数据科学领导者加快工作或职业发展。如果您对本博客系列中涉及的数据科学工作感兴趣，请发送电子邮件至 writeforus(at)dominodatalab(dot)com。)*