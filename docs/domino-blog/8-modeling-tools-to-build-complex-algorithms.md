# 构建复杂算法的 8 个建模工具

> 原文：<https://www.dominodatalab.com/blog/8-modeling-tools-to-build-complex-algorithms>

对于一个模型驱动的企业来说，获得适当的工具意味着亏损经营和一系列后期项目在你面前徘徊或者超出生产力和盈利能力预测的区别。这一点也不夸张。借助合适的工具，您的数据科学团队可以专注于他们最擅长的领域，即测试、开发和部署新模型，同时推动前瞻性创新。

## 什么是建模工具？

一般来说，模型是一系列算法，当给定适当的数据时，这些算法可以解决问题。正如人脑可以通过将我们从过去的经验中学到的教训应用到新的情况来解决问题一样，模型是根据一组数据训练的，可以用新的数据集解决问题。

### 建模工具的重要性

如果没有完整和无限制的建模工具，数据科学团队将会被束缚在工作中。开源创新比比皆是，并将继续下去。数据科学家需要获得现代工具，这些工具受到 IT 部门的青睐，但可以通过自助服务的方式获得。

## 建模工具的类型

在选择工具之前，你应该首先知道你的最终目标——机器学习或深度学习。

机器学习使用主要基于传统统计学习方法的算法来识别数据中的模式。这对分析结构化数据最有帮助。

深度学习有时被认为是机器学习的子集。基于神经网络的概念，它对于分析图像、视频、文本和其他非结构化数据非常有用。深度学习模型也往往更加资源密集型，需要更多的 CPU 和 GPU 能力。

### 深度学习建模工具:

1.  PyTorch: 是一个免费的开源库，主要用于深度学习应用，如自然语言处理和计算机视觉。它基于火炬图书馆。但是，它更喜欢 Python，而不是 Torch 使用的 Lua 编程语言。Pytorch 的大部分是由脸书的 AI 研究实验室开发的，该实验室尊重修改后的 BSD 开源许可。基于 PyTorch 构建的深度学习模型的例子包括特斯拉的 Autopilot 和优步的 Pyro。
2.  **TensorFlow:** 类似于 PyTorch，这是 Google 创建的一个开源 Python 库。相比之下，它的主要优势之一是支持 Python 之外的其他语言。它也被认为比 PyTorch 更面向生产。然而，这是有争议的，因为这两个工具都在不断更新它们的特性。您可以在 TensorFlow 内创建深度学习模型，或者对基于 TensorFlow 构建的模型使用包装器库。Airbnb 用它来给照片加标题和分类，而 GE Healthcare 用它来识别大脑核磁共振成像的解剖结构。
3.  **Keras:** 是一个构建在 TensorFlow 之上的 API。虽然 TensorFlow 确实有自己的 API 来减少所需的编码量，但 Keras 通过在 TensorFlow 库的基础上添加一个简化的接口来扩展这些功能。它不仅减少了大多数任务所需的用户操作数量，而且您还可以仅用几行代码来设计和测试人工[神经网络模型。它被新手用来快速学习深度学习，以及从事高级项目的团队，包括 NASA、CERN 和 NIH。](/blog/deep-learning-illustrated-building-natural-language-processing-models)
4.  Ray: 是一个开源的库框架，它提供了一个简单的 API，用于[将应用从单台计算机扩展到大型集群](/blog/evaluating-ray-distributed-python-for-massive-scalability)。Ray 包括一个名为 RLib 的可扩展强化学习库和一个名为 Tune 的可扩展超参数调优库。
5.  **Horovod:** 是另一个分布式深度学习训练框架，可以和 PyTorch、TensorFlow、Keras、Apache MXNet 一起使用。它与 Ray 的相似之处在于，它主要是为同时跨多个 GPU 进行扩展而设计的。Horovod 最初由优步开发，是开源的，可以通过 GitHub 获得。

### 机器学习建模工具

6. **Scikit-Learn:** 是 Python 中最健壮的机器学习库之一。该库包含一系列用于机器学习和统计建模的工具，包括分类、回归、聚类和降维以及预测数据分析。这是一个基于 BSD 许可的开源但商业上可用的库，构建在 NumPy、SciPy 和 matplotlib 之上。

7. **XGBoost:** 是另一个开源的机器学习库，为 Python、C++、Java、R、Perl、Scala 提供了一个[正则化梯度提升框架](/blog/credit-card-fraud-detection-using-xgboost-smote-and-threshold-moving)。它不仅在各种平台上提供稳定的模型性能，而且是目前最快的梯度提升框架之一。

8. **Apache Spark:** 是一个开源的统一分析引擎，专为[扩展数据处理](/blog/pca-on-very-large-neuroimaging-datasets-using-pyspark)需求而设计。它为具有并行数据的多个编程集群提供了一个简单的用户界面。也很快。

## Domino 数据实验室建模工具

虽然这些是今天用于 AI/ML 模型的一些最流行的工具，但这绝不是一个详尽的列表。要在同一个平台上探索这些工具的功能，请看一下 Domino 的企业 MLOps，它提供了[14 天免费试用](https://www.dominodatalab.com/trial/)或[点击这里观看快速演示](/demo)。

[![14-day free trial  Try The Domino Enterprise MLOps Platform Get started](img/4b2c6aa363d959674d8585491f0e18b8.png)](https://cta-redirect.hubspot.com/cta/redirect/6816846/28f05935-b374-4903-9806-2b4e86e1069d)