# 选择正确的机器学习框架

> 原文：<https://www.dominodatalab.com/blog/choosing-the-right-machine-learning-framework>

## 什么是机器学习框架？

机器学习(ML)框架是允许数据科学家和开发人员更快更容易地构建和部署机器学习模型的接口。机器学习几乎被用于每个行业，特别是[金融](/blog/deep-learning-machine-learning-uses-in-financial-services)、保险、[医疗](https://analytics-solution.pharmatechoutlook.com/vendor/domino-data-lab-imbuing-innovation-in-modeldriven-businesses-cid-324-mid-38.html)和[营销](https://formation.ai/blog/ai-and-machine-learning-for-marketers-101/)。使用这些工具，企业可以扩展他们的机器学习工作，同时保持高效的 ML 生命周期。

公司可以选择构建自己的定制机器学习框架，但大多数组织都会选择适合自己需求的现有框架。在本文中，我们将展示为您的项目选择正确的机器学习框架的关键考虑因素，并简要回顾四个流行的 ML 框架。

## 如何选择合适的机器学习框架

在为您的项目选择机器学习框架时，您应该考虑以下几个关键因素。

### 评估您的需求

当你开始寻找一个机器学习框架时，问这三个问题:

1.  你会使用深度学习的框架还是经典的机器学习算法？
2.  对于人工智能(AI)模型开发，你首选的编程语言是什么？
3.  哪些硬件、软件和云服务用于扩展？

Python 和 R 是机器学习中广泛使用的语言，但其他语言如 C、Java 和 Scala 也是可用的。今天，大多数机器学习应用程序都是用 Python 编写的，并且正在从 R 过渡，因为 R 是由统计学家设计的，使用起来有些笨拙。Python 是一种更现代的编程语言，它提供了简单明了的语法，并且更易于使用。

### 参数最优化

机器学习算法使用不同的方法来分析训练数据，并将所学应用于新的例子。

算法有参数，你可以把它想象成一个仪表板，上面有控制算法如何运行的开关和转盘。他们调整要考虑的变量的权重，定义在多大程度上考虑离群值，并对算法进行其他调整。在选择机器学习框架时，重要的是要考虑这种调整应该是自动的还是手动的。

### 扩展培训和部署

在 AI 算法开发的[训练阶段](/blog/what-is-machine-learning-model-training)，可扩展性就是可以分析的数据量和分析的速度。通过分布式算法和处理，以及通过使用硬件加速，主要是[图形处理单元(GPU)](/data-science-dictionary/gpu)，可以提高性能。

在 AI 项目的部署阶段，可伸缩性与可以同时访问模型的并发用户或应用程序的数量有关。

因为在培训和[部署阶段](/blog/machine-learning-model-deployment)有不同的要求，所以组织倾向于在一种类型的环境中开发模型(例如，在云中运行的基于 Python 的机器学习框架),并在对性能和高可用性有严格要求的不同环境中运行它们，例如，在本地数据中心。

在选择一个框架时，考虑它是否支持这两种类型的可伸缩性，并查看它是否支持您计划的开发和生产环境是很重要的。

## 顶级机器学习框架

让我们来看看目前使用的一些最流行的机器学习框架:

*   TensorFlow
*   PyTorch
*   Sci-Kit 学习
*   H2O

### TensorFlow

TensorFlow 由 Google 创建，并作为开源项目发布。它是一个多功能和强大的机器学习工具，具有广泛和灵活的函数库，允许您构建分类模型、回归模型、神经网络和大多数其他类型的机器学习模型。这还包括根据您的特定要求定制机器学习算法的能力。TensorFlow 在 CPU 和 GPU 上都可以运行。TensorFlow 的主要挑战是它对于初学者来说不容易使用。

TensorFlow 的主要特性:

*   **计算图形的可见性** —TensorFlow 可以轻松地可视化算法计算过程的任何部分(称为图形)，这是 Numpy 或 SciKit 等旧框架所不支持的。
*   **模块化** —TensorFlow 高度模块化，您可以独立使用其组件，而不必使用整个框架。
*   **分布式训练** —TensorFlow 为 CPU 和 GPU 上的分布式训练提供了强大的支持。
*   **并行神经网络训练** —TensorFlow 提供管道，让你并行训练多个神经网络和多个 GPU，在大型分布式系统上非常高效。

随着 TensorFlow 2.0 的发布，TensorFlow 增加了几个重要的新功能:

*   **在多种平台上部署** -使用 SavedModel 格式提高移动设备、物联网和其他环境的兼容性，使您可以将 Tensorflow 模型导出到几乎任何平台。
*   **急切执行** -在 Tensorflow 1.x 中，用户需要构建整个计算图并运行它，以便测试和调试他们的工作。Tensorflow 2.0 和 PyTorch 一样，支持热切执行。这意味着模型可以在构建时被修改和调试，而不需要运行整个模型。
*   Keras 的更紧密集成 -以前，Keras 受 TensorFlow 支持，但不作为库的一部分集成。在 TensorFlow 2.x 中，Keras 是 TensorFlow 附带的官方高级 API。
*   **改进了对分布式计算的支持** -改进了使用 GPU 的训练性能，比 Tensorflow 1.x 快三倍，以及与多个 GPU 和 Google TensorFlow 处理单元(TPU)配合工作的能力。

### PyTorch

PyTorch 是一个基于 Torch 和 Caffe2 的机器学习框架，非常适合神经网络设计。PyTorch 是开源的，支持基于云的软件开发。它支持用户界面开发的 Lua 语言。它与 Python 集成，并与 Numba 和 Cython 等流行库兼容。与 [Tensorflow 不同，PyTorch 更直观](/blog/tensorflow-pytorch-or-keras-for-deep-learning)并且初学者掌握起来更快。

PyTorch 的主要特点:

*   通过使用本地 Python 代码进行模型开发，支持快速执行和更大的灵活性。
*   从开发模式快速切换到图形模式，在 C++运行时环境中提供高性能和更快的开发。
*   使用异步执行和对等通信来提高模型训练和生产环境中的性能。
*   提供端到端的工作流，允许您使用 Python 开发模型并在 iOS 和 Android 上部署。PyTorch API 的扩展处理将机器学习模型嵌入移动应用程序所需的常见预处理和集成任务。

### Sci-Kit 学习

SciKit Learn 是开源的，对于机器学习的新手来说非常用户友好，并且附带了详细的文档。它允许开发人员在使用中或运行时更改算法的预设参数，从而易于调整和排除模型故障。

SciKit-Learn 通过广泛的 Python 库支持机器学习开发。它是数据挖掘和分析的最佳工具之一。Sci-Kit Learn 具有广泛的预处理能力，并支持用于聚类、分类、回归、维度减少和模型选择的算法和模型设计。

Scikit-Learn 的主要功能:

*   **支持大多数监督学习算法**—线性回归、支持向量机(SVM)、决策树、贝叶斯等。
*   **支持无监督学习算法**—聚类分析、因子分解、主成分分析(PCA)和无监督神经网络。
*   **执行特征提取和交叉验证**—从可以提取的文本和图像中提取特征，并在新的看不见的数据上测试模型的准确性。
*   **支持聚类和集成技术**—可以组合来自多个模型的预测，并且可以对未标记的数据进行分组。

### H2O

H2O 是一个开源的 ML 框架，旨在解决决策支持系统流程的组织问题。它与其他框架集成，包括我们上面讨论的框架，来处理实际的模型开发和训练。H2O 广泛应用于风险和欺诈趋势分析、保险客户分析、医疗保健行业的患者分析、广告成本和投资回报以及客户智能。

H2O 组件包括:

*   **深水**—将 H2O 与 TensorFlow 和 Caffe 等其他框架集成在一起。
*   **苏打水**—将 H2O 与大数据处理平台 Spark 整合。
*   **Steam**—企业版，支持训练和部署机器学习模型，通过 API 使其可用，并将它们集成到应用程序中。
*   **无人驾驶 AI**—使非技术人员能够准备数据、调整参数，并使用 ML 来确定解决特定业务问题的最佳算法。

## 使用 Domino 的机器学习框架

通过 Domino 的[环境管理](https://docs.dominodatalab.com/en/4.1/reference/environments/Environment_management.html)特性，为您的用例选择正确的 ML 框架比以往任何时候都更容易。您可以轻松构建环境，并让它们在最佳计算资源上运行，无论是 CPU、GPU 还是 APU。

Domino 中的环境很容易配置，并且包括以下主要特性

*   **版本控制** -召回升级可能破坏模型代码或显著改变结果的环境的先前版本
*   **选择您自己的 IDE GUI**——包括在 Domino workbench 解决方案中使用的任何基于 HTML 浏览器的 GUI
*   **与数据科学家同事轻松共享您的环境** -在服务器托管实例中，获得与笔记本电脑相同的灵活性，能够与同事即时共享代码和环境
*   **针对不同用例的不同环境** -为了获得最佳的服务器利用率，只安装对您的代码需要运行的工作流至关重要的包，拥有多个环境以充分利用您的服务器资源

关于机器学习框架如何在 Domino 中运行的示例，请查看我们下面展示 PyTorch、Tensorflow 和 Ludwig 的一些文章。

*   [用 36 行代码改进 Zillow 的 Zestimate】](/blog/zillow-kaggle)
*   [图像分类器的数据漂移检测](/blog/data-drift-detection-for-image-classifiers)
*   [路德维希深度学习实践指南](/blog/a-practitioners-guide-to-deep-learning-with-ludwig)