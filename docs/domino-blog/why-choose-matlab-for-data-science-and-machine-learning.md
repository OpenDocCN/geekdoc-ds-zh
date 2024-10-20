# 用于数据科学和机器学习的 MATLAB

> 原文：<https://www.dominodatalab.com/blog/why-choose-matlab-for-data-science-and-machine-learning>

使用数据解决问题的机会比以往任何时候都多，随着不同行业采用这些方法，可用的数据一直在稳步增加，工具的数量也在增加。新数据科学家提出的一个典型问题与学习最佳编程语言有关，要么是为了更好地理解编码，要么是为了让他们的技能经得起未来考验。通常，这个问题围绕一些常见的疑点展开，比如 [R 和 Python](/r-vs-python-data-science) 。也有人问我 Java 和[的问题，我在其他地方提供了这个问题的答案](https://jrogel.com/programming-first-steps-java-or-python/)。实际上，可能没有单一的最佳工具可以使用，我一直主张在数据科学实践中使用工具箱方法。我想推荐其中一个工具:MATLAB。

## MATLAB 是什么？

MATLAB 于 20 世纪 80 年代中期首次出现在商业领域，它对专家工具箱的使用已经成为这种语言和生态系统的一个定义性特征。许多工程和科学课程已经接受 MATLAB 作为教学工具。因此，它被许多领域的科学家和工程师广泛使用，为数据分析、可视化等提供了出色的功能。

MATLAB 是一个高级技术计算环境，它在一个地方集成了计算、可视化和开发。它的交互式环境可作为开发、设计和使用应用程序的平台，其优势在于拥有各种数学函数，如统计、线性代数、傅立叶分析和优化算法等。

MATLAB 提供了有用的开发工具，可以改进代码维护和性能，以及与 Fortran、C/C++等其他编程语言的集成。NET，或者 Java。这就是我写 [Essential MATLAB 和 Octave](https://jrogel.com/essential-matlab-and-octave-video/) 的一些原因，这本书介绍了我自己的物理、数学和工程学生用 MATLAB 解决计算问题。

今天，MATLAB 是一种广泛使用的编程语言，许多行业都信任它，当用户试图将机器学习技术集成到他们的应用程序中时，它可以从中受益。

## 数据科学为什么要选择 MATLAB？

在数据科学和机器学习领域，MATLAB 可能不是首先想到的编程环境之一。部分原因可能是因为 Python、R 和 Scala 等语言吸引了人们的注意力；也可能是因为作为一种专有语言有时被视为一种障碍。然而，我认为，在许多行业和应用中，如航空航天、军事、医疗或金融，拥有一套受支持和外部验证的工具是一种优势，有多年的发展和商业成功作为后盾。

从技术角度来看，数据科学家和机器学习从业者需要一种语言，使他们能够操纵适合向量或矩阵运算的对象。一种编程语言，其名称实际上是“矩阵实验室”的缩写，它保证矩阵是表达所需计算操作的自然方式，其语法接近原始的线性代数符号。换句话说，对于 MATLAB 来说，基本的运算对象是一个矩阵元素。这样，例如整数可以被认为是 1×1 矩阵。这意味着为向量或矩阵构建的各种数学算法从一开始就内置到 MATLAB 中:叉积和点积、行列式、逆矩阵等。是天生可用的。反过来，这意味着机器学习技术需要的许多实现工作在 MATLAB 中变得更加容易。例如，考虑自然语言处理中语料库的表示:我们需要大型矩阵来表示文档。例如，矩阵的列可以表示文档中的单词，行可以是我们语料库中的句子、页面或文档。在机器视觉的情况下，将图像表示为矩阵并不罕见，MATLAB 提供了对这类对象的操作。

此外，MATLAB 中可用的工具箱数量使得创建结构化数据管道变得很容易，不需要我们担心兼容性问题，所有这些都是在相同的计算环境中完成的。一些工具箱已经成为语言的一部分很长时间了，例如[符号数学](https://uk.mathworks.com/products/symbolic.html)、[优化](https://uk.mathworks.com/products/optimization.html)和[曲线拟合](https://uk.mathworks.com/products/curvefitting.html?s_tid=srchtitle_curve%20fitting_1)工具箱，但是新的工具箱，例如[文本分析](https://uk.mathworks.com/products/text-analytics.html)、[统计和机器学习](https://uk.mathworks.com/products/statistics.html?s_tid=srchtitle_Statistics_1)以及[深度学习](https://uk.mathworks.com/solutions/deep-learning.html?s_tid=srchtitle_deep%20learning_1)工具箱正在将 MATLAB 重新放回游戏中。

MATLAB 强大的工程资质意味着有现成的机制可以直接从电路板、测量仪器和成像设备等硬件获取数据。这些功能，加上模拟工具，如 [SIMULINK，](https://www.mathworks.com/products/simulink.html)使得在一个有凝聚力的环境中使用机器学习技术变得不可抗拒。以防你从未听说过 SIMULINK，它是一个用于动态系统建模的交互式图形环境。它让用户创建虚拟原型，这些原型可以作为数字双胞胎，在飞行中尝试事物或分析假设情景。

让我们将 MATLAB 的使用映射到典型的数据科学工作流程，看看它如何为我们提供支持:

*   数据访问和探索——MATLAB 允许我们接收各种数据格式，包括文本文件、电子表格和 MATLAB 文件，还包括图像、音频、视频、XML 或拼花格式的数据。正如我们上面提到的，直接从硬件读取数据也是可能的。由于提供的交互式 IDE 和生态系统的数据可视化功能，可以实现数据探索。
*   数据预处理和清理-作为数据探索的自然下一步，MATLAB 使得使用实时编辑器清理异常值以及查找、填充或删除丢失的数据、删除趋势或规范化属性变得容易。MATLAB 还为用户提供特定领域的图像、视频和音频预处理工具。这意味着我们可以在训练 [MATLAB 的 Deep Network Designer 应用程序](https://uk.mathworks.com/help/deeplearning/ug/build-networks-with-deep-network-designer.html)之前对我们的数据应用合适的步骤，以构建复杂的网络架构或修改已训练的网络以进行迁移学习。
*   预测建模-工具箱可用于实现逻辑回归、分类树或支持向量机，以及专用的深度学习工具，用于在图像、时间序列和文本数据上实现卷积神经网络(ConvNets，CNN)和长短期记忆(LSTM)网络。

## 用 MATLAB 进行更多的机器学习

你的 MATLAB 之旅可能始于一台台式电脑，作为你工程或科学课程的一部分。今天，MATLAB 在专用的云资源中可用，例如在[Domino Enterprise MLOps Platform](https://www.dominodatalab.com/product/domino-enterprise-mlops-platform)中，您的模型可以在合适的 GPU 上训练。这种情况已经持续了一段时间，在[之前的博客文章](https://blog.dominodatalab.com/simple-parallelization)中，我们研究了 Domino 平台支持的一些更流行的语言中的代码并行化。

使用 MATLAB 实现数据科学和机器学习模型的可能性是无限的:从模型比较、用于特性和模型选择的 AutoML 以及超参数调整或扩展专用集群处理的能力，到用 C++等语言生成适合高性能计算的代码，以及与 SIMULINK 等仿真平台的集成。

借助深度学习工具箱等工具，MATLAB 不仅可以为深度神经网络的训练和部署(包括网络架构的设计)奠定坚实的基础，还可以为图像、视频和音频格式等数据的准备和标记提供支持。MATLAB 还使我们能够在同一环境中使用 PyTorch 或 TensorFlow 等框架。

## 摘要

数据科学家可以使用几个很棒的工具。尽管像 R 和 Python 这样的语言引人注目，但在 MATLAB 中训练的工程师、科学家、计量经济学家和金融工程师可以继续使用这个丰富而强大的生态系统提供的功能。随着 Mathworks 向其用户提供的持续支持和开发，以及与 Domino 的合作关系，MATLAB 将继续在数据科学和机器学习领域发展。

[![Video Demo  MATLAB with Domino Video Demo  This video will show how MATLAB and the Domino platform work together to help researchers and data scientists be more productive Watch the demo](img/35e8f5529200d691d24f055f32c7b0c9.png)](https://cta-redirect.hubspot.com/cta/redirect/6816846/fe952271-aa70-48ef-87c9-f1432053f854)