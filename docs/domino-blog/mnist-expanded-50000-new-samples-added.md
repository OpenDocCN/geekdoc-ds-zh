# MNIST 扩展:新增 50，000 个样品

> 原文：<https://www.dominodatalab.com/blog/mnist-expanded-50000-new-samples-added>

*这篇文章提供了关于在 MNIST 数据集中重新发现 50，000 个样本的概要。*

## MNIST:过度拟合的潜在危险

最近， [Chhavi Yadav](https://twitter.com/chhaviyadav_/status/1133372241729732608?s=20) (NYU)和 [Leon Bottou](https://leon.bottou.org/start) (脸书 AI Research 和 NYU)在他们的论文《[悬案:丢失的 MNIST 数字](https://arxiv.org/pdf/1905.10498.pdf)》中指出，他们如何重建 [MNIST](https://www.dominodatalab.com/blog/benchmarking-nvidia-cuda-9-amazon-ec2-p3-instances-using-fashion-mnist) (修改后的美国国家标准与技术研究所)数据集，并向测试集添加 50，000 个样本，总共 60，000 个样本。20 多年来，许多数据科学家和研究人员一直使用包含 10，000 个样本的 MNIST 测试集来训练和测试模型。然而，行业意识到 MNIST(和其他流行数据集)的流行和使用也可能增加过度拟合的潜在危险。这导致研究人员通过重建数据集、测量准确性，然后分享他们的过程来寻找解决日益增长的过度拟合危险的方法。共享过程增加了可再现性的可能性，并在整个行业内建立现有的工作。

例如，在 Yadav 和 Bottou 的论文中，他们指出

*“数百份出版物报道了相同测试集[10，000 个样本]上越来越好的性能。他们是否对测试集进行了过度调整？我们能相信从这个数据集中得出的任何新结论吗？”....以及“5 万份样本是如何丢失的”*

## MNIST 重建步骤

为了解决这些关于过度拟合的潜在危险的问题，Yadav 和 Bottou 重建了 MNIST 数据集。当[论文深入到关于过程的细节](https://arxiv.org/pdf/1905.10498.pdf)时，[github 上的自述文件](https://github.com/facebookresearch/qmnist)提供了他们采取的步骤的摘要:

*“1。根据在介绍 MNIST 数据集的【独立】[论文中找到的信息，从第一重建算法开始。](https://leon.bottou.org/papers/bottou-cortes-94)*

*2。使用[匈牙利算法](https://en.wikipedia.org/wiki/Hungarian_algorithm)找到 MNIST 训练数字和我们重建的训练数字之间的最佳成对匹配。直观地检查最差的匹配，试图理解 MNIST 的作者可以做什么不同的事情来证明这些差异，同时不改变现有的接近匹配。*

*3。尝试重建算法的新变体，将它们的输出与 MNIST 训练集中它们的最佳对应物进行匹配，并重复该过程。"*

[摘录来源](https://github.com/facebookresearch/qmnist)

这种过程共享有助于支持研究的可重复性，并有助于推动行业向前发展。

## 找到的 MNIST 数字

通过这项工作，Yadav 和 Bottou 重新发现了丢失的 50，000 个样本

*“本着与【Recht et al .】、 [2018](https://arxiv.org/abs/1806.00451) 、 [2019](https://arxiv.org/abs/1902.10811) 相同的精神，5 万个丢失的 MNIST 测试数字的重新发现为量化官方 MNIST 测试集在四分之一世纪的实验研究中的退化提供了一个机会。”*

他们也能够

*“跟踪每个 MNIST 数字到其 [NIST](https://www.nist.gov/srd/nist-special-database-19) 源图像和相关元数据”...“这些新的测试样本使我们能够精确地调查标准测试集上报告的结果如何受到长时间重复实验的影响。我们的结果证实了 Recht 等人[ [2018](https://arxiv.org/abs/1806.00451) ， [2019](https://arxiv.org/abs/1902.10811) ]观察到的趋势，尽管是在不同的数据集上，并且是在更加可控的设置中。所有这些结果本质上表明“测试集腐烂”问题是存在的，但远没有人们担心的那么严重。虽然重复使用相同测试样本的做法会影响绝对性能数字，但它也提供了配对优势，有助于长期的模型选择。”*

然而，潜在影响仍在继续。Yadav 和 Battou 的工作也在 Yann LeCun 的 twitter 上分享。

> MNIST 重生了，恢复了，扩张了。
> 现在有了额外的 50，000 个训练样本。
> 
> 如果你多次使用原始的 MNIST 测试集，你的模型很可能会超出测试集。是时候用这些额外的样本来测试它们了。[https://t.co/l7QA1u94jF](https://t.co/l7QA1u94jF)
> 
> — Yann LeCun (@ylecun) [May 29, 2019](https://twitter.com/ylecun/status/1133735660563697665?ref_src=twsrc%5Etfw)