# 机器学习模型验证的重要性及其工作原理

> 原文：<https://www.dominodatalab.com/blog/what-is-model-validation>

模型验证是开发机器学习或人工智能(ML/AI)的核心组件。虽然它独立于培训和部署，但它应该贯穿整个 [数据科学生命周期](https://blog.dominodatalab.com/how-enterprise-mlops-works-throughout-the-data-science-lifecycle) 。

## 什么是模型验证？

模型验证是一组过程和活动，旨在确保 ML/AI 模型按其应有的方式运行，包括其设计目标和对最终用户的效用。验证的一个重要部分是测试模型，但是验证并没有就此结束。

作为模型风险管理的一个组成部分，验证被设计成确保模型不会产生比它解决的更多的问题，并且它符合治理需求。除了测试之外，验证过程还包括检查模型的构造、用于创建模型的 [工具](https://blog.dominodatalab.com/8-modeling-tools-to-build-complex-algorithms) 及其使用的数据，以确保模型能够有效运行。

### 模型验证的作用

在对[模型进行训练](https://www.dominodatalab.com/blog/what-is-machine-learning-model-training)之后，需要一个过程来确保模型按照预期的方式运行，并解决其设计要解决的问题。这就是模型验证的目的。

以公正的方式进行验证是很重要的。因此，验证团队通常独立于训练模型的数据科学团队和将使用模型的人。通常，较小的组织会将模型验证外包给第三方。在一些受到高度监管的行业，这通常是一个熟悉法规的团队，以确保合规性。

### 模型验证与模型评估

[模型验证](https://medium.com/yogesh-khuranas-blogs/difference-between-model-validation-and-model-evaluation-1a931d908240) 与模型评估完全分开。评估是培训阶段的一部分。它是通过训练数据完成的:你选择一个[算法](https://www.dominodatalab.com/blog/7-machine-learning-algorithms)，在训练数据上训练它，然后将其性能与其他模型进行比较。

评估完成后，您可以继续验证获胜的模型。验证永远不会在测试数据上进行，而是在一个新的数据集上进行——测试数据。

## 如何验证模型

模型验证应该在模型测试之后、部署之前完成。在部署之后，可能还需要进行模型验证，尤其是在对它进行了任何更改的情况下。此外，作为监控过程的一部分，验证应在部署后定期进行，例如每年进行一次。

有 [三个主要的验证区域](https://axenehp.com/beginners-guide-model-validation/) :输入、计算和输出。

### 投入

输入部分包括模型计算中使用的假设和数据。数据应该与其来源一致，并根据行业基准以及团队在该模型或类似模型上的经验进行衡量。如果数据来自另一个模型，那么所使用的父模型的输出也应该被验证。

回溯测试通常是验证的一部分。例如，在[预测模型](https://www.dominodatalab.com/blog/introduction-to-predictive-modeling)中，可以向模型提供测试数据，然后可以将模型结果与来自测试数据的实际结果进行比较。

### 计算

检查模型逻辑很重要，以确保计算是合理和稳定的，并且输入被正确地合并。测试模型计算组件的两种常见方法是敏感性测试和动态验证。

敏感性测试包括量化模型输出的不确定性与其输入数据的不确定性之间的对应关系。动态验证包括在模型运行时更改属性，以测试其响应。

### 输出

模型的输出不仅包括计算结果，还包括数据的格式和表示。输出需要清晰，没有误导用户的风险。测试模型输出的一个好方法是将其与类似模型的输出进行比较，如果有类似模型的话。另外两种测试方法是历史回溯测试和版本控制。

## 最后的想法

如果没有合格的验证，您的组织将冒着潜在的灾难性风险，将有缺陷的模型部署到生产中。在商业中，这可能会使公司处于危险境地，甚至承担法律责任。在同一个组织中重复使用的一个有缺陷的模型可能会造成数百万甚至数十亿的损失。在卫生和科学等领域，一个未经验证的模型会危及人类生命。

除了模型本身，许多利用[机器学习模型](https://www.dominodatalab.com/blog/a-guide-to-machine-learning-models)的组织，包括毕马威和 Axene Health Partners，都将 [模型治理](https://blog.dominodatalab.com/the-role-of-model-governance-in-machine-learning-and-artificial-intelligence) 和文档监督作为验证流程的基本组成部分。当然，验证过程本身也必须完全记录在案。

认真对待数据科学计划的模型驱动型组织依赖于像 Domino 这样的企业级 MLOps 平台。Domino 不仅提供了数据科学团队需要的库和协作工具，还提供了大量的文档和治理工具，可以指导整个数据科学生命周期。

[![The Practical Guide to  Accelerating the Data Science Lifecycle  Lessons from the field on becoming a model-driven businesses.   Read the Guide](img/733c37e12c2c7c37295fb3198e3a226a.png)](https://cta-redirect.hubspot.com/cta/redirect/6816846/c77ca351-ae85-425a-9ee3-c264b3bc4a69)