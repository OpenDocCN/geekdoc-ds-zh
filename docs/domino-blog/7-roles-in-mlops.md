# 企业 MLOps 中的 7 个关键角色和职责

> 原文：<https://www.dominodatalab.com/blog/7-roles-in-mlops>

任何 ML/AI 项目的主要挑战之一是将它从数据科学生命周期开发阶段的数据科学家手中转移到部署阶段的工程师手中。

数据科学家的参与在生命周期的哪个阶段结束？谁对可操作的模型负责？开发和部署之间的过渡应该持续多长时间？与数据工程师或 DevOps 工程师相比，数据科学家做什么？

这些问题的答案很少是一成不变的，即使是在小商店里。对于一个企业来说，当您添加额外的团队成员时，问题会变得更加复杂，每个成员都有不同的角色。

## 企业 MLOps 流程概述

[数据科学生命周期](https://www.dominodatalab.com/blog/how-enterprise-mlops-works-throughout-the-data-science-lifecycle)包含四个阶段，提供了整个流程的缩略图，并指出不同团队成员应该关注的地方。

*   **管理:**管理阶段着重于理解项目的目标和需求，并对工作进行优先级排序。数据科学家与业务、领导层、最终用户和数据专家合作，以确定项目范围、估算价值、估算成本、绘制解决方案蓝图、创建模拟交付成果、创建目标、就时间表以及验证和批准关口达成一致。他们为未来的数据科学家和审计人员记录这项工作。
*   **开发:**开发阶段是数据科学家基于各种不同的建模技术构建和评估各种模型的阶段。数据科学家创建一个模型，并用算法和数据对其进行测试。他们可能依赖数据分析师，或者得到他们的帮助。数据工程师通过提供干净的数据来提供帮助。基础架构工程师通过为数据科学家提供 IT 基础架构来提供帮助。当数据科学家需要帮助理解数据中存在的复杂关系时，就会召集数据专家。
*   **部署:**部署阶段是数据科学家基于各种不同的建模技术构建和评估各种模型的阶段。经过测试的模型在生产环境中从数据科学家过渡到开发人员和基础架构工程师。如果模型需要改写成另一种语言，软件开发人员就会接手。
*   **监控:**监控阶段是生命周期的操作阶段，在此阶段，组织确保模型交付预期的业务价值和性能。该模型通常由工程师进行[监控，如果出现问题，在需要时引入数据科学家。如果不像预测的那样，数据科学家会对模型进行故障诊断。如果数据管道出现问题，数据工程师会提供帮助。然后，在下一个开发阶段，双方都使用学到的信息和资源。](https://www.dominodatalab.com/blog/model-monitoring-best-practices-maintaining-data-science-at-scale)

然而，典型生命周期中的角色和职责很少如此清晰地描述。

## MLOps 团队中的 7 个关键角色

在较小的[数据科学](https://www.dominodatalab.com/blog/an-in-depth-view-of-data-science)运营中，一个人可能有多个角色，但是在一个企业中，每个团队成员都应该能够专注于他们的专业。有七个主要角色，尽管通常还涉及其他几个角色。例如，业务经理将参与构思和验证阶段，而法律团队中的某个人将在模型交付之前监督项目的合规性。

### 1.数据科学家

数据科学家通常被视为任何 MLOps 团队的核心人物，负责分析和处理数据。他们建立并测试 ML 模型，然后将模型发送到生产单位。在一些企业中，他们还负责监控模型一旦投入生产后的性能。

### 2.数据分析师

数据分析师与产品经理和业务部门合作，从用户数据中发掘洞察力。他们通常擅长不同类型的任务，如市场分析、财务分析或风险分析。许多人拥有与数据科学家相当的定量技能，而其他人可以归类为公民数据科学家，他们知道需要做什么，但缺乏编码技能和统计背景，无法像数据科学家那样单独工作。

### 3.数据工程师

数据工程师管理如何收集、处理和存储数据，以便从软件中可靠地导入和导出。他们可能拥有特定领域的专业知识，如 SQL 数据库、云平台以及特定的分布系统、数据结构或算法。它们通常对数据科学成果的可操作性至关重要。

### 4.DevOps 工程师

DevOps 工程师为数据科学家和其他角色提供对专业工具和基础架构(例如，存储、分布式计算、[GPU](https://www.dominodatalab.com/blog/machine-learning-gpu)等)的访问。)他们在数据科学生命周期中的需求。他们开发方法来平衡独特的数据科学要求与业务其余部分的要求，以提供与现有流程和 CI/CD 管道的集成。

### 5.ML 架构师

ML 架构师为要使用的 MLOps 开发策略、蓝图和流程，同时识别生命周期中的任何固有风险。他们识别和评估最好的工具，并召集工程师和开发人员团队一起工作。在整个项目生命周期中，他们监督 MLOps 流程。它们统一了数据科学家、数据工程师和软件开发人员的工作。

### 6.软件开发人员

软件开发人员与数据工程师和数据科学家一起工作，专注于 ML 模型和支持基础设施的生产化。他们根据 ML 架构师的蓝图开发解决方案，选择和构建必要的工具，并实施风险缓解策略。

### 7.领域专家/业务翻译

领域专家/业务翻译对业务领域和流程有深入的了解。他们帮助技术团队理解什么是可能的，以及如何将业务问题构建成 ML 问题。它们帮助业务团队理解模型提供的价值以及如何使用它们。在深入理解数据至关重要的任何阶段，它们都非常有用。

## MLOps 流程中可能的难点

由于流程中有如此多的阶段，企业运营中涉及如此多的人员，团队之间以及孤岛之间的沟通和协作会很快产生许多问题。例如，当团队不理解哪个模型使用了什么数据，数据来自哪里，以及它是如何被跟踪的时候，问题就出现了。这就产生了对数据科学家的依赖，他们需要提供所有必要的信息并管理从一个阶段到另一个阶段的过渡，这就成为了数据科学治理的一个问题。当 MLOps 过程中的变更和进展没有被适当地记录时，问题就出现了，这会给团队成员造成不准确的数据集和整体混乱。

一个关键点是确保模型从一个阶段有效地过渡到另一个阶段，而不会丢失前一个阶段的关键信息。这是通过企业 MLOps 平台实现的，该平台简化了复杂的数据科学流程。例如，数据科学家可以轻松获得他们需要的工具和计算，而不必依赖基础设施工程师。

## 建立标准以避免 MLOps 中的错误

[管理好 MLOps】最重要的一个方面是确保每个成员都清楚自己在团队中的角色。例如，让数据科学家而不是工程师负责监控项目的部署阶段，这在他们能够访问监控工具时相对容易，并且当模型出现问题时，可以由 MLOps 平台自动执行 pinged 操作。](https://www.dominodatalab.com/resources/a-guide-to-enterprise-mlops/)

每个专业都应该指定一名负责人，负责签署项目的每个阶段。例如，首席数据科学家将监督测试阶段完成的工作，并负责确定模型何时可以由业务部门进行验证。

## Domino 的企业 MLOps 平台

使用 [Domino 的企业 MLOps 平台](https://www.dominodatalab.com/product/domino-data-science-platform/)，团队成员能够在整个数据科学生命周期中轻松履行他们的职责。它缩短了关键过渡阶段的时间和精力，并且集成的工作流提供了一致性，无论谁在做这项工作。它还提供了对自动监控工具和自动生成的报告的访问，只需要很少的时间就可以检查模型的进度。因为所需的信息就在他们的手边，额外的协作不会占用他们手头其他任务的时间或精力。

[![The Complete Guide to  Enterprise MLOps Principles  Learn the key to becoming a model-driven business Read the Guide](img/9c077285252ec960ecf5eff9b9d6c5dc.png)](https://cta-redirect.hubspot.com/cta/redirect/6816846/4670a0fa-8832-4636-93d7-a77ea2f9611c)