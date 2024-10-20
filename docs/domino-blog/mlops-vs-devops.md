# MLOps vs DevOps:有什么区别？

> 原文：<https://www.dominodatalab.com/blog/mlops-vs-devops>

机器学习操作(machine Learning Operations)([MLOps](https://www.dominodatalab.com/resources/a-guide-to-enterprise-mlops/))是一个在过去十年中变得流行的术语，通常用于描述一系列旨在可靠、高效地在生产中部署和维护机器学习(ML)模型的实践。

然而，这个定义一次又一次地被证明是过于狭隘的，因为它忽略了数据科学生命周期的关键方面，包括 ML 模型的管理和开发。

面对当今的业务挑战，模型驱动的业务应该以采用企业 MLOps 为目标。我们将企业 MLOps 定义为“大规模端到端数据科学生命周期的流程系统。它为数据科学家、工程师和其他 It 专业人员提供了一个场所，使他们能够在机器学习(ML)模型的开发、部署、监控和持续管理方面与支持技术高效地合作。它允许组织在整个组织内快速高效地扩展数据科学和 MLOps 实践，而不会牺牲安全性或质量。”

但是，值得注意的是，只有在学习了另一个学科:开发运营(DevOps)的成功和最佳实践之后，这种对 MLOps 的整体看法才是可能的。

顾名思义，DevOps 是软件开发(Dev)和运营(Ops)的结合，旨在加快应用和服务的开发周期。DevOps 引入的最重要的变化是摆脱孤岛，促进软件开发和 IT 团队之间的协作，以及实施最佳实践和工具，以实现利益相关者之间的流程自动化和集成的目标。

虽然 DevOps 和 MLOps 的目标相似，但它们并不相同。本文将探讨 MLOps 和 DevOps 之间的差异。

## 为什么是 MLOps

ML 模型是由数据科学家构建的。然而，模型开发只是组成企业 ML 生产工作流程的一小部分。

为了操作 ML 模型并使模型为生产做好准备，数据科学家需要与其他专业人员密切合作  [，包括数据工程师、ML 工程师以及软件开发人员。然而，在不同的职能部门之间建立有效的沟通和协作是非常具有挑战性的。每个角色都有独特的职责。例如，数据科学家开发 ML 模型，ML 工程师部署这些模型。](https://www.dominodatalab.com/blog/7-roles-in-mlops)

任务的性质也不同。例如，处理数据比软件开发更具有研究性。有效的协作是具有挑战性的，沟通不畅会导致最终产品交付的重大延误。

因此，MLOps 可以解释为一系列技术和最佳实践，有助于大规模管理、开发、部署和监控数据科学模型。Enterprise MLOps 采用这些相同的原则，并将其应用于大规模生产环境，这些环境的模型更依赖于安全、治理和合规性系统。

## MLOps 和 DevOps 之间的区别

MLOps 和 DevOps 之间有几个主要区别，包括:

### 数据科学模型是概率性的，而不是确定性的

ML 模型产生概率预测。这意味着它们的结果可能因输入数据和底层算法及架构而异。如果用于训练模型的环境发生变化，模型性能会迅速降低。例如，在新冠肺炎疫情期间，当输入数据的性质发生剧烈变化时，大多数 ML 模型的性能都会受到影响。

软件系统是确定性的，并且将以相同的方式执行，直到底层代码被改变，例如在升级期间。

### 数据科学的变化率更高

DevOps 允许您通过一套标准的实践和过程来开发和管理复杂的软件产品。数据科学产品和 ML 产品也是软件。然而，它们很难管理，因为除了代码之外，它们还涉及数据和模型。

您可以将 ML 模型视为一种数学表达式或算法，经过训练可以识别所提供数据中的特定模式，并根据这些模式生成预测。使用 ML 模型的挑战在于其开发所必需的活动部分，例如数据集、代码、模型、超参数、管道、配置文件、部署设置和模型性能指标等等。这就是为什么 MLOps 是必要的，模型不仅仅是软件，而且需要不同的策略来进行开发、监控和大规模部署。

### 不同的责任

MLOps 源于 DevOps。但是，每个人的责任不同。

DevOps 的职责包括以下内容:

*   **持续集成，持续部署(CI/CD):** CI/CD 是指将软件开发的不同阶段自动化的过程，包括构建、测试和部署。它允许您的团队不断地修复错误、实现新特性，并将代码交付到产品中，这样您就可以快速高效地交付软件。
*   **正常运行时间:** 这是指您的服务器保持可用的总时间，是一个通用的基础设施可靠性指标。DevOps 专注于维持高正常运行时间，以确保软件的高可用性。它帮助企业方便地运营关键业务，而不会面临重大服务中断。
*   **监控/记录:** 监控让您可以分析应用程序在整个软件生命周期中的性能。它还允许您有效地分析基础设施的稳定性，并涉及几个过程，包括日志记录、警报和跟踪。日志记录提供了关于关键组件的数据。您的团队可以利用这些信息来改进软件。相比之下，预警有助于您提前发现问题。它包含帮助您快速解决问题的调试信息。跟踪提供性能和行为信息洞察。您可以使用这些信息来增强生产应用的稳定性和可伸缩性。

相比之下，MLOps 的职责包括以下内容:

*   **管理阶段。** 这个阶段围绕着识别要解决的问题，定义预期要获得的结果，建立项目需求，以及分配角色和工作优先级。一旦模型被部署到生产中，在监控阶段获得的结果将被审查，以评估所获得的成功程度。此审查的成功和错误可用于改进当前项目和未来项目。换句话说，这一阶段包括以下方面:

> *   **Project management:** Access control of resources such as information, snapshot of central repository, status tracking, and more.
> *   **Enterprise-wide knowledge management:** It is very important to improve efficiency by reusing key knowledge. [Domino Knowledge Center](https://www.dominodatalab.com/product/system-of-record) is a good example of this type of central knowledge base.
> *   **Technical resource governance:** includes cost tracking capability, role-based information access rights, tools and computing resource management.

*   **发育阶段。** 在此阶段，数据科学家基于各种不同的建模技术构建和评估各种模型。这意味着在开发阶段，数据科学家可以使用诸如  [Domino 的数据科学工作台](https://www.dominodatalab.com/product/integrated-model-factory)之类的环境来试验和执行 R & D，在那里他们可以轻松地访问数据、计算资源、Jupyter 或 RStudio 之类的工具、库、代码版本控制、作业调度等等。
*   **部署阶段。** 一旦一个模型被证明是成功的，它就可以被部署到生产中，并通过帮助业务流程决策来交付价值。为此，这一阶段涉及的方面包括:

> *   Flexible hosting of allows data scientists to freely deploy licensed model APIs, web applications and other data products to various hosting infrastructures quickly and efficiently.
> *   Packaging the model in a container is convenient for the external system to consume on a large scale.
> *   A data pipeline that supports the ingestion, arrangement and management of the data.

*   **监视阶段。** 这是负责测量模型性能的阶段，从而检查它是否如预期的那样运行。换句话说，它评估模型是否向业务交付了预期的价值。理想情况下，此阶段为 MLOps 团队提供的功能包括:

> *   Verify that the pipeline is allowed to pass the CI/CD principle.
> *   Testing and deploying scoring pipeline. These tools are helpful for A/B testing of model versions in production and tracking the results to inform business decisions.
> *   Model library, which collects all model APIs and other assets in order to know their health, usage, history and so on. It can be evaluated. Tools like Domino's integrated model monitoring can comprehensively monitor all models of an organization immediately after deployment, and can also proactively detect any problems and alert you.
> *   A mechanism that uses the history and context of the original model to help retrain and rebuild the model.

#### MLOps 和 DevOps:职责重叠

虽然 MLOps 和 DevOps 有所不同，但它们也有一些重叠的职责:

*   这两个学科广泛地使用版本控制系统来跟踪对每个工件的变更。
*   DevOps 和 MLOps 大量使用监控来检测问题、测量性能，并对它们各自的工件执行优化。
*   两者都使用 CI/CD 工具，尽管在 DevOps 中，这些工具旨在自动化应用程序的创建。另一方面，在 MLOps 中，它们被用来建立和训练 ML 模型。
*   DevOps 和 MLOps 鼓励持续改进各自的流程。同样，DevOps 和 MLOps 促进不同学科之间的协作和集成，以实现共同的目标。
*   这两个规程都需要所有涉众的深度承诺，以在整个组织中实施它们的原则和最佳实践

## 包扎

在本文中，您了解了 MLOps 团队的职责与 DevOps 团队的职责有何不同，以及它们之间的重叠之处。数据科学团队需要高水平的资源和灵活的基础架构，而 MLOps 团队提供并维护这些资源。

MLOps 是成功的关键。它允许你通过克服约束来实现你的目标，包括有限的预算和资源。此外，MLOps 还能帮助您提高敏捷性和速度，从而更好地适应模型。

[![The Complete Guide to  Enterprise MLOps Principles  Learn the key to becoming a model-driven business Read the Guide](img/9c077285252ec960ecf5eff9b9d6c5dc.png)](https://cta-redirect.hubspot.com/cta/redirect/6816846/4670a0fa-8832-4636-93d7-a77ea2f9611c)