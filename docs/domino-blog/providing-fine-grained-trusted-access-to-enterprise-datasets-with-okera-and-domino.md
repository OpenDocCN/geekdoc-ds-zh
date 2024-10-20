# 使用 Okera 和 Domino 提供对企业数据集的细粒度可信访问

> 原文：<https://www.dominodatalab.com/blog/providing-fine-grained-trusted-access-to-enterprise-datasets-with-okera-and-domino>

Domino 和 Okera——让数据科学家能够在可再生和即时调配的计算环境中访问可信数据集。

在过去几年中，我们看到了两种趋势的加速发展——组织存储和利用的数据量不断增加，随之而来的是对数据科学家的需求，他们需要帮助理解这些数据，以做出关键的业务决策。数据量和需要访问数据的用户数量的激增带来了新的挑战，其中最主要的挑战是如何提供对大规模数据的安全访问，以及如何让数据科学家能够一致、可重复且方便地访问他们所需的计算工具。

这些模式出现在多个行业和用例中。例如，在制药业，有大量的数据是为临床试验和新药物及治疗方法的商业生产而产生的，而这种情况在新冠肺炎出现后更是加速了。这些数据支持组织内的各种用例，从帮助生产分析师了解生产进展情况，到允许研究科学家查看不同试验和人群横截面的一组治疗结果。

Domino 数据实验室是世界领先的数据科学平台，允许数据科学家轻松访问可重复且易于配置的计算环境。他们可以处理数据，而不用担心设置 Apache Spark 集群或获得正确版本的库。他们可以轻松地与其他用户共享结果，并创建重复作业，以便随着时间的推移产生新的结果。

在当今日益注重隐私的环境中，越来越多类型的数据被视为敏感数据。这些数据集必须按照行业特定的法规进行保护，如 HIPAA，或一系列新兴的消费者数据隐私法规，包括 GDPR、CCPA 和不同司法管辖区的其他法规。这可能成为数据消费者的障碍；尽管 Domino Data Lab 使访问计算资源变得很容易，但是访问他们需要的所有数据却是一个真正的挑战。

传统上，解决这个问题的方法是完全拒绝访问这些数据(这种情况并不少见)，或者通过省略特定用户不允许查看的数据(例如 PII、PHI 等)，为每个可能的用例创建和维护许多数据集的多个副本。这种创建数据副本的过程不仅需要花费大量时间(通常需要几个月)并增加存储成本(当谈到数 Pb 的数据时，存储成本会迅速增加)，而且还会成为一场管理噩梦。数据经理需要跟踪所有这些副本以及创建它们的目的，并记住它们需要与新数据保持同步，甚至更糟的是，由于新类型的数据被认为是敏感的，未来可能会进行修订和转换。

安全数据访问和[数据治理](/choosing-a-data-governance-framework)的领先提供商 Okera ，允许您使用基于属性的访问策略定义细粒度的数据访问控制。结合 Domino Data Labs 和 Okera 的强大功能，您的数据科学家只能访问允许的列、行和单元格，轻松删除或编辑与训练模型无关的敏感数据，如 PII 和 PHI。此外，Okera 连接到公司现有的技术和业务元数据目录(如 Collibra)，使数据科学家能够轻松发现、访问和利用新的、经批准的信息源。

对于法规遵从性团队来说，Okera 和 Domino 数据实验室的结合是极其强大的。它不仅允许法规遵从性管理可以访问的信息，还允许审核和了解数据的实际访问方式——何时、由谁、通过什么工具、查看了多少数据等。这可以识别数据泄露，并找出应该进一步减少数据访问的地方，例如通过删除对不常用数据的访问来降低暴露风险。

那么这看起来像什么？考虑一个例子，一个数据科学家想要从亚马逊 S3 将一个 CSV 文件加载到 pandas 数据帧中进行进一步的分析，例如为下游的 ML 过程建立一个模型。在 Domino Data Lab 中，用户将使用他们有权访问的环境之一，并拥有一些类似于下面这样的代码:

```py
import boto3

import io
s3 = boto3.client('s3')

obj = s3.get_object(Bucket='clinical-trials', Key='drug-xyz/trial-july2020/data.csv')

df = pd.read_csv(io.BytesIO(obj['Body'].read()))
```

上述片段中嵌入的一个关键细节是数据科学家如何获得访问文件的许可的问题。这可以通过 IAM 权限来实现，既可以将用户凭证存储在 Domino 内部的安全环境变量中，也可以使用 [keycloak 功能](https://admin.dominodatalab.com/en/latest/keycloak.html#aws-credential-propagation)在 Domino 和 AWS 之间进行凭证传播。

最后，如果不允许数据科学家查看 CSV 文件中的某些列、行或单元格，就没有办法授予对该文件的访问权限。

当 Domino Data Lab 与 Okera 集成时，相同的代码看起来就像这样:

```py
import os

from okera.integration import domino

ctx = domino.context()

with ctx.connect(host=os.environ['OKERA_HOST'], port=int(os.environ['OKERA_PORT'])) as conn:

    df = conn.scan_as_pandas('drug_xyz.trial_july2020')
```

Domino Data Lab 中当前用户的身份被自动透明地传播到 Okera，并应用了所有必需的细粒度访问控制策略。这意味着，如果执行用户仅被允许查看某些行(例如，来自美国参与者的试验结果，以遵守数据位置法规)或查看某些列，但不公开 PII(例如，通过不公开参与者的姓名，但仍然能够有意义地创建聚合)，这将反映在返回的查询结果中，而不会向数据科学家公开底层敏感数据。最后，这种数据访问也要经过审计，审计日志可以作为数据集用于查询和检查。

除了能够在维护细粒度访问控制策略的同时安全访问数据的好处之外，数据科学家现在更容易找到他们需要访问的数据。以前，这涉及到筛选对象存储，如亚马逊 S3 或 Azure ADLS，但通过 Okera 和 Domino Data Lab 的结合，数据科学家可以轻松地检查和搜索 Okera 的元数据注册表，以找到他们有权访问的数据，这些数据已经过主题专家的验证、鉴定和记录，预览这些数据，并获得如何在 Domino Data Lab 环境中访问这些数据的简单说明。

随着您的组织在数据方面的投资和数据科学家工作效率的提高，他们拥有正确的工具和访问正确数据的权限变得至关重要。有了 Okera 和 Domino Data Lab 的结合，整体就不仅仅是各部分的总和了。如果您已经在利用 Domino Data Lab，添加 Okera 可以允许您解锁数据进行分析，这在以前是由于隐私和安全问题而被禁止的。如果您已经在使用 Okera，那么添加 Domino Data Lab 可以提高您的数据科学家的工作效率，让他们能够轻松地访问可重复且易于配置的计算环境。

有关 Okera 及其与 Domino 的合作关系的更多信息，请参见 [Okera 的博客文章](https://www.okera.com/blogs/okera-domino-speeding-business-outcomes-with-data-science/)或[关于整合 Okera 与 Domino 的文档](https://docs.dominodatalab.com/en/latest/user_guide/b83660/connect-to-okera-from-domino/)。