# 在混合/云计算的未来，平衡医疗保健人工智能创新与治理

> 原文：<https://www.dominodatalab.com/blog/balancing-healthcare-ai-innovation-with-governance>

合作伙伴营销主管[大卫·舒尔曼](/blog/author/david-schulman)和全球健康和生命科学主管[卡洛琳·法尔斯](//dominodatalab.com/blog/author/caroline-phares)

从早期疾病检测到智能临床试验设计，再到个性化医疗，人工智能/人工智能在医疗保健和生命科学领域的前景非常广阔，“超过 57%的医疗保健和生命科学组织将人工智能/人工智能评为重要或非常重要，”根据 Ventana Research 在其最近的白皮书 *[中对首席数据和分析高管的顶级人工智能考虑](/resources/top-5-ai-considerations-for-data-and-analytics-executives)。*

*   [让桑加速了深度学习模型](https://www.dominodatalab.com/customers/janssen) 的训练，在某些情况下速度快了 10 倍，通过全切片图像分析更快更准确地诊断和表征癌细胞。
*   [礼来公司将电子病历、](https://www.dominodatalab.com/resources/data-science-innovation-across-healthcare) *保险赔付、真实世界证据(RWE)和其他来源的多模态数据相结合，将人群水平的模式转化为下一个最佳的患者治疗措施(*注:36:56 转化为面板记录)。
*   [Evidation 使用来自应用程序和可穿戴技术(如智能手机、活动追踪器和智能手表)的患者生成的健康数据(PGHD)持续测量个人健康](https://www.dominodatalab.com/customers/evidation) 。
*   [NVIDIA 的医疗人工智能全球负责人 Mona Flores](https://www.dominodatalab.com/resources/data-science-innovators-playbook?) 描述了使用带有隐私保护数据的联合学习来识别疾病并预测需要哪些治疗。

但是机器学习模型需要数据，而平衡数据科学家进行实验和创新所需的开放性和灵活性与治理和数据隐私法规，是管理全球医疗保健和生命科学企业软件和基础设施的数据和分析高管面临的一项重大挑战。

## 管理分布式多模式数据的激增

医疗保健和生命科学组织处理大量不同的数据，包括电子病历、账单、患者就诊报告、保险索赔、放射图像、PGHD 等等。根据 Ventana Research 的数据，这些数据通常分布在云区域、提供商和内部数据源之间——32%的组织报告使用 20 个以上的数据源，而 58%的组织自我报告使用“大数据”, Pb 大小的数据库变得越来越常见。例如，让桑的组织病理学图像的大小可能在 2gb 到 5gb 之间，而更大的临床试验可能超过 10 万张图像。

虽然托管云数据库前景光明，但入口和出口成本会极大地阻碍数据科学的发展。Ventana Research 指出，提取和分析 1pb 的数据可能需要高达 50，000 美元。数据重力性能考虑因素(即，通过协同定位数据和计算来减少模型延迟)以及数据驻留/主权法规进一步使数据收集和处理复杂化，通常会将数据集锁定在单个地理位置。这与 HIPAA 和 GDPR 等法规相结合，凸显了混合云和多云配置对于确保适当的数据管理和地理围栏的重要性。Ventana 研究强调指出:

> “到 2026 年，几乎所有的跨国组织都将投资本地数据处理基础设施和服务，以减轻与数据传输相关的风险。”

## 使用混合/多云多点操作管理 AI/ML

虽然数据是分布式的，但数据科学是一项团队运动，需要快速的实验、轻松的数据访问和轻松的可复制性来实现真正的创新。AI/ML/Analytics“卓越中心”(COE)战略正变得越来越普遍，通过协作来混合知识，同时提供基础设施和治理。Ventana Research 指出，80%的组织认识到了治理 AI/ML 的重要性。 [约翰逊&约翰逊](https://blogs.nvidia.com/blog/2021/09/02/johnson-and-johnson-domino-data-science-mlops/) 有一个内部数据科学委员会，帮助“将公司的数据科学社区整合到业务工作流中，从而更快地应用机器学习模型、反馈和影响。”此外，这些 Coe 通过确保数据科学家能够访问所需的数据、工具和基础架构来促进创新，以便他们能够专注于构建突破性的模型，而不是开发运维。

许多医疗保健和生命科学机器学习用例，如计算机视觉(如让桑的深度学习用例)，需要专门构建的人工智能基础设施，包括英伟达等公司的 GPU。考虑到数据传输成本、安全性、法规和性能，与跨云或地理位置传输或复制数据集相比，对数据进行计算/处理通常更有意义。

理论上，云基础设施上的 GPU 解决了这个问题——直到考虑到成本和性能。 [Protocol 最近报道了](https://www.protocol.com/enterprise/ai-machine-learning-cloud-data) 公司将 ML 数据和模型转移回内部、本地设置的趋势，“花更少的钱，获得更好的性能”数据科学工作负载本质上是可变的，训练模型所需的大规模爆发可能难以预测。将其中一些工作负载转移回内部基础架构可以显著降低成本，同时提高性能。

对于 ML CoEs 而言，在混合/多云和本地环境中，从单一控制台管理 AI/ML 变得更具挑战性，尤其是在分布式数据以及全球公司中存在的本地和云基础架构的混合环境中。从数据管理到分析再到数据科学，数据和分析高管在庞大的数据和分析技术体系中难以做出决策。

## 医疗保健和生命科学组织的数据科学平台考虑事项

随着我们最近的 [Nexus 混合云数据科学平台](https://www.dominodatalab.com/blog/why-hybrid-cloud-is-the-next-frontier-for-scaling-enterprise-data-science) 的发布，Domino 数据实验室处于人工智能工作负载的混合/多云支持的前沿。真正的混合数据科学平台使数据科学家能够在公司运营的每个环境中以安全、受监管的方式访问数据、计算资源和代码。我们与 NVIDIA 的深度 [合作以及对](https://www.dominodatalab.com/partners/nvidia) [更广泛的数据和分析生态系统](https://www.dominodatalab.com/partners) 的支持，为数据和分析高管提供了培养 AI/ML 创新的信心，同时提供了企业范围治理所需的灵活性。

Ventana Research 强调了开放和灵活的数据科学平台的重要性，“面对不断发展的混合战略、不断变化的数据科学创新，让您的数据科学实践经得起未来考验，并从专门构建的人工智能基础设施或云中实现价值最大化。”要了解更多信息，请查看他们的白皮书。

[![Whitepaper  Top 5 AI Considerations for Data & Analytics Executives  Discover the next generation of AI infrastructure for the hybid cloud future Read the whitepaper](img/41bc19f79d99c5f3f05c31ceb8a3ed38.png)](https://cta-redirect.hubspot.com/cta/redirect/6816846/48b0b460-9689-4aba-a19e-245c132f539f)