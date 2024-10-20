# 文本和语音的自然语言处理指南

> 原文：<https://www.dominodatalab.com/blog/a-guide-to-natural-language-processsing>

虽然人类自诞生以来就一直在使用语言，但对语言的完全理解是一项终生的追求，即使对专家来说也常常达不到。让计算机技术理解语言、翻译甚至创作原创作品，代表了一系列仍在解决过程中的问题。

## 什么是自然语言处理？

自然语言处理(NLP)是不同学科的融合，从计算机科学和计算语言学到人工智能，共同用于分析、提取和理解来自人类语言的信息，包括文本和口语。它不仅仅是将单词作为信息块来处理。相反，NLP 可以识别语言中的层次结构，提取思想和辨别意义的细微差别。它包括理解句法、语义、词法和词汇。NLP 在数据科学领域有几个使用案例，例如:

*   摘要
*   语法纠正
*   翻译
*   实体识别
*   语音识别
*   关系抽取
*   话题分割
*   情感分析
*   文本挖掘
*   自动回答问题

## 自然语言生成

NLP 的子集， [自然语言生成(NLG)](https://www.ibm.com/cloud/learn/natural-language-processing) 是一种可以用英语或其他人类语言写出想法的语言技术类型。当给模型输入数据时，它可以生成人类语言的文本。通过文本到语音转换技术，它还可以产生人类语音。这是一个三阶段的过程:

*   **文字策划:**内容大致概述。
*   **句子规划:**内容放入句子和段落，考虑标点符号和文本流，包括代词和连词的使用。
*   **实现:**组装后的文本在输出前进行语法、拼写和标点的编辑。

通过开源模型(如 GPT 3)和框架(如 [PyTorch)的新发现和扩展，自然语言生成已经迅速扩展到商业组织中。](https://www.dominodatalab.com/data-science-dictionary/pytorch)

## 自然语言理解

NLP 的另一个子集是自然语言理解(NLU ),它确定文本或语音中句子的含义。虽然这对人类来说似乎很自然，但对于 [机器学习](//blog.dominodatalab.com/a-guide-to-machine-learning-models) ，它涉及一系列复杂的分析，包括:

*   **句法分析:**处理句子的语法结构，辨别意思。
*   **语义分析:**寻找可能是一个句子显性或隐性的意义。
*   **本体:**确定单词和短语之间的关系。

只有将这些分析放在一起，NLU 才能理解“食人鲨”这样的短语；依赖于先前句子的短语，如“我喜欢那样”；甚至是有多重含义的单个单词，比如自动反义词“疏忽”

## NLP 技术和工具

在你开始学习 NLP 之前，你需要访问标记数据(用于监督学习)、算法、代码和框架。你可以使用几种不同的技术，包括 [深度学习技术](https://blog.dominodatalab.com/deep-learning-illustrated-building-natural-language-processing-models) 取决于你的需求。一些最常见的自然语言处理技术包括:

*   **情感分析:**最广泛使用的自然语言处理技术，用于分析客户评论、社交媒体、调查和其他人们表达意见的文本内容。最基本的输出使用三分制(积极、中立和消极)，但如果需要，情感分析分数可以定制为更复杂的等级。它可以使用[监督学习技术](/blog/supervised-vs-unsupervised-learning)，包括朴素贝叶斯、随机森林或梯度推进，或者无监督学习。
*   **命名实体识别:**从文本中提取实体的基本技术。它可以识别人名、地点、日期、组织或标签。
*   **文本摘要:**主要用于摘要新闻和研究文章。提取模型通过提取文本来总结内容，而抽象模型生成自己的文本来总结输入文本。
*   **方面挖掘:**识别文本中的不同方面。当与情感分析一起使用时，它可以提取完整的信息和文本的意图。
*   **主题建模:**确定文本文档中涵盖的抽象主题。由于这使用了无监督学习，因此不需要带标签的数据集。流行的主题建模算法包括潜在狄利克雷分配、潜在语义分析和概率潜在语义分析。

现在比较流行的 NLP 框架有 NLTK，PyTorch， [spaCy](https://blog.dominodatalab.com/natural-language-in-python-using-spacy) ，TensorFlow，Stanford CoreNLP， [Spark NLP](https://www.dominodatalab.com/data-science-dictionary/apache-spark) 和百度 ERNIE。在生产环境中，每个 NLP 框架都有其优缺点，所以数据科学家通常不会只依赖一个框架。Kaggle 提供了一系列的 [NLP 教程](https://www.kaggle.com/c/word2vec-nlp-tutorial/overview/description) ，涵盖了基础知识，面向具有 Python 知识的初学者，以及使用 Google 的 Word2vec 进行深度学习。工具包括 50，000 个 IMDB 电影评论的标记数据集和所需的代码。

## 自然语言处理的应用

NLP 用于人们经常使用的各种应用程序。比如 Google Translate 就是用 TensorFlow 开发的。虽然它的早期版本经常被嘲笑，但它一直在通过谷歌的 [GNMT 神经翻译模型](https://ai.googleblog.com/2020/06/recent-advances-in-google-translate.html) 使用深度学习进行不断改进，以产生 100 多种语言的准确和听起来自然的翻译。

[【脸书】](https://research.fb.com/category/natural-language-processing-and-speech/) 的翻译服务也取得了显著的成功，用 [深度学习](https://blog.dominodatalab.com/deep-learning-introduction) 和自然语言处理，以及语言识别、文本规范化和词义消歧解决了复杂的问题。

今天自然语言处理的其他应用包括 [情感分析](https://blog.dominodatalab.com/deep-learning-illustrated-building-natural-language-processing-models) ，它允许应用程序检测情感和观点的细微差别，并识别讽刺或讽刺之类的事情。情感分析也用于文本分类，它自动处理非结构化文本，以确定它应该如何分类。例如，负面产品评论中的讽刺性评论可以被正确分类，而不是误解为正面评论。

### 带 Domino 数据实验室的 NLP

除了您可能在网上或社交媒体中使用的应用程序，还有许多商业应用程序依赖于 NLP。例如，在保险业中，NLP 模型可以分析报告和应用程序，以帮助确定公司是否应该接受所请求的风险。

丹麦第二大保险公司 Topdanmark， [使用 Domino](/blog/machine-learning-model-deployment) [数据科学平台](https://www.dominodatalab.com/resources/field-guide/data-science-platforms/)构建并部署了NLP 模型，使其 65%至 75%的案例实现自动化，客户等待时间从一周缩短至几秒钟。要开始探索 Domino 企业 MLOps 平台的优势，请注册一个[14 天免费试用版](https://www.dominodatalab.com/trial/?_ga=2.42355429.1300734614.1636060935-555552642.1632667115) 。

[![14-day free trial  Try The Domino Enterprise MLOps Platform Get started](img/4b2c6aa363d959674d8585491f0e18b8.png)](https://cta-redirect.hubspot.com/cta/redirect/6816846/28f05935-b374-4903-9806-2b4e86e1069d)