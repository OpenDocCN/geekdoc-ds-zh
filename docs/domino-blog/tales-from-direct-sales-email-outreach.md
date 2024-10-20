# 直销电子邮件推广的故事

> 原文：<https://www.dominodatalab.com/blog/tales-from-direct-sales-email-outreach>

直接电子邮件外联是我们的销售渠道之一:我们找到在感兴趣的公司工作的数据科学家，并向他们发送电子邮件。我们已经通过几次迭代和 A/B 测试改进了我们的模板，我们得到了超过 5%的响应率，这似乎是合理的。但是我们也得到一些令人困惑的回答。我想分享其中的一些，不是因为酸葡萄，而是因为我认为它们揭示了销售企业数据科学软件所面临的一些挑战。澄清一下，我不是在抱怨；我分享-有两个原因。第一，可能人家会有建议。其次，这可能有助于其他试图做企业销售的人。

以下是一些回答，代表了我们遇到的一些反复出现的主题:

## 我们自己建造它

到目前为止，我们听到的最常见的拒绝是“不，谢谢，我们自己构建我们的技术解决方案。”这里有一个例子:

```py

Hey Nick,
Sorry for the delay in response. 
I have chatted with a few of the developers here and they have decided to tailor build our platform in house. 
I am not entirely sure why, but this has been a common trend with [company]. Thank you for your time though.
```

或者另一个例子:

```py
Hi Nick,
Congrats! Seems like an awesome product, my team was definitely impressed. I don't think it's a fit for us as we prefer to homegrow and err on the side of needing control but would love to let you know if that changes.
```

或者简单地说:

```py
We have already built (and continue to build) the in-house technology we need.
```

我对[“购买 vs 构建”的想法可以写一篇长长的博文](/blog/reflections-on-buy-vs-build)，但这里有一些:

首先，开发人员更喜欢构建他们自己的解决方案一点也不奇怪。大多数开发人员之所以成为开发人员，正是因为他们喜欢构建东西，而且大多数公司都有一种文化，奖励那些做事情*的人*，而不是根据总体业务目标和约束找到最佳解决方案。因此，询问开发人员(甚至大多数工程经理)“你愿意构建一个解决方案还是购买一个”可能会被误导，因为对他们来说，权衡所有因素以做出最佳决策并不自然——相反，他们有某种直觉。当你有一堆锤子的时候，你会找到钉东西的理由。

第二，我认为像“我们通常更喜欢本土解决方案”这样的原则没有更多的限定是没有意义的。你总是会购买一些软件和构建一些软件。如果你认为你真的更喜欢构建自己的解决方案，问问自己是否构建了自己的编程语言、文字处理器、电子邮件客户端、操作系统、数据库服务器、IDE、防火墙软件等。

是否构建您自己的解决方案应取决于其功能对您的业务的重要性和差异化程度。

如果您要解决的功能和使用情形对您的业务来说是独一无二的，或者对您的竞争优势来说是至关重要的，以至于您必须完全控制它们，那么您应该构建一个自主开发的解决方案。(当然，这也取决于您自己是否有交付和维护解决方案的工程能力。)

相反，如果您需要的功能在很大程度上是商品化的，即许多其他公司需要做同样的事情，您应该考虑购买一个解决方案。

对此有两点看法:

1.  我们遇到的许多公司认为他们的用例比实际情况更独特。大多数人都想认为“我们做事的方式”是特别的——但通常它与其他公司的做法没有太大区别。在我们的具体环境中:当然，您公司的特定分析模型是高度差异化的，但这并不意味着您有独特的需求来支持构建这些模型的数据科学家。
2.  当考虑购买与构建的权衡时，许多公司大大低估了开发成本和总拥有成本。开发成本应该包括业务用户参与需求和验收测试的时间。它们还应该包括将这些开发人员从其他可能的项目中撤出的机会成本。总拥有成本估计应该包括用于持续支持和维护的开发人员资源。

## 低估生产功能的工作量

这里有一个与潜在客户的交流，他实际上用 Domino 实现了一个有效的概念验证。在大约一个小时的时间里，他完成了他的用例(将 Python 预测模型公开为 web 服务，供现有的 web 应用程序使用)。但是公司的决策者决定不使用我们的服务。以下是我们听到的反馈:

```py
As I understand it, the reasoning was pretty simple - they didn't want to be dependent on an external service that they would have to pay for when the exact functionality that they need is achievable using ZeroMQ and some shell scripts.
```

我认为这表明低估了真正强化生产应用程序所需的成本。将 ZeroMQ 和一些 shell 脚本拼凑在一起可能会产生一个原型，但需要真正的工程来确保解决方案满足公司对可用性和正常运行时间的要求；有一个部署新版本的好故事；并且当问题不可避免地发生时，公司具有支持和维护所需的内部知识。

他的理由中“必须为”部分似乎也是被误导的。为了将 Python 中的预测模型与现有的 Java web 应用程序集成在一起，使用 ZeroMQ 和 shell 脚本来实现定制的解决方案肯定不是免费的。除了前期工作之外，还需要持续利用资源进行支持和维护。

## 过程重于结果

一家非常大的咨询公司回信说，他们对了解 Domino 不感兴趣，因为:

```py
We are pretty far down the path evaluating several different technologies in this space and don’t want to incur the cost of spinning up another.
```

尽管我尽可能客观，但这对我来说似乎仍然不合逻辑。如果你要做一个重要的决定，并投入几个月(可能是几年)的时间来寻找解决问题的好方法，难道不值得花 30 分钟去了解另一个可能的解决方法吗？从期望值的角度来看，这似乎是显而易见的。

我不确定这是否是更深层次的症状，但我的直觉是这反映了大型组织的官僚作风。最终的业务目标和实际的所有活动之间存在脱节。有人没有看到更好的业务成果的机会，而是看到了对他正在运行的流程的干扰。

* * *

不幸的是，我没有很好的技巧来解除这些反应——事实上，解除它们可能是不可能的，因为大多数人在确信某事时不会改变主意。但是如果我们弄明白了，我会在以后的帖子里分享我们的见解！如果你有什么想法，请在评论中发表或者给我们发邮件。