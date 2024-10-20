# 评估 Ray:分布式 Python 的大规模可伸缩性

> 原文：<https://www.dominodatalab.com/blog/evaluating-ray-distributed-python-for-massive-scalability>

*[迪安](mailto:dean@anyscale.io) [Wampler](https://twitter.com/deanwampler) 提供了 Ray 的概要，这是一个用于将 Python 系统从单机扩展到大型集群的开源系统。如果您对其他见解感兴趣，请注册参加[即将举行的射线峰会](https://events.linuxfoundation.org/ray-summit/)。*

## 介绍

这篇文章是为做技术决策的人写的，我指的是数据科学团队领导、架构师、开发团队领导，甚至是参与组织中使用的技术的战略决策的经理。如果你的团队已经开始使用 [Ray](https://ray.io) 并且你想知道它是什么，这篇文章就是为你准备的。如果您想知道 Ray 是否应该成为您基于 Python 的应用程序(尤其是 ML 和 AI)的技术策略的一部分，那么这篇文章就是为您准备的。如果你想对 Ray 有更深入的技术介绍，可以在 [Ray 项目博客](https://medium.com/distributed-computing-with-ray/ray-for-the-curious-fa0e019e17d3)上看到这篇文章。

## 雷是什么？

Ray 是一个 T2 开源系统，用于将 Python 应用从单机扩展到大型集群。它的设计是由下一代 ML 和 AI 系统的独特挑战驱动的，但它的特性使 Ray 成为所有需要跨集群伸缩的基于 Python 的应用程序的绝佳选择，特别是如果它们具有分布式状态。Ray 还提供了一个侵入性最小且直观的 API，因此您无需在分布式系统编程方面花费太多精力和专业知识就能获得这些好处。

开发人员在他们的代码中指出哪些部分应该分布在集群中并异步运行，然后 Ray 为您处理分布。如果在本地运行，应用程序可以使用机器中的所有内核(您也可以指定一个限制)。当一台机器不够用时，可以直接在一个机器集群上运行 Ray，让应用程序利用这个集群。此时唯一需要更改的代码是在应用程序中初始化 Ray 时传递的选项。

使用 Ray 的 ML 库，如用于强化学习(RL)的 [RLlib](https://docs.ray.io/en/latest/rllib/index.html) 、用于超参数调整的 [Tune](https://ray.readthedocs.io/en/latest/tune.html) 和用于模型服务(实验性)的 Serve，都是用 Ray 在内部实现的，因为它具有可伸缩、分布式计算和状态管理的优势，同时提供了特定于领域的 API 来实现它们的目的。

## Ray 的动机:训练强化学习(RL)模型

为了理解 Ray 的动机，考虑训练强化学习(RL)模型的例子。RL 是一种机器学习类型，最近被用于[击败世界上最好的围棋选手](https://deepmind.com/research/case-studies/alphago-the-story-so-far)，并实现雅达利和类似游戏的专家游戏。

可伸缩 RL 需要许多 Ray 设计提供的功能:

*   **高度并行和高效地执行*任务*** (数百万或更多)——当训练模型时，我们一遍又一遍地重复相同的计算，以找到最佳的模型*方法*(“超参数”)，一旦选择了最佳结构，就找到工作最佳的模型*参数*。当任务依赖于其他任务的结果时，我们还需要正确的任务排序。
*   **自动容错** -对于所有这些任务，它们中的一部分可能会因为各种原因而失败，因此我们需要一个支持任务监控和故障恢复的系统。
*   **多样的计算模式**——模型训练涉及大量的计算数学。特别是，大多数 RL 模型训练还需要模拟器的高效执行——例如，我们想要击败的游戏引擎或代表真实世界活动的模型，如自动驾驶。使用的计算模式(算法、内存访问模式等。)是更典型的通用计算系统，它与数据系统中常见的计算模式有很大不同，在数据系统中，高吞吐量的转换和记录的聚合是标准。另一个区别是这些计算的动态性质。想想一个游戏玩家(或模拟器)是如何适应游戏的发展状态，改进策略，尝试新战术等。这些不同的需求出现在各种新的基于 ML 的系统中，如机器人、自主车辆、计算机视觉系统、自动对话系统等。
*   **分布式状态管理** -使用 RL，需要在训练迭代之间跟踪当前模型参数和模拟器状态。因为任务是分布式的，所以这种状态变成分布式的。正确的状态管理还需要有状态操作的正确排序..

当然，其他 ML/AI 系统需要部分或全部这些能力。大规模运行的一般 Python 应用程序也是如此。

## 雷的要旨

像 RLlib、Tune 和 Serve 这样的 Ray 库使用 Ray，但大多数情况下对用户隐藏它。然而，使用 Ray API 本身很简单。假设您有一个“昂贵”的函数要对数据记录重复运行。如果它是无状态的，这意味着它在调用之间不维护任何状态，并且您想要并行地调用它，那么您需要做的就是通过添加如下的`​@ray.remote​`注释将该函数转换成一个 Ray 任务:

```py
@ray.remote

def slow(record):

    new_record = expensive_process(record)

    return new_record

```

然后初始化 Ray 并在数据集上调用它，如下所示:

```py
ray.init() # Arguments can specify the cluster location, etc.

futures = [slow.remote(r) for r in records]

```

注意我们是如何使用`​slow.remote`调用函数`slow`的。每一个呼叫都会立即返回，并带来一个未来。我们收集了它们。如果我们在集群中运行，Ray 管理可用的资源，并将这个任务放在一个节点上，该节点具有运行该功能所必需的资源。

我们现在可以要求 Ray 在使用完`​ray.wait`后返回每个结果。这里有一个惯用的方法:

```py
while len(futures) > 0:

     finished, rest = ray.wait(futures)

     # Do something with “finished”, which has 1 value:

     value = ray.get(finished[0]) # Get the value from the future

     print(value)

     futures = rest

```

如前所述，我们将等待一个 slow 调用完成，此时`ray.wait`将返回两个列表。第一个只有一个条目，即已完成的慢速调用的*未来*的 id。我们传入的期货列表的其余部分将在第二个列表中— `​rest`。我们调用`​ray.get`来检索已完成的未来的价值。*(注意:这是一个阻塞调用，但它会立即返回，因为我们已经知道它已经完成了。)*我们通过将我们的列表重置为剩余的列表来结束循环，然后重复直到所有的远程调用都已完成并且结果已被处理。

您还可以向`ray.wait`传递参数，一次返回多个参数，并设置超时。如果您没有等待一组并发任务，也可以通过调用`​ray.get(future_id)​`来等待特定的未来。

如果没有参数，ray.init 假定本地执行并使用所有可用的 CPU 核心。您可以提供参数来指定要运行的集群、要使用的 CPU 或 GPU 核心的数量等。

假设一个远程函数通过另一个远程函数调用传递了未来。Ray 将自动对这些依赖关系进行排序，以便按照所需的顺序对它们进行评估。你不必自己做任何事情来处理这种情况。

假设您有一个有状态计算要做。当我们使用上面的`ray.get`时，我们实际上是从分布式对象存储中检索值。如果你愿意，你可以自己显式地把对象放在那里，用`​ray.put`返回一个 id，稍后你可以把它传递给`ray.get`再次检索它。

## 用参与者模型处理有状态计算

Ray 支持一种更直观、更灵活的方式来管理 actor 模型的设置和检索状态。它使用常规的 Python 类，这些类被转换成具有相同`@ray.remote`注释的远程参与者。为了简单起见，假设您需要计算 slow 被调用的次数。这里有一个这样的类:

```py
@ray.remote

class CountedSlows:

    def __init__(self, initial_count = 0):

        self.count = initial_count

    def slow(self, record):

        self.count += 1

        new_record = expensive_process(record)

        return new_record

    def get_count(self):

        return self.count

```

除了注释之外，这看起来像一个普通的 Python 类声明，尽管通常情况下您不会仅仅为了检索计数而定义`get_count`方法。我很快会回到这个话题。

现在以类似的方式使用它。注意类的实例是如何构造的，以及实例上的方法是如何调用的，使用前面的`remote`:

```py
cs = CountedSlows.remote() # Note how actor construction works

futures = [cs.slow.remote(r) for r in records]

while len(futures) > 0:

    finished, rest = ray.wait(futures)

    value = ray.get(finished[0])

print(value)
futures = rest

count_future_id = cs.get_count.remote()

ray.get(count_future_id)
```

最后一行应该打印与原始集合大小相等的数字。注意，我调用了方法`get_count`来检索属性`count`的值。目前，Ray 不支持像`​count`那样直接检索实例*属性*，所以与常规 Python 类相比，添加方法来检索它是一个必要的区别。

## Ray 统一了任务和演员

在上述两种情况下，Ray 跟踪任务和参与者在集群中的位置，消除了在用户代码中明确知道和管理这些位置的需要。actors 内部的状态变化以线程安全的方式处理，不需要显式的并发原语。因此，Ray 为应用程序提供了直观的、分布式的状态管理，这意味着 Ray 通常是实现*有状态* [无服务器](https://en.wikipedia.org/wiki/Serverless_computing)应用程序的优秀平台。此外，当在同一台机器上的任务和参与者之间通信时，通过共享内存透明地管理状态，参与者和任务之间的零拷贝序列化，以获得最佳性能。

**注意:**让我强调一下 Ray 在这里提供的一个重要好处。如果没有 Ray，当您需要在一个集群上横向扩展应用程序时，您必须决定创建多少个实例，将它们放在集群中的什么位置(或者使用 Kubernetes 之类的系统)，如何管理它们的生命周期，它们之间如何交流信息和协调，等等。等。Ray 为您做了这一切，而您只需付出最少的努力。你只需编写普通的 Python 代码。它是简化微服务架构设计和管理的强大工具。

## 采用射线

如果你已经在使用其他的并发 API，比如[多处理](https://docs.python.org/3.8/library/multiprocessing.html)、 [asyncio](https://docs.python.org/3.8/library/asyncio.html?highlight=asyncio#module-asyncio) 或者 [joblib](https://joblib.readthedocs.io/en/latest/) 会怎么样？虽然它们在单台机器上可以很好地进行扩展，但是它们不提供对集群的扩展。Ray 最近介绍了这些 API 的实验性实现，允许您的应用程序扩展到一个集群。代码中唯一需要更改的是 import 语句。例如，如果您正在使用`multiprocessing.Pool`，这是通常的导入语句:

```py
from multiprocessing.pool import Pool

```

要使用射线实现，请使用以下语句:

```py
from ray.experimental.multiprocessing.pool import Pool
```

这就够了。

Dask 怎么样，它似乎提供了许多与 Ray 相同的功能？如果您想要分布式集合，比如 numpy 数组和 Pandas 数据帧，Dask 是一个不错的选择。(一个使用 Ray 的名为[摩丁](https://github.com/modin-project/modin)的研究项目最终将满足这一需求。)Ray 是为更一般的场景设计的，在这些场景中，需要分布式状态管理，异构任务执行必须在大规模上非常有效，就像我们需要强化学习一样。

## 结论

我们已经看到 Ray 的抽象和特性如何使它成为一个简单易用的工具，同时提供强大的分布式计算和状态管理功能。虽然 Ray 的设计是由高性能、高要求的 ML/AI 应用程序的特定需求推动的，但它具有广泛的适用性，甚至提供了一种新的方法来实现基于微服务的架构。我希望你觉得这个关于 Ray 的简短解释很有趣。请尝试一下，让我知道你的想法！发送至: [dean@anyscale.io](mailto:dean@anyscale.io)

#### 了解更多信息

有关 Ray 的更多信息，请查看以下内容:

*   [2020 年 5 月 27 日至 28 日在旧金山举行的射线峰会](https://events.linuxfoundation.org/ray-summit/)。聆听案例研究、研究项目和对 Ray 的深入探讨，以及数据科学和人工智能社区领导者的早间主题演讲！
*   [雷网站](https://ray.io/)是雷所有事情的起点。
*   几个基于笔记本的[射线教程](https://github.com/ray-project/tutorial)让你尝试射线。
*   Ray GitHub 页面是你可以找到所有 Ray 源代码的地方。
*   Ray 文档说明了一切:[登陆页面](https://ray.readthedocs.io/en/latest/)，[安装说明](https://ray.readthedocs.io/en/latest/installation.html)。
*   直接向 [Ray Slack workspace](https://docs.google.com/forms/d/e/1FAIpQLSfAcoiLCHOguOm8e7Jnn-JJdZaCxPGjgVCvFijHB5PLaQLeig/viewform?usp=send_form) 或 [ray-dev Google Group](https://groups.google.com/forum/?nomobile=true#!forum/ray-dev) 提出关于 Ray 的问题。
*   推特上的[雷账号](https://twitter.com/raydistributed)。
*   一些射线项目:
    *   [RLlib](https://docs.ray.io/en/latest/rllib/index.html) :用 Ray 进行可扩展强化学习(还有这篇 [RLlib 研究论文](https://arxiv.org/abs/1712.09381))
    *   [调谐](https://ray.readthedocs.io/en/latest/tune.html):利用射线进行高效的超参数调谐
    *   Serve:灵活、可扩展的模型，使用 Ray 服务
    *   [摩丁](https://github.com/modin-project/modin):用射线加速熊猫的研究项目
    *   [FLOW](https://flow-project.github.io/) :使用强化学习进行交通控制建模的计算框架
    *   [Anyscale](https://anyscale.io/) :雷背后的公司
*   有关更多技术细节:
    *   一篇详细描述射线系统的研究论文
    *   一篇[研究论文，描述了用于深度学习的 Ray 内部的灵活原语](https://pdfs.semanticscholar.org/0e8f/5cd8d8dbbe4a55427e90ed35977e238b1eed.pdf?_ga=2.179761508.1978760042.1576357334-1293756462.1576357334)
    *   [用雷](https://ray-project.github.io/2017/10/15/fast-python-serialization-with-ray-and-arrow.html)和[快速序列化阿帕奇箭头](https://arrow.apache.org/)