# Ray for Data Science:大规模分布式 Python 任务

> 原文：<https://www.dominodatalab.com/blog/ray-for-data-science-distributed-python-tasks-at-scale>

**编者按:** 本文原载于[帕特森咨询博客](http://www.pattersonconsultingtn.com/blog/blog_index.html)经许可已转载。

## 我们为什么需要雷？

训练机器学习模型，尤其是神经网络，是计算密集型的。但是，大部分负载可以划分为较小的任务，并分布在大型集群上。几年前，加州大学伯克利分校的人工智能研究人员需要一种简单的方法来为他们正在研究和开发的算法做这件事。他们是专业的 Python 程序员，但是他们不想花费大量的时间和精力使用大多数可用的工具包。他们不需要太多复杂或精细的控制。他们只需要尽可能简单的 API，在幕后做出良好的默认选择，以在集群上扩展工作，利用可用资源，重新启动失败的任务，并管理计算结果以方便使用。

出于这种需求，出现了 Ray，这是一个开源系统，用于将 Python(现在是 Java)应用程序从单机扩展到大型集群。Ray 吸引我的地方在于它的 API 简单、简洁，使用起来很直观，尤其是对于没有分布式计算经验的人来说，但是它对于很多问题都非常灵活。

您当然可以找到具有更大扩展性和灵活性的更复杂的工具包，但是它们总是需要更多的努力来学习和使用。对于许多项目来说，这可能是正确的选择，但是当您需要一个易于使用的框架，并且您不需要对它的工作方式进行绝对控制时，例如为了实现最大可能的性能和效率，那么 Ray 是理想的。

## 光线任务

让我们看一个简单的例子，它说明了使用 Ray 在集群上进行“低级”分布是多么容易。然后，我将简要介绍几个利用 Ray 的机器学习高级工具包。Ray API 文档和 [Ray 网站](https://ray.io/)提供了比我在这里所能涵盖的更多的信息。

假设我们想要实现一个简单的 DNS 服务器。我们可以如下开始。如果您在家独自玩游戏，请将这段代码复制并粘贴到 Python 解释器中:

```py
import time    # We'll use sleep to simulate long operations.

addresses = { # Some absolutely correct addresses.

    "google.com": "4.3.2.1",

    "microsoft.com": "4.3.2.2",

    "amazon.com": "4.3.2.3",
}

def lookup(name): # A function to return an address for a name.

    time.sleep(0.5) # It takes a long time to run!

    return name, addresses[name] # Also return the name with the address.

start = time.time() # How long will this take?

for name in addresses: # Use the keys in addresses...

    n, address = lookup(name) # ... but go through lookup for the values.

    delta = time.time() - start

    print(f"{name}:\t {address} ({delta:.3f} seconds)")
```

```py
# The results:
# google.com: 4.3.2.1 (0.504 seconds)
# microsoft.com: 4.3.2.2 (1.008 seconds)
# amazon.com: 4.3.2.3 (1.511 seconds)
```

末尾的注释显示，每个查询大约需要 0.5 秒。让我们用 Ray 来减少这个开销。

首先，您需要使用`pip install ray`安装 Ray。这就是我们在这个练习中需要做的全部工作，我们将只在一个进程中运行 Ray，但是它将根据我们的需要在我们的 CPU 内核中利用尽可能多的线程。如果你想在一个集群中运行 Ray，你可以按照[文档](https://docs.ray.io/en/latest/cluster/getting-started.html)中描述的步骤进行设置。

在 pip 安装 Ray 后重启 Python 解释器。现在我们可以创建一个 Ray 任务来并行运行这些查询。

```py
import time

import ray.   # Import the Ray library

ray.init() # Initialize Ray in this application

addresses = {
"google.com": "4.3.2.1",

    "microsoft.com": "4.3.2.2",

    "amazon.com": "4.3.2.3",
}

@ray.remote # Decorator turns functions into Ray Tasks.

def ray_lookup(name): # Otherwise, it's identical to lookup().

    time.sleep(0.5)

    return name, addresses[name]

start = time.time()

for name in addresses:

    reference = ray_lookup.remote(name) # Start async. task with foo.remote().

    n, address = ray.get(reference) # Block to get the result.

    delta = time.time() - start

    print(f"{name}:\t {address} ({delta:.3f} seconds)")
```

```py
# google.com: 4.3.2.1 (0.520 seconds)
# microsoft.com: 4.3.2.2 (1.024 seconds)
# amazon.com: 4.3.2.3 (1.530 seconds)
```

我们没有改进我们的结果，但我们会马上解决这个问题。首先，我们来讨论一下有什么新内容。

您导入光线库并调用`ray.init()`在您的应用程序中初始化它。您可以向`ray.init()`传递参数，以连接到正在运行的集群，配置一些行为，等等。

当你用`@ray.remote`修饰一个 Python 函数的时候，你把它转换成了一个 Ray 任务。当被调用时，它将在集群中的某个地方异步运行，或者在我们的例子中，只在笔记本电脑的 CPU 核心上异步运行。这已经提供了轻松打破 Python 本身的单线程限制的能力。所有的核心都是属于我们的！

请注意循环是如何变化的。当您调用一个任务时，您将`.remote(...)`添加到函数中。这种必要变化的一个好处是为读者提供了文档；很明显，一个 Ray 任务正在被调用。

一旦任务完成，任务立即返回一个引用，该引用可用于检索任务的结果。我们通过调用`ray.get(reference)`来立即实现这一点，它会一直阻塞到任务完成。

这就是我们没有提高成绩的原因。我们等待每个任务完成，一次一个。然而，这很容易解决。我们应该“启动”所有的任务，然后立刻等待结果，而不是立即调用`ray.get(reference)`:

```py
start = time.time()

references = [ray_lookup.remote(name) for name in addresses]

ns_addresses = ray.get(references) # Wait on all of them together.

for name, address in ns_addresses:

    delta = time.time() - start

    print(f"{name}:\t {address} ({delta:.3f} seconds)")
```

```py
# google.com: 4.3.2.1 (0.513 seconds)
# microsoft.com: 4.3.2.2 (0.513 seconds)
# amazon.com: 4.3.2.3 (0.513 seconds)
```

好多了！它仍然需要至少 0.5 秒，因为没有一个任务能比这更快完成。对`ray.get(array)`的调用仍然阻塞，直到所有调用完成。还有一个`ray.wait()` API 调用，可以用来避免阻塞，并在结果可用时处理它们。详见[文档](https://docs.ray.io/en/latest/package-ref.html#ray-wait)和[光线教程](https://github.com/anyscale/academy/blob/master/ray-crash-course/01-Ray-Tasks.ipynb)。

## 射线演员

分布式编程的一大挑战是管理分布式状态。Ray 用行动者的概念解决了这个问题。如果您曾经为 JVM 使用过 Erlang 语言或 Akka 系统，那么您就使用过 actors。

基本上，actor 是用 Python 类实现的，任何状态都由类实例中的字段保存。这些实例的 Ray actor 封装确保了许多 Ray 任务或其他 actor 同时与 actor 交互时的线程安全。我们的演员就像“迷你服务器”。

让我们使用一个 actor 来保存 DNS 数据。到目前为止，我们在尝试访问单个字典时遇到了瓶颈，它“卡”在了我们的驱动程序 ipython 进程中。有了 actors，我们可以在一个集群上运行任意多的 actors，并将负载分配给它们。在 Ray 中甚至有一些工具，您可以在其中查询正在运行的演员。我们现在不看这两个特性，我们只用一个演员。

首先，这位是`DNSServer`雷演员:

```py
import ray

@ray.remote

class DNSServer(object):

    def __init__(self, initial_addresses):

        # A dictionary of names to IP addresses.

        self.addresses = initial_addresses

    def lookup(self, name):

        return name, self.addresses[name]

    def get_addresses(self):

        return self.addresses

    def update_address(self, name, ip):

        self.addresses[name] = ip
```

除了熟悉的`@ray.remote`装饰器，这看起来像一个常规的 Python 类，尽管我们也添加了一个`get_addresses`方法。在普通的 Python 对象中，您可以只读取像地址这样的字段。射线参与者需要 getter 方法来读取字段。

现在让我们使用它。为了方便起见，我将展示整个 Python 脚本，包括我们上面已经定义的一些内容。让我们从演员的设置开始:

```py
import ray

import time

from dns_server import DNSServer

#ray.init() # Uncomment if this is a new ipython session.

server = DNSServer.remote({ # Construct actor instances with .remote

    "google.com": "4.3.2.1",

    "microsoft.com": "4.3.2.2",

    "amazon.com": "4.3.2.3",
})

server.update_address.remote("twitter.com", "4.3.2.4")

server.update_address.remote("instagram.com", "4.3.2.5")

ref = server.get_addresses.remote()

names_addresses = ray.get(ref)

for name, address in names_addresses.items():

    print(f"{name}:\t {address}")
```

```py
# google.com: 4.3.2.1
# microsoft.com: 4.3.2.2
# amazon.com: 4.3.2.3
# twitter.com: 4.3.2.4
# instagram.com: 4.3.2.5
```

注意，实例是用 remote 构造的，方法是用 remote 调用的，就像我们对任务所做的那样。现在我们可以使用 actor:

```py
@ray.remote

def ray_lookup(name):             # Now use the server.

  time.sleep(0.5)

  return name, server.lookup.remote(name)

start = time.time()

refs = [ray_lookup.remote(name) for name in names_addresses.keys()]

names_refs2 = ray.get(refs)

for name, ref2 in names_refs2:

    delta = time.time() - start

    name2, address = ray.get(ref2)

    print(f"{name}:\t {address} ({delta:.3f} seconds)")
```

```py
# google.com: 4.3.2.1 (0.512 seconds)
# microsoft.com: 4.3.2.2 (0.512 seconds)
# amazon.com: 4.3.2.3 (0.516 seconds)
# twitter.com: 4.3.2.4 (0.517 seconds)
# instagram.com: 4.3.2.5 (0.519 seconds)
```

我们并不真的需要通过`ray_lookup`来调用服务器，但我们还是会这样做。产生了两个级别的引用。首先，`ray_lookup`返回服务器返回的 IP 地址的名称和引用。因此`names_refs`是一个名称引用对的数组。然后，当我们在每个引用上调用`ray.get(ref2)`时，我们得到名称和地址的另一个副本。值得打印出每个对`ray.get`的调用返回的内容，以了解发生了什么。

如果您编写了大量 Python 代码，并且偶尔发现自己需要并行化工作以提高速度，无论是在笔记本电脑上还是在集群中，我希望您能够体会到 Ray 在这方面的简洁。您甚至可以用参与者来管理状态。正如上一个例子所示，您必须小心地管理通过引用的“间接性”,这样它才不会变得太复杂，但是在大多数现实世界的应用程序中，这并不难做到。

Ray API 对于各种应用程序来说都是非常通用和灵活的。虽然它出现在 ML/AI 世界，但它完全不限于数据科学应用。

## 机器学习的射线

然而，因为 Ray 出现在 ML/AI 研究社区，所以大多数使用 Ray 的可用库都是以 ML 和 AI 为中心的。我现在讨论几个。

### 雷·里布

当 Deep Mind 使用强化学习实现专家级游戏时，强化学习成为 ML 中的热门话题，首先是在雅达利游戏中，然后是击败世界上最好的围棋选手。 [Ray RLlib](https://docs.ray.io/en/master/rllib/index.html) 是世界领先的 RL 库之一，其性能与 RL 算法的定制实现不相上下，但它非常通用，可以支持各种 RL 算法和环境，如游戏、机器人等。，为此你可以训练一个 RL 系统。下面是一个使用命令行工具的快速示例，尽管您也可以使用 Python API。首先你需要安装 RLlib，`pip install ‘ray[rllib]’`。然后，您可以运行以下命令。$是命令提示符(*NIX 或 Windows)。该命令包含两行。产量很大。我将显示第一条和最后一条“状态”消息，我根据页面进行了编辑:

```py
$ rllib train --run PPO --env CartPole-v1 --stop='{"training_iteration": 20}' --checkpoint-freq 10 --checkpoint-at-end
```

```py
…

== Status ==

Memory usage on this node: 19.4/32.0 GiB

Using FIFO scheduling algorithm.

Resources requested: 3/8 CPUs, 0/0 GPUs, 0.0/11.52 GiB heap, 0.0/3.96 GiB objects

Result logdir: /Users/deanwampler/ray_results/default

Number of trials: 1 (1 RUNNING)

+--------------------+------------+-------+------+--------------+-------+----------+

| Trial name         | status     | loc   | iter |     time (s) |    ts |   reward |

+--------------------+------------+-------+------+--------------+-------+----------+

| PPO_CartPole-v1_…. | RUNNING    | ip:N  |    1 |      7.39127 |  4000 |  22.2011 |

+--------------------+------------+-------+------+--------------+-------+----------+
```

```py
…
== Status ==
Memory usage on this node: 19.3/32.0 GiB
Using FIFO scheduling algorithm.
Resources requested: 0/8 CPUs, 0/0 GPUs, 0.0/11.52 GiB heap, 0.0/3.96 GiB objects
Result logdir: /Users/deanwampler/ray_results/default
Number of trials: 1 (1 TERMINATED)
+--------------------+------------+-------+------+--------------+-------+----------+
| Trial name | status | loc | iter | time (s) | ts | reward |
|--------------------+------------+-------+------+--------------+-------+----------|
| PPO_CartPole-v1_…. | TERMINATED | ip:N | 20 | 95.503 | 80000 | 494.77 |
+--------------------+------------+-------+------+--------------+-------+----------+
```

这个命令在横拉杆环境中训练一个神经网络，我一会儿会描述它。它使用一种流行的分布式训练算法，称为 PPO(“近似策略优化”)，其停止条件是它应该只运行 20 次迭代。您还可以指定与性能相关的停止条件，比如奖励值。- checkpoint*标志控制“代理”(在 CartPole 环境中运行的东西)的检查点的保存频率。这包括正在训练的简单神经网络。我们主要关心最终的代理，它是- checkpoint-at-end 保存的，但是- checkpoint-freq 标志在作业由于某种原因失败时很有用。我们可以从最后一个检查点重新开始。

侧手翻是训练环境的“hello world”。它是 [OpenAI](http://openai.com/) 库“健身房”的一部分，用于“练习”RL 算法。这个想法是训练一辆手推车来平衡一根垂直的柱子，手推车可以向左或向右移动，而柱子被限制在二维空间内。

我说模型是检查点，但是在哪里呢？默认情况下，它会写入您的`$HOME/ray_results directory`。我们可以使用最后一个检查点来“推出”模型，看看它的效果如何。这里我省略了以`PPO_CartPole-v1`开头的完整目录名。同样，该命令包含两行:

```py
$ rllib rollout ~/ray_results/default/PPO_CartPole-v1.../checkpoint_20/checkpoint-20 --run PPO
```

```py
…

Episode #0: reward: 500.0

Episode #1: reward: 484.0

...

Episode #19: reward: 458.0

Episode #20: reward: 488.0

Episode #21: reward: 367.0
```

将弹出一个对话框，动画展示该卷展栏，以便您可以看到它的工作效果。你会看到杆子开始向左或向右倾斜，而手推车会移动试图保持它直立。奖励点数计算购物车成功保持垂直且不触及左边界或右边界的迭代次数，最多 500 次迭代。效果非常好。

因此，命令行 rllib 对于许多快速训练运行和实验来说是非常好的，但是当您需要更深入地研究时，有一个完整的 Python API。参见 [Anyscale Academy RLlib 教程](https://github.com/anyscale/academy/tree/master/ray-rllib)获取深入示例。

### 射线调谐

我说我们用 RLlib 训练了一个神经网络，但是那个网络的参数是什么，是最优的吗？默认情况下，使用具有两个 256 个参数的隐藏层的网络。这些是许多超参数中的两个，我们可能想要调整它们来优化神经网络的架构和我们的 RL 代理的有效性。

[雷调](http://tune.io/)就是为此而造。使用它自己的 CLI 或 API，您可以指定想要优化的超参数，通常指定一个允许值的范围，以及任何固定的超参数，然后 tune 会为您运行实验以找到最佳性能。Tune 使用多种算法来优化这个潜在代价高昂的过程，例如提前终止看起来不太理想的超参数集的训练运行。

### 其他射线库

Ray 还附带了其他几个库，如[文档](https://docs.ray.io/en/latest/index.html)所示，许多第三方库现在都使用 Ray 来实现可伸缩的计算。

## 从这里去哪里

我希望你觉得这个简短的介绍雷有趣。要了解更多关于 Ray 的信息，请查看以下资源:

*   [Ray 网站](https://ray.io/) -社区信息、[教程](https://github.com/anyscale/academy)、[文档](https://docs.ray.io/en/latest/index.html)。
*   我给过的关于雷的更深入的演讲:
    *   [从笔记本电脑到集群的射线可扩展性](https://deanwampler.github.io/polyglotprogramming/papers/ClusterWideScalingOfMLWithRay.pdf)
    *   [用射线强化学习](https://deanwampler.github.io/polyglotprogramming/papers/ReinforcementLearningWithRayRLlib.pdf)
    *   [用于自然语言处理的射线](https://deanwampler.github.io/polyglotprogramming/papers/RayForNLP.pdf)
*   Anyscale -开发 Ray 和基于 Ray 的产品和服务的商业公司。[博客](https://www.anyscale.com/blog)和[事件](https://www.anyscale.com/events)页面提供了大量关于 Ray 在各种组织和行业中使用的有用信息。

[![Webinar  50x Faster Performance with Ray and NVIDIA GPUs  See Ray in action, boosted by performance gains from NVIDIA GPU-acceleration. Watch the webinar](img/11a384e7b01583cacaef89b7e8813253.png)](https://cta-redirect.hubspot.com/cta/redirect/6816846/17946acf-bac1-417d-8261-1b048f64d48b)