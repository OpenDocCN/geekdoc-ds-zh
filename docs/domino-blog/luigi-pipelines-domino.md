# 用 Luigi 和 Domino 编排管道

> 原文：<https://www.dominodatalab.com/blog/luigi-pipelines-domino>

建立数据管道听起来可能是一项艰巨的任务。在本文中，我们将研究如何使用 Luigi——一个专门设计来帮助管道构建的库——来构建一个与 Domino 协同工作的数据管道。

## 什么是数据管道？

数据管道是进程(“任务”)的集合，每个进程对其他任务输出的数据进行操作。随着管道的转换，信息在管道中“流动”。

例如，统计模型开发的管道可能如下所示:

```py
clean data -> extract features -> fit model -> predict
```

更复杂的集合模型可能如下所示:

```py
clean data -> extract features -> fit models -> combine models -> predict
```

最简单的管道就是脚本中的一系列步骤。这通常就足够了，但并不总是如此。

例如，如果您后来添加了新的数据源或任务，并使管道的依赖结构变得复杂，您的脚本可能无法很好地伸缩。

此外，当任务失败时，最好不要重复前面的步骤，尤其是当任务快要结束时。一个失败的阶段可能会留下不完整或部分的输出，如果其他任务无意中使用了损坏的数据，这可能会引起连锁反应。

最后，脚本化管道可能没有考虑硬件限制，或者它可能无法充分满足某些任务对额外资源的需求(GPU、多个 CPU、额外内存等)。

## 路易吉

Luigi 是一个 [Python 库](https://github.com/spotify/luigi)，它通过解决这些和其他相关问题简化了数据管道的创建和执行。它起源于 Spotify，也被 Foursquare、Stripe、Asana 和 Buffer 等公司使用。它可以管理各种任务，包括用[其他编程语言](https://datapipelinearchitect.com/luigi-only-python/)的操作，这使得它对于现代数据科学中典型的多语言项目非常有用。

使用 Luigi，您可以根据数据依赖性(其他任务的输出)来定义每个任务。在运行时，中央调度程序会自动为您选择执行顺序。Luigi 还支持原子文件操作(有助于确保可再现性)和自动缓存先前运行的管道步骤(可以真正节省时间)。

如果你对学习 Luigi 感兴趣，从官方文档中的概述开始。然后继续讨论[执行模型](https://luigi.readthedocs.org/en/stable/execution_model.html)和[通用模式](https://luigi.readthedocs.org/en/stable/luigi_patterns.html)。你可能也想看看这个项目原作者的幻灯片概述。

## Domino 上的多机管道

即使在单台机器上，Luigi 也是一个有效的工具。将它与 Domino API 的功能结合起来，通过允许管道在任意多个运行之间伸缩，赋予了它更大的能力。这可以通过添加几个额外的类来实现。

例如，您可以在混合的 GPU 和高内存实例上并行训练一组模型，然后收集生成的模型文件用于比较或其他用途——所有这些都通过一个命令启动。

## 远程执行

`DominoRunTask`是`luigi.Task`的子类，表示在不同的 Domino 机器上的远程执行。只需告诉命令远程运行，给它预期的输出位置，并指定硬件层。在运行时，`DominoRunTask`将启动一个新的 Domino 运行，等待它完成，然后将输出复制回主管道运行，以便后续任务可以使用。

这里有一个简单的例子:

```py
import luigi

from domino_luigi import DominoRunTask

class Hello(luigi.Task):

    def output(self):

        return luigi.LocalTarget('hello_world_pipeline/hello.txt')

def run(self):

    with self.output().open('w') as f:

        f.write('hello\n')

class DominoHelloWorld(luigi.Task):

    def requires(self):

        return DominoRunTask(

            domino_run_command=['example_pipeline.py', 'Hello'], # --workers N

            domino_run_tier='Small',

            output_file_path=Hello().output().path,

        )

    def output(self):

        return luigi.LocalTarget('hello_world_pipeline/hello_world.txt') def run(self):

    def run(self):    

        with self.input().open('r') as in_f:

            with self.output().open('w') as out_f:

                out_f.write(in_f.read() + 'world\n')
```

这将开始第二次运行，创建输出文件`hello_world_pipeline/hello.txt`。这个文件然后被初始运行检索，并且文件的内容被用来创建`hello_world_pipeline/hello_world.txt`。

注意事项:

*   按照 Luigi 模式，外部运行的输出必须是单个文件。这是为了保证每个任务的原子性，可以防止数据损坏，有利于可复制性。如果您的外部作业创建了许多文件，只需创建一个 zip 存档，并将其作为输出。
*   `DominoRunTask`接受一个名为`commit_id`的参数，该参数指示 Domino 项目的状态，该状态将被用作运行的输入。如果没有提供这个参数，它默认为当前 Domino 运行的输入`commit_id`。使用 UI 的文件浏览器时，可以在 URL 中看到`commit_id`，例如`https://app.dominodatalab.com/u/domino/luigi-multi-run/browse?commitId=1dee27202efaabe0baf780e6b67a3c0b739a2f4c`。

## 多机系综

`DominoRunTask`可应用于预测模型开发。以这组模型为例:

*   用 R 写的随机森林，带有 [H2O 图书馆](https://www.h2o.ai/)。
*   使用 [scikit-learn](https://scikit-learn.org) python 库的渐变增强模型。
*   用 [nolearn](https://github.com/dnouri/nolearn) 构建的神经网络(建立在[千层面](https://github.com/Lasagne/Lasagne)和 theano 之上)。

每个模型都可以在自己的机器上训练。随机森林模型可能在 *X-Large [硬件层](https://support.dominodatalab.com/hc/en-us/articles/204187149-What-hardware-will-my-code-run-on-)* 上运行，而神经网络使用 *GPU 层*。由于每个训练任务都是它自己的运行，它们可以被单独检查。

您可以使用一个 Domino 命令开始训练和评估模型的整个过程:

```py
domino run example_ensemble.py ScoreReport --workers 3
```

这对于按照时间表重新训练[来说很方便。](https://support.dominodatalab.com/hc/en-us/articles/204843165-Scheduling-Runs)

在开发过程中，您可以通过运行不同的任务来处理管道的子组件:

```py
domino run example_ensemble.py TrainRForest
```

成功完成后，生成的模型文件可用于管道的后续运行，而无需重新训练。

注意事项:

*   在这个示例项目中，随机森林训练和预测代码在 [`forest_train.R`](https://app.dominodatalab.com/u/domino/luigi-multi-run/view/forest_train.R) 和 [`forest_predict.R`](https://app.dominodatalab.com/u/domino/luigi-multi-run/view/forest_predict.R) 中找到。请注意压缩包含 H2O 模型的目录所采取的额外步骤；添加这些是为了确保任务的原子性。
*   这些模型尚未调整。目的是以可扩展的方式说明概念验证，而不是呈现最佳匹配。