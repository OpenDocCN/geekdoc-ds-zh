# Dask 并行计算:一步一步的教程

> 原文：<https://www.dominodatalab.com/blog/dask-step-by-step-tutorial>

现在，随着时间的推移，计算能力不断提高是正常的。每月，有时甚至每周，新的设备被创造出来，具有更好的特性和更强的处理能力。然而，这些增强需要大量的硬件资源。接下来的问题是，你能使用一个设备提供的所有计算资源吗？在大多数情况下，答案是否定的，你会得到一个  `out of memory` 错误。但是，如何在不改变项目底层架构的情况下利用所有的计算资源呢？

这就是达斯克的用武之地。

在许多 ML 用例中，您必须处理巨大的数据集，如果不使用并行计算，您就无法处理这些数据集，因为整个数据集无法在一次迭代中处理。Dask 帮助您加载大型数据集，用于数据预处理和模型构建。在本文中，您将了解更多关于 Dask 的知识，以及它如何帮助并行化。

## Dask 是什么？

当开发人员处理小型数据集时，他们可以执行任何任务。然而，随着数据的增加，它有时不再适合内存。

Dask 是一个开源库，在 ML 和数据分析中提供高效的并行化。在 Dask 的帮助下，您可以轻松扩展各种 ML 解决方案，并配置您的项目以利用大部分可用的计算能力。

Dask 帮助开发人员扩展他们的整个 Python 生态系统，它可以与您的笔记本电脑或容器集群一起工作。Dask 是针对大于内存的数据集的一站式解决方案，因为它提供了多核和分布式并行执行。它还为编程中的不同概念提供了一个通用实现。

在 Python 中，像 pandas、NumPy 和 scikit-learn 这样的库经常被用来读取数据集和创建模型。所有这些库都有相同的问题:它们不能管理大于内存的数据集，这就是 Dask 的用武之地。

与 Apache Hadoop 或 Apache Spark 不同，在 Apache Hadoop 或 Apache Spark 中，您需要改变整个生态系统来加载数据集和训练 ML 模型，Dask 可以轻松地与这些库集成，并使它们能够利用并行计算。

最好的部分是你不需要重写你的整个代码库，你只需要根据你的用例进行最小的修改就可以实现并行计算。

### Dask 数据帧和 pandas 数据帧之间的差异

数据帧是数据的表格表示，其中信息存储在行和列中。pandas 和 Dask 是用来读取这些数据帧的两个库。

pandas 是一个 Python 库，用于将不同文件(例如 CSV、TSV 和 Excel)中的数据读入数据帧。它最适合处理相对少量的数据。如果你有大量的数据，你会收到一个  `out of memory` (OOM)错误。

为了解决 OOM 错误并实现并行执行，可以使用 Dask 库来读取 pandas 无法处理的大型数据集。Dask 数据帧是不同熊猫数据帧的集合，这些数据帧沿着一个索引分割:

![Dask DataFrame courtesy of Gourav Bais](img/b8713707dd57d360904d5e201c820ec0.png)

当您使用 Dask 数据帧读取数据时，会创建多个小熊猫数据帧，沿索引拆分，然后存储在磁盘上(如果内存有限)。当您在 Dask 数据帧上应用任何操作时，存储在磁盘上的所有数据帧都会触发该操作。

### Dask 的使用案例

Dask 数据帧可用于各种用例，包括:

*   **并行化数据科学应用:** 要在任何数据科学和 ML 解决方案中实现并行化，Dask 是首选，因为并行化不限于单个应用。您还可以在同一硬件/集群上并行处理多个应用。因为 Dask 使应用程序能够利用整个可用的硬件，所以不同的应用程序可以并行运行，而不会出现资源争用导致的问题。
*   **图像处理:** 这需要大量的硬件资源。如果你在 pandas 或 NumPy 上使用传统的方法，你可能会中止这个项目，因为这些库/工具不能执行多重处理。Dask 帮助您在多个内核上读取数据，并允许通过并行方式进行处理。

有关不同真实世界用例的更多信息，请查看 Dask Stories 文档中的 [Dask 用例](https://stories.dask.org/en/latest/)页面。

## Dask 实现

要实现 Dask，您需要确保您的系统中安装了 Python。如果没有，现在就下载并安装它。确保您安装的版本高于 3.7。(3.10 最新)。

### 软件包安装

如前所述，Dask 是一个 Python 库，可以像其他 Python 库一样安装。要在您的系统中安装软件包，您可以使用 Python 软件包管理器 pip 并编写以下命令:

```py
## install dask with command prompt

pip install dask

## install dask with jupyter notebook

! pip install dask
```

### 配置

当您安装完 Dask 后，您就可以使用默认配置(在大多数情况下都有效)来使用它了。在本教程中，您不需要显式地更改或配置任何东西，但是如果您想要配置 Dask，您可以查看这个[配置文档](https://docs.dask.org/en/stable/configuration.html)。

### 并行化准备

首先，看一个计算两个数平方和的简单代码。为此，您需要创建两个不同的函数:一个用于计算数字的平方，另一个用于计算数字的平方和:

```py
## import dependencies

from time import sleep

## calculate square of a number

def calculate_square(x):

    sleep(1)

    x= x**2

    return x

## calculate sum of two numbers

def get_sum(a,b):

    sleep(1)

    return a+b
```

`calculate_square()` 函数将一个数字作为输入，并返回该数字的平方。  `get_sum()` 函数接受两个数字作为输入，并返回这两个数字的和。故意增加一秒钟的延迟，因为这有助于您注意到使用和不使用 Dask 时执行时间的显著差异。现在，在不使用 Dask 的情况下，看看这个逻辑的执行时间:

```py
%%time

## call functions sequentially, one after the other

## calculate square of first number

x = calculate_square(10)

## calculate square of second number

y = calculate_square(20)

## calculate sum of two numbers

z = get_sum(x,y)

print(z)
```

前面的代码计算数字的平方，并将它们存储在变量  `x` 和  `y`中。然后计算并打印  `x` 和  `y` 之和。  `%%time` 命令用于计算函数执行所花费的 [CPU 时间](https://www.cs.fsu.edu/~hawkes/cda3101lects/chap2/index.html?$$$cputime.html$$$)和 [wall 时间](https://en.wiktionary.org/wiki/wall_time)。现在，您应该有这样的输出:

![Simple execution calculate_square function](img/98917be9f39fe5a1f443d6ef66de530a.png)

您可能会注意到，即使在故意延迟三秒钟之后，整个代码也花了 3.1 秒运行。

现在是时候看看 Dask 如何帮助并行化这段代码了(计算平方和的初始函数保持不变)。

首先，您需要导入一个名为  `delayed` 的 Python Dask 依赖项，用于实现并行化:

```py
## import dask dependencies

import dask

from dask import delayed
```

导入 Dask 依赖项后，就可以使用它一次执行多个作业了:

```py
%%time

## Wrapping the function calls using dask.delayed

x = delayed(calculate_square)(10)

y = delayed(calculate_square)(20)

z = delayed(get_sum)(x, y)

print(z)
```

在这个代码片段中，您使用 Dask `delayed` 函数将您的普通 Python 函数/方法包装成延迟函数，现在您应该得到如下所示的输出:

![Dask execution of function](img/97b5333a68416764fae916385b8ee4bd.png)

你可能会注意到  `z` 并没有给你答案。这是由于在这个实例中，  `z` 被认为是延迟函数的懒惰对象。它包含了计算最终结果所需的一切，包括不同的函数及其各自的输入。要得到结果，必须调用  `compute()` 方法。

然而，在调用  `compute()` 方法之前，先检查一下使用  `visualize()` 方法的并行执行会是什么样子:

```py
%%time

## visualize the task graph

z.visualize()
```

> **请注意:** 如果您在可视化图形时遇到任何错误，可能会有依赖性问题。你必须安装 Graphviz 来可视化任何图形。用  `pip install graphviz`可以做到这一点。

执行图表应该如下所示:

![Task graph for Dask](img/ccf9ab592a841248aa027092dc610920.png)

```py
%%time

## get the result using compute method

z.compute()
```

要查看输出，需要调用  `compute()` 方法:

![Dask `compute()` method](img/b25e057d99b0c0376a4e3626b697e6a0.png)

您可能会注意到结果中有一秒钟的时间差。这是因为  `calculate_square()` 方法是并行化的(在前面的图中形象化了)。Dask 的一个好处是，您不需要定义不同方法的执行顺序，也不需要定义要并行化哪些方法。达斯克会为你做的。

为了进一步理解并行化，看看如何将  `delayed()` 方法扩展到 Python 循环:

```py
## Call above functions in a for loop

output = []

## iterate over values and calculate the sum

for i in range(5):

    a = delayed(calculate_square)(i)

    b = delayed(calculate_square)(i+10)

    c = delayed(get_sum)(a, b)

    output.append(c)

total = dask.delayed(sum)(output)

## Visualizing the graph

total.visualize()
```

这里，代码迭代一个 for 循环，并使用 delayed 方法计算两个值的平方和:  `a` 和  `b`。这段代码的输出将是一个有多个分支的图形，看起来像这样:

![Dask with for-loop](img/9290d73c54f5c162f9d04c640b3b3474.png)

到目前为止，您已经了解了 pandas 数据帧和 Dask 数据帧之间的差异。然而，当你从事并行化工作时，理论是不够的。现在您需要回顾技术实现，Dask 数据帧在实践中如何工作，以及它们与 pandas 数据帧相比如何。

这里使用的样本数据集可以从这个 [GitHub repo](https://github.com/gouravsinghbais/Time-Series-Forecasting-with-Tensorflow-and-InfluxDB/blob/master/sunspots-dataset/Sunspots.csv) 下载。

### 技术实现

要开始测试 Dask 数据帧的技术实现，请运行以下代码:

```py
## import dependencies

import pandas as pd

import dask.dataframe as dd

## read the dataframe

pandas_dataframe = pd.read_csv('Sunspots.csv')

dask_dataframe = dd.read_csv('Sunspots.csv')

## calculate mean of a column

print(pandas_dataframe['Monthly Mean Total Sunspot Number'].mean())

print(dask_dataframe['Monthly Mean Total Sunspot Number'].mean().compute())
```

这段代码应该对 pandas 和 Dask 数据帧产生相同的结果；但是内部处理就大不一样了。熊猫数据帧将被加载到内存中，只有当它适合内存时，你才能对它进行操作。如果不适合，就会抛出错误。

相比之下，Dask 创建多个 pandas 数据帧，并在可用内存不足时将它们存储在磁盘上。当您在 Dask 数据帧上调用任何操作时，它将应用于构成 Dask 数据帧的所有数据帧。

除了延迟方法之外，Dask 还提供了其他一些特性，比如并行和分布式执行、高效的数据存储和延迟执行。你可以在 Dask 的官方文档中读到它。

## 使用 Dask 的最佳实践

每一种使开发更容易的工具或技术都有一些自己定义的规则和最佳实践，包括 Dask。这些最佳实践可以帮助您提高效率，让您专注于发展。Dask 最著名的一些最佳实践包括:

### 从基础开始

您不需要总是使用并行执行或分布式计算来找到问题的解决方案。最好从正常的执行开始，如果这不起作用，您可以继续使用其他解决方案。

### 仪表板是关键

当你在研究多重处理或多线程概念时，你知道事情正在发生，但是你不知道如何发生。Dask 仪表板通过提供清晰的可视化效果，帮助您查看员工的状态，并允许您采取适当的执行措施。

### 高效使用存储

尽管 Dask 支持并行执行，但您不应该将所有数据存储在大文件中。您必须根据可用内存空间来管理您的操作；否则，你会耗尽内存，或者在这个过程中会有一个滞后。

这些只是帮助您简化开发过程的一些最佳实践。其他建议包括:

*   避免非常大的分区: 这样它们就能放入一个工作者的可用内存中。
*   避免非常大的图: 因为那会造成任务开销。
*   **学习定制的技巧:** 为了提高你的流程效率。
*   **不再需要时停止使用 Dask:**比如当你迭代一个小得多的数据量时。
*   **能坚持就坚持分布式 RAM:**这样做，访问 RAM 内存会更快。
*   **进程和线程:** 注意将数字工作与文本数据分开，以保持效率。
*   **用 Dask 加载数据:**例如，如果你需要处理大型 Python 对象，让 Dask 创建它们(而不是在 Dask 之外创建它们，然后交给框架)。
*   **避免重复调用 compute:**，因为这样会降低性能。

有关更多信息，请查看 Dask [**最佳实践** 文章](https://docs.dask.org/en/stable/best-practices.html)。

## 通过 Domino 使用 Dask

Domino Data Lab 帮助您在 Domino 实例的基础设施上动态地提供和编排 Dask 集群。这使得 Domino 用户能够快速访问 Dask，而不需要依赖他们的 IT 团队来为他们设置和管理集群。

当您启动 Domino workspace 进行交互式工作或启动 Domino job 进行批处理时，Domino 会创建、管理并提供一个容器化的 Dask 集群供您执行。

要了解更多关于在 Domino 上使用 Dask 的信息，请查看我们的 GitHub 页面。

## Dask:解决并行性、硬件浪费和内存问题的一站式解决方案

早期的程序员、开发人员和数据科学家没有强大的系统来开发 ML 或数据科学解决方案。但是现在，随着强大的硬件和定期的改进，这些解决方案变得越来越受欢迎。然而，这带来了一个问题:数据库不是为利用这些解决方案而设计的，因为它们没有内存容量或者不能利用计算能力。

有了 Dask，您不必担心并行性、硬件浪费或内存问题。Dask 是所有这些问题的一站式解决方案。

当您开始构建一个解决方案(记住内存问题或计算能力)时，您需要将解决方案部署到目标受众可以访问的地方。Domino Enterprise MLOps 平台通过提供机器学习工具、基础设施和工作材料来加速数据科学工作的开发和部署，使团队能够轻松协调和构建解决方案。