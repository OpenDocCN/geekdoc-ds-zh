# 使用 R 和 Python 的多核数据科学

> 原文：<https://www.dominodatalab.com/blog/multicore-data-science-r-python>

这篇文章展示了许多不同的包和方法，用于利用 R 和 Python 的并行处理。

## R 和 Python 中的多核数据科学

时间是宝贵的。数据科学涉及越来越苛刻的处理要求。从训练越来越大的模型，到特征工程，到超参数调整，处理能力通常是实验和构思的瓶颈。

利用云上的大型机器实例，数据科学家可以在更大的数据集、更复杂的模型和要求更高的配置上使用他们已经知道并喜欢的统计编程环境，如 R 和 Python。拥有 128 个内核和 2tb 内存的大规模硬件，如 [AWS X1 instances](//blog.dominodatalab.com/high-performance-computing-with-amazons-x1-instance/) ，已经突破了不需要复杂、难以管理、不熟悉的分布式系统就能完成的极限。

利用更大的数据集和更强的处理能力，数据科学家可以进行更多实验，对解决方案更有信心，并为业务构建更准确的模型。并行处理过去需要专业的技能和理解，利用细粒度多线程和消息传递的基本构建块。现代数据科学堆栈提供了高级 API，数据科学家可以利用大型计算实例，同时在他们最高效的抽象级别工作。

在关于 R 和 Python 视频的多核数据科学中，我们介绍了许多 R 和 Python 工具，这些工具允许数据科学家利用大规模架构来收集、写入、转换和操作数据，以及在多核架构上训练和验证模型。您将看到样本代码、真实世界的基准测试，以及使用 Domino 在 AWS X1 实例上运行的实验。

对于 R，我们涵盖了并行包、数据表、插入符号和 multidplyr。在 Python 中，我们涵盖了 paratext、joblib 和 scikit-learn。

最后，我们展示了数据科学家可以利用多核架构的强大语言无关工具。在视频中，我们使用了 H2O.ai 和 xgboost——这两个尖端的机器学习工具可以通过设置单个参数来利用多核机器。

[https://www.youtube.com/embed/6RwPR4OzwaI?feature=oembed](https://www.youtube.com/embed/6RwPR4OzwaI?feature=oembed)

下面是完整视频中包含的材料示例。

## r 包

### 平行包装

Package parallel 最初包含在 R 2.14.0 中，它为 apply 的大部分功能提供了嵌入式并行替代，集成了随机数生成处理。并行可以在许多不同层次的计算中完成:这个包主要关注“粗粒度并行化”。关键的一点是，这些计算块是不相关的，不需要以任何方式进行通信。

在这个示例代码中，我们利用 parallel 从一个文件夹中并行读取一个包含 100 个 csv 文件的目录。这可以让我们利用多核架构来更快地解析和接收磁盘上的数据。

```py
library(parallel)

numCores <- detectCores() # get the number of cores available

results <- mclapply(1:100,

             FUN=function(i) read.csv(paste0("./data/datafile-", i, ".csv")),

             mc.cores = numCores)

```

这段代码中引入的第一个多核概念在第 3 行，在这里我们调用 detectCores()函数来检索该进程可用的内核数。这将查询底层操作系统，并返回一个表示处理器数量的整数。值得注意的是，最佳并行性通常并不意味着使用所有可用的内核，因为它可能会使其他资源饱和并导致系统崩溃，因此请记住进行实验和基准测试。

在第 5 行，我们调用从并行包导入的 mclapply()(或多核 lapply)函数。这几乎是 R 的古老的 lapply 功能的替代物。

从业者应该意识到两个不同之处:

1.  mc.cores 参数为用户提供了一种设置要利用的内核数量的方法(在我们的示例中，只需检测到所有内核)。
2.  FUN 中的代码是在一个单独的 R 进程中执行的，因此继承了分叉语义。

主要的要点是，访问全局变量和处理全局状态必然与所有代码在单个进程中执行时不同。

Parallel 提供了一个很好的工具，通过一些小的调整就可以快速扩展现有的代码。随着内核数量的增加，衡量真正的性能非常重要，并且要记住，如果使用全局变量，分叉语义将需要对代码进行一些重组，但是如果您想将代码速度提高 100 倍以上，这一切都是值得的！

### 数据表

Data.table 是一个古老而强大的包，主要由 Matt Dowle 编写。它是 R 的数据帧结构的高性能实现，具有增强的语法。有无数的基准测试展示了 data.table 包的强大功能。它为最常见的分析操作提供了高度优化的表格数据结构。

Matt Dowle 警告不要在多核环境中使用 data.table，那么为什么要在多核网络研讨会和博客文章中讨论它呢？在 2016 年 11 月宣布，data.table 获得了 fwrite 的完全并行化版本！允许 R 以显著的加速写出数据！

在这个示例代码中，我们使用 data.table fwrite 包来编写一个利用多核架构的大型 CSV。

```py
library(parallel)

library(data.table)

numCores <- detectCores()

nrows_mult <- 1000

ncols_mult<- 1000

big_data <- do.call("rbind", replicate(nrows_mult, iris, simplify = FALSE))

big_data <- do.call("cbind", replicate(ncols_mult, big_data, simplify = FALSE))

fwrite(big_data, "/tmp/bigdata.csv", nThread=numCores)
```

在第 6-10 行中，我们多次复制 iris 数据集，使其成为一个非常大的内存数据结构。使用标准工具写入磁盘会花费大量时间。

为了利用多个内核，在第 12 行，我们使用参数 nThread 调用 fwrite()函数，参数是我们在第 4 行检测到的内核数量。在这种情况下，并行性的数量是有限制的，因为 io 子系统可能无法跟上大量的线程，但正如视频中的基准测试所示，它可以产生重大影响；有时候，写出文件的速度快 50%是成功与失败的区别。

### 脱字号

caret 包(分类和回归训练)是一组功能，用于简化创建预测模型的过程。该软件包包含用于数据分割、预处理、[特征选择](/data-science-dictionary/feature-selection)、[模型调整](/data-science-dictionary/model-tuning)使用重采样、变量重要性估计和其他功能的工具。

r 中有许多不同的建模函数。有些函数有不同的模型训练和/或预测语法。该包最初是作为一种为函数本身提供统一接口的方式，以及标准化常见任务，如参数调整和变量重要性。

caret 包无缝、轻松地利用多核功能。它利用了模型训练中的许多操作(例如使用不同超参数的训练和交叉验证)是并行的这一事实。

在示例代码中，我们利用 caret 的多核能力来训练 glmnet 模型，同时进行超参数扫描和交叉验证:

```py
library(parallel)

library(doMC)

library(caret)

numCores <- detectCores()

registerDoMc(cores = numCores)

model_fit<- train(price ~ ., data=diamonds, method="glmnet", preProcess=c("center", "scale"), tuneLength=10)
```

这段代码现在应该看起来很熟悉了。主要的区别出现在第 2 行，这里我们包含了 doMC 包。这个包为 caret 包提供了一个多核“后端”,并处理所有内核间的任务分配。

在第 6 行，我们记录了 doMC 集群可用的内核数量。

在第 8 行，我们训练模型，对值做一些预处理。

不需要将核心数量传递给 caret，因为它会自动从已经创建的集群中继承该信息。

有许多参数可以传递给 caret。在本例中，我们传递的 tuneLength 为 10，这创建了一个由超参数组成的大型调整网格。这将创建几十个甚至几百个具有不同配置的模型，并在优化最有意义的指标的基础上为我们提供最佳模型，在本例中为 RMSE。

### 多层

Multidplyr 是一个用于将数据帧划分到多个内核的后端。您告诉 multidplyr 如何用 partition()分割数据，然后数据会保留在每个节点上，直到您用 collect()显式检索它。这最大限度地减少了移动数据所花费的时间，并最大限度地提高了并行性能。

由于与节点间通信相关的开销，对于少于 1000 万次观察的基本 dplyr 动词，您不会看到太多的性能改进。但是，如果您使用 do()进行更复杂的操作，您会发现改进速度会更快。

在示例代码中，我们使用 multidplyr 在数据集上训练大量 GAM 模型:

```py
library(multidplyr)
library(dplyr)
library(parallel)
library(nycflights13)

numCores <- detectCores()

cluster <- create_cluster(cores)

by_dest <- flights %>%
 count(dest) %>%
 filter(n >= 365) %>%
 semi_join(flights, .) %>%
 mutate(yday = lubridate::yday(ISOdate(year, month, day))) %>%
 partition(dest, cluster = cluster)

cluster_library(by_dest, "mgcv")
models <- by_dest %>%
 do(mod = gam(dep_delay ~ s(yday) + s(dep_time), data = .))
```

这段代码比我们前面的例子稍微复杂一些。主要区别在于，在第 8 行中，我们使用第 6 行中检测到的内核数量来显式初始化集群。

multidplyr 包可以处理启动集群的所有潜在挑战，并通过简化的界面以透明的方式为我们处理这些挑战。

第 10-15 行我们获取航班数据集，对其进行一些特征工程，然后将其划分到第 15 行的集群中。

这意味着我们获取数据集的子集，并将其发送到每个内核进行处理。这种抽象比其他抽象(如 caret 的抽象)处于更低的层次，但它确实让我们有相当大的能力来决定代码如何在多个内核之间准确分布。

在第 17 行，我们向集群广播，为了执行下面的管道，它将需要“mgcv”库。

最后，在第 18 行和第 19 行，我们跨集群并行训练了大量 GAM 模型，每个模型都按目的地进行了划分。

Multidplyr 仍然是一个早期的包，Hadley 和他的团队正在使用它来研究和理解如何将多核技术引入 tidyverse。它现在对许多用例都很有用，并且是未来事物的一个激动人心的预览。

## Python 包

### 副档名

读取 CSVs can 是一项耗时的任务，通常会成为数据处理的瓶颈。如果您正在利用具有几十个内核的大规模硬件，那么当您在读取 CSV 文件时，看到您的服务器大部分时间处于空闲状态，因为一个内核的利用率为 100%,这可能会令人感到羞愧。

paratext 库为 CSV 读取和解析提供了多核心处理。ParaText 是一个 C++库，用于在多核机器上并行读取文本文件。alpha 版本包括一个 CSV 阅读器和 Python 绑定。除了标准库之外，库本身没有依赖关系。

根据我们的基准，ParaText 是世界上最快的 CSV 阅读库。

在示例代码中，我们使用 paratext 读取一个非常大的 CSV，同时利用所有可用的内核。

```py
import paratext

import pandas as pd

mydf = paratext.load_csv_to_pandas("data/big_data.csv")
```

在这段极其简单的代码中，我们加载 paratext 库并使用 load_csv_to_pandas 函数从 big_data.csv 创建 pandas 数据帧。该函数将自动利用最佳数量的内核并提供显著的加速。

paratext 的惟一挑战是在您的特定环境中构建它并不容易。然而，在安装之后，它以最小的努力提供了显著的性能。

### Joblib

Joblib 是一组用于 Python 中轻量级管道的工具。特别是，joblib 提供:

1.  输出值的透明磁盘缓存和惰性重新评估(记忆模式)
2.  简单易行的并行计算
3.  记录和跟踪执行

Joblib 经过优化，特别是在处理大型数据时，速度更快、更健壮，并且针对 numpy 数组进行了特定的优化。Joblib 是 Python 中并行处理的基本构建块，不仅用于数据科学，还用于许多其他分布式和多核处理任务。

在示例代码中，joblib 用于查找哪些线段完全包含在群体中的其他线段中，这是一项令人尴尬的并行任务:

```py

import numpy as np

from matplotlib.path import Path

from joblib import Parallel, delayed

## Create pairs of points for line segments

all_segments = zip(np.random.rand(10000,2), np.random.rand(10000,2))

test_segments = zip(np.random.rand(800,2),np.random.rand(800,2))

## Check if one line segment contains another.

def check_paths(path):

    for other_path in all_segments:

        res='no cross'

        chck = Path(other_path)

        if chck.contains_path(path)==1:

            res= 'cross'

            break

    return res

res = Parallel(n_jobs=128) (delayed(check_paths) (Path(test_segment)) for test_segment in test_segments)
```

直到第 19 行的所有代码都在设置我们的环境。生成要验证的线段和线段，并创建一个名为 check_paths 的函数来遍历线段并检查其中是否包含其他线段。

第 19 行是我们对 joblib 的调用，其中我们创建了一个有 128 个线程的并行对象(这是在 AWS X1 实例上运行的)。它遍历 test_segments 中的值，为该 test_segment 创建一个 path 对象，然后为创建的对象调用 check_paths 函数。

注意，对 check_paths 的调用包装在 delayed()中，这允许 joblib 调度它，而不是立即执行它。

### Scikit-learn

Scikit-learn 是一个免费的 Python 编程语言的机器学习库。它具有各种分类、回归和聚类算法，包括支持向量机、随机森林、梯度推进、k-means 和 DBSCAN。它旨在与 Python 数字和科学库 NumPy 和 SciPy 进行互操作。

Scikit-learn 通过简单使用 n_jobs 参数使利用大型多核服务器变得容易。这适用于*多*模型、网格搜索、交叉验证等。

在示例代码中，我们训练了一个 RandomForestClassifier 来利用多个并行内核预测 iris 数据集中的物种:

```py
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics, datasets

iris = datasets.load_iris()

X = iris.data[:, :2] # we only take the first two features.

Y = iris.target

md = RandomForestClassifier(n_estimators = 500, n_jobs = -1)

md.fit(X, y)
```

scikit-learn 多核功能的强大之处在 online 8 中展示:为了利用任何系统上可用的所有内核，我们只需将值-1 传递给 n_jobs 参数。

没有必要建立一个集群，自省机器，或任何其他东西...这个简单的参数通常可以在 scikit-learn 的机器学习模型的训练中提供 100 倍的加速。

不是所有的模型都提供 n_jobs 参数，但是 scikit-learn 文档提供了一种方法来确定您的特定分类器是否支持这种简单的并行化。

## 了解更多，包括 H2O 和 Xgboost

观看关于 R 和 Python 的[多核数据科学的完整视频](https://www.youtube.com/watch?v=6RwPR4OzwaI)，了解 h2o 和 xgboost 中的多核功能，这是当今最流行的两个机器学习包。

在超级计算机级别的硬件上使用世界上最尖端的软件是一种真正的特权。看到这些工具在我的数据科学实践中让我变得更有效率是令人兴奋的，并且希望以类似的方式影响你。

[Domino Enterprise MLOps platform](https://www.dominodatalab.com/product/domino-data-science-platform/)通过强大的环境管理工具提供对大规模计算环境的访问，使得在大型硬件上测试该软件和管理这些包的多个版本的配置变得容易。

如果你有兴趣了解更多，你可以[请求一个 Domino](https://www.dominodatalab.com/demo/?utm_source=blog&utm_medium=post&utm_campaign=multicore-data-science-r-python) 的演示。