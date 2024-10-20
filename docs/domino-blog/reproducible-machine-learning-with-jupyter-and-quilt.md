# Jupyter 和 Quilt 的可重复机器学习

> 原文：<https://www.dominodatalab.com/blog/reproducible-machine-learning-with-jupyter-and-quilt>

## Jupyter 和 Quilt 的可重复机器学习

Jupyter 笔记本记录了代码和数据的交互。代码依赖关系很容易表达:

```py
import numpy as np

import pandas as pd
```

另一方面，数据依赖关系更加混乱:自定义脚本从网络上获取文件，解析各种格式的文件，填充数据结构，并处理数据。因此，跨机器、跨协作者以及随着时间的推移再现数据依赖性可能是一个挑战。Domino 的再现性引擎通过将代码、数据和模型组装到一个统一的中心来应对这一挑战。

我们可以将可再生机器学习视为一个包含三个变量的方程:

**代码+数据+模型=可复制的机器学习**

开源社区为复制第一个变量*代码*提供了强有力的支持。像 git、pip 和 Docker 这样的工具可以确保代码的版本化和统一的可执行性。然而，*数据*带来了完全不同的挑战。数据比代码大，有多种格式，需要高效地写入磁盘和读入内存。在这篇文章中，我们将探索一个开放源代码的数据路由器，[被子](https://quiltdata.com/)，它版本化和编组*数据*。Quilt 为数据所做的与 pip 为代码所做的一样:将数据打包到可重用的版本化构建块中，这些构建块可以在 Python 中访问。

在下一节中，我们将设置 Quilt 与 Jupyter 一起工作。然后，我们将通过一个例子来重现一个随机森林分类器。

## 推出带被子的木星笔记本

为了访问 Quilt，Domino cloud 用户可以在项目设置中选择“默认 2017-02 + Quilt”计算环境。或者，将以下几行添加到文件下的 `requirements.txt`:

```py
quilt==2.8.0

 scikit-learn==0.19.1
```

接下来，启动一个 Jupyter 工作区，用 Python 打开一个 Jupyter 笔记本。

## 用于机器学习的被子包

让我们用来自 [Wes McKinney 的 Python for Data Analysis，第二版](http://wesmckinney.com/pages/book.html)的数据建立一个机器学习模型。访问这些数据的旧方法是克隆 [Wes 的 git 库](https://github.com/wesm/pydata-book)，导航文件夹，检查文件，确定格式，解析文件，然后将解析后的数据加载到 Python 中。

有了被子，过程就简单多了:

```py
import quilt

quilt.install("akarve/pydata_book/titanic", tag="features",

force=True)

# Python versions prior to 2.7.9 will display an SNIMissingWarning
```

上面的代码具体化了 akarve/pydata_book 包的“titanic”文件夹中的数据。我们使用“features”标签来获取合作者已经完成了一些特性工程的包的特定版本。每个被子包都有一个用于文档的[目录条目](https://quiltdata.com/package/akarve/pydata_book)，一个惟一的散列，和一个历史日志(`$ quilt log akarve/pydata_book`)。

我们可以从 Wes 的书中导入如下数据:

```py
from quilt.data.akarve import pydata_book as pb
```

如果我们在 Jupyter 中评估`pb.titanic`，我们会看到它是一个包含 DataNodes 的 GroupNode:

```py
<GroupNode>
features
genderclassmodel
gendermodel
model_pkl
test
train
```

我们可以如下访问`pb.titanic`中的数据:

```py
features = pb.titanic.features()

train = pb.titanic.train()

trainsub = train[features.values[0]]
```

请注意上面代码示例中的括号。括号指示 Quilt“将数据从磁盘加载到内存中”Quilt 加载表格数据，如`features`所示，作为 pandas 数据帧。

让我们将训练数据转换成可在 scikit-learn 中使用的 numpy 数组:

```py
trainvecs = trainsub.values

trainlabels = train['Survived'].values
```

现在，让我们根据我们的数据训练一个随机森林分类器，然后进行五重交叉验证来衡量我们的准确性:

```py
from sklearn.model_selection import cross_val_score as cvs

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(max_depth=4, random_state=0)

rfc.fit(trainvecs, trainlabels)

scores = cvs(rfc, trainvecs, trainlabels, cv=5)

scores.mean()
```

该模型的平均准确率为 81%。让我们将模型序列化。

```py
from sklearn.externals import joblib

joblib.dump(rfc, 'model.pkl')
```

我们现在可以将序列化的模型添加到 Quilt 包中，这样合作者就可以用训练数据和训练好的模型来复制我们的实验。为了简单起见，`titanic`子包已经包含了我们训练过的随机森林模型。您可以按如下方式加载模型:

```py
from sklearn.externals import joblib

model = joblib.load(pb.titanic.model_pkl2())

# requires scikit-learn version 0.19.1
```

为了验证它是否与我们上面训练的模型相同，请重复交叉验证:

```py
scores = cvs(model, trainvecs, trainlabels, cv=5)

scores.mean()
```

## 表达数据依赖关系

通常一台 Jupyter 笔记本依赖于多个数据包。我们可以将`quilt.yml`中的数据依赖关系表达如下:

```py
packages:

  -  uciml/iris

  -  asah/mnist

  -  akarve/pydata_book/titanic:tag:features
```

精神上`quilt.yml`和`requirements.txt`很像，但是对于数据来说。使用`quilt.yml`的结果是，你的代码库仍然小而快。`quilt.yml`伴随您的 Jupyter 笔记本文件，以便任何想要复制您的笔记本的人可以在终端中键入被子安装并开始工作。

## 结论

我们展示了 Quilt 如何与 Domino 的 Reproducibility Engine 协同工作，以使 Jupyter 笔记本变得便携，并为机器学习提供[可复制性。](/blog/reproducible-data-science)[棉被的社区版](https://quiltdata.com/)由开源核心驱动。欢迎代码贡献者。