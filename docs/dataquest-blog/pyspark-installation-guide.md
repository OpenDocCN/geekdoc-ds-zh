# 教程:用 Jupyter 笔记本安装和集成 PySpark

> 原文：<https://www.dataquest.io/blog/pyspark-installation-guide/>

October 26, 2015At Dataquest, we’ve released an [interactive course on Spark](https://www.dataquest.io/course/spark-map-reduce/), with a focus on PySpark. We explore the fundamentals of Map-Reduce and how to utilize PySpark to clean, transform, and munge data. In this post, we’ll dive into how to install PySpark locally on your own computer and how to integrate it into the Jupyter Notebbok workflow. Some familarity with the command line will be necessary to complete the installation.

## 概述

概括地说，以下是安装 PySpark 并将其与 Jupyter notebook 集成的步骤:

1.  安装下面所需的软件包
2.  下载并构建 Spark
3.  设置您的环境变量
4.  为 PySpark 创建一个 Jupyter 配置文件

## 必需的包

*   Java SE 开发工具包
*   Scala 构建工具
*   Spark 1.5.1(撰写本文时)
*   Python 2.6 或更高版本(我们更喜欢使用 Python 3.4+)
*   Jupyter 笔记型电脑

## Java

Spark 需要 Java 7+，可以从 Oracle 网站下载:

*   [Mac 下载链接](https://www.java.com/en/download/faq/java_mac.xml)
*   [Linux 下载链接](https://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html)

## 火花

前往

[Spark 下载页面](https://spark.apache.org/downloads.html)，保留步骤 1 到 3 中的默认选项，下载压缩版(。tgz 文件)的链接。一旦你下载了 Spark，我们建议解压文件夹，并将解压后的文件夹移动到你的主目录。

## Scala 构建工具

1.  要构建 Spark，您需要 Scala 构建工具，您可以安装:

*   Mac: `brew install sbt`
*   Linux: [指令](https://www.scala-sbt.org/release/tutorial/Installing-sbt-on-Linux.html)

2.  导航到 Spark 解压到的目录，并在该目录中运行`sbt assembly`(这需要一段时间！).

## 测试

为了测试 Spark 构建是否正确，在同一个文件夹(Spark 所在的位置)中运行以下命令:

```py
bin/pyspark
```

交互式 PySpark shell 应该会启动。这是交互式 PySpark shell，类似于 Jupyter，但是如果你运行

在 shell 中，您将看到已经初始化的 SparkContext 对象。您可以在这个 shell 中交互地编写和运行命令，就像使用 Jupyter 一样。

## 环境变量

环境变量是计算机上的任何程序都可以访问的全局变量，包含您希望所有程序都可以访问的特定设置和信息。在我们的例子中，我们需要指定 Spark 的位置，并添加一些特殊的参数，我们将在后面引用。使用

[`nano`](https://askubuntu.com/questions/54221/how-to-edit-files-in-a-terminal-with-nano) 或`vim`打开`~/.bash_profile`并在末尾添加以下几行:

```py
export SPARK_HOME="$HOME/spark-1.5.1"
export PYSPARK_SUBMIT_ARGS="--master local[2]"
```

替换

将 Spark 解压缩到的文件夹的位置(还要确保版本号匹配！).

## Jupyter 简介

最后一步是专门为 PySpark 创建一个 Jupyter 概要文件，并进行一些自定义设置。要创建此配置文件，请运行:

```py
Jupyter profile create pyspark
```

使用

`nano`或`vim`在以下位置创建以下 Python 脚本:

```py
~/.jupyter/profile_pyspark/startup/00-pyspark-setup.py
```

然后向其中添加以下内容:

```py
 import os
import sys

spark_home = os.environ.get('SPARK_HOME', None)
sys.path.insert(0, spark_home + "/python")
sys.path.insert(0, os.path.join(spark_home, 'python/lib/py4j-0.8.2.1-src.zip'))

filename = os.path.join(spark_home, 'python/pyspark/shell.py')
exec(compile(open(filename, "rb").read(), filename, 'exec'))

spark_release_file = spark_home + "/RELEASE"

if os.path.exists(spark_release_file) and "Spark 1.5" in open(spark_release_file).read():
    pyspark_submit_args = os.environ.get("PYSPARK_SUBMIT_ARGS", "")
    if not "pyspark-shell" in pyspark_submit_args: 
        pyspark_submit_args += " pyspark-shell"
        os.environ["PYSPARK_SUBMIT_ARGS"] = pyspark_submit_args 
```

如果您使用的是比 Spark 1.5 更高的版本，请在脚本中将“Spark 1.5”替换为您正在使用的版本。

## 运行

使用启动 Jupyter Notebook

`pyspark`配置文件，运行:

```py
jupyter notebook --profile=pyspark
```

要测试 PySpark 是否正确加载，创建一个新的笔记本并运行

以确保 SparkContext 对象被正确初始化。

## 接下来的步骤

如果您想更详细地了解 spark，您可以访问我们的

Dataquest 上的[互动火花课程](https://www.dataquest.io/course/spark-map-reduce/)。