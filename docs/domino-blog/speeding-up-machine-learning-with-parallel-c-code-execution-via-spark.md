# 通过 Spark 的并行 C/C++代码执行加速机器学习

> 原文：<https://www.dominodatalab.com/blog/speeding-up-machine-learning-with-parallel-c-code-execution-via-spark>

C 编程语言是在 50 多年前引入的，从那以后，它一直占据着最常用编程语言的位置。随着 1985 年 C++扩展的引入以及类和对象的增加，C/C++对在所有主要操作系统、数据库和一般性能关键应用程序的开发中保持了中心地位。由于其效率，C/C++支持大量的机器学习库(如 TensorFlow、Caffe、CNTK)和广泛使用的工具(如 MATLAB、SAS)。当想到机器学习和大数据时，C++可能不是第一个想到的东西，但它在需要闪电般快速计算的领域无处不在——从谷歌的 Bigtable 和 GFS 到几乎所有与 GPU 相关的东西(CUDA、OCCA、openCL 等)。)

不幸的是，当谈到并行数据处理时，C/C++并不倾向于提供开箱即用的解决方案。这就是大规模数据处理之王 Apache Spark 的用武之地。

## 通过 RDD.pipe 调用 C/C++

根据其 [文档](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.RDD.pipe.html),*管道*操作符使 Spark 能够使用外部应用程序处理 RDD 数据。*管道*的使用不是 C/C++特有的，它可以用来调用用任意语言编写的外部代码(包括 shell 脚本)。在内部，Spark 使用一种特殊类型的 RDD[piped rdd](https://spark.apache.org/docs/0.7.3/api/core/spark/rdd/PipedRDD.html)，通过指定的外部程序来传输每个数据分区的内容。

让我们看一个简单的例子。下面是一个基本的 C 程序，它从标准输入(stdin)中读取并鹦鹉学舌般地返回字符串，前缀为“Hello”。该程序存储在一个名为 *hello.c* 的文件中

```py
/* hello.c */

#include <stdio.h>

#include <stdlib.h>

int main(int argc, char *argv[]) {

  char *buffer = NULL;

  int read;

  size_t len;

  while (1) {

    read = getline(&buffer, &len, stdin);

    if (-1 != read)

      printf("Hello, %s", buffer);

    else

      break;

  }

free(buffer);

return 0;
}

```

让我们编译并测试它。请注意，您可以按 Ctrl+D 来中断执行。

```py
$ gcc -o hello hello.c

$ ./hello 

Dave

Hello, Dave

Kiko
Hello, Kiko

Dirk
Hello, Dirk

$
```

到目前为止一切顺利。现在让我们看看如何将这段代码传送到 Spark RDD。首先，在我们尝试实际执行编译后的 C 代码之前，我们需要确保两件事情已经就绪:

*   访问 Spark 集群——这是一个显而易见的先决条件，在 Domino MLOps 平台中构建按需 Spark 集群是一个简单的操作[](https://docs.dominodatalab.com/en/4.3.1/reference/spark/on_demand_spark/On_demand_spark_overview.html)，只需点击几下鼠标即可完成。或者，你可以在 [独立模式](https://spark.apache.org/docs/latest/spark-standalone.html) 下运行 Spark。
*   我们计划使用的外部代码应该对所有负责运行应用程序的 Spark 执行者可用。如果我们使用的是 Domino 按需 Spark，所有执行者都默认访问一个 [共享数据集](https://docs.dominodatalab.com/en/5.0.1/reference/data/data_in_domino/datasets.html) 。我们要做的就是把编译好的 C/C++程序放到数据集的文件夹里。或者，我们可以修改驱动程序以包含对[Spark context . addfile()](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.SparkContext.addFile.html)的调用，这将强制 Spark 在每个节点上下载作业引用的外部代码。我们将展示如何处理上述两种情况。

假设我们使用的是 Domino，我们现在可以将编译好的 hello 程序放到一个数据集中。如果我们项目的名称是 SparkCPP，那么默认情况下，我们可以从工作区访问位于/domino/datasets/local/spark CPP/的数据集。让我们复制那里的 hello 程序。

```py
$ cp hello /domino/datasets/local/SparkCPP/

```

接下来，我们创建主程序 test_hello.py，我们将使用它来测试管道。这个 PySpark 应用程序将简单地并行化一个 RDD，并直接传输位于数据集内部的外部代码。

```py
from pyspark.sql import SparkSession

spark = SparkSession \

        .builder \

        .appName("MyAppName") \

        .getOrCreate()

# Suppress logging beyond this point so it is easier to inspect the output

spark.sparkContext.setLogLevel("ERROR")

rdd = spark.sparkContext.parallelize(["James","Cliff","Kirk","Robert", "Jason", "Dave"])

rdd = rdd.pipe("/domino/datasets/local/SparkCPP/hello")

for item in rdd.collect():

    print(item)

spark.stop()

```

对于远程集群(即独立且不在 Domino 中运行的集群)，我们可以修改代码来显式分发 hello 程序。注意，管道命令的访问路径发生了变化，因为 hello 代码现在位于每个执行器的工作目录中。我们还假设编译后的 C 程序的绝对路径是/mnt/hello。

```py
from pyspark.sql import SparkSession

spark = SparkSession \

        .builder \

        .appName("HelloApp") \

        .getOrCreate()

spark.sparkContext.addFile("/mnt/hello")

rdd = spark.sparkContext.parallelize(["James","Cliff","Kirk","Robert", "Jason", "Dave"])

rdd = rdd.pipe("./hello")

for item in rdd.collect():

    print(item)

spark.stop()

```

通过 spark-submit 运行程序在两种情况下都会产生相同的输出:

```py
$ spark-submit test_hello.py

2022-02-10 09:55:02,365 INFO spark.SparkContext: Running Spark version 3.0.0

…

Hello, James

Hello, Cliff

Hello, Kirk

Hello, Robert

Hello, Jason

Hello, Dave

$ 
```

我们看到我们的 C 程序正确地连接了 RDD 的所有条目。此外，这个过程是在所有可用的工作人员上并行执行的。

## 使用编译的库和 UDF

现在让我们来看一些更高级的东西。我们将看到如何使用一个编译过的库(共享库或共享对象的集合，如果我们想使用 Linux 术语的话)并将其中的一个函数映射到一个 Spark 用户定义函数(UDF)。

我们首先在一个名为 *fact.c* 的文件中放置一个简单的 C 函数，它计算一个数 n 的阶乘:

```py
/* fact.c */

int fact(int n) {

  int f=1;

  if (n>1) {

    for(int i=1;i<=num;i++)

        f=f*i;

   }

  return f;

}
```

上面的代码非常简单明了。然而，我们将把它编译成一个。所以库并定义了一个事实()的 UDF 包装器。在我们这样做之前，我们需要考虑有哪些选项可以帮助我们实现 Python 和 C/C++的接口。

*   SWIG(Simplified Wrapper and Interface Generator)——一个成熟的框架，将 C 和 C++编写的程序与各种高级编程语言连接起来，如 Perl、Python、Javascript、Tcl、Octave、R 和 [许多其他的](http://www.swig.org/compat.html#SupportedLanguages) 。SWIG 最初开发于 1995 年，是一个成熟而稳定的包，惟一的缺点是它的灵活性是有代价的——它很复杂，需要一些额外的预编译步骤。值得注意的是，因为 SWIG 或多或少与语言无关，所以可以用它将 Python 与框架支持的任何语言进行接口。它自动包装整个库的能力(假设我们可以访问头文件)也非常方便。
*   CFFI (C 对外函数接口)——根据它的 [文档](https://cffi.readthedocs.io/en/latest/overview.html#overview) 来看，使用 CFFI 的主要方式是作为一些已经编译好的共享对象的接口，这些共享对象是通过其他方式提供的。该框架通过动态运行时接口增强了 Python，使其易于使用和集成。然而，运行时开销会导致性能下降，因此它落后于静态编译的代码。
*   Cython——Python 的超集，cy thon 是一种为速度而设计的编译语言，它为 Python 提供了额外的受 C 启发的语法。与 SWIG 相比，Cython 的学习曲线不太陡峭，因为它自动将 Python 代码翻译成 C，所以程序员不必用低级语言实现逻辑来获得性能提升。不利的一面是，Cython 在构建时需要一个额外的库，并且需要单独安装，这使得部署过程变得复杂。

当然，还有其他选择(Boost。Python 和 Pybindgen 跃入脑海)，但是对每个 C/C++接口工具的全面概述超出了本文的范围。现在，我们将提供一个使用 SWIG 的例子，这或多或少是我们的最爱，因为它的成熟和多功能性。

在 Linux 上安装 SWIG 再简单不过了——它归结为一个简单的 *sudo 来安装 swig* 命令。如果你想看其他操作系统的安装过程，请随意看它的 [文档](http://www.swig.org/Doc4.0/SWIGDocumentation.html) 。还有一个简单的 [教程](http://www.swig.org/tutorial.html) 提供了一个 10 分钟的“入门”介绍，对于想要熟悉使用框架的人来说是一个简单的方法。

假设我们已经安装了 SWIG，构建共享库所需的第一步是为它创建一个接口。这个文件作为 SWIG 的输入。我们将把这个文件命名为我们的 *fact()* 函数 fact.i (i 代表接口),并将函数定义放在里面，如下所示:

```py
 /* fact.i */

 %module fact

 %{

 extern int fact(int n);

 %}

 extern int fact(int n);

```

然后我们运行 SWIG 来构建一个 Python 模块。

```py
$ swig -python fact.i
```

我们看到 SWIG 生成了两个新文件:

*   fact.py(基于接口文件中的模块名)是一个我们可以直接导入的 Python 模块
*   fact_wrap.c 文件，它应该被编译并与外部代码的其余部分链接

```py
$ ls -lah

total 132K

drwxrwxr-x 2 ubuntu ubuntu 4.0K Feb 10 14:32 .

drwxr-xr-x 6 ubuntu ubuntu 4.0K Feb 10 11:13 ..

-rw-rw-r-- 1 ubuntu ubuntu  110 Feb 10 11:20 fact.c

-rw-rw-r-- 1 ubuntu ubuntu   55 Feb 10 14:32 fact.i

-rw-rw-r-- 1 ubuntu ubuntu 3.0K Feb 10 14:32 fact.py

-rw-rw-r-- 1 ubuntu ubuntu 112K Feb 10 14:32 fact_wrap.c

$ 
```

接下来，我们使用 GCC 来编译函数和包装器:

```py
$ gcc -fpic -I /usr/include/python3.6m -c fact.c fact_wrap.c

$
```

最后，我们将目标文件捆绑到一个名为 _fact.so 的共享库中:

```py
$ gcc -shared fact.o fact_wrap.o -o _fact.so

$ ls -lah

total 240K

drwxrwxr-x 2 ubuntu ubuntu 4.0K Feb 10 14:36 .

drwxr-xr-x 6 ubuntu ubuntu 4.0K Feb 10 11:13 ..

-rw-rw-r-- 1 ubuntu ubuntu  108 Feb 10 14:33 fact.c

-rw-rw-r-- 1 ubuntu ubuntu   89 Feb 10 14:35 fact.i

-rw-rw-r-- 1 ubuntu ubuntu 1.3K Feb 10 14:35 fact.o

-rw-rw-r-- 1 ubuntu ubuntu 3.0K Feb 10 14:35 fact.py

-rwxrwxr-x 1 ubuntu ubuntu  51K Feb 10 14:36 _fact.so

-rw-rw-r-- 1 ubuntu ubuntu 112K Feb 10 14:35 fact_wrap.c

-rw-rw-r-- 1 ubuntu ubuntu  50K Feb 10 14:35 fact_wrap.o

$
```

我们现在需要将共享库放在一个所有 worker 节点都可以访问的地方。选项大致相同——共享文件系统(例如 Domino dataset)或在执行期间通过 SparkContext.addFile()添加库。

这一次，我们将使用共享数据集。不要忘记将编译后的库和 Python 包装器类都复制到共享文件系统中。

```py
$ mkdir /domino/datasets/local/SparkCPP/fact_library
$ cp _fact.so /domino/datasets/local/SparkCPP/fact_library/
$ cp fact.py /domino/datasets/local/SparkCPP/fact_library/
```

这是我们将用来测试外部库调用的 Spark 应用程序。

```py
import sys

from pyspark.sql import SparkSession

from pyspark.sql.functions import udf

from pyspark.sql.types import IntegerType

def calculate_fact(val):

    sys.path.append("/domino/datasets/local/SparkCPP/fact_library")

    import fact

    return fact.fact(val)

factorial = udf(lambda x: calculate_fact(x))

spark = SparkSession \

        .builder \

        .appName("FactApp") \

        .getOrCreate()

df = spark.createDataFrame([1, 2, 3, 4, 5], IntegerType()).toDF("Value")

df.withColumn("Factorial", \

              factorial(df.Value
)) \
              factorial(df.Value)) \

  .show(truncate=False)

spark.stop()
```

这里有几件事需要注意。首先，我们使用 *calculate_fact* ()作为用户定义的函数(UDF)。UDF 是用户可编程的例程，作用于单个 RDD 行，是 SparkSQL 最有用的特性之一。它们允许我们扩展 SparkSQL 的标准功能，并促进代码重用。在上面的代码中，我们使用 org.apache.spark.sql.functions 包中的 *udf* ()，它采用一个 Python 函数并返回一个 UserDefinedFunction 对象。然而，上面的 Python 函数是对我们的共享库和底层 C 实现的调用。第二件要注意的事情是，我们使用了 *sys.path.append* ()，它告诉 Python 添加一个特定的路径，以便解释器在加载模块时进行搜索。我们这样做是为了迫使 python 寻找位于/domino/datasets/local/spark CPP/fact _ library 下的事实模块。注意， *append* ()和 *import* 调用是 *calculate_fact* ()函数的一部分。我们这样做是因为我们希望在所有工作节点上附加路径和导入模块。如果我们将这些与脚本顶部的其他导入捆绑在一起，导入将只发生在 Spark 驱动程序级别，作业将会失败。

下面是运行上面的 Spark 应用程序的输出。：

```py
$ spark-submit test_fact.py 

2022-02-11 16:25:51,615 INFO spark.SparkContext: Running Spark version 3.0.0

…

2022-02-11 16:26:07,605 INFO codegen.CodeGenerator: Code generated in 17.433013 ms

+-----+---------+

|Value|Factorial|

+-----+---------+

|1    |1        |

|2    |2        |

|3    |6        |

|4    |24       |

|5    |120      |

+-----+---------+

…

2022-02-11 16:26:08,663 INFO util.ShutdownHookManager: Deleting directory /tmp/spark-cf80e57a-d1d0-492f-8f6d-31d806856915

$ 
```

## 
总结

在本文中，我们研究了如何通过访问外部 C/C++库提供的闪电般的操作来增强 Spark 这个无可争议的大数据处理之王。注意，这种方法并不仅限于 C 和 C++，因为 SWIG 等框架提供了对许多其他语言的访问，如 Lua、Go 和 Scilab。其他项目如 [gfort2py](https://github.com/rjfarmer/gfort2py) 也提供了对 Fortran 的访问。

此外，人们可以使用 NVIDIA CUDA 来编写、编译和运行调用 CPU 功能和启动 GPU 内核的 C/C++程序。然后可以通过 Spark 经由共享库访问这些代码，提供并行化和 GPU 加速的自定义操作。

#### 额外资源

*   Domino Enterprise MLOps 平台提供了对按需 Spark 计算的灵活访问。它提供了直接在云或支持 Domino 实例的本地基础设施上动态配置和编排 Spark 集群的能力。它还可以直接运行 NVIDIA 的 [NGC 目录](https://catalog.ngc.nvidia.com/containers) 中的容器。注册并了解该平台如何帮助您加快组织中的模型速度。
*   了解更多关于 [Spark、Ray 和 Dask](https://blog.dominodatalab.com/spark-dask-ray-choosing-the-right-framework)T2 的信息，以及如何为您的机器学习管道选择正确的框架。