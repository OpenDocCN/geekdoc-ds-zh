# 让 PySpark 与 spaCy 一起工作:克服序列化错误

> 原文：<https://www.dominodatalab.com/blog/making-pyspark-work-spacy-overcoming-serialization-errors>

*在这篇客座博文中， [Holden Karau](https://twitter.com/holdenkarau) ， [Apache Spark Committer](https://spark.apache.org/committers.html) ，提供了关于如何使用 [spaCy](https://spacy.io/) 处理文本数据的见解。 [Karau](https://twitter.com/holdenkarau) 是谷歌的[开发者拥护者](http://bit.ly/holdenGKE)，也是“[高性能火花](https://www.amazon.com/High-Performance-Spark-Practices-Optimizing/dp/1491943203/)”和“[学习火花](https://www.amazon.com/Learning-Spark-Lightning-Fast-Data-Analysis/dp/1449358624/)”的合著者。她在 [Twitch](https://www.twitch.tv/holdenkarau) 和 [YouTube](https://www.youtube.com/user/holdenkarau) 上有一个关于她的演讲、代码评审和代码会议的知识库。她还在从事分布式计算和儿童 T21 的工作。*

## spaCy 能帮上什么忙？

你是一名数据科学家，需要用 Python 并行化你的 NLP 代码，但却不断遇到问题吗？这篇博文展示了如何克服使用流行的 NLP 库空间时出现的序列化困难。(虽然这些技术有点复杂，但是您可以将它们隐藏在一个单独的文件中，假装一切正常。)这篇文章关注的是 spaCy 的使用，我还有另外一篇关于 NLTK 的文章，我将在我的博客上发表。如果你是 Scala 或 Java 用户，请尽快寻找 JVM 上的帖子。

不过，在你过于兴奋之前，有一个警告——如果你认为调试 Apache Spark 很难，那么调试这些序列化技巧会更难，所以你应该看看[我的调试 Spark 视频](https://www.youtube.com/watch?v=s5p15QT0Zj8)，并关注 Safari 上的深度课程，当它可用时。

## 当然是字数统计

现在，如果我们不把重点放在字数上，这就不是一篇关于大数据的博文，但是我们要做一点改变。就像我的第一份工作一样，在某些时候，老板(或客户或其他影响你支付租金能力的人)可能会找到你，要求你“本地化”你激动人心的 WordCount**项目。在这一点上，如果你从小到大只说英语和法语，你可能会回答，“那很好；所有语言都使用空格来分隔单词”，但是您可能会很快发现 split 函数对于像日语这样的语言来说并不是一个通用的标记器。

在意识到标记化其他语言实际上是多么复杂之后，我们可能会开始对我们承诺的两周交付时间感到紧张，但幸运的是，标记化是 NLP 工具的一个基本部分，许多现有的库可以在多种人类(非计算机)语言上工作。

## 常规(非箭头)Python UDFs

我们将要使用的 Python 库是 [spaCy](https://www.dominodatalab.com/data-science-dictionary/spacy) ，这是一个世界级的自然语言处理工具。虽然你不需要了解 spaCy 来理解这篇博文，但是如果你想了解更多关于 spaCy 的知识，这里有一个[的精彩文档集](https://spacy.io/usage/)。

首先，我们将弄清楚如何在 PySpark 内部使用 spaCy，而不用担心跨语言管道的细节。在 PySpark 的旧版本中，用户注册了 UDF，如:

```py
def spacy_tokenize(x):
# Note this is expensive, in practice you would use something like SpacyMagic, see footnote for link; which caches
# spacy.load so it isn’t happening multiple times
    nlp = spacy.load(lang)
# If you are working with Python 2 and getting regular strings add x = unicode(x)
    doc = nlp(text)
    return [token.text for token in doc]

tokenize = session.udf.register("tokenize", spacy_tokenize)
```

这给了我们一个可以在 Python 中调用的函数，它将使用 spaCy 来标记输入，尽管是用英语，因为我真的不懂其他任何东西。以标准的 [Spark SQL 字数](https://spark.apache.org/examples.html)为例，我们可以修改它以避免 Spark RDDs:

```py
df = spark.read.format("text").load("README.md")

tokenized = df.select(explode(split(df.value, " ")))

result = tokenized.groupBy(tokenized.col).count()

result.show() # Or save
```

然后，我们可以换入新函数，并使用 spaCy 进行标记化:

```py
tokenized = df.select(explode(spacy_tokenize(df.value, ' ')))
```

如果我们运行这个，它会因为几个不同的原因而变得相当慢。首先，spaCy.load 是一个昂贵的调用；在我自己的系统上，导入和加载空间几乎只需要一秒钟。第二个原因是在 Java 和 Python 之间来回复制数据的序列化开销。

## 火花之箭 UDFs

Spark 的基于箭头的 UDF 通常更快，原因有几个。Apache Arrow 的核心为我们提供了一种 JVM 和 Python、[以及许多其他语言](https://github.com/apache/arrow)都能理解的格式，并且以一种有利于矢量化操作的方式进行组织。在 Spark 中创建基于箭头的 UDF 需要一点重构，因为我们操作的是批处理而不是单个记录。新的基于箭头的 PySpark 矢量化 UDF 可以注册如下:

```py
@pandas_udf("integer", PandasUDFType.SCALAR) &nbsp;# doctest: +SKIP
def pandas_tokenize(x):
    return x.apply(spacy_tokenize)
tokenize_pandas = session.udf.register("tokenize_pandas", pandas_tokenize)
```

如果您的集群还没有为基于 Arrow 的 PySpark UDFs(有时也称为 Pandas UDFs)设置，那么您需要确保您的所有机器上都安装了 Spark 2.3+和 PyArrow 的匹配版本。(查看 setup.py，了解您的 Spark 版本所需的版本)。

## 使用 PySpark UDFs(常规和箭头)

两者的用法看起来很相似。

### 在 SQL 中:

```py
spark.sql("SELECT tokenize(str_column) FROM db")
spark.sql("SELECT tokenize_pandas(str_column) FROM db")
```

### 使用 Dataframe API:

```py
dataframe.select(tokenize(dataframe.str_column))
dataframe.select(tokenize_pandas(dataframe.str_column))
```

我们当然可以用它们来做 Python 中的(强制)字数统计示例:

```py
dataframe.select(explode(tokenize_pandas(dataframe.str_column)).alias("tokens")).groupBy("tokens").sum("*").collect()
```

虽然这两个看起来非常相似，但它们实际上可能具有非常不同的性能含义，因此我们自然会关注第二个更快的示例，在本例中是熊猫/箭头驱动的示例。

## 处处超越“spacy.load()”

序列化问题是 PySpark 面临的最大性能挑战之一。如果你试图通过将``spacy.load()``移到函数调用之外来优化它，spaCy 会尝试自己序列化，这可能会很大，包括 [cdefs](https://cython.readthedocs.io/en/latest/src/tutorial/cdef_classes.html) 。Cdefs 不能被 pickle 序列化，尽管通过一些小心的包装，我们仍然可以使用依赖于它们的代码。这甚至可能不起作用，而是通过使用一个全局变量(对不起！)和一个包装函数，我们可以确保重用空间:

```py
# spaCy isn't serializable but loading it is semi-expensive
NLP = None
def get_spacy_magic_for(lang):
    global NLP
    if NLP is None:
        NLP = {}
    if lang not in NLP:
        NLP[lang] = spacy.load(lang)
    return NLP[lang]
```

然后在我们的代码中，我们通过我们的朋友``get_spacy_magic``访问空间。如果您在常规文件中工作，而不是在笔记本/REPL 中工作，您可以使用更干净的基于类的方法，但是由于深奥的序列化原因，在带有 PySpark 的 REPL 中使用类有一些问题。

由于这段代码不够漂亮，您可能会问自己减少负载有多重要。举个例子，在我的 X1 Carbon 上加载 en 语言大约需要一秒钟，如果每个元素增加一秒钟的开销，我们就很容易失去并行化这个工作负载的好处。

spaCy load prefork 在 spaCy load 方面有一些新的有趣的技巧，但这是另一篇博客文章的主题。(请再次关注我的[博客](http://blog.holdenkarau.com/) / [媒体](https://medium.com/@holdenkarau) / [推特](https://twitter.com/holdenkarau)，我会在那里分享我的博客。)

## 包扎

这种方法对于 WordCount 来说已经足够好了(我的意思是，哪个大数据系统不行呢？)，但它仍然让我们缺少一些想要的信息。例如，在这种情况下和未来的 NLTK 文章中，Python 中收集的信息比我们目前在标量转换中可以轻松返回的信息多得多，但在 [SPARK-21187](https://issues.apache.org/jira/browse/SPARK-21187) 中围绕这一点的工作仍在继续。如果您尝试直接返回 spaCy 文档，您将遇到序列化问题，因为它引用了 cloud pickle 不知道如何处理的内存块。(参见[云泡菜#182](https://github.com/cloudpipe/cloudpickle/issues/182) 了解一些背景)。

如果这对你来说是令人兴奋的，并且你想有所贡献，那么非常欢迎你加入我们的 [Sparkling ML 项目](https://github.com/sparklingpandas/sparklingml)，Apache Arrow 或通用改进版 [Apache Spark Python 集成](http://spark.apache.org/)。此外，如果你有兴趣看到这篇文章的幕后，你可以看看这个[现场编码会议](https://www.youtube.com/watch?v=EPvd5BhhevM&list=PLRLebp9QyZtYF46jlSnIu2x1NDBkKa2uw)和相应的[闪闪发光的 ml 回购](https://github.com/sparklingpandas/sparklingml)。

*^(作者注:SpacyMagic link 这里的[是](https://github.com/sparklingpandas/sparklingml/blob/91bed86546943d683ba4a1fc5ae3a2fef7e2175e/sparklingml/transformation_functions.py#L80)。)*