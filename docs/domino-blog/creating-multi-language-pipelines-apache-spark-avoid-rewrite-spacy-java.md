# 使用 Apache Spark 创建多语言管道或避免将 spaCy 重写为 Java

> 原文：<https://www.dominodatalab.com/blog/creating-multi-language-pipelines-apache-spark-avoid-rewrite-spacy-java>

*在这篇客座博文中， [Holden Karau](https://twitter.com/holdenkarau) ， [Apache Spark Committer](https://spark.apache.org/committers.html) ，提供了关于如何使用 Apache Spark 创建多语言管道以及避免将 [spaCy](https://www.dominodatalab.com/data-science-dictionary/spacy) 重写为 Java 的见解。她已经写了一篇关于使用 spaCy 为 Domino 处理文本数据的补充博客文章。 [Karau](https://twitter.com/holdenkarau) 是谷歌的开发者拥护者，也是[高性能火花](https://www.amazon.com/High-Performance-Spark-Practices-Optimizing/dp/1491943203/)和[学习火花](https://www.amazon.com/Learning-Spark-Lightning-Fast-Data-Analysis/dp/1449358624/)的合著者。她在 Twitch 和 Youtube 上也有一个关于她的演讲、代码评审和代码会议的知识库。*

## 介绍

作为一名 Scala/Java data/ML 工程师，您是否发现自己需要使用一些现有的 PySpark 代码，但不想重写所有使用过的库？或者您只是厌倦了 JVM 中的 NLP 选项，想尝试一下 spaCy？Apache Spark 新的基于 Apache Arrow 的 UDF 不仅提供了性能改进，还可以与实验特性相结合，以允许跨语言管道的开发。这篇博文关注的是 spaCy 的使用，我还有另外一篇关于 NLTK 的文章，我将会发布在[我的博客](http://blog.holdenkarau.com/)上。如果您希望在单语言管道中使用 spaCy 和 PySpark，那么我之前的文章《[让 PySpark 与 spaCy 一起工作:克服序列化错误](https://blog.dominodatalab.com/making-pyspark-work-spacy-overcoming-serialization-errors/)》会对您有所帮助

如果您选择在生产中使用这些技术，请预先注意在 Spark 中调试多语言管道会增加额外的层次和复杂性。你应该看看[霍尔登的调试 Spark 视频](https://www.youtube.com/watch?v=s5p15QT0Zj8)，如果你有公司费用账户，还有一个在 Safari 上提供的关于[调试 Apache Spark 的深度潜水。](https://www.oreilly.com/library/view/debugging-apache-spark/9781492039174/)

## 从 Scala / Java 调用和注册 Python UDFs

我们的第一步需要为 JVM 设置一种方法来调用 Python，并让 Python 将 UDF 注册到 Java 中。

在 startup.py 中，我们可以创建一个入口点，这样我们的 Java 代码就可以调用 Py4J。这个连接 Python 和 Scala 的样板有点烦人。如果你想看细节(或者破坏魔法)，你可以看看 [Initialize.scala](https://github.com/sparklingpandas/sparklingml/blob/master/src/main/scala/com/sparklingpandas/sparklingml/util/python/Initialize.scala) 和 [startup.py](https://github.com/sparklingpandas/sparklingml/blob/master/sparklingml/startup.py) 。其中的关键部分是一个注册提供者，它在 Scala & Python:

```py
# This class is used to allow the Scala process to call into Python

# It may not run in the same Python process as your regular Python

# shell if you are running PySpark normally.

class PythonRegistrationProvider(object):

  """Provide an entry point for Scala to call to register functions."""

  def __init__(self, gateway):

    self.gateway = gateway

    self._sc = None

    self._session = None

    self._count = 0

  def registerFunction(self, ssc, jsession, function_name, params):

    jvm = self.gateway.jvm

    # If we don't have a reference to a running SparkContext

    # Get the SparkContext from the provided SparkSession.

    if not self._sc:

        master = ssc.master()

        jsc = jvm.org.apache.spark.api.java.JavaSparkContext(ssc)

        jsparkConf = ssc.conf()

        sparkConf = SparkConf(_jconf=jsparkConf)

        self._sc = SparkContext(

           master=master,

           conf=sparkConf,

           gateway=self.gateway,

           jsc=jsc)

        self._session = SparkSession.builder.getOrCreate()

    if function_name in functions_info:

        function_info = functions_info[function_name]

        if params:

            evaledParams = ast.literal_eval(params)

        else:

          evaledParams = []

        func = function_info.func(*evaledParams)

        ret_type = function_info.returnType()

        self._count = self._count + 1

        registration_name = function_name + str(self._count)

        udf = UserDefinedFunction(func, ret_type, registration_name)

        # Configure non-standard evaluation types (e.g. Arrow)

        udf.evalType = function_info.evalType()

        judf = udf._judf

        return judf

    else:

       return None

    class Java:

       package = "com.sparklingpandas.sparklingml.util.python"

       className = "PythonRegisterationProvider"

       implements = [package + "." + className]
```

在 Scala 方面，它看起来像这样:

```py
/**
* Abstract trait to implement in Python to allow Scala to call in to perform
* registration.
*/
trait PythonRegisterationProvider {
// Takes a SparkContext, SparkSession, String, and String
// Returns UserDefinedPythonFunction but types + py4j :(
def registerFunction(
sc: SparkContext, session: Object,
functionName: Object, params: Object): Object
}
```

构建用户定义的函数是在基类 [PythonTransformer](https://github.com/sparklingpandas/sparklingml/blob/master/src/main/scala/com/sparklingpandas/sparklingml/util/python/PythonTransformer.scala) 中完成的。主调用通过以下方式完成:

```py
 // Call the registration provider from startup.py to get a Python UDF back.
   val pythonUdf = Option(registrationProvider.registerFunction(
     session.sparkContext,
     session,
     pythonFunctionName,
     miniSerializeParams()))
   val castUdf = pythonUdf.map(_.asInstanceOf[UserDefinedPythonFunction])
```

因为我们现在需要在 JVM 和 python 之间传递一个参数(例如，我们正在使用的语言)，所以包装器必须有逻辑来指向所需的 Python 函数和用于配置它的参数:

```py
final val lang = new Param[String](this, "lang", "language for tokenization")

/** @group getParam */
final def getLang: String = $(lang)

final def setLang(value: String): this.type = set(this.lang, value)

def this() = this(Identifiable.randomUID("SpacyTokenizePython"))

override val pythonFunctionName = "spaCytokenize"
override protected def outputDataType = ArrayType(StringType)
override protected def validateInputType(inputType: DataType): Unit = {
if (inputType != StringType) {
throw new IllegalArgumentException(
s"Expected input type StringType instead found ${inputType}")
}
}

override def copy(extra: ParamMap) = {
defaultCopy(extra)
}

def miniSerializeParams() = {
"[\"" + $(lang) + "\"]"
}
}
```

然后我们需要重构我们的 Python 代码，以便 Scala 代码可以很容易地调用这些参数。在 [SparklingML](https://github.com/sparklingpandas/sparklingml) 内部，我们有一个可以使用的基类，[scalarvectorizedtransformation function](https://github.com/sparklingpandas/sparklingml/blob/f120129d4a7c17147ce51953834ccef1e3f90b6a/sparklingml/transformation_functions.py#L37)，来处理一些样板文件，看起来像是:

```py
class SpacyTokenize(ScalarVectorizedTransformationFunction):

    @classmethod

    def setup(cls, sc, session, *args):

        pass

    @classmethod

    def func(cls, *args):

        lang = args[0]

        def inner(inputSeries):

            """Tokenize the inputString using spaCy for the provided language."""

            nlp = SpacyMagic.get(lang) # Optimization for spacy.load

            def tokenizeElem(elem):

                result_itr =  [token.text for token in nlp(elem)]

                return list(result_itr)

            return inputSeries.apply(tokenizeElem)

        return inner

    @classmethod

    def returnType(cls, *args):

        return ArrayType(StringType())

functions_info["spacytokenize"] = SpacyTokenize
```

## SpaCyMagic:处处超越` spacy.load()'

PySpark 面临的一大挑战是序列化问题，对于多语言管道，这一挑战几乎加倍。

```py
# Spacy isn't serializable but loading it is semi-expensive

@ignore_unicode_prefix

class SpacyMagic(object):

    """
Simple Spacy Magic to minimize loading time.

    >>> SpacyMagic.get("en")

    <spacy.lang.en.English ...
"""
    _spacys = {}

    @classmethod

    def get(cls, lang):

        if lang not in cls._spacys:

            import spacy

            try:

                try:

                    cls._spacys[lang] = spacy.load(lang)

                except Exception:

                    spacy.cli.download(lang)

                    cls._spacys[lang] = spacy.load(lang)

            except Exception as e:

                raise Exception("Failed to find or download language {0}: {1}"
.format(lang, e))
    return cls._spacys[lang]
```

然后在我们的代码中，我们通过我们的朋友 SpacyMagic 访问 spaCy。

spaCy load pre-fork 在 Spark 2.4 中有一些有趣的新技巧，但这是另一篇博客文章的主题。(再一次，请关注我的[博客](http://blog.holdenkarau.com/)，我将在那里分享它。)

## 用跨语言字数统计把它们联系在一起

如果我们不把重点放在字数上，这看起来就不像是一篇真正的大数据博客文章，但我们将通过使用 Scala 和 Python 的不同方式来完成它。在 Scala 中，我们可以使用我们创建的转换器对单词计数进行标记:

```py
val data = session.load(...)
val transformer = new SpacyTokenizePython()
transformer.setLang("en")
transformer.setInputCol("input")
transformer.setOutputCol("tokens")
val tokens = transformer.transform(data)
val counts = tokens.groupBy("tokens").count()
counts.write.format("json").save("...")
```

## 包扎

虽然这些技术对于像 WordCount 这样的简单跨语言管道已经足够好了，但是在构建更复杂的管道时，我们还必须考虑一些其他方面。例如，即使在我们的简单示例中，spaCy 除了标记之外还有更多信息，并且当前在我们的 Panda 的 UDF 中缺少复合类型的数组，这使得返回复杂的结果很困难。

如果你觉得这很有趣，希望你能加入我们，为闪亮的 ML 项目、Apache Arrow 或 Apache Spark Python 和 Arrow 集成做出贡献。

### 进一步阅读

如果你有兴趣看到这篇文章的幕后，你可以看看这个[现场编码会议](https://www.youtube.com/watch?v=EPvd5BhhevM&list=PLRLebp9QyZtYF46jlSnIu2x1NDBkKa2uw)和相应的[闪闪发光的 ml 回购](https://github.com/sparklingpandas/sparklingml)。如果你有兴趣向孩子们介绍分布式系统的神奇世界，请注册了解我的下一本书“[分布式计算 4 孩子](http://www.distributedcomputing4kids.com/)”。如果一切都着火了，这些调试资源( [free talk](https://www.youtube.com/watch?v=s5p15QT0Zj8) 和[subscription Safari deep-dive](https://www.oreilly.com/library/view/debugging-apache-spark/9781492039174/))至少应该能帮到你一点点。