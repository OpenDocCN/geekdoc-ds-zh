# 教程:使用 Python 脚本和命令行转换数据

> 原文：<https://www.dataquest.io/blog/tutorial-python-scripts-command-line/>

October 1, 2019![python-scripts-command-line-hackernews](img/1154fbff27d44f04ed33d0150aa1d692.png)

在本教程中，我们将深入探讨如何使用 Python 脚本和命令行来转换数据。

但是首先，有必要问一个你可能会想的问题:“Python 如何适应命令行，当我知道我可以使用 IPython 笔记本或 Jupyter lab 完成所有数据科学工作时，我为什么还要使用命令行与 Python 交互？”

笔记本是快速数据可视化和探索的好工具，但是 Python 脚本是将我们学到的任何东西投入生产的方法。假设你想做一个网站，帮助人们用理想的标题和 sublesson 时间制作黑客新闻帖子。为此，您需要脚本。

本教程假设对函数有基本的了解，一点命令行经验也不会有什么坏处。如果你以前没有使用过 Python，请随意查看我们的课程，包括 Python 函数的基础知识，或者深入学习我们的 T2 数据科学课程。最近，我们发布了两个新的交互式命令行课程:[命令行元素](https://www.dataquest.io/course/command-line-elements)和[命令行中的文本处理](https://www.dataquest.io/course/text-processing-cli/)，所以如果你想更深入地了解命令行，我们也推荐这些课程

也就是说，不要太担心先决条件！我们会边走边解释我们正在做的一切，所以让我们开始吧！

## 熟悉数据

[黑客新闻](https://news.ycombinator.com/)是一个网站，用户可以提交来自互联网的文章(通常是关于技术和创业公司的)，其他人可以“投票支持”这些文章，表明他们喜欢这些文章。一个分包商获得的支持票越多，在社区里就越受欢迎。热门文章会出现在黑客新闻的“首页”，在那里它们更有可能被其他人看到。

这就引出了一个问题:是什么让一篇黑客新闻文章获得成功？我们能找到什么样的模式最有可能被投票支持吗？

我们将使用的数据集是 Arnaud Drizard 使用黑客新闻 API 编译的，可以在这里找到。我们从数据中随机抽取了 10000 行，并删除了所有无关的列。我们的数据集只有四列:

*   `sublesson_time` —故事提交的时间。
*   `upvotes` —分包商获得的支持票数。
*   `url` —分包商的基本域。
*   `headline`——子标题。用户可以编辑它，并且它不必与原始文章的标题相匹配。

我们将编写脚本来回答三个关键问题:

*   什么词最常出现在标题中？
*   黑客新闻最常提交哪些域名？
*   大多数文章是在什么时候提交的？

记住:对于编程来说，完成一项任务有多种方法。在本教程中，我们将通过一种方法来解决这些问题，但肯定还有其他方法可能同样有效，所以请随意尝试，并尝试提出自己的方法！

## 使用命令行和 Python 脚本读入数据

首先，让我们在桌面上创建一个名为`Transforming_Data_with_Python`的文件夹。要使用命令行创建文件夹，您可以使用`mkdir`命令，后跟文件夹的名称。例如，如果你想创建一个名为`test`的文件夹，你可以导航到桌面目录，然后输入`mkdir test`。

稍后我们将讨论为什么创建一个文件夹，但是现在，让我们使用`cd`命令导航到创建的文件夹。`cd`命令允许我们使用命令行改变目录。

虽然使用命令行创建文件有多种方法，但是我们可以利用一种称为[管道和重定向输出](https://app.dataquest.io/m/391)的技术来同时完成两件事:将来自 stdout 的输出(命令行生成的标准输出)重定向到一个文件中，并创建一个新文件！换句话说，我们可以让它创建一个新文件，并让它输出该文件的内容，而不是让命令行只打印它的输出。

为此，我们可以使用`>`和`>>`，这取决于我们想要对文件做什么。如果文件不存在，两者都会创建一个文件；然而，`>`会用重定向的输出覆盖文件中已经存在的文本，而`>>`会将任何重定向的输出附加到文件中。

我们希望将数据读入这个文件，并创建一个描述性的文件名和函数名，所以我们将创建一个名为`load_data()`的函数，并将它放在名为`read.py`的文件中。让我们使用读入数据的命令行来创建我们的函数。为此，我们将使用`printf`函数。(我们将使用`printf`,因为它允许我们打印换行符和制表符，我们希望使用它们来使我们的脚本对我们自己和他人来说更具可读性)。

为此，我们可以在命令行中键入以下内容

```py
 printf "import pandas as pd\n\ndef load_data():\n\thn_stories = pd.read_csv('hn_stories.csv')\n\thn_stories.colummns = ['sublesson_time', 'upvotes', 'url', 'headline']\n\treturn(hn_stores)\n" > read.py 
```

检查上面的代码，有很多事情正在发生。让我们一点一点地分解它。在函数中，我们是:

*   记住，我们想让我们的脚本可读，我们使用`printf`命令来生成一些输出，使用命令行来保存我们生成输出时的格式。
*   进口熊猫。
*   将我们的数据集(`hn_stories.csv`)读入熊猫数据帧。
*   使用`df.columns`将列名添加到我们的数据框架中。
*   创建一个名为`load_data()`的函数，它包含读入和处理数据集的代码。
*   利用换行符(`\n`)和制表符(`\t`)来保存格式，以便 Python 可以读取脚本。
*   使用`>`操作符将`printf`的输出重定向到一个名为`read.py`的文件。因为`read.py`还不存在，所以创建了这个文件。

在我们运行上面的代码之后，我们可以在命令行中键入`cat read.py`并执行命令来检查`read.py`的内容。如果一切运行正常，我们的`read.py`文件将如下所示:

```py
 import pandas as pd

def load_data():
    hn_stories = pd.read_csv("hn_stories.csv")
    hn_stories.columns = ['sublesson_time', 'upvotes', 'url', 'headline']
    return(hn_stories) 
```

创建这个文件后，我们的目录结构应该如下所示

```py
Transforming_Data_with_Python
 | read.py
 | 
 | 
 |
 |
___ 
```

## 正在创建`__init__.py`

对于这个项目的其余部分，我们将创建更多的脚本来回答我们的问题，并使用`load_data()`函数。虽然我们可以将这个函数粘贴到每个使用它的文件中，但是如果我们正在处理的项目很大，这可能会变得很麻烦。

为了解决这个问题，我们可以创建一个名为`__init__.py`的文件。本质上，`__init__.py`允许文件夹将它们的目录文件视为包。最简单的形式，`__init__.py`可以是一个空文件。它必须存在，目录文件才能被视为包。你可以在 [Python 文档](https://docs.python.org/3/tutorial/modules.html)中找到更多关于包和模块的信息。

因为`load_data()`是`read.py`中的一个函数，我们可以像导入包一样导入那个函数:`from read import load_data()`。

还记得使用命令行创建文件有多种方法吗？我们可以使用另一个命令来创建`__init__.py`,这一次，我们将使用`touch`命令来创建文件。`touch`是一个命令，当您运行该命令时，它会为您创建一个空文件:

```py
 touch __init__.py 
```

创建这个文件后，目录结构将如下所示

```py
Transforming_Data_with_Python
 | __init__.py
 | read.py
 | 
 |
 |
___ 
```

## 探索标题中的单词

现在我们已经创建了一个脚本来读入和处理数据，并创建了`__init__.py`，我们可以开始分析数据了！我们首先要探究的是出现在标题中的独特词汇。为此，我们希望做到以下几点:

*   使用命令行创建一个名为`count.py`的文件。
*   从`read.py`导入`load_data`，调用函数读入数据集。
*   将所有标题组合成一个长字符串。当你组合标题时，我们希望在每个标题之间留有空间。对于这一步，我们将使用 [`Series.str.cat`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.cat.html) 来加入字符串。
*   把那一长串拆分成单词。
*   使用[计数器](https://docs.python.org/3/library/collections.html#collections.Counter)类来计算每个单词在字符串中出现的次数。
*   使用`.most_common()`方法将 100 个最常用的单词存储到`wordCount`。

如果您使用命令行创建该文件，它看起来是这样的:

```py
 printf "from read import load_data\nfrom collections import Counter\n\nstories = load_data()\nheadlines = stories['headline'].str.cat(sep = ' ').lower()\nwordCount = Counter(headlines.split(' ')).most_common(100)\nprint(wordCount)\n" > count.py 
```

运行上面的代码后，可以在命令行中键入`cat count.py`并执行命令来检查`count.py`的内容。如果一切运行正常，您的`count.py`文件将如下所示:

```py
 from read import load_data
from collections import Counter

stories = load_data()
headlines = stories["headline"].str.cat(sep = ' ').lower()
wordCount = Counter(headlines.split(' ')).most_common(100)
print(wordCount) 
```

该目录现在应该是这样的:

```py
Transforming_Data_with_Python
 | __init__.py
 | read.py
 | count.py
 |
 |
___ 
```

现在我们已经创建了 Python 脚本，我们可以从命令行运行我们的脚本来获得一百个最常见单词的列表。为了运行这个脚本，我们从命令行输入命令`python count.py`。

脚本运行后，您将看到打印的结果:

```py
 [('the', 2045), ('to', 1641), ('a', 1276), ('of', 1170), ('for', 1140), ('in', 1036), ('and', 936), ('', 733), ('is', 620), ('on', 568), ('hn:', 537), ('with', 537), ('how', 526), ('-', 487), ('your', 480), ('you', 392), ('ask', 371), ('from', 310), ('new', 304), ('google', 303), ('why', 262), ('what', 258), ('an', 243), ('are', 223), ('by', 219), ('at', 213), ('show', 205), ('web', 192), ('it', 192), ('–', 184), ('do', 183), ('app', 178), ('i', 173), ('as', 161), ('not', 160), ('that', 160), ('data', 157), ('about', 154), ('be', 154), ('facebook', 150), ('startup', 147), ('my', 131), ('|', 127), ('using', 125), ('free', 125), ('online', 123), ('apple', 123), ('get', 122), ('can', 115), ('open', 114), ('will', 112), ('android', 110), ('this', 110), ('out', 109), ('we', 106), ('its', 102), ('now', 101), ('best', 101), ('up', 100), ('code', 98), ('have', 97), ('or', 96), ('one', 95), ('more', 93), ('first', 93), ('all', 93), ('software', 93), ('make', 92), ('iphone', 91), ('twitter', 91), ('should', 91), ('video', 90), ('social', 89), ('&', 88), ('internet', 88), ('us', 88), ('mobile', 88), ('use', 86), ('has', 84), ('just', 80), ('world', 79), ('design', 79), ('business', 79), ('5', 78), ('apps', 77), ('source', 77), ('cloud', 76), ('into', 76), ('api', 75), ('top', 74), ('tech', 73), ('javascript', 73), ('like', 72), ('programming', 72), ('windows', 72), ('when', 71), ('ios', 70), ('live', 69), ('future', 69), ('most', 68)] 
```

在我们的网站上滚动浏览它们有点笨拙，但你可能会注意到最常见的单词如`the`、`to`、`a`、`for`等。这些词被称为*停用词*——对人类语言有用但对数据分析没有任何帮助的词。你可以在我们关于空间的教程中找到更多关于停用词的信息。如果您想扩展这个项目，从我们的分析中删除停用词将是一个有趣的下一步。

尽管包含了停用词，我们还是可以发现一些趋势。除了停用词，这些词中的绝大多数都是与科技和创业相关的术语。鉴于 HackerNews 专注于科技初创公司，这并不令人惊讶，但我们可以看到一些有趣的具体趋势。例如，谷歌是这个数据集中最常被提及的品牌。脸书，苹果，推特是其他品牌的热门话题。

## 探索域子层次

现在，我们已经探索了不同的标题并显示了前 100 个最常用的单词，我们现在可以探索域的子部分了！为此，我们可以执行以下操作:

*   使用命令行创建一个名为`domains.py`的文件。
*   从`read.py`导入`load_data`，调用函数读入数据集。
*   在 pandas 中使用`value_counts()`方法来计算一列中每个值出现的次数。
*   遍历序列并打印索引值及其相关的总计。

下面是它在命令行中的样子:

```py
 printf "from read import load_data\n\nstories = load_data()\ndomains = stories['url'].value_counts()\nfor name, row in domains.items():\n\tprint('{0}: {1}'.format(name, row))\n" > domains.py 
```

同样，如果我们在命令行中键入`cat domains.py`来检查`domains.py`，我们应该会看到:

```py
 from read import load_data

stories = load_data()
domains = stories['url'].value_counts()
for name, row in domains.items():
    print('{0}: {1}'.format(name, row)) 
```

创建该文件后，我们的目录如下所示:

```py
Transforming_Data_with_Python
 | __init__.py
 | read.py
 | count.py
 | domains.py
 |
___ 
```

## 探索 Sublesson 时代

我们想知道大多数文章是什么时候提交的。一个简单的方法就是看看文章是在什么时候提交的。为了解决这个问题，我们需要使用`sublesson_time`列。

`sublesson_time`列包含如下所示的时间戳:`2011-11-09T21:56:22Z`。这些时间用 [UTC](https://en.wikipedia.org/wiki/Coordinated_Universal_Time) 表示，这是大多数软件为保持一致性而使用的通用时区(想象一个数据库中的时间都有不同的时区；工作起来会很痛苦)。

为了从时间戳中获取小时，我们可以使用`dateutil`库。`dateutil`中的`parser`模块包含解析函数，该函数可以接收时间戳，并返回一个`datetime`对象。这里是文档的链接。解析时间戳后，产生的 date 对象的`hour`属性将告诉您文章提交的时间。

为此，我们可以执行以下操作:

*   使用命令行创建一个名为`times.py`的文件。
*   写一个函数从时间戳中提取小时。这个函数应该首先使用`dateutil.parser.parse`解析时间戳，然后从结果`datetime`对象中提取小时，然后使用`.hour`返回小时。
*   使用 pandas `apply()`方法制作一列子小时数。
*   在 pandas 中使用`value_counts()`方法来计算每小时出现的次数。
*   打印出结果。

下面是我们在命令行中的操作方法:

```py
 printf "from dateutil.parser import parse\nfrom read import load_data\n\n\ndef extract_hour(timestamp):\n\tdatetime = parse(timestamp)\n\thour = datetime.hour\n\treturn hour\n\nstories = load_data()\nstories['hour'] = stories['sublesson_time'].apply(extract_hour)\ntime = stories['hour'].value_counts()\nprint(time)" > times.py 
```

下面是它作为一个单独的`.py`文件的样子(如上所述，您可以通过从命令行运行`cat times.py`检查该文件来确认):

```py
 from dateutil.parser import parse
from read import load_data

def extract_hour(timestamp):
    datetime = parse(timestamp)
    hour = datetime.hour
    return hour

nstories = load_data()
stories['hour'] = stories['sublesson_time'].apply(extract_hour)
time = stories['hour'].value_counts()
print(time) 
```

让我们再一次更新我们的目录:

```py
Transforming_Data_with_Python
 | __init__.py
 | read.py
 | count.py
 | domains.py
 | times.py
___ 
```

现在我们已经创建了 Python 脚本，我们可以从命令行运行我们的脚本来获取某个小时内发布的文章数量的列表。为此，您可以从命令行键入命令`python times.py`。运行该脚本，您将看到以下结果:

```py
 17    646
16    627
15    618
14    602
18    575
19    563
20    538
13    531
21    497
12    398
23    394
22    386
11    347
10    324
7     320
0     317
1     314
2     298
9     298
3     296
4     282
6     279
5     275
8     274 
```

你会注意到大多数转租是在下午公布的。但是，请记住，这些时间是用 UTC 表示的。如果您对扩展这个项目感兴趣，可以尝试在脚本中添加一个部分，将 UTC 的输出转换为您当地的时区。

## 后续步骤

在本教程中，我们探索了数据并构建了一个简短脚本的目录，这些脚本相互协作以提供我们想要的答案。这是构建我们的数据分析项目的生产版本的第一步。

当然，这仅仅是开始！在本教程中，我们没有使用任何 upvotes 数据，因此这将是扩展您的分析的一个很好的下一步:

*   什么样的标题长度能获得最多的支持？
*   什么时候投票最多？
*   随着时间的推移，向上投票的总数是如何变化的？

我们鼓励您思考自己的问题，并在继续探索该数据集时发挥创造力！

## 这个教程有帮助吗？

选择你的道路，不断学习有价值的数据技能。

![arrow down left](img/2215dd1efd21629477b52ea871afdd98.png)![arrow right down](img/2e703f405f987a154317ac045ee00a68.png)[Python Tutorials](/python-tutorials-for-data-science/)

通过我们的免费教程练习您的 Python 编程技能。

[Data science courses](/data-science-courses/)

通过我们的交互式浏览器数据科学课程，投入到 Python、R、SQL 等语言的学习中。