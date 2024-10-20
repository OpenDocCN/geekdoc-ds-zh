# 如何使用 Python Docstrings 进行有效的代码文档编制

> 原文：<https://www.dataquest.io/blog/documenting-in-python-with-docstrings/>

August 15, 2022![Documenting in Python with docstrings](img/c718b31526a4b3d944afe661a0db97ad.png)

## 对任何数据科学家或软件工程师来说，记录代码都是一项关键技能。了解如何使用 docstrings 来实现这一点。

## 为什么 Python 中的文档很重要？

Python 的[禅告诉我们“可读性很重要”，“显式比隐式好。”这些都是 Python 的必要特征。当我们写代码时，我们是为了最终用户、开发者和我们自己。](https://peps.python.org/pep-0020/#the-zen-of-python)

请记住，当我们阅读 [`pandas`](https://pandas.pydata.org/docs/) 文档或 [`scikit-learn`](https://scikit-learn.org/0.18/documentation.html) 文档时，我们也是最终用户。这两个软件包都有很好的文档，用户使用它们通常不会有任何问题，因为它们包含了大量的示例和教程。它们还有内置文档，我们可以在您的首选 IDE 中直接访问。

现在想象一下在没有任何参考的情况下使用这些包。我们需要深入研究他们的代码，以了解它做什么，以及我们如何使用它。有一些软件包完全没有文档，通常要花更多的时间来理解它们下面是什么以及我们如何使用它们。如果包很大，函数分布在多个文件中，那么工作量会增加一倍或两倍。

为什么好的文档很重要现在应该更明显了。

接下来，其他开发人员可能想要为我们的项目做出贡献。如果我们的代码有良好的文档记录，他们可以做得更快更有效。如果你不得不花几个小时弄清楚一个项目的不同部分是做什么的，你会愿意在空闲时间为这个项目做贡献吗？我不会。

最后，即使只是我们的私人项目或一个小脚本，我们无论如何都需要文档。我们永远不知道什么时候我们会回到我们的老项目中去修改或改进它。如果它明确地告诉我们它的用途以及它的功能、代码行和模块是做什么的，那么使用它会非常愉快和容易。在编写代码时，我们往往会很快忘记我们在想什么，所以花一些时间解释我们为什么要做一些事情，在返回代码时总是会节省更多的时间(即使它是一天前才编写的)。所以，花些时间在文档上，它会在以后回报你。

## Python 文档字符串示例

让我们来看一些 Python 中文档字符串的例子！例如，下面的普通函数采用两个变量，或者返回它们的和(默认情况下),或者返回它们之间的差:

```py
def sum_subtract(a, b, operation="sum"):
    if operation == "sum":
        return a + b
    elif operation == "subtract":
        return a - b
    else:
        print("Incorrect operation.")

print(sum_subtract(1, 2, operation="sum"))
```

```py
 3
```

这个函数非常简单，但是为了展示 Python docstrings 的强大功能，让我们编写一些文档:

```py
def sum_subtract(a, b, operation="sum"):
    """
    Return sum or difference between the numbers 'a' and 'b'.
    The type of operation is defined by the 'operation' argument.
    If the operation is not supported, print 'Incorrect operation.'
    """
    if operation == "sum":
        return a + b
    elif operation == "subtract":
        return a - b
    else:
        print("Incorrect operation.")
```

这个函数的 Python docstring 用两边的三个双引号括起来。正如您所看到的，这个字符串解释了这个函数的作用，并指出我们可以如何改变它的功能，以及如果它不支持我们希望它执行的操作会发生什么。这是一个简单的例子，您可能会认为这个函数太明显了，不需要任何解释，但是一旦函数变得复杂并且数量增加，您至少需要一些文档来避免迷失在自己的代码中。

引擎盖下发生了什么？如果我们在`sum_subtract()`函数上运行`help()`函数，就会弹出 docstring。所有文档完整的包都有(几乎)所有函数的 docstrings。例如，让我们看看熊猫的数据帧:

```py
import pandas as pd
```

```py
help(pd.DataFrame)

# Help on class DataFrame in module pandas.core.frame:

# class DataFrame(pandas.core.generic.NDFrame, pandas.core.arraylike.OpsMixin)
#  |  DataFrame(data=None, index: 'Axes | None' = None, columns: 'Axes | None' = None, dtype: 'Dtype | None' = None, copy: 'bool | None' = None)
#  |
#  |  Two-dimensional, size-mutable, potentially heterogeneous tabular data.
#  |
#  |  Data structure also contains labeled axes (rows and columns).
#  |  Arithmetic operations align on both row and column labels. Can be
#  |  thought of as a dict-like container for Series objects. The primary
#  |  pandas data structure.
#  |
#  |  Parameters
#  |  ----------
#  |  data : ndarray (structured or homogeneous), Iterable, dict, or DataFrame
#  |      Dict can contain Series, arrays, constants, dataclass or list-like objects. If
#  |      data is a dict, column order follows insertion-order. If a dict contains Series
#  |      which have an index defined, it is aligned by its index.
#  |
#  |      .. versionchanged:: 0.25.0
#  |         If data is a list of dicts, column order follows insertion-order.
#  |
#  |  index : Index or array-like
#  |      Index to use for resulting frame. Will default to RangeIndex if
#  |      no indexing information part of input data and no index provided.
#  |  columns : Index or array-like
#  |      Column labels to use for resulting frame when data does not have them,
#  |      defaulting to RangeIndex(0, 1, 2, ..., n). If data contains column labels,
#  |      will perform column selection instead.
#  |  dtype : dtype, default None
#  |      Data type to force. Only a single dtype is allowed. If None, infer.
#  |  copy : bool or None, default None
#  |      Copy data from inputs.
#  |      For dict data, the default of None behaves like ``copy=True``.  For DataFrame
#  |      or 2d ndarray input, the default of None behaves like ``copy=False``.
#  |
#  |      .. versionchanged:: 1.3.0

# Docstring continues...
```

现在，如果我们看一下 [`pandas`源代码](https://github.com/pandas-dev/pandas/blob/main/pandas/core/frame.py)，就会发现`help`向我们展示了`DataFrame`类的 docstring(搜索`class DataFrame(NDFrame, OpsMixin)`)。

从技术上讲，docstring 被分配给这个对象的一个自动生成的属性，称为`__doc__`。我们还可以打印出该属性，并看到它与之前完全相同:

```py
print(pd.DataFrame.__doc__[:1570])  # Truncated
```

```py
 Two-dimensional, size-mutable, potentially heterogeneous tabular data.

    Data structure also contains labeled axes (rows and columns).
    Arithmetic operations align on both row and column labels. Can be
    thought of as a dict-like container for Series objects. The primary
    pandas data structure.

    Parameters
    ----------
    data : ndarray (structured or homogeneous), Iterable, dict, or DataFrame
        Dict can contain Series, arrays, constants, dataclass or list-like objects. If
        data is a dict, column order follows insertion-order. If a dict contains Series
        which have an index defined, it is aligned by its index.

        .. versionchanged:: 0.25.0
           If data is a list of dicts, column order follows insertion-order.

    index : Index or array-like
        Index to use for resulting frame. Will default to RangeIndex if
        no indexing information part of input data and no index provided.
    columns : Index or array-like
        Column labels to use for resulting frame when data does not have them,
        defaulting to RangeIndex(0, 1, 2, ..., n). If data contains column labels,
        will perform column selection instead.
    dtype : dtype, default None
        Data type to force. Only a single dtype is allowed. If None, infer.
    copy : bool or None, default None
        Copy data from inputs.
        For dict data, the default of None behaves like ``copy=True``.  For DataFrame
        or 2d ndarray input, the default of None behaves like ``copy=False``.

        .. versionchanged:: 1.3.0
```

## 文档字符串和代码注释之间的区别

在我们了解了 docstrings 的样子之后，是时候了解它们与常规代码注释的区别了。主要思想是它们(通常)服务于不同的目的。

文档字符串解释了一个函数/类的用途(例如，它的描述、参数和输出——以及任何其他有用的信息)，而注释解释了特定代码字符串的作用。换句话说，代码注释是给想要修改代码的人用的，文档字符串是给想要使用代码的人用的。

此外，在计划代码时，代码注释可能是有用的(例如，通过实现[伪代码](https://en.wikipedia.org/wiki/Pseudocode)，或者留下你的想法或想法的临时注释，这些想法或想法不是为最终用户准备的)。

## 文档字符串格式

让我们看看不同类型的文档字符串。首先，浏览 Python 的 PEP 页面是一个极好的主意。我们可以找到 [PEP 257](https://peps.python.org/pep-0257/) ，里面总结了 Python docstrings。我强烈建议你通读，即使你可能并不完全理解。要点如下:

1.  使用三个双引号将文档字符串括起来。
2.  Docstring 以点结束。
3.  它应该描述函数的命令(即函数做什么，所以我们通常以“Return…”开始短语)。
4.  如果我们需要添加更多的信息(例如，关于参数)，那么我们应该在函数/类的概要和更详细的描述之间留一个空行(关于它的更多信息将在本教程后面给出)。
5.  如果能增加可读性，多行文档字符串也不错。
6.  应该有参数、输出、异常等的描述。

请注意，这些只是建议，并不是严格的规则。例如，我怀疑我们是否需要向`sum_subtract()`函数添加更多关于参数或输出的信息，因为摘要已经足够描述了。

有四种主要类型的文档字符串，它们都遵循上述建议:

1.  [NumPy/SciPy docstrings](https://numpydoc.readthedocs.io/en/latest/format.html)
2.  [谷歌文档字符串](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings)
3.  [重组文本](https://docutils.sourceforge.io/rst.html)
4.  [Epytext](http://epydoc.sourceforge.net/epytext.html)

你可以看看所有的，然后决定你最喜欢哪一个。尽管 reStructuredText 是官方的 Python 文档样式，但 NumPy/SciPy 和 Google docstrings 会更频繁地出现。

让我们来看一个真实世界的数据集，并编写一个函数来应用于它的一个列。

[这个数据集](https://www.kaggle.com/datasets/deepcontractor/top-video-games-19952021-metacritic)包含了 1995-2021 年 [Metacritic](https://www.metacritic.com/) 上的顶级视频游戏。

```py
from datetime import datetime
import pandas as pd

all_games = pd.read_csv("all_games.csv")
print(all_games.head())
```

```py
 name        platform        release_date  \
0  The Legend of Zelda: Ocarina of Time     Nintendo 64   November 23, 1998   
1              Tony Hawk's Pro Skater 2     PlayStation  September 20, 2000   
2                   Grand Theft Auto IV   PlayStation 3      April 29, 2008   
3                           SoulCalibur       Dreamcast   September 8, 1999   
4                   Grand Theft Auto IV        Xbox 360      April 29, 2008   

                                             summary  meta_score user_review  
0  As a young boy, Link is tricked by Ganondorf, ...          99         9.1  
1  As most major publishers' development efforts ...          98         7.4  
2  [Metacritic's 2008 PS3 Game of the Year; Also ...          98         7.7  
3  This is a tale of souls and swords, transcendi...          98         8.4  
4  [Metacritic's 2008 Xbox 360 Game of the Year; ...          98         7.9 
```

例如，我们可能对创建一个列感兴趣，该列计算一个游戏在多少**天**前发布。为此，我们还将使用`datetime`包([教程一](https://www.dataquest.io/blog/python-datetime/)、[教程二](https://www.dataquest.io/blog/python-datetime-tutorial/))。

首先，让我们编写这个函数并测试它:

```py
def days_release(date):
    current_date = datetime.now()
    release_date_dt = datetime.strptime(date, "%B %d, %Y") # Convert date string into datetime object
    return (current_date - release_date_dt).days

print(all_games["release_date"].apply(days_release))
```

```py
0        8626
1        7959
2        5181
3        8337
4        5181
         ... 
18795    3333
18796    6820
18797    2479
18798    3551
18799    4845
Name: release_date, Length: 18800, dtype: int64
```

该函数如预期的那样工作，但是最好用 docstrings 来补充它。首先，让我们使用 **SciPy/NumPy 格式**:

```py
def days_release(date):
    """
    Return the difference in days between the current date and game release date.

    Parameter
    ---------
    date : str
        Release date in string format.

    Returns
    -------
    int64
        Integer difference in days.
    """
    current_date = datetime.now()
    release_date_dt = datetime.strptime(date, "%B %d, %Y") # Convert date string into datetime object
    return (current_date - release_date_dt).days
```

在上面的`Returns`部分，我没有重复(大部分)已经在总结部分的内容。

接下来，一个 **Google docstrings** 的例子:

```py
def days_release(date):
    """Return the difference in days between the current date and game release date.

    Args:
        date (str): Release date in string format.

    Returns:
        int64: Integer difference in days.
    """
    current_date = datetime.now()
    release_date_dt = datetime.strptime(date, "%B %d, %Y") # Convert date string into datetime object
    return (current_date - release_date_dt).days
```

一个**重组文本**的例子:

```py
def days_release(date):
    """Return the difference in days between the current date and game release date.

    :param date: Release date in string format.
    :type date : str
    :returns: Integer difference in days.
    :rtype: int64
    """
    current_date = datetime.now()
    release_date_dt = datetime.strptime(date, "%B %d, %Y") # Convert date string into datetime object
    return (current_date - release_date_dt).days
```

最后，一个 **Epytext** 的例子:

```py
def days_release(date):
    """Return the difference in days between the current date and the game release date.
    @type date: str
    @param date: Release date in string format.
    @rtype: int64
    @returns: Integer difference in days.
    """
    current_date = datetime.now()
    release_date_dt = datetime.strptime(date, "%B %d, %Y") # Convert date string into datetime object
    return (current_date - release_date_dt).days
```

大多数函数文档字符串至少应该有它们的功能、输入和输出描述的摘要。此外，如果它们更复杂，它们可能包括例子、注释、例外等等。

最后，我没有过多地讨论类，因为它们在数据科学中并不常见，但是除了它们的方法(函数)所包含的内容之外，它们还应该包含以下内容:

1.  属性描述
2.  方法列表及其功能摘要
3.  属性的默认值

## 编写文档字符串脚本

Python docstrings 还描述了小脚本的功能和使用指南，这些小脚本可能适用于我们一些日常任务的自动化。例如，我经常需要将丹麦克朗转换成欧元。我可以每次在谷歌上输入一个查询，或者我可以下载一个应用程序来帮我完成，但这一切似乎都过于复杂和多余。我知道一欧元大约是 7.45 丹麦克朗，因为我几乎总是在 Linux 终端上工作，所以我决定编写一个小的 CLI 程序，将一种货币转换成另一种货币:

```py
"""
DKK-EUR and EUR-DKK converter.

This script allows the conversion of Danish kroner to euros and euros to Danish kroner with a fixed exchange rate
of 7.45 Danish krone for one euro.

It is required to specify the conversion type: either dkk-eur or eur-dkk as well as the type of the input currency
(EUR or DKK).

This file contains the following functions:

    * dkk_converter - converts Danish kroner to euros
    * eur_converter - converts euros to Danish kroner
    * main - main function of the script
"""

import argparse

# Create the parser
parser = argparse.ArgumentParser(
    prog="DKK-EUR converter", description="Converts DKK to EUR and vice versa"
)

# Add arguments
parser.add_argument("--dkk", "-d", help="Danish kroner")
parser.add_argument("--eur", "-e", help="Euros")
parser.add_argument(
    "--type",
    "-t",
    help="Conversion type",
    choices=["dkk-eur", "eur-dkk"],
    required=True,
)

# Parse the arguments
args = parser.parse_args()

def dkk_converter(amount):
    """Convert Danish kroner to euros."""
    amount = float(amount)
    print(f"{amount} DKK is {round(amount / 7.45, 2)} EUR.")

def eur_converter(amount):
    """Convert euros to Danish kroner."""
    amount = float(amount)
    print(f"{amount} EUR is {round(amount * 7.45, 2)} DKK.")

def main():
    """Main function."""
    if args.type == "dkk-eur":
        dkk_converter(args.dkk)
    elif args.type == "eur-dkk":
        eur_converter(args.eur)
    else:
        print("Incorrect conversion type")

main()
```

该脚本应该有足够的文档记录，以允许用户应用它。在文件的顶部，docstring 应该描述脚本的主要目的、简要指南以及它包含的函数或类。此外，如果使用任何第三方包，应该在 docstrings 中声明，以便用户在使用脚本之前安装它。

如果您使用`argparse`模块创建 CLI 应用程序，那么它的每个参数都应该在`help`菜单中描述，以便最终用户可以选择`-h`选项并确定输入。此外，`description`参数应该填入`ArgumentParser`对象中。

最后，如前所述，所有功能都应正确记录。这里，我省略了“参数”和“返回”描述符，因为从我的角度来看，它们是不必要的。

另外，请注意我是如何使用代码注释的，以及它们与 docstrings 有何不同。如果你有一个自己写的脚本，添加一些文档来练习。

## 关于如何编写 Python 文档字符串和文档的一般建议

为什么文档很重要，为什么文档字符串是 Python 文档的重要组成部分，这应该很清楚了。最后，让我给你一些关于 Python 文档字符串和文档的一般性建议。

1.  为人们而不是计算机写文档。它应该是描述性的，但简洁明了。
2.  不要过度使用 docstrings。有时它们是不必要的，一个小的代码注释就足够了(甚至根本没有注释)。假设开发人员有一些 Python 的基础知识。
3.  文档字符串不应该解释函数如何工作，而是应该解释如何使用它。有时，可能有必要解释一段代码的内部机制，因此使用代码注释。
4.  不要用文档字符串代替注释，也不要用注释代替代码。

## 摘要

以下是我们在本教程中学到的内容:

1.  文档是 Python 项目的重要组成部分——它对最终用户、开发人员和您都很重要。
2.  文档字符串是给使用代码的的**，注释是给修改代码的**的**。**
3.  [PEP 257](https://peps.python.org/pep-0257/) 总结 Python 文档字符串。
4.  有四种主要的 docstring 格式:[NumPy/SciPy docstring](https://numpydoc.readthedocs.io/en/latest/format.html)、[Google docstring](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings)、 [reStructuredText](https://docutils.sourceforge.io/rst.html) 和 [Epytext](http://epydoc.sourceforge.net/epytext.html) 。前两种是最常见的。
5.  脚本文档字符串应该描述脚本的功能、用法以及其中包含的功能。

我希望你今天学到了一些新东西。请随时在 LinkedIn 或 T2 GitHub 上与我联系。编码快乐！