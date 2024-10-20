# 教程:用 Python 清理 MoMA 艺术收藏的数据

> 原文：<https://www.dataquest.io/blog/data-cleaning-with-python/>

August 13, 2015![moma-art-data-science-python-tutorial](img/155fd6a20ae905f3431c8b6f551f42ad.png)Art is a messy business. Over centuries, artists have created everything from simple paintings to complex sculptures, and art historians have been cataloging everything they can along the way. The Museum of Modern Art, or MoMA for short, is considered one of the most influential museums in the world and recently released a dataset of all the artworks they’ve cataloged in their collection. This dataset contains basic information on metadata for each artwork and is part of MoMA’s push to make art more accessible to everyone. The museum has put out a disclaimer, however, that the dataset is still a work in progress – an evolving artwork in its own right perhaps. Because it’s still in progress, the dataset has data quality issues and needs some cleanup before we can analyze it. In this post, we’ll introduce a few tools for cleaning data (or munging data, as it’s sometimes called) in Python. and discuss how we can use them to diagnose data quality issues and correct them. You can download the dataset for yourself on [Github](https://github.com/MuseumofModernArt/collection).

## 为什么使用 Python 进行数据清理？

Python 是一种编程语言，从政府机构(如 SEC)到互联网公司(如 Dropbox)，各种机构都用它来创建功能强大的软件。Python 已经成为大量处理数据的组织中用于数据分析的主要语言。虽然我们不会在这篇文章中深入太多关于 Python 如何工作的细节，但是如果你有兴趣了解更多，我们强烈推荐你阅读[学习 Python](https://www.dataquest.io/course/python-for-data-science-fundamentals) 系列。

## 用熊猫探索数据

在这篇文章中，我们将只处理数据集的前 100 行。我们首先需要将 Pandas 库导入到我们的环境中，然后将数据集读入 Pandas 数据帧。DataFrame 是我们数据集的速度优化表示，内置于 Pandas 中，我们可以使用它来快速浏览和分析我们的数据。一旦我们将数据集读入 DataFrame 对象`artworks_100`，我们将:

```py
 import pandas
artworks_100 = pandas.read_csv("MOMA_Artworks.csv")
artworks_100.head(10) 
```

|  | 标题 | 艺术家 | 艺术家简历 | 日期 | 中等 | 规模 | 作者姓名或来源附注 | MoMANumber | 分类 | 部门 | 获得日期 | 经监管机构批准 | ObjectID | 统一资源定位器 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Zero | 费迪南德布罗斯项目，奥地利维也纳 | 奥图·华格纳 | (奥地利人，1841 年至 1918 年) | One thousand eight hundred and ninety-six | 墨水和剪贴在纸上的画页 | 19 1/8 x 66 1/2 英寸(48.6 x 168.9 厘米) | 部分和承诺的礼物乔卡罗尔和… | 885.1996 | A&D 建筑制图 | 建筑与设计 | 1996-04-09 | Y | Two | https://www.moma.org/collection/works/2 |
| one | 音乐之城，国家高级音乐学院… | 克里斯蒂安·德·波扎姆帕克 | (法国人，生于 1944 年) | One thousand nine hundred and eighty-seven | 油漆和彩色铅笔打印 | 16 x 11 3/4 英寸(40.6 x 29.8 厘米) | 建筑师向莉莉·奥金奇致敬的礼物… | 1.1995 | A&D 建筑制图 | 建筑与设计 | 1995-01-17 | Y | three | https://www.moma.org/collection/works/3 |
| Two | 澳大利亚维也纳郊外维也纳项目附近的别墅… | 埃米尔·霍普 | (奥地利人，1876 年至 1957 年) | One thousand nine hundred and three | 石墨、钢笔、彩色铅笔、墨水和水粉… | 13 1/2 x 12 1/2 英寸(34.3 x 31.8 厘米) | 乔·卡罗尔和罗纳德·s·劳德的礼物 | 1.1997 | A&D 建筑制图 | 建筑与设计 | 1997-01-15 | Y | four | https://www.moma.org/collection/works/4 |
| three | 纽约曼哈顿抄本项目… | 伯纳德·屈米 | (法国人和瑞士人，1944 年生于瑞士) | One thousand nine hundred and eighty | 彩色合成照相复制 | 20 x 20 英寸(50.8 x 50.8 厘米) | 购买和部分赠送的建筑师在… | 2.1995 | A&D 建筑制图 | 建筑与设计 | 1995-01-17 | Y | five | https://www.moma.org/collection/works/5 |
| four | 奥地利维也纳郊外的别墅项目 | 埃米尔·霍普 | (奥地利人，1876 年至 1957 年) | One thousand nine hundred and three | 石墨，彩色铅笔，墨水，水粉在 tr… | 15 1/8 x 7 1/2 英寸(38.4 x 19.1 厘米) | 乔·卡罗尔和罗纳德·s·劳德的礼物 | 2.1997 | A&D 建筑制图 | 建筑与设计 | 1997-01-15 | Y | six | https://www.moma.org/collection/works/6 |
| five | 纽约曼哈顿抄本项目… | 伯纳德·屈米 | (法国人和瑞士人，1944 年生于瑞士) | 1976-77 | 明胶银照片 | 14 x 18 英寸(35.6 x 45.7 厘米) | 购买和部分赠送的建筑师在… | 3.1995.1 | A&D 建筑制图 | 建筑与设计 | 1995-01-17 | Y | seven | https://www.moma.org/collection/works/7 |
| six | 纽约曼哈顿抄本项目… | 伯纳德·屈米 | (法国人和瑞士人，1944 年生于瑞士) | 1976-77 | 明胶银照片 | 每个:14 x 18 英寸(35.6 x 45.7 厘米) | 购买和部分赠送的建筑师在… | 3.1995.1-24 | A&D 建筑制图 | 建筑与设计 | 1995-01-17 | Y | eight | https://www.moma.org/collection/works/8 |
| seven | 纽约曼哈顿抄本项目… | 伯纳德·屈米 | (法国人和瑞士人，1944 年生于瑞士) | 1976-77 | 明胶银照片 | 14 x 18 英寸(35.6 x 45.7 厘米) | 购买和部分赠送的建筑师在… | 3.1995.10 | A&D 建筑制图 | 建筑与设计 | 1995-01-17 | Y | nine | https://www.moma.org/collection/works/9 |
| eight | 纽约曼哈顿抄本项目… | 伯纳德·屈米 | (法国人和瑞士人，1944 年生于瑞士) | 1976-77 | 明胶银照片 | 14 x 18 英寸(35.6 x 45.7 厘米) | 购买和部分赠送的建筑师在… | 3.1995.11 | A&D 建筑制图 | 建筑与设计 | 1995-01-17 | Y | Ten | https://www.moma.org/collection/works/10 |
| nine | 纽约曼哈顿抄本项目… | 伯纳德·屈米 | (法国人和瑞士人，1944 年生于瑞士) | 1976-77 | 明胶银照片 | 14 x 18 英寸(35.6 x 45.7 厘米) | 购买和部分赠送的建筑师在… | 3.1995.12 | A&D 建筑制图 | 建筑与设计 | 1995-01-17 | Y | Eleven | https://www.moma.org/collection/works/11 |

## 在熊猫中处理日期

如果仔细观察`Date`列，您可能会注意到一些值是年份范围(`1976-77`)，而不是个别年份(`1976`)。年份范围很难处理，我们不能像绘制单个年份那样简单地绘制它们。让我们利用熊猫的`value_counts()`功能来寻找任何其他奇怪之处。首先，要选择一个列，使用括号符号并在引号中指定列名`artworks_100['Date']`，然后附加`.value_counts()`以使用`artworks_100['Date'].value_counts()`获得值的分布。

```py
artworks_100['Date'].value_counts()
```

```py
1976-77    25
1980-81    15
1979       12
Unknown     7
1980        5
1917        5
1978        5
1923        4
1935        3
1987        2
1903        2
1970        1
1896        1
1975        1
1984        1
1918        1
1986        1
n.d.        1
1906        1
1905        1
1930        1
1974        1
1936        1
1968        1
1900        1
c. 1917     1
dtype: int64
```

## 清理数据模式

除了年份范围之外，我们还需要注意其他三种模式。下面是对`Date`列中不规则值类型的快速总结:

*   `Pattern 1`:“1976-77”(年份范围)
*   `Pattern 2`:“约 1917 年”
*   `Pattern 3`:“未知”
*   `Pattern 4`:“未注明。”

如果我们想出处理每个模式的规则，我们可以用 Python 写出逻辑，并将所有不规则的值转换成适当的格式。一旦我们写出了我们的逻辑，我们就可以逐行迭代 DataFrame，并在必要时更改`Date`列的值。虽然模式 1 或 2 的行有日期值，但对我们来说格式很差，而模式 3 和 4 的行实际上没有关于艺术品制作时间的信息。因此，我们处理模式 1 和 2 的代码应该着重于将值重新格式化为干净的日期，而我们处理模式 3 和 4 的代码应该只将那些列标识为缺少日期信息。为了简单起见，我们可以保持模式 3 的行不变(如`"Unknown"`)，并将模式 4 的行转换为模式 3。

## 模式 1

因为模式 1 的所有行都是仅跨越两年的年份范围(例如`1980-81`)，所以我们可以选择一年并让它替换该范围。为了简单起见，让我们选择范围中的第一年，因为它包含所有四位数字(`1980`)，而范围中的第二年只有最后两位数字(`81`)。我们还需要一种可靠的方法来识别哪些行实际上表现出模式 1，所以我们只更新那些行，而让其他行保持不变。我们需要让其他模式保持不变，要么是因为它们已经是正确的日期格式，要么是因为它们需要在以后使用我们为处理其他模式而编写的逻辑进行修改。

由于年份范围包含一个分隔两年的连字符`-`，我们可以在每一行的`Date`值中寻找`-`并将它分成两个独立的年份。核心 Python 库包含一个名为`.split()`的函数，在这种情况下，如果找到连字符，它将返回两年的列表，如果没有找到，则返回原始值。因为我们只寻找第一年，我们可以在每一行的`Date`上调用`.split("-")`，检查结果列表是否包含两个元素，如果包含，则返回第一个元素。让我们写一个函数`clean_split_dates(row)`来完成这个任务:

```py
 def clean_split_dates(row):
    # Initial date contains the current value for the Date column
    initial_date = str(row['Date'])
    # Split initial_date into two elements if "-" is found
    split_date = initial_date.split('-')
     # If a "-"  is found, split_date will contain a list with at least two items
    if len(split_date) > 1:
        final_date = split_date[0]
    # If no "-" is found, split_date will just contain 1 item, the initial_date
    else:
        final_date = initial_date
    return final_date
# Assign the results of "clean_split_dates" to the 'Date' column. 
# We want Pandas to go row-wise so we set "axis=1". We would use "axis=0" for column-wise.
artworks_100['Date'] = artworks.apply(lambda row: clean_split_dates(row), axis=1)
artworks_100['Date'].value_counts()
```

```py
 1976       25
1980       20
1979       12
Unknown     7
1917        5
1978        5
1923        4
1935        3
1987        2
1903        2
c. 1917     1
1918        1
1975        1
1968        1
n.d.        1
1905        1
1896        1
1984        1
1930        1
1970        1
1974        1
1986        1
1936        1
1900        1
1906        1
dtype: int64 
```

## 使用 Python 清理数据:后续步骤

我们在最后的`Date`列上运行`.value_counts()`,以验证所有年份范围都已从数据框架中删除。如果您有兴趣了解更多关于数据清理的信息，请查看 Dataquest 网站上我们的交互式[数据清理课程](https://www.dataquest.io/course/data-exploration)。这个六部分的课程使用 Python 和 pandas 库来教你如何清理和处理数据。该课程包括两个指导项目，帮助你综合你的技能，并开始一个[数据科学作品集](https://www.dataquest.io/blog/data-science-portfolio-project/)，你可以用它来向雇主展示你的技能。最棒的是，无需安装！该课程完全在一个交互式环境中教授，您可以编写自己的代码，并立即在浏览器中看到结果。