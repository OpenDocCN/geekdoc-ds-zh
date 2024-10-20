# 熊猫备忘单—用于数据科学的 Python

> 原文：<https://www.dataquest.io/blog/pandas-cheat-sheet/>

March 4, 2020![](img/02a84fd526339d91307f931e386864ef.png)

如果你对用 Python 处理数据感兴趣，你几乎肯定会使用 pandas 库。但是，即使你已经学习了熊猫——也许是在我们的交互式熊猫课程中——也很容易忘记做某事的具体语法。这就是为什么我们创建了一个熊猫备忘单来帮助你容易地参考最常见的熊猫任务。

在我们开始阅读备忘单之前，值得一提的是，您不应该仅仅依赖于此。如果你还没有学过熊猫，我们强烈建议你完成我们的熊猫课程。这张小抄将帮助你快速找到并回忆起你已经知道的关于熊猫的事情；它不是为了从头开始教你熊猫而设计的！

不时查看官方的熊猫文档也是个好主意，即使你能在小抄里找到你需要的。阅读文档是每个数据专业人员都需要的技能，文档涉及的细节比我们在一张纸上所能描述的要多得多！

如果你想用熊猫来完成一个特定的任务，我们也推荐你查看一下我们的免费 Python 教程 的完整列表**[；他们中的许多人除了使用其他 Python 库之外，还使用 pandas。例如，在我们的](https://www.dataquest.io/python-tutorials-for-data-science/) [Python 日期时间教程](https://www.dataquest.io/blog/python-datetime-tutorial/)中，您还将学习如何在 pandas 中处理日期和时间。**

## 熊猫小抄:指南

首先，给这个页面加个书签可能是个好主意，当你要找某个特定的东西时，用 Ctrl+F 可以很容易地搜索到它。然而，我们也制作了一份 PDF 版本的备忘单，如果你想打印出来，你可以从这里下载。

在这个备忘单中，**我们将使用下面的简写方式**:

`df` |任意熊猫数据帧对象`s` |任意熊猫系列对象

向下滚动时，您会看到我们已经使用副标题组织了相关命令，以便您可以根据您要完成的任务快速搜索并找到正确的语法。

另外，快速提醒—要使用下面列出的命令，您需要首先导入相关的库，如下所示:

```py
import pandas as pd
import numpy as np
```

## 导入数据

使用这些命令从各种不同的源和格式导入数据。

`pd.read_csv(filename)` |从 CSV 文件`pd.read_table(filename)` |从分隔文本文件(如 TSV) `pd.read_excel(filename)` |从 Excel 文件`pd.read_sql(query, connection_object)` |从 SQL 表/数据库`pd.read_json(json_string)`读取|从 JSON 格式的字符串、URL 或文件读取。`pd.read_html(url)` |解析 html URL、字符串或文件，并将表格提取到数据帧列表中`pd.read_clipboard()` |获取剪贴板中的内容，并将其传递给 read_table() `pd.DataFrame(dict)` |从 dict、列名关键字、数据值列表中##导出数据

使用这些命令将数据帧导出为 CSV 格式。xlsx、SQL 或 JSON。

`df.to_csv(filename)` |写入 CSV 文件`df.to_excel(filename)` |写入 Excel 文件`df.to_sql(table_name, connection_object)` |写入 SQL 表`df.to_json(filename)` |写入 JSON 格式的文件##创建测试对象

这些命令对于创建测试段非常有用。

`pd.DataFrame(np.random.rand(20,5))` | 5 列 20 行随机浮动`pd.Series(my_list)` |从可迭代的 my_list 中创建一个序列`df.index = pd.date_range('1900/1/30', periods=df.shape[0])` |添加一个日期索引##查看/检查数据

使用这些命令查看熊猫数据帧或系列的特定部分。

`df.head(n)` |数据帧的前 n 行`df.tail(n)` |数据帧的后 n 行`df.shape` |行数和列数`df.info()` |索引、数据类型和内存信息`df.describe()` |数字列的汇总统计信息`s.value_counts(dropna=False)` |查看唯一值和计数`df.apply(pd.Series.value_counts)` |所有列的唯一值和计数##选择

使用这些命令选择数据的特定子集。

`df[col]` |返回标签为 col 的列作为系列`df[[col1, col2]]` |返回列作为新的数据帧`s.iloc[0]` |按位置选择`s.loc['index_one']` |按索引选择`df.iloc[0,:]` |第一行`df.iloc[0,0]` |第一列的第一个元素##数据清理

使用这些命令执行各种数据清理任务。

`df.columns = ['a','b','c']` |重命名列`pd.isnull()` |检查空值， 返回布尔数组`pd.notnull()` |与 pd.isnull()相反`df.dropna()` |删除所有包含空值的行`df.dropna(axis=1)` |删除所有包含空值的列`df.dropna(axis=1,thresh=n)` |删除所有包含少于 n 个非空值的行`df.fillna(x)` |用 x 替换所有空值`s.fillna(s.mean())` |用平均值替换所有空值(平均值可以用来自[统计模块](https://docs.python.org/3/library/statistics.html) ) `s.astype(float)` | 将序列的数据类型转换为浮点数`s.replace(1,'one')` |将所有等于 1 的值替换为“一”`s.replace([1,3],['one','three'])` |将所有 1 替换为“一”，将所有 3 替换为“三”`df.rename(columns=lambda x: x + 1)` |对列进行大规模重命名`df.rename(columns={'old_name': 'new_ name'})` |选择性重命名`df.set_index('column_one')` |更改索引`df.rename(index=lambda x: x + 1)` |对索引进行大规模重命名##筛选、排序和分组依据

使用这些命令对数据进行筛选、排序和分组。

`df[df[col] > 0.5]` |列`col`大于`0.5`T3 的行】|列`0.7 > col > 0.5`T5 的行】|按列 1 升序排序`df.sort_values(col2,ascending=False)` |按`col2`降序排序`df.sort_values([col1,col2],ascending=[True,False])` |按`col1`降序排序`col2``df.groupby(col)`|返回一列值的 groupby 对象`df.groupby([col1,col2])` |返回多列值的 groupby 对象`df.groupby(col1)[col2]` |返回`col2`中值的平均值， 按`col1`中的值分组(平均值可以用统计模块中的几乎任何函数替换)`df.pivot_table(index=col1,values=[col2,col3],aggfunc=mean)` |创建一个按`col1`分组的数据透视表，并计算`col2`和`col3`的平均值`df.groupby(col1).agg(np.mean)` |求出每个唯一列 1 组的所有列的平均值`df.apply(np.mean)` |对每列应用函数`np.mean()``nf.apply(np.max,axis=1)`|对每行应用函数`np.max()`# #联接/组合

使用这些命令将多个数据帧合并成一个数据帧。

`df1.append(df2)` |将`df1`中的行添加到`df2`的末尾(列应该相同)`pd.concat([df1, df2],axis=1)` |将`df1`中的列添加到`df2`的末尾(行应该相同)`df1.join(df2,on=col1,how='inner')` | SQL 样式将`df1`中的列与`df2`中的列连接，其中`col`的行具有相同的值。`'how'`可以是`'left'`、`'right'`、`'outer'`、`'inner'` ##统计中的一种

使用这些命令执行各种统计测试。(这些都可以同样适用于一个系列。)

`df.describe()` |数字列的汇总统计数据`df.mean()` |返回所有列的平均值`df.corr()` |返回数据帧中列之间的相关性`df.count()` |返回每个数据帧列中非空值的数量`df.max()` |返回每列中的最高值`df.min()` |返回每列中的最低值`df.median()` |返回每列的中间值`df.std()` |返回每列的标准偏差##下载此备忘单的可打印版本

如果您想下载此备忘单的可打印版本[，您可以在此处](https://drive.google.com/file/d/1UHK8wtWbADvHKXFC937IS6MTnlSZC_zB/view)下载。

## 更多资源

如果你想了解关于这个话题的更多信息，请查看 Dataquest 的交互式 [Pandas 和 NumPy Fundamentals](https://www.dataquest.io/course/pandas-fundamentals/) 课程，以及我们的[Python 数据分析师](https://www.dataquest.io/path/data-analyst)和[Python 数据科学家](https://www.dataquest.io/path/data-scientist)路径，它们将帮助你在大约 6 个月内做好工作准备。

![YouTube video player for 6a5jbnUNE2E](img/1abf55e66817f421c9b041572037fe56.png)

*[https://www.youtube.com/embed/6a5jbnUNE2E?rel=0](https://www.youtube.com/embed/6a5jbnUNE2E?rel=0)*

 *提升您的数据技能。

[查看计划](/subscribe)*