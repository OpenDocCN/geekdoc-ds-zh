# 熊猫的日期时间:初学者简单指南(2022)

> 原文：<https://www.dataquest.io/blog/datetime-in-pandas/>

March 22, 2022![](img/90e58b95b66819d2e84e114a29164d1b.png)

我们被不同类型和形式的数据所包围。毫无疑问，时间序列数据是最有趣和最基本的数据类别之一。时间序列数据无处不在，它在各个行业都有很多应用。患者健康指标、股票价格变化、天气记录、经济指标、服务器、网络、传感器和应用程序性能监控都是时间序列数据的例子。

我们可以将时间序列数据定义为在不同时间间隔获得并按时间顺序排列的数据点的集合。

Pandas 库主要用于分析金融时间序列数据，并为处理时间、日期和时间序列数据提供一个全面的框架。

本教程将讨论在熊猫中处理日期和时间的不同方面。完成本教程后，您将了解以下内容:

*   `Timestamp`和`Period`对象的功能
*   如何使用时序数据框架
*   如何对时间序列进行切片
*   `DateTimeIndex`对象及其方法
*   如何对时间序列数据进行重采样

在本教程中，我们假设你知道熊猫系列和数据帧的基础知识。如果你不熟悉熊猫图书馆，你可能想试试我们的[熊猫和 NumPy 基础——data quest](https://www.dataquest.io/course/pandas-fundamentals/)。

让我们开始吧。

## 探索熊猫`Timestamp`和`Period`物品

pandas 库提供了一个名为 Timestamp 的具有纳秒精度的 DateTime 对象来处理日期和时间值。Timestamp 对象派生自 NumPy 的 datetime64 数据类型，这使得它比 Python 的 datetime 对象更准确，速度也快得多。让我们使用时间戳构造函数创建一些时间戳对象。打开 Jupyter 笔记本或 VS 代码，运行以下代码:

```py
import pandas as pd
import numpy as np
from IPython.display import display

print(pd.Timestamp(year=1982, month=9, day=4, hour=1, minute=35, second=10))
print(pd.Timestamp('1982-09-04 1:35.18'))
print(pd.Timestamp('Sep 04, 1982 1:35.18'))
```

```py
1982-09-04 01:35:10
1982-09-04 01:35:10
1982-09-04 01:35:10
```

运行上面的代码会返回输出，这些输出都表示相同的时间或时间戳实例。

如果您将单个整数或浮点值传递给`Timestamp`构造函数，它将返回一个时间戳，该时间戳相当于 Unix 纪元(1970 年 1 月 1 日)之后的纳秒数:

```py
print(pd.Timestamp(5000))
```

```py
1970-01-01 00:00:00.000005
```

对象包含许多方法和属性，帮助我们访问时间戳的不同方面。让我们试试它们:

```py
time_stamp = pd.Timestamp('2022-02-09')
print('{}, {} {}, {}'.format(time_stamp.day_name(), time_stamp.month_name(), time_stamp.day, time_stamp.year))
```

```py
Wednesday, February 9, 2022
```

`Timestamp`类的一个实例代表一个时间点，而`Period`对象的一个实例代表一段时间，比如一年、一个月等等。

例如，公司监控他们一年的收入。Pandas 库提供了一个名为`Period`的对象来处理句点，如下所示:

```py
year = pd.Period('2021')
display(year)
```

```py
Period('2021', 'A-DEC')
```

您可以在这里看到，它创建了一个表示 2021 年期间的`Period`对象，`'A-DEC'`表示该期间是每年的，在 12 月结束。

`Period`对象提供了许多有用的方法和属性。例如，如果要返回时间段的开始和结束时间，请使用以下属性:

```py
print('Start Time:', year.start_time)
print('End Time:', year.end_time)
```

```py
Start Time: 2021-01-01 00:00:00
End Time: 2021-12-31 23:59:59.999999999
```

要创建月期间，您可以将特定月份传递给它，如下所示:

```py
month = pd.Period('2022-01')
display(month)
print('Start Time:', month.start_time)
print('End Time:', month.end_time)
```

```py
Period('2022-01', 'M')

Start Time: 2022-01-01 00:00:00
End Time: 2022-01-31 23:59:59.999999999
```

`'M'`表示周期的频率为每月。您还可以使用`freq`参数明确指定周期的频率。下面的代码创建了一个表示 2022 年 1 月 1 日期间的 period 对象:

```py
day = pd.Period('2022-01', freq='D')
display(day)
print('Start Time:', day.start_time)
print('End Time:', day.end_time)
```

```py
Period('2022-01-01', 'D')

Start Time: 2022-01-01 00:00:00
End Time: 2022-01-01 23:59:59.999999999
```

我们还可以对周期对象执行算术运算。让我们以每小时的频率创建一个新的 period 对象，看看如何进行计算:

```py
hour = pd.Period('2022-02-09 16:00:00', freq='H')
display(hour)
display(hour + 2)
display(hour - 2)
```

```py
Period('2022-02-09 16:00', 'H')

Period('2022-02-09 18:00', 'H')

Period('2022-02-09 14:00', 'H')
```

我们可以使用熊猫日期偏移得到相同的结果:

```py
display(hour + pd.offsets.Hour(+2))
display(hour + pd.offsets.Hour(-2))
```

```py
Period('2022-02-09 18:00', 'H')

Period('2022-02-09 14:00', 'H')
```

要创建一个日期序列，可以使用 pandas `range_dates()`方法。让我们在代码片段中尝试一下:

```py
week = pd.date_range('2022-2-7', periods=7)
for day in week:
    print('{}-{}\t{}'.format(day.day_of_week, day.day_name(), day.date()))
```

```py
0-Monday    2022-02-07
1-Tuesday   2022-02-08
2-Wednesday 2022-02-09
3-Thursday  2022-02-10
4-Friday    2022-02-11
5-Saturday  2022-02-12
6-Sunday    2022-02-13
```

`week`的数据类型是一个`DatetimeIndex`对象，`week`中的每个日期都是`Timestamp`的一个实例。所以我们可以使用适用于一个`Timestamp`对象的所有方法和属性。

## 创建时间序列数据框架

首先，让我们通过从 CSV 文件中读取数据来创建一个数据帧，该文件包含与 50 台服务器相关的关键信息，连续 34 天每小时记录一次:

```py
df = pd.read_csv('https://raw.githubusercontent.com/m-mehdi/pandas_tutorials/main/server_util.csv')
display(df.head())
```

|  | 日期时间 | 服务器 id | cpu _ 利用率 | 空闲内存 | 会话计数 |
| --- | --- | --- | --- | --- | --- |
| Zero | 2019-03-06 00:00:00 | One hundred | Zero point four | Zero point five four | fifty-two |
| one | 2019-03-06 01:00:00 | One hundred | Zero point four nine | Zero point five one | Fifty-eight |
| Two | 2019-03-06 02:00:00 | One hundred | Zero point four nine | Zero point five four | Fifty-three |
| three | 2019-03-06 03:00:00 | One hundred | Zero point four four | Zero point five six | forty-nine |
| four | 2019-03-06 04:00:00 | One hundred | Zero point four two | Zero point five two | Fifty-four |

让我们看看数据帧的内容。每个数据帧行代表服务器的基本性能指标，包括特定时间戳的 CPU 利用率、空闲内存和会话计数。数据帧分解成一小时的片段。例如，从午夜到凌晨 4 点记录的性能指标位于数据帧的前五行。

现在，让我们详细了解一下数据帧的特征，比如它的大小和每一列的数据类型:

```py
print(df.info())
```

```py
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 40800 entries, 0 to 40799
Data columns (total 5 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   datetime         40800 non-null  object 
 1   server_id        40800 non-null  int64  
 2   cpu_utilization  40800 non-null  float64
 3   free_memory      40800 non-null  float64
 4   session_count    40800 non-null  int64  
dtypes: float64(2), int64(2), object(1)
memory usage: 1.6+ MB
None
```

运行上面的语句将返回行数和列数、总内存使用量、每列的数据类型等。

根据上面的信息， *datetime* 列的数据类型是`object`，这意味着时间戳被存储为字符串值。要将 *datetime* 列的数据类型从`string`对象转换为`datetime64`对象，我们可以使用 pandas `to_datetime()`方法，如下所示:

```py
df['datetime'] = pd.to_datetime(df['datetime'])
```

当我们通过导入 CSV 文件创建 DataFrame 时，日期/时间值被视为字符串对象，而不是日期时间对象。pandas `to_datetime()`方法将存储在 DataFrame 列中的日期/时间值转换为 DateTime 对象。将日期/时间值作为 DateTime 对象使得操作它们更加容易。运行以下语句并查看更改:

```py
print(df.info())
```

```py
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 40800 entries, 0 to 40799
Data columns (total 5 columns):
 #   Column           Non-Null Count  Dtype         
---  ------           --------------  -----         
 0   datetime         40800 non-null  datetime64[ns]
 1   server_id        40800 non-null  int64         
 2   cpu_utilization  40800 non-null  float64       
 3   free_memory      40800 non-null  float64       
 4   session_count    40800 non-null  int64         
dtypes: datetime64[ns](1), float64(2), int64(2)
memory usage: 1.6 MB
None
```

现在，*日期时间*列的数据类型是一个`datetime64[ns]`对象。`[ns]`表示基于纳秒的时间格式，用于指定 DateTime 对象的精度。

此外，我们可以让 pandas `read_csv()`方法将某些列解析为 DataTime 对象，这比使用`to_datetime()`方法更简单。让我们来试试:

```py
df = pd.read_csv('https://raw.githubusercontent.com/m-mehdi/pandas_tutorials/main/server_util.csv', parse_dates=['datetime'])
print(df.head())
```

```py
 datetime  server_id  cpu_utilization  free_memory  session_count
0 2019-03-06 00:00:00        100             0.40         0.54             52
1 2019-03-06 01:00:00        100             0.49         0.51             58
2 2019-03-06 02:00:00        100             0.49         0.54             53
3 2019-03-06 03:00:00        100             0.44         0.56             49
4 2019-03-06 04:00:00        100             0.42         0.52             54
```

运行上面的代码会创建一个 DataFrame，其中 *datetime* 列的数据类型是 datetime 对象。

在进入下一节之前，让我们对`datetime`列应用一些基本方法。

首先，让我们看看如何在 DataFrame 中返回最早和最晚的日期。为此，我们可以简单地对`datetime`列应用`max()`和`min()`方法，如下所示:

```py
display(df.datetime.min())
display(df.datetime.max())
```

```py
Timestamp('2019-03-06 00:00:00')

Timestamp('2019-04-08 23:00:00')
```

要选择两个特定日期之间的 DataFrame 行，我们可以创建一个布尔掩码，并使用`.loc`方法来过滤特定日期范围内的行:

```py
mask = (df.datetime >= pd.Timestamp('2019-03-06')) & (df.datetime < pd.Timestamp('2019-03-07'))
display(df.loc[mask])
```

|  | 日期时间 | 服务器 id | cpu _ 利用率 | 空闲内存 | 会话计数 |
| --- | --- | --- | --- | --- | --- |
| Zero | 2019-03-06 00:00:00 | One hundred | Zero point four | Zero point five four | fifty-two |
| one | 2019-03-06 01:00:00 | One hundred | Zero point four nine | Zero point five one | Fifty-eight |
| Two | 2019-03-06 02:00:00 | One hundred | Zero point four nine | Zero point five four | Fifty-three |
| three | 2019-03-06 03:00:00 | One hundred | Zero point four four | Zero point five six | forty-nine |
| four | 2019-03-06 04:00:00 | One hundred | Zero point four two | Zero point five two | Fifty-four |
| … | … | … | … | … | … |
| Forty thousand and three | 2019-03-06 19:00:00 | One hundred and forty-nine | Zero point seven four | Zero point two four | Eighty-one |
| Forty thousand and four | 2019-03-06 20:00:00 | One hundred and forty-nine | Zero point seven three | Zero point two three | Eighty-one |
| Forty thousand and five | 2019-03-06 21:00:00 | One hundred and forty-nine | Zero point seven nine | Zero point two nine | Eighty-three |
| Forty thousand and six | 2019-03-06 22:00:00 | One hundred and forty-nine | Zero point seven three | Zero point two nine | Eighty-two |
| Forty thousand and seven | 2019-03-06 23:00:00 | One hundred and forty-nine | Zero point seven five | Zero point two four | Eighty-four |

1200 行× 5 列

## 切片时间序列

为了使时间戳切片成为可能，我们需要将`datetime`列设置为数据帧的索引。要将列设置为数据帧的索引，请使用`set_index`方法:

```py
df.set_index('datetime', inplace=True)
print(df)
```

```py
 datetime    server_id  cpu_utilization  free_memory  session_count
2019-03-06 00:00:00        100             0.40         0.54             52
2019-03-06 01:00:00        100             0.49         0.51             58
2019-03-06 02:00:00        100             0.49         0.54             53
2019-03-06 03:00:00        100             0.44         0.56             49
2019-03-06 04:00:00        100             0.42         0.52             54
...                        ...              ...          ...            ...
2019-04-08 19:00:00        149             0.73         0.20             81
2019-04-08 20:00:00        149             0.75         0.25             83
2019-04-08 21:00:00        149             0.80         0.26             82
2019-04-08 22:00:00        149             0.75         0.29             82
2019-04-08 23:00:00        149             0.75         0.24             80

[40800 rows x 4 columns]
```

使用`.loc`方法选择等于一个索引的所有行:

```py
print(df.loc['2019-03-07 02:00:00'].head(5))
```

```py
 datetime       server_id  cpu_utilization  free_memory  session_count
2019-03-07 02:00:00        100             0.44         0.50             56
2019-03-07 02:00:00        101             0.78         0.21             87
2019-03-07 02:00:00        102             0.75         0.27             80
2019-03-07 02:00:00        103             0.76         0.28             85
2019-03-07 02:00:00        104             0.74         0.24             77
```

您可以在索引列中选择与特定时间戳部分匹配的行。让我们来试试:

```py
print(df.loc['2019-03-07'].head(5))
```

```py
 datetime      server_id  cpu_utilization  free_memory  session_count
2019-03-07 00:00:00        100             0.51         0.52             55
2019-03-07 01:00:00        100             0.46         0.50             49
2019-03-07 02:00:00        100             0.44         0.50             56
2019-03-07 03:00:00        100             0.45         0.52             51
2019-03-07 04:00:00        100             0.42         0.50             53
```

选择字符串可以是任何标准的日期格式，让我们看一些例子:

```py
df.loc['Apr 2019']
df.loc['8th April 2019']
df.loc['April 05, 2019 5pm']
```

我们还可以使用`.loc`方法在一个日期范围内对行进行切片。以下语句将返回从 2019 年 4 月 3 日开始到 2019 年 4 月 4 日结束的所有行；开始和结束日期都包括在内:

```py
display(df.loc['03-04-2019':'04-04-2019'])
```

但是运行它会引发一个令人讨厌的未来警告。为了消除警告，我们可以在对行进行切片之前对索引进行排序:

```py
display(df.sort_index().loc['03-04-2019':'04-04-2019'])
```

| 日期时间 | 服务器 id | cpu _ 利用率 | 空闲内存 | 会话计数 |
| --- | --- | --- | --- | --- |
| 2019-03-06 00:00:00 | One hundred | Zero point four | Zero point five four | fifty-two |
| 2019-03-06 00:00:00 | One hundred and thirty-five | Zero point five | Zero point five five | Fifty-five |
| 2019-03-06 00:00:00 | One hundred and ten | Zero point five four | Zero point four | Sixty-one |
| 2019-03-06 00:00:00 | One hundred and thirty-six | Zero point five eight | Zero point four | Sixty-four |
| 2019-03-06 00:00:00 | One hundred and nine | Zero point five seven | Zero point four one | Sixty-one |
| … | … | … | … | … |
| 2019-04-04 23:00:00 | One hundred and forty-three | Zero point four three | Zero point five two | Fifty |
| 2019-04-04 23:00:00 | One hundred and eleven | Zero point five three | Zero point five two | Fifty-nine |
| 2019-04-04 23:00:00 | One hundred and forty-nine | Zero point seven five | Zero point two four | eighty-five |
| 2019-04-04 23:00:00 | One hundred and thirty-eight | Zero point four | Zero point five six | Forty-seven |
| 2019-04-04 23:00:00 | One hundred and seven | Zero point six three | Zero point three three | Seventy-three |

36000 行× 4 列

## `DateTimeIndex`方法

一些熊猫数据帧方法仅适用于`DateTimeIndex`。我们将在本节中研究其中的一些，但是首先，让我们确保我们的数据帧有一个`DateTimeIndex`:

```py
print(type(df.index))
```

```py
<class 'pandas.core.indexes.datetimes.DatetimeIndex'>
```

要返回在特定时间收集的服务器监控数据，而不考虑日期，请使用`at_time()`方法:

```py
display(df.at_time('09:00'))
```

| 日期时间 | 服务器 id | cpu _ 利用率 | 空闲内存 | 会话计数 |
| --- | --- | --- | --- | --- |
| 2019-03-06 09:00:00 | One hundred | Zero point four eight | Zero point five one | Fifty-one |
| 2019-03-07 09:00:00 | One hundred | Zero point four five | Zero point four nine | fifty-six |
| 2019-03-08 09:00:00 | One hundred | Zero point four five | Zero point five three | Fifty-three |
| 2019-03-09 09:00:00 | One hundred | Zero point four five | Zero point five one | Fifty-three |
| 2019-03-10 09:00:00 | One hundred | Zero point four nine | Zero point five five | Fifty-five |
| … | … | … | … | … |
| 2019-04-04 09:00:00 | One hundred and forty-nine | Zero point seven five | Zero point two one | Eighty |
| 2019-04-05 09:00:00 | One hundred and forty-nine | Zero point seven one | Zero point two six | Eighty-three |
| 2019-04-06 09:00:00 | One hundred and forty-nine | Zero point seven five | Zero point three | Eighty-three |
| 2019-04-07 09:00:00 | One hundred and forty-nine | Zero point eight one | Zero point two eight | Seventy-seven |
| 2019-04-08 09:00:00 | One hundred and forty-nine | Zero point eight two | Zero point two four | Eighty-six |

1700 行× 4 列

此外，要选择所有日期午夜到凌晨 2 点之间的所有服务器数据，请使用`between_time()`方法。让我们来试试:

```py
display(df.between_time('00:00','02:00'))
```

| 日期时间 | 服务器 id | cpu _ 利用率 | 空闲内存 | 会话计数 |
| --- | --- | --- | --- | --- |
| 2019-03-06 00:00:00 | One hundred | Zero point four | Zero point five four | fifty-two |
| 2019-03-06 01:00:00 | One hundred | Zero point four nine | Zero point five one | Fifty-eight |
| 2019-03-06 02:00:00 | One hundred | Zero point four nine | Zero point five four | Fifty-three |
| 2019-03-07 00:00:00 | One hundred | Zero point five one | Zero point five two | Fifty-five |
| 2019-03-07 01:00:00 | One hundred | Zero point four six | Zero point five | forty-nine |
| … | … | … | … | … |
| 2019-04-07 01:00:00 | One hundred and forty-nine | Zero point seven four | Zero point two one | seventy-eight |
| 2019-04-07 02:00:00 | One hundred and forty-nine | Zero point seven six | Zero point two six | Seventy-four |
| 2019-04-08 00:00:00 | One hundred and forty-nine | Zero point seven five | Zero point two eight | Seventy-five |
| 2019-04-08 01:00:00 | One hundred and forty-nine | Zero point six nine | Zero point two seven | Seventy-nine |
| 2019-04-08 02:00:00 | One hundred and forty-nine | Zero point seven eight | Zero point two | eighty-five |

5100 行× 4 列

我们可以使用`first()`方法根据特定的日期偏移量选择第一个数据帧行。例如，将`5B`作为日期偏移量传递给该方法将返回前五个工作日内的所有索引行。类似地，将`1W`传递给`last()`方法将返回上周内所有带有索引的 DataFrame 行。请注意，数据帧必须按其索引排序，以确保这些方法有效。让我们试试这两个例子:

```py
display(df.sort_index().first('5B'))
```

| 日期时间 | 服务器 id | cpu _ 利用率 | 空闲内存 | 会话计数 |
| --- | --- | --- | --- | --- |
| 2019-03-06 | One hundred | Zero point four | Zero point five four | fifty-two |
| 2019-03-06 | One hundred and thirty-five | Zero point five | Zero point five five | Fifty-five |
| 2019-03-06 | One hundred and ten | Zero point five four | Zero point four | Sixty-one |
| 2019-03-06 | One hundred and thirty-six | Zero point five eight | Zero point four | Sixty-four |
| 2019-03-06 | One hundred and nine | Zero point five seven | Zero point four one | Sixty-one |
| … | … | … | … | … |
| 2019-03-12 | One hundred and thirty-four | Zero point five three | Zero point four five | Sixty-one |
| 2019-03-12 | One hundred and forty-four | Zero point six eight | Zero point three one | Seventy-three |
| 2019-03-12 | One hundred and thirteen | Zero point seven six | Zero point two four | Eighty-three |
| 2019-03-12 | One hundred and fourteen | Zero point five eight | Zero point four eight | Sixty-seven |
| 2019-03-12 | One hundred and thirty-one | Zero point five eight | Zero point four two | Sixty-seven |

7250 行× 4 列

```py
display(df.sort_index().last('1W'))
```

| 日期时间 | 服务器 id | cpu _ 利用率 | 空闲内存 | 会话计数 |
| --- | --- | --- | --- | --- |
| 2019-04-08 00:00:00 | One hundred and six | Zero point four four | Zero point six two | forty-nine |
| 2019-04-08 00:00:00 | One hundred and twelve | Zero point seven two | Zero point two nine | Eighty-one |
| 2019-04-08 00:00:00 | One hundred | Zero point four three | Zero point five four | Fifty-one |
| 2019-04-08 00:00:00 | One hundred and thirty-seven | Zero point seven five | Zero point two eight | Eighty-three |
| 2019-04-08 00:00:00 | One hundred and ten | Zero point six one | Zero point four | Sixty-two |
| … | … | … | … | … |
| 2019-04-08 23:00:00 | One hundred and twenty-eight | Zero point six four | Zero point four one | Sixty-four |
| 2019-04-08 23:00:00 | One hundred and twenty-seven | Zero point six seven | Zero point three three | seventy-eight |
| 2019-04-08 23:00:00 | One hundred and twenty-six | Zero point seven one | Zero point three three | Seventy-three |
| 2019-04-08 23:00:00 | One hundred and twenty-three | Zero point seven one | Zero point two two | Eighty-three |
| 2019-04-08 23:00:00 | One hundred and forty-nine | Zero point seven five | Zero point two four | Eighty |

1200 行× 4 列

```py
df.sort_index().last('2W')
```

| 日期时间 | 服务器 id | cpu _ 利用率 | 空闲内存 | 会话计数 |
| --- | --- | --- | --- | --- |
| 2019-04-01 00:00:00 | One hundred and twenty | Zero point five four | Zero point four eight | Sixty-three |
| 2019-04-01 00:00:00 | One hundred and four | Zero point seven three | Zero point three one | Eighty-three |
| 2019-04-01 00:00:00 | One hundred and three | Zero point seven seven | Zero point two two | Eighty-two |
| 2019-04-01 00:00:00 | One hundred and twenty-four | Zero point three nine | Zero point five five | forty-nine |
| 2019-04-01 00:00:00 | One hundred and twenty-seven | Zero point six eight | Zero point three seven | Seventy-three |
| … | … | … | … | … |
| 2019-04-08 23:00:00 | One hundred and twenty-eight | Zero point six four | Zero point four one | Sixty-four |
| 2019-04-08 23:00:00 | One hundred and twenty-seven | Zero point six seven | Zero point three three | seventy-eight |
| 2019-04-08 23:00:00 | One hundred and twenty-six | Zero point seven one | Zero point three three | Seventy-three |
| 2019-04-08 23:00:00 | One hundred and twenty-three | Zero point seven one | Zero point two two | Eighty-three |
| 2019-04-08 23:00:00 | One hundred and forty-nine | Zero point seven five | Zero point two four | Eighty |

9600 行× 4 列

## 重采样时间序列数据

`resample()`方法背后的逻辑类似于`groupby()`方法。它对任何可能时间段内的数据进行分组。虽然我们可以使用`resample()`方法进行上采样和下采样，但我们将重点关注如何使用它来执行下采样，这可以降低时间序列数据的频率——例如，将每小时的时间序列数据转换为每天的时间序列数据，或将每天的时间序列数据转换为每月的时间序列数据。

以下示例返回 ID 为 100 的服务器每天的平均 CPU 利用率、可用内存和活动会话数。为此，我们首先需要过滤服务器 ID 为 100 的数据帧的行，然后将每小时的数据重新采样为每天的数据。最后，对结果应用`mean()`方法，以获得三个指标的日平均值:

```py
df[df.server_id == 100].resample('D')['cpu_utilization', 'free_memory', 'session_count'].mean()
```

| 日期时间 | cpu _ 利用率 | 空闲内存 | 会话计数 |
| --- | --- | --- | --- |
| 2019-03-06 | 0.470417 | 0.535417 | 53.000000 |
| 2019-03-07 | 0.455417 | 0.525417 | 53.666667 |
| 2019-03-08 | 0.478333 | 0.532917 | 54.541667 |
| 2019-03-09 | 0.472917 | 0.523333 | 54.166667 |
| 2019-03-10 | 0.465000 | 0.527500 | 54.041667 |
| 2019-03-11 | 0.469583 | 0.528750 | 53.916667 |
| 2019-03-12 | 0.475000 | 0.533333 | 53.750000 |
| 2019-03-13 | 0.462917 | 0.521667 | 52.541667 |
| 2019-03-14 | 0.472083 | 0.532500 | 54.875000 |
| 2019-03-15 | 0.470417 | 0.530417 | 53.500000 |
| 2019-03-16 | 0.463750 | 0.530833 | 54.416667 |
| 2019-03-17 | 0.472917 | 0.532917 | 52.041667 |
| 2019-03-18 | 0.475417 | 0.535000 | 53.333333 |
| 2019-03-19 | 0.460833 | 0.546667 | 54.791667 |
| 2019-03-20 | 0.467083 | 0.529167 | 54.375000 |
| 2019-03-21 | 0.465833 | 0.543333 | 54.375000 |
| 2019-03-22 | 0.468333 | 0.528333 | 54.083333 |
| 2019-03-23 | 0.462500 | 0.539167 | 53.916667 |
| 2019-03-24 | 0.467917 | 0.537917 | 54.958333 |
| 2019-03-25 | 0.461250 | 0.530000 | 54.000000 |
| 2019-03-26 | 0.456667 | 0.531250 | 54.166667 |
| 2019-03-27 | 0.466667 | 0.530000 | 53.291667 |
| 2019-03-28 | 0.468333 | 0.532083 | 53.291667 |
| 2019-03-29 | 0.472917 | 0.538750 | 53.541667 |
| 2019-03-30 | 0.463750 | 0.526250 | 54.458333 |
| 2019-03-31 | 0.465833 | 0.522500 | 54.833333 |
| 2019-04-01 | 0.468333 | 0.527083 | 53.333333 |
| 2019-04-02 | 0.464583 | 0.515000 | 53.708333 |
| 2019-04-03 | 0.472500 | 0.533333 | 54.583333 |
| 2019-04-04 | 0.472083 | 0.531250 | 53.291667 |
| 2019-04-05 | 0.451250 | 0.540000 | 53.833333 |
| 2019-04-06 | 0.464167 | 0.531250 | 53.750000 |
| 2019-04-07 | 0.472500 | 0.530417 | 54.541667 |
| 2019-04-08 | 0.464583 | 0.534167 | 53.875000 |

我们还可以通过链接`groupby()`和`resample()`方法来查看每个服务器 ID 的相同结果。
以下语句返回每个服务器每个月的最大 CPU 利用率和空闲内存。让我们来试试:

```py
df.groupby(df.server_id).resample('M')['cpu_utilization', 'free_memory'].max()
```

| 服务器 id | 日期时间 | cpu _ 利用率 | 空闲内存 |
| --- | --- | --- | --- |
| One hundred | 2019-03-31 | Zero point five six | Zero point six two |
| 2019-04-30 | Zero point five five | Zero point six one |
| One hundred and one | 2019-03-31 | Zero point nine one | Zero point three two |
| 2019-04-30 | Zero point eight nine | Zero point three |
| One hundred and two | 2019-03-31 | Zero point eight five | Zero point three six |
| … | … | … | … |
| One hundred and forty-seven | 2019-04-30 | Zero point six one | Zero point five seven |
| One hundred and forty-eight | 2019-03-31 | Zero point eight four | Zero point three five |
| 2019-04-30 | Zero point eight three | Zero point three four |
| One hundred and forty-nine | 2019-03-31 | Zero point eight five | Zero point three six |
| 2019-04-30 | Zero point eight three | Zero point three four |

100 行× 2 列

在我们结束本教程之前，让我们画出每个月每台服务器的平均 CPU 利用率。这个结果让我们对几个月来每台服务器的平均 CPU 利用率的变化有了足够的了解。

```py
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(24, 8))
df.groupby(df.server_id).resample('M')['cpu_utilization'].mean()\
.plot.bar(color=['green', 'gray'], ax=ax, title='The Average Monthly CPU Utilization Comparison')
```

```py
<AxesSubplot:title={'center':'The Average Monthly CPU Utilization Comparison'}, xlabel='server_id,datetime'>
```

[![](img/aad84d2ead644237498e4dc88b494a26.png)](https://www.dataquest.io/wp-content/uploads/2022/03/output-57-1.webp)

## 结论

Pandas 是一个优秀的分析工具，尤其是在处理时间序列数据时。该库为处理时间索引数据帧提供了丰富的工具。本教程涵盖了在 pandas 中处理日期和时间的基本方面。

如果你想了解更多关于在 pandas 中处理时间序列数据的信息，你可以查看 Dataquest 博客上的[Pandas 时间序列分析](https://www.dataquest.io/blog/tutorial-time-series-analysis-with-pandas/)教程，当然，还有关于时间序列/日期功能的 [Pandas 文档](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html)。