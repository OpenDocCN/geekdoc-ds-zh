# 教程:在 Pandas 中索引数据帧

> 原文：<https://www.dataquest.io/blog/tutorial-indexing-dataframes-in-pandas/>

February 15, 2022![](img/57c67b0ad55fd3aa50a4ffab3ddbaef5.png)

在本教程中，我们将讨论索引熊猫数据帧意味着什么，为什么我们需要它，存在什么样的数据帧索引，以及应该使用什么语法来选择不同的子集。

## 什么是熊猫的索引数据帧？

索引熊猫数据帧意味着从该数据帧中选择特定的数据子集(如行、列、单个单元格)。Pandas 数据帧具有由行和列表示的固有表格结构，其中每一行和列在数据帧内具有唯一的标签(名称)和位置编号(类似于坐标)，并且每个数据点通过其在特定行和列的交叉点处的位置来表征。

称为数据帧索引的行标签可以是整数或字符串值，称为列名的列标签通常是字符串。由于数据帧索引和列名只包含唯一的值，我们可以使用这些标签来引用数据帧的特定行、列或数据点。另一方面，我们可以通过每个行、列或数据点在数据帧结构中的位置来描述它们。位置编号是整数，从第一行或第一列的 0 开始，随后每一行/列增加 1，因此它们也可以用作特定数据帧元素(行、列或数据点)的唯一坐标。这种通过标签或位置号引用数据帧元素的能力正是使数据帧索引成为可能的原因。

pandas 数据帧索引的一个具体(实际上也是最常见的)例子是切片。它用于访问数据帧元素的*序列，而不是单个数据帧元素*的*。*

Pandas 数据帧索引可用于各种任务:根据预定义的标准提取数据子集、重新组织数据、获取数据样本、数据操作、修改数据点的值等。

为了从数据帧中选择一个子集，我们使用了索引操作符`[]`、属性操作符`.`，以及熊猫数据帧索引的适当方法，例如`loc`、`iloc`、`at`、`iat`以及其他一些方法。

本质上，有两种主要的方法来索引熊猫数据帧:**基于标签的**和**基于位置的**(又名**基于位置的**或**基于整数的**)。此外，可以根据预定义的条件应用**布尔数据帧索引**，甚至混合不同类型的数据帧索引。让我们详细考虑一下所有这些方法。

为了进一步的实验，我们将创建一个假的数据帧:

```py
import pandas as pd

df = pd.DataFrame({'col_1': list(range(1, 11)), 'col_2': list(range(11, 21)), 'col_3': list(range(21, 31)),
                   'col_4': list(range(31, 41)), 'col_5': list(range(41, 51)), 'col_6': list(range(51, 61)),
                   'col_7': list(range(61, 71)), 'col_8': list(range(71, 81)), 'col_9': list(range(81, 91))})
df
```

|  | 第一栏 | 第二栏 | 第三栏 | col_4 | 第五栏 | col_6 | col_7 | 第八栏 | col_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Zero | one | Eleven | Twenty-one | Thirty-one | Forty-one | Fifty-one | Sixty-one | Seventy-one | Eighty-one |
| one | Two | Twelve | Twenty-two | Thirty-two | forty-two | fifty-two | Sixty-two | seventy-two | Eighty-two |
| Two | three | Thirteen | Twenty-three | Thirty-three | Forty-three | Fifty-three | Sixty-three | Seventy-three | Eighty-three |
| three | four | Fourteen | Twenty-four | Thirty-four | forty-four | Fifty-four | Sixty-four | Seventy-four | Eighty-four |
| four | five | Fifteen | Twenty-five | Thirty-five | Forty-five | Fifty-five | Sixty-five | Seventy-five | eighty-five |
| five | six | Sixteen | Twenty-six | Thirty-six | Forty-six | fifty-six | Sixty-six | Seventy-six | Eighty-six |
| six | seven | Seventeen | Twenty-seven | Thirty-seven | Forty-seven | Fifty-seven | Sixty-seven | Seventy-seven | Eighty-seven |
| seven | eight | Eighteen | Twenty-eight | Thirty-eight | Forty-eight | Fifty-eight | sixty-eight | seventy-eight | Eighty-eight |
| eight | nine | Nineteen | Twenty-nine | Thirty-nine | forty-nine | Fifty-nine | sixty-nine | Seventy-nine | eighty-nine |
| nine | Ten | Twenty | Thirty | Forty | Fifty | Sixty | Seventy | Eighty | Ninety |

## 基于标签的数据帧索引

顾名思义，这种方法意味着根据行和列标签选择数据帧子集。让我们探索四种基于标签的数据帧索引方法:使用索引操作符`[]`、属性操作符`.`、`loc`索引器和`at`索引器。

### 使用索引运算符

如果我们需要从 pandas 数据帧的一列或多列中选择所有数据，我们可以简单地使用索引操作符`[]`。为了从单个列中选择所有数据，我们传递该列的名称:

```py
df['col_2']
```

```py
0    11
1    12
2    13
3    14
4    15
5    16
6    17
7    18
8    19
9    20
Name: col_2, dtype: int64
```

最终的对象是一个熊猫系列。相反，如果我们想要一个单列数据帧作为输出，我们需要包含第二对方括号`[[]]`:

```py
print(df[['col_2']])
type(df[['col_2']])
```

```py
 col_2
0     11
1     12
2     13
3     14
4     15
5     16
6     17
7     18
8     19
9     20

pandas.core.frame.DataFrame
```

也可以从 pandas 数据帧中选择多个列，并将列名列表传递给索引操作符。此操作的结果将始终是一个熊猫数据帧:

```py
df[['col_5', 'col_1', 'col_8']]
```

|  | 第五栏 | 第一栏 | 第八栏 |
| --- | --- | --- | --- |
| Zero | Forty-one | one | Seventy-one |
| one | forty-two | Two | seventy-two |
| Two | Forty-three | three | Seventy-three |
| three | forty-four | four | Seventy-four |
| four | Forty-five | five | Seventy-five |
| five | Forty-six | six | Seventy-six |
| six | Forty-seven | seven | Seventy-seven |
| seven | Forty-eight | eight | seventy-eight |
| eight | forty-nine | nine | Seventy-nine |
| nine | Fifty | Ten | Eighty |

正如我们所看到的，列在输出数据帧中出现的顺序与列表中的顺序相同。当我们想要重新组织原始数据时，这是很有帮助的。

如果所提供的列表中至少有一个列名不在数据帧中，将抛出`KeyError`:

```py
df[['col_5', 'col_1', 'col_8', 'col_100']]
```

```py
---------------------------------------------------------------------------

KeyError                                  Traceback (most recent call last)

~\AppData\Local\Temp/ipykernel_2100/3615225278.py in <module>
----> 1 df[['col_5', 'col_1', 'col_8', 'col_100']]

~\anaconda3\lib\site-packages\pandas\core\frame.py in __getitem__(self, key)
   3462             if is_iterator(key):
   3463                 key = list(key)
-> 3464             indexer = self.loc._get_listlike_indexer(key, axis=1)[1]
   3465 
   3466         # take() does not accept boolean indexers

~\anaconda3\lib\site-packages\pandas\core\indexing.py in _get_listlike_indexer(self, key, axis)
   1312             keyarr, indexer, new_indexer = ax._reindex_non_unique(keyarr)
   1313 
-> 1314         self._validate_read_indexer(keyarr, indexer, axis)
   1315 
   1316         if needs_i8_conversion(ax.dtype) or isinstance(

~\anaconda3\lib\site-packages\pandas\core\indexing.py in _validate_read_indexer(self, key, indexer, axis)
   1375 
   1376             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 1377             raise KeyError(f"{not_found} not in index")
   1378 
   1379 

KeyError: "['col_100'] not in index"
```

### 使用属性运算符

要仅选择数据帧中的一列，我们可以通过其名称作为属性直接访问它:

```py
df.col_3
```

```py
0    21
1    22
2    23
3    24
4    25
5    26
6    27
7    28
8    29
9    30
Name: col_3, dtype: int64
```

上面这段代码相当于`df['col_3']`:

```py
df['col_3']
```

```py
0    21
1    22
2    23
3    24
4    25
5    26
6    27
7    28
8    29
9    30
Name: col_3, dtype: int64
```

然而，将列作为属性访问的方法有很多缺点。它不适用于以下情况:

*   如果列名包含空格或标点符号(下划线`_`除外)，
*   如果列名与 pandas 方法名一致(例如，“max”、“first”、“sum”)，
*   如果列名不是字符串类型(尽管使用这样的列名通常不是一个好的做法)，
*   为了选择多列，
*   用于创建新列(在这种情况下，尝试使用属性访问只会创建新属性而不是新列)。

让我们暂时在数据帧的列名中引入一些混乱，看看如果我们尝试使用属性 access 会发生什么:

```py
df.columns = ['col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col 6', 'col-7', 8, 'last']
df.columns
```

```py
Index(['col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col 6', 'col-7', 8,
       'last'],
      dtype='object')
```

上面，我们更改了最后四列的名称。现在，让我们看看如何在这些列上使用属性操作符:

```py
# The column name contains a white space
df.col 6
```

```py
 File "C:\Users\Utente\AppData\Local\Temp/ipykernel_2100/3654995016.py", line 2
    df.col 6
           ^
SyntaxError: invalid syntax
```

```py
# The column name contains a punctuation mark (except for the underscore)
df.col-7
```

```py
---------------------------------------------------------------------------

AttributeError                            Traceback (most recent call last)

~\AppData\Local\Temp/ipykernel_2100/1932640420.py in <module>
      1 # The column name contains a punctuation mark
----> 2 df.col-7

~\anaconda3\lib\site-packages\pandas\core\generic.py in __getattr__(self, name)
   5485         ):
   5486             return self[name]
-> 5487         return object.__getattribute__(self, name)
   5488 
   5489     def __setattr__(self, name: str, value) -> None:

AttributeError: 'DataFrame' object has no attribute 'col'
```

```py
# The column name coincides with a pandas method name
df.last
```

```py
 <bound method NDFrame.last of    col_1  col_2  col_3  col_4  col_5  col 6  col-7   8  last
    0      1     11     21     31     41     51     61  71    81
    1      2     12     22     32     42     52     62  72    82
    2      3     13     23     33     43     53     63  73    83
    3      4     14     24     34     44     54     64  74    84
    4      5     15     25     35     45     55     65  75    85
    5      6     16     26     36     46     56     66  76    86
    6      7     17     27     37     47     57     67  77    87
    7      8     18     28     38     48     58     68  78    88
    8      9     19     29     39     49     59     69  79    89
    9     10     20     30     40     50     60     70  80    90>
```

```py
# The column name is not a string 
df.8
```

```py
 File "C:\Users\Utente\AppData\Local\Temp/ipykernel_2100/2774159673.py", line 2
    df.8
      ^
SyntaxError: invalid syntax
```

```py
# An attempt to create a new column using the attribute access
df.new = 0
print(df.new)     # an attribute was created
print(df['new'])  # a column was not created
```

```py
0

---------------------------------------------------------------------------

KeyError                                  Traceback (most recent call last)

~\anaconda3\lib\site-packages\pandas\core\indexes\base.py in get_loc(self, key, method, tolerance)
   3360             try:
-> 3361                 return self._engine.get_loc(casted_key)
   3362             except KeyError as err:

~\anaconda3\lib\site-packages\pandas\_libs\index.pyx in pandas._libs.index.IndexEngine.get_loc()

~\anaconda3\lib\site-packages\pandas\_libs\index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas\_libs\hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas\_libs\hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'new'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)

~\AppData\Local\Temp/ipykernel_2100/2886494677.py in <module>
      2 df.new = 0
      3 print(df.new)     # an attribute was created
----> 4 print(df['new'])  # a column was not created

~\anaconda3\lib\site-packages\pandas\core\frame.py in __getitem__(self, key)
   3456             if self.columns.nlevels > 1:
   3457                 return self._getitem_multilevel(key)
-> 3458             indexer = self.columns.get_loc(key)
   3459             if is_integer(indexer):
   3460                 indexer = [indexer]

~\anaconda3\lib\site-packages\pandas\core\indexes\base.py in get_loc(self, key, method, tolerance)
   3361                 return self._engine.get_loc(casted_key)
   3362             except KeyError as err:
-> 3363                 raise KeyError(key) from err
   3364 
   3365         if is_scalar(key) and isna(key) and not self.hasnans:

KeyError: 'new'
```

注意，在上述所有情况下，语法`df[column_name]`都可以完美地工作。此外，在整个项目中使用相同的编码风格，包括数据帧索引的方式，提高了整体代码的可读性，因此保持一致并坚持更通用的风格是有意义的(在我们的例子中，使用方括号`[]`)。

让我们恢复原来的列名，并继续下一种基于标签的数据帧索引方法:

```py
df.columns = ['col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6', 'col_7', 'col_8', 'col_9']
df
```

|  | 第一栏 | 第二栏 | 第三栏 | col_4 | 第五栏 | col_6 | col_7 | 第八栏 | col_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Zero | one | Eleven | Twenty-one | Thirty-one | Forty-one | Fifty-one | Sixty-one | Seventy-one | Eighty-one |
| one | Two | Twelve | Twenty-two | Thirty-two | forty-two | fifty-two | Sixty-two | seventy-two | Eighty-two |
| Two | three | Thirteen | Twenty-three | Thirty-three | Forty-three | Fifty-three | Sixty-three | Seventy-three | Eighty-three |
| three | four | Fourteen | Twenty-four | Thirty-four | forty-four | Fifty-four | Sixty-four | Seventy-four | Eighty-four |
| four | five | Fifteen | Twenty-five | Thirty-five | Forty-five | Fifty-five | Sixty-five | Seventy-five | eighty-five |
| five | six | Sixteen | Twenty-six | Thirty-six | Forty-six | fifty-six | Sixty-six | Seventy-six | Eighty-six |
| six | seven | Seventeen | Twenty-seven | Thirty-seven | Forty-seven | Fifty-seven | Sixty-seven | Seventy-seven | Eighty-seven |
| seven | eight | Eighteen | Twenty-eight | Thirty-eight | Forty-eight | Fifty-eight | sixty-eight | seventy-eight | Eighty-eight |
| eight | nine | Nineteen | Twenty-nine | Thirty-nine | forty-nine | Fifty-nine | sixty-nine | Seventy-nine | eighty-nine |
| nine | Ten | Twenty | Thirty | Forty | Fifty | Sixty | Seventy | Eighty | Ninety |

### 使用`loc`步进器

如果我们不仅需要从数据帧中选择列，还需要选择行(或者只选择行)，我们可以使用`loc`方法，也称为`loc`索引器。这种方法也意味着使用索引操作符`[]`。这是通过标签访问数据帧行和列的最常见方式。

在继续之前，让我们看看数据帧的当前标签是什么。为此，我们将使用属性`columns`和`index`:

```py
print(df.columns)
print(df.index)
```

```py
Index(['col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6', 'col_7', 'col_8',
       'col_9'],
      dtype='object')
RangeIndex(start=0, stop=10, step=1)
```

注意，数据帧行的标签由一种特定类型的对象表示，在我们的例子中，它由从 0 到 9 的有序整数组成。这些整数是有效的行标签，它们可以在应用`loc`索引器时使用。例如，要提取带有*标签* 0 的行，这也是我们的数据帧的第一行，我们可以使用以下语法:

```py
df.loc[0]
```

```py
col_1     1
col_2    11
col_3    21
col_4    31
col_5    41
col_6    51
col_7    61
col_8    71
col_9    81
Name: 0, dtype: int64
```

一般来说，语法`df.loc[row_label]`用于从数据帧中提取特定的行作为熊猫系列对象。

然而，为了进一步实验`loc`索引器，让我们将行标签重命名为更有意义的字符串数据类型:

```py
df.index = ['row_1', 'row_2', 'row_3', 'row_4', 'row_5', 'row_6', 'row_7', 'row_8', 'row_9', 'row_10']
df
```

|  | 第一栏 | 第二栏 | 第三栏 | col_4 | 第五栏 | col_6 | col_7 | 第八栏 | col_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 第 1 行 | one | Eleven | Twenty-one | Thirty-one | Forty-one | Fifty-one | Sixty-one | Seventy-one | Eighty-one |
| 第 2 行 | Two | Twelve | Twenty-two | Thirty-two | forty-two | fifty-two | Sixty-two | seventy-two | Eighty-two |
| 第三行 | three | Thirteen | Twenty-three | Thirty-three | Forty-three | Fifty-three | Sixty-three | Seventy-three | Eighty-three |
| 第 4 行 | four | Fourteen | Twenty-four | Thirty-four | forty-four | Fifty-four | Sixty-four | Seventy-four | Eighty-four |
| 第 5 行 | five | Fifteen | Twenty-five | Thirty-five | Forty-five | Fifty-five | Sixty-five | Seventy-five | eighty-five |
| 第 6 行 | six | Sixteen | Twenty-six | Thirty-six | Forty-six | fifty-six | Sixty-six | Seventy-six | Eighty-six |
| 第 7 行 | seven | Seventeen | Twenty-seven | Thirty-seven | Forty-seven | Fifty-seven | Sixty-seven | Seventy-seven | Eighty-seven |
| 第 8 行 | eight | Eighteen | Twenty-eight | Thirty-eight | Forty-eight | Fifty-eight | sixty-eight | seventy-eight | Eighty-eight |
| 第 9 行 | nine | Nineteen | Twenty-nine | Thirty-nine | forty-nine | Fifty-nine | sixty-nine | Seventy-nine | eighty-nine |
| 第 10 行 | Ten | Twenty | Thirty | Forty | Fifty | Sixty | Seventy | Eighty | Ninety |

让我们使用`loc`索引器和新的行标签再次提取数据帧第一行的值:

```py
df.loc['row_1']
```

```py
col_1     1
col_2    11
col_3    21
col_4    31
col_5    41
col_6    51
col_7    61
col_8    71
col_9    81
Name: row_1, dtype: int64
```

如果我们想要访问**多个不同的行**，不一定按照原始数据帧中的顺序，我们必须传递一个行标签列表:

```py
df.loc[['row_6', 'row_2', 'row_9']]
```

|  | 第一栏 | 第二栏 | 第三栏 | col_4 | 第五栏 | col_6 | col_7 | 第八栏 | col_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 第 6 行 | six | Sixteen | Twenty-six | Thirty-six | Forty-six | fifty-six | Sixty-six | Seventy-six | Eighty-six |
| 第 2 行 | Two | Twelve | Twenty-two | Thirty-two | forty-two | fifty-two | Sixty-two | seventy-two | Eighty-two |
| 第 9 行 | nine | Nineteen | Twenty-nine | Thirty-nine | forty-nine | Fifty-nine | sixty-nine | Seventy-nine | eighty-nine |

结果数据帧中的行以与列表中相同的顺序出现。

如果提供的列表中至少有一个标签不在数据帧中，将抛出`KeyError`:

```py
df.loc[['row_6', 'row_2', 'row_100']]
```

```py
---------------------------------------------------------------------------

KeyError                                  Traceback (most recent call last)

~\AppData\Local\Temp/ipykernel_2100/3587512526.py in <module>
----> 1 df.loc[['row_6', 'row_2', 'row_100']]

~\anaconda3\lib\site-packages\pandas\core\indexing.py in __getitem__(self, key)
    929 
    930             maybe_callable = com.apply_if_callable(key, self.obj)
--> 931             return self._getitem_axis(maybe_callable, axis=axis)
    932 
    933     def _is_scalar_access(self, key: tuple):

~\anaconda3\lib\site-packages\pandas\core\indexing.py in _getitem_axis(self, key, axis)
   1151                     raise ValueError("Cannot index with multidimensional key")
   1152 
-> 1153                 return self._getitem_iterable(key, axis=axis)
   1154 
   1155             # nested tuple slicing

~\anaconda3\lib\site-packages\pandas\core\indexing.py in _getitem_iterable(self, key, axis)
   1091 
   1092         # A collection of keys
-> 1093         keyarr, indexer = self._get_listlike_indexer(key, axis)
   1094         return self.obj._reindex_with_indexers(
   1095             {axis: [keyarr, indexer]}, copy=True, allow_dups=True

~\anaconda3\lib\site-packages\pandas\core\indexing.py in _get_listlike_indexer(self, key, axis)
   1312             keyarr, indexer, new_indexer = ax._reindex_non_unique(keyarr)
   1313 
-> 1314         self._validate_read_indexer(keyarr, indexer, axis)
   1315 
   1316         if needs_i8_conversion(ax.dtype) or isinstance(

~\anaconda3\lib\site-packages\pandas\core\indexing.py in _validate_read_indexer(self, key, indexer, axis)
   1375 
   1376             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 1377             raise KeyError(f"{not_found} not in index")
   1378 
   1379 

KeyError: "['row_100'] not in index"
```

我们可能需要从原始数据帧中选择**多个连续的行**，而不是选择不同的行。在这种情况下，我们可以应用*切片*，即指定由冒号分隔的开始和结束行标签:

```py
df.loc['row_7':'row_9']
```

|  | 第一栏 | 第二栏 | 第三栏 | col_4 | 第五栏 | col_6 | col_7 | 第八栏 | col_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 第 7 行 | seven | Seventeen | Twenty-seven | Thirty-seven | Forty-seven | Fifty-seven | Sixty-seven | Seventy-seven | Eighty-seven |
| 第 8 行 | eight | Eighteen | Twenty-eight | Thirty-eight | Forty-eight | Fifty-eight | sixty-eight | seventy-eight | Eighty-eight |
| 第 9 行 | nine | Nineteen | Twenty-nine | Thirty-nine | forty-nine | Fifty-nine | sixty-nine | Seventy-nine | eighty-nine |

注意，对于`loc`索引器，*的开始和停止边界都是包含的*，这不是 Python 中常见的切片风格，通常停止边界是唯一的。

如果我们需要所有的行，包括特定行的(例如`df.loc[:'row_4']`)，或者从特定行开始，直到末端(例如`df.loc['row_4':]`)，可以将其中一个切片末端打开:

```py
# Selecting all the rows up to and including 'row_4'
df.loc[:'row_4']
```

|  | 第一栏 | 第二栏 | 第三栏 | col_4 | 第五栏 | col_6 | col_7 | 第八栏 | col_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 第 1 行 | one | Eleven | Twenty-one | Thirty-one | Forty-one | Fifty-one | Sixty-one | Seventy-one | Eighty-one |
| 第 2 行 | Two | Twelve | Twenty-two | Thirty-two | forty-two | fifty-two | Sixty-two | seventy-two | Eighty-two |
| 第三行 | three | Thirteen | Twenty-three | Thirty-three | Forty-three | Fifty-three | Sixty-three | Seventy-three | Eighty-three |
| 第 4 行 | four | Fourteen | Twenty-four | Thirty-four | forty-four | Fifty-four | Sixty-four | Seventy-four | Eighty-four |

为了选择**多行和多列**，实际上意味着这些行和列的交叉点处的数据帧的数据点的子集，我们向`loc`索引器传递两个由逗号分隔的参数:必要的行标签和列标签。对于行标签和列标签，它可以是一个标签列表、一个标签片(或一个开放标签片)或一个字符串形式的单个标签。一些例子:

```py
df.loc[['row_4', 'row_2'], ['col_5', 'col_2', 'col_9']]
```

|  | 第五栏 | 第二栏 | col_9 |
| --- | --- | --- | --- |
| 第 4 行 | forty-four | Fourteen | Eighty-four |
| 第 2 行 | forty-two | Twelve | Eighty-two |

```py
df.loc[['row_4', 'row_2'], 'col_5':'col_7']
```

|  | 第五栏 | col_6 | col_7 |
| --- | --- | --- | --- |
| 第 4 行 | forty-four | Fifty-four | Sixty-four |
| 第 2 行 | forty-two | fifty-two | Sixty-two |

```py
df.loc['row_4':'row_6', 'col_5':]
```

|  | 第五栏 | col_6 | col_7 | 第八栏 | col_9 |
| --- | --- | --- | --- | --- | --- |
| 第 4 行 | forty-four | Fifty-four | Sixty-four | Seventy-four | Eighty-four |
| 第 5 行 | Forty-five | Fifty-five | Sixty-five | Seventy-five | eighty-five |
| 第 6 行 | Forty-six | fifty-six | Sixty-six | Seventy-six | Eighty-six |

```py
df.loc[:'row_4', 'col_5']
```

```py
row_1    41
row_2    42
row_3    43
row_4    44
Name: col_5, dtype: int64
```

一个特殊的情况是当我们需要从一个数据帧中显式地获取一个值时。为此，我们将必要的行和列标签作为由逗号分隔的两个参数进行传递:

```py
df.loc['row_6', 'col_3']
```

```py
26
```

### 使用`at`步进器

对于上一节的最后一种情况，即从数据帧中只选择一个值，有一种更快的方法——使用`at`索引器。语法与`loc`索引器的语法相同，只是这里我们总是使用由逗号分隔的两个标签(对于行和列):

```py
df.at['row_6', 'col_3']
```

```py
26
```

## 基于位置的数据帧索引

使用这种方法，也称为基于位置或基于整数的方法，每个 dataframe 元素(行、列或数据点)通过其位置编号而不是标签来引用。位置号是整数，从第一行或第一列的 0 开始(典型的基于 Python 0 的索引)，随后的每一行/列增加 1。

基于标签和基于位置的数据帧索引方法之间的关键区别在于数据帧切片的方式:对于基于位置的索引，它是纯 Python 风格的，即范围的开始界限是包含性的，而停止界限是*排他性的*。对于基于标签的索引，停止界限是*包含*。

### 使用索引运算符

要从 pandas 数据帧的**多个连续行**中检索所有数据，我们可以简单地使用索引操作符`[]`和一系列必要的行位置(可以是一个开放的范围):

```py
df[3:6]
```

|  | 第一栏 | 第二栏 | 第三栏 | col_4 | 第五栏 | col_6 | col_7 | 第八栏 | col_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 第 4 行 | four | Fourteen | Twenty-four | Thirty-four | forty-four | Fifty-four | Sixty-four | Seventy-four | Eighty-four |
| 第 5 行 | five | Fifteen | Twenty-five | Thirty-five | Forty-five | Fifty-five | Sixty-five | Seventy-five | eighty-five |
| 第 6 行 | six | Sixteen | Twenty-six | Thirty-six | Forty-six | fifty-six | Sixty-six | Seventy-six | Eighty-six |

```py
df[:3]
```

|  | 第一栏 | 第二栏 | 第三栏 | col_4 | 第五栏 | col_6 | col_7 | 第八栏 | col_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 第 1 行 | one | Eleven | Twenty-one | Thirty-one | Forty-one | Fifty-one | Sixty-one | Seventy-one | Eighty-one |
| 第 2 行 | Two | Twelve | Twenty-two | Thirty-two | forty-two | fifty-two | Sixty-two | seventy-two | Eighty-two |
| 第三行 | three | Thirteen | Twenty-three | Thirty-three | Forty-three | Fifty-three | Sixty-three | Seventy-three | Eighty-three |

请注意，索引操作符不适用于选择单个行。

### 使用`iloc`步进器

这是通过位置编号选择数据帧行和列的最常见方式。这个方法的语法非常类似于`loc`索引器的语法，也意味着使用索引操作符`[]`。

为了访问**单行**或**多行**，我们将*一个参数*传递给`iloc`索引器，该索引器表示相应的行位置、行位置列表(如果行是不同的)或行位置范围(如果行是连续的):

```py
# Selecting one row 
df.iloc[3]
```

```py
col_1     4
col_2    14
col_3    24
col_4    34
col_5    44
col_6    54
col_7    64
col_8    74
col_9    84
Name: row_4, dtype: int64
```

```py
# Selecting disparate rows in the necessary order
df.iloc[[9, 8, 7]]
```

|  | 第一栏 | 第二栏 | 第三栏 | col_4 | 第五栏 | col_6 | col_7 | 第八栏 | col_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 第 10 行 | Ten | Twenty | Thirty | Forty | Fifty | Sixty | Seventy | Eighty | Ninety |
| 第 9 行 | nine | Nineteen | Twenty-nine | Thirty-nine | forty-nine | Fifty-nine | sixty-nine | Seventy-nine | eighty-nine |
| 第 8 行 | eight | Eighteen | Twenty-eight | Thirty-eight | Forty-eight | Fifty-eight | sixty-eight | seventy-eight | Eighty-eight |

```py
# Selecting a slice of sequential rows
df.iloc[3:6]
```

|  | 第一栏 | 第二栏 | 第三栏 | col_4 | 第五栏 | col_6 | col_7 | 第八栏 | col_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 第 4 行 | four | Fourteen | Twenty-four | Thirty-four | forty-four | Fifty-four | Sixty-four | Seventy-four | Eighty-four |
| 第 5 行 | five | Fifteen | Twenty-five | Thirty-five | Forty-five | Fifty-five | Sixty-five | Seventy-five | eighty-five |
| 第 6 行 | six | Sixteen | Twenty-six | Thirty-six | Forty-six | fifty-six | Sixty-six | Seventy-six | Eighty-six |

```py
# Selecting a slice of sequential rows (an open-ending range)
df.iloc[:3]
```

|  | 第一栏 | 第二栏 | 第三栏 | col_4 | 第五栏 | col_6 | col_7 | 第八栏 | col_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 第 1 行 | one | Eleven | Twenty-one | Thirty-one | Forty-one | Fifty-one | Sixty-one | Seventy-one | Eighty-one |
| 第 2 行 | Two | Twelve | Twenty-two | Thirty-two | forty-two | fifty-two | Sixty-two | seventy-two | Eighty-two |
| 第三行 | three | Thirteen | Twenty-three | Thirty-three | Forty-three | Fifty-three | Sixty-three | Seventy-three | Eighty-three |

注意，在最后两段代码中，不包括范围的停止边界，如前所述。

在所有其他情况下，要选择任意大小的数据子集(**一列**、**多列**、**多行多列在一起**、**单个数据点**)，我们将*两个参数*传递给`iloc`索引器:必要的行位置号和列位置号。对于这两个参数，值的潜在类型可以是:

*   整数(选择单行/列)，
*   整数列表(多个不同的行/列)，
*   整数范围(多个连续的行/列)，
*   整数的开放式范围(多个连续的行/列，直到并且*，不包括*特定位置编号(例如`df.iloc[:4, 1]`)，或者从特定编号开始直到结束(例如`df.loc[4:, 1]`)，
*   冒号(选择所有行/列)。

一些例子:

```py
# Selecting an individual data point
df.iloc[0, 0]
```

```py
1
```

```py
# Selecting one row and multiple disparate columns 
df.iloc[0, [2, 8, 3]]
```

```py
col_3    21
col_9    81
col_4    31
Name: row_1, dtype: int64
```

```py
# Selecting multiple disparate rows and multiple sequential columns 
df.iloc[[8, 1, 9], 5:9]
```

|  | col_6 | col_7 | 第八栏 | col_9 |
| --- | --- | --- | --- | --- |
| 第 9 行 | Fifty-nine | sixty-nine | Seventy-nine | eighty-nine |
| 第 2 行 | fifty-two | Sixty-two | seventy-two | Eighty-two |
| 第 10 行 | Sixty | Seventy | Eighty | Ninety |

```py
# Selecting multiple sequential rows and multiple sequential columns (with open-ending ranges)
df.iloc[:3, 6:]
```

|  | col_7 | 第八栏 | col_9 |
| --- | --- | --- | --- |
| 第 1 行 | Sixty-one | Seventy-one | Eighty-one |
| 第 2 行 | Sixty-two | seventy-two | Eighty-two |
| 第三行 | Sixty-three | Seventy-three | Eighty-three |

```py
# Selecting all rows and multiple disparate columns 
df.iloc[:, [1, 3, 7]]
```

|  | 第二栏 | col_4 | 第八栏 |
| --- | --- | --- | --- |
| 第 1 行 | Eleven | Thirty-one | Seventy-one |
| 第 2 行 | Twelve | Thirty-two | seventy-two |
| 第三行 | Thirteen | Thirty-three | Seventy-three |
| 第 4 行 | Fourteen | Thirty-four | Seventy-four |
| 第 5 行 | Fifteen | Thirty-five | Seventy-five |
| 第 6 行 | Sixteen | Thirty-six | Seventy-six |
| 第 7 行 | Seventeen | Thirty-seven | Seventy-seven |
| 第 8 行 | Eighteen | Thirty-eight | seventy-eight |
| 第 9 行 | Nineteen | Thirty-nine | Seventy-nine |
| 第 10 行 | Twenty | Forty | Eighty |

注意，如果我们向`iloc`索引器传递至少一个越界的位置号(无论是整行/列的单个整数还是列表中的一个整数)，将会抛出一个`IndexError`:

```py
df.iloc[[1, 3, 100]]
```

```py
---------------------------------------------------------------------------

IndexError                                Traceback (most recent call last)

~\anaconda3\lib\site-packages\pandas\core\indexing.py in _get_list_axis(self, key, axis)
   1529         try:
-> 1530             return self.obj._take_with_is_copy(key, axis=axis)
   1531         except IndexError as err:

~\anaconda3\lib\site-packages\pandas\core\generic.py in _take_with_is_copy(self, indices, axis)
   3627         """
-> 3628         result = self.take(indices=indices, axis=axis)
   3629         # Maybe set copy if we didn't actually change the index.

~\anaconda3\lib\site-packages\pandas\core\generic.py in take(self, indices, axis, is_copy, **kwargs)
   3614 
-> 3615         new_data = self._mgr.take(
   3616             indices, axis=self._get_block_manager_axis(axis), verify=True

~\anaconda3\lib\site-packages\pandas\core\internals\managers.py in take(self, indexer, axis, verify)
    861         n = self.shape[axis]
--> 862         indexer = maybe_convert_indices(indexer, n, verify=verify)
    863 

~\anaconda3\lib\site-packages\pandas\core\indexers.py in maybe_convert_indices(indices, n, verify)
    291         if mask.any():
--> 292             raise IndexError("indices are out-of-bounds")
    293     return indices

IndexError: indices are out-of-bounds

The above exception was the direct cause of the following exception:

IndexError                                Traceback (most recent call last)

~\AppData\Local\Temp/ipykernel_2100/47693281.py in <module>
----> 1 df.iloc[[1, 3, 100]]

~\anaconda3\lib\site-packages\pandas\core\indexing.py in __getitem__(self, key)
    929 
    930             maybe_callable = com.apply_if_callable(key, self.obj)
--> 931             return self._getitem_axis(maybe_callable, axis=axis)
    932 
    933     def _is_scalar_access(self, key: tuple):

~\anaconda3\lib\site-packages\pandas\core\indexing.py in _getitem_axis(self, key, axis)
   1555         # a list of integers
   1556         elif is_list_like_indexer(key):
-> 1557             return self._get_list_axis(key, axis=axis)
   1558 
   1559         # a single integer

~\anaconda3\lib\site-packages\pandas\core\indexing.py in _get_list_axis(self, key, axis)
   1531         except IndexError as err:
   1532             # re-raise with different error message
-> 1533             raise IndexError("positional indexers are out-of-bounds") from err
   1534 
   1535     def _getitem_axis(self, key, axis: int):

IndexError: positional indexers are out-of-bounds
```

但是，对于整数范围(一部分连续的行或列)，允许超出边界的位置数:

```py
df.iloc[1, 5:100]
```

```py
col_6    52
col_7    62
col_8    72
col_9    82
Name: row_2, dtype: int64
```

这也适用于上一节中讨论的索引运算符:

```py
df[7:1000]
```

|  | 第一栏 | 第二栏 | 第三栏 | col_4 | 第五栏 | col_6 | col_7 | 第八栏 | col_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 第 8 行 | eight | Eighteen | Twenty-eight | Thirty-eight | Forty-eight | Fifty-eight | sixty-eight | seventy-eight | Eighty-eight |
| 第 9 行 | nine | Nineteen | Twenty-nine | Thirty-nine | forty-nine | Fifty-nine | sixty-nine | Seventy-nine | eighty-nine |
| 第 10 行 | Ten | Twenty | Thirty | Forty | Fifty | Sixty | Seventy | Eighty | Ninety |

### 使用`iat`步进器

要从数据帧中只选择一个值，我们可以使用`iat`索引器，它比`iloc`执行得更快。语法与`iloc`索引器相同，只是这里我们总是使用两个整数(行号和列号):

```py
df.iat[1, 2]
```

```py
22
```

## 布尔数据帧索引

除了基于标签或基于位置的 pandas 数据帧索引之外，还可以根据特定条件从数据帧中选择一个子集:

```py
df[df['col_2'] > 15]
```

|  | 第一栏 | 第二栏 | 第三栏 | col_4 | 第五栏 | col_6 | col_7 | 第八栏 | col_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 第 6 行 | six | Sixteen | Twenty-six | Thirty-six | Forty-six | fifty-six | Sixty-six | Seventy-six | Eighty-six |
| 第 7 行 | seven | Seventeen | Twenty-seven | Thirty-seven | Forty-seven | Fifty-seven | Sixty-seven | Seventy-seven | Eighty-seven |
| 第 8 行 | eight | Eighteen | Twenty-eight | Thirty-eight | Forty-eight | Fifty-eight | sixty-eight | seventy-eight | Eighty-eight |
| 第 9 行 | nine | Nineteen | Twenty-nine | Thirty-nine | forty-nine | Fifty-nine | sixty-nine | Seventy-nine | eighty-nine |
| 第 10 行 | Ten | Twenty | Thirty | Forty | Fifty | Sixty | Seventy | Eighty | Ninety |

上面这段代码返回数据帧中所有列的值，其中列`col_2`的值大于 15。所应用的条件(在我们的例子中是–`df['col_2'] > 15`)是一个布尔向量，其长度与数据帧索引相同，并检查数据帧的每一行是否满足定义的标准。

我们还可以使用任何其他比较运算符:

*   `==`等于，
*   `!=`不等于，
*   `>`大于，
*   `<`小于，
*   `>=`大于或等于，
*   `<=`小于或等于。

也可以为字符串列定义布尔条件(比较运算符`==`和`!=`在这种情况下有意义)。

此外，我们可以在同一列或多列上定义几个标准。用于此目的的运算符有`&` ( *和*)、`|` ( *或*)、`~` ( *而非*)。每个条件必须放在单独的一对括号中:

```py
# Selecting all the rows of the dataframe where the value of `col_2` is greater than 15 but not equal to 19
df[(df['col_2'] > 15) & (df['col_2'] != 19)]
```

|  | 第一栏 | 第二栏 | 第三栏 | col_4 | 第五栏 | col_6 | col_7 | 第八栏 | col_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 第 6 行 | six | Sixteen | Twenty-six | Thirty-six | Forty-six | fifty-six | Sixty-six | Seventy-six | Eighty-six |
| 第 7 行 | seven | Seventeen | Twenty-seven | Thirty-seven | Forty-seven | Fifty-seven | Sixty-seven | Seventy-seven | Eighty-seven |
| 第 8 行 | eight | Eighteen | Twenty-eight | Thirty-eight | Forty-eight | Fifty-eight | sixty-eight | seventy-eight | Eighty-eight |
| 第 10 行 | Ten | Twenty | Thirty | Forty | Fifty | Sixty | Seventy | Eighty | Ninety |

```py
# Selecting all the rows of the dataframe where the value of `col_2` is greater than 15 
# or the value of `col_5` is equal to 42
df[(df['col_2'] > 15) | (df['col_5'] == 42)]
```

|  | 第一栏 | 第二栏 | 第三栏 | col_4 | 第五栏 | col_6 | col_7 | 第八栏 | col_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 第 2 行 | Two | Twelve | Twenty-two | Thirty-two | forty-two | fifty-two | Sixty-two | seventy-two | Eighty-two |
| 第 6 行 | six | Sixteen | Twenty-six | Thirty-six | Forty-six | fifty-six | Sixty-six | Seventy-six | Eighty-six |
| 第 7 行 | seven | Seventeen | Twenty-seven | Thirty-seven | Forty-seven | Fifty-seven | Sixty-seven | Seventy-seven | Eighty-seven |
| 第 8 行 | eight | Eighteen | Twenty-eight | Thirty-eight | Forty-eight | Fifty-eight | sixty-eight | seventy-eight | Eighty-eight |
| 第 9 行 | nine | Nineteen | Twenty-nine | Thirty-nine | forty-nine | Fifty-nine | sixty-nine | Seventy-nine | eighty-nine |
| 第 10 行 | Ten | Twenty | Thirty | Forty | Fifty | Sixty | Seventy | Eighty | Ninety |

```py
# Selecting all the rows of the dataframe where the value of `col_2` is NOT greater than 15 
df[~(df['col_2'] > 15)]
```

|  | 第一栏 | 第二栏 | 第三栏 | col_4 | 第五栏 | col_6 | col_7 | 第八栏 | col_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 第 1 行 | one | Eleven | Twenty-one | Thirty-one | Forty-one | Fifty-one | Sixty-one | Seventy-one | Eighty-one |
| 第 2 行 | Two | Twelve | Twenty-two | Thirty-two | forty-two | fifty-two | Sixty-two | seventy-two | Eighty-two |
| 第三行 | three | Thirteen | Twenty-three | Thirty-three | Forty-three | Fifty-three | Sixty-three | Seventy-three | Eighty-three |
| 第 4 行 | four | Fourteen | Twenty-four | Thirty-four | forty-four | Fifty-four | Sixty-four | Seventy-four | Eighty-four |
| 第 5 行 | five | Fifteen | Twenty-five | Thirty-five | Forty-five | Fifty-five | Sixty-five | Seventy-five | eighty-five |

## 数据帧索引方法的组合

最后，我们可以以各种方式组合基于标签、基于位置和布尔数据帧的索引方法。为此，我们应该再次应用`loc`索引器，并使用行的`index`属性和列的`columns`属性来访问位置号:

```py
df.loc[df.index[[3, 4]], ['col_3', 'col_7']]
```

|  | 第三栏 | col_7 |
| --- | --- | --- |
| 第 4 行 | Twenty-four | Sixty-four |
| 第 5 行 | Twenty-five | Sixty-five |

```py
df.loc['row_3':'row_6', df.columns[[0, 5]]]
```

|  | 第一栏 | col_6 |
| --- | --- | --- |
| 第三行 | three | Fifty-three |
| 第 4 行 | four | Fifty-four |
| 第 5 行 | five | Fifty-five |
| 第 6 行 | six | fifty-six |

```py
df.loc[df['col_4'] > 35, 'col_4':'col_7']
```

|  | col_4 | 第五栏 | col_6 | col_7 |
| --- | --- | --- | --- | --- |
| 第 6 行 | Thirty-six | Forty-six | fifty-six | Sixty-six |
| 第 7 行 | Thirty-seven | Forty-seven | Fifty-seven | Sixty-seven |
| 第 8 行 | Thirty-eight | Forty-eight | Fifty-eight | sixty-eight |
| 第 9 行 | Thirty-nine | forty-nine | Fifty-nine | sixty-nine |
| 第 10 行 | Forty | Fifty | Sixty | Seventy |

## 结论

总之，在本教程中，我们深入探讨了在 pandas 中索引数据帧。我们学到了很多东西:

*   什么是熊猫数据帧索引
*   数据帧索引的目的
*   什么是数据帧切片
*   什么是行标签和列标签
*   如何检查数据帧的当前行和列标签
*   什么是行列位置号
*   熊猫数据帧索引的主要方法
*   基于标签和基于位置的数据帧索引的关键区别
*   为什么使用属性 access 来选择数据帧列并不可取
*   布尔数据帧索引的工作原理
*   最常见的方法涉及每种数据帧索引方法、用于选择不同类型子集的语法以及它们的局限性
*   如何组合不同的索引方法