# 数据科学的 Python 备忘单:基础

> 原文：<https://www.dataquest.io/blog/python-cheat-sheet/>

July 20, 2017

第一次学习 Python 数据科学时，很难记住所有需要的语法，这是很常见的。虽然在 Dataquest，我们提倡习惯于查阅 [Python 文档](https://docs.python.org/3/)，但有时有一份方便的参考资料是很好的，所以我们整理了这份备忘单来帮助你！

这个备忘单是我们的 [Python 中级数据科学备忘单](https://www.dataquest.io/blog/data-science-python-cheat-sheet/)的伴侣

如果你想[学习 Python](https://www.dataquest.io/blog/learn-python-the-right-way/) ，我们有一个 [Python 编程:初学者](https://www.dataquest.io/course/python-for-data-science-fundamentals/)课程，可以开始你的数据科学之旅。

[下载此备忘单的可打印 PDF 文件](https://s3.amazonaws.com/dq-blog-files/python-cheat-sheet-basic.pdf)

## 关键基础知识，打印和获取帮助

`x = 3` |给变量`x`
`print(x)`赋值 3】 |打印 x
`type(x)`的值|返回变量的类型`x`(在本例中，`int`为整数)
`help(x)` |显示数据类型
`help(print)`的文档|显示`print()`函数的文档

## 读取文件

```py
f = open("my_file.txt", "r")
file_as_string = f.read()
```

打开文件`my_file.txt`并将其内容分配给`string`

```py
import csv
f = open("my_dataset.csv", "r")
csvreader = csv.reader(f)
csv_as_list = list(csvreader)
```

打开 CSV 文件`my_dataset.csv`并将其数据分配给列表列表`csv_as_list`

## 用线串

`s = "hello"` |将字符串`"hello"`赋给变量`s`

```py
s = """
She said,"there's a good idea.
""""
```

将多行字符串赋给变量`s`。也用于创建同时包含“和”字符的字符串。

`len(s)` |返回`s`
`s.startswith("hel")`中的字符个数|测试`s`是否以子串`"hel"`
`s.endswith("lo")`开头|测试`s`是否以子串`"lo"`
`"{} plus {} is {}".format(3,1,4)`结尾|返回值为`3`，`1`， 和`4`插入
`s.replace("e","z")` |返回一个基于`s`的新字符串，所有出现的`"e"`替换为`"z"`
`s.strip()` |返回一个基于`s`的新字符串，去掉字符串开头和结尾的任何空格
`s.split(" ")` |将字符串 s 拆分成一个字符串列表，在字符`" "`上进行分隔并返回该列表

## 数字类型和数学运算

`i = int("5")` |将字符串`"5"`转换为整数`5`并将结果赋给`i`
`f = float("2.5")` |将字符串`"2.5"`转换为浮点值`2.5`并将结果赋给`f`
`5 + 5` |加法
`5 - 5` |减法
`10 / 2` |除法
`5 * 2` |乘法
`3 ** 2` |将`3`提升为`2`(或\(3^{2}\)
`27 ** (1/3)`|第`3`根

## 列表

`l = [100, 21, 88, 3]` |分配一个包含整数`100`、`21`、`88`的列表， 并将`3`与变量`l`
`l = list()` |创建一个空列表并将结果赋给`l`
`l[0]` |返回列表中的第一个值`l`
`l[-1]` |返回列表中的最后一个值`l`
`l[1:3]` |返回包含`l`
`len(l)`的第二个和第三个值的切片(列表)】|返回`l`
`sum(l)`中元素的个数|返回`l`
的值之和 从`l`
`max(l)`返回最小值|从`l`
`l.append(16)`返回最大值|将值`16`追加到`l`
`l.sort()`的末尾|对`l`中的项目进行升序排序
`" ".join(["A", "B", "C", "D"])` |将列表`["A", "B", "C", "D"]`转换为字符串`"A B C D"`

## 字典

`d = {"CA": "Canada", "GB": "Great Britain", "IN": "India"}` |创建一个带有`"CA"`、`"GB"`和`"IN"`关键字以及`"Canada"`、`"Great Britain"`和`"India"`
`d["GB"]`对应值的字典|从带有`"GB"`
`d.get("AU","Sorry")`关键字的字典`d`中返回值|从带有`"AU"`关键字的字典`d`中返回值， 如果在`d`
`d.keys()`中找不到关键字`"AU"`则返回字符串`"Sorry"`|从`d`
`d.values()`返回关键字列表|从`d`
`d.items()`返回值列表|从`d`返回`(key, value)`对列表

## 模块和功能

*通过缩进定义函数体*

`import random` |导入模块`random`
`from random import random` |从模块`random`导入函数`random`

```py
def calculate(addition_one,addition_two,exponent=1,factor=1):
    result = (value_one + value_two) ** exponent * factor
    return result
```

用两个必需的和两个可选的命名参数定义一个新函数`calculate`，它计算并返回一个结果。

`addition(3,5,factor=10)` |用值`3`和`5`以及指定参数`10`运行加法函数

## 布尔比较

`x == 5` |测试`x`是否等于`5`
`x != 5` |测试`x`是否等于`5`
`x > 5` |测试`x`是否大于`5`
`x < 5` |测试`x`是否小于`5`
`x >= 5` |测试`x`是否大于等于`5`
`x <= 5` |测试`x`是否小于等于`5`
`x == 5 or name == "alfred"` | 测试`x`是否等于`5`或`name`是否等于`"alfred"`
`x == 5 and name == "alfred"` |测试`x`是否等于`5``name`是否等于`"alfred"`
`5 in l` |检查列表`l`
`"GB" in d`中是否有`5`的值|检查`d`的键中是否有`"GB"`的值

## 语句和循环

*if 语句和循环体通过缩进定义*

```py
if x > 5:
    print("{} is greater than five".format(x))
elif x < 0:
    print("{} is negative".format(x))
else:
    print("{} is between zero and five".format(x))
```

测试变量`x`的值，并根据该值运行代码体

```py
for value in l:
    print(value)
```

迭代`l`中的每个值，在每次迭代中运行循环体中的代码。

```py
while x < 10:
    x += 1
```

运行循环体中的代码，直到`x`的值不再小于`10`

[下载此备忘单的可打印 PDF 文件](https://s3.amazonaws.com/dq-blog-files/python-cheat-sheet-basic.pdf)

### 准备好继续学习了吗？

永远不要想接下来我该学什么？又来了！

在我们的 [Python for Data Science 路径](/path/data-scientist/)中，您将了解到:

*   使用 **matplotlib** 和 **pandas** 进行数据清理、分析和可视化
*   假设检验、概率和**统计**
*   机器学习、**深度学习**和决策树
*   ...还有更多！

立即开始学习我们的 **60+免费任务**:

[Try Dataquest (it's free!)](https://app.dataquest.io/signup)