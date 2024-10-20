# 教程:如何在熊猫中使用应用方法

> 原文：<https://www.dataquest.io/blog/tutorial-how-to-use-the-apply-method-in-pandas/>

February 18, 2022![](img/8c81ee55db042b188cee7c7fedcf9b6c.png)

`apply()`方法是最常用的数据预处理方法之一。它简化了对熊猫**系列**中的每个元素以及熊猫**数据帧**中的每一行或每一列应用函数。在本教程中，我们将学习如何在 pandas 中使用`apply()`方法——您需要了解 Python 和 [lambda 函数](https://app.dataquest.io/m/49)的基础知识。如果你不熟悉这些或者需要提高你的 Python 技能，你可能想试试我们的[免费 Python 基础课程](https://www.dataquest.io/course/python-for-data-science-fundamentals/)。

让我们开始吧。

## 在熊猫系列上应用函数

系列构成了熊猫的基础。它们本质上是一维数组，带有称为索引的轴标签。

创建一个系列对象有不同的方法(例如，我们可以用列表或字典初始化一个系列)。让我们定义一个包含两个列表的 Series 对象，这两个列表包含学生姓名作为索引，以厘米为单位的身高作为数据:

```py
import pandas as pd
import numpy as np
from IPython.display import display

students = pd.Series(data=[180, 175, 168, 190], 
                     index=['Vik', 'Mehdi', 'Bella', 'Chriss'])
display(students)
print(type(students))
```

```py
Vik       180
Mehdi     175
Bella     168
Chriss    190
dtype: int64

<class>
```

上面的代码返回了`students`对象的内容及其数据类型。

`students`对象的数据类型是*系列*，因此我们可以使用`apply()`方法对其数据应用任何函数。让我们看看如何将学生的身高从厘米转换为英尺:

```py
def cm_to_feet(h):
    return np.round(h/30.48, 2)

print(students.apply(cm_to_feet))
```

```py
Vik       5.91
Mehdi     5.74
Bella     5.51
Chriss    6.23
dtype: float64
```

学生身高换算成英尺，带两位小数。为此，我们首先定义一个进行转换的函数，然后将不带括号的函数名传递给`apply()`方法。`apply()`方法获取序列中的每个元素，并对其应用`cm_to_feet()`函数。

## 在熊猫数据帧上应用函数

在这一节中，我们将学习如何使用`apply()`方法来操作数据帧中的列和行。

首先，让我们使用下面的代码片段[创建一个包含公司员工个人详细信息的虚拟数据帧](https://www.dataquest.io/blog/tutorial-how-to-create-and-use-a-pandas-dataframe/):

```py
data = pd.DataFrame({'EmployeeName': ['Callen Dunkley', 'Sarah Rayner', 'Jeanette Sloan', 'Kaycee Acosta', 'Henri Conroy', 'Emma Peralta', 'Martin Butt', 'Alex Jensen', 'Kim Howarth', 'Jane Burnett'],
                    'Department': ['Accounting', 'Engineering', 'Engineering', 'HR', 'HR', 'HR', 'Data Science', 'Data Science', 'Accounting', 'Data Science'],
                    'HireDate': [2010, 2018, 2012, 2014, 2014, 2018, 2020, 2018, 2020, 2012],
                    'Sex': ['M', 'F', 'F', 'F', 'M', 'F', 'M', 'M', 'M', 'F'],
                    'Birthdate': ['04/09/1982', '14/04/1981', '06/05/1997', '08/01/1986', '10/10/1988', '12/11/1992', '10/04/1991', '16/07/1995', '08/10/1992', '11/10/1979'],
                    'Weight': [78, 80, 66, 67, 90, 57, 115, 87, 95, 57],
                    'Height': [176, 160, 169, 157, 185, 164, 195, 180, 174, 165],
                    'Kids': [2, 1, 0, 1, 1, 0, 2, 0, 3, 1]
                    })
display(data)
```

|  | 员工姓名 | 部门 | 你在说什么 | 性 | 出生年月日 | 重量 | 高度 | 小孩子 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Zero | 卡伦·邓克利 | 会计 | Two thousand and ten | M | 04/09/1982 | seventy-eight | One hundred and seventy-six | Two |
| one | 莎拉·雷纳 | 工程 | Two thousand and eighteen | F | 14/04/1981 | Eighty | One hundred and sixty | one |
| Two | 珍妮特·斯隆 | 工程 | Two thousand and twelve | F | 06/05/1997 | Sixty-six | One hundred and sixty-nine | Zero |
| three | 凯茜·阿科斯塔 | 人力资源（部） | Two thousand and fourteen | F | 08/01/1986 | Sixty-seven | One hundred and fifty-seven | one |
| four | 亨利康罗伊 | 人力资源（部） | Two thousand and fourteen | M | 10/10/1988 | Ninety | One hundred and eighty-five | one |
| five | 艾玛·佩拉尔塔 | 人力资源（部） | Two thousand and eighteen | F | 12/11/1992 | Fifty-seven | One hundred and sixty-four | Zero |
| six | 马丁·巴特 | 数据科学 | Two thousand and twenty | M | 10/04/1991 | One hundred and fifteen | One hundred and ninety-five | Two |
| seven | 艾利克斯詹森 | 数据科学 | Two thousand and eighteen | M | 16/07/1995 | Eighty-seven | one hundred and eighty  | Zero |
| eight | 金·豪沃思 | 会计 | Two thousand and twenty | M | 08/10/1992 | Ninety-five | One hundred and seventy-four | three |
| nine | 简·伯内特 | 数据科学 | Two thousand and twelve | F | 11/10/1979 | Fifty-seven | One hundred and sixty-five | one |

* * *

**注**

在这一部分，我们将处理由公司人力资源团队发起的虚拟请求。我们将通过不同的场景来学习如何使用`apply()`方法。我们将在每个场景中探索一个新的用例，并使用`apply()`方法解决它。

* * *

### 场景 1

让我们假设人力资源团队想要发送一封邀请电子邮件，以对所有员工的友好问候开始(例如，*嘿，莎拉！*)。他们要求您创建两列来分别存储雇员的名和姓，以便于引用雇员的名。为此，我们可以使用一个 lambda 函数，该函数在用指定的分隔符将一个字符串拆分成一个列表之后，将它拆分成一个列表；`split()`方法的默认分隔符是任何空格。让我们看看代码:

```py
data['FirstName'] = data['EmployeeName'].apply(lambda x : x.split()[0])
data['LastName'] = data['EmployeeName'].apply(lambda x : x.split()[1])
display(data)
```

|  | 员工姓名 | 部门 | 你在说什么 | 性 | 出生年月日 | 重量 | 高度 | 小孩子 | 西方人名的第一个字 | 姓 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Zero | 卡伦·邓克利 | 会计 | Two thousand and ten | M | 04/09/1982 | seventy-eight | One hundred and seventy-six | Two | 卡伦 | 邓克利 |
| one | 莎拉·雷纳 | 工程 | Two thousand and eighteen | F | 14/04/1981 | Eighty | One hundred and sixty | one | 撒拉 | 雷纳 |
| Two | 珍妮特·斯隆 | 工程 | Two thousand and twelve | F | 06/05/1997 | Sixty-six | One hundred and sixty-nine | Zero | 细斜纹布 | 斯隆 |
| three | 凯茜·阿科斯塔 | 人力资源（部） | Two thousand and fourteen | F | 08/01/1986 | Sixty-seven | One hundred and fifty-seven | one | 凯茜 | 阿科斯塔 |
| four | 亨利康罗伊 | 人力资源（部） | Two thousand and fourteen | M | 10/10/1988 | Ninety | One hundred and eighty-five | one | 亨利 | 康罗伊 |
| five | 艾玛·佩拉尔塔 | 人力资源（部） | Two thousand and eighteen | F | 12/11/1992 | Fifty-seven | One hundred and sixty-four | Zero | 女子名 | 佩拉尔塔 |
| six | 马丁·巴特 | 数据科学 | Two thousand and twenty | M | 10/04/1991 | One hundred and fifteen | One hundred and ninety-five | Two | 马丁 | 屁股 |
| seven | 艾利克斯詹森 | 数据科学 | Two thousand and eighteen | M | 16/07/1995 | Eighty-seven | one hundred and eighty  | Zero | 亚历克斯 | 詹森 |
| eight | 金·豪沃思 | 会计 | Two thousand and twenty | M | 08/10/1992 | Ninety-five | One hundred and seventy-four | three | 金姆（人名） | 霍沃斯 |
| nine | 简·伯内特 | 数据科学 | Two thousand and twelve | F | 11/10/1979 | Fifty-seven | One hundred and sixty-five | one | 简（女子名） | 伯内特 |

在上面的代码中，我们对`EmployeeName`列应用了 lambda 函数，从技术上讲，这是一个 Series 对象。lambda 函数将雇员的全名分为名和姓。因此，代码创建了另外两列，包含雇员的名字和姓氏。

### 场景 2

现在，让我们假设人力资源团队想要知道每个员工的年龄和员工的平均年龄，因为他们想要确定员工的年龄是否影响工作满意度和工作参与度。

要完成这项工作，第一步是定义一个函数，获取雇员的出生日期并返回他们的年龄:

```py
from datetime import datetime, date

def calculate_age(birthdate):
    birthdate = datetime.strptime(birthdate, '%d/%m/%Y').date()
    today = date.today()
    return today.year - birthdate.year - (today.month < birthdate.month)
```

`calculate_age()`函数以适当的格式获取一个人的出生日期，并在对其进行简单计算后，返回其年龄。

下一步是使用`apply()`方法对数据帧的`Birthdate`列应用函数，如下所示:

```py
data['Age'] = data['Birthdate'].apply(calculate_age)
display(data)
```

|  | 员工姓名 | 部门 | 你在说什么 | 性 | 出生年月日 | 重量 | 高度 | 小孩子 | 西方人名的第一个字 | 姓 | 年龄 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Zero | 卡伦·邓克利 | 会计 | Two thousand and ten | M | 04/09/1982 | seventy-eight | One hundred and seventy-six | Two | 卡伦 | 邓克利 | Thirty-nine |
| one | 莎拉·雷纳 | 工程 | Two thousand and eighteen | F | 14/04/1981 | Eighty | One hundred and sixty | one | 撒拉 | 雷纳 | Forty |
| Two | 珍妮特·斯隆 | 工程 | Two thousand and twelve | F | 06/05/1997 | Sixty-six | One hundred and sixty-nine | Zero | 细斜纹布 | 斯隆 | Twenty-four |
| three | 凯茜·阿科斯塔 | 人力资源（部） | Two thousand and fourteen | F | 08/01/1986 | Sixty-seven | One hundred and fifty-seven | one | 凯茜 | 阿科斯塔 | Thirty-six |
| four | 亨利康罗伊 | 人力资源（部） | Two thousand and fourteen | M | 10/10/1988 | Ninety | One hundred and eighty-five | one | 亨利 | 康罗伊 | Thirty-three |
| five | 艾玛·佩拉尔塔 | 人力资源（部） | Two thousand and eighteen | F | 12/11/1992 | Fifty-seven | One hundred and sixty-four | Zero | 女子名 | 佩拉尔塔 | Twenty-nine |
| six | 马丁·巴特 | 数据科学 | Two thousand and twenty | M | 10/04/1991 | One hundred and fifteen | One hundred and ninety-five | Two | 马丁 | 屁股 | Thirty |
| seven | 艾利克斯詹森 | 数据科学 | Two thousand and eighteen | M | 16/07/1995 | Eighty-seven | one hundred and eighty  | Zero | 亚历克斯 | 詹森 | Twenty-six |
| eight | 金·豪沃思 | 会计 | Two thousand and twenty | M | 08/10/1992 | Ninety-five | One hundred and seventy-four | three | 金姆（人名） | 霍沃斯 | Twenty-nine |
| nine | 简·伯内特 | 数据科学 | Two thousand and twelve | F | 11/10/1979 | Fifty-seven | One hundred and sixty-five | one | 简（女子名） | 伯内特 | forty-two |

上面的单行语句对`Birthdate`列的每个元素应用了`calculate_age()`函数，并将返回值存储在`Age`列中。

最后一步是计算雇员的平均年龄，如下所示:

```py
print(data['Age'].mean())
```

```py
32.8
```

### 场景 3

该公司的人力资源经理正在探索为所有员工提供医疗保险的方案。潜在的提供商需要关于雇员的信息。由于数据框架包含每个员工的体重和身高，我们假设人力资源经理要求您提供每个员工的体重指数(身体质量指数),以便她可以从潜在的医疗保健提供商那里获得报价。

为了完成这项任务，首先，我们需要定义一个计算身体质量指数(身体质量指数)的函数。身体质量指数的公式是以千克为单位的重量除以以米为单位的高度的平方。因为员工的身高是以厘米为单位的，所以我们需要将身高除以 100 来获得以米为单位的身高。让我们实现这个函数:

```py
def calc_bmi(weight, height):
    return np.round(weight/(height/100)**2, 2)
```

下一步是在数据帧上应用函数:

```py
data['BMI'] = data.apply(lambda x: calc_bmi(x['Weight'], x['Height']), axis=1)
```

lambda 函数获取每一行的体重和身高值，然后对它们应用`calc_bmi()`函数来计算它们的 BMI。`axis=1`参数意味着迭代数据帧中的行。

```py
display(data)
```

|  | 员工姓名 | 部门 | 你在说什么 | 性 | 出生年月日 | 重量 | 高度 | 小孩子 | 西方人名的第一个字 | 姓 | 年龄 | 身体质量指数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Zero | 卡伦·邓克利 | 会计 | Two thousand and ten | M | 04/09/1982 | seventy-eight | One hundred and seventy-six | Two | 卡伦 | 邓克利 | Thirty-nine | Twenty-five point one eight |
| one | 莎拉·雷纳 | 工程 | Two thousand and eighteen | F | 14/04/1981 | Eighty | One hundred and sixty | one | 撒拉 | 雷纳 | Forty | Thirty-one point two five |
| Two | 珍妮特·斯隆 | 工程 | Two thousand and twelve | F | 06/05/1997 | Sixty-six | One hundred and sixty-nine | Zero | 细斜纹布 | 斯隆 | Twenty-four | Twenty-three point one one |
| three | 凯茜·阿科斯塔 | 人力资源（部） | Two thousand and fourteen | F | 08/01/1986 | Sixty-seven | One hundred and fifty-seven | one | 凯茜 | 阿科斯塔 | Thirty-six | Twenty-seven point one eight |
| four | 亨利康罗伊 | 人力资源（部） | Two thousand and fourteen | M | 10/10/1988 | Ninety | One hundred and eighty-five | one | 亨利 | 康罗伊 | Thirty-three | Twenty-six point three |
| five | 艾玛·佩拉尔塔 | 人力资源（部） | Two thousand and eighteen | F | 12/11/1992 | Fifty-seven | One hundred and sixty-four | Zero | 女子名 | 佩拉尔塔 | Twenty-nine | Twenty-one point one nine |
| six | 马丁·巴特 | 数据科学 | Two thousand and twenty | M | 10/04/1991 | One hundred and fifteen | One hundred and ninety-five | Two | 马丁 | 屁股 | Thirty | Thirty point two four |
| seven | 艾利克斯詹森 | 数据科学 | Two thousand and eighteen | M | 16/07/1995 | Eighty-seven | one hundred and eighty  | Zero | 亚历克斯 | 詹森 | Twenty-six | Twenty-six point eight five |
| eight | 金·豪沃思 | 会计 | Two thousand and twenty | M | 08/10/1992 | Ninety-five | One hundred and seventy-four | three | 金姆（人名） | 霍沃斯 | Twenty-nine | Thirty-one point three eight |
| nine | 简·伯内特 | 数据科学 | Two thousand and twelve | F | 11/10/1979 | Fifty-seven | One hundred and sixty-five | one | 简（女子名） | 伯内特 | forty-two | Twenty point nine four |

最后一步是根据身体质量指数度量对员工进行分类。低于 18.5 的身体质量指数是第一组，在 18.5 和 24.9 之间是第二组，在 25 和 29.9 之间是第三组，超过 30 是第四组。为了实现该解决方案，我们将定义一个返回各种身体质量指数指标的函数，然后将其应用于数据帧的`BMI`列，以查看每个员工属于哪个类别:

```py
def indicator(bmi):
    if (bmi < 18.5):
        return 'Group One'
    elif (18.5 <= bmi < 25):
        return 'Group Two'
    elif (25 <= bmi < 30):
        return 'Group Three'
    else:
        return 'Group Four'

data['BMI_Indicator'] = data['BMI'].apply(indicator)
display(data)
```

|  | 员工姓名 | 部门 | 你在说什么 | 性 | 告发 | 重量 | 高度 | 小孩子 | 西方人名的第一个字 | 姓 | 年龄 | 身体质量指数 | 身体质量指数指标 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Zero | 卡伦·邓克利 | 会计 | Two thousand and ten | M | 04/09/1982 | seventy-eight | One hundred and seventy-six | Two | 卡伦 | 邓克利 | Thirty-nine | Twenty-five point one eight | 第三组 |
| one | 莎拉·雷纳 | 工程 | Two thousand and eighteen | F | 14/04/1981 | Eighty | One hundred and sixty | one | 撒拉 | 雷纳 | Forty | Thirty-one point two five | 第四组 |
| Two | 珍妮特·斯隆 | 工程 | Two thousand and twelve | F | 06/05/1997 | Sixty-six | One hundred and sixty-nine | Zero | 细斜纹布 | 斯隆 | Twenty-four | Twenty-three point one one | 第二组 |
| three | 凯茜·阿科斯塔 | 人力资源（部） | Two thousand and fourteen | F | 08/01/1986 | Sixty-seven | One hundred and fifty-seven | one | 凯茜 | 阿科斯塔 | Thirty-six | Twenty-seven point one eight | 第三组 |
| four | 亨利康罗伊 | 人力资源（部） | Two thousand and fourteen | M | 10/10/1988 | Ninety | One hundred and eighty-five | one | 亨利 | 康罗伊 | Thirty-three | Twenty-six point three | 第三组 |
| five | 艾玛·佩拉尔塔 | 人力资源（部） | Two thousand and eighteen | F | 12/11/1992 | Fifty-seven | One hundred and sixty-four | Zero | 女子名 | 佩拉尔塔 | Twenty-nine | Twenty-one point one nine | 第二组 |
| six | 马丁·巴特 | 数据科学 | Two thousand and twenty | M | 10/04/1991 | One hundred and fifteen | One hundred and ninety-five | Two | 马丁 | 屁股 | Thirty | Thirty point two four | 第四组 |
| seven | 艾利克斯詹森 | 数据科学 | Two thousand and eighteen | M | 16/07/1995 | Eighty-seven | one hundred and eighty  | Zero | 亚历克斯 | 詹森 | Twenty-six | Twenty-six point eight five | 第三组 |
| eight | 金·豪沃思 | 会计 | Two thousand and twenty | M | 08/10/1992 | Ninety-five | One hundred and seventy-four | three | 金姆（人名） | 霍沃斯 | Twenty-nine | Thirty-one point three eight | 第四组 |
| nine | 简·伯内特 | 数据科学 | Two thousand and twelve | F | 11/10/1979 | Fifty-seven | One hundred and sixty-five | one | 简（女子名） | 伯内特 | forty-two | Twenty point nine four | 第二组 |

### 场景 4

让我们假设新的一年即将来临，公司管理层已经宣布，那些有十年以上工作经验的员工将获得额外的奖金。人力资源经理想知道谁有资格获得奖金。

为了准备请求的信息，您需要对`HireDate`列应用下面的 lambda 函数，如果当前年份和雇佣年份之间的差值大于或等于十年，则返回`True`，否则返回`False`。

```py
mask = data['HireDate'].apply(lambda x: date.today().year - x >= 10)
print(mask)
```

```py
0     True
1    False
2     True
3    False
4    False
5    False
6    False
7    False
8    False
9     True
Name: HireDate, dtype: bool
```

运行上面的代码创建一个包含`True`或`False`值的 pandas 系列，称为布尔掩码。

为了显示合格的雇员，我们使用布尔掩码来过滤数据帧行。让我们运行下面的语句，看看结果:

```py
display(data[mask])
```

|  | 员工姓名 | 部门 | 你在说什么 | 性 | 告发 | 重量 | 高度 | 小孩子 | 西方人名的第一个字 | 姓 | 年龄 | 身体质量指数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Zero | 卡伦·邓克利 | 会计 | Two thousand and ten | M | 04/09/1982 | seventy-eight | One hundred and seventy-six | Two | 卡伦 | 邓克利 | Thirty-nine | Twenty-five point one eight |
| Two | 珍妮特·斯隆 | 工程 | Two thousand and twelve | F | 06/05/1997 | Sixty-six | One hundred and sixty-nine | Zero | 细斜纹布 | 斯隆 | Twenty-four | Twenty-three point one one |
| nine | 简·伯内特 | 数据科学 | Two thousand and twelve | F | 11/10/1979 | Fifty-seven | One hundred and sixty-five | one | 简（女子名） | 伯内特 | forty-two | Twenty point nine four |

### 场景 5

假设明天是母亲节，公司为所有有孩子的女员工策划了一份母亲节礼物。人力资源团队要求您准备一份有资格获得礼品的员工名单。为了完成这项任务，我们需要编写一个简单的 lambda 函数，它考虑到了`Sex`和`Kids`列，以提供所需的结果，如下所示:

```py
data[data.apply(lambda x: True if x ['Gender'] == 'F' and x['Kids'] > 0 else False, axis=1)]
```

|  | 员工姓名 | 部门 | 你在说什么 | 性 | 出生年月日 | 重量 | 高度 | 小孩子 | 西方人名的第一个字 | 姓 | 年龄 | 身体质量指数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| one | 莎拉·雷纳 | 工程 | Two thousand and eighteen | F | 14/04/1981 | Eighty | One hundred and sixty | one | 撒拉 | 雷纳 | Forty | Thirty-one point two five |
| three | 凯茜·阿科斯塔 | 人力资源（部） | Two thousand and fourteen | F | 08/01/1986 | Sixty-seven | One hundred and fifty-seven | one | 凯茜 | 阿科斯塔 | Thirty-six | Twenty-seven point one eight |
| nine | 简·伯内特 | 数据科学 | Two thousand and twelve | F | 11/10/1979 | Fifty-seven | One hundred and sixty-five | one | 简（女子名） | 伯内特 | forty-two | Twenty point nine four |

运行上面的代码会返回将收到礼物的员工列表。

如果女性雇员至少有一个孩子，lambda 函数返回`True`；否则返回`False`。对数据帧应用 lambda 函数的结果是一个布尔掩码，我们直接用它来过滤数据帧的行。

## 结论

在本教程中，我们通过不同的例子学习了`apply()`方法做什么以及如何使用它。apply()方法是一种强大而有效的方法，可以在 pandas 中对一个系列或数据帧的每个值应用函数。因为`apply()`方法使用 Python 的 C 扩展，所以当遍历 pandas 数据帧的所有行时，它执行得更快。但是，这并不是一个普遍的规则，因为当通过一列执行相同的操作时，它会变慢。