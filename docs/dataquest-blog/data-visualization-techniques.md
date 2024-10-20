# 6 个必须知道的数据可视化技术(2023)

> 原文：<https://www.dataquest.io/blog/data-visualization-techniques/>

December 8, 2022![Common data visualization techniques](img/35ad296711dedd0f045e42530bd7e757.png)

尽管对于任何数据从业者来说，可视化都是一项至关重要的技能，但在许多新人的数据科学学习道路上，可视化往往被视为理所当然。例如，绘制一个简单的图表来显示上个月的收入增加，这似乎是微不足道的。

与其他任务相比，数据可视化似乎过于简单。但是恰当地展示你的数据是一门艺术，可以决定你的项目是被接受还是被拒绝。为了让你的视觉效果脱颖而出，每个小细节都很重要。

在这篇博文中，我们将介绍最常见的数据可视化技术，并给出每种情况下的实际例子。这些技术可以使用各种工具来实现，从 Excel 到[特定的数据可视化编程语言](https://industrywired.com/top-10-programming-languages-used-for-data-visualization-in-2022/)和数据可视化软件，如 [Power BI](https://powerbi.microsoft.com/) 和 [Tableau](https://www.tableau.com/) 。

我们将使用 Python 来创建您将在本文中看到的情节，但是不要担心，跟随它不需要任何编码经验。此外，如果你不想马上进入编码，你可以使用市场上最重要的数据可视化工具之一的免费版本 [Tableau Public](https://public.tableau.com/app/discover) 开始创建你自己的可视化。

如果你想马上进入编码领域，你应该知道有很多选择。为了使事情更简单，这里有一个 Python 中最常用的数据-viz 工具的比较。

## 我们可视化的数据集

在本文中，我们将使用两个数据集作为示例。让我们快速浏览一下，这样您就可以对我们将要创建的可视化有所了解。

第一个数据集包含特定地方的天气和太阳能电池板发电的每日数据。以下是它包含的变量:

*   `temperature_min`:最低温度，单位为摄氏度
*   `temperature_max`:最高温度，单位为摄氏度
*   `humidity_min`:最小湿度，单位为%。
*   `humidity_max`:最大湿度，单位为%。
*   `rain_vol`:降雨量，单位为毫米。
*   `kwh`:发电量。

[点击此处](https://dq-blog.s3.amazonaws.com/6-common-data-visualization-techniques/power_and_weather21.csv)下载该数据集。

第二个数据集包含一家信用卡公司的客户数据，用于预测客户流失。以下是它包含的变量:

*   `education_level`:每个客户的教育程度。
*   `credit_limit`:客户的信用额度。
*   `total_trans_count`:成交笔数。
*   `total_trans_amount`:交易总金额。
*   `churn`:客户是否有过搅动。

[点击此处](https://dq-blog.s3.amazonaws.com/6-common-data-visualization-techniques/churn_predict_over_time.csv)下载该数据集。

## 1.表格和热图

### 桌子

让我们用最简单的可视化数据的方法来热身。表格通常是数据的原始形式，但它们也是可视化汇总数据的一种有价值的技术。然而，我们应该谨慎:表格可能包含太多的信息，这可能会给你的受众带来解释问题。

下表包含一年中每个季度的平均最高温度和发电量:

| 四分之一 | 最高温度 | 千瓦小时 |
| one | Twenty-nine point two seven | Thirty-two point eight eight |
| Two | Twenty-six point four five | Twenty-five point one nine |
| three | Twenty-seven point one one | Twenty-four point three six |
| four | Twenty-six point nine two | Twenty-six point seven seven |

在我们读取数据后，该表仅用一行 Python 代码创建:

```py
df = pd.read_csv('power_and_weather21.csv')
round(df.groupby('quarter')['temperature_max','kwh'].mean(), 2)
```

如果想在 Excel 中做，只需要插入一个数据透视表，将一年中各个季度的值相加即可。

表格不一定很难阅读，但是要找到你要找的信息可能要花一些时间。这里的主要提示是，为了在演示文稿中使用表格，您应该确保它们不要太长或有很多列。格式化的其他小技巧:移除或增加边框和不必要的网格线的透明度，以突出数据。这样做的目的是增加用于表示数据的墨水(在本例中是像素)相对于整个绘图中墨水(像素)总量的比例。这个比例越高，情节越强调数据。这个概念被称为数据-油墨比。

### 热图

热图使用颜色使表格更容易阅读。下图包含与我们刚刚看到的表格相同的数据:

![](img/aefd3fbfa9680216009783716ff25842.png)

色标更容易看出第一季度的温度和发电量最高。

热图是让表格更容易阅读和得出结论的一种很好的方式。

既然我们在讨论使用不同工具的可能性，这里有一些资源可以帮助你使用 [Google Sheets](https://workspace.google.com/products/sheets/) 创建相同的热图。

如果您还没有下载[发电数据集](https://github.com/dataquestio/dq-blog/blob/ot%C3%A1vio-common-data-visualization-techniques/Draft%20Blog%20Post/power_and_weather21.csv)，并将其上传到 Sheets。然后你只需要[创建一个数据透视表](https://support.google.com/a/users/answer/9308944?hl=en)和[添加条件过滤](https://support.google.com/docs/answer/78413?hl=en&co=GENIE.Platform%3DDesktop#zippy=)到你想要创建热图的列上。这个过程是完全可定制的，你可以让你的图表看起来像你想要的样子！

## 2.折线图

折线图适合绘制时间序列——一个或多个序列如何随时间变化。虽然绘制时间序列不是这种图表的唯一用途，但它是最适合它的。这是因为这条线给读者一种连续性的感觉，如果你要比较两个类别，这看起来并不好，我们将在下面看到。

例如，下图显示了一年中最高温度的变化情况:

![](img/a6c80237d022dccee90462f97c54fcf3.png)

我们也可以在单个图表中绘制更多系列。下面是每个月的日最高温度:

![](img/ef19e884d132571ff17721a5187d43d6.png)

然而，如果我们把太多的系列放在一起，即使每个系列都是不同的颜色，也会变得有点乱。试着把注意力集中在你想传达给听众的信息上。这是以另一种方式绘制的同一图表:

![](img/9a916308ee8c8d1ac9e4e0f177bd5579.png)

请注意，尽管我们仍然将所有的系列绘制在同一个图表上，但我们的观众能够将注意力集中在九月。

当一次绘制多条线时，混乱的图表和良好的图表之间的区别在于使用格式向您的受众传递信息。上面两张图片使用了相同的数据，但是传递了完全不同的信息。

虽然我们可以写一整篇文章来讲述如何让我们的图更漂亮，但主要的想法是让我们的可视化尽可能的清晰，并把重点放在数据上。

这里有一些你可以使用的额外资源:

如果你想学习如何用数据讲故事， [Cole Knaflic 的书《用数据讲故事》](https://www.storytellingwithdata.com/books)是最好的资源。同样，在这个话题上，[科尔在谷歌](https://www.youtube.com/watch?v=8EMW7io4rSI&ab_channel=TalksatGoogle)上的讲座绝对值得你花时间，如果你想改进你的情节的话。

## 3.面积图

面积图只是折线图的一种变体，主要区别在于面积图对直线和 x 轴之间的区域应用了阴影。这种图表用于比较多个变量随时间的变化情况。

例如，使用[这个客户数据集](https://github.com/dataquestio/dq-blog/blob/ot%C3%A1vio-common-data-visualization-techniques/Draft%20Blog%20Post/churn_predict_over_time.csv)的修改版本，我们可以绘制每年客户总数的面积图。这个新数据包含了不同年份的数据，增加了无数可能性。

![](img/a7caf4c6fb0e0b0b168fd6a47eba9b95.png)

不过，我们需要注意不要不必要地使用面积图。上面的图看起来就像一个普通的折线图，填充区域几乎没有什么不同，除了可能会让读者感到困惑。

然而，当绘制多个变量时，我们可以看到堆积面积图更有意义。下图不仅显示了每年的客户总数，还显示了搅动和未搅动客户的数量:

![](img/8e6c8cd023804911316fb4d0ffdce234.png)

现在我们可以看到，尽管客户总数在增加，但客户流失率似乎也在以更高的速度增长。

如果 Tableau 是你选择的工具，这里有一个关于使用 Tableau 创建面积图的快速[教程。](https://help.tableau.com/current/pro/desktop/en-us/qs_area_charts.htm)

## 4.条形图

条形图是最常见、最易读的图表之一。它们最常见的用例是显示不同类别之间的值如何变化。它们是餐桌的很好替代品。

它们很容易理解，你的读者只需快速看一下图表就能理解你的信息。例如，在下面的图表中，我们可以很容易地看到，在这个数据集中，非老客户比老客户多得多:

![](img/df8a7ecaa5c24ab171657de2cf735a60.png)

尽管如此，重要的是要注意，如果我们没有正确使用它们，这种图表是如何具有欺骗性的。下图显示了每组客户的平均收入:

![](img/fe4d47932abcce0db3e3fd2e8e0239ee.png)

快速浏览一下图表，你可能会得出这样的结论:这两组人之间存在显著差异，喝醉的顾客比没喝醉的顾客赚的钱少得多。然而，事实并非如此。注意 y 轴从 60000 开始，完全改变了剧情的视觉冲击。当 y 轴从零开始时，这个图表看起来是这样的:

![](img/a9cc68bc15636167e9b06d92ab76d830.png)

尽管被搅动的顾客平均来说赚的钱还是少了，但是差别很小，酒吧的高度看起来几乎一样。使用条形图和修改 y 轴上的单位时，我们需要非常小心。

请注意，我们将值写在了条形上，以使各组之间的差异显而易见。在定制图表时，Python 允许很多可能性。这是这个的源代码:

```py
import pandas as pd
import matplotlib.pyplot as plt
df_churn = pd.read_csv('churn_predict.csv')
fig = plt.figure(figsize=(12,6))
df_income = df_churn.groupby('churn')['estimated_income'].mean().reset_index().sort_values('estimated_income', ascending=False)
plt.bar(df_income['churn'], df_income['estimated_income'])

for _, row in df_income.iterrows():
    plt.text(row['churn'], row['estimated_income'] - 10000, 
             '$'+str(int(row['estimated_income'])), ha='center',
             fontsize=14, color='w')

plt.title('Average Income by Churn Status')
plt.tight_layout()
plt.show()
```

### 水平条形图

水平条形图是条形图的变体，其效果与其原始形式一样好。这是一个强烈推荐的图表，用于可视化每个类别的值，尤其是当我们有许多名称较长的类别时。

这是因为图表与我们习惯阅读的方向相同——从左到右。例如，看看下面的图表，它代表了受教育程度不同的客户数量:

![](img/4e8c645d6b0ae4b895d0e19b7764c18c.png)

请注意，我们在图的最左边部分有每个条形的标签，这是我们看它时眼睛首先去的地方，然后我们流入已经知道它的意思的数据。

此外，画一条假想的垂直线比画一条水平线更容易比较每个条形的大小。更容易理解的是，毕业生比其他人更容易流失。

在常规条形图中，我们需要注意 y 轴，而在水平条形图中，我们关注的是 x 轴。重要的是要确保它总是从零开始，或者有一个非常好的理由不这样做。

条形图中的颜色选择也是一个有趣的点，尤其是当我们有几个条形图时。如果每个条形代表一个类别，我们应该用颜色来区分它们。但是如果条形代表一个类别中的变化，那么渐变可能是展示这种关系的好方法。

这里有一篇关于如何为图表选择颜色的完整文章。

### 堆积条形图

条形图的另一种变体是堆积条形图。这种类型的可视化用于在每个条形内显示另一个变量。这是我们刚刚看到的堆积面积图的条形图版本。

例如，我们可以使用第一个条形图，按照另一个类别(教育水平)对每个条形图进行细分:

![](img/106ac1e30292bc2633f621f5e9e1ed56.png)

这个新的图表不仅显示了被搅动和未被搅动的顾客的数量，还显示了他们是如何被另一个类别划分的。

我们需要小心从这样的图表中得出的见解。举例来说，很容易说没有受过教育的毕业生比受过教育的毕业生多，但考虑到这两个群体的规模，就很难说这是真的了。

因此，一个好的建议是使用百分比而不是名义金额。这被称为规范化数据。这使得条形具有相同的高度，这使得比较更加容易:

![](img/84bdab939b707c6edf6b90861318c0ca.png)

请注意，尽管第一个图表中的标称值明显不同，但归一化值基本相同。

为了让你的情节更好，这里有一些设计技巧来帮助你的可视化。

### 瀑布图

瀑布图是条形图的另一种变体。就像堆积条形图一样，瀑布图的目的是可视化分解成其他变量的条形图的组成。回到我们的第一个数据集，这里有一个瀑布图，显示了今年前六个月以及下半年按月份细分的发电量:

![](img/6c1d8abda83929bbc98562fdc33f533d.png)

请注意，每个绿色条从上一个结束的高度开始。如果我们把所有的绿色条加在一起，就会得出图表两边的两个蓝色条的差值。

此外，由于每个月都有自己的条形，所以整个图只能显示一个条形的分解情况，在本例中，就是当年的发电量。

瀑布图和堆积条形图之间的另一个区别是，瀑布图可以在主变量的分解中表示负值。想象一下，由于某种奇怪的原因，我们有几个月的负发电量。这是该图的样子:

![](img/604fcf0693fe9dd1b59dff9d1d2e263c.png)

创建这种图表时要记住的一件重要事情是充分利用颜色。在这种情况下，我们需要三种颜色:一种用于图表左侧和右侧的合并条形，另外两种颜色分别用于正负条形。

红色条代表负值。它们从前一个绿色条的顶部开始，并在该标记下方结束。下一个小节从红色小节的底部开始。

如果你喜欢 Power BI，这里有一个关于如何使用 Power BI 创建瀑布图的[完整指南。](https://learn.microsoft.com/en-us/power-bi/visuals/power-bi-visualization-waterfall-charts?tabs=po)

## 5.散点图

散点图用于显示两个变量之间的关系，并检查它们之间是否存在相关性。它们不像以前的可视化那样直观，没有准备的读者可能要花一些时间才能理解它们。

例如，下图显示了交易总数和这些交易的总金额之间存在有趣的相关性:

![](img/2ca4ba4a43dc4f6c1cebb0af64d338f1.png)

这很有道理:一个人用信用卡购物的次数越多，他们的总金额就越大。

我们还可以在此图表中添加另一个维度，使用一种颜色来代表客户的分类变量:

![](img/e68ecbd424cf727287f443dcd589efe3.png)

我们可以看到，当交易数量或交易总额达到一定程度时，客户变得非常不容易流失。

我们还可以向散点图添加另一个轴来创建三维图表，但通常不建议这样做，因为这会使图表变得混乱，更难阅读。

当查看清楚显示相关性的散点图时，我们在得出结论时必须非常小心。尽管这两个变量可能密切相关，但这并不一定意味着存在因果关系。换句话说，相关性并不意味着因果关系。例如，冰淇淋销售散点图将显示与鲨鱼攻击次数的密切相关。这并不意味着购买冰淇淋会导致鲨鱼攻击，但它确实暗示了一种可能性，即无论是什么在驱动一个也在驱动另一个。在这种情况下，外面的温度导致人们在更热的日子里买冰淇淋或去游泳。

有一系列的统计测试和分析用来确定因果关系。在 Dataquest，你可以通过使用 [Python](https://www.dataquest.io/course/probability-statistics-intermediate/) 和 [R](https://www.dataquest.io/course/hypothesis-testing-r/) 来了解更多。

## 6.饼图

饼图也用于可视化一个分类变量如何被细分成不同的类别。然而，我们在使用这种图表时应该非常小心。那是因为他们通常很难准确地阅读。当两个或多个类别具有相似的值时，人眼几乎不可能区分哪个类别大于另一个类别，因为我们不太擅长估计角度。

下面是客户数据集中的`education_level`变量类别的饼图:

![](img/7ecd584f2794b6a847b0814927bf29ad.png)

如你所见，判断`Post-Graduate`或`Doctorate`是否代表更大的客户群是一个挑战。但是，如果我们在图表中添加百分比，就更容易区分了:

![](img/5de1fd8015e711ab910ffd6d983a6c85.png)

在这种情况下，条形图可能会使读者更容易比较不同类别的数值。

当饼图只有两三个类别时，效果最好，因此可以进行的比较较少。例如，下面是一个饼状图，显示了不满意客户和不满意客户的数量:

![](img/732129959f6cf4b076a9bc5d0ab3ca7a.png)

因为我们只有两个类别可以比较，所以确认哪个类别比另一个大要容易得多。

现在我们已经介绍了一些可视化技术，并且您已经有了绘制它们的数据集，如果您想开始使用像 PowerBI 这样的无代码工具，我们没有[一个](https://www.youtube.com/watch?v=G0v2DrqoTJs&t=2s)，而是[两个视频教程](https://www.youtube.com/watch?v=ukwBAi4ytms&t=1183s)来帮助您。

## 结论

虽然涵盖所有可能的数据可视化技术已经超出了本文的范围，但是我们在这里介绍的技术可能足以完成大多数基本的可视化工作。

如果你想让你的可视化更加有效和漂亮，这里有一个[教程完全集中在如何做到这一点](https://www.dataquest.io/blog/how-to-communicate-with-data/)。

然而，如果您有兴趣更深入地研究数据可视化并使用真实数据创建自己的图表，Dataquest 提供了多种课程，将带您从零到英雄，同时在以下方面创建美丽的可视化:

*   [Python](https://www.dataquest.io/path/data-analysis-and-visualization-with-python/)
*   [R](https://www.dataquest.io/path/data-visualization-with-r/)
*   [功率 BI](https://www.dataquest.io/path/analyzing-data-with-microsoft-power-bi-skill-path/) ，
*   [Tableau](https://www.dataquest.io/path/data-visualization-in-tableau/) ，还有更多。

一定要检查他们！