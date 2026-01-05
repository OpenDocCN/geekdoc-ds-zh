# 3 双变量统计 - 案例研究：美国总统选举

> 原文：[https://bookdown.org/conradziller/introstatistics/bivariate-statistics-case-study-united-states-presidential-election.html](https://bookdown.org/conradziller/introstatistics/bivariate-statistics-case-study-united-states-presidential-election.html)

## 3.1 简介

研究2020年美国总统选举的结果，在理解当代美国政治以及选举行为研究方面具有深远的相关性。分析投票模式揭示了影响政治偏好的人口统计、意识形态和社会经济因素之间的相互作用。这些见解不仅有助于选举策略，而且加深了我们对社会分裂和分歧的理解。2020年选举还因唐纳德·特朗普及其同谋试图推翻选举结果而引人注目。这导致了动荡，并助长了1月6日美国国会大厦的袭击（例如，参见[https://www.britannica.com/event/January-6-U-S-Capitol-attack](https://www.britannica.com/event/January-6-U-S-Capitol-attack)和[https://statesuniteddemocracy.org/resources/doj-charges-trump/](https://statesuniteddemocracy.org/resources/doj-charges-trump/)）。

![](../Images/4214df26bd25da76fee2ee7bb72854f6.png)

来源：[https://unsplash.com/de/s/fotos/us-election](https://unsplash.com/de/s/fotos/us-electionl)

在选举之前，美国国家选举研究（ANES）的研究人员团队进行了一次大规模的代表性调查（基于美国人口的随机样本）来研究投票意向。我们将在这个案例研究中使用这些数据。我们特别感兴趣的是，哪些人更有可能支持唐纳德·特朗普而不是乔·拜登。为此，我们将以交叉表的形式呈现数据。我们还将通过调查美国各州平均教育和平均特朗普支持率之间的关系，更深入地研究相关性分析这一主题。

## 3.2 数据概述和描述性分析

数据是从CSV文件（逗号分隔值）中读取的。通过使用“head”命令，显示了数据的前六行。

```r
data_us <- read.csv("data/data_election.csv")
knitr::kable(head(data_us, 10), booktabs = TRUE,  caption = 'A table of the first 10 rows of the vote data.') %>%
 kable_paper() %>%
 scroll_box(width = "100%", height = "100%")
```

表3.1：表3.2：投票数据的前10行表格。

| vote | trump | education | age |
| --- | --- | --- | --- |
| trump | 1 | 3 | 46 |
| other | NA | 2 | 37 |
| biden | 0 | 1 | 40 |
| biden | 0 | 3 | 41 |
| trump | 1 | 4 | 72 |
| biden | 0 | 2 | 71 |
| trump | 1 | 3 | 37 |
| trump | 1 | 1 | 45 |
| refused / don’t know | NA | 3 | 43 |
| biden | 0 | 1 | 37 |

我们可以从这里看出，以下变量包含在数据集中：

`vote` = 2020年美国总统选举中的投票意向：“trump”，“biden”，“other”，“refused or don’t know”。

`trump` = `vote`变量的数值编码。1=`trump`，0=“biden”，NA=“other / refused / don’t know”。

`education` = 受访者最高的教育资格：1=无学位或高中，2=一些大学，3=副学士或学士学位，4=硕士学位或研究生学位，NA=未指定

`age` = 受访者的年龄（以年为单位）。

* * *

**问题：** 每个变量的测量尺度（名义、有序、指数）是什么？

你的回答：

解答：

`vote` 和 `trump` –> 名义（即，关于某事是否为真的信息，类别不能排序且没有数值意义）

`education` –> 有序（即，类别（或变量值）可以排序，关于变量值是否“更高”/“更多”或“更低”/“更少”的信息）

`age` –> 指数（即，排名变量值之间的间隔可以进行比较；例如，从20岁上升到22岁与从52岁上升到54岁是等效的）

* * *

### 3.2.1 频率表

下表显示了变量 `vote` 的值的观察频率。

```r
dim(data_us) #total number of cases
```

```r
## [1] 7272    4
```

```r
table(data_us$vote)
```

```r
## 
##                biden                other refused / don't know 
##                 3759                  274                  223 
##                trump 
##                 3016
```

我们可以看到，3,016名受访者选择了特朗普，而3,759人选择了拜登。几百个回答的差异。对于只有几个类别的变量，查看观察到的绝对数值可能是有信息的。然而，使用相对频率（即比例）通常更有信息量。让我们继续吧！

```r
prop.table(table(data_us$vote))
```

```r
## 
##                biden                other refused / don't know 
##           0.51691419           0.03767877           0.03066557 
##                trump 
##           0.41474147
```

特朗普支持者和拜登支持者群体之间的相对差异在百分比点上似乎相当微小。为了能够将调查数据中的数字与官方投票结果进行比较，我们需要排除“拒绝（例如，因为受访者表示不会投票）/不知道”这一类别。为此，我们将此类别指定为“缺失值”，

```r
data_us$vote[data_us$vote == "refused / don't know"] <- NA # Set "refused / don't know" to NA (i.e., missing), so that this category is no longer displayed in the table command
 prop.table(table(data_us$vote))
```

```r
## 
##      biden      other      trump 
## 0.53326713 0.03887076 0.42786211
```

* * *

**问题：** 调查结果如何预测实际选举结果？我们能从这一点上得出样本是否具有代表性的结论吗？

你的回答：

解答：

在2020年美国总统大选中，拜登以51.3%对46.9%的得票率战胜了特朗普。调查结果显示，在调查进行时，53.3%的人会选择拜登（相比之下，特朗普为42.8%）。虽然调查正确预测了拜登获胜，但结果与实际选举结果相差约2个百分点。从样本推断到潜在总体总是存在一些不确定性。我们可以量化不确定性范围，我们将在统计推断案例研究中这样做。除了统计不确定性的作用之外，一些其他因素（例如，政治竞选或投票日的情境环境）可能也对选举结果有贡献，并可能解释了调查结果与实际投票之间的差异。

调查的代表性与其特征相关，例如受访者的随机抽样以及被采访者的系统性无响应。对于美国选举研究，我们有充分的理由假设这些特征是既定的，结果能够代表美国人口。

* * *

我们下面也将排除在调查和选举中可能选择的“其他候选人”类别（这些候选人在选举中总共获得了不到 2% 的选票）。我们将在稍后展示如何重新编码变量。在此期间，我们可以依赖变量 `trump`，其中“拒绝/不知道”和“其他”类别已经设置为 NA（*不可用*，在数据分析中等于*缺失值*）。

```r
prop.table(table(data_us$trump))
```

```r
## 
##         0         1 
## 0.5548339 0.4451661
```

### 3.2.2 关于缺失数据和缺失值的说明

在调查数据中，缺失数据可能通过参与者的故意非响应或其他过程（例如，一个人搬到了新的地址或无资格投票）产生。如果生成缺失值的过程未知，这可能导致结果偏差。关于如何处理缺失数据的有用示例可以在以下链接找到：[https://cran.r-project.org/web/packages/finalfit/vignettes/missing.html](https://cran.r-project.org/web/packages/finalfit/vignettes/missing.html)

关于如何处理数据分析中的缺失值，请记住，在变量 `trump` 中，“不会投票”这一类别已被重新编码为 NA，这意味着向程序声明不将这些案例包含在统计分析中。请注意，对于 R 中的某些函数，必须明确排除缺失值（例如，使用选项“`na.rm=TRUE`”）。其他函数会自动执行此操作。

例如，`*table*` 命令会自动排除缺失值，如果您想显示缺失值的频率，您必须指定“`exclude=NULL`”。然而，`*mean*` 命令不会自动排除缺失值，因此我们必须指定选项“`na.rm=TRUE`”（即，“*NA remove* = TRUE”）。

## 3.3 “谁支持特朗普？” - 深入二元统计

在以下内容中，我们关注的是在美国总统选举中哪些群体更有可能支持特朗普。为此，我们关注受访者的教育和年龄。教育通过几个类别来衡量。因此，设置交叉表（或交叉表 = 包含两个变量的表）是合适的。相比之下，年龄以年为单位衡量，因此包含更多的类别。这将使涉及年龄的交叉表变得相当繁琐。我们可以将年龄重新编码为几个年龄类别，并在交叉表中使用它们，或者我们可以比较投票意向的年龄分布。下面，我们将探讨这两种选项。

### 3.3.1 教育与特朗普投票

首先，我们感兴趣的是人们的投票选择是否在不同教育群体中有所不同。如果是这样，我们就可以说教育和投票意图在统计学上相关或相互关联。如果不是，两者将是独立的。按照惯例，“要解释的变量”（也称为结果或因变量）由交叉表的行表示，而“解释变量”（也称为因果因素或自变量）由列表示。我们计算列百分比以进行解释（即，每个单元格的观察数除以每列的总观察数。因此，每列的所有比例加起来等于100%）。在我们的案例中，因果关系非常明确。我们有充分的理由相信（如果有的话）`教育`决定了投票意图（`trump`），而不是相反（即，投票意图会导致教育程度更高或更低）。请注意，在许多其他情况下，因果关系并不那么明确，我们可以自由选择哪个变量放在行中，哪个放在列中。

```r
tab_xtab(var.row = data_us$trump, var.col = data_us$education, show.col.prc = TRUE, show.obs = TRUE, show.summary = FALSE) #show.col.prc = TRUE displays column percentages
```

| trump | 教育 | 总计 |
| --- | --- | --- |
| 1 | 2 | 3 | 4 |
| 0 | 567 45.2 % | 687 50.1 % | 1483 55.4 % | 974 70.5 % | 3711 55.5 % |
| 1 | 687 54.8 % | 685 49.9 % | 1195 44.6 % | 407 29.5 % | 2974 44.5 % |
| 总计 | 1254 100 % | 1372 100 % | 2678 100 % | 1381 100 % | 6685 100 % |

对于列百分比，解释首先从解释变量的两个类别（相邻或极端）中选择，然后比较结果的相关类别。

> ***解释：在低学历群体（教育程度=1）中，54.8%的受访者支持特朗普，而在高学历群体（教育程度=4）中，只有29.5%的人支持特朗普。从这个比较中，我们已可以得出结论，教育和特朗普支持之间有很强的相关性。教育程度越高，对特朗普的偏好平均来说越低。***

注意，在变量之间没有系统性关系的情况下，相邻或相对列值之间的差异将是零或接近零（两个值将紧密对应于44.5%的总比例）。

对解释的重要补充将是（1）量化实证关系的强度（例如，弱、中、强）和（2）对我们可以有多大的信心，认为在样本中发现的实证关系反映了样本所抽取的总体属性。为此目的有相应的程序，我们将在本案例研究的末尾回到这些程序。在此之前，让我们暂时回到交叉表。在这里，我们看到与上面相同的交叉表，但行百分比代替了列百分比。

```r
tab_xtab(var.row = data_us$trump, var.col = data_us$education, show.row.prc = TRUE, show.obs = TRUE, show.summary = FALSE) #show.row.prc = TRUE displays row percentages
```

| trump | 教育 | 总计 |
| --- | --- | --- |
| 1 | 2 | 3 | 4 |
| 0 | 567 15.3 % | 687 18.5 % | 1483 40 % | 974 26.2 % | 3711 100 % |
| 1 | 687 23.1 % | 685 23 % | 1195 40.2 % | 407 13.7 % | 2974 100 % |
| 总计 | 1254 18.8 % | 1372 20.5 % | 2678 40.1 % | 1381 20.7 % | 6685 100 % |

* * *

**问题：** 从带有行百分比的交叉表中可以得出哪些结论？

**你的回答：**

**解答：**

行百分比是每个单元格的观测数除以每行的总观测数（每行的所有比例相加等于100%）。解释将与之前类似，但我们现在关注的是比较列变量选定值对应的行变量的不同类别。让我们选择教育 = 1：在那些不支持特朗普（=拜登支持者）的人中，有15.3%的人受教育程度低，而在特朗普支持者中，有23.1%的人受教育程度低。7.8个百分点的差异表明，特朗普支持和低教育程度系统性地相关。因此，带有行百分比的交叉表分析的结论与列百分比分析的结论相当（假设我们调整了解释方案）。此外，在这种情况下，行百分比还显示了投票变量每个类别中教育变量的描述性*分布*。然而，通常很难一眼看出分布的比较。

* * *

作为第三种可能性，以下是一个带有总计百分比的交叉表：

```r
tab_xtab(var.row = data_us$trump, var.col = data_us$education, show.cell.prc = TRUE, show.obs = TRUE, show.summary = FALSE) #show.cell.prc = TRUE displays cell or total percentages
```

| trump | 教育 | 总计 |
| --- | --- | --- |
| 1 | 2 | 3 | 4 |
| 0 | 567 8.5 % | 687 10.3 % | 1483 22.2 % | 974 14.6 % | 3711 55.6 % |
| 1 | 687 10.3 % | 685 10.2 % | 1195 17.9 % | 407 6.1 % | 2974 44.5 % |
| 总计 | 1254 18.8 % | 1372 20.5 % | 2678 40.1 % | 1381 20.7 % | 6685 100 % |

* * *

**问题：** 从带有单元格百分比的交叉表中可以得出哪些结论？

**你的回答：**

**解答：**

**总计百分比**是每个单元格的观测数除以总观测数。这显示了两个变量值组合的相对频率。目标是描述性理解案例的分布。（注意：这在应用研究中很少使用。）

* * *

### 3.3.2 年龄与特朗普投票

由于“年龄”变量以年为单位记录，因此有大量的类别，交叉表会相当繁琐，因此信息量较少。因此，我们首先创建显示`trump`两个类别中年龄变量分布的直方图，然后比较它们。

```r
histogram( ~ age | trump ,
 breaks = 10, 
 ylim=c(0,12),
 type = "percent", 
 main = "Left: trump=0 (in support of Biden), Right: trump=1 (in support of Trump)",
 ylab = "Percent of observations",
 xlab = "Age in years",
 layout = c(2,1),
 scales = list(relation="free"),
 col = 'grey',
 data = data_us)
```

![](../Images/857c066cee6269727bd9fb220f29bdb0.png)

* * *

**问题：** 从图表中可以推断出什么？

**你的回答：**

**解答：**

支持拜登的受访者（左侧）的年龄分布比支持特朗普的受访者（右侧）的分布更均匀。后者也向左偏斜更多，其中心更偏向右侧，这表明支持特朗普的受访者平均年龄比支持拜登的受访者大。

* * *

此外，以下是按投票组计算的年龄的均值和标准差。

**拜登支持者的年龄均值、中位数和标准差：**

```r
mean(data_us$age[data_us$trump == 0], na.rm=TRUE)
```

```r
## [1] 51.22253
```

```r
median(data_us$age[data_us$trump == 0], na.rm=TRUE)
```

```r
## [1] 52
```

```r
sd(data_us$age[data_us$trump == 0], na.rm=TRUE)
```

```r
## [1] 17.22161
```

**特朗普支持者的年龄均值、中位数和标准差：**

```r
mean(data_us$age[data_us$trump == 1], na.rm=TRUE)
```

```r
## [1] 54.2871
```

```r
median(data_us$age[data_us$trump == 1], na.rm=TRUE)
```

```r
## [1] 56
```

```r
sd(data_us$age[data_us$trump == 1], na.rm=TRUE)
```

```r
## [1] 16.51148
```

通过比较两组的平均值和中间值，我们发现支持特朗普的群体比支持拜登的群体年龄更大。拜登群体中的标准差（可以认为是受访者年龄与平均值之间的平均偏差）略大于特朗普群体的标准差（16.5年），为17.2年。这意味着拜登群体中受访者的年龄分布略大于特朗普群体。

在下一步中，我们将年龄变量重新编码为“年轻”（1）、“中年”（2）和“老年”（3）这三个类别。然后我们使用带有列百分比的交叉表。

```r
data_us <- data_us %>% 
 mutate(age_cat = 
 case_when(age <= 35 ~ 1, 
 age > 36 & age <= 59 ~ 2, 
 age > 59 ~ 3)) 
 tab_xtab(var.row = data_us$trump, var.col = data_us$age_cat, show.col.prc = TRUE, show.obs = TRUE, show.summary = FALSE)
```

| 特朗普 | 年龄类别 | 总计 |
| --- | --- | --- |
| 1 | 2 | 3 |
| 0 | 838 63.1 % | 1373 55.7 % | 1375 52.4 % | 3586 55.9 % |
| 1 | 491 36.9 % | 1092 44.3 % | 1250 47.6 % | 2833 44.1 % |
| 总计 | 1329 100 % | 2465 100 % | 2625 100 % | 6419 100 % |

* * *

**问题：** 解释交叉表。可以得出什么结论？为什么我们不一定直接对度量变量进行分类并使用重新编码的变量在交叉表中？

你的答案：

解答：

在查看年轻受访者（年龄类别 = 1）这一类别时，36.9%的人更喜欢特朗普作为总统，而老年受访者（年龄类别 = 3）中，有47.6%的人选择特朗普。这种10.7个百分点的差异表明了年龄与特朗普支持之间的系统性（正面）经验关系：老年受访者倾向于比年轻选民更喜欢特朗普。

不总是推荐对度量变量进行分类的原因是，分类过程涉及信息损失。通常，信息越多越好，因为它会产生更详细的结果。请注意，这里有一些例外，例如，为了简化对经验关系的描述。

* * *

这里是使用堆叠条形图可视化交叉表的一种方法。第一张图对应于带有列百分比的交叉表，第二张图对应于带有行百分比的交叉表。

```r
# Age variable will be subdivided into the three dummy variables "young", "middle-aged", and "old"
data_us <- data_us %>% 
 mutate(age_cat_nom = 
 case_when(age <= 35 ~ "young", 
 age > 36 & age <= 59 ~ "middle-aged", 
 age > 59 ~ "old")) 
 data_us$age_cat_nom  <- factor(data_us$age_cat_nom, levels=c('young', 'middle-aged', 'old'))

# Recoding of electoral participation 
data_us <- data_us %>% 
 mutate(trump_nom =
 case_when(trump == 0 ~ "Favoring Biden", 
 trump == 1 ~ "Favoring Trump"))
 data_us_counted <- data_us  %>%  count(age_cat_nom, trump_nom)
data_us_counted <- subset(data_us_counted, !is.na(data_us_counted$age_cat_nom)) # removal of missing values
data_us_counted <- subset(data_us_counted, !is.na(data_us_counted$trump_nom)) # removal of missing values
 ggplot(data_us_counted, aes(fill=trump_nom, y=n, x=age_cat_nom)) +
 geom_bar(position="fill", stat="identity") +
 scale_y_continuous(labels = scales::percent) +
 labs(x = "Age groups",
 y = "Shares",
 fill = "Voting intention US presidential election") +
 theme_minimal()
```

![](../Images/3a708062e933d62613551ae8fef2f89c.png)

```r
ggplot(data_us_counted, aes(fill=age_cat_nom, y=n, x=trump_nom)) +
 geom_bar(position="fill", stat="identity") +
 scale_y_continuous(labels = scales::percent) +
 coord_flip() +
 labs(x = "Voting intention US presidential election",
 y = "Shares",
 fill = "Age groups") +
 theme_minimal()
```

![](../Images/252d77af5e2b86b359df6cdc8a279436.png)

## 3.4 散点图和相关性

散点图和相关性适合展示两个变量之间的关系。在以下内容中，我们使用美国各州对特朗普的平均支持率（%）(`perc_trump`)作为要解释的变量（即结果或因变量）。以下地图展示了各州投票模式的具体变化。作为解释变量，我们使用该州拥有高水平教育（硕士或研究生）的人口比例（%）(`perc_higheducation`)。这两个变量都来自ANES研究的个人层面数据，并已汇总到州级（即，每个地区的平均值存储在一个新的数据集中）。

![美国地区总统选举结果](../Images/df8ff4c600f60647e1fd13df8a34e904.png)美国地区总统选举结果

来源：[https://www.governing.com/assessments/what-painted-us-so-indelibly-red-and-blue](https://www.governing.com/assessments/what-painted-us-so-indelibly-red-and-blue)

接下来，读取数据并显示前六行：

```r
data_states <- read.csv("data/data_states.csv")
knitr::kable(
 head(data_states, 10), booktabs = TRUE,
 caption = 'A table of the first 10 rows of the regional vote data.') %>%
 kable_paper() %>%
 scroll_box(width = "100%", height = "100%")
```

表3.3：表3.4：区域投票数据的前10行表。

| 州 | 特朗普支持率 | 高等教育比例 |
| --- | --- | --- |

|

1.  南达科他州

| 0.7333334 | 0.0588235 |
| --- | --- |

|

1.  内布拉斯加州

| 0.5416667 | 0.0816327 |
| --- | --- |

|

1.  南卡罗来纳州

| 0.5000000 | 0.0840336 |
| --- | --- |

|

1.  北达科他州

| 0.7619048 | 0.0869565 |
| --- | --- |

|

1.  蒙大拿州

| 0.5238095 | 0.0869565 |
| --- | --- |

|

1.  阿肯色州

| 0.7142857 | 0.1132075 |
| --- | --- |

|

1.  路易斯安那州

| 0.6022728 | 0.1170213 |
| --- | --- |

|

1.  威斯康星州

| 0.5031056 | 0.1180124 |
| --- | --- |

|

1.  阿拉斯加

| 0.7500000 | 0.1250000 |
| --- | --- |

|

1.  犹他州

| 0.5526316 | 0.1250000 |
| --- | --- |

变量 `state` 识别地区。

### 3.4.1 散点图

为了对两个变量之间的经验关系有一个初步的了解，我们使用散点图。每个点代表美国50个州和华盛顿特区的其中一个。请注意，惯例是，因变量显示在y轴上，而自变量显示在x轴上。

```r
sc1 <- ggplot(data=data_states, aes(x = perc_higheducation, y = perc_trump)) + 
 geom_point() + 
 xlab("Share of highly-educated persons in %") +
 ylab("Support of Trump in %") +
 scale_y_continuous(labels = scales::percent) +
 scale_x_continuous(labels = scales::percent)
 sc1
```

![](../Images/74133aa4cb55fe6fb934060c4fbadb92.png)

* * *

**问题：** 从图中可以推断出什么？

你的答案：

解答：

这种模式表明两个变量之间存在**负**相关关系。如果一个地区的受过高等教育的人的比例增加（在x轴上向右移动），我们会看到这与对特朗普的支持度降低（y轴上的得分降低）有关。

* * *

### 3.4.2 相关性

在下一步中，我们将通过图形方法来展示如何确定相关性。为此，我们绘制了一条垂直线和一条水平线，它们通过代表黑色点的观察云来表示变量的平均值。我们看到点主要集中在左上象限（=高教育比例且特朗普支持率低）和右下象限（=高教育比例且特朗普支持率低）。这表明较高的平均教育水平往往与较低的特朗普平均支持率相关。换句话说：相关性是负的。

```r
#Obtaining means for each variable
mean(data_states$perc_trump, na.rm=TRUE)
```

```r
## [1] 0.470922
```

```r
mean(data_states$perc_higheducation, na.rm=TRUE)
```

```r
## [1] 0.1927645
```

```r
#Scatter plot using percentages as scale units
sc2 <- ggplot(data=data_states, aes(x = perc_higheducation, y = perc_trump)) + 
 geom_point() + 
 xlab("Share of highly-educated persons in %") +
 ylab("Support of Trump in %") +
 scale_y_continuous(labels = scales::percent) +
 scale_x_continuous(labels = scales::percent) +
 geom_hline(yintercept=0.470922, linetype="dashed", color = "red", size=1) +
 geom_vline(xintercept=0.1927645, linetype="dashed", color = "red", size=1)
sc2
```

![](../Images/f144be86d1419e69cadcc105fd9d9f43.png)

为了量化这种关系，我们可以计算协方差，然后是相关系数（=标准化协方差，调整到-1到+1的值域）。广泛使用的皮尔逊相关系数如下所示：

\(r=\frac{Covariance(x,y)}{SD(x)*SD(y)}\)

\(r=\frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i-\bar{y})^2}}\)

分子 \({\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}\) 特别重要。它代表 **协方差**，这是一个 *未标准化* 的 x 和 y 之间关联的度量，因为其值很大程度上取决于 x 和 y 的测量尺度。同时，它还用于确定经验关联的方向。为此，将观测值分类到均值以上/以下象限是有用的。位于两个均值以上的观测值（即右上角）具有正号，并作为乘积贡献于“正”协方差或相关系数。对于位于左下象限的观测值也是如此，因为在这里两个差异都具有负号，这再次作为乘积贡献于正协方差（从而相关系数）。相反，位于左上和右下象限的观测值贡献于负协方差（从而相关系数）。如果观测值在所有象限中均匀分布，正负乘积会相互抵消，这将导致协方差（从而相关系数）为零。从协方差到相关系数的步骤是将协方差通过 x 和 y 的标准差进行 *标准化*，从而消除了协方差所测量的尺度单位，并将度量范围转换为 -1 到 +1。

这里是相关系数可能情景的概述：

![不同的相关系数](../Images/def5619515dfea46c9ae284c25c25731.png)不同的相关系数

来源：[https://en.wikipedia.org/wiki/Correlation#/media/File:Correlation_examples2.svg](https://en.wikipedia.org/wiki/Correlation#/media/File:Correlation_examples2.svg)

展示相关系数（从左到右）为 -1、-0.8、-0.4、0、0.4、0.8 和 1 的散点图。

在下一步中，计算了高学历人群比例与特朗普支持者比例之间的相关系数，并绘制了表示它们关系的线条。这条线本质上以回归斜率的形式描绘了双变量关联，它始终沿着相关系数的方向运行。然而，其解释与相关系数不同。

```r
sc3 <- ggplot(data=data_states, aes(x = perc_higheducation, y = perc_trump)) + 
 geom_point() + 
 xlab("Share of highly-educated persons in %") +
 ylab("Support of Trump in %") +
 scale_y_continuous(labels = scales::percent) +
 scale_x_continuous(labels = scales::percent) +
 geom_smooth(method = lm, se = FALSE)
sc3
```

![](../Images/d49a117c960c96522d1a4ec77b2e09f0.png)

```r
vars <- c("perc_higheducation", "perc_trump")
cor.vars <- data_states[vars]
rcorr(as.matrix(cor.vars))
```

```r
##                    perc_higheducation perc_trump
## perc_higheducation               1.00      -0.69
## perc_trump                      -0.69       1.00
## 
## n= 51 
## 
## 
## P
##                    perc_higheducation perc_trump
## perc_higheducation                     0        
## perc_trump          0
```

在相关系数和回归系数的情况下，解释总是从提及 x（或 **自变量**）开始，它对 y（或 **因变量**）产生作用。

+   **相关系数**：“如果 x 增加，那么 y 增加/减少（取决于相关系数的符号），平均而言。相关系数 *r* 的大小表明一个 *弱/中等/强* 的经验关联。”（经验法则：+/- 0.1 *弱*，+/- 0.3 *中等*，+/- 0.5 或更高 *强* 关联）

+   **回归系数**：“如果x增加一个单位，那么y平均增加/减少（取决于回归系数的符号）*系数*个单位。”所提到的具体系数估计值在回归输出中给出。通过回归系数的标准化，可以确定经验关系的强度是强、中还是弱。请注意，回归分析及其解释的细节包含在回归分析的案例研究中。

+   相关性是协方差的标准形式，它具有直观可解释的优点（-1到+1），无需参考两个变量所测量的基础尺度。然而，付出的代价是它隐藏了y在x增加一个单位时变化的幅度（而不是表示在形成线性关系时观察值聚集的密度）。二元回归量化了x增加1时y预期的增加量。因此，我们需要对x和y之间的协方差进行标准化（或“校正”），以适应x的测量尺度。这是通过将协方差除以x的方差来实现的：\(b=\frac{Covariance(x,y)}{Var(x)}\)

> ***示例中相关系数的解释：高教育程度人群与特朗普支持率之间的相关性为负。如果高教育程度人群的份额增加，那么平均而言，特朗普的支持率会下降。相关系数-0.69表明这两个变量之间存在强烈的经验关系。***

## 3.5 卡方检验和Cramér的V值

现在我们回到交叉表，以及如何陈述显示变量之间是否存在相关性，如果存在，那么相关性有多强。为此，我们再次回顾个体层面的调查数据，并显示`education`和`trump`之间的交叉表。

```r
tab_xtab(var.row = data_us$trump, var.col = data_us$education, show.col.prc = TRUE, show.obs = TRUE, show.summary = TRUE) #setting show.summary = TRUE displays summary statistics at the bottom of the table
```

| trump | education | Total |
| --- | --- | --- |
| 1 | 2 | 3 | 4 |
| 0 | 567 45.2 % | 687 50.1 % | 1483 55.4 % | 974 70.5 % | 3711 55.5 % |
| 1 | 687 54.8 % | 685 49.9 % | 1195 44.6 % | 407 29.5 % | 2974 44.5 % |
| Total | 1254 100 % | 1372 100 % | 2678 100 % | 1381 100 % | 6685 100 % |
| χ²=196.388 · df=3 · Cramer’s V=0.171 · p=0.000 |

已经非常明显，低学历的人比高学历的人更有可能支持特朗普（而高学历的人更倾向于支持拜登）。为了量化这种经验关联，我们可以计算 Cramér 的 V 值。如果涉及名义变量（此处：`trump`），此度量量化经验关系。它基于独立性卡方检验。卡方值表示实际观察值与理论分布的偏差程度，其中所有观察值在交叉表的单元格中均匀分布（相对于边际分布）。测量的观察值与理论分布的偏差越大，卡方值越高，随后 Cramér 的 V 值也越高，该值已根据表格大小进行归一化，范围在 0（=无关联）和 1（=极其强烈的关联）之间。

卡方检验的交叉表公式如下：\(\chi^2= \sum{\frac{(O-E)^2}{E}}\)

\(O\) 指的是观察频率，\(E\) 指的是预期频率。预期频率是通过将每个单元格的边际频率（“总数”）相乘，然后除以案例总数（\(E = \frac{R\times C}{n}\)）获得的。对于上表中的第一个单元格，这将等于 (1254*3711)/6685=696.12。因此，理论上第一个单元格中应有 696 个观察值，而实际观察到的是 567。相当不同。这通常是通过统计软件对每个单元格进行操作，并按照公式进行求和。

Cramér 的 V 值是基于卡方并针对不同大小的交叉表进行调整的度量。它由以下公式给出：\(V=\sqrt{\frac{\chi^2}{n\times (m-1)}}\)，其中 \(n\) 是观察值的数量，\(M\) 是具有较少类别的变量（此处 `trump` 有 2 个类别）的类别数（或变量值）。

关于解释，以下指南适用：

+   Cramér 的 V 值介于 0 到 1 之间，不可能有负值

+   因此，Cramér 的 V 值不提供关于关联是正还是负的信息，我们必须从表中找出这一点（例如，通过计算列值之间的百分比差异 –> 见上文）

+   关于关系大小的经验法则：

    +   在一个小型表格（例如，2x2）中：介于 0.1 和 0.3 之间 –> 弱关联，超过 0.3 且小于 0.5 –> 中等关联，超过 0.5 –> 强关联

    +   在一个大型表格（例如，5x5）中：介于 0.05 和 0.15 之间 –> 弱关联，超过 0.15 且小于 0.25 –> 中等关联，超过 0.25 –> 强关联

在 R 中，可以使用以下命令计算卡方和 Cramér 的 V 值：

```r
chisq.test(data_us$trump, data_us$education)
```

```r
## 
##  Pearson's Chi-squared test
## 
## data:  data_us$trump and data_us$education
## X-squared = 196.39, df = 3, p-value < 2.2e-16
```

```r
cramerV_tabelle <- table(data_us$trump, data_us$education)
cramerV(cramerV_tabelle)
```

```r
## Cramer V 
##   0.1714
```

* * *

**问题：**在这种情况下，如何解释 Cramér 的 V 值？

你的答案：

解决方案：

> ***解释：变量 `education` 和 `trump` 之间存在关联。根据小表的规则，这种关联的值为 0.17，属于较弱。在我们的案例中，根据上述解释，我们知道这种关联是负面的（教育水平越高，对特朗普的平均支持率越低）。***

* * *

* * *

**问题：为什么我们没有计算皮尔逊相关系数？**

你的答案：

解答：

皮尔逊相关系数需要（准）度量变量，并且具有多个类别。对于有序变量，我们有如肯德尔tau系数或斯皮尔曼相关系数等工具（有关不同相关类型的更多信息，请参阅此处：[https://ademos.people.uic.edu/Chapter22.html](https://ademos.people.uic.edu/Chapter22.html)）。如果涉及名义变量，我们使用如卡方系数（Cramér’s V）等系数。

* * *

[2 单变量统计 - 案例研究：社会人口统计报告](univariate-statistics-case-study-socio-demographic-reporting.html)[4 统计推断 - 案例研究：对政府的满意度](statistical-inference---case-study-satisfaction-with-government.html)
