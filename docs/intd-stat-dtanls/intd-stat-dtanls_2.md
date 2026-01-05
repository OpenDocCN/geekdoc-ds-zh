# 2 单变量统计——案例研究社会人口统计报告

> 原文：[https://bookdown.org/conradziller/introstatistics/univariate-statistics-case-study-socio-demographic-reporting.html](https://bookdown.org/conradziller/introstatistics/univariate-statistics-case-study-socio-demographic-reporting.html)

## 2.1 简介

2020 年关于德国北莱茵-威斯特法伦州的社会人口统计报告继续了自 1992 年开始的国家社会报告。这些报告旨在向公众提供关于北莱茵-威斯特法伦州（NRW）的社会和人口状况及动态的信息。

社会指标指的是人口状况和发展（例如，人口数据，有移民背景的人的比例），经济（例如，失业率，GDP），健康，教育，住房，公共财政和社会不平等。

我们在这个案例研究中处理的报告及其基础数据可以通过 [https://www.sozialberichte.nrw.de](https://www.sozialberichte.nrw.de) 在线访问。

![社会报告封面](../Images/94ed8dc6172bfcc8f9239a2818b4c276.png)社会报告封面

基于社会人口统计报告数据进行的分析具有政治相关性。数据分析的见解可能为政治家和公务员提供需要通过政治措施解决的社会问题的信息。因此，从数据分析中得出的解释必须准确。在以下案例研究中，我们将更深入地研究数据，并使用单变量统计方法。请注意，以下的分析和解释旨在说明数据分析技术，并不反映任何政策建议。

我们将使用以下指标或*变量*（来源 [https://www.inkar.de](https://www.inkar.de)）：

+   人口密度

+   外国人比例

+   失业者比例

+   犯罪率

+   平均年龄

## 2.2 准备数据

在我们能够分析数据之前，它们必须被数据分析程序读取。R 使用许多所谓的“包”（即简化或便于特定数据分析部分的软件附加组件）。例如，包“readxl”允许我们读取存储在 Excel 格式的数据。另一个例子，包“ggplot2”允许我们创建打印就绪的图表。包通常具有特定的语法，我们在分析过程中会参考这些语法。

```r
library(readxl) # This command loads the package "readxl". In the markdown document, all required packages are installed in the background using the command "install.packages('packagename')". In the downloadable script files, the installation and activation of the packages is automated. 
 data_nrw <- read_excel("data/inkar_nrw.xls") # Reads data from Excel format; the directory where the data is to be found can be changed (e.g., "C:/user/documents/folderxyz/inkar_nrw.xls")
```

在读取文件“inkar_nrw.xlsx”之后，它被存储为 R 中的一个对象。我们现在可以处理这些数据了。作为第一步，我们打印数据框的前十行：

```r
kable(head(data_nrw, 10), format = "html", caption = "Selected social indicators NRW") %>%
 kable_paper() %>% #Font scheme of the table
 scroll_box(width = "100%", height = "100%") #Scrollable box
```

表 2.1：NRW 选择性社会指标

| kkziff | 区域 | 聚合 | 非国民 | 人口 | 面积 | 失业 | 平均年龄 | 犯罪率 | KN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 05111 | 杜塞尔多夫，城市 | 自治市 | 20.69 | 620523 | 217 | 7.77 | 42.65 | 9998.762 | 5111000 |
| 05112 | 杜伊斯堡，城市 | 自治市 | 21.86 | 495885 | 233 | 12.10 | 43.05 | 8640.908 | 5112000 |
| 05113 | 埃森，城市 | 自治市 | 16.87 | 582415 | 210 | 11.04 | 43.68 | 7472.201 | 5113000 |
| 05114 | 克雷菲尔德，城市 | 自治市 | 17.54 | 226844 | 138 | 11.11 | 44.28 | 8866.532 | 5114000 |
| 05116 | 莫恩格лад巴赫，城市 | 自治市 | 17.09 | 259665 | 170 | 10.02 | 43.74 | 8256.396 | 5116000 |
| 05117 | 路德维希港，城市 | 自治市 | 16.12 | 170921 | 91 | 8.31 | 45.23 | 5285.058 | 5117000 |
| 05119 | 奥伯豪森，城市 | 自治市 | 15.83 | 209566 | 77 | 10.76 | 44.52 | 7378.869 | 5119000 |
| 05120 | 雷姆舍德，城市 | 自治市 | 19.15 | 111516 | 75 | 7.97 | 44.37 | 5635.991 | 5120000 |
| 05122 | 索林根，刀城 | 自治市 | 17.13 | 159193 | 90 | 8.15 | 44.11 | 5933.624 | 5122000 |
| 05124 | 武珀塔尔，城市 | 自治市 | 20.81 | 355004 | 168 | 9.95 | 43.11 | 8059.420 | 5124000 |

数据包含以下变量，涵盖了北莱茵-威斯特法伦州（参考年份为2020年）的所有53个区和大型城市：

`kkziff` = 县（*Landkreis*）或城市（*kreisfreie Stadt*）的标识码

`area` = 县/城市名称

`nonnationall` = 外籍居民比例（%）

`population` = 人口数量

`flaeche` = 土地面积（平方公里）

`unemp` = 失业率（%）

`avage` = 人口平均年龄

`crimerate` = 每10万人中的犯罪率（案件数）

`KN` = 我们需要用于下面地图的备选id码

## 2.3 使用地图表示空间模式

表示空间结构数据的一种方法是地图的形式。

```r
# Read a so-called shape file for NRW which contains the polygones (e.g., district boundaries) necessary to build maps and merging with structural data
nrw_shp <- st_read("data/dvg2krs_nw.shp")
```

```r
## Reading layer `dvg2krs_nw' from data source 
##   `C:\Users\ziller\OneDrive - Universitaet Duisburg-Essen\Statistik Bookdown\IntroStats\IS\data\dvg2krs_nw.shp' 
##   using driver `ESRI Shapefile'
## Simple feature collection with 53 features and 8 fields
## Geometry type: MULTIPOLYGON
## Dimension:     XY
## Bounding box:  xmin: 280375 ymin: 5577680 xmax: 531791.2 ymax: 5820212
## Projected CRS: ETRS89 / UTM zone 32N
```

```r
nrw_shp$KN <- as.numeric(as.character(nrw_shp$KN))
nrw_shp <- nrw_shp %>%
 right_join(data_nrw, by = c("KN"))
 # Building the map using the "ggplot2" package
 ggplot() + 
 geom_sf(data = nrw_shp, aes(fill = unemp), color = 'gray', size = 0.1) + 
ggtitle("North-Rhine Westphalia, Germany") +
 guides(fill=guide_colorbar(title="Unemployment in %")) + 
 scale_fill_gradientn(colours=rev(magma(3)),
 na.value = "grey100", 
 ) 
```

![](../Images/3e8407e259dabffce793d31c3884598c.png)

* * *

**问题：** 地图传达了哪些信息？哪些信息是隐含的？

你的回答：

解决方案：

+   地图可以给出数据空间分布的良好第一印象。

+   然而，洞察力取决于进一步（隐含）的信息（“地图中间的高失业城市具体有哪些特点？哪些地区是乡村，哪些是城市？”）。

+   对地图进行一般性陈述以及比较地图（尤其是如果颜色化的基础断点不同）是困难的。

* * *

## 2.4 分布：对数据结构的初步了解

要了解数据的分布，可以使用直方图（例如，最小值或最大值是多少？观察值是否围绕分布的中心聚集，或者它们分布广泛？）。直方图显示了变量值范围内的观察频率。单个观察值（即，在我们的案例中是地区）在数据集中的出现方式并不重要。直方图根据观察值在特定变量上的值对观察值进行排序。显示的柱状高度与具有相应变量值的观察频率成比例。柱状宽度（也称为箱）代表一个值区间，并且可以自定义（即，几个宽柱包含比许多窄柱更多的观察值）。

```r
library([lattice](https://lattice.r-forge.r-project.org/))
 data_nrw$popdens <- data_nrw$population/data_nrw$flaeche # This generates the variable for the population density
 histogram( ~ popdens + unemp + avage + crimerate,
 breaks = 10, 
 type = "percent", 
 xlab = "",
 ylab = "Percent of observations", 
 layout = c(2,1),
 scales = list(relation="free"),
 col = 'grey',
 data = data_nrw)
```

![](../Images/0d609b47a0edeb9a05625752e8e7446c.png)![](../Images/a076dcbcd37b73dc7b7e46b0f0b5452b.png)

```r
## Alternative approach with the "hist"-function
#hist(data_nrw$unemp)
#hist(data_nrw$avage)
#hist(data_nrw$crimerate)
```

* * *

**问题：** 历史图可以识别数据的哪些特征？哪些方面仍然隐藏？

你的答案：

解答：

可以看到什么：数据的分布，包括大多数观察值所在的中心（s），偏斜模式，以及异常值。

隐藏的是什么：总结统计量的具体估计（例如，平均值、方差）。

* * *

## 2.5 平均值

一个变量的平均值也被称为集中趋势的度量。因此，它代表了测量特征（即变量）的典型（或可预期）值的良好第一印象。比较平均值和特定观察值的变量值可以提供有关观察值是否接近或远离平均值的信息。除了算术平均值外，还存在其他度量（例如，几何平均值），但在此我们不考虑它们。

**计算平均值时，将一个变量的所有观察值相加，然后将总和除以观察值的总数（n）。**

\(\bar x = \frac{1}{n}\sum_{i=1}^n x_i\) 或 \(\mu = \frac{1}{N}\sum_{i=1}^N x_i\)

\(\bar x\) 和 n 指的是我们正在处理的数据（例如，来自一个*样本*），\(\mu\) 和 N 指的是我们想要对其做出统计陈述的*总体*。（注意：这种区别在统计测试中变得相关，在那里我们使用随机样本来推断潜在更大的总体。）

让我们看看数据集中变量的平均值和其他信息：

```r
summary(data_nrw)
```

```r
##     kkziff              area             aggregat          nonnational   
##  Length:53          Length:53          Length:53          Min.   : 6.05  
##  Class :character   Class :character   Class :character   1st Qu.: 9.82  
##  Mode  :character   Mode  :character   Mode  :character   Median :12.19  
##                                                           Mean   :13.45  
##                                                           3rd Qu.:16.87  
##                                                           Max.   :21.86  
##    population         flaeche           unemp            avage      
##  Min.   : 111516   Min.   :  51.0   Min.   : 3.110   Min.   :40.97  
##  1st Qu.: 226844   1st Qu.: 170.0   1st Qu.: 5.740   1st Qu.:43.33  
##  Median : 308335   Median : 543.0   Median : 6.950   Median :44.15  
##  Mean   : 338218   Mean   : 643.6   Mean   : 7.432   Mean   :44.01  
##  3rd Qu.: 408662   3rd Qu.:1112.0   3rd Qu.: 8.850   3rd Qu.:44.62  
##  Max.   :1083498   Max.   :1960.0   Max.   :14.870   Max.   :45.81  
##    crimerate           KN             popdens      
##  Min.   : 3718   Min.   :5111000   Min.   : 116.3  
##  1st Qu.: 4906   1st Qu.:5166000   1st Qu.: 278.5  
##  Median : 5636   Median :5512000   Median : 784.7  
##  Mean   : 6244   Mean   :5506094   Mean   :1072.0  
##  3rd Qu.: 7568   3rd Qu.:5770000   3rd Qu.:1768.8  
##  Max.   :10500   Max.   :5978000   Max.   :3077.3
```

* * *

**问题：** 解释你选择的两种平均值。为什么变量“面积”没有显示平均值？

你的答案：

解答：

观察到的地区的平均失业率为7.4%。人口变量的平均值为338,218，这意味着平均而言，大约有338,218人居住在该地区。

只有对于度量或准度量变量（通常是具有五个或更多类别的有序变量）才能计算平均值。`area`是一个名义变量。

* * *

展示的表格包含的信息比平均值更多：

+   **中位数**是另一个中心度量（除了平均值和众数，即给定观察集出现最频繁的变量值）：

    +   中位数是观察到的变量值频率分布的中点（即它将有序观察数据分为两个相等的部分）。

    +   对于偏斜数据，通常更倾向于使用中位数，因为中位数对异常值（即极端值）不敏感。

+   离散度度量，如**范围**（即最大值 - 最小值）和**标准差**，都提供了有关数据分布的信息。

+   位置度量，如**四分位数**和四分位距，通常使用箱线图进行图形表示。

## 2.6 离散度度量：方差和标准差

暂时想象一下，我们不仅有了NRW的社会人口统计报告，还拥有另一个德国州的报告。比较两个州在犯罪率等县际特征上的差异，可能会发现它们具有完全相同的平均值。然而，州1的犯罪率分布极为不均，某些县的犯罪率特别高或低。相比之下，州2的犯罪率围绕平均值分布得更加均匀。确定数据的离散程度（或分散度）是重要的信息，对进一步的多项统计分析非常有用。这也可能具有实际意义，因为犯罪率的分散模式不同可能导致不同的犯罪对策措施。让我们从计算方差开始。

**方差是偏差平方的总和**。

\(\sigma^2 = \frac{\sum_{i=1}^n (x_i-\mu)^2}{N}\)（总体符号）\(s^2 = \frac{\sum_{i=1}^n (x_i-\bar x)^2}{n-1}\)（样本符号；如果我们应用统计推断并使用随机样本，我们需要在分母“n-1”处应用校正——称为贝塞尔校正）

**标准差将值重新转换回变量测量的尺度上，因此更容易解释，也更常用**。

\(s = \sqrt{\frac{\sum_{i=1}^n (x_i-\bar x)^2}{n-1}}\)

在这里，你可以找到变量`unemp`和`crimerate`分散的图形表示。请注意，x轴上观察值的偏差完全是随机的，以提高图表的可读性（否则，所有观察值都会像一串垂直的珍珠项链一样排列）。红色线代表平均值。

```r
s_unemp <- ggplot(data=data_nrw, aes(y=unemp, x=reorder(area, area), color=area)) +
 geom_jitter(height = 0) + 
 ggtitle("Unemployment rate 2020 in %")
s_unemp + geom_hline(yintercept=7.432, linetype="solid", color = "red", size=0.1) + theme(legend.position="bottom") + theme(legend.text = element_text(size=5)) + theme(axis.title.x=element_blank(),        axis.text.x=element_blank(),        axis.ticks.x=element_blank())
```

![图片](../Images/34533e1c6a4b7e5afda02c1a679b778c.png)

```r
s_crime <- ggplot(data=data_nrw, aes(y=crimerate, x=reorder(area, area), color=area)) +
 geom_jitter(height = 0) + 
 ggtitle("Crime rate 2020")
 s_crime + geom_hline(yintercept=6244, linetype="solid", color = "red", size=0.1)  + theme(legend.position="bottom") + theme(legend.text = element_text(size=5)) + theme(axis.title.x=element_blank(),        axis.text.x=element_blank(),        axis.ticks.x=element_blank())
```

![图片](../Images/3b051a03bee8c053a4ff845313f85bda.png)

* * *

**问题：**你能找出哪个区的失业率最高？哪个县的犯罪率最低？我们能从图表中推断出这两个变量中哪个的分散度更高？

你的答案：

解答：

很难将点的颜色与图例中的颜色匹配。Gelsenkirchen的失业率最高，达到14.9%。犯罪率最低的区是Lippe。

由于变量的缩放不同，无法从这个图表中推断出哪个变量的分散度更高。要做到这一点，我们需要计算所谓的变异系数（见下文）。

* * *

使用R计算的标准差、范围和变异系数（失业率）：

```r
sd(data_nrw$unemp)
```

```r
## [1] 2.477591
```

```r
range <- max(data_nrw$unemp, na.rm=TRUE) - min(data_nrw$unemp, na.rm=TRUE)
range 
```

```r
## [1] 11.76
```

```r
varcoef <- sd(data_nrw$unemp) / mean(data_nrw$unemp) * 100 #how much percent of the mean is the standard deviation?
varcoef
```

```r
## [1] 33.33815
```

使用R计算的标准差、最小/最大值和变异系数（犯罪率）：

```r
sd(data_nrw$crimerate)
```

```r
## [1] 1775.389
```

```r
range <- max(data_nrw$crimerate, na.rm=TRUE) - min(data_nrw$crimerate, na.rm=TRUE)
range 
```

```r
## [1] 6782.056
```

```r
varcoef <- sd(data_nrw$crimerate) / mean(data_nrw$crimerate) * 100
varcoef
```

```r
## [1] 28.43261
```

要比较不同尺度上测量的变量的分散度，我们不能使用方差或标准差（因为它们是在尺度的单位上测量的），而应使用变异系数（即标准差除以平均值再乘以100）。在这里，更高的值意味着数据的分散度更高。

> ***解释：变量失业的分布更加异质（变异系数 = 33.3）比犯罪率变量（变异系数 = 28.4）的分布。***

## 2.7 位置度量：箱线图和四分位距

四分位数将有序数据分为四个部分（Q1 到 Q4，其中 Q4 是观察值的最大值），中位数由 Q2 表示。因此，25% 的观察值低于或等于 Q1，75% 的观察值高于 Q1。Q3-Q1 = 四分位距（IQR），它反映了中间 50% 的观察值所在的范围。

![四分位数](../Images/7182b21d77cf421b22eb68ef86f85830.png)四分位数

失业率的四分位数和四分位距：

```r
quantile(data_nrw$unemp)
```

```r
##    0%   25%   50%   75%  100% 
##  3.11  5.74  6.95  8.85 14.87
```

```r
unemp.score.quart <- quantile(data_nrw$unemp, names = FALSE)
unemp.score.quart[4] - unemp.score.quart[2]
```

```r
## [1] 3.11
```

犯罪率的四分位数和四分位距：

```r
quantile(data_nrw$crimerate)
```

```r
##        0%       25%       50%       75%      100% 
##  3718.411  4906.122  5635.991  7568.376 10500.467
```

```r
crimerate.score.quart <- quantile(data_nrw$crimerate, names = FALSE)
crimerate.score.quart[4] - crimerate.score.quart[2]
```

```r
## [1] 2662.254
```

箱线图可以图形化地表示关键数值 Q1、Q2、Q3。胡须的上下端通常代表观察到的最小值和最大值。为了标记远离中位数（所谓的异常值）的观察值，胡须的最大宽度分别限制为 Q1 - 1.5 x IQR（对于下胡须）和 Q3 + 1.5 x IQR（对于上胡须）。异常值由胡须范围之外的点表示。最低和最高的异常值将分别标记最小/最大值。对于犯罪率，胡须的末端代表最小/最大值。对于失业率，我们发现一个异常值代表观察值的最大值（即，Gelsenkirchen 的失业率为 14.9%）。

此外，箱线图中单个部分越大，该区域的数据分散度就越大（这也可以给人留下数据分布偏斜的印象）。

```r
boxplot(data_nrw$unemp, 
 col = 'blue', 
 horizontal = FALSE,
 ylab = 'in %', 
 main = 'Unemployment rate')
```

![图片](../Images/194d9556a8307f91be6421ee44339ed3.png)

```r
boxplot(data_nrw$crimerate, 
 col = 'orange', 
 horizontal = FALSE,
 ylab = 'in cases per 100.000 inhab.', 
 main = 'Crime rate')
```

![图片](../Images/57838b3ff7c316b4139a4ef1e0528985.png)

## 2.8 结论

提供的例子说明了单变量描述性统计的一些基本原理。单变量统计对于了解数据的结构非常重要，这也许会为更复杂的统计方法提供信息。在进一步步骤方面，我们可以应用学到的方法并提问：哪些县受到犯罪最严重的困扰？哪些县的失业率最低，哪些县最高？人们大量迁移到或离开的地方在哪里？这可能会激发更多的问题，比如：为什么观察到的结构是这样的？这些问题涉及解释性分析。在以下内容中，我们将展望这一点，并在此过程中引用后续章节中详细解释的进一步主题。

## 2.9 展望：失业是否导致犯罪增加？

使用观察数据（例如，来自调查或官方记录）来测试“失业导致犯罪增加”等因果主张是英勇的，并且基于需要满足的各种假设。因果推断（即非实验数据）之所以如此困难，主要原因在于必须消除所有可能的替代解释。否则，我们无法确定失业是原因，还是与失业相关联的其他未观察到的现象。

对于因果推断，必须满足以下三个条件：

+   X（原因）和Y（效果）必须经验性地相互关联（例如，相互关联）。

+   X必须在时间上先于Y（例如，可以用面板数据映射）。

+   最重要的是：必须排除所有可能的替代解释（例如，通过随机实验或使用多元回归中的控制变量）。

我们将在回归分析的案例研究中重新审视这些假设。

### 2.9.1 双变量分析：相关性

获取相关方向印象的一个好方法是使用散点图。

```r
sc1 <- ggplot(data=data_nrw, aes(x = unemp, y = crimerate)) + 
 geom_point() + 
 xlab("Unemployment rate 2020 in %") +
 ylab("Crime rate")
sc1
```

![图片](../Images/079ab43b5a418b983371785f868044a9.png)

可以添加描述两个变量之间线性关系的线条：

```r
sc1 <- ggplot(data=data_nrw, aes(x = unemp, y = crimerate)) + 
 geom_point() + 
 geom_smooth(method = lm, se = FALSE) +
 xlab("Unemployment rate in %") +
 ylab("Crime rate")
sc1
```

![图片](../Images/9d6f6220261bee4f635dcd40dc62ea29.png)

> **解释：图中显示的观察数据和拟合线表明存在正相关。（失业率越高，犯罪率越高。）**

皮尔逊相关系数“r”量化了相关性：-1 = 完全负相关，0 = 无相关性，+1 = 完全正相关。

```r
vars <- c("unemp", "crimerate")
cor.vars <- data_nrw[vars]
rcorr(as.matrix(cor.vars))
```

```r
##           unemp crimerate
## unemp      1.00      0.72
## crimerate  0.72      1.00
## 
## n= 53 
## 
## 
## P
##           unemp crimerate
## unemp            0       
## crimerate  0
```

> **解释：在这种情况下，相关系数为r = 0.72，p值小于0.001。这意味着我们找到了强烈的正相关。相关性也是统计显著的（如果p值低于0.05，我们通常称之为统计显著的结果），这意味着我们可以相当肯定，结果不是由于偶然，而是可以解释为系统性的。**

### 2.9.2 使用控制变量的回归

相关性表明失业和犯罪之间存在正相关（且统计显著）的双变量关系。但这种关系是因果的吗？评估这种关系的一种方法是通过使用多元回归控制替代解释。我们使用人口密度作为替代解释。毕竟，失业和犯罪可能主要发生在城市环境中。如果是这样，部分相关性可能不是由于失业，而是由于人口密度，失业可以说部分“传递”了人口密度的影响（如果未包含在回归模型中）。让我们看看这一点。

+   模型1将失业作为解释变量

+   模型2包括失业**和**人口密度

```r
model1 <- lm(crimerate ~ 1 + unemp, data = data_nrw)
model2 <- lm(crimerate ~ 1 + unemp + popdens, data = data_nrw)
 stargazer(model1, model2, type = "text")
```

```r
## 
## =================================================================
##                                  Dependent variable:             
##                     ---------------------------------------------
##                                       crimerate                  
##                              (1)                    (2)          
## -----------------------------------------------------------------
## unemp                     517.460***             215.409**       
##                            (69.412)              (100.125)       
##                                                                  
## popdens                                           1.054***       
##                                                   (0.275)        
##                                                                  
## Constant                 2,398.594***           3,513.062***     
##                           (543.248)              (563.493)       
##                                                                  
## -----------------------------------------------------------------
## Observations                  53                     53          
## R2                          0.521                  0.630         
## Adjusted R2                 0.512                  0.615         
## Residual Std. Error  1,240.129 (df = 51)    1,101.398 (df = 50)  
## F Statistic         55.575*** (df = 1; 51) 42.557*** (df = 2; 50)
## =================================================================
## Note:                                 *p<0.1; **p<0.05; ***p<0.01
```

> ***解释：如果失业率上升一个百分点，每10万人中犯罪案件将增加517起（模型1）。这种关系在统计学上是显著的。如果我们现在在模型2中控制人口密度（从而保持人口密度的恒定影响），失业率每增加一个单位，犯罪案件将仅增加215起。人口密度也与犯罪呈正相关。这两个系数估计在p < 0.01的水平上都是统计学上显著的。因此，控制人口密度的影响似乎是一个好主意。我们可能现在更接近于失业对犯罪的真实影响。然而，我们并不确定这一点，因为其他可能的替代解释仍然存在（例如，年龄构成或地方社会政策的作用，以及个人层面的替代解释）***。

* * *

**免责声明：如果这个展望过于复杂，请不要担心。我们将在后续的案例研究中详细讲解所有步骤**。

* * *

[1 前言](index.html)[3 双变量统计 - 案例研究：美国总统选举](bivariate-statistics-case-study-united-states-presidential-election.html)
