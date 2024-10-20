# Kaggle 入门:房价竞争

> 原文：<https://www.dataquest.io/blog/kaggle-getting-started/>

May 5, 2017

成立于 2010 年的 [Kaggle](https://www.kaggle.com) 是一个数据科学平台，用户可以在这里分享、合作和竞争。Kaggle 的一个关键功能是“竞赛”，它为用户提供了在真实世界数据上练习的能力，并与国际社会一起测试他们的技能。

本指南将教你如何接近并参加 Kaggle 竞赛，包括探索数据、创建和设计特征、构建模型以及提交预测。我们将使用 [Python 3](https://www.python.org/) 和 [Jupyter 笔记本](https://jupyter.org/)。

![Getting Started With Kaggle](img/d1e7475f6844aac84e21cb4e208d1879.png)

## 竞争

我们将完成[房价:高级回归技术](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)竞赛。

我们将按照以下步骤成功完成 Kaggle 竞赛分包:

*   获取数据
*   探索数据
*   设计和转换特征和目标变量
*   建立一个模型
*   做出并提交预测

## 步骤 1:获取数据并创建我们的环境

我们需要为比赛获取[数据](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)。特性的描述和其他一些有用的信息包含在一个文件中，这个文件有一个明显的名字`data_description.txt`。

下载数据并将其保存到一个文件夹中，您可以在其中保存比赛所需的一切。

我们先来看一下`train.csv`的数据。在我们训练了一个模型之后，我们将使用`test.csv`数据进行预测。

首先，导入 [Pandas](https://www.dataquest.io/blog/pandas-python-tutorial/) ，这是一个用 Python 处理数据的极好的库。接下来我们将导入 [Numpy](https://www.dataquest.io/blog/numpy-tutorial-python/) 。

```py
 import pandas as pd
import numpy as np
```

我们可以用熊猫读入 csv 文件。`pd.read_csv()`方法从一个 csv 文件创建一个数据帧。

```py
 train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
```

让我们检查一下数据的大小。

```py
 print ("Train data shape:", train.shape)
print ("Test data shape:", test.shape)
```

```py
Train data shape: (1460, 81)
Test data shape: (1459, 80)
```

我们看到`test`只有 80 列，而`train`有 81 列。当然，这是因为测试数据不包括最终销售价格信息！

接下来，我们将使用`DataFrame.head()`方法查看几行。

```py
train.head()
```

|  | 身份 | MSSubClass | MSZoning | 地段临街 | 地段面积 | 街道 | 胡同 | LotShape | 陆地等高线 | 公用事业 | 日志配置 | 风景 | 附近 | 情况 | 情况 | 建筑类型 | HouseStyle | 总体平等 | 总体代码 | 年造的 | YearRemodAdd | 屋顶样式 | 室友 | 外部 1st | 外部第二 | MasVnrType | 马斯夫纳雷亚 | exteequal | 外部 | 基础 | BsmtQual | BsmtCond | bsm 曝光 | BsmtFinType1 | BsmtFinSF1 | BsmtFinType2 | BsmtFinSF2 | BsmtUnfSF | 总计 BsmtSF | 加热 | 加热 QC | 中央空气 | 与电有关的 | 1stFlrSF | 2ndFlrSF | 低质量 FinSF | GrLivArea | BsmtFullBath | 半沐浴 | 全浴 | 半浴 | BedroomAbvGr | KitchenAbvGr | KitchenQual | TotRmsAbvGrd | 功能的 | 壁炉 | 壁炉曲 | GarageType | 车库 | 车库整理 | 车库汽车 | 车库区域 | GarageQual | GarageCond | 铺面车道 | WoodDeckSF | OpenPorchSF | 封闭门廊 | 3SsnPorch | 纱窗阳台 | 游泳池区域 | PoolQC | 栅栏 | 杂项功能 | 米沙尔 | mos old | YrSold | 标度类型 | 销售条件 | 销售价格 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Zero | one | Sixty | RL | Sixty-five | Eight thousand four hundred and fifty | 安排 | 圆盘烤饼 | 车辆注册号 | 单板层积材 | AllPub | 里面的 | Gtl | CollgCr | 标准 | 标准 | 1Fam | 2 历史 | seven | five | Two thousand and three | Two thousand and three | 三角形建筑部分 | 康普什 | 乙烯树脂 | 乙烯树脂 | BrkFace | One hundred and ninety-six | 钆 | 钽 | PConc | 钆 | 钽 | 不 | GLQ | Seven hundred and six | 未溶化的 | Zero | One hundred and fifty | Eight hundred and fifty-six | 加萨 | 前夫;前妻;前男友;前女友 | Y | SBrkr | Eight hundred and fifty-six | Eight hundred and fifty-four | Zero | One thousand seven hundred and ten | one | Zero | Two | one | three | one | 钆 | eight | 典型 | Zero | 圆盘烤饼 | 附上 | Two thousand and three | RFn | Two | Five hundred and forty-eight | 钽 | 钽 | Y | Zero | Sixty-one | Zero | Zero | Zero | Zero | 圆盘烤饼 | 圆盘烤饼 | 圆盘烤饼 | Zero | Two | Two thousand and eight | 陆军部(War Department) | 常态 | Two hundred and eight thousand five hundred |
| one | Two | Twenty | RL | Eighty | Nine thousand six hundred | 安排 | 圆盘烤饼 | 车辆注册号 | 单板层积材 | AllPub | FR2 | Gtl | 维肯 | Feedr | 标准 | 1Fam | 1 历史 | six | eight | One thousand nine hundred and seventy-six | One thousand nine hundred and seventy-six | 三角形建筑部分 | 康普什 | 金属 | 金属 | 没有人 | Zero | 钽 | 钽 | CBlock | 钆 | 钽 | 钆 | ALQ | Nine hundred and seventy-eight | 未溶化的 | Zero | Two hundred and eighty-four | One thousand two hundred and sixty-two | 加萨 | 前夫;前妻;前男友;前女友 | Y | SBrkr | One thousand two hundred and sixty-two | Zero | Zero | One thousand two hundred and sixty-two | Zero | one | Two | Zero | three | one | 钽 | six | 典型 | one | 钽 | 附上 | One thousand nine hundred and seventy-six | RFn | Two | Four hundred and sixty | 钽 | 钽 | Y | Two hundred and ninety-eight | Zero | Zero | Zero | Zero | Zero | 圆盘烤饼 | 圆盘烤饼 | 圆盘烤饼 | Zero | five | Two thousand and seven | 陆军部(War Department) | 常态 | One hundred and eighty-one thousand five hundred |
| Two | three | Sixty | RL | Sixty-eight | Eleven thousand two hundred and fifty | 安排 | 圆盘烤饼 | IR1 | 单板层积材 | AllPub | 里面的 | Gtl | CollgCr | 标准 | 标准 | 1Fam | 2 历史 | seven | five | Two thousand and one | Two thousand and two | 三角形建筑部分 | 康普什 | 乙烯树脂 | 乙烯树脂 | BrkFace | One hundred and sixty-two | 钆 | 钽 | PConc | 钆 | 钽 | 锰 | GLQ | Four hundred and eighty-six | 未溶化的 | Zero | Four hundred and thirty-four | Nine hundred and twenty | 加萨 | 前夫;前妻;前男友;前女友 | Y | SBrkr | Nine hundred and twenty | Eight hundred and sixty-six | Zero | One thousand seven hundred and eighty-six | one | Zero | Two | one | three | one | 钆 | six | 典型 | one | 钽 | 附上 | Two thousand and one | RFn | Two | Six hundred and eight | 钽 | 钽 | Y | Zero | forty-two | Zero | Zero | Zero | Zero | 圆盘烤饼 | 圆盘烤饼 | 圆盘烤饼 | Zero | nine | Two thousand and eight | 陆军部(War Department) | 常态 | Two hundred and twenty-three thousand five hundred |
| three | four | Seventy | RL | Sixty | Nine thousand five hundred and fifty | 安排 | 圆盘烤饼 | IR1 | 单板层积材 | AllPub | 角落 | Gtl | 克劳福德 | 标准 | 标准 | 1Fam | 2 历史 | seven | five | One thousand nine hundred and fifteen | One thousand nine hundred and seventy | 三角形建筑部分 | 康普什 | Wd Sdng | Wd Shng | 没有人 | Zero | 钽 | 钽 | 贝尔蒂尔 | 钽 | 钆 | 不 | ALQ | Two hundred and sixteen | 未溶化的 | Zero | Five hundred and forty | Seven hundred and fifty-six | 加萨 | 钆 | Y | SBrkr | Nine hundred and sixty-one | Seven hundred and fifty-six | Zero | One thousand seven hundred and seventeen | one | Zero | one | Zero | three | one | 钆 | seven | 典型 | one | 钆 | 荷兰的 | One thousand nine hundred and ninety-eight | 未溶化的 | three | Six hundred and forty-two | 钽 | 钽 | Y | Zero | Thirty-five | Two hundred and seventy-two | Zero | Zero | Zero | 圆盘烤饼 | 圆盘烤饼 | 圆盘烤饼 | Zero | Two | Two thousand and six | 陆军部(War Department) | 反常的 | One hundred and forty thousand |
| four | five | Sixty | RL | Eighty-four | Fourteen thousand two hundred and sixty | 安排 | 圆盘烤饼 | IR1 | 单板层积材 | AllPub | FR2 | Gtl | 诺里奇 | 标准 | 标准 | 1Fam | 2 历史 | eight | five | Two thousand | Two thousand | 三角形建筑部分 | 康普什 | 乙烯树脂 | 乙烯树脂 | BrkFace | Three hundred and fifty | 钆 | 钽 | PConc | 钆 | 钽 | 音像的 | GLQ | Six hundred and fifty-five | 未溶化的 | Zero | Four hundred and ninety | One thousand one hundred and forty-five | 加萨 | 前夫;前妻;前男友;前女友 | Y | SBrkr | One thousand one hundred and forty-five | One thousand and fifty-three | Zero | Two thousand one hundred and ninety-eight | one | Zero | Two | one | four | one | 钆 | nine | 典型 | one | 钽 | 附上 | Two thousand | RFn | three | Eight hundred and thirty-six | 钽 | 钽 | Y | One hundred and ninety-two | Eighty-four | Zero | Zero | Zero | Zero | 圆盘烤饼 | 圆盘烤饼 | 圆盘烤饼 | Zero | Twelve | Two thousand and eight | 陆军部(War Department) | 常态 | Two hundred and fifty thousand |

我们的文件夹中应该有`data dictionary`可供比赛使用。你也可以在这里找到它[。](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)

以下是您将在数据描述文件中找到的内容的简要版本:

*   `SalePrice` —以美元为单位的房产销售价格。这是你试图预测的目标变量。
*   `MSSubClass` —建筑类
*   `MSZoning` —一般分区分类
*   `LotFrontage` —与物业相连的街道的线性英尺数
*   `LotArea` —以平方英尺为单位的批量
*   `Street` —道路通道的类型
*   `Alley` —小巷通道的类型
*   `LotShape` —物业的总体形状
*   `LandContour` —财产的平坦度
*   `Utilities` —可用的公用设施类型
*   `LotConfig` —批次配置

诸如此类。

比赛要求你预测每套房子的最终价格。
在这一点上，我们应该开始思考我们对房价的了解，[艾姆斯，爱荷华](https://en.wikipedia.org/wiki/Ames,_Iowa)，以及我们可能期望在这个数据集中看到什么。

查看数据，我们看到了我们预期的特征，比如`YrSold`(房屋最后出售的年份)和`SalePrice`。其他的我们可能没有预料到，比如`LandSlope`(房屋所在土地的坡度)和`RoofMatl`(用于建造屋顶的材料)。稍后，我们将不得不决定如何处理这些和其他特性。

我们希望在项目的探索阶段进行一些绘图，并且我们也需要将该功能导入到我们的环境中。绘图使我们能够可视化数据的分布，检查异常值，并看到我们可能会忽略的其他模式。我们将使用 [Matplotlib](https://matplotlib.org/) ，一个流行的可视化库。

```py
 import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)
```

## 步骤 2:探索数据和工程特征

挑战在于预测房屋的最终售价。该信息存储在`SalePrice`栏中。我们试图预测的值通常被称为**目标变量**。

我们可以使用`Series.describe()`来获取更多信息。

```py
train.SalePrice.describe()
```

```py
 count      1460.000000
mean     180921.195890
std       79442.502883
min       34900.000000
25%      129975.000000
50%      163000.000000
75%      214000.00000
0max      755000.000000
Name: SalePrice, dtype: float64
```

给你任何系列的更多信息。`count`显示系列中的总行数。对于数值数据，`Series.describe()`也给出了`mean`、`std`、`min`和`max`的值。

在我们的数据集中，房屋的平均销售价格接近于`$180,000`，大部分价格都在`$130,000`到`$215,000`的范围内。

接下来，我们将检查[偏斜度](https://mathworld.wolfram.com/Skewness.html)，这是对值分布形状的度量。

执行回归时，当目标变量有偏差时，有时对其进行对数变换是有意义的。这样做的一个原因是为了提高数据的线性度。虽然论证超出了本教程的范围，但是更多信息可以在[这里](https://en.wikipedia.org/wiki/Data_transformation_%28statistics%29)找到。

重要的是，最终模型生成的预测也将进行对数转换，因此我们需要稍后将这些预测转换回原始形式。

`np.log()`将转换变量，`np.exp()`将反转转换。

我们使用`plt.hist()`来绘制`SalePrice`的直方图。请注意，右边的分布有一个较长的尾部。分布是正偏的。

```py
 print ("Skew is:", train.SalePrice.skew())
plt.hist(train.SalePrice, color='blue')
plt.show()
```

```py
 Skew is: 1.88287575977
```

![Kaggle Tutorial](img/f8fecce7de79f19cb64ac7a26aa917b1.png)

现在我们使用`np.log()`来转换`train.SalePric`并再次计算偏斜度，同时重新绘制数据。更接近 0 的值意味着我们已经改善了数据的偏斜度。我们可以直观地看到，数据将更像一个[正态分布](https://en.wikipedia.org/wiki/Normal_distribution)。

```py
 target = np.log(train.SalePrice)
print ("Skew is:", target.skew())
plt.hist(target, color='blue')
plt.show()
```

```py
Skew is: 0.121335062205
```

![](img/e293d90516e874d0dc1d97dcb9538514.png)

现在我们已经转换了目标变量，让我们考虑一下我们的特性。首先，我们将检查数字特征并绘制一些图表。`.select_dtypes()`方法将返回匹配指定数据类型的列的子集。

### 使用数字要素

```py
 numeric_features = train.select_dtypes(include=[np.number])
numeric_features.dtypes
```

```py
 Id                 int64
MSSubClass         int64
LotFrontage      float64
LotArea            int64
OverallQual        int64
OverallCond        int64
YearBuilt          int64
YearRemodAdd       int64
MasVnrArea       float64
BsmtFinSF1         int64
BsmtFinSF2         int64
BsmtUnfSF          int64
TotalBsmtSF        int64
1stFlrSF           int64
2ndFlrSF           int64
LowQualFinSF       int64
GrLivArea          int64
BsmtFullBath       int64
BsmtHalfBath       int64
FullBath           int64
HalfBath           int64
BedroomAbvGr       int64
KitchenAbvGr       int64
TotRmsAbvGrd       int64
Fireplaces         int64
GarageYrBlt      float64
GarageCars         int64
GarageArea         int64
WoodDeckSF         int64
OpenPorchSF        int64
EnclosedPorch      int64
3SsnPorch          int64
ScreenPorch        int64
PoolArea           int64
MiscVal            int64
MoSold             int64
YrSold             int64
SalePrice          int64
dtype: object
```

`DataFrame.corr()`方法显示列之间的相关性(或关系)。我们将检查特征和目标之间的相关性。

```py
 corr = numeric_features.corr()
print (corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
print (corr['SalePrice'].sort_values(ascending=False)[-5:]) 
```

```py
 SalePrice      1.000000
OverallQual    0.790982
GrLivArea      0.708624
GarageCars     0.640409
GarageArea     0.623431
Name: SalePrice, dtype: float64 
YrSold          -0.028923
OverallCond     -0.077856
MSSubClass      -0.084284
EnclosedPorch   -0.128578
KitchenAbvGr    -0.135907
Name: SalePrice, dtype: float64 
```

前五个特征是最[正相关的](https://en.wikipedia.org/wiki/Correlation_and_dependence)和`SalePrice`，而接下来的五个是最负相关的。

让我们更深入地了解一下`OverallQual`。我们可以使用`.unique()`方法来获得唯一值。

```py
train.OverallQual.unique()
```

```py
array([ 7,  6,  8,  5,  9,  4, 10,  3,  1,  2])
```

`OverallQual`数据是 1 到 10 区间内的整数值。

我们可以创建一个[数据透视表](https://en.wikipedia.org/wiki/Pivot_table)来进一步研究`OverallQual`和`SalePrice`之间的关系。熊猫医生展示了如何完成这项任务。我们设定了`index='OverallQual'`和`values='SalePrice'`。我们选择看这里的`median`。

```py
quality_pivot = train.pivot_table(index='OverallQual',
                  values='SalePrice', aggfunc=np.median) 
```

```py
quality_pivot
```

```py
OverallQual
1      50150
2      60000
3      86250
4     108000
5     133000
6     160000
7     200141
8     269750
9     345000
10    432390
Name: SalePrice, dtype: int64
```

为了帮助我们更容易地可视化这个数据透视表，我们可以使用`Series.plot()`方法创建一个条形图。

```py
 quality_pivot.plot(kind='bar', color='blue')
plt.xlabel('Overall Quality')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)plt.show() 
```

![](img/6346add7cc8d2fa686f0c65f3e1ed2b2.png)

请注意，中间销售价格随着整体质量的提高而严格提高。

接下来，让我们使用`plt.scatter()`生成一些散点图，并可视化地面生活区`GrLivArea`和`SalePrice`之间的关系。

```py
 plt.scatter(x=train['GrLivArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Above grade (ground) living area square feet')
plt.show() 
```

![](img/7a79846befb398d12598a00ecc32d922.png)

乍一看，我们发现居住面积的增加对应着价格的上涨。我们将为`GarageArea`做同样的事情。

```py
 plt.scatter(x=train['GarageArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()
```

![](img/547ee14445201bc0ed853fe9ebce0106.png)

注意有很多家的`Garage Area`用`0`表示，表示没有车库。我们稍后将转换其他特性来反映这一假设。也有一些[离群值](https://en.wikipedia.org/wiki/Outlier)。异常值会影响回归模型，使我们的估计回归线远离真实的总体回归线。因此，我们将从数据中删除这些观察值。剔除异常值是一门艺术，也是一门科学。有许多处理异常值的技术。

我们将创建一个删除了一些离群值的新数据帧。

```py
train = train[train['GarageArea'] < 1200]
```

让我们再看一看。

```py
 plt.scatter(x=train['GarageArea'], y=np.log(train.SalePrice))
plt.xlim(-200,1600) # This forces the same scale as before
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()
```

![](img/068d924940aeda6a0a61f736a2c7b8a0.png)

### 处理空值

接下来，我们将检查空值或缺失值。

我们将创建一个数据框架来查看顶部的空列。将`train.isnull().sum()`方法链接在一起，我们返回每一列中 null 值的一系列计数。

```py
 nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
nulls 
```

|  | 空计数 |
| --- | --- |
| 特征 |  |
| --- | --- |
| PoolQC | One thousand four hundred and forty-nine |
| 杂项功能 | One thousand four hundred and two |
| 胡同 | One thousand three hundred and sixty-four |
| 栅栏 | One thousand one hundred and seventy-four |
| 壁炉曲 | Six hundred and eighty-nine |
| 地段临街 | Two hundred and fifty-eight |
| GarageCond | Eighty-one |
| 车库类型 | Eighty-one |
| 车库 | Eighty-one |
| 车库整理 | Eighty-one |
| GarageQual | Eighty-one |
| bsm 曝光 | Thirty-eight |
| BsmtFinType2 | Thirty-eight |
| BsmtFinType1 | Thirty-seven |
| BsmtCond | Thirty-seven |
| BsmtQual | Thirty-seven |
| 马斯夫纳雷亚 | eight |
| MasVnrType | eight |
| 与电有关的 | one |
| 公用事业 | Zero |
| YearRemodAdd | Zero |
| MSSubClass | Zero |
| 基础 | Zero |
| 外部 | Zero |
| exteequal | Zero |

[文档](https://www.kaggle.com/dejavu23/house-prices-eda-to-ml-beginner/data)可以帮助我们理解缺失的值。在`PoolQC`的情况下，该列指的是池质量。当`PoolArea`为`0`时，池质量为`NaN`，否则没有池。我们可以在许多与车库相关的栏目中找到类似的关系。

让我们来看看另一个专栏，`MiscFeature`。我们将使用`Series.unique()`方法返回唯一值的列表。

```py
print ("Unique values are:", train.MiscFeature.unique())
```

```py
Unique values are: [nan 'Shed' 'Gar2' 'Othr' 'TenC']
```

我们可以使用文档来找出这些值的含义:

```py
 MiscFeature: Miscellaneous feature not covered in other categories  

   Elev Elevator   
   Gar2 2nd Garage (if not described in garage section)
   Othr Other
   Shed Shed (over 100 SF)
   TenC Tennis Court
   NA   None
```

这些值描述了房子是否有超过 100 平方英尺的小屋、第二个车库等等。我们以后可能会用到这些信息。为了在处理缺失数据时做出最佳决策，收集领域知识非常重要。

### 争论非数字特征

现在让我们考虑非数字特性。

```py
 categoricals = train.select_dtypes(exclude=[np.number])
categoricals.describe()
```

|  | MSZoning | 街道 | 胡同 | LotShape | 陆地等高线 | 公用事业 | 日志配置 | 风景 | 附近 | 情况 | 情况 | 建筑类型 | HouseStyle | 屋顶样式 | 室友 | 外部 1st | 外部第二 | MasVnrType | exteequal | 外部 | 基础 | BsmtQual | BsmtCond | bsm 曝光 | BsmtFinType1 | BsmtFinType2 | 加热 | 加热 QC | 中央空气 | 与电有关的 | KitchenQual | 功能的 | 壁炉曲 | 车库类型 | 车库整理 | GarageQual | GarageCond | 铺面车道 | PoolQC | 栅栏 | 杂项功能 | 标度类型 | 销售条件 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 数数 | One thousand four hundred and fifty-five | One thousand four hundred and fifty-five | Ninety-one | One thousand four hundred and fifty-five | One thousand four hundred and fifty-five | One thousand four hundred and fifty-five | One thousand four hundred and fifty-five | One thousand four hundred and fifty-five | One thousand four hundred and fifty-five | One thousand four hundred and fifty-five | One thousand four hundred and fifty-five | One thousand four hundred and fifty-five | One thousand four hundred and fifty-five | One thousand four hundred and fifty-five | One thousand four hundred and fifty-five | One thousand four hundred and fifty-five | One thousand four hundred and fifty-five | One thousand four hundred and forty-seven | One thousand four hundred and fifty-five | One thousand four hundred and fifty-five | One thousand four hundred and fifty-five | One thousand four hundred and eighteen | One thousand four hundred and eighteen | One thousand four hundred and seventeen | One thousand four hundred and eighteen | One thousand four hundred and seventeen | One thousand four hundred and fifty-five | One thousand four hundred and fifty-five | One thousand four hundred and fifty-five | One thousand four hundred and fifty-four | One thousand four hundred and fifty-five | One thousand four hundred and fifty-five | Seven hundred and sixty-six | One thousand three hundred and seventy-four | One thousand three hundred and seventy-four | One thousand three hundred and seventy-four | One thousand three hundred and seventy-four | One thousand four hundred and fifty-five | six | Two hundred and eighty-one | Fifty-three | One thousand four hundred and fifty-five | One thousand four hundred and fifty-five |
| 独一无二的 | five | Two | Two | four | four | Two | five | three | Twenty-five | nine | eight | five | eight | six | seven | Fifteen | Sixteen | four | four | five | six | four | four | four | six | six | six | five | Two | five | four | seven | five | six | three | five | five | three | three | four | four | nine | six |
| 顶端 | RL | 安排 | Grvl | 车辆注册号 | 单板层积材 | AllPub | 里面的 | Gtl | 名称 | 标准 | 标准 | 1Fam | 1 历史 | 三角形建筑部分 | 康普什 | 乙烯基 | 乙烯树脂 | 没有人 | 钽 | 钽 | PConc | 钽 | 钽 | 不 | 未溶化的 | 未溶化的 | 加萨 | 前夫;前妻;前男友;前女友 | Y | SBrkr | 钽 | 典型 | 钆 | 附上 | 未溶化的 | 钽 | 钽 | Y | 前夫;前妻;前男友;前女友 | MnPrv | 棚 | 陆军部(War Department) | 常态 |
| 频率 | One thousand one hundred and forty-seven | One thousand four hundred and fifty | Fifty | Nine hundred and twenty-one | One thousand three hundred and nine | One thousand four hundred and fifty-four | One thousand and forty-eight | One thousand three hundred and seventy-eight | Two hundred and twenty-five | One thousand two hundred and fifty-seven | One thousand four hundred and forty-one | One thousand two hundred and sixteen | Seven hundred and twenty-two | One thousand one hundred and thirty-nine | One thousand four hundred and thirty | Five hundred and fourteen | Five hundred and three | Eight hundred and sixty-three | Nine hundred and five | One thousand two hundred and seventy-eight | Six hundred and forty-four | Six hundred and forty-seven | One thousand three hundred and six | Nine hundred and fifty-one | Four hundred and twenty-eight | One thousand two hundred and fifty-one | One thousand four hundred and twenty-three | Seven hundred and thirty-seven | One thousand three hundred and sixty | One thousand three hundred and twenty-nine | Seven hundred and thirty-three | One thousand three hundred and fifty-five | Three hundred and seventy-seven | Eight hundred and sixty-seven | Six hundred and five | One thousand three hundred and six | One thousand three hundred and twenty-one | One thousand three hundred and thirty-five | Two | One hundred and fifty-seven | Forty-eight | One thousand two hundred and sixty-six | One thousand one hundred and ninety-six |

`count`列表示非空观察值的数量，而`unique`表示唯一值的数量。`top`是最常出现的值，顶部值的频率由`freq`显示。

对于其中的许多特性，我们可能想要使用[一键编码](https://www.quora.com/What-is-one-hot-encoding-and-when-is-it-used-in-data-science)来利用信息进行建模。
One-hot 编码是一种将分类数据转换为数字的技术，因此模型可以理解某个特定的观察值是否属于某个类别。

### 转换和工程特征

在转换特征时，重要的是要记住，在拟合模型*之前应用于训练数据的任何转换都必须*应用于测试数据。

我们的模型期望来自`train`集合的特征的形状与来自`test`集合的特征的形状相匹配。这意味着在处理`train`数据时发生的任何特征工程应再次应用于`test`器械组。

为了演示这是如何工作的，考虑一下`Street`数据，它表明是否有`Gravel`或`Paved`道路通往该地产。

```py
 print ("Original: \n") 
print (train.Street.value_counts(), "\n")
```

```py
 Original: 
Pave    1450
Grvl       5
Name: Street, dtype: int64
```

在`Street`列中，唯一值是`Pave`和`Grvl`，它们描述了通往该地产的道路类型。在训练组中，只有 5 个家庭有砾石通道。我们的模型需要数字数据，所以我们将使用一键编码将数据转换为布尔列。

我们创建一个名为`enc_street`的新列。`pd.get_dummies()`方法将为我们处理这个问题。

如前所述，我们需要对`train`和`test`数据都这样做。

```py
 train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
test['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
```

```py
 print ('Encoded: \n') 
print (train.enc_street.value_counts())
```

```py
 Encoded: 

1    1450
0       5
Name: enc_street, dtype: int64
```

价值观一致。我们设计了我们的第一个功能！[特征工程](https://en.wikipedia.org/wiki/Feature_engineering)是使数据的特征适用于机器学习和建模的过程。当我们将`Street`特征编码成一列布尔值时，我们设计了一个特征。

让我们尝试设计另一个功能。我们将通过构建和绘制一个数据透视表来查看`SaleCondition`，就像我们对`OverallQual`所做的那样。

```py
 condition_pivot = train.pivot_table(index='SaleCondition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show() 
```

![](img/35389be492715fbd2136a102de2938a7.png)

请注意，`Partial`的销售价格中值明显高于其他产品。我们将把它编码为一个新特性。我们选择所有`SaleCondition`等于`Patrial`的房子，并赋值`1`，否则赋值`0`。

遵循我们在上面`Street`中使用的类似方法。

```py
 def encode(x):
 return 1 if x == 'Partial' else 0
train['enc_condition'] = train.SaleCondition.apply(encode)
test['enc_condition'] = test.SaleCondition.apply(encode) 
```

让我们探索一下这个新特性的剧情。

```py
 condition_pivot = train.pivot_table(index='enc_condition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Encoded Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show() 
```

![](img/bba7290604bec21844e88f78a4afb10f.png)

这看起来很棒。您可以继续使用更多功能来提高模型的最终性能。

在为建模准备数据之前，我们需要处理缺失的数据。我们将用平均值填充缺失值，然后将结果赋给`data`。这是一种[插补](https://en.wikipedia.org/wiki/Interpolation)的方法。`DataFrame.interpolate()`方法使这变得简单。

这是处理缺失值的一种快速而简单的方法，并且可能不会使模型在新数据上达到最佳性能。处理丢失的值是建模过程的重要部分，在这里创造力和洞察力可以发挥很大的作用。这是本教程中您可以扩展的另一个领域。

```py
data = train.select_dtypes(include=[np.number]).interpolate().dropna()
```

检查是否所有列都有 0 空值。

```py
sum(data.isnull().sum() != 0)
```

```py
0
```

## 第三步:建立一个线性模型

让我们执行最后的步骤，为建模准备数据。我们将特性和目标变量分开建模。我们将特性分配给`X`，目标变量分配给`y`。我们使用如上所述的`np.log()`来转换模型的 y 变量。`data.drop([features], axis=1)`告诉熊猫我们想要排除哪些列。出于显而易见的原因，我们不会将`SalePrice`包括在内，而`Id`只是一个与`SalePrice`没有关系的指标。

```py
 y = np.log(train.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1)
```

让我们对数据进行分区并开始建模。
我们将使用 [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) 中的`train_test_split()`函数来创建一个训练集和一个拒绝集。以这种方式划分数据允许我们评估我们的模型在以前从未见过的数据上可能如何执行。如果我们根据所有测试数据训练模型，将很难判断是否发生了[过度拟合](https://en.wikipedia.org/wiki/Overfitting)。

`train_test_split()`返回四个对象:

*   `X_train`是我们用于训练的特征子集。
*   `X_test`是将成为我们“坚持”集的子集，我们将用它来测试模型。
*   `y_train`是对应于`X_train`的目标变量`SalePrice`。
*   `y_test`是对应于`X_test`的目标变量`SalePrice`。

第一参数值`X`表示预测器数据集，而`y`是目标变量。接下来，我们设定`random_state=42`。这提供了可重复的结果，因为 sci-kit learn 的`train_test_split`将随机划分数据。`test_size`参数告诉该函数在`test`分区中应该有多大比例的数据。在本例中，大约 33%的数据专用于拒绝集。

```py
 from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                          X, y, random_state=42, test_size=.33)
```

### 开始建模

我们将首先创建一个[线性回归](https://en.wikipedia.org/wiki/Linear_regression)模型。首先，我们实例化模型。

```py
 from sklearn import linear_model
lr = linear_model.LinearRegression()
```

接下来，我们需要拟合模型。首先实例化模型，然后拟合模型。模型拟合是一个因不同类型的模型而异的过程。简而言之，我们正在估计预测值和目标变量之间的关系，这样我们就可以根据新数据做出准确的预测。

我们用`X_train`和`y_train`拟合模型，用`X_test`和`y_test`评分。`lr.fit()`方法将对我们传递的特征和目标变量进行线性回归拟合。

```py
model = lr.fit(X_train, y_train)
```

### 评估性能并可视化结果

现在，我们要评估模型的性能。每个竞赛可能会对分包商的评价有所不同。在这次竞赛中，Kaggle 将使用[均方根误差(RMSE)](https://en.wikipedia.org/wiki/Root-mean-square_deviation) 来评估我们的分包合同。我们还会看看[的 r 平方值](https://en.wikipedia.org/wiki/Coefficient_of_determination)。r 平方值衡量数据与拟合回归线的接近程度。它取 0 到 1 之间的值，1 表示目标中的所有差异都可以用数据来解释。一般来说，较高的 r 平方值意味着更好的拟合。

默认情况下，`model.score()`方法返回 r 平方值。

```py
print ("R^2 is: \n", model.score(X_test, y_test))
```

```py
R^2 is:
  0.888247770926
```

这意味着我们的特征解释了目标变量中大约 89%的方差。点击上面的链接了解更多信息。

接下来，我们来考虑`rmse`。为此，使用我们建立的模型对测试数据集进行预测。

```py
predictions = model.predict(X_test)
```

给定一组预测值，`model.predict()`方法将返回一个预测列表。拟合模型后使用`model.predict()`。

`mean_squared_error`函数接受两个数组并计算`rmse`。

```py
 from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, predictions))
```

```py
RMSE is:
  0.0178417945196
```

解释这个值比 r 平方值更直观。RMSE 测量我们的预测值和实际值之间的距离。

我们可以用散点图来直观地观察这种关系。

```py
 actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.7,
            color='b') #alpha helps to show overlapping data
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()
```

![](img/ae531bcde82e477bfebe09dac90e7bdf.png)

如果我们的预测值与实际值相同，这个图将是直线`y=x`，因为每个预测值`x`将等于每个实际值`y`。

### 尝试改进模型

接下来，我们将尝试使用[脊正则化](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)来减少不太重要的特征的影响。岭正则化是缩小不太重要的特征的回归系数的过程。

我们将再次实例化该模型。脊正则化模型采用参数`alpha`，该参数控制正则化的强度。

我们将通过循环几个不同的 alpha 值进行实验，看看这会如何改变我们的结果。

```py
 for i in range (-2, 3):
    alpha = 10**i
    rm = linear_model.Ridge(alpha=alpha)
    ridge_model = rm.fit(X_train, y_train)
    preds_ridge = ridge_model.predict(X_test)

    plt.scatter(preds_ridge, actual_values, alpha=.75, color='b')
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title('Ridge Regularization with alpha = {}'.format(alpha))
    overlay = 'R^2 is: {}\nRMSE is: {}'.format(
                    ridge_model.score(X_test, y_test),
                    mean_squared_error(y_test, preds_ridge))
    plt.annotate(s=overlay,xy=(12.1,10.6),size='x-large')
    plt.show() 
```

![](img/3edb82ebb697172f389f221562533fd3.png)![](img/84cc3f4aa61bc09211b6e1a64e7e00f0.png)![](img/1708996d9c3a5d257af4f0bf39c82ba5.png)![](img/baf6af091d879d8bf552f83fcdf5bedc.png)![](img/ecdcc4edd56162c2aa9eee1efab07a91.png)

这些型号的性能几乎与第一种型号相同。在我们的例子中，调整 alpha 并没有实质性地改进我们的模型。当您添加更多功能时，规范化会很有帮助。添加更多功能后，重复此步骤。

## 第四步:转租

我们需要为`test.csv`数据集中的每个观察值创建一个包含预测的`SalePrice`的`csv`。

我们将登录我们的 Kaggle 帐户，并前往[转租页面](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/submit)进行转租。
我们将使用`DataFrame.to_csv()`创建一个 csv 来提交。
第一列必须包含测试数据中的 ID。

```py
 sublesson = pd.DataFrame()
sublesson['Id'] = test.Id 
```

现在，像我们上面所做的那样，从模型的测试数据中选择特性。

```py
 feats = test.select_dtypes(
        include=[np.number]).drop(['Id'], axis=1).interpolate()
```

接下来，我们生成我们的预测。

```py
predictions = model.predict(feats)
```

现在我们将把预测转换成正确的形式。记住，要反转`log()`，我们要做`exp()`。
所以我们将把`np.exp()`应用到我们的预测中，因为我们之前已经取了对数。

```py
final_predictions = np.exp(predictions)
```

看区别。

```py
 print ("Original predictions are: \n", predictions[:5], "\n")
print ("Final predictions are: \n", final_predictions[:5])
```

```py
 Original predictions are:
  [ 11.76725362  11.71929504  12.07656074  12.20632678  12.11217655] 
Final predictions are:
  [ 128959.49172586  122920.74024358  175704.82598102  200050.83263756  182075.46986405]
```

让我们分配这些预测，并检查一切看起来都很好。

```py
 sublesson['SalePrice'] = final_predictions
sublesson.head()
```

|  | 身份 | 销售价格 |
| --- | --- | --- |
| Zero | One thousand four hundred and sixty-one | 128959.491726 |
| one | One thousand four hundred and sixty-two | 122920.740244 |
| Two | One thousand four hundred and sixty-three | 175704.825981 |
| three | One thousand four hundred and sixty-four | 200050.832638 |
| four | One thousand four hundred and sixty-five | 182075.469864 |

一旦我们确信已经以正确的格式安排了数据，我们就可以像 Kaggle 期望的那样导出到`.csv file`中。我们通过了`index=False`,因为熊猫会为我们创建一个新的索引。

```py
sublesson.to_csv('sublesson1.csv', index=False)
```

### 提交我们的结果

我们在工作目录中创建了一个名为`sublesson1.csv`的文件，它符合正确的格式。进入[转租页面](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/submit)进行转租。

我们从大约 2400 名竞争者中挑选了 1602 名。差不多是中游水平，还不错！注意我们这里的分数是`.15097`，比我们在测试数据上观察到的分数要好。这是一个好结果，但不会总是如此。

## 后续步骤

您可以通过以下方式扩展本教程并改进您的结果:

*   使用和转换训练集中的其他功能
*   尝试不同的建模技术，如随机森林回归或梯度推进
*   使用[集合模型](https://en.wikipedia.org/wiki/Ensemble_learning)

我们创建了一组名为`categoricals`的分类特征，它们并没有全部包含在最终的模型中。请返回并尝试包含这些功能。还有其他方法可能有助于分类数据，特别是`pd.get_dummies()`方法。在处理完这些特性之后，重复测试数据的转换，并创建另一个子分类。

研究模型和参加 Kaggle 竞赛可能是一个反复的过程——尝试新想法、了解数据和测试更新的模型和技术非常重要。

有了这些工具，您可以在工作的基础上改进结果。

祝你好运！