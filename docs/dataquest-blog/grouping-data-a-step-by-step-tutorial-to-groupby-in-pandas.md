# 数据分组:Pandas 中分组的分步指南

> 原文：<https://www.dataquest.io/blog/grouping-data-a-step-by-step-tutorial-to-groupby-in-pandas/>

February 2, 2022![](img/ea7ec6f7195d2a1dc05d5a24c0eceb35.png)

在本教程中，我们将探索如何在 Python 的 pandas 库中创建一个 GroupBy 对象，以及这个对象是如何工作的。我们将详细了解分组过程的每一步，哪些方法可以应用于 GroupBy 对象，以及我们可以从中提取哪些信息。

## Groupby 流程的 3 个步骤

任何 groupby 流程都涉及以下 3 个步骤的某种组合:

*   **根据定义的标准将原始对象分成组。**
*   **对每组应用**一个函数。
*   **结合**结果。

让我们用一个来自 Kaggle [诺贝尔奖数据集](https://www.kaggle.com/imdevskp/nobel-prize)的例子来一步一步地探索这个分离-应用-组合链:

```py
import pandas as pd
import numpy as np

pd.set_option('max_columns', None)

df = pd.read_csv('complete.csv')
df = df[['awardYear', 'category', 'prizeAmount', 'prizeAmountAdjusted', 'name', 'gender', 'birth_continent']]
df.head()
```

|  | 获奖年份 | 种类 | 奖金数额 | prizeAmountAdjusted | 名字 | 性别 | 出生地 _ 大陆 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Zero | Two thousand and one | 经济科学 | Ten million | Twelve million two hundred and ninety-five thousand and eighty-two | A·迈克尔·斯宾思 | 男性的 | 北美洲 |
| one | One thousand nine hundred and seventy-five | 物理学 | Six hundred and thirty thousand | Three million four hundred and four thousand one hundred and seventy-nine | Aage N. Bohr | 男性的 | 欧洲 |
| Two | Two thousand and four | 化学 | Ten million | Eleven million seven hundred and sixty-two thousand eight hundred and sixty-one | Aaron Ciechanover | 男性的 | 亚洲 |
| three | One thousand nine hundred and eighty-two | 化学 | One million one hundred and fifty thousand | Three million one hundred and two thousand five hundred and eighteen | 阿伦·克卢格 | 男性的 | 欧洲 |
| four | One thousand nine hundred and seventy-nine | 物理学 | Eight hundred thousand | Two million nine hundred and eighty-eight thousand and forty-eight | Abdus Salam | 男性的 | 亚洲 |

### 将原始对象拆分成组

在这个阶段，我们调用熊猫`DataFrame.groupby()`函数。我们使用它根据预定义的标准将数据分组，按照行(默认情况下，`axis=0`)或列(`axis=1`)。换句话说，这个函数将标签映射到组名。

例如，在我们的例子中，我们可以按奖项类别对诺贝尔奖的数据进行分组:

```py
grouped = df.groupby('category')
```

也可以使用多个列来执行数据分组，传递列的列表。让我们先按奖项类别对数据进行分组，然后，在每个已创建的组中，我们将根据获奖年份应用附加分组:

```py
grouped_category_year = df.groupby(['category', 'awardYear'])
```

现在，如果我们尝试打印我们创建的两个 GroupBy 对象中的一个，我们实际上将看不到任何组:

```py
print(grouped)
```

```py
<pandas.core.groupby.generic.DataFrameGroupBy object at 0x0000026083789DF0>
```

需要注意的是，创建 GroupBy 对象只检查我们是否传递了正确的映射；它不会真正执行拆分-应用-组合链的任何操作，直到我们显式地对这个对象使用某种方法或提取它的某些属性。

为了简单地检查产生的 GroupBy 对象，并检查组是如何被精确地分割的，我们可以从中提取出`groups`或`indices`属性。它们都返回一个字典，其中键是创建的组，值是原始数据帧中每个组的实例的轴标签(对于`groups`属性)或索引(对于`indices`属性)的列表:

```py
grouped.indices
```

```py
{'Chemistry': array([  2,   3,   7,   9,  10,  11,  13,  14,  15,  17,  19,  39,  62,
         64,  66,  71,  75,  80,  81,  86,  92, 104, 107, 112, 129, 135,
        153, 169, 175, 178, 181, 188, 197, 199, 203, 210, 215, 223, 227,
        239, 247, 249, 258, 264, 265, 268, 272, 274, 280, 282, 284, 289,
        296, 298, 310, 311, 317, 318, 337, 341, 343, 348, 352, 357, 362,
        365, 366, 372, 374, 384, 394, 395, 396, 415, 416, 419, 434, 440,
        442, 444, 446, 448, 450, 455, 456, 459, 461, 463, 465, 469, 475,
        504, 505, 508, 518, 522, 523, 524, 539, 549, 558, 559, 563, 567,
        571, 572, 585, 591, 596, 599, 627, 630, 632, 641, 643, 644, 648,
        659, 661, 666, 667, 668, 671, 673, 679, 681, 686, 713, 715, 717,
        719, 720, 722, 723, 725, 726, 729, 732, 738, 742, 744, 746, 751,
        756, 759, 763, 766, 773, 776, 798, 810, 813, 814, 817, 827, 828,
        829, 832, 839, 848, 853, 855, 862, 866, 880, 885, 886, 888, 889,
        892, 894, 897, 902, 904, 914, 915, 920, 921, 922, 940, 941, 943,
        946, 947], dtype=int64),
 'Economic Sciences': array([  0,   5,  45,  46,  58,  90,  96, 139, 140, 145, 152, 156, 157,
        180, 187, 193, 207, 219, 231, 232, 246, 250, 269, 279, 283, 295,
        305, 324, 346, 369, 418, 422, 425, 426, 430, 432, 438, 458, 467,
        476, 485, 510, 525, 527, 537, 538, 546, 580, 594, 595, 605, 611,
        636, 637, 657, 669, 670, 678, 700, 708, 716, 724, 734, 737, 739,
        745, 747, 749, 750, 753, 758, 767, 800, 805, 854, 856, 860, 864,
        871, 882, 896, 912, 916, 924], dtype=int64),
 'Literature': array([ 21,  31,  40,  49,  52,  98, 100, 101, 102, 111, 115, 142, 149,
        159, 170, 177, 201, 202, 220, 221, 233, 235, 237, 253, 257, 259,
        275, 277, 278, 286, 312, 315, 316, 321, 326, 333, 345, 347, 350,
        355, 359, 364, 370, 373, 385, 397, 400, 403, 406, 411, 435, 439,
        441, 454, 468, 479, 480, 482, 483, 492, 501, 506, 511, 516, 556,
        569, 581, 602, 604, 606, 613, 614, 618, 631, 633, 635, 640, 652,
        653, 655, 656, 665, 675, 683, 699, 761, 765, 771, 774, 777, 779,
        780, 784, 786, 788, 796, 799, 803, 836, 840, 842, 850, 861, 867,
        868, 878, 881, 883, 910, 917, 919, 927, 928, 929, 930, 936],
       dtype=int64),
 'Peace': array([  6,  12,  16,  25,  26,  27,  34,  36,  44,  47,  48,  54,  61,
         65,  72,  78,  79,  82,  95,  99, 116, 119, 120, 126, 137, 146,
        151, 166, 167, 171, 200, 204, 205, 206, 209, 213, 225, 236, 240,
        244, 255, 260, 266, 267, 270, 287, 303, 320, 329, 356, 360, 361,
        377, 386, 387, 388, 389, 390, 391, 392, 393, 433, 447, 449, 471,
        477, 481, 489, 491, 500, 512, 514, 517, 528, 529, 530, 533, 534,
        540, 542, 544, 545, 547, 553, 555, 560, 562, 574, 578, 590, 593,
        603, 607, 608, 609, 612, 615, 616, 617, 619, 620, 628, 634, 639,
        642, 664, 677, 688, 697, 703, 705, 710, 727, 736, 787, 793, 795,
        806, 823, 846, 847, 852, 865, 875, 876, 877, 895, 926, 934, 935,
        937, 944, 948, 949], dtype=int64),
 'Physics': array([  1,   4,   8,  20,  23,  24,  30,  32,  38,  51,  59,  60,  67,
         68,  69,  70,  74,  84,  89,  97, 103, 105, 108, 109, 114, 117,
        118, 122, 125, 127, 128, 130, 133, 141, 143, 144, 155, 162, 163,
        164, 165, 168, 173, 174, 176, 179, 183, 195, 212, 214, 216, 222,
        224, 228, 230, 234, 238, 241, 243, 251, 256, 263, 271, 276, 291,
        292, 297, 301, 306, 307, 308, 323, 327, 328, 330, 335, 336, 338,
        349, 351, 353, 354, 363, 367, 375, 376, 378, 381, 382, 398, 399,
        402, 404, 405, 408, 410, 412, 413, 420, 421, 424, 428, 429, 436,
        445, 451, 453, 457, 460, 462, 470, 472, 487, 495, 498, 499, 509,
        513, 515, 521, 526, 532, 535, 536, 541, 548, 550, 552, 557, 561,
        564, 565, 566, 573, 576, 577, 579, 583, 586, 588, 592, 601, 610,
        621, 622, 623, 629, 647, 650, 651, 654, 658, 674, 676, 682, 684,
        690, 691, 693, 694, 695, 696, 698, 702, 707, 711, 714, 721, 730,
        731, 735, 743, 752, 755, 770, 772, 775, 781, 785, 790, 792, 797,
        801, 802, 808, 822, 833, 834, 835, 844, 851, 870, 872, 879, 884,
        887, 890, 893, 900, 901, 903, 905, 907, 908, 909, 913, 925, 931,
        932, 933, 938, 942, 945], dtype=int64),
 'Physiology or Medicine': array([ 18,  22,  28,  29,  33,  35,  37,  41,  42,  43,  50,  53,  55,
         56,  57,  63,  73,  76,  77,  83,  85,  87,  88,  91,  93,  94,
        106, 110, 113, 121, 123, 124, 131, 132, 134, 136, 138, 147, 148,
        150, 154, 158, 160, 161, 172, 182, 184, 185, 186, 189, 190, 191,
        192, 194, 196, 198, 208, 211, 217, 218, 226, 229, 242, 245, 248,
        252, 254, 261, 262, 273, 281, 285, 288, 290, 293, 294, 299, 300,
        302, 304, 309, 313, 314, 319, 322, 325, 331, 332, 334, 339, 340,
        342, 344, 358, 368, 371, 379, 380, 383, 401, 407, 409, 414, 417,
        423, 427, 431, 437, 443, 452, 464, 466, 473, 474, 478, 484, 486,
        488, 490, 493, 494, 496, 497, 502, 503, 507, 519, 520, 531, 543,
        551, 554, 568, 570, 575, 582, 584, 587, 589, 597, 598, 600, 624,
        625, 626, 638, 645, 646, 649, 660, 662, 663, 672, 680, 685, 687,
        689, 692, 701, 704, 706, 709, 712, 718, 728, 733, 740, 741, 748,
        754, 757, 760, 762, 764, 768, 769, 778, 782, 783, 789, 791, 794,
        804, 807, 809, 811, 812, 815, 816, 818, 819, 820, 821, 824, 825,
        826, 830, 831, 837, 838, 841, 843, 845, 849, 857, 858, 859, 863,
        869, 873, 874, 891, 898, 899, 906, 911, 918, 923, 939], dtype=int64)}
```

为了找到 GroupBy 对象中的组数，我们可以从中提取出`ngroups`属性或者调用 Python 标准库的`len`函数:

```py
print(grouped.ngroups)
print(len(grouped))
```

```py
6
6
```

如果我们需要可视化每个组的所有或部分条目，我们可以迭代 GroupBy 对象:

```py
for name, entries in grouped:
    print(f'First 2 entries for the "{name}" category:')
    print(30*'-')
    print(entries.head(2), '\n\n')
```

```py
First 2 entries for the "Chemistry" category:
------------------------------
   awardYear   category  prizeAmount  prizeAmountAdjusted               name  \
2       2004  Chemistry     10000000             11762861  Aaron Ciechanover   
3       1982  Chemistry      1150000              3102518         Aaron Klug   

  gender birth_continent  
2   male            Asia  
3   male          Europe   

First 2 entries for the "Economic Sciences" category:
------------------------------
   awardYear           category  prizeAmount  prizeAmountAdjusted  \
0       2001  Economic Sciences     10000000             12295082   
5       2019  Economic Sciences      9000000              9000000   

                name gender birth_continent  
0  A. Michael Spence   male   North America  
5   Abhijit Banerjee   male            Asia   

First 2 entries for the "Literature" category:
------------------------------
    awardYear    category  prizeAmount  prizeAmountAdjusted  \
21       1957  Literature       208629              2697789   
31       1970  Literature       400000              3177966   

                     name gender birth_continent  
21           Albert Camus   male          Africa  
31  Alexandr Solzhenitsyn   male          Europe   

First 2 entries for the "Peace" category:
------------------------------
    awardYear category  prizeAmount  prizeAmountAdjusted  \
6        2019    Peace      9000000              9000000   
12       1980    Peace       880000              2889667   

                     name gender birth_continent  
6          Abiy Ahmed Ali   male          Africa  
12  Adolfo Pérez Esquivel   male   South America   

First 2 entries for the "Physics" category:
------------------------------
   awardYear category  prizeAmount  prizeAmountAdjusted          name gender  \
1       1975  Physics       630000              3404179  Aage N. Bohr   male   
4       1979  Physics       800000              2988048   Abdus Salam   male   

  birth_continent  
1          Europe  
4            Asia   

First 2 entries for the "Physiology or Medicine" category:
------------------------------
    awardYear                category  prizeAmount  prizeAmountAdjusted  \
18       1963  Physiology or Medicine       265000              2839286   
22       1974  Physiology or Medicine       550000              3263449   

             name gender birth_continent  
18   Alan Hodgkin   male          Europe  
22  Albert Claude   male          Europe 
```

相反，如果我们想以 DataFrame 的形式选择一个组，我们应该对 GroupBy 对象使用方法`get_group()`:

```py
grouped.get_group('Economic Sciences')
```

|  | 获奖年份 | 种类 | 奖金数额 | prizeAmountAdjusted | 名字 | 性别 | 出生地 _ 大陆 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Zero | Two thousand and one | 经济科学 | Ten million | Twelve million two hundred and ninety-five thousand and eighty-two | A·迈克尔·斯宾思 | 男性的 | 北美洲 |
| five | Two thousand and nineteen | 经济科学 | Nine million | Nine million | 阿比吉特·班纳吉 | 男性的 | 亚洲 |
| Forty-five | Two thousand and twelve | 经济科学 | Eight million | Eight million three hundred and sixty-one thousand two hundred and four | 埃尔文·E·罗斯 | 男性的 | 北美洲 |
| Forty-six | One thousand nine hundred and ninety-eight | 经济科学 | Seven million six hundred thousand | Nine million seven hundred and thirteen thousand seven hundred and one | 阿马蒂亚·森 | 男性的 | 亚洲 |
| Fifty-eight | Two thousand and fifteen | 经济科学 | Eight million | Eight million three hundred and eighty-four thousand five hundred and seventy-two | 安格斯·迪顿 | 男性的 | 欧洲 |
| … | … | … | … | … | … | … | … |
| Eight hundred and eighty-two | Two thousand and two | 经济科学 | Ten million | Twelve million thirty-four thousand six hundred and sixty | 弗农·L·史密斯 | 男性的 | 北美洲 |
| Eight hundred and ninety-six | One thousand nine hundred and seventy-three | 经济科学 | Five hundred and ten thousand | Three million three hundred and thirty-one thousand eight hundred and eighty-two | 瓦西里·列昂季耶夫 | 男性的 | 欧洲 |
| Nine hundred and twelve | Two thousand and eighteen | 经济科学 | Nine million | Nine million | 威廉·诺德豪斯 | 男性的 | 北美洲 |
| Nine hundred and sixteen | One thousand nine hundred and ninety | 经济科学 | Four million | Six million three hundred and twenty-nine thousand one hundred and fourteen | 威廉·夏普 | 男性的 | 北美洲 |
| Nine hundred and twenty-four | One thousand nine hundred and ninety-six | 经济科学 | Seven million four hundred thousand | Nine million four hundred and ninety thousand four hundred and twenty-four | 威廉·维克瑞 | 男性的 | 北美洲 |

84 行× 7 列

### 按组应用函数

在分割原始数据并(可选地)检查结果组之后，我们可以对每个组执行以下操作之一或它们的组合(不一定按照给定的顺序):

*   **聚合:**计算每个组的汇总统计数据(例如，组大小、平均值、中间值或总和)，并输出多个数据点的单一数字。
*   **转换:**按组进行一些操作，比如计算每组的 z 值。
*   **过滤:**根据预定义的条件，如组大小、平均值、中间值或总和，拒绝一些组。这也可以包括从每个组中过滤出特定的行。

#### 聚合

要聚合 GroupBy 对象的数据(即，按组计算汇总统计数据)，我们可以在对象上使用`agg()`方法:

```py
# Showing only 1 decimal for all float numbers
pd.options.display.float_format = '{:.1f}'.format

grouped.agg(np.mean)
```

|  | 获奖年份 | 奖金数额 | prizeAmountAdjusted |
| --- | --- | --- | --- |
| 种类 |  |  |  |
| --- | --- | --- | --- |
| 化学 | One thousand nine hundred and seventy-two point three | Three million six hundred and twenty-nine thousand two hundred and seventy-nine point four | Six million two hundred and fifty-seven thousand eight hundred and sixty-eight point one |
| 经济科学 | One thousand nine hundred and ninety-six point one | Six million one hundred and five thousand eight hundred and forty-five point two | Seven million eight hundred and thirty-seven thousand seven hundred and seventy-nine point two |
| 文学 | One thousand nine hundred and sixty point nine | Two million four hundred and ninety-three thousand eight hundred and eleven point two | Five million five hundred and ninety-eight thousand two hundred and fifty-six point three |
| 和平 | One thousand nine hundred and sixty-four point five | Three million one hundred and twenty-four thousand eight hundred and seventy-nine point two | Six million one hundred and sixty-three thousand nine hundred and six point nine |
| 物理学 | One thousand nine hundred and seventy-one point one | Three million four hundred and seven thousand nine hundred and thirty-eight point six | Six million eighty-six thousand nine hundred and seventy-eight point two |
| 生理学或医学 | One thousand nine hundred and seventy point four | Three million seventy-two thousand nine hundred and seventy-two point nine | Five million seven hundred and thirty-eight thousand three hundred point seven |

上面的代码生成一个 DataFrame，将组名作为它的新索引，并按组显示每个数值列的平均值。

我们可以不使用`agg()`方法，而是直接在 GroupBy 对象上应用相应的 pandas 方法。最常用的方法有`mean()`、`median()`、`mode()`、`sum()`、`size()`、`count()`、`min()`、`max()`、`std()`、`var()`(计算每组的方差)、`describe()`(按组输出描述性统计量)、`nunique()`(给出每组唯一值的个数)。

```py
grouped.sum()
```

|  | 获奖年份 | 奖金数额 | prizeAmountAdjusted |
| --- | --- | --- | --- |
| 种类 |  |  |  |
| --- | --- | --- | --- |
| 化学 | Three hundred and sixty-two thousand nine hundred and twelve | Six hundred and sixty-seven million seven hundred and eighty-seven thousand four hundred and eighteen | One billion one hundred and fifty-one million four hundred and forty-seven thousand seven hundred and twenty-six |
| 经济科学 | One hundred and sixty-seven thousand six hundred and seventy-four | Five hundred and twelve million eight hundred and ninety-one thousand | Six hundred and fifty-eight million three hundred and seventy-three thousand four hundred and forty-nine |
| 文学 | Two hundred and twenty-seven thousand four hundred and sixty-eight | Two hundred and eighty-nine million two hundred and eighty-two thousand one hundred and two | Six hundred and forty-nine million three hundred and ninety-seven thousand seven hundred and thirty-one |
| 和平 | Two hundred and sixty-three thousand two hundred and forty-eight | Four hundred and eighteen million seven hundred and thirty-three thousand eight hundred and seven | Eight hundred and twenty-five million nine hundred and sixty-three thousand five hundred and twenty-one |
| 物理学 | Four hundred and nineteen thousand eight hundred and thirty-seven | Seven hundred and twenty-five million eight hundred and ninety thousand nine hundred and twenty-eight | One billion two hundred and ninety-six million five hundred and twenty-six thousand three hundred and fifty-two |
| 生理学或医学 | Four hundred and thirty-one thousand five hundred and eight | Six hundred and seventy-two million nine hundred and eighty-one thousand and sixty-six | One billion two hundred and fifty-six million six hundred and eighty-seven thousand eight hundred and fifty-seven |

通常，我们只对某些特定列的统计数据感兴趣，所以我们需要指定它(或它们)。在上面的例子中，我们肯定不想对所有年份求和。相反，我们可能希望按奖项类别对奖项值求和。为此，我们可以选择 GroupBy 对象的`prizeAmountAdjusted`列，就像我们选择 DataFrame 的一列一样，并对其应用`sum()`函数:

```py
grouped['prizeAmountAdjusted'].sum()
```

```py
category
Chemistry                 1151447726
Economic Sciences          658373449
Literature                 649397731
Peace                      825963521
Physics                   1296526352
Physiology or Medicine    1256687857
Name: prizeAmountAdjusted, dtype: int64
```

对于上面的这段代码(以及下面的一些例子)，我们可以使用一个等价的语法，在选择必要的列之前对 GroupBy 对象应用函数:`grouped.sum()['prizeAmountAdjusted']`。但是，前面的语法更可取，因为它的性能更好，尤其是在大型数据集上。

如果我们需要合计两列或更多列的数据，我们使用双方括号:

```py
grouped[['prizeAmount', 'prizeAmountAdjusted']].sum()
```

|  | 奖金数额 | prizeAmountAdjusted |
| --- | --- | --- |
| 种类 |  |  |
| --- | --- | --- |
| 化学 | Six hundred and sixty-seven million seven hundred and eighty-seven thousand four hundred and eighteen | One billion one hundred and fifty-one million four hundred and forty-seven thousand seven hundred and twenty-six |
| 经济科学 | Five hundred and twelve million eight hundred and ninety-one thousand | Six hundred and fifty-eight million three hundred and seventy-three thousand four hundred and forty-nine |
| 文学 | Two hundred and eighty-nine million two hundred and eighty-two thousand one hundred and two | Six hundred and forty-nine million three hundred and ninety-seven thousand seven hundred and thirty-one |
| 和平 | Four hundred and eighteen million seven hundred and thirty-three thousand eight hundred and seven | Eight hundred and twenty-five million nine hundred and sixty-three thousand five hundred and twenty-one |
| 物理学 | Seven hundred and twenty-five million eight hundred and ninety thousand nine hundred and twenty-eight | One billion two hundred and ninety-six million five hundred and twenty-six thousand three hundred and fifty-two |
| 生理学或医学 | Six hundred and seventy-two million nine hundred and eighty-one thousand and sixty-six | One billion two hundred and fifty-six million six hundred and eighty-seven thousand eight hundred and fifty-seven |

可以对 GroupBy 对象的一列或多列同时应用多个函数。为此，我们再次需要`agg()`方法和感兴趣的函数列表:

```py
grouped[['prizeAmount', 'prizeAmountAdjusted']].agg([np.sum, np.mean, np.std])
```

|  | 奖金数额 | prizeAmountAdjusted |
| --- | --- | --- |
|  | 总和 | 意思是 | 标准 | 总和 | 意思是 | 标准 |
| --- | --- | --- | --- | --- | --- | --- |
| 种类 |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| 化学 | Six hundred and sixty-seven million seven hundred and eighty-seven thousand four hundred and eighteen | Three million six hundred and twenty-nine thousand two hundred and seventy-nine point four | Four million seventy thousand five hundred and eighty-eight point four | One billion one hundred and fifty-one million four hundred and forty-seven thousand seven hundred and twenty-six | Six million two hundred and fifty-seven thousand eight hundred and sixty-eight point one | Three million two hundred and seventy-six thousand and twenty-seven point two |
| 经济科学 | Five hundred and twelve million eight hundred and ninety-one thousand | Six million one hundred and five thousand eight hundred and forty-five point two | Three million seven hundred and eighty-seven thousand six hundred and thirty point one | Six hundred and fifty-eight million three hundred and seventy-three thousand four hundred and forty-nine | Seven million eight hundred and thirty-seven thousand seven hundred and seventy-nine point two | Three million three hundred and thirteen thousand one hundred and fifty-three point two |
| 文学 | Two hundred and eighty-nine million two hundred and eighty-two thousand one hundred and two | Two million four hundred and ninety-three thousand eight hundred and eleven point two | Three million six hundred and fifty-three thousand seven hundred and thirty-four | Six hundred and forty-nine million three hundred and ninety-seven thousand seven hundred and thirty-one | Five million five hundred and ninety-eight thousand two hundred and fifty-six point three | Three million twenty-nine thousand five hundred and twelve point one |
| 和平 | Four hundred and eighteen million seven hundred and thirty-three thousand eight hundred and seven | Three million one hundred and twenty-four thousand eight hundred and seventy-nine point two | Three million nine hundred and thirty-four thousand three hundred and ninety point nine | Eight hundred and twenty-five million nine hundred and sixty-three thousand five hundred and twenty-one | Six million one hundred and sixty-three thousand nine hundred and six point nine | Three million one hundred and eighty-nine thousand eight hundred and eighty-six point one |
| 物理学 | Seven hundred and twenty-five million eight hundred and ninety thousand nine hundred and twenty-eight | Three million four hundred and seven thousand nine hundred and thirty-eight point six | Four million thirteen thousand and seventy-three | One billion two hundred and ninety-six million five hundred and twenty-six thousand three hundred and fifty-two | Six million eighty-six thousand nine hundred and seventy-eight point two | Three million two hundred and ninety-four thousand two hundred and sixty-eight point five |
| 生理学或医学 | Six hundred and seventy-two million nine hundred and eighty-one thousand and sixty-six | Three million seventy-two thousand nine hundred and seventy-two point nine | Three million eight hundred and ninety-eight thousand five hundred and thirty-nine point three | One billion two hundred and fifty-six million six hundred and eighty-seven thousand eight hundred and fifty-seven | Five million seven hundred and thirty-eight thousand three hundred point seven | Three million two hundred and forty-one thousand seven hundred and eighty-one |

此外，我们可以考虑通过传递一个字典，对 GroupBy 对象的不同列应用不同的聚合函数:

```py
grouped.agg({'prizeAmount': [np.sum, np.size], 'prizeAmountAdjusted': np.mean})
```

|  | 奖金数额 | prizeAmountAdjusted |
| --- | --- | --- |
|  | 总和 | 大小 | 意思是 |
| --- | --- | --- | --- |
| 种类 |  |  |  |
| --- | --- | --- | --- |
| 化学 | Six hundred and sixty-seven million seven hundred and eighty-seven thousand four hundred and eighteen | One hundred and eighty-four | Six million two hundred and fifty-seven thousand eight hundred and sixty-eight point one |
| 经济科学 | Five hundred and twelve million eight hundred and ninety-one thousand | Eighty-four | Seven million eight hundred and thirty-seven thousand seven hundred and seventy-nine point two |
| 文学 | Two hundred and eighty-nine million two hundred and eighty-two thousand one hundred and two | One hundred and sixteen | Five million five hundred and ninety-eight thousand two hundred and fifty-six point three |
| 和平 | Four hundred and eighteen million seven hundred and thirty-three thousand eight hundred and seven | One hundred and thirty-four | Six million one hundred and sixty-three thousand nine hundred and six point nine |
| 物理学 | Seven hundred and twenty-five million eight hundred and ninety thousand nine hundred and twenty-eight | Two hundred and thirteen | Six million eighty-six thousand nine hundred and seventy-eight point two |
| 生理学或医学 | Six hundred and seventy-two million nine hundred and eighty-one thousand and sixty-six | Two hundred and nineteen | Five million seven hundred and thirty-eight thousand three hundred point seven |

#### 转换

与聚合方法不同(也与过滤方法不同，我们将很快看到)，转换方法返回一个新的数据帧，该数据帧具有与原始数据帧相同的形状和索引，但具有转换后的单个值。这里需要注意的是，转换不能修改原始数据帧中的任何值，这意味着此类操作不能就地执行。

转换 GroupBy 对象的数据的最常见的 pandas 方法是`transform()`。例如，它有助于计算每个组的 z 分数:

```py
grouped[['prizeAmount', 'prizeAmountAdjusted']].transform(lambda x: (x - x.mean()) / x.std())
```

|  | 奖金数额 | prizeAmountAdjusted |
| --- | --- | --- |
| Zero | One | One point three |
| one | -0.7 | -0.8 |
| Two | one point six | One point seven |
| three | -0.6 | -1.0 |
| four | -0.6 | -0.9 |
| … | … | … |
| Nine hundred and forty-five | -0.7 | -0.8 |
| Nine hundred and forty-six | -0.8 | -1.1 |
| Nine hundred and forty-seven | -0.9 | Zero point three |
| Nine hundred and forty-eight | -0.5 | -1.0 |
| Nine hundred and forty-nine | -0.7 | -1.0 |

950 行× 2 列

通过变换方法，我们还可以用组均值、中值、众数或任何其他值替换缺失数据:

```py
grouped['gender'].transform(lambda x: x.fillna(x.mode()[0]))
```

```py
0        male
1        male
2        male
3        male
4        male
        ...  
945      male
946      male
947    female
948      male
949      male
Name: gender, Length: 950, dtype: object
```

我们还可以使用其他一些 pandas 方法来按对象转换 GroupBy 数据:`bfill()`、`ffill()`、`diff()`、`pct_change()`、`rank()`、`shift()`、`quantile()`等。

#### 过滤

过滤方法基于预定义的条件丢弃组或每个组中的特定行，并返回原始数据的子集。例如，我们可能希望只保留所有组中某一列的值，其中该列的组平均值大于预定义的值。在我们的数据框架中，让我们过滤掉所有组，使`prizeAmountAdjusted`列的组平均值小于 7，000，000，并且在输出中只保留这一列:

```py
grouped['prizeAmountAdjusted'].filter(lambda x: x.mean() > 7000000)
```

```py
0      12295082
5       9000000
45      8361204
46      9713701
58      8384572
         ...   
882    12034660
896     3331882
912     9000000
916     6329114
924     9490424
Name: prizeAmountAdjusted, Length: 84, dtype: int64
```

另一个例子是过滤掉具有超过一定数量的元素的组:

```py
grouped['prizeAmountAdjusted'].filter(lambda x: len(x) < 100)
```

```py
0      12295082
5       9000000
45      8361204
46      9713701
58      8384572
         ...   
882    12034660
896     3331882
912     9000000
916     6329114
924     9490424
Name: prizeAmountAdjusted, Length: 84, dtype: int64
```

在上面的两个操作中，我们使用了将 lambda 函数作为参数传递的`filter()`方法。这种应用于整个组的函数根据该组与预定义统计条件的比较结果返回`True`或`False`。换句话说，`filter()`方法中的函数决定在新的数据帧中**保留哪些组**。

除了过滤掉整个组，还可以从每个组中丢弃某些行。这里有一些有用的方法是`first()`、`last()`和`nth()`。将其中一个应用于 GroupBy 对象会相应地返回每个组的第一个/最后一个/第 n 个条目:

```py
grouped.last()
```

|  | 获奖年份 | 奖金数额 | prizeAmountAdjusted | 名字 | 性别 | 出生地 _ 大陆 |
| --- | --- | --- | --- | --- | --- | --- |
| 种类 |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| 化学 | One thousand nine hundred and eleven | One hundred and forty thousand six hundred and ninety-five | Seven million three hundred and twenty-seven thousand eight hundred and sixty-five | 玛丽·居里 | 女性的 | 欧洲 |
| 经济科学 | One thousand nine hundred and ninety-six | Seven million four hundred thousand | Nine million four hundred and ninety thousand four hundred and twenty-four | 威廉·维克瑞 | 男性的 | 北美洲 |
| 文学 | One thousand nine hundred and sixty-eight | Three hundred and fifty thousand | Three million fifty-two thousand three hundred and twenty-six | 川端康成 | 男性的 | 亚洲 |
| 和平 | One thousand nine hundred and sixty-three | Two hundred and sixty-five thousand | Two million eight hundred and thirty-nine thousand two hundred and eighty-six | 红十字国际委员会 | 男性的 | 亚洲 |
| 物理学 | One thousand nine hundred and seventy-two | Four hundred and eighty thousand | Three million three hundred and forty-five thousand seven hundred and twenty-five | 约翰巴丁 | 男性的 | 北美洲 |
| 生理学或医学 | Two thousand and sixteen | Eight million | Eight million three hundred and one thousand and fifty-one | 大隅良典 | 男性的 | 亚洲 |

对于`nth()`方法，我们必须传递一个整数，该整数代表我们想要为每个组返回的条目的索引:

```py
grouped.nth(1)
```

|  | 获奖年份 | 奖金数额 | prizeAmountAdjusted | 名字 | 性别 | 出生地 _ 大陆 |
| --- | --- | --- | --- | --- | --- | --- |
| 种类 |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| 化学 | One thousand nine hundred and eighty-two | One million one hundred and fifty thousand | Three million one hundred and two thousand five hundred and eighteen | 阿伦·克卢格 | 男性的 | 欧洲 |
| 经济科学 | Two thousand and nineteen | Nine million | Nine million | 阿比吉特·班纳吉 | 男性的 | 亚洲 |
| 文学 | One thousand nine hundred and seventy | Four hundred thousand | Three million one hundred and seventy-seven thousand nine hundred and sixty-six | 亚历山大·巴甫洛夫·索尔仁尼琴 | 男性的 | 欧洲 |
| 和平 | One thousand nine hundred and eighty | Eight hundred and eighty thousand | Two million eight hundred and eighty-nine thousand six hundred and sixty-seven | 阿道夫·佩雷斯·埃斯基维尔 | 男性的 | 南美。参见 AMERICA |
| 物理学 | One thousand nine hundred and seventy-nine | Eight hundred thousand | Two million nine hundred and eighty-eight thousand and forty-eight | Abdus Salam | 男性的 | 亚洲 |
| 生理学或医学 | One thousand nine hundred and seventy-four | Five hundred and fifty thousand | Three million two hundred and sixty-three thousand four hundred and forty-nine | 阿尔伯特克劳德 | 男性的 | 欧洲 |

上面这段代码收集了所有组的第二个条目，记住 Python 中的 0 索引。

过滤出每组中的行的另外两种方法是`head()`和`tail()`，对应返回每组的第一个/最后一个 *n* 行(默认为 5 行):

```py
grouped.head(3)
```

|  | 获奖年份 | 种类 | 奖金数额 | prizeAmountAdjusted | 名字 | 性别 | 出生地 _ 大陆 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Zero | Two thousand and one | 经济科学 | Ten million | Twelve million two hundred and ninety-five thousand and eighty-two | A·迈克尔·斯宾思 | 男性的 | 北美洲 |
| one | One thousand nine hundred and seventy-five | 物理学 | Six hundred and thirty thousand | Three million four hundred and four thousand one hundred and seventy-nine | Aage N. Bohr | 男性的 | 欧洲 |
| Two | Two thousand and four | 化学 | Ten million | Eleven million seven hundred and sixty-two thousand eight hundred and sixty-one | Aaron Ciechanover | 男性的 | 亚洲 |
| three | One thousand nine hundred and eighty-two | 化学 | One million one hundred and fifty thousand | Three million one hundred and two thousand five hundred and eighteen | 阿伦·克卢格 | 男性的 | 欧洲 |
| four | One thousand nine hundred and seventy-nine | 物理学 | Eight hundred thousand | Two million nine hundred and eighty-eight thousand and forty-eight | Abdus Salam | 男性的 | 亚洲 |
| five | Two thousand and nineteen | 经济科学 | Nine million | Nine million | 阿比吉特·班纳吉 | 男性的 | 亚洲 |
| six | Two thousand and nineteen | 和平 | Nine million | Nine million | 阿比·艾哈迈德·阿里 | 男性的 | 非洲 |
| seven | Two thousand and nine | 化学 | Ten million | Ten million nine hundred and fifty-eight thousand five hundred and four | 艾达·约纳什 | 女性的 | 亚洲 |
| eight | Two thousand and eleven | 物理学 | Ten million | Ten million five hundred and forty-five thousand five hundred and fifty-seven | 亚当·g·里斯 | 男性的 | 北美洲 |
| Twelve | One thousand nine hundred and eighty | 和平 | Eight hundred and eighty thousand | Two million eight hundred and eighty-nine thousand six hundred and sixty-seven | 阿道夫·佩雷斯·埃斯基维尔 | 男性的 | 南美。参见 AMERICA |
| Sixteen | Two thousand and seven | 和平 | Ten million | Eleven million three hundred and one thousand nine hundred and eighty-nine | 阿尔戈尔 | 男性的 | 北美洲 |
| Eighteen | One thousand nine hundred and sixty-three | 生理学或医学 | Two hundred and sixty-five thousand | Two million eight hundred and thirty-nine thousand two hundred and eighty-six | 艾伦·霍奇金 | 男性的 | 欧洲 |
| Twenty-one | One thousand nine hundred and fifty-seven | 文学 | Two hundred and eight thousand six hundred and twenty-nine | Two million six hundred and ninety-seven thousand seven hundred and eighty-nine | 阿尔伯特·加缪 | 男性的 | 非洲 |
| Twenty-two | One thousand nine hundred and seventy-four | 生理学或医学 | Five hundred and fifty thousand | Three million two hundred and sixty-three thousand four hundred and forty-nine | 阿尔伯特克劳德 | 男性的 | 欧洲 |
| Twenty-eight | One thousand nine hundred and thirty-seven | 生理学或医学 | One hundred and fifty-eight thousand four hundred and sixty-three | Four million seven hundred and sixteen thousand one hundred and sixty-one | 阿尔伯特·圣奎奇 | 男性的 | 欧洲 |
| Thirty-one | One thousand nine hundred and seventy | 文学 | Four hundred thousand | Three million one hundred and seventy-seven thousand nine hundred and sixty-six | 亚历山大·巴甫洛夫·索尔仁尼琴 | 男性的 | 欧洲 |
| Forty | Two thousand and thirteen | 文学 | Eight million | Eight million three hundred and sixty-five thousand eight hundred and sixty-seven | 爱丽丝·门罗 | 女性的 | 北美洲 |
| Forty-five | Two thousand and twelve | 经济科学 | Eight million | Eight million three hundred and sixty-one thousand two hundred and four | 埃尔文·E·罗斯 | 男性的 | 北美洲 |

### 合并结果

拆分-应用-合并链的最后一步——合并结果——是由熊猫在幕后完成的。它包括获取在 GroupBy 对象上执行的所有操作的输出，并将它们重新组合在一起，产生一个新的数据结构，如序列或数据帧。将这个数据结构赋给一个变量，我们可以用它来解决其他任务。

## 结论

在本教程中，我们介绍了使用 pandas groupby 函数和处理结果对象的许多方面。特别是，我们了解到以下情况:

*   分组过程包括的步骤
*   分割-应用-合并链如何一步一步地工作
*   如何创建 GroupBy 对象
*   如何简要检查 GroupBy 对象
*   GroupBy 对象的属性
*   可应用于 GroupBy 对象的操作
*   如何按组计算汇总统计数据，有哪些方法可用于此目的
*   如何对 GroupBy 对象的一列或多列同时应用多个函数
*   如何对 GroupBy 对象的不同列应用不同的聚合函数
*   如何以及为什么转换原始数据帧中的值
*   如何筛选 GroupBy 对象的组或每个组的特定行
*   熊猫如何组合一个分组过程的结果
*   分组过程产生的数据结构

有了这些信息，你就可以开始在熊猫小组中运用你的技能了。