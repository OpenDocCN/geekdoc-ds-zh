# 教程:在 Python 中使用带有大型数据集的 Pandas

> 原文：<https://www.dataquest.io/blog/pandas-big-data/>

August 4, 2017

您是否知道当您处理大数据集时，Python 和 pandas 可以减少多达 90%的内存使用？

当使用 pandas 处理小数据(小于 100 兆字节)的 Python 时，性能很少成为问题。当我们转移到更大的数据(100 兆字节到数千兆字节)时，性能问题会使运行时间变得更长，并导致代码由于内存不足而完全失败。

虽然 Spark 等工具可以处理大型数据集(100 到数 TB)，但充分利用它们的能力通常需要更昂贵的硬件。与熊猫不同，它们缺乏丰富的功能集来进行高质量的数据清理、探索和分析。对于中等大小的数据，我们最好尝试从 pandas 中获得更多，而不是切换到不同的工具。

在这篇文章中，我们将了解 Python 对 pandas 的内存使用，如何通过为列选择适当的数据类型，将数据帧的内存占用减少近 90%。

![](img/ca0dd6f50814b03f58b6e062ed579352.png)

## 使用棒球比赛日志

我们将研究 130 年来美国职业棒球大联盟比赛的数据，这些数据最初来自于[回顾表](https://www.retrosheet.org/gamelogs/index.html)。

最初，数据在 127 个单独的 CSV 文件中，但是我们使用了 [csvkit](https://csvkit.readthedocs.io/en/1.0.2/) 来合并文件，并在第一行中添加了列名。如果你想下载我们的数据版本来跟进这篇文章，我们已经在这里提供了[。](https://data.world/dataquest/mlb-game-logs)

让我们从在 Python 中导入熊猫和我们的数据开始，看看前五行。

```py
 import pandas as pd
gl = pd.read_csv('game_logs.csv')
gl.head() 
```

|  | 日期 | 游戏的数量 | 星期几 | 虚拟姓名 | v _ 联赛 | v _ 游戏 _ 号码 | hgame | h _ 联赛 | h _ 游戏 _ 号码 | v _ 分数 | h 分数 | 长度 _ 超时 | 白天夜晚 | 完成 | 丧失 | 抗议 | 公园 id | 出席 | 长度 _ 分钟 | v _ 线 _ 分数 | h_line_score | 蝙蝠 | v_hits | v _ 双打 | v_triples | v _ 本垒打 | 虚拟银行 | v _ 牺牲 _ 命中 | v _ 牺牲 _ 苍蝇 | 垂直命中间距 | v_walks | v _ 有意行走 | v _ 删除线 | 五 _ 被盗 _ 基地 | 偷窃被抓 | v _ 接地 _ 进 _ 双 | v _ 优先 _ 捕捉器 _ 干扰 | v _ 左 _ 上 _ 下 | v _ 投手 _ 已用 | 个人获得的跑步记录 | v _ team _ earned _ runs | v _ wild _ pitches | v_balks | v _ 输出 | vassistx | v _ 错误 | v _ 传球 _ 球 | 双人游戏 | 三重播放 | 蝙蝠 | 点击次数 | h _ 双打 | h_triples | h _ 本垒打 | h_rbi | h _ 牺牲 _ 命中 | h _ 祭祀 _ 苍蝇 | h _ hit _ by _ 音高 | h _ 行走 | h _ 有意行走 | h _ 删除线 | h _ 被盗 _ 基地 | 偷东西被抓 | h _ 接地 _ 进 _ 双 | h _ 优先 _ 捕捉器 _ 干扰 | 左满垒 | h_pitchers_used | 个人获得的跑步记录 | h _ team _ earned _ runs | h _ wild _ pitches | h_balks | h _ 输出 | h _ 助攻 | h _ 错误 | h _ 传球 _ 球 | 双人游戏 | 三重播放 | 惠普 _ 裁判 _id | 惠普裁判姓名 | 1b _ 裁判 _id | 1b _ 裁判员 _ 姓名 | 2b _ 裁判 _id | 2b _ 裁判员 _ 姓名 | 3b _ 裁判 _id | 3b _ 裁判员 _ 姓名 | lf _ 裁判员 _id | lf _ 裁判员 _ 姓名 | rf _ 裁判员 _id | rf _ 裁判员 _ 姓名 | 虚拟经理 id | 虚拟经理姓名 | h _ 经理 _id | 经理姓名 | winning _ pitcher _ id | 获胜投手姓名 | losing_pitcher_id | 失去投手姓名 | 正在保存 _pitcher_id | 保存 _pitcher_name | 中奖 _ 打点 _ 击球手 _id | 获奖 _rbi_batter_id_name | v_starting_pitcher_id | v_starting_pitcher_name | h_starting_pitcher_id | h_starting_pitcher_name | 虚拟玩家 1 号 | v _ 玩家 _ 1 _ 姓名 | 虚拟玩家 1 定义位置 | v _ 玩家 _2_id | v _ 玩家 _ 2 _ 姓名 | 虚拟玩家 2 定义位置 | 虚拟玩家 3 号 id | v _ 玩家 _ 3 _ 姓名 | 虚拟玩家 3 定义位置 | v _ 玩家 _4_id | v _ 玩家 _ 4 _ 姓名 | 虚拟玩家 4 定义位置 | v _ 玩家 _5_id | v_player_5_name | 虚拟玩家 5 定义位置 | v_player_6_id | v_player_6_name | 虚拟玩家 6 定义位置 | v _ 玩家 _7_id | v_player_7_name | 虚拟玩家 7 定义位置 | v_player_8_id | v_player_8_name | 虚拟玩家 8 定义位置 | v _ 玩家 _9_id | v_player_9_name | 虚拟玩家 9 定义位置 | h_player_1_id | h_player_1_name | h _ 播放器 _1_def_pos | h_player_2_id | h_player_2_name | h _ 播放器 _2_def_pos | h_player_3_id | h_player_3_name | h _ 播放器 _3_def_pos | h_player_4_id | h_player_4_name | h _ 播放器 _4_def_pos | h_player_5_id | h_player_5_name | h _ 播放器 _5_def_pos | h_player_6_id | h_player_6_name | h _ 播放器 _6_def_pos | h_player_7_id | h_player_7_name | h _ 播放器 _7_def_pos | h_player_8_id | h_player_8_name | h _ 播放器 _8_def_pos | h_player_9_id | h_player_9_name | h _ 播放器 _9_def_pos | 附加信息 | 收购信息 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Zero | Eighteen million seven hundred and ten thousand five hundred and four | Zero | 星期四 | CL1 | 钠 | one | FW1 | 钠 | one | Zero | Two | Fifty-four | D | 圆盘烤饼 | 圆盘烤饼 | 圆盘烤饼 | 对于 01 | Two hundred | One hundred and twenty | 000000000 | 010010000 | Thirty | Four | One | Zero | Zero | Zero | Zero | Zero | Zero | One | -1.0 | Six | One | -1.0 | -1.0 | -1.0 | Four | One | One | One | Zero | Zero | Twenty-seven | Nine | Zero | Three | Zero | Zero | Thirty-one | Four | One | Zero | Zero | Two | Zero | Zero | Zero | One | -1.0 | Zero | Zero | -1.0 | -1.0 | -1.0 | Three | One | Zero | Zero | Zero | Zero | Twenty-seven | Three | Three | One | One | Zero | boakj901 | 约翰·布莱克 | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | paboc101 | 查理·帕博尔 | lennb101 | 比尔·列侬 | mathb101 | 鲍比·马修斯 | 普拉塔 101 | 艾尔·普拉特 | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | 普拉塔 101 | 艾尔·普拉特 | mathb101 | 鲍比·马修斯 | whitd102 | 迪肯·怀特 | Two | kimbg101 | 吉恩·金博尔 | Four | paboc101 | 查理·帕博尔 | Seven | allia101 | 阿特·艾利森 | Eight | 白色 104 | 埃尔默·怀特 | Nine | 普拉塔 101 | 艾尔·普拉特 | One | 萨图特 101 | 以斯拉·萨顿 | Five | 卡莱 102 | 吉姆·卡尔顿 | Three | bassj101 | 约翰·巴斯 | Six | selmf101 | 弗兰克·塞尔曼 | Five | mathb101 | 鲍比·马修斯 | One | 福尔马林 101 | Jim Foran | Three | goldw101 | 沃利·戈德史密斯 | Six | lennb101 | 比尔·列侬 | Two | 职业 101 | 汤姆·凯里 | Four | 碎肉 101 | 埃德·明彻 | Seven | mcdej101 | 詹姆斯·麦克德莫特 | Eight | kellb105 | 比尔·凯利 | Nine | 圆盘烤饼 | Y |
| one | Eighteen million seven hundred and ten thousand five hundred and five | Zero | Fri | BS1 | 钠 | one | WS3 | 钠 | one | Twenty | Eighteen | Fifty-four | D | 圆盘烤饼 | 圆盘烤饼 | 圆盘烤饼 | WAS01 | Five thousand | One hundred and forty-five | One hundred and seven million four hundred and thirty-five | Six hundred and forty million one hundred and thirteen thousand and thirty | Forty-one | Thirteen | One | Two | Zero | Thirteen | Zero | Zero | Zero | Eighteen | -1.0 | Five | Three | -1.0 | -1.0 | -1.0 | Twelve | One | Six | Six | One | Zero | Twenty-seven | Thirteen | Ten | One | Two | Zero | Forty-nine | Fourteen | Two | Zero | Zero | Eleven | Zero | Zero | Zero | Ten | -1.0 | Two | One | -1.0 | -1.0 | -1.0 | Fourteen | One | Seven | Seven | Zero | Zero | Twenty-seven | Twenty | Ten | Two | Three | Zero | 多布斯 901 | 亨利·多布森 | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | wrigh101 | 哈里·赖特 | younn801 | 尼克·杨 | spala101 | 阿尔·斯伯丁 | 巴西 102 | 阿萨·布雷纳德 | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | spala101 | 阿尔·斯伯丁 | 巴西 102 | 阿萨·布雷纳德 | 箭牌 101 | 乔治·赖特 | Six | 巴恩斯 102 | 罗斯·巴恩斯 | Four | birdd102 | 戴夫·伯索尔 | Nine | mcvec101 | 卡尔·麦克维 | Two | wrigh101 | 哈里·赖特 | Eight | goulc101 | 查理·古尔德 | Three | 沙赫 101 | 哈里·斯查费 | Five | conef101 | 弗雷德·科恩 | Seven | spala101 | 阿尔·斯伯丁 | One | watef102 | 弗雷德·沃特曼 | Five | forcd101 | 戴维力量 | Six | mille105 | 埃弗里特·米尔斯 | Three | allid101 | 道格·艾利森 | Two | hallg101 | 乔治·霍尔 | Seven | 利昂娜 101 | 安迪·伦纳德 | Four | 巴西 102 | 阿萨·布雷纳德 | One | burrh101 | 亨利·巴勒斯 | Nine | berth101 | 亨利·贝思龙 | Eight | HTBF | Y |
| Two | Eighteen million seven hundred and ten thousand five hundred and six | Zero | 坐 | CL1 | 钠 | Two | RC1 | 钠 | one | Twelve | four | Fifty-four | D | 圆盘烤饼 | 圆盘烤饼 | 圆盘烤饼 | RCK01 | One thousand | One hundred and forty | Six hundred and ten million twenty thousand and three | 010020100 | Forty-nine | Eleven | One | One | Zero | Eight | Zero | Zero | Zero | Zero | -1.0 | One | Zero | -1.0 | -1.0 | -1.0 | Ten | One | Zero | Zero | Two | Zero | Twenty-seven | Twelve | Eight | Five | Zero | Zero | Thirty-six | Seven | Two | One | Zero | Two | Zero | Zero | Zero | Zero | -1.0 | Three | Five | -1.0 | -1.0 | -1.0 | Five | One | Three | Three | One | Zero | Twenty-seven | Twelve | Thirteen | Three | Zero | Zero | mawnj901 | J.H .曼尼 | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | paboc101 | 查理·帕博尔 | hasts101 | 斯科特·黑斯廷斯 | 普拉塔 101 | 艾尔·普拉特 | 鱼 c102 | 切诺基费希尔 | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | 普拉塔 101 | 艾尔·普拉特 | 鱼 c102 | 切诺基费希尔 | whitd102 | 迪肯·怀特 | Two | kimbg101 | 吉恩·金博尔 | Four | paboc101 | 查理·帕博尔 | Seven | allia101 | 阿特·艾利森 | Eight | 白色 104 | 埃尔默·怀特 | Nine | 普拉塔 101 | 艾尔·普拉特 | One | 萨图特 101 | 以斯拉·萨顿 | Five | 卡莱 102 | 吉姆·卡尔顿 | Three | bassj101 | 约翰·巴斯 | Six | mackd101 | 丹尼·麦克 | Three | addyb101 | 鲍勃·艾迪 | Four | 鱼 c102 | 切诺基费希尔 | One | hasts101 | 斯科特·黑斯廷斯 | Eight | ham-r101 | Ralph Ham | Five | ansoc101 | 安森角 | Two | sagep101 | 小马的事 | Six | birdg101 | 乔治·伯德 | Seven | 搅拌 101 | Gat Stires | Nine | 圆盘烤饼 | Y |
| three | Eighteen million seven hundred and ten thousand five hundred and eight | Zero | 孟人 | CL1 | 钠 | three | CH1 | 钠 | one | Twelve | Fourteen | Fifty-four | D | 圆盘烤饼 | 圆盘烤饼 | 圆盘烤饼 | CHI01 | Five thousand | One hundred and fifty | One hundred and one million four hundred and three thousand one hundred and eleven | 077000000 | Forty-six | Fifteen | Two | One | Two | Ten | Zero | Zero | Zero | Zero | -1.0 | One | Zero | -1.0 | -1.0 | -1.0 | Seven | One | Six | Six | Zero | Zero | Twenty-seven | Fifteen | Eleven | Six | Zero | Zero | Forty-three | Eleven | Two | Zero | Zero | Eight | Zero | Zero | Zero | Four | -1.0 | Two | One | -1.0 | -1.0 | -1.0 | Six | One | Four | Four | Zero | Zero | Twenty-seven | Fourteen | Seven | Two | Zero | Zero | willg901 | 加德纳·威拉德 | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | paboc101 | 查理·帕博尔 | woodj106 | 吉米·伍德 | zettg101 | 乔治·蔡特林 | 普拉塔 101 | 艾尔·普拉特 | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | 普拉塔 101 | 艾尔·普拉特 | zettg101 | 乔治·蔡特林 | whitd102 | 迪肯·怀特 | Two | kimbg101 | 吉恩·金博尔 | Four | paboc101 | 查理·帕博尔 | Seven | allia101 | 阿特·艾利森 | Eight | 白色 104 | 埃尔默·怀特 | Nine | 普拉塔 101 | 艾尔·普拉特 | One | 萨图特 101 | 以斯拉·萨顿 | Five | 卡莱 102 | 吉姆·卡尔顿 | Three | bassj101 | 约翰·巴斯 | Six | mcatb101 | 布·麦卡蒂 | Three | kingm101 | 马歇尔·金 | Eight | hodec101 | 查理·霍兹 | Two | woodj106 | 吉米·伍德 | Four | simmj101 | 乔·西蒙斯 | Nine | 小精灵 101 | 汤姆·福利 | Seven | 粗呢 101 | 艾德·达菲 | Six | pinke101 | 艾德平克曼 | Five | zettg101 | 乔治·蔡特林 | One | 圆盘烤饼 | Y |
| four | Eighteen million seven hundred and ten thousand five hundred and nine | Zero | 周二 | BS1 | 钠 | Two | TRO | 钠 | one | nine | five | Fifty-four | D | 圆盘烤饼 | 圆盘烤饼 | 圆盘烤饼 | TRO01 | Three thousand two hundred and fifty | One hundred and forty-five | 000002232 | One hundred and one million and three thousand | Forty-six | Seventeen | Four | One | Zero | Six | Zero | Zero | Zero | Two | -1.0 | Zero | One | -1.0 | -1.0 | -1.0 | Twelve | One | Two | Two | Zero | Zero | Twenty-seven | Twelve | Five | Zero | One | Zero | Thirty-six | Nine | Zero | Zero | Zero | Two | Zero | Zero | Zero | Three | -1.0 | Zero | Two | -1.0 | -1.0 | -1.0 | Seven | One | Three | Three | One | Zero | Twenty-seven | Eleven | Seven | Three | Zero | Zero | leroi901 | 艾萨克·勒罗伊 | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | wrigh101 | 哈里·赖特 | pikel101 | 唇矛 | spala101 | 阿尔·斯伯丁 | 麦克莫吉 101 | 约翰·麦克穆林 | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | spala101 | 阿尔·斯伯丁 | 麦克莫吉 101 | 约翰·麦克穆林 | 箭牌 101 | 乔治·赖特 | Six | 巴恩斯 102 | 罗斯·巴恩斯 | Four | birdd102 | 戴夫·伯索尔 | Nine | mcvec101 | 卡尔·麦克维 | Two | wrigh101 | 哈里·赖特 | Eight | goulc101 | 查理·古尔德 | Three | 沙赫 101 | 哈里·斯查费 | Five | conef101 | 弗雷德·科恩 | Seven | spala101 | 阿尔·斯伯丁 | One | flync101 | 快船弗林 | Nine | mcgem101 | 迈克·麦吉瑞 | Two | yorkt101 | 汤姆·约克 | Eight | 麦克莫吉 101 | 约翰·麦克穆林 | One | 金斯 101 | 史蒂夫·金 | Seven | beave101 | 爱德华·比弗斯 | Four | 贝尔 s101 | 史蒂夫·贝伦 | Five | pikel101 | 唇矛 | Three | cravb101 | 比尔·克雷默 | Six | HTBF | Y |

我们在下面总结了一些重要的列，但是如果您想查看所有列的指南，我们为整个数据集创建了一个[数据字典:](https://data.world/dataquest/mlb-game-logs/workspace/data-dictionary)

*   `date` —比赛日期。
*   `v_name` —客队名称。
*   `v_league` —客队联赛。
*   `h_name` —主队名称。
*   `h_league` —主队联赛。
*   `v_score` —客队得分。
*   `h_score` —主队得分。
*   `v_line_score` —客队线路得分，如`010000(10)00`。
*   `h_line_score` —主队线路得分，如`010000(10)0X`。
*   `park_id` —举办比赛的公园的 ID。
*   `attendance` —游戏出席率。

我们可以使用 [`DataFrame.info()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.info.html) 方法来给出一些关于数据帧的高级信息，包括它的大小、数据类型和内存使用的信息。

默认情况下，pandas 大约使用数据帧的内存来节省时间。因为我们对准确性感兴趣，所以我们将把`memory_usage`参数设置为`'deep'`来获得一个准确的数字。

```py
gl.info(memory_usage='deep')
```

```py
<class 'pandas.core.frame.DataFrame'>RangeIndex: 171907 entries, 0 to 171906
Columns: 161 entries, date to acquisition_infodtypes: float64(77), int64(6), object(78)
memory usage: 861.6 MB
```

我们可以看到有 171，907 行和 161 列。Pandas 已经为我们自动检测了类型，有 83 个数字列和 78 个对象列。对象列用于字符串或包含混合数据类型的列。

因此，我们可以更好地了解我们可以在哪里减少这种内存使用，让我们看看 Python 和 pandas 如何在内存中存储数据。

## 数据帧的内部表示

在底层，pandas 将列分组为相同类型的值块。下面是 pandas 如何存储我们的数据帧的前 12 列的预览。

![](img/38864fbf323506895ebaa325ba1ac322.png)

您会注意到这些块没有维护对列名的引用。这是因为块是为存储数据帧中的实际值而优化的。 [BlockManager 类](https://kite.com/python/docs/pandas.core.internals.BlockManager)负责维护行和列索引与实际块之间的映射。它充当一个 API，提供对底层数据的访问。每当我们选择、编辑或删除值时，dataframe 类都会与 BlockManager 类接口，将我们的请求转换为函数和方法调用。

每种类型在 [`pandas.core.internals`](https://kite.com/python/docs/pandas.core.internals.BlockManager) 模块中都有专门的类。Pandas 使用 ObjectBlock 类表示包含 string 列的块，使用 FloatBlock 类表示包含 float 列的块。对于表示数字值(如整数和浮点数)的块，pandas 合并列并将其存储为 NumPy ndarray。NumPy ndarray 是围绕 C 数组构建的，值存储在连续的内存块中。由于这种存储方案，访问一部分值的速度非常快。

因为每种数据类型都是单独存储的，所以我们将按数据类型检查内存使用情况。让我们从查看数据类型的平均内存使用量开始。

```py
 for dtype in ['float','int','object']:
    selected_dtype = gl.select_dtypes(include=[dtype])
    mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
    mean_usage_mb = mean_usage_b / 1024 ** 2
    print("Average memory usage for {} columns: {:03.2f} MB".format(dtype,mean_usage_mb))
```

```py
Average memory usage for float columns: 1.29 MB
Average memory usage for int columns: 1.12 MB
Average memory usage for object columns: 9.53 MB
```

我们可以立即看到，我们的 78 个`object`列使用了大部分内存。我们将在后面讨论这些，但首先让我们看看是否可以改善数字列的内存使用。

## 了解子类型

正如我们之前简单提到的，pandas 将数值表示为 NumPy ndarrays，并将它们存储在连续的内存块中。这种存储模型占用更少的空间，并允许我们快速访问值本身。因为 pandas 使用相同数量的字节表示相同类型的每个值，并且 NumPy ndarray 存储值的数量，pandas 可以快速准确地返回数字列消耗的字节数。

pandas 中的许多类型有多个子类型，可以使用较少的字节来表示每个值。例如，`float`类型有`float16`、`float32`和`float64`子类型。类型名的数字部分表示类型用来表示值的位数。例如，我们刚刚列出的子类型分别使用`2`、`4`、`8`和`16`字节。下表显示了最常见熊猫类型的子类型:

| 内存使用 | 漂浮物 | （同 Internationalorganizations）国际组织 | uint | 日期时间 | 弯曲件 | 目标 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 字节 |  | int8 | uint8 |  | 弯曲件 |  |
| 2 字节 | 浮动 16 | int16 | uint16 |  |  |  |
| 4 字节 | float32 | int32 | uint32 |  |  |  |
| 8 字节 | float64 | int64 | uint64 | 日期时间 64 |  |  |
| 可变的 |  |  |  |  |  | 目标 |

一个`int8`值用`1`字节(或`8`位)存储一个值，可以用二进制表示`256`值(`2^8`)。这意味着我们可以用这个子类型来表示从`-128`到`127`(包括`0`)的值。

我们可以使用 [`numpy.iinfo`](https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.iinfo.html) 类来验证每个整数子类型的最小值和最大值。让我们看一个例子:

```py
 import numpy as np
int_types = ["uint8", "int8", "int16"]
for it in int_types:
    print(np.iinfo(it))
```

```py
 Machine parameters for uint8----------------------------------------
min = 0
max = 255
Machine parameters for int8-----------------------------------------
min = -128
max = 127
Machine parameters for int16----------------------------------------
min = -32768
max = 32767 
```

我们在这里可以看到`uint`(无符号整数)和`int`(有符号整数)的区别。这两种类型具有相同的存储容量，但是通过只存储正值，无符号整数允许我们更有效地存储只包含正值的列。

## 使用子类型优化数字列

我们可以使用函数`pd.to_numeric()`来**向下转换**我们的数值类型。我们将使用`DataFrame.select_dtypes`只选择整数列，然后我们将优化类型并比较内存使用情况。

```py
 # We're going to be calculating memory usage a lot,
# so we'll create a function to save us some time!
def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)
gl_int = gl.select_dtypes(include=['int'])
converted_int = gl_int.apply(pd.to_numeric,downcast='unsigned')
print(mem_usage(gl_int))
print(mem_usage(converted_int))
compare_ints = pd.concat([gl_int.dtypes,converted_int.dtypes],axis=1)
compare_ints.columns = ['before','after']
compare_ints.apply(pd.Series.value_counts)
```

```py
7.87 MB
1.48 MB
```

|  | 以前 | 在...之后 |
| --- | --- | --- |
| uint8 | 圆盘烤饼 | Five |
| uint32 | 圆盘烤饼 | One |
| int64 | Six | 圆盘烤饼 |

我们可以看到内存使用量从 7.9 兆字节下降到 1.5 兆字节，降幅超过 80%。尽管对原始数据帧的整体影响并不大，因为整数列很少。

让我们对浮动列做同样的事情。

```py
 gl_float = gl.select_dtypes(include=['float'])
converted_float = gl_float.apply(pd.to_numeric,downcast='float')
print(mem_usage(gl_float))
print(mem_usage(converted_float))
compare_floats = pd.concat([gl_float.dtypes,converted_float.dtypes],axis=1)
compare_floats.columns = ['before','after']
compare_floats.apply(pd.Series.value_counts) 
```

```py
100.99 MB
50.49 MB
```

|  | 以前 | 在...之后 |
| --- | --- | --- |
| float32 | 圆盘烤饼 | Seventy-seven |
| float64 | Seventy-seven | 圆盘烤饼 |

我们可以看到，所有的浮动列都从`float64`转换成了`float32`，从而减少了 50%的内存使用。

让我们创建原始数据帧的副本，将这些优化的数字列分配到原始数据帧的位置，并查看我们现在的整体内存使用情况。

```py
 optimized_gl = gl.copy()
optimized_gl[converted_int.columns] = converted_int
optimized_gl[converted_float.columns] = converted_float
print(mem_usage(gl))
print(mem_usage(optimized_gl)) 
```

```py
861.57 MB
```

```py
804.69 MB
```

虽然我们已经极大地减少了数字列的内存使用，但总的来说，我们只将数据帧的内存使用减少了 7%。我们的大部分收益将来自优化对象类型。

在此之前，让我们仔细看看 pandas 中字符串是如何存储的，与数字类型相比

## 比较数字存储和字符串存储

`object`类型使用 Python 字符串对象表示值，部分原因是 NumPy 中缺少对缺失字符串值的支持。因为 Python 是一种高级的解释型语言，所以它对值在内存中的存储方式没有细粒度的控制。

这种限制导致字符串以碎片的方式存储，从而消耗更多的内存，并且访问速度较慢。对象列中的每个元素实际上是一个指针，包含实际值在内存中位置的“地址”。

下图显示了数字数据如何存储在 NumPy 数据类型中，以及字符串如何使用 Python 的内置类型存储。

![](img/4c23a8466b0e51a135fe482d47aea340.png)

*图改编自优帖[Python 为什么慢](https://jakevdp.github.io/blog/2014/05/09/why-python-is-slow/)。*

您可能已经注意到，我们之前的图表将`object`类型描述为使用可变数量的内存。虽然每个指针占用 1 个字节的内存，但是每个实际的字符串值使用的内存量与单独存储在 Python 中的字符串使用的内存量相同。让我们使用`sys.getsizeof()`来证明这一点，首先查看单个字符串，然后查看熊猫系列中的项目。

```py
 from sys import getsizeof
s1 = 'working out'
s2 = 'memory usage for'
s3 = 'strings in python is fun!'
s4 = 'strings in python is fun!'
for s in [s1, s2, s3, s4]:
    print(getsizeof(s))
```

```py
 60
65
74
74
```

```py
obj_series = pd.Series(['working out',
    'memory usage for',
    'strings in python is fun!',
    'strings in python is fun!'])
obj_series.apply(getsizeof) 
```

```py
 0 60
1 65
2 74
3 74
dtype: int64
```

您可以看到，存储在 pandas 系列中的字符串的大小与它们在 Python 中作为单独字符串的用法相同。

## 使用范畴优化对象类型

熊猫在 0.15 版本中引入了[种类](https://pandas.pydata.org/pandas-docs/stable/categorical.html)。`category`类型使用整数值来表示列中的值，而不是原始值。Pandas 使用单独的映射字典将整数值映射到原始值。每当一列包含有限的一组值时，这种安排就很有用。当我们将一个列转换为`category` dtype 时，pandas 使用空间效率最高的`int`子类型，它可以表示一个列中的所有唯一值。

![](img/420c8cfc6cd9b9c11c03ddd959c2bb4e.png)

为了了解我们可以在哪里使用这种类型来减少内存，让我们来看看每个对象类型的唯一值的数量。

```py
 gl_obj = gl.select_dtypes(include=['object']).copy()
gl_obj.describe()
```

|  | 星期几 | 虚拟姓名 | v _ 联赛 | hgame | h _ 联赛 | 白天夜晚 | 完成 | 丧失 | 抗议 | 公园 id | v _ 线 _ 分数 | h_line_score | 惠普 _ 裁判 _id | 惠普裁判姓名 | 1b _ 裁判 _id | 1b _ 裁判员 _ 姓名 | 2b _ 裁判 _id | 2b _ 裁判员 _ 姓名 | 3b _ 裁判 _id | 3b _ 裁判员 _ 姓名 | lf _ 裁判员 _id | lf _ 裁判员 _ 姓名 | rf _ 裁判员 _id | rf _ 裁判员 _ 姓名 | 虚拟经理 id | 虚拟经理姓名 | h _ 经理 _id | 经理姓名 | winning _ pitcher _ id | 获胜投手姓名 | losing_pitcher_id | 失去投手姓名 | 正在保存 _pitcher_id | 保存 _pitcher_name | 中奖 _ 打点 _ 击球手 _id | 获奖 _rbi_batter_id_name | v_starting_pitcher_id | v_starting_pitcher_name | h_starting_pitcher_id | h_starting_pitcher_name | 虚拟玩家 1 号 | v _ 玩家 _ 1 _ 姓名 | v _ 玩家 _2_id | v _ 玩家 _ 2 _ 姓名 | 虚拟玩家 3 号 id | v _ 玩家 _ 3 _ 姓名 | v _ 玩家 _4_id | v _ 玩家 _ 4 _ 姓名 | v _ 玩家 _5_id | v_player_5_name | v_player_6_id | v_player_6_name | v _ 玩家 _7_id | v_player_7_name | v_player_8_id | v_player_8_name | v _ 玩家 _9_id | v_player_9_name | h_player_1_id | h_player_1_name | h_player_2_id | h_player_2_name | h_player_3_id | h_player_3_name | h_player_4_id | h_player_4_name | h_player_5_id | h_player_5_name | h_player_6_id | h_player_6_name | h_player_7_id | h_player_7_name | h_player_8_id | h_player_8_name | h_player_9_id | h_player_9_name | 附加信息 | 收购信息 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 数数 | One hundred and seventy-one thousand nine hundred and seven | One hundred and seventy-one thousand nine hundred and seven | One hundred and seventy-one thousand nine hundred and seven | One hundred and seventy-one thousand nine hundred and seven | One hundred and seventy-one thousand nine hundred and seven | One hundred and forty thousand one hundred and fifty | One hundred and sixteen | One hundred and forty-five | one hundred and eighty  | One hundred and seventy-one thousand nine hundred and seven | One hundred and forty-seven thousand two hundred and seventy-one | One hundred and forty-seven thousand two hundred and seventy-one | One hundred and seventy-one thousand eight hundred and eighty-eight | One hundred and seventy-one thousand eight hundred and ninety-one | One hundred and forty-seven thousand and forty | One hundred and seventy-one thousand eight hundred and ninety-one | Eighty-eight thousand five hundred and forty | One hundred and seventy-one thousand one hundred and twenty-seven | One hundred and sixteen thousand seven hundred and twenty-three | One hundred and seventy-one thousand one hundred and thirty-five | Two hundred and three | One hundred and seventy-one thousand nine hundred and two | nine | One hundred and seventy-one thousand nine hundred and two | One hundred and seventy-one thousand nine hundred and seven | One hundred and seventy-one thousand nine hundred and seven | One hundred and seventy-one thousand nine hundred and seven | One hundred and seventy-one thousand nine hundred and seven | One hundred and forty thousand two hundred and twenty-nine | One hundred and forty thousand two hundred and twenty-nine | One hundred and forty thousand two hundred and twenty-nine | One hundred and forty thousand two hundred and twenty-nine | Forty-eight thousand and eighteen | One hundred and forty thousand eight hundred and thirty-eight | One hundred and five thousand six hundred and ninety-nine | One hundred and forty thousand eight hundred and thirty-eight | One hundred and seventy-one thousand eight hundred and sixty-three | One hundred and seventy-one thousand eight hundred and sixty-three | One hundred and seventy-one thousand eight hundred and sixty-three | One hundred and seventy-one thousand eight hundred and sixty-three | One hundred and forty thousand eight hundred and thirty-eight | One hundred and forty thousand eight hundred and thirty-eight | One hundred and forty thousand eight hundred and thirty-eight | One hundred and forty thousand eight hundred and thirty-eight | One hundred and forty thousand eight hundred and thirty-eight | One hundred and forty thousand eight hundred and thirty-eight | One hundred and forty thousand eight hundred and thirty-eight | One hundred and forty thousand eight hundred and thirty-eight | One hundred and forty thousand eight hundred and thirty-eight | One hundred and forty thousand eight hundred and thirty-eight | One hundred and forty thousand eight hundred and thirty-eight | One hundred and forty thousand eight hundred and thirty-eight | One hundred and forty thousand eight hundred and thirty-eight | One hundred and forty thousand eight hundred and thirty-eight | One hundred and forty thousand eight hundred and thirty-eight | One hundred and forty thousand eight hundred and thirty-eight | One hundred and forty thousand eight hundred and thirty-five | One hundred and forty thousand eight hundred and thirty-five | One hundred and forty thousand eight hundred and thirty-eight | One hundred and forty thousand eight hundred and thirty-eight | One hundred and forty thousand eight hundred and thirty-eight | One hundred and forty thousand eight hundred and thirty-eight | One hundred and forty thousand eight hundred and thirty-eight | One hundred and forty thousand eight hundred and thirty-eight | One hundred and forty thousand eight hundred and thirty-eight | One hundred and forty thousand eight hundred and thirty-eight | One hundred and forty thousand eight hundred and thirty-eight | One hundred and forty thousand eight hundred and thirty-eight | One hundred and forty thousand eight hundred and thirty-eight | One hundred and forty thousand eight hundred and thirty-eight | One hundred and forty thousand eight hundred and thirty-eight | One hundred and forty thousand eight hundred and thirty-eight | One hundred and forty thousand eight hundred and thirty-eight | One hundred and forty thousand eight hundred and thirty-eight | One hundred and forty thousand eight hundred and thirty-eight | One hundred and forty thousand eight hundred and thirty-eight | One thousand four hundred and fifty-six | One hundred and forty thousand eight hundred and forty-one |
| 独一无二的 | seven | One hundred and forty-eight | seven | One hundred and forty-eight | seven | Two | One hundred and sixteen | three | five | Two hundred and forty-five | Thirty-six thousand three hundred and sixty-seven | Thirty-seven thousand eight hundred and fifty-nine | One thousand one hundred and forty-nine | One thousand one hundred and forty-six | Six hundred and seventy-eight | Six hundred and seventy-eight | Three hundred and twenty-four | Three hundred and twenty-five | Three hundred and sixty-two | Three hundred and sixty-three | Thirty-one | Thirty-two | eight | nine | Six hundred and forty-eight | Six hundred and forty-eight | Six hundred and fifty-nine | Six hundred and fifty-nine | Five thousand one hundred and twenty-three | Five thousand and eighty-four | Five thousand six hundred and fifty-three | Five thousand six hundred and six | Three thousand one hundred and thirty-three | Three thousand one hundred and seventeen | Five thousand seven hundred and thirty-nine | Five thousand six hundred and seventy-four | Five thousand one hundred and ninety-three | Five thousand one hundred and twenty-nine | Five thousand one hundred and seventy | Five thousand one hundred and twenty-five | Two thousand eight hundred and seventy | Two thousand eight hundred and forty-seven | Three thousand seven hundred and nine | Three thousand six hundred and seventy-three | Two thousand nine hundred and eighty-nine | Two thousand nine hundred and sixty-four | Two thousand five hundred and eighty-one | Two thousand five hundred and sixty-three | Three thousand seven hundred and fifty-seven | Three thousand seven hundred and twenty-two | Four thousand seven hundred and ninety-four | Four thousand seven hundred and thirty-six | Five thousand three hundred and one | Five thousand two hundred and forty-one | Four thousand eight hundred and twelve | Four thousand seven hundred and sixty-three | Five thousand six hundred and forty-three | Five thousand five hundred and eighty-five | Two thousand eight hundred and two | Two thousand seven hundred and eighty-two | Three thousand six hundred and forty-eight | Three thousand six hundred and fourteen | Two thousand eight hundred and eighty-one | Two thousand eight hundred and fifty-eight | Two thousand five hundred and thirty-three | Two thousand five hundred and seventeen | Three thousand six hundred and ninety-six | Three thousand six hundred and sixty | Four thousand seven hundred and seventy-four | Four thousand seven hundred and twenty | Five thousand two hundred and fifty-three | Five thousand one hundred and ninety-seven | Four thousand seven hundred and sixty | Four thousand seven hundred and ten | Five thousand one hundred and ninety-three | Five thousand one hundred and forty-two | Three hundred and thirty-two | one |
| 顶端 | 坐 | 测链 | 荷兰 | 测链 | 荷兰 | D | 19510725,,6,6,46 | H | V | STL07 | 000000000 | 000000000 | klemb901 | 比尔挤压 | 康恩 901 | (无) | westj901 | (无) | mcgob901 | (无) | sudoe901 | (无) | gormt101 | (无) | mackc101 | 康妮·麦克 | mackc101 | 康妮·麦克 | johnw102 | 华特·强森 | rixee101 | 荷兰人伦纳德 | rivem002 | (无) | pujoa001 | (无) | younc102 | 年纪轻的 | younc102 | 年纪轻的 | suzui001 | 铃木一朗 | 福克斯 n101 | 内莉·福克斯 | speat101 | Tris 扬声器 | bottj101 | 吉姆·博顿利 | 海尔 101 | 哈里·赫尔曼 | grimc101 | 查理·格林 | grimc101 | 查理·格林 | 洛普亚 102 | 阿尔·洛佩兹 | grifa001 | 阿尔弗雷多·格里芬 | suzui001 | 铃木一朗 | 福克斯 n101 | 内莉·福克斯 | speat101 | Tris 扬声器 | gehrl101 | 卢·格里克 | 海尔 101 | 哈里·赫尔曼 | grimc101 | 查理·格林 | grimc101 | 查理·格林 | 洛普亚 102 | 阿尔·洛佩兹 | spahw101 | 沃伦·斯帕恩 | HTBF | Y |
| 频率 | Twenty-eight thousand eight hundred and ninety-one | Eight thousand eight hundred and seventy | Eighty-eight thousand eight hundred and sixty-six | Nine thousand and twenty-four | Eighty-eight thousand eight hundred and sixty-seven | Eighty-two thousand seven hundred and twenty-four | one | sixty-nine | Ninety | Seven thousand and twenty-two | Ten thousand one hundred and two | Eight thousand and twenty-eight | Three thousand five hundred and forty-five | Three thousand five hundred and forty-five | Two thousand and twenty-nine | Twenty-four thousand eight hundred and fifty-one | Eight hundred and fifteen | Eighty-two thousand five hundred and eighty-seven | One thousand one hundred and twenty-nine | Fifty-four thousand four hundred and twelve | Thirty | One hundred and seventy-one thousand six hundred and ninety-nine | Two | One hundred and seventy-one thousand eight hundred and ninety-three | Three thousand nine hundred and one | Three thousand nine hundred and one | Three thousand eight hundred and forty-eight | Three thousand eight hundred and forty-eight | Three hundred and eighty-five | Three hundred and eighty-five | Two hundred and fifty-one | Two hundred and ninety-five | Five hundred and twenty-three | Ninety-two thousand eight hundred and twenty | Two hundred and eighty-eight | Thirty-five thousand one hundred and thirty-nine | Four hundred and three | Four hundred and forty-one | Four hundred and twelve | Four hundred and fifty-one | Eight hundred and ninety-three | Eight hundred and ninety-three | Eight hundred and fifty-two | Eight hundred and fifty-two | One thousand two hundred and twenty-four | One thousand two hundred and twenty-four | Eight hundred and sixteen | Eight hundred and sixteen | Six hundred and sixty-three | Six hundred and sixty-three | Four hundred and sixty-five | Four hundred and sixty-five | Four hundred and eighty-five | Four hundred and eighty-five | Six hundred and eighty-seven | Six hundred and eighty-seven | Three hundred and thirty-three | Three hundred and thirty-three | Nine hundred and twenty-seven | Nine hundred and twenty-seven | Eight hundred and fifty-nine | Eight hundred and fifty-nine | One thousand one hundred and sixty-five | One thousand one hundred and sixty-five | Seven hundred and fifty-two | Seven hundred and fifty-two | Six hundred and twelve | Six hundred and twelve | Four hundred and twenty-seven | Four hundred and twenty-seven | Four hundred and ninety-one | Four hundred and ninety-one | Six hundred and seventy-six | Six hundred and seventy-six | Three hundred and thirty-nine | Three hundred and thirty-nine | One thousand one hundred and twelve | One hundred and forty thousand eight hundred and forty-one |

快速浏览一下就会发现，在我们的数据集中，相对于大约 172，000 个游戏，许多列几乎没有唯一值。

在我们深入研究之前，我们将从选择一个对象列开始，看看当我们将它转换为分类类型时，幕后发生了什么。我们将使用数据集的第二列`day_of_week`。

看着上面的桌子。我们可以看到它只包含七个唯一值。我们将使用`.astype()`方法将其转换为分类。

```py
 dow = gl_obj.day_of_week
print(dow.head())
dow_cat = dow.astype('category')
print(dow_cat.head())
```

```py
 0 Thu
1 Fri
2 Sat
3 Mon
4 Tue
Name: day_of_week, dtype: object
0 Thu
1 Fri
2 Sat
3 Mon
4 Tue
Name: day_of_week, dtype: category
Categories (7, object): [Fri, Mon, Sat, Sun, Thu, Tue, Wed] 
```

如您所见，除了列的类型发生了变化之外，数据看起来完全相同。让我们来看看到底发生了什么。

在下面的代码中，我们使用`Series.cat.codes`属性返回整数值，`category`类型使用这些整数值来表示每个值。

```py
dow_cat.head().cat.codes
```

```py
 0 4
1 0
2 2
3 1
4 5
dtype: int8
```

您可以看到每个惟一值都被赋予了一个整数，该列的底层数据类型现在是`int8`。该列没有任何丢失的值，但是如果有，`category`子类型通过将它们设置为`-1`来处理丢失的值。

最后，让我们看看这个列在转换为`category`类型之前和之后的内存使用情况。

```py
 print(mem_usage(dow))
print(mem_usage(dow_cat))
```

```py
9.84 MB
0.16 MB
```

我们的内存使用量从 9.8MB 减少到了 0.16MB，减少了 98%！请注意，这个特定的列可能代表了我们的最佳情况之一——一个包含大约 172，000 个条目的列，其中只有 7 个唯一值。

虽然将所有列都转换成这种类型听起来很吸引人，但是了解其中的利弊也很重要。最大的问题是无法进行数值计算。如果不先转换成真正的数字数据类型，我们就不能对`category`列进行算术运算，也不能使用像`Series.min()`和`Series.max()`这样的方法。

我们应该坚持将`category`类型主要用于`object`列，其中不到 50%的值是唯一的。如果一列中的所有值都是唯一的，那么`category`类型将会使用*更多的*内存。这是因为除了整数类别代码之外，该列还存储所有的原始字符串值。你可以在 [pandas 文档](https://pandas.pydata.org/pandas-docs/stable/categorical.html#gotchas)中阅读更多关于`category`类型的限制。

我们将编写一个循环来迭代每个`object`列，检查唯一值的数量是否小于 50%，如果是，则将其转换为 category 类型。

```py
 converted_obj = pd.DataFrame()
for col in gl_obj.columns:
    num_unique_values = len(gl_obj[col].unique())
    num_total_values = len(gl_obj[col])
    if num_unique_values / num_total_values < 0.5:
        converted_obj.loc[:,col] = gl_obj[col].astype('category')
    else:
        converted_obj.loc[:,col] = gl_obj[col]
```

和以前一样，

```py
 print(mem_usage(gl_obj))
print(mem_usage(converted_obj))
compare_obj = pd.concat([gl_obj.dtypes,converted_obj.dtypes],axis=1)
compare_obj.columns = ['before','after']
compare_obj.apply(pd.Series.value_counts)
```

```py
752.72 MB
51.67 MB
```

|  | 以前 | 在...之后 |
| --- | --- | --- |
| 目标 | Seventy-eight | 圆盘烤饼 |
| 种类 | 圆盘烤饼 | Seventy-eight |

在这种情况下，我们所有的对象列都被转换为`category`类型，然而这并不是所有数据集的情况，所以您应该确保使用上面的过程进行检查。

此外，我们的`object`列的内存使用从 752MB 减少到 52MB，减少了 93%。让我们将这一点与数据帧的其余部分结合起来，看看我们相对于开始时的 861MB 内存使用量处于什么位置。

```py
 optimized_gl[converted_obj.columns] = converted_obj
mem_usage(optimized_gl)
```

```py
'103.64 MB'
```

哇，我们真的取得了一些进展！我们还可以做一个优化——如果你还记得我们的类型表，有一个`datetime`类型可以用于我们数据集的第一列。

```py
 date = optimized_gl.date
print(mem_usage(date))
date.head()
```

```py
0.66 MB
```

```py
0 18710504
1 18710505
2 18710506
3 18710508
4 18710509
Name: date, dtype: uint32
```

您可能记得这是作为整数类型读入的，并且已经优化到了`unint32`。因此，将它转换成`datetime`实际上会使它的内存使用加倍，因为`datetime`类型是 64 位类型。无论如何，将它转换成`datetime`是有价值的，因为它将允许我们更容易地进行时间序列分析。

我们将使用`pandas.to_datetime()`函数进行转换，使用`format`参数告诉它我们的日期数据存储在`YYYY-MM-DD`中。

```py
 optimized_gl['date'] = pd.to_datetime(date,format='%Y%m%d')
print(mem_usage(optimized_gl))
optimized_gl.date.head()
```

```py
104.29 MB
```

```py
 0 1871-05-04
1 1871-05-05
2 1871-05-06
3 1871-05-08
4 1871-05-09
Name: date, dtype: datetime64[ns]
```

## 读取数据时选择类型

到目前为止，我们已经探索了减少现有数据帧的内存占用的方法。通过首先读取数据帧，然后迭代节省内存的方法，我们能够更好地了解每次优化可以节省的内存量。但是，正如我们在本课前面提到的，我们通常没有足够的内存来表示数据集中的所有值。当我们甚至不能首先创建数据帧时，我们如何应用节省内存的技术呢？

幸运的是，当我们读取。 [pandas.read_csv()](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) 函数有几个不同的参数允许我们这样做。`dtype`参数接受以(string)列名作为键、以 NumPy 类型对象作为值的字典。

首先，我们将把每一列的最终类型存储在一个包含列名键的字典中，首先删除日期列，因为它需要单独处理。

```py
 dtypes = optimized_gl.drop('date',axis=1).dtypes
dtypes_col = dtypes.index
dtypes_type = [i.name for i in dtypes.values]
column_types = dict(zip(dtypes_col, dtypes_type))
# rather than print all 161 items, we'll
# sample 10 key/value pairs from the dict
# and print it nicely using prettyprint
preview = first2pairs = {key:value for key,value in list(column_types.items())[:10]}
import pprintpp
pp = pp = pprint.PrettyPrinter(indent=4)
pp.pprint(preview)
```

```py
 {
'acquisition_info': 'category',
'h_caught_stealing': 'float32',
'h_player_1_name': 'category',
'h_player_9_name': 'category',
'v_assists': 'float32',
'v_first_catcher_interference': 'float32',
'v_grounded_into_double': 'float32',
'v_player_1_id': 'category',
'v_player_3_id': 'category',
'v_player_5_id': 'category'
}
```

现在，我们可以使用字典和几个日期参数，用几行代码读入正确类型的数据:

```py
 read_and_optimized = pd.read_csv('game_logs.csv',dtype=column_types,parse_dates=['date'],infer_datetime_format=True)
print(mem_usage(read_and_optimized))
read_and_optimized.head()
```

```py
104.28 MB
```

|  | 日期 | 游戏的数量 | 星期几 | 虚拟姓名 | v _ 联赛 | v _ 游戏 _ 号码 | hgame | h _ 联赛 | h _ 游戏 _ 号码 | v _ 分数 | h 分数 | 长度 _ 超时 | 白天夜晚 | 完成 | 丧失 | 抗议 | 公园 id | 出席 | 长度 _ 分钟 | v _ 线 _ 分数 | h_line_score | 蝙蝠 | v_hits | v _ 双打 | v_triples | v _ 本垒打 | 虚拟银行 | v _ 牺牲 _ 命中 | v _ 牺牲 _ 苍蝇 | 垂直命中间距 | v_walks | v _ 有意行走 | v _ 删除线 | 五 _ 被盗 _ 基地 | 偷窃被抓 | v _ 接地 _ 进 _ 双 | v _ 优先 _ 捕捉器 _ 干扰 | v _ 左 _ 上 _ 下 | v _ 投手 _ 已用 | 个人获得的跑步记录 | v _ team _ earned _ runs | v _ wild _ pitches | v_balks | v _ 输出 | vassistx | v _ 错误 | v _ 传球 _ 球 | 双人游戏 | 三重播放 | 蝙蝠 | 点击次数 | h _ 双打 | h_triples | h _ 本垒打 | h_rbi | h _ 牺牲 _ 命中 | h _ 祭祀 _ 苍蝇 | h _ hit _ by _ 音高 | h _ 行走 | h _ 有意行走 | h _ 删除线 | h _ 被盗 _ 基地 | 偷东西被抓 | h _ 接地 _ 进 _ 双 | h _ 优先 _ 捕捉器 _ 干扰 | 左满垒 | h_pitchers_used | 个人获得的跑步记录 | h _ team _ earned _ runs | h _ wild _ pitches | h_balks | h _ 输出 | h _ 助攻 | h _ 错误 | h _ 传球 _ 球 | 双人游戏 | 三重播放 | 惠普 _ 裁判 _id | 惠普裁判姓名 | 1b _ 裁判 _id | 1b _ 裁判员 _ 姓名 | 2b _ 裁判 _id | 2b _ 裁判员 _ 姓名 | 3b _ 裁判 _id | 3b _ 裁判员 _ 姓名 | lf _ 裁判员 _id | lf _ 裁判员 _ 姓名 | rf _ 裁判员 _id | rf _ 裁判员 _ 姓名 | 虚拟经理 id | 虚拟经理姓名 | h _ 经理 _id | 经理姓名 | winning _ pitcher _ id | 获胜投手姓名 | losing_pitcher_id | 失去投手姓名 | 正在保存 _pitcher_id | 保存 _pitcher_name | 中奖 _ 打点 _ 击球手 _id | 获奖 _rbi_batter_id_name | v_starting_pitcher_id | v_starting_pitcher_name | h_starting_pitcher_id | h_starting_pitcher_name | 虚拟玩家 1 号 | v _ 玩家 _ 1 _ 姓名 | 虚拟玩家 1 定义位置 | v _ 玩家 _2_id | v _ 玩家 _ 2 _ 姓名 | 虚拟玩家 2 定义位置 | 虚拟玩家 3 号 id | v _ 玩家 _ 3 _ 姓名 | 虚拟玩家 3 定义位置 | v _ 玩家 _4_id | v _ 玩家 _ 4 _ 姓名 | 虚拟玩家 4 定义位置 | v _ 玩家 _5_id | v_player_5_name | 虚拟玩家 5 定义位置 | v_player_6_id | v_player_6_name | 虚拟玩家 6 定义位置 | v _ 玩家 _7_id | v_player_7_name | 虚拟玩家 7 定义位置 | v_player_8_id | v_player_8_name | 虚拟玩家 8 定义位置 | v _ 玩家 _9_id | v_player_9_name | 虚拟玩家 9 定义位置 | h_player_1_id | h_player_1_name | h _ 播放器 _1_def_pos | h_player_2_id | h_player_2_name | h _ 播放器 _2_def_pos | h_player_3_id | h_player_3_name | h _ 播放器 _3_def_pos | h_player_4_id | h_player_4_name | h _ 播放器 _4_def_pos | h_player_5_id | h_player_5_name | h _ 播放器 _5_def_pos | h_player_6_id | h_player_6_name | h _ 播放器 _6_def_pos | h_player_7_id | h_player_7_name | h _ 播放器 _7_def_pos | h_player_8_id | h_player_8_name | h _ 播放器 _8_def_pos | h_player_9_id | h_player_9_name | h _ 播放器 _9_def_pos | 附加信息 | 收购信息 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Zero | 1871-05-04 | Zero | 星期四 | CL1 | 钠 | one | FW1 | 钠 | one | Zero | Two | Fifty-four | D | 圆盘烤饼 | 圆盘烤饼 | 圆盘烤饼 | 对于 01 | Two hundred | One hundred and twenty | 000000000 | 010010000 | Thirty | Four | One | Zero | Zero | Zero | Zero | Zero | Zero | One | -1.0 | Six | One | -1.0 | -1.0 | -1.0 | Four | One | One | One | Zero | Zero | Twenty-seven | Nine | Zero | Three | Zero | Zero | Thirty-one | Four | One | Zero | Zero | Two | Zero | Zero | Zero | One | -1.0 | Zero | Zero | -1.0 | -1.0 | -1.0 | Three | One | Zero | Zero | Zero | Zero | Twenty-seven | Three | Three | One | One | Zero | boakj901 | 约翰·布莱克 | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | paboc101 | 查理·帕博尔 | lennb101 | 比尔·列侬 | mathb101 | 鲍比·马修斯 | 普拉塔 101 | 艾尔·普拉特 | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | 普拉塔 101 | 艾尔·普拉特 | mathb101 | 鲍比·马修斯 | whitd102 | 迪肯·怀特 | Two | kimbg101 | 吉恩·金博尔 | Four | paboc101 | 查理·帕博尔 | Seven | allia101 | 阿特·艾利森 | Eight | 白色 104 | 埃尔默·怀特 | Nine | 普拉塔 101 | 艾尔·普拉特 | One | 萨图特 101 | 以斯拉·萨顿 | Five | 卡莱 102 | 吉姆·卡尔顿 | Three | bassj101 | 约翰·巴斯 | Six | selmf101 | 弗兰克·塞尔曼 | Five | mathb101 | 鲍比·马修斯 | One | 福尔马林 101 | Jim Foran | Three | goldw101 | 沃利·戈德史密斯 | Six | lennb101 | 比尔·列侬 | Two | 职业 101 | 汤姆·凯里 | Four | 碎肉 101 | 埃德·明彻 | Seven | mcdej101 | 詹姆斯·麦克德莫特 | Eight | kellb105 | 比尔·凯利 | Nine | 圆盘烤饼 | Y |
| one | 1871-05-05 | Zero | Fri | BS1 | 钠 | one | WS3 | 钠 | one | Twenty | Eighteen | Fifty-four | D | 圆盘烤饼 | 圆盘烤饼 | 圆盘烤饼 | WAS01 | Five thousand | One hundred and forty-five | One hundred and seven million four hundred and thirty-five | Six hundred and forty million one hundred and thirteen thousand and thirty | Forty-one | Thirteen | One | Two | Zero | Thirteen | Zero | Zero | Zero | Eighteen | -1.0 | Five | Three | -1.0 | -1.0 | -1.0 | Twelve | One | Six | Six | One | Zero | Twenty-seven | Thirteen | Ten | One | Two | Zero | Forty-nine | Fourteen | Two | Zero | Zero | Eleven | Zero | Zero | Zero | Ten | -1.0 | Two | One | -1.0 | -1.0 | -1.0 | Fourteen | One | Seven | Seven | Zero | Zero | Twenty-seven | Twenty | Ten | Two | Three | Zero | 多布斯 901 | 亨利·多布森 | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | wrigh101 | 哈里·赖特 | younn801 | 尼克·杨 | spala101 | 阿尔·斯伯丁 | 巴西 102 | 阿萨·布雷纳德 | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | spala101 | 阿尔·斯伯丁 | 巴西 102 | 阿萨·布雷纳德 | 箭牌 101 | 乔治·赖特 | Six | 巴恩斯 102 | 罗斯·巴恩斯 | Four | birdd102 | 戴夫·伯索尔 | Nine | mcvec101 | 卡尔·麦克维 | Two | wrigh101 | 哈里·赖特 | Eight | goulc101 | 查理·古尔德 | Three | 沙赫 101 | 哈里·斯查费 | Five | conef101 | 弗雷德·科恩 | Seven | spala101 | 阿尔·斯伯丁 | One | watef102 | 弗雷德·沃特曼 | Five | forcd101 | 戴维力量 | Six | mille105 | 埃弗里特·米尔斯 | Three | allid101 | 道格·艾利森 | Two | hallg101 | 乔治·霍尔 | Seven | 利昂娜 101 | 安迪·伦纳德 | Four | 巴西 102 | 阿萨·布雷纳德 | One | burrh101 | 亨利·巴勒斯 | Nine | berth101 | 亨利·贝思龙 | Eight | HTBF | Y |
| Two | 1871-05-06 | Zero | 坐 | CL1 | 钠 | Two | RC1 | 钠 | one | Twelve | four | Fifty-four | D | 圆盘烤饼 | 圆盘烤饼 | 圆盘烤饼 | RCK01 | One thousand | One hundred and forty | Six hundred and ten million twenty thousand and three | 010020100 | Forty-nine | Eleven | One | One | Zero | Eight | Zero | Zero | Zero | Zero | -1.0 | One | Zero | -1.0 | -1.0 | -1.0 | Ten | One | Zero | Zero | Two | Zero | Twenty-seven | Twelve | Eight | Five | Zero | Zero | Thirty-six | Seven | Two | One | Zero | Two | Zero | Zero | Zero | Zero | -1.0 | Three | Five | -1.0 | -1.0 | -1.0 | Five | One | Three | Three | One | Zero | Twenty-seven | Twelve | Thirteen | Three | Zero | Zero | mawnj901 | J.H .曼尼 | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | paboc101 | 查理·帕博尔 | hasts101 | 斯科特·黑斯廷斯 | 普拉塔 101 | 艾尔·普拉特 | 鱼 c102 | 切诺基费希尔 | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | 普拉塔 101 | 艾尔·普拉特 | 鱼 c102 | 切诺基费希尔 | whitd102 | 迪肯·怀特 | Two | kimbg101 | 吉恩·金博尔 | Four | paboc101 | 查理·帕博尔 | Seven | allia101 | 阿特·艾利森 | Eight | 白色 104 | 埃尔默·怀特 | Nine | 普拉塔 101 | 艾尔·普拉特 | One | 萨图特 101 | 以斯拉·萨顿 | Five | 卡莱 102 | 吉姆·卡尔顿 | Three | bassj101 | 约翰·巴斯 | Six | mackd101 | 丹尼·麦克 | Three | addyb101 | 鲍勃·艾迪 | Four | 鱼 c102 | 切诺基费希尔 | One | hasts101 | 斯科特·黑斯廷斯 | Eight | ham-r101 | Ralph Ham | Five | ansoc101 | 安森角 | Two | sagep101 | 小马的事 | Six | birdg101 | 乔治·伯德 | Seven | 搅拌 101 | Gat Stires | Nine | 圆盘烤饼 | Y |
| three | 1871-05-08 | Zero | 孟人 | CL1 | 钠 | three | CH1 | 钠 | one | Twelve | Fourteen | Fifty-four | D | 圆盘烤饼 | 圆盘烤饼 | 圆盘烤饼 | CHI01 | Five thousand | One hundred and fifty | One hundred and one million four hundred and three thousand one hundred and eleven | 077000000 | Forty-six | Fifteen | Two | One | Two | Ten | Zero | Zero | Zero | Zero | -1.0 | One | Zero | -1.0 | -1.0 | -1.0 | Seven | One | Six | Six | Zero | Zero | Twenty-seven | Fifteen | Eleven | Six | Zero | Zero | Forty-three | Eleven | Two | Zero | Zero | Eight | Zero | Zero | Zero | Four | -1.0 | Two | One | -1.0 | -1.0 | -1.0 | Six | One | Four | Four | Zero | Zero | Twenty-seven | Fourteen | Seven | Two | Zero | Zero | willg901 | 加德纳·威拉德 | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | paboc101 | 查理·帕博尔 | woodj106 | 吉米·伍德 | zettg101 | 乔治·蔡特林 | 普拉塔 101 | 艾尔·普拉特 | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | 普拉塔 101 | 艾尔·普拉特 | zettg101 | 乔治·蔡特林 | whitd102 | 迪肯·怀特 | Two | kimbg101 | 吉恩·金博尔 | Four | paboc101 | 查理·帕博尔 | Seven | allia101 | 阿特·艾利森 | Eight | 白色 104 | 埃尔默·怀特 | Nine | 普拉塔 101 | 艾尔·普拉特 | One | 萨图特 101 | 以斯拉·萨顿 | Five | 卡莱 102 | 吉姆·卡尔顿 | Three | bassj101 | 约翰·巴斯 | Six | mcatb101 | 布·麦卡蒂 | Three | kingm101 | 马歇尔·金 | Eight | hodec101 | 查理·霍兹 | Two | woodj106 | 吉米·伍德 | Four | simmj101 | 乔·西蒙斯 | Nine | 小精灵 101 | 汤姆·福利 | Seven | 粗呢 101 | 艾德·达菲 | Six | pinke101 | 艾德平克曼 | Five | zettg101 | 乔治·蔡特林 | One | 圆盘烤饼 | Y |
| four | 1871-05-09 | Zero | 周二 | BS1 | 钠 | Two | TRO | 钠 | one | nine | five | Fifty-four | D | 圆盘烤饼 | 圆盘烤饼 | 圆盘烤饼 | TRO01 | Three thousand two hundred and fifty | One hundred and forty-five | 000002232 | One hundred and one million and three thousand | Forty-six | Seventeen | Four | One | Zero | Six | Zero | Zero | Zero | Two | -1.0 | Zero | One | -1.0 | -1.0 | -1.0 | Twelve | One | Two | Two | Zero | Zero | Twenty-seven | Twelve | Five | Zero | One | Zero | Thirty-six | Nine | Zero | Zero | Zero | Two | Zero | Zero | Zero | Three | -1.0 | Zero | Two | -1.0 | -1.0 | -1.0 | Seven | One | Three | Three | One | Zero | Twenty-seven | Eleven | Seven | Three | Zero | Zero | leroi901 | 艾萨克·勒罗伊 | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | wrigh101 | 哈里·赖特 | pikel101 | 唇矛 | spala101 | 阿尔·斯伯丁 | 麦克莫吉 101 | 约翰·麦克穆林 | 圆盘烤饼 | (无) | 圆盘烤饼 | (无) | spala101 | 阿尔·斯伯丁 | 麦克莫吉 101 | 约翰·麦克穆林 | 箭牌 101 | 乔治·赖特 | Six | 巴恩斯 102 | 罗斯·巴恩斯 | Four | birdd102 | 戴夫·伯索尔 | Nine | mcvec101 | 卡尔·麦克维 | Two | wrigh101 | 哈里·赖特 | Eight | goulc101 | 查理·古尔德 | Three | 沙赫 101 | 哈里·斯查费 | Five | conef101 | 弗雷德·科恩 | Seven | spala101 | 阿尔·斯伯丁 | One | flync101 | 快船弗林 | Nine | mcgem101 | 迈克·麦吉瑞 | Two | yorkt101 | 汤姆·约克 | Eight | 麦克莫吉 101 | 约翰·麦克穆林 | One | 金斯 101 | 史蒂夫·金 | Seven | beave101 | 爱德华·比弗斯 | Four | 贝尔 s101 | 史蒂夫·贝伦 | Five | pikel101 | 唇矛 | Three | cravb101 | 比尔·克雷默 | Six | HTBF | Y |

通过优化列，我们成功地将 pandas 的内存使用量从 861.6 MB 减少到 104.28 MB，降幅高达 88%。

## 分析棒球比赛

现在我们已经优化了我们的数据，我们可以执行一些分析。我们先来看一下游戏天数的分布。

```py
 optimized_gl['year'] = optimized_gl.date.dt.year
games_per_day = optimized_gl.pivot_table(index='year',columns='day_of_week',values='date',aggfunc=len)
games_per_day = games_per_day.divide(games_per_day.sum(axis=1),axis=0)
ax = games_per_day.plot(kind='area',stacked='true')
ax.legend(loc='upper right')
ax.set_ylim(0,1)
plt.show()
```

![](img/48a0053ab5c8683b187fa1ad50dde295.png)

我们可以看到，在 20 世纪 20 年代以前，星期天的棒球比赛很少，直到上世纪下半叶才逐渐流行起来。

我们也可以很清楚地看到，在过去的 50 年里，游戏日的分布相对稳定。

让我们也来看看游戏长度在这些年里是如何变化的。

```py
 game_lengths = optimized_gl.pivot_table(index='year', values='length_minutes')
game_lengths.reset_index().plot.scatter('year','length_minutes')
plt.show() 
```

![](img/d4b7b138ede0aaa87c780829c3fd0661.png)

看起来棒球比赛从 20 世纪 40 年代开始持续变长。

## 总结和后续步骤

我们已经了解了 pandas 如何使用不同类型存储数据，然后我们利用这些知识，通过使用一些简单的技术，将 pandas 数据帧的内存使用量减少了近 90%:

*   将数字列向下转换为更有效的类型。
*   将字符串列转换为分类类型。

如果你想更深入地研究熊猫中更大的数据，这篇博文中的许多内容都可以在我们的互动课程[处理熊猫课程](https://app.dataquest.io/m/163)中的大数据集中找到，你可以免费开始。

## 获取免费的数据科学资源

免费注册获取我们的每周时事通讯，包括数据科学、 **Python** 、 **R** 和 **SQL** 资源链接。此外，您还可以访问我们免费的交互式[在线课程内容](/data-science-courses)！

[SIGN UP](https://app.dataquest.io/signup)