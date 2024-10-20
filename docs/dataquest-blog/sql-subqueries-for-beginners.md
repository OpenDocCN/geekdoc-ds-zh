# SQL 子查询:初学者指南(带代码示例)

> 原文：<https://www.dataquest.io/blog/sql-subqueries-for-beginners/>

June 10, 2022![SQL Subqueries](img/602b4fc38b359549cfbb83f693d050d2.png)

## 每个数据科学家都需要了解 SQL 数据库，包括子查询。下面介绍一下。

在本文中，我们将介绍 SQL 子查询的基础知识，它们的语法，它们如何有用，以及在查询数据库时何时以及如何使用它们。

本文假设您对使用 SQL 选择数据有一些基本的了解，比如数据分组、聚合函数、过滤和基本连接。

### 什么是子查询

子查询只不过是另一个查询中的一个查询。我们主要使用它们向主查询结果添加新列、创建过滤器或创建从中选择数据的统一源。

子查询将总是在括号中，并且它可以出现在主查询中的不同位置，这取决于目标—通常在`SELECT`、`FROM`或`WHERE`类中。此外，子查询的数量是无限的，这意味着您可以根据需要拥有任意多的嵌套查询。

### 数据库

为了编写一些真正的 SQL 代码，我们将使用 [Chinook 数据库](https://github.com/lerocha/chinook-database)作为例子。这是一个示例数据库，可用于多种类型的数据库。

该数据库包含关于一个虚构的数字音乐商店的信息，例如关于该音乐商店的艺术家、歌曲、播放列表、音乐流派和专辑的数据，以及关于该商店的员工、客户和购买的信息。

这是数据库的模式，因此您可以更好地理解我们将要编写的查询是如何工作的:

![chinook-schema.svg](img/671be3887cb11447acef13e49bb729fe.png)

### 创建新列的子查询

子查询的第一个用例包括使用它向主查询的输出中添加新列。语法看起来是这样的:

```py
 SELECT column_1,
       columns_2,
       (SELECT
            ...
        FROM table_2
        GROUP BY 1)
FROM table_1
GROUP BY 1
```

我们来看一个实际的例子。

在这里，我们希望看到应用程序中用户添加每首歌曲的播放列表的数量。

主查询返回两列:歌曲名称和用户添加的播放列表数量。需要子查询的是第二列。子查询在这里是必要的，因为我们必须将分配给播放列表的`track_id`与曲目表中的`track_id`进行匹配，然后对每个曲目进行计数。

```py
 SELECT t.name,
    (SELECT 
         count(playlist_id)
    FROM playlist_track pt
    WHERE pt.track_id = t.track_id
    ) as number_of_playlists
FROM track t 
GROUP BY 1
ORDER BY number_of_playlists DESC
LIMIT 50
```

然后我们得到这个输出:

| 名字 | 播放列表的数量 |
| --- | --- |
| 《仲夏夜之梦》,作品 61。配乐:第七首《夜曲》 | five |
| BWV 988“戈德堡变奏曲”:咏叹调 | five |
| 万福马利亚。亦称 HAIL　MARY | five |
| 卡门:序曲 | five |
| 卡米娜·布拉纳:哦，福尔图娜 | five |
| 乡村骑士\行动\间奏曲 | five |
| F 小调第二钢琴协奏曲，作品 21: II。稍缓慢曲 | five |
| G 大调小提琴、弦乐和通奏低音协奏曲，作品 3，第 9 首:第一乐章:快板 | five |
| 来自地球的歌，来自于青春 | five |
| 死亡之旅:女武神之旅 | five |
| 魔笛 k 620 "地狱复仇在我心中沸腾" | five |
| 绿袖幻想曲 | five |
| 伊图斯:你们崇拜上帝 | five |
| 木星，欢乐的使者 | five |
| 卡累利阿组曲，作品 11: 2。叙事曲 | five |
| Koyaanisqatsi | five |
| 耶利米哀歌，第一集 | five |
| metopes，第 29 页:Calypso | five |
| 梅女士，德乌斯 | five |

### 做数学

创建新列时，子查询对于执行某些计算可能也很有用。当然，如果是这种情况，子查询的输出必须是一个数字。

在下一个查询中，我们想知道数据库中每种风格的曲目所占的百分比。语法与上一个示例基本相同，只是子查询是创建新列的一部分，而不是整个新列。

对于这个任务，我们需要将每个流派的歌曲数量除以曲目表中的歌曲总数。我们可以通过以下查询轻松访问曲目总数:

```py
 SELECT 
    count(*) as total_tracks
FROM track

| total_tracks |
|--------------|
| 3503         |
```

我们可以使用以下查询找到每个流派的曲目总数:

```py
 SELECT 
    g.name as genre,
    count(t.track_id) as number_of_tracks
FROM genre g
INNER JOIN track t on g.genre_id = t.genre_id
GROUP BY 1
ORDER BY 2 DESC
```

| 类型 | 轨道数量 |
| --- | --- |
| 岩石 | One thousand two hundred and ninety-seven |
| 拉丁语 | Five hundred and seventy-nine |
| 金属 | Three hundred and seventy-four |
| 另类&朋克 | Three hundred and thirty-two |
| 爵士乐 | One hundred and thirty |
| 电视节目 | Ninety-three |
| 布鲁斯音乐 | Eighty-one |
| 经典的 | Seventy-four |
| 戏剧 | Sixty-four |
| R&B/灵魂 | Sixty-one |
| 瑞格舞(西印度群岛的一种舞蹈及舞曲) | Fifty-eight |
| 流行音乐 | Forty-eight |
| 配乐 | Forty-three |
| 供选择的 | Forty |
| 嘻哈/说唱 | Thirty-five |
| 电子/舞蹈 | Thirty |
| 重金属 | Twenty-eight |
| 世界 | Twenty-eight |
| 科幻与幻想 | Twenty-six |
| 轻松音乐 | Twenty-four |
| 喜剧 | Seventeen |
| 波沙·诺瓦 | Fifteen |
| 科幻小说 | Thirteen |
| 摇滚乐 | Twelve |
| 歌剧 | one |

如果我们将这两个查询组合起来，使第一个查询成为子查询，则输出是每个流派的歌曲百分比:

```py
 SELECT 
    g.name as genre,
    round(cast(count(t.track_id) as float) / (SELECT count(*) FROM track), 2) as perc
FROM genre g
INNER JOIN track t on g.genre_id = t.genre_id
GROUP BY 1
ORDER BY 2 DESC
```

| 类型 | 滤液（percolate 的简写） |
| --- | --- |
| 岩石 | Zero point three seven |
| 拉丁语 | Zero point one seven |
| 金属 | Zero point one one |
| 另类&朋克 | Zero point zero nine |
| 爵士乐 | Zero point zero four |
| 电视节目 | Zero point zero three |
| 布鲁斯音乐 | Zero point zero two |
| 经典的 | Zero point zero two |
| 戏剧 | Zero point zero two |
| R&B/灵魂 | Zero point zero two |
| 瑞格舞(西印度群岛的一种舞蹈及舞曲) | Zero point zero two |
| 供选择的 | Zero point zero one |
| 轻松音乐 | Zero point zero one |
| 电子/舞蹈 | Zero point zero one |
| 重金属 | Zero point zero one |
| 嘻哈/说唱 | Zero point zero one |
| 流行音乐 | Zero point zero one |
| 科幻与幻想 | Zero point zero one |
| 配乐 | Zero point zero one |
| 世界 | Zero point zero one |
| 波沙·诺瓦 | Zero |
| 喜剧 | Zero |
| 歌剧 | Zero |
| 摇滚乐 | Zero |
| 科幻小说 | Zero |

### 作为筛选的子查询

使用 SQL 子查询作为主查询的过滤器是我最喜欢的用例之一。在这个场景中，子查询将位于`WHERE`子句中，我们可以根据子查询的输出，使用`IN`、`=`、`<>`、`>`和`<`等操作符进行过滤。

这是语法:

```py
 SELECT
    column_1,
    columns_2
FROM table_1
WHERE column_1 in 
    (SELECT
        ...
    FROM table_2)
```

在我们的例子中，假设我们想知道每个雇员被分配到多少个至少花费 100 美元的顾客。让我们分两步来做。

首先，让我们获得每个雇员的客户数量。这是一个简单的查询。

```py
SELECT employee_id,
       e.last_name,
       count(distinct customer_id) as number_of_customers
FROM employee e 
INNER JOIN customer c on e.employee_id = c.support_rep_id
GROUP BY 1,2
ORDER BY 3 DESC
```

这是输出:

| 员工 id | 姓氏 | 客户数量 |
| --- | --- | --- |
| three | 雄孔雀 | Twenty-one |
| four | 公园 | Twenty |
| five | 约翰逊 | Eighteen |

现在，让我们看看哪些顾客在店里至少消费了 100 美元。这是查询:

```py
 SELECT
    c.customer_id,
    round(sum(i.total), 2) as total
FROM customer c
INNER JOIN invoice i on c.customer_id = i.customer_id
GROUP BY c.customer_id
HAVING sum(i.total) > 100
ORDER BY 2 DESC
```

这是输出:

| 客户标识 | 总数 |
| --- | --- |
| five | One hundred and forty-four point five four |
| six | One hundred and twenty-eight point seven |
| Forty-six | One hundred and fourteen point eight four |
| Fifty-eight | One hundred and eleven point eight seven |
| one | One hundred and eight point nine |
| Thirteen | One hundred and six point nine two |
| Thirty-four | One hundred and two point nine six |

现在，为了组合这两个查询，第一个将是主查询，第二个将在`WHERE`子句中过滤主查询。

它是这样工作的:

```py
 SELECT employee_id,
       e.last_name,
       count(distinct customer_id) as number_of_customers
FROM employee e 
INNER JOIN customer c on e.employee_id = c.support_rep_id
WHERE customer_id in (
        SELECT
            c.customer_id
        FROM customer c
        INNER JOIN invoice i on c.customer_id = i.customer_id
        GROUP BY c.customer_id
        HAVING sum(i.total) > 100)
GROUP BY 1, 2
ORDER BY 3 DESC
```

这是最终输出:

| 员工 id | 姓氏 | 客户数量 |
| --- | --- | --- |
| three | 雄孔雀 | three |
| four | 公园 | three |
| five | 约翰逊 | one |

请注意两点:

1.  当将查询 2 放在主查询的`WHERE`子句中时，我们删除了`total_purchased`列。这是因为我们希望这个查询只返回一列，也就是主查询用作过滤器的那一列。如果我们没有这样做，我们将会看到这样的错误消息(取决于 SQL 的版本):

```py
 sub-select returns 2 columns - expected 1
```

2.  我们使用了`IN`操作符。顾名思义，我们想要检查哪些客户在购买金额超过 100 美元的列的列表中。

要使用像`=`或`<>`这样的数学运算符，子查询应该返回一个数字，而不是一列。在这个例子中并不是这样的，但是我们可以在必要的时候轻松地修改代码。

### 作为新表的子查询

我们将在本文中看到的使用 SQL 子查询的最后一种方法是用它来创建一个新的统一数据源，您可以从中提取数据。

当主查询变得太复杂时，我们使用这种方法，我们希望保持代码的可读性和组织性——当我们为了不同的目的重复使用这种新的数据源时，我们也不希望一遍又一遍地重写它。

通常看起来是这样的:

```py
 SELECT
    column_1,
    column_2
FROM
    (SELECT 
        ...
    FROM table_1
    INNER JOIN table_2)
WHERE column_1 > 100
```

例如，这将是我们的子查询:

```py
 SELECT c.customer_id,
       c.last_name,
       c.country,
       c.state,
       count(i.customer_id) as number_of_purchases,
       round(sum(i.total), 2) as total_purchased,
       (SELECT 
            count(il.track_id) n_tracks
        FROM invoice_line il   
        INNER JOIN invoice i on i.invoice_id = il.invoice_id
        WHERE i.customer_id = c.customer_id
        ) as count_tracks
FROM customer c 
INNER JOIN invoice i on i.customer_id = c.customer_id
GROUP BY 1, 2, 3, 4
ORDER BY 6 DESC
```

其结果是这个新表:

| 客户标识 | 姓氏 | 国家 | 状态 | 购买数量 | 总计 _ 已购买 | 计数 _ 曲目 |
| --- | --- | --- | --- | --- | --- | --- |
| five | 威彻尔 | 捷克共和国 | 没有人 | Eighteen | One hundred and forty-four point five four | One hundred and forty-six |
| six | 光着身子 | 捷克共和国 | 没有人 | Twelve | One hundred and twenty-eight point seven | One hundred and thirty |
| Forty-six | 奥赖利 | 爱尔兰 | 都柏林 | Thirteen | One hundred and fourteen point eight four | One hundred and sixteen |
| Fifty-eight | Pareek | 印度 | 没有人 | Thirteen | One hundred and eleven point eight seven | One hundred and thirteen |
| one | 贡萨尔维斯 | 巴西 | 特殊卡 | Thirteen | One hundred and eight point nine | One hundred and ten |
| Thirteen | 拉莫斯 | 巴西 | DF | Fifteen | One hundred and six point nine two | One hundred and eight |
| Thirty-four | 费尔南德斯 | 葡萄牙 | 没有人 | Thirteen | One hundred and two point nine six | One hundred and four |
| three | 特里布莱 | 加拿大 | 质量控制 | nine | Ninety-nine point nine nine | One hundred and one |
| forty-two | 吉拉德 | 法国 | 没有人 | Eleven | Ninety-nine point nine nine | One hundred and one |
| Seventeen | 史密斯（姓氏） | 美利坚合众国 | 西澳大利亚州 | Twelve | Ninety-eight point zero one | Ninety-nine |
| Fifty | 穆尼奥斯 | 西班牙 | 没有人 | Eleven | Ninety-eight point zero one | Ninety-nine |
| Fifty-three | 休斯 | 联合王国 | 没有人 | Eleven | Ninety-eight point zero one | Ninety-nine |
| Fifty-seven | 罗哈斯 | 辣椒 | 没有人 | Thirteen | Ninety-seven point zero two | Ninety-eight |
| Twenty | 面粉厂主 | 美利坚合众国 | 加拿大 | Twelve | Ninety-five point zero four | Ninety-six |
| Thirty-seven | 齐默尔曼 | 德国 | 没有人 | Ten | Ninety-four point zero five | Ninety-five |
| Twenty-two | 李科克 | 美利坚合众国 | 佛罗里达州 | Twelve | Ninety-two point zero seven | Ninety-three |
| Twenty-one | 追赶 | 美利坚合众国 | 女士 | Eleven | Ninety-one point zero eight | Ninety-two |
| Thirty | 弗朗西斯 | 加拿大 | 在…上 | Thirteen | Ninety-one point zero eight | Ninety-two |
| Twenty-six | 坎宁安 | 美利坚合众国 | 谢谢 | Twelve | Eighty-six point one three | Eighty-seven |
| Thirty-six | 裁缝店 | 德国 | 没有人 | Eleven | Eighty-five point one four | Eighty-six |
| Twenty-seven | 灰色的 | 美利坚合众国 | 阿塞拜疆（Azerbaijan 的缩写） | nine | Eighty-four point one five | eighty-five |
| Two | 克勒 | 德国 | 没有人 | Eleven | Eighty-two point one seven | Eighty-three |
| Twelve | 阿尔梅达 | 巴西 | 交叉路口(road junction) | Eleven | Eighty-two point one seven | Eighty-three |
| Thirty-five | 桑帕约 | 葡萄牙 | 没有人 | Sixteen | Eighty-two point one seven | Eighty-three |
| Fifty-five | 泰勒 | 澳大利亚 | 新南威尔士 | Ten | Eighty-one point one eight | Eighty-two |

在这个新表中，我们整合了数据库中每个客户的 ID、姓氏、国家、州、购买次数、花费的总金额以及购买的曲目数量。

现在，我们可以看到美国哪些用户购买了至少 50 首歌曲:

```py
 SELECT  
    new_table.*
FROM
    (SELECT c.customer_id,
        c.last_name,
        c.country,
        c.state,
        count(i.customer_id) as number_of_purchases,
        round(sum(i.total), 2) as total_purchased,
        (SELECT 
                count(il.track_id) n_tracks
            FROM invoice_line il   
            INNER JOIN invoice i on i.invoice_id = il.invoice_id
            WHERE i.customer_id = c.customer_id
            ) as count_tracks
    FROM customer c 
    INNER JOIN invoice i on i.customer_id = c.customer_id
    GROUP BY 1, 2, 3, 4
    ORDER BY 6 DESC) as new_table
WHERE 
    new_table.count_tracks >= 50
    AND new_table.country = 'USA' 
```

请注意，我们只需要选择列，并在 SQL 子查询中应用我们想要的过滤器。

这是输出:

| 客户标识 | 姓氏 | 国家 | 状态 | 购买数量 | 总计 _ 已购买 | 计数 _ 曲目 |
| --- | --- | --- | --- | --- | --- | --- |
| Seventeen | 史密斯（姓氏） | 美利坚合众国 | 西澳大利亚州 | Twelve | Ninety-eight point zero one | Ninety-nine |
| Twenty | 面粉厂主 | 美利坚合众国 | 加拿大 | Twelve | Ninety-five point zero four | Ninety-six |
| Twenty-two | 李科克 | 美利坚合众国 | 佛罗里达州 | Twelve | Ninety-two point zero seven | Ninety-three |
| Twenty-one | 追赶 | 美利坚合众国 | 女士 | Eleven | Ninety-one point zero eight | Ninety-two |
| Twenty-six | 坎宁安 | 美利坚合众国 | 谢谢 | Twelve | Eighty-six point one three | Eighty-seven |
| Twenty-seven | 灰色的 | 美利坚合众国 | 阿塞拜疆（Azerbaijan 的缩写） | nine | Eighty-four point one five | eighty-five |
| Eighteen | 布鲁克斯 | 美利坚合众国 | 纽约州 | eight | Seventy-nine point two | Eighty |
| Twenty-five | 斯蒂文斯 | 美利坚合众国 | WI | Ten | Seventy-six point two three | Seventy-seven |
| Sixteen | 哈里斯 | 美利坚合众国 | 加拿大 | eight | Seventy-four point two five | Seventy-five |
| Twenty-eight | 巴尼特 | 美利坚合众国 | 世界时 | Ten | Seventy-two point two seven | Seventy-three |
| Twenty-four | 罗尔斯顿 | 美利坚合众国 | 伊利诺伊 | eight | Seventy-one point two eight | seventy-two |
| Twenty-three | 戈登 | 美利坚合众国 | 马萨诸塞州 | Ten | Sixty-six point three three | Sixty-seven |
| Nineteen | 戈耶 | 美利坚合众国 | 加拿大 | nine | Fifty-four point four five | Fifty-five |

我们还可以看到每个州购买了至少 50 首歌曲的用户数量:

```py
 SELECT  
    state,
    count(*)
FROM
    (SELECT c.customer_id,
        c.last_name,
        c.country,
        c.state,
        count(i.customer_id) as number_of_purchases,
        round(sum(i.total), 2) as total_purchased,
        (SELECT 
                count(il.track_id) n_tracks
            FROM invoice_line il   
            INNER JOIN invoice i on i.invoice_id = il.invoice_id
            WHERE i.customer_id = c.customer_id
            ) as count_tracks
    FROM customer c 
    INNER JOIN invoice i on i.customer_id = c.customer_id
    GROUP BY 1, 2, 3, 4
    ORDER BY 6 DESC) as new_table
WHERE 
    new_table.count_tracks >= 50
    AND new_table.country = 'USA'  
GROUP BY new_table.state
ORDER BY 2 desc
```

现在，我们只需要添加聚合函数`count`和`GROUP BY`子句。我们继续使用子查询，就好像它是一个新的数据源。

输出:

| 状态 | 计数(*) |
| --- | --- |
| 加拿大 | three |
| 阿塞拜疆（Azerbaijan 的缩写） | one |
| 佛罗里达州 | one |
| 伊利诺伊 | one |
| 马萨诸塞州 | one |
| 女士 | one |
| 纽约州 | one |
| 谢谢 | one |
| 世界时 | one |
| 西澳大利亚州 | one |
| WI | one |

还可以使用这个新的 SQL 表进行一些计算，并按订单选择平均花费最高的前 10 名用户:

```py
 SELECT  
    customer_id,
    last_name,
    round(total_purchased / number_of_purchases, 2) as avg_purchase
FROM
    (SELECT c.customer_id,
        c.last_name,
        c.country,
        c.state,
        count(i.customer_id) as number_of_purchases,
        round(sum(i.total), 2) as total_purchased,
        (SELECT 
                count(il.track_id) n_tracks
            FROM invoice_line il   
            INNER JOIN invoice i on i.invoice_id = il.invoice_id
            WHERE i.customer_id = c.customer_id
            ) as count_tracks
    FROM customer c 
    INNER JOIN invoice i on i.customer_id = c.customer_id
    GROUP BY 1, 2, 3, 4
    ORDER BY 6 DESC) as new_table
ORDER BY 3 DESC
LIMIT 10
```

我们使用子查询中的两列来执行计算，并得到以下结果:

| 客户标识 | 姓氏 | 平均购买量 |
| --- | --- | --- |
| three | 特里布莱 | Eleven point one one |
| six | 光着身子 | Ten point seven two |
| Twenty-nine | 褐色的 | Ten point one five |
| Eighteen | 布鲁克斯 | Nine point nine |
| Thirty-seven | 齐默尔曼 | Nine point four |
| Twenty-seven | 灰色的 | Nine point three five |
| Sixteen | 哈里斯 | Nine point two eight |
| forty-two | 吉拉德 | Nine point zero nine |
| Fifty | 穆尼奥斯 | Eight point nine one |
| Fifty-three | 休斯 | Eight point nine one |

根据用户的需要，使用该子查询中的数据还有许多其他方法，甚至可以根据需要构建一个更大的子查询。

如果为此目的使用子查询过于频繁，那么根据数据库的体系结构，在数据库中创建一个视图，并使用这个新视图作为新的统一数据源可能会很有意思。请咨询您的数据工程团队！

### 结论

SQL 子查询是一个非常重要的工具，不仅对数据科学家来说如此，对任何经常使用 SQL 的人来说也是如此——花时间去理解它们绝对是值得的。

在本文中，我们介绍了以下内容:

*   如何以及何时使用子查询
*   使用子查询在主查询中创建新列
*   使用子查询作为过滤器
*   使用子查询作为新的数据源