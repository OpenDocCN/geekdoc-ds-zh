# SQL 备忘单—用于数据分析的 SQL 参考指南

> 原文：<https://www.dataquest.io/blog/sql-cheat-sheet/>

January 20, 2021

无论您是通过我们的交互式 SQL 课程学习 SQL，还是通过其他方式学习 SQL，拥有一份 SQL 备忘单都会非常有帮助。

将这篇文章加入书签，或者下载并打印 PDF 文件，并在下次编写 SQL 查询时方便地快速参考！



[![](img/3ca399c8b6bd56aa05009fec309047f8.png "sql-cheat-sheet")](https://www.dataquest.io/wp-content/uploads/2021/02/sql-cheat-sheet.jpg)

我们的 SQL 备忘单比这份手写的更深入一点！

在准备阅读备忘单之前，需要复习一下 SQL 吗？查看我们的交互式[在线 SQL 基础课程](https://www.dataquest.io/path/sql-skills/)，阅读[为什么你应该学习 SQL](https://www.dataquest.io/blog/why-sql-is-the-most-important-language-to-learn/) ，或者做一些关于 [SQL 认证和你是否需要一个](https://www.dataquest.io/blog/sql-certification/)的研究。

## SQL 基础知识

SQL 代表结构化语言 **Q** 查询 **L** 。它是一个用于查询——请求、过滤和输出——来自关系数据库的数据的系统。

SQL 开发于 20 世纪 70 年代，最初被称为 SEQUEL。因此，今天这个词有时读作“Sequel”，有时读作“S.Q.L”。这两种发音都是可以接受的

尽管 SQL 有许多“风格”,但某种形式的 SQL 可以用于查询大多数关系数据库系统的数据，包括 MySQL、SQLite、Oracle、Microsoft SQL Server、PostgreSQL、IBM DB2、Microsoft Azure SQL 数据库、Apache Hive 等。数据库。

## SQL 备忘单:基础

#### 使用 SQL 执行计算

执行单一计算:
`SELECT 1320+17;`

执行多重计算:
`SELECT 1320+17, 1340-3, 7*191, 8022/6;`

使用多个数字进行计算:
`SELECT 1*2*3, 1+2+3;`

重命名结果:
`SELECT 2*3 AS mult, 1+2+3 AS nice_sum;`

#### 选择表格、列和行:

记住:在 SQL 中，子句的顺序很重要。SQL 使用以下优先顺序:`FROM`、`SELECT`、`LIMIT`。

显示整个表格:

```py
SELECT *
  FROM table_name;
```

从表中选择特定列:

```py
SELECT column_name_1, column_name_2
  FROM table_name;
```

显示表格的前 10 行:

```py
SELECT *
  FROM table_name
  LIMIT 10;
```

#### 向 SQL 查询添加注释

添加单行注释:

```py
-- First comment
SELECT column_1, column_2, column_3 -- Second comment
  FROM table_name; -- Third comment
```

添加块注释:

```py
/*
This comment
spans over
multiple lines
 */
SELECT column_1, column_2, column_3
  FROM table_name;
```

## SQL Intermediate:连接和复杂查询

这些示例中有许多使用了真实 SQL 数据库中的表名和列名，学员在我们的 interactive SQL 课程中会用到这些数据库。要了解更多信息，[注册一个免费帐户](https://app.dataquest.io/signup)并试用一个！

#### 在 SQL 中联接数据:

使用内部联接联接表:

```py
SELECT column_name_1, column_name_2 FROM table_name_1
INNER JOIN table_name_2 ON table_name_1.column_name_1 = table_name_2.column_name_1;
```

使用左连接来连接表:

```py
SELECT * FROM facts
LEFT JOIN cities ON cities.facts_id = facts.id;
```

使用右连接来连接表:

```py
SELECT f.name country, c.name city
FROM cities c
RIGHT JOIN facts f ON f.id = c.facts;
```

使用完全外部联接来联接表:

```py
SELECT f.name country, c.name city
FROM cities c
FULL OUTER JOIN facts f ON f.id = c.facts_id;
```

在不指定列名的情况下对列排序:

```py
SELECT name, migration_rate FROM FACTS
ORDER BY 2 desc; -- 2 refers to migration_rate column
```

在子查询中使用联接，但有一个限制:

```py
SELECT c.name capital_city, f.name country
FROM facts f
INNER JOIN (
        SELECT * FROM cities
				WHERE capital = 1
				) c ON c.facts_id = f.id
LIMIT 10;
```

连接两个以上表中的数据:

```py
SELECT [column_names] FROM [table_name_one]
   [join_type] JOIN [table_name_two] ON [join_constraint]
	 [join_type] JOIN [table_name_three] ON [join_constraint]
	 ...
	 ...
	 ...
	 [join_type] JOIN [table_name_three] ON [join_constraint]
```

#### 其他常见的 SQL 操作:

将列合并成一列:

```py
SELECT
		album_id,
		artist_id,
		"album id is " || album_id col_1,
		"artist id is " || artist_id col2,
		album_id || artist_id col3
FROM album LIMIT 3;
```

字符串的匹配部分:

```py
SELECT
	first_name,
	last_name,
	phone
FROM customer
WHERE first_name LIKE "%Jen%";
```

在带有 CASE 的 SQL 中使用 if/then 逻辑:

```py
CASE
	WHEN [comparison_1] THEN [value_1]
	WHEN [comparison_2] THEN [value_2]
	ELSE [value_3]
	END
AS [new_column_name]
```

使用 WITH 子句:

```py
WITH track_info AS
(
	SELECT
		t.name,
		ar.name artist,
		al.title album_name,
	FROM track t
	INNER JOIN album al ON al.album_id = t.album_id
	INNER JOIN artist ar ON ar.artist_id = al.artist_id
)
SELECT * FROM track_info
WHERE album_name = "Jagged Little Pill";
```

创建视图:

```py
CREATE VIEW chinook.customer_2 AS
SELECT * FROM chinook.customer;
```

删除视图:

```py
DROP VIEW chinook.customer_2;
```

选择出现在一个或多个 SELECT 语句中的行:

```py
[select_statement_one]
UNION
[select_statement_two];
```

选择在两个 SELECT 语句中都出现的行:

```py
SELECT * from customer_usa
INTERSECT
SELECT * from customer_gt_90_dollars;
```

选择出现在第一个 SELECT 语句中但不出现在第二个 SELECT 语句中的行:

```py
SELECT * from customer_usa
EXCEPT
SELECT * from customer_gt_90_dollars;
```

用语句链接:

```py
WITH
usa AS
	(
	SELECT * FROM customer
	WHERE country = "USA"
	),
last_name_g AS
	(
	SELECT * FROM usa
	WHERE last_name LIKE "G%"
	),
state_ca AS
	(
	SELECT * FROM last_name_g
	WHERE state = "CA"
	)
SELECT
	first_name,
	last_name,
	country,
	state
FROM state_ca
```

## 重要概念和资源:

#### 保留字

保留字是在编程语言中不能用作标识符的字(如变量名或函数名)，因为它们在语言本身中有特定的含义。下面是 SQL 中的保留字列表。

![YouTube video player for JFlukJudHrk](img/dfad4b71ed1a0869abbcbbe9fac03787.png)

*[https://www.youtube.com/embed/JFlukJudHrk?feature=oembed](https://www.youtube.com/embed/JFlukJudHrk?feature=oembed)*

 *## 下载 SQL 备忘单 PDF

点击下面的按钮下载备忘单(PDF，3 MB，彩色)。

[下载 SQL 备忘单](https://www.dataquest.io/wp-content/uploads/2021/01/dataquest-sql-cheat-sheet.pdf)

寻找不仅仅是一个快速参考？Dataquest 的交互式 [SQL 课程](https://www.dataquest.io/path/sql-skills/)将帮助您在学习构建您将需要为真实世界的数据工作编写的复杂查询时，亲自动手操作 SQL。

点击下面的按钮**注册一个免费帐户**并立即开始学习！

### 用正确的方法学习 SQL！

*   编写真正的查询
*   使用真实数据
*   就在你的浏览器里！

当你可以 ***边做边学*** 的时候，为什么要被动的看视频讲座？

[Sign up & start learning!](https://app.dataquest.io/signup)*