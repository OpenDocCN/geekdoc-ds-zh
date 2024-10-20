# SQL 与 T-SQL:了解差异

> 原文：<https://www.dataquest.io/blog/sql-vs-t-sql/>

March 4, 2021![SQL or T-SQL - which one should you learn?](img/1d9a1223d7f1c61ae1e106fd39a59d83.png)

SQL 或 T-SQL——你需要学习哪一个？

SQL 和 T-SQL 在数据库和数据科学行业中都有大量的使用。但是它们到底是什么呢？这两种查询语言在名称和功能上都非常相似，因此很难理解它们之间的区别。

在这篇文章中，我们将:定义什么是标准 SQL 和 T-SQL，研究它们之间的区别，提供各自的例子，总结你应该学习哪一个以及为什么。

*   定义什么是标准 SQL 和 T-SQL
*   调查它们之间的差异
*   提供每个的示例
*   总结你应该学习哪些内容，为什么要学

## 什么是标准 SQL？

标准 SQL，通常简称为“SQL”，是一种称为查询语言的编程语言。查询语言用于与数据库通信。

SQL 用于添加、检索或更新存储在数据库中的数据。它被用于许多不同类型的数据库。也就是说，如果你学习了 SQL 的[基础，你将在数据职业生涯中处于有利地位。](https://www.dataquest.io/blog/sql-basics/)

数据库和存储在其中的数据是许多公司运营的核心部分。一个简单的例子是，零售商可能会将订单或客户信息存储在数据库中。SQL 是一种允许公司处理这些数据的编程语言。

## 什么是 T-SQL？

T-SQL 代表 Transact-SQL，有时也称为 TSQL，是主要在 Microsoft SQL Server 中使用的 SQL 语言的扩展。这意味着它提供了 SQL 的所有功能，但增加了一些额外的功能。

你可以把它想象成一种 SQL 方言——它非常类似于常规的 SQL，但是它有一些额外的和不同的地方，使它独一无二。

尽管标准 SQL 有清晰而严格的规范，但它确实允许数据库公司添加自己的扩展，以使它们有别于其他产品。对于 Microsoft SQL Server 数据库来说，T-SQL 就是这样一个例子— T-SQL 是该软件的核心，并且在其中运行大多数操作。

大多数主要的数据库供应商都为自己的产品提供了自己的 SQL 语言扩展，T-SQL 就是其中使用最广泛的一个例子(因为 Microsoft SQL server 很流行)。

简而言之:当您在 Microsoft SQL Server 中编写查询时，您可以有效地使用 T-SQL。无论应用程序的用户界面如何，所有与 SQL Server 通信的应用程序都是通过向服务器发送 T-SQL 语句来进行通信的。

但是，除了 SQL Server，其他数据库管理系统(DBMS)也支持 T-SQL。另一款微软产品 Microsoft Azure SQL Database 支持 T-SQL 的大部分功能。

T-SQL 旨在使支持它的数据库的工作更容易、更高效。

## SQL 和 T-SQL 有什么区别？

现在我们已经介绍了这两者的基础知识，让我们来看看它们的主要区别:

### 差异#1

明显的区别在于它们的设计目的:SQL 是一种用于操作存储在数据库中的数据的查询语言。T-SQL 也是一种查询语言，但它是 SQL 的扩展，主要用于 Microsoft SQL Server 数据库和软件。

### 差异#2

SQL 是开源的。T-SQL 由微软开发并拥有。

### 差异#3

SQL 语句一次执行一条，也称为“非过程化”T-SQL 以一种“过程化”的方式执行语句，这意味着代码将作为一个块，在逻辑上以结构化的顺序进行处理。

每种方法都有优点和缺点，但从学习者的角度来看，这种差异并不太重要。您将能够以任何一种语言获取和处理您想要的数据，只是根据您使用的语言和查询的具体情况，您处理数据的方式会有所不同。

### 差异#4

除了这些更普遍的差异之外，SQL 和 T-SQL 还有一些稍微不同的命令关键字。T-SQL 还具有不属于常规 SQL 的功能。

这方面的一个例子是我们如何选择最上面的 *X* 行。在标准 SQL 中，我们会使用 LIMIT 关键字。在 T-SQL 中，我们使用 TOP 关键字。

这两个命令做同样的事情，正如我们在下面的例子中看到的。这两个查询都将返回 users 表中按 age 列排序的前十行。

#### SQL 示例

```py
SELECT *
FROM users
ORDER BY age
LIMIT 10;
```

#### T-SQL 示例

```py
SELECT TOP 10 (*)
FROM users
ORDER BY age;
```

### 差异#5

最后，如前所述，T-SQL 提供了常规 SQL 中没有的功能。ISNULL 函数就是一个例子。这将替换来自特定列的空值。对于年龄列中值为 NULL 的任何行，下面的将返回年龄“0”。

```py
SELECT ISNULL(0, age)
FROM users;
```

(当然，在标准 SQL 中也有这样做的[种方式](https://stackoverflow.com/questions/9877533/replace-nulls-values-in-sql-using-select-statement)，但是命令略有不同。)

这些只是一些代码差异，让您对两者的比较有所了解，当然，还有更多。通过我们丰富的指南，您可以了解更多关于 SQL 命令的信息。当然，微软有与 T-SQL 一起工作的[文档。](https://docs.microsoft.com/en-us/sql/t-sql/language-reference?view=sql-server-ver15)

## 学哪个比较好？

如果你想以任何方式使用数据库，或者如果你正在寻找一份数据工作，学习 SQL 是必要的。

因为 T-SQL 是 SQL 的扩展，所以在开始之前，您需要学习 SQL 的基础知识。如果您首先学习 T-SQL，无论如何，您最终都会学到标准 SQL 的知识。

对于大多数事情来说，你选择学什么应该取决于你想要达到的目标。如果您打算使用 Microsoft SQL server，那么学习更多关于 T-SQL 的知识是值得的。如果你是一个初学使用数据库的人，那就从学习 SQL 开始吧。

如果你想了解关于这个话题的更多信息，请查看 Dataquest 的交互式[SQL 和数据库介绍](https://www.dataquest.io/course/funds-sql-i/)课程，以及我们的 [SQL 基础知识](https://www.dataquest.io/path/sql-skills/)，帮助你在大约 2 个月内掌握这些技能。

### 用正确的方法学习 SQL！

*   编写真正的查询
*   使用真实数据
*   就在你的浏览器里！

当你可以 ***边做边学*** 的时候，为什么要被动的看视频讲座？

[Sign up & start learning!](https://app.dataquest.io/signup)