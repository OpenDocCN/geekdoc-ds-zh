# 乘法作为卷积

> 原文：[`www.algorithm-archive.org/contents/convolutions/multiplication/multiplication.html`](https://www.algorithm-archive.org/contents/convolutions/multiplication/multiplication.html)

作为简短的旁白，我们将触及一个相当有趣的话题：整数乘法和卷积之间的关系。作为一个例子，让我们考虑以下乘法：

在这种情况下，我们可能会将数字排列成这样：

在这里，每一列代表 10 的另一个幂，例如在数字 123 中，有 1 个 100，2 个 10，和 3 个 1。因此，我们可以使用类似的符号来进行卷积，通过反转第二组数字并将其向右移动，在每一步进行逐元素乘法：

对于这些操作，任何空白空间应被视为一个 . 最后，我们将得到一组新的数字：

现在剩下的只是执行*进位*操作，将 10 位上的任何数字移动到其左边的相邻数字。例如，数字 58 或 58。对于这些数字，

这将给我们，整数乘法的正确答案。我并不是建议我们教小学生学习卷积，但我确实觉得这是一个大多数人不知道的有趣事实：整数乘法可以用卷积来完成。

当我们讨论 Schonhage-Strassen 算法时，这将进一步详细讨论，该算法使用这一事实来执行极大整数的乘法。

## 许可

##### 文本

本章的文本由[James Schloss](https://github.com/leios)编写，并许可在[Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/legalcode)下。

[](https://creativecommons.org/licenses/by-sa/4.0/)

![](https://creativecommons.org/licenses/by-sa/4.0/)

##### 拉取请求

在初始许可([#560](https://github.com/algorithm-archivists/algorithm-archive/pull/560))之后，以下拉取请求已修改了本章的文本或图形：

+   无
