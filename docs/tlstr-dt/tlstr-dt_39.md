# 在线附录 E — R Markdown

> 原文：[`tellingstorieswithdata.com/24-rmarkdown.html`](https://tellingstorieswithdata.com/24-rmarkdown.html)

1.  附录

1.  E  R Markdown

Quarto 是 R Markdown 的继任者，然而它相对较新，许多人仍然使用 R Markdown。就大部分内容而言，第三章中涵盖的方面适用于 Quarto 和 R Markdown。然而，在本附录中，我们为 R Markdown 提供了 Quarto 中提供的指导的等效内容，其中有一些方面不同。

## E.1 R 代码块

我们可以在 R Markdown 文档中的代码块中包含 R 和许多其他语言的代码。然后当我们编织文档时，代码将运行并被包含在文档中。

要创建一个 R 代码块，我们首先使用三个反引号，然后在花括号内告诉 R Markdown 这是一个 R 代码块。这个代码块内的任何内容都将被视为 R 代码并按此运行。例如，我们可以加载`tidyverse`和`AER`，并绘制过去两周内调查受访者看医生的次数的图表。

```py
```{r}

library(tidyverse)

library(AER)

data("DoctorVisits", package = "AER")

DoctorVisits |>

ggplot(aes(x = illness)) +

geom_histogram(stat = "count")

```py
```

该代码的输出是 Figure 3.1

![](img/fea3d502a7503de9e0f6f6d602220d17.png)

过去两周内疾病数量的统计，基于 1977-1978 年澳大利亚健康调查

注意，与 Quarto 不同，所有选项都在顶行的大括号之间。无法使用 Quarto 的注释符号。

## E.2 交叉引用

在交叉引用图表、表格和方程时可能很有用。这使得在文本中引用它们变得更加容易。为了对图表进行交叉引用，我们引用创建或包含该图表的 R 代码块名称。例如，`(Figure \@ref(fig:uniquename))`将生成：(Figure 3.2)，因为 R 代码块名称是`uniquename`。我们还需要在代码块名称前加上‘fig’以让 R Markdown 知道这是一个图表。然后我们在 R 代码块中包含一个‘fig.cap’来指定标题。

```py
```{r uniquename, fig.cap = "Number of illnesses in the past two weeks, based on the 1977--1978 Australian Health Survey", echo = TRUE}

```py

 *```

library(tidyverse)

library(AER)

data("DoctorVisits", package = "AER")

DoctorVisits |>

ggplot(aes(x = illness)) +

geom_histogram(stat = "count")

```py

 *![](img/534ae4780883ed6a61c57f477b332c80.png)

Number of illnesses in the past two weeks, based on the 1977–1978 Australian Health Survey*  *We can take a similar, but slightly different, approach to cross-reference tables. For instance, `(Table \@ref(tab:docvisittablermarkdown))` will produce: (Table E.1). In this case we specify ‘tab’ before the unique reference to the table, so that R Markdown knows that it is a table. For tables we need to include the caption in the main content, as a ‘caption’, rather than in a ‘fig.cap’ chunk option as is the case for figures.

Table E.1: Number of visits to the doctor in the past two weeks, based on the 1977–1978 Australian Health Survey

| visits | n |
| --- | --- |
| 0 | 4141 |
| 1 | 782 |
| 2 | 174 |
| 3 | 30 |
| 4 | 24 |
| 5 | 9 |
| 6 | 12 |
| 7 | 12 |
| 8 | 5 |
| 9 | 1 |

Finally, we can also cross-reference equations. To that we need to add a tag `(\#eq:macroidentity)` which we then reference. For instance, use `Equation \@ref(eq:macroidentity).` to produce (Equation E.1)

```

\begin{equation}

Y = C + I + G + (X - M) (\#eq:macroidentity)

\end{equation}

```

$$ Y = C + I + G + (X - M) \tag{E.1}$$

当使用交叉引用时，确保 R 代码块具有简单的标签。通常，尽量保持名称简单但唯一，如果可能，避免使用标点符号并坚持使用字母。不要在标签中使用下划线，因为这会导致错误。**
