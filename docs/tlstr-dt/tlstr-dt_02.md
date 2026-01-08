# 错误和更新

> 原文：[`tellingstorieswithdata.com/00-errata.html`](https://tellingstorieswithdata.com/00-errata.html)

*Chapman and Hall/CRC 于 2023 年 7 月出版了本书。您可以在[此处](https://www.routledge.com/Telling-Stories-with-Data-With-Applications-in-R/Alexander/p/book/9781032134772)购买。

本在线版本对印刷版有所更新。与印刷版相匹配的在线版本可在[此处](https://rohanalexander.github.io/telling_stories-published/)找到。*  **最后更新：2024 年 11 月 21 日.*

本书由 Piotr Fryzlewicz 在 *The American Statistician* (Fryzlewicz 2024) 和 Nick Cox 在 [Amazon](https://www.amazon.com/gp/customer-reviews/R3S602G9RUDOF/ref=cm_cr_dp_d_rvw_ttl?ie=UTF8&ASIN=1032134771) 上进行评论。我非常感激他们抽出大量时间提供评论，以及他们的纠正和建议。

自本书于 2023 年 7 月出版以来，世界发生了许多变化。生成式 AI 的兴起改变了人们的编码方式，由于 Quarto，Python 与 R 的集成变得更加容易，包持续更新（不用说，还有一批新的学生开始阅读本书）。拥有在线版本的一个优点是我可以做出改进。

我感谢以下人士的纠正和建议：Andrew Black, Clay Ford, Crystal Lewis, David Jankoski, Donna Mulkern, Emi Tanaka, Emily Su, Inessa De Angelis, James Wade, Julia Kim, Krishiv Jain, Seamus Ross, Tino Kanngiesser 和 Zak Varty。

## 错误

印刷版中存在以下错误，但在线版本已更新。如果您注意到以下未提及的错误，请提交[问题](https://github.com/RohanAlexander/telling_stories/issues)或发送电子邮件：rohan.alexander@utoronto.ca。

+   p. xxi: 在致谢中添加 Alex Hayes。

+   p. 20: 在“使用 `tidyverse` 和 `janitor` 包”中添加“packages”。

+   p. 34: `"daily-shelter-overnight-service-occupancy-capacity-2021"` 应为 `"daily-shelter-overnight-service-occupancy-capacity-2021.csv"`（注意添加了“.csv”）。

+   p. 34: 将第一个代码块替换为第二个：

```py
toronto_shelters_clean <-
 clean_names(toronto_shelters) |>
 select(occupancy_date, id, occupied_beds)

head(toronto_shelters_clean)
```

*```py
toronto_shelters_clean <-
 clean_names(toronto_shelters) |>
 mutate(occupancy_date = ymd(occupancy_date)) |> 
 select(occupancy_date, occupied_beds)

head(toronto_shelters_clean)
```

**   p. 38: “At this point we can make a nice graph of the number of ridings won by each party in the 2019 Canadian Federal Election.” 应指代 2021 年的选举。

+   p. 41: 删除多余的 “:::”。

+   p. 66: “New Project$dots” 应为 “New Project…”。

+   p. 138: `scale_color_brewer(palette = "Set1")` 是不必要的，应删除。

+   p. 138: 图表说明应指代通货膨胀而非失业。

+   p. 154: 代码块后缺少“work”，在“if”之前。

+   p. 188: “Leonhard Euler” 应为 “Carl Friedrich Gauss”.

+   p. 279: “detonated” 应为 “denoted”。

+   p. 342: Q5 选项 b 在选项 c 中重复。

+   p. 347: 《R for Data Science》的“探索性数据分析”章节是 11 章，而不是 12 章。

+   p. 353: 修复“the the”。

+   p. 355: “...结果估计为 5,814，两者都太低。” 应该改为 “...结果估计为 11,197，前者太低，后者太高。”

+   p. 371: 引用图 11.11a 的句子令人困惑，需要更清晰地引用该图。

+   p. 587: 链接应为：https://fivethirtyeight.com/features/police-misconduct-costs-cities-millions-every-year-but-thats-where-the-accountability-ends/

Fryzlewicz, Piotr. 2024. “用数据讲故事：R 语言应用。” *《美国统计学家》*，四月，1–5 页。[`doi.org/10.1080/00031305.2024.2339562`](https://doi.org/10.1080/00031305.2024.2339562).***
