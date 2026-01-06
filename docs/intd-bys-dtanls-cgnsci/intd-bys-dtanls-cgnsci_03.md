# 认知科学贝叶斯数据分析简介

> [`bruno.nicenboim.me/bayescogsci/index.html`](https://bruno.nicenboim.me/bayescogsci/index.html)

*布鲁诺·尼岑博伊姆，丹尼尔·J·沙德，以及夏拉万·瓦希什*

*2025-02-12*

# 前言

本书旨在作为使用概率编程语言 Stan（Carpenter 等人 2017）及其前端`brms`（Bürkner 2024）进行贝叶斯数据分析和认知建模的相对温和的入门介绍。我们的目标受众是进行计划行为实验的认知科学家（例如，语言学家、心理学家和计算机科学家），他们希望从基础和原则性的方式学习贝叶斯数据分析方法。我们的目标是使贝叶斯统计成为实验语言学、心理语言学、心理学和相关学科数据分析工具箱的标准部分。

对于贝叶斯数据分析，已经存在许多优秀的入门教科书。为什么还要写另一本书呢？我们的文本与其他尝试在两个方面有所不同。首先，我们的主要重点是展示如何分析涉及重复测量的计划实验中的数据；这类实验数据涉及独特的复杂性。我们提供了许多涉及时间测量的数据集示例（例如，自我调节阅读，阅读时的眼动追踪，声音起始时间），事件相关电位，瞳孔大小，准确性（例如，回忆任务，是非问题），分类答案（例如，图片命名），选择反应时间（例如，斯特鲁普任务，运动检测任务）等。其次，从一开始，我们就强调一种特定的工作流程，其核心是模拟数据；我们旨在教授一种哲学，即在数据收集**之前**就认真思考假设的潜在生成过程。我们希望通过这本书教授的数据分析方法涉及先验预测和后验预测检查、敏感性分析和使用模拟数据进行模型验证的循环。我们试图培养一种感觉，即如何从理论上有趣的参数的后验分布中得出推论，而无需诉诸“显著”或“不显著”这样的二元决策。我们希望这将为以更细致的方式报告和解释数据分析结果设定新的标准，并导致发表文献中更多谨慎的断言。

请在[`github.com/bnicenboim/bayescogsci/issues`](https://github.com/bnicenboim/bayescogsci/issues)报告错别字、错误或改进建议。

## 为什么阅读这本书，它的目标受众是谁？

心理学、心理语言学和其他领域普遍认为，统计分析是次要的，应该是快速且容易的。例如，一位资深的数学心理学家曾告诉本书的最后一作者：“如果你需要运行比配对 t 检验更复杂的任何东西，你提出的问题就是错误的。”在这里，我们持有不同的观点：科学和统计建模是一个统一的事物。统计模型应该代表一些合理的近似，这些近似是假设在起作用的潜在认知过程。

本书的目标读者是那些希望将统计学视为他们科学工作中平等伙伴的学生和研究人员。我们期望读者愿意花时间理解和运行计算分析。

任何对贝叶斯数据分析的严谨介绍都至少需要被动了解概率论、微积分和线性代数。然而，我们并不要求读者在开始阅读本书时就有这些背景知识。相反，相关思想是非正式地、及时地引入的，一旦需要就立即引入。读者永远不需要具备解决概率问题、求解积分或计算导数，或手动执行矩阵运算（如矩阵求逆）的主动能力。有几个地方讨论变得技术性，需要一些微积分或相关主题的知识。然而，对所需数学不熟悉的读者可以简单地跳过这些部分，因为这些部分实际上并不是跟随本书主线所必需的。

我们期望的是对算术、基本集合理论和初等概率理论（例如，求和和乘积规则，条件概率）的熟悉，以及简单的矩阵运算，如加法和乘法，以及简单的代数运算。在开始本书之前，快速浏览 Gill 的第一章是非常推荐的。我们还假设，当需要时，读者愿意查找他们可能已经忘记的概念（例如，对数）。我们还提供了一些数学基本概念的自学课程（针对非数学家），读者可以自学：请参阅[`vasishth.github.io/FoM/`](https://vasishth.github.io/FoM/)。

我们还期望读者已经了解并且/或者愿意学习足够的编程语言 R（R Core Team 2023），以便能够重现所提供的示例并完成练习。如果读者对 R 完全陌生，在开始本书之前，他们应该首先查阅像[R for data science](https://r4ds.had.co.nz/)和[Efficient R programming](https://csgillespie.github.io/efficientR/)这样的书籍。熟悉 Python 的读者可能会发现 Jozsef Arato 对前五章的 Python 移植版很有用([`github.com/jozsarato/bayescogdat`](https://github.com/jozsarato/bayescogdat))。

我们还假设读者已经接触过简单的线性建模和线性混合模型（Bates, Mächler, et al. 2015; Baayen, Davidson, and Bates 2008)。在实践中这意味着读者应该使用 R 中的`lm()`和`lmer()`函数。对基本统计概念，如两个变量之间的相关性，也有一定的了解。

本书不适合数据分析的初学者。数据分析的新手应该从像 Kerns (2014)这样的免费教科书开始，然后阅读我们关于频率派数据分析的介绍，该介绍也免费在线提供（Vasishth, Schad, et al. 2021)。后者将为读者准备这里所展示的材料。

## 为本书培养正确的思维方式

读者应该带到这本书中的一个非常重要的特点是“我能做到”的精神。会有很多地方会变得困难，读者将不得不放慢速度，与材料互动，或者刷新他们对算术或初中代数的理解。这种“我能做到”的精神的基本原则在 Burger 和 Starbird (2012)的书中得到了很好的总结；也参见 Levy (2021)。虽然我们无法用几句话总结这些书籍的所有见解，但受 Burger 和 Starbird (2012)的书的启发，以下是对读者需要培养的心态类型的一个简短列举：

+   在基本、看似简单的材料上花时间；确保你对其有深刻的理解。寻找你理解中的空白。阅读不同书籍或文章中对同一材料的不同呈现可以带来新的见解。

+   让错误和错误成为你的老师。我们本能地回避我们的错误，但错误最终是我们的朋友；它们有潜力教会我们比正确答案更多的东西。从这个意义上说，一个正确的解决方案可能不如一个错误的解决方案有趣。

+   当你被某些练习或问题吓倒时，立即放弃并承认失败。这会放松你的心情；你已经放弃了，没有更多的事情要做。然后，过一段时间后，尝试解决一个更简单的问题版本。有时，将问题分解成更小的部分，每个部分可能更容易解决，这很有用。

+   提出你自己的问题。不要等待被问问题；发展你自己的问题，然后尝试解决它们。

+   不要期望在第一次阅读时就能理解所有内容。只需在心里记下你理解上的差距，稍后再回来解决这些问题。

+   定期退后一步，尝试勾勒出你正在学习的更广泛的图景。在不查阅任何资料的情况下写下你所知道的内容，这是一种有助于实现这一目标的有用方法。不要等待老师给你总结你应该学到的要点；自己发展这样的总结。

+   培养寻找信息的能力。当你面对不知道的事情或一些晦涩的错误信息时，使用谷歌寻找一些答案。

+   不要犹豫重新阅读一个章节；通常，只有当一个人再次回顾材料后，才能真正理解一个主题。

作为讲师，我们多年来注意到，具有这种心态的学生通常表现很好。一些学生已经拥有这种精神，但其他人需要明确地培养它。我们坚信每个人都可以培养这种心态，但可能需要努力去获得它。无论如何，这种态度对于这类书籍来说是必要的。

## 如何阅读本书

本书中的章节旨在按顺序阅读，但在第一次阅读本书时，读者可以自由地跳过在线提供的可选深入材料。这些资源提供了更正式的发展（有助于过渡到更高级的教科书，如 Gelman 等人 2014），或处理章节中呈现的主题的边缘方面。

根据读者的目标，以下是本书的一些建议阅读路径：

+   对于面向完全初学者的短期课程，请阅读第一章到第五章。我们通常在每年举办的为期五天的暑期学校课程中涵盖这五章。这些章节中的大部分内容也包含在可在线免费提供的为期四周的课程中：[`open.hpi.de/courses/bayesian-statistics2023`](https://open.hpi.de/courses/bayesian-statistics2023)。

+   对于专注于使用 R 包`brms`的回归模型的课程，请阅读第一章到第七章，以及可选的 13 章。

+   对于专注于涉及 Stan 的复杂模型的进阶课程，请阅读第八章到第十八章。

## 本书中使用的一些约定

我们采用以下约定：

+   所有分布名称均为小写，除非它们也是专有名词（例如，Poisson、Bernoulli）。

+   单变量正态分布由均值和标准差（而非方差）参数化。

+   图表的代码仅在认为具有教学意义的情况下提供。在其他情况下，代码保持隐藏，但可以在书的网络版本中找到（[`bruno.nicenboim.me/bayescogsci/`](https://bruno.nicenboim.me/bayescogsci/)）。请注意，书中所有 R 代码都可以从每个章节的 Rmd 源文件中提取，这些源文件与书一同发布（[`github.com/bnicenboim/bayescogsci`](https://github.com/bnicenboim/bayescogsci)）。

## 在线材料

整本书，包括所有数据和源代码，都可以在 [`bruno.nicenboim.me/bayescogsci/`](https://bruno.nicenboim.me/bayescogsci/) 上免费在线获取。每个章节的附加可选材料和练习也在线提供。如有需要，可以提供练习的答案。

## 需要的软件

在开始之前，请安装

+   [R](https://cran.r-project.org/) 和 [RStudio](https://www.rstudio.com/)，或您偏好的任何其他集成开发环境，例如 [Visual Studio Code](https://code.visualstudio.com/) 和 [Emacs Speaks Statistics](https://ess.r-project.org/)。

+   R 包 `rstan`。在撰写本书时，CRAN 版本的 `rstan` 落后于 Stan 的最新发展，因此建议从 `https://mc-stan.org/r-packages/` 安装 `rstan`，如[`github.com/stan-dev/rstan/wiki/RStan-Getting-Started`](https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started)所示。

+   书中许多章节使用了 R 包 `dplyr`、`purrr`、`tidyr`、`extraDistr`、`brms`、`hypr` 和 `lme4`，并且可以按常规方式安装：`install.packages(c("dplyr","purrr","tidyr", "extraDistr", "brms","hypr","lme4"))`。我们使用 `ggplot2` 进行绘图；如果您不熟悉 `ggplot2`，请参阅相关文档（例如，Wickham, Chang, et al. 2024）。

+   以下 R 包是可选的：`tictoc`、`rootSolve`、`SHELF`、`cmdstanr` 和 `SBC`。

+   一些包及其依赖项，例如 `intoo`、`barsurf`、`bivariate` 和 `SIN` 可能需要从存档或 github 版本手动安装。对于此类包，请访问 CRAN 存档：[`cran.r-project.org/src/contrib/Archive/`](https://cran.r-project.org/src/contrib/Archive/)；下载相关包的 tar.gz 文件；然后使用命令行中的命令 `R CMD INSTALL package.tar.gz` 进行安装。

+   本书使用的数据和 Stan 模型可以使用 `remotes::install_github("bnicenboim/bcogsci")` 命令安装。此命令使用 `remotes` 包中的 `install_github()` 函数。（因此，此包也应存在于系统中。）

在每个 R 会话中，加载以下包，并设置以下 Stan 选项。

```r
library(MASS)
## be careful to load dplyr after MASS
library(dplyr)
library(tidyr)
library(purrr)
library(extraDistr)
library(ggplot2)
library(loo)
library(bridgesampling)
library(brms)
library(bayesplot)
library(tictoc)
library(hypr)
library(bcogsci)
library(lme4)
library(rstan)
# This package is optional, see https://mc-stan.org/cmdstanr/:
library(cmdstanr)
# This package is optional, see https://hyunjimoon.github.io/SBC/:
library(SBC)
library(SHELF)
library(rootSolve)

## Save compiled models:
rstan_options(auto_write = FALSE)
## Parallelize the chains using all the cores:
options(mc.cores = parallel::detectCores())
# To solve some conflicts between packages:
select <-  dplyr::select
extract <-  rstan::extract
```

## 致谢

我们感谢波茨坦大学的多代学生，ESSLLI 的各种夏季学校，LOT 冬季学校，Open HPI 关于贝叶斯统计学的 MOOC 课程（[`open.hpi.de/courses/bayesian-statistics2023`](https://open.hpi.de/courses/bayesian-statistics2023)），我们在各种机构教授的其他短期课程，以及每年在德国波茨坦举办的[语言学和心理学统计方法（SMLP）](https://vasishth.github.io/smlp/)夏季学校。这些课程中的参与者大大帮助了我们改进这里展示的材料。特别感谢 Anna Laurinavichyute，Paula Lissón 和 Himanshu Yadav 共同教授 SMLP 中的贝叶斯课程。我们还感谢 Vasishth 实验室的成员，特别是 Dorothea Pregla，对本书早期草稿的评论。我们还想感谢 Douglas Bates，Ben Bolker，Christian Robert（又称 Xi'an），Robin Ryder，Nicolas Chopin，Michael Betancourt，Andrew Gelman，Stan 的开发者（特别是 Bob Carpenter 和 Paul-Christian Bürkner），Philip D. Loewen，和 Leendert Van Maanen 提供的评论和建议；感谢 Pavel Logačev 的反馈，以及 Athanassios Protopapas，Patricia Mirabile，Masataka Ogawa，Alex Swiderski，Andrew Ellis，Jakub Szewczyk，Chi Hou Pau，Alec Shaw，Patrick Wen，Riccardo Fusaroli，Abdulrahman Dallak，Elizabeth Pankratz，João Veríssimo，Jean-Pierre Haeberly，Chris Hammill，Florian Wickelmaier，Ole Seeth，Jules Bouton，Siqi Zheng，Michael Gaunt，Benjamin Senst，Chris Moreh，Richard Hatcher，Noelia Stetie，Robert Lew，Leonardo Cerliani，Stefan Riedel，Raluca Rilla，Arne Schernich，Sven Koch，Joy Sarow，Iñigo Urrestarazu-Porta，Jan Winkowski，Adrian Staub，Brian Dillon，Job Schepens，Katja Politt，Cui Ding，Marc Tortorello，Michael Vrazitulis，Marisol Murujosa，Carla Bombi Ferrer，和 Ander Egurtzegi 捕捉书中的错别字、不清晰的段落和错误。特别感谢 Daniel Heck，Alexandre Cremers，Henrik Singmann 和 Martin Modrák 阅读（本书的部分内容）并捕捉到许多错误和错别字。对于任何我们忘记提及的人，我们表示歉意。还要感谢 Jeremy Oakley 以及英国谢菲尔德大学数学与统计学院的其他统计学家，他们提供了有益的讨论，以及受到谢菲尔德大学在线授课的硕士课程启发的练习想法。

最后，我们衷心感谢 Iohanna，Oliver，Luc，Milena，Luisa，Andrea 和 Atri 在撰写本书的漫长过程中给予的爱和支持。

没有以下软件，这本书将无法写成：R (版本 4.3.2; R 核心团队 2023) 和 R 包 *afex* (Singmann 等人 2020)、*barsurf* (版本 0.7.0; Spurdle 2020)、*bayesplot* (版本 1.11.1; Gabry 和 Mahr 2024)、*bcogsci* (版本 0.0.0.9000; Nicenboim、Schad 和 Vasishth 2024)、*bibtex* (Francois 2017)、*bivariate* (版本 0.7.0; Spurdle 2021)、*bookdown* (版本 0.39; Xie 2024a)、*brms* (版本 2.21.0; Bürkner 2024)、*citr* (Aust 2019)、*cmdstanr* (版本 0.8.0; Gabry 等人 2024)、*cowplot* (版本 1.1.3; Wilke 2024)、*digest* (版本 0.6.35; Antoine Lucas 等人 2021)、*dplyr* (版本 1.1.4; Wickham、François 等人 2023)、*DT* (Xie、Cheng 和 Tan 2019)、*extraDistr* (版本 1.10.0; Wolodzko 2023)、*forcats* (Wickham 2019a)、*gdtools* (Gohel 等人 2019)、*ggplot2* (版本 3.5.1; Wickham、Chang 等人 2024)、*gridExtra* (版本 2.3; Auguie 2017)、*htmlwidgets* (版本 1.6.4; Vaidyanathan 等人 2023)、*intoo* (版本 0.4.0; Spurdle 和 Bode 2020)、*kableExtra* (版本 1.4.0; Zhu 2024)、*knitr* (版本 1.47; Xie 2024b)、*lme4* (版本 1.1.35.3; Bates、Mächler 等人 2015)、*loo* (版本 2.7.0; Yao 等人 2017)、*MASS* (版本 7.3.60; Ripley 2023)、*Matrix* (版本 1.6.1.1; Bates、Maechler 和 Jagan 2023)、*miniUI* (版本 0.1.1.1; Cheng 2018)、*papaja* (版本 0.1.2; Aust 和 Barth 2020)、*pdftools* (版本 3.4.0; Ooms 2023)、*purrr* (版本 1.0.2; Wickham 和 Henry 2023)、*Rcpp* (版本 1.0.12; Eddelbuettel 等人 2024)、*readr* (Wickham、Hester 和 Francois 2018)、*RefManageR* (McLean 2017)、*remotes* (版本 2.5.0; Hester 等人 2021)、*rethinking* (版本 2.40; McElreath 2021)、*RJ-2021-048* (Bengtsson 2021)、*rmarkdown* (版本 2.27; Allaire 等人 2024)、*rootSolve* (版本 1.8.2.4; Soetaert 和 Herman 2009)、*rstan* (版本 2.35.0.9000; Guo 等人 2024)、*SBC* (版本 0.3.0.9000; Kim 等人 2024)、*servr* (Xie 2019)、*SHELF* (版本 1.10.0; Oakley 2024)、*SIN* (版本 0.6; Drton 2013)、*StanHeaders* (版本 2.35.0.9000; Goodrich 等人 2024)、*stringr* (版本 1.5.1; Wickham 2019b)、*texPreview* (Sidi 和 Polhamus 2020)、*tibble* (版本 3.2.1; Müller 和 Wickham 2020)、*tictoc* (版本 1.2.1; Izrailev 2024)、*tidyr* (版本 1.3.1; Wickham、Vaughan 和 Girlich 2024)、*tidyverse* (Wickham、Averick 等人 2019)、*tinylabels* (版本 0.2.4; Barth 2023) 和 *webshot* (Chang 2018)。

Bruno Nicenboim（荷兰蒂尔堡），Daniel Schad（德国波茨坦），Shravan Vasishth（德国波茨坦）

### 参考文献

Allaire, J. J., Yihui Xie, Christophe Dervieux, Jonathan McPherson, Javier Luraschi, Kevin Ushey, Aron Atkins 等. 2024\. *rmarkdown：R 的动态文档*. [`github.com/rstudio/rmarkdown`](https://github.com/rstudio/rmarkdown).

Antoine Lucas, Dirk Eddelbuettel 及其他贡献者，包括 Jarek Tuszynski, Henrik Bengtsson, Simon Urbanek, Mario Frasca, Bryan Lewis, Murray Stokely 等. 2021\. *Digest：创建 R 对象的紧凑哈希摘要*. [`CRAN.R-project.org/package=digest`](https://CRAN.R-project.org/package=digest).

Auguie, Baptiste. 2017\. *GridExtra：用于"Grid"图形的杂项函数*. [`CRAN.R-project.org/package=gridExtra`](https://CRAN.R-project.org/package=gridExtra).

Aust, Frederik. 2019\. *citr：RStudio 插件，用于插入 Markdown 引用*. [`CRAN.R-project.org/package=citr`](https://CRAN.R-project.org/package=citr).

Aust, Frederik, 和 Marius Barth. 2020\. *papaja：使用 R Markdown 创建 APA 论文*. [`github.com/crsh/papaja`](https://github.com/crsh/papaja).

Baayen, R. Harald, Douglas J. Davidson, 和 Douglas M. Bates. 2008\. “使用交叉随机效应的混合效应模型.” *记忆与语言杂志* 59 (4): 390–412.

Barth, Marius. 2023\. *tinylabels：轻量级变量标签*. [`github.com/mariusbarth/tinylabels`](https://github.com/mariusbarth/tinylabels).

Bates, Douglas M., Martin Mächler, Ben Bolker, 和 Steve Walker. 2015\. “使用 lme4 拟合线性混合效应模型.” *统计软件杂志* 67 (1): 1–48\. [`doi.org/10.18637/jss.v067.i01`](https://doi.org/10.18637/jss.v067.i01).

Bates, Douglas M., Martin Maechler, 和 Mikael Jagan. 2023\. *矩阵：稀疏和稠密矩阵类和方法*. [`Matrix.R-forge.R-project.org`](https://Matrix.R-forge.R-project.org).

Bengtsson, Henrik. 2021\. “在 R 中使用 Futures 的并行和分布式处理统一框架.” *R 杂志* 13 (2): 208–27\. [`doi.org/10.32614/RJ-2021-048`](https://doi.org/10.32614/RJ-2021-048).

Burger, Edward B., 和 Michael Starbird. 2012\. *有效思维的五个要素*. 普林斯顿大学出版社.

Bürkner, Paul-Christian. 2024\. *brms：使用“Stan”的贝叶斯回归模型*. [`github.com/paul-buerkner/brms`](https://github.com/paul-buerkner/brms).

Carpenter, Bob, Andrew Gelman, Matthew D. Hoffman, Daniel Lee, Ben Goodrich, Michael J. Betancourt, Marcus Brubaker, Jiqiang Guo, Peter Li, 和 Allen Riddell. 2017\. “Stan：一种概率编程语言.” *统计软件杂志* 76 (1).

Chang, Winston. 2018\. *webshot：网页截图*. [`CRAN.R-project.org/package=webshot`](https://CRAN.R-project.org/package=webshot).

Cheng, Joe. 2018\. *miniUI: 适用于小屏幕的 Shiny UI 小部件*. [`CRAN.R-project.org/package=miniUI`](https://CRAN.R-project.org/package=miniUI).

Drton, Mathias. 2013\. *SIN: 使用 SIN 方法选择高斯图马尔可夫模型*. [`CRAN.R-project.org/package=SIN`](https://CRAN.R-project.org/package=SIN).

Eddelbuettel, Dirk, Romain Francois, J. J. Allaire, Kevin Ushey, Qiang Kou, Nathan Russell, Inaki Ucar, Douglas M. Bates, and John Chambers. 2024\. *Rcpp: R 与 C++的无缝集成*. [`www.rcpp.org`](https://www.rcpp.org).

Francois, Romain. 2017\. *Bibtex: Bibtex 解析器*. [`CRAN.R-project.org/package=bibtex`](https://CRAN.R-project.org/package=bibtex).

Gabry, Jonah, Rok Češnovar, Andrew Johnson, and Steve Bronder. 2024\. *cmdstanr: “CmdStan”的 R 接口*. [`mc-stan.org/cmdstanr/`](https://mc-stan.org/cmdstanr/).

Gabry, Jonah, and Tristan Mahr. 2024\. *bayesplot: 贝叶斯模型的绘图*. [`mc-stan.org/bayesplot/`](https://mc-stan.org/bayesplot/).

Gelman, Andrew, John B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari, and Donald B. Rubin. 2014\. *《贝叶斯数据分析》*. 第三版. Boca Raton, FL: Chapman; Hall/CRC Press.

Gill, Jeff. 2006\. *《政治和社会研究的必要数学》*. 剑桥大学出版社剑桥.

Gohel, David, Hadley Wickham, Lionel Henry, and Jeroen Ooms. 2019\. *gdtools: 图形渲染的实用工具*. [`CRAN.R-project.org/package=gdtools`](https://CRAN.R-project.org/package=gdtools).

Goodrich, Ben, Andrew Gelman, Bob Carpenter, Matthew D. Hoffman, Daniel Lee, Michael J. Betancourt, Marcus Brubaker, et al. 2024\. *StanHeaders: Stan 的 C++头文件*. [`mc-stan.org/`](https://mc-stan.org/).

Guo, Jiqiang, Jonah Gabry, Ben Goodrich, Andrew Johnson, Sebastian Weber, and Hamada S. Badr. 2024\. *rstan: Stan 的 R 接口*. [`mc-stan.org/rstan/`](https://mc-stan.org/rstan/).

Hester, Jim, Gábor Csárdi, Hadley Wickham, Winston Chang, Martin Morgan, and Dan Tenenbaum. 2021\. *Remotes: 从远程仓库安装 R 包，包括‘Github’*. [`CRAN.R-project.org/package=remotes`](https://CRAN.R-project.org/package=remotes).

Izrailev, Sergei. 2024\. *Tictoc: 用于计时 R 脚本以及“Stack”和“Stacklist”结构的实现*. [`github.com/jabiru/tictoc`](https://github.com/jabiru/tictoc).

Kerns, G. J. 2014\. *使用 R 的概率论与数理统计导论*. 第二版.

Kim, Shinyoung, Hyunji Moon, Martin Modrák, and Teemu Säilynoja. 2024\. *SBC: Rstan/Cmdstanr 模型的基于模拟的校准*. [`hyunjimoon.github.io/SBC/`](https://hyunjimoon.github.io/SBC/).

Levy, Dan. 2021\. *《分析性思维的格言：传奇哈佛教授理查德·泽克豪瑟的智慧》*. Dan Levy.

McElreath, Richard. 2021\. *Rethinking: 统计重思的书籍包*.

McLean, Mathew William. 2017\. “RefManageR：在 R 中导入和管理 Bibtex 和 Biblatex 参考文献.” *开源软件杂志*. [`doi.org/10.21105/joss.00338`](https://doi.org/10.21105/joss.00338).

Müller, Kirill, 和 Hadley Wickham. 2020\. *Tibble：简单数据框*. [`CRAN.R-project.org/package=tibble`](https://CRAN.R-project.org/package=tibble).

Nicenboim, Bruno, Daniel J. Schad, 和 Shravan Vasishth. 2024\. *bcogsci: 认知科学贝叶斯数据分析入门书籍的数据和模型*.

Oakley, Jeremy E. 2024\. *SHELF：支持 Sheffield 启发式框架的工具*. [`github.com/OakleyJ/SHELF`](https://github.com/OakleyJ/SHELF).

Ooms, Jeroen. 2023\. *pdftools：PDF 文档的文本提取、渲染和转换*. [`docs.ropensci.org/pdftools/`](https://docs.ropensci.org/pdftools/).

R Core Team. 2023\. *R：统计计算的语言和环境*. 奥地利维也纳：R 统计计算基金会. [`www.R-project.org/`](https://www.R-project.org/).

Ripley, Brian D. 2023\. *MASS：Venables 和 Ripley 的 MASS 的支持函数和数据集*. [`www.stats.ox.ac.uk/pub/MASS4/`](http://www.stats.ox.ac.uk/pub/MASS4/).

Sidi, Jonathan, 和 Daniel Polhamus. 2020\. *TexPreview：编译和预览“LaTeX”片段*. [`CRAN.R-project.org/package=texPreview`](https://CRAN.R-project.org/package=texPreview).

Singmann, Henrik, Ben Bolker, Jake Westfall, Frederik Aust, 和 Mattan S. Ben-Shachar. 2020\. *Afex：因子实验分析*. [`CRAN.R-project.org/package=afex`](https://CRAN.R-project.org/package=afex).

Soetaert, Karline, 和 Peter M. J. Herman. 2009\. *生态建模实用指南：使用 R 作为模拟平台*.

Spurdle, Abby. 2020\. *Barsurf：与热图相关的绘图和平滑多波段颜色插值*. [`CRAN.R-project.org/package=barsurf`](https://CRAN.R-project.org/package=barsurf).

Spurdle, Abby. 2021\. *Bivariate: 双变量概率分布*. [`sites.google.com/site/spurdlea/r`](https://sites.google.com/site/spurdlea/r).

Spurdle, Abby, 和 Emil Bode. 2020\. *Intoo：最小化语言扩展*. [`CRAN.R-project.org/package=intoo`](https://CRAN.R-project.org/package=intoo).

Vaidyanathan, Ramnath, Yihui Xie, J. J. Allaire, Joe Cheng, 和 Kenton Russell. 2023\. *htmlwidgets：R 的 HTML 小部件*. [`github.com/ramnathv/htmlwidgets`](https://github.com/ramnathv/htmlwidgets).

Vasishth, Shravan, Daniel J. Schad, Audrey Bürki, 和 Reinhold Kliegl. 2021\. “语言学和心理学中的线性混合模型：全面介绍.” [`vasishth.github.io/Freq_CogSci/`](https://vasishth.github.io/Freq_CogSci/).

Wickham, Hadley. 2019a. *forcats：处理分类变量（因子）的工具*. [`CRAN.R-project.org/package=forcats`](https://CRAN.R-project.org/package=forcats).

魏克汉姆，哈德利. 2019b. *stringr: 常见字符串操作的简单、一致的包装器*. [`CRAN.R-project.org/package=stringr`](https://CRAN.R-project.org/package=stringr).

魏克汉姆，哈德利，玛拉·阿维里克，詹妮弗·布莱恩，温斯顿·张，露西·D’阿戈斯蒂诺·麦戈文，罗曼·弗朗索瓦，加勒特·格罗勒蒙德等. 2019. “欢迎来到 tidyverse.” *开源软件杂志* 4 (43): 1686. [`doi.org/10.21105/joss.01686`](https://doi.org/10.21105/joss.01686).

魏克汉姆，哈德利，温斯顿·张，利昂内尔·亨利，托马斯·林·佩德森，高桥耕介，克劳斯·O·威尔克，卡拉·伍，山本博昭，特恩·范登·布兰德. 2024. *ggplot2: 使用图形语法创建优雅的数据可视化*. [`ggplot2.tidyverse.org`](https://ggplot2.tidyverse.org).

魏克汉姆，哈德利，罗曼·弗朗索瓦，利昂内尔·亨利，基里尔·穆勒，戴维斯·沃恩. 2023. *dplyr: 数据操作的语法*. [`dplyr.tidyverse.org`](https://dplyr.tidyverse.org).

魏克汉姆，哈德利，利昂内尔·亨利. 2023. *purrr: 函数式编程工具*. [`purrr.tidyverse.org/`](https://purrr.tidyverse.org/).

魏克汉姆，哈德利，吉姆·赫斯特，罗曼·弗朗索瓦. 2018. *readr: 读取矩形文本数据*. [`CRAN.R-project.org/package=readr`](https://CRAN.R-project.org/package=readr).

魏克汉姆，哈德利，戴维斯·沃恩，马克西米利安·吉尔利希. 2024. *Tidyr: 整洁混乱数据*. [`tidyr.tidyverse.org`](https://tidyr.tidyverse.org).

威尔克，克劳斯·O. 2024. *cowplot: 为 ‘ggplot2’ 提供简化的绘图主题和绘图注释*. [`wilkelab.org/cowplot/`](https://wilkelab.org/cowplot/).

伍洛德科，蒂莫捷乌什. 2023. *extraDistr: 额外的单变量和多变量分布*. [`github.com/twolodzko/extraDistr`](https://github.com/twolodzko/extraDistr).

谢益辉. 2019. *servr: 一个简单的 HTTP 服务器，用于提供静态文件或动态文档*. [`CRAN.R-project.org/package=servr`](https://CRAN.R-project.org/package=servr).

谢益辉. 2024a. *bookdown: 使用 R Markdown 编写书籍和技术文档*. [`github.com/rstudio/bookdown`](https://github.com/rstudio/bookdown).

谢益辉. 2024b. *knitr: R 中动态报告生成的通用包*. [`yihui.org/knitr/`](https://yihui.org/knitr/).

谢益辉，乔·程，谭晓莺. 2019. *DT: JavaScript 库 ‘Datatables’ 的包装器*. [`CRAN.R-project.org/package=DT`](https://CRAN.R-project.org/package=DT).

姚宇灵，阿基·维哈里，丹尼尔·P·辛普森，安德鲁·格尔曼. 2017. “使用堆叠平均贝叶斯预测分布.” *贝叶斯分析*. [`doi.org/10.1214/17-BA1091`](https://doi.org/10.1214/17-BA1091).

朱浩. 2024. *KableExtra: 使用 ‘Kable’ 和管道语法构建复杂表格*. [`haozhu233.github.io/kableExtra/`](http://haozhu233.github.io/kableExtra/).
