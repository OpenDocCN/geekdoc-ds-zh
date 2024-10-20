# 教程:使用 rvest 在 R 中进行 Web 抓取

> 原文：<https://www.dataquest.io/blog/web-scraping-in-r-rvest/>

April 13, 2020![we'll do some web scraping in R with rvest to gather data about the weather](img/d59656bf0c2dfca3acf421f8aa64cb85.png)

互联网上有很多数据集，你可以用它们来做你自己的个人项目。有时候你很幸运，你可以访问一个 [API](https://www.dataquest.io/blog/r-api-tutorial/) ，在那里你可以直接用 r 请求数据。其他时候，你就没那么幸运了，你不能以简洁的格式获得数据。当这种情况发生时，我们需要求助于**网络抓取**，这是一种通过在网站的 HTML 代码中找到我们想要分析的数据来获取数据的技术。

在本教程中，我们将介绍如何在 r 中进行网络抓取的基础知识。我们将从[国家气象局](https://www.weather.gov/)网站抓取天气预报数据，并将其转换为可用的格式。

> install.packages("Dataquest ")

从我们的[R 课程简介](/course/intro-to-r/)开始学习 R——不需要信用卡！

[SIGN UP](https://app.dataquest.io/signup)

当我们找不到我们要找的数据时，网络抓取为我们提供了机会，并给了我们实际创建数据集所需的工具。由于我们使用 R 来进行 web 抓取，如果我们使用的站点更新了，我们可以简单地再次运行我们的代码来获得更新的数据集。

## 理解网页

在我们开始学习如何抓取网页之前，我们需要了解网页本身是如何构造的。

从用户的角度来看，网页上的文本、图像和链接都以一种美观易读的方式组织起来。但是网页本身是用特定的编码语言编写的，然后由我们的网络浏览器解释。当我们进行网络抓取时，我们需要处理网页本身的实际内容:浏览器解释之前的代码。

用于构建网页的主要语言称为超文本标记语言(HTML)、层叠样式表(CSS)和 Javascript。HTML 给出了网页的实际结构和内容。CSS 赋予网页风格和外观，包括字体和颜色等细节。Javascript 赋予了网页功能。

在本教程中，我们将主要关注如何使用 R web scraping 来读取组成网页的 HTML 和 CSS。

## 超文本标记语言

与 R 不同，HTML 不是一种编程语言。相反，它被称为*标记语言*——它描述了网页的内容和结构。HTML 使用**标签**来组织，这些标签被`<>`符号包围。不同的标签执行不同的功能。许多标签一起将形成并包含网页的内容。

最简单的 HTML 文档如下所示:

```py
<html>
<head>
```

虽然上面是一个合法的 HTML 文档，但它没有文本或其他内容。如果我们将其保存为. html 文件，并使用 web 浏览器打开它，我们会看到一个空白页面。

注意单词`html`被`<>`括号包围，这表明它是一个标签。为了给这个 HTML 文档添加更多的结构和文本，我们可以添加以下内容:

```py
<head>
</head>
<body>
<p>
Here's a paragraph of text!
</p>
<p>
Here's a second paragraph of text!
</p>
</body>
</html>
```

这里我们添加了`<head>`和`<body>`标签，它们为文档增加了更多的结构。`<p>`标签是我们在 HTML 中用来指定段落文本的。

HTML 中有很多很多的标签，但是我们不可能在本教程中涵盖所有的标签。如果有兴趣，可以去[这个网站](https://developer.mozilla.org/en-US/docs/Web/HTML/Element)看看。重要的是要知道标签有特定的名字(`html`、`body`、`p`等)。)来使它们在 HTML 文档中可识别。

请注意，每个标签都是“成对”的，即每个标签都伴随着另一个具有相似名称的标签。也就是说，开始标签`<html>`与另一个标签`</html>`配对，指示 HTML 文档的开始和结束。这同样适用于`<body>`和`<p>`。

认识到这一点很重要，因为它允许标签互相嵌套*。<主体>和<头部>标签嵌套在`<html>`内，`<p>`嵌套在`<body>`内。这种嵌套给了 HTML 一种“树状”结构:*

 *当我们使用 R 进行网页抓取时，这种树状结构将告诉我们如何寻找某些标签，所以记住这一点很重要。如果一个标签中嵌套了其他标签，我们会将包含标签称为*父标签*，并将其中的每个标签称为“子标签”。如果父标签中有多个子标签，则子标签统称为“兄弟”。这些父、子和兄弟的概念让我们了解了标签的层次结构。

## 半铸钢ˌ钢性铸铁(Cast Semi-Steel)

HTML 提供了网页的内容和结构，而 CSS 提供了网页应该如何设计的信息。没有 CSS，网页是非常简单的。这里有一个简单的没有 CSS 的 HTML 文档来演示这一点。

当我们说造型的时候，我们指的是一个*宽，宽*范围的东西。样式可以指特定 HTML 元素的颜色或它们的位置。像 HTML 一样，CSS 材料的范围如此之大，以至于我们无法涵盖语言中每一个可能的概念。如果你有兴趣，你可以在这里了解更多。

在我们深入研究 R web 抓取代码之前，我们需要了解的两个概念是**类**和**id**。

首先说一下班级。如果我们在制作一个网站，我们经常会希望网站的相似元素看起来一样。例如，我们可能希望列表中的许多项目都以相同的颜色显示，红色。

我们可以通过将一些包含颜色信息的 CSS 直接插入到每一行文本的 HTML 标记中来实现，就像这样:

```py
<p style=”color:red” >Text 1</p>
<p style=”color:red” >Text 2</p>
<p style=”color:red” >Text 3</p>
```

`style`文本表明我们正试图将 CSS 应用于`<p>`标签。在引号内，我们看到一个键值对“color:red”。`color`指的是`<p>`标签中文本的颜色，而红色描述了应该是什么颜色。

但是正如我们在上面看到的，我们已经多次重复了这个键值对。这并不理想——如果我们想改变文本的颜色，我们必须一行一行地改变。

我们可以用一个`class`选择器代替所有这些`<p>`标签中的`style`文本:

```py
<p class=”red-text” >Text 1</p>
<p class=”red-text” >Text 2</p>
<p class=”red-text” >Text 3</p>
```

`class`选择器，我们可以更好地表明这些`<p>`标签在某种程度上是相关的。在一个单独的 CSS 文件中，我们可以创建红色文本类，并通过编写以下内容来定义它的外观:

```py
.red-text {
    color : red;
}
```

将这两个元素组合成一个网页将产生与第一组红色`<p>`标签相同的效果，但是它允许我们更容易地进行快速更改。

当然，在本教程中，我们感兴趣的是网页抓取，而不是构建网页。但是当我们抓取网页时，我们经常需要选择一个特定的 HTML 标签类，所以我们需要了解 CSS 类如何工作的基础知识。

类似地，我们可能经常想要抓取使用 **id** 标识的特定数据。CSS ids 用于给单个元素一个可识别的名称，很像一个类如何帮助定义一类元素。

```py
<p id=”special” >This is a special tag.</p>
```

如果一个 id 被附加到一个 HTML 标签上，当我们用 r 执行实际的 web 抓取时，它会使我们更容易识别这个标签。

如果您还不太理解类和 id，不要担心，当我们开始操作代码时，它会变得更加清晰。

有几个 R 库被设计用来获取 HTML 和 CSS，并且能够遍历它们来寻找特定的标签。我们将在本教程中使用的库是`rvest`。

## rvest 图书馆

由传奇人物哈德利·威克姆维护的 [`rvest`图书馆](https://github.com/tidyverse/rvest)，是一个让用户轻松从网页上抓取(“收获”)数据的图书馆。

`rvest`是`tidyverse`库中的一个，所以它可以很好地与捆绑包中包含的其他库一起工作。`rvest`从 web 抓取库 BeautifulSoup 获取灵感，该库来自 Python。(相关:o [ur BeautifulSoup Python 教程。](https://www.dataquest.io/blog/web-scraping-python-using-beautiful-soup/)

## 在 R 中抓取网页

为了使用`rvest`库，我们首先需要安装它并用`library()`函数导入它。

```py
install.packages(“rvest”)
```

```py
library(rvest)
```

为了开始解析网页，我们首先需要从包含它的计算机服务器请求数据。在 revest 中，服务于此目的的函数是`read_html()`函数。

`read_html()`接受一个 web URL 作为参数。让我们从前面的简单的无 CSS 页面开始，看看这个函数是如何工作的。

```py
simple <- read_html("https://dataquestio.github.io/web-scraping-pages/simple.html")
```

`read_html()`函数返回一个列表对象，它包含我们前面讨论过的树状结构。

```py
simple
```

```py
{html_document}
<html>
[1] <head>\n<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">\n<title>A simple exa ...
[2] <body>\n        <p>Here is some simple content for this page.</p>\n    </body>
```

假设我们想要将包含在单个`<p>`标签中的文本存储到一个变量中。为了访问这段文本，我们需要弄清楚如何*定位*这段特定的文本。这通常是 CSS 类和 id 可以帮助我们的地方，因为优秀的开发人员通常会使 CSS 在他们的网站上高度具体化。

在这种情况下，我们没有这样的 CSS，但是我们知道我们想要访问的`<p>`标签是页面上唯一的同类标签。为了捕获文本，我们需要分别使用`html_nodes()`和`html_text()`函数来搜索这个`<p>`标签并检索文本。下面的代码实现了这一点:

```py
simple %>%
html_nodes("p") %>%
html_text()
```

```py
"Here is some simple content for this page."
```

`simple`变量已经包含了我们试图抓取的 HTML，所以剩下的任务就是从其中搜索我们想要的元素。由于我们使用的是`tidyverse`，我们可以将 HTML 通过管道传输到不同的函数中。

我们需要将特定的 HTML 标签或 CSS 类传递给`html_nodes()`函数。我们需要`<p>`标签，所以我们将字符“p”传入函数。`html_nodes()`也返回一个列表，但是它返回 HTML 中的所有节点，这些节点具有您给它的特定 HTML 标签或 CSS class/id。一个*节点*是指树状结构上的一个点。

一旦我们拥有了所有这些节点，我们就可以将`html_nodes()`的输出传递给`html_text()`函数。我们需要得到< p >标签的实际文本，所以这个函数可以帮助我们。

这些功能共同构成了许多常见 web 抓取任务的主体。一般来说，R(或任何其他语言)中的 web 抓取可以归结为以下三个步骤:

*   获取您想要抓取的网页的 HTML】
*   决定你想阅读页面的哪一部分，并找出你需要什么 HTML/CSS 来选择它
*   选择 HTML 并以你需要的方式进行分析

## 目标网页

在本教程中，我们将关注国家气象局的网站。假设我们对创建自己的天气应用程序感兴趣。我们需要天气数据本身来填充它。

天气数据每天都在更新，所以当我们需要时，我们将使用网络抓取从 NWS 网站获取这些数据。

出于我们的目的，我们将从旧金山获取数据，但是每个城市的网页看起来都一样，所以同样的步骤也适用于任何其他城市。旧金山页面的屏幕截图如下所示:

我们对天气预报和每天的温度特别感兴趣。每天都有白天预报和夜晚预报。既然我们已经确定了我们需要的网页部分，我们就可以在 HTML 中挖掘，看看我们需要选择什么标签或类来捕获这个特定的数据。

## 使用 Chrome 开发工具

令人欣慰的是，大多数现代浏览器都有一个工具，允许用户直接检查任何网页的 HTML 和 CSS。在 Google Chrome 和 Firefox 中，它们被称为开发者工具，在其他浏览器中也有类似的名称。对于本教程来说，最有用的工具是检查器。

你可以在浏览器的右上角找到开发者工具。如果你使用 Firefox，你应该可以看到开发者工具，如果你使用 Chrome，你可以通过`View -> More Tools -> Developer Tools`。这将在您的浏览器窗口中打开开发者工具:

我们之前处理的 HTML 是最基本的，但是你在浏览器中看到的大多数网页都极其复杂。开发者工具将使我们更容易挑选出我们想要抓取和检查 HTML 的网页的确切元素。

我们需要查看天气页面的 HTML 中的温度，所以我们将使用 Inspect 工具来查看这些元素。Inspect 工具将挑选出我们正在寻找的准确的 HTML，所以我们不必自己寻找！

通过点击元素本身，我们可以看到七天的天气预报包含在下面的 HTML 中。我们压缩了一些内容，使其更具可读性:

```py
<div id="seven-day-forecast-container">
<ul id="seven-day-forecast-list" class="list-unstyled">
<li class="forecast-tombstone">
<div class="tombstone-container">
<p class="period-name">Tonight<br><br></p>
<p><img src="newimages/medium/nskc.png" alt="Tonight: Clear, with a low around 50\. Calm wind. " title="Tonight: Clear, with a low around 50\. Calm wind. " class="forecast-icon"></p>
<p class="short-desc" style="height: 54px;">Clear</p>
<p class="temp temp-low">Low: 50 °F</p></div>
</li>
# More elements like the one above follow, one for each day and night
</ul>
</div>
```

## 利用我们所学的知识

既然我们已经确定了我们需要在网页中定位哪些特定的 HTML 和 CSS，我们可以使用`rvest`来捕获它。

从上面的 HTML 来看，似乎每个温度都包含在类`temp`中。一旦我们有了所有这些标签，我们就可以从中提取文本。

```py
forecasts <- read_html("https://forecast.weather.gov/MapClick.php?lat=37.7771&lon=-122.4196#.Xl0j6BNKhTY") %>%
    html_nodes(“.temp”) %>%
    html_text()

forecasts
```

```py
[1] "Low: 51 °F" "High: 69 °F" "Low: 49 °F" "High: 69 °F"
[5] "Low: 51 °F" "High: 65 °F" "Low: 51 °F" "High: 60 °F"
[9] "Low: 47 °F"
```

使用这个代码，`forecasts`现在是一个与低温和高温相对应的字符串向量。

现在我们有了我们感兴趣的 R 变量的实际数据，我们只需要做一些常规的数据分析，将向量转换成我们需要的格式。例如:

```py
library(readr)
parse_number(forecasts)
```

```py
[1] 51 69 49 69 51 65 51 60 47
```

## 后续步骤

`rvest`库使得使用与`tidyverse`库相同的技术来执行 web 抓取变得简单方便。

本教程应该给你必要的工具来启动一个小型的 web 抓取项目，并开始探索更高级的 web 抓取过程。一些与网络抓取极其兼容的网站是体育网站、有股票价格甚至新闻文章的网站。

或者，你可以继续扩展这个项目。你还能为你的天气应用收集到哪些其他的天气预报元素？

如果你想了解更多关于这个主题的知识，请查看 Dataquest 在 R 课程中的交互式[网页抓取，以及我们的](https://www.dataquest.io/course/scraping-in-r/)[API 和 R](https://www.dataquest.io/path/apis-and-web-scraping-with-r/) 网页抓取，它们将帮助你在大约 2 个月内掌握这些技能。

### 准备好提升你的 R 技能了吗？

我们的[R 路数据分析师](/path/data-analyst-r/)涵盖了您获得工作所需的所有技能，包括:

*   使用 **ggplot2** 进行数据可视化
*   使用 **tidyverse** 软件包的高级数据清理技能
*   R 用户的重要 SQL 技能
*   **统计**和概率的基础知识
*   ...还有**多得多的**

没有要安装的东西，**没有先决条件**，也没有时间表。

[Start learning for free!](https://app.dataquest.io/signup)*