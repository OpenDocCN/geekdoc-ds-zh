# 如何为算法档案做出贡献

> 原文：[`www.algorithm-archive.org/contents/how_to_contribute/how_to_contribute.html`](https://www.algorithm-archive.org/contents/how_to_contribute/how_to_contribute.html)

*算法档案*是一个作为社区学习和教授算法的努力。因此，它需要在社区成员之间建立一定的信任。有关如何贡献的具体细节，请参阅[如何贡献指南](https://github.com/algorithm-archivists/algorithm-archive/wiki/How-to-Contribute)。如果你在 git 和版本控制方面遇到困难，也请查看[这个视频系列](https://www.youtube.com/playlist?list=PL5NSPcN6fRq2vwgdb9noJacF945CeBk8x)以获取更多详细信息。

此外，我们还有一个[常见问题解答](https://github.com/algorithm-archivists/algorithm-archive/wiki/FAQ)和一个[代码风格指南](https://github.com/algorithm-archivists/algorithm-archive/wiki/Code-style-guide)，该指南目前正在为迄今为止提交到算法档案的所有语言编写。

目前，我们不接受章节提交；然而，我们将在不久的将来允许这样做。现在，以下是向算法档案提交代码的基本步骤：

1.  **风格**: 我们正在为算法档案中的所有语言开发一个[代码风格指南](https://github.com/algorithm-archivists/algorithm-archive/wiki/Code-style-guide)。大部分情况下，遵循你选择的语言的标准风格指南。你的代码应该对任何人都是可读和可理解的——特别是那些刚开始学习这门语言的人。此外，记住你的代码将在这个书中展示，所以尽量保持大约 80 列宽，尽量去除任何视觉杂乱，并保持变量名简洁易懂。

1.  **许可**: 此项目的所有代码都将位于`LICENSE.md`中的 MIT 许可下；然而，文本将位于 Creative Commons Attribution-NonCommercial 4.0 International 许可下。

1.  **CONTRIBUTORS.md**: 在贡献代码后，请在`CONTRIBUTORS.md`的末尾添加你的名字，使用`echo "- name" >> CONTRIBUTORS.md`。

1.  **构建算法档案**: 在每次提交之前，你应在自己的机器上构建算法档案。为此，安装[Node](https://nodejs.org/)并使用`npm install`然后在主目录（`README.md`所在位置）中运行`npm run serve`。这将提供一个本地 URL，你可以通过浏览器查看档案。使用这个服务器确保你的算法档案版本在更新的章节中可以干净地工作！

要**提交代码**，前往你想要提交的章节的`code/`目录，并为你的选择语言添加另一个目录。

如果你能够**审查代码**，并且有能力审查一种语言（并且希望被要求这样做），请将自己添加到代码审查者列表中。

我们使用两个 GitBook 插件来允许用户在不同算法之间切换语言。一个是 theme-api，另一个是 include-codeblock api。为了让这两个插件协同工作，Markdown 文件中需要以下语句：

```
{% method %}
{% sample lang="jl" %}
import:1-17, lang:"julia"
{% endmethod %} 
```

在这个例子中，我们启动了 theme-api 的 `method` 并从代码目录中的一个示例 Julia 片段中导入第 1-17 行。请注意，为了标准化语言的命名方案，我们要求每种语言的 `sample lang` 是其代码文件的扩展名，例如 `cpp` 对于 C++，`hs` 对于 Haskell 等。这保持了 theme-api 在不同语言中的标题一致性。另外，请注意，根据算法的不同，可能还需要编写一些文本中的代码片段。

随着项目的增长，我会更新这个页面。如果您想参与持续的讨论，请随时加入我们的 Discord 服务器：[`discord.gg/pb976sY`](https://discord.gg/pb976sY)。感谢所有支持并考虑为算法档案做出贡献！

## 许可证

##### 代码示例

代码示例许可在 MIT 许可证下（可在 [LICENSE.md](https://github.com/algorithm-archivists/algorithm-archive/blob/main/LICENSE.md) 中找到）。

##### 文本

本章的文本由 [James Schloss](https://github.com/leios) 编写，并许可在 [Creative Commons Attribution-ShareAlike 4.0 国际许可协议](https://creativecommons.org/licenses/by-sa/4.0/legalcode) 下。

[](https://creativecommons.org/licenses/by-sa/4.0/)

![](https://creativecommons.org/licenses/by-sa/4.0/)

##### 提交请求

在初始许可后 ([#560](https://github.com/algorithm-archivists/algorithm-archive/pull/560))，以下提交请求已修改了本章的文本或图形：

+   无
