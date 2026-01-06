# 量子信息

> 原文：[`www.algorithm-archive.org/contents/quantum_information/quantum_information.html`](https://www.algorithm-archive.org/contents/quantum_information/quantum_information.html)

量子信息理论是...强烈的。它需要我们对经典信息理论和量子力学有强烈和根本的理解。它以任何方式都不是显而易见的，并且值得有自己的一套教科书。实际上，关于这个主题已经有大量的教科书了。本节的目的不是超越任何基本知识。相反，我们将尝试将知识提炼成一个简短、直观的总结，希望帮助人们更好地理解这个主题，并在此基础上进一步探索。

在撰写本文时，真正的量子计算机尚不存在。我们有一些能够模拟量子位的系统，它们并不是真正的通用量子计算机。目前市场上最接近的系统是 D-WAVE，它拥有令人印象深刻的 128 个量子位！

介绍量子信息理论有很多起点，所以我们将一步一步地进行：

1.  **量子位逻辑**：什么是量子位以及它与经典位有什么不同？

1.  **量子门和量子电路**：你如何从根本上构建一个量子算法？

1.  **野外的量子计算机**：当前创建量子计算机的实验技术以及它们为什么不适合作为真正的量子计算机

1.  **当前量子算法概述**：有一些算法在量子计算机上运行时承诺带来巨大的优势，当它们最终在实验中实现时，应该真正地颠覆行业。

作为备注，第 3 项可能对于一个算法书籍来说显得有些不合适，我也有同感；然而，目前正在进行大量的研究以实现第一个真正的量子计算机，并且有几种潜在的系统可能适用于这个目的。这些系统将改变我们未来对量子计算的看法和接口，讨论该领域可能的发展方向以及我们何时可以期待在家中使用量子计算机是很重要的。

现在，能够编译量子代码的语言并不多。不久前，我们尝试制作一个量子电路编译器，它是基于 SPICE 电路模拟器的模型，但这远非一种计算机语言。在目前这个时间点，我们无法预测当我们最终拥有真正的量子机器时，量子计算语言会是什么样子，所以目前我们不会要求社区为与量子信息相关的章节提供代码。

事实上，很难想象如何在 C 语言中充分实现 Shor 算法。像往常一样，这个部分将在我们添加更多算法到列表中时进行更新。

## 许可证

##### 代码示例

代码示例许可在 MIT 许可下（可在[LICENSE.md](https://github.com/algorithm-archivists/algorithm-archive/blob/main/LICENSE.md)中找到）。

##### 文本

本章的文本由[James Schloss](https://github.com/leios)编写，并许可在[Creative Commons Attribution-ShareAlike 4.0 国际许可协议](https://creativecommons.org/licenses/by-sa/4.0/legalcode)下使用。

[](https://creativecommons.org/licenses/by-sa/4.0/)

![](https://creativecommons.org/licenses/by-sa/4.0/)

##### 拉取请求

在初始许可([#560](https://github.com/algorithm-archivists/algorithm-archive/pull/560))之后，以下拉请求已修改本章的文本或图形：

+   无
