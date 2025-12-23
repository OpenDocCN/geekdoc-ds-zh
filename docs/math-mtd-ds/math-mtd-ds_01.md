# 数据科学中的数学方法#

> 原文：[`mmids-textbook.github.io/`](https://mmids-textbook.github.io/)

**作者:** [Sebastien Roch](https://people.math.wisc.edu/~roch/)，威斯康星大学麦迪逊分校数学系

**完整标题:** 数据科学中的数学方法：使用 Python 连接理论与应用

![_images/9781009509404i.jpg](img/9781009509404i.jpg)

**印刷版:** 这本教科书的印刷版已由剑桥大学出版社出版。您可以通过[这里](https://www.amazon.com/Mathematical-Methods-Data-Science-Applications/dp/1009509403)订购。在线版本将继续提供并维护。更多资源可以在[出版商网站](https://www.cambridge.org/highereducation/books/mathematical-methods-in-data-science/6CB866F77A7CA33109EF99910CFA40BC#overview)找到。有关错别字的列表，请参阅这里。本书基于为[MATH 535: 数据科学中的数学方法](https://people.math.wisc.edu/~roch/mmids/)开发的 Jupyter 笔记本，这是一门在[UW-Madison](https://math.wisc.edu/)提供的为期一个学期的本科和研究生水平的高级课程。在线版本使用[Jupyter Book](https://jupyterbook.org/stable/)生成。有关其他此类书籍的集合，请参阅[这里](https://executablebooks.org/en/latest/gallery/)。

**描述:** 这本关于数据与 AI 数学的教科书有几个目标受众：

+   *针对数学或物理、经济学、工程等其他定量领域的学生：它旨在从严格的数学角度邀请学生进入数据科学和 AI。*

+   *针对数据科学相关领域（本科或研究生水平）的数学倾向学生：它可以作为机器学习、AI 和统计学课程的数学伴侣。*

在内容上，这是一门关于多变量微积分、线性代数和概率的后续课程，由数据科学应用激发并举例说明。因此，读者应熟悉这些领域的基础知识，以及接触过证明——但不假设有数据科学的知识。此外，虽然重点是数学概念和方法，但全书都使用了编码。对[Python](https://docs.python.org/3/tutorial/index.html)的基本熟悉程度就足够了。本书介绍了某些专业软件包的入门，特别是[Numpy](https://numpy.org)、[NetworkX](https://networkx.org)和[PyTorch](https://pytorch.org)。

+   1. 引言：第一个数据科学问题

+   2. 最小二乘法：几何、代数和数值方面

+   3. 优化理论与算法

+   4.奇异值分解

+   5.图谱谱理论

+   6. 从简单到复杂的概率模型

+   7.图上的随机游走和马尔可夫链

+   8. 神经网络、反向传播和随机梯度下降

重要

要运行本书中的代码，你需要导入以下库。

```py
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import torch
import mmids 
```

文件 `mmids.py` 在[这里](https://raw.githubusercontent.com/MMiDS-textbook/MMiDS-textbook.github.io/main/utils/mmids.py)。所有数据集都可以在[笔记的 GitHub 页面](https://github.com/MMiDS-textbook/MMiDS-textbook.github.io/tree/main/utils/datasets)上下载。

每一章的末尾都提供了仅包含代码的 Jupyter 笔记本。建议在[Google Colaboratory](https://colab.google)上运行它们。这些笔记本也以幻灯片格式提供。幻灯片是用 Jupyter 制作的；因此，讲师可以直接从笔记本中创建自己的定制版本。

注意

如果你发现错别字（在线或印刷版），请通过右上角的按钮在 GitHub 上创建一个问题。

**补充材料**：这个在线版本还包含补充了印刷版书籍中可以找到的内容。具体来说，在每一章的末尾，你将找到一个名为“*在线补充材料*”的部分，其中包括：

+   *仅代码*：包含章节中所有代码的 Jupyter 笔记本和幻灯片

+   *自我评估测验*：每个部分的扩展、交互式自我评估测验

+   *自动测验*：包含随机测验和自动生成答案的 Jupyter 笔记本

+   *热身练习解答*：所有奇数编号热身练习的解答

+   *附加章节*：额外的内容，通常比已出版的书籍更高级（例如，不要求在正文中证明的更高级结果）

**图片来源**：侧边栏标志是用[Midjourney](https://www.midjourney.com/)制作的
