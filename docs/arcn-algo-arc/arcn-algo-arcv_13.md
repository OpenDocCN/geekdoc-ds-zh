# 仿射变换

> 原文：[`www.algorithm-archive.org/contents/affine_transformations/affine_transformations.html`](https://www.algorithm-archive.org/contents/affine_transformations/affine_transformations.html)

仿射变换是一类数学运算，它包括旋转、缩放、平移、剪切以及一些在数学和计算机图形学中经常使用的类似变换。首先，我们将在仿射变换和线性变换之间画一条明确的（尽管很细）界限，然后再讨论实践中通常使用的扩展矩阵形式。

## 仿射（和线性）变换的快速介绍

让我们从二维平面上的一个给定点，  开始。如果我们把这个点当作一个  向量，我们可以通过乘以一个  变换矩阵来将其转换成另一个  向量。同样，一个三维点可以看作一个  向量，需要使用一个  变换矩阵。这类操作被称为线性变换，通常表示为，

这里，  是一个  变换矩阵，其中  是输入和输出向量的长度，  和  分别。虽然这些变换很强大，但它们都是以原点为中心的。仿射变换扩展了线性变换的这种限制，并允许我们将初始向量的位置平移，使得

在这里，  是一个  平移向量。为了理解这些变换的力量，看到它们在实际中的应用是非常重要的：

| 描述 | 变换 |
| --- | --- |

| 沿着  的缩放 |  <res/a11_square_white.mp4>

您的浏览器不支持视频标签。

| 沿着  的缩放 |  <res/a22_square_white.mp4>

您的浏览器不支持视频标签。

| 沿着  的剪切 |  <res/a12_square_white.mp4>

您的浏览器不支持视频标签。

| 沿着  的剪切 |  <res/a21_square_white.mp4>

您的浏览器不支持视频标签。

| 沿着  的平移 |  <res/a13_square_white.mp4>

您的浏览器不支持视频标签。

| 沿着  的平移 |  <res/a23_square_white.mp4>

您的浏览器不支持视频标签。

对于所有这些可视化，我们展示了一组 4 个点，这些点被分配到正方形的顶点上。最初，  被设置为恒等矩阵，  ，这样输入向量就没有变换或平移。从那里，  和  的每个元素都单独修改，结果变换可以在左侧看到。每个元素修改的量在矩阵表示中以数值形式显示，并在下方的小旋钮上显示。

希望这些可视化展示出，集合中的每个元素都是可以操作的旋钮，通过这些旋钮可以对输入向量的集合执行特定的变换。当然，一次移动多个旋钮是完全可能的，这就是为什么深入研究一个大家都喜欢的例子——旋转——是值得的。

### 旋转：一个特别的旁注

我必须坦白，当我最初学习如何使用线性变换进行旋转时，我并没有真正理解它是如何工作的。因此，我认为深入探讨这个话题很重要，希望为那些新手（以及可能那些经常使用旋转矩阵但并不完全理解它的人）提供一个直观的解释。

如果有人将上面显示的旋钮组合起来以产生旋转效果，他们可能会先沿着一个方向进行剪切，然后沿着另一个方向进行剪切，这将产生一个“伪旋转”效果。这无疑是朝着正确方向迈出的一步，但如果在修改剪切分量时其他分量保持为 1，那么这些点也会进一步远离原点。因此，在 x 轴和 y 轴上还需要进行额外的缩放。这将在下面的动画中展示：

<res/semi_rotate_white.mp4>

您的浏览器不支持视频标签。

在这里，我们可以看到（至少对于小于π的角度），旋转仅仅是朝相反方向进行剪切并相应地进行缩放。现在唯一的问题是，“我们如何知道需要剪切和缩放多少？”

好吧，答案并不特别令人惊讶。如果我们想旋转我们的点，我们可能已经在想象沿着某个角度θ的圆周进行旋转。我们知道单位矩阵应该对应于没有旋转的对象，即θ为 0。因此，我们知道有两个元素应该从 1 开始（注意：），而其他两个应该从 0 开始（注意：）。我们还知道剪切应该朝相反的方向进行，所以我们可以猜测旋转矩阵可能是这样的：

在这种情况下，我们想要剪切的量应该在θ为 0 时从 0 开始，然后在θ为π/2 时变为π。同时，缩放因子应该在θ为 0 时从 1 开始，然后在θ为π/2 时变为 0。

这看起来是正确的，但值得深入思考一下。如果缩放因子在θ为 0 时为 0，这当然意味着正方形的所有点也都在 0 处，对吧？毕竟，任何被缩放因子为 0 的都应该变成 0！但是，并不完全是这样。在这种情况下，

这意味着即使缩放分量是 0，剪切分量也是π。这可能会有些令人困惑，所以让我们用向量 v 乘以这两个矩阵：

在这里，我们可以看到，当乘以单位矩阵时，向量保持不变，但当乘以第二个矩阵时，x 和 y 分量会翻转。本质上，所有的向量大小都移动到了“剪切”分量中，而没有任何部分保留在“缩放”分量中。

我的观点是，尽管将我们的两个旋钮视为 x 轴和 y 轴上的缩放因子是有用的，但这并不一定描绘了整个画面，而且重要的是要考虑这些不同分量是如何共同工作的。

在继续展示矩阵应用于正方形的效果之前，考虑两个与之一致但仅修改了  或  分量的相关矩阵是值得的。

| 描述 | 变换 |
| --- | --- |

| 正弦波 |  <res/sines_white.mp4>

您的浏览器不支持视频标签。

| 仅余余弦 |  <res/cosines_white.mp4>

您的浏览器不支持视频标签。

在这里，我们看到两种完全不同的行为：

1.  在仅余正弦的情况下，我们看到当  从  绕过来时，正方形似乎像预期的那样增长并旋转，但在  时，它突然决定向相反方向移动。

1.  在仅余余弦的情况下，我们看到正方形在  完全翻转。

在观看下一个视频之前，稍微思考一下这两种不同的交互在实际中如何协同工作是很重要的。当你准备好了，请点击播放按钮：

<res/rotation_square_white.mp4>

您的浏览器不支持视频标签。

至少对我来说，思考为什么上面的两个动画组合在一起会产生旋转需要一些思考。在思考时，在  时正弦分量开始鼓励正方形缓慢地振荡回原始位置是有道理的，但同时也被同时变为负值的余弦分量拉向相反方向。这种“巧合”就是产生旋转效果的原因。

总体来说，旋转矩阵是线性变换的一个有趣且实用的应用，它真正帮助我理解了整个操作类如何被用来创建更复杂的运动。

### 仿射变换的保证

在这个阶段，我们已经从功能的角度讨论了仿射变换；然而，(正如通常一样)还有很多可以讨论的。这个特定的章节旨在为可能需要使用它们来解决各种应用的人提供对变换的直观感受，所以我犹豫着不深入探讨更严格的定义；然而，讨论仿射变换的某些特性，这些特性使它们适用于广泛的用途是很重要的。具体来说，仿射变换保持以下特性：

1.  **点之间的共线性**。这意味着在仿射变换之前位于同一直线上的任何点在变换后也必须位于同一直线上。直线仍然可以改变斜率或位置。

1.  **直线之间的平行性**。在变换前平行的任何直线在变换后也必须平行。

1.  **平行线段长度的比例**。这意味着如果你有两个不同的线段，其中一个由  和  参数化，而另一个由  和  参数化，那么在变换前后  必须相同。

1.  **任何变换形状的凸性**。如果一个形状在变换前没有凹面部分（即指向其中心的点），那么在变换后也不能有凹面部分。

1.  **点的集合的重心**。重心是系统的总质心，就像盘子的平衡点。本质上，重心两侧有相等数量的“物质”。这个位置在变换后必须相对于每个点保持在相同的位置。

再次强调，我们还有很多可以讨论的，但我感觉如果我们需要它们来讨论后续算法，我们将把更严格的讨论留到以后。相反，我相信继续讨论仿射变换的相对常见实现：增广矩阵形式是有用的。

## 增广矩阵实现

如前所述，仿射变换基本上是变换矩阵和平移的组合。对于二维输入向量，增广矩阵形式将这两个组合成一个大的变换矩阵。如果你像我一样，这可能会有些令人困惑。毕竟，如果二维向量由一个  数组描述，那么你怎么用一个  数组进行矩阵乘法呢？

坦白说，答案 *感觉* 有点儿蹩脚：我们只是在输入、输出和平移向量的末尾添加一个 1，使得：

所以，使用

我们将执行以下计算：

做这件事，我们发现，正如我们在前面的例子中所发现的那样。好的，现在我们需要讨论为什么这会起作用。

将 1 添加到二维向量的末尾实际上将它们转换成三维向量，其中  维简单地设置为 1。最容易可视化的方式是想象一个更大的立方体的顶面，所以这里是在那个立方体上之前相同的向量操作：

| 描述 | 变换 |
| --- | --- |

| 沿着缩放 |  <res/a11_cube_white.mp4>

您的浏览器不支持视频标签。  |

| 沿着缩放 |  <res/a22_cube_white.mp4>

您的浏览器不支持视频标签。  |

| 沿着剪切 |  <res/a12_cube_white.mp4>

您的浏览器不支持视频标签。  |

| 沿着剪切 |  <res/a21_cube_white.mp4>

您的浏览器不支持视频标签。  |

| 沿着平移 |  <res/a13_cube_white.mp4>

您的浏览器不支持视频标签。  |

| 沿着平移 |  <res/a23_cube_white.mp4>

您的浏览器不支持视频标签。  |

剪切和缩放操作看起来和之前差不多；然而，现在的平移操作现在明显是沿着整个立方体的剪切！这个操作作为二维平移的唯一原因是因为我们只关心通过立方体的切片在 。

现在，我之所以总觉得这种实现有点儿蹩脚，是因为有一个小小的魔法，每个人都保持沉默：矩阵的最后一行。在上述所有操作中，它简单地设置为  并且再也没有被触及过...但这非常令人不满意！

如果我们实际上移动那些旋钮并修改最后一行会发生什么？嗯...

| 描述 | 变换 |
| --- | --- |

| 沿着 x 和 y 方向进行剪切 |  <res/a31_cube_white.mp4>

您的浏览器不支持视频标签。  |

| 沿着 x 和 y 方向进行剪切 |  <res/a32_cube_white.mp4>

您的浏览器不支持视频标签。  |

| 沿着 z 方向进行缩放 |  <res/a33_cube_white.mp4>

您的浏览器不支持视频标签。  |

在这种情况下，前两个分量是沿着 x 和 y 方向的剪切，以及沿着 z 和 w 方向的剪切，而最后一个分量是沿着 z 方向的缩放。如果有人从上方拍照，这些变换都不会可见。因为我们高度专注于仿射变换的上下视图，所以这些操作在技术上不是仿射的；然而，它们仍然是线性的，并且展示立方体的所有可能的线性变换仍然很有意义。

最后，让我们回到旋转示例：

<res/rotation_cube_white.mp4>

您的浏览器不支持视频标签。

在这里，我们可以看到我们可以将任何仿射变换嵌入到三维空间中，并且仍然会看到与二维情况相同的结果。我认为这是一个很好的结尾：仿射变换是多维空间中的线性变换。

## 视频说明

这里有一个描述仿射变换的视频：

[`www.youtube-nocookie.com/embed/E3Phj6J287o`](https://www.youtube-nocookie.com/embed/E3Phj6J287o)

## 许可证

##### 代码示例

代码示例受 MIT 许可协议保护（可在[LICENSE.md](https://github.com/algorithm-archivists/algorithm-archive/blob/main/LICENSE.md)中找到）。

##### 文本

本章的文本由[James Schloss](https://github.com/leios)编写，并授权于[Creative Commons Attribution-ShareAlike 4.0 国际许可协议](https://creativecommons.org/licenses/by-sa/4.0/legalcode)。

[](https://creativecommons.org/licenses/by-sa/4.0/)

![](https://creativecommons.org/licenses/by-sa/4.0/)

##### 图像/图形

+   视频文件"A11 正方形"由[James Schloss](https://github.com/leios)创建，并授权于[Creative Commons Attribution-ShareAlike 4.0 国际许可协议](https://creativecommons.org/licenses/by-sa/4.0/legalcode)。

+   视频文件"A22 正方形"由[James Schloss](https://github.com/leios)创建，并授权于[Creative Commons Attribution-ShareAlike 4.0 国际许可协议](https://creativecommons.org/licenses/by-sa/4.0/legalcode)。

+   视频文件"A12 正方形"由[James Schloss](https://github.com/leios)创建，并授权于[Creative Commons Attribution-ShareAlike 4.0 国际许可协议](https://creativecommons.org/licenses/by-sa/4.0/legalcode)。

+   视频文件"A21 正方形"由[James Schloss](https://github.com/leios)创建，并授权于[Creative Commons Attribution-ShareAlike 4.0 国际许可协议](https://creativecommons.org/licenses/by-sa/4.0/legalcode)。

+   视频文件"A13 正方形"由[James Schloss](https://github.com/leios)制作，并授权于[Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/legalcode)。

+   视频文件"A23 正方形"由[James Schloss](https://github.com/leios)制作，并授权于[Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/legalcode)。

+   视频文件"半旋转"由[James Schloss](https://github.com/leios)制作，并授权于[Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/legalcode)。

+   视频文件"正弦波"由[James Schloss](https://github.com/leios)制作，并授权于[Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/legalcode)。

+   视频文件"余弦波"由[James Schloss](https://github.com/leios)制作，并授权于[Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/legalcode)。

+   视频文件"旋转正方形"由[James Schloss](https://github.com/leios)制作，并授权于[Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/legalcode)。

+   视频文件"A11 立方体"由[James Schloss](https://github.com/leios)制作，并授权于[Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/legalcode)。

+   视频文件"A22 立方体"由[James Schloss](https://github.com/leios)制作，并授权于[Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/legalcode)。

+   视频文件"A12 立方体"由[James Schloss](https://github.com/leios)制作，并授权于[Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/legalcode)。

+   视频文件"A21 立方体"由[James Schloss](https://github.com/leios)制作，并授权于[Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/legalcode)。

+   视频文件"A13 立方体"由[James Schloss](https://github.com/leios)制作，并授权于[Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/legalcode)。

+   视频文件"A23 立方体"由[James Schloss](https://github.com/leios)制作，并授权于[Creative Commons Attribution-ShareAlike 4.0 国际许可协议](https://creativecommons.org/licenses/by-sa/4.0/legalcode)。

+   视频文件"A31 立方体"由[James Schloss](https://github.com/leios)制作，并授权于[Creative Commons Attribution-ShareAlike 4.0 国际许可协议](https://creativecommons.org/licenses/by-sa/4.0/legalcode)。

+   视频文件"A32 立方体"由[James Schloss](https://github.com/leios)制作，并授权于[Creative Commons Attribution-ShareAlike 4.0 国际许可协议](https://creativecommons.org/licenses/by-sa/4.0/legalcode)。

+   视频文件"A33 立方体"由[James Schloss](https://github.com/leios)制作，并授权于[Creative Commons Attribution-ShareAlike 4.0 国际许可协议](https://creativecommons.org/licenses/by-sa/4.0/legalcode)。

+   视频文件"旋转立方体"由[James Schloss](https://github.com/leios)制作，并授权于[Creative Commons Attribution-ShareAlike 4.0 国际许可协议](https://creativecommons.org/licenses/by-sa/4.0/legalcode)。
