# 从 Python 到 Numpy

## 版权所有 (c) 2017 - 尼古拉斯·P·鲁吉耶 <Nicolas.Rougier@inria.fr>

![img/37ef03690ff06d35c7a6a4721a5bfdf6.png](img/37ef03690ff06d35c7a6a4721a5bfdf6.png) ![img/b9aad74680f5251444b0d4b4ffa1af9a.png](img/b9aad74680f5251444b0d4b4ffa1af9a.png) ![img/02dcd8f1f04a6181a76ede131dfccf71.png](img/02dcd8f1f04a6181a76ede131dfccf71.png) ![img/2cdafe9a0083e1ba44f35436fa336345.png](img/2cdafe9a0083e1ba44f35436fa336345.png)最新版本 - 2017 年 5 月 DOI: [10.5281/zenodo.225783](http://doi.org/10.5281/zenodo.225783)![img/247a52e4201671bc71b669dffc84bc84.png](img/247a52e4201671bc71b669dffc84bc84.png)

已经有很多关于 Numpy 的书籍（参见参考文献），一个合理的问题可能是质疑是否真的需要另一本书。正如你可能通过阅读这些行所猜测的，我个人的答案是肯定的，主要是因为我认为在从 Python 到 Numpy 的向量化迁移方面，还有不同的方法空间。有很多你在书中找不到的技术，而这些技术大多是通过经验学到的。本书的目标是解释这些技术中的一些，并在这个过程中提供一个体验的机会。

**网站:** [`www.labri.fr/perso/nrougier/from-python-to-numpy`](http://www.labri.fr/perso/nrougier/from-python-to-numpy)

**目录**

+   前言

    +   关于作者

    +   关于本书

    +   许可证

+   简介

    +   简单示例

    +   可读性与速度

+   数组的解剖结构

    +   简介

    +   内存布局

    +   视图和副本

    +   结论

+   代码向量化

    +   简介

    +   统一向量化

    +   时间向量化

    +   空间向量化

    +   结论

+   问题向量化

    +   简介

    +   路径查找

    +   流体动力学

    +   蓝色噪声采样

    +   结论

+   自定义向量化

    +   简介

    +   类型化列表

    +   内存感知数组

    +   结论

+   超越 Numpy

    +   返回 Python

    +   Numpy 及其相关

    +   Scipy 及其相关

    +   结论

+   结论

+   快速参考

    +   数据类型

    +   创建

    +   索引

    +   重塑

    +   广播

+   参考文献

    +   教程

    +   文章

    +   书籍

**免责声明：**所有外部图片都应附有相关信用。如果缺少信用，请告诉我，我会进行更正。同样，所有摘录都应注明来源（主要是维基百科）。如果没有注明来源，则视为错误，我会尽快更正。

# 前言

**目录**

+   关于作者

+   关于本书

    +   先决条件

    +   约定

    +   如何贡献

    +   出版

+   许可证

## 关于作者

[Nicolas P. Rougier](http://www.labri.fr/perso/nrougier/)是法国国家计算机科学和控制研究机构 Inria 的全职研究科学家。这是一个公共科学和技术机构（EPST），在研究与教育部的双重监督下。Nicolas P. Rougier 在[Mnemosyne](http://www.inria.fr/en/teams/mnemosyne)项目中工作，该项目位于整合神经科学和计算神经科学的交汇处，与[神经退行性疾病研究所](http://www.imn-bordeaux.org/en/)、波尔多计算机科学研究实验室([LaBRI](https://www.labri.fr/))、[波尔多大学](http://www.u-bordeaux.com/)和国家级科学研究中心([CNRS](http://www.cnrs.fr/index.php))合作。

他已经使用 Python 超过 15 年，使用 numpy 超过 10 年进行神经科学、机器学习和高级可视化（OpenGL）建模。Nicolas P. Rougier 是多个在线资源和教程（Matplotlib、numpy、OpenGL）的作者，并在波尔多大学以及世界各地的各种会议和学校（SciPy、EuroScipy 等）教授 Python、numpy 和科学可视化。他还是广受欢迎的文章《制作更好图形的十简单规则》（http://dx.doi.org/10.1371/journal.pcbi.1003833）和流行的[matplotlib 教程](http://www.labri.fr/perso/nrougier/teaching/matplotlib/matplotlib.html)的作者。

## 关于本书

本书是用[restructured text](http://docutils.sourceforge.net/rst.html)格式编写的，并使用来自[docutils](http://docutils.sourceforge.net/) Python 包的`rst2html.py`命令行生成。

如果你想重新构建 html 输出，从顶级目录开始，输入：

```py
$ rst2html.py --link-stylesheet --cloak-email-addresses \
                  --toc-top-backlinks --stylesheet=book.css \
                  --stylesheet-dirs=. book.rst book.html

```

源代码可以从[`github.com/rougier/from-python-to-numpy`](https://github.com/rougier/from-python-to-numpy)获取。

### 先决条件

这不是一本 Python 入门指南，你应该具备 Python 的中级水平，理想情况下 numpy 的入门水平。如果不是这样，请查看 bibliography 部分，以获取精选资源列表。

### 约定

我们将使用常规的命名约定。除非明确说明，每个脚本都应该导入 numpy、scipy 和 matplotlib，如下所示：

```py
import numpy as np
    import scipy as sp
    import matplotlib.pyplot as plt

```

我们将使用不同软件包的最新版本（截至写作日期，即 2017 年 1 月）：

| 软件包 | 版本 |
| --- | --- |
| Python | 3.6.0 |
| Numpy | 1.12.0 |
| Scipy | 0.18.1 |
| Matplotlib | 2.0.0 |

### 如何贡献

如果你想为这本书做出贡献，你可以：

+   审阅章节（请与我联系）

+   报告问题（[`github.com/rougier/from-python-to-numpy/issues`](https://github.com/rougier/from-python-to-numpy/issues)）

+   建议改进（[`github.com/rougier/from-python-to-numpy/pulls`](https://github.com/rougier/from-python-to-numpy/pulls)）

+   正确的英文（[`github.com/rougier/from-python-to-numpy/issues`](https://github.com/rougier/from-python-to-numpy/issues)）

+   为书籍设计一个更好、更响应式的 html 模板。

+   关注项目（[`github.com/rougier/from-python-to-numpy`](https://github.com/rougier/from-python-to-numpy)）

### 发布

如果你是一位对出版这本书感兴趣的编辑，如果你同意将这个版本和所有后续版本开放获取（即在此地址[`www.labri.fr/perso/nrougier/from-python-to-numpy`](http://www.labri.fr/perso/nrougier/from-python-to-numpy)上在线），你知道如何处理[restructured text](http://docutils.sourceforge.net/rst.html)（Word 不是选择），你提供真正的增值服务以及支持服务，更重要的是，你有一个真正惊人的 latex 书籍模板（并且警告：我对字体排印和设计有点挑剔：[Edward Tufte](https://www.edwardtufte.com/tufte/)是我的英雄）。还在这里吗？

## 许可

**书籍**

本作品根据[Creative Commons Attribution-Non Commercial-Share Alike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/)授权。你可以：

+   **分享** — 以任何媒体或格式复制和重新分发材料

+   **改编** — 混合、转换和在此基础上构建材料

只要您遵守许可条款，许可方不能撤销这些自由。

**代码**

代码根据 OSI 批准的 BSD 2-Clause 许可授权。

# 简介

**目录**

+   简单示例

+   可读性 vs 速度

## 简单示例

注意

你可以使用常规的 python shell 或从 IPython 会话或 Jupyter 笔记本中执行下面的任何代码。在这种情况下，你可能想使用魔法命令`%timeit`而不是我写的自定义命令。

Numpy 主要关于向量化。如果你熟悉 Python，这将是你面临的主要困难，因为你需要改变你的思维方式，你的新朋友（包括其他）被称为“向量”、“数组”、“视图”或“ufuncs”。

让我们用一个非常简单的例子来举例，随机游走。一种可能的对象式方法是定义一个`RandomWalker`类，并编写一个 walk 方法，该方法会在每次（随机）步骤后返回当前位置。这很好，可读性不错，但速度较慢：

**面向对象的方法**

```py
class RandomWalker:
        def __init__(self):
            self.position = 0

        def walk(self, n):
            self.position = 0
            for i in range(n):
                yield self.position
                self.position += 2*random.randint(0, 1) - 1

    walker = RandomWalker()
    walk = [position for position in walker.walk(1000)]

```

基准测试给我们：

```py
>>> from tools import timeit
    >>> walker = RandomWalker()
    >>> timeit("[position for position in walker.walk(n=10000)]", globals())
    10 loops, best of 3: 15.7 msec per loop

```

**过程式方法**

对于这样一个简单的问题，我们可能可以省略类定义，只关注 walk 方法，该方法计算每次随机步骤后的连续位置。

```py
def random_walk(n):
        position = 0
        walk = [position]
        for i in range(n):
            position += 2*random.randint(0, 1)-1
            walk.append(position)
        return walk

    walk = random_walk(1000)

```

这种新方法节省了一些 CPU 周期，但并不多，因为这个函数基本上与面向对象的方法相同，我们节省的几个周期可能来自 Python 内部的面向对象机制。

```py
>>> from tools import timeit
    >>> timeit("random_walk(n=10000)", globals())
    10 loops, best of 3: 15.6 msec per loop

```

**向量化方法**

但我们可以使用提供创建迭代器以实现高效循环的[itertools](https://docs.python.org/3.6/library/itertools.html) Python 模块做得更好。如果我们观察到随机游走是步骤的累积，我们可以通过首先生成所有步骤并累积它们而不使用任何循环来重写函数：

```py
def random_walk_faster(n=1000):
        from itertools import accumulate
        # Only available from Python 3.6
        steps = random.choices([-1,+1], k=n)
        return [0]+list(accumulate(steps))

     walk = random_walk_faster(1000)

```

事实上，我们只是将函数向量化了。我们不再通过循环来选择连续步骤并将它们添加到当前位置，而是首先一次性生成所有步骤，并使用[accumulate](https://docs.python.org/3.6/library/itertools.html#itertools.accumulate)函数来计算所有位置。我们消除了循环，这使得事情变得更快：

```py
>>> from tools import timeit
    >>> timeit("random_walk_faster(n=10000)", globals())
    10 loops, best of 3: 2.21 msec per loop

```

与上一个版本相比，我们获得了 85%的计算时间，这并不坏。但这个新版本的优势在于它使 numpy 向量化变得超级简单。我们只需将 itertools 调用转换为 numpy 调用即可。

```py
def random_walk_fastest(n=1000):
        # No 's' in numpy choice (Python offers choice & choices)
        steps = np.random.choice([-1,+1], n)
        return np.cumsum(steps)

    walk = random_walk_fastest(1000)

```

不太难，但我们使用 numpy 获得了 500 倍的提升：

```py
>>> from tools import timeit
    >>> timeit("random_walk_fastest(n=10000)", globals())
    1000 loops, best of 3: 14 usec per loop

```

这本书是关于向量化，无论是代码层面还是问题层面。在查看自定义向量化之前，我们会看到这种差异很重要。

## 可读性 vs 速度

在进入下一章之前，我想提醒你，一旦你熟悉了 numpy，你可能会遇到一个潜在的问题。这是一个非常强大的库，你可以用它做出奇迹，但大多数情况下，这要以可读性为代价。如果你在编写代码时没有注释，几周（或可能是几天）后你将无法知道一个函数在做什么。例如，你能说出下面两个函数在做什么吗？你可能可以猜出第一个，但第二个可能不太可能（或者你的名字是[Jaime Fernández del Río](http://stackoverflow.com/questions/7100242/python-numpy-first-occurrence-of-subarray)并且你不需要读这本书）。

```py
def function_1(seq, sub):
        return [i for i in range(len(seq) - len(sub)) if seq[i:i+len(sub)] == sub]

    def function_2(seq, sub):
        target = np.dot(sub, sub)
        candidates = np.where(np.correlate(seq, sub, mode='valid') == target)[0]
        check = candidates[:, np.newaxis] + np.arange(len(sub))
        mask = np.all((np.take(seq, check) == sub), axis=-1)
        return candidates[mask]

```

如你所猜，第二个函数是第一个函数的向量化-优化-更快-numpy 版本。它比纯 Python 版本快 10 倍，但几乎不可读。

# 数组解剖

**内容**

+   介绍

+   内存布局

+   视图和副本

    +   直接和间接访问

    +   临时副本

+   结论

## 介绍

如 前言 中所述，你应该对 numpy 有基本的了解才能阅读这本书。如果不是这样，你最好先从入门教程开始，然后再回来。因此，我这里只简要回顾一下 numpy 数组的基本结构，特别是关于内存布局、视图、副本和数据类型。如果你想让你的计算从 numpy 哲学中受益，这些是关键的概念。

让我们考虑一个简单的例子，我们想要清除具有 `np.float32` 数据类型的数组中的所有值。如何编写它以最大化速度？下面的语法相当明显（至少对于熟悉 numpy 的人来说是这样），但上述问题要求找到最快的操作。

```py
>>> Z = np.ones(4*1000000, np.float32)
    >>> Z[...] = 0

```

如果你更仔细地观察数组的 dtype 和大小，你可以观察到这个数组可以被转换为（即视为）许多其他“兼容”的数据类型。这里的兼容是指 `Z.size * Z.itemsize` 可以被新的 dtype 元素大小整除。

```py
>>> timeit("Z.view(np.float16)[...] = 0", globals())
    100 loops, best of 3: 2.72 msec per loop
    >>> timeit("Z.view(np.int16)[...] = 0", globals())
    100 loops, best of 3: 2.77 msec per loop
    >>> timeit("Z.view(np.int32)[...] = 0", globals())
    100 loops, best of 3: 1.29 msec per loop
    >>> timeit("Z.view(np.float32)[...] = 0", globals())
    100 loops, best of 3: 1.33 msec per loop
    >>> timeit("Z.view(np.int64)[...] = 0", globals())
    100 loops, best of 3: 874 usec per loop
    >>> timeit("Z.view(np.float64)[...] = 0", globals())
    100 loops, best of 3: 865 usec per loop
    >>> timeit("Z.view(np.complex128)[...] = 0", globals())
    100 loops, best of 3: 841 usec per loop
    >>> timeit("Z.view(np.int8)[...] = 0", globals())
    100 loops, best of 3: 630 usec per loop

```

有趣的是，清除所有值的明显方法并不是最快的。通过将数组转换为更大的数据类型，例如 `np.float64`，我们获得了 25% 的速度提升。但是，通过将数组视为字节数组（`np.int8`），我们获得了 50% 的速度提升。这种加速的原因可以在 numpy 的内部机制和编译器优化中找到。这个简单的例子说明了 numpy 的哲学，我们将在下一节中看到。

## 内存布局

[numpy 文档](https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html) 对 `ndarray` 类定义得非常清楚：

> *`ndarray` 类的实例由一个连续的一维内存段（由数组或某些其他对象拥有）组成，并结合一个索引方案，该方案将 N 个整数映射到块中项的位置。*

换句话说，数组主要是一个连续的内存块，其部分可以通过索引方案访问。这种索引方案反过来由一个 [形状](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html#numpy.ndarray.shape) 和一个 [数据类型](https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html) 定义，这正是当你定义一个新数组时所需要的：

```py
Z = np.arange(9).reshape(3,3).astype(np.int16)

```

这里，我们知道 Z 的元素大小是 2 字节（`int16`），形状是 (3,3)，维度数是 2（`len(Z.shape)`）。

```py
>>> print(Z.itemsize)
    2 >>> print(Z.shape)
    (3, 3) >>> print(Z.ndim)
    2

```

此外，由于 Z 不是一个视图，我们可以推断出数组的步长（[strides](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.strides.html#numpy.ndarray.strides)），这些步长定义了在遍历数组时每个维度需要步进的字节数。

```py
>>> strides = Z.shape[1]*Z.itemsize, Z.itemsize
    >>> print(strides)
    (6, 2) >>> print(Z.strides)
    (6, 2)

```

在所有这些信息的基础上，我们知道如何访问一个特定的项（由索引元组设计），以及更精确地，如何计算起始和结束偏移量：

```py
offset_start = 0
    for i in range(ndim):
        offset_start += strides[i]*index[i]
    offset_end = offset_start + Z.itemsize

```

让我们使用[tobytes](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.tobytes.html)转换方法来验证这是否正确：

```py
>>> Z = np.arange(9).reshape(3,3).astype(np.int16)
    >>> index = 1,1
    >>> print(Z[index].tobytes())
    b'\x04\x00'
    >>> offset = 0
    >>> for i in range(Z.ndim):
    ...     offset + = Z.strides[i]*index[i]
    >>> print(Z.tobytes()[offset_start:offset_end]
    b'\x04\x00'

```

这个数组可以从不同的角度（即布局）来考虑：

**项目布局**

```py
               shape[1]
                     (=3)
                ┌───────────┐

             ┌  ┌───┬───┬───┐  ┐
             │  │ 0 │ 1 │ 2 │  │
             │  ├───┼───┼───┤  │
    shape[0] │  │ 3 │ 4 │ 5 │  │ len(Z)
     (=3)    │  ├───┼───┼───┤  │  (=3)
             │  │ 6 │ 7 │ 8 │  │
             └  └───┴───┴───┘  ┘

```

**扁平化项目布局**

```py
┌───┬───┬───┬───┬───┬───┬───┬───┬───┐
    │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │
    └───┴───┴───┴───┴───┴───┴───┴───┴───┘

    └───────────────────────────────────┘
                   Z.size
                    (=9)

```

**内存布局（C 顺序，大端序）**

```py
                         strides[1]
                               (=2)
                      ┌─────────────────────┐

              ┌       ┌──────────┬──────────┐ ┐
              │ p+00: │ 00000000 │ 00000000 │ │
              │       ├──────────┼──────────┤ │
              │ p+02: │ 00000000 │ 00000001 │ │ strides[0]
              │       ├──────────┼──────────┤ │   (=2x3)
              │ p+04  │ 00000000 │ 00000010 │ │
              │       ├──────────┼──────────┤ ┘
              │ p+06  │ 00000000 │ 00000011 │
              │       ├──────────┼──────────┤
    Z.nbytes  │ p+08: │ 00000000 │ 00000100 │
    (=3x3x2)  │       ├──────────┼──────────┤
              │ p+10: │ 00000000 │ 00000101 │
              │       ├──────────┼──────────┤
              │ p+12: │ 00000000 │ 00000110 │
              │       ├──────────┼──────────┤
              │ p+14: │ 00000000 │ 00000111 │
              │       ├──────────┼──────────┤
              │ p+16: │ 00000000 │ 00001000 │
              └       └──────────┴──────────┘

                      └─────────────────────┘
                            Z.itemsize
                         Z.dtype.itemsize
                               (=2)

```

如果我们现在对`Z`进行切片，结果是基础数组`Z`的视图：

```py
V = Z[::2,::2]

```

这种视图是通过形状、dtype **和** 步长来指定的，因为步长不能再从 dtype 和形状中推导出来：

**项目布局**

```py
               shape[1]
                     (=2)
                ┌───────────┐

             ┌  ┌───┬╌╌╌┬───┐  ┐
             │  │ 0 │   │ 2 │  │            ┌───┬───┐
             │  ├───┼╌╌╌┼───┤  │            │ 0 │ 2 │
    shape[0] │  ╎   ╎   ╎   ╎  │ len(Z)  →  ├───┼───┤
     (=2)    │  ├───┼╌╌╌┼───┤  │  (=2)      │ 6 │ 8 │
             │  │ 6 │   │ 8 │  │            └───┴───┘
             └  └───┴╌╌╌┴───┘  ┘

```

**扁平化项目布局**

```py
┌───┬╌╌╌┬───┬╌╌╌┬╌╌╌┬╌╌╌┬───┬╌╌╌┬───┐       ┌───┬───┬───┬───┐
    │ 0 │   │ 2 │   ╎   ╎   │ 6 │   │ 8 │   →   │ 0 │ 2 │ 6 │ 8 │
    └───┴╌╌╌┴───┴╌╌╌┴╌╌╌┴╌╌╌┴───┴╌╌╌┴───┘       └───┴───┴───┴───┘
    └─┬─┘   └─┬─┘           └─┬─┘   └─┬─┘
      └───┬───┘               └───┬───┘
          └───────────┬───────────┘
                   Z.size
                    (=4)

```

**内存布局（C 顺序，大端序）**

```py
              ┌        ┌──────────┬──────────┐ ┐             ┐
                ┌─┤  p+00: │ 00000000 │ 00000000 │ │             │
                │ └        ├──────────┼──────────┤ │ strides[1]  │
              ┌─┤    p+02: │          │          │ │   (=4)      │
              │ │ ┌        ├──────────┼──────────┤ ┘             │
              │ └─┤  p+04  │ 00000000 │ 00000010 │               │
              │   └        ├──────────┼──────────┤               │ strides[0]
              │      p+06: │          │          │               │   (=12)
              │            ├──────────┼──────────┤               │
    Z.nbytes ─┤      p+08: │          │          │               │
      (=8)    │            ├──────────┼──────────┤               │
              │      p+10: │          │          │               │
              │   ┌        ├──────────┼──────────┤               ┘
              │ ┌─┤  p+12: │ 00000000 │ 00000110 │
              │ │ └        ├──────────┼──────────┤
              └─┤    p+14: │          │          │
                │ ┌        ├──────────┼──────────┤
                └─┤  p+16: │ 00000000 │ 00001000 │
                  └        └──────────┴──────────┘

                           └─────────────────────┘
                                 Z.itemsize
                              Z.dtype.itemsize
                                    (=2)

```

## 视图和副本

视图和副本是优化您数值计算的重要概念。即使我们在前面的部分已经操作过它们，整个故事还是要复杂一些。

### 直接和间接访问

首先，我们必须区分[索引](https://docs.scipy.org/doc/numpy/user/basics.indexing.html#)和[复杂索引](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing)。第一个总是返回一个视图，而第二个将返回一个副本。这种差异很重要，因为在第一种情况下，修改视图会修改基础数组，而在第二种情况下则不是这样：

```py
>>> Z = np.zeros(9)
    >>> Z_view = Z[:3]
    >>> Z_view[...] = 1
    >>> print(Z)
    [ 1\.  1\.  1\.  0\.  0\.  0\.  0\.  0\.  0.] >>> Z = np.zeros(9)
    >>> Z_copy = Z[[0,1,2]]
    >>> Z_copy[...] = 1
    >>> print(Z)
    [ 0\.  0\.  0\.  0\.  0\.  0\.  0\.  0\.  0.]

```

因此，如果您需要复杂的索引，最好保留您复杂索引的副本（尤其是如果计算它很复杂的话），并且使用它：

```py
>>> Z = np.zeros(9)
    >>> index = [0,1,2]
    >>> Z[index] = 1
    >>> print(Z)
    [ 1\.  1\.  1\.  0\.  0\.  0\.  0\.  0\.  0.]

```

如果您不确定索引的结果是视图还是副本，您可以检查结果的`base`是什么。如果是`None`，则结果是一个副本：

```py
>>> Z = np.random.uniform(0,1,(5,5))
    >>> Z1 = Z[:3,:]
    >>> Z2 = Z[[0,1,2], :]
    >>> print(np.allclose(Z1,Z2))
    True >>> print(Z1.base is Z)
    True >>> print(Z2.base is Z)
    False >>> print(Z2.base is None)
    True

```

注意，一些 numpy 函数在可能的情况下会返回一个视图（例如[ravel](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ravel.html)），而另一些则总是返回一个副本（例如[flatten](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flatten.html#numpy.ndarray.flatten)）：

```py
>>> Z = np.zeros((5,5))
    >>> Z.ravel().base is Z
    True >>> Z[::2,::2].ravel().base is Z
    False >>> Z.flatten().base is Z
    False

```

### 临时副本

可以像上一节那样显式地创建副本，但最通用的情况是中间副本的隐式创建。这是您在数组上进行某些算术运算时的情况：

```py
>>> X = np.ones(10, dtype=np.int)
    >>> Y = np.ones(10, dtype=np.int)
    >>> A = 2*X + 2*Y

```

在上面的例子中，创建了三个中间数组。一个用于存储`2*X`的结果，一个用于存储`2*Y`的结果，最后一个用于存储`2*X+2*Y`的结果。在这种情况下，数组足够小，这并不会真正造成差异。然而，如果您的数组很大，那么您必须小心处理这样的表达式，并考虑是否可以以不同的方式完成。例如，如果只有最终结果重要，您之后不需要`X`或`Y`，那么另一种解决方案将是：

```py
>>> X = np.ones(10, dtype=np.int)
    >>> Y = np.ones(10, dtype=np.int)
    >>> np.multiply(X, 2, out=X)
    >>> np.multiply(Y, 2, out=Y)
    >>> np.add(X, Y, out=X)

```

使用这种替代方案，没有创建临时数组。问题是还有许多其他情况需要创建这样的副本，这会影响性能，如下面的例子所示：

```py
>>> X = np.ones(1000000000, dtype=np.int)
    >>> Y = np.ones(1000000000, dtype=np.int)
    >>> timeit("X = X + 2.0*Y", globals())
    100 loops, best of 3: 3.61 ms per loop >>> timeit("X = X + 2*Y", globals())
    100 loops, best of 3: 3.47 ms per loop >>> timeit("X += 2*Y", globals())
    100 loops, best of 3: 2.79 ms per loop >>> timeit("np.add(X, Y, out=X); np.add(X, Y, out=X)", globals())
    1000 loops, best of 3: 1.57 ms per loop

```

## 结论

作为结论，我们将进行一个练习。给定两个向量`Z1`和`Z2`。我们想知道`Z2`是否是`Z1`的视图，如果是，这个视图是什么？

```py
>>> Z1 = np.arange(10)
    >>> Z2 = Z1[1:-1:2]

```

```py
   ╌╌╌┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬╌╌
    Z1    │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │
       ╌╌╌┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴╌╌
       ╌╌╌╌╌╌╌┬───┬╌╌╌┬───┬╌╌╌┬───┬╌╌╌┬───┬╌╌╌╌╌╌╌╌╌╌
    Z2        │ 1 │   │ 3 │   │ 5 │   │ 7 │
       ╌╌╌╌╌╌╌┴───┴╌╌╌┴───┴╌╌╌┴───┴╌╌╌┴───┴╌╌╌╌╌╌╌╌╌╌

```

首先，我们需要检查`Z1`是否是`Z2`的基

```py
>>> print(Z2.base is Z1)
    True

```

在这一点上，我们知道`Z2`是`Z1`的视图，这意味着`Z2`可以表示为`Z1[start:stop:step]`。困难之处在于找到`start`、`stop`和`step`。对于`step`，我们可以使用任何数组的`strides`属性，该属性给出每个维度中从一个元素到另一个元素的字节数。在我们的情况下，由于两个数组都是一维的，我们可以直接比较第一个步长：

```py
>>> step = Z2.strides[0] // Z1.strides[0]
    >>> print(step)
    2

```

下一个难度是找到`start`和`stop`索引。为此，我们可以利用`byte_bounds`方法，该方法返回数组的端点指针。

```py
  byte_bounds(Z1)[0]                  byte_bounds(Z1)[-1]
              ↓                                   ↓
       ╌╌╌┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬╌╌
    Z1    │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │
       ╌╌╌┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴╌╌

          byte_bounds(Z2)[0]      byte_bounds(Z2)[-1]
                  ↓                       ↓
       ╌╌╌╌╌╌╌┬───┬╌╌╌┬───┬╌╌╌┬───┬╌╌╌┬───┬╌╌╌╌╌╌╌╌╌╌
    Z2        │ 1 │   │ 3 │   │ 5 │   │ 7 │
       ╌╌╌╌╌╌╌┴───┴╌╌╌┴───┴╌╌╌┴───┴╌╌╌┴───┴╌╌╌╌╌╌╌╌╌╌

```

```py
>>> offset_start = np.byte_bounds(Z2)[0] - np.byte_bounds(Z1)[0]
    >>> print(offset_start) # bytes
    8

    >>> offset_stop = np.byte_bounds(Z2)[-1] - np.byte_bounds(Z1)[-1]
    >>> print(offset_stop) # bytes
    -16

```

使用`itemsize`将这些偏移量转换为索引是直接的，同时考虑到`offset_stop`是负数（`Z2`的逻辑结束边界小于`Z1`数组的结束边界）。因此，我们需要将 Z1 的元素大小加到正确的结束索引上。

```py
>>> start = offset_start // Z1.itemsize
    >>> stop = Z1.size + offset_stop // Z1.itemsize
    >>> print(start, stop, step)
    1, 8, 2

```

最后我们测试我们的结果：

```py
>>> print(np.allclose(Z1[start:stop:step], Z2))
    True

```

作为练习，你可以通过考虑以下内容来改进这个最初且非常简单的实现：

+   负步长

+   多维数组

练习的解决方案。

# 代码向量化

**目录**

+   介绍

+   统一向量化

    +   生命游戏

    +   Python 实现

    +   NumPy 实现

    +   练习

    +   来源

    +   参考文献

+   时间向量化

    +   Python 实现

    +   NumPy 实现

    +   更快的 NumPy 实现

    +   可视化

    +   练习

    +   来源

    +   参考文献

+   空间向量化

    +   鸟群

    +   Python 实现

    +   NumPy 实现

    +   练习

    +   来源

    +   参考文献

+   结论

## 介绍

代码向量化意味着你试图解决的问题本质上是可向量的，并且只需要几个 NumPy 技巧来使其更快。当然，这并不意味着它容易或直接，但至少它不需要完全重新思考你的问题（正如它将在问题向量化章节中那样）。然而，这可能需要一些经验来看到代码可以向量化的地方。让我们通过一个简单的例子来说明这一点，我们想要对两个整数列表求和。一种简单的方法是使用纯 Python：

```py
def add_python(Z1,Z2):
        return [z1+z2 for (z1,z2) in zip(Z1,Z2)]

```

这个最初的天真解决方案可以非常容易地使用 NumPy 向量化：

```py
def add_numpy(Z1,Z2):
        return np.add(Z1,Z2)

```

毫不意外，对两种方法进行基准测试表明第二种方法快了一个数量级。

```py
>>> Z1 = random.sample(range(1000), 100)
    >>> Z2 = random.sample(range(1000), 100)
    >>> timeit("add_python(Z1, Z2)", globals())
    1000 loops, best of 3: 68 usec per loop
    >>> timeit("add_numpy(Z1, Z2)", globals())
    10000 loops, best of 3: 1.14 usec per loop

```

不仅第二种方法更快，而且它还自然地适应了 `Z1` 和 `Z2` 的形状。这就是我们没有写 `Z1 + Z2` 的原因，因为如果 `Z1` 和 `Z2` 都是列表，那么它将不起作用。在第一个 Python 方法中，内层的 `+` 根据两个对象的性质被解释为不同，因此如果我们考虑两个嵌套列表，我们会得到以下输出：

```py
>>> Z1 = [[1, 2], [3, 4]]
    >>> Z2 = [[5, 6], [7, 8]]
    >>> Z1 + Z2
    [[1, 2], [3, 4], [5, 6], [7, 8]] >>> add_python(Z1, Z2)
    [[1, 2, 5, 6], [3, 4, 7, 8]] >>> add_numpy(Z1, Z2)
    [[ 6  8]
     [10 12]]

```

第一种方法是将两个列表连接在一起，第二种方法是将内部列表连接在一起，最后一种方法计算的是（数值上）预期的结果。作为一个练习，你可以重写 Python 版本，使其能够接受任何深度的嵌套列表。

## 均匀向量化

均匀向量化是向量化最简单的形式，其中所有元素在每个时间步骤中执行相同的计算，没有针对任何特定元素的处理。一个典型的例子是由约翰·康威（见下文）发明的生命游戏，它是细胞自动机最早的例子之一。那些细胞自动机可以方便地被视为由细胞组成的数组，这些细胞通过邻居的概念连接在一起，它们的向量化是直接的。让我首先定义这个游戏，然后我们将看到如何向量化它。

**图 4.1**

织锦螺在其壳上表现出细胞自动机模式。图片由 [Richard Ling](https://commons.wikimedia.org/wiki/File:Textile_cone.JPG)，2005 年拍摄。

![img/c091c3d73318960a6f67f0411911ec50.png](img/c091c3d73318960a6f67f0411911ec50.png)

### 生命游戏

注

来自维基百科关于 [生命游戏](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) 条目的摘录

生命游戏是由英国数学家约翰·霍顿·康威在 1970 年设计的细胞自动机。它是细胞自动机最著名的例子。这个“游戏”实际上是一个零玩家游戏，意味着其演变由其初始状态决定，不需要来自人类玩家的输入。玩家通过与创建初始配置并观察其演变来与生命游戏互动。

生命游戏的宇宙是一个无限的两维正交网格，由正方形细胞组成，每个细胞处于两种可能的状态之一，即活着或死亡。每个细胞与其八个邻居相互作用，这些邻居是直接水平、垂直或对角相邻的细胞。在每一个时间步骤中，以下转换发生：

1.  任何拥有少于两个活邻居的活细胞都会死亡，就像由于人口不足而导致的需要一样。

1.  任何拥有超过三个活邻居的活细胞都会死亡，就像由于过度拥挤一样。

1.  任何拥有两个或三个活邻居的活细胞在下一代中保持不变。

1.  任何拥有恰好三个活邻居的死亡细胞变成活细胞。

初始模式构成了系统的“种子”。第一代是通过将上述规则同时应用于种子中的每个细胞来创建的——出生和死亡是同时发生的，这个离散的时刻有时被称为“滴答”。（换句话说，每一代都是前一代的纯函数。）规则会反复应用以创建更多代。

### Python 实现

注意

我们本可以使用更高效的 python [数组接口](http://docs.python.org/3/library/array.html)，但使用熟悉的列表对象更方便。

在纯 Python 中，我们可以使用表示棋盘的列表的列表来编码生命游戏，其中细胞将进化。这样的棋盘将配备一个 0 的边界，这可以通过避免在计算邻居数量时对边界进行特定测试来加速事情。

```py
Z = [[0,0,0,0,0,0],
         [0,0,0,1,0,0],
         [0,1,0,1,0,0],
         [0,0,1,1,0,0],
         [0,0,0,0,0,0],
         [0,0,0,0,0,0]]

```

考虑到边界，然后计算邻居就很简单了：

```py
def compute_neighbours(Z):
        shape = len(Z), len(Z[0])
        N  = [[0,]*(shape[0]) for i in range(shape[1])]
        for x in range(1,shape[0]-1):
            for y in range(1,shape[1]-1):
                N[x][y] = Z[x-1][y-1]+Z[x][y-1]+Z[x+1][y-1] \
                        + Z[x-1][y]            +Z[x+1][y]   \
                        + Z[x-1][y+1]+Z[x][y+1]+Z[x+1][y+1]
        return N

```

要迭代一步时间，我们只需计算每个内部细胞的邻居数量，然后根据上述四个规则更新整个棋盘：

```py
def iterate(Z):
        N = compute_neighbours(Z)
        for x in range(1,shape[0]-1):
            for y in range(1,shape[1]-1):
                 if Z[x][y] == 1 and (N[x][y] < 2 or N[x][y] > 3):
                     Z[x][y] = 0
                 elif Z[x][y] == 0 and N[x][y] == 3:
                     Z[x][y] = 1
        return Z

```

下图展示了在 4x4 区域内对初始状态为[滑翔机](https://en.wikipedia.org/wiki/Glider_(Conway%27s_Life))的四个迭代过程，该结构由理查德·K·盖于 1970 年发现。

**图 4.2**

已知滑翔机模式在 4 次迭代中会沿对角线复制自己一步。

![img/d3d45b9f3ee5417c34c6fd84b90f9af4.png](img/d3d45b9f3ee5417c34c6fd84b90f9af4.png)

### Numpy 实现

从 Python 版本开始，生命游戏的向量化需要两部分，一部分负责计算邻居，另一部分负责执行规则。如果记住我们已经在竞技场周围添加了一个空边界，邻居计数相对简单。通过考虑竞技场的部分视图，我们实际上可以直观地访问邻居，如下面的一维情况所示：

```py
               ┏━━━┳━━━┳━━━┓───┬───┐
            Z[:-2] ┃ 0 ┃ 1 ┃ 1 ┃ 1 │ 0 │ (left neighbours)
                   ┗━━━┻━━━┻━━━┛───┴───┘
                         ↓︎
               ┌───┏━━━┳━━━┳━━━┓───┐
       Z[1:-1] │ 0 ┃ 1 ┃ 1 ┃ 1 ┃ 0 │ (actual cells)
               └───┗━━━┻━━━┻━━━┛───┘
                         ↑
           ┌───┬───┏━━━┳━━━┳━━━┓
    Z[+2:] │ 0 │ 1 ┃ 1 ┃ 1 ┃ 0 ┃ (right neighbours)
           └───┴───┗━━━┻━━━┻━━━┛

```

转到二维情况只需要一点算术，以确保考虑所有八个邻居。

```py
N = np.zeros(Z.shape, dtype=int)
    N[1:-1,1:-1] += (Z[ :-2, :-2] + Z[ :-2,1:-1] + Z[ :-2,2:] +
                     Z[1:-1, :-2]                + Z[1:-1,2:] +
                     Z[2:  , :-2] + Z[2:  ,1:-1] + Z[2:  ,2:])

```

对于规则执行，我们可以编写一个使用 numpy 的[argwhere](http://docs.scipy.org/doc/numpy/reference/generated/numpy.argwhere.html)方法的第一个版本，这将给我们给定条件为真的索引。

```py
# Flatten arrays
    N_ = N.ravel()
    Z_ = Z.ravel()

    # Apply rules
    R1 = np.argwhere( (Z_==1) & (N_ < 2) )
    R2 = np.argwhere( (Z_==1) & (N_ > 3) )
    R3 = np.argwhere( (Z_==1) & ((N_==2) | (N_==3)) )
    R4 = np.argwhere( (Z_==0) & (N_==3) )

    # Set new values
    Z_[R1] = 0
    Z_[R2] = 0
    Z_[R3] = Z_[R3]
    Z_[R4] = 1

    # Make sure borders stay null
    Z[0,:] = Z[-1,:] = Z[:,0] = Z[:,-1] = 0

```

即使这个第一个版本没有使用嵌套循环，但由于使用了四个`argwhere`调用，它远非最优，因为这些调用可能相当慢。我们可以将规则分解为将存活的（保持为 1）和将出生的细胞，为此，我们可以利用 Numpy 的布尔能力，并自然地编写：

注意

我们没有写`Z = 0`，因为这会将值 0 赋给`Z`，然后它将变成一个简单的标量。

```py
birth = (N==3)[1:-1,1:-1] & (Z[1:-1,1:-1]==0)
    survive = ((N==2) | (N==3))[1:-1,1:-1] & (Z[1:-1,1:-1]==1)
    Z[...] = 0
    Z[1:-1,1:-1][birth | survive] = 1

```

如果你查看`出生`和`存活`行，你会看到这两个变量是数组，可以在清除后将其`Z`值设置为 1。

**图 4.3**

生命游戏。灰度表示细胞过去活跃的程度。

[从 Python 到 NumPy 的数据：生命游戏](https://www.labri.fr/perso/nrougier/from-python-to-numpy/data/game-of-life.mp4)

您的浏览器不支持视频标签。

### 练习

化学物种的反应和扩散可以产生各种图案，这些图案常在自然界中看到。灰色-斯科特方程式模拟了这样的反应。有关这个化学系统的更多信息，请参阅文章《简单系统中的复杂模式》（约翰·E·皮尔森，科学，第 261 卷，1993 年）。让我们考虑两种化学物种 *U* 和 *V*，它们分别具有浓度 *u* 和 *v* 以及扩散率 *Du* 和 *Dv*。*V* 以转换率 *k* 转换为 *P*。*f* 代表喂养 *U* 并从 *U*、*V* 和 *P* 中排出的过程的速率。这可以写成：

| 化学反应 | 方程式 |
| --- | --- |
| *U*＋2*V*→3*V* | *u̇*＝*Du*∇²*u*－*uv*²＋*f*(1－*u*) |
| *V*→*P* | *v̇*＝*Dv*∇²*v*＋*uv*²－(*f*＋*k*)*v* |

基于生命游戏的示例，尝试实现这样的反应-扩散系统。以下是一组有趣的参数以供测试：

| 名称 | 杜 | 杜 v | f | k |
| --- | --- | --- | --- | --- |
| 细菌 1 | 0.16 | 0.08 | 0.035 | 0.065 |
| 细菌 2 | 0.14 | 0.06 | 0.035 | 0.065 |
| 珊瑚 | 0.16 | 0.08 | 0.060 | 0.062 |
| 指纹 | 0.19 | 0.05 | 0.060 | 0.062 |
| 螺旋 | 0.10 | 0.10 | 0.018 | 0.050 |
| 密集螺旋 | 0.12 | 0.08 | 0.020 | 0.050 |
| 螺旋快速 | 0.10 | 0.16 | 0.020 | 0.050 |
| 不稳定 | 0.16 | 0.08 | 0.020 | 0.055 |
| 蠕虫 1 | 0.16 | 0.08 | 0.050 | 0.065 |
| 蠕虫 2 | 0.16 | 0.08 | 0.054 | 0.063 |
| 斑马鱼 | 0.16 | 0.08 | 0.035 | 0.060 |

下图显示了该模型在特定参数集下的动画。

**图 4.4**

反应-扩散灰色-斯科特模型。从左到右，*细菌 1*、*珊瑚*和*密集螺旋*。

[从 Python 到 NumPy 的数据：灰色-斯科特模型 1](https://www.labri.fr/perso/nrougier/from-python-to-numpy/data/gray-scott-1.mp4)

您的浏览器不支持视频标签。

[从 Python 到 NumPy 的数据：灰色-斯科特模型](https://www.labri.fr/perso/nrougier/from-python-to-numpy/data/gray-scott-2.mp4)

您的浏览器不支持视频标签。

[从 Python 到 NumPy 的数据：灰色-斯科特模型 3](https://www.labri.fr/perso/nrougier/from-python-to-numpy/data/gray-scott-3.mp4)

您的浏览器不支持视频标签。

### 来源

+   game_of_life_python.py

+   game_of_life_numpy.py

+   gray_scott.py（练习的解决方案）

### 参考文献

+   [约翰·康威的新单人纸牌游戏 "生命"](https://web.archive.org/web/20090603015231/http://ddi.cs.uni-potsdam.de/HyFISCH/Produzieren/lis_projekt/proj_gamelife/ConwayScientificAmerican.htm), 马丁·加德纳, 科学美国人 223, 1970.

+   [反应-扩散的灰色-斯科特模型](http://groups.csail.mit.edu/mac/projects/amorphous/GrayScott/), 阿贝尔森，亚当斯，库尔，汉森，纳加帕尔，苏斯曼，1997.

+   [通过灰色-斯科特模型进行反应-扩散](http://mrob.com/pub/comp/xmorphia/), 罗伯特·P·穆纳福，1996.

## 时间向量化

曼德布罗特集是复数 *c* 的集合，对于这些复数，函数 *f**c* = *z*² + *c* 在从 *z* = 0 迭代时不会发散，即对于序列 *f**c*，*f**c*)，等等，其绝对值保持有界。它非常容易计算，但可能需要非常长的时间，因为需要确保给定的数字不会发散。这通常是通过迭代计算直到最大迭代次数来完成的，之后，如果数字仍然在某个范围内，则认为它不是发散的。当然，你进行的迭代越多，你得到的精度就越高。

**图 4.5**

罗马 esco 西兰花，显示出近似自然分形的自相似形态。图片由 [Jon Sullivan](https://commons.wikimedia.org/wiki/File:Fractal_Broccoli.jpg)，2004 年拍摄。

![img/c00398a6e63dcdcf8495aa58e80c0f72.png](img/c00398a6e63dcdcf8495aa58e80c0f72.png)

### Python 实现

纯 Python 实现如下编写：

```py
def mandelbrot_python(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon=2.0):
        def mandelbrot(z, maxiter):
            c = z
            for n in range(maxiter):
                if abs(z) > horizon:
                    return n
                z = z*z + c
            return maxiter
        r1 = [xmin+i*(xmax-xmin)/xn for i in range(xn)]
        r2 = [ymin+i*(ymax-ymin)/yn for i in range(yn)]
        return [mandelbrot(complex(r, i),maxiter) for r in r1 for i in r2]

```

这段代码中有趣（且缓慢）的部分是 `mandelbrot` 函数，它实际上计算序列 *f**c*))。此类代码的向量化并不完全直接，因为内部的 `return` 表明对元素进行了微分处理。一旦它发散，我们就不需要再迭代了，并且可以安全地返回发散时的迭代次数。问题是然后在 numpy 中做同样的事情。但如何做呢？

### Numpy 实现

技巧在于在每次迭代中搜索尚未发散的值，并只更新这些值的相关信息。因为我们从 *Z* = 0 开始，我们知道每个值至少会被更新一次（当它们等于 0 时，它们尚未发散），并且一旦它们发散，就会停止更新。为此，我们将使用 numpy 的花式索引和 `less(x1,x2)` 函数，该函数返回 `(x1 < x2)` 的逐元素真值。

```py
def mandelbrot_numpy(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon=2.0):
        X = np.linspace(xmin, xmax, xn, dtype=np.float32)
        Y = np.linspace(ymin, ymax, yn, dtype=np.float32)
        C = X + Y[:,None]*1j
        N = np.zeros(C.shape, dtype=int)
        Z = np.zeros(C.shape, np.complex64)
        for n in range(maxiter):
            I = np.less(abs(Z), horizon)
            N[I] = n
            Z[I] = Z[I]**2 + C[I]
        N[N == maxiter-1] = 0
        return Z, N

```

这里是基准测试：

```py
>>> xmin, xmax, xn = -2.25, +0.75, int(3000/3)
    >>> ymin, ymax, yn = -1.25, +1.25, int(2500/3)
    >>> maxiter = 200
    >>> timeit("mandelbrot_python(xmin, xmax, ymin, ymax, xn, yn, maxiter)", globals())
    1 loops, best of 3: 6.1 sec per loop >>> timeit("mandelbrot_numpy(xmin, xmax, ymin, ymax, xn, yn, maxiter)", globals())
    1 loops, best of 3: 1.15 sec per loop

```

### 更快的 Numpy 实现

收益大约是 5 倍，没有我们预期的那么多。部分问题是 `np.less` 函数在每次迭代中都意味着 *xn* × *yn* 测试，而我们知道某些值已经发散了。即使这些测试在 C 级别（通过 numpy）执行，成本仍然相当显著。Dan Goodman [`thesamovar.wordpress.com/`](https://thesamovar.wordpress.com/) 提出的一种另一种方法是，在每次迭代中处理一个动态数组，该数组只存储尚未发散的点。这需要更多的行，但结果更快，与 Python 版本相比，速度提高了 10 倍。

```py
def mandelbrot_numpy_2(xmin, xmax, ymin, ymax, xn, yn, itermax, horizon=2.0):
        Xi, Yi = np.mgrid[0:xn, 0:yn]
        Xi, Yi = Xi.astype(np.uint32), Yi.astype(np.uint32)
        X = np.linspace(xmin, xmax, xn, dtype=np.float32)[Xi]
        Y = np.linspace(ymin, ymax, yn, dtype=np.float32)[Yi]
        C = X + Y*1j
        N_ = np.zeros(C.shape, dtype=np.uint32)
        Z_ = np.zeros(C.shape, dtype=np.complex64)
        Xi.shape = Yi.shape = C.shape = xn*yn

        Z = np.zeros(C.shape, np.complex64)
        for i in range(itermax):
            if not len(Z): break

            # Compute for relevant points only
            np.multiply(Z, Z, Z)
            np.add(Z, C, Z)

            # Failed convergence
            I = abs(Z) > horizon
            N_[Xi[I], Yi[I]] = i+1
            Z_[Xi[I], Yi[I]] = Z[I]

            # Keep going with those who have not diverged yet
            np.negative(I,I)
            Z = Z[I]
            Xi, Yi = Xi[I], Yi[I]
            C = C[I]
        return Z_.T, N_.T

```

基准测试给我们：

```py
>>> timeit("mandelbrot_numpy_2(xmin, xmax, ymin, ymax, xn, yn, maxiter)", globals())
    1 loops, best of 3: 510 msec per loop

```

### 可视化

为了可视化我们的结果，我们可以直接使用 matplotlib 的`imshow`命令显示`N`数组，但这会导致“带状”图像，这是我们所使用的逃逸计数算法的已知后果。这种带状可以通过使用分数逃逸计数来消除。这可以通过测量迭代点落在逃逸截止点之外有多远来实现。请参阅下文关于逃逸计数的重新归一化的参考。以下是使用重新计数归一化，并添加了功率归一化颜色图（伽马=0.3）以及浅阴影的结果图片。

**图 4.6**

使用重新计数归一化、功率归一化颜色图（伽马=0.3）和浅阴影渲染的曼德尔布罗特集。

![img/9ea19f540f154a1bc6fcf2fa7a4ba552.png](img/9ea19f540f154a1bc6fcf2fa7a4ba552.png)

### 练习

注意

你应该查看[ufunc.reduceat](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.reduceat.html)方法，该方法在单个轴上执行具有指定切片的（局部）归约。

我们现在想使用[闵可夫斯基-博利冈德维度](https://en.wikipedia.org/wiki/Minkowski–Bouligand_dimension)来测量曼德尔布罗特集的分数维数。为此，我们需要使用递减的盒子大小进行盒子计数（见下文图示）。正如你可以想象的那样，我们不能使用纯 Python，因为它会非常慢。这个练习的目的是编写一个使用 numpy 的函数，该函数接受一个二维浮点数组并返回维度。我们将考虑数组中的值是归一化的（即所有值都在 0 和 1 之间）。

**图 4.7**

大不列颠海岸线的闵可夫斯基-博利冈德维度大约为 1.24。

![img/3790c715b411fa4b5650d93efabbaf3d.png](img/3790c715b411fa4b5650d93efabbaf3d.png)

### 来源

+   mandelbrot.py

+   mandelbrot_python.py

+   mandelbrot_numpy_1.py

+   mandelbrot_numpy_2.py

+   fractal_dimension.py（练习的解决方案）

### 参考文献

+   [如何在 Python 中快速计算曼德尔布罗特集](https://www.ibm.com/developerworks/community/blogs/jfp/entry/How_To_Compute_Mandelbrodt_Set_Quickly?lang=en)，Jean Francois Puget，2015。

+   [我的圣诞礼物：Python 中的曼德尔布罗特集计算](https://www.ibm.com/developerworks/community/blogs/jfp/entry/My_Christmas_Gift?lang=en)，Jean Francois Puget，2015。

+   [使用 Python 和 Numpy 快速处理分形](https://thesamovar.wordpress.com/2009/03/22/fast-fractals-with-python-and-numpy/)，Dan Goodman，2009。

+   [重新归一化曼德尔布罗特逃逸](http://linas.org/art-gallery/escape/escape.html)，Linas Vepstas，1997。

## 空间向量化

空间向量化指的是元素共享相同的计算，但只与其他元素的一个子集交互。在“生命游戏”示例中已经是这种情况，但在某些情况下，由于子集是动态的并且需要在每次迭代中更新，因此会带来额外的困难。例如，在粒子系统中，粒子主要与本地邻居交互。对于模拟集群行为的“boids”也是如此。

**图 4.8**

集群鸟类是生物学中自组织的例子。图片由[Christoffer A Rasmussen](https://commons.wikimedia.org/wiki/File:Fugle,_ørnsø_073.jpg)，2012 年拍摄。

![img/7b37fde927b910d7cbe8fb3dcbf545d4.png](img/7b37fde927b910d7cbe8fb3dcbf545d4.png)

### Boids

注意

来自维基百科条目[Boids](https://en.wikipedia.org/wiki/Boids)的摘录

Boids 是一个由克雷格·雷诺兹在 1986 年开发的人工生命程序，它模拟了鸟类的集群行为。名称“boid”是“bird-oid object”（鸟形对象）的缩写，指的是类似鸟的对象。

与大多数人工生命模拟一样，Boids 是一个涌现行为的例子；也就是说，Boids 的复杂性源于遵循一组简单规则的个人代理（在这种情况下是 boids）的交互。在最简单的 Boids 世界中应用的规则如下：

+   **分离**：避免拥挤本地鸟群成员

+   **对齐**：朝向本地鸟群成员的平均航向

+   **凝聚**：朝向本地鸟群成员的平均位置（质心）移动

**图 4.9**

Boids 受三个局部规则（分离、凝聚和对齐）的支配，这些规则作为计算速度和加速度。

![img/c8b3883aeb2fb8322a4b531ca31ccaed.png](img/c8b3883aeb2fb8322a4b531ca31ccaed.png)

### Python 实现

由于每个 Boid 都是一个具有位置和速度等几个属性的自主实体，因此从编写 Boid 类开始似乎是自然的：

```py
import math
    import random
    from vec2 import vec2

    class Boid:
        def __init__(self, x=0, y=0):
            self.position = vec2(x, y)
            angle = random.uniform(0, 2*math.pi)
            self.velocity = vec2(math.cos(angle), math.sin(angle))
            self.acceleration = vec2(0, 0)

```

`vec2`对象是一个处理所有常见 2 分量向量操作的非常简单的类。它将在主`Boid`类中为我们节省一些编写工作。请注意，Python 包索引中有些向量包，但对于这样一个简单的例子来说，这将是过度设计。

对于常规 Python 来说，Boid 是一个难题，因为 Boid 与本地邻居有交互。然而，由于 Boids 是移动的，要找到这样的本地邻居，需要在每个时间步计算每个 Boid 与其他每个 Boid 的距离，以便对那些在给定交互半径内的 Boid 进行排序。因此，编写三个规则的原型方式如下：

```py
def separation(self, boids):
        count = 0
        for other in boids:
            d = (self.position - other.position).length()
            if 0 < d < desired_separation:
                count += 1
                ...
        if count > 0:
            ...

     def alignment(self, boids): ...
     def cohesion(self, boids): ...

```

完整的源代码在下面的参考文献部分给出，在这里描述它将太长，而且没有真正的困难。

为了完整地描述，我们还可以创建一个`Flock`对象：

```py
class Flock:
        def __init__(self, count=150):
            self.boids = []
            for i in range(count):
                boid = Boid()
                self.boids.append(boid)

        def run(self):
            for boid in self.boids:
                boid.run(self.boids)

```

使用这种方法，我们可以有高达 50 个 boids，直到计算时间变得太慢，无法进行平滑动画。正如你可能猜到的，我们可以使用 numpy 做得更好，但让我首先指出这个 Python 实现的主要问题。如果你查看代码，你肯定会注意到有很多冗余。更确切地说，我们没有利用欧几里得距离是自反的这一事实，即 |*x* − *y*| = |*y* − *x*|。在这个原始的 Python 实现中，每个规则（函数）计算 *n*² 个距离，而如果适当缓存，(*n*²)/(2) 就足够了。此外，每个规则都会重新计算每个距离，而不缓存其他函数的结果。最终，我们计算了 3*n*² 个距离，而不是 (*n*²)/(2)。

### Numpy 实现

如你所料，numpy 实现采取了不同的方法，我们将把所有的 boids 收集到一个`position`数组和`velocity`数组中：

```py
n = 500
    velocity = np.zeros((n, 2), dtype=np.float32)
    position = np.zeros((n, 2), dtype=np.float32)

```

第一步是计算所有 boids 的局部邻域，为此我们需要计算所有配对距离：

```py
dx = np.subtract.outer(position[:, 0], position[:, 0])
    dy = np.subtract.outer(position[:, 1], position[:, 1])
    distance = np.hypot(dx, dy)

```

我们本可以使用 scipy 的[cdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)，但我们需要稍后使用`dx`和`dy`数组。一旦这些数组被计算出来，使用[hypot](https://docs.scipy.org/doc/numpy/reference/generated/numpy.hypot.html)方法会更快。请注意，距离形状是`(n, n)`，每一行都对应一个 boid，即每一行给出了到所有其他 boids（包括自身）的距离。

从这些距离中，我们现在可以计算每个三个规则的局部邻域，利用我们可以混合它们的事实。实际上，我们可以计算一个严格正（即没有自我交互）的距离掩码，并将其与其他距离掩码相乘。

注意

如果我们假设 boids 不能占据相同的位置，你如何更有效地计算`mask_0`？

```py
mask_0 = (distance > 0)
    mask_1 = (distance < 25)
    mask_2 = (distance < 50)
    mask_1 *= mask_0
    mask_2 *= mask_0
    mask_3 = mask_2

```

然后，我们计算给定半径内的邻居数量，并确保它至少为 1，以避免除以零。

```py
mask_1_count = np.maximum(mask_1.sum(axis=1), 1)
    mask_2_count = np.maximum(mask_2.sum(axis=1), 1)
    mask_3_count = mask_2_count

```

我们准备好编写我们的三个规则：

**对齐**

```py
# Compute the average velocity of local neighbours
    target = np.dot(mask, velocity)/count.reshape(n, 1)

    # Normalize the result
    norm = np.sqrt((target*target).sum(axis=1)).reshape(n, 1)
    target *= np.divide(target, norm, out=target, where=norm != 0)

    # Alignment at constant speed
    target *= max_velocity

    # Compute the resulting steering
    alignment = target - velocity

```

**凝聚力**

```py
# Compute the gravity center of local neighbours
    center = np.dot(mask, position)/count.reshape(n, 1)

    # Compute direction toward the center
    target = center - position

    # Normalize the result
    norm = np.sqrt((target*target).sum(axis=1)).reshape(n, 1)
    target *= np.divide(target, norm, out=target, where=norm != 0)

    # Cohesion at constant speed (max_velocity)
    target *= max_velocity

    # Compute the resulting steering
    cohesion = target - velocity

```

**分离**

```py
# Compute the repulsion force from local neighbours
    repulsion = np.dstack((dx, dy))

    # Force is inversely proportional to the distance
    repulsion = np.divide(repulsion, distance.reshape(n, n, 1)**2, out=repulsion,
                          where=distance.reshape(n, n, 1) != 0)

    # Compute direction away from others
    target = (repulsion*mask.reshape(n, n, 1)).sum(axis=1)/count.reshape(n, 1)

    # Normalize the result
    norm = np.sqrt((target*target).sum(axis=1)).reshape(n, 1)
    target *= np.divide(target, norm, out=target, where=norm != 0)

    # Separation at constant speed (max_velocity)
    target *= max_velocity

    # Compute the resulting steering
    separation = target - velocity

```

所有的三个结果导向（分离、对齐和凝聚力）都需要限制其幅度。我们将此作为读者的练习。这些规则的组合以及速度和位置的更新都是简单直接的：

```py
acceleration = 1.5 * separation + alignment + cohesion
    velocity += acceleration
    position += velocity

```

我们最终使用自定义定向散点图来可视化结果。

**图 4.10**

Boids 是一个由克雷格·雷诺兹在 1986 年开发的人工生命程序，它模拟了鸟类的集群行为。

<https://www.labri.fr/perso/nrougier/from-python-to-numpy/data/boids.mp4>

您的浏览器不支持视频标签。

### 练习

我们现在可以可视化我们的鸟群了。最简单的方法是使用 matplotlib 动画函数和散点图。不幸的是，散点图不能单独定向，我们需要使用 matplotlib 的 `PathCollection` 创建自己的对象。一个简单的三角形路径可以定义为：

```py
v= np.array([(-0.25, -0.25),
                 ( 0.00,  0.50),
                 ( 0.25, -0.25),
                 ( 0.00,  0.00)])
    c = np.array([Path.MOVETO,
                  Path.LINETO,
                  Path.LINETO,
                  Path.CLOSEPOLY])

```

这个路径可以在数组内部重复多次，并且每个三角形都可以独立。

```py
n = 500
    vertices = np.tile(v.reshape(-1), n).reshape(n, len(v), 2)
    codes = np.tile(c.reshape(-1), n)

```

现在我们有一个 `(n,4,2)` 的顶点数组和一个 `(n,4)` 的数组来表示 `n` 个鸟群，我们感兴趣的是操作顶点数组以反映每个 `n` 个鸟群的平移、缩放和旋转。

注意

旋转确实很棘手。

你会如何编写 `translate`、`scale` 和 `rotate` 函数？

### 来源

+   boid_python.py

+   boid_numpy.py（练习的解决方案）

### 参考文献

+   [鸟群](https://processing.org/examples/flocking.html)，Daniel Shiffman，2010。

+   [鸟群、牧群和学校：一种分布式行为模型](http://www.red3d.com/cwr/boids/)，Craig Reynolds，SIGGRAPH，1987

## 结论

通过这些示例，我们已经看到了三种代码向量化形式：

+   均匀向量化，其中元素无条件且在相同时间内共享相同的计算。

+   时间向量化，其中元素共享相同的计算，但需要不同数量的迭代。

+   空间向量化，其中元素共享相同的计算，但基于动态空间参数。

可能还有更多这样的直接代码向量化形式。正如之前解释的，这种向量化是最简单的一种，尽管我们看到了它实施起来可能非常棘手，需要一些经验，一些帮助，或者两者都需要。例如，鸟群练习的解决方案是由 [Divakar](http://stackoverflow.com/users/3293881/divakar) 在 [Stack Overflow](http://stackoverflow.com/questions/40822983/multiple-individual-2d-rotation-at-once) 上提供的，在解释了我的问题之后。

# 问题向量化

**目录**

+   引言

+   路径查找

    +   构建迷宫

    +   广度优先搜索

    +   Bellman-Ford 方法

    +   来源

    +   参考文献

+   流体动力学

    +   拉格朗日方法与欧拉方法

    +   Numpy 实现

    +   来源

    +   参考文献

+   蓝色噪声采样

    +   DART 方法

    +   Bridson 方法

    +   来源

    +   参考文献

+   结论

## 引言

问题向量化比代码向量化要难得多，因为它意味着你从根本上必须重新思考你的问题，以便使其可向量化。大多数情况下，这意味着你必须使用不同的算法来解决你的问题，或者更糟糕的是……发明一个新的算法。因此，困难在于跳出思维定势。

为了说明这一点，让我们考虑一个简单的问题，给定两个向量 `X` 和 `Y`，我们想要计算所有索引对 `i`，`j` 的 `X[i]*Y[j]` 的和。一个简单且明显的方法是写出：

```py
def compute_python(X, Y):
        result = 0
        for i in range(len(X)):
            for j in range(len(Y)):
                result += X[i] * Y[j]
        return result

```

然而，这个最初且简单的方法需要两个循环，而我们已知它将会很慢：

```py
>>> X = np.arange(1000)
    >>> timeit("compute_python(X,X)")
    1 loops, best of 3: 0.274481 sec per loop

```

那么如何向量化这个问题呢？如果你记得你的线性代数课程，你可能已经识别出表达式 `X[i] * Y[j]` 与矩阵乘法表达式非常相似。所以也许我们可以从一些 numpy 速度提升中受益。一个错误的方法是写出：

```py
def compute_numpy_wrong(X, Y):
        return (X*Y).sum()

```

这是错误的，因为 `X*Y` 表达式实际上会计算一个新的向量 `Z`，使得 `Z[i] = X[i] * Y[i]`，而这不是我们想要的。相反，我们可以利用 numpy 的广播功能，首先重塑两个向量，然后相乘：

```py
def compute_numpy(X, Y):
        Z = X.reshape(len(X),1) * Y.reshape(1,len(Y))
        return Z.sum()

```

这里我们有 `Z[i,j] == X[i,0]*Y[0,j]`，如果我们对 `Z` 的每个元素求和，我们就能得到预期的结果。让我们看看在这个过程中我们获得了多少速度提升：

```py
>>> X = np.arange(1000)
    >>> timeit("compute_numpy(X,X)")
    10 loops, best of 3: 0.00157926 sec per loop

```

这更好，我们得到了一个因子 ~150。但我们还能做得更好。

如果你再次仔细地看纯 Python 版本，你会发现内循环使用了 `X[i]`，它不依赖于 `j` 索引，这意味着它可以从内循环中移除。代码可以重写为：

```py
def compute_numpy_better_1(X, Y):
        result = 0
        for i in range(len(X)):
            Ysum = 0
            for j in range(len(Y)):
                Ysum += Y[j]
            result += X[i]*Ysum
        return result

```

但由于内循环不依赖于 `i` 索引，我们不妨只计算一次：

```py
def compute_numpy_better_2(X, Y):
        result = 0
        Ysum = 0
        for j in range(len(Y)):
            Ysum += Y[j]
        for i in range(len(X)):
            result += X[i]*Ysum
        return result

```

还不错，我们移除了内循环，这意味着将 *O*(*n*²) 复杂度转换为 *O*(*n*) 复杂度。使用相同的方法，我们现在可以写出：

```py
def compute_numpy_better_3(x, y):
        Ysum = 0
        for j in range(len(Y)):
            Ysum += Y[j]
        Xsum = 0
        for i in range(len(X)):
            Xsum += X[i]
        return Xsum*Ysum

```

最后，意识到我们只需要 `X` 和 `Y` 的和的乘积，我们可以利用 `np.sum` 函数并写出：

```py
def compute_numpy_better(x, y):
        return np.sum(y) * np.sum(x)

```

它更短，更清晰，并且快得多：

```py
>>> X = np.arange(1000)
    >>> timeit("compute_numpy_better(X,X)")
    1000 loops, best of 3: 3.97208e-06 sec per loop

```

我们确实重新表述了我们的问题，利用了 ∑[*ij*]*X*[*i*]*Y*[*j*] = ∑[*i*]*X*[*i*]∑[*j*]*Y*[*j*] 的事实，并且在同时我们了解到有两种向量化：代码向量和问题向量化。后者是最难的，但也是最重要的，因为这是你可以期望在速度上获得巨大提升的地方。在这个简单的例子中，我们通过代码向量化获得了 150 倍的因子，但通过问题向量化我们获得了 70,000 倍的因子，仅仅是通过以不同的方式编写我们的问题（尽管你不能期望在所有情况下都能获得如此巨大的速度提升）。然而，代码向量化仍然是一个重要的因素，如果我们用 Python 的方式重写最后一个解决方案，改进是好的，但不如 numpy 版本那么好：

```py
def compute_python_better(x, y):
        return sum(x)*sum(y)

```

这个新的 Python 版本比之前的 Python 版本快得多，但仍然，它比 numpy 版本慢 50 倍：

```py
>>> X = np.arange(1000)
    >>> timeit("compute_python_better(X,X)")
    1000 loops, best of 3: 0.000155677 sec per loop

```

## 寻路

寻路问题就是在一个图中找到最短路径。这可以分为两个独立的问题：在一个图中找到两个节点之间的路径，以及找到最短路径。我们将通过迷宫中的寻路来展示这一点。因此，第一个任务是构建一个迷宫。

**图 5.1**

位于英国 Longleat 庄园的树篱迷宫。图片由[Prince Rurik](https://commons.wikimedia.org/wiki/File:Longleat_maze.jpg)，2005 年拍摄。

![img/760876f935b5a4bdc2877549c1cabec9.png](img/760876f935b5a4bdc2877549c1cabec9.png)

### 构建迷宫

存在着[许多迷宫生成算法](https://en.wikipedia.org/wiki/Maze_generation_algorithm)，但我倾向于使用我已经使用了几年的算法，但其起源对我来说是未知的。我已经在引用的维基百科条目中添加了代码。如果你知道原始作者，请随时完善它。该算法通过创建长度为 `p`（复杂度）的 `n`（密度）岛屿来工作。岛屿是通过选择具有奇数坐标的随机起始点来创建的，然后选择一个随机方向。如果在给定方向上两步的单元格是空的，那么在这个方向上一步和两步都添加墙壁。这个过程为这个岛屿迭代 `n` 步。创建 `p` 个岛屿。`n` 和 `p` 被表示为 `float` 以适应迷宫的大小。低复杂度时，岛屿非常小，迷宫容易解决。低密度时，迷宫有更多的“大空房间”。

```py
def build_maze(shape=(65, 65), complexity=0.75, density=0.50):
        # Only odd shapes
        shape = ((shape[0]//2)*2+1, (shape[1]//2)*2+1)

        # Adjust complexity and density relatively to maze size
        n_complexity = int(complexity*(shape[0]+shape[1]))
        n_density = int(density*(shape[0]*shape[1]))

        # Build actual maze
        Z = np.zeros(shape, dtype=bool)

        # Fill borders
        Z[0, :] = Z[-1, :] = Z[:, 0] = Z[:, -1] = 1

        # Islands starting point with a bias in favor of border
        P = np.random.normal(0, 0.5, (n_density, 2))
        P = 0.5 - np.maximum(-0.5, np.minimum(P, +0.5))
        P = (P*[shape[1], shape[0]]).astype(int)
        P = 2*(P//2)

        # Create islands
        for i in range(n_density):
            # Test for early stop: if all starting point are busy, this means we
            # won't be able to connect any island, so we stop.
            T = Z[2:-2:2, 2:-2:2]
            if T.sum() == T.size: break
            x, y = P[i]
            Z[y, x] = 1
            for j in range(n_complexity):
                neighbours = []
                if x > 1:          neighbours.append([(y, x-1), (y, x-2)])
                if x < shape[1]-2: neighbours.append([(y, x+1), (y, x+2)])
                if y > 1:          neighbours.append([(y-1, x), (y-2, x)])
                if y < shape[0]-2: neighbours.append([(y+1, x), (y+2, x)])
                if len(neighbours):
                    choice = np.random.randint(len(neighbours))
                    next_1, next_2 = neighbours[choice]
                    if Z[next_2] == 0:
                        Z[next_1] = 1
                        Z[next_2] = 1
                        y, x = next_2
                else:
                    break
        return Z

```

这里有一个动画展示了生成过程。

**图 5.2**

带有复杂度和密度控制的渐进式迷宫构建。

<https://www.labri.fr/perso/nrougier/from-python-to-numpy/data/maze-build.mp4>

您的浏览器不支持视频标签。

### 广度优先

广度优先（以及深度优先）搜索算法通过从根节点开始检查所有可能性，并在找到解决方案（目标节点）时停止，来解决在两个节点之间找到路径的问题。此算法以线性时间运行，其复杂度为 *O*(|*V*| + |*E*|)（其中 *V* 是顶点数，*E* 是边数）。只要你有合适的数据结构，编写这样的算法并不特别困难。在我们的情况下，迷宫的数组表示并不是最合适的，我们需要将其转换为[Valentin Bryukhanov](http://bryukh.com)提出的实际图。

```py
def build_graph(maze):
        height, width = maze.shape
        graph = {(i, j): [] for j in range(width)
                            for i in range(height) if not maze[i][j]}
        for row, col in graph.keys():
            if row < height - 1 and not maze[row + 1][col]:
                graph[(row, col)].append(("S", (row + 1, col)))
                graph[(row + 1, col)].append(("N", (row, col)))
            if col < width - 1 and not maze[row][col + 1]:
                graph[(row, col)].append(("E", (row, col + 1)))
                graph[(row, col + 1)].append(("W", (row, col)))
        return graph

```

注意

如果我们使用了深度优先算法，我们无法保证找到最短路径，只能找到一条路径（如果存在的话）。

一旦完成这个步骤，编写广度优先算法就很简单了。我们从起始节点开始，只访问当前深度的节点（记住，是广度优先）并迭代这个过程，直到达到最终节点（如果可能的话）。问题是：通过这种方式探索图，我们能得到最短路径吗？在这个特定情况下，“是的”，因为我们没有带权重的图，即所有边都有相同的权重（或成本）。

```py
def breadth_first(maze, start, goal):
        queue = deque([([start], start)])
        visited = set()
        graph = build_graph(maze)

        while queue:
            path, current = queue.popleft()
            if current == goal:
                return np.array(path)
            if current in visited:
                continue
            visited.add(current)
            for direction, neighbour in graph[current]:
                p = list(path)
                p.append(neighbour)
                queue.append((p, neighbour))
        return None

```

### Bellman-Ford 方法

Bellman-Ford 算法是一种能够通过扩散过程在图中找到最优路径的算法。最优路径是通过沿着产生的梯度上升找到的。这个算法的时间复杂度是二次的 *O*(|*V*||*E*|)（其中 *V* 是顶点的数量，*E* 是边的数量）。然而，在我们的简单情况下，我们不会遇到最坏的情况。算法如图所示（从左到右，从上到下阅读）。一旦完成，我们就可以从起始节点开始沿着梯度上升。您可以在图中看到这会导致最短路径。

**图 5.3**

在简单迷宫上的值迭代算法。一旦到达入口，通过向上爬升值梯度很容易找到最短路径。

![img/23273028ed7c308c5053f969e41f7471.png](img/23273028ed7c308c5053f969e41f7471.png) ![img/0bd752138eda95a4a062032e9ef4100e.png](img/0bd752138eda95a4a062032e9ef4100e.png) ![img/e33abc8e918425a64e9453cf0a651e41.png](img/e33abc8e918425a64e9453cf0a651e41.png) ![img/89b5382abc5d8c757df5ae510473f636.png](img/89b5382abc5d8c757df5ae510473f636.png) ![img/28a4fc5dbe7ab7c7cd41be569274fec7.png](img/28a4fc5dbe7ab7c7cd41be569274fec7.png) ![img/c980df26414ea32c86e51335a027d6b9.png](img/c980df26414ea32c86e51335a027d6b9.png) ![img/37b7fa5200493002586ba83473dfa57f.png](img/37b7fa5200493002586ba83473dfa57f.png) ![img/2cd153d230f00b7cdb9e8c87ee555fad.png](img/2cd153d230f00b7cdb9e8c87ee555fad.png) ![img/0970a7d64f8fce9622ca4d1bf1fb440d.png](img/0970a7d64f8fce9622ca4d1bf1fb440d.png) ![img/a731b6cc95670022e66d5585c8d7e130.png](img/a731b6cc95670022e66d5585c8d7e130.png)

我们首先将出口节点设置为值 1，而其他所有节点都设置为 0，除了墙壁。然后我们迭代一个过程，使得每个单元格的新值是当前单元格值和折扣（以下情况中`gamma=0.9`）的 4 个邻居值之间的最大值。一旦起始节点值变为严格正值，这个过程就开始。

如果我们利用`generic_filter`（来自`scipy.ndimage`）进行扩散过程，NumPy 的实现就很简单：

```py
def diffuse(Z):
        # North, West, Center, East, South
        return max(gamma*Z[0], gamma*Z[1], Z[2], gamma*Z[3], gamma*Z[4])

    # Build gradient array
    G = np.zeros(Z.shape)

    # Initialize gradient at the entrance with value 1
    G[start] = 1

    # Discount factor
    gamma = 0.99

    # We iterate until value at exit is > 0\. This requires the maze
    # to have a solution or it will be stuck in the loop.
    while G[goal] == 0.0:
        G = Z * generic_filter(G, diffuse, footprint=[[0, 1, 0],
                                                      [1, 1, 1],
                                                      [0, 1, 0]])

```

但在这个特定情况下，它相当慢。我们最好自己想出一个解决方案，重用生命游戏代码的一部分：

```py
# Build gradient array
    G = np.zeros(Z.shape)

    # Initialize gradient at the entrance with value 1
    G[start] = 1

    # Discount factor
    gamma = 0.99

    # We iterate until value at exit is > 0\. This requires the maze
    # to have a solution or it will be stuck in the loop.
    G_gamma = np.empty_like(G)
    while G[goal] == 0.0:
        np.multiply(G, gamma, out=G_gamma)
        N = G_gamma[0:-2,1:-1]
        W = G_gamma[1:-1,0:-2]
        C = G[1:-1,1:-1]
        E = G_gamma[1:-1,2:]
        S = G_gamma[2:,1:-1]
        G[1:-1,1:-1] = Z[1:-1,1:-1]*np.maximum(N,np.maximum(W,
                                    np.maximum(C,np.maximum(E,S))))

```

一旦完成，我们就可以沿着梯度上升找到如图所示的最近路径：

**图 5.4**

使用 Bellman-Ford 算法进行路径查找。渐变颜色表示从迷宫的终点（右下角）传播的值。路径是通过从目标向上爬升梯度找到的。

![img/d808f0dc7c41b6768554a47834b568e9.png](img/d808f0dc7c41b6768554a47834b568e9.png)

### 来源

+   maze_build.py

+   maze_numpy.py

### 参考文献

+   [迷宫算法](http://bryukh.com/labyrinth-algorithms/), 瓦伦丁·布里亚库诺夫，2014.

## 流体动力学

**图 5.5**

德国海德堡内卡河在不同放大级别下的流体动力学流动。图片由[Steven Mathey](https://commons.wikimedia.org/wiki/File:Self_Similar_Turbulence.png)，2012 年拍摄。

![img/89e537df1608439e1d4339ff3bad9a78.png](img/89e537df1608439e1d4339ff3bad9a78.png)

### 拉格朗日与欧拉方法

注意

来自维基百科关于[拉格朗日和欧拉流场指定](https://en.wikipedia.org/wiki/Lagrangian_and_Eulerian_specification_of_the_flow_field)条目的摘录

在经典场论中，场的拉格朗日指定是一种观察流体运动的方法，观察者跟随单个流体包随其在空间和时间中的移动。通过时间绘制单个包的位置给出包的路径线。这可以想象为坐在船上，随河流漂流。

欧拉流场指定是一种观察流体运动的方法，它关注随着时间流逝，流体流经空间中的特定位置。这可以通过坐在河岸上，观察水流过固定位置来可视化。

换句话说，在欧拉方法中，你将空间的一部分划分为单元格，每个单元格包含一个速度向量和其他信息，例如密度和温度。在拉格朗日方法中，我们需要基于粒子的物理，具有动态交互，通常需要大量的粒子。这两种方法都有优缺点，选择哪种方法取决于你问题的本质。当然，你也可以将这两种方法混合成一种混合方法。

然而，基于粒子的模拟最大的问题是粒子交互需要找到邻近的粒子，正如我们在 boids 案例中看到的那样，这会产生成本。如果我们只针对 Python 和 numpy，那么选择欧拉方法可能更好，因为与拉格朗日方法相比，向量化几乎将是微不足道的。

### Numpy 实现

我不会解释计算流体动力学背后的所有理论，因为首先，我做不到（我对这个领域根本不是专家）而且网上有许多资源很好地解释了这一点（查看下面的参考文献，特别是 L. Barba 的教程）。那么为什么选择计算流体动力学作为例子呢？因为结果（几乎）总是美丽而迷人的。我无法抗拒（看看下面的电影）。

我们将通过实施计算机图形学中的一个方法来进一步简化问题，该方法的目标不是正确性，而是令人信服的行为。Jos Stam 为 1999 年的 SIGGRAPH 撰写了一篇非常好的文章，描述了一种在长时间内保持流体稳定的技术（即其长期解不会发散）。[Alberto Santini](https://github.com/albertosantini/python-fluid)很久以前（使用 numarray！）编写了一个 Python 复制版本，这样我就只需要将其适配到现代 numpy，并使用现代 numpy 技巧进行一点加速。

我不会对代码进行注释，因为它太长了，但您也可以阅读原始论文以及 [Philip Rideout](http://prideout.net/blog/?p=58) 在他的博客上的解释。以下是我使用这种技术制作的几个电影。

**图 5.6**

使用 Jos Stam 的稳定流体算法进行的烟雾模拟。最右侧的视频来自 [glumpy](http://glumpy.github.io) 包，并使用 GPU（帧缓冲区操作，即不使用 OpenCL 也不使用 CUDA）进行更快的计算。

<https://www.labri.fr/perso/nrougier/from-python-to-numpy/data/smoke-1.mp4>

您的浏览器不支持视频标签。

<https://www.labri.fr/perso/nrougier/from-python-to-numpy/data/smoke-2.mp4>

您的浏览器不支持视频标签。

<https://www.labri.fr/perso/nrougier/from-python-to-numpy/data/smoke-gpu.mp4>

您的浏览器不支持视频标签。

### 源代码

+   smoke_1.py

+   smoke_2.py

+   smoke_solver.py

+   smoke_interactive.py

### 参考文献

+   [12 步到纳维-斯托克斯方程](https://github.com/barbagroup/CFDPython), Lorena Barba, 2013.

+   [稳定流体](http://www.dgp.toronto.edu/people/stam/reality/Research/pdf/ns.pdf), Jos Stam, 1999.

+   [简单流体模拟](http://prideout.net/blog/?p=58), Philip Rideout, 2010

+   [在 GPU 上进行快速流体动力学模拟](http://http.developer.nvidia.com/GPUGems/gpugems_ch38.html), Mark Harris, 2004.

+   [将沙子作为流体进行动画](https://www.cs.ubc.ca/%7Erbridson/docs/zhu-siggraph05-sandfluid.pdf), Yongning Zhu & Robert Bridson, 2005.

## 蓝色噪声采样

蓝色噪声指的是具有随机且均匀分布且没有任何频谱偏差的样本集。这种噪声在渲染、抖动、点彩等多种图形应用中非常有用。已经提出了许多不同的方法来实现这种噪声，但最简单的方法无疑是 DART 方法。

**图 5.7**

文森特·梵高 1889 年的《星夜》的细节。细节已经使用 voronoi 单元中心为蓝色噪声样本进行重采样。

![img/b5243aadd074137f551ae1b47c8d5cc0.png](img/b5243aadd074137f551ae1b47c8d5cc0.png)

### DART 方法

DART 方法是最早且最简单的方法之一。它通过顺序地绘制均匀随机点，并且只接受那些与每个先前接受的样本保持最小距离的点。因此，这种顺序方法非常慢，因为每个新的候选点都需要与先前接受的候选点进行测试。你接受的点越多，该方法就越慢。让我们考虑单位表面和每个点之间要强制执行的半径 `r`。

知道在平面上圆的密集排列是蜜蜂蜂巢的六边形晶格，我们知道这种密度是 *d* = (1)/(6)*π*√(3)（实际上 [我在写这本书时学到了这一点](https://en.wikipedia.org/wiki/Circle_packing)）。考虑半径为 *r* 的圆，我们最多可以排列 (*d*)/(*π**r*²) = (√(3))/(6*r*²) = (1)/(2*r*²√(3)) 的圆。我们知道可以排列在表面上的圆片数量的理论上限，但我们可能不会达到这个上限，因为随机放置。此外，由于在几个点被接受之后，许多点将被拒绝，我们需要在停止整个过程之前设定连续失败尝试的数量上限。

```py
import math
    import random

    def DART_sampling(width=1.0, height=1.0, r = 0.025, k=100):
        def distance(p0, p1):
            dx, dy = p0[0]-p1[0], p0[1]-p1[1]
            return math.hypot(dx, dy)

        points = []
        i = 0
        last_success = 0
        while True:
            x = random.uniform(0, width)
            y = random.uniform(0, height)
            accept = True
            for p in points:
                if distance(p, (x, y)) < r:
                    accept = False
                    break
            if accept is True:
                points.append((x, y))
                if i-last_success > k:
                    break
                last_success = i
            i += 1
        return points

```

我将 DART 方法的向量化留作练习。想法是预先计算足够的均匀随机样本以及成对距离，并测试它们的顺序包含。

### Bridson 方法

如果前一种方法的向量化没有真正的困难，速度提升并不那么好，质量仍然很低，并且依赖于 `k` 参数。数值越高越好，因为它基本上控制了尝试插入新样本的难度。但是，当已经有大量样本被接受时，只有机会才能找到插入新样本的位置。我们可以增加 `k` 值，但这会使方法变得更慢，而且没有任何质量保证。是时候跳出思维定式了，幸运的是，Robert Bridson 已经为我们做到了这一点，并提出了一个简单而有效的方法：

> **步骤 0**. 初始化一个用于存储样本和加速空间搜索的 n 维背景网格。我们选择单元格大小为 (*r*)/(√(*n*))，这样每个网格单元格最多包含一个样本，因此网格可以作为一个简单的 n 维整数数组实现：默认值 -1 表示没有样本，一个非负整数表示位于单元格中的样本索引。
> 
> **步骤 1**. 随机地从域中均匀选择初始样本 *x*[0]。将其插入到背景网格中，并用此索引（零）初始化“活动列表”（样本索引数组）。
> 
> **步骤 2**. 当活动列表不为空时，从其中随机选择一个索引（比如说 *i*）。生成最多 *k* 个点，这些点均匀地选择在半径 *r* 和 2*r* 之间的球形环带中，围绕 *x*[*i*]。依次检查每个点，看它是否在现有样本距离 *r* 以内（使用背景网格仅测试附近的样本）。如果一个点足够远离现有样本，则将其作为下一个样本发射，并将其添加到活动列表中。如果在 *k* 次尝试后没有找到这样的点，则从活动列表中删除 *i*。

实现没有真正的问题，留作读者的练习。请注意，这种方法不仅速度快，而且即使在高 *k* 参数下，也比 DART 方法提供更好的质量（更多样本）。

**图 5.8**

均匀、网格抖动和 Bridson 采样的比较。

![img/3f0249fec14faeea13174847e8028b41.png](img/3f0249fec14faeea13174847e8028b41.png)

### 来源

+   DART_sampling_python.py

+   DART_sampling_numpy.py（练习的解决方案）

+   Bridson_sampling.py（练习的解决方案）

+   sampling.py

+   mosaic.py

+   voronoi.py

### 参考文献

+   [可视化算法](https://bost.ocks.org/mike/algorithms/) Mike Bostock, 2014.

+   [ stippling 和蓝噪声](http://www.joesfer.com/?p=108) Jose Esteve, 2012.

+   [泊松盘采样](http://devmag.org.za/2009/05/03/poisson-disk-sampling/) Herman Tulleken, 2009.

+   [任意维度的快速泊松盘采样](http://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf), Robert Bridson, SIGGRAPH, 2007.

## 结论

我们最后研究的一个例子确实是一个很好的例子，在这个例子中，将问题向量化比将代码向量化（而且过早地）更重要。在这个特定的情况下，我们很幸运地有人为我们完成了这项工作，但这种情况并不总是如此，在这种情况下，向量化我们找到的第一个解决方案的诱惑可能会很大。我希望你现在已经相信，一旦你找到了一个解决方案，通常寻找替代方案是一个好主意。你（几乎）总是可以通过向量化你的代码来提高速度，但在过程中，你可能会错过巨大的改进。

# 自定义向量化

**内容**

+   介绍

+   类型化列表

    +   创建

    +   访问

    +   练习

    +   来源

+   内存感知数组

    +   Glumpy

    +   数组子类

    +   计算范围

    +   跟踪待处理数据

    +   来源

+   结论

## 介绍

NumPy 的一个优点是它可以用来构建新对象或[子类 ndarray 对象](https://docs.scipy.org/doc/numpy/user/basics.subclassing.html)。这个过程可能有点繁琐，但这是值得的，因为它允许你改进 `ndarray` 对象以适应你的问题。在下一节中，我们将研究两个实际案例（类型化列表和内存感知数组），这些案例在[glumpy](http://glumpy.github.io)项目（我维护的项目）中得到了广泛的应用，而最后一个（双精度数组）是一个更学术的案例。

## 类型化列表

打印列表（也称为参差不齐数组）是一个所有项都具有相同数据类型（在 numpy 的意义上）的项列表。它们提供了列表和 ndarray API（当然有一些限制），但由于它们的 API 在某些情况下可能不兼容，我们必须做出选择。例如，关于`+`运算符，我们将选择使用 numpy API，其中值被添加到每个单独的项上，而不是通过追加新项（`1`）来扩展列表。

```py
>>> l = TypedList([[1,2], [3]])
    >>> print(l)
    [1, 2], [3]
    >>> print(l+1)
    [2, 3], [4]

```

从列表 API 中，我们希望我们的新对象能够提供无缝插入、追加和删除项的可能性。

### 创建

由于对象按定义是动态的，因此提供一种通用且功能强大的创建方法以避免以后需要进行操作是很重要的。例如，插入/删除这样的操作成本很高，我们希望避免它们。以下是一个创建`TypedList`对象的建议（以及其他建议）。

```py
def __init__(self, data=None, sizes=None, dtype=float)
        """
        Parameters
        ----------

        data : array_like
            An array, any object exposing the array interface, an object
            whose __array__ method returns an array, or any (nested) sequence.

        sizes:  int or 1-D array
            If `itemsize is an integer, N, the array will be divided
            into elements of size N. If such partition is not possible,
            an error is raised.

            If `itemsize` is 1-D array, the array will be divided into
            elements whose successive sizes will be picked from itemsize.
            If the sum of itemsize values is different from array size,
            an error is raised.

        dtype: np.dtype
            Any object that can be interpreted as a numpy data type.
        """

```

此 API 允许创建一个空列表或从某些外部数据创建一个列表。请注意，在后一种情况下，我们需要指定如何将数据分割成几个项，否则它们将分割成 1-size 项。它可以是一个常规分区（即每个项是 2 个数据长）或一个自定义分区（即数据必须分割成 1、2、3 和 4 个项的大小）。

```py
>>> L = TypedList([[0], [1,2], [3,4,5], [6,7,8,9]])
    >>> print(L)
    [ [0] [1 2] [3 4 5] [6 7 8] ]

    >>> L = TypedList(np.arange(10), [1,2,3,4])
    [ [0] [1 2] [3 4 5] [6 7 8] ]

```

在这一点上，问题是是否要子类化`ndarray`类或使用内部`ndarray`来存储我们的数据。在我们的特定情况下，子类化`ndarray`实际上没有意义，因为我们并不真正想要提供`ndarray`接口。相反，我们将使用`ndarray`来存储列表数据，这种设计选择将为我们提供更多的灵活性。

```py
╌╌╌╌┬───┐┌───┬───┐┌───┬───┬───┐┌───┬───┬───┬───┬╌╌╌╌╌
        │ 0 ││ 1 │ 2 ││ 3 │ 4 │ 5 ││ 6 │ 7 │ 8 │ 9 │
     ╌╌╌┴───┘└───┴───┘└───┴───┴───┘└───┴───┴───┴───┴╌╌╌╌╌╌
       item 1  item 2    item 3         item 4

```

为了存储每个项的极限，我们将使用一个`items`数组，它将负责存储每个项的位置（起始和结束）。对于列表的创建，有两种不同的情况：没有提供数据或提供了数据。第一种情况很简单，只需要创建`_data`和`_items`数组。请注意，它们的大小不是`null`，因为每次我们插入新项时调整数组大小会非常昂贵。相反，最好预留一些空间。

**第一种情况。**没有提供数据，只有 dtype。

```py
self._data = np.zeros(512, dtype=dtype)
    self._items = np.zeros((64,2), dtype=int)
    self._size = 0
    self._count = 0

```

**第二种情况。**已经提供了数据以及一个项大小列表（对于其他情况，请参阅下面的完整代码）

```py
self._data = np.array(data, copy=False)
    self._size = data.size
    self._count = len(sizes)
    indices = sizes.cumsum()
    self._items = np.zeros((len(sizes),2),int)
    self._items[1:,0] += indices[:-1]
    self._items[0:,1] += indices

```

### 访问

一旦完成，每个列表方法只需要进行一点计算和操作不同的键来获取、插入或设置项。以下是`__getitem__`方法的代码。没有真正的困难，但可能的负步骤：

```py
def __getitem__(self, key):
        if type(key) is int:
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("Tuple index out of range")
            dstart = self._items[key][0]
            dstop  = self._items[key][1]
            return self._data[dstart:dstop]

        elif type(key) is slice:
            istart, istop, step = key.indices(len(self))
            if istart > istop:
                istart,istop = istop,istart
            dstart = self._items[istart][0]
            if istart == istop:
                dstop = dstart
            else:
                dstop  = self._items[istop-1][1]
            return self._data[dstart:dstop]

        elif isinstance(key,str):
            return self._data[key][:self._size]

        elif key is Ellipsis:
            return self.data

        else:
            raise TypeError("List indices must be integers")

```

### 练习

列表的修改稍微复杂一些，因为它需要正确管理内存。由于这并不构成真正的困难，所以我们将其留作读者的练习。对于懒惰的人，可以看看下面的代码。注意负步长、关键范围和数组扩展。当底层数组需要扩展时，最好扩展得比实际需要的多，以避免未来的扩展。

**设置项**

```py
L = TypedList([[0,0], [1,1], [0,0]])
    L[1] = 1,1,1

```

```py
╌╌╌╌┬───┬───┐┌───┬───┐┌───┬───┬╌╌╌╌╌
        │ 0 │ 0 ││ 1 │ 1 ││ 2 │ 2 │
     ╌╌╌┴───┴───┘└───┴───┘└───┴───┴╌╌╌╌╌╌
         item 1   item 2   item 3

    ╌╌╌╌┬───┬───┐┌───┬───┲━━━┓┌───┬───┬╌╌╌╌╌
        │ 0 │ 0 ││ 1 │ 1 ┃ 1 ┃│ 2 │ 2 │
     ╌╌╌┴───┴───┘└───┴───┺━━━┛└───┴───┴╌╌╌╌╌╌
         item 1     item 2     item 3

```

**删除项**

```py
L = TypedList([[0,0], [1,1], [0,0]])
    del L[1]

```

```py
╌╌╌╌┬───┬───┐┏━━━┳━━━┓┌───┬───┬╌╌╌╌╌
        │ 0 │ 0 │┃ 1 ┃ 1 ┃│ 2 │ 2 │
     ╌╌╌┴───┴───┘┗━━━┻━━━┛└───┴───┴╌╌╌╌╌╌
         item 1   item 2   item 3

    ╌╌╌╌┬───┬───┐┌───┬───┬╌╌╌╌╌
        │ 0 │ 0 ││ 2 │ 2 │
     ╌╌╌┴───┴───┘└───┴───┴╌╌╌╌╌╌
         item 1    item 2

```

**插入**

```py
L = TypedList([[0,0], [1,1], [0,0]])
    L.insert(1, [3,3])

```

```py
╌╌╌╌┬───┬───┐┌───┬───┐┌───┬───┬╌╌╌╌╌
        │ 0 │ 0 ││ 1 │ 1 ││ 2 │ 2 │
     ╌╌╌┴───┴───┘└───┴───┘└───┴───┴╌╌╌╌╌╌
         item 1   item 2   item 3

    ╌╌╌╌┬───┬───┐┏━━━┳━━━┓┌───┬───┐┌───┬───┬╌╌╌╌╌
        │ 0 │ 0 │┃ 3 ┃ 3 ┃│ 1 │ 1 ││ 2 │ 2 │
     ╌╌╌┴───┴───┘┗━━━┻━━━┛└───┴───┘└───┴───┴╌╌╌╌╌╌
         item 1   item 2   item 3   item 4

```

### 源代码

+   array_list.py（练习的解决方案）

## 内存感知数组

### Glumpy

[Glumpy](http://glumpy.github.io) 是一个基于 OpenGL 的 Python 交互式可视化库，其目标是使创建快速、可扩展、美观、交互式和动态的可视化变得容易。

**图 6.1**

使用密度波理论模拟螺旋星系。

![img/4a1bec6566a39698532f9fb071ef6ffc.png](img/4a1bec6566a39698532f9fb071ef6ffc.png)

**图 6.2**

使用集合和 2 个 GL 调用实现的虎形显示

![img/97405080ef01bac0bf56599757b7489f.png](img/97405080ef01bac0bf56599757b7489f.png)

Glumpy 是基于与 numpy 数组的紧密和无缝集成。这意味着你可以像操作常规 numpy 数组一样操作 GPU 数据，而 glumpy 会处理其余部分。但一个例子胜过千言万语：

```py
from glumpy import gloo

    dtype = [("position", np.float32, 2),  # x,y
             ("color",    np.float32, 3)]  # r,g,b
    V = np.zeros((3,3),dtype).view(gloo.VertexBuffer)
    V["position"][0,0] = 0.0, 0.0
    V["position"][1,1] = 0.0, 0.0

```

`V` 是一个 `VertexBuffer`，它既是 `GPUData` 也是 numpy 数组。当 `V` 被修改时，glumpy 会负责计算自上次上传到 GPU 内存以来最小的连续脏内存块。当这个缓冲区要在 GPU 上使用时，glumpy 会负责在最后时刻上传“脏”区域。这意味着如果你从未使用 `V`，则永远不会上传任何内容到 GPU！在上面的例子中，最后计算的“脏”区域由从偏移量 0 开始的 88 字节组成，如下所示：

![img/51d636eb5c4a87df76ffe74319ca2360.png](img/51d636eb5c4a87df76ffe74319ca2360.png)

注意

当创建缓冲区时，它被标记为完全脏，但为了说明，这里就假装不是这种情况。

因此，glumpy 最终将上传 88 字节，而实际上只修改了 16 字节。你可能想知道这是否是最优的。实际上，大多数情况下是这样的，因为将一些数据上传到缓冲区需要在 GL 端进行大量操作，并且每次调用都有固定的成本。

### 数组子类

如 [ndarray 子类化](https://docs.scipy.org/doc/numpy/user/basics.subclassing.html) 文档中所述，`ndarray` 的子类化由于 `ndarray` 类的新实例可以通过三种不同的方式产生而变得复杂：

+   显式构造函数调用

+   视图转换

+   从模板新建

然而，我们的情况更简单，因为我们只对视图转换感兴趣。因此，我们只需要定义在每次实例创建时都会被调用的 `__new__` 方法。这样，`GPUData` 类将配备两个属性：

+   `extents`：这表示相对于基本数组的视图的完整范围。它存储为字节偏移量和字节大小。

+   `pending_data`：这表示相对于 `extents` 属性的连续 *脏* 区域，以（字节偏移量，字节大小）的形式。

```py
class GPUData(np.ndarray):
        def __new__(cls, *args, **kwargs):
            return np.ndarray.__new__(cls, *args, **kwargs)

        def __init__(self, *args, **kwargs):
            pass

        def __array_finalize__(self, obj):
            if not isinstance(obj, GPUData):
                self._extents = 0, self.size*self.itemsize
                self.__class__.__init__(self)
                self._pending_data = self._extents
            else:
                self._extents = obj._extents

```

### 计算范围

每次请求数组的部分视图时，我们需要在可以访问基本数组的同时计算这个部分视图的范围。

```py
def __getitem__(self, key):
        Z = np.ndarray.__getitem__(self, key)
        if not hasattr(Z,'shape') or Z.shape == ():
            return Z
        Z._extents = self._compute_extents(Z)
        return Z

    def _compute_extents(self, Z):
        if self.base is not None:
            base = self.base.__array_interface__['data'][0]
            view = Z.__array_interface__['data'][0]
            offset = view - base
            shape = np.array(Z.shape) - 1
            strides = np.array(Z.strides)
            size = (shape*strides).sum() + Z.itemsize
            return offset, offset+size
        else:
            return 0, self.size*self.itemsize

```

### 跟踪待处理数据

另一个额外的困难是我们不希望所有视图都跟踪脏区域，而只跟踪基本数组。这就是为什么在 `__array_finalize__` 方法的第二种情况下我们不实例化 `self._pending_data` 的原因。这将在我们需要更新某些数据时处理，例如在 `__setitem__` 调用期间：

```py
def __setitem__(self, key, value):
        Z = np.ndarray.__getitem__(self, key)
        if Z.shape == ():
            key = np.mod(np.array(key)+self.shape, self.shape)
            offset = self._extents[0]+(key * self.strides).sum()
            size = Z.itemsize
            self._add_pending_data(offset, offset+size)
            key = tuple(key)
        else:
            Z._extents = self._compute_extents(Z)
            self._add_pending_data(Z._extents[0], Z._extents[1])
        np.ndarray.__setitem__(self, key, value)

    def _add_pending_data(self, start, stop):
        base = self.base
        if isinstance(base, GPUData):
            base._add_pending_data(start, stop)
        else:
            if self._pending_data is None:
                self._pending_data = start, stop
            else:
                start = min(self._pending_data[0], start)
                stop = max(self._pending_data[1], stop)
                self._pending_data = start, stop

```

### 来源

+   gpudata.py

## 结论

如在 numpy 网站上所述，numpy 是 Python 科学计算的基础包。然而，正如本章所示，numpy 强度的使用远远超出了仅仅是一个 *通用数据的多维容器*。在一种情况下（`TypedList`）将 `ndarray` 作为私有属性，或在另一种情况下（`GPUData`）直接子类化 `ndarray` 类以跟踪内存，我们已经看到如何扩展 numpy 的功能以适应非常特定的需求。限制只在于你的想象力和经验。

# 超越 Numpy

**内容**

+   回到 Python

+   Numpy 及其相关

    +   NumExpr

    +   Cython

    +   Numba

    +   Theano

    +   PyCUDA

    +   PyOpenCL

+   Scipy 及其相关

    +   scikit-learn

    +   scikit-image

    +   SymPy

    +   Astropy

    +   Cartopy

    +   Brian

    +   Glumpy

+   结论

## 回到 Python

你几乎读完了这本书，希望你已经了解到 numpy 是一个非常灵活且强大的库。然而在此期间，记住 Python 也是一种相当强大的语言。实际上，在某些特定情况下，它可能比 numpy 更强大。让我们考虑一个有趣的练习，这是由 Tucker Balch 在他的 [Coursera 的计算投资课程](https://www.coursera.org/learn/computational-investing)中提出的。练习的编写如下：

> *编写尽可能简洁的代码来计算所有“合法”的 4 只股票的分配，使得分配在 1.0 块中，并且分配总和为 10.0。*

[Yaser Martinez](http://yasermartinez.com/blog/index.html) 收集了社区的不同答案和提出的解决方案，产生了令人惊讶的结果。但让我们从最明显的 Python 解决方案开始：

```py
def solution_1():
        # Brute force
        # 14641 (=11*11*11*11) iterations & tests
        Z = []
        for i in range(11):
            for j in range(11):
                for k in range(11):
                    for l in range(11):
                        if i+j+k+l == 10:
                            Z.append((i,j,k,l))
        return Z

```

这种解决方案是最慢的解决方案，因为它需要 4 个循环，更重要的是，它测试了 0 到 10 之间 4 个整数的所有不同组合（11641），以保留和为 10 的组合。我们当然可以使用 itertools 来消除 4 个循环，但代码仍然很慢：

```py
import itertools as it

    def solution_2():
        # Itertools
        # 14641 (=11*11*11*11) iterations & tests
        return [(i,j,k,l)
                for i,j,k,l in it.product(range(11),repeat=4) if i+j+k+l == 10]

```

Nick Popplas 提出的最佳解决方案之一利用了我们可以拥有智能嵌套循环的事实，这将允许我们直接构建每个元组，而无需任何测试，如下所示：

```py
def solution_3():
        return [(a, b, c, (10 - a - b - c))
                for a in range(11)
                for b in range(11 - a)
                for c in range(11 - a - b)]

```

Yaser Martinez 提供的最佳 numpy 解决方案采用了一种不同的策略，使用一组有限的测试：

```py
def solution_4():
        X123 = np.indices((11,11,11)).reshape(3,11*11*11)
        X4 = 10 - X123.sum(axis=0)
        return np.vstack((X123, X4)).T[X4 > -1]

```

如果我们对这些方法进行基准测试，我们得到：

```py
>>> timeit("solution_1()", globals())
    100 loops, best of 3: 1.9 msec per loop >>> timeit("solution_2()", globals())
    100 loops, best of 3: 1.67 msec per loop >>> timeit("solution_3()", globals())
    1000 loops, best of 3: 60.4 usec per loop >>> timeit("solution_4()", globals())
    1000 loops, best of 3: 54.4 usec per loop

```

numpy 解决方案是最快的，但纯 Python 解决方案也是可以比较的。但让我介绍一下 Python 解决方案的一个小修改：

```py
def solution_3_bis():
        return ((a, b, c, (10 - a - b - c))
                for a in range(11)
                for b in range(11 - a)
                for c in range(11 - a - b))

```

如果我们对其进行基准测试，我们得到：

```py
>>> timeit("solution_3_bis()", globals())
    10000 loops, best of 3: 0.643 usec per loop

```

你没看错，我们仅仅通过将方括号替换为圆括号就提高了 100 倍。这是怎么做到的？解释可以通过查看返回对象的类型来找到：

```py
>>> print(type(solution_3()))
    <class 'list'> >>> print(type(solution_3_bis()))
    <class 'generator'>

```

`solution_3_bis()` 返回一个生成器，可以用来生成完整的列表或遍历所有不同的元素。在任何情况下，巨大的速度提升都来自于不实例化完整的列表，因此重要的是要考虑你是否需要一个实际的结果实例，或者一个简单的生成器可能就足够了。

## Numpy & co

除了 numpy 之外，还有几个其他 Python 包值得一看，因为它们使用不同的技术（编译、虚拟机、即时编译、GPU、压缩等）解决了类似但不同类别的问题。根据您具体的问题和硬件，一个包可能比另一个包更好。让我们用一个非常简单的例子来说明它们的用法，其中我们想要根据两个浮点向量计算一个表达式：

```py
import numpy as np
    a = np.random.uniform(0, 1, 1000).astype(np.float32)
    b = np.random.uniform(0, 1, 1000).astype(np.float32)
    c = 2*a + 3*b

```

### NumExpr

[numexpr](https://github.com/pyhttps://www.labri.fr/perso/nrougier/from-python-to-numpy/data/numexpr/wiki/Numexpr-Users-Guide) 包提供了一组通过使用基于向量的虚拟机逐元素快速评估数组表达式的例程。它与 SciPy 的 weave 包类似，但不需要单独编译 C 或 C++ 代码的步骤。

```py
import numpy as np
    import numexpr as ne

    a = np.random.uniform(0, 1, 1000).astype(np.float32)
    b = np.random.uniform(0, 1, 1000).astype(np.float32)
    c = ne.evaluate("2*a + 3*b")

```

### Cython

[Cython](http://cython.org) 是一个针对 Python 编程语言及其扩展 Cython 编程语言（基于 Pyrex）的优化静态编译器。它使得编写 Python 的 C 扩展变得和 Python 本身一样简单。

```py
import numpy as np

    def evaluate(np.ndarray a, np.ndarray b):
        cdef int i
        cdef np.ndarray c = np.zeros_like(a)
        for i in range(a.size):
            c[i] = 2*a[i] + 3*b[i]
        return c

    a = np.random.uniform(0, 1, 1000).astype(np.float32)
    b = np.random.uniform(0, 1, 1000).astype(np.float32)
    c = evaluate(a, b)

```

### Numba

[Numba](http://numba.pydata.org) 为您提供了使用直接在 Python 中编写的性能函数来加速应用程序的能力。通过一些注解，面向数组的和数学密集型的 Python 代码可以被即时编译成原生机器指令，其性能与 C、C++ 和 Fortran 相似，而无需切换语言或 Python 解释器。

```py
from numba import jit
    import numpy as np

    @jit
    def evaluate(np.ndarray a, np.ndarray b):
        c = np.zeros_like(a)
        for i in range(a.size):
            c[i] = 2*a[i] + 3*b[i]
        return c

    a = np.random.uniform(0, 1, 1000).astype(np.float32)
    b = np.random.uniform(0, 1, 1000).astype(np.float32)
    c = evaluate(a, b)

```

### Theano

[Theano](http://www.deeplearning.net/software/theano/) 是一个 Python 库，允许您高效地定义、优化和评估涉及多维数组的数学表达式。Theano 具有与 numpy 的紧密集成、透明地使用 GPU、高效的符号微分、速度和稳定性优化、动态 C 代码生成以及广泛的单元测试和自我验证功能。

```py
import numpy as np
    import theano.tensor as T

    x = T.fvector('x')
    y = T.fvector('y')
    z = 2*x + 3*y
    f = function([x, y], z)

    a = np.random.uniform(0, 1, 1000).astype(np.float32)
    b = np.random.uniform(0, 1, 1000).astype(np.float32)
    c = f(a, b)

```

### PyCUDA

[PyCUDA](http://mathema.tician.de/software/pycuda) 允许您从 Python 访问 Nvidia 的 CUDA 并行计算 API。

```py
import numpy as np
    import pycuda.autoinit
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule

    mod = SourceModule("""
        __global__ void evaluate(float *c, float *a, float *b)
        {
          const int i = threadIdx.x;
          c[i] = 2*a[i] + 3*b[i];
        }
    """)

    evaluate = mod.get_function("evaluate")

    a = np.random.uniform(0, 1, 1000).astype(np.float32)
    b = np.random.uniform(0, 1, 1000).astype(np.float32)
    c = np.zeros_like(a)

    evaluate(drv.Out(c), drv.In(a), drv.In(b), block=(400,1,1), grid=(1,1))

```

### PyOpenCL

[PyOpenCL](http://mathema.tician.de/software/pyopencl) 允许您从 Python 访问 GPU 和其他大规模并行计算设备。

```py
import numpy as np
    import pyopencl as cl

    a = np.random.uniform(0, 1, 1000).astype(np.float32)
    b = np.random.uniform(0, 1, 1000).astype(np.float32)
    c = np.empty_like(a)

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    gpu_a = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    gpu_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)

    evaluate = cl.Program(ctx, """
        __kernel void evaluate(__global const float *gpu_a;
                               __global const float *gpu_b;
                               __global       float *gpu_c)
        {
            int gid = get_global_id(0);
            gpu_c[gid] = 2*gpu_a[gid] + 3*gpu_b[gid];
        }
    """).build()

    gpu_c = cl.Buffer(ctx, mf.WRITE_ONLY, a.nbytes)
    evaluate.evaluate(queue, a.shape, None, gpu_a, gpu_b, gpu_c)
    cl.enqueue_copy(queue, c, gpu_c)

```

## Scipy & co

如果有多个针对 numpy 的附加包，那么就有成千上万的 scipy 附加包。实际上，科学领域的每个领域可能都有自己的包，而且我们迄今为止所研究的多数例子都可以通过调用相关包中的方法来解决。但当然，这并不是目标，如果你有一些空闲时间，自己编程通常是一个很好的练习。目前最大的困难是找到这些相关包。以下是一个非常简短的、维护良好、经过良好测试的包列表，这些包可能会简化你的科学生活（取决于你的领域）。当然，还有很多其他的包，根据你的具体需求，你可能不需要自己编写所有代码。要查看一个详尽的列表，请查看 [Awesome python 列表](https://awesome-python.com)。

### scikit-learn

[scikit-learn](http://scikit-learn.org/stable/) 是一个用于 Python 编程语言的免费软件机器学习库。它包含各种分类、回归和聚类算法，包括支持向量机、随机森林、梯度提升、k-means 和 DBSCAN，并且设计为与 Python 的数值和科学库 numpy 和 SciPy 互操作。

### scikit-image

[scikit-image](http://scikit-image.org) 是一个专注于图像处理的 Python 包，它使用原生的 numpy 数组作为图像对象。本章描述了如何在各种图像处理任务中使用 scikit-image，并强调了与其他科学 Python 模块（如 numpy 和 SciPy）之间的联系。

### SymPy

[SymPy](http://www.sympy.org/en/index.html) 是一个用于符号数学的 Python 库。它的目标是成为一个功能齐全的计算机代数系统（CAS），同时保持代码尽可能简单，以便于理解并易于扩展。SymPy 完全用 Python 编写。

### Astropy

[Astropy](http://www.astropy.org) 项目是一个社区努力，旨在为 Python 开发一个单一的核心天文包，并促进 Python 天文包之间的互操作性。

### Cartopy

[Cartopy](http://scitools.org.uk/cartopy/)是一个 Python 包，旨在使绘制用于数据分析可视化的地图尽可能容易。Cartopy 利用了强大的 PROJ.4、numpy 和 shapely 库，并为 matplotlib 提供了一个简单直观的绘图界面，用于创建高质量的地图。

### Brian

[Brian](http://www.briansimulator.org)是一个免费的开源模拟器，用于模拟突触神经网络。它用 Python 编程语言编写，几乎在所有平台上都可用。我们相信，模拟器不仅应该节省处理器的处理时间，还应该节省科学家的研究时间。因此，Brian 被设计成易于学习和使用，高度灵活且易于扩展。

### Glumpy

[Glumpy](http://glumpy.github.io)是一个基于 OpenGL 的 Python 交互式可视化库。它的目标是使创建快速、可扩展、美观、交互式和动态的可视化变得容易。该网站的主要文档组织为几个部分：

## 结论

Numpy 是一个非常通用的库，但并不意味着你必须在每种情况下都使用它。在本章中，我们看到了一些值得一看的替代方案（包括 Python 本身）。一如既往，选择权在你。你必须考虑对你来说，在开发时间、计算时间和维护努力方面，哪种解决方案是最好的。一方面，如果你设计自己的解决方案，你将不得不对其进行测试和维护，但作为交换，你将能够自由地按照自己的意愿来设计它。另一方面，如果你决定依赖第三方包，你将在开发上节省时间，并从社区支持中受益，尽管你可能需要根据你的特定需求调整包。选择权在你。

# 结论

你已经到达了这本书的结尾。我希望你在阅读过程中学到了一些东西，我写作时确实学到了很多。试图解释某件事通常是一个很好的练习，可以检验你对这个知识的掌握。当然，我们只是触及了 numpy 的表面，还有很多东西等待去发现。请查看由真正的专家撰写的书籍目录，查看由制作 numpy 的人撰写的文档，并且不要犹豫在邮件列表上提出你的问题，因为 numpy 社区非常友好。

如果这本书中有一个信息需要保留，那就是“过早优化是万恶之源”。我们已经看到，代码向量化可以极大地提高你的计算效率，在某些情况下甚至可以提升几个数量级。然而，问题向量化通常更加强大。如果你在设计过程中过早地使用代码向量化，你就无法跳出思维定式，你肯定会错过一些真正强大的替代方案，因为你无法像我们在问题向量化章节中看到的那样正确地识别你的问题。这需要一些经验，你必须有耐心：经验不是一蹴而就的过程。

最后，一旦你考虑了 NumPy 的替代方案，自定义向量化是一个值得考虑的选项。当你发现没有任何方法可行时，NumPy 仍然提供了一个巧妙的框架来锻造你自己的工具。而且谁知道呢，这可能是你和你所在社区的一次激动人心的冒险的开始，就像它发生在我与 [glumpy](http://glumpy.github.io) 和 [vispy](http://vispy.org) 软件包一样。

# 快速参考

**内容**

+   数据类型

+   创建

+   索引

+   重塑

+   广播

## 数据类型

| 类型 | 名称 | 字节数 | 描述 |
| --- | --- | --- | --- |
| `bool` | `b` | 1 | 布尔值（真或假），存储为一个字节 |
| `int` | `l` | 4-8 | 平台（长）整数（通常是 int32 或 int64） |
| `intp` | `p` | 4-8 | 用于索引的整数（通常是 int32 或 int64） |
| `int8` | `i1` | 1 | 字节（-128 到 127） |
| `int16` | `i2` | 2 | 整数（-32768 到 32767） |
| `int32` | `i4` | 4 | 整数（-2147483648 到 2147483647） |
| `int64` | `i8` | 8 | 整数（-9223372036854775808 到 9223372036854775807） |
| `uint8` | `u1` | 1 | 无符号整数（0 到 255） |
| `uint16` | `u2` | 2 | 无符号整数（0 到 65535） |
| `uint32` | `u4` | 4 | 无符号整数（0 到 4294967295） |
| `uint64` | `u8` | 8 | 无符号整数（0 到 18446744073709551615） |
| `float` | `f8` | 8 | 简写为 float64 |
| `float16` | `f2` | 2 | 半精度浮点数：符号位，5 位指数，10 位尾数 |
| `float32` | `f` | 4 | 单精度浮点数：符号位，8 位指数，23 位尾数 |
| `float64` | `d` | 8 | 双精度浮点数：符号位，11 位指数，52 位尾数 |
| `complex` | `c16` | 16 | 简写为 complex128。 |
| `complex64` | `c8` | 8 | 复数，由两个 32 位浮点数表示 |
| `complex128` | `c16` | 16 | 复数，由两个 64 位浮点数表示 |

`bool`、`int`、`float` 和 `complex` 在 NumPy 中被理解，但名称为 `np.bool_`，额外有一个下划线。此外，在 C 编程语言中使用的名称，如 `intc`、`long` 或 `double`，也被定义。

## 创建

```py
Z = np.zeros(9)

```

```py
┌───┬───┬───┬───┬───┬───┬───┬───┬───┐
    │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │
    └───┴───┴───┴───┴───┴───┴───┴───┴───┘

```

```py
Z = np.ones(9)

```

```py
┌───┬───┬───┬───┬───┬───┬───┬───┬───┐
    │ 1 │ 1 │ 1 │ 1 │ 1 │ 1 │ 1 │ 1 │ 1 │
    └───┴───┴───┴───┴───┴───┴───┴───┴───┘

```

```py
Z = np.array([1,0,0,0,0,0,0,1,0])

```

```py
┌───┬───┬───┬───┬───┬───┬───┬───┬───┐
    │ 1 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 1 │ 0 │
    └───┴───┴───┴───┴───┴───┴───┴───┴───┘

```

```py
Z = 2*np.ones(9)

```

```py
┌───┬───┬───┬───┬───┬───┬───┬───┬───┐
    │ 2 │ 2 │ 2 │ 2 │ 2 │ 2 │ 2 │ 2 │ 2 │
    └───┴───┴───┴───┴───┴───┴───┴───┴───┘

```

```py
Z = np.arange(9)

```

```py
┌───┬───┬───┬───┬───┬───┬───┬───┬───┐
    │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │
    └───┴───┴───┴───┴───┴───┴───┴───┴───┘

```

```py
Z = np.arange(9).reshape(9,1)

```

```py
┌───┐
    │ 0 │
    ├───┤
    │ 1 │
    ├───┤
    │ 2 │
    ├───┤
    │ 3 │
    ├───┤
    │ 4 │
    ├───┤
    │ 5 │
    ├───┤
    │ 6 │
    ├───┤
    │ 7 │
    ├───┤
    │ 8 │
    └───┘

```

```py
Z = np.arange(9).reshape(3,3)

```

```py
┌───┬───┬───┐
    │ 0 │ 1 │ 2 │
    ├───┼───┼───┤
    │ 3 │ 4 │ 5 │
    ├───┼───┼───┤
    │ 6 │ 7 │ 8 │
    └───┴───┴───┘

```

```py
Z = np.random.randint(0,9,(3,3))

```

```py
┌───┬───┬───┐
    │ 4 │ 5 │ 7 │
    ├───┼───┼───┤
    │ 0 │ 2 │ 6 │
    ├───┼───┼───┤
    │ 8 │ 4 │ 0 │
    └───┴───┴───┘

```

```py
Z = np.linspace(0, 1, 5)

```

```py
┌──────┬──────┬──────┬──────┬──────┐
    │ 0.00 │ 0.25 │ 0.50 │ 0.75 │ 1.00 │
    └──────┴──────┴──────┴──────┴──────┘

```

```py
np.grid[0:3,0:3]

```

```py
┌───┬───┬───┐   ┌───┬───┬───┐
    │ 0 │ 0 │ 0 │   │ 0 │ 1 │ 2 │
    ├───┼───┼───┤   ├───┼───┼───┤
    │ 1 │ 1 │ 1 │   │ 0 │ 1 │ 2 │
    ├───┼───┼───┤   ├───┼───┼───┤
    │ 2 │ 2 │ 2 │   │ 0 │ 1 │ 2 │
    └───┴───┴───┘   └───┴───┴───┘

```

## 索引

```py
Z = np.arange(9).reshape(3,3)
    Z[0,0]

```

```py
┏━━━┓───┬───┐   ┏━━━┓
    ┃ 0 ┃ 1 │ 2 │ → ┃ 0 ┃ (scalar)
    ┗━━━┛───┼───┤   ┗━━━┛
    │ 3 │ 4 │ 5 │
    ├───┼───┼───┤
    │ 6 │ 7 │ 8 │
    └───┴───┴───┘

```

```py
Z = np.arange(9).reshape(3,3)
    Z[-1,-1]

```

```py
┌───┬───┬───┐
    │ 0 │ 1 │ 2 │
    ├───┼───┼───┤
    │ 3 │ 4 │ 5 │
    ├───┼───┏━━━┓   ┏━━━┓
    │ 6 │ 7 ┃ 8 ┃ → ┃ 8 ┃ (scalar)
    └───┴───┗━━━┛   ┗━━━┛

```

```py
Z = np.arange(9).reshape(3,3)
    Z[1]

```

```py
┌───┬───┬───┐
    │ 0 │ 1 │ 2 │
    ┏━━━┳━━━┳━━━┓   ┏━━━┳━━━┳━━━┓
    ┃ 3 ┃ 4 ┃ 5 ┃ → ┃ 3 ┃ 4 ┃ 5 ┃
    ┗━━━┻━━━┻━━━┛   ┗━━━┻━━━┻━━━┛
    │ 6 │ 7 │ 8 │      (view)
    └───┴───┴───┘

```

```py
Z = np.arange(9).reshape(3,3)
    Z[:,2]

```

```py
┌───┬───┏━━━┓   ┏━━━┓
    │ 0 │ 1 ┃ 2 ┃   ┃ 2 ┃
    ├───┼───┣━━━┫   ┣━━━┫
    │ 3 │ 4 ┃ 5 ┃ → ┃ 5 ┃ (view)
    ├───┼───┣━━━┫   ┣━━━┫
    │ 6 │ 7 ┃ 8 ┃   ┃ 8 ┃
    └───┴───┗━━━┛   ┗━━━┛

```

```py
Z = np.arange(9).reshape(3,3)
    Z[1:,1:]

```

```py
┌───┬───┬───┐
    │ 0 │ 1 │ 2 │    (view)
    ├───┏━━━┳━━━┓   ┏━━━┳━━━┓
    │ 3 ┃ 4 ┃ 5 ┃   ┃ 4 ┃ 5 ┃
    ├───┣━━━╋━━━┫ → ┣━━━╋━━━┫
    │ 6 ┃ 7 ┃ 8 ┃   ┃ 7 ┃ 8 ┃
    └───┗━━━┻━━━┛   ┗━━━┻━━━┛

```

```py
Z = np.arange(9).reshape(3,3)
    Z[::2,::2]

```

```py
┏━━━┓───┏━━━┓   ┏━━━┳━━━┓
    ┃ 0 ┃ 1 ┃ 2 ┃   ┃ 0 ┃ 2 ┃
    ┗━━━┛───┗━━━┛ → ┣━━━╋━━━┫
    │ 3 │ 4 │ 5 │   ┃ 6 ┃ 8 ┃
    ┏━━━┓───┏━━━┓   ┗━━━┻━━━┛
    ┃ 6 ┃ 7 ┃ 8 ┃    (view)
    ┗━━━┛───┗━━━┛

```

```py
Z = np.arange(9).reshape(3,3)
    Z[[0,1],[0,2]]

```

```py
┏━━━┓───┬───┐
    ┃ 0 ┃ 1 │ 2 │
    ┗━━━┛───┏━━━┓   ┏━━━┳━━━┓
    │ 3 │ 4 ┃ 5 ┃ → ┃ 0 ┃ 5 ┃
    ├───┼───┗━━━┛   ┗━━━┻━━━┛
    │ 6 │ 7 │ 8 │    (copy)
    └───┴───┴───┘

```

## 重塑

```py
Z = np.array([0,0,0,0,0,0,0,0,0,0,1,0])

```

```py
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┏━━━┓───┐
    │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 ┃ 1 ┃ 0 │
    └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┗━━━┛───┘

```

```py
Z = np.array([0,0,0,0,0,0,0,0,0,0,1,0]).reshape(12,1)

```

```py
┌───┐
    │ 0 │
    ├───┤
    │ 0 │
    ├───┤
    │ 0 │
    ├───┤
    │ 0 │
    ├───┤
    │ 0 │
    ├───┤
    │ 0 │
    ├───┤
    │ 0 │
    ├───┤
    │ 0 │
    ├───┤
    │ 0 │
    ├───┤
    │ 0 │
    ┏━━━┓
    ┃ 1 ┃
    ┗━━━┛
    │ 0 │
    └───┘

```

```py
Z = np.array([0,0,0,0,0,0,0,0,0,0,1,0]).reshape(3,4)

```

```py
┌───┬───┬───┬───┐
    │ 0 │ 0 │ 0 │ 0 │
    ├───┼───┼───┼───┤
    │ 0 │ 0 │ 0 │ 0 │
    ├───┼───┏━━━┓───┤
    │ 0 │ 0 ┃ 1 ┃ 0 │
    └───┴───┗━━━┛───┘

```

```py
Z = np.array([0,0,0,0,0,0,0,0,0,0,1,0]).reshape(4,3)

```

```py
┌───┬───┬───┐
    │ 0 │ 0 │ 0 │
    ├───┼───┼───┤
    │ 0 │ 0 │ 0 │
    ├───┼───┼───┤
    │ 0 │ 0 │ 0 │
    ├───┏━━━┓───┤
    │ 0 ┃ 1 ┃ 0 │
    └───┗━━━┛───┘

```

```py
Z = np.array([0,0,0,0,0,0,0,0,0,0,1,0]).reshape(6,2)

```

```py
┌───┬───┐
    │ 0 │ 0 │
    ├───┼───┤
    │ 0 │ 0 │
    ├───┼───┤
    │ 0 │ 0 │
    ├───┼───┤
    │ 0 │ 0 │
    ├───┼───┤
    │ 0 │ 0 │
    ┏━━━┓───┤
    ┃ 1 ┃ 0 │
    ┗━━━┛───┘

```

```py
Z = np.array([0,0,0,0,0,0,0,0,0,0,1,0]).reshape(2,6)

```

```py
┌───┬───┬───┬───┬───┬───┐
    │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │
    ├───┼───┼───┼───┏━━━┓───┤
    │ 0 │ 0 │ 0 │ 0 ┃ 1 ┃ 0 │
    └───┴───┴───┴───┗━━━┛───┘

```

## 广播

```py
Z1 = np.arange(9).reshape(3,3)
    Z2 = 1
    Z1 + Z2

```

```py
┌───┬───┬───┐   ┌───┐   ┌───┬───┬───┐   ┏━━━┓───┬───┐   ┌───┬───┬───┐
    │ 0 │ 1 │ 2 │ + │ 1 │ = │ 0 │ 1 │ 2 │ + ┃ 1 ┃ 1 │ 1 │ = │ 1 │ 2 │ 3 │
    ├───┼───┼───┤   └───┘   ├───┼───┼───┤   ┗━━━┛───┼───┤   ├───┼───┼───┤
    │ 3 │ 4 │ 5 │           │ 3 │ 4 │ 5 │   │ 1 │ 1 │ 1 │   │ 4 │ 5 │ 6 │
    ├───┼───┼───┤           ├───┼───┼───┤   ├───┼───┼───┤   ├───┼───┼───┤
    │ 6 │ 7 │ 8 │           │ 6 │ 7 │ 8 │   │ 1 │ 1 │ 1 │   │ 7 │ 8 │ 9 │
    └───┴───┴───┘           └───┴───┴───┘   └───┴───┴───┘   └───┴───┴───┘

```

```py
Z1 = np.arange(9).reshape(3,3)
    Z2 = np.arange(3)[::-1].reshape(3,1)
    Z1 + Z2

```

```py
┌───┬───┬───┐   ┌───┐   ┌───┬───┬───┐   ┏━━━┓───┬───┐   ┌───┬───┬───┐
    │ 0 │ 1 │ 2 │ + │ 2 │ = │ 0 │ 1 │ 2 │ + ┃ 2 ┃ 2 │ 2 │ = │ 2 │ 3 │ 4 │
    ├───┼───┼───┤   ├───┤   ├───┼───┼───┤   ┣━━━┫───┼───┤   ├───┼───┼───┤
    │ 3 │ 4 │ 5 │   │ 1 │   │ 3 │ 4 │ 5 │   ┃ 1 ┃ 1 │ 1 │   │ 4 │ 5 │ 6 │
    ├───┼───┼───┤   ├───┤   ├───┼───┼───┤   ┣━━━┫───┼───┤   ├───┼───┼───┤
    │ 6 │ 7 │ 8 │   │ 0 │   │ 6 │ 7 │ 8 │   ┃ 0 ┃ 0 │ 0 │   │ 6 │ 7 │ 8 │
    └───┴───┴───┘   └───┘   └───┴───┴───┘   ┗━━━┛───┴───┘   └───┴───┴───┘

```

```py
Z1 = np.arange(9).reshape(3,3)
    Z2 = np.arange(3)[::-1]
    Z1 + Z2

```

```py
┌───┬───┬───┐   ┌───┬───┬───┐   ┌───┬───┬───┐   ┏━━━┳━━━┳━━━┓   ┌───┬───┬───┐
    │ 0 │ 1 │ 2 │ + │ 2 │ 1 │ 0 │ = │ 0 │ 1 │ 2 │ + ┃ 2 ┃ 1 ┃ 0 ┃ = │ 2 │ 2 │ 2 │
    ├───┼───┼───┤   └───┴───┴───┘   ├───┼───┼───┤   ┗━━━┻━━━┻━━━┛   ├───┼───┼───┤
    │ 3 │ 4 │ 5 │                   │ 3 │ 4 │ 5 │   │ 2 │ 1 │ 0 │   │ 5 │ 5 │ 5 │
    ├───┼───┼───┤                   ├───┼───┼───┤   ├───┼───┼───┤   ├───┼───┼───┤
    │ 6 │ 7 │ 8 │                   │ 6 │ 7 │ 8 │   │ 2 │ 1 │ 0 │   │ 8 │ 8 │ 8 │
    └───┴───┴───┘                   └───┴───┴───┘   └───┴───┴───┘   └───┴───┴───┘

```

```py
Z1 = np.arange(3).reshape(3,1)
    Z2 = np.arange(3).reshape(1,3)
    Z1 + Z2

```

```py
┌───┐   ┌───┬───┬───┐   ┏━━━┓───┬───┐   ┏━━━┳━━━┳━━━┓   ┌───┬───┬───┐
    │ 0 │ + │ 0 │ 1 │ 2 │ = ┃ 0 ┃ 0 │ 0 │ + ┃ 0 ┃ 1 ┃ 2 ┃ = │ 0 │ 1 │ 2 │
    ├───┤   └───┴───┴───┘   ┣━━━┫───┼───┤   ┗━━━┻━━━┻━━━┛   ├───┼───┼───┤
    │ 1 │                   ┃ 1 ┃ 1 │ 1 │   │ 0 │ 1 │ 2 │   │ 1 │ 2 │ 3 │
    ├───┤                   ┣━━━┫───┼───┤   ├───┼───┼───┤   ├───┼───┼───┤
    │ 2 │                   ┃ 2 ┃ 2 │ 2 │   │ 0 │ 1 │ 2 │   │ 2 │ 3 │ 4 │
    └───┘                   ┗━━━┛───┴───┘   └───┴───┴───┘   └───┴───┴───┘

```

# 参考文献

这是一个精选的 numpy 相关资源列表（文章、书籍和教程），涵盖了 numpy 的不同方面。其中一些非常特定于 numpy/Scipy，而另一些则提供了对数值计算的更广泛视角。

**内容**

+   教程

+   文章

+   书籍

## 教程

+   [100 Numpy 练习](http://www.labri.fr/perso/nrougier/teaching/numpy.100/index.html), Nicolas P. Rougier, 2016.

+   [NumPy 教程](http://www.labri.fr/perso/nrougier/teaching/numpy/numpy.html), Nicolas P. Rougier, 2015。

+   [Python 课程](http://www.python-course.eu/numpy.php), Bernd Klein, 2015.

+   [NumPy 和 Scipy 简介](https://engineering.ucsb.edu/~shell/che210d/numpy.pdf), M. Scott Shell, 2014。

+   [Python Numpy 教程](http://cs231n.github.io/python-numpy-tutorial/), Justin Johnson, 2014.

+   [快速入门教程](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html), Numpy 开发者, 2009.

+   [NumPy 工具包](http://mentat.za.net/numpy/numpy_advanced_slides/), Stéfan van der Walt, 2008。

## 文章

+   [NumPy 数组：高效数值计算的结构](https://hal.inria.fr/inria-00564007/document)Stéfan van der Walt, Chris Colbert & Gael Varoquaux, 计算科学工程，13(2)，2011。

    在 Python 世界中，NumPy 数组是数值数据的标准表示形式，并允许在高级语言中高效实现数值计算。正如这个努力所显示的，可以通过三种技术来提高 NumPy 的性能：向量化计算、避免在内存中复制数据以及最小化操作次数。

+   [用于突触神经网络模拟的矢量化算法](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.397.6097)Romain Brette & Dan F. M. Goodman, Neural Computation, 23(6), 2010.

    高级语言（Matlab、Python）在神经科学中很受欢迎，因为它们灵活且能加速开发。然而，对于模拟突触神经网络，解释的开销是一个瓶颈。我们描述了一组算法，使用基于向量的操作，以高效的方式使用高级语言模拟大型突触神经网络。这些算法构成了用 Python 语言编写的突触神经网络模拟器 Brian 的核心。向量化模拟使得将高级语言的灵活性与通常与编译语言相关的计算效率结合起来成为可能。

+   [科学计算中的 Python](http://dl.acm.org/citation.cfm?id=1251830)Travis E. Oliphant, 计算科学工程，9(3)，2007。

    仅就 Python 本身而言，它是一种优秀的“引导”语言，适用于用其他语言编写的科学代码。然而，通过添加一些基本工具，Python 可以转变为适合科学和工程代码的高级语言，这种语言通常足够快，可以立即使用，同时足够灵活，可以通过额外的扩展来加速。

## 书籍

+   [SciPy 讲义](http://www.scipy-lectures.org)，Gaël Varoquaux，Emmanuelle Gouillart，Olav Vahtras 等人，2016。

    一份学习使用 Python 进行数值计算、科学和数据的文档。关于科学 Python 生态系统的教程：对核心工具和技术进行快速介绍。不同的章节各自对应 1 到 2 小时的课程，从入门到专家级别。

+   [Python 数据科学手册](http://shop.oreilly.com/product/0636920034919.do)Jake van der Plas, O'Reilly, 2016.

    《Python 数据科学手册》提供了对数据密集型科学、研究和发现中核心的计算和统计方法的参考。具有编程背景并希望有效地使用 Python 进行数据科学任务的人将学习如何面对各种问题：例如，你如何将这种数据格式读入你的脚本？你如何操作、转换和清理这些数据？你如何使用这些数据来获得洞察力、回答问题或构建统计或机器学习模型？

+   [优雅的 SciPy：科学 Python 的艺术](http://shop.oreilly.com/product/0636920038481.do)Juan Nunez-Iglesias，Stéfan van der Walt，Harriet Dashnow，O'Reilly，2016。

    欢迎来到科学 Python 及其社区！通过这本实用的书，你将学习 SciPy 及其相关库的基本部分，并品尝到易于阅读的漂亮代码，你可以在实践中使用。越来越多的科学家在编程，SciPy 库就在这里帮助你。找到有用的函数并正确、高效、易于阅读地使用它们是两件非常不同的事情。你将通过一些最好的代码示例进行学习，这些代码被选中以涵盖 SciPy 和相关库的广泛范围——包括 scikit-learn、scikit-image、toolz 和 pandas。

+   [学习 IPython 进行交互式计算和数据可视化](https://www.packtpub.com/big-data-and-business-intelligence/learning-ipython-interactive-computing-and-data-visualization-sec)Cyrille Rossant，Packt Publishing，2015。

    这本书是 Python 数据分析平台的入门友好指南。在介绍 Python 语言、IPython 和 Jupyter Notebook 之后，你将学习如何在现实世界的例子中分析和可视化数据，如何在 Notebook 中为图像处理创建图形用户界面，以及如何使用 NumPy、Numba、Cython 和 ipyparallel 进行快速数值计算以进行科学模拟。到本书结束时，你将能够对各种数据进行深入分析。

+   [SciPy 和 NumPy](https://www.safaribooksonline.com/library/view/scipy-and-numpy/9781449361600/)Eli Bressert，O'Reilly Media, Inc.，2012

    你是 SciPy 和 NumPy 的新手吗？你想通过示例和简洁的介绍快速轻松地学习它们吗？那么这本书就是为你准备的。你将能够穿透在线文档的复杂性，发现如何轻松地掌握这些 Python 库。

+   [数据分析中的 Python](http://shop.oreilly.com/product/0636920023784.do) Wes McKinney, O'Reilly Media, Inc., 2012.

    在寻找关于如何在 Python 中操作、处理、清洗和计算结构化数据的完整指南吗？这本实践性强的书籍包含了大量的实际案例研究，展示了如何有效地解决一系列数据分析问题，使用多个 Python 库。

+   [NumPy 指南](http://csc.ucdavis.edu/~chaos/courses/nlp/Software/NumPyBook.pdf) Travis Oliphant, 2006

    本书仅简要概述了 NumPy 中基本对象周围的一些基础设施，以提供包含在较旧的 Numeric 包中的附加功能（例如，线性代数、随机数组、FFT）。NumPy 中的这个基础设施包括基本的线性代数例程、傅里叶变换能力和随机数生成器。此外，f2py 模块在其自己的文档中有描述，因此在书的第二部分中仅简要提及。
