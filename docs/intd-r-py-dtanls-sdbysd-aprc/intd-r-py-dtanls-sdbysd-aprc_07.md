# 第四章 Numpy 数组与 R 的矩阵和数组类型

> 原文：[`randpythonbook.netlify.app/numpy-ndarrays-versus-rs-matrix-and-array-types`](https://randpythonbook.netlify.app/numpy-ndarrays-versus-rs-matrix-and-array-types)

有时你想要一个所有元素都是同一类型的元素集合，但你希望将它们存储在二维或三维结构中。例如，假设你需要使用矩阵乘法来编写一些线性回归软件，或者你需要使用张量来完成你正在进行的计算机视觉项目。

## 4.1 Python 中的 Numpy `ndarray`

在 Python 中，你仍然可以使用数组来完成这些任务。你将很高兴地了解到我们之前讨论的 Numpy `array`s 是 [Numpy 的 N 维数组](https://numpy.org/doc/stable/reference/arrays.ndarray.html) 的一个特例。每个数组都将附带大量的 [方法](https://numpy.org/doc/stable/reference/arrays.ndarray.html#array-methods) 和 [属性](https://numpy.org/doc/stable/reference/arrays.ndarray.html#array-attributes)（更多关于面向对象编程的内容请见第十四章）。下面演示了一些。

```py
import numpy as np
a = np.array([[1,2],[3,4]], np.float)
a
## array([[1., 2.],
##        [3., 4.]])
a.shape
## (2, 2)
a.ndim
## 2
a.dtype
## dtype('float64')
a.max()
## 4.0
a.resize((1,4)) # modification is **in place**
a
## array([[1., 2., 3., 4.]])
```

矩阵和逐元素乘法也非常有用。

```py
b = np.ones(4).reshape((4,1)) 
np.dot(b,a) # matrix mult.
## array([[1., 2., 3., 4.],
##        [1., 2., 3., 4.],
##        [1., 2., 3., 4.],
##        [1., 2., 3., 4.]])
b @ a # infix matrix mult. from PEP 465
## array([[1., 2., 3., 4.],
##        [1., 2., 3., 4.],
##        [1., 2., 3., 4.],
##        [1., 2., 3., 4.]])
a * np.arange(4) # elementwise mult.
## array([[ 0.,  2.,  6., 12.]])
```

我应该提到，Numpy 中还有一个 `matrix` 类型；然而，本文中并未对其进行描述，因为这更倾向于使用 Numpy `array`s（Albon 2018）。

在 R 和 Python 中，都有 `matrix` 类型和 `array` 类型。在 R 中，更常见的是使用 `matrix`s 而不是 `array`s，而在 Python 中则相反！

## 4.2 R 中的 `matrix` 和 `array` 类

在 Python 中，向你的“容器”添加维度很简单。你继续使用 Numpy 数组，只需更改 `.shape` 属性（可能通过调用 `.reshape()` 或类似的方法）。在 R 中，1 维、2 维和 3 维容器之间的区别更为明显。每个都有自己的类。存储相同类型对象的二维容器属于 `matrix` 类。具有 3 个或更多维度的容器属于 `array` 类⁷。在本节中，我将简要介绍如何使用这两个类。更多信息，请参阅 (Matloff 2011) 的第三章。

就像 `vector`s 一样，`matrix` 对象不一定要用于执行矩阵运算。是的，它们要求所有元素都是同一类型，但将包含 `character`s 的 `matrix` 对象“相乘”实际上并没有太多意义。

我通常使用 `matrix()` 函数或 `as.matrix()` 函数创建 `矩阵` 对象。在我看来，`matrix()` 更可取。第一个参数明确是一个包含你想要在 `矩阵` 中包含的所有展平数据的 `向量`。另一方面，`as.matrix()` 更灵活；它接受各种 R 对象（例如 `data.frame`），并尝试根据具体情况确定如何处理它们。换句话说，`as.matrix()` 是一个 *通用函数*。有关通用函数的更多信息，请参阅 14.2.2。

与 `matrix()` 相关的一些其他注意事项：`byrow=` 默认为 `FALSE`，如果你想要任何不是 1 列 `矩阵` 的东西，你还需要指定 `ncol=` 和/或 `nrow=`。

```py
A <-  matrix(1:4)
A
##      [,1]
## [1,]    1
## [2,]    2
## [3,]    3
## [4,]    4
matrix(1:4, ncol = 2)
##      [,1] [,2]
## [1,]    1    3
## [2,]    2    4
matrix(1:4, ncol = 2, byrow = T)
##      [,1] [,2]
## [1,]    1    2
## [2,]    3    4
as.matrix(
 data.frame(
 firstCol = c(1,2,3),
 secondCol = c("a","b","c"))) # coerces numbers to characters!
##      firstCol secondCol
## [1,] "1"      "a" 
## [2,] "2"      "b" 
## [3,] "3"      "c"
dim(A)
## [1] 4 1
nrow(A)
## [1] 4
ncol(A)
## [1] 1
```

`array()` 用于创建 `数组` 对象。这种类型的使用频率低于 `matrix` 类型，但这并不意味着你应该避免学习它。这主要反映了人们更喜欢处理的数据集类型，以及矩阵代数通常比张量代数更容易理解。然而，你将无法永远避免 3 维数据集（3 维，而不是 3 列 `矩阵`），尤其是如果你在神经影像学或计算机视觉等领域工作。

```py
myArray <-  array(rep(1:3, each = 4), dim = c(2,2,3))
myArray
## , , 1
## 
##      [,1] [,2]
## [1,]    1    1
## [2,]    1    1
## 
## , , 2
## 
##      [,1] [,2]
## [1,]    2    2
## [2,]    2    2
## 
## , , 3
## 
##      [,1] [,2]
## [1,]    3    3
## [2,]    3    3
```

你可以使用 `%*%` 运算符将 `矩阵` 对象相乘。如果你在做这个，那么转置运算符（即 `t()`）也很有用。你仍然可以使用逐元素（Hadamard）乘法。这使用更熟悉的乘法运算符 `*` 定义。

```py
# calculate a quadratic form y'Qy
y <-  matrix(c(1,2,3))
Q <-  diag(1, 3) # diag() gets and sets diagonal matrices
t(y) %*%  Q %*%  y
##      [,1]
## [1,]   14
```

有时你需要访问或修改 `矩阵` 对象的个别元素。你可以使用熟悉的 `[` 和 `[<-` 运算符来完成此操作。这里有一个设置示例。你不需要担心不同类型之间的强制转换。

```py
Qcopy <-  Q
Qcopy[1,1] <-  3
Qcopy[2,2] <-  4
Qcopy
##      [,1] [,2] [,3]
## [1,]    3    0    0
## [2,]    0    4    0
## [3,]    0    0    1
```

这里有一些提取示例。请注意，如果可能的话，`[` 将将 `矩阵` 强制转换为 `向量`。如果你希望避免这种情况，可以指定 `drop=FALSE`。

```py
Q
##      [,1] [,2] [,3]
## [1,]    1    0    0
## [2,]    0    1    0
## [3,]    0    0    1
Q[1,1]
## [1] 1
Q[2,]
## [1] 0 1 0
Q[2,,drop=FALSE]
##      [,1] [,2] [,3]
## [1,]    0    1    0
class(Q)
## [1] "matrix" "array"
class(Q[2,])
## [1] "numeric"
class(Q[2,,drop=FALSE]) 
## [1] "matrix" "array"
row(Q) >  1
##       [,1]  [,2]  [,3]
## [1,] FALSE FALSE FALSE
## [2,]  TRUE  TRUE  TRUE
## [3,]  TRUE  TRUE  TRUE
Q[row(Q) >  1] # column-wise ordering
## [1] 0 0 1 0 0 1
```

有其他函数以更有趣的方式操作一个或多个 `矩阵` 对象，但其中大部分将在未来的章节中介绍。例如，我们将在第十五部分中描述 `apply()` 如何与 `矩阵` 一起工作，我们将在第十二部分中讨论以不同方式组合 `矩阵` 对象。

## 4.3 练习

### 4.3.1 R 问题

考虑以下数据集。设 $N = 20$ 为行数。对于 $i=1,\ldots,N$，定义 $\mathbf{x}_i \in \mathbb{R}⁴$ 为第 $i$ 行的数据。

```py
d <-  matrix(c(
 -1.1585476,  0.06059602, -1.854421163,  1.62855626,
 0.5619835,  0.74857327, -0.830973409,  0.38432716,
 -1.6949202,  1.24726626,  0.068601035, -0.32505127,
 2.8260260, -0.68567999, -0.109012111, -0.59738648,
 -0.3128249, -0.21192009, -0.317923437, -1.60813901,
 0.3830597,  0.68000706,  0.787044622,  0.13872087,
 -0.2381630,  1.02531172, -0.606091651,  1.80442260,
 1.5429671, -0.05174198, -1.950780046, -0.87716787,
 -0.5927925, -0.40566883, -0.309193162,  1.25575250,
 -0.8970403, -0.10111751,  1.555160257, -0.54434356,
 2.4060504, -0.08199934, -0.472715155,  0.25254794,
 -1.0145770, -0.83132666, -0.009597552, -1.71378699,
 -0.3590219,  0.84127504,  0.062052945, -1.00587841,
 -0.1335952, -0.02769315, -0.102229046, -1.08526057,
 0.1641571, -0.08308289, -0.711009361,  0.06809487,
 2.2450975,  0.32619749,  1.280665384,  1.75090469,
 1.2147885,  0.10720830, -2.018215962,  0.34602861,
 0.7309219, -0.60083707, -1.007344145, -1.77345958,
 0.1791807, -0.49500051,  0.402840566,  0.60532646,
 1.0454594,  1.09878293,  2.784986486, -0.22579848), ncol = 4)
```

对于以下问题，请确保只使用转置函数 `t()`、矩阵乘法（即 `%*%`）和标量乘法/除法。你可以在交互模式下使用其他函数来检查你的工作，但请不要在提交中使用它们。

1.  计算样本均值 $\bar{\mathbf{x}} = \frac{1}{N} \sum_{i=1}^N \mathbf{x}_i$。用 `colMeans()` 检查你的工作，**但不要在提交的代码中使用该函数**。将其分配给变量 `xbar`。确保它是一个 $4 \times 1$ 的 `矩阵` 对象。

1.  计算以下数据的 $4 \times 4$ 样本协方差。将变量命名为 `mySampCov`，并确保它也是一个 `矩阵` 对象。**你可以用 `cov()` 检查你的工作，但不要在提交的代码中使用它**。样本协方差的公式为 $$\begin{equation} \frac{1}{N-1} \sum_{i=1}^N (\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^\intercal \end{equation}$$

创建一个名为 `P` 的 `矩阵`，它有一百行，一百列，所有元素非负，每个对角线元素为 $1/10$，所有行之和为 $1$。这个矩阵被称为**随机矩阵**，它描述了马尔可夫链如何随机随时间移动。

创建一个名为 `X` 的 `矩阵`，它有一千行，四列，每个元素都设置为 $0$ 或 $1$，其第一列设置为全 $1$，第二列在第二 $250$ 个元素中设置为 $1$，其他地方设置为 $0$，第三列在第三 $250$ 个位置中设置为 $1$，其他地方设置为 $0$，第四列在最后 $250$ 个位置中设置为 $1$，其他地方设置为 $0$。换句话说，它看起来像这样

$$\begin{equation} \begin{bmatrix} \mathbf{1}_{250} & \mathbf{0}_{250} & \mathbf{0}_{250} & \mathbf{0}_{250} \\ \mathbf{1}_{250} & \mathbf{1}_{250} & \mathbf{0}_{250} & \mathbf{0}_{250} \\ \mathbf{1}_{250} & \mathbf{0}_{250} & \mathbf{1}_{250} & \mathbf{0}_{250} \\ \mathbf{1}_{250} & \mathbf{0}_{250} & \mathbf{0}_{250} & \mathbf{1}_{250} \\ \end{bmatrix} \end{equation}$$ 其中 $\mathbf{1}_{250}$ 和 $\mathbf{0}_{250}$ 是长度为 $250$ 的列向量，它们的元素分别全部设置为 $1$ 或 $0$。

1.  计算投影（或帽）矩阵 $\mathbf{H} := \mathbf{X}\left(\mathbf{X}^\intercal \mathbf{X}\right)^{-1} \mathbf{X}^\intercal$。将其作为一个 `矩阵`，并命名为 `H`。

1.  对于一个随机向量，**可交换**的协方差矩阵是一个具有所有相同方差和所有相同协方的协方差矩阵。换句话说，它有两个独特的元素：对角线元素应该相同，非对角线元素也应该相同。在 R 中，生成十个 $100 \times 100$ 的**可交换**协方差矩阵，每个矩阵的方差为 $2$，可能的协方差取值在集合 $0,.01,.02, ..., .09$ 中。将这些十个协方差矩阵存储在一个三维数组中。第一个索引应该是每个矩阵的行索引，第二个索引应该是每个矩阵的列索引，第三个索引应该是“层”或“切片”，表示你拥有的 $10$ 个矩阵中的哪一个。将这个数组命名为 `myCovMats`

1.  在 R 中，生成一百个 $10 \times 10$ 的**可交换**协方差矩阵，每个矩阵的方差为 $2$，可能的协方差值在集合 $0,0.0009090909, ..., 0.0890909091, .09.$ 中。将这些 $100$ 个协方差矩阵存储在一个三维数组中。第一个索引应该是每个矩阵的行索引，第二个索引应该是每个矩阵的列索引，第三个索引应该是“层”或“切片”，表示你有哪些 $100$ 个矩阵。将这个数组命名为 `myCovMats2`

### 4.3.2 Python 问题

设 $\mathbf{X}$ 是一个 $n \times 1$ 的随机向量。如果它的概率密度函数可以写成

$$\begin{equation} f(\mathbf{x}; \mathbf{m}, \mathbf{C}) = (2\pi)^{-n/2}\text{det}\left( \mathbf{C} \right)^{-1/2}\exp\left[- \frac{1}{2} (\mathbf{x}- \mathbf{m})^\intercal \mathbf{C}^{-1} (\mathbf{x}- \mathbf{m}) \right] \end{equation}$$

评估这个密度需要小心。没有一种函数对所有情况都是最优的。这里有一些快速考虑的事项。

+   如果协方差矩阵是高维的，使用 `np.linalg.solve` 或 `np.linalg.inv` 来求逆非常大的矩阵会变得非常慢。如果你对协方差矩阵的结构有特殊的假设，请使用它！另外，了解当你尝试求不可逆矩阵的逆时会发生什么也是一个好主意。例如，你能依赖错误被抛出，还是它将返回一个错误的结果？

+   从上一个实验中回忆起来，对接近 $-\infty$ 的数字进行指数运算可能会出现数值下溢。最好是优先评估对数密度（以 $e$ 为底，自然对数）。还有[特殊函数可以评估对数行列式](https://numpy.org/doc/stable/reference/generated/numpy.linalg.slogdet.html)，这些函数不太可能下溢/溢出！

完成以下问题。**不要使用预定义的函数，如 `scipy.stats.norm` 和 `scipy.stats.multivariate_normal`，在你的提交中，但你可以使用它们来检查你的工作。只使用“标准”函数和 Numpy n 维数组。**使用以下定义的 $\mathbf{x}$ 和 $\mathbf{m}$：

```py
import numpy as np
x = np.array([1.1, .9, 1.0]).reshape((3,1))
m = np.ones(3).reshape((3,1))
```

1.  设 $\mathbf{C} = \begin{bmatrix} 10 & 0 & 0 \\ 0 & 10 & 0 \\ 0 & 0 & 10 \end{bmatrix}$。评估并分配对数密度到一个名为 `log_dens1` 的 `float` 类型的变量。你能在不定义一个用于 $\mathbf{C}$ 的 numpy 数组的情况下完成这个任务吗？

1.  设 $\mathbf{C} = \begin{bmatrix} 10 & 0 & 0 \\ 0 & 11 & 0 \\ 0 & 0 & 12 \end{bmatrix}$。评估并将对数密度分配给一个类似 `float` 的变量 `log_dens2`。你能在不定义 $\mathbf{C}$ 的 numpy 数组的情况下完成这个任务吗？

1.  设 $\mathbf{C} = \begin{bmatrix} 10 & -.9 & -.9 \\ -.9 & 11 & -.9 \\ -.9 & -.9 & 12 \end{bmatrix}$. 评估并将对数密度分配给一个类似 `float` 的变量 `log_dens3`。你能在不定义 $\mathbf{C}$ 的 numpy 数组的情况下完成这个任务吗？

考虑这个来自 (Cortez 等人 2009)，由 (Dua 和 Graff 2017) 提供的 [葡萄酒数据集](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)。使用以下代码读取它。注意，你可能需要先使用 `os.chdir()`。

```py
import pandas as pd
d = pd.read_csv("winequality-red.csv", sep = ";")
d.head()
```

1.  通过删除 `"quality"` 列，并从每个元素中减去列均值来创建 **设计矩阵**（在数学上表示为 $\mathbf{X}$）。将变量命名为 `X`，并确保它是一个 Numpy `ndarray`，而不是 Pandas `DataFrame`。

1.  计算 $\mathbf{X}^\intercal \mathbf{X}$ 的 **谱分解**。换句话说，找到“特殊”矩阵⁸ $\mathbf{V}$ 和 $\boldsymbol{\Lambda}$，使得 $\mathbf{X}^\intercal \mathbf{X} = \mathbf{V} \boldsymbol{\Lambda} \mathbf{V}^\intercal$。注意，*特征向量*存储在矩阵 $\mathbf{V} := \begin{bmatrix} \mathbf{V}_1 & \cdots & \mathbf{V}_{11}\end{bmatrix}$ 的列中，而标量 *特征值*存储为对角元素 $\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \ldots, \lambda_{11})$。将特征向量存储在一个名为 `eig_vecs` 的 `ndarray` 中，并将特征值存储在一个名为 `eig_vals` 的 Numpy `array` 中。提示：使用 `np.linalg.eig()`。如果你对线性代数不太熟悉，不必过于担心刷新你的记忆关于特征向量和特征值是什么。

1.  计算 $\mathbf{X}$ 的 **奇异值分解**。换句话说，找到“特殊”⁹ 矩阵 $\mathbf{U}$，$\mathbf{\Sigma}$，和 $\mathbf{V}$，使得 $\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\intercal$。使用 `np.linalg.svd`，并且不必过于担心数学细节。这两个分解是相关的。如果你做对了，两个 $\mathbf{V}$ 矩阵应该相同，而 $\boldsymbol{\Sigma}$ 的元素应该是 $\boldsymbol{\Lambda}$ 元素的平方根。将特征向量作为名为 `eig_vecs_v2` 的 `ndarray` 的列存储，并将奇异值（$\boldsymbol{\Sigma}$ 的对角元素）存储在一个名为 `sing_vals` 的 Numpy `array` 中。

1.  计算第一个主成分向量，并将其命名为 `first_pc_v1`。其数学公式是 $\mathbf{X} \mathbf{U}_1$，其中 $\mathbf{U}_1$ 是与最大特征值 $\lambda_1$ 相关的特征向量。这可以理解为，在某种意义上，通过平均所有其他预测因子所能创建的最具信息量的预测因子。

### 参考文献

Albon, Chris. 2018. *Machine Learning with Python Cookbook: Practical Solutions from Preprocessing to Deep Learning*. 1st ed. O’Reilly Media, Inc.

Cortez, Paulo, António Cerdeira, Fernando Almeida, Telmo Matos, and José Reis. 2009. “通过数据挖掘物理化学性质建模葡萄酒偏好。” *Decis. Support Syst.* 47 (4): 547–53. [`dblp.uni-trier.de/db/journals/dss/dss47.html#CortezCAMR09`](http://dblp.uni-trier.de/db/journals/dss/dss47.html#CortezCAMR09).

Dua, Dheeru, and Casey Graff. 2017. “UCI Machine Learning Repository.” University of California, Irvine, School of Information; Computer Sciences. [`archive.ics.uci.edu/ml`](http://archive.ics.uci.edu/ml).

Matloff, Norman. 2011. *The Art of R Programming: A Tour of Statistical Software Design*. 1st ed. USA: No Starch Press.

* * *

1.  从技术角度来看，这些容器之间的区别更为微妙。在 R 中，一个 `array` 可以有一维、二维或更多维度，它只是一个带有额外维度属性的向量。此外，二维数组与 `matrix` 是相同的。↩

1.  对于这个问题，你不必过于担心这些矩阵的性质。↩

1.  再次强调，对于这个问题，你不必过于担心这些矩阵的性质。↩
