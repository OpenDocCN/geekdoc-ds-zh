# 7.1. 激励示例：发现数学主题#

> 原文：[`mmids-textbook.github.io/chap07_rwmc/01_motiv/roch-mmids-rwmc-motiv.html`](https://mmids-textbook.github.io/chap07_rwmc/01_motiv/roch-mmids-rwmc-motiv.html)

网络分析中的一项常见任务是识别图中的“中心”顶点。中心性是一个模糊的概念。它可以根据上下文和网络类型以许多不同的方式定义。引用自[Wikipedia](https://en.wikipedia.org/wiki/Centrality)：

> 在图论和网络分析中，中心性指标为图中的节点分配数字或排名，对应于它们的网络位置。应用包括识别社交网络中最有影响力的人（们）、互联网或城市网络中的关键基础设施节点、疾病的超级传播者以及大脑网络。 [...] 中心性指标是对“什么特征定义了一个重要的顶点？”这一问题的回答。答案是以图顶点上的实值函数的形式给出的，其中产生的值预期将提供排名，以识别最重要的节点。“重要性”一词有广泛的含义，导致中心性的许多不同定义。

在无向图中，一个自然的方法是查看顶点的度作为其重要性的度量（也称为度中心性）。但这并不是唯一的方法。例如，可以查看到所有其他节点的平均距离（其倒数是[接近中心性](https://en.wikipedia.org/wiki/Closeness_centrality)）或通过顶点之间的最短路径对的数量（称为[中介中心性](https://en.wikipedia.org/wiki/Betweenness_centrality)）。

如果图是定向的，事情就会变得稍微复杂一些。例如，现在不仅有入度，还有出度。

让我们看看一个具有实际重要性的特定例子，即万维网（从现在起，简称 Web）。在这种情况下，顶点是网页，从 \(u\) 到 \(v\) 的有向边表示从页面 \(u\) 到页面 \(v\) 的超链接。Web 太大，无法在此分析。相反，我们将考虑它的一个微小（但仍然有趣！）的子集，即[Wolfram 的 MathWorld](https://mathworld.wolfram.com)的页面，这是一个出色的数学资源。

MathWorld 的每一页都关注一个特定的数学概念，例如[无标度网络](https://mathworld.wolfram.com/Scale-FreeNetwork.html)。描述了定义和显著性质。对我们来说重要的是，在“参见”部分列出了其他相关的数学概念，并附有它们在 MathWorld 页面的链接。在无标度网络的情况下，[小世界网络](https://mathworld.wolfram.com/SmallWorldNetwork.html)主题被引用，以及其他主题。

生成的有向图可通过[NetSet](https://netset.telecom-paris.fr/index.html)数据集获取，并可在此[下载](https://netset.telecom-paris.fr/pages/mathworld.html)。我们现在加载它。为了方便，我们将其重新格式化为`mathworld-adjacency.csv`和`mathworld-titles.csv`文件，这些文件可在本书的[GitHub](https://github.com/MMiDS-textbook/MMiDS-textbook.github.io/tree/main/utils/datasets)上找到。

```py
data_edges = pd.read_csv('mathworld-adjacency.csv')
data_edges.head() 
```

|  | 来源 | 目标 |
| --- | --- | --- |
| 0 | 0 | 2 |
| 1 | 1 | 47 |
| 2 | 1 | 404 |
| 3 | 1 | 2721 |
| 4 | 2 | 0 |

它由一系列有向边组成。例如，第一个是从顶点`0`到顶点`2`的边。第二个是从`1`到`47`等等。

总共有 \(49069\) 条边。

```py
data_edges.shape[0] 
```

```py
49069 
```

第二个文件包含页面的标题。

```py
data_titles = pd.read_csv('mathworld-titles.csv')
data_titles.head() 
```

|  | 标题 |
| --- | --- |
| 0 | 亚历山大角形球体 |
| 1 | 异域球体 |
| 2 | 安托万角形球体 |
| 3 | 平面 |
| 4 | 普安卡曲面 |

因此，上面的第一条边是从`亚历山大角形球体`到`安托万角形球体`。也就是说，[后者](https://mathworld.wolfram.com/AntoinesHornedSphere.html)被列在[前者](https://mathworld.wolfram.com/AlexandersHornedSphere.html)的“参见”部分。

有 \(12362\) 个主题。

```py
n = data_titles.shape[0]
print(n) 
```

```py
12362 
```

我们通过逐个添加边来构建图。我们首先将`df_edges`转换为 NumPy 数组。

```py
edgelist = data_edges[['from','to']].to_numpy()
print(edgelist) 
```

```py
[[    0     2]
 [    1    47]
 [    1   404]
 ...
 [12361 12306]
 [12361 12310]
 [12361 12360]] 
```

```py
G = nx.empty_graph(n, create_using=nx.DiGraph)
for i in range(edgelist.shape[0]):
    G.add_edge(edgelist[i,0], edgelist[i,1]) 
```

回到中心性问题，我们现在可以尝试测量不同节点的的重要性。例如，`亚历山大角形球体`的入度为：

```py
G.in_degree(0) 
```

```py
5 
```

而`安托万角形球体`的则是：

```py
G.in_degree(2) 
```

```py
1 
```

这表明前者比后者更中心，至少在它被引用得更频繁这个意义上。

但这是否是正确的度量？考虑以下情况：`安托万角形球体`只收到一个引用，但它来自一个看似相对重要的顶点，`亚历山大角形球体`。如何在量化其在网络中的重要性时考虑这一点？

我们将在本章后面回到这个问题。为了暗示即将发生的事情，我们将发现“随机探索图”提供了对中心性的强大视角。
