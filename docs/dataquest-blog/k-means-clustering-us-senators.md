# 教程:K-Means 聚类美国参议员

> 原文：<https://www.dataquest.io/blog/k-means-clustering-us-senators/>

February 15, 2015![us-senators-python-data-tutorial](img/d271510463de468301609eb398079bd9.png)Clustering is a powerful way to split up datasets into groups based on similarity. A very popular clustering algorithm is [K-means clustering](https://en.wikipedia.org/wiki/K-means_clustering). In K-means clustering, we divide data up into a fixed number of clusters while trying to ensure that the items in each cluster are as similar as possible. In this post, we’ll explore cluster US Senators using an interactive Python environment. We’ll use the voting history from the [114th Congress](https://en.wikipedia.org/wiki/114th_United_States_Congress) to split Senators into clusters.

## 载入数据

我们有一个包含第 114 届参议院所有投票的 csv 文件。你可以在这里下载文件。每行包含一位参议员的投票。投票以`0`表示“反对”，`1`表示“赞成”，`0.5`表示“弃权”。以下是前三行数据:

```py
name,party,state,00001,00004,00005,00006,00007,00008,00009,00010,00020,00026,00032,00038,00039,00044,00047
Alexander,R,TN,0,1,1,1,1,0,0,1,1,1,0,0,0,0,0
Ayotte,R,NH,0,1,1,1,1,0,0,1,0,1,0,1,0,1,0
```

我们可以使用`pandas`将 csv 文件读入 Python。

```py
 import pandas as pd
# Read in the csv file
votes = pd.read_csv("114_congress.csv")

# As you can see, there are 100 senators, and they voted on 15 bills (we subtract 3 because the first 3 columns aren't bills).
print(votes.shape)

# We have more "Yes" votes than "No" votes overall
print(pd.value_counts(votes.iloc[:,3:].values.ravel())) 
```

```py
 (100, 18)
1.0    803
0.0    669
0.5    28
dtype: int64 
```

## 初始 k 均值聚类

*K-means* 聚类将尝试从参议员中进行聚类。每个集群将包含投票尽可能相似的参议员。我们需要预先指定我们想要的集群数量。让我们试试`2`看看看起来怎么样。

```py
 import pandas as pd

# The kmeans algorithm is implemented in the scikits-learn library
from sklearn.cluster import KMeans

# Create a kmeans model on our data, using 2 clusters. random_state helps ensure that the algorithm returns the same results each time.
kmeans_model = KMeans(n_clusters=2, random_state=1).fit(votes.iloc[:, 3:])

# These are our fitted labels for clusters -- the first cluster has label 0, and the second has label 1.
labels = kmeans_model.labels_

# The clustering looks pretty good!
# It's separated everyone into parties just based on voting history
print(pd.crosstab(labels, votes["party"]))
```

```py
 party   D  I   R
row_0           
0      41  2   0
1       3  0  54 
```

## 探索错误群组中的人

我们现在可以找出哪些参议员属于“错误的”群体。这些参议员属于与对方有关联的群体。

```py
 # Let's call these types of voters "oddballs" (why not?)
# There aren't any republican oddballs
democratic_oddballs = votes[(labels == 1) & (votes["party"] == "D")]

# It looks like Reid has abstained a lot, which changed his cluster.
# Manchin seems like a genuine oddball voter.
print(democratic_oddballs["name"])
```

```py
 42    Heitkamp
56     Manchin
74        Reid
Name: name, dtype: object
```

## 绘制群集图

让我们通过绘制它们来更深入地探索我们的集群。每一列数据都是图上的一个维度，我们无法可视化 15 个维度。我们将使用*主成分分析*将投票列压缩成两列。然后，我们可以根据他们的投票划分出我们所有的参议员，并根据他们的 K 均值聚类对他们进行着色。

```py
 import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
pca_2 = PCA(2)

# Turn the vote data into two columns with PCA
plot_columns = pca_2.fit_transform(votes.iloc[:,3:18])

# Plot senators based on the two dimensions, and shade by cluster label
# You can see the plot by clicking "plots" to the bottom right
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=votes["label"])
plt.show() 
```

## 尝试更多群集

虽然两个集群很有趣，但它没有告诉我们任何我们不知道的事情。更多的集群可以显示每个政党的翅膀，或跨党派团体。让我们尝试使用 5 个集群来看看会发生什么。

```py
 import pandas as pdfrom sklearn.cluster 
import KMeanskmeans_model = KMeans(n_clusters=5, random_state=1).fit(votes.iloc[:, 3:])
labels = kmeans_model.labels_

# The republicans are still pretty solid, but it looks like there are two democratic "factions"
print(pd.crosstab(labels, votes["party"])) 
```

```py
 party   D  I   R
row_0   6  0   0
1       0  0  52
2      31  1   0
3       0  0   2
4       7  1   0 
```

## 关于 k 均值聚类的更多信息

关于 K-means 聚类的更多信息，你可以查看我们关于 K-means 聚类的 Dataquest 课程。