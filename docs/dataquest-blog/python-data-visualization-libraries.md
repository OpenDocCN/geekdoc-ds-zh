# 教程:比较 Python 中的 7 种数据可视化工具

> 原文：<https://www.dataquest.io/blog/python-data-visualization-libraries/>

November 12, 2015The Python [scientific stack](https://www.scipy.org/about.html) is fairly mature, and there are libraries for a variety of use cases, including [machine learning](https://scikit-learn.org/), and [data analysis](https://pandas.pydata.org/). Data visualization is an important part of being able to explore data and communicate results, but has lagged a bit behind other tools such as R in the past. Luckily, many new Python data visualization libraries have been created in the past few years to close the gap. [matplotlib](https://matplotlib.org/index.html) has emerged as the main data visualization library, but there are also libraries such as [vispy](https://vispy.org/), [bokeh](https://bokeh.pydata.org/en/latest/), [seaborn](https://seaborn.pydata.org), [pygal](https://github.com/Kozea/pygal), [folium](https://github.com/python-visualization/folium), and [networkx](https://github.com/networkx/networkx) that either build on matplotlib or have functionality that it doesn’t support. In this post, we’ll use a real-world dataset, and use each of these libraries to make visualizations. As we do that, we’ll discover what areas each library is best in, and how to leverage the Python data visualization ecosystem most effectively. At [Dataquest](https://www.dataquest.io), we’ve built interactive courses that teaches you about Python data visualization tools. If you want to learn in more depth, check out our [data visualization courses](https://www.dataquest.io/course/python-for-data-science-fundamentals).

## 探索数据集

在我们开始可视化数据之前，让我们快速浏览一下将要使用的数据集。我们将使用来自

[开放航班](https://openflights.org/data.html)。我们将使用[航线](https://openflights.org/data.html#route)、[机场](https://openflights.org/data.html#airport)和[航空公司](https://openflights.org/data.html#airline)的数据。路线数据中的每一行对应于两个机场之间的航线。机场数据中的每一行都对应于世界上的一个机场，并且有关于它的信息。航空公司数据中的每一行代表一家航空公司。我们首先读入数据:

```py
 # Import the pandas library.
import pandas
# Read in the airports data.
airports = pandas.read_csv("airports.csv", header=None, dtype=str)
airports.columns = ["id", "name", "city", "country", "code", "icao", "latitude", "longitude", "altitude", "offset", "dst", "timezone"]
# Read in the airlines data.airlines = pandas.read_csv("airlines.csv", header=None, dtype=str)
airlines.columns = ["id", "name", "alias", "iata", "icao", "callsign", "country", "active"]
# Read in the routes data.routes = pandas.read_csv("routes.csv", header=None, dtype=str)
routes.columns = ["airline", "airline_id", "source", "source_id", "dest", "dest_id", "codeshare", "stops", "equipment"]
```

数据没有列标题，所以我们通过将

`columns`属性。我们希望将每一列都作为一个字符串读入——这将使以后基于 id 匹配行时跨数据帧的比较更容易。我们通过在读入数据时设置`dtype`参数来实现这一点。我们可以快速浏览一下每个数据帧:

```py
airports.head()
```

|  | 身份证明（identification） | 名字 | 城市 | 国家 | 密码 | 国际民航组织 | 纬度 | 经度 | 海拔 | 抵消 | 夏令时 | 时区 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Zero | one | Goroka | Goroka | 巴布亚新几内亚 | GKA | AYGA | -6.081689 | 145.391881 | Five thousand two hundred and eighty-two | Ten | U | 太平洋/莫尔兹比港 |
| one | Two | 马当 | 马当 | 巴布亚新几内亚 | 杂志 | 月，月 | -5.207083 | 145.788700 | Twenty | Ten | U | 太平洋/莫尔兹比港 |
| Two | three | 芒特哈根 | 芒特哈根 | 巴布亚新几内亚 | HGU | AYMH | -5.826789 | 144.295861 | Five thousand three hundred and eighty-eight | Ten | U | 太平洋/莫尔兹比港 |
| three | four | Nadzab | Nadzab | 巴布亚新几内亚 | 莱城 | AYNZ | -6.569828 | 146.726242 | Two hundred and thirty-nine | Ten | U | 太平洋/莫尔兹比港 |
| four | five | 莫尔斯比港杰克逊国际机场 | 莫尔兹比港 | 巴布亚新几内亚 | 砰的一声 | AYPY | -9.443383 | 147.220050 | One hundred and forty-six | Ten | U | 太平洋/莫尔兹比港 |

```py
airlines.head()
```

|  | 身份证明（identification） | 名字 | 别名 | international air transport association 国际航空运输协会 | 国际民航组织 | callsign | 国家 | 活跃的 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Zero | one | 私人航班 | \N | – | 圆盘烤饼 | 圆盘烤饼 | 圆盘烤饼 | Y |
| one | Two | 135 航空公司 | \N | 圆盘烤饼 | GNL | 一般 | 美国 | 普通 |
| Two | three | 1 时代航空公司 | \N | 1T | RNX | 下次 | 南非 | Y |
| three | four | 2 Sqn 第一初级飞行训练学校 | \N | 圆盘烤饼 | WYT | 圆盘烤饼 | 联合王国 | 普通 |
| four | five | 213 飞行单位 | \N | 圆盘烤饼 | TFU | 圆盘烤饼 | 俄罗斯 | 普通 |

```py
routes.head()
```

|  | 航空公司 | 航空公司 id | 来源 | 来源标识 | 建筑环境及 HVAC 系统模拟的软件平台 | 目的地标识 | 代码共享 | 停止 | 装备 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Zero | 2B | Four hundred and ten | 年度等价利率 | Two thousand nine hundred and sixty-five | KZN | Two thousand nine hundred and ninety | 圆盘烤饼 | Zero | CR2 |
| one | 2B | Four hundred and ten | ASF | Two thousand nine hundred and sixty-six | KZN | Two thousand nine hundred and ninety | 圆盘烤饼 | Zero | CR2 |
| Two | 2B | Four hundred and ten | ASF | Two thousand nine hundred and sixty-six | MRV | Two thousand nine hundred and sixty-two | 圆盘烤饼 | Zero | CR2 |
| three | 2B | Four hundred and ten | CEK | Two thousand nine hundred and sixty-eight | KZN | Two thousand nine hundred and ninety | 圆盘烤饼 | Zero | CR2 |
| four | 2B | Four hundred and ten | CEK | Two thousand nine hundred and sixty-eight | OVB | Four thousand and seventy-eight | 圆盘烤饼 | Zero | CR2 |

我们可以单独对每个数据集进行各种有趣的探索，但只有将它们结合起来，我们才能看到最大的收获。Pandas 将在我们进行分析时为我们提供帮助，因为它可以轻松地过滤矩阵或对矩阵应用函数。我们将深入研究一些有趣的指标，比如分析航空公司和航线。在这样做之前，我们需要做一些数据清理工作:

```py
routes = routes[routes["airline_id"] != "\\N"]
```

这一行确保我们在

`airline_id`列。

## 制作直方图

既然我们理解了数据的结构，我们就可以继续前进，开始绘制图表来探索它。对于我们的第一个图，我们将使用 matplotlib。matplotlib 是 Python 堆栈中一个相对低级的绘图库，因此与其他库相比，它通常需要更多的命令来制作好看的绘图。另一方面，你可以用 matplotlib 制作几乎任何类型的情节。它非常灵活，但是这种灵活性是以冗长为代价的。我们将首先制作一个直方图，按航空公司显示航线长度的分布。A

[直方图](https://en.wikipedia.org/wiki/Histogram)将所有路线长度划分为范围(或“箱”)，并计算每个范围内有多少条路线。这能告诉我们航空公司是飞更短的航线，还是更长的航线。为了做到这一点，我们需要首先计算路线长度。第一步是距离公式。我们将使用哈弗线距离，它计算纬度，经度对之间的距离。

```py
 import math
def haversine(lon1, lat1, lon2, lat2):
    # Convert coordinates to floats.
    lon1, lat1, lon2, lat2 = [float(lon1), float(lat1), float(lon2), float(lat2)]
    # Convert to radians from degrees.
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    # Compute distance.
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    km = 6367 * c
    return km
```

然后我们可以做一个函数来计算

`source`和`dest`机场为单条航线。为此，我们需要从 routes 数据帧中获取`source_id`和`dest_id`机场，然后将它们与`airports`数据帧中的`id`列进行匹配，以获取这些机场的纬度和经度。然后，就是做计算的问题了。函数如下:

```py
 def calc_dist(row):
    dist = 0 
    try:
        # Match source and destination to get coordinates.
        source = airports[airports["id"] == row["source_id"]].iloc[0]
        dest = airports[airports["id"] == row["dest_id"]].iloc[0]
        # Use coordinates to compute distance.
        dist = haversine(dest["longitude"], dest["latitude"], source["longitude"], source["latitude"])
    except (ValueError, IndexError):
        pass
    return dist
```

如果中有无效的值，该函数可能会失败

`source_id`或`dest_id`列，所以我们将添加一个`try/except`块来捕捉这些。最后，我们将使用 pandas 在`routes`数据帧中应用距离计算功能。这将给我们一个包含所有路线长度的熊猫系列。路线长度均以千米为单位。

```py
route_lengths = routes.apply(calc_dist, axis=1)
```

现在我们已经有了一系列的路径长度，我们可以创建一个直方图，它会将值绑定到范围内，并计算每个范围内有多少条路径:

```py
 import matplotlib.pyplot as plt

plt.hist(route_lengths, bins=20)
```

![mplhist2](img/9fbcba1496d58ee1624e8910b3071e06.png)我们用`import matplotlib.pyplot as plt`导入 matplotlib 绘图函数。然后我们用`%matplotlib inline`设置 matplotlib 来显示 ipython 笔记本中的图形。最后，我们可以用`plt.hist(route_lengths, bins=20)`做一个直方图。正如我们所看到的，航空公司飞行的短途航线比长途航线多。

## 使用 Seaborn

我们可以用 Python 的高级绘图库 seaborn 进行类似的绘图。Seaborn 建立在 matplotlib 的基础上，简化了通常与统计工作有关的某些类型的绘图。我们可以使用

`distplot`绘制一个直方图的函数，其顶部有一个核密度估计值。核密度估计是一条曲线——本质上是直方图的平滑版本，更容易看到其中的模式。

```py
 import seaborn
seaborn.distplot(route_lengths, bins=20)
```

如你所见，seaborn 也有比 matplotlib 更好的默认风格。Seaborn 没有自己版本的所有 matplotlib 图，但这是一种很好的方式，可以快速获得比默认 matplotlib 图更深入的好看的图。如果你需要更深入，做更多的统计工作，它也是一个很好的库。

## 条形图

直方图很棒，但也许我们想看看航空公司的平均航线长度。我们可以用柱状图来代替——每个航空公司都有一个单独的柱状图，告诉我们每个航空公司的平均长度。这将让我们看到哪些运营商是地区性的，哪些是国际性的。我们可以使用 python 数据分析库 pandas 来计算每家航空公司的平均航线长度。

```py
 import numpy
# Put relevant columns into a dataframe.
route_length_df = pandas.DataFrame({"length": route_lengths, "id": routes["airline_id"]})
# Compute the mean route length per airline.
airline_route_lengths = route_length_df.groupby("id").aggregate(numpy.mean)
# Sort by length so we can make a better chart.
airline_route_lengths = airline_route_lengths.sort("length", ascending=False) 
```

我们首先用航线长度和航线 id 制作一个新的数据帧。我们分开了

`route_length_df`根据`airline_id`分组，基本上每个航空公司制作一个数据帧。然后，我们使用熊猫`aggregate`函数获取每个航空公司数据帧中`length`列的平均值，并将每个结果重新组合成一个新的数据帧。然后，我们对数据帧进行排序，使航线最多的航空公司排在最前面。然后我们可以用 matplotlib 来绘制:

```py
plt.bar(range(airline_route_lengths.shape[0]), airline_route_lengths["length"])
```

![](img/cd446d28214aa26841954714543d619d.png)matplotlib`plt.bar`方法根据每家航空公司飞行的平均航线长度(`airline_route_lengths["length"]`)来绘制每家航空公司。上面这个图的问题是，我们不能轻易看出哪个航空公司有什么航线长度。为了做到这一点，我们需要能够看到轴标签。这有点难，因为有这么多的航空公司。一种更简单的方法是让图具有交互性，这将允许我们放大和缩小来查看标签。我们可以使用散景库来实现这一点——它使得制作交互式、可缩放的绘图变得简单。要使用散景，我们首先需要预处理我们的数据:

```py
 def lookup_name(row):
    try:
        # Match the row id to the id in the airlines dataframe so we can get the name.
        name = airlines["name"][airlines["id"] == row["id"]].iloc[0]
    except (ValueError, IndexError):
        name = ""
    return name
# Add the index (the airline ids) as a column.
airline_route_lengths["id"] = airline_route_lengths.index.copy()
# Find all the airline names.
airline_route_lengths["name"] = airline_route_lengths.apply(lookup_name, axis=1)
# Remove duplicate values in the index.
airline_route_lengths.index = range(airline_route_lengths.shape[0]) 
```

上面的代码将获取

`airline_route_lengths`，并在`name`栏中添加，该栏包含各航空公司的名称。我们还添加了`id`列，这样我们就可以进行查找(apply 函数没有传入索引)。最后，我们重置索引列，使其包含所有唯一的值。如果没有这个，散景就不能正常工作。现在，我们可以进入图表部分了:

```py
 import numpy as np
from bokeh.io import output_notebook
from bokeh.charts import Bar, showoutput_notebook()
p = Bar(airline_route_lengths, 'name', values='length', title="Average airline route lengths")
show(p) 
```

我们打电话

`output_notebook`设置散景以显示 ipython 笔记本中的图形。然后，我们使用我们的数据框架和某些列绘制一个柱状图。最后，`show`函数显示剧情。在您的笔记本中生成的图不是图像，它实际上是一个 javascript 小部件。因此，我们在下面显示了一个截图，而不是实际的图表。![bokehbar2](img/fbd4f22dd4a398cebd7cce38ce6e7f34.png)有了这个图，我们可以放大看哪个航空公司飞的航线最长。上面的图像使标签看起来很紧凑，但是放大后更容易看到。

## 水平条形图

Pygal 是一个 python 数据分析库，可以快速制作有吸引力的图表。我们可以用它来按长度划分路线。我们首先将路线分为短、中和长路线，并计算每条路线在我们的

`route_lengths`。

```py
 long_routes = len([k for k in route_lengths if k > 10000]) / len(route_lengths)
medium_routes = len([k for k in route_lengths if k < 10000 and k > 2000]) / len(route_lengths)
short_routes = len([k for k in route_lengths if k < 2000]) / len(route_lengths) 
```

然后，我们可以在 pygal 水平条形图中将每个点绘制成一个条形:

```py
 import pygal
from IPython.display import SVG
chart = pygal.HorizontalBar()
chart.title = 'Long, medium, and short routes'
chart.add('Long', long_routes * 100)
chart.add('Medium', medium_routes * 100)
chart.add('Short', short_routes * 100)
chart.render_to_file('/blog/conteimg/routes.svg')
SVG(filename='/blog/conteimg/routes.svg') 
```

![routes](img/f5679e4d15e500560dfe1f074c4a7a13.png)在上面，我们首先创建一个空图表。然后，我们添加元素，包括标题和栏。每个条形被传递一个百分比值(在`100`中),显示该类型的路线有多常见。最后，我们将图表呈现到一个文件中，并使用 IPython 的 SVG 显示功能来加载和显示该文件。这个图看起来比默认的 matplotlib 图表要好一些，但是我们确实需要编写更多的代码来创建它。Pygal 可能适用于小型演示质量的图形。

## 散点图

散点图使我们能够比较多列数据。我们可以做一个简单的散点图来比较航空公司 id 号和航空公司名称的长度:

```py
name_lengths = airlines["name"].apply(lambda x: len(str(x)))
plt.scatter(airlines["id"].astype(int), name_lengths)
```

![mplscatter](img/ea9ce24cc8aa46d0ccf84f0f0bb2ad2d.png)首先，我们使用熊猫`apply`方法计算每个名字的长度。这将找到每个航空公司名称的字符长度。然后，我们使用 matplotlib 绘制一个散点图，比较航空公司 id 和名称长度。当我们绘图时，我们将`airlines`的`id`列转换为整数类型。如果我们不这样做，这个图就不会工作，因为它需要 x 轴上的数值。我们可以看到相当多的长名字出现在早期的 id 中。这可能意味着成立较早的航空公司往往名字更长。我们可以用 seaborn 来验证这种预感。Seaborn 有一个散点图的扩展版本，一个联合图，显示了两个变量的相关程度，以及每个变量的单独分布。

```py
 data = pandas.DataFrame({"lengths": name_lengths, "ids": airlines["id"].astype(int)})
seaborn.jointplot(x="ids", y="lengths", data=data) 
```

上图显示这两个变量之间没有任何真正的相关性——r 的平方值很低。

## 静态地图

我们的数据天生非常适合地图绘制——我们有机场的纬度和经度对，也有出发地和目的地机场的纬度和经度对。我们能做的第一张地图是显示全世界所有机场的地图。我们可以用

matplotlib 的[底图](https://matplotlib.org/basemap/)扩展。这使得绘制世界地图和添加点，是非常可定制的。

```py
 # Import the basemap package
from mpl_toolkits.basemap import Basemap
# Create a map on which to draw.  We're using a mercator projection, and showing the whole world.
m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')
# Draw coastlines, and the edges of the map.
m.drawcoastlines()
m.drawmapboundary()
# Convert latitude and longitude to x and y coordinatesx, y = m(list(airports["longitude"].astype(float)), list(airports["latitude"].astype(float)))
# Use matplotlib to draw the points onto the map.
m.scatter(x,y,1,marker='o',color='red')
# Show the plot.
plt.show()
```

在上面的代码中，我们首先用一个

[墨卡托投影](https://en.wikipedia.org/wiki/Mercator_projection)。墨卡托投影是一种将整个世界投影到二维表面上的方法。然后，我们用红点在地图上画出机场。![mplmap](img/aa9138730bfeb21d256312c376155b23.png)上面地图的问题是很难看到每个机场的位置——它们只是在机场密度高的区域合并成一个红色的斑点。就像 bokeh 一样，有一个交互式地图库，我们可以使用它来放大地图，并帮助我们找到各个机场。

```py
 import folium
# Get a basic world map.
airports_map = folium.Map(location=[30, 0], zoom_start=2)
# Draw markers on the map.
for name, row in airports.iterrows():
    # For some reason, this one airport causes issues with the map.
    if row["name"] != "South Pole Station":
        airports_map.circle_marker(location=`, row["longitude"]], popup=row["name"])
# Create and show the map.airports_map.create_map('airports.html')
airports_map`

![foliummap](img/5ab4ef075cd0b1530e78258b06f1d180.png)leave 使用 fleet . js 制作全交互地图。您可以点击每个机场，在弹出窗口中查看其名称。上面是截图，但是实际的图印象深刻得多。follow 还允许你广泛地修改选项，以制作更好的标记，或者在地图上添加更多的东西。

## 画大圆

在地图上看到所有的航线会很酷。幸运的是，我们可以使用底图来做到这一点。我们会抽签

[大圈](https://en.wikipedia.org/wiki/Great_circle)连接出发地和目的地机场。每个圆圈将显示一架飞机的航线。不幸的是，路线太多了，把它们都显示出来会很混乱。相反，我们将展示第一条`3000`路线。

```
 # Make a base map with a mercator projection. 
# Draw the coastlines.
m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')
m.drawcoastlines()
# Iterate through the first 3000 rows.
for name, row in routes[:3000].iterrows():
    try:
        # Get the source and dest airports.
        source = airports[airports["id"] == row["source_id"]].iloc[0]
        dest = airports[airports["id"] == row["dest_id"]].iloc[0]
        # Don't draw overly long routes.
        if abs(float(source["longitude"]) - float(dest["longitude"])) < 90:
            # Draw a great circle between source and dest airports.
            m.drawgreatcircle(float(source["longitude"]), float(source["latitude"]), float(dest["longitude"]), float(dest["latitude"]),linewidth=1,color='b')
    except (ValueError, IndexError):
        pass
    # Show the map.
plt.show()
```py

上面的代码将绘制一张地图，然后在上面绘制路线。我们添加了一些过滤器，以防止过长的路线遮蔽其他路线。

## 绘制网络图

我们要做的最后一项探索是绘制机场网络图。每个机场将是网络中的一个节点，如果机场之间有路线，我们将在节点之间画边。如果有多条路线，我们将增加边权重，以显示机场之间的联系更加紧密。我们将使用 networkx 库来完成这项工作。首先，我们需要计算机场之间的边权重。

```
 # Initialize the weights dictionary.
weights = {}
# Keep track of keys that have been added once -- we only want edges with a weight of more than 1 to keep our network size manageable.added_keys = []
# Iterate through each route.
for name, row in routes.iterrows():
    # Extract the source and dest airport ids.
    source = row["source_id"]
    dest = row["dest_id"]
        # Create a key for the weights dictionary.
    # This corresponds to one edge, and has the start and end of the route.
    key = "{0}_{1}".format(source, dest)
    # If the key is already in weights, increment the weight.
    if key in weights:
        weights[key] += 1
    # If the key is in added keys, initialize the key in the weights dictionary, with a weight of 2.
    elif key in added_keys:
        weights[key] = 2
    # If the key isn't in added_keys yet, append it.
    # This ensures that we aren't adding edges with a weight of 1.
    else:
        added_keys.append(key)
```py

一旦上述代码运行完毕，权重字典将包含两个机场之间权重大于 2 的每条边。因此，任何由两条或多条航线连接的机场都会出现。现在，我们需要绘制图表。

```
 # Import networkx and initialize the graph.
import networkx as nx
graph = nx.Graph()
# Keep track of added nodes in this set so we don't add twice.
nodes = set()
# Iterate through each edge.
for k, weight in weights.items():
    try:
        # Split the source and dest ids and convert to integers.
        source, dest = k.split("_")
        source, dest = [int(source), int(dest)]
        # Add the source if it isn't in the nodes.
        if source not in nodes:
            graph.add_node(source)
        # Add the dest if it isn't in the nodes.
        if dest not in nodes:
            graph.add_node(dest)
        # Add both source and dest to the nodes set.
        # Sets don't allow duplicates.
        nodes.add(source)
        nodes.add(dest)
                # Add the edge to the graph.
        graph.add_edge(source, dest, weight=weight)
    except (ValueError, IndexError):
        passpos=nx.spring_layout(graph)
# Draw the nodes and edges.nx.draw_networkx_nodes(graph,pos, node_color='red', node_size=10, alpha=0.8)
nx.draw_networkx_edges(graph,pos,width=1.0,alpha=1)
# Show the plot.
plt.show() 
```py

![nxgraph](img/fa29a36440468a0369a75f93703be0e2.png)

## 结论

用于数据可视化的 Python 库激增，几乎可以实现任何类型的可视化。大多数库都建立在 matplotlib 之上，并简化了某些用例。如果您想更深入地了解如何使用 matplotlib、seaborn 和其他工具可视化数据，请查看我们的交互式

[探索性数据可视化](https://www.dataquest.io/course/exploratory-data-visualization/)和[通过数据可视化讲故事](https://www.dataquest.io/course/storytelling-data-visualization/)课程。

```