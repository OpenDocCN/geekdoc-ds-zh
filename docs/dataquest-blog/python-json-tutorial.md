# 教程:在 Python 中使用 Pandas 和 JSON 处理大型数据集

> 原文：<https://www.dataquest.io/blog/python-json-tutorial/>

March 1, 2016

处理大型 JSON 数据集可能是一件痛苦的事情，尤其是当它们太大而无法放入内存时。在这种情况下，命令行工具和 Python 的结合可以提供一种有效的方式来探索和分析数据。在这篇关注学习 python 编程的文章中，我们将看看如何利用像 Pandas 这样的工具来探索和绘制马里兰州蒙哥马利县的警察活动。我们将从查看 JSON 数据开始，然后继续用 Python 探索和分析 JSON。

当数据存储在 SQL 数据库中时，它往往遵循一种看起来像表的严格结构。下面是一个来自 SQLite 数据库的示例:

```py
id|code|name|area|area_land|area_water|population|population_growth|birth_rate|death_rate|migration_rate|created_at|updated_at1|af|Afghanistan|652230|652230|0|32564342|2.32|38.57|13.89|1.51|2015-11-01 13:19:49.461734|2015-11-01 13:19:49.4617342|al|Albania|28748|27398|1350|3029278|0.3|12.92|6.58|3.3|2015-11-01 13:19:54.431082|2015-11-01 13:19:54.4310823|ag|Algeria|2381741|2381741|0|39542166|1.84|23.67|4.31|0.92|2015-11-01 13:19:59.961286|2015-11-01 13:19:59.961286
```

如您所见，数据由行和列组成，其中每一列都映射到一个定义的属性，如`id`或`code`。在上面的数据集中，每行代表一个国家，每列代表该国的一些事实。

但是，随着我们捕获的数据量的增加，我们往往不知道数据在存储时的确切结构。这就是所谓的非结构化数据。一个很好的例子是网站上访问者的事件列表。以下是发送到服务器的事件列表示例:

```py
{'event_type': 'started-lesson', 'keen': {'created_at': '2015-06-12T23:09:03.966Z', 'id': '557b668fd2eaaa2e7c5e916b', 'timestamp': '2015-06-12T23:09:07.971Z'}, 'sequence': 1} {'event_type': 'started-screen', 'keen': {'created_at': '2015-06-12T23:09:03.979Z', 'id': '557b668f90e4bd26c10b6ed6', 'timestamp': '2015-06-12T23:09:07.987Z'}, 'lesson': 1, 'sequence': 4, 'type': 'code'} {'event_type': 'started-screen', 'keen': {'created_at': '2015-06-12T23:09:22.517Z', 'id': '557b66a246f9a7239038b1e0', 'timestamp': '2015-06-12T23:09:24.246Z'}, 'lesson': 1, 'sequence': 3, 'type': 'code'},
```

如您所见，上面列出了三个独立的事件。每个事件都有不同的字段，一些字段*嵌套在其他字段*中。这种类型的数据很难存储在常规 SQL 数据库中。这种非结构化数据通常以一种叫做[JavaScript Object Notation](https://json.org/)(JSON)的格式存储。JSON 是一种将列表和字典等数据结构编码成字符串的方法，可以确保它们易于被机器读取。尽管 JSON 以单词 Javascript 开头，但它实际上只是一种格式，可以被任何语言读取。

Python 有很好的 JSON 支持，有 [json](https://docs.python.org/3/library/json.html) 库。我们既可以将*列表*和*字典*转换为 JSON，也可以将字符串转换为*列表*和*字典*。JSON 数据看起来很像 Python 中的*字典*，其中存储了键和值。

在这篇文章中，我们将在命令行上探索一个 JSON 文件，然后将它导入 Python 并使用 Pandas 来处理它。

## 数据集

我们将查看包含马里兰州蒙哥马利县交通违规信息的数据集。你可以在这里下载数据。该数据包含违规发生的地点、汽车类型、收到违规的人的人口统计信息以及其他一些有趣的信息。使用该数据集，我们可以回答很多问题，包括:

*   什么类型的车最有可能因为超速被拦下来？
*   警察一天中什么时候最活跃？
*   “速度陷阱”有多普遍？还是说门票在地理上分布相当均匀？
*   人们被拦下来最常见的原因是什么？

不幸的是，我们事先不知道 JSON 文件的结构，所以我们需要做一些探索来弄清楚它。我们将使用 [Jupyter 笔记本](https://jupyter.org/)进行这次探索。

## 探索 JSON 数据

尽管 JSON 文件只有 600MB，但我们会把它当作大得多的文件来处理，这样我们就可以研究如何分析一个不适合内存的 JSON 文件。我们要做的第一件事是看一下`md_traffic.json`文件的前几行。JSON 文件只是一个普通的文本文件，因此我们可以使用所有标准的命令行工具与它进行交互:

```py
%%bashhead md_traffic.json
```

```py
<codeclass="language-bash">
{ "meta" : {
"view" : {
"id" : "4mse-ku6q",
"name" : "Traffic Violations",
"averageRating" : 0,
"category" : "Public Safety",
"createdAt" : 1403103517,
"description" : "This dataset contains traffic violation information from all electronic traffic violations issued in the County. Any information that can be used to uniquely identify the vehicle, the vehicle owner or the officer issuing the violation will not be published.\r\n\r\nUpdate Frequency: Daily",
"displayType" : "table",
```

由此，我们可以看出 JSON 数据是一个字典，并且格式良好。`meta`是顶级键，缩进两个空格。我们可以通过使用 [grep](https://en.wikipedia.org/wiki/Grep) 命令打印任何有两个前导空格的行来获得所有顶级密钥:

```py
%%bashgrep -E '^ {2}"' md_traffic.json
```

```py
"meta" : {"data" : [
[ 1889194, "92AD0076-5308-45D0-BDE3-6A3A55AD9A04", 1889194, 1455876689, "498050", 1455876689, "498050", null, "2016-02-18T00:00:00", "09:05:00", "MCP", "2nd district, Bethesda", "DRIVER USING HANDS TO USE HANDHELD TELEPHONE WHILEMOTOR VEHICLE IS IN MOTION", "355/TUCKERMAN LN", "-77.105925", "39.03223", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "MD", "02 - Automobile", "2010", "JEEP", "CRISWELL", "BLUE", "Citation", "21-1124.2(d2)", "Transportation Article", "No", "WHITE", "F", "GERMANTOWN", "MD", "MD", "A - Marked Patrol", [ "{\"address\":\"\",\"city\":\"\",\"state\":\"\",\"zip\":\"\"}", "-77.105925", "39.03223", null, false ] ]
```

这表明`meta`和`data`是`md_traffic.json`数据中的顶级键。列表的列表*似乎与`data`相关联，并且这可能包含我们的交通违规数据集中的每个记录。每个内部*列表*都是一条记录，第一条记录出现在`grep`命令的输出中。这与我们在操作 CSV 文件或 SQL 表时习惯使用的结构化数据非常相似。以下是数据的部分视图:*

```py
[ [1889194, "92AD0076-5308-45D0-BDE3-6A3A55AD9A04", 1889194, 1455876689, "498050"], [1889194, "92AD0076-5308-45D0-BDE3-6A3A55AD9A04", 1889194, 1455876689, "498050"], ...]
```

这看起来很像我们习惯使用的行和列。我们只是缺少了告诉我们每一列的含义的标题。我们也许能在`meta`键下找到这个信息。

`meta`通常指关于数据本身的信息。让我们更深入地研究一下`meta`，看看那里包含了什么信息。从`head`命令中，我们知道至少有`3`级按键，其中`meta`包含一个按键`view`，包含`id`、`name`、`averageRating`等按键。我们可以通过使用 grep 打印出任何带有`2-6`前导空格的行来打印出 JSON 文件的完整密钥结构:

```py
%%bashgrep -E '^ {2,6}"' md_traffic.json
```

```py
"meta" : { "view" : { "id" : "4mse-ku6q", "name" : "Traffic Violations", "averageRating" : 0, "category" : "Public Safety", "createdAt" : 1403103517, "description" : "This dataset contains traffic violation information from all electronic traffic violations issued in the County. Any information that can be used to uniquely identify the vehicle, the vehicle owner or the officer issuing the violation will not be published.\r\n\r\nUpdate Frequency: Daily", "displayType" : "table", "downloadCount" : 2851, "iconUrl" : "fileId:r41tDc239M1FL75LFwXFKzFCWqr8mzMeMTYXiA24USM", "indexUpdatedAt" : 1455885508, "newBackend" : false, "numberOfComments" : 0, "oid" : 8890705, "publicationAppendEnabled" : false, "publicationDate" : 1411040702, "publicationGroup" : 1620779, "publicationStage" : "published", "rowClass" : "", "rowsUpdatedAt" : 1455876727, "rowsUpdatedBy" : "ajn4-zy65", "state" : "normal", "tableId" : 1722160, "totalTimesRated" : 0, "viewCount" : 6910, "viewLastModified" : 1434558076, "viewType" : "tabular", "columns" : [ { "disabledFeatureFlags" : [ "allow_comments" ], "grants" : [ { "metadata" : { "owner" : { "query" : { "rights" : [ "read" ], "sortBys" : [ { "tableAuthor" : { "tags" : [ "traffic", "stop", "violations", "electronic issued." ], "flags" : [ "default" ]"data" : [ [ 1889194, "92AD0076-5308-45D0-BDE3-6A3A55AD9A04", 1889194, 1455876689, "498050", 1455876689, "498050", null, "2016-02-18T00:00:00", "09:05:00", "MCP", "2nd district, Bethesda", "DRIVER USING HANDS TO USE HANDHELD TELEPHONE WHILEMOTOR VEHICLE IS IN MOTION", "355/TUCKERMAN LN", "-77.105925", "39.03223", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "MD", "02 - Automobile", "2010", "JEEP", "CRISWELL", "BLUE", "Citation", "21-1124.2(d2)", "Transportation Article", "No", "WHITE", "F", "GERMANTOWN", "MD", "MD", "A - Marked Patrol", [ "{\"address\":\"\",\"city\":\"\",\"state\":\"\",\"zip\":\"\"}", "-77.105925", "39.03223", null, false ] ]
```

这向我们展示了与`md_traffic.json`相关的完整密钥结构，并告诉我们 JSON 文件的哪些部分与我们相关。在这种情况下，`columns`键看起来很有趣，因为它可能包含关于`data`键中的列表的*列表中的列的信息。*

## 提取列上的信息

现在我们知道了哪个键包含列的信息，我们需要读入这些信息。因为我们假设 JSON 文件不适合内存，所以我们不能使用 [json](https://docs.python.org/3/library/json.html) 库直接读取它。相反，我们需要以内存高效的方式迭代读取它。

我们可以使用 [ijson](https://pypi.python.org/pypi/ijson/) 包来完成这个任务。ijson 将迭代解析 json 文件，而不是一次读入所有内容。这比直接读入整个文件要慢，但它使我们能够处理内存中容纳不下的大文件。要使用 ijson，我们指定一个要从中提取数据的文件，然后指定一个要提取的关键路径:

```py
import ijson
filename = "md_traffic.json"
with open(filename, 'r') as f:
    objects = ijson.items(f, 'meta.view.columns.item')
    columns = list(objects)
```

在上面的代码中，我们打开了`md_traffic.json`文件，然后我们使用 ijson 中的`items`方法从文件中提取一个列表。我们使用`meta.view.columns`符号指定列表的路径。回想一下`meta`是一个顶级键，里面包含`view`，里面包含`columns`。然后，我们指定`meta.view.columns.item`来表示我们应该提取`meta.view.columns`列表中的每一项。`items`函数将返回一个[生成器](https://wiki.python.org/moin/Generators)，所以我们使用 list 方法将生成器转换成一个 Python 列表。我们可以打印出清单中的第一项:

```py
print(columns[0])
```

```py
{'renderTypeName': 'meta_data', 'name': 'sid', 'fieldName': ':sid', 'position': 0, 'id': -1, 'format': {}, 'dataTypeName': 'meta_data'}
```

从上面的输出来看，`columns`中的每一项都是一个字典，包含了每一列的信息。为了得到我们的头，看起来`fieldName`是要提取的相关键。要获得我们的列名，我们只需从`columns`中的每一项提取`fieldName`键:

```py
column_names = [col["fieldName"] for col in columns]column_names
```

```py
[':sid', ':id', ':position', ':created_at', ':created_meta', ':updated_at', ':updated_meta', ':meta', 'date_of_stop', 'time_of_stop', 'agency', 'subagency', 'description', 'location', 'latitude', 'longitude', 'accident', 'belts', 'personal_injury', 'property_damage', 'fatal', 'commercial_license', 'hazmat', 'commercial_vehicle', 'alcohol', 'work_zone', 'state', 'vehicle_type', 'year', 'make', 'model', 'color', 'violation_type', 'charge', 'article', 'contributed_to_accident', 'race', 'gender', 'driver_city', 'driver_state', 'dl_state', 'arrest_type', 'geolocation']
```

太好了！现在我们已经有了列名，我们可以开始提取数据本身了。

## 提取数据

您可能还记得数据被锁在`data`键中的列表的*列表中。我们需要将这些数据读入内存来操作它。幸运的是，我们可以使用刚刚提取的列名来只获取相关的列。这将节省大量空间。如果数据集更大，您可以迭代地处理成批的行。所以读取第一个`10000000`行，做一些处理，然后是下一个`10000000`，以此类推。在这种情况下，我们可以定义我们关心的列，并再次使用`ijson`迭代处理 JSON 文件:*

```py
good_columns = [
    "date_of_stop",
    "time_of_stop",
    "agency",
    "subagency",
    "description",
    "location",
    "latitude",
    "longitude",
    "vehicle_type",
    "year",
    "make",
    "model",
    "color",
    "violation_type",
    "race",
    "gender",
    "driver_state",
    "driver_city",
    "dl_state",
    "arrest_type"]
data = []
with open(filename, 'r') as f:
    objects = ijson.items(f, 'data.item')
    for row in objects:
        selected_row = []
        for item in good_columns:
            selected_row.append(row[column_names.index(item)])
            data.append(selected_row)
```

既然我们已经读入了数据，我们可以打印出`data`中的第一项:

```py
data[0]
```

```py
['2016-02-18T00:00:00', '09:05:00', 'MCP', '2nd district, Bethesda', 'DRIVER USING HANDS TO USE HANDHELD TELEPHONE WHILEMOTOR VEHICLE IS IN MOTION', '355/TUCKERMAN LN', '-77.105925', '39.03223', '02 - Automobile', '2010', 'JEEP', 'CRISWELL', 'BLUE', 'Citation', 'WHITE', 'F', 'MD', 'GERMANTOWN', 'MD', 'A - Marked Patrol']
```

## 将数据读入熊猫

现在我们有了作为列表的*列表的数据，以及作为*列表*的列标题，我们可以创建一个[熊猫](https://pandas.pydata.org/)数据框架来分析数据。如果您对 Pandas 不熟悉，它是一个数据分析库，使用一种称为 Dataframe 的高效表格数据结构来表示您的数据。Pandas 允许您将一个列表转换成一个数据框架，并单独指定列名。*

```py
import pandas as pd
stops = pd.DataFrame(data, columns=good_columns)
```

现在我们有了数据框架中的数据，我们可以做一些有趣的分析。这是一个根据汽车颜色划分的停靠站数量表:

```py
stops["color"].value_counts()
```

```py
BLACK 161319
SILVER 150650
WHITE 122887
GRAY 86322
RED 66282
BLUE 61867
GREEN 35802
GOLD 27808
TAN 18869
BLUE, DARK 17397
MAROON 15134
BLUE, LIGHT 11600
BEIGE 10522
GREEN, DK 10354
N/A 9594
GREEN, LGT 5214
BROWN 4041
YELLOW 3215
ORANGE 2967
BRONZE 1977
PURPLE 1725
MULTICOLOR 723
CREAM 608
COPPER 269
PINK 137
CHROME 21
CAMOUFLAGE 17
dtype: int64
```

迷彩似乎是一种*非常*流行的汽车颜色。下表列出了是哪种警察部门创造了这一记录:

```py
stops["arrest_type"].value_counts()
```

```py
A - Marked Patrol 671165
Q - Marked Laser 87929
B - Unmarked Patrol 25478
S - License Plate Recognition 11452
O - Foot Patrol 9604
L - Motorcycle 8559
E - Marked Stationary Radar 4854
R - Unmarked Laser 4331
G - Marked Moving Radar (Stationary) 2164
M - Marked (Off-Duty) 1301
I - Marked Moving Radar (Moving) 842
F - Unmarked Stationary Radar 420
C - Marked VASCAR 323
D - Unmarked VASCAR 210
P - Mounted Patrol 171
N - Unmarked (Off-Duty) 116
H - Unmarked Moving Radar (Stationary) 72
K - Aircraft Assist 41
J - Unmarked Moving Radar (Moving) 35
dtype: int64
```

随着闯红灯相机和高速激光的兴起，有趣的是，巡逻车仍然是迄今为止最主要的引用来源。

## 转换列

我们现在几乎准备好进行一些基于时间和位置的分析，但是我们需要首先将`longitude`、`latitude`和`date`列从字符串转换为浮点。我们可以使用下面的代码来转换`latitude`和`longitude`:

```py
import numpy as np
def parse_float(x):
    try:
        x = float(x)
    except Exception:
        x = 0
    return x
stops["longitude"] = stops["longitude"].apply(parse_float)
stops["latitude"] = stops["latitude"].apply(parse_float)
```

奇怪的是，一天中的时间和停止的日期存储在两个单独的列中，`time_of_stop`和`date_of_stop`。我们将对两者进行解析，并将它们转换成一个日期时间列:

```py
import datetime
def parse_full_date(row):
    date = datetime.datetime.strptime(row["date_of_stop"], "%Y-%m-%dT%H:%M:%S")
    time = row["time_of_stop"].split(":")
    date = date.replace(hour=int(time[0]), minute = int(time[1]), second = int(time[2]))
    return date
stops["date"] = stops.apply(parse_full_date, axis=1)
```

我们现在可以画出哪一天的交通堵塞最多:

```py
import matplotlib.pyplot as plt
%matplotlib inline plt.hist(stops["date"].dt.weekday, bins=6)
```

```py
(array([ 112816., 142048., 133363., 127629., 131735., 181476.]), array([ 0., 1., 2., 3., 4., 5., 6.]), <a list of 6 Patch objects>)
```

![](img/b9befb22f6ef8990058494a0b8d203ae.png)

在这个情节中，周一是`0`，周日是`6`。看起来周日停得最多，周一最少。这也可能是数据质量问题，因为某些原因导致无效日期出现在周日。你必须更深入地挖掘`date_of_stop`专栏才能最终找到答案(这超出了本文的范围)。

我们还可以绘制出最常见的交通停车时间:

```py
plt.hist(stops["date"].dt.hour, bins=24)
```

```py
(array([ 44125., 35866., 27274., 18048., 11271., 7796., 11959., 29853., 46306., 42799., 43775., 37101., 34424., 34493., 36006., 29634., 39024., 40988., 32511., 28715., 31983.,  43957., 60734., 60425.]), array([ 0\. , 0.95833333, 1.91666667, 2.875 , 3.83333333, 4.79166667, 5.75 , 6.70833333, 7.66666667, 8.625 , 9.58333333, 10.54166667, 11.5 , 12.45833333, 13.41666667, 14.375 , 15.33333333, 16.29166667, 17.25 , 18.20833333, 19.16666667, 20.125 , 21.08333333, 22.04166667, 23\. ]), <a list of 24 Patch objects>)
```

![](img/45eb9e6123cb64fdc139e9b940942f31.png)

看起来大多数停靠站发生在午夜左右，最少发生在凌晨 5 点左右。这可能是有道理的，因为人们在深夜从酒吧和晚宴开车回家，可能会受到损害。这也可能是一个数据质量问题，为了得到完整的答案，有必要浏览一下`time_of_stop`栏。

## 设置站点子集

现在我们已经转换了位置和日期列，我们可以绘制出交通站点。因为映射在 CPU 资源和内存方面非常密集，所以我们需要首先过滤掉从`stops`开始使用的行:

```py
last_year = stops[stops["date"] > datetime.datetime(year=2015, month=2, day=18)]
```

在上面的代码中，我们选择了过去一年中出现的所有行。我们可以进一步缩小范围，只选择发生在高峰时间(每个人都去上班的早晨)的行:

```py
morning_rush = last_year[(last_year["date"].dt.weekday < 5) & (last_year["date"].dt.hour > 5) & (last_year["date"].dt.hour < 10)]
print(morning_rush.shape)last_year.shape
```

```py
(29824, 21)
```

```py
(234582, 21)
```

使用优秀的[叶子](https://github.com/python-visualization/folium)包，我们现在可以看到所有的停止发生在哪里。通过利用[传单](https://leafletjs.com/)，Folium 允许你用 Python 轻松地创建交互式地图。为了保持性能，我们将只显示`morning_rush`的前`1000`行:

```py
import folium
from folium import plugins
stops_map = folium.Map(location=[39.0836, -77.1483], zoom_start=11)
marker_cluster = folium.MarkerCluster().add_to(stops_map)
for name, row in morning_rush.iloc[:1000].iterrows():
    folium.Marker([row["longitude"], row["latitude"]], popup=row["description"]).add_to(marker_cluster)
stops_map.create_map('stops.html')stops_map
```

[https://www.dataquest.io/blog/large_files/json_folium.html](https://www.dataquest.io/blog/large_files/json_folium.html)

这说明很多交通站点都集中在县城右下方周围。我们可以通过热图进一步扩展我们的分析:

```py
stops_heatmap = folium.Map(location=[39.0836, -77.1483], zoom_start=11)stops_heatmap.add_children(plugins.HeatMap([`, row["latitude"]] for name, row in morning_rush.iloc[:1000].iterrows()]))stops_heatmap.save("heatmap.html")stops_heatmap`

处理大型 JSON 数据集可能是一件痛苦的事情，尤其是当它们太大而无法放入内存时。在这种情况下，命令行工具和 Python 的结合可以提供一种有效的方式来探索和分析数据。在这篇关注学习 python 编程的文章中，我们将看看如何利用像 Pandas 这样的工具来探索和绘制马里兰州蒙哥马利县的警察活动。我们将从查看 JSON 数据开始，然后继续用 Python 探索和分析 JSON。

当数据存储在 SQL 数据库中时，它往往遵循一种看起来像表的严格结构。下面是一个来自 SQLite 数据库的示例:

```
id|code|name|area|area_land|area_water|population|population_growth|birth_rate|death_rate|migration_rate|created_at|updated_at1|af|Afghanistan|652230|652230|0|32564342|2.32|38.57|13.89|1.51|2015-11-01 13:19:49.461734|2015-11-01 13:19:49.4617342|al|Albania|28748|27398|1350|3029278|0.3|12.92|6.58|3.3|2015-11-01 13:19:54.431082|2015-11-01 13:19:54.4310823|ag|Algeria|2381741|2381741|0|39542166|1.84|23.67|4.31|0.92|2015-11-01 13:19:59.961286|2015-11-01 13:19:59.961286
```py

如您所见，数据由行和列组成，其中每一列都映射到一个定义的属性，如`id`或`code`。在上面的数据集中，每行代表一个国家，每列代表该国的一些事实。

但是，随着我们捕获的数据量的增加，我们往往不知道数据在存储时的确切结构。这就是所谓的非结构化数据。一个很好的例子是网站上访问者的事件列表。以下是发送到服务器的事件列表示例:

```
{'event_type': 'started-lesson', 'keen': {'created_at': '2015-06-12T23:09:03.966Z', 'id': '557b668fd2eaaa2e7c5e916b', 'timestamp': '2015-06-12T23:09:07.971Z'}, 'sequence': 1} {'event_type': 'started-screen', 'keen': {'created_at': '2015-06-12T23:09:03.979Z', 'id': '557b668f90e4bd26c10b6ed6', 'timestamp': '2015-06-12T23:09:07.987Z'}, 'lesson': 1, 'sequence': 4, 'type': 'code'} {'event_type': 'started-screen', 'keen': {'created_at': '2015-06-12T23:09:22.517Z', 'id': '557b66a246f9a7239038b1e0', 'timestamp': '2015-06-12T23:09:24.246Z'}, 'lesson': 1, 'sequence': 3, 'type': 'code'},
```py

如您所见，上面列出了三个独立的事件。每个事件都有不同的字段，一些字段*嵌套在其他字段*中。这种类型的数据很难存储在常规 SQL 数据库中。这种非结构化数据通常以一种叫做[JavaScript Object Notation](https://json.org/)(JSON)的格式存储。JSON 是一种将列表和字典等数据结构编码成字符串的方法，可以确保它们易于被机器读取。尽管 JSON 以单词 Javascript 开头，但它实际上只是一种格式，可以被任何语言读取。

Python 有很好的 JSON 支持，有 [json](https://docs.python.org/3/library/json.html) 库。我们既可以将*列表*和*字典*转换为 JSON，也可以将字符串转换为*列表*和*字典*。JSON 数据看起来很像 Python 中的*字典*，其中存储了键和值。

在这篇文章中，我们将在命令行上探索一个 JSON 文件，然后将它导入 Python 并使用 Pandas 来处理它。

## 数据集

我们将查看包含马里兰州蒙哥马利县交通违规信息的数据集。你可以在这里下载数据[。该数据包含违规发生的地点、汽车类型、收到违规的人的人口统计信息以及其他一些有趣的信息。使用该数据集，我们可以回答很多问题，包括:](https://catalog.data.gov/dataset/traffic-violations-56dda)

*   什么类型的车最有可能因为超速被拦下来？

*   警察一天中什么时候最活跃？

*   “速度陷阱”有多普遍？还是说门票在地理上分布相当均匀？

*   人们被拦下来最常见的原因是什么？

不幸的是，我们事先不知道 JSON 文件的结构，所以我们需要做一些探索来弄清楚它。我们将使用 [Jupyter 笔记本](https://jupyter.org/)进行这次探索。

## 探索 JSON 数据

尽管 JSON 文件只有 600MB，但我们会把它当作大得多的文件来处理，这样我们就可以研究如何分析一个不适合内存的 JSON 文件。我们要做的第一件事是看一下`md_traffic.json`文件的前几行。JSON 文件只是一个普通的文本文件，因此我们可以使用所有标准的命令行工具与它进行交互:

```
%%bashhead md_traffic.json
```py

```
<codeclass="language-bash">
{ "meta" : {
"view" : {
"id" : "4mse-ku6q",
"name" : "Traffic Violations",
"averageRating" : 0,
"category" : "Public Safety",
"createdAt" : 1403103517,
"description" : "This dataset contains traffic violation information from all electronic traffic violations issued in the County. Any information that can be used to uniquely identify the vehicle, the vehicle owner or the officer issuing the violation will not be published.\r\n\r\nUpdate Frequency: Daily",
"displayType" : "table",
```py

由此，我们可以看出 JSON 数据是一个字典，并且格式良好。`meta`是顶级键，缩进两个空格。我们可以通过使用 [grep](https://en.wikipedia.org/wiki/Grep) 命令打印任何有两个前导空格的行来获得所有顶级密钥:

```
%%bashgrep -E '^ {2}"' md_traffic.json
```py

```
"meta" : {"data" : [
[ 1889194, "92AD0076-5308-45D0-BDE3-6A3A55AD9A04", 1889194, 1455876689, "498050", 1455876689, "498050", null, "2016-02-18T00:00:00", "09:05:00", "MCP", "2nd district, Bethesda", "DRIVER USING HANDS TO USE HANDHELD TELEPHONE WHILEMOTOR VEHICLE IS IN MOTION", "355/TUCKERMAN LN", "-77.105925", "39.03223", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "MD", "02 - Automobile", "2010", "JEEP", "CRISWELL", "BLUE", "Citation", "21-1124.2(d2)", "Transportation Article", "No", "WHITE", "F", "GERMANTOWN", "MD", "MD", "A - Marked Patrol", [ "{\"address\":\"\",\"city\":\"\",\"state\":\"\",\"zip\":\"\"}", "-77.105925", "39.03223", null, false ] ]
```py

这表明`meta`和`data`是`md_traffic.json`数据中的顶级键。列表的列表*似乎与`data`相关联，并且这可能包含我们的交通违规数据集中的每个记录。每个内部*列表*都是一条记录，第一条记录出现在`grep`命令的输出中。这与我们在操作 CSV 文件或 SQL 表时习惯使用的结构化数据非常相似。以下是数据的部分视图:*

```
[ [1889194, "92AD0076-5308-45D0-BDE3-6A3A55AD9A04", 1889194, 1455876689, "498050"], [1889194, "92AD0076-5308-45D0-BDE3-6A3A55AD9A04", 1889194, 1455876689, "498050"], ...]
```py

这看起来很像我们习惯使用的行和列。我们只是缺少了告诉我们每一列的含义的标题。我们也许能在`meta`键下找到这个信息。

`meta`通常指关于数据本身的信息。让我们更深入地研究一下`meta`，看看那里包含了什么信息。从`head`命令中，我们知道至少有`3`级按键，其中`meta`包含一个按键`view`，包含`id`、`name`、`averageRating`等按键。我们可以通过使用 grep 打印出任何带有`2-6`前导空格的行来打印出 JSON 文件的完整密钥结构:

```
%%bashgrep -E '^ {2,6}"' md_traffic.json
```py

```
"meta" : { "view" : { "id" : "4mse-ku6q", "name" : "Traffic Violations", "averageRating" : 0, "category" : "Public Safety", "createdAt" : 1403103517, "description" : "This dataset contains traffic violation information from all electronic traffic violations issued in the County. Any information that can be used to uniquely identify the vehicle, the vehicle owner or the officer issuing the violation will not be published.\r\n\r\nUpdate Frequency: Daily", "displayType" : "table", "downloadCount" : 2851, "iconUrl" : "fileId:r41tDc239M1FL75LFwXFKzFCWqr8mzMeMTYXiA24USM", "indexUpdatedAt" : 1455885508, "newBackend" : false, "numberOfComments" : 0, "oid" : 8890705, "publicationAppendEnabled" : false, "publicationDate" : 1411040702, "publicationGroup" : 1620779, "publicationStage" : "published", "rowClass" : "", "rowsUpdatedAt" : 1455876727, "rowsUpdatedBy" : "ajn4-zy65", "state" : "normal", "tableId" : 1722160, "totalTimesRated" : 0, "viewCount" : 6910, "viewLastModified" : 1434558076, "viewType" : "tabular", "columns" : [ { "disabledFeatureFlags" : [ "allow_comments" ], "grants" : [ { "metadata" : { "owner" : { "query" : { "rights" : [ "read" ], "sortBys" : [ { "tableAuthor" : { "tags" : [ "traffic", "stop", "violations", "electronic issued." ], "flags" : [ "default" ]"data" : [ [ 1889194, "92AD0076-5308-45D0-BDE3-6A3A55AD9A04", 1889194, 1455876689, "498050", 1455876689, "498050", null, "2016-02-18T00:00:00", "09:05:00", "MCP", "2nd district, Bethesda", "DRIVER USING HANDS TO USE HANDHELD TELEPHONE WHILEMOTOR VEHICLE IS IN MOTION", "355/TUCKERMAN LN", "-77.105925", "39.03223", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "MD", "02 - Automobile", "2010", "JEEP", "CRISWELL", "BLUE", "Citation", "21-1124.2(d2)", "Transportation Article", "No", "WHITE", "F", "GERMANTOWN", "MD", "MD", "A - Marked Patrol", [ "{\"address\":\"\",\"city\":\"\",\"state\":\"\",\"zip\":\"\"}", "-77.105925", "39.03223", null, false ] ]
```py

这向我们展示了与`md_traffic.json`相关的完整密钥结构，并告诉我们 JSON 文件的哪些部分与我们相关。在这种情况下，`columns`键看起来很有趣，因为它可能包含关于`data`键中的列表的*列表中的列的信息。*

## 提取列上的信息

现在我们知道了哪个键包含列的信息，我们需要读入这些信息。因为我们假设 JSON 文件不适合内存，所以我们不能使用 [json](https://docs.python.org/3/library/json.html) 库直接读取它。相反，我们需要以内存高效的方式迭代读取它。

我们可以使用 [ijson](https://pypi.python.org/pypi/ijson/) 包来完成这个任务。ijson 将迭代解析 json 文件，而不是一次读入所有内容。这比直接读入整个文件要慢，但它使我们能够处理内存中容纳不下的大文件。要使用 ijson，我们指定一个要从中提取数据的文件，然后指定一个要提取的关键路径:

```
import ijson
filename = "md_traffic.json"
with open(filename, 'r') as f:
    objects = ijson.items(f, 'meta.view.columns.item')
    columns = list(objects)
```py

在上面的代码中，我们打开了`md_traffic.json`文件，然后我们使用 ijson 中的`items`方法从文件中提取一个列表。我们使用`meta.view.columns`符号指定列表的路径。回想一下`meta`是一个顶级键，里面包含`view`，里面包含`columns`。然后，我们指定`meta.view.columns.item`来表示我们应该提取`meta.view.columns`列表中的每一项。`items`函数将返回一个[生成器](https://wiki.python.org/moin/Generators)，所以我们使用 list 方法将生成器转换成一个 Python 列表。我们可以打印出清单中的第一项:

```
print(columns[0])
```py

```
{'renderTypeName': 'meta_data', 'name': 'sid', 'fieldName': ':sid', 'position': 0, 'id': -1, 'format': {}, 'dataTypeName': 'meta_data'}
```py

从上面的输出来看，`columns`中的每一项都是一个字典，包含了每一列的信息。为了得到我们的头，看起来`fieldName`是要提取的相关键。要获得我们的列名，我们只需从`columns`中的每一项提取`fieldName`键:

```
column_names = [col["fieldName"] for col in columns]column_names
```py

```
[':sid', ':id', ':position', ':created_at', ':created_meta', ':updated_at', ':updated_meta', ':meta', 'date_of_stop', 'time_of_stop', 'agency', 'subagency', 'description', 'location', 'latitude', 'longitude', 'accident', 'belts', 'personal_injury', 'property_damage', 'fatal', 'commercial_license', 'hazmat', 'commercial_vehicle', 'alcohol', 'work_zone', 'state', 'vehicle_type', 'year', 'make', 'model', 'color', 'violation_type', 'charge', 'article', 'contributed_to_accident', 'race', 'gender', 'driver_city', 'driver_state', 'dl_state', 'arrest_type', 'geolocation']
```py

太好了！现在我们已经有了列名，我们可以开始提取数据本身了。

## 提取数据

您可能还记得数据被锁在`data`键中的列表的*列表中。我们需要将这些数据读入内存来操作它。幸运的是，我们可以使用刚刚提取的列名来只获取相关的列。这将节省大量空间。如果数据集更大，您可以迭代地处理成批的行。所以读取第一个`10000000`行，做一些处理，然后是下一个`10000000`，以此类推。在这种情况下，我们可以定义我们关心的列，并再次使用`ijson`迭代处理 JSON 文件:*

```
good_columns = [
    "date_of_stop",
    "time_of_stop",
    "agency",
    "subagency",
    "description",
    "location",
    "latitude",
    "longitude",
    "vehicle_type",
    "year",
    "make",
    "model",
    "color",
    "violation_type",
    "race",
    "gender",
    "driver_state",
    "driver_city",
    "dl_state",
    "arrest_type"]
data = []
with open(filename, 'r') as f:
    objects = ijson.items(f, 'data.item')
    for row in objects:
        selected_row = []
        for item in good_columns:
            selected_row.append(row[column_names.index(item)])
            data.append(selected_row)
```py

既然我们已经读入了数据，我们可以打印出`data`中的第一项:

```
data[0]
```py

```
['2016-02-18T00:00:00', '09:05:00', 'MCP', '2nd district, Bethesda', 'DRIVER USING HANDS TO USE HANDHELD TELEPHONE WHILEMOTOR VEHICLE IS IN MOTION', '355/TUCKERMAN LN', '-77.105925', '39.03223', '02 - Automobile', '2010', 'JEEP', 'CRISWELL', 'BLUE', 'Citation', 'WHITE', 'F', 'MD', 'GERMANTOWN', 'MD', 'A - Marked Patrol']
```py

## 将数据读入熊猫

现在我们有了作为列表的*列表的数据，以及作为*列表*的列标题，我们可以创建一个[熊猫](https://pandas.pydata.org/)数据框架来分析数据。如果您对 Pandas 不熟悉，它是一个数据分析库，使用一种称为 Dataframe 的高效表格数据结构来表示您的数据。Pandas 允许您将一个列表转换成一个数据框架，并单独指定列名。*

```
import pandas as pd
stops = pd.DataFrame(data, columns=good_columns)
```py

现在我们有了数据框架中的数据，我们可以做一些有趣的分析。这是一个根据汽车颜色划分的停靠站数量表:

```
stops["color"].value_counts()
```py

```
BLACK 161319
SILVER 150650
WHITE 122887
GRAY 86322
RED 66282
BLUE 61867
GREEN 35802
GOLD 27808
TAN 18869
BLUE, DARK 17397
MAROON 15134
BLUE, LIGHT 11600
BEIGE 10522
GREEN, DK 10354
N/A 9594
GREEN, LGT 5214
BROWN 4041
YELLOW 3215
ORANGE 2967
BRONZE 1977
PURPLE 1725
MULTICOLOR 723
CREAM 608
COPPER 269
PINK 137
CHROME 21
CAMOUFLAGE 17
dtype: int64
```py

迷彩似乎是一种*非常*流行的汽车颜色。下表列出了是哪种警察部门创造了这一记录:

```
stops["arrest_type"].value_counts()
```py

```
A - Marked Patrol 671165
Q - Marked Laser 87929
B - Unmarked Patrol 25478
S - License Plate Recognition 11452
O - Foot Patrol 9604
L - Motorcycle 8559
E - Marked Stationary Radar 4854
R - Unmarked Laser 4331
G - Marked Moving Radar (Stationary) 2164
M - Marked (Off-Duty) 1301
I - Marked Moving Radar (Moving) 842
F - Unmarked Stationary Radar 420
C - Marked VASCAR 323
D - Unmarked VASCAR 210
P - Mounted Patrol 171
N - Unmarked (Off-Duty) 116
H - Unmarked Moving Radar (Stationary) 72
K - Aircraft Assist 41
J - Unmarked Moving Radar (Moving) 35
dtype: int64
```py

随着闯红灯相机和高速激光的兴起，有趣的是，巡逻车仍然是迄今为止最主要的引用来源。

## 转换列

我们现在几乎准备好进行一些基于时间和位置的分析，但是我们需要首先将`longitude`、`latitude`和`date`列从字符串转换为浮点。我们可以使用下面的代码来转换`latitude`和`longitude`:

```
import numpy as np
def parse_float(x):
    try:
        x = float(x)
    except Exception:
        x = 0
    return x
stops["longitude"] = stops["longitude"].apply(parse_float)
stops["latitude"] = stops["latitude"].apply(parse_float)
```py

奇怪的是，一天中的时间和停止的日期存储在两个单独的列中，`time_of_stop`和`date_of_stop`。我们将对两者进行解析，并将它们转换成一个日期时间列:

```
import datetime
def parse_full_date(row):
    date = datetime.datetime.strptime(row["date_of_stop"], "%Y-%m-%dT%H:%M:%S")
    time = row["time_of_stop"].split(":")
    date = date.replace(hour=int(time[0]), minute = int(time[1]), second = int(time[2]))
    return date
stops["date"] = stops.apply(parse_full_date, axis=1)
```py

我们现在可以画出哪一天的交通堵塞最多:

```
import matplotlib.pyplot as plt
%matplotlib inline plt.hist(stops["date"].dt.weekday, bins=6)
```py

```
(array([ 112816., 142048., 133363., 127629., 131735., 181476.]), array([ 0., 1., 2., 3., 4., 5., 6.]), <a list of 6 Patch objects>)
```py

![](img/b9befb22f6ef8990058494a0b8d203ae.png)

在这个情节中，周一是`0`，周日是`6`。看起来周日停得最多，周一最少。这也可能是数据质量问题，因为某些原因导致无效日期出现在周日。你必须更深入地挖掘`date_of_stop`专栏才能最终找到答案(这超出了本文的范围)。

我们还可以绘制出最常见的交通停车时间:

```
plt.hist(stops["date"].dt.hour, bins=24)
```py

```
(array([ 44125., 35866., 27274., 18048., 11271., 7796., 11959., 29853., 46306., 42799., 43775., 37101., 34424., 34493., 36006., 29634., 39024., 40988., 32511., 28715., 31983.,  43957., 60734., 60425.]), array([ 0\. , 0.95833333, 1.91666667, 2.875 , 3.83333333, 4.79166667, 5.75 , 6.70833333, 7.66666667, 8.625 , 9.58333333, 10.54166667, 11.5 , 12.45833333, 13.41666667, 14.375 , 15.33333333, 16.29166667, 17.25 , 18.20833333, 19.16666667, 20.125 , 21.08333333, 22.04166667, 23\. ]), <a list of 24 Patch objects>)
```py

![](img/45eb9e6123cb64fdc139e9b940942f31.png)

看起来大多数停靠站发生在午夜左右，最少发生在凌晨 5 点左右。这可能是有道理的，因为人们在深夜从酒吧和晚宴开车回家，可能会受到损害。这也可能是一个数据质量问题，为了得到完整的答案，有必要浏览一下`time_of_stop`栏。

## 设置站点子集

现在我们已经转换了位置和日期列，我们可以绘制出交通站点。因为映射在 CPU 资源和内存方面非常密集，所以我们需要首先过滤掉从`stops`开始使用的行:

```
last_year = stops[stops["date"] > datetime.datetime(year=2015, month=2, day=18)]
```py

在上面的代码中，我们选择了过去一年中出现的所有行。我们可以进一步缩小范围，只选择发生在高峰时间(每个人都去上班的早晨)的行:

```
morning_rush = last_year[(last_year["date"].dt.weekday < 5) & (last_year["date"].dt.hour > 5) & (last_year["date"].dt.hour < 10)]
print(morning_rush.shape)last_year.shape
```py

```
(29824, 21)
```py

```
(234582, 21)
```py

使用优秀的[叶子](https://github.com/python-visualization/folium)包，我们现在可以看到所有的停止发生在哪里。通过利用[传单](https://leafletjs.com/)，Folium 允许你用 Python 轻松地创建交互式地图。为了保持性能，我们将只显示`morning_rush`的前`1000`行:

```
import folium
from folium import plugins
stops_map = folium.Map(location=[39.0836, -77.1483], zoom_start=11)
marker_cluster = folium.MarkerCluster().add_to(stops_map)
for name, row in morning_rush.iloc[:1000].iterrows():
    folium.Marker([row["longitude"], row["latitude"]], popup=row["description"]).add_to(marker_cluster)
stops_map.create_map('stops.html')stops_map
```py

[https://www.dataquest.io/blog/large_files/json_folium.html](https://www.dataquest.io/blog/large_files/json_folium.html)

这说明很多交通站点都集中在县城右下方周围。我们可以通过热图进一步扩展我们的分析:

```
stops_heatmap = folium.Map(location=[39.0836, -77.1483], zoom_start=11)stops_heatmap.add_children(plugins.HeatMap([`, row["latitude"]] for name, row in morning_rush.iloc[:1000].iterrows()]))stops_heatmap.save("heatmap.html")stops_heatmap`

[data:text/html;base64,CiAgICAgICAgPCFET0NUWVBFIGh0bWw+CiAgICAgICAgPGhlYWQ+CiAgICAgICAgICAgIAogICAgICAgIAogICAgICAgICAgICA8bWV0YSBodHRwLWVxdWl2PSJjb250ZW50LXR5cGUiIGNvbnRlbnQ9InRleHQvaHRtbDsgY2hhcnNldD1VVEYtOCIgLz4KICAgICAgICAKICAgICAgICAgICAgCiAgICAgICAgCiAgICAgICAgICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9sZWFmbGV0LzAuNy4zL2xlYWZsZXQuanMiPjwvc2NyaXB0PgogICAgICAgIAogICAgICAgIAogICAgICAgIAogICAgICAgICAgICAKICAgICAgICAKICAgICAgICAgICAgPHNjcmlwdCBzcmM9Imh0dHBzOi8vYWpheC5nb29nbGVhcGlzLmNvbS9hamF4L2xpYnMvanF1ZXJ5LzEuMTEuMS9qcXVlcnkubWluLmpzIj48L3NjcmlwdD4KICAgICAgICAKICAgICAgICAKICAgICAgICAKICAgICAgICAgICAgCiAgICAgICAgCiAgICAgICAgICAgIDxzY3JpcHQgc3JjPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9qcy9ib290c3RyYXAubWluLmpzIj48L3NjcmlwdD4KICAgICAgICAKICAgICAgICAKICAgICAgICAKICAgICAgICAgICAgCiAgICAgICAgCiAgICAgICAgICAgIDxzY3JpcHQgc3JjPSJodHRwczovL3Jhd2dpdGh1Yi5jb20vbHZvb2dkdC9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAvZGV2ZWxvcC9kaXN0L2xlYWZsZXQuYXdlc29tZS1tYXJrZXJzLmpzIj48L3NjcmlwdD4KICAgICAgICAKICAgICAgICAKICAgICAgICAKICAgICAgICAgICAgCiAgICAgICAgCiAgICAgICAgICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9sZWFmbGV0Lm1hcmtlcmNsdXN0ZXIvMC40LjAvbGVhZmxldC5tYXJrZXJjbHVzdGVyLXNyYy5qcyI+PC9zY3JpcHQ+CiAgICAgICAgCiAgICAgICAgCiAgICAgICAgCiAgICAgICAgICAgIAogICAgICAgIAogICAgICAgICAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvbGVhZmxldC5tYXJrZXJjbHVzdGVyLzAuNC4wL2xlYWZsZXQubWFya2VyY2x1c3Rlci5qcyI+PC9zY3JpcHQ+CiAgICAgICAgCiAgICAgICAgCiAgICAgICAgCiAgICAgICAgICAgIAogICAgICAgIAogICAgICAgICAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vY2RuanMuY2xvdWRmbGFyZS5jb20vYWpheC9saWJzL2xlYWZsZXQvMC43LjMvbGVhZmxldC5jc3MiIC8+CiAgICAgICAgCiAgICAgICAgCiAgICAgICAgCiAgICAgICAgICAgIAogICAgICAgIAogICAgICAgICAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vYm9vdHN0cmFwLzMuMi4wL2Nzcy9ib290c3RyYXAubWluLmNzcyIgLz4KICAgICAgICAKICAgICAgICAKICAgICAgICAKICAgICAgICAgICAgCiAgICAgICAgCiAgICAgICAgICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIiAvPgogICAgICAgIAogICAgICAgIAogICAgICAgIAogICAgICAgICAgICAKICAgICAgICAKICAgICAgICAgICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2ZvbnQtYXdlc29tZS80LjEuMC9jc3MvZm9udC1hd2Vzb21lLm1pbi5jc3MiIC8+CiAgICAgICAgCiAgICAgICAgCiAgICAgICAgCiAgICAgICAgICAgIAogICAgICAgIAogICAgICAgICAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vcmF3Z2l0LmNvbS9sdm9vZ2R0L0xlYWZsZXQuYXdlc29tZS1tYXJrZXJzLzIuMC9kZXZlbG9wL2Rpc3QvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuY3NzIiAvPgogICAgICAgIAogICAgICAgIAogICAgICAgIAogICAgICAgICAgICAKICAgICAgICAKICAgICAgICAgICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9sZWFmbGV0Lm1hcmtlcmNsdXN0ZXIvMC40LjAvTWFya2VyQ2x1c3Rlci5EZWZhdWx0LmNzcyIgLz4KICAgICAgICAKICAgICAgICAKICAgICAgICAKICAgICAgICAgICAgCiAgICAgICAgCiAgICAgICAgICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvbGVhZmxldC5tYXJrZXJjbHVzdGVyLzAuNC4wL01hcmtlckNsdXN0ZXIuY3NzIiAvPgogICAgICAgIAogICAgICAgIAogICAgICAgIAogICAgICAgICAgICAKICAgICAgICAKICAgICAgICAgICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL3Jhdy5naXRodWJ1c2VyY29udGVudC5jb20vcHl0aG9uLXZpc3VhbGl6YXRpb24vZm9saXVtL21hc3Rlci9mb2xpdW0vdGVtcGxhdGVzL2xlYWZsZXQuYXdlc29tZS5yb3RhdGUuY3NzIiAvPgogICAgICAgIAogICAgICAgIAogICAgICAgIAogICAgICAgICAgICAKICAgICAgICAgICAgPHN0eWxlPgoKICAgICAgICAgICAgaHRtbCwgYm9keSB7CiAgICAgICAgICAgICAgICB3aWR0aDogMTAwJTsKICAgICAgICAgICAgICAgIGhlaWdodDogMTAwJTsKICAgICAgICAgICAgICAgIG1hcmdpbjogMDsKICAgICAgICAgICAgICAgIHBhZGRpbmc6IDA7CiAgICAgICAgICAgICAgICB9CgogICAgICAgICAgICAjbWFwIHsKICAgICAgICAgICAgICAgIHBvc2l0aW9uOmFic29sdXRlOwogICAgICAgICAgICAgICAgdG9wOjA7CiAgICAgICAgICAgICAgICBib3R0b206MDsKICAgICAgICAgICAgICAgIHJpZ2h0OjA7CiAgICAgICAgICAgICAgICBsZWZ0OjA7CiAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgIDwvc3R5bGU+CiAgICAgICAgICAgIAogICAgICAgIAogICAgICAgICAgICAKICAgICAgICAgICAgPHN0eWxlPiAjbWFwX2EzMmQ2NzVlN2Y5MDQ5YTlhMTQzOTk4MjkwMzE0YTk4IHsKICAgICAgICAgICAgICAgIHBvc2l0aW9uIDogcmVsYXRpdmU7CiAgICAgICAgICAgICAgICB3aWR0aCA6IDEwMC4wJTsKICAgICAgICAgICAgICAgIGhlaWdodDogMTAwLjAlOwogICAgICAgICAgICAgICAgbGVmdDogMC4wJTsKICAgICAgICAgICAgICAgIHRvcDogMC4wJTsKICAgICAgICAgICAgICAgIH0KICAgICAgICAgICAgPC9zdHlsZT4KICAgICAgICAKICAgICAgICAKICAgICAgICAgICAgCiAgICAgICAgCiAgICAgICAgICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2xlYWZsZXQuZ2l0aHViLmlvL0xlYWZsZXQuaGVhdC9kaXN0L2xlYWZsZXQtaGVhdC5qcyI+PC9zY3JpcHQ+CiAgICAgICAgCiAgICAgICAgCiAgICAgICAgCiAgICAgICAgCiAgICAgICAgPC9oZWFkPgogICAgICAgIDxib2R5PgogICAgICAgICAgICAKICAgICAgICAKICAgICAgICAgICAgCiAgICAgICAgICAgIDxkaXYgY2xhc3M9ImZvbGl1bS1tYXAiIGlkPSJtYXBfYTMyZDY3NWU3ZjkwNDlhOWExNDM5OTgyOTAzMTRhOTgiID48L2Rpdj4KICAgICAgICAKICAgICAgICAKICAgICAgICAKICAgICAgICA8L2JvZHk+CiAgICAgICAgPHNjcmlwdD4KICAgICAgICAgICAgCiAgICAgICAgCiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIHNvdXRoV2VzdCA9IEwubGF0TG5nKC05MCwgLTE4MCk7CiAgICAgICAgICAgIHZhciBub3J0aEVhc3QgPSBMLmxhdExuZyg5MCwgMTgwKTsKICAgICAgICAgICAgdmFyIGJvdW5kcyA9IEwubGF0TG5nQm91bmRzKHNvdXRoV2VzdCwgbm9ydGhFYXN0KTsKCiAgICAgICAgICAgIHZhciBtYXBfYTMyZDY3NWU3ZjkwNDlhOWExNDM5OTgyOTAzMTRhOTggPSBMLm1hcCgnbWFwX2EzMmQ2NzVlN2Y5MDQ5YTlhMTQzOTk4MjkwMzE0YTk4JywgewogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY2VudGVyOlszOS4wODM2LC03Ny4xNDgzXSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHpvb206IDExLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbWF4Qm91bmRzOiBib3VuZHMsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBsYXllcnM6IFtdLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY3JzOiBMLkNSUy5FUFNHMzg1NwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIH0pOwogICAgICAgICAgICAKICAgICAgICAKICAgICAgICAKICAgICAgICAgICAgCiAgICAgICAgICAgIHZhciB0aWxlX2xheWVyXzMxYjEzMTJiNmVhODQ0NWE4MmJiMGVlMDBjOTc2NjBjID0gTC50aWxlTGF5ZXIoCiAgICAgICAgICAgICAgICAnaHR0cHM6Ly97c30udGlsZS5vcGVuc3RyZWV0bWFwLm9yZy97en0ve3h9L3t5fS5wbmcnLAogICAgICAgICAgICAgICAgewogICAgICAgICAgICAgICAgICAgIG1heFpvb206IDE4LAogICAgICAgICAgICAgICAgICAgIG1pblpvb206IDEsCiAgICAgICAgICAgICAgICAgICAgYXR0cmlidXRpb246ICdEYXRhIGJ5IDxhIGhyZWY9Imh0dHA6Ly9vcGVuc3RyZWV0bWFwLm9yZyI+T3BlblN0cmVldE1hcDwvYT4sIHVuZGVyIDxhIGhyZWY9Imh0dHA6Ly93d3cub3BlbnN0cmVldG1hcC5vcmcvY29weXJpZ2h0Ij5PRGJMPC9hPi4nLAogICAgICAgICAgICAgICAgICAgIGRldGVjdFJldGluYTogZmFsc2UKICAgICAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hMzJkNjc1ZTdmOTA0OWE5YTE0Mzk5ODI5MDMxNGE5OCk7CgogICAgICAgIAogICAgICAgIAogICAgICAgICAgICAKICAgICAgICAgICAgdmFyIGhlYXRfbWFwX2I3OGIzMzNmZWMzMTQxY2U4ZDNiMGY1NzJiZTM0YjUwID0gTC5oZWF0TGF5ZXIoCiAgICAgICAgICAgICAgICBbWzM5LjAzMjIzLCAtNzcuMTA1OTI1XSwgWzAuMCwgMC4wXSwgWzM5LjA0MTM0NSwgLTc3LjA1MzkxNjY2NjY2NjddLCBbMzkuMDQwNTY1LCAtNzcuMDU2MjgxNjY2NjY2N10sIFszOS4wMTM0NDgzMzMzMzMzLCAtNzcuMTA2MTMzMzMzMzMzM10sIFszOS4wMTM1NjgzMzMzMzMzLCAtNzcuMTA2MjM4MzMzMzMzM10sIFszOS4wMTM2NCwgLTc3LjEwNjIyNjY2NjY2NjddLCBbMzkuMDIzMDA1LCAtNzcuMTAzMjAxNjY2NjY2N10sIFszOS4yNDA2NTMzMzMzMzMzLCAtNzcuMjM1MDIxNjY2NjY2N10sIFszOS4yNDA2NTMzMzMzMzMzLCAtNzcuMjM1MDIxNjY2NjY2N10sIFszOS4yNDA2NTMzMzMzMzMzLCAtNzcuMjM1MDIxNjY2NjY2N10sIFszOS4wMDE3OCwgLTc3LjAzODg1MTY2NjY2NjddLCBbMC4wLCAwLjBdLCBbMzguOTk4MTYsIC03Ni45OTUwODgzMzMzMzMzXSwgWzAuMCwgMC4wXSwgWzAuMCwgMC4wXSwgWzAuMCwgMC4wXSwgWzAuMCwgMC4wXSwgWzAuMCwgMC4wXSwgWzAuMCwgMC4wXSwgWzM5LjA0MTM0NSwgLTc3LjA1MzkxNjY2NjY2NjddLCBbMzkuMDQ3MDM4MzMzMzMzMywgLTc3LjA1MzE5XSwgWzM5LjA0NzAzODMzMzMzMzMsIC03Ny4wNTMxOV0sIFszOS4wNTcxODE2NjY2NjY3LCAtNzcuMDQ5NzE4MzMzMzMzM10sIFszOS4wNTcxODE2NjY2NjY3LCAtNzcuMDQ5NzE4MzMzMzMzM10sIFszOS4wNDA1NjUsIC03Ny4wNTYyODE2NjY2NjY3XSwgWzM4Ljk5OTE5MTY2NjY2NjcsIC03Ny4wMzM4MTY2NjY2NjY3XSwgWzM4Ljk5NzI3LCAtNzcuMDI3NDc2NjY2NjY2N10sIFszOS4wMTkxOTY2NjY2NjY3LCAtNzcuMDE0MjExNjY2NjY2N10sIFszOS4wOTYzMTE2NjY2NjY3LCAtNzcuMjA2NDRdLCBbMzkuMTAyMDA1LCAtNzcuMjEwODgzMzMzMzMzM10sIFszOS4xMDIwMDUsIC03Ny4yMTA4ODMzMzMzMzMzXSwgWzM5LjEwMjAwNSwgLTc3LjIxMDg4MzMzMzMzMzNdLCBbMzkuMDI5Mzk1LCAtNzcuMDYzODE1XSwgWzM5LjAyOTM5NSwgLTc3LjA2MzgxNV0sIFszOS4wMjkzOTUsIC03Ny4wNjM4MTVdLCBbMzkuMDI5Mzk1LCAtNzcuMDYzODE1XSwgWzM5LjAzMTIwNSwgLTc3LjA3MjQyODMzMzMzMzNdLCBbMzkuMDMxMzA1LCAtNzcuMDcyNTI1XSwgWzM4Ljk5MDE2MTY2NjY2NjcsIC03Ny4wMDE1NDgzMzMzMzMzXSwgWzM5LjExMzgyNSwgLTc3LjE5MzY1XSwgWzM5LjAwMzUwNjY2NjY2NjcsIC03Ny4wMzU2MDE2NjY2NjY3XSwgWzM5LjE2NDE3MzMzMzMzMzMsIC03Ny4xNTcyM10sIFszOS4xOTA2NjE2NjY2NjY3LCAtNzcuMjQyMDYzMzMzMzMzM10sIFszOS4xMTg4NCwgLTc3LjAyMTc1MTY2NjY2NjddLCBbMzkuMTI3NzA2NjY2NjY2NywgLTc3LjE2NzA4MzMzMzMzMzNdLCBbMzkuMTUxNzEsIC03Ny4yNzgyNF0sIFszOS4wNjUwMDUsIC03Ny4yNjU5NTVdLCBbMzkuMTAxODI1LCAtNzcuMTQxODRdLCBbMzguOTg1NjUzMzMzMzMzMywgLTc3LjIyNjU0NV0sIFszOS4wNzQ3OCwgLTc2Ljk5ODJdLCBbMzkuMDczOTMxNjY2NjY2NywgLTc3LjAwMDk5MzMzMzMzMzNdLCBbMzkuMTk4OTI4MzMzMzMzMywgLTc3LjI0NDgxMTY2NjY2NjddLCBbMzkuMjc4MTk1LCAtNzcuMzIyNjk2NjY2NjY2N10sIFszOS4yNzgxOTUsIC03Ny4zMjI2OTY2NjY2NjY3XSwgWzM5LjIyODQ4MzMzMzMzMzMsIC03Ny4yODI2MV0sIFszOS4xNzc4NzY2NjY2NjY3LCAtNzcuMjcwNjQ4MzMzMzMzM10sIFszOS4xNzgyNzY2NjY2NjY3LCAtNzcuMjcyNDcxNjY2NjY2N10sIFszOS4xNzgyMzUsIC03Ny4yNzIzMTMzMzMzMzMzXSwgWzM5LjE3NzQzNjY2NjY2NjcsIC03Ny4yNzEzMjY2NjY2NjY3XSwgWzAuMCwgMC4wXSwgWzM5LjE3MTE5ODMzMzMzMzMsIC03Ny4yNzc4MjY2NjY2NjY3XSwgWzM5LjE3MjA0NjY2NjY2NjcsIC03Ny4yNzY2ODgzMzMzMzMzXSwgWzM5LjE2ODg5NjY2NjY2NjcsIC03Ny4yNzk1N10sIFszOS4xNzEwNiwgLTc3LjI3NzkyNV0sIFszOS4xMTEyOTgzMzMzMzMzLCAtNzcuMTYxMjYxNjY2NjY2N10sIFszOS4wNjk1NTMzMzMzMzMzLCAtNzcuMTM4NDQzMzMzMzMzM10sIFszOS4wNDExMDgzMzMzMzMzLCAtNzcuMTYyMjYzMzMzMzMzM10sIFszOS4wOTU1NywgLTc3LjA0NDE5ODMzMzMzMzNdLCBbMzkuMjg3MTIsIC03Ny4yMDgwMV0sIFszOS4yNzU5NjUsIC03Ny4yMTQxNTVdLCBbMzkuMjc2NzI1LCAtNzcuMjEzNDY4MzMzMzMzM10sIFswLjAsIDAuMF0sIFszOC45OTg4MTUsIC03Ni45OTUxMDgzMzMzMzMzXSwgWzM4Ljk5Nzc0ODMzMzMzMzMsIC03Ni45OTQzM10sIFszOC45OTc3NDgzMzMzMzMzLCAtNzYuOTk0MzNdLCBbMzguOTk3NTY1LCAtNzYuOTk0MDkzMzMzMzMzM10sIFszOC45OTQ3OCwgLTc3LjAwNzgwMTY2NjY2NjddLCBbMzguOTk0NzgsIC03Ny4wMDc4MDE2NjY2NjY3XSwgWzM4Ljk5NDc4LCAtNzcuMDA3ODAxNjY2NjY2N10sIFszOC45OTQ3OCwgLTc3LjAwNzgwMTY2NjY2NjddLCBbMzkuMjIwMzEzMzMzMzMzMywgLTc3LjA2MDE2NjY2NjY2NjddLCBbMzkuMDc4NzMzMzMzMzMzMywgLTc3LjAwMTUyMzMzMzMzMzNdLCBbMzkuMDgxNjM4MzMzMzMzMywgLTc3LjAwMTEwMTY2NjY2NjddLCBbMzkuMTA1NzI4MzMzMzMzMywgLTc3LjA4NTIwMzMzMzMzMzNdLCBbMzkuMTA1NjQzMzMzMzMzMywgLTc3LjA4NTM4XSwgWzM5LjEwNzQ0LCAtNzcuMDgyNjNdLCBbMzkuMTA1MzcsIC03Ny4wODU2NzY2NjY2NjY3XSwgWzM5LjEwNTk1MTY2NjY2NjcsIC03Ny4wODUwMDE2NjY2NjY3XSwgWzM5LjEwNTA2MTY2NjY2NjcsIC03Ny4wODU4OF0sIFszOS4xMDQ0MDUsIC03Ny4wODY5NjY2NjY2NjY3XSwgWzM5LjI3ODE5NSwgLTc3LjMyMjY5NjY2NjY2NjddLCBbMzkuMjc4MTk1LCAtNzcuMzIyNjk2NjY2NjY2N10sIFszOS4yNzgxOTUsIC03Ny4zMjI2OTY2NjY2NjY3XSwgWzM5LjA4OTA2NjY2NjY2NjcsIC03Ny4yMTM5NV0sIFszOS4yMjg0ODMzMzMzMzMzLCAtNzcuMjgyNjFdLCBbMzkuMjA1ODA1LCAtNzcuMjQzMTkzMzMzMzMzM10sIFszOS4wMDk5NDY2NjY2NjY3LCAtNzcuMDk3MzMzMzMzMzMzM10sIFszOS4wMDk5NDY2NjY2NjY3LCAtNzcuMDk3MzMzMzMzMzMzM10sIFszOS4wMzAzMDgzMzMzMzMzLCAtNzcuMTA1MDldLCBbMzkuMDIwMjksIC03Ny4wNzcxNjgzMzMzMzMzXSwgWzM5LjAyMDI5LCAtNzcuMDc3MTY4MzMzMzMzM10sIFszOS4wNDc0MjE2NjY2NjY3LCAtNzcuMDc1ODkzMzMzMzMzM10sIFswLjAsIDAuMF0sIFszOS4wNDExMjMzMzMzMzMzLCAtNzcuMTU5MjgzMzMzMzMzM10sIFszOS4wMDc0OTE2NjY2NjY3LCAtNzcuMTEyNTkzMzMzMzMzM10sIFszOC45OTAxNjE2NjY2NjY3LCAtNzcuMDAxNTQ4MzMzMzMzM10sIFszOS4xMTM4MjUsIC03Ny4xOTM2NV0sIFszOS4xMTM4MjUsIC03Ny4xOTM2NV0sIFszOS4xNzE5OCwgLTc3LjI3NzAzODMzMzMzMzNdLCBbMzkuMDAzNTA2NjY2NjY2NywgLTc3LjAzNTYwMTY2NjY2NjddLCBbMzkuMDAzNTA2NjY2NjY2NywgLTc3LjAzNTYwMTY2NjY2NjddLCBbMzkuMDQ1NDI1LCAtNzYuOTkwNzM2NjY2NjY2N10sIFszOS4wNDU0MjUsIC03Ni45OTA3MzY2NjY2NjY3XSwgWzM5LjA0NTQyNSwgLTc2Ljk5MDczNjY2NjY2NjddLCBbMzkuMDQ1NDI1LCAtNzYuOTkwNzM2NjY2NjY2N10sIFszOS4wNDEzNDUsIC03Ny4wNTM5MTY2NjY2NjY3XSwgWzM5LjA0MTM0NSwgLTc3LjA1MzkxNjY2NjY2NjddLCBbMzkuMDQ3MDM4MzMzMzMzMywgLTc3LjA1MzE5XSwgWzM5LjA0NzAzODMzMzMzMzMsIC03Ny4wNTMxOV0sIFszOS4wNDcwMzgzMzMzMzMzLCAtNzcuMDUzMTldLCBbMzkuMDQ3Mjc1LCAtNzcuMDU0MDU1XSwgWzM5LjA2NjkyMTY2NjY2NjcsIC03Ny4wMTg4N10sIFszOS4wNjY5MjE2NjY2NjY3LCAtNzcuMDE4ODddLCBbMzkuMDY2OTIxNjY2NjY2NywgLTc3LjAxODg3XSwgWzM4Ljk4NzI4MzMzMzMzMzMsIC03Ny4wMjM0NjY2NjY2NjY3XSwgWzM4Ljk4Njk2MzMzMzMzMzMsIC03Ny4xMDc2MjMzMzMzMzMzXSwgWzM5LjA4Mzg2NSwgLTc3LjAwMDM0MzMzMzMzMzNdLCBbMzkuMTc2MjMsIC03Ny4yNzExMTE2NjY2NjY3XSwgWzM5LjEyMzkzMTY2NjY2NjcsIC03Ny4xNzk0MDY2NjY2NjY3XSwgWzM5LjEyMzkzMTY2NjY2NjcsIC03Ny4xNzk0MDY2NjY2NjY3XSwgWzM5LjAwNjY1MzMzMzMzMzMsIC03Ny4xOTEzN10sIFszOS4wMzk5MDgzMzMzMzMzLCAtNzcuMTIyMjhdLCBbMzkuMTE0MTk4MzMzMzMzMywgLTc3LjE2Mzg2ODMzMzMzMzNdLCBbMzkuMDI0Nzk4MzMzMzMzMywgLTc3LjA3NjQwNjY2NjY2NjddLCBbMzkuMDUzMTU4MzMzMzMzMywgLTc3LjExMjk1NjY2NjY2NjddLCBbMzkuMDUzMTU4MzMzMzMzMywgLTc3LjExMjk1NjY2NjY2NjddLCBbMzkuMDU0NDgzMzMzMzMzMywgLTc3LjExNzUxNV0sIFszOC45OTIxMjE2NjY2NjY3LCAtNzcuMDMzNzUzMzMzMzMzM10sIFszOC45ODgwMjY2NjY2NjY3LCAtNzcuMDI3XSwgWzM4Ljk4OTgxNjY2NjY2NjcsIC03Ny4wMjY0MjVdLCBbMzkuMDM4NDI1LCAtNzcuMDU4NTYzMzMzMzMzM10sIFszOS4wMzUwNjMzMzMzMzMzLCAtNzcuMTczOTgxNjY2NjY2N10sIFszOS4wMzUwNjMzMzMzMzMzLCAtNzcuMTczOTgxNjY2NjY2N10sIFszOS4wOTc3NzMzMzMzMzMzLCAtNzcuMjA3ODE1XSwgWzM5LjA5Nzc3MzMzMzMzMzMsIC03Ny4yMDc4MTVdLCBbMzkuMDQ1NDI1LCAtNzYuOTkwNzM2NjY2NjY2N10sIFszOS4wNDU0MjUsIC03Ni45OTA3MzY2NjY2NjY3XSwgWzM5LjA0NTQyNSwgLTc2Ljk5MDczNjY2NjY2NjddLCBbMzkuMDQ1NDI1LCAtNzYuOTkwNzM2NjY2NjY2N10sIFszOS4wNDU0MjUsIC03Ni45OTA3MzY2NjY2NjY3XSwgWzM5LjA0NTQyNSwgLTc2Ljk5MDczNjY2NjY2NjddLCBbMzkuMDgyNjE1LCAtNzYuOTQ3MTAzMzMzMzMzM10sIFszOS4wODI2MTUsIC03Ni45NDcxMDMzMzMzMzMzXSwgWzM5LjA4MjYxNSwgLTc2Ljk0NzEwMzMzMzMzMzNdLCBbMzkuMDc3MzA1LCAtNzYuOTQyNzM4MzMzMzMzM10sIFszOS4wNzczMDUsIC03Ni45NDI3MzgzMzMzMzMzXSwgWzM5LjA4MzM5ODMzMzMzMzMsIC03Ni45MzkyMDVdLCBbMzkuMDgzMzk4MzMzMzMzMywgLTc2LjkzOTIwNV0sIFszOS4xMzkxNzE2NjY2NjY3LCAtNzcuMTkzNzI1XSwgWzM5LjEzOTE3MTY2NjY2NjcsIC03Ny4xOTM3MjVdLCBbMzkuMTkwNDAzMzMzMzMzMywgLTc3LjI2NzEyMTY2NjY2NjddLCBbMzkuMTkwNDAzMzMzMzMzMywgLTc3LjI2NzEyMTY2NjY2NjddLCBbMzkuMTY0MzgxNjY2NjY2NywgLTc3LjQ3Njk3NV0sIFszOS4xNjQzODE2NjY2NjY3LCAtNzcuNDc2OTc1XSwgWzM5LjE2NDM4MTY2NjY2NjcsIC03Ny40NzY5NzVdLCBbMzguOTg2NTgzMzMzMzMzMywgLTc3LjEwNzE3ODMzMzMzMzNdLCBbMzguOTg2Njc1LCAtNzcuMTA3Mjg2NjY2NjY2N10sIFszOC45ODY2NTY2NjY2NjY3LCAtNzcuMTA3MzFdLCBbMzguOTg2NjU2NjY2NjY2NywgLTc3LjEwNzMxXSwgWzM4Ljk4Njk2MzMzMzMzMzMsIC03Ny4xMDc2MjMzMzMzMzMzXSwgWzM4Ljk4NjYxNjY2NjY2NjcsIC03Ny4xMDcyMl0sIFszOC45ODY3MjUsIC03Ny4xMDczNDY2NjY2NjY3XSwgWzM5LjAxMTY1MTY2NjY2NjcsIC03Ny4xOTkwODMzMzMzMzMzXSwgWzM5LjA1NDA2NjY2NjY2NjcsIC03Ny4xMzk2ODVdLCBbMzkuMTU0OTI2NjY2NjY2NywgLTc3LjIxNDQ3MzMzMzMzMzNdLCBbMzkuMTU0OTI2NjY2NjY2NywgLTc3LjIxNDQ3MzMzMzMzMzNdLCBbMzkuMTc2NDMzMzMzMzMzMywgLTc3LjM1ODk0XSwgWzM5LjE3NjQzMzMzMzMzMzMsIC03Ny4zNTg5NF0sIFszOS4xMTg3MDMzMzMzMzMzLCAtNzcuMDc3NDA2NjY2NjY2N10sIFszOS4wMTMyMDUsIC03Ny4xMDY4MjVdLCBbMzkuMDEzMTYxNjY2NjY2NywgLTc3LjEwNjg3XSwgWzM5LjA3OTkwNSwgLTc3LjAwMTQwMzMzMzMzMzNdLCBbMzkuMTQ4MTgxNjY2NjY2NywgLTc3LjI3ODY3ODMzMzMzMzNdLCBbMzkuMTQ3NjI4MzMzMzMzMywgLTc3LjI3NTk4MTY2NjY2NjddLCBbMzkuMTcwNDg1LCAtNzcuMjc4Mzk1XSwgWzM5LjE3MjkzNSwgLTc3LjI2NDUxNjY2NjY2NjddLCBbMzkuMTcxODA2NjY2NjY2NywgLTc3LjI2MzQ5MzMzMzMzMzNdLCBbMzkuMDk2NjksIC03Ni45Mzc1NTE2NjY2NjY3XSwgWzM5LjA3NTU5LCAtNzYuOTk0NTI4MzMzMzMzM10sIFszOS4xNTc2MjE2NjY2NjY3LCAtNzcuMDQ4OTA4MzMzMzMzM10sIFszOS4wMTIwMjE2NjY2NjY3LCAtNzcuMTk5NzA4MzMzMzMzM10sIFszOS4wMTM0MzY2NjY2NjY3LCAtNzcuMjAzMTIzMzMzMzMzM10sIFszOS4wNzQ5MjE2NjY2NjY3LCAtNzcuMDkzOTJdLCBbMzkuMTg2Nzc1LCAtNzcuNDA1MzUzMzMzMzMzM10sIFszOS4xODY3NzUsIC03Ny40MDUzNTMzMzMzMzMzXSwgWzM5LjA5NjEwNjY2NjY2NjcsIC03Ny4xNzcxMzMzMzMzMzMzXSwgWzM5LjA5MTU5ODMzMzMzMzMsIC03Ny4xNzUyODE2NjY2NjY3XSwgWzM5LjA2NDQxNjY2NjY2NjcsIC03Ny4xNTYyNTY2NjY2NjY3XSwgWzM5LjA0NTUzNSwgLTc3LjE0OTIyMTY2NjY2NjddLCBbMzkuMDU2MzMsIC03Ny4xMTcyMTgzMzMzMzMzXSwgWzM5LjAzNDcwMTY2NjY2NjcsIC03Ny4wNzA4MTMzMzMzMzMzXSwgWzM5LjAzNDU5NjY2NjY2NjcsIC03Ny4wNzE2MTgzMzMzMzMzXSwgWzM5LjA4MzM5ODMzMzMzMzMsIC03Ni45MzkyMDVdLCBbMzkuMDY5MzY1LCAtNzcuMTA0OTE4MzMzMzMzM10sIFszOS4wNjkzNjUsIC03Ny4xMDQ5MTgzMzMzMzMzXSwgWzM5LjA2OTM2NSwgLTc3LjEwNDkxODMzMzMzMzNdLCBbMzkuMTI1MDkzMzMzMzMzMywgLTc3LjE4MTg3MTY2NjY2NjddLCBbMzkuMTI1MDkzMzMzMzMzMywgLTc3LjE4MTg3MTY2NjY2NjddLCBbMzkuMTI1MDkzMzMzMzMzMywgLTc3LjE4MTg3MTY2NjY2NjddLCBbMzkuMTI1MDkzMzMzMzMzMywgLTc3LjE4MTg3MTY2NjY2NjddLCBbMzkuMTIyODM1LCAtNzcuMTc4MTI4MzMzMzMzM10sIFszOS4xMzkxNzE2NjY2NjY3LCAtNzcuMTkzNzI1XSwgWzM5LjAxMjE2NSwgLTc3LjE5OTg2ODMzMzMzMzNdLCBbMzkuMDA2NjUzMzMzMzMzMywgLTc3LjE5MTM3XSwgWzM5LjAxNDc3LCAtNzcuMjA0NDI1XSwgWzM5LjExODMxMTY2NjY2NjcsIC03Ny4zMTY1Ml0sIFszOS4wMTQ5NTMzMzMzMzMzLCAtNzcuMDAwNDMzMzMzMzMzM10sIFszOS4wMTQ5NTMzMzMzMzMzLCAtNzcuMDAwNDMzMzMzMzMzM10sIFszOS4wNTM3NjY2NjY2NjY3LCAtNzcuMDUwNzk4MzMzMzMzM10sIFszOC45ODgwMjY2NjY2NjY3LCAtNzcuMDI3XSwgWzM4Ljk4ODAyNjY2NjY2NjcsIC03Ny4wMjddLCBbMzguOTg5ODE2NjY2NjY2NywgLTc3LjAyNjQyNV0sIFszOS4wMDQ3MTUsIC03Ny4wNzc3NjY2NjY2NjY3XSwgWzM5LjAzMDI3LCAtNzcuMDg0ODU1XSwgWzM5LjAzMDI3LCAtNzcuMDg0ODU1XSwgWzM5LjE0NDk1NjY2NjY2NjcsIC03Ny4yMTg5OTgzMzMzMzMzXSwgWzM5LjE0NDk1NjY2NjY2NjcsIC03Ny4yMTg5OTgzMzMzMzMzXSwgWzM5LjAyOTc5MzMzMzMzMzMsIC03Ny4wNzQ1NTgzMzMzMzMzXSwgWzM5LjAyOTc5MzMzMzMzMzMsIC03Ny4wNzQ1NTgzMzMzMzMzXSwgWzM5LjAyOTc5MzMzMzMzMzMsIC03Ny4wNzQ1NTgzMzMzMzMzXSwgWzM5LjAyOTc5MzMzMzMzMzMsIC03Ny4wNzQ1NTgzMzMzMzMzXSwgWzM5LjAyOTgxODMzMzMzMzMsIC03Ny4wNzM3NzMzMzMzMzMzXSwgWzM5LjAyOTgxODMzMzMzMzMsIC03Ny4wNzM3NzMzMzMzMzMzXSwgWzM5LjAyOTgxODMzMzMzMzMsIC03Ny4wNzM3NzMzMzMzMzMzXSwgWzM5LjAyOTczNSwgLTc3LjA3NDY5MzMzMzMzMzNdLCBbMzkuMDI5NzM1LCAtNzcuMDc0NjkzMzMzMzMzM10sIFszOS4wMjk3MzUsIC03Ny4wNzQ2OTMzMzMzMzMzXSwgWzM5LjAyOTI3LCAtNzcuMDY4ODA4MzMzMzMzM10sIFszOS4wMjkyNywgLTc3LjA2ODgwODMzMzMzMzNdLCBbMzkuMDI5NDI2NjY2NjY2NywgLTc3LjA3NDg4MTY2NjY2NjddLCBbMzkuMDI5NDI2NjY2NjY2NywgLTc3LjA3NDg4MTY2NjY2NjddLCBbMzkuMDI5NDI2NjY2NjY2NywgLTc3LjA3NDg4MTY2NjY2NjddLCBbMzkuMTI3MDczMzMzMzMzMywgLTc3LjExODQyXSwgWzM5LjE4MDUzMTY2NjY2NjcsIC03Ny4yNjIzMl0sIFszOS4wMzgxNTMzMzMzMzMzLCAtNzYuOTk4OTU4MzMzMzMzM10sIFszOS4wMzgxNTMzMzMzMzMzLCAtNzYuOTk4OTU4MzMzMzMzM10sIFszOS4wMzgxNTMzMzMzMzMzLCAtNzYuOTk4OTU4MzMzMzMzM10sIFszOC45ODUxMjE2NjY2NjY3LCAtNzcuMDk0MDMzMzMzMzMzM10sIFszOS4wMDc4MiwgLTc2Ljk4NDM4XSwgWzM5LjE1ODA0LCAtNzcuMjI4MDQ2NjY2NjY2N10sIFszOS4wMDc5NzgzMzMzMzMzLCAtNzYuOTg0NDAxNjY2NjY2N10sIFszOS4wMDg2MTUsIC03Ni45ODQxOTY2NjY2NjY3XSwgWzM5LjAwODYxNSwgLTc2Ljk4NDE5NjY2NjY2NjddLCBbMzkuMDA4NzgsIC03Ni45ODQxODY2NjY2NjY3XSwgWzM5LjAwODc3MzMzMzMzMzMsIC03Ni45ODQyNDY2NjY2NjY3XSwgWzM5LjAwOTI5NjY2NjY2NjcsIC03Ni45ODQwMjE2NjY2NjY3XSwgWzM5LjAwOTE2MTY2NjY2NjcsIC03Ni45ODM5ODgzMzMzMzMzXSwgWzM5LjAwODk4NSwgLTc2Ljk4NDE4XSwgWzM5LjAwODM2MzMzMzMzMzMsIC03Ni45ODQyNzY2NjY2NjY3XSwgWzM5LjAwODEwMzMzMzMzMzMsIC03Ni45ODQzODE2NjY2NjY3XSwgWzM5LjAwNzk3NjY2NjY2NjcsIC03Ni45ODQ0MDMzMzMzMzMzXSwgWzM5LjAwNzk3NjY2NjY2NjcsIC03Ni45ODQ0MDMzMzMzMzMzXSwgWzM5LjAwNzgyLCAtNzYuOTg0MzhdLCBbMzguOTk5MDQ1LCAtNzYuOTk3ODZdLCBbMzguOTk5MDQ1LCAtNzYuOTk3ODZdLCBbMzguOTk5MDQ1LCAtNzYuOTk3ODZdLCBbMC4wLCAwLjBdLCBbMC4wLCAwLjBdLCBbMC4wLCAwLjBdLCBbMzkuMTU4MDQsIC03Ny4yMjgwNDY2NjY2NjY3XSwgWzM5LjE3OTU1NjY2NjY2NjcsIC03Ny4yNjQ4ODVdLCBbMzkuMDMzMzI4MzMzMzMzMywgLTc3LjA0ODk3MzMzMzMzMzNdLCBbMzkuMDAxMzE2NjY2NjY2NywgLTc2Ljk5NTcwMTY2NjY2NjddLCBbMzkuMTE5NDYxNjY2NjY2NywgLTc3LjAxODg4XSwgWzM4Ljk4NjQzLCAtNzcuMDk0NzEzMzMzMzMzM10sIFszOS4wMjY4ODY2NjY2NjY3LCAtNzcuMDc3MDg1XSwgWzM4Ljk4MjAyNSwgLTc3LjA5OTg4MzMzMzMzMzNdLCBbMzguOTgyMDI1LCAtNzcuMDk5ODgzMzMzMzMzM10sIFszOS4wNDY3NjE2NjY2NjY3LCAtNzcuMTEyNV0sIFszOS4yNjYzNywgLTc3LjIwMTE5XSwgWzM5LjI2NjM3LCAtNzcuMjAxMTldLCBbMzkuMTcwODE1LCAtNzcuMjc4MDk2NjY2NjY2N10sIFszOS4xNzE2OTgzMzMzMzMzLCAtNzcuMjc3MzRdLCBbMzkuMTc5NTU2NjY2NjY2NywgLTc3LjI2NDg4NV0sIFszOS4wMDI3MiwgLTc3LjAwNjM3NV0sIFszOS4wMDYzNzY2NjY2NjY3LCAtNzcuMDA3MzQ2NjY2NjY2N10sIFszOS4wMDAxOSwgLTc3LjAxMjEzXSwgWzAuMCwgMC4wXSwgWzAuMCwgMC4wXSwgWzAuMCwgMC4wXSwgWzM5LjAyNjcxODMzMzMzMzMsIC03Ny4wNDY4NjE2NjY2NjY3XSwgWzM5LjAzMzMyODMzMzMzMzMsIC03Ny4wNDg5NzMzMzMzMzMzXSwgWzM5LjE5MDU5MTY2NjY2NjcsIC03Ny4yNzMyMTMzMzMzMzMzXSwgWzM5LjE5MDU5MTY2NjY2NjcsIC03Ny4yNzMyMTMzMzMzMzMzXSwgWzM5LjE4MjgxLCAtNzcuMjc0MjNdLCBbMzkuMTgyODEsIC03Ny4yNzQyM10sIFszOS4xODI4MSwgLTc3LjI3NDIzXSwgWzM5LjAwMDczODMzMzMzMzMsIC03Ny4wMTA5N10sIFszOS4wMDQ1MywgLTc3LjAxNDgyNV0sIFszOS4wMDUwNSwgLTc3LjAyMjEzODMzMzMzMzNdLCBbMzkuMTgyMzcsIC03Ny4yNjQyNV0sIFszOS4xODIzNywgLTc3LjI2NDI1XSwgWzM5LjE4MjM3LCAtNzcuMjY0MjVdLCBbMzkuMTgyMzcsIC03Ny4yNjQyNV0sIFszOS4xODIzNywgLTc3LjI2NDI1XSwgWzM5LjE4MjM3LCAtNzcuMjY0MjVdLCBbMzkuMTgyMzcsIC03Ny4yNjQyNV0sIFszOS4wNDU0MjUsIC03Ni45OTA3MzY2NjY2NjY3XSwgWzM5LjA0NTQyNSwgLTc2Ljk5MDczNjY2NjY2NjddLCBbMzkuMDQ1NDI1LCAtNzYuOTkwNzM2NjY2NjY2N10sIFszOS4wNDU0MjUsIC03Ni45OTA3MzY2NjY2NjY3XSwgWzM5LjA0NTQyNSwgLTc2Ljk5MDczNjY2NjY2NjddLCBbMzkuMDQ1NDI1LCAtNzYuOTkwNzM2NjY2NjY2N10sIFszOS4wNDU0MjUsIC03Ni45OTA3MzY2NjY2NjY3XSwgWzM5LjA0NTQyNSwgLTc2Ljk5MDczNjY2NjY2NjddLCBbMC4wLCAwLjBdLCBbMC4wLCAwLjBdLCBbMC4wLCAwLjBdLCBbMzguOTg0MjgxNjY2NjY2NywgLTc3LjA5NDEwNjY2NjY2NjddLCBbMzguOTgxOTM2NjY2NjY2NywgLTc3LjA5OTc1MzMzMzMzMzNdLCBbMzkuMDI2NTY2NjY2NjY2NywgLTc3LjAyMDIyNV0sIFszOS4wMjY1NjY2NjY2NjY3LCAtNzcuMDIwMjI1XSwgWzM5LjAyNjU2NjY2NjY2NjcsIC03Ny4wMjAyMjVdLCBbMzkuMDAxMzE2NjY2NjY2NywgLTc2Ljk5NTcwMTY2NjY2NjddLCBbMzkuMDI0Mzk4MzMzMzMzMywgLTc3LjAxODA3NV0sIFszOS4wMTgwMzMzMzMzMzMzLCAtNzcuMDA3MzY4MzMzMzMzM10sIFszOS4wMDIyMzMzMzMzMzMzLCAtNzcuMDA0ODY1XSwgWzM5LjAwMjUyODMzMzMzMzMsIC03Ny4wMDUxMDgzMzMzMzMzXSwgWzM5LjAwMTg0LCAtNzcuMDA0Nzc4MzMzMzMzM10sIFszOC45OTQzNjE2NjY2NjY3LCAtNzYuOTkyNzQxNjY2NjY2N10sIFszOS4wMzIwNDY2NjY2NjY3LCAtNzcuMDcyNDc1XSwgWzM5LjAzMjAxLCAtNzcuMDcyNDkzMzMzMzMzM10sIFszOS4wMzE5MjY2NjY2NjY3LCAtNzcuMDcyNDQ4MzMzMzMzM10sIFszOS4wMzE4NiwgLTc3LjA3MjI3NV0sIFszOS4wMzE5ODY2NjY2NjY3LCAtNzcuMDcyNDI2NjY2NjY2N10sIFszOS4xMzA3MTMzMzMzMzMzLCAtNzcuMTg3ODk2NjY2NjY2N10sIFszOS4xMzA3MTMzMzMzMzMzLCAtNzcuMTg3ODk2NjY2NjY2N10sIFszOS4xMzI4MTgzMzMzMzMzLCAtNzcuMTg5NTNdLCBbMzkuMTMyODE4MzMzMzMzMywgLTc3LjE4OTUzXSwgWzM5LjEzMDczLCAtNzcuMTg3OTddLCBbMzkuMTMwNzMsIC03Ny4xODc5N10sIFszOS4xMzA3MywgLTc3LjE4Nzk3XSwgWzM5LjEzMzg0ODMzMzMzMzMsIC03Ny4yNDQ3NjVdLCBbMzkuMTM2MTY4MzMzMzMzMywgLTc3LjI0Mjk5NV0sIFszOS4xMzIyNjgzMzMzMzMzLCAtNzcuMTY0ODkzMzMzMzMzM10sIFszOS4xMzIyNjgzMzMzMzMzLCAtNzcuMTY0ODkzMzMzMzMzM10sIFszOS4xODk3ODMzMzMzMzMzLCAtNzcuMjQ5NzAzMzMzMzMzM10sIFszOS4wMTIzODUsIC03Ny4wODA1OTVdLCBbMzkuMTI0NjcxNjY2NjY2NywgLTc3LjIwMDQ1MzMzMzMzMzNdLCBbMzkuMTAyMTg1LCAtNzcuMTc3OTUzMzMzMzMzM10sIFszOS4xMDIxODUsIC03Ny4xNzc5NTMzMzMzMzMzXSwgWzM5LjEwMjE4NSwgLTc3LjE3Nzk1MzMzMzMzMzNdLCBbMzguOTg1NzUxNjY2NjY2NywgLTc3LjA3NzAzNV0sIFszOC45ODU3NTE2NjY2NjY3LCAtNzcuMDc3MDM1XSwgWzM4Ljk4NjY3NSwgLTc3LjA3NDc4MTY2NjY2NjddLCBbMzguOTg2Njc1LCAtNzcuMDc0NzgxNjY2NjY2N10sIFszOC45ODY2NzUsIC03Ny4wNzQ3ODE2NjY2NjY3XSwgWzM4Ljk4NTc3ODMzMzMzMzMsIC03Ny4wNzY3MzE2NjY2NjY3XSwgWzM4Ljk4NTc3ODMzMzMzMzMsIC03Ny4wNzY3MzE2NjY2NjY3XSwgWzM5LjEzNDc4MzMzMzMzMzMsIC03Ny40MDA1NzY2NjY2NjY3XSwgWzM5LjEzNzc1MzMzMzMzMzMsIC03Ny4zOTc5NzMzMzMzMzMzXSwgWzM5LjE1NjY3NjY2NjY2NjcsIC03Ny40NDczNDVdLCBbMzkuMTcxNTI4MzMzMzMzMywgLTc3LjQ2NjM2NV0sIFszOS4xNzEzNzY2NjY2NjY3LCAtNzcuNDY1NTAxNjY2NjY2N10sIFszOS4xNzEzNzY2NjY2NjY3LCAtNzcuNDY1NTAxNjY2NjY2N10sIFszOS4xMjM5NzE2NjY2NjY3LCAtNzcuMTc1NTY1XSwgWzM5LjEyNDAxNSwgLTc3LjE3NTg0NjY2NjY2NjddLCBbMzkuMTI0MDE1LCAtNzcuMTc1ODQ2NjY2NjY2N10sIFszOS4xMjQwMTUsIC03Ny4xNzU4NDY2NjY2NjY3XSwgWzM5LjE5NDcyODMzMzMzMzMsIC03Ny4xNTQxNF0sIFszOS4xOTQ4MjY2NjY2NjY3LCAtNzcuMTU0MTI2NjY2NjY2N10sIFszOS4xOTQ3OTE2NjY2NjY3LCAtNzcuMTU0MDkzMzMzMzMzM10sIFszOS4xOTQ3OTE2NjY2NjY3LCAtNzcuMTU0MDkzMzMzMzMzM10sIFszOS4xNjM3NDUsIC03Ny4yMTQ2OTMzMzMzMzMzXSwgWzM5LjE2Mzc0NSwgLTc3LjIxNDY5MzMzMzMzMzNdLCBbMzkuMTYzNzQ1LCAtNzcuMjE0NjkzMzMzMzMzM10sIFszOS4xNjM3NDUsIC03Ny4yMTQ2OTMzMzMzMzMzXSwgWzM5LjExMjgxNjY2NjY2NjcsIC03Ny4xNjIxNjVdLCBbMzkuMTU5NTE4MzMzMzMzMywgLTc3LjI3Njg0NjY2NjY2NjddLCBbMzkuMTU5NTE4MzMzMzMzMywgLTc3LjI3Njg0NjY2NjY2NjddLCBbMzkuMTIwNzgsIC03Ny4wMTU3NjE2NjY2NjY3XSwgWzM5LjE5ODA0NSwgLTc3LjI2MzA2MzMzMzMzMzNdLCBbMzkuMTc4NDk4MzMzMzMzMywgLTc3LjI0MDc1XSwgWzM5LjE4MDQ1LCAtNzcuMjYyNDAzMzMzMzMzM10sIFszOS4xODA0NSwgLTc3LjI2MjQwMzMzMzMzMzNdLCBbMzkuMDIyNjQ4MzMzMzMzMywgLTc3LjA3MTc2NjY2NjY2NjddLCBbMzkuMDMyOTgxNjY2NjY2NywgLTc3LjA3MjczNjY2NjY2NjddLCBbMzkuMDMzMDIsIC03Ny4wNzI1MzMzMzMzMzMzXSwgWzM5LjE1MzUzLCAtNzcuMjEyNzIxNjY2NjY2N10sIFszOS4xMzQ2MjE2NjY2NjY3LCAtNzcuNDAxMzU1XSwgWzM5LjEzNDYyMTY2NjY2NjcsIC03Ny40MDEzNTVdLCBbMzkuMTM0NjIxNjY2NjY2NywgLTc3LjQwMTM1NV0sIFszOS4xMzQ2MjE2NjY2NjY3LCAtNzcuNDAxMzU1XSwgWzM5LjA4NDYyLCAtNzcuMDc5OTQ2NjY2NjY2N10sIFszOC45OTE5NzUsIC03Ny4wOTU3MzgzMzMzMzMzXSwgWzM4Ljk4NjY3MTY2NjY2NjcsIC03Ny4wOTQ3MzVdLCBbMzguOTg2NDMsIC03Ny4wOTQ3MTMzMzMzMzMzXSwgWzM4Ljk4NjQzLCAtNzcuMDk0NzEzMzMzMzMzM10sIFszOC45OTA4OSwgLTc3LjA5NTUxNV0sIFszOS4wMTAwMTE2NjY2NjY3LCAtNzcuMDkxNzU2NjY2NjY2N10sIFszOS4wMTIzODUsIC03Ny4wODA1OTVdLCBbMC4wLCAwLjBdLCBbMzkuMTM2MTgxNjY2NjY2NywgLTc3LjE1MjAzXSwgWzM5LjE4MzI0ODMzMzMzMzMsIC03Ny4yNzAzNTVdLCBbMzkuMDk4NTg4MzMzMzMzMywgLTc2LjkzNjQ3NV0sIFswLjAsIDAuMF0sIFswLjAsIDAuMF0sIFswLjAsIDAuMF0sIFswLjAsIDAuMF0sIFswLjAsIDAuMF0sIFswLjAsIDAuMF0sIFswLjAsIDAuMF0sIFswLjAsIDAuMF0sIFswLjAsIDAuMF0sIFszOS4xMTQ3NjMzMzMzMzMzLCAtNzcuMTk3MzkzMzMzMzMzM10sIFszOS4xMTQzMjY2NjY2NjY3LCAtNzcuMTk1NDMzMzMzMzMzM10sIFszOS4xNTM1MywgLTc3LjIxMjcyMTY2NjY2NjddLCBbMzkuMTM0NzczMzMzMzMzMywgLTc3LjQwMDYzMzMzMzMzMzNdLCBbMzkuMTM0NzczMzMzMzMzMywgLTc3LjQwMDYzMzMzMzMzMzNdLCBbMzkuMTM0NzczMzMzMzMzMywgLTc3LjQwMDYzMzMzMzMzMzNdLCBbMzkuMTM0Nzc2NjY2NjY2NywgLTc3LjQwMDU4NV0sIFszOS4xNTY2NzY2NjY2NjY3LCAtNzcuNDQ3MzQ1XSwgWzM5LjE3MTUyMzMzMzMzMzMsIC03Ny40NjYyMl0sIFszOS4xNzE1MTY2NjY2NjY3LCAtNzcuNDY2NzUxNjY2NjY2N10sIFszOS4xNzE1MjgzMzMzMzMzLCAtNzcuNDY2MzY1XSwgWzM5LjA0MzI1MTY2NjY2NjcsIC03Ni45ODMyNzVdLCBbMzkuMDQzMjUxNjY2NjY2NywgLTc2Ljk4MzI3NV0sIFszOS4wNDMyNTE2NjY2NjY3LCAtNzYuOTgzMjc1XSwgWzM5LjA0MzI1MTY2NjY2NjcsIC03Ni45ODMyNzVdLCBbMzkuMDQzMjUxNjY2NjY2NywgLTc2Ljk4MzI3NV0sIFszOS4xMTUyMTgzMzMzMzMzLCAtNzcuMDE2Nl0sIFszOS4xMTUyMTgzMzMzMzMzLCAtNzcuMDE2Nl0sIFszOS4xMTk0NjE2NjY2NjY3LCAtNzcuMDE4ODhdLCBbMzkuMTE5NDYxNjY2NjY2NywgLTc3LjAxODg4XSwgWzM5LjE5ODA0NSwgLTc3LjI2MzA2MzMzMzMzMzNdLCBbMzkuMDg2NzIzMzMzMzMzMywgLTc3LjE2NzQzMTY2NjY2NjddLCBbMzkuMDg1ODE4MzMzMzMzMywgLTc2Ljk0NTQzODMzMzMzMzNdLCBbMzkuMDg1ODE4MzMzMzMzMywgLTc2Ljk0NTQzODMzMzMzMzNdLCBbMzkuMDg1ODE4MzMzMzMzMywgLTc2Ljk0NTQzODMzMzMzMzNdLCBbMzkuMDg1ODE4MzMzMzMzMywgLTc2Ljk0NTQzODMzMzMzMzNdLCBbMzguOTg1MDM1LCAtNzcuMDk0NDMzMzMzMzMzM10sIFszOC45ODQyODE2NjY2NjY3LCAtNzcuMDk0MTA2NjY2NjY2N10sIFszOC45ODE5MzY2NjY2NjY3LCAtNzcuMDk5NzUzMzMzMzMzM10sIFszOC45ODE5MzY2NjY2NjY3LCAtNzcuMDk5NzUzMzMzMzMzM10sIFszOC45ODIwOTE2NjY2NjY3LCAtNzcuMDk5MjYxNjY2NjY2N10sIFszOC45ODE5MzMzMzMzMzMzLCAtNzcuMTAwMTFdLCBbMzkuMDI5ODg4MzMzMzMzMywgLTc3LjA0ODc4NjY2NjY2NjddLCBbMzkuMDI5ODg4MzMzMzMzMywgLTc3LjA0ODc4NjY2NjY2NjddLCBbMzkuMTM0NjIxNjY2NjY2NywgLTc3LjQwMTM1NV0sIFszOS4xMzQ2MjE2NjY2NjY3LCAtNzcuNDAxMzU1XSwgWzM5LjEzNDYyMTY2NjY2NjcsIC03Ny40MDEzNTVdLCBbMzkuMTM0NjIxNjY2NjY2NywgLTc3LjQwMTM1NV0sIFszOS4xMzQ2MjE2NjY2NjY3LCAtNzcuNDAxMzU1XSwgWzM5LjA0MzI5NjY2NjY2NjcsIC03Ny4wNTIwNzY2NjY2NjY3XSwgWzM5LjA0MzI5NjY2NjY2NjcsIC03Ny4wNTIwNzY2NjY2NjY3XSwgWzM4Ljk5NDg1ODMzMzMzMzMsIC03Ny4wMjU3MTY2NjY2NjY3XSwgWzM5LjEzMTg0NSwgLTc3LjIwNjg0NjY2NjY2NjddLCBbMzkuMDk4NTc1LCAtNzcuMTM3NzM4MzMzMzMzM10sIFszOS4wMzEwOTgzMzMzMzMzLCAtNzcuMDc1NzI1XSwgWzM5LjAzMTI0LCAtNzcuMDc1MDI4MzMzMzMzM10sIFszOS4wMjY4ODY2NjY2NjY3LCAtNzcuMDc3MDg1XSwgWzM4Ljk4MjAyNSwgLTc3LjA5OTg4MzMzMzMzMzNdLCBbMzkuMDMyMDA1LCAtNzcuMDcyNDQxNjY2NjY2N10sIFszOS4wMzE5MjY2NjY2NjY3LCAtNzcuMDcyNDQ4MzMzMzMzM10sIFszOS4wMzE5NzY2NjY2NjY3LCAtNzcuMDcyNDM2NjY2NjY2N10sIFszOS4wMzE5NzY2NjY2NjY3LCAtNzcuMDcyNDM2NjY2NjY2N10sIFszOS4wMzE1OTUsIC03Ny4wNzE5OTE2NjY2NjY3XSwgWzM5LjAzMTk3MTY2NjY2NjcsIC03Ny4wNzIzMzMzMzMzMzMzXSwgWzM5LjAzMjA0NjY2NjY2NjcsIC03Ny4wNzI0OTgzMzMzMzMzXSwgWzM5LjAzMTk4NjY2NjY2NjcsIC03Ny4wNzI0MjY2NjY2NjY3XSwgWzM5LjA0ODc5MzMzMzMzMzMsIC03Ny4xMTk4NDY2NjY2NjY3XSwgWzM5LjA4NDY4LCAtNzcuMDgwMDIzMzMzMzMzM10sIFszOS4wODQ2NDY2NjY2NjY3LCAtNzcuMDgwMDAzMzMzMzMzM10sIFszOS4wODQ1NzgzMzMzMzMzLCAtNzcuMDc5OTA1XSwgWzM5LjA4NDY3NSwgLTc3LjA4MDAwMTY2NjY2NjddLCBbMzkuMDg0NjIsIC03Ny4wNzk5NDY2NjY2NjY3XSwgWzM5LjExOTYxLCAtNzcuMDk2OTg1XSwgWzM4Ljk5NDM2MTY2NjY2NjcsIC03Ni45OTI3NDE2NjY2NjY3XSwgWzM5LjEzNjE3ODMzMzMzMzMsIC03Ny4yNDMwOV0sIFszOC45OTY5MTMzMzMzMzMzLCAtNzYuOTkzNjRdLCBbMzguOTk2OTEzMzMzMzMzMywgLTc2Ljk5MzY0XSwgWzM5LjA0MDg4ODMzMzMzMzMsIC03Ny4wNTIxNDVdLCBbMzkuMDQwODg4MzMzMzMzMywgLTc3LjA1MjE0NV0sIFszOS4wNDA4ODgzMzMzMzMzLCAtNzcuMDUyMTQ1XSwgWzM5LjA0MDg4ODMzMzMzMzMsIC03Ny4wNTIxNDVdLCBbMzkuMTEyNjYsIC03Ni45MzI4MTMzMzMzMzMzXSwgWzM5LjE4OTc4MzMzMzMzMzMsIC03Ny4yNDk3MDMzMzMzMzMzXSwgWzM5LjEwNTM5MzMzMzMzMzMsIC03Ny4xOTc1ODE2NjY2NjY3XSwgWzM5LjE2NjQzNjY2NjY2NjcsIC03Ny4yODExODgzMzMzMzMzXSwgWzM5LjExOTAxNjY2NjY2NjcsIC03Ny4xODE5NzE2NjY2NjY3XSwgWzM5LjA5NzExLCAtNzcuMjI5NjEzMzMzMzMzM10sIFszOS4xMTA3NSwgLTc3LjI1MjQzMzMzMzMzMzNdLCBbMzkuMTEwNjM4MzMzMzMzMywgLTc3LjI1Mjc0NV0sIFszOS4xMTU1OTgzMzMzMzMzLCAtNzcuMjUyODg1XSwgWzM5LjExNTU5ODMzMzMzMzMsIC03Ny4yNTI4ODVdLCBbMzkuMTE1NzM2NjY2NjY2NywgLTc3LjI1MjkxMzMzMzMzMzNdLCBbMzkuMTE0NSwgLTc3LjI1MzM4MzMzMzMzMzNdLCBbMzkuMDk2NzY1LCAtNzcuMjY1NzMzMzMzMzMzM10sIFszOS4wOTY3NjUsIC03Ny4yNjU3MzMzMzMzMzMzXSwgWzM5LjA5NDQ2NjY2NjY2NjcsIC03Ny4yNjU3NDY2NjY2NjY3XSwgWzM5LjA5NDQ2NjY2NjY2NjcsIC03Ny4yNjU3NDY2NjY2NjY3XSwgWzM5LjE0NTgxMTY2NjY2NjcsIC03Ny4xOTIyNDVdLCBbMzkuMTMwMTgzMzMzMzMzMywgLTc3LjEyMzM3NjY2NjY2NjddLCBbMzkuMTMwMTgzMzMzMzMzMywgLTc3LjEyMzM3NjY2NjY2NjddLCBbMzkuMTIyMzUsIC03Ny4wNzM5NTMzMzMzMzMzXSwgWzM5LjE1ODQ3ODMzMzMzMzMsIC03Ny4wNjQ3NDgzMzMzMzMzXSwgWzM5LjA3NDc2ODMzMzMzMzMsIC03Ni45OTgyMzY2NjY2NjY3XSwgWzM5LjA3NDc2ODMzMzMzMzMsIC03Ni45OTgyMzY2NjY2NjY3XSwgWzM5LjA3NDc2ODMzMzMzMzMsIC03Ni45OTgyMzY2NjY2NjY3XSwgWzM5LjA3NTAyODMzMzMzMzMsIC03Ni45OTkwMl0sIFszOS4wNzg0NDE2NjY2NjY3LCAtNzcuMDcyMjczMzMzMzMzM10sIFszOS4wMTg5NiwgLTc3LjA1MDFdLCBbMzkuMTI0MTU2NjY2NjY2NywgLTc3LjIwMDAzMzMzMzMzMzNdLCBbMzkuMTgzMDk1LCAtNzcuMjU3NTg1XSwgWzM5LjE2OTMwMTY2NjY2NjcsIC03Ny4yNzkzMTE2NjY2NjY3XSwgWzM5LjE2OTMwMTY2NjY2NjcsIC03Ny4yNzkzMTE2NjY2NjY3XSwgWzM5LjE3MDA0MTY2NjY2NjcsIC03Ny4yNzg4NTMzMzMzMzMzXSwgWzM5LjE2ODg5ODMzMzMzMzMsIC03Ny4yNzk2MDgzMzMzMzMzXSwgWzM5LjE3MDU1LCAtNzcuMjc4Mzg1XSwgWzM5LjE3MDI2NSwgLTc3LjI3ODYwODMzMzMzMzNdLCBbMzkuMTAxODcsIC03Ny4yMTkzMTE2NjY2NjY3XSwgWzM5LjEwMTg3LCAtNzcuMjE5MzExNjY2NjY2N10sIFswLjAsIDAuMF0sIFswLjAsIDAuMF0sIFswLjAsIDAuMF0sIFszOS4wNzQ2MDgzMzMzMzMzLCAtNzYuOTk4MDYxNjY2NjY2N10sIFszOS4wNzQ2MDgzMzMzMzMzLCAtNzYuOTk4MjE4MzMzMzMzM10sIFszOS4wNzQ5MSwgLTc2Ljk5ODM1NV0sIFszOS4wNDE2MywgLTc3LjE0NzUzNV0sIFszOS4wNDQ0MTY2NjY2NjY3LCAtNzcuMTExOTExNjY2NjY2N10sIFszOS4xMTcwMzMzMzMzMzMzLCAtNzcuMjUxOTAzMzMzMzMzM10sIFszOS4xMTc1NzE2NjY2NjY3LCAtNzcuMjUxMjM2NjY2NjY2N10sIFszOS4xMjA5NzgzMzMzMzMzLCAtNzcuMTU4MTQxNjY2NjY2N10sIFszOS4xMjA5NzgzMzMzMzMzLCAtNzcuMTU4MTQxNjY2NjY2N10sIFszOS4xMjA5NzgzMzMzMzMzLCAtNzcuMTU4MTQxNjY2NjY2N10sIFszOS4xMjA5NzgzMzMzMzMzLCAtNzcuMTU4MTQxNjY2NjY2N10sIFszOS4xMTcwMTE2NjY2NjY3LCAtNzcuMjUyNzFdLCBbMzkuMTE4Mzg2NjY2NjY2NywgLTc3LjI1MTE2NV0sIFszOS4wNTkwMzMzMzMzMzMzLCAtNzcuMTYxMzY2NjY2NjY2N10sIFszOS4wNTkwMzMzMzMzMzMzLCAtNzcuMTYxMzY2NjY2NjY2N10sIFszOS4wNTkwMzMzMzMzMzMzLCAtNzcuMTYxMzY2NjY2NjY2N10sIFszOS4wNTkwMzMzMzMzMzMzLCAtNzcuMTYxMzY2NjY2NjY2N10sIFszOS4xODk5MTUsIC03Ny4yNDE5MTVdLCBbMzkuMDg3NDI2NjY2NjY2NywgLTc3LjE3Mzk1MTY2NjY2NjddLCBbMzkuMTE1Njc4MzMzMzMzMywgLTc3LjI1Mjk2MTY2NjY2NjddLCBbMzkuMTA4NiwgLTc3LjE4NTQyNjY2NjY2NjddLCBbMzkuMTEzNTYxNjY2NjY2NywgLTc3LjI0MjgyMzMzMzMzMzNdLCBbMzkuMDc0NzY4MzMzMzMzMywgLTc2Ljk5ODIzNjY2NjY2NjddLCBbMzkuMDc0NzY4MzMzMzMzMywgLTc2Ljk5ODIzNjY2NjY2NjddLCBbMzkuMDc0NzY4MzMzMzMzMywgLTc2Ljk5ODIzNjY2NjY2NjddLCBbMzkuMDc0NzY4MzMzMzMzMywgLTc2Ljk5ODIzNjY2NjY2NjddLCBbMzkuMDMyMjYxNjY2NjY2NywgLTc3LjA3NDI3MTY2NjY2NjddLCBbMzkuMDMzNTkxNjY2NjY2NywgLTc3LjA3Mjg4NV0sIFszOS4wMzM5NCwgLTc3LjA3MjIzNjY2NjY2NjddLCBbMzkuMDMzNjk2NjY2NjY2NywgLTc3LjA3MzAzMzMzMzMzMzNdLCBbMzkuMDMzNDkzMzMzMzMzMywgLTc3LjA3MzM1NV0sIFszOS4wMzQ1OTE2NjY2NjY3LCAtNzcuMDcxMzQ1XSwgWzM5LjAzMzkyNjY2NjY2NjcsIC03Ny4wNzIyNjE2NjY2NjY3XSwgWzM5LjEwMTg3LCAtNzcuMjE5MzExNjY2NjY2N10sIFszOS4xMDE4NywgLTc3LjIxOTMxMTY2NjY2NjddLCBbMzkuMTQ1ODExNjY2NjY2NywgLTc3LjE5MjI0NV0sIFszOS4xNDU4MTE2NjY2NjY3LCAtNzcuMTkyMjQ1XSwgWzM5LjE0NTgxMTY2NjY2NjcsIC03Ny4xOTIyNDVdLCBbMzkuMTE1ODEsIC03Ny4xODU5NTgzMzMzMzMzXSwgWzM5LjExNTgxLCAtNzcuMTg1OTU4MzMzMzMzM10sIFszOS4xMTU4MSwgLTc3LjE4NTk1ODMzMzMzMzNdLCBbMzkuMTE1ODEsIC03Ny4xODU5NTgzMzMzMzMzXSwgWzM5LjEyMTMxNjY2NjY2NjcsIC03Ny4xNjE2MTVdLCBbMzkuMTAxMTE1LCAtNzcuMDQ1MjQ2NjY2NjY2N10sIFszOS4wNzQ2MDgzMzMzMzMzLCAtNzYuOTk4MDYxNjY2NjY2N10sIFszOS4wMjQzNTE2NjY2NjY3LCAtNzcuMDQ1MDMzMzMzMzMzM10sIFszOS4wMjQzNTE2NjY2NjY3LCAtNzcuMDQ1MDMzMzMzMzMzM10sIFszOS4wMjQzNTE2NjY2NjY3LCAtNzcuMDQ1MDMzMzMzMzMzM10sIFszOS4wMjQzNTE2NjY2NjY3LCAtNzcuMDQ1MDMzMzMzMzMzM10sIFszOS4wMjQzNTE2NjY2NjY3LCAtNzcuMDQ1MDMzMzMzMzMzM10sIFszOS4wMjQzNTE2NjY2NjY3LCAtNzcuMDQ1MDMzMzMzMzMzM10sIFszOS4wNTcxMjE2NjY2NjY3LCAtNzcuMDQ5NjU2NjY2NjY2N10sIFszOS4wNTcxMjE2NjY2NjY3LCAtNzcuMDQ5NjU2NjY2NjY2N10sIFszOS4wNTcxMjE2NjY2NjY3LCAtNzcuMDQ5NjU2NjY2NjY2N10sIFszOS4wNTcxMjE2NjY2NjY3LCAtNzcuMDQ5NjU2NjY2NjY2N10sIFszOS4wNTcxMjE2NjY2NjY3LCAtNzcuMDQ5NjU2NjY2NjY2N10sIFszOS4wNTcxMjE2NjY2NjY3LCAtNzcuMDQ5NjU2NjY2NjY2N10sIFszOC45OTk5OSwgLTc3LjAyNTcwNV0sIFszOC45OTk5OSwgLTc3LjAyNTcwNV0sIFszOS4xNTc1NDgzMzMzMzMzLCAtNzcuMTg4NzJdLCBbMzkuMTU3NTg1LCAtNzcuMTg5MTIzMzMzMzMzM10sIFszOS4wNDU0MjUsIC03Ni45OTA3MzY2NjY2NjY3XSwgWzM5LjA0NTQyNSwgLTc2Ljk5MDczNjY2NjY2NjddLCBbMzkuMDQ1NDI1LCAtNzYuOTkwNzM2NjY2NjY2N10sIFszOS4wNDU0MjUsIC03Ni45OTA3MzY2NjY2NjY3XSwgWzM5LjEzOTk3LCAtNzcuMTgyNzFdLCBbMzkuMTM5OTcsIC03Ny4xODI3MV0sIFszOS4xMzk5NywgLTc3LjE4MjcxXSwgWzM5LjEzOTk3LCAtNzcuMTgyNzFdLCBbMzkuMTM5OTcsIC03Ny4xODI3MV0sIFszOS4xMzk5NywgLTc3LjE4MjcxXSwgWzM5LjEzOTk3LCAtNzcuMTgyNzFdLCBbMzkuMDMxNTE2NjY2NjY2NywgLTc3LjA0ODM3NV0sIFszOS4wNDQ0MTY2NjY2NjY3LCAtNzcuMTExOTExNjY2NjY2N10sIFszOS4wMzE1MTY2NjY2NjY3LCAtNzcuMDQ4Mzc1XSwgWzM5LjAzMTUxNjY2NjY2NjcsIC03Ny4wNDgzNzVdLCBbMzkuMDAxMTcsIC03Ni45ODU1NjVdLCBbMzkuMDAxMTcsIC03Ni45ODU1NjVdLCBbMzkuMDQxNzE1LCAtNzYuOTk0OTI2NjY2NjY2N10sIFszOS4wNDU0MjUsIC03Ni45OTA3MzY2NjY2NjY3XSwgWzM5LjA0NTQyNSwgLTc2Ljk5MDczNjY2NjY2NjddLCBbMzkuMDQ1NDI1LCAtNzYuOTkwNzM2NjY2NjY2N10sIFszOS4wNDU0MjUsIC03Ni45OTA3MzY2NjY2NjY3XSwgWzM5LjA2NzQxMTY2NjY2NjcsIC03Ni45MzcyNV0sIFszOS4wOTA4NTE2NjY2NjY3LCAtNzcuMDc5NzQ4MzMzMzMzM10sIFszOS4wMDg0NDUsIC03Ni45OTg0NTMzMzMzMzMzXSwgWzM5LjAwODQ0NSwgLTc2Ljk5ODQ1MzMzMzMzMzNdLCBbMzguOTk5OTksIC03Ny4wMjU3MDVdLCBbMzkuMjg4NTA1LCAtNzcuMjAxOV0sIFszOS4wNTY1NTUsIC03Ny4wNTAwNDVdLCBbMzguOTg3NjA4MzMzMzMzMywgLTc3LjA5ODYwNjY2NjY2NjddLCBbMzguOTkzNDI2NjY2NjY2NywgLTc3LjAyNjg5MzMzMzMzMzNdLCBbMzguOTkzNDI2NjY2NjY2NywgLTc3LjAyNjg5MzMzMzMzMzNdLCBbMzkuMTE0NDgzMzMzMzMzMywgLTc3LjI2NjQyODMzMzMzMzNdLCBbMzkuMDQ1NDI1LCAtNzYuOTkwNzM2NjY2NjY2N10sIFszOS4xMzY2NzE2NjY2NjY3LCAtNzcuMzk1MDg1XSwgWzM5LjE2NzYsIC03Ny4yODAzNTgzMzMzMzMzXSwgWzM4Ljk5NjcxNjY2NjY2NjcsIC03Ny4wMTMwMTE2NjY2NjY3XSwgWzM4Ljk5NjcxNjY2NjY2NjcsIC03Ny4wMTMwMTE2NjY2NjY3XSwgWzM5LjExNjc1ODMzMzMzMzMsIC03Ny4yNjExODMzMzMzMzMzXSwgWzM5LjExNjc1ODMzMzMzMzMsIC03Ny4yNjExODMzMzMzMzMzXSwgWzM5LjExNjc1ODMzMzMzMzMsIC03Ny4yNjExODMzMzMzMzMzXSwgWzM5LjExNjc1ODMzMzMzMzMsIC03Ny4yNjExODMzMzMzMzMzXSwgWzM5LjExNjc1ODMzMzMzMzMsIC03Ny4yNjExODMzMzMzMzMzXSwgWzM5LjExNjc1ODMzMzMzMzMsIC03Ny4yNjExODMzMzMzMzMzXSwgWzM5LjA0ODUxODMzMzMzMzMsIC03Ny4wNTY3ODMzMzMzMzMzXSwgWzM5LjE1MDkyODMzMzMzMzMsIC03Ny4yMDU1NzVdLCBbMzkuMTcwMTc2NjY2NjY2NywgLTc3LjI3ODcyODMzMzMzMzNdLCBbMzkuMDE1NjcxNjY2NjY2NywgLTc3LjA0MzAzODMzMzMzMzNdLCBbMzkuMDE1NjcxNjY2NjY2NywgLTc3LjA0MzAzODMzMzMzMzNdLCBbMzkuMDE1NjcxNjY2NjY2NywgLTc3LjA0MzAzODMzMzMzMzNdLCBbMzguOTkxODE4MzMzMzMzMywgLTc3LjAzMDgxMzMzMzMzMzNdLCBbMzguOTkxODE4MzMzMzMzMywgLTc3LjAzMDgxMzMzMzMzMzNdLCBbMzkuMTQ5NDM4MzMzMzMzMywgLTc3LjE4OTEyXSwgWzM5LjE0OTQzODMzMzMzMzMsIC03Ny4xODkxMl0sIFszOS4xNDk0MzgzMzMzMzMzLCAtNzcuMTg5MTJdLCBbMzkuMTcwMzQxNjY2NjY2NywgLTc3LjI3ODUzMzMzMzMzMzNdLCBbMzguOTgyMDUxNjY2NjY2NywgLTc3LjA5OTgxMzMzMzMzMzNdLCBbMzkuMjA1MDc1LCAtNzcuMjcxMzU1XSwgWzM4Ljk4MjA1MTY2NjY2NjcsIC03Ny4wOTk4MTMzMzMzMzMzXSwgWzM5LjE3MjE4ODMzMzMzMzMsIC03Ny4yMDM4NTVdLCBbMzkuMDIwODMsIC03Ny4wNzM0NTE2NjY2NjY3XSwgWzM5LjAyMDgzLCAtNzcuMDczNDUxNjY2NjY2N10sIFszOC45ODIwNTE2NjY2NjY3LCAtNzcuMDk5ODEzMzMzMzMzM10sIFszOS4xNTAzODgzMzMzMzMzLCAtNzcuMDYwODc2NjY2NjY2N10sIFszOS4xNTAzODgzMzMzMzMzLCAtNzcuMDYwODc2NjY2NjY2N10sIFszOS4yMTc1MDgzMzMzMzMzLCAtNzcuMjc3ODA4MzMzMzMzM10sIFszOS4xNjk4ODgzMzMzMzMzLCAtNzcuMjc5MDA2NjY2NjY2N10sIFszOS4yMTc1MDgzMzMzMzMzLCAtNzcuMjc3ODA4MzMzMzMzM10sIFszOS4xNTAzNzE2NjY2NjY3LCAtNzcuMTkwODY1XSwgWzM5LjE1MDM3MTY2NjY2NjcsIC03Ny4xOTA4NjVdLCBbMzkuMTUwMzcxNjY2NjY2NywgLTc3LjE5MDg2NV0sIFszOC45ODIwNTE2NjY2NjY3LCAtNzcuMDk5ODEzMzMzMzMzM10sIFszOC45ODIwNTE2NjY2NjY3LCAtNzcuMDk5ODEzMzMzMzMzM10sIFszOS4yMjkwOTY2NjY2NjY3LCAtNzcuMjY3MzY2NjY2NjY2N10sIFszOC45OTQ3MjY2NjY2NjY3LCAtNzcuMDI0MjAzMzMzMzMzM10sIFszOS4xMTA4MzgzMzMzMzMzLCAtNzcuMTg3ODU2NjY2NjY2N10sIFszOS4xNzAyNTE2NjY2NjY3LCAtNzcuMjc4NjA2NjY2NjY2N10sIFszOS4xMDU1MTY2NjY2NjY3LCAtNzcuMTgyMzg1XSwgWzM5LjEwNTUxNjY2NjY2NjcsIC03Ny4xODIzODVdLCBbMzkuMTAxNzkxNjY2NjY2NywgLTc3LjMwNzUyNV0sIFszOS4xNjc5NDMzMzMzMzMzLCAtNzcuMjgwMjI1XSwgWzM5LjE0ODQ5LCAtNzcuMTkyMTRdLCBbMzkuMTQ4NDksIC03Ny4xOTIxNF0sIFszOC45ODIwNTE2NjY2NjY3LCAtNzcuMDk5ODEzMzMzMzMzM10sIFszOS4wOTUzMzE2NjY2NjY3LCAtNzcuMTc2ODkxNjY2NjY2N10sIFszOS4xMDE3OTE2NjY2NjY3LCAtNzcuMzA3NTI1XSwgWzM5LjE0MzUxLCAtNzcuNDA2MjUxNjY2NjY2N10sIFszOS4wNDEzMDgzMzMzMzMzLCAtNzcuMTEzMTc4MzMzMzMzM10sIFszOS4xNTk3MDgzMzMzMzMzLCAtNzcuMjg5OTI1XSwgWzM5LjEwMTY5NjY2NjY2NjcsIC03Ny4zMDc0NTVdLCBbMzkuMTAxNjk2NjY2NjY2NywgLTc3LjMwNzQ1NV0sIFszOS4wNzcxNCwgLTc3LjE2ODM4MTY2NjY2NjddLCBbMzguOTgyMDUxNjY2NjY2NywgLTc3LjA5OTgxMzMzMzMzMzNdLCBbMzkuMTQ1ODIzMzMzMzMzMywgLTc3LjE4NDQ2MzMzMzMzMzNdLCBbMzkuMDg0NjM4MzMzMzMzMywgLTc3LjA3OTkxXSwgWzM5LjE3MDI3NSwgLTc3LjI3ODYyXSwgWzM5LjA0MTc3LCAtNzcuMTEzMjFdLCBbMzkuMDQxNzcsIC03Ny4xMTMyMV0sIFszOS4xMDE3ODY2NjY2NjY3LCAtNzcuMzA3NTI2NjY2NjY2N10sIFszOS4wNzUxNiwgLTc3LjE2MTkyMTY2NjY2NjddLCBbMzkuMDc1MTYsIC03Ny4xNjE5MjE2NjY2NjY3XSwgWzM4Ljk4MTk4NSwgLTc3LjA5OTc5MzMzMzMzMzNdLCBbMzkuMDc1MTYsIC03Ny4xNjE5MjE2NjY2NjY3XSwgWzM4Ljk4MTk4NSwgLTc3LjA5OTc5MzMzMzMzMzNdLCBbMzkuMTU5NzAxNjY2NjY2NywgLTc3LjI4OTk0XSwgWzAuMCwgMC4wXSwgWzM5LjE0NTQ0ODMzMzMzMzMsIC03Ny4xODQ0MDgzMzMzMzMzXSwgWzM5LjEwMTc4NjY2NjY2NjcsIC03Ny4zMDc1MjY2NjY2NjY3XSwgWzM5LjEwMTc4NjY2NjY2NjcsIC03Ny4zMDc1MjY2NjY2NjY3XSwgWzM5LjA3MTc3NSwgLTc3LjE2MTk0MzMzMzMzMzNdLCBbMzkuMDcxNzc1LCAtNzcuMTYxOTQzMzMzMzMzM10sIFszOS4wODQ1OCwgLTc3LjA3OTg1ODMzMzMzMzNdLCBbMzkuMDg0NTgsIC03Ny4wNzk4NTgzMzMzMzMzXSwgWzM5LjA0MTQwMzMzMzMzMzMsIC03Ny4xMTMxNl0sIFszOS4wMzY5NywgLTc3LjA1Mjk0ODMzMzMzMzNdLCBbMzkuMDQxNzE1LCAtNzYuOTk0OTI2NjY2NjY2N10sIFszOS4xMDE3MTE2NjY2NjY3LCAtNzcuMzA3NDY1XSwgWzM4Ljk5Nzc4ODMzMzMzMzMsIC03Ni45OTE2Nl0sIFszOS4wODQ2MzgzMzMzMzMzLCAtNzcuMDc5OTFdLCBbMzkuMDQxMjg1LCAtNzcuMTEzMTY1XSwgWzAuMCwgMC4wXSwgWzM5LjE1MTIyODMzMzMzMzMsIC03Ny4xODg3MjY2NjY2NjY3XSwgWzM5LjE1MTIyODMzMzMzMzMsIC03Ny4xODg3MjY2NjY2NjY3XSwgWzM5LjE1MTIyODMzMzMzMzMsIC03Ny4xODg3MjY2NjY2NjY3XSwgWzM5LjIwNjczLCAtNzcuMjQxNzM4MzMzMzMzM10sIFszOS4wNTM4ODE2NjY2NjY3LCAtNzcuMTUzNTA1XSwgWzM5LjA1Mzg4MTY2NjY2NjcsIC03Ny4xNTM1MDVdLCBbMzkuMjIwMzI2NjY2NjY2NywgLTc3LjA2MDE2NjY2NjY2NjddLCBbMzkuMDQxNDk1LCAtNzcuMDUwODQxNjY2NjY2N10sIFszOS4wNDE0OTUsIC03Ny4wNTA4NDE2NjY2NjY3XSwgWzM5LjA0MTQ5NSwgLTc3LjA1MDg0MTY2NjY2NjddLCBbMC4wLCAwLjBdLCBbMzkuMDQxNzE1LCAtNzYuOTk0OTI2NjY2NjY2N10sIFszOS4wNDE3MTUsIC03Ni45OTQ5MjY2NjY2NjY3XSwgWzM5LjExNTg1NSwgLTc2Ljk2MTc3MTY2NjY2NjddLCBbMzkuMTczNDE4MzMzMzMzMywgLTc3LjE4OTMyXSwgWzM5LjE3MzQxODMzMzMzMzMsIC03Ny4xODkzMl0sIFszOS4wNDE0MzUsIC03Ny4xMTMxMzgzMzMzMzMzXSwgWzM5LjIyMDMyNjY2NjY2NjcsIC03Ny4wNjAxNjY2NjY2NjY3XSwgWzM4Ljk5NzQ0NSwgLTc2Ljk5MjAxXSwgWzM5LjEyMzExMzMzMzMzMzMsIC03Ny4yMjk5NDgzMzMzMzMzXSwgWzM5LjEyMzExMzMzMzMzMzMsIC03Ny4yMjk5NDgzMzMzMzMzXSwgWzM5LjEyMzExMzMzMzMzMzMsIC03Ny4yMjk5NDgzMzMzMzMzXSwgWzM5LjEyMzExMzMzMzMzMzMsIC03Ny4yMjk5NDgzMzMzMzMzXSwgWzM5LjEyMzExMzMzMzMzMzMsIC03Ny4yMjk5NDgzMzMzMzMzXSwgWzM5LjEyMzExMzMzMzMzMzMsIC03Ny4yMjk5NDgzMzMzMzMzXSwgWzAuMCwgMC4wXSwgWzM5LjAyMTQ5ODMzMzMzMzMsIC03Ny4wMTQ0NDE2NjY2NjY3XSwgWzM5LjA0NzQyMTY2NjY2NjcsIC03Ny4xNTA1MzgzMzMzMzMzXSwgWzM4Ljk5NzUwMTY2NjY2NjcsIC03Ni45ODc4OTE2NjY2NjY3XSwgWzM5LjEyMTc3LCAtNzcuMTU5NTg1XSwgWzM5LjA1MDk4NSwgLTc3LjEwOTUwODMzMzMzMzNdLCBbMzkuMDA3MDI4MzMzMzMzMywgLTc3LjAyMTE1NjY2NjY2NjddLCBbMzkuMDA3MDI4MzMzMzMzMywgLTc3LjAyMTE1NjY2NjY2NjddLCBbMC4wLCAwLjBdLCBbMC4wLCAwLjBdLCBbMC4wLCAwLjBdLCBbMzkuMjIwMzI2NjY2NjY2NywgLTc3LjA2MDE2NjY2NjY2NjddLCBbMC4wLCAwLjBdLCBbMzkuMTUwNzM2NjY2NjY2NywgLTc3LjE5MTQxNV0sIFszOS4xNTA3MzY2NjY2NjY3LCAtNzcuMTkxNDE1XSwgWzM5LjE2ODcyMTY2NjY2NjcsIC03Ny4yNzk3MDgzMzMzMzMzXSwgWzM5LjE2ODcyMTY2NjY2NjcsIC03Ny4yNzk3MDgzMzMzMzMzXSwgWzM5LjExNTc3MTY2NjY2NjcsIC03Ni45NjE1ODY2NjY2NjY3XSwgWzM5LjEwMTY2ODMzMzMzMzMsIC03Ny4zMDc0NTVdLCBbMzkuMTAxNjY4MzMzMzMzMywgLTc3LjMwNzQ1NV0sIFszOS4wMzc4MzE2NjY2NjY3LCAtNzcuMDUzNDExNjY2NjY2N10sIFszOS4wMzc4MzE2NjY2NjY3LCAtNzcuMDUzNDExNjY2NjY2N10sIFszOS4xMDk5MDUsIC03Ny4xNTM5NV0sIFszOC45OTY3NTMzMzMzMzMzLCAtNzYuOTkxNTA1XSwgWzM5LjAzMTAxLCAtNzcuMDA0OTgxNjY2NjY2N10sIFszOS4wMzEwMSwgLTc3LjAwNDk4MTY2NjY2NjddLCBbMzkuMDM1NjcxNjY2NjY2NywgLTc3LjEyNDIzNV0sIFszOS4wMzU2NzE2NjY2NjY3LCAtNzcuMTI0MjM1XSwgWzM5LjIyMDMyNjY2NjY2NjcsIC03Ny4wNjAxNjY2NjY2NjY3XSwgWzM5LjA0NTQyNSwgLTc2Ljk5MDczNjY2NjY2NjddLCBbMzkuMDQ1NDI1LCAtNzYuOTkwNzM2NjY2NjY2N10sIFszOS4yNjYxLCAtNzcuMjAxMDY1XSwgWzM4Ljk5MDYyLCAtNzcuMDAxODFdLCBbMzguOTkwNjIsIC03Ny4wMDE4MV0sIFszOS4xMTU4NTMzMzMzMzMzLCAtNzcuMDc1NDQxNjY2NjY2N10sIFszOS4xMTU4NTMzMzMzMzMzLCAtNzcuMDc1NDQxNjY2NjY2N10sIFszOS4wNzgzMTE2NjY2NjY3LCAtNzcuMTI5MDQ1XSwgWzM5LjA3ODMxMTY2NjY2NjcsIC03Ny4xMjkwNDVdLCBbMzkuMDM5NjcxNjY2NjY2NywgLTc3LjA1NDU1MTY2NjY2NjddLCBbMzkuMjI1ODY1LCAtNzcuMjYzNDU1XSwgWzM5LjIyNTg2NSwgLTc3LjI2MzQ1NV0sIFszOS4yMjAzMjY2NjY2NjY3LCAtNzcuMDYwMTY2NjY2NjY2N10sIFszOS4yMjAzMjY2NjY2NjY3LCAtNzcuMDYwMTY2NjY2NjY2N10sIFszOS4xNjI0MDY2NjY2NjY3LCAtNzcuMjUwNjEzMzMzMzMzM10sIFszOS4xNjI0MDY2NjY2NjY3LCAtNzcuMjUwNjEzMzMzMzMzM10sIFszOS4wMzEwMzgzMzMzMzMzLCAtNzcuMDA0ODRdLCBbMzguOTk3MjgxNjY2NjY2NywgLTc2Ljk5NDAwODMzMzMzMzNdLCBbMzguOTk3MjgxNjY2NjY2NywgLTc2Ljk5NDAwODMzMzMzMzNdLCBbMzkuMTUwMzc4MzMzMzMzMywgLTc3LjE5MDk1XSwgWzM5LjE1MDM3ODMzMzMzMzMsIC03Ny4xOTA5NV0sIFszOS4xNTAzNzgzMzMzMzMzLCAtNzcuMTkwOTVdLCBbMzkuMDc3NzAxNjY2NjY2NywgLTc3LjEyNTk3NjY2NjY2NjddLCBbMzkuMDc3NzAxNjY2NjY2NywgLTc3LjEyNTk3NjY2NjY2NjddLCBbMzkuMDQ5NDg1LCAtNzcuMDY4MDk1XSwgWzM5LjA0OTQ4NSwgLTc3LjA2ODA5NV0sIFszOS4wNDk0ODUsIC03Ny4wNjgwOTVdLCBbMzkuMDQ5NDg1LCAtNzcuMDY4MDk1XSwgWzM5LjA0OTQ4NSwgLTc3LjA2ODA5NV0sIFszOC45OTY3NSwgLTc2Ljk5MTcyNV0sIFszOC45OTcxOCwgLTc3LjAyNjAyMzMzMzMzMzNdLCBbMzguOTk3MTgsIC03Ny4wMjYwMjMzMzMzMzMzXSwgWzAuMCwgMC4wXSwgWzAuMCwgMC4wXSwgWzAuMCwgMC4wXSwgWzM5LjA5NjY5ODMzMzMzMzMsIC03Ny4yNjU2NDVdLCBbMzkuMDk2Njk4MzMzMzMzMywgLTc3LjI2NTY0NV0sIFszOS4wOTc1MSwgLTc2Ljk0ODE2ODMzMzMzMzNdLCBbMzkuMjIwMjY2NjY2NjY2NywgLTc3LjA2MDE5MzMzMzMzMzNdLCBbMzguOTk5NjQsIC03Ni45OTMyNjVdLCBbMzguOTk5NjQsIC03Ni45OTMyNjVdLCBbMzkuMDI0NzIzMzMzMzMzMywgLTc3LjAxMTIzMzMzMzMzMzNdLCBbMzguOTk5ODAxNjY2NjY2NywgLTc2Ljk5MjMyNjY2NjY2NjddLCBbMzkuMTA3NzU1LCAtNzYuOTk4OTk4MzMzMzMzM10sIFszOS4wOTAwOCwgLTc3LjA1MzYzODMzMzMzMzNdLCBbMzkuMTA3NzU1LCAtNzYuOTk4OTk4MzMzMzMzM10sIFszOS4xMDc3NTUsIC03Ni45OTg5OTgzMzMzMzMzXSwgWzM5LjEwNzc1NSwgLTc2Ljk5ODk5ODMzMzMzMzNdLCBbMzkuMTA3NzU1LCAtNzYuOTk4OTk4MzMzMzMzM10sIFszOS4xNTA0MTE2NjY2NjY3LCAtNzcuMTkwOTZdLCBbMzkuMTI4NDk1LCAtNzcuMTg2NDVdLCBbMzkuMTI4NDk1LCAtNzcuMTg2NDVdLCBbMzkuMTI4NDk1LCAtNzcuMTg2NDVdLCBbMzkuMDY3NjcxNjY2NjY2NywgLTc3LjAwMTY0ODMzMzMzMzNdLCBbMzkuMDkwMTMsIC03Ny4wNTM3Nl0sIFszOS4wOTAxMywgLTc3LjA1Mzc2XSwgWzM5LjA5MDEzLCAtNzcuMDUzNzZdLCBbMzkuMDc1MzAxNjY2NjY2NywgLTc3LjExNTE4NjY2NjY2NjddLCBbMzkuMTM2NjcxNjY2NjY2NywgLTc3LjM5NTA4NV0sIFszOS4wNDM3MSwgLTc3LjA2MjQ2MTY2NjY2NjddLCBbMC4wLCAwLjBdLCBbMzkuMTM2NjcxNjY2NjY2NywgLTc3LjM5NTA4NV0sIFszOS4wNzQ0ODgzMzMzMzMzLCAtNzcuMTY1MjY2NjY2NjY2N10sIFszOS4yNjYxLCAtNzcuMjAxMDY1XSwgWzM5LjAwOTQ3NSwgLTc2Ljk5OTQwMzMzMzMzMzNdLCBbMzkuMDA5NDc1LCAtNzYuOTk5NDAzMzMzMzMzM10sIFszOS4wNzU0NzUsIC03Ny4xMTUwNzMzMzMzMzMzXSwgWzM5LjA3NTQ3NSwgLTc3LjExNTA3MzMzMzMzMzNdLCBbMzkuMDc1NDc1LCAtNzcuMTE1MDczMzMzMzMzM10sIFszOS4wNzU0NzUsIC03Ny4xMTUwNzMzMzMzMzMzXSwgWzM5LjA3NTQ3NSwgLTc3LjExNTA3MzMzMzMzMzNdLCBbMzkuMDkxMjIzMzMzMzMzMywgLTc3LjIwNDgwODMzMzMzMzNdLCBbMzkuMDc1NzM1LCAtNzcuMDAxODc4MzMzMzMzM10sIFszOS4wNzU3MzUsIC03Ny4wMDE4NzgzMzMzMzMzXSwgWzM5LjA3NTczNSwgLTc3LjAwMTg3ODMzMzMzMzNdLCBbMzkuMDc1NzM1LCAtNzcuMDAxODc4MzMzMzMzM10sIFszOS4wNzU3MzUsIC03Ny4wMDE4NzgzMzMzMzMzXSwgWzM5LjA0NTQyNSwgLTc2Ljk5MDczNjY2NjY2NjddLCBbMzguOTk0NDAzMzMzMzMzMywgLTc3LjAyODMzMzMzMzMzMzNdLCBbMzkuMDMyMjQzMzMzMzMzMywgLTc3LjA3NTA0NV0sIFszOS4wNTI1MzgzMzMzMzMzLCAtNzcuMTY2MzU4MzMzMzMzM10sIFszOS4wMzIyNDMzMzMzMzMzLCAtNzcuMDc1MDQ1XSwgWzM5LjA2NjEzMzMzMzMzMzMsIC03Ny4wMDA1MzMzMzMzMzMzXSwgWzM5LjI2NjEsIC03Ny4yMDEwNjVdLCBbMC4wLCAwLjBdLCBbMC4wLCAwLjBdLCBbMzkuMTIxMTIsIC03Ny4wMTE1MjY2NjY2NjY3XSwgWzAuMCwgMC4wXSwgWzAuMCwgMC4wXSwgWzAuMCwgMC4wXSwgWzAuMCwgMC4wXSwgWzM5LjAzMTEwNSwgLTc3LjAwNDkzNjY2NjY2NjddLCBbMzkuMTg0NjMsIC03Ny4yMTA5OTMzMzMzMzMzXSwgWzM5LjE4NDYzLCAtNzcuMjEwOTkzMzMzMzMzM10sIFszOS4wODM4MDY2NjY2NjY3LCAtNzYuOTUwMTczMzMzMzMzM10sIFszOS4xODczNTY2NjY2NjY3LCAtNzcuMjU3MzRdLCBbMzkuMTg3MzU2NjY2NjY2NywgLTc3LjI1NzM0XSwgWzM5LjE4NzM1NjY2NjY2NjcsIC03Ny4yNTczNF0sIFszOS4xODczNTY2NjY2NjY3LCAtNzcuMjU3MzRdLCBbMzkuMTg3MzU2NjY2NjY2NywgLTc3LjI1NzM0XSwgWzM5LjE4NzM1NjY2NjY2NjcsIC03Ny4yNTczNF0sIFszOC45OTkzNjMzMzMzMzMzLCAtNzcuMDE3Nzg1XSwgWzM4Ljk5MDcsIC03Ny4wMjE5MTVdLCBbMzkuMTQ5Mjc4MzMzMzMzMywgLTc3LjA2NjYyXSwgWzM5LjE0OTI3ODMzMzMzMzMsIC03Ny4wNjY2Ml0sIFszOS4xNDkyNzgzMzMzMzMzLCAtNzcuMDY2NjJdLCBbMzkuMTQ5Mjc4MzMzMzMzMywgLTc3LjA2NjYyXSwgWzM5LjE0OTI3ODMzMzMzMzMsIC03Ny4wNjY2Ml0sIFszOS4xNDkyNzgzMzMzMzMzLCAtNzcuMDY2NjJdLCBbMzkuMTYwMTk1LCAtNzcuMjg5NDcxNjY2NjY2N10sIFszOS4xNDgyODMzMzMzMzMzLCAtNzcuMjc2OTc2NjY2NjY2N10sIFszOS4wODk5NTMzMzMzMzMzLCAtNzcuMDI3NzZdLCBbMzkuMDg5OTUzMzMzMzMzMywgLTc3LjAyNzc2XSwgWzAuMCwgMC4wXSwgWzM5LjE0ODU4MTY2NjY2NjcsIC03Ny4yNzkwOV0sIFszOC45OTY3ODY2NjY2NjY3LCAtNzcuMDI3NjIxNjY2NjY2N10sIFszOC45ODIzMTE2NjY2NjY3LCAtNzcuMDc1MzI2NjY2NjY2N10sIFszOC45OTc5NTgzMzMzMzMzLCAtNzcuMDI2NzU2NjY2NjY2N10sIFszOC45OTc5NTgzMzMzMzMzLCAtNzcuMDI2NzU2NjY2NjY2N10sIFszOS4xNTk3MDE2NjY2NjY3LCAtNzcuMjg5OTczMzMzMzMzM10sIFszOS4xNDg1MzE2NjY2NjY3LCAtNzcuMjc2NzczMzMzMzMzM10sIFszOS4wNjcwOTMzMzMzMzMzLCAtNzcuMDAxMzY4MzMzMzMzM10sIFszOS4xNTk3NjY2NjY2NjY3LCAtNzcuMjg5OTE4MzMzMzMzM10sIFszOS4xNDgyNjUsIC03Ny4yNzc0XSwgWzM5LjAxMjYwNjY2NjY2NjcsIC03Ny4xMDc5NTY2NjY2NjY3XSwgWzM5LjIyMDM0LCAtNzcuMDYwMTZdLCBbMC4wLCAwLjBdLCBbMzkuMTU5ODAzMzMzMzMzMywgLTc3LjI4OTgyMzMzMzMzMzNdLCBbMzkuMTEzODYsIC03Ny4xOTYzMzY2NjY2NjY3XSwgWzM5LjE0ODM3MTY2NjY2NjcsIC03Ny4yNzY5MV0sIFszOC45ODIxNjMzMzMzMzMzLCAtNzcuMDc2NDMzMzMzMzMzM10sIFszOC45ODIxNjMzMzMzMzMzLCAtNzcuMDc2NDMzMzMzMzMzM10sIFszOS4wNjY3MDgzMzMzMzMzLCAtNzcuMDAxMDgxNjY2NjY2N10sIFszOS4xMjM5NDUsIC03Ny4xNzU1NjY2NjY2NjY3XSwgWzM5LjEyMzk0NSwgLTc3LjE3NTU2NjY2NjY2NjddLCBbMzkuMjIwMzQsIC03Ny4wNjAxNl0sIFszOS4wMDIxNDE2NjY2NjY3LCAtNzcuMDM5MDU2NjY2NjY2N10sIFswLjAsIDAuMF0sIFszOS4xMjM5NDUsIC03Ny4xNzU1NjY2NjY2NjY3XSwgWzM5LjE1OTU5NjY2NjY2NjcsIC03Ny4yODk4OTE2NjY2NjY3XSwgWzM5LjExMzAzLCAtNzcuMTkyNjYxNjY2NjY2N10sIFszOS4wMTI3MjE2NjY2NjY3LCAtNzcuMTA3OTk1XSwgWzM5LjEyMzY2MTY2NjY2NjcsIC03Ny4xNzUxM10sIFszOS4xMjM2NjE2NjY2NjY3LCAtNzcuMTc1MTNdLCBbMzkuMjIwMzQsIC03Ny4wNjAxNl0sIFszOS4wNjEwNDUsIC03Ni45OTg3MzMzMzMzMzMzXSwgWzM4Ljk5MDE5MzMzMzMzMzMsIC03Ny4wOTUzNTMzMzMzMzMzXSwgWzAuMCwgMC4wXSwgWzM4Ljk5MDA2MzMzMzMzMzMsIC03Ny4wOTUxMTVdLCBbMzkuMDY5NDUxNjY2NjY2NywgLTc3LjE2OTQ5MzMzMzMzMzNdLCBbMzkuMDEzMTA4MzMzMzMzMywgLTc3LjEwNzYzODMzMzMzMzNdLCBbMzkuMjIwMzQsIC03Ny4wNjAxNl0sIFszOS4yMjAzNCwgLTc3LjA2MDE2XSwgWzM5LjA1MjQ1ODMzMzMzMzMsIC03Ny4xNjU3NjVdLCBbMzkuMTU5Njk4MzMzMzMzMywgLTc3LjI4OTk0XSwgWzAuMCwgMC4wXSwgWzM5LjA1MjQ1NjY2NjY2NjcsIC03Ny4xNjU4MzMzMzMzMzMzXSwgWzM5LjAxMjA1NSwgLTc3LjEwODMzNjY2NjY2NjddLCBbMzkuMTI2MDc1LCAtNzcuMTczOTVdLCBbMzkuMTI2MDc1LCAtNzcuMTczOTVdLCBbMzkuMTU5Njk4MzMzMzMzMywgLTc3LjI4OTkxXSwgWzM5LjAxMTQ3ODMzMzMzMzMsIC03Ni45Nzk1NDE2NjY2NjY3XSwgWzM5LjE1MTg1ODMzMzMzMzMsIC03Ny4yODM1XSwgWzAuMCwgMC4wXSwgWzM5LjAwMDE0ODMzMzMzMzMsIC03Ni45OTA4NF0sIFszOS4wMDAxNDgzMzMzMzMzLCAtNzYuOTkwODRdLCBbMzguOTk0NjY4MzMzMzMzMywgLTc3LjAyNzYyNjY2NjY2NjddLCBbMzkuMTE5MTY4MzMzMzMzMywgLTc3LjAxOTg1XSwgWzM5LjExNjA1MzMzMzMzMzMsIC03Ny4wMTc3NjE2NjY2NjY3XSwgWzAuMCwgMC4wXSwgWzM5LjE0OTI3ODMzMzMzMzMsIC03Ny4wNjY2Ml0sIFszOS4xMTkwMywgLTc3LjAyMDM0MzMzMzMzMzNdLCBbMzkuMDg3MjcxNjY2NjY2NywgLTc3LjIxMTU1NjY2NjY2NjddLCBbMzkuMDg4Nzk4MzMzMzMzMywgLTc3LjIxMDI1MTY2NjY2NjddLCBbMC4wLCAwLjBdLCBbMzguOTY4NjU4MzMzMzMzMywgLTc3LjEwOTEyNV0sIFszOS4xMjE3MzE2NjY2NjY3LCAtNzYuOTkyNTldLCBbMzkuMTE5MDE2NjY2NjY2NywgLTc3LjAyMDM2NjY2NjY2NjddLCBbMzkuMTIxNzMxNjY2NjY2NywgLTc2Ljk5MjU5XSwgWzM5LjE0OTI3ODMzMzMzMzMsIC03Ny4wNjY2Ml0sIFszOC45OTk1LCAtNzcuMDk3NzkxNjY2NjY2N10sIFszOS4xMDYzNSwgLTc3LjE4OTUyMzMzMzMzMzNdLCBbMzkuMTI4MTM1LCAtNzcuMTYxMzhdLCBbMzkuMTE4ODg1LCAtNzcuMDIwOTddLCBbMzkuMDE1NzcsIC03Ny4wNDMyODY2NjY2NjY3XSwgWzM5LjAyODA5MTY2NjY2NjcsIC03Ny4wMTMzODVdLCBbMzkuMTI4MTM1LCAtNzcuMTYxMzhdLCBbMC4wLCAwLjBdLCBbMC4wLCAwLjBdLCBbMC4wLCAwLjBdLCBbMC4wLCAwLjBdLCBbMzkuMDg2Mzc4MzMzMzMzMywgLTc2Ljk5OTUzMTY2NjY2NjddLCBbMzkuMTE1ODQsIC03Ny4wNzM2NV0sIFswLjAsIDAuMF0sIFszOS4xMjgxNjUsIC03Ny4xNjEyNzgzMzMzMzMzXSwgWzM5LjE1MTc1MTY2NjY2NjcsIC03Ny4yMTEwNTY2NjY2NjY3XSwgWzM5LjE1MTc1MTY2NjY2NjcsIC03Ny4yMTEwNTY2NjY2NjY3XSwgWzM5LjAxNDk0ODMzMzMzMzMsIC03Ny4wOTk2NDY2NjY2NjY3XSwgWzM5LjI0MTU4LCAtNzcuMTQzNjMzMzMzMzMzM10sIFszOS4wMzQzMzMzMzMzMzMzLCAtNzcuMDcxNjU4MzMzMzMzM10sIFszOS4wMTM3OTgzMzMzMzMzLCAtNzcuMDk5MzJdLCBbMzkuMDMyMjEsIC03Ny4wNzQzNzE2NjY2NjY3XSwgWzM5LjA5OTQ0MTY2NjY2NjcsIC03Ny4yMTIxNl0sIFszOS4wOTk0NDE2NjY2NjY3LCAtNzcuMjEyMTZdLCBbMzguOTg1NTkxNjY2NjY2NywgLTc3LjA5MzcyNV0sIFszOS4wOTk0NDE2NjY2NjY3LCAtNzcuMjEyMTZdLCBbMzguOTg0MzQ1LCAtNzcuMDk0MjkxNjY2NjY2N10sIFszOS4wNTI1NDUsIC03Ni45NzY5MDMzMzMzMzMzXSwgWzM5LjA1MjU0NSwgLTc2Ljk3NjkwMzMzMzMzMzNdLCBbMzkuMDUyNTQ1LCAtNzYuOTc2OTAzMzMzMzMzM10sIFszOS4wMTg1NzgzMzMzMzMzLCAtNzcuMDE0NzEzMzMzMzMzM10sIFszOS4wNDU0MjUsIC03Ni45OTA3MzY2NjY2NjY3XSwgWzAuMCwgMC4wXSwgWzM5LjAyMDQzODMzMzMzMzMsIC03Ny4xMDMxMzMzMzMzMzMzXSwgWzM5LjA0NTQyNSwgLTc2Ljk5MDczNjY2NjY2NjddLCBbMzkuMDg5NzkxNjY2NjY2NywgLTc3LjEzMDI5MTY2NjY2NjddLCBbMzkuMDg5NzkxNjY2NjY2NywgLTc3LjEzMDI5MTY2NjY2NjddLCBbMzkuMDQ1NDI1LCAtNzYuOTkwNzM2NjY2NjY2N10sIFszOC45OTE3NjMzMzMzMzMzLCAtNzcuMDk1NzI4MzMzMzMzM10sIFszOS4wNDU0MjUsIC03Ni45OTA3MzY2NjY2NjY3XSwgWzM4Ljk5MTE2MTY2NjY2NjcsIC03Ny4wOTQwMzE2NjY2NjY3XSwgWzM4Ljk5NDMyLCAtNzcuMDI4MDFdLCBbMzkuMTQwMDcsIC03Ny4xNTIxMzE2NjY2NjY3XSwgWzM5LjEyOTMxNjY2NjY2NjcsIC03Ny4yMzI0MTgzMzMzMzMzXSwgWzM5LjEyOTMxNjY2NjY2NjcsIC03Ny4yMzI0MTgzMzMzMzMzXSwgWzM5LjE5NzM3ODMzMzMzMzMsIC03Ny4yODI0MjVdLCBbMzkuMTk3Mzc4MzMzMzMzMywgLTc3LjI4MjQyNV0sIFszOS4xOTczNzgzMzMzMzMzLCAtNzcuMjgyNDI1XSwgWzM5LjEwOTU5NjY2NjY2NjcsIC03Ny4wNzc0NzY2NjY2NjY3XSwgWzM5LjAxMDE5NjY2NjY2NjcsIC03Ny4wMDA2NTVdLCBbMzguOTkwMywgLTc3LjA5NTM2XSwgWzM4Ljk5MDM2ODMzMzMzMzMsIC03Ny4wMDE4MjE2NjY2NjY3XSwgWzM4Ljk5MDM2ODMzMzMzMzMsIC03Ny4wMDE4MjE2NjY2NjY3XSwgWzM5LjAwMDUwMzMzMzMzMzMsIC03Ni45ODk4OTVdLCBbMzkuMjYwMzMxNjY2NjY2NywgLTc3LjIyMzExMTY2NjY2NjddLCBbMzkuMjYwMzMxNjY2NjY2NywgLTc3LjIyMzExMTY2NjY2NjddLCBbMzguOTkwMzIxNjY2NjY2NywgLTc3LjA5NTI1ODMzMzMzMzNdLCBbMzkuMjY2MDA4MzMzMzMzMywgLTc3LjIwMTQ3NjY2NjY2NjddXSwKICAgICAgICAgICAgICAgIHsKICAgICAgICAgICAgICAgICAgICBtaW5PcGFjaXR5OiAwLjUsCiAgICAgICAgICAgICAgICAgICAgbWF4Wm9vbTogMTgsCiAgICAgICAgICAgICAgICAgICAgbWF4OiAxLjAsCiAgICAgICAgICAgICAgICAgICAgcmFkaXVzOiAyNSwKICAgICAgICAgICAgICAgICAgICBibHVyOiAxNSwKICAgICAgICAgICAgICAgICAgICBncmFkaWVudDogbnVsbAogICAgICAgICAgICAgICAgICAgIH0pCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwX2EzMmQ2NzVlN2Y5MDQ5YTlhMTQzOTk4MjkwMzE0YTk4KTsKICAgICAgICAKICAgICAgICAKICAgICAgICAKICAgICAgICA8L3NjcmlwdD4KICAgICAgICA=](data:text/html;base64,CiAgICAgICAgPCFET0NUWVBFIGh0bWw+CiAgICAgICAgPGhlYWQ+CiAgICAgICAgICAgIAogICAgICAgIAogICAgICAgICAgICA8bWV0YSBodHRwLWVxdWl2PSJjb250ZW50LXR5cGUiIGNvbnRlbnQ9InRleHQvaHRtbDsgY2hhcnNldD1VVEYtOCIgLz4KICAgICAgICAKICAgICAgICAgICAgCiAgICAgICAgCiAgICAgICAgICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9sZWFmbGV0LzAuNy4zL2xlYWZsZXQuanMiPjwvc2NyaXB0PgogICAgICAgIAogICAgICAgIAogICAgICAgIAogICAgICAgICAgICAKICAgICAgICAKICAgICAgICAgICAgPHNjcmlwdCBzcmM9Imh0dHBzOi8vYWpheC5nb29nbGVhcGlzLmNvbS9hamF4L2xpYnMvanF1ZXJ5LzEuMTEuMS9qcXVlcnkubWluLmpzIj48L3NjcmlwdD4KICAgICAgICAKICAgICAgICAKICAgICAgICAKICAgICAgICAgICAgCiAgICAgICAgCiAgICAgICAgICAgIDxzY3JpcHQgc3JjPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9qcy9ib290c3RyYXAubWluLmpzIj48L3NjcmlwdD4KICAgICAgICAKICAgICAgICAKICAgICAgICAKICAgICAgICAgICAgCiAgICAgICAgCiAgICAgICAgICAgIDxzY3JpcHQgc3JjPSJodHRwczovL3Jhd2dpdGh1Yi5jb20vbHZvb2dkdC9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAvZGV2ZWxvcC9kaXN0L2xlYWZsZXQuYXdlc29tZS1tYXJrZXJzLmpzIj48L3NjcmlwdD4KICAgICAgICAKICAgICAgICAKICAgICAgICAKICAgICAgICAgICAgCiAgICAgICAgCiAgICAgICAgICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9sZWFmbGV0Lm1hcmtlcmNsdXN0ZXIvMC40LjAvbGVhZmxldC5tYXJrZXJjbHVzdGVyLXNyYy5qcyI+PC9zY3JpcHQ+CiAgICAgICAgCiAgICAgICAgCiAgICAgICAgCiAgICAgICAgICAgIAogICAgICAgIAogICAgICAgICAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvbGVhZmxldC5tYXJrZXJjbHVzdGVyLzAuNC4wL2xlYWZsZXQubWFya2VyY2x1c3Rlci5qcyI+PC9zY3JpcHQ+CiAgICAgICAgCiAgICAgICAgCiAgICAgICAgCiAgICAgICAgICAgIAogICAgICAgIAogICAgICAgICAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vY2RuanMuY2xvdWRmbGFyZS5jb20vYWpheC9saWJzL2xlYWZsZXQvMC43LjMvbGVhZmxldC5jc3MiIC8+CiAgICAgICAgCiAgICAgICAgCiAgICAgICAgCiAgICAgICAgICAgIAogICAgICAgIAogICAgICAgICAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vYm9vdHN0cmFwLzMuMi4wL2Nzcy9ib290c3RyYXAubWluLmNzcyIgLz4KICAgICAgICAKICAgICAgICAKICAgICAgICAKICAgICAgICAgICAgCiAgICAgICAgCiAgICAgICAgICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIiAvPgogICAgICAgIAogICAgICAgIAogICAgICAgIAogICAgICAgICAgICAKICAgICAgICAKICAgICAgICAgICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2ZvbnQtYXdlc29tZS80LjEuMC9jc3MvZm9udC1hd2Vzb21lLm1pbi5jc3MiIC8+CiAgICAgICAgCiAgICAgICAgCiAgICAgICAgCiAgICAgICAgICAgIAogICAgICAgIAogICAgICAgICAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vcmF3Z2l0LmNvbS9sdm9vZ2R0L0xlYWZsZXQuYXdlc29tZS1tYXJrZXJzLzIuMC9kZXZlbG9wL2Rpc3QvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuY3NzIiAvPgogICAgICAgIAogICAgICAgIAogICAgICAgIAogICAgICAgICAgICAKICAgICAgICAKICAgICAgICAgICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9sZWFmbGV0Lm1hcmtlcmNsdXN0ZXIvMC40LjAvTWFya2VyQ2x1c3Rlci5EZWZhdWx0LmNzcyIgLz4KICAgICAgICAKICAgICAgICAKICAgICAgICAKICAgICAgICAgICAgCiAgICAgICAgCiAgICAgICAgICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvbGVhZmxldC5tYXJrZXJjbHVzdGVyLzAuNC4wL01hcmtlckNsdXN0ZXIuY3NzIiAvPgogICAgICAgIAogICAgICAgIAogICAgICAgIAogICAgICAgICAgICAKICAgICAgICAKICAgICAgICAgICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL3Jhdy5naXRodWJ1c2VyY29udGVudC5jb20vcHl0aG9uLXZpc3VhbGl6YXRpb24vZm9saXVtL21hc3Rlci9mb2xpdW0vdGVtcGxhdGVzL2xlYWZsZXQuYXdlc29tZS5yb3RhdGUuY3NzIiAvPgogICAgICAgIAogICAgICAgIAogICAgICAgIAogICAgICAgICAgICAKICAgICAgICAgICAgPHN0eWxlPgoKICAgICAgICAgICAgaHRtbCwgYm9keSB7CiAgICAgICAgICAgICAgICB3aWR0aDogMTAwJTsKICAgICAgICAgICAgICAgIGhlaWdodDogMTAwJTsKICAgICAgICAgICAgICAgIG1hcmdpbjogMDsKICAgICAgICAgICAgICAgIHBhZGRpbmc6IDA7CiAgICAgICAgICAgICAgICB9CgogICAgICAgICAgICAjbWFwIHsKICAgICAgICAgICAgICAgIHBvc2l0aW9uOmFic29sdXRlOwogICAgICAgICAgICAgICAgdG9wOjA7CiAgICAgICAgICAgICAgICBib3R0b206MDsKICAgICAgICAgICAgICAgIHJpZ2h0OjA7CiAgICAgICAgICAgICAgICBsZWZ0OjA7CiAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgIDwvc3R5bGU+CiAgICAgICAgICAgIAogICAgICAgIAogICAgICAgICAgICAKICAgICAgICAgICAgPHN0eWxlPiAjbWFwX2EzMmQ2NzVlN2Y5MDQ5YTlhMTQzOTk4MjkwMzE0YTk4IHsKICAgICAgICAgICAgICAgIHBvc2l0aW9uIDogcmVsYXRpdmU7CiAgICAgICAgICAgICAgICB3aWR0aCA6IDEwMC4wJTsKICAgICAgICAgICAgICAgIGhlaWdodDogMTAwLjAlOwogICAgICAgICAgICAgICAgbGVmdDogMC4wJTsKICAgICAgICAgICAgICAgIHRvcDogMC4wJTsKICAgICAgICAgICAgICAgIH0KICAgICAgICAgICAgPC9zdHlsZT4KICAgICAgICAKICAgICAgICAKICAgICAgICAgICAgCiAgICAgICAgCiAgICAgICAgICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2xlYWZsZXQuZ2l0aHViLmlvL0xlYWZsZXQuaGVhdC9kaXN0L2xlYWZsZXQtaGVhdC5qcyI+PC9zY3JpcHQ+CiAgICAgICAgCiAgICAgICAgCiAgICAgICAgCiAgICAgICAgCiAgICAgICAgPC9oZWFkPgogICAgICAgIDxib2R5PgogICAgICAgICAgICAKICAgICAgICAKICAgICAgICAgICAgCiAgICAgICAgICAgIDxkaXYgY2xhc3M9ImZvbGl1bS1tYXAiIGlkPSJtYXBfYTMyZDY3NWU3ZjkwNDlhOWExNDM5OTgyOTAzMTRhOTgiID48L2Rpdj4KICAgICAgICAKICAgICAgICAKICAgICAgICAKICAgICAgICA8L2JvZHk+CiAgICAgICAgPHNjcmlwdD4KICAgICAgICAgICAgCiAgICAgICAgCiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIHNvdXRoV2VzdCA9IEwubGF0TG5nKC05MCwgLTE4MCk7CiAgICAgICAgICAgIHZhciBub3J0aEVhc3QgPSBMLmxhdExuZyg5MCwgMTgwKTsKICAgICAgICAgICAgdmFyIGJvdW5kcyA9IEwubGF0TG5nQm91bmRzKHNvdXRoV2VzdCwgbm9ydGhFYXN0KTsKCiAgICAgICAgICAgIHZhciBtYXBfYTMyZDY3NWU3ZjkwNDlhOWExNDM5OTgyOTAzMTRhOTggPSBMLm1hcCgnbWFwX2EzMmQ2NzVlN2Y5MDQ5YTlhMTQzOTk4MjkwMzE0YTk4JywgewogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY2VudGVyOlszOS4wODM2LC03Ny4xNDgzXSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHpvb206IDExLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbWF4Qm91bmRzOiBib3VuZHMsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBsYXllcnM6IFtdLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY3JzOiBMLkNSUy5FUFNHMzg1NwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIH0pOwogICAgICAgICAgICAKICAgICAgICAKICAgICAgICAKICAgICAgICAgICAgCiAgICAgICAgICAgIHZhciB0aWxlX2xheWVyXzMxYjEzMTJiNmVhODQ0NWE4MmJiMGVlMDBjOTc2NjBjID0gTC50aWxlTGF5ZXIoCiAgICAgICAgICAgICAgICAnaHR0cHM6Ly97c30udGlsZS5vcGVuc3RyZWV0bWFwLm9yZy97en0ve3h9L3t5fS5wbmcnLAogICAgICAgICAgICAgICAgewogICAgICAgICAgICAgICAgICAgIG1heFpvb206IDE4LAogICAgICAgICAgICAgICAgICAgIG1pblpvb206IDEsCiAgICAgICAgICAgICAgICAgICAgYXR0cmlidXRpb246ICdEYXRhIGJ5IDxhIGhyZWY9Imh0dHA6Ly9vcGVuc3RyZWV0bWFwLm9yZyI+T3BlblN0cmVldE1hcDwvYT4sIHVuZGVyIDxhIGhyZWY9Imh0dHA6Ly93d3cub3BlbnN0cmVldG1hcC5vcmcvY29weXJpZ2h0Ij5PRGJMPC9hPi4nLAogICAgICAgICAgICAgICAgICAgIGRldGVjdFJldGluYTogZmFsc2UKICAgICAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hMzJkNjc1ZTdmOTA0OWE5YTE0Mzk5ODI5MDMxNGE5OCk7CgogICAgICAgIAogICAgICAgIAogICAgICAgICAgICAKICAgICAgICAgICAgdmFyIGhlYXRfbWFwX2I3OGIzMzNmZWMzMTQxY2U4ZDNiMGY1NzJiZTM0YjUwID0gTC5oZWF0TGF5ZXIoCiAgICAgICAgICAgICAgICBbWzM5LjAzMjIzLCAtNzcuMTA1OTI1XSwgWzAuMCwgMC4wXSwgWzM5LjA0MTM0NSwgLTc3LjA1MzkxNjY2NjY2NjddLCBbMzkuMDQwNTY1LCAtNzcuMDU2MjgxNjY2NjY2N10sIFszOS4wMTM0NDgzMzMzMzMzLCAtNzcuMTA2MTMzMzMzMzMzM10sIFszOS4wMTM1NjgzMzMzMzMzLCAtNzcuMTA2MjM4MzMzMzMzM10sIFszOS4wMTM2NCwgLTc3LjEwNjIyNjY2NjY2NjddLCBbMzkuMDIzMDA1LCAtNzcuMTAzMjAxNjY2NjY2N10sIFszOS4yNDA2NTMzMzMzMzMzLCAtNzcuMjM1MDIxNjY2NjY2N10sIFszOS4yNDA2NTMzMzMzMzMzLCAtNzcuMjM1MDIxNjY2NjY2N10sIFszOS4yNDA2NTMzMzMzMzMzLCAtNzcuMjM1MDIxNjY2NjY2N10sIFszOS4wMDE3OCwgLTc3LjAzODg1MTY2NjY2NjddLCBbMC4wLCAwLjBdLCBbMzguOTk4MTYsIC03Ni45OTUwODgzMzMzMzMzXSwgWzAuMCwgMC4wXSwgWzAuMCwgMC4wXSwgWzAuMCwgMC4wXSwgWzAuMCwgMC4wXSwgWzAuMCwgMC4wXSwgWzAuMCwgMC4wXSwgWzM5LjA0MTM0NSwgLTc3LjA1MzkxNjY2NjY2NjddLCBbMzkuMDQ3MDM4MzMzMzMzMywgLTc3LjA1MzE5XSwgWzM5LjA0NzAzODMzMzMzMzMsIC03Ny4wNTMxOV0sIFszOS4wNTcxODE2NjY2NjY3LCAtNzcuMDQ5NzE4MzMzMzMzM10sIFszOS4wNTcxODE2NjY2NjY3LCAtNzcuMDQ5NzE4MzMzMzMzM10sIFszOS4wNDA1NjUsIC03Ny4wNTYyODE2NjY2NjY3XSwgWzM4Ljk5OTE5MTY2NjY2NjcsIC03Ny4wMzM4MTY2NjY2NjY3XSwgWzM4Ljk5NzI3LCAtNzcuMDI3NDc2NjY2NjY2N10sIFszOS4wMTkxOTY2NjY2NjY3LCAtNzcuMDE0MjExNjY2NjY2N10sIFszOS4wOTYzMTE2NjY2NjY3LCAtNzcuMjA2NDRdLCBbMzkuMTAyMDA1LCAtNzcuMjEwODgzMzMzMzMzM10sIFszOS4xMDIwMDUsIC03Ny4yMTA4ODMzMzMzMzMzXSwgWzM5LjEwMjAwNSwgLTc3LjIxMDg4MzMzMzMzMzNdLCBbMzkuMDI5Mzk1LCAtNzcuMDYzODE1XSwgWzM5LjAyOTM5NSwgLTc3LjA2MzgxNV0sIFszOS4wMjkzOTUsIC03Ny4wNjM4MTVdLCBbMzkuMDI5Mzk1LCAtNzcuMDYzODE1XSwgWzM5LjAzMTIwNSwgLTc3LjA3MjQyODMzMzMzMzNdLCBbMzkuMDMxMzA1LCAtNzcuMDcyNTI1XSwgWzM4Ljk5MDE2MTY2NjY2NjcsIC03Ny4wMDE1NDgzMzMzMzMzXSwgWzM5LjExMzgyNSwgLTc3LjE5MzY1XSwgWzM5LjAwMzUwNjY2NjY2NjcsIC03Ny4wMzU2MDE2NjY2NjY3XSwgWzM5LjE2NDE3MzMzMzMzMzMsIC03Ny4xNTcyM10sIFszOS4xOTA2NjE2NjY2NjY3LCAtNzcuMjQyMDYzMzMzMzMzM10sIFszOS4xMTg4NCwgLTc3LjAyMTc1MTY2NjY2NjddLCBbMzkuMTI3NzA2NjY2NjY2NywgLTc3LjE2NzA4MzMzMzMzMzNdLCBbMzkuMTUxNzEsIC03Ny4yNzgyNF0sIFszOS4wNjUwMDUsIC03Ny4yNjU5NTVdLCBbMzkuMTAxODI1LCAtNzcuMTQxODRdLCBbMzguOTg1NjUzMzMzMzMzMywgLTc3LjIyNjU0NV0sIFszOS4wNzQ3OCwgLTc2Ljk5ODJdLCBbMzkuMDczOTMxNjY2NjY2NywgLTc3LjAwMDk5MzMzMzMzMzNdLCBbMzkuMTk4OTI4MzMzMzMzMywgLTc3LjI0NDgxMTY2NjY2NjddLCBbMzkuMjc4MTk1LCAtNzcuMzIyNjk2NjY2NjY2N10sIFszOS4yNzgxOTUsIC03Ny4zMjI2OTY2NjY2NjY3XSwgWzM5LjIyODQ4MzMzMzMzMzMsIC03Ny4yODI2MV0sIFszOS4xNzc4NzY2NjY2NjY3LCAtNzcuMjcwNjQ4MzMzMzMzM10sIFszOS4xNzgyNzY2NjY2NjY3LCAtNzcuMjcyNDcxNjY2NjY2N10sIFszOS4xNzgyMzUsIC03Ny4yNzIzMTMzMzMzMzMzXSwgWzM5LjE3NzQzNjY2NjY2NjcsIC03Ny4yNzEzMjY2NjY2NjY3XSwgWzAuMCwgMC4wXSwgWzM5LjE3MTE5ODMzMzMzMzMsIC03Ny4yNzc4MjY2NjY2NjY3XSwgWzM5LjE3MjA0NjY2NjY2NjcsIC03Ny4yNzY2ODgzMzMzMzMzXSwgWzM5LjE2ODg5NjY2NjY2NjcsIC03Ny4yNzk1N10sIFszOS4xNzEwNiwgLTc3LjI3NzkyNV0sIFszOS4xMTEyOTgzMzMzMzMzLCAtNzcuMTYxMjYxNjY2NjY2N10sIFszOS4wNjk1NTMzMzMzMzMzLCAtNzcuMTM4NDQzMzMzMzMzM10sIFszOS4wNDExMDgzMzMzMzMzLCAtNzcuMTYyMjYzMzMzMzMzM10sIFszOS4wOTU1NywgLTc3LjA0NDE5ODMzMzMzMzNdLCBbMzkuMjg3MTIsIC03Ny4yMDgwMV0sIFszOS4yNzU5NjUsIC03Ny4yMTQxNTVdLCBbMzkuMjc2NzI1LCAtNzcuMjEzNDY4MzMzMzMzM10sIFswLjAsIDAuMF0sIFszOC45OTg4MTUsIC03Ni45OTUxMDgzMzMzMzMzXSwgWzM4Ljk5Nzc0ODMzMzMzMzMsIC03Ni45OTQzM10sIFszOC45OTc3NDgzMzMzMzMzLCAtNzYuOTk0MzNdLCBbMzguOTk3NTY1LCAtNzYuOTk0MDkzMzMzMzMzM10sIFszOC45OTQ3OCwgLTc3LjAwNzgwMTY2NjY2NjddLCBbMzguOTk0NzgsIC03Ny4wMDc4MDE2NjY2NjY3XSwgWzM4Ljk5NDc4LCAtNzcuMDA3ODAxNjY2NjY2N10sIFszOC45OTQ3OCwgLTc3LjAwNzgwMTY2NjY2NjddLCBbMzkuMjIwMzEzMzMzMzMzMywgLTc3LjA2MDE2NjY2NjY2NjddLCBbMzkuMDc4NzMzMzMzMzMzMywgLTc3LjAwMTUyMzMzMzMzMzNdLCBbMzkuMDgxNjM4MzMzMzMzMywgLTc3LjAwMTEwMTY2NjY2NjddLCBbMzkuMTA1NzI4MzMzMzMzMywgLTc3LjA4NTIwMzMzMzMzMzNdLCBbMzkuMTA1NjQzMzMzMzMzMywgLTc3LjA4NTM4XSwgWzM5LjEwNzQ0LCAtNzcuMDgyNjNdLCBbMzkuMTA1MzcsIC03Ny4wODU2NzY2NjY2NjY3XSwgWzM5LjEwNTk1MTY2NjY2NjcsIC03Ny4wODUwMDE2NjY2NjY3XSwgWzM5LjEwNTA2MTY2NjY2NjcsIC03Ny4wODU4OF0sIFszOS4xMDQ0MDUsIC03Ny4wODY5NjY2NjY2NjY3XSwgWzM5LjI3ODE5NSwgLTc3LjMyMjY5NjY2NjY2NjddLCBbMzkuMjc4MTk1LCAtNzcuMzIyNjk2NjY2NjY2N10sIFszOS4yNzgxOTUsIC03Ny4zMjI2OTY2NjY2NjY3XSwgWzM5LjA4OTA2NjY2NjY2NjcsIC03Ny4yMTM5NV0sIFszOS4yMjg0ODMzMzMzMzMzLCAtNzcuMjgyNjFdLCBbMzkuMjA1ODA1LCAtNzcuMjQzMTkzMzMzMzMzM10sIFszOS4wMDk5NDY2NjY2NjY3LCAtNzcuMDk3MzMzMzMzMzMzM10sIFszOS4wMDk5NDY2NjY2NjY3LCAtNzcuMDk3MzMzMzMzMzMzM10sIFszOS4wMzAzMDgzMzMzMzMzLCAtNzcuMTA1MDldLCBbMzkuMDIwMjksIC03Ny4wNzcxNjgzMzMzMzMzXSwgWzM5LjAyMDI5LCAtNzcuMDc3MTY4MzMzMzMzM10sIFszOS4wNDc0MjE2NjY2NjY3LCAtNzcuMDc1ODkzMzMzMzMzM10sIFswLjAsIDAuMF0sIFszOS4wNDExMjMzMzMzMzMzLCAtNzcuMTU5MjgzMzMzMzMzM10sIFszOS4wMDc0OTE2NjY2NjY3LCAtNzcuMTEyNTkzMzMzMzMzM10sIFszOC45OTAxNjE2NjY2NjY3LCAtNzcuMDAxNTQ4MzMzMzMzM10sIFszOS4xMTM4MjUsIC03Ny4xOTM2NV0sIFszOS4xMTM4MjUsIC03Ny4xOTM2NV0sIFszOS4xNzE5OCwgLTc3LjI3NzAzODMzMzMzMzNdLCBbMzkuMDAzNTA2NjY2NjY2NywgLTc3LjAzNTYwMTY2NjY2NjddLCBbMzkuMDAzNTA2NjY2NjY2NywgLTc3LjAzNTYwMTY2NjY2NjddLCBbMzkuMDQ1NDI1LCAtNzYuOTkwNzM2NjY2NjY2N10sIFszOS4wNDU0MjUsIC03Ni45OTA3MzY2NjY2NjY3XSwgWzM5LjA0NTQyNSwgLTc2Ljk5MDczNjY2NjY2NjddLCBbMzkuMDQ1NDI1LCAtNzYuOTkwNzM2NjY2NjY2N10sIFszOS4wNDEzNDUsIC03Ny4wNTM5MTY2NjY2NjY3XSwgWzM5LjA0MTM0NSwgLTc3LjA1MzkxNjY2NjY2NjddLCBbMzkuMDQ3MDM4MzMzMzMzMywgLTc3LjA1MzE5XSwgWzM5LjA0NzAzODMzMzMzMzMsIC03Ny4wNTMxOV0sIFszOS4wNDcwMzgzMzMzMzMzLCAtNzcuMDUzMTldLCBbMzkuMDQ3Mjc1LCAtNzcuMDU0MDU1XSwgWzM5LjA2NjkyMTY2NjY2NjcsIC03Ny4wMTg4N10sIFszOS4wNjY5MjE2NjY2NjY3LCAtNzcuMDE4ODddLCBbMzkuMDY2OTIxNjY2NjY2NywgLTc3LjAxODg3XSwgWzM4Ljk4NzI4MzMzMzMzMzMsIC03Ny4wMjM0NjY2NjY2NjY3XSwgWzM4Ljk4Njk2MzMzMzMzMzMsIC03Ny4xMDc2MjMzMzMzMzMzXSwgWzM5LjA4Mzg2NSwgLTc3LjAwMDM0MzMzMzMzMzNdLCBbMzkuMTc2MjMsIC03Ny4yNzExMTE2NjY2NjY3XSwgWzM5LjEyMzkzMTY2NjY2NjcsIC03Ny4xNzk0MDY2NjY2NjY3XSwgWzM5LjEyMzkzMTY2NjY2NjcsIC03Ny4xNzk0MDY2NjY2NjY3XSwgWzM5LjAwNjY1MzMzMzMzMzMsIC03Ny4xOTEzN10sIFszOS4wMzk5MDgzMzMzMzMzLCAtNzcuMTIyMjhdLCBbMzkuMTE0MTk4MzMzMzMzMywgLTc3LjE2Mzg2ODMzMzMzMzNdLCBbMzkuMDI0Nzk4MzMzMzMzMywgLTc3LjA3NjQwNjY2NjY2NjddLCBbMzkuMDUzMTU4MzMzMzMzMywgLTc3LjExMjk1NjY2NjY2NjddLCBbMzkuMDUzMTU4MzMzMzMzMywgLTc3LjExMjk1NjY2NjY2NjddLCBbMzkuMDU0NDgzMzMzMzMzMywgLTc3LjExNzUxNV0sIFszOC45OTIxMjE2NjY2NjY3LCAtNzcuMDMzNzUzMzMzMzMzM10sIFszOC45ODgwMjY2NjY2NjY3LCAtNzcuMDI3XSwgWzM4Ljk4OTgxNjY2NjY2NjcsIC03Ny4wMjY0MjVdLCBbMzkuMDM4NDI1LCAtNzcuMDU4NTYzMzMzMzMzM10sIFszOS4wMzUwNjMzMzMzMzMzLCAtNzcuMTczOTgxNjY2NjY2N10sIFszOS4wMzUwNjMzMzMzMzMzLCAtNzcuMTczOTgxNjY2NjY2N10sIFszOS4wOTc3NzMzMzMzMzMzLCAtNzcuMjA3ODE1XSwgWzM5LjA5Nzc3MzMzMzMzMzMsIC03Ny4yMDc4MTVdLCBbMzkuMDQ1NDI1LCAtNzYuOTkwNzM2NjY2NjY2N10sIFszOS4wNDU0MjUsIC03Ni45OTA3MzY2NjY2NjY3XSwgWzM5LjA0NTQyNSwgLTc2Ljk5MDczNjY2NjY2NjddLCBbMzkuMDQ1NDI1LCAtNzYuOTkwNzM2NjY2NjY2N10sIFszOS4wNDU0MjUsIC03Ni45OTA3MzY2NjY2NjY3XSwgWzM5LjA0NTQyNSwgLTc2Ljk5MDczNjY2NjY2NjddLCBbMzkuMDgyNjE1LCAtNzYuOTQ3MTAzMzMzMzMzM10sIFszOS4wODI2MTUsIC03Ni45NDcxMDMzMzMzMzMzXSwgWzM5LjA4MjYxNSwgLTc2Ljk0NzEwMzMzMzMzMzNdLCBbMzkuMDc3MzA1LCAtNzYuOTQyNzM4MzMzMzMzM10sIFszOS4wNzczMDUsIC03Ni45NDI3MzgzMzMzMzMzXSwgWzM5LjA4MzM5ODMzMzMzMzMsIC03Ni45MzkyMDVdLCBbMzkuMDgzMzk4MzMzMzMzMywgLTc2LjkzOTIwNV0sIFszOS4xMzkxNzE2NjY2NjY3LCAtNzcuMTkzNzI1XSwgWzM5LjEzOTE3MTY2NjY2NjcsIC03Ny4xOTM3MjVdLCBbMzkuMTkwNDAzMzMzMzMzMywgLTc3LjI2NzEyMTY2NjY2NjddLCBbMzkuMTkwNDAzMzMzMzMzMywgLTc3LjI2NzEyMTY2NjY2NjddLCBbMzkuMTY0MzgxNjY2NjY2NywgLTc3LjQ3Njk3NV0sIFszOS4xNjQzODE2NjY2NjY3LCAtNzcuNDc2OTc1XSwgWzM5LjE2NDM4MTY2NjY2NjcsIC03Ny40NzY5NzVdLCBbMzguOTg2NTgzMzMzMzMzMywgLTc3LjEwNzE3ODMzMzMzMzNdLCBbMzguOTg2Njc1LCAtNzcuMTA3Mjg2NjY2NjY2N10sIFszOC45ODY2NTY2NjY2NjY3LCAtNzcuMTA3MzFdLCBbMzguOTg2NjU2NjY2NjY2NywgLTc3LjEwNzMxXSwgWzM4Ljk4Njk2MzMzMzMzMzMsIC03Ny4xMDc2MjMzMzMzMzMzXSwgWzM4Ljk4NjYxNjY2NjY2NjcsIC03Ny4xMDcyMl0sIFszOC45ODY3MjUsIC03Ny4xMDczNDY2NjY2NjY3XSwgWzM5LjAxMTY1MTY2NjY2NjcsIC03Ny4xOTkwODMzMzMzMzMzXSwgWzM5LjA1NDA2NjY2NjY2NjcsIC03Ny4xMzk2ODVdLCBbMzkuMTU0OTI2NjY2NjY2NywgLTc3LjIxNDQ3MzMzMzMzMzNdLCBbMzkuMTU0OTI2NjY2NjY2NywgLTc3LjIxNDQ3MzMzMzMzMzNdLCBbMzkuMTc2NDMzMzMzMzMzMywgLTc3LjM1ODk0XSwgWzM5LjE3NjQzMzMzMzMzMzMsIC03Ny4zNTg5NF0sIFszOS4xMTg3MDMzMzMzMzMzLCAtNzcuMDc3NDA2NjY2NjY2N10sIFszOS4wMTMyMDUsIC03Ny4xMDY4MjVdLCBbMzkuMDEzMTYxNjY2NjY2NywgLTc3LjEwNjg3XSwgWzM5LjA3OTkwNSwgLTc3LjAwMTQwMzMzMzMzMzNdLCBbMzkuMTQ4MTgxNjY2NjY2NywgLTc3LjI3ODY3ODMzMzMzMzNdLCBbMzkuMTQ3NjI4MzMzMzMzMywgLTc3LjI3NTk4MTY2NjY2NjddLCBbMzkuMTcwNDg1LCAtNzcuMjc4Mzk1XSwgWzM5LjE3MjkzNSwgLTc3LjI2NDUxNjY2NjY2NjddLCBbMzkuMTcxODA2NjY2NjY2NywgLTc3LjI2MzQ5MzMzMzMzMzNdLCBbMzkuMDk2NjksIC03Ni45Mzc1NTE2NjY2NjY3XSwgWzM5LjA3NTU5LCAtNzYuOTk0NTI4MzMzMzMzM10sIFszOS4xNTc2MjE2NjY2NjY3LCAtNzcuMDQ4OTA4MzMzMzMzM10sIFszOS4wMTIwMjE2NjY2NjY3LCAtNzcuMTk5NzA4MzMzMzMzM10sIFszOS4wMTM0MzY2NjY2NjY3LCAtNzcuMjAzMTIzMzMzMzMzM10sIFszOS4wNzQ5MjE2NjY2NjY3LCAtNzcuMDkzOTJdLCBbMzkuMTg2Nzc1LCAtNzcuNDA1MzUzMzMzMzMzM10sIFszOS4xODY3NzUsIC03Ny40MDUzNTMzMzMzMzMzXSwgWzM5LjA5NjEwNjY2NjY2NjcsIC03Ny4xNzcxMzMzMzMzMzMzXSwgWzM5LjA5MTU5ODMzMzMzMzMsIC03Ny4xNzUyODE2NjY2NjY3XSwgWzM5LjA2NDQxNjY2NjY2NjcsIC03Ny4xNTYyNTY2NjY2NjY3XSwgWzM5LjA0NTUzNSwgLTc3LjE0OTIyMTY2NjY2NjddLCBbMzkuMDU2MzMsIC03Ny4xMTcyMTgzMzMzMzMzXSwgWzM5LjAzNDcwMTY2NjY2NjcsIC03Ny4wNzA4MTMzMzMzMzMzXSwgWzM5LjAzNDU5NjY2NjY2NjcsIC03Ny4wNzE2MTgzMzMzMzMzXSwgWzM5LjA4MzM5ODMzMzMzMzMsIC03Ni45MzkyMDVdLCBbMzkuMDY5MzY1LCAtNzcuMTA0OTE4MzMzMzMzM10sIFszOS4wNjkzNjUsIC03Ny4xMDQ5MTgzMzMzMzMzXSwgWzM5LjA2OTM2NSwgLTc3LjEwNDkxODMzMzMzMzNdLCBbMzkuMTI1MDkzMzMzMzMzMywgLTc3LjE4MTg3MTY2NjY2NjddLCBbMzkuMTI1MDkzMzMzMzMzMywgLTc3LjE4MTg3MTY2NjY2NjddLCBbMzkuMTI1MDkzMzMzMzMzMywgLTc3LjE4MTg3MTY2NjY2NjddLCBbMzkuMTI1MDkzMzMzMzMzMywgLTc3LjE4MTg3MTY2NjY2NjddLCBbMzkuMTIyODM1LCAtNzcuMTc4MTI4MzMzMzMzM10sIFszOS4xMzkxNzE2NjY2NjY3LCAtNzcuMTkzNzI1XSwgWzM5LjAxMjE2NSwgLTc3LjE5OTg2ODMzMzMzMzNdLCBbMzkuMDA2NjUzMzMzMzMzMywgLTc3LjE5MTM3XSwgWzM5LjAxNDc3LCAtNzcuMjA0NDI1XSwgWzM5LjExODMxMTY2NjY2NjcsIC03Ny4zMTY1Ml0sIFszOS4wMTQ5NTMzMzMzMzMzLCAtNzcuMDAwNDMzMzMzMzMzM10sIFszOS4wMTQ5NTMzMzMzMzMzLCAtNzcuMDAwNDMzMzMzMzMzM10sIFszOS4wNTM3NjY2NjY2NjY3LCAtNzcuMDUwNzk4MzMzMzMzM10sIFszOC45ODgwMjY2NjY2NjY3LCAtNzcuMDI3XSwgWzM4Ljk4ODAyNjY2NjY2NjcsIC03Ny4wMjddLCBbMzguOTg5ODE2NjY2NjY2NywgLTc3LjAyNjQyNV0sIFszOS4wMDQ3MTUsIC03Ny4wNzc3NjY2NjY2NjY3XSwgWzM5LjAzMDI3LCAtNzcuMDg0ODU1XSwgWzM5LjAzMDI3LCAtNzcuMDg0ODU1XSwgWzM5LjE0NDk1NjY2NjY2NjcsIC03Ny4yMTg5OTgzMzMzMzMzXSwgWzM5LjE0NDk1NjY2NjY2NjcsIC03Ny4yMTg5OTgzMzMzMzMzXSwgWzM5LjAyOTc5MzMzMzMzMzMsIC03Ny4wNzQ1NTgzMzMzMzMzXSwgWzM5LjAyOTc5MzMzMzMzMzMsIC03Ny4wNzQ1NTgzMzMzMzMzXSwgWzM5LjAyOTc5MzMzMzMzMzMsIC03Ny4wNzQ1NTgzMzMzMzMzXSwgWzM5LjAyOTc5MzMzMzMzMzMsIC03Ny4wNzQ1NTgzMzMzMzMzXSwgWzM5LjAyOTgxODMzMzMzMzMsIC03Ny4wNzM3NzMzMzMzMzMzXSwgWzM5LjAyOTgxODMzMzMzMzMsIC03Ny4wNzM3NzMzMzMzMzMzXSwgWzM5LjAyOTgxODMzMzMzMzMsIC03Ny4wNzM3NzMzMzMzMzMzXSwgWzM5LjAyOTczNSwgLTc3LjA3NDY5MzMzMzMzMzNdLCBbMzkuMDI5NzM1LCAtNzcuMDc0NjkzMzMzMzMzM10sIFszOS4wMjk3MzUsIC03Ny4wNzQ2OTMzMzMzMzMzXSwgWzM5LjAyOTI3LCAtNzcuMDY4ODA4MzMzMzMzM10sIFszOS4wMjkyNywgLTc3LjA2ODgwODMzMzMzMzNdLCBbMzkuMDI5NDI2NjY2NjY2NywgLTc3LjA3NDg4MTY2NjY2NjddLCBbMzkuMDI5NDI2NjY2NjY2NywgLTc3LjA3NDg4MTY2NjY2NjddLCBbMzkuMDI5NDI2NjY2NjY2NywgLTc3LjA3NDg4MTY2NjY2NjddLCBbMzkuMTI3MDczMzMzMzMzMywgLTc3LjExODQyXSwgWzM5LjE4MDUzMTY2NjY2NjcsIC03Ny4yNjIzMl0sIFszOS4wMzgxNTMzMzMzMzMzLCAtNzYuOTk4OTU4MzMzMzMzM10sIFszOS4wMzgxNTMzMzMzMzMzLCAtNzYuOTk4OTU4MzMzMzMzM10sIFszOS4wMzgxNTMzMzMzMzMzLCAtNzYuOTk4OTU4MzMzMzMzM10sIFszOC45ODUxMjE2NjY2NjY3LCAtNzcuMDk0MDMzMzMzMzMzM10sIFszOS4wMDc4MiwgLTc2Ljk4NDM4XSwgWzM5LjE1ODA0LCAtNzcuMjI4MDQ2NjY2NjY2N10sIFszOS4wMDc5NzgzMzMzMzMzLCAtNzYuOTg0NDAxNjY2NjY2N10sIFszOS4wMDg2MTUsIC03Ni45ODQxOTY2NjY2NjY3XSwgWzM5LjAwODYxNSwgLTc2Ljk4NDE5NjY2NjY2NjddLCBbMzkuMDA4NzgsIC03Ni45ODQxODY2NjY2NjY3XSwgWzM5LjAwODc3MzMzMzMzMzMsIC03Ni45ODQyNDY2NjY2NjY3XSwgWzM5LjAwOTI5NjY2NjY2NjcsIC03Ni45ODQwMjE2NjY2NjY3XSwgWzM5LjAwOTE2MTY2NjY2NjcsIC03Ni45ODM5ODgzMzMzMzMzXSwgWzM5LjAwODk4NSwgLTc2Ljk4NDE4XSwgWzM5LjAwODM2MzMzMzMzMzMsIC03Ni45ODQyNzY2NjY2NjY3XSwgWzM5LjAwODEwMzMzMzMzMzMsIC03Ni45ODQzODE2NjY2NjY3XSwgWzM5LjAwNzk3NjY2NjY2NjcsIC03Ni45ODQ0MDMzMzMzMzMzXSwgWzM5LjAwNzk3NjY2NjY2NjcsIC03Ni45ODQ0MDMzMzMzMzMzXSwgWzM5LjAwNzgyLCAtNzYuOTg0MzhdLCBbMzguOTk5MDQ1LCAtNzYuOTk3ODZdLCBbMzguOTk5MDQ1LCAtNzYuOTk3ODZdLCBbMzguOTk5MDQ1LCAtNzYuOTk3ODZdLCBbMC4wLCAwLjBdLCBbMC4wLCAwLjBdLCBbMC4wLCAwLjBdLCBbMzkuMTU4MDQsIC03Ny4yMjgwNDY2NjY2NjY3XSwgWzM5LjE3OTU1NjY2NjY2NjcsIC03Ny4yNjQ4ODVdLCBbMzkuMDMzMzI4MzMzMzMzMywgLTc3LjA0ODk3MzMzMzMzMzNdLCBbMzkuMDAxMzE2NjY2NjY2NywgLTc2Ljk5NTcwMTY2NjY2NjddLCBbMzkuMTE5NDYxNjY2NjY2NywgLTc3LjAxODg4XSwgWzM4Ljk4NjQzLCAtNzcuMDk0NzEzMzMzMzMzM10sIFszOS4wMjY4ODY2NjY2NjY3LCAtNzcuMDc3MDg1XSwgWzM4Ljk4MjAyNSwgLTc3LjA5OTg4MzMzMzMzMzNdLCBbMzguOTgyMDI1LCAtNzcuMDk5ODgzMzMzMzMzM10sIFszOS4wNDY3NjE2NjY2NjY3LCAtNzcuMTEyNV0sIFszOS4yNjYzNywgLTc3LjIwMTE5XSwgWzM5LjI2NjM3LCAtNzcuMjAxMTldLCBbMzkuMTcwODE1LCAtNzcuMjc4MDk2NjY2NjY2N10sIFszOS4xNzE2OTgzMzMzMzMzLCAtNzcuMjc3MzRdLCBbMzkuMTc5NTU2NjY2NjY2NywgLTc3LjI2NDg4NV0sIFszOS4wMDI3MiwgLTc3LjAwNjM3NV0sIFszOS4wMDYzNzY2NjY2NjY3LCAtNzcuMDA3MzQ2NjY2NjY2N10sIFszOS4wMDAxOSwgLTc3LjAxMjEzXSwgWzAuMCwgMC4wXSwgWzAuMCwgMC4wXSwgWzAuMCwgMC4wXSwgWzM5LjAyNjcxODMzMzMzMzMsIC03Ny4wNDY4NjE2NjY2NjY3XSwgWzM5LjAzMzMyODMzMzMzMzMsIC03Ny4wNDg5NzMzMzMzMzMzXSwgWzM5LjE5MDU5MTY2NjY2NjcsIC03Ny4yNzMyMTMzMzMzMzMzXSwgWzM5LjE5MDU5MTY2NjY2NjcsIC03Ny4yNzMyMTMzMzMzMzMzXSwgWzM5LjE4MjgxLCAtNzcuMjc0MjNdLCBbMzkuMTgyODEsIC03Ny4yNzQyM10sIFszOS4xODI4MSwgLTc3LjI3NDIzXSwgWzM5LjAwMDczODMzMzMzMzMsIC03Ny4wMTA5N10sIFszOS4wMDQ1MywgLTc3LjAxNDgyNV0sIFszOS4wMDUwNSwgLTc3LjAyMjEzODMzMzMzMzNdLCBbMzkuMTgyMzcsIC03Ny4yNjQyNV0sIFszOS4xODIzNywgLTc3LjI2NDI1XSwgWzM5LjE4MjM3LCAtNzcuMjY0MjVdLCBbMzkuMTgyMzcsIC03Ny4yNjQyNV0sIFszOS4xODIzNywgLTc3LjI2NDI1XSwgWzM5LjE4MjM3LCAtNzcuMjY0MjVdLCBbMzkuMTgyMzcsIC03Ny4yNjQyNV0sIFszOS4wNDU0MjUsIC03Ni45OTA3MzY2NjY2NjY3XSwgWzM5LjA0NTQyNSwgLTc2Ljk5MDczNjY2NjY2NjddLCBbMzkuMDQ1NDI1LCAtNzYuOTkwNzM2NjY2NjY2N10sIFszOS4wNDU0MjUsIC03Ni45OTA3MzY2NjY2NjY3XSwgWzM5LjA0NTQyNSwgLTc2Ljk5MDczNjY2NjY2NjddLCBbMzkuMDQ1NDI1LCAtNzYuOTkwNzM2NjY2NjY2N10sIFszOS4wNDU0MjUsIC03Ni45OTA3MzY2NjY2NjY3XSwgWzM5LjA0NTQyNSwgLTc2Ljk5MDczNjY2NjY2NjddLCBbMC4wLCAwLjBdLCBbMC4wLCAwLjBdLCBbMC4wLCAwLjBdLCBbMzguOTg0MjgxNjY2NjY2NywgLTc3LjA5NDEwNjY2NjY2NjddLCBbMzguOTgxOTM2NjY2NjY2NywgLTc3LjA5OTc1MzMzMzMzMzNdLCBbMzkuMDI2NTY2NjY2NjY2NywgLTc3LjAyMDIyNV0sIFszOS4wMjY1NjY2NjY2NjY3LCAtNzcuMDIwMjI1XSwgWzM5LjAyNjU2NjY2NjY2NjcsIC03Ny4wMjAyMjVdLCBbMzkuMDAxMzE2NjY2NjY2NywgLTc2Ljk5NTcwMTY2NjY2NjddLCBbMzkuMDI0Mzk4MzMzMzMzMywgLTc3LjAxODA3NV0sIFszOS4wMTgwMzMzMzMzMzMzLCAtNzcuMDA3MzY4MzMzMzMzM10sIFszOS4wMDIyMzMzMzMzMzMzLCAtNzcuMDA0ODY1XSwgWzM5LjAwMjUyODMzMzMzMzMsIC03Ny4wMDUxMDgzMzMzMzMzXSwgWzM5LjAwMTg0LCAtNzcuMDA0Nzc4MzMzMzMzM10sIFszOC45OTQzNjE2NjY2NjY3LCAtNzYuOTkyNzQxNjY2NjY2N10sIFszOS4wMzIwNDY2NjY2NjY3LCAtNzcuMDcyNDc1XSwgWzM5LjAzMjAxLCAtNzcuMDcyNDkzMzMzMzMzM10sIFszOS4wMzE5MjY2NjY2NjY3LCAtNzcuMDcyNDQ4MzMzMzMzM10sIFszOS4wMzE4NiwgLTc3LjA3MjI3NV0sIFszOS4wMzE5ODY2NjY2NjY3LCAtNzcuMDcyNDI2NjY2NjY2N10sIFszOS4xMzA3MTMzMzMzMzMzLCAtNzcuMTg3ODk2NjY2NjY2N10sIFszOS4xMzA3MTMzMzMzMzMzLCAtNzcuMTg3ODk2NjY2NjY2N10sIFszOS4xMzI4MTgzMzMzMzMzLCAtNzcuMTg5NTNdLCBbMzkuMTMyODE4MzMzMzMzMywgLTc3LjE4OTUzXSwgWzM5LjEzMDczLCAtNzcuMTg3OTddLCBbMzkuMTMwNzMsIC03Ny4xODc5N10sIFszOS4xMzA3MywgLTc3LjE4Nzk3XSwgWzM5LjEzMzg0ODMzMzMzMzMsIC03Ny4yNDQ3NjVdLCBbMzkuMTM2MTY4MzMzMzMzMywgLTc3LjI0Mjk5NV0sIFszOS4xMzIyNjgzMzMzMzMzLCAtNzcuMTY0ODkzMzMzMzMzM10sIFszOS4xMzIyNjgzMzMzMzMzLCAtNzcuMTY0ODkzMzMzMzMzM10sIFszOS4xODk3ODMzMzMzMzMzLCAtNzcuMjQ5NzAzMzMzMzMzM10sIFszOS4wMTIzODUsIC03Ny4wODA1OTVdLCBbMzkuMTI0NjcxNjY2NjY2NywgLTc3LjIwMDQ1MzMzMzMzMzNdLCBbMzkuMTAyMTg1LCAtNzcuMTc3OTUzMzMzMzMzM10sIFszOS4xMDIxODUsIC03Ny4xNzc5NTMzMzMzMzMzXSwgWzM5LjEwMjE4NSwgLTc3LjE3Nzk1MzMzMzMzMzNdLCBbMzguOTg1NzUxNjY2NjY2NywgLTc3LjA3NzAzNV0sIFszOC45ODU3NTE2NjY2NjY3LCAtNzcuMDc3MDM1XSwgWzM4Ljk4NjY3NSwgLTc3LjA3NDc4MTY2NjY2NjddLCBbMzguOTg2Njc1LCAtNzcuMDc0NzgxNjY2NjY2N10sIFszOC45ODY2NzUsIC03Ny4wNzQ3ODE2NjY2NjY3XSwgWzM4Ljk4NTc3ODMzMzMzMzMsIC03Ny4wNzY3MzE2NjY2NjY3XSwgWzM4Ljk4NTc3ODMzMzMzMzMsIC03Ny4wNzY3MzE2NjY2NjY3XSwgWzM5LjEzNDc4MzMzMzMzMzMsIC03Ny40MDA1NzY2NjY2NjY3XSwgWzM5LjEzNzc1MzMzMzMzMzMsIC03Ny4zOTc5NzMzMzMzMzMzXSwgWzM5LjE1NjY3NjY2NjY2NjcsIC03Ny40NDczNDVdLCBbMzkuMTcxNTI4MzMzMzMzMywgLTc3LjQ2NjM2NV0sIFszOS4xNzEzNzY2NjY2NjY3LCAtNzcuNDY1NTAxNjY2NjY2N10sIFszOS4xNzEzNzY2NjY2NjY3LCAtNzcuNDY1NTAxNjY2NjY2N10sIFszOS4xMjM5NzE2NjY2NjY3LCAtNzcuMTc1NTY1XSwgWzM5LjEyNDAxNSwgLTc3LjE3NTg0NjY2NjY2NjddLCBbMzkuMTI0MDE1LCAtNzcuMTc1ODQ2NjY2NjY2N10sIFszOS4xMjQwMTUsIC03Ny4xNzU4NDY2NjY2NjY3XSwgWzM5LjE5NDcyODMzMzMzMzMsIC03Ny4xNTQxNF0sIFszOS4xOTQ4MjY2NjY2NjY3LCAtNzcuMTU0MTI2NjY2NjY2N10sIFszOS4xOTQ3OTE2NjY2NjY3LCAtNzcuMTU0MDkzMzMzMzMzM10sIFszOS4xOTQ3OTE2NjY2NjY3LCAtNzcuMTU0MDkzMzMzMzMzM10sIFszOS4xNjM3NDUsIC03Ny4yMTQ2OTMzMzMzMzMzXSwgWzM5LjE2Mzc0NSwgLTc3LjIxNDY5MzMzMzMzMzNdLCBbMzkuMTYzNzQ1LCAtNzcuMjE0NjkzMzMzMzMzM10sIFszOS4xNjM3NDUsIC03Ny4yMTQ2OTMzMzMzMzMzXSwgWzM5LjExMjgxNjY2NjY2NjcsIC03Ny4xNjIxNjVdLCBbMzkuMTU5NTE4MzMzMzMzMywgLTc3LjI3Njg0NjY2NjY2NjddLCBbMzkuMTU5NTE4MzMzMzMzMywgLTc3LjI3Njg0NjY2NjY2NjddLCBbMzkuMTIwNzgsIC03Ny4wMTU3NjE2NjY2NjY3XSwgWzM5LjE5ODA0NSwgLTc3LjI2MzA2MzMzMzMzMzNdLCBbMzkuMTc4NDk4MzMzMzMzMywgLTc3LjI0MDc1XSwgWzM5LjE4MDQ1LCAtNzcuMjYyNDAzMzMzMzMzM10sIFszOS4xODA0NSwgLTc3LjI2MjQwMzMzMzMzMzNdLCBbMzkuMDIyNjQ4MzMzMzMzMywgLTc3LjA3MTc2NjY2NjY2NjddLCBbMzkuMDMyOTgxNjY2NjY2NywgLTc3LjA3MjczNjY2NjY2NjddLCBbMzkuMDMzMDIsIC03Ny4wNzI1MzMzMzMzMzMzXSwgWzM5LjE1MzUzLCAtNzcuMjEyNzIxNjY2NjY2N10sIFszOS4xMzQ2MjE2NjY2NjY3LCAtNzcuNDAxMzU1XSwgWzM5LjEzNDYyMTY2NjY2NjcsIC03Ny40MDEzNTVdLCBbMzkuMTM0NjIxNjY2NjY2NywgLTc3LjQwMTM1NV0sIFszOS4xMzQ2MjE2NjY2NjY3LCAtNzcuNDAxMzU1XSwgWzM5LjA4NDYyLCAtNzcuMDc5OTQ2NjY2NjY2N10sIFszOC45OTE5NzUsIC03Ny4wOTU3MzgzMzMzMzMzXSwgWzM4Ljk4NjY3MTY2NjY2NjcsIC03Ny4wOTQ3MzVdLCBbMzguOTg2NDMsIC03Ny4wOTQ3MTMzMzMzMzMzXSwgWzM4Ljk4NjQzLCAtNzcuMDk0NzEzMzMzMzMzM10sIFszOC45OTA4OSwgLTc3LjA5NTUxNV0sIFszOS4wMTAwMTE2NjY2NjY3LCAtNzcuMDkxNzU2NjY2NjY2N10sIFszOS4wMTIzODUsIC03Ny4wODA1OTVdLCBbMC4wLCAwLjBdLCBbMzkuMTM2MTgxNjY2NjY2NywgLTc3LjE1MjAzXSwgWzM5LjE4MzI0ODMzMzMzMzMsIC03Ny4yNzAzNTVdLCBbMzkuMDk4NTg4MzMzMzMzMywgLTc2LjkzNjQ3NV0sIFswLjAsIDAuMF0sIFswLjAsIDAuMF0sIFswLjAsIDAuMF0sIFswLjAsIDAuMF0sIFswLjAsIDAuMF0sIFswLjAsIDAuMF0sIFswLjAsIDAuMF0sIFswLjAsIDAuMF0sIFswLjAsIDAuMF0sIFszOS4xMTQ3NjMzMzMzMzMzLCAtNzcuMTk3MzkzMzMzMzMzM10sIFszOS4xMTQzMjY2NjY2NjY3LCAtNzcuMTk1NDMzMzMzMzMzM10sIFszOS4xNTM1MywgLTc3LjIxMjcyMTY2NjY2NjddLCBbMzkuMTM0NzczMzMzMzMzMywgLTc3LjQwMDYzMzMzMzMzMzNdLCBbMzkuMTM0NzczMzMzMzMzMywgLTc3LjQwMDYzMzMzMzMzMzNdLCBbMzkuMTM0NzczMzMzMzMzMywgLTc3LjQwMDYzMzMzMzMzMzNdLCBbMzkuMTM0Nzc2NjY2NjY2NywgLTc3LjQwMDU4NV0sIFszOS4xNTY2NzY2NjY2NjY3LCAtNzcuNDQ3MzQ1XSwgWzM5LjE3MTUyMzMzMzMzMzMsIC03Ny40NjYyMl0sIFszOS4xNzE1MTY2NjY2NjY3LCAtNzcuNDY2NzUxNjY2NjY2N10sIFszOS4xNzE1MjgzMzMzMzMzLCAtNzcuNDY2MzY1XSwgWzM5LjA0MzI1MTY2NjY2NjcsIC03Ni45ODMyNzVdLCBbMzkuMDQzMjUxNjY2NjY2NywgLTc2Ljk4MzI3NV0sIFszOS4wNDMyNTE2NjY2NjY3LCAtNzYuOTgzMjc1XSwgWzM5LjA0MzI1MTY2NjY2NjcsIC03Ni45ODMyNzVdLCBbMzkuMDQzMjUxNjY2NjY2NywgLTc2Ljk4MzI3NV0sIFszOS4xMTUyMTgzMzMzMzMzLCAtNzcuMDE2Nl0sIFszOS4xMTUyMTgzMzMzMzMzLCAtNzcuMDE2Nl0sIFszOS4xMTk0NjE2NjY2NjY3LCAtNzcuMDE4ODhdLCBbMzkuMTE5NDYxNjY2NjY2NywgLTc3LjAxODg4XSwgWzM5LjE5ODA0NSwgLTc3LjI2MzA2MzMzMzMzMzNdLCBbMzkuMDg2NzIzMzMzMzMzMywgLTc3LjE2NzQzMTY2NjY2NjddLCBbMzkuMDg1ODE4MzMzMzMzMywgLTc2Ljk0NTQzODMzMzMzMzNdLCBbMzkuMDg1ODE4MzMzMzMzMywgLTc2Ljk0NTQzODMzMzMzMzNdLCBbMzkuMDg1ODE4MzMzMzMzMywgLTc2Ljk0NTQzODMzMzMzMzNdLCBbMzkuMDg1ODE4MzMzMzMzMywgLTc2Ljk0NTQzODMzMzMzMzNdLCBbMzguOTg1MDM1LCAtNzcuMDk0NDMzMzMzMzMzM10sIFszOC45ODQyODE2NjY2NjY3LCAtNzcuMDk0MTA2NjY2NjY2N10sIFszOC45ODE5MzY2NjY2NjY3LCAtNzcuMDk5NzUzMzMzMzMzM10sIFszOC45ODE5MzY2NjY2NjY3LCAtNzcuMDk5NzUzMzMzMzMzM10sIFszOC45ODIwOTE2NjY2NjY3LCAtNzcuMDk5MjYxNjY2NjY2N10sIFszOC45ODE5MzMzMzMzMzMzLCAtNzcuMTAwMTFdLCBbMzkuMDI5ODg4MzMzMzMzMywgLTc3LjA0ODc4NjY2NjY2NjddLCBbMzkuMDI5ODg4MzMzMzMzMywgLTc3LjA0ODc4NjY2NjY2NjddLCBbMzkuMTM0NjIxNjY2NjY2NywgLTc3LjQwMTM1NV0sIFszOS4xMzQ2MjE2NjY2NjY3LCAtNzcuNDAxMzU1XSwgWzM5LjEzNDYyMTY2NjY2NjcsIC03Ny40MDEzNTVdLCBbMzkuMTM0NjIxNjY2NjY2NywgLTc3LjQwMTM1NV0sIFszOS4xMzQ2MjE2NjY2NjY3LCAtNzcuNDAxMzU1XSwgWzM5LjA0MzI5NjY2NjY2NjcsIC03Ny4wNTIwNzY2NjY2NjY3XSwgWzM5LjA0MzI5NjY2NjY2NjcsIC03Ny4wNTIwNzY2NjY2NjY3XSwgWzM4Ljk5NDg1ODMzMzMzMzMsIC03Ny4wMjU3MTY2NjY2NjY3XSwgWzM5LjEzMTg0NSwgLTc3LjIwNjg0NjY2NjY2NjddLCBbMzkuMDk4NTc1LCAtNzcuMTM3NzM4MzMzMzMzM10sIFszOS4wMzEwOTgzMzMzMzMzLCAtNzcuMDc1NzI1XSwgWzM5LjAzMTI0LCAtNzcuMDc1MDI4MzMzMzMzM10sIFszOS4wMjY4ODY2NjY2NjY3LCAtNzcuMDc3MDg1XSwgWzM4Ljk4MjAyNSwgLTc3LjA5OTg4MzMzMzMzMzNdLCBbMzkuMDMyMDA1LCAtNzcuMDcyNDQxNjY2NjY2N10sIFszOS4wMzE5MjY2NjY2NjY3LCAtNzcuMDcyNDQ4MzMzMzMzM10sIFszOS4wMzE5NzY2NjY2NjY3LCAtNzcuMDcyNDM2NjY2NjY2N10sIFszOS4wMzE5NzY2NjY2NjY3LCAtNzcuMDcyNDM2NjY2NjY2N10sIFszOS4wMzE1OTUsIC03Ny4wNzE5OTE2NjY2NjY3XSwgWzM5LjAzMTk3MTY2NjY2NjcsIC03Ny4wNzIzMzMzMzMzMzMzXSwgWzM5LjAzMjA0NjY2NjY2NjcsIC03Ny4wNzI0OTgzMzMzMzMzXSwgWzM5LjAzMTk4NjY2NjY2NjcsIC03Ny4wNzI0MjY2NjY2NjY3XSwgWzM5LjA0ODc5MzMzMzMzMzMsIC03Ny4xMTk4NDY2NjY2NjY3XSwgWzM5LjA4NDY4LCAtNzcuMDgwMDIzMzMzMzMzM10sIFszOS4wODQ2NDY2NjY2NjY3LCAtNzcuMDgwMDAzMzMzMzMzM10sIFszOS4wODQ1NzgzMzMzMzMzLCAtNzcuMDc5OTA1XSwgWzM5LjA4NDY3NSwgLTc3LjA4MDAwMTY2NjY2NjddLCBbMzkuMDg0NjIsIC03Ny4wNzk5NDY2NjY2NjY3XSwgWzM5LjExOTYxLCAtNzcuMDk2OTg1XSwgWzM4Ljk5NDM2MTY2NjY2NjcsIC03Ni45OTI3NDE2NjY2NjY3XSwgWzM5LjEzNjE3ODMzMzMzMzMsIC03Ny4yNDMwOV0sIFszOC45OTY5MTMzMzMzMzMzLCAtNzYuOTkzNjRdLCBbMzguOTk2OTEzMzMzMzMzMywgLTc2Ljk5MzY0XSwgWzM5LjA0MDg4ODMzMzMzMzMsIC03Ny4wNTIxNDVdLCBbMzkuMDQwODg4MzMzMzMzMywgLTc3LjA1MjE0NV0sIFszOS4wNDA4ODgzMzMzMzMzLCAtNzcuMDUyMTQ1XSwgWzM5LjA0MDg4ODMzMzMzMzMsIC03Ny4wNTIxNDVdLCBbMzkuMTEyNjYsIC03Ni45MzI4MTMzMzMzMzMzXSwgWzM5LjE4OTc4MzMzMzMzMzMsIC03Ny4yNDk3MDMzMzMzMzMzXSwgWzM5LjEwNTM5MzMzMzMzMzMsIC03Ny4xOTc1ODE2NjY2NjY3XSwgWzM5LjE2NjQzNjY2NjY2NjcsIC03Ny4yODExODgzMzMzMzMzXSwgWzM5LjExOTAxNjY2NjY2NjcsIC03Ny4xODE5NzE2NjY2NjY3XSwgWzM5LjA5NzExLCAtNzcuMjI5NjEzMzMzMzMzM10sIFszOS4xMTA3NSwgLTc3LjI1MjQzMzMzMzMzMzNdLCBbMzkuMTEwNjM4MzMzMzMzMywgLTc3LjI1Mjc0NV0sIFszOS4xMTU1OTgzMzMzMzMzLCAtNzcuMjUyODg1XSwgWzM5LjExNTU5ODMzMzMzMzMsIC03Ny4yNTI4ODVdLCBbMzkuMTE1NzM2NjY2NjY2NywgLTc3LjI1MjkxMzMzMzMzMzNdLCBbMzkuMTE0NSwgLTc3LjI1MzM4MzMzMzMzMzNdLCBbMzkuMDk2NzY1LCAtNzcuMjY1NzMzMzMzMzMzM10sIFszOS4wOTY3NjUsIC03Ny4yNjU3MzMzMzMzMzMzXSwgWzM5LjA5NDQ2NjY2NjY2NjcsIC03Ny4yNjU3NDY2NjY2NjY3XSwgWzM5LjA5NDQ2NjY2NjY2NjcsIC03Ny4yNjU3NDY2NjY2NjY3XSwgWzM5LjE0NTgxMTY2NjY2NjcsIC03Ny4xOTIyNDVdLCBbMzkuMTMwMTgzMzMzMzMzMywgLTc3LjEyMzM3NjY2NjY2NjddLCBbMzkuMTMwMTgzMzMzMzMzMywgLTc3LjEyMzM3NjY2NjY2NjddLCBbMzkuMTIyMzUsIC03Ny4wNzM5NTMzMzMzMzMzXSwgWzM5LjE1ODQ3ODMzMzMzMzMsIC03Ny4wNjQ3NDgzMzMzMzMzXSwgWzM5LjA3NDc2ODMzMzMzMzMsIC03Ni45OTgyMzY2NjY2NjY3XSwgWzM5LjA3NDc2ODMzMzMzMzMsIC03Ni45OTgyMzY2NjY2NjY3XSwgWzM5LjA3NDc2ODMzMzMzMzMsIC03Ni45OTgyMzY2NjY2NjY3XSwgWzM5LjA3NTAyODMzMzMzMzMsIC03Ni45OTkwMl0sIFszOS4wNzg0NDE2NjY2NjY3LCAtNzcuMDcyMjczMzMzMzMzM10sIFszOS4wMTg5NiwgLTc3LjA1MDFdLCBbMzkuMTI0MTU2NjY2NjY2NywgLTc3LjIwMDAzMzMzMzMzMzNdLCBbMzkuMTgzMDk1LCAtNzcuMjU3NTg1XSwgWzM5LjE2OTMwMTY2NjY2NjcsIC03Ny4yNzkzMTE2NjY2NjY3XSwgWzM5LjE2OTMwMTY2NjY2NjcsIC03Ny4yNzkzMTE2NjY2NjY3XSwgWzM5LjE3MDA0MTY2NjY2NjcsIC03Ny4yNzg4NTMzMzMzMzMzXSwgWzM5LjE2ODg5ODMzMzMzMzMsIC03Ny4yNzk2MDgzMzMzMzMzXSwgWzM5LjE3MDU1LCAtNzcuMjc4Mzg1XSwgWzM5LjE3MDI2NSwgLTc3LjI3ODYwODMzMzMzMzNdLCBbMzkuMTAxODcsIC03Ny4yMTkzMTE2NjY2NjY3XSwgWzM5LjEwMTg3LCAtNzcuMjE5MzExNjY2NjY2N10sIFswLjAsIDAuMF0sIFswLjAsIDAuMF0sIFswLjAsIDAuMF0sIFszOS4wNzQ2MDgzMzMzMzMzLCAtNzYuOTk4MDYxNjY2NjY2N10sIFszOS4wNzQ2MDgzMzMzMzMzLCAtNzYuOTk4MjE4MzMzMzMzM10sIFszOS4wNzQ5MSwgLTc2Ljk5ODM1NV0sIFszOS4wNDE2MywgLTc3LjE0NzUzNV0sIFszOS4wNDQ0MTY2NjY2NjY3LCAtNzcuMTExOTExNjY2NjY2N10sIFszOS4xMTcwMzMzMzMzMzMzLCAtNzcuMjUxOTAzMzMzMzMzM10sIFszOS4xMTc1NzE2NjY2NjY3LCAtNzcuMjUxMjM2NjY2NjY2N10sIFszOS4xMjA5NzgzMzMzMzMzLCAtNzcuMTU4MTQxNjY2NjY2N10sIFszOS4xMjA5NzgzMzMzMzMzLCAtNzcuMTU4MTQxNjY2NjY2N10sIFszOS4xMjA5NzgzMzMzMzMzLCAtNzcuMTU4MTQxNjY2NjY2N10sIFszOS4xMjA5NzgzMzMzMzMzLCAtNzcuMTU4MTQxNjY2NjY2N10sIFszOS4xMTcwMTE2NjY2NjY3LCAtNzcuMjUyNzFdLCBbMzkuMTE4Mzg2NjY2NjY2NywgLTc3LjI1MTE2NV0sIFszOS4wNTkwMzMzMzMzMzMzLCAtNzcuMTYxMzY2NjY2NjY2N10sIFszOS4wNTkwMzMzMzMzMzMzLCAtNzcuMTYxMzY2NjY2NjY2N10sIFszOS4wNTkwMzMzMzMzMzMzLCAtNzcuMTYxMzY2NjY2NjY2N10sIFszOS4wNTkwMzMzMzMzMzMzLCAtNzcuMTYxMzY2NjY2NjY2N10sIFszOS4xODk5MTUsIC03Ny4yNDE5MTVdLCBbMzkuMDg3NDI2NjY2NjY2NywgLTc3LjE3Mzk1MTY2NjY2NjddLCBbMzkuMTE1Njc4MzMzMzMzMywgLTc3LjI1Mjk2MTY2NjY2NjddLCBbMzkuMTA4NiwgLTc3LjE4NTQyNjY2NjY2NjddLCBbMzkuMTEzNTYxNjY2NjY2NywgLTc3LjI0MjgyMzMzMzMzMzNdLCBbMzkuMDc0NzY4MzMzMzMzMywgLTc2Ljk5ODIzNjY2NjY2NjddLCBbMzkuMDc0NzY4MzMzMzMzMywgLTc2Ljk5ODIzNjY2NjY2NjddLCBbMzkuMDc0NzY4MzMzMzMzMywgLTc2Ljk5ODIzNjY2NjY2NjddLCBbMzkuMDc0NzY4MzMzMzMzMywgLTc2Ljk5ODIzNjY2NjY2NjddLCBbMzkuMDMyMjYxNjY2NjY2NywgLTc3LjA3NDI3MTY2NjY2NjddLCBbMzkuMDMzNTkxNjY2NjY2NywgLTc3LjA3Mjg4NV0sIFszOS4wMzM5NCwgLTc3LjA3MjIzNjY2NjY2NjddLCBbMzkuMDMzNjk2NjY2NjY2NywgLTc3LjA3MzAzMzMzMzMzMzNdLCBbMzkuMDMzNDkzMzMzMzMzMywgLTc3LjA3MzM1NV0sIFszOS4wMzQ1OTE2NjY2NjY3LCAtNzcuMDcxMzQ1XSwgWzM5LjAzMzkyNjY2NjY2NjcsIC03Ny4wNzIyNjE2NjY2NjY3XSwgWzM5LjEwMTg3LCAtNzcuMjE5MzExNjY2NjY2N10sIFszOS4xMDE4NywgLTc3LjIxOTMxMTY2NjY2NjddLCBbMzkuMTQ1ODExNjY2NjY2NywgLTc3LjE5MjI0NV0sIFszOS4xNDU4MTE2NjY2NjY3LCAtNzcuMTkyMjQ1XSwgWzM5LjE0NTgxMTY2NjY2NjcsIC03Ny4xOTIyNDVdLCBbMzkuMTE1ODEsIC03Ny4xODU5NTgzMzMzMzMzXSwgWzM5LjExNTgxLCAtNzcuMTg1OTU4MzMzMzMzM10sIFszOS4xMTU4MSwgLTc3LjE4NTk1ODMzMzMzMzNdLCBbMzkuMTE1ODEsIC03Ny4xODU5NTgzMzMzMzMzXSwgWzM5LjEyMTMxNjY2NjY2NjcsIC03Ny4xNjE2MTVdLCBbMzkuMTAxMTE1LCAtNzcuMDQ1MjQ2NjY2NjY2N10sIFszOS4wNzQ2MDgzMzMzMzMzLCAtNzYuOTk4MDYxNjY2NjY2N10sIFszOS4wMjQzNTE2NjY2NjY3LCAtNzcuMDQ1MDMzMzMzMzMzM10sIFszOS4wMjQzNTE2NjY2NjY3LCAtNzcuMDQ1MDMzMzMzMzMzM10sIFszOS4wMjQzNTE2NjY2NjY3LCAtNzcuMDQ1MDMzMzMzMzMzM10sIFszOS4wMjQzNTE2NjY2NjY3LCAtNzcuMDQ1MDMzMzMzMzMzM10sIFszOS4wMjQzNTE2NjY2NjY3LCAtNzcuMDQ1MDMzMzMzMzMzM10sIFszOS4wMjQzNTE2NjY2NjY3LCAtNzcuMDQ1MDMzMzMzMzMzM10sIFszOS4wNTcxMjE2NjY2NjY3LCAtNzcuMDQ5NjU2NjY2NjY2N10sIFszOS4wNTcxMjE2NjY2NjY3LCAtNzcuMDQ5NjU2NjY2NjY2N10sIFszOS4wNTcxMjE2NjY2NjY3LCAtNzcuMDQ5NjU2NjY2NjY2N10sIFszOS4wNTcxMjE2NjY2NjY3LCAtNzcuMDQ5NjU2NjY2NjY2N10sIFszOS4wNTcxMjE2NjY2NjY3LCAtNzcuMDQ5NjU2NjY2NjY2N10sIFszOS4wNTcxMjE2NjY2NjY3LCAtNzcuMDQ5NjU2NjY2NjY2N10sIFszOC45OTk5OSwgLTc3LjAyNTcwNV0sIFszOC45OTk5OSwgLTc3LjAyNTcwNV0sIFszOS4xNTc1NDgzMzMzMzMzLCAtNzcuMTg4NzJdLCBbMzkuMTU3NTg1LCAtNzcuMTg5MTIzMzMzMzMzM10sIFszOS4wNDU0MjUsIC03Ni45OTA3MzY2NjY2NjY3XSwgWzM5LjA0NTQyNSwgLTc2Ljk5MDczNjY2NjY2NjddLCBbMzkuMDQ1NDI1LCAtNzYuOTkwNzM2NjY2NjY2N10sIFszOS4wNDU0MjUsIC03Ni45OTA3MzY2NjY2NjY3XSwgWzM5LjEzOTk3LCAtNzcuMTgyNzFdLCBbMzkuMTM5OTcsIC03Ny4xODI3MV0sIFszOS4xMzk5NywgLTc3LjE4MjcxXSwgWzM5LjEzOTk3LCAtNzcuMTgyNzFdLCBbMzkuMTM5OTcsIC03Ny4xODI3MV0sIFszOS4xMzk5NywgLTc3LjE4MjcxXSwgWzM5LjEzOTk3LCAtNzcuMTgyNzFdLCBbMzkuMDMxNTE2NjY2NjY2NywgLTc3LjA0ODM3NV0sIFszOS4wNDQ0MTY2NjY2NjY3LCAtNzcuMTExOTExNjY2NjY2N10sIFszOS4wMzE1MTY2NjY2NjY3LCAtNzcuMDQ4Mzc1XSwgWzM5LjAzMTUxNjY2NjY2NjcsIC03Ny4wNDgzNzVdLCBbMzkuMDAxMTcsIC03Ni45ODU1NjVdLCBbMzkuMDAxMTcsIC03Ni45ODU1NjVdLCBbMzkuMDQxNzE1LCAtNzYuOTk0OTI2NjY2NjY2N10sIFszOS4wNDU0MjUsIC03Ni45OTA3MzY2NjY2NjY3XSwgWzM5LjA0NTQyNSwgLTc2Ljk5MDczNjY2NjY2NjddLCBbMzkuMDQ1NDI1LCAtNzYuOTkwNzM2NjY2NjY2N10sIFszOS4wNDU0MjUsIC03Ni45OTA3MzY2NjY2NjY3XSwgWzM5LjA2NzQxMTY2NjY2NjcsIC03Ni45MzcyNV0sIFszOS4wOTA4NTE2NjY2NjY3LCAtNzcuMDc5NzQ4MzMzMzMzM10sIFszOS4wMDg0NDUsIC03Ni45OTg0NTMzMzMzMzMzXSwgWzM5LjAwODQ0NSwgLTc2Ljk5ODQ1MzMzMzMzMzNdLCBbMzguOTk5OTksIC03Ny4wMjU3MDVdLCBbMzkuMjg4NTA1LCAtNzcuMjAxOV0sIFszOS4wNTY1NTUsIC03Ny4wNTAwNDVdLCBbMzguOTg3NjA4MzMzMzMzMywgLTc3LjA5ODYwNjY2NjY2NjddLCBbMzguOTkzNDI2NjY2NjY2NywgLTc3LjAyNjg5MzMzMzMzMzNdLCBbMzguOTkzNDI2NjY2NjY2NywgLTc3LjAyNjg5MzMzMzMzMzNdLCBbMzkuMTE0NDgzMzMzMzMzMywgLTc3LjI2NjQyODMzMzMzMzNdLCBbMzkuMDQ1NDI1LCAtNzYuOTkwNzM2NjY2NjY2N10sIFszOS4xMzY2NzE2NjY2NjY3LCAtNzcuMzk1MDg1XSwgWzM5LjE2NzYsIC03Ny4yODAzNTgzMzMzMzMzXSwgWzM4Ljk5NjcxNjY2NjY2NjcsIC03Ny4wMTMwMTE2NjY2NjY3XSwgWzM4Ljk5NjcxNjY2NjY2NjcsIC03Ny4wMTMwMTE2NjY2NjY3XSwgWzM5LjExNjc1ODMzMzMzMzMsIC03Ny4yNjExODMzMzMzMzMzXSwgWzM5LjExNjc1ODMzMzMzMzMsIC03Ny4yNjExODMzMzMzMzMzXSwgWzM5LjExNjc1ODMzMzMzMzMsIC03Ny4yNjExODMzMzMzMzMzXSwgWzM5LjExNjc1ODMzMzMzMzMsIC03Ny4yNjExODMzMzMzMzMzXSwgWzM5LjExNjc1ODMzMzMzMzMsIC03Ny4yNjExODMzMzMzMzMzXSwgWzM5LjExNjc1ODMzMzMzMzMsIC03Ny4yNjExODMzMzMzMzMzXSwgWzM5LjA0ODUxODMzMzMzMzMsIC03Ny4wNTY3ODMzMzMzMzMzXSwgWzM5LjE1MDkyODMzMzMzMzMsIC03Ny4yMDU1NzVdLCBbMzkuMTcwMTc2NjY2NjY2NywgLTc3LjI3ODcyODMzMzMzMzNdLCBbMzkuMDE1NjcxNjY2NjY2NywgLTc3LjA0MzAzODMzMzMzMzNdLCBbMzkuMDE1NjcxNjY2NjY2NywgLTc3LjA0MzAzODMzMzMzMzNdLCBbMzkuMDE1NjcxNjY2NjY2NywgLTc3LjA0MzAzODMzMzMzMzNdLCBbMzguOTkxODE4MzMzMzMzMywgLTc3LjAzMDgxMzMzMzMzMzNdLCBbMzguOTkxODE4MzMzMzMzMywgLTc3LjAzMDgxMzMzMzMzMzNdLCBbMzkuMTQ5NDM4MzMzMzMzMywgLTc3LjE4OTEyXSwgWzM5LjE0OTQzODMzMzMzMzMsIC03Ny4xODkxMl0sIFszOS4xNDk0MzgzMzMzMzMzLCAtNzcuMTg5MTJdLCBbMzkuMTcwMzQxNjY2NjY2NywgLTc3LjI3ODUzMzMzMzMzMzNdLCBbMzguOTgyMDUxNjY2NjY2NywgLTc3LjA5OTgxMzMzMzMzMzNdLCBbMzkuMjA1MDc1LCAtNzcuMjcxMzU1XSwgWzM4Ljk4MjA1MTY2NjY2NjcsIC03Ny4wOTk4MTMzMzMzMzMzXSwgWzM5LjE3MjE4ODMzMzMzMzMsIC03Ny4yMDM4NTVdLCBbMzkuMDIwODMsIC03Ny4wNzM0NTE2NjY2NjY3XSwgWzM5LjAyMDgzLCAtNzcuMDczNDUxNjY2NjY2N10sIFszOC45ODIwNTE2NjY2NjY3LCAtNzcuMDk5ODEzMzMzMzMzM10sIFszOS4xNTAzODgzMzMzMzMzLCAtNzcuMDYwODc2NjY2NjY2N10sIFszOS4xNTAzODgzMzMzMzMzLCAtNzcuMDYwODc2NjY2NjY2N10sIFszOS4yMTc1MDgzMzMzMzMzLCAtNzcuMjc3ODA4MzMzMzMzM10sIFszOS4xNjk4ODgzMzMzMzMzLCAtNzcuMjc5MDA2NjY2NjY2N10sIFszOS4yMTc1MDgzMzMzMzMzLCAtNzcuMjc3ODA4MzMzMzMzM10sIFszOS4xNTAzNzE2NjY2NjY3LCAtNzcuMTkwODY1XSwgWzM5LjE1MDM3MTY2NjY2NjcsIC03Ny4xOTA4NjVdLCBbMzkuMTUwMzcxNjY2NjY2NywgLTc3LjE5MDg2NV0sIFszOC45ODIwNTE2NjY2NjY3LCAtNzcuMDk5ODEzMzMzMzMzM10sIFszOC45ODIwNTE2NjY2NjY3LCAtNzcuMDk5ODEzMzMzMzMzM10sIFszOS4yMjkwOTY2NjY2NjY3LCAtNzcuMjY3MzY2NjY2NjY2N10sIFszOC45OTQ3MjY2NjY2NjY3LCAtNzcuMDI0MjAzMzMzMzMzM10sIFszOS4xMTA4MzgzMzMzMzMzLCAtNzcuMTg3ODU2NjY2NjY2N10sIFszOS4xNzAyNTE2NjY2NjY3LCAtNzcuMjc4NjA2NjY2NjY2N10sIFszOS4xMDU1MTY2NjY2NjY3LCAtNzcuMTgyMzg1XSwgWzM5LjEwNTUxNjY2NjY2NjcsIC03Ny4xODIzODVdLCBbMzkuMTAxNzkxNjY2NjY2NywgLTc3LjMwNzUyNV0sIFszOS4xNjc5NDMzMzMzMzMzLCAtNzcuMjgwMjI1XSwgWzM5LjE0ODQ5LCAtNzcuMTkyMTRdLCBbMzkuMTQ4NDksIC03Ny4xOTIxNF0sIFszOC45ODIwNTE2NjY2NjY3LCAtNzcuMDk5ODEzMzMzMzMzM10sIFszOS4wOTUzMzE2NjY2NjY3LCAtNzcuMTc2ODkxNjY2NjY2N10sIFszOS4xMDE3OTE2NjY2NjY3LCAtNzcuMzA3NTI1XSwgWzM5LjE0MzUxLCAtNzcuNDA2MjUxNjY2NjY2N10sIFszOS4wNDEzMDgzMzMzMzMzLCAtNzcuMTEzMTc4MzMzMzMzM10sIFszOS4xNTk3MDgzMzMzMzMzLCAtNzcuMjg5OTI1XSwgWzM5LjEwMTY5NjY2NjY2NjcsIC03Ny4zMDc0NTVdLCBbMzkuMTAxNjk2NjY2NjY2NywgLTc3LjMwNzQ1NV0sIFszOS4wNzcxNCwgLTc3LjE2ODM4MTY2NjY2NjddLCBbMzguOTgyMDUxNjY2NjY2NywgLTc3LjA5OTgxMzMzMzMzMzNdLCBbMzkuMTQ1ODIzMzMzMzMzMywgLTc3LjE4NDQ2MzMzMzMzMzNdLCBbMzkuMDg0NjM4MzMzMzMzMywgLTc3LjA3OTkxXSwgWzM5LjE3MDI3NSwgLTc3LjI3ODYyXSwgWzM5LjA0MTc3LCAtNzcuMTEzMjFdLCBbMzkuMDQxNzcsIC03Ny4xMTMyMV0sIFszOS4xMDE3ODY2NjY2NjY3LCAtNzcuMzA3NTI2NjY2NjY2N10sIFszOS4wNzUxNiwgLTc3LjE2MTkyMTY2NjY2NjddLCBbMzkuMDc1MTYsIC03Ny4xNjE5MjE2NjY2NjY3XSwgWzM4Ljk4MTk4NSwgLTc3LjA5OTc5MzMzMzMzMzNdLCBbMzkuMDc1MTYsIC03Ny4xNjE5MjE2NjY2NjY3XSwgWzM4Ljk4MTk4NSwgLTc3LjA5OTc5MzMzMzMzMzNdLCBbMzkuMTU5NzAxNjY2NjY2NywgLTc3LjI4OTk0XSwgWzAuMCwgMC4wXSwgWzM5LjE0NTQ0ODMzMzMzMzMsIC03Ny4xODQ0MDgzMzMzMzMzXSwgWzM5LjEwMTc4NjY2NjY2NjcsIC03Ny4zMDc1MjY2NjY2NjY3XSwgWzM5LjEwMTc4NjY2NjY2NjcsIC03Ny4zMDc1MjY2NjY2NjY3XSwgWzM5LjA3MTc3NSwgLTc3LjE2MTk0MzMzMzMzMzNdLCBbMzkuMDcxNzc1LCAtNzcuMTYxOTQzMzMzMzMzM10sIFszOS4wODQ1OCwgLTc3LjA3OTg1ODMzMzMzMzNdLCBbMzkuMDg0NTgsIC03Ny4wNzk4NTgzMzMzMzMzXSwgWzM5LjA0MTQwMzMzMzMzMzMsIC03Ny4xMTMxNl0sIFszOS4wMzY5NywgLTc3LjA1Mjk0ODMzMzMzMzNdLCBbMzkuMDQxNzE1LCAtNzYuOTk0OTI2NjY2NjY2N10sIFszOS4xMDE3MTE2NjY2NjY3LCAtNzcuMzA3NDY1XSwgWzM4Ljk5Nzc4ODMzMzMzMzMsIC03Ni45OTE2Nl0sIFszOS4wODQ2MzgzMzMzMzMzLCAtNzcuMDc5OTFdLCBbMzkuMDQxMjg1LCAtNzcuMTEzMTY1XSwgWzAuMCwgMC4wXSwgWzM5LjE1MTIyODMzMzMzMzMsIC03Ny4xODg3MjY2NjY2NjY3XSwgWzM5LjE1MTIyODMzMzMzMzMsIC03Ny4xODg3MjY2NjY2NjY3XSwgWzM5LjE1MTIyODMzMzMzMzMsIC03Ny4xODg3MjY2NjY2NjY3XSwgWzM5LjIwNjczLCAtNzcuMjQxNzM4MzMzMzMzM10sIFszOS4wNTM4ODE2NjY2NjY3LCAtNzcuMTUzNTA1XSwgWzM5LjA1Mzg4MTY2NjY2NjcsIC03Ny4xNTM1MDVdLCBbMzkuMjIwMzI2NjY2NjY2NywgLTc3LjA2MDE2NjY2NjY2NjddLCBbMzkuMDQxNDk1LCAtNzcuMDUwODQxNjY2NjY2N10sIFszOS4wNDE0OTUsIC03Ny4wNTA4NDE2NjY2NjY3XSwgWzM5LjA0MTQ5NSwgLTc3LjA1MDg0MTY2NjY2NjddLCBbMC4wLCAwLjBdLCBbMzkuMDQxNzE1LCAtNzYuOTk0OTI2NjY2NjY2N10sIFszOS4wNDE3MTUsIC03Ni45OTQ5MjY2NjY2NjY3XSwgWzM5LjExNTg1NSwgLTc2Ljk2MTc3MTY2NjY2NjddLCBbMzkuMTczNDE4MzMzMzMzMywgLTc3LjE4OTMyXSwgWzM5LjE3MzQxODMzMzMzMzMsIC03Ny4xODkzMl0sIFszOS4wNDE0MzUsIC03Ny4xMTMxMzgzMzMzMzMzXSwgWzM5LjIyMDMyNjY2NjY2NjcsIC03Ny4wNjAxNjY2NjY2NjY3XSwgWzM4Ljk5NzQ0NSwgLTc2Ljk5MjAxXSwgWzM5LjEyMzExMzMzMzMzMzMsIC03Ny4yMjk5NDgzMzMzMzMzXSwgWzM5LjEyMzExMzMzMzMzMzMsIC03Ny4yMjk5NDgzMzMzMzMzXSwgWzM5LjEyMzExMzMzMzMzMzMsIC03Ny4yMjk5NDgzMzMzMzMzXSwgWzM5LjEyMzExMzMzMzMzMzMsIC03Ny4yMjk5NDgzMzMzMzMzXSwgWzM5LjEyMzExMzMzMzMzMzMsIC03Ny4yMjk5NDgzMzMzMzMzXSwgWzM5LjEyMzExMzMzMzMzMzMsIC03Ny4yMjk5NDgzMzMzMzMzXSwgWzAuMCwgMC4wXSwgWzM5LjAyMTQ5ODMzMzMzMzMsIC03Ny4wMTQ0NDE2NjY2NjY3XSwgWzM5LjA0NzQyMTY2NjY2NjcsIC03Ny4xNTA1MzgzMzMzMzMzXSwgWzM4Ljk5NzUwMTY2NjY2NjcsIC03Ni45ODc4OTE2NjY2NjY3XSwgWzM5LjEyMTc3LCAtNzcuMTU5NTg1XSwgWzM5LjA1MDk4NSwgLTc3LjEwOTUwODMzMzMzMzNdLCBbMzkuMDA3MDI4MzMzMzMzMywgLTc3LjAyMTE1NjY2NjY2NjddLCBbMzkuMDA3MDI4MzMzMzMzMywgLTc3LjAyMTE1NjY2NjY2NjddLCBbMC4wLCAwLjBdLCBbMC4wLCAwLjBdLCBbMC4wLCAwLjBdLCBbMzkuMjIwMzI2NjY2NjY2NywgLTc3LjA2MDE2NjY2NjY2NjddLCBbMC4wLCAwLjBdLCBbMzkuMTUwNzM2NjY2NjY2NywgLTc3LjE5MTQxNV0sIFszOS4xNTA3MzY2NjY2NjY3LCAtNzcuMTkxNDE1XSwgWzM5LjE2ODcyMTY2NjY2NjcsIC03Ny4yNzk3MDgzMzMzMzMzXSwgWzM5LjE2ODcyMTY2NjY2NjcsIC03Ny4yNzk3MDgzMzMzMzMzXSwgWzM5LjExNTc3MTY2NjY2NjcsIC03Ni45NjE1ODY2NjY2NjY3XSwgWzM5LjEwMTY2ODMzMzMzMzMsIC03Ny4zMDc0NTVdLCBbMzkuMTAxNjY4MzMzMzMzMywgLTc3LjMwNzQ1NV0sIFszOS4wMzc4MzE2NjY2NjY3LCAtNzcuMDUzNDExNjY2NjY2N10sIFszOS4wMzc4MzE2NjY2NjY3LCAtNzcuMDUzNDExNjY2NjY2N10sIFszOS4xMDk5MDUsIC03Ny4xNTM5NV0sIFszOC45OTY3NTMzMzMzMzMzLCAtNzYuOTkxNTA1XSwgWzM5LjAzMTAxLCAtNzcuMDA0OTgxNjY2NjY2N10sIFszOS4wMzEwMSwgLTc3LjAwNDk4MTY2NjY2NjddLCBbMzkuMDM1NjcxNjY2NjY2NywgLTc3LjEyNDIzNV0sIFszOS4wMzU2NzE2NjY2NjY3LCAtNzcuMTI0MjM1XSwgWzM5LjIyMDMyNjY2NjY2NjcsIC03Ny4wNjAxNjY2NjY2NjY3XSwgWzM5LjA0NTQyNSwgLTc2Ljk5MDczNjY2NjY2NjddLCBbMzkuMDQ1NDI1LCAtNzYuOTkwNzM2NjY2NjY2N10sIFszOS4yNjYxLCAtNzcuMjAxMDY1XSwgWzM4Ljk5MDYyLCAtNzcuMDAxODFdLCBbMzguOTkwNjIsIC03Ny4wMDE4MV0sIFszOS4xMTU4NTMzMzMzMzMzLCAtNzcuMDc1NDQxNjY2NjY2N10sIFszOS4xMTU4NTMzMzMzMzMzLCAtNzcuMDc1NDQxNjY2NjY2N10sIFszOS4wNzgzMTE2NjY2NjY3LCAtNzcuMTI5MDQ1XSwgWzM5LjA3ODMxMTY2NjY2NjcsIC03Ny4xMjkwNDVdLCBbMzkuMDM5NjcxNjY2NjY2NywgLTc3LjA1NDU1MTY2NjY2NjddLCBbMzkuMjI1ODY1LCAtNzcuMjYzNDU1XSwgWzM5LjIyNTg2NSwgLTc3LjI2MzQ1NV0sIFszOS4yMjAzMjY2NjY2NjY3LCAtNzcuMDYwMTY2NjY2NjY2N10sIFszOS4yMjAzMjY2NjY2NjY3LCAtNzcuMDYwMTY2NjY2NjY2N10sIFszOS4xNjI0MDY2NjY2NjY3LCAtNzcuMjUwNjEzMzMzMzMzM10sIFszOS4xNjI0MDY2NjY2NjY3LCAtNzcuMjUwNjEzMzMzMzMzM10sIFszOS4wMzEwMzgzMzMzMzMzLCAtNzcuMDA0ODRdLCBbMzguOTk3MjgxNjY2NjY2NywgLTc2Ljk5NDAwODMzMzMzMzNdLCBbMzguOTk3MjgxNjY2NjY2NywgLTc2Ljk5NDAwODMzMzMzMzNdLCBbMzkuMTUwMzc4MzMzMzMzMywgLTc3LjE5MDk1XSwgWzM5LjE1MDM3ODMzMzMzMzMsIC03Ny4xOTA5NV0sIFszOS4xNTAzNzgzMzMzMzMzLCAtNzcuMTkwOTVdLCBbMzkuMDc3NzAxNjY2NjY2NywgLTc3LjEyNTk3NjY2NjY2NjddLCBbMzkuMDc3NzAxNjY2NjY2NywgLTc3LjEyNTk3NjY2NjY2NjddLCBbMzkuMDQ5NDg1LCAtNzcuMDY4MDk1XSwgWzM5LjA0OTQ4NSwgLTc3LjA2ODA5NV0sIFszOS4wNDk0ODUsIC03Ny4wNjgwOTVdLCBbMzkuMDQ5NDg1LCAtNzcuMDY4MDk1XSwgWzM5LjA0OTQ4NSwgLTc3LjA2ODA5NV0sIFszOC45OTY3NSwgLTc2Ljk5MTcyNV0sIFszOC45OTcxOCwgLTc3LjAyNjAyMzMzMzMzMzNdLCBbMzguOTk3MTgsIC03Ny4wMjYwMjMzMzMzMzMzXSwgWzAuMCwgMC4wXSwgWzAuMCwgMC4wXSwgWzAuMCwgMC4wXSwgWzM5LjA5NjY5ODMzMzMzMzMsIC03Ny4yNjU2NDVdLCBbMzkuMDk2Njk4MzMzMzMzMywgLTc3LjI2NTY0NV0sIFszOS4wOTc1MSwgLTc2Ljk0ODE2ODMzMzMzMzNdLCBbMzkuMjIwMjY2NjY2NjY2NywgLTc3LjA2MDE5MzMzMzMzMzNdLCBbMzguOTk5NjQsIC03Ni45OTMyNjVdLCBbMzguOTk5NjQsIC03Ni45OTMyNjVdLCBbMzkuMDI0NzIzMzMzMzMzMywgLTc3LjAxMTIzMzMzMzMzMzNdLCBbMzguOTk5ODAxNjY2NjY2NywgLTc2Ljk5MjMyNjY2NjY2NjddLCBbMzkuMTA3NzU1LCAtNzYuOTk4OTk4MzMzMzMzM10sIFszOS4wOTAwOCwgLTc3LjA1MzYzODMzMzMzMzNdLCBbMzkuMTA3NzU1LCAtNzYuOTk4OTk4MzMzMzMzM10sIFszOS4xMDc3NTUsIC03Ni45OTg5OTgzMzMzMzMzXSwgWzM5LjEwNzc1NSwgLTc2Ljk5ODk5ODMzMzMzMzNdLCBbMzkuMTA3NzU1LCAtNzYuOTk4OTk4MzMzMzMzM10sIFszOS4xNTA0MTE2NjY2NjY3LCAtNzcuMTkwOTZdLCBbMzkuMTI4NDk1LCAtNzcuMTg2NDVdLCBbMzkuMTI4NDk1LCAtNzcuMTg2NDVdLCBbMzkuMTI4NDk1LCAtNzcuMTg2NDVdLCBbMzkuMDY3NjcxNjY2NjY2NywgLTc3LjAwMTY0ODMzMzMzMzNdLCBbMzkuMDkwMTMsIC03Ny4wNTM3Nl0sIFszOS4wOTAxMywgLTc3LjA1Mzc2XSwgWzM5LjA5MDEzLCAtNzcuMDUzNzZdLCBbMzkuMDc1MzAxNjY2NjY2NywgLTc3LjExNTE4NjY2NjY2NjddLCBbMzkuMTM2NjcxNjY2NjY2NywgLTc3LjM5NTA4NV0sIFszOS4wNDM3MSwgLTc3LjA2MjQ2MTY2NjY2NjddLCBbMC4wLCAwLjBdLCBbMzkuMTM2NjcxNjY2NjY2NywgLTc3LjM5NTA4NV0sIFszOS4wNzQ0ODgzMzMzMzMzLCAtNzcuMTY1MjY2NjY2NjY2N10sIFszOS4yNjYxLCAtNzcuMjAxMDY1XSwgWzM5LjAwOTQ3NSwgLTc2Ljk5OTQwMzMzMzMzMzNdLCBbMzkuMDA5NDc1LCAtNzYuOTk5NDAzMzMzMzMzM10sIFszOS4wNzU0NzUsIC03Ny4xMTUwNzMzMzMzMzMzXSwgWzM5LjA3NTQ3NSwgLTc3LjExNTA3MzMzMzMzMzNdLCBbMzkuMDc1NDc1LCAtNzcuMTE1MDczMzMzMzMzM10sIFszOS4wNzU0NzUsIC03Ny4xMTUwNzMzMzMzMzMzXSwgWzM5LjA3NTQ3NSwgLTc3LjExNTA3MzMzMzMzMzNdLCBbMzkuMDkxMjIzMzMzMzMzMywgLTc3LjIwNDgwODMzMzMzMzNdLCBbMzkuMDc1NzM1LCAtNzcuMDAxODc4MzMzMzMzM10sIFszOS4wNzU3MzUsIC03Ny4wMDE4NzgzMzMzMzMzXSwgWzM5LjA3NTczNSwgLTc3LjAwMTg3ODMzMzMzMzNdLCBbMzkuMDc1NzM1LCAtNzcuMDAxODc4MzMzMzMzM10sIFszOS4wNzU3MzUsIC03Ny4wMDE4NzgzMzMzMzMzXSwgWzM5LjA0NTQyNSwgLTc2Ljk5MDczNjY2NjY2NjddLCBbMzguOTk0NDAzMzMzMzMzMywgLTc3LjAyODMzMzMzMzMzMzNdLCBbMzkuMDMyMjQzMzMzMzMzMywgLTc3LjA3NTA0NV0sIFszOS4wNTI1MzgzMzMzMzMzLCAtNzcuMTY2MzU4MzMzMzMzM10sIFszOS4wMzIyNDMzMzMzMzMzLCAtNzcuMDc1MDQ1XSwgWzM5LjA2NjEzMzMzMzMzMzMsIC03Ny4wMDA1MzMzMzMzMzMzXSwgWzM5LjI2NjEsIC03Ny4yMDEwNjVdLCBbMC4wLCAwLjBdLCBbMC4wLCAwLjBdLCBbMzkuMTIxMTIsIC03Ny4wMTE1MjY2NjY2NjY3XSwgWzAuMCwgMC4wXSwgWzAuMCwgMC4wXSwgWzAuMCwgMC4wXSwgWzAuMCwgMC4wXSwgWzM5LjAzMTEwNSwgLTc3LjAwNDkzNjY2NjY2NjddLCBbMzkuMTg0NjMsIC03Ny4yMTA5OTMzMzMzMzMzXSwgWzM5LjE4NDYzLCAtNzcuMjEwOTkzMzMzMzMzM10sIFszOS4wODM4MDY2NjY2NjY3LCAtNzYuOTUwMTczMzMzMzMzM10sIFszOS4xODczNTY2NjY2NjY3LCAtNzcuMjU3MzRdLCBbMzkuMTg3MzU2NjY2NjY2NywgLTc3LjI1NzM0XSwgWzM5LjE4NzM1NjY2NjY2NjcsIC03Ny4yNTczNF0sIFszOS4xODczNTY2NjY2NjY3LCAtNzcuMjU3MzRdLCBbMzkuMTg3MzU2NjY2NjY2NywgLTc3LjI1NzM0XSwgWzM5LjE4NzM1NjY2NjY2NjcsIC03Ny4yNTczNF0sIFszOC45OTkzNjMzMzMzMzMzLCAtNzcuMDE3Nzg1XSwgWzM4Ljk5MDcsIC03Ny4wMjE5MTVdLCBbMzkuMTQ5Mjc4MzMzMzMzMywgLTc3LjA2NjYyXSwgWzM5LjE0OTI3ODMzMzMzMzMsIC03Ny4wNjY2Ml0sIFszOS4xNDkyNzgzMzMzMzMzLCAtNzcuMDY2NjJdLCBbMzkuMTQ5Mjc4MzMzMzMzMywgLTc3LjA2NjYyXSwgWzM5LjE0OTI3ODMzMzMzMzMsIC03Ny4wNjY2Ml0sIFszOS4xNDkyNzgzMzMzMzMzLCAtNzcuMDY2NjJdLCBbMzkuMTYwMTk1LCAtNzcuMjg5NDcxNjY2NjY2N10sIFszOS4xNDgyODMzMzMzMzMzLCAtNzcuMjc2OTc2NjY2NjY2N10sIFszOS4wODk5NTMzMzMzMzMzLCAtNzcuMDI3NzZdLCBbMzkuMDg5OTUzMzMzMzMzMywgLTc3LjAyNzc2XSwgWzAuMCwgMC4wXSwgWzM5LjE0ODU4MTY2NjY2NjcsIC03Ny4yNzkwOV0sIFszOC45OTY3ODY2NjY2NjY3LCAtNzcuMDI3NjIxNjY2NjY2N10sIFszOC45ODIzMTE2NjY2NjY3LCAtNzcuMDc1MzI2NjY2NjY2N10sIFszOC45OTc5NTgzMzMzMzMzLCAtNzcuMDI2NzU2NjY2NjY2N10sIFszOC45OTc5NTgzMzMzMzMzLCAtNzcuMDI2NzU2NjY2NjY2N10sIFszOS4xNTk3MDE2NjY2NjY3LCAtNzcuMjg5OTczMzMzMzMzM10sIFszOS4xNDg1MzE2NjY2NjY3LCAtNzcuMjc2NzczMzMzMzMzM10sIFszOS4wNjcwOTMzMzMzMzMzLCAtNzcuMDAxMzY4MzMzMzMzM10sIFszOS4xNTk3NjY2NjY2NjY3LCAtNzcuMjg5OTE4MzMzMzMzM10sIFszOS4xNDgyNjUsIC03Ny4yNzc0XSwgWzM5LjAxMjYwNjY2NjY2NjcsIC03Ny4xMDc5NTY2NjY2NjY3XSwgWzM5LjIyMDM0LCAtNzcuMDYwMTZdLCBbMC4wLCAwLjBdLCBbMzkuMTU5ODAzMzMzMzMzMywgLTc3LjI4OTgyMzMzMzMzMzNdLCBbMzkuMTEzODYsIC03Ny4xOTYzMzY2NjY2NjY3XSwgWzM5LjE0ODM3MTY2NjY2NjcsIC03Ny4yNzY5MV0sIFszOC45ODIxNjMzMzMzMzMzLCAtNzcuMDc2NDMzMzMzMzMzM10sIFszOC45ODIxNjMzMzMzMzMzLCAtNzcuMDc2NDMzMzMzMzMzM10sIFszOS4wNjY3MDgzMzMzMzMzLCAtNzcuMDAxMDgxNjY2NjY2N10sIFszOS4xMjM5NDUsIC03Ny4xNzU1NjY2NjY2NjY3XSwgWzM5LjEyMzk0NSwgLTc3LjE3NTU2NjY2NjY2NjddLCBbMzkuMjIwMzQsIC03Ny4wNjAxNl0sIFszOS4wMDIxNDE2NjY2NjY3LCAtNzcuMDM5MDU2NjY2NjY2N10sIFswLjAsIDAuMF0sIFszOS4xMjM5NDUsIC03Ny4xNzU1NjY2NjY2NjY3XSwgWzM5LjE1OTU5NjY2NjY2NjcsIC03Ny4yODk4OTE2NjY2NjY3XSwgWzM5LjExMzAzLCAtNzcuMTkyNjYxNjY2NjY2N10sIFszOS4wMTI3MjE2NjY2NjY3LCAtNzcuMTA3OTk1XSwgWzM5LjEyMzY2MTY2NjY2NjcsIC03Ny4xNzUxM10sIFszOS4xMjM2NjE2NjY2NjY3LCAtNzcuMTc1MTNdLCBbMzkuMjIwMzQsIC03Ny4wNjAxNl0sIFszOS4wNjEwNDUsIC03Ni45OTg3MzMzMzMzMzMzXSwgWzM4Ljk5MDE5MzMzMzMzMzMsIC03Ny4wOTUzNTMzMzMzMzMzXSwgWzAuMCwgMC4wXSwgWzM4Ljk5MDA2MzMzMzMzMzMsIC03Ny4wOTUxMTVdLCBbMzkuMDY5NDUxNjY2NjY2NywgLTc3LjE2OTQ5MzMzMzMzMzNdLCBbMzkuMDEzMTA4MzMzMzMzMywgLTc3LjEwNzYzODMzMzMzMzNdLCBbMzkuMjIwMzQsIC03Ny4wNjAxNl0sIFszOS4yMjAzNCwgLTc3LjA2MDE2XSwgWzM5LjA1MjQ1ODMzMzMzMzMsIC03Ny4xNjU3NjVdLCBbMzkuMTU5Njk4MzMzMzMzMywgLTc3LjI4OTk0XSwgWzAuMCwgMC4wXSwgWzM5LjA1MjQ1NjY2NjY2NjcsIC03Ny4xNjU4MzMzMzMzMzMzXSwgWzM5LjAxMjA1NSwgLTc3LjEwODMzNjY2NjY2NjddLCBbMzkuMTI2MDc1LCAtNzcuMTczOTVdLCBbMzkuMTI2MDc1LCAtNzcuMTczOTVdLCBbMzkuMTU5Njk4MzMzMzMzMywgLTc3LjI4OTkxXSwgWzM5LjAxMTQ3ODMzMzMzMzMsIC03Ni45Nzk1NDE2NjY2NjY3XSwgWzM5LjE1MTg1ODMzMzMzMzMsIC03Ny4yODM1XSwgWzAuMCwgMC4wXSwgWzM5LjAwMDE0ODMzMzMzMzMsIC03Ni45OTA4NF0sIFszOS4wMDAxNDgzMzMzMzMzLCAtNzYuOTkwODRdLCBbMzguOTk0NjY4MzMzMzMzMywgLTc3LjAyNzYyNjY2NjY2NjddLCBbMzkuMTE5MTY4MzMzMzMzMywgLTc3LjAxOTg1XSwgWzM5LjExNjA1MzMzMzMzMzMsIC03Ny4wMTc3NjE2NjY2NjY3XSwgWzAuMCwgMC4wXSwgWzM5LjE0OTI3ODMzMzMzMzMsIC03Ny4wNjY2Ml0sIFszOS4xMTkwMywgLTc3LjAyMDM0MzMzMzMzMzNdLCBbMzkuMDg3MjcxNjY2NjY2NywgLTc3LjIxMTU1NjY2NjY2NjddLCBbMzkuMDg4Nzk4MzMzMzMzMywgLTc3LjIxMDI1MTY2NjY2NjddLCBbMC4wLCAwLjBdLCBbMzguOTY4NjU4MzMzMzMzMywgLTc3LjEwOTEyNV0sIFszOS4xMjE3MzE2NjY2NjY3LCAtNzYuOTkyNTldLCBbMzkuMTE5MDE2NjY2NjY2NywgLTc3LjAyMDM2NjY2NjY2NjddLCBbMzkuMTIxNzMxNjY2NjY2NywgLTc2Ljk5MjU5XSwgWzM5LjE0OTI3ODMzMzMzMzMsIC03Ny4wNjY2Ml0sIFszOC45OTk1LCAtNzcuMDk3NzkxNjY2NjY2N10sIFszOS4xMDYzNSwgLTc3LjE4OTUyMzMzMzMzMzNdLCBbMzkuMTI4MTM1LCAtNzcuMTYxMzhdLCBbMzkuMTE4ODg1LCAtNzcuMDIwOTddLCBbMzkuMDE1NzcsIC03Ny4wNDMyODY2NjY2NjY3XSwgWzM5LjAyODA5MTY2NjY2NjcsIC03Ny4wMTMzODVdLCBbMzkuMTI4MTM1LCAtNzcuMTYxMzhdLCBbMC4wLCAwLjBdLCBbMC4wLCAwLjBdLCBbMC4wLCAwLjBdLCBbMC4wLCAwLjBdLCBbMzkuMDg2Mzc4MzMzMzMzMywgLTc2Ljk5OTUzMTY2NjY2NjddLCBbMzkuMTE1ODQsIC03Ny4wNzM2NV0sIFswLjAsIDAuMF0sIFszOS4xMjgxNjUsIC03Ny4xNjEyNzgzMzMzMzMzXSwgWzM5LjE1MTc1MTY2NjY2NjcsIC03Ny4yMTEwNTY2NjY2NjY3XSwgWzM5LjE1MTc1MTY2NjY2NjcsIC03Ny4yMTEwNTY2NjY2NjY3XSwgWzM5LjAxNDk0ODMzMzMzMzMsIC03Ny4wOTk2NDY2NjY2NjY3XSwgWzM5LjI0MTU4LCAtNzcuMTQzNjMzMzMzMzMzM10sIFszOS4wMzQzMzMzMzMzMzMzLCAtNzcuMDcxNjU4MzMzMzMzM10sIFszOS4wMTM3OTgzMzMzMzMzLCAtNzcuMDk5MzJdLCBbMzkuMDMyMjEsIC03Ny4wNzQzNzE2NjY2NjY3XSwgWzM5LjA5OTQ0MTY2NjY2NjcsIC03Ny4yMTIxNl0sIFszOS4wOTk0NDE2NjY2NjY3LCAtNzcuMjEyMTZdLCBbMzguOTg1NTkxNjY2NjY2NywgLTc3LjA5MzcyNV0sIFszOS4wOTk0NDE2NjY2NjY3LCAtNzcuMjEyMTZdLCBbMzguOTg0MzQ1LCAtNzcuMDk0MjkxNjY2NjY2N10sIFszOS4wNTI1NDUsIC03Ni45NzY5MDMzMzMzMzMzXSwgWzM5LjA1MjU0NSwgLTc2Ljk3NjkwMzMzMzMzMzNdLCBbMzkuMDUyNTQ1LCAtNzYuOTc2OTAzMzMzMzMzM10sIFszOS4wMTg1NzgzMzMzMzMzLCAtNzcuMDE0NzEzMzMzMzMzM10sIFszOS4wNDU0MjUsIC03Ni45OTA3MzY2NjY2NjY3XSwgWzAuMCwgMC4wXSwgWzM5LjAyMDQzODMzMzMzMzMsIC03Ny4xMDMxMzMzMzMzMzMzXSwgWzM5LjA0NTQyNSwgLTc2Ljk5MDczNjY2NjY2NjddLCBbMzkuMDg5NzkxNjY2NjY2NywgLTc3LjEzMDI5MTY2NjY2NjddLCBbMzkuMDg5NzkxNjY2NjY2NywgLTc3LjEzMDI5MTY2NjY2NjddLCBbMzkuMDQ1NDI1LCAtNzYuOTkwNzM2NjY2NjY2N10sIFszOC45OTE3NjMzMzMzMzMzLCAtNzcuMDk1NzI4MzMzMzMzM10sIFszOS4wNDU0MjUsIC03Ni45OTA3MzY2NjY2NjY3XSwgWzM4Ljk5MTE2MTY2NjY2NjcsIC03Ny4wOTQwMzE2NjY2NjY3XSwgWzM4Ljk5NDMyLCAtNzcuMDI4MDFdLCBbMzkuMTQwMDcsIC03Ny4xNTIxMzE2NjY2NjY3XSwgWzM5LjEyOTMxNjY2NjY2NjcsIC03Ny4yMzI0MTgzMzMzMzMzXSwgWzM5LjEyOTMxNjY2NjY2NjcsIC03Ny4yMzI0MTgzMzMzMzMzXSwgWzM5LjE5NzM3ODMzMzMzMzMsIC03Ny4yODI0MjVdLCBbMzkuMTk3Mzc4MzMzMzMzMywgLTc3LjI4MjQyNV0sIFszOS4xOTczNzgzMzMzMzMzLCAtNzcuMjgyNDI1XSwgWzM5LjEwOTU5NjY2NjY2NjcsIC03Ny4wNzc0NzY2NjY2NjY3XSwgWzM5LjAxMDE5NjY2NjY2NjcsIC03Ny4wMDA2NTVdLCBbMzguOTkwMywgLTc3LjA5NTM2XSwgWzM4Ljk5MDM2ODMzMzMzMzMsIC03Ny4wMDE4MjE2NjY2NjY3XSwgWzM4Ljk5MDM2ODMzMzMzMzMsIC03Ny4wMDE4MjE2NjY2NjY3XSwgWzM5LjAwMDUwMzMzMzMzMzMsIC03Ni45ODk4OTVdLCBbMzkuMjYwMzMxNjY2NjY2NywgLTc3LjIyMzExMTY2NjY2NjddLCBbMzkuMjYwMzMxNjY2NjY2NywgLTc3LjIyMzExMTY2NjY2NjddLCBbMzguOTkwMzIxNjY2NjY2NywgLTc3LjA5NTI1ODMzMzMzMzNdLCBbMzkuMjY2MDA4MzMzMzMzMywgLTc3LjIwMTQ3NjY2NjY2NjddXSwKICAgICAgICAgICAgICAgIHsKICAgICAgICAgICAgICAgICAgICBtaW5PcGFjaXR5OiAwLjUsCiAgICAgICAgICAgICAgICAgICAgbWF4Wm9vbTogMTgsCiAgICAgICAgICAgICAgICAgICAgbWF4OiAxLjAsCiAgICAgICAgICAgICAgICAgICAgcmFkaXVzOiAyNSwKICAgICAgICAgICAgICAgICAgICBibHVyOiAxNSwKICAgICAgICAgICAgICAgICAgICBncmFkaWVudDogbnVsbAogICAgICAgICAgICAgICAgICAgIH0pCiAgICAgICAgICAgICAgICAuYWRkVG8obWFwX2EzMmQ2NzVlN2Y5MDQ5YTlhMTQzOTk4MjkwMzE0YTk4KTsKICAgICAgICAKICAgICAgICAKICAgICAgICAKICAgICAgICA8L3NjcmlwdD4KICAgICAgICA=)

后续步骤
在这篇关注学习 python 编程的文章中，我们学习了如何使用 Python 从原始 JSON 数据到使用命令行工具、ijson、Pandas、matplotlib 和 folium 的全功能地图。如果你想了解关于这些工具的更多信息，请查看我们在 [Dataquest](https://www.dataquest.io) 上的[数据分析](https://www.dataquest.io/subject/data-analysis)、[数据可视化](https://www.dataquest.io/subject/data-visualization)和[命令行](https://www.dataquest.io/subject/the-command-line)课程。
如果您想进一步探索这个数据集，这里有一些有趣的问题需要回答:

*   车站的类型因地点而异吗？
*   收入和停留次数有什么关系？
*   人口密度如何与停靠点数量相关联？
*   午夜前后什么类型的停车最常见？

```py

## 后续步骤

在这篇关注学习 python 编程的文章中，我们学习了如何使用 Python 从原始 JSON 数据到使用命令行工具、ijson、Pandas、matplotlib 和 folium 的全功能地图。如果你想了解关于这些工具的更多信息，请查看我们在 [Dataquest](https://www.dataquest.io) 上的[数据分析](https://www.dataquest.io/subject/data-analysis)、[数据可视化](https://www.dataquest.io/subject/data-visualization)和[命令行](https://www.dataquest.io/subject/the-command-line)课程。

如果您想进一步探索这个数据集，这里有一些有趣的问题需要回答:

*   车站的类型因地点而异吗？

*   收入和停留次数有什么关系？

*   人口密度如何与停靠点数量相关联？

*   午夜前后什么类型的停车最常见？

## 这个教程有帮助吗？

选择你的道路，不断学习有价值的数据技能。

    ![arrow down left](img/2215dd1efd21629477b52ea871afdd98.png)   ![arrow right down](img/2e703f405f987a154317ac045ee00a68.png)      [    Python Tutorials ](/python-tutorials-for-data-science/) 

在我们的免费教程中练习 Python 编程技能。

   [    Data science courses ](/data-science-courses/) 

通过我们的交互式浏览器数据科学课程，投入到 Python、R、SQL 等语言的学习中。

```