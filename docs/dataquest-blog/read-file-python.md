# 教程:如何轻松读取 Python 中的文件(文本、CSV、JSON)

> 原文：<https://www.dataquest.io/blog/read-file-python/>

April 18, 2022![Read Files in Python](img/3bb28cce9869f6a1a6b7158a5ac6ddfd.png)

## 用 Python 读取文件

文件无处不在:在计算机上、移动设备上，以及在云中。无论您使用哪种编程语言，处理文件对每个程序员来说都是必不可少的。

文件处理是一种创建文件、写入数据和从中读取数据的机制。好消息是 Python 增加了处理不同文件类型的包。

在本教程中，我们将学习如何处理不同类型的文件。然而，我们将更加关注用 Python 读取文件的[。](http://https://app.dataquest.io/c/79/m/452/reading-and-writing-to-files/1/reading-files-in-python "reading files with Python")

完成本教程后，您将知道如何执行以下操作:

*   打开文件并使用`with`上下文管理器
*   Python 中的文件模式
*   阅读文本
*   读取 CSV 文件
*   读取 JSON 文件

让我们开始吧。

## 打开文件

在访问文件内容之前，我们需要打开文件。Python 提供了一个内置函数，可以帮助我们以不同的模式打开文件。`open()`函数接受两个基本参数:文件名和模式；默认模式是`'r'`，以只读方式打开文件。这些模式定义了我们如何访问一个文件以及如何操作它的内容。`open()`函数提供了一些不同的模式，我们将在本教程的后面讨论。

首先，让我们通过打开一个文本文件来尝试这个功能。下载包含 Python 之禅的文本文件，并将其存储在与您的代码相同的路径中。

```py
f = open('zen_of_python.txt', 'r')
print(f.read())
f.close()
```

```py
 The Zen of Python, by Tim Peters

    Beautiful is better than ugly.
    Explicit is better than implicit.
    Simple is better than complex.
    Complex is better than complicated.
    Flat is better than nested.
    Sparse is better than dense.
    Readability counts.
    Special cases aren't special enough to break the rules.
    Although practicality beats purity.
    Errors should never pass silently.
    Unless explicitly silenced.
    In the face of ambiguity, refuse the temptation to guess.
    There should be one-- and preferably only one --obvious way to do it.
    Although that way may not be obvious at first unless you're Dutch.
    Now is better than never.
    Although never is often better than *right* now.
    If the implementation is hard to explain, it's a bad idea.
    If the implementation is easy to explain, it may be a good idea.
    Namespaces are one honking great idea -- let's do more of those!
```

在上面的代码中，`open()`函数以阅读模式打开文本文件，允许我们从文件中获取信息，而不会意外地更改它。在第一行中，`open()`函数的输出被赋给了`f`变量，一个表示文本文件的对象。在上面代码的第二行，我们使用`read()`方法读取整个文件并打印其内容。`close()`方法关闭文件的最后一行。我们必须在使用完打开的文件后关闭它们，以释放我们的计算机资源并避免引发异常。

在 Python 中，我们可以使用`with`上下文管理器来确保程序在文件关闭后释放所使用的资源，即使发生了异常。让我们来试试:

```py
with open('zen_of_python.txt') as f:
    print(f.read())
```

```py
 The Zen of Python, by Tim Peters

    Beautiful is better than ugly.
    Explicit is better than implicit.
    Simple is better than complex.
    Complex is better than complicated.
    Flat is better than nested.
    Sparse is better than dense.
    Readability counts.
    Special cases aren't special enough to break the rules.
    Although practicality beats purity.
    Errors should never pass silently.
    Unless explicitly silenced.
    In the face of ambiguity, refuse the temptation to guess.
    There should be one-- and preferably only one --obvious way to do it.
    Although that way may not be obvious at first unless you're Dutch.
    Now is better than never.
    Although never is often better than *right* now.
    If the implementation is hard to explain, it's a bad idea.
    If the implementation is easy to explain, it may be a good idea.
    Namespaces are one honking great idea -- let's do more of those!
```

上面的代码使用`with`语句创建了一个上下文，表明文件对象不再在上下文之外打开。绑定变量`f`表示文件对象，所有文件对象方法都可以通过该变量访问。`read()`方法在第二行读取整个文件，然后`print()`函数输出文件内容。

当程序到达`with`语句块上下文的末尾时，它会关闭文件以释放资源，并确保其他程序可以使用它们。一般来说，当您处理那些一旦不再需要就需要关闭的对象(比如文件、数据库和网络连接)时，强烈推荐使用`with`语句。

注意，即使在退出`with`上下文管理器块之后，我们仍然可以访问`f`变量；但是，该文件已关闭。让我们尝试一些文件对象属性，看看该变量是否仍然有效并且可以访问:

```py
print("Filename is '{}'.".format(f.name))
if f.closed:
    print("File is closed.")
else:
    print("File isn't closed.")
```

```py
 Filename is 'zen_of_python.txt'.
    File is closed.
```

然而，不可能从文件中读取或向文件中写入。当文件关闭时，任何访问其内容的尝试都会导致以下错误:

```py
f.read()
```

```py
 ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    ~\AppData\Local\Temp/ipykernel_9828/3059900045.py in <module>
    ----> 1 f.read()

    ValueError: I/O operation on closed file.
```

## Python 中的文件模式

正如我们在上一节中提到的，我们需要在打开文件时指定模式。下表显示了 Python 中不同的文件模式:

| 方式 | 描述 |
| --- | --- |
| `'r'` | 它打开一个只读文件。 |
| `'w'` | 它打开一个文件进行写入。如果文件存在，它会覆盖它，否则，它会创建一个新文件。 |
| `'a'` | 它只打开一个附加文件。如果文件不存在，它会创建文件。 |
| `'x'` | 它会创建一个新文件。如果文件存在，它将失败。 |
| `'+'` | 它打开一个文件进行更新。 |

我们还可以指定以文本模式打开文件，默认模式是`'t'`，或者二进制模式是`'b'`。让我们看看如何使用简单的语句复制一个图像文件， [dataquest_logo.png](https://raw.githubusercontent.com/m-mehdi/tutorials/8210bc95fdde6e46c393bd56298cee1a49ea08b1/dataquest_logo.png) :

```py
with open('dataquest_logo.png', 'rb') as rf:
    with open('data_quest_logo_copy.png', 'wb') as wf:
        for b in rf:
            wf.write(b)
```

上面的代码复制了 Dataquest 徽标图像，并将其存储在相同的路径中。`'rb'`模式以二进制模式打开文件进行读取，`'wb'`模式以文本模式打开文件进行写入。

## 读取文本文件

阅读文本文件有不同的方法。本节将回顾一些阅读文本文件内容的有用方法。

到目前为止，我们已经了解了使用`read()`方法可以读取文件的全部内容。如果我们只想从文本文件中读取几个字节呢？为此，在`read()`方法中指定字节数。让我们来试试:

```py
with open('zen_of_python.txt') as f:
    print(f.read(17))
```

```py
The Zen of Python
```

上面的简单代码读取 *zen_of_python.txt* 文件的前 17 个字节并打印出来。

有时，一次读取一行文本文件的内容更有意义。在这种情况下，我们可以使用`readline()`方法。让我们开始吧:

```py
with open('zen_of_python.txt') as f:
    print(f.readline())
```

```py
The Zen of Python, by Tim Peters
```

上面的代码返回文件的第一行。如果我们再次调用该方法，它将返回文件中的第二行，等等。，如下所示:

```py
with open('zen_of_python.txt') as f:
    print(f.readline())
    print(f.readline())
    print(f.readline())
    print(f.readline())
```

```py
The Zen of Python, by Tim Peters  

Beautiful is better than ugly.

Explicit is better than implicit.
```

这种有用的方法帮助我们以增量方式读取整个文件。下面的代码通过逐行迭代输出整个文件，直到跟踪文件读写位置的文件指针到达文件末尾。当`readline()`方法到达文件末尾时，它返回一个空字符串`''`。用 open('zen_of_python.txt ')作为 f:

```py
with open('zen_of_python.txt') as f:
    line = f.readline()
    while line:
        print(line, end='')
        line = f.readline() 
```

```py
 The Zen of Python, by Tim Peters

    Beautiful is better than ugly.
    Explicit is better than implicit.
    Simple is better than complex.
    Complex is better than complicated.
    Flat is better than nested.
    Sparse is better than dense.
    Readability counts.
    Special cases aren't special enough to break the rules.
    Although practicality beats purity.
    Errors should never pass silently.
    Unless explicitly silenced.
    In the face of ambiguity, refuse the temptation to guess.
    There should be one-- and preferably only one --obvious way to do it.
    Although that way may not be obvious at first unless you're Dutch.
    Now is better than never.
    Although never is often better than *right* now.
    If the implementation is hard to explain, it's a bad idea.
    If the implementation is easy to explain, it may be a good idea.
    Namespaces are one honking great idea -- let's do more of those!
```

上面的代码在 while 循环之外读取文件的第一行，并将其赋给`line`变量。在 while 循环中，它打印存储在`line`变量中的字符串，然后读取文件的下一行。while 循环迭代这个过程，直到`readline()`方法返回一个空字符串。在 while 循环中，空字符串的计算结果为`False`，因此迭代过程终止。

读取文本文件的另一个有用的方法是`readlines()`方法。对 file 对象应用此方法将返回包含文件每行的字符串列表。让我们看看它是如何工作的:

```py
with open('zen_of_python.txt') as f:
    lines = f.readlines()
```

让我们检查一下`lines`变量的数据类型，然后打印出来:

```py
print(type(lines))
print(lines)
```

```py
<class 'list'>
['The Zen of Python, by Tim Peters\n', '\n', 'Beautiful is better than ugly.\n', 'Explicit is better than implicit.\n', 'Simple is better than complex.\n', 'Complex is better than complicated.\n', 'Flat is better than nested.\n', 'Sparse is better than dense.\n', 'Readability counts.\n', "Special cases aren't special enough to break the rules.\n", 'Although practicality beats purity.\n', 'Errors should never pass silently.\n', 'Unless explicitly silenced.\n', 'In the face of ambiguity, refuse the temptation to guess.\n', 'There should be one-- and preferably only one --obvious way to do it.\n', "Although that way may not be obvious at first unless you're Dutch.\n", 'Now is better than never.\n', 'Although never is often better than *right* now.\n', "If the implementation is hard to explain, it's a bad idea.\n", 'If the implementation is easy to explain, it may be a good idea.\n', "Namespaces are one honking great idea -- let's do more of those!"]
```

这是一个字符串列表，列表中的每一项都是文本文件中的一行。`\n`转义字符代表文件中的新行。此外，我们可以通过索引或切片操作来访问列表中的每一项:

```py
print(lines)
print(lines[3:5])
print(lines[-1])
```

```py
['The Zen of Python, by Tim Peters\n', '\n', 'Beautiful is better than ugly.\n', 'Explicit is better than implicit.\n', 'Simple is better than complex.\n', 'Complex is better than complicated.\n', 'Flat is better than nested.\n', 'Sparse is better than dense.\n', 'Readability counts.\n', "Special cases aren't special enough to break the rules.\n", 'Although practicality beats purity.\n', 'Errors should never pass silently.\n', 'Unless explicitly silenced.\n', 'In the face of ambiguity, refuse the temptation to guess.\n', 'There should be one-- and preferably only one --obvious way to do it.\n', "Although that way may not be obvious at first unless you're Dutch.\n", 'Now is better than never.\n', 'Although never is often better than *right* now.\n', "If the implementation is hard to explain, it's a bad idea.\n", 'If the implementation is easy to explain, it may be a good idea.\n', "Namespaces are one honking great idea -- let's do more of those!"]
['Explicit is better than implicit.\n', 'Simple is better than complex.\n']
Namespaces are one honking great idea -- let's do more of those!
```

## 读取 CSV 文件

到目前为止，我们已经学习了如何处理常规文本文件。然而，有时数据以 CSV 格式出现，数据专业人员检索所需信息并处理 CSV 文件的内容是很常见的。

我们将在本节中使用 CSV 模块。CSV 模块提供了读取 CSV 文件中存储的逗号分隔值的有用方法。我们现在就尝试一下，但是首先，你需要下载 [`chocolate.csv`](https://raw.githubusercontent.com/m-mehdi/tutorials/main/chocolate.csv) 文件，并将其存储在当前工作目录中:

```py
import csv
with open('chocolate.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        print(row)
```

```py
 ['Company', 'Bean Origin or Bar Name', 'REF', 'Review Date', 'Cocoa Percent', 'Company Location', 'Rating', 'Bean Type', 'Country of Origin']
    ['A. Morin', 'Agua Grande', '1876', '2016', '63%', 'France', '3.75', 'Â\xa0', 'Sao Tome']
    ['A. Morin', 'Kpime', '1676', '2015', '70%', 'France', '2.75', 'Â\xa0', 'Togo']
    ['A. Morin', 'Atsane', '1676', '2015', '70%', 'France', '3', 'Â\xa0', 'Togo']
    ['A. Morin', 'Akata', '1680', '2015', '70%', 'France', '3.5', 'Â\xa0', 'Togo']
    ['Acalli', 'Chulucanas, El Platanal', '1462', '2015', '70%', 'U.S.A.', '3.75', 'Â\xa0', 'Peru']
    ['Acalli', 'Tumbes, Norandino', '1470', '2015', '70%', 'U.S.A.', '3.75', 'Criollo', 'Peru']
    ['Adi', 'Vanua Levu', '705', '2011', '60%', 'Fiji', '2.75', 'Trinitario', 'Fiji']
    ['Adi', 'Vanua Levu, Toto-A', '705', '2011', '80%', 'Fiji', '3.25', 'Trinitario', 'Fiji']
    ['Adi', 'Vanua Levu', '705', '2011', '88%', 'Fiji', '3.5', 'Trinitario', 'Fiji']
    ['Adi', 'Vanua Levu, Ami-Ami-CA', '705', '2011', '72%', 'Fiji', '3.5', 'Trinitario', 'Fiji']
    ['Aequare (Gianduja)', 'Los Rios, Quevedo, Arriba', '370', '2009', '55%', 'Ecuador', '2.75', 'Forastero (Arriba)', 'Ecuador']
    ['Aequare (Gianduja)', 'Los Rios, Quevedo, Arriba', '370', '2009', '70%', 'Ecuador', '3', 'Forastero (Arriba)', 'Ecuador']
    ['Ah Cacao', 'Tabasco', '316', '2009', '70%', 'Mexico', '3', 'Criollo', 'Mexico']
    ["Akesson's (Pralus)", 'Bali (west), Sukrama Family, Melaya area', '636', '2011', '75%', 'Switzerland', '3.75', 'Trinitario', 'Indonesia']
    ["Akesson's (Pralus)", 'Madagascar, Ambolikapiky P.', '502', '2010', '75%', 'Switzerland', '2.75', 'Criollo', 'Madagascar']
    ["Akesson's (Pralus)", 'Monte Alegre, D. Badero', '508', '2010', '75%', 'Switzerland', '2.75', 'Forastero', 'Brazil']
    ['Alain Ducasse', 'Trinite', '1215', '2014', '65%', 'France', '2.75', 'Trinitario', 'Trinidad']
    ['Alain Ducasse', 'Vietnam', '1215', '2014', '75%', 'France', '2.75', 'Trinitario', 'Vietnam']
    ['Alain Ducasse', 'Madagascar', '1215', '2014', '75%', 'France', '3', 'Trinitario', 'Madagascar']
    ['Alain Ducasse', 'Chuao', '1061', '2013', '75%', 'France', '2.5', 'Trinitario', 'Venezuela']
    ['Alain Ducasse', 'Piura, Perou', '1173', '2013', '75%', 'France', '2.5', 'Â\xa0', 'Peru']
    ['Alexandre', 'Winak Coop, Napo', '1944', '2017', '70%', 'Netherlands', '3.5', 'Forastero (Nacional)', 'Ecuador']
    ['Alexandre', 'La Dalia, Matagalpa', '1944', '2017', '70%', 'Netherlands', '3.5', 'Criollo, Trinitario', 'Nicaragua']
    ['Alexandre', 'Tien Giang', '1944', '2017', '70%', 'Netherlands', '3.5', 'Trinitario', 'Vietnam']
    ['Alexandre', 'Makwale Village, Kyela', '1944', '2017', '70%', 'Netherlands', '3.5', 'Forastero', 'Tanzania']
    ['Altus aka Cao Artisan', 'Momotombo', '1728', '2016', '60%', 'U.S.A.', '2.75', 'Â\xa0', 'Nicaragua']
    ['Altus aka Cao Artisan', 'Bolivia', '1133', '2013', '60%', 'U.S.A.', '3', 'Â\xa0', 'Bolivia']
    ['Altus aka Cao Artisan', 'Peru', '1133', '2013', '60%', 'U.S.A.', '3.25', 'Â\xa0', 'Peru']
    ['Amano', 'Morobe', '725', '2011', '70%', 'U.S.A.', '4', 'Â\xa0', 'Papua New Guinea']
    ['Amano', 'Dos Rios', '470', '2010', '70%', 'U.S.A.', '3.75', 'Â\xa0', 'Dominican Republic']
    ['Amano', 'Guayas', '470', '2010', '70%', 'U.S.A.', '4', 'Â\xa0', 'Ecuador']
    ['Amano', 'Chuao', '544', '2010', '70%', 'U.S.A.', '3', 'Trinitario', 'Venezuela']
    ['Amano', 'Montanya', '363', '2009', '70%', 'U.S.A.', '3', 'Â\xa0', 'Venezuela']
    ['Amano', 'Bali, Jembrana', '304', '2008', '70%', 'U.S.A.', '2.75', 'Â\xa0', 'Indonesia']
    ['Amano', 'Madagascar', '129', '2007', '70%', 'U.S.A.', '3.5', 'Trinitario', 'Madagascar']
    ['Amano', 'Cuyagua', '147', '2007', '70%', 'U.S.A.', '3', 'Â\xa0', 'Venezuela']
    ['Amano', 'Ocumare', '175', '2007', '70%', 'U.S.A.', '3.75', 'Criollo', 'Venezuela']
    ['Amatller (Simon Coll)', 'Ghana', '322', '2009', '70%', 'Spain', '3', 'Forastero', 'Ghana']
    ['Amatller (Simon Coll)', 'Ecuador', '327', '2009', '70%', 'Spain', '2.75', 'Â\xa0', 'Ecuador']
    ['Amatller (Simon Coll)', 'Ecuador', '464', '2009', '85%', 'Spain', '2.75', 'Â\xa0', 'Ecuador']
    ['Amatller (Simon Coll)', 'Ghana', '464', '2009', '85%', 'Spain', '3', 'Forastero', 'Ghana']
    ['Amazona', 'LamasdelChanka, San Martin, Oro Verde coop', '1145', '2013', '72%', 'Peru', '3.25', 'Â\xa0', 'Peru']
    ['Ambrosia', 'Venezuela', '1498', '2015', '70%', 'Canada', '3.25', 'Â\xa0', 'Venezuela']
    ['Ambrosia', 'Peru', '1498', '2015', '68%', 'Canada', '3.5', 'Â\xa0', 'Peru']
    ['Amedei', 'Piura, Blanco de Criollo', '979', '2012', '70%', 'Italy', '3.75', 'Â\xa0', 'Peru']
    ['Amedei', 'Porcelana', '111', '2007', '70%', 'Italy', '4', 'Criollo (Porcelana)', 'Venezuela']
    ['Amedei', 'Nine', '111', '2007', '75%', 'Italy', '4', 'Blend', 'Â\xa0']
    ['Amedei', 'Chuao', '111', '2007', '70%', 'Italy', '5', 'Trinitario', 'Venezuela']
    ['Amedei', 'Ecuador', '123', '2007', '70%', 'Italy', '3', 'Trinitario', 'Ecuador']
    ['Amedei', 'Jamaica', '123', '2007', '70%', 'Italy', '3', 'Trinitario', 'Jamaica']
    ['Amedei', 'Grenada', '123', '2007', '70%', 'Italy', '3.5', 'Trinitario', 'Grenada']
    ['Amedei', 'Venezuela', '123', '2007', '70%', 'Italy', '3.75', 'Trinitario (85% Criollo)', 'Venezuela']
    ['Amedei', 'Madagascar', '123', '2007', '70%', 'Italy', '4', 'Trinitario (85% Criollo)', 'Madagascar']
    ['Amedei', 'Trinidad', '129', '2007', '70%', 'Italy', '3.5', 'Trinitario', 'Trinidad']
    ['Amedei', 'Toscano Black', '170', '2007', '63%', 'Italy', '3.5', 'Blend', 'Â\xa0']
    ['Amedei', 'Toscano Black', '40', '2006', '70%', 'Italy', '5', 'Blend', 'Â\xa0']
    ['Amedei', 'Toscano Black', '75', '2006', '66%', 'Italy', '4', 'Blend', 'Â\xa0']
    ['AMMA', 'Catongo', '1065', '2013', '75%', 'Brazil', '3.25', 'Forastero (Catongo)', 'Brazil']
    ['AMMA', 'Monte Alegre, 3 diff. plantations', '572', '2010', '85%', 'Brazil', '2.75', 'Forastero (Parazinho)', 'Brazil']
    ['AMMA', 'Monte Alegre, 3 diff. plantations', '572', '2010', '50%', 'Brazil', '3.75', 'Forastero (Parazinho)', 'Brazil']
    ['AMMA', 'Monte Alegre, 3 diff. plantations', '572', '2010', '75%', 'Brazil', '3.75', 'Forastero (Parazinho)', 'Brazil']
    ['AMMA', 'Monte Alegre, 3 diff. plantations', '572', '2010', '60%', 'Brazil', '4', 'Forastero (Parazinho)', 'Brazil']
    ['Anahata', 'Elvesia', '1259', '2014', '75%', 'U.S.A.', '3', 'Â\xa0', 'Dominican Republic']
    ['Animas', 'Alto Beni', '1852', '2016', '75%', 'U.S.A.', '3.5', 'Â\xa0', 'Bolivia']
    ['Ara', 'Madagascar', '1375', '2014', '75%', 'France', '3', 'Trinitario', 'Madagascar']
    ['Ara', 'Chiapas', '1379', '2014', '72%', 'France', '2.5', 'Â\xa0', 'Mexico']
    ['Arete', 'Camino Verde', '1602', '2015', '75%', 'U.S.A.', '3.5', 'Â\xa0', 'Ecuador']
    ['Artisan du Chocolat', 'Congo', '300', '2008', '72%', 'U.K.', '3.75', 'Forastero', 'Congo']
    ['Artisan du Chocolat (Casa Luker)', 'Orinoqua Region, Arauca', '1181', '2013', '72%', 'U.K.', '2.75', 'Trinitario', 'Colombia']
    ['Askinosie', 'Mababa', '1780', '2016', '68%', 'U.S.A.', '3.75', 'Trinitario', 'Tanzania']
    ['Askinosie', 'Tenende, Uwate', '647', '2011', '72%', 'U.S.A.', '3.75', 'Trinitario', 'Tanzania']
    ['Askinosie', 'Cortes', '661', '2011', '70%', 'U.S.A.', '3.75', 'Trinitario', 'Honduras']
    ['Askinosie', 'Davao', '331', '2009', '77%', 'U.S.A.', '3.75', 'Trinitario', 'Philippines']
    ['Askinosie', 'Xoconusco', '141', '2007', '75%', 'U.S.A.', '2.5', 'Trinitario', 'Mexico']
    ['Askinosie', 'San Jose del Tambo', '175', '2007', '70%', 'U.S.A.', '3', 'Forastero (Arriba)', 'Ecuador']
    ['Bahen & Co.', 'Houseblend', '999', '2012', '70%', 'Australia', '2.5', 'Blend', 'Â\xa0']
    ['Bakau', 'Bambamarca, 2015', '1454', '2015', '70%', 'Peru', '2.75', 'Â\xa0', 'Peru']
    ['Bakau', 'Huallabamba, 2015', '1454', '2015', '70%', 'Peru', '3.5', 'Â\xa0', 'Peru']
    ['Bar Au Chocolat', 'Bahia', '1554', '2015', '70%', 'U.S.A.', '3.5', 'Â\xa0', 'Brazil']
    ['Bar Au Chocolat', 'Maranon Canyon', '1295', '2014', '70%', 'U.S.A.', '4', 'Forastero (Nacional)', 'Peru']
    ['Bar Au Chocolat', 'Duarte Province', '983', '2012', '70%', 'U.S.A.', '3.25', 'Â\xa0', 'Dominican Republic']
    ['Bar Au Chocolat', 'Chiapas', '983', '2012', '70%', 'U.S.A.', '3.5', 'Â\xa0', 'Mexico']
    ['Bar Au Chocolat', 'Sambirano', '983', '2012', '70%', 'U.S.A.', '3.75', 'Trinitario', 'Madagascar']
    ["Baravelli's", 'single estate', '955', '2012', '80%', 'Wales', '2.75', 'Â\xa0', 'Costa Rica']
    ['Batch', 'Dominican Republic, Batch 3', '1840', '2016', '65%', 'U.S.A.', '3.5', 'Â\xa0', 'Domincan Republic']
    ['Batch', 'Brazil', '1868', '2016', '70%', 'U.S.A.', '3.75', 'Â\xa0', 'Brazil']
    ['Batch', 'Ecuador', '1880', '2016', '65%', 'U.S.A.', '3.25', 'Â\xa0', 'Ecuador']
    ['Beau Cacao', 'Asajaya E, NW Borneo, b. #132/4500', '1948', '2017', '73%', 'U.K.', '3', 'Â\xa0', 'Malaysia']
    ['Beau Cacao', 'Serian E., NW Borneo, b. #134/3800', '1948', '2017', '72%', 'U.K.', '3.25', 'Â\xa0', 'Malaysia']
    ['Beehive', 'Brazil, Batch 20316', '1784', '2016', '80%', 'U.S.A.', '2.75', 'Â\xa0', 'Brazil']
    ['Beehive', 'Dominican Republic, Batch 31616', '1784', '2016', '70%', 'U.S.A.', '2.75', 'Â\xa0', 'Domincan Republic']
    ['Beehive', 'Ecuador, Batch 31516', '1784', '2016', '70%', 'U.S.A.', '2.75', 'Â\xa0', 'Ecuador']
    ['Beehive', 'Ecuador', '1788', '2016', '90%', 'U.S.A.', '2.75', 'Â\xa0', 'Ecuador']
    ['Belcolade', 'Costa Rica', '586', '2010', '64%', 'Belgium', '2.75', 'Â\xa0', 'Costa Rica']
    ['Belcolade', 'Papua New Guinea', '586', '2010', '64%', 'Belgium', '2.75', 'Â\xa0', 'Papua New Guinea']
    ['Belcolade', 'Peru', '586', '2010', '64%', 'Belgium', '2.75', 'Â\xa0', 'Peru']
    ['Belcolade', 'Ecuador', '586', '2010', '71%', 'Belgium', '3.5', 'Â\xa0', 'Ecuador']
    ['Bellflower', 'Kakao Kamili, Kilombero Valley', '1800', '2016', '70%', 'U.S.A.', '3.5', 'Criollo, Trinitario', 'Tanzania']
    ['Bellflower', 'Alto Beni, Palos Blanco', '1804', '2016', '70%', 'U.S.A.', '3.25', 'Â\xa0', 'Bolivia']
    ['Vintage Plantations (Tulicorp)', 'Los Rios, Rancho Grande 2007', '153', '2007', '65%', 'U.S.A.', '3', 'Forastero (Arriba)', 'Ecuador']
    ['Violet Sky', 'Sambirano Valley', '1458', '2015', '77%', 'U.S.A.', '2.75', 'Trinitario', 'Madagascar']
    ['Wm', 'Wild Beniano, 2016, batch 128, Heirloom', '1912', '2016', '76%', 'U.S.A.', '3.5', 'Â\xa0', 'Bolivia']
    ['Wm', 'Ghana, 2013, batch 129', '1916', '2016', '75%', 'U.S.A.', '3.75', 'Â\xa0', 'Ghana']
    ['Woodblock', 'La Red', '769', '2011', '70%', 'U.S.A.', '3.5', 'Â\xa0', 'Dominican Republic']
    ['Xocolat', 'Hispaniola', '1057', '2013', '66%', 'Domincan Republic', '3', 'Â\xa0', 'Dominican Republic']
    ['Xocolla', 'Sambirano, batch 170102', '1948', '2017', '70%', 'U.S.A.', '2.75', 'Â\xa0', 'Madagascar']
    ['Xocolla', 'Hispaniola, batch 170104', '1948', '2017', '70%', 'U.S.A.', '2.5', 'Â\xa0', 'Dominican Republic']
    ['Zart Pralinen', 'UNOCACE', '1824', '2016', '70%', 'Austria', '2.75', 'Nacional (Arriba)', 'Ecuador']
    ['Zart Pralinen', 'San Juan Estate', '1824', '2016', '85%', 'Austria', '2.75', 'Trinitario', 'Trinidad']
    ['Zokoko', 'Guadalcanal', '1716', '2016', '78%', 'Australia', '3.75', 'Â\xa0', 'Solomon Islands']
```

CSV 文件的每一行都形成一个列表，其中的每一项都可以方便地访问，如下所示:

```py
import csv
with open('chocolate.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        print("The {} company is located in {}.".format(row[0], row[5]))
```

```py
 The Company company is located in Company Location.
    The A. Morin company is located in France.
    The A. Morin company is located in France.
    The A. Morin company is located in France.
    The A. Morin company is located in France.
    The Acalli company is located in U.S.A..
    The Acalli company is located in U.S.A..
    The Adi company is located in Fiji.
    The Adi company is located in Fiji.
    The Adi company is located in Fiji.
    The Adi company is located in Fiji.
    The Aequare (Gianduja) company is located in Ecuador.
    The Aequare (Gianduja) company is located in Ecuador.
    The Ah Cacao company is located in Mexico.
    The Akesson's (Pralus) company is located in Switzerland.
    The Akesson's (Pralus) company is located in Switzerland.
    The Akesson's (Pralus) company is located in Switzerland.
    The Alain Ducasse company is located in France.
    The Alain Ducasse company is located in France.
    The Alain Ducasse company is located in France.
    The Alain Ducasse company is located in France.
    The Alain Ducasse company is located in France.
    The Alexandre company is located in Netherlands.
    The Alexandre company is located in Netherlands.
    The Alexandre company is located in Netherlands.
    The Alexandre company is located in Netherlands.
    The Altus aka Cao Artisan company is located in U.S.A..
    The Altus aka Cao Artisan company is located in U.S.A..
    The Altus aka Cao Artisan company is located in U.S.A..
    The Amano company is located in U.S.A..
    The Amano company is located in U.S.A..
    The Amano company is located in U.S.A..
    The Amano company is located in U.S.A..
    The Amano company is located in U.S.A..
    The Amano company is located in U.S.A..
    The Amano company is located in U.S.A..
    The Amano company is located in U.S.A..
    The Amano company is located in U.S.A..
    The Amatller (Simon Coll) company is located in Spain.
    The Amatller (Simon Coll) company is located in Spain.
    The Amatller (Simon Coll) company is located in Spain.
    The Amatller (Simon Coll) company is located in Spain.
    The Amazona company is located in Peru.
    The Ambrosia company is located in Canada.
    The Ambrosia company is located in Canada.
    The Amedei company is located in Italy.
    The Amedei company is located in Italy.
    The Amedei company is located in Italy.
    The Amedei company is located in Italy.
    The Amedei company is located in Italy.
    The Amedei company is located in Italy.
    The Amedei company is located in Italy.
    The Amedei company is located in Italy.
    The Amedei company is located in Italy.
    The Amedei company is located in Italy.
    The Amedei company is located in Italy.
    The Amedei company is located in Italy.
    The Amedei company is located in Italy.
    The AMMA company is located in Brazil.
    The AMMA company is located in Brazil.
    The AMMA company is located in Brazil.
    The AMMA company is located in Brazil.
    The AMMA company is located in Brazil.
    The Anahata company is located in U.S.A..
    The Animas company is located in U.S.A..
    The Ara company is located in France.
    The Ara company is located in France.
    The Arete company is located in U.S.A..
    The Artisan du Chocolat company is located in U.K..
    The Artisan du Chocolat (Casa Luker) company is located in U.K..
    The Askinosie company is located in U.S.A..
    The Askinosie company is located in U.S.A..
    The Askinosie company is located in U.S.A..
    The Askinosie company is located in U.S.A..
    The Askinosie company is located in U.S.A..
    The Askinosie company is located in U.S.A..
    The Bahen & Co. company is located in Australia.
    The Bakau company is located in Peru.
    The Bakau company is located in Peru.
    The Bar Au Chocolat company is located in U.S.A..
    The Bar Au Chocolat company is located in U.S.A..
    The Bar Au Chocolat company is located in U.S.A..
    The Bar Au Chocolat company is located in U.S.A..
    The Bar Au Chocolat company is located in U.S.A..
    The Baravelli's company is located in Wales.
    The Batch company is located in U.S.A..
    The Batch company is located in U.S.A..
    The Batch company is located in U.S.A..
    The Beau Cacao company is located in U.K..
    The Beau Cacao company is located in U.K..
    The Beehive company is located in U.S.A..
    The Beehive company is located in U.S.A..
    The Beehive company is located in U.S.A..
    The Beehive company is located in U.S.A..
    The Belcolade company is located in Belgium.
    The Belcolade company is located in Belgium.
    The Belcolade company is located in Belgium.
    The Belcolade company is located in Belgium.
    The Bellflower company is located in U.S.A..
    The Bellflower company is located in U.S.A..
    The Vintage Plantations (Tulicorp) company is located in U.S.A..
    The Violet Sky company is located in U.S.A..
    The Wm company is located in U.S.A..
    The Wm company is located in U.S.A..
    The Woodblock company is located in U.S.A..
    The Xocolat company is located in Domincan Republic.
    The Xocolla company is located in U.S.A..
    The Xocolla company is located in U.S.A..
    The Zart Pralinen company is located in Austria.
    The Zart Pralinen company is located in Austria.
    The Zokoko company is located in Australia.
```

可以使用列名而不是使用它们的索引，这对开发人员来说通常更方便。在这种情况下，我们不使用`reader()`方法，而是使用返回字典对象集合的`DictReader()`方法。让我们来试试:

```py
import csv
with open('chocolate.csv') as f:
    dict_reader = csv.DictReader(f, delimiter=',')
    for row in dict_reader:
        print("The {} company is located in {}.".format(row['Company'], row['Company Location']))
```

```py
 The A. Morin company is located in France.
    The A. Morin company is located in France.
    The A. Morin company is located in France.
    The A. Morin company is located in France.
    The Acalli company is located in U.S.A..
    The Acalli company is located in U.S.A..
    The Adi company is located in Fiji.
    The Adi company is located in Fiji.
    The Adi company is located in Fiji.
    The Adi company is located in Fiji.
    The Aequare (Gianduja) company is located in Ecuador.
    The Aequare (Gianduja) company is located in Ecuador.
    The Ah Cacao company is located in Mexico.
    The Akesson's (Pralus) company is located in Switzerland.
    The Akesson's (Pralus) company is located in Switzerland.
    The Akesson's (Pralus) company is located in Switzerland.
    The Alain Ducasse company is located in France.
    The Alain Ducasse company is located in France.
    The Alain Ducasse company is located in France.
    The Alain Ducasse company is located in France.
    The Alain Ducasse company is located in France.
    The Alexandre company is located in Netherlands.
    The Alexandre company is located in Netherlands.
    The Alexandre company is located in Netherlands.
    The Alexandre company is located in Netherlands.
    The Altus aka Cao Artisan company is located in U.S.A..
    The Altus aka Cao Artisan company is located in U.S.A..
    The Altus aka Cao Artisan company is located in U.S.A..
    The Amano company is located in U.S.A..
    The Amano company is located in U.S.A..
    The Amano company is located in U.S.A..
    The Amano company is located in U.S.A..
    The Amano company is located in U.S.A..
    The Amano company is located in U.S.A..
    The Amano company is located in U.S.A..
    The Amano company is located in U.S.A..
    The Amano company is located in U.S.A..
    The Amatller (Simon Coll) company is located in Spain.
    The Amatller (Simon Coll) company is located in Spain.
    The Amatller (Simon Coll) company is located in Spain.
    The Amatller (Simon Coll) company is located in Spain.
    The Amazona company is located in Peru.
    The Ambrosia company is located in Canada.
    The Ambrosia company is located in Canada.
    The Amedei company is located in Italy.
    The Amedei company is located in Italy.
    The Amedei company is located in Italy.
    The Amedei company is located in Italy.
    The Amedei company is located in Italy.
    The Amedei company is located in Italy.
    The Amedei company is located in Italy.
    The Amedei company is located in Italy.
    The Amedei company is located in Italy.
    The Amedei company is located in Italy.
    The Amedei company is located in Italy.
    The Amedei company is located in Italy.
    The Amedei company is located in Italy.
    The AMMA company is located in Brazil.
    The AMMA company is located in Brazil.
    The AMMA company is located in Brazil.
    The AMMA company is located in Brazil.
    The AMMA company is located in Brazil.
    The Anahata company is located in U.S.A..
    The Animas company is located in U.S.A..
    The Ara company is located in France.
    The Ara company is located in France.
    The Arete company is located in U.S.A..
    The Artisan du Chocolat company is located in U.K..
    The Artisan du Chocolat (Casa Luker) company is located in U.K..
    The Askinosie company is located in U.S.A..
    The Askinosie company is located in U.S.A..
    The Askinosie company is located in U.S.A..
    The Askinosie company is located in U.S.A..
    The Askinosie company is located in U.S.A..
    The Askinosie company is located in U.S.A..
    The Bahen & Co. company is located in Australia.
    The Bakau company is located in Peru.
    The Bakau company is located in Peru.
    The Bar Au Chocolat company is located in U.S.A..
    The Bar Au Chocolat company is located in U.S.A..
    The Bar Au Chocolat company is located in U.S.A..
    The Bar Au Chocolat company is located in U.S.A..
    The Bar Au Chocolat company is located in U.S.A..
    The Baravelli's company is located in Wales.
    The Batch company is located in U.S.A..
    The Batch company is located in U.S.A..
    The Batch company is located in U.S.A..
    The Beau Cacao company is located in U.K..
    The Beau Cacao company is located in U.K..
    The Beehive company is located in U.S.A..
    The Beehive company is located in U.S.A..
    The Beehive company is located in U.S.A..
    The Beehive company is located in U.S.A..
    The Belcolade company is located in Belgium.
    The Belcolade company is located in Belgium.
    The Belcolade company is located in Belgium.
    The Belcolade company is located in Belgium.
    The Bellflower company is located in U.S.A..
    The Bellflower company is located in U.S.A..
    The Vintage Plantations (Tulicorp) company is located in U.S.A..
    The Violet Sky company is located in U.S.A..
    The Wm company is located in U.S.A..
    The Wm company is located in U.S.A..
    The Woodblock company is located in U.S.A..
    The Xocolat company is located in Domincan Republic.
    The Xocolla company is located in U.S.A..
    The Xocolla company is located in U.S.A..
    The Zart Pralinen company is located in Austria.
    The Zart Pralinen company is located in Austria.
    The Zokoko company is located in Australia.
```

## 读取 JSON 文件

我们主要用于存储和交换数据的另一种流行的文件格式是 JSON。JSON 代表 JavaScript 对象符号，它允许我们用逗号分隔的键值对来存储数据。

在这一节中，我们将加载一个 JSON 文件，并将其作为一个 JSON 对象使用，而不是作为一个文本文件。为此，我们需要导入 JSON 模块。然后，在`with`上下文管理器中，我们使用属于`json`对象的`load()`方法。它加载文件的内容，并将其作为字典存储在`context`变量中。让我们试一试，但是在运行代码之前，下载 [`movie.json`](https://raw.githubusercontent.com/m-mehdi/tutorials/main/movie.json) 文件，放在当前工作目录下。

```py
import json
with open('movie.json') as f:
    content = json.load(f)
    print(content)
```

```py
 {'Title': 'Bicentennial Man', 'Release Date': 'Dec 17 1999', 'MPAA Rating': 'PG', 'Running Time min': 132, 'Distributor': 'Walt Disney Pictures', 'Source': 'Based on Book/Short Story', 'Major Genre': 'Drama', 'Creative Type': 'Science Fiction', 'Director': 'Chris Columbus', 'Rotten Tomatoes Rating': 38, 'IMDB Rating': 6.4, 'IMDB Votes': 28827}
```

让我们检查一下`content`变量的数据类型:

```py
print(type(content))
```

```py
 <class 'dict'>
```

它的数据类型是字典。所以我们可以用它的键访问存储在 JSON 文件中的每一条信息。让我们看看如何从中检索数据:

```py
print('{} directed by {}'.format(content['Title'], content['Director']))
```

```py
 Bicentennial Man directed by Chris Columbus
```

## 结论

本教程讨论了 Python 中的文件处理，重点是读取文件的内容。您了解了 open()内置函数、带有上下文管理器的，以及如何读取常见的文件类型，如文本、CSV 和 JSON。