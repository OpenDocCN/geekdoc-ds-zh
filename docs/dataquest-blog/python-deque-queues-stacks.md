# Python 的 Deque:如何轻松实现队列和堆栈

> 原文：<https://www.dataquest.io/blog/python-deque-queues-stacks/>

September 8, 2022![collections.deque](img/7242a3d5738e61e3b8d7631af6d3cec6.png)

如果您使用 Python，您可能对列表很熟悉，并且您可能也经常使用它们。它们是很棒的数据结构，有许多有用的方法，允许用户通过添加、删除和排序条目来修改列表。然而，在一些用例中，列表可能看起来是一个很好的选择，但它并不是。

在本文中，我们将介绍当您需要在 Python 中实现队列和堆栈时，`collections`模块中的`deque()`函数如何成为更好的选择。

为了阅读本文，我们假设您对 Python 编程有一定的经验，使用列表及其方法，因为它们中的大多数都可以推断出`collections.deque`的用法。

## 队列和堆栈

首先:在进入`collections`模块之前，让我们先理解队列和堆栈的概念。

### 行列

队列是一种以先进先出(FIFO)方式存储项目的数据结构。也就是说，添加到队列中的第一个项目将是从队列中移除的第一个项目。

比方说，你正在你最喜欢的音乐播放器中添加歌曲。你还没有开始，所以你的队列是空的。然后你添加三首歌曲:`song_1`、`song_2`和`song_3`，按照这个顺序。现在，您的队列如下所示:

```py
queue = ['song_1', 'song_2', 'song_3']
```

然后，您开始播放队列中的歌曲。第一个出场并移出队列的就是第一个加入的:`song_1` —这样，**先进先出**。根据这条规则，如果您播放了几首歌曲，然后又添加了三首，这将是您的队列:

```py
queue = ['song_3', 'song_4', 'song_5', 'song_6']
```

### 大量

堆栈是一种以后进先出(LIFO)方式存储项目的数据结构。也就是说，添加到堆栈中的最后一项将是从堆栈中移除的第一项。

栈的一个经典例子是一堆盘子。假设你有几个朋友过来吃饭，用了五个盘子。晚饭后，你把那五个盘子堆在水槽里。所以你会有这个堆栈:

```py
stack = ['plate_1', 'plate_2', 'plate_3', 'plate_4', 'plate_5']
```

当你开始洗碗时，你从最上面的一堆开始，`plate_5`，这意味着**最后一个进来，先出去**。所以在你洗完三个盘子后，一个朋友拿来另一个她用来做甜点的盘子，放在盘子堆的最上面。您的堆栈现在看起来像这样:

```py
stack = ['plate_1', 'plate_2', 'plate_6']
```

这意味着`plate_6`将是下一个被清洗的。

## 为什么不是列表？

既然我们已经理解了队列和堆栈的概念，那么看起来用 Python 实现这些结构用列表就可以了。毕竟它们是用来代表歌曲的队列和上面的盘子堆的，对吧？嗯，没那么多。

列表不是 Python 中队列或堆栈的最佳数据结构，原因有两个。首先，列表不是线程安全的，这意味着如果一个列表被多个线程同时访问和修改，事情可能会变得不太顺利，最终可能会出现不一致的数据或错误消息。

此外，当您需要从列表的左端插入或删除元素时，列表也不是很有效。如果您使用`list.append()`在右端插入一个元素，或者使用`list.pop()`删除最右边的元素，列表将会很好地执行。但是，当您试图在左端执行这些操作时，列表需要将所有其他元素向右移动，这意味着列表的大小会影响执行此类操作所需的时间，从而导致性能下降。

## 使用`collections.deque`

来自`collections`的`deque`对象是一个类似列表的对象，支持快速追加，从两边弹出。它还支持线程安全、内存高效的操作，并且在用作队列和堆栈时，它被特别设计为比列表更高效。

名称 deque 是双端队列的缩写，发音像“deck”

### 创建一个`deque`对象

`deque`将一个 iterable 作为参数，它将成为一个`deque`对象。如果没有传递，它将为空:

```py
from collections import deque

queue = deque()
print(queue)
```

```py
deque([])
```

但是我们也可以将任何 iterable 传递给`deque`。下面，我们可以看到如何将字典中的列表、元组、集合、键和值转换成一个`deque`对象:

```py
songs_list = ['song_1', 'song_2', 'song_3']
songs_tuple = ('song_1', 'song_2', 'song_3')
songs_set = {'song_1', 'song_2', 'song_3'}
songs_dict = {'1': 'song_1', '2': 'song_2', '3': 'song_3'}

deque_from_list = deque(songs_list)
deque_from_tuple = deque(songs_tuple)
deque_from_set = deque(songs_set)
deque_from_dict_keys = deque(songs_dict.keys())
deque_from_dict_values = deque(songs_dict.values())

print(deque_from_list)
print(deque_from_tuple)
print(deque_from_set)
print(deque_from_dict_keys)
print(deque_from_dict_values)
```

```py
deque(['song_1', 'song_2', 'song_3'])
deque(['song_1', 'song_2', 'song_3'])
deque(['song_3', 'song_1', 'song_2'])
deque(['1', '2', '3'])
deque(['song_1', 'song_2', 'song_3'])
```

现在我们已经初始化了`deque`对象，我们可以使用 append 和 pop 方法从右端插入和删除项目:

```py
queue = deque(songs_list)
print(queue)

queue.append('song_4')
print(queue)

queue.pop()
print(queue)
```

```py
deque(['song_1', 'song_2', 'song_3'])
deque(['song_1', 'song_2', 'song_3', 'song_4'])
deque(['song_1', 'song_2', 'song_3'])
```

注意`song_4` 被插入到最右边的位置，然后被删除。

### 从左侧追加和删除

与列表相比，`deque`最大的优势之一是可以从左端添加和删除条目。

在列表中我们使用`insert()`方法，而对于`deque`，我们可以使用`appendleft()`方法。以下是每种方法的工作原理:

```py
songs_list.insert(0, 'new_song')
print(songs_list)

queue.appendleft('new_song')
print(queue)
```

```py
['new_song', 'song_2', 'song_3']
deque(['new_song', 'song_1', 'song_2', 'song_3'])
```

删除左端的项目也是一样。在一个列表中，我们使用索引为零的`pop()`方法作为参数，表示第一项应该被删除。在一个`deque`中，我们有`popleft()`方法来执行这个任务:

```py
songs_list.pop(0)
print(songs_list)

queue.popleft()
print(queue)
```

```py
['song_2', 'song_3']
deque(['song_1', 'song_2', 'song_3'])
```

如前所述，`deque`对象对于左端的这些操作更有效，尤其是当队列的大小增加时。

根据队列和堆栈的概念，我们使用`popleft()`方法从列表中删除第一个插入的条目。我们从右边追加，从左边移除:**先进先出**。

然而，如果队列为空，`pop()`和`popleft()`方法都会引发错误。在`try`和`except`子句中使用这些方法来防止错误是一个很好的实践。我们将在本文后面看到一个例子。

最后，我们还可以使用`extendleft()`方法将多个元素插入队列。此方法采用一个 iterable。这里有一个例子:

```py
queue.extendleft(['new_song_1', 'new_song_2'])
print(queue)
```

```py
deque(['new_song_2', 'new_song_1', 'song_1', 'song_2', 'song_3'])
```

### 队列上的内置函数和迭代

就像列表、元组和集合一样，`deque`对象也支持 Python 的内置函数。

例如，我们可以使用`len()`来检查一个队列的大小:

```py
print(len(queue))
```

```py
5
```

我们可以使用`sorted()`函数对一个队列进行排序:

```py
print(sorted(queue))
```

```py
['new_song_1', 'new_song_2', 'song_1', 'song_2', 'song_3']
```

我们使用`reversed()`函数来反转`deque`对象中项目的顺序:

```py
print(deque(reversed(queue)))
```

```py
deque(['song_3', 'song_2', 'song_1', 'new_song_1', 'new_song_2'])
```

还支持`max()`和`min()`功能:

```py
new_queue = deque([1, 2, 3])
print(max(new_queue))
print(min(new_queue))
```

```py
3
1
```

当然，我们可以遍历队列:

```py
for song in queue:
    print(song)
```

```py
new_song_2
new_song_1
song_1
song_2
song_3
```

### 实现队列

现在，让我们将一个简化版本的队列付诸实践。我们将继续使用歌曲队列的例子，这意味着我们的队列将继续接收新的歌曲，同时播放队列中最老的歌曲，然后删除它们。虽然我们实现的是队列，但是我们可以使用相同的概念以非常相似的方式实现堆栈。

首先，我们将编写一个函数，将一个`deque`对象作为队列和歌曲列表。然后，该函数选择一首随机歌曲，并将其添加到队列中。该函数还打印出添加了哪首歌曲以及当前队列。这个过程会无限下去。

```py
def add_song(queue, songs):
    while True:
        index = randint(0, len(songs))
        song = songs[index]
        queue.append(song)
        print(f'Song Added: {song}, Queue: {queue}')
        sleep(randint(0, 5))
```

我们现在需要一个功能来删除正在播放的歌曲。这个函数实际上不会播放一首歌曲，因为它的目标只是用 Python 来表示队列的功能。相反，这个函数接收一个队列，删除最左边的元素，并打印被删除的项和当前队列。如果队列为空，该函数将打印出来。

```py
def play_song(queue):
    while True:
        try:
            song = queue.popleft()
            print(f'Song Played: {song}, Queue: {queue}')
        except IndexError:
            print('Queue is empty.')
        sleep(randint(0, 5))
```

接下来，我们将创建一个歌曲列表并实例化一个`deque`对象。最后，我们将使用 Python 的`threading`模块同时运行这两个函数。这是最后的代码:

```py
from threading import Thread
from collections import deque
from time import sleep
from random import randint

songs = [f'song_{i+1}' for i in range(100)]

def add_song(queue, songs):
    while True:
        index = randint(0, len(songs))
        song = songs[index]
        queue.append(song)
        print(f'Song Added: {song}, Queue: {queue}')
        sleep(randint(0, 5))

def play_song(queue):
    while True:
        try:
            song = queue.popleft()
            print(f'Song Played: {song}, Queue: {queue}')
        except IndexError:
            print('Queue is empty.')
        sleep(randint(0, 5))

queue = deque()

Thread(target=add_song, args=(queue, songs)).start()
Thread(target=play_song, args=(queue,)).start()
```

如果我们运行上面的代码，我们将得到如下输出:

```py
Song Added: song_60, Queue: deque(['song_60'])
Song Played: song_60, Queue: deque([])
Queue is empty.
Queue is empty.
Song Added: song_13, Queue: deque(['song_13'])
Song Played: song_13, Queue: deque([])
Queue is empty.
Song Added: song_59, Queue: deque(['song_59'])
Song Added: song_46, Queue: deque(['song_59', 'song_46'])
Song Added: song_48, Queue: deque(['song_59', 'song_46', 'song_48'])
Song Played: song_59, Queue: deque(['song_46', 'song_48'])
Song Added: song_3, Queue: deque(['song_46', 'song_48', 'song_3'])
Song Played: song_46, Queue: deque(['song_48', 'song_3'])
Song Added: song_98, Queue: deque(['song_48', 'song_3', 'song_98'])
Song Played: song_48, Queue: deque(['song_3', 'song_98'])
```

请注意，代码在队列的右端添加歌曲，并从左端删除它们，同时遵守用户定义的歌曲顺序。此外，由于`deque`支持多线程，我们对两个函数同时访问和修改同一个对象没有问题。

## 结论

在本文中，我们介绍了来自`collections`的`deque`对象如何成为在 Python 中实现队列和堆栈的绝佳选择。我们还介绍了以下内容:

*   队列和堆栈的概念

*   为什么在这种情况下列表不是最好的选择

*   如何使用`deque`对象

*   实现在生产中工作的队列的简化版本