# Python 中的多线程:终极指南(带编码示例)

> 原文：<https://www.dataquest.io/blog/multithreading-in-python/>

July 14, 2022![Multithreading in Python](img/09d29377ff3a4ba0a8962b8186ad5c19.png)

## 在本教程中，我们将向您展示如何通过使用 Python 中的多线程技术在代码中实现并行性。

“并行”、“多线程”——这些术语是什么意思，它们之间有什么联系？我们将在本教程中回答您的所有问题，包括以下内容:

*   什么是并发？
*   并发和并行有什么区别？
*   进程和线程有什么区别？
*   Python 中的多线程是什么？
*   什么是全局解释器锁？

这里我们假设您了解 Python 的基础知识，包括基本的结构和功能。如果你对这些不熟悉(或者急于复习)，试试我们的 [Python 数据分析基础——Data quest](https://www.dataquest.io/path/python-basics-for-data-analysis/)。

### 什么是并发？

在我们进入多线程细节之前，让我们比较一下并发编程和顺序编程。

在顺序计算中，程序的组成部分是逐步执行的，以产生正确的结果；然而，在并发计算中，不同的程序组件是独立或半独立的。因此，处理器可以独立地同时运行不同状态的组件。并发的主要优势在于改善程序的运行时间，这意味着由于处理器同时运行独立的任务，因此处理器运行整个程序并完成主要任务所需的时间更少。

### 并发和并行有什么区别？

并发是指在一个处理器和内核上，两个或多个任务可以在重叠的时间段内启动和完成。并行性是指多个任务或任务的分布式部分在多个处理器上同时独立运行。因此，在只有一个处理器和一个内核的机器上，并行是不可能的。

想象两个排队的顾客；并发意味着单个收银员通过在两个队列之间切换来服务客户。并行意味着两个收银员同时为两个顾客队列服务。

### 进程和线程有什么区别？

进程是一个正在执行的程序，它有自己的地址空间、内存、数据堆栈等。操作系统根据任何调度策略将资源分配给进程，并通过将 CPU 时间分配给不同的执行进程来管理进程的执行。

线程类似于进程。但是，它们在相同的进程中执行，并共享相同的上下文。因此，与其他线程共享信息或通信比它们是单独的进程更容易访问。

### Python 中的多线程

Python 虚拟机不是线程安全的解释器，这意味着解释器在任何给定时刻只能执行一个线程。这个限制是由 Python 全局解释器锁(GIL)强制实施的，它本质上限制了一次只能运行一个 Python 线程。换句话说，GIL 确保在同一时间，在单个处理器上，只有一个线程在同一个进程中运行。

基本上，线程化可能不会加快所有任务的速度。与 CPU 绑定任务相比，花费大量时间等待外部事件的 I/O 绑定任务更有可能利用线程。

* * *

**注**

Python 附带了两个用于实现多线程程序的内置模块，包括`thread`和`threading`模块。`thread`和`threading`模块为创建和管理线程提供了有用的特性。然而，在本教程中，我们将把重点放在`threading`模块上，这是一个改进很多的高级模块，用于实现严肃的多线程程序。此外，Python 提供了`Queue`模块，允许我们创建一个队列数据结构来跨多个线程安全地交换信息。

* * *

让我们从一个简单的例子开始，来理解使用多线程编程的好处。

假设我们有一个整数列表，我们要计算每个数字的平方和立方，然后在屏幕上打印出来。

该程序包括如下两个独立的任务(功能):

```py
import time
def calc_square(numbers):
    for n in numbers:
        print(f'\n{n} ^ 2 = {n*n}')
        time.sleep(0.1)

def calc_cube(numbers):
    for n in numbers:
        print(f'\n{n} ^ 3 = {n*n*n}')
        time.sleep(0.1)
```

上面的代码实现了两个功能，`calc_square()`和`calc_cube()`。`calc_square()`函数计算列表中每个数字的平方，`calc_cube()`计算列表中每个数字的立方。函数体中的`time.sleep(0.1)`语句在每次迭代结束时暂停代码执行 0.1 秒。我们将这个语句添加到函数中，使 CPU 空闲一会儿，并模拟一个 I/O 绑定的任务。在真实的场景中，I/O 绑定的任务可能会等待外围设备或 web 服务响应。

```py
numbers = [2, 3, 5, 8]
start = time.time()
calc_square(numbers)
calc_cube(numbers)
end = time.time()

print('Execution Time: {}'.format(end-start))
```

如您所见，程序的顺序执行几乎需要一秒钟。现在让我们利用 CPU 的空闲时间，利用多线程技术，减少总的执行时间。多线程技术通过在其他任务等待 I/O 响应的同时将 CPU 时间分配给一个任务来减少运行时间。让我们看看它是如何工作的:

```py
import threading

start = time.time()

square_thread = threading.Thread(target=calc_square, args=(numbers,))
cube_thread = threading.Thread(target=calc_cube, args=(numbers,))

square_thread.start()
cube_thread.start()

square_thread.join()
cube_thread.join()

end = time.time()

print('Execution Time: {}'.format(end-start))
```

```py
 2 ^ 2 = 4

    2 ^ 3 = 8

    3 ^ 3 = 27

    3 ^ 2 = 9

    5 ^ 3 = 125
    5 ^ 2 = 25

    8 ^ 2 = 64
    8 ^ 3 = 512

    Execution Time: 0.4172379970550537
```

太棒了，执行时间不到半秒，这是一个相当大的改进，这要归功于多线程。让我们一行一行地研究代码。

首先，我们需要导入“线程”模块，这是一个具有各种有用特性的高级线程模块。

我们使用“Thread”构造方法来创建一个线程实例。在此示例中,“Thread”方法以元组的形式接受两个输入，即函数名(` target `)及其参数(` args `)。当一个线程开始执行时，这个函数将被执行。当我们实例化“Thread”类时，将自动调用“Thread”构造方法，它将创建一个新线程。但是新线程不会立即开始执行，这是一个很有价值的同步特性，一旦所有线程都被分配，我们就可以启动线程。

要开始线程执行，我们需要分别调用每个线程实例的“start”方法。因此，这两行同时执行 square 和 cube 线程:

```py
square_thread.start()
cube_thread.start()
```

我们需要做的最后一件事是调用“join”方法，它告诉一个线程等待，直到另一个线程的执行完成:

```py
square_thread.join()
cube_thread.join()
```

当`sleep`方法暂停执行`calc_square()`函数 0.1 秒时，`calc_cube()`函数被执行并打印出列表中某个值的立方，然后进入休眠状态，`calc_square()`函数将被执行。换句话说，操作系统在线程之间来回切换，每次运行一点点，这导致了整个进程运行时间的改进。

### `threading`方法

Python `threading`模块提供了一些有用的函数，帮助您高效地管理我们的多线程程序:

| * *方法名** | * *描述** | 方法的结果 |
| --- | --- | --- |
| threading.active_count() | 返回当前活动线程的数量 | 8 |
| threading . current _ thread() | 返回当前线程，对应于调用者的控制线程 | < _MainThread(主线程，已启动 4303996288) > |
| threading.enumerate() | 返回当前所有活动线程的列表，包括主线程；被终止的线程和尚未启动的线程被排除在外 | [ < _MainThread(MainThread，已启动 4303996288) >，<Thread(Thread-4(_ Thread _ main)，已启动守护进程 6182760448) >，<heart(Thread-5，已启动守护进程 6199586816) >，<Thread(Thread-6(_ watch _ pipe _ FD)，已启动守护进程 6217560064) >，】 |
| threading.main_thread() | 返回主线程 | < _MainThread(主线程，已启动 4303996288) > |

### 结论

多线程是高级编程中实现高性能应用程序的一个广泛概念，本教程涉及 Python 中多线程的基础知识。我们讨论了并发和并行计算的基本术语，然后实现了一个基本的多线程程序来计算数字列表的平方和立方。我们将在以后讨论一些更高级的多线程技术。