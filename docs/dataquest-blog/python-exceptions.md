# Python 异常:终极初学者指南(带示例)

> 原文：<https://www.dataquest.io/blog/python-exceptions/>

June 7, 2022![Python exceptions](img/06a09a06714ab9f85d27c68bfa89cde2.png)

## 本教程涵盖了 Python 中的异常，包括为什么会出现异常、如何识别异常以及如何解决异常。

在本教程中，我们将在 Python 中定义异常，我们将确定它们为什么重要以及它们与语法错误有何不同，我们将学习如何在 Python 中解决异常。

### Python 中有哪些异常？

当运行 Python 代码时遇到意外情况时，程序会停止执行并抛出错误。Python 中基本上有两类错误:**语法错误**和**异常**。为了理解这两种类型之间的区别，让我们运行下面这段代码:

```py
print(x
print(1)
```

```py
 File "C:\Users\Utente\AppData\Local\Temp/ipykernel_4732/4217672763.py", line 2
    print(1)
    ^
SyntaxError: invalid syntax
```

由于我们忘记了关闭括号，引发了语法错误。当我们在 Python 中使用语法不正确的语句时，总是会出现这种错误。解析器通过一个小箭头`^`显示检测到语法错误的地方。还要注意，随后的行`print(1)`没有被执行，因为错误发生时 Python 解释器停止了工作。

让我们修复这个错误并重新运行代码:

```py
print(x)
print(1)
```

```py
---------------------------------------------------------------------------

NameError                                 Traceback (most recent call last)

~\AppData\Local\Temp/ipykernel_4732/971139432.py in <module>
----> 1 print(x)
      2 print(1)

NameError: name 'x' is not defined
```

现在，当我们修复了错误的语法后，我们得到了另一种类型的错误:异常。换句话说，**异常是当语法正确的 Python 代码产生错误**时发生的一种错误。箭头指示发生异常的行，而错误信息的最后一行指定了异常的确切类型，并提供了其描述以方便调试。在我们的例子中，它是一个`NameError`，因为我们试图打印一个之前没有定义的变量`x`的值。同样在这种情况下，我们的第二行代码`print(1)`没有被执行，因为 Python 程序的正常流程被中断了。

为了防止程序突然崩溃，捕捉和处理异常非常重要。例如，当给定的异常发生时，提供代码执行的替代版本。这是我们接下来要学的。

### 标准内置类型的异常

Python 提供了在各种情况下抛出的多种类型的异常。让我们看看最常见的内置异常及其示例:

*   **`NameError`**–当名字不存在于局部或全局变量中时引发:

```py
print(x)
```

```py
---------------------------------------------------------------------------

NameError                                 Traceback (most recent call last)

~\AppData\Local\Temp/ipykernel_4732/1353120783.py in <module>
----> 1 print(x)

NameError: name 'x' is not defined
```

*   **`TypeError`**–对不适用的数据类型运行操作时引发:

```py
print(1+'1')
```

```py
---------------------------------------------------------------------------

TypeError                                 Traceback (most recent call last)

~\AppData\Local\Temp/ipykernel_4732/218464413.py in <module>
----> 1 print(1+'1')

TypeError: unsupported operand type(s) for +: 'int' and 'str'
```

*   **`ValueError`**–当操作或函数接受无效的参数值时引发:

```py
print(int('a'))
```

```py
---------------------------------------------------------------------------

ValueError                                Traceback (most recent call last)

~\AppData\Local\Temp/ipykernel_4732/3219923713.py in <module>
----> 1 print(int('a'))

ValueError: invalid literal for int() with base 10: 'a'
```

*   **`IndexError`**–在 iterable 中不存在索引时引发:

```py
print('dog'[3])
```

```py
---------------------------------------------------------------------------

IndexError                                Traceback (most recent call last)

~\AppData\Local\Temp/ipykernel_4732/2163152480.py in <module>
----> 1 print('dog'[3])

IndexError: string index out of range
```

*   **`IndentationError`**–缩进不正确时引发:

```py
for i in range(3):
print(i)
```

```py
 File "C:\Users\Utente\AppData\Local\Temp/ipykernel_4732/3296739069.py", line 2
    print(i)
    ^
IndentationError: expected an indented block
```

*   **`ZeroDivisionError`**–在试图将一个数除以零时引发:

```py
print(1/0)
```

```py
---------------------------------------------------------------------------

ZeroDivisionError                         Traceback (most recent call last)

~\AppData\Local\Temp/ipykernel_4732/165659023.py in <module>
----> 1 print(1/0)

ZeroDivisionError: division by zero
```

*   **`ImportError`**–导入语句不正确时引发:

```py
from numpy import pandas
```

```py
---------------------------------------------------------------------------

ImportError                               Traceback (most recent call last)

~\AppData\Local\Temp/ipykernel_4732/3955928270.py in <module>
----> 1 from numpy import pandas

ImportError: cannot import name 'pandas' from 'numpy' (C:\Users\Utente\anaconda3\lib\site-packages\numpy\__init__.py)
```

*   **`AttributeError`**–在试图分配或引用不适用于给定 Python 对象的属性时引发:

```py
print('a'.sum())
```

```py
---------------------------------------------------------------------------

AttributeError                            Traceback (most recent call last)

~\AppData\Local\Temp/ipykernel_4732/2316121794.py in <module>
----> 1 print('a'.sum())

AttributeError: 'str' object has no attribute 'sum'
```

*   **`KeyError`**–字典中没有该键时引发:

```py
animals = {'koala': 1, 'panda': 2}
print(animals['rabbit'])
```

```py
---------------------------------------------------------------------------

KeyError                                  Traceback (most recent call last)

~\AppData\Local\Temp/ipykernel_4732/1340951462.py in <module>
      1 animals = {'koala': 1, 'panda': 2}
----> 2 print(animals['rabbit'])

KeyError: 'rabbit'
```

关于 Python 内置异常的完整列表，请参考 [Python 文档](https://docs.python.org/3/library/exceptions.html)。

## 在 Python 中处理异常

由于引发异常会导致程序执行的中断，我们必须提前处理这个异常以避免这种不良情况。

### `try`和`except`语句

Python 中用于检测和处理异常的最基本命令是`try`和`except`。

`try`语句用于运行一段容易出错的代码，并且必须始终跟有`except`语句。如果`try`块执行后没有出现异常，则`except`块被跳过，程序按预期运行。在相反的情况下，如果抛出异常，`try`块的执行会立即停止，程序通过运行`except`块中确定的替代代码来处理引发的异常。之后，Python 脚本继续工作并执行剩余的代码。

让我们通过我们最初的一小段代码`print(x)`的例子来看看它是如何工作的，它在前面提出了一个`NameError`:

```py
try:
    print(x)
except:
    print('Please declare the variable x first')

print(1)
```

```py
Please declare the variable x first
1
```

既然我们在`except`块中处理了异常，我们就收到了一条有意义的定制消息，告诉我们到底哪里出了问题以及如何修复。更何况这一次，程序并不是一遇到异常就停止工作，执行剩下的代码。

在上面的例子中，我们只预测和处理了一种类型的异常，更具体地说，是一个`NameError`。这种方法的缺点是，`except`子句中的这段代码将以相同的方式对待**所有类型的异常**，并输出相同的消息`Please declare the variable x first`。为了避免这种混淆，我们可以在`except`命令之后明确指出需要捕捉和处理的异常类型:

```py
try:
    print(x)
except NameError:
    print('Please declare the variable x first')
```

```py
Please declare the variable x first
```

#### 处理多个异常

清楚地说明要捕获的异常类型不仅是为了代码的可读性。更重要的是，使用这种方法，我们可以预测各种特定的异常，并相应地处理它们。

为了理解这个概念，我们来看一个简单的函数，它总结了一个输入字典的值:

```py
def print_dict_sum(dct):
    print(sum(dct.values()))

my_dict = {'a': 1, 'b': 2, 'c': 3}
print_dict_sum(my_dict)
```

```py
6
```

尝试运行这个函数时，如果我们不小心向它传递了一个错误的输入，我们可能会遇到不同的问题。例如，我们可以在字典名称中犯一个错误，导致一个不存在的变量:

```py
print_dict_sum(mydict)
```

```py
---------------------------------------------------------------------------

NameError                                 Traceback (most recent call last)

~\AppData\Local\Temp/ipykernel_4732/2473187932.py in <module>
----> 1 print_dict_sum(mydict)

NameError: name 'mydict' is not defined
```

输入字典的某些值可以是字符串而不是数字:

```py
my_dict = {'a': '1', 'b': 2, 'c': 3}
print_dict_sum(my_dict)
```

```py
---------------------------------------------------------------------------

TypeError                                 Traceback (most recent call last)

~\AppData\Local\Temp/ipykernel_4732/2621846538.py in <module>
      1 my_dict = {'a': '1', 'b': 2, 'c': 3}
----> 2 print_dict_sum(my_dict)

~\AppData\Local\Temp/ipykernel_4732/3241128871.py in print_dict_sum(dct)
      1 def print_dict_sum(dct):
----> 2     print(sum(dct.values()))
      3 
      4 my_dict = {'a': 1, 'b': 2, 'c': 3}
      5 print_dict_sum(my_dict)

TypeError: unsupported operand type(s) for +: 'int' and 'str'
```

另一个选项允许我们为此函数传入一个不合适的数据类型的参数:

```py
my_dict = 'a'
print_dict_sum(my_dict)
```

```py
---------------------------------------------------------------------------

AttributeError                            Traceback (most recent call last)

~\AppData\Local\Temp/ipykernel_4732/1925769844.py in <module>
      1 my_dict = 'a'
----> 2 print_dict_sum(my_dict)

~\AppData\Local\Temp/ipykernel_4732/3241128871.py in print_dict_sum(dct)
      1 def print_dict_sum(dct):
----> 2     print(sum(dct.values()))
      3 
      4 my_dict = {'a': 1, 'b': 2, 'c': 3}
      5 print_dict_sum(my_dict)

AttributeError: 'str' object has no attribute 'values'
```

因此，我们至少有三种不同类型的异常应该被不同地处理:`NameError`、`TypeError`和`AttributeError`。为此，我们可以在单个`try`块之后添加多个`except`块(每个异常类型一个，在我们的例子中是三个):

```py
try: 
    print_dict_sum(mydict)
except NameError:
    print('Please check the spelling of the dictionary name')
except TypeError:
    print('It seems that some of the dictionary values are not numeric')
except AttributeError:
    print('You should provide a Python dictionary with numeric values')
```

```py
Please check the spelling of the dictionary name
```

在上面的代码中，我们在`try`子句中提供了一个不存在的变量名作为函数的输入。代码应该抛出一个`NameError`,但是它在一个后续的`except`子句中被处理，相应的消息被输出。

我们也可以在函数定义中处理异常。**重要:**我们不能为任何函数参数处理`NameError`异常，因为在这种情况下，异常发生在函数体开始之前。例如，在下面的代码中:

```py
def print_dict_sum(dct):
    try:
        print(sum(dct.values()))
    except NameError:
        print('Please check the spelling of the dictionary name')
    except TypeError:
        print('It seems that some of the dictionary values are not numeric')
    except AttributeError:
        print('You should provide a Python dictionary with numeric values')

print_dict_sum({'a': '1', 'b': 2, 'c': 3})
print_dict_sum('a')
print_dict_sum(mydict)
```

```py
It seems that some of the dictionary values are not numeric
You should provide a Python dictionary with numeric values

---------------------------------------------------------------------------

NameError                                 Traceback (most recent call last)

~\AppData\Local\Temp/ipykernel_4732/3201242278.py in <module>
     11 print_dict_sum({'a': '1', 'b': 2, 'c': 3})
     12 print_dict_sum('a')
---> 13 print_dict_sum(mydict)

NameError: name 'mydict' is not defined
```

函数内部成功处理了`TypeError`和`AttributeError`，并输出了相应的消息。相反，由于上述原因，`NameError`尽管引入了单独的`except`条款，却没有得到妥善处理。因此，函数*的任何参数的`NameError`都不能在函数体*内部处理。

如果以相同的方式处理，可以将几个异常组合成一个元组放在一个`except`子句中:

```py
def print_dict_sum(dct):
    try:
        print(sum(dct.values()))
    except (TypeError, AttributeError):
        print('You should provide a Python DICTIONARY with NUMERIC values')

print_dict_sum({'a': '1', 'b': 2, 'c': 3})
print_dict_sum('a')
```

```py
You should provide a Python DICTIONARY with NUMERIC values
You should provide a Python DICTIONARY with NUMERIC values
```

#### `else`声明

除了`try`和`except`子句，我们可以使用可选的`else`命令。如果存在，`else`命令必须放在所有`except`子句之后，并且只有在`try`子句中没有出现异常时才执行。

例如，在下面的代码中，我们尝试除以零:

```py
try:
    print(3/0)
except ZeroDivisionError:
    print('You cannot divide by zero')
else:
    print('The division is successfully performed')
```

```py
You cannot divide by zero
```

异常在`except`块中被捕获并处理，因此`else`子句被跳过。让我们看看如果我们提供一个非零数字会发生什么:

```py
try:
    print(3/2)
except ZeroDivisionError:
    print('You cannot divide by zero')
else:
    print('The division is successfully performed')
```

```py
1.5
The division is successfully performed
```

因为没有出现异常，所以执行了`else`块并输出相应的消息。

#### `finally`声明

另一个可选语句是`finally`，如果提供，它必须放在包括`else`(如果存在)
在内的所有条款之后，并且在任何情况下都要执行，不管`try`条款中是否提出了例外。

让我们将`finally`块添加到前面的两段代码中，观察结果:

```py
try:
    print(3/0)
except ZeroDivisionError:
    print('You cannot divide by zero')
else:
    print('The division is successfully performed')
finally:
    print('This message is always printed')
```

```py
You cannot divide by zero
This message is always printed
```

```py
try:
    print(3/2)
except ZeroDivisionError:
    print('You cannot divide by zero')
else:
    print('The division is successfully performed')
finally:
    print('This message is always printed')
```

```py
1.5
The division is successfully performed
This message is always printed
```

在第一种情况下，出现了异常，在第二种情况下，没有出现异常。然而，在这两种情况下，`finally`子句输出相同的消息。

## 引发异常

有时，我们可能需要故意引发一个异常，并在某个条件发生时停止程序。为此，我们需要`raise`关键字和以下语法:

```py
raise ExceptionClass(exception_value)
```

在上面，`ExceptionClass`是要引发的异常的类型(例如，`TypeError`),`exception_value`是可选的定制描述性消息，如果引发异常，将显示该消息。

让我们看看它是如何工作的:

```py
x = 'blue'
if x not in ['red', 'yellow', 'green']:
    raise ValueError
```

```py
---------------------------------------------------------------------------

ValueError                                Traceback (most recent call last)

~\AppData\Local\Temp/ipykernel_4732/2843707178.py in <module>1 x = 'blue'
      2 if x not in ['red', 'yellow', 'green']:
----> 3     raise ValueError

ValueError: 
```

在上面这段代码中，我们没有为异常提供任何参数，因此代码没有输出任何消息(默认情况下，异常值为`None`)。

```py
x = 'blue'
if x not in ['red', 'yellow', 'green']:
    raise ValueError('The traffic light is broken')
```

```py
---------------------------------------------------------------------------

ValueError                                Traceback (most recent call last)

~\AppData\Local\Temp/ipykernel_4732/359420707.py in <module>
      1 x = 'blue'
      2 if x not in ['red', 'yellow', 'green']:
----> 3     raise ValueError('The traffic light is broken')

ValueError: The traffic light is broken
```

我们运行了相同的代码，但是这次我们提供了异常参数。在这种情况下，我们可以看到一条输出消息，该消息提供了更多的上下文信息，以说明发生该异常的确切原因。

### 结论

在本教程中，我们讨论了 Python 中异常的许多方面。特别是，我们了解到以下情况:

*   如何在 Python 中定义异常，以及它们与语法错误有何不同
*   Python 中存在哪些内置异常，何时引发
*   为什么捕捉和处理异常很重要
*   如何在 Python 中处理一个或多个异常
*   捕捉和处理异常的不同子句如何协同工作
*   为什么指定要处理的异常类型很重要
*   为什么我们不能在函数定义中为函数的任何参数处理一个`NameError`
*   如何引发异常
*   如何向引发的异常添加描述性消息，以及为什么这是一种好的做法

有了这些技能，您就可以处理任何需要在 Python 中解决异常的真实世界的数据科学任务。