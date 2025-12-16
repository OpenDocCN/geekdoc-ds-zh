# 11.1 Python 中的文件输入和输出🔗

> 原文：[`dcic-world.org/2025-08-27/python-fileio.html`](https://dcic-world.org/2025-08-27/python-fileio.html)

|   11.1.1 基本文件操作 |
| --- |
|   11.1.2 逐步读取 CSV 文件 |
|   11.1.3 处理和过滤数据 |
|   11.1.4 写入 CSV 文件 |

在 Pandas 简介 中，我们加载了来自 CSV（逗号分隔值）文件的数据，但我们让 Pandas 处理底层细节：读取文件并将内容转换为 DataFrame。

在本章中，我们将学习使用 Python 的基本文件操作来读取和写入文件，以简化的 CSV 处理器为例。

虽然 Pandas 当然可以完成我们将在本章中做的一切（以及更多），但了解文件操作是如何工作的有助于你成为一个更全面的程序员，也许有一天你可能会创建或参与像 Pandas 这样的库。

#### 11.1.1 基本文件操作🔗 "链接到此处")

Python 提供了用于处理文件的内置函数。在你可以对文件进行任何其他操作之前，你必须“打开”它：

```py
file = open('data.csv', 'r')
```

这条语句以读取模式 (`'r'`) 打开一个名为 `'data.csv'` 的文件。`open` 函数返回一个文件对象，我们可以用它来读取文件（如果以适当的模式打开）或写入文件。

> 现在就做！
> 
> > 你认为如果你尝试打开一个不存在的文件会发生什么？试一试，看看你会得到什么错误信息。

一旦我们有了文件对象，我们可以通过几种方式读取其内容：

重要：文件对象会记住它在文件中的位置。这就是为什么调用 `.readline()` 两次不会返回相同的行，它会返回一行，然后是下一行。但是，这也意味着如果你运行了 `.read()` 或 `.readlines()`，你就已经读取了整个文件，这意味着文件对象的当前位置现在在文件末尾，这意味着调用任何其他方法都会返回空结果——`.read()` 返回空字符串，`.readline()` 返回空列表。你可以使用 `.seek()` 移动文件对象指向的位置，但它的具体工作原理超出了我们的讨论范围！

```py
# Read the entire file as one string
content = file.read()

# Or, we can read one line at a time
line = file.readline()
another_line = file.readline()

# We can also read all remaining lines into a list of strings
all_lines = file.readlines()
```

当我们完成文件的使用时，我们始终应该关闭它：

```py
file.close()
```

> 现在就做！
> 
> > 你认为为什么在完成文件操作后关闭文件可能很重要？

关闭文件很重要，因为它可以释放系统资源并确保，如果我们正在向文件写入（与这个例子不同，我们只是在读取），所有挂起的写入实际上都会被保存！然而，手动记住关闭文件可能会出错。Python 提供了一种更可靠的方法，使用 `with` 语句：

```py
with open('data.csv', 'r') as file:
    content = file.read()
    # file is automatically closed when this block ends
```

除了不让我们记住关闭文件外，这种方法还保证了即使在处理文件时发生错误，文件也会被关闭。

#### 11.1.2 逐步读取 CSV 文件🔗 "链接至此")

让我们手动处理读取 CSV 文件，作为练习使用文件的一个实际（尽管规模较小）的例子。

假设我们有一个名为 `orders.csv` 的文件，其内容如下：（你可以使用你选择的编辑器创建此文件，例如 VSCode）。

| 菜品，数量，价格，订单类型 |
| --- |
| 披萨，2，25.0，堂食 |
| 沙拉，1，8.75，外卖 |
| 汉堡，3，30.0，堂食 |
| 披萨，1，12.50，外卖 |

这是我们可以逐步读取和解析此文件的方法：

```py
# Step 1: Open and read the file into variable `lines`
with open('orders.csv', 'r') as file:
    lines = file.readlines()

# Step 2: Clean data: remove newline characters and split by commas
data = []
for line in lines:
    cells = line.strip().split(',')
    data.append(cells)

# Step 3: Separate header (first row) from data rows (rest of file)
header = data[0]
rows = data[1:]

print("Header:", header)
print("First row:", rows[0])
```

让我们分解每个步骤的作用：

1.  `file.readlines()` 将文件的所有行读取到字符串列表中

1.  我们使用 for 循环遍历每一行，使用 `line.strip()` 从每行的末尾移除换行符 (`'\n'`)，然后通过 `.split(',')` 将行转换为字符串列表，该字符串通过给定的字符串（不包括）分割：

1.  我们将第一行（标题）与数据行分开，以便更容易处理——表示 `data[1:]` 是一种特殊的方式，表示我们想要 "从索引 1 到列表的末尾——即，列表的末尾"。

> 立刻行动！
> 
> > 如果你的 CSV 中的一个单元格包含逗号，你的代码会做什么？例如，如果菜品名称是 "Mac and cheese, deluxe"，你将如何处理这种情况？

#### 11.1.3 处理和过滤数据🔗 "链接至此")

一旦我们将数据作为列表的列表形式，我们就可以使用我们学过的相同编程技术来处理它，通过使用 `.index()` 方法来返回给定字符串在字符串列表中的数字偏移量——这就是我们将如何找到我们感兴趣的列，然后使用它来索引行。

例如，让我们过滤出只有外卖订单：

```py
# Returns the index (i.e., offset, base 0) where 'order_type' exists in the header list.
order_type_index = header.index('order_type')

# Filter for takeout orders
takeout_orders = []
for row in rows:
    if row[order_type_index] == 'takeout':
        takeout_orders.append(row)

print("Found " + str(len(takeout_orders)) + " takeout orders")
```

我们还可以根据需要转换数据类型。例如，如果我们想计算总收入，我们不仅需要找到每行的数量和价格，还需要在乘法之前将行中的字符串转换为数字：

```py
quantity_index = header.index('quantity')
price_index = header.index('price')

total_revenue = 0
for row in rows:
    quantity = int(row[quantity_index])
    price = float(row[price_index])
    total_revenue += quantity * price

print("Total revenue: $" + str(total_revenue))
```

> 立刻行动！
> 
> > 如果其中一个数量单元格包含无效数据，比如字符串 "three" 而不是数字 3，会发生什么？你如何使你的代码更加健壮以处理此类错误？

#### 11.1.4 写入 CSV 文件🔗 "链接至此")

写入 CSV 文件遵循类似的模式。我们需要：

1.  以写入模式打开文件

1.  将我们的数据转换为适当的字符串格式

1.  将字符串写入文件

这是将我们的过滤后的外卖订单写入新文件的方法：

```py
# Prepare data to write (header + filtered rows)
output_data = [header] + takeout_orders

# Write to file
with open('takeout_orders.csv', 'w') as file:
    for row in output_data:
        # Join the row elements with commas and add a newline
        line = ','.join(row) + '\n'
        file.write(line)
```

这里关键步骤是：

+   `','.join(row)` 将列表元素合并成一个带有逗号的单独字符串

+   我们添加 `'\n'` 以在每行后创建一个新行

+   `file.write()` 将字符串写入文件

注意，我们为每一行调用一次 `.write()` ——我们本可以将所有行合并成一个字符串，然后只调用一次 `.write()`，但这样做没有必要——就像文件对象会记住我们从哪里读取一样，它们也会记住我们曾经在哪里写入，所以下一次调用 `.write()` 将在上一行的字符串之后添加下一个字符串。

> 现在就做！
> 
> > 尝试编写一个程序，读取 CSV 文件，添加一个包含计算值的新列（如总价 = 数量 × 价格），并将结果写入新文件。

#### 11.1.1 基本文件操作🔗 "链接到此处")

Python 提供了用于处理文件的内置函数。在你可以对文件进行任何其他操作之前，你必须“打开”它：

```py
file = open('data.csv', 'r')
```

这个语句以读取模式（`'r'`）打开一个名为 `'data.csv'` 的文件。`open` 函数返回一个文件对象，我们可以用它来读取文件（如果以适当的模式打开）或写入文件。

> 现在就做！
> 
> > 你认为如果你尝试打开一个不存在的文件会发生什么？试一试，看看你会得到什么错误信息。

一旦我们有了文件对象，我们可以通过几种方式读取其内容：

重要：文件对象会记住它在文件中的读取位置。这就是为什么调用 `.readline()` 两次不会返回相同的行，它会返回一行，然后是下一行。但是，这也意味着如果你运行了 `.read()` 或 `.readlines()`，你就已经读取了整个文件，这意味着文件对象的当前位置现在在文件末尾，这意味着调用任何其他方法都会返回空结果——`.read()` 返回空字符串，`.readline()` 返回空列表。你可以使用 `.seek()` 移动文件对象指向的位置，但它的具体工作原理超出了我们的讨论范围！

```py
# Read the entire file as one string
content = file.read()

# Or, we can read one line at a time
line = file.readline()
another_line = file.readline()

# We can also read all remaining lines into a list of strings
all_lines = file.readlines()
```

当我们完成文件的使用后，我们应始终关闭它：

```py
file.close()
```

> 现在就做！
> 
> > 你认为为什么在完成文件使用后关闭文件可能很重要？

关闭文件很重要，因为它可以释放系统资源，并确保如果我们正在写入文件（与这个例子中我们只读取不同），所有挂起的写入实际上都会被保存！然而，手动记住关闭文件可能会出错。Python 提供了一种更可靠的方法，使用 `with` 语句：

```py
with open('data.csv', 'r') as file:
    content = file.read()
    # file is automatically closed when this block ends
```

除了不需要我们记住关闭文件外，这种方法还保证了即使在处理文件时发生错误，文件也会被关闭。

#### 11.1.2 逐步读取 CSV 文件🔗 "链接到此处")

让我们通过手动读取 CSV 文件来练习使用文件，作为一个实际（尽管规模较小）的例子。

假设我们有一个名为 `orders.csv` 的文件，其内容如下：（你可以使用你选择的编辑器创建此文件，例如 VSCode）。

| 菜品,数量,价格,订单类型 |
| --- |
| 披萨,2,25.0,堂食 |
| 沙拉,1,8.75,外卖 |
| 汉堡,3,30.0,堂食 |
| 披萨,1,12.50,外卖 |

下面是如何逐步读取和解析这个文件的步骤：

```py
# Step 1: Open and read the file into variable `lines`
with open('orders.csv', 'r') as file:
    lines = file.readlines()

# Step 2: Clean data: remove newline characters and split by commas
data = []
for line in lines:
    cells = line.strip().split(',')
    data.append(cells)

# Step 3: Separate header (first row) from data rows (rest of file)
header = data[0]
rows = data[1:]

print("Header:", header)
print("First row:", rows[0])
```

让我们分解每个步骤的作用：

1.  `file.readlines()` 读取文件中的所有行到一个字符串列表中

1.  我们使用 for 循环遍历每一行，使用 `line.strip()` 从每一行的末尾移除换行符 (`'\n'`)，然后通过 `.split(',')` 将行转换为字符串列表，该操作通过给定的字符串（不包括）来分割字符串。

1.  我们将第一行（标题行）与数据行分开，以便更容易处理——表示法 `data[1:]` 是一种特殊的方式，表示我们想要 "从索引 1 开始直到列表的末尾——即，列表的末尾。

> 立即行动！
> 
> > 如果你的 CSV 中的一个单元格包含逗号，我们的代码会做什么？例如，如果一道菜名是 "Mac and cheese, deluxe"，该如何处理这种情况？

#### 11.1.3 处理和筛选数据🔗 "链接到此处")

一旦我们将数据作为列表的列表，我们就可以使用我们学到的相同编程技术来处理它，通过使用 `.index()` 方法返回给定字符串在字符串列表中的数字偏移量——这就是我们将如何找到我们感兴趣的列，然后使用它来索引行。

例如，让我们筛选出仅外卖订单：

```py
# Returns the index (i.e., offset, base 0) where 'order_type' exists in the header list.
order_type_index = header.index('order_type')

# Filter for takeout orders
takeout_orders = []
for row in rows:
    if row[order_type_index] == 'takeout':
        takeout_orders.append(row)

print("Found " + str(len(takeout_orders)) + " takeout orders")
```

我们还可以根据需要转换数据类型。例如，如果我们想计算总收入，我们不仅需要找到每一行的数量和价格，还需要在乘法之前将行中的字符串转换为数字：

```py
quantity_index = header.index('quantity')
price_index = header.index('price')

total_revenue = 0
for row in rows:
    quantity = int(row[quantity_index])
    price = float(row[price_index])
    total_revenue += quantity * price

print("Total revenue: $" + str(total_revenue))
```

> 立即行动！
> 
> > 如果其中一个数量单元格包含无效数据，比如字符串 "three" 而不是数字 3，会发生什么？你该如何使你的代码更加健壮以处理此类错误？

#### 11.1.4 编写 CSV 文件🔗 "链接到此处")

编写 CSV 文件遵循类似的模式。我们需要：

1.  以写入模式打开文件

1.  将我们的数据转换为适当的字符串格式

1.  将字符串写入文件

这是将我们的筛选外卖订单写入新文件的方法：

```py
# Prepare data to write (header + filtered rows)
output_data = [header] + takeout_orders

# Write to file
with open('takeout_orders.csv', 'w') as file:
    for row in output_data:
        # Join the row elements with commas and add a newline
        line = ','.join(row) + '\n'
        file.write(line)
```

这里关键步骤是：

+   `','.join(row)` 将列表元素合并成一个带有逗号的单一字符串

+   我们添加 `'\n'` 在每一行后创建一个新行

+   `file.write()` 将字符串写入文件

注意，我们为每一行调用 `.write()` 一次——我们本可以将所有行合并成一个字符串，然后只调用一次 `.write()`，但这样做没有必要——就像文件对象记得我们从它们那里读取的位置一样，它们也记得我们写入的位置，所以下一次调用 `.write()` 将在上一条字符串之后添加下一条字符串。

> 立即行动！
> 
> > 尝试编写一个程序，该程序读取 CSV 文件，添加一个包含计算值的新列（如总价 = 数量 × 价格），并将结果写入新文件。
