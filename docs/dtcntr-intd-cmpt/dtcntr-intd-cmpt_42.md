# 13.2 可变列表🔗

> 原文：[`dcic-world.org/2025-08-27/mutable-lists.html`](https://dcic-world.org/2025-08-27/mutable-lists.html)

|  |
| --- |

让我们再次扩展我们对更新的研究，这次是查看更新列表。我们将从列表开始。

想象一下，Shaunae 想要使用一个程序来维护她的购物清单。她创建了一个包含两个项目的初始列表：

```py
shaunae_list = ["bread", "coffee"]
```

> 现在就做！
> 
> > Shaunae 想要向她的清单中添加鸡蛋。请编写一行代码来完成这个任务。

你可以有两种方法来完成这个任务：

```py
# approach 1
shaunae_list = shaunae_list + ["eggs"]

#approach 2
shaunae_list.append("eggs")
```

这两种方法之间的区别是什么？区别在于对堆的影响。

+   第一个版本创建了一个包含 `"eggs"` 的新列表，然后将两个列表的元素合并到一个新列表中。

+   第二个版本将 `"eggs"` 插入堆中现有的列表中。

让我们看看每个版本的目录。以下是第一个版本的最终目录：

目录

+   ```py
    shaunae_list
    ```

    → 1010

堆

+   1005: `List(len:2)`

+   1006: `"bread"`

+   1007: `"coffee"`

+   1008: `List(len:1)`

+   1009: `"eggs"`

+   1010: `List(len:3)`

+   1011: `"bread"`

+   1012: `"coffee"`

+   1013: `"eggs"`

`shaunae_list` 的原始版本在地址 1005，包含 `"eggs"` 的列表在 1008，合并后的列表在 1010。

相比之下，第二个版本的最终目录将如下所示：

目录

+   ```py
    shaunae_list
    ```

    → 1010

堆

+   1005: `List(len:3)`

+   1006: `"bread"`

+   1007: `"coffee"`

+   1008: `"eggs"`

注意这里，原始列表的长度和内容已更改，以包含新添加的 `"eggs"`。

> 现在就做！
> 
> > 你认为哪种方法更好？为什么？

乍一看，第二种方法可能看起来更好，因为它不会创建额外的非必要列表。两种方法在 `shaunae_List` 中的内容相同，所以使用额外空间似乎没有多少好处。

除非，当然，我们希望在以后仍然能够访问 `shaunae_list` 的旧版本。旧列表仍然在堆中（尽管我们的当前程序没有名称可以通过它来访问那个旧列表）。如果我们以这种方式编写程序会怎样呢？

```py
shaunae_list = ["bread", "coffee"]
prev_list = shaunae_list
shaunae_list = ["paint", "brushes"] + shaunae_list
```

现在，如果 Shaunae 发现她在最后一次更新中将艺术用品购物清单放在了杂货清单上，她可以通过将她的列表变量重置为之前的列表来“撤销”更新：

```py
shaunae_list = prev_list
```

撤销修改（就像文档编辑工具中的撤销功能一样）只是说明保留旧版本数据一段时间可以有所帮助的一个例子。这里的重点不是对撤销计算进行复杂处理，而是要说明在某些情况下，创建新列表比更新旧列表更可取。

我们何时想要更新而不是保留现有列表？

记得我们讨论的别名问题吗？我们想要两个人，Elena 和 Jorge，共享对一个共同银行账户的访问权限。我们是否可能想要一个共享的购物清单？当然，Shaunae 和她的室友 Jonella 就共享一个购物清单，这样他们都可以添加项目，同时让其中一个人去商店。

> 现在就做！
> 
> > 设置一个共享的购物清单，可以通过两个名称`shaunae_list`和`jonella_list`访问。然后，通过其中一个名称添加一个项目到列表中，并检查该项目是否出现在另一个名称下。

你可能已经编写了如下内容：

```py
shaunae_list = ["bread", "coffee"]
jonella_list = shaunae_list
jonella_list.append("eggs")
```

如果你将此代码加载到提示符中并查看末尾的两个列表，你会看到它们具有相同的值。

相反，如果我们像以下这样编写代码，那么只有一个列表会看到新项目：

```py
>>> jonella_list = ["apples"] + jonella_list
>>> jonella_list
["apples", "bread", "coffee", "eggs"]
>>> shaunae_list
["bread", "coffee", "eggs"]
```

> 现在行动！
> 
> > 为上述程序绘制内存图。

#### 练习：创建账户列表🔗 "链接到此处")

在函数内修改顶级变量中，我们编写了一个函数来为银行创建新账户。该函数在创建新账户时返回每个新账户。这意味着每个新创建的账户都必须与目录中的名称相关联（否则我们无法从堆中访问它）。

维护所有创建的账户的列表或字典更有意义。我们只需要一个名称来表示账户集合，但仍然可以根据需要访问单个账户。例如，我们可能想要一个类似以下内容的`all_accts`列表：

```py
all_accts = [Account(8623, 100),
             Account(8624, 300),
             Account(8625, 225),
             ...
             ]
```

> 现在行动！
> 
> > 编写一个程序，创建一个空的`all_accts`列表，然后每次调用`create_acct`时向其中添加一个新的`Account`。你需要修改`create_acct`来实现这一点。以下是作为起点的现有代码。
> > 
> > ```py
> > next_id = 1
> > 
> > def create_acct(init_bal: float) -> Account:
> >   global next_id
> >   new_acct = Account(next_id, init_bal, [])
> >   next_id = next_id + 1
> >   return new_acct
> > ```
> > 
> 现在行动！
> 
> > 你在代码中包含了一行`global all_accts`吗？为什么或为什么不包含？

如果你使用`append`来更新`all_accts`列表，那么你就不需要包含`global all_accts`。回想一下，`global`是必需的，以告诉 Python 更新顶级目录中的变量而不是局部目录中的变量。然而，如果你使用`all_accts.append`，你是在修改堆而不是目录。如果你的代码只修改堆内容，则不需要`global`。

#### 练习：创建账户列表🔗 "链接到此处")

在函数内修改顶级变量中，我们编写了一个函数来为银行创建新账户。该函数在创建新账户时返回每个新账户。这意味着每个新创建的账户都必须与目录中的名称相关联（否则我们无法从堆中访问它）。

维护所有创建的账户的列表或字典更有意义。我们只需要一个名称来表示账户集合，但仍然可以根据需要访问单个账户。例如，我们可能想要一个类似以下内容的`all_accts`列表：

```py
all_accts = [Account(8623, 100),
             Account(8624, 300),
             Account(8625, 225),
             ...
             ]
```

> 现在行动！
> 
> > 编写一个程序，创建一个空的 `all_accts` 列表，然后每次调用 `create_acct` 时向其中添加一个新的 `Account`。你需要修改 `create_acct` 来实现这一点。以下是将作为起始点的现有代码。 
> > 
> > ```py
> > next_id = 1
> > 
> > def create_acct(init_bal: float) -> Account:
> >   global next_id
> >   new_acct = Account(next_id, init_bal, [])
> >   next_id = next_id + 1
> >   return new_acct
> > ```
> > 
> 现在就做！
> 
> > 你在代码中包含了一条像 `global all_accts` 这样的行吗？为什么或为什么不包含？

如果你使用了 `append` 来更新 `all_accts` 列表，那么你就不需要包含 `global all_accts`。回想一下，`global` 是用来告诉 Python 更新顶级目录中的变量而不是本地目录中的变量的。然而，如果你使用 `all_accts.append`，你是在修改堆而不是目录。如果你的代码只是修改堆内容，那么不需要 `global`。
