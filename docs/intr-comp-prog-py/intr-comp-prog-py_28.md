# 第二十七章：PYTHON 3.8 快速参考

## 对数值类型的常见操作

`**i+j**` 是 `i` 和 `j` 的和。

`**i–j**` 是 `i` 减去 `j`。

`**i*j**` 是 `i` 和 `j` 的乘积。

`**i//j**` 是向下取整除法。

`**i/j**` 是浮点除法。

`**i%j**` 是整型 `i` 除以整型 `j` 的余数。

`**i**j**` 是 `i` 的 `j` 次幂。

`**x += y**` 等同于 `x = x + y`。`***=**` 和 `**-=**` 也以相同方式工作。

比较运算符有 `==` （等于）、 `!=` （不等于）、 `>` （大于）、 `>=` （至少）、 `<` （小于）和 `<=` （最多）。

## 布尔运算符

`**x == y**` 如果 `x` 和 `y` 相等，则返回 `True`。

`**x != y**` 如果 `x` 和 `y` 不相等，则返回 `True`。

`**<, >, <=, >=**` 具有其通常的含义。

`**a and b**` 如果 `a` 和 `b` 都为 `True`，则为 `True`，否则为 `False`。

`**a or b**` 如果 `a` 或 `b` 至少有一个为 `True`，则为 `True`，否则为 `False`。

`**not a**` 如果 `a` 为 `False`，则为 `True`；如果 `a` 为 `True`，则为 `False`。

## 对序列类型的常见操作

`**seq[i]**` 返回序列中的第 `i` 个元素。

`**len(seq)**` 返回序列的长度。

`**seq1 + seq2**` 连接两个序列。（范围不适用。）

`**n*seq**` 返回一个重复 `seq` `n` 次的序列。（范围不适用。）

`**seq[start:end]**` 返回一个新的序列，它是 `seq` 的切片。

`**e in seq**` 测试 `e` 是否包含在序列中。

`**e not in seq**` 测试 `e` 是否不包含在序列中。

`**for e in seq**` 遍历序列中的元素。

## **常见字符串方法**

`**s.count(s1)**` 计算字符串 `s1` 在 `s` 中出现的次数。

`**s.find(s1)**` 返回子字符串 `s1` 在 `s` 中第一次出现的索引；如果 `s1` 不在 `s` 中，则返回 `-1`。

`**s.rfind(s1)**` 与 `find` 相同，但从 `s` 的末尾开始。

`**s.index(s1)**` 与 `find` 相同，但如果 `s1` 不在 `s` 中，则引发异常。

`**s.rindex(s1)**` 与 `index` 相同，但从 `s` 的末尾开始。

`**s.lower()**` 将所有大写字母转换为小写。

`**s.replace(old, new)**` 将字符串 `old` 的所有出现替换为字符串 `new`。

`**s.rstrip()**` 移除末尾的空白字符。

`**s.split(d)**` 使用 `d` 作为分隔符分割 `s`。返回 `s` 的子字符串列表。

## **常见列表方法**

`**L.append(e)**` 将对象 `e` 添加到列表 `L` 的末尾。

`**L.count(e)**` 返回元素 `e` 在列表 `L` 中出现的次数。

`**L.insert(i, e)**` 在列表 `L` 的索引 `i` 处插入对象 `e`。

`**L.extend(L1)**` 将列表 `L1` 中的项追加到列表 `L` 的末尾。

`**L.remove(e)**` 从列表 `L` 中删除 `e` 的第一次出现。

`**L.index(e)**` 返回 `e` 在列表 `L` 中第一次出现的索引。如果 `e` 不在 `L` 中，则引发 `ValueError`。

`**L.pop(i)**` 移除并返回索引 `i` 处的项；`i` 默认为 `-1`。如果 `L` 为空，则引发 `IndexError`。

`**L.sort()**` 具有对 `L` 中元素进行排序的副作用。

`**L.reverse()**` 具有反转 `L` 中元素顺序的副作用。

`**L.copy()**` 返回 `L` 的浅拷贝。

`**L.deepcopy()**` 返回 `L` 的深拷贝。

## **字典的常见操作**

`**len(d)**` 返回 `d` 中项目的数量。

`**d.keys()**` 返回 `d` 中键的视图。

`**d.values()**` 返回 `d` 中值的视图。

`**d.items()**` 返回 `d` 中的 (键, 值) 对的视图。

`**k in d**` 如果键 `k` 在 `d` 中，则返回 `True`。

`**d[k]**` 返回 `d` 中键为 `k` 的项目。如果 `k` 不在 `d` 中，则引发 `KeyError`。

`**d.get(k, v)**` 如果 `k` 在 `d` 中，则返回 `d[k]`，否则返回 `v`。

`**d[k] = v**` 将值 `v` 关联到键 `k`。如果 `k` 已经关联了一个值，则该值会被替换。

`**del d[k]**` 从 `d` 中删除键为 `k` 的元素。如果 `k` 不在 `d` 中，则引发 `KeyError`。

`**for k in d**` 遍历 `d` 中的键。

## 常见的输入/输出机制

`**input(msg)**` 打印 `msg`，然后返回输入的值作为字符串。

`**print(s1, …, sn)**` 打印字符串 `s1, …, sn`，并用空格分隔。

`**open('file_name', 'w')**` 创建一个用于写入的文件。

`**open('file_name', 'r')**` 打开现有文件以进行读取。

`**open('file_name', 'a')**` 打开现有文件以进行追加。

`**file_handle.read()**` 返回包含文件内容的字符串。

`**file_handle.readline()**` 返回文件中的下一行。

`**file_handle.readlines()**` 返回包含文件行的列表。

`**file_handle.write(s)**` 将字符串 `s` 写入文件末尾。

`**file_handle.writelines(L)**` 将 `L` 的每个元素写入文件作为单独的行。

`**file_handle.close()**` 关闭文件。
