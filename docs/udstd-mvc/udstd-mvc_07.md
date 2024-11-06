# 问答

## 问题和答案，技巧和窍门

**我应该使用简单的容器方法（例如带有通知的键/值字典）作为模型，而不是一个完整的对象吗？**

一个简单的容器，比如一个键/值字典，从技术上讲可以被用作模型，通常还会加入通知功能。通常，这些模型对于键有一个约定，通常是一个字符串。这种解决方案对于简单的模型来说是可以的，但随着存储数据量的增长，它的非结构化特性和访问方式将导致纠缠不清、不一致和未记录的存储。这个模型将成为一个“大杂烩数据”的集合，几乎没有清晰度或强制一致性。

```
Enforcing access through well defined object relations and interfaces is
recommended for models beyond the most trivial cases. 
```

**如何在视图中报告错误？**

```
This is more of a Human Interface design question, but the choice can influence the
design choices at the level of Model and View. It also depends on the data. Individual
values that are incorrect can be marked in red. Typically, the user would input some data.
the value would be checked for validity, and if found invalid, the View would be changed
to express this information. This case probably enjoys using a Local Model, so that changes
can be discarded and the original Model is left untouched. It also means that the Model
must be able to accept and preserve invalid data, because this invalid data
may be a step stone to reach a correct state after additional modifications. 
```
