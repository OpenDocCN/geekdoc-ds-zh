# 一个基于 Mongo 的游戏缓存插件

> 原文：<https://www.dominodatalab.com/blog/a-mongo-based-cache-plugin-for-play>

一个与工程相关的快速帖子:我们在 Mongo 中为 Play 构建了一个使用上限集合的缓存插件。如果你想用的话，可以在 Github 上找到它。

## 动机

Domino 的 web 前端是在 [Scala](http://www.scala-lang.org/) 中构建的，在 [Play](https://playframework.com/) 上——现在，有点勉强地托管在 Heroku 上。

我们有多个 dyno 在运行，我们希望每个 dyno 访问同一个缓存实例。因此，位于 web 进程内存中或机器磁盘上的缓存是不起作用的。

我们最初尝试通过 Memcachier 使用 Memcached，但是我们得到了我们永远无法解决的间歇性超时，即使有 Memcachier 人员的帮助。

## 解决办法

Mongo 中有上限的集合具有非常适合缓存的特性:快速写入和 FIFO 驱逐。我们的插件为每个缓存项增加了一个可选的过期时间。

项目中的自述文件[中有自己使用的说明。](https://github.com/dominodatalab/play-mongo-cache)

## 警告

这并不意味着疯狂的吞吐量。我们没有在“网络规模”上做任何事情，我怀疑这种解决方案能否承受巨大的负载。这是一个相当基本的解决方案，对于许多用例来说可能已经足够好了——而不是被设计成工业优势的东西。