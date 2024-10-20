# 获得对 Domino 实例的 Shell 访问

> 原文：<https://www.dominodatalab.com/blog/gain-shell-access-domino-instances>

注意:请注意，对于 4.x 以上的 Domino 版本，不赞成通过 SSH 直接访问容器。通过工作区终端(例如 JupyterLab、VSCode 等)间接访问 SSH。)在所有 Domino 版本中仍然可用。

Domino 提供了一个可管理的、可伸缩的计算环境，为数据科学家提供了方便，无论他们是对探索 Python 笔记本中的数据感兴趣，还是对使用 R 脚本训练模型感兴趣。虽然 Domino 为团队提供了许多定制和共享环境的机会，但是有些问题最好通过 shell 访问来解决。通过 Domino 的最新版本，用户可以做到这一点。

一旦启动了交互式会话，您将能够按照运行细节中的说明使用 SSH 访问正在运行的实例。看看这个短片: