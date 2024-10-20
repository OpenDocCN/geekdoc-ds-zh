# 教程:为数据科学运行 Dockerized Jupyter 服务器

> 原文：<https://www.dataquest.io/blog/docker-data-science/>

November 22, 2015

在 Dataquest，我们提供了一个易于使用的环境来开始学习数据科学。这个环境预配置了最新版本的 Python、众所周知的数据科学库和一个可运行的代码编辑器。它允许全新的数据科学家和有经验的数据科学家立即开始运行代码。虽然我们在数据集上提供了无缝的学习体验，但当你想切换到自己的数据集时，你必须转移到本地开发环境。

可悲的是，建立自己的本地环境是数据科学家最令人沮丧的经历。处理不一致的包版本、由于错误而失败的冗长安装以及模糊的安装指令甚至使开始编程都变得困难。当在使用不同操作系统的团队中工作时，这些问题被夸大到了更高的程度。对于许多人来说，设置是学习如何编码的最大障碍。

幸运的是，有助于解决这些发展困境的技术已经出现。在这篇文章中，我们将要探索的是一个叫做 [Docker](https://www.docker.com/) 的容器化工具。自 2013 年以来，Docker 使启动支持不同项目基础设施需求的多种数据科学环境变得快速而简单。

在本教程中，我们将向您展示如何使用 Docker 设置您自己的 Jupyter 笔记本服务器。我们将介绍 Docker 和容器化的基础知识，如何安装 Docker，以及如何下载和运行 Docker 化的应用程序。最后，你应该能够运行你自己的本地 Jupyter 服务器，拥有最新的数据科学库。![docker_whale](img/f1b8e9ca48af2c44a2c3f7a416411b93.png)

码头鲸是来帮忙的

## 码头工人和集装箱化概述

在我们深入了解 Docker 之前，了解一些导致 Docker 等技术兴起的初步软件概念是很重要的。在简介中，我们简要描述了在多个操作系统的团队中工作以及安装第三方库的困难。这些类型的问题从软件开发开始就已经存在了。

一个解决方案是使用[虚拟机](https://en.wikipedia.org/wiki/Virtual_machine)。虚拟机允许您从本地机器上运行的操作系统模拟替代操作系统。一个常见的例子是在 Linux 虚拟机上运行 Windows 桌面。虚拟机本质上是一个完全隔离的操作系统，其应用程序独立于您自己的系统运行。这在开发团队中非常有用，因为每个成员都可以运行完全相同的系统，而不管他们机器上的操作系统是什么。

然而，虚拟机并不是万能的。它们很难设置，需要大量的系统资源才能运行，并且需要很长时间才能启动。

![vm windows on mac](img/54c43c7d5bfd5b113ad56af8db5b36a7.png)

在 mac 上的虚拟机中使用 Windows 的示例

基于这一虚拟化概念，完全虚拟机隔离的另一种方法称为容器化。容器类似于虚拟机，因为它们也在隔离的环境中运行应用程序。然而，容器化的环境不是运行一个包含所有库的完整操作系统，而是一个运行在容器引擎之上的轻量级进程。

容器引擎运行容器，并为主机操作系统提供向容器共享只读库及其内核的能力。这使得正在运行的容器的大小为几十兆字节，而虚拟机的大小可能为几十千兆字节或更多。Docker 是一种容器引擎。它允许我们下载并运行包含预配置的轻量级操作系统、一组库和应用特定包的**映像**。当我们运行一个映像时，Docker 引擎产生的进程被称为**容器**。

如前所述，容器消除了配置问题，确保了跨平台的兼容性，将我们从底层操作系统或硬件的限制中解放出来。与虚拟机类似，基于不同技术(例如 Mac OS X 与微软 Windows)构建的系统可以部署完全相同的容器。与虚拟环境相比，使用容器有多种优势:

*   快速上手的能力。当您只想开始分析时，您不需要等待软件包安装。
*   跨平台一致。Python 包是跨平台的，但是有些包在 Windows 和 Linux 上的行为不同，有些包的依赖项不能安装在 Windows 上。Docker 容器总是运行在 Linux 环境中，所以它们是一致的。
*   检查点和恢复的能力。您可以将软件包安装到 Docker 映像中，然后创建该检查点的新映像。这使您能够快速撤消更改或回滚配置。

在官方 Docker 文档中可以找到对容器化和虚拟机之间的差异的很好的概述。在下一节中，我们将介绍如何在您的系统上设置和运行 Docker。

## 安装 Docker

有一个用于 Windows 和 Mac 的图形安装程序，使得安装 Docker 很容易。以下是每个操作系统的说明:

*   [窗户](https://docs.docker.com/docker-for-windows/install/)
*   [苹果操作系统](https://docs.docker.com/docker-for-mac/install/)
*   [Linux](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/)

在本教程的剩余部分，我们将介绍运行 Docker 的 Linux 和 macOS (Unix)指令。我们提供的示例应该在您的终端应用程序(Unix)或 DOS 提示符(Windows)下运行。虽然我们将突出显示 Unix shell 命令，但是 Windows 命令应该是类似的。如果有任何差异，我们建议检查官方 Docker 文档。要检查 Docker 客户端是否安装正确，这里有几个测试命令:

*   `docker version`:返回本地机器上运行的 Docker 版本的信息。
*   `docker help`:返回 Docker 命令列表。
*   `docker run hello-world`:运行 hello-world 镜像并验证 Docker 是否正确安装并运行。

## 从映像运行 Docker 容器

安装 Docker 后，我们现在可以下载和运行图像了。回想一下，映像包含运行应用程序的轻量级操作系统和库，而运行的映像称为容器。你可以把一个图像想象成*可执行的*文件，这个文件产生的运行进程就是容器。让我们从运行一个基本的 Docker 映像开始。

在 shell 提示符下输入`docker run`命令(如下)。确保输入完整的命令:`docker run ubuntu:16.04`如果您的 Docker 安装正确，您应该看到如下输出:

```py
Unable to find image 'ubuntu:16.04' locally
16.04: Pulling from library/ubuntu
297061f60c36: Downloading [============> ] 10.55MB/43.03MB
e9ccef17b516: Download complete
dbc33716854d: Download complete
8fe36b178d25: Download complete 686596545a94: Download complete
```

我们来分解一下之前的命令。首先，我们开始将`run`参数传递给`docker`引擎。这告诉 Docker 下一个参数，`ubuntu:16.04`是我们想要运行的图像。我们传入的图像参数由图像名称、`ubuntu`和相应的**标签**、`16.04`组成。您可以将标签视为图像版本。此外，如果您将图像标签留空，Docker 将运行**最新的**图像版本(即`docker run ubuntu` < - > `docker run ubuntu:latest`)。

一旦我们发出命令，Docker 通过检查映像是否在您的本地机器上来启动运行过程。如果 Docker 找不到图像，它将检查 [**Docker hub**](https://hub.docker.com/) 并下载图像。Docker hub 是一个**图像库**，这意味着它托管开源社区构建的图像，可供下载。

最后，下载完图像后，Docker 会将它作为一个容器运行。但是，请注意，当`ubuntu`容器启动时，它会立即退出。退出的原因是因为我们没有传入额外的参数来为正在运行的容器提供上下文。让我们试着运行另一个带有一些可选参数的图像到`run`命令。在下面的命令中，我们将提供`-i`和`-t`标志，启动一个**交互**会话和模拟终端(TTY)。

运行以下命令来访问 Docker 容器中运行的 Python 提示符:`docker run -i -t python:3.6`这相当于:`docker run -it python:3.6`

## 运行 Jupyter 笔记本

运行前面的命令后，您应该已经输入了 Python 提示符。在提示符下，您可以像平常一样编写 Python 代码，但是代码将在正在运行的 Docker 容器中执行。当您退出提示符时，您将退出 Python 进程，并离开关闭 Docker 容器的交互式容器模式。到目前为止，我们已经运行了 Ubuntu 和 Python 映像。这些类型的图像非常适合开发，但是它们本身并不令人兴奋。相反，我们将运行一个 Jupyter 映像，它是一个构建在`ubuntu`映像之上的**应用程序特定的**映像。

我们将使用的 Jupyter 图片来自 Jupyter 的开发社区。这些图像的蓝图被称为 **Dockerfile** ，可以在他们的 [Github repo](https://github.com/jupyter/docker-stacks/tree/master/base-notebook) 中找到。本教程中我们不会详细讨论 Dockerfiles，所以就把它们看作是创建图像的*源代码*。图像的 Dockerfile 通常托管在 Github 上，而构建的图像则托管在 Docker Hub 上。首先，让我们在一个 Jupyter 映像上调用 Docker run 命令。

我们将运行只安装了 Python 和 Jupyter 的`minimal-notebook`。输入下面的命令:`docker run jupyter/minimal-notebook`使用这个命令，我们将从`jupyter` Docker hub 帐户中提取`minimal-notebook`的最新图像。如果您看到以下输出，就知道它已经成功运行了:

```py
[C 06:14:15.384 NotebookApp]
Copy/paste this URL into your browser when you connect for the first time,
to login with a token:
https://localhost:8888/?token=166aead826e247ff182296400d370bd08b1308a5da5f9f87
```

与运行非 Dockerized Jupyter 笔记本类似，您将拥有一个到 Jupyter 本地主机服务器的链接和一个给定的令牌。但是，如果您尝试导航到提供的链接，您将无法访问服务器。在你的浏览器上，你会看到一个“网站无法访问”的页面。

这是因为 Jupyter 服务器运行在它自己独立的 Docker 容器中。这意味着除非明确指示，否则所有端口、目录或任何其他文件都不会与您的本地计算机共享。为了访问 Docker 容器中的 Jupyter 服务器，我们需要通过传入`-p <host_port>:<container_port>`标志和参数来打开主机和容器之间的端口。`docker run -p 8888:8888 jupyter/minimal-notebook` ![jupyter](img/59f2458a9c5d2c5c5062575ed215c10d.png)

如果您看到上面的屏幕，这是您在浏览器中导航到 URL 后应该看到的内容，您正在 Docker 容器中成功地进行开发。概括地说，不需要下载 Python、一些运行时库和 Jupyter 包，只需要安装 Docker、下载官方的 Jupyter 映像并运行容器。接下来，我们将对此进行扩展，并了解如何从您的主机(本地机器)与运行的容器共享笔记本。

## 在主机和容器之间共享笔记本

首先，我们将在我们的主机上创建一个目录，在那里我们将保存所有的笔记本。在您的主目录中，创建一个名为`notebooks`的新目录。这里有一个命令可以做到这一点:`mkdir ~/notebooks`

现在我们有了笔记本的专用目录，我们可以在主机和容器之间共享这个目录。类似于打开端口，我们需要向`run`命令传递另一个额外的参数。这个参数的标志是`-v <host_directory>:<container_directory>`，它告诉 Docker 引擎**将给定的主机目录挂载到容器目录。在 Jupyter Docker 文档中，它将容器的工作目录指定为`/home/jovyan`。因此，我们将使用 mount `run`标志把我们的`~/notebooks`目录挂载到容器的工作目录。`docker run -p 8888:8888 -v ~/notebooks:/home/jovyan jupyter/minimal-notebook`安装好目录后，转到 Jupyter 服务器并创建一个新笔记本。将笔记本从“未命名”重命名为“示例笔记本”。![rename_notebook](img/15283c817e0516bfa22b7e101a99a45e.png)在您的主机上，检查`~/notebooks`目录。在那里，您应该会看到一个 iPython 文件:`Example Notebook.ipynb`！**

## 安装附加软件包

在我们的`minimal-notebook` Docker 映像中，有预装的 Python 包可供使用。其中一个是我们一直在使用的，Jupyter 笔记本，这是我们在浏览器上访问的笔记本服务器。其他包是隐式安装的，比如`requests`包，您可以在笔记本中导入它。请注意，这些预安装的包是捆绑在映像中的，不是我们自己安装的。

正如我们已经提到的，使用容器进行开发的主要好处之一就是不必安装包。但是，如果图像缺少您想要使用的数据科学包，比如说用于机器学习的`tensorflow`，该怎么办？在容器中安装软件包的一种方法是使用 [`docker exec`命令](https://docs.docker.com/engine/reference/commandline/exec/)。

`exec`命令与`run`命令有相似的参数，但是它不使用参数启动容器，它*在已经运行的容器上执行*。因此，在从图像创建容器的 insead 中，像 docker run 一样，`docker exec`需要一个正在运行的**容器 ID** 或**容器名称**，它们被称为**容器标识符**。要定位一个正在运行的容器的标识符，您需要调用 [`docker ps`命令](https://docs.docker.com/engine/reference/commandline/ps/)，该命令列出了所有正在运行的容器和一些附加信息。例如，下面是我们的`docker ps`在`minimal-notebook`容器运行时的输出。

```py
$ docker ps
CONTAINER ID IMAGE COMMAND CREATED STATUS PORTS NAMES
874108dfc9d9 jupyter/minimal-notebook "tini -- start-noteb…" Less than a second ago Up 4 seconds 0.0.0.0:8900->8888/tcp thirsty_almeida
```

现在我们有了容器 ID，我们可以在容器中安装 Python 包了。从 [`docker exec`文档](https://docs.docker.com/engine/reference/commandline/exec/)中，我们传入 runnable 命令作为标识符的参数，然后在容器中执行。回想一下，安装 Python 包的命令是`pip install <package name>`。为了安装`tensorflow`，我们将在 shell 中运行以下代码。**注意:**，您的容器 ID 将与提供的示例不同。`docker exec 874108dfc9d9 pip install tensorflow`

如果成功，您应该会看到 pip 安装输出日志。您将注意到的一件事是，在这个 Docker 容器中安装`tensorflow`相对较快(假设您有快速的互联网)。如果你以前安装过`tensorflow`，你会知道这是一个非常繁琐的设置，所以你可能会惊讶于这个过程是多么的轻松。这是一个快速安装过程的原因，因为`minimal-notebook`映像的编写考虑到了数据科学优化。基于 Jupyter 社区对安装最佳实践的想法，已经预安装了 C 库和其他 Linux 系统级包。这是使用开源社区 developer Docker 图像的最大好处，因为它们通常针对您将要进行的开发工作类型进行了优化。

## 扩展 Docker 图像

到目前为止，您已经安装了 Docker，运行了您的第一个容器，访问了 Docker 化的 Jupyter 容器，并在一个正在运行的容器上安装了`tensorflow`。现在，假设您已经使用`tensorflow`库完成了一天的 Jupyter 容器工作，并且您想要关闭容器以回收处理速度和内存。

要停止容器，您可以运行 [`docker stop`](https://docs.docker.com/engine/reference/commandline/stop/) 或 [`docker rm`](https://docs.docker.com/engine/reference/commandline/rm/) 命令。第二天，您再次运行容器，并准备开始着手 tensorflow 工作。然而，当你去运行笔记本电池时，你被一个`ImportError`挡住了。如果你前一天已经安装了`tensorflow`，怎么会发生这种情况？

![import_error_tensorflow](img/fef84eaf3b56b965445c136baf56edb1.png)

问题出在`docker exec`命令上。回想一下，当您运行`exec`时，您正在对运行容器的*执行给定的命令。容器只是映像的运行进程，其中映像是包含所有预安装库的可执行文件。所以，当你在容器上安装`tensorflow`时，它只是为那个**特定的**实例安装的。*

因此，关闭容器就是从内存中删除该实例，当您从映像中重新启动一个新容器时，只有映像中包含的库才能再次使用。将`tensorflow`保存到映像上已安装软件包列表的唯一方法是**修改**原始 docker 文件并构建一个新映像，或者**扩展**docker 文件并从这个新 docker 文件构建一个新映像。

不幸的是，这些步骤都需要理解 Dockerfiles，这是一个 Docker 概念，我们在本教程中不会详细介绍。但是，我们使用的是 Jupyter 社区开发的 Docker 映像，所以让我们检查一下是否已经有一个使用`tensorflow`构建的 Docker 映像。再次查看 [Jupyter github 库](https://github.com/jupyter/docker-stacks)，可以看到有一个`tensorflow`笔记本！

不仅仅是`tensorflow`，还有很多其他的选择。下面的树形图来自他们的文档，描述了 docker 文件和每个使用中的可用图像之间的关系扩展。![jupyter_docker_stacks](img/83606c28b5a3195a5dae86f23960b659.png)因为`tensorflow-notebook`是从`minimal-notebook`扩展而来的，我们可以从之前的`docker run`命令开始，只改变图像的名称。以下是我们如何在您的浏览器上运行预装了`tensorflow` : `docker run -p 8888:8888 -v ~/notebooks:/home/jovyan jupyter/tensorflow-notebook`的 Dockerized Jupyter 笔记本服务器，使用前几节描述的相同方法导航到正在运行的服务器。在那里，在一个代码单元中运行`import tensorflow`，你应该看不到`ImportError`！

## 后续步骤

在本教程中，我们讨论了虚拟化和容器化之间的区别，如何安装和运行 Dockerized 应用程序，以及使用开源社区开发人员 Docker 映像的好处。我们使用了一个容器化的 Jupyter 笔记本服务器作为例子，展示了在 Docker 容器中使用 Jupyter 服务器是多么容易。完成本教程后，您应该会对使用 Jupyter 社区图像感到舒适，并能够在日常工作中融入一个文档化的数据科学设置。

虽然我们涵盖了许多 Docker 概念，但这些只是帮助您入门的基础。关于 Docker 以及它的功能有多强大，还有很多东西需要学习。掌握 Docker 不仅有助于您的本地开发时间，而且可以在与数据科学家团队合作时节省时间和金钱。如果喜欢这个例子，这里有一些改进可以帮助你学习更多的 Docker 概念:

1.  以**分离**模式作为后台进程运行 Jupyter 服务器。
2.  命名您的运行容器以保持您的进程干净。
3.  创建自己的 docker 文件，扩展包含必要数据科学库的`minimal-notebook`。
4.  从扩展 Dockerfile 文件构建一个映像。

### 成为一名数据工程师！

现在就学习成为一名数据工程师所需的技能。注册一个免费帐户，访问我们的交互式 Python 数据工程课程内容。

[Sign up now!](https://app.dataquest.io/signup)

*(免费)*

![YouTube video player for ddM21fz1Tt0](img/5a85348206993fc2a430506128b76684.png)

*[https://www.youtube.com/embed/ddM21fz1Tt0?rel=0](https://www.youtube.com/embed/ddM21fz1Tt0?rel=0)*