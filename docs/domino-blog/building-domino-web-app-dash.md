# 用 Dash 构建 Domino Web 应用程序

> 原文：<https://www.dominodatalab.com/blog/building-domino-web-app-dash>

*[Randi R. Ludwig](https://www.linkedin.com/in/randi-r-ludwig-717150114) ，戴尔 EMC 的数据科学家和[ATX 数据科学女性](https://www.meetup.com/Women-in-Data-Science-ATX/)的组织者，在这篇文章中介绍了如何使用 Dash 构建 Domino web 应用程序。*

## 用 Dash 构建 Domino Web 应用程序

假设您是一名数据科学家，使用 Domino 作为您的探索和开发平台。你做了一个分析，突出了一些真正有用的结果！您知道，随着新数据的出现，您的团队将需要检查这一分析并探索结果。这是为您的分析和结果构建交互式界面的完美案例。

Dash 是一个用于构建交互式 web 应用的框架，特别有助于开发像仪表板这样的分析界面。R 用户有闪亮，Pythonistas 有破折号。Dash 是由 plot.ly 团队开发的，是基于 Flask 框架构建的，所以 Flask 在哪里工作，Dash 也应该在哪里工作。在提供交互式图形界面的同时，Dash 还提供了对许多功能的支持。如果你有兴趣了解更多，Dash 团队提供了[优秀教程](https://plot.ly/dash/getting-started)。想了解 Dash 能做什么的更多例子，请查看这套广泛的[资源](https://github.com/acrotrend/awesome-dash)。

### 多米诺骨牌上的 Dash

要在 Domino 数据实验室中部署 Dash，您需要做一些事情。首先，您需要在启用 Python 的环境中安装 Dash 依赖项。截至 2018 年 1 月，您需要将以下命令添加到 Domino 环境的 Docker 文件中:

```py
pip install dash==0.19.0  # The core dash backend

pip install dash-renderer==0.11.1  # The dash front-end

pip install dash-html-components==0.8.0  # HTML components

pip install dash-core-components==0.15.2  # Supercharged components

pip install plotly --upgrade  # Latest Plotly graphing library

```

检查[https://plot.ly/dash/getting-started](https://plot.ly/dash/getting-started)的最新软件包版本。

## Domino 应用程序框架

部署在 Domino 上的所有应用程序都使用通用的结构。虽然您的应用程序代码可以是您想要的任何代码，但是在您打算部署的同一个项目中，您需要包含一个 app.sh 文件。对于 Dash 应用程序，这个文件非常简单，只需要包含一行:

```py
> python app.py

```

如果您在本地开发应用程序，您将使用相同的命令来运行该应用程序。(顺便说一下，在你的本地机器上开发一个应用程序来测试内容、界面等的许多调整。然后将 app.py 文件上传到 Domino 是一个很好的开发策略。)

## Domino 的破折号设置

一旦您有了一个应用程序并准备在 Domino 上部署，与您在本地机器上的配置相比，在 app.py 文件中还需要三个额外的配置。首先，您必须让 Domino 能够找到您安装的 Dash 依赖项。默认情况下，Dash 会在“/”处的根文件夹中查找您的所有文件。因为 Domino 使用反向代理和子文件夹，所以您必须告诉 Domino 在当前工作目录中查找所有相关的文件和资源。为此，您需要为 Dash 提供所需的配置:

```py
# Configure path for dependencies. This is required for Domino.

app.config.update({

#### as the proxy server may remove the prefix

'routes_pathname_prefix': '',

    #### the front-end will prefix this string to the requests

    #### that are made to the proxy server

    'requests_pathname_prefix': ''

    })

```

^(*注意:此 app.config 语法仅适用于 dash 0.18.3 及更新版本。*)

下一步是通过添加以下行使 Flask 服务器对 Domino web app 服务器可见:

```py
# This enables the Flask server to be visible to the Domino server

server = app.server

```

最后，在 app.py 文件的末尾，您需要显式地包含 web 应用服务器的主机和端口。对于所有的 Domino 应用程序，这都是一样的。

```py
# host & port need to be explicitly defined for Domino

if __name__ == '__main__':

    #app.run_server() # if running locally

    app.run_server(host='0.0.0.0',port=8888) # on Domino

```

有了这三个变化，您的应用程序应该可以在 Domino 上运行，并且对更广阔的世界可见！要在上下文中查看这段代码，请查看 [my GitHub](https://github.com/randirl17/Dash-on-Domino) 。要了解更多关于在 Domino 上部署 Shiny 或 Flask 应用程序的信息，请查看[文档](https://support.dominodatalab.com/hc/en-us/articles/209150326-Getting-Started-with-App-publishing)。