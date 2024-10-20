# 数据科学编程最佳实践

> 原文：<https://www.dataquest.io/blog/programming-best-practices-for-data-science/>

June 8, 2018

数据科学生命周期通常由以下组件组成:

*   资料检索
*   数据清理
*   数据探索和可视化
*   统计或预测建模

虽然这些组件有助于理解不同的阶段，但它们并不能帮助我们思考我们的*编程*工作流程。

通常情况下，整个数据科学生命周期最终会成为任意混乱的笔记本单元，要么是 Jupyter 笔记本，要么是一个杂乱的脚本。此外，大多数数据科学问题需要我们在数据检索、数据清理、数据探索、数据可视化和统计/预测建模之间切换。

但是有更好的方法！在这篇文章中，我将回顾大多数人在进行专门针对数据科学的编程工作时会转换的两种心态:原型心态和生产心态。

| 原型思维优先: | 生产思维优先考虑: |
| 小段代码的迭代速度 | 整个管道上的迭代速度 |
| 更少的抽象(直接修改代码和数据对象) | 更抽象(改为修改参数值) |
| 更少的代码结构(更少的模块化) | 更多的代码结构(更多的模块化) |
| 帮助您和其他人理解代码和数据 | 帮助计算机自动运行代码 |

我个人在整个过程中都使用 [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html) (原型和生产)。我建议至少使用 JupyterLab **做原型**。

#### 租借俱乐部数据

为了帮助更具体地理解原型和生产思维模式之间的区别，让我们使用一些真实的数据。我们将使用来自点对点贷款网站 [Lending Club](https://www.lendingclub.com) 的贷款数据。与银行不同，Lending Club 本身并不放贷。相反，Lending Club 是一个市场，贷方向出于各种原因(房屋维修、婚礼费用等)寻求贷款的个人提供贷款。).我们可以使用这些数据来建立模型，预测给定的贷款申请是否会成功。在这篇文章中，我们不会深入构建一个机器学习管道来进行预测，但我们会在我们的[机器学习项目演练课程](https://www.dataquest.io/course/machine-learning-project)中涉及到它。

Lending Club 提供有关已完成贷款(贷款申请得到 Lending Club 的批准，他们找到了贷款人)和已拒绝贷款(贷款申请被 Lending Club 拒绝，资金从未转手)的详细历史数据。导航到他们的[数据下载页面](https://www.lendingclub.com/info/download-data.action)，选择**下的 **2007-2011** 。**

![lendingclub](img/11b673623fbb593a21b1471dec40851a.png)

#### 原型思维

在原型思维中，我们对快速迭代感兴趣，并试图理解数据的一些属性和真相。创建一个新的 Jupyter 笔记本，并添加一个说明以下内容的减价单元格:

*   为了更好地了解 Lending Club 平台，您做了哪些研究
*   你下载的数据集有什么信息吗

首先，让我们将 CSV 文件读入 pandas。

```py
import pandas as pd
loans_2007 = pd.read_csv('LoanStats3a.csv')
loans_2007.head(2)
```

我们得到两个输出，第一个是警告。

```py
/home/srinify/anaconda3/envs/dq2/lib/python3.6/site-packages/IPython/core/interactiveshell.py:2785: DtypeWarning: Columns (0,1,2,3,4,7,13,18,24,25,27,28,29,30,31,32,34,36,37,38,39,40,41,42,43,44,46,47,49,50,51,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,123,124,125,126,127,128,129,130,131,132,133,134,135,136,142,143,144) have mixed types. Specify dtype option on import or set low_memory=False.  interactivity=interactivity, compiler=compiler, result=result)
```

然后是数据帧的前 5 行，我们将避免在这里显示(因为它很长)。

我们还获得了以下数据帧输出:

|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 招股说明书发行的票据(https://www.lendingclub.com/info/prospectus.action) |
| d | 成员 id | 贷款金额 | 资助 _amnt | 资助 _amnt_inv | 学期 | 利息率 | 部分 | 等级 | 路基 | 员工 _ 职位 | 员工长度 | 房屋所有权 | 年度公司 | 验证 _ 状态 | 问题 _d | 贷款 _ 状态 | pymnt_plan | 全球资源定位器(Uniform Resource Locator) | desc | 目的 | 标题 | 邮政编码 | 地址状态 | 弥散张量成像 | delinq _ 年 | 最早 _cr_line | 最近 6 个月 | mths _ since _ last _ delinq | 月 _ 自 _ 最后 _ 记录 | open_acc | 发布 _ 记录 | 革命 _ 平衡 | 革命报 | 总计 _acc | 初始列表状态 | out_prncp | out_prncp_inv | total_pymnt | total_pymnt_inv | total_rec_prncp | total_rec_int | total_rec_late_fee | 追回款 | 收款 _ 回收 _ 费用 | last_pymnt_d | last_pymnt_amnt | 下一个 _pymnt_d | 最后一笔贷款 | 收藏 _ 12 _ 月 _ 月 _ 日 _ 医学 | 月 _ 自 _ 最后 _ 主要 _ 日志 | 策略代码 | 应用程序类型 | 年度 _ 公司 _ 联合 | dti _ 联合 | 验证 _ 状态 _ 联合 | acc _ now _ delinq | 总计 _ 合计 _ 金额 | tot_cur_bal | open_acc_6m | open_act_il | open_il_12m | open_il_24m | mths_since_rcnt_il | 总计 _bal_il | il_util | 开 _rv_12m | open_rv_24m | 最大余额 | all_util | 总计 _ 收入 _ 高收入 | inq_fi(消歧义) | 总计 _ 铜 _ 铊 | inq_last_12m | acc _ open _ past _ 月 | avg _ cur _ ball | bc_open_to_buy | bc_util | 12 个月内收费 | delinq _ amnt | 莫 _ 辛 _ 旧 _ 日 _ 账户 | mo_sin_old_rev_tl_op | mo_sin_rcnt_rev_tl_op | mo_sin_rcnt_tl | acc 死亡 | 月 _ 自 _ 最近 _ 公元前 | 月 _ 自 _ 最近 _bc_dlq | 月 _ 自 _ 最近 _ 入 q | mths _ since _ recent _ revol _ delinq | num_accts_ever_120_pd | 数字 _ 活动 _ 商业 _ 时间 | 数量 _ 活动 _ 版本 _tl | num _ bc _ 语句 | 数字 _bc_tl | 数字 il TL | S7-1200 可编程控制器 | num_rev_accts | 数量 _ 收益 _ tl _ 余额 _gt_0 | 数量 _ 饱和度 | num_tl_120dpd_2m | num_tl_30dpd | 数量 _tl_90g_dpd_24m | 数量 _tl_op_past_12m | pct_tl_nvr_dlq | percent_bc_gt_75 | pub _ rec _ 破产 | tax _ links-税捐连结 | tot_hi_cred_lim | total _ bal _ ex _ mort-总计 _ bal _ ex _ 死亡 _ 终止 _ 终止 _ 终止 _ 终止 _ 终止 _ 终止 _ 终止 _ 终止 _ 终止 _ 终止 _ 终止 _ 终止 _ 终止 _ 终止 _ 终止 _ 终止 _ 终止 _ 终止 _ 终止 _ 终止 _ 终止 | 总计 _ bc _ 限制 | total_il_high_credit_limit | 斜接 _ 边 _ 边 _ 边 _ 边 _ 边 _ 边 _ 边 _ 边 _ 边 _ 边 _ 边 _ 边 _ 边 _ 边 _ 边 _ 边 _ 边 _ 边 _ 边 _ 边 _ 边 _ 边 _ 边 _ 边 _ 边 _ 边 _ 边 | sec_app_earliest_cr_line | sec _ app _ inq _ last _ 个月 | 秒 _ 应用程序 _ 抵押 _ 帐户 | sec_app_open_acc | sec_app_revol_util | sec_app_open_act_il | 秒 _ 应用 _ 数量 _ 收入 _ 帐户 | sec _ app _ chargeoff _ within _ 12 _ 个月 | sec _ app _ collections _ 12 _ mths _ ex _ med | sec _ app _ mths _ since _ last _ major _ derog | 艰难 _ 标志 | 困难类型 | 困难 _ 原因 | 艰苦状况 | 延期 _ 期限 | 困难 _ 金额 | 艰难开始日期 | 困难 _ 结束 _ 日期 | 付款 _ 计划 _ 开始 _ 日期 | 艰辛 _ 长度 | 困难 _dpd | 困难 _ 贷款 _ 状态 | 原始 _ 预计 _ 附加 _ 应计 _ 利息 | 困难 _ 收益 _ 余额 _ 金额 | 困难 _ 最后 _ 付款 _ 金额 | 支付方式 | 债务 _ 结算 _ 标志 | 债务 _ 结算 _ 标志 _ 日期 | 结算 _ 状态 | 结算日期 | 结算 _ 金额 | 结算 _ 百分比 | 结算 _ 期限 |

这个警告让我们知道，如果我们在调用`pandas.read_csv()`时将`low_memory`参数设置为`False`，那么每一列的熊猫类型推断将会得到改进。

第二个输出更成问题，因为数据帧存储数据的方式有问题。JupyterLab 内置了一个终端环境，因此我们可以打开它并使用 bash 命令`head`来观察原始文件的前两行:

```py
head -2 LoanStats3a.csv
```

虽然第二行包含了我们在 CSV 文件中期望的列名，但是当 pandas 试图解析该文件时，第一行似乎抛出了 DataFrame 的格式:

```py
Notes offered by Prospectus (https://www.lendingclub.com/info/prospectus.action)
```

添加一个详述您的观察结果的 Markdown 单元格，并添加一个将观察结果考虑在内的 code 单元格。

```py
 import pandas as pd
loans_2007 = pd.read_csv('LoanStats3a.csv', skiprows=1, low_memory=False)
```

从 [Lending Club 下载页面](https://www.lendingclub.com/info/download-data.action)阅读数据字典，了解哪些栏目不包含有用的功能信息。`desc`和`url`列似乎很符合这个标准。

```py
loans_2007 = loans_2007.drop(['desc', 'url'],axis=1)
```

下一步是删除丢失行超过 50%的任何列。使用一个单元格来研究哪些列符合该条件，使用另一个单元格来实际删除这些列。

```py
loans_2007.isnull().sum()/len(loans_2007)
```

```py
loans_2007 = loans_2007.dropna(thresh=half_count, axis=1)
```

因为我们使用 Jupyter 笔记本来跟踪我们的想法和代码，所以我们依靠环境(通过 IPython 内核)来跟踪状态的变化。这让我们可以自由行动，移动单元格，多次运行相同的代码，等等。

一般来说，原型思维中的代码应该关注:

*   易懂
    *   来描述我们的观察和假设
    *   实际逻辑的小段代码
    *   大量的可视化和计数
*   最小抽象
    *   大多数代码不应该在函数中(应该感觉更面向对象)

假设我们又花了一个小时探索数据，并编写描述我们所做的数据清理的 markdown 单元格。然后，我们可以切换到生产思维模式，使代码更加健壮。

#### 生产思维

在生产思维中，我们希望专注于编写可以推广到更多情况的代码。在我们的例子中，我们希望我们的数据清理代码适用于来自 Lending Club(来自其他时间段)的任何数据集。概括我们代码的最好方法是把它变成一个数据管道。使用来自[函数编程](https://www.dataquest.io/blog/introduction-functional-programming-python/)的原理设计了一条数据流水线，其中数据在函数内被修改*，然后在*函数间传递*。*

这是使用单个函数封装数据清理代码的管道的第一次迭代:

```py
 import pandas as pd
def import_clean(file_list):
    frames = []
    for file in file_list:
        loans = pd.read_csv(file, skiprows=1, low_memory=False)
        loans = loans.drop(['desc', 'url'], axis=1)
        half_count = len(loans)/2
        loans = loans.dropna(thresh=half_count, axis=1)
        loans = loans.drop_duplicates()
        # Drop first group of features
        loans = loans.drop(["funded_amnt", "funded_amnt_inv",
 "grade", "sub_grade", "emp_title", "issue_d"], axis=1)
        # Drop second group of features
        loans = loans.drop(["zip_code", "out_prncp",
 "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
 "total_rec_prncp"], axis=1)
        # Drop third group of features
        loans = loans.drop(["total_rec_int", "total_rec_late_fee",
 "recoveries", "collection_recovery_fee", "last_pymnt_d",
 "last_pymnt_amnt"], axis=1)
        frames.append(loans)
    return frames

frames = import_clean(['LoanStats3a.csv'])
```

在上面的代码中，我们**将前面的代码抽象为一个函数。这个函数的输入是文件名列表，输出是 DataFrame 对象列表。**

总的来说，生产思维应该集中在:

*   健康的抽象概念
    *   代码应该一般化以兼容相似的数据源
    *   代码不应该太笼统，以至于难以理解
*   管道稳定性
    *   可靠性应该与其运行频率相匹配(每天？周刊？每月？)

#### 在思维模式之间切换

假设我们试图对 Lending Club 的所有数据集运行该函数，但 Python 返回了错误。一些潜在的错误来源:

*   某些文件中的列名不同
*   由于 50%缺失值阈值而被删除的列中的差异
*   基于该文件的熊猫类型推断的不同列类型

在这种情况下，我们实际上应该切换回我们的原型笔记本，并进一步研究。当我们确定希望我们的管道更加灵活，并考虑到数据中的特定变化时，我们可以将这些变化重新合并到管道逻辑中。

下面是一个例子，我们调整了函数以适应不同的跌落阈值:

```py
import pandas as pd
def import_clean(file_list, threshold=0.5):
    frames = []
    for file in file_list:
        loans = pd.read_csv(file, skiprows=1, low_memory=False)
        loans = loans.drop(['desc', 'url'], axis=1)
        threshold_count = len(loans)*threshold
        loans = loans.dropna(thresh=half_count, axis=1)
        loans = loans.drop_duplicates()
        # Drop first group of features
        loans = loans.drop(["funded_amnt", "funded_amnt_inv",
 "grade", "sub_grade", "emp_title", "issue_d"], axis=1)
        # Drop second group of features
        loans = loans.drop(["zip_code", "out_prncp",
 "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
 "total_rec_prncp"], axis=1)
        # Drop third group of features
        loans = loans.drop(["total_rec_int", "total_rec_late_fee",
 "recoveries", "collection_recovery_fee", "last_pymnt_d",
 "last_pymnt_amnt"], axis=1)
        frames.append(loans)
    return frames

frames = import_clean(['LoanStats3a.csv'], threshold=0.7)
```

默认值仍然是`0.5`，但是如果我们愿意，我们可以将其改写为`0.7`。

以下是一些使管道更灵活的方法，按优先级递减:

*   使用可选的、位置的和必需的参数
*   在函数中使用 if / then 语句和布尔输入值
*   使用新的数据结构(字典、列表等)。)来表示特定数据集的自定义操作

这个管道可以扩展到数据科学工作流的所有阶段。这里有一些框架代码，预览这是什么样子。

```py
 import pandas as pd
def import_clean(file_list, threshold=0.5):
    ## Code
def visualize(df_list):
    # Find the most important features and generate pairwise scatter plots
    # Display visualizations and write to file.
    plt.savefig("scatter_plots.png")
def combine(df_list):
    # Combine dataframes and generate train and test sets
    # Drop features all dataframes don't share
    # Return both train and test dataframes
    return train,test
def train(train_df):
    # Train model
    return model
def validate(train_df, test-df):
    # K-fold cross validation
    # Return metrics dictionary
    return metrics_dict
    frames = import_clean(['LoanStats3a.csv', 'LoanStats2012.csv'],
 threshold=0.7)
visualize(frames)
train_df, test_df = combine(frames)
model = train(train_df)
metrics = test(train_df, test_df)
print(metrics)
```

#### 后续步骤

如果您有兴趣加深理解并进一步实践，我推荐以下步骤:

*   了解如何将您的管道变成一个独立的脚本，可以作为一个模块运行，也可以从命令行运行:[https://docs.python.org/3/library/**主**。html](https://docs.python.org/3/library/__main__.html)
*   了解如何使用 Luigi 构建可以在云中运行的更复杂的管道:[用 Python 和 Luigi 构建数据管道](https://marcobonzanini.com/2015/10/24/building-data-pipelines-with-python-and-luigi/)
*   了解更多关于数据工程的信息:[Data quest 上的数据工程帖子](https://www.dataquest.io/blog/tag/data-engineering/)