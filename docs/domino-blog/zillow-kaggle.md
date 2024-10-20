# 用 36 行代码改进 Zillow 的 Zestimate

> 原文：<https://www.dominodatalab.com/blog/zillow-kaggle>

Zillow 和 Kaggle 最近开始了一场[100 万美元的比赛来提高热情](https://www.kaggle.com/c/zillow-prize-1)。我们使用 H2O 的 AutoML 生成一个解决方案。

新的超低价格竞争受到了媒体的大量报道，这是有充分理由的。如果你能提高他们的 Zestimate 功能的准确性，Zillow 已经投入了 100 万美元。这是 Zillow 对房子价值的估计。正如他们在比赛描述中所说，提高这个估价可以更准确地反映美国近 1.1 亿套房屋的价值！

我们构建了一个项目，作为利用数据科学社区正在构建的一些令人惊叹的技术的快捷方式！在这个项目中有一个脚本`take_my_job.R`，它使用了令人惊叹的 H2O AutoML 框架。

H2O 的机器学习图书馆是行业领导者，他们将人工智能带入大众的最新尝试是 AutoML 功能。通过一个函数调用，它可以并行训练许多模型，将它们集成在一起，并建立一个强大的预测模型。

这个脚本只有 36 行:

```py
library(data.table)
library(h2o)

data_path <- Sys.getenv("DOMINO_EARINO_ZILLOW_HOME_VALUE_PREDICTION_DATA_WORKING_DIR")

properties_file <- file.path(data_path, "properties_2016.csv")
train_file <- file.path(data_path, "train_2016.csv")
properties <- fread(properties_file, header=TRUE, stringsAsFactors=FALSE,
                   colClasses = list(character = 50))
train      <- fread(train_file)

properties_train = merge(properties, train, by="parcelid",all.y=TRUE)
```

在前 12 行中，我们设置了环境，并将数据作为 R data.table 对象导入。我们在第 4 行使用 Domino 环境变量功能，不必在脚本中硬编码任何路径，因为硬编码的路径通常会带来巨大的挑战。

在第 12 行，我们通过将属性文件与训练数据集合并来创建训练集，该数据集包含我们将要预测的 logerror 列。

```py
h2o.init(nthreads = -1)

Xnames <- names(properties_train)[which(names(properties_train)!="logerror")]
Y <- "logerror"

dx_train <- as.h2o(properties_train)
dx_predict <- as.h2o(properties)

md <- h2o.automl(x = Xnames, y = Y,
stopping_metric="RMSE",
training_frame = dx_train,
leaderboard_frame = dx_train)
```

这段代码就是利用 [H2O 的 AutoML 基础设施](https://www.dominodatalab.com/blog/deep-learning-with-h2o-ai)所需要的全部！

在第 14 行，我们正在初始化 H2O，让它使用和机器内核一样多的线程。第 16 行和第 17 行用于设置预测变量和响应变量的名称。在第 19 行和第 20 行，我们将 data.table 对象上传到 H2O(这本来可以用 h2o.importFile 来避免)。在第 22-25 行中，我们告诉 H2O 在训练数据集上使用 RMSE 作为早期停止指标，为我们建立它能做到的最好的模型。

```py
properties_target<- h2o.predict(md@leader, dx_predict)
predictions <- round(as.vector(properties_target$predict), 4)

result <- data.frame(cbind(properties$parcelid, predictions, predictions * .99,
predictions * .98, predictions * .97, predictions * .96,
predictions * .95))

colnames(result)<-c("parcelid","201610","201611","201612","201710","201711","201712")
options(scipen = 999)
write.csv(result, file = "submission_automl.csv", row.names = FALSE )
```

第 27-36 行是我们最后的预测和簿记。

在第 27 行，我们使用训练好的 AutoML 对象来预测我们的响应。然后，我们将答案四舍五入到 4 位数的精度，构建结果 data.frame，设置名称，并将其写出。

我添加的唯一一点技巧是将每一列的 logerror 缩小 1%，假设 Zillow 的团队总是让他们的模型变得更好一点。

在我没有任何意见的情况下，这个包建立了一个提供公共排行榜分数`0.0673569`的模型。不惊人，但考虑到我还没看过这些数据，这已经很了不起了。将 H2O 的算法与灵活的可扩展计算和简单的环境配置结合在一起，使这个项目变得快速而简单。

## 包裹

虽然手工构建的解决方案在 Kaggle 排行榜上的得分明显高于这一个，但完全自动化的解决方案表现相当好仍然令人兴奋。全自动数据科学的未来令人兴奋，我们迫不及待地继续支持社区开发的惊人工具！