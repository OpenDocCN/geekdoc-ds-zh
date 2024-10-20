# 建立说话人识别模型

> 原文：<https://www.dominodatalab.com/blog/building-a-speaker-recognition-model>

系统通过声音识别一个人的能力是一种收集其生物特征信息的非侵入性方式。与指纹检测、视网膜扫描或面部识别不同，说话人识别只是使用麦克风来记录一个人的声音，从而避免了对昂贵硬件的需求。此外，在疫情这样的表面可能存在传染病的情况下，语音识别可以很容易地部署在涉及某种形式接触的其他生物识别系统的位置。现有的身份认证应用包括:

*   信用卡验证，
*   通过电话安全进入呼叫中心，
*   通过声音识别嫌疑人，
*   利用声音定位和追踪恐怖分子来打击恐怖主义的国家安全措施
*   在远程位置检测讲话者的存在
*   音频数据中说话人的注释和索引
*   基于语音的签名和安全的楼宇访问

## 说话人确认系统概述:

说话人确认是更广泛的说话人识别任务中的一个子领域。说话人识别是通过声音来识别说话人的过程。说话人的识别通常是以下使用情况之一所需要的:

1.  *说话人识别* -给定一个现有的说话人列表，将一个未知说话人的声音分配给列表中最接近的匹配。这里的系统将通过某种有意义的感觉来识别话语与哪个现有说话者的模型相匹配。例如，将生成一个对数概率列表，并且具有最高值的讲话者(在现有讲话者列表中)将被分类为说出“未知”话语的讲话者。
    *   这个用例的一个可能的应用是识别音频会议室的单个成员发言的时间戳。
    *   如果未知话语是由现有说话者列表之外的说话者说出的，则该模型仍然会将其映射到该列表中的某个说话者。因此，假设传入的语音 ***具有*** 来自现有列表中存在的说话者。

2.  *说话人验证* -给定一个说话人模型，系统验证输入的语音是否来自训练该模型的同一说话人。它决定了个人是否是他们所声称的那个人。
    *   这个用例的一个可能应用是使用说话者的声音作为生物认证令牌。

### 重要注意事项:

*   虽然个人的声音可以被克隆，但声音生物特征的目的是作为多因素认证方案中的“因素”之一。

*   虽然存在更先进的说话人确认模型，但本博客将构成语音信号处理的基础。此外，深度学习方法建立在信号处理的基础上。神经网络的输入应该是从语音信号中提取的特征(MFCCs、LPCCS、40-log mel 滤波器组能量),而不是整个语音信号。
*   在这个项目中，我们关注第二个用例。因此，我们的目标是通过他们的声音来验证一个人，前提是他们以前在系统中注册过。

## 说话人确认的一般流程包括三个阶段: ***开发******注册******确认*** 。

*   开发是学习与说话人无关的参数的过程，这些参数是捕获特定语音特征所需要的。在开发期间，UBM(通用背景模型)训练发生在通过大样本说话人来学习说话人独立参数的情况下。

*   登记是学习说话者声音的独特特征的过程，并且用于在验证期间创建声称的模型来代表登记的说话者。这里，说话者声音的独特特征被用于创建特定于说话者的模型。说话者说话持续一段时间(通常约 30 秒),在此期间，系统会模拟他的声音并将其保存在一个. npy 文件中。

*   在验证中，将声明者的声音的独特特征与先前登记的声明说话者模型进行比较，以确定声明的身份是否正确。这里，将讲话者的输入音频与先前保存的讲话者模型进行比较。

说话人确认通常以两种方式进行

1.  **TISV** -(文本独立说话人验证)-语音的词汇和语音内容没有限制，说话人在注册和验证期间可以说出任何短语或句子。

2.  **TDSV** -(文本相关说话人验证)-语音的词典和语音内容受到限制，并且如果用于登记的词典用于验证，则系统只能返回登记和验证之间的匹配。

## 使用的方法和算法:

*   如果您不熟悉数字信号处理的基础知识，强烈建议您阅读[信号处理基础知识](https://blog.dominodatalab.com/fundamentals-of-signal-processing)。
*   I 向量是在注册和验证期间从语音话语中提取的嵌入或向量。正是基于这种嵌入来评估余弦相似性，以预测登记和验证话语中说话者之间的匹配或不匹配。

*   为了理解 i-Vector 方法，深入研究 GMM(高斯混合模型)是必要的。自然界中随机变量的许多概率分布可以建模为高斯分布或高斯混合分布。它们是很好理解的，并且参数估计是迭代完成的。GMMs 作为一种生成方法，有效地为说话人确认系统建模说话人，在说话人确认系统中，返回输入话语是由训练的高斯生成的概率。GMM 的参数是高斯平均值、协方差和混合物的权重。

M 个分量高斯密度的加权和的 GMM，如下式所示

$ $ \ begin { equation } p(x | \ lambda)= \sum_{k=1}^{m} w _ { k } \乘以 g(x|\mu _{k}，\Sigma_{k} )\end{equation}$$

其中\(x\)是一个 D 维特征向量(在我们的例子中是 39 维 MFCCs)，\(w_{k}，\mu_{k}，\适马 _ { k }；k = 1，2，...........M\)，是混合权重、均值和协方差。\(g(x | \mu_k，\适马 _k)，k = 1，2，3.........M\)是分量高斯密度。

每个分量密度是一个 D 变量高斯函数，

$ \ begin { equation } g(\ \ boldssymbol { x } | \ boldssymbol } _ k，\ boldssymbol } _ k)= \ left[1 \ mathbin {/} \ left(2 \ mathbin {/} 2 } \ right)\ text \ { exp } \ \ 0.5(\ \ boldssymbol { x }-\ boldssymbol } _ k })} \ end { equation } $)

其中均值向量\(\boldsymbol{\mu}_k\)和协方差矩阵\(\boldsymbol{\Sigma}_{k}\)，混合权重满足约束\( \sum_{k=1}^M w_i = 1\)。完整的 GMM 由来自所有分量密度的平均向量、协方差矩阵和混合权重来参数化，并且这些参数共同表示为\(\lambda = \{w_k，\boldsymbol{\mu}_k，\ bold symbol { \适马}_{k} \}，k = 1，2，3.........M \)。

### 期望最大化

*   期望最大化(EM)算法基于最大化训练数据的期望对数似然来学习 GMM 参数。

*   随机变量相对于分布的期望值是求和(离散分布)或积分(连续分布)到无穷大时的算术平均值。

*   对于未知参数的给定值，似然函数测量统计模型与数据样本的拟合优度。在这种情况下，均值、方差和混合权重。它由样本的联合概率分布形成。

*   EM 算法的动机是使用训练话语特征从当前模型估计新的和改进的模型，使得新模型生成训练特征的概率大于或等于旧模型。这是一种迭代技术，新模型成为下一次迭代的当前模型。

$ $ \ begin { equation } \prod_{n=1}^n p(\ bold symbol { x } _ n | \ lambda)\ geq \prod_{n=1}^n p(\ bold symbol { x } _ n | \lambda^{\text{old}})\ end { equation } $ $

当输入的提取特征即 MFCCs 具有潜在变量时，最大似然估计 GMM 参数。EM 算法通过选择随机值或初始化来寻找模型参数，并使用这些参数来推测第二组数据。

k 均值算法用于迭代初始化 GMM 参数，其中通过评估混合均值来执行 MFCCs 特征向量的混合。

#### 电子步骤

$ $ \ begin { equation } p(k | \ bold symbol { x })= \ bold symbol { w } _ k g(\ bold symbol { x } | \ bold symbol { \ mu } _ k，\ bold symbol { \ sigma } _ k)\ mathbin {/} p(\ bold symbol { x } | \lambda^{\text{old}})\ end { equation } $ $

最大化导致 GMM 参数被估计

#### m 步

$ \ begin { aligned } \ boldssymbol { w } _ k & =(n _ k \ mathbin {/} t)∞p(k | \ boldssymbol { x } _ n })\ \【t 0 } \ boldssymbol } \ mu } _ k }&=(1 \ mathbin {/} n _ k)？\ p(k | \ boldssymbol { x } _ n })\ boldssymbol { x }

因此，我们评估了 GMM 的三个参数，即权重、均值和协方差。

## UBM(通用背景模型)培训

在典型的说话人确认任务中，由于可用于训练说话人模型的数据量有限，所以不能使用期望最大化以可靠的方式直接估计说话人模型。由于这个原因，最大后验概率(MAP)自适应经常被用于训练说话人确认系统的说话人模型。这种方法从通用背景模型中估计说话人模型。UBM 是一种高阶 GMM，在从感兴趣的说话人群体的广泛样本中获得的大量语音数据上进行训练。它被设计成捕捉说话者模型的一般形式，并表示与说话者无关的特征分布。使用 EM 算法来估计 UBM 参数。

*   GMM 所基于的低层特征是 MFCCs。此外，在 GMM-UBM 模型中，MFCC 声学特征被用作输入特征。GMM 平均超向量是 GMM 平均向量的串联。I 向量方法提供了高维 GMM 超向量和传统低维 MFCC 特征表示之间的中间说话人表示。

*   注册过程包括从说话者注册过程中的 30 秒长的话语中提取的 I 向量。这个 I 向量作为说话者模型保存在数据库中。稍后在验证期间，通过余弦相似性得分，将从测试话语(大约 15 秒长)中提取的 I 向量与登记的 I 向量进行比较。如果该分数超过阈值，则它被声明为匹配或不匹配。

## 资料组

用于语音处理的一些流行数据集是 LibriSpeech、LibriVox、[vox Cele 1](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html)、[vox Cele 2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)、TIMIT 等。

本演示中用于预训练 UBM 的数据集是 VoxCeleb1。该数据集文件分布如下:

### 验证分割

| **vox Cele 1** | **开发** | **测试** |
| **、扬声器数量** | One thousand two hundred and eleven | Forty |
| **、视频数量** | Twenty-one thousand eight hundred and nineteen | Six hundred and seventy-seven |
| **、话语数量** | One hundred and forty-eight thousand six hundred and forty-two | Four thousand eight hundred and seventy-four |

基于名为[关于说话人确认中通用背景模型训练的研究](https://ieeexplore.ieee.org/document/5713236)的论文，可以基于来自大约 60 个说话人的数据，即大约 1.5 小时长的数据来训练 UBM。预训练模型基于说话者的这个子集。

上述数据集远远超过所需的说话人数量，已被用作评估说话人确认性能的基准数据集。

由于 UBM 的培训甚至是针对部分演讲者的培训也需要一段时间，因此在本博客中展示了针对 5 位演讲者的培训。相同的代码可以用于更多的扬声器。

## 模型实现

#### 从子集 VoxCeleb1 提取 MFCCs 并训练 iVector 提取器

1.  对于这个博客的范围，来自 dev 的 5 名测试人员和来自 test 的 5 名测试人员来自 VoxCeleb1 数据集。UBM 训练，I 向量训练是在 dev 集上执行的。测试集用于评估模型性能指标。实际上，用于训练 UBM 的扬声器数量要高得多。

2.  无声检测 **-** 包含人类语音的. wav 文件通常由语音和无声片段组成。提取的 MFCCs 应该来自检测到话音活动/语音活动的语音帧，而不是无声段。如果 MFCCs 是从来源于无声段的帧中提取的，则不是目标说话人特征的特征将在 GMM 中建模。可以训练隐马尔可夫模型(HMM)来学习语音文件中的静音和语音片段(参考[pyaudionanalysis](https://github.com/tyiannak/pyAudioAnalysis))。为了训练 HMM，将语音段和静音段的时间戳馈送给模型。
3.  鲍勃留下来

*   MFCCs 由函数[4]提取

```py
bob.kaldi.mfcc(data, rate=8000, preemphasis_coefficient=0.97, raw_energy=True, frame_length=25, frame_shift=10, num_ceps=13, num_mel_bins=23, cepstral_lifter=22, low_freq=20, high_freq=0, dither=1.0, snip_edges=True, normalization=True)
```

*   训练通用背景模型。通过以下函数[4]在 Voxceleb1 的 5 说话人开发集上训练 UBM 模型

#### 对于全局对角 GMM 模型

```py
bob.kaldi.ubm_train(feats, ubmname, num_threads=4, num_frames=500000, min_gaussian_weight=0.0001, num_gauss=2048, num_gauss_init=0, num_gselect=30, num_iters_init=20, num_iters=4, remove_low_count_gaussians=True)
```

#### 全协方差 UBM 模型

```py
bob.kaldi.ubm_full_train(feats, dubm, fubmfile, num_gselect=20, num_iters=4, min_gaussian_weight=0.0001)
```

*   训练 i-Vector (extractor) (dev)集-I-Vector 训练是通过使用 dev 集和 UBM 模型的 MFCC 特征来完成的。这导致来自语音发声的 600 维阵列/嵌入。用于训练 I 向量的函数是

```py
bob.kaldi.ivector_train(feats, fubm, ivector_extractor, num_gselect=20, ivector_dim=600, use_weights=False, num_iters=5, min_post=0.025, num_samples_for_weights=3, posterior_scale=1.0)
```

*   提取 I 向量——一旦 I 向量训练完成，就可以通过下面的函数在任何语音上提取 I 向量。wav 文件[4]

```py
bob.kaldi.ivector_extract(feats, fubm, ivector_extractor, num_gselect=20, min_post=0.025, posterior_scale=1.0)
```

## 模型性能

通过计算真阳性率(TPR)和假阳性率(FPR)在测试集上评估模型性能。TPR 和 FPR 按以下方式确定:

每个说话者都有一定数量的话语，与之对应的是 I 向量。I 向量通过余弦相似性得分相互比较。如果分数高于某个阈值，则说话者被认为是匹配的。匹配被归类为阳性，不匹配被归类为阴性。CSS 通过比较测试 I 向量(w_{\text{test}})和目标 I 向量(w_{\text{target}})之间的角度来工作

$ $ \ begin { equation } \ bold symbol { S }(\ hat { \ bold symbol { w } } _ { \ text { target })= \乐浪\ hat { \ bold symbol { w } } _ { \ text { target } }，\ hat { \ bold symbol { w } } _ { \ text { test } } \ rangle \ math bin {/} \ left(| | \ hat { \ bold symbol { w } } _ { \ text { target } } | | \\\\\\\\\\\\\\\\\\\\\\\| | \ hat { \ bold symbol { w } } _ { \ text { test } } | | \ right)\ end { equation } $ $

在这种情况下，为每个说话者确定真阳性率和假阳性率，并计算总平均值。这反映了分类器的性能。

本文有一个附带的项目，包含数据和 Python 代码。此外，伴随项目的[的 README.md 文件中的指令提供了关于下载预训练模型的指导，其中 UBM 针对更多数量的扬声器进行训练。正如所料，该模型的性能优于本博客中使用的较小模型。你可以通过注册一个](https://try.dominodatalab.com/u/nmanchev/speaker-verification/overview) [免费多米诺骨牌试用](https://www.dominodatalab.com/trial/) 来获得它。

[![New call-to-action](img/a5df909512743f17c9529d550f77a27d.png)](https://cta-redirect.hubspot.com/cta/redirect/6816846/5cb2ca1e-fedf-47bf-b9b7-c98585a796f2) 

**参考文献:**

*[*https://www . allaboutcircuits . com/technical-articles/an-introduction-to-digital-signal-processing/*](https://www.allaboutcircuits.com/technical-articles/an-introduction-to-digital-signal-processing/)*

 **[*https://www . researchgate . net/publication/268277174 _ Speaker _ Verification _ using _ I-vector _ Features*](https://www.researchgate.net/publication/268277174_Speaker_Verification_using_I-vector_Features)*

 **【3】*[](https://ieeexplore.ieee.org/document/5713236)

 **【4】*[*https://www . idiap . ch/software/bob/docs/bob/bob . kaldi/master/py _ API . html # module-bob . kaldi*](https://www.idiap.ch/software/bob/docs/bob/bob.kaldi/master/py_api.html#module-bob.kaldi)

*[](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html)*

 ***【6】*[*https://www . idiap . ch/software/bob/docs/bob/bob . kaldi/master/py _ API . html # module-bob . kaldi*](https://www.idiap.ch/software/bob/docs/bob/bob.kaldi/master/py_api.html#module-bob.kaldi)

*[*https://www . researchgate . net/publication/268277174 _ Speaker _ Verification _ using _ I-vector _ Features*](https://www.researchgate.net/publication/268277174_Speaker_Verification_using_I-vector_Features)*

 **【8】*[*https://engineering . purdue . edu/~ ee 538/DSP _ Text _ 3rd edition . pdf*](https://engineering.purdue.edu/~ee538/DSP_Text_3rdEdition.pdf)*【9】*[*Kamil，Oday。(2018).帧分块加窗语音信号。4.87-94.*](https://www.researchgate.net/publication/331635757_Frame_Blocking_and_Windowing_Speech_Signal)******