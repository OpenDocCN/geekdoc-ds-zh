# 初创公司如何在成长过程中获得适当的安全保障

> 原文：<https://review.firstround.com/how-early-stage-startups-can-enlist-the-right-amount-of-security-as-they-grow>

**[迈克尔·格拉汉姆](https://twitter.com/grahamvsworld "null")** 回忆起启动他安全职业生涯的副业。作为一家大型运输公司的分析师，他安装了一系列开源网络入侵检测设备(这在当时是一种全新的技术)，并开始重建底层操作系统以使其正常工作。他的目标是捕获足够的流量，以便更清楚地诊断最近的网络问题。两周后，臭名昭著的电脑蠕虫[红色代码](https://en.wikipedia.org/wiki/Code_Red_(computer_worm) "null")来袭。虽然追踪包裹和飞机的技术完好无损，但几乎整个公司网络都瘫痪了。当公司开始着手隔离和控制问题时，他开始深入研究他的项目的初期日志系统。他发现了承包商的笔记本电脑——放在他下面两层楼的地方——最初的感染就是从那里开始的。

如果那个兼职项目是他的全职工作呢？他本可以为公司省下很多麻烦和金钱。就在那时，格雷厄姆知道安全不是次要的；这是一项关键的业务职能。识别威胁并迅速采取行动是决定一个公司能否高效运营的关键。识别和最小化这些危险决定了一个公司是否能正确规划。在过去的 15 年里，Graham 在安全架构领域担任过高级职位，包括 [Harrah's Entertainment](https://en.wikipedia.org/wiki/Caesars_Entertainment_Corporation "null") 、 [Zynga](https://www.zynga.com/ "null") 、 [Evernote](https://evernote.com/ "null") 和 **[Box](https://www.box.com/ "null")** ，他目前是安全架构和工程的高级经理。在他的职业生涯中，他领导了跨阶段和跨部门的安全工作:从初创公司到财富 500 强公司，从公共教育到制造业。

在这次独家采访中，Graham 概述了初创公司在每个成长阶段应对安全问题的不同方式，包括他们应该何时雇佣第一批安全人员。他还分享了如何在更广泛的组织中更好地定位安全性，以及他对未来十年该职能发展的看法。

# 如何看待初创企业的安全支出

许多资源紧张的初创公司通过评估公司的财务支出来衡量他们对安全的承诺水平。相反，格雷厄姆建议根据公司可能面临的风险来定义安全支出。“对所有公司来说，损失多少钱是有限度的。所以，如果你花的钱超过这个数目，你肯定会把事情搞砸，”格雷厄姆说。“根据你如何处理客户数据以及如何利用这些数据赚钱，你可能损失的资金也是有限的。如果你花的钱超过这个数目，你也会陷入困境。”

格雷厄姆承认，这些主张与许多营销信息背道而驰。大多数创业公司让客户面临的风险比他们出售的产品更大。“如今营销中使用了大量的社会资本。诸如“您完全可以相信我们会保护您的数据”之类的陈述格雷厄姆说:“很抱歉，但当你是一家六人公司时，事实并非如此。你没有检查所有的东西。你没有做非常深入的代码审查。你没有处理好每一个可能的现实威胁。作为一家初创公司，您正在进行大量艰难的优先级排序，但不幸的是，安全性很可能并不总是排在第一位。"

这一现实既归因于早期创业公司的挑战性道路，也归因于更广泛的行业力量。“撇开公司阶段的环境不谈，这种现实也是在风险资本驱动的环境中做生意的副作用。格雷厄姆说:“你在用自己的时间和他人的金钱来尽快获得客户。“在这种模式下，当信任相关时，获得客户可以迅速转变为代表信任。这不是花尽你所能让信任变得合理。无论您是否同意这种立场，在构建威胁模型来保护漏洞时，都要承认这种心态。一旦您确定了那些易受影响的点，就更容易建立控制来更好地保护它们。然后，就像创业世界中的其他事情一样，你必须合理安排你的支出以产生影响。”

安全不是将风险降低到零。这是关于随着环境的变化不断重新平衡风险。

要评估风险，首先要寻找一个权衡的结果，这个结果将随着业务的发展而被快速和频繁地重新评估。格雷厄姆说:“每一个早期威胁都是存在的，所以创业公司能够在没有未知瘫痪的情况下运营的唯一方法是在威胁的滑动尺度上确定自己的容忍点。”“您可以虔诚地对所有内容进行逐行代码审查，或者花几个小时审查进入您的操作系统或应用程序容器的每个软件包。你可以把安全提升到第 n 级，把任何风险都烧光。但你也可能会把产品的速度和盈利能力都拿掉。”

鉴于他所引用的具体案例的细节的敏感性，Graham 不仅从他的角度，而且从他在该领域的同行的角度，抽象出了他如何看待初创公司安全性的演变。从他的角度来看，这是早期公司如何平衡安全的风险和回报。鉴于初创公司的增长轨迹多种多样，他大致按照公司员工人数进行了分类:

**1-9 名员工。**“在这个阶段，避免会让你陷入困境的额外安全措施。使用双因素身份验证。不要把你的服务器放在互联网上，暴露你所有的服务。把它们放在别人的云上，并确保你已经锁定了你的资产，以至于机会主义者对它们不感兴趣。很可能没有人瞄准你或者你的潜在客户。如果有人在您的基础设施中随机出现，您只会遇到真正的安全问题。让这不太可能。在 [AWS](https://aws.amazon.com/ "null") 上开店大约需要一天的时间来满足你的初始需求。只管去做，然后继续发展你的产品。”

**10-19 名员工。**“员工人数达到两位数，你开始分配职责。你已经过了每个人都在做每件事的阶段，但仍然处于决定做某件事只需要五分钟同步，而不是团队会议的阶段。这是你希望有人负责安全的阶段。这不需要——也很可能不会——是她的全部工作，但它必须在她的权限范围内。很可能是你的首席技术官或任何可能成长为这个头衔的人。这个人监控基础设施，决定代码如何被推送，并且知道秘密在哪里。她还没有做出关于风险的艰难决策，因为其他人对业务、代码库和基础设施都很熟悉。但她能够采取防御姿态，确保没有人做任何会吸引不必要注意的事情。在这个阶段，公司与威胁它的任何人都处于一种脆弱的、不同步的关系中。攻击者比初创公司拥有更多的资源和时间来防御入侵。”

20-49 名员工。“如果你活下来了，你就有了一个产品，并与客户建立了关系。这意味着你是他们信任的伙伴，不管你是否意识到这一点。他们给你某种程度的信息，甚至可能是他们客户的信息，而你正在利用这些信息做事。现在你不仅有你可能的威胁，而且还有那些可能以你的客户为目标的人。这就是它变得危险的地方，因为如果你在处理有趣的数据，你就有可能成为真正的威胁者。此外，你可能会雇佣会编程的工程师，但他们可能不知道如何审查代码或设计一个安全的系统。最后，现在您开始扩展您的基础架构及其相应的安全控制。有了这么多活动部件，裂缝开始出现。老实说，我认为很多公司都是靠运气度过这一关的。我不确定还有比这更好的答案。”

50 名员工。“你已经 50 岁了。可能会有一位顾问、联合创始人或投资者认为，如果一切进展顺利，你可能会在一年内拥有 100 名员工。如果你还没有，现在是你应该雇佣第一个全职保安的时候了。你会听到这样的话:“我们只能招 10 个非工程人员……我们需要法律、合同、人力资源等方面的人。”通常情况下，全职员工不会被聘用，下一个转折点是 200 到 300 人左右。这是感觉像一个“真正的”公司将会成功的关键。然后，人们被雇佣来解决运营、安全、人才等方面的大问题。但是当那个人进来时，他们不仅仅是从零开始——他们获得了与他们学科相关的所有技术债务，因为以前没有人完全拥有它。"

# 在船上，你的第一个全职保安雇佣 30-100 名员工。

格雷厄姆提供了广泛的范围，不仅解释了初创公司的不同增长路径，也反驳了他作为安全专业人士的偏见。“当然，我鼓励人们在第一次安全招聘时雇佣 30 人左右，而不是 100 人。格雷厄姆说:“这是因为创业公司的技术和业务同时呈指数级增长，变得更快、更复杂。“危险在于，早期初创公司的业务领导通常将安全性视为一个技术问题，而工程领导则认为这是一个可以通过应用他们现有的技术知识和严谨性来解决的问题。这是一个幼稚的错误。”

尽早雇用安全主管的理由与在关键业务关头雇用任何专业人员的理由相同:不仅是因为她的判断，也因为她的能力。格雷厄姆说:“你雇佣的是一个你信任的人，而不是因为她会加入并开始盲目地在所有事情上涂抹安全程序和严谨性。”。“你希望她与团队交流，查看代码，并告诉你哪里会有麻烦。她应该能够证明，她能够在公司发展的过程中保护公司，而不仅仅是在散兵坑里保护初创公司。最后，她应该[开始修复旧的安全技术债务，并拦截和减少新的债务](http://firstround.com/review/shims-jigs-and-other-woodworking-concepts-to-conquer-technical-debt/ "null")

不可否认，一个 30 人和 100 人的公司之间的差异是显著的，选择合适的时间进行第一次安全招聘在回想起来总是更清晰。“如果你在做一些‘高度信任’的事情，比如存储某人的税务文件，或者运送一个信息客户端，那么雇佣一个接近 30 人的团队。Graham 说:“如果在发布 MVP 之前有合规性要求，那么你显然属于这一类。“还有一些公司的产品不包含安全性，不包括互联网上的任何东西都应该有防止被黑客攻击的措施。这些公司并没有打造高信任度的产品。我仍然建议公司尽早聘用第一名安全人员，但当员工人数达到 100 人时，这些类型的初创公司可以聘请安全负责人。”

![](img/aafa4c7f538163e62e3b274ef0addc09.png)

Michael Graham

格雷厄姆制定这个安全雇佣时间表的原因是，当一家初创公司成为“真正的公司”时，安全基础尚未建立，这是很危险的。“如果你等到那个时候才加强安保，公司和第一个安保人员都会后悔的。在一个组织中，Graham 在一次入侵后加入帮助组建安全团队。“在面试过程中，格雷厄姆和其他几名候选人建议他们雇佣一个团队，而不是一个保安，”格雷厄姆说。“组织已经过了一个人可以应对挑战的阶段。它通常是一家有安全意识的公司，大多数正确的安全措施和对违规的响应都是正确的。幸运的是，他们凭借自己的工程技术勉强过关，但他们的方法已经暴露了长期的安全漏洞。”

安全的偏见是避免负面后果。工程师寻求优化积极的结果。两者在技术上可能都有能力处理安全实例，但最终，他们的动机将是构建截然不同的安全系统。“在出现漏洞的情况下，安全人员会安装一个系统来进一步降低风险。例如，密码可能在入侵后被泄露。Graham 说:“安全架构师会使用哈希机制来加大‘第二次攻击’的难度。“这会节省工程和客户成功团队的时间和资源。这些团队本可以更有把握地说:‘我们有理由相信，他们需要几个月的时间来破解这些密码。如果你在其他地方使用这个密码，去修改它，但不要担心。"

# 如何在您的初创企业中巧妙定位安全性

初创公司将安全与“老大哥”联系在一起并不罕见，那些从事安全工作的人感觉自己就像是在照看员工。对于许多代人来说，安全性意味着管理防火墙，与该功能的任何交互在很大程度上都与策略执行有关。安全成为工作中行为的事实上的警察，监控你如何浏览互联网和你安装了什么，直到公司内部的社会规范。

“安全性作为一个商业概念仍在从那个时期恢复过来。员工认为安全性是一个障碍，是一个业务难题，也是一个昂贵的成本中心。格雷厄姆说:“这在某种程度上是对的，但不是全部。“员工认为保安在房间里扮演‘老大哥’我不知道在现代安全团队中有谁会这样看待自己。这种假设是过去遗留下来的。"

鉴于这种偏见，**今天的安全职能部门——以及一家初创公司的创始人——的责任是宣扬其作为合作保护者和建议来源的真正角色**。“这是一场艰苦的战斗，因为安全部门和整个初创公司都受到资源的限制，并且经常在追赶。这通常会妨碍他们主动提供内部建议。格雷厄姆说:“他们更像是一个保姆，只是在漏洞被破坏之前抓住它们。”。

以下五种方法可以帮助安全从老大哥或保姆转变为顾问:

# 培训同事对信任决策采取双重态度。

初创公司会采取重大措施来促进用户的特定行为。当涉及到安全措施时，可以在内部应用相同的方法。“我们的目标不是让员工自己进行威胁建模。谈到风险，大多数人会想到可能发生的最糟糕的事件，希望这种可能性消失，然后继续前进。格雷厄姆说:“观点的轻微转变会带来很大的不同。相反，引入多一个检查点，让员工暂停一下。一种方法是向员工展示他们在工作中的信任决定——从安全的角度来看。对于工程师来说，这可能是在验证系统的特定点验证一切正常。如果一切正常，安全应该没问题。”

如果员工能够始终如一地快速仔细检查，那么安全团队需要扑灭的火灾就会大大减少。“如果你能教人们在工作中遇到安全问题时三思而行，你就能解决三分之二的安全问题，”Graham 说。“通常引发问题的人会实施修复。即使这种修复只是花 20 秒钟标记一个看起来不寻常的电子邮件附件。如果员工因为知道查看文件是做出信任决定而暂停，则安全性获胜。然后，it 可以专注于解决高度技术性和特定领域的安全问题，而不是救火。”

为了养成三思的习惯或本能，员工可以问自己六个问题:

我在等这封邮件或电话吗？

我如何确认这个要求我做或分享的人就是他所说的那个人？

如果我告诉某人一些敏感的事情，我确定她应该知道吗？

我对别人会用我正在构建的东西做什么做了什么假设？

如果我在谈论我们如何管理秘密、认证或会话，我是否向别人征求过建议？

如果我的决定是错的，我们怎么知道呢？

这种自我询问不会取代帮助建立威胁模型、实施控制甚至只是提供快速建议的安全主管或团队，但这是一种帮助防止重大失误的广泛方式。问题背后的主旨和主题总是“这是意料之外的吗？”以及“我是不是在做一个很大的假设？”

看到什么，就说什么。PSA 对初创公司的安全性来说是正确的。这是防止员工成为零号病人的最好方法。

# 杀死羞耻游戏。

鼓励员工主动寻求安全的主要方法是立即走出责骂的循环。“人们不会与你互动，因为他们害怕会被告知一些他们不想听到的事情。他们可能害怕会被告诫。这并不是安全性所独有的。格雷厄姆说:“公司内部的任何咨询部门都是如此，比如人力资源或法律部门。“当每一次互动都充满压力时，就会建立一种糟糕的关系，这种关系最终会变得非常昂贵。归结起来就是:不要骂，不要为难人。羞耻不会改变行为。这会让你试图避免的一切更有可能发生。”

# 奖励侦探。

许多公司都有认可超出工作范围的员工的计划——安全部门也应该这样做。“无论是简单的一封电子邮件还是全体员工的认可，都要找到一种低调的方式来认可提出安全问题的员工。格雷厄姆说:“无论是公开还是私下，脱帽致敬都应该反映出你公司的文化。

“在 Box，我们有我们所谓的安全之星计划。格雷厄姆说:“我们分发实物贴纸，现在正在设计一个新的奖杯作为奖励，因为当人们花点时间提醒我们时，这是一个巨大的胜利。”“如果这是一场针对公司一群人的网络钓鱼活动，我们将表彰抓住它的员工。我们突出显示这个人，感谢他们这么快就通知了我们。员工是我们的头号侦查来源。”

安全方面，我不需要阻止每一件发生的坏事。我需要有足够的检测能力和能够通知我的同事，这样我们才能及时做出反应。

# 忘记教育，强调邮件。

培训员工的其他途径包括教育方法，如模块、手册或定期召开的安全摘要会议。Graham 不太看好这些策略，尽管法规遵从性经常推动这些项目向前发展，但他并不建议将它们作为您防御的关键支柱。“我从未见过真正有效的安全教育项目——至少没有一个项目能提高参与者的独立决策能力，”他说。“最终，你想要一个开放、舒适的渠道。有效的教育是让人们重视电子邮件安全。你需要公司里的每个人都知道——事实上——就某件事发电子邮件通知安全部门从来都不是错误的做法。你的‘谢谢’有一个快速响应的过程。”

抛开所有的手和电子邮件不谈，事实证明，对格雷厄姆来说，最成功的方法是签字，并对收到的邮件做出快速回复。“保持信息简短明了；你如何签字是最重要的。如果你不确定，那么“电子邮件安全”应该放在你对任何人说的几乎所有话的末尾。格雷厄姆说:“我不会把它作为电子邮件签名，但每次想到它，我都会用不同的方式打出来，所以它是真实的，而不是自动的。”。“当我收到一封电子邮件时——不管是什么邮件——我都会感谢发邮件的人。他停下来考虑情况，推迟了工作给我们写信。那是巨大的。那是模范行为。”

# 可见。

从员工的角度来看，很多关于安全性的东西最初都是未知的:恶意黑客、漏洞的来源和危害的程度。正因为如此，安全团队不能隐身，这一点很重要。“对于团队来说，保持可见性和可识别性是安全管理的职责之一。不是为了强化刻板印象，但是很多人都处于安全之中，因为他们发现问题比人更有趣。格雷厄姆说:“这并不是说他们不关心人，他们只是深深着迷于技术。“他们觉得解构比建造更有趣。他们发现破坏一个创建良好的系统比创建一个良好的系统更有趣。你在安全领域有很多人，他们想弄清楚其他人是如何破坏的，并在非常深入的技术层面上做这件事，这需要极大的关注。”

密集的学习并不总是有利于工作中的社交参与，但安全领导必须如此。“尤其是在一家初创公司的员工人数真正增加之后，抓住容易的机会让自己在公司中引人注目是非常重要的。有多少员工能从一排同事中挑出一个安全人员？安全部门需要更加公开他们的工作。格雷厄姆说:“员工更有可能接触到他们能认出的名字或面孔。“否则，如果你不在或不友好，那么‘老大哥’或‘保姆’的属性就会重新回到保安的名声中。在 Evernote、Zynga 和 Box，安全主管都是社交型的、在场的、内部公认的。这种行为可能是安全领导者职业生涯中最艰难的个人成长里程碑。我所知道的最好的人都很有魅力，鼓励互动，并对此很认真。”

# 安全的未来

Graham 从事安全工作近 20 年，在安全方面有着独特的历史理解。展望未来，他认为随着该领域从其他行业汲取实践经验，它将走向成熟。“在过去几年中，安全性更多地借鉴了统计和保险行业。每个人都听过这样一句老话:“你无法预测黑客会做什么。”“这是错误的，”格雷厄姆说。当然，我无法预测一个黑客会通过一个漏洞在一个系统上做什么，但我们绝对可以跟踪黑客作为一个集体会做什么，并开始预测导致的违规类型。"

格雷厄姆将绑架保险与此相提并论。“每一次绑架都是一个独特的事件，特定的人决定以特定的方式追捕一个人。然而，尽管是一个独特的事件，我可以得到保险。他说:“这意味着保险业已经搞清楚了大背景下的机制和经济学，他们已经算出了平均值，并且知道如果他们根据特定的分类决定为不同类型的人投保，他们就可以在保单有效期内获利。”

合规领域的统计分析最近也取得了一些进展。“有一些非常大的公司表现得好像他们知道如何去做，但事实是这是一个新兴领域。格雷厄姆说:“最近发生的事情是，统计人员与安全团队合作，测试实际产生有用输出的实践。“我认为在不到十年的时间里，我们将能够以对业务有意义的方式开始对安全性进行长期预测。这是我现在在 Box 工作的一部分。不久之后，世界各地的安全团队将能够不再把风险视为一系列抽象的威胁，而是用美元数字来表示损失或节省的收益。”

在安全领域，我们即将用美元数字准确量化风险和违规行为。这会让与领导层的对话和决策变得更好。

在创始人对工程师、产品市场适应性和盈利能力的快速追求中，很容易看到安全性可能会从优先列表中消失。这就是为什么格雷厄姆概述了具有重大影响的简单、高度杠杆化的调整。首先，从一开始就采取一些小的安全措施，但是在你雇佣了 30 名员工之后，再雇佣你的第一个全职保安。通过培训员工暂停信任决策、消除羞耻游戏和消除任何与主动员工拓展的摩擦，为更大的成果建立安全性。安全领导，把一个名字和你的职能联系起来。

“保护人民和财产是一项古老的服务。然而，当这些人是动态的数字原住民时，你谈论的是 T2 的知识产权，这是一个不同的——仍在发展中的——游戏。Graham 说:“有成熟稳定的框架和语言可以在一开始就解决一半的安全问题，但是没有人使用它们，因为它们没什么意思。”。“在许多方面，这个领域仍然是开放的前沿。安全是一种新兴的元实践，与构成技术行业的不断发展的领域一起工作。只要这些核心实践保持不变，安全性将继续是一门科学，也是一门艺术。随着我们努力使其更加基于统计数据，安全性将转变为更加严格的、基于工程的实践。但与此同时，如果你不确定，就给安全团队发电子邮件。”