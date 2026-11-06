# v3 执行候选协议定稿与批准边界

日期：2026-11-06。基准：`aa0247c8c38243504028bfad24795e0db7120264`。本轮固定研究目标保持“大规模复杂父场景DRL训练—冻结策略快速完成不同规模局部任务的完整规划”；监督、规则和教师仅为辅助与对照。

**协议定稿核验：PASS；文件仍待用户批准，活动设计仍为v2。** 本文与新协议不授予真实教师生成、科学optimizer、S3或原60任务访问权限。原v1/v2、v3提案、S0及活动pointer保持原字节。

## 建议批准的准确文件

| 文件 | SHA256／作用 |
|---|---|
| [V3_EXECUTION_CANDIDATE.json](V3_EXECUTION_CANDIDATE.json) | `0c5bc7d7a9557edf2af76f51a9f418550f0b4ce1ec6a8b47ddea95c6256e78bb`；推荐的完整待批准协议 |
| [REGISTRY.json](REGISTRY.json) | `75d110bef7e9954e9ace14198d262a940af6aacb17abad71cf83771ace8b363a`；原v3 registry逐字节复制，不新增或改变声明 |
| [H_BINDING.json](H_BINDING.json) | `4bc694068672afe0b558d16a41232400fce2410c5450824ded9d6725f0ed9b54`；冻结H源文件逐项hash及canonical聚合规则 |

H identity为`5db67cacc1128b6143c7a04b9b84a70c37bdc0ab917a0eebae0b8aca132c4d14`，由原S0的H文件hash映射按排序紧凑JSON计算SHA256。H两文件本身不是完整运行环境；最终`EXECUTION_SOURCE_BINDING.json`还必须绑定端点导出、验证、导航、环境、训练器及新的阶段授权适配器全部源码和依赖，并引用工程回归结果。协议只引用独立绑定文件，避免把自身hash嵌入自身造成循环。

## 八项定稿结果

| 核对项 | 明确的执行定义与批准性质 |
|---|---|
| 1．S1教师重分配 | 256条完整声明见[清单](S1_TEACHERS_256.csv)。mixed/blocked × closed/open四格各64；尺寸4×3/6×6/8×8为86/84/86；64个父标签出现3尺寸，32个出现2尺寸。256个map/task输入均属于原96×8训练银行。与v2相比213条的端点、尺寸或origin改变，属于必须批准的样本分布修订，不能称工程修复。 |
| 2．S2 RNG | 采用详细RNG条款：protocol SHA、phase、optimizer seed、update、slot、role派生独立流；S2_data随optimizer seed分别可重放，不宣称三seed共享相同数据日程。registry的11632026保留为bootstrap种子声明，不再混作共享S2抽样流。新候选SHA不同会事前改变命名派生流，map/task seed银行未变；不能声称与旧提案轨迹逐位相同。 |
| 3．单合法首动作 | unique_first=1时两辅助角色均强制同一合法首动作，但独立采样后缀，仍各权重.25且不计forced首步logprob；主行正常计首步概率。不静默去重、重加权或另换样本。 |
| 4．优化器与谱系 | 每seed独立初始化并保存untrained；S1固定256步masked teacher CE；冻结SL、RL和继续SL从本seed同一SL固定终点逐位clone，后两者fresh Adam，S1动量仅存证不继承。Adam1e-4、betas(.9,.999)、eps1e-8、wd0、clip1等详细配方沿用原v3。工程/坏checkpoint不能充当正式来源。 |
| 5．调用与墙钟 | S1-A共享教师最多256 native+256H=512次、1800秒、0optimizer。S1-B每seed256更新/batch4/1800秒、0新教师。S2每seedRL512+继续SL512、RL6144+开发/其他2048=8192次、共同7200秒。输入生成、F&A、导航、读盘、恢复、源SL载入/分叉、开发和IO都归调用阶段，不设免费离线准备。 |
| 6．S3声明与A | [108条声明](S3_DECLARATIONS_108.csv)保持原v3 bytes。与v2相比72例仅A条件变化；seen各尺寸A0/1/2各4例，unseen各尺寸各8例。只读metadata，不生成目标task、路线或成本。请求上限5292/seed准确分为756独立质量请求+4536独立计时请求，后者为108×7方法×3重复×2cache；质量计算不得免费复用到计时。 |
| 7．聚合与AND门 | 先每物理parent平均两endpoint成本，再平均三个匹配seed成本，再计算parent相对收益并等权汇总；不先求子例相对收益。parent-cluster bootstrap20000、seed11632026；seen/unseen分开。整体RL主张需全部6个group×size质量格对两SL控制均过门，整体部署主张需全部12个group×size×cache格过门，组合主张再对二者取AND；不以池化救失败格。既定阈值数值不变，但v2→v3的聚合与AND必要条件是需批准的协议变化。 |
| 8．历史边界 | v1→v2纠正实际物理布局身份与map模式；该历史amendment的“预算门槛不变”仅适用于该次转换。v2→v3另有配方、分布、预算范围、A与统计决策变更，不能沿用该不变声明。新候选把历史amendment单独列出并明确scope。 |

正式provider沿用已验收的`make_complex_scene -> resample_tasks -> F/make_allow -> canonical几何/source绑定 -> 全父目标有效连接检查 -> 局部提取`链，不再沿用旧v3文字中尚未作为正式provider实现的“task-free factory”说法。临时constructor任务不是另一个科学样本，但其所有构造和重采样成本必须计入准备。任一声明任务不可行都保留收费失败并停止，不能换seed、删例或重抽。

## 阶段批准和独立启动

候选规定顺序为S1-A一次共享教师池→S1-B三个固定seed按11622026/27/28完成→另行批准S2并按同顺序执行。S1-B仍允许单seed独立调用和合法恢复；只有三seed固定端点都完成，才满足S2的前置管理条件。这一操作门在候选差异中显式记录，不改变算法或总预算。

新的阶段授权衔接必须把稳定科学身份（协议/registry/源码/H/实验ID和权重谱系）与每阶段不同授权文件分离，保存源产物hash及其原阶段授权证据。后续只批准S2不能重写S1来源、重开已完成S1、重置账或自动取得S3权限。实现和有限回归由独立阶段适配器交付；此定稿报告不提前声称适配器验收结果。

本轮没有生成三个正式untrained权重。它们在S1-B获准后按各固定seed实际初始化并保存；当前工程权重和坏恢复夹具均不是正式模型。

## 完整差异表与草稿纠正

[JSON差异](PROTOCOL_LEAF_DIFF.json)与[CSV差异](PROTOCOL_LEAF_DIFF.csv)按JSON-pointer逐叶、逐数组元素展开，无截断，包含六个比较集：

| 比较集 | 叶变化行数 |
|---|---:|
| v1→v2 | 24 |
| v2→原v3提案 | 112 |
| 原v3提案→新候选 | 156 |
| v2→新候选 | 245 |
| v2 registry→v3 registry | 860 |
| 原v3 registry→候选registry | 0 |

合计1397行包含比较集之间的重叠与版本metadata，不能解读为1397项独立方法改动。与旧评审“108字段差异”数字不同，是本表把数组元素完全拆分的口径不同；原报告原字节未修改。

本次初稿曾将S3的756次质量请求误称未分配余量。独立审查在任何任务执行前纠正，原初稿SHA`5e5887ab3cd955abdd12ee8559665cc94a2d1643377bcf22afb94ebccad866ee`原字节保存在`drafts/`，见[DRAFT_HISTORY.json](DRAFT_HISTORY.json)。最终请求总额仍5292，方法、种子和阈值均未变；没有使用S3任务或结果作修订。

## 核验与当前决定

[定稿核验收据](PROTOCOL_FINALIZATION_RECEIPT.json)核对全部JSON、CSV逐行对应、原registry字节、数值优化配方及预算、256/768/108声明、H源hash和状态。自动metadata核验0.0240412秒，不包括此前静态阅读/起草；scene、nav、teacher、model、forward/backward、optimizer、S3与原60语义调用均为0。

候选可提交用户审阅批准；当前未批准、未激活，正式科学执行不因此获得GO。阶段衔接代码和新源码绑定须通过工程审查，并由用户分别批准准确协议与对应阶段，才可进入真实S1-A或S1-B。性能结论只能由之后正式运行产生。
