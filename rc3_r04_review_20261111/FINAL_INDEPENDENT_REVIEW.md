# R04独立终审

**交付判定：GO_ENGINEERING_AND_DESIGN。科学状态：NOT_READY / NOT_AUTHORIZED。**

本终审接续`STATIC_REVIEW.md/.json`，保留初审REVISE及其修复记录。只读取本轮新收据、登记/授权、事件尾记录和必要设计文档，校验交付收据的SHA256；没有重新执行单元测试、CSV算术或任何模型、训练、场景、导航、教师、native、H、原60或S3操作。下面数字是本轮执行收据的记录，不是终审另跑的统计结果。

## 工程结果与资源

| 证据 | 已执行结果 | 能支持的结论 |
| --- | --- | --- |
| `UNIT_TEST_RESULTS.json` | 37项通过，failure/error均空 | 人工张量、fit-only人工记录的公式、单位、mask、guard/假状态恢复及schema契约通过 |
| `OFFLINE_INTERFACE_RESULTS.json` | 固定前8组保存logit通过，真实SL加载/encoder调用/完整路线均0 | 人工零206维feature下零头合法logit与mask精确一致，standalone mapping往返一致；人工PPO/C1 loss及梯度隔离通过 |
| `EVIDENCE_AUDIT.json` | 41,624,938项checks全部通过；新增streaming覆盖399,350条prediction-component行、1,568条summary、784组 | 已保存CSV的来源与算术一致；继承纯审计函数的部分仍是维护复核，新增分层矩统计是独立实现 |
| `FROZEN_BYTES_CHECK.json` | 619个source、32个artifact复核PASS | 本轮runner绑定的历史/R04输入字节在执行前后保持一致 |
| `RESOURCE_ACCOUNTING.json` | PASS，wall 150.83922809999967秒，上限900秒 | 本轮登记工程完成且未触及资源上限 |

资源收据记录：module forward 256/2048，backward 4/64，autograd.grad 0/64；source reads 1327/10000、hashes 1316/5000；37/37 unit case、1/1 offline suite、1/1 saved CSV audit。synthetic deserialization为0，真实权重加载为0。Actor/Critic optimizer、scientific run、新场景、navigation、teacher、native、H、原60与S3读取计数全部为0。手工参数赋值及fake Adam形状对象没有被描述成学习更新或真实Adam验收。

资源账绑定的registration hash与实际`ENGINEERING_REGISTRATION.json`一致；`ENGINEERING_AUTHORIZATION.json`也绑定相同hash，范围只含本轮零优化器工程。资源账与事件close的状态、wall、计数、journal tail一致。本终审读取事件尾记录，没有另行重算整条事件链，也未重新校验全部619项输入；完整执行前后字节结论由新runner的收据承担。

## 数学定义与四个最终回答

数学定义与原型接口一致，可以作为候选设计交付：固定初始n贯穿整条轨迹；H是有符号整轨迹会计信用，不是物理剩余距离；268维观测不被宣称Markov充分。C1在归一分量空间输出，raw适配再交旧loss只归一一次；signed H与三分量value组合一致。训练归约与评价聚合分别定义；新M1使用SUM匹配未来PPO，前轮MEAN离线证据不追改。

相对Actor只消除pre-tanh共同加性方向；post-tanh均值未保证为0，额外post-center可能突破幅度界。±0.25及pair差至多0.5保留，gap>0.5的SL唯一首选仍不可翻转。guard在人工近tie用例允许首选翻转，因而不是路线成本、安全或真实IK证书。

四个最终回答均准确，需保留当前限定：

1. **C1可作为新Critic的候选基础。** 依据是已有两种口径相对C0的数值改善和最小输出单位修改；它仍没有胜过条件常数，不能被写成已获联合训练资格或稳定泛化证据。
2. **条件baseline尚不构成必要模块。** 它应保留为强对照；可训练条件残差须另行证明超越anchor自身、global-anchor残差和C1，当前仅有表/lookup/分量合成接口。
3. **居中不代表共同偏移问题已经解决。** 公式不变性及反例已获工程验证，但没有新Actor训练、相对排序收益或完整成本收益证据。
4. **固定距离特征下C0/C1×A0/A1是本提案的最小可识别四臂pilot。** 需要同SL256/零残差起点、各臂自身在线轨迹、匹配监督控制及完整raw/H成本与请求时间；预算与停止规则依然只是未批准候选。不能用C1+A1单臂对比历史R03终点作因素归因。

C0/C1同净张量对应不同物理初值，故不能作纯梯度尺度因果声明。G3是保守项目进入门槛，不是校准过的总体5%错误率多指标检验，也不是H2成立的逻辑前提。若固定16父缺尺寸覆盖，应报INSUFFICIENT；不换父、不删例、不使用state数增加独立样本。

## G2/G3与科学阶段阻断

**G2仍未通过。** 旧adapter的导出方法会调用新head不存在的旧方法；旧production export可能把同形状relative权重错标为absolute。wrapper/export/loader/checkpoint须共同迁移并绑定Actor/Critic语义、206/268特征、归约、SL/H与来源身份。typed C1还须显式接入旧trainer的admission、critic step、rollout和日志。真实SL恢复、actor-only完整请求等价、真实Adam候选接受/拒绝回滚，以及RNG/游标/累计资源恢复，均未由本轮工程覆盖。真实optimizer验收必须另行登记和授权。

**G3仍未满足、未执行新M1。** 历史C1在两个聚合均未胜条件常数，C2存在归一H信用退化。本轮算术PASS不改变该结论。新的数据/source/H/environment/seed/初始化绑定、固定reserve父与覆盖检查、0/128/256结果和预先规定的门槛仍须独立阶段授权后实施。不得把G3失败直接写成A1/H2被否证。

**协议批准与阶段执行授权均未获得。** 本目录设计和工程交付GO不激活R04科学协议、不恢复R03-256、不继承旧剩余额度、不解封原60/S3、不自动追加条件baseline/C2/特征、步数或seed。原活动指针与历史证据维持冻结。

本轮可交付“方案与接口原型工程通过、历史算术复核通过”。不能交付“规划性能改善”“真实机械臂认证”或“联合PPO已获许可”。GitHub提交、推送及真实远端SHA核对由根代理完成，本独立终审不预先声称已同步。
