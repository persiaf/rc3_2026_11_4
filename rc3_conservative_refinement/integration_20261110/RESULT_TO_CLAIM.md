# 结果与可支持主张

## 正式R03结论

**claim_supported=no；confidence=high（仅对本冻结配置），provisional，paper_claim_audit=unavailable。** 独立agent审查而非人类审稿，完整判断和读取范围见[独立正式审查](INDEPENDENT_FORMAL_REVIEW.md)及[结构化记录](INDEPENDENT_FORMAL_REVIEW.json)。

64/128各66个视图的ON/OFF完整raw/H路线及J/L/K均与FrozenSL完全相同；所有相关成本点估计/描述性CI为0/[0,0]。128继续监督只出现两条raw路线、一条H路线变化，ON未稳定优于该对照。ON完整请求中位时间为SL的1.1434–1.1642倍、Detour的1.1199–1.1750倍。Critic value MSE为warm固定常数的2.4122/2.3107倍。三组各128次接受、0拒绝，不支持guard拒绝造成无收益的解释。

可以支持：真实206维PPO、冻结/事务回滚、actor-only部署及有限工程复现链已运行；按预注册规则完成一个有界负结果配置，证据完整。不能支持：新距离信息有效、DRL优于SL/继续监督、部署价值、未暴露布局泛化、真实IK/实机能力。单分量预测改善、优势正负、有限梯度和软件PASS均不替代规划收益；无收益原因尚不能唯一归因。

128后已终闭合，256为NOT_STARTED，不补跑。下一步仅为[保存证据诊断及新协议提案](NEXT_STEP_DECISION.md)，保留大规模父布局DRL核心及全部冻结边界。正式数据见[事实报告](FORMAL_EXPERIMENT_REPORT.md)、[完整raw/H表](runs/conservative_paired_11622026_v1/analysis/ALL_RAW_H_ROUTES.csv)和[资源账](FORMAL_RESOURCE_ACCOUNTING.json)。

## 正式执行前工程快照

状态说明：以下表格是正式执行前的工程结论快照。之后用户已批准准确R03并授权正式阶段；当前正式结果已在上节独立判读，以下仅为当时工程结论。

| 证据 | 可以支持 | 不能支持 |
|---|---|---|
| 90项合成/接口测试；4种真实输入ON/OFF/plainSL严格一致 | 零修正初始接口与mask/路线正确，有限工程范围内无基座解冻 | 成本改善、长期泛化或真实机器人可用 |
| 真实有限PPO损失/优势/梯度与一次严格拒绝 | 新206维优化器链路可执行，拒绝后Actor/Adam精确恢复 | guard保证未知状态、路线成本单调改善 |
| 连续/暂停恢复与Actor-only重载一致 | 已检查闭合状态包含网络、Adam、RNG/游标/账，部署无Critic/Adam | 所有故障均可自动恢复、最终正式部署通过 |
| Fixed3/Detour16合法示例 | 原生基线接口可复用且无额外共同H | 新DRL优于原生规划 |
| 单次冷请求含真实完整开销 | 指定工程请求的实际计时与启动成本可追溯 | 中位/p95速度优势、正式负载性能 |
| 计划中的ON/OFF/继续监督/布局聚类评价 | 协议已具体、可审查 | 未执行结果、收益或失败 |

工程快照时，正式训练、配对性能、Critic校准和长期收益均为**NOT_STARTED**；当前训练状态见STAGE_STATUS.json和同一run账。旧small256 P2未通过的负证据不改变，不解释为本轮训练结果。
