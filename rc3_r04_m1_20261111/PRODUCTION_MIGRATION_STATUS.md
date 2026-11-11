# R04-M1生产接口迁移状态

本阶段实现的是有来源识别和明确单位的C0/C1离线校准入口。它不迁移旧R03生产文件、不恢复R03的256终点，也不是新的联合PPO入口。下面将“代码已经实现”与“实际工程检查是否运行”分开记录。

## 本阶段已实现的接口

| 接口 | 实现与语义 | 本文形成时的验证状态 |
| --- | --- | --- |
| `MatchedCritic` | 268-128-128-3固定网络；C0为raw L/K/signed H，C1为按固定初始n归一的L/K/H。两配置同seed、同初始净张量；模型参数不包含Actor。 | 静态审查完成，实际14项单元测试待执行。 |
| `ComponentPrediction` | typed输出显式携带variant和初始任务数。C0 value保持`−(rawL+0.75rawK−rawH)/n`，C1 value为`−(normalizedL+0.75normalizedK−normalizedH)`。 | raw／normalized及元数据分别暴露，避免把字符串送入旧tensor日志循环。 |
| `as_training_dict()` | 仅返回旧loss所需raw分量与value。C1先乘回n，再由旧loss除n一次；不能把已经归一的预测冒充raw再次除n。 | 原loss等价的人工fixture已写入测试，尚未运行。 |
| `component_state_loss()`／`matched_loss()` | C0保持原raw误差除n的算术；C1直接使用归一误差。均无额外value项。固定state SUM/trajectory，再episode MEAN，仅fit轨迹可进入拟合。 | 归约、holdout拒绝与typed输入拒绝用例待执行。 |
| 输入与标签 | 有限非空T×268；n为固定正整数初始任务数，拒绝剩余任务数替代。完整L/K/H/RTG形状与组合一致；H允许为负，包含对已走路径前缀的修正。 | 实际输入检查与源hash在运行中执行；不生产新标签。 |
| `make_checkpoint()`／`restore_checkpoint()` | 新schema区分C0/C1单位与角色，核对网络、损失、source/config/data/schedule/protocol/authorization及恢复输入hash；精确恢复模型、Adam、CPU Torch/Python/NumPy RNG、游标与累计资源。Adam每参数step必须匹配游标，矩张量维度、dtype、CPU设备、有限性均检查。 | Step0人工序列化测试待执行；真实拟合端点由本次runner保存后读回，无额外optimizer尝试。 |
| 初始n来源 | checkpoint明确绑定`RESTORED_INPUT_INDEX.json`中的`task_count`及原`M1_DATA_AND_SOURCE_IDENTITY.json`来源记录；调用者不能因张量shape相同而忽略恢复输入身份。 | 精确恢复输入索引由实际已保存前缀重建后生成并纳入checkpoint身份。 |
| `make_export()`／`load_export()` | 单独的Critic离线校准导出schema，保存语义、来源及端点，剔除Adam/RNG；拒绝未知字段、旧schema、Actor用途及C0/C1互换。加载后eval且不启用梯度。 | 人工与实际端点预测往返待执行；这不是Actor-only部署包。 |
| 计账及停止 | 每次实际Critic尝试先扣账，保存失败和累计墙钟，两个配置同批次日程、0/128/256端点及既定128停止。 | 实际资源与停止判定见运行收据，不能以接口存在预填成功。 |

恢复校验必须由调用者将包内游标、资源与**当前持久账本**对照；包内自带hash不能取代外部来源验证。实际端点恢复可以证明保存的权重、Adam、RNG及预测保持一致，并按原日程继续；没有额外完整连续运行对照时，不能声称已经独立验证“两次完整训练完全等价”。失败后也不能换目录重置预算。

## A0/A1与G2仍未完成的生产项

1. **Actor语义传递。** 旧`runtime.export_actor`仅检查206维、hidden和幅度，不能识别absolute A0与pre-tanh居中A1。相同shape的权重不表示相同策略。需要将correction semantic ID贯穿Actor父adapter、导出器、文件loader与checkpoint，并拒绝错误角色。
2. **父adapter导出方法兼容。** 旧`LookaheadResidualActor.actor_only_package()`调用旧correction对象的同名方法；R04的独立`RelativeCorrection`原型没有这条旧方法路径。仅standalone mapping往返不能证明它可直接接入旧生产adapter。
3. **真实Actor／Adam候选事务。** 仍需另行登记并验收每次候选更新前的参数及Adam快照、guard接受、guard拒绝后严格恢复、拒绝参数不会被后续使用。M1的Critic checkpoint检查不包含Actor、guard或动作变化，不能替代这项G2。
4. **完整训练状态恢复。** 未来联合runner必须同时绑定Actor A0/A1、Critic C0/C1、固定SL、H、mask、特征、奖励、端点、两优化器、RNG、查询游标和累计资源。必须验证真实暂停／恢复与约定事务，不接受R03残差或旧raw Critic的同shape静默加载。
5. **PPO生产typed接口。** M1已实现单次归一化loss及明确日志单位，但旧PPO的`_admit`、`_critic_step`、rollout及tensor日志循环仍未统一迁移为新typed schema。本轮未调用旧202维入口，也未启动206维Actor训练。
6. **Actor部署完整请求。** 新A0/A1生产loader仍需加载准确SL256、排除Critic和Adam，并在实际已授权任务中验证logits／mask／完整路线与J/L/K一致；需要冷启动、完整请求及I/O计时。M1的Critic导出不满足这些部署验收。

基座永久冻结、旧mask和代理约束的历史检查继续保留，但本轮没有生成新路线，没有真实双臂IK或实机验证。旧R03、旧离线校准、R04原提案、旧权重和用户Word文件均应保持原字节。

## 本阶段结束后的唯一准入路径

以`M1_GATE_DECISION.json`逐项结果为准：G3为GO后，下一步是完成上述G2最终工程验收；这不授予M3执行许可。G3为FAIL或INSUFFICIENT时，本C1配置停止，不自动转M2、追加种子、延长步数或启动Actor训练。MSE通过不等于路线收益；MSE失败不否证居中Actor的机制假设。

## 实际执行记录

本文在首次优化器前写成，此时实际测试与拟合状态为**待执行**。完成后应引用`UNIT_TEST_RESULTS.json`、`CHECKPOINT_INTERFACE_RESULTS.json`、`RESOURCE_ACCOUNTING.json`及最终报告追加真实结果，不覆盖上述接口边界与G2缺项。

### 2026-11-11追加：本次执行已停止

`RESOURCE_ACCOUNTING.json`记录累计**2859.2566758999983秒**及`FAILED_PRESERVED`。`FAILURE.json`显示runner在首次工程测试之前的来源hash检查处检测到1800秒预算已超限。实际消费保留，不能写成“在1800秒内完成”，也不能归零后重跑。

| 迁移项目 | 本次实际状态 |
| --- | --- |
| C0/C1 typed单位、loss、checkpoint、导出源码 | 已实现并完成静态审查；不是运行通过。 |
| 14项零优化器单元测试 | **NOT_RUN**。 |
| 准确SL恢复、268状态重建及保存输入检查 | **NOT_RUN**；实际恢复完整轨迹0。 |
| C0/C1优化器与step0/128/256评价 | **NOT_RUN**；Critic优化器尝试0。 |
| 真实checkpoint／Adam／RNG往返 | **NOT_RUN**；未生成实际拟合端点。 |
| 实际Critic导出预测等价 | **NOT_RUN**；不能用源码存在代替通过证据。 |
| G3 | **INSUFFICIENT**；没有本次校准数值可以判定C1科学优劣。 |
| G2／M3 | **未进入，未通过，未启动**；上述六项Actor生产缺口仍然存在。 |

本配置因资源耗尽停止；不能继续消耗本次已无剩余额度的工程或科学预算。此结果不支持路线收益，不构成C1误差退化的证据，也不否证Actor居中假设。原静态审查表中的“待执行”是执行前历史状态，本追加是当前实际状态。
