# R04-M1 生产迁移状态

本阶段完成真实环境中的 C0/C1 单位、来源、保存和恢复接口检查，并完成离线校准。**G3=FAIL，本配置在 step128 停止；不进入 G2/M3。** 工程接口通过与科学准入失败分别记录，不能互相替代。

| 项目 | 本次状态 | 实际证据及边界 |
| --- | --- | --- |
| 真实依赖及来源分类 | GO | Torch 2.12.1+cu126、NumPy 2.2.6、Python 3.10.20；torch.ops/classes 正确识别为虚拟命名空间，普通源码按真实 SHA 核验，used_for_scientific_attempt=true。 |
| 原 14 项 Critic 接口测试 | GO | 真实项目依赖中全部通过，测试 optimizer=0。 |
| 268 维输入与完整标签恢复 | GO | 固定 116 条、7992 个状态全部恢复；第一条 149×268，初始任务数 226；沿用保存的 L/K/H/RTG，无新标签。 |
| C0 raw / C1 normalized 类型 | GO | C0 是原始分量，value 除初始 n；C1 已是分量/n，组合 value 不再除 n。H 为可有符号的全路径信用，不解释为剩余物理距离。 |
| 初始张量与随机流 | GO | 同 seed11622026、同初始净张量；正式初始化前重置随机流。相同净张量的物理输出不同，已保存 step0。 |
| checkpoint 恢复 | GO，限 Critic | step0/128 各 C0/C1，核模型、Adam、RNG、游标、已消费资源和源码/授权/协议/数据身份；预测逐位一致。 |
| Critic 校准 export | GO，限离线 | 含角色、raw/normalized 单位及来源身份；重新加载预测一致。它不是 Actor-only 部署产物，不能替代 Actor 部署加载器。 |
| 恢复后继续拟合 | GO，限 step0 | step0 checkpoint 载入新同角色 Critic+Adam 后实际完成 128 步；step128 仅恢复/导出验收即停止，无 step129。未执行独立连续运行对照。 |
| 离线 Critic G3 | FAIL | C1 value/条件常数为 2.863347186（state-pooled）、3.382871356（父等权），95%CI 上界 4.180973505、4.576223660；均超门槛。 |
| G2 真实 Actor/Adam 事务 | NOT_RUN / NOT_PASSED | Actor 更新为 0，Critic checkpoint 检查不等于 Actor 接受/拒绝事务检查。 |
| M2 / M3 | NOT_AUTHORIZED_NOT_RUN | 没有条件锚点拟合、联合 PPO、追加配置或新种子。 |

依据：[REAL_PREFLIGHT](attempts/02/REAL_PREFLIGHT.json)、[CHECKPOINT_INTERFACE_RESULTS](attempts/02/CHECKPOINT_INTERFACE_RESULTS.json)、[INITIALIZATION](attempts/02/INITIALIZATION.json)、[M1_GATE_DECISION](attempts/02/M1_GATE_DECISION.json)、[INDEPENDENT_NUMERICAL_RECHECK](attempts/02/INDEPENDENT_NUMERICAL_RECHECK.json)。

**生产使用所需的固定语义**

每条轨迹传入固定正整数 initial_task_count=n，不能用当前剩余任务数替代。C0 原始预测转换为 L/n、K/n、H/n；C1 typed 输出直接提供这些归一量，禁止二次除 n。组合 value 始终为 `-(L/n+0.75K/n-H/n)`，单位是每初始任务的负成本。完整 H 标签包括终端 H 对已经经过路径前缀的修正，不能截断为未来后缀收益。

新输出包绑定执行源码、科学协议、数据索引、批次日程、阶段授权、恢复索引与配置语义；原模型、旧 R03、旧校准、R04 提案和失败入口原字节保留。本次源码绑定见 [EXECUTION_SOURCE_COMMIT](attempts/02/EXECUTION_SOURCE_COMMIT.json)。各 checkpoint/export 保存在 [models](attempts/02/models)，不能当成已通过 G3 的生产 Critic 发布。

**A0/A1 仍未完成的生产适配与验收**

以下是尚未通过的具体事项，不构成本轮执行授权：

1. 在真实训练器和部署器中统一选择 A0 旧残差、A1 合法候选均值居中的残差，明确模型格式与版本识别；A1 的非法候选不进入均值、零初始化严格等价 SL、±0.25 边界需真实接口验收。
2. 对真实 Actor 参数和独立 Adam 做候选更新前快照、guard 接受/拒绝、精确回滚；同时核 optimizer、RNG、游标与预算，不用 Critic 恢复代替这项 G2。
3. 真实 rollout 中同步登记 mask、合法排序、首选、去共同偏移残差及更新前后 ratio；保持旧 guard 阈值和失败处理，不因 G3 失败放宽 Actor 限制。
4. Actor-only 导出/加载的版本、单位与来源识别，以及同输入 logits、mask、完整路线 J/L/K 和完整请求时间的一致性仍未在新 A0/A1 上验收。本次无规划请求，不能给出新部署路线结论。

目前下一步是停止本 C1 配置并保留数值证据。任何不同 Critic 或后续 G2/M3 方案均需要独立范围与授权；不追加当前 256 终点以求通过。M1 的校准失败不能作为居中 Actor 假设被否证的证据。
