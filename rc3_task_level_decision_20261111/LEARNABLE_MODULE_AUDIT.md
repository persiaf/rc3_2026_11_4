# 当前可学习模块：代码与保存证据核对

范围为最近实际运行的Frozen-reference PG/A1实现。更早Integrated-PPO使用202维、±2残差和独立Critic；R03使用206维ON/OFF残差及Critic，不能把这些配置与当前A1混成同一个模型。这里只读代码和既有结果，未加载权重、重做前向或生成路线。

| 环节 | 已实现、可改变什么 | 固定或受限的部分 | 代码证据 |
|---|---|---|---|
| 主评分/表示 | 冻结SL随观测计算候选打分和状态特征；输入会随已选动作变化 | 当前训练不更新SL编码/主评分参数，不能学习新的基座表示 | [load_actor与冻结检查](../rc3_conservative_refinement/runtime.py#L112)、[_assert_frozen](../rc3_conservative_refinement/runtime.py#L142) |
| A1修正 | 206→64→64→1网络改变每个合法候选的相对logit及采样概率；因此可改变选点、顺序，以及随动作演化的后续覆盖/停靠数 | `delta=.25*tanh(f-legal_mean(f))`，末层零初始化；没有连续位姿、任务窗口或stateID决策分支 | [生产A1](../rc3_frozen_reference_pg_20261111/actor.py#L11)、[原公式](../rc3_r04_review_20261111/relative_actor.py#L43) |
| 幅度 | 基座间隔较小时可能发生首选翻转 | 任意两合法候选的修正差不超过0.5；基座领先>0.5者不能被该残差反超。gap≤0.5仅是必要条件，不保证网络/联合保护可行 | [有界delta与logits](../rc3_r04_review_20261111/relative_actor.py#L90) |
| 动作集合/mask | Actor在合法候选中作选择；其动作会改变剩余任务和下一步mask | 请求已给定任务/窗口/入出口，候选位姿及全局导航D固定；mask排除无剩余覆盖、不可达候选，无可学习STOP；覆盖完成自动结束 | [LocalProblem](../rc3_global_local/problem.py#L76)、[mask](../rc3_global_local/environment.py#L19)、[step](../rc3_global_local/environment.py#L30) |
| 解码 | 在线训练从当前Actor概率采样；冻结评价取argmax，当前同一状态打分有直接作用 | 不能把“有梯度/参数变化/概率变化”当成贪心完整路线变化，更不能当收益 | [实际采样与贪心分支](../rc3_conservative_refinement/runtime.py#L219) |
| 训练参考 | PG-global/PG-cond改变detach的完整回报优势，可能影响更新方向 | B表固定，只用尺寸/可观测进度；不学习表，不参与部署选点，也不替代Actor；当前无神经Critic | [reference.values](../rc3_frozen_reference_pg_20261111/reference.py#L11)、[实际Trainer](../rc3_frozen_reference_pg_20261111/trainer.py#L16) |
| 参数更新目标 | 每机会仅更新`actor.correction.parameters()`，当前策略独立采样；监督对照用已有标签CE | PG保留base-KL系数.02、ratio clip .1、批次优势RMS缩放和单候选更新；这些训练目标约束与下面的硬guard不同，不能单独认作失败原因 | [实际损失与Adam对象](../rc3_frozen_reference_pg_20261111/trainer.py#L16) |
| guard | 允许满足所检查状态分布边界的更新，拒绝则恢复Actor/Adam | 增量KL≤.001、概率最大变化≤.02、相对SL的TV≤.1；约束已给状态，不保证未见状态、相同路线或成本单调改进。guard不是部署时选点器 | [GuardLimits](../rc3_conservative_refinement/guard.py#L20)、[check_update](../rc3_conservative_refinement/guard.py#L66) |
| 完整回报 | 每次动作承担行走/停靠成本，终末并入固定H信用，累计对应完整任务目标 | gamma=lambda=1与完整回报已经实现，不能把“改用长期回报”冒充下一版尚未实现的新机制 | [环境终末信用](../rc3_global_local/environment.py#L30)、[PG完整回报优势](../rc3_frozen_reference_pg_20261111/trainer.py#L38) |
| 确定性H | 对Actor raw方案执行规范化、2-opt、冗余停靠删除和候选替换；H后方案依赖输入raw方案 | H源码不学习。它不是固定输出同一条路线，但可能压缩/覆盖Actor的raw差异，不能把其绝对降本归给Actor | [finish](../rc3_global_local/terminal.py#L36)、[joint_search](../rc3_fast/solver.py#L96) |
| 具体服务与轨迹 | Actor选点/顺序影响任务在哪个停靠点被服务及导航腿顺序 | 同点服务任务由覆盖关系自动确定，导出固定优先左臂否则右臂；不学习臂选择、IK、关节轨迹或导航边路线 | [export](../rc3_global_local/terminal.py#L11) |
| 独立验证 | 校验完整覆盖、同臂F&A、入出口、导航长度/碰撞代理、目标算术 | 不替Actor学习；`arm_joint_trajectories=None`，明确无真实关节轨迹证书 | [validation](../rc3_global_local/validation.py#L17) |

**保存证据与实现相互约束：** 最近在线PG三组各128次候选全部自然接受，不能说其零收益由guard拒绝造成；PG-cond有效残差远小于通常基座间隔只是描述性诊断，不是唯一因果证明。后续固定32状态的QP确有80次拒绝，其中67仅hinge、1仅KL、12两者都有，不能沿用“从不拒绝”，也不能说全由guard造成。见[在线完整报告](../rc3_frozen_reference_pg_20261111/EXPERIMENT_REPORT.md)与[QP原报告](../rc3_shared_constraint_opt_20261111/SHARED_OPTIMIZATION_REPORT.md)。

共享A1在原32状态存在真实32/32参数，是局部表达存在性证据；CE/Adam/QP固定终点失败不构成表示不可能、DRL普遍无效或路线收益结论。标签相关共享构造及零输出隐藏层赋值路径都不是可用的正式训练初始化/搜索方向。

供老师决定的“SL初始化、训练期主策略可更新、部署期冻结”见[短材料](TEACHER_DECISION_BRIEF.md)，是建议，不是本轮代码改动或已执行效果。
