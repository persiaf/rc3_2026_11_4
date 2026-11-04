# G2 实现与公开 PPO／学习局部搜索工作的对应关系

核对日期：2026-11-03。G2 实现为**预算约束下的局部搜索动作选择器**：先由 Fixed3 提供经验证的完整初解，再由策略选择共享菜单中的修复、扰动或 STOP。图网络不直接预测整条底盘折线；服务点、任务覆盖、DP 修复和导航折线仍由共同求解器产生。

本文件区分方法来源、实际本地实现和未使用组件。G0/G1 已核验的官方仓库固定提交继续沿用，详情见 `../rc3_rl_research/design/source_notes.json`；本轮额外阅读 GAE 原论文、PPO 原论文、MPNN 原论文和 Gymnasium 官方时间限制说明。未下载或执行第三方训练程序，未将原项目的模型参数用作本轮初始化；G2 生产代码未因本说明而改变。

## 1. 来源与采用范围

| 官方来源 | 核对的原则 | G2 中对应部分 | 没有采用或不能据此声称的内容 |
|---|---|---|---|
| [PPO 原论文](https://arxiv.org/pdf/1707.06347)，2017，公式 7、9 与算法 1 | 从当前策略采样，旧动作概率比裁剪，多轮小批量更新 | `train_ppo.py::ppo_update` | 不是论文机器人／Atari 实验的复现；裁剪本身不保证本问题收益单调 |
| [GAE 原论文](https://arxiv.org/html/1506.02438v6)，公式 16 | TD 残差的指数加权优势估计 | `ppo_math.py::compute_gae` | 没有采用原论文的 TRPO 优化器或关节力矩控制任务 |
| [CleanRL 固定提交的离散 PPO](https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/cleanrl/ppo.py#L170-L295) | 可审计的 rollout 保存、优势估计、概率重评估、梯度裁剪与 KL 监测 | 本地 `collector.py`、`train_ppo.py` 的训练契约 | 没有调用 CleanRL 的训练器；该固定版本合并 termination/truncation 的处理没有照搬 |
| [NeuOpt 原作者 TSP 转移](https://github.com/yining043/NeuOpt/blob/ccf6b5f0f6a8fda2792b4be11d4ec35390a8139b/problems/problem_tsp.py#L85-L112) 与 [训练器](https://github.com/yining043/NeuOpt/blob/ccf6b5f0f6a8fda2792b4be11d4ec35390a8139b/agent/ppo.py#L400-L542) | 搜索中的 current 与 best 分开保存，连续改进动作保存旧状态／动作／log-prob | `env_g2.py` 的 current/best/anchor；原始观察快照与旧动作重评估 | 没有使用神经 k-opt 分解解码器、GIRE 不可行区域探索或 D2A；本地动作是覆盖修复窗口 |
| [RL4CO 的 n_step_PPO](https://github.com/ai4co/rl4co/blob/dbd18e7a06ef9c8e35f54ab93adf6136ceeffad0/rl4co/models/rl/ppo/n_step_ppo.py#L140-L218) | 路由改进可表达为连续环境转移，保存原动作后重评估 | 持续局部搜索的环境与采集结构 | 没有继承 RL4CO 类或使用 TensorDict／Lightning；本轮使用 GAE，而非原类的固定 n-step return 实现 |
| [MPNN 原论文](https://proceedings.mlr.press/v70/gilmer17a.html)，2017 | 在图边上传递消息并聚合节点表示的一般范式 | `actor_critic.py` 的稀疏邻接聚合 | 是范式层面的对应；不是量子化学网络、数据或预训练参数的迁移 |
| [Gymnasium 官方时间限制说明](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/) | 任务内终止与外部采集截断应有不同 bootstrap 语义 | 真终止 value=0；采集分段仍保留后继 value | 不是把所有名为“超时”的事件统一置为 done；本项目有自己的预算结束和最终验证语义 |

三个仓库引用均固定到 G0/G1 已读取的提交，不把后续主分支变化当作本轮依据。RL4CO 同时存在普通 [PPO](https://github.com/ai4co/rl4co/blob/dbd18e7a06ef9c8e35f54ab93adf6136ceeffad0/rl4co/models/rl/ppo/ppo.py#L17-L50) 与 [StepwisePPO](https://github.com/ai4co/rl4co/blob/dbd18e7a06ef9c8e35f54ab93adf6136ceeffad0/rl4co/models/rl/ppo/stepwise_ppo.py)，因此不把整个框架概括成单步构造。本地采用多步改进任务定义，是针对当前问题的实现选择。

## 2. PPO 与 GAE 实际做了什么

`train_ppo.py` 用采样时保存的 `old_logp` 和同一原始观察、同一动作重新计算概率比 `ratio=exp(logp-old_logp)`；最小化两项负优势代理损失的较大值，与 PPO clipped surrogate 最大化等价。每轮首次更新前要求旧参数重评估的 log-prob 差不超过 `3e-5`。同时使用未裁剪的平方 value loss、熵项、梯度范数裁剪和近似 KL 提前停止；没有 value clipping、学习率退火或 PPO 外的重放训练。公开依据为上述 PPO 论文与固定 CleanRL 源码；这些具体开关与数值属于本轮协议。

正式协议为：8 个 CPU 环境、每次每环境 32 个真实动作；每轮 256 条样本，4 个优化 epoch，minibatch 64，Adam 学习率 `1e-4`，clip `0.2`，GAE `lambda=0.95`，`gamma=1`，entropy 系数 `0.01`，value 系数 `0.5`，梯度裁剪 `0.5`，target KL `0.02`。预登记训练总量为 32,768 个真实动作；本文件不以预登记数量代替完成回执。

`collector.py` 按 T-major 顺序保存 `(T,B)` 转移。真正终止后工作进程可返回新回合观察，但该条已结束转移的 `next_value` 被置零；新回合价值不会泄漏到旧回合。rollout 分段只为一次优化收集数据，当前环境和尝试记忆继续保留。GAE 在真终止处截断，在批次尾端使用仍存活环境的 value bootstrap。这与 GAE 递推及 Gymnasium 的终止／截断区分相符；本地另外检查采样顺序、恢复后的观察及 RNG，以防工程实现破坏上述数学含义。

本轮 `gamma=1` 是有限预算目标的选择：质量奖励按已验证 best 的改善计量，再扣累计工作费用，便于与实际返回方案核对。它不是照用公开路由实现的默认折扣，也不意味着函数逼近 PPO 有收敛保证。势函数系数固定为 0；不能把 GIRE 或势函数塑形作为本轮效果来源。

## 3. 正式网络是局部图消息传递模型

正式模型为隐藏维度 64、两轮消息传递的共享动作 actor–critic，共 65,602 个可训练参数。两层 64 维动作 MLP 作为管线调试对照；正式编码额外接收每个动作对应的**任务—候选服务点覆盖图**，并非将 22 个统计量改名为“图”。

`graph_observation.py` 为每个动作提供 30 维动作特征、9 维任务节点特征和 14 维候选节点特征。包含待修复 required 集合、候选 pool、保留路线的覆盖次数、路线位置、进入／离开端点的导航图距离、current–best 差距、剩余工作／步数及动作尝试信息。覆盖边使用 `C=max_arm(F)` 的单臂独立任务约化；候选导航邻边从真实导航成本中选择，保留距离输入。双臂协作组合、关节轨迹和真实 IK 不由这个图模型处理。

`actor_critic.py` 对覆盖邻居做均值聚合，对候选导航邻居做 `1/(1+D)` 加权消息，再通过本地 MLP 更新；将两类节点池化到对应动作，与动作编码融合。actor 对任意数量的候选动作共享打分，critic 只汇聚合法动作；padding 和非法动作不参与概率或 critic 池化。没有 attention、Transformer、循环 k-opt 解码器、dropout 或 batch normalization，也没有在参数更新后复用旧 learned embedding。缓存仅限绑定问题身份的原始静态坐标与覆盖关系。

这可归入 MPNN 的消息传递与图聚合范式，但上述覆盖边、导航边、局部窗口特征和共享动作头均是本轮本地实现。[Attention Model](https://arxiv.org/abs/1803.08475v3) 使用注意力与 REINFORCE／greedy rollout baseline；它提供了神经路由的背景参考，**不是当前网络或 PPO 训练器的实际实现**。因此不使用“复现 Attention Model／NeuOpt／RL4CO 后得到本结果”的表述。

## 4. BC 初始化和控制组是本地实验设计

G2 教师在相同的环境和有序动作菜单上做深度 2、最多 8 分支的有限前瞻。初始 64 场景、额外 32 场景提供监督数据；`evidence_mask` 区分查询过与未查询的合法动作，未查询动作不参与 BC 归一化或负标签损失。近并列动作使用平均负对数概率；场景、路线和同路线状态分层赋权。教师只训练 actor，没有把有限深度值当作 horizon=16 的 critic 真值。

固定初始 BC 训练 400 次更新后，三条分支使用同一份参数：冻结 BC；继续 BC（记录 512、1024、2048 更新）；新的 on-policy PPO（记录 8192、16384、32768 个真实交互）。继续 BC 与 PPO 均重新创建 optimizer。PPO 不把教师文件当作新采样 rollout，监督阶段的查询成本另记。这是为区分“多做监督训练”和“新的策略交互是否有额外价值”的本地控制设计，未归因给上述开源仓库的现成组件。

本轮正式比较限定单个训练种子，24 个验证场景在多个检查点重复使用；其配对区间仅是该训练种子条件下、该验证场景集合的探索性统计。不能宣称它已证明跨随机种子的稳定收益。最终采用预登记的末次模型，不按验证集挑选最佳检查点。

## 5. STOP、工作预算与时间预算的边界

共享搜索动作菜单包含 STOP。STOP、任务内 horizon=16、工作软预算 2,000,000 耗尽是本项目回合结束条件，结束时执行返回方案核验并处理回退差额。失败动作、菜单构建和实际修复工作进入相同计数契约；遇到内部工作量无法确定的异常时停止训练采集，不将其记录成低成本失败。已尝试的确定性无效／无变化提案按共享记忆处理，规则与网络使用同一环境。

**当前“预算感知”具体指工作预算和剩余决策步数。** 策略观察没有剩余墙钟秒数；0.4 秒比较由外层运行器设置包含预处理与最终验证的完整规划软 deadline。它检验此部署方式的质量—耗时结果，不证明网络已学习按任意墙钟预算条件化决策。软 deadline 允许正在进行的计算或最终验证产生超时，统计必须保留完整耗时及超时量。

部署 deadline 触发时，本地环境显式最终化并把末尾费用／回退修正计入完整返回账本；该路径用于完整规划评估。它与训练中“只切断一个采集批次，环境仍可继续”的情况不同。正式训练仅按工作／步数终止，不能将外层评估结束事件冒充训练中的有限时间任务。Gymnasium 文档支持首先明确任务定义再决定 bootstrap，本项目的具体事件合并和零动作仅计指标规则是额外实现。

## 6. 可以准确写入方法说明的版本

本研究采用新写的轻量稀疏局部图 actor–critic，以有限前瞻行为克隆初始化，再以 PPO-Clip 与 GAE 对共享覆盖修复动作进行新交互训练。其设计借鉴公开 PPO 的采样与概率重评估契约、学习局部搜索的 current/best 分离，以及图消息传递的关系编码方式。覆盖约束、共同尝试记忆、实际工作计费、STOP、可行参照方案和最终验证均通过本地环境实现。

这段表述说明实际系统组成，不提前判断其是否优于 Detour。训练完成情况以 `ppo_receipt.json` 为准，正确性门槛见 `TRAINING_REVIEW.md`，算法质量与耗时以完整最终比较为准。Frozen v4/v5/v5.1、真实机械臂标定和 IK 的原有范围没有改变。
