# 开源实现核验与 G0/G1 采用边界

核验日期：2026-11-03。已使用 GitHub 插件取得官方仓库提交与文件，按下述 commit 固定引用；没有执行下载的源码。用户设计正文已提取到 `design/research_design.txt`。

**本轮是 G0 多步价值诊断与 G1 环境验证，不是 PPO 训练，也没有训练新模型。** 下表说明研究接口的依据，不能用作本项目已获学习收益的证据。实现是否满足接口，以本轮测试与诊断产物为准；v4/v5/v5.1 继续冻结。

| 来源 | 本次固定版本 | 实际核验范围 |
|---|---|---|
| NeuOpt | `ccf6b5f0f6a8fda2792b4be11d4ec35390a8139b` | TSP 状态转移、原作者 n-step PPO、分解动作解码 |
| CleanRL | `fe8d8a03c41a7ef5b523e2e354bd01c363e786bb` | 离散 PPO 的采样存储、GAE、clip、终止处理 |
| RL4CO | `dbd18e7a06ef9c8e35f54ab93adf6136ceeffad0` | 普通 PPO、n_step_PPO、StepwisePPO，以及 DACT/N2S/NeuOpt 的继承关系 |

## 1. NeuOpt：搜索 current，保留 best

原作者 TSP 的 step 先执行 k-opt 并把生成的路线返回为下一 current；更新 best-so-far 成本，并仅对新纪录降本给奖励。该处没有“只有改进 current 才允许前进”的门槛。因此，current 暂时变差与 best 单调不增是不同概念。见 [problem_tsp.py L85–112](https://github.com/yining043/NeuOpt/blob/ccf6b5f0f6a8fda2792b4be11d4ec35390a8139b/problems/problem_tsp.py#L85-L112)。

本项目采用这个状态分离思想，加上不可变、已验证的 anchor。探索允许成本有限变差，但仍须覆盖全部任务、满足共同可达约束及导航可行性；最终只返回验证过的 best 或 anchor。NeuOpt 论文另外研究了不可行区域探索与 GIRE，**本轮不采用该不可行探索机制**，也不把它的 TSP/CVRP 结果转述成桌面覆盖规划性能。[NeuOpt 论文](https://arxiv.org/abs/2310.18264v1)

原作者训练器保存旧状态、上下文、动作和 log-prob，并在 PPO 重评估时强制使用原动作。解码器按掩码生成动作概率，重评估沿用固定动作。见 [训练器 L410–452、L483–504](https://github.com/yining043/NeuOpt/blob/ccf6b5f0f6a8fda2792b4be11d4ec35390a8139b/agent/ppo.py#L410-L504)、[解码器 L408–433](https://github.com/yining043/NeuOpt/blob/ccf6b5f0f6a8fda2792b4be11d4ec35390a8139b/nets/graph_layers.py#L408-L433)。其分解 k-opt 动作不是本项目的窗口动作菜单；本项目由此要求额外保存**采样时有序菜单、mask、动作 ID、完整 current 和预算上下文**。这是一项本地设计约束，不是声称原仓库已经实现了我们的菜单缓存。

## 2. RL4CO：明确选择多步改进的语义

普通 [PPO 类 L17–50](https://github.com/ai4co/rl4co/blob/dbd18e7a06ef9c8e35f54ab93adf6136ceeffad0/rl4co/models/rl/ppo/ppo.py#L17-L50) 明确把整段自回归构造视为单步 MDP，解码节点选择并不分别作为 PPO 的环境转移。因此不能仅换成该类，就宣称已经学习了重组动作的多步后效。

[n_step_PPO L140–218](https://github.com/ai4co/rl4co/blob/dbd18e7a06ef9c8e35f54ab93adf6136ceeffad0/rl4co/models/rl/ppo/n_step_ppo.py#L140-L218) 则采集连续改进步，每步保存 TensorDict、动作、log-prob、reward，用末端 critic 递推 n-step return，再对旧状态和原动作计算概率比。DACT、N2S、NeuOpt 的 model 均继承该类，见 [DACT](https://github.com/ai4co/rl4co/blob/dbd18e7a06ef9c8e35f54ab93adf6136ceeffad0/rl4co/models/zoo/dact/model.py#L5-L14)、[N2S](https://github.com/ai4co/rl4co/blob/dbd18e7a06ef9c8e35f54ab93adf6136ceeffad0/rl4co/models/zoo/n2s/model.py#L5-L14)、[NeuOpt](https://github.com/ai4co/rl4co/blob/dbd18e7a06ef9c8e35f54ab93adf6136ceeffad0/rl4co/models/zoo/neuopt/model.py#L5-L14)。

该版本还存在单独的 [StepwisePPO](https://github.com/ai4co/rl4co/blob/dbd18e7a06ef9c8e35f54ab93adf6136ceeffad0/rl4co/models/rl/ppo/stepwise_ppo.py)，故“RL4CO 的 PPO 全都是整段单步”不准确。本项目未来需要的是连续重组动作上的 PPO/GAE 契约，可参考 n-step 存储与重评估模式，不直接照搬它的固定训练长度和末端 bootstrap。该源码默认 gamma=0.999、n_step=5，是原实现参数，不能替代本任务的有限预算目标定义。

## 3. CleanRL：借鉴可审计训练骨架，单独处理终止

[cleanrl/ppo.py L170–295](https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/cleanrl/ppo.py#L170-L295) 直接展示 observations/actions/log-probs/rewards/values 的保存、GAE、旧动作重评估、概率比裁剪、value loss、entropy、梯度裁剪和 KL 监测。后续可将这些步骤移植到共享动作评分器；当前 G0/G1 只准备可重放记录，不创建 optimizer 或 PPO 更新。

**该固定版本 L205–230 将 terminations 与 truncations 合并后决定是否 bootstrap。** 它不是本项目区分这两类结束的正确示范。根据 [Gymnasium 官方终止说明](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)，任务内有限 horizon 结束属于真正终止，剩余 horizon 应在观察中；外部采样截断仍有后续价值。

本项目语义固定为：

| 结束原因 | 任务终止 | bootstrap |
|---|---:|---:|
| STOP、任务定义的工作预算耗尽、任务 horizon 到达 | 是 | 0 |
| 仅采集器分段截断，任务仍可继续 | 否 | 截断时真实后继状态的 value |

若使用自动 reset 的向量环境，不得误用 reset 后 observation 做上述 bootstrap。GAE 的 bootstrap mask 与“是否跨采样/回合边界继续累加”的 trace mask 也须分别处理；断开的采样段不能串到另一回合。RL4CO n_step_PPO 源码在分段末端取 critic，没有实现本项目的 STOP/工作预算语义，亦不能原样复制。

## 4. 本地三层单调限制与隔离修改点

以下仅为源码审阅，未修改这些冻结文件。

| 冻结位置 | 当前限制 | 新研究入口须改变的部分 |
|---|---|---|
| `rc3_improve/repair.py:80–85,134–199` | 候选池保留被移除点，使旧解可恢复；末尾仅返回严格改进方案，否则回原解 | 保留 improve 行为；另建 proposal 接口，让有界重组能够提出合法但暂时更差的方案。不能只删末尾判断后就认为产生了非单调邻域 |
| `rc3_v3/environment.py:113–134` | 仅在 after < before 时更新路线，否则忽略方案，reward 是即时 current 降本 | 在独立环境保存 current/best/anchor，接受契约控制 current，质量奖励依 best 更新 |
| `rc3_v4/runtime.py:112–116` | 每步禁止 current 成本增加，异常则回退 | 新 runtime 将探索 current 的允许变差与最终输出 best 的成本界分开 |
| `rc3_v4/runtime.py:130–136` | 最终验证并要求结果不劣于 anchor | 保留这种输出保障思想；不允许放宽后的 current 越过最终验证直接导出 |

G1 应测试：同一有序菜单与 mask 可重放；候选不足/DP 失败不污染 current；已用工作计入预算；best 不增；返还方案与 reward 一致；变差再恢复旧 best 不反复领取质量收益。G0 应在预先固定开发状态和共同菜单上比较单步、深度 2/3 的可达 best，统计有限分支诊断的收益与失败，不能把穷举诊断当作已训练策略。

## 5. 奖励、gamma 与势函数的适用条件

本项目的基础奖励为

`r_t = 100*(J(best_t)-J(best_{t+1}))/max(J(anchor),eps) - lambda_work*work_t/work_scale`。

在有限 horizon、gamma=1、anchor 分母及 work_scale 在回合内固定时，逐步相加恰为最终 best 相对初始 best 的改善减去累计工作罚；初始 best=anchor 时即为相对 anchor 改善。这是由该本地奖励的望远镜求和得到的结论，并非原作者已证明本项目有效。gamma<1 会偏好较早得到的改善，通常不再精确等价于单纯最终质量减总工作；有限回合使用 gamma=1 不意味着函数逼近 PPO 有收敛保证。固定工作预算也不等于硬墙钟截止时间。

[Ng、Harada、Russell（1999）Theorem 1、Corollary 2](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf) 给出势函数差分形式与策略不变性条件。本项目如启用 shaping，采用与训练一致的折扣：

`r'_t = r_t + gamma*Phi(s_{t+1}) - Phi(s_t)`。

直接推导得折扣 shaping 总和为 `-Phi(s_0)+gamma^T*Phi(s_T)`。为消除可变 STOP/预算终止时的尾项，所有真正终态统一令 Phi=0；外部 rollout 截断不是终态，不能提前清零。不得对负差分裁剪，否则循环可能产生额外收益并破坏望远镜抵消。若势函数用未验证启发量，仍须它由完整状态确定，且不改变环境转移/动作集合；压缩图特征是否足够表达 Markov 状态另需审查。

## 6. 本轮交付与后续门槛

本次源码核验支持先验证多步机制的设计决策；它不支持“PPO 已超过 Detour”或“多步学习必然有效”。只有 G0/G1 环境与机制结果通过后，才进入同一 BC 初始化的公平分支实验、短 PPO、独立跨规模测试。未来 PPO 还须单独核验 old/new log-prob 比值、padding/mask、真实结束与截断 bootstrap、训练/验证/测试隔离。

原始 PPO 方法的依据是 [Schulman 等（2017）](https://arxiv.org/abs/1707.06347v2)，不是原有 gain 监督模型。所有文件/blob 哈希、行号与读取范围记录在 `design/source_notes.json`。未复制第三方训练实现，未改变旧模型或冻结结果。

