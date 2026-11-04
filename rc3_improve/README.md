# 学习选择局部修复窗口：覆盖保持规划与多步 PPO

本目录在 `rc3_fast` 的已验证 Fixed1 初解上，学习选择有限预算内值得修复的路线片段。神经网络输出窗口动作；确定性局部求解器联合改变多个停靠；输出层只保留独立检查通过且成本不增的计划。

本轮实现包含真正的多步状态转移、收益排序预训练和 PPO 微调。这里的学习任务不同于此前的 13 个构造配置选择。训练完成与性能优势是两件事：**最终质量、速度、PPO 相对监督模型的增益，以运行目录和独立测试结果为准。**

最终版本的实际结果见 [RESULTS.md](RESULTS.md)，版本口径见 [VERSION_HISTORY.md](VERSION_HISTORY.md)。最终模型约 315 KiB；BC/PPO 三种子各 100/30 轮，合计 4,320 次参数更新、23,040 次真实动作，训练约 422 秒，数据与标签生成另约 76 秒。独立复算确认训练验证成本与最终导出成本一致。

**最终结论：已完成训练，但未证明应替代强基线。** 新百桌测试中，PPO 相对 Fixed1 改善 0.7814%，BC 改善 0.8021%，Fixed3 改善 1.5925%；对应完整规划中位数为 193.36、212.33、146.52 ms（Fixed1 为 48.45 ms）。本轮计时包含初解和候选独立验证，不能与旧版不同检查流程的计时直接混用。PPO 相对 BC、几何规则的百桌差异置信区间包含 0；可靠性保持与强化学习收益需要分开评价。系统实用默认仍建议保留 Fixed1/Fixed3，当前神经模型作为已训练的研究对照。

所有 648 份测试方案及其相对 Fixed1 的成本界限已从磁盘独立复验；26 项单测、4 类结构化失败、缓存与模型重载均通过，见 `verification_v2/verification.json`。

## 方法

初解为 `rc3_fast.solve_route(problem, 1)`，目标仍为 `J=路线长度+0.75×停靠数`。默认动作集合由长度 3、步长 2 的连续停靠窗口组成，包含最后一个可能的窗口。

删除局部片段后，只重新覆盖未被其余路线覆盖的任务。候选池保留原片段候选，并包含能够服务这些任务的其他可达配置；已保留的外部停靠不重复加入。固定入口/出口的覆盖状态动态规划，最多处理 8 项必要任务和 80 个候选。超出限制的窗口被屏蔽。精确性限于这个声明的局部任务与候选范围，不是全局精确解。

策略输入是每个窗口的 24 维可观察统计，包括覆盖缺口、候选数、操作选择数、图上局部路程和绕行、几何分布。模型为 64 维嵌入、两层四头 Transformer 和 actor/critic 头，带候选 padding/动作 mask。它是窗口集合注意力模型，没有实现任务—服务配置的完整异构图网络；图上距离通过窗口特征进入观察。

每步选窗口后实际调用修复器，更新路线，再重新生成窗口和观察；无效修改不改变当前解。默认回合预算 8 次修复调用，奖励为 `100×(J_before-J_after)/J_initial`。奖励之和等于本回合相对初解的百分比改善。重复尝试过的局部签名被屏蔽。

训练包括：

- **BC**：用训练场景中枚举修复的非负收益归一化为软动作标签，预训练 100 轮。没有任何正收益的初始场景不进入 BC 更新；PPO 仍从全部训练场景采样。
- **PPO**：每种子 30 轮，每轮最多采样 32 个训练实例；8 步真实环境交互。GAE 使用 γ=1、λ=0.95；预算耗尽是回合终止，因此末状态 bootstrap 为零。clip 为 0.2，值损失系数 0.5，熵系数 0.01；每批 rollout 做 4 轮更新。BC 和 PPO 使用不同学习率。
- 三个种子分别为 17、29、43。BC 与 PPO 检查点分别按独立验证集选择并保存，因此可以独立测试 PPO 是否优于预训练。验证较差的 PPO 不应替代较优 BC 成为推荐模型。

## 数据范围

本轮数据为 **96 个训练＋24 个验证**场景，全部只含 10×10 桌阵列，均分为 regular/mixed/blocked 三种分布。测试清单在训练前生成并冻结：12/36/64/100 桌各 24 个，共 96 个同分布或向小规模迁移场景；另外 144/196 桌共 12 个外推场景。物理桌子、机器人和可达性尺寸保持原定义。

训练程序只读取缓存中的 train/validation 问题与训练标签，测试地图不进入训练缓存。生成器避开本地已有案例清单的种子，避免复用已分析的留出场景。缓存 pickle 由本项目本地生成；不要用未知来源文件替换它。

## 运行

从包含 `rc3_baseline`、`rc3_fast`、`rc3_learning` 和本目录的工作目录执行。本机训练环境已经安装 PyTorch；新功能部署仍需要 PyTorch 来加载注意力模型。

```powershell
# 生成新的同配置数据目录，已有输出目录不覆盖
python -B -m rc3_improve.data --output rc3_improve/data_repeat --train-per-mode 32 --val-per-mode 8

# 三种子：BC＋PPO，全部只使用百桌训练/验证
.\rc3_learning\.venv\Scripts\python.exe -B -m rc3_improve.train --data rc3_improve/data_repeat --output rc3_improve/run_repeat --bc-epochs 100 --rl-epochs 30 --episodes 32 --horizon 8 --seeds 17 29 43 --device cuda

# 同一冻结模型直接评价全部 108 个新测试场景
.\rc3_learning\.venv\Scripts\python.exe -B -m rc3_improve.evaluate --data rc3_improve/data_repeat --model rc3_improve/run_repeat/ppo_best.pt --bc-model rc3_improve/run_repeat/bc_best.pt --output rc3_improve/eval_repeat --horizon 8 --repeats 3

# 单元检查：修复、动作掩码、参数更新、真实多步状态重放和输出兜底
.\rc3_learning\.venv\Scripts\python.exe -B -m unittest discover -s rc3_improve -t . -p "test_*.py" -v
```

本轮实际运行目录为 `data_v2_20261102/` 和 `run_v2_20261102/`。`training_summary.json` 是训练完成标志；`history.csv` 保存训练和验证过程，`progress.json` 是阶段性进度，不能代替完成记录。`bc_best.pt` 与 `ppo_best.pt` 是分别冻结的可加载检查点。

最终交付采用 v2：训练状态、教师标签和导出方案都先删除零新增覆盖停靠，再计算同一目标成本。v1 的数据、模型和结果只作为开发记录保留，见 `VERSION_HISTORY.md`。实际训练统计与检查点选择以 `run_v2_20261102/training_summary.json` 为准，最终可部署注意力模型为 `deployment_v2/policy.pt`，约 315 KiB。

直接生成单个模型方案及服务 CSV：

```powershell
.\rc3_learning\.venv\Scripts\python.exe -B -m rc3_improve.plan --checkpoint rc3_improve/deployment_v2/policy.pt --scale 10x10 --mode blocked --budget 8 --output rc3_improve/demo/scene_plan_100.json
python -B rc3_baseline/validator.py rc3_improve/demo
```

系统集成使用 `from rc3_improve.plan import RepairPlanner`，创建 `RepairPlanner(checkpoint, horizon=8)` 后调用 `.plan(scene)`。同一对象会缓存最多两张静态地图，任务变动仍重算可达性。先检查返回的 `status`，成功后从 `plans['ppo']` 获取折线路径和服务记录。减少 `--budget` 可减少搜索，但本轮正式质量比较的预算统一为 8。

Python 调用：

```python
from rc3_fast.scenes import make_complex_scene
from rc3_improve.learning import load_policy
from rc3_improve.evaluate import evaluate_case

policies = {
    "bc": load_policy("rc3_improve/run_v2_20261102/bc_best.pt", device="cpu"),
    "ppo": load_policy("rc3_improve/run_v2_20261102/ppo_best.pt", device="cpu"),
}
scene = make_complex_scene(10, 10, 8675309, "blocked")
rows, output = evaluate_case(scene, policies, horizon=8, repeats=1)
plan = output["plans"]["ppo"]
path_xy = plan["route_polyline"]
service_stops = plan["service_stops"]
```

不可行输入会保留失败记录，此时不能假设 `plans` 中一定有完整计划；先检查相应 `rows` 的 `status`。

## 可靠性与公平比较

共同参考解先展开为完整路径并独立验证。学习建议、局部修复或候选验证发生异常时保留已验证 incumbent；接受候选的成本不高于该参考方案。保证以参考解成功生成且通过检查为前提，不等于全局最优或实机执行保证。

主要同预算对照是 random、detour、BC、PPO，均使用相同初解、局部求解器和 8 次修复预算。Fixed1/Fixed3 为实用参考，不应把其运行次数称为与 8 次局部修复等价。完整时延包含初解、窗口特征、模型、修复和独立校验；导航预处理另外报告。所有输入均保留结果，不因失败中断整批或剔除分母。

开源借鉴与源码核查见 [OPEN_SOURCE_NOTES.md](OPEN_SOURCE_NOTES.md)：借鉴 Learning-to-Delegate 的子问题选择、NeuOpt 的多步改进与历史最优奖励、POMO/RL4CO 的策略梯度训练思想。本项目未完整复现这些算法，也不能引用其论文收益证明本方法有效。

## 当前限制

- 双臂仍为肩部偏置球壳/扇形代理，升降固定，不模拟真实 IK、机械臂轨迹碰撞或同步双臂协作。
- 当前观察压缩为窗口统计，没有显式剩余回合预算特征；critic 对有限预算的价值建模存在限制。
- 已尝试窗口按局部签名屏蔽；即使外部覆盖变化，同一签名也不会重试，可能漏掉后续改善机会。这影响搜索质量，不改变 incumbent 的可行性。
- BC 只有初始路线上的监督标签；PPO 才会访问修复后的状态，仍可能遇到分布变化和稀疏改善奖励。
- 未证明神经网络或 PPO 比同预算几何规则更快、更优。必须保留负面结果，并区分局部搜索扩展的收益与学习选择的独立收益。
