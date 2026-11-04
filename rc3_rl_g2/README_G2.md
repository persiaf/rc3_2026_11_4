# G2：可达性约束下的预算感知图策略与 PPO 小试

**训练已完成；固定工作预算评测中，最终检查点在 24／24 个验证场景根节点 STOP，返回成本等于 Fixed3 初解。** 本轮交付的是可复验训练平台和研究权重，当前证据不支持用 PPO 替换默认规则规划。完整对照、执行口径和独立审阅见 [REPORT_G2.md](REPORT_G2.md) 与 [G2_AUDIT.md](G2_AUDIT.md)。

本目录在冻结 G0/G1 环境上新增图 actor-critic、有限前瞻行为克隆、共享规则基线、并行 PPO、检查点恢复和完整路径导出。原 v4/v5/v5.1 与 `rc3_rl_research/` 保持只读。研究对象仍是**二维静态任务、给定候选服务配置和代理可达关系**；当前不进行真实标定、关节运动或 IK 验证。

本轮是预登记的**单训练种子、百桌场景、32,768 个真实 PPO 动作**小试。`outputs/protocol.json` 规定运行规模，`outputs/training/ppo_receipt.json` 的 `completed`、`actual_actions` 和逐轮账本证明实际完成情况；协议文件本身不是完成证据。本文档整理接口与复现范围，具体训练是否完成、质量比较和后续判断以最终运行回执及结果报告为准。只有质量与时间比较支持时，才能讨论学习策略的使用价值；本目录不会自动取代原 Fixed3／Detour 默认入口。

**实际执行发生一次进程中断：** 在 18,432 步一致检查点恢复。因此本轮最终计数按“已提交、用于更新的真实 on-policy 动作”解释；另可能有 0～256 个被丢弃的未落盘动作和 0～16 次未提交优化尝试，其精确工作量及时间未知。它们不影响恢复链上的最终权重，但不能把全部历史物理调用写成恰好 32,768。训练器记录时间不含中断间隔或未知尾段，见 `outputs/interruption_receipt.json` 和最终独立审计。

## 安装位置与入口

将 `rc3_rl_g2/` 与冻结的 `rc3_rl_research/`、`rc3_baseline/`、`rc3_fast/`、`rc3_improve/`、`rc3_v3/`、`rc3_v5/` 等模块放在同一解压根目录。G2 是增量研究包，不单独替代原 v5。命令均在该根目录执行。

训练和模型推理需要 Python、NumPy、SciPy、PyTorch、threadpoolctl。实际训练使用本机 `rc3_learning/.venv/Scripts/python.exe` 的 Python 3.10 环境；虚拟环境不随交付包复制。下文 `python` 指向装有上述依赖的解释器。冻结审计仅需 Python 标准库。

用保存的检查点直接生成完整方案：

```powershell
python -B -m rc3_rl_g2.runtime_g2 --checkpoint rc3_rl_g2/outputs/deployment/policy_ppo.pt --nx 10 --ny 10 --seed 20261103 --mode mixed --output rc3_rl_g2/outputs/my_plan.json
```

上例要求正式最终检查点已实际产生。已有 `bc_initial.pt` 或 `bc_continued_2048.pt` 也可用同一入口推理。若只运行共享规则：

```powershell
python -B -m rc3_rl_g2.runtime_g2 --method detour_memory --nx 10 --ny 10 --seed 20261103 --mode blocked --time-limit-ms 400 --output rc3_rl_g2/outputs/my_detour.json
```

`--method` 支持 `learned`、`detour_memory`、`ils`、`random`、`stop`。模型推理采用贪心动作选择。`--scene 文件.json` 可替代随机生成场景，支持原始场景或含 `scene` 字段的封装。命令行主动拒绝外部约束包；这类输入必须显式使用 `ResearchAdapterG2.from_external(...)`，保留完整的 F 与 A，不能在读场景时丢掉约束。

输出 JSON 包含原场景、服务访问顺序、左右臂分配、完整底盘折线、证书及诊断；同名 `.trace.json.gz` 保存动作和费用证据。`complete_s` 与 `runtime_and_model_load_s` 分开报告，后者包括本进程导入运行库与载入模型的开销。输出通过的是给定模型下的检查，并不代表已验证真实机械臂执行。

## 环境与模型实际决定什么

搜索以经验证的 Fixed3 路线为 anchor，分别维护 current 与 best。动作从共享有限菜单中选择局部 improve、允许有限变差的 replace 或 STOP。候选必须通过认证，且满足 `J(candidate) ≤ J(best) + 0.02 × J(anchor)` 才可成为 current；best 只在改进时更新。终止时独立复验 best，必要时返回已认证 anchor，并修正最终回报。

代价 `J = L + 0.75 K` 是路径长度与停靠次数的等效路程，不是机器人真实完成时间。默认势函数权重为零，完整回合满足：

```text
return = 100 × (J_anchor − J_returned) / max(J_anchor, 1e−8)
         − 0.01 × work_used / 10000
```

work 是菜单扫描、候选及 DP 工作的可核对代理计数，不是秒或 FLOPs。默认 H=16、work_budget=2,000,000；一次已经启动的原子菜单构建或修复可以越界，随后终止并报告超量。`--time-limit-ms 400` 同样是**软期限**，初解、原子修复和强制终验可能使完整场景耗时超过 400 ms。

图策略读取局部任务—服务候选二部图、覆盖边、窗口入口/出口的真实导航距离、保留覆盖、路线顺序、剩余预算、current/best 差距与尝试记录。正式模型为 hidden=64、2 轮稀疏消息传递的图 actor-critic，无 dropout 和 BatchNorm。MLP 仅用于数值及短管线对照，本轮正式 PPO 使用图模型。

所有 G2 规则与学习策略共用同一个有序菜单、候选上限、独立认证、接受界和费用。菜单在截断到最多 16 个非 STOP 动作前，过滤相同 current 路线上已知无变化、失败及已认证超接受界的确定性提案，并用后续候选补位；首个查询照常付费。proposal key 不随剩余预算或历史改变，没有免费全菜单求解，也没有模型专用的提案缓存。

`detour_memory` 使用廉价 detour 排序；ILS 有明确的 improve、replace 扰动、停滞计数和 STOP 条件。ILS 参数仅在 12 个规则开发场景预选，冻结为连续 2 次无 best 改善触发扰动、最多 1 次扰动；不使用验证场景调参。原 Fixed3／Detour 另保留原入口，Fixed3 不承担 G2 首菜单费用。详见 `ENVIRONMENT_G2.md` 与 `outputs/rule_development/selected_ils.json`。

## 数据、同初始化三分支与实际计算账本

所有正式新场景均为 10×10 桌阵列，模式包括 regular、mixed、blocked。预登记种子分为初始教师 64、补充教师 32、规则开发 12、PPO 训练池 384、反复观察的验证集 24，以及短管线 6 个场景。各 split 互斥，验证种子排除已知旧场景；本轮没有跨规模或多训练种子证据。

教师重新在 G2 环境中执行 depth=2、每层最多 8 个分支（包含 STOP）的有限前瞻，初始 64 场景产生 512 个状态、4,096 次真实查询，新增 32 场景产生 256 个状态、2,048 次查询。未知合法动作独立标记 `evidence_mask=False`，不会因数值占位为零被当成负例。标签只监督 actor，不用局部深度二回报训练 horizon=16 的 critic。两批原始账本的离线复核为 96／96；没有把离线复算写成重新执行全部查询。详见 `TEACHER_NOTES.md`。

| 分支 | 共同起点与新增计算 | 用途 |
|---|---|---|
| 冻结 BC | 64 场景、400 次 BC 更新后的同一 `bc_initial.pt` | 判断初始教师监督本身的效果 |
| 继续 BC | 从相同初始参数再做 2,048 次 BC 更新；合并 96 场景并重算等场景权重 | 区分更多监督学习与在线强化学习 |
| PPO | 从相同初始参数使用新环境交互，128 轮 × 8 环境 × 32 动作 = 32,768 动作 | 围绕实际成本与工作罚改进策略 |

三分支使用同一初始权重，不表示三者计算成本相等。PPO 不将教师样本当作 on-policy 数据；STOP 是真实动作，reset、离线教师查询与短管线动作均不加入正式 PPO 动作数。PPO 每轮最多 4 epochs，minibatch=64，gamma=1、GAE lambda=.95、lr=1e-4、clip=.2、梯度裁剪 .5，target KL=.02 可提前终止本轮更新，实际 optimizer 更新数以账本为准。

预登记在 8,192／16,384／32,768 动作验证；最终模型固定取 32,768 动作，不按验证排名挑较早检查点。本轮不会自动扩展到 65,536 动作或多个种子。验证集被反复观察，属于开发比较，不能作为最终独立确认集。

正式运行预先选择 CPU，8 个 worker，每个 worker 只保留 1 个 adapter，Torch 与 worker 数学库各限制 1 线程。原因是本机约 16 GiB 内存、当时约 2 GiB 可用，额外 CUDA 上下文会加大内存压力。硬件探测只用于实施选择，GPU 单独更新较快不等于端到端规划较快。详情见 `outputs/training/execution_settings.json`。

## 训练、恢复与验收命令

以下命令用于已有冻结协议和教师数据的新运行或明确恢复，不应在正式进程仍运行时重复启动，也不应通过覆盖原记录伪装重新复现：

复训时使用独立副本，保留提供的 `protocol.json` 和教师数据，并先把该副本已有的 `outputs/training/` 移到单独归档位置，再创建新的训练输出目录；不要覆盖交付证据。恢复命令中的检查点动作数必须与日志末行一致。完整交付后 `ppo_latest.pt` 可能早于最终日志，不能把它当作自动回滚入口；最终研究权重固定为 `ppo_32768.pt`。

```powershell
python -B -m rc3_rl_g2.train_ppo --stage bc --device cpu
python -B -m rc3_rl_g2.train_ppo --stage bc-continued --device cpu
python -B -m rc3_rl_g2.train_ppo --stage ppo --device cpu
python -B -m rc3_rl_g2.train_ppo --stage ppo --device cpu --resume rc3_rl_g2/outputs/training/ppo_latest.pt
```

训练检查点绑定模型结构、协议、父 BC SHA、当前生产代码与冻结求解依赖，并保存优化器、RNG、动作/迭代/更新计数及 8 个 worker 的活动状态。环境和 worker 的恢复先在独立副本验证，再提交；NaN/Inf、错误数值类型、路线/奖励/菜单不一致或错误 RNG 会被拒绝。已有日志领先于检查点时会停止，避免静默倒退并重复计算动作。生产代码或绑定版本变化后不能声称无缝延续同一正式实验。

rollout 的长度边界只截断采集，不结束正在进行的搜索，GAE 保留 bootstrap。外部截止事件通过稳定 event_id 恰好一次并入最后一个真实动作，终止 bootstrap 归零；零动作截止只进入指标，不制造 actor 样本。最后一轮可能仍有活动回合，保存在检查点中，应与已完成回合分开报告。worker 的回合墙钟可能包含模型优化和验证等待，不能拿来代替部署规划耗时。

接口使用：`SearchEnvG2(adapter, config=SearchConfig(), seed=0, record_transitions=True)`；支持 `observe/step/state_dict/load_state_dict/terminate/finalize`。轻日志训练模式 `record_transitions=False` 仍保存合法续采样所需状态。详细公共合同见 `INTERFACES.md`，恢复及数值审阅见 `TRAINING_REVIEW.md`。

```powershell
python -B -m unittest rc3_rl_g2.test_env_g2 rc3_rl_g2.test_prepared_cache rc3_rl_g2.test_collector_contract rc3_rl_g2.test_model rc3_rl_g2.test_training_guards -v
python -B -m rc3_rl_g2.tools.freeze_audit --phase final
```

第一条运行合同与合成回归，不重新执行整套训练。第二条只读核对原 v4/v5/v5.1 与 G01 归档 SHA、G01 的 279 个清单 payload、G2 正式 18 个生产源、教师/规则绑定与协议；输出 `outputs/freeze_after.json`。源码一致不等于模型收益通过，完整结果另由逐场景方案、返回/工作账本及配对统计支持。

## 缓存的用途与比较边界

搜索认证缓存只复用同一静态问题、验证器版本和完全相同路线的证书，所有共享策略使用相同规则；final 与 load 始终独立复验，有外部 service_checker 时关闭该缓存。逻辑认证次数、真实执行次数与命中数分别记录。

`prepared_cache.prepare_case(case, cache_dir)` 是**训练专用**准备缓存，减少低 RAM 下重复 reset 的场景准备工作。压缩 NPZ 内只含数值数组和 JSON，读取校验外层 SHA、来源版本、输入身份、有限性、矩阵关系与 anchor 成本，禁止 object pickle；载入重新认证 anchor。F、D、arm_dist 恢复为不可写。损坏或错版本条目拒绝，文件以短锁与原子替换发布；外部约束或 service_checker 输入不走此缓存。

`cold_prepare_s`、首次写入、cache_load_s、waiting_s、字节与 source_identity 分开记录，首次准备不可隐去。三种小场景和百桌的 cold/cache 同动作、work、回报及完整方案证据保存在 `outputs/prepared_cache_tests/equivalence_receipt.json`。它证明实现一致性，不能据此宣称部署提速；多策略正式完整时间评测使用 fresh adapter，不使用 `prepared_train`。

## 产物与交付范围

| 路径 | 用途 |
|---|---|
| `outputs/protocol.json`、`protocol.sha256` | 场景、预算与训练设置的冻结协议 |
| `outputs/formal_source_manifest.json` | 18 个正式生产文件绑定 |
| `outputs/teacher_initial/`、`outputs/teacher_extra/` | 正式教师标签、全部查询、spec 与复核回执 |
| `outputs/rule_development/` | 规则参数选择证据与冻结参数 |
| `outputs/training/` | 三分支模型、逐轮/逐回合账本、正式 PPO 完成回执 |
| `outputs/pipeline/` | 256 个 MLP＋768 个图模型动作的第二轮数值接线记录 |
| `outputs/freeze_before.json`、`freeze_after.json` | 新增工作前后冻结完整性核对 |
| `outputs/prepared_cache_tests/` | cold/cache 一致性与损坏输入拒绝证据 |

两个历史类别必须与正式结果分开：`outputs/pipeline_before_resume_guards/` 是较早的 MLP＋图模型调试管线，与第二轮合计 2,048 个真实调试交互，均排除正式 32,768 动作；这些历史检查点并非最终源码绑定，也不是正式初始化。`outputs/teacher_debug_v1_bound_rejection/` 是首次中断的旧尝试记忆教师采集，39 个完整场景的 2,496 查询是可核对小计，两个中断尾部的未落盘工作未知；正式教师按同一预登记种子重新采集，没有将旧标签混入训练。

交付 ZIP **排除** `outputs/prepared_train/` 的可重建数 GB 缓存、`.venv/`、`__pycache__/` 和临时锁/写入文件。保留 `prepared_cache.py`、可重建协议、一次 cold/cache 回执及首次准备账本。历史调试记录若为体积原因仅保留回执与摘要，须在交付清单明确原记录位置及省略范围，不能把已发生的调试计算写成零。

教师 `teacher.pkl` 与 PyTorch 检查点仅供读取本项目生成的可信本地文件；准备缓存使用非 pickle 格式。当前交付不支持在线任务持续到达、真实 IK 或跨机器人参数泛化，结论以这次单种子受控比较的实际证据为限。

## 构建可迁移交付包

`tools/package_delivery.py` 只在显式执行时打包，不会被训练入口调用。完成正式训练、评估及最终审计后运行：

```powershell
python -B -m rc3_rl_g2.tools.freeze_audit --phase final
python -B -m rc3_rl_g2.tools.package_delivery build --output rc3_rl_g2_delivery.zip
```

构建要求完成回执为 32,768 动作、18 个正式源匹配、冻结审计为 final 且通过。ZIP 收录 G2 源码与产物，按上述规则排除大缓存、虚拟环境、临时文件、打包自身与迁移测试树；静态 import 闭包另收录 `rc3_rl_research`、`rc3_fast`、`rc3_baseline`、`rc3_improve`、`rc3_v3`、`rc3_v4`、`rc3_v5` 的原清单 Python 文件。每个旧文件均核对冻结 SHA 后按原字节复制，没有附带旧实验二进制。`rc3_v4` 是 G2 评估模块导入原规则运行器的实际依赖，不应遗漏。

包内 `DELIVERY_MANIFEST.json` 列出全部 payload 的 SHA256/字节数，清单自身不循环求哈希；`FROZEN_DEPENDENCIES.json` 保存旧依赖身份与闭包，`delivery_metadata/` 保存两份原清单。ZIP 的 SHA 与排除说明写入工作区 `outputs/delivery_package_receipt.json`。构建会自动解压到独占的新目录 `rc3_rl_g2/outputs/portable_test/<包名>-<SHA前12位>/` 并核对 payload，拒绝覆盖既有目录或原源码。

迁移 smoke 为独立显式步骤，下面的目录替换为构建回执中的 `portable_directory`：

```powershell
python -B -m rc3_rl_g2.tools.package_delivery smoke --directory rc3_rl_g2/outputs/portable_test/<实际新目录> --python rc3_learning/.venv/Scripts/python.exe
```

smoke 在新目录启动隔离子进程，检查导入来源均在该目录、训练命令帮助、环境/终止事件合同，并各生成一份小场景 STOP 与最终 PPO 方案，随后重新构建场景进行独立方案验证。它不会重新训练或测量正式性能。日志和回执仅写入迁移目录下新建的 `portable_smoke_*`，不改交付 payload。

任何位置解压后的只读完整性复验均可从解压根目录运行：

```powershell
python -B -m rc3_rl_g2.tools.package_delivery verify --directory .
```

这个 verify 核对本次交付子集。原 `freeze_audit` 的四份旧归档与 G01 全部 279 payload 检查仍需原工作区，本可迁移包只保留其最终回执和来源清单，不声称携带全部旧实验。依赖安装与模型推理命令见包根目录 `PORTABLE_README.md`。
