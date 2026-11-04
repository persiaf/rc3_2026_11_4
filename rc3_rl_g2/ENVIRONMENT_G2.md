# G2 环境、尝试记忆与共享规则合同

本模块只读引用 G0/G1 的场景、修复器、认证适配层和基础转移。新环境 schema 为 `rc3-rl-g2-v1`，正式菜单 schema 为 `rc3-g2-menu-v2`。不修改旧文件，也不追溯修改 G0 的费用或展示轨迹。

## 可恢复状态

`SearchEnvG2` 保留原观察/step/terminate/finalize 接口。`load_state_dict` 先在分离副本中完成验证，再一次性提交：

- 成本、奖励、计时与菜单特征检查有限数；明确的数值字段拒绝 bool。
- 步数、工作和计数必须是合法非负整数，工作累计与分项一致。
- anchor/current/best 重新认证，核对路线、成本、best 界和接受界；保存的证书与导出方案也核对。
- 真终止、返回证书、回退标志、最终修正、超量和停止原因保持一致。
- 质量奖励、工作罚、势函数与总回报满足相同回合的闭合式。
- 菜单依据同问题、完整路线、配置、预算、历史和尝试记忆重新构造并逐项比较；损坏 mask、features 或动作表不会被直接信任。
- 最后验证 RNG 状态；任一失败不会部分恢复目标环境。

`record_transitions=False` 供训练减少持久日志体积；仍保留预算、历史、记忆及全部合法续采样状态。`fork()` 是教师树对已可信内存状态的分支，adapter 共享，其他状态深复制，proposer/action_builder/certifier 重新绑定子环境。磁盘状态必须走验证后的 load。

外部终止依然产生独立 event，包含稳定 event_id。collector 负责将它恰好一次并入最后一个真实动作；零动作截止只记录指标。环境不会自动构造虚假的 actor 样本。自然 STOP、horizon、预算结束仍在 step 内完成。

## 共享尝试记忆

`proposal_key` 包含 problem_id、完整 current 路线、算子、窗口、禁用节点与实际 task/candidate caps；不包含 budget、best、step 或 history。重放 context 则包含全部动态上下文，两者用途不同。

正式 v2 对所有方法共同过滤三类已知结果：

1. 提案返回相同路线；
2. 提案无效或认证失败；
3. **经过独立认证后**的成本超过当时 `best + epsilon*anchor`。同一路线和固定求解条件下，best 只会下降，之前超界的确定性结果不会重新变得可接受。

第 3 类是在首次教师采集中发现基线仍可能重复拒绝后的训练前修正。首轮采集保留在 `outputs/teacher_debug_v1_bound_rejection/`，不作为正式教师标签；正式采集按同一批种子重新绑定源码后执行。候选首次被拒绝时依然执行独立认证并支付完整查询费用，没有模型专属能力。

过滤发生在 16 个非 STOP 菜单截断之前，用后续未尝试候选补位。所有扫描上下文仍计入菜单工作；没有 proposal 缓存，也没有免费试求全菜单收益。观察公开 `proposal_keys`、`action_attempts` 和记忆大小；被过滤记录仍保存在完整状态中。

## 规则对照

`make_policy` 提供：

- `detour_memory`：在统一菜单中选廉价 detour 最大动作；不重复查询已屏蔽结果。
- `ils`：默认连续 3 次无 best 改善后进行一次 replace 扰动，最多 2 次扰动；达到扰动上限后再次停滞则 STOP。缺少 improve 时也触发扰动/停止。原样接受不重置停滞计数。
- `random`：在实际合法菜单（含 STOP）均匀采样。
- `stop`：共同菜单后的立即停止。

上述参数是公开的首版配置，不冒称已经充分调优的最优规则。原 Fixed3 无需共同首菜单，应由外层评测保留其原始入口。所有规则与 BC/PPO 共用动作、费用、接受和终验语义。

## 认证证据复用

`ResearchAdapterG2` 对固定静态问题缓存完整相同路线的证书；键包含输入 problem_id、验证器源码版本和路线，最多保存 64 条 LRU 证据。只有搜索 accept 阶段可以命中，final 和 load 阶段始终独立复验。存在外部 service_checker 回调时关闭该缓存。

`cache_stats` 单独记录命中、未命中、绕过、真实认证调用和耗时；env.counters 的认证数是逻辑调用数，不能把它当作未缓存实际执行次数。各方法时间比较应采用 fresh adapter 或同等缓存条件；教师树分支共享缓存时必须将物理工作单列。工作奖励仍沿用菜单/提案计数定义，缓存不会修改历史工作罚。

本轮 `outputs/environment_probe.json` 使用旧 G01 case_01 检查缓存开/关在三个控制入口的动作、work、回报、路线和最终状态完全一致。它是单个已知场景的实施与探索性吞吐检查，不是新验证样本或统计加速结论。正式训练吞吐与部署时间以根执行器的新记录为准。

## 回归范围

`python -B -m unittest rc3_rl_g2.test_env_g2 -v`：14 项针对性回归，包括 NaN/Inf/bool、失败原子性、菜单损坏、活动及终态恢复、记忆补位、独立 fork、轻日志恢复、STOP-only、外部 event 总账、认证缓存与 final 故障，以及共享超界拒绝记忆。它们验证环境合同，不替代 PPO 的 log-prob、GAE、梯度及训练结果验收。

## 训练专用磁盘准备缓存

低 RAM 的 worker 只保留少数 adapter 时，`prepared_cache.prepare_case(case, cache_dir)` 可在重置时复用此前已准备的静态 Problem 与 Fixed3 初解。它不修改上面四个生产模块，不用于多方法部署时间评测。

格式为带 SHA256 外层头的压缩 NPZ，内含明确的数值数组与 UTF8 JSON；读取前检查校验和、成员集合和尺寸，NumPy 采用 `allow_pickle=False`。不加载任意 Python pickle。键绑定完整 case、生成/导航/适配/验证器源码和 NumPy/SciPy 版本。载入核对矩阵形状、有限性、稀疏/稠密可达一致性、problem identity 与 anchor 成本；恢复 F、D、arm_dist 的不可写属性，并对 anchor 额外独立认证一次。最终搜索认证仍独立执行。外部约束回调输入不走该缓存。

文件在同目录临时写入并原子替换；短写锁使两个并发同场景请求只产生一次 cold 准备。损坏或版本不匹配的既有条目直接报错，不悄悄作为 miss 重算掩盖问题。它是本地生成的训练缓存与完整性检查，不是外部文件来源的身份认证机制。

返回统计分别记录 cold_prepare_s、cache_write_s、cache_load_s、waiting_s、total_s、原首次准备时间、source_identity 与字节数。首次压缩写入会额外耗时，小场景读取也可能比重新生成慢。一次百桌实施探测中，准备约 0.34 s、首次写约 0.27 s，读取及 anchor 复验约 0.15 s，文件约 6.1 MB；这只说明本地复用的可行性，不是部署性能结论。`outputs/prepared_train/` 可能达到数 GB，必须排除出交付 ZIP，首次准备成本仍计入训练账本。

`validate_worker_snapshot(state,cases,config,seed)` 在独立副本中检查 worker 外围计数、case/order/position/episode、RNG、有限计时，以及 episode_reward 与 env.total_reward 一致；调用者完成该检查及环境 load 后才能提交 live worker 字段。

`test_prepared_cache` 的 6 项回归覆盖 3 类小场景和真实百桌 fresh/cache 同动作、同 work、同回报及方案，损坏校验和、错版本、重算校验和后的 NaN 和 object-pickle 数组拒绝、外部输入拒绝、并发单次发布，以及 worker 元数据损坏拒绝。记录位于 `outputs/prepared_cache_tests/`。
