# S2 模型、日志与证据索引

实验 `rc3_v3_global_local_20261106_01`，固定三个 seed；每个 RL512 与 continued-SL512 都是预定终点。全部二进制权重、Adam、RNG 和大型 RL journal 使用 Git LFS。源码、sidecar、commit receipt、日志和计划保持各自原字节。

| Seed | untrained | frozen SL256 来源 | RL512 | continued-SL512 |
| --- | --- | --- | --- | --- |
| 11622026 | [untrained](../rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01/seed_11622026/S1/untrained.pt) | [SL256](../rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01/seed_11622026/S1/update_0256.pt) | [RL512](../rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01/seed_11622026/RL/update_0512.pt) | [continued-SL512](../rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01/seed_11622026/continued_SL/update_0512.pt) |
| 11622027 | [untrained](../rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01/seed_11622027/S1/untrained.pt) | [SL256](../rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01/seed_11622027/S1/update_0256.pt) | [RL512](../rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01/seed_11622027/RL/update_0512.pt) | [continued-SL512](../rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01/seed_11622027/continued_SL/update_0512.pt) |
| 11622028 | [untrained](../rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01/seed_11622028/S1/untrained.pt) | [SL256](../rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01/seed_11622028/S1/update_0256.pt) | [RL512](../rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01/seed_11622028/RL/update_0512.pt) | [continued-SL512](../rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01/seed_11622028/continued_SL/update_0512.pt) |

frozen-SL 是各 seed 的原 S1 SL256 精确张量快照，不重新训练或创建新的科学起点。RL/continued-SL 的 `start_from_SL.pt`、每32次更新的 checkpoint、`.pt.json` 与 `.pt.commit.json` 在各自目录；`S2/FORK_RECEIPT.json` 证明三个分支起点相同且 Adam 动量未继承。完整文件与张量身份见 [WEIGHT_LINEAGE.json](WEIGHT_LINEAGE.json)。

- [原始 run：计划、输入、日志与所有检查点](../rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01/)
- [三个 S2 COMPLETE 收据](../rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01/stage_admission/S2/)
- [真实 S2 阶段授权](../rc3_global_local_execution_v3/authorizations/20261107_S2/S2_AUTHORIZATION.json)
- [实际顺序执行记录](production_invocations/ALL3_COMPLETE.json)
- [逐更新数据](analysis/updates.csv)、[逐轨迹数据](analysis/trajectories.csv)、[任务分组统计](analysis/group_summary.csv)
- [终局调用与失败、缓存](analysis/terminal_attempts.csv)、[共享资源账](RESOURCE_ACCOUNTING.json)
- [九组矢量 PDF 与 PNG 曲线](analysis/figures/)、[图注与解释](FIGURE_CAPTIONS.md)
- [独立审查](INDEPENDENT_REVIEW.md)、[原证据字节审计](FROZEN_BASELINE_AUDIT.json)、[结果到主张](RESULT_TO_CLAIM.md)
- [版本清单](../project_management/GIT_MANIFEST.json)、[Git 核验](GIT_VERSION_AUDIT.json)、[实际发布收据](PUBLICATION_RECEIPT.json)

训练曲线只描述实际抽样，H 的贡献单列。固定终点不按开发结果选择；这里没有 S3 或原60保留任务的性能结果。
