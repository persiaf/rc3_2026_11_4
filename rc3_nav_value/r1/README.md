# R1 CNR-VN 单种子筛选

入口为 `python -B -m rc3_nav_value.r1.run register`，随后 `execute`。注册会固定源代码、H、种子与预算，拒绝覆盖。执行拒绝隐式重跑；任何失败尝试保留在调用账本中。使用项目既有 `rc3_learning/.venv/Scripts/python.exe`。

执行前计划见 `paper/EXPERIMENT_PLAN.md`；机器协议见 `protocol.json`。本轮最多两轮近似策略迭代，不读取保留测试、不自动启动 R2/R3；旧源码、模型和 R0 保持冻结。

结果读取 `quality.jsonl`、`labels_initial.jsonl`、`labels_round.jsonl`、`audit.jsonl`、`terminal.jsonl`、`training.jsonl` 与 `resources.json`。模型张量及配置在 checkpoints 内，逐场景完整部署方案在 plans 内。终局实际调用、标签使用、缓存命中、内部搜索及模型更新分别记账。最终离线分析不得重新构造路线或执行 H。

本次运行已完成并收尾：主门槛及两种缓存部署门槛均未通过，无参考晋升。请勿重复执行run；结果见父目录REPORT_R1.md及analysis.json。后验QA通过，不启动R2/R3。
