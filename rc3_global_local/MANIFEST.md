# 本轮产物索引

中文为主要说明语言；本目录隔离旧实验。

| 文件 | 用途 |
|---|---|
| RESEARCH_OBJECTIVE.md / README.md | 固定DRL研究目标、范围与入口 |
| KERNEL_REVIEW.md | 原约束/开端点H复用的只读核查 |
| LITERATURE_VERIFICATION.md / literature_sources.json | 六篇＋RL4CO一级来源核实及迁移边界 |
| paper/2026-11-06_EXPERIMENT_PLAN.md / EXPERIMENT_PLAN.md | 日期版与固定入口设计 |
| paper/EXPERIMENT_TRACKER.md | 实际阶段状态 |
| test_environment.py / unit_receipt.json | 独立环境契约测试、历史运行和零真实调用账 |
| test_policy.py / policy_unit_receipt.json | 独立神经结构测试、forward/backward/临时版本变更分账 |
| PREREG_REVIEW.md / prereg_review.json | 事前审查和当前源码绑定 |
| S0_protocol.json / S2_S3_protocol.json | 当前S0与未执行未来设计分别冻结 |
| seeds.json / seed_audit.json / execution_environment.json | 全部分配、历史种子核查、环境版本 |
| registration_receipt.json / run_started.json | 数据与真实探针之前的协议绑定 |
| parents/ / views/ | 父地图完整输入数组与子问题映射 |
| plans/ / episodes.jsonl / terminal_journal.jsonl | 完整轨迹、逐步事件、终局请求与重放账 |
| initial_policy.pt / gradient_probe.json | 未更新初始权重和无优化器梯度证据 |
| resources.json / STATUS.json / REPORT_S0.md / claims.json | 真实计账、状态、报告及受限主张 |
