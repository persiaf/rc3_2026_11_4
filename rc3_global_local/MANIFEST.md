# 本轮产物索引

中文为主要说明语言；本目录隔离旧实验。

| 文件 | 用途 |
|---|---|
| RESEARCH_OBJECTIVE.md / RESEARCH_STATUS.md / README.md | 固定DRL研究目标、阶段边界与入口 |
| KERNEL_REVIEW.md | 原约束/开端点H复用的只读核查 |
| LITERATURE_VERIFICATION.md / literature_sources.json | 六篇＋RL4CO一级来源核实及迁移边界 |
| paper/2026-11-06_EXPERIMENT_PLAN.md / EXPERIMENT_PLAN.md | 日期版与固定入口设计 |
| paper/2026-11-06_v2_EXPERIMENT_PLAN.md | 正式执行前的物理布局独立性修订，固定入口同步v2 |
| paper/EXPERIMENT_TRACKER.md | 实际阶段状态 |
| test_environment.py / unit_receipt.json | 独立环境契约测试、历史运行和零真实调用账 |
| test_policy.py / policy_unit_receipt.json | 独立神经结构测试、forward/backward/临时版本变更分账 |
| PREREG_REVIEW.md / prereg_review.json | 事前审查和当前源码绑定 |
| S0_protocol.json / S2_S3_protocol.json | 当前S0与未执行未来设计分别冻结 |
| FUTURE_DESIGN_STATUS.json / S2_S3_protocol_v2.json / future_seeds_v2.json | 训练前修正regular物理布局别名；v2为未来唯一有效设计，v1原字节保留 |
| seeds.json / seed_audit.json / execution_environment.json | 全部分配、历史种子核查、环境版本 |
| registration_receipt.json / run_started.json | 数据与真实探针之前的协议绑定 |
| parents/ / views/ | 父地图完整输入数组与子问题映射 |
| plans/ / episodes.jsonl / terminal_journal.jsonl | 完整轨迹、逐步事件、终局请求与重放账 |
| initial_policy.pt / gradient_probe.json | 未更新初始权重和无优化器梯度证据 |
| resources.json / STATUS.json / REPORT_S0.md / claims.json | 真实计账、状态、报告及受限主张 |
| audit_saved.py / audit_saved.json / FINAL_REVIEW.md | 独立保存数据复算及未来v2静态审查；几何验收历史与零新增研究调用分账 |
| closeout_integrity.py / baseline_integrity.json | 旧科学证据字节核对，根README/PROJECT_VERSION状态维护例外明列 |
| archive_receipt.json | 本地收尾、关键证据哈希与正式训练未启动状态 |
