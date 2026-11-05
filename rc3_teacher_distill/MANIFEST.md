# 本轮输出索引

输出语言：中文；文件、JSON键及代码使用原有英文命名。均位于本研究目录，不覆盖旧证据。

| 输出 | 版本与用途 |
|---|---|
| T0_DIAGNOSTIC.md / t0_diagnostics.json | 只读证据诊断与输入哈希 |
| paper/2026-11-05_EXPERIMENT_PLAN.md | 时间戳实验设计原版 |
| paper/EXPERIMENT_PLAN.md | 同内容固定入口 |
| paper/EXPERIMENT_TRACKER.md | 执行状态 |
| protocol.json / registration_receipt.json | 首次场景、标签、训练前登记与字节收据 |
| seeds.json / seed_audit.json | 全部分配与历史使用排查 |
| PREREG_REVIEW.md | 独立事前审查与修复核对 |
| test_candidate_menu.py / unit_receipt.json | 独立单测与验收 |
| labels.jsonl / scenes/ / run/ | 新标签、完整前缀、路线、版本化终局与资源账 |
| weights/ / *_training.json / *_actions.json | 匹配监督端点与动作结果 |
| STATUS.json / T1_GATE.json / T2_NOT_RUN.json | 阶段门控；条件未启动有明确记录 |
| REPORT_T012.md / paper_claims.json | 可引用结论与范围 |
| T2_RESULTS.json / quality_time.csv | 仅当T2获准且完成才产生 |
