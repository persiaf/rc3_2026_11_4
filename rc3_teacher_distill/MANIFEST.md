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
| saved_result_audit.py / saved_result_audit.json / FINAL_REVIEW.md | 仅对保存记录独立复算与结果审查，无新增研究调用 |
| calibration_replay.json | 冻结张量与已有特征的验收预测补录；390成员前向/3120行独立计账，不修改原研究资源或停止判断 |
| execution_environment.json | 研究后安装环境快照，明确区别于事前协议 |
| EXPERIMENT_RESULT_TO_CLAIM.md / paper/review-traces/ | 结果到主张审查、原始评审与路由记录 |
| baseline_integrity.json / closeout_integrity.py | 原维护基准全部已登记文件字节核对 |
| archive_receipt.json | 本地分支收尾状态、关键证据哈希及研究/验收分账，不含推送授权 |

2026-11-06：T1未通过，T2_RESULTS.json和quality_time.csv未生成，交付T2_NOT_RUN.json记录条件停止；不将未执行的T2写成完成。
