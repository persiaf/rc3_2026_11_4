# R1 文档清单

| 文件 | 来源/用途 |
|---|---|
| 2026-11-05_EXPERIMENT_PLAN.md | 执行前时间戳计划 |
| EXPERIMENT_PLAN.md | 同版固定入口 |
| EXPERIMENT_TRACKER.md | 阶段状态，依据原始账本更新 |
| EXPERIMENT_RESULT_TO_CLAIM.md | 独立主张判定摘要，当前配置收尾 |
| ../protocol.json、registration_receipt.json | 首次研究调用前冻结的机器协议与源码哈希 |
| ../analysis.json、training_summary.json、resources.json | 完整筛选、拟合、计时与资源 |
| ../per_scene.csv、per_request.csv、quality_time.csv | 逐场景成本、全部请求计时和汇总 |
| ../qa_saved.json、frozen_integrity_before/after.json | 保存证据验收与旧文件字节核验 |
| ../labels_*.jsonl、audit.jsonl、terminal.jsonl、call_journal.jsonl、cache_hits.jsonl | 全部标签、实际未知选择、终局调用及缓存来源 |
| ../checkpoints/、scenes/、plans/、training.jsonl、execution.log | 原始研究证据与权重；逐文件SHA见项目总清单 |
| ../../REPORT_R1.md、../findings.md | R1事实、主张范围与当前配置收尾 |
| review-traces/experiment-result-to-claim/2026-11-05_run01/ | 独立审稿原回复、结构判定及路由 |

运行协议、源代码与种子文件在父目录，以 registration_receipt.json 记录 protocol.json 冻结哈希。最终所有项目文件哈希由 project_management/GIT_MANIFEST.json 管理。
