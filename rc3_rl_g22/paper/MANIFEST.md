# G2.2 文档与证据索引

日期：2026-11-04。新实验仅在rc3_rl_g22目录，旧版本保持冻结。

| 产物 | 来源／用途 |
| --- | --- |
| 2026-11-04_EXPERIMENT_PLAN.md、EXPERIMENT_PLAN.md | 规划前固定的评估范围 |
| EXPERIMENT_TRACKER.md | 已完成72槽位及0训练/PPO的追踪 |
| PAPER_PLAN.md | 两组反馈图的展示范围 |
| ../REPORT_G22.md | 从审计结果生成的中文主报告 |
| ../outputs/protocol.json、protocol.sha256 | 场景、权重、上限、方法及代码身份 |
| ../outputs/plan_index.json | 72份完整路径、SHA及结果索引 |
| ../outputs/action_feedback.csv | 即时／策略后缀／STOP解析记账和有限教师范围 |
| ../outputs/plan_audit.json、inference_replay.json | 保存计划及52模型输入检查 |
| ../outputs/freeze_before.json、freeze_after.json | 原G2及G2.1未变 |
| ../EXPERIMENT_RESULT_TO_CLAIM.md、review-traces/ | 独立数据复算及主张边界 |
| ../figures/ | PDF、PNG、生成脚本、图注与图表复核 |

每个新文件的逐项SHA由交付根目录MANIFEST.sha256.json提供；同级压缩包回执覆盖清单本身。新解压目录检查只读取保存结果，不重跑规划。
