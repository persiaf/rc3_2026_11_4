# 研究材料清单

- 2026-11-04_EXPERIMENT_PLAN.md：事前计划原本。
- EXPERIMENT_PLAN.md：相同内容固定入口。
- EXPERIMENT_TRACKER.md：阶段执行状态。
- ../protocol.yaml：机器可读JSON/YAML协议。
- ../OPEN_SOURCE_NOTES.md：已查阅来源及未运行对照边界。
- FIGURE_PLAN.md、../figures/：可重生成学习曲线、质量—时间图、表格与独立图表QA。
- ../outputs/registration.json、execution_sources.json、wall_protocol.json：本地事前登记、训练前修订源码和壁钟上限。
- ../outputs/small_exact.json、references.json、final_evaluation.json：12例精确参照、C0的480测量及C2的672测量；独立开发场景仍24个。
- ../outputs/checkpoints/、training.jsonl、sampled_routes.jsonl.gz、selection/：全部检查点、512更新日志、16,384优化采样路线及选择记录。
- ../outputs/verification.json、replay_recovery.json、verification_attempt1.log、verification_completion.log：数值重放、UTF8失败来源及显式完成记录。
- ../outputs/resource_accounting.json、analysis.json：完整资源与统计口径。
- ../REPORT_C012.md、EXPERIMENT_RESULT_TO_CLAIM.md、findings.md：最终报告与路由。
- review-traces/experiment-result-to-claim/2026-11-04_run01/：独立提示、原回应、结构化判定、路由及独立复算脚本／JSON。
- 根项目project_management/GIT_MANIFEST.json：所有交付文件的逐项SHA256清单，以Git版本管理，不生成额外ZIP备份。

## C2.1（2026-11-05）

- ../REPORT_C21.md、../c21.py：冻结路线归因报告与有界执行／保存复算入口。
- ../c21/protocol.json、call_ledger.jsonl、execution_environment.json：事前固定参数、输入／权重哈希与72次调用账。
- ../c21/rows.jsonl、analysis.json、per_scene.csv：三来源H0/H1候选集合、J/L/K及三类增量。
- ../c21/plans/、verification.json、resources.json、frozen_integrity.json：72方案、独立终验、资源分类和冻结字节核对。
- ../c21/REQUEST.txt：原始用户审查意见及本轮任务范围。
- review-traces/experiment-result-to-claim/2026-11-05_run01/：独立结果到主张审查及算术复核；无额外规划或证书。

## C2.2（2026-11-05）

- ../REPORT_C22.md、../c22.py、../c22_close.py：目标对照报告、一次性执行与保存记录复算。
- ../c22/protocol.json、instances.jsonl、mechanism_seeds.json、wall_protocol.json：事前协议、4096任务流、48新机制种子及壁钟。
- ../c22/paper/2026-11-05_EXPERIMENT_PLAN.md及EXPERIMENT_PLAN.md、EXPERIMENT_TRACKER.md：本阶段计划与状态；原阶段计划保留。
- ../c22/training.jsonl、training_routes.jsonl.gz、training_summary.json、checkpoints/：256新更新、8192sample与4096baseline路线及两终点。
- ../c22/evaluation.jsonl、analysis.json、per_scene.csv、scenes/、plans/：288质量结果、配对统计与48场景执行方案。
- ../c22/resources.json、verification.json、frozen_integrity.json、REQUEST.txt：资源分类、合法性重放、原文件字节审计与用户任务。
- review-traces/experiment-result-to-claim/2026-11-05_run02/：独立算术与主张审查，不执行新规划或训练。

## 当前同结构支线收尾（2026-11-05）

- ../REPORT_CLOSEOUT.md：收尾决定、统计解释与论文边界。
- ../NAVIGATION_INFORMATION_GATE.md：未来独立任务门槛，未启动搜索或训练。
- ../closeout.py、../closeout/analysis.json：保存记录复算与输入哈希，零新研究调用。
- ../closeout/STATUS.json、frozen_integrity.json、REQUEST.txt：关闭状态、保留权重、原文件字节核对与任务依据。
- review-traces/experiment-result-to-claim/2026-11-05_run03/：独立统计与主张复核。
