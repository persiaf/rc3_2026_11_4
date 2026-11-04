# 输出清单

| 日期 | 文件 | 作用 |
|---|---|---|
| 2026-11-02 | EXPERIMENT_PLAN_20261102_v3.md / EXPERIMENT_PLAN.md | 本轮改进的主张、控制、划分与执行门槛 |
| 2026-11-02 | EXPERIMENT_TRACKER_20261102_v3.md / EXPERIMENT_TRACKER.md | 实验执行状态 |
| 2026-11-02 | EXPERIMENT_PLAN_20261102_v4.md / EXPERIMENT_PLAN.md | 执行后修订、最终协议和证据边界；不替代原测试前预案 |
| 2026-11-02 | EXPERIMENT_TRACKER_20261102_v4.md / EXPERIMENT_TRACKER.md | 全部实现、训练和测试完成状态 |
| 2026-11-02 | ../rc3_fast/EXPERIMENT_REPORT.md | 192 留出场景＋24 外推场景的结果、置信区间和局限 |
| 2026-11-02 | ../rc3_fast/deployment/selection.json | 模型测试前冻结记录，包含最终模型 SHA-256 |
| 2026-11-02 | EXPERIMENT_PLAN_20261102_v5.md | 多步局部修复 PPO 执行前计划 |
| 2026-11-02 | EXPERIMENT_PLAN_20261102_v6.md / EXPERIMENT_PLAN.md | 统一训练、标签、导出成本后的重新训练协议 |
| 2026-11-02 | ../rc3_improve/OPEN_SOURCE_NOTES.md | 四个开源项目的实际源码核查 |
| 2026-11-02 | ../rc3_review_import/IMPORT_AUDIT.md | 不完整压缩包的恢复记录与范围 |
| 2026-11-02 | EXPERIMENT_TRACKER_20261102_v6.md / EXPERIMENT_TRACKER.md | 新修复策略 v2 实际训练与完成状态 |
| 2026-11-02 | ../rc3_improve/RESULTS.md | 新独立测试、PPO/BC/规则/强基线对比与局限 |
| 2026-11-02 | ../rc3_improve/verification_v2/verification.json | 648 方案、参考界限、失败和缓存/重载/26单测的复验 |
| 2026-11-03 | EXPERIMENT_PLAN_20261102_v7.md / EXPERIMENT_PLAN.md | v3 Fixed3中间状态监督、停止机制修订与冻结协议 |
| 2026-11-03 | EXPERIMENT_TRACKER_20261102_v7.md / EXPERIMENT_TRACKER.md | 两轮训练、正式60场景测试及完整复验 |
| 2026-11-03 | ../rc3_v3/RESULTS.md | v3质量/完整时间对照与负结果 |
| 2026-11-03 | ../rc3_v3/deployment/policy.npz | 59744字节的便携MLP模型；仅验证选模 |
| 2026-11-03 | ../rc3_v3/verification/verification.json | 2583独立方案与成本界、861均值和23测试全部通过 |
| 2026-11-03 | ../rc3_v3/engineering_profile.json | 同18验证场景Fixed3工程提速，保持逐例目标成本 |

- 2026-11-03 v8: EXPERIMENT_PLAN_20261103_v8.md, EXPERIMENT_TRACKER_20261103_v8.md; external service evidence, exact reference, controlled structural input study.

- 2026-11-03 v8 results: rc3_v4/RESULTS.md, rc3_v4/results/RESULTS.md, rc3_v4/training/summary.json, rc3_v4/small_results/READOUT.md, rc3_v4/ROBOT_INPUT_STATUS.md.

- 2026-11-03 v8 internal result review: rc3_v4/claims_review.md, rc3_v4/findings.md, paper/review-traces/experiment-result-to-claim/2026-11-03_run02/. Verdict partial; retain Fixed3/Detour.

- 2026-11-03 v4 final delivery: rc3_v4_delivery.zip; 271 entries; SHA256 d08e572b582f2e22d855fc2c37e88f0ebaa793296501ed08496325a03d525d76; CRC and relocated CPU-only smoke passed. Receipts: rc3_v4_delivery.manifest.json, rc3_v4_delivery.smoke.json.

- 2026-11-03 v9: isolated rc3_v5 maintenance; paper/EXPERIMENT_PLAN_20261103_v9.md, EXPERIMENT_TRACKER_20261103_v9.md; common F/A coverage, frozen graph cache, separate Windows/Linux old-MLP supplements, pinned AlohaMini1 assets and blocked-input preflight. Evidence: rc3_v5/RESULTS.md, EVIDENCE_MAP.md, claims_review.md; reviewer trace run03. No retraining or retrospective v4 adoption change.

- 2026-11-03 v5 final delivery: rc3_v5_delivery.zip; 542 entries (541 payload hashes); 81,542,583 bytes; SHA256 0901d94faa949a2378af99b8314580b77669fe666a807fc55a89d7b3a97a4d81. CRC, every payload hash, unchanged 270 v4 files, relocated CPU-only planner/external/assets/protocol smoke all passed. Receipts: rc3_v5_delivery.manifest.json, rc3_v5_delivery.smoke.json. User confirmed real robot calibration is not yet available and authorized completing the currently verifiable scope; actual IK/collision integration remains deferred until calibrated inputs exist.

- 2026-11-03 v5 acceptance freeze / v5.1 maintenance: planning release v5 remains frozen (archive SHA256 0901d94faa949a2378af99b8314580b77669fe666a807fc55a89d7b3a97a4d81). Isolated gripper-chain preflight fix at rc3_v5_1/robot_input.py; 19 regressions and 3 CLI checks passed, relocated overlay checks passed. No training, performance remeasurement, calibration or IK. rc3_v5_1_maintenance.zip: 88 entries, 251710 bytes, SHA256 20559f39164ef20eb79f4850467cb2493c43425aac12bb672bd10154d869070f. Source/acceptance provenance and frozen display-case index: rc3_v5_1/README.md, ACCEPTANCE_SCOPE.md, DISPLAY_INDEX.md, display_manifest.json. Receipts: rc3_v5_1_maintenance.manifest.json and .smoke.json.

- 2026-11-03 frozen paper figures v1: 3 ordered figure groups plus common-region zoom (4 vector PDFs, 4 SVGs, 4 PNGs), L/K/J table, Chinese captions/argument links, LaTeX includes and standalone source-copy regeneration. No planning, inference, training, IK or performance measurement. Independent review PASS; 6 source hashes/90 pointers/7 displayed plans checked; relocated regeneration passed. Package rc3_paper_figures_v1.zip: 53 entries, 1518126 bytes, SHA256 cf4596952f46737049ca005d8ebede3575a91e786f9ac7f1e220f321bf30f829. Entry: paper_figures_v1/index.html; provenance: sources/source_manifest.json; review: FIGURE_REVIEW.md. LaTeX snippets were not compiled against a thesis template.
