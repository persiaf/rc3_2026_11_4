# Independent RESULT-TO-CLAIM EVALUATION prompt

Reviewer: independent agent paper architect/reviewer, not implementer and not human reviewer.
Experiment: conservative_paired_11622026_v1, branch codex/rc3-conservative-paired-ppo-20261110.

Intended claim: 同SL256冻结起点，206维ON相比同结构OFF、FrozenSL和匹配继续监督，在已训练父布局和独立父布局的不同窗口上稳定改善完整raw/H J/L/K，并保留完整请求时间价值。单seed/仅3已暴露独立父布局不能证明泛化。

Read only the 12 listed files below. No input/source-task/old60/oldS3/Word/raw request JSON reads; no models, planner, H, validator, optimizer or new experiment. Light host CPU arithmetic <=60s cumulatively; failures count and no large analysis rerun. Formal stage closed; saved-only analysis audit PASS is engineering evidence, not scientific success. Native raw is native final search/H output, not a shared-pre-H ablation. Do not fill endpoint256.

Required checks: exact per-case raw/H cost and route identities at64/128 for ON/OFF/continued_SL vs FrozenSL; parent cluster points and CIs; all6 methods costs and full timing; state-pooled Critic stop vs parent-macro components; PPO G/V/A distributions; actual guard accept/reject/greedy counts and their limits; resource/stop/data-exposure limits.

Return claim_supported yes|partial|no, supports/dont_support/missing_evidence/suggested_revision/next_experiments_needed/confidence. Set paper_claim_audit=unavailable and provisional. Only propose a new-protocol review path; no experiment execution or DRL-objective change. Save only new INDEPENDENT_FORMAL_REVIEW.md/.json and this trace prompt/raw_response/verdict/ROUTING, without overwriting existing evidence, implementing changes or committing.

Allowed result/protocol files:
- rc3_conservative_refinement/integration_20261110/FORMAL_EXPERIMENT_REPORT.md
- rc3_conservative_refinement/integration_20261110/FORMAL_RESOURCE_ACCOUNTING.json
- rc3_conservative_refinement/integration_20261110/PROTOCOL_REVIEW.md
- rc3_conservative_refinement/integration_20261110/PROTOCOL_DRAFT_R03.json
- rc3_conservative_refinement/integration_20261110/runs/conservative_paired_11622026_v1/analysis/AUDIT.json
- rc3_conservative_refinement/integration_20261110/runs/conservative_paired_11622026_v1/analysis/QUALITY_CONTRASTS.json
- rc3_conservative_refinement/integration_20261110/runs/conservative_paired_11622026_v1/analysis/STOP_SUMMARY.json
- rc3_conservative_refinement/integration_20261110/runs/conservative_paired_11622026_v1/analysis/CRITIC_CALIBRATION_PARENT_MACRO.csv
- rc3_conservative_refinement/integration_20261110/runs/conservative_paired_11622026_v1/analysis/GUARD_UPDATES.csv
- rc3_conservative_refinement/integration_20261110/runs/conservative_paired_11622026_v1/analysis/PER_STATE_ADVANTAGES.csv
- rc3_conservative_refinement/integration_20261110/runs/conservative_paired_11622026_v1/analysis/ALL_RAW_H_ROUTES.csv
- rc3_conservative_refinement/integration_20261110/runs/conservative_paired_11622026_v1/analysis/ALL_REQUEST_TIMING.csv
