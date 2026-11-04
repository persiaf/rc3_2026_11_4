# RESULT-TO-CLAIM EVALUATION

Intended claim: 学习选择局部覆盖修复窗口，能在固定计算预算下改善路径质量，并支持快速可靠规划。

Read-only evidence: `results/metrics.json`, `results/summary.json`, `run_v3_stopfix_20261102/training_summary.json`; model frozen before test. Baselines Fixed3, random-window and detour-window selection share the exact same navigation/repair/validation modules. Call budgets 1/2/4/8; soft wall budgets 200/400/800 ms on nine 100-table scenes only.

Please provide claim_supported yes/partial/no; what_results_support; what_results_dont_support; missing_evidence; suggested_claim_revision; next_experiments_needed; confidence.

Key caveats: 54 primary vs 6 stress cases; 100-table subset n=9. Three timing repeats are not independent scenes. One validation-selected checkpoint, not multi-seed test uncertainty. Wall horizon cap256 extrapolates training horizon8. Canonical wall CI group is in_distribution_100; duplicated groups are not extra samples. Proxy reachability and fixed height, no real IK/hardware validation.

Raw facts shared to avoid transcription errors: main54 calls8 vsFixed3 +0.78645% reduction, 177.39 vs52.80ms warm; vsdetour +0.06015% CI[-0.10963,0.24071], vsrandom +0.14598% CI[0.00299,0.27549]. 100-table calls8 learned gain0.66997%, detour0.91434%; wall400 vsdetour -0.32212% CI[-0.64694,-0.01667], wall800 -0.26545% CI[-0.54041,0]. Independent verification2583/2583 plans,23tests. Engineering profile is separate18validation scenes: unchanged Fixed3 costs,141.56→77.36ms,median paired speedup1.8022; not learning benefit.

Review roles: delegated to `/root/local_repair` by `/root/review_archive`. Both are internal contributors; this is not a zero-context external paper audit. The skill's referenced shared review-tracing document is unavailable locally, so this trace explicitly retains prompt, response, parsed verdict and routing instead of asserting compliance with an unread template.
