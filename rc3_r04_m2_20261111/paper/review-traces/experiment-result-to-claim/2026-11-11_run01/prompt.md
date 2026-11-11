# Independent result-to-claim review prompt

Role: delegated, independent paper architect/reviewer for the `experiment-result-to-claim` skill. The executor collects and records evidence; the reviewer judges the claim.

Intended claim: **“CB 神经残差有足够增益值得保留。”** Judge `claim_supported: yes | partial | no`, the supported and unsupported statements, evidence gaps, revised claim, any next experiments needed, confidence, and routing.

Fixed research objective: train a DRL planning policy in large, complex, multitask parent scenes, then freeze it to rapidly generate complete routes for variable-size local tasks. Supervised components and statistical references are training aids or controls, not replacements for the DRL core.

Review scope: read only the current M2 small tables, protocol, receipts and scientific source. Do not run training, planning, old large-CSV audits, or supplementary experiments. Do not modify frozen evidence or the user's Word document. Write only the five review-trace files in this directory. Leave M1 G3 FAIL, M1 step256 NOT_RUN, and all historical resource records unchanged.

Evidence bundle:

- `rc3_r04_m2_20261111/attempts/02/M2_PROTOCOL.json` (registered before the first numerical calculation).
- `M2_DECISION.json`, `M2_METRICS.csv`, `PAIRED_PARENT_INTERVALS.csv`.
- `INDEPENDENT_NUMERICAL_RECHECK.json`, `DATA_REUSE_VERIFICATION.json`, `C1_REFERENCE_IDENTITY.json`, `WORKER_RESOURCE_ACCOUNTING.json`.
- `REGISTRATION_RECEIPT.json`, `ENTRY_CONSUMPTION.json` for preserved failure/time context.
- `rc3_r04_m2_20261111/{model,analysis,audit,worker,entry2}.py`.

Design: 33 fit physical parents, 16 already exposed holdout physical parents; saved 268-dimensional observable features and saved labels. Conditions are size and an observable remaining-fraction bin. Fit-only component anchors; same seeded 268-128-128-3 network, zero output layer, Adam schedule and 128 updates per CG/CB arm. C1 is the exact fixed M1 step128 export, not retrained. Value is composed from the same normalized L/K/H components. This is an exposed-holdout development mechanism screen, not independent generalization or online planning evaluation.

Pre-registered CB retention standard (all conditions must pass): in both state-pooled and equal-parent aggregations, Value MSE ratio <=0.95 and physical-parent paired bootstrap 95% upper bound <1 against both B-cond0 and CG128; each of the three normalized component ratios <=1.10 against both references; every size has at least 3 parents and Value ratio <=1 against both. CG versus its own B-global0 anchor uses <=0.95 and upper bound <1 in both aggregations for diagnosis only. Beating C1 alone does not establish residual value. Failed CB means stop this candidate; no added steps, conditions/features, network changes, joint PPO, R03 or old M3.

Collected results, verified directly in current small tables:

| Comparison, Value MSE ratio | State pooled (95% interval) | Parent equal (95% interval) |
|---|---|---|
| CB128 / B-cond0 | 0.8320688975 [0.7670512093, 0.9401564508] | 0.8933313876 [0.7939030075, 1.0154808881] |
| CB128 / CG128 | 0.3314353461 [0.2438564520, 0.4433031976] | 0.2970036941 [0.2351711948, 0.3857891615] |
| CG128 / B-global0 | 0.6357037709 [0.5638181022, 0.7159908494] | 0.6804749391 [0.5935048834, 0.7605210257] |
| B-cond0 / B-global0 | 0.2532178524 [0.1870803718, 0.3230372862] | 0.2262358330 [0.1776218167, 0.2871715247] |
| CG128 / B-cond0 | 2.5105013915 [1.7861740680, 3.7336653774] | 3.0078123790 [2.1208913526, 4.1334885346] |
| CB128 / C1_fixed128 | 0.2905930586 [0.2174618286, 0.3807799590] | 0.2640748742 [0.2115637326, 0.3361640998] |
| CG128 / C1_fixed128 | 0.8767714791 [0.8531682605, 0.8995854253] | 0.8891299317 [0.8616549203, 0.9117718801] |

CB has 29/32 retention checks passing. Failures: parent-equal ALL Value versus B-cond0 has upper bound 1.0154808881; 6x6 Value versus B-cond0 ratios are 1.0374282881 (pooled) and 1.1066706396 (equal parent), each based on 12 parents. All normalized component guards pass. Production decision is `STOP_DELETE_CB_RESIDUAL`.

Arithmetic receipt: PASS, 5,880 checks across 55,944 new exported system-state rows; 32 independent gate checks; no extra optimizer attempts. This audits arithmetic, not planning performance. Resource receipt: 256 completed Critic updates = CG128 + CB128; Actor, new scenes, navigation, teacher, native, H, original60 and oldS3 all zero. Initial default-Python import failed without Torch before updates/data; entry2 binds the exact M1 interpreter, preserves prior consumption and original protocol/scientific sources, and uses a distinct output directory.

The root and M2 `paper/PAPER_CLAIM_AUDIT.json` files are absent. Mark `paper_claim_audit=unavailable`, and downstream paper use `provisional — no paper claim audit available`. Independent reviewer judgment itself is complete, not pending external review. Skill-linked shared `review-tracing.md` and `assurance-contract.md` are absent from the local skills tree; use the skill body's specified full five-file trace and record that limitation.

Do not flatten the evidence into “all gains came from initial statistics”: CG has incremental evidence over its global anchor, and CB has visible pooled/point improvement over its conditional anchor. Distinguish those observations from the failed pre-registered retention decision. State which training-period reference to retain and what online evidence remains absent, without authorizing or executing it.
