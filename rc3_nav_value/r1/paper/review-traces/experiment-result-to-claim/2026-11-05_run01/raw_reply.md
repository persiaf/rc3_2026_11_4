# R1 RESULT-TO-CLAIM EVALUATION

Independent paper architect/reviewer; 2026-11-05. Evidence-only review. No scene generation, model instantiation or inference, optimizer step, solver, H, or new experiment was executed. The only writes by this reviewer are this response and `verdict.json` in the same trace directory.

**Main claim C1: `no`. Supporting representation claim C2: `no`. Research gate: fail. Deployment gate: fail in both cache modes. Close this configuration under its frozen stopping rule.** Confidence in these experimental facts and the screening decision is **high**. Any downstream paper-writing judgment is **provisional — no paper claim audit available**; `paper/PAPER_CLAIM_AUDIT.json` is absent.

## Evidence and independent checks

I read the frozen protocol and experiment plan, analysis, resources, promotion decisions, both invariant files, saved QA, training summary, quality-time CSV, calibration/model-load records, and the relevant frozen timing implementation. I independently read the saved `quality.jsonl`, `audit.jsonl`, `labels_round.jsonl`, `terminal.jsonl`, `call_journal.jsonl`, and `training.jsonl` records. All calculations below use these saved records and standard-library/NumPy arithmetic only; I did not import the project runtime.

- Recomputed all eight final comparisons using paired scene resampling with the registered seed 11412026 and 20,000 draws. Their means and percentile 95% intervals agree with `analysis.json` to numerical precision. Recomputed both round-one promotion comparisons separately.
- Verified the complete quality grid: 224 initial, 128 round-one, 128 round-two and 1,728 timing records, totaling 2,208. All have stored valid export and independent scalar certificates. The maximum error in `J=L+0.75K` is 2.85e-14. These are stored certificate checks, not a new geometric validation; feasibility remains limited to the registered discrete proxy and excludes IK.
- Compared all 32 raw routes, final routes, raw J and final J for A/C and B/D in both rounds: exact equality. Independently read checkpoint ZIP tensor-storage payloads without deserialization or model instantiation: all 44 storage entries are byte-identical for each corresponding pair in each round. This corroborates both invariant receipts.
- Recomputed all 18 method/cache timing summaries from the 1,728 raw records, including all 32 scenes and three repetitions per method/cache. The saved medians, scene-median p95 values, and separate raw-call p95 values agree. For D, Fixed3 and Detour16, `total_s=preparation_s+decision_s+oracle_s` and request wall equals total plus the saved bookkeeping residual exactly at saved precision.
- Rehashed all 882 files listed in `qa_saved.json`: no missing file and no hash mismatch. The saved QA reports 1,012,842 checks, zero failures. I did not claim to rerun that verifier or to independently reconstruct unsaved masks, Adam moments, or geometric certificates. The before/after frozen-integrity receipts both report 8,549 files, 2,963,640,420 bytes, and zero failures; this review did not itself rehash that entire historical inventory.
- The journal independently contains 9,286 starts, 9,286 completions, zero error events and 6,130 explicit cache hits. The 9,286 unique completed terminal IDs divide into init 4,096, round 2,551, audit 431 and validation 2,208, all within their respective caps. There are exactly 6,144 logged optimizer steps: two warm runs of 1,024 and eight group/round runs of 512. Per-configuration inherited logical steps are 2,048; 8,192 is not the actual optimizer-step total. Post-calibration elapsed time is 2,225.6702 s against the frozen 7,200 s cap. Resources report zero retained-test queries, R2 and R3 not started; saved identities/declarations do not prove absence of every possible unsaved read.

## 1. Main claim and reference-promotion interpretation

Relative gain is the mean of 32 per-scene `(J_reference-J_candidate)/J_reference` values, reported in percent below. It is not the percentage change of the two mean costs.

| Comparison | Mean relative gain (%) | Paired exploratory 95% CI (%) |
|---|---:|---:|
| C1 vs warm_base, promotion | +0.602016 | [-0.771501, +1.981853] |
| D1 vs warm_nav, promotion | +1.135547 | [-0.113863, +2.500297] |
| D2 vs B2 | 0.000000 | [0.000000, 0.000000] |
| D2 vs warm_nav | +0.596353 | [-1.169072, +2.493845] |
| D2 vs old128+H | -6.145567 | [-7.783351, -4.493346] |
| D2 vs native Fixed3 | -7.838244 | [-9.430081, -6.214013] |
| D2 vs native Detour16 | -9.175324 | [-10.887831, -7.423742] |

Both promotion means exceed 0.5%, but both two-sided 95% lower bounds are below zero. The registered rule requires all conditions jointly, so neither reference was promoted. Round-two labels for A/C retain the warm_base reference hash; B/D retain the warm_nav reference hash. No g1 target or changed-reference state distribution was realized.

Consequently, D2 and B2 are the same fitted policy in this run. Their zero difference is not an empirical estimate of the effect of a successful reference-policy update: that update never occurred. This result establishes that the registered R1 pipeline failed to generate an accepted reference update and failed its intended incremental-benefit screen. It does **not** establish that approximate policy iteration is generally ineffective. Two rounds of gradients under unchanged references cannot be described as completed effective API. Likewise, the point improvement over warm_nav is uncertain, while the degradation relative to old128+H is clearly negative within this exploratory comparison. All feasibility conditions pass, but every non-feasibility main-gate criterion fails.

## 2. Representation and learning evidence

Final mean J values are A2=C2=305.606685 and B2=D2=308.288794. The final B2-vs-A2 (equivalently D2-vs-C2) mean gain is **-0.943622%**, CI **[-2.425401%, +0.622741%]**. Under the shared initial labels, warm_nav-vs-warm_base is **-0.719578%**, CI **[-1.847786%, +0.377471%]**. Neither comparison supports a positive route-quality benefit from candidate-navigation relations in this configuration. Both intervals cross zero, so they also do not establish a general harmful effect of navigation relations.

There is limited queried-action fitting, not demonstrated all-menu learning. For D2, the saved queried-action MSE is 0.000952995, queried pair-sign accuracy 59.8665%, top-one hit rate 22.65625%, and mean queried-top-one advantage regret 2.530828%. A2 has a higher MSE (0.001156372) but higher pair-sign accuracy (62.5036%), higher top-one hit rate (25%), and smaller queried-top-one regret (2.078083%). These mixed diagnostics do not rescue a representation-benefit claim or explain causation. Round-specific reference targets differ across representations; regression losses are not a clean causal route-quality comparison.

All groups share the full ordered prefix GRU, true current/home D, exact masks, and prefix-edge costs. The controlled candidate-relation distinction concerns this finite architecture and training process; it cannot establish the information necessity of original G1–G4. The later B/A and D/C comparisons are comparisons of complete learning processes, whose references and state distributions may differ across representations. The common-label warm comparison is closer to a representation contrast, and is still negative and uncertain here.

## 3. Unknown-action audit

All 1,024 registered training-state audit opportunities are saved; 920 (89.84375%) select an action outside that state's queried menu. The 920 opportunity-level requests are not 920 new terminal calls: exact cache reuse reduces actual audit charges to 431. Counts of action selection, label uses and charged H calls must remain distinct.

D selects unknown actions in 124/128 states (96.875%) in each round. For D2's 124 unknown selections:

- Mean true normalized advantage against the unchanged reference is +0.117644%; mean optimistic prediction bias is +1.334950 percentage points.
- Mean positive relative regret to the best **queried** action is 3.639229%; the worst such relative loss is 12.444897%. Both quantities normalize by Yref. The queried best is not the all-menu oracle optimum.
- In 46.774194% of these unknown selections, the prediction is positive while the true advantage is negative. This is the joint-event fraction over all unknown selections, not an error rate conditional on a positive prediction.

D1's corresponding unknown mean advantage is -0.024905%, with +0.788251 pp optimistic bias and 3.622938% mean positive regret. These records expose a weak transfer from queried supervision to the model's actual full-menu choice on seen training states. They do not independently prove the complete menu's best action, diagnose a unique failure mechanism, or measure unseen-state generalization. They were not used as promotion/selection evidence, consistent with the protocol.

## 4. Complete quality–time deployment gate

Mean J is 308.288794 for D2, 286.450683 for native Fixed3, and 283.160033 for native Detour16. D2's mean relative degradation against Detour16 is 9.175324%, with a one-sided 95% upper bound of **10.598363%**, far above the permitted 0.5%.

| Cache | Method | Median full time (s) | p95 of scene-median full time (s) |
|---|---|---:|---:|
| cold | D2 | 0.648615 | 0.928438 |
| cold | native Detour16 | 0.570436 | 0.875049 |
| cold | native Fixed3 | 0.307744 | 0.441431 |
| hot | D2 | 0.521323 | 0.770080 |
| hot | native Detour16 | 0.430001 | 0.713648 |
| hot | native Fixed3 | 0.160462 | 0.272575 |

The D/Detour16 median and p95 ratios are cold **1.137051 / 1.061012** and hot **1.212374 / 1.079076**. Required limits are median <=0.7 and p95 <=1 in each mode. Both fail. Native Fixed3 also dominates D on all registered frontier axes (mean J, median full time, p95 full time) in both modes. Strong deployment support is absent.

The timing code includes fresh C/F/D preparation, per-request features/static encoding and construction, complete H or the full native baseline pipeline, export and independent scalar validation; GPU boundaries synchronize. Hot reuse is navigation-cache reuse, not neural embedding reuse. Persisting journal/summary files is excluded and saved separately as a residual; scene generation/JSON loading and model loading are outside this definition. The old128 load record is 0.0590271 s; this does not turn reported deployment-compute times into full application-startup latency. D's median saved wall residual across both modes is 0.0064153 s. Learned methods use this machine's GPU while strong rules use CPU, so the timing result has this hardware/protocol scope. Fixed3/Detour16 use their native pipelines, whereas learned/prior/old policies use common H: their names and pipeline definitions must not be conflated.

## 5. Evidence boundaries and missing evidence

**Failed under this frozen configuration:** accepted first-round promotion; the predeclared D2-vs-B2 >=0.5% increment and positive lower bound; positive lower bounds against warm_nav and old128+H; representation-based improvement in terminal routes; Detour16 quality tolerance and both complete-time requirements; absence of Fixed3 dominance. These failures are sufficient for the registered closeout decision. A lack of positive representation evidence is not a statistically established universal negative effect.

**Not tested or not identified:** an API iteration with an actually changed accepted reference; seed-to-seed robustness; an untouched confirmatory test (the same dev32 was used for promotion and final screening); unseen-state action-generalization performance; global all-menu optimality; information necessity of G1–G4; arbitrary architectures or RL algorithms; other deployment hardware; physical feasibility, calibration and IK. Training contains one task instance per training map. Single-seed scene-bootstrap intervals do not represent training-seed uncertainty, and development reuse makes them exploratory. Audit completeness/QA consistency cannot close these research gaps.

Evidence reproducibility also has saved-artifact limits: Adam moments and step tensors were not saved, individual eight sampled label indices were not logged, initial shared untrained tensors and full unqueried-action scores/masks cannot be recovered independently, and cross-batch forced-completion argmax equivalence cannot be certified from saved compact data. Endpoint tensor identities, matched schedules and source pinning provide useful checks but must not be advertised as reconstructing those unsaved states. No ledger, timing or saved-QA inconsistency found in this review blocks reporting the observed failures.

## 6. Suggested claim revision

> In a preregistered single-seed development screen using matched architecture, query budgets and updates, neither candidate representation produced a first-round learner that passed the frozen reference-promotion rule. The second round therefore retained the initial reference, and fixed-reference and nominal API learners were exactly identical within each representation. The final navigation learner did not pass the incremental-quality screen, showed no reliable route-quality advantage from candidate-navigation relations, and failed the complete quality–time deployment criteria against native strong-rule baselines. These results delimit this configuration; they do not test the benefit of an accepted changed-reference API iteration or establish a general failure of API or navigation information.

This is suitable as a bounded negative/diagnostic result, with the numerical table and exploratory scope retained. R0's teacher search opportunity remains separate historical evidence; its 2.2507% teacher gain is not a new network gain and contributes nothing to passing R1.

## 7. Routing and future evidence

`routing_action = close_current_configuration_no_automatic_R2_R3`. Record the negative screening result and preserve the complete evidence. Do not append updates, seeds, rounds, replacement runs, post-hoc endpoint selection, or ablation studies to rescue this failed configuration. Do not route it to confirmed positive-claim paper writing.

No further experiment is needed to decide this configuration's frozen screen: it has failed. Only a separate explicit task and newly frozen protocol could examine a substantively changed method. Such a future study would need to realize and document an accepted reference change before attributing any later contrast to API, establish credible full-menu choice on independent states, and use matched multiple seeds plus an untouched test for broader learning claims. A deployment claim would additionally need the complete native-baseline quality–time criteria on declared hardware. These are requirements for future claims, not authorization or a recommendation to continue training this run.
