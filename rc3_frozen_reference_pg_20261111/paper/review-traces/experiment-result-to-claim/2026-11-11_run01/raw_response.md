# Independent paper architect/reviewer judgment

claim_supported: **no**. confidence: **high for the bounded, observed development-screen conclusion**. The review is complete; downstream paper writing remains **provisional — no paper claim audit available**.

The intended positive claim fails its predeclared complete-route gate. This is a completed negative screen for the current configuration, not an unexecuted stage or a guard-rejection failure, and not evidence that frozen conditional references or DRL cannot work in general.

## Main result

I checked the actual QUALITY_CONTRASTS.csv primary rows and FINAL_DECISION.json against ENDPOINT_RESULTS.md. At64, PG-cond has exactly0% parent-equally-weighted raw-J and common-H J saving, with descriptive paired95% intervals [0,0], against FrozenSL, matched continued_SL_A1, and PG-global in both cohorts. At128, these same comparisons against FrozenSL and PG-global remain0% [0,0], and all selection-cohort comparisons against continued_SL_A1 remain0% [0,0].

At128 in the3-parent exposed development cohort, PG-cond versus continued_SL_A1 has:
- common-H J saving **-0.004556013374%**, paired95% **[-0.013668040122%,0%]**;
- raw-J saving **+0.000333203642%**, paired95% **[0%,+0.000999610926%]**.

These tiny opposite-sign effects neither meet0.5% nor have a strictly positive lower bound. They must not be summarized as all-zero or as a successful raw-route benefit. The per-size no-regression condition also fails at128 for8x8 common-H J versus continued_SL_A1: **-0.002312177173%** across11 parents (development-only3 parents: **-0.008477982966%**). All route-validity checks pass. Thus route validity is established within the saved model constraints, while the quality criterion fails at both endpoints. FINAL_DECISION=STOP_CURRENT_CONFIGURATION is appropriate.

The zero-width bootstrap intervals describe the observed parent cost differences. They establish neither population equivalence nor a universal zero treatment effect.

## What the results support

1. A functioning bounded, fresh online-PG comparison was executed. All three roles start with the same correction tensor digest; engineering weights are excluded. Source inspection confirms the current-policy hash/logit admission check, separate trajectory collection per RL role, frozen reference composition without a second n division, detached full-return advantage, original RMS scaling, one Adam candidate, and absent neural Critic.
2. The accounting records384 completed real Actor attempts,1536 own-policy complete trajectories,768 existing-label replays and792 evaluations, totaling2328 completed requests and0 failed requests. ONLINE_SUMMARY records128/128 naturally accepted opportunities in each role,0 rejection/rollback. The engineered injected rejection is a distinct transactional test, not the reason for the scientific stop.
3. The new-output audit reports PASS over1,326,384 arithmetic checks and145,309 diagnostic states, plus independent cost/paired-statistic recomputation. I inspected its implementation and receipts; I did not rerun the archives or historical CSVs.
4. For this seed, exposure set, accurateSL256, A1 schema, optimizer/guard/loss and128-opportunity budget, replacing B-global with B-cond did **not** produce retained complete-route objective improvement over either PG-global or FrozenSL, and did not improve matched continuation SL under the gate.
5. Complete-request timing is available. Across132 evaluation requests per method, the measured medians are PG-cond0.5091343s, FrozenSL0.4327211s, PG-global0.4883516s, continued_SL_A10.49720145s, Fixed30.3653685s and Detour160.43133615s. This does not support a speed advantage for PG-cond. These are saved-input requests inside an existing Python runtime, with cold model loads where applicable, uncontrolled filesystem cache, validation and main evidence I/O; they are not per-request fresh-process cold starts.

## Diagnostics and causal interpretation

For PG-cond's60,136 training diagnostic states, pooled scalar G-b_cond std is0.458068354538 versus G-b_global std0.943237948064. The conditional reference changes centering and reduces this descriptive pooled scalar dispersion on observed trajectories. It does not establish lower policy-gradient estimator variance, a calibrated value function, independent generalization, or route-quality improvement. There is no neural Critic and no Critic-learning claim.

PG-cond's effective correction range is small (median0.0003690995; observed max0.0008917212), with one base-to-applied top1 change among60,136 training states. These observations are consistent with limited decision movement in this run. They do not identify the unique cause of zero endpoint gains, prove an A1 bottleneck, or authorize a changed amplitude/lr/guard/budget. Action flips, nonzero gradients, changed weights, accepted updates and software PASS are not complete-route benefit.

## Native-H scope and reporting issue

The actual native pipeline differs from the protocol's overbroad promise of every native pre/post-H intermediate route. In runtime.py, only the shared baselines.postprocess wrapper populates H_invocations. Native recipe/two_opt/joint_search calls use other entry points. Native final routes, native_info and actual internal counts/inclusive-exclusive timing remain available; their raw_J is the final native route convention. It is valid to report native final quality/time as a descriptive baseline under that convention, but invalid to claim a no-native-H ablation or an isolated native-H gain.

This limitation **does not invalidate the main stop decision**: PG-cond/FrozenSL/continued_SL_A1/PG-global all retain raw and common-H route metrics and share the relevant comparison path. No native-H intermediate is needed to determine their zero/tiny route-quality contrasts. It does limit native postprocessing attribution and fulfillment of that original archival promise.

At evidence inspection, ENDPOINT_RESULTS.md still said every native internal pre/post-H route and call was archived, contrary to the accurate narrowing in SCHEMA_AND_ACCOUNTING_NOTES.md. The executor subsequently reported correcting that active summary and its delivery-helper wording, recording a reversible one-sentence documentation patch in POSTPROCESS_FORMAT_REVISION.json while preserving executed helper identity and frozen protocol/scientific source bytes. This correction is reported by the executor; this reviewer did not reopen or rerun the evidence. The archival limitation remains as described above and does not justify a native rerun or reopening this completed screen.

## Unsupported claims and remaining evidence

Unsupported: a useful route gain from B-cond over B-global; improvement over FrozenSL/matchedSL; isolated A1 or distance-summary efficacy; generalization across unseen parents or seeds; reaching the fixed large-parent/frozen-local DRL research goal; real robot IK or physical safety certification; isolated native-H benefit; fresh-process cold-start performance.

To support a future positive planning claim, a separately specified and authorized design would first need a measurable, reproducible complete-route gain over the same strong matched controls with its gate fixed in advance, followed by genuinely unexposed parent/task evaluations and multiple seeds. The current3/8-parent exposed cohorts are development evidence only. A mechanism study could distinguish why the bounded correction rarely changes route choices, but the present run does not identify a cause. No supplementary run is necessary to conclude that this current candidate failed its screen.

## Suggested claim wording and route

“在固定SL256、A1距离特征、冻结参考表和128次机会/组的单seed暴露开发筛选中，条件冻结参考改变了训练状态的中心化回报分布，但64与128终点均未达到完整路线收益门槛；相对FrozenSL与PG-global的raw/H目标节省为0。相对匹配继续SL仅出现极小、方向不一致的开发集变化，并有一项8x8 H目标点退化。因此停止当前配置，不将训练诊断作为规划收益。”

Route: **pivot / narrow the claim to this bounded negative result; STOP_CURRENT_CONFIGURATION**. Do not extend to256, resume oldM3/M1/M2/R03, supplement automatically, read held-out tasks, or proceed as a confirmed positive paper result. Maintain the fixed DRL research objective and preserve completed evidence. A future design/authorization is a separate decision. Parent executor should record the verdict in project notes; the active native-H summary correction was subsequently reported complete by the executor. This reviewer wrote only the five permitted trace files.

PAPER_CLAIM_AUDIT.json is absent. The independent reviewer verdict above is complete, not pending external review; only downstream paper-to-evidence assurance remains provisional.

