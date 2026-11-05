# C2.2 independent result-to-claim reviewer prompt

Review the fixed raw-cost versus complete-final-cost matched warm-start experiment against the original user design and preregistration. Before execution inspect objective sign, full original H0 then joint_search(passes=2) and normalize, fresh independent Adam, original update128 fixed greedy baseline, independent task and policy RNG, fixed 128 endpoints and explicit resource caps.

After completion independently recompute the saved data: 4,096 shared instances, 8,192 sampled routes, 4,096 actual baseline routes / 8,192 uses, 8,192 training complete-H calls; calibration separate. Check source/checkpoint hashes, log means, fixed-endpoint evaluator, all summaries and scene-paired bootstrap g/h, and the exact preregistered three-way rule. Inspect all 288 final-plan hashes and certificate receipts without creating new certificates.

Primary claim: final-cost training B gives a clear useful mechanism signal versus matched raw-cost A, and B also improves the frozen warm start. Supporting interpretation must distinguish avoiding A deterioration from improving the start. Stronger deployment, cross-seed and cross-scale claims require evidence absent from this round.

Report claim_supported yes|partial|no, confidence, what is supported, what is not, missing evidence, revised wording and routing. Do not invent exact per-trajectory loss verification when only mean_logp was saved. Acknowledge no paper claim audit is available, so paper-writing status is provisional.

Strict scope: saved-record arithmetic, source and data inspection only. Do not instantiate a policy, run inference, training, postprocessing, planners, scene generation, or geometry validation. Do not query the reserved 60 tests. Write only this trace. This is an independent agent review, not external human review.
