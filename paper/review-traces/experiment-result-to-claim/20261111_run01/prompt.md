# Independent RESULT-TO-CLAIM review prompt

Role: independent paper architect/reviewer. Stage: rc3_critic_diagnostics_20261111.
Intended claim: offline Critic scaling and a composed-value loss improve held-out-parent numerical calibration; Actor nonresponse has identifiable bound/common-offset mechanisms. No route-benefit claim is established.

Read only saved PREREGISTRATION (necessary config; do not dump all hashes), CALIBRATION_SPLIT, MECHANISM_SUMMARY, CRITIC_MATCHED_CALIBRATION.csv, CRITIC_STRATIFIED_CALIBRATION.csv, NUMERICAL_RECHECK status/coverage, RESOURCE_ACCOUNTING, RECONSTRUCTION_RECEIPT, MISSING_SAVED_FIELDS, plus necessary original R03 NEXT_STEP_DECISION/FORMAL_EXPERIMENT_REPORT context. Check only existence/verdict of paper/PAPER_CLAIM_AUDIT.json per skill.

No new model forward, optimizer, fit, numerical audit script, route/task generation, original60 or S3 semantic access. All 768 Critic steps and the one complete independent numerical recheck have finished within cumulative 418.65731119999964 seconds; no science may be appended.

Design: 128 historical saved trajectories, 64 ON/OFF pairs, 50 physical parents; complete-parent split 33 fit / 17 calibration holdout, never state-random. 268 inputs/no new features. C0 raw-component parameterization and original normalized component loss; C1 direct per-initial_n normalized outputs; C2 C1 plus composed-value loss. Each has 256 steps, same net tensors and batch schedule. C0 and C1/2 have different physical initialization meanings. Saved online Critic predictions mix changing collection checkpoints and are not a fourth fitted configuration.

Answer:
1. Main bias, scale and conditional/state-error evidence for original Critic.
2. Whether C1/C2 improve holdout; report both state_pooled and parent_equal, both fit-only global/conditional constants and initialization. C2 parent_equal value MSE .4180583911 beats conditional .4512044008; pooled .3516746747 loses to conditional .3244870376.
3. Proportion under strict >.5 base-gap impossibility under residual +/- .25; list single-legal states separately.
4. Whether actual common offset dominates; 92 state rankings changed but top1 changes zero; distinguish pre/post ratios and theoretical bound from actual residual.
5. Recommend only one next-version change; distinguish enough evidence for a proposal from formal authorization or route benefit. Do not automatically enlarge residual range/budget/rounds.

Return claim_supported yes|partial|no, what_results_support, what_results_dont_support, missing_evidence, suggested_claim_revision, next_experiments_needed, confidence. paper claim audit missing => unavailable/provisional. Do not modify old/frozen files; create only this stage's independent review .md/.json and findings.md, and paper/review-traces/experiment-result-to-claim/20261111_run01/{prompt.md,raw_response.md,verdict.json,ROUTING.md}. Do not commit.
