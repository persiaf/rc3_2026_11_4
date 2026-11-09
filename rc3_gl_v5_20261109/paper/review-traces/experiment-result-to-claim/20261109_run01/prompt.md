# Independent result-to-claim review prompt

Role: independent Codex paper architect/reviewer, delegated under experiment-result-to-claim/SKILL.md. This is an independent agent role review, not external human peer review.

Experiment: rc3_gl_v5_frozen_greedy_20261109_01. Judge the intended claim that G128 improves frozen greedy plus fixed H complete-route costs against all three controls: the common SL256 initialization, natural-start LOO B128, and continued-SL C128.

The fixed research goal remains large-parent-scene DRL training followed by frozen rapid complete planning on local subscenes. Supervision and rules remain auxiliary. Do not substitute another research goal.

The only major treatment is replacing detached LOO with a frozen SL256, same-input greedy plus same-H cost baseline. Preserve the original B512 input schedule, seed11622026, fresh Adam from the exact same SL256 tensors, three natural trajectories and original stream names, weights 1/.25/.25, all first-action log probabilities, batch4, clip1, lr1e-4, network, reward, H and precision. The reference has separate model/prefix/cache ownership, consumes no policy RNG, and retains signed advantages.

The predeclared screen is an AND: for each SL256/B128/C128 control, the parent-layout mean relative reduction in H-final cost and its 95% lower bound must both be strictly positive; all432 evaluations must be feasible, finite and identity-valid. There is no additional Top1 gate. Evaluate exactly72 new tasks/views on12 B-trained physical layouts, six mixed and six blocked, six size/endpoint cases per layout. These are new task draws on seen geometry, one optimization seed. Do not call this unseen-environment or independent-multiple-seed confirmation.

Read only the GL-v5 protocol, input registry, source/engineering binding, phase receipt, saved432 evaluation CSV and referenced new result artifacts, training CSV, resource accounting, and necessary new source. Use Python/NumPy arithmetic on saved data to independently verify comparisons and identities. Do not instantiate or forward a scientific model; do not run an optimizer, navigation, native planner, H, or any new task generator. Do not inspect old S3 or original60 semantics. Do not modify scientific sources, results, frozen evidence, protocols or Git.

Use layout-first averaging, equal weight12 layouts, 20000 stratified bootstrap replicates, PCG64 seed11642026 and linear percentiles. For exact replay, reset the generator for each metric/control; draw mixed indices in one (20000,6) block, then blocked in one (20000,6) block.

Explicitly answer: does G improve SL; does it beat B/C; does either raw or H-final comparison improve; what extra training computation was consumed; is direct independent-multiple-seed confirmation warranted? Expected-score baseline invariance is not monotonic greedy improvement. A negative candidate result does not establish that DRL is theoretically ineffective. Diagnose clipping/variance only as observations or hypotheses unless causal evidence exists.

Write CLAIM_REVIEW_VERDICT.json and only this module's review-trace files. Required fields include claim_supported, confidence, what_results_support, what_results_dont_support, missing_evidence, suggested_claim_revision, routing_action, next_experiments_needed, paper_claim_audit='unavailable', provisional=true and pending_external_review=false. If the separate INDEPENDENT_AUDIT.json is not yet present, state that explicitly rather than fabricating PASS. No paper plan or paper claim audit is present. The skill's shared review-tracing reference is unavailable locally; preserve prompt, raw response, parsed verdict and routing directly. No new scientific execution is authorized by this review.
