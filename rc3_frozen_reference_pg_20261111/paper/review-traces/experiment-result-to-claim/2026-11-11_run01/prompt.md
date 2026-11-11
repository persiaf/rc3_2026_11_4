# Independent result-to-claim review prompt

Role: independent paper architect/reviewer, delegated by the executor under experiment-result-to-claim. Judge the evidence; do not run training, models, planning, historical CSV audits, or additional experiments. Write only the five review-trace files in this directory.

Intended claim: “冻结B-cond相比B-global能使当前A1 Actor产生值得保留的完整路线收益，并改善FrozenSL和匹配继续SL。”

Fixed research goal: train a DRL planning policy in large complex multitask parent scenes; freeze its Actor to rapidly complete full constrained task and route planning for different-size local scenes. Navigation, coverage, proxy reachability and safety evidence remain distinct from real IK/robot certification.

Read-only evidence:
- PROTOCOL.json, IMPLEMENTATION_REVIEW.md, SCHEMA_AND_ACCOUNTING_NOTES.md
- ENDPOINT_RESULTS.md, ONLINE_REFERENCE_SUMMARY.csv, REQUEST_TIMING_SUMMARY.csv, ONLINE_SUMMARY.json
- execution_v2/engineering/CHECKS.json, execution_v2/ENGINEERING_ENTRY_CONSUMPTION.json
- execution_v2/science/FINAL_DECISION.json, QUALITY_CONTRASTS.csv, INDEPENDENT_NEW_OUTPUT_AUDIT.json, RESOURCE_ACCOUNTING.json, INITIALIZATION.json, ACTUAL_ENVIRONMENT.json
- execution_v2/SCIENCE_ENTRY_CONSUMPTION.json
- actor.py, reference.py, trainer.py, runtime.py, worker.py, analysis.py, audit.py
- Imported A1 source rc3_r04_review_20261111/relative_actor.py, read solely to inspect the actually used legal-pre-tanh correction.

Experiment: three fresh A1 roles, same accurate frozen SL256, 206 distance-ON features, 64/64 residual head, zero final layer, +/-0.25 legal-pre-tanh-centered correction. PG-global and PG-cond separately sample their own current-policy full trajectories with preregistered common random numbers. One candidate and one real Adam update per opportunity; no neural Critic, no TD; detached complete G minus frozen already-normalized component reference. Matched continued_SL_A1 replays existing labels. Both 64/128 endpoints are fixed; no 256 extension.

Predeclared gate: >=0.5% parent-equally-weighted saving and paired physical-parent bootstrap95 lower>0, against FrozenSL and matched SL, then PG-global, for raw_J and common-H J in each exposed cohort; no size point regression vs SL controls; valid complete routes. Only 8 trained-parent/new-task selection parents and 3 exposed development parents, single seed.

Executor reports engineering complete (19 requests,6 real attempts,41.2348255024s cumulative including prior failed entry); science complete (2328 requests,384 real Actor attempts,1536 online full rollouts,768 existing-label replays,792 endpoint requests), zero failures/Critic/new teacher/new scene/navigation solve/original60/oldS3. Independent new-output audit PASS:1,326,384 checks/145,309 diagnostic states. Verify from listed files rather than treating this summary as judgment.

The executor's initial “all comparisons zero at128” summary was explicitly corrected before verdict: FrozenSL/PG-global remain zero; the development cohort's continued_SL_A1 comparison has tiny opposite-sign raw/H changes. Read actual CSV and endpoint table.

Known record limitation: Actor/FrozenSL raw/common-H routes are retained. Native Fixed3/Detour16 use their original full recipe/search, no added common H; their raw means native final route, not a new no-H control. Shared postprocess H_invocations does not capture every native two_opt/joint_search intermediate. Native final route/native_info/internal counts and times are recorded. Keep original protocol/source bytes and do not invent missing native intermediates.

Evaluate claim_supported yes|partial|no; actual supported/unsupported claims; evidence gaps; revised wording; future evidence needed, without authorizing or running it; confidence; route confirm|supplement|pivot. Do not treat advantage signs, scalar dispersion, MSE, loss, weights, guard acceptance or action changes as route gains; do not claim isolated A1/distance efficacy. Check for paper/PAPER_CLAIM_AUDIT.json; if absent, label downstream paper-writing verdict provisional, while marking this independent review completed.

Shared review-tracing/assurance templates were not present at their skill-relative paths; use the skill body's five-file trace contract. Parent executor owns updates outside this trace and Git delivery/resource cumulative accounting.

