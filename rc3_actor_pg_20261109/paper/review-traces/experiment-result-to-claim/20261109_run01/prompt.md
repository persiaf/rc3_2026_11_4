# Independent paper architect/reviewer task

Experiment: `rc3_actor_pg_screen_20261109_01`. Branch supplied by executor:
`codex/actor-pg-screen-20261109`; frozen baseline
`925a0a1a8957299e3e849b29ac501a99212aa902`.

The parent delegated this review under the user-requested independent review and
the `experiment-result-to-claim` skill. The reviewer must judge the results rather
than accept the executor's summary. The target remains DRL training on large,
complex multi-task parent scenes, then frozen rapid complete local planning with
navigation, coverage, same-arm reachability/allow and safety constraints.

Review the new experiment's protocol, authorization, implementation review,
source binding `57b0d0d2236b4b1f942a393ce2fc41e6408f4e7ec899df038bdfc4b19a65eee6`,
input registry, deployment grid, engineering verification and saved raw evidence.
Do not change production source, original evidence, Git, old protocols or active
stage pointers. Write only this trace and `INDEPENDENT_REVIEW.md/.json`.

The allowed audit is read-only byte hashing, static code review, static
`torch.load`, stored-gradient arithmetic and saved-data statistical analysis.
No new model forward, rollout, optimizer, backward, scene/task generation,
navigation solve, native baseline, H or teacher call is allowed. Do not read old
60 held-out task semantics or refill/rerun any old S3 slot. Frozen historical
files may be byte-hashed without semantic interpretation.

Independently verify:

1. All 12 actor exports have exactly equal parameter names, shapes, dtypes,
   values and tensor hashes to their original checkpoints; originals remain
   unchanged. No quantization, pruning or distillation.
2. The 756 requests equal the registered grid and method order; raw/final routes,
   J/L/K, assignments and sources match across D0 full checkpoint per request,
   D1 actor-only per request and D2 actor parameters resident in a separate
   process. Separate cold/hot navigation and startup; verify 540 fresh encodings,
   H counts and D2 identity-key behavior.
3. Recompute complete-request median/p95 and retain startup costs. Fixed
   D0→D1→D2 order is not a randomized crossover. A startup-plus-request span sum
   excludes intervening receipt/debit IO and is not exact process-to-response.
4. A/B start at the common exact SL256 weights with separate fresh Adam and
   share 512 view identities. Check 128+128 updates, full likelihood/action/mask/
   reward/LOO traces, exact source binding, clipping, saved complete gradients,
   resulting weights and full checkpoints every 32 updates including Adam,
   RNG, query/cache/ledger and cursor identities. Check budgets and failures.
5. Recompute B vs A AND B vs shared SL, before and after common H, using 18
   DEV cases nested in three physical parents. Average costs within each parent,
   form relative gains and weight parents equally. Recompute 20,000 parent
   bootstrap draws with seed 11632026. Do not claim 18 independent layouts or
   three optimizer seeds. B jointly changes forced-start sampling and mixture
   weights; it is not a first-action-only causal ablation.
6. Also report grand absolute J means. These may have a different direction from
   equal-parent mean relative gains; do not use alternate aggregation to overturn
   the preregistered candidate screen or claim that lower absolute J worsened.
7. Check frozen evidence/source hashes with zero scientific calls. If no supplied
   paper or `paper/PAPER_CLAIM_AUDIT.json` exists, mark the paper audit unavailable
   and downstream writing provisional, without rejecting the actual result report.

Return separate verdicts for actor speed/exact quality, B vs A, B vs SL and the
overall DRL research claim. Include `claim_supported: yes|partial|no`, confidence,
technical `GO|REVISE|BLOCKED`, missing evidence, smallest defensible claims and
routing. No new experiment is authorized by this review. Preserve the DRL goal;
do not replace it with a supervised or rule-based research objective.

Tracing contract: save this prompt, raw reviewer response, parsed verdict,
routing, `pendingExternalReview` and a hash manifest. The referenced optional
shared `review-tracing.md` and `assurance-contract.md` files were not installed
under the skill's relative path; the explicit SKILL.md requirements above are
implemented directly. This is an actual independent delegated review, so
`pendingExternalReview` is false.
