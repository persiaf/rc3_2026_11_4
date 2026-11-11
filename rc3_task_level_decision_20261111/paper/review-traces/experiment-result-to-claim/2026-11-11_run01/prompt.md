# Independent paper architect/reviewer prompt

Apply `experiment-result-to-claim` to the evidence-summary package `rc3_task_level_decision_20261111`, based on commit `e9b3fb86c3ee541ed815cd16dd0bb2d4d6967654`.

Intended claim: the package accurately summarizes existing complete-task results and implementation boundaries, and is sufficient for a teacher to decide whether a future main-policy training mechanism should be proposed. This is not a claim that the proposed mechanism has been implemented or is effective.

Review targets: `TEACHER_DECISION_BRIEF.md`, `TASK_LEVEL_COMPARISON.csv/.md`, `SAVED_PARENT_CONTRASTS.csv`, `LEARNABLE_MODULE_AUDIT.md`, `FIXED32_CLOSURE.json/.md`, `TABLE_BUILD_RECEIPT.json`, and static source `build_existing_tables.py`.

Evidence boundaries:

- Three experiments remain separate: Integrated selected five methods; conservative R03 fixed64/128 with two cohorts and six methods; latest frozen-reference PG/A1 with the same endpoints/cohorts/method count. Total table rows:53.
- Quality is within-parent case mean followed by equal parent weighting. DEV is3 exposed development parents/18 cases; SEL is8 exposed training parents/48 selection cases. These are not new independent generalization tests.
- Use only existing permitted result CSV/JSON and static code. Do not train, run a model forward, plan, invoke H/native/teacher/navigation, rerun the builder or bootstrap, read historical large diagnostic CSVs, read original60/oldS3, or read Word documents.
- Integrated selected RL128 and residual supervision256 have different selected update counts despite matched maximum opportunity. Saved `full_audit_s` is a development-request clock. R03 uses `through_record_IO_s`; latest PG uses `full_request_s` through its recorded I/O/journal. Do not compare speeds across these clocks.
- Native raw fields contain native final search/postprocessing results and cannot identify a no-H ablation.
- Keep the Integrated nonzero historical effects and P2 failure; do not claim that all RL effects were exactly zero.
- Fixed32 shared witness32/32 is an existence result on known states. CE14/32, Adam-rank18/32, QP13/32 remain failures. Preserve original counters, caps and units, including323/128/112 candidates,32 QPs,9050 dual iterations and22 isolated path assignments. No automatic continuation or new budget is authorized.
- Static implementation review must distinguish frozen SL score/representation, bounded trainable A1 logits, legal positive-gain selection, sampling versus greedy evaluation, fixed reference, supplied-state guard, complete return, raw-dependent deterministic H, automatic arm assignment and proxy validation. Full gamma=lambda=1 return already exists.
- A trainable main Actor initialized from SL is a future direction for teacher decision, not a demonstrated root cause, implemented change, or promised benefit. Shared32/32 prevents a blanket representation-impossibility conclusion; guard acceptance prevents attributing current PG zero gains to guard rejection.

Perform limited numeric spotchecks from the allowed sources and assess `claim_supported`, actual support, unsupported extensions, missing evidence, suggested revision, future-experiment need, and confidence. Write only the five review-trace files in this directory. The reviewer must not alter the package, authorize experiments, or route this summary verdict into confirmed novel-performance paper claims.
