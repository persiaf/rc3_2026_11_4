# Independent R08-B result-to-claim review

Verdict: **claim_supported=no; confidence=high; OPPORTUNITY_INSUFFICIENT**. This independent reviewer does not approve entry to C. `pending_external_review=false`; `paper_claim_audit=unavailable`, so any later paper-writing verdict is provisional without a paper claim audit.

## Review prompt and scope

Judge whether the fixed R08-B opportunity run, based on commit `27deb2daf46f091cc2b9a7e3c475b7242eb2753e`, meets the original opportunity gates and supports entering C. Check menu ordering/cancellation/dedup, lower solver/original H, proxy legality, raw-versus-H cost roles, independent CSV arithmetic, query prefixes, accounting and the D projection. Read saved allowed sources only. Do not execute any forward pass, planner, H, optimizer, native, scene, teacher or navigation operation. Do not read old60, oldS3, confirmation or final task semantics. Do not invoke Git or delegate another agent.

Reviewed the four original R08-A normative documents; PROTOCOL; menu.py/lower.py/science.py; ANALYSIS_SOURCE_BINDING; RESULTS; ACTION_RESULTS/TASK_PARENT_RESULTS/PREFIX_SUMMARY CSV; RESOURCE_ACCOUNTING; SUPERVISOR_RECEIPT; all saved menu/task JSON; and five individually addressed gzip members (F058 anchor/edit03/edit07; L176 edit04; L136 edit01). Main B report/claim/next-step documents were not yet present at this review. New scientific calls=0; Git commands=0.

## Independent gate calculation

Gains were recalculated directly from anchor_J and edit_J for all 671 CSV rows, rather than trusting relative_gain. Maximum difference=0; material flag mismatches=0. There are 48 tasks and 48 distinct physical parents. Missing slots=97, so 48+671=719 actual requests rather than 816 nominal; missing slots remain zero opportunity under the fixed task denominator.

|Original gate|Independent result|Decision|
|---|---:|---|
|At least 8 full / 4 local parents with material opportunity|11 full / 8 local|PASS|
|Overall material-task fraction >=30%|19/48 =39.5833%|PASS|
|Full and local fractions each >=25%|11/32=34.375%; 8/16=50%|PASS|
|Parent-equal nonnegative best-edit oracle mean >=1.5%|1.1954778056%|FAIL|
|Material edits with net explicit drop=2 or add=2 >=50%|11/74 =14.8648648649%|FAIL|
|All original proxy plans valid|Original aggregate reports 719/719; source path and five raw/H certificates consistent|PASS in recorded evidence; sampled independently|
|D engineering projection <=21600s|12660.580331580939s|PASS as projection|

The registered conjunction fails. The correct scientific result is OPPORTUNITY_INSUFFICIENT, not a technical-abort result and not partial approval for C.

## Implementation and attribution

`menu.py` implements sorted adjacent-first alternating block classes, at most eight drop pairs, at most eight pool candidates using coverage-count/D/ID order, deletion-only when U is empty, one/two additions when U is nonempty, intersection cancellation before arity and ranking, local first two before global net-edit dedup, and no duplicate backfill. All 48 saved menus reproduce that dedup stage with zero structural errors and 671 unique-per-menu actions. Explicit net-joint labels agree with post-cancellation arity. The source reads no prospective completed J/H to rank the menu. The U-empty operationalization and tie order are registered explicitly; no post-result threshold change was used.

`lower.py` removes the net drop set, inserts additions in global-ID order at finite minimum navigation increment (earliest-position tie), checks complete coverage/actual finite edges, calls eight-pass-capped 2-opt, then one normalize. There is no extra pre-H joint-search call. `science.py` starts edits from the raw FrozenSL incumbent, freshly reproduces the saved frozen raw and H reference, validates raw output, calls original H once per request, and requires final original model validation. The ledger's 719 H calls and internal joint-search calls are compatible with H-internal work; they are not evidence that the lower solver secretly expands its action family.

The five exact compressed members were individually sought and SHA-checked. Full/local examples retain complete original-model raw/H certificates, same-arm F&A, valid endpoints and no collision; cost samples match CSV raw_J versus raw evidence and J versus H evidence. The input role on sampled tasks is train and the loader source is explicitly preexisting hash-pinned input. H identity is `5db67cacc1128b6143c7a04b9b84a70c37bdc0ab917a0eebae0b8aca132c4d14`. No historical task body was opened by the reviewer. All CSV J=L+0.75K identities pass; maximum delta-identity error=5.684e-14.

Crucial scope: an explicit 1-to-1 edit can cause much larger cleanup and H set changes. F058 edit03 is one such material edit: its raw normalize removes 40 configurations and H changes the final set. That does not make it an explicit joint success. Only 11 material edits meet the registered net joint arity. Changes in L do not isolate an order-only causal effect, changes in K do not prove an independent set-selection mechanism, and changed task/arm responsibilities are induced by the retained deterministic export. No controlled fixed-set or no-H causal comparison was run. The negative opportunity gate remains meaningful for this finite menu plus registered lower/H pipeline; it does not prove all 2-exchange spaces or DRL are ineffective.

## Query-prefix and timing check

The saved prefix table was independently reconstructed by fixed menu index, using a nonnegative best gain and the fixed 48-task denominator. Total times include each task's full anchor request, full menu build and cumulative selected edit request times.

|Prefix|Material tasks|Oracle mean|Mean accumulated seconds|
|---:|---:|---:|---:|
|1|6.25%|0.16098%|0.82663|
|2|10.4167%|0.41445%|0.93101|
|4|29.1667%|0.70396%|1.13335|
|8|29.1667%|0.77568%|1.54244|
|16|39.5833%|1.19548%|2.18763|

Full/local prefix values also match the saved table. Prefix16 uses only the 671 actual legal edits, not fictitious filled slots. These are oracle/query opportunity observations, not frozen-policy speed or equal-compute superiority over R07.

Independently recomputed mean complete request=0.141131841168s, p95 edit=0.199678300003s, p95 menu=0.221039465001s. The D forecast is `max(mean request,p95 edit)*22428 + p95 menu*14656 +1920 +3000 +22.641020055606 =12660.580331581s`. Including 14656 complete menu constructions avoids the invalid B sixteen-action-amortized menu estimate. The last setup component is the reported measured setup, not independently reconstructed from a separate setup ledger. One-edit B timing cannot certify actual four-step feature/Actor/optimizer/checkpoint throughput; C's real engineering measurement was not run.

The supervisor completed with worker exit0, root exit and entire private tree exit confirmed, no timeout, no surviving assigned process, and no deadline overshoot. The resource snapshot's stage wall is 540.0096s; this does not certify the yet-unfinished Git/closeout portion of the 1800s total. The parent must finish delivery before the absolute deadline. Actor/Critic updates, native, new teacher/scene/navigation and protected-set reads are recorded zero.

## Supported claim, missing evidence and routing

Supported: the exact fixed 48-task menu/pipeline was successfully exercised and contains useful edits on some inputs, but its registered opportunity is insufficient on both the mean-gain and actual joint-edit-share thresholds. No high-impact implementation issue was found that would invalidate this bounded opportunity conclusion in the reviewed sources/samples.

Not supported: entry to C, learned improvement, joint causal benefit, superior same-budget search, real robot execution, multi-step training throughput or new-parent generalization. The saved certificates explicitly do not certify physical joint trajectories.

Route: close this fixed R08-B neighbourhood, preserve all positive and negative examples, and report C/D/E as NOT_RUN_NOT_AUTHORIZED with null unexecuted outcomes. Do not enlarge menus/blocks/passes, change seeds/gates, create new training evidence or revive an old branch. Any future revised research proposal needs its own frozen protocol and actual authorization. This review recommends no immediate experiment.

## Follow-up alignment review

After the initial review, EXPERIMENT_REPORT.md and NEXT_STEP_DECISION.md became available and were read without repeating any numerical check or scientific call. Their task/action/gain counts, failed thresholds, query-prefix figures, complete-menu-inclusive D estimate, attribution limits and STOP at B agree with this review. The verdict is unchanged. `main_reports_available_at_review` is now true to record this follow-up.

Accounting clarification: lower.py imported `two_opt` directly, so patching the solver module inside the H instrument does not count that retained alias. The 2873 recorded H-internal two_opt calls plus 671 successful lower-solver calls give 3544 total calls. The total lower_s timing already covers the lower work, but its two_opt part was not separately timed; no new measured per-2-opt cost is claimed. The completed report states this correctly.
