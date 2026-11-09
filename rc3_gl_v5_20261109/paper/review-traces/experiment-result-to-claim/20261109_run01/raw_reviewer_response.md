# Raw independent reviewer response

Reviewer: delegated independent Codex paper architect/reviewer. Verdict: **claim_supported = no; confidence = high for this fixed single-seed development screen**. Route: **REVISE; retain the negative result and the fixed DRL research goal**.

I read the GL-v5 protocol, registration/input identity records, new objective/reference/mechanism/data/run source, engineering verification, final phase receipt, saved training curves, resource accounting,432 saved endpoint rows and their432 referenced new result JSONs. I used only saved-data JSON/CSV/NumPy arithmetic. I did not import the scientific project, load a model, run policy forwards/backward/optimizers, construct new tasks, or invoke navigation/native/H. Old S3 and original60 semantics were not read. I did not alter scientific artifacts or Git.

## Independent checks and interpretation

The registry has128 batches of4 inputs:256 full-scene and256 local inputs over96 physical layouts. Every selected development layout's map seed occurs in that training registry. Each of the12 development task seeds is absent from the registered training task seeds. There are12 new parent/task draws,72 case IDs,12 physical keys, exactly six mixed and six blocked physical layouts, and six size/endpoint cases per layout:4x3/6x6/8x8 crossed with closed/open endpoints. Each of the six methods has exactly72 saved rows.

This supports a test on new tasks on trained geometry. It does not support transfer to unseen physical layouts, robustness over optimization seeds, or an S3 conclusion. The deterministic physical-key selection rule and pre-optimizer bank freeze are recorded; I did not re-read historical B semantic files to re-execute the selection process. The separate numerical/provenance audit is responsible for its broader cross-file replay.

I reproduced every raw and H-final parent-layout mean comparison directly from the432 rows. For the six primary SL/B/C comparisons, the stored20000-replicate confidence limits were reproduced exactly by resetting PCG64(seed11642026) per comparison, drawing the mixed (20000,6) index block followed by blocked (20000,6), concatenating the sampled layout gains, averaging each12-layout replicate, and taking linear2.5/97.5 percentiles. A preliminary mathematically equivalent interleaved stream order gave small Monte Carlo differences; the registered block order resolves them exactly. Neither calculation changes the failed gate.

Define gain=(mean-control-cost minus mean-G-cost)/mean-control-cost within each physical layout before averaging the12 layout gains. Positive means G is better.

| Comparator | Raw gain | Raw95% interval | H-final gain | H-final95% interval |
|---|---:|---:|---:|---:|
| SL256 | -3.2267955% | [-5.5680816%, -0.5000452%] | -2.8641351% | [-4.6180194%, -1.2430283%] |
| B128 | -2.5805358% | [-3.8274202%, -1.4060759%] | -2.4678596% | [-4.1868361%, -0.9169384%] |
| C128 | -5.0590604% | [-7.0656972%, -3.2329255%] | -3.2426334% | [-4.9424995%, -1.7734140%] |

All three H-final means and entire intervals are negative. G wins on only3/12 layouts against SL,2/12 against B, and0/12 against C after H. The predeclared AND criterion therefore fails against every primary control. This is stronger than an inconclusive near-zero result for this selected development bank, while the bootstrap intervals remain conditional on a single trained actor and12 selected layouts.

The H-final relative gains against the native complete-route pipelines are also negative: Fixed3 -5.2272952%, Detour16 -6.6059809%. These comparisons do not rescue the candidate. Fixed3 and Detour16 use their registered endpoint pipelines; they must not receive an extra common H pass.

G does not improve the SL initialization, and does not beat B128 or C128, in the registered parent-averaged raw or H-final comparisons. H does improve G's own raw routes: the mean raw cost151.8736448752 falls to126.0017051553, an absolute mean reduction25.8719397199. That is evidence for the registered H postprocessing effect on these G routes, not evidence that G learned a better policy than the controls. Comparing G+H only against raw controls would not answer the preregistered question.

All432 referenced result artifacts have valid=true; all432 CSV artifact hashes match the saved JSON bytes; all reported cost components are finite and H-final J does not exceed raw J. Neural identities have a single explicit study_method, checkpoint_role, actor hash, tensor hash and fixed update per role. SL is fixed at256; B/C/G are fixed at128. G's COMPLETE tensor hash matches its evaluation role. The G branch receipt records the exact SL tensor hash1817f58aa6506456ff12f0c5fe0a914e156aef8af319f4687dde90782f55a63e, fresh Adam and zero inherited state entries. The engineering receipt is bound to SOURCE_BINDING SHA256 0f5e381a80790d531c03c9fd4452c755ab728712bf32ee59317135c42351ddee. These are consistency checks, not new execution or proof of real-arm calibration/IK validity.

## Computation and diagnostics

The final phase receipt records COMPLETE, no error,989.3444799 seconds within the7200-second cap, including22.5167148 registration seconds. There were128 true optimizer updates,1536 sampled terminal requests,512 frozen-reference requests and432 evaluation requests:2480 total. Relative to B's1536 sampled training terminal requests, the512 references add33.3333% training terminal requests, bringing that part to2048. This is a request-count increase, not an assertion of33.3333% more elapsed time or FLOPs.

The reference account records512 separate frozen model loads, fresh encodings and greedy constructions;486 actual full-H executions and26 reference-only terminal-cache hits. The sample account records1536 actual full-H executions and zero terminal-cache hits. Reference subcomponent clocks are3.7899901 seconds loading/verifying,7.6565062 encoding,32.5462068 greedy construction and38.6735359 reference-H/validation. The resource accounting warns against summing nested clocks. G's recorded internal update clock totals569.3650876 seconds versus historical B's482.0828944; the87.2821932-second difference is descriptive and is not a paired deployment latency or controlled speed claim. The outer989.3444799 seconds remains the inclusive science budget measure. Read-only review/publication is maintenance, not extra scientific training.

The saved128-row training curve contains updates1..128 and clipping in all128, with pre-clip gradient norms ranging66.50137..4425.47069. The saved advantage convention is reference_J minus sample_J;1151 are negative,367 positive,18 zero. Mean advantage=-8.4859183 and mean sample J231.9462967 exceeds mean reference J223.4603784. The source keeps every sign and all first-action log probabilities. Loss/gradient existence, successful parameter updates, analytic toy-gradient PASS and the score-function identity do not establish greedy route improvement.

Clipping at every update and strong scene-size differences in gradient scale are diagnostic observations. The experiment does not identify whether clipping, variance, the baseline, step size, multi-size weighting, or another interaction caused degradation. It does not establish lower variance than LOO from a matched variance measurement. The state-only frozen baseline leaves the expected score objective unchanged under its mathematical assumptions, while finite samples, Adam and clipping do not guarantee monotonic greedy quality. A single failed baseline replacement does not refute DRL generally.

## Claim and routing

Suggested supported statement: In a single seed11622026 run from the same SL256 initialization, replacing detached LOO with a frozen same-input SL256 greedy+H reference for128 updates completed within budget but failed the preregistered development screen. Across12 already-trained physical layouts and72 new size/endpoint views, G's H-final cost was worse than SL256, B128 and C128 by parent-averaged2.8641%,2.4679% and3.2426%; all corresponding bootstrap gain intervals were strictly negative. The added reference consumed512 requests (+33.33% relative to sampled training terminal requests).

Direct independent-multiple-seed confirmation of this G recipe is not warranted as a successful-candidate follow-up: all primary gates fail. Retain the negative result; keep the original DRL research goal; do not auto-expand updates, alter seeds, replace cases, open held-out evidence, or route to confirmed paper claims/ablation studies. Complete the already-running read-only numerical audit and reconcile any discrepancy. Further scientific work would require a separate, explicitly registered and authorized mechanism revision with its own bounded budget and preserved lineage. The current evidence cannot determine such a revision's outcome.

At this review's issuance, rc3_gl_v5_20261109/INDEPENDENT_AUDIT.json does not yet exist. The separate exhaustive numerical/provenance audit is still running; I do not mark it PASS or imply that my checks replace it. paper/PAPER_CLAIM_AUDIT.json and PAPER_PLAN.md are absent. The writing verdict is provisional because there is no paper-to-evidence audit, not because reviewer delegation was unavailable. pending_external_review=false denotes that an independent delegated Codex reviewer performed this review; it does not represent external human peer review.
