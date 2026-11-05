# Independent R0 result-to-claim judgment

`claim_supported: yes` — **strictly for the registered bounded R0 opportunity claim**. Confidence is high for this saved-record result. Paper-level writing remains **provisional — no paper claim audit available**.

I independently read the registration, amendment, source, ledger, evaluations, checkpoints, case results, verification, analysis, resource account and report. I ran only `review_numeric.py`, which reads saved records and hashes and performs arithmetic plus the registered scalar bootstrap. I did not import project code, generate/prepare scenes, run a completion/H/native planner, issue a geometry certificate, forward a model or update an optimizer.

## Numerical gate

| Registered condition | Independent result | Judgment |
| --- | ---: | --- |
| All cases in primary denominator | 32/32; regular/mixed/blocked = 11/11/10 | Pass |
| Mean scene-paired gain vs root g_B+H >= 0.5% | 2.250654204783714% | Pass |
| Positive absolute J difference > 1e-7 in >= 12 cases | 32/32 | Pass |
| Largest share of positive relative gains <= 25% | 8.944762348022874% | Pass |
| All final plans feasible | All 32 cases; stored verification passed and 96 plan byte hashes matched | Pass |

The registered exploratory scene bootstrap is reproduced: mean 2.250654204783714%, 95% interval [1.7990165472075776%, 2.734123017711682%]. It is a descriptive bootstrap of these 32 development scenes, not evidence of learned-policy generalization, independent final-test performance or training-seed robustness. Neither 256 checkpoints nor 2048 labels are independent scene samples.

Mean J is 284.545602725675 for native Fixed3,284.2691595927062 for root g_B+H and 277.8744753213 for the bounded teacher. The teacher's mean paired improvement against native Fixed3 is 2.3424715302003327%; root's paired change against native is 0.09364809460546068%. These are distinct comparators. The primary gate uses root g_B+H throughout.

## Protocol and resource integrity

The source hashes match the amended frozen protocol. Full H is present in static code: original H0 has two normalize+two_opt(passes=60) rounds and final normalize, followed by joint_search(passes=2) and final normalize. Native Fixed3 invokes its three frozen recipes; it is not merely one H call and is not substituted for root g_B+H in the gate.

The ledger contains 2144 paired starts/completions within the 2304 cap:32 native,32 root,2016 actual query calls,32 teacher finalizations and 32 uncached root repeats. There are 2112 actual full-H evaluation records. The 2048 candidate-label uses include 32 exact root-cache hits;2048−32=2016 actual query calls. The 2144 full-H result uses (candidate uses plus 32 roots,32 teacher finalizations and 32 audits) minus 32 cache hits equal 2112 actual full-H calls. These two occurrences of 2144 have different meanings and should retain their labels.

Recorded actual function counts match: full H has 8100 two_opt calls and 2112 joint_search calls, including nested two_opt invocations once each. Native Fixed3 has 96 solve_route,96 construct,288 two_opt and 96 joint_search calls. These counts are function invocations, not iterations, FLOPs, or an equal-search-effort comparison.

All 256 saved checkpoint selections match the registered tolerance and tie preference. Every candidate's cost and route agrees with its saved evaluation key; cached query values trace to the corresponding root. The maximum saved reference-continuity discrepancy and terminal-continuity discrepancy are both exactly 0. The 32 uncached root repeats match both raw and final routes. The stored numerical verification covers 2112 evaluations and 173740 raw service transitions;96 plan hashes match the stored scalar-certificate list. I consumed those existing geometry-validation results rather than re-certifying geometry.

The calibration wall formula independently yields 180 seconds, matches the protocol receipts and remains unchanged. Main execution is 25.96944760000042 seconds; calibration is separately 1.0035623000003397 seconds. This is an offline teacher computation, not a neural deployment benchmark.

## Amendment and provenance

The canonical sorted-TSV inventory digest is`8631c20f068b709c9a64201bd4526a619667c611e7b0796affb7eb38a36aa536`; the raw JSONL digest is`2ac9effcc4b840906ef617c09ec067d6463f03e0d29dce16b4a08fc08b64a6a5`. They are different byte representations. The original code indeed compared these incompatible representations before entering calibration, before writing its start receipt or making a scene/planner call. The archived original source/protocol hashes and current amended hashes match the amendment. The source diff changes only inventory interpretation and boundary-receipt checks; the original and amended protocols retain identical seeds, H, teacher, gate and budgets. The current protocol also identifies the amendment. This is an honestly recorded pre-run correction, not a hidden post-result method change. A local ledger and static ordering support this provenance; they do not constitute an external forensic execution history.

The seed audit is appropriately phrased as no recorded prior use in tracked text/protocol/gzip logs and listed historical banks, with a separate filename/boundary receipt. It explicitly excludes arbitrary binary/archive internals and unrecorded external jobs. I find no justification for strengthening this statement to universal proof that every historical binary contains no such integer.

No numerical or protocol blocker was found.

## What is supported

On exactly these 32 registered development cases, the fixed bounded query teacher retains distributed, measurable improvements relative to root g_B+H after the entire shared postprocessor.120/256 checkpoints have a strict improvement over the reference within the queried candidate set. The immediate gain/distance rule is worse than the best queried final-H cost at 114 checkpoints, with mean candidate-set regret 1.0155297574851594 J. Raw-cost minimization is worse after H at 108 checkpoints, with mean final-cost regret 0.7269231244085962 J. These are evidence that candidate ranking can matter after H in this finite query protocol.

## What is not supported

- That a finite network can learn these decisions or generalize to new scenes or a reserved test bank.
- That navigation/D information is necessary, indispensable, or absent from another observation representation. An oracle ranking difference does not establish an information-theoretic witness.
- That policy gradients, reinforcement learning, a new architecture or a learned representation contribute any gain. There were zero neural forwards and optimizer updates.
- That the teacher is optimal over all legal actions, or that its candidate-limited regrets measure all-action regret.
- That the teacher's additional completion search is an efficient deployment method, beats Detour16, or establishes real robot/IK reliability.
- That this result reopens the already closed same-structure pure policy-gradient branch or authorizes R1/R2/R3.

## Revised claim and routing

Use: “On the 32 pre-registered development scenes, a bounded deterministic teacher using up to 8 queried actions at up to 8 actual-prefix checkpoints improves mean scene-paired final cost by 2.2507% relative to root g_B+H; all 32 scenes improve, and the largest scene contributes 8.9448% of total positive relative gain. This establishes a candidate-limited opportunity signal after full shared postprocessing.”

Confirm and archive **only this R0 claim**. No additional experiment is needed to decide the registered R0 gate. No automatic R1, ablation planning, budget expansion, held-out evaluation or reopening of the old policy-gradient route follows. If a future user task authorizes broader research, learning/generalization needs a separate protocol and learned-versus-appropriate-baseline evidence; D-necessity needs a separate information-gap design; RL benefit needs controlled learning comparisons. Those are unmet requirements for other claims, not actions authorized by this review.

The parent executor should record this narrow claim in project findings/tracker and preserve the trace. A paper claim audit is unavailable, so downstream manuscript-level confirmation stays provisional. The delegated independent result review itself is complete and is not `[pending external review]`.
