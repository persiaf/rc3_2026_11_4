# R07 independent result-to-claim review

Role: independent paper architect/reviewer under experiment-result-to-claim. Review date: 2026-11-13. Requested task: judge only whether the already executed R07 opportunity warrants continuing the current configured constructive learning branch. No new forward pass, planning, H, optimizer, teacher, old held-out reading, or Git operation was performed by this reviewer. The computational work here was CSV arithmetic and extraction of four existing archive members.

## Review prompt and evidence scope

Review PROTOCOL.json, RESULTS.json, STATE_PAIRS.csv, PARENT_OPPORTUNITIES.csv, REPRODUCIBILITY.json, TEACHER_GAPS.json, EXPERIMENT_REPORT.md and RESOURCE_ACCOUNTING.json in rc3_r07_20261113; inspect science.py and the F003:early / L136:early baseline–alternative archive pairs as needed. Independently aggregate the numbers, judge preset gates, inspect the intervention and continuation, assess runtime scope, and distinguish oracle opportunity from learning benefit. Save a verdict without launching supplementary experiments.

## Independent CSV aggregation

The CSV has 144 states from 48 distinct physical parents, three states per parent. All rows satisfy delta_J = alternative_J - baseline_J exactly at the stored floating-point precision. The maximum discrepancy in delta_J = delta_L + 0.75 delta_K is 2.85e-14.

| Scope | States / parents | Improved / tied / degraded | At least 0.5% improved | Mean delta J |
|---|---:|---:|---:|---:|
| Overall | 144 / 48 | 33 / 76 / 35 | 23 / 144 = 15.9722% | -0.13831475 |
| Full | 96 / 32 | 24 / 51 / 21 | 18 / 96 = 18.7500% | -0.18967860 |
| Local | 48 / 16 | 9 / 25 / 14 | 5 / 48 = 10.4167% | -0.03558707 |

I recomputed parent means directly from state rows and compared them with PARENT_OPPORTUNITIES.csv. The maximum oracle summary discrepancy is 3.47e-18. Parent-equal signed relative gain is -0.0135040%; parent-equal average max(0, relative gain) is 0.5217912%. Parents with at least one material improvement: full 12/32 and local 4/16. These agree with the supplied report. Reported stratified physical-parent bootstrap intervals are descriptive; their numerical resampling was not repeated in this bounded review. The signed interval [-0.4226%, 0.4556%] spans zero.

## Preset opportunity gates

| Gate | Observed | Required | Result |
|---|---:|---:|---|
| Overall material-improvement fraction | 15.9722% | at least 20% | FAIL |
| Full material-improvement fraction | 18.7500% | at least 15% | PASS |
| Local material-improvement fraction | 10.4167% | at least 15% | FAIL |
| Full parents with material improvement | 12 | at least 8 | PASS |
| Local parents with material improvement | 4 | at least 4 | PASS |
| Parent-equal oracle per-state gain | 0.5217912% | at least 0.5% | PASS |

The predeclared conjunction fails. Passing the oracle mean by 0.0218 percentage points does not override the failed incidence gates. The correct recorded result is OPPORTUNITY_INSUFFICIENT.

## Intervention, continuation and H

science.py restores the same saved prefix, checks a fingerprint containing problem identity, prefix, current node, remaining tasks, raw length and mask, and checks the same top-1/top-2 ranking. The only branch-dependent action is the first forced choice; subsequent actions use the same frozen SL greedy policy. The actor is in eval / no-grad mode, and its tensor digest is checked before and after each request. Different subsequent actions are expected consequences of different resulting states, not a different continuation policy.

I extracted the two selected baseline–alternative pairs by exact archive offsets, verified their raw SHA-256 values, and checked shared prefix, fingerprint and first mask, the raw-route prefix, and the first suffix action. All checks passed. Both branches call the identical finish(problem, raw, stats) terminal wrapper with no branch-dependent search argument. Each sampled request records one full H call, four two-opt calls and one joint-search call. This corroborates the common H call path and observed work counts. The underlying terminal source and the H identity-to-source hash mapping were not independently reopened within this review's read whitelist. The main thread separately supplied the dependency receipt and terminal.py SHA-256 6f4c334176f2d4fcf86e9b02eaaf4ce98c5fa5d8b8bee335e04ed6cea62785af; that is external corroboration, not an additional check performed here.

H processes the complete raw route. In the F003:early sample it changes the earlier prefix. Consequently these are offline full-route construction counterfactuals followed by H, not suffix-only values after a physically committed execution prefix. They are conditional costs under this frozen continuation and common H, not optimal Q values. REPRODUCIBILITY.json reports all 16 planned repeat requests identical in actions, masks, raw/H routes and metrics. science.py also requires every primary baseline to reproduce the saved SL raw/H route and costs.

## Runtime and resource interpretation

Independent timing aggregation gives pair mean/p95 0.349579 / 0.746889 seconds and alternative-only mean/p95 0.173519 / 0.371714 seconds. Mean summed H time is 0.012232 seconds per pair. These are resident-actor, cached-input request timings including prefix restoration, encoding, complete continuation, H, export/validation and archive. Input restoration and selection are separately recorded; cold model load and overall maintenance/review/Git are not represented by the per-pair speed number. Two-candidate choice requires two full continuations/H calls unless an exact baseline is reusable. No equal-cache deployment speed advantage is established.

RESOURCE_ACCOUNTING.json records 304 completed requests, zero failed, 304 H calls, zero actor/critic optimizer attempts and zero new teachers/scenes/navigation or held-out-set requests. Its 843.293-second wall value is the supplied execution snapshot, not independently established as the final all-inclusive delivery wall clock; the main thread must close that ledger through delivery. The review itself issued zero scientific requests.

## Claim judgment and routing

**claim_supported: no**, for the positive claim that the preset opportunity is sufficient to continue the current configured constructive learning branch. The negative decision is supported: this fixed training-layout/state/top-2 probe fails its preset continuation gate. It detects real conditional cost differences and a few beneficial alternatives, with heterogeneous local behavior. It does not establish that all constructive planning or all DRL approaches are impossible.

The oracle statistic chooses the favorable branch after observing both costs at each sampled state. It is an optimistic selection statistic over these candidate pairs, not a trained policy return, a demonstrated learnable selector, or additive gains along a single route. The three sampled prefixes cannot be combined as three simultaneously achievable improvements. Existing same-task teachers are better in 40/48 cases, tied in one and worse in seven, with mean relative gap 3.8139%; this contextual gap does not show that this top-2 intervention or a learned actor can recover it.

Suggested claim: "On 48 fixed training physical layouts and 144 predefined states, frozen-SL top-1/top-2 full-route counterfactuals under the same frozen continuation and H produced material improvements in 15.97% of states (local: 10.42%), below the preset overall/local gates. The configured branch therefore did not meet the criterion for further learning work."

Routing: **pivot at the design level / close this configured opportunity branch**. Do not launch PG, Q, multiseed training, new reward baselines or additional held-out evaluation from this review. Any future overall planning-decision redesign needs its own protocol, implementation review and explicit execution authorization. No extra experiment is necessary to decide the gate already executed.

Missing evidence for broader claims consists of a learned selector's benefit, independent-layout generalization, deployable suffix/online behavior and calibrated physical IK/joint trajectories; these are scope limits, not requests to run them now. Confidence is high for CSV arithmetic and the failed preset gate, and limited to the stated sample/protocol for broader interpretation. paper_claim_audit is unavailable in this bounded review, so downstream paper writing remains provisional. pending_external_review is false because this review was performed by the delegated independent reviewer role.
