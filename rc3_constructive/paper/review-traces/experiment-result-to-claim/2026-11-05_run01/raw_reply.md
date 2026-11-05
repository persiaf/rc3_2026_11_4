# Independent reviewer response: C2.1

**No blocking factual or arithmetic mismatch found.** The report correctly separates the large deterministic H1 contribution from the smaller, uncertain learning increment. The decision to stop at bounded attribution and retain Fixed3／Detour as defaults is consistent with the evidence.

## Audit scope and outcome

I read the C2.1 execution code, fixed joint_search/two_opt implementation, protocol, rows, analysis, resources, existing verification and original final-evaluation records. My independent audit imports only NumPy and standard-library modules, never project planner/model/validator modules. It recomputed all H0/H1 means, each delta and its bootstrap interval, learning comparisons, timing summaries, candidate overlaps and scene-mode summaries; verified original-route identity, numeric plan fields, 134 frozen input hashes, 72 output plan hashes, and the ordered 72 start/72 completion ledger. All checks passed. The numeric audit did not rerun geometric validation or H1; it checked saved certificates and hashes. New planning calls, neural calls, optimizer updates, test queries and geometry certificates from this review are all zero.

The first invocation of the arithmetic audit script failed at reading a path because its initial repository-parent calculation was off by one. The path was corrected; the second invocation passed. That failure occurred before any numerical results or research calls and introduced no additional experiment or certificate.

The shared postprocessor is genuinely the same for all three sources: one call to the frozen joint_search with passes=2, then the common normalization; internal 2-opt has at most 60 passes. Original candidate rules and deterministic tie-breaking are retained. Cyclic source order gives the three sources balanced order positions. H0 is read from saved outputs and all old repeated routes are consistent. Full Fixed3 retains its complete original capability. This is a fair construction-source comparison conditional on fixed H1, while it is not an equal-total-work comparison against Fixed3 and cannot by itself establish deployment efficiency.

## Claim A: fixed common H1 closes much of the observed quality gap

- **claim_supported: yes**, restricted to the existing frozen routes and 24 development scenes.
- **confidence: high** for this descriptive claim.
- All 72 source/scene routes improve under H1. The prior, initialization and RL mean gains are 16.6874, 17.0207 and 16.9018 J respectively. Similar magnitudes for all sources show the added deterministic pipeline capability has substantial value here.
- RL mean J falls from 290.8602889981 to 273.9584714056. Full Fixed3 remains 272.3429627863. The mean absolute gap falls from 18.5173262118 to 1.6155086193; 91.2756917451% of that particular gap is removed.
- The 91.3% figure is descriptive gap shrinkage, not a decomposition of causal root causes. H1 jointly changes point selection, deletion and route order; no one component can claim the entire gain.
- Remaining RL−Fixed3 difference is exactly ΔL=+2.5842586193 plus 0.75ΔK=−0.96875, giving ΔJ=+1.6155086193. RL uses fewer stops on average but travels further.
- This does not establish global optimality, an isolated selection/order mechanism, all-scenario generalization, or stable learning benefit.

Suggested wording: “On the 24 reused development scenes, a fixed common deterministic postprocessor markedly reduced the cost of all three frozen construction sources and removed about 91.3% of the RL pipeline's mean absolute cost gap to full Fixed3. This is conditional pipeline evidence and does not identify independent subcomponent causes.”

## Claim B: stable independent deployment advantage of learned construction

- **claim_supported: no**.
- **confidence: high** that this claim is not established by current data, not that it is impossible for future methods.
- With H1 held common, RL−prior improvement is +4.4382016425 J (95% interval −0.4252276646 to +9.3661029579), or +1.3397079747% (−0.4115622573% to +3.1155507351%). RL wins only 13 of 24 paired scenes.
- Against initialization, RL's improvement is +3.7444516425 J (−1.0789470300 to +8.6796315295), or +1.0900229689% (−0.6192651731% to +2.8727969669%), with 12 wins and 12 losses.
- RL＋H1 is still worse than Fixed3 in mean cost; mean scene-relative degradation is +0.8170352258%, with interval −0.3036072767% to +1.9974188708% and 11 wins/13 losses. A confidence interval crossing zero cannot certify equivalence, noninferiority or a deployment gate.
- These are reused development scenes involved in rollout baseline decisions and one selected training seed. Bootstrap intervals describe scene variation conditional on the frozen selected model. They do not cover training instability or replace independent confirmation.
- The approximately 2.9 ms median added H1 time is a single-call incremental measurement. It cannot be added to an old complete-pipeline median to construct a new complete-time result, and no new quality–time frontier is established.

The narrower claim “there is a limited mean learning signal conditional on H1” is **partial**, confidence medium, because the paired intervals remain compatible with harm. Suggested wording: “Common H1 preserves a modest average advantage of the trained constructor over prior and initialization, but this advantage is inconsistent across scenes and remains unconfirmed.”

## Missing evidence and routing

Missing evidence includes frozen-method independent confirmation, training-seed repeatability, complete same-machine pipeline timing, and controlled causal component/target tests. Navigation encoding is not established as the primary deficit: remaining type-specific gaps do not rise monotonically with obstacle complexity, and the slices are exploratory.

No further experiment is necessary to accept the bounded descriptive claim A. For claim B, narrow the current claim and stop this capped stage. Do not run C3, retained tests or more training within this task. A future separately authorized raw-vs-post training-target comparison could be a focused mechanism test if network, initialization, task stream, optimizer and deployment H are controlled; it is not assured to improve greedy inference or deployment outcomes.

Record this result and the residual uncertainty in experiment notes. Retain all original evidence and exact frozen weights. Because paper/PAPER_CLAIM_AUDIT.json is absent, downstream paper-writing confirmation remains **provisional — no paper claim audit available**. No algorithmic-superiority claim should be routed to confirmed paper writing.
