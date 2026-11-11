# R04-M2 implementation review

Verdict: **GO**, limited to the explicitly authorized offline M2 test. Execution permission is separately recorded in STAGE_AUTHORIZATION.json after exact source registration.

- Fixed research target remains large-parent DRL and frozen local complete-route planning. M2 only screens a training-period Critic reference. It does not replace the Actor or DRL with a table.
- Model: matched 268→128→128→3 ReLU residual nets, identical seeded tensors and zero final weight/bias. CG uses the fit-global component anchor; CB uses the fit-only size × observable remaining-fraction anchor. No target enters forward(features, window).
- Labels: reuse the exact saved restored268 artifact from real M1 attempt02. Validate its LFS byte hash, data/source/index/base identities and all116 trajectory/parent/feature identities. Do not replay trajectories, navigate, or generate labels. Labels L/K/H only estimate fit anchors or enter the fitting loss/evaluation.
- Link: L/K use stable inverse-softplus anchor plus residual, with pre-fixed1e-6 anchor/output floor. H remains signed. Value always composes the same normalized components with0.75 K. Zero-step and extreme-link tolerances are registered before computation.
- Optimizer: independent Adam, same M1 settings, clipping and first128 common batches; state_SUM per trajectory then episode_MEAN; no extra Value loss. Each attempted update is debited before backward, including failures. Actor optimizer0.
- Environment, reward, mask, safety, H, endpoints, native baselines and SL256/S1→S2 lineage are inherited unchanged, with no runtime planner/Actor request in M2. Exact original base artifacts are byte-checked; no encoder reconstruction is needed.
- Dataset: fixed33 fit and16 physically disjoint exposed holdout parents; no holdout table/scale/bias fitting, no resampling or missing-cell replacement. The exposed holdout supports development mechanism screening only.
- Endpoints: save0/128 only, exact model/Adam/RNG/cursor/table/resource roundtrip; old C1step128 is a verified saved export, no retraining. No intermediate model selection.
- Decision: preregistered Value ratio≤0.95 and paired95% upper<1 against BOTH own condition anchor and CG, both aggregations, plus component≤1.10 and per-size≤1 checks. CG is checked against its own global anchor; C1 superiority alone never establishes residual value.
- Resources:256 Critic attempts maximum, all planner/newscene/teacher/native/H and Actor counts0; active engineering+fitting+evaluation+audit wall≤1800s. Reuse the already accepted Windows Job supervisor and real Torch source classifier; do not rerun historical engineering or million-row audits. Old consumption remains separate and cumulative.
- Boundaries: M1step128 G3FAIL, step256NOT_RUN, R03 and oldM3 remain frozen. Original60, oldS3 and user Word documents are untouched. No automatic jointPPO, steps, features, seed, network or gate change.

The separate M2 binding does not overwrite the historical v2/v3 active design pointers. Scientific identity contains no mutable downstream authorization hash. The new source binding and protocol must be committed before numerical execution.
