# G2.1 causal controls: prepared, not evaluated

These controls consume the same action menu and mask as G2. They create no
environment, do not repair candidates, and do not inspect future trajectory
prefixes or teacher Q labels. The tests use synthetic dictionaries only. No
development scene, training scene or validation scene is executed by this work.

## Defined controls

`make_policy('detour_early', max_proposals=4, patience=1)` ranks legal non-STOP
actions by the existing cheap `detour_excess` feature, with the existing lower
action-index tie break. At root it proposes one action if one is legal. After
each committed proposal it reads G2's realized `quality_reward`, `work_penalty`,
`work_units`, `best_improved` and `route_changed`. A proposal resets stagnation
only when it changes the route, improves the certified best route, and its
realized quality reward exceeds its already-paid work penalty by more than
`1e-12`. Otherwise stagnation increases. It chooses STOP once patience or the
proposal cap is reached. It does not estimate any unexecuted proposal's return.
Accepted no-op and accepted non-best perturbation do not count as progress.

The rule's net-gain check includes the committed step's menu and proposal work
because those are included in G2's `work_penalty`. It excludes potential shaping
and future menu work. The environment still charges an explicit STOP normally.
This rule is not a guarantee of positive total return: the first query and the
query that discovers stagnation have already been paid for. Its parameters are
development candidates, not a claim of superiority over frozen Detour.

`make_policy('one_step', base_name='detour_memory')` allows the base policy at
root, then chooses STOP after at most one committed non-STOP action. Repeated
`select` calls do not consume this cap. `base_name='stop'` retains immediate STOP.
An injected `selector` with `select/update/state_dict/load_state_dict` can wrap
a frozen learned selector: STOP must remain legal at root, and the wrapper
does not call the selector again after its first proposal. A learned adapter
must use the normal already-built model observation; this module does not
construct graph observations or load/train any network. This adapter and its
end-to-end integration remain future work, rather than a completed experiment.

## Predeclared development grid and selection rule

Use only the existing 12 `rule_dev` scene definitions, and only after the
offline root-opportunity gates authorize further environment interactions.

| Candidate | max_proposals | patience |
| --- | ---: | ---: |
| E1 | 1 | 1 |
| E2 | 2 | 1 |
| E3 | 4 | 1 |
| E4 | 4 | 2 |

Hold the environment configuration, scene seed, shared menu, legal mask,
work accounting and final validation fixed. Select the largest mean shared
environment return over the 12 scenes. Break ties within `1e-12` by smaller
mean paid work, then smaller proposal cap, then smaller patience. Freeze the
chosen configuration before validation. Record all 48 runs and complete timing;
do not select per scene. Use the same selected parameters in every later paired
comparison and report fixed-work and wall-clock comparisons separately.

The immediate-STOP control remains separate. E1 and detour `one_step` should
produce the same shared-environment action sequence; this is a useful future
integration check, not evidence of performance at the present stage. A saved
future-best-prefix calculation is an oracle diagnostic and is not in this grid.

## Integration and synthetic checks

The interface matches G2 policies. A new G2.1 evaluator should explicitly import
`make_policy` from `rc3_rl_g21.baselines`; the frozen G2 evaluator hard-codes its
own factory and cannot accept these names without such an integration. Do not
monkey-patch or edit the frozen G2 module. The G2.1 factory delegates old names
to the original factory and does not change their implementations.

Run from the shared project root:

```powershell
.\rc3_learning\.venv\Scripts\python.exe -B -m unittest rc3_rl_g21.test_baselines -v
```

The suite covers causal feedback, cap/patience, mask/tie behavior, realized
net gain, no-op rejection as progress, STOP retention, one-step inference cap,
save/load, malformed input handling and factory compatibility. It makes zero
scene-generation, solver-prepare, environment-construction or step calls.

## Prepared evaluation integration (not an executed experiment)

`evaluate_g21.py` adapts the frozen G2 evaluation loop without modifying it.
It handles early detour, immediate STOP, original Fixed3/Detour, the frozen
initial BC, a gate-approved G2.1 checkpoint and a learned one-step control.
The learned one-step control lets the model choose STOP at root; after one
actual proposal it selects explicit STOP without another graph/model inference.
Per-case output retains STOP decisions, returned J, full reward ledger, paid
work, proposal count, full measured planning duration and complete plan evidence.

Before any scene preparation it requires: unchanged G2.1 protocol, final toy
receipt with passing numerical counts, root-written `opportunity_gate.json`,
final prescribed supervised checkpoint hash, matching final diagnostic/receipt,
and all four full-action opportunity thresholds. A toy checkpoint is forbidden
as deployment input. The gate attestation records `passed`, `protocol_sha256`,
`checkpoint_path`, `checkpoint_sha256`; optional but recommended receipt hashes
are `receipt_sha256`, `diagnostic_sha256`, `extra_diagnostic_sha256`.
The `final_diagnostic_sha256` alias is also accepted; if both diagnostic fields
are supplied they must agree. Those hashes are checked when supplied. The evaluator does not create this
authorization or automatically trigger environment work after offline training.

All mutable outputs and prepared inputs must lie under `rc3_rl_g21/outputs`.
The frozen G2 safe prepared-cache implementation is reused only with new G2.1
paths. Cold mode uses a fresh directory per case/method, records cache writing
inside total time, and rejects an existing prepared input. Warm mode prepares
inputs explicitly outside measured episodes and labels times as warm-input
load-plus-planning times. Original Fixed3/Detour retain the frozen uncached
problem-preparation boundary, so their anchor is not computed twice. Their
latency has no prepared-cache serialization overhead and is separately labelled;
cache-inclusive cold timings should not be advertised as unchanged native solver
latency. Warm shared-method timing must not be called cold end-to-end latency.

Once root explicitly authorizes interactions, example commands are:

```powershell
.\rc3_learning\.venv\Scripts\python.exe -B -m rc3_rl_g21.evaluate_g21 --select-early --output rc3_rl_g21/outputs/rule_development
.\rc3_learning\.venv\Scripts\python.exe -B -m rc3_rl_g21.evaluate_g21 --compare --output rc3_rl_g21/outputs/evaluation_bc
```

The first command evaluates all 4 configurations only on existing rule_dev12
and freezes one. The second compares all 7 methods on the same existing G2
validation24, reporting paired return versus STOP and paired J/work/time among
shared methods. These are reused development scenes, not independent new test
evidence. Original Fixed3/Detour receive no invented shared-work reward.

No command above has been run at preparation time. The integration checks in
`test_evaluate_g21.py` only create temporary synthetic JSON and mock the solver
and environment boundaries. They establish guard/helper behavior, not that a
complete scenario run or deployment timing has been validated.
