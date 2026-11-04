# G2 training preflight review

The inspected CPU training implementation has no remaining blocking defect in the reviewed PPO, BC, rollout, or checkpoint contracts. This conclusion concerns implementation correctness before the registered single-seed run; it does not establish a performance advantage over the rules or BC controls.

Reviewed production files: `train_ppo.py`, `collector.py`, `worker.py`, and `checkpoints.py`, with the graph, GAE, terminal-event, and prepared-input interfaces checked as dependencies. This review did not change those production files. Exact inspected hashes and executed checks are recorded in `outputs/training_preflight_review.json`.

## Confirmed contracts

- Observations, actions, and old log probabilities are flattened in T-major order, matching the T × B values/rewards/terminal arrays and flattened GAE. A two-worker, three-step numerical probe checks the actual collector implementation against manually calculated advantages.
- Worker autoreset observations are used for the next episode, while a true terminal transition has zero next value. A collector cutoff preserves the live environment and bootstraps its value; it is not reported as environment termination.
- PPO uses freshly collected interactions and checks unchanged-parameter log probabilities before each update. Registered settings give 8 × 32 = 256 actual actions per rollout and 128 rollouts = 32,768 actions. STOP is a real sampled action; resets and pipeline probes do not add formal actions.
- BC normalizes over queried teacher evidence only. A test executes `bc_train` with a synthetic actor whose unknown-action logit is 1000: that action's gradient is exactly zero and its parameter remains unchanged. The teacher does not supervise the critic with its depth-two local value.
- Frozen BC, continued BC, and PPO start from the same initial checkpoint parameters. Continued BC and PPO create fresh optimizers. PPO does not use teacher samples as on-policy rollout data.
- CPU and CUDA RNG state are explicitly stored and restored. Checkpoint payloads load on CPU before model placement, avoiding accidental placement of CPU RNG byte tensors on CUDA. CPU RNG is tested with actual draws; the CUDA save/restore branch is tested with mocked state APIs to avoid opening another large CUDA context during CPU training.
- Checkpoints bind model architecture, protocol, parent BC identity, implementation files, and frozen solver dependencies. Nonfinite model/optimizer state is rejected. Resume rejects invalid or inconsistent action/iteration/update counters and a log that is ahead of the checkpoint.
- Resume restores live worker state and action memory. A reached validation boundary is reconciled on resume; saved validation must match its scenes, configuration, budget, and semantic model identity. Cumulative elapsed and validation time are retained.
- A proposal with unknown internal work is not admitted to learning as a cheap ordinary failure. The worker saves a failure context and stops the rollout.

## Executed evidence

`python -B -m unittest rc3_rl_g2.test_training_guards -v`: **8/8 passed** on the final reviewed source. These include the actual BC trainer path, source binding, identical checkpoint initialization, nonfinite state rejection, counter/log checks, CPU RNG draws, and the mocked CUDA state contract.

`python -B -m rc3_rl_g2.tools.training_audit_checks`: **3/3 passed** on the final reviewed source. This is a synthetic contract probe, not formal training or evaluation data. The collector probe gives GAE values `[[-9, -88.7], [27.25, -106], [15, 16]]` and preserves the final live observations at the rollout boundary.

The retained second pipeline run contains 256 MLP and 768 graph actions, 64 optimizer updates, finite advantages/gradients, unchanged log-probability error zero, parameter changes, and identical restored policy sampling. Its saved receipts report all checks passed. The earlier pipeline is separately retained in `outputs/pipeline_before_resume_guards`; the two runs incurred **2,048 real debugging interactions total**, excluded from the 32,768 formal actions.

The historical pipeline checkpoint source list is **not identical** to the final formal source list. After the second run, checkpoint dependency binding gained four frozen legacy modules, and saved-validation reuse gained additional input-identity checks. These changes do not alter the pipeline's executed environment, network, collector, or PPO update equations. The historical pipeline checkpoints are numerical evidence only and are not formal initialization checkpoints. The final source is additionally covered by the eight guard tests and the three numerical probes above; no identical-source full pipeline rerun is claimed.

The formal `outputs/training/bc_initial.pt` was independently read after the final binding changes: it has 400 BC updates, CPU device, the registered protocol digest, and a source identity exactly matching the final reviewed implementation.

## Issues corrected before this conclusion

The initial inspection identified broad wildcard source binding, incomplete CUDA RNG capture, insufficient resume-counter validation, potentially uncounted proposal failure work, missing resumed checkpoint validation, and lost cumulative time on resume. The root implementation added the relevant guards; this review reread those changes and executed the tests above. This document does not represent these corrected findings as still-open failures.

## Reporting and verification limits

- `worker` episode `complete_s` spans worker reset to episode completion and may include collector optimizer/validation waits between rollouts. It is an episode wall-clock diagnostic, not deployment planning latency. Use `evaluate_g2` complete-scene timing for deployment comparisons.
- The final rollout may leave live unfinished episodes. Their state is preserved in the final checkpoint; report completed episodes separately from these active episodes.
- The source still uses literal 256 in formal iteration accounting. This matches the frozen 8 × 32 protocol. Changing worker count or rollout length requires changing that accounting and rerunning the corresponding tests before a new protocol is used.
- Full formal training crash/restart injection has not been performed. The executed tests cover state/counter/RNG contracts and pipeline live-state restore, not every operating-system failure mode.
- CUDA RNG testing here is a mocked API-branch test, not a new hardware roundtrip. The selected formal training device is CPU.
- Numerical gates cannot establish final learned quality, stable superiority, or multi-seed generalization. Those require the registered held-out comparisons and their actual results.
