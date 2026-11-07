# Saved-data analysis contract

`analyze_s2.py`, `plot_s2.py`, `paper_plot_style.py`, `make_figures.py`, and the nine `gen_fig*.py` files are analysis-only additions. They are outside the approved execution source manifest. They do not change the runtime, protocol, reward, mask, training distribution, budget, or stage authorization.

The analyzer imports Python standard-library modules only. It never imports the research package or PyTorch, loads `.pt` tensors, constructs a model or optimizer, runs inference, generates scenes/navigation/trajectories, calls a teacher/H, or reads S3/old60 tasks. Plotting imports Matplotlib and reads the derived CSV files only. Analysis and plotting are postprocessing, not new scientific training/evaluation or optimizer engineering acceptance.

Run only after the scientific execution has stopped:

```powershell
.\rc3_learning\.venv\Scripts\python.exe -B rc3_s2_results_20261107/analyze_s2.py
.\rc3_learning\.venv\Scripts\python.exe -B rc3_s2_results_20261107/make_figures.py
```

The default output is `analysis/`. Existing nonempty derived directories are refused. Use a new `--out` path inside this report directory for a separately labelled revision, and pass the same directory with `make_figures.py --analysis`. The script refuses a started but unclosed stage by default. `--allow-open-snapshot` is an explicit diagnostic option; even then, all input files and their inventory must remain stable throughout the read, and the seed is labelled `OPEN_SNAPSHOT`. Missing seeds stay `NOT_EXECUTED`; failed or interrupted runs are never promoted to 512 updates.

All JSON and JSONL input bytes are SHA-256 hashed and recorded in `input_manifest.csv`. Per-file size/mtime checks and a final inventory check reject concurrently changing evidence. JSON parsing uses Python's exact integers. `sampling_seed_decimal` is exported as the exact decimal string and must not be reimported through JavaScript numbers or a spreadsheet's floating point numeric format.

The output schema contains:

- `updates.csv`: only completed update records, separate `RL` and `continued_SL` phases; losses, wall time, pre/post clipping gradient norms, per-update reward/cost/H-gain/advantage/entropy summaries.
- `trajectories.csv`: all three recorded starts per completed RL instance, exact seed strings, raw and H costs, LOO baselines/advantages, entropies, cache flags, component timing and metric-source references.
- `instances.csv`, `gradient_norms.csv`: per-instance summaries and every named parameter's recorded pre-clipping norm.
- `terminal_attempts.csv`: charged starts, completions, errors, pending attempts, saved plan metrics and validation results. A saved JSON file can precede a failed clock check, so file presence is kept separate from completed status.
- `group_summary.csv`: observed distributions by seed, window (full10x10/local4x3/local6x6/local8x8), obstacle mode, endpoint condition, start role, and their combinations. These are trajectory distributions, not independent observations or confidence intervals.
- `events_summary.csv`, `checkpoint_boundaries.csv`, `development_feasibility.csv`: saved event counts and checkpoint JSON metadata; only case counts/feasibility for the fixed development checks, with no development quality ranking or checkpoint selection.
- `analysis_checks.csv`: static arithmetic, metric joins, exact-integer seeds, contiguous completed logs, saved lineage and event/ledger reconciliation. This is a postprocessing consistency check, not a substitute for independent scientific review.
- `ANALYSIS_SUMMARY.json`: a `per_seed` dictionary exposes status, actual RL/continued-SL update counts, cumulative shared stage wall/ledger, observed raw/H J/L/K and H gain, failure/invalid/pending counts, phase-separated clipping and objective diagnostics, advantage/entropy distributions, and first32/last32 costs separately by full/local groups. A `seeds` list contains the same records for simple iteration.

The preferred L/K source joins a trajectory to an already saved terminal using exact `(problem identity, raw_route)` and matching `raw_J`/`J_H`; the final route is cross-checked. If no saved match exists, the output explicitly labels the static algebra `K = len(route)-2`, `L = J-0.75*K`. This calculation calls no geometry, planner or validator. Repeated identical routes may match several attempts; a metric reference does not identify chronological ownership. `not_assigned_by_metric_join` is therefore a noncausal label. Aggregate work outside completed updates is counted from RL completed terminal events minus noncache completed-log trajectories, plus event cache hits minus logged cache-hit trajectories, with cumulative optimizer-attempt/update differences reported separately.

Terminal requests, attempts, completions, cache hits, errors and pending work remain distinct. Requests charged before a clock check may stop before an attempt/cache event; that difference is exposed from the ledger. RL and continued-SL share a cumulative clock/ledger, so their resource contexts are never summed. Development attempts are reported separately from RL attempts.

The curves describe changing on-policy training inputs. First32/last32 summaries compare different observed tasks and are not paired improvement tests. `raw_J-J_H` measures H's postprocessing on those routes; it is not independent network gain. CE and REINFORCE losses have different meanings/scales. These artifacts do not establish deployment/generalization, baseline superiority, or a DRL-specific gain. Saved proxy reachability/roadmap validation is not physical IK or joint-trajectory certification.
