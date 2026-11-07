"""Saved-data-only S2 analysis; never imports the research runtime or torch.

Run from the repository root after scientific processes have stopped:
  .\rc3_learning\.venv\Scripts\python.exe -B rc3_s2_results_20261107/analyze_s2.py

Only persisted S2 train/development logs, terminal JSONs, checkpoint JSON
sidecars and S2 admission receipts are read. No model or optimizer is created,
no tensor checkpoint is loaded, and no plan, navigation or query is generated.
All input bytes are hashed with stable pre/post stat checks. Python json/int
preserves the exact recorded 64-bit sampling seeds; CSV seeds are decimal text.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

SEEDS = (11622026, 11622027, 11622028)
PHASES = ("RL", "continued_SL")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01"
EXPECTED_PROTOCOL = "0c5bc7d7a9557edf2af76f51a9f418550f0b4ce1ec6a8b47ddea95c6256e78bb"
EXPECTED_SOURCE = "0f0ffd2d2dda2fefefcfcb5cb26dc5197b5d11bd1332b0a5546992409049bfe0"
STOP_PENALTY = 0.75


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def q(values, p):
    v = sorted(values)
    if not v:
        return None
    k = (len(v) - 1) * p
    a = int(k)
    return v[a] + (v[min(a + 1, len(v) - 1)] - v[a]) * (k - a)


def dist(values, prefix=""):
    v = [float(x) for x in values if finite(x)]
    out = {"n": len(v), "mean": statistics.fmean(v) if v else None,
           "std_population": statistics.pstdev(v) if v else None,
           "min": min(v) if v else None, "p05": q(v, .05),
           "p25": q(v, .25), "median": q(v, .5), "p75": q(v, .75),
           "p95": q(v, .95), "max": max(v) if v else None,
           "exact_zero_n": sum(x == 0 for x in v),
           "abs_le_1e_minus12_n": sum(abs(x) <= 1e-12 for x in v)}
    return {prefix + k: value for k, value in out.items()}


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


class Inputs:
    def __init__(self):
        self.manifest = []

    @staticmethod
    def stamp(path):
        s = path.stat()
        return (s.st_size, s.st_mtime_ns)

    def label(self, path):
        try:
            return path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            return str(path.resolve())

    def json(self, path):
        before = self.stamp(path)
        data = path.read_bytes()
        if self.stamp(path) != before:
            raise RuntimeError(f"Input changed during read: {path}")
        self.manifest.append(dict(path=self.label(path), bytes=len(data),
                                  sha256=hashlib.sha256(data).hexdigest(), mtime_ns=before[1]))
        return json.loads(data.decode("utf-8-sig"))

    def jsonl(self, path):
        if not path.exists():
            return
        before = self.stamp(path)
        h = hashlib.sha256()
        count = 0
        with path.open("rb") as f:
            for number, line in enumerate(f, 1):
                h.update(line)
                if line.strip():
                    try:
                        row = json.loads(line.decode("utf-8-sig"))
                    except Exception as exc:
                        raise RuntimeError(f"Unreadable saved JSONL {path}:{number}") from exc
                    count += 1
                    yield number, row
        if self.stamp(path) != before:
            raise RuntimeError(f"Input changed during read: {path}")
        self.manifest.append(dict(path=self.label(path), bytes=before[0],
                                  sha256=h.hexdigest(), records=count, mtime_ns=before[1]))


def write_csv(path, rows, minimum_fields=()):
    fields = list(minimum_fields)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: dumps(v) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})


def same(a, b, tol=1e-7):
    return finite(a) and finite(b) and abs(a - b) <= tol


def main():
    analysis_started = time.perf_counter()
    analysis_cpu_started = time.process_time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "analysis")
    parser.add_argument("--allow-open-snapshot", action="store_true",
                        help="Explicitly label started but unclosed seeds as snapshots; stable reads still required")
    args = parser.parse_args()
    run = args.run.resolve()
    out = args.out.resolve()
    report_root = Path(__file__).resolve().parent
    if not out.is_relative_to(report_root):
        raise ValueError("Derived outputs must remain inside the new S2 report directory")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose a new output directory; existing derived evidence is not overwritten")
    inp = Inputs()
    # Compare the inventory again before producing outputs so a newly started
    # stage or appended journal cannot silently become a final closed report.
    def inventory():
        paths = []
        for seed in SEEDS:
            for folder in (run / f"stage_admission/S2/{seed}", run / f"seed_{seed}/S2",
                           run / f"seed_{seed}/RL", run / f"seed_{seed}/continued_SL"):
                if folder.exists():
                    paths.extend(p for p in folder.rglob("*") if p.is_file())
        return {str(p): inp.stamp(p) for p in paths}
    initial_inventory = inventory()
    records = []
    for seed in SEEDS:
        admission = run / f"stage_admission/S2/{seed}"
        receipt = None
        status = "NOT_EXECUTED"
        for name, label in (("COMPLETE.json", "COMPLETE"), ("FAILED.json", "FAILED_INCOMPLETE")):
            if (admission / name).exists():
                if receipt is not None:
                    raise ValueError(f"Conflicting final receipts for {seed}")
                receipt = inp.json(admission / name)
                status = label
        if receipt is None and (admission / "START.json").exists():
            interruptions = sorted(admission.glob("INTERRUPTED_*.json"))
            if interruptions:
                receipt = inp.json(interruptions[-1])
                status = "INTERRUPTED_SAFE_UPDATE_BOUNDARY"
            else:
                status = "OPEN_SNAPSHOT"
                if not args.allow_open_snapshot:
                    raise RuntimeError(f"S2 seed {seed} is started but not closed; defer analysis until stopped")
        records.append((seed, status, receipt, admission))
    updates, trajectories, instances, norms, attempts, events_summary = [], [], [], [], [], []
    dev, checkpoints, seed_summary, checks = [], [], [], []
    all_terminal_count = 0
    for seed, status, receipt, admission in records:
        base = run / f"seed_{seed}"
        if (admission / "START.json").exists():
            start = inp.json(admission / "START.json")
            binding = start.get("science_binding", {})
            checks.append(dict(seed=seed, check="protocol_binding", passed=binding.get("protocol_sha256") == EXPECTED_PROTOCOL))
            checks.append(dict(seed=seed, check="source_binding", passed=binding.get("source_manifest_sha256") == EXPECTED_SOURCE))
        final = inp.json(base / "S2/FINAL.json") if (base / "S2/FINAL.json").exists() else None
        fork = inp.json(base / "S2/FORK_RECEIPT.json") if (base / "S2/FORK_RECEIPT.json").exists() else None
        by_key = {}
        attempt_map = {}
        view_specs = {}
        event_counts = collections.Counter()
        terminal_role_event_counts = collections.Counter()
        for phase in PHASES:
            folder = base / phase
            for line_no, row in inp.jsonl(folder / "events.jsonl"):
                event = row.get("event")
                event_counts[f"{phase}:{event}"] += 1
                if event == "view":
                    view_specs[row["identity"]] = row["spec"]
                role = row.get("role")
                if event and event.startswith("terminal_"):
                    terminal_role_event_counts[(role, event)] += 1
                if event == "terminal_cache_hit":
                    continue
                if event in ("terminal_start", "terminal_complete", "terminal_error"):
                    key = (role, row["ordinal"])
                    entry = attempt_map.setdefault(key, dict(seed=seed, phase_log=phase, role=role,
                                                              ordinal=row["ordinal"], started=False,
                                                              completed=False, failed=False))
                    if event == "terminal_start":
                        entry.update(started=True, problem=row["problem"], raw_route=row["raw_route"],
                                     source_event_line=line_no)
                    elif event == "terminal_complete":
                        entry.update(completed=True, wall_s=row.get("wall_s"),
                                     completed_J=row.get("J"), completed_L=row.get("L"), completed_K=row.get("K"))
                    else:
                        entry.update(failed=True, wall_s=row.get("wall_s"), error=row.get("error"), message=row.get("message"))
            # A file can exist even if a later clock check marked its attempt failed.
            # Its presence is therefore recorded separately from completion.
            for file in sorted((folder / "plans").glob("*.json")) if (folder / "plans").exists() else ():
                if not (file.name.startswith("RL_") or file.name.startswith("development_")):
                    raise ValueError(f"Unexpected S2 terminal file: {file}")
                raw = inp.json(file)
                r, problem = raw["result"], raw["problem"]
                role, ordinal = file.stem.rsplit("_", 1)
                entry = attempt_map.setdefault((role, int(ordinal)), dict(seed=seed, phase_log=phase,
                    role=role, ordinal=int(ordinal), started=False, completed=False, failed=False))
                validation = r.get("plan", {}).get("validation", {})
                spec = view_specs.get(problem["identity"], {})
                entry.update(terminal_file=inp.label(file), problem=problem["identity"],
                    raw_route=r["raw_route"], route=r["route"], J=r["J"], L=r["L"], K=r["K"],
                    raw_J=r["raw_J"], raw_L=r["raw_L"], raw_K=r["raw_K"],
                    H_gain=r["raw_J"] - r["J"], H_s=r["H_s"], export_s=r["export_s"],
                    validation_s=r["validation_s"], valid_model_plan=validation.get("valid_model_plan"),
                    all_tasks_served=validation.get("all_tasks_served"), collision_count=validation.get("collision_count"),
                    endpoint_constraints_satisfied=validation.get("endpoint_constraints_satisfied"),
                    physical_joint_trajectory_certified=validation.get("physical_joint_trajectory_certified"),
                    window_group="full10x10" if problem["window"]["width"] == problem["window"]["height"] == 10 else
                         f"local{problem['window']['width']}x{problem['window']['height']}",
                    mode=spec.get("mode"), endpoints=spec.get("endpoints"), metric_join_references=0)
                key = (problem["identity"], tuple(r["raw_route"]))
                by_key.setdefault(key, []).append(entry)
                for name in ("J", "L", "K", "raw_J", "raw_L", "raw_K", "H_s", "export_s", "validation_s"):
                    checks.append(dict(seed=seed, check="finite_terminal_" + name, source=inp.label(file), passed=finite(r[name])))
                checks.append(dict(seed=seed, check="saved_objective_algebra", source=inp.label(file),
                    passed=same(r["J"], r["L"] + STOP_PENALTY * r["K"]) and
                           same(r["raw_J"], r["raw_L"] + STOP_PENALTY * r["raw_K"])))
                all_terminal_count += 1
        seed_updates = []
        for phase in PHASES:
            folder = base / phase
            seen_updates = []
            for line_no, row in inp.jsonl(folder / "updates.jsonl"):
                u = row["update"]
                seen_updates.append(u)
                upd = {k: v for k, v in row.items() if k not in ("instances", "sampling_specs", "gradient_norms", "teacher_indices")}
                upd.update(seed=seed, source_log=inp.label(folder / "updates.jsonl"), source_line=line_no,
                           teacher_indices=row.get("teacher_indices"), loss_scale="REINFORCE" if phase == "RL" else "masked_route_mean_CE")
                for metric in ("loss", "gradient_norm_before_clip", "gradient_norm_after_clip", "elapsed_s", "stage_elapsed_s"):
                    checks.append(dict(seed=seed, phase=phase, update=u, check="finite_update_" + metric,
                                       passed=finite(row.get(metric))))
                if phase == "RL":
                    collected = []
                    for slot, inst in enumerate(row["instances"]):
                        spec = row["sampling_specs"][slot]
                        meta = dict(seed=seed, phase=phase, update=u, cursor_after=row["cursor_after"], slot=slot,
                            problem=inst["problem"], parent_identity=inst["parent"], mode=spec["mode"],
                            window_group="full10x10" if spec["width"] == spec["height"] == 10 else f"local{spec['width']}x{spec['height']}",
                            width=spec["width"], height=spec["height"], endpoints=spec["endpoints"],
                            map_seed=str(spec["map_seed"]), task_seed=str(spec["task_seed"]), parent_index=spec["parent"],
                            A_condition=spec["A_condition"], x0=spec["x0"], y0=spec["y0"], empty=inst["empty"])
                        instances.append(dict(meta, instance_loss=inst["loss"], **dist(inst["advantages"], "advantage_")))
                        checks.append(dict(seed=seed, phase=phase, update=u, slot=slot, check="finite_instance_loss_baseline_advantage",
                            passed=all(finite(v) for v in [inst["loss"], *inst["baseline"], *inst["advantages"]])))
                        for ri, tr in enumerate(inst["rows"]):
                            key = (inst["problem"], tuple(tr["raw_route"]))
                            matches = [m for m in by_key.get(key, ()) if same(m.get("J"), tr["J"]) and same(m.get("raw_J"), tr["raw_J"])]
                            valid_matches = [m for m in matches if m["completed"]]
                            terminal = valid_matches[-1] if valid_matches else (matches[-1] if matches else None)
                            item = dict(meta, row_role=tr["role"], sampling_seed_decimal=str(tr["sampling_seed"]),
                                forced_first=tr["forced_first"], unique_legal_first=tr["unique_legal_first"],
                                action_count=len(tr["actions"]), return_=tr["return_"], reward_negative_JH=-tr["J"],
                                raw_J=tr["raw_J"], H_J=tr["J"], H_gain=tr["raw_J"] - tr["J"],
                                baseline=inst["baseline"][ri], advantage=inst["advantages"][ri],
                                mixture_weight=inst["mixture_weights"][ri], logprob_sum=tr["logprob_sum"],
                                terminal_cache_hit=tr["terminal_cache_hit"], H_s=tr["H_s"], export_s=tr["export_s"],
                                validation_s=tr["validation_s"], raw_route=tr["raw_route"], final_route=tr["final_route"],
                                source_log=inp.label(folder / "updates.jsonl"), source_line=line_no,
                                **dist(tr["entropies"], "decision_entropy_"))
                            if terminal:
                                terminal["metric_join_references"] += 1
                                item.update(raw_L=terminal["raw_L"], raw_K=terminal["raw_K"], H_L=terminal["L"], H_K=terminal["K"],
                                    metric_source="matched_saved_terminal_by_problem_raw_route_and_J", terminal_file=terminal["terminal_file"],
                                    terminal_attempt_completed=terminal["completed"], valid_model_plan=terminal["valid_model_plan"])
                                checks.append(dict(seed=seed, check="trajectory_saved_route_match", update=u, slot=slot, row_role=tr["role"],
                                                   passed=tr["final_route"] == terminal["route"]))
                            else:
                                # Static frozen objective algebra, without a geometry query.
                                # These are labelled derivations, never fabricated terminal observations.
                                raw_k, h_k = len(tr["raw_route"]) - 2, len(tr["final_route"]) - 2
                                item.update(raw_K=raw_k, H_K=h_k, raw_L=tr["raw_J"] - STOP_PENALTY * raw_k,
                                    H_L=tr["J"] - STOP_PENALTY * h_k, metric_source="static_algebra_J_minus_0.75K", terminal_file=None,
                                    terminal_attempt_completed=None, valid_model_plan=None)
                            checks.append(dict(seed=seed, check="return_equals_negative_JH", update=u, slot=slot,
                                               row_role=tr["role"], passed=same(tr["return_"], -tr["J"])))
                            checks.append(dict(seed=seed, phase=phase, update=u, slot=slot, row_role=tr["role"],
                                check="finite_saved_trajectory_diagnostics", passed=all(finite(v) for v in
                                [tr["return_"], tr["J"], tr["raw_J"], tr["logprob_sum"], tr["H_s"], tr["export_s"], tr["validation_s"],
                                 *tr["rewards"], *[v for v in tr["logprobs"] if v is not None],
                                 *[v for v in tr["entropies"] if v is not None]])))
                            checks.append(dict(seed=seed, phase=phase, update=u, slot=slot, row_role=tr["role"],
                                check="saved_reward_sum_equals_complete_return", passed=same(sum(tr["rewards"]), tr["return_"])))
                            checks.append(dict(seed=seed, check="sampling_seed_exact_integer", update=u, slot=slot,
                                row_role=tr["role"], passed=isinstance(tr["sampling_seed"], int) and
                                not isinstance(tr["sampling_seed"], bool) and 0 <= tr["sampling_seed"] < 2**64))
                            trajectories.append(item)
                            collected.append(item)
                    for metric in ("return_", "raw_J", "H_J", "raw_L", "H_L", "raw_K", "H_K", "H_gain", "advantage", "H_s", "export_s", "validation_s"):
                        upd.update(dist([x[metric] for x in collected], metric + "_"))
                    upd.update(dist([v for i in row["instances"] for t in i["rows"] for v in t["entropies"]], "decision_entropy_"))
                    upd["cache_hit_n"] = sum(t["terminal_cache_hit"] for t in collected)
                    upd["completed_trajectory_n"] = len(collected)
                    upd["window_group"] = "full10x10" if u % 2 == 0 else "mixed_local_sizes"
                else:
                    upd["supervised_decision_n"] = sum(x["decisions"] for x in row["instances"])
                    upd["teacher_or_H_queries"] = row.get("teacher_or_H_queries", 0)
                for name, value in row.get("gradient_norms", {}).items():
                    norms.append(dict(seed=seed, phase=phase, update=u, cursor_after=row["cursor_after"], parameter=name,
                                      pre_clip_gradient_norm=value))
                    checks.append(dict(seed=seed, phase=phase, update=u, parameter=name,
                                       check="finite_parameter_gradient_norm", passed=finite(value)))
                updates.append(upd)
                seed_updates.append(upd)
            checks.append(dict(seed=seed, phase=phase, check="contiguous_completed_update_log", passed=seen_updates == list(range(len(seen_updates)))))
            for f in sorted(folder.glob("*.pt.commit.json")):
                r = inp.json(f)
                if r.get("phase") != phase:
                    raise ValueError("Checkpoint phase disagrees with containing phase: " + inp.label(f))
                checkpoints.append(dict(r, seed=seed, phase=phase, source=inp.label(f)))
            for f in sorted(folder.glob("development_*.json")):
                r = inp.json(f)
                dev.append(dict(seed=seed, phase=phase, update=r["update"], observed_cases=len(r["rows"]),
                                feasible_cases=sum(x["feasible"] for x in r["rows"]),
                                purpose=r["purpose"], source=inp.label(f)))
        for entry in attempt_map.values():
            entry["pending_without_completion_or_error"] = entry["started"] and not entry["completed"] and not entry["failed"]
            # Repeated identical routes may have multiple persisted attempts.
            # A deterministic metric join is not a chronological request join.
            entry["not_assigned_by_metric_join"] = entry.get("metric_join_references", 0) == 0
            attempts.append(entry)
        events_summary.extend(dict(seed=seed, event=event, count=n) for event, n in sorted(event_counts.items()))
        resource_contexts = (receipt or {}).get("cumulative_stage_resources", (receipt or {}).get("resources", {}))
        science_contexts = {k: v for k, v in resource_contexts.items() if k in (f"{seed}/RL", f"{seed}/continued_SL")}
        # RL and continued_SL share the same ledger and clock; choose the largest
        # cumulative snapshot, never sum contexts or duplicate phase ledgers.
        context = max(science_contexts.values(), key=lambda x: x.get("elapsed_wall_s", 0), default={})
        ledger = context.get("ledger", (final or {}).get("ledger", {}))
        phase_counts = {phase: sum(x["phase"] == phase for x in seed_updates) for phase in PHASES}
        observed_terminal = {role: dict(charged_attempts=sum(e["role"] == role and e["started"] for e in attempt_map.values()),
             completed=sum(e["role"] == role and e["completed"] for e in attempt_map.values()),
             failed=sum(e["role"] == role and e["failed"] for e in attempt_map.values()),
             pending=sum(e["role"] == role and e["pending_without_completion_or_error"] for e in attempt_map.values()),
             cache_hits=terminal_role_event_counts[(role, "terminal_cache_hit")])
             for role in ("RL", "development")}
        for role in observed_terminal:
            expected = ledger.get("terminal", {}).get(role)
            if expected:
                observed_terminal[role]["ledger_terminal_requests"] = expected["terminal_requests"]
                observed_terminal[role]["requests_stopped_before_attempt_or_cache_event"] = (
                    expected["terminal_requests"] - expected["terminal_attempts"] - expected["terminal_cache_hits"])
                for event_key, ledger_key in (("charged_attempts", "terminal_attempts"), ("completed", "terminal_completed"),
                                              ("failed", "terminal_failed"), ("cache_hits", "terminal_cache_hits")):
                    checks.append(dict(seed=seed, role=role, check="event_ledger_" + ledger_key,
                        passed=observed_terminal[role][event_key] == expected[ledger_key]))
        seed_tr = [r for r in trajectories if r["seed"] == seed]
        phase_diagnostics = {}
        for phase in PHASES:
            phase_rows = [r for r in seed_updates if r["phase"] == phase]
            before = [r["gradient_norm_before_clip"] for r in phase_rows]
            after = [r["gradient_norm_after_clip"] for r in phase_rows]
            phase_diagnostics[phase] = dict(completed_updates=len(phase_rows),
                loss=dist([r["loss"] for r in phase_rows]),
                loss_definition="complete-return mixed REINFORCE" if phase == "RL" else "route-mean masked teacher CE",
                gradient_before=dist(before), gradient_after=dist(after),
                gradient_before_min=min(before) if before else None,
                gradient_before_max=max(before) if before else None,
                clipped_updates_n=sum(x > 1. + 1e-7 for x in before),
                update_wall_s=dist([r["elapsed_s"] for r in phase_rows]),
                all_zero_advantage_updates_n=sum(r.get("zero_signal", False) for r in phase_rows) if phase == "RL" else None)
        descriptive_blocks = {}
        rl_indices = sorted({r["update"] for r in seed_tr})
        for block_name, block_indices in (("first32", set(rl_indices[:32])), ("last32", set(rl_indices[-32:]))):
            rows = [r for r in seed_tr if r["update"] in block_indices]
            block = dict(update_indices=sorted(block_indices), paired_improvement=False, groups={})
            for window_group in ("all_observed", "full10x10", "local4x3", "local6x6", "local8x8", "all_local"):
                members = rows if window_group == "all_observed" else [r for r in rows if
                    (r["window_group"].startswith("local") if window_group == "all_local" else r["window_group"] == window_group)]
                block["groups"][window_group] = {metric: dist([r[metric] for r in members]) for metric in
                    ("return_", "raw_J", "H_J", "raw_L", "H_L", "raw_K", "H_K", "H_gain")}
            descriptive_blocks[block_name] = block
        seed_summary.append(dict(seed=seed, status=status, completed_updates=phase_counts,
            RL_updates=phase_counts["RL"], continuedSL_updates=phase_counts["continued_SL"],
            fixed_endpoint_complete=status == "COMPLETE" and phase_counts == {"RL": 512, "continued_SL": 512} and final is not None,
            shared_S2_wall_s=context.get("elapsed_wall_s", (final or {}).get("shared_S2_wall_s")),
            final_stage_error=(receipt or {}).get("error"), terminal_events=observed_terminal,
            cumulative_ledger=ledger, resource_context_names=list(science_contexts),
            resource_ledger=ledger, stage_wall_s=context.get("elapsed_wall_s", (final or {}).get("shared_S2_wall_s")),
            raw_mean_J=statistics.fmean(r["raw_J"] for r in seed_tr) if seed_tr else None,
            H_mean_J=statistics.fmean(r["H_J"] for r in seed_tr) if seed_tr else None,
            raw_mean_L=statistics.fmean(r["raw_L"] for r in seed_tr) if seed_tr else None,
            H_mean_L=statistics.fmean(r["H_L"] for r in seed_tr) if seed_tr else None,
            raw_mean_K=statistics.fmean(r["raw_K"] for r in seed_tr) if seed_tr else None,
            H_mean_K=statistics.fmean(r["H_K"] for r in seed_tr) if seed_tr else None,
            mean_H_gain=statistics.fmean(r["H_gain"] for r in seed_tr) if seed_tr else None,
            advantage_distribution=dist([r["advantage"] for r in seed_tr]),
            all_zero_advantage_instances_n=sum(r["advantage_exact_zero_n"] == r["advantage_n"] for r in instances if r["seed"] == seed),
            entropy_trajectory_means=dist([r["decision_entropy_mean"] for r in seed_tr]),
            entropy_decision_weighted_mean=(sum(r["decision_entropy_mean"] * r["decision_entropy_n"] for r in seed_tr if r["decision_entropy_n"]) /
                sum(r["decision_entropy_n"] for r in seed_tr)) if sum(r["decision_entropy_n"] for r in seed_tr) else None,
            invalid_saved_plans_n=sum(r.get("valid_model_plan") is False for r in attempt_map.values()),
            failed_terminal_attempts_n=sum(r["failed"] for r in attempt_map.values()),
            pending_terminal_attempts_n=sum(r["pending_without_completion_or_error"] for r in attempt_map.values()),
            RL_completed_attempts_outside_completed_update_logs_n=(observed_terminal["RL"]["completed"] -
                sum(not r["terminal_cache_hit"] for r in seed_tr)),
            RL_cache_hits_outside_completed_update_logs_n=(observed_terminal["RL"]["cache_hits"] -
                sum(r["terminal_cache_hit"] for r in seed_tr)),
            phase_diagnostics=phase_diagnostics, first32_last32_descriptive_costs=descriptive_blocks,
            finite_checks=dict(total=sum(r["seed"] == seed and r["check"].startswith("finite_") for r in checks),
                failed=sum(r["seed"] == seed and r["check"].startswith("finite_") and not r["passed"] for r in checks)),
            shared_contexts_not_additive=True, fork_receipt_present=fork is not None,
            development_checks=[x for x in dev if x["seed"] == seed],
            unfinished_optimizer_attempts=max(0, ledger.get("optimizer_attempts", sum(phase_counts.values())) - ledger.get("optimizer_updates", sum(phase_counts.values())))))
        if status == "COMPLETE":
            checks.append(dict(seed=seed, check="complete_receipt_has_both_fixed_endpoints",
                passed=phase_counts == {"RL": 512, "continued_SL": 512} and final is not None and final.get("status") == "COMPLETE_FIXED_ENDPOINTS"))
    groups = []
    metrics = ("return_", "raw_J", "H_J", "raw_L", "H_L", "raw_K", "H_K", "H_gain", "advantage", "H_s", "export_s", "validation_s")
    for dimensions in (("seed",), ("seed", "window_group"), ("seed", "mode"), ("seed", "endpoints"),
                       ("seed", "window_group", "mode", "endpoints"), ("seed", "row_role"),
                       ("seed", "window_group", "mode", "endpoints", "row_role")):
        buckets = collections.defaultdict(list)
        for row in trajectories:
            buckets[tuple(row[k] for k in dimensions)].append(row)
        for key, rows in sorted(buckets.items(), key=lambda x: str(x[0])):
            meta = dict(group_dimensions="|".join(dimensions), **dict(zip(dimensions, key)),
                        trajectory_n=len(rows), instance_n=len({(r["update"], r["slot"]) for r in rows}),
                        update_n=len({r["update"] for r in rows}))
            for metric in metrics:
                groups.append(dict(meta, metric=metric, **dist([r[metric] for r in rows])))
    if inventory() != initial_inventory:
        raise RuntimeError("S2 input inventory changed during analysis; no final derived outputs written")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "updates.csv", updates, ("seed", "phase", "update", "cursor_after", "loss"))
    write_csv(out / "trajectories.csv", trajectories, ("seed", "phase", "update", "slot", "row_role", "sampling_seed_decimal"))
    write_csv(out / "instances.csv", instances)
    write_csv(out / "gradient_norms.csv", norms)
    write_csv(out / "terminal_attempts.csv", attempts)
    write_csv(out / "group_summary.csv", groups)
    write_csv(out / "events_summary.csv", events_summary)
    write_csv(out / "development_feasibility.csv", dev, ("seed", "phase", "update", "observed_cases", "feasible_cases", "purpose"))
    write_csv(out / "checkpoint_boundaries.csv", checkpoints)
    write_csv(out / "analysis_checks.csv", checks)
    write_csv(out / "input_manifest.csv", sorted(inp.manifest, key=lambda x: x["path"]))
    metric_summary = {str(seed): {metric: dist([r[metric] for r in trajectories if r["seed"] == seed]) for metric in metrics} for seed in SEEDS}
    summary = dict(schema="rc3-s2-saved-data-analysis-v1", analysis_only=True, runtime_changed=False,
        run=inp.label(run), protocol_sha256=EXPECTED_PROTOCOL, source_manifest_sha256=EXPECTED_SOURCE,
        optimizer_or_inference_or_scene_nav_H_teacher_calls=0, S3_or_original60_semantic_reads=0,
        fixed_seeds=list(SEEDS), seeds=seed_summary, per_seed={str(x["seed"]): x for x in seed_summary}, completed_update_rows=len(updates),
        RL_trajectory_rows=len(trajectories), saved_terminal_files=all_terminal_count,
        metric_summary=metric_summary, analysis_checks=dict(total=len(checks), failed=sum(not x["passed"] for x in checks)),
        finite_checks=dict(total=sum(r["check"].startswith("finite_") for r in checks),
                           failed=sum(r["check"].startswith("finite_") and not r["passed"] for r in checks)),
        metric_sources=dict(collections.Counter(r["metric_source"] for r in trajectories)),
        trajectory_cache_hits=sum(x["terminal_cache_hit"] for x in trajectories),
        limitations=["On-policy train inputs change across updates; cost/reward curves are descriptive and are not paired improvement tests.",
          "H_gain = raw_J - H_J measures the frozen H postprocessor, not an independent network gain.",
          "RL REINFORCE loss and continued-SL cross-entropy have different definitions/scales and are never compared as quality.",
          "Only preregistered development feasibility/resource checks at updates256/512; no quality selection or best-checkpoint choice.",
          "Sampled trajectories (three starts per instance) are dependent; group SD/quantiles describe observed draws, not confidence intervals.",
          "No deployment, generalization, frozen-SL comparison, native baseline superiority, or DRL-independent planning gain follows from these logs.",
          "Saved model-plan validation covers roadmap and proxy per-arm reachability constraints; it is not physical IK/joint-trajectory certification.",
          "Identical-route metric joins prove metric provenance, not chronological ownership of a request. Per-attempt not_assigned_by_metric_join is noncausal; incomplete-update work is counted separately by event-minus-completed-log totals and the cumulative ledger.",
          "Missing seeds/endpoints stay not executed/incomplete; no synthetic endpoint or replacement task is created."],
        derivation="For unmatched saved-trajectory metrics only: K = len(route)-2 and L = J-0.75*K; values are explicitly labelled static algebra. Preferred metrics join saved terminals by exact problem and raw_route with matching raw_J/H_J.",
        input_manifest="input_manifest.csv", output_directory=inp.label(out),
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        analysis_runtime=dict(python=sys.version, executable=sys.executable,
            wall_s=time.perf_counter() - analysis_started, process_cpu_s=time.process_time() - analysis_cpu_started,
            scientific_calls=0, ledger_category="postprocessing_only_separate_from_scientific_stage"),
        completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with (out / "ANALYSIS_SUMMARY.json").open("x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": inp.label(out), "updates": len(updates), "trajectories": len(trajectories),
                      "saved_terminals": all_terminal_count, "checks_failed": summary["analysis_checks"]["failed"],
                      "seeds": [{"seed": x["seed"], "status": x["status"], "updates": x["completed_updates"]} for x in seed_summary]}, ensure_ascii=False))
    return 1 if summary["analysis_checks"]["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
