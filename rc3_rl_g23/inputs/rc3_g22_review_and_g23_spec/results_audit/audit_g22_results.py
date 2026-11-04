#!/usr/bin/env python3
"""Offline G2.2 ledger audit. Only stdlib; never imports a planner or neural model.

Usage:
  python audit_g22_results.py --source ../source/rc3_rl_g22 --output RESULTS_RECOMPUTED.json
Inputs are existing JSON / gzip JSON records. No training, environment step,
scene generation, planner call, neural inference, or timing experiment occurs.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

METHODS = ("fixed3", "detour", "stop", "detour_early", "continued_ce", "ce_gain")
SHARED = METHODS[2:]
TOL = 1e-9


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values):
    values = list(values)
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                minimum=min(values), maximum=max(values), total=sum(values))


def pair(a, b, field, higher_better=False):
    diffs = {i: a[i][field] - b[i][field] for i in sorted(a)}
    direction = -1 if higher_better else 1
    wins = [i for i, d in diffs.items() if direction * d < -TOL]
    ties = [i for i, d in diffs.items() if abs(d) <= TOL]
    losses = [i for i, d in diffs.items() if direction * d > TOL]
    return dict(field=field, higher_better=higher_better, difference="first minus second",
                statistics=summarize(diffs.values()), win_count=len(wins), tie_count=len(ties),
                loss_count=len(losses), win_cases=wins, tie_cases=ties, loss_cases=losses,
                differences_by_case=diffs)


def audit(source):
    failures = []
    check_count = 0

    def check(condition, name):
        nonlocal check_count
        check_count += 1
        if not condition:
            failures.append(name)

    def close(a, b, name, tolerance=1e-7):
        check(math.isclose(a, b, rel_tol=0, abs_tol=tolerance), name)

    outputs = source / "outputs"
    evaluation = outputs / "evaluation"
    protocol = read(outputs / "protocol.json")
    records = read(evaluation / "results.json")
    published = read(outputs / "summary.json")
    recorded_plan_audit = read(outputs / "plan_audit.json")
    audit_by_key = {(r["case_id"], r["method"]): r for r in recorded_plan_audit["checks"]}
    case_ids = sorted({r["case_id"] for r in records})
    by = {(r["case_id"], r["method"]): r for r in records}
    check(case_ids == list(range(12)), "case IDs exactly 0..11")
    check(len(records) == len(by) == 72, "72 distinct case-method records")
    check(set(by) == set(itertools.product(case_ids, METHODS)), "complete 12 by 6 grid")
    check(sha(outputs / "protocol.json") == (outputs / "protocol.sha256").read_text().strip(), "protocol checksum")
    for name, expected in protocol["implementation_sha256"].items():
        check(sha(source / name) == expected, "implementation checksum: " + name)

    attempts = [json.loads(s) for s in (evaluation / "attempts.jsonl").read_text().splitlines() if s]
    schedule = read(evaluation / "schedule.json")
    check(attempts == schedule, "attempt ledger exactly scheduled slots")
    check(len(attempts) == 72, "no extra recorded plan attempts")
    for i, r in enumerate(records):
        check({k: r[k] for k in ["slot", "case_id", "method"]} == schedule[i], "result slot matches schedule: " + str(i))
    source_scenes = {}
    for c in protocol["cases"]:
        scene_path = source / "inputs" / ("scene_%02d.json" % c["case_id"])
        check(sha(scene_path) == c["scene_sha256"], "scene checksum " + str(c["case_id"]))
        source_scenes[c["case_id"]] = read(scene_path)

    recomputed = []
    raw_counts = collections.Counter()
    method_counters = {m: collections.Counter() for m in METHODS}
    plan_checks = []
    for r in records:
        case, method = r["case_id"], r["method"]
        tag = f"case_{case:02d}_{method}"
        plan_path, evidence_path = evaluation / tag / "plan.json", evaluation / tag / "evidence.json.gz"
        plan, evidence, scene = read(plan_path), read(evidence_path), source_scenes[case]
        audit_row = audit_by_key[case, method]
        check(sha(plan_path) == audit_row["plan_sha256"], tag + " saved plan audit identity")
        check(sha(evidence_path) == audit_row["evidence_sha256"], tag + " saved evidence audit identity")
        check(plan == evidence["plan"], tag + " evidence embeds same plan")
        check(r["status"] == "complete" and r["valid"], tag + " recorded completion")
        check(not r["failures"] and not r["fallback"], tag + " no failure or fallback")
        check(r["time_overrun_s"] == 0 and r["complete_s"] <= protocol["soft_complete_limit_s"], tag + " recorded soft time limit")
        validation = plan["validation"]
        check(validation["valid_model_plan"] and validation["all_tasks_served"] and not validation["errors"], tag + " recorded proxy validation")
        check(validation["collision_count"] == validation["point_collision_count"] == 0, tag + " recorded collision results")
        check(validation == audit_row["independent_validation"], tag + " matching saved independent validation")
        task_ids = [t["id"] for t in scene["tasks"]]
        assignments = [(a["task_id"], a["candidate_id"], a["arm"]) for a in plan["assignments"]]
        check(collections.Counter(a[0] for a in assignments) == collections.Counter(task_ids), tag + " each task assigned exactly once")
        check(not plan["unserved_task_ids"], tag + " no unserved tasks")
        sparse = {x["task_id"]: x for x in evidence["sparse_feasibility"]}
        check(all(arm in ["L", "R"] and cid in sparse[tid]["left" if arm == "L" else "right"]
                  for tid, cid, arm in assignments), tag + " assignment arm feasible in saved sparse relation")
        stops = plan["service_stops"]
        stop_tasks = [(a["task_id"], s["candidate_id"], a["arm"]) for s in stops for a in s["tasks"]]
        check(collections.Counter(assignments) == collections.Counter(stop_tasks), tag + " stop assignments consistent")
        check([s["candidate_id"] for s in stops] == plan["visit_order_candidate_ids"], tag + " stop sequence consistent")
        K = len(stops)
        polyline = plan["route_polyline"]
        L = sum(math.dist(a, b) for a, b in zip(polyline, polyline[1:]))
        J = L + .75 * K
        close(plan["stop_penalty_equivalent_m"], .75, tag + " penalty")
        for value in [r["L"], plan["route_length_m"], sum(plan["route_leg_costs_m"]), validation["independent_polyline_length_m"]]:
            close(L, value, tag + " length arithmetic")
        check(K == r["K"], tag + " service stop count")
        close(J, r["J"], tag + " J arithmetic")
        close(J, plan["objective_length_plus_stops_m"], tag + " plan J arithmetic")
        anchor = by[case, "fixed3"]["J"]
        close(r["anchor_J"], anchor, tag + " common Fixed3 anchor")
        gain = 100 * (anchor - r["J"]) / anchor
        close(gain, r["quality_gain_percent"], tag + " quality gain")
        raw_counts["plans"] += 1
        raw_counts["original_detour_repairs"] += (r["proposal_calls"] if method == "detour" else 0)
        step_rows = evidence["actions"]
        recorded_policy_inputs = evidence.get("policy_observations", [])
        R = delta_R = None
        if method in SHARED:
            counters = evidence["counters"]
            for k, v in counters.items():
                if type(v) is int:
                    method_counters[method][k] += v
            raw_counts["shared_episodes"] += 1
            raw_counts["shared_actions"] += len(step_rows)
            raw_counts["shared_stop_actions"] += sum(a["action"]["kind"] == "stop" for a in step_rows)
            raw_counts["shared_proposals"] += sum(a["proposal_calls"] for a in step_rows)
            raw_counts["recorded_policy_inputs"] += len(recorded_policy_inputs)
            check(len(step_rows) == r["steps"], tag + " action count")
            check(sum(a["proposal_calls"] for a in step_rows) == r["proposal_calls"] == counters["proposal_calls"], tag + " proposal count")
            check(counters["menu_work"] + counters["proposal_work"] == r["work_used"], tag + " work ledger")
            check(not counters["work_count_incomplete"] and not r["work_count_incomplete"], tag + " complete work accounting")
            check(not evidence["events"], tag + " no extra recorded reward events")
            check(step_rows[-1]["action"]["kind"] == "stop", tag + " final explicit STOP")
            close(sum(a["reward"] for a in step_rows), r["return_"], tag + " action reward sum")
            close(evidence["finalization"]["quality_correction"], 0., tag + " zero final correction")
            close(evidence["finalization"]["best_cost"], r["J"], tag + " returns recorded best cost")
            work = 0
            for i, a in enumerate(step_rows):
                check(a["step"] == i, tag + " action index")
                check(a["before_work"] == work, tag + " work continuity")
                work = a["after_work"]
                check(a["proposal_calls"] == (0 if a["action"]["kind"] == "stop" else 1), tag + " action type proposal count")
            check(work == r["work_used"], tag + " final work continuity")
            R = gain - protocol["config"]["lambda_work"] * r["work_used"] / protocol["config"]["work_scale"]
            delta_R = R - by[case, "stop"]["return_"]
            close(R, r["return_"], tag + " R arithmetic")
            close(delta_R, r["return_vs_stop"], tag + " paired STOP R")
            if method == "stop":
                close(r["root_stop_return_analytical"], R, tag + " actual root STOP accounting")
            if method in ["continued_ce", "ce_gain"]:
                check(len(recorded_policy_inputs) == len(step_rows), tag + " model input count")
                for observation, action in zip(recorded_policy_inputs, step_rows):
                    check(observation["executed"], tag + " recorded inference executed")
                    check(observation["action_id"] == action["action_id"], tag + " inference action recorded consistently")
                    logits = observation["scores"]["logits"]
                    chosen = max(range(len(logits)), key=lambda i: -math.inf if logits[i] is None else logits[i])
                    check(chosen == action["action_id"], tag + " recorded greedy argmax")
                    check(observation["observation"]["mask"][chosen], tag + " legal recorded action")
        else:
            check(r["return_"] is None and r["work_used"] is None, tag + " no imputed shared R")
            check(not step_rows, tag + " no shared action rows for original method")
            if method == "detour":
                check(evidence["diagnostics"]["dp_calls"] == 16, tag + " original Detour 16 repairs")
        plan_checks.append(dict(case_id=case, method=method, L_from_polyline=L, K_from_stops=K, J_from_geometry_and_stops=J,
                                recorded_proxy_valid=validation["valid_model_plan"], saved_arm_relation_checked=True,
                                plan_sha256=sha(plan_path), evidence_sha256=sha(evidence_path)))
        recomputed.append(dict(case_id=case, method=method, mode=r["mode"], task_count=r["task_count"],
                               J=r["J"], L=L, K=K, quality_gain_percent=gain, return_=R, return_vs_stop=delta_R,
                               work_used=r["work_used"], complete_ms=r["complete_s"] * 1000,
                               steps=r["steps"], proposal_calls=r["proposal_calls"], stop_at_root=r["stop_at_root"]))

    by_method = {m: {r["case_id"]: r for r in recomputed if r["method"] == m} for m in METHODS}
    summaries = {}
    for m in METHODS:
        subset = list(by_method[m].values())
        fields = ["J", "L", "K", "quality_gain_percent", "complete_ms", "steps", "proposal_calls"]
        if m in SHARED:
            fields += ["return_", "return_vs_stop", "work_used"]
        summary = {f: summarize(r[f] for r in subset) for f in fields}
        summary["root_stop_count"] = sum(r["stop_at_root"] for r in subset)
        summary["improved_quality_cases"] = sum(r["quality_gain_percent"] > TOL for r in subset)
        summary["raw_work_counters"] = dict(method_counters[m])
        if m in SHARED:
            summary["R_vs_actual_STOP"] = pair(by_method[m], by_method["stop"], "return_", higher_better=True)
            work = method_counters[m]
            summary["menu_fraction_of_counted_work"] = work["menu_work"] / (work["menu_work"] + work["proposal_work"])
        timing_rows = [r["timing"] for r in records if r["method"] == m]
        summary["recorded_timing_components_ms"] = {k: summarize(t[k] * 1000 for t in timing_rows)
                                                   for k, value in timing_rows[0].items() if isinstance(value, (int, float))}
        summaries[m] = summary
        for ours, theirs in [(summary["J"]["mean"], "mean_J"), (summary["L"]["mean"], "mean_L"),
                              (summary["K"]["mean"], "mean_K"), (summary["quality_gain_percent"]["mean"], "mean_gain_percent"),
                              (summary["complete_ms"]["median"], "median_complete_ms"),
                              (summary["proposal_calls"]["total"], "total_proposals")]:
            close(ours, published["methods"][m][theirs], "published summary matches " + m + " " + theirs)
        if m in SHARED:
            for field, name in [("return_", "mean_return"), ("return_vs_stop", "mean_return_vs_stop"), ("work_used", "mean_work")]:
                close(summary[field]["mean"], published["methods"][m][name], "published summary matches " + m + " " + name)

    pairs = {}
    for a, b in itertools.combinations(METHODS, 2):
        entry = {f: pair(by_method[a], by_method[b], f) for f in ["J", "complete_ms"]}
        if a in SHARED and b in SHARED:
            entry["R"] = pair(by_method[a], by_method[b], "return_", higher_better=True)
        pairs[a + "__vs__" + b] = entry
    raw_counts["scenes"] = len(case_ids)
    raw_counts["learned_policy_episodes"] = sum(r["method"] in ["continued_ce", "ce_gain"] for r in records)
    raw_counts["learned_proposals"] = sum(r["proposal_calls"] for r in records if r["method"] in ["continued_ce", "ce_gain"])
    receipt = read(evaluation / "receipt.json")
    check(raw_counts["shared_proposals"] + raw_counts["original_detour_repairs"] == receipt["proposal_calls"], "total 239 different-protocol proposal / repair calls")
    check(raw_counts["shared_actions"] == receipt["shared_actions"], "shared actions receipt")
    check(raw_counts["plans"] == receipt["attempted_plans"] == receipt["complete_plans"] == receipt["valid_plans"], "plan receipt counts")
    check(receipt["optimizer_updates"] == receipt["ppo_actions"] == receipt["new_teacher_branches"] == 0, "recorded no training or new teacher branches")

    before = read(outputs / "freeze_before.json")
    after = read(outputs / "freeze_after.json")
    freeze_keys_equal = {k: before[k] == after[k] for k in before}
    check(all(freeze_keys_equal.values()), "before / after frozen snapshot equality")
    check(after["before_snapshot_sha256"] == sha(outputs / "freeze_before.json"), "freeze before hash identity")
    setup = read(evaluation / "model_setup.json")
    replay = read(outputs / "inference_replay.json")
    for name, identity in protocol["models"].items():
        check(setup[name]["checkpoint_sha256"] == identity["sha256"], name + " model setup identity")
        check(before["g21_payloads"][identity["path"]]["actual"] == identity["sha256"], name + " model snapshot identity")
    check(replay["state_evaluations"] == raw_counts["recorded_policy_inputs"], "saved replay count matches actual inputs")
    check(all(m["passed"] and m["actions_equal"] and m["model_unchanged"] and m["max_logit_error"] == m["max_probability_error"] == 0
              for m in replay["models"]), "saved inference replay result")
    check(not protocol["training_allowed"] and not protocol["ppo_allowed"], "protocol training and PPO disallowed")

    return dict(
        schema="independent-g22-recorded-results-audit-v1",
        scope={"source": str(source.resolve()), "training_updates": 0, "new_environment_queries": 0,
               "new_plans": 0, "new_timing_experiments": 0, "new_neural_inferences": 0,
               "method": "stdlib recomputation from saved records only",
               "validation_limit": "Geometric length, task/arm ledger, stored feasibility and certificate identity checked; this script does not rerun collision/reachability geometry or IK.",
               "frozen_model_limit": "Saved hashes and before/after snapshots checked; model tensors not loaded or replayed by this script."},
        input_sha256={"protocol": sha(outputs / "protocol.json"), "results": sha(evaluation / "results.json"),
                      "summary": sha(outputs / "summary.json"), "evaluate.py": sha(source / "evaluate.py")},
        all_passed=not failures, check_count=check_count, failures=failures,
        formulas={"J": "L + 0.75 K", "gain_percent": "100 (J_fixed3 - J) / J_fixed3; average scene percentages",
                  "R_shared_only": "gain_percent - 0.01 W / 10000 = gain_percent - W / 1000000",
                  "delta_R": "R_method - actual R_stop on same scene", "pair_tolerance": TOL},
        counts=dict(raw_counts), methods=summaries, paired=pairs, rows=recomputed, plan_checks=plan_checks,
        frozen_evidence={"before_after_keys_equal": freeze_keys_equal, "model_setup": setup,
                         "saved_inference_replay": replay, "old_gate_files_exist": after["g21_gate_files_exist"]},
        timing_contract={"main": protocol["timing_scope"], "ordering": protocol["method_order"],
                         "single_sample_per_case_method": True, "soft_deadline_s": protocol["soft_complete_limit_s"],
                         "max_observed_complete_ms": max(r["complete_ms"] for r in recomputed),
                         "note": "Reported measurements only. A difference of method medians is not the median of paired differences."},
        limitations=["12 previously used development scenes; no independent held-out generalization claim.",
                     "Original Detour16 performs 192 repair calls; shared policies have different actions/work protocol and 19/13/15 proposals. No equal-work superiority claim.",
                     "Only one timing sample per case-method; no stable latency or real-time guarantee.",
                     "R uses a work proxy dominated by menus; R is not elapsed time.",
                     "72 saved artifacts and certificates do not constitute 72 independent scenes or 72 distinct routes.",
                     "Proxy-model feasibility is not real-arm IK, collision-free dual-arm motion, service duration, or physical safety certification.",
                     "CE and CE+gain checkpoints frozen; this evaluation provides no incremental PPO/DRL training benefit evidence."])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["all_passed", "check_count", "failures", "counts"]}, ensure_ascii=False))
    raise SystemExit(0 if result["all_passed"] else 1)


if __name__ == "__main__":
    main()
