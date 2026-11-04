"""Read-only G2 teacher audit and a predeclared, deterministic G2.1 mini-set.

Only NumPy/stdlib are used. No environment, policy, Torch, or oracle is loaded.
Pickles are deserialized only after matching pinned and saved SHA256 receipts.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


PINNED_SHA256 = {
    "initial": "f53d32640dc6a61f7014a0724bc7565c71c254a2ddc087f244a09e9d62d97ac8",
    "extra": "46a7c5133c6b7e4529f207365f49aa0b2e8f839ce1cea46f471fd885105d0a8a",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def recomputed_weights(samples):
    groups = defaultdict(lambda: defaultdict(list))
    for index, sample in enumerate(samples):
        groups[str(sample["scene_id"])][tuple(sample["current_route"])].append(index)
    result = np.zeros(len(samples), dtype=np.float64)
    for routes in groups.values():
        for indices in routes.values():
            result[indices] = 1.0 / (len(groups) * len(routes) * len(indices))
    return result


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0}
    return dict(count=len(values), minimum=float(values.min()),
                q25=float(np.quantile(values, .25)), median=float(np.median(values)),
                q75=float(np.quantile(values, .75)), maximum=float(values.max()),
                mean=float(values.mean()))


def load_split(workspace, split):
    directory = workspace / "rc3_rl_g2" / "outputs" / ("teacher_" + split)
    teacher_path = directory / "teacher.pkl"
    raw = teacher_path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    receipt = json.loads((directory / "audit_receipt.json").read_text(encoding="utf-8"))
    if not actual == PINNED_SHA256[split] == summary["teacher_sha256"] == receipt["teacher_sha256"]:
        raise ValueError("Teacher SHA256 mismatch before deserialization: " + split)
    if receipt.get("all_passed") is not True:
        raise ValueError("Saved teacher audit did not pass: " + split)
    data = pickle.loads(raw)
    if data["schema"] != "rc3-g2-finite-return-teacher-v1":
        raise ValueError("Unexpected teacher schema")
    samples = data["samples"]
    weights = recomputed_weights(samples)
    original_weights = np.asarray([s["weight"] for s in samples], dtype=float)
    if not np.allclose(weights, original_weights, rtol=0, atol=1e-15):
        raise ValueError("Saved scene/route weights disagree")
    rows = []
    root_scenes = set()
    for index, sample in enumerate(samples):
        q = np.asarray(sample["q_values"], dtype=float)
        evidence = np.asarray(sample["evidence_mask"], dtype=bool)
        legal = np.asarray(sample["legality_mask"], dtype=bool)
        ties = np.asarray(sample["ties"], dtype=bool)
        stop_id = int(sample["stop_id"])
        if not evidence[stop_id] or np.any(evidence & ~legal):
            raise ValueError("Illegal evidence or unevidenced STOP")
        if not np.all(np.isfinite(q[evidence])):
            raise ValueError("Non-finite evidenced return")
        expected_ties = evidence & (q >= q[evidence].max() - sample["tie_tolerance"])
        if not np.array_equal(ties, expected_ties):
            raise ValueError("Saved near-best labels disagree")
        if sample["path"]:
            continue
        scene_id = str(sample["scene_id"])
        if scene_id in root_scenes:
            raise ValueError("More than one root per scene")
        root_scenes.add(scene_id)
        case_index = int(scene_id.split(":", 1)[0])
        menus_path = directory / ("case_%04d" % case_index) / "menus.jsonl.gz"
        with gzip.open(menus_path, "rt", encoding="utf-8") as stream:
            menus = [json.loads(line) for line in stream]
        root_menus = [m for m in menus if m["path"] == []]
        if len(root_menus) != 1 or root_menus[0]["node_id"] != 0:
            raise ValueError("Saved root menu identity mismatch")
        menu = root_menus[0]
        if menu["queried_action_ids"] != sample["queried_action_ids"]:
            raise ValueError("Saved root query/menu mismatch")
        search_evidence = evidence.copy()
        search_evidence[stop_id] = False
        if not search_evidence.any():
            raise ValueError("Root has no queried non-STOP action")
        best_nonstop_q = float(q[search_evidence].max())
        delta = best_nonstop_q - float(q[stop_id])
        best_nonstop_ids = np.flatnonzero(search_evidence & (q == best_nonstop_q)).tolist()
        rows.append(dict(split=split, sample_index=index, case_index=case_index,
            scene_id=scene_id, node_id=int(menu["node_id"]), root=True, path=[],
            state_signature=str(sample["state_signature"]),
            problem_id=str(sample["problem_id"]), teacher_depth=int(sample["teacher_depth"]),
            stop_id=stop_id, q_stop=float(q[stop_id]), q_best_queried_nonstop=best_nonstop_q,
            delta=float(delta), label_margin=float(delta),
            best_queried_nonstop_action_ids=best_nonstop_ids,
            queried_action_ids=[int(x) for x in sample["queried_action_ids"]],
            near_best_action_ids=np.flatnonzero(ties).tolist(),
            stop_near_best=bool(ties[stop_id]),
            stop_unique_near_best=bool(ties[stop_id] and ties.sum() == 1),
            tie_tolerance=float(sample["tie_tolerance"]),
            original_weight=float(sample["weight"]),
            route_repeat_count=int(sample["route_repeat_count"]),
            scene_route_count=int(sample["scene_route_count"]),
            root_menu_file=str(menus_path.relative_to(workspace)).replace("\\", "/"),
            root_menu_sha256=sha256(menus_path)))
    if rows != sorted(rows, key=lambda r: (r["case_index"], r["sample_index"])):
        raise ValueError("Root record ordering differs from numeric case order")
    provenance = dict(split=split, teacher_file=str(teacher_path.relative_to(workspace)).replace("\\", "/"),
        teacher_sha256=actual, summary_sha256=sha256(directory / "summary.json"),
        audit_receipt_sha256=sha256(directory / "audit_receipt.json"),
        saved_audit_all_passed=True, sha_checked_before_pickle=True,
        saved_weights_recomputed=True)
    return samples, rows, provenance


def statistics(samples, roots, weights):
    root_indices = [i for i, sample in enumerate(samples) if not sample["path"]]
    positive = [r["delta"] for r in roots if r["delta"] > 0]
    return dict(samples=len(samples), scenes=len(set(str(s["scene_id"]) for s in samples)),
        roots=len(roots), raw_root_fraction=len(roots) / len(samples),
        root_weight_mass=float(weights[root_indices].sum()),
        weight_total=float(weights.sum()),
        positive_delta_count=len(positive), delta_greater_than_point1_count=sum(r["delta"] > .1 for r in roots),
        delta_at_least_point1_count=sum(r["delta"] >= .1 for r in roots),
        stop_near_best_count=sum(r["stop_near_best"] for r in roots),
        stop_unique_near_best_count=sum(r["stop_unique_near_best"] for r in roots),
        negative_selection_eligible_count=sum(r["delta"] <= -.02 and r["stop_unique_near_best"] for r in roots),
        positive_delta_distribution=distribution(positive),
        all_delta_distribution=distribution([r["delta"] for r in roots]))


def audit_saved_ppo_logs(workspace):
    """Recompute historical action frequencies; this does not run PPO."""
    directory = workspace / "rc3_rl_g2" / "outputs" / "training"
    iterations_path = directory / "ppo_iterations.jsonl"
    episodes_path = directory / "ppo_episodes.jsonl"
    iterations = [json.loads(line) for line in iterations_path.read_text(encoding="utf-8").splitlines() if line]
    episodes = [json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines() if line]
    previous = stop = non_stop = final_8192_non_stop = 0
    final_actions = int(iterations[-1]["actual_actions"])
    for item in iterations:
        actions = int(item["actual_actions"]) - previous
        if actions <= 0:
            raise ValueError("Historical training counters are not strictly increasing")
        stop_float = item["stop_fraction"] * actions
        stop_count = round(stop_float)
        if not np.isclose(stop_float, stop_count, rtol=0, atol=1e-8):
            raise ValueError("Historical STOP fraction is not an exact action count")
        stop += stop_count
        non_stop += actions - stop_count
        if previous >= final_actions - 8192:
            final_8192_non_stop += actions - stop_count
        elif int(item["actual_actions"]) > final_actions - 8192:
            raise ValueError("Last-8192 boundary cuts through an aggregate iteration")
        previous = int(item["actual_actions"])
    proposals = [int(item["counters"]["proposal_calls"]) for item in episodes]
    result = dict(actual_actions=final_actions, stop_actions=stop, non_stop_actions=non_stop,
        completed_episodes=len(episodes), logged_episode_proposals=sum(proposals),
        episodes_with_search=sum(x > 0 for x in proposals),
        episodes_with_one_proposal=sum(x == 1 for x in proposals),
        episodes_with_at_least_two_proposals=sum(x >= 2 for x in proposals),
        last_8192_non_stop_actions=final_8192_non_stop,
        historical_logs_only=True, iterations_sha256=sha256(iterations_path),
        episodes_sha256=sha256(episodes_path))
    if stop + non_stop != final_actions or sum(proposals) != non_stop:
        raise ValueError("Historical iteration and episode action counts disagree")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()
    datasets, roots, provenance = {}, {}, {}
    for split in ("initial", "extra"):
        datasets[split], roots[split], provenance[split] = load_split(args.workspace, split)
    positive = [dict(r, selection_label="continue", selection_stratum="positive")
                for r in roots["initial"] if r["delta"] >= .1][:8]
    negative = [dict(r, selection_label="stop", selection_stratum="negative")
                for r in roots["initial"] if r["delta"] <= -.02 and r["stop_unique_near_best"]][:8]
    if len(positive) != 8 or len(negative) != 8:
        raise ValueError("Predeclared thresholds provide insufficient roots; no threshold changed")
    chosen = sorted(positive + negative, key=lambda r: r["sample_index"])
    combined = datasets["initial"] + datasets["extra"]
    stats = {s: statistics(datasets[s], roots[s], recomputed_weights(datasets[s]))
             for s in ("initial", "extra")}
    stats["combined"] = statistics(combined, roots["initial"] + roots["extra"], recomputed_weights(combined))
    args.out.mkdir(parents=True, exist_ok=True)
    selection = dict(schema="rc3-g21-fixed16-selection-v1", source_split="initial",
        selection_rule="First eight qualifying roots in original numeric case_index order in each stratum; output samples retain original sample order.",
        thresholds=dict(positive_delta_at_least=.1, negative_delta_at_most=-.02,
                        negative_requires_unique_near_best_stop=True, per_stratum=8),
        delta_definition="max Q over evidenced non-STOP actions minus evidenced Q_STOP; unknown actions excluded",
        provenance=provenance["initial"], sample_indices=[r["sample_index"] for r in chosen],
        positive_sample_indices=[r["sample_index"] for r in positive],
        negative_sample_indices=[r["sample_index"] for r in negative],
        positive_margin_distribution=distribution([r["delta"] for r in positive]),
        negative_margin_distribution=distribution([r["delta"] for r in negative]),
        samples=chosen)
    selection_path = args.out / "selection.json"
    if selection_path.exists():
        prior = json.loads(selection_path.read_text(encoding="utf-8"))
        if prior != selection:
            raise ValueError("Existing selection differs: refusing to replace fixed mini-set")
    write_json(selection_path, selection)
    all_roots = roots["initial"] + roots["extra"]
    columns = ("split", "sample_index", "case_index", "scene_id", "node_id", "root", "delta",
               "q_stop", "q_best_queried_nonstop", "stop_near_best", "stop_unique_near_best", "original_weight")
    with (args.out / "root_labels.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_roots)
    report = dict(schema="rc3-g21-teacher-readonly-audit-v1", statistics=stats,
        historical_ppo_log_audit=audit_saved_ppo_logs(args.workspace),
        sources=provenance, selection_sha256=sha256(selection_path),
        script_sha256=sha256(__file__), root_labels_sha256=sha256(args.out / "root_labels.csv"),
        scope=dict(torch_used=False, policy_inference_performed=False, environment_queries=0,
                   source_files_modified=False, conclusions="Teacher evidence only; not policy performance or real robot feasibility."),
        checked_review_claims=dict(initial_positive_39_of_64=stats["initial"]["positive_delta_count"] == 39,
            combined_positive_65_of_96=stats["combined"]["positive_delta_count"] == 65,
            combined_above_point1_60_of_96=stats["combined"]["delta_greater_than_point1_count"] == 60,
            initial_stop_near_best_25_of_64=stats["initial"]["stop_near_best_count"] == 25,
            combined_stop_near_best_31_of_96=stats["combined"]["stop_near_best_count"] == 31))
    for split in ("initial", "extra"):
        source = args.workspace / provenance[split]["teacher_file"]
        if sha256(source) != PINNED_SHA256[split]:
            raise ValueError("Frozen teacher changed during audit")
    report["source_sha256_unchanged_after_audit"] = True
    write_json(args.out / "data_audit.json", report)
    print(json.dumps(dict(selection=str(selection_path), selection_sha256=report["selection_sha256"],
                          statistics=stats, checked_review_claims=report["checked_review_claims"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
