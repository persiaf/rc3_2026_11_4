"""Offline R1 analysis; imports only the standard library and NumPy.

Input ``quality.jsonl`` has one object per completed route:
  phase: initial | round1 | round2 | timing
  method: A | B | C | D | warm_base | warm_nav | old128 | prior | gBH |
          fixed3 | detour16
  case: integer 0..31; J: positive final cost; L: nonnegative final length;
  K: nonnegative integer final stop count; valid: Boolean; total_s: full time.
Timing additionally requires cachemode: cold | hot and rep: integer 0..2.
rawroute/finalroute (or raw_route/final_route) are optional diagnostics. Initial
rows are required for the seven non-A/B/C/D methods; both rounds require all
four trained configurations. Timing requires nine methods (all except gBH and
warm_base), every case, both cache modes, and all three repetitions.

Input ``audit.jsonl`` records *every* audit opportunity, including selected
actions already queried: group, round (1|2), case, prefix (ordered IDs or stable
prefix identifier), queried (unique action IDs), selected, Yref, Yselected,
predicted (selected-action relative-advantage prediction), known_best (minimum
queried final cost). There are at most 128 opportunities/group/round. Optional
``labels_round.jsonl`` supplies the exact registered observed prefix sets; each
group/round must cover all 32 cases and remain within the 128-state cap. Audits
must match those sets and queried costs/actions exactly. Without labels the
legacy strict requirement of exactly 128 audit opportunities is retained.
Only an unknown selected action needs a new terminal evaluation. Diagnostics concern
actions at seen training states, not independent-state generalization.

No route is generated, validated, repaired, or postprocessed here. A stored
``valid`` flag is reported evidence from the independent validator. Missing,
malformed or duplicate primary data produce ``inconclusive`` instead of a
successful-subset result. Development intervals are exploratory: this same
development set is used for reference promotion and final R1 screening.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


BOOTSTRAP_SEED = 11412026
BOOTSTRAP_RESAMPLES = 20000
CASES = tuple(range(32))
GROUPS = ("A", "B", "C", "D")
INITIAL_METHODS = ("warm_base", "warm_nav", "old128", "prior", "gBH",
                   "fixed3", "detour16")
TIMING_METHODS = GROUPS + ("warm_nav", "old128", "prior", "fixed3", "detour16")
CACHE_MODES = ("cold", "hot")
PHASES = ("initial", "round1", "round2", "timing")
AUDIT_OPPORTUNITIES = 128


def _finite(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _status(criteria: dict[str, bool], complete: bool = True) -> str:
    if not complete:
        return "inconclusive"
    return "pass" if all(criteria.values()) else "fail"


def load_jsonl(path: str | Path) -> tuple[list[dict], list[str]]:
    rows, errors = [], []
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("row must be an object")
                    rows.append(row)
                except (ValueError, TypeError) as exc:
                    errors.append(f"{source.name}:{line_number}: {exc}")
    except OSError as exc:
        errors.append(f"{source}: {exc}")
    return rows, errors


def _quality_index(rows: Iterable[dict]) -> tuple[dict, list[str]]:
    index, errors = {}, []
    for number, row in enumerate(rows, 1):
        label = f"quality row {number}"
        if not isinstance(row, dict):
            errors.append(f"{label}: not an object")
            continue
        phase, method, case = row.get("phase"), row.get("method"), row.get("case")
        if phase not in PHASES or method not in GROUPS + INITIAL_METHODS:
            errors.append(f"{label}: unknown phase or method")
            continue
        if not _int(case) or case not in CASES:
            errors.append(f"{label}: case must be integer 0..31")
            continue
        mode, rep = row.get("cachemode"), row.get("rep")
        if phase == "timing" and (mode not in CACHE_MODES or not _int(rep)
                                   or rep not in (0, 1, 2)):
            errors.append(f"{label}: timing needs cold/hot and rep 0..2")
            continue
        fields_ok = (all(_finite(row.get(k)) for k in ("J", "L", "total_s"))
                     and row["J"] > 0 and row["L"] >= 0 and row["total_s"] >= 0
                     and _int(row.get("K")) and row["K"] >= 0
                     and isinstance(row.get("valid"), bool))
        if not fields_ok:
            errors.append(f"{label}: invalid/missing J,L,K,valid,total_s")
            continue
        key = (phase, method, case, mode, rep) if phase == "timing" else (
            phase, method, case)
        if key in index:
            errors.append(f"{label}: duplicate {key!r}")
        else:
            index[key] = row
    return index, errors


def _route_set(index: dict, phase: str, method: str) -> tuple[list[dict], list[int]]:
    missing = [case for case in CASES if (phase, method, case) not in index]
    return [index[(phase, method, case)] for case in CASES
            if (phase, method, case) in index], missing


def _summary(rows: list[dict], missing: list[int]) -> dict:
    if missing:
        return {"status": "inconclusive", "missing_cases": missing,
                "observed_cases": len(rows), "expected_cases": len(CASES)}
    return {"status": "complete", "cases": len(rows),
            "mean_J": float(np.mean([r["J"] for r in rows])),
            "mean_L": float(np.mean([r["L"] for r in rows])),
            "mean_K": float(np.mean([r["K"] for r in rows])),
            "feasible_count": sum(r["valid"] for r in rows),
            "feasible_rate": float(np.mean([r["valid"] for r in rows])),
            "invalid_cases": [r["case"] for r in rows if not r["valid"]]}


def paired_bootstrap(candidate: Iterable[float], reference: Iterable[float]) -> dict:
    """Paired scene bootstrap, larger relative improvement is better.

    Fractions, rather than percentages, are the primary numerical units.
    Identical seeded resampling indices are used for every 32-case comparison.
    The CI is percentile, two-sided 95%; the degradation bound is one-sided 95%.
    """
    cand, ref = np.asarray(list(candidate), dtype=float), np.asarray(list(reference), dtype=float)
    if (cand.shape != ref.shape or cand.shape != (len(CASES),)
            or not np.isfinite(cand).all() or not np.isfinite(ref).all()
            or np.any(cand <= 0) or np.any(ref <= 0)):
        return {"status": "inconclusive", "reason": "requires all 32 positive finite pairs"}
    gain = (ref - cand) / ref
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(CASES), (BOOTSTRAP_RESAMPLES, len(CASES)))
    samples = gain[indices].mean(axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    losses = -gain
    worst = int(np.argmax(losses))
    return {"status": "complete", "cases": len(CASES),
            "mean_relative_improvement": float(gain.mean()),
            "mean_relative_improvement_pct": float(100 * gain.mean()),
            "paired_95_ci": [float(low), float(high)],
            "mean_cost_reduction": float((ref - cand).mean()),
            "one_sided_95_degradation_upper": float(np.quantile(-samples, 0.95)),
            "positive_gain_cases": int(np.count_nonzero(gain > 1e-12)),
            "worst_relative_loss": float(losses[worst]), "worst_case": CASES[worst],
            "seed": BOOTSTRAP_SEED, "resamples": BOOTSTRAP_RESAMPLES,
            "per_case_relative_improvement": [float(x) for x in gain]}


def _comparison(index: dict, candidate_phase: str, candidate: str,
                reference_phase: str, reference: str) -> dict:
    cand, cmiss = _route_set(index, candidate_phase, candidate)
    ref, rmiss = _route_set(index, reference_phase, reference)
    if cmiss or rmiss:
        return {"status": "inconclusive", "candidate_missing": cmiss,
                "reference_missing": rmiss}
    result = paired_bootstrap([r["J"] for r in cand], [r["J"] for r in ref])
    result["all_feasible"] = all(r["valid"] for r in cand + ref)
    return result


def promotion_decisions(quality_rows: Iterable[dict]) -> dict:
    """Only round1 C/D versus their own frozen common warm starting policy."""
    index, errors = _quality_index(quality_rows)
    decisions = {}
    for group, warm in (("C", "warm_base"), ("D", "warm_nav")):
        comparison = _comparison(index, "round1", group, "initial", warm)
        complete = comparison["status"] == "complete" and not errors
        criteria = {}
        if comparison["status"] == "complete":
            criteria = {"mean_improvement_at_least_0.005":
                        comparison["mean_relative_improvement"] >= 0.005,
                        "paired_95_lower_positive": comparison["paired_95_ci"][0] > 0,
                        "all_feasible": comparison["all_feasible"]}
        status = _status(criteria, complete)
        decisions[group] = {"status": status, "promote": status == "pass",
                            "otherwise": "retain g0; no effective reference update",
                            "criteria": criteria, "comparison": comparison}
    return {"status": "inconclusive" if errors else "complete",
            "errors": errors, "decisions": decisions}


def _timing(index: dict) -> tuple[dict, list[str]]:
    report, errors = {}, []
    for mode in CACHE_MODES:
        report[mode] = {}
        for method in TIMING_METHODS:
            missing, scene_medians, raw, invalid = [], [], [], []
            for case in CASES:
                keys = [("timing", method, case, mode, rep) for rep in range(3)]
                absent = [key for key in keys if key not in index]
                if absent:
                    missing.extend([{"case": key[2], "rep": key[4]} for key in absent])
                    continue
                rows = [index[key] for key in keys]
                times = [r["total_s"] for r in rows]
                raw.extend(times)
                scene_medians.append(float(np.median(times)))
                invalid.extend([{"case": case, "rep": r["rep"]}
                                for r in rows if not r["valid"]])
                phase = "round2" if method in GROUPS else "initial"
                quality = index.get((phase, method, case))
                values = [r["J"] for r in rows]
                if quality is not None:
                    values.append(quality["J"])
                if max(values) - min(values) > max(1e-7, 1e-9 * max(values)):
                    errors.append(f"timing/quality cost mismatch {mode}/{method}/case{case}")
                other = index.get(("timing", method, case,
                                   "hot" if mode == "cold" else "cold", 0))
                if other is not None and not math.isclose(values[0], other["J"],
                                                          rel_tol=1e-9, abs_tol=1e-7):
                    errors.append(f"cold/hot cost mismatch {method}/case{case}")
            if missing:
                report[mode][method] = {"status": "inconclusive", "missing": missing,
                                        "expected_calls": 96, "observed_calls": len(raw)}
            else:
                report[mode][method] = {"status": "complete", "cases": 32, "calls": 96,
                    "median_s": float(np.median(scene_medians)),
                    "p95_s": float(np.quantile(scene_medians, 0.95)),
                    "raw_call_p95_s": float(np.quantile(raw, 0.95)),
                    "per_scene_median_s": scene_medians,
                    "all_feasible": not invalid, "invalid": invalid}
    return report, sorted(set(errors))


def _audit_summary(rows: list[dict]) -> dict:
    unknown = [r for r in rows if r["selected"] not in r["queried"]]
    def metrics(subset: list[dict]) -> dict:
        if not subset:
            return {"count": 0}
        true = np.asarray([(r["Yref"] - r["Yselected"]) / r["Yref"] for r in subset])
        prediction = np.asarray([r["predicted"] for r in subset])
        loss = np.asarray([(r["Yselected"] - r["known_best"]) / r["Yref"] for r in subset])
        absolute = np.asarray([r["Yselected"] - r["known_best"] for r in subset])
        bias = prediction - true  # positive = overly optimistic selected action
        worst = int(np.argmax(loss))
        return {"count": len(subset), "mean_true_relative_advantage": float(true.mean()),
            "mean_prediction_bias": float(bias.mean()),
            "mean_positive_optimism": float(np.maximum(bias, 0).mean()),
            "advantage_mse": float(np.mean((prediction - true)**2)),
            "mean_signed_gap_to_known_best": float(absolute.mean()),
            "mean_relative_signed_gap_to_known_best": float(loss.mean()),
            "mean_positive_relative_regret": float(np.maximum(loss, 0).mean()),
            "optimistic_sign_error_rate": float(np.mean((prediction > 0) & (true < 0))),
            "worst_relative_loss": float(loss[worst]),
            "worst_state": {k: subset[worst][k] for k in ("group", "round", "case", "prefix")}}
    return {"opportunities": len(rows), "unknown_count": len(unknown),
            "unknown_rate": len(unknown) / len(rows) if rows else None,
            "all_selected": metrics(rows), "unknown_selected": metrics(unknown)}


def _label_states(rows: Iterable[dict]) -> tuple[dict, list[str]]:
    expected, errors = {}, []
    for number, row in enumerate(rows, 1):
        label = f"round-label row {number}"
        if not isinstance(row, dict):
            errors.append(f"{label}: not an object")
            continue
        group, rnd, case = row.get("group"), row.get("round"), row.get("case")
        if (group not in GROUPS or not _int(rnd) or rnd not in (1, 2)
                or not _int(case) or case not in CASES or "prefix" not in row):
            errors.append(f"{label}: missing/invalid group,round,case,prefix")
            continue
        try:
            prefix_key = json.dumps(row["prefix"], sort_keys=True, separators=(",", ":"), allow_nan=False)
            actions, costs = row.get("actions"), row.get("Y")
            valid = (isinstance(actions, list) and bool(actions) and len(set(actions)) == len(actions)
                     and isinstance(costs, list) and len(costs) == len(actions)
                     and all(_finite(y) and y > 0 for y in costs)
                     and _finite(row.get("Yref")) and row["Yref"] > 0
                     and math.isclose(row["Yref"], costs[0], rel_tol=1e-9, abs_tol=1e-7))
        except (TypeError, ValueError):
            valid, prefix_key = False, ""
        if not valid:
            errors.append(f"{label}: invalid queried actions or terminal costs")
            continue
        key = (group, rnd, case, prefix_key)
        if key in expected:
            errors.append(f"{label}: duplicate prefix {key!r}")
        else:
            expected[key] = row
    for group in GROUPS:
        for rnd in (1, 2):
            subset = [key for key in expected if key[:2] == (group, rnd)]
            missing = sorted(set(CASES) - {key[2] for key in subset})
            if missing:
                errors.append(f"round-label {group}/{rnd}: missing cases {missing}")
            if len(subset) > AUDIT_OPPORTUNITIES:
                errors.append(f"round-label {group}/{rnd}: {len(subset)} states exceeds cap128")
    return expected, errors


def analyze_audits(rows: Iterable[dict], labels_rows: Iterable[dict] | None = None) -> dict:
    buckets, errors, seen = defaultdict(list), [], set()
    expected = None
    if labels_rows is not None:
        expected, label_errors = _label_states(labels_rows)
        errors.extend(label_errors)
    for number, row in enumerate(rows, 1):
        label = f"audit row {number}"
        if not isinstance(row, dict):
            errors.append(f"{label}: not an object")
            continue
        group, rnd, case = row.get("group"), row.get("round"), row.get("case")
        if (group not in GROUPS or not _int(rnd) or rnd not in (1, 2)
                or not _int(case) or case not in CASES or "prefix" not in row):
            errors.append(f"{label}: missing/invalid group,round,case,prefix")
            continue
        queried = row.get("queried")
        try:
            unique = len(set(queried)) == len(queried) if isinstance(queried, list) else False
            selected_hashable = hash(row["selected"]) is not None
            prefix_key = json.dumps(row["prefix"], sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError, KeyError):
            unique, selected_hashable, prefix_key = False, False, ""
        if (not unique or not queried or not selected_hashable
                or not all(_finite(row.get(k)) for k in ("Yref", "Yselected", "predicted", "known_best"))
                or row["Yref"] <= 0 or row["Yselected"] < 0 or row["known_best"] < 0
                or row["known_best"] > row["Yref"] + 1e-7):
            errors.append(f"{label}: invalid actions or advantage-cost fields")
            continue
        key = (group, rnd, case, prefix_key)
        if key in seen:
            errors.append(f"{label}: duplicate state {key!r}")
            continue
        seen.add(key)
        buckets[(group, rnd)].append(row)
        if expected is not None and key in expected:
            reference = expected[key]
            if (row["queried"] != reference["actions"]
                    or not math.isclose(row["Yref"], reference["Yref"], rel_tol=1e-9, abs_tol=1e-7)
                    or not math.isclose(row["known_best"], min(reference["Y"]), rel_tol=1e-9, abs_tol=1e-7)):
                errors.append(f"{label}: audit queried menu/reference cost differs from saved label")
            if row["selected"] in reference["actions"]:
                cost = reference["Y"][reference["actions"].index(row["selected"])]
                if not math.isclose(row["Yselected"], cost, rel_tol=1e-9, abs_tol=1e-7):
                    errors.append(f"{label}: selected queried-action cost differs from saved label")
    if expected is not None:
        for key in sorted(set(expected) - seen):
            errors.append(f"audit missing registered prefix {key!r}")
        for key in sorted(seen - set(expected)):
            errors.append(f"audit extra unregistered prefix {key!r}")
    report, complete = {}, not errors
    for group in GROUPS:
        report[group] = {}
        for rnd in (1, 2):
            subset = buckets[(group, rnd)]
            summary = _audit_summary(subset)
            count = (sum(key[:2] == (group, rnd) for key in expected)
                     if expected is not None else AUDIT_OPPORTUNITIES)
            summary["status"] = "complete" if len(subset) == count and count > 0 else "inconclusive"
            summary["expected_opportunities"] = count
            summary["opportunity_cap"] = AUDIT_OPPORTUNITIES
            complete = complete and summary["status"] == "complete"
            report[group][str(rnd)] = summary
    return {"status": "complete" if complete else "inconclusive", "errors": errors,
            "coverage_rule": "exact saved labels with all32 cases and cap128" if expected is not None else "legacy exactly128 per group/round",
            "scope": "selected-action diagnostics at seen training prefixes; not independent state validation",
            "by_group_round": report,
            "observed_diagnostics": _audit_summary([r for subset in buckets.values() for r in subset])}


def analyze(quality_rows: Iterable[dict], audit_rows: Iterable[dict],
            source_errors: Iterable[str] = (), labels_rows: Iterable[dict] | None = None) -> dict:
    quality_rows, audit_rows = list(quality_rows), list(audit_rows)
    index, errors = _quality_index(quality_rows)
    errors.extend(source_errors)
    quality, missing = {}, []
    for phase, methods in (("initial", INITIAL_METHODS), ("round1", GROUPS), ("round2", GROUPS)):
        quality[phase] = {}
        for method in methods:
            rows, absent = _route_set(index, phase, method)
            quality[phase][method] = _summary(rows, absent)
            if absent:
                missing.append(f"{phase}/{method}: {absent}")
    comparisons = {}
    for method in ("B", "C", "warm_nav", "old128", "fixed3", "detour16"):
        phase = "round2" if method in GROUPS else "initial"
        comparisons[f"D_vs_{method}"] = _comparison(index, "round2", "D", phase, method)
    # Same-label warm representation comparison is cleaner than the later
    # different-reference/state-distribution comparisons.
    comparisons["warm_nav_vs_warm_base"] = _comparison(index, "initial", "warm_nav", "initial", "warm_base")
    comparisons["B_vs_A"] = _comparison(index, "round2", "B", "round2", "A")
    timing, timing_errors = _timing(index)
    errors.extend(timing_errors)
    audits = analyze_audits(audit_rows, labels_rows)
    timing_complete = all(timing[m][method]["status"] == "complete"
                          for m in CACHE_MODES for method in TIMING_METHODS)
    complete = not errors and not missing and timing_complete and audits["status"] == "complete"
    all_feasible = all(r["valid"] for r in index.values()) if index else False
    research_criteria = {"all_recorded_routes_feasible": all_feasible}
    primary_complete = all(comparisons[f"D_vs_{m}"]["status"] == "complete"
                           for m in ("B", "warm_nav", "old128"))
    if primary_complete:
        for method in ("B", "warm_nav", "old128"):
            comp = comparisons[f"D_vs_{method}"]
            research_criteria[f"D_vs_{method}_paired_95_lower_positive"] = comp["paired_95_ci"][0] > 0
        research_criteria["D_vs_B_mean_improvement_at_least_0.005"] = (
            comparisons["D_vs_B"]["mean_relative_improvement"] >= 0.005)
    research = {"status": _status(research_criteria, complete and primary_complete),
                "criteria": research_criteria,
                "interpretation": "single-seed development screen; does not authorize or establish R2"}
    deployment = {}
    for mode in CACHE_MODES:
        dm, dt, ft = (timing[mode][m] for m in ("D", "detour16", "fixed3"))
        cmp = comparisons["D_vs_detour16"]
        ready = (complete and cmp["status"] == "complete"
                 and all(t["status"] == "complete" for t in (dm, dt, ft)))
        criteria, ratios, dominated = {}, {}, None
        if ready and min(dt["median_s"], dt["p95_s"]) > 0:
            ratios = {"median_ratio_to_detour16": dm["median_s"] / dt["median_s"],
                      "p95_ratio_to_detour16": dm["p95_s"] / dt["p95_s"]}
            d_j = quality["round2"]["D"]["mean_J"]
            f_j = quality["initial"]["fixed3"]["mean_J"]
            d_values, f_values = (d_j, dm["median_s"], dm["p95_s"]), (f_j, ft["median_s"], ft["p95_s"])
            dominated = (all(f <= d for f, d in zip(f_values, d_values))
                         and any(f < d for f, d in zip(f_values, d_values)))
            criteria = {"all_recorded_routes_feasible": all_feasible,
                "detour16_one_sided_95_degradation_upper_at_most_0.005":
                    cmp["one_sided_95_degradation_upper"] <= 0.005,
                "median_ratio_at_most_0.7": ratios["median_ratio_to_detour16"] <= 0.7,
                "p95_ratio_at_most_1.0": ratios["p95_ratio_to_detour16"] <= 1.0,
                "not_dominated_by_fixed3": not dominated}
        else:
            ready = False
        deployment[mode] = {"status": _status(criteria, ready), "criteria": criteria,
            **ratios, "fixed3_dominates": dominated,
            "frontier_axes": "mean J, median of scene-median time, p95 of scene-median time"}
    return {"schema": "rc3.r1.analysis.v1", "status": "complete" if complete else "inconclusive",
        "errors": errors, "missing_quality": missing,
        "bootstrap": {"seed": BOOTSTRAP_SEED, "resamples": BOOTSTRAP_RESAMPLES,
                      "unit": "scene", "relative_units": "fraction"},
        "evidence_scope": "single matched training seed; exploratory development evidence; reserved60 untouched",
        "quality": quality, "comparisons": comparisons,
        "round1_promotion": promotion_decisions(quality_rows), "timing": timing,
        "timing_definition": "three repeats -> median per scene -> median/p95 across 32 scenes; raw-call p95 separate",
        "unknown_action_audits": audits, "research_gate": research,
        "deployment_gate": deployment,
        "next_stage": "R2 requires a separate explicit task and independently frozen protocol"}


def analyze_files(quality_path: str | Path, audit_path: str | Path,
                  labels_path: str | Path | None = None) -> dict:
    quality, qerrors = load_jsonl(quality_path)
    audits, aerrors = load_jsonl(audit_path)
    labels, lerrors = load_jsonl(labels_path) if labels_path is not None else (None, [])
    return analyze(quality, audits, qerrors + aerrors + lerrors, labels)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--labels", type=Path, help="saved labels_round.jsonl for exact audit coverage")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze_files(args.quality, args.audit, args.labels)
    content = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    else:
        print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
