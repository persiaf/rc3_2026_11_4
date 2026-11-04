#!/usr/bin/env python3
"""Read saved G2.2 JSON only; do not import project code or execute a planner.

Usage:
  python audit_trajectories.py --source /path/to/rc3_rl_g22 --output /path/to/audit.json

The stopping calculation uses only decision states actually visited by each
frozen policy, pays the already-built pending menu, and assumes a hypothetical
prefix final check would not roll back its incumbent. It is a hindsight
accounting diagnostic, not a deployable policy, counterfactual rollout, global
bound, or timing measurement. Teacher labels are checked for internal
consistency only: the original teacher.pkl is deliberately not loaded.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


SHARED = ("stop", "detour_early", "continued_ce", "ce_gain")
MODELS = ("continued_ce", "ce_gain")
CLASSES = ("action_queried", "action_unqueried", "state_unseen")
EPS = 1e-8


def read(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, message):
    check(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-7), f"{message}: {a} != {b}")


def signs(values):
    values = list(values)
    return {"positive": sum(v > EPS for v in values),
            "tie": sum(abs(v) <= EPS for v in values),
            "negative": sum(v < -EPS for v in values)}


def gain(anchor, cost):
    return 100.0 * (anchor - cost) / max(anchor, 1e-8)


def action_group(rows):
    return {
        "proposals": len(rows),
        "best_improvements": sum(v["best_improved"] for v in rows),
        "positive_immediate_reward": sum(v["reward"] > EPS for v in rows),
        "positive_actual_suffix_return": sum(v["actual_policy_suffix_return"] > EPS for v in rows),
        "actual_suffix_better_than_analytic_same_state_stop": sum(v["actual_return_vs_this_stop"] > EPS for v in rows),
        "accepted": sum(v["accepted"] for v in rows),
        "route_changed": sum(v["route_changed"] for v in rows),
        "current_cost_improved": sum(v["current_cost_change"] < -EPS for v in rows),
        "current_cost_worsened": sum(v["current_cost_change"] > EPS for v in rows),
        "current_cost_equal": sum(abs(v["current_cost_change"]) <= EPS for v in rows),
        "mean_immediate_reward": mean(v["reward"] for v in rows) if rows else None,
        "menu_work": sum(v["menu_work"] for v in rows),
        "proposal_work": sum(v["proposal_work"] for v in rows),
    }


def audit(source):
    protocol = read(source / "outputs/protocol.json")
    results = read(source / "outputs/evaluation/results.json")
    feedback = read(source / "outputs/action_feedback.json")
    cfg = protocol["config"]
    check(cfg["potential_weight"] == 0, "This calculation requires zero potential")
    work_rate = cfg["lambda_work"] / cfg["work_scale"]
    by_result = {(v["case_id"], v["method"]): v for v in results}
    by_feedback = {(v["case_id"], v["method"], v["step"]): v for v in feedback}
    check(len(results) == len(by_result) == 72, "Expected 72 unique result rows")
    check(len(feedback) == len(by_feedback) == 95, "Expected 95 unique feedback rows")

    episodes, actions, repeated_records = [], [], []
    evidence_hashes = {}
    state_labels, action_labels = {}, {}
    for result in results:
        if result["method"] not in SHARED:
            continue
        case, method = result["case_id"], result["method"]
        path = source / "outputs/evaluation" / f"case_{case:02d}_{method}" / "evidence.json.gz"
        evidence_hashes[str(path.relative_to(source))] = sha(path)
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        trace = data["actions"]
        root_observation = trace[0]["environment_observation"]
        anchor = result["anchor_J"]
        check(result["shared_environment"] and trace, "Missing shared trace")
        check(not data["events"], "This run should have no exceptional terminal events")
        check(not data["finalization"]["fallback_used"], "Unexpected final fallback")
        close(data["finalization"]["quality_correction"], 0.0, "Unexpected final correction")
        check(trace[-1]["action"]["kind"] == "stop", "Expected actual explicit STOP")
        check(result["stop_reason"] == "stop", "Unexpected terminal reason")
        close(data["finalization"]["returned_cost"], result["J"], "Returned J")
        close(result["J"], trace[-1]["after_best_J"], "Returned incumbent")
        close(result["J"], data["plan"]["objective_length_plus_stops_m"], "Plan objective")
        check(data["final_best"] == trace[-1]["environment_observation"]["best"], "Final incumbent route")
        close(result["quality_gain_percent"], gain(anchor, result["J"]), "Final quality")

        work_so_far, reward_so_far, proposals_so_far = 0, 0.0, 0
        candidates, local_actions, seen_operations = [], [], {}
        root_return = by_result[(case, "stop")]["return_"]
        for index, raw in enumerate(trace):
            info, obs = raw["info"], raw["environment_observation"]
            aid = raw["action_id"]
            tag = f"case {case} {method} step {index}"
            check(raw["step"] == index, tag + " step order")
            check(obs["mask"][aid] and obs["actions"][aid] == raw["action"], tag + " selected legality")
            check(raw["before_work"] == work_so_far, tag + " work prefix")
            close(raw["prefix_return"], reward_so_far, tag + " reward prefix")
            close(obs["anchor_cost"], anchor, tag + " anchor")
            for name in ("current", "best"):
                close(raw[f"before_{name}_J"], obs[f"{name}_cost"], tag + " state cost")
                close(raw[f"before_{name}_J"], info[f"previous_{name}_cost"], tag + " prior cost")
                close(raw[f"after_{name}_J"], info[f"{name}_cost"], tag + " after cost")
            check(raw["after_best_J"] <= raw["before_best_J"] + 1e-7, tag + " incumbent monotonicity")
            check(raw["after_best_J"] >= result["J"] - 1e-7, tag + " final incumbent is best seen")
            check(info["menu_work_units"] == obs["pending_menu_work"], tag + " pending menu charge")
            check(info["work_units"] == info["menu_work_units"] + info["proposal_work_units"], tag + " action work components")
            check(raw["after_work"] == raw["before_work"] + info["work_units"], tag + " work delta")
            quality = gain(anchor, raw["after_best_J"]) - gain(anchor, raw["before_best_J"])
            close(raw["immediate_best_quality_percent"], quality, tag + " immediate quality")
            close(info["quality_reward"], quality, tag + " reward quality")
            close(info["work_penalty"], work_rate * info["work_units"], tag + " work penalty")
            close(info["shaping_reward"], 0.0, tag + " zero shaping")
            close(raw["reward"], quality - work_rate * info["work_units"], tag + " reward")
            close(raw["realized_policy_suffix_return"], result["return_"] - reward_so_far, tag + " actual suffix")

            paid_if_stop = work_so_far + obs["pending_menu_work"]
            return_if_stop = gain(anchor, raw["before_best_J"]) - work_rate * paid_if_stop
            close(return_if_stop, raw["hypothetical_stop_total_return"], tag + " pending-menu STOP")
            if index == 0:
                close(return_if_stop, root_return, tag + " true root STOP calibration")
            candidate = {"decision_step": index, "proposals_completed": proposals_so_far,
                         "J": raw["before_best_J"], "quality_gain_percent": gain(anchor, raw["before_best_J"]),
                         "work_before_decision": work_so_far, "pending_menu_work": obs["pending_menu_work"],
                         "work_if_stop": paid_if_stop, "return_if_stop": return_if_stop,
                         "return_vs_actual_root_stop": return_if_stop - root_return,
                         "actual_action": raw["action"]["kind"], "actual_stop": raw["action"]["kind"] == "stop"}
            candidates.append(candidate)

            status = raw["teacher_status"]
            check(status in CLASSES, tag + " teacher class")
            known = status == "action_queried"
            check((raw["known_Q"] is not None) == known, tag + " known-Q null discipline")
            check((raw["known_delta_vs_stop"] is not None) == known, tag + " known delta null discipline")
            check((raw["teacher_depth"] is None) == (status == "state_unseen"), tag + " teacher depth scope")
            check((raw["teacher_state_signature"] is None) == (status == "state_unseen"), tag + " teacher state scope")
            state_key = (case, digest(obs))
            state_label = (status == "state_unseen", raw["teacher_depth"], raw["teacher_state_signature"])
            check(state_key not in state_labels or state_labels[state_key] == state_label, tag + " repeated exact-state label")
            state_labels[state_key] = state_label
            action_key = (*state_key, aid)
            action_label = (status, raw["known_Q"], raw["known_delta_vs_stop"])
            check(action_key not in action_labels or action_labels[action_key] == action_label, tag + " repeated exact-action label")
            action_labels[action_key] = action_label
            saved = by_feedback[(case, method, index)]
            for key in ("teacher_status", "action_id", "proposal_calls"):
                check(saved[key] == raw[key], tag + " exported feedback " + key)
            for key in ("reward", "before_best_J", "after_best_J", "hypothetical_stop_total_return", "realized_policy_suffix_return"):
                close(saved[key], raw[key], tag + " exported feedback " + key)
            close(saved["policy_return_vs_analytic_stop"], result["return_"] - return_if_stop, tag + " exported same-state delta")

            next_current = trace[index + 1]["environment_observation"]["current"] if index + 1 < len(trace) else data["final_current"]
            check(info["route_changed"] == (obs["current"] != next_current), tag + " current route changed flag")
            if index + 1 < len(trace):
                close(trace[index + 1]["before_current_J"], raw["after_current_J"], tag + " current continuity")
                close(trace[index + 1]["before_best_J"], raw["after_best_J"], tag + " incumbent continuity")
            rec = {"case_id": case, "method": method, "step": index, "action_id": aid,
                   "kind": raw["action"]["kind"], "teacher_status": status,
                   "action": raw["action"], "current_route_sha256": digest(obs["current"]),
                   "menu_context_key": obs["menu_context_key"],
                   "action_context_key": obs["action_context_keys"][aid],
                   "proposal_key": obs["proposal_keys"][aid],
                   "prior_attempts": obs["action_attempts"][aid].get("attempts", 0),
                   "prior_no_change": obs["action_attempts"][aid].get("no_change", False),
                   "exact_observation_sha256": state_key[1], "proposals": raw["proposal_calls"],
                   "current_before_J": raw["before_current_J"], "current_after_J": raw["after_current_J"],
                   "current_cost_change": raw["after_current_J"] - raw["before_current_J"],
                   "best_before_J": raw["before_best_J"], "best_after_J": raw["after_best_J"],
                   "best_improved": raw["after_best_J"] < raw["before_best_J"] - EPS,
                   "quality_gain_percent": quality, "reward": raw["reward"],
                   "accepted": info["accepted"], "route_changed": info["route_changed"],
                   "menu_work": info["menu_work_units"], "proposal_work": info["proposal_work_units"],
                   "work": info["work_units"], "actual_policy_suffix_return": result["return_"] - reward_so_far,
                   "analytic_stop_remaining_return": return_if_stop - reward_so_far,
                   "actual_return_vs_this_stop": result["return_"] - return_if_stop,
                   "proposal_failure_reason": info["failure_reason"]}
            root_matches = [i for i, action in enumerate(root_observation["actions"])
                            if action == raw["action"] and root_observation["mask"][i]]
            rec["matching_legal_root_action_ids"] = root_matches
            rec["same_current_route_as_root"] = obs["current"] == root_observation["current"]
            rec["same_saved_proposal_key_as_matching_legal_root_action"] = any(
                root_observation["proposal_keys"][i] == rec["proposal_key"] for i in root_matches)
            if raw["proposal_calls"]:
                operation = digest({"problem_id": obs["problem_id"], "current": obs["current"],
                                    "action": raw["action"], "protocol_config": cfg})
                if operation in seen_operations:
                    prior, prior_rec = seen_operations[operation]
                    repeated_records.append({
                        "case_id": case, "method": method,
                        "earlier_step": prior["step"], "later_step": index,
                        "same_recorded_problem_current_action_config": True,
                        "previous_route_unchanged": not prior["info"]["route_changed"],
                        "later_route_unchanged": not info["route_changed"],
                        "earlier_proposal_key": prior_rec["proposal_key"], "later_proposal_key": rec["proposal_key"],
                        "same_proposal_key": prior_rec["proposal_key"] == rec["proposal_key"],
                        "same_entire_saved_proposal_record": prior["info"]["proposal"] == info["proposal"],
                        "earlier_full_state_sha256": prior_rec["exact_observation_sha256"],
                        "later_full_state_sha256": rec["exact_observation_sha256"],
                        "earlier_action_context_key": prior_rec["action_context_key"],
                        "later_action_context_key": rec["action_context_key"],
                        "later_selected_attempt_memory": obs["action_attempts"][aid],
                        "later_reward": raw["reward"], "later_menu_work": info["menu_work_units"],
                        "later_proposal_work": info["proposal_work_units"],
                        "scope": "Observed serialized-input and output equality only; does not establish general solver determinism, safe cache equivalence, or counterfactual rollout effects."})
                seen_operations[operation] = (raw, rec)
            local_actions.append(rec)
            work_so_far += info["work_units"]
            reward_so_far += raw["reward"]
            proposals_so_far += raw["proposal_calls"]

        check(work_so_far == result["work_used"], "Final work ledger")
        check(proposals_so_far == result["proposal_calls"], "Proposal ledger")
        close(reward_so_far, result["return_"], "Final reward ledger")
        close(reward_so_far, gain(anchor, result["J"]) - work_rate * work_so_far, "Final reward identity")
        close(candidates[-1]["return_if_stop"], result["return_"], "Actual terminal STOP included")
        # Existing endpoint is included; tie preference is the earliest decision.
        best = max(candidates, key=lambda v: (v["return_if_stop"], -v["decision_step"]))
        same_quality = next(v for v in candidates if abs(v["J"] - result["J"]) <= 1e-7)
        detour = by_result[(case, "detour")]
        episode = {"case_id": case, "method": method, "anchor_J": anchor,
                   "actual_final_J": result["J"], "actual_quality_gain_percent": result["quality_gain_percent"],
                   "actual_return": result["return_"], "actual_root_stop_return": root_return,
                   "actual_return_vs_root_stop": result["return_"] - root_return,
                   "actual_work": work_so_far, "actual_proposals": proposals_so_far,
                   "actual_stop_decision_step": len(trace) - 1,
                   "hindsight_best_visited_stop": best,
                   "hindsight_return_uplift": best["return_if_stop"] - result["return_"],
                   "hindsight_work_reduction": work_so_far - best["work_if_stop"],
                   "hindsight_quality_loss_percent_points": result["quality_gain_percent"] - best["quality_gain_percent"],
                   "earliest_visited_stop_preserving_final_J": same_quality,
                   "preserve_J_work_reduction": work_so_far - same_quality["work_if_stop"],
                   "preserve_J_return_uplift": same_quality["return_if_stop"] - result["return_"],
                   "proposals_after_final_best_was_found": proposals_so_far - same_quality["proposals_completed"],
                   "original_detour_J": detour["J"],
                   "actual_quality_gap_to_detour_percent_points": detour["quality_gain_percent"] - result["quality_gain_percent"],
                   "hindsight_quality_gap_to_detour_percent_points": detour["quality_gain_percent"] - best["quality_gain_percent"],
                   "visited_stop_candidates": candidates, "actions": local_actions}
        episodes.append(episode)
        actions.extend(local_actions)

    summaries = {}
    for method in SHARED:
        own = [v for v in episodes if v["method"] == method]
        all_act = [v for v in actions if v["method"] == method]
        proposals = [v for v in all_act if v["proposals"]]
        total_work = sum(v["work"] for v in all_act)
        menu_work = sum(v["menu_work"] for v in all_act)
        summaries[method] = {
            "episodes": len(own), "actions": len(all_act),
            "stop_after_proposals_histogram": dict(sorted(Counter(v["actual_proposals"] for v in own).items())),
            "proposal_diagnostics": action_group(proposals),
            "selected_actions_with_prior_attempt_count_positive": sum(v["prior_attempts"] > 0 for v in proposals),
            "selected_actions_flagged_prior_no_change": sum(v["prior_no_change"] for v in proposals),
            "same_recorded_inputs_repeats": sum(v["method"] == method for v in repeated_records),
            "same_recorded_inputs_repeats_after_no_change": sum(v["method"] == method and v["previous_route_unchanged"] for v in repeated_records),
            "actual_mean_quality_gain_percent": mean(v["actual_quality_gain_percent"] for v in own),
            "actual_mean_return_vs_root_stop": mean(v["actual_return_vs_root_stop"] for v in own),
            "actual_vs_root_stop_counts": signs(v["actual_return_vs_root_stop"] for v in own),
            "hindsight_mean_return_vs_root_stop": mean(v["hindsight_best_visited_stop"]["return_vs_actual_root_stop"] for v in own),
            "hindsight_vs_root_stop_counts": signs(v["hindsight_best_visited_stop"]["return_vs_actual_root_stop"] for v in own),
            "hindsight_mean_return_uplift": mean(v["hindsight_return_uplift"] for v in own),
            "hindsight_earlier_stop_cases": sum(v["hindsight_best_visited_stop"]["decision_step"] < v["actual_stop_decision_step"] for v in own),
            "hindsight_same_final_J_cases": sum(v["hindsight_quality_loss_percent_points"] <= EPS for v in own),
            "hindsight_mean_quality_gain_percent": mean(v["hindsight_best_visited_stop"]["quality_gain_percent"] for v in own),
            "preserve_J_total_work_reduction": sum(v["preserve_J_work_reduction"] for v in own),
            "preserve_J_mean_return_vs_root_stop": mean(v["earliest_visited_stop_preserving_final_J"]["return_vs_actual_root_stop"] for v in own),
            "preserve_J_earlier_stop_cases": sum(v["preserve_J_work_reduction"] > 0 for v in own),
            "proposals_after_final_best_was_found": sum(v["proposals_after_final_best_was_found"] for v in own),
            "actual_mean_quality_gap_to_detour_percent_points": mean(v["actual_quality_gap_to_detour_percent_points"] for v in own),
            "hindsight_mean_quality_gap_to_detour_percent_points": mean(v["hindsight_quality_gap_to_detour_percent_points"] for v in own),
            "actual_J_strictly_worse_than_detour_cases": sum(v["actual_final_J"] > v["original_detour_J"] + EPS for v in own),
            "hindsight_J_strictly_worse_than_detour_cases": sum(v["hindsight_best_visited_stop"]["J"] > v["original_detour_J"] + EPS for v in own),
            "work_total": total_work, "menu_work_total": menu_work,
            "menu_fraction_of_counted_work": menu_work / total_work,
            "teacher_proposal_classes": {label: action_group([v for v in proposals if v["teacher_status"] == label]) for label in CLASSES},
        }

    worsening = [v for v in actions if v["current_cost_change"] > EPS]
    episodes_with_worsening = {(v["case_id"], v["method"]) for v in worsening}
    check(len(episodes) == 48 and len(actions) == 95, "Shared episode/action counts")
    check(sum(v["proposals"] for v in actions) == 47, "Shared proposal count")
    cases_0_1_8 = [v for v in episodes if (v["case_id"], v["method"]) in ((0, "continued_ce"), (1, "ce_gain"), (8, "ce_gain"))]
    return {
        "schema": "g22-saved-trajectory-audit-v1", "new_environment_queries": 0,
        "new_planner_runs": 0, "new_training_updates": 0, "new_timing_measurements": 0,
        "scope": "48 saved shared episodes on 12 old development scenes; descriptive and conditional hindsight accounting only",
        "formula": {"quality_percent": "100*(anchor_J-best_J)/anchor_J",
                    "work_rate": work_rate,
                    "hindsight_stop_return": "100*(anchor_J-best_before_J)/anchor_J - work_rate*(work_before+pending_menu_work)",
                    "assumption": "Zero potential; hypothetical prefix final verification has no rollback or quality correction. Pending menu is paid. No extra counted work is imputed for verification; this is not a wall-clock model.",
                    "selection": "Maximum over the actually visited decision states, including root and actual terminal STOP; no unvisited alternatives or actions",
                    "quality_property": "Since incumbent best is monotone and actual finalization has no rollback, an earlier visited STOP cannot improve returned J; it can preserve or worsen it."},
        "teacher_label_scope": {"source_semantics": "Exact full environment observation digest, then evidence mask of selected action; same route or action id alone is insufficient",
                                "independently_rebuilt_from_teacher_pkl": False,
                                "checked": "95 exported feedback rows against raw saved actions; null discipline and repeated full-observation/action consistency",
                                "unique_observed_full_states": len(state_labels),
                                "unique_observed_full_state_actions": len(action_labels)},
        "checks": {"all_passed": True, "shared_episodes": len(episodes), "shared_actions": len(actions),
                   "shared_proposals": sum(v["proposals"] for v in actions),
                   "all_return_final_incumbent_without_rollback": True,
                   "all_root_stop_calculations_match_actual_root_stop": True,
                   "all_actual_terminal_returns_match_stop_formula": True,
                   "all_best_costs_nonincreasing": True},
        "method_summary": summaries,
        "repeat_diagnostics": {"records": repeated_records,
                               "total_repeats_after_prior_no_change": sum(v["previous_route_unchanged"] for v in repeated_records),
                               "scope": "Within-episode comparisons of selected proposals only; no hypothetical cache or execution performed."},
        "uphill_observations": {"actions": worsening, "episode_count": len(episodes_with_worsening),
                                "worsening_actions_with_later_best_improvement": sum(any(a["best_improved"] and a["step"] > w["step"] for a in actions if (a["case_id"], a["method"]) == (w["case_id"], w["method"])) for w in worsening)},
        "cases_0_1_8": cases_0_1_8, "episodes": episodes,
        "input_sha256": {"protocol.json": sha(source / "outputs/protocol.json"),
                         "results.json": sha(source / "outputs/evaluation/results.json"),
                         "action_feedback.json": sha(source / "outputs/action_feedback.json"),
                         "evidence": evidence_hashes},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    check(not output.is_relative_to(source), "Write audit output outside the source package")
    data = audit(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"all_passed": data["checks"]["all_passed"],
                      "shared_episodes": data["checks"]["shared_episodes"],
                      "shared_actions": data["checks"]["shared_actions"],
                      "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
