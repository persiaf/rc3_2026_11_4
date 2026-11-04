"""Read saved evidence only. No project imports, inference, scene creation or replay."""
from pathlib import Path
import collections
import gzip
import hashlib
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "outputs"
TRACE = Path(__file__).resolve().parent


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def boot(a):
    a = np.asarray(a, float)
    rng = np.random.default_rng(114)
    samples = a[rng.integers(len(a), size=(10000, len(a)))].mean(axis=1)
    return dict(mean=float(a.mean()), ci95=np.quantile(samples, [.025, .975]).tolist(),
                upper_one_sided95=float(np.quantile(samples, .95)))


records = read("final_evaluation.json")
reference_records = read("references.json")
assert len(records) == 672
assert len({(r["case"], r["rep"], r["cache_mode"], r["method"]) for r in records}) == 672
assert len(reference_records) == 480
methods = ["fixed3", "detour8", "detour16", "insertion", "prior", "initial", "model"]
assert {r["case"] for r in records} == set(range(24))
assert {r["rep"] for r in records} == {0, 1}
assert all(r["raw_valid"] and r["valid"] for r in records)
assert all(not r["fallback"] for r in records)
assert all(abs(r["J"] - r["L"] - .75*r["K"]) < 1e-7 for r in records)
assert all(r["J"] <= r["raw_J"] + 1e-7 for r in records)
summary = {}
for mode in ["cold", "hot"]:
    summary[mode] = {}
    for method in methods:
        rows = [r for r in records if r["cache_mode"] == mode and r["method"] == method]
        assert len(rows) == 48
        means = [np.mean([r["total_s"] for r in rows if r["case"] == case])*1000 for case in range(24)]
        times = [r["total_s"]*1000 for r in rows]
        summary[mode][method] = dict(mean_J=float(np.mean([r["J"] for r in rows])),
            mean_raw_J=float(np.mean([r["raw_J"] for r in rows])),
            pooled_repetitions_median_ms=float(np.median(times)),
            pooled_repetitions_p95_ms=float(np.quantile(times, .95)),
            scene_averaged_median_ms=float(np.median(means)),
            scene_averaged_p95_ms=float(np.quantile(means, .95)),
            pooled_common_fraction_median=float(np.median([(r["preparation_s"]+r["export_validation_s"])/r["total_s"] for r in rows])),
            scene_averaged_common_fraction_median=float(np.median([
                np.mean([r["preparation_s"]+r["export_validation_s"] for r in rows if r["case"] == case]) /
                np.mean([r["total_s"] for r in rows if r["case"] == case]) for case in range(24)])))
    m, d = summary[mode]["model"], summary[mode]["detour16"]
    summary[mode]["time_ratios_model_detour16"] = {
        prefix + metric: m[prefix+metric+"_ms"] / d[prefix+metric+"_ms"]
        for prefix in ["pooled_repetitions_", "scene_averaged_"] for metric in ["median", "p95"]}
    summary[mode]["dominators_mean_J_and_median_time"] = [
        method for method in methods if method != "model"
        and summary[mode][method]["mean_J"] <= m["mean_J"]
        and summary[mode][method]["scene_averaged_median_ms"] <= m["scene_averaged_median_ms"]]

paired = {}
for comparator in methods[:-1]:
    values = []
    for case in range(24):
        ours = np.mean([r["J"] for r in records if r["case"] == case and r["method"] == "model"])
        base = np.mean([r["J"] for r in records if r["case"] == case and r["method"] == comparator])
        values.append(float(100*(ours-base)/base))
    paired[comparator] = boot(values)

train_summary = read("training_summary.json")
curve = read("validation_curve.json")
selected_update = min(curve, key=lambda row:(row["mean_post_J"], row["update"]))["update"]
assert selected_update == train_summary["selected_update"] == 128
initial = read("selection/update0000.json")
chosen = read(f"selection/update{selected_update:04}.json")
assert [r["case"] for r in initial] == [r["case"] for r in chosen] == list(range(48))
a = np.array([r["post_J"] for r in initial])
b = np.array([r["post_J"] for r in chosen])
rng = np.random.default_rng(114)
idx = rng.integers(48, size=(10000,48))
gains = 100*(a[idx].mean(1)-b[idx].mean(1))/a[idx].mean(1)
selection = dict(ratio_of_means_percent=float(100*(a.mean()-b.mean())/a.mean()),
    ratio_of_means_ci95=np.quantile(gains,[.025,.975]).tolist(),
    mean_paired_relative_percent=float(np.mean(100*(a-b)/a)),
    initial_mean_post_J=float(a.mean()), selected_mean_post_J=float(b.mean()),
    selected_update=selected_update,
    reuse_caveat="48 scenes select the checkpoint and estimate gain; interval conditional on selection, not independent test evidence")
small = read("small_exact.json")
exact = []
for row in small:
    f = row["full_exact"]
    r = row["baseline_candidate_restriction"]["result"]
    strict = f["objective"] < r["lower_bound"]-1e-8
    assert strict == row["strict_outside_set_value"]
    assert f["optimal"] and r["optimal"] and f["status"] == r["status"] == "optimal"
    assert abs(f["objective"]-f["lower_bound"]) < 1e-8
    assert abs(r["objective"]-r["lower_bound"]) < 1e-8
    assert row["candidates"] == 96
    exact.append(dict(case=row["case"], tasks=row["tasks"], full_UB=f["objective"], full_LB=f["lower_bound"],
        restricted_UB=r["objective"], restricted_LB=r["lower_bound"], strict_outside_set=strict))
assert len(exact) == 12 and sum(x["strict_outside_set"] for x in exact) == 6
small_gaps = {}
for method in ["fixed3", "detour8", "detour16", "insertion"]:
    gaps = np.array([100*(r["baselines"][method]["J"]/r["full_exact"]["objective"]-1) for r in small])
    small_gaps[method] = dict(mean_gap_percent=float(gaps.mean()), max_gap_percent=float(gaps.max()))
history = [json.loads(line) for line in (OUT/"training.jsonl").read_text(encoding="utf-8").splitlines()]
assert [r["update"] for r in history] == list(range(1,513))
assert all(r["sampled_trajectories"] == 32*r["update"] == r["baseline_greedy_trajectories"] for r in history)
assert all(math.isfinite(r["loss"]) and math.isfinite(r["gradient_norm"]) for r in history)
episodes=steps=0
maps=collections.Counter()
task_seeds=set()
with gzip.open(OUT/"sampled_routes.jsonl.gz", "rt", encoding="utf-8") as stream:
    for line in stream:
        row = json.loads(line)
        assert row["update"] == episodes//32+1
        assert row["task_seed"] == 7200000000+episodes
        assert row["map_seed"] == 7100000000+episodes%24
        assert row["route"][0] == row["route"][-1] == 0
        assert all(isinstance(x,int) and 1 <= x <= 800 for x in row["route"][1:-1])
        maps[row["map_seed"]] += 1
        task_seeds.add(row["task_seed"])
        episodes += 1
        steps += len(row["route"])-2
assert episodes == 16384 and len(maps) == 24 and len(task_seeds) == 16384
registration = read("registration.json")
execution = read("execution_sources.json")
wall = read("wall_protocol.json")
old_hashes = {name:sha(ROOT.parent/name) == value for name,value in registration["source_sha256"].items()}
execution_hashes = {name:sha(ROOT/name) == value for name,value in execution["source_sha256"].items()}
assert all(old_hashes.values()) and all(execution_hashes.values())
assert sha(ROOT/"protocol.yaml") == registration["protocol_sha256"] == execution["protocol_sha256"] == wall["protocol_sha256"]
assert sha(OUT/"execution_sources.json") == wall["execution_sources_sha256"]
assert sha(OUT/"training.jsonl") == train_summary["history_sha256"]
assert sha(OUT/"sampled_routes.jsonl.gz") == train_summary["stored_sampled_routes_sha256"]
assert wall["training_wall_limit_s"] == min(7200,max(1800,math.ceil(512*(max(wall["greedy_forward_backward_probe_s"])+wall["input_preparation_32_s"])*2+600)))
analysis = read("analysis.json")
for mode in ["cold", "hot"]:
    for method in methods:
        independent = summary[mode][method]
        supplied = analysis["summary"][mode][method]
        for left,right in [("scene_averaged_median_ms","median_ms"),("scene_averaged_p95_ms","p95_ms"),
                           ("pooled_repetitions_median_ms","measurement_median_ms"),
                           ("pooled_repetitions_p95_ms","measurement_p95_ms"),
                           ("scene_averaged_common_fraction_median","common_fraction_median"),
                           ("mean_J","mean_J"),("mean_raw_J","mean_raw_J")]:
            assert abs(independent[left]-supplied[right]) < 1e-8, (mode,method,left,right)
recovery = read("replay_recovery.json")
recovery_log = (ROOT/recovery["original_log_path"]).read_bytes()
assert hashlib.sha256(recovery_log).hexdigest() == recovery["original_log_sha256"]
assert hashlib.sha256(recovery["original_verifier_source"].encode()).hexdigest() == recovery["original_source_sha256"]
assert '"replayed_sampled_episodes": 16384, "steps": 1584734' in recovery_log.decode("utf-8")
assert "cli.read_text()" in recovery_log.decode("utf-8") and "UnicodeDecodeError" in recovery_log.decode("utf-8")
assert recovery["completed_replay_episodes"] == episodes
assert recovery["completed_replay_transitions"] == steps
verification = read("verification.json")
resource = read("resource_accounting.json")
assert verification["all_passed"] and verification["sampled_route_replays"] == episodes
assert verification["replayed_service_transitions"] == steps
assert verification["new_neural_inferences"] == verification["new_optimizer_updates"] == 0
assert verification["saved_plan_geometry_checks"] == len(verification["saved_plans"]) == 541
assert verification["verification_geometry_certificates_total"] == 544
assert verification["initial_attempt_additional_geometry_certificates"] == 543
for row in verification["saved_plans"]:
    path = (ROOT/row["file"]).resolve()
    assert path.is_relative_to(ROOT)
    assert row["valid"] and sha(path) == row["sha256"]
assert resource["all_verification_geometry_certificates"] == 544+543 == 1087
assert resource["all_stochastic_trajectories"] == sum(resource["stochastic_trajectories"].values()) == 16384+7 == 16391
assert resource["all_greedy_trajectories"] == sum(resource["greedy_trajectories"].values()) == 16384+432+64+2+12+192+1 == 17087
assert resource["greedy_trajectories"]["training_validation"] == 5*48+4*2*24
assert resource["greedy_trajectories"]["final_development_initial_and_selected"] == 24*2*2*2
assert abs(sum(r["preparation_s"] for r in history)-resource["training_input_preparation_wall_s"]) < 1e-7
original_updates = read("baseline_updates.json")
directional = []
for old, corrected in zip(original_updates,resource["baseline_replacement_directional_p"]):
    lower = old["candidate_mean_J"] < old["baseline_mean_J"]
    true_p = old["one_sided_p"] if lower else 1-old["one_sided_p"]
    assert corrected["candidate_lower"] == lower
    assert abs(corrected["improvement_direction_p"]-true_p) < 1e-12
    assert corrected["replaced"] == old["replaced"] == bool(lower and true_p < .05)
    directional.append(dict(update=old["update"],candidate_lower=lower,directional_improvement_p=true_p,replaced=old["replaced"]))
for mode in ["cold", "hot"]:
    assert analysis["primary"][mode]["passed"] is False
assert analysis["continuation_signal"]["passed"] is False
assert not (ROOT/"paper/PAPER_CLAIM_AUDIT.json").exists()
result = dict(schema="independent-saved-evidence-recalculation-v1", imports="stdlib and NumPy only; no project modules",
    counts=dict(c0_measurement_rows=len(reference_records), c2_measurement_rows=len(records), combined_rows=1152,
                development_scenes=24, selection_scenes=48, independent_test_scenes=0,
                optimizer_updates=len(history), optimizer_sampled_trajectories=episodes, sampled_service_transitions=steps,
                training_map_seeds=len(maps), distinct_training_task_seeds=len(task_seeds)),
    timing=summary, paired_degradation_percent=paired, selection_gain=selection, exact=exact,small_gaps=small_gaps,
    source_hash_checks=dict(imported_frozen_sources=old_hashes, execution_sources=execution_hashes,
                           protocol_and_history_and_route_digest_match=True),
    corrected_analysis_comparison="All method mean J/raw J, scene-averaged median/p95/common fractions and retained pooled medians/p95 match to <1e-8",
    recovery_receipt_checks=dict(original_source_and_log_hashes_match=True,completed_replay_line_matches=True,
        failure_at_cli_read_confirmed=True,provenance="Explicitly posthoc, not an automatic pre-error receipt"),
    final_verification_checks=dict(saved_plan_hashes_checked=541,final_geometry_certificates=544,
        initial_attempt_geometry_certificates=543,total_actual_geometry_certificates=1087,
        numeric_replays_not_repeated=True,reviewer_performed_new_route_replays=False),
    resource_arithmetic=dict(stochastic_trajectories=16391,greedy_trajectories=17087,total_neural_trajectories=33478,
        c1_counts_provenance="Posthoc suite history supplied by executor; final seven-test log/source inspected, no tests rerun by reviewer",
        corrected_directional_p=directional),
    evidence_sha256={name:sha(OUT/name) for name in ["analysis.json","verification.json","resource_accounting.json","replay_recovery.json","final_evaluation.json","small_exact.json"]},
    figure_sha256={name:sha(ROOT/"figures"/name) for name in ["quality_time.png","quality_time.pdf","learning_curve.png","learning_curve.pdf","CAPTIONS.md","latex_includes.tex"]},
    paper_claim_audit="unavailable; downstream paper-writing verdict provisional",
    timing_reporting_note="Protocol says timing repetitions averaged inside scene. Original analysis pools 48 repetitions; this review also reports 24 scene-averaged values. Gates have same truth values.")
(TRACE/"independent_recalculation.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(dict(counts=result["counts"], paired_detour16=paired["detour16"], selection_gain=selection,
                     time_ratios={mode:summary[mode]["time_ratios_model_detour16"] for mode in summary},
                     dominators={mode:summary[mode]["dominators_mean_J_and_median_time"] for mode in summary},
                     exact=exact),ensure_ascii=False,indent=2))
