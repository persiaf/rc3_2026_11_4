"""Independent review of saved D5 revision evidence only; no project imports."""
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path.cwd()
RUN = ROOT / "rc3_d5_revision_20261112"
OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archived(pointer):
    with (ROOT / pointer["path"]).open("rb") as stream:
        stream.seek(pointer["offset"])
        blob = stream.read(pointer["compressed_bytes"])
    raw = gzip.decompress(blob)
    assert hashlib.sha256(raw).hexdigest() == pointer["raw_sha256"]
    return json.loads(raw)


def mean(xs):
    xs = list(xs)
    return math.fsum(xs) / len(xs)


def percentile(xs, q):
    xs = sorted(xs)
    index = (len(xs) - 1) * q
    low = math.floor(index)
    high = math.ceil(index)
    return xs[low] + (xs[high] - xs[low]) * (index - low)


def gains(candidate, control, metric="J", group=None):
    a = {x["case_id"]: x for x in candidate if group is None or x["group"] == group}
    b = {x["case_id"]: x for x in control if group is None or x["group"] == group}
    assert set(a) == set(b)
    parents = collections.defaultdict(list)
    for key, row in a.items():
        assert row["valid"] and b[key]["valid"]
        assert row["physical_key"] == b[key]["physical_key"]
        parents[row["physical_key"]].append(key)
    paired = {}
    for parent, keys in sorted(parents.items()):
        assert len(keys) == 6
        base = mean(b[k]["metrics"][metric] for k in keys)
        cand = mean(a[k]["metrics"][metric] for k in keys)
        paired[parent] = {"candidate_parent_mean": cand, "control_parent_mean": base,
                          "gain": (base - cand) / base}
    values = [x["gain"] for x in paired.values()]
    return {"parent_count": len(paired), "gain": mean(values), "parent_gains": paired,
            "positive_parents": sum(v > 1e-12 for v in values),
            "negative_parents": sum(v < -1e-12 for v in values)}


protocol = read(RUN / "REVISION_PROTOCOL.json")
expected = {x["id"]: x for x in protocol["evaluation"]["development"]}
grids = {}
records = {}
summary = {}
errors = []
raw_h_checks = 0
for path in sorted((RUN / "evaluations/development").glob("*.json")):
    report = read(path)
    rows = report["rows"]
    assert report["complete"] and len(rows) == 66
    assert {x["case_id"] for x in rows} == set(expected)
    assert len({x["case_id"] for x in rows}) == len(rows)
    parents = collections.Counter(x["physical_key"] for x in rows)
    assert len(parents) == 11 and set(parents.values()) == {6}
    full_times = []
    recs = {}
    for row in rows:
        case = expected[row["case_id"]]
        assert row["valid"] and row["physical_key"] == case["physical_key"]
        assert row["group"] == case["group"]
        x = archived(row["record"])
        assert x["valid"] and x["task_id"] == row["case_id"]
        assert x["physical_key"] == row["physical_key"]
        assert x["method"] == row["method"] and x["spec"] == case["spec"]
        assert x["metrics"] == row["metrics"]
        for prefix, plan_name in [("raw_", "raw_plan"), ("", "plan")]:
            m = row["metrics"]
            plan = x[plan_name]
            cert = plan["validation"]
            assert cert["valid_model_plan"] and cert["all_tasks_served"]
            assert cert["endpoint_constraints_satisfied"] and cert["collision_count"] == 0
            assert not cert["physical_joint_trajectory_certified"]
            assert not plan["unserved_task_ids"]
            assert abs(m[prefix + "J"] - (m[prefix + "L"] + .75 * m[prefix + "K"])) < 1e-7
            assert abs(m[prefix + "J"] - plan["objective_length_plus_stops_m"]) < 1e-7
            assert abs(m[prefix + "L"] - plan["route_length_m"]) < 1e-7
            assert abs(m[prefix + "L"] - cert["independent_polyline_length_m"]) < 1e-7
            assert m[prefix + "K"] == len(plan["visit_order_candidate_ids"])
            raw_h_checks += 1
        if row["method"] in ("Fixed3", "Detour16"):
            assert "Native final" in x["raw_semantics"]
            assert x["raw_route"] == x["route"]
        else:
            assert x["H_identity"] == protocol["H_identity"]
            assert abs(x["complete_return"] + x["metrics"]["J"]) < 1e-7
            assert abs(x["reward_sum"] + x["metrics"]["J"]) < 1e-7
        recs[row["case_id"]] = x
        assert row["full_request_s"] > 0
        full_times.append(row["full_request_s"])
    grids[path.stem] = rows
    records[path.stem] = recs
    summary[path.stem] = {"requests": len(rows), "success_rate": 1.0,
                          "parent_count": len(parents),
                          "mean_metrics": {k: mean(x["metrics"][k] for x in rows)
                                           for k in ["J", "L", "K", "raw_J", "raw_L", "raw_K"]},
                          "full_request_p50_s": percentile(full_times, .5),
                          "full_request_p95_s": percentile(full_times, .95)}

comparisons = {}
groups = ["trained_parent_new_tasks_exposed_selection", "independent_parent_exposed_development"]
for step in [1024, 2048, 4096]:
    pg = grids[f"2026111101_D5-PG_{step}"]
    for control in ["baseline_FrozenSL", f"2026111101_D5-SL-U_{step}", "baseline_Fixed3", "baseline_Detour16"]:
        key = f"D5-PG_{step}_vs_{control}"
        comparisons[key] = {m: gains(pg, grids[control], m) for m in ["J", "L", "K", "raw_J"]}
        comparisons[key]["groups"] = {group: gains(pg, grids[control], "J", group) for group in groups}
comparisons["SL-U_4096_vs_FrozenSL"] = {m: gains(grids["2026111101_D5-SL-U_4096"], grids["baseline_FrozenSL"], m)
                                       for m in ["J", "L", "K", "raw_J"]}
pg = records["2026111101_D5-PG_4096"]
base = records["baseline_FrozenSL"]
route_changes = {}
for label, key in [("raw", "raw_plan"), ("H", "plan")]:
    pairs = [(pg[k][key]["visit_order_candidate_ids"], base[k][key]["visit_order_candidate_ids"]) for k in pg]
    route_changes[label] = {"stop_set_changed": sum(set(a) != set(b) for a, b in pairs),
                            "visit_order_changed": sum(a != b for a, b in pairs),
                            "same_set_order_changed": sum(set(a) == set(b) and a != b for a, b in pairs)}

local = read(RUN / "ORIGINAL_LOCAL_TEACHERS.json")
local_entries = {x["id"]: x for x in local["records"]}
assert len(local_entries) == 256
assert all(sha(ROOT / x["path"]) == x["sha256"] for x in local["files"])
fullmf = read(RUN / "FULL_TEACHER_MANIFEST.json")
full_registration = {x["id"]: x for x in read(RUN / "FULL_TASK_REGISTRATION.json")["records"]}
assert len(fullmf["records"]) == 96
teacher_sizes = collections.Counter()
for tid, entry in fullmf["records"].items():
    path = ROOT / entry["path"]
    assert sha(path) == entry["sha256"]
    with gzip.open(path, "rt", encoding="utf-8") as f:
        label = json.load(f)
    assert label["replay_complete"] and label["H_identity"] == protocol["H_identity"]
    assert label["problem_identity"] == entry["problem_identity"]
    assert label["physical_key"] == full_registration[tid]["physical_key"]
    assert label["spec"] == full_registration[tid]["spec"]
    assert label["plan"]["validation"]["valid_model_plan"]
    assert all(x["legal"] for x in label["environment_replay"])
    assert label["environment_replay"][-1]["remaining"] == 0
    teacher_sizes[(label["spec"]["width"], label["spec"]["height"])] += 1
schedule_counts = {}
for seed in protocol["seeds"]:
    schedule = read(RUN / "schedules" / f"{seed}.json")["batches"]
    assert len(schedule) == 4096
    counts = collections.Counter()
    totals = collections.Counter()
    for i, batch in enumerate(schedule):
        # Episode IDs must be unique; repeated task IDs at an epoch boundary
        # are allowed by the registered immutable task schedule.
        assert len(batch) == 8
        assert all(x in full_registration for x in batch[:4])
        assert all(x in local_entries for x in batch[4:])
        local_sizes = collections.Counter((local_entries[x]["spec"]["width"], local_entries[x]["spec"]["height"]) for x in batch[4:])
        assert tuple(local_sizes[(w, h)] for w, h in [(4, 3), (6, 6), (8, 8)]) == tuple(protocol["optimization"]["local_rotation"][i % 3])
        counts.update(batch)
        totals["full"] += 4
        totals.update({f"{w}x{h}": n for (w, h), n in local_sizes.items()})
    schedule_counts[str(seed)] = counts
    print("schedule",seed,dict(totals))
assert all(schedule_counts[str(protocol["seeds"][0])] == schedule_counts[str(s)] for s in protocol["seeds"])
stream_updates = {}
for role in ["D5-PG", "D5-SL"]:
    with gzip.open(RUN / f"streams/2026111101/{role}/UPDATES.jsonl.gz", "rt", encoding="utf-8") as f:
        updates = [json.loads(line) for line in f if line.strip()]
    assert [x["cursor"] for x in updates] == list(range(1, 4097))
    assert updates[0]["before"] == "1817f58aa6506456ff12f0c5fe0a914e156aef8af319f4687dde90782f55a63e"
    assert all(updates[i]["before"] == updates[i-1]["after"] for i in range(1, len(updates)))
    for x in updates:
        assert x["critic_updates"] == 0
        assert [z["episode_id"] for z in x["loss_sources"]] == [f"2026111101:{role}:{x['cursor']}:{slot}" for slot in range(8)]
        assert not x["update_acceptance_guard"] and not x["permanent_SL_guard"] and not x["additional_A1_amplitude"]
    stream_updates[role] = {"actual_updates": len(updates), "logical_cursor_unique": True,
                            "weight_chain_continuous": True, "episodes_per_update": 8,
                            "initial_tensor_sha": updates[0]["before"], "final_tensor_sha": updates[-1]["after"]}

training_keys = {x["physical_key"] for x in full_registration.values()}
exposure = {}
for group in groups:
    cases = [x for x in expected.values() if x["group"] == group]
    keys = {x["physical_key"] for x in cases}
    exposure[group] = {"tasks": len(cases), "parents": len(keys), "parents_overlap_training": len(keys & training_keys),
                       "exposed_for_development_selection": True}

main1 = comparisons["D5-PG_4096_vs_baseline_FrozenSL"]
main2 = comparisons["D5-PG_4096_vs_2026111101_D5-SL-U_4096"]
passed = main1["J"]["gain"] >= .01 and main2["J"]["gain"] >= .005
passed = passed and all(x["gain"] >= 0 for c in [main1, main2] for x in c["groups"].values())
assert passed is False
decision = read(RUN / "P1_DECISION.json")
assert decision["passed"] == passed
assert abs(decision["comparisons"]["FrozenSL"]["J"]["mean"] - main1["J"]["gain"]) < 1e-12
assert abs(decision["comparisons"]["SL-U"]["J"]["mean"] - main2["J"]["gain"]) < 1e-12
output = {"scope": "Pure saved-evidence statistics and SHA/arithmetic checks; no project imports, model construction, forward, optimizer, planning, H, native, teacher or navigation",
          "evaluation_records_verified": sum(len(x) for x in grids.values()), "raw_H_plan_checks": raw_h_checks,
          "saved_plan_certificates_only": True, "grids": summary, "comparisons": comparisons,
          "route_changes_4096_vs_FrozenSL": route_changes,
          "data_admission": {"full": len(fullmf["records"]), "local": len(local_entries), "old_label_hashes_match": True,
                             "full_label_saved_replay_and_validation_match": True,
                             "cross_seed_exposure_multisets_identical": True, "same_per_seed_schedule_in_code": True},
          "stream_updates": stream_updates, "evaluation_exposure": exposure,
          "P1_recomputed": {"passed": passed, "status": "NO_DEVELOPMENT_SIGNAL"},
          "limitations": ["Saved validation certificates were checked for internal consistency; geometry/mask replay was not rerun.",
                          "Actual PG task-to-schedule and all 32768 masks are covered by the separate saved-output audit, not independently reexecuted here.",
                          "Full-request times are observed development measurements with unequal input cache warmth, not deployment latency comparisons."]}
(OUT / "REVIEWER_NUMERICAL_RECOMPUTATION.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"evaluations_verified": output["evaluation_records_verified"], "main_gain_FrozenSL": main1["J"]["gain"],
                  "main_gain_SL_U": main2["J"]["gain"], "P1_passed": passed,
                  "route_changes": route_changes, "exposure": exposure, "grid_summary": summary}, ensure_ascii=False, indent=2))
