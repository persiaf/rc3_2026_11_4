"""Independent, static S3 result audit. No project imports or scientific calls.

Reads saved JSON/CSV and binary hashes only. The only outputs are in this trace.
No torch/model loading, scene generation, navigation, H, or optimizer execution.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
RES = ROOT / "rc3_s3_scope_results_20261109/results"
SCOPE = ROOT / "rc3_s3_connectivity_20261109"
RUN = SCOPE / "runs/rc3_v3_global_local_20261106_01/scope_r1"
SEEDS = (11622026, 11622027, 11622028)
SIZES = ("4x3", "6x6", "8x8")
METHODS = ("Fixed3_endpoint_adapter", "Detour16_endpoint_adapter", "untrained",
           "frozen_supervised", "continued_SL", "RL_raw", "RL_plus_H")
ROLES = {"untrained": "untrained", "frozen_supervised": "SL256",
         "continued_SL": "continuedSL512", "RL_raw": "RL512", "RL_plus_H": "RL512"}
METRICS = ("J", "L", "K", "raw_J", "raw_L", "raw_K")
EXPECTED_BINDING = "311a58bd867e95c852ec3246c54a398468eb526db391fbb6d12239c6a3379949"
START = time.perf_counter()
START_UTC = datetime.now(timezone.utc).isoformat()
checks = Counter()


def require(ok, message):
    if not ok:
        raise AssertionError(message)
    checks[message.split(":", 1)[0]] += 1


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def identity(x):
    return hashlib.sha256(canonical(x)).hexdigest()


def digest(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def near(a, b, context):
    require(math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-10), context)


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
                            + "\n", encoding="utf-8")


def csvwrite(name, rows):
    if rows:
        with (OUT / name).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


registry = read(ROOT / "rc3_global_local_execution_v3/REGISTRY.json")
protocol = read(ROOT / "rc3_global_local_execution_v3/V3_EXECUTION_CANDIDATE.json")
scientific = read(RES / "SCIENTIFIC_ANALYSIS.json")
account = read(RES / "RESOURCE_ACCOUNTING.json")
baseline = read(SCOPE / "BASELINE_IMPORT.json")
binding = read(SCOPE / "SOURCE_BINDING.json")
amendment = read(SCOPE / "INPUT_SCOPE_AMENDMENT.json")
recovery = read(SCOPE / "RECOVERY_DECISION.json")
require(digest(SCOPE / "SOURCE_BINDING.json") == EXPECTED_BINDING, "source-binding byte hash")
fixed_hashes = {
    "rc3_global_local_execution_v3/V3_EXECUTION_CANDIDATE.json": binding["protocol_sha256"],
    "rc3_global_local_execution_v3/REGISTRY.json": binding["registry_sha256"],
    "rc3_s3_evaluator/S3_MODEL_INDEX.json": binding["model_index_sha256"],
    "rc3_navigation_compat_20261106/SOURCE_BINDING_UPDATED.json": binding["training_source_sha256"],
    "rc3_s3_evaluator/S3_SOURCE_BINDING.json": binding["original_evaluator_sha256"],
    "rc3_s3_connectivity_20261109/INPUT_SCOPE_AMENDMENT.json": binding["input_contract_amendment_sha256"],
    "rc3_s3_connectivity_20261109/FROZEN_EVIDENCE_INDEX.json": binding["frozen_evidence_index_sha256"],
    "rc3_s3_connectivity_20261109/BASELINE_IMPORT.json": binding["baseline_import_sha256"],
    "rc3_s3_results_20261108/authorizations/S3_AUTHORIZATION.json": recovery["original_stage_authorization_sha256"],
}
for p, h in fixed_hashes.items():
    require(digest(ROOT / p) == h, "scientific lineage hash:" + p)
for p, h in binding["source_sha256"].items():
    require(digest(ROOT / p) == h, "bound execution source hash:" + p)
frozen = read(SCOPE / "FROZEN_EVIDENCE_INDEX.json")["files"]
for p, entry in frozen.items():
    require(digest(ROOT / p) == entry["sha256"], "frozen evidence byte hash:" + p)
    require((ROOT / p).stat().st_size == entry["bytes"], "frozen evidence byte size:" + p)
modelaudit = read(RES / "MODEL_SOURCE_AUDIT.json")
for entry in modelaudit["fixed_model_source_files"]:
    require(digest(ROOT / entry["path"]) == entry["sha256"], "fixed model static hash:" + entry["path"])
models = read(ROOT / "rc3_s3_evaluator/S3_MODEL_INDEX.json")["models"]
require(len(modelaudit["fixed_model_source_files"]) == 36, "36 model source files")
require(not amendment["original_two_failed_slots_refilled"], "amendment preserves old missing slots")
require(amendment["scientific_input_admission_changed"], "input contract scientifically amended")

with (RES / "REQUEST_METRICS.csv").open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
index = {r["request_id"]: r for r in rows}
require(len(index) == len(rows) == 15874, "unique saved CSV requests")
states = read(RES / "REQUEST_STATE_INDEX.json")
state_index = {q["request"]["id"]: q for q in states["requests"]}
require(len(state_index) == len(states["requests"]) == 15876, "unique registered states")
expected = {}
nominal_cases = {}
for c in registry["S3"]:
    cid = f'S3_{c["case"]:03d}'
    nominal_cases[cid] = c
    for seed in SEEDS:
        for kind in ("quality", "cold", "hot"):
            for rep in range(1 if kind == "quality" else 3):
                for method in METHODS:
                    q = dict(case_id=cid, seed=seed, kind=kind, repeat=rep, method=method)
                    rid = identity(q)
                    expected[rid] = dict(q, id=rid)
require(set(expected) == set(state_index), "exact immutable declared request grid")
for rid, state in state_index.items():
    require(state["request"] == expected[rid], "declared request identity:" + rid)
    require((rid in index) == (state["state"] == "COMPLETED"), "state CSV consistency:" + rid)
missing = sorted(set(expected) - set(index))
require(missing == sorted(baseline["missing_failure_ids"]), "exact two old missing requests unfilled")
require(len(missing) == 2, "two retained missing requests")
require(set(scientific["missing_ids"]) == set(missing), "executor missingness matches")
coverage = {}
for kind in ("quality", "cold", "hot"):
    coverage[kind] = dict(expected=sum(q["kind"] == kind for q in expected.values()),
                          observed=sum(r["kind"] == kind for r in rows),
                          missing=[expected[rid] for rid in missing if expected[rid]["kind"] == kind])

artifacts = read(RES / "RESULT_ARTIFACT_INDEX.json")["artifacts"]
artifact_index = {a["path"]: a for a in artifacts}
require(len(artifact_index) == len(artifacts) == len(rows), "artifact index unique coverage")
stable = {}
raw_by_case_seed = {}
case_sources = {}
sample_routes = []
sample_keys = set()
assignments = Counter()
for r in rows:
    rid = r["request_id"]
    q = expected[rid]
    for key in ("case_id", "method", "kind"):
        require(r[key] == q[key], "CSV request field consistency:" + rid)
    require(int(r["seed"]) == q["seed"] and int(r["repeat"]) == q["repeat"], "CSV seed repeat identity:" + rid)
    a = artifact_index[r["artifact_path"]]
    p = ROOT / r["artifact_path"]
    payload = p.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == a["sha256"] == r["artifact_sha256"], "saved result byte hash:" + rid)
    require(len(payload) == a["bytes"], "saved result byte size:" + rid)
    d = json.loads(payload)
    require(d["request"] == q, "raw artifact request identity:" + rid)
    require(d["case_id"] == r["case_id"] and d["physical_key"] == r["physical_key"], "raw artifact case physical identity:" + rid)
    for k in METRICS:
        near(r[k], d["metrics"][k], "CSV saved metrics:" + rid)
    for k in ("raw_route", "route"):
        require(identity(d[k]) == d[k + "_sha256"] == r[k + "_sha256"], "actual saved route hash:" + rid)
    near(r["H_gain_J"], float(r["raw_J"]) - float(r["J"]), "H gain arithmetic:" + rid)
    near(r["H_gain_J"], d["H_gain_J"], "CSV H gain:" + rid)
    require(r["valid"] == "True" and d["valid"] is True, "all saved results valid:" + rid)
    require(math.isfinite(float(r["complete_time_s"])) and float(r["complete_time_s"]) > 0, "complete finite positive clock:" + rid)
    plan = d["plan"]
    val = plan["validation"]
    require(val["valid_model_plan"] and val["all_tasks_served"] and val["effective_same_arm_F_and_A"]
            and val["endpoint_constraints_satisfied"] and val["collision_count"] == 0
            and not val["errors"], "saved independent validator flags:" + rid)
    require(not plan["physical_joint_trajectory_certified"], "physical trajectory not falsely certified:" + rid)
    target = plan["target_task_ids_global"]
    assigned = plan["assignments"]
    require(set(target) == {a["task_id"] for a in assigned} and len(target) == len(assigned), "saved target coverage:" + rid)
    require(len(target) == int(r["task_count"]) == int(r["tasks_covered"]), "CSV saved target count:" + rid)
    require(r["same_arm_assignments_valid"] == "True", "CSV same-arm status:" + rid)
    require(set(a["candidate_id"] for a in assigned) <= set(plan["selected_candidate_ids"]), "assigned candidates are visited:" + rid)
    require(all(a["arm"] in ("L", "R") for a in assigned), "saved arm symbols:" + rid)
    assignments.update(a["arm"] for a in assigned)
    near(plan["route_length_m"], r["L"], "saved plan CSV L:" + rid)
    near(sum(plan["route_leg_costs_m"]), r["L"], "saved route legs L:" + rid)
    near(plan["objective_length_plus_stops_m"], r["J"], "saved plan CSV J:" + rid)
    near(float(r["L"]) + .75 * float(r["K"]), r["J"], "objective decomposition:" + rid)
    require(int(float(r["K"])) == len(plan["service_stops"]), "saved stops K:" + rid)
    require(plan["navigation_global_nodes"][0] == plan["entry_global_node"]
            and plan["navigation_global_nodes"][-1] == plan["exit_global_node"], "saved route endpoints:" + rid)
    source = d["source"]
    require(source["view_identity"] == r["input_identity"] and source["D_sha256"] == r["D_sha256"], "saved CSV input identity:" + rid)
    c = nominal_cases[r["case_id"]]
    require(all(source["content"]["spec"][k] == c[k] for k in c), "registry saved target spec:" + rid)
    require(source["full_parent_tables"] == 100 and source["full_parent_geometry"] == r["physical_key"], "full parent retained:" + rid)
    require(d["group"] == r["group"] and "x".join(map(str, d["size"])) == r["size"] and d["endpoint"] == r["endpoint"], "raw CSV strata:" + rid)
    stable_input = (source["content"], source["F_sha256"], source["A_sha256"], source["D_sha256"],
                    source["view_identity"], source["global_task_ids"], source["global_candidate_ids"])
    cid = r["case_id"]
    require(cid not in case_sources or case_sources[cid] == stable_input, "same saved case input across all methods seeds revisions:" + rid)
    case_sources[cid] = stable_input
    if "scope_evidence" in source:
        e = source["scope_evidence"]
        require(e["full_parent_check"] == "DIAGNOSTIC" and e["target_check"] == "HARD"
                and not e["target_unservable_task_ids"], "scope actual target hard constraint:" + rid)
    if q["method"] in ROLES:
        require(d["weight"] == models[str(q["seed"])][ROLES[q["method"]]], "all neural request matched seed fixed endpoint:" + rid)
    key = (cid, q["seed"], q["method"])
    value = (d["raw_route_sha256"], d["route_sha256"], tuple(d["metrics"][k] for k in METRICS))
    require(key not in stable or stable[key] == value, "repeats same frozen route costs:" + rid)
    stable[key] = value
    if q["method"] in ("RL_raw", "RL_plus_H"):
        pair = (cid, q["seed"])
        rawvalue = (d["raw_route"], tuple(d["metrics"][k] for k in ("raw_J", "raw_L", "raw_K")))
        require(pair not in raw_by_case_seed or raw_by_case_seed[pair] == rawvalue, "RL raw H actual original-route equality:" + rid)
        raw_by_case_seed[pair] = rawvalue
        if q["method"] == "RL_raw":
            require(d["raw_route"] == d["route"] and d["pipeline_counters"].get("full_H_calls", 0) == 0,
                    "RL raw no H attribution:" + rid)
        else:
            require(d["pipeline_counters"]["full_H_calls"] == 1, "RL plus H exactly one H:" + rid)
    sample_key = (r["group"], r["size"], r["method"], r["kind"], r["persistence_revision"])
    if sample_key not in sample_keys:
        sample_keys.add(sample_key)
        poly = plan["route_polyline"]
        poly_length = sum(math.hypot(y[0]-x[0], y[1]-x[1]) for x, y in zip(poly, poly[1:]))
        require(abs(poly_length-float(r["L"])) < 1e-7, "sample saved polyline length independent arithmetic:" + rid)
        sample_routes.append(dict(request_id=rid, artifact=r["artifact_path"], case=cid, seed=q["seed"],
                                  method=q["method"], kind=q["kind"], group=r["group"], size=r["size"],
                                  persistence_revision=r["persistence_revision"], J=float(r["J"]),
                                  L=float(r["L"]), independent_polyline_length=poly_length,
                                  K=float(r["K"]), route_sha256=r["route_sha256"]))

require(len(case_sources) == 108, "108 exact saved cases")
require(len(stable) == 2268, "2268 deterministic case seed method units")
oldids = set(baseline["baselines"][str(SEEDS[0])]["terminal_ids"])
require(len(oldids & set(index)) == 2057 and set(missing) < oldids, "2057 old successes two old missing carried")
require(sum(r["persistence_revision"] == EXPECTED_BINDING for r in rows) == 13817, "13817 new success artifacts")
print("Static artifacts, models, sources and frozen bytes verified", flush=True)

# Independently verify every new append-only journal and timing/source ledger.
journals = []
total_new = 0
for seed in SEEDS:
    folder = RUN / str(seed)
    prev, count, starts, finishes, errs = "0"*64, 0, {}, {}, []
    session_clocks = []
    for line in (folder / "journal.jsonl").open(encoding="utf-8"):
        e = json.loads(line)
        h = e.pop("sha256")
        require(e["previous"] == prev and e["sequence"] == count and identity(e) == h,
                "new journal append chain:" + str(seed))
        prev, count = h, count + 1
        if e["event"] == "request_start":
            q = e["request"]
            require(q == expected[q["id"]] and q["id"] not in starts and q["id"] not in oldids,
                    "new unique request paid before work:" + q["id"])
            starts[q["id"]] = q
        elif e["event"] == "request_complete":
            rid = e["id"]
            require(rid in starts and rid not in finishes, "completion has unique new debit:" + rid)
            require(e["artifact_sha256"] == index[rid]["artifact_sha256"], "new ledger artifact CSV hash:" + rid)
            near(e["complete_time_s"], index[rid]["complete_time_s"], "new ledger complete CSV clock:" + rid)
            finishes[rid] = e
        elif e["event"] == "request_error":
            errs.append(e)
        elif e["event"] == "session_close":
            session_clocks.append(e["elapsed_s"])
    head = read(folder / "HEAD.json")
    require(head["sequence"] == count and head["sha256"] == prev, "new journal closed head:" + str(seed))
    require(set(starts) == set(finishes) and not errs, "all new debits complete no failures:" + str(seed))
    a = account["actual_by_seed"][str(seed)]
    require(len(starts) == a["new_debits"] == a["new_success"], "new accounting exact per seed:" + str(seed))
    require(a["total_debits"] == 5292 and a["charged_wall_s"] <= 14400, "registered per seed budgets kept:" + str(seed))
    seal = read(folder / "SCHEDULE_SEAL.json")
    require(seal["schedule_sha256"] == digest(folder / "SCHEDULE_COMPLETE.json"), "schedule completion byte seal:" + str(seed))
    finals = []
    for session in a["sessions"]:
        fp = ROOT / session["path"] / "FINAL.json"
        require(digest(fp) == session["final_sha256"], "session final byte seal:" + str(seed))
        finals.append(read(fp))
    near(sum(x["elapsed_s"] for x in finals), a["measured_new_wall_s"], "sealed exact new wall:" + str(seed))
    near(a["measured_new_wall_s"] + baseline["baselines"][str(seed)]["wall_carry_s"],
         a["charged_wall_s"], "inherited wall cannot reset:" + str(seed))
    require(seal["final_sha256"] == a["sessions"][-1]["final_sha256"], "schedule final hash binding:" + str(seed))
    total_new += len(starts)
    journals.append(dict(seed=seed, journal_events=count, journal_tail=prev, new_debits=len(starts),
                         successes=len(finishes), new_failures=len(errs),
                         journal_inner_session_wall_s=sum(session_clocks),
                         sealed_measured_new_wall_s=a["measured_new_wall_s"],
                         charged_cumulative_wall_s=a["charged_wall_s"]))
require(total_new == 13817, "13817 new paid requests independent journal count")

# Independently recompute frozen aggregation. Only quality rows enter quality CIs.
bykey = {(r["case_id"], int(r["seed"]), r["method"], r["kind"], int(r["repeat"])):r for r in rows}
cases = {}
for r in rows:
    cases[r["case_id"]] = {k:r[k] for k in ("case_id", "physical_key", "group", "size", "endpoint")}
quality, deployment, all_comparisons, method_table, timing_revisions, endpoints = [], [], [], [], [], []
numeric_delta = []


def boot(v, ix):
    v = np.asarray(v, dtype=float)
    b = v[ix].mean(axis=1)
    return dict(mean=float(v.mean()), lower95=float(np.quantile(b,.025)),
                upper95=float(np.quantile(b,.975)), upper_one_sided95=float(np.quantile(b,.95)))


def stats(rr):
    t = np.array([float(r["complete_time_s"]) for r in rr])
    return dict(n=len(t), median_s=float(np.median(t)), p95_s=float(np.quantile(t,.95)))


for group, nparents in (("seen", 6), ("unseen", 12)):
    for size in SIZES:
        cc = [c for c in cases.values() if c["group"] == group and c["size"] == size]
        parents = sorted({c["physical_key"] for c in cc})
        require(len(parents) == nparents and len(cc) == 2*nparents, "true physical parent counts:" + group+size)
        require(all(len([c for c in cc if c["physical_key"] == p]) == 2 for p in parents), "two endpoints per physical parent:" + group+size)
        cids = {c["case_id"] for c in cc}
        sub = [r for r in rows if r["case_id"] in cids]
        qmissing = [q for q in coverage["quality"]["missing"] if q["case_id"] in cids]
        qcell = dict(group=group, size=size, physical_parents=nparents,
                     status="INCOMPLETE" if qmissing else "COMPLETE", passed=None, comparisons=[])
        pool = None
        ix = np.random.Generator(np.random.PCG64(11632026)).integers(nparents, size=(20000,nparents))
        if not qmissing:
            costs = {}
            for m in METHODS:
                costs[m] = np.array([[np.mean([float(bykey[c["case_id"],seed,m,"quality",0]["J"])
                                                for c in cc if c["physical_key"] == p])
                                      for seed in SEEDS] for p in parents])
            pool = {m:v.mean(axis=1) for m,v in costs.items()}
            for actor, control in [("RL_plus_H",m) for m in METHODS if m != "RL_plus_H"] + [("RL_raw","frozen_supervised"),("RL_raw","continued_SL")]:
                g = (pool[control]-pool[actor])/pool[control]
                b = boot(g, ix)
                sg = [float(((costs[control][:,i]-costs[actor][:,i])/costs[control][:,i]).mean()) for i in range(3)]
                comparison = dict(group=group,size=size,actor=actor,control=control,**b,seed_gains=sg)
                all_comparisons.append(comparison)
                if actor == "RL_plus_H" and control in ("frozen_supervised","continued_SL"):
                    comparison = dict(comparison, passed=b["mean"] >= .005 and b["lower95"] > 0 and all(x>0 for x in sg))
                    qcell["comparisons"].append(comparison)
            qcell["passed"] = all(c["passed"] for c in qcell["comparisons"])
            original = next(c for c in scientific["quality_cells"] if c["group"] == group and "x".join(map(str,c["size"])) == size)
            for c in qcell["comparisons"]:
                original_c = next(x for x in original["controls"] if x["control"] == c["control"])
                for k in ("mean","lower95","upper95","upper_one_sided95"):
                    near(c[k], original_c[k], "quality numeric executor agreement:" + group+size+k)
                    numeric_delta.append(abs(c[k]-original_c[k]))
                for a,b in zip(c["seed_gains"],original_c["seed_gains"]):near(a,b,"quality seed executor agreement:"+group+size)
                require(c["passed"] == original_c["passed"], "quality gate executor agreement:" + group+size)
        quality.append(qcell)
        for m in METHODS:
            qrows = [r for r in sub if r["method"] == m and r["kind"] == "quality"]
            table = dict(group=group,size=size,method=m,quality_observed=len(qrows),quality_expected=6*nparents)
            table.update({k:float(np.mean([float(r[k]) for r in qrows])) for k in (*METRICS,"H_gain_J")})
            for cache in ("cold","hot"):
                rr = [r for r in sub if r["method"] == m and r["kind"] == cache]
                table.update({cache+"_"+k:v for k,v in stats(rr).items()})
                for ver in sorted({r["persistence_revision"] for r in rr}):
                    vv = [r for r in rr if r["persistence_revision"] == ver]
                    timing_revisions.append(dict(group=group,size=size,method=m,cache=cache,persistence_revision=ver,**stats(vv)))
            method_table.append(table)
            for ep in ("closed_depot","open_first_last_connected"):
                er = [r for r in qrows if r["endpoint"] == ep]
                endpoints.append(dict(group=group,size=size,method=m,endpoint=ep,n=len(er),
                                      **{k:float(np.mean([float(r[k]) for r in er])) for k in METRICS}))
        for cache in ("cold","hot"):
            tm = {m:stats([r for r in sub if r["method"]==m and r["kind"]==cache]) for m in METHODS}
            tmissing = [q for q in coverage[cache]["missing"] if q["case_id"] in cids]
            complete = not qmissing and not tmissing
            tcell = dict(group=group,size=size,cache=cache,status="COMPLETE" if complete else "INCOMPLETE",
                         quality_missing=qmissing,timing_missing=tmissing,passed=None,timing=tm,
                         descriptive_median_ratio=tm["RL_plus_H"]["median_s"]/tm["Detour16_endpoint_adapter"]["median_s"],
                         descriptive_p95_ratio=tm["RL_plus_H"]["p95_s"]/tm["Detour16_endpoint_adapter"]["p95_s"])
            if complete:
                d = boot((pool["RL_plus_H"]-pool["Detour16_endpoint_adapter"])/pool["Detour16_endpoint_adapter"],ix)
                fixed = (pool["Fixed3_endpoint_adapter"].mean(),tm["Fixed3_endpoint_adapter"]["median_s"],tm["Fixed3_endpoint_adapter"]["p95_s"])
                rl = (pool["RL_plus_H"].mean(),tm["RL_plus_H"]["median_s"],tm["RL_plus_H"]["p95_s"])
                dom = all(a<=b for a,b in zip(fixed,rl)) and any(a<b for a,b in zip(fixed,rl))
                med,p95 = tcell["descriptive_median_ratio"],tcell["descriptive_p95_ratio"]
                tcell.update(degradation=d,Fixed3_dominates=bool(dom),
                             passed=d["upper_one_sided95"]<=.005 and med<=.7 and p95<=1 and not dom)
                old = next(c for c in scientific["deployment_cells"] if c["group"]==group and "x".join(map(str,c["size"]))==size and c["cache"]==cache)
                for k in d:near(d[k],old["degradation"][k],"deployment quality executor agreement:"+group+size+cache+k)
                near(med,old["median_ratio"],"deployment median executor agreement:"+group+size+cache)
                near(p95,old["p95_ratio"],"deployment p95 executor agreement:"+group+size+cache)
                require(bool(tcell["passed"]) == old["passed"] and bool(dom)==old["Fixed3_dominates"],"deployment gate executor agreement:"+group+size+cache)
            deployment.append(tcell)

csvwrite("independent_method_quality_time.csv", method_table)
csvwrite("independent_timing_by_revision.csv", timing_revisions)
csvwrite("independent_endpoint_quality.csv", endpoints)
csvwrite("independent_route_samples.csv", sample_routes)
write("independent_matched_comparisons.json", all_comparisons)
write("independent_quality_and_deployment.json", dict(quality=quality,deployment=deployment))
result = dict(
    audit="PASS", audit_started_utc=START_UTC, audit_finished_utc=datetime.now(timezone.utc).isoformat(),
    audit_wall_s=time.perf_counter()-START, python=sys.version, numpy=np.__version__,
    scientific_calls=0, model_deserializations=0, project_module_imports=0,
    source_binding_sha256=EXPECTED_BINDING, bound_source_files_hashed=len(binding["source_sha256"]),
    frozen_evidence_files_hashed=len(frozen), fixed_model_files_hashed=36,
    saved_artifacts_hashed_and_metrics_routes_validated=len(rows), saved_assignment_symbols=dict(assignments),
    constraint_limit="Saved validator evidence, coverage and route arithmetic independently checked; no rerun of proxy F/A construction, IK or navigation. The earlier executor's full F&A source audit remains separately identified.",
    polyline_route_samples_independently_summed=len(sample_routes), coverage=coverage,
    missing_requests=[expected[rid] for rid in missing], inherited_debits=2059,inherited_success=2057,
    inherited_missing=2,new_debits=total_new,new_success=total_new,new_failures=0,
    total_debits=total_new+2059,total_success=len(rows),
    physical_parents=dict(seen=6,unseen=12), optimization_repeats=3,
    quality_cells=dict(complete=sum(c["status"]=="COMPLETE" for c in quality),
                       failed=sum(c["passed"] is False for c in quality),
                       passed=sum(c["passed"] is True for c in quality),
                       incomplete=sum(c["status"]=="INCOMPLETE" for c in quality)),
    deployment_cells=dict(complete=sum(c["status"]=="COMPLETE" for c in deployment),
                          failed=sum(c["passed"] is False for c in deployment),
                          passed=sum(c["passed"] is True for c in deployment),
                          incomplete=sum(c["status"]=="INCOMPLETE" for c in deployment)),
    overall_frozen_grid="INCOMPLETE_NO_OVERALL_PASS; complete cells already fail both AND classes",
    max_quality_numeric_difference=max(numeric_delta),journals=journals,
    total_charged_stage_wall_s=sum(j["charged_cumulative_wall_s"] for j in journals),
    inherited_original_exact_wall_s=None,inherited_conservative_reservation_s=592,
    checked_saved_case_inputs=108,deterministic_case_seed_method_units=len(stable),
    repeat_RL_raw_route_pairs_checked=len(raw_by_case_seed),
    model_source_lineage="All neural request metadata exactly matches its own fixed seed endpoint, all 36 static source files hash verified; no checkpoint deserialization.",
    paper_claim_audit="unavailable" if not (ROOT/"paper/PAPER_CLAIM_AUDIT.json").exists() else "present",
    check_counts=dict(checks), input_hashes={str(p.relative_to(ROOT)):digest(p) for p in
        [RES/"REQUEST_METRICS.csv",RES/"SCIENTIFIC_ANALYSIS.json",RES/"RESOURCE_ACCOUNTING.json",RES/"REQUEST_STATE_INDEX.json",RES/"RESULT_ARTIFACT_INDEX.json"]})
write("independent_audit_results.json",result)
print(json.dumps({k:v for k,v in result.items() if k not in ("check_counts","input_hashes","journals")},ensure_ascii=False,indent=2))
