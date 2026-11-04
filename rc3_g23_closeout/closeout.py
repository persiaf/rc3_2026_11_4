"""Adopt the G2.3 closeout review without executing the research project.

Only stdlib audit scripts supplied in the reviewed input are executed. They read
saved JSON/gzip/ZIP records; no project imports, model forward, branch replay,
training, planning, or planner timing is performed. All writes stay in this
separate closeout directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parent
EXPECTED_G23 = "02df5a5ff9dd896aed60b3fc579b8509855f48a60426b784d868250e013c3473"
REVIEW_PREFIX = "rc3_g23_closeout_review"


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    with path.open("rb") as stream:
        h = hashlib.file_digest(stream, "sha256") if hasattr(hashlib, "file_digest") else None
        if h is not None:
            return h.hexdigest()
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("All closeout writes must stay in the separate closeout directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def extract_checked(archive, destination, prefix):
    """Write only regular members in the named archive subtree; never overwrite differences."""
    destination = destination.resolve()
    if not destination.is_relative_to(ROOT):
        raise ValueError("Extraction must be within the closeout directory")
    members = {}
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            name = PurePosixPath(item.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in item.filename or ":" in item.filename:
                raise ValueError("Unsafe ZIP path")
            if not name.parts or name.parts[0] != prefix or item.is_dir():
                if item.is_dir():
                    continue
                raise ValueError("Unexpected ZIP root")
            if ((item.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError("Symlink ZIP member")
            if item.filename in members:
                raise ValueError("Duplicate ZIP member")
            data = z.read(item)
            target = (destination / Path(*name.parts)).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("ZIP member escaped destination")
            if target.exists():
                if target.read_bytes() != data:
                    raise ValueError("Refusing to replace differing extracted input: " + str(target))
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            members[item.filename] = {"sha256": sha_bytes(data), "bytes": len(data)}
    return members


def match_members(base, members):
    failures = []
    for relative, expected in members.items():
        path = base / relative
        if not path.is_file() or path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
            failures.append(relative)
    return {"checked": len(members), "matched": len(members) - len(failures), "failures": failures}


def normalize_results(data):
    # The supplied result report deliberately includes its absolute input location.
    data = json.loads(json.dumps(data))
    data["scope"].pop("source", None)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT.parent)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    review_zip = workspace / "rc3_g23_closeout_review.zip"
    source_zip = workspace / "rc3_rl_g23_delivery.zip"
    if sha(source_zip) != EXPECTED_G23:
        raise ValueError("Frozen G2.3 archive checksum mismatch")
    archive_paths = [workspace / p for p in (
        "rc3_v4_delivery.zip", "rc3_v5_delivery.zip", "rc3_v5_1_maintenance.zip",
        "rc3_rl_research/outputs/rc3_rl_research_g01.zip", "rc3_rl_g2_delivery.zip",
        "rc3_rl_g21_delivery.zip", "rc3_rl_g22_delivery.zip", "rc3_rl_g23_delivery.zip",
    )]
    before = {str(p.relative_to(workspace)).replace("\\", "/"): sha(p) for p in archive_paths}
    review_members = extract_checked(review_zip, ROOT / "inputs", REVIEW_PREFIX)
    review = ROOT / "inputs" / REVIEW_PREFIX
    supplied_manifest = read(review / "MANIFEST.sha256.json")["files"]
    expected_names = {REVIEW_PREFIX + "/MANIFEST.sha256.json"}
    for item in supplied_manifest:
        relative = REVIEW_PREFIX + "/" + item["path"]
        expected_names.add(relative)
        actual = review_members[relative]
        if actual != {"sha256": item["sha256"], "bytes": item["size_bytes"]}:
            raise ValueError("Review manifest mismatch: " + relative)
    if set(review_members) != expected_names:
        raise ValueError("Unlisted review members")
    source_members = extract_checked(source_zip, ROOT / "work" / "fresh_source", "rc3_rl_g23")
    fresh = ROOT / "work" / "fresh_source" / "rc3_rl_g23"
    source_manifest = read(fresh / "MANIFEST.sha256.json")["files"]
    if set(source_members) != set(source_manifest) | {"rc3_rl_g23/MANIFEST.sha256.json"}:
        raise ValueError("G2.3 archive member/manifest disagreement")
    for name, expected in source_manifest.items():
        if source_members[name] != expected:
            raise ValueError("G2.3 manifest mismatch: " + name)
    live_before = match_members(workspace, source_members)
    if live_before["failures"]:
        raise ValueError("Live frozen G2.3 differs from its delivery")
    jobs = [
        ("results", "results_audit/audit_g23_results.py", "results_audit/RESULTS_RECOMPUTED.json"),
        ("model_budget", "model_budget_audit/audit_model_budget.py", "model_budget_audit/MODEL_BUDGET_RECOMPUTED.json"),
    ]
    executions = []
    for label, script, reference in jobs:
        output = ROOT / "outputs" / (label + ".json")
        command = [sys.executable, "-B", str(review / script), "--source", str(fresh), "--output", str(output)]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        log_path = ROOT / "outputs" / (label + ".log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError("Read-only audit failed: " + label)
        actual, expected = read(output), read(review / reference)
        equal = normalize_results(actual) == normalize_results(expected) if label == "results" else actual == expected
        if not equal:
            raise ValueError("Recomputed statistics differ from supplied review: " + label)
        executions.append({"audit": label, "script": script, "script_sha256": sha(review / script),
                           "returncode": result.returncode, "output": str(output.relative_to(ROOT)).replace("\\", "/"),
                           "output_sha256": sha(output), "matches_supplied_statistics": equal,
                           "normalization": "remove scope.source only" if label == "results" else "none"})
    results = read(ROOT / "outputs/results.json")
    budget = read(ROOT / "outputs/model_budget.json")
    live_after = match_members(workspace, source_members)
    fresh_after = match_members(ROOT / "work/fresh_source", source_members)
    after = {str(p.relative_to(workspace)).replace("\\", "/"): sha(p) for p in archive_paths}
    if before != after or live_after["failures"] or fresh_after["failures"]:
        raise ValueError("Frozen content changed")
    receipt = {
        "schema": "g23-closeout-adoption-v1", "date": "2026-11-04",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "CLOSE_LOCAL_WINDOW_PLUS_STOP_AS_CURRENT_DRL_CORE",
        "future_construction_research": "CONDITIONAL_NOTE_ONLY_NOT_STARTED",
        "default_planners": ["Fixed3", "Detour"],
        "input_review": {"zip_sha256": sha(review_zip), "manifest_payloads_checked": len(supplied_manifest),
                         "archive_members": len(review_members)},
        "frozen_g23": {"zip_sha256": EXPECTED_G23, "manifest_payloads_checked": len(source_manifest),
                       "all_archive_members": len(source_members), "live_before": live_before,
                       "live_after": live_after, "fresh_extraction_after": fresh_after},
        "frozen_archives": {"before": before, "after": after, "all_unchanged": before == after},
        "executions": executions,
        "scope": {"new_training_updates": 0, "new_environment_queries": 0, "new_neural_inferences": 0,
                  "new_plans": 0, "new_performance_measurements": 0, "new_geometric_or_IK_checks": 0,
                  "project_code_imported": False, "checkpoint_pickle_executed": False,
                  "method": "Saved-record arithmetic, source text/hash checks and raw ZIP storage hash comparison only"},
        "joint_quality_time": results["proposed_method_joint_quality_time"],
        "updates": {"all_seeds_and_controls": budget["fits"]["optimizer_updates"],
                    "each_final_multistep_model": budget["fits"]["multi_model_updates_per_seed"]},
        "seed2302_round1": {"saved_rows": budget["labels"]["equal_suffix_rows"],
                            "one_multi_suffix_equal": budget["labels"]["seed2302_round1_one_and_multi_equal_in_saved_suffixes"]},
        "primary_passed": results["decision"]["registered_primary_passed"],
        "limitations": ["No original planning, training, branch replay, geometry/IK checks, or performance timing rerun.",
                       "Six test scenes; 180 outputs and 36 multistep runs are not independent scene counts.",
                       "Seed differences combine training-data subsampling and training randomness.",
                       "Closeout is an investment decision under the registered conditions, not a universal DRL impossibility claim.",
                       "Historical archive hashes checked; legacy extracted payloads outside G2.3 not exhaustively rehashed."]
    }
    write(ROOT / "outputs/closeout_receipt.json", receipt)
    print(json.dumps({"all_passed": True, "audits": len(executions), "frozen_archives": len(before),
                      "frozen_g23_members": live_after["matched"], "joint": receipt["joint_quality_time"]["counts"],
                      "new_queries": 0, "new_updates": 0}, ensure_ascii=False))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
