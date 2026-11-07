"""Finalize mutable S2 project metadata after all seeds close and reports exist.

Do not run during training. No Git mutation/network command, tensor load, model
construction, test, scene generation or scientific query is performed. This is
supplementary publication maintenance and never replaces approved source 0f0f.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
RUN = ROOT / "rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01"
SEEDS = (11622026, 11622027, 11622028)
VERSION = "2026.11.07-s2-closeout"
BASELINE = "f69de88efa9655cf20633b61b6b0b46dd600666c"
PROTOCOL = "0c5bc7d7a9557edf2af76f51a9f418550f0b4ce1ec6a8b47ddea95c6256e78bb"
SOURCE = "0f0ffd2d2dda2fefefcfcb5cb26dc5197b5d11bd1332b0a5546992409049bfe0"
APPROVAL = "9a45d6e7a6b7ccbc8d43b614646ecefe417ba10eaabfe2055c4e896f1e6aa7c4"
AUTH = "4dba652b2156b88848c4c1c49826952414a5b6d08baa7ce250d710b3ea312423"
S3_PENDING = "REVISE_PENDING_FROZEN_PROTOCOL_EVALUATOR_AND_SEPARATE_AUTHORIZATION"
SOURCE_PATH = ROOT / "rc3_navigation_compat_20261106/SOURCE_BINDING_UPDATED.json"
APPROVAL_PATH = ROOT / "rc3_global_local_execution_v3/authorizations/20261106_S1_B/SOURCE_REVISION_APPROVAL.json"
AUTH_PATH = ROOT / "rc3_global_local_execution_v3/authorizations/20261107_S2/S2_AUTHORIZATION.json"


def fail(message):
    raise ValueError(message)


def check(condition, message):
    if not condition:
        fail(message)


def exact_int(value, name):
    check(type(value) is int and value >= 0, name + " must remain an exact nonnegative Python integer")
    return value


def reject_constant(value):
    fail("Nonfinite JSON constant: " + value)


def label(path):
    return path.resolve().relative_to(ROOT).as_posix()


class Inputs:
    def __init__(self):
        self.entries = {}

    def raw(self, path):
        path = path.resolve()
        check(path.is_relative_to(ROOT), "Input outside repository")
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
        check((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
              "Input changed during metadata read: " + label(path))
        self.entries[label(path)] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                                         mtime_ns=before.st_mtime_ns)
        return data

    def digest(self, path):
        if label(path) not in self.entries:
            self.raw(path)
        return self.entries[label(path)]["sha256"]

    def read(self, path):
        return json.loads(self.raw(path).decode("utf-8-sig"), parse_constant=reject_constant)

    def unchanged(self):
        for rel, old in self.entries.items():
            now = (ROOT / rel).stat()
            check((now.st_size, now.st_mtime_ns) == (old["bytes"], old["mtime_ns"]),
                  "Input changed before metadata publication: " + rel)


def git(*args, success=(0,)):
    # Fixed read-only argv only. No shell and no add/commit/fetch/push invocation.
    result = subprocess.run(["git", *args], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    check(result.returncode in success, "Read-only Git command failed: " + " ".join(args))
    return result.stdout.decode("utf-8", errors="strict").strip(), result.returncode


def safe_origin(value):
    parsed = urlsplit(value)
    if parsed.scheme in ("http", "https"):
        return urlunsplit((parsed.scheme, parsed.hostname or "", parsed.path, "", ""))
    return value


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def insert_readme_block(raw, block, old_version=None):
    check(b"<!-- S2_CURRENT_BEGIN -->" not in raw, "S2 README block already exists; refuse duplicate metadata write")
    marker = b"<!-- S1B_CURRENT_BEGIN -->"
    at = raw.find(marker)
    check(at >= 0, "Historical S1B marker not found")
    before, historical = raw[:at], raw[at:]
    if old_version is not None:
        previous = old_version.encode("ascii")
        check(before.count(previous) == 1, "Root README version header is ambiguous")
        before = before.replace(previous, VERSION.encode("ascii"), 1)
        before = before.replace(b"2026-11-06", b"2026-11-07", 1)
    after_marker = raw[at + len(marker):]
    newline = b"\r\n" if after_marker.startswith(b"\r\n") else b"\n"
    inserted = newline.join(block.encode("utf-8").split(b"\n")) + newline + newline
    result = before + inserted + historical
    check(result.endswith(historical), "Historical README bytes were not preserved")
    return result


def append_version_object(raw, old, value):
    check("S2_scientific_execution" not in old, "S2 version record already exists")
    old_version = old["version"].encode("ascii")
    pattern = rb'("version"[ \t\r\n]*:[ \t\r\n]*")' + re.escape(old_version) + rb'(")'
    edited, n = re.subn(pattern, lambda m: m.group(1) + VERSION.encode("ascii") + m.group(2), raw, count=1)
    check(n == 1, "Version ASCII edit did not uniquely match")
    close = edited.rfind(b"}")
    check(close >= 0 and not edited[close + 1:].strip(), "Unexpected root JSON closing bytes")
    body = edited[:close]
    trimmed = body.rstrip(b" \t\r\n")
    old_whitespace = body[len(trimmed):]
    newline = b"\r\n" if old_whitespace.startswith(b"\r\n") else b"\n"
    # Only the new object is serialized. Every old Chinese/history/mixed-newline
    # byte is retained; version ASCII and one comma are the sole old-body edits.
    fragment = json_bytes({"S2_scientific_execution": value})[2:-3]
    fragment = newline.join(fragment.split(b"\n"))
    result = trimmed + b"," + old_whitespace + fragment + newline + edited[close:]
    parsed = json.loads(result.decode("utf-8-sig"), parse_constant=reject_constant)
    check(parsed["version"] == VERSION and parsed["S2_scientific_execution"] == value, "New version record invalid")
    check(all(parsed[k] == v for k, v in old.items() if k != "version"), "Historical version fields changed")
    return result


def main():
    started = time.perf_counter()
    inp = Inputs()
    destinations = [HERE / "GIT_VERSION_AUDIT.json", HERE / "SUPPLEMENTARY_SOURCE_BINDING.json"]
    check(not any(p.exists() for p in destinations), "New publication metadata already exists; refuse replacement")
    required_reports = ("README.md", "IMPLEMENTATION_REVIEW.md", "EXPERIMENT_REPORT.md", "RESULT_TO_CLAIM.md",
                        "RESOURCE_ACCOUNTING.json", "WEIGHT_LINEAGE.json", "STAGE_STATUS.json", "NEXT_STEP_DECISION.md")
    for name in required_reports:
        check((HERE / name).is_file(), "Final report absent: " + name)
        inp.raw(HERE / name)
    stage = inp.read(HERE / "STAGE_STATUS.json")
    resource = inp.read(HERE / "RESOURCE_ACCOUNTING.json")
    check(stage.get("protocol_sha256") == PROTOCOL and stage.get("source_manifest_sha256") == SOURCE,
          "Final stage source/protocol identity differs")
    check(stage.get("S2_user_authorization_sha256") == AUTH and stage.get("S3_authorized") is False
          and stage.get("S3_runtime") == "NOT_STARTED" and stage.get("S3_target_semantic_consumption") == 0,
          "Final authority/sealed-task boundary differs")
    check(stage.get("S3_execution_readiness") == S3_PENDING, "Whole S3 execution must remain pending evaluator and authority")
    check(resource.get("protocol_sha256") == PROTOCOL and resource.get("source_manifest_sha256") == SOURCE,
          "Final resource identity differs")
    check(inp.digest(ROOT / "rc3_global_local_execution_v3/V3_EXECUTION_CANDIDATE.json") == PROTOCOL,
          "Frozen protocol bytes differ")
    check(inp.digest(SOURCE_PATH) == SOURCE and inp.digest(APPROVAL_PATH) == APPROVAL and inp.digest(AUTH_PATH) == AUTH,
          "Approved source binding/source revision approval/S2 authorization bytes differ")
    source_metadata = inp.read(SOURCE_PATH)
    authorization = inp.read(AUTH_PATH)
    sources = authorization["source_sha256"]
    check(len(sources) == 34, "Approved execution source inventory is not 34")
    verified_sources = {rel: dict(expected_sha256=h, observed_sha256=inp.digest(ROOT / rel)) for rel, h in sources.items()}
    check(all(v["expected_sha256"] == v["observed_sha256"] for v in verified_sources.values()),
          "An approved execution source changed")
    active_path = ROOT / "rc3_global_local_execution_v3/ACTIVE_EXECUTION_STATUS.json"
    active = inp.read(active_path)
    protected = {k: v for k, v in active.items() if k in (
        "active_protocol", "active_protocol_sha256", "source_manifest", "source_manifest_sha256", "registry_sha256",
        "H_identity", "protocol_approval", "protocol_approval_sha256", "source_revision_approval",
        "source_revision_approval_sha256", "approved_source_revision", "original_S1_A_source_manifest_sha256")
        or k.startswith("S1_")}
    check(active.get("active_protocol_sha256") == PROTOCOL and active.get("source_manifest_sha256") == SOURCE
          and active.get("source_revision_approval_sha256") == APPROVAL, "Existing active stable identities differ")
    registry_path = ROOT / "rc3_global_local_execution_v3/REGISTRY.json"
    check(inp.digest(registry_path) == active.get("registry_sha256"), "Frozen registry bytes differ")
    # H_BINDING remains an opaque byte receipt; it is not a model/task read.
    check(inp.digest(ROOT / "rc3_global_local_execution_v3/H_BINDING.json") == source_metadata.get("H_binding_sha256")
          and source_metadata.get("H_sha256") == active.get("H_identity"), "Frozen H binding/identity differs")
    by_stage = {exact_int(x["seed"], "stage seed"): x for x in stage["seeds"]}
    by_resource = {exact_int(x["seed"], "resource seed"): x for x in resource["per_seed"]}
    check(set(by_stage) == set(SEEDS) and set(by_resource) == set(SEEDS), "Final seed inventory differs")
    counts, completion_hashes, terminal_hashes, walls = {}, {}, {}, {}
    for seed in SEEDS:
        item, spent = by_stage[seed], by_resource[seed]
        folder = RUN / f"stage_admission/S2/{seed}"
        terminal = [folder / name for name in ("COMPLETE.json", "FAILED.json") if (folder / name).is_file()]
        check(len(terminal) == 1, f"Seed {seed} is open or has conflicting terminal receipts")
        receipt = inp.read(terminal[0])
        terminal_hashes[str(seed)] = inp.digest(terminal[0])
        check(item.get("stage_receipt_sha256") == terminal_hashes[str(seed)], f"Final stage receipt hash differs: {seed}")
        check(receipt.get("status") == item.get("status"), f"Actual terminal status differs: {seed}")
        phase_counts = {p: exact_int(item["completed_updates"][p], f"{seed}/{p} count") for p in ("RL", "continued_SL")}
        check(phase_counts == spent["completed_updates"], f"Stage/resource actual counts differ: {seed}")
        contexts = receipt.get("cumulative_stage_resources", {})
        shared = [v for k, v in contexts.items() if k in (f"{seed}/RL", f"{seed}/continued_SL")]
        chosen = max(shared, key=lambda x: x.get("elapsed_wall_s", 0), default={})
        check(chosen.get("ledger") == spent.get("ledger"), f"Final cumulative shared ledger differs: {seed}")
        check(chosen.get("elapsed_wall_s") == spent.get("cumulative_shared_wall_s"), f"Final shared wall differs: {seed}")
        counts[str(seed)] = phase_counts
        walls[str(seed)] = chosen.get("elapsed_wall_s")
        if receipt["status"] == "COMPLETE":
            check(phase_counts == {"RL": 512, "continued_SL": 512}, f"A COMPLETE seed lacks fixed endpoints: {seed}")
            completion_hashes[str(seed)] = terminal_hashes[str(seed)]
    actual_updates = sum(sum(v.values()) for v in counts.values())
    check(actual_updates == stage.get("actual_optimizer_updates") == resource.get("actual_completed_optimizer_updates"),
          "Final actual optimizer totals differ")
    all_complete = len(completion_hashes) == 3
    check(not all_complete or stage.get("status") == "COMPLETE_ALL3_FIXED_ENDPOINTS", "Final all-three status differs")
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    branch, _ = git("branch", "--show-current")
    head, _ = git("rev-parse", "HEAD")
    auth_commit, _ = git("rev-parse", "--verify", "13a1495^{commit}")
    _, baseline_relation = git("merge-base", "--is-ancestor", BASELINE, head, success=(0, 1))
    check(baseline_relation == 0, "Current HEAD does not descend from the exact S2 baseline")
    _, authorization_relation = git("merge-base", "--is-ancestor", auth_commit, head, success=(0, 1))
    check(authorization_relation == 0, "Current HEAD does not descend from the actual S2 authorization commit")
    origin, _ = git("remote", "get-url", "origin")
    tracking, tracking_rc = git("rev-parse", "--verify", "refs/remotes/origin/" + branch, success=(0, 128))
    main_head, main_rc = git("rev-parse", "--verify", "refs/remotes/origin/main", success=(0, 128))
    main_contains_auth = None
    if main_rc == 0:
        _, main_relation = git("merge-base", "--is-ancestor", auth_commit, main_head, success=(0, 1))
        main_contains_auth = main_relation == 0
    history, _ = git("log", "--format=%H%x09%P%x09%s", auth_commit + ".." + head)
    commits = [dict(zip(("commit", "parents", "subject"), line.split("\t", 2))) for line in history.splitlines()]
    progress = inp.read(HERE / "DELIVERY_PROGRESS.json")
    first, second = progress.get("preuploaded_LFS", {}), progress.get("second_preupload", {})
    check(first.get("objects") == 35 and first.get("reported_size") == "995 MB" and first.get("exit_code") == 0,
          "First LFS preupload receipt differs")
    check(second.get("cumulative_objects_confirmed") == 70 and second.get("reported_size") == "2.0 GB"
          and second.get("exit_code") == 0, "Second cumulative LFS preupload receipt differs")
    supplementary_scripts = {label(p): dict(bytes=len(inp.raw(p)), sha256=inp.digest(p)) for p in sorted(HERE.glob("*.py"))}
    supplementary = dict(schema="rc3-s2-supplementary-source-binding-v1", generated_utc=now,
        status="SUPPLEMENTARY_ONLY_NOT_A_PRODUCTION_SOURCE_REVISION", scripts=supplementary_scripts,
        scopes=["saved-data analysis and figures", "independent audit", "report generation", "progress observation",
                "orchestration of the already-authorized unchanged production CLI", "Git publication metadata maintenance"],
        approved_production_source_manifest=label(SOURCE_PATH), approved_production_source_manifest_sha256=SOURCE,
        approved_source_revision_approval_sha256=APPROVAL, frozen_protocol_sha256=PROTOCOL,
        approved_34_execution_sources_unchanged=True, approved_34_execution_source_byte_checks=verified_sources,
        supersedes_approved_execution_binding=False, creates_new_scientific_authority=False,
        algorithms_rewards_masks_distributions_H_thresholds_changed=False,
        current_script_scientific_calls=0, S3_target_semantic_consumption=0)
    version_path = ROOT / "PROJECT_VERSION.json"
    version_raw = inp.raw(version_path)
    old_version = json.loads(version_raw.decode("utf-8-sig"), parse_constant=reject_constant)
    check(old_version["version"] != VERSION, "Version is already finalized")
    s2_object = dict(status=stage["status"], experiment_id=RUN.name, branch=branch,
        baseline_commit=BASELINE, user_authorization_commit=auth_commit, S2_user_authorization_sha256=AUTH,
        approved_protocol_sha256=PROTOCOL, approved_source_manifest_sha256=SOURCE,
        source_revision_approval_sha256=APPROVAL, authorized_stages=["S1-A", "S1-B", "S2"],
        actual_optimizer_updates=actual_updates, actual_updates_by_seed=counts,
        actual_stage_shared_wall_s_by_seed=walls, completion_sha256=completion_hashes,
        original_S1_A_S1_B_bytes_preserved=True, original_S1_B_updates_separate=768,
        report="rc3_s2_results_20261107/README.md", S3_model_lineage_readiness=stage["S3_model_lineage_readiness"],
        S3_execution_readiness=S3_PENDING, S3_authorized=False, S3_executed=False,
        S3_target_semantic_consumption=0, old60="SEALED", metadata_closed_utc=now,
        main_merged=main_contains_auth, branch_push_complete=False,
        publication_receipt="rc3_s2_results_20261107/PUBLICATION_RECEIPT.json",
        historical_metadata_scope="Earlier objects retain their original-turn status; current state is this S2 object and ACTIVE_EXECUTION_STATUS.")
    active.update(status="APPROVED_V3_S2_COMPLETE_ALL3_FIXED_ENDPOINTS" if all_complete else "APPROVED_V3_S2_CLOSED_INCOMPLETE",
        authorized_stages=["S1-A", "S1-B", "S2"], S2_authorized=True, S3_authorized=False,
        S1_B_S2_S3_authorized=False, S1_B_S2_S3_authorized_scope="S1-B and S2 explicitly authorized; S3 not authorized",
        S2_runtime=stage["status"], S2_scientific_optimizer_updates=actual_updates,
        S2_actual_updates_by_seed=counts, S2_completed_seeds=[s for s in SEEDS if str(s) in completion_hashes],
        S2_actual_optimizer_updates_completed_seeds=sum(sum(counts[str(s)].values()) for s in SEEDS if str(s) in completion_hashes),
        S2_completion_sha256=completion_hashes, S2_terminal_receipt_sha256=terminal_hashes,
        S2_shared_stage_wall_s_by_seed=walls, S2_metadata_closed_utc=now,
        S2_report="rc3_s2_results_20261107/README.md", S2_authorization=label(AUTH_PATH),
        result_report="rc3_s2_results_20261107/README.md",
        stage_complete_receipt_scope="Historical S1-A COMPLETE receipt retained; current S2 completion is the per-seed S2_completion_sha256 map and stage_admission/S2 records.",
        audit_receipt_scope="Historical S1-A audit retained; current S2 audit is rc3_s2_results_20261107/INDEPENDENT_REVIEW.json.",
        working_tree_execution_sources="APPROVED_COMPATIBILITY_REVISION_0f0ffd2; original source approval 9a45d6e retained; independent S2 authorization 4dba652 consumed for all3 seeds, now complete; S3 not authorized.",
        S2_authorization_sha256=AUTH, S3_model_lineage_readiness=stage["S3_model_lineage_readiness"],
        S3_execution_readiness=S3_PENDING, S3_runtime="NOT_STARTED", S3_target_tasks="SEALED",
        S3_target_semantic_consumption=0, original60="SEALED")
    check(all(active[k] == v for k, v in protected.items()), "An existing stable identity or S1 history changed")
    dev = resource.get("development_feasibility_denominators", {})
    dev_n = dev.get("actual_repeated_DEV_evaluations")
    root_block = ("<!-- S2_CURRENT_BEGIN -->\n"
        f"**最新执行状态：S2 `{stage['status']}`，实际完成 {actual_updates} 次正式更新。** "
        + "；".join(f"seed{s}: RL{counts[str(s)]['RL']} / continued-SL{counts[str(s)]['continued_SL']}" for s in SEEDS)
        + f"。S3模型谱系就绪 `{stage['S3_model_lineage_readiness']}`；整个S3执行仍为 `{S3_PENDING}`，无阶段权限、未启动。"
        + f"开发记录是原18个DEV任务在256/512及三seed下的 {dev_n} 次重复评价（全完成计划108次），不是108个独立案例，也不是封存S3的108目标；S3语义消费0。\n\n"
        + "训练回报/成本曲线只作诊断，H收益单列；未证明DRL质量提升、泛化或SL对RL优势。原S1-A/S1-B与冻结科学证据保持；新增RL journal的路径专属LFS规则属于交付维护。"
        + "见[本轮结果与全部报告](rc3_s2_results_20261107/README.md)、[下一步](rc3_s2_results_20261107/NEXT_STEP_DECISION.md)、[当前执行状态](rc3_global_local_execution_v3/ACTIVE_EXECUTION_STATUS.json)及[发布收据（待实际推送后生成）](rc3_s2_results_20261107/PUBLICATION_RECEIPT.json)。\n\n"
        + "下面S1-B及更早段落保留其当时状态，当前以本S2记录及活动执行状态为准。\n<!-- S2_CURRENT_END -->")
    execution_block = ("<!-- S2_CURRENT_BEGIN -->\n"
        f"**当前：S2 `{stage['status']}`，实际{actual_updates}次更新，三seed权重和资源按真实收据归档。** "
        + f"源码0f0ffd2、原批准9a45与冻结协议身份保持。S3模型谱系就绪 `{stage['S3_model_lineage_readiness']}`；执行就绪 `{S3_PENDING}`，S3无授权/未启动，目标及原60封存。"
        + "生产入口目前只实现S1-A/S1-B/S2；后续须补齐冻结协议评估器的冷/暖计时、共同H、父场景聚合与108声明一致性，目标语义访问仍需独立阶段授权。"
        + f"本轮开发仅原18DEV任务的 {dev_n} 次重复评价，不是S3的108目标。"
        + "见[本轮报告](../rc3_s2_results_20261107/README.md)、[当前准确状态](ACTIVE_EXECUTION_STATUS.json)及[待推送后生成的发布收据](../rc3_s2_results_20261107/PUBLICATION_RECEIPT.json)。下文保留历史。\n<!-- S2_CURRENT_END -->")
    git_audit = dict(schema="rc3-s2-prepush-git-version-audit-v1", generated_utc=now,
        baseline_commit=BASELINE, S2_authorization_commit=auth_commit,
        current_branch=branch, current_precloseout_HEAD=head, post_authorization_commits=commits,
        origin_url=safe_origin(origin), origin_tracking_ref="refs/remotes/origin/" + branch,
        observed_local_origin_tracking_HEAD=tracking if tracking_rc == 0 else None,
        origin_tracking_equals_baseline=tracking == BASELINE if tracking_rc == 0 else None,
        remote_branch_HEAD_live_verified=False, branch_push_complete=False, remote_synchronized_claim=False,
        remote_verification_scope="Local tracking refs only; parent must verify the true remote SHA after its actual final push.",
        observed_origin_main_tracking_HEAD=main_head if main_rc == 0 else None,
        local_origin_main_contains_S2_authorization=main_contains_auth, main_merged=main_contains_auth,
        LFS_preupload_progress=dict(source=label(HERE / "DELIVERY_PROGRESS.json"),
            source_sha256=inp.digest(HERE / "DELIVERY_PROGRESS.json"), first=first, second_cumulative=second,
            interpretation="First reports 35 objects/995 MB; second confirms cumulative 70 objects/2.0 GB. Do not add displayed sizes as unique network bytes.",
            unique_network_bytes="NOT_MEASURED", branch_ref_push_proven=False),
        stable_source_identity=dict(protocol_sha256=PROTOCOL, approved_source_manifest_sha256=SOURCE,
            existing_source_revision_approval_sha256=APPROVAL, actual_S2_user_authorization_sha256=AUTH,
            approved_34_source_hashes_verified=True, registry_hash_verified=True),
        original_S1_A_S1_B_science_files_modified=False, unrelated_live_Word_or_untracked_files_read_or_staged=False,
        git_add_commit_fetch_push_or_merge_calls=0, scientific_queries=0, S3_target_semantic_consumption=0,
        first_seed_outer_process_start="NOT_SEPARATELY_INSTRUMENTED; status-update timestamps are not process starts; use actual receipt wall with its stated clock scope",
        source_stage_wall_s_by_seed=walls, supplementary_source_binding="rc3_s2_results_20261107/SUPPLEMENTARY_SOURCE_BINDING.json",
        future_publication_receipt="rc3_s2_results_20261107/PUBLICATION_RECEIPT.json",
        metadata_maintenance_wall_s_before_serialization=time.perf_counter()-started,
        input_bytes=inp.entries)
    output = {
        ROOT / "README.md": insert_readme_block(inp.raw(ROOT / "README.md"), root_block, old_version["version"]),
        version_path: append_version_object(version_raw, old_version, s2_object),
        ROOT / "rc3_global_local_execution_v3/README.md": insert_readme_block(inp.raw(ROOT / "rc3_global_local_execution_v3/README.md"), execution_block),
        active_path: json_bytes(active),
        HERE / "GIT_VERSION_AUDIT.json": json_bytes(git_audit),
        HERE / "SUPPLEMENTARY_SOURCE_BINDING.json": json_bytes(supplementary),
    }
    inp.unchanged()
    for path, data in output.items():
        with path.open("xb" if path in destinations else "wb") as stream:
            stream.write(data)
    print(json.dumps(dict(status="MUTABLE_METADATA_CLOSED_PREPUSH_NOT_SYNCED", version=VERSION,
                          actual_updates=actual_updates, S3_authorized=False,
                          output_sha256={label(p): hashlib.sha256(b).hexdigest() for p, b in output.items()}),
                     ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
