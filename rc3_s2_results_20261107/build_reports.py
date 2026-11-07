"""Build S2 closure reports from stable saved metadata; run only after training stops.

This writer never imports the runtime/torch, deserializes tensor checkpoints,
reads train journals, creates a model, or evaluates a task. Python json/int keeps
all recorded integer identifiers exact. Existing final reports are not replaced.
Run from repository root with the existing Python environment and -B -X utf8.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
REPORT = Path(__file__).resolve().parent
SEEDS = (11622026, 11622027, 11622028)
PHASES = ("RL", "continued_SL")
RUN = ROOT / "rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01"
PROTOCOL = "0c5bc7d7a9557edf2af76f51a9f418550f0b4ce1ec6a8b47ddea95c6256e78bb"
SOURCE = "0f0ffd2d2dda2fefefcfcb5cb26dc5197b5d11bd1332b0a5546992409049bfe0"
AUTH = "4dba652b2156b88848c4c1c49826952414a5b6d08baa7ce250d710b3ea312423"
REVISION_APPROVAL = "9a45d6e7a6b7ccbc8d43b614646ecefe417ba10eaabfe2055c4e896f1e6aa7c4"
BASELINE = "f69de88efa9655cf20633b61b6b0b46dd600666c"
AUTH_PATH = ROOT / "rc3_global_local_execution_v3/authorizations/20261107_S2/S2_AUTHORIZATION.json"
REVISION_PATH = ROOT / "rc3_global_local_execution_v3/authorizations/20261106_S1_B/SOURCE_REVISION_APPROVAL.json"
LFS_PATTERN = "rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01/seed_*/RL/updates.jsonl"
S3_EXECUTION_READINESS = "REVISE_PENDING_FROZEN_PROTOCOL_EVALUATOR_AND_SEPARATE_AUTHORIZATION"
FINAL_NAMES = ("EXPERIMENT_REPORT.md", "RESULT_TO_CLAIM.md", "RESOURCE_ACCOUNTING.json",
               "WEIGHT_LINEAGE.json", "STAGE_STATUS.json", "NEXT_STEP_DECISION.md")
LIMITS = dict(per_seed_RL_terminal_attempts=6144, per_seed_other_terminal_attempts=2048,
              per_seed_total_terminal_attempts=8192, per_seed_shared_wall_s=7200,
              per_seed_RL_updates=512, per_seed_continued_SL_updates=512,
              all_seed_optimizer_updates=3072, all_seed_terminal_attempts=24576,
              all_seed_wall_sum_s=21600)
LIMITATIONS = (
    "训练输入与采样分布随 update 改变；return/J/L/K 曲线及首末区间仅描述实际训练抽样，不能证明 DRL 质量提升或新规模泛化。",
    "H_gain=raw_J-H_J 归属于冻结 H 后处理；不得解释为独立网络收益。",
    "RL 的 REINFORCE objective 与 continued-SL 的交叉熵定义和尺度不同，不以损失大小或下降幅度比较规划质量。",
    "256/512 固定开发检查复用原 18 个声明 DEV 任务；三 seed 全完成时 18×2×3=108 次重复开发评价，绝非 108 个唯一独立案例，也不是封存 S3 的 108 个目标案例。仅作可行性与资源检查，不作质量筛选、最佳 checkpoint 选择或泛化判断；本轮 S3 语义消费为 0。",
    "未新增 frozen-SL、未训练策略的路线评价，未执行 SL 对 RL 的冻结路线比较；不据此宣称优于原生基线、规则或教师。",
    "保存的模型计划验证只覆盖路线与代理可达约束；真实物理标定、IK 与机械臂关节轨迹认证尚未完成。",
    "S3 与原 60 历史保留任务继续封存；模型谱系就绪 GO 只覆盖 S2 权重证据，不代表 S3 执行器就绪或阶段授权。未启动与缺失终点不写成失败或成功。",
    "ledger 的动态计数覆盖有限：缺失模型初始化键不代表实际构造为零；source policy/source Adam 重建与三分支复制在阶段时钟内。没有新增科学未训练起点，原 S1 端点继续复用。",
)


def reject_constant(value):
    raise ValueError("Nonfinite JSON constant: " + value)


def label(path):
    return path.resolve().relative_to(ROOT).as_posix()


class SavedInputs:
    def __init__(self):
        self.manifest = {}

    def raw(self, path):
        path = path.resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Input outside repository: " + str(path))
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("Saved input changed during read: " + label(path))
        self.manifest[label(path)] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                                         mtime_ns=before.st_mtime_ns)
        return data

    def digest(self, path):
        path = path.resolve()
        if label(path) in self.manifest:
            return self.manifest[label(path)]["sha256"]
        # Checkpoints are opaque bytes: no pickle/torch/tensor load.
        self.raw(path)
        return self.manifest[label(path)]["sha256"]

    def read(self, path):
        return json.loads(self.raw(path).decode("utf-8-sig"), parse_constant=reject_constant)

    def unchanged(self):
        for rel, meta in self.manifest.items():
            current = (ROOT / rel).stat()
            if (current.st_size, current.st_mtime_ns) != (meta["bytes"], meta["mtime_ns"]):
                raise RuntimeError("Input changed before report closure: " + rel)


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(x in "0123456789abcdef" for x in value)


def require(condition, message, issues):
    if not condition:
        issues.append(message)
    return bool(condition)


def exact_count(value, name, issues):
    if type(value) is not int or value < 0:
        issues.append(name + " is absent or is not a nonnegative exact integer")
        return None
    return value


def finite_tree(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(x) for x in value.values())
    if isinstance(value, list):
        return all(finite_tree(x) for x in value)
    return True


def fmt(value, digits=6):
    if value is None:
        return "未知／未执行"
    if type(value) is int:
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(headers, rows):
    return "\n".join(("| " + " | ".join(headers) + " |",
                       "| " + " | ".join("---" for _ in headers) + " |",
                       *("| " + " | ".join(fmt(x) for x in row) + " |" for row in rows)))


def checkpoint(inp, path, seed, phase, cursor, binding, issues, required=True):
    files = [path, path.with_suffix(".pt.json"), path.with_suffix(".pt.commit.json")]
    missing = [label(p) for p in files if not p.is_file()]
    if missing:
        if required:
            issues.append("Missing checkpoint boundary: " + ", ".join(missing))
        return None
    side = inp.read(files[1])
    commit = inp.read(files[2])
    actual = inp.digest(path)
    expected_binding = dict(binding, optimizer_seed=seed, run=label(RUN), phase=phase)
    ok = (side.get("schema") == "rc3-trainer-checkpoint-integrity-v1"
          and side.get("phase") == phase and side.get("binding") == expected_binding
          and side.get("cursors", {}).get(phase) == cursor and side.get("weight_version") == cursor
          and side.get("sha256") == actual and side.get("bytes") == path.stat().st_size
          and valid_sha(side.get("model_tensor_sha256"))
          and commit.get("schema") == "rc3-checkpoint-boundary-commit-v1"
          and commit.get("phase") == phase and commit.get("checkpoint_sha256") == actual
          and commit.get("sidecar_sha256") == inp.digest(files[1])
          and commit.get("cursors") == side.get("cursors"))
    require(ok, "Checkpoint hash/binding/closed boundary mismatch: " + label(path), issues)
    return dict(path=label(path), checkpoint_sha256=actual, checkpoint_bytes=path.stat().st_size,
                sidecar_path=label(files[1]), sidecar_sha256=inp.digest(files[1]),
                commit_path=label(files[2]), commit_sha256=inp.digest(files[2]),
                phase=phase, cursor=cursor, weight_version=side.get("weight_version"),
                model_tensor_sha256=side.get("model_tensor_sha256"), hash_binding_closed_boundary_ok=ok,
                tensor_payload_deserialized=False)


def actual_status(inp, seed):
    folder = RUN / f"stage_admission/S2/{seed}"
    complete, failed, start = [folder / (x + ".json") for x in ("COMPLETE", "FAILED", "START")]
    if complete.exists() and failed.exists():
        raise ValueError(f"Conflicting S2 terminal receipts: {seed}")
    if complete.exists():
        return "COMPLETE", inp.read(complete), complete
    if failed.exists():
        return "FAILED_INCOMPLETE", inp.read(failed), failed
    if start.exists():
        interruptions = sorted(folder.glob("INTERRUPTED_*.json"))
        if interruptions:
            return "INTERRUPTED_SAFE_UPDATE_BOUNDARY", inp.read(interruptions[-1]), interruptions[-1]
        raise RuntimeError(f"Seed {seed} is still open; do not write final reports during training")
    return "NOT_EXECUTED", None, None


def main():
    t0 = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=REPORT / "analysis/ANALYSIS_SUMMARY.json")
    parser.add_argument("--independent-audit", type=Path)
    args = parser.parse_args()
    for name in FINAL_NAMES:
        if (REPORT / name).exists():
            raise FileExistsError("Final report already exists; do not silently replace evidence: " + name)
    inp, issues = SavedInputs(), []
    summary = inp.read(args.summary)
    if summary.get("schema") != "rc3-s2-saved-data-analysis-v1":
        raise ValueError("Unexpected analysis schema")
    require(summary.get("analysis_only") is True and summary.get("runtime_changed") is False,
            "Analysis is not explicitly saved-data only", issues)
    require(summary.get("protocol_sha256") == PROTOCOL and summary.get("source_manifest_sha256") == SOURCE,
            "Analysis protocol/source binding differs", issues)
    require(summary.get("fixed_seeds") == list(SEEDS), "Analysis fixed seed order differs", issues)
    require(summary.get("analysis_checks", {}).get("failed") == 0, "Saved-data analysis checks failed or unknown", issues)
    require(summary.get("finite_checks", {}).get("failed") == 0, "Saved-diagnostic finite checks failed or unknown", issues)
    require(finite_tree(summary), "Nonfinite value in analysis summary", issues)
    require(summary.get("optimizer_or_inference_or_scene_nav_H_teacher_calls") == 0
            and summary.get("S3_or_original60_semantic_reads") == 0,
            "Analysis-only zero-call/sealed-read declaration differs", issues)
    auth = inp.read(AUTH_PATH)
    require(inp.digest(AUTH_PATH) == AUTH and auth.get("status") == "AUTHORIZED"
            and auth.get("authorized_by") == "user" and auth.get("optimizer_seeds") == list(SEEDS)
            and auth.get("stages") == ["S2"] and auth.get("S3_authorized") is False,
            "Real S2 user authorization differs", issues)
    require(inp.digest(REVISION_PATH) == REVISION_APPROVAL, "Existing source revision approval changed", issues)
    require(inp.digest(ROOT / "rc3_global_local_execution_v3/V3_EXECUTION_CANDIDATE.json") == PROTOCOL,
            "Frozen execution protocol bytes changed", issues)
    binding_meta = inp.read(ROOT / "rc3_navigation_compat_20261106/SOURCE_BINDING_UPDATED.json")
    require(inp.digest(ROOT / "rc3_navigation_compat_20261106/SOURCE_BINDING_UPDATED.json") == SOURCE,
            "Approved source manifest bytes changed", issues)
    source_hashes = auth.get("source_sha256", {})
    source_check = {rel: dict(expected_sha256=expected, observed_sha256=inp.digest(ROOT / rel))
                    for rel, expected in source_hashes.items()}
    require(len(source_check) == 34 and all(x["expected_sha256"] == x["observed_sha256"] for x in source_check.values()),
            "Approved 34 execution source bytes differ", issues)
    # Binding structure is taken from the immutable source manifest, not reconstructed
    # with mutable downstream authorization hashes.
    binding = binding_meta.get("science_binding", binding_meta.get("binding"))
    if binding is None:
        # START holds the exact stable lineage used by each checkpoint. It is
        # independently compared with the immutable approval identities below.
        first_start = next((RUN / f"stage_admission/S2/{s}/START.json" for s in SEEDS
                            if (RUN / f"stage_admission/S2/{s}/START.json").exists()), None)
        binding = inp.read(first_start)["science_binding"] if first_start else None
    if binding is not None:
        require(binding.get("protocol_sha256") == PROTOCOL and binding.get("source_manifest_sha256") == SOURCE
                and binding.get("approval_sha256") == REVISION_APPROVAL
                and binding.get("source_sha256") == source_hashes,
                "Stable scientific binding differs from approved source identities", issues)
    pre_review_path = REPORT / "IMPLEMENTATION_REVIEW.md"
    pre_review = inp.raw(pre_review_path).decode("utf-8-sig")
    require("GO" in pre_review, "Existing pre-run implementation GO is not found", issues)
    preservation = inp.read(REPORT / "FROZEN_BASELINE_AUDIT.json")
    require(preservation.get("status") == "PASS" and preservation.get("baseline_commit") == BASELINE
            and preservation.get("mismatch_count") == 0 and preservation.get("read_error_count") == 0,
            "Independent baseline-byte preservation is not PASS", issues)
    attributes = inp.raw(ROOT / ".gitattributes").decode("utf-8-sig")
    rule_lines = [x for x in attributes.splitlines() if x.strip().startswith(LFS_PATTERN + " ")]
    require(any("filter=lfs" in x and "diff=lfs" in x and "merge=lfs" in x and "-text" in x for x in rule_lines),
            "New S2 RL journal path-specific LFS metadata rule is absent", issues)
    independent = inp.read(args.independent_audit) if args.independent_audit else None
    if independent is not None:
        require(independent.get("schema") == "rc3-s2-independent-review-v1"
                and independent.get("status") in ("PASS_S2_CLOSED", "PASS_EVIDENCE_INCOMPLETE_STAGE"),
                "Supplied independent audit does not carry a recognized saved-evidence PASS status", issues)
    seeds, lineage, resources = [], {}, []
    total_updates = 0
    for seed in SEEDS:
        observed = summary.get("per_seed", {}).get(str(seed))
        if not isinstance(observed, dict):
            raise ValueError(f"Missing analysis seed entry: {seed}")
        status, receipt, receipt_path = actual_status(inp, seed)
        require(observed.get("status") == status, f"Analysis/admission status differs for {seed}", issues)
        if status != "NOT_EXECUTED":
            admission = RUN / f"stage_admission/S2/{seed}"
            local_auth = inp.read(admission / "AUTHORIZATION.json")
            local_start = inp.read(admission / "START.json")
            require(local_auth.get("authorization_sha256") == AUTH
                    and local_auth.get("science_binding") == binding
                    and local_start.get("science_binding") == binding
                    and local_start.get("authorization_record_sha256") == inp.digest(admission / "AUTHORIZATION.json"),
                    f"{seed} admission authorization or stable source lineage differs", issues)
            if receipt and status in ("COMPLETE", "FAILED_INCOMPLETE"):
                require(receipt.get("science_binding") == binding
                        and receipt.get("authorization_record_sha256") == inp.digest(admission / "AUTHORIZATION.json")
                        and receipt.get("start_record_sha256") == inp.digest(admission / "START.json"),
                        f"{seed} final receipt/start/authorization hash chain differs", issues)
        counts = {p: exact_count(observed.get("completed_updates", {}).get(p), f"{seed}/{p} update count", issues)
                  for p in PHASES}
        if None in counts.values():
            raise ValueError(f"Cannot report an inexact update count: {seed}")
        total_updates += sum(counts.values())
        require(all(x <= 512 for x in counts.values()), f"{seed} exceeds a fixed phase update cap", issues)
        require(observed.get("finite_checks", {}).get("failed") == 0,
                f"{seed} saved-diagnostic finite checks failed or unknown", issues)
        contexts = (receipt or {}).get("cumulative_stage_resources", (receipt or {}).get("resources", {}))
        shared = {k: v for k, v in contexts.items() if k in (f"{seed}/RL", f"{seed}/continued_SL")}
        context = max(shared.values(), key=lambda x: x.get("elapsed_wall_s", 0), default={})
        ledger = context.get("ledger", {})
        if not ledger and status == "NOT_EXECUTED":
            ledger = {}
        require(observed.get("cumulative_ledger", {}) == ledger,
                f"{seed} analysis ledger differs from final shared stage receipt", issues)
        wall = context.get("elapsed_wall_s")
        if wall is None and not shared:
            wall = contexts.get("admission_wall_s")
        wall_known = isinstance(wall, (int, float)) and not isinstance(wall, bool) and math.isfinite(wall)
        require(status == "NOT_EXECUTED" or wall_known, f"{seed} final cumulative wall is unknown", issues)
        if wall_known:
            require(wall <= 7200, f"{seed} cumulative shared wall exceeds 7200s", issues)
        attempts = {role: value.get("terminal_attempts") for role, value in ledger.get("terminal", {}).items()}
        terminal_total = sum(attempts.values()) if all(type(x) is int and x >= 0 for x in attempts.values()) else None
        if status == "NOT_EXECUTED":
            terminal_total = 0
        elif not attempts and counts == {"RL": 0, "continued_SL": 0} and not shared:
            terminal_total = 0  # Admission failed before a science context existed.
        else:
            require(bool(attempts) and terminal_total is not None, f"{seed} terminal ledger is absent/inexact", issues)
        rl_attempts = attempts.get("RL", 0) if terminal_total is not None else None
        other_attempts = terminal_total - rl_attempts if terminal_total is not None else None
        if terminal_total is not None:
            require(rl_attempts <= 6144 and other_attempts <= 2048 and terminal_total <= 8192,
                    f"{seed} terminal budget cap exceeded", issues)
        if ledger:
            require(ledger.get("optimizer_updates") == sum(counts.values()),
                    f"{seed} paid ledger/completed update count differs", issues)
            require(ledger.get("native_teacher_attempts", 0) == 0
                    and ledger.get("terminal", {}).get("teacher_H", {}).get("terminal_attempts", 0) == 0,
                    f"{seed} new teacher query recorded", issues)
        final_path = RUN / f"seed_{seed}/S2/FINAL.json"
        final = inp.read(final_path) if final_path.exists() else None
        fork_path = RUN / f"seed_{seed}/S2/FORK_RECEIPT.json"
        fork = inp.read(fork_path) if fork_path.exists() else None
        item = dict(seed=seed, status=status, completed_updates=counts,
                    stage_receipt=label(receipt_path) if receipt_path else None,
                    stage_receipt_sha256=inp.digest(receipt_path) if receipt_path else None,
                    actual_error=(receipt or {}).get("error"), endpoint_complete=False)
        own = dict(seed=seed, status=status, fork_receipt=label(fork_path) if fork else None,
                   fork_receipt_sha256=inp.digest(fork_path) if fork else None,
                   fork_receipt_contents=fork, source_S1=None, untrained_reference=None,
                   frozen_SL_reference=None, branches={}, missing_endpoint_is_not_synthesized=True)
        if status != "NOT_EXECUTED" and binding is not None:
            own["source_S1"] = checkpoint(inp, RUN / f"seed_{seed}/S1/update_0256.pt", seed, "S1", 256, binding, issues)
            own["untrained_reference"] = checkpoint(inp, RUN / f"seed_{seed}/S1/untrained.pt", seed, "S1", 0, binding, issues)
            own["frozen_SL_reference"] = own["source_S1"]
            require(fork is not None, f"{seed} exact fork receipt missing", issues)
            if fork and own["source_S1"]:
                common = own["source_S1"]["model_tensor_sha256"]
                require(fork.get("source_S1_checkpoint_sha256") == own["source_S1"]["checkpoint_sha256"]
                        and fork.get("common_tensor_sha256") == common and fork.get("frozen_SL") is True
                        and fork.get("frozen_untrained_checkpoint") == "../S1/untrained.pt"
                        and fork.get("source_S1_optimizer_inherited") is False
                        and all(fork.get("branches", {}).get(p, {}).get("tensor_sha256") == common
                                and fork.get("branches", {}).get(p, {}).get("optimizer_state_entries") == 0
                                for p in ("frozen_SL", *PHASES)),
                        f"{seed} own-S1 exact-weight/fresh-Adam fork record differs", issues)
            for phase in PHASES:
                start = checkpoint(inp, RUN / f"seed_{seed}/{phase}/start_from_SL.pt", seed, phase, 0,
                                   binding, issues, required=counts[phase] > 0 or status == "COMPLETE")
                endpoint = checkpoint(inp, RUN / f"seed_{seed}/{phase}/update_0512.pt", seed, phase, 512,
                                      binding, issues, required=counts[phase] == 512 or status == "COMPLETE")
                if start and fork:
                    require(start["model_tensor_sha256"] == fork.get("common_tensor_sha256"),
                            f"{seed}/{phase} initial exact S1 tensor identity differs", issues)
                own["branches"][phase] = dict(actual_completed_updates=counts[phase], start_from_own_S1=start,
                                              fixed_endpoint=endpoint, eligible_as_completed_endpoint=status == "COMPLETE")
        if status == "COMPLETE":
            require(counts == {"RL": 512, "continued_SL": 512} and final is not None
                    and final.get("status") == "COMPLETE_FIXED_ENDPOINTS"
                    and final.get("no_S3_queries") is True and final.get("no_online_rule_fallback") is True
                    and final.get("ledger") == ledger
                    and final.get("frozen_SL_tensor_sha256") == (fork or {}).get("common_tensor_sha256"),
                    f"{seed} complete receipt lacks the actual fixed endpoints/closure", issues)
            item["endpoint_complete"] = (counts == {"RL": 512, "continued_SL": 512} and final is not None
                                         and all(own["branches"].get(p, {}).get("fixed_endpoint") for p in PHASES))
        if receipt and status in ("COMPLETE", "FAILED_INCOMPLETE"):
            # Admission artifacts are checkpoints, sidecars, fork/final and a
            # small resource snapshot; no full train journal is read here.
            for rel, expected in receipt.get("artifacts", {}).items():
                p = (RUN / rel).resolve()
                if not p.is_relative_to(RUN) or p.suffix == ".jsonl":
                    raise ValueError("Unexpected admission artifact path: " + rel)
                require(p.is_file() and inp.digest(p) == expected,
                        f"{seed} final admission artifact changed: {rel}", issues)
        resources.append(dict(seed=seed, status=status, completed_updates=counts,
                              cumulative_shared_wall_s=wall if wall_known else (0 if status == "NOT_EXECUTED" else None),
                              receipt=label(receipt_path) if receipt_path else None,
                              shared_resource_context_names=list(shared), contexts_not_additive=True,
                              ledger=ledger, optimizer_attempts=ledger.get("optimizer_attempts"),
                              optimizer_updates=ledger.get("optimizer_updates"),
                              unfinished_optimizer_attempts=observed.get("unfinished_optimizer_attempts"),
                              terminal_RL_attempts=rl_attempts, terminal_other_attempts=other_attempts,
                              terminal_total_attempts=terminal_total,
                              failed_terminal_attempts=observed.get("failed_terminal_attempts_n"),
                              pending_terminal_attempts=observed.get("pending_terminal_attempts_n"),
                              RL_completed_attempts_outside_completed_update_logs=observed.get("RL_completed_attempts_outside_completed_update_logs_n"),
                              RL_cache_hits_outside_completed_update_logs=observed.get("RL_cache_hits_outside_completed_update_logs_n")))
        seeds.append(item)
        lineage[str(seed)] = own
    require(total_updates == summary.get("completed_update_rows"), "Total actual update count differs from analysis rows", issues)
    require(total_updates <= 3072, "All-seed optimizer update cap exceeded", issues)
    terminal_total = sum(x["terminal_total_attempts"] for x in resources) if all(x["terminal_total_attempts"] is not None for x in resources) else None
    wall_total = sum(x["cumulative_shared_wall_s"] for x in resources) if all(x["cumulative_shared_wall_s"] is not None for x in resources) else None
    if terminal_total is not None:
        require(terminal_total <= 24576, "All-seed terminal cap exceeded", issues)
    if wall_total is not None:
        require(wall_total <= 21600, "All-seed wall-sum cap exceeded", issues)
    dev_checks = [row for seed in SEEDS for row in summary["per_seed"][str(seed)].get("development_checks", [])]
    dev_counts_exact = all(type(row.get(k)) is int and row[k] >= 0
                           for row in dev_checks for k in ("observed_cases", "feasible_cases"))
    dev_observed = sum(row["observed_cases"] for row in dev_checks) if dev_counts_exact else None
    dev_feasible = sum(row["feasible_cases"] for row in dev_checks) if dev_counts_exact else None
    development_denominators = dict(
        declared_original_DEV_task_count=18,
        fixed_development_update_boundaries=[256, 512], optimizer_seed_count=3,
        expected_repeated_DEV_evaluations_if_all_complete=108,
        expected_denominator_formula="18 original declared DEV tasks x 2 fixed checkpoints x 3 seeds",
        actual_saved_development_check_records=len(dev_checks),
        actual_repeated_DEV_evaluations=dev_observed,
        actual_feasible_repeated_DEV_evaluations=dev_feasible,
        unique_independent_case_count_is_not_108=True,
        evaluations_are_repeated_observations_of_original_18_DEV_tasks=True,
        these_are_not_the_sealed_S3_108_target_cases=True,
        S3_target_semantic_consumption=0,
        purpose="feasibility/resources only; no quality selection, independence, generalization or comparative-quality claim",
        checkpoint_records=dev_checks)
    complete = all(x["status"] == "COMPLETE" and x["endpoint_complete"] for x in seeds)
    if complete and independent is not None:
        require(independent.get("status") == "PASS_S2_CLOSED",
                "Independent review does not support the claimed all-three closed S2 stage", issues)
    model_lineage_readiness = "GO" if complete and not issues else "REVISE"
    stage_status = "COMPLETE_ALL3_FIXED_ENDPOINTS" if complete else (
        "STOPPED_FAILED_OR_INCOMPLETE" if any(x["status"] == "FAILED_INCOMPLETE" for x in seeds)
        else "INTERRUPTED_OR_NOT_EXECUTED")
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    shared = dict(schema="rc3-s2-closure-v1", generated_utc=now, experiment_id=RUN.name,
                  baseline_commit=BASELINE, protocol_sha256=PROTOCOL, source_manifest_sha256=SOURCE,
                  source_revision_approval_sha256=REVISION_APPROVAL,
                  S2_user_authorization_sha256=AUTH, S2_user_authorization_commit_abbreviated="13a1495",
                  S3_authorized=False, S3_runtime="NOT_STARTED", original60="SEALED",
                  S3_model_lineage_readiness=model_lineage_readiness,
                  S3_execution_readiness=S3_EXECUTION_READINESS,
                  S3_target_semantic_consumption=0,
                  S3_target_tasks="SEALED", scientific_calls_by_report_writer=0,
                  analysis_summary=label(args.summary), analysis_summary_sha256=inp.digest(args.summary))
    outer_process_records = {}
    for seed in SEEDS[1:]:
        process_record = REPORT / f"production_invocations/{seed}_PROCESS_RESULT.json"
        if process_record.is_file():
            outer_process_records[str(seed)] = inp.read(process_record)
    status_out = dict(shared, status=stage_status, seeds=seeds, actual_optimizer_updates=total_updates,
                      development_feasibility_denominators=development_denominators,
                      S3_model_lineage_GO_is_execution_authorization=False,
                      S3_execution_readiness_evidence=dict(
                          production_authority_stages=["S1-A", "S1-B", "S2"],
                          production_runner_stage_choices=["S1-A", "S1-B", "S2"],
                          authority_rejects_S3=True,
                          TrainingRegistry_semantic_splits=["train", "dev"],
                          S3_existing_artifacts="Frozen protocols and 108 declaration CSV rows; no S3 production evaluator is implemented in the current approved source.",
                          code_references=["rc3_global_local_execution_v3/authority.py", "rc3_global_local_execution_v3/runner.py", "rc3_global_local_trainer/data.py"],
                          target_task_semantic_reads_this_round=0),
                      model_lineage_gate_issues=issues, validation_scope="saved diagnostics plus opaque-byte checkpoint/sidecar/binding integrity; no tensor deserialization or new evaluation")
    lineage_out = dict(shared, seeds=lineage,
                       interpretation="Frozen-SL uses the original own-seed S1 update0256; untrained uses the original own-seed S1 untrained checkpoint. No regenerated reference policy. Fresh Adam is evidenced by the original fork receipt and hashed start boundary; no S1 optimizer momentum is inherited.",
                       reference_route_evaluations_added=0, tensor_payload_deserializations=0)
    accounting = dict(shared, limits=LIMITS, per_seed=resources,
                      development_feasibility_denominators=development_denominators,
                      actual_completed_optimizer_updates=total_updates,
                      actual_terminal_attempts=terminal_total, actual_shared_stage_wall_sum_s=wall_total,
                      scientific_optimizer_updates_added_by_analysis_or_reporting=0,
                      original_S1_B_scientific_optimizer_updates_separate=768,
                      new_engineering_optimizer_updates=0,
                      new_teacher_generation_queries=0, S3_queries=0, old60_evaluations=0,
                      failures_are_charged=True, safe_restore_does_not_reset_resources=True,
                      model_construction_accounting=dict(
                          new_scientific_untrained_starts=False,
                          original_own_seed_S1_endpoints_reused=True,
                          physical_model_initializations="NOT_INFERRED_FROM_ABSENT_COUNTERS",
                          source_policy_and_source_Adam_reconstruction_and_three_branch_copies="inside cumulative stage clock",
                          dynamic_counter_coverage="Ledger does not necessarily expose every physical constructor/deepcopy operation."),
                      outer_process_timing=dict(
                          first_seed_outer_CLI_startup="NOT_SEPARATELY_INSTRUMENTED",
                          helper_process_result_records=outer_process_records,
                          nested_clock_rule="Helper outer process wall includes production startup and stage time; show separately, never add to shared-stage wall."),
                      source_34_byte_checks=source_check,
                      independent_baseline_byte_audit=dict(path=label(REPORT / "FROZEN_BASELINE_AUDIT.json"),
                          sha256=inp.digest(REPORT / "FROZEN_BASELINE_AUDIT.json"),
                          status=preservation.get("status"), active_wall_s=preservation.get("active_wall_seconds"),
                          original_S1_A_S1_B_and_historical_science_preserved=preservation.get("status") == "PASS",
                          gitattributes_scope="pre-LFS-metadata-update baseline check only; no repeated baseline scan"),
                      intentional_git_metadata_change=dict(path=".gitattributes", rule=rule_lines,
                          purpose="Only newly created S2 RL updates.jsonl paths use LFS; frozen scientific evidence and 34 execution sources are unchanged."),
                      postprocessing_separate_from_stage=dict(analysis=summary.get("analysis_runtime"),
                          report_writer_wall_s_before_serialization=time.perf_counter()-t0,
                          scope="Read-only saved-data metadata/hash processing; not formal scientific stage time, optimizer work or performance evidence."),
                      independent_audit=dict(path=label(args.independent_audit), sha256=inp.digest(args.independent_audit), result=independent) if args.independent_audit else None,
                      report_inputs=inp.manifest, model_lineage_gate_issues=issues)
    seed_table = table(["seed", "实际状态", "RL updates", "continued-SL updates", "共享秒数", "RL终局尝试", "其他终局尝试", "总终局尝试"],
                       [(r["seed"], r["status"], r["completed_updates"]["RL"], r["completed_updates"]["continued_SL"],
                         r["cumulative_shared_wall_s"], r["terminal_RL_attempts"], r["terminal_other_attempts"], r["terminal_total_attempts"]) for r in resources])
    costs = table(["seed", "训练 raw J 均值", "训练 H后 J 均值", "冻结 H gain 均值", "无效保存计划", "失败终局尝试"],
                  [(s, summary["per_seed"][str(s)].get("raw_mean_J"), summary["per_seed"][str(s)].get("H_mean_J"),
                    summary["per_seed"][str(s)].get("mean_H_gain"), summary["per_seed"][str(s)].get("invalid_saved_plans_n"),
                    summary["per_seed"][str(s)].get("failed_terminal_attempts_n")) for s in SEEDS])
    dev_text = (f"开发可行性分母：当前保存记录对应 `{fmt(dev_observed)}` 次重复 DEV 评价，"
                f"其中可行 `{fmt(dev_feasible)}` 次。原声明 DEV 任务只有 18 个；三 seed 全完成时为 "
                "`18个DEV任务 × 2个固定checkpoint(256/512) × 3个seed = 108次重复评价`。"
                "即使汇总为 108/108，也只表示这些重复开发评价全部可行，绝非 108 个唯一独立案例，"
                "更不是封存 S3 的 108 个目标案例。本轮 S3 目标语义消费为 `0`。"
                "这些数据只支持开发可行性／资源检查，不作质量筛选或泛化主张。")
    limit_text = "\n".join("- " + x for x in LIMITATIONS)
    issues_text = "\n".join("- " + x for x in issues) if issues else "无保存证据／哈希／数值／预算技术阻断项。"
    basis = (f"固定协议 `{PROTOCOL}`，已批准源码绑定 `{SOURCE}`；原真实源码修订批准 `{REVISION_APPROVAL}` 保持。\n\n"
             f"本轮真实用户 S2 授权 `{AUTH}`（训练前提交 `13a1495`）一次覆盖三固定 seed；它不授权 S3。")
    readme = (f"# S2正式DRL科学训练闭环\n\n状态：`{stage_status}`。实际完成更新 `{total_updates}`；S3 模型谱系就绪 `{model_lineage_readiness}`，执行就绪 `{S3_EXECUTION_READINESS}`，执行授权仍为无。\n\n"
              + seed_table + "\n\n" + basis
              + "\n\n" + dev_text
              + "\n\n报告：[实验](EXPERIMENT_REPORT.md)、[主张边界](RESULT_TO_CLAIM.md)、[真实资源账](RESOURCE_ACCOUNTING.json)、[权重谱系](WEIGHT_LINEAGE.json)、[阶段状态](STAGE_STATUS.json)、[下一步](NEXT_STEP_DECISION.md)。"
              + "\n\n[完整权重与产物入口](ARTIFACT_INDEX.md)、[梯度、优势、熵、分窗口与首末区间诊断](TRAINING_DIAGNOSTICS.md)、[独立审查](INDEPENDENT_REVIEW.md)、[单独记账的报告与传输异常](MAINTENANCE_NOTES.json)。"
              + "\n\n[分析方法](ANALYSIS_METHOD.md)、[保存数据分析](analysis/ANALYSIS_SUMMARY.json)、[图注](FIGURE_CAPTIONS.md) 与图 01–09 的独立脚本保留。图形仅用于训练诊断。"
              + "\n\n原 S1-A/S1-B 与历史证据经单次独立字节审计保留；`.gitattributes` 仅为新 S2 RL journal 增加路径专属 LFS 规则，审计中的该文件检查注明为规则更新前。"
              + "\n\n" + limit_text + "\n")
    review = (f"# S2实施审查与运行后闭环\n\n既有训练前技术准入为 GO；本文件补充运行后证据，不追溯签发训练许可。\n\n{basis}\n\n"
              + f"34 份执行源码保持；未重开工程准入，新增工程 optimizer=0。真实 S2 更新 `{total_updates}` 单列科学账。模型／环境／奖励／mask／端点／H／教师谱系沿用既有已批准审查，不在后处理中改动。\n\n"
              + f"运行后 S2 端点／S3 模型谱系就绪：`{model_lineage_readiness}`；固定端点、保存诊断 finite、来源与 hash、资源门槛共同判定。S3 执行就绪仍为 `{S3_EXECUTION_READINESS}`：当前生产 authority/runner 只实现 S1-A、S1-B、S2，TrainingRegistry 只有 train/dev；声明 CSV 和协议不等于已实现评估器。该结论不说明规划质量提升，也不授予 S3 权限。\n\n{issues_text}\n\n"
              + "训练前本轮准入原文（原字节哈希记于 RESOURCE_ACCOUNTING.json）：\n\n> "
              + pre_review.strip().replace("\n", "\n> ") + "\n")
    review = review.replace("\n> \n", "\n>\n")
    experiment = (f"# S2实验报告\n\n`{stage_status}`，按实际阶段 admission 收据闭环；未知／未启动保持其真实状态，不作为失败计数。\n\n{basis}\n\n"
                  + seed_table + f"\n\n实际已完成 optimizer updates={total_updates}；终局尝试={fmt(terminal_total)}；累计共享阶段墙钟之和={fmt(wall_total)} 秒。RL 与 continued-SL 两 context 是同一份累计 ledger 与时钟，禁止相加。失败、未完成 update 中已付终局以及缓存命中分别留在原 ledger 与分析表中。\n\n"
                  + "实际训练抽样的描述统计：\n\n" + costs
                  + f"\n\n保存数据分析检查失败数 `{summary.get('analysis_checks', {}).get('failed')}`，finite 检查失败数 `{summary.get('finite_checks', {}).get('failed')}`；这只说明所检查保存诊断的数值与记录一致性。\n\n"
                  + dev_text + "\n\n"
                  + limit_text + "\n\n完整逐更新／逐轨迹／梯度／终局尝试／开发可行性明细见 analysis；本报告生成与保存数据分析未运行新的模型、场景、导航、H 或教师查询。\n")
    claims = ("# 结果到主张\n\n"
              + table(["事项", "证据与可支持范围"], [
                  ("S2实际执行", f"收据与日志记录 {total_updates} 次完成更新；按 seed/phase 展示真实状态。"),
                  ("固定端点", "仅实际存在并通过哈希、sidecar 与 commit 绑定的文件列为端点；缺失不补造。"),
                  ("数值与资源", "保存诊断 finite、梯度、ledger 与硬预算检查属于工程／训练闭环，不能转写为质量提升。"),
                  ("冻结来源", "原 own-seed S1 权重、fresh Adam 分叉收据、S1-A/S1-B 原字节以及 34 源码身份保持。"),
                  ("开发可行性分母", "原18个DEV任务的重复评价；全完成时18×2×3=108次，不是108个独立案例或S3目标。S3语义消费0。"),
                  ("DRL独立收益／泛化／SL对RL", "未建立；训练输入改变且未执行冻结参考路线比较。"),
                  ("S3模型谱系", f"就绪判定 {model_lineage_readiness}；只支持实际 S2 端点来源与保存诊断闭环。"),
                  ("S3执行", f"{S3_EXECUTION_READINESS}；尚无 S3 生产评估器，未授权、未启动，目标任务与原60仍封存。")])
              + "\n\n" + limit_text + "\n")
    next_step = (f"# 下一步决策\n\nS3 模型谱系就绪：`{model_lineage_readiness}`。S3 执行就绪：`{S3_EXECUTION_READINESS}`。S3 执行授权：`NOT_AUTHORIZED`。\n\n"
                 + ("三 seed 的 RL512 与 continued-SL512 实际完成且当前保存诊断／哈希／资源门槛通过，仅说明后续评估可引用的模型谱系就绪。\n\n" if model_lineage_readiness == "GO" else
                    "保留真实已花资源与不完整阶段，停止隐式续跑；先处置下面的技术问题，或按原协议与原授权审查已关闭边界安全恢复的合法性。不能清零、免费重试、换 seed 或扩预算。\n\n")
                 + issues_text
                 + "\n\n源码现状：authority.STAGES 与生产 runner 的 --stage choices 只有 S1-A/S1-B/S2，authority 明确拒绝 S3；TrainingRegistry 只有 train/dev。已有 S3 文件属于冻结协议与 108 行声明，不能代替生产评估器。\n\n"
                 + "下一项工程任务是在不访问保留目标任务的前提下，实现并验证符合既有冻结协议的 heldout 评估器：冷／暖启动时间口径、各方法共用冻结 H、父场景聚合，以及 108 条声明的身份与覆盖一致性。该任务须保持固定研究目标、冻结协议和现有门槛；本轮不实施评估器，也不消费 S3。目标任务语义访问及真正阶段执行仍须未来独立明确授权。\n\n"
                 + "上述缺口来自当前源码能力，不引入新的前置审批循环或研究方向变更。任何实质算法、奖励、分布、预算或门槛变化仍须新增协议，绑定源码／H／环境／seed 并获独立批准；本报告不自动激活提案。S3／原60继续封存，不自主补做 frozen-SL／SL对RL 路线比较或新泛化评价。\n")
    outputs = {"README.md": readme, "IMPLEMENTATION_REVIEW.md": review,
               "EXPERIMENT_REPORT.md": experiment, "RESULT_TO_CLAIM.md": claims,
               "RESOURCE_ACCOUNTING.json": accounting, "WEIGHT_LINEAGE.json": lineage_out,
               "STAGE_STATUS.json": status_out, "NEXT_STEP_DECISION.md": next_step}
    payloads = {name: ((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
                       if isinstance(value, dict) else value).encode("utf-8") for name, value in outputs.items()}
    inp.unchanged()
    for name, data in payloads.items():
        mode = "wb" if name in ("README.md", "IMPLEMENTATION_REVIEW.md") else "xb"
        with (REPORT / name).open(mode) as stream:
            stream.write(data)
    print(json.dumps(dict(status=stage_status, S3_model_lineage_readiness=model_lineage_readiness,
                          S3_execution_readiness=S3_EXECUTION_READINESS, S3_authorized=False,
                          actual_optimizer_updates=total_updates, issue_count=len(issues), issues=issues,
                          outputs={name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}),
                     ensure_ascii=False, allow_nan=False))
    return 0 if model_lineage_readiness == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
