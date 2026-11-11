"""Fixed metadata-only sampling and saved-prefix feature reconstruction.

This module never calls a planner, terminal processor, teacher, optimizer, task
generator, or navigation solver.  ``build_selection`` uses only identities and
schedule metadata; ``reconstruct_episode`` is admitted numerical work and must
only be called after the owning run has durably registered its exact inputs.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
RUN = "rc3_conservative_refinement/integration_20261110/runs/conservative_paired_11622026_v1"
SCHEDULE = "rc3_conservative_refinement/integration_20261110/SCHEDULE_R03.json"
REFERENCES = "rc3_integrated_rppo/DATA_REFERENCES.json"
EVENTS = RUN + "/events.jsonl"
SIZES = ((10, 10), (4, 3), (6, 6), (8, 8))
SEED = 11622026


def _nothing(kind, amount=1, **context):
    return None


def _relative_file(name):
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ValueError("Only project-relative existing evidence paths are allowed")
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError("Missing evidence or path outside repository: " + name)
    return path


def _bytes(name, charge):
    path = _relative_file(name)
    charge("source_reads", 1, path=name, bytes=path.stat().st_size)
    start = time.perf_counter()
    payload = path.read_bytes()
    charge("timing", 0, category="source_read", seconds=time.perf_counter() - start)
    charge("source_hashes", 1, path=name)
    start = time.perf_counter()
    digest = hashlib.sha256(payload).hexdigest()
    charge("timing", 0, category="source_hash", seconds=time.perf_counter() - start)
    return payload, dict(path=name, sha256=digest, bytes=len(payload))


def _key(seed, *parts):
    return hashlib.sha256((str(seed) + "|" + "|".join(map(str, parts))).encode()).hexdigest()


def build_selection(seed=SEED, charge=None):
    """Freeze 64 ON/OFF pairs without reading costs or per-state target labels.

    Eight pairs are selected per phase x size stratum.  A hash order within each
    physical parent is consumed in parent round-robin order.  Parents and rows
    therefore affect sampling only through immutable metadata, never outcomes.
    The complete selected physical-parent set is divided once into approximately
    two-thirds fit / one-third holdout; all states and paired conditions follow
    the same physical-parent assignment.
    """
    if type(seed) is not int or seed != SEED:
        raise ValueError("The fixed diagnostic seed is 11622026")
    charge = charge or _nothing
    raw_schedule, schedule_source = _bytes(SCHEDULE, charge)
    raw_references, reference_source = _bytes(REFERENCES, charge)
    raw_events, event_source = _bytes(EVENTS, charge)
    schedule = json.loads(raw_schedule)
    references = json.loads(raw_references)
    events = [json.loads(line) for line in raw_events.splitlines() if line]
    if schedule["seed"] != SEED:
        raise ValueError("Saved training schedule has another seed")
    completed = {e["request_id"]: e for e in events if e["event"] == "request_done"}
    checkpoints = {}
    updates = {}
    for event in events:
        if event["event"] == "checkpoint":
            rnd = event["progress"]["round"]
            if rnd not in checkpoints or event["sequence"] > checkpoints[rnd]["sequence"]:
                checkpoints[rnd] = event
        elif event["event"] == "ppo_opportunity":
            pair = event["study_method"], event["opportunity"]
            if pair in updates:
                raise ValueError("Duplicate PPO opportunity identity")
            updates[pair] = event
    candidates = defaultdict(list)
    for entry in schedule["training"]:
        opportunity = entry["opportunity"]
        if not 1 <= opportunity <= 128:
            continue
        phase = "early_001_064" if opportunity <= 64 else "late_065_128"
        for slot, spec in enumerate(entry["specs"]):
            size = (spec["width"], spec["height"])
            if size not in SIZES:
                raise ValueError("Unexpected registered training size")
            request_ids = {role: f"train:{role}:{opportunity}:{slot}" for role in ("ON", "OFF")}
            if any(rid not in completed for rid in request_ids.values()):
                raise ValueError("A registered training pair is missing; no replacement allowed")
            physical = references["geometry"][str(spec["map_seed"])]["physical_key"]
            candidates[(phase, size)].append(dict(
                pair_id=f"pair_{opportunity:03}_{slot}", opportunity=opportunity,
                slot=slot, spec=dict(spec), physical_key=physical, phase=phase,
                width=size[0], height=size[1], request_ids=request_ids))
    selected = []
    coverage = []
    for phase in ("early_001_064", "late_065_128"):
        for size in SIZES:
            rows = candidates[(phase, size)]
            parents = defaultdict(list)
            for row in rows:
                parents[row["physical_key"]].append(row)
            parent_order = sorted(parents, key=lambda p: _key(seed, "stratum-parent", phase, *size, p))
            for parent in parent_order:
                parents[parent].sort(key=lambda r: _key(seed, "within-parent", r["pair_id"]))
            chosen = []
            offset = 0
            while len(chosen) < 8:
                added = 0
                for parent in parent_order:
                    if offset < len(parents[parent]):
                        chosen.append(parents[parent][offset])
                        added += 1
                        if len(chosen) == 8:
                            break
                if not added:
                    raise ValueError("A fixed phase/size stratum has fewer than eight pairs")
                offset += 1
            selected.extend(chosen)
            coverage.append(dict(phase=phase, width=size[0], height=size[1],
                                 candidate_pairs=len(rows), candidate_parents=len(parents),
                                 selected_pairs=len(chosen),
                                 selected_parents=len({r["physical_key"] for r in chosen})))
    physical_parents = sorted({r["physical_key"] for r in selected},
                              key=lambda p: _key(seed, "calibration-parent-fold", p))
    if len(physical_parents) < 2:
        raise ValueError("At least two complete physical parents are needed for isolated folds")
    nfit = max(1, min(len(physical_parents) - 1, (2 * len(physical_parents)) // 3))
    fit = physical_parents[:nfit]
    holdout = physical_parents[nfit:]
    fold = {p: "fit" for p in fit} | {p: "holdout" for p in holdout}
    trajectories = []
    for pair in selected:
        opportunity = pair["opportunity"]
        for role in ("ON", "OFF"):
            request_id = pair["request_ids"][role]
            request = completed[request_id]
            update = updates[(role, opportunity)]
            if not update["accepted"]:
                raise ValueError("Unexpected rejected historical update; no silent reinterpretation")
            before, after = checkpoints[opportunity - 1], checkpoints[opportunity]
            trajectories.append(dict(
                trajectory_id=request_id, request_id=request_id,
                request_path=request["path"], request_sha256=request["artifact_sha256"],
                pair_id=pair["pair_id"], method=role, condition=role,
                physical_key=pair["physical_key"], parent_id=pair["physical_key"],
                phase=pair["phase"], width=pair["width"], height=pair["height"],
                opportunity=opportunity, slot=pair["slot"], spec=dict(pair["spec"]),
                split=fold[pair["physical_key"]],
                before_checkpoint=dict(path=before["path"], sha256=before["checkpoint_sha256"],
                                       sequence=before["sequence"], round=opportunity-1),
                after_checkpoint=dict(path=after["path"], sha256=after["checkpoint_sha256"],
                                      sequence=after["sequence"], round=opportunity),
                after_actor_tensor_sha256=update["actor_tensor_sha"],
                update_event_sequence=update["sequence"],
                full_old_logits_persisted=False,
                full_old_logits_recovery="exact before-checkpoint forward; compare saved selected logit/probability"))
    if len(selected) != 64 or len(trajectories) != 128:
        raise AssertionError("Fixed trajectory cap changed")
    split = dict(
        seed=seed, unit="complete physical_key, never task-specific parent_identity or state",
        fit_physical_parents=fit, holdout_physical_parents=holdout,
        state_random_split=False, fold_selection_uses_targets_or_costs=False,
        rule="physical parents sorted by SHA256(seed|calibration-parent-fold|physical_key); first floor(2*N/3) fit, rest holdout",
        selected_trajectory_ids=[r["trajectory_id"] for r in trajectories],
        trajectory_fold={r["trajectory_id"]: r["split"] for r in trajectories},
        coverage=coverage)
    return dict(
        seed=seed, trajectories=trajectories, pairs=selected, split=split,
        selection_rule="phase(1:64/65:128) x size(10x10,4x3,6x6,8x8), eight ON/OFF pairs per stratum; hash-ordered physical-parent round-robin then hash-ordered within-parent rows",
        selection_uses_costs=False, maximum_trajectories=128,
        source_metadata=[schedule_source, reference_source, event_source])


def reconstruct_episode(row, actor, inputs, charge):
    """Reconstruct saved pre-action states without taking any new terminal step.

    The supplied actor is the frozen ON feature adapter.  Correction weights are
    irrelevant here: only its base encoder and feature builder are called.  OFF
    is obtained by zeroing the two registered mean/max coordinates afterwards.
    Full old logits were not persisted and are deliberately marked missing;
    the caller must recover them with the correct before-update correction.
    """
    import numpy as np
    import torch
    from rc3_global_local.environment import Environment
    from rc3_conservative_refinement.adapter import pack_candidate_states

    if actor.distances_enabled is not True:
        raise ValueError("Reconstruction must create ON features before deriving OFF")
    if any(p.requires_grad or p.grad is not None for p in actor.encoder.parameters()):
        raise ValueError("Feature encoder must remain frozen")
    raw, source = _bytes(row["request_path"], charge)
    if source["sha256"] != row["request_sha256"]:
        raise ValueError("Saved trajectory changed")
    record = json.loads(raw)
    if record["condition"] != row["method"]:
        raise ValueError("Saved ON/OFF condition mismatch")
    charge("saved_view_loads", 1, trajectory_id=row["trajectory_id"])
    start = time.perf_counter()
    problem, evidence = inputs.view(row["spec"])
    charge("timing", 0, category="saved_view_restore", seconds=time.perf_counter() - start)
    if (problem.identity != record["input_identity"]
            or problem.parent.identity != record["parent_identity"]
            or evidence["physical_key"] != row["physical_key"]):
        raise ValueError("Saved problem/physical parent identity mismatch")
    if (problem.description() != record["problem"]
            or problem.task_ids.tolist() != record["problem"]["task_ids_global"]
            or problem.candidate_ids.tolist() != record["problem"]["candidate_ids_global"]):
        raise ValueError("Global task/candidate identity or problem description mismatch")
    targets = record["targets"]
    task_count = len(problem.C)
    if targets["task_count"] != task_count or targets["scale"] != max(1, task_count):
        raise ValueError("Original target initial-task normalization differs")
    actions = record["actions_local_indices_0based"]
    audits = record["post_service_audit"]
    trace = record["trace"]
    if (not actions or len(actions) != len(audits) or len(actions) != len(trace)
            or len(actions) != len(record["V_old"])):
        raise ValueError("Missing complete saved actions, audits, trace or values")
    for name in ("raw_remaining_L", "raw_remaining_K", "returns"):
        if len(targets[name]) != len(actions):
            raise ValueError("Missing complete saved component targets")
    if not all(math.isfinite(float(v)) for name in ("raw_remaining_L", "raw_remaining_K", "returns")
               for v in targets[name]) or not math.isfinite(float(targets["H_gain"])):
        raise ValueError("Nonfinite original targets")

    def forbidden_terminal(*args, **kwargs):
        raise RuntimeError("Terminal/H/planning is forbidden during saved-state reconstruction")

    env = Environment(problem, forbidden_terminal)
    states = []
    with torch.no_grad():
        charge("base_encoder_forwards", 1, trajectory_id=row["trajectory_id"])
        start = time.perf_counter()
        encoded = actor.encode(problem)
        charge("timing", 0, category="base_encoder_forward", seconds=time.perf_counter() - start)
        for t, action in enumerate(actions):
            if type(action) is not int or not 0 <= action < len(problem.candidate_ids):
                raise ValueError("Saved action is not a local zero-based candidate index")
            if env.done or not bool(env.mask()[action]):
                raise ValueError("Saved action is illegal in its reconstructed prefix")
            before = problem.task_ids[np.flatnonzero(env.remaining)].tolist()
            served_indices = np.flatnonzero(env.remaining & problem.C[:, action])
            served = problem.task_ids[served_indices].tolist()
            after_remaining = env.remaining & ~problem.C[:, action]
            after = problem.task_ids[np.flatnonzero(after_remaining)].tolist()
            audit, event = audits[t], trace[t]
            if (before != audit["remaining_tasks_global_before"]
                    or served != audit["hypothetical_served_tasks_global"]
                    or served != event["served_tasks_global"]
                    or after != audit["hypothetical_remaining_tasks_global"]
                    or after != event["remaining_tasks_global"]
                    or action != audit["action_local_index_0based"]
                    or action+1 != event["action_local_node"]
                    or int(problem.candidate_ids[action]) != event["action_global_candidate"]
                    or bool(event["done"]) != (t == len(actions)-1)):
                raise ValueError("Reconstructed task/candidate/prefix state differs from saved audit")
            charge("base_state_forwards", 1, trajectory_id=row["trajectory_id"], state=t)
            charge("feature_states", 1, trajectory_id=row["trajectory_id"], state=t)
            start = time.perf_counter()
            features = actor.features(env, encoded)
            charge("timing", 0, category="base_state_features", seconds=time.perf_counter() - start)
            if not math.isclose(float(features["z0"][action]), float(audit["selected_base_logit"]),
                                rel_tol=0., abs_tol=1e-6):
                raise ValueError("Reconstructed SL selected logit differs from saved value")
            extra = features["post_service_distances"]
            if (int(extra["remaining_after_count"][action]) != len(after)
                    or int(extra["unreachable_remaining_count"][action]) != audit["unreachable_remaining_count"]):
                raise ValueError("Reconstructed provider/remaining summaries differ")
            for feature_name, audit_name in (("raw_mean", "mean_distance_m"), ("raw_max", "max_distance_m")):
                value = float(extra[feature_name][action])
                saved = audit[audit_name]
                if saved is None:
                    if math.isfinite(value):
                        raise ValueError("Saved unreachable distance became finite")
                elif not math.isclose(value, float(saved), rel_tol=0., abs_tol=1e-6):
                    raise ValueError("Reconstructed saved distance differs")
            states.append(features)
            # Set only the exact next saved prefix.  No env.step or terminal is
            # called, including at the final service action.
            node = action + 1
            env.raw_L += float(problem.D[env.current, node])
            env.current = node
            env.prefix.append(node)
            env.remaining = after_remaining
            env.done = not bool(after_remaining.any())
            if env.done != (t == len(actions)-1):
                raise ValueError("Saved trace ends before/after full task coverage")
    packed = pack_candidate_states(states)
    x_on = packed["x"]
    x_off = x_on.clone()
    x_off[..., 202:204] = 0.
    batch = dict(packed, xON=x_on, xOFF=x_off)
    return dict(
        record=record, batch=batch, actions=torch.tensor(actions, dtype=torch.long),
        actions0=list(actions), task_count=task_count, targets=targets,
        parent_id=row["physical_key"], physical_key=row["physical_key"],
        trajectory_id=row["trajectory_id"], request_id=row["request_id"],
        pair_id=row["pair_id"], method=row["method"], condition=row["method"],
        width=row["width"], height=row["height"], phase=row["phase"], split=row["split"],
        opportunity=row["opportunity"], slot=row["slot"],
        old_selected_logits=[a["selected_actor_logit"] for a in audits],
        old_selected_probabilities=[a["selected_probability"] for a in audits],
        original_V_old=list(record["V_old"]),
        full_old_logits=None, full_old_logits_persisted=False,
        missing_saved_data=["full_206_candidate_features", "full_268_critic_features", "full_base_logits", "full_action_masks", "full_old_actor_logits"],
        recovery="Features/masks reconstructed from exact saved problem and complete action prefix; no new task, label, action choice, H or route",
        source=source, input_evidence=evidence)
