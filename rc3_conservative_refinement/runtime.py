"""Real SL256 feature adapter, complete on-policy traces, and actor-only loading.

This module never invokes the old 202-dimensional optimizer or runner.  The
historical Actor is used only for its frozen SL feature construction; its unused
202-dimensional residual is removed immediately.  No inputs are selected or
generated here: callers must supply a separately admitted problem factory.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

import numpy as np
import torch

from .adapter import LookaheadResidualActor, pack_candidate_states
from .costs import remaining_cost_targets


ROOT = Path(__file__).resolve().parents[1]
SL_PATH = ('rc3_integrated_rppo/runs/rc3_integrated_rppo_v1_20261110_01/'
           'actors/11622026/SL256_0256_1817f58aa650.pt')
SL_SHA256 = '60f5aa4e59983e92d565f0599823b710965cf2c96a041e7eb483d23a8eb0556c'
SL_TENSOR_SHA256 = '1817f58aa6506456ff12f0c5fe0a914e156aef8af319f4687dde90782f55a63e'
UPSTREAM = {
    'path': 'rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01/seed_11622026/S1/update_0256.pt',
    'checkpoint_sha256': 'caed2d90d5283f56d816bbf5c103e395e93f2f351fa31e0085274ad204f76bfb',
    'sidecar_path': 'rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01/seed_11622026/S1/update_0256.pt.json',
    'sidecar_sha256': '37e1378d39d7749eb0d26fa12cd311573113bfdee343a7be4bdc830dac52b75a',
    'commit_path': 'rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01/seed_11622026/S1/update_0256.pt.commit.json',
    'commit_sha256': '708de69f1e85f3c84630b5db6357c12144dd7967dd53c56827647a65ac71824f',
}
DEPLOYMENT_SCHEMA = 'rc3-conservative-actor-only-206-v1'
FEATURE_SCHEMA = 'original202 + mean_over10,max_over10,remaining_fraction,unreachable_fraction'


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tensor_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _condition(value):
    if type(value) is bool:
        return value
    if value == 'ON':
        return True
    if value == 'OFF':
        return False
    raise ValueError('Use the explicit ON/OFF condition or bool')


def _seed(seed, purpose):
    _require(type(seed) is int and seed >= 0, 'Use a nonnegative integer seed')
    return int(hashlib.sha256(f'rc3-conservative-206:{seed}:{purpose}'.encode()).hexdigest()[:15], 16)


def _read_sl256_payload():
    """One exact restored artifact, with immutable upstream bytes checked."""
    path = ROOT / SL_PATH
    _require(file_sha256(path) == SL_SHA256, 'SL256 file hash changed or is an unrestored LFS pointer')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    _require(payload['schema'] == 'integrated-D1-actor-v1' and payload['architecture'] == 'base',
             'Expected the frozen base-only historical export')
    _require(payload['checkpoint_role'] == 'SL256' and payload['seed'] == 11622026
             and payload['update'] == 256, 'SL256 role/seed/endpoint mismatch')
    _require(payload['base_tensor_sha'] == payload['actor_tensor_sha'] == SL_TENSOR_SHA256
             and tensor_digest(payload['model']) == SL_TENSOR_SHA256, 'SL256 tensors differ')
    source = payload['training_source']
    _require(all(source.get(k) == v for k, v in UPSTREAM.items()), 'SL256 upstream lineage differs')
    for path_key, sha_key in (('path', 'checkpoint_sha256'), ('sidecar_path', 'sidecar_sha256'),
                              ('commit_path', 'commit_sha256')):
        _require(file_sha256(ROOT / source[path_key]) == source[sha_key], 'Upstream SL256 evidence changed')
    return payload


def load_frozen_sl():
    """Plain historical SL256 baseline; no new206 feature computation."""
    tick = time.perf_counter()
    from rc3_global_local.policy import GlobalLocalPolicy
    payload = _read_sl256_payload()
    with torch.device('meta'):
        model = GlobalLocalPolicy()
    model.load_state_dict(payload['model'], strict=True, assign=True)
    model.eval().requires_grad_(False)
    model.base_identity = dict(path=SL_PATH, sha256=SL_SHA256, tensor_sha256=SL_TENSOR_SHA256,
                               training_source=dict(payload['training_source']))
    model.loading_seconds = time.perf_counter() - tick
    return model


def load_actor(condition, seed=11622026, *, max_adjustment=0.25):
    """Load the exact seed11622026 SL256, then create equal ON/OFF residuals.

    ``seed`` controls only the NEW residual initialization.  It never selects a
    different supervised endpoint.  The loading duration is separately exposed;
    it is not silently added to every hot inference request.
    """
    tick = time.perf_counter()
    enabled = _condition(condition)
    init_seed = _seed(seed, 'residual_init')
    from rc3_integrated_rppo.model import Actor as HistoricalFeatureEncoder
    payload = _read_sl256_payload()
    with torch.random.fork_rng():
        torch.manual_seed(init_seed)
        encoder = HistoricalFeatureEncoder(payload['model'], 11622026, device='cpu')
        # These methods use only encoder.base and base_sha.  Avoid carrying the
        # unused historical 202-dimensional residual into either runtime model.
        del encoder.head
        actor = LookaheadResidualActor(encoder, distances_enabled=enabled,
                                       max_adjustment=max_adjustment)
    _require(tensor_digest(actor.encoder.base.state_dict()) == SL_TENSOR_SHA256, 'Loaded base differs')
    actor.base_identity = dict(path=SL_PATH, sha256=SL_SHA256, tensor_sha256=SL_TENSOR_SHA256,
                               training_source=dict(payload['training_source']))
    actor.initialization_seed = seed
    actor.loading_seconds = time.perf_counter() - tick
    actor.eval()
    _assert_frozen(actor)
    return actor


def _assert_frozen(actor):
    _require(not actor.encoder.training and not actor.encoder.base.training,
             'Frozen encoder/base entered training mode')
    _require(all(not parameter.requires_grad and parameter.grad is None
                 for parameter in actor.encoder.parameters()), 'Frozen encoder gained gradients')
    _require(tensor_digest(actor.encoder.base.state_dict()) == SL_TENSOR_SHA256,
             'Frozen SL256 parameters changed')


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def rollout(problem, actor, critic=None, *, greedy=False, sampling_seed=0,
            check=None, equivalence=False, state_audit=None):
    """Collect exactly one complete trace from THIS policy, without an update.

    The timer starts with an already constructed ``problem``; ``plan_request``
    is the public full-request wrapper when input preparation must be measured.
    States are detached and contain the new 206-dimensional schema exclusively.
    Both raw and H plans are independently exported and constraint-validated.
    """
    from rc3_global_local.environment import Environment
    from rc3_global_local.terminal import finish, export
    from rc3_global_local.validation import validate
    from rc3_nav_value.r1.oracle import _instrument, _empty_internal

    tick = time.perf_counter()
    check = (lambda: None) if check is None else check
    check()
    _assert_frozen(actor)
    _require(type(greedy) is bool and type(equivalence) is bool, 'Explicit bool mode required')
    rng = torch.Generator(device='cpu').manual_seed(_seed(sampling_seed, 'action_sampling'))
    before = tensor_digest(actor.correction.state_dict())
    stats = dict(full_H_calls=0, export_attempts=0, exports=0,
                 validation_attempts=0, validation_completed=0)
    internal = _empty_internal()
    components = dict(encoder_and_provider_prepare_s=0., dynamic_features_s=0.,
                      actor_forward_sampling_s=0., critic_forward_s=0.,
                      terminal_pipeline_s=0., raw_export_s=0., raw_validation_s=0.)

    def terminal(p, raw):
        check()
        start = time.perf_counter()
        with _instrument(internal):
            result = finish(p, raw, stats)
        components['terminal_pipeline_s'] += time.perf_counter() - start
        return result

    env = Environment(problem, terminal)
    features, logits, actions, values, critic_predictions = [], [], [], [], []
    audit = []
    independent_state_audits = []
    with torch.no_grad():
        start = time.perf_counter()
        encoded = actor.encode(problem) if not env.done else None
        components['encoder_and_provider_prepare_s'] = time.perf_counter() - start
        while not env.done:
            check()
            start = time.perf_counter()
            f = actor.features(env, encoded)
            components['dynamic_features_s'] += time.perf_counter() - start
            if state_audit is not None:
                state_check = state_audit(env, f)
                if state_check is not None:
                    independent_state_audits.append(state_check)
            start = time.perf_counter()
            z = actor.logits(f)
            if equivalence:
                _require(torch.equal(z, f['z0']) and torch.equal(torch.isfinite(z), f['mask']),
                         'Zero correction does not exactly reproduce SL logits/mask')
            _require(z.device.type == 'cpu', 'This admitted runtime uses CPU inference/sampling')
            lp = torch.log_softmax(z, dim=-1)
            action = int(z.argmax()) if greedy else int(torch.multinomial(lp.exp(), 1, generator=rng))
            components['actor_forward_sampling_s'] += time.perf_counter() - start
            start = time.perf_counter()
            if critic is None:
                value = 0.
                prediction = None
            else:
                prediction = critic(f['v'], len(problem.C), problem.penalty)
                value = float(prediction['value'])
                _require(all(bool(torch.isfinite(v).all()) for v in prediction.values()),
                         'Nonfinite rollout Critic prediction')
                prediction = _json_value(prediction)
            components['critic_forward_s'] += time.perf_counter() - start
            left = env.remaining.copy()
            served = np.flatnonzero(left & problem.C[:, action])
            after = left & ~problem.C[:, action]
            extra = f['post_service_distances']
            _require(int(extra['remaining_after_count'][action]) == int(after.sum()),
                     'Post-service remaining set count differs from Environment')
            audit.append(dict(
                action_local_index_0based=action, action_local_node=action + 1,
                action_global_candidate_id=int(problem.candidate_ids[action]),
                remaining_tasks_global_before=problem.task_ids[np.flatnonzero(left)].tolist(),
                hypothetical_served_tasks_global=problem.task_ids[served].tolist(),
                hypothetical_remaining_tasks_global=problem.task_ids[np.flatnonzero(after)].tolist(),
                remaining_after_count=int(extra['remaining_after_count'][action]),
                unreachable_remaining_count=int(extra['unreachable_remaining_count'][action]),
                mean_distance_m=float(extra['raw_mean'][action]) if math.isfinite(float(extra['raw_mean'][action])) else None,
                max_distance_m=float(extra['raw_max'][action]) if math.isfinite(float(extra['raw_max'][action])) else None,
                nonfinite_distance_encoding='null requires the explicit unreachable count above',
                provider_semantics='All service providers in the immutable problem; not current mask',
                selected_base_logit=float(f['z0'][action]), selected_actor_logit=float(z[action]),
                selected_probability=float(lp[action].exp())))
            features.append(f)
            logits.append(z.detach().clone())
            actions.append(action)
            values.append(value)
            critic_predictions.append(prediction)
            event = env.step(action + 1)
            _require(event['served_tasks_global'] == problem.task_ids[served].tolist()
                     and event['remaining_tasks_global'] == problem.task_ids[np.flatnonzero(after)].tolist(),
                     'Global task IDs and local candidate IDs were confused')

    raw = [0, *env.prefix, 0]
    result = env.result
    _require(result is not None, 'An incomplete rollout cannot enter PPO')
    result['plan']['method'] = 'ON' if actor.distances_enabled else 'OFF'
    result['plan']['model_identity'] = dict(base=actor.base_identity, correction_tensor_sha256=before,
                                            feature_schema=FEATURE_SCHEMA)
    raw_legs = [float(problem.D[a, b]) for a, b in zip(raw, raw[1:])]
    targets = remaining_cost_targets(raw_legs, result['J'], len(problem.C), problem.penalty)
    _require(np.allclose(env.rewards if actions else [], targets['dense_rewards'], rtol=0., atol=1e-7),
             'Real dense rewards disagree with remaining-cost accounting')
    _require(abs(sum(env.rewards) + result['J']) < 1e-7, 'Complete return is not -H-final J')
    _require(abs(targets['J_raw'] - result['raw_J']) < 1e-7, 'Raw legs/cost mismatch')
    check()
    start = time.perf_counter()
    raw_plan = export(problem, raw, 'conservative_raw')
    components['raw_export_s'] = time.perf_counter() - start
    start = time.perf_counter()
    raw_certificate = validate(problem, raw_plan)
    components['raw_validation_s'] = time.perf_counter() - start
    _require(raw_certificate['valid_model_plan'], f'Invalid raw plan: {raw_certificate}')
    raw_plan['validation'] = raw_certificate
    _assert_frozen(actor)
    _require(before == tensor_digest(actor.correction.state_dict()), 'Rollout changed correction parameters')
    _require(all(math.isfinite(float(result[k])) for k in ('J', 'L', 'K', 'raw_J', 'raw_L', 'raw_K')),
             'Nonfinite complete route metrics')
    batch = pack_candidate_states(features)
    record = dict(schema='rc3-conservative-real-rollout-v1', problem=problem.description(),
                  input_identity=problem.identity, parent_identity=problem.parent.identity,
                  condition='ON' if actor.distances_enabled else 'OFF', sampling_seed=sampling_seed,
                  greedy=greedy, base_identity=actor.base_identity, correction_tensor_sha256=before,
                  raw_route=raw, route=list(result['route']), raw_global_route=problem.global_route(raw),
                  global_route=problem.global_route(result['route']), trace=list(env.trace),
                  actions=[a + 1 for a in actions], actions_local_indices_0based=actions,
                  actions_global_candidate_ids=[int(problem.candidate_ids[a]) for a in actions],
                  rewards=list(env.rewards), targets=targets, V_old=values,
                  critic_predictions=critic_predictions, post_service_audit=audit,
                  independent_state_audits=independent_state_audits,
                  metrics={k: result[k] for k in ('J', 'L', 'K', 'raw_J', 'raw_L', 'raw_K')},
                  plan=result['plan'], raw_plan=raw_plan, counts=stats,
                  raw_additional_export_attempts=1, raw_additional_validation_attempts=1,
                  internal_search=internal, components_s=components,
                  H_s=result['H_s'], export_s=result['export_s'], validation_s=result['validation_s'],
                  rng_end_sha256=hashlib.sha256(rng.get_state().numpy().tobytes()).hexdigest(),
                  timing_boundary='Already-prepared problem through both exports/validation, target and state packing; excludes caller input preparation, model loading and persistence')
    record = _json_value(record)
    # Ensure persisted evidence cannot silently serialize inf/NaN placeholders.
    json.dumps(record, ensure_ascii=False, allow_nan=False)
    record['prepared_problem_request_s'] = time.perf_counter() - tick
    check()
    return dict(batch=batch,
                old_logits=torch.stack(logits) if logits else torch.empty((0, len(problem.candidate_ids))),
                actions=torch.tensor(actions, dtype=torch.long), V_old=torch.tensor(values, dtype=torch.float32),
                G=torch.tensor(targets['returns'], dtype=torch.float32), targets=targets,
                task_count=len(problem.C), actor_tensor_sha=before, record=record)


collect = rollout


def audit_real_feature_state(env, features):
    """Slow independent engineering check; never called in timed deployment.

    Cross-check global IDs against the parent's per-arm table and obstacle
    distances, then recompute task-wise nearest providers with scalar loops.
    Providers deliberately include candidates excluded by the CURRENT mask.
    This inspects only the explicitly supplied, already admitted state.
    """
    problem = env.problem
    global_tasks = np.asarray(problem.task_ids, dtype=np.int64)
    global_candidates = np.asarray(problem.candidate_ids, dtype=np.int64)
    parent_effective = problem.parent.effective
    expected_coverage = parent_effective[global_tasks[:, None], global_candidates[None, :]].any(-1)
    _require(np.array_equal(problem.C, expected_coverage), 'Global task/candidate IDs do not match per-arm coverage')
    nodes = global_candidates + 1
    parent_D = problem.parent.D
    expected_D = np.empty_like(problem.D)
    expected_D[0, 0] = parent_D[problem.entry, problem.exit]
    expected_D[0, 1:] = parent_D[problem.entry, nodes]
    expected_D[1:, 0] = parent_D[nodes, problem.exit]
    expected_D[1:, 1:] = parent_D[np.ix_(nodes, nodes)]
    _require(np.array_equal(problem.D, expected_D), 'Global candidate IDs do not match navigation node IDs')
    extra = features['post_service_distances']
    legal = env.mask()
    _require(np.array_equal(extra['valid_action'], legal), 'Feature action mask differs')
    checked_pairs = 0
    providers_outside_current_mask = set()
    records = []
    for action in np.flatnonzero(legal):
        after = np.asarray(env.remaining, dtype=bool) & ~expected_coverage[:, action]
        distances = []
        for task in np.flatnonzero(after):
            providers = np.flatnonzero(expected_coverage[task] & np.isfinite(problem.D[1:, 0]))
            providers_outside_current_mask.update(int(global_candidates[j]) for j in providers if not legal[j])
            available = [float(parent_D[int(global_candidates[action]) + 1,
                                         int(global_candidates[j]) + 1]) for j in providers]
            distances.append(min(available, default=math.inf))
            checked_pairs += 1
        count = len(distances)
        unreachable = sum(not math.isfinite(value) for value in distances)
        mean = math.fsum(distances) / count if count and not unreachable else (math.inf if count else 0.)
        maximum = max(distances, default=0.)
        _require(int(extra['remaining_after_count'][action]) == count
                 and int(extra['unreachable_remaining_count'][action]) == unreachable,
                 'Post-service task count or unreachable count differs')
        for key, expected in (('raw_mean', mean), ('raw_max', maximum)):
            actual = float(extra[key][action])
            _require((math.isinf(actual) and actual > 0 and math.isinf(expected))
                     or math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10),
                     f'Independent provider distance check failed: {key}')
        records.append(dict(candidate_global_id=int(global_candidates[action]),
                            remaining_task_ids_global=global_tasks[np.flatnonzero(after)].tolist(),
                            unreachable_task_count=unreachable,
                            mean_distance_m=mean if math.isfinite(mean) else None,
                            max_distance_m=maximum if math.isfinite(maximum) else None))
    return dict(status='PASS', input_identity=problem.identity,
                checked_legal_candidates=int(legal.sum()), checked_candidate_task_pairs=checked_pairs,
                providers_outside_current_mask_global=sorted(providers_outside_current_mask),
                current_mask_was_not_used_to_crop_providers=True,
                global_id_coverage_and_navigation_check=True, per_candidate=records)


def sl_route(problem, model, *, check=None):
    """Greedy plain-SL comparison with the same raw/H validation and full timer.

    It deliberately never prepares the new lookahead features.  As with bare
    ``rollout``, its input is an already prepared problem; external input and
    cold-load timing must be added through real enclosing measurements.
    """
    from rc3_global_local.environment import Environment
    from rc3_global_local.terminal import finish, export
    from rc3_global_local.validation import validate
    from rc3_nav_value.r1.oracle import _instrument, _empty_internal
    start = time.perf_counter()
    check = (lambda: None) if check is None else check
    check()
    _require(not model.training and all(not p.requires_grad and p.grad is None for p in model.parameters()),
             'Plain SL baseline is not frozen')
    _require(tensor_digest(model.state_dict()) == SL_TENSOR_SHA256, 'Plain SL baseline differs')
    stats = dict(full_H_calls=0, export_attempts=0, exports=0,
                 validation_attempts=0, validation_completed=0)
    internal = _empty_internal()
    def terminal(p, raw):
        check()
        with _instrument(internal):
            return finish(p, raw, stats)
    env = Environment(problem, terminal)
    with torch.no_grad():
        encoded = model.encode(problem) if not env.done else None
        while not env.done:
            check()
            logits = model.score(env, encoded)
            _require(bool(torch.isfinite(logits[torch.as_tensor(env.mask())]).all()), 'Invalid plain SL logits')
            env.step(int(logits.argmax()) + 1)
    result = env.result
    raw = [0, *env.prefix, 0]
    raw_plan = export(problem, raw, 'FrozenSL_raw')
    raw_plan['validation'] = validate(problem, raw_plan)
    _require(raw_plan['validation']['valid_model_plan'], 'Invalid plain SL raw plan')
    _require(tensor_digest(model.state_dict()) == SL_TENSOR_SHA256, 'Plain SL changed during inference')
    record = _json_value(dict(schema='rc3-conservative-frozen-sl-rollout-v1', study_method='FrozenSL',
                    input_identity=problem.identity, parent_identity=problem.parent.identity,
                    problem=problem.description(), raw_route=raw, route=result['route'],
                    raw_global_route=problem.global_route(raw), global_route=problem.global_route(result['route']),
                    trace=env.trace, plan=result['plan'], raw_plan=raw_plan,
                    metrics={k: result[k] for k in ('J', 'L', 'K', 'raw_J', 'raw_L', 'raw_K')},
                    counts=stats, internal_search=internal, model_identity=model.base_identity,
                    H_s=result['H_s'], export_s=result['export_s'], validation_s=result['validation_s'],
                    timing_boundary='Already-prepared problem through raw/H export and validation; excludes caller input preparation, loading and persistence'))
    json.dumps(record, allow_nan=False)
    record['prepared_problem_request_s'] = time.perf_counter() - start
    check()
    return record


def plan_request(problem_factory, *, actor=None, actor_factory=None, critic=None, **kwargs):
    """Measure a real whole request, including input load/preparation and cold load.

    A hot request supplies ``actor``; a cold model request supplies
    ``actor_factory``.  The caller controls navigation cache policy inside the
    admitted problem factory.  Persistence is timed by the external ledger.
    """
    _require((actor is None) != (actor_factory is None), 'Provide exactly one actor or actor_factory')
    start = time.perf_counter()
    check = kwargs.get('check')
    if check is not None:
        check()
    tick = time.perf_counter()
    problem = problem_factory()
    preparation_s = time.perf_counter() - tick
    if check is not None:
        check()
    tick = time.perf_counter()
    if actor_factory is not None:
        actor = actor_factory()
    model_load_s = time.perf_counter() - tick if actor_factory is not None else 0.
    if check is not None:
        check()
    output = rollout(problem, actor, critic, **kwargs)
    output['record']['input_preparation_s'] = preparation_s
    output['record']['model_startup_s'] = model_load_s
    output['record']['service_time_before_IO_s'] = time.perf_counter() - start
    output['record']['full_request_timing_boundary'] = 'Before problem factory through input preparation, optional model load, provider preparation, features, actions, H, raw/H export, validation and evidence construction; persistence measured externally'
    return output


def export_actor(path, actor, metadata=None):
    """Write one immutable correction-only artifact, never Critic or Adam."""
    _assert_frozen(actor)
    metadata = {} if metadata is None else dict(metadata)
    json.dumps(metadata, allow_nan=False)
    _require(not any(word in str(key).lower() for key in metadata for word in ('critic', 'optimizer', 'adam')),
             'Deployment metadata may not carry training-state payloads')
    package = dict(schema=DEPLOYMENT_SCHEMA, feature_schema=FEATURE_SCHEMA, input_dim=206,
                   hidden=actor.correction.head[0].out_features,
                   max_adjustment=actor.correction.max_adjustment,
                   distances_enabled=actor.distances_enabled, initialization_seed=actor.initialization_seed,
                   base=dict(actor.base_identity), metadata=metadata,
                   correction_tensor_sha256=tensor_digest(actor.correction.state_dict()),
                   correction_state={name: value.detach().cpu().clone()
                                     for name, value in actor.correction.state_dict().items()},
                   critic_included=False, optimizer_included=False)
    path = Path(path)
    _require(not path.exists(), 'Actor export is immutable; destination already exists')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending.' + uuid.uuid4().hex)
    try:
        with temporary.open('xb') as stream:
            torch.save(package, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link provides no-clobber publication; removing our known temporary
        # leaves the same complete bytes at the final immutable path.
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return dict(path=str(path), sha256=file_sha256(path), bytes=path.stat().st_size,
                correction_tensor_sha256=package['correction_tensor_sha256'],
                base=dict(actor.base_identity), schema=DEPLOYMENT_SCHEMA,
                condition='ON' if actor.distances_enabled else 'OFF')


def load_deployed_actor(path, *, expected_sha256):
    """Validate a new206 actor-only artifact and reconstruct exact frozen SL256."""
    start = time.perf_counter()
    _require(type(expected_sha256) is str and len(expected_sha256) == 64
             and file_sha256(path) == expected_sha256, 'Actor-only file hash differs')
    package = torch.load(path, map_location='cpu', weights_only=True)
    allowed = {'schema', 'feature_schema', 'input_dim', 'hidden', 'max_adjustment',
               'distances_enabled', 'initialization_seed', 'base', 'metadata',
               'correction_tensor_sha256', 'correction_state', 'critic_included', 'optimizer_included'}
    _require(set(package) == allowed, 'Unexpected training objects or missing actor-only metadata')
    _require(package['schema'] == DEPLOYMENT_SCHEMA and package['feature_schema'] == FEATURE_SCHEMA
             and package['input_dim'] == 206 and package['hidden'] == 64, 'Incompatible actor-only schema')
    _require(package['critic_included'] is False and package['optimizer_included'] is False,
             'Critic/optimizer forbidden in deployment')
    _require(package['base']['path'] == SL_PATH and package['base']['sha256'] == SL_SHA256
             and package['base']['tensor_sha256'] == SL_TENSOR_SHA256, 'Unexpected frozen base')
    _require(tensor_digest(package['correction_state']) == package['correction_tensor_sha256'],
             'Correction tensor hash differs')
    actor = load_actor(package['distances_enabled'], package['initialization_seed'],
                       max_adjustment=package['max_adjustment'])
    _require(package['base'] == actor.base_identity, 'Deployment SL lineage differs')
    actor.correction.load_state_dict(package['correction_state'], strict=True)
    _require(tensor_digest(actor.correction.state_dict()) == package['correction_tensor_sha256'],
             'Loaded correction differs')
    actor.eval().requires_grad_(False)
    actor.deployment_identity = dict(path=str(path), sha256=expected_sha256,
                                     metadata=package['metadata'])
    actor.loading_seconds = time.perf_counter() - start
    _assert_frozen(actor)
    return actor
