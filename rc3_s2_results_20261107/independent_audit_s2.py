"""Independent saved-evidence audit. No model, optimizer, scene or solver objects.

Streams JSONL one record at a time; loads only one saved checkpoint at a time.
The only writes are new audit_seed_<seed>.json files in this report directory.
"""
from pathlib import Path
import argparse
import collections
import copy
import datetime
import gc
import hashlib
import json
import math
import random
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01'
PROTOCOL = '0c5bc7d7a9557edf2af76f51a9f418550f0b4ce1ec6a8b47ddea95c6256e78bb'
SOURCE = '0f0ffd2d2dda2fefefcfcb5cb26dc5197b5d11bd1332b0a5546992409049bfe0'
SEEDS = (11622026, 11622027, 11622028)


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def tensor_digest(model):
    h = hashlib.sha256()
    for k, v in sorted(model.items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class Audit:
    def __init__(self, seed):
        self.seed = seed
        self.counts = collections.Counter()
        self.failures = []
        self.max_errors = collections.defaultdict(float)
        self.artifacts = {}

    def ck(self, value, group, detail=''):
        self.counts[group] += 1
        if not bool(value):
            self.failures.append(dict(group=group, detail=str(detail)))

    def close(self, a, b, group, detail='', tol=1e-7):
        error = abs(float(a) - float(b))
        self.max_errors[group] = max(self.max_errors[group], error)
        self.ck(math.isfinite(error) and error <= tol, group, detail)

    def record(self, path):
        result = digest(path)
        self.artifacts[path.relative_to(ROOT).as_posix()] = result
        return result

    def derive(self, phase, u=-1, slot=-1, role='data'):
        payload = json.dumps([PROTOCOL, phase, self.seed, u, slot, role],
                             ensure_ascii=False, separators=(',', ':')).encode('utf8')
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little') % (2**63)


def stream(path):
    if path.exists():
        with path.open('rb') as f:
            for number, line in enumerate(f, 1):
                yield number, line, json.loads(line.decode('utf8'))


def schedule(rng, registry, u):
    result = []
    for slot in range(4):
        pi, ti = int(rng.integers(96)), int(rng.integers(8))
        p = registry['S2_train'][pi]
        if u % 2 == 0:
            w = h = 10
            x = y = 0
            ep, li = 'closed_depot', None
        else:
            li = ((u - 1) // 2) * 4 + slot
            w, h = ((4, 3), (6, 6), (8, 8))[li % 3]
            x, y = int(rng.integers(11 - w)), int(rng.integers(11 - h))
            ep = 'closed_depot' if li % 2 == 0 else 'open_first_last_connected'
        result.append(dict(role='train', parent=pi, task_draw=ti,
            map_seed=p['map_seed'], task_seed=p['task_seeds'][ti], mode=p['mode'],
            A_condition=(pi + ti) % 3, width=w, height=h, x0=x, y0=y,
            endpoints=ep, update=u, slot=slot, local_index=li))
    return result


def run(seed):
    started = time.perf_counter()
    a = Audit(seed)
    base = RUN / f'seed_{seed}'
    admission = RUN / f'stage_admission/S2/{seed}'
    complete = read(admission / 'COMPLETE.json')
    final = read(base / 'S2/FINAL.json')
    fork = read(base / 'S2/FORK_RECEIPT.json')
    auth_path = ROOT / 'rc3_global_local_execution_v3/authorizations/20261107_S2/S2_AUTHORIZATION.json'
    authority = read(auth_path)
    manifest_path = ROOT / 'rc3_navigation_compat_20261106/SOURCE_BINDING_UPDATED.json'
    binding = read(manifest_path)
    a.ck(a.record(manifest_path) == SOURCE, 'identity')
    for name, h in binding['source_sha256'].items():
        a.ck(a.record(ROOT / name) == h, 'identity', name)
    for name, expected in ((binding['protocol_path'], PROTOCOL),
                           (binding['registry_path'], binding['registry_sha256'])):
        a.ck(a.record(ROOT / name) == expected, 'identity', name)
    a.ck(complete['status'] == 'COMPLETE' and complete['stage_key'] == f'S2/{seed}', 'identity')
    a.ck(complete['error'] is None and not (admission / 'FAILED.json').exists(), 'identity')
    a.ck(final['status'] == 'COMPLETE_FIXED_ENDPOINTS', 'identity')
    a.ck(complete['science_binding']['source_manifest_sha256'] == SOURCE, 'identity')
    a.ck(complete['science_binding']['protocol_sha256'] == PROTOCOL, 'identity')
    a.ck(a.record(admission / 'AUTHORIZATION.json') == complete['authorization_record_sha256'], 'identity')
    a.ck(a.record(admission / 'START.json') == complete['start_record_sha256'], 'identity')
    savedauth = read(admission / 'AUTHORIZATION_SOURCE.json')
    a.ck(savedauth['authorization_sha256'] == a.record(auth_path), 'identity')
    a.ck(savedauth['declaration'] == authority, 'identity')
    expectedparents = {'S1-A', *(f'S1-B/{s}' for s in SEEDS), *(f'S2/{s}' for s in SEEDS[:SEEDS.index(seed)])}
    a.ck(set(complete['predecessors']) == expectedparents, 'identity')
    for key, h in complete['predecessors'].items():
        a.ck(a.record(RUN / 'stage_admission' / key / 'COMPLETE.json') == h, 'identity', key)
    for name, h in complete['artifacts'].items():
        a.ck(a.record(RUN / name) == h, 'completion_artifacts', name)

    sourcepath = base / 'S1/update_0256.pt'
    source = torch.load(sourcepath, map_location='cpu', weights_only=False)
    source_model = source['model']
    sourcehash = tensor_digest(source_model)
    a.ck(sourcehash == fork['common_tensor_sha256'] == final['frozen_SL_tensor_sha256'], 'fork')
    a.ck(a.record(sourcepath) == fork['source_S1_checkpoint_sha256'], 'fork')
    a.ck(fork['source_S1_optimizer_inherited'] is False and fork['frozen_SL'] is True, 'fork')
    a.ck(set(fork['branches']) == {'frozen_SL', 'RL', 'continued_SL'}, 'fork')
    for name, value in fork['branches'].items():
        a.ck(value['tensor_sha256'] == sourcehash and value['optimizer_state_entries'] == 0, 'fork', name)
    source_rng = source['global_rng']['torch'].clone()
    del source
    registry = read(ROOT / binding['registry_path'])
    teacher_index = read(RUN / 'teachers256/POOL_INDEX.json')
    teacher_labels = [read(RUN / 'teachers256' / x['file']) for x in teacher_index['labels']]
    rng_states = {}
    prefixes = {}
    update_hashes = {}
    counters = collections.Counter()
    role_events = {role: collections.Counter() for role in ('RL', 'development')}
    terminal_starts = {}
    event_views = collections.Counter()
    generated_specs = []
    # Read event chronology and preserve exact byte-prefix digests for checkpoint IO closure.
    for phase in ('RL', 'continued_SL'):
        eventpath = base / phase / 'events.jsonl'
        h = hashlib.sha256()
        prefixes[(phase, 'events.jsonl')] = {h.hexdigest(): 0}
        for n, line, row in stream(eventpath):
            h.update(line)
            prefixes[(phase, 'events.jsonl')][h.hexdigest()] = n
            event = row['event']
            counters[f'event:{event}'] += 1
            if event == 'view':
                a.ck(row['spec']['role'] in ('train', 'development'), 'data_roles')
                event_views[row['identity']] += 1
            if event == 'task_scene_start':
                generated_specs.append(row['spec'])
                a.ck(row['spec']['role'] in ('train', 'development'), 'data_roles')
            if event.startswith('terminal_'):
                role = row['role']
                role_events[role][event] += 1
                if event == 'terminal_start':
                    terminal_starts[(role, row['ordinal'])] = (row['problem'], tuple(row['raw_route']))
        if eventpath.exists():
            a.artifacts[eventpath.relative_to(ROOT).as_posix()] = h.hexdigest()
    # Saved terminals are checked statically. No route, collision or reachability solver runs.
    terminals = {}
    for file in sorted((base / 'RL/plans').glob('*.json')):
        saved = read(file)
        r, p, problem = saved['result'], saved['result']['plan'], saved['problem']
        role, ordinal = file.stem.rsplit('_', 1)
        key = (problem['identity'], tuple(r['raw_route']))
        a.ck(terminal_starts[(role, int(ordinal))] == key, 'terminal_identity', file.name)
        a.ck(problem['identity'] == p['problem_identity'], 'terminal_identity')
        a.ck(problem['parent_identity'] == p['parent_identity'], 'terminal_identity')
        a.ck(p['target_task_ids_global'] == problem['task_ids_global'], 'terminal_identity')
        for name in ('J', 'L', 'K', 'raw_J', 'raw_L', 'raw_K', 'H_s', 'export_s', 'validation_s'):
            a.ck(math.isfinite(r[name]), 'terminal_finite', name)
        a.close(r['J'], r['L'] + .75 * r['K'], 'terminal_objective')
        a.close(r['raw_J'], r['raw_L'] + .75 * r['raw_K'], 'terminal_objective')
        a.close(r['J'], p['objective_length_plus_stops_m'], 'terminal_objective')
        a.close(r['L'], p['route_length_m'], 'terminal_objective')
        a.close(r['L'], sum(p['route_leg_costs_m']), 'terminal_leg_sum', tol=1e-6)
        polyline = p['route_polyline']
        length = sum(math.hypot(b[0] - x[0], b[1] - x[1]) for x, b in zip(polyline, polyline[1:]))
        a.close(r['L'], length, 'saved_polyline_length', tol=1e-6)
        a.ck(r['J'] <= r['raw_J'] + 1e-7, 'H_monotonic')
        a.ck(r['K'] == len(r['route']) - 2 == len(p['service_stops']), 'terminal_route')
        a.ck(r['raw_K'] == len(r['raw_route']) - 2, 'terminal_route')
        a.ck(p['status'] == 'complete' and p['unserved_task_ids'] == [], 'saved_certificate')
        validation = p['validation']
        for name in ('valid_model_plan', 'all_tasks_served', 'effective_same_arm_F_and_A', 'endpoint_constraints_satisfied'):
            a.ck(validation[name] is True, 'saved_certificate', name)
        a.ck(validation['collision_count'] == 0 and validation['errors'] == [], 'saved_certificate')
        a.ck(p['physical_joint_trajectory_certified'] is False and validation['physical_joint_trajectory_certified'] is False, 'certificate_scope')
        a.ck(p['arm_joint_trajectories'] is None, 'certificate_scope')
        a.ck({v['task_id'] for v in p['assignments']} == set(problem['task_ids_global']), 'saved_assignment_coverage')
        a.ck(p['entry_global_node'] == problem['entry_global_node'] and p['exit_global_node'] == problem['exit_global_node'], 'terminal_identity')
        if key in terminals:
            old = terminals[key]
            a.close(r['J'], old[0], 'repeated_route_determinism')
            a.ck(tuple(r['route']) == old[2], 'repeated_route_determinism')
        terminals[key] = (r['J'], r['raw_J'], tuple(r['route']))
        counters[f'saved_plan:{role}'] += 1
    for phase in ('RL', 'continued_SL'):
        rng = np.random.Generator(np.random.PCG64(a.derive('S2_data' if phase == 'RL' else 'continuedSL_data')))
        h = hashlib.sha256()
        prefixes[(phase, 'updates.jsonl')] = {h.hexdigest(): 0}
        rng_states[phase] = {0: copy.deepcopy(rng.bit_generator.state)}
        elapsed_before = -1.
        for n, line, row in stream(base / phase / 'updates.jsonl'):
            h.update(line)
            prefixes[(phase, 'updates.jsonl')][h.hexdigest()] = n
            u = n - 1
            a.ck(row['phase'] == phase and row['update'] == u, 'update_cursor')
            a.ck(row['cursor_after'] == row['weight_version_after'] == n, 'update_cursor')
            a.ck(row['optimizer_updated'] is True and len(row['instances']) == 4, 'update_cursor')
            a.ck(row['stage_elapsed_s'] > elapsed_before, 'update_wall')
            elapsed_before = row['stage_elapsed_s']
            a.ck(0 <= row['gradient_norm_after_clip'] <= 1. + 1e-5, 'gradient_clip')
            a.ck(math.isfinite(row['loss']) and all(math.isfinite(v) for v in row['gradient_norms'].values()), 'update_finite')
            a.close(row['loss'], sum(i['loss'] for i in row['instances']) / 4, 'batch_loss')
            update_hashes[(phase, n)] = row['model_tensor_sha256']
            if phase == 'RL':
                a.ck(row['sampling_specs'] == schedule(rng, registry, u), 'schedule')
                for slot, instance in enumerate(row['instances']):
                    a.ck(instance['update'] == u and instance['slot'] == slot and instance['weight_version'] == u, 'instance_binding')
                    a.ck(instance['mixture_weights'] == [1., .25, .25] and instance['gamma'] == 1., 'instance_binding')
                    rr = instance['rows']
                    a.ck(len(rr) == 3, 'row_count')
                    returns = []
                    for role, z in zip(('primary', 'forced_low', 'forced_high'), rr):
                        counters['RL_rows'] += 1
                        a.ck(z['role'] == role and z['sampling_seed'] == a.derive('RL', u, slot, role), 'row_rng')
                        a.close(sum(z['rewards']), z['return_'], 'reward_sum')
                        a.close(z['return_'], -z['J'], 'reward_negative_J')
                        a.ck(z['raw_route'] == [0, *z['actions'], 0], 'raw_route')
                        a.ck(len(z['actions']) == len(z['legal_counts']) == len(z['masks_local_nodes']) == len(z['logprobs']) == len(z['entropies']), 'row_alignment')
                        for j, (act, count, mask, lp, ent) in enumerate(zip(z['actions'], z['legal_counts'], z['masks_local_nodes'], z['logprobs'], z['entropies'])):
                            counters['RL_actions'] += 1
                            a.ck(count == len(mask) and mask == sorted(set(mask)), 'saved_mask')
                            a.ck(act in mask, 'saved_mask_action')
                            if j == 0 and role != 'primary':
                                a.ck(lp is None and ent is None and act == (mask[0] if role == 'forced_low' else mask[-1]), 'forced_first')
                            else:
                                a.ck(lp is not None and math.isfinite(lp) and lp <= 1e-7, 'saved_logprob')
                                a.ck(ent is not None and math.isfinite(ent) and ent >= -1e-7, 'saved_entropy')
                        a.close(sum(x for x in z['logprobs'] if x is not None), z['logprob_sum'], 'float32_logsum', tol=.002)
                        metric = terminals[(instance['problem'], tuple(z['raw_route']))]
                        a.close(z['J'], metric[0], 'row_saved_terminal')
                        a.close(z['raw_J'], metric[1], 'row_saved_terminal')
                        a.ck(tuple(z['final_route']) == metric[2], 'row_saved_terminal')
                        returns.append(z['return_'])
                        counters['row_cache_hits'] += bool(z['terminal_cache_hit'])
                    baselines = [(sum(returns) - r) / 2 for r in returns]
                    advantages = [r - b for r, b in zip(returns, baselines)]
                    for actual, expected in zip(instance['baseline'], baselines):
                        a.close(actual, expected, 'LOO_baseline')
                    for actual, expected in zip(instance['advantages'], advantages):
                        a.close(actual, expected, 'LOO_advantage')
                    loss = sum(-w * adv * z['logprob_sum'] for w, adv, z in zip((1., .25, .25), advantages, rr)) / 1.5
                    a.close(instance['loss'], loss, 'mixed_REINFORCE_objective', tol=1e-6)
            else:
                indices = rng.integers(0, 256, 4).tolist()
                a.ck(row['teacher_indices'] == indices, 'continuedSL_schedule')
                for instance, index in zip(row['instances'], indices):
                    teacher = teacher_labels[index]
                    a.ck(instance['problem'] == teacher['view_identity'] and instance['actions'] == teacher['actions'], 'continuedSL_teacher_lineage')
                    a.ck(instance['decisions'] == len(teacher['actions']) and instance['teacher_or_H_queries'] == 0, 'continuedSL_teacher_lineage')
                    a.ck(instance['label_replay_terminal_calls'] == 1, 'continuedSL_label_replay')
                    counters['SL_actions'] += instance['decisions']
            counters[f'updates:{phase}'] += 1
            if n % 32 == 0:
                rng_states[phase][n] = copy.deepcopy(rng.bit_generator.state)
        a.artifacts[(base / phase / 'updates.jsonl').relative_to(ROOT).as_posix()] = h.hexdigest()
        a.ck(counters[f'updates:{phase}'] == 512, 'fixed_endpoints')
    # Verify all 34 S2 checkpoints without creating policy or optimizer instances.
    parameter_deltas = {}
    final_ledgers = {}
    checkpoint_count = 0
    for phase in ('RL', 'continued_SL'):
        folder = base / phase
        paths = [folder / 'start_from_SL.pt'] + [folder / f'update_{n:04}.pt' for n in range(32, 513, 32)]
        a.ck(set(folder.glob('*.pt')) == set(paths), 'checkpoint_inventory')
        for cp in paths:
            cursor = 0 if cp.name == 'start_from_SL.pt' else int(cp.stem.split('_')[1])
            meta = read(cp.with_suffix('.pt.json'))
            commit = read(cp.with_suffix('.pt.commit.json'))
            a.ck(a.record(cp) == meta['sha256'] == commit['checkpoint_sha256'], 'checkpoint_file_hash', cp.name)
            a.ck(a.record(cp.with_suffix('.pt.json')) == commit['sidecar_sha256'], 'checkpoint_file_hash')
            state = torch.load(cp, map_location='cpu', weights_only=False)
            checkpoint_count += 1
            th = tensor_digest(state['model'])
            a.ck(th == state['model_tensor_sha256'] == meta['model_tensor_sha256'], 'checkpoint_tensor_hash')
            expected_cursors = {'S1': 0, 'RL': 0, 'continued_SL': 0}
            expected_cursors[phase] = cursor
            a.ck(state['cursors'] == meta['cursors'] == commit['cursors'] == expected_cursors, 'checkpoint_cursor')
            a.ck(state['weight_version'] == meta['weight_version'] == cursor, 'checkpoint_cursor')
            a.ck(state['config']['binding'] == meta['binding'], 'checkpoint_binding')
            a.ck(meta['binding']['optimizer_seed'] == seed and meta['binding']['phase'] == phase and meta['binding']['source_manifest_sha256'] == SOURCE, 'checkpoint_binding')
            a.ck(state['model_training'] is True and state['stopped'] is False, 'checkpoint_boundary')
            a.ck(set(state['model']) == set(source_model), 'checkpoint_tensor_keys')
            for name, tensor in state['model'].items():
                a.ck(tensor.shape == source_model[name].shape and tensor.dtype == source_model[name].dtype and bool(torch.isfinite(tensor).all()), 'checkpoint_tensor_finite')
            group = state['optimizer']['param_groups'][0]
            a.ck(group['lr'] == 1e-4 and tuple(group['betas']) == (.9, .999) and group['eps'] == 1e-8 and group['weight_decay'] == 0., 'Adam_recipe')
            if cursor == 0:
                a.ck(th == sourcehash and state['optimizer']['state'] == {}, 'fresh_branch_start')
                a.ck(state['streams']['numpy'] == {} and state['streams']['torch'] == {}, 'fresh_branch_start')
                a.ck(torch.equal(state['global_rng']['torch'], source_rng), 'source_global_RNG')
            else:
                a.ck(th == update_hashes[(phase, cursor)], 'checkpoint_log_endpoint')
                a.ck(len(state['optimizer']['state']) == len(source_model), 'Adam_state')
                for entry in state['optimizer']['state'].values():
                    a.ck(float(entry['step']) == cursor, 'Adam_step')
                    a.ck(bool(torch.isfinite(entry['exp_avg']).all()) and bool(torch.isfinite(entry['exp_avg_sq']).all()), 'Adam_finite')
                data_label = 'S2_data' if phase == 'RL' else 'continuedSL_data'
                a.ck(state['streams']['numpy'] == {data_label: rng_states[phase][cursor]}, 'named_PCG64_state')
            streams = state['streams']
            a.ck(streams['protocol_sha256'] == PROTOCOL and streams['optimizer_seed'] == seed, 'named_rng_binding')
            expected_torchn = 12 * cursor if phase == 'RL' else 0
            a.ck(len(streams['torch']) == len(streams['torch_meta']) == expected_torchn, 'named_torch_inventory')
            a.ck(set(streams['torch']) == set(streams['torch_meta']), 'named_torch_inventory')
            for key, record in streams['torch_meta'].items():
                a.ck(record['phase'] == 'RL' and 0 <= record['update'] < cursor and 0 <= record['slot'] < 4 and record['role'] in ('primary', 'forced_low', 'forced_high'), 'named_torch_metadata')
                a.ck(record['seed'] == a.derive(record['phase'], record['update'], record['slot'], record['role']), 'named_torch_seed')
                a.ck(key == json.dumps([record['phase'], record['update'], record['slot'], record['role'], record['device']], separators=(',', ':')), 'named_torch_metadata')
                torch.Generator(device='cpu').set_state(streams['torch'][key])
                a.ck(streams['torch'][key].dtype == torch.uint8, 'named_torch_state_loadable')
            random.Random().setstate(state['global_rng']['python'])
            np.random.RandomState().set_state(state['global_rng']['numpy'])
            torch.Generator(device='cpu').set_state(state['global_rng']['torch'])
            a.ck(state['global_rng']['cuda'] is None, 'global_RNG_loadable')
            a.ck(state['extra']['journal_sha256'] == commit['journal_sha256'], 'checkpoint_journal')
            for name, value in commit['journal_sha256'].items():
                if value is None:
                    a.ck(cursor == 0, 'checkpoint_journal')
                else:
                    a.ck(value in prefixes[(phase, name)], 'checkpoint_journal', str((phase, cursor, name)))
                    if name == 'updates.jsonl':
                        a.ck(prefixes[(phase, name)].get(value) == cursor, 'checkpoint_journal')
            a.close(commit['snapshot_elapsed_s'], state['extra']['clock']['elapsed_s'], 'checkpoint_clock')
            a.ck(commit['stage_elapsed_after_checkpoint_IO_s'] >= commit['snapshot_elapsed_s'] >= 0, 'checkpoint_clock')
            a.ck(state['extra']['clock']['limit_s'] == 7200 and commit['stage_elapsed_after_checkpoint_IO_s'] < 7200, 'checkpoint_clock')
            a.ck(state['terminal_state']['limits'] == {'RL': 6144, 'development': 2048}, 'terminal_budget_state')
            a.ck(len(state['terminal_state']['cache']) == commit['terminal_cache_entries'] <= 128, 'terminal_budget_state')
            a.ck(state['terminal_state']['stats'] == state['ledger']['terminal'], 'terminal_budget_state')
            if cursor == 512:
                delta = math.sqrt(sum(float((state['model'][k].double() - v.double()).square().sum()) for k, v in source_model.items()))
                norm = math.sqrt(sum(float(v.double().square().sum()) for v in source_model.values()))
                parameter_deltas[phase] = dict(tensor_sha256=th, relative_L2_change_from_S1=delta / norm)
                a.ck(delta > 0. and th != sourcehash, 'endpoint_changed')
                final_ledgers[phase] = copy.deepcopy(state['ledger'])
            del state
            gc.collect()
    a.ck(parameter_deltas['RL']['tensor_sha256'] != parameter_deltas['continued_SL']['tensor_sha256'], 'endpoint_changed')
    resources = complete['cumulative_stage_resources']
    a.ck(set(resources) == {f'{seed}/RL'}, 'shared_resources')
    resource = resources[f'{seed}/RL']
    ledger = resource['ledger']
    a.ck(ledger == final['ledger'] == final_ledgers['continued_SL'], 'shared_resources')
    a.ck(resource['wall_cap_s'] == 7200 and final['shared_S2_wall_s'] <= resource['elapsed_wall_s'] < 7200, 'shared_wall')
    for role in ('RL', 'development'):
        stats = ledger['terminal'][role]
        ev = role_events[role]
        for event, key in (('terminal_start', 'terminal_attempts'), ('terminal_complete', 'terminal_completed'), ('terminal_error', 'terminal_failed'), ('terminal_cache_hit', 'terminal_cache_hits')):
            a.ck(ev[event] == stats[key], 'terminal_event_ledger', role + '/' + key)
        a.ck(stats['terminal_requests'] == stats['terminal_attempts'] + stats['terminal_cache_hits'], 'terminal_query_accounting')
        a.ck(stats['terminal_attempts'] == stats['terminal_completed'] == stats['full_H_calls'] == stats['exports'] == stats['validation_completed'] == counters[f'saved_plan:{role}'], 'terminal_query_accounting')
        a.ck(stats['terminal_failed'] == 0 and stats['terminal_attempts'] <= (6144 if role == 'RL' else 2048), 'terminal_query_accounting')
    a.ck(ledger['terminal']['RL']['terminal_requests'] == counters['RL_rows'] == 6144, 'real_work_accounting')
    a.ck(ledger['terminal']['development']['terminal_requests'] == 36, 'real_work_accounting')
    a.ck(ledger['terminal']['RL']['terminal_cache_hits'] == counters['row_cache_hits'], 'real_work_accounting')
    a.ck(ledger['environment_actions'] == counters['RL_actions'], 'real_work_accounting')
    a.ck(ledger['supervised_label_actions'] == counters['SL_actions'], 'real_work_accounting')
    a.ck(ledger['teacher_label_reads'] == ledger['supervised_label_terminal_replays'] == 2048, 'real_work_accounting')
    a.ck(ledger['optimizer_attempts'] == ledger['optimizer_step_calls'] == ledger['optimizer_updates'] == 1024, 'real_work_accounting')
    a.ck(ledger['policy_trajectories'] == 6144 and ledger['forced_first_actions'] == 4096 and ledger['backward_calls'] == ledger['backward_completed'] == 4096, 'real_work_accounting')
    a.ck(ledger.get('native_teacher_attempts', 0) == 0, 'no_new_teacher_native')
    for key in ('task_scene_generation_attempts', 'task_scene_generations', 'proxy_feasibility_calls'):
        a.ck(ledger.get(key, 0) == len(generated_specs), 'data_resource_accounting')
    a.ck(ledger.get('navigation_attempts', 0) == ledger.get('navigation_completed', 0) == counters['event:navigation_start'] == counters['event:navigation_complete'], 'data_resource_accounting')
    a.ck(ledger['parent_requests'] == ledger['view_extractions'] == counters['event:view'] == 4132, 'data_resource_accounting')
    a.ck(final['no_S3_queries'] is True and final['no_online_rule_fallback'] is True, 'execution_scope')
    for update in (256, 512):
        dev = read(base / f'RL/development_{update:04}.json')
        a.ck(dev['update'] == update and len(dev['rows']) == 18, 'fixed_development')
        a.ck(all(x['feasible'] is True for x in dev['rows']), 'fixed_development')
        a.ck('no checkpoint selection' in dev['purpose'], 'fixed_development')
    result = dict(schema='rc3-s2-independent-seed-audit-v1', seed=seed,
        status='PASS_CLOSED_SEED' if not a.failures else 'FAIL_INDEPENDENT_CHECKS',
        scientific_stage_status='COMPLETE_FIXED_ENDPOINTS', checks_total=sum(a.counts.values()),
        check_groups=dict(a.counts), failures=a.failures, max_abs_errors=dict(a.max_errors),
        source_manifest_sha256=SOURCE, protocol_sha256=PROTOCOL,
        completion_sha256=a.record(admission / 'COMPLETE.json'), source_SL256_tensor_sha256=sourcehash,
        checkpoints_audited=checkpoint_count, parameter_deltas=parameter_deltas,
        observed_counts=dict(counters), shared_stage_wall_s=resource['elapsed_wall_s'],
        resource_ledger=ledger, audited_artifact_sha256=a.artifacts,
        independent_certificate_scope='Saved verdicts plus raw saved route/objective/polyline arithmetic and assignment coverage; no new geometric/IK/route evaluation.',
        RNG_scope='Independent SHA/PCG64 schedules and metadata/state validity; no policy inference or sampling replay from logits.',
        no_model_or_optimizer_instantiation=True, no_new_query_or_scene_generation=True,
        no_S3_or_original60_semantic_access=True, streaming_JSONL=True,
        measured_audit_compute_s=time.perf_counter() - started,
        completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    out = HERE / f'audit_seed_{seed}.json'
    with out.open('x', encoding='utf8', newline='\n') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({k: result[k] for k in ('seed', 'status', 'checks_total', 'failures', 'checkpoints_audited', 'measured_audit_compute_s', 'max_abs_errors')}, ensure_ascii=False))
    return int(bool(a.failures))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    raise SystemExit(run(p.parse_args().seed))
