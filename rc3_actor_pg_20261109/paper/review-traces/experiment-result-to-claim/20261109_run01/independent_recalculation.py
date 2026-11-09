"""Independent saved-evidence audit. No model, optimizer, rollout or planner is run.

Only stdlib, NumPy and torch deserialization/tensor arithmetic are used. The
scientific source modules are read as bytes/text, never imported. All writes
are confined to this review trace. Frozen historical evidence is byte-hashed
only; in particular no old held-out task/result JSON is semantically opened.
"""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

TRACE = Path(__file__).resolve().parent
HOME = TRACE.parents[3]
ROOT = HOME.parent
START = time.perf_counter()
UTC = datetime.now(timezone.utc).isoformat()
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_grad_enabled(False)


def denied(*args, **kwargs):
    raise AssertionError('Scientific execution is forbidden in independent review')


# Fail closed if later edits accidentally attempt scientific computation.
torch.nn.Module.__call__ = denied
torch.optim.Optimizer.__init__ = denied
torch.autograd.backward = denied
torch.Tensor.backward = denied

OUT = dict(schema='rc3-independent-readonly-result-audit-v1', started_utc=UTC,
           reviewer='independent delegated paper architect/reviewer',
           script_scope='Saved-data/hash/tensor/statistics only; no scientific modules imported',
           checks={}, warnings=[], numeric_tolerances=dict(
               exact_parameters='name/shape/dtype/torch.equal/hash exact',
               saved_return_J_absolute=1e-7, scalar_loss_relative=1e-10,
               float32_logprob_sum_absolute=0.002,
               gradient_norm_relative=1e-6,
               clip_vector_absolute=2e-7, clip_vector_relative=2e-6,
               Adam_saved_delta_absolute=2e-7, Adam_saved_delta_relative=2e-5))


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf8')


def ident(v):
    return hashlib.sha256(canonical(v)).hexdigest()


def check(ok, msg):
    if not ok:
        raise AssertionError(msg)


def close(a, b, abs_tol=1e-7, rel_tol=1e-10):
    return math.isclose(float(a), float(b), abs_tol=abs_tol, rel_tol=rel_tol)


def tensorhash(d):
    h = hashlib.sha256()
    for name, value in sorted(d.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def exact(a, b):
    return set(a) == set(b) and all(a[k].shape == b[k].shape and a[k].dtype == b[k].dtype
                                  and torch.equal(a[k], b[k]) for k in a)


def write_json(name, value):
    p = TRACE / name
    check(p.parent == TRACE, 'Output outside review trace')
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf8')


def chain(folder):
    events = []
    tail = '0' * 64
    with (folder/'events.jsonl').open(encoding='utf8') as f:
        for line in f:
            d = json.loads(line)
            check(d['sequence'] == len(events) and d['previous'] == tail, 'Event sequence/parent mismatch')
            payload = {k: v for k, v in d.items() if k != 'sha256'}
            check(d['sha256'] == ident(payload), 'Event self hash mismatch')
            tail = d['sha256']
            events.append(d)
    h = read(folder/'HEAD.json')
    check(h['sha256'] == tail and h['sequence'] == len(events), 'Anchor mismatch')
    check(not (folder/'HEAD.pending').exists(), 'Unclosed event anchor')
    errors = [e for e in events if any(x in e['event'].lower() for x in ('error', 'failure', 'failed'))]
    check(not errors, 'Failure event exists')
    return events, tail


def completed(folder):
    events, tail = chain(folder)
    starts, rows, seen = {}, [], set()
    for e in events:
        if e['event'] == 'request_start':
            q = e['request']
            check(q['id'] not in starts, 'Duplicate predebit')
            check(q['id'] == ident({k: v for k, v in q.items() if k != 'id'}), 'Request ID mismatch')
            starts[q['id']] = q
        if e['event'] == 'request_complete':
            check(e['id'] in starts and e['id'] not in seen, 'Unbilled/duplicate completion')
            p = ROOT / e['artifact_path']
            check(sha(p) == e['artifact_sha256'], 'Result file hash mismatch')
            r = read(p)
            check(r['request'] == starts[e['id']] and r['valid'], 'Result request/validity differs')
            v = r['plan']['validation']
            check(v['valid_model_plan'] and v['all_tasks_served'] and v['endpoint_constraints_satisfied']
                  and v['effective_same_arm_F_and_A'] and v['collision_count'] == 0 and not v['errors'],
                  'Invalid saved plan certificate')
            check(not v['physical_joint_trajectory_certified'], 'Unexpected physical IK claim')
            check(not r['plan']['unserved_task_ids'], 'Unserved tasks')
            check(r['raw_route_sha256'] == ident(r['raw_route']) and r['route_sha256'] == ident(r['route']), 'Route hash differs')
            check(all(math.isfinite(x) and x >= 0 for x in r['metrics'].values()), 'Nonfinite metrics')
            check(close(r['metrics']['J'], r['metrics']['L'] + .75*r['metrics']['K']), 'Saved objective arithmetic')
            check(close(r['metrics']['raw_J'], r['metrics']['raw_L'] + .75*r['metrics']['raw_K']), 'Raw objective arithmetic')
            check(close(r['metrics']['J'], r['plan']['objective_length_plus_stops_m']), 'Plan J differs')
            check(close(r['metrics']['L'], r['plan']['route_length_m']), 'Plan L differs')
            check(close(r['H_gain_J'], r['metrics']['raw_J']-r['metrics']['J']), 'H gain differs')
            check(e['counts']['artifact_write_completed'] == 1 and e['complete_time_s'] > 0, 'Durable request write/time absent')
            check(e['complete_time_s'] >= max(e['seconds'].values(), default=0), 'Complete time excludes a nested component')
            seen.add(e['id'])
            rows.append((r, e))
    check(set(starts) == seen == {p.stem for p in (folder/'results').glob('*.json')}, 'Incomplete/orphan request set')
    check(read(folder/'COMPLETE.json')['event_tail'] == tail, 'Completion tail differs')
    return rows, events


def main():
    protocol = read(HOME/'PROTOCOL.json')
    binding = read(HOME/'SOURCE_BINDING.json')
    authorization = read(HOME/'AUTHORIZATION.json')
    registry = read(HOME/'INPUT_REGISTRY.json')
    check(sha(HOME/'SOURCE_BINDING.json') == '57b0d0d2236b4b1f942a393ce2fc41e6408f4e7ec899df038bdfc4b19a65eee6', 'Source binding differs from assigned identity')
    for name, key in [('PROTOCOL.json', 'protocol_sha256'), ('INPUT_REGISTRY.json', 'input_registry_sha256'),
                      ('DEPLOYMENT_GRID.json', 'deployment_grid_sha256'), ('AUTHORIZATION.json', 'authorization_sha256')]:
        check(sha(HOME/name) == binding[key], 'Registration binding: '+name)
    check(authorization['protocol_sha256'] == sha(HOME/'PROTOCOL.json'), 'Authorization protocol differs')
    check(authorization['phases'] == ['DEPLOYMENT', 'A128', 'B128', 'DEV54'] and not authorization['old_budget_reused'], 'Authorization scope')
    check(ident(registry['training_batches']) == protocol['mechanism']['common_schedule_sha256'], 'Schedule identity')
    for rel, expected in binding['source_sha256'].items():
        check(sha(ROOT/rel) == expected, 'Source bytes changed: '+rel)
    for rel, d in registry['source_files'].items():
        check((ROOT/rel).stat().st_size == d['bytes'] and sha(ROOT/rel) == d['sha256'], 'Saved input bytes changed: '+rel)
    frozen = read(HOME/'FROZEN_PRIOR_EVIDENCE.json')
    hashed_bytes = 0
    for rel, d in frozen['files'].items():
        check((ROOT/rel).stat().st_size == d['bytes'] and sha(ROOT/rel) == d['sha256'], 'Frozen bytes changed: '+rel)
        hashed_bytes += d['bytes']
    OUT['checks']['bindings'] = dict(status='PASS', bound_source_files=len(binding['source_sha256']), input_source_files=len(registry['source_files']),
                                    frozen_prior_files=len(frozen['files']), frozen_bytes_hashed_only=hashed_bytes,
                                    old_heldout_semantic_reads=0)
    print('PASS bindings/frozen hash', flush=True)

    # Actor-only deserialization; never construct GlobalLocalPolicy.
    actor_index = read(HOME/'actors/ACTOR_INDEX.json')
    source_index = read(ROOT/'rc3_s3_evaluator/S3_MODEL_INDEX.json')
    actor_reports = []
    original_source_files = set()
    for seed, roles in actor_index['models'].items():
        for role, e in roles.items():
            src = source_index['models'][seed][role]
            for pk, hk in [('path', 'checkpoint_sha256'), ('sidecar_path', 'sidecar_sha256'), ('commit_path', 'commit_sha256')]:
                check(sha(ROOT/src[pk]) == src[hk], 'Original model source modified')
                original_source_files.add(src[pk])
            check(sha(ROOT/e['path']) == e['checkpoint_sha256'], 'Actor file SHA')
            actor = torch.load(ROOT/e['path'], map_location='cpu', weights_only=True)
            original = torch.load(ROOT/src['path'], map_location='cpu', weights_only=False)
            check(set(actor) == {'schema', 'architecture', 'model', 'model_tensor_sha256', 'provenance'}, 'Actor-only schema')
            check(actor['architecture'] == actor_index['architecture'] and actor['provenance'] == e['provenance'], 'Actor architecture/provenance')
            check(actor['provenance']['training_checkpoint_path'] == src['path'] and actor['provenance']['training_checkpoint_sha256'] == src['checkpoint_sha256'], 'Actor source mapping')
            check(exact(actor['model'], original['model']), 'Actor parameter name/shape/dtype/value mismatch')
            check(tensorhash(actor['model']) == tensorhash(original['model']) == e['model_tensor_sha256'] == src['model_tensor_sha256'], 'Actor tensor hash mismatch')
            for d in e['parameters']:
                t = actor['model'][d['name']]
                check(list(t.shape) == d['shape'] and str(t.dtype) == d['dtype'] and hashlib.sha256(t.numpy().tobytes()).hexdigest() == d['value_sha256'], 'Per-parameter receipt mismatch')
                check(torch.isfinite(t).all().item(), 'Nonfinite actor')
            actor_reports.append(dict(seed=int(seed), role=role, parameters=len(actor['model']), exact=True,
                                      checkpoint_bytes=src['checkpoint_bytes'], actor_bytes=e['checkpoint_bytes'], tensor_sha256=e['model_tensor_sha256']))
            del original, actor
    check(len(actor_reports) == 12 and len(original_source_files) == 36, 'Actor count')
    OUT['checks']['actors'] = dict(status='PASS', exact_models=12, preserved_source_files=36, actors=actor_reports)

    methods = protocol['deployment']['methods']
    grid = read(HOME/'DEPLOYMENT_GRID.json')['requests']
    expected = []
    rng = np.random.Generator(np.random.PCG64(11642026))
    for mode in ['D0', 'D1', 'D2']:
        for case in registry['development_cases'][:6]:
            for kind in ['cold', 'hot']:
                for repeat in range(3):
                    for method in rng.permutation(methods).tolist():
                        q = dict(seed=11622026, mode=mode, case_id=case['id'], method=method, kind=kind, repeat=repeat)
                        expected.append(dict(q, id=ident(q)))
    check(grid == expected, 'Independent grid/order regeneration')
    groups = collections.defaultdict(list)
    components = collections.defaultdict(list)
    counts = collections.Counter()
    stable, pairs, startups, mode_counts = {}, {}, {}, {}
    all_request_rows = []
    for mode in ['D0', 'D1', 'D2']:
        folder = HOME/'runs/DEPLOYMENT'/mode
        rows, events = completed(folder)
        check([r['request'] for r, e in rows] == [q for q in grid if q['mode'] == mode], 'Actual grid/order differs')
        startups[mode] = read(folder/'STARTUP.json')
        c = read(folder/'COMPLETE.json')
        mode_counts[mode] = dict(requests=len(rows), model_loads=c['model_loads'], resident_hits=c['resident_hits'],
                                 startup_s=startups[mode]['process_to_ready_s'])
        check(c['model_loads'] == (4 if mode == 'D2' else 180) and c['resident_hits'] == (180 if mode == 'D2' else 0), 'Lifecycle loading counts')
        check(startups[mode]['resident_models'] == (4 if mode == 'D2' else 0), 'Resident model count')
        check(len([e for e in events if e['event'] == 'hot_prime']) == 126, 'Hot priming count')
        for r, e in rows:
            q = r['request']
            k = (q['case_id'], q['method'])
            value = (r['raw_route'], r['route'], r['metrics'], r['source'], r['plan']['assignments'])
            check(k not in stable or stable[k] == value, 'Lifecycle/cache/repeat changed scientific output')
            stable[k] = value
            if q['method'] in ['RL_raw', 'RL_plus_H']:
                pairs.setdefault((q['case_id'], mode, q['kind'], q['repeat']), {})[q['method']] = r['raw_route']
            neural = q['method'] not in methods[:2]
            check(e['counts'].get('fresh_encodings', 0) == int(neural), 'Encoding cached across requests')
            check(e['counts'].get('common_H_invocations', 0) == int(q['method'] in ['untrained','frozen_supervised','continued_SL','RL_plus_H']), 'H usage differs')
            check(e['counts'].get('navigation_only_cache_hits', 0) == int(q['kind'] == 'hot'), 'Navigation cache key behavior')
            check(e['counts'].get('navigation_solves', 0) == int(q['kind'] == 'cold'), 'Cold navigation solves')
            counts.update(e['counts'])
            g = (q['method'], mode, q['kind'])
            groups[g].append(e['complete_time_s'])
            for comp, seconds in e['seconds'].items():
                components[g+(comp,)].append(seconds)
            all_request_rows.append((r, e))
    check(len(all_request_rows) == 756 and len(stable) == 42 and len(pairs) == 108, 'Deployment cardinality')
    check(all(set(v) == {'RL_raw', 'RL_plus_H'} and v['RL_raw'] == v['RL_plus_H'] for v in pairs.values()), 'Raw/H pairing')
    check(counts['fresh_encodings'] == 540 and counts['common_H_invocations'] == 432, 'Deployment counters')
    timing = []
    for (method, mode, nav), values in sorted(groups.items()):
        check(len(values) == 18, 'Timing sample count')
        timing.append(dict(method=method, mode=mode, navigation=nav, n=len(values), median_s=float(np.median(values)),
                           p95_s=float(np.quantile(values, .95)), sum_complete_s=float(np.sum(values))))
    published_timing = list(csv.DictReader((HOME/'results/DEPLOYMENT_TIMING.csv').open(encoding='utf8')))
    by_timing = {(d['method'], d['mode'], d['navigation']): d for d in published_timing}
    for d in timing:
        p = by_timing[d['method'], d['mode'], d['navigation']]
        check(close(d['median_s'], p['median_s']) and close(d['p95_s'], p['p95_s']), 'Published timing mismatch')
    comp_summary = [dict(method=k[0], mode=k[1], navigation=k[2], component=k[3], n=len(v), median_s=float(np.median(v)), p95_s=float(np.quantile(v,.95)))
                    for k, v in sorted(components.items())]
    OUT['checks']['deployment'] = dict(status='PASS', requests=756, grid_and_order_exact=True, exact_case_method_groups=42, raw_H_pairs=108,
                                       fresh_encodings=540, H_invocations=432, modes=mode_counts, counts=dict(counts), timings=timing,
                                       component_timings=comp_summary, startup=startups,
                                       D2_static_cache_key='(seed, actor checkpoint SHA256); resident dictionary stores only model parameters/modules',
                                       dynamic_encoding_scope='greedy creates local encoded variable per request; no catalog/NavigationPool encoding cache')
    write_json('deployment_recomputed.json', OUT['checks']['deployment'])
    print('PASS 12 actors / 756 deployment requests / timing', flush=True)

    # Full saved-gradient and logged-likelihood audit; no gradient recomputation.
    train_stats, same_inputs = {}, []
    for arm in ['A128', 'B128']:
        folder = HOME/'runs'/arm
        events, tail = chain(folder)
        initial = torch.load(folder/'checkpoints/update_0000.pt', map_location='cpu', weights_only=False)
        check(not initial['optimizer']['state'] and initial['cursors']['RL'] == 0 and initial['weight_version'] == 0, 'Not fresh Adam/common zero cursor')
        sl = torch.load(ROOT/actor_index['models']['11622026']['SL256']['path'], map_location='cpu', weights_only=True)
        check(exact(initial['model'], sl['model']), 'Common SL starting parameters differ')
        check(initial['model_tensor_sha256'] == tensorhash(initial['model']) == read(folder/'WEIGHT_BRANCH.json')['initial_tensor_sha256'], 'Common tensor identity')
        config = initial['config']
        check(config['mixture_weights'] == ([1., .25, .25] if arm == 'A128' else [1.,1.,1.]), 'Arm objective weights')
        check(config['gamma'] == 1 and config['entropy_weight'] == 0 and not config['advantage_standardization'] and config['batch_instances'] == 4, 'Unregistered objective/config')
        group = initial['optimizer']['param_groups'][0]
        check(group['lr'] == 1e-4 and tuple(group['betas']) == (.9,.999) and group['eps'] == 1e-8 and group['weight_decay'] == 0, 'Adam hyperparameters')
        names = list(initial['model'])
        old_arrays = {k:v.numpy().copy() for k,v in initial['model'].items()}
        moments = {k:np.zeros_like(v) for k,v in old_arrays.items()}
        variances = {k:np.zeros_like(v) for k,v in old_arrays.items()}
        stats = collections.Counter()
        max_errors = collections.defaultdict(float)
        grad_rows = []
        inputs = []
        term_starts = [e for e in events if e['event'] == 'terminal_request_predebit']
        term_ends = [e for e in events if e['event'] == 'terminal_request_complete']
        update_starts = [e for e in events if e['event'] == 'update_start']
        update_ends = [e for e in events if e['event'] == 'update_complete']
        check(len(term_starts) == len(term_ends) == 1536 and len(update_starts) == len(update_ends) == 128, 'Training predebits/completions')
        check([e['request'] for e in term_starts] == list(range(1,1537)) == [e['request'] for e in term_ends], 'Terminal debit monotonicity')
        for u in range(1,129):
            path = folder/'updates'/f'update_{u:04}.json'
            d = read(path)
            check(d['cursor_after'] == u and d['weight_version_after'] == u and d['update'] == u-1 and d['optimizer_updated'], 'Update/cursor mismatch')
            check(d['specs'] == registry['training_batches'][u-1] and len(d['instances']) == 4, 'Training input schedule differs')
            check(sha(path) == update_ends[u-1]['update_artifact_sha256'], 'Update receipt hash')
            check(update_starts[u-1]['weight_tensor_before'] == tensorhash({k:torch.from_numpy(v) for k,v in old_arrays.items()}), 'Pre-update lineage mismatch')
            inputs.append((d['specs'], d['view_identities']))
            instance_losses = []
            for slot, ins in enumerate(d['instances']):
                check(ins['problem'] == d['view_identities'][slot] and ins['weight_version'] == u-1 and ins['update'] == u-1 and ins['slot'] == slot, 'Stale trajectory/update input')
                check(len(ins['rows']) == 3, 'Trajectory count')
                weights = [1., .25, .25] if arm == 'A128' else [1.,1.,1.]
                check(ins['mixture_weights'] == weights, 'Mixture weights log mismatch')
                returns = np.array([r['return_'] for r in ins['rows']], dtype=np.float64)
                loo = (returns.sum()-returns)/2
                adv = returns-loo
                check(np.allclose(loo, ins['baseline'], rtol=0, atol=1e-10) and np.allclose(adv, ins['advantages'], rtol=0, atol=1e-10), 'Detached LOO numeric mismatch')
                loss = 0.
                seeds = set()
                for ri, row in enumerate(ins['rows']):
                    stats['trajectories'] += 1
                    stats['actions'] += len(row['actions'])
                    role = (['primary','forced_low','forced_high'] if arm == 'A128' else ['policy_0','policy_1','policy_2'])[ri]
                    check(row['role'] == role, 'Trajectory role differs')
                    payload = json.dumps([protocol['original_v3_protocol_sha256'], 'RL', 11622026, u-1, slot, role], ensure_ascii=False, separators=(',', ':')).encode()
                    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little') % (2**63)
                    check(row['sampling_seed'] == seed and seed not in seeds, 'Sampling stream derivation/independence')
                    seeds.add(seed)
                    check(close(sum(row['rewards']), row['return_']) and close(row['return_'], -row['J']), 'Reward sum/terminal cost')
                    check(close(row['J'], row['L']+.75*row['K']) and close(row['raw_J'], row['raw_L']+.75*row['raw_K']), 'Training terminal arithmetic')
                    check(len(row['actions']) == len(row['masks_local_nodes']) == len(row['legal_counts']) == len(row['logprobs']), 'Action/mask/logprob cardinality')
                    check(row['raw_route'] == [0,*row['actions'],0], 'Training raw route trace')
                    for i,(a, mask, count, lp) in enumerate(zip(row['actions'], row['masks_local_nodes'], row['legal_counts'], row['logprobs'])):
                        check(a in mask and len(set(mask)) == len(mask) == count and count > 0, 'Illegal logged action/mask')
                        check(lp is None or math.isfinite(lp) and lp <= 1e-6, 'Invalid logprob')
                        if lp is not None and count == 1:check(abs(lp) < 1e-7, 'Single legal action has nonzero logprob')
                    if arm == 'B128' or ri == 0:
                        check(row['forced_first'] is None and all(lp is not None for lp in row['logprobs']), 'Policy full likelihood omits first action')
                        stats['full_policy_first_logprobs'] += 1
                    else:
                        check(row['forced_first'] == row['actions'][0] == (row['masks_local_nodes'][0][0] if ri == 1 else row['masks_local_nodes'][0][-1]), 'Forced start differs')
                        check(row['logprobs'][0] is None and all(lp is not None for lp in row['logprobs'][1:]), 'Forced likelihood differs')
                        stats['forced_first_logprobs_omitted'] += 1
                    # PyTorch logged a sequential float32 accumulation, so compare in float32 order.
                    running = np.float32(0.)
                    for lp in row['logprobs']:
                        if lp is not None:running = np.float32(running+np.float32(lp))
                    err = abs(float(running)-row['logprob_sum'])
                    max_errors['logprob_sum_abs'] = max(max_errors['logprob_sum_abs'], err)
                    check(err <= .002, 'Logged full logprob sum mismatch')
                    loss += -weights[ri]*adv[ri]*row['logprob_sum']/sum(weights)
                    request = (u-1)*12 + slot*3 + ri + 1
                    check(row['terminal_request'] == request, 'Training terminal request identity')
                    start, end = term_starts[request-1], term_ends[request-1]
                    check(start['problem'] == end['problem'] == ins['problem'] and start['raw_route'] == end['raw_route'] == row['raw_route'], 'Terminal debit source mismatch')
                    check(all(close(row[k], end[k]) for k in ['J','L','K','raw_J','raw_L','raw_K']), 'Terminal ledger metrics mismatch')
                    tp = read(folder/'terminal_plans'/f'RL_{request:06}.json')
                    check(tp['problem']['identity'] == ins['problem'] and tp['result']['raw_route'] == row['raw_route'] and tp['result']['route'] == row['final_route'], 'Saved complete terminal route mismatch')
                    check(all(close(row[k], tp['result'][k]) for k in ['J','L','K','raw_J','raw_L','raw_K']), 'Saved terminal objective mismatch')
                    v = tp['result']['plan']['validation']
                    check(v['valid_model_plan'] and v['all_tasks_served'] and v['endpoint_constraints_satisfied'] and not v['errors'] and v['collision_count'] == 0, 'Saved training terminal plan invalid')
                    check(not row['terminal_cache_hit'] and not tp['result']['cache_hit'], 'Unexpected terminal cache hit')
                max_errors['instance_loss_abs'] = max(max_errors['instance_loss_abs'], abs(loss-ins['loss']))
                check(close(loss, ins['loss'], abs_tol=1e-6, rel_tol=1e-10), 'LOO weighted instance objective mismatch')
                instance_losses.append(ins['loss'])
            check(close(sum(instance_losses)/4, d['loss'], abs_tol=1e-6, rel_tol=1e-10), 'Batch mean loss mismatch')
            gradfile = d['full_gradient_and_weight_artifact']
            gp = ROOT/gradfile['path']
            check(sha(gp) == gradfile['sha256'], 'Full gradient/weight artifact SHA')
            g = torch.load(gp, map_location='cpu', weights_only=True)
            check(g['update'] == u and tensorhash(g['model']) == d['model_tensor_sha256'] == g['model_tensor_sha256'], 'Resulting model tensor mismatch')
            check(set(g['gradient_before_clip']) == set(g['gradient_after_clip']) == set(g['model']) == set(names), 'Gradient parameter coverage')
            before_norms = {k:float(t.norm()) if t is not None else 0. for k,t in g['gradient_before_clip'].items()}
            after_norms = {k:float(t.norm()) if t is not None else 0. for k,t in g['gradient_after_clip'].items()}
            before = math.sqrt(sum(v*v for v in before_norms.values()))
            after = math.sqrt(sum(v*v for v in after_norms.values()))
            check(close(before,d['gradient_norm_before_clip'],rel_tol=1e-6) and close(after,d['gradient_norm_after_clip'],rel_tol=1e-6), 'Gradient norms differ')
            check(all(close(before_norms[k],d['gradient_norms'][k],rel_tol=1e-6) for k in names), 'Per-parameter gradient norm mismatch')
            check(before > 0 and after <= 1.000001 and all(torch.isfinite(t).all() for t in g['model'].values()), 'Invalid/zero/nonfinite gradient or model')
            pre_values = [t for t in g['gradient_before_clip'].values() if t is not None]
            torch_norm = float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(t,2) for t in pre_values]),2))
            coefficient = min(1., 1./(torch_norm+1e-6))
            delta_max = 0.
            for name in names:
                pre, post = g['gradient_before_clip'][name], g['gradient_after_clip'][name]
                check(pre is not None and post is not None and pre.shape == post.shape == g['model'][name].shape, 'Missing/shape gradient')
                check(torch.isfinite(pre).all() and torch.isfinite(post).all(), 'Nonfinite saved gradient')
                expected_post = pre * np.float32(coefficient)
                check(torch.allclose(post, expected_post, rtol=2e-6, atol=2e-7), 'Global clip vector mismatch')
                ga = post.numpy()
                # Pure array identities from recorded g; does not instantiate an optimizer or change any model.
                moments[name] = np.float32(.9)*moments[name] + np.float32(.1)*ga
                variances[name] = np.float32(.999)*variances[name] + np.float32(.001)*ga*ga
                denom = np.sqrt(variances[name])/math.sqrt(1.-.999**u)+np.float32(1e-8)
                expected_delta = (np.float32(1e-4/(1.-.9**u))*moments[name])/denom
                observed_delta = old_arrays[name]-g['model'][name].numpy()
                delta_error = float(np.max(np.abs(observed_delta-expected_delta)))
                delta_max = max(delta_max,delta_error)
                check(np.allclose(observed_delta,expected_delta,rtol=2e-5,atol=2e-7), 'Saved Adam model delta inconsistent with recorded gradient')
                old_arrays[name] = g['model'][name].numpy().copy()
            max_errors['Adam_model_delta_abs'] = max(max_errors['Adam_model_delta_abs'],delta_max)
            stats['clipped_updates'] += before > 1
            stats['zero_signal_updates'] += bool(d['zero_signal'])
            grad_rows.append(dict(update=u,gradient_norm_before=before,gradient_norm_after=after,Adam_model_delta_max_abs=delta_max))
            if u % 32 == 0:
                cp = folder/'checkpoints'/f'update_{u:04}.pt'
                saved = torch.load(cp,map_location='cpu',weights_only=False)
                receipt,commit = read(cp.with_suffix('.pt.json')),read(cp.with_suffix('.pt.commit.json'))
                check(sha(cp) == receipt['sha256'] == commit['checkpoint_sha256'] and sha(cp.with_suffix('.pt.json')) == commit['sidecar_sha256'], 'Full checkpoint byte/sidecar/commit')
                check(exact(saved['model'],g['model']) and saved['cursors']['RL'] == u and saved['weight_version'] == u, 'Full checkpoint cursor/model')
                check(saved['config'] == initial['config'] and saved['terminal_state']['request_count'] == u*12, 'Full checkpoint protocol/query cursor')
                check(saved['terminal_state']['original_terminal']['cache_size'] == 128 and len(saved['terminal_state']['original_terminal']['cache']) <= 128, 'Persisted terminal cache bound')
                check(set(saved['global_rng']) == {'python','numpy','torch','cuda'} and isinstance(saved['global_rng']['torch'],torch.Tensor), 'Global RNG missing')
                streams = saved['streams']
                check(streams['protocol_sha256'] == protocol['original_v3_protocol_sha256'] and streams['optimizer_seed'] == 11622026 and len(streams['torch']) == u*12 and set(streams['torch']) == set(streams['torch_meta']), 'Named RNG checkpoint mismatch')
                for pid,name in zip(saved['optimizer']['param_groups'][0]['params'],names):
                    state = saved['optimizer']['state'][pid]
                    check(int(state['step']) == u, 'Adam step cursor mismatch')
                    m_err = float(np.max(np.abs(state['exp_avg'].numpy()-moments[name])))
                    v_err = float(np.max(np.abs(state['exp_avg_sq'].numpy()-variances[name])))
                    max_errors['Adam_first_moment_abs'] = max(max_errors['Adam_first_moment_abs'],m_err)
                    max_errors['Adam_second_moment_abs'] = max(max_errors['Adam_second_moment_abs'],v_err)
                    check(np.allclose(state['exp_avg'].numpy(),moments[name],atol=2e-7,rtol=2e-5), 'Adam first moment inconsistency')
                    check(np.allclose(state['exp_avg_sq'].numpy(),variances[name],atol=2e-7,rtol=2e-5), 'Adam second moment inconsistency')
        complete = read(folder/'COMPLETE.json')
        ledger = complete['ledger']
        check(complete['event_tail'] == tail and complete['updates'] == 128 and complete['terminal_requests'] == 1536, 'Training completion differs')
        check(ledger['optimizer_updates'] == ledger['optimizer_attempts'] == ledger['optimizer_step_calls'] == 128 and ledger['backward_calls'] == ledger['backward_completed'] == 512, 'Update/failure resource reconciliation')
        check(ledger['terminal']['RL']['terminal_requests'] == ledger['terminal']['RL']['terminal_completed'] == ledger['terminal']['RL']['full_H_calls'] == 1536 and ledger['terminal']['RL']['terminal_failed'] == ledger['terminal']['RL']['terminal_cache_hits'] == 0, 'Terminal resource/cache ledger')
        check(ledger['environment_actions'] == stats['actions'] and ledger['encode_calls'] == 1536, 'Action/encoding counts differ')
        check(len(list((folder/'terminal_plans').glob('*.json'))) == 1536 and len(list((folder/'updates').glob('*.json'))) == 128, 'Orphan/missing training evidence')
        train_stats[arm] = dict(status='PASS', optimizer_updates=128,terminal_requests=1536, fresh_Adam=True,
                               initial_tensor_sha256=initial['model_tensor_sha256'], final_tensor_sha256=complete['tensor_sha256'],
                               full_checkpoint_cursors=[0,32,64,96,128], counts=dict(stats), maximum_numeric_errors=dict(max_errors),
                               gradient_checks=grad_rows, stored_full_gradient_tensors_verified=128*25*2,
                               new_backward_or_optimizer_calls=0,
                               audit_limit='Stored gradients/likelihood arithmetic and clipped Adam deltas checked; gradients/logits not recomputed with model forward/backward')
        same_inputs.append(inputs)
        print('PASS',arm,'128 updates / 1536 trajectories / full gradient and checkpoint lineage',flush=True)
    check(same_inputs[0] == same_inputs[1] and len(same_inputs[0]) == 128, 'A/B matched input identities')
    full_batches=sum(all(s['width']==10 and s['height']==10 for s in batch) for batch in registry['training_batches'])
    check(full_batches == 64, 'Full/local 64/64 batch mix')
    OUT['checks']['training'] = dict(status='PASS',arms=train_stats,identical_instance_views=512,full_batches=64,local_batches=64,
                                    same_start_exact=True,detached_LOO='Explicit .detach in both bound sources; all saved numeric baselines and objectives independently recomputed')

    dev,events = completed(HOME/'runs/DEV54')
    check(len(dev)==54,'DEV request cardinality')
    cases = registry['development_cases']
    expected_dev=[(c['id'],a) for c in cases for a in ['SL256','A128','B128']]
    check([(r['case_id'],r['study_method']) for r,e in dev] == expected_dev,'DEV fixed model/case order')
    index={(r['case_id'],r['study_method']):r for r,e in dev}
    for c in cases:
        for arm in ['SL256','A128','B128']:
            r=index[c['id'],arm]
            check(r['physical_key']==c['physical_key'] and r['size']==c['size'] and r['endpoint']==c['endpoint'],'DEV physical case identity')
            check(r['source']['F_sha256']==c['F_sha256'] and r['source']['A_sha256']==c['A_sha256'] and r['source']['D_sha256']==c['D_sha256'],'DEV F/A/D source')
    parents=sorted({c['physical_key'] for c in cases})
    check(len(parents)==3 and all(sum(c['physical_key']==p for c in cases)==6 for p in parents),'Parent clusters')
    bootstrap_indices=np.random.Generator(np.random.PCG64(11632026)).integers(3,size=(20000,3))
    comparisons=[]
    parent_means=[]
    for metric in ['raw_J','J']:
        for control in ['A128','SL256']:
            gain=[]
            for p in parents:
                ids=sorted(c['id'] for c in cases if c['physical_key']==p)
                a=float(np.mean([index[c,'B128']['metrics'][metric] for c in ids]))
                b=float(np.mean([index[c,control]['metrics'][metric] for c in ids]))
                gain.append((b-a)/b)
                parent_means.append(dict(physical_key=p,metric=metric,control=control,B128_mean=a,control_mean=b,relative_gain=(b-a)/b,cases=ids))
            values=np.array(gain);means=values[bootstrap_indices].mean(1)
            comparisons.append(dict(actor='B128',control=control,metric=metric,mean=float(values.mean()),
                lower95=float(np.quantile(means,.025)),upper95=float(np.quantile(means,.975)),upper_one_sided95=float(np.quantile(means,.95)),parent_gains=gain,
                exploratory_positive=bool(values.mean()>0 and np.quantile(means,.025)>0)))
    pub=read(HOME/'results/MECHANISM_COMPARISONS.json')['comparisons']
    for a,b in zip(comparisons,pub):
        check(a['actor']==b['actor'] and a['control']==b['control'] and a['metric']==b['metric'],'Comparison identity')
        check(all(close(a[k],b[k],abs_tol=1e-13,rel_tol=1e-12) for k in ['mean','lower95','upper95','upper_one_sided95']) and np.allclose(a['parent_gains'],b['parent_gains'],atol=1e-13,rtol=1e-12),'Published mechanism statistics mismatch')
    # Also report A against SL to expose the "less degradation" alternative.
    A_vs_SL=[]
    for metric in ['raw_J','J']:
        vals=[]
        for p in parents:
            ids=[c['id'] for c in cases if c['physical_key']==p]
            a=float(np.mean([index[c,'A128']['metrics'][metric] for c in ids]));b=float(np.mean([index[c,'SL256']['metrics'][metric] for c in ids]))
            vals.append((b-a)/b)
        means=np.array(vals)[bootstrap_indices].mean(1)
        A_vs_SL.append(dict(metric=metric,mean=float(np.mean(vals)),lower95=float(np.quantile(means,.025)),upper95=float(np.quantile(means,.975)),parent_gains=vals))
    strata=[]
    for size in [[4,3],[6,6],[8,8]]:
        for ep in ['closed_depot','open_first_last_connected']:
            ids=[c['id'] for c in cases if c['size']==size and c['endpoint']==ep]
            for arm in ['SL256','A128','B128']:
                strata.append(dict(size=size,endpoint=ep,arm=arm,cases=len(ids),**{metric:float(np.mean([index[c,arm]['metrics'][metric] for c in ids])) for metric in ['J','L','K','raw_J','raw_L','raw_K']}))
    OUT['checks']['development']=dict(status='PASS',requests=54,cases=18,physical_parent_clusters=3,optimizer_seeds=1,
        bootstrap_replicates=20000,bootstrap_seed=11632026,parent_means=parent_means,comparisons=comparisons,A_vs_SL=A_vs_SL,
        strata=strata,candidate_gate='GO_CANDIDATE_ONLY' if all(c['exploratory_positive'] for c in comparisons if c['metric']=='J') else 'REVISE',
        uncertainty='Descriptive parent percentile intervals over only three layouts; neither 18 independent layouts nor three optimizer seeds; no simultaneous coverage')
    write_json('mechanism_recomputed.json', OUT['checks']['development'])

    resources={}
    for phase in ['DEPLOYMENT','A128','B128','DEV54']:
        f=read(HOME/'runs'/phase/'PHASE_FINAL.json');s=read(HOME/'runs'/phase/'STARTED.json')
        check(f['status']=='COMPLETE' and f['error'] is None and f['wall_s']<f['cap_s']==s['wall_cap_s'],'Phase failed/exceeded budget')
        check(s['source_binding_sha256']==sha(HOME/'SOURCE_BINDING.json') and s['protocol_sha256']==sha(HOME/'PROTOCOL.json'),'Stage bound source/protocol')
        resources[phase]=dict(wall_s=f['wall_s'],cap_s=f['cap_s'],status=f['status'])
    OUT['checks']['resources']=dict(status='PASS',phases=resources,deployment_requests=756,train_terminal_requests=3072,
        DEV_terminal_requests=54,mechanism_total=3126,optimizer_updates=256,terminal_failures=0,scientific_request_failures=0,
        old_budgets_reused=False,engineering_separate=True)
    OUT['paper_claim_audit']='unavailable' if not (ROOT/'paper/PAPER_CLAIM_AUDIT.json').exists() else 'available_not_part_of_supplied_paper'
    OUT['downstream_writing']='provisional — no supplied paper claim audit' if OUT['paper_claim_audit']=='unavailable' else 'provisional pending paper-specific review'
    OUT['status']='PASS_SAVED_EVIDENCE_AUDIT'
    OUT['audit_wall_s']=time.perf_counter()-START
    OUT['finished_utc']=datetime.now(timezone.utc).isoformat()
    OUT['new_scientific_calls']=dict(model_forward=0,rollout=0,optimizer_updates=0,gradient_backward=0,new_scenes=0,
        navigation_solve=0,native=0,H=0,teacher=0,old60_semantic_reads=0,old_S3_reruns=0)
    write_json('independent_numeric_audit.json',OUT)
    print(json.dumps(dict(status=OUT['status'],wall_s=OUT['audit_wall_s'],comparisons=comparisons,new_scientific_calls=0),ensure_ascii=False),flush=True)


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        OUT['status']='AUDIT_SCRIPT_STOPPED'
        OUT['error']=repr(exc)
        OUT['audit_wall_s']=time.perf_counter()-START
        write_json('independent_numeric_audit.json',OUT)
        raise
