"""Read-only mechanical audit of actual S1-B logs, checkpoints and lineage.

Creates no model, optimizer, environment, navigation solution or teacher query.
Run from the repository root with the existing research Python environment.
"""
from pathlib import Path
import hashlib
import json
import math
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from rc3_global_local_trainer.rng import derive_seed

HERE = Path(__file__).resolve().parent
RUN = ROOT/'rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01'
SEEDS = (11622026, 11622027, 11622028)
PH = '0c5bc7d7a9557edf2af76f51a9f418550f0b4ce1ec6a8b47ddea95c6256e78bb'
SH = '0f0ffd2d2dda2fefefcfcb5cb26dc5197b5d11bd1332b0a5546992409049bfe0'
AH = 'ee952e9a3b3d917b96aef803196d16e9807fe7ac527c1b7244210f04d50950e2'
PR = '9a45d6e7a6b7ccbc8d43b614646ecefe417ba10eaabfe2055c4e896f1e6aa7c4'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))

def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def state_hash(state):
    h = hashlib.sha256()
    for n, v in sorted(state.items()):
        h.update(n.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def write_new(p, value):
    with Path(p).open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')

def main():
    started = time.perf_counter(); checks = 0
    def check(condition, message):
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(message)
    source = ROOT/'rc3_navigation_compat_20261106/SOURCE_BINDING_UPDATED.json'
    check(digest(source) == SH, 'Approved source manifest changed')
    for name, h in read(source)['source_sha256'].items():
        check(digest(ROOT/name) == h, 'Source bytes changed: '+name)
    authdir = ROOT/'rc3_global_local_execution_v3/authorizations/20261106_S1_B'
    check(digest(authdir/'SOURCE_REVISION_APPROVAL.json') == PR, 'Approval receipt changed')
    check(digest(authdir/'S1_B_AUTHORIZATION.json') == AH, 'Stage authorization changed')
    check(read(authdir/'S1_B_AUTHORIZATION.json')['status'] == 'AUTHORIZED', 'Not authorized')
    original = read(ROOT/'rc3_navigation_compat_20261106/LEGACY_TEACHER_REPLAY_MANIFEST.json')
    check(digest(RUN/'teachers256/POOL_INDEX.json') == original['pool_index_sha256'], 'Teacher index changed')
    labels = [read(RUN/'teachers256'/f'label_{i:03}.json') for i in range(256)]
    oldmanifest = json.loads(__import__('subprocess').check_output(
        ['git', 'show', 'ffcfcabc4ccc0b071a368b5e6e49684be3de4d64:project_management/GIT_MANIFEST.json'], cwd=ROOT))
    frozen_files = []
    prefix = RUN.relative_to(ROOT).as_posix()+'/'
    for name, r in oldmanifest['files'].items():
        if name.startswith(prefix):
            check((ROOT/name).stat().st_size == r['bytes'] and digest(ROOT/name) == r['sha256'], 'S1-A frozen evidence changed: '+name)
            frozen_files.append(dict(file=name, **r))
    check(len(frozen_files) == 1482, 'Original S1-A byte-audit population differs')
    rows_out = []; artifacts = []; previous = None
    for seed in SEEDS:
        folder = RUN/f'seed_{seed}/S1'; admission = RUN/f'stage_admission/S1-B/{seed}'
        check(not (admission/'FAILED.json').exists(), 'Failed stage cannot be treated as complete')
        complete = read(admission/'COMPLETE.json')
        start = read(admission/'START.json')
        final = read(folder/'FINAL.json')
        check(complete['status'] == 'COMPLETE', 'Stage not complete')
        check(start['science_binding']['source_manifest_sha256'] == SH, 'Start source binding mismatch')
        check(start['science_binding']['approval_sha256'] == PR, 'Start approval binding mismatch')
        auth = read(admission/'AUTHORIZATION.json')
        check(digest(admission/'AUTHORIZATION.json') == complete['authorization_record_sha256'], 'Completion authorization record mismatch')
        check(read(admission/'AUTHORIZATION_SOURCE.json')['authorization_sha256'] == AH, 'Completion authorization source mismatch')
        check(auth['authorization_sha256'] == AH and digest(admission/'START.json') == complete['start_record_sha256'], 'Stage record linkage mismatch')
        for relative, h in complete['artifacts'].items():
            p = (RUN/relative).resolve()
            check(p.is_relative_to(RUN.resolve()) and digest(p) == h, 'Stage artifact closure mismatch: '+relative)
        logs = [json.loads(s) for s in (folder/'updates.jsonl').read_text().splitlines()]
        update_lines = (folder/'updates.jsonl').read_bytes().splitlines(keepends=True)
        event_lines = (folder/'events.jsonl').read_bytes().splitlines(keepends=True)
        check(len(logs) == 256, 'Incomplete update log')
        rng = np.random.Generator(np.random.PCG64(derive_seed(PH, 'S1_data', seed)))
        rng_states = {0: rng.bit_generator.state}
        seen = set(); anomaly = 0
        for i, r in enumerate(logs):
            check(r['update'] == i and r['cursor_after'] == i+1 and r['weight_version_after'] == i+1, 'Cursor mismatch')
            expected = rng.integers(0, 256, 4).tolist()
            check(r['teacher_indices'] == expected, 'Named data RNG mismatch')
            check(r['batch_instances'] == 4 and len(r['instances']) == 4 and r['optimizer_updated'], 'Not a real batch4 update')
            check(r['teacher_or_H_queries'] == 0, 'New teacher/H call')
            check(math.isfinite(r['loss']) and r['loss'] >= 0, 'Nonfinite/negative CE')
            gb, ga = r['gradient_norm_before_clip'], r['gradient_norm_after_clip']
            check(math.isfinite(gb) and gb > 0 and math.isfinite(ga) and 0 < ga <= 1.00001, 'Abnormal gradient')
            check(all(math.isfinite(x) for x in r['gradient_norms'].values()), 'Nonfinite per-tensor gradient')
            check(0 <= r['stage_elapsed_s'] <= 1800 and (i == 0 or r['stage_elapsed_s'] >= logs[i-1]['stage_elapsed_s']), 'Wall budget/monotonicity mismatch')
            for idx, inst in zip(expected, r['instances']):
                seen.add(idx)
                check(inst['actions'] == labels[idx]['actions'] and inst['problem'] == labels[idx]['view_identity'], 'Teacher action/identity mismatch')
                check(inst['teacher_or_H_queries'] == 0 and inst['decisions'] == len(inst['actions']), 'Invalid teacher replay accounting')
                check(len(inst['legal_counts']) == inst['decisions'] and min(inst['legal_counts']) > 0, 'Mask evidence absent')
            if (i+1) % 32 == 0:
                rng_states[i+1] = rng.bit_generator.state
        initial = endpoint = None; checkpoint_rows = []
        for cursor in [0, *range(32, 257, 32)]:
            path = folder/('untrained.pt' if cursor == 0 else f'update_{cursor:04}.pt')
            sidepath = path.with_suffix('.pt.json'); closepath = path.with_suffix('.pt.commit.json')
            side, close = read(sidepath), read(closepath)
            check(digest(path) == side['sha256'] == close['checkpoint_sha256'], 'Checkpoint file checksum mismatch')
            check(path.stat().st_size == side['bytes'] and digest(sidepath) == close['sidecar_sha256'], 'Checkpoint integrity sidecar mismatch')
            state = torch.load(path, map_location='cpu', weights_only=False)
            binding = state['config']['binding']
            check(binding == side['binding'] and binding['optimizer_seed'] == seed and binding['phase'] == 'S1', 'Checkpoint seed/phase mismatch')
            check(binding['protocol_sha256'] == PH and binding['source_manifest_sha256'] == SH and binding['approval_sha256'] == PR, 'Checkpoint scientific lineage mismatch')
            check(state['cursors'] == {'S1':cursor, 'RL':0, 'continued_SL':0}, 'Checkpoint cursor mismatch')
            check(state['weight_version'] == cursor and state['stopped'] is False, 'Checkpoint version/stopped mismatch')
            check(state['config']['batch_instances'] == 4 and state['config']['clip_global_norm'] == 1 and state['config']['max_optimizer_updates'] == 256, 'Training recipe mismatch')
            group = state['optimizer']['param_groups'][0]
            check(group['lr'] == 1e-4 and tuple(group['betas']) == (.9,.999) and group['eps'] == 1e-8 and group['weight_decay'] == 0, 'Adam recipe mismatch')
            for journal, lines, count in [('updates.jsonl',update_lines,cursor),('events.jsonl',event_lines,cursor*4)]:
                expected_journal_hash = hashlib.sha256(b''.join(lines[:count])).hexdigest() if count else None
                check(state['extra']['journal_sha256'][journal] == expected_journal_hash == close['journal_sha256'][journal], 'Checkpoint/log boundary prefix mismatch')
            check(state_hash(state['model']) == side['model_tensor_sha256'] == state['model_tensor_sha256'], 'Tensor checksum mismatch')
            check(all(bool(torch.isfinite(t).all()) for t in state['model'].values()), 'Nonfinite checkpoint model')
            check(state['streams']['optimizer_seed'] == seed and state['streams']['protocol_sha256'] == PH, 'Checkpoint RNG seed mismatch')
            stream = state['streams']['numpy'].get('S1_data')
            check((stream is None if cursor == 0 else stream == rng_states[cursor]), 'Checkpoint RNG cursor mismatch')
            optim = state['optimizer']
            steps = [float(v['step']) for v in optim['state'].values()]
            check((not steps if cursor == 0 else len(steps) == len(state['model']) and all(v == cursor for v in steps)), 'Adam step counters mismatch')
            check(all(bool(torch.isfinite(v).all()) for entry in optim['state'].values() for v in entry.values() if isinstance(v, torch.Tensor)), 'Nonfinite Adam state')
            check('torch' in state['global_rng'] and 'python' in state['global_rng'] and 'numpy' in state['global_rng'], 'Global RNG missing')
            if cursor:
                check(side['model_tensor_sha256'] == logs[cursor-1]['model_tensor_sha256'], 'Log/checkpoint model mismatch')
            checkpoint_rows.append(dict(cursor=cursor, file=path.relative_to(ROOT).as_posix(), sha256=side['sha256'], tensor_sha256=side['model_tensor_sha256']))
            for p in (path, sidepath, closepath):
                artifacts.append(dict(file=p.relative_to(ROOT).as_posix(), sha256=digest(p), bytes=p.stat().st_size))
            if cursor == 0: initial = state
            if cursor == 256: endpoint = state
        check(final['status'] == 'COMPLETE_FIXED_ENDPOINT' and final['optimizer_updates'] == 256, 'Final endpoint not complete')
        ledger = final['ledger']
        for key in ['native_teacher_attempts','task_scene_generation_attempts','task_scene_generations','navigation_attempts','navigation_completed','proxy_feasibility_calls']:
            check(ledger.get(key, 0) == 0, 'Unexpected generation/query: '+key)
        check(ledger['optimizer_updates'] == 256 and ledger['model_initializations'] == 1, 'Update/init accounting mismatch')
        check(all(all(x == 0 for x in role.values()) for role in ledger['terminal'].values()), 'Terminal call/cache counters not zero')
        resources = complete['cumulative_stage_resources']
        wall = next(iter(resources.values()))['elapsed_wall_s']
        check(wall < 1800, 'Completed stage exceeded wall cap')
        if previous is not None:
            # The production START snapshot records the exact preceding seed completion.
            expected_hash = digest(RUN/f'stage_admission/S1-B/{previous}/COMPLETE.json')
            check(complete['predecessors'].get(f'S1-B/{previous}') == expected_hash, 'Previous seed completion is absent from admission chain')
        deltas = [(endpoint['model'][n].double()-v.double()).flatten() for n,v in initial['model'].items()]
        before = torch.cat([v.double().flatten() for v in initial['model'].values()])
        delta = torch.cat(deltas)
        row = dict(seed=seed, updates=256, scientific_wall_s=wall, unique_teachers_sampled=len(seen),
            first_batch_loss=logs[0]['loss'], final_batch_loss=logs[-1]['loss'],
            first32_mean_loss=float(np.mean([r['loss'] for r in logs[:32]])), last32_mean_loss=float(np.mean([r['loss'] for r in logs[-32:]])),
            gradient_before_min=min(r['gradient_norm_before_clip'] for r in logs), gradient_before_max=max(r['gradient_norm_before_clip'] for r in logs),
            gradient_after_max=max(r['gradient_norm_after_clip'] for r in logs), clipped_updates=sum(r['gradient_norm_before_clip']>1 for r in logs),
            nonfinite_or_invalid_updates=anomaly, anomaly_fraction=0.,
            parameter_count=before.numel(), changed_parameters=int((delta != 0).sum()), changed_tensors=sum(bool((d != 0).any()) for d in deltas),
            parameter_L2_change=float(delta.norm()), relative_L2_change=float(delta.norm()/before.norm()), parameter_max_abs_change=float(delta.abs().max()),
            initialization_tensor_sha256=initial['model_tensor_sha256'], SL256_tensor_sha256=endpoint['model_tensor_sha256'],
            data_stream_seed=derive_seed(PH,'S1_data',seed), data_stream_state_sha256=hashlib.sha256(json.dumps(endpoint['streams']['numpy'],sort_keys=True).encode()).hexdigest(),
            checkpoints=checkpoint_rows, ledger=ledger, completion_sha256=digest(admission/'COMPLETE.json'))
        rows_out.append(row); previous = seed
        row['initialization_torch_rng_sha256'] = hashlib.sha256(initial['global_rng']['torch'].numpy().tobytes()).hexdigest()
    for key in ['initialization_tensor_sha256','SL256_tensor_sha256','data_stream_seed','data_stream_state_sha256']:
        check(len({r[key] for r in rows_out}) == 3, 'Seeds are not distinct: '+key)
    check(len({r['initialization_torch_rng_sha256'] for r in rows_out}) == 3, 'Initialization Torch RNG snapshots are not distinct')
    result = dict(schema='rc3-s1b-mechanical-audit-v1', status='PASS', checks=checks, seeds=rows_out,
        total_scientific_optimizer_updates=768, scientific_policy_initializations=3,
        original_S1A_files_byte_checked=len(frozen_files), original_S1A_bytes=sum(r['bytes'] for r in frozen_files),
        original_S1A_all_bytes_unchanged=True, checkpoint_artifacts=artifacts,
        diagnostic_model_initializations=0, diagnostic_optimizer_updates=0, new_teacher_queries=0,
        audit_wall_s=time.perf_counter()-started, analysis_script_sha256=digest(__file__))
    write_new(HERE/'MECHANICAL_AUDIT.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['seeds','checkpoint_artifacts']},ensure_ascii=False))

if __name__ == '__main__':
    main()
