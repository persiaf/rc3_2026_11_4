"""Explicitly authorized M2 runner; this module never auto-starts training.

Run only with --authorization PATH --stage init|paired.  An independently
approved authorization must bind the exact frozen protocol and source hash.
The current M0/M1 authorization is not accepted.  Stage output directories
are canonical, and closed journals/checkpoints are required for continuation.

Recoverable boundaries: an atomic STATE.json after every completed SL/RL
update, and after a whole endpoint evaluation.  A crash within an update or
evaluation leaves paid transactions; this runner refuses to replay them.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

from .common import ROOT, HOME, Ledger, read, put, sha, verify_binding, append

EXPECTED_LIMITS = {
    'init': dict(request_attempts=4096, optimizer_attempts=512, wall_s=14400),
    'paired': dict(request_attempts=75728, optimizer_attempts=4096, wall_s=115200),
}


def _reference(ref):
    path = (ROOT / ref['path']).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file() or sha(path) != ref['sha256']:
        raise ValueError('Missing or changed registered reference: ' + str(ref.get('path')))
    return path


def authorize(protocol_path, authorization_path, stage):
    """Perform all authorization checks before constructing a Ledger/model."""
    protocol_path = Path(protocol_path).resolve()
    authorization_path = Path(authorization_path).resolve()
    if not authorization_path.is_file():
        raise PermissionError('Independent exact M2 stage authorization is required; M0/M1 does not authorize M2')
    protocol, authorization = read(protocol_path), read(authorization_path)
    if protocol.get('schema') != 'relation-M2-protocol-v1' or protocol.get('status') not in ('FROZEN', 'FROZEN_NOT_AUTHORIZED'):
        raise PermissionError('M2 protocol is not an exact frozen executable protocol')
    if (authorization.get('status') != 'APPROVED' or authorization.get('stage') != stage
            or authorization.get('protocol_sha256') != sha(protocol_path)
            or authorization.get('scientific_id') != protocol.get('scientific_id')
            or authorization.get('execution_source_binding_sha256') != protocol['source_binding']['sha256']
            or not authorization.get('stage_authorization_id')):
        raise PermissionError('M2 authorization does not bind this stage/protocol/scientific identity/source')
    if not authorization.get('protocol_approval'):
        raise PermissionError('A separate exact protocol approval receipt is required')
    approval = read(_reference(authorization['protocol_approval']))
    if (approval.get('status') != 'APPROVED' or approval.get('protocol_sha256') != sha(protocol_path)
            or approval.get('scientific_id') != protocol['scientific_id']):
        raise PermissionError('Protocol approval does not match this stable scientific identity')
    if stage not in EXPECTED_LIMITS or protocol.get('limits') != EXPECTED_LIMITS:
        raise ValueError('Registered M2 cumulative limits differ from the implemented fixed schedule')
    if protocol.get('reference_floor') != 1.0:
        raise ValueError('The implemented training reference floor is exactly 1.0')
    if protocol['initialization']['seed'] != 2026111301 or protocol.get('sampling_seed') != 2026111303:
        raise ValueError('Registered science seeds differ')
    if protocol.get('device') not in ('cpu', 'cuda'):
        raise ValueError('A tested CPU or CUDA device must be frozen')
    binding = _reference(protocol['source_binding'])
    verify_binding(binding)
    _reference(protocol['schedule'])
    _reference(protocol['evaluation_cases'])
    _reference(protocol['evaluation_grid'])
    _reference(protocol['runtime_environment'])
    _reference(protocol['initialization']['random_export'])
    m1 = read(_reference(protocol['m1_receipt']))
    if m1.get('status') not in ('PASS', 'ENGINEERING_PASS'):
        raise PermissionError('M2 requires a completed PASS M1 receipt')
    if stage == 'paired':
        if not authorization.get('init_receipt'):
            raise PermissionError('Paired stage must bind its exact SL512 completion receipt')
        upstream = read(_reference(authorization['init_receipt']))
        if (upstream.get('status') != 'COMPLETE' or upstream.get('scientific_id') != protocol['scientific_id']
                or upstream.get('stage') != 'init' or upstream.get('cursor') != 512
                or upstream.get('protocol_sha256') != sha(protocol_path)):
            raise PermissionError('Paired parent SL512 completion does not match the frozen scientific identity')
    return protocol, authorization


def _verify_runtime(protocol):
    """Check the actual interpreter/libraries after deterministic setup."""
    import numpy as np
    import torch
    expected = read(_reference(protocol['runtime_environment']))
    actual = dict(python=sys.version, executable=str(Path(sys.executable).resolve()),
                  torch=torch.__version__, numpy=np.__version__,
                  torch_init_sha256=sha(torch.__file__), cuda_runtime=torch.version.cuda,
                  cuda_available=torch.cuda.is_available(), device=protocol['device'],
                  torch_threads=torch.get_num_threads(), deterministic=torch.are_deterministic_algorithms_enabled(),
                  CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                  TF32=bool(torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32))
    for key, value in actual.items():
        comparison = str(Path(expected[key]).resolve()) if key == 'executable' else expected[key]
        if comparison != value:
            raise ValueError('Actual runtime differs from frozen environment: ' + key)
    if protocol['device'] == 'cuda' and torch.cuda.get_device_name() != expected['gpu']:
        raise ValueError('GPU model differs from registered environment')
    return actual


def _load_schedule(protocol):
    from .evaluate import METHODS
    schedule = read(_reference(protocol['schedule']))
    cases = read(_reference(protocol['evaluation_cases']))
    if (schedule.get('schema') != 'relation-M2-data-schedule-v1'
            or schedule.get('seed') != 2026111302 or schedule.get('paired_stream_seed') != 2026111402):
        raise ValueError('Registered scheduling seeds/schema differ')
    for phase, count, batch, full in (('init', 512, 8, 4), ('paired', 2048, 4, 2)):
        rows = schedule[phase]
        if len(rows) != count:
            raise ValueError('Incomplete immutable schedule')
        for index, row in enumerate(rows, 1):
            if row['update'] != index or len(row['tasks']) != batch:
                raise ValueError('Update ordinal/batch differs from preregistered schedule')
            if any(tid not in schedule['training_metadata'] for tid in row['tasks']):
                raise ValueError('Schedule includes a task outside the registered training pool')
            sizes = [(schedule['training_metadata'][tid]['spec']['width'],
                      schedule['training_metadata'][tid]['spec']['height']) for tid in row['tasks']]
            if sum(size == (10, 10) for size in sizes) != full:
                raise ValueError('Full-parent schedule proportion changed')
    if len(cases['LOCAL66']) != 66 or cases['FULL12_ids'] != [f'F{i:03}' for i in range(12)]:
        raise ValueError('Fixed exposed evaluation populations changed')
    grid = read(_reference(protocol['evaluation_grid']))
    expected_grid = [dict(endpoint=endpoint, method=method, population=population, case_id=case['id'])
                     for endpoint in (0, 512, 2048) for population in ('LOCAL66', 'FULL12')
                     for case in cases[population] for method in METHODS]
    if (grid.get('schema') != 'relation-M2-evaluation-grid-v1' or grid.get('requests') != expected_grid
            or grid.get('normal_count') != 1404 or grid.get('automatic_retry') is not False):
        raise ValueError('Registered evaluation grid differs from exact execution order')
    return schedule, cases


def _stage_ledger(protocol, authorization, authorization_path, stage):
    # A stable stage path prevents a new output directory from resetting use.
    scientific_id = protocol['scientific_id']
    if not scientific_id or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for ch in scientific_id):
        raise ValueError('Scientific identity must be a stable path-safe registered identifier')
    folder = HOME / 'science' / scientific_id / stage
    receipt = dict(stage=stage, scientific_id=scientific_id,
                   authorization_sha256=sha(authorization_path), authorization=authorization)
    entry = folder / 'STAGE_AUTHORIZATION.json'
    if entry.exists() and read(entry) != receipt:
        raise PermissionError('Same-stage continuation requires the original authorization')
    resource = folder/'RESOURCE_ACCOUNTING.json'
    if resource.exists() and read(resource)['status'] in ('COMPLETE', 'SCIENTIFIC_PASS', 'NO_DEVELOPMENT_SIGNAL', 'STOP_SEVERE_DEGRADATION'):
        raise RuntimeError('This stage is finalized; read its existing receipt instead of rerunning it')
    ledger = Ledger(folder, protocol['limits'][stage], 'M2:' + scientific_id + ':' + stage)
    if not entry.exists():
        put(entry, receipt)
    return ledger


def _initial_actor(protocol, ledger):
    from .runtime import load_export, instrument
    ref = protocol['initialization']['random_export']
    if ref.get('engineering_only') is not False:
        raise PermissionError('M2 must use its independently registered scientific random initialization')
    actor = load_export(ref, device=protocol['device'])
    actor.train().requires_grad_(True)
    return instrument(actor, ledger)


def _save_init(actor, opt, ledger, cursor):
    from .runtime import checkpoint
    state_path = ledger.folder/'STATE.json'
    sequence = read(state_path)['commit_sequence'] + 1 if state_path.exists() else 0
    # Write the inactive slot first. A crash never overwrites the checkpoint
    # referenced by the last atomically committed STATE.json.
    ref = checkpoint(actor, opt, ledger, ledger.folder/f'current_slot{sequence % 2}.pt', cursor, engineering_only=False)
    state = dict(schema='relation-M2-init-state-v1', cursor=cursor, checkpoint=ref, commit_sequence=sequence,
                 next_action='COMPLETE' if cursor == 512 else 'UPDATE')
    put(state_path, state)
    return state


def _init(protocol, authorization, authorization_path, protocol_path):
    from .data import Pool
    from .runtime import setup, optimizer, sl_update, restore, export_actor
    schedule, _ = _load_schedule(protocol)
    ledger = _stage_ledger(protocol, authorization, authorization_path, 'init')
    try:
        setup(protocol['initialization']['seed'], protocol['device'])
        _verify_runtime(protocol)
        actor = _initial_actor(protocol, ledger)
        opt = optimizer(actor)
        pool = Pool(ledger)
        state_path = ledger.folder/'STATE.json'
        if state_path.exists():
            state = read(state_path)
            saved = restore(actor, opt, ledger, state['checkpoint'], engineering_only=False)
            ledger.prior_wall = max(ledger.prior_wall, float(saved['resource_elapsed_s']))
            ledger.save()
            cursor = state['cursor']
            if saved['cursor'] != cursor:
                raise ValueError('Initialization cursor differs from committed checkpoint')
        else:
            if ledger.ids or ledger.counts['optimizer_attempts']:
                raise RuntimeError('Paid initialization work has no closed checkpoint; no free replay')
            cursor = 0
            state = _save_init(actor, opt, ledger, cursor)
        for row in schedule['init'][cursor:]:
            ledger.check()
            update = row['update']
            sl_update(actor, opt, row['tasks'], pool, ledger, f'init_{update:04}')
            state = _save_init(actor, opt, ledger, update)
        if state['cursor'] != 512 or ledger.counts['optimizer_completed'] != 512:
            raise RuntimeError('SL512 did not complete the exact fixed endpoint')
        if ledger.counts['request_completed'] != 4096 or ledger.counts['request_attempts'] != 4096:
            raise RuntimeError('SL512 processing ledger differs from the 4096 registered teacher replays')
        export = export_actor(actor, ledger.folder/'SL512.export.pt', False, protocol['scientific_id'])
        receipt = dict(status='COMPLETE', stage='init', scientific_id=protocol['scientific_id'], cursor=512,
                       protocol_sha256=sha(protocol_path), checkpoint=state['checkpoint'], export=export,
                       actual_counts=dict(ledger.counts), engineering_only=False)
        put(ledger.folder/'COMPLETE.json', receipt)
        ledger.close('COMPLETE')
        return receipt
    except BaseException:
        ledger.close('TECHNICAL_INCOMPLETE')
        raise


def _save_pair(actors, opts, ledger, cursor):
    from .runtime import checkpoint
    state_path = ledger.folder/'STATE.json'
    sequence = read(state_path)['commit_sequence'] + 1 if state_path.exists() else 0
    refs = {name: checkpoint(actors[name], opts[name], ledger, ledger.folder/f'{name}.slot{sequence % 2}.pt',
                             dict(cursor), engineering_only=False) for name in ('RL', 'SL')}
    if refs['RL']['rng_sha256'] != refs['SL']['rng_sha256']:
        raise ValueError('Two-stream checkpoint did not capture the same closed RNG boundary')
    state = dict(schema='relation-M2-paired-state-v1', cursor=cursor, checkpoints=refs, commit_sequence=sequence)
    put(state_path, state)
    return state


def _sample_seed(base, update, slot, trajectory):
    value = f'relation-M2:{base}:{update}:{slot}:{trajectory}'.encode()
    return int(hashlib.sha256(value).hexdigest()[:15], 16)


def _paired(protocol, authorization, authorization_path, protocol_path):
    from .data import Pool
    from .runtime import setup, load_export, instrument, optimizer, collect, rl_update, sl_update, restore, export_actor
    from .evaluate import evaluate_endpoint
    schedule, cases = _load_schedule(protocol)
    upstream = read(_reference(authorization['init_receipt']))
    _reference(upstream['export'])
    if upstream['export'].get('engineering_only') is not False:
        raise PermissionError('Paired initialization must be a scientific SL512 export')
    ledger = _stage_ledger(protocol, authorization, authorization_path, 'paired')
    try:
        setup(protocol['sampling_seed'], protocol['device'])
        _verify_runtime(protocol)
        actors = {name: instrument(load_export(upstream['export'], device=protocol['device']), ledger)
                  for name in ('RL', 'SL')}
        for actor in actors.values():
            actor.train().requires_grad_(True)
        opts = {name: optimizer(actor) for name, actor in actors.items()}
        pool = Pool(ledger)
        state_path = ledger.folder/'STATE.json'
        if state_path.exists():
            state = read(state_path)
            cursor = state['cursor']
            for name in ('RL', 'SL'):
                saved = restore(actors[name], opts[name], ledger, state['checkpoints'][name], engineering_only=False)
                ledger.prior_wall = max(ledger.prior_wall, float(saved['resource_elapsed_s']))
                ledger.save()
                if saved['cursor'] != cursor:
                    raise ValueError('Paired stream cursor mismatch')
        else:
            if ledger.ids or ledger.counts['optimizer_attempts']:
                raise RuntimeError('Paid paired work has no exact closed boundary; no free replay')
            cursor = dict(update=0, next='EVALUATE', endpoint=0, evaluations={})
            state = _save_pair(actors, opts, ledger, cursor)
        while cursor['next'] != 'COMPLETE':
            ledger.check()
            if cursor['next'] == 'EVALUATE':
                endpoint = cursor['endpoint']
                exports = {'RelationInit': upstream['export']}
                for stream, method in (('RL', 'RelationRL'), ('SL', 'RelationSL')):
                    exports[method] = export_actor(actors[stream], ledger.folder/f'{stream}_{endpoint:04}.export.pt',
                                                  False, protocol['scientific_id'])
                outcome = evaluate_endpoint(ledger, pool, cases, endpoint, exports, protocol['device'])
                cursor['evaluations'][str(endpoint)] = outcome
                status = outcome['gate']['status']
                if status == 'INSUFFICIENT':
                    raise RuntimeError('Incomplete fixed evaluation')
                if status == 'STOP_SEVERE_DEGRADATION' or endpoint == 2048:
                    cursor.update(next='COMPLETE', final_status=status)
                else:
                    cursor.update(next='RL')
                state = _save_pair(actors, opts, ledger, dict(cursor))
                continue
            update = cursor['update'] + 1
            row = schedule['paired'][update - 1]
            if row['update'] != update:
                raise ValueError('Immutable paired schedule cursor mismatch')
            if cursor['next'] == 'RL':
                groups = []
                for slot, tid in enumerate(row['tasks']):
                    group = []
                    for k in range(4):
                        group.append(collect(actors['RL'], pool, tid, ledger,
                                             f'paired_{update:04}_sample_{slot}_{k}',
                                             seed=_sample_seed(protocol['sampling_seed'], update, slot, k)))
                    groups.append(group)
                rl_update(actors['RL'], opts['RL'], groups, pool, ledger, f'paired_{update:04}_RL')
                cursor.update(next='SL')
                state = _save_pair(actors, opts, ledger, dict(cursor))
            elif cursor['next'] == 'SL':
                sl_update(actors['SL'], opts['SL'], row['tasks'], pool, ledger, f'paired_{update:04}_SL')
                cursor.update(update=update, next='EVALUATE' if update in (512, 2048) else 'RL')
                if update in (512, 2048):
                    cursor['endpoint'] = update
                state = _save_pair(actors, opts, ledger, dict(cursor))
            else:
                raise ValueError('Unknown persisted paired cursor')
        expected_requests = cursor['update']*36 + len(cursor['evaluations'])*468
        if (ledger.counts['request_completed'] != expected_requests
                or ledger.counts['request_attempts'] != expected_requests
                or ledger.counts['optimizer_completed'] != cursor['update']*2
                or ledger.counts['optimizer_attempts'] != cursor['update']*2):
            raise RuntimeError('Paired closed ledger differs from the exact executed schedule')
        receipt = dict(status=cursor['final_status'], stage='paired', scientific_id=protocol['scientific_id'],
                       protocol_sha256=sha(protocol_path), completed_paired_updates=cursor['update'],
                       checkpoints=state['checkpoints'], evaluations=cursor['evaluations'],
                       actual_counts=dict(ledger.counts), downstream_authorized=False)
        put(ledger.folder/'COMPLETE.json', receipt)
        ledger.close(cursor['final_status'])
        return receipt
    except BaseException:
        ledger.close('TECHNICAL_INCOMPLETE')
        raise


def _supervise(protocol, authorization_path, protocol_path, stage):
    """Use the existing tested Windows Job Object supervisor in a new folder."""
    from rc3_r04_m1_recovery_20261111 import supervisor
    scientific_id = protocol['scientific_id']
    if not scientific_id or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for ch in scientific_id):
        raise ValueError('Invalid stable scientific identifier')
    folder = HOME/'science'/scientific_id/stage
    resource = folder/'RESOURCE_ACCOUNTING.json'
    old = read(resource) if resource.exists() else None
    if old and old['status'] in ('COMPLETE', 'SCIENTIFIC_PASS', 'NO_DEVELOPMENT_SIGNAL', 'STOP_SEVERE_DEGRADATION'):
        raise RuntimeError('A finalized stage cannot be launched again')
    controls = folder/'controls'
    attempts = sorted(controls.glob('attempt_*'))
    supervised = 0.
    for attempt in attempts:
        receipt = attempt/'SUPERVISOR_RECEIPT.json'
        if not receipt.exists():
            raise RuntimeError('An earlier supervisor attempt has no final accounting; resolve it without resetting use')
        supervised += read(receipt)['elapsed_s']
    prior = max(float(old['elapsed_s']) if old else 0., supervised)
    remaining = protocol['limits'][stage]['wall_s'] - prior
    if remaining <= 30:
        raise TimeoutError('No cumulative M2 stage wall budget remains')
    if old and prior > old['elapsed_s']:
        old['elapsed_s'] = prior
        put(resource, old)
    number = len(attempts) + 1
    output = controls/f'attempt_{number:04}'
    if output.exists():
        raise RuntimeError('Supervisor attempt name already consumed')
    environment = read(_reference(protocol['runtime_environment']))
    command = [environment['executable'], '-B', '-X', 'utf8', '-m', HOME.name+'.m2',
               '--protocol', str(Path(protocol_path).resolve()), '--authorization', str(Path(authorization_path).resolve()),
               '--stage', stage, '--worker']
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1')
    marker = hashlib.sha256((scientific_id+':'+stage).encode()).hexdigest()[:16]
    original_stage = supervisor.STAGE
    # The legacy supervisor source stays byte-identical.  Its new claims are
    # directed to this scientific stage, never a historical experiment folder.
    supervisor.STAGE = controls
    entry_m, entry_w = time.monotonic(), time.time()
    receipt = None
    try:
        receipt = supervisor.launch_supervised(command, output, f'relation-M2-{marker}-{number:04}',
            wall_limit_s=remaining, closeout_reserve_s=10, termination_tolerance_s=2,
            cwd=ROOT, env=env, entry_monotonic=entry_m, entry_wall=entry_w)
    finally:
        supervisor.STAGE = original_stage
        elapsed = max(time.monotonic()-entry_m, time.time()-entry_w)
        if resource.exists():
            accounting = read(resource)
        else:
            accounting = dict(identity='M2:'+scientific_id+':'+stage, limits=protocol['limits'][stage],
                              counts={}, pending={}, status='TECHNICAL_INCOMPLETE', scientific_updates=0, elapsed_s=0.)
        accounting['elapsed_s'] = max(accounting['elapsed_s'], prior+elapsed)
        if receipt is None or receipt['status'] != 'COMPLETE':
            accounting['status'] = 'SUPERVISOR_STOPPED'
        put(resource, accounting)
        append(folder/'EVENTS.jsonl', dict(event='supervisor_accounting', elapsed_s=accounting['elapsed_s'],
                                         attempt=number, supervisor_elapsed_s=elapsed,
                                         receipt=None if receipt is None else (output/'SUPERVISOR_RECEIPT.json').relative_to(ROOT).as_posix()))
    if receipt['status'] != 'COMPLETE':
        raise RuntimeError('Supervised worker failed or timed out; its requests and wall time remain consumed')
    result = read(folder/'COMPLETE.json')
    print(dict(stage=stage, status=result['status'], scientific_id=scientific_id,
               supervisor_receipt=(output/'SUPERVISOR_RECEIPT.json').relative_to(ROOT).as_posix()))


def main(argv=None):
    parser = argparse.ArgumentParser(description='M2 requires a separate exact approved stage authorization')
    parser.add_argument('--protocol', default=str(HOME/'M2_PROTOCOL.json'))
    parser.add_argument('--authorization', required=True)
    parser.add_argument('--stage', choices=('init', 'paired'), required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    protocol, authorization = authorize(args.protocol, args.authorization, args.stage)
    if not args.worker:
        return _supervise(protocol, args.authorization, args.protocol, args.stage)
    if not os.environ.get('RC3_SUPERVISOR_RUN_ID', '').startswith('relation-M2-'):
        raise PermissionError('M2 worker requires the registered outer process-tree supervisor')
    result = (_init(protocol, authorization, args.authorization, args.protocol) if args.stage == 'init' else
              _paired(protocol, authorization, args.authorization, args.protocol))
    print(dict(stage=args.stage, status=result['status'], scientific_id=protocol['scientific_id']))


if __name__ == '__main__':
    main()
