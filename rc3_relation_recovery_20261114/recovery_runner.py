"""Prepared M2a-only executor. Requires NEW exact approvals; not run this turn.

There is no M2b path and no second-attempt/restart path in recovery version 1.
An earlier incomplete recovery requires another disposition, never a fresh
directory or silent replay. The frozen original stage remains read-only.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from .persistence import Ledger, PersistenceError, put, read, sha, digest
from .recovery_adapter import checked_ref, original_origin, restore_sl238, save_update_boundary
from .runtime_boundary import require_recovery_authorization

ROOT = Path(__file__).resolve().parents[1]
HOME = Path(__file__).resolve().parent
CANONICAL = ROOT / 'rc3_relation_drl_20261113/science/relation_navcover_m2_v1_20261113/init/recovery_v1_20261114'


def admit(proposal_path, approval_path, authorization_path):
    if not Path(approval_path).is_file() or not Path(authorization_path).is_file():
        raise PermissionError('Separate new recovery approval and stage authorization are absent')
    proposal, approval, auth = read(proposal_path), read(approval_path), read(authorization_path)
    require_recovery_authorization(proposal, approval, auth)
    if (proposal['recovery_id'] != 'relation_navcover_m2a_recovery_v1_20261114'
            or proposal['scientific_id'] != 'relation_navcover_m2_v1_20261113'
            or proposal['canonical_recovery_folder'] != CANONICAL.relative_to(ROOT).as_posix()
            or proposal['cumulative_limits'] != dict(request_attempts=4102, optimizer_attempts=512, wall_s=14400)
            or proposal['historical_consumption'] != dict(request_attempts=1910, optimizer_attempts=238, wall_s=2221.312)
            or proposal['remaining_limits'] != dict(request_attempts=2192, optimizer_attempts=274, wall_s=12178.688)
            or proposal['M2b_authorized'] is not False):
        raise ValueError('Recovery proposal exceeds the exact reviewed M2a scope')
    for key in ('original_protocol', 'original_source_binding', 'original_stage_authorization',
                'sl238_audit', 'historical_binding', 'new_source_binding', 'recovery_schedule'):
        checked_ref(proposal[key])
    for key in ('original_source_binding', 'historical_binding', 'new_source_binding'):
        for name, expected in read(checked_ref(proposal[key]))['files'].items():
            if sha(ROOT / name) != expected:
                raise ValueError('Changed bound input/source: ' + name)
    audit = read(checked_ref(proposal['sl238_audit']))
    schedule = read(checked_ref(proposal['recovery_schedule']))
    original = read(checked_ref(read(checked_ref(proposal['original_protocol']))['schedule']))['init'][238:]
    if ([(r['update'], r['tasks']) for r in schedule['rows']] != [(r['update'], r['tasks']) for r in original]
            or len(schedule['rows']) != 274):
        raise ValueError('Exact original task schedule suffix required')
    if (audit['historical_wall_s'] != 2221.312 or audit['original_unique_request_debits'] != 1910
            or audit['checkpoint']['cursor'] != 238):
        raise ValueError('Archived recovery origin mismatch')
    return proposal, approval, auth, audit, schedule


def worker(proposal, approval, auth, audit, schedule):
    if not os.environ.get('RC3_SUPERVISOR_RUN_ID', '').startswith('relation-M2a-recovery-v1-'):
        raise PermissionError('Reviewed process-tree supervision is mandatory')
    if (CANONICAL / 'EVENTS.v2.jsonl').exists():
        raise RuntimeError('A recovery attempt is already consumed; this version does not restart it')
    events = [json.loads(x) for x in checked_ref(audit['journal']).read_text(encoding='utf8').splitlines()]
    origin = original_origin(audit, events)
    ledger = Ledger(CANONICAL, proposal['cumulative_limits'], 'M2:relation_navcover_m2_v1_20261113:init', origin=origin)
    try:
        from . import runtime as rt
        from rc3_relation_drl_20261113.data import Pool
        from rc3_relation_drl_20261113.m2 import _verify_runtime
        protocol = read(checked_ref(proposal['original_protocol']))
        rt.setup(protocol['initialization']['seed'], protocol['device'])
        _verify_runtime(protocol)
        config = __import__('torch').load(checked_ref(audit['checkpoint']), map_location='cpu', weights_only=False)['config']
        actor = rt.RelationActor(**{k: v for k, v in config.items() if k != 'model_id'}).to(protocol['device'])
        actor.train().requires_grad_(True)
        opt = rt.optimizer(actor)
        restore_sl238(actor, opt, ledger, proposal, approval, auth)
        actor = rt.instrument(actor, ledger)
        pool = Pool(ledger)
        put(CANONICAL / 'ORIGIN_STATE.json', dict(checkpoint=audit['checkpoint'], counts=origin['counts'],
                                               ids=origin['ids'], historical_overlap_preserved=True))
        ledger.publish()
        state = None
        for row in schedule['rows']:
            ledger.check()
            if row['update'] != ledger.cursor + 1:
                raise ValueError('Unexpected optimizer cursor')
            rt.sl_update(actor, opt, row['tasks'], pool, ledger, row['optimizer_id'])
            state = save_update_boundary(actor, opt, ledger, row['update'])
        if (ledger.cursor != 512 or ledger.pending or ledger.counts['optimizer_attempts'] != 512
                or ledger.counts['optimizer_completed'] != 512 or ledger.counts['request_attempts'] != 4102
                or ledger.counts['request_completed'] != 4102 or ledger.counts['request_failed'] != 1):
            raise ValueError('Recovered endpoint differs from exact cumulative disposition')
        export = rt.export_actor(actor, CANONICAL / 'SL512_RECOVERED.export.pt', False, proposal['scientific_id'])
        ledger.close('COMPLETE_RECOVERED')
        put(CANONICAL / 'WORKER_COMPLETE.json', dict(status='COMPLETE_RECOVERED', stage='M2a_recovery',
            scientific_id=proposal['scientific_id'], proposal_digest=digest(proposal), checkpoint=state['checkpoint'],
            export=export, actual_counts=dict(ledger.counts), historical_resource=audit['resource'],
            historical_failed_event_preserved=True, active_optimized_teacher_slots=4096,
            technical_recompute_requests=6, M2b_authorized=False))
    except BaseException as exc:
        if isinstance(exc, PersistenceError) and exc.phase == 'journal_append_uncertain':
            ledger.release()
        else:
            try:
                ledger.close('TECHNICAL_INCOMPLETE')
            except BaseException:
                ledger.release()
        raise


def supervise(args, admitted, entry_m, entry_w):
    proposal, approval, auth, audit, schedule = admitted
    if CANONICAL.exists():
        raise RuntimeError('Canonical recovery directory already consumed; no alternate directory/restart')
    CANONICAL.mkdir(parents=False)
    put(CANONICAL / 'RECOVERY_AUTHORIZATION.json', dict(proposal_digest=digest(proposal), approval=approval, authorization=auth,
                                                      original_stage_authorization=proposal['original_stage_authorization']))
    from rc3_r04_m1_recovery_20261111 import supervisor
    protocol = read(checked_ref(proposal['original_protocol']))
    executable = read(checked_ref(protocol['runtime_environment']))['executable']
    command = [executable, '-B', '-X', 'utf8', '-m', 'rc3_relation_recovery_20261114.recovery_runner',
               '--proposal', str(Path(args.proposal).resolve()), '--approval', str(Path(args.approval).resolve()),
               '--authorization', str(Path(args.authorization).resolve()), '--worker']
    controls = CANONICAL / 'controls'
    old_stage, supervisor.STAGE = supervisor.STAGE, controls
    receipt = None
    try:
        receipt = supervisor.launch_supervised(command, controls / 'attempt_0001', 'relation-M2a-recovery-v1-0001',
            wall_limit_s=12178.688, closeout_reserve_s=10, termination_tolerance_s=2,
            cwd=ROOT, env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1'),
            entry_monotonic=entry_m, entry_wall=entry_w)
    finally:
        supervisor.STAGE = old_stage
        elapsed = max(time.monotonic() - entry_m, time.time() - entry_w)
        put(CANONICAL / 'OUTER_ACCOUNTING.json', dict(prior_wall_s=2221.312, current_attempt_wall_s=elapsed,
            cumulative_wall_s=2221.312 + elapsed, cumulative_cap_s=14400,
            worker_status=None if receipt is None else receipt['status'],
            prior_request_debits=1910, historical_records_untouched=True))
    if receipt is None or receipt['status'] != 'COMPLETE' or 2221.312 + elapsed > 14400:
        raise RuntimeError('Recovery failed or exceeded wall; retain all consumption and stop')
    result = read(CANONICAL / 'WORKER_COMPLETE.json')
    put(CANONICAL / 'RECOVERY_COMPLETE.json', dict(result, supervisor_receipt_sha256=sha(controls / 'attempt_0001/SUPERVISOR_RECEIPT.json'),
                                                outer_accounting_sha256=sha(CANONICAL / 'OUTER_ACCOUNTING.json')))


def main(argv=None):
    entry_m, entry_w = time.monotonic(), time.time()
    parser = argparse.ArgumentParser(description='Requires new exact recovery approval; no M2b support')
    parser.add_argument('--proposal', default=str(HOME / 'RECOVERY_PROTOCOL_DRAFT.json'))
    parser.add_argument('--approval', required=True)
    parser.add_argument('--authorization', required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    admitted = admit(args.proposal, args.approval, args.authorization)
    if args.worker:
        return worker(*admitted)
    return supervise(args, admitted, entry_m, entry_w)


if __name__ == '__main__':
    main()
