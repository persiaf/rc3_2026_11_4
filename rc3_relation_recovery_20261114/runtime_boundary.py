"""Reviewed persistence hooks for a future separately authorized M2a runner.

These hooks run no scientific calculation. Call begin before computation;
persist_request after computation; commit_update only after the optimizer
transaction and checkpoint bytes have completed. No retry accepts a callback
that could replay a teacher or call an optimizer.
"""
import time
from .persistence import PersistenceError


def persist_request(ledger, rid, record, tick):
    try:
        ledger.record_request(rid, record, tick)
        elapsed = time.perf_counter() - tick
        ledger.event('request_full_time', id=rid, full_request_s=elapsed,
                     includes='request computation, output and accounting snapshot')
        ledger.publish()
        return elapsed
    except BaseException as exc:
        # A committed completion remains complete even when its snapshot fails.
        ledger.fail('request', rid, exc)
        raise


def commit_update(ledger, cursor, checkpoint_path, state_value):
    try:
        ledger.commit_checkpoint(cursor, checkpoint_path, state_value)
        ledger.publish()
    except BaseException:
        ledger.stopped = True
        raise


def require_recovery_authorization(proposal, approval, stage_authorization):
    # Not a launch entry point. Prepared proposals and the original M2a approval
    # cannot authorize the new code or the six additional paid requests.
    from .persistence import digest
    if (not approval or not stage_authorization or approval.get('status') != 'APPROVED'
            or stage_authorization.get('status') != 'APPROVED'
            or approval.get('proposal_sha256') != digest(proposal)
            or stage_authorization.get('proposal_sha256') != digest(proposal)
            or stage_authorization.get('stage') != 'M2a_recovery'
            or stage_authorization.get('original_stage_authorization') != proposal['original_stage_authorization']):
        raise PermissionError('New exact recovery approval and stage authorization are required')
    return True
