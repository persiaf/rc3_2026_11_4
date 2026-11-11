"""Recovery-only bookkeeping around frozen independent M1 arithmetic.

No model, planner, optimizer or prior large CSV is imported/read here. The
predecessor checker independently recomputes only this attempt's saved tables.
Its scientific formulas remain frozen; this wrapper adds attempt lineage and
deadline-aware, uncached file hashing with optional trace events.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time

FROZEN = {
    'M1_DATA_AND_SOURCE_IDENTITY.json': '8cb63231f748e2dbf47f3c18135ea721af589a2d0d4d4e55857e69ca582feeff',
    'BATCH_SCHEDULE.json': '3676a96d0c05efb417a6be0ab00c7d3bae574cacfb6bf46e33a06bb773db90a3',
    'M1_EXECUTION_PROTOCOL.json': '99e7030e0293b87cf4302f5d76f6802fceb3d2d9c0f3816de24d46cc18ed3982',
}
OLD_RESOURCE_SHA = '2256fe392db3bf3830e364b5db695cf14b2220f0e4fa2690d0e929f67bb8d70e'
OLD_WALL = 2859.2566758999983
RUN_ID = 'rc3-r04-m1-recovery-20261111-01'


def recheck(folder, *, heartbeat=lambda: None, trace=lambda event, **fields: None):
    """Read-only arithmetic review; heartbeat must not alter EVENTS/resources."""
    folder = Path(folder).resolve()
    old = folder.parent/'rc3_r04_m1_20261111'
    checks = []

    def identity(path):
        path = Path(path)
        start = time.perf_counter()
        trace('file_hash_start', path=str(path), role='independent_new_attempt_review', cache=False)
        digest = hashlib.sha256(); size = 0
        with path.open('rb') as stream:
            while True:
                heartbeat()
                block = stream.read(1 << 20)
                if not block: break
                size += len(block); digest.update(block)
        heartbeat()
        answer = dict(path=path.name, bytes=size, sha256=digest.hexdigest())
        trace('file_hash_end', path=str(path), bytes=size, elapsed_s=time.perf_counter()-start,
              sha256=answer['sha256'], role='independent_new_attempt_review', cache=False)
        return answer

    for name, expected in FROZEN.items():
        prior, current = identity(old/name), identity(folder/name)
        if prior['sha256'] != expected or current['sha256'] != expected:
            raise ValueError('Frozen scientific metadata changed: '+name)
        checks.append(dict(check='exact_prior_bytes_'+name, status='PASS', sha256=expected))
    if identity(old/'RESOURCE_ACCOUNTING.json')['sha256'] != OLD_RESOURCE_SHA:
        raise ValueError('Original failed attempt resource bytes changed')
    old_resource = json.loads((old/'RESOURCE_ACCOUNTING.json').read_text(encoding='utf-8-sig'))
    link = json.loads((folder/'ATTEMPT_LINK.json').read_text(encoding='utf-8-sig'))
    prior = link['old_attempt']; current = link['new_attempt']
    if (old_resource['wall_s'] != OLD_WALL or old_resource['status'] != 'FAILED_PRESERVED'
            or old_resource['counts']['critic_optimizer_attempts'] != 0
            or prior['resource_sha256'] != OLD_RESOURCE_SHA or prior['wall_s'] != OLD_WALL
            or prior['status'] != 'FAILED_PRESERVED' or prior['critic_optimizer_attempts'] != 0
            or current['run_id'] != RUN_ID or current['wall_cap_s'] != 1800
            or current['max_attempts'] != 1 or link['engineering']['cap_s'] != 300
            or link['prior_consumption_not_reset'] is not True
            or link['scientific_files_unchanged'] != FROZEN):
        raise ValueError('Recovery attempt linkage/caps differ from explicit authorization')
    checks.append(dict(check='old_failed_consumption_kept_and_new_attempt_distinct', status='PASS'))
    # Reuse the already independently implemented arithmetic, not the production
    # metric implementation; do not call any old full-audit entry point.
    from rc3_r04_m1_20261111 import independent_review as frozen
    original_identity = frozen._file_identity
    frozen._file_identity = identity
    try:
        result = frozen.recheck(folder, heartbeat=heartbeat)
    finally:
        frozen._file_identity = original_identity
    result['recovery_lineage_checks'] = checks
    result['recovery_run_id'] = RUN_ID
    result['retained_original_attempt_wall_s'] = OLD_WALL
    result['review_implementation_scope'] = (
        'Frozen independently implemented M1 arithmetic reused on NEW saved tables only; '
        'new wrapper checks exact metadata/old failure linkage and instruments uncached hashes. '
        'Not a new independent mathematical implementation and not a rerun of old large CSV audits.')
    result['review_wrapper_identity'] = identity(Path(__file__))
    return result
