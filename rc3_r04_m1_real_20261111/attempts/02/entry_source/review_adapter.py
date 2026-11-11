"""Reuse frozen independent M1 arithmetic on this task's saved tables only.

The sole source substitution fixes the historical split location for nested
attempt directories. No arithmetic, scientific criterion, or audit counter is
changed. This is a provenance/bookkeeping adapter, not a new implementation of
the independent mathematical review.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
FROZEN = {
    'M1_DATA_AND_SOURCE_IDENTITY.json': '8cb63231f748e2dbf47f3c18135ea721af589a2d0d4d4e55857e69ca582feeff',
    'BATCH_SCHEDULE.json': '3676a96d0c05efb417a6be0ab00c7d3bae574cacfb6bf46e33a06bb773db90a3',
    'M1_EXECUTION_PROTOCOL.json': '99e7030e0293b87cf4302f5d76f6802fceb3d2d9c0f3816de24d46cc18ed3982',
}
FROZEN_REVIEW_SHA256 = '394129fc1713affee803dea987635dadd444dbfc64ef60fe5fe6fb2b5fbecb34'
PRIOR_RECEIPTS = (
    ('rc3_r04_m1_20261111/RESOURCE_ACCOUNTING.json',
     '2256fe392db3bf3830e364b5db695cf14b2220f0e4fa2690d0e929f67bb8d70e',
     'wall_s', 2859.2566758999983),
    ('rc3_r04_m1_recovery_20261111/RESOURCE_ACCOUNTING.json',
     '12d6b3af248cec43e496f973f215dc6e50dd2a69e5d183a4b52616039941e5b2',
     'wall_s', 11.586754083633423),
    ('rc3_r04_m1_recovery_20261111/ENGINEERING_ACCOUNTING.json',
     '4ed739cec8a6c7686d5002595f984a94fbf91f983cbca3c8eebb72bd9f52baf1',
     'total_wall_s', 14.386909658434888),
)
ORIGINAL_LINE = "    old_split_path = folder.parent/'rc3_critic_diagnostics_20261111/CALIBRATION_SPLIT.json'"
REPLACEMENT_LINE = "    old_split_path = Path(__file__).resolve().parents[1]/'rc3_critic_diagnostics_20261111/CALIBRATION_SPLIT.json'"


def recheck(folder, *, heartbeat=lambda: None,
            trace=lambda event, **fields: None):
    """Read-only numerical audit inside the already running worker budget.

    ``heartbeat`` must check deadlines without mutating EVENTS/resource files,
    which the frozen checker hashes both before and after its arithmetic.
    Current-attempt resource counts are checked here; task-wide retries and
    cumulative wall accounting are checked separately by the outer controller.
    """
    folder = Path(folder).resolve()
    allowed = ROOT/'rc3_r04_m1_real_20261111'/'attempts'
    if folder.parent != allowed or folder.name not in ('01', '02', '03'):
        raise ValueError('Independent review requires a registered current attempt directory')
    checks = []

    def identity(path):
        path = Path(path)
        started = time.perf_counter()
        trace('file_hash_start', path=str(path), role='independent_new_attempt_review', cache=False)
        digest = hashlib.sha256()
        size = 0
        with path.open('rb') as stream:
            while True:
                heartbeat()
                block = stream.read(1 << 20)
                if not block:
                    break
                size += len(block)
                digest.update(block)
        heartbeat()
        result = dict(path=path.name, bytes=size, sha256=digest.hexdigest())
        trace('file_hash_end', path=str(path), bytes=size,
              elapsed_s=time.perf_counter()-started, sha256=result['sha256'],
              role='independent_new_attempt_review', cache=False)
        return result

    binding_identity = identity(folder/'SOURCE_BINDING.json')
    binding = json.loads((folder/'SOURCE_BINDING.json').read_text(encoding='utf-8-sig'))
    expected_run_id = 'rc3-r04-m1-real-20261111-'+folder.name
    if binding['run_id'] != expected_run_id:
        raise ValueError('Registered current attempt run_id differs')
    checks.append(dict(check='registered_current_attempt_identity', status='PASS',
                       run_id=expected_run_id, source_binding=binding_identity))
    for name, expected in FROZEN.items():
        for path in (ROOT/'rc3_r04_m1_20261111'/name, folder/name):
            if identity(path)['sha256'] != expected:
                raise ValueError('Frozen scientific metadata changed: '+str(path))
        checks.append(dict(check='exact_frozen_scientific_bytes', name=name,
                           sha256=expected, status='PASS'))
    prior = []
    for relative, expected, field, wall in PRIOR_RECEIPTS:
        path = ROOT/relative
        if identity(path)['sha256'] != expected:
            raise ValueError('Original resource receipt changed: '+relative)
        receipt = json.loads(path.read_text(encoding='utf-8-sig'))
        if receipt[field] != wall:
            raise ValueError('Original consumed wall differs: '+relative)
        if field == 'wall_s' and (receipt['status'] != 'FAILED_PRESERVED' or
                receipt['counts']['critic_optimizer_attempts'] != 0 or
                receipt['counts']['actor_optimizer_attempts'] != 0):
            raise ValueError('Original failed attempt classification differs: '+relative)
        prior.append(dict(path=relative, sha256=expected, wall_s=wall))
    checks.append(dict(check='both_failures_and_engineering_original_bytes_preserved',
                       status='PASS', receipts=prior))

    source_path = ROOT/'rc3_r04_m1_20261111'/'independent_review.py'
    source_identity = identity(source_path)
    if source_identity['sha256'] != FROZEN_REVIEW_SHA256:
        raise ValueError('Frozen independent arithmetic source changed')
    source_bytes = source_path.read_bytes()
    heartbeat()
    if hashlib.sha256(source_bytes).hexdigest() != FROZEN_REVIEW_SHA256:
        raise ValueError('Frozen arithmetic source changed between read and compile')
    source = source_bytes.decode('utf-8')
    if source.count(ORIGINAL_LINE) != 1:
        raise ValueError('Independent arithmetic path substitution is not unique')
    adapted = source.replace(ORIGINAL_LINE, REPLACEMENT_LINE, 1)
    # __file__ stays bound to the original verified source so the replacement
    # resolves only its repository root and original source identity stays true.
    namespace = dict(__name__='rc3_r04_m1_real_20261111.frozen_independent_arithmetic',
                     __file__=str(source_path))
    exec(compile(adapted, str(source_path), 'exec'), namespace)
    namespace['_file_identity'] = identity
    result = namespace['recheck'](folder, heartbeat=heartbeat)
    result['current_attempt_directory'] = folder.relative_to(ROOT).as_posix()
    result['current_run_id'] = expected_run_id
    result['preserved_prior_receipts'] = prior
    result['preserved_prior_science_and_engineering_wall_s'] = sum(item['wall_s'] for item in prior)
    result['exact_metadata_and_prior_receipt_checks'] = checks
    result['review_source_adaptation'] = dict(
        frozen_source=source_path.relative_to(ROOT).as_posix(),
        frozen_source_sha256=FROZEN_REVIEW_SHA256,
        exact_original_line=ORIGINAL_LINE, exact_replacement_line=REPLACEMENT_LINE,
        replacement_count=1,
        adapted_source_sha256=hashlib.sha256(adapted.encode('utf-8')).hexdigest(),
        numerical_formula_changes=0)
    result['review_implementation_scope'] = (
        'Frozen independently implemented M1 saved-table arithmetic reused on NEW tables only. '
        'One exact path substitution supports nested attempts; all numerical formulas, '
        'gates, source split checks, hash chain and per-worker counters are unchanged. '
        'Not a new independent mathematical implementation; no prior large CSV audit rerun. '
        'Cumulative retry accounting is verified separately from the per-attempt audit.')
    result['review_wrapper_identity'] = identity(Path(__file__))
    return result
