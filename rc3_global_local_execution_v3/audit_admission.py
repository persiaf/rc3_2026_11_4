"""Independent read-only audit of saved stage-admission regression evidence.

No model, optimizer, task generation, navigation, teacher, H or retained test
evaluation. Input paths are repository-relative; the output is restricted here.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
BASE = 'aa0247c8c38243504028bfad24795e0db7120264'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def study_path(name):
    p = (ROOT / name).resolve()
    if not p.is_relative_to(HERE) or not p.is_file():
        raise ValueError('Expected saved evidence inside new execution study: ' + str(name))
    return p


def audit(result_name, binding_name):
    start = time.perf_counter(); cpu = time.process_time()
    rp, bp = study_path(result_name), study_path(binding_name)
    r, binding = read(rp), read(bp)
    if r.get('status') != 'PASS':
        raise AssertionError('Saved regression is not PASS')
    sources = r.get('source_sha256') or binding.get('source_sha256')
    if not isinstance(sources, dict) or not sources:
        raise AssertionError('No source binding in saved regression')
    if binding.get('source_sha256') != sources:
        raise AssertionError('Result and execution source binding differ')
    for name, h in sources.items():
        p = (ROOT / name).resolve()
        if not p.is_relative_to(ROOT) or digest(p) != h:
            raise AssertionError('Bound source changed: ' + name)
    proto = HERE / 'V3_EXECUTION_CANDIDATE.json'
    registry = HERE / 'REGISTRY.json'
    p, reg = read(proto), read(registry)
    if p.get('status') in ('APPROVED', 'ACTIVE', 'AUTHORIZED') or p.get('activated') is True:
        raise AssertionError('Unapproved candidate unexpectedly activated')
    if p.get('seeds_sha256') != digest(registry):
        raise AssertionError('Protocol registry hash mismatch')
    old_registry = ROOT / 'rc3_global_local_impl_review/PROPOSED_SEEDS_V3.json'
    if registry.read_bytes() != old_registry.read_bytes():
        raise AssertionError('Registry drift from reviewed v3 allocations')
    assert len(reg['S1_teachers']) == 256 and len(reg['S2_train']) == 96
    assert all(len(x['task_seeds']) == 8 for x in reg['S2_train'])
    assert len(reg['S3']) == 108 and reg['policy_seeds'] == [11622026, 11622027, 11622028]
    mode_endpoint = Counter((reg['S2_train'][x['parent']]['mode'], x['endpoints']) for x in reg['S1_teachers'])
    assert sorted(mode_endpoint.values()) == [64, 64, 64, 64]
    a_cells = Counter((x['group'], tuple(x['size']), x['A_condition']) for x in reg['S3'])
    assert len(a_cells) == 18
    assert sorted(a_cells.values()) == [4] * 9 + [8] * 9
    old_manifest = json.loads(subprocess.check_output(['git', 'show', BASE + ':project_management/GIT_MANIFEST.json'], cwd=ROOT))
    # Check original trainer evidence plus old protocol/registry/pointer bytes;
    # the root agent performs the separate full-project byte manifest audit.
    protected = {n: v for n, v in old_manifest['files'].items() if n.startswith('rc3_global_local_trainer/')}
    extra = ['rc3_global_local/FUTURE_DESIGN_STATUS.json', 'rc3_global_local/S2_S3_protocol.json',
             'rc3_global_local/S2_S3_protocol_v2.json', 'rc3_global_local/future_seeds_v2.json',
             'rc3_global_local_impl_review/PROPOSED_PROTOCOL_V3.json',
             'rc3_global_local_impl_review/PROPOSED_SEEDS_V3.json']
    for n in extra:
        protected[n] = old_manifest['files'][n]
    for name, expected in protected.items():
        f = ROOT / name
        if f.stat().st_size != expected['bytes'] or digest(f) != expected['sha256']:
            raise AssertionError('Frozen prior evidence changed: ' + name)
    return dict(schema='rc3-independent-stage-admission-audit-v1', reviewer='/root/v3_admission_independent',
        status='PASS_SAVED_EVIDENCE_PENDING_REPORT_FINALIZATION', base=BASE,
        result=str(rp.relative_to(ROOT)).replace('\\', '/'), result_sha256=digest(rp),
        binding=str(bp.relative_to(ROOT)).replace('\\', '/'), binding_sha256=digest(bp),
        source_files_checked=len(sources), prior_frozen_files_checked=len(protected),
        candidate_sha256=digest(proto), registry_sha256=digest(registry),
        candidate_status=p.get('status'), active_v2_pointer_sha256=digest(ROOT / extra[0]),
        S1_mode_endpoint_counts={str(k): v for k, v in sorted(mode_endpoint.items())},
        S3_metadata_cases=108, target_tasks_materialized=False,
        scientific_calls=dict(model=0, optimizer=0, task_generation=0, navigation=0, teacher=0, H=0,
                              S3_target_queries=0, original60_semantic_reads=0),
        scope='Read saved JSON/source hashes and prior original bytes only. Metadata allocation is not task feasibility.',
        wall_s=time.perf_counter()-start, cpu_s=time.process_time()-cpu,
        script_sha256=digest(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--result', required=True); ap.add_argument('--binding', required=True)
    ap.add_argument('--output', default='rc3_global_local_execution_v3/INDEPENDENT_AUDIT.json')
    args = ap.parse_args(); dest = (ROOT / args.output).resolve()
    if not dest.is_relative_to(HERE):
        raise ValueError('Output outside new study')
    r = audit(args.result, args.binding)
    history = read(dest).get('verification_run_history', []) if dest.exists() else []
    history.append({k: r[k] for k in ('status', 'wall_s', 'cpu_s', 'result_sha256', 'script_sha256')})
    r['verification_run_history'] = history
    dest.write_text(json.dumps(r, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')
    print(json.dumps({k: r[k] for k in ('status', 'wall_s', 'source_files_checked', 'prior_frozen_files_checked')}))


if __name__ == '__main__':
    main()
