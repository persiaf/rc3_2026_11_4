"""Append-only interpretation of stored adapter role metadata. No science imports."""
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOME = ROOT / 'rc3_gl_v4_20261109'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def write_new(p, value):
    with p.open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    start = time.perf_counter()
    folder = HOME / 'runs/SCIENCE/DEV216'
    catalog_path = folder / 'ACTOR_INDEX.json'
    catalog = read(catalog_path)['models']['11622026']
    independent = read(HOME / 'INDEPENDENT_AUDIT.json')
    need(independent['status'] == 'PASS' and independent['actor_packages_exact'] == 5,
         'Independent exact stored-tensor audit required')
    need(set(catalog) == {'SL256', 'A128', 'B128', 'C128'}, 'Unexpected roles')
    original = read(ROOT / 'rc3_s3_evaluator/S3_MODEL_INDEX.json')['models']['11622026']['SL256']
    for role, entry in catalog.items():
        need(entry['role'] == entry['cursor'] == entry['provenance']['role'] == role,
             'Actor role conflict')
        need(entry['seed'] == entry['provenance']['seed'] == 11622026, 'Seed conflict')
        need(sha(ROOT / entry['path']) == entry['checkpoint_sha256'], 'Actor bytes changed')
        source = entry['provenance']
        need(sha(ROOT / source['training_checkpoint_path']) == source['training_checkpoint_sha256'],
             'Source checkpoint changed')
        if role == 'SL256':
            need(source['training_checkpoint_path'] == original['path'] and
                 source['training_checkpoint_sha256'] == original['checkpoint_sha256'] and
                 entry['model_tensor_sha256'] == original['model_tensor_sha256'], 'Wrong common SL source')
        else:
            done = read(HOME / 'runs/SCIENCE' / role / 'COMPLETE.json')
            need(done['status'] == 'COMPLETE_FIXED_128' and done['updates'] == 128 and
                 source['training_checkpoint_path'] == done['checkpoint'] and
                 entry['model_tensor_sha256'] == done['tensor_sha256'], 'Not fixed128 endpoint')

    events = []; previous = '0' * 64
    for line in (folder / 'events.jsonl').read_bytes().splitlines():
        e = json.loads(line); h = e.pop('sha256')
        canonical = json.dumps(e, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf8')
        need(e['sequence'] == len(events) and e['previous'] == previous and
             hashlib.sha256(canonical).hexdigest() == h, 'Invalid event chain')
        e['sha256'] = h; events.append(e); previous = h
    head = read(folder / 'HEAD.json')
    need(head['sha256'] == previous and head['sequence'] == len(events), 'Bad journal anchor')
    starts = {e['request']['id']: e['request'] for e in events if e['event'] == 'request_start'}
    with (HOME / 'results/ENDPOINT_EVALUATION.csv').open(encoding='utf8', newline='') as f:
        rows = list(csv.DictReader(f))
    csv_index = {r['artifact_path']: r for r in rows}
    need(len(rows) == len(csv_index) == len(starts) == 216, 'Incomplete or duplicate grid')
    mapping = []; counts = Counter()
    for e in events:
        if e['event'] != 'request_complete':
            continue
        p = ROOT / e['artifact_path']; before = sha(p); r = read(p); q = r['request']
        need(before == e['artifact_sha256'], 'Result bytes do not match committed event')
        need(starts[q['id']] == q, 'Request was not the admitted request')
        role = q['study_method']; row = csv_index[e['artifact_path']]
        need(r['study_method'] == row['model'] == role and row['artifact_sha256'] == before,
             'Declared study/CSV identity conflict')
        if role in catalog:
            entry = catalog[role]
            need(r['weight'] == entry, 'Actual model catalog is not declared model')
            need(row['tensor_sha256'] == entry['model_tensor_sha256'], 'CSV tensor mismatch')
            need(q['method'] == r['plan']['method'] == 'frozen_supervised', 'Unexpected generic adapter')
            need(r['training_weight_role'] == 'SL256', 'Unexpected legacy metadata')
            mismatch = role != r['training_weight_role']
            counts['neural'] += 1; counts['legacy_default_aliases'] += int(mismatch)
            counts['correct_SL_defaults'] += int(not mismatch)
        else:
            need(role in ('Fixed3', 'Detour16') and r['weight'] is None and
                 r['training_weight_role'] is None, 'Unexpected native model/source')
            entry = None; mismatch = False; counts['native'] += 1
        for key in ('J', 'L', 'K', 'raw_J', 'raw_L', 'raw_K'):
            need(float(row[key]) == r['metrics'][key], 'Numeric CSV differs from stored result')
        need(sha(p) == before, 'Read-only audit altered result')
        mapping.append(dict(request_id=q['id'], case_id=q['case_id'], artifact_path=e['artifact_path'],
            artifact_sha256=before, declared_study_method=role,
            actual_parameter_role=None if entry is None else entry['role'],
            actor_path=None if entry is None else entry['path'],
            actor_sha256=None if entry is None else entry['checkpoint_sha256'],
            model_tensor_sha256=None if entry is None else entry['model_tensor_sha256'],
            training_source=None if entry is None else entry['provenance'],
            original_training_weight_role=r['training_weight_role'],
            original_plan_method=r['plan']['method'], legacy_default_alias=mismatch))
    need(len(mapping) == 216 and counts == Counter(neural=144, native=72,
         legacy_default_aliases=108, correct_SL_defaults=36), 'Unexpected identity findings')
    need(read(folder / 'COMPLETE.json')['event_tail'] == previous, 'Completion mismatch')
    result = dict(status='PASS_ACTUAL_MODEL_IDENTITIES_WITH_DISCLOSED_METADATA_DEFECT',
        source_binding_sha256=sha(HOME / 'SOURCE_BINDING.json'),
        actor_index_sha256=sha(catalog_path), independent_audit_sha256=sha(HOME / 'INDEPENDENT_AUDIT.json'),
        journal_tail=previous, counts=dict(counts), records=mapping,
        new_scientific_calls=0, model_materializations=0, optimizer_updates=0,
        original_artifacts_changed=0, wall_s=time.perf_counter() - start)
    out = HOME / 'METHOD_IDENTITY_AUDIT.json'; write_new(out, result)
    write_new(HOME / 'METHOD_IDENTITY_CLARIFICATION.json', dict(
        finding='GLV4_META_001', severity='Derived adapter role metadata; actual model identities and metrics verified',
        cause='run.complete_request passes generic frozen_supervised to the old evaluation adapter for all four neural study roles. Catalog override_role loads each correct actor, but the inherited training_weight_role and plan.method labels were not relabeled.',
        scope='DEV216: 108 A/B/C artifacts retain default training_weight_role=SL256; 36 true SL and 72 native labels are consistent.',
        authoritative_identity='request.study_method and row.study_method, exact actor path/hash/tensor identity and training provenance, all strictly cross-validated by METHOD_IDENTITY_AUDIT.json',
        disposition='Keep all frozen scientific source, raw result bytes, journals, parameter tensors, costs and CSV bytes unchanged. Append this exact source-bound interpretation map; do not use the generic pipeline label as the scientific group identity.',
        future_use='A later evaluator revision should separate adapter kind from study model identity. No such unapproved new scientific execution or rerun is performed here.',
        numerical_impact='None: 216 CSV group/metric pairs match declared study method and actual bound model source. This is not a claim that the original role metadata was correct.',
        audit_sha256=sha(out), original_artifacts_changed=0, new_scientific_calls=0))
    print(dict(status=result['status'], **dict(counts), wall_s=result['wall_s']))


if __name__ == '__main__':
    main()
