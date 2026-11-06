"""Independent byte/row/source cross-check; no scientific runtime imports."""
from __future__ import annotations

import datetime
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
BASE = '0b715b1eedfd3107a1e5188c25c60fe7fde113f1'
RUN = ROOT/'rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, default=HERE/'TEACHER_REPLAY_VERIFICATION.json')
    parser.add_argument('--negative', type=Path, default=HERE/'NAVIGATION_COMPATIBILITY_TESTS.json')
    parser.add_argument('--bridge', type=Path, default=HERE/'STAGE_BRIDGE_VERIFICATION.json')
    args = parser.parse_args()
    started = time.perf_counter()
    assertions = 0

    def need(condition, message):
        nonlocal assertions
        assertions += 1
        if not condition:
            raise AssertionError(message)

    replay_path = args.replay
    negative_path = args.negative
    bridge_path = args.bridge
    replay, negative, bridge = map(read, (replay_path, negative_path, bridge_path))
    old_audit = read(ROOT/'rc3_s1a_results_20261106/INDEPENDENT_AUDIT.json')
    origin = read(RUN/'RUN_MANIFEST.json')['binding']
    original_manifest = json.loads(subprocess.check_output(
        ['git', 'show', BASE+':project_management/GIT_MANIFEST.json'], cwd=ROOT))
    prefix = RUN.relative_to(ROOT).as_posix()+'/'
    original_files = {n:m for n,m in original_manifest['files'].items() if n.startswith(prefix)}
    actual_files = {p.relative_to(ROOT).as_posix() for p in RUN.rglob('*') if p.is_file()}
    need(actual_files == set(original_files), 'S1-A artifact file set differs')
    for name, meta in original_files.items():
        path = ROOT/name
        need(path.stat().st_size == meta['bytes'] and digest(path) == meta['sha256'],
             'S1-A original bytes differ: '+name)

    sources = replay['production_source_sha256']
    need(replay['verifier_sha256'] == digest(HERE/'verify_teacher_replay.py'), 'Production replay checker changed after execution')
    need(negative['test_script_sha256'] == digest(HERE/'test_navigation_compat.py'), 'Schema checker changed after execution')
    need(bridge['checker_sha256'] == digest(HERE/'bridge_regression.py'), 'Stage bridge checker changed after execution')
    need(bridge['legacy_manifest_sha256'] == digest(HERE/'LEGACY_TEACHER_REPLAY_MANIFEST.json'), 'Stage bridge declaration changed after execution')
    need(bridge['source_sha256'] == sources, 'Stage bridge verifier used a different source revision')
    for name, expected in sources.items():
        need(digest(ROOT/name) == expected, 'Verified production source changed: '+name)
    changed = sorted(n for n,h in origin['source_sha256'].items() if sources.get(n) != h)
    permitted = {'rc3_global_local_trainer/data.py', 'rc3_global_local_trainer/approval.py',
        'rc3_global_local_trainer/runner.py', 'rc3_global_local_execution_v3/authority.py',
        'rc3_global_local_execution_v3/runner.py'}
    need(set(changed).issubset(permitted), 'Scientific math/source changed beyond compatibility scope')
    need(set(sources)-set(origin['source_sha256']) == {'rc3_global_local_trainer/navigation_compat.py'},
         'Unexpected new runtime source')
    need(replay['status'] == negative['status'] == bridge['status'] == 'PASS', 'A required regression did not pass')
    need(replay['teachers_replayed'] == 256 and replay['saved_plans_revalidated'] == 512,
         'Incomplete real production replay')
    need(replay['original_source_binding_sha256'] == origin['source_manifest_sha256'],
         'Original approved teacher source binding was substituted')
    need(replay['original_bytes_preserved'] is True, 'Production replay did not preserve old data')
    need(replay['original_artifacts_before'] == replay['original_artifacts_after'], 'Production replay before/after differ')
    for result in (replay, negative):
        need(not any(result['forbidden_scientific_call_counters'].values()), 'Forbidden scientific entry was attempted')
    need(not any(bridge['scientific_calls'].values()) and not bridge['forbidden_operation_attempts'],
         'Bridge executed a scientific operation')
    need(all(r['status'] in ('PASS', 'PASS_REJECTED') for r in negative['results']), 'Negative/codec regression failed')
    need(all(r['result'] == 'EXPECTED_REJECTION' for r in bridge['expected_rejections']), 'Invalid stage request was accepted')

    rows = replay['teacher_rows']
    need([r['index'] for r in rows] == list(range(256)), 'Replayed row order/count differs')
    full = reduced = decisions = 0
    for row, old in zip(rows, old_audit['teacher_rows']):
        i = row['index']
        label_path = RUN/'teachers256'/f'label_{i:03}.json'
        label = read(label_path)
        spec = label['spec']
        folder = RUN/'data/task_draws'/f"{spec['map_seed']}_{spec['task_seed']}_A{spec['A_condition']}"
        receipt = read(folder/'receipt.json')
        need(row['passed'] and row['spec'] == spec == old['spec'], 'Teacher declaration differs: '+str(i))
        need(row['original_parent_identity'] == row['loaded_parent_identity'] == receipt['parent_identity'],
             'Parent identity differs: '+str(i))
        need(row['original_view_identity'] == row['loaded_view_identity'] == label['view_identity'],
             'View identity differs: '+str(i))
        need(row['label_sha256'] == old['label_sha256'] == digest(label_path), 'Label/source bytes differ: '+str(i))
        need(row['scene_sha256'] == receipt['scene_sha256'] == digest(folder/'scene.json'), 'Scene hash differs: '+str(i))
        need(row['task_arrays_sha256'] == receipt['arrays_sha256'] == digest(folder/'arrays.npz'), 'F/A artifact hash differs: '+str(i))
        need(row['original_task_source_sha256'] == receipt['source_sha256'] == origin['source_sha256'],
             'Task source revision was relabeled: '+str(i))
        need(row['actions'] == label['actions'] == label['H_result']['route'][1:-1], 'Teacher actions differ: '+str(i))
        need(row['label_state_checks']['all_legal'] and row['label_state_checks']['coverage_complete'],
             'Production action/mask transition failed: '+str(i))
        decisions += row['label_state_checks']['decisions']
        for metric in ('J', 'L', 'K'):
            value = row['teacher_'+metric]
            need(abs(value-label['H_result'][metric]) < 1e-7 and abs(value-old['teacher_'+metric]) < 1e-7,
                 'Teacher terminal '+metric+' differs: '+str(i))
        need(len(row['saved_plan_checks']) == 2 and all(p['valid'] for p in row['saved_plan_checks']),
             'Saved native/H plan was not revalidated: '+str(i))
        if row['navigation_schema_variant'] == 'full_with_recovered_rejected_edge_count':
            full += 1
            need(old['navigation_identity_variant'].startswith('original_navigation_'), 'Recovered wrong historical schema')
        else:
            reduced += 1
            need(row['navigation_schema_variant'] == 'legacy_reduced_original_identity'
                 and old['navigation_identity_variant'] == 'artifact_reload_without_rejected_edge_count',
                 'Reduced original identity was changed')
    need((full, reduced) == (96, 160), 'Historical96/160 split changed')
    for name in ('rc3_global_local_execution_v3/V3_EXECUTION_CANDIDATE.json',
                 'rc3_global_local_execution_v3/REGISTRY.json',
                 'rc3_global_local_execution_v3/EXECUTION_SOURCE_BINDING.json',
                 'rc3_global_local_execution_v3/H_BINDING.json',
                 'rc3_global_local/FUTURE_DESIGN_STATUS.json'):
        need(digest(ROOT/name) == original_manifest['files'][name]['sha256'], 'Frozen protocol/binding/pointer changed: '+name)

    result = dict(schema='rc3-navigation-compat-independent-review-v1', status='PASS',
        reviewer='/root/compat_independent_review', checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        script_sha256=digest(Path(__file__)), source_sha256=sources, changed_runtime_sources=changed,
        assertions=assertions, original_S1_A_files_hashed=len(original_files),
        original_S1_A_bytes_hashed=sum(v['bytes'] for v in original_files.values()),
        original_S1_A_all_bytes_preserved=True, teachers_cross_checked=256,
        recovered_original_full=full, preserved_original_reduced=reduced,
        supervision_decisions_cross_checked=decisions, saved_plans_revalidated_by_production_regression=512,
        input_receipts={p.name:digest(p) for p in (replay_path, negative_path, bridge_path)},
        negative_tests=negative['tests'], stage_bridge_assertions=bridge['assertions'],
        stage_bridge_expected_rejections=len(bridge['expected_rejections']),
        new_scene_navigation_teacher_H_model_optimizer_calls=0,
        S1_B_technical_decision='GO', S1_B_scientific_execution='NOT_EXECUTED_REQUIRES_EXPLICIT_UPDATED_SOURCE_APPROVAL_AND_STAGE_AUTHORIZATION',
        scientific_claim='Compatibility/data replay only; no learning or planning-performance improvement established.',
        wall_s=time.perf_counter()-started,
        scope='Standard-library byte/row/source comparison only. No scientific runtime imports, generation, models or optimizer. S3/old60 task semantics not read.')
    output = HERE/'INDEPENDENT_REVIEW.json'
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps({k:result[k] for k in ('status','assertions','teachers_cross_checked','wall_s')}))


if __name__ == '__main__':
    main()
