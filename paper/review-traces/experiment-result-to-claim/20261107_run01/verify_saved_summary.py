"""Independent standard-library-only S2 closeout verification.

Consumes existing audits, source bytes, receipts and derived CSVs. No runtime
imports, tensors, inference, scene generation, geometric checks or optimization.
Run from repository root after final analysis and figures have closed.
"""
from pathlib import Path
import collections
import csv
import hashlib
import json
import math
import statistics
import time

ROOT = Path(__file__).resolve().parents[4]
REPORT = ROOT / 'rc3_s2_results_20261107'
RUN = ROOT / 'rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01'
SEEDS = (11622026, 11622027, 11622028)
PROTOCOL = '0c5bc7d7a9557edf2af76f51a9f418550f0b4ce1ec6a8b47ddea95c6256e78bb'
SOURCE = '0f0ffd2d2dda2fefefcfcb5cb26dc5197b5d11bd1332b0a5546992409049bfe0'
AUTH = '4dba652b2156b88848c4c1c49826952414a5b6d08baa7ce250d710b3ea312423'


def run():
    started = time.perf_counter()
    checks = collections.Counter()
    failures = []
    artifacts = {}

    def ck(value, group, detail=''):
        checks[group] += 1
        if not value:
            failures.append({'group': group, 'detail': str(detail)})

    def sha(path):
        value = hashlib.sha256(path.read_bytes()).hexdigest()
        artifacts[path.relative_to(ROOT).as_posix()] = value
        return value

    def read(path):
        sha(path)
        return json.loads(path.read_text(encoding='utf8'))

    def rows(name):
        path = REPORT / 'analysis' / name
        sha(path)
        with path.open(encoding='utf8', newline='') as f:
            return list(csv.DictReader(f))

    binding_path = ROOT / 'rc3_navigation_compat_20261106/SOURCE_BINDING_UPDATED.json'
    binding = read(binding_path)
    ck(sha(binding_path) == SOURCE, 'source_manifest')
    ck(len(binding['source_sha256']) == 34, 'source_manifest')
    for name, expected in binding['source_sha256'].items():
        ck(sha(ROOT / name) == expected, 'approved_source_bytes', name)
    ck(sha(ROOT / binding['protocol_path']) == PROTOCOL, 'protocol_bytes')
    ck(sha(ROOT / binding['registry_path']) == binding['registry_sha256'], 'registry_bytes')
    auth_path = ROOT / 'rc3_global_local_execution_v3/authorizations/20261107_S2/S2_AUTHORIZATION.json'
    auth = read(auth_path)
    ck(sha(auth_path) == AUTH and auth['stages'] == ['S2'] and
       auth['optimizer_seeds'] == list(SEEDS) and auth['S3_authorized'] is False,
       'actual_S2_authorization')
    authority = (ROOT / 'rc3_global_local_execution_v3/authority.py').read_text(encoding='utf8')
    runner = (ROOT / 'rc3_global_local_execution_v3/runner.py').read_text(encoding='utf8')
    registry_source = (ROOT / 'rc3_global_local_trainer/data.py').read_text(encoding='utf8')
    ck("STAGES = ('S1-A', 'S1-B', 'S2')" in authority, 'S3_execution_boundary')
    ck("choices=('S1-A','S1-B','S2')" in runner, 'S3_execution_boundary')
    ck("Only declared train/development inputs are admitted; S3/old60 are sealed" in registry_source,
       'S3_execution_boundary')
    summary = read(REPORT / 'analysis/ANALYSIS_SUMMARY.json')
    ck(summary['analysis_checks']['failed'] == summary['finite_checks']['failed'] == 0, 'analysis_checks')
    ck(summary['source_manifest_sha256'] == SOURCE and summary['protocol_sha256'] == PROTOCOL, 'analysis_binding')
    ck(summary['completed_update_rows'] == 3072 and summary['RL_trajectory_rows'] == 18432,
       'analysis_totals')
    ck(summary['saved_terminal_files'] == 18539 and summary['trajectory_cache_hits'] == 1, 'analysis_totals')
    updates, trajectories = rows('updates.csv'), rows('trajectories.csv')
    attempts, dev = rows('terminal_attempts.csv'), rows('development_feasibility.csv')
    audit_results = []
    selected_metrics = {}
    for seed in SEEDS:
        audit = read(REPORT / f'audit_seed_{seed}.json')
        audit_results.append(audit)
        ck(audit['status'] == 'PASS_CLOSED_SEED' and not audit['failures'], 'seed_audit')
        ck(audit['source_manifest_sha256'] == SOURCE and audit['protocol_sha256'] == PROTOCOL,
           'seed_audit_binding')
        completion = RUN / f'stage_admission/S2/{seed}/COMPLETE.json'
        receipt = read(completion)
        ck(sha(completion) == audit['completion_sha256'], 'completion_identity')
        ck(receipt['status'] == 'COMPLETE' and receipt['error'] is None, 'completion_status')
        for name, expected in receipt['predecessors'].items():
            ck(sha(RUN / 'stage_admission' / name / 'COMPLETE.json') == expected,
               'actual_predecessor_completion', name)
        ck(sha(RUN / f'stage_admission/S2/{seed}/AUTHORIZATION.json') == receipt['authorization_record_sha256'],
           'actual_admission_authorization')
        ss = summary['per_seed'][str(seed)]
        ck(ss['status'] == 'COMPLETE' and ss['RL_updates'] == ss['continuedSL_updates'] == 512,
           'actual_fixed_endpoints')
        ck(ss['resource_ledger'] == audit['resource_ledger'], 'ledger_cross_derivation')
        ck(abs(ss['stage_wall_s'] - audit['shared_stage_wall_s']) < 1e-7 and ss['stage_wall_s'] < 7200,
           'shared_wall_cross_derivation')
        ur = [r for r in updates if int(r['seed']) == seed]
        tr = [r for r in trajectories if int(r['seed']) == seed]
        ck(len(ur) == 1024 and len(tr) == 6144, 'CSV_counts')
        for phase in ('RL', 'continued_SL'):
            pr = sorted((r for r in ur if r['phase'] == phase), key=lambda r: int(r['update']))
            ck(len(pr) == 512 and [int(r['cursor_after']) for r in pr] == list(range(1, 513)),
               'CSV_fixed_cursor', f'{seed}/{phase}')
            ck(all(math.isfinite(float(r['loss'])) for r in pr), 'CSV_finite_losses')
        selected_metrics[str(seed)] = {}
        for metric, field in (('raw_J', 'raw_mean_J'), ('H_J', 'H_mean_J'), ('raw_L', 'raw_mean_L'),
                              ('H_L', 'H_mean_L'), ('raw_K', 'raw_mean_K'), ('H_K', 'H_mean_K'),
                              ('H_gain', 'mean_H_gain')):
            mean = statistics.fmean(float(r[metric]) for r in tr)
            selected_metrics[str(seed)][field] = mean
            ck(abs(mean - ss[field]) < 1e-8, 'trajectory_mean_recomputed', f'{seed}/{field}')
        for r in tr:
            ck(abs(float(r['raw_J']) - float(r['raw_L']) - .75 * float(r['raw_K'])) < 1e-7,
               'CSV_raw_objective')
            ck(abs(float(r['H_J']) - float(r['H_L']) - .75 * float(r['H_K'])) < 1e-7,
               'CSV_H_objective')
            ck(abs(float(r['H_gain']) - float(r['raw_J']) + float(r['H_J'])) < 1e-7,
               'CSV_H_gain')
        dr = [r for r in dev if int(r['seed']) == seed]
        ck(len(dr) == 2 and {int(r['update']) for r in dr} == {256, 512}, 'DEV_fixed_cursors')
        ck(all(int(r['observed_cases']) == int(r['feasible_cases']) == 18 for r in dr), 'DEV_repeated_denominator')
        terminal = audit['resource_ledger']['terminal']
        ck(sum(v['terminal_attempts'] for v in terminal.values()) <= 8192 and
           terminal['RL']['terminal_attempts'] <= 6144 and terminal['development']['terminal_attempts'] <= 2048,
           'terminal_budget')
        ck(all(v['terminal_failed'] == 0 for v in terminal.values()), 'terminal_no_failure')
        ck(audit['checkpoints_audited'] == 34, 'all_saved_checkpoint_inventory')
    ck(len({x['source_SL256_tensor_sha256'] for x in audit_results}) == 3, 'seed_source_tensor_independence')
    ck(len({x['parameter_deltas']['RL']['tensor_sha256'] for x in audit_results}) == 3,
       'RL_endpoint_tensor_independence')
    ck(len({x['parameter_deltas']['continued_SL']['tensor_sha256'] for x in audit_results}) == 3,
       'SL_endpoint_tensor_independence')
    figures = read(REPORT / 'analysis/figures/FIGURE_BUILD.json')
    ck(len(figures['outputs']) == 18, 'figure_inventory')
    for output in figures['outputs']:
        p = REPORT / 'analysis/figures' / output['file']
        ck(p.stat().st_size == output['bytes'] and sha(p) == output['sha256'], 'figure_bytes')
    for name, expected in figures['input_csv_sha256'].items():
        ck(sha(REPORT / 'analysis' / name) == expected, 'figure_input_binding')
    for name, expected in figures['source_sha256'].items():
        ck(sha(REPORT / name) == expected, 'figure_source_binding')
    read(REPORT / 'FROZEN_BASELINE_AUDIT.json')
    sha(REPORT / 'FIGURE_CAPTIONS.md')
    sha(REPORT / 'ANALYSIS_METHOD.md')
    out = dict(schema='rc3-s2-independent-summary-checks-v1',
               status='PASS' if not failures else 'FAIL', checks_total=sum(checks.values()),
               check_groups=dict(checks), failures=failures, selected_metrics=selected_metrics,
               saved_paid_terminal_rows=len(attempts),
               original_raw_seed_audit_assertions=sum(a['checks_total'] for a in audit_results),
               artifact_sha256=artifacts,
               scientific_optimizer_model_inference_scene_navigation_teacher_H_calls=0,
               S3_original60_semantic_access=0, elapsed_wall_s=time.perf_counter() - started)
    destination = Path(__file__).with_name('summary_checks.json')
    with destination.open('x', encoding='utf8', newline='\n') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ('artifact_sha256', 'selected_metrics')},
                     ensure_ascii=False))
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(run())
