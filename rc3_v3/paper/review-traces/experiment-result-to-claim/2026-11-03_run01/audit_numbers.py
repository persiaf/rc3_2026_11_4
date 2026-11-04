"""Read-only recomputation of frozen aggregate statistics and Markdown tables."""
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

ROOT = Path('rc3_v3')
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
rows = read(ROOT / 'results/metrics.json')
summary = read(ROOT / 'results/summary.json')
errors = []


def included(row, group):
    n = row['tables']
    predicates = {'all': True, 'primary_up_to_196': 0 < n <= 196,
                  'in_distribution_100': n == 100, 'smaller_transfer': 0 < n < 100,
                  'larger_transfer': 100 < n <= 196, 'stress_over_196': n > 196}
    return predicates[group] if group in predicates else n == int(group.split('_')[1])


def same(actual, expected, label, tolerance=1e-8):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=tolerance):
        errors.append(dict(label=label, actual=actual, expected=expected))


for entry in summary['summaries']:
    subset = [r for r in rows if included(r, entry['group']) and r['config'] == entry['config']]
    good = [r for r in subset if r['status'] == 'complete']
    expected = dict(total=len(subset), success=len(good), failed=len(subset)-len(good),
                    mean_objective_m=statistics.mean(r['objective_m'] for r in good),
                    mean_improvement_pct=statistics.mean((r['reference_objective_m']-r['objective_m'])/r['reference_objective_m']*100 for r in good),
                    median_planning_ms=statistics.median(r['planning_s'] for r in good)*1000,
                    p95_planning_ms=float(np.percentile([r['planning_s'] for r in good], 95))*1000,
                    median_total_ms=statistics.median(r['total_s'] for r in good)*1000,
                    mean_dp_calls=statistics.mean(r['dp_calls'] for r in good),
                    mean_overrun_rate=statistics.mean(r['overrun_rate'] for r in good),
                    mean_fallback_rate=statistics.mean(r['fallback_rate'] for r in good))
    for field, value in expected.items():
        same(entry[field], value, '/'.join([entry['group'], entry['config'], field]))

key = lambda r: (r['seed'], r['nx'], r['ny'], r['mode'])
lookup = {(key(r), r['config']): r for r in rows}
for entry in summary['comparisons']:
    subset = [r for r in rows if included(r, entry['group']) and r['config'] == entry['config'] and r['status'] == 'complete']
    pairs = [(r, lookup[key(r), entry['reference']]) for r in subset]
    delta = [(b['objective_m']-a['objective_m'])/b['objective_m']*100 for a, b in pairs]
    expected = dict(paired_n=len(delta), mean_reduction_pct=statistics.mean(delta),
                    wins=sum(d > 1e-7 for d in delta), ties=sum(abs(d) <= 1e-7 for d in delta),
                    losses=sum(d < -1e-7 for d in delta),
                    median_planning_speedup=statistics.median(b['planning_s']/a['planning_s'] for a, b in pairs))
    for field, value in expected.items():
        same(entry[field], value, '/'.join([entry['group'], entry['config'], entry['reference'], field]))

slookup = {(r['group'], r['config']): r for r in summary['summaries']}
clookup = {(r['group'], r['config'], r['reference']): r for r in summary['comparisons']}
checked_report = 0
for line in (ROOT / 'RESULTS.md').read_text(encoding='utf-8').splitlines():
    if not line.startswith('| '):
        continue
    cells = [x.strip() for x in line.strip('|').split('|')]
    if len(cells) == 7 and (cells[0], cells[1]) in slookup:
        record = slookup[cells[0], cells[1]]
        assert cells[2] == f"{record['success']}/{record['total']}"
        for cell, field, tol in zip(cells[3:], ['mean_improvement_pct', 'median_planning_ms', 'median_total_ms', 'mean_dp_calls'], [.000501, .00501, .00501, .00501]):
            same(cell, record[field], 'report/' + field, tol)
        checked_report += 1
    elif len(cells) == 6 and (cells[0], 'learned_calls_' + cells[1], cells[2]) in clookup:
        record = clookup[cells[0], 'learned_calls_' + cells[1], cells[2]]
        same(cells[3], record['paired_n'], 'report/n')
        same(cells[4], record['mean_reduction_pct'], 'report/mean', .000501)
        for value, expected in zip(json.loads(cells[5]), record['ci95_pct']):
            same(value, expected, 'report/ci', .000501)
        checked_report += 1
    elif len(cells) == 5 and ('in_distribution_100', cells[0]) in slookup:
        record = slookup['in_distribution_100', cells[0]]
        for cell, field, tol in zip(cells[1:4], ['mean_improvement_pct', 'median_planning_ms', 'mean_dp_calls'], [.000501, .00501, .00501]):
            same(cell, record[field], 'report/wall/' + field, tol)
        same(cells[4], record['mean_overrun_rate']*100, 'report/overrun', .0501)
        checked_report += 1

manifest = read(ROOT / 'results/manifest.json')
training = read(ROOT / 'run_v3_stopfix_20261102/training_summary.json')
deployment = read(ROOT / 'deployment/manifest.json')
model = ROOT / 'deployment/policy.npz'
hash_ok = sha(model) == manifest['model_sha256'] == deployment['checkpoint_sha256'] == training['checkpoints']['best.npz']
engineering = read(ROOT / 'engineering_profile.json')
grouped = {}
for row in engineering['runs']:
    grouped.setdefault((row['scene_id'], row['method']), []).append(row)
old, new, old_total, new_total = [], [], [], []
for scene_id in range(engineering['validation_scenes']):
    a, b = grouped[scene_id, 'old_fixed3'], grouped[scene_id, 'new_fixed3']
    old.append(statistics.median(r['planning_s'] for r in a))
    new.append(statistics.median(r['planning_s'] for r in b))
    old_total.append(statistics.median(r['total_s'] for r in a))
    new_total.append(statistics.median(r['total_s'] for r in b))
    same(max(r['objective_m'] for r in a+b), min(r['objective_m'] for r in a+b), 'engineering/cost', 1e-7)
engineering_recomputed = dict(old_median_planning_ms=statistics.median(old)*1000,
                              new_median_planning_ms=statistics.median(new)*1000,
                              median_per_scene_speedup=statistics.median(a/b for a, b in zip(old, new)),
                              ratio_of_planning_medians=statistics.median(old)/statistics.median(new),
                              old_median_total_ms=statistics.median(old_total)*1000,
                              new_median_total_ms=statistics.median(new_total)*1000,
                              new_faster_scenes=sum(a>b for a,b in zip(old,new)))
for field, value in engineering_recomputed.items():
    same(engineering['summary'][field], value, 'engineering/' + field)
audit = dict(status='pass' if not errors and hash_ok else 'fail', reviewed_at='2026-11-03',
             summary_rows_recomputed=len(summary['summaries']), comparison_means_counts_speedups_recomputed=len(summary['comparisons']),
             report_table_rows_crosschecked=checked_report, metrics_rows=len(rows),
             successful_rows=sum(r['status'] == 'complete' for r in rows), cases=len(set(key(r) for r in rows)),
             model_hash_consistent=hash_ok, model_bytes=model.stat().st_size,
             engineering_runs_recomputed=len(engineering['runs']), engineering_summary=engineering_recomputed,
             ci_scope='CI endpoints checked against frozen summary, not independently regenerated. Canonical wall group is in_distribution_100; repeats and duplicated group rows are not extra scenes.',
             source_hashes={str(p): sha(p) for p in [ROOT/'results/manifest.json', ROOT/'results/metrics.json', ROOT/'results/summary.json', ROOT/'RESULTS.md', ROOT/'run_v3_stopfix_20261102/training_summary.json', ROOT/'engineering_profile.json', ROOT/'verification/verification.json']},
             errors=errors)
Path(__file__).with_name('numeric_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in audit.items() if k not in ['source_hashes', 'ci_scope']}, ensure_ascii=False))
