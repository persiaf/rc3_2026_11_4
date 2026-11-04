"""Independent review reconstruction; run from the delivered project root."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from rc3_baseline.validator import validate

OUT = Path(__file__).parent
ROOT = Path.cwd()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def near(a, b, tolerance=1e-7):
    return abs(a-b) <= tolerance

def main():
    manifest = read('rc3_v4/results/manifest.json')
    assert manifest['status'] == 'complete'
    saved = read('rc3_v4/results/metrics.json')
    summary = read('rc3_v4/results/summary.json')
    errors, records, totals = [], [], Counter()
    hashes = []
    for group in ('source_sha256', 'model_sha256', 'data_sha256'):
        for name, expected in manifest[group].items():
            ok = Path(name).is_file() and sha(name) == expected
            hashes.append(dict(group=group, path=name, matched=ok))
            if not ok:
                errors.append('hash: '+name)
    files = sorted(Path('rc3_v4/results/plans').glob('*.json'))
    for file in files:
        data = read(file)
        meta = data['metadata']
        grouped = defaultdict(list)
        totals['scenes'] += 1
        totals['tables_'+str(meta['tables'])] += 1
        for label, run in data['runs'].items():
            p = data['plans'][label]
            check = validate(data['scene'], p, data['feasibility'])
            line = p['route_polyline']
            length = math.fsum(math.dist(a,b) for a,b in zip(line, line[1:]))
            objective = length + .75*len(p['selected_candidate_ids'])
            valid = check['valid_model_plan'] and check['all_tasks_served']
            totals['runs'] += 1
            totals['valid_plans'] += int(valid)
            if not valid or not near(objective, run['objective_m']) or not near(p['stop_penalty_equivalent_m'], .75):
                errors.append(f'{file.name}/{label}: independent plan or objective failed')
            grouped[run['config']['id']].append((run, objective))
        ref = float(np.mean([p[1] for p in grouped['fixed3']]))
        for name, pairs in grouped.items():
            if len(pairs) != pairs[0][0]['config']['repeats']:
                errors.append(f'{file.name}/{name}: repeat count')
            if any(value > ref+1e-7 or not near(run['reference_objective_m'], ref) for run,value in pairs):
                errors.append(f'{file.name}/{name}: independent Fixed3 bound')
            records.append(dict(**meta, config=name, repeats=len(pairs),
                objective=float(np.mean([p[1] for p in pairs])),
                planning_s=float(np.median([p[0]['planning_s'] for p in pairs])),
                gain=float(np.mean([100*(ref-value)/max(ref,1e-8) for _,value in pairs])),
                calls=float(np.mean([p[0]['dp_calls'] for p in pairs])),
                total_s=float(np.median([p[0]['planning_s'] for p in pairs]))+data['preprocessing_s']))
    lookup = {(r['case_index'],r['config']):r for r in records}
    for r in saved:
        rebuilt = lookup[(r['case_index'],r['config'])]
        if not near(rebuilt['objective'],r['objective_m']) or not near(rebuilt['planning_s'],r['planning_s']):
            errors.append('saved aggregate differs from raw plans: '+str((r['case_index'],r['config'])))
    comparisons = []
    for r in summary['comparisons']:
        if r['group'] != 'main_100':
            continue
        aa = sorted([a for a in records if a['tables']==100 and a['config']==r['config']],
                    key=lambda a:(a['seed'],a['nx'],a['ny'],a['mode']))
        bb = {b['case_index']:b for b in records if b['tables']==100 and b['config']==r['reference']}
        delta = np.array([100*(bb[a['case_index']]['objective']-a['objective']) /
                          max(bb[a['case_index']]['objective'],1e-8) for a in aa])
        rng = np.random.default_rng(731203)
        ci = np.quantile(delta[rng.integers(len(delta),size=(20000,len(delta)))].mean(1),[.025,.975])
        ok = (near(float(delta.mean()),r['mean_reduction_pct'])
              and np.allclose(ci,r['ci95_pct'],rtol=0,atol=1e-7)
              and len(delta)==r['paired_n'])
        comparisons.append(dict(current=r['config'],reference=r['reference'],n=len(delta),
            mean_reduction_pct=float(delta.mean()),ci95_pct=ci.tolist(),reported_matches=bool(ok)))
        if not ok:
            errors.append('paired comparison mismatch: '+r['config']+'/'+r['reference'])
    tables = []
    for n in (36,100,196,400):
        names = sorted({r['config'] for r in records if r['tables']==n})
        for name in names:
            rows = [r for r in records if r['tables']==n and r['config']==name]
            tables.append(dict(tables=n,config=name,n=len(rows),mean_gain_pct=float(np.mean([r['gain'] for r in rows])),
                median_ms=1000*float(np.median([r['planning_s'] for r in rows])),
                cold_median_ms=1000*float(np.median([r['total_s'] for r in rows])),
                mean_calls=float(np.mean([r['calls'] for r in rows]))))
    small = list(csv.DictReader(Path('rc3_v4/small_results/metrics.csv').open(encoding='utf-8')))
    small_stats = []
    for name in sorted({r['method'] for r in small}):
        rows = [r for r in small if r['penalty']=='0.75' and r['method']==name and r['optimal_gap_pct']]
        small_stats.append(dict(method=name,certified_cases=len(rows),
            mean_gap_pct=float(np.mean([float(r['optimal_gap_pct']) for r in rows]))))
    expected = dict(scenes=39,runs=855,valid_plans=855,tables_36=9,tables_100=18,tables_196=9,tables_400=3)
    if dict(totals) != expected or len(records)!=357:
        errors.append('denominator mismatch')
    units = read('rc3_v4/verification/summary.json')
    out = dict(all_passed=not errors,errors=errors,counts=dict(totals),method_records=len(records),
        raw_reconstruction_matches=len(saved),hash_checks=hashes,primary_paired=comparisons,
        tables=tables,small_primary_recomputed=small_stats,unit_tests_at_review=units['unit_tests'],
        unit_suite_all_passed=units['all_passed'],
        primary_scope='100 tables, 18 fresh scenes; seed/scale/wall results are supplementary',
        independence='Reviewer script does not call rc3_v4.report; uses shared independent geometric validator and separate scalar path cost/bootstrap code.')
    (OUT/'numeric_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(all_passed=out['all_passed'],counts=out['counts'],method_records=len(records),primary_pairs=len(comparisons),errors=errors)))

if __name__=='__main__':
    main()
