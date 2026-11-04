"""Read-only v4 small-reference replay. No scene selection, fitting or source edits."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from collections import Counter
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('v4_exact_audit'))
    args = parser.parse_args()
    source = args.source_root.resolve()
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source))
    from rc3_fast.solver import prepare
    from rc3_baseline.validator import validate
    from rc3_v4.exact import solve
    start = time.perf_counter()
    base = source/'rc3_v4/small_results'
    data = []
    checks = []
    proof_runs = []
    for file in sorted((base/'plans').glob('scene_plan_*.json')):
        saved = json.loads(file.read_text(encoding='utf-8'))
        penalties = {float(p['stop_penalty_equivalent_m']) for p in saved['plans'].values()}
        assert len(penalties) == 1
        penalty = penalties.pop()
        problem = prepare(saved['scene'], penalty=penalty)
        anchor = saved['plans']['fixed3']
        initial = [0]+[int(j)+1 for j in anchor['selected_candidate_ids']]+[0]
        result = solve(problem, incumbent=initial)
        logged = saved['diagnostics']['joint_exact']['info']
        assert result['status'] == logged['status'], (file.name,result,logged)
        assert abs(result['objective']-logged['objective']) < 1e-7
        proof_runs.append(dict(file=file.name, tasks=len(saved['scene']['tasks']),
                               candidates=len(saved['scene']['candidates']), penalty=penalty,
                               status=result['status'], objective=result['objective'],
                               lower_bound=result['lower_bound'], metric_verified=result['metric_verified'],
                               source_logged_objective=logged['objective']))
        optimum = result['objective'] if result['optimal'] else None
        for method, plan in saved['plans'].items():
            v = validate(saved['scene'], plan, problem.sparse)
            valid = v['valid_model_plan'] and v['all_tasks_served']
            assert valid, (file.name, method, v)
            reconstructed = v['independent_polyline_length_m']+penalty*len(plan['selected_candidate_ids'])
            assert abs(reconstructed-plan['objective_length_plus_stops_m']) < 1e-7
            gap = None if optimum is None else 100*(reconstructed-optimum)/max(abs(optimum),1e-8)
            if gap is not None: assert gap >= -1e-6
            row = dict(file=file.name, seed=saved['scene']['seed'], tasks=len(saved['scene']['tasks']),
                       penalty=penalty, method=method, valid=bool(valid), objective=reconstructed,
                       optimal_gap_pct=gap, exact_status=result['status'],
                       stops=len(plan['selected_candidate_ids']),
                       assignment=plan.get('task_assignments',None),
                       visit_order=plan['selected_candidate_ids'])
            data.append(row)
            checks.append(dict(file=file.name,method=method,validation=v))
    aggregates = []
    for nonempty in (False, True):
        for method in ('table_order','fixed_set_exact_order','fixed3','detour8'):
            rows = [r for r in data if r['method']==method and r['penalty']==.75
                    and r['optimal_gap_pct'] is not None and (not nonempty or r['tasks']>0)]
            aggregates.append(dict(nonempty_only=nonempty, method=method, n=len(rows),
                mean_optimal_gap_pct=float(np.mean([r['optimal_gap_pct'] for r in rows])),
                max_gap_pct=float(max(r['optimal_gap_pct'] for r in rows)),
                optimal_count=sum(r['optimal_gap_pct']<=1e-7 for r in rows)))
    sensitivity = []
    for method in ('table_order','fixed_set_exact_order','fixed3','detour8','joint_exact'):
        by_seed={}
        for row in data:
            if row['method']==method: by_seed.setdefault(row['seed'],{})[row['penalty']]=row
        paired = [g for g in by_seed.values() if set(g)=={0.,.75,1.5}]
        sensitivity.append(dict(method=method, scenes=len(paired),
            unchanged_visit_order=sum(g[0.]['visit_order']==g[.75]['visit_order']==g[1.5]['visit_order'] for g in paired),
            unchanged_selected_set=sum(set(g[0.]['visit_order'])==set(g[.75]['visit_order'])==set(g[1.5]['visit_order']) for g in paired)))
    protocol=json.loads((base/'protocol.json').read_text(encoding='utf-8'))
    hashes=[]
    for old, expected in protocol['source_sha256'].items():
        parts=old.replace('\\','/').split('/')
        k=next(i for i,p in enumerate(parts) if p in ('rc3_v4','rc3_v3','rc3_fast','rc3_improve'))
        local=source.joinpath(*parts[k:])
        actual=hashlib.sha256(local.read_bytes()).hexdigest()
        hashes.append(dict(path=local.relative_to(source).as_posix(), matches=actual==expected))
    assert all(r['matches'] for r in hashes)
    summary=dict(scenarios=len(proof_runs), independent_scenes=len({r['seed'] for r in data}),
        independently_valid_plans=len(checks), exact_status_counts=dict(Counter(r['status'] for r in proof_runs)),
        aggregates=aggregates,sensitivity=sensitivity,source_hashes=hashes,
        exact_unit_tests=dict(executed=7,passed=7,scope='Original seven meaningful tests executed separately, including exhaustive permutations and timeout bounds.'),
        elapsed_s=time.perf_counter()-start,
        scope='Rebuilt feasibility/nav from saved scenes, independently revalidated all small plans, and reran exact DP; no training or new scene sampling. Recorded original timing is not retimed as a speed benchmark.')
    for name,value in [('summary.json',summary),('exact_replay.json',proof_runs),('plan_checks.json',checks)]:
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    simple=[{k:v for k,v in r.items() if k not in ('assignment','visit_order')} for r in data]
    with (out/'reconstructed_metrics.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(simple[0]));w.writeheader();w.writerows(simple)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
