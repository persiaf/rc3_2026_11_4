"""C2.1: bounded frozen-route attribution. No policy imports or route decoding."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
from rc3_fast import solver
from rc3_baseline.validator import validate
from .baselines import normalize
from .problem import ROOT, Cache, digest, write

OUT = ROOT / 'c21'
METHODS = ('prior', 'initial', 'model')
BASE = '0a5fa0297847bd143915420450997ffd3373983b'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def source_rows():
    records = read(ROOT/'outputs/final_evaluation.json')
    rows = {(r['case'], r['method']): r for r in records
            if r['rep'] == 0 and r['cache_mode'] == 'cold'}
    assert len(rows) == 24 * 7
    # All four saved measurements must agree on routes and quality.
    for r in records:
        ref = rows[r['case'], r['method']]
        assert r['route'] == ref['route'] and r['raw_route'] == ref['raw_route']
        for key in ('J', 'L', 'K', 'raw_J'):
            assert abs(r[key] - ref[key]) < 1e-7
    return rows


def frozen_inputs():
    paths = [ROOT/'outputs/final_evaluation.json', ROOT/'outputs/training_summary.json',
             ROOT/'protocol.yaml', ROOT/'model.py', ROOT/'baselines.py',
             ROOT/'problem.py', ROOT/'c21.py', ROOT.parent/'rc3_fast/solver.py',
             ROOT.parent/'rc3_baseline/validator.py']
    paths += sorted((ROOT/'outputs/checkpoints').glob('*.pt'))
    paths += sorted((ROOT/'outputs/scenes').glob('dev_*.json'))
    for case in range(24):
        for method in (*METHODS, 'fixed3'):
            paths.append(ROOT/'outputs/final_plans'/f'dev_{case:02}_{method}.json')
    return {str(p.relative_to(ROOT.parent)).replace('\\', '/'): digest(p) for p in paths}


def check_inputs():
    p = read(OUT/'protocol.json')
    assert all(digest(ROOT.parent/name) == sha for name, sha in p['input_sha256'].items())
    return p


def register():
    target = OUT/'protocol.json'
    if target.exists():
        raise FileExistsError('C2.1 already registered; no implicit retuning')
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == BASE
    rows = source_rows()
    assert read(ROOT/'outputs/training_summary.json')['selected_update'] == 128
    assert len(list((ROOT/'outputs/scenes').glob('dev_*.json'))) == 24
    assert len(list((ROOT/'outputs/checkpoints').glob('*.pt'))) == 5
    for case in range(24):
        scene = read(ROOT/'outputs/scenes'/f'dev_{case:02}.json')
        assert scene['seed'] == 7300000000 + case
        for method in (*METHODS, 'fixed3'):
            plan = read(ROOT/'outputs/final_plans'/f'dev_{case:02}_{method}.json')
            assert [0]+[j+1 for j in plan['visit_order_candidate_ids']]+[0] == rows[case, method]['route']
    write(target, dict(schema='rc3-c21-frozen-route-v1', date='2026-11-05', base_commit=BASE,
        intended_claim='Does learned construction retain value under identical H1, and close the Fixed3 quality gap?',
        sources=list(METHODS), development_cases=24, selected_update=128,
        H0='Already saved final_evaluation rep=0 cold routes; no new H0 calls',
        H1='normalize(joint_search(copy(H0), C, D, passes=2)); one joint_search call per route',
        stopping='Frozen solver: at most 2 sweeps; no-change early exit; internal two_opt default passes=60. No restart, recipe selection or parameter search.',
        candidate_rule='All original depot-connected candidates covering current unique tasks; existing visited nodes excluded except current node; deterministic argmin tie order',
        objective='J=L+0.75*K; same navigation, coverage and proxy geometry as C012',
        max_new_postprocess_calls=72, execution_order='case ascending; cyclic source order by case%3',
        cpu_threads=1, inference_calls=0, optimizer_updates=0, new_raw_routes=0,
        future_test_scenes_queried=0, full_pipeline_timing=False,
        timing_scope='Single additional H1 call only, including final normalize. Preparation and export/validation separately timed. No sum with historical full-pipeline medians.',
        verification_scope='72 export certificates during run; separate verify phase rechecks 72 saved H1 plans with scalar validator; frozen source metrics checked numerically without replanning',
        uncertainty='Exploratory paired scene bootstrap 10000 seed 114, two-sided 95%; development used in rollout and diagnosis, selected one training seed; no independent significance claim',
        attribution='Joint H1 gain combines deletion, replacement and order interactions; cannot identify a pure selection, order or RL contribution',
        continuation='No training or C3 authorized by this protocol; report conditional next decision only',
        input_sha256=frozen_inputs()))
    print('Registered C2.1: 24 scenes x 3 sources, maximum 72 H1 calls', flush=True)


def metrics(item, route):
    assert route[0] == route[-1] == 0
    nodes = route[1:-1]
    assert len(nodes) == len(set(nodes)) and all(1 <= j <= item.C.shape[1] for j in nodes)
    assert item.C[:, np.asarray(nodes)-1].any(1).all()
    L = float(sum(item.D[a, b] for a, b in zip(route, route[1:])))
    assert np.isfinite(L)
    K = len(nodes)
    return dict(J=L+item.base.penalty*K, L=L, K=K,
                route=list(route), candidate_ids=sorted(j-1 for j in nodes))


def append(path, row):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        stream.flush()


def run():
    p = check_inputs()
    ledger, result = OUT/'call_ledger.jsonl', OUT/'rows.jsonl'
    if ledger.exists() or result.exists():
        raise FileExistsError('Refusing implicit replay, including incomplete prior attempts')
    rows = source_rows()
    cache = Cache(1)
    started = time.perf_counter()
    write(OUT/'execution_environment.json', dict(python=platform.python_version(),
        platform=platform.platform(), numpy=np.__version__, threadpools=threadpool_info(),
        protocol_sha256=digest(OUT/'protocol.json')))
    count = 0
    for case in range(24):
        scene = read(ROOT/'outputs/scenes'/f'dev_{case:02}.json')
        tick = time.perf_counter(); item = cache.prepare(scene); preparation = time.perf_counter()-tick
        fixed = metrics(item, rows[case, 'fixed3']['route'])
        order = METHODS[case % 3:] + METHODS[:case % 3]
        for method in order:
            original = rows[case, method]
            h0 = metrics(item, original['route'])
            for key in ('J', 'L', 'K'):
                assert abs(h0[key]-original[key]) < 1e-7
                assert abs(fixed[key]-rows[case, 'fixed3'][key]) < 1e-7
            assert count < p['max_new_postprocess_calls']
            count += 1
            append(ledger, dict(call=count, case=case, method=method, status='started'))
            tick = time.perf_counter()
            route = solver.joint_search(list(h0['route']), item.C, item.D, passes=2)
            route = normalize(item, route)
            added = time.perf_counter()-tick
            h1 = metrics(item, route)
            assert h1['J'] <= h0['J']+1e-7
            tick = time.perf_counter(); plan = item.certify(route, 'c21_'+method+'_H1'); export_s = time.perf_counter()-tick
            assert [0]+[j+1 for j in plan['visit_order_candidate_ids']]+[0] == route
            assert abs(plan['objective_length_plus_stops_m']-h1['J']) < 1e-7
            path = OUT/'plans'/f'dev_{case:02}_{method}_H1.json'; write(path, plan)
            row = dict(call=count, case=case, method=method, scene_seed=scene['seed'],
                scene_mode=scene['complexity'], H0=h0, H1=h1, fixed3=fixed,
                raw_J=original['raw_J'], raw_route=original['raw_route'],
                delta_H_J=h0['J']-h1['J'], delta_H_L=h0['L']-h1['L'],
                delta_H_penalty=item.base.penalty*(h0['K']-h1['K']),
                removed_candidate_ids=sorted(set(h0['candidate_ids'])-set(h1['candidate_ids'])),
                added_candidate_ids=sorted(set(h1['candidate_ids'])-set(h0['candidate_ids'])),
                H1_added_s=added, export_validation_s=export_s,
                shared_scene_preparation_s=preparation, valid=plan['validation']['valid_model_plan'],
                plan_path=str(path.relative_to(ROOT.parent)).replace('\\', '/'), plan_sha256=digest(path))
            append(result, row)
            append(ledger, dict(call=count, case=case, method=method, status='completed', J=h1['J']))
        print(json.dumps(dict(cases_done=case+1, H1_calls=count)), flush=True)
    assert count == 72
    check_inputs()
    write(OUT/'resources.json', dict(new_bounded_H1_calls=count, internal_joint_search_calls=count,
        max_joint_sweeps_per_call=2, max_internal_two_opt_calls=144, max_two_opt_passes_per_internal_call=60,
        new_H0_calls=0, new_neural_routes=0, neural_forward_calls=0, optimizer_updates=0,
        export_geometry_certificates=72, separate_saved_plan_geometry_checks=0,
        distinct_development_scenes=24, new_raw_construction_scenes=0,
        reserved_test_scenes_queried=0, full_pipeline_measurements=0,
        elapsed_run_s=time.perf_counter()-started, protocol_sha256=digest(OUT/'protocol.json'),
        scope='New planning diagnosis: 72 bounded compound postprocessors, not zero planning. Verification/preparation separately accounted; no retraining. Internal bounds are bounds, not measured iteration counts.'))


def saved_rows():
    rows = [json.loads(s) for s in (OUT/'rows.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 72 and len({(r['case'], r['method']) for r in rows}) == 72
    return rows


def stats(values):
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(114)
    samples = a[rng.integers(len(a), size=(10000, len(a)))].mean(1)
    return dict(mean=float(a.mean()), ci95=np.quantile(samples, [.025, .975]).tolist(),
                positive_cases=int((a > 1e-7).sum()), zero_cases=int((abs(a) <= 1e-7).sum()),
                negative_cases=int((a < -1e-7).sum()), scene_values=a.tolist())


def analyze():
    check_inputs(); rows = saved_rows()
    summary = {}
    for method in METHODS:
        rr = sorted((r for r in rows if r['method'] == method), key=lambda r:r['case'])
        summary[method] = {h: {key:float(np.mean([r[h][key] for r in rr])) for key in ('J', 'L', 'K')}
                           for h in ('H0', 'H1')}
        for name, values in (
            ('delta_H_J', [r['delta_H_J'] for r in rr]),
            ('delta_H_L', [r['delta_H_L'] for r in rr]),
            ('delta_H_penalty', [r['delta_H_penalty'] for r in rr]),
            ('delta_H_relative_percent', [100*r['delta_H_J']/r['H0']['J'] for r in rr]),
            ('gap_Fixed3_J', [r['H1']['J']-r['fixed3']['J'] for r in rr]),
            ('gap_Fixed3_relative_percent', [100*(r['H1']['J']-r['fixed3']['J'])/r['fixed3']['J'] for r in rr])):
            summary[method][name] = stats(values)
        times = [1000*r['H1_added_s'] for r in rr]
        summary[method]['H1_added_ms'] = dict(median=float(np.median(times)), p95=float(np.quantile(times,.95)),
                                            min=float(min(times)), max=float(max(times)), repetitions=1)
        summary[method]['mean_removed_candidate_ids'] = float(np.mean([len(r['removed_candidate_ids']) for r in rr]))
        summary[method]['mean_added_candidate_ids'] = float(np.mean([len(r['added_candidate_ids']) for r in rr]))
    by = {(r['case'],r['method']):r for r in rows}
    paired = {}
    for base in ('prior', 'initial'):
        for h in ('H0', 'H1'):
            gain = [by[c,base][h]['J']-by[c,'model'][h]['J'] for c in range(24)]
            relative = [100*gain[c]/by[c,base][h]['J'] for c in range(24)]
            paired[f'model_vs_{base}_{h}'] = dict(absolute_J=stats(gain), relative_percent=stats(relative))
    overlap = {}
    for h in ('H0', 'H1'):
        for other in ('prior', 'initial', 'fixed3'):
            values = []
            for c in range(24):
                a = set(by[c,'model'][h]['candidate_ids'])
                b = set(by[c,'model']['fixed3']['candidate_ids'] if other == 'fixed3' else by[c,other][h]['candidate_ids'])
                values.append(len(a & b)/len(a | b))
            overlap[f'model_vs_{other}_{h}'] = stats(values)
    mode_summary = {}
    for mode in ('regular', 'mixed', 'blocked'):
        rr = [r for r in rows if r['scene_mode'] == mode and r['method'] == 'model']
        mode_summary[mode] = dict(cases=len(rr), mean_delta_H_J=float(np.mean([r['delta_H_J'] for r in rr])),
            mean_gap_fixed3_percent=float(np.mean([100*(r['H1']['J']/r['fixed3']['J']-1) for r in rr])))
    result = dict(schema='rc3-c21-attribution-v1', summary=summary, learning_increment=paired,
        candidate_jaccard=overlap, model_by_scene_mode=mode_summary,
        fixed3={k:float(np.mean([by[c,'model']['fixed3'][k] for c in range(24)])) for k in ('J','L','K')},
        scope='Exploratory conditional development-set comparisons only; no independent tests, training-seed stability or full-pipeline timing. Candidate changes and L/K accounting do not isolate causal subcomponents.')
    write(OUT/'analysis.json', result)
    flat = []
    for r in sorted(rows, key=lambda r:(r['case'],r['method'])):
        row = {k:r[k] for k in ('case','method','scene_seed','scene_mode','delta_H_J','delta_H_L','delta_H_penalty','H1_added_s','export_validation_s')}
        for h in ('H0','H1','fixed3'):
            for k in ('J','L','K'):row[h+'_'+k] = r[h][k]
        row['learning_increment_H1_prior_minus_model_J'] = by[r['case'],'prior']['H1']['J']-by[r['case'],'model']['H1']['J']
        row['gap_Fixed3_H1_J'] = r['H1']['J']-r['fixed3']['J']
        for key in ('removed_candidate_ids','added_candidate_ids'):
            row[key] = json.dumps(r[key])
        flat.append(row)
    with (OUT/'per_scene.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(json.dumps({m:dict(H0=s['H0']['J'],H1=s['H1']['J'],gain=s['delta_H_J']['mean'],
        fixed_gap_percent=s['gap_Fixed3_relative_percent']['mean']) for m,s in summary.items()}), flush=True)


def verify():
    check_inputs(); rows = saved_rows()
    if (OUT/'verification.json').exists():
        raise FileExistsError('Separate verification already completed; no implicit duplicate certificates')
    ledger = [json.loads(s) for s in (OUT/'call_ledger.jsonl').read_text().splitlines()]
    assert len(ledger) == 144
    assert [r['call'] for r in ledger if r['status']=='started'] == list(range(1,73))
    assert [r['call'] for r in ledger if r['status']=='completed'] == list(range(1,73))
    cache=Cache(1); certificates=[];tick=time.perf_counter()
    for case in range(24):
        scene = read(ROOT/'outputs/scenes'/f'dev_{case:02}.json');item=cache.prepare(scene)
        for r in [r for r in rows if r['case']==case]:
            for h in ('H0','H1','fixed3'):
                actual = metrics(item, r[h]['route'])
                assert actual['candidate_ids'] == r[h]['candidate_ids']
                for key in ('J','L','K'):assert abs(actual[key]-r[h][key])<1e-7
            assert abs(r['delta_H_J']-r['delta_H_L']-r['delta_H_penalty'])<1e-7
            path=ROOT.parent/r['plan_path']; assert digest(path)==r['plan_sha256']
            plan=read(path); cert=validate(scene,plan,item.base.sparse)
            assert cert['valid_model_plan'], cert
            assert [0]+[j+1 for j in plan['visit_order_candidate_ids']]+[0] == r['H1']['route']
            assert abs(plan['objective_length_plus_stops_m']-r['H1']['J'])<1e-7
            certificates.append(dict(file=r['plan_path'],sha256=r['plan_sha256'],valid=True))
    check_inputs()
    write(OUT/'verification.json', dict(all_passed=True, separate_geometry_checks=72,
        numerical_route_checks=216, new_postprocess_calls=0, new_neural_inferences=0,
        optimizer_updates=0, ledger_started=72, ledger_completed=72,
        elapsed_s=time.perf_counter()-tick, certificates=certificates,
        scope='Saved-route numeric accounting and independent scalar geometry/assignment validator; no H0/H1 rerun; no IK'))
    resources=read(OUT/'resources.json');resources['separate_saved_plan_geometry_checks']=72
    resources['geometry_certificates_total']=144;write(OUT/'resources.json',resources)
    print('PASS: 72 separate geometry checks, 216 numerical route checks; no new planner calls', flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase', choices=['register','run','analyze','verify'])
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        globals()[args.phase]()
