"""Frozen post-V4 descriptive supplement: legacy MLP vs Detour at 400 ms.

Reuses the unmodified original run_once/runtime implementation. No training,
checkpoint selection, threshold changes, or hard-deadline reinterpretation.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

sys.dont_write_bytecode = True


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.source_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists():
        raise ValueError('refusing to overwrite an existing frozen supplement')
    (out/'plans').mkdir(exist_ok=True)
    sys.path.insert(0, str(root))
    import numpy as np
    import scipy
    from rc3_v4.evaluate import run_once, source_paths
    from rc3_v4.legacy_policy import load_policy
    from rc3_fast.solver import prepare
    from rc3_fast.scenes import make_complex_scene
    from rc3_baseline.validator import validate
    from rc3_improve.evaluate import safe_json

    original_manifest = read(root/'rc3_v4/results/manifest.json')
    all_cases = read(root/'rc3_v4/data/test_cases.json')
    cases = [(index, case) for index, case in enumerate(all_cases)
             if int(case['nx'])*int(case['ny']) == 100]
    if len(all_cases) != 39 or len(cases) != 18 or len({c['seed'] for _, c in cases}) != 18:
        raise ValueError('unexpected frozen test set')
    model = root/'rc3_v3/deployment/policy.npz'
    if sha(model) != original_manifest['model_sha256'][r'rc3_v3\deployment\policy.npz']:
        raise ValueError('frozen v3 NPZ model hash mismatch')
    specs = [dict(id='detour_wall400', method='detour', policy=None, kind='wall',
                  horizon=256, seconds=.4, repeats=3),
             dict(id='v3_mlp_wall400', method='learned', policy='v3_mlp', kind='wall',
                  horizon=256, seconds=.4, repeats=3)]
    source_plan_paths = []
    for index, case in cases:
        path = root/'rc3_v4/results/plans'/f'scene_plan_{index:03d}_{case["seed"]}.json'
        if not path.is_file():
            raise ValueError(f'missing original scene: {path}')
        source_plan_paths.append(path)
    frozen_paths = [root/p for p in source_paths()] + [
        model, root/'rc3_v4/data/test_cases.json', root/'rc3_v4/results/manifest.json',
        *source_plan_paths]
    hashes = {str(path.relative_to(root)): sha(path) for path in frozen_paths}
    policy = load_policy(model)  # Original NPZ reader uses allow_pickle=False.
    cpu_model = ''
    if Path('/proc/cpuinfo').exists():
        cpu_model = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                          if line.startswith('model name')), '')
    cpu_quota = Path('/sys/fs/cgroup/cpu.max').read_text().strip() if Path('/sys/fs/cgroup/cpu.max').exists() else None
    manifest = dict(status='running', frozen_utc=datetime.now(timezone.utc).isoformat(),
        study='post_freeze_descriptive_legacy_mlp_vs_detour_400ms',
        inference_scope='Descriptive supplement after observing the original V4 results. No retrospective change to original adoption gates.',
        source_root=str(root), original_platform=original_manifest['platform'],
        platform=platform.platform(), python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        cpu_model=cpu_model, cpu_quota=cpu_quota,
        model_sha256=sha(model), script_sha256=sha(Path(__file__)), frozen_source_sha256=hashes,
        cases=[dict(original_case_index=i, **c) for i, c in cases],
        configurations=specs, repeats=3, planned_runs=108,
        timing_scope=original_manifest['timing_scope'],
        shared_preprocessing='Prepare each exact saved original scene once; shared across randomized runs; excluded from planning timer.',
        acceptance='Unmodified original evaluate.run_once: runtime certification/fallback + outside-timer independent recheck and Fixed3 bound. Also compare with the separately revalidated original saved Fixed3.',
        runtime='Unmodified rc3_v4.evaluate.run_once -> rc3_v4.runtime.run; original 400 ms soft budget, horizon 256.',
        warmup=dict(nx=2, ny=2, seed=130041, mode='regular', horizon=1, seconds=None),
        schedule='For each original scene: np.random.default_rng(scene.seed + 753911).permutation(6) of two methods x three repeats.',
        aggregation='Mean cost and median planning time within scene; then mean paired cost reduction and median scene planning time. Preserve every repeat; no best-repeat selection.',
        ci=dict(method='percentile paired scene bootstrap', samples=20000, seed=731203,
                sorting=['seed','nx','ny','mode'], training_seed_uncertainty=False,
                repeated_runs_treated_as_independent=False, multiple_comparison_adjustment=False),
        no_training_or_tuning=True, original_adoption_gate_unchanged=True)
    write(out/'manifest.json', manifest)
    warm = prepare(make_complex_scene(2, 2, 130041, 'regular'))
    warm_records = []
    for method, policy_arg in [('learned', policy), ('detour', None)]:
        _, result = run_once(warm, dict(method=method, horizon=1, seconds=None), policy_arg)
        warm_records.append(dict(method=method, result=result))
        if result['status'] != 'complete':
            raise ValueError('original development warmup failed')
    write(out/'warmup.json', safe_json(warm_records))
    start = time.perf_counter()
    repeats, scene_rows, validations = [], [], []
    for counter, ((index, case), source_path) in enumerate(zip(cases, source_plan_paths)):
        original = read(source_path)
        meta = dict(original['metadata'])
        if any(meta[k] != case[k] for k in ('seed','nx','ny','mode')):
            raise ValueError('original case metadata mismatch')
        scene = original['scene']
        tick = time.perf_counter()
        problem = prepare(scene)
        preprocessing_s = time.perf_counter()-tick
        fixed_labels = [label for label, record in original['runs'].items()
                        if record['config']['id'] == 'fixed3']
        fixed_plan = original['plans'][sorted(fixed_labels)[0]]
        fixed_check = validate(scene, fixed_plan, problem.sparse)
        if not fixed_check['valid_model_plan'] or not fixed_check['all_tasks_served']:
            raise ValueError('original Fixed3 failed current independent revalidation')
        fixed_cost = fixed_plan['objective_length_plus_stops_m']
        payload = dict(status='running', metadata=meta, source_plan=str(source_path.relative_to(root)),
                       source_plan_sha256=sha(source_path), scene=scene, feasibility=problem.sparse,
                       preprocessing_s=preprocessing_s, fixed3_reference_plan=fixed_plan,
                       fixed3_reference_validation=fixed_check, plans={}, runs={}, execution_order=[])
        schedule = [(spec, repeat) for spec in specs for repeat in range(3)]
        permutation = np.random.default_rng(meta['seed']+753911).permutation(len(schedule))
        for order, pos in enumerate(permutation):
            spec, repeat = schedule[int(pos)]
            label = f'{spec["id"]}_rep{repeat}'
            plan, record = run_once(problem, spec, policy if spec['policy'] else None)
            record.update(config=spec, repeat=repeat, execution_order=order)
            extra_bound = (record['status'] == 'complete' and
                           record['objective_m'] <= fixed_cost+1e-7 and
                           abs(record['reference_objective_m']-fixed_cost) <= 1e-7)
            record['bound_against_revalidated_saved_fixed3'] = bool(extra_bound)
            payload['execution_order'].append(label)
            payload['runs'][label] = record
            if plan is not None:
                payload['plans'][label] = plan
            validations.append(dict(seed=meta['seed'], label=label, status=record['status'],
                valid=record.get('independent_validation', {}).get('valid_model_plan', False),
                all_tasks_served=record.get('independent_validation', {}).get('all_tasks_served', False),
                fixed3_bound=bool(extra_bound)))
            repeats.append(dict(original_case_index=index, seed=meta['seed'], nx=meta['nx'], ny=meta['ny'],
                mode=meta['mode'], tasks=len(scene['tasks']), method=spec['id'], repeat=repeat,
                execution_order=order, status=record['status'], objective_m=record.get('objective_m'),
                reference_m=fixed_cost, planning_s=record['planning_s'], preprocessing_s=preprocessing_s,
                dp_calls=record.get('dp_calls'), accepted=record.get('accepted'),
                fallback_used=record.get('fallback_used'), stop_reason=record.get('stop_reason'),
                overrun_s=record.get('overrun_s'), deadline_exceeded=record.get('deadline_exceeded'),
                valid=validations[-1]['valid'], all_tasks_served=validations[-1]['all_tasks_served'],
                fixed3_bound=bool(extra_bound)))
        payload['status'] = 'complete' if all(v['status']=='complete' and v['bound_against_revalidated_saved_fixed3']
                                                for v in payload['runs'].values()) else 'partial_failure'
        write(out/'plans'/f'scene_plan_{index:03d}_{meta["seed"]}.json', safe_json(payload))
        for spec in specs:
            rows = [r for r in repeats if r['seed']==meta['seed'] and r['method']==spec['id']]
            good = all(r['status']=='complete' and r['valid'] and r['fixed3_bound'] for r in rows)
            scene_rows.append(dict(original_case_index=index, seed=meta['seed'], nx=meta['nx'], ny=meta['ny'],
                mode=meta['mode'], method=spec['id'], valid=good, reference_m=fixed_cost,
                mean_objective_m=float(np.mean([r['objective_m'] for r in rows])) if good else None,
                median_planning_ms=float(np.median([r['planning_s'] for r in rows]))*1000,
                mean_dp_calls=float(np.mean([r['dp_calls'] for r in rows])) if good else None,
                preprocessing_ms=preprocessing_s*1000,
                min_planning_ms=min(r['planning_s'] for r in rows)*1000,
                max_planning_ms=max(r['planning_s'] for r in rows)*1000,
                min_calls=min(r['dp_calls'] for r in rows) if good else None,
                max_calls=max(r['dp_calls'] for r in rows) if good else None,
                min_objective_m=min(r['objective_m'] for r in rows) if good else None,
                max_objective_m=max(r['objective_m'] for r in rows) if good else None))
        write(out/'progress.json', dict(completed_scenes=counter+1, total_scenes=18,
              completed_runs=len(repeats), elapsed_s=time.perf_counter()-start))
        print(f'completed {counter+1}/18 scenes, {len(repeats)}/108 runs, {time.perf_counter()-start:.1f}s', flush=True)
    write_csv(out/'repeat_metrics.csv', repeats)
    write_csv(out/'scene_metrics.csv', scene_rows)
    write(out/'validation.json', validations)
    changed = [path for path, expected in hashes.items() if sha(root/path) != expected]
    all_passed = all(v['status']=='complete' and v['valid'] and v['all_tasks_served'] and v['fixed3_bound']
                     for v in validations) and not changed
    summary = dict(status='complete' if all_passed else 'audit_failure', scenes=18, runs=len(repeats),
        valid_runs=sum(v['valid'] and v['all_tasks_served'] for v in validations),
        fixed3_bounds=sum(v['fixed3_bound'] for v in validations), changed_frozen_files=changed,
        all_passed=all_passed, inference_scope=manifest['inference_scope'],
        bootstrap=manifest['ci'], platform=manifest['platform'], methods={})
    if all_passed:
        for spec in specs:
            ss = [s for s in scene_rows if s['method']==spec['id']]
            rr = [r for r in repeats if r['method']==spec['id']]
            summary['methods'][spec['id']] = dict(
                scenes=len(ss), runs=len(rr), mean_objective_m=float(np.mean([s['mean_objective_m'] for s in ss])),
                mean_reduction_vs_fixed3_pct=float(np.mean([100*(s['reference_m']-s['mean_objective_m'])/s['reference_m'] for s in ss])),
                median_scene_planning_ms=float(np.median([s['median_planning_ms'] for s in ss])),
                p95_scene_planning_ms=float(np.quantile([s['median_planning_ms'] for s in ss], .95)),
                p95_all_run_planning_ms=float(np.quantile([r['planning_s']*1000 for r in rr], .95)),
                max_run_planning_ms=max(r['planning_s']*1000 for r in rr),
                mean_dp_calls=float(np.mean([r['dp_calls'] for r in rr])),
                median_dp_calls=float(np.median([r['dp_calls'] for r in rr])),
                deadline_exceeded_runs=sum(r['deadline_exceeded'] for r in rr),
                max_overrun_ms=max(r['overrun_s']*1000 for r in rr),
                mean_overrun_ms=float(np.mean([r['overrun_s']*1000 for r in rr])),
                fallback_runs=sum(r['fallback_used'] for r in rr),
                stop_reasons={key:sum(r['stop_reason']==key for r in rr) for key in sorted({r['stop_reason'] for r in rr})})
        def key(row):
            return row['seed'], row['nx'], row['ny'], row['mode']
        left = {key(s):s for s in scene_rows if s['method']=='v3_mlp_wall400'}
        right = {key(s):s for s in scene_rows if s['method']=='detour_wall400'}
        pairs=[]
        for k in sorted(left):
            a,b=left[k],right[k]
            pairs.append(dict(seed=k[0], nx=k[1], ny=k[2], mode=k[3],
                mlp_objective_m=a['mean_objective_m'], detour_objective_m=b['mean_objective_m'],
                reduction_pct=100*(b['mean_objective_m']-a['mean_objective_m'])/b['mean_objective_m'],
                mlp_median_ms=a['median_planning_ms'], detour_median_ms=b['median_planning_ms'],
                paired_time_ratio=a['median_planning_ms']/b['median_planning_ms'],
                mlp_mean_calls=a['mean_dp_calls'], detour_mean_calls=b['mean_dp_calls']))
        values=np.array([p['reduction_pct'] for p in pairs])
        indices=np.random.default_rng(731203).integers(0,len(values),size=(20000,len(values)))
        interval=np.quantile(values[indices].mean(axis=1),[.025,.975]).tolist()
        summary['paired_mlp_vs_detour']=dict(scenes=len(values),mean_reduction_pct=float(values.mean()),
            ci95_pct=interval,wins=int((values>1e-9).sum()),ties=int((np.abs(values)<=1e-9).sum()),
            losses=int((values < -1e-9).sum()),
            median_paired_time_ratio=float(np.median([p['paired_time_ratio'] for p in pairs])),
            ratio_of_median_times=summary['methods']['v3_mlp_wall400']['median_scene_planning_ms']/summary['methods']['detour_wall400']['median_scene_planning_ms'])
        write_csv(out/'paired_scene_metrics.csv',pairs)
    summary['elapsed_s']=time.perf_counter()-start
    write(out/'summary.json',summary)
    manifest.update(status=summary['status'], completed_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_s=summary['elapsed_s'], attempted_runs=len(repeats), saved_plans=sum(v['status']=='complete' for v in validations),
        changed_frozen_files=changed)
    write(out/'manifest.json',manifest)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return 0 if all_passed else 2


if __name__=='__main__':
    raise SystemExit(main())
