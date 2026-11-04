"""Validation-only fixed-call controls; exhaustive oracle is offline analysis.

Uses only validation scenes/routes from the local compact training cache. No test
manifest, test cost or test label is opened. Timings are diagnostic and must not
be used as a deployment speed comparison, especially during parallel training.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import pickle
import time

import numpy as np

from rc3_baseline.baseline import write_json, write_csv
from rc3_fast.solver import prepare, route_objective, export_plan
from .data import TeacherLabels
from .environment import Env, FEATURE_NAMES, FEATURE_SCHEMA


METHODS=('random','detour','oracle_greedy8')


def certify(problem, route, method, expected):
    plan=export_plan(problem,route,method)
    actual=float(plan['objective_length_plus_stops_m'])
    validation=plan.get('validation',{})
    if not validation.get('valid_model_plan') or not validation.get('all_tasks_served'):
        raise AssertionError('independent exported plan validation failed')
    if not math.isfinite(actual) or abs(actual-expected)>1e-7:
        raise AssertionError('canonical and independently exported objective differ')
    return actual,validation


def run_control(problem, initial_route, reference_cost, method, horizon=8):
    if method not in METHODS:
        raise ValueError('unknown validation control')
    if horizon!=8:
        raise ValueError('this protocol is fixed at eight repair attempts')
    env=Env(problem,horizon=horizon,initial_route=initial_route)
    if abs(env.reference_cost-reference_cost)>1e-7:
        raise AssertionError('stored Fixed3 reference differs from reconstructed route')
    rng=np.random.default_rng(int(problem.scene.get('seed',0))+82991)
    teacher_stats=defaultdict(int)
    labeller=TeacherLabels(problem,reference_cost,teacher_stats,FEATURE_NAMES) if method=='oracle_greedy8' else None
    required_column=FEATURE_NAMES.index('required_tasks')
    detour_column=FEATURE_NAMES.index('detour_excess')
    selected_dp_calls=0
    trace=[]
    stop_reason='horizon'
    started=time.perf_counter()
    while env.steps<horizon:
        features,mask=env.observe()
        if env.done or not mask.any():
            stop_reason='no_legal_action';break
        predicted_gain=None
        if method=='oracle_greedy8':
            gains=labeller.evaluate(env,features,mask)
            action=int(np.where(mask,gains,-np.inf).argmax())
            predicted_gain=float(gains[action])
            if predicted_gain<=1e-8:
                stop_reason='oracle_all_legal_zero';break
        elif method=='detour':
            action=int(np.where(mask,features[:,detour_column],-np.inf).argmax())
        else:
            action=int(rng.choice(np.flatnonzero(mask)))
        selected_dp_calls+=int(features[action,required_column]>0)
        reward,_,info=env.step(action)
        if predicted_gain is not None and abs(reward-predicted_gain)>1e-5:
            raise AssertionError('oracle label and actual transition gain disagree')
        trace.append(dict(step=int(env.steps-1),action=action,gain_pct=float(reward),
                          oracle_label_pct=predicted_gain,accepted=bool(info['accepted']),
                          objective_m=float(route_objective(problem,env.route))))
    expected=float(route_objective(problem,env.route))
    actual,validation=certify(problem,env.route,f'validation_{method}',expected)
    gain=100*(reference_cost-actual)/max(reference_cost,1e-8)
    if actual>reference_cost+1e-8 or abs(sum(row['gain_pct'] for row in trace)-gain)>1e-7:
        raise AssertionError('incremental rewards and final independently exported gain differ')
    # Oracle labels evaluate every legal action, then Env.step executes the
    # selected repair again. Both workloads are explicitly counted.
    offline_calls=int(teacher_stats['teacher_repair_calls'])
    offline_dp=int(teacher_stats['teacher_dp_calls'])
    return dict(method=method,reference_objective_m=float(reference_cost),objective_m=actual,
        improvement_pct=float(gain),attempts=int(env.steps),selected_dp_calls=selected_dp_calls,
        accepted=sum(row['accepted'] for row in trace),stop_reason=stop_reason,
        offline_label_repair_calls=offline_calls,offline_label_dp_calls=offline_dp,
        offline_label_cache_hits=int(teacher_stats['teacher_cache_hits']),
        total_repair_calls=int(env.steps)+offline_calls,total_dp_calls=selected_dp_calls+offline_dp,
        offline_label_repair_s=float(teacher_stats['teacher_repair_s']),
        offline_label_dp_s=float(teacher_stats['teacher_dp_s']),
        diagnostic_elapsed_s=time.perf_counter()-started,
        valid_model_plan=bool(validation['valid_model_plan']),
        all_tasks_served=bool(validation['all_tasks_served']),
        export_cost_matches=True,cost_nonincreasing=True,route=list(map(int,env.route)),trace=trace)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True,help='Local v3 data directory containing data.pkl')
    parser.add_argument('--output',type=Path,required=True,help='New validation-controls JSON path')
    args=parser.parse_args(argv)
    if args.output.exists():parser.error('use a new output path')
    data_path=args.data/'data.pkl'
    with data_path.open('rb') as handle:
        data=pickle.load(handle)  # Only the cache generated locally by this project.
    if data.get('schema')!=FEATURE_SCHEMA or data.get('feature_names')!=FEATURE_NAMES:
        raise ValueError('validation cache schema mismatch')
    scenes=data['scenes']['validation']
    routes=data['initial_routes']['validation']
    references=data['reference_costs']['validation']
    cases=data['case_metadata']['validation']
    # Training fields are not inspected or evaluated. There is no test data in
    # this compact pickle, and this script never opens a test manifest.
    del data
    if not len(scenes)==len(routes)==len(references)==len(cases)==18:
        raise ValueError('frozen validation protocol requires exactly 18 scenes')
    if any(int(case['nx'])*int(case['ny'])!=100 for case in cases):
        raise ValueError('validation protocol requires only 100-table scenes')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();rows=[]
    progress_path=args.output.with_name(args.output.stem+'.progress.json')
    for index,(scene,route,reference,case) in enumerate(zip(scenes,routes,references,cases)):
        problem=prepare(scene)
        reference_actual,reference_validation=certify(problem,route,'validation_fixed3',float(reference))
        record=dict(scene_id=index,seed=int(case['seed']),mode=case['mode'],tables=100,
                    reference_objective_m=reference_actual,reference_valid_model_plan=True,
                    methods={})
        for method in METHODS:
            record['methods'][method]=run_control(problem,route,reference_actual,method)
        rows.append(record)
        write_json(progress_path,dict(status='running',completed=len(rows),total=18,
            elapsed_s=time.perf_counter()-started,cases=rows))
        gains=' '.join(f'{m}={record["methods"][m]["improvement_pct"]:.4f}%' for m in METHODS)
        print(f'validation {index+1}/18 {case["mode"]}: {gains}',flush=True)
        del problem
    summary={}
    for method in METHODS:
        records=[row['methods'][method] for row in rows]
        summary[method]=dict(mean_improvement_pct=float(np.mean([r['improvement_pct'] for r in records])),
            mean_attempts=float(np.mean([r['attempts'] for r in records])),
            mean_selected_dp_calls=float(np.mean([r['selected_dp_calls'] for r in records])),
            improved_cases=sum(r['improvement_pct']>1e-8 for r in records),
            total_offline_label_repair_calls=sum(r['offline_label_repair_calls'] for r in records),
            total_offline_label_dp_calls=sum(r['offline_label_dp_calls'] for r in records),
            total_repair_calls=sum(r['total_repair_calls'] for r in records),
            total_dp_calls=sum(r['total_dp_calls'] for r in records),
            all_export_checks_passed=all(r['export_cost_matches'] and r['valid_model_plan'] and r['all_tasks_served'] for r in records))
    result=dict(status='complete',schema=FEATURE_SCHEMA,validation_instances=18,horizon=8,
        reference='Stored canonical Fixed3 (0,1,7) route shared by all methods',
        timing_scope='Diagnostic only; may overlap training. Oracle exhaustive label generation is offline extra work, never model deployment latency.',
        oracle_scope='Greedy maximum one-step gain, stopping when all legal gains are zero; not a global or eight-step optimal upper bound.',
        dp_count_scope='Only positive-required-task repair calls count as DP. Oracle label calls and actual selected Env.step calls are both counted.',
        label_time_scope='Teacher repair/canonicalization/objective; not isolated inner DP timing.',
        test_accessed=False,elapsed_s=time.perf_counter()-started,
        data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),summary=summary,cases=rows)
    write_json(args.output,result)
    flat=[]
    for row in rows:
        for method,values in row['methods'].items():
            flat.append(dict(scene_id=row['scene_id'],seed=row['seed'],mode=row['mode'],
                             **{k:v for k,v in values.items() if k not in ('trace','route')}))
    write_csv(args.output.with_suffix('.csv'),flat)
    write_json(progress_path,dict(status='complete',completed=18,total=18,summary=summary))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
