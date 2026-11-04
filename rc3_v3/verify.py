"""Independently revalidate v3 plans, Fixed3 bounds, portable policy and failures."""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

import numpy as np
from rc3_baseline.baseline import write_json,write_csv,schedule_rows,SCHEDULE_COLUMNS
from rc3_baseline.validator import validate_file
from rc3_fast.scenes import make_complex_scene
from rc3_fast.solver import prepare
from rc3_improve.evaluate import safe_json
from .environment import FEATURE_NAMES,FEATURE_SCHEMA
from .model import load_policy
from .plan import Planner,DEFAULT_CHECKPOINT
from .runtime import run


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def plan_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--model',type=Path,default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output',type=Path,default=Path('rc3_v3/verification'))
    args=parser.parse_args(argv)
    manifest=read(args.results/'manifest.json')
    if manifest.get('status')!='complete':raise ValueError('wait for completed frozen test evaluation')
    args.output.mkdir(parents=True,exist_ok=True);started=time.perf_counter();errors=[]
    model_hash=sha(args.model)
    if model_hash!=manifest['model_sha256']:errors.append('deployment model differs from frozen evaluated model')
    if sha(Path(manifest['wall_config_path']))!=manifest['wall_config_sha256']:
        errors.append('frozen soft wall budget changed')
    data_checks=[]
    for name,expected in manifest['data_sha256'].items():
        same=sha(Path(name))==expected;data_checks.append(dict(path=name,unchanged=same))
        if not same:errors.append(f'changed evaluation data: {name}')
    source_checks=[dict(path=name,unchanged=Path(name).exists() and sha(Path(name))==expected)
                   for name,expected in manifest['source_sha256'].items()]
    if args.model.suffix.lower()!='.npz':raise ValueError('portable deployment verification expects the final NPZ')
    with np.load(args.model,allow_pickle=False) as archive:
        arrays={k:archive[k].copy() for k in archive.files}
    metadata=json.loads(str(arrays['metadata']))
    phase_ok=(metadata.get('phase')=='gain_supervised' and metadata.get('schema')==FEATURE_SCHEMA
              and metadata.get('feature_names')==FEATURE_NAMES)
    default_ok=DEFAULT_CHECKPOINT.exists() and sha(DEFAULT_CHECKPOINT)==model_hash
    if not phase_ok:errors.append('final model phase/schema mismatch')
    if not default_ok:errors.append('default deployment path does not contain the evaluated frozen model')
    wrong_phase_rejected=False
    with tempfile.TemporaryDirectory() as folder:
        bad=Path(folder)/'wrong_phase.npz'
        wrong=dict(metadata,phase='ppo')
        np.savez_compressed(bad,**(arrays|{'metadata':np.array(json.dumps(wrong))}))
        try:load_policy(bad)
        except ValueError:wrong_phase_rejected=True
    if not wrong_phase_rejected:errors.append('a checkpoint with the wrong training phase was accepted')
    policy=load_policy(args.model)

    files=sorted((args.results/'plans').glob('scene_plan_*.json'))
    if len(files)!=manifest['cases']:errors.append('saved scene count does not match denominator')
    checks=[];bounds=[];cost_lookup={};failed_files=[]
    for index,path in enumerate(files):
        data=read(path)
        if not data.get('plans'):
            failed_files.append(dict(file=path.name,status=data.get('status'),pending_task_ids=data.get('pending_task_ids')))
            continue
        current=validate_file(path);checks.extend(current)
        problem=prepare(data['scene'])
        reference,info=run(problem,method='fixed3')
        reference_cost=reference['objective_length_plus_stops_m']
        for check in current:
            label=check['method'];plan=data['plans'].get(label)
            if not check.get('valid_model_plan') or not check.get('all_tasks_served') or plan is None:
                errors.append(f'independent validation failed: {path.name} {label}');continue
            independent_cost=check['independent_polyline_length_m']+problem.penalty*len(plan['selected_candidate_ids'])
            good=independent_cost<=reference_cost+1e-7
            bounds.append(dict(file=path.name,plan=label,objective_m=independent_cost,reference_m=reference_cost,nonworsening=good))
            if not good:errors.append(f'Fixed3 bound violated: {path.name} {label}')
            spec_id=label.rsplit('_rep',1)[0]
            scene=data['scene'];key=(scene['seed'],scene['nx'],scene['ny'],scene.get('complexity','custom'),spec_id)
            cost_lookup.setdefault(key,[]).append(independent_cost)
        if (index+1)%9==0:print(f'independently verified {index+1}/{len(files)} scenes',flush=True)
    if len(checks)!=manifest['saved_repeated_plans']:errors.append('saved repeated-plan count mismatch')
    rows=read(args.results/'metrics.json')
    if len(rows)!=manifest['method_budget_cases']:errors.append('method-budget-case denominator mismatch')
    metric_checks=[]
    for row in rows:
        if row['status']!='complete':continue
        key=(row['seed'],row['nx'],row['ny'],row['mode'],row['config'])
        values=cost_lookup.get(key,[])
        matched=(len(values)==manifest['repeats'] and abs(float(np.mean(values))-row['objective_m'])<=1e-7)
        metric_checks.append(dict(seed=row['seed'],config=row['config'],matched_independent_repeat_mean=matched))
        if not matched:errors.append(f'reported mean objective not reproduced from saved plans: {key}')
    write_json(args.output/'independent_plans.json',dict(checks=checks,reference_bounds=bounds,
               repeated_mean_checks=metric_checks,failed_scene_files=failed_files))

    forbidden=set()
    for name in manifest['data_sha256']:
        if name.endswith('_cases.json'):
            forbidden.update(case['seed'] for case in read(Path(name)) if isinstance(case,dict) and 'seed' in case)
    seed=771901
    while seed in forbidden:seed+=1
    scene=make_complex_scene(10,10,seed,'blocked')
    # A deterministic call budget, not different wall-time limits, establishes
    # reproducibility and cache correctness.
    planner=Planner(args.model,'learned',horizon=8,time_limit_s=None)
    cold=planner.plan(scene);warm=planner.plan(copy.deepcopy(scene))
    reloaded=Planner(args.model,'learned',horizon=8,time_limit_s=None).plan(copy.deepcopy(scene))
    complete=all(result.get('status')=='complete' for result in (cold,warm,reloaded))
    same=complete and cold['plans']==warm['plans']==reloaded['plans']
    cached=complete and not cold['timing']['cache_hit'] and warm['timing']['cache_hit']
    if not same or not cached:errors.append('cold/warm/reloaded full-plan equivalence or cache check failed')
    path=args.output/'scene_plan_portable_fresh.json';write_json(path,safe_json(cold))
    if complete:
        write_csv(path.with_suffix('.csv'),list(schedule_rows(cold)),SCHEDULE_COLUMNS)
        fresh_checks=validate_file(path)
        if not all(c['valid_model_plan'] and c['all_tasks_served'] for c in fresh_checks):errors.append('portable saved output invalid')
    else:fresh_checks=[]
    default_result=Planner().plan(copy.deepcopy(scene))
    default_fixed3=(default_result.get('status')=='complete' and set(default_result['plans'])=={'fixed3'}
                    and default_result['diagnostics']['dp_calls']==0)
    if not default_fixed3:errors.append('default planner did not use the declared Fixed3 mode')
    empty=copy.deepcopy(scene);empty['tasks']=[]
    empty_result=planner.plan(empty)
    empty_ok=(empty_result.get('status')=='complete' and empty_result['diagnostics']['objective_m']==0
              and next(iter(empty_result['plans'].values()))['validation']['valid_model_plan'])
    if not empty_ok:errors.append('empty task set handling failed')
    failure_records=[]
    for name in ('unreachable_task','blocked_depot','variable_lift','invalid_task_ids'):
        bad=copy.deepcopy(scene)
        if name=='unreachable_task':bad['tasks'][0]['z']=5.
        elif name=='blocked_depot':
            x,y=bad['start'];bad['obstacles'].append(dict(xmin=x-.1,xmax=x+.1,ymin=y-.1,ymax=y+.1))
        elif name=='variable_lift':bad['candidates'][0]['h']+=.1
        else:bad['tasks'][0]['id']=99999
        expected=[t['id'] for t in bad['tasks']]
        outcome=planner.plan(bad)
        passed=(outcome.get('status')=='failed' and not outcome.get('plans')
                and outcome.get('pending_task_ids')==expected and outcome.get('task_ids')==expected)
        failure_records.append(dict(case=name,passed=passed,code=outcome.get('code'),detail=outcome.get('detail'),
                                    tasks_preserved=len(outcome.get('pending_task_ids',[]))))
        write_json(args.output/f'failure_{name}.json',safe_json(outcome))
        if not passed:errors.append(f'failure case did not retain every task: {name}')
    deployment_no_torch=('torch' not in sys.modules)

    # The runtime suite also injects model failure, invalid actions and final
    # certification failure, so no additional costly duplicate test loop is used.
    package=Path(__file__).resolve().parent;capture=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(package),pattern='test_*.py',top_level_dir=str(package.parent))
    tested=unittest.TextTestRunner(stream=capture,verbosity=2).run(suite)
    (args.output/'unit_tests.log').write_text(capture.getvalue(),encoding='utf-8')
    if not tested.wasSuccessful():errors.append('unit or injected runtime failure tests failed')
    report=dict(status='complete',all_passed=not errors,errors=errors,results=str(args.results),
                model=str(args.model),model_sha256=model_hash,model_phase=metadata.get('phase'),
                model_schema=metadata.get('schema'),phase_schema_valid=phase_ok,wrong_phase_rejected=wrong_phase_rejected,
                default_checkpoint_matches_evaluated=default_ok,default_planner_is_fixed3=default_fixed3,
                source_checks=source_checks,data_checks=data_checks,saved_scenes=len(files),
                independently_validated_repeated_plans=len(checks),reference_bound_checks=len(bounds),
                independently_reconstructed_mean_costs=len(metric_checks),method_budget_cases=len(rows),
                failed_method_budget_cases=sum(row['status']!='complete' for row in rows),
                portable_deployment=dict(seed=seed,budget_calls=8,wall_limit=None,full_plan_reload_and_cache_equal=same,
                    cache_hits_correct=cached,cold_hash=plan_hash(cold.get('plans',{})),warm_hash=plan_hash(warm.get('plans',{})),
                    reload_hash=plan_hash(reloaded.get('plans',{})),cold_timing=cold.get('timing'),warm_timing=warm.get('timing'),
                    numpy_deployment_without_torch=deployment_no_torch,empty_tasks_valid=empty_ok,independent_checks=fresh_checks),
                failure_checks=failure_records,
                unit_tests=dict(run=tested.testsRun,passed=tested.wasSuccessful(),failures=len(tested.failures),errors=len(tested.errors),
                    skipped=[dict(test=str(t),reason=reason) for t,reason in tested.skipped]),
                elapsed_s=time.perf_counter()-started,
                scope='Independent fixed-height arm proxy and navigation-graph plans only; wall deadlines are soft; no IK or hardware claim.')
    write_json(args.output/'verification.json',safe_json(report))
    print(json.dumps(dict(all_passed=report['all_passed'],plans=len(checks),bound_checks=len(bounds),
                          repeated_mean_checks=len(metric_checks),unit_tests=tested.testsRun,elapsed_s=report['elapsed_s'])),flush=True)
    if errors:raise SystemExit(2)


if __name__=='__main__':main()
