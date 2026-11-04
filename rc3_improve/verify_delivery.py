"""Independent saved-plan, frozen-weight, deployment, cache and failure checks."""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import time
import unittest

from rc3_baseline.baseline import write_json, write_csv, schedule_rows, SCHEDULE_COLUMNS
from rc3_baseline.validator import validate_file
from rc3_fast.scenes import make_complex_scene
from rc3_fast.solver import prepare, solve_route, export_plan
from .evaluate import safe_json
from .plan import RepairPlanner


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def plan_hash(plans):return hashlib.sha256(json.dumps(plans,sort_keys=True,allow_nan=False).encode()).hexdigest()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('rc3_improve/verification_v2'))
    args=parser.parse_args(argv)
    training=read(args.run/'training_summary.json')
    manifest=read(args.results/'manifest.json')
    if training.get('status')!='complete' or manifest.get('status')!='complete':
        raise ValueError('verification must wait for completed training and frozen evaluation')
    args.output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    errors=[]
    frozen_weights=[]
    for name,expected in manifest['frozen_models'].items():
        path=Path(name)
        actual=digest(path)
        frozen_weights.append(dict(path=str(path),expected=expected,actual=actual,unchanged=actual==expected))
        if actual!=expected:errors.append(f'frozen model changed: {path}')
    checkpoint_hash=digest(args.checkpoint)
    if checkpoint_hash not in manifest['frozen_models'].values():
        errors.append('deployment checkpoint is not one of the evaluated frozen checkpoints')
    source_checks=[]
    for name,expected in manifest.get('source_sha256',{}).items():
        path=Path(name)
        actual=digest(path) if path.exists() else None
        source_checks.append(dict(path=name,unchanged=actual==expected,expected=expected,actual=actual))
    data_checks=[]
    for name,expected in manifest.get('data_sha256',{}).items():
        path=Path(name);actual=digest(path)
        data_checks.append(dict(path=name,unchanged=actual==expected))
        if actual!=expected:errors.append(f'evaluation data changed: {name}')

    plan_files=sorted((args.results/'plans').glob('scene_plan_*.json'))
    raw_checks=[];failed_scene_files=[];reference_checks=[]
    if len(plan_files)!=manifest['cases']:errors.append('saved scene-file count differs from evaluation denominator')
    for index,path in enumerate(plan_files):
        data=read(path)
        if not data.get('plans'):
            failed_scene_files.append(dict(file=path.name,status=data.get('status'),pending_task_ids=data.get('pending_task_ids')))
            continue
        # Uses the independent validator on the file, never its stored valid flag.
        checks=validate_file(path)
        raw_checks.extend(checks)
        if any(not c['valid_model_plan'] or not c['all_tasks_served'] for c in checks):
            errors.append(f'independent saved-plan validation failed: {path.name}')
        # Recreate the named Fixed1 reference once per scene; never trust a saved
        # reference cost to establish the nonworsening certificate.
        problem=prepare(data['scene'])
        reference=export_plan(problem,solve_route(problem,1),'verification_fixed1')
        reference_cost=reference['objective_length_plus_stops_m']
        for method,plan in data['plans'].items():
            check=next(c for c in checks if c['method']==method)
            independent_cost=check['independent_polyline_length_m']+problem.penalty*len(plan['selected_candidate_ids'])
            bounded=independent_cost<=reference_cost+1e-7
            reference_checks.append(dict(file=path.name,method=method,independent_objective_m=independent_cost,
                                         recreated_reference_m=reference_cost,nonworsening=bounded))
            if not bounded:errors.append(f'reference bound failed: {path.name} {method}')
        if (index+1)%24==0:print(f'revalidated {index+1}/{len(plan_files)} saved scenes',flush=True)
    if len(raw_checks)!=manifest['successful_plans']:
        errors.append('independent plan count differs from recorded successful plan count')
    metrics=read(args.results/'metrics.json')
    if len(metrics)!=manifest['cases']*len(manifest['methods']):
        errors.append('evaluation method-case denominator is incomplete')
    actual_failures=sum(row['status']!='complete' for row in metrics)
    if actual_failures!=manifest['failed_method_cases']:
        errors.append('failure count differs from manifest')
    write_json(args.output/'independent_saved_plans.json',dict(files=len(plan_files),plans=len(raw_checks),
               checks=raw_checks,reference_checks=reference_checks,failed_scene_files=failed_scene_files))

    forbidden=set()
    for name in manifest.get('data_sha256',{}):
        if name.endswith('_cases.json'):
            forbidden.update(r['seed'] for r in read(Path(name)) if isinstance(r,dict) and 'seed' in r)
    seed=6241030
    while seed in forbidden:seed+=1
    scene=make_complex_scene(10,10,seed,'blocked')
    method=training.get('recommended_stage','ppo')
    if method not in ('bc','ppo'):method='ppo'
    horizon=int(training['config']['horizon'])
    planner=RepairPlanner(args.checkpoint,method,horizon)
    cold=planner.plan(scene);warm=planner.plan(copy.deepcopy(scene))
    reload_result=RepairPlanner(args.checkpoint,method,horizon).plan(copy.deepcopy(scene))
    all_complete=all(r['status']=='complete' for r in (cold,warm,reload_result))
    equal=all_complete and cold['plans']==warm['plans']==reload_result['plans']
    cache_correct=all_complete and not cold['timing']['cache_hit'] and warm['timing']['cache_hit']
    if not all_complete or not equal or not cache_correct:errors.append('portable reload/cache equality check failed')
    fresh_path=args.output/f'scene_plan_fresh_{method}.json'
    write_json(fresh_path,safe_json(cold))
    if cold['status']=='complete':
        write_csv(fresh_path.with_suffix('.csv'),list(schedule_rows(cold)),SCHEDULE_COLUMNS)
        fresh_checks=validate_file(fresh_path)
        if not all(c['valid_model_plan'] and c['all_tasks_served'] for c in fresh_checks):
            errors.append('fresh API output failed independent reload validation')
    else:fresh_checks=[]
    deployment=dict(seed=seed,method=method,horizon=horizon,all_complete=all_complete,
                    cold_warm_reload_same_full_plan=equal,cache_hit_correct=cache_correct,
                    cold_plan_sha256=plan_hash(cold['plans']),warm_plan_sha256=plan_hash(warm['plans']),
                    reloaded_plan_sha256=plan_hash(reload_result['plans']),
                    cold_timing=cold.get('timing'),warm_timing=warm.get('timing'),independent_checks=fresh_checks)

    failure_records=[]
    for name in ('no_arm_reach','blocked_depot','variable_lift','invalid_task_ids'):
        invalid=copy.deepcopy(scene)
        if name=='no_arm_reach':invalid['tasks'][0]['z']=5.
        elif name=='blocked_depot':
            x,y=invalid['start']
            invalid['obstacles'].append(dict(xmin=x-.1,xmax=x+.1,ymin=y-.1,ymax=y+.1))
        elif name=='variable_lift':invalid['candidates'][0]['h']+=.1
        else:invalid['tasks'][0]['id']=99999
        expected_ids=[t['id'] for t in invalid['tasks']]
        result=planner.plan(invalid)
        retained=result.get('pending_task_ids')==expected_ids and result.get('task_ids')==expected_ids
        passed=result['status']=='failed' and not result.get('plans') and retained
        failure_records.append(dict(case=name,passed=passed,status=result.get('status'),code=result.get('code'),
                                    detail=result.get('detail'),task_count=len(expected_ids),all_task_ids_retained=retained))
        write_json(args.output/f'failure_{name}.json',safe_json(result))
        if not passed:errors.append(f'structured failure check failed: {name}')

    capture=io.StringIO()
    package=Path(__file__).resolve().parent
    suite=unittest.defaultTestLoader.discover(str(package),pattern='test_*.py',top_level_dir=str(package.parent))
    tested=unittest.TextTestRunner(stream=capture,verbosity=2).run(suite)
    (args.output/'unit_tests.log').write_text(capture.getvalue(),encoding='utf-8')
    tests=dict(run=tested.testsRun,passed=tested.wasSuccessful(),failures=len(tested.failures),errors=len(tested.errors),
               skipped=[dict(test=str(test),reason=reason) for test,reason in tested.skipped])
    if not tested.wasSuccessful():errors.append('unit tests failed')
    report=dict(status='complete',all_passed=not errors,errors=errors,training_run=str(args.run),results=str(args.results),
                checkpoint=str(args.checkpoint),checkpoint_sha256=checkpoint_hash,frozen_weights=frozen_weights,
                source_checks=source_checks,data_checks=data_checks,saved_scene_files=len(plan_files),
                independently_validated_plans=len(raw_checks),reference_bound_checks=len(reference_checks),
                evaluation_method_cases=len(metrics),evaluation_failed_method_cases=actual_failures,
                deployment=deployment,structured_failures=failure_records,unit_tests=tests,
                elapsed_s=time.perf_counter()-started,
                scope='Current fixed-height independent-arm reachability proxy and corridor graph only; no hardware execution or IK claim.')
    write_json(args.output/'verification.json',safe_json(report))
    print(json.dumps(dict(all_passed=report['all_passed'],saved_plans=len(raw_checks),reference_bounds=len(reference_checks),
                          failure_cases=len(failure_records),unit_tests=tests['run'],elapsed_s=report['elapsed_s'])),flush=True)
    if errors:raise SystemExit(2)


if __name__=='__main__':main()
