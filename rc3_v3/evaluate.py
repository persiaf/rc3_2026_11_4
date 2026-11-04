"""Frozen v3 evaluation: DP-call budgets and validation-calibrated soft deadlines."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

from rc3_baseline.baseline import write_json,write_csv
from rc3_fast.scenes import make_complex_scene
from rc3_fast.solver import prepare
from rc3_improve.evaluate import safe_json


SEARCH_METHODS=('random','detour','learned')
CALL_BUDGETS=(1,2,4,8)


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def configurations(wall_seconds=(),include_wall=False):
    specs=[dict(id='fixed3',method='fixed3',kind='anchor',budget=0.,horizon=8,seconds=None)]
    specs += [dict(id=f'{m}_calls_{k}',method=m,kind='calls',budget=k,horizon=k,seconds=None)
              for k in CALL_BUDGETS for m in SEARCH_METHODS]
    if include_wall:
        specs += [dict(id=f'{m}_wall_{round(s*1000)}ms',method=m,kind='wall',budget=s,
                       horizon=256,seconds=s) for s in wall_seconds for m in SEARCH_METHODS]
    return specs


def calibrate(data,output,repeats=3,min_wall_ms=200.):
    """Use only validation references to freeze the deadline grid; no test access."""
    from .data import make_case_scene
    from .runtime import run
    path=data/'validation_cases.json'
    cases=read(path)
    timings=[];records=[]
    for case in cases:
        problem=prepare(make_case_scene(case));samples=[]
        for _ in range(repeats):
            tick=time.perf_counter();plan,info=run(problem,method='fixed3')
            samples.append(time.perf_counter()-tick)
            if not plan['validation']['valid_model_plan']:raise ValueError('invalid calibration reference')
        median=float(np.median(samples));timings.append(median)
        records.append(dict(seed=case['seed'],reference_s=median,samples_s=samples))
    p90=float(np.quantile(timings,.9))
    base=max(min_wall_ms/1000,math.ceil(2*p90/.025)*.025)
    result=dict(status='frozen_from_validation_only',validation_cases_sha256=sha(path),
                reference_source_sha256={str(p):sha(p) for p in [Path(__file__).parent/'runtime.py',
                    Path(__file__).parent/'environment.py',Path('rc3_fast/solver.py'),Path('rc3_baseline/baseline.py')]},
                rule='base=max(min_wall_ms, 2*p90(reference) rounded upward to 25 ms); grid=base*[1,2,4]',
                min_wall_ms=min_wall_ms,p90_reference_s=p90,seconds=[base,2*base,4*base],
                wall_horizon_safety_cap=256,training_horizon=8,repeats=repeats,records=records,
                scope='SOFT complete-planning deadline; includes anchor and final validation, excludes navigation preprocessing. Atomic DP and final verification can overrun. Wall horizon is outside the training budget.')
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise ValueError('do not overwrite a frozen wall-budget file')
    write_json(output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


def run_once(problem,spec,policy):
    from .runtime import run
    tick=time.perf_counter()
    try:
        plan,info=run(problem,method=spec['method'],policy=policy if spec['method']=='learned' else None,
                      horizon=spec['horizon'],time_limit_s=spec['seconds'])
        elapsed=time.perf_counter()-tick
        if not plan['validation']['valid_model_plan'] or not plan['validation']['all_tasks_served']:
            raise ValueError('runtime returned an invalid or incomplete plan')
        objective=plan['objective_length_plus_stops_m']
        if objective>info['reference_objective_m']+1e-7:raise ValueError('Fixed3 reference bound violated')
        return plan,dict(status='complete',objective_m=objective,reference_objective_m=info['reference_objective_m'],
                          planning_s=elapsed,inner_planning_s=info['planning_s'],
                          dp_calls=info.get('dp_calls',info.get('attempts',0)),accepted=info.get('accepted',0),
                          fallback_used=info.get('fallback_used',False),stop_reason=info.get('stop_reason'),
                          deadline_exceeded=bool(spec['seconds'] is not None and elapsed>spec['seconds']),
                          overrun_s=max(0.,elapsed-spec['seconds']) if spec['seconds'] is not None else 0.,
                          runtime=info)
    except Exception as error:
        return None,dict(status='failed',error=str(error),planning_s=time.perf_counter()-tick)


def evaluate_case(case,policy,wall_seconds,repeats=3):
    from .data import make_case_scene
    meta=dict(seed=case.get('seed'),nx=case.get('nx'),ny=case.get('ny'),mode=case.get('mode','custom'))
    tables=case.get('nx',0)*case.get('ny',0)
    meta['tables']=tables
    specs=configurations(wall_seconds,include_wall=tables==100)
    try:
        scene=make_case_scene(case)
        task_ids=[t['id'] for t in scene['tasks']]
        meta.update(tasks=len(task_ids),tables=len(scene['tables']))
        problem=prepare(scene)
    except Exception as error:
        scene=locals().get('scene',dict(input_case=case))
        task_ids=[t.get('id') for t in scene.get('tasks',[]) if isinstance(t,dict)]
        rows=[dict(**meta,config=s['id'],method=s['method'],kind=s['kind'],budget=s['budget'],
                   status='failed',error=str(error),pending_task_ids=task_ids) for s in specs]
        return rows,dict(status='failed',scene=scene,plans={},task_ids=task_ids,pending_task_ids=task_ids,error=str(error))
    # No anchor is shared: every timed run builds and validates its own Fixed3.
    output=dict(scene=scene,feasibility=problem.sparse,task_ids=task_ids,plans={},runs={})
    samples={s['id']:[] for s in specs};rng=np.random.default_rng(int(case['seed'])+67341)
    for repeat in range(repeats):
        for index in rng.permutation(len(specs)):
            spec=specs[int(index)]
            plan,info=run_once(problem,spec,policy)
            samples[spec['id']].append(info)
            label=f'{spec["id"]}_rep{repeat}'
            output['runs'][label]=info
            if plan is not None:output['plans'][label]=plan
    rows=[]
    for spec in specs:
        values=samples[spec['id']]
        base=dict(**meta,config=spec['id'],method=spec['method'],kind=spec['kind'],budget=spec['budget'],
                  horizon_cap=spec['horizon'],repeats=repeats,preprocessing_s=problem.preprocessing_s)
        if any(v['status']!='complete' for v in values):
            rows.append(dict(**base,status='failed',errors=[v.get('error') for v in values if v['status']!='complete'],
                             successful_repeats=sum(v['status']=='complete' for v in values),pending_task_ids=task_ids))
            continue
        costs=np.asarray([v['objective_m'] for v in values])
        references=np.asarray([v['reference_objective_m'] for v in values])
        if spec['kind']!='wall' and float(np.ptp(costs))>1e-7:
            raise AssertionError('DP-budget repeat changed a deterministic route cost')
        if float(np.ptp(references))>1e-7:raise AssertionError('reference changed across repeated runs')
        planning=float(np.median([v['planning_s'] for v in values]))
        rows.append(dict(**base,status='complete',objective_m=float(costs.mean()),
                         objective_min_m=float(costs.min()),objective_max_m=float(costs.max()),
                         reference_objective_m=float(references[0]),
                         improvement_pct=float(np.mean((references-costs)/np.maximum(references,1e-8)*100)),
                         planning_s=planning,total_s=planning+problem.preprocessing_s,
                         dp_calls=float(np.mean([v['dp_calls'] for v in values])),accepted=float(np.mean([v['accepted'] for v in values])),
                         fallback_rate=float(np.mean([v['fallback_used'] for v in values])),
                         overrun_rate=float(np.mean([v['deadline_exceeded'] for v in values])),
                         mean_overrun_s=float(np.mean([v['overrun_s'] for v in values])),
                         max_overrun_s=max(v['overrun_s'] for v in values),
                         stop_reasons=[v['stop_reason'] for v in values],pending_task_ids=[]))
    output['status']='complete' if all(r['status']=='complete' for r in rows) else 'partial_failure'
    return rows,output


def summarize(rows):
    groups={'all':rows,'primary_up_to_196':[r for r in rows if 0<r['tables']<=196],
            'in_distribution_100':[r for r in rows if r['tables']==100],
            'smaller_transfer':[r for r in rows if 0<r['tables']<100],
            'larger_transfer':[r for r in rows if 100<r['tables']<=196],
            'stress_over_196':[r for r in rows if r['tables']>196],
            **{f'tables_{n}':[r for r in rows if r['tables']==n] for n in sorted({r['tables'] for r in rows})}}
    summaries,comparisons=[],[];rng=np.random.default_rng(71291)
    key=lambda r:(r['seed'],r['nx'],r['ny'],r['mode'])
    for group,subset in groups.items():
        if not subset:continue
        config_ids=list(dict.fromkeys(r['config'] for r in subset))
        for cid in config_ids:
            all_rows=[r for r in subset if r['config']==cid];good=[r for r in all_rows if r['status']=='complete']
            first=all_rows[0]
            summaries.append(dict(group=group,config=cid,method=first['method'],kind=first['kind'],budget=first['budget'],
                                  total=len(all_rows),success=len(good),failed=len(all_rows)-len(good),
                                  mean_objective_m=float(np.mean([r['objective_m'] for r in good])) if good else None,
                                  mean_improvement_pct=float(np.mean([r['improvement_pct'] for r in good])) if good else None,
                                  median_planning_ms=float(np.median([r['planning_s'] for r in good])*1000) if good else None,
                                  p95_planning_ms=float(np.quantile([r['planning_s'] for r in good],.95)*1000) if good else None,
                                  median_total_ms=float(np.median([r['total_s'] for r in good])*1000) if good else None,
                                  mean_dp_calls=float(np.mean([r['dp_calls'] for r in good])) if good else None,
                                  mean_overrun_rate=float(np.mean([r['overrun_rate'] for r in good])) if good else None,
                                  mean_fallback_rate=float(np.mean([r['fallback_rate'] for r in good])) if good else None))
        lookup={(key(r),r['config']):r for r in subset if r['status']=='complete'}
        for cid in (c for c in config_ids if c.startswith('learned_')):
            current=[r for r in subset if r['config']==cid and r['status']=='complete']
            for reference in ('fixed3',cid.replace('learned_','detour_',1),cid.replace('learned_','random_',1)):
                pairs=[(r,lookup[key(r),reference]) for r in current if (key(r),reference) in lookup]
                if not pairs:continue
                delta=np.asarray([(b['objective_m']-a['objective_m'])/max(b['objective_m'],1e-8)*100 for a,b in pairs])
                ci=np.quantile(delta[rng.integers(len(delta),size=(5000,len(delta)))].mean(1),[.025,.975]).tolist()
                comparisons.append(dict(group=group,config=cid,reference=reference,paired_n=len(pairs),
                                        mean_reduction_pct=float(delta.mean()),ci95_pct=ci,
                                        wins=int((delta>1e-7).sum()),ties=int((np.abs(delta)<=1e-7).sum()),losses=int((delta< -1e-7).sum()),
                                        median_planning_speedup=float(np.median([b['planning_s']/a['planning_s'] for a,b in pairs]))))
    return dict(summaries=summaries,comparisons=comparisons,
                note='Wall quality averages timed repetitions within each scene before paired statistics; repetitions are not independent scenes. Wall deadlines are soft and horizons exceed training horizon. Primary results stop at 196 tables; stress_over_196 is separate. all is descriptive only and includes stress.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',type=Path);parser.add_argument('--wall-config',type=Path)
    parser.add_argument('--calibrate',action='store_true');parser.add_argument('--min-wall-ms',type=float,default=200.)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args(argv)
    if args.repeats<1:parser.error('repeats must be positive')
    if args.calibrate:
        calibrate(args.data,args.output,args.repeats,args.min_wall_ms);return
    if args.model is None or args.wall_config is None:parser.error('--model and frozen --wall-config required for evaluation')
    if args.output.exists() and any(args.output.iterdir()):parser.error('use a new or empty output directory')
    config=read(args.wall_config)
    if config['status']!='frozen_from_validation_only' or config['validation_cases_sha256']!=sha(args.data/'validation_cases.json'):
        raise ValueError('wall budgets are not frozen from this validation split')
    from .model import load_policy
    policy=load_policy(args.model)
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'plans').mkdir()
    cases=read(args.data/'test_cases.json')
    source_paths=[*Path(__file__).parent.glob('*.py'),Path('rc3_fast/solver.py'),Path('rc3_fast/scenes.py'),
                  Path('rc3_baseline/baseline.py'),Path('rc3_baseline/validator.py'),
                  Path('rc3_improve/repair.py'),Path('rc3_improve/environment.py'),Path('rc3_improve/evaluate.py')]
    manifest=dict(status='running',cases=len(cases),repeats=args.repeats,model_path=str(args.model),model_sha256=sha(args.model),
                  source_sha256={str(p):sha(p) for p in source_paths},
                  data_sha256={str(args.data/name):sha(args.data/name) for name in ('manifest.json','train_cases.json','validation_cases.json','test_cases.json')},
                  wall_config_path=str(args.wall_config),wall_config_sha256=sha(args.wall_config),wall_seconds=config['seconds'],
                  wall_subset='Only scenes with 100 tables; no wall-budget results for transfer sizes',
                  wall_subset_cases=sum(c['nx']*c['ny']==100 for c in cases),call_budgets=CALL_BUDGETS,
                  primary_cases=sum(c['nx']*c['ny']<=196 for c in cases),
                  stress_cases=sum(c['nx']*c['ny']>196 for c in cases),
                  wall_horizon_cap=256,wall_budget_extrapolation_beyond_training=True,
                  python=platform.python_version(),numpy=np.__version__,
                  timing_scope='Same prepared Problem for all methods. Each timed run rebuilds Fixed3, includes features/model/all repairs/final independent validation. Cold total adds common navigation preprocessing. Excludes process/model loading and disk I/O. Soft deadlines may overrun.',
                  repeat_aggregation='Mean objective over repeats within scene, median planning time; all repeated final plans saved; no best-repeat selection.')
    write_json(args.output/'manifest.json',manifest)
    warmup=prepare(make_complex_scene(4,3,1901,'regular'))
    run_once(warmup,configurations()[3],policy)
    started=time.perf_counter();rows=[];saved_plans=0
    for index,case in enumerate(cases):
        case_rows,output=evaluate_case(case,policy,config['seconds'],args.repeats)
        rows.extend(case_rows);saved_plans+=len(output['plans'])
        write_json(args.output/'plans'/f'scene_plan_{index:04d}_{case.get("seed",index)}.json',safe_json(output))
        if (index+1)%6==0:print(f'evaluated {index+1}/{len(cases)} scenes elapsed={time.perf_counter()-started:.1f}s',flush=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    write_csv(args.output/'metrics.csv',rows,fields);write_json(args.output/'metrics.json',safe_json(rows))
    write_json(args.output/'summary.json',summarize(rows))
    manifest.update(status='complete',elapsed_s=time.perf_counter()-started,method_budget_cases=len(rows),
                    successful_method_budget_cases=sum(r['status']=='complete' for r in rows),
                    failed_method_budget_cases=sum(r['status']!='complete' for r in rows),saved_repeated_plans=saved_plans)
    write_json(args.output/'manifest.json',manifest);print(json.dumps(manifest|{'source_sha256':'saved','data_sha256':'saved'}),flush=True)


if __name__=='__main__':main()
