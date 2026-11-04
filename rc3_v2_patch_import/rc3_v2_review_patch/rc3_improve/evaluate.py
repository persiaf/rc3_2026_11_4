"""Fresh-scene comparison with independently validated, nonworsening incumbents."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from rc3_baseline.baseline import write_json, write_csv
from rc3_fast.scenes import make_complex_scene
from rc3_fast.solver import prepare, solve_route, export_plan
from .repair import FEATURE_NAMES


METHODS = ('fixed1', 'fixed3', 'random', 'detour', 'bc', 'ppo')


def safe_json(value):
    """Invalid inputs remain visible in diagnostics without emitting invalid JSON."""
    if isinstance(value, dict):
        return {str(k):safe_json(v) for k,v in value.items()}
    if isinstance(value, (list,tuple,np.ndarray)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value,float) and not np.isfinite(value):
        return str(value)
    if value is None or isinstance(value,(str,int,float,bool)):
        return value
    return repr(value)


def run_method(problem, method, policy=None, horizon=8, env_factory=None, fixed_ids=(0,1,7)):
    if method not in METHODS:
        raise ValueError('unknown method')
    if env_factory is None:
        from .environment import ImprovementEnv
        env_factory = ImprovementEnv
    started = time.perf_counter()
    env = env_factory(problem,horizon=horizon)
    incumbent = export_plan(problem,env.route,method)
    initial_cost = incumbent['objective_length_plus_stops_m']
    current_cost = initial_cost
    trace = [dict(event='validated_reference',objective_m=initial_cost)]
    attempts, accepted, fallback = 0,0,False
    timing = dict(feature_s=0.,model_s=0.,repair_s=0.,candidate_validation_s=0.)
    if method == 'fixed3':
        for recipe in fixed_ids:
            if recipe == 1:
                continue
            attempts += 1
            try:
                route = solve_route(problem,int(recipe))
                candidate = export_plan(problem,route,method)
                cost = candidate['objective_length_plus_stops_m']
                improve = cost < current_cost-1e-9
                if improve:
                    incumbent,current_cost = candidate,cost
                    accepted += 1
                trace.append(dict(event='recipe',recipe=int(recipe),accepted=improve,objective_m=current_cost))
            except Exception as error:
                fallback=True
                trace.append(dict(event='candidate_failed',error=str(error),objective_m=current_cost))
    elif method != 'fixed1':
        rng = np.random.default_rng(int(problem.scene.get('seed',0))+82991)
        for _ in range(horizon):
            if env.done:
                break
            previous = list(env.route)
            try:
                tick=time.perf_counter();features,mask=env.observe();timing['feature_s']+=time.perf_counter()-tick
                mask=np.asarray(mask,dtype=bool)
                if not mask.any():
                    trace.append(dict(event='no_eligible_window',objective_m=current_cost));break
                tick=time.perf_counter()
                if method=='random':
                    action=int(rng.choice(np.flatnonzero(mask)))
                elif method=='detour':
                    action=int(np.where(mask,features[:,FEATURE_NAMES.index('detour_excess')],-np.inf).argmax())
                else:
                    scores=np.asarray(policy.scores(features,mask))
                    if scores.shape!=mask.shape or not np.isfinite(scores[mask]).all():
                        raise ValueError('invalid learned scores')
                    action=int(np.where(mask,scores,-np.inf).argmax())
                timing['model_s']+=time.perf_counter()-tick
                attempts += 1
                tick=time.perf_counter();_,_,info=env.step(action);timing['repair_s']+=time.perf_counter()-tick
                changed=list(env.route)!=previous
                if changed:
                    tick=time.perf_counter();candidate=export_plan(problem,env.route,method)
                    timing['candidate_validation_s']+=time.perf_counter()-tick
                    cost=candidate['objective_length_plus_stops_m']
                    if cost>current_cost+1e-8:
                        raise ValueError('candidate worsens independently recomputed objective')
                    if cost < current_cost-1e-9:
                        incumbent,current_cost=candidate,cost;accepted+=1
                    else:
                        env.route=previous
                trace.append(dict(event='repair',action=action,changed=changed,objective_m=current_cost,
                                  accepted=bool(changed and list(env.route)!=previous),info=safe_json(info)))
            except Exception as error:
                env.route=previous
                fallback=True
                trace.append(dict(event='proposal_failed',error=str(error),objective_m=current_cost))
                break
    if current_cost>initial_cost+1e-8:
        raise AssertionError('validated incumbent guarantee violated')
    return incumbent,dict(status='complete',reference_objective_m=initial_cost,objective_m=current_cost,
                          planning_s=time.perf_counter()-started,attempts=attempts,accepted=accepted,
                          fallback_used=fallback,cost_nonincreasing=True,trace=trace,**timing)


def evaluate_case(scene, policies, horizon=8, repeats=3, fixed_ids=(0,1,7)):
    """Retain a row for every method even when preprocessing/solving fails."""
    if not isinstance(scene,dict):
        scene=dict(input_rejected=repr(scene),tasks=[])
    tasks=scene.get('tasks',[])
    tables=scene.get('tables',[])
    task_ids=[t.get('id') for t in tasks if isinstance(t,dict)] if isinstance(tasks,list) else []
    case={k:scene.get(k) for k in ('seed','nx','ny')}
    case.update(mode=scene.get('complexity','custom'),tasks=len(task_ids),tables=len(tables) if isinstance(tables,list) else 0)
    output=dict(scene=scene,task_ids=task_ids,plans={},diagnostics={})
    rows=[]
    try:
        problem=prepare(scene)
        output['feasibility']=problem.sparse
        cover=problem.F.any(2);reachable=np.isfinite(problem.nav['costs'][0,1:])
        no_service=np.flatnonzero(~cover.any(1)).tolist()
        disconnected=np.flatnonzero(cover.any(1)&~(cover&reachable).any(1)).tolist()
        if no_service or disconnected:
            raise ValueError(f'no_service={no_service}; disconnected={disconnected}; finite proxy model only')
    except Exception as error:
        for method in METHODS:
            rows.append(dict(**case,method=method,status='failed',error=str(error),pending_task_ids=task_ids))
        output.update(status='failed',pending_task_ids=task_ids,error=str(error))
        return rows,output
    samples={method:[] for method in METHODS}
    rng=np.random.default_rng(int(scene.get('seed',0))+19731)
    for _ in range(repeats):
        for method in rng.permutation(METHODS):
            try:
                plan,info=run_method(problem,str(method),policies.get(str(method)),horizon,fixed_ids=fixed_ids)
                samples[method].append(info)
                output['plans'][str(method)]=plan
                output['diagnostics'][str(method)]=info
            except Exception as error:
                samples[method].append(dict(status='failed',error=str(error)))
    for method in METHODS:
        results=samples[method]
        if any(r['status']!='complete' for r in results):
            output['plans'].pop(method,None)
            rows.append(dict(**case,method=method,status='failed',error=str(results),pending_task_ids=task_ids))
            continue
        info=results[0];plan=output['plans'][method]
        if any(abs(r['objective_m']-info['objective_m'])>1e-7 for r in results):
            raise AssertionError('repeated deterministic planning produced different costs')
        duration=float(np.median([r['planning_s'] for r in results]))
        rows.append(dict(**case,method=method,status='complete',error='',pending_task_ids=[],
                         objective_m=info['objective_m'],reference_objective_m=info['reference_objective_m'],
                         improvement_pct=(info['reference_objective_m']-info['objective_m'])/max(info['reference_objective_m'],1e-8)*100,
                         length_m=plan['route_length_m'],stops=len(plan['service_stops']),
                         planning_s=duration,preprocessing_s=problem.preprocessing_s,total_s=duration+problem.preprocessing_s,
                         attempts=info['attempts'],accepted=info['accepted'],fallback_used=any(r['fallback_used'] for r in results),
                         valid=plan['validation']['valid_model_plan'],cost_nonincreasing=info['cost_nonincreasing'],
                         **{k:float(np.median([r[k] for r in results])) for k in ('feature_s','model_s','repair_s','candidate_validation_s')}))
    output['status']='complete' if all(r['status']=='complete' for r in rows) else 'partial_failure'
    return rows,output


def summarize(rows):
    rng=np.random.default_rng(319)
    groups={'all':rows,'in_distribution_100':[r for r in rows if r['tables']==100],
            'smaller_transfer':[r for r in rows if 0<r['tables']<100],
            'larger_transfer':[r for r in rows if r['tables']>100],
            **{f'tables_{n}':[r for r in rows if r['tables']==n] for n in sorted({r['tables'] for r in rows})}}
    summaries,comparisons=[],[]
    for group,subset in groups.items():
        if not subset:continue
        for method in METHODS:
            total=[r for r in subset if r['method']==method];good=[r for r in total if r['status']=='complete']
            summaries.append(dict(group=group,method=method,total=len(total),success=len(good),failed=len(total)-len(good),
                                  fallback_cases=sum(r['fallback_used'] for r in good),
                                  mean_repair_attempts=float(np.mean([r['attempts'] for r in good])) if good else None,
                                  mean_objective_m=float(np.mean([r['objective_m'] for r in good])) if good else None,
                                  mean_improvement_pct=float(np.mean([r['improvement_pct'] for r in good])) if good else None,
                                  median_planning_ms=float(np.median([r['planning_s'] for r in good])*1000) if good else None,
                                  median_total_ms=float(np.median([r['total_s'] for r in good])*1000) if good else None))
        key=lambda r:(r['seed'],r['nx'],r['ny'],r['mode'])
        lookup={(key(r),r['method']):r for r in subset if r['status']=='complete'}
        for reference in ('fixed1','fixed3','random','detour','bc'):
            pairs=[(r,lookup[key(r),reference]) for r in subset if r['method']=='ppo' and r['status']=='complete' and (key(r),reference) in lookup]
            if not pairs: continue
            delta=np.array([(b['objective_m']-a['objective_m'])/max(b['objective_m'],1e-8)*100 for a,b in pairs])
            ci=np.quantile(delta[rng.integers(len(delta),size=(5000,len(delta)))].mean(1),[.025,.975]).tolist()
            comparisons.append(dict(group=group,method='ppo',reference=reference,paired_n=len(pairs),
                                    mean_reduction_pct=float(delta.mean()),ci95_pct=ci,wins=int((delta>1e-7).sum()),
                                    ties=int((np.abs(delta)<=1e-7).sum()),losses=int((delta< -1e-7).sum())))
    return dict(summaries=summaries,comparisons=comparisons)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--bc-model',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--horizon',type=int,default=8);parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args(argv)
    if args.horizon<1 or args.repeats<1: parser.error('positive horizon/repeats required')
    if args.output.exists() and any(args.output.iterdir()): parser.error('new or empty output required')
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'plans').mkdir()
    from .learning import load_policy
    policies={name:load_policy(path,device='cpu') for name,path in [('ppo',args.model),('bc',args.bc_model)]}
    cases=json.loads((args.data/'test_cases.json').read_text(encoding='utf-8'))
    with np.load(Path('rc3_fast/deployment/selector.npz'),allow_pickle=False) as archive:
        fixed_ids=json.loads(str(archive['metadata']))['fixed_portfolios']['top3']
    manifest=dict(status='running',cases=len(cases),horizon=args.horizon,repeats=args.repeats,methods=METHODS,
                  frozen_models={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.model,args.bc_model)},
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 [*Path(__file__).parent.glob('*.py'),Path('rc3_fast/solver.py'),Path('rc3_fast/scenes.py'),
                                  Path('rc3_baseline/baseline.py'),Path('rc3_baseline/validator.py')]},
                  data_sha256={str(args.data/name):hashlib.sha256((args.data/name).read_bytes()).hexdigest()
                               for name in ('manifest.json','train_cases.json','validation_cases.json','test_cases.json')},
                  timing_scope='Reference construction, all proposal/model/feature/repair time, independent initial and accepted-candidate validation; preprocessing separate; excludes model/process loading and file I/O.')
    write_json(args.output/'manifest.json',manifest);rows=[];started=time.perf_counter()
    for i,case in enumerate(cases):
        try:
            scene=case['scene'] if 'scene' in case else make_complex_scene(case['nx'],case['ny'],case['seed'],case.get('mode','regular'))
        except Exception as error:
            scene=dict(seed=case.get('seed',i),nx=case.get('nx'),ny=case.get('ny'),tasks=[],
                       complexity=case.get('mode','unknown'),input_case=case,generation_error=str(error))
        result,output=evaluate_case(scene,policies,args.horizon,args.repeats,fixed_ids)
        rows.extend(result);write_json(args.output/'plans'/f'scene_plan_{i:04d}_{case.get("seed",i)}.json',safe_json(output))
        if (i+1)%8==0:print(f'evaluated {i+1}/{len(cases)} elapsed={time.perf_counter()-started:.1f}s',flush=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    write_csv(args.output/'metrics.csv',rows,fields);write_json(args.output/'metrics.json',safe_json(rows))
    write_json(args.output/'summary.json',summarize(rows))
    manifest.update(status='complete',elapsed_s=time.perf_counter()-started,successful_plans=sum(r['status']=='complete' for r in rows),failed_method_cases=sum(r['status']!='complete' for r in rows))
    write_json(args.output/'manifest.json',manifest);print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
