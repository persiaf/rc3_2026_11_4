"""Paired validation-only profile of existing old/new Fixed3 validation schedules.

This is an engineering measurement after model freeze, not model selection.
The same prepared Problem is used by both implementations. Each timed run
constructs its own three reference candidates. Independent rechecks run outside
the measured call; validation performed inside each implementation is included.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import platform
import time

import numpy as np

from rc3_baseline.baseline import write_json, write_csv
from rc3_baseline.validator import validate
from rc3_fast.solver import prepare
from rc3_improve.evaluate import run_method as old_fixed3
from .runtime import run as new_fixed3
from .environment import FEATURE_NAMES, FEATURE_SCHEMA


METHODS=('old_fixed3','new_fixed3')
REPEATS=3


def profile_case(problem, case, reference_cost):
    rng=np.random.default_rng(int(case['seed'])+730112)
    runs=[]
    for repeat in range(REPEATS):
        for order,method in enumerate(rng.permutation(METHODS)):
            method=str(method)
            entry=dict(scene_id=int(case['scene_id']),seed=int(case['seed']),mode=case['mode'],
                       tables=int(case['nx'])*int(case['ny']),repeat=repeat,order=order,method=method)
            try:
                # No cached initial route or reference plan is passed to either
                # implementation. The only shared object is scene preprocessing.
                tick=time.perf_counter()
                if method=='old_fixed3':
                    plan,info=old_fixed3(problem,'fixed3',fixed_ids=(0,1,7))
                else:
                    plan,info=new_fixed3(problem,'fixed3')
                complete_s=time.perf_counter()-tick
                cost=float(plan['objective_length_plus_stops_m'])
                internal=float(info['objective_m'])
                if not math.isfinite(cost) or abs(cost-internal)>1e-7 or abs(cost-reference_cost)>1e-7:
                    raise AssertionError('output/internal/stored canonical Fixed3 costs disagree')
                validation=validate(problem.scene,plan,problem.sparse)
                if not validation.get('valid_model_plan') or not validation.get('all_tasks_served'):
                    raise AssertionError(f'independent final record recheck failed: {validation}')
                if not plan.get('validation',{}).get('valid_model_plan'):
                    raise AssertionError('implementation did not certify its returned plan')
                entry.update(status='complete',planning_s=complete_s,
                    internal_reported_planning_s=float(info['planning_s']),
                    shared_preprocessing_s=float(problem.preprocessing_s),
                    total_s=complete_s+float(problem.preprocessing_s),
                    objective_m=cost,stored_reference_m=float(reference_cost),
                    cost_matches_stored=True,valid_model_plan=True,all_tasks_served=True,
                    independently_rechecked=True,fallback_used=bool(info.get('fallback_used',False)),
                    stops=len(plan['service_stops']))
            except Exception as error:
                entry.update(status='failed',error=f'{type(error).__name__}: {error}')
            runs.append(entry)
    return runs


def summarize(runs):
    grouped={}
    for row in runs:grouped.setdefault(row['scene_id'],[]).append(row)
    paired=[]
    for scene_id,rows in grouped.items():
        if len(rows)!=2*REPEATS or any(r['status']!='complete' for r in rows):continue
        selected={m:[r for r in rows if r['method']==m] for m in METHODS}
        if any(len(r)!=REPEATS for r in selected.values()):continue
        costs=[r['objective_m'] for r in rows]
        if max(costs)-min(costs)>1e-7:
            raise AssertionError('old/new or repeated Fixed3 costs differ')
        medians={m:float(np.median([r['planning_s'] for r in selected[m]])) for m in METHODS}
        old,new=medians['old_fixed3'],medians['new_fixed3']
        preprocessing=float(rows[0]['shared_preprocessing_s'])
        paired.append(dict(scene_id=scene_id,seed=rows[0]['seed'],mode=rows[0]['mode'],tables=rows[0]['tables'],
            objective_m=costs[0],all_six_costs_equal=True,all_six_independent_rechecks_passed=True,
            old_planning_median_s=old,new_planning_median_s=new,
            shared_preprocessing_s=preprocessing,old_total_median_s=old+preprocessing,
            new_total_median_s=new+preprocessing,planning_speedup_old_over_new=old/max(new,1e-12),
            planning_reduction_pct=100*(old-new)/max(old,1e-12),new_faster=new<old,
            order_by_repeat=[[r['method'] for r in rows if r['repeat']==i] for i in range(REPEATS)]))
    if not paired:return dict(paired_scenes=0),paired
    olds=np.array([r['old_planning_median_s'] for r in paired])
    news=np.array([r['new_planning_median_s'] for r in paired])
    pre=np.array([r['shared_preprocessing_s'] for r in paired])
    summary=dict(paired_scenes=len(paired),old_median_planning_ms=float(np.median(olds)*1000),
        new_median_planning_ms=float(np.median(news)*1000),
        median_per_scene_speedup=float(np.median(olds/news)),
        ratio_of_planning_medians=float(np.median(olds)/np.median(news)),
        median_per_scene_reduction_pct=float(np.median(100*(olds-news)/olds)),
        new_faster_scenes=int((news<olds).sum()),new_slower_scenes=int((news>olds).sum()),
        exact_time_ties=int((news==olds).sum()),new_win_ratio=float((news<olds).mean()),
        old_median_total_ms=float(np.median(olds+pre)*1000),
        new_median_total_ms=float(np.median(news+pre)*1000),
        median_per_scene_total_speedup=float(np.median((olds+pre)/(news+pre))),
        all_costs_equal=True,all_independent_rechecks_passed=True)
    return summary,paired


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('rc3_v3/engineering_profile.json'))
    args=parser.parse_args(argv)
    if args.output.exists():parser.error('use a new output path')
    data_path=args.data/'data.pkl'
    with data_path.open('rb') as handle:data=pickle.load(handle)
    if data.get('schema')!=FEATURE_SCHEMA or data.get('feature_names')!=FEATURE_NAMES:
        raise ValueError('validation cache feature schema mismatch')
    scenes=data['scenes']['validation'];cases=data['case_metadata']['validation']
    references=data['reference_costs']['validation']
    del data
    if not len(scenes)==len(cases)==len(references)==18:
        raise ValueError('profile requires the same eighteen frozen validation cases')
    if Counter(c['mode'] for c in cases)!=Counter(regular=6,mixed=6,blocked=6):
        raise ValueError('validation modes must have six cases each')
    if any(int(c['nx'])*int(c['ny'])!=100 for c in cases):
        raise ValueError('profile is restricted to the frozen 100-table validation set')
    sources=[Path(__file__),Path('rc3_improve/evaluate.py'),Path('rc3_improve/environment.py'),
             Path('rc3_v3/runtime.py'),Path('rc3_v3/environment.py'),Path('rc3_fast/solver.py'),
             Path('rc3_baseline/validator.py')]
    metadata=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Post-freeze engineering profile of existing Fixed3 validation scheduling; not model selection',
        methods=METHODS,repeats=REPEATS,validation_scenes=18,test_accessed=False,model_loaded=False,
        order='Seeded independent random method order for each of three repetitions per scene',
        timing_scope='Outer wall time of the full planner call, including independent internal validation; excludes scene preprocessing, imports, file I/O and the extra external recheck. Shared preprocessing is separately reported and added identically to both total-time columns.',
        independence='Every run constructs its own candidates (0,1,7); no cached route/plan is passed. Both use exactly the same preprocessed Problem.',
        limits='Only this hardware and eighteen 100-table validation scenes; development measurement after model freeze. No cross-version comparison on different test sets and no learned-policy speed claim.',
        hardware=dict(platform=platform.platform(),processor=platform.processor(),machine=platform.machine(),
                      logical_cpu_count=os.cpu_count(),python=platform.python_version(),numpy=np.__version__),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    progress=args.output.with_name(args.output.stem+'.progress.json')
    write_json(progress,metadata)
    started=time.perf_counter();runs=[]
    for index,(scene,case,reference) in enumerate(zip(scenes,cases,references)):
        problem=prepare(scene)
        case=dict(case,scene_id=index)
        runs.extend(profile_case(problem,case,float(reference)))
        summary,paired=summarize(runs)
        write_json(progress,dict(**metadata,completed_scenes=index+1,summary=summary,runs=runs,paired=paired))
        print(f'profile {index+1}/18; complete={sum(r["status"]=="complete" for r in runs)}/{len(runs)}',flush=True)
        del problem
    summary,paired=summarize(runs)
    failures=[r for r in runs if r['status']!='complete']
    metadata.update(status='complete' if not failures else 'partial_failure',elapsed_s=time.perf_counter()-started,
        successful_runs=len(runs)-len(failures),failed_runs=len(failures),summary=summary,paired=paired,runs=runs)
    write_json(args.output,metadata)
    write_csv(args.output.with_suffix('.csv'),paired)
    write_csv(args.output.with_name(args.output.stem+'_runs.csv'),runs)
    write_json(progress,dict(status=metadata['status'],completed_scenes=18,summary=summary))
    print(json.dumps(summary),flush=True)
    if failures:raise RuntimeError(f'{len(failures)} profile runs failed; all records retained')


if __name__=='__main__':main()
