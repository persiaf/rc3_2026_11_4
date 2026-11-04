"""Frozen-weight graph-cache correctness and optional validation-only timing.

Default execution checks equivalence only. Pass --profile to explicitly start
the 18-scene, 3-repeat randomized paired timing experiment. It never trains,
chooses checkpoints, or changes the original v4 adoption decision.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np

from rc3_baseline.baseline import write_json, write_csv
from rc3_baseline.validator import validate
from rc3_fast.scenes import make_complex_scene
from rc3_fast.solver import prepare, export_plan
from rc3_v3.environment import Env
from rc3_v3.model import select_action
from rc3_v4.features import observation, collate
from rc3_v4.model import NumpyPolicy
from rc3_v4.policy import GraphPolicy
from rc3_v4.runtime import run
from .policy import CachedGraphPolicy


CHECKPOINT = Path('rc3_v4/training/structured.npz')
CASES = Path('rc3_v3/data_v3_20261102/validation_cases.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol(cases, profile):
    paths=[Path(__file__),Path(__file__).with_name('policy.py'),CHECKPOINT,CASES,
        *[Path('rc3_v4')/f for f in ('model.py','policy.py','features.py','runtime.py')],
        Path('rc3_v3/environment.py'),Path('rc3_improve/repair.py'),Path('rc3_improve/environment.py'),
        Path('rc3_fast/solver.py'),Path('rc3_fast/scenes.py'),
        Path('rc3_baseline/baseline.py'),Path('rc3_baseline/validator.py')]
    return dict(status='running',frozen_utc=datetime.now(timezone.utc).isoformat(),
        scope='Validation-only post-freeze engineering maintenance. Not retraining, new test data or adoption evidence.',
        profile=profile,cases=cases,scenes=18,repeats=3 if profile else 0,horizon=8,
        expected_planning_runs=108 if profile else 36,expected_step_checks=0 if profile else 144,
        checkpoint_sha256=sha(CHECKPOINT),source_sha256={str(p):sha(p) for p in paths},
        python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
        ordering_seed=1150317,
        timing_scope='Whole planner including certified Fixed3 anchor, common/graph features, model, repair, final certificate and thread session; preprocessing/model load, extra recheck and disk I/O excluded.',
        cache_contract='F/D/scene semantics fixed for Env lifetime; new Env on changes; graph key includes route/window layout; frozen weights; dynamic heads/masks recomputed.')


def check_case(problem, original, cached, case_index):
    env=Env(problem,8);checks=[]
    with original.inference_session(),cached.inference_session():
        while True:
            f,m=env.observe()
            if env.done or not m.any():break
            left=original.outputs(collate([observation(env,f,m)]));right=cached.outputs_env(env,f,m)
            errors=[]
            for a,b in zip(left,right):
                if not np.array_equal(np.isfinite(a),np.isfinite(b)):raise AssertionError('nonfinite pattern differs')
                finite=np.isfinite(a);errors.append(float(np.max(abs(a[finite]-b[finite]))) if finite.any() else 0.)
            a=select_action(left[0][0],m);b=select_action(right[0][0],m)
            equal=all(np.array_equal(x,y) for x,y in zip(left,right))
            checks.append(dict(scene=case_index,step=env.steps,all_heads_bitwise_equal=equal,
                               max_output_error=max(errors),original_action=a,cached_action=b))
            if not equal or a!=b:raise AssertionError('cache changes a head or action')
            if a is None:break
            env.step(a)
    expected=export_plan(problem,env.route,'learned')
    plans={};runs={}
    for name,policy in [('original',GraphPolicy(original)),('cached',cached)]:
        plan,info=run(problem,'learned',policy,8)
        check=validate(problem.scene,plan,problem.sparse)
        if info['failures'] or not check['valid_model_plan'] or not check['all_tasks_served']:
            raise AssertionError('invalid full-plan equivalence run')
        if plan!=expected:raise AssertionError('full plan object changes with cache')
        plans[name]=plan;runs[name]=dict(runtime=info,independent_validation=check)
    return checks,plans,runs


def profile_case(problem, original, cached, case_index):
    policies={'original':GraphPolicy(original),'cached':cached}
    rng=np.random.default_rng(1150317+case_index);plans={};runs={};records=[];reference=None
    # Randomize all six method/repetition pairs, not only within each pair.
    schedule=[(name,repeat) for name in policies for repeat in range(3)]
    for position in rng.permutation(len(schedule)):
        name,repeat=schedule[int(position)];policy=policies[name]
        counters={k:float(getattr(policy,k,0.)) for k in ('graph_s','inference_s','cache_hits','cache_misses')}
        tick=time.perf_counter();plan,info=run(problem,'learned',policy,8);elapsed=time.perf_counter()-tick
        check=validate(problem.scene,plan,problem.sparse)
        if info['failures'] or not check['valid_model_plan'] or not check['all_tasks_served']:
            raise AssertionError('invalid benchmark plan')
        if reference is None:reference=plan
        if plan!=reference:raise AssertionError('profile final plan differs from frozen computation')
        if info['objective_m']>info['reference_objective_m']+1e-8:raise AssertionError('cost bound failed')
        key=f'{name}_rep{repeat}';plans[key]=plan
        record=dict(scene=case_index,seed=int(problem.scene['seed']),method=name,repeat=repeat,
            planning_ms=elapsed*1000,runtime_planning_ms=info['planning_s']*1000,
            model_ms=info['model_s']*1000,
            graph_ms=(getattr(policy,'graph_s',0.)-counters['graph_s'])*1000,
            inference_ms=(getattr(policy,'inference_s',0.)-counters['inference_s'])*1000,
            cache_hits=getattr(policy,'cache_hits',0)-counters['cache_hits'],
            cache_misses=getattr(policy,'cache_misses',0)-counters['cache_misses'],
            objective=info['objective_m'],calls=info['dp_calls'],accepted=info['accepted'],
            validation_passed=True,plan_object_equal=True)
        records.append(record);runs[key]=dict(record=record,runtime=info,independent_validation=check)
    return records,plans,runs


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',action='store_true')
    p.add_argument('--output',type=Path)
    a=p.parse_args(argv)
    output=a.output or Path('rc3_v5/cache_results')/('benchmark' if a.profile else 'equivalence')
    if output.exists() and any(output.iterdir()):raise ValueError('use a new/empty result directory')
    output.mkdir(parents=True,exist_ok=True);(output/'plans').mkdir()
    cases=json.loads(CASES.read_text(encoding='utf-8'))
    if len(cases)!=18 or len({c['seed'] for c in cases})!=18:raise ValueError('expected frozen 18 validation scenes')
    frozen=protocol(cases,a.profile);write_json(output/'manifest.json',frozen)
    original=NumpyPolicy.load(CHECKPOINT);cached=CachedGraphPolicy.load(CHECKPOINT)
    # Predeclared small warmup; no threshold or checkpoint selection follows it.
    warm=prepare(make_complex_scene(2,2,1150318,'regular'))
    for policy in (GraphPolicy(original),cached):run(warm,'learned',policy,1)
    start=time.perf_counter();checks=[];records=[];saved=0
    for i,case in enumerate(cases):
        problem=prepare(make_complex_scene(case['nx'],case['ny'],case['seed'],case['mode']))
        if a.profile:
            rec,plans,runs=profile_case(problem,original,cached,i);records.extend(rec)
        else:
            rec,plans,runs=check_case(problem,original,cached,i);checks.extend(rec)
        saved+=len(plans)
        write_json(output/'plans'/f'scene_plan_{i:03d}_{case["seed"]}.json',
                   dict(scene=problem.scene,feasibility=problem.sparse,plans=plans,runs=runs))
        write_json(output/'progress.json',dict(scenes=i+1,total=18,saved_plans=saved,head_checks=len(checks)))
        print(f'cache {"profile" if a.profile else "equivalence"} {i+1}/18',flush=True)
    changed=[s for s,h in frozen['source_sha256'].items() if sha(s)!=h]
    result=dict(scope=frozen['scope'],profile=a.profile,scenes=18,head_checks=len(checks),
        all_heads_bitwise_equal=all(r['all_heads_bitwise_equal'] for r in checks),
        all_actions_equal=all(r['original_action']==r['cached_action'] for r in checks),
        saved_plans=saved,all_full_plan_objects_equal=True,all_independently_valid=True,
        all_hashes_unchanged=not changed,changed_sources=changed,elapsed_s=time.perf_counter()-start)
    if a.profile:
        medians=[]
        for i in range(18):
            record=dict(scene=i)
            for name in ('original','cached'):
                chosen=[r for r in records if r['scene']==i and r['method']==name]
                for key in ('planning_ms','model_ms','graph_ms','inference_ms'):
                    record[name+'_'+key]=float(np.median([r[key] for r in chosen]))
            record['speedup']=record['original_planning_ms']/record['cached_planning_ms']
            record['reduction_pct']=100*(1-record['cached_planning_ms']/record['original_planning_ms'])
            medians.append(record)
        result['summary']={name:{k:float(np.median([r[name+'_'+k] for r in medians]))
                                  for k in ('planning_ms','model_ms','graph_ms','inference_ms')}
                           for name in ('original','cached')}
        result.update(median_paired_speedup=float(np.median([r['speedup'] for r in medians])),
            faster_scenes=sum(r['speedup']>1 for r in medians),slower_scenes=sum(r['speedup']<1 for r in medians),
            median_paired_reduction_pct=float(np.median([r['reduction_pct'] for r in medians])),
            ratio_of_group_medians=result['summary']['original']['planning_ms']/result['summary']['cached']['planning_ms'])
        write_csv(output/'scene_medians.csv',medians);write_csv(output/'repeat_metrics.csv',records)
    else:write_json(output/'head_checks.json',checks)
    result['all_passed']=not changed and saved==(108 if a.profile else 36)
    frozen.update(status='complete' if result['all_passed'] else 'failed',elapsed_s=result['elapsed_s'])
    write_json(output/'manifest.json',frozen);write_json(output/'summary.json',result)
    print(json.dumps(result),flush=True)
    if not result['all_passed']:raise SystemExit(2)


if __name__=='__main__':main()
