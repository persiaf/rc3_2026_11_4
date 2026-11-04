"""Frozen input-ablation evaluation; no test-dependent selection or tuning.

Each method constructs its own certified Fixed3 anchor. Navigation preparation is
shared, model loading and disk writes are outside timing, and every repeated plan
is retained. The 400 ms budget is a soft deadline, including its anchor.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PureWindowsPath
import platform
import time
import numpy as np

from rc3_baseline.baseline import write_json
from rc3_baseline.validator import validate
from rc3_fast.scenes import make_complex_scene
from rc3_fast.solver import prepare
from rc3_improve.evaluate import safe_json
from .runtime import run

TRAIN_SEEDS = (17, 29, 43)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_source_file(recorded):
    """Explicit relocation of the three declared v4 training modules.

    Training logs retain the original absolute provenance. Only these named
    modules are mapped to the current package root, and their full hashes must
    still match. No basename search or missing-file bypass is performed.
    """
    source=PureWindowsPath(recorded)
    if source.parent.name!='rc3_v4' or source.name not in ('train.py','model.py','features.py'):
        raise ValueError('unrecognized training source identity: '+recorded)
    return Path('rc3_v4')/source.name


def configurations(tables, repeats=3):
    result = [dict(id='fixed3', method='fixed3', policy=None, kind='calls',
                   horizon=0, seconds=None, repeats=repeats)]
    for label, method, policy in [('detour8','detour',None),
                                  ('v3_mlp8','learned','v3_mlp'),
                                  ('control8','learned','control'),
                                  ('structured8','learned','structured')]:
        result.append(dict(id=label,method=method,policy=policy,kind='calls',
                           horizon=8,seconds=None,repeats=repeats))
    if tables == 100:
        for label in ('detour','control','structured'):
            result.append(dict(id=label+'_wall400',method='detour' if label=='detour' else 'learned',
                policy=None if label=='detour' else label,kind='wall',horizon=256,
                seconds=.4,repeats=repeats))
        for label in ('control','structured'):
            for seed in TRAIN_SEEDS:
                result.append(dict(id=f'{label}_seed{seed}_calls8',method='learned',
                    policy=f'{label}_seed{seed}',kind='seed',horizon=8,seconds=None,
                    repeats=1,training_seed=seed))
    return result


def run_once(problem, spec, policy=None):
    """Timing ends before an extra external recheck of the already certified plan."""
    counters={k:float(getattr(policy,k,0.)) for k in ('graph_s','inference_s','calls','stops')}
    started=time.perf_counter()
    plan=None
    try:
        plan,info=run(problem,method=spec['method'],policy=policy,horizon=spec['horizon'],
                      time_limit_s=spec['seconds'])
        elapsed=time.perf_counter()-started
        check=validate(problem.scene,plan,problem.sparse)
        objective=float(plan['objective_length_plus_stops_m'])
        reference=float(info['reference_objective_m'])
        if not check['valid_model_plan'] or not check['all_tasks_served']:
            raise ValueError('additional independent validation failed: '+str(check['errors']))
        if not np.isfinite([objective,reference]).all() or objective>reference+1e-7:
            raise ValueError('certified Fixed3 objective bound violated')
        result=dict(status='complete',objective_m=objective,reference_objective_m=reference,
            planning_s=elapsed,inner_planning_s=info['planning_s'],
            dp_calls=info['dp_calls'],accepted=info['accepted'],
            fallback_used=info['fallback_used'],stop_reason=info['stop_reason'],
            overrun_s=max(0.,elapsed-spec['seconds']) if spec['seconds'] is not None else 0.,
            deadline_exceeded=spec['seconds'] is not None and elapsed>spec['seconds'],
            independent_validation=check,runtime=info)
    except Exception as error:
        result=dict(status='failed',error=str(error),planning_s=locals().get('elapsed',time.perf_counter()-started),
                    pending_task_ids=[t['id'] for t in problem.scene['tasks']])
        if 'info' in locals():result['runtime']=info
        if 'check' in locals():result['independent_validation']=check
    for name,old in counters.items():
        result['policy_'+name]=float(getattr(policy,name,0.))-old
    return plan,result


def evaluate_case(case, policies, repeats=3, case_index=0):
    meta=dict(case_index=case_index,seed=int(case['seed']),nx=int(case['nx']),ny=int(case['ny']),
              mode=case['mode'],tables=int(case['nx'])*int(case['ny']))
    specs=configurations(meta['tables'],repeats)
    started=time.perf_counter()
    scene=None
    try:
        scene=make_complex_scene(meta['nx'],meta['ny'],meta['seed'],meta['mode'])
        generated=time.perf_counter()-started
        started=time.perf_counter();problem=prepare(scene);preprocessing=time.perf_counter()-started
    except Exception as error:
        runs={f'{s["id"]}_rep{r}':dict(status='failed',error=str(error),config=s,repeat=r,
                    pending_task_ids=[t.get('id') for t in (scene or {}).get('tasks',[])])
              for s in specs for r in range(s['repeats'])}
        return dict(status='failed',metadata=meta,scene=scene,feasibility=[],plans={},runs=runs,
                    preparation_error=str(error))
    meta['tasks']=len(scene['tasks'])
    output=dict(status='running',metadata=meta,scene=scene,feasibility=problem.sparse,
                preprocessing_s=preprocessing,reported_preprocessing_s=problem.preprocessing_s,
                scene_generation_s=generated,plans={},runs={},execution_order=[])
    # Every timed run rebuilds and certifies its own reference. The shared
    # Problem contains only common navigation/reachability preprocessing.
    schedule=[(s,r) for s in specs for r in range(s['repeats'])]
    rng=np.random.default_rng(meta['seed']+753911)
    for index in rng.permutation(len(schedule)):
        spec,repeat=schedule[int(index)];label=f'{spec["id"]}_rep{repeat}'
        policy=policies.get(spec['policy'])
        plan,record=run_once(problem,spec,policy)
        output['execution_order'].append(label)
        output['runs'][label]=dict(record,config=spec,repeat=repeat)
        if plan is not None:output['plans'][label]=plan
    output['status']='complete' if all(r['status']=='complete' for r in output['runs'].values()) else 'partial_failure'
    return output


def source_paths():
    return [Path('rc3_v4')/f for f in ('evaluate.py','report.py','runtime.py','features.py','policy.py','model.py','legacy_policy.py')]+[
        Path('rc3_v3')/f for f in ('model.py','environment.py')]+[
        Path('rc3_fast')/f for f in ('solver.py','scenes.py')]+[
        Path('rc3_baseline')/f for f in ('baseline.py','validator.py')]+[
        Path('rc3_improve')/f for f in ('repair.py','environment.py','evaluate.py')]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('rc3_v4/data'))
    parser.add_argument('--training',type=Path,default=Path('rc3_v4/training'))
    parser.add_argument('--output',type=Path,default=Path('rc3_v4/results'))
    parser.add_argument('--legacy-model',type=Path,default=Path('rc3_v3/deployment/policy.npz'))
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args(argv)
    if args.repeats!=3:parser.error('the frozen formal protocol uses exactly three repetitions')
    if args.output.exists() and any(args.output.iterdir()):parser.error('use a new or empty output directory')
    training=read(args.training/'summary.json')
    if training.get('status')!='complete':raise ValueError('training must be complete before freezing the test')
    for path,expected in training['source_sha256'].items():
        if sha(training_source_file(path))!=expected:raise ValueError(f'training source changed before evaluation: {path}')
    if sha(args.data/'data.pkl')!=training['data_sha256']:
        raise ValueError('training data changed before evaluation')
    cases=read(args.data/'test_cases.json')
    data_manifest=read(args.data/'manifest.json')
    if sha(args.data/'test_cases.json')!=data_manifest['test_sha256']:
        raise ValueError('test metadata changed after data preparation')
    sizes={n:sum(int(c['nx'])*int(c['ny'])==n for c in cases) for n in (36,100,196,400)}
    if len(cases)!=39 or sizes!={36:9,100:18,196:9,400:3} or len({c['seed'] for c in cases})!=39:
        raise ValueError('test metadata does not match the frozen 39-case protocol')
    model_paths={'v3_mlp':args.legacy_model}
    for variant in ('control','structured'):
        selected=training['selected'][variant]
        model_paths[variant]=args.training/(variant+'.npz')
        if sha(model_paths[variant])!=sha((args.training/selected['checkpoint']).with_suffix('.npz')):
            raise ValueError('deployment choice differs from validation-selected checkpoint')
        for seed in TRAIN_SEEDS:model_paths[f'{variant}_seed{seed}']=args.training/f'{variant}_seed{seed}.npz'
    from .policy import load_policy
    from .legacy_policy import load_policy as load_v3
    policies={k:(load_v3(path) if k=='v3_mlp' else load_policy(path)) for k,path in model_paths.items()}
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'plans').mkdir()
    manifest=dict(status='running',frozen_utc=datetime.now(timezone.utc).isoformat(),
        cases=len(cases),test_sizes=sizes,repeats=args.repeats,
        selected_training_records=training['selected'],training_seeds=list(TRAIN_SEEDS),
        model_sha256={str(p):sha(p) for p in model_paths.values()},model_paths={k:str(p) for k,p in model_paths.items()},
        source_sha256={str(p):sha(p) for p in source_paths()},
        data_sha256={str(p):sha(p) for p in [args.data/'test_cases.json',args.data/'manifest.json',args.training/'summary.json',args.data/'data.pkl']},
        planned_runs=855,planned_main_method_cases=195,planned_wall_method_cases=54,planned_seed_method_cases=108,
        stop_penalty_equivalent_m=.75,
        wall_seconds=.4,wall_tables=100,wall_horizon=256,training_horizon=8,
        timing_scope='Full planner including anchor, feature construction, model, repairs, final independent validation, and session thread control. Common navigation/reachability preprocessing listed separately. Excludes model/process loading, scene generation, extra post-timing audit and disk I/O. Atomic repair/final verification may overrun the soft deadline.',
        repeat_rule='Mean objective and median planning time within scene; preserve every repeat, never select the best test repeat.',
        selection_rule='Both control and structured checkpoints selected on the original 18 validation scenes only; all three training seeds reported on the 18 new 100-table cases.',
        legacy_policy_note='v3_mlp uses the frozen original weights with an output-equivalent v4 engineering adapter: cached BLAS controller once per full planning session. No retraining or weight changes.',
        acceptance=dict(primary_tables=100,minimum_reduction_pct=.2,ci95_lower_must_be_positive=True,
            references=['control8','detour8'],max_median_time_ratio_to_detour=1.25,all_plans_valid_required=True),
        python=platform.python_version(),numpy=np.__version__,platform=platform.platform())
    # This file is written BEFORE constructing the first test scene.
    write_json(args.output/'manifest.json',manifest)
    # Predeclared small development warmup, not a member of the frozen test set.
    warm=prepare(make_complex_scene(2,2,130041,'regular'))
    for name,policy in policies.items():
        run_once(warm,dict(method='learned',horizon=1,seconds=None),policy)
    run_once(warm,dict(method='detour',horizon=1,seconds=None))
    started=time.perf_counter();saved=0;successful=0;attempted=0
    for index,case in enumerate(cases):
        output=evaluate_case(case,policies,args.repeats,index)
        path=args.output/'plans'/f'scene_plan_{index:03d}_{case["seed"]}.json'
        write_json(path,safe_json(output))
        saved+=len(output['plans']);attempted+=len(output['runs'])
        successful+=sum(r['status']=='complete' for r in output['runs'].values())
        write_json(args.output/'progress.json',dict(completed_scenes=index+1,total_scenes=len(cases),
            attempted_runs=attempted,successful_runs=successful,saved_plans=saved,elapsed_s=time.perf_counter()-started))
        print(f'evaluated {index+1}/{len(cases)} scenes; plans={saved}; elapsed={time.perf_counter()-started:.1f}s',flush=True)
    changed=[p for field in ('model_sha256','source_sha256','data_sha256') for p,expected in manifest[field].items() if sha(p)!=expected]
    manifest.update(status='complete' if not changed else 'integrity_failure',elapsed_s=time.perf_counter()-started,
        attempted_runs=attempted,successful_runs=successful,failed_runs=attempted-successful,
        saved_plans=saved,changed_frozen_files=changed)
    write_json(args.output/'manifest.json',manifest)
    from .report import generate
    report=generate(args.output)
    print(json.dumps(dict(status=manifest['status'],runs=attempted,valid=report['verification']['valid_plans'],
                         adopted=report['decision']['adopt_structured'])),flush=True)
    return 0 if manifest['status']=='complete' and report['verification']['all_passed'] else 2


if __name__=='__main__':raise SystemExit(main())
