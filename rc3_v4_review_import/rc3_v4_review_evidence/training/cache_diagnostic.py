"""Validation-only graph embedding cache equivalence and complete-plan timing."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--scenes',type=int,default=1)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args();sys.path.insert(0,str(args.source_root.resolve()));args.output_dir.mkdir(parents=True,exist_ok=True)
    from rc3_fast.scenes import make_complex_scene
    from rc3_fast.solver import prepare
    from rc3_v3.environment import Env
    from rc3_v3.model import select_action
    from rc3_v4.features import observation,collate
    from rc3_v4.model import NumpyPolicy
    from rc3_v4.policy import GraphPolicy
    from rc3_v4.runtime import run
    from graph_embedding_cache import CachedGraphPolicy
    cases=json.loads((args.source_root/'rc3_v3/data_v3_20261102/validation_cases.json').read_text())[:args.scenes]
    checkpoint=args.source_root/'rc3_v4/training/structured.npz'
    original=NumpyPolicy.load(checkpoint);cached=CachedGraphPolicy.load(checkpoint)
    records=[];checks=[];start=time.perf_counter()
    for i,case in enumerate(cases):
        problem=prepare(make_complex_scene(case['nx'],case['ny'],case['seed'],case['mode']))
        env=Env(problem,8)
        misses,hits=cached.cache_misses,cached.cache_hits
        with original.inference_session(),cached.inference_session():
            while not env.done:
                f,m=env.observe()
                if not m.any():break
                old=original.outputs(collate([observation(env,f,m)]))
                new=cached.outputs_env(env,f,m)
                errors=[]
                for a,b in zip(old,new):
                    assert np.array_equal(np.isfinite(a),np.isfinite(b))
                    finite=np.isfinite(a)
                    errors.append(float(np.abs(a[finite]-b[finite]).max()) if finite.any() else 0.)
                old_action=select_action(old[0][0],m,'binary_threshold',.9)
                new_action=select_action(new[0][0],m,'binary_threshold',.9)
                checks.append(dict(scene=i,step=env.steps,all_heads_bitwise_equal=all(np.array_equal(a,b) for a,b in zip(old,new)),max_output_error=max(errors),old_action=old_action,new_action=new_action))
                assert max(errors)<1e-7 and old_action==new_action
                if old_action is None:break
                env.step(old_action)
        # Interleave full original/cached calls; final certification and context
        # handling are included. Preprocessing/model loading are shared/excluded.
        policies={'original':GraphPolicy(original),'cached':CachedGraphPolicy.load(checkpoint)}
        rng=np.random.default_rng(110300+i)
        first_plan=None
        for repeat in range(args.repeats):
            for method in rng.permutation(['original','cached']):
                policy=policies[method]
                before={k:getattr(policy,k,0) for k in ('graph_s','inference_s','cache_hits','cache_misses')}
                plan,info=run(problem,'learned',policy,8)
                assert plan['validation']['valid_model_plan'] and plan['validation']['all_tasks_served'] and not info['failures']
                # Timing metadata excluded from identity comparison.
                identity={k:v for k,v in plan.items() if k not in ('timing','elapsed_s','planning_s')}
                if first_plan is None:first_plan=identity
                assert identity==first_plan
                records.append(dict(scene=i,seed=case['seed'],method=method,repeat=repeat,
                    planning_ms=info['planning_s']*1000,model_ms=info['model_s']*1000,
                    graph_ms=(getattr(policy,'graph_s',0)-before['graph_s'])*1000,
                    inference_ms=(getattr(policy,'inference_s',0)-before['inference_s'])*1000,
                    objective=info['objective_m'],calls=info['dp_calls'],accepted=info['accepted'],
                    cache_hits=getattr(policy,'cache_hits',0)-before['cache_hits'],cache_misses=getattr(policy,'cache_misses',0)-before['cache_misses']))
    summaries=[]
    for method in ('original','cached'):
        vals=[]
        for i in range(len(cases)):
            selected=[r for r in records if r['method']==method and r['scene']==i]
            vals.append({k:float(np.median([r[k] for r in selected])) for k in ('planning_ms','model_ms','graph_ms','inference_ms','objective','cache_hits','cache_misses')})
        summaries.append(dict(method=method,scenes=len(cases),**{k:float(np.median([r[k] for r in vals])) for k in vals[0]}))
    result=dict(scope='Post-freeze validation-only engineering diagnostic; not a new test, retraining or adoption decision.',scenes=len(cases),repeats=args.repeats,head_checks=len(checks),all_bitwise_equal=all(c['all_heads_bitwise_equal'] for c in checks),all_actions_equal=all(c['old_action']==c['new_action'] for c in checks),all_final_plan_objects_equal=True,all_independently_valid=True,summaries=summaries,records=records,checks=checks,elapsed_s=time.perf_counter()-start)
    (args.output_dir/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','checks')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
