"""Fresh 100-table training only, with disjoint smaller/larger frozen test scenes."""
import argparse,json,pickle,time
from pathlib import Path
import numpy as np
from rc3_baseline.baseline import write_json
from rc3_fast.scenes import make_complex_scene
from rc3_fast.solver import prepare,route_objective,export_plan
from .environment import ImprovementEnv,canonical_route
from .repair import repair_window,route_is_valid,FEATURE_NAMES


def teacher_gain(problem, route, window):
    """Improvement label uses the canonical route that will actually be exported."""
    reference=canonical_route(problem,route)
    candidate=repair_window(problem,reference,window)
    if not route_is_valid(problem,candidate):
        return 0.
    candidate=canonical_route(problem,candidate)
    return max(0.,route_objective(problem,reference)-route_objective(problem,candidate))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--train-per-mode',type=int,default=32);p.add_argument('--val-per-mode',type=int,default=8)
    p.add_argument('--seed',type=int,default=6241130)
    a=p.parse_args()
    if a.output.exists():raise ValueError('use a new output directory')
    a.output.mkdir(parents=True)
    rng=np.random.default_rng(a.seed)
    # Avoid every previous case seed, including historical failed/development runs.
    forbidden=set()
    for folder in ('rc3_fast','rc3_learning','rc3_improve'):
        for path in Path(folder).rglob('*cases.json'):
            try:
                value=json.loads(path.read_text(encoding='utf-8'))
                if isinstance(value,list):forbidden.update(r['seed'] for r in value if isinstance(r,dict) and 'seed' in r)
            except (ValueError,KeyError):pass
    def seed():
        while True:
            n=int(rng.integers(100000,2**31-1))
            if n not in forbidden:forbidden.add(n);return n
    modes=('regular','mixed','blocked')
    cases={s:[dict(nx=10,ny=10,seed=seed(),mode=m) for m in modes for _ in range(n)]
           for s,n in [('train',a.train_per_mode),('validation',a.val_per_mode)]}
    cases['test']=[dict(nx=nx,ny=ny,seed=seed(),mode=m,split='test') for nx,ny in ((4,3),(6,6),(8,8),(10,10)) for m in modes for _ in range(8)]
    cases['test'] += [dict(nx=n,ny=n,seed=seed(),mode=m,split='ood') for n in (12,14) for m in modes for _ in range(2)]
    for s,records in cases.items():write_json(a.output/f'{s}_cases.json',records)
    manifest=dict(status='running',feature_names=FEATURE_NAMES,splits={k:len(v) for k,v in cases.items()},seed=a.seed,training_scales=['10x10'],test_frozen_before_training=True,
                  objective_definition='canonical route cost matching export_plan; zero-new-coverage stops removed before scoring')
    write_json(a.output/'manifest.json',manifest)
    cache,labels={},[];started=time.perf_counter()
    for split in ('train','validation'):
        cache[split]=[]
        for i,r in enumerate(cases[split]):
            problem=prepare(make_complex_scene(r['nx'],r['ny'],r['seed'],r['mode']))
            env=ImprovementEnv(problem)
            export_plan(problem,env.route,'reference')
            cache[split].append(problem)
            if split=='train':
                x,mask=env.observe()
                gain=np.asarray([teacher_gain(problem,env.route,w) for w in env.windows],np.float32)
                labels.append(dict(x=x,mask=mask,gain=gain))
            if (i+1)%12==0:print(f'{split} {i+1}/{len(cases[split])} elapsed {time.perf_counter()-started:.1f}s',flush=True)
    with (a.output/'cache.pkl').open('wb') as f:pickle.dump(dict(problems=cache,labels=labels),f,pickle.HIGHEST_PROTOCOL)
    manifest.update(status='complete',seconds=time.perf_counter()-started,teacher_positive_windows=int(sum((r['gain']>1e-7).sum() for r in labels)))
    write_json(a.output/'manifest.json',manifest);print(manifest,flush=True)

if __name__=='__main__':main()
