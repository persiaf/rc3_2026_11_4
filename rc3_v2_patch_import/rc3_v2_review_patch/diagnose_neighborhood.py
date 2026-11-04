"""Fresh development diagnostic, not a held-out evaluation or a runtime benchmark.

Use the uploaded implementation unchanged. Compare identical local repairs under
random, detour and exhaustive best-immediate-improvement window selection. The
last strategy is an expensive diagnostic oracle, not a global upper bound.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--source-root', type=Path)
_args, _ = _pre.parse_known_args()
if _args.source_root is not None:
    ROOT = _args.source_root.resolve()
else:
    ROOT = next((candidate for parent in Path(__file__).resolve().parents
                 for candidate in (parent, parent/'v2_upload', parent/'source')
                 if (candidate/'rc3_improve/repair.py').is_file()
                 and (candidate/'rc3_fast/solver.py').is_file()), None)
if ROOT is None or not (ROOT/'rc3_improve/repair.py').is_file():
    raise SystemExit('Pass --source-root pointing to the extracted v2 delivery folder.')
sys.path.insert(0, str(ROOT))
from rc3_fast.scenes import make_complex_scene
from rc3_fast.solver import prepare, solve_route, route_objective, export_plan
from rc3_improve.environment import ImprovementEnv, canonical_route
from rc3_improve.repair import windows, window_features, repair_window, _local


def scan(problem, route, candidates, mask):
    before = route_objective(problem, route)
    gains = np.full(len(candidates), -np.inf)
    dp_calls = 0
    for a in np.flatnonzero(mask):
        result = canonical_route(problem, repair_window(problem, route, candidates[a]))
        gains[a] = max(0., before - route_objective(problem, result))
        dp_calls += 1
    return gains, dp_calls


def run(problem, start_route, strategy, horizon):
    env = ImprovementEnv(problem, horizon=horizon)
    env.route = list(start_route)
    env.initial_route = list(start_route)
    env.reference_cost = route_objective(problem, start_route)
    rng = np.random.default_rng(int(problem.scene['seed']) + 82991)
    started = time.perf_counter()
    calls, accepted, steps, history = 0, 0, 0, []
    for _ in range(horizon):
        f, mask = env.observe()
        if env.done:
            break
        if strategy == 'random':
            action = int(rng.choice(np.flatnonzero(mask)))
        elif strategy == 'detour':
            action = int(np.where(mask, f[:,8], -np.inf).argmax())
        else:
            gains, n = scan(problem, env.route, env.windows, mask)
            calls += n
            action = int(gains.argmax())
            if gains[action] <= 1e-8:
                break
        reward, done, info = env.step(action)
        calls += 1
        accepted += int(info['accepted'])
        steps += 1
        history.append(dict(action=action, reward=reward, **info))
        if done:
            break
    search_s = time.perf_counter() - started
    plan = export_plan(problem, env.route, f'diagnostic_{strategy}')
    return dict(objective_m=plan['objective_length_plus_stops_m'],
                improvement_pct=100*(env.reference_cost-plan['objective_length_plus_stops_m'])/env.reference_cost,
                steps=steps, accepted=accepted, dp_calls=calls, search_s=search_s,
                valid=plan['validation']['valid_model_plan'], route=env.route, trace=history)


def initial_diagnostics(problem, route):
    out = []
    before = route_objective(problem,route)
    for width, stride in ((3,2),(3,1),(4,2)):
        candidates = windows(problem, route, width=width, stride=stride)
        f = window_features(problem, route, candidates)
        mask = f[:,-1] > 0
        gains, n = scan(problem, route, candidates, mask)
        valid_gains = gains[mask]
        bound = []
        for w in candidates:
            start,end,required,pool,cover = _local(problem,route,w)
            local=route[start-1:end+1]
            old=sum(problem.nav['costs'][a,b] for a,b in zip(local,local[1:]))+problem.penalty*(end-start)
            max_gain = int(cover[np.ix_(required,pool-1)].sum(0).max()) if len(required) else 0
            min_stops = int(np.ceil(len(required)/max_gain)) if max_gain else 0
            bound.append(float(old-problem.nav['costs'][local[0],local[-1]]-problem.penalty*min_stops))
        out.append(dict(width=width,stride=stride,total=len(candidates),eligible=int(mask.sum()),
                        task_cap_exceeded=int((f[:,3]>8).sum()),candidate_cap_exceeded=int((f[:,4]>80).sum()),
                        max_required=int(f[:,3].max()),max_pool=int(f[:,4].max()),positive=int((valid_gains>1e-8).sum()),
                        best_gain_m=float(valid_gains.max(initial=0)),sum_positive_gain_m=float(valid_gains[valid_gains>0].sum()),
                        max_local_only_analytical_bound_m=max(bound,default=0),
                        detour_first_gain_m=float(gains[int(np.where(mask,f[:,8],-np.inf).argmax())]) if mask.any() else 0,
                        all_window_gains_m=[None if not np.isfinite(g) else float(g) for g in gains]))
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=ROOT)
    parser.add_argument('--cases',type=int,default=8)
    parser.add_argument('--horizon',type=int,default=8)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'fresh_repeat')
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a new or empty output directory so saved evidence is not overwritten.')
    args.output.mkdir(parents=True,exist_ok=True)
    rows,details=[] ,[]
    modes=['regular','mixed','blocked']
    for i in range(args.cases):
        seed=4200000200+i
        mode=modes[i%3]
        problem=prepare(make_complex_scene(10,10,seed,mode))
        variants=[canonical_route(problem,solve_route(problem,k)) for k in (1,0,7)]
        start1=variants[0]
        start3=min(variants,key=lambda r:route_objective(problem,r))
        case=dict(seed=seed,mode=mode,tasks=len(problem.F),
                  fixed1_J=route_objective(problem,start1),fixed3_J=route_objective(problem,start3),
                  fixed1_stops=len(start1)-2,fixed3_stops=len(start3)-2)
        detail=dict(**case,initial_windows={})
        for start_name, route in [('fixed1',start1),('fixed3',start3)]:
            detail['initial_windows'][start_name]=initial_diagnostics(problem,route)
            for strategy in ('random','detour','best_immediate'):
                result=run(problem,route,strategy,args.horizon)
                rows.append(dict(**case,start=start_name,strategy=strategy,
                                 **{k:v for k,v in result.items() if k not in ('route','trace')}))
                detail[f'{start_name}_{strategy}']=result
        details.append(detail)
        (args.output/'details.json').write_text(json.dumps(details,indent=2))
        with (args.output/'metrics.csv').open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        print(json.dumps({**case,'strategies':[{k:r[k] for k in ('start','strategy','objective_m','search_s')} for r in rows[-6:]]}),flush=True)
    metadata=dict(purpose='Exploratory fresh development diagnostics; not a frozen test or a runtime benchmark.',
                  data_seed_start=4200000200,cases=args.cases,horizon=args.horizon,
                  strategy_scope='best_immediate scans all eligible windows, then takes maximum immediate gain; not a globally optimal sequence.',
                  bound_scope='Local travel-plus-stops only, before canonical_route. Additional outside-window redundant-stop removal can exceed this bound; do not use as an upper bound on total environment reward.',
                  timing_scope='Search loop only; excludes shared preprocessing, construction, final export and independent validation.',
                  source_root=str(ROOT),valid_plans=sum(r['valid'] for r in rows),total_plans=len(rows))
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2))


if __name__=='__main__': main()
