"""Validation-only STOP ablation; never reads test metadata, labels or results."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np

from rc3_fast.solver import prepare, route_objective, export_plan
from .environment import Env, canonical_route, repair_window
from .model import load_policy


def rollout(policy,problem,initial_route,use_stop,horizon):
    env=Env(problem,horizon,initial_route=initial_route)
    stopped=None
    while True:
        features,mask=env.observe()
        if env.done:break
        logits,gain,binary=policy.outputs(features,mask)
        action=int(np.argmax(logits if use_stop else logits[:-1]))
        if use_stop and action==len(mask):
            legal=np.r_[mask,True]
            p=np.exp(logits[legal]-logits[legal].max());p/=p.sum()
            before=route_objective(problem,env.route)
            residual=[]
            for j in np.flatnonzero(mask):
                candidate=canonical_route(problem,repair_window(problem,env.route,env.windows[int(j)]))
                residual.append(max(0.,before-route_objective(problem,candidate)))
            stopped=dict(step=env.steps,legal_actions=int(mask.sum()),stop_probability=float(p[-1]),
                         stop_logit=float(logits[-1]),best_action_logit=float(logits[:-1][mask].max()),
                         positive_remaining_actions=int(sum(g>1e-7 for g in residual)),
                         best_remaining_gain_pct=100*max(residual,default=0.)/max(env.reference_cost,1e-8))
            break
        env.step(action)
    plan=export_plan(problem,env.route,'stop_diagnostic')
    cost=plan['objective_length_plus_stops_m']
    assert abs(cost-route_objective(problem,env.route))<1e-7
    return dict(gain_pct=100*(env.reference_cost-cost)/max(env.reference_cost,1e-8),
                calls=env.steps,cost=cost,stop=stopped)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--horizon',type=int,default=8)
    args=p.parse_args()
    with (args.data/'data.pkl').open('rb') as handle:data=pickle.load(handle)
    policy=load_policy(args.checkpoint,'cpu')
    started=time.perf_counter();rows=[]
    for i,(scene,initial) in enumerate(zip(data['scenes']['validation'],data['initial_routes']['validation'])):
        problem=prepare(scene)
        original=rollout(policy,problem,initial,True,args.horizon)
        without=rollout(policy,problem,initial,False,args.horizon)
        rows.append(dict(scene=i,seed=scene['seed'],original_stop=original,without_stop=without))
    stops=[r['original_stop']['stop'] for r in rows if r['original_stop']['stop'] is not None]
    summary=dict(validation_cases=len(rows),original_mean_gain_pct=float(np.mean([r['original_stop']['gain_pct'] for r in rows])),
                 without_stop_mean_gain_pct=float(np.mean([r['without_stop']['gain_pct'] for r in rows])),
                 original_mean_calls=float(np.mean([r['original_stop']['calls'] for r in rows])),
                 without_stop_mean_calls=float(np.mean([r['without_stop']['calls'] for r in rows])),
                 stopped_cases=len(stops),stops_with_positive_residual=sum(s['positive_remaining_actions']>0 for s in stops),
                 stop_probabilities=[s['stop_probability'] for s in stops])
    report=dict(scope='Validation-only training diagnostic. Full residual DP enumeration used only to diagnose stopped states, never counted as deployable policy inference.',
                checkpoint=str(args.checkpoint),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                horizon=args.horizon,summary=summary,cases=rows,seconds=time.perf_counter()-started)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
