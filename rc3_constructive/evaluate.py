"""Matched cold/hot CPU measurements and final-path certification; no hidden fallback."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from rc3_fast.solver import route_objective
from .problem import ROOT, Cache, registered_scene, protocol, write
from .env import batch, unpack_routes, Env
from .model import Policy
from . import baselines


def load_policy(path, device='cpu'):
    tick=time.perf_counter();data=torch.load(path,map_location=device,weights_only=False)
    model=Policy();model.load_state_dict(data['state']);model.to(device).eval()
    return model,time.perf_counter()-tick


def greedy_routes(model, items, device='cpu'):
    data=batch(items,device)
    with torch.inference_mode():
        costs,_,actions,lengths=model(data)
    return unpack_routes(actions,lengths),costs.detach().cpu().numpy()


def run_one(scene, method, cache, model=None, device='cpu', save=None):
    started=time.perf_counter();item=cache.prepare(scene);prep=time.perf_counter()-started
    raw_plan=None;fallback=False;calls=0;raw_audit_s=0.
    if method in ('fixed3','detour8','detour16'):
        plan,info,times=baselines.reference(item,method)
        fallback=info['fallback_used'];calls=info['dp_calls'];raw_route=None
        raw_J=plan['objective_length_plus_stops_m'];post=times['postprocess_s'];finalize=times['export_validation_s'];decision=times['decision_s']
        raw_route=[0]+[j+1 for j in plan['visit_order_candidate_ids']]+[0]
    else:
        tick=time.perf_counter()
        if method in ('model','initial'):
            if model is None:raise ValueError('missing loaded policy')
            raw_route=greedy_routes(model,[item],device)[0][0]
        elif method=='insertion':raw_route=baselines.insertion(item)
        elif method=='prior':raw_route=baselines.prior(item)
        else:raise ValueError(method)
        decision=time.perf_counter()-tick
        raw_J=route_objective(item.base,raw_route);finalize=0.
        tick=time.perf_counter();route=baselines.postprocess(item,raw_route);post=time.perf_counter()-tick
        tick=time.perf_counter();plan=item.certify(route,method);finalize+=time.perf_counter()-tick
        if plan['objective_length_plus_stops_m']>raw_J+1e-7:raise ValueError('postprocess worsened objective')
    total=time.perf_counter()-started
    if method not in ('fixed3','detour8','detour16'):
        # Separate research audit, outside the deployment pipeline being timed.
        tick=time.perf_counter();raw_plan=item.certify(raw_route,method+'_raw');raw_audit_s=time.perf_counter()-tick
        if abs(raw_J-raw_plan['objective_length_plus_stops_m'])>1e-7:raise ValueError('raw export changed route cost')
    if not plan['validation']['valid_model_plan']:raise ValueError('failed independent certificate')
    row=dict(method=method,L=plan['route_length_m'],K=len(plan['service_stops']),J=plan['objective_length_plus_stops_m'],raw_J=raw_J,
             total_s=total,preparation_s=prep,decision_s=decision,postprocess_s=post,export_validation_s=finalize,
             timer_other_s=max(0.,total-prep-decision-post-finalize),cache_hit=item.cache_hit,
             raw_valid=True,valid=True,raw_audit_s=raw_audit_s,fallback=bool(fallback),dp_calls=int(calls),
             raw_route=raw_route,route=[0]+[j+1 for j in plan['visit_order_candidate_ids']]+[0],tasks=len(scene['tasks']))
    if save is not None:
        write(Path(save).with_suffix('.json'),plan)
        if raw_plan is not None:write(Path(save).with_name(Path(save).name+'_raw').with_suffix('.json'),raw_plan)
    return row


def measure(methods, models=None, label='references'):
    protocol_data=protocol();out=ROOT/'outputs';target=out/(label+'.json')
    if target.exists():raise FileExistsError('Refusing implicit measurement replay: '+str(target))
    records=[];models=models or {}
    for index in range(protocol_data['splits']['development']['count']):
        scene=registered_scene('development',index)
        write(out/'scenes'/f'dev_{index:02}.json',scene)
        # Alternate method order, cache order and repetition order deterministically.
        for rep in range(protocol_data['evaluation']['repetitions']):
            ordered=methods[rep%len(methods):]+methods[:rep%len(methods)]
            if index%2:ordered=list(reversed(ordered))
            for mode in (['cold','hot'] if (index+rep)%2==0 else ['hot','cold']):
                for method in ordered:
                    cache=Cache(2)
                    if mode=='hot':cache.prepare(scene)
                    save=out/'plans'/f'dev_{index:02}_{method}' if mode=='cold' and rep==0 else None
                    row=run_one(scene,method,cache,models.get(method),save=save)
                    row.update(case=index,rep=rep,cache_mode=mode,scene_seed=scene['seed'],scene_mode=scene['complexity'])
                    records.append(row)
        write(target,records)
        print(json.dumps({'evaluation':label,'cases_done':index+1,'rows':len(records)}),flush=True)
    return records


def selection_values(model, scenes, cache=None):
    cache=cache or Cache(8);items=[cache.prepare(s) for s in scenes];rows=[]
    for start in range(0,len(items),8):
        subset=items[start:start+8];routes,costs=greedy_routes(model,subset,'cpu')
        for offset,(item,route,cost) in enumerate(zip(subset,routes,costs)):
            env=Env(item);env.replay([j-1 for j in route[1:-1]])
            if abs(cost+sum(env.rewards))>1e-6:raise ValueError('decoder and replay reward differ')
            processed=baselines.postprocess(item,route)
            rows.append(dict(case=start+offset,raw_J=float(cost),post_J=route_objective(item.base,processed),raw_route=route,route=processed))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['references','final'],required=True);args=p.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if args.phase=='references':measure(['fixed3','detour8','detour16','insertion','prior'])
        else:
            selected=json.loads((ROOT/'outputs/training_summary.json').read_text())['selected_checkpoint']
            initial,load0=load_policy(ROOT/'outputs/checkpoints/update0000.pt')
            model,load1=load_policy(ROOT/selected)
            write(ROOT/'outputs/model_load.json',dict(initial_s=load0,selected_s=load1,scope='Tensor checkpoint and policy construction; Python/Torch import excluded'))
            measure(['initial','model'],dict(initial=initial,model=model),label='final_model')


if __name__=='__main__':main()
