"""Validation-only audit of GPU batches, sequential CPU Torch and portable NPZ.

This is correctness verification, not a latency benchmark or model-selection
routine. It never opens test_cases.json or any held-out test result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
import torch

from rc3_fast.solver import prepare, route_objective, export_plan
from rc3_v3.environment import Env, FEATURE_NAMES, FEATURE_SCHEMA
from rc3_v3.model import batch_states, load_policy, select_action


def sequential(policy,problem,route,horizon):
    env=Env(problem,horizon,initial_route=route)
    actions=[]
    while True:
        features,mask=env.observe()
        if env.done:break
        action=policy.select(features,mask)
        actions.append(action)
        if action is None:break
        env.step(action)
    return env,actions


def audit_checkpoint(path,problems,routes,device,horizon):
    cpu=load_policy(path,'cpu')
    batch_model=load_policy(path,device)
    npz_path=path.with_suffix('.npz')
    portable=load_policy(npz_path)
    envs=[Env(p,horizon,initial_route=r) for p,r in zip(problems,routes)]
    stopped=np.zeros(len(envs),bool)
    batch_actions=[[] for _ in envs]
    score_errors=dict(cpu_vs_numpy=0.,batch_vs_cpu=0.)
    mismatches=[];states_checked=0
    if any((p.stop_rule,p.stop_threshold)!=(cpu.stop_rule,cpu.stop_threshold)
           for p in (batch_model,portable)):
        mismatches.append(dict(kind='stopping_metadata'))
    while True:
        observations=[env.observe() for env in envs]
        active=[i for i,env in enumerate(envs) if not stopped[i] and not env.done]
        if not active:break
        x,mask=batch_states([observations[i] for i in active],device)
        with torch.no_grad():
            batch_result=tuple(value.cpu().numpy() for value in batch_model(x,mask))
        for row,index in enumerate(active):
            features,valid=observations[index]
            n=len(valid)
            cpu_result=cpu.outputs(features,valid)
            np_result=portable.outputs(features,valid)
            aligned=(np.r_[batch_result[0][row,:n],batch_result[0][row,-1]],
                     batch_result[1][row,:n],batch_result[2][row,:n])
            for head,(a,b,c) in enumerate(zip(cpu_result,np_result,aligned)):
                legal=np.r_[valid,True] if head==0 else valid
                score_errors['cpu_vs_numpy']=max(score_errors['cpu_vs_numpy'],float(np.abs(a[legal]-b[legal]).max(initial=0)))
                score_errors['batch_vs_cpu']=max(score_errors['batch_vs_cpu'],float(np.abs(a[legal]-c[legal]).max(initial=0)))
                if not np.allclose(a[legal],b[legal],atol=3e-5,rtol=3e-5):
                    mismatches.append(dict(scene=index,step=envs[index].steps,kind='cpu_npz_scores',head=head))
                if not np.allclose(a[legal],c[legal],atol=3e-5,rtol=3e-5):
                    mismatches.append(dict(scene=index,step=envs[index].steps,kind='batch_cpu_scores',head=head))
            actions=[select_action(result[0],valid,cpu.stop_rule,cpu.stop_threshold)
                     for result in (aligned,cpu_result,np_result)]
            if len(set(actions))!=1:
                mismatches.append(dict(scene=index,step=envs[index].steps,kind='same_state_action',
                                       batch=actions[0],cpu=actions[1],numpy=actions[2]))
            action=actions[0];batch_actions[index].append(action)
            if action is None:stopped[index]=True
            else:envs[index].step(action)
            states_checked+=1
    cases=[]
    for index,(problem,route) in enumerate(zip(problems,routes)):
        cpu_env,cpu_actions=sequential(cpu,problem,route,horizon)
        np_env,np_actions=sequential(portable,problem,route,horizon)
        costs=[route_objective(problem,e.route) for e in (envs[index],cpu_env,np_env)]
        action_match=batch_actions[index]==cpu_actions==np_actions
        cost_match=max(costs)-min(costs)<=1e-7
        if not action_match or not cost_match:
            mismatches.append(dict(scene=index,kind='independent_rollout',batch_actions=batch_actions[index],
                                   cpu_actions=cpu_actions,numpy_actions=np_actions,costs=costs))
        exported=export_plan(problem,np_env.route,'parity_numpy_validation')
        export_match=abs(costs[-1]-exported['objective_length_plus_stops_m'])<=1e-7
        if not export_match:
            mismatches.append(dict(scene=index,kind='export_objective'))
        cases.append(dict(scene_index=index,seed=problem.scene.get('seed'),actions_match=action_match,
                          costs_match=cost_match,export_match=export_match,
                          actions=batch_actions[index],batch_cpu_numpy_costs=costs,
                          independent_validation=exported['validation']['valid_model_plan']))
    return dict(checkpoint=str(path),numpy_checkpoint=str(npz_path),
                stop_rule=cpu.stop_rule,stop_threshold=cpu.stop_threshold,
                checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                numpy_sha256=hashlib.sha256(npz_path.read_bytes()).hexdigest(),
                states_checked=states_checked,score_max_abs_error=score_errors,
                all_passed=not mismatches,mismatches=mismatches,cases=cases)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--checkpoints',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--horizon',type=int,default=8)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.horizon<1:parser.error('horizon must be positive')
    with (args.data/'data.pkl').open('rb') as handle:
        data=pickle.load(handle)  # Our locally produced cache only.
    if data['schema']!=FEATURE_SCHEMA or data['feature_names']!=FEATURE_NAMES:
        raise ValueError('data schema mismatch')
    validation_scenes=data['scenes']['validation']
    routes=data['initial_routes']['validation']
    problems=[prepare(scene) for scene in validation_scenes]
    started=time.perf_counter();records=[]
    for path in args.checkpoints:
        record=audit_checkpoint(path,problems,routes,args.device,args.horizon)
        records.append(record)
        print(json.dumps(dict(checkpoint=path.name,passed=record['all_passed'],
                              states=record['states_checked'],errors=record['score_max_abs_error'],
                              mismatches=len(record['mismatches']))),flush=True)
    report=dict(status='pass' if all(r['all_passed'] for r in records) else 'mismatch',
                scope='Validation scenes only. Same-state three-head numeric parity and independently executed full action/cost rollouts. No test cases or costs inspected.',
                schema=FEATURE_SCHEMA,device=args.device,validation_scenes=len(problems),
                horizon=args.horizon,seconds=time.perf_counter()-started,torch=torch.__version__,
                data_sha256=hashlib.sha256((args.data/'data.pkl').read_bytes()).hexdigest(),checkpoints=records)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=report['status'],seconds=report['seconds'],output=str(args.output))),flush=True)
    if report['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
