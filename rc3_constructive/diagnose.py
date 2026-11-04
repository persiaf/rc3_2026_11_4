"""C0 exact references, C1 checks, and pre-optimizer throughput calibration."""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from rc3_fast.scenes import make_complex_scene
from rc3_v4.exact import solve_metric
from .problem import ROOT,Cache,protocol,write,digest,resample_tasks,registered_scene
from .env import batch,Env,unpack_routes
from .model import Policy
from .evaluate import run_one


def register():
    target=ROOT/'outputs/registration.json'
    if target.exists():raise FileExistsError('Protocol already registered')
    workspace=ROOT.parent
    sources=['rc3_baseline/baseline.py','rc3_baseline/validator.py','rc3_fast/solver.py','rc3_fast/scenes.py','rc3_fast/planner.py',
             'rc3_v3/environment.py','rc3_v4/runtime.py','rc3_v4/exact.py','rc3_v5/external.py','rc3_improve/environment.py','rc3_improve/repair.py']
    write(target,dict(protocol_sha256=digest(ROOT/'protocol.yaml'),source_sha256={p:digest(workspace/p) for p in sources},
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=workspace).decode().strip(),
        hardware=dict(cpu=platform.processor(),cpu_threads=os.cpu_count(),gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                      gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
                      python=platform.python_version(),torch=torch.__version__,numpy=np.__version__),
        new_code_sha256={p.name:digest(p) for p in ROOT.glob('*.py')},
        scope='Registered before C0 measurements and optimizer updates; wall cap is a later, separately frozen throughput-dependent record.'))


def exact_cases():
    target=ROOT/'outputs/small_exact.json'
    if target.exists():raise FileExistsError('Refusing implicit exact rerun')
    spec=protocol()['splits']['small_exact'];records=[]
    for n in spec['task_counts']:
        for i in range(spec['per_count']):
            case=len(records);seed=spec['seed_start']+case
            layout=make_complex_scene(*spec['shape'],seed,('regular','mixed','blocked','regular')[i])
            scene=resample_tasks(layout,seed+1000,exact_count=n);item=Cache(0).prepare(scene)
            values={}
            for method in ('fixed3','detour8','detour16','insertion'):
                values[method]=run_one(scene,method,Cache(0))
            incumbent=values['detour16']['route']
            exact=solve_metric(item.C,item.D,.75,time_limit_s=30,incumbent=incumbent)
            chosen=values['fixed3']['route'][1:-1];indices=[0]+chosen
            remap={node:j+1 for j,node in enumerate(chosen)}
            restricted=solve_metric(item.C[:,np.asarray(chosen)-1],item.D[np.ix_(indices,indices)],.75,
                time_limit_s=30,incumbent=[0]+[remap[node] for node in chosen]+[0])
            record=dict(case=case,scene=scene,tasks=n,candidates=item.C.shape[1],baselines=values,full_exact=exact,
                        baseline_candidate_restriction=dict(candidate_nodes=chosen,result=restricted,
                        interpretation='May omit baseline stops; not a fixed-all-stops TSP'),
                        strict_outside_set_value=(exact['objective'] is not None and restricted['lower_bound'] is not None
                                                  and exact['objective']<restricted['lower_bound']-1e-8))
            if exact['route'] is not None:
                plan=item.certify(exact['route'],'global_exact');write(ROOT/'outputs/small_plans'/f'{case:02}.json',plan)
            records.append(record);write(target,records)
            print(json.dumps({'small_exact_case':case,'tasks':n,'status':exact['status'],'strict_outside_set_value':record['strict_outside_set_value']}),flush=True)


def calibrate():
    out=ROOT/'outputs';target=out/'wall_protocol.json'
    if target.exists():raise FileExistsError('Throughput/wall cap already frozen')
    registered=json.loads((out/'registration.json').read_text())
    if any(digest(ROOT.parent/name)!=value for name,value in registered['source_sha256'].items()):
        raise ValueError('Frozen imported source changed')
    execution=dict(protocol_sha256=digest(ROOT/'protocol.yaml'),
        source_sha256={name:digest(ROOT/name) for name in ('problem.py','env.py','model.py','baselines.py','evaluate.py','train.py','diagnose.py')},
        amendment='Before any neural timing/optimizer: sanitize masked infinite distances to avoid 0*infinity parameter gradients; add regression. No change to any legal-action score or C0 rule timing.')
    write(out/'execution_sources.json',execution)
    config=protocol()['training'];torch.manual_seed(config['seed']);device='cuda' if torch.cuda.is_available() else 'cpu'
    model=Policy().to(device);cache=Cache(24)
    scenes=[make_complex_scene(10,10,protocol()['splits']['throughput_only']['seed_start']+i,('regular','mixed','blocked')[i%3]) for i in range(32)]
    started=time.perf_counter();items=[cache.prepare(s) for s in scenes];prep=time.perf_counter()-started
    data=batch(items,device);durations=[]
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        if device=='cuda':torch.cuda.synchronize()
        tick=time.perf_counter();cost,logp,_,_=model(data,sample=False)
        (cost.float()/100*logp).mean().backward()
        if device=='cuda':torch.cuda.synchronize()
        durations.append(time.perf_counter()-tick)
    timings={}
    for candidate_device in ['cpu']+(['cuda'] if device=='cuda' else []):
        policy=copy.deepcopy(model).to(candidate_device).eval();tensor=batch(items[:1],candidate_device)
        if candidate_device=='cuda':torch.cuda.synchronize()
        tick=time.perf_counter()
        with torch.inference_mode():policy(tensor)
        if candidate_device=='cuda':torch.cuda.synchronize()
        timings[candidate_device]=time.perf_counter()-tick
    max_update=max(durations)+prep
    wall=int(min(7200,max(1800,math.ceil(512*max_update*2+600))))
    write(target,dict(protocol_sha256=digest(ROOT/'protocol.yaml'),execution_sources_sha256=digest(out/'execution_sources.json'),device=device,greedy_forward_backward_probe_s=durations,
        input_preparation_32_s=prep,estimated_cold_update_upper_s=max_update,training_wall_limit_s=wall,
        registration_before_optimizer=True,optimizer_updates=0,probe_greedy_trajectories=64,
        deployment_device='cpu',single_instance_initial_inference_s=timings,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
        rule='Predeclared min(7200,max(1800,ceil(512*max_probe_update_s*2+600))); entire measured 32-case preparation included conservatively.',
        note='No optimizer step and no sampled trajectory in calibration; gradients discarded; two greedy backward passes only.'))
    print(json.dumps(json.loads(target.read_text()),ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['register','exact','calibrate']);a=p.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        dict(register=register,exact=exact_cases,calibrate=calibrate)[a.phase]()


if __name__=='__main__':main()
