"""One capped REINFORCE run; greedy rollout baseline, no PPO or teacher labels."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import copy
import gzip
import json
import math
import time
import numpy as np
import torch
from scipy.stats import ttest_rel
from threadpoolctl import threadpool_limits
from rc3_fast.scenes import make_complex_scene
from .problem import ROOT,Cache,protocol,write,digest,resample_tasks,registered_scene
from .env import batch,unpack_routes,Env
from .model import Policy
from .evaluate import selection_values


def cpu_copy(model): return copy.deepcopy(model).to('cpu').eval()


def checkpoint(path,model,optimizer,update):
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(schema='rc3-constructive-policy-v1',update=update,
        state={k:v.detach().cpu() for k,v in model.state_dict().items()},
        optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        protocol_sha256=digest(ROOT/'protocol.yaml')),path)


def main():
    out=ROOT/'outputs';summary_path=out/'training_summary.json'
    if summary_path.exists() or (out/'training.jsonl').exists():
        raise FileExistsError('Refusing implicit training replay/resume; existing resource record must be preserved')
    p=protocol();config=p['training'];wall=json.loads((out/'wall_protocol.json').read_text())
    if wall['protocol_sha256']!=digest(ROOT/'protocol.yaml'):raise ValueError('protocol changed after calibration')
    registered=json.loads((out/'registration.json').read_text())
    if registered['protocol_sha256']!=wall['protocol_sha256']:raise ValueError('registration changed')
    execution=json.loads((out/'execution_sources.json').read_text())
    if digest(out/'execution_sources.json')!=wall['execution_sources_sha256']:
        raise ValueError('execution source snapshot changed')
    if any(digest(ROOT/name)!=value for name,value in execution['source_sha256'].items()):
        raise ValueError('production code changed after throughput registration')
    torch.set_num_threads(1);torch.manual_seed(config['seed']);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    device=wall['device'];started=time.perf_counter();model=Policy().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    baseline=copy.deepcopy(model).eval();cache=Cache(24)
    maps=[make_complex_scene(10,10,config['map_seed_start']+i,('regular','mixed','blocked')[i%3]) for i in range(config['map_bank_size'])]
    dev=[registered_scene('development',i) for i in range(24)]
    selected_scenes=[registered_scene('selection',i) for i in range(48)]
    initial_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    checkpoint(out/'checkpoints/update0000.pt',model,optimizer,0)
    evaluations=[];baseline_updates=[];sampled=0;baseline_trajectories=0;validation_trajectories=0;update=0

    def validate(step):
        nonlocal validation_trajectories
        cpu=cpu_copy(model);rows=selection_values(cpu,selected_scenes)
        validation_trajectories+=len(rows)
        write(out/'selection'/f'update{step:04}.json',rows)
        values=dict(update=step,mean_raw_J=float(np.mean([r['raw_J'] for r in rows])),
                    mean_post_J=float(np.mean([r['post_J'] for r in rows])),checkpoint=f'outputs/checkpoints/update{step:04}.pt')
        evaluations.append(values);write(out/'validation_curve.json',evaluations)
        return values

    validate(0)
    with (out/'training.jsonl').open('x',encoding='utf-8') as history,gzip.open(out/'sampled_routes.jsonl.gz','wt',encoding='utf-8') as replay:
        for update in range(1,config['max_updates']+1):
            if time.perf_counter()-started>=wall['training_wall_limit_s']:
                update-=1;break
            tick=time.perf_counter();items=[];seeds=[]
            for b in range(config['batch_size']):
                ordinal=(update-1)*config['batch_size']+b
                layout=maps[ordinal%len(maps)];task_seed=config['task_seed_start']+ordinal
                scene=resample_tasks(layout,task_seed);items.append(cache.prepare(scene))
                seeds.append(dict(map_seed=layout['seed'],task_seed=task_seed,mode=layout['complexity']))
            preparation_s=time.perf_counter()-tick;data=batch(items,device)
            if device=='cuda':torch.cuda.synchronize()
            tick=time.perf_counter();model.train();cost,logp,actions,lengths=model(data,sample=True)
            sampled+=len(items)
            with torch.no_grad():baseline_cost,_,_,_=baseline(data,sample=False)
            baseline_trajectories+=len(items)
            loss=(((cost-baseline_cost).detach().float()/100.)*logp).mean()
            if not torch.isfinite(loss):raise ValueError('nonfinite loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'])
            if not torch.isfinite(grad):raise ValueError('nonfinite gradient')
            optimizer.step()
            if device=='cuda':torch.cuda.synchronize()
            compute_s=time.perf_counter()-tick;routes=unpack_routes(actions,lengths);costs=cost.detach().cpu().tolist()
            # Store all sampled trajectories, not full repeated distance matrices.
            for seed,route,value,item in zip(seeds,routes,costs,items):
                # Numeric bookkeeping per sampled trajectory; geometry certificates are separate.
                calculated=sum(item.D[a,b] for a,b in zip(route,route[1:]))+.75*(len(route)-2)
                if abs(value-calculated)>1e-6:raise ValueError('training cost/route mismatch')
                replay.write(json.dumps(dict(update=update,**seed,route=route,J=value),separators=(',',':'))+'\n')
            replay.flush()
            row=dict(update=update,sampled_trajectories=sampled,baseline_greedy_trajectories=baseline_trajectories,
                     loss=float(loss.item()),gradient_norm=float(grad.item()),sample_mean_J=float(cost.mean().item()),
                     baseline_mean_J=float(baseline_cost.mean().item()),mean_service_stops=float(lengths.float().mean().item()),
                     preparation_s=preparation_s,compute_s=compute_s,elapsed_s=time.perf_counter()-started,
                     prior_weight=float(torch.nn.functional.softplus(model.prior_weight).item()))
            history.write(json.dumps(row)+'\n');history.flush()
            if update%16==0:print(json.dumps(row),flush=True)
            if update%config['checkpoint_every']==0:
                checkpoint(out/f'checkpoints/update{update:04}.pt',model,optimizer,update)
                values=validate(update)
                candidate=selection_values(cpu_copy(model),dev);previous=selection_values(cpu_copy(baseline),dev)
                validation_trajectories+=48
                a=np.array([x['raw_J'] for x in candidate]);b=np.array([x['raw_J'] for x in previous])
                test=ttest_rel(a,b);pval=float(test.pvalue/2) if np.isfinite(test.pvalue) else 1.
                replace=bool(a.mean()<b.mean() and pval<.05)
                if replace:baseline=copy.deepcopy(model).eval()
                baseline_updates.append(dict(update=update,replaced=replace,one_sided_p=pval,
                                             candidate_mean_J=float(a.mean()),baseline_mean_J=float(b.mean())))
                write(out/'baseline_updates.json',baseline_updates)
                print(json.dumps({'validation':values,'rollout_baseline_update':baseline_updates[-1]}),flush=True)
            if sampled>config['max_sampled_trajectories']:raise AssertionError('sampling cap exceeded')
    if update%config['checkpoint_every']!=0:
        checkpoint(out/f'checkpoints/update{update:04}.pt',model,optimizer,update);validate(update)
    chosen=min(evaluations,key=lambda x:(x['mean_post_J'],x['update']))
    changed={k:bool(not torch.equal(initial_state[k],v.detach().cpu())) for k,v in model.state_dict().items()}
    if not all(changed.values()):
        # Count and disclose; zero change in one bias is not automatically a failed optimizer.
        pass
    summary=dict(schema='rc3-constructive-c2-run-v1',seed=config['seed'],device=device,updates=update,
        sampled_trajectories=sampled,baseline_greedy_trajectories=baseline_trajectories,
        cpu_validation_greedy_trajectories=validation_trajectories,calibration_greedy_trajectories=64,
        training_wall_limit_s=wall['training_wall_limit_s'],actual_elapsed_s=time.perf_counter()-started,
        stopped_for_wall_limit=update<config['max_updates'],training_maps=len(maps),
        distinct_training_task_seeds=sampled,selected_checkpoint=chosen['checkpoint'],selected_update=chosen['update'],
        selection=chosen,changed_parameter_tensors=sum(changed.values()),total_parameter_tensors=len(changed),
        protocol_sha256=digest(ROOT/'protocol.yaml'),history_sha256=digest(out/'training.jsonl'),
        stored_sampled_routes_sha256=digest(out/'sampled_routes.jsonl.gz'),
        c3_started=False,counts_scope='Trajectory counts include repeated maps; sampled main trajectories are at most 16384. Calibration is greedy only; validation and baseline forwards counted separately.')
    write(summary_path,summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
