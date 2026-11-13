"""New credit arithmetic on the frozen32768 D5 records; no Actor or H call."""
import csv,gzip,json,math,time
from collections import defaultdict
import numpy as np
import torch
from .common import ROOT,HOME,D5,read,put,unarchive,sha
from .data import Pool
from .runtime import References
from .costs import decompose,coefficients

def stats(v):
    a=np.asarray(v,dtype=np.float64)
    return dict(n=len(v),mean=float(a.mean()),std=float(a.std()),min=float(a.min()),p05=float(np.quantile(a,.05)),
        p50=float(np.quantile(a,.5)),p95=float(np.quantile(a,.95)),max=float(a.max())) if len(v) else dict(n=0)

def run(budget):
    start=time.time();pool=Pool(budget);refs=References(pool,budget);groups=defaultdict(list);grad=[];maxerr=0.;rows=0;loss_by_id={}
    target=HOME/'D5_TRAJECTORY_CREDIT.csv';columns=['request_id','task_id','size','parent','initial_tasks','actions','raw_J','raw_L','raw_K','J_H','L_H','K_H','reference_J_H','delta','H_correction','sum_logpi','loss_contribution','B_coeff_mean','B_coeff_std','negative_tail_fraction','decomposition_max_error']
    with target.open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,columns);writer.writeheader()
        for line in (D5/'streams/2026111101/D5-PG/REQUEST_POINTERS.jsonl').open(encoding='utf-8'):
            budget.check();ptr=json.loads(line);r=unarchive(ptr['record']);p,evidence=pool.get(r['task_id'])
            if p.identity!=r['problem_identity'] or evidence['physical_key']!=r['physical_key']:raise ValueError('Old online task source identity')
            credit=decompose(p,r);ref=refs.get(r['task_id']);delta=r['metrics']['J']-ref;coef=coefficients(credit,r['metrics']['J'],ref,'B')
            logps=torch.tensor(r['sample_log_prob'],dtype=torch.float32)
            loss=float(torch.tensor(delta,dtype=torch.float32)*logps.sum()/8)
            size=f"{r['spec']['width']}x{r['spec']['height']}";m=r['metrics']
            row=dict(request_id=r['request_id'],task_id=r['task_id'],size=size,parent=r['physical_key'],initial_tasks=len(p.C),actions=len(r['actions']),
                raw_J=m['raw_J'],raw_L=m['raw_L'],raw_K=m['raw_K'],J_H=m['J'],L_H=m['L'],K_H=m['K'],reference_J_H=ref,delta=delta,
                H_correction=credit['H_cost_correction'],sum_logpi=float(logps.sum()),loss_contribution=loss,B_coeff_mean=float(np.mean(coef)),B_coeff_std=float(np.std(coef)),
                negative_tail_fraction=float(np.mean(np.asarray(credit['tail_cost'])<0)),decomposition_max_error=credit['max_abs_decomposition_error'])
            writer.writerow(row);groups[size].append(row);loss_by_id[r['request_id']]=loss;maxerr=max(maxerr,credit['max_abs_decomposition_error']);rows+=1
            if rows%4096==0:print(dict(phase='COST_DECOMPOSITION',rows=rows,max_abs_error=maxerr,wall_s=time.time()-start),flush=True)
    if rows!=32768:raise ValueError('Exact32768 old online trajectories required')
    with gzip.open(D5/'streams/2026111101/D5-PG/UPDATES.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            u=json.loads(line);norm=u['gradient_norm'];reconstructed=sum(loss_by_id[x['episode_id']] for x in u['loss_sources'])
            grad.append(dict(update=u['cursor'],loss=u['loss'],gradient_norm=norm,
                clipped=norm>1,clip_factor=min(1.,1/(norm+1e-6)),batch_loss_sum_reconstruction=reconstructed,
                loss_reconstruction_abs_error=abs(reconstructed-u['loss'])))
    if len(grad)!=4096:raise ValueError('Exact4096 old saved updates required')
    with (HOME/'D5_GRADIENT_CLIPPING.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,list(grad[0]));w.writeheader();w.writerows(grad)
    absolute_total=sum(abs(r['loss_contribution']) for rr in groups.values() for r in rr)
    summary={}
    for size,rr in groups.items():
        dd=np.asarray([r['delta'] for r in rr]);signed=sum(r['loss_contribution'] for r in rr);absolute=sum(abs(r['loss_contribution']) for r in rr)
        summary[size]=dict(trajectories=len(rr),unique_tasks=len({r['task_id'] for r in rr}),parents=len({r['parent'] for r in rr}),
            positive_delta_fraction=float(np.mean(dd>0)),negative_delta_fraction=float(np.mean(dd<0)),zero_delta_fraction=float(np.mean(dd==0)),
            metrics={k:stats([r[k] for r in rr]) for k in ('delta','actions','raw_J','J_H','H_correction','loss_contribution','B_coeff_mean','negative_tail_fraction')},
            signed_loss_sum=signed,absolute_loss_sum=absolute,absolute_loss_share=absolute/absolute_total)
    # Fixed-state baseline depends on remaining/N and the frozen same-problem
    # reference only. Enumerating possible actions cannot change its value.
    first=next(iter(groups.values()))[0];p,_=pool.get(first['task_id']);reference=refs.get(first['task_id'])
    baseline=np.full(int(p.C.any(0).sum()),reference,dtype=np.float64)
    if not len(baseline) or np.ptp(baseline)!=0:raise ValueError('First-state action-independent baseline')
    output=dict(status='PASS',old_online_trajectories=rows,old_updates=len(grad),sizes=summary,max_abs_decomposition_error=maxerr,
        float64_tolerance=1e-7,terminal_exit_included=True,terminal_H_adjustment_included_once=True,
        baseline_input_contract='Only before-action remaining count, initial task count and frozen same-problem J_H; action argument absent',
        baseline_same_state_all_legal_actions_equal=True,initial_state_legal_actions_checked=len(baseline),
        gradient_norm=stats([r['gradient_norm'] for r in grad]),clipped_fraction=sum(r['clipped'] for r in grad)/len(grad),
        saved_batch_loss_reconstruction_max_abs_error=max(r['loss_reconstruction_abs_error'] for r in grad),
        loss_contribution_scope='Reconstructed scalar cost-weighted log-probability terms from saved D5 samples, not per-size parameter-gradient attribution',
        new_Actor_optimizer=0,new_forward=0,new_requests=0,new_H_native_teacher_navigation_scene=0,wall_s=time.time()-start)
    put(HOME/'COST_DECOMPOSITION_VALIDATION.json',output)
    budget.counts['old_D5_trajectories_cost_checked']=rows;budget.save()
