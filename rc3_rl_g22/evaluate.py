"""Run at most 72 registered frozen plans; no optimizer, rollout training, or forks."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import csv
import gzip
import json
import math
import time
from dataclasses import asdict
import numpy as np
from .common import ROOT,WORKSPACE,METHODS,read,write,sha,digest,jsonable,check_contract,teacher_index,feedback_label,finish_feedback,schedule

def authorize():
    path=ROOT/'outputs/protocol.json';p=read(path)
    if sha(path)!=(path.with_suffix('.sha256')).read_text().strip():raise ValueError('Protocol identity mismatch')
    check_contract(p)
    for name,expected in p['implementation_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Evaluation implementation changed: '+name)
    for identity in p['models'].values():
        if sha(WORKSPACE/identity['path'])!=identity['sha256']:raise ValueError('Frozen checkpoint changed')
    for c in p['cases']:
        if sha(WORKSPACE/c['scene_path'])!=c['scene_sha256']:raise ValueError('Frozen scene changed')
    if sha(WORKSPACE/'rc3_rl_g2/outputs/teacher_extra/teacher.pkl')!=p['extra_teacher_sha256']:
        raise ValueError('Teacher changed before trusted loading')
    return p

def model_identity(model):
    import hashlib
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        value=tensor.detach().cpu().numpy();h.update(name.encode());h.update(value.dtype.str.encode());h.update(str(value.shape).encode());h.update(value.tobytes())
    return h.hexdigest()

def gzip_write(path,value):
    with gzip.open(path,'wt',encoding='utf-8') as f:
        json.dump(jsonable(value),f,ensure_ascii=False,allow_nan=False)

def greedy(model,raw):
    import torch
    with torch.no_grad():
        logits,values=model([raw]);scores=logits[0,:len(raw['mask'])]
        if not torch.isfinite(scores[torch.as_tensor(raw['mask'],dtype=torch.bool)]).all():raise ValueError('Nonfinite legal actor score')
        action=int(scores.argmax().item());prob=scores.softmax(0)
    if not raw['mask'][action]:raise ValueError('Greedy policy selected illegal action')
    return action,dict(logits=[float(v) if math.isfinite(float(v)) else None for v in scores],
        probabilities=prob.cpu().tolist(),value=float(values[0]))

def run_original(scene,method,p,started,input_s):
    from rc3_fast.solver import prepare
    from rc3_v4.runtime import run
    tick=time.perf_counter();problem=prepare(scene,penalty=.75);preparation_s=time.perf_counter()-tick
    remaining=max(0.,started+p['soft_complete_limit_s']-time.perf_counter())
    plan,diagnostic=run(problem,method=method,horizon=p['original_detour_horizon'],time_limit_s=remaining)
    elapsed=time.perf_counter()-started
    row=dict(J=float(plan['objective_length_plus_stops_m']),L=float(plan['route_length_m']),K=len(plan['service_stops']),
        anchor_J=float(diagnostic['reference_objective_m']),return_=None,work_used=None,shared_environment=False,
        steps=int(diagnostic['dp_calls']),proposal_calls=int(diagnostic['dp_calls']),
        route_changes=int(diagnostic['provisional_improvements']),best_improvements=int(diagnostic['provisional_improvements']),
        stop_reason=diagnostic['stop_reason'],explicit_stop=False,stop_at_root=False,
        fallback=bool(diagnostic['fallback_used']),failures=diagnostic['failures'],complete_s=elapsed,
        time_overrun_s=max(0.,elapsed-p['soft_complete_limit_s']),work_overrun=None,
        timing=dict(input_read_s=input_s,prepare_s=preparation_s,frozen_runtime=diagnostic),
        work_scope='Original frozen algorithm; no shared-work reward imputed')
    evidence=dict(plan=plan,sparse_feasibility=problem.sparse,diagnostics=diagnostic,
                  actions=[],events=[],scope='Original solver without shared action protocol')
    return row,evidence

def run_shared(scene,method,model,p,case,index,started,input_s):
    from rc3_rl_research.types import SearchConfig
    from rc3_rl_g2.adapter_g2 import ResearchAdapterG2
    from rc3_rl_g2.env_g2 import SearchEnvG2
    from rc3_rl_g2.graph_observation import build_policy_observation
    from rc3_rl_g21.baselines import make_policy
    config=SearchConfig(**p['config']);tick=time.perf_counter()
    adapter=ResearchAdapterG2.from_scene(scene,.75);preparation_s=time.perf_counter()-tick
    env=SearchEnvG2(adapter,config,case['seed']+991,record_transitions=True)
    selector=None if model is not None else make_policy('stop' if method=='stop' else 'detour_early',**({} if method=='stop' else p['early_stop']))
    deadline=started+p['soft_complete_limit_s'];rows=[];raw_records=[];failures=[]
    graph_s=inference_s=label_s=selection_s=0.;root_stop_return=None
    while not env.done:
        if time.perf_counter()>=deadline:
            env.terminate('complete_soft_deadline');break
        observation=env.observe()
        if time.perf_counter()>=deadline:
            env.terminate('complete_soft_deadline');break
        if root_stop_return is None:
            root_stop_return=-config.lambda_work*observation['pending_menu_work']/config.work_scale
        try:
            if model is not None:
                tick=time.perf_counter();raw=build_policy_observation(env);graph_s+=time.perf_counter()-tick
                tick=time.perf_counter();action,scores=greedy(model,raw);inference_s+=time.perf_counter()-tick
                raw_records.append(dict(step=env.steps,observation=raw,action_id=action,scores=scores,executed=False))
            else:
                tick=time.perf_counter();action=int(selector.select(observation,env.rng));selection_s+=time.perf_counter()-tick
            tick=time.perf_counter();label=feedback_label(index,observation,action);label_s+=time.perf_counter()-tick
            if time.perf_counter()>=deadline:
                env.terminate('complete_soft_deadline');break
            prefix=float(env.total_reward);before_calls=env.counters['proposal_calls'];before_work=env.work_used
            before_best=env.best_cost;before_current=env.current_cost
            hypothetical_stop_total=100*(env.anchor_cost-before_best)/max(env.anchor_cost,1e-8)-config.lambda_work*(before_work+observation['pending_menu_work'])/config.work_scale
            next_observation,reward,terminal,truncated,info=env.step(action)
            if raw_records and model is not None:raw_records[-1]['executed']=True
            row=dict(step=len(rows),action_id=action,action=observation['actions'][action],
                environment_observation=observation,prefix_return=prefix,reward=float(reward),
                before_best_J=before_best,after_best_J=env.best_cost,before_current_J=before_current,after_current_J=env.current_cost,
                before_work=before_work,after_work=env.work_used,
                proposal_calls=env.counters['proposal_calls']-before_calls,
                immediate_best_quality_percent=100*(before_best-env.best_cost)/max(env.anchor_cost,1e-8),
                realized_reward_so_far=env.total_reward,terminated=terminal,truncated=truncated,
                hypothetical_stop_total_return=hypothetical_stop_total,
                hypothetical_scope='Retrospective zero-potential accounting at a visited decision, assuming no final validation rollback; not a deployed oracle',
                info=info,**label)
            rows.append(row)
            if selector is not None:selector.update(info)
        except Exception as error:
            failures.append(f'{type(error).__name__}: {error}')
            env.terminate('evaluation_error');break
    certificate,finalization=env.finalize();elapsed=time.perf_counter()-started
    expected=100*(env.anchor_cost-certificate.cost)/max(env.anchor_cost,1e-8)-config.lambda_work*env.work_used/config.work_scale
    if abs(expected-env.total_reward)>1e-7:raise ValueError('Shared returned-plan reward mismatch')
    finish_feedback(rows,env.total_reward,env.events)
    if certificate.cost>env.anchor_cost+1e-7:raise ValueError('Worsened certified anchor')
    if method=='stop' and rows and (len(rows)!=1 or abs(env.total_reward-root_stop_return)>1e-9):
        raise ValueError('Actual immediate STOP differs from zero-potential accounting')
    plan=certificate.plan
    row=dict(J=certificate.cost,L=float(plan['route_length_m']),K=len(plan['service_stops']),anchor_J=env.anchor_cost,
        return_=env.total_reward,work_used=env.work_used,shared_environment=True,steps=env.steps,
        proposal_calls=env.counters['proposal_calls'],route_changes=sum(r['info']['route_changed'] for r in rows),
        best_improvements=env.counters['best_improvements'],stop_reason=env.stop_reason,
        explicit_stop=bool(rows and rows[-1]['action']['kind']=='stop'),stop_at_root=bool(len(rows)==1 and rows[0]['action']['kind']=='stop'),
        stop_after_proposals=env.counters['proposal_calls'],fallback=finalization['fallback_used'],
        failures=failures,work_count_incomplete=env.counters['work_count_incomplete'],
        proposal_failure_reasons=[r['info']['failure_reason'] for r in rows if r['info']['failure_reason']],
        complete_s=elapsed,time_overrun_s=max(0.,elapsed-p['soft_complete_limit_s']),work_overrun=finalization['work_overrun'],
        root_stop_return_analytical=root_stop_return,
        timing=dict(input_read_s=input_s,prepare_and_anchor_s=preparation_s,graph_s=graph_s,inference_s=inference_s,
            feedback_lookup_s=label_s,selection_s=selection_s,**env.timings),
        work_scope='Frozen shared menu/proposal work, with mandatory final certificate')
    evidence=dict(plan=plan,sparse_feasibility=adapter.problem.sparse,actions=rows,events=env.events,
        policy_observations=raw_records,finalization=finalization,counters=env.counters,
        final_current=list(env.current),final_best=list(env.best),anchor=list(env.anchor),
        adapter_problem_id=adapter.problem_id,cache_stats=adapter.cache_stats)
    return row,evidence

def main():
    p=authorize();output=ROOT/'outputs/evaluation'
    if output.exists():raise FileExistsError('No automatic replay or plan retries; preserve the registered 72 slots')
    from rc3_rl_g2.train_ppo import configure
    from rc3_rl_g2.checkpoints import load_model
    from rc3_rl_g2.teacher_dataset import load_teacher
    configure(2026110401)
    models={};setup={}
    for name,identity in p['models'].items():
        tick=time.perf_counter();model,_=load_model(WORKSPACE/identity['path'],'cpu');model.eval()
        models[name]=model;setup[name]=dict(load_s=time.perf_counter()-tick,parameters_sha256=model_identity(model),checkpoint_sha256=identity['sha256'])
    teacher=load_teacher(WORKSPACE/'rc3_rl_g2/outputs/teacher_extra/teacher.pkl')['samples']
    indices={c['case_id']:teacher_index(teacher,c) for c in p['cases']}
    output.mkdir();write(output/'model_setup.json',setup)
    write(output/'schedule.json',[dict(slot=i,case_id=c['case_id'],method=m) for i,(c,m) in enumerate(schedule(p['cases']))])
    rows=[]
    with (output/'attempts.jsonl').open('x',encoding='utf-8') as attempts,(output/'results.jsonl').open('x',encoding='utf-8') as results:
        for slot,(case,method) in enumerate(schedule(p['cases'])):
            if slot>=p['max_plans']:raise ValueError('Plan cap exceeded')
            ticket=dict(slot=slot,case_id=case['case_id'],method=method)
            attempts.write(json.dumps(ticket)+'\n');attempts.flush()
            destination=output/f"case_{case['case_id']:02d}_{method}";destination.mkdir()
            started=time.perf_counter()
            try:
                scene=read(WORKSPACE/case['scene_path']);input_s=time.perf_counter()-started
                if method in ('fixed3','detour'):row,evidence=run_original(scene,method,p,started,input_s)
                else:row,evidence=run_shared(scene,method,models.get(method),p,case,indices[case['case_id']],started,input_s)
                if not evidence['plan']['validation'].get('valid_model_plan'):raise ValueError('Final proxy certification failed')
                if method in models and model_identity(models[method])!=setup[method]['parameters_sha256']:
                    raise ValueError('Frozen model parameters changed')
                row.update(status='complete' if not row['failures'] else 'complete_with_error',valid=True,
                    task_count=len(scene['tasks']),table_count=len(scene['tables']),
                    return_vs_stop=None,quality_gain_percent=100*(row['anchor_J']-row['J'])/max(row['anchor_J'],1e-8),
                    loaded_model_s=setup.get(method,{}).get('load_s'),
                    complete_plus_model_load_s=row['complete_s']+setup.get(method,{}).get('load_s',0.),
                    scene_sha256=case['scene_sha256'],mode=case['mode'],**ticket)
                write(destination/'plan.json',evidence['plan']);gzip_write(destination/'evidence.json.gz',evidence)
            except Exception as error:
                row=dict(status='failed',valid=False,error=f'{type(error).__name__}: {error}',complete_s=time.perf_counter()-started,mode=case['mode'],**ticket)
                write(destination/'failure.json',row)
            rows.append(row);results.write(json.dumps(jsonable(row),ensure_ascii=False,allow_nan=False)+'\n');results.flush()
            print(json.dumps({k:row.get(k) for k in ('slot','case_id','method','status','J','return_','proposal_calls','complete_s')}),flush=True)
    if len(rows)!=72:raise ValueError('Incomplete registered schedule')
    for row in rows:
        if row.get('shared_environment'):
            stop=next(r for r in rows if r['case_id']==row['case_id'] and r['method']=='stop')
            if stop['status']!='complete':raise ValueError('Missing actual paired STOP')
            row['return_vs_stop']=row['return_']-stop['return_']
    write(output/'results.json',rows)
    flat=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in rows]
    with (output/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=list(dict.fromkeys(k for r in flat for k in r));writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(flat)
    for name,identity in p['models'].items():
        if sha(WORKSPACE/identity['path'])!=identity['sha256'] or model_identity(models[name])!=setup[name]['parameters_sha256']:
            raise ValueError('Frozen model identity changed')
    write(output/'receipt.json',dict(completed=True,attempted_plans=len(rows),complete_plans=sum(r['status']=='complete' for r in rows),
        valid_plans=sum(r['valid'] for r in rows),protocol_sha256=sha(ROOT/'outputs/protocol.json'),
        proposal_calls=sum(r.get('proposal_calls',0) for r in rows),shared_actions=sum(r.get('steps',0) for r in rows if r.get('shared_environment')),
        model_parameters_unchanged=True,optimizer_updates=0,ppo_actions=0,new_teacher_branches=0,
        scope='Frozen-policy development planning feedback only; no additional branch queries or training'))

if __name__=='__main__':main()
