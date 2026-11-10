"""Independent real 206-dimensional PPO entry, requiring exact stage approval.

Run ``--help`` for the interface.  Proposal generation does not authorize an
optimizer. This entry never invokes an old trainer or changes its active pointer.
"""
from __future__ import annotations

import time
ENTRY_START = time.perf_counter()

import argparse
import copy
from dataclasses import asdict
import gc
import hashlib
import os
import platform
from pathlib import Path
import random
import time

import numpy as np
import torch

from .adapter import pack_candidate_states
from .analysis import critic_calibration, csv_new, parent_contrast, summarize
from .guard import GuardLimits
from .model import RemainingCostCritic
from .runtime import (load_actor, load_frozen_sl, load_deployed_actor, export_actor,
                      plan_request, sl_route, tensor_digest, SL_PATH, SL_SHA256)
from .saved_inputs import SavedInputs, ANCHORS
from .storage import ROOT, Ledger, read, write_new, digest, identity
from .trainer import ConservativePPO, PPOConfig

HOME = Path(__file__).resolve().parent
STAGE = HOME/'integration_20261110'
H_ID = '5db67cacc1128b6143c7a04b9b84a70c37bdc0ab917a0eebae0b8aca132c4d14'
RUN_ID = 'conservative_paired_11622026_v1'


def configure(seed=11622026):
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def verify_binding(path, sha):
    if digest(path) != sha:
        raise ValueError('Exact source binding changed')
    binding = read(path)
    for name, expected in binding['source_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Bound source changed: '+name)
    for name, expected in binding.get('artifacts', {}).items():
        if digest(ROOT/name) != expected:
            raise ValueError('Bound artifact changed: '+name)
    if binding['H_identity'] != H_ID:
        raise ValueError('Wrong frozen H')
    return binding


def init_trainer(condition, protocol, ledger, role):
    actor=load_actor(condition, protocol['seed'], max_adjustment=protocol['actor']['amplitude'])
    with torch.random.fork_rng():
        torch.manual_seed(protocol['critic']['initialization_seed'])
        critic=RemainingCostCritic()
    def log(row):
        if row['event']=='optimizer_attempt':
            resource='actor_optimizer_attempts' if row['role'] in ('actor','matched_SL') else 'critic_optimizer_attempts'
            ledger.debit({resource:1}, role=role, optimizer_role=row['role'],
                         opportunity=row['opportunity'])
        ledger.emit(dict(row, study_method=role))
    return ConservativePPO(actor,critic,config=PPOConfig(**protocol['ppo']),
                           limits=GuardLimits(**protocol['guard']),log=log,context={'study_method':role})


def collect(ledger, inputs, trainer, spec, request_id, sampling_seed):
    holder={}
    def compute():
        holder['episode']=plan_request(lambda:inputs.view(spec)[0], actor=trainer.actor,
                                       critic=trainer.critic, sampling_seed=sampling_seed,
                                       check=ledger.check)
        return holder['episode']['record']
    ledger.request(request_id,compute,role='training_rollout')
    episode=holder.pop('episode')
    episode['episode_id']=request_id
    episode['record']['request_id']=request_id
    return episode


def teacher_episode(actor, inputs, index, episode_id):
    from rc3_global_local.environment import Environment
    problem, actions, evidence=inputs.teacher(index)
    env=Environment(problem,lambda p,r:p.metrics(r))
    states=[]
    with torch.no_grad():
        encoded=actor.encode(problem) if not env.done else None
        for node in actions:
            if env.done or not env.mask()[node-1]:
                raise ValueError('Existing teacher trajectory is illegal; no relabeling')
            states.append(actor.features(env,encoded))
            env.step(node)
    if not env.done:
        raise ValueError('Existing teacher trajectory is incomplete')
    return dict(batch=pack_candidate_states(states), actions=torch.tensor(actions,dtype=torch.long)-1,
                task_count=len(problem.C), episode_id=episode_id,
                record=dict(input_identity=problem.identity,teacher_source=evidence['teacher_source']))


def checkpoint(ledger, trainers, progress, constants):
    metadata=dict(binding=ledger.binding, journal_sequence=len(ledger.events), journal_tail=ledger.tail,
                  resource_counts=copy.deepcopy(ledger.counts), progress=copy.deepcopy(progress))
    payload=dict(metadata=metadata, constants=copy.deepcopy(constants),
                 trainers={role:t.state_dict(dict(metadata,role=role)) for role,t in trainers.items()})
    folder=ledger.folder/'checkpoints'
    path=folder/f"closed_{len(ledger.events):06}.pt"
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:
        torch.save(payload,f)
        f.flush();os.fsync(f.fileno())
    ledger.emit(dict(event='checkpoint',path=path.relative_to(ROOT).as_posix(),checkpoint_sha256=digest(path),
                     progress=progress,resource_counts=ledger.counts))


def restore(ledger, trainers):
    events=[e for e in ledger.events if e['event']=='checkpoint']
    if not events:
        raise ValueError('No closed checkpoint; recovery cannot restart paid work')
    event=events[-1]
    if any(e['event'] not in ('session_start','session_close') for e in ledger.events[event['sequence']+1:]):
        raise ValueError('Dirty post-checkpoint transaction; automatic replay refused')
    path=ROOT/event['path']
    if digest(path)!=event['checkpoint_sha256']:
        raise ValueError('Closed checkpoint bytes changed')
    payload=torch.load(path,map_location='cpu',weights_only=False)
    expected=dict(binding=ledger.binding,journal_sequence=event['sequence'],journal_tail=event['previous'],
                  resource_counts=ledger.counts,progress=event['progress'])
    if payload['metadata']!=expected or event['resource_counts']!=ledger.counts:
        raise ValueError('Checkpoint cursor/request/optimizer journal differs')
    for role,t in trainers.items():
        t.load_state_dict(payload['trainers'][role],dict(expected,role=role))
    return copy.deepcopy(event['progress']),payload['constants']


def approval(protocol_path, protocol, source_path, protocol_approval, stage_authorization):
    sha=digest(protocol_path)
    p,s=read(protocol_approval),read(stage_authorization)
    shared=dict(status='APPROVED',authorized_by='user',protocol_sha256=sha,
                source_binding_sha256=digest(source_path),experiment=RUN_ID)
    if any(record.get(k)!=v for record in (p,s) for k,v in shared.items()):
        raise ValueError('Exact independent protocol approval and stage authorization required')
    if s.get('caps')!=protocol['caps'] or s.get('stages')!=['warm','train','evaluate']:
        raise ValueError('Stage caps or scope differs from reviewed protocol')
    if s.get('protocol_approval_sha256')!=digest(protocol_approval):
        raise ValueError('Stage receipt does not bind its separate approval')
    for record in (p,s):
        user_source=ROOT/record['user_instruction_path']
        if digest(user_source)!=record['user_instruction_sha256'] or not record.get('explicit_parameters_approved'):
            raise ValueError('Missing actual user approval source; PENDING is not permission')
    return dict(protocol_sha256=sha,protocol_approval_sha256=digest(protocol_approval),
                stage_authorization_sha256=digest(stage_authorization))


def evaluate(ledger, inputs, trainers, cases, endpoint, repeats, constants, output):
    from rc3_global_local_impl_review.endpoint_baselines import run_endpoint
    entries={}
    for role,t in trainers.items():
        entries[role]=export_actor(output/'actors'/f'{role}_{endpoint:04}.pt',t.actor,
                                   dict(experiment=RUN_ID,role=role,opportunity=endpoint,
                                        source_binding_sha256=ledger.binding['source_binding_sha256']))
    rows=[]; calibration=[]
    for repeat in range(repeats):
        for ci,case in enumerate(cases):
            # Fixed, balanced rotating order; independent of observed outcomes.
            roles=['FrozenSL','continued_SL','ON','OFF','Fixed3','Detour16']
            offset=(ci+endpoint+repeat)%len(roles)
            for role in roles[offset:]+roles[:offset]:
                rid=f'eval:{endpoint}:{repeat}:{case["id"]}:{role}'
                holder={}
                def compute():
                    started=time.perf_counter();tick=time.perf_counter()
                    problem,evidence=inputs.view(case['spec'])
                    prep=time.perf_counter()-tick;tick=time.perf_counter()
                    if role in entries:
                        entry=entries[role]
                        model=load_deployed_actor(entry['path'],expected_sha256=entry['sha256'])
                        load_s=time.perf_counter()-tick
                        from .runtime import rollout
                        episode=rollout(problem,model,greedy=True,check=ledger.check)
                        record=episode['record'];holder['episode']=episode
                    elif role=='FrozenSL':
                        model=load_frozen_sl();load_s=time.perf_counter()-tick
                        record=sl_route(problem,model,check=ledger.check)
                    else:
                        load_s=0.;stats={}
                        plan,info=run_endpoint(problem,'fixed3' if role=='Fixed3' else 'detour16',
                                               time_limit_s=max(.001,ledger.caps['wall_s']-ledger.wall_charged-(time.perf_counter()-ledger.started)),
                                               reject_fallback=True,counters=stats)
                        route=info['local_route'];metrics=problem.metrics(route)
                        record=dict(input_identity=problem.identity,plan=plan,raw_plan=plan,route=route,raw_route=route,
                                    metrics=dict(metrics,raw_J=metrics['J'],raw_L=metrics['L'],raw_K=metrics['K']),
                                    counts=stats,internal_search=info['internal_search'],
                                    raw_semantics='Native final route; native already includes its search/H, no extra common H')
                    record.update(case_id=case['id'],cohort=case['group'],spec=case['spec'],physical_key=evidence['physical_key'],
                                  method=role,endpoint=endpoint,repeat=repeat,scope_evidence=evidence,
                                  valid=bool(record['plan']['validation']['valid_model_plan']),
                                  timing=dict(request_before_IO_s=time.perf_counter()-started,
                                              model_load_s=load_s,input_prepare_s=prep),
                                  timer_scope='Cold model, saved navigation input load, all provenance/preparation/provider/features/policy/H/export/validation; filesystem-cache state uncontrolled; record IO reported separately')
                    return record
                record=ledger.request(rid,compute,role=role,endpoint=endpoint)
                rows.append(record)
                if role in ('ON','OFF') and repeat==0:
                    # Critic diagnosis is OUTSIDE deployment request timing and
                    # never passed to the actor-only loader or greedy policy.
                    e=holder['episode'];b=e['batch']
                    tick=time.perf_counter()
                    if b is not None:
                        with torch.no_grad():
                            pred=trainers[role].critic(b['v'],e['task_count'])
                        e['record']['critic_predictions']=[{k:float(v[i]) for k,v in pred.items()} for i in range(len(e['actions']))]
                        for row in critic_calibration([e],constants[role]):
                            row.update(endpoint=endpoint,cohort=case['group'],case_id=case['id'],method=role)
                            calibration.append(row)
                    ledger.emit(dict(event='critic_evaluation',role=role,case_id=case['id'],endpoint=endpoint,
                                     elapsed_s=time.perf_counter()-tick,deployment_timer_includes_critic=False))
                holder.clear();gc.collect()
    summarize(rows,output/f'evaluation_{endpoint:04}')
    csv_new(output/f'evaluation_{endpoint:04}'/'CRITIC_CALIBRATION.csv',calibration)
    return rows,calibration


def run(args):
    protocol_path=Path(args.protocol).resolve();protocol=read(protocol_path)
    source_path=ROOT/protocol['source_binding_path']
    verify_binding(source_path,protocol['source_binding_sha256'])
    engineering=protocol['engineering_gate']
    if digest(ROOT/engineering['resource_path'])!=engineering['resource_sha256'] or digest(ROOT/engineering['checks_path'])!=engineering['checks_sha256']:
        raise ValueError('Engineering acceptance receipt changed')
    resource=read(ROOT/engineering['resource_path']);checks=read(ROOT/engineering['checks_path'])
    if (resource['status']!='PASS' or resource['source_binding_sha256']!=digest(source_path)
            or not checks['initial_on_off_same_correction'] or not checks['forced_rejection_exact']
            or checks['actor_only_exact']['status']!='PASS' or checks['continuous_resume_exact']['status']!='PASS'
            or not checks.get('matched_supervision_real') or checks.get('fixed3_real',{}).get('status')!='PASS'
            or checks.get('detour16_real',{}).get('status')!='PASS'
            or len(checks['zero_exact'])!=4):
        raise ValueError('Exact-source real engineering gate is not PASS')
    authorized=approval(protocol_path,protocol,source_path,args.protocol_approval,args.stage_authorization)
    configure(protocol['seed'])
    actual=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,
                platform=platform.platform(),device='cpu',threads=torch.get_num_threads(),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    if actual!=protocol['environment']:
        raise ValueError('Actual execution environment differs from approved binding')
    inputs=SavedInputs()
    if digest(ROOT/protocol['schedule_path'])!=protocol['schedule_sha256']:
        raise ValueError('Approved exact input/teacher schedule changed')
    schedule=read(ROOT/protocol['schedule_path'])
    expected=[dict(opportunity=i,specs=inputs.training(protocol['seed'],i),
                   teacher_indices=inputs.teacher_indices(protocol['seed'],i)) for i in range(1,257)]
    if schedule['training']!=expected or schedule['evaluation']!=inputs.evaluation_cases or schedule['physical_groups']!=inputs.physical_groups:
        raise ValueError('Runtime data schedule differs from reviewed schedule')
    output=ROOT/protocol['output']
    if output!=STAGE/'runs'/RUN_ID:
        raise ValueError('Output is fixed; moving directories cannot reset budgets')
    binding=dict(experiment=RUN_ID,**authorized,source_binding_sha256=digest(source_path),
                 H_identity=H_ID,seed=protocol['seed'],environment=protocol['environment'])
    ledger=Ledger(output,binding,protocol['caps'],resume=args.resume,started_at=ENTRY_START)
    status='FAILED'
    try:
        trainers={role:init_trainer(role=='ON',protocol,ledger,role) for role in ('ON','OFF','continued_SL')}
        initial={role:tensor_digest(t.actor.correction.state_dict()) for role,t in trainers.items()}
        if len(set(initial.values()))!=1:
            raise ValueError('ON/OFF/control residual initialization differs')
        constants={role:{} for role in ('ON','OFF')}
        progress=dict(warm_batches={'ON':0,'OFF':0},round=0,evaluated=[],quality_no_gain=[],critic_bad=[])
        if args.resume:
            progress,constants=restore(ledger,trainers)
        else:
            checkpoint(ledger,trainers,progress,constants)
        for role in ('ON','OFF'):
            trainer=trainers[role]
            for batch_no in range(progress['warm_batches'][role],4):
                episodes=[collect(ledger,inputs,trainer,spec,f'warm:{role}:{batch_no}:{slot}',
                                  protocol['seed']+100000+batch_no*6+slot)
                          for slot,spec in enumerate(inputs.training(protocol['seed'],batch_no+1))]
                values=constants[role].setdefault('_samples',{'raw_remaining_L':[], 'raw_remaining_K':[], 'terminal_H_credit':[], 'value':[]})
                for e in episodes:
                    values['raw_remaining_L'].extend(e['targets']['raw_remaining_L'])
                    values['raw_remaining_K'].extend(e['targets']['raw_remaining_K'])
                    values['terminal_H_credit'].extend([e['targets']['H_gain']]*len(e['actions']))
                    values['value'].extend(e['targets']['returns'])
                for epoch in range(8):
                    ledger.check();trainer.critic_only_update(episodes)
                for k,v in values.items():
                    constants[role][k]=float(np.mean(v))
                progress['warm_batches'][role]=batch_no+1
                del episodes;gc.collect();checkpoint(ledger,trainers,progress,constants)
        for opportunity in range(progress['round']+1,protocol['opportunities']+1):
            specs=inputs.training(protocol['seed'],opportunity)
            for role in ('ON','OFF'):
                trainer=trainers[role]
                episodes=[collect(ledger,inputs,trainer,spec,f'train:{role}:{opportunity}:{slot}',
                                  protocol['seed']+opportunity*100+slot)
                          for slot,spec in enumerate(specs)]
                trainer.update(episodes)
                del episodes;gc.collect()
            control=trainers['continued_SL'];teachers=[]
            for slot,index in enumerate(inputs.teacher_indices(protocol['seed'],opportunity)):
                ledger.debit({'teacher_episodes':1},opportunity=opportunity,teacher_index=index)
                teachers.append(teacher_episode(control.actor,inputs,index,f'teacher:{opportunity}:{slot}'))
            control.supervised_update(teachers)
            del teachers;gc.collect()
            progress['round']=opportunity
            stop_reasons=[]
            if any(t.counts['consecutive_rejected']>=t.config.consecutive_rejection_limit for t in trainers.values()):
                stop_reasons.append('Sixteen consecutive guard rejections; no threshold relaxation')
            if opportunity in protocol['checkpoints']:
                rows,cal=evaluate(ledger,inputs,trainers,inputs.evaluation_cases,opportunity,
                                  3 if opportunity==256 else 1,constants,output)
                progress['evaluated'].append(opportunity)
                selected=[r for r in rows if r['repeat']==0]
                no_gain=all(parent_contrast(selected,role,'FrozenSL')['point']<=0 for role in ('ON','OFF'))
                progress['quality_no_gain'].append(no_gain)
                bad=[]
                for role in ('ON','OFF'):
                    subset=[r for r in cal if r['method']==role and r['component']=='value']
                    mse=float(np.mean([r['squared_error'] for r in subset]))
                    constant=float(np.mean([r['constant_squared_error'] for r in subset]))
                    bad.append(mse>1.25*max(1e-12,constant))
                progress['critic_bad'].append(bad)
                if len(progress['quality_no_gain'])>=2 and all(progress['quality_no_gain'][-2:]):
                    stop_reasons.append('Two endpoints: neither ON nor OFF improves the frozen supervised start')
                if len(progress['critic_bad'])>=2 and any(all(v[i] for v in progress['critic_bad'][-2:]) for i in (0,1)):
                    stop_reasons.append('Two endpoints: Critic MSE exceeds fixed warm constant by25%')
                ledger.emit(dict(event='endpoint_decision',opportunity=opportunity,stop_reasons=stop_reasons,
                                 no_gain=no_gain,critic_bad=bad))
            if stop_reasons:
                progress['stop_reasons']=stop_reasons
            checkpoint(ledger,trainers,progress,constants)
            print(f'Closed opportunity {opportunity}; ON/OFF accepted '+str([trainers[r].counts['actor_accepted'] for r in ('ON','OFF')]),flush=True)
            if stop_reasons:
                status='STOPPED_REGISTERED_RULE';break
            if args.pause_after==opportunity:
                status='PAUSED_CLEAN_BOUNDARY';break
        else:
            status='COMPLETE_BOUNDED_CONFIGURATION'
        ledger.emit(dict(event='stage_result',status=status,progress=progress,
                         trainers={role:t.counts for role,t in trainers.items()},input_counts=inputs.counts))
        if status=='PAUSED_CLEAN_BOUNDARY':
            # The result is an administrative message, not new computation.
            # Record a matching checkpoint so the clean resume check is exact.
            checkpoint(ledger,trainers,progress,constants)
    finally:
        ledger.close(status)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',required=True)
    p.add_argument('--protocol-approval',required=True)
    p.add_argument('--stage-authorization',required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--pause-after',type=int,default=None,help='Clean stop after a whole opportunity; does not extend budgets')
    run(p.parse_args())


if __name__=='__main__':
    main()
