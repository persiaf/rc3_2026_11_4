"""Explicitly authorized, single-seed G21 PPO pilot; no implicit validation runs.

No resume or automatic extension. Each attempted run keeps its own worker
ledgers, including committed transitions from a partially failed vector batch.
"""
from __future__ import annotations
import os
import sys
sys.dont_write_bytecode=True
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from collections import deque
import json
from pathlib import Path
import time

import numpy as np

from rc3_rl_g2.checkpoints import sha256, load_model, save_model, capture_rng
from rc3_rl_g2.train_ppo import configure, ppo_update
from .train_supervised import checked_samples, diagnose, read, write

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parent


def distribution(values):
    values=np.asarray(values,dtype=float)
    if not len(values):
        return dict(n=0,mean=None,std=None,min=None,max=None,p10=None,p50=None,p90=None,positive_fraction=None)
    if not np.isfinite(values).all():raise ValueError('Nonfinite raw statistic')
    q=np.quantile(values,[.1,.5,.9])
    return dict(n=int(len(values)),mean=float(values.mean()),std=float(values.std()),
        min=float(values.min()),max=float(values.max()),p10=float(q[0]),p50=float(q[1]),p90=float(q[2]),
        positive_fraction=float((values>0).mean()))


def critic_fit(values,returns):
    values=np.asarray(values,dtype=float).reshape(-1);returns=np.asarray(returns,dtype=float).reshape(-1)
    if values.shape!=returns.shape or not len(values) or not np.isfinite(values).all() or not np.isfinite(returns).all():
        raise ValueError('Finite paired critic predictions and returns required')
    variance=float(np.var(returns))
    return dict(mse=float(np.mean((returns-values)**2)),return_variance=variance,
        explained_variance=None if variance<=1e-12 else float(1-np.var(returns-values)/variance),
        target='Current rollout GAE-lambda return; not exact value or finite teacher Q')


def raw_rollout_metrics(batch, settings):
    from rc3_rl_g2.ppo_math import compute_gae
    advantages,returns=compute_gae(batch['rewards'],batch['values'],batch['next_values'],batch['terminated'],
        settings['gamma'],settings['gae_lambda'])
    raw=advantages.cpu().numpy().reshape(-1);targets=returns.cpu().numpy().reshape(-1)
    kinds=np.asarray(batch['action_kinds']);stop=kinds=='stop'
    if len(raw)!=len(kinds):raise ValueError('Action/advantage ordering mismatch')
    return dict(raw_advantages=dict(all=distribution(raw),stop=distribution(raw[stop]),
                                   continue_actions=distribution(raw[~stop])),
        critic_before=critic_fit(batch['values'],targets)),raw,targets


def opportunity_fraction(diagnostic):
    selected=[r for r in diagnostic['rows'] if r['is_root'] and r['known_positive']]
    if not selected:raise ValueError('Opportunity diagnostic needs known-positive roots')
    return sum(r['full_known_positive_continue'] for r in selected)/len(selected)


def collapse_decision(recent_kinds,current_detection,initial_detection):
    if len(recent_kinds)!=1024:raise ValueError('Collapse monitor requires latest 1024 committed actions')
    non_stop=sum(k!='stop' for k in recent_kinds)
    no_improvement=current_detection<=initial_detection+1e-12
    vanished=non_stop/1024<=.01
    return dict(stop=no_improvement and vanished,nonstop_last1024=non_stop,
        nonstop_fraction_last1024=non_stop/1024,known_positive_detection=current_detection,
        initial_known_positive_detection=initial_detection,failed_to_improve=no_improvement,
        continue_sampling_at_most_one_percent=vanished)


def validate_authorization(auth,protocol,protocol_sha):
    if auth.get('authorized') is not True or auth.get('protocol_sha256')!=protocol_sha:
        raise ValueError('Explicit matching G21 PPO authorization required')
    if auth.get('seed')!=protocol['seed']:raise ValueError('Only the preregistered single seed is allowed')
    maximum=auth.get('actions_max')
    if type(maximum) is not int or maximum not in (1024,2048,4096) or maximum>protocol['ppo']['actions_max']:
        raise ValueError('Authorized cap must be 1024, 2048 or 4096 and cannot exceed protocol')
    gates=auth.get('gates',{})
    if set(gates)!={'toy','opportunity','early_stop_rules_frozen','one_step_control_ready'} or any(v is not True for v in gates.values()):
        raise ValueError('All preregistered readiness gates must be explicitly true')
    artifacts=auth.get('required_artifacts',[])
    if not artifacts or any(set(x)!= {'path','sha256'} for x in artifacts):
        raise ValueError('Hashed gate/baseline evidence artifacts required')
    for key in ('initial_checkpoint','initial_checkpoint_sha256','g2_protocol_sha256'):
        if not isinstance(auth.get(key),str) or not auth[key]:raise ValueError('Missing authorization '+key)
    return maximum


def authorized_inputs():
    protocol_path=ROOT/'outputs/protocol.json';p=read(protocol_path);digest=sha256(protocol_path)
    if digest!=(ROOT/'outputs/protocol.sha256').read_text().strip():raise ValueError('Protocol changed')
    auth_path=ROOT/'outputs/ppo_authorization.json'
    if not auth_path.exists():raise ValueError('PPO is gated: outputs/ppo_authorization.json is absent')
    auth=read(auth_path);maximum=validate_authorization(auth,p,digest)
    for item in auth['required_artifacts']:
        path=(WORKSPACE/item['path']).resolve()
        if not path.is_relative_to(WORKSPACE) or sha256(path)!=item['sha256']:
            raise ValueError('Readiness evidence identity mismatch')
    initial=(WORKSPACE/auth['initial_checkpoint']).resolve()
    if not initial.is_relative_to(WORKSPACE) or sha256(initial)!=auth['initial_checkpoint_sha256']:
        raise ValueError('Selected BC initialization changed')
    g2_path=WORKSPACE/'rc3_rl_g2/outputs/protocol.json'
    if sha256(g2_path)!=auth['g2_protocol_sha256']:raise ValueError('Frozen G2 hyperparameter protocol changed')
    g2=read(g2_path)
    if not read(ROOT/'outputs/toy16/receipt.json')['gate']['passed']:raise ValueError('Actual toy receipt fails gate')
    if g2['config']['potential_weight']!=0:raise ValueError('STOP comparison assumes frozen zero-potential semantics')
    return p,digest,auth,sha256(auth_path),maximum,initial,g2


def train(device='cpu'):
    # Authorization and identities are checked before any worker/environment.
    p,digest,auth,auth_sha,maximum,initial,g2=authorized_inputs()
    import torch
    from .collector_pilot import PilotCollector
    configure(p['seed']);settings=dict(g2['ppo'])
    for key in ('actions','validation_actions','final_checkpoint','no_automatic_extension'):
        settings.pop(key,None)
    batch_size=settings['workers']*settings['rollout_steps']
    if batch_size!=256 or maximum%batch_size:raise ValueError('Frozen G2 rollout is 8 workers x 32 actions')
    out=ROOT/'outputs/ppo_pilot'
    if out.exists():raise FileExistsError('Keep previous pilot records; automatic restart/resume is disabled')
    model,_=load_model(initial,device);model.train()
    optimizer=torch.optim.Adam(model.parameters(),lr=settings['learning_rate'])
    rng=np.random.default_rng(p['seed']+501)
    samples=checked_samples();extra_samples=checked_samples('extra')
    initial_diagnostic=diagnose(model,samples)
    initial_detection=opportunity_fraction(initial_diagnostic)
    # Recheck the operational gate numerically, not just an authorization flag.
    extra_diagnostic=diagnose(model,extra_samples)
    gates=p['opportunity_gate'];roots=[r for r in initial_diagnostic['rows'] if r['is_root']]
    stop_roots=[r for r in roots if r['clear_stop']]
    if (initial_detection<gates['initial_positive_full_known_min_fraction'] or
        opportunity_fraction(extra_diagnostic)<gates['extra_positive_full_known_min_fraction'] or
        not stop_roots or sum(r['full_greedy_stop'] for r in stop_roots)/len(stop_roots)<gates['initial_clear_stop_full_stop_min_fraction'] or
        sum(r['full_greedy_unknown'] for r in roots)/len(roots)>gates['initial_root_full_unknown_max_fraction']):
        raise ValueError('Selected BC does not pass the preregistered full-action opportunity gate')
    out.mkdir(parents=True)
    request=dict(protocol_sha256=digest,authorization_sha256=auth_sha,
        initial_checkpoint_sha256=sha256(initial),g2_protocol_sha256=auth['g2_protocol_sha256'],
        source_sha256={f:sha256(ROOT/f) for f in ('train_ppo_pilot.py','collector_pilot.py','worker_pilot.py')},
        seed=p['seed'],actions_max=maximum,settings=settings,environment_config=g2['config'],
        training_cases=g2['splits']['train'],device=device,prepared_cache='rc3_rl_g21/outputs/prepared_train',
        diagnostics='Existing teacher inputs only; zero diagnostic environment queries',
        additional_rollout_validation=False,automatic_extension=False,automatic_resume=False,
        stop_reference='Analytical root STOP from already built menu; no extra environment action',
        timing_scope='Training episode wall time includes optimizer/worker waits; not deployment planner latency')
    write(out/'request.json',request);write(out/'opportunity_0000.json',initial_diagnostic)
    write(out/'extra_opportunity_0000.json',extra_diagnostic)
    started=time.perf_counter();recent=deque(maxlen=1024);updates=0;actual=0
    total_proposals=0;completed_episodes=0;multi_proposal_episodes=0;early_stop=None;final_checkpoint=None
    try:
        with PilotCollector(g2['splits']['train'],g2['config'],p['seed'],out,
            workers=settings['workers'],actions_max=maximum) as collector:
            with (out/'iterations.jsonl').open('x',encoding='utf-8') as iteration_log, \
                 (out/'episodes.jsonl').open('x',encoding='utf-8') as episode_log, \
                 (out/'advantages.jsonl').open('x',encoding='utf-8') as advantage_log:
                for iteration in range(1,maximum//batch_size+1):
                    batch=collector.collect(model,settings['rollout_steps']);actual=collector.committed
                    raw_metrics,raw_adv,targets=raw_rollout_metrics(batch,settings)
                    for index,transition in enumerate(batch['transitions']):
                        advantage_log.write(json.dumps(dict(committed_index=actual-batch_size+index+1,
                            raw_advantage=float(raw_adv[index]),gae_return=float(targets[index]),
                            value=float(batch['values'].reshape(-1)[index]),
                            old_log_probability=float(batch['old_logp'][index]),**transition),allow_nan=False)+'\n')
                    advantage_log.flush()
                    row=ppo_update(model,optimizer,batch,settings,rng);updates+=row['updates']
                    with torch.no_grad():
                        new_values=[]
                        for start in range(0,batch_size,settings['minibatch']):
                            new_values.extend(model(batch['observations'][start:start+settings['minibatch']])[1].cpu().tolist())
                    recent.extend(batch['action_kinds']);proposals=int(batch['actual_proposal_calls'].sum())
                    total_proposals+=proposals;completed_episodes+=len(batch['episodes'])
                    multi_proposal_episodes+=sum(e['has_at_least_two_proposals'] for e in batch['episodes'])
                    row.update(raw_metrics,critic_after=critic_fit(new_values,targets),iteration=iteration,
                        actual_actions=actual,optimizer_updates=updates,actual_proposal_calls=proposals,
                        cumulative_actual_proposal_calls=total_proposals,
                        completed_episodes=len(batch['episodes']),
                        episodes_with_at_least_two_proposals=sum(e['has_at_least_two_proposals'] for e in batch['episodes']),
                        return_minus_root_stop=distribution([e['return_minus_root_stop'] for e in batch['episodes']]),
                        stop_fraction=batch['action_kinds'].count('stop')/batch_size,
                        changed_fraction=float(batch['changed'].mean()),collect_s=batch['elapsed_s'],
                        inference_s=batch['inference_s'],worker_s=batch['worker_s'],elapsed_s=time.perf_counter()-started)
                    iteration_log.write(json.dumps(row,allow_nan=False)+'\n');iteration_log.flush()
                    for episode in batch['episodes']:
                        episode_log.write(json.dumps(dict(training_actions=actual,**episode),allow_nan=False)+'\n')
                    episode_log.flush()
                    print(json.dumps(dict(stage='g21_ppo_pilot',actions=actual,proposals=proposals,
                        stop_fraction=row['stop_fraction'],critic_mse=row['critic_before']['mse'])),flush=True)
                    if actual in p['ppo']['checkpoints']:
                        diagnostic=diagnose(model,samples);write(out/f'opportunity_{actual:04d}.json',diagnostic)
                        extra=diagnose(model,extra_samples);write(out/f'extra_opportunity_{actual:04d}.json',extra)
                        decision=collapse_decision(list(recent),opportunity_fraction(diagnostic),initial_detection)
                        write(out/f'collapse_{actual:04d}.json',decision)
                        final_checkpoint=out/f'ppo_{actual:04d}.pt'
                        save_model(final_checkpoint,model,protocol_sha256=digest,authorization_sha256=auth_sha,
                            parent_bc_sha256=sha256(initial),actual_actions=actual,iteration=iteration,
                            optimizer_updates=updates,optimizer=optimizer.state_dict(),workers=collector.state_dict(),
                            numpy_rng=rng.bit_generator.state,torch_rng=capture_rng(device),pilot_source_sha256=request['source_sha256'])
                        if decision['stop']:
                            early_stop=decision;break
            live=collector.state_dict()
            if sum(s['committed'] for s in live)!=actual:
                raise AssertionError('Worker committed totals disagree with collector')
        receipt=dict(completed=True,actual_actions=actual,actions_max=maximum,optimizer_updates=updates,
            seed=p['seed'],early_stop=early_stop,actual_proposal_calls=total_proposals,
            completed_episodes=completed_episodes,episodes_with_at_least_two_proposals=multi_proposal_episodes,
            live_unfinished_episodes=sum(not s['env_state']['terminated'] and s['env_state']['steps']>0 for s in live),
            final_checkpoint=str(final_checkpoint.relative_to(WORKSPACE)),final_checkpoint_sha256=sha256(final_checkpoint),
            initial_checkpoint_sha256=sha256(initial),protocol_sha256=digest,authorization_sha256=auth_sha,
            elapsed_s=time.perf_counter()-started,on_policy=True,teacher_used_in_ppo_updates=False,
            extra_environment_validation_actions=0,automatic_extension=False,automatic_resume=False,
            interpretation='One-seed bounded pilot; diagnostic teacher sets previously used in development; no superiority claim from training returns')
        write(out/'receipt.json',receipt);return receipt
    except Exception as error:
        write(out/'failure.json',dict(completed=False,collector_acknowledged_actions=actual,
            error=repr(error),scope='Inspect all worker committed JSONL files for partial in-flight batches; no automatic resume'))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args();print(json.dumps(train(args.device)))
