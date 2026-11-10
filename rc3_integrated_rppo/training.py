"""Matched residual SL, replayable state checkpoints and conditional seed runs."""
import copy
import gc
import math
import random
import time
from .common import *
from .data import base_state
from .model import Actor,Critic,batch_logits
from .ppo import PPO,adam,norm,same
from .rollout import collect,teacher_episode
from .eval import export_actor,evaluate_set,gains


def frame(actor,seed,stage,round_):
    path=OUT/'frames'/str(seed)/stage/f'{round_:04}.pt'
    state={k:v.detach().clone() for k,v in actor.head.state_dict().items()}
    if not path.exists():atomic(path,dict(head=state,base_sha=actor.base_sha,seed=seed,stage=stage,round=round_,experiment=EXP))
    else:
        saved=torch.load(path,map_location='cpu',weights_only=True)
        require(saved['base_sha']==actor.base_sha and same(saved['head'],state),'Rollout frame differs from closed upstream model')
    actor.source_identity=dict(path=label(path),sha256=digest(path));return actor.source_identity


def teacher_batch(actor,indices,inputs):
    out=[]
    for i in indices:
        ref=inputs.ref['teachers'][i];require(digest(ROOT/ref['path'])==ref['sha256'],'Teacher bytes changed')
        d=read(ROOT/ref['path']);s=dict(d['spec']);s['source_family']='bank'
        p,_=inputs.view(s);require(p.identity==d['view_identity'],'Teacher view changed')
        out.append(teacher_episode(actor,p,d['actions']))
    return out


def supervised_step(actor,opt,episodes,log,counts,round_):
    opt.zero_grad(set_to_none=True);loss=0.;decisions=0;correct=0
    for e in episodes:
        if e['batch'] is None:continue
        z=batch_logits(actor,e['batch']);lp=torch.log_softmax(z,-1)
        actions=e['actions'].to(lp.device);term=-lp.gather(1,actions[:,None]).mean()/len(episodes)
        term.backward();loss+=float(term.detach());decisions+=len(z);correct+=int((z.argmax(-1)==actions).sum())
        counts['SL_forwards']=counts.get('SL_forwards',0)+1;counts['SL_backwards']=counts.get('SL_backwards',0)+1
    require(decisions>0 and math.isfinite(loss),'Empty/nonfinite matched SL batch')
    g=norm(actor.head.parameters());require(math.isfinite(g),'Nonfinite SL gradient')
    torch.nn.utils.clip_grad_norm_(actor.head.parameters(),1.,error_if_nonfinite=True)
    log(dict(event='optimizer_attempt',role='residual_SL',round=round_));counts['SL_optimizer_attempts']=counts.get('SL_optimizer_attempts',0)+1
    opt.step();counts['SL_optimizer_steps']=counts.get('SL_optimizer_steps',0)+1
    require(all(bool(torch.isfinite(p).all()) for p in actor.head.parameters()),'Nonfinite matched SL weights')
    return dict(loss=loss,gradient_norm=g,gradient_clipped=g>1.,label_top1=correct/decisions,decisions=decisions)


def value_error(critic,episodes):
    values=[]
    with torch.no_grad():
        for e in episodes:
            if e['batch'] is not None:values.append(float((critic(e['batch']['v'])-e['G']).square().mean()))
    require(values,'No valid critic diagnostic episodes');return float(np.mean(values))


def checkpoint(actor,critic,ppo,control,control_opt,seed,stage,round_,counts,log,clock,extra=None):
    payload=dict(experiment=EXP,seed=seed,stage=stage,closed_round=round_,source_binding_sha256=digest(source_path()),
        protocol_sha256=digest(HOME/'PROTOCOL.json'),base_tensor_sha=actor.base_sha,actor=actor.head.state_dict(),critic=critic.state_dict(),
        actor_opt=ppo.actor_opt.state_dict(),critic_opt=ppo.critic_opt.state_dict(),actor_cursor=ppo.actor_cursor,critic_cursor=ppo.critic_cursor,
        failed_batches=ppo.failed_batches,halved=ppo.halved,control=None if control is None else control.head.state_dict(),
        control_opt=None if control_opt is None else control_opt.state_dict(),counts=copy.deepcopy(counts),
        torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
        log_sequence=len(log.events),log_tail=log.tail,request_count=sum(e['event']=='request_start' for e in log.events),
        optimizer_attempt_count=sum(e['event']=='optimizer_attempt' for e in log.events),wall_used_s=clock.elapsed(),extra=extra or {})
    folder=OUT/'checkpoints'/str(seed)/stage;path=folder/'CURRENT.pt';ref=atomic(path,payload)
    if round_==0 or round_%16==0 or round_ in (64,128,256,512):
        archive=folder/f'closed_{round_:04}.pt'
        if not archive.exists():
            # Same exact serialized checkpoint bytes; a checkpoint is not an actor-only package.
            archive.write_bytes(path.read_bytes())
        else:require(digest(archive)==digest(path),'Closed checkpoint cannot be overwritten')
    log(dict(event='checkpoint_closed',seed=seed,stage=stage,round=round_,checkpoint=ref,actor_cursor=ppo.actor_cursor,
        critic_cursor=ppo.critic_cursor,base_sha=actor.base_sha,actor_tensor_sha=tensor_digest(actor.head.state_dict())))
    return ref


def restore(seed,stage,actor,critic,ppo,control,control_opt,log,counts):
    path=OUT/'checkpoints'/str(seed)/stage/'CURRENT.pt'
    if not path.exists():return None
    matches=[e for e in log.events if e['event']=='checkpoint_closed' and e['seed']==seed and e['stage']==stage
        and e['checkpoint']['path']==label(path) and e['checkpoint']['sha256']==digest(path)]
    require(len(matches)==1,'CURRENT not uniquely sealed by its closed-checkpoint event')
    d=torch.load(path,map_location='cpu',weights_only=False)
    require(d['experiment']==EXP and d['seed']==seed and d['stage']==stage and d['source_binding_sha256']==digest(source_path())
        and d['protocol_sha256']==digest(HOME/'PROTOCOL.json') and d['base_tensor_sha']==actor.base_sha,'Checkpoint scientific source mismatch')
    require(d['closed_round']==matches[0]['round'] and d['actor_cursor']==matches[0]['actor_cursor'] and d['critic_cursor']==matches[0]['critic_cursor'],
        'Closed checkpoint cursor/round mismatch')
    n=d['log_sequence'];require(n<=len(log.events) and (log.events[n-1]['sha256'] if n else '0'*64)==d['log_tail'],'Checkpoint not in current immutable ledger')
    # Paid queries after a closed round may be reconstructed. A dirty optimizer
    # transaction is never silently replayed or refunded from an older checkpoint.
    require(sum(e['event']=='optimizer_attempt' for e in log.events)==d['optimizer_attempt_count'],'Unclosed optimizer boundary; safe resume refused')
    actor.head.load_state_dict(d['actor']);critic.load_state_dict(d['critic']);ppo.actor_opt.load_state_dict(d['actor_opt']);ppo.critic_opt.load_state_dict(d['critic_opt'])
    ppo.actor_cursor=d['actor_cursor'];ppo.critic_cursor=d['critic_cursor'];ppo.failed_batches=d['failed_batches'];ppo.halved=d['halved']
    if control is not None:
        require(d['control'] is not None,'Missing matched control');control.head.load_state_dict(d['control']);control_opt.load_state_dict(d['control_opt'])
    counts.update(d['counts']);torch.set_rng_state(d['torch_rng']);np.random.set_state(d['numpy_rng']);random.setstate(d['python_rng'])
    require(same(actor.head.state_dict(),d['actor']) and same(ppo.actor_opt.state_dict(),d['actor_opt']),'Checkpoint restore changed tensors/Adam')
    return d


def batch_rollouts(actor,critic,seed,stage,round_,specs,kind,inputs,store,clock,*,scale=True):
    frame(actor,seed,stage,round_);episodes=[]
    for slot,spec in enumerate(specs):
        p,_=inputs.view(spec)
        for row in range(2):
            q=dict(kind=kind,stage=stage,seed=seed,round=round_,slot=slot,row=row,role='RL',input_spec_sha256=identity(spec))
            if stage in ('no_anchor_KL','no_task_scale'):q['sampling_stage']='main'
            e=collect(actor,critic,p,q,store,clock,scale_tasks=scale);e['size']=f"{spec['width']}x{spec['height']}";episodes.append(e)
    return episodes


def warm_seed(actor,critic,ppo,seed,inputs,store,clock,counts,stage='main',scale=True):
    result_path=OUT/'seeds'/str(seed)/stage/'WARM_COMPLETE.json'
    if result_path.exists():
        info=read(result_path);path=ROOT/info['critic_state']['path'];require(digest(path)==info['critic_state']['sha256'],'Warm state changed')
        require(any(e['event']=='warm_complete' and e['seed']==seed and e['stage']==stage and e['receipt_sha256']==digest(result_path) for e in store.log.events),
            'Warm completion not sealed; no free replay')
        s=torch.load(path,map_location='cpu',weights_only=True);critic.load_state_dict(s['critic']);ppo.critic_opt.load_state_dict(s['optimizer']);ppo.critic_cursor=s['cursor'];counts.update(s['counts']);return info
    episodes=[]
    for r,specs in enumerate(inputs.declared['warm'][str(seed)]):
        batch=batch_rollouts(actor,critic,seed,stage+'_warm',r,specs,'warm',inputs,store,clock,scale=scale)
        for e in batch:
            e['batch']=None if e['batch'] is None else dict(v=e['batch']['v']);e.pop('old_logp_all');e.pop('base_logp')
        episodes.extend(batch)
    valid=set(inputs.declared['warm_valid_parent_ids']);train=[];val=[]
    for e in episodes:
        spec=e['record']['request'];r=spec['round'];slot=spec['slot'];parent=inputs.declared['warm'][str(seed)][r][slot]['parent']
        (val if parent in valid else train).append(e)
    require(train and val,'Warm physical split is empty')
    initial=value_error(critic,val);steps=[]
    for step in range(224):
        clock.check();batch=[train[(step*24+i)%len(train)] for i in range(24)];steps.append(ppo.critic_step(batch))
    training=value_error(critic,train);validation=value_error(critic,val);extra=0
    if validation>initial or validation>1.5*training:
        for epoch in range(2):
            for start in range(0,len(train),24):
                if len(steps)>=256:break
                clock.check();steps.append(ppo.critic_step(train[start:start+24]));extra+=1
    info=dict(seed=seed,episodes=192,training_episodes=len(train),valid_episodes=len(val),physical_split=True,
        initial_valid_MSE=initial,train_MSE=value_error(critic,train),valid_MSE=value_error(critic,val),critic_steps=len(steps),
        extra_steps=extra,actor_steps=0,step_diagnostics=steps,source_binding_sha256=digest(source_path()))
    require(all(torch.count_nonzero(v)==0 for v in actor.head[-1].state_dict().values()),'Warm changed zero residual')
    info['critic_state']=atomic(result_path.with_suffix('.pt'),dict(critic=critic.state_dict(),optimizer=ppo.critic_opt.state_dict(),cursor=ppo.critic_cursor,counts=counts.copy()))
    write_new(result_path,info);store.log(dict(event='warm_complete',seed=seed,stage=stage,receipt_sha256=digest(result_path)));return info


def selection(seed,round_,actor,control,inputs,store,clock):
    entries={role:export_actor(a,seed,role,round_,dict(stage='main',closed_round=round_)) for role,a in [('RL',actor),('residual_SL',control)]}
    roles=[(seed,k,v) for k,v in entries.items()]
    rows=evaluate_set(inputs.declared['selection'],roles,'selection','selection',round_,store,inputs,clock)
    return entries,rows


def seed_run(seed,inputs,store,clock,*,variant='main',rounds=512,anchor=True,scale=True):
    counts={};state,source=base_state(seed);actor=Actor(state,seed,TRAIN_DEVICE);critic=Critic(seed);control=Actor(state,seed,TRAIN_DEVICE) if variant=='main' else None
    ppo=PPO(actor,critic,store.log,counts,anchor=anchor,context=dict(seed=seed,stage=variant));control_opt=adam(control.head.parameters(),3e-5) if control is not None else None
    d=restore(seed,variant,actor,critic,ppo,control,control_opt,store.log,counts)
    folder=OUT/'seeds'/str(seed)/variant
    if d is None:
        require(not any(e['event']=='optimizer_attempt' and e.get('seed')==seed and e.get('stage')==variant for e in store.log.events),'Unknown optimizer source')
        if variant=='main':
            warm=warm_seed(actor,critic,ppo,seed,inputs,store,clock,counts)
        else:
            # Identical pi0 warm states/actions only; saved rewards remain raw.
            warm=warm_variant(actor,critic,ppo,seed,inputs,store,clock,scale)
        checkpoint(actor,critic,ppo,control,control_opt,seed,variant,0,counts,store.log,clock,dict(warm=warm))
        start=1
    else:start=d['closed_round']+1
    base_entry=export_actor(actor,seed,'SL256',256,source)
    if variant=='main':
        sl_rows=evaluate_set(inputs.declared['selection'],[(seed,'SL256',base_entry)],'selection','baseline',256,store,inputs,clock)
    curve=[];entries={};checks={};stopped_early=False;completed_round=d['closed_round'] if d is not None else 0
    # A closed training checkpoint may precede its deterministic selection call.
    if variant=='main' and d is not None and d['closed_round'] in (64,128,256,512):
        r0=d['closed_round'];found=any(e['event']=='selection_complete' and e['seed']==seed and e['round']==r0 for e in store.log.events)
        if not found:
            ent,rr=selection(seed,r0,actor,control,inputs,store,clock);allrows=sl_rows+rr
            sz={str(size):gains([x for x in allrows if (x['spec']['width'],x['spec']['height'])==size],'SL256',seed) for size in ((4,3),(6,6),(8,8))}
            store.log(dict(event='selection_complete',seed=seed,round=r0,entries=ent,checks=dict(gain_SL=gains(allrows,'SL256',seed),size_gains=sz)))
        if seed==SEEDS[0] and r0==128:
            prior={str(e['round']):e['checks'] for e in store.log.events if e['event']=='selection_complete' and e['seed']==seed}
            bad=lambda c:c['gain_SL']<-.02 and all(v<0 for v in c['size_gains'].values())
            if bad(prior['64']) and bad(prior['128']):stopped_early=True;start=rounds+1
    for r in range(start,rounds+1):
        tick=time.perf_counter();clock.check();completed_round=r
        episodes=batch_rollouts(actor,critic,seed,variant,r,inputs.declared['training'][str(seed)][r-1],
            'train' if variant=='main' else 'ablation',inputs,store,clock,scale=scale)
        # Old likelihoods/values/features are detached and identical for all epochs.
        outcome=ppo.actor_epochs(episodes,clock,f'{seed}:{variant}:{r}')
        records=[e['record'] for e in episodes]
        bysize={}
        for slot in range(12):
            s=inputs.declared['training'][str(seed)][r-1][slot];k=f"{s['width']}x{s['height']}"
            bysize.setdefault(k,[]).extend([records[2*slot]['result']['J'],records[2*slot+1]['result']['J']])
        row=dict(seed=seed,variant=variant,round=r,raw_J=float(np.mean([e['result']['raw_J'] for e in records])),
            H_J=float(np.mean([e['result']['J'] for e in records])),raw_L=float(np.mean([e['result']['raw_L'] for e in records])),
            H_L=float(np.mean([e['result']['L'] for e in records])),raw_K=float(np.mean([e['result']['raw_K'] for e in records])),
            H_K=float(np.mean([e['result']['K'] for e in records])),cost_by_size={k:float(np.mean(v)) for k,v in bysize.items()},
            mean_steps=float(np.mean([e['policy_steps'] for e in records])),PPO=outcome)
        del episodes;gc.collect()
        sl=[]
        if control is not None and outcome['accepted_epochs']:
            examples=teacher_batch(control,inputs.declared['SL_batches'][str(seed)][r-1],inputs)
            for _ in range(outcome['accepted_epochs']):clock.check();sl.append(supervised_step(control,control_opt,examples,store.log,counts,r))
            del examples;gc.collect()
        row.update(matched_SL=sl,elapsed_s=time.perf_counter()-tick,cumulative_s=clock.elapsed(),counts=counts.copy())
        store.log(dict(event='round_complete',**row));curve.append(row)
        checkpoint(actor,critic,ppo,control,control_opt,seed,variant,r,counts,store.log,clock)
        if variant=='main' and r in (64,128,256,512):
            ent,rr=selection(seed,r,actor,control,inputs,store,clock);entries[str(r)]=ent
            allrows=sl_rows+rr;sz={}
            for size in ((4,3),(6,6),(8,8)):
                subset=[x for x in allrows if tuple(x['spec'][k] for k in ('width','height'))==size];sz[str(size)]=gains(subset,'SL256',seed)
            checks[str(r)]=dict(gain_SL=gains(allrows,'SL256',seed),size_gains=sz)
            store.log(dict(event='selection_complete',seed=seed,round=r,entries=ent,checks=checks[str(r)]))
            if r==128 and seed==SEEDS[0]:
                # Rebuild earlier diagnostics from immutable events on recovery.
                for e in store.log.events:
                    if e['event']=='selection_complete' and e['seed']==seed:checks[str(e['round'])]=e['checks']
                bad=lambda c:c['gain_SL']<-.02 and all(v<0 for v in c['size_gains'].values())
                if bad(checks['64']) and bad(checks['128']):stopped_early=True;break
        print(f'{seed} {variant} round={r} J_H={row["H_J"]:.4f} actor_steps={ppo.actor_cursor} wall={clock.elapsed():.1f}',flush=True)
    # Recoverable selection artifacts are ledger references, not best-so-far weights.
    if variant=='main':
        events=[e for e in store.log.events if e['event']=='selection_complete' and e['seed']==seed]
        for e in events:entries[str(e['round'])]=e['entries'];checks[str(e['round'])]=e['checks']
    info=dict(seed=seed,variant=variant,status='STOPPED_PREDECLARED_EARLY' if stopped_early else 'COMPLETE',rounds=completed_round,
        actor_accepted_steps=ppo.actor_cursor,critic_steps=ppo.critic_cursor,matched_SL_steps=counts.get('SL_optimizer_steps',0),
        counts=counts,candidates=entries,checks=checks,base_entry=base_entry,source_model=source,
        final_actor=export_actor(actor,seed,'RL' if variant=='main' else variant,completed_round,dict(stage=variant,closed_round=completed_round)),wall_used_s=clock.elapsed())
    if not (folder/'COMPLETE.json').exists():
        write_new(folder/'COMPLETE.json',info)
        store.log(dict(event='seed_stage_complete',seed=seed,stage=variant,round=completed_round,receipt=label(folder/'COMPLETE.json'),receipt_sha256=digest(folder/'COMPLETE.json')))
    return info


def read_complete(seed,stage,log):
    p=OUT/'seeds'/str(seed)/stage/'COMPLETE.json'
    if not p.exists():return None
    matches=[e for e in log.events if e['event']=='seed_stage_complete' and e['seed']==seed and e['stage']==stage and e['receipt']==label(p) and e['receipt_sha256']==digest(p)]
    require(len(matches)==1,'Stage completion not sealed; cannot skip or replay its optimizers')
    d=read(p);require(d['seed']==seed and d['variant']==stage and d['rounds']==matches[0]['round'],'Stage completion source conflict')
    return d


def warm_variant(actor,critic,ppo,seed,inputs,store,clock,scale):
    episodes=[]
    frame(actor,seed,'ablation_warm_replay',0)
    for r,specs in enumerate(inputs.declared['warm'][str(seed)]):
        for slot,s in enumerate(specs):
            p,_=inputs.view(s)
            for row in range(2):
                q=dict(kind='warm',stage='main_warm',seed=seed,round=r,slot=slot,row=row,role='RL',input_spec_sha256=identity(s))
                require(any(identity(q)+':'+str(a) in store.done for a in (0,1)),'Ablation cannot request new warm trajectories')
                e=collect(actor,critic,p,q,store,clock,scale_tasks=scale)
                if not scale:
                    from .ppo import returns_to_go
                    e['G']=returns_to_go(e['record']['rewards'],1)
                e['batch']=None if e['batch'] is None else dict(v=e['batch']['v']);e.pop('old_logp_all');e.pop('base_logp');episodes.append(e)
    valid=set(inputs.declared['warm_valid_parent_ids']);train=[];val=[]
    for e in episodes:
        q=e['record']['request'];parent=inputs.declared['warm'][str(seed)][q['round']][q['slot']]['parent'];(val if parent in valid else train).append(e)
    initial=value_error(critic,val);steps=[]
    for i in range(224):clock.check();steps.append(ppo.critic_step([train[(i*24+j)%len(train)] for j in range(24)]))
    if value_error(critic,val)>initial or value_error(critic,val)>1.5*value_error(critic,train):
        for _ in range(2):
            for i in range(0,len(train),24):
                if len(steps)>=256:break
                clock.check();steps.append(ppo.critic_step(train[i:i+24]))
    return dict(reused_pi0_episodes=192,new_terminal_requests=0,critic_steps=len(steps),scaled=scale,initial_valid_MSE=initial,valid_MSE=value_error(critic,val))
