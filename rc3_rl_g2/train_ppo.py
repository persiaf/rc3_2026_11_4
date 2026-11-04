"""Single-seed BC controls and fresh-interaction PPO; no G0 offline PPO reuse."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parent

def dump(path,value):
    Path(path).parent.mkdir(exist_ok=True,parents=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def protocol():
    raw=(ROOT/'outputs/protocol.json').read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if digest!=(ROOT/'outputs/protocol.sha256').read_text().split()[0]:raise ValueError('protocol changed')
    return json.loads(raw),digest

def configure(seed):
    import torch
    from threadpoolctl import threadpool_limits
    threadpool_limits(1);torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    np.random.seed(seed);torch.manual_seed(seed)

def tensors_equal(a,b):
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(tensors_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(tensors_equal(x,y) for x,y in zip(a,b))
    return a==b

def ppo_update(model,optimizer,batch,settings,rng):
    import torch
    from .ppo_math import compute_gae
    obs=batch['observations'];N=len(obs);device=model.device
    advantages,returns=compute_gae(batch['rewards'],batch['values'],batch['next_values'],batch['terminated'],
                                  settings['gamma'],settings['gae_lambda'])
    advantages=advantages.flatten().to(device);returns=returns.flatten().to(device)
    advantages=(advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
    old=torch.as_tensor(batch['old_logp'],device=device)
    old_error=0.
    with torch.no_grad():
        for start in range(0,N,settings['minibatch']):
            end=min(start+settings['minibatch'],N)
            lp,_,_=model.evaluate_actions(obs[start:end],batch['actions'][start:end])
            old_error=max(old_error,float((lp-old[start:end]).abs().max()))
    if old_error>3e-5:raise AssertionError(f'old/new probability mismatch before update: {old_error}')
    norms=[];kls=[];clips=[];losses=[];updates=0;early_stop=False
    tick=time.perf_counter()
    for epoch in range(settings['epochs']):
        indices=rng.permutation(N)
        for start in range(0,N,settings['minibatch']):
            idx=indices[start:start+settings['minibatch']]
            lp,entropy,values=model.evaluate_actions([obs[int(i)] for i in idx],batch['actions'][idx])
            logratio=lp-old[idx];ratio=logratio.exp()
            kl=float(((ratio-1)-logratio).mean().detach())
            if not np.isfinite(kl):raise AssertionError('nonfinite KL')
            if kl>settings['target_kl']:
                early_stop=True;break
            policy_loss=torch.maximum(-advantages[idx]*ratio,
                -advantages[idx]*ratio.clamp(1-settings['clip'],1+settings['clip'])).mean()
            value_loss=(values-returns[idx]).square().mean()
            loss=policy_loss+settings['value_coefficient']*value_loss-settings['entropy_coefficient']*entropy.mean()
            if not torch.isfinite(loss):raise AssertionError('nonfinite PPO loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),settings['grad_clip'])
            if not torch.isfinite(norm):raise AssertionError('nonfinite gradient')
            optimizer.step();updates+=1
            norms.append(float(norm));kls.append(kl);losses.append(float(loss.detach()))
            clips.append(float(((ratio-1).abs()>settings['clip']).float().mean().detach()))
        if early_stop:break
    return dict(updates=updates,old_new_logp_error=old_error,early_kl_stop=early_stop,
                loss=float(np.mean(losses)) if losses else 0.,approx_kl=float(np.mean(kls)) if kls else kl,
                clip_fraction=float(np.mean(clips)) if clips else 0.,grad_norm_max=max(norms,default=0.),
                optimizer_s=time.perf_counter()-tick,advantages_finite=True,gradients_finite=True)

def pipeline(device='cpu'):
    import torch
    from .actor_critic import ActorCritic
    from .collector import VectorCollector
    from .checkpoints import save_model,capture_rng,restore_rng
    p,digest=protocol();configure(p['training_seed']);reports=[]
    for kind,total in [('mlp',p['pipeline']['mlp_actions']),('graph',p['pipeline']['graph_actions'])]:
        model=ActorCritic(kind=kind,hidden=64,message_steps=2).to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=p['ppo']['learning_rate'])
        rng=np.random.default_rng(p['training_seed']+51);before={k:v.clone() for k,v in model.state_dict().items()}
        rows=[];ep=[];started=time.perf_counter()
        with VectorCollector(p['splits']['pipeline'],p['config'],seed=p['training_seed']+91,workers=8) as collector:
            for iteration in range(total//256):
                batch=collector.collect(model,32);ep.extend(batch['episodes'])
                row=ppo_update(model,optimizer,batch,p['ppo'],rng);row.update(actual_actions=len(batch['actions']),collect_s=batch['elapsed_s'])
                rows.append(row)
                if iteration==0:
                    states=collector.state_dict();raw_before=copy.deepcopy(collector.observations)
                    torch_state=capture_rng(device);expected=model.act(raw_before)
                    collector.load_state_dict(states)
                    if not tensors_equal(raw_before,collector.observations):raise AssertionError('restored raw policy observations changed')
                    restore_rng(torch_state);actual=model.act(collector.observations)
                    if any(not torch.equal(x,y) for x,y in zip(expected,actual)):raise AssertionError('restored policy sampling changed')
                    restore_rng(torch_state)
            changed=any(not torch.equal(v,before[k]) for k,v in model.state_dict().items())
            if not changed:raise AssertionError('pipeline parameters did not update')
            save_model(ROOT/f'outputs/pipeline/{kind}.pt',model,protocol_sha256=digest,actual_actions=total,
                       optimizer=optimizer.state_dict(),workers=collector.state_dict(),torch_rng=capture_rng(device))
        report=dict(kind=kind,actual_actions=total,updates=sum(x['updates'] for x in rows),
            completed_episodes=len(ep),all_episode_ledgers_verified=True,restore_sampling_identical=True,
            parameters_changed=changed,elapsed_s=time.perf_counter()-started,iterations=rows)
        reports.append(report);dump(ROOT/f'outputs/pipeline/{kind}_receipt.json',report)
        print(json.dumps(dict(stage='pipeline',kind=kind,actions=total,updates=report['updates'],passed=True)),flush=True)
    dump(ROOT/'outputs/pipeline/summary.json',dict(all_passed=True,reports=reports,scope='Numerical/interaction correctness only; not superiority evidence'))

def bc_train(device='cpu',continued=False):
    import torch
    from .actor_critic import ActorCritic
    from .teacher_dataset import load_teacher,assign_scene_route_weights
    from .checkpoints import save_model,load_model,sha256,capture_rng
    p,digest=protocol();configure(p['training_seed']);s=p['bc']
    initial_path=ROOT/'outputs/teacher_initial/teacher.pkl';samples=load_teacher(initial_path)['samples']
    initial_checkpoint=ROOT/'outputs/training/bc_initial.pt'
    if continued:
        model,_=load_model(initial_checkpoint,device)
        samples+=load_teacher(ROOT/'outputs/teacher_extra/teacher.pkl')['samples']
        assign_scene_route_weights(samples)
        steps=s['continued_updates'];checkpoints=s['continued_checkpoints'];prefix='bc_continued'
    else:
        model=ActorCritic(kind='graph',hidden=64,message_steps=2).to(device)
        steps=s['initial_updates'];checkpoints=[steps];prefix='bc_initial'
    optimizer=torch.optim.Adam(model.parameters(),lr=s['learning_rate'])
    rng=np.random.default_rng(p['training_seed']+203+(1 if continued else 0))
    weights=np.asarray([x['weight'] for x in samples],dtype=float);weights/=weights.sum()
    rows=[];started=time.perf_counter();out=ROOT/'outputs/training';out.mkdir(exist_ok=True)
    for step in range(1,steps+1):
        idx=rng.choice(len(samples),size=s['batch_size'],replace=True,p=weights);batch=[samples[int(i)] for i in idx]
        logits,_=model([r['observation'] for r in batch]);losses=[]
        for j,row in enumerate(batch):
            n=len(row['evidence_mask']);evidence=torch.as_tensor(row['evidence_mask'],device=model.device,dtype=torch.bool)
            ties=torch.as_tensor(row['ties'],device=model.device,dtype=torch.bool)
            if ties.shape!=(n,) or not ties.any() or (ties&~evidence).any():raise ValueError('teacher tie/evidence mismatch')
            logp=logits[j,:n].masked_fill(~evidence,-torch.inf).log_softmax(0)
            losses.append(-logp[ties].mean())
        loss=torch.stack(losses).mean()
        if not torch.isfinite(loss):raise AssertionError('nonfinite BC loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.5)
        if not torch.isfinite(norm):raise AssertionError('nonfinite BC gradient')
        optimizer.step()
        rows.append(dict(update=step,loss=float(loss.detach()),grad_norm=float(norm)))
        if step%100==0:print(json.dumps(dict(stage=prefix,updates=step,loss=rows[-1]['loss'],elapsed_s=time.perf_counter()-started)),flush=True)
        if step in checkpoints:
            path=initial_checkpoint if not continued else out/f'{prefix}_{step:04d}.pt'
            save_model(path,model,protocol_sha256=digest,updates=step,optimizer=optimizer.state_dict(),
                teacher_initial_sha256=sha256(initial_path),
                teacher_extra_sha256=sha256(ROOT/'outputs/teacher_extra/teacher.pkl') if continued else None,
                parent_bc_sha256=sha256(initial_checkpoint) if continued else None,
                torch_rng=capture_rng(device),numpy_rng=rng.bit_generator.state)
    dump(out/f'{prefix}_training.json',dict(updates=steps,samples=len(samples),unique_scenes=len({r['scene_id'] for r in samples}),
        elapsed_s=time.perf_counter()-started,rows=rows,device=device,actor_only=True,critic_supervised=False,
        scope='Masked supervised actor loss over queried evidence only; no unknown negative labels'))

def evaluate_checkpoint(model,step,p):
    from .evaluate_g2 import evaluate_methods,_model_identity
    from rc3_rl_research.types import SearchConfig
    destination=ROOT/f'outputs/validation/ppo_{step}'
    if (destination/'results.json').exists():
        rows=json.loads((destination/'results.json').read_text(encoding='utf-8'))
        if len(rows)!=len(p['splits']['validation']) or any(r['status']!='complete' for r in rows):
            raise ValueError('saved checkpoint validation is incomplete or failed')
        request=json.loads((destination/'request.json').read_text(encoding='utf-8'))
        if request.get('cases')!=p['splits']['validation'] or request.get('config')!=p['config'] or request.get('time_limit_s') is not None:
            raise ValueError('saved validation scene/config/budget identity differs')
        expected=_model_identity(model)
        # Require the exact semantic model identity somewhere in the recorded request.
        if expected['sha256'] not in json.dumps(request,sort_keys=True):raise ValueError('saved validation model identity differs')
        return rows
    return evaluate_methods(p['splits']['validation'],{f'ppo_{step}':model},destination,
                            config=SearchConfig(**p['config']))

def train(device='cpu',resume=None):
    import torch
    from .collector import VectorCollector
    from .checkpoints import save_model,load_model,sha256,capture_rng,restore_rng,validate_training_state
    p,digest=protocol();configure(p['training_seed']);settings=p['ppo']
    initial=ROOT/'outputs/training/bc_initial.pt';model,_=load_model(initial,device)
    optimizer=torch.optim.Adam(model.parameters(),lr=settings['learning_rate'])
    rng=np.random.default_rng(p['training_seed']+501);start_iteration=0;total_updates=0
    out=ROOT/'outputs/training';episodes_file=out/'ppo_episodes.jsonl';metrics_file=out/'ppo_iterations.jsonl'
    pending=None;base_elapsed=0.;validation_s=0.
    if resume:
        model,pending=load_model(resume,device)
        if pending['protocol_sha256']!=digest or pending['parent_bc_sha256']!=sha256(initial):raise ValueError('resume identities mismatch')
        optimizer=torch.optim.Adam(model.parameters(),lr=settings['learning_rate']);optimizer.load_state_dict(pending['optimizer'])
        if torch.device(pending['training_device']).type!=torch.device(device).type:
            raise ValueError('exact training resume requires the original device type')
        rng.bit_generator.state=pending['numpy_rng'];restore_rng(pending['torch_rng'])
        start_iteration=pending['iteration'];total_updates=pending['optimizer_updates']
        recorded=[]
        if metrics_file.exists():
            recorded=[json.loads(line) for line in metrics_file.read_text(encoding='utf-8').splitlines() if line.strip()]
        validate_training_state(pending,p,sha256(initial),recorded[-1] if recorded else None)
        base_elapsed=pending.get('elapsed_s',0.);validation_s=pending.get('validation_s',0.)
    elif metrics_file.exists():raise ValueError('formal training already has records; use explicit --resume')
    settings_path=out/'execution_settings.json'
    if not settings_path.exists():dump(settings_path,dict(device=device,torch_threads=1,numpy_worker_threads=1,
        workers=settings['workers'],adapter_lru_per_worker=1,
        decision='CPU model chosen from pre-training probes because only about 2 GiB host RAM was available; CUDA context added about 1 GiB. GPU update benchmark is retained, not used to claim end-to-end speed.',
        training_case_sampling='Each worker permutes its disjoint registered 48-case subset; next permutation after exhaustion. Fresh adapter if not in one-entry LRU.',
        pipeline_actions_excluded=True))
    started=time.perf_counter()
    if pending is not None:
        preserved_rng=capture_rng(device)
        for boundary in settings['validation_actions']:
            if boundary<=pending['actual_actions']:
                target=out/f'ppo_{boundary:05d}.pt'
                if not target.exists():raise ValueError('reached validation checkpoint is missing')
                validation_model,_=load_model(target,device)
                tick=time.perf_counter();evaluate_checkpoint(validation_model,boundary,p)
                validation_s+=time.perf_counter()-tick
        restore_rng(preserved_rng)
    with VectorCollector(p['splits']['train'],p['config'],seed=p['training_seed'],workers=settings['workers']) as collector:
        if pending is not None:collector.load_state_dict(pending['workers'])
        with metrics_file.open('a',encoding='utf-8') as metrics,episodes_file.open('a',encoding='utf-8') as episodes:
            for iteration in range(start_iteration+1,settings['actions']//256+1):
                batch=collector.collect(model,settings['rollout_steps'])
                row=ppo_update(model,optimizer,batch,settings,rng);total_updates+=row['updates']
                actions=iteration*256
                row.update(iteration=iteration,actual_actions=actions,optimizer_updates=total_updates,
                    rollout_return_mean=float(batch['rewards'].mean()),completed_episodes=len(batch['episodes']),
                    episode_return_mean=float(np.mean([e['return_value'] for e in batch['episodes']])) if batch['episodes'] else None,
                    stop_fraction=batch['action_kinds'].count('stop')/256,
                    changed_fraction=float(batch['changed'].mean()),collect_s=batch['elapsed_s'],
                    inference_s=batch['inference_s'],worker_s=batch['worker_s'],
                    elapsed_s=base_elapsed+time.perf_counter()-started,validation_s=validation_s)
                metrics.write(json.dumps(row,allow_nan=False)+'\n');metrics.flush()
                for ep in batch['episodes']:episodes.write(json.dumps(dict(training_actions=actions,**ep),allow_nan=False)+'\n')
                episodes.flush()
                print(json.dumps(dict(stage='ppo',actions=actions,episodes=row['completed_episodes'],
                    mean_return=row['episode_return_mean'],stop_fraction=row['stop_fraction'],
                    collect_s=row['collect_s'],optimizer_s=row['optimizer_s'],elapsed_s=row['elapsed_s'])),flush=True)
                if actions in settings['validation_actions'] or iteration%8==0:
                    target=out/(f'ppo_{actions:05d}.pt' if actions in settings['validation_actions'] else 'ppo_latest.pt')
                    save_model(target,model,protocol_sha256=digest,parent_bc_sha256=sha256(initial),
                        actual_actions=actions,iteration=iteration,optimizer_updates=total_updates,
                        optimizer=optimizer.state_dict(),workers=collector.state_dict(),
                        numpy_rng=rng.bit_generator.state,torch_rng=capture_rng(device),validation_s=validation_s,
                        elapsed_s=base_elapsed+time.perf_counter()-started)
                if actions in settings['validation_actions']:
                    tick=time.perf_counter();evaluate_checkpoint(model,actions,p);validation_s+=time.perf_counter()-tick
                    save_model(out/f'ppo_{actions:05d}.pt',model,protocol_sha256=digest,parent_bc_sha256=sha256(initial),
                        actual_actions=actions,iteration=iteration,optimizer_updates=total_updates,
                        optimizer=optimizer.state_dict(),workers=collector.state_dict(),numpy_rng=rng.bit_generator.state,
                        torch_rng=capture_rng(device),validation_s=validation_s,elapsed_s=base_elapsed+time.perf_counter()-started)
    dump(out/'ppo_receipt.json',dict(actual_actions=settings['actions'],optimizer_updates=total_updates,
        seed=p['training_seed'],parent_bc_sha256=sha256(initial),protocol_sha256=digest,
        elapsed_s=base_elapsed+time.perf_counter()-started,validation_s=validation_s,device=device,
        completed=True,on_policy=True,pipeline_actions_excluded=True,teacher_used_in_ppo_updates=False,
        automatic_extension=False,final_checkpoint='ppo_32768.pt'))

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--stage',choices=['pipeline','bc','bc-continued','ppo'],required=True)
    a.add_argument('--device',choices=['cpu','cuda'],default='cpu');a.add_argument('--resume',type=Path)
    args=a.parse_args()
    if args.stage=='pipeline':pipeline(args.device)
    elif args.stage=='bc':bc_train(args.device,False)
    elif args.stage=='bc-continued':bc_train(args.device,True)
    else:train(args.device,args.resume)

if __name__=='__main__':main()
