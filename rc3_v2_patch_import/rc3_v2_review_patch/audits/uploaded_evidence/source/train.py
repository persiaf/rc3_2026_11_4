"""Gain-ranked BC followed by genuine multi-step on-policy clipped PPO."""
import argparse,hashlib,json,pickle,time,copy
from pathlib import Path
import numpy as np
import torch
from torch.distributions import Categorical
from rc3_baseline.baseline import write_json,write_csv
from rc3_fast.solver import route_objective
from .environment import ImprovementEnv
from .learning import Policy,batch_states,save_policy,load_policy


def collect(model,problems,horizon,greedy=False):
    envs=[ImprovementEnv(p,horizon) for p in problems]
    episodes=[[] for _ in envs]
    model.eval();device=model.mean.device
    while True:
        observations=[e.observe() for e in envs]
        active=[i for i,e in enumerate(envs) if not e.done]
        if not active:break
        states=[observations[i] for i in active]
        x,mask=batch_states(states,device)
        with torch.no_grad():
            logits,values=model(x,mask);dist=Categorical(logits=logits)
            actions=logits.argmax(-1) if greedy else dist.sample()
            logp=dist.log_prob(actions).cpu().numpy();values=values.cpu().numpy();actions=actions.cpu().numpy()
        for j,i in enumerate(active):
            reward,done,info=envs[i].step(int(actions[j]))
            episodes[i].append(dict(state=states[j],action=int(actions[j]),logp=float(logp[j]),
                                    value=float(values[j]),reward=float(reward),accepted=info['accepted']))
    gains=[100*(e.reference_cost-route_objective(e.problem,e.route))/max(e.reference_cost,1e-8) for e in envs]
    return episodes,gains


def advantages(episodes,gamma=1.,lam=.95):
    records=[]
    for episode in episodes:
        gae,next_value=0.,0.
        for step in reversed(episode):
            delta=step['reward']+gamma*next_value-step['value']
            gae=delta+gamma*lam*gae
            step['advantage']=gae;step['return']=gae+step['value']
            next_value=step['value']
        records.extend(episode)
    return records


def ppo_update(model,optimizer,records,rng,epochs=4,batch_size=128):
    if not records:return dict(loss=0.,kl=0.,updates=0)
    device=model.mean.device
    adv=np.asarray([r['advantage'] for r in records],np.float32)
    adv=(adv-adv.mean())/max(float(adv.std()),1e-6)
    loss_values=[];kls=[];updates=0
    model.train()
    for _ in range(epochs):
        for ids in np.array_split(rng.permutation(len(records)),int(np.ceil(len(records)/batch_size))):
            rows=[records[i] for i in ids]
            x,mask=batch_states([r['state'] for r in rows],device)
            logits,value=model(x,mask);dist=Categorical(logits=logits)
            actions=torch.tensor([r['action'] for r in rows],device=device)
            old=torch.tensor([r['logp'] for r in rows],device=device)
            returns=torch.tensor([r['return'] for r in rows],device=device)
            advantage=torch.tensor(adv[ids],device=device)
            logp=dist.log_prob(actions);ratio=(logp-old).exp()
            policy=-torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean()
            loss=policy+.5*(value-returns).square().mean()-.01*dist.entropy().mean()
            if not torch.isfinite(loss):raise ValueError('nonfinite PPO loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()
            loss_values.append(float(loss.detach()));kls.append(float((old-logp).mean().detach()));updates+=1
    return dict(loss=float(np.mean(loss_values)),kl=float(np.mean(kls)),updates=updates)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bc-epochs',type=int,default=100);p.add_argument('--rl-epochs',type=int,default=30)
    p.add_argument('--episodes',type=int,default=32);p.add_argument('--horizon',type=int,default=8)
    p.add_argument('--seeds',nargs='+',type=int,default=[17,29,43]);p.add_argument('--device',default='cuda')
    a=p.parse_args()
    if a.output.exists():raise ValueError('use a new output directory')
    a.output.mkdir(parents=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    # The pickle is our own locally generated cache, never an uploaded pickle.
    with (a.data/'cache.pkl').open('rb') as f:data=pickle.load(f)
    training,validation=data['problems']['train'],data['problems']['validation']
    examples=[e for e in data['labels'] if e['gain'][e['mask']].sum()>1e-8]
    flat=np.concatenate([e['x'] for e in data['labels']])
    mean,std=flat.mean(0),flat.std(0).clip(.05)
    history=[];runs=[];started=time.perf_counter();total_updates=0;total_steps=0
    for seed in a.seeds:
        rng=np.random.default_rng(seed);torch.manual_seed(seed)
        model=Policy(mean,std).to(a.device)
        optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
        best_bc=-np.inf;best_state=None;best_epoch=0
        for epoch in range(a.bc_epochs+1):
            losses=[]
            if epoch:
                model.train()
                for ids in np.array_split(rng.permutation(len(examples)),max(1,int(np.ceil(len(examples)/8)))):
                    rows=[examples[i] for i in ids]
                    x,mask=batch_states([(e['x'],e['mask']) for e in rows],a.device)
                    target=torch.zeros(mask.shape,device=a.device)
                    for i,e in enumerate(rows):
                        y=e['gain']*e['mask'];target[i,:len(y)]=torch.tensor(y/max(y.sum(),1e-8),device=a.device)
                    logits,_=model(x,mask)
                    logp=torch.where(mask,torch.log_softmax(logits,dim=-1),0.)
                    loss=-(target*logp).sum(-1).mean()
                    optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                    losses.append(float(loss.detach()));total_updates+=1
            if epoch%10==0 or epoch==a.bc_epochs:
                _,gains=collect(model,validation,a.horizon,greedy=True)
                score=float(np.mean(gains))
                if score>best_bc:
                    best_bc,best_epoch=score,epoch;best_state=copy.deepcopy(model.state_dict())
                history.append(dict(seed=seed,phase='bc',epoch=epoch,loss=float(np.mean(losses)) if losses else None,val_improvement_pct=score,train_improvement_pct=None,steps=total_steps,elapsed_s=time.perf_counter()-started))
                print(f'seed {seed} BC {epoch} val gain {score:.3f}% best {best_bc:.3f}%',flush=True)
        model.load_state_dict(best_state)
        bc_path=a.output/f'seed{seed}_bc.pt';save_policy(bc_path,model,seed=seed,phase='bc',epoch=best_epoch,horizon=a.horizon)
        optimizer=torch.optim.Adam(model.parameters(),lr=5e-5)
        best_rl=-np.inf;rl_state=None;rl_epoch=None
        for epoch in range(1,a.rl_epochs+1):
            ids=rng.choice(len(training),min(a.episodes,len(training)),replace=False)
            episodes,gains=collect(model,[training[i] for i in ids],a.horizon)
            records=advantages(episodes);total_steps+=len(records)
            stats=ppo_update(model,optimizer,records,rng);total_updates+=stats['updates']
            score=None
            if epoch%5==0 or epoch==1 or epoch==a.rl_epochs:
                _,vg=collect(model,validation,a.horizon,True);score=float(np.mean(vg))
                if score>best_rl:
                    best_rl,rl_epoch=score,epoch;rl_state=copy.deepcopy(model.state_dict())
                print(f'seed {seed} PPO {epoch}/{a.rl_epochs} train gain {np.mean(gains):.3f}% val {score:.3f}% steps {total_steps}',flush=True)
            history.append(dict(seed=seed,phase='ppo',epoch=epoch,loss=stats['loss'],val_improvement_pct=score,train_improvement_pct=float(np.mean(gains)),steps=total_steps,elapsed_s=time.perf_counter()-started))
        model.load_state_dict(rl_state)
        rl_path=a.output/f'seed{seed}_ppo.pt';save_policy(rl_path,model,seed=seed,phase='ppo',epoch=rl_epoch,horizon=a.horizon)
        runs.append(dict(seed=seed,bc_val_gain=best_bc,ppo_val_gain=best_rl,bc_epoch=best_epoch,ppo_epoch=rl_epoch,
                         bc_checkpoint=bc_path.name,ppo_checkpoint=rl_path.name))
        write_csv(a.output/'history.csv',history);write_json(a.output/'progress.json',runs)
    # Freeze BC and PPO separately, so the test can measure added RL contribution.
    bc=max(runs,key=lambda r:r['bc_val_gain']);rl=max(runs,key=lambda r:r['ppo_val_gain'])
    import shutil
    shutil.copy2(a.output/bc['bc_checkpoint'],a.output/'bc_best.pt')
    shutil.copy2(a.output/rl['ppo_checkpoint'],a.output/'ppo_best.pt')
    summary=dict(status='complete',runs=runs,selected_bc_seed=bc['seed'],selected_ppo_seed=rl['seed'],
                 recommended_stage='ppo' if rl['ppo_val_gain']>bc['bc_val_gain'] else 'bc',
                 training_instances=len(training),validation_instances=len(validation),bc_positive_scenes=len(examples),
                 total_optimizer_updates=total_updates,total_environment_steps=total_steps,training_s=time.perf_counter()-started,
                 config=vars(a)|{'data':str(a.data),'output':str(a.output)},
                 torch=torch.__version__,gpu=torch.cuda.get_device_name() if a.device=='cuda' else None,
                 checkpoints={name:hashlib.sha256((a.output/name).read_bytes()).hexdigest() for name in ('bc_best.pt','ppo_best.pt')})
    write_json(a.output/'training_summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
