"""Per-action PPO ratios, episode sums, exact masked KL and transactional rollback."""
import copy
import math
import numpy as np
import torch
from .common import require,tensor_digest
from .model import batch_logits


def exact_kl(old_logp,new_logp,mask):
    require(old_logp.shape==new_logp.shape==mask.shape,'KL distribution shape')
    a=old_logp.masked_fill(~mask,0.);b=new_logp.masked_fill(~mask,0.)
    return (a.exp()*mask*(a-b)).sum(-1)


def state_loss(new_logp,old_logp,adv,base_logp,mask,actions,beta=.02):
    require(not old_logp.requires_grad and not adv.requires_grad,'Old likelihood/advantage is attached')
    taken=new_logp.gather(1,actions[:,None]).squeeze(1)
    rho=(taken-old_logp).exp();require(bool(torch.isfinite(rho).all()),'Nonfinite PPO probability ratio')
    gain=torch.minimum(rho*adv,rho.clamp(.9,1.1)*adv)
    kl=exact_kl(base_logp,new_logp,mask)
    return -gain+beta*kl,rho,kl


def adam(parameters,lr):return torch.optim.Adam(parameters,lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0.)


def norm(parameters):return math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in parameters if p.grad is not None))


def same(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def returns_to_go(rewards,scale):
    total=0.;out=[]
    for r in reversed(rewards):total+=float(r)/scale;out.append(total)
    return torch.tensor(list(reversed(out)),dtype=torch.float32)


class PPO:
    def __init__(self,actor,critic,log,counts,*,anchor=True,context=None):
        self.actor,self.critic,self.counts=actor,critic,counts
        self.log=lambda row:log(dict(context or {},**row))
        self.actor_opt=adam(actor.head.parameters(),3e-5);self.critic_opt=adam(critic.parameters(),1e-4)
        self.actor_cursor=0;self.critic_cursor=0;self.halved=False;self.failed_batches=0;self.anchor=anchor
    def inc(self,k,n=1):self.counts[k]=self.counts.get(k,0)+n
    def critic_step(self,episodes):
        self.critic_opt.zero_grad(set_to_none=True);E=len(episodes);loss=0.
        for e in episodes:
            if e['batch'] is None:continue
            predicted=self.critic(e['batch']['v']);part=.5*(predicted-e['G']).square().sum()/E
            self.inc('critic_forwards');part.backward();self.inc('critic_backwards');loss+=float(part.detach())
        g=norm(self.critic.parameters());require(math.isfinite(g),'Nonfinite critic gradient')
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(),1.,error_if_nonfinite=True)
        self.log(dict(event='optimizer_attempt',role='critic',cursor=self.critic_cursor));self.inc('critic_optimizer_attempts')
        self.critic_opt.step();self.critic_cursor+=1;self.inc('critic_optimizer_steps')
        require(all(bool(torch.isfinite(p).all()) for p in self.critic.parameters()),'Nonfinite critic parameters')
        return dict(loss=loss,gradient_norm=g,clipped=g>1.)
    def diagnostic(self,episodes):
        old=[];base=[];changes=0;sat=0;total=0;entropy=[]
        with torch.no_grad():
            for e in episodes:
                if e['batch'] is None:continue
                b=e['batch'];lp=torch.log_softmax(batch_logits(self.actor,b),-1);self.inc('KL_actor_forwards')
                mask=b['mask'].to(lp.device);base_lp=e['base_logp'].to(lp.device)
                old.extend(exact_kl(e['old_logp_all'].to(lp.device),lp,mask).double().tolist())
                base.extend(exact_kl(base_lp,lp,mask).double().tolist())
                entropy.extend((-(lp.exp()*lp.masked_fill(~mask,0.)).sum(-1)).double().tolist())
                changes+=int((lp.argmax(-1)!=base_lp.argmax(-1)).sum());total+=len(lp)
                f=self.actor.head(b['x'].reshape(-1,202).to(lp.device)).detach();sat+=int((torch.tanh(f).abs()>.95).sum())
        require(old and np.isfinite(old).all() and np.isfinite(base).all(),'Empty/nonfinite KL states')
        return dict(old_mean=float(np.mean(old)),old_p95=float(np.quantile(old,.95)),base_mean=float(np.mean(base)),
            state_count=total,base_action_changes=changes,residual_saturated_candidate_values=sat,entropy=float(np.mean(entropy)))
    def actor_epochs(self,episodes,clock,round_id):
        # At most one persistent LR halving/retry per rollout batch.
        self.halved=False
        values=torch.cat([e['G']-e['V_old'] for e in episodes if e['batch'] is not None]).detach()
        scale=float(torch.sqrt(values.double().square().mean()))+1e-8
        for e in episodes:
            if e['batch'] is not None:e['A']=((e['G']-e['V_old'])/scale).detach()
        rows=[];accepted=0;batch_bad=False
        for epoch in range(4):
            clock.check();before=copy.deepcopy(self.actor.head.state_dict());opt=copy.deepcopy(self.actor_opt.state_dict());cursor=self.actor_cursor
            for attempt in range(2):
                self.actor_opt.zero_grad(set_to_none=True);loss=0.;ratio_clipped=0;total=0;size_grads={}
                for e in episodes:
                    if e['batch'] is None:continue
                    b=e['batch'];lp=torch.log_softmax(batch_logits(self.actor,b),-1);self.inc('PPO_actor_forwards')
                    device=lp.device
                    term,rho,kl=state_loss(lp,e['old_logp'].to(device),e['A'].to(device),e['base_logp'].to(device),b['mask'].to(device),e['actions'].to(device),.02 if self.anchor else 0.)
                    prior=[torch.zeros_like(p) if p.grad is None else p.grad.clone() for p in self.actor.head.parameters()]
                    part=term.sum()/len(episodes);part.backward();self.inc('actor_backwards');loss+=float(part.detach())
                    key=e.get('size','synthetic');delta=[p.grad-x for p,x in zip(self.actor.head.parameters(),prior)]
                    if key not in size_grads:size_grads[key]=[v.clone() for v in delta]
                    else:
                        for a,b0 in zip(size_grads[key],delta):a.add_(b0)
                    ratio_clipped+=int(((rho<.9)|(rho>1.1)).sum());total+=len(rho)
                g=norm(self.actor.head.parameters());require(math.isfinite(g),'Nonfinite Actor gradient')
                torch.nn.utils.clip_grad_norm_(self.actor.head.parameters(),1.,error_if_nonfinite=True)
                self.log(dict(event='optimizer_attempt',role='actor',round=round_id,epoch=epoch,attempt=attempt,cursor=self.actor_cursor));self.inc('actor_optimizer_attempts')
                pre_step=[p.detach().clone() for p in self.actor.head.parameters()]
                self.actor_opt.step();self.inc('actor_optimizer_calls')
                step_norm=math.sqrt(sum(float((p.detach()-b0).double().square().sum()) for p,b0 in zip(self.actor.head.parameters(),pre_step)))
                require(all(bool(torch.isfinite(p).all()) for p in self.actor.head.parameters()),'Nonfinite Actor parameters')
                d=self.diagnostic(episodes);bad=self.anchor and (d['old_mean']>.01 or d['old_p95']>.05 or d['base_mean']>.10)
                row=dict(round=round_id,epoch=epoch+1,attempt=attempt,loss=loss,gradient_norm=g,gradient_clipped=g>1.,
                    ratio_clip_fraction=ratio_clipped/max(total,1),advantage_RMS_scale=scale,raw_advantage_mean=float(values.mean()),
                    raw_advantage_std=float(values.std(unbiased=False)),negative_advantages=int((values<0).sum()),
                    old_values_fixed=True,old_probabilities_fixed=True,accepted=not bad,lr=self.actor_opt.param_groups[0]['lr'],
                    Adam_parameter_step_norm=step_norm,gradient_norm_by_size={k:math.sqrt(sum(float(x.double().square().sum()) for x in v)) for k,v in size_grads.items()},**d)
                if bad:
                    self.actor.head.load_state_dict(before);self.actor_opt.load_state_dict(opt);self.actor_cursor=cursor
                    require(same(self.actor.head.state_dict(),before) and same(self.actor_opt.state_dict(),opt),'KL rollback did not restore Actor/Adam')
                    self.inc('actor_rolled_back_calls');row['rollback_exact']=True
                    if attempt==0 and not self.halved:
                        self.halved=True
                        for group in self.actor_opt.param_groups:group['lr']*=.5
                        opt=copy.deepcopy(self.actor_opt.state_dict());row['lr_halved']=True
                    else:batch_bad=True
                else:
                    self.actor_cursor+=1;self.inc('actor_accepted_steps');accepted+=1
                rows.append(row);self.log(dict(event='PPO_epoch',**row))
                if not bad or batch_bad:break
            if batch_bad:break
        # Each sampled value target remains fixed even though critic is fitted now.
        critic_rows=[self.critic_step(episodes) for _ in range(4)]
        self.failed_batches=self.failed_batches+1 if batch_bad else 0
        require(self.failed_batches<3,'Three consecutive batches cannot satisfy fixed KL constraints')
        require(tensor_digest(self.actor.base.state_dict())==self.actor.base_sha,'Frozen base mutated')
        return dict(epochs=rows,critic=critic_rows,accepted_epochs=accepted,batch_noncompliant=batch_bad,
            actor_cursor=self.actor_cursor,critic_cursor=self.critic_cursor,base_sha=self.actor.base_sha)
