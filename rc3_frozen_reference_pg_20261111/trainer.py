"""One real guarded Actor candidate per opportunity; no Critic exists."""
import copy
import math
import random
import numpy as np
import torch
from rc3_conservative_refinement.trainer import masked_log_softmax,ppo_state_loss,checkpoint_equal
from rc3_conservative_refinement.guard import GuardLimits,check_update,restore_if_rejected
from rc3_conservative_refinement.runtime import tensor_digest,_assert_frozen
from .common import archive,put,sha
from .actor import load_actor,CORRECTION_SCHEMA

def rng():return dict(torch=torch.get_rng_state().clone(),numpy=np.random.get_state(),python=random.getstate())
def set_rng(s):torch.set_rng_state(s['torch']);np.random.set_state(s['numpy']);random.setstate(s['python'])

class Trainer:
    def __init__(self,actor,reference,ledger,identity):
        self.actor,self.reference,self.ledger,self.identity=actor,reference,ledger,identity
        self.optimizer=torch.optim.Adam(actor.correction.parameters(),lr=3e-5,betas=(.9,.999),eps=1e-8,weight_decay=0.)
        self.counts=dict(opportunities=0,accepted=0,rejected=0,consecutive_rejected=0,failed=0)
        self.used=set();self.failed=False
    def update(self,episodes,*,force_reject=False):
        if self.failed or not episodes or self.counts['consecutive_rejected']>=16:raise RuntimeError('Trainer stop boundary')
        actor=self.actor;_assert_frozen(actor);actor.train()
        current=tensor_digest(actor.correction.state_dict());supervised=actor.role=='continued_SL_A1'
        ids=[e['episode_id'] for e in episodes]
        if len(set(ids))!=len(ids) or self.used.intersection(ids):raise ValueError('Repeated trajectories')
        for e in episodes:
            if e['role']!=actor.role or e['stage']!=self.ledger.stage:raise ValueError('Exact episode role/stage mismatch')
            if not supervised and e['actor_tensor_sha']!=current:raise ValueError('Off-policy episode')
            if e['batch']['x'].shape[-1]!=206:raise ValueError('A1 feature schema')
        before=copy.deepcopy(actor.correction.state_dict());opt_before=copy.deepcopy(self.optimizer.state_dict())
        self.used.update(ids);self.counts['opportunities']+=1;op=self.counts['opportunities']
        self.optimizer.zero_grad(set_to_none=True)
        try:
            admitted=[];all_adv=[]
            for e in episodes:
                b=e['batch'];a=e['actions'];old=actor.correction(b['z0'],b['x'],b['mask']).detach()
                if not supervised and not torch.allclose(old[b['mask']],e['old_logits'][b['mask']],atol=1e-6,rtol=0):raise ValueError('Collection logits not current policy')
                old_lp=masked_log_softmax(old,b['mask']).gather(1,a[:,None]).squeeze(1).detach()
                refs=self.reference.values(b['v'],e['window'])
                if supervised:adv=None
                else:
                    base=refs['conditional_value'] if actor.role=='PG-cond' else refs['global_value']
                    adv=(e['G'].detach()-base.detach()).detach();all_adv.append(adv)
                    if not torch.isfinite(adv).all():raise ValueError('Nonfinite full return advantage')
                admitted.append((e,old,old_lp,refs,adv))
            rms=float(torch.cat(all_adv).double().square().mean().sqrt())+1e-8 if all_adv else 1.
            loss_total=0.
            for e,old,old_lp,refs,adv in admitted:
                b=e['batch'];a=e['actions'];lp=masked_log_softmax(actor.correction(b['z0'],b['x'],b['mask']),b['mask'])
                if supervised:part=-lp.gather(1,a[:,None]).mean()/len(episodes)
                else:
                    objective,_,_=ppo_state_loss(lp,old_lp,(adv/rms).detach(),masked_log_softmax(b['z0'],b['mask']).detach(),b['mask'],a,beta=.02,ratio_clip=.1)
                    part=objective.sum()/len(episodes)
                if not torch.isfinite(part):raise ValueError('Nonfinite Actor loss')
                part.backward();loss_total+=float(part.detach())
            norm=torch.nn.utils.clip_grad_norm_(actor.correction.parameters(),1.,error_if_nonfinite=True)
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in actor.correction.parameters()):raise ValueError('Actor gradient invalid')
            self.ledger.optimizer(actor.role,op);self.optimizer.step();self.ledger.counts['actor_optimizer_completed']+=1
            if any(not torch.isfinite(p).all() for p in actor.correction.parameters()):raise ValueError('Actor parameters invalid')
            with torch.no_grad():
                candidate=[actor.correction(e['batch']['z0'],e['batch']['x'],e['batch']['mask']).detach() for e in episodes]
                guards=[check_update(e['batch']['z0'],old,new,e['batch']['mask'],GuardLimits()) for (e,old,_,_,_),new in zip(admitted,candidate)]
            accepted=all(g['accepted'] for g in guards) and not force_reject
            rollback=restore_if_rejected(actor.correction,self.optimizer,before,opt_before,{'accepted':accepted})
            self.counts['accepted' if accepted else 'rejected']+=1
            self.counts['consecutive_rejected']=0 if accepted else self.counts['consecutive_rejected']+1
            self.optimizer.zero_grad(set_to_none=True);_assert_frozen(actor)
            with torch.no_grad():
                for (e,old,old_lp,refs,adv),new in zip(admitted,candidate):
                    b=e['batch'];mask=b['mask'];a=e['actions']
                    applied=actor.correction.details(b['z0'],b['x'],mask)
                    oldfull=masked_log_softmax(old,mask);newfull=masked_log_softmax(new,mask)
                    basefull=masked_log_softmax(b['z0'],mask)
                    appliedfull=masked_log_softmax(applied['logits'],mask)
                    safe=lambda z:z.masked_fill(~mask,0.)
                    kl=(oldfull.exp()*(safe(oldfull)-safe(newfull))).sum(1).clamp_min(0.)
                    tv=.5*(newfull.exp()-basefull.exp()).abs().sum(1)
                    candidate_ratio=(newfull.gather(1,a[:,None]).squeeze(1)-old_lp).exp()
                    applied_ratio=(appliedfull.gather(1,a[:,None]).squeeze(1)-old_lp).exp()
                    state_rows=[]
                    for t in range(len(a)):
                        legal=mask[t].nonzero(as_tuple=False).flatten();order=legal[torch.argsort(b['z0'][t,legal],descending=True,stable=True)]
                        delta=applied['delta'][t,legal]
                        gap=None if len(order)<2 else float(b['z0'][t,order[0]]-b['z0'][t,order[1]])
                        state_rows.append(dict(state=t,legal_count=len(legal),base_gap=gap,
                            effective_delta_range=float(delta.max()-delta.min()),delta_min=float(delta.min()),delta_max=float(delta.max()),
                            delta_std=float(delta.std(unbiased=False)),post_tanh_mean=float(delta.mean()),
                            delta_at_base_top1=float(applied['delta'][t,order[0]]),delta_at_base_top2=None if len(order)<2 else float(applied['delta'][t,order[1]]),
                            raw_legal_mean=float(applied['legal_raw_mean'][t,0]),
                            candidate_ratio=float(candidate_ratio[t]),applied_ratio=float(applied_ratio[t]),candidate_step_KL=float(kl[t]),candidate_base_TV=float(tv[t]),
                            base_top1=int(order[0]),applied_top1=int(applied['logits'][t].argmax()),actual_action=int(a[t]),
                            G=None if supervised else float(e['G'][t]),B_global=float(refs['global_value'][t]),B_cond=float(refs['conditional_value'][t]),
                            centered_global=None if supervised else float(e['G'][t]-refs['global_value'][t]),
                            centered_cond=None if supervised else float(e['G'][t]-refs['conditional_value'][t]),
                            A_sample=None if supervised else float(adv[t]),condition_key=refs['condition_keys'][t],unknown_fallback=refs['fallback'][t]))
                    archive(self.ledger.path/'ONLINE_UPDATE_DIAGNOSTICS.jsonl.gz',dict(role=actor.role,opportunity=op,episode_id=e['episode_id'],
                        accepted=accepted,forced_engineering_rejection=force_reject,window=e['window'],task_count=e['task_count'],states=state_rows))
            report=dict(role=actor.role,opportunity=op,accepted=accepted,rollback_exact=bool(rollback.get('restored')),
                forced_engineering_rejection=force_reject,guard=guards,loss=loss_total,gradient_norm=float(norm),advantage_rms=rms,
                actor_before=current,actor_after=tensor_digest(actor.correction.state_dict()),counts=copy.deepcopy(self.counts))
            self.ledger.event('actor_candidate_complete',**report)
            return report
        except BaseException:
            actor.correction.load_state_dict(before);self.optimizer.load_state_dict(opt_before);self.optimizer.zero_grad(set_to_none=True)
            self.counts['failed']+=1;self.failed=True;self.ledger.event('failed_candidate_rolled_back',role=actor.role,opportunity=op,attempts_remain_debited=True)
            raise
    def save(self,path):
        package=dict(schema='rc3-frozen-reference-pg-training-A1-v1',correction_schema=CORRECTION_SCHEMA,role=self.actor.role,
            stage=self.ledger.stage,identity=self.identity,base=self.actor.base_identity,table_sha256=self.reference.sha256,
            actor=copy.deepcopy(self.actor.correction.state_dict()),adam=copy.deepcopy(self.optimizer.state_dict()),rng=rng(),
            counts=copy.deepcopy(self.counts),used_ids=sorted(self.used),resource=copy.deepcopy(self.ledger.snapshot()),
            request_ids=sorted(self.ledger.ids),critic_included=False)
        with path.open('xb') as f:torch.save(package,f)
        return dict(path=str(path),sha256=sha(path))
    def restore(self,entry):
        if sha(entry['path'])!=entry['sha256']:raise ValueError('Training checkpoint bytes')
        p=torch.load(entry['path'],map_location='cpu',weights_only=False)
        if p['schema']!='rc3-frozen-reference-pg-training-A1-v1' or p['correction_schema']!=CORRECTION_SCHEMA:raise ValueError('Checkpoint schema')
        if p['role']!=self.actor.role or p['stage']!=self.ledger.stage or p['identity']!=self.identity or p['base']!=self.actor.base_identity or p['table_sha256']!=self.reference.sha256 or p['critic_included']:raise ValueError('Checkpoint role/source/table/base mismatch')
        if any(p['resource']['counts'][k]>self.ledger.counts[k] for k in ('requests','actor_optimizer_attempts')) or not set(p['request_ids'])<=self.ledger.ids:raise ValueError('Checkpoint resource/query ledger cannot reset')
        self.actor.correction.load_state_dict(p['actor']);self.optimizer.load_state_dict(p['adam']);self.counts=copy.deepcopy(p['counts']);self.used=set(p['used_ids']);set_rng(p['rng'])
        if not checkpoint_equal(self.optimizer.state_dict(),p['adam']) or not checkpoint_equal(rng(),p['rng']):raise ValueError('Exact Adam/RNG restore')
        _assert_frozen(self.actor);return p
