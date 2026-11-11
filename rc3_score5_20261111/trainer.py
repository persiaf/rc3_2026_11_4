import copy
import math
import random
import torch
import numpy as np
from rc3_conservative_refinement.runtime import tensor_digest
from rc3_conservative_refinement.trainer import checkpoint_equal
from .actor import ALLOW
from .runtime import replay_logits
from .common import ROOT,read,sha,archive

def step_guard(old,new,states):
    rows=[]
    for a,b,s in zip(old,new,states):
        mask=torch.as_tensor(s.mask());finite=bool(torch.isfinite(b[mask]).all() and torch.isneginf(b[~mask]).all())
        lp=torch.log_softmax(a[mask].double(),-1);lq=torch.log_softmax(b[mask].double(),-1);p,q=lp.exp(),lq.exp()
        kl=float((p*(lp-lq)).sum().clamp_min(0));change=float((p-q).abs().max())
        rows.append(dict(accepted=finite and kl<=.001 and change<=.02,finite_mask=finite,incremental_KL=kl,max_probability_change=change))
    return dict(accepted=all(r['accepted'] for r in rows),states=rows,reference='PREVIOUS_POLICY_ONLY_NO_SL_TV_OR_BASE_KL_OR_RESIDUAL_BOUND')

def rng():return dict(torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate())
def set_rng(r):torch.set_rng_state(r['torch']);np.random.set_state(r['numpy']);random.setstate(r['python'])

class Trainer:
    def __init__(self,actor,ledger,identity):
        self.actor,self.ledger,self.identity=actor,ledger,identity
        self.optimizer=torch.optim.Adam(actor.trainable_parameters(),lr=3e-5,betas=(.9,.999),eps=1e-8,weight_decay=0.)
        actor.validate_optimizer(self.optimizer);self.used=set();self.cursor=0;self.consecutive_rejects=0
        self.table=read(ROOT/'rc3_r04_m2_20261111/attempts/02/FIT_ONLY_COMPONENT_ANCHORS.json')
    def reference(self,episode):
        p=episode['problem'];window=f'{p.window["width"]}x{p.window["height"]}';values=[]
        for s in episode['states']:
            q=min(3,max(0,math.floor(4*(1-float(s.remaining.mean())))))
            c=self.table['conditional'].get(f'{window}|q{q}',self.table['global'])
            values.append(-c[0]-.75*c[1]+c[2])
        return torch.tensor(values,dtype=torch.float32).detach()
    def update(self,episodes,force_reject=False):
        actor=self.actor;actor.train();actor.validate_optimizer(self.optimizer)
        if actor.frozen_digest()!=actor.initial_frozen_digest:raise ValueError('Frozen representation changed')
        if not episodes or self.consecutive_rejects>=16:raise ValueError('Empty/stopped update')
        current=tensor_digest(actor.state_dict());supervised=actor.role=='Score5-SL'
        ids=[e['episode_id'] for e in episodes]
        if len(ids)!=len(set(ids)) or self.used.intersection(ids):raise ValueError('Repeated batch')
        for e in episodes:
            if e['role']!=actor.role or e['stage']!=self.ledger.stage:raise ValueError('Episode role/stage')
            if not supervised and e['actor_tensor_sha']!=current:raise ValueError('Off-policy online trajectory')
        before={n:dict(actor.named_parameters())[n].detach().clone() for n in ALLOW};opt_before=copy.deepcopy(self.optimizer.state_dict())
        with torch.no_grad():old=[replay_logits(actor,e,self.ledger) for e in episodes]
        if not supervised:
            for e,zs in zip(episodes,old):
                for z,recorded,s in zip(zs,e['old_logits'],e['states']):
                    mask=torch.tensor(s.mask())
                    if not torch.allclose(z[mask],recorded[mask],atol=1e-6,rtol=0):raise ValueError('Current policy differs from collection')
        advantages=[(e['G']-self.reference(e)).detach() for e in episodes] if not supervised else []
        rms=float(torch.cat(advantages).double().square().mean().sqrt())+1e-8 if advantages else 1.
        self.optimizer.zero_grad(set_to_none=True);total=0.
        for i,e in enumerate(episodes):
            # No detached206 feature input. Fresh graph/key for each score,
            # frozen encoded representation/prefix only inside this episode.
            zs=replay_logits(actor,e,self.ledger)
            for t,(z,a) in enumerate(zip(zs,e['actions'])):
                part=-torch.log_softmax(z,-1)[a]/len(episodes)
                part=part/len(zs) if supervised else part*(advantages[i][t]/rms)
                if not torch.isfinite(part):raise ValueError('Nonfinite loss')
                part.backward();self.ledger.counts['backward_calls']+=1;total+=float(part.detach())
        gradients={n:float(p.grad.norm()) if p.grad is not None else None for n,p in actor.named_parameters()}
        if any(p.grad is not None for n,p in actor.named_parameters() if n not in ALLOW):raise ValueError('Frozen gradient')
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in actor.trainable_parameters()):raise ValueError('Invalid allowed gradient')
        norm=torch.nn.utils.clip_grad_norm_(actor.trainable_parameters(),1.,error_if_nonfinite=True)
        self.used.update(ids);self.cursor+=1;self.ledger.optimizer(actor.role,self.cursor)
        self.optimizer.step();self.ledger.counts['actor_optimizer_completed']+=1;actor.weight_event()
        try:
            with torch.no_grad():new=[replay_logits(actor,e,self.ledger) for e in episodes]
            reports=[step_guard(a,b,e['states']) for a,b,e in zip(old,new,episodes)]
            accepted=all(r['accepted'] for r in reports) and not force_reject
        except BaseException:
            self._restore(before,opt_before);raise
        if not accepted:self._restore(before,opt_before)
        self.consecutive_rejects=0 if accepted else self.consecutive_rejects+1
        if actor.frozen_digest()!=actor.initial_frozen_digest:raise ValueError('Frozen weight changed')
        actor.validate_optimizer(self.optimizer);self.optimizer.zero_grad(set_to_none=True)
        result=dict(role=actor.role,cursor=self.cursor,accepted=accepted,forced_rejection=force_reject,loss=total,gradient_norm=float(norm),gradient_norms=gradients,
            guard=reports,before=current,after=tensor_digest(actor.state_dict()),revision=actor.revision,frozen_sha256=actor.frozen_digest(),
            rollback_exact=not accepted,cache_policy='Fresh original encode per replay after every weight event; graph keys rebuilt per grad score')
        archive(self.ledger.path/'UPDATES.jsonl.gz',result);self.ledger.event('score5_update',**{k:v for k,v in result.items() if k not in ('gradient_norms','guard')})
        return result
    def _restore(self,before,opt):
        with torch.no_grad():
            for n,v in before.items():dict(self.actor.named_parameters())[n].copy_(v)
        self.optimizer.load_state_dict(opt);self.actor.weight_event()
        assert all(torch.equal(dict(self.actor.named_parameters())[n],v) for n,v in before.items())
        assert checkpoint_equal(self.optimizer.state_dict(),opt)
    def save(self,path):
        payload=dict(schema='score5-training-v1',role=self.actor.role,stage=self.ledger.stage,identity=self.identity,
            base=self.actor.base_identity,model=copy.deepcopy(self.actor.state_dict()),optimizer=copy.deepcopy(self.optimizer.state_dict()),
            allowlist=list(ALLOW),revision=self.actor.revision,cursor=self.cursor,used=sorted(self.used),rejects=self.consecutive_rejects,rng=rng(),
            resource=copy.deepcopy(self.ledger.snapshot()),request_ids=sorted(self.ledger.ids),engineering_forbidden_as_scientific_initialization=self.ledger.stage=='engineering')
        with path.open('xb') as f:torch.save(payload,f)
        return dict(path=str(path),sha256=sha(path))
    def restore(self,entry):
        if sha(entry['path'])!=entry['sha256']:raise ValueError('Checkpoint SHA')
        p=torch.load(entry['path'],map_location='cpu',weights_only=False)
        if p['schema']!='score5-training-v1' or p['role']!=self.actor.role or p['stage']!=self.ledger.stage or p['identity']!=self.identity or p['base']!=self.actor.base_identity or p['allowlist']!=list(ALLOW):raise ValueError('Checkpoint role/source mismatch')
        if p['resource']['counts']['actor_optimizer_attempts']>self.ledger.counts['actor_optimizer_attempts'] or not set(p['request_ids'])<=self.ledger.ids:raise ValueError('Resource reset forbidden')
        self.actor.load_state_dict(p['model']);self.actor.revision=p['revision'];self.actor.cache_epoch+=1
        self.optimizer.load_state_dict(p['optimizer']);self.actor.validate_optimizer(self.optimizer)
        self.cursor=p['cursor'];self.used=set(p['used']);self.consecutive_rejects=p['rejects'];set_rng(p['rng'])
        if self.actor.frozen_digest()!=self.actor.initial_frozen_digest:raise ValueError('Restored frozen identity')
        return p
