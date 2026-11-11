import copy
import random
import numpy as np
import torch
from rc3_conservative_refinement.runtime import tensor_digest
from rc3_conservative_refinement.trainer import checkpoint_equal
from .actor import ALLOW
from .common import archive,sha

def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state())
def set_rng(r):random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'])

class Trainer:
    """One Adam step, full terminal scalar return; no Critic, guard or A1 limit."""
    def __init__(self,actor,ledger,identity):
        self.actor=actor;self.ledger=ledger;self.identity=identity;self.cursor=0;self.used=set()
        self.optimizer=torch.optim.Adam(actor.trainable_parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
        actor.validate_optimizer(self.optimizer)
    def update(self,episodes):
        a=self.actor;a.validate_optimizer(self.optimizer);current=tensor_digest(a.state_dict());ids=[e['episode_id'] for e in episodes]
        if len(ids)!=len(set(ids)) or set(ids)&self.used:raise ValueError('Episode reuse')
        if any(e['role']!=a.role or e['actor_tensor_sha']!=current for e in episodes):raise ValueError('Foreign/off-policy trajectory')
        before=copy.deepcopy(a.state_dict());opt_before=copy.deepcopy(self.optimizer.state_dict());self.optimizer.zero_grad(set_to_none=True)
        total=0.;loss_sources=[]
        try:
            for e in episodes:
                cache=a.encode(e['problem']);self.ledger.counts['cache_rebuilds']+=1
                logits=[a.score(s,cache) for s in e['states']]
                if not logits:continue
                logps=torch.stack([torch.log_softmax(z,-1)[action] for z,action in zip(logits,e['actions'])])
                if a.role=='D5-SL':loss=-logps.mean()/len(episodes);factor=None
                else:
                    factor=torch.tensor(e['J_H']-e['reference_J_H'],dtype=torch.float32).detach()
                    loss=factor*logps.sum()/len(episodes)
                    if e['complete_return']!=-e['J_H']:raise ValueError('Complete return contract')
                total+=float(loss.detach());loss.backward();self.ledger.counts['backward_calls']+=1
                loss_sources.append(dict(episode_id=e['episode_id'],sample_J=e.get('J_H'),reference_J=e.get('reference_J_H'),factor=None if factor is None else float(factor),action_count=len(logps)))
            grads={n:None if p.grad is None else float(p.grad.norm()) for n,p in a.named_parameters()}
            if any(p.grad is not None for n,p in a.named_parameters() if not p.requires_grad):raise ValueError('Frozen gradient')
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in a.trainable_parameters()):raise ValueError('Missing/nonfinite allowed gradient')
            norm=torch.nn.utils.clip_grad_norm_(a.trainable_parameters(),1.,error_if_nonfinite=True)
            self.ledger.optimizer(a.role,self.cursor+1);self.optimizer.step();self.ledger.counts['actor_optimizer_completed']+=1;a.weight_event()
            if any(not torch.isfinite(p).all() for p in a.parameters()):raise ValueError('Nonfinite updated parameter')
            if not a.full and a.frozen_digest()!=a.initial_frozen_digest:raise ValueError('Frozen representation changed')
        except BaseException:
            a.load_state_dict(before);self.optimizer.load_state_dict(opt_before);a.weight_event();self.optimizer.zero_grad(set_to_none=True);raise
        self.cursor+=1;self.used.update(ids)
        result=dict(role=a.role,cursor=self.cursor,loss=total,gradient_norm=float(norm),gradient_norms=grads,
            before=current,after=tensor_digest(a.state_dict()),revision=a.revision,loss_sources=loss_sources,
            permanent_SL_guard=False,update_acceptance_guard=False,additional_A1_amplitude=False,critic_updates=0)
        archive(self.ledger.path/'UPDATES.jsonl.gz',result);self.optimizer.zero_grad(set_to_none=True);return result
    def save(self,path):
        payload=dict(schema='decoder-training-v1',identity=self.identity,role=self.actor.role,model=copy.deepcopy(self.actor.state_dict()),
            optimizer=copy.deepcopy(self.optimizer.state_dict()),revision=self.actor.revision,cursor=self.cursor,used=sorted(self.used),rng=rng(),
            counts=copy.deepcopy(self.ledger.counts),request_ids=sorted(self.ledger.ids),engineering_only=True)
        with path.open('xb') as f:torch.save(payload,f)
        return dict(path=str(path),sha256=sha(path))
    def restore(self,entry):
        if sha(entry['path'])!=entry['sha256']:raise ValueError('Checkpoint file SHA')
        p=torch.load(entry['path'],map_location='cpu',weights_only=False)
        if p['schema']!='decoder-training-v1' or p['identity']!=self.identity or p['role']!=self.actor.role:raise ValueError('Checkpoint provenance/role')
        if any(p['counts'].get(k,0)>self.ledger.counts[k] for k in ('requests','actor_optimizer_attempts')) or not set(p['request_ids'])<=self.ledger.ids:
            raise ValueError('Resource rollback forbidden')
        self.actor.load_state_dict(p['model']);self.actor.revision=p['revision'];self.actor.cache_epoch+=1
        self.optimizer.load_state_dict(p['optimizer']);self.actor.validate_optimizer(self.optimizer)
        self.cursor=p['cursor'];self.used=set(p['used']);set_rng(p['rng']);return p
