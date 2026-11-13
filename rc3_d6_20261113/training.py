import copy,os,time
from pathlib import Path
import torch
from rc3_decoder_program_20261111.trainer import Trainer as OriginalTrainer,rng,set_rng
from rc3_decoder_program_20261111.actor import ALLOW
from rc3_conservative_refinement.runtime import tensor_digest
from .common import ROOT,HOME,read,put,sha,identity,archive
from .runtime import generator_text,restore_generator
from .costs import coefficients

class Trainer(OriginalTrainer):
    def __init__(self,actor,budget,stream,generator):
        super().__init__(actor,budget,identity());self.stream=stream;self.sampling=generator
        self.path=HOME/'streams'/stream;self.path.mkdir(parents=True,exist_ok=True)
    def update(self,episodes):
        a=self.actor;a.validate_optimizer(self.optimizer);current=tensor_digest(a.state_dict());ids=[e['episode_id'] for e in episodes]
        if len(episodes)!=8 or len(set(ids))!=8 or self.used&set(ids):raise ValueError('Exact full batch or duplicate episode')
        if any(e['role']!=a.role or e['actor_tensor_sha']!=current or e['stream']!=self.stream for e in episodes):raise ValueError('Foreign/off-policy stream')
        before=copy.deepcopy(a.state_dict());opt_before=copy.deepcopy(self.optimizer.state_dict());self.optimizer.zero_grad(set_to_none=True)
        loss_sources=[];total=0.
        try:
            for e in episodes:
                cache=a.encode(e['problem']);self.ledger.counts['cache_rebuilds']+=1
                logits=[a.score(s,cache) for s in e['states']]
                logps=torch.stack([torch.log_softmax(z,-1)[act] for z,act in zip(logits,e['actions'])])
                if self.stream=='SL':loss=-logps.mean()/8;weights=None
                elif self.stream=='A':
                    factor=torch.tensor(e['J_H']-e['reference_J_H'],dtype=torch.float32).detach()
                    weights=factor.expand(len(logps));loss=factor*logps.sum()/8
                else:
                    values=coefficients(e['credit'],e['J_H'],e['reference_J_H'],self.stream)
                    weights=torch.tensor(values,dtype=torch.float32).detach()
                    if weights.requires_grad or e['complete_return']!=-e['J_H']:raise ValueError('Detached complete costs')
                    loss=(weights*logps).sum()/8
                total+=float(loss.detach());loss.backward();self.ledger.counts['backward_calls']+=1
                loss_sources.append(dict(episode_id=e['episode_id'],task_id=e['task_id'],action_count=len(logps),loss=float(loss.detach()),
                    weights=None if weights is None else weights.tolist(),reference_J=e.get('reference_J_H'),J_H=e.get('J_H')))
            grads={n:None if p.grad is None else float(p.grad.norm()) for n,p in a.named_parameters()}
            if any(p.grad is not None for n,p in a.named_parameters() if not p.requires_grad):raise ValueError('Frozen gradient')
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in a.trainable_parameters()):raise ValueError('Missing/nonfinite five-tensor gradient')
            norm=torch.nn.utils.clip_grad_norm_(a.trainable_parameters(),1.,error_if_nonfinite=True)
            self.ledger.optimizer(self.stream,self.cursor+1);self.optimizer.step();self.ledger.counts['actor_optimizer_completed']+=1;a.weight_event()
            if any(not torch.isfinite(p).all() for p in a.parameters()) or a.frozen_digest()!=a.initial_frozen_digest:raise ValueError('Updated finite/frozen contract')
        except BaseException:
            a.load_state_dict(before);self.optimizer.load_state_dict(opt_before);a.weight_event();self.optimizer.zero_grad(set_to_none=True)
            self.ledger.event('update_rollback',stream=self.stream,cursor=self.cursor,exact_model_and_Adam_restored=True);raise
        self.cursor+=1;self.used.update(ids)
        result=dict(stream=self.stream,role=a.role,cursor=self.cursor,loss=total,gradient_norm=float(norm),gradient_norms=grads,
            clipped=bool(norm>1),clip_factor=min(1.,1./(float(norm)+1e-6)),before=current,after=tensor_digest(a.state_dict()),
            loss_sources=loss_sources,epoch=1,entropy_weight=0,critic_updates=0)
        archive(self.path/'UPDATES.jsonl.gz',result);self.optimizer.zero_grad(set_to_none=True);self.ledger.save();return result
    def save_current(self):
        payload=dict(schema='d6-training-v1',identity=self.identity,stream=self.stream,role=self.actor.role,
            model=copy.deepcopy(self.actor.state_dict()),optimizer=copy.deepcopy(self.optimizer.state_dict()),revision=self.actor.revision,
            cursor=self.cursor,used=sorted(self.used),rng=rng(),sampling_state=generator_text(self.sampling),
            counts=dict(self.ledger.counts),request_ids=sorted(self.ledger.ids),engineering_only=False)
        pointer=self.path/'LATEST.json';prior=read(pointer) if pointer.exists() else None
        slot=1 if prior and Path(prior['path']).name=='LATEST_0.pt' else 0
        target=self.path/f'LATEST_{slot}.pt';temp=target.with_suffix('.pending.pt')
        with temp.open('wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
        replace_file(temp,target)
        entry=dict(path=target.relative_to(ROOT).as_posix(),sha256=sha(target),cursor=self.cursor,stream=self.stream)
        put(pointer,entry,pointer.exists())
        if self.cursor in (0,128,256):
            saved=self.path/f'step{self.cursor:05}.training.pt'
            if not saved.exists():
                with saved.open('xb') as f:torch.save(payload,f)
        return entry
    def restore_current(self,entry):
        p=ROOT/entry['path']
        if sha(p)!=entry['sha256']:raise ValueError('Checkpoint SHA')
        q=torch.load(p,map_location='cpu',weights_only=False)
        if q['schema']!='d6-training-v1' or q['identity']!=self.identity or q['stream']!=self.stream or q['role']!=self.actor.role or q['engineering_only']:raise ValueError('Checkpoint identity')
        if any(q['counts'].get(k,0)>self.ledger.counts[k] for k in ('requests','actor_optimizer_attempts')) or not set(q['request_ids'])<=self.ledger.ids:raise ValueError('Resource rollback')
        if self.ledger.counts['optimizer_'+self.stream]!=q['cursor']:raise ValueError('Ambiguous optimizer not durably checkpointed; no free repeated update')
        self.actor.load_state_dict(q['model']);self.actor.revision=q['revision'];self.actor.cache_epoch+=1
        self.optimizer.load_state_dict(q['optimizer']);self.actor.validate_optimizer(self.optimizer);set_rng(q['rng']);restore_generator(self.sampling,q['sampling_state'])
        self.cursor=q['cursor'];self.used=set(q['used'])
        if self.actor.frozen_digest()!=self.actor.initial_frozen_digest:raise ValueError('Checkpoint frozen representation')
        self.ledger.event('checkpoint_restored',stream=self.stream,cursor=self.cursor,checkpoint=entry,Adam_rng_sampler_cursor_and_budget_restored=True)
        return q

def replace_file(src,dst):
    for i,delay in enumerate((0.,.01,.02,.04,.08,.16,.32)):
        if delay:time.sleep(delay)
        try:os.replace(src,dst);return
        except OSError as e:
            if getattr(e,'winerror',None) not in (5,32,33) or i==6:raise
