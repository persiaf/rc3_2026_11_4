import copy,os,time
from pathlib import Path
import torch
from rc3_decoder_program_20261111.trainer import Trainer as OriginalTrainer,rng,set_rng
from rc3_conservative_refinement.runtime import tensor_digest
from .common import ROOT,HOME,identity,read,put,sha
from .runtime import generator_text,restore_generator

class StreamLedger:
    def __init__(self,parent,seed,role,engineering,trainer):
        self.parent=parent;self.seed=seed;self.role=role;self.trainer=trainer
        self.key=f'{"engineering" if engineering else "scientific"}_{seed}_{role}_optimizer_attempts'
        self.path=HOME/('acceptance/updates' if engineering else f'streams/{seed}/{role}');self.path.mkdir(parents=True,exist_ok=True)
    def __getattr__(self,name):return getattr(self.parent,name)
    def optimizer(self,role,op):
        if self.parent.counts[self.key]-self.trainer.cursor>=64:raise RuntimeError('Per-stream64 extra attempt boundary')
        self.parent.optimizer(role,op);self.parent.counts[self.key]+=1;self.parent.event('stream_optimizer_attempt',seed=self.seed,role=role,logical_update=op);self.parent.save()

class Trainer(OriginalTrainer):
    """Same update math/Adam as accepted v1; production identities and atomic recovery."""
    def __init__(self,actor,budget,seed,sampling,engineering=False):
        super().__init__(actor,budget,identity());self.seed=seed;self.sampling=sampling;self.engineering=engineering
        self.ledger=StreamLedger(budget,seed,actor.role,engineering,self)
        self.train_s=0.;self.training_awake_s=0.;self.extra_attempts=0
    def save_current(self,folder,keep=False):
        folder.mkdir(parents=True,exist_ok=True)
        payload=dict(schema='d5-revision-training-r1',identity=self.identity,role=self.actor.role,seed=self.seed,engineering_only=self.engineering,
            model=copy.deepcopy(self.actor.state_dict()),optimizer=copy.deepcopy(self.optimizer.state_dict()),revision=self.actor.revision,
            cursor=self.cursor,used=sorted(self.used),rng=rng(),sampling_state=generator_text(self.sampling),train_s=self.train_s,training_awake_s=self.training_awake_s,
            counts=copy.deepcopy(dict(self.ledger.counts)),request_ids=sorted(self.ledger.ids),extra_attempts=self.extra_attempts)
        pointer=folder/'LATEST.json';prior=read(pointer) if pointer.exists() else None
        slot=1 if prior and Path(prior['path']).name=='LATEST_0.pt' else 0
        target=folder/f'LATEST_{slot}.pt';tmp=folder/f'LATEST_{slot}.pending.pt'
        with tmp.open('wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
        os.replace(tmp,target);entry=dict(path=target.relative_to(ROOT).as_posix(),sha256=sha(target),cursor=self.cursor,role=self.actor.role,seed=self.seed,train_s=self.train_s)
        put(pointer,entry,pointer.exists())
        if keep:
            saved=folder/f'update_{self.cursor:05}.pt'
            if not saved.exists():
                with saved.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
            put(folder/f'update_{self.cursor:05}.json',dict(entry,path=saved.relative_to(ROOT).as_posix(),sha256=sha(saved)),(folder/f'update_{self.cursor:05}.json').exists())
        return entry
    def restore_current(self,entry):
        path=ROOT/entry['path']
        if sha(path)!=entry['sha256']:raise ValueError('Scientific checkpoint SHA')
        p=torch.load(path,map_location='cpu',weights_only=False)
        if p['schema']!='d5-revision-training-r1' or p['identity']!=self.identity or p['role']!=self.actor.role or p['seed']!=self.seed or p['engineering_only']!=self.engineering:
            raise ValueError('Checkpoint scientific role/seed/stage/source mismatch')
        for key in ('requests','actor_optimizer_attempts'):
            if p['counts'].get(key,0)>self.ledger.counts[key]:raise ValueError('Resource rollback forbidden')
        if not set(p['request_ids'])<=self.ledger.ids:raise ValueError('Checkpoint query history absent from durable attempt journal')
        self.actor.load_state_dict(p['model']);self.actor.revision=p['revision'];self.actor.cache_epoch+=1
        self.optimizer.load_state_dict(p['optimizer']);self.actor.validate_optimizer(self.optimizer)
        self.cursor=p['cursor'];self.used=set(p['used']);self.train_s=p['train_s'];self.training_awake_s=p.get('training_awake_s');self.extra_attempts=p['extra_attempts'];set_rng(p['rng']);restore_generator(self.sampling,p['sampling_state'])
        timer=path.parent/'TRAINING_TIME.json'
        if timer.exists():
            ts=read(timer)
            if ts['cursor']==self.cursor and ts.get('checkpoint_sha256')==entry['sha256']:
                self.train_s=ts['train_s'];self.training_awake_s=ts.get('training_awake_s',self.training_awake_s)
        if self.training_awake_s is None:
            migration=read(HOME/'R04_TIMING_MIGRATION_RECEIPT.json')
            if migration['checkpoint_sha256']!=entry['sha256'] or migration['cursor']!=self.cursor or abs(migration['original_stream_wall_s']-self.train_s)>1e-8:
                raise ValueError('Legacy checkpoint needs exact registered timing migration; no guessed refund')
            self.training_awake_s=self.train_s-migration['official_historical_sleep_s']
        if not self.actor.full and self.actor.frozen_digest()!=self.actor.initial_frozen_digest:raise ValueError('Restored frozen representation')
        return p
