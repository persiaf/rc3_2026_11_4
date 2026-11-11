import copy
import torch
from rc3_global_local.policy import GlobalLocalPolicy
from rc3_score5_20261111.actor import Score5Actor,ALLOW,StaleCacheError
from rc3_conservative_refinement import runtime as source
from .common import ROLES,sha

class Actor(Score5Actor):
    def __init__(self,role):
        if role not in ROLES:raise ValueError('Exact program role required')
        # Same original network; no added parameters, residual head or state-ID input.
        self.full=role=='Full-PG'
        super().__init__('Score5-PG' if role!='D5-SL' else 'Score5-SL')
        self.role=role
    def configure(self):
        for name,p in self.named_parameters():p.requires_grad_(self.full or name in ALLOW);p.grad=None
    def train(self,mode=True):
        if self.full:return GlobalLocalPolicy.train(self,mode)
        return super().train(mode)
    def trainable_parameters(self):
        expected={n for n,_ in self.named_parameters()} if self.full else set(ALLOW)
        actual={n for n,p in self.named_parameters() if p.requires_grad}
        if self.deployed or actual!=expected:raise ValueError('Trainable parameter identity differs')
        return [p for n,p in self.named_parameters() if n in expected]
    def validate_optimizer(self,opt):
        want=self.trainable_parameters();got=[p for group in opt.param_groups for p in group['params']]
        if len(got)!=len(want) or len({id(p) for p in got})!=len(want) or {id(p) for p in got}!={id(p) for p in want}:
            raise ValueError('Optimizer owns an incorrect parameter or another model')
    def encode(self,p):
        if not self.full:return super().encode(p)
        e=GlobalLocalPolicy.encode(self,p);e.update(score5_revision=self.revision,score5_epoch=self.cache_epoch);return e
    def score(self,s,e):
        if not self.full:return super().score(s,e)
        if e['score5_revision']!=self.revision or e['score5_epoch']!=self.cache_epoch:raise StaleCacheError('FullActor stale cache')
        return GlobalLocalPolicy.score(self,s,e)

def load_actor(role):
    payload=source._read_sl256_payload();actor=Actor(role);actor.load_state_dict(payload['model'],strict=True)
    actor.base_identity=dict(path=source.SL_PATH,sha256=source.SL_SHA256,tensor_sha256=source.SL_TENSOR_SHA256,training_source=copy.deepcopy(payload['training_source']))
    actor.initial_frozen_digest=actor.frozen_digest();actor.train()
    assert source.tensor_digest(actor.state_dict())==source.SL_TENSOR_SHA256
    assert sum(p.numel() for p in actor.parameters())==65088
    assert sum(p.numel() for p in actor.trainable_parameters())==(65088 if actor.full else 25152)
    return actor

def export(path,actor,engineering=True):
    if not actor.full and actor.frozen_digest()!=actor.initial_frozen_digest:raise ValueError('Frozen representation changed')
    package=dict(schema='rc3-decoder-program-actor-v1',role=actor.role,model=copy.deepcopy(actor.state_dict()),base=actor.base_identity,
        revision=actor.revision,tensor_sha256=source.tensor_digest(actor.state_dict()),engineering_only=engineering,critic_included=False)
    with path.open('xb') as f:torch.save(package,f)
    return dict(path=str(path),sha256=sha(path),role=actor.role,engineering_only=engineering)

def load_export(entry,engineering=True):
    if sha(entry['path'])!=entry['sha256']:raise ValueError('Export file SHA')
    p=torch.load(entry['path'],map_location='cpu',weights_only=True)
    if p['schema']!='rc3-decoder-program-actor-v1' or p['role']!=entry['role'] or p['engineering_only']!=engineering or p['critic_included']:
        raise ValueError('Actor role/usage/schema mismatch')
    a=load_actor(p['role'])
    if p['base']!=a.base_identity:raise ValueError('SL source changed')
    a.load_state_dict(p['model']);a.revision=p['revision'];a.cache_epoch+=1
    if source.tensor_digest(a.state_dict())!=p['tensor_sha256']:raise ValueError('Export tensor SHA')
    if not a.full and a.frozen_digest()!=a.initial_frozen_digest:raise ValueError('Export frozen tensors changed')
    a.deployed=True;a.eval().requires_grad_(False);return a
