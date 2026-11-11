import copy
import torch
from rc3_global_local.policy import GlobalLocalPolicy,static_data
from rc3_conservative_refinement import runtime as old
from .common import ROLES,sha

ALLOW=('key.weight','dynamic.weight','nav.weight','query.weight','query.bias')
SCHEMA='rc3-score5-original-actor-only-v1'

class StaleCacheError(ValueError):pass

class Score5Actor(GlobalLocalPolicy):
    def __init__(self,role):
        if role not in ROLES:raise ValueError('Exact Score5 role required')
        super().__init__();self.role=role;self.revision=0;self.cache_epoch=0;self.deployed=False
        self.configure()
    def configure(self):
        for name,p in self.named_parameters():p.requires_grad_(name in ALLOW);p.grad=None
    def train(self,mode=True):
        super().train(mode)
        for name,module in self.named_children():
            if name not in ('key','dynamic','nav','query'):module.eval()
        return self
    def frozen_digest(self):return old.tensor_digest({n:p for n,p in self.state_dict().items() if n not in ALLOW})
    def trainable_parameters(self):
        actual=tuple(n for n,p in self.named_parameters() if p.requires_grad)
        if set(actual)!=set(ALLOW) or len(actual)!=5:raise ValueError('Trainability differs from exact five-tensor allowlist')
        if self.deployed:raise ValueError('Deployed Actor cannot train')
        return [dict(self.named_parameters())[n] for n in ALLOW]
    def validate_optimizer(self,optimizer):
        expected=self.trainable_parameters();got=[p for g in optimizer.param_groups for p in g['params']]
        if len(got)!=5 or len({id(p) for p in got})!=5 or {id(p) for p in got}!={id(p) for p in expected}:
            raise ValueError('Optimizer must own exactly the five tensors of this instance')
    def weight_event(self):self.revision+=1;self.cache_epoch+=1
    def encode(self,p):
        device=next(self.parameters()).device
        with torch.no_grad():
            data={k:v.to(device) for k,v in static_data(p).items()};C=data['C'];t=self.task_in(data['task']);c=self.candidate_in(data['candidate'])
            tn=self.task_norm(t+torch.relu(self.to_task(C@c/C.sum(1).clamp_min(1)[:,None])))
            cn=self.candidate_norm(c+torch.relu(self.to_candidate(C.T@tn/C.sum(0).clamp_min(1)[:,None])))
            g=self.global_net(data['global_']).mean(0)
        encoded=dict(**data,t=tn.detach(),c=cn.detach(),g=g.detach(),problem_identity=p.identity,model_identity=id(self),
            parameter_versions=tuple(w._version for w in self.parameters()),history_prefix=(),history_hidden=torch.zeros(self.hidden,device=device),
            score5_revision=self.revision,score5_epoch=self.cache_epoch)
        # Frozen/collection inference may reuse numerical keys within THIS
        # weight version. Grad-enabled score ALWAYS recomputes key projection.
        if not torch.is_grad_enabled():
            with torch.no_grad():encoded['inference_keys']=self.key(encoded['c']).detach()
        return encoded
    def score(self,env,encoded):
        if (encoded['score5_revision']!=self.revision or encoded['score5_epoch']!=self.cache_epoch
                or encoded['parameter_versions']!=tuple(p._version for p in self.parameters())):
            raise StaleCacheError('Cache belongs to a different Score5 weight event')
        if any(encoded[k].requires_grad for k in ('t','c','g','history_hidden')):raise ValueError('Frozen representation/prefix cache retained an autograd graph')
        if torch.is_grad_enabled():keys=self.key(encoded['c'])
        else:
            if 'inference_keys' not in encoded:encoded['inference_keys']=self.key(encoded['c']).detach()
            keys=encoded['inference_keys']
        work=dict(encoded,keys=keys)
        # Unchanged original query/dynamic/nav/key logits, prefix recurrence,
        # intrinsic10*tanh score and legal mask. No206 feature or residual path.
        logits=super().score(env,work)
        encoded['history_prefix']=work['history_prefix'];encoded['history_hidden']=work['history_hidden'].detach()
        return logits

def load_actor(role):
    payload=old._read_sl256_payload()
    actor=Score5Actor(role);actor.load_state_dict(payload['model'],strict=True)
    actor.base_identity=dict(path=old.SL_PATH,sha256=old.SL_SHA256,tensor_sha256=old.SL_TENSOR_SHA256,training_source=copy.deepcopy(payload['training_source']))
    actor.initial_frozen_digest=actor.frozen_digest();actor.train()
    assert sum(p.numel() for p in actor.trainable_parameters())==25152
    assert old.tensor_digest(actor.state_dict())==old.SL_TENSOR_SHA256
    return actor

def export(path,actor,usage):
    if usage not in ('engineering','science'):raise ValueError('Exact usage role required')
    actor.trainable_parameters()
    if actor.frozen_digest()!=actor.initial_frozen_digest:raise ValueError('Frozen parameters changed')
    package=dict(schema=SCHEMA,role=actor.role,usage=usage,allowlist=list(ALLOW),base=actor.base_identity,
        model=copy.deepcopy(actor.state_dict()),tensor_sha256=old.tensor_digest(actor.state_dict()),frozen_sha256=actor.initial_frozen_digest,
        revision=actor.revision,engineering_forbidden_as_scientific_initialization=usage=='engineering',critic_included=False)
    with path.open('xb') as f:torch.save(package,f)
    return dict(path=str(path),sha256=sha(path),role=actor.role,usage=usage)

def load_export(entry,role,usage):
    if sha(entry['path'])!=entry['sha256']:raise ValueError('Export file SHA')
    p=torch.load(entry['path'],map_location='cpu',weights_only=True)
    if set(p)!={'schema','role','usage','allowlist','base','model','tensor_sha256','frozen_sha256','revision','engineering_forbidden_as_scientific_initialization','critic_included'}:raise ValueError('Foreign Actor schema')
    if p['schema']!=SCHEMA or p['role']!=role or p['usage']!=usage or p['allowlist']!=list(ALLOW) or p['critic_included']:raise ValueError('Actor role/usage/parameter identity')
    actor=load_actor(role)
    if p['base']!=actor.base_identity or p['frozen_sha256']!=actor.initial_frozen_digest:raise ValueError('SL provenance differs')
    actor.load_state_dict(p['model']);actor.revision=p['revision'];actor.cache_epoch+=1
    if actor.frozen_digest()!=actor.initial_frozen_digest or old.tensor_digest(actor.state_dict())!=p['tensor_sha256']:raise ValueError('Export tensor identity')
    actor.deployed=True;actor.eval().requires_grad_(False)
    return actor
