"""Production A1 schema: explicit roles and strict Actor-only artifacts."""
import copy
import time
import torch
from rc3_conservative_refinement import runtime as old
from rc3_conservative_refinement.adapter import LookaheadResidualActor
from rc3_r04_review_20261111.relative_actor import RelativeCorrection,SCHEMA as CORRECTION_SCHEMA
from .common import ROLES,sha

SCHEMA='rc3-frozen-reference-pg-actor-only-A1-v1'
class Actor(LookaheadResidualActor):
    def __init__(self,encoder,role):
        if role not in ROLES:raise ValueError('Unknown Actor role')
        super().__init__(encoder,distances_enabled=True,max_adjustment=.25)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(old._seed(11622026,'A1_residual_init'))
            self.correction=RelativeCorrection()
        self.role=role;self.correction_schema=CORRECTION_SCHEMA

def load_actor(role):
    tick=time.perf_counter();temporary=old.load_actor(True,11622026)
    actor=Actor(temporary.encoder,role);actor.base_identity=copy.deepcopy(temporary.base_identity)
    actor.initialization_seed=11622026;actor.loading_seconds=time.perf_counter()-tick
    actor.eval();old._assert_frozen(actor)
    return actor

def export(path,actor,*,usage,opportunity):
    old._assert_frozen(actor)
    if usage not in ('engineering','science'):raise ValueError('Explicit stage role')
    package=dict(schema=SCHEMA,correction_schema=CORRECTION_SCHEMA,feature_schema=old.FEATURE_SCHEMA,
        role=actor.role,usage=usage,opportunity=opportunity,base=actor.base_identity,input_dim=206,
        hidden=[64,64],amplitude=.25,distances_enabled=True,seed=11622026,
        correction=copy.deepcopy(actor.correction.state_dict()),
        correction_sha256=old.tensor_digest(actor.correction.state_dict()))
    with path.open('xb') as f:torch.save(package,f)
    return dict(path=str(path),sha256=sha(path),role=actor.role,usage=usage,opportunity=opportunity)

def load_export(entry,*,role,usage):
    if sha(entry['path'])!=entry['sha256']:raise ValueError('Actor-only bytes differ')
    p=torch.load(entry['path'],map_location='cpu',weights_only=True)
    allowed={'schema','correction_schema','feature_schema','role','usage','opportunity','base','input_dim','hidden','amplitude','distances_enabled','seed','correction','correction_sha256'}
    if set(p)!=allowed or p['schema']!=SCHEMA or p['correction_schema']!=CORRECTION_SCHEMA:raise ValueError('Foreign/training schema')
    if p['role']!=role or p['usage']!=usage:raise ValueError('Actor role or stage mismatch')
    if (p['feature_schema'],p['input_dim'],p['hidden'],p['amplitude'],p['distances_enabled'],p['seed'])!=(old.FEATURE_SCHEMA,206,[64,64],.25,True,11622026):raise ValueError('Actor feature semantics mismatch')
    actor=load_actor(role)
    if actor.base_identity!=p['base']:raise ValueError('Accurate SL256 lineage mismatch')
    actor.correction.load_state_dict(p['correction'],strict=True)
    if old.tensor_digest(actor.correction.state_dict())!=p['correction_sha256']:raise ValueError('Correction hash mismatch')
    actor.eval().requires_grad_(False)
    return actor
