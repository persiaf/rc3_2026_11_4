"""Exact actor tensors; no optimizer, RNG or task caches in inference files."""
import copy
import time
import torch
from rc3_global_local.policy import GlobalLocalPolicy,tensor_hash
from rc3_global_local_trainer.checkpoint import _state_tensor_hash
from rc3_s3_evaluator.provenance import ModelCatalog
from .common import HOME,ROOT,SEED,MODEL_ROLES,read,write_new,digest,require,label,torch_new

ARCH=dict(class_name='rc3_global_local.policy.GlobalLocalPolicy',hidden=64,
    global_dim=14,task_dim=5,candidate_dim=7,input_contract='original whole-parent/local coverage/prefix features',
    policy_source_sha256=digest(ROOT/'rc3_global_local/policy.py'))


def compare_tensors(a,b):
    require(set(a)==set(b),'Parameter names differ')
    report=[]
    for name in sorted(a):
        x,y=a[name],b[name]
        require(x.shape==y.shape and x.dtype==y.dtype and torch.equal(x,y),'Parameter shape/dtype/value differs: '+name)
        require(bool(torch.isfinite(x).all()),'Nonfinite actor tensor')
        import hashlib
        h=hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()
        report.append(dict(name=name,shape=list(x.shape),dtype=str(x.dtype),value_sha256=h))
    require(_state_tensor_hash(a)==_state_tensor_hash(b),'Tensor hash differs')
    return report


def export_one(source,folder,seed,role,expected_tensor=None):
    start=time.perf_counter();sha=digest(source)
    state=torch.load(source,map_location='cpu',weights_only=False)
    values={k:v.detach().cpu().clone() for k,v in state['model'].items()}
    tensor=_state_tensor_hash(values);require(tensor==state['model_tensor_sha256'],'Training payload tensor identity differs')
    if expected_tensor:require(tensor==expected_tensor,'Wrong fixed endpoint tensor')
    payload=dict(schema='rc3-actor-only-v1',architecture=ARCH,model=values,model_tensor_sha256=tensor,
        provenance=dict(training_checkpoint_path=label(source),training_checkpoint_sha256=sha,seed=seed,role=role))
    path=folder/f'{seed}_{role}.pt';file=torch_new(path,payload)
    saved=torch.load(path,map_location='cpu',weights_only=True)
    require(set(saved)==set(payload) and saved['architecture']==ARCH and saved['provenance']==payload['provenance'],'Actor has extra/mismatched state')
    params=compare_tensors(state['model'],saved['model'])
    require(digest(source)==sha,'Export modified training checkpoint')
    entry=dict(path=file['path'],checkpoint_sha256=file['sha256'],checkpoint_bytes=file['bytes'],
        model_tensor_sha256=tensor,phase='frozen_actor',cursor=role,role=role,seed=seed,provenance=payload['provenance'],
        parameters=params,export_wall_s=time.perf_counter()-start)
    write_new(path.with_suffix('.pt.json'),entry);return entry


def export_all():
    index=read(ROOT/'rc3_s3_evaluator/S3_MODEL_INDEX.json');models={}
    for seed,roles in index['models'].items():
        models[seed]={}
        for role,m in roles.items():
            require(digest(ROOT/m['path'])==m['checkpoint_sha256'],'Frozen model source changed')
            models[seed][role]=export_one(ROOT/m['path'],HOME/'actors',int(seed),role,m['model_tensor_sha256'])
    write_new(HOME/'actors/ACTOR_INDEX.json',dict(models=models,architecture=ARCH,source_model_index_sha256=digest(ROOT/'rc3_s3_evaluator/S3_MODEL_INDEX.json'),
        exactly_equal=True,quantization_pruning_distillation=False))
    return models


def load_actor(entry):
    p=ROOT/entry['path'];require(digest(p)==entry['checkpoint_sha256'],'Actor file changed')
    state=torch.load(p,map_location='cpu',weights_only=True)
    require(state['schema']=='rc3-actor-only-v1' and state['architecture']==ARCH and state['provenance']==entry['provenance'],'Wrong actor architecture/source')
    require(_state_tensor_hash(state['model'])==entry['model_tensor_sha256'],'Actor tensor identity differs')
    before=torch.get_rng_state().clone()
    with torch.device('meta'):model=GlobalLocalPolicy(hidden=64)
    model.load_state_dict(state['model'],strict=True,assign=True);model.eval().requires_grad_(False)
    require(tensor_hash(model)==entry['model_tensor_sha256'] and torch.equal(before,torch.get_rng_state()),'Actor materialization changed values/RNG')
    return model


class Catalog:
    def __init__(self,mode,models,seed=SEED,override_role=None):
        require(mode in ('D0','D1','D2'),'Unknown model lifecycle')
        self.mode,self.models,self.seed,self.override_role=mode,models,seed,override_role
        self.resident={};self.full=ModelCatalog() if mode=='D0' else None
        self.loads=0;self.hits=0
        if mode=='D2':
            for role,entry in models[str(seed)].items():
                self.resident[(seed,entry['checkpoint_sha256'])]=load_actor(entry);self.loads+=1

    def load(self,seed,method,meter):
        require(seed==self.seed,'Foreign actor seed')
        if self.mode=='D0':
            self.loads+=1;return self.full.load(seed,method,meter)
        role=self.override_role or MODEL_ROLES[method];e=self.models[str(seed)][role]
        with meter.span('model_load_and_verify' if self.mode=='D1' else 'resident_actor_access'):
            if self.mode=='D1':model=load_actor(e);self.loads+=1;meter.add('frozen_policy_materializations')
            else:
                key=(seed,e['checkpoint_sha256']);require(key in self.resident,'Wrong resident model identity')
                model=self.resident[key];self.hits+=1;meter.add('resident_model_hits')
            require(tensor_hash(model)==e['model_tensor_sha256'],'Resident actor mutated')
        return model,e
