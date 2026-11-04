"""Trusted local checkpoints with explicit architecture and protocol identity."""
import hashlib
import math
from pathlib import Path

def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_identity():
    root=Path(__file__).resolve().parent
    core=('actor_critic.py','graph_observation.py','env_g2.py','action_memory.py','adapter_g2.py',
          'collector.py','collector_contract.py','worker.py','train_ppo.py','ppo_math.py',
          'teacher_dataset.py','checkpoints.py','prepared_cache.py')
    legacy=('rc3_rl_research/types.py','rc3_rl_research/adapter.py','rc3_rl_research/action_builder.py',
            'rc3_rl_research/repair_proposal.py','rc3_rl_research/rl_env.py',
            'rc3_fast/solver.py','rc3_fast/scenes.py','rc3_baseline/baseline.py','rc3_baseline/validator.py',
            'rc3_improve/environment.py','rc3_improve/repair.py','rc3_v3/environment.py','rc3_v5/external.py')
    return {**{'rc3_rl_g2/'+name:sha256(root/name) for name in core},
            **{name:sha256(root.parent/name) for name in legacy}}

def capture_rng(device='cpu'):
    import torch
    return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.device(device).type=='cuda' else None)

def restore_rng(state):
    import torch
    torch.set_rng_state(state['cpu'].cpu())
    if state.get('cuda') is not None:torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])

def validate_training_state(payload,protocol,parent_sha,logged_row=None):
    settings=protocol['ppo'];batch=settings['workers']*settings['rollout_steps']
    for key in ('iteration','actual_actions','optimizer_updates'):
        if type(payload[key]) is not int or payload[key]<0:raise ValueError('invalid training counter '+key)
    if payload['actual_actions']!=payload['iteration']*batch or payload['actual_actions']>settings['actions']:
        raise ValueError('checkpoint iteration/action counts disagree')
    max_updates=payload['iteration']*settings['epochs']*((batch+settings['minibatch']-1)//settings['minibatch'])
    if payload['optimizer_updates']>max_updates:raise ValueError('impossible optimizer update count')
    if payload['parent_bc_sha256']!=parent_sha:raise ValueError('BC initialization identity changed')
    for key in ('elapsed_s','validation_s'):
        value=payload.get(key,0.)
        if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or value<0:
            raise ValueError('invalid saved elapsed time')
    if logged_row is not None and any(logged_row[key]!=payload[key] for key in ('iteration','actual_actions','optimizer_updates')):
        raise ValueError('training log/checkpoint counters disagree; audit uncheckpointed interactions first')
    return True

def _finite_parameters(value):
    import torch
    if isinstance(value,dict):
        for item in value.values():_finite_parameters(item)
    elif isinstance(value,(tuple,list)):
        for item in value:_finite_parameters(item)
    elif isinstance(value,torch.Tensor) and not torch.isfinite(value).all():raise ValueError('nonfinite saved parameter/optimizer tensor')
    elif isinstance(value,float) and not math.isfinite(value):raise ValueError('nonfinite saved optimizer scalar')

def load_model(path,device='cpu'):
    import torch
    from .actor_critic import ActorCritic
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload.get('source_sha256')!=source_identity():raise ValueError('checkpoint implementation identity changed')
    _finite_parameters(payload['model'])
    if 'optimizer' in payload:_finite_parameters(payload['optimizer'])
    architecture=payload['architecture']
    model=ActorCritic(**{k:architecture[k] for k in ('kind','hidden','message_steps')}).to(device)
    model.load_state_dict(payload['model']);return model,payload

def save_model(path,model,**metadata):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(schema='rc3-g2-checkpoint-v1',architecture=model.architecture(),
                 model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                 source_sha256=source_identity(),training_device=str(model.device),**metadata)
    temp=path.with_suffix('.pending');torch.save(payload,temp);temp.replace(path)
    return sha256(path)
