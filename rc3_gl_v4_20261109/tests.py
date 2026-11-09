"""Only new-start mathematics on synthetic logits, no scientific policy forward."""
import itertools
import math
import time
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from .common import *


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.logits=torch.nn.Parameter(torch.tensor([.2,-.3],dtype=torch.float64))
    def encode(self,view):return None
    def score(self,env,encoded):return self.logits


class ToyEnvironment:
    def __init__(self,view,terminal):
        self.done=False;self.prefix=[];self.rewards=[];self.result=None
    def mask(self):return np.ones(2,dtype=bool)
    def step(self,node):
        self.prefix.append(node);self.done=True;cost=float(3-node)
        self.result=dict(J=cost,raw_J=cost);self.rewards=[-cost]


def verify():
    start=time.perf_counter()
    from rc3_global_local_trainer.rng import NamedStreams
    from .mechanism import NaturalTrainer,AuditTrainer
    require(digest(HOME/'PROTOCOL.json')==read(HOME/'SOURCE_BINDING.json')['protocol_sha256'],'Unregistered math probe')
    weights=[1.,.25,.25];view=SimpleNamespace(C=np.zeros((1,2)),identity='synthetic-only',parent=SimpleNamespace(identity='synthetic-parent'))
    results={}
    for name,cls,target in [('A',AuditTrainer,'rc3_global_local_trainer.engine.Environment'),('B',NaturalTrainer,'rc3_gl_v4_20261109.mechanism.Environment')]:
        model=ToyModel();streams=NamedStreams(OLD_PROTOCOL,SEED)
        trainer=cls(model,None,streams,None,dict(phase='RL',mixture_weights=weights))
        with patch(target,ToyEnvironment):loss,d=trainer.policy_instance_loss(view,0,0)
        require(d['mixture_weights']==weights,'B silently changed mixture weights')
        if name=='B':require(all(r['forced_first'] is None and r['logprobs'][0] is not None for r in d['rows']),'Natural first action omitted')
        else:require(d['rows'][0]['logprobs'][0] is not None and all(r['logprobs'][0] is None for r in d['rows'][1:]),'Original forced convention changed')
        g=torch.autograd.grad(loss,model.logits)[0]
        z=model.logits.detach().clone().requires_grad_(True);lp=torch.log_softmax(z,0)
        terms=[-w*a*lp[r['actions'][0]-1] for w,a,r in zip(weights,d['advantages'],d['rows']) if r['logprobs'][0] is not None]
        manual=sum(terms)/sum(weights);ref=torch.autograd.grad(manual,z)[0]
        require(torch.allclose(g,ref,atol=1e-12,rtol=0),'Actual objective gradient differs from independent expression')
        require(len({r['sampling_seed'] for r in d['rows']})==3,'Trajectory RNG stream collision')
        results[name]=dict(weights=d['mixture_weights'],gradient=g.tolist(),reference_gradient=ref.tolist(),first_logprobs=[r['logprobs'][0] for r in d['rows']],advantages=d['advantages'])
    # Enumerate the stochastic estimator, not the sampled-loss implementation.
    p=1/(1+math.exp(-.3));estimate=0.
    for actions in itertools.product((0,1),repeat=3):
        prob=math.prod(p if a else 1-p for a in actions);returns=[-2.+a for a in actions]
        advantages=[returns[i]-sum(returns[j] for j in range(3) if j!=i)/2 for i in range(3)]
        estimate+=prob*sum(-w*a*(x-p) for w,a,x in zip(weights,advantages,actions))/sum(weights)
    analytic=-p*(1-p)
    require(abs(estimate-analytic)<1e-12,'Weighted natural-start LOO is not unbiased in analytic toy')
    registry=read(HOME/'INPUT_REGISTRY.json');require(len(registry['training_batches'])==128 and sum(map(len,registry['training_batches']))==512,'Bad frozen schedule')
    require(identity(registry['training_batches'])==registry['training_schedule_sha256'],'Schedule identity changed')
    require(time.perf_counter()-start<120,'Engineering wall cap')
    write_new(HOME/'ENGINEERING_VERIFICATION.json',dict(status='PASS',source_binding_sha256=digest(HOME/'SOURCE_BINDING.json'),
        synthetic_objective_checks=results,expected_gradient_enumeration=estimate,analytic_gradient=analytic,
        exact_A_B_schedule_views=512,actual_optimizer_updates=0,scientific_policy_forwards=0,
        synthetic_logits_only=True,navigation_native_H_teacher=0,wall_s=time.perf_counter()-start))
    print('GL-v4 new-start synthetic mathematics PASS; zero scientific calls',flush=True)
