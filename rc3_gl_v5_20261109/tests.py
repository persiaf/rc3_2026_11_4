"""New baseline mathematics and explicit role checks on synthetic fixtures only."""
import copy
import itertools
import math
import time
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from .common import *
from .objective import fixed_reference_objective
from .identities import actor_identity
from .reference import FrozenReference,rng_guard,rng_fingerprint


def verify():
    start=time.perf_counter()
    from rc3_gl_v4_20261109.tests import ToyModel,ToyEnvironment
    from rc3_global_local_trainer.rng import NamedStreams
    from .mechanism import GreedyBaselineTrainer
    view=SimpleNamespace(C=np.zeros((1,2)),identity='SYNTHETIC_GLV5',parent=SimpleNamespace(identity='SYNTHETIC_PARENT'))
    def make():
        model=ToyModel();tr=GreedyBaselineTrainer(model,None,NamedStreams(OLD_PROTOCOL,SEED),None,dict(phase='RL',batch_instances=4))
        tr.reference=lambda v:dict(J=1.5,synthetic=True)
        tr.instance_gradient_norms=[]
        return model,tr
    model,tr=make()
    with patch('rc3_gl_v5_20261109.mechanism.Environment',ToyEnvironment):
        loss,d=tr.policy_instance_loss(view,0,0)
    z=model.logits.detach().clone().requires_grad_(True);lp=torch.log_softmax(z,0)
    manual=sum(w*(r['J']-1.5)*lp[r['actions'][0]-1] for w,r in zip((1,.25,.25),d['rows']))/1.5
    got=torch.autograd.grad(loss,model.logits)[0];want=torch.autograd.grad(manual,z)[0]
    require(torch.allclose(got,want,atol=1e-12,rtol=0),'Production gradient/sign differs')
    require(all(r['forced_first'] is None and r['logprobs'][0] is not None for r in d['rows']),'First action likelihood omitted')
    gradients={}
    for mode in ('accumulate','joint'):
        m,t=make()
        with patch('rc3_gl_v5_20261109.mechanism.Environment',ToyEnvironment):b=t.batch_gradients([view]*4,0,mode=mode)
        gradients[mode]=m.logits.grad.clone()
    require(torch.allclose(gradients['accumulate'],gradients['joint'],atol=1e-12,rtol=0),'Macro batch gradient mismatch')
    baseline=torch.tensor(1.5,dtype=torch.float64,requires_grad=True);x=torch.tensor(.2,dtype=torch.float64,requires_grad=True)
    objective,a=fixed_reference_objective([x,x,x],[1.,2.,2.],baseline)
    require(a.tolist()==[.5,-.5,-.5] and not a.requires_grad,'Negative advantages dropped or baseline attached')
    require(torch.autograd.grad(objective,baseline,allow_unused=True)[0] is None,'Baseline receives gradient')
    p=1/(1+math.exp(-.3));expected=[]
    for baseline_value in (-2.,1.5,20.):
        value=0.
        for actions in itertools.product((0,1),repeat=3):
            probability=math.prod(p if q else 1-p for q in actions)
            costs=[2-q for q in actions]
            value+=probability*sum(w*(c-baseline_value)*(q-p) for w,c,q in zip((1,.25,.25),costs,actions))/1.5
        require(abs(value+p*(1-p))<1e-12,'Frozen baseline changed expected objective')
        expected.append(value)
    streams=NamedStreams(OLD_PROTOCOL,SEED);streams.torch('RL',0,0,'primary');before=rng_fingerprint(streams)
    require(rng_guard(streams,lambda: {'J':1.})=={'J':1.} and rng_fingerprint(streams)==before,'Reference pollutes RNG')
    rejected=0
    try:rng_guard(streams,lambda:torch.rand(1,generator=streams.torch('RL',0,0,'primary')))
    except ValueError:rejected+=1
    entry=read(HOME/'INPUT_REGISTRY.json')['actors']['B128']
    for variant in ('role','source_role','endpoint'):
        wrong=copy.deepcopy(entry)
        if variant=='role':wrong['role']='SL256'
        elif variant=='source_role':wrong['provenance']['role']='SL256'
        else:wrong['provenance']['training_checkpoint_path']=wrong['provenance']['training_checkpoint_path'].replace('0128','0127')
        try:actor_identity('B128',wrong)
        except ValueError:rejected+=1
    require(rejected==4,'RNG/role negative test accepted a mismatch')
    # Instantiate the actual cache owners, without calling either real terminal.
    from .mechanism import RecordingTerminal
    fixture=HOME/'engineering_fixture';clock=Deadline(time.perf_counter()+30);events=[]
    ref=FrozenReference(fixture/'reference',clock,events.append,read(HOME/'INPUT_REGISTRY.json')['actors']['SL256'])
    sample=RecordingTerminal({},fixture/'sample',H_ID,clock,events.append)
    require(ref.terminal.cache is not sample.inner.cache and ref.terminal.folder!=sample.inner.folder,'Caches/prefix owners shared')
    ref.terminal.cache[('SYNTHETIC','H',(0,1,0))]={'J':1}
    require(not sample.inner.cache and sample.request_count==0 and ref.requests==0,'Cache contamination or hidden query')
    # Reference production __call__ with only mocked model/greedy/H, no scientific policy or planner.
    frozen=ToyModel().eval().requires_grad_(False)
    from rc3_global_local.policy import tensor_hash
    fake=copy.deepcopy(ref.entry);fake['model_tensor_sha256']=tensor_hash(frozen)
    class ToyCatalog:
        def __init__(self,*a,**k):pass
        def load(self,*a,**k):return frozen,fake
    ref.terminal=lambda p,r:dict(J=1.,L=1.,K=0,raw_J=2.,raw_L=2.,raw_K=0,route=r,cache_hit=False)
    streams=NamedStreams(OLD_PROTOCOL,SEED)
    with patch('rc3_gl_v5_20261109.reference.Catalog',ToyCatalog),patch('rc3_gl_v5_20261109.reference.greedy',lambda *a:([0,1,0],[])):
        rng_guard(streams,lambda:ref(view))
    require(time.perf_counter()-start<120,'Engineering time cap')
    write_new(HOME/'ENGINEERING_VERIFICATION.json',dict(status='PASS',source_binding_sha256=digest(HOME/'SOURCE_BINDING.json'),
        production_loss_gradient=got.tolist(),independent_loss_gradient=want.tolist(),batch_gradients={k:v.tolist() for k,v in gradients.items()},
        expected_gradients_with_different_state_baselines=expected,analytic_gradient=-p*(1-p),
        signed_advantages_preserved=[.5,-.5,-.5],baseline_gradient=None,first_actions_all_scored=True,
        disjoint_cache_owners=True,production_reference_wrapper_synthetic_test=True,rejected_mismatches=rejected,
        scientific_requests=0,scientific_policy_forwards=0,optimizer_updates=0,navigation_native_H=0,
        fixture_only_mock_reference_requests=1,wall_s=time.perf_counter()-start))
    print('GL-v5 synthetic mathematics/RNG/role/cache PASS',flush=True)
