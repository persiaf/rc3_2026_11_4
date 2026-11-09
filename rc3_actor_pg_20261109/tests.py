"""Pre-registered zero-science tensor, lifecycle and exact gradient unit tests."""
import io
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from .common import *
from .artifacts import compare_tensors,Catalog,ARCH,load_actor
from .mechanism import IIDTrainer,AuditTrainer
from rc3_global_local_trainer.rng import NamedStreams
from rc3_global_local.policy import tensor_hash


class ToyEnvironment:
    def __init__(self,p,terminal):
        self.problem=p;self.terminal=terminal;self.done=False;self.prefix=[];self.rewards=[];self.result=None
    def mask(self):
        mask=np.ones(3,bool);mask[np.array(self.prefix,dtype=int)-1]=False
        if not self.prefix and self.problem.one_first:mask=np.array([False,True,False])
        return mask
    def step(self,node):
        require(1<=node<=3 and self.mask()[node-1],'Illegal toy action')
        self.prefix.append(node);self.done=len(self.prefix)==2
        if self.done:
            self.result=self.terminal(self.problem,[0,*self.prefix,0]);self.rewards.append(-self.result['J'])
        else:self.rewards.append(0.)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.theta=torch.nn.Parameter(torch.tensor([.2,-.4,.7],dtype=torch.float64));self.encodings=[]
    def encode(self,p):
        e=dict(problem=p.identity);self.encodings.append(e);return e
    def score(self,env,encoded):return self.theta.masked_fill(~torch.tensor(env.mask()),-torch.inf)


def toy_view(one=False):return SimpleNamespace(identity='synthetic-view',parent=SimpleNamespace(identity='synthetic-parent'),C=np.ones((2,3)),one_first=one)
def toy_terminal(p,raw):
    j=float(10+2*raw[1]+raw[2]);return dict(J=j,raw_J=j,L=j-1.5,K=2,route=raw)


class Verification(unittest.TestCase):
    def test_exact_tensor_schema_and_negative_values(self):
        good={'a':torch.arange(6,dtype=torch.float32).reshape(2,3)}
        self.assertEqual(compare_tensors(good,{k:v.clone() for k,v in good.items()})[0]['shape'],[2,3])
        bad=[{'b':good['a']},{'a':good['a'].double()},{'a':good['a'].reshape(3,2)},
             {'a':good['a']+1},{'a':torch.full((2,3),float('inf'))}]
        for value in bad:
            with self.assertRaises(ValueError):compare_tensors(good,value)

    def test_B_likelihood_gradient_matches_independent_formula(self):
        model=ToyModel();trainer=IIDTrainer(model,None,NamedStreams(OLD_PROTOCOL,SEED),toy_terminal)
        with patch('rc3_actor_pg_20261109.mechanism.Environment',ToyEnvironment):loss,d=trainer.policy_instance_loss(toy_view(),0,0)
        loss.backward();actual=model.theta.grad.clone()
        t=model.theta.detach().clone().requires_grad_();terms=[]
        returns=[r['return_'] for r in d['rows']]
        for i,row in enumerate(d['rows']):
            lp=torch.zeros((),dtype=t.dtype);mask=torch.ones(3,dtype=torch.bool)
            for node in row['actions']:
                lp=lp+torch.log_softmax(t.masked_fill(~mask,-torch.inf),0)[node-1];mask=mask.clone();mask[node-1]=False
            advantage=returns[i]-sum(returns[j] for j in range(3) if j!=i)/2
            terms.append(-advantage*lp)
        independent=sum(terms)/3;independent.backward()
        torch.testing.assert_close(loss,independent,rtol=0,atol=1e-12)
        torch.testing.assert_close(actual,t.grad,rtol=0,atol=1e-12)
        self.assertEqual(d['mixture_weights'],[1,1,1]);self.assertEqual(len({r['sampling_seed'] for r in d['rows']}),3)
        self.assertTrue(all(r['forced_first'] is None and len(r['logprobs'])==2 and r['logprobs'][0] is not None for r in d['rows']))
        self.assertEqual(len({id(e) for e in model.encodings}),3)
        self.assertAlmostEqual(sum(d['advantages']),0.)

    def test_A_original_forced_and_single_first(self):
        import rc3_global_local_trainer.engine as original
        trainer=AuditTrainer(ToyModel(),None,NamedStreams(OLD_PROTOCOL,SEED),toy_terminal)
        with patch.object(original,'Environment',ToyEnvironment):_,d=trainer.policy_instance_loss(toy_view(True),0,0)
        self.assertEqual(d['mixture_weights'],[1,.25,.25]);self.assertEqual(len(d['rows']),3)
        self.assertTrue(all(r['actions'][0]==2 and r['unique_legal_first']==1 for r in d['rows']))
        self.assertIsNotNone(d['rows'][0]['logprobs'][0]);self.assertTrue(all(r['logprobs'][0] is None for r in d['rows'][1:]))

    def test_B_macro_batch_gradient_accumulation(self):
        a=IIDTrainer(ToyModel(),None,NamedStreams(OLD_PROTOCOL,SEED),toy_terminal)
        b=IIDTrainer(ToyModel(),None,NamedStreams(OLD_PROTOCOL,SEED),toy_terminal)
        with patch('rc3_actor_pg_20261109.mechanism.Environment',ToyEnvironment):
            x=a.batch_gradients([toy_view() for _ in range(4)],0,'accumulate')
            y=b.batch_gradients([toy_view() for _ in range(4)],0,'joint')
        self.assertAlmostEqual(x['loss'],y['loss'],places=12)
        torch.testing.assert_close(a.model.theta.grad,b.model.theta.grad,rtol=0,atol=1e-12)

    def test_actor_resident_parameter_only_identity(self):
        model=ToyModel().eval().requires_grad_(False)
        e=dict(checkpoint_sha256='a'*64,model_tensor_sha256=tensor_hash(model))
        with patch('rc3_actor_pg_20261109.artifacts.load_actor',return_value=model) as loader:
            c=Catalog('D2',{str(SEED):{'untrained':e}})
            x,_=c.load(SEED,'untrained',Meter());y,_=c.load(SEED,'untrained',Meter())
            self.assertIs(x,y);self.assertEqual(loader.call_count,1);self.assertEqual(c.hits,2)
            self.assertFalse(any('encod' in k or 'route' in k for k in c.__dict__))
            with self.assertRaises(ValueError):c.load(SEED+1,'untrained',Meter())
            with self.assertRaises(ValueError):Catalog('invalid',{})

    def test_actor_tampering_and_foreign_architecture_rejected(self):
        model=ToyModel();path=HOME/'synthetic_fixtures/invalid_actor.pt'
        torch_new(path,dict(schema='rc3-actor-only-v1',architecture=dict(ARCH,hidden=32),model=model.state_dict(),
            model_tensor_sha256=tensor_hash(model),provenance=dict(seed=SEED,role='synthetic')))
        e=dict(path=label(path),checkpoint_sha256=digest(path),model_tensor_sha256=tensor_hash(model),provenance=dict(seed=SEED,role='synthetic'))
        with self.assertRaisesRegex(ValueError,'architecture/source'):load_actor(e)
        with self.assertRaisesRegex(ValueError,'file changed'):load_actor(dict(e,checkpoint_sha256='0'*64))

    def test_streams_independent_and_replay(self):
        s=NamedStreams(OLD_PROTOCOL,SEED);state=s.state_dict()
        values=[torch.rand(4,generator=s.torch('RL',0,0,role)) for role in ('policy_0','policy_1','policy_2')]
        self.assertFalse(torch.equal(values[0],values[1]));self.assertFalse(torch.equal(values[1],values[2]))
        s.load_state_dict(state)
        self.assertTrue(all(torch.equal(v,torch.rand(4,generator=s.torch('RL',0,0,role))) for v,role in zip(values,('policy_0','policy_1','policy_2'))))

    def test_durable_log_and_single_writer(self):
        folder=HOME/'synthetic_fixtures/journal';log=Log(folder)
        try:
            log(dict(event='synthetic_predebit',request=1));log(dict(event='synthetic_complete',request=1))
            with self.assertRaises(Exception):Log(folder)
        finally:log.close()
        events,tail=read_chain(folder/'events.jsonl');self.assertEqual(len(events),2)
        self.assertEqual(read(folder/'HEAD.json')['sha256'],tail)


def main():
    started=time.perf_counter();require((HOME/'PROTOCOL.json').exists() and (HOME/'SOURCE_BINDING.json').exists(),'Tests must follow preregistration/source freeze')
    buffer=io.StringIO();result=unittest.TextTestRunner(stream=buffer,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Verification))
    with (HOME/'ENGINEERING_TEST_OUTPUT.txt').open('x',encoding='utf8') as f:f.write(buffer.getvalue())
    elapsed=time.perf_counter()-started
    write_new(HOME/'ENGINEERING_VERIFICATION.json',dict(status='PASS' if result.wasSuccessful() and elapsed<=120 else 'FAIL',
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),wall_s=elapsed,wall_cap_s=120,
        source_binding_sha256=digest(HOME/'SOURCE_BINDING.json'),real_optimizer_updates=0,real_model_inference=0,
        new_scene_navigation_native_H_teacher=0,synthetic_tensor_autograd_only=True))
    print(buffer.getvalue());require(result.wasSuccessful() and elapsed<=120,'Bound engineering tests failed')


if __name__=='__main__':main()
