"""Paid frozen SL256 baseline; separate model, prefix, cache, and request account."""
import hashlib
import json
import torch
from .common import *
from .identities import actor_identity
from rc3_gl_v4_20261109.artifacts import Catalog
from rc3_gl_v4_20261109.evaluation import greedy
from rc3_global_local.policy import tensor_hash
from rc3_global_local_trainer.runner import BudgetedTerminal


def rng_fingerprint(streams):
    h=hashlib.sha256()
    for k,v in sorted(streams._torch.items()):
        h.update(k.encode());h.update(v.get_state().numpy().tobytes())
    for k,v in sorted(streams._numpy.items()):
        h.update(k.encode());h.update(json.dumps(v.bit_generator.state,sort_keys=True).encode())
    h.update(torch.get_rng_state().numpy().tobytes())
    return h.hexdigest()


def rng_guard(streams, fn):
    before=rng_fingerprint(streams);result=fn()
    require(before==rng_fingerprint(streams),'Frozen reference consumed policy/global RNG')
    return result


class FrozenReference:
    def __init__(self,folder,clock,log,entry):
        self.folder,self.clock,self.log,self.entry=folder,clock,log,entry
        self.identity=actor_identity('SL256',entry);self.requests=0;self.ledger={};self.operations=[]
        self.terminal=BudgetedTerminal(self.ledger,folder/'H_plans',H_ID,{'frozen_SL_reference':512},clock,log,cache_size=128)

    def __call__(self,p):
        self.clock.check();require(self.requests<512,'Reference request budget exhausted')
        self.requests+=1;n=self.requests;self.log(dict(event='reference_request_predebit',request=n,problem=p.identity,model_identity=self.identity))
        meter=Meter(self.log,self.clock.check)
        try:
            catalog=Catalog('D1',{str(SEED):{'SL256':self.entry}},override_role='SL256')
            model,entry=catalog.load(SEED,'frozen_supervised',meter)
            require(all(not x.requires_grad and x.grad is None for x in model.parameters()),'Reference is not frozen')
            raw,trace=greedy(model,p,meter)
            with meter.span('reference_H_and_validation'):result=self.terminal(p,raw)
            require(tensor_hash(model)==entry['model_tensor_sha256'],'Reference parameters mutated')
            record=dict(request=n,problem=p.identity,parent=p.parent.identity,model_identity=self.identity,
                raw_route=raw,trace=trace,result=result,counts=meter.counts,seconds=meter.seconds,
                cache_namespace='frozen_SL_reference_only',policy_sampling_RNG_used=False)
            path=self.folder/'greedy_results'/f'reference_{n:04}.json';write_new(path,record)
            self.operations.append(dict(request=n,counts=meter.counts,seconds=meter.seconds))
            answer=dict(J=result['J'],L=result['L'],K=result['K'],raw_J=result['raw_J'],raw_L=result['raw_L'],raw_K=result['raw_K'],
                request=n,artifact_path=label(path),artifact_sha256=digest(path),model_identity=self.identity,
                raw_route=raw,route=result['route'],cache_hit=result['cache_hit'])
            self.log(dict(event='reference_request_complete',**answer));self.clock.check()
            return answer
        except BaseException as exc:
            self.log(dict(event='reference_request_error',request=n,error=repr(exc)));raise

    def state_dict(self):
        return dict(requests=self.requests,model_identity=self.identity,ledger=self.ledger,
            operations=self.operations,terminal=self.terminal.state_dict(),dynamic_prefix_cached=False)
