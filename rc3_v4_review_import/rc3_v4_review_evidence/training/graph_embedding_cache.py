"""Isolated audit adapter: cache an invariant graph embedding within one Env.

The source policy is unchanged. F, D, candidate/task semantics and model weights
must remain fixed for the duration of an Env, exactly as required by its own
route/feature caches. A new environment or route/window layout invalidates this
cache. Budget statistics, valid-action masks, all heads and STOP pooling are
recomputed on every action. This is not a new trained policy.
"""
from __future__ import annotations
import time
import numpy as np
from rc3_v4.model import NumpyPolicy
from rc3_v4.features import local_graphs,collate
from rc3_v3.model import select_action


class CapturingModel(NumpyPolicy):
    def linear(self,x,name):
        value=super().linear(x,name)
        if name=='graph_pool':
            self.captured_graph=value.copy()
        return value


class CachedGraphPolicy:
    def __init__(self,model):
        self.model=model
        self._env=None;self._key=None;self._graph=None
        self.graph_s=0.;self.inference_s=0.;self.calls=0;self.stops=0
        self.cache_hits=0;self.cache_misses=0;self.last_outputs=None

    @classmethod
    def load(cls,path):
        model=CapturingModel.load(path)
        return cls(model)

    def inference_session(self):return self.model.inference_session()

    def outputs_env(self,env,features,mask):
        tick=time.perf_counter()
        key=(tuple(env.route),tuple((w['start'],w['end']) for w in env.windows))
        if self._env is not env or self._key!=key:
            graphs=local_graphs(env.problem,env.route,env.windows)
            batch=collate([dict(features=features,mask=mask,graphs=graphs)])
            self.graph_s+=time.perf_counter()-tick
            tick=time.perf_counter()
            out=self.model.outputs(batch)
            self._graph=self.model.captured_graph.copy()
            self._env,self._key=env,key
            self.cache_misses+=1
        else:
            base=np.asarray(features,np.float32)[None,:,:]
            mask=np.asarray(mask,bool)[None,:]
            if base.shape!=(1,len(env.windows),28) or mask.shape!=(1,len(env.windows)) or not mask.any() or not np.isfinite(base).all():
                raise ValueError('invalid current observation')
            self.graph_s+=time.perf_counter()-tick
            tick=time.perf_counter()
            m=self.model
            common=m.linear((base-m.w['mean'])/m.w['std'],'base_embed')
            h=np.maximum(common+self._graph,0.)
            h=np.maximum(m.linear(h,'mlp.1'),0.)
            h=np.maximum(m.linear(h,'mlp.3'),0.)
            rank=np.where(mask,m.linear(h,'rank')[...,0],-np.inf)
            pooled=(h*mask[...,None]).sum(1)/mask.sum(1).astype(np.float32)[:,None]
            stop=m.linear(np.maximum(m.linear(pooled,'stop.0'),0.),'stop.2')
            out=(np.concatenate((rank,stop),-1),m.linear(h,'gain')[...,0],m.linear(h,'positive')[...,0])
            self.cache_hits+=1
        self.inference_s+=time.perf_counter()-tick
        self.last_outputs=out
        return out

    def select_env(self,env,features,mask):
        out=self.outputs_env(env,features,mask)
        action=select_action(out[0][0],mask,'binary_threshold',.9)
        self.calls+=1;self.stops+=int(action is None)
        return action
