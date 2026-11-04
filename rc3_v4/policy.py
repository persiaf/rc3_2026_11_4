"""Environment adapter and one-session BLAS management for frozen policies."""
import time
from contextlib import nullcontext
from .features import local_graphs, collate
from rc3_v3.model import select_action


class GraphPolicy:
    def __init__(self, model):
        self.model=model
        self._env=None;self._key=None;self._graphs=None
        self.graph_s=0.;self.inference_s=0.;self.calls=0;self.stops=0

    def inference_session(self):
        return self.model.inference_session() if hasattr(self.model,'inference_session') else nullcontext()

    def select_env(self, env, features, mask):
        tick=time.perf_counter();key=tuple(env.route)
        if self._env is not env or self._key!=key:
            self._graphs=local_graphs(env.problem,env.route,env.windows)
            self._env,self._key=env,key
        batch=collate([dict(features=features,mask=mask,graphs=self._graphs)])
        self.graph_s+=time.perf_counter()-tick
        tick=time.perf_counter()
        logits=self.model.outputs(batch)[0][0]
        action=select_action(logits,mask,'binary_threshold',.9)
        self.inference_s+=time.perf_counter()-tick
        self.calls+=1;self.stops+=int(action is None)
        return action


def load_policy(path):
    from .model import load_policy as load
    return GraphPolicy(load(path))
