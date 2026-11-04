"""Frozen v3 MLP weights with one thread-control session per full plan.

This is an engineering adapter, not a retrained policy. It uses the original
versioned loader and the same NumPy MLP arithmetic, but does not scan loaded BLAS
libraries or create a threadpool-limits context at each action.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from rc3_v3.model import NumpyPolicy, select_action


class LegacyMLPPolicy:
    def __init__(self, original):
        if not isinstance(original,NumpyPolicy) or original.metadata['architecture']!='mlp':
            raise ValueError('legacy adapter supports only a validated v3 NumPy MLP')
        self.w,self.metadata=original.w,dict(original.metadata)
        self.stop_rule,self.stop_threshold=original.stop_rule,original.stop_threshold
        try:
            import scipy.linalg  # noqa: F401 - discover both NumPy and SciPy BLAS once
            from threadpoolctl import ThreadpoolController
            self._controller=ThreadpoolController()
        except ImportError:
            self._controller=None

    def inference_session(self):
        return self._controller.limit(limits=1,user_api='blas') if self._controller is not None else nullcontext()

    def linear(self,x,name):
        return x @ self.w[name+'.weight'].T+self.w[name+'.bias']

    def outputs(self,features,mask):
        x,mask=np.asarray(features,np.float32),np.asarray(mask,bool)
        if x.shape!=(len(mask),len(self.w['mean'])) or not mask.any() or not np.isfinite(x).all():
            raise ValueError('invalid observation')
        h=self.linear((x-self.w['mean'])/self.w['std'],'embed')
        h=np.maximum(h,0)
        h=np.maximum(self.linear(h,'mlp.1'),0)
        h=np.maximum(self.linear(h,'mlp.3'),0)
        rank=self.linear(h,'rank').reshape(-1)
        rank[~mask]=-np.inf
        pooled=h[mask].mean(0)
        stop=self.linear(np.maximum(self.linear(pooled,'stop.0'),0),'stop.2').reshape(-1)
        return np.r_[rank,stop],self.linear(h,'gain').reshape(-1),self.linear(h,'positive').reshape(-1)

    def scores(self,features,mask):
        return self.outputs(features,mask)[0][:-1]

    def select(self,features,mask):
        return select_action(self.outputs(features,mask)[0],mask,self.stop_rule,self.stop_threshold)


def load_policy(path):
    if Path(path).suffix.lower()!='.npz':
        raise ValueError('legacy engineering adapter requires the frozen NumPy NPZ checkpoint')
    return LegacyMLPPolicy(NumpyPolicy.load(path))


def validation_states(cases_path):
    """Validation-only initial and one-step states, preserving common decisions."""
    from rc3_fast.scenes import make_complex_scene
    from rc3_fast.solver import prepare
    from rc3_v3.environment import Env,FEATURE_NAMES
    cases=json.loads(Path(cases_path).read_text(encoding='utf-8'))
    states=[]
    for case in cases:
        p=prepare(make_complex_scene(case['nx'],case['ny'],case['seed'],case['mode']))
        env=Env(p,8)
        for step in range(2):
            features,mask=env.observe()
            if not mask.any():break
            states.append(dict(seed=case['seed'],step=step,features=features,mask=mask))
            # The shared state sequence follows the rule selector, not either
            # measured implementation, preventing timing from changing states.
            action=int(np.where(mask,features[:,FEATURE_NAMES.index('detour_excess')],-np.inf).argmax())
            env.step(action)
    return states


def profile(model_path,cases_path,output,repeats=30):
    """Measure only inference/context overhead; this is not end-to-end speed."""
    original=NumpyPolicy.load(model_path);optimized=load_policy(model_path)
    states=validation_states(cases_path);records=[];all_equal=True;max_error=0.
    for state in states:
        f,m=state['features'],state['mask']
        old=original.outputs(f,m)
        with optimized.inference_session():new=optimized.outputs(f,m)
        for a,b in zip(old,new):
            finite=np.isfinite(a)&np.isfinite(b)
            if finite.any():max_error=max(max_error,float(np.abs(a[finite]-b[finite]).max()))
            all_equal &= bool(np.array_equal(a,b))
        if original.select(f,m)!=optimized.select(f,m):
            raise AssertionError('legacy adapter changed a validation decision')
        rng=np.random.default_rng(int(state['seed'])+state['step']);samples={'old':[],'optimized':[]}
        for _ in range(repeats):
            for method in rng.permutation(['old','optimized']):
                tick=time.perf_counter()
                if method=='old':original.outputs(f,m)
                else:
                    # Eight repeated actions share one session in deployment.
                    # Keep one session over the repeated inference burst; then
                    # divide its full measured time by eight.
                    with optimized.inference_session():
                        for __ in range(8):optimized.outputs(f,m)
                seconds=time.perf_counter()-tick
                samples[method].append(seconds if method=='old' else seconds/8)
        records.append(dict(seed=state['seed'],step=state['step'],windows=len(m),
            old_per_action_median_ms=float(np.median(samples['old'])*1000),
            optimized_amortized_action_median_ms=float(np.median(samples['optimized'])*1000),
            raw_seconds=samples))
    old=np.array([r['old_per_action_median_ms'] for r in records])
    new=np.array([r['optimized_amortized_action_median_ms'] for r in records])
    result=dict(status='complete',model_sha256=hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
        cases_sha256=hashlib.sha256(Path(cases_path).read_bytes()).hexdigest(),
        cases=len({r['seed'] for r in records}),states=len(records),repeats=repeats,
        all_heads_bitwise_equal=bool(all_equal),max_absolute_error=max_error,all_actions_equal=True,
        median_old_action_ms=float(np.median(old)),median_optimized_amortized_action_ms=float(np.median(new)),
        median_paired_inference_speedup=float(np.median(old/new)),records=records,
        scope='Validation states only. Old creates a BLAS threadpool context per action; optimized amortizes one cached-controller context over eight identical inference calls. Includes context entry/exit, excludes model loading, feature construction, navigation, repair and final validation. Not an end-to-end planner acceleration claim. Frozen weights unchanged; no retraining.')
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,default=Path('rc3_v3/deployment/policy.npz'))
    p.add_argument('--cases',type=Path,default=Path('rc3_v3/data_v3_20261102/validation_cases.json'))
    p.add_argument('--output',type=Path,default=Path('rc3_v4/legacy_profile.json'))
    p.add_argument('--repeats',type=int,default=30)
    a=p.parse_args(argv)
    if a.repeats<1:p.error('repeats must be positive')
    result=profile(a.model,a.cases,a.output,a.repeats)
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))


if __name__=='__main__':main()
