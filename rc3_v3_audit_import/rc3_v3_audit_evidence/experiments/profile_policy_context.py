"""One validation-state microbenchmark of repeated threadpool discovery overhead.

No source edits; a process-local patch only eliminates an inner redundant BLAS
context while an outer context already limits the entire measurement to one.
"""
import json
from contextlib import nullcontext
from pathlib import Path
import platform
import sys
import time
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]/'v3_upload'
sys.path.insert(0,str(ROOT))
from rc3_fast.scenes import make_complex_scene
from rc3_fast.solver import prepare
from rc3_v3.environment import Env
from rc3_v3.model import load_policy


def main():
    base=ROOT/'rc3_v3'
    case=json.loads((base/'data_v3_20261102/validation_cases.json').read_text())[0]
    problem=prepare(make_complex_scene(case['nx'],case['ny'],case['seed'],case['mode']))
    env=Env(problem,horizon=8);features,mask=env.observe()
    policy=load_policy(base/'deployment/policy.npz')
    rows=[];reference=None
    rng=np.random.default_rng(20261103)
    iterations=200
    with threadpool_limits(limits=1,user_api='blas'):
        reference=policy.outputs(features,mask)
        for block in range(5):
            for label in rng.permutation(['per_call_discovery','already_managed_once']):
                ctx=(patch('threadpoolctl.threadpool_limits',lambda *a,**k:nullcontext())
                     if label=='already_managed_once' else nullcontext())
                with ctx:
                    actual=policy.outputs(features,mask)
                    exact=all(np.array_equal(a,b) for a,b in zip(actual,reference))
                    tick=time.perf_counter()
                    for _ in range(iterations):policy.outputs(features,mask)
                    elapsed=time.perf_counter()-tick
                rows.append(dict(block=block,method=str(label),calls=iterations,
                                 mean_call_ms=elapsed/iterations*1000,outputs_exactly_equal=exact))
    medians={label:float(np.median([r['mean_call_ms'] for r in rows if r['method']==label]))
             for label in ('per_call_discovery','already_managed_once')}
    result=dict(validation_case=case,windows=len(mask),legal_windows=int(mask.sum()),features=features.shape[1],
                architecture=policy.metadata['architecture'],platform=platform.platform(),python=platform.python_version(),numpy=np.__version__,
                scope='One fixed real validation observation, 5 randomized-order blocks of 200 calls. Both conditions run inside one outer BLAS-thread=1 context. No source changes, no retraining, no full planner timing.',
                median_of_block_mean_call_ms=medians,
                all_outputs_exactly_equal=all(r['outputs_exactly_equal'] for r in rows),
                caveat='This isolates per-call threadpool-library discovery and context overhead on the current Linux machine; it does not reproduce the original Windows benchmark or establish end-to-end speedup.',
                runs=rows)
    (OUT/'policy_threadpool_microprofile.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='runs'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
