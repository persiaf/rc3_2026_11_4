"""Validation-only CPU feature/inference gate before training."""
import hashlib
import pickle
import time
from pathlib import Path
import numpy as np
from rc3_baseline.baseline import write_json
from rc3_fast.solver import prepare
from rc3_v3.environment import Env
from .features import observation,collate
from .model import make_policy,save_policy,export_numpy,load_policy


def main():
    import argparse
    import torch
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('rc3_v4/profile'))
    args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(17)
    folder=args.output;folder.mkdir(exist_ok=False)
    with Path('rc3_v4/data/data.pkl').open('rb') as f:data=pickle.load(f)
    flat=np.concatenate([r['features'][r['mask']] for r in data['states']])
    model=make_policy(flat.mean(0),flat.std(0).clip(.05),'structured')
    save_policy(folder/'untrained.pt',model,diagnostic_only=True)
    export_numpy(folder/'untrained.pt',folder/'untrained.npz');policy=load_policy(folder/'untrained.npz')
    records=[]
    for i,scene in enumerate(data['scenes']['validation']):
        problem=prepare(scene);env=Env(problem,8,initial_route=data['initial_routes']['validation'][i])
        for repeat in range(3):
            tick=time.perf_counter();b=collate([observation(env)]);feature=time.perf_counter()-tick
            tick=time.perf_counter()
            with policy.inference_session():policy.outputs(b)
            infer=time.perf_counter()-tick
            records.append(dict(scene=i,repeat=repeat,features_s=feature,inference_s=infer,total_s=feature+infer,
                                batch_bytes=sum(v.nbytes for v in b.values())))
    med=float(np.median([r['total_s'] for r in records]));passed=med<.05
    result=dict(passed=passed,threshold_s=.05,median_step_s=med,max_batch_bytes=max(r['batch_bytes'] for r in records),
        scope='18 validation initial states, 3 repeats; untrained CPU NumPy; includes common features and BLAS session.',records=records,
        feature_sha256=hashlib.sha256(Path('rc3_v4/features.py').read_bytes()).hexdigest(),
        model_sha256=hashlib.sha256(Path('rc3_v4/model.py').read_bytes()).hexdigest())
    write_json(folder/'result.json',result);print({k:v for k,v in result.items() if k!='records'},flush=True)
    if not passed:raise SystemExit(2)


if __name__=='__main__':main()
