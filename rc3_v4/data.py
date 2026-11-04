"""Augment the frozen teacher states without adding oracle labels or test tuning."""
import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path
import numpy as np
from rc3_baseline.baseline import write_json
from rc3_fast.solver import prepare
from rc3_v3.data import historical_seeds
from rc3_v3.environment import adaptive_windows
from .features import local_graphs


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('rc3_v4/data'))
    p.add_argument('--source',type=Path,default=Path('rc3_v3/data_v3_20261102/data.pkl'))
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=a.resume)
    started=time.perf_counter()
    with a.source.open('rb') as f: data=pickle.load(f)  # Only the existing locally generated cache.
    if hashlib.sha256(a.source.read_bytes()).hexdigest()!='310401a8e08393bc8d595a84857c4563a69dddd994b38041654dbcc03e585999':
        raise ValueError('v3 teacher cache changed')
    if a.resume:
        if (a.output/'data.pkl').exists(): raise ValueError('refuse to overwrite completed cache')
        cases=json.loads((a.output/'test_cases.json').read_text(encoding='utf-8'))
    else:
        forbidden,audit=historical_seeds(Path('.'))
        rng=np.random.default_rng(7130081);cases=[]
        for nx,ny,count in ((10,10,6),(6,6,3),(14,14,3),(20,20,1)):
            for mode in ('regular','mixed','blocked'):
                for _ in range(count):
                    while True:
                        seed=int(rng.integers(100000,2**31-1))
                        if seed not in forbidden: break
                    forbidden.add(seed);cases.append(dict(nx=nx,ny=ny,mode=mode,seed=seed))
        write_json(a.output/'test_cases.json',cases)
        write_json(a.output/'split_audit.json',audit)
    rows=data['states']
    for sid,scene in enumerate(data['scenes']['train']):
        problem=prepare(scene)
        for row in [r for r in rows if r['scene_id']==sid]:
            route=list(map(int,row['route']))
            windows,_=adaptive_windows(problem,route)
            if len(windows)!=len(row['mask']): raise AssertionError('window order/length changed')
            row['graphs']=local_graphs(problem,route,windows)
        if (sid+1)%12==0: print(f'graph scenes {sid+1}/96 seconds={time.perf_counter()-started:.1f}',flush=True)
    data['graph_schema']='rc3-v4-local-graph-v1'
    with (a.output/'data.pkl').open('wb') as f: pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
    manifest=dict(status='complete',source=str(a.source),source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
        schema=data['graph_schema'],states=len(rows),train_scenes=len(data['scenes']['train']),
        validation_scenes=len(data['scenes']['validation']),test_scenes=len(cases),
        generation_s=time.perf_counter()-started,data_bytes=(a.output/'data.pkl').stat().st_size,
        data_sha256=hashlib.sha256((a.output/'data.pkl').read_bytes()).hexdigest(),
        test_sha256=hashlib.sha256((a.output/'test_cases.json').read_bytes()).hexdigest(),
        note='Teacher labels and 28D observations unchanged; no new test costs read; graph tensors use only F/D and current route.')
    write_json(a.output/'manifest.json',manifest);print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
