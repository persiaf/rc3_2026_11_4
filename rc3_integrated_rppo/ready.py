"""Fresh-process ready boundary: immutable sources/model files, no task encode."""
import os
import time
from .common import *


def main():
    start=float(os.environ['RC3_READY_LAUNCH_PERF']);verify_sources();models={}
    for seed in SEEDS:
        d=read(OUT/'seeds'/str(seed)/'SELECTION_DECISION.json')
        for role in ('RL','residual_SL'):
            e=d['chosen'][role]['entry'];require(digest(ROOT/e['path'])==e['actor_sha'],'Ready model file mismatch');models[e['path']]=e['actor_sha']
        p=read(OUT/'seeds'/str(seed)/'main/COMPLETE.json')['base_entry'];require(digest(ROOT/p['path'])==p['actor_sha'],'Ready SL file mismatch');models[p['path']]=p['actor_sha']
    print(json.dumps(dict(ready_s=time.perf_counter()-start,models=models,source_binding_sha256=digest(HOME/'SOURCE_BINDING.json'),
        model_forwards=0,task_embeddings=0,navigation_solves=0,complete_requests=0,scope='Cold process import, immutable runtime/model identity validation; common for every method')))


if __name__=='__main__':main()
