"""Post-training, interleaved all-method evaluation; preserve C0 measurements."""
from __future__ import annotations
import json
import time
import torch
from threadpoolctl import threadpool_limits
from .problem import ROOT,Cache,registered_scene,write
from .evaluate import load_policy,run_one


def main():
    out=ROOT/'outputs';target=out/'final_evaluation.json'
    if target.exists():raise FileExistsError('Refusing implicit final-evaluation replay')
    trained=json.loads((out/'training_summary.json').read_text())
    initial,t0=load_policy(out/'checkpoints/update0000.pt')
    model,t1=load_policy(ROOT/trained['selected_checkpoint'])
    write(out/'model_load.json',dict(initial_s=t0,selected_s=t1,scope='CPU checkpoint load and policy construction; standing process imports excluded'))
    models=dict(initial=initial,model=model);methods=['fixed3','detour8','detour16','insertion','prior','initial','model'];records=[]
    for case in range(24):
        scene=registered_scene('development',case)
        for rep in range(2):
            ordered=methods[rep:]+methods[:rep]
            if case%2:ordered=list(reversed(ordered))
            for mode in (['cold','hot'] if (case+rep)%2==0 else ['hot','cold']):
                for name in ordered:
                    cache=Cache(2)
                    if mode=='hot':cache.prepare(scene)
                    save=out/'final_plans'/f'dev_{case:02}_{name}' if mode=='cold' and rep==0 else None
                    row=run_one(scene,name,cache,models.get(name),save=save)
                    row.update(case=case,rep=rep,cache_mode=mode,scene_seed=scene['seed'],scene_mode=scene['complexity'])
                    records.append(row)
        write(target,records);print(json.dumps({'final_cases':case+1,'rows':len(records)}),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):main()
