"""Research inference entry: complete constructive route, shared postprocess, final check."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import time
import torch
from threadpoolctl import threadpool_limits
from rc3_fast.scenes import make_complex_scene
from rc3_fast.solver import route_objective
from .problem import ROOT,Cache,write,digest
from .evaluate import load_policy,greedy_routes
from .baselines import postprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=Path);parser.add_argument('--scale',default='10x10')
    parser.add_argument('--mode',choices=['regular','mixed','blocked'],default='regular')
    parser.add_argument('--seed',type=int,default=7300000000);parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    path=args.checkpoint
    if path is None:
        summary=json.loads((ROOT/'outputs/training_summary.json').read_text())
        path=ROOT/summary['selected_checkpoint']
    model,load_s=load_policy(path)
    if args.scene:scene=json.loads(args.scene.read_text(encoding='utf-8-sig'))
    else:
        nx,ny=map(int,args.scale.lower().split('x'));scene=make_complex_scene(nx,ny,args.seed,args.mode)
    started=time.perf_counter();item=Cache(1).prepare(scene);prepare_s=time.perf_counter()-started
    tick=time.perf_counter();raw_route=greedy_routes(model,[item])[0][0];decode_s=time.perf_counter()-tick
    raw_J=route_objective(item.base,raw_route)
    tick=time.perf_counter();route=postprocess(item,raw_route);post_s=time.perf_counter()-tick
    tick=time.perf_counter();plan=item.certify(route,'constructive_research');check_s=time.perf_counter()-tick
    total=time.perf_counter()-started
    assert plan['objective_length_plus_stops_m']<=raw_J+1e-7
    result=dict(status='complete',scene=item.base.scene,plans={'constructive':plan},feasibility=item.base.sparse,
        diagnostics=dict(raw_route=raw_route,raw_J=raw_J,J=plan['objective_length_plus_stops_m'],
                         checkpoint=str(path),checkpoint_sha256=digest(path),fallback_used=False),
        timing=dict(model_load_s=load_s,preparation_s=prepare_s,decode_s=decode_s,postprocess_s=post_s,
                    export_validation_s=check_s,total_s=total),
        scope='Research policy, proxy/table-model validation only. No Fixed3 online fallback or cost bound. Process startup, input generation/read, model load and serialization excluded from per-request total and disclosed separately.')
    write(args.output,result)
    with args.output.with_suffix('.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['sequence','task_id','candidate_id','arm','x','y','theta','h']);writer.writeheader()
        for stop in plan['service_stops']:
            for task in stop['tasks']:writer.writerow(dict(sequence=stop['sequence'],candidate_id=stop['candidate_id'],**task,**stop['pose']))
    print(json.dumps(dict(status='complete',J=plan['objective_length_plus_stops_m'],L=plan['route_length_m'],K=len(plan['service_stops']),timing=result['timing'])),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):main()
