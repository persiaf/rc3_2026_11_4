"""Stable rule/legacy entries and a cached frozen structured research policy."""
import argparse
import json
from pathlib import Path
from rc3_v4.plan import Planner as FrozenPlanner, DEFAULT_CHECKPOINT
from rc3_baseline.baseline import write_json,write_csv,schedule_rows,SCHEDULE_COLUMNS,scale_argument
from rc3_fast.scenes import make_complex_scene
from rc3_improve.evaluate import safe_json


class Planner(FrozenPlanner):
    def __init__(self,checkpoint=None,method='fixed3',horizon=8,cache_size=2,time_limit_s=None):
        if method=='learned':
            super().__init__(method='fixed3',horizon=horizon,cache_size=cache_size,time_limit_s=time_limit_s)
            from .policy import load_policy
            self.policy=load_policy(checkpoint or DEFAULT_CHECKPOINT)
            self.method='learned'
        else:
            super().__init__(checkpoint,method,horizon,cache_size,time_limit_s)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['fixed3','detour','legacy','learned','random'],default='fixed3')
    p.add_argument('--checkpoint',type=Path);p.add_argument('--scene',type=Path)
    p.add_argument('--scale',type=scale_argument,default=(10,10))
    p.add_argument('--mode',choices=['regular','mixed','blocked'],default='blocked')
    p.add_argument('--seed',type=int,default=771903);p.add_argument('--budget',type=int,default=8)
    p.add_argument('--time-limit-ms',type=float);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    scene=json.loads(a.scene.read_text(encoding='utf-8-sig')) if a.scene else make_complex_scene(*a.scale,a.seed,a.mode)
    if str(scene.get('schema','')).startswith('rc3-external-') or ('planning' in scene and 'audit' in scene):
        p.error('Use python -m rc3_v5.external for external service certificates; proxy mode must not discard F/A.')
    scene=scene.get('scene',scene)
    planner=Planner(a.checkpoint,a.method,a.budget,time_limit_s=None if a.time_limit_ms is None else a.time_limit_ms/1000)
    result=planner.plan(scene);a.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(a.output,safe_json(result))
    if result['status']=='complete':
        write_csv(a.output.with_suffix('.csv'),list(schedule_rows(result)),SCHEDULE_COLUMNS)
        print(json.dumps(dict(status='complete',method=a.method,objective=result['diagnostics']['objective_m'],
                             timing=result['timing'],dp_calls=result['diagnostics']['dp_calls'])),flush=True)
    else:
        print(json.dumps(dict(status='failed',code=result['code'],detail=result['detail'])),flush=True)
        raise SystemExit(2)


if __name__=='__main__':main()
