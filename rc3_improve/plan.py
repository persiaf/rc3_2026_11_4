"""CPU deployment of a frozen repair policy with validated-incumbent fallback."""
import argparse,json,time
from collections import OrderedDict
from pathlib import Path
import numpy as np
from rc3_baseline.baseline import write_json,write_csv,schedule_rows,SCHEDULE_COLUMNS,scale_argument
from rc3_fast.planner import map_key
from rc3_fast.scenes import make_complex_scene
from rc3_fast.solver import prepare
from .evaluate import run_method,safe_json


class RepairPlanner:
    def __init__(self,checkpoint=None,method='ppo',horizon=8,cache_size=2):
        if method not in ('ppo','bc','detour','random','fixed1','fixed3') or horizon<1 or cache_size<0:
            raise ValueError('invalid method, horizon or cache size')
        self.policy=None
        if method in ('ppo','bc'):
            if checkpoint is None:raise ValueError('checkpoint required for learned policy')
            from .learning import load_policy
            self.policy=load_policy(checkpoint,'cpu')
        self.method,self.horizon,self.cache_size=method,horizon,cache_size
        self.cache=OrderedDict()

    def plan(self,scene):
        started=time.perf_counter();code='unsupported_input'
        if not isinstance(scene,dict):
            return dict(status='failed',code=code,detail='scene must be a mapping',plans={},task_ids=[],pending_task_ids=[])
        ids=[t.get('id') for t in scene.get('tasks',[]) if isinstance(t,dict)]
        try:
            for kind in ('tasks','tables','candidates'):
                if ([t['id'] for t in scene[kind]]!=list(range(len(scene[kind]))) or
                    any(type(t['id']) is not int for t in scene[kind])):
                    raise ValueError(f'{kind} IDs must be contiguous from zero')
            for t in scene['tasks']:
                if not isinstance(t['table_id'],int) or not 0<=t['table_id']<len(scene['tables']):
                    raise ValueError('invalid task table id')
                if not np.isfinite([t[k] for k in ('x','y','z')]).all():raise ValueError('nonfinite task')
            for c in scene['candidates']:
                if not np.isfinite([c[k] for k in ('x','y','theta','h')]).all():raise ValueError('nonfinite candidate')
                if abs(c['h']-scene['robot']['fixed_lift_height'])>1e-9:raise ValueError('variable lift not supported')
            key=map_key(scene);hit=key in self.cache
            problem=prepare(scene,nav=self.cache.get(key))
            if self.cache_size:
                self.cache[key]=problem.nav;self.cache.move_to_end(key)
                while len(self.cache)>self.cache_size:self.cache.popitem(last=False)
            cover=problem.F.any(2)
            missing=np.flatnonzero(~cover.any(1)).tolist()
            unreachable=np.flatnonzero(cover.any(1)&~(cover&np.isfinite(problem.nav['costs'][0,1:])).any(1)).tolist()
            if missing:code='no_service_candidate';raise ValueError(f'tasks: {missing}')
            if unreachable:code='disconnected';raise ValueError(f'tasks: {unreachable}')
            code='solver_failure'
            plan,info=run_method(problem,self.method,self.policy,self.horizon)
            return dict(status='complete',scene=scene,feasibility=problem.sparse,plans={self.method:plan},
                        diagnostics=info,timing=dict(preprocessing_s=problem.preprocessing_s,planning_s=info['planning_s'],
                                                     total_s=time.perf_counter()-started,cache_hit=hit),
                        guarantee='independently validated discrete proxy plan; objective <= validated Fixed1 reference')
        except Exception as error:
            return safe_json(dict(status='failed',code=code,detail=str(error),scene=scene,plans={},
                                  task_ids=ids,pending_task_ids=ids,total_s=time.perf_counter()-started,
                                  physical_infeasibility_claim=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=Path('rc3_improve/deployment_v2/policy.pt'))
    p.add_argument('--method',choices=['ppo','bc','detour','random','fixed1','fixed3'],default='ppo')
    p.add_argument('--scale',type=scale_argument,default=(10,10));p.add_argument('--mode',choices=['regular','mixed','blocked'],default='blocked')
    p.add_argument('--seed',type=int,default=6241030);p.add_argument('--scene',type=Path);p.add_argument('--budget',type=int,default=8)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    scene=json.loads(a.scene.read_text(encoding='utf-8')) if a.scene else make_complex_scene(*a.scale,a.seed,a.mode)
    scene=scene.get('scene',scene)
    planner=RepairPlanner(a.checkpoint,a.method,a.budget)
    result=planner.plan(scene);a.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(a.output,safe_json(result))
    if result['status']=='complete':
        write_csv(a.output.with_suffix('.csv'),list(schedule_rows(result)),SCHEDULE_COLUMNS)
        print(json.dumps(dict(status=result['status'],timing=result['timing'],objective=result['diagnostics']['objective_m'],
                              reference=result['diagnostics']['reference_objective_m'],fallback=result['diagnostics']['fallback_used'])),flush=True)
    else:print(json.dumps({k:result[k] for k in ('status','code','detail','pending_task_ids')}),flush=True)

if __name__=='__main__':main()
