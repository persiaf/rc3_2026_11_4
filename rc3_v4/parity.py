"""Validation-only parity of real trajectories, exported plans and three runtimes."""
import argparse
import hashlib
import pickle
from pathlib import Path
import numpy as np
from rc3_baseline.baseline import write_json
from rc3_fast.solver import prepare,export_plan
from rc3_v3.environment import Env
from rc3_v3.model import select_action
from .features import collate,observation
from .model import load_policy


def main():
    import torch
    p=argparse.ArgumentParser();p.add_argument('--training',type=Path,default=Path('rc3_v4/training'));a=p.parse_args()
    torch.set_num_threads(1)
    with Path('rc3_v4/data/data.pkl').open('rb') as f:data=pickle.load(f)
    problems=[prepare(s) for s in data['scenes']['validation']]
    results=[]
    for variant in ('control','structured'):
        cpu=load_policy(a.training/(variant+'.pt'),'cpu');gpu=load_policy(a.training/(variant+'.pt'),'cuda')
        numpy=load_policy(a.training/(variant+'.npz'))
        states=0;maximum=0.;mismatches=[];plans=0
        with numpy.inference_session():
            for i,problem in enumerate(problems):
                env=Env(problem,8,initial_route=data['initial_routes']['validation'][i])
                while True:
                    base,mask=env.observe()
                    if env.done or not mask.any():break
                    batch=collate([observation(env,base,mask)])
                    outputs=[cpu.outputs(batch),gpu.outputs(batch),numpy.outputs(batch)]
                    for other in outputs[1:]:
                        for x,y in zip(outputs[0],other):
                            finite=np.isfinite(x)&np.isfinite(y)
                            maximum=max(maximum,float(abs(x[finite]-y[finite]).max()))
                    actions=[select_action(v[0][0],mask) for v in outputs]
                    if len(set(actions))!=1:mismatches.append(dict(scene=i,step=env.steps,actions=actions))
                    states+=1
                    if actions[-1] is None:break
                    env.step(actions[-1])
                plan=export_plan(problem,env.route,'v4_parity_validation')
                assert plan['validation']['valid_model_plan'];plans+=1
        results.append(dict(variant=variant,states=states,max_output_error=maximum,action_mismatches=mismatches,
                            independent_final_plans=plans,passed=not mismatches and maximum<1e-4))
    out=dict(all_passed=all(r['passed'] for r in results),results=results,
        sha256={str(a.training/(v+ext)):hashlib.sha256((a.training/(v+ext)).read_bytes()).hexdigest()
                for v in ('control','structured') for ext in ('.pt','.npz')})
    write_json(a.training/'parity.json',out);print(out,flush=True)
    if not out['all_passed']:raise SystemExit(2)


if __name__=='__main__':main()
