"""Run a saved G2 policy on a static proxy scene and export a certified plan."""
import argparse
import gzip
import json
from pathlib import Path
import time

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--scene',type=Path)
    p.add_argument('--nx',type=int,default=10);p.add_argument('--ny',type=int,default=10)
    p.add_argument('--seed',type=int,default=20261103)
    p.add_argument('--mode',choices=['regular','mixed','blocked'],default='regular')
    p.add_argument('--method',choices=['learned','detour_memory','ils','random','stop'],default='learned')
    p.add_argument('--time-limit-ms',type=float);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.method=='learned' and a.checkpoint is None:p.error('--checkpoint is required for learned')
    startup=time.perf_counter()
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1);threadpool_limits(1)
    from rc3_fast.scenes import make_complex_scene
    from rc3_rl_research.types import SearchConfig
    from .checkpoints import load_model,sha256
    from .evaluate_g2 import _run_shared,_write
    model=None;checkpoint=None
    if a.checkpoint:
        model,checkpoint=load_model(a.checkpoint,'cpu');model.eval()
    load_s=time.perf_counter()-startup
    source=json.loads(a.scene.read_text(encoding='utf-8-sig')) if a.scene else make_complex_scene(a.nx,a.ny,a.seed,a.mode)
    if ('planning' in source and 'audit' in source) or str(source.get('schema','')).startswith('rc3-external-'):
        p.error('This CLI is for proxy scenes; external F/A must use the explicit ResearchAdapterG2.from_external API and must not be discarded')
    scene=source.get('scene',source)
    settings={}
    selection=Path(__file__).parent/'outputs/rule_development/selected_ils.json'
    if a.method=='ils' and selection.exists():settings=json.loads(selection.read_text())['selected']
    row,evidence=_run_shared(scene,a.method,model,SearchConfig(),
                            None if a.time_limit_ms is None else a.time_limit_ms/1000,settings,a.seed)
    row['runtime_and_model_load_s']=load_s
    identity=None if a.checkpoint is None else dict(checkpoint=str(a.checkpoint),sha256=sha256(a.checkpoint),
        architecture=checkpoint['architecture'],training_actions=checkpoint.get('actual_actions'))
    output=dict(schema='rc3-g2-service-plan-v1',status='complete',scene=scene,plans={a.method:evidence['plan']},
        diagnostics=row,model=identity,scope='2D static proxy reachability; no physical IK')
    _write(a.output,output)
    _write(a.output.with_suffix('.trace.json.gz'),evidence,compressed=True)
    print(json.dumps(dict(output=str(a.output),J=row['J'],return_value=row['total_reward'],
                         complete_s=row['complete_s'],runtime_and_model_load_s=load_s),ensure_ascii=False))

if __name__=='__main__':main()
