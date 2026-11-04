"""Reproduce selected CUDA validation costs with the deployment Python/NumPy."""
import json
import pickle
import sys
from pathlib import Path
import numpy as np
from rc3_baseline.baseline import write_json
from rc3_fast.solver import prepare
from .policy import load_policy
from .runtime import run


def main():
    with Path('rc3_v4/data/data.pkl').open('rb') as f:data=pickle.load(f)
    training=json.loads(Path('rc3_v4/training/summary.json').read_text(encoding='utf-8'))
    rows=[]
    problems=[prepare(s) for s in data['scenes']['validation']]
    for variant in ('control','structured'):
        policy=load_policy(Path('rc3_v4/training')/(variant+'.npz'));gains=[]
        for p in problems:
            plan,info=run(p,'learned',policy,8)
            if not plan['validation']['valid_model_plan'] or info['failures']:raise AssertionError('bad deployment validation')
            gains.append(100*(info['reference_objective_m']-info['objective_m'])/max(info['reference_objective_m'],1e-8))
        expected=training['selected'][variant]['validation']['gains_pct']
        error=float(np.max(abs(np.asarray(gains)-expected)))
        rows.append(dict(variant=variant,gains_pct=gains,max_cost_gain_error=error,passed=error<1e-7))
    out=dict(all_passed=all(r['passed'] for r in rows) and 'torch' not in sys.modules,torch_imported='torch' in sys.modules,
        python=sys.version,numpy=np.__version__,results=rows)
    write_json(Path('rc3_v4/training/portable_validation.json'),out);print(out,flush=True)
    if not out['all_passed']:raise SystemExit(2)


if __name__=='__main__':main()
