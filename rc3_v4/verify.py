"""Regression suite plus CPU-only cold/warm planner and external-certificate smoke."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import unittest
from rc3_baseline.baseline import write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('rc3_v4/verification'))
    parser.add_argument('--cpu-python',default=sys.executable);args=parser.parse_args()
    args.output.mkdir(exist_ok=True);started=time.perf_counter();errors=[]
    stream=io.StringIO();suite=unittest.TestLoader().discover('rc3_v4',pattern='test_*.py',top_level_dir='.')
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (args.output/'unit_tests.log').write_text(stream.getvalue(),encoding='utf-8')
    if not result.wasSuccessful():errors.append('unit test suite failed')
    script=r'''
import json,sys
from pathlib import Path
from rc3_fast.scenes import make_complex_scene
from rc3_v4.plan import Planner
from rc3_baseline.validator import validate
from rc3_v4.external_example import make_demo
from rc3_v4.external import plan_external,validate_external
scene=make_complex_scene(4,3,730920,'blocked')
records=[]
for method in ('fixed3','detour','learned','legacy'):
 p=Planner(method=method)
 cold=p.plan(scene);warm=p.plan(scene)
 assert cold['status']==warm['status']=='complete'
 assert cold['plans']==warm['plans'] and warm['timing']['cache_hit']
 plan=cold['plans'][method];v=validate(scene,plan,cold['feasibility'])
 assert v['valid_model_plan'] and v['all_tasks_served']
 assert cold['diagnostics']['cost_nonincreasing']
 records.append(dict(method=method,objective=cold['diagnostics']['objective_m'],cold=cold['timing'],warm=warm['timing']))
bad=dict(scene,tasks=[dict(t,z=9.) for t in scene['tasks']])
failure=Planner().plan(bad)
assert failure['status']=='failed' and failure['pending_task_ids']==[t['id'] for t in bad['tasks']]
b=make_demo();e=plan_external(b,'detour');assert e['status']=='complete'
assert validate_external(b,e['plans']['detour'])['valid_model_plan']
assert 'torch' not in sys.modules
print(json.dumps(dict(passed=True,torch_imported=False,records=records,invalid_tasks_rejected=True,external_valid=True)))
'''
    run=subprocess.run([args.cpu_python,'-B','-c',script],capture_output=True)
    if run.returncode:
        errors.append('CPU-only smoke failed: '+run.stderr.decode('utf-8',errors='replace'))
        smoke=dict(passed=False)
    else:
        smoke=json.loads(run.stdout.decode('utf-8'));write_json(args.output/'cpu_smoke.json',smoke)
    parity=Path('rc3_v4/training/parity.json')
    parity_ok=parity.exists() and json.loads(parity.read_text(encoding='utf-8'))['all_passed']
    if not parity_ok:errors.append('real-trajectory parity is missing or failed')
    summary=dict(all_passed=not errors,errors=errors,unit_tests=result.testsRun,
        failures=len(result.failures),test_errors=len(result.errors),skipped=len(result.skipped),
        cpu_smoke=smoke.get('passed',False),parity_passed=parity_ok,elapsed_s=time.perf_counter()-started,
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in Path('rc3_v4').glob('*.py')})
    write_json(args.output/'summary.json',summary);print(json.dumps(summary),flush=True)
    if errors:raise SystemExit(2)


if __name__=='__main__':main()
