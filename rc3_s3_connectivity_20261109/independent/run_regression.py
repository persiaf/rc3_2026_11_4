"""One source-pinned, bounded independent zero-science regression controller."""
import time
START=time.perf_counter()
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HOME=ROOT/'rc3_s3_connectivity_20261109'
sys.path.insert(0,str(ROOT))
from rc3_s3_evaluator.common import read,write_new,digest,label,require,now


def main():
    require(not (HOME/'independent/REGISTRATION.json').exists(),'One independent run already registered; no silent rerun')
    sources={label(p):digest(p) for p in HOME.glob('*.py')}
    sources.update({label(p):digest(p) for p in (HOME/'independent').glob('*.py')})
    inputs={}
    for name in ('CONNECTIVITY_WITNESS.json','FROZEN_EVIDENCE_INDEX.json','diagnosis/navigation.npz','diagnosis/navigation.json','diagnosis/FINAL.json'):
        inputs[label(HOME/name)]=digest(HOME/name)
    for case in range(3):
        for suffix in ('scene.json','arrays.npz','receipt.json'):
            p=ROOT/'rc3_global_local/parents'/f'{case:02}_{suffix}';inputs[label(p)]=digest(p)
    registration=dict(schema='rc3-s3-scope-independent-zero-science-regression-v1',utc=now(),source_sha256=sources,
        input_sha256=inputs,attempts_max=1,wall_cap_s=120,tests=8,
        navigation=0,scene_generation=0,FA_generation=0,model=0,native=0,H=0,teacher=0,optimizer=0,
        formal_S3_requests=0,original60_semantic_reads=0,
        scope='Saved S3_042 diagnosis arrays,3 saved S0 parents and manually invented synthetic fixtures; existing full-parent-passing input semantics and source/budget ledger')
    write_new(HOME/'independent/REGISTRATION.json',registration)
    write_new(HOME/'independent/START.json',dict(utc=now(),registration_sha256=digest(HOME/'independent/REGISTRATION.json'),attempts_charged=1))
    error=None;code=None
    try:
        with (HOME/'independent/regression.log').open('xb') as stream:
            proc=subprocess.Popen([sys.executable,'-B','-X','utf8',str(HOME/'independent/test_scope.py')],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            try:code=proc.wait(timeout=max(.001,120-(time.perf_counter()-START)))
            except subprocess.TimeoutExpired:proc.kill();proc.wait();raise TimeoutError('Independent120s bound exhausted')
        require(code==0,'Independent regression child failed')
        child=read(HOME/'independent/CHILD_RESULT.json')
        require(child['status']=='PASS' and child['tests_run']==8 and all(v==0 for v in child['observed_scientific_function_entries'].values()),'Nonzero scientific call or failing test')
        for name,sha in sources.items():require(digest(ROOT/name)==sha,'Source changed during independent regression')
        for name,sha in inputs.items():require(digest(ROOT/name)==sha,'Input changed during independent regression')
    except BaseException as exc:error=repr(exc)
    elapsed=time.perf_counter()-START
    result=dict(status='PASS' if not error and elapsed<=120 else 'FAIL',error=error,returncode=code,elapsed_s=elapsed,
        registration_sha256=digest(HOME/'independent/REGISTRATION.json'),source_snapshot_consistent=not error,
        scientific_calls=read(HOME/'independent/CHILD_RESULT.json')['observed_scientific_function_entries'] if (HOME/'independent/CHILD_RESULT.json').exists() else None,
        formal_S3_requests=0,new_navigation_calls=0)
    write_new(HOME/'independent/REGRESSION_RESULT.json',result)
    print(result)
    require(result['status']=='PASS','Independent regression did not pass; evidence retained')


if __name__=='__main__':main()
