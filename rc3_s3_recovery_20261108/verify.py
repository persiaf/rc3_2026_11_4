"""Cumulative zero-science engineering budget; immutable run snapshots/results."""
import io
import json
import subprocess
import sys
import time
from pathlib import Path
from rc3_s3_evaluator.common import ROOT,read,write_new,digest,label,now,require
HOME=Path(__file__).resolve().parent


def sources():return {label(p):digest(p) for p in sorted(HOME.glob('*.py'))}


def main():
    protocol=HOME/'ENGINEERING_PROTOCOL.json'
    if not protocol.exists():write_new(protocol,dict(status='REGISTERED_BEFORE_FAULT_REGRESSION',utc=now(),
        cumulative_wall_s_max=900,run_max=8,test_methods_per_run_max=40,
        permitted='Synthetic temporary filesystem/events/statistics and read-only original180 evidence; no new target generation/model inference/navigation/native/H/optimizer',
        scientific_generation_inference_navigation_native_H_optimizer_max=0))
    previous=sorted((HOME/'verification').glob('RESULT_*.json'))
    used=sum(read(p)['elapsed_s'] for p in previous);require(len(previous)<8 and used<900,'Engineering budget exhausted')
    n=len(previous)+1;registration=HOME/'verification'/f'REGISTRATION_{n:02}.json'
    write_new(registration,dict(protocol_sha256=digest(protocol),source_sha256=sources(),
        remaining_wall_s=900-used,prior_wall_s=used,real_scientific_calls_max=0,utc=now()))
    tick=time.perf_counter()
    p=subprocess.run([sys.executable,'-B','-X','utf8','-m','unittest','rc3_s3_recovery_20261108.test_recovery','-v'],
        cwd=ROOT,capture_output=True,timeout=900-used)
    elapsed=time.perf_counter()-tick
    completed_sources=sources();consistent=completed_sources==read(registration)['source_sha256']
    stdout=p.stdout.decode('utf8');lines=[line for line in stdout.splitlines() if line.startswith('OBSERVED_ZERO_SCIENCE_COUNTERS=')]
    counters=json.loads(lines[-1].split('=',1)[1]) if lines else None
    passed=p.returncode==0 and consistent and counters is not None and not any(counters.values())
    write_new(HOME/'verification'/f'RESULT_{n:02}.json',dict(status='PASS' if passed else 'FAIL',
        returncode=p.returncode,elapsed_s=elapsed,cumulative_wall_s=used+elapsed,
        registration_sha256=digest(registration),source_sha256=completed_sources,source_snapshot_consistent=consistent,
        observed_scientific_function_entries=counters,stdout=stdout,stderr=p.stderr.decode('utf8'),
        new_target_generations=0,new_model_inferences=0,new_navigation=0,new_native=0,new_H=0,new_optimizer=0))
    print(p.stderr.decode('utf8'));print('observed_zero_science',counters);raise SystemExit(0 if passed else 1)


if __name__=='__main__':main()
