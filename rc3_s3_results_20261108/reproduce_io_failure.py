"""Preregistered synthetic filesystem fault only; never touches formal run files."""
from pathlib import Path
import tempfile
import time
from unittest.mock import patch
from rc3_s3_evaluator.common import *
from rc3_s3_evaluator.journal import Journal

OUT=ROOT/'rc3_s3_results_20261108'


def main():
    registration=OUT/'IO_FAULT_REPRO_REGISTRATION.json'
    write_new(registration,dict(status='REGISTERED_BEFORE_SYNTHETIC_FAULT',utc=now(),
        source_binding_sha256=digest(HERE/'S3_SOURCE_BINDING.json'),journal_source_sha256=digest(HERE/'journal.py'),
        probe_source_sha256=digest(Path(__file__)),wall_s_max=10,synthetic_append_attempts_max=4,
        synthetic_failures='Injected PermissionError at os.replace; then existing pending file blocks error append',
        real_generation=0,real_inference=0,real_navigation=0,real_native=0,real_H=0,optimizer=0,
        formal_run_file_writes=0))
    tick=time.perf_counter();checks=[]
    with tempfile.TemporaryDirectory(prefix='s3_io_fault_') as temp:
        p=Path(temp)/'fixture';j=Journal(p,dict(synthetic=True),[],0)
        with patch('rc3_s3_evaluator.journal.os.replace',side_effect=PermissionError(5,'synthetic access denied')):
            try:j.append(dict(event='operation_start',operation='synthetic'))
            except PermissionError:checks.append('first_replace_denied')
        require((p/'HEAD.pending').exists() and not (p/'HEAD.json').exists(),'Expected preserved pending anchor')
        checks.append('pending_preserved')
        try:j.append(dict(event='operation_end',operation='synthetic',elapsed_s=0))
        except FileExistsError:checks.append('secondary_error_append_blocked_by_pending')
        require(len(j.path.read_bytes().splitlines())==2,'Both append-only records must survive')
        checks.append('journal_records_retained')
        try:Journal(p,dict(synthetic=True),[],0)
        except ValueError:checks.append('automatic_resume_refused')
    elapsed=time.perf_counter()-tick
    require(len(checks)==5 and elapsed<10,'Synthetic reproduction did not match bounds')
    write_new(OUT/'IO_FAULT_REPRO_RESULT.json',dict(status='PASS_FAILURE_MECHANISM_REPRODUCED',checks=checks,
        elapsed_s=elapsed,registration_sha256=digest(registration),synthetic_append_attempts=2,
        new_real_requests=0,new_target_generations=0,new_inferences=0,new_native=0,new_H=0,optimizer=0,
        formal_run_file_writes=0,
        conclusion='Confirms approved journal error cascade under injected replacement failure; does not identify external process/driver behind original Windows access denial'))
    print(checks)


if __name__=='__main__':main()
