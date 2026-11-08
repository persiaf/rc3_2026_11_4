"""Independent synthetic probes only. No production data provider or model calls."""
import errno
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch
from rc3_s3_evaluator.common import ROOT,read,write_new,digest,label,now,require
from rc3_s3_recovery_20261108 import authority,runner

OUT=Path(__file__).resolve().parent
HOME=ROOT/'rc3_s3_recovery_20261108'


class OpeningFailureJournal:
    def __init__(self,*args,**kwargs):self.events=[];self.results={}
    def __enter__(self):return self
    def __exit__(self,*exc):pass
    def clean(self,*args,**kwargs):pass
    def append(self,e):
        if e['event']=='session_open':raise PermissionError(errno.EACCES,'reviewer synthetic session-open anchor denial')


def main():
    source={label(p):digest(p) for p in HOME.glob('*.py')}
    registration=OUT/'REVIEW_PROBE_REGISTRATION.json'
    write_new(registration,dict(status='REGISTERED_BEFORE_PROBE',utc=now(),source_sha256=source,
        script_sha256=digest(Path(__file__)),wall_s_max=10,checks_max=8,
        permitted='Temporary synthetic metadata only; fake Journal/source/provider; no production formal entry',
        target_generation_max=0,model_inference_max=0,navigation_max=0,native_max=0,H_max=0,optimizer_max=0))
    tick=time.perf_counter();checks=[]
    with tempfile.TemporaryDirectory(prefix='s3_reviewer_') as tmp:
        root=Path(tmp)/'experiment';cont=root/'continuations'/'persistence_r1'
        fake_legacy=dict(synthetic=True)
        fake_binding=dict(experiment_id='synthetic',source='fixture')
        with patch.object(authority,'RUN',root),patch.object(authority,'CONT',cont):
            authority.initialize(fake_binding,fake_legacy)
            marker=root/'PERSISTENCE_RECOVERY_ACTIVATION.json'
            original=(digest(marker),digest(cont/'AUTHORITY.json'),digest(cont/'BASELINE_IMPORT.json'))
            authority.initialize(fake_binding,fake_legacy)
            require(original==(digest(marker),digest(cont/'AUTHORITY.json'),digest(cont/'BASELINE_IMPORT.json')),'Repeated import changed immutable files')
            checks.append('idempotent_activation_preserves_existing_bytes')
            write_new(cont/'synthetic_resource.json',dict(debits=123,wall_s=456))
            renamed=cont.parent/'renamed_segment'
            require(cont.resolve().is_relative_to(Path(tmp).resolve()) and renamed.resolve().is_relative_to(Path(tmp).resolve()),'Unsafe synthetic rename')
            cont.rename(renamed)
            rejected=False
            try:authority.initialize(fake_binding,fake_legacy)
            except ValueError:rejected=True
            require(rejected and not cont.exists() and read(renamed/'synthetic_resource.json')['debits']==123,'Renamed continuation reset budget')
            checks.append('renamed_continuation_cannot_reset_consumption')
    with tempfile.TemporaryDirectory(prefix='s3_reviewer_') as tmp:
        cont=Path(tmp)/'continuation'
        baseline=dict(debits=163,success=162,infra_failure=1,terminal_ids=[],wall_carry_s=592,old_exact_wall_s=None)
        with patch.object(runner,'CONT',cont),patch.object(runner,'Inputs',return_value=SimpleNamespace(cases=[])), \
             patch.object(runner,'grid',return_value=[]),patch.object(runner,'baseline_for',return_value=baseline), \
             patch.object(runner,'Journal',OpeningFailureJournal), \
             patch.object(runner.subprocess,'Popen',side_effect=AssertionError('A scientific worker must never be launched')):
            rejected=False
            try:runner.run_seed(11622026,time.perf_counter(),None,None,dict(source='synthetic'))
            except ValueError:rejected=True
            finals=list(cont.rglob('FINAL.json'))
            require(rejected and len(finals)==1,'Early parent failure lost the per-seed final receipt')
            final=read(finals[0]);require(final['status']=='FAILED' and final['elapsed_s']>0,'Failure clock reset or falsely complete')
            checks.append('session_open_failure_has_independent_per_seed_clock')
            require((finals[0].parent/'FINAL_SEAL.json').exists(),'Missing failure clock seal')
            checks.append('failure_clock_seal_retained')
    with tempfile.TemporaryDirectory(prefix='s3_reviewer_') as tmp:
        p=Path(tmp)/'seed';token='fixture';start=time.perf_counter()
        final=runner.finish_session(p,token,start,'CLOSED_BOUNDARY',None,lambda elapsed:time.sleep(.03))
        require(final['elapsed_s']>=.03 and final['status']=='CLOSED_BOUNDARY','Heavy closure work outside final clock')
        checks.append('closure_validation_time_precedes_final_clock_seal')
    elapsed=time.perf_counter()-tick
    require(elapsed<10,'Reviewer probe wall cap')
    write_new(OUT/'REVIEW_PROBE_RESULT.json',dict(status='PASS',checks=checks,elapsed_s=elapsed,
        registration_sha256=digest(registration),source_sha256=source,formal_requests=0,
        target_generations=0,model_inferences=0,navigation=0,native=0,H=0,optimizer=0,old_evidence_writes=0))
    print(checks)


if __name__=='__main__':main()
