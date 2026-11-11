import time
START_M,START_W=time.monotonic(),time.time()
import io
import os
import random
import platform
import traceback
from pathlib import Path
from .common import HOME,ROOT,RUN,Ledger,read,put,sha,binding_identity,verify_binding

def main():
    b=verify_binding();ledger=Ledger(START_M,START_W)
    try:
        from rc3_r04_m1_real_20261111.provenance_use import ScientificTracedProvenance,verify_real_environment
        trace=ScientificTracedProvenance(ROOT,ledger.path,b,ledger.check,RUN+'-P0');trace.install_import_guard()
        with trace.stage('registered_real_Torch_and_saved_sources'):
            import torch
            import numpy as np
            from . import actor,trainer,runtime,engineering
            from rc3_conservative_refinement import saved_inputs as si,runtime as source
            actual=verify_real_environment(trace,torch,dict(actor=actor,trainer=trainer,runtime=runtime))
            torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
            torch.manual_seed(2026111101);np.random.seed(2026111101);random.seed(2026111101)
            env=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device='cpu',threads=1,deterministic_algorithms=True)
            if env!=read(HOME/'ENVIRONMENT_METADATA.json')['registered_environment']:raise ValueError('Registered hardware/software contract differs')
            si.digest=source.file_sha256=trace.hash_file;si.read=trace.read_json
            from rc3_global_local_trainer import navigation_compat,approval
            from rc3_global_local_impl_review import runtime_guards
            navigation_compat.read=approval.read=trace.read_json
            navigation_compat.digest=approval.digest=runtime_guards.file_hash=trace.hash_file
            original_np_load=np.load;original_torch_load=torch.load
            def np_load(p,*a,**kw):return original_np_load(io.BytesIO(trace.read_bytes(p)) if isinstance(p,(str,os.PathLike)) else p,*a,**kw)
            def torch_load(p,*a,**kw):return original_torch_load(io.BytesIO(trace.read_bytes(p)) if isinstance(p,(str,os.PathLike)) and not Path(p).resolve().is_relative_to(HOME) else p,*a,**kw)
            np.load=np_load;torch.load=torch_load
            inputs=si.SavedInputs()
            for t in read(ROOT/'rc3_score5_20261111/SAVED_TEACHER_METADATA.json')['teachers']:inputs.allowed[si.spec_key(t['spec'])]=t['spec']
            if inputs.evaluation_cases!=read(HOME/'PROGRAM_PROTOCOL.json')['evaluation']['development']:raise ValueError('Development identities differ')
            put(ledger.path/'ACTUAL_ENVIRONMENT.json',dict(environment=env,real_Torch_classification=actual))
        with trace.stage('necessary_ordinary_task_P0'):
            engineering.run(ledger,inputs,binding_identity())
        trace.check_loaded_modules();put(ledger.path/'ACTUAL_DEPENDENCY_VERIFICATION.json',trace.summary())
        ledger.status='COMPLETE';put(ledger.path/'RESOURCE_ACCOUNTING.json',dict(ledger.snapshot(),input_counts=inputs.counts))
        data=read(HOME/'DATA_ADMISSION.json');overall='GO_P1' if data['status']=='PASS' else 'P6_TECHNICAL_DATA_INCOMPLETE'
        status=read(HOME/'STATUS.json');status.update(phase='P6' if overall!='GO_P1' else 'P1',status=overall,P0_engineering='PASS',
            P1_admission=data['status'],scientific_training='NOT_RUN',scientific_results=None,next_action='Document and publish precise data-admission failure; no new teacher or scientific run')
        put(HOME/'STATUS.json',status,replace=True)
        ledger.event('P0_complete',engineering='PASS',data_admission=data['status'],next_phase=status['phase'])
        print(dict(status=overall,P0_engineering='PASS',data_admission=data['status'],requests=ledger.counts['requests'],Actor_attempts=ledger.counts['actor_optimizer_attempts']),flush=True)
    except BaseException as e:
        ledger.status='FAILED_PRESERVED';put(ledger.path/'FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),resource=ledger.snapshot()))
        put(ledger.path/'RESOURCE_ACCOUNTING.json',ledger.snapshot())
        status=read(HOME/'STATUS.json');status.update(status='P0_TECHNICAL_FAILURE_PRESERVED',next_action='Source-bound repair within remaining P0 budget; never reset failure consumption')
        put(HOME/'STATUS.json',status,replace=True);raise

if __name__=='__main__':main()
