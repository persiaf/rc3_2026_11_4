import time
START_M,START_W=time.monotonic(),time.time()
import io
import random
import platform
import subprocess
import traceback
from pathlib import Path
from .common import HOME,ROOT,read,put,sha,Ledger,admit,BINDING

def main(stage,authorization=None):
    admit(stage,authorization)
    ledger=Ledger(stage,START_M,START_W)
    try:
        from rc3_r04_m1_real_20261111.provenance_use import ScientificTracedProvenance,verify_real_environment
        b=read(BINDING);trace=ScientificTracedProvenance(ROOT,ledger.path,b,ledger.check,'score5-'+stage+'-r02');trace.install_import_guard()
        assert sha(HOME/'PROTOCOL.json')==b['protocol_sha256'] and sha(HOME/'SCHEDULE.json')==b['schedule_sha256']
        with trace.stage('real_Torch_and_exact_saved_input_sources'):
            import torch
            import numpy as np
            from . import actor,trainer,runtime,engineering,science
            from rc3_conservative_refinement import saved_inputs as si,runtime as old
            actual=verify_real_environment(trace,torch,dict(actor=actor,trainer=trainer,runtime=runtime))
            torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
            torch.manual_seed(11622026);np.random.seed(11622026);random.seed(11622026)
            env=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device='cpu',threads=1,deterministic_algorithms=True)
            assert env==b['environment']
            si.digest=old.file_sha256=trace.hash_file;si.read=trace.read_json
            from rc3_global_local_trainer import navigation_compat,approval
            from rc3_global_local_impl_review import runtime_guards
            navigation_compat.read=approval.read=trace.read_json
            navigation_compat.digest=approval.digest=runtime_guards.file_hash=trace.hash_file
            np_raw=np.load;torch_raw=torch.load
            def np_load(p,*a,**kw):return np_raw(io.BytesIO(trace.read_bytes(p)) if isinstance(p,(str,__import__('os').PathLike)) else p,*a,**kw)
            def torch_load(p,*a,**kw):return torch_raw(io.BytesIO(trace.read_bytes(p)) if isinstance(p,(str,__import__('os').PathLike)) and not Path(p).resolve().is_relative_to(HOME) else p,*a,**kw)
            np.load=np_load;torch.load=torch_load
            inputs=si.SavedInputs()
            if inputs.evaluation_cases!=read(HOME/'SCHEDULE.json')['evaluation']:raise ValueError('Saved evaluation identity differs')
            put(ledger.path/'ACTUAL_ENVIRONMENT.json',dict(environment=env,real_Torch_classification=actual,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
        identity=dict(protocol_sha256=sha(HOME/'PROTOCOL.json'),schedule_sha256=sha(HOME/'SCHEDULE.json'),source_binding_sha256=sha(BINDING))
        with trace.stage(stage+'_real_execution'):
            (engineering.run if stage=='engineering' else science.run)(ledger,trace,inputs,identity)
        trace.check_loaded_modules();put(ledger.path/'ACTUAL_DEPENDENCY_VERIFICATION.json',trace.summary())
        ledger.status='COMPLETE';ledger.event('stage_complete',counts=ledger.counts,input_counts=inputs.counts)
        put(ledger.path/'RESOURCE_ACCOUNTING.json',dict(ledger.snapshot(),input_counts=inputs.counts));print(ledger.snapshot(),flush=True)
        if stage=='engineering':put(HOME/'ENGINEERING_ACCEPTANCE.json',dict(status=read(ledger.path/'CHECKS.json')['status'],path=str(ledger.path/'CHECKS.json'),source_binding_sha256=sha(BINDING),models_forbidden_as_scientific_initialization=True))
    except BaseException as e:
        ledger.status='FAILED_PRESERVED';put(ledger.path/'FAILURE.json',dict(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),resource=ledger.snapshot()))
        put(ledger.path/'RESOURCE_ACCOUNTING.json',ledger.snapshot());raise
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['engineering','science']);p.add_argument('--authorization');a=p.parse_args();main(a.stage,a.authorization)
