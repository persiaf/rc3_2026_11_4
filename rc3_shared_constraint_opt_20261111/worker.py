import time
START_M,START_W=time.monotonic(),time.time()
import random
import platform
import subprocess
import traceback
from .common import HOME,ROOT,Ledger,read,put,sha
def main():
    ledger=Ledger(START_M,START_W)
    try:
        from rc3_r04_m1_real_20261111.provenance_use import ScientificTracedProvenance,verify_real_environment
        binding=read(HOME/'SOURCE_BINDING.json')
        trace=ScientificTracedProvenance(ROOT,ledger.path,binding,ledger.check,'shared-constraint-opt-20261111-01');trace.install_import_guard()
        assert read(HOME/'DIAGNOSTIC_AUTHORIZATION.json')['source_binding_sha256']==sha(HOME/'SOURCE_BINDING.json')
        assert sha(HOME/'PROTOCOL.json')==binding['protocol_sha256']
        with trace.stage('real_Torch_and_exact_reused_runtime_sources'):
            import numpy as np
            import torch
            from . import solver,experiment
            actual=verify_real_environment(trace,torch,dict(solver=solver,experiment=experiment))
            torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
            torch.manual_seed(11622026);np.random.seed(11622026);random.seed(11622026)
            env=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device='cpu',threads=1,deterministic_algorithms=True)
            assert env==binding['environment']
            put(ledger.path/'ACTUAL_ENVIRONMENT.json',dict(environment=env,real_Torch_classification=actual,
                code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
        experiment.run(ledger,trace)
        trace.check_loaded_modules();put(ledger.path/'ACTUAL_DEPENDENCY_VERIFICATION.json',trace.summary())
        ledger.status='COMPLETE';ledger.event('stage_complete',counts=ledger.counts)
        put(ledger.path/'RESOURCE_ACCOUNTING.json',ledger.snapshot());print(ledger.snapshot(),flush=True)
    except BaseException as e:
        ledger.status='FAILED_PRESERVED'
        put(ledger.path/'FAILURE.json',dict(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),resource=ledger.snapshot()))
        put(ledger.path/'RESOURCE_ACCOUNTING.json',ledger.snapshot());raise
if __name__=='__main__':main()
