import io
import platform
import random
import subprocess
from pathlib import Path
from .common import ROOT,HOME,read,put,sha

def setup(ledger):
    from rc3_r04_m1_real_20261111.provenance_use import ScientificTracedProvenance,verify_real_environment
    binding=read(HOME/'SOURCE_BINDING.json')
    trace=ScientificTracedProvenance(ROOT,ledger.path,binding,ledger.check,'actor-positive-scale-'+ledger.stage+'-01')
    trace.install_import_guard()
    assert read(HOME/(ledger.stage.upper()+'_AUTHORIZATION.json'))['source_binding_sha256']==sha(HOME/'SOURCE_BINDING.json')
    assert sha(HOME/'PROTOCOL.json')==binding['protocol_sha256'] and sha(HOME/'SCHEDULE.json')==binding['schedule_sha256']
    with trace.stage('real_environment_and_exact_saved_interfaces'):
        import numpy as np
        import torch
        from rc3_conservative_refinement import saved_inputs as si,runtime as rt
        from . import witness,positive,online
        actual=verify_real_environment(trace,torch,dict(witness=witness,positive=positive,online=online))
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        torch.manual_seed(11622026);np.random.seed(11622026);random.seed(11622026)
        env=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device='cpu',threads=1,deterministic_algorithms=True)
        assert env==binding['environment']
        si.digest=rt.file_sha256=trace.hash_file;si.read=trace.read_json
        from rc3_global_local_trainer import navigation_compat,approval
        from rc3_global_local_impl_review import runtime_guards
        navigation_compat.read=approval.read=trace.read_json
        navigation_compat.digest=approval.digest=runtime_guards.file_hash=trace.hash_file
        raw_np=np.load;raw_torch=torch.load
        def load_np(p,*a,**kw):
            if isinstance(p,(str,__import__('os').PathLike)):p=io.BytesIO(trace.read_bytes(p))
            return raw_np(p,*a,**kw)
        def load_torch(p,*a,**kw):
            if isinstance(p,(str,__import__('os').PathLike)) and not Path(p).resolve().is_relative_to(HOME):p=io.BytesIO(trace.read_bytes(p))
            return raw_torch(p,*a,**kw)
        np.load=load_np;torch.load=load_torch
        from rc3_conservative_refinement import model
        def forbidden(*a,**kw):raise RuntimeError('Neural Critic construction forbidden')
        model.RemainingCostCritic.__init__=forbidden
        inputs=si.SavedInputs()
        put(ledger.path/'ACTUAL_ENVIRONMENT.json',dict(environment=env,real_Torch_classification=actual,
            code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    return trace,inputs
