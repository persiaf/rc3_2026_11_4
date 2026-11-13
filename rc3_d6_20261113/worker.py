import time
START_M,START_W=time.monotonic(),time.time()
import os,io,traceback,random,platform,sys
from pathlib import Path
from .common import ROOT,HOME,RUN,read,put,sha,Budget,state

def main(phase):
    b=read(HOME/'SOURCE_BINDING.json');auth=read(HOME/'AUTHORIZATION.json')
    if not auth['authorized'] or auth['source_binding_sha256']!=sha(HOME/'SOURCE_BINDING.json') or auth['protocol_sha256']!=sha(HOME/'PROTOCOL.json'):raise PermissionError('D6 exact authorization')
    for name,digest in b['source_sha256'].items():
        if name.startswith(HOME.name+'/') and sha(ROOT/name)!=digest:raise ValueError('Registered D6 source changed: '+name)
    for name,digest in b['D6_artifacts'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Registered D6 input changed: '+name)
    budget=Budget(phase)
    try:
        from rc3_r04_m1_real_20261111.provenance_use import ScientificTracedProvenance,verify_real_environment
        trace=ScientificTracedProvenance(ROOT,budget.path,b,budget.check,RUN+'-'+phase);trace.install_import_guard()
        with trace.stage('real_Torch_bound_data'):
            import torch,numpy as np
            from . import data,runtime,training
            actual=verify_real_environment(trace,torch,dict(data=data,runtime=runtime,training=training))
            torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
            torch.manual_seed(2026111101);np.random.seed(2026111101);random.seed(2026111101)
            environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device='cpu',threads=1,deterministic_algorithms=True)
            if environment!=b['environment']:raise ValueError('Real execution environment differs')
            from rc3_conservative_refinement import saved_inputs as si,runtime as source
            si.digest=source.file_sha256=trace.hash_file;si.read=trace.read_json
            from rc3_global_local_trainer import navigation_compat,approval
            from rc3_global_local_impl_review import runtime_guards
            navigation_compat.read=approval.read=trace.read_json;navigation_compat.digest=approval.digest=runtime_guards.file_hash=trace.hash_file
            npload=np.load;torchload=torch.load
            def np_load(p,*a,**kw):return npload(io.BytesIO(trace.read_bytes(p)) if isinstance(p,(str,os.PathLike)) else p,*a,**kw)
            def torch_load(p,*a,**kw):return torchload(io.BytesIO(trace.read_bytes(p)) if isinstance(p,(str,os.PathLike)) and not Path(p).resolve().is_relative_to(HOME) else p,*a,**kw)
            np.load=np_load;torch.load=torch_load
            put(budget.path/'ENVIRONMENT.json',dict(environment=environment,real_Torch_classifier=actual))
        with trace.stage(phase):
            if phase=='diagnose':
                from .diagnostics import run
                run(budget)
            elif phase.startswith('eval'):
                from .science import evaluate
                evaluate(budget,int(phase[4:]))
            else:
                from .science import train
                stream,point=phase.split('_');train(budget,stream,int(point))
        trace.check_loaded_modules();put(budget.path/'ACTUAL_DEPENDENCY_VERIFICATION.json',trace.summary());budget.close()
        print(dict(status='COMPLETE',phase=phase,wall_s=budget.elapsed(),counts=dict(budget.counts)),flush=True)
    except BaseException as e:
        put(budget.path/'FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),counts=dict(budget.counts)))
        budget.close('FAILED_PRESERVED');state(status='TECHNICAL_FAILURE_PRESERVED',phase=phase,error=repr(e));raise

if __name__=='__main__':main(sys.argv[1])
