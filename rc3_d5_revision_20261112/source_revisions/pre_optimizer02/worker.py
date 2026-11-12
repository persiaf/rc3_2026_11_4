import time
START_M,START_W=time.monotonic(),time.time()
import os,io,traceback,argparse,random,platform
from pathlib import Path
from .common import ROOT,HOME,RUN,read,put,sha,state,Budget

def main(phase):
    binding_file=HOME/('DATA_SOURCE_BINDING.json' if phase=='data' else 'SCIENTIFIC_SOURCE_BINDING.json')
    b=read(binding_file)
    if not read(HOME/'AUTHORIZATION.json')['authorized'] or sha(HOME/'REVISION_PROTOCOL.json')!=read(HOME/'AUTHORIZATION.json')['protocol_sha256']:
        raise PermissionError('Revision authorization identity')
    for name,digest in b['source_sha256'].items():
        if name.startswith(HOME.name+'/') and sha(ROOT/name)!=digest:raise ValueError('Frozen revision source changed '+name)
    pool_name='data' if phase in ('data','joint-data') else 'engineering' if phase=='acceptance' else 'reference' if phase=='reference' else 'deployment' if phase=='deployment' else 'independent' if phase in ('confirmation','final') else 'science'
    budget=Budget(pool_name,os.environ['D5_JOB_ID'])
    try:
        from rc3_r04_m1_real_20261111.provenance_use import ScientificTracedProvenance,verify_real_environment
        trace=ScientificTracedProvenance(ROOT,budget.path,b,budget.check,RUN+'-'+phase);trace.install_import_guard()
        with trace.stage('real_Torch_and_bound_data_sources'):
            import torch
            import numpy as np
            from . import data
            actual=verify_real_environment(trace,torch,dict(data=data))
            torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
            torch.manual_seed(2026111101);np.random.seed(2026111101);random.seed(2026111101)
            env=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device='cpu',threads=1,deterministic_algorithms=True)
            if env!=b['environment']:raise ValueError('Registered execution environment changed')
            from rc3_conservative_refinement import saved_inputs as si,runtime as source
            si.digest=source.file_sha256=trace.hash_file;si.read=trace.read_json
            from rc3_global_local_trainer import navigation_compat,approval
            from rc3_global_local_impl_review import runtime_guards
            navigation_compat.read=approval.read=trace.read_json;navigation_compat.digest=approval.digest=runtime_guards.file_hash=trace.hash_file
            original_np=np.load;original_torch=torch.load
            def np_load(p,*a,**kw):return original_np(io.BytesIO(trace.read_bytes(p)) if isinstance(p,(str,os.PathLike)) else p,*a,**kw)
            def torch_load(p,*a,**kw):return original_torch(io.BytesIO(trace.read_bytes(p)) if isinstance(p,(str,os.PathLike)) and not Path(p).resolve().is_relative_to(HOME) else p,*a,**kw)
            np.load=np_load;torch.load=torch_load
            put(budget.path/'ENVIRONMENT.json',dict(actual_stage=phase,environment=env,real_Torch_classifier=actual,
                note='Inherited used_for_scientific_attempt classifier flag is generic; actual role is actual_stage'))
        with trace.stage(phase+'_execution'):
            if phase=='data':data.run_data(budget)
            else:
                from . import science
                if phase in ('deployment','confirmation','final'):
                    from . import later_stages
                    {'deployment':later_stages.deployment,'confirmation':later_stages.confirmation,'final':later_stages.final}[phase](budget)
                else:{'joint-data':science.validate_joint,'acceptance':science.acceptance,'reference':science.references,'science':science.pipeline}[phase](budget)
        trace.check_loaded_modules();put(budget.path/'ACTUAL_DEPENDENCY_VERIFICATION.json',trace.summary())
        budget.close();print(dict(phase=phase,status='COMPLETE',job_wall_s=budget.elapsed(),counts=dict(budget.counts)),flush=True)
    except BaseException as exc:
        budget.close('FAILED_PRESERVED');put(budget.path/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),resources=budget.snapshot()))
        state(status='TECHNICAL_FAILURE_PRESERVED',failed_phase=phase,failed_job=budget.job,error=repr(exc));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['data','joint-data','acceptance','reference','science','deployment','confirmation','final']);a=p.parse_args();main(a.phase)
