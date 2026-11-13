import os,sys,io,time,traceback,platform
from pathlib import Path
from .common import ROOT,HOME,RUN,read,put,sha,state,Budget

def main():
    b=read(HOME/'SOURCE_BINDING.json');a=read(HOME/'AUTHORIZATION.json')
    if not a['authorized'] or sha(HOME/'SOURCE_BINDING.json')!=a['source_binding_sha256'] or sha(HOME/'PROTOCOL.json')!=a['protocol_sha256']:raise ValueError('R07 authorization identity')
    for n,v in b['source_sha256'].items():
        if n.startswith(HOME.name+'/') and sha(ROOT/n)!=v:raise ValueError('R07 frozen source: '+n)
    for n,v in b['R07_artifacts'].items():
        if sha(ROOT/n)!=v:raise ValueError('R07 frozen input: '+n)
    budget=Budget();out=HOME/'execution';out.mkdir(exist_ok=True)
    try:
        from rc3_r04_m1_real_20261111.provenance_use import ScientificTracedProvenance,verify_real_environment
        trace=ScientificTracedProvenance(ROOT,out,b,budget.check,RUN);trace.install_import_guard()
        with trace.stage('real_Torch_and_source_restore'):
            import torch,numpy as np
            from . import science
            actual=verify_real_environment(trace,torch,dict(science=science))
            torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
            env=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device='cpu',threads=1,deterministic_algorithms=True)
            if env!=b['environment']:raise ValueError('Exact runtime environment changed')
            from rc3_conservative_refinement import saved_inputs as si,runtime as old
            si.digest=old.file_sha256=trace.hash_file;si.read=trace.read_json
            from rc3_global_local_trainer import navigation_compat,approval
            from rc3_global_local_impl_review import runtime_guards
            navigation_compat.read=approval.read=trace.read_json;navigation_compat.digest=approval.digest=runtime_guards.file_hash=trace.hash_file
            oldnp=np.load;oldtorch=torch.load
            def np_load(p,*aa,**kw):return oldnp(io.BytesIO(trace.read_bytes(p)) if isinstance(p,(str,os.PathLike)) else p,*aa,**kw)
            def torch_load(p,*aa,**kw):return oldtorch(io.BytesIO(trace.read_bytes(p)) if isinstance(p,(str,os.PathLike)) and not Path(p).resolve().is_relative_to(HOME) else p,*aa,**kw)
            np.load=np_load;torch.load=torch_load
            put(out/'ENVIRONMENT.json',dict(environment=env,real_Torch_classifier=actual))
        with trace.stage('fixed_task_state_selection_and_paired_rollouts'):science.run(budget)
        trace.check_loaded_modules();put(out/'ACTUAL_DEPENDENCY_VERIFICATION.json',trace.summary());budget.save()
        print(dict(status='COMPLETE',counts=dict(budget.counts)),flush=True)
    except BaseException as e:
        put(out/'FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),counts=dict(budget.counts)));budget.save();state(status='FAILED_PRESERVED',error=repr(e));raise
if __name__=='__main__':main()
