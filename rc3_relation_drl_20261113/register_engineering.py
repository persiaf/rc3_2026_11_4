"""Pre-register engineering identity, cases, numerical tolerances and limits."""
import ast,importlib.util,platform,sys
from pathlib import Path
from .common import ROOT,HOME,H_ID,read,put,sha,digest

def local_dependencies(seeds):
    todo=list(seeds);seen=set()
    while todo:
        path=Path(todo.pop())
        if path in seen or not path.is_file():continue
        seen.add(path);tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        pkg=path.relative_to(ROOT).with_suffix('').parts[:-1]
        for node in ast.walk(tree):
            modules=[]
            if isinstance(node,ast.Import):modules=[x.name.split('.') for x in node.names]
            elif isinstance(node,ast.ImportFrom):
                prefix=list(pkg[:len(pkg)-node.level+1]) if node.level else []
                base=prefix+(node.module.split('.') if node.module else [])
                modules=[base]+[base+x.name.split('.') for x in node.names]
            for parts in modules:
                if not parts or not parts[0].startswith('rc3_'):continue
                base=ROOT.joinpath(*parts)
                if base.with_suffix('.py').is_file():todo.append(base.with_suffix('.py'))
                elif (base/'__init__.py').is_file():todo.append(base/'__init__.py')
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(seen)}

def main():
    if (HOME/'ENGINEERING_REGISTRATION.json').exists():raise ValueError('Registration already fixed; write an explicit revision instead')
    files=local_dependencies([HOME/x for x in ('engineering.py','test_actor.py','register_engineering.py')])
    for name in ('ENGINEERING_CASES.json',):files[(HOME/name).relative_to(ROOT).as_posix()]=sha(HOME/name)
    frozen=['rc3_global_local/FUTURE_DESIGN_STATUS.json','rc3_global_local/S2_S3_protocol_v2.json','rc3_global_local_execution_v3/ACTIVE_EXECUTION_STATUS.json',
        'rc3_d5_revision_20261112/JOINT_DATA_INDEX.json','rc3_d5_revision_20261112/JOINT_DATA_ADMISSION.json','rc3_d5_revision_20261112/FULL_TEACHER_MANIFEST.json',
        'rc3_d5_revision_20261112/REFERENCE_MANIFEST.json','rc3_d5_revision_20261112/NEXT_STEP_DECISION.md','rc3_d6_20261113/NEXT_STEP_DECISION.md',
        'rc3_r07_20261113/NEXT_STEP_DECISION.md','rc3_r08_joint_runtime/NEXT_STEP_DECISION.md']
    for f in frozen:files[f]=sha(ROOT/f)
    binding=dict(schema='relation-engineering-source-v1',base_commit='198a8ff672595b5cbd7b8b7e80c4f04b1d2716fa',files=files,H_identity=H_ID,
        no_old_source_mutation=True,scope='M0/M1 only; exact current source, source-data lineage independently checked by saved loader')
    put(HOME/'ENGINEERING_SOURCE_BINDING.json',binding)
    import torch,numpy
    environment=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,numpy=numpy.__version__,torch_path=torch.__file__,
        torch_init_sha256=sha(torch.__file__),cuda_runtime=torch.version.cuda,cuda_available=torch.cuda.is_available(),
        gpu=torch.cuda.get_device_name(0),device='cuda',torch_threads=1,deterministic=True,CUBLAS_WORKSPACE_CONFIG=':4096:8',TF32=False)
    put(HOME/'ENVIRONMENT.json',environment)
    reg=dict(schema='relation-M0-M1-registration-v1',status='REGISTERED_BEFORE_EXECUTION',authorization=dict(scope='M0_M1_ONLY',
        source='Current user message: 第五章导航—覆盖关系DRL方案：实施前修订与工程启动',
        approved='Plan revision, unique relation Actor, M0/M1 bounded engineering and freeze M2 protocol',M2_execution_authorized=False),
        seed=2026111391,sampling_seed=2026111392,device='cuda',cases=read(HOME/'ENGINEERING_CASES.json')['cases'],
        limits=dict(request_attempts=128,optimizer_attempts=32,wall_s=7200),
        planned=dict(nonempty_cases=6,empty_boundary_cases=2,policy_requests=48,gradient_full_replays=24,teacher_full_replays=6,empty_requests=8,frozen_equivalence_requests=4,total_requests=90,optimizer_calls=12),
        tolerance=dict(cost_reward_abs=1e-7,logprob_atol=2e-5,logprob_rtol=1e-5,mask_exact=True,frozen_routes_exact=True,restored_weight_Adam_RNG_bytes_exact=True),
        source_binding_sha256=sha(HOME/'ENGINEERING_SOURCE_BINDING.json'),environment_sha256=sha(HOME/'ENVIRONMENT.json'),
        synthetic_unit_tests=dict(count=10,complete_requests=0,optimizer_calls=0,backward_calls=1,manual_parameter_mutations=1),
        recovery='Two separate worker processes at closed case3 checkpoint; no resource reset; dirty requests/updates block free replay',
        engineering_models_for_science=False,original60_oldS3_fixed32_semantic_reads=0)
    put(HOME/'ENGINEERING_REGISTRATION.json',reg)
    print({'source_files':len(files),'registration_sha256':sha(HOME/'ENGINEERING_REGISTRATION.json'),'limits':reg['limits'],'planned_requests':90})

if __name__=='__main__':main()
