"""Metadata/source registration only: no Torch import or model computation."""
import ast
import collections
import platform
import subprocess
import time
from .common import HOME,ROOT,BASE,RUN,sha,read,put

def main():
    start=time.perf_counter()
    if (HOME/'SOURCE_BINDING.json').exists():raise FileExistsError('Do not reset program registration')
    subprocess.run(['git','merge-base','--is-ancestor',BASE,'HEAD'],cwd=ROOT,check=True)
    attachment=ROOT/'RC3_长期目标与Codex持续执行任务书_20261111 (2).txt'
    with (HOME/'NORMATIVE_TASKBOOK.txt').open('xb') as f:f.write(attachment.read_bytes())
    metadata=read(ROOT/'rc3_score5_20261111/SAVED_TEACHER_METADATA.json')
    teachers=metadata['teachers'];counts=collections.Counter((t['spec']['width'],t['spec']['height']) for t in teachers)
    expected=sha(ROOT/'rc3_score5_20261111/SAVED_TEACHER_METADATA.json')
    assert expected=='5c0db1f1208d6e1a448d529272258221214106668bf0ccd02f941c8c816cbd15'
    reg=read(ROOT/'rc3_global_local_execution_v3/REGISTRY.json')
    assert len(reg['S1_teachers'])==256 and [(t['spec']['width'],t['spec']['height']) for t in teachers]==[(s['width'],s['height']) for s in reg['S1_teachers']]
    available_full=counts[(10,10)]
    admission=dict(status='PASS' if available_full else 'BLOCKED_MISSING_FULL_TEACHERS',
        source='rc3_score5_20261111/SAVED_TEACHER_METADATA.json',source_sha256=expected,
        legal_teacher_pool_size=256,physical_training_parents=96,
        existing_labels_by_size={f'{w}x{h}':counts[(w,h)] for w,h in ((10,10),(4,3),(6,6),(8,8))},
        requirement='Same PG/SL task schedule, 50% full10x10, every SL task has an existing legal full sequence',
        complete_teacher_labels_missing=available_full==0,
        minimum_32768_slot_full_occurrences=16384,minimum_distinct_missing_full_task_labels=1,
        no_implicit_new_teacher_queries=True,no_local_only_SL_substitution=True,
        original60_and_oldS3_reads=0,new_teacher_pool_scan=0,actual_teacher_labels_read_this_registration=0,
        registration_reads='Only existing saved metadata plus S1_teacher and S2_train registry fields; no task/result semantics from sealed branches')
    put(HOME/'DATA_ADMISSION.json',admission)
    prior=read(ROOT/'rc3_score5_20261111/ENGINEERING_ENTRY_CONSUMPTION_R02.json')
    put(HOME/'PRECURSOR_CONSUMPTION.json',dict(wall_s=prior['cumulative_wall_s'],failed_complete_requests=1,Actor_attempts=0,
        H_calls=1,H_evidence='runtime terminal reached terminal.finish export_attempts access after unchanged H; failed counter transfer omitted this call',
        internal_search_details=None,internal_search_status='UNKNOWN_NOT_PERSISTED_IN_PRECURSOR_FAILURE',
        first_failure='rc3_score5_20261111/engineering/FAILURE_EARLY.json',second_failure='rc3_score5_20261111/engineering_r02/FAILURE.json',
        original_resource_files_preserved=True,previous_metadata_label_reads=512))
    old=read(ROOT/'rc3_score5_20261111/SCHEDULE.json')
    selected=[next(t for t in teachers if (t['spec']['width'],t['spec']['height'])==size) for size in ((4,3),(6,6),(8,8))]
    protocol=dict(schema='rc3-decoder-program-v1',run_id=RUN,base_commit=BASE,normative_sha256=sha(HOME/'NORMATIVE_TASKBOOK.txt'),
        seeds=[2026111101,2026111102,2026111103],deploy_seed=2026111101,
        model=dict(architecture='Unchanged GlobalLocalPolicy hidden64',total_scalars=65088,D5_trainable_scalars=25152,
            allowlist=['key.weight','dynamic.weight','nav.weight','query.weight','query.bias'],frozen_scalars=39936,
            FullPG_parameters='All original Actor parameters',main_input='Original global14/task5/candidate7/prefixGRU64; not detached206 residual input'),
        optimization=dict(lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=0,clip_norm=1,entropy_weight=0,
            batch=8,full=4,local_rotation=[[2,1,1],[1,2,1],[1,1,2]],PG_updates=4096,SL_U=4096,SL_max=16384,
            PG_loss='mean_i detach(J_H_sample_i-J_H_FrozenSL_same_task_i)*sum_t logpi(a_it)',
            SL_loss='mean_episode mean_action masked CE',gamma=1,critic_updates=0,guard=False,extra_A1_amplitude=False,
            reference='FrozenSL same-task complete greedy+unchanged H; no B-cond/global table, no normalization or critic',
            formal_initialization='Exact legal SL256 only, never engineering or diagnostic weights'),
        budget=dict(total_wall_s=345600,training_pool_wall_s=259200,P0_wall_s=21600,deployment_wall_s=28800,
            independent_wall_s=28800,reproduction_wall_s=7200,each_training_stream_s=28800,
            actual_optimizer_attempts=80000,PG_logical_updates=24576,SL_logical_updates=49152,
            complete_planning_attempts=300000,teacher_label_replays=393216,per_stream_failure_replay_limit=64,
            same_root_cause_repairs=2,new_teacher_queries=0),
        stage_rules='Exact gates, data boundaries, timing, statistics and exits in frozen MASTER_PLAN.md and NORMATIVE_TASKBOOK.txt',
        evaluation=dict(development=old['evaluation'],checkpoints=[0,1024,2048,4096],retained_layers=[24,24],tasks_per_parent=6,
            bootstrap=20000,statistics_seed=2026111199,method_order_seed=2026111198,cold_warm_repeats=3,
            final_endpoint=4096,no_best_checkpoint_or_seed_selection=True),
        P0=dict(ordinary_training_specs=[old['engineering'][0]]+[t['spec'] for t in selected],teacher_indices=[t['index'] for t in selected],
            complete_request_cap=60,Actor_attempt_cap=24,new_planned_complete_requests=13,new_planned_Actor_attempts=5,
            checks=['exact initialization logits/mask/raw/H','five-tensor finite graph, frozen bitwise, strict optimizer identity',
                'complete scalar return=-J_H, legal probability zero','real PG and existing-label SL updates',
                'FullActor graph with the same PG recipe','weight/cache versions after step and restore',
                'frozen export equality','exact resume Model/Adam/RNG/cursor/queries/next update'],
            teacher_Top1_or_fixed32_gate=False,engineering_models_scientific_initialization=False),
        admission=dict(P0_engineering_authorized=True,P1_requires=['P0 engineering PASS','DATA_ADMISSION PASS','8h-stream throughput admissible'],
            data_admission=admission['status'],known_failure_exit='P6_TECHNICAL_DATA_INCOMPLETE'))
    put(HOME/'PROGRAM_PROTOCOL.json',protocol)
    user_text='我批准你展示的完整计划，以及其中明确的资源上限、数据边界、阶段门槛、恢复规则和失败出口。请开始实施，并在这一版本范围内按条件连续推进。本次批准覆盖计划中明确列出的后续条件阶段；条件满足时直接执行，并登记进入该阶段的依据。允许范围内的工程修复、断点恢复，以及提交和推送本周期独立 codex/ 分支。保留用户文件、历史记录和封存数据。不授权重开关闭支线、读取原60或旧S3、超预算、购买额外算力、合并main或重写历史。科学结果未通过时，按计划完成分析和交付；只有确需改变本次批准边界或受到平台权限阻塞时，再说明具体原因。'
    put(HOME/'APPROVAL_RECEIPT.json',dict(execution_approved=True,authority='Current explicit user message',user_message=user_text,
        normative_document_sha256=sha(HOME/'NORMATIVE_TASKBOOK.txt'),resolved_plan_sha256=sha(HOME/'MASTER_PLAN.md'),
        approval_scope='Taskbook v1 default recipe/caps/conditions; no teacher-data boundary expansion; source-compatible engineering fixes',
        independent_approved_plan_file_present_before_registration=False,
        clarification='No prior full-cycle MASTER_PLAN/PROGRAM_PROTOCOL existed in this checkout; the old score5 proposal is not treated as this approved recipe. This receipt binds the user approval to taskbook defaults, preserves the known data gate failure, and does not invent prior displayed-plan file identity.',
        conditional_stages_auto_admit_only_when_gates_pass=True))
    ps=subprocess.check_output(['powershell','-NoProfile','-Command',"$c=Get-CimInstance Win32_ComputerSystem;$p=Get-CimInstance Win32_Processor;[pscustomobject]@{CPU=$p.Name;Cores=$p.NumberOfCores;Logical=$p.NumberOfLogicalProcessors;RAM_bytes=$c.TotalPhysicalMemory}|ConvertTo-Json -Compress"],text=True,encoding='utf8').strip()
    put(HOME/'ENVIRONMENT_METADATA.json',dict(hardware=__import__('json').loads(ps),GPU='NVIDIA GeForce RTX4070 Laptop GPU 8188MiB (not enabled)',
        selected_device='cpu',threads=1,deterministic_algorithms=True,python_command='rc3_learning/.venv/Scripts/python.exe -B -X utf8',
        registered_environment=read(ROOT/'rc3_score5_20261111/SOURCE_BINDING_R02.json')['environment']))
    b=read(ROOT/'rc3_score5_20261111/SOURCE_BINDING_R02.json')
    for p in HOME.glob('*.py'):
        ast.parse(p.read_text(encoding='utf8'));b['source_sha256'][p.relative_to(ROOT).as_posix()]=sha(p)
    b.update(schema='decoder-program-source-v1',run_id=RUN,program_artifacts={name:sha(HOME/name) for name in (
        'MASTER_PLAN.md','PROGRAM_PROTOCOL.json','DATA_ADMISSION.json','APPROVAL_RECEIPT.json','NORMATIVE_TASKBOOK.txt','PRECURSOR_CONSUMPTION.json','ENVIRONMENT_METADATA.json')})
    put(HOME/'SOURCE_BINDING.json',b)
    put(HOME/'STATUS.json',dict(run_id=RUN,phase='P0',status='REGISTERED_P0_ENGINEERING_ONLY_DATA_GATE_KNOWN_BLOCKED',
        plan_sha256=sha(HOME/'MASTER_PLAN.md'),protocol_sha256=sha(HOME/'PROGRAM_PROTOCOL.json'),source_binding_sha256=sha(HOME/'SOURCE_BINDING.json'),
        scientific_training='NOT_RUN',P1_admission=admission['status'],next_action='Necessary ordinary-task engineering, then P6 if the fixed data gate remains blocked'))
    put(HOME/'REGISTRATION_RECEIPT.json',dict(status='FROZEN_BEFORE_ANY_NEW_MODEL_FORWARD_REQUEST_OR_OPTIMIZER',wall_s=time.perf_counter()-start,
        registration_python=platform.python_version(),new_forward_calls=0,new_optimizer_attempts=0,new_planning_requests=0,
        old_precursor_resources_preserved_and_carried=True))
    print('Program source, default full-cycle protocol, user approval and BLOCKED_MISSING_FULL_TEACHERS data gate registered; no new model computation.')

if __name__=='__main__':main()
