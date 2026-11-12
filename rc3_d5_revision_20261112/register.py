"""Pre-computation task identities, resource transfer and user authorization."""
import copy,collections,random,hashlib,ast,subprocess,time
from .common import ROOT,HOME,BASE,RUN,H_ID,read,put,sha

def main():
    start=time.perf_counter()
    if (HOME/'REVISION_PROTOCOL.json').exists():raise FileExistsError('No data-registration reset')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=BASE:raise ValueError('Wrong revision base')
    old=read(ROOT/'rc3_decoder_program_20261111/PROGRAM_PROTOCOL.json');m=read(ROOT/'rc3_score5_20261111/SAVED_TEACHER_METADATA.json')
    registry=read(ROOT/'rc3_global_local_execution_v3/REGISTRY.json');ref=read(ROOT/'rc3_integrated_rppo/DATA_REFERENCES.json')
    bank=read(ROOT/'rc3_integrated_rppo/runs/rc3_integrated_rppo_v1_20261110_01/inputs/INDEX.json')
    first={}
    for row in sorted(m['teachers'],key=lambda x:(x['spec']['parent'],x['index'])):first.setdefault(row['spec']['parent'],row)
    if set(first)!=set(range(96)) or len({x['physical_key'] for x in first.values()})!=96:raise ValueError('Wrong96 parents')
    full=[];local=[]
    for parent,row in sorted(first.items()):
        spec=copy.deepcopy(row['spec']);spec.update(width=10,height=10,x0=0,y0=0,endpoints='closed_depot',role='train')
        pkey=f"{spec['map_seed']}_{spec['task_seed']}_A{spec['A_condition']}";saved=bank.get(pkey,ref['bank'].get(pkey))
        if saved is None:raise ValueError('Missing saved parent '+pkey)
        full.append(dict(id=f'F{parent:03}',parent=parent,spec=spec,physical_key=row['physical_key'],expected_parent_identity=saved['parent_identity'],
            input_files=saved['files'],selection='Minimum original teacher index within each registered training parent; no cost used',source_teacher_metadata_index=row['index']))
    original_files=[]
    for row in m['teachers']:
        src=row['source'];path=ROOT/src['path']
        if sha(path)!=src['sha256']:raise ValueError('Original local teacher changed')
        original_files.append(src);local.append(dict(id=f"L{row['index']:03}",teacher_index=row['index'],spec=row['spec'],physical_key=row['physical_key'],source=src,view_identity=row['view_identity']))
    put(HOME/'FULL_TASK_REGISTRATION.json',dict(schema='d5-full-training96-r1',records=full,registered_before_new_native_or_H=True))
    put(HOME/'ORIGINAL_LOCAL_TEACHERS.json',dict(index=ref['teacher_pool_index'],files=original_files,records=local,old_bytes_verified=True))
    usage={};schedules={}
    categories={'F':[x['id'] for x in full]}
    for size in ((4,3),(6,6),(8,8)):categories[f'{size[0]}x{size[1]}']=[x['id'] for x in local if (x['spec']['width'],x['spec']['height'])==size]
    for seed in old['seeds']:
        exposed={};counts={'F':16384,'4x3':5462,'6x6':5461,'8x8':5461}
        for cat,ids in categories.items():
            seq=[ids[i%len(ids)] for i in range(counts[cat])]
            rs=int(hashlib.sha256(f'{RUN}:schedule:{seed}:{cat}'.encode()).hexdigest()[:16],16);random.Random(rs).shuffle(seq);exposed[cat]=seq
        cursor=collections.Counter();batches=[]
        for batch in range(4096):
            ids=exposed['F'][cursor['F']:cursor['F']+4];cursor['F']+=4
            for cat,n in zip(('4x3','6x6','8x8'),old['optimization']['local_rotation'][batch%3]):ids+=exposed[cat][cursor[cat]:cursor[cat]+n];cursor[cat]+=n
            if len(ids)!=8:raise ValueError('Batch count')
            batches.append(ids)
        path=HOME/'schedules'/f'{seed}.json';put(path,dict(seed=seed,batches=batches,methods=['D5-PG','Full-PG','D5-SL'],same_task_order=True))
        u=dict(collections.Counter(x for b in batches for x in b));usage[str(seed)]=u;schedules[str(seed)]=dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path))
    if not usage[str(old['seeds'][0])]==usage[str(old['seeds'][1])]==usage[str(old['seeds'][2])]:raise ValueError('Unequal exposure between seeds')
    put(HOME/'SCHEDULE_EXPOSURE.json',dict(unique_full=96,unique_local=256,unique_total=352,slots_per_seed=32768,batches=4096,
        schedules=schedules,uses_by_seed=usage,same_task_order_per_seed=True,label_use_is_distinct_from_new_teacher_queries=True))
    protocol=dict(schema='d5-data-revision-and-scientific-execution-r1',run_id=RUN,base_commit=BASE,
        original_protocol=dict(path='rc3_decoder_program_20261111/PROGRAM_PROTOCOL.json',sha256=sha(ROOT/'rc3_decoder_program_20261111/PROGRAM_PROTOCOL.json')),
        original_master_plan=dict(path='rc3_decoder_program_20261111/MASTER_PLAN.md',sha256=sha(ROOT/'rc3_decoder_program_20261111/MASTER_PLAN.md')),
        model=old['model'],optimization=old['optimization'],seeds=old['seeds'],deploy_seed=old['deploy_seed'],H_identity=H_ID,
        evaluation=old['evaluation'],wall_limits=dict(data=3600,engineering=21600,training=259200,deployment=28800,independent=28800,reproduction=7200),
        data=dict(full_task_identities=96,native_attempts=104,uniform_H_attempts=104,technical_recovery_extra=8,
            teacher='Unchanged endpoint Detour16, horizon16, original portfolio; reject fallback/failures; then unchanged uniform terminal.finish H once',
            reuse='Only exact source-bound complete teacher manifest entries; none in the registered original256 pool; no guessed old route reuse',
            retry='Only same problem after documented I/O/process interruption; source/input/feasibility/limit error stops; at most8, never replacement',
            replay='Current Environment.mask/step at every action; terminal metrics only for label validation, not a new H or PG return label',new_scenes=0,new_navigation=0),
        budget_transfer=dict(global_wall_s=345600,global_complete_requests=300000,teacher_suballocation_wall_s=3600,
            teacher_suballocation_complete_requests=208,source='Within original72h training pool and42404 recovery/technical request reserve',
            prior_requests=15,prior_Actor_attempts=5,prior_engineering_wall_s=read(ROOT/'rc3_decoder_program_20261111/RESOURCE_ACCOUNTING.json')['wall']['charged_compute_job_wall_s'],
            total_budget_increase=0,engineering_remaining_requests=45,engineering_remaining_Actor_attempts=19),
        targeted_engineering=dict(request_cap_remaining=45,Actor_cap_remaining=19,planned_batches=1,
            checks=['new full teacher loader and replay','8-task complete PG batch gradients/frozen bytes','exact model/Adam/sampling and next online trajectory resume','correct purpose and non-refundable accounting','actual full batch throughput'],
            throughput='Using measured full8 collection+backward+checkpoint cost,4096 opportunities plus full unique reference cost must fit8h stream; no artificial cap bypass'),
        stage_gates='Exactly original MASTER_PLAN P1-P5 thresholds and fixed4096 endpoints; only data allowance and new execution identity change',
        old_v1_never_rewritten=True,critic_updates=0,closed_branches_reopened=False)
    put(HOME/'REVISION_PROTOCOL.json',protocol)
    put(HOME/'AUTHORIZATION.json',dict(authorized=True,authority='Current user request: 补齐D5完整教师数据和科学驱动，进入既定P1',
        scope='96 fixed complete teachers;104native/104uniformH/3600s within original totals; targeted remaining-P0 acceptance; conditional original P1-P5',
        protocol_sha256=sha(HOME/'REVISION_PROTOCOL.json'),base_commit=BASE,old_v1_state_kept=True))
    prior=read(ROOT/'rc3_decoder_program_20261111/RESOURCE_ACCOUNTING.json');c=dict(prior['counts']);c.update(engineering_requests=15,engineering_optimizer_attempts=5,
        scientific_optimizer_attempts=0,teacher_native_attempts=0,teacher_uniform_H_attempts=0,teacher_recovery_attempts=0,backward_calls=c.get('backward_calls',0))
    for field in ('teacher_requests','reference_requests','train_requests','eval_requests','deployment_requests','confirmation_requests','final_requests'):c[field]=0
    wall=prior['wall']['charged_compute_job_wall_s']
    put(HOME/'RESOURCE_ACCOUNTING.json',dict(schema='d5-cumulative-budget-r1',counts=c,wall=dict(total_s=wall,engineering_s=wall,training_s=0,data_s=0,deployment_s=0,independent_s=0,reproduction_s=0),jobs={},
        inherited_resource=dict(path='rc3_decoder_program_20261111/RESOURCE_ACCOUNTING.json',sha256=sha(ROOT/'rc3_decoder_program_20261111/RESOURCE_ACCOUNTING.json')),
        old_missing_details_preserved='Original13 request search and precursor forward remain UNKNOWN; new counters are actual, no zero replacement'))
    put(HOME/'STATUS.json',dict(run_id=RUN,phase='DATA',status='REGISTERED',base_commit=BASE,old_v1_status='P6_TECHNICAL_DATA_INCOMPLETE',
        old_v1_scientific_stages='NOT_RUN',new_execution=True,next_action='Generate/reuse exact registered96 complete strong teachers',scientific_result=None))
    bind=read(ROOT/'rc3_decoder_program_20261111/SOURCE_BINDING_R02.json')
    for path in HOME.glob('*.py'):ast.parse(path.read_text(encoding='utf8'));bind['source_sha256'][path.relative_to(ROOT).as_posix()]=sha(path)
    bind['revision_artifacts']={x:sha(HOME/x) for x in ('REVISION_PROTOCOL.json','AUTHORIZATION.json','FULL_TASK_REGISTRATION.json','ORIGINAL_LOCAL_TEACHERS.json','SCHEDULE_EXPOSURE.json')}
    bind['revision_run_id']=RUN;put(HOME/'DATA_SOURCE_BINDING.json',bind)
    elapsed=time.perf_counter()-start
    put(HOME/'REGISTRATION_RECEIPT.json',dict(wall_s=elapsed,before_new_teacher_forward_or_optimizer=True,new_native=0,new_H=0,new_Actor=0))
    resources=read(HOME/'RESOURCE_ACCOUNTING.json')
    for key in ('total_s','training_s','data_s'):resources['wall'][key]+=elapsed
    put(HOME/'RESOURCE_ACCOUNTING.json',resources,True)
    print('Registered96 fixed full tasks,352-task joint schedule x3seeds, original-budget transfer and data source before new calls.')

if __name__=='__main__':main()
