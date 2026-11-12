"""Bind production sources after data completion, before any new optimizer."""
import ast,hashlib,json,time
from .common import ROOT,HOME,BASE,RUN,read,put,sha

def main():
    start=time.perf_counter()
    if (HOME/'SCIENTIFIC_SOURCE_BINDING.json').exists():raise FileExistsError('Do not reset science source binding')
    if read(HOME/'FULL_TEACHER_MANIFEST.json')['status']!='PASS':raise ValueError('Data stage not passed')
    put(HOME/'TARGETED_ACCEPTANCE_PROTOCOL.json',dict(schema='targeted-D5-production-acceptance-r1',seed=2026111101,
        new_request_cap=40,new_Actor_attempt_cap=4,carried_requests=15,carried_Actor_attempts=5,
        measured_PG_batch_size=8,SL_batch_size=8,restored_next_scheduled_batch_size=8,
        checks=['new joint full/local teacher loader','one full8 PG batch including actual update and checkpoint','full8 SL gradient/frozen bytes',
            'restore then independently sample the next scheduled8 on both branches, exact actions/masks/logprobs/raw/H','exact next full8 Adam update',
            'fresh formal initialization role; no P0 weights','actual32-40 requests and4 optimizer accounting','measured4096 throughput fits8h'],
        initial_reference_requests=8,resume_reference_requests_max=8,online_initial_requests=8,online_resumed_requests=16,
        scope='All inside remaining45 request/19 Actor P0 quota; no old P0 rerun or teacherTop1 gate',
        numerical_tolerance='Actions/masks/logprobs/routes/metrics/weights/Adam/sampler exact; reward sum vs-J tolerance1e-7',
        checkpoint_timing='Training timer includes required checkpoint/C export. Pure timing receipt own write excluded from stream matching, still charged in pool wall.',
        source_revision='Production stages extend the registered data source; original data producer20bf35f40 and its binding remain archived'))
    b=read(HOME/'DATA_SOURCE_BINDING.json')
    for path in HOME.glob('*.py'):
        ast.parse(path.read_text(encoding='utf8'));b['source_sha256'][path.relative_to(ROOT).as_posix()]=sha(path)
    artifacts={x:sha(HOME/x) for x in ('REVISION_PROTOCOL.json','AUTHORIZATION.json','FULL_TASK_REGISTRATION.json','ORIGINAL_LOCAL_TEACHERS.json',
        'SCHEDULE_EXPOSURE.json','FULL_TEACHER_MANIFEST.json','JOINT_DATA_INDEX.json','TARGETED_ACCEPTANCE_PROTOCOL.json')}
    payload=dict(run_id=RUN,base_commit=BASE,revision_artifacts=artifacts,source_sha256=b['source_sha256'],environment=b['environment'])
    ident=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    put(HOME/'SCIENTIFIC_IDENTITY.json',dict(identity=ident,**payload,stable_identity_excludes_downstream_admission_receipts=True))
    b.update(schema='d5-r1-production-scientific-source',inherited_source_envelope_base=b.get('base_commit'),base_commit=BASE,revision_run_id=RUN,protocol_sha256=sha(HOME/'REVISION_PROTOCOL.json'),
        schedule_sha256={str(s):sha(HOME/'schedules'/f'{s}.json') for s in (2026111101,2026111102,2026111103)},
        revision_artifacts=artifacts,scientific_identity=ident,data_producer_binding_sha256=sha(HOME/'DATA_SOURCE_BINDING.json'))
    b['artifact_sha256'][str((HOME/'SCIENTIFIC_IDENTITY.json').relative_to(ROOT)).replace('\\','/')]=sha(HOME/'SCIENTIFIC_IDENTITY.json')
    put(HOME/'SCIENTIFIC_SOURCE_BINDING.json',b)
    put(HOME/'SCIENTIFIC_AUTHORIZATION.json',dict(authorized=True,authority='Current explicit data-revision request plus already approved original conditional P1-P5',
        scientific_identity=ident,source_binding_sha256=sha(HOME/'SCIENTIFIC_SOURCE_BINDING.json'),
        preconditions=['joint352current-Environment teacher admission','targeted remaining-P0 acceptance','actual full8 throughput within8h','reference manifest PASS'],
        automatic_conditional_stage_progression=True,old_v1_no_state_changes=True))
    elapsed=time.perf_counter()-start;r=read(HOME/'RESOURCE_ACCOUNTING.json')
    r['wall']['total_s']+=elapsed;r['wall']['engineering_s']+=elapsed;put(HOME/'RESOURCE_ACCOUNTING.json',r,True)
    put(HOME/'SCIENTIFIC_REGISTRATION_RECEIPT.json',dict(wall_s=elapsed,before_new_optimizer=True,scientific_identity=ident))
    print('Bound executable production loop/reference/evaluation/SL-U-C/conditional stages/atomic recovery before targeted optimizer; no new execution yet.')

if __name__=='__main__':main()
