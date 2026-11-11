"""Small control/source registration only; no scientific or synthetic execution."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
from .provenance import build_manifest

ROOT=Path(__file__).resolve().parents[1]
HOME=Path(__file__).resolve().parent
OLD=ROOT/'rc3_r04_m1_20261111'
BASE='ff749ff1e6a994138ebafb1b88bdbc077a528216'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def put(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def main():
    started=time.perf_counter()
    if (HOME/'ATTEMPT_LINK.json').exists():raise RuntimeError('Immutable recovery registration exists')
    names=('M1_DATA_AND_SOURCE_IDENTITY.json','BATCH_SCHEDULE.json','M1_EXECUTION_PROTOCOL.json')
    identities={}
    for name in names:
        payload=(OLD/name).read_bytes()
        with (HOME/name).open('xb') as f:f.write(payload)
        identities[name]=hashlib.sha256(payload).hexdigest()
    old=json.loads((OLD/'RESOURCE_ACCOUNTING.json').read_text(encoding='utf-8'))
    if old['wall_s']!=2859.2566758999983 or old['status']!='FAILED_PRESERVED':raise ValueError('Old failure identity differs')
    link=dict(schema='rc3-r04-m1-distinct-recovery-attempt-link-v1',
        old_attempt=dict(directory=OLD.name,base_commit=BASE,resource_path=(OLD/'RESOURCE_ACCOUNTING.json').relative_to(ROOT).as_posix(),
            resource_sha256=sha(OLD/'RESOURCE_ACCOUNTING.json'),wall_s=old['wall_s'],status=old['status'],critic_optimizer_attempts=0),
        new_attempt=dict(run_id='rc3-r04-m1-recovery-20261111-01',directory=HOME.name,wall_cap_s=1800,max_attempts=1),
        engineering=dict(cap_s=300,run_id='rc3-r04-m1-engineering-20261111-01',required_before_science=True),
        scientific_files_unchanged=identities,prior_consumption_not_reset=True,
        cumulative_disclosure='old2859.2566759 + actual engineering + actual new scientific attempt, all separately retained',
        user_authority='2026-11-11 explicit instruction: minimal runtime repair; <=300s synthetic acceptance; onGO directly one new <=1800s M1; frozen116 inputs/settings; no Actor/G2/M3/new requests')
    put(HOME/'ATTEMPT_LINK.json',link)
    controls=dict(schema='rc3-m1-runtime-recovery-controls-v1',base_commit=BASE,
        scientific_protocol_sha256=identities['M1_EXECUTION_PROTOCOL.json'],scientific_modifications=[],
        worker_module=HOME.name+'.worker',supervisor_module=HOME.name+'.supervisor',
        source_verification='actual imports/read dependencies only; exact SHA source bytes compiled; lazy frozen catalogue is not an eager whole-repository audit',
        work_admission_cutoff_s=1650,worker_cooperative_deadline_s=1780,supervisor_force_termination_s=1790,
        total_scientific_cap_s=1800,termination_tolerance_s=5,supervisor_poll_s=.05,
        host_suspend='Use max(monotonic,UTC) from supervisor entry; retainactualoverrun andviolation. Temporarysystemexecutionrequest doesnotpreventforcedsleep.',
        failed_attempt_policy='one new science launch only, durable exclusive run_id; no automatic retry or resetting prior consumption',
        outputs='new directory only, raw/normalized meanings and fixed scientificunits unchanged')
    put(HOME/'RECOVERY_EXECUTION.json',controls)
    engineering=dict(schema='rc3-m1-synthetic-timeout-preregistration-v1',cap_s=300,
        outer_force_stop_s=290,outer_termination_tolerance_s=5,each_case_wall_cap_s=8,
        each_case_force_stop_s=4,each_case_termination_tolerance_s=2,
        cases=['normal','duplicate_run_id','blocked','unresponsive','child_tree','worker_receipt_failure','supervisor_receipt_failure'],
        child_scope='PrivateJobassignedbeforeresume; ActiveProcesses0 plusroot/knownchildexit; knownsiblingsurvives andowner cleansonlyitsownhandle',
        counters=dict(Actor_updates=0,Critic_updates=0,model_forwards=0,new_tasks=0),
        failure_policy='Any failed engineeringcase stops task before science; retain all consumption and errors')
    put(HOME/'ENGINEERING_REGISTRATION.json',engineering)
    manifest=build_manifest(ROOT,HOME)
    manifest.update(base_commit=BASE,protocol_sha256=identities['M1_EXECUTION_PROTOCOL.json'],
        data_sha256=identities['M1_DATA_AND_SOURCE_IDENTITY.json'],schedule_sha256=identities['BATCH_SCHEDULE.json'],
        startup_files_sha256={name:sha(HOME/name) for name in ('ATTEMPT_LINK.json','RECOVERY_EXECUTION.json','ENGINEERING_REGISTRATION.json')})
    put(HOME/'SOURCE_BINDING.json',manifest)
    authority=link['user_authority']
    put(HOME/'STAGE_AUTHORIZATION.json',dict(schema='rc3-m1-recovery-explicit-user-authorization-v1',authorized=True,
        authority=authority,source_binding_sha256=sha(HOME/'SOURCE_BINDING.json'),
        scientific_protocol_sha256=identities['M1_EXECUTION_PROTOCOL.json'],runtime_controls_sha256=sha(HOME/'RECOVERY_EXECUTION.json'),
        run_id=link['new_attempt']['run_id'],science_cap_s=1800,engineering_cap_s=300,engineering_GO_required=True,
        max_science_attempts=1,critic_attempts=512,Actor=0,M2=False,G2=False,M3=False))
    put(HOME/'REGISTRATION_RECEIPT.json',dict(status='REGISTERED_NO_EXECUTION_YET',
        utc=datetime.now(timezone.utc).isoformat(),maintenance_wall_s=time.perf_counter()-started,
        source_binding_sha256=sha(HOME/'SOURCE_BINDING.json'),scientific_bytes_unchanged=identities,
        old_source_catalogue_entries=manifest['original_source_catalogue_count'],old_files_eagerly_hashed=0,
        model_reads=0,model_forwards=0,optimizer_attempts=0))
    print(json.dumps(dict(status='REGISTERED',source_binding_sha256=sha(HOME/'SOURCE_BINDING.json'),
        registered_trajectories=116,engineering_cap_s=300,science_cap_s=1800)))

if __name__=='__main__':main()
