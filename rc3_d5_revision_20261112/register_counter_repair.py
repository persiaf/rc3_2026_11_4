"""Register one proven counter-alias fix and non-refunding acceptance recovery."""
import ast,json,time,hashlib,collections
from .common import ROOT,HOME,RUN,H_ID,read,put,sha,unarchive

def main():
    start=time.perf_counter()
    if (HOME/'COUNTER_CORRECTION_RECEIPT.json').exists():raise FileExistsError('No repeated counter correction')
    r=read(HOME/'RESOURCE_ACCOUNTING.json');old=HOME/'source_revisions/accounting_failure01'
    before=read(old/'RESOURCE_ACCOUNTING.json')
    events=[json.loads(x) for x in (HOME/'EVENTS.jsonl').read_text(encoding='utf8').splitlines()]
    attempts=[e for e in events if e['event']=='request_attempt' and e['job']=='acceptance-01']
    actual=15+len(attempts)
    if len(attempts)!=29 or actual!=44 or r['counts']['requests']!=236 or r['counts']['actor_optimizer_attempts']!=7:raise ValueError('Actual consumption proof differs; no assumed reset')
    purposes=collections.Counter(e['purpose'] for e in attempts)
    if r['counts']['engineering_requests']!=61 or purposes!={'reference':12,'engineering':17}:raise ValueError('Counter alias not the exact diagnosed case')
    receipt=dict(status='CORRECTED_PROVEN_DUPLICATE_FIELD',prior_recorded_engineering_counter=61,actual_engineering_requests=44,
        old_carried_engineering_requests=15,new_actual_attempts=29,purpose_counts=dict(purposes),global_requests_unchanged=236,Actor_attempts_unchanged=7,
        diagnosis='engineering purpose increment and engineering stage increment used the same field;17 purpose-engineering requests counted twice',
        raw_failure_preserved='jobs/acceptance-01/FAILURE.json',raw_resources_preserved='source_revisions/accounting_failure01/RESOURCE_ACCOUNTING.json',
        no_actual_request_optimizer_or_time_refund=True,old_v1_unchanged=True)
    put(HOME/'COUNTER_CORRECTION_RECEIPT.json',receipt)
    r['counts']['engineering_requests']=actual
    for p,n in collections.Counter(e['purpose'] for e in events if e['event']=='request_attempt').items():r['counts']['purpose_'+p+'_requests']=n+(15 if p=='engineering' else 0)
    put(HOME/'RESOURCE_ACCOUNTING.json',r,True)
    mf=read(HOME/'REFERENCE_MANIFEST.json')
    with (HOME/'references/records/REQUEST_POINTERS.jsonl').open(encoding='utf8') as f:pointers=[json.loads(x) for x in f]
    for row in pointers:
        rec=unarchive(row['record']);tid=rec['task_id']
        if rec['actor_tensor_sha256']!='1817f58aa6506456ff12f0c5fe0a914e156aef8af319f4687dde90782f55a63e' or not rec['plan']['validation']['valid_model_plan']:raise ValueError('Completed reference is not exact validated SL256')
        if tid in mf['records']:continue
        path=HOME/'references'/(tid+'.json.gz')
        sources={f:sha(ROOT/f) for f in ('rc3_global_local/environment.py','rc3_global_local/terminal.py','rc3_global_local/validation.py','rc3_global_local/policy.py')}
        key=hashlib.sha256(json.dumps(dict(problem=rec['problem_identity'],SL=rec['actor_tensor_sha256'],H=H_ID,cost='J=L+.75K; complete unnormalized',sources=sources),sort_keys=True).encode()).hexdigest()
        mf['records'][tid]=dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),key=key,first_compute_s=row['full_request_s'],source='VALIDATED_COMPLETED_BEFORE_COUNTER_GUARD')
        mf['engineering_first_reference_s']+=row['full_request_s']
    if len(mf['records'])!=12:raise ValueError('Expected12 completed references for exact reuse')
    put(HOME/'REFERENCE_MANIFEST.json',mf,True)
    put(HOME/'RECOVERY_ACCEPTANCE_PROTOCOL.json',dict(root_cause='Confirmed double P0 count field alias',repair_number=1,max_same_root_repairs=2,
        actual_engineering_requests_before=44,actual_Actor_attempts_before=7,new_engineering_request_cap=15,new_Actor_attempt_cap=3,
        projected_engineering_total_requests_max=59,projected_engineering_Actor_attempts_max=10,
        order=['Complete only missing fixed FrozenSL references in existing training/reference pool; no optimizer admission required for constant cache',
            'Restore real cursor1 engineering checkpoint and reuse9 completed own-policy resume traces; collect only7 missing trajectories',
            'Compare actual next scheduled full8 on both branches and perform2 full8 updates',
            'Measure one fresh third scheduled full8 including1 update/checkpoint because prior timing receipt was lost'],
        no_retraining_initial_PG_or_SL=True,no_scientific_method_hyperparameter_or_gate_change=True,
        scope='Within original60 request/24 Actor P0 and original96h/300000 overall; no budget expansion or count reset'))
    b=read(HOME/'SCIENTIFIC_SOURCE_BINDING.json')
    for name in ('rc3_d5_revision_20261112/runtime.py','rc3_d5_revision_20261112/training.py','rc3_decoder_program_20261111/actor.py','rc3_decoder_program_20261111/trainer.py',
        'rc3_global_local/policy.py','rc3_global_local/environment.py','rc3_global_local/terminal.py'):
        if sha(ROOT/name)!=b['source_sha256'][name]:raise ValueError('Mathematical Actor/Trainer/runtime changed during accounting repair')
    for path in HOME.glob('*.py'):ast.parse(path.read_text(encoding='utf8'));b['source_sha256'][path.relative_to(ROOT).as_posix()]=sha(path)
    b.update(active_repair='R02_engineering_counter_and_saved_batch_resume',compatible_scientific_identity=b['scientific_identity'],
        original_scientific_source_binding_sha256=sha(HOME/'SCIENTIFIC_SOURCE_BINDING.json'),repair_registration_sha256=sha(HOME/'RECOVERY_ACCEPTANCE_PROTOCOL.json'))
    put(HOME/'SCIENTIFIC_SOURCE_BINDING_R02.json',b)
    put(HOME/'REPAIR_AUTHORIZATION_R02.json',dict(authorized=True,authority='Current user permits necessary bounded engineering repair/resume',
        scientific_identity=b['scientific_identity'],source_binding_sha256=sha(HOME/'SCIENTIFIC_SOURCE_BINDING_R02.json'),protocol_sha256=sha(HOME/'REVISION_PROTOCOL.json'),
        recovery_protocol_sha256=sha(HOME/'RECOVERY_ACCEPTANCE_PROTOCOL.json'),no_scientific_checkpoint_identity_change=True))
    put(HOME/'ACTIVE_EXECUTION_SOURCE.json',dict(binding='SCIENTIFIC_SOURCE_BINDING_R02.json',authorization='REPAIR_AUTHORIZATION_R02.json'))
    r=read(HOME/'RESOURCE_ACCOUNTING.json');elapsed=time.perf_counter()-start;r['wall']['total_s']+=elapsed;r['wall']['engineering_s']+=elapsed;put(HOME/'RESOURCE_ACCOUNTING.json',r,True)
    put(HOME/'COUNTER_REPAIR_REGISTRATION.json',dict(wall_s=elapsed,source_binding_sha256=sha(HOME/'SCIENTIFIC_SOURCE_BINDING_R02.json'),actual_consumption_kept=True))
    print('Counter alias corrected from actual receipts:44requests/7Actor;29requests and2updates not replayed. R02 source+bounded recovery registered.')

if __name__=='__main__':main()
