"""Bind timing-only migration before a real zero-forward clock acceptance."""
import ast,datetime,time,collections,json
from .common import ROOT,HOME,read,put,sha,state

def main():
    started=time.perf_counter();now=time.time();out=HOME/'R04_TIMING_SOURCE_REGISTRATION.json'
    if out.exists():raise FileExistsError('No repeated timing migration')
    pause=read(HOME/'R04_MAINTENANCE_PAUSE_REGISTRATION.json');receipt=read(HOME/'controls/science-02/SUPERVISOR_RECEIPT.json')
    if not receipt['root_exit_confirmed'] or receipt['job_active_processes_at_end']!=0:raise RuntimeError('Owned old tree still active')
    folder=HOME/'streams/2026111101/D5-PG';cp=read(folder/'LATEST.json');timer=read(folder/'TRAINING_TIME.json');active=read(folder/'ACTIVE_BATCH.json')
    if cp['cursor']!=1771 or cp['sha256']!=timer['checkpoint_sha256'] or sha(ROOT/cp['path'])!=cp['sha256']:raise ValueError('Exact migration checkpoint differs')
    if active['op']!=1772 or len(active['episodes'])!=8:raise ValueError('Expected8 completed current-actor episodes survive')
    r=read(HOME/'RESOURCE_ACCOUNTING.json')
    if r['counts']['scientific_optimizer_attempts']!=1771 or r['counts']['actor_optimizer_attempts']!=1781:raise ValueError('Actual optimizer count differs; no reset')
    sleep=pause['historical_low_power_interval_s']
    put(HOME/'R04_TIMING_MIGRATION_RECEIPT.json',dict(checkpoint_sha256=cp['sha256'],cursor=cp['cursor'],original_stream_wall_s=timer['train_s'],
        official_historical_sleep_s=sleep,matching_awake_s=timer['train_s']-sleep,
        source='Official Power-Troubleshooter event1, same interval fully contained in completed legacy stream',
        sleep_event_sha256=sha(HOME/'PLATFORM_SLEEP_EVENT_01.json'),whole_old_stream_and_total_wall_unchanged=True,
        model_Adam_RNG_query_budget_and_cursor_unchanged=True,preserved_checkpoint='source_revisions/clock_pause01/step01771_training.pt',
        migration_applies_once_to_exact_legacy_checkpoint=True))
    put(out,dict(status='FROZEN_BEFORE_CLOCK_TEST_AND_RESUME',scope='Only timing metadata and exact recovery; inherited optimizer update math unchanged',
        registered_maintenance_stop_sha256=sha(HOME/'R04_MAINTENANCE_PAUSE_REGISTRATION.json'),clock_test_wall_limit_s=10,
        awake_measurement='Windows QueryUnbiasedInterruptTime; includes awake I/O and processing, excludes system low power',
        interrupted_batch_time='Exact previous batch UTC start to confirmed supervisor exit minus union of official sleep-event overlaps; metadata query failure means INCOMPLETE, not assumed zero',
        all_budget_guards='Original total96h, pools and per-stream8h continue using all elapsed wall including sleep, without refund',
        matching='Only SL-C target/selection use awake training wall plus unchanged exact same-task reference allocation; original2% tolerance',
        recovered_current_actor_episodes=8,repeated_teacher_or_navigation=0,new_optimizer_configuration=False,new_forward_or_optimizer_for_clock_test=0,
        incomplete_forward_backward_tail_after_maintenance='UNKNOWN; completed optimizer attempts and all wall charged, never filled with zero',
        no_quality_gate_endpoint_model_data_schedule_reward_mask_H_change=True))
    b=read(HOME/'SCIENTIFIC_SOURCE_BINDING_R03.json')
    for name in ('runtime.py','data.py','evaluation.py','later_stages.py'):
        path=HOME/name
        if sha(path)!=b['source_sha256'][path.relative_to(ROOT).as_posix()]:raise ValueError('Training data/sampling/evaluation math changed')
    for name in ('rc3_decoder_program_20261111/actor.py','rc3_decoder_program_20261111/trainer.py','rc3_global_local/policy.py','rc3_global_local/environment.py','rc3_global_local/terminal.py'):
        if sha(ROOT/name)!=b['source_sha256'][name]:raise ValueError('Actor/optimizer/H changed')
    for name in ('clocking.py','clock_acceptance.py','register_clock_repair.py','worker.py','science.py','training.py'):
        path=HOME/name;ast.parse(path.read_text(encoding='utf8'));b['source_sha256'][path.relative_to(ROOT).as_posix()]=sha(path)
    b.update(active_repair='R04_all_wall_budget_and_awake_matching_split',prior_execution_binding_sha256=sha(HOME/'SCIENTIFIC_SOURCE_BINDING_R03.json'),repair_registration_sha256=sha(out))
    for name in ('R04_TIMING_SOURCE_REGISTRATION.json','R04_TIMING_MIGRATION_RECEIPT.json','PLATFORM_SLEEP_EVENT_01.json'):b['revision_artifacts'][name]=sha(HOME/name)
    put(HOME/'SCIENTIFIC_SOURCE_BINDING_R04.json',b)
    put(HOME/'REPAIR_AUTHORIZATION_R04.json',dict(authorized=True,authority='Current explicit engineering-fix/resume permission; original SL-C excludes waiting; no increased resources',
        scientific_identity=b['scientific_identity'],source_binding_sha256=sha(HOME/'SCIENTIFIC_SOURCE_BINDING_R04.json'),protocol_sha256=sha(HOME/'REVISION_PROTOCOL.json'),
        requires_R03_IO_acceptance_PASS=True,requires_R04_clock_acceptance_PASS=True,no_scientific_checkpoint_identity_change=True))
    put(HOME/'ACTIVE_EXECUTION_SOURCE.json',dict(binding='SCIENTIFIC_SOURCE_BINDING_R04.json',authorization='REPAIR_AUTHORIZATION_R04.json'),True)
    pause_start=datetime.datetime.fromisoformat(pause['utc']).timestamp();manual=now-pause_start
    # This finite earlier source-authoring interval excludes subsequent idle monitoring.
    prepare=max(0.,(HOME/'PROVENANCE_AND_ACCOUNTING_NOTES.md').stat().st_ctime-(HOME/'build_saved_tables.py').stat().st_ctime)
    r['wall']['engineering_s']+=manual+prepare;r['wall']['total_s']+=manual+prepare
    if r['wall']['engineering_s']>=21600-20 or r['wall']['total_s']>=345600-20:raise RuntimeError('Existing cumulative wall boundary')
    r.setdefault('recording_limits',{})['maintenance_1772_uncommitted_forward_backward_tail']='UNKNOWN; actual optimizer attempts1771 and supervisor wall fully retained'
    r['jobs']['science-02']['exit_classification']='REGISTERED_MAINTENANCE_FOR_TIMING_INTERFACE; original supervisor status preserved'
    elapsed=time.perf_counter()-started;r['wall']['engineering_s']+=elapsed;r['wall']['total_s']+=elapsed;put(HOME/'RESOURCE_ACCOUNTING.json',r,True)
    put(HOME/'R04_REGISTRATION_RECEIPT.json',dict(registration_wall_s=elapsed,engineering_repair_interval_s=manual,
        prior_saved_output_source_authoring_s=prepare,scientific_requests_refunded=0,scientific_Actor_attempts_refunded=0,source_binding_sha256=sha(HOME/'SCIENTIFIC_SOURCE_BINDING_R04.json')))
    state(status='REGISTERED_TIMING_RECOVERY',recovery_checkpoint_cursor=1771,active_job=None,
        error='Registered maintenance separates awake matching from full wall; all old consumption retained')
    print(dict(status='R04_FROZEN',cursor=1771,whole_wall_s=timer['train_s'],matching_awake_s=timer['train_s']-sleep,all_optimizer_counts_kept=1781))

if __name__=='__main__':main()
