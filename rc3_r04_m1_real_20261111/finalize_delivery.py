"""Small saved-receipt aggregation only; no science or historical CSV audit."""
import time
START=time.perf_counter()
import hashlib
import json
from pathlib import Path

HOME=Path(__file__).resolve().parent
ROOT=HOME.parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,v):
    with p.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def main():
    attempts=[];entries=[];counts={}
    for index in ('01','02'):
        p=HOME/'attempts'/index
        registered=read(p/'REGISTRATION_RECEIPT.json')
        consumed=read(p/'ENTRY_CONSUMPTION.json')
        worker=read(p/'RESOURCE_ACCOUNTING.json')
        attempts.append(dict(index=index,run_id=consumed['run_id'],
            source_binding_sha256=sha(p/'SOURCE_BINDING.json'),
            execution_source_commit=read(p/'EXECUTION_SOURCE_COMMIT.json')['commit'],
            source_snapshot=f'attempts/{index}/entry_source',
            source_binding=f'attempts/{index}/SOURCE_BINDING.json',
            resource=f'attempts/{index}/RESOURCE_ACCOUNTING.json',resource_sha256=sha(p/'RESOURCE_ACCOUNTING.json'),
            consumption=f'attempts/{index}/ENTRY_CONSUMPTION.json',consumption_sha256=sha(p/'ENTRY_CONSUMPTION.json'),
            worker_status=worker['status'],supervisor_status=consumed['status'],
            registration_wall_s=registered['wall_s'],entry_wall_s=consumed['wall_s'],
            step0_started=consumed['step0_started'],counts=worker['counts'],
            root_exit_confirmed=consumed['root_exit_confirmed'],job_active_processes_at_end=consumed['job_active_processes_at_end']))
        entries.extend([dict(kind='registration',index=index,wall_s=registered['wall_s']),
                        dict(kind='full_supervised_entry',index=index,wall_s=consumed['wall_s'])])
        for key,value in worker['counts'].items():counts[key]=counts.get(key,0)+value
    entries.append(dict(kind='pre_registration_stdlib_AST_tool_execution',wall_s=.555))
    entries.append(dict(kind='root_saved_phase_trace_extract_tool_execution',wall_s=.286423))
    report=read(HOME/'REPORT_ACCOUNTING.json')
    entries.append(dict(kind='report_small_summary_extract_and_render',wall_s=report['report_reading_rendering_debit_s'],
                        receipt='REPORT_ACCOUNTING.json',conservative_tool_envelope=True))
    final_review=read(HOME/'FINAL_REVIEW_ACCOUNTING.json')
    entries.append(dict(kind='independent_final_small_table_review',wall_s=final_review['measured_tool_process_wall_s'],
                        receipt='FINAL_REVIEW_ACCOUNTING.json'))
    prior=read(HOME/'attempts/02/ATTEMPT_LINK.json')
    put(HOME/'ATTEMPT_LIST.json',dict(base_commit='992981414239da51c662b5060921ceea2312b7a1',
        branch='codex/rc3-m1-real-preflight-20261111',worker_launches=2,max_worker_launches=3,
        automatic_further_entry_allowed=False,scientific_stop='G3_FAIL_ORIGINAL_128_STOP',attempts=attempts,
        old_failures=prior['old_attempts'],old_engineering_wall_s=prior['old_engineering_wall_s'],
        first_entry_repair_review='attempts/01/ENTRY_REPAIR_AUTHORIZATION.json'))
    traces=[json.loads(x) for x in (HOME/'attempts/02/PREFLIGHT_TRACE.jsonl').read_text(encoding='utf-8').splitlines()]
    stages=[dict(stage=x.get('name'),seconds=x.get('seconds'),status=x.get('status'))
            for x in traces if x['event']=='stage_end']
    put(HOME/'TIMING_SUMMARY.json',dict(attempt02_stage_seconds=stages,
        nesting='heavy_imports is within preflight; saved_result_curves within save_summaries. Do not double-sum nested stages.',
        dominant_stage='restore_116_saved_trajectories',dominant_stage_s=96.00137090000135,
        first_entry_failure='WinError5 during atomic heartbeat replacement; lock holder/cause UNKNOWN',
        timeout_supervision='Previously accepted real interruption tests reused without rerun; both new process trees exited and active count0',
        scientific_stop='Original midpoint rule, not a runtime timeout'))
    put(HOME/'STAGE_STATUS.json',dict(execution='COMPLETE',scientific_G3='FAIL',
        restored_trajectories=116,restored_states=7992,step0='COMPLETE',step128='COMPLETE',
        step256='NOT_RUN_REGISTERED_MIDPOINT_STOP',C0_updates=128,C1_updates=128,
        numerical_recheck='PASS',next_step='STOP_THIS_C1_CONFIGURATION',G2='NOT_RUN',M2='NOT_RUN',M3='NOT_RUN'))
    # Original per-worker receipts remain byte-for-byte unchanged.
    entries.append(dict(kind='this_saved_receipt_aggregation',wall_s=time.perf_counter()-START))
    charged=sum(x['wall_s'] for x in entries)
    put(HOME/'RESOURCE_ACCOUNTING.json',dict(schema='rc3-m1-task-cumulative-closeout-v1',
        status='COMPLETE_G3_FAIL',new_task_cap_s=1800,new_task_accounted_wall_s=charged,
        runtime_registration_and_entries_wall_s=149.47272907448104,
        accounting_entries=entries,counts=counts,per_configuration_optimizer_attempts=dict(C0=128,C1=128),
        optimizer_cap=512,per_configuration_cap=256,worker_launches=2,worker_launch_cap=3,
        execution_wall_violation=charged>1800,old_consumption_preserved_s=prior['all_prior_wall_s'],
        cumulative_all_research_accounted_wall_s=prior['all_prior_wall_s']+charged,
        old_attempts=prior['old_attempts'],old_engineering_wall_s=prior['old_engineering_wall_s'],
        accounting_scope='Runtime/imports/checks/tests/restore/fit/evaluate/save/numerical audit and post-run small-table review included. Code authorship, prose editing, Git/network publication are maintenance, not scientific computation.',
        no_double_count='Worker wall already includes previous task consumption; aggregate uses each supervisor ENTRY_CONSUMPTION wall exactly once, plus registration and separately recorded review work.',
        retired_budget_not_reused=True,original_receipts_unchanged=True,
        further_scientific_execution_allowed=False,stop_reason='C1 conditional ratios exceed1.25 at128 in both aggregations'))
    print(json.dumps(dict(status='CLOSED',new_task_accounted_wall_s=charged,
                         all_research_accounted_wall_s=prior['all_prior_wall_s']+charged)),flush=True)

if __name__=='__main__':main()
