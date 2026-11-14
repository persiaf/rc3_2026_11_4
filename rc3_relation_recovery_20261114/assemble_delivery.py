"""Assemble review metadata from closed engineering/audit artifacts only."""
import hashlib
import json
import subprocess
from pathlib import Path
from .persistence import read, sha, digest

ROOT = Path(__file__).resolve().parents[1]
HOME = Path(__file__).resolve().parent
SOURCE_COMMIT = 'd6dd4ff603762d36058e96cc540ad29405dc61e6'


def write(name, value):
    (HOME / name).write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode())


def ref(path):
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path))


def main():
    audit = read(HOME / 'SL238_RECOVERY_AUDIT.json')
    original = read(ROOT / audit['protocol']['path'])
    files = dict(read(ROOT / audit['source_binding']['path'])['files'])
    final_test = read(HOME / 'engineering_results/ATTEMPT_04.json')
    tested_source = read(HOME / 'engineering_results/ATTEMPT_04_SOURCE.json')['files']
    assert final_test['success'] and final_test['tests'] == 36
    core = ['__init__.py', 'common.py', 'persistence.py', 'observe.py', 'runtime.py',
            'runtime_boundary.py', 'recovery_adapter.py', 'recovery_runner.py']
    for name in core:
        path = HOME / name
        key = path.relative_to(ROOT).as_posix()
        current = sha(path)
        assert current == tested_source[key]
        committed = subprocess.check_output(['git', 'show', SOURCE_COMMIT + ':' + key], cwd=ROOT)
        assert hashlib.sha256(committed).hexdigest() == current
        files[key] = current
    assert 'rc3_r04_m1_recovery_20261111/supervisor.py' in files
    write('NEW_SOURCE_BINDING.json', dict(schema='relation-M2a-recovery-source-v1',
        source_commit=SOURCE_COMMIT, original_binding=audit['source_binding'],
        final_engineering_test=ref(HOME / 'engineering_results/ATTEMPT_04.json'), files=files))
    entries = [json.loads(x) for x in (HOME / 'engineering_results/SUITE_DEBITS.jsonl').read_text(encoding='utf8').splitlines()]
    ends = [e for e in entries if e['event'] == 'end']
    reports = [read(HOME / f'engineering_results/ATTEMPT_{i:02}.json') for i in range(1, 5)]
    assert len(ends) == 4
    resources = dict(schema='M2a-persistence-repair-resource-v1', verdict='ENGINEERING_PASS',
        scientific_optimizer_calls=0, formal_teacher_replays=0, DRL_trajectories=0, new_parent_tests=0,
        model_constructions=0, model_forwards=0, true_engineering_optimizer_calls=0, CUDA_RNG_restores=0,
        read_only_checkpoint_tensor_loads=1, read_only_teacher_records_checked=1910,
        engineering_suite_invocations=4, suite_cap=4, tests_executed=sum(r['tests'] for r in reports),
        historical_failed_suite_invocations=1, final_suite_tests=36, final_suite_failed_tests=0,
        cumulative_synthetic_begin_method_calls=sum(e['synthetic_begin_calls'] for e in ends),
        synthetic_count_scope='Includes rejected calls and fake optimizer accounting events; no actual optimizer',
        cumulative_suite_process_wall_s=sum(e['wall_s'] for e in ends), suite_wall_cap_s=600,
        read_only_audit_tool_wall_s=8.2484207,
        audit_clock_scope='exec_command wall_time_seconds; separate from suite subprocess clock',
        total_turn_wall_s=None, total_turn_wall_note='Editing, source review and Git transfer not comprehensively instrumented',
        original_science=dict(status='TECHNICAL_INCOMPLETE', request_debits=1910, optimizer_calls=238, wall_s=2221.312),
        requested_future_increment=dict(technical_recompute_requests=6, scientific_updates_added_to_original_target=0,
            cumulative_request_cap=4102, remaining_requests=2192, remaining_optimizer_calls=274, remaining_wall_s=12178.688),
        formal_recovery_executed=False, M2b_executed=False)
    write('RESOURCE_ACCOUNTING.json', resources)
    fault = dict(historical_failure=ref(ROOT / 'rc3_relation_m2_20261113/FAILED_REQUEST_DETAIL.json'),
        actual_historical_lock_owner='UNKNOWN_NO_HISTORICAL_HANDLE_TRACE',
        old_monitor_source=ref(ROOT / 'rc3_relation_m2_20261113/observe.py'),
        current_file_permission_observation=dict(date_local='2026-11-14', target=audit['resource']['path'],
            attributes='Archive', read_only=False, directory_Administrators='FullControl',
            directory_Authenticated_Users='Modify', method='PowerShell Get-Acl/Get-Item read-only',
            historical_ACL_state_proven=False),
        independent_windows_reproductions=[e for e in final_test['evidence'] if e['case'].startswith(('test_01_', 'test_02_', 'test_05_'))])
    write('FAULT_EVIDENCE.json', fault)
    protocol = dict(schema='relation-M2a-recovery-proposal-v1', status='DRAFT_NOT_APPROVED',
        scientific_id='relation_navcover_m2_v1_20261113', recovery_id='relation_navcover_m2a_recovery_v1_20261114',
        interrupted_commit=audit['stop_commit'], implementation_commit=SOURCE_COMMIT,
        original_protocol=audit['protocol'], original_source_binding=audit['source_binding'],
        original_stage_authorization=audit['original_stage_authorization'],
        original_initialization_lineage=original['initialization'], original_schedule=original['schedule'],
        original_optimizer=original['optimizer'], retained_problem=original['retained'],
        new_source_binding=ref(HOME / 'NEW_SOURCE_BINDING.json'), sl238_audit=ref(HOME / 'SL238_RECOVERY_AUDIT.json'),
        historical_binding=ref(HOME / 'HISTORICAL_BINDING.json'), recovery_schedule=ref(HOME / 'RECOVERY_SCHEDULE.json'),
        checkpoint=audit['checkpoint'], model_Adam_RNG_hashes=audit['checkpoint'],
        canonical_recovery_folder='rc3_relation_drl_20261113/science/relation_navcover_m2_v1_20261113/init/recovery_v1_20261114',
        historical_consumption=dict(request_attempts=1910, optimizer_attempts=238, wall_s=2221.312),
        remaining_limits=dict(request_attempts=2192, optimizer_attempts=274, wall_s=12178.688),
        cumulative_limits=dict(request_attempts=4102, optimizer_attempts=512, wall_s=14400),
        technical_recompute_increment=6, historical_paid_uncommitted=audit['paid_after_checkpoint'],
        batch239_tasks=audit['batch239_tasks'], first_recovery_update=239, last_recovery_update=512,
        later_batches=273, tasks_per_update=8, recompute_entire_batch239=True,
        lost_graph_reuse=False, historical_complete_failed_overlap_preserved=True,
        ledger_origin='Original 1910 debits, 238 optimizer calls, all2148 IDs,2221.312s; original counts retained including raw complete/failed overlap',
        new_request_id_pattern='recovery_v1_init_{update:04}_teacher_{slot}', optimizer_id_pattern='init_{update:04}',
        restore='Exact SL238 model/Adam/RNG; construct objects before RNG restore; fresh empty gradients; no old graph reuse',
        training_rule='Original SL loss/gradient/optimizer/mask/teacher identity and validation; only I/O and attempt identities changed',
        persistence=dict(authority='fsynced hash-chained EVENTS.v2.jsonl', snapshot='derived only',
            replace_attempts=6, retry_delays_s=[.02, .04, .08, .16, .32], retry_scope='os.replace only',
            uncertain_append='STOP; no retry/truncation', monitor='immutable published status; never active ledger',
            cursor='checkpoint bytes + STATE + checkpoint_commit event'),
        supervision=dict(source='rc3_r04_m1_recovery_20261111/supervisor.py',
            wall_limit_s=12178.688, closeout_reserve_s=10, termination_tolerance_s=2,
            tolerance_adds_budget=False, outer_clock_starts_before_admission=True,
            actual_wall_receipt='OUTER_ACCOUNTING.json', parent_consumption_added_once=True),
        failure_policy=dict(technical_error='STOP_TECHNICAL_INCOMPLETE', source_or_data_mismatch='STOP',
            illegal_teacher_or_nonfinite='STOP', wall_or_resource_limit='STOP',
            replacement_samples=False, second_recovery_attempt_allowed=False, automatic_replay=False),
        completion=dict(required_counts=dict(request_attempts=4102, request_completed_raw=4102,
            historical_request_failed_raw=1, optimizer_attempts=512, optimizer_completed=512),
            optimized_training_slots=4096, raw_compute_requests=4102,
            receipt='RECOVERY_COMPLETE.json', preserves_old_TECHNICAL_INCOMPLETE=True,
            overwrite_original_COMPLETE=False),
        authorization_requirements=dict(new_protocol_approval=True, new_stage_authorization=True,
            stage='M2a_recovery', binds='canonical JSON SHA256 of this entire proposal plus exact original authorization reference',
            old_M2a_authorization_sufficient=False),
        scientific_updates_this_turn=0, formal_teacher_replays_this_turn=0, M2b_authorized=False,
        engineering_acceptance=ref(HOME / 'engineering_results/ATTEMPT_04.json'))
    write('RECOVERY_PROTOCOL_DRAFT.json', protocol)
    write('CURRENT_STAGE_STATUS.json', dict(schema='M2a-persistence-recovery-review-v1', conclusion='ENGINEERING_PASS',
        original_science='TECHNICAL_INCOMPLETE', recovery_proposal='DRAFT_NOT_APPROVED',
        proposal_file_sha256=sha(HOME / 'RECOVERY_PROTOCOL_DRAFT.json'), proposal_canonical_sha256=digest(protocol),
        implementation_commit=SOURCE_COMMIT, source_binding_sha256=sha(HOME / 'NEW_SOURCE_BINDING.json'),
        scientific_optimizer_calls=0, formal_teacher_replays=0, DRL_trajectories=0, new_parent_tests=0,
        engineering_tests_passed=36, original_evidence_preserved=True,
        recovery_stage_authorization_created=False, recovery_run_directory_created=False,
        M2b_started=False, next_action='STOP_PENDING_SEPARATE_RECOVERY_APPROVAL_AND_STAGE_AUTHORIZATION',
        publication='LOCAL_PREPARED'))
    print(json.dumps(dict(bound_files=len(files), final_test_count=36,
        suite_wall_s=resources['cumulative_suite_process_wall_s'], proposal_sha256=sha(HOME / 'RECOVERY_PROTOCOL_DRAFT.json'),
        proposal_canonical_sha256=digest(protocol))))


if __name__ == '__main__':
    main()
