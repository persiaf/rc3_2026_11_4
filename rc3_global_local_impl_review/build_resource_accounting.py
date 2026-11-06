"""Consolidate saved receipts only. No project/model imports or scientific calls."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE / name).read_text(encoding='utf8'))


def main():
    e, g = read('ENGINEERING_RESULT.json'), read('GEOMETRY_GATE.json')
    b, guard = read('BASELINE_UNIT_RECEIPT.json'), read('GUARD_TEST_RECEIPT.json')
    closed = [json.loads(p.read_text(encoding='utf8')) for p in (HERE / 'baseline_plans').glob('*closed_reference.json')]
    assert len(closed) == 18
    closed_dp = sum(int(d['info'].get('dp_calls', 0)) for d in closed)
    endpoint_dp = sum(int(d.get('DP_calls', 0)) for d in e['baseline_rows'])
    times = [r['elapsed_s'] for r in e['calibration']]
    result = {
      'schema': 'rc3-preflight-resource-accounting-v1', 'date': '2026-11-06',
      'scope': 'Implementation review and bounded engineering checks; not formal S1/S2/S3 performance',
      'status': 'PREPARATION_PASS_FORMAL_TRAINING_REVISE',
      'scientific_execution_totals': {
        'new_target_task_scene_generation': 0, 'new_navigation_preparation': 0,
        'saved_S0_parent_artifact_loads': 3, 'real_engineering_complete_plans': 63,
        'endpoint_native_calls': 36, 'endpoint_open_requests_subset': 18,
        'full_candidate_closed_native_reference_calls': 18,
        'policy_complete_trajectories': 9, 'policy_encode_calls': 9,
        'policy_score_calls': 1063, 'environment_actions': 1069,
        'forced_first_actions_no_score': 6, 'model_instantiations': 1, 'checkpoint_loads': 1,
        'full_common_H_calls': 9, 'terminal_cache_hits': 0,
        'backward_calls': 3, 'optimizer_steps': 0, 'new_trained_checkpoints': 0,
        'formal_S1_training': 0, 'formal_S2_training': 0,
        'S3_target_task_queries': 0, 'original60_semantic_reads_or_evaluations': 0,
        'unexpected_failures': 0
      },
      'physical_geometry_only': {'counters': g['counts'], 'wall_s': g['elapsed_wall_s'],
        'new_unique_assets': 111, 'reference_reconstructions': 3,
        'target_tasks_nav_or_outcomes_materialized': False,
        'qualification': 'Includes unseen physical geometry gate only, not S3 target-task consumption.'},
      'engineering': {
        'counters': e['counters'], 'native_internal_counters': e['native_counters'],
        'policy_terminal_counters': e['terminal_counters'],
        'policy_internal_search': e['internal_policy_terminal'],
        'native_DP_calls': {'endpoint': endpoint_dp, 'closed_full_reference': closed_dp, 'total': endpoint_dp + closed_dp},
        'export_and_validation_attempts': 88, 'validation_passed': 88,
        'elapsed_wall_s': e['elapsed_wall_s'],
        'wall_scope': 'Measured runner snapshot; contains nested baseline/policy/IO/instrumentation; excludes final receipt write and interpreter startup. Not deployment latency.',
        'errors': e['error'], 'model_tensor_hash_before': e['model_hash_before'],
        'model_tensor_hash_after': e['model_hash_after'],
        'hardware': e['hardware'], 'peak_process_RSS_bytes': e['peak_process_RSS_bytes'],
        'calibration_start_RSS_bytes': e['calibration_start_RSS_bytes'],
        'calibration_peak_RSS_bytes': e['calibration_peak_RSS_bytes'],
        'three_rollout_plus_backward_s_by_parent': times,
        'RL_kernel_512x4_max_observed_extrapolation_s': max(times) * 4 * 512,
        'extrapolation_excludes': ['teacher generation', 'SL initialization', 'continued SL', 'data/navigation preparation', 'development evaluation', 'checkpoint/recovery and IO', 'trained policy distribution changes'],
        'formal_batch4_peak_or_GPU_measured': False,
        '7200_second_stage_budget_certified': False
      },
      'synthetic_unit_tests_separate': {
        'guard': guard,
        'endpoint_suite_latest_tests': b['latest_tests'],
        'endpoint_suite_history': b['run_history'],
        'endpoint_suite_all_runs_totals': b['all_runs_totals'],
        'expected_negative_entry_cases': b['expected_negative_entry_cases'],
        'unexpected_test_failures': b['unexpected_test_failures'],
        'DP_call_total_available': False,
        'DP_limitation': 'Unit tests checked per-return info.dp_calls; cumulative unit DP count was not captured. Do not conflate with real 54 baseline calls.',
        'notes': '61 tiny synthetic native entries include historical tests and expected negative branches; no real S0/S3 scenes, common H or model calls.'
      },
      'saved_evidence_verification_and_maintenance': {
        'layout_independent_review_receipt': 'LAYOUT_GATE_REVIEW.json',
        'layout_review_wall_s': read('LAYOUT_GATE_REVIEW.json')['elapsed_s'],
        'independent_preparation_review_complete_run_history': 'PREPARATION_AUDIT.json/verification_run_history',
        'frozen_byte_check': 'FROZEN_BYTE_CHECK.json',
        'final_archive_verification': 'ARCHIVE_VERIFICATION.json',
        'Git_and_LFS_audit': 'GIT_VERSION_AUDIT.json',
        'semantic_calls_added_by_these_verifiers': 0,
        'sealed_asset_byte_hashing': 'Frozen byte checks hash original bytes without semantic loading, task generation or test evaluation.',
        'static_review_code_writing_orchestration_wall_s': None,
        'total_task_wall_s': None,
        'timing_limitation': 'Static review/authoring/Git commands were not all independently timed. Recorded phase wall times overlap; no invented full task wall or total GPU/FLOPs estimate.'
      },
      'no_push_no_merge': True,
      'source_receipts': ['ENGINEERING_RESULT.json', 'GEOMETRY_GATE.json', 'BASELINE_UNIT_RECEIPT.json', 'GUARD_TEST_RECEIPT.json', 'PREPARATION_AUDIT.json']
    }
    (HERE / 'RESOURCE_ACCOUNTING.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'plans': 63, 'DP': result['engineering']['native_DP_calls'], 'optimizer_steps': 0, 'RL_kernel_estimate_s': max(times)*4*512}))


if __name__ == '__main__':
    main()
