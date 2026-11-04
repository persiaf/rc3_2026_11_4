"""Reconcile trajectory accounting; never execute a policy or a training update."""
import json
from .problem import ROOT,write


def main():
    out=ROOT/'outputs';train=json.loads((out/'training_summary.json').read_text());wall=json.loads((out/'wall_protocol.json').read_text())
    history=[json.loads(line) for line in (out/'training.jsonl').read_text().splitlines()]
    assert len(history)==train['updates'] and history[-1]['sampled_trajectories']==train['sampled_trajectories']
    evaluation=json.loads((out/'final_evaluation.json').read_text());verification=json.loads((out/'verification.json').read_text())
    updates=json.loads((out/'baseline_updates.json').read_text())
    directional=[]
    for row in updates:
        lower=row['candidate_mean_J']<row['baseline_mean_J']
        directional.append(dict(update=row['update'],candidate_lower=lower,
            improvement_direction_p=row['one_sided_p'] if lower else 1-row['one_sided_p'],replaced=row['replaced']))
        assert row['replaced']==(lower and row['one_sided_p']<.05)
    greedy=dict(training_rollout_baseline=train['baseline_greedy_trajectories'],
                training_validation=train['cpu_validation_greedy_trajectories'],
                calibration_forward_backward_probes=wall['probe_greedy_trajectories'],
                calibration_single_instance_latency_probes=len(wall['single_instance_initial_inference_s']),
                c1_contract_checks=12,
                final_development_initial_and_selected=sum(r['method'] in ('initial','model') for r in evaluation),
                cli_example=1 if (out/'cli_example.json').exists() else 0)
    sampled=dict(c2_optimizer_rollouts=train['sampled_trajectories'],c1_contract_checks=7)
    write(out/'resource_accounting.json',dict(optimizer_updates=len(history),
        stochastic_trajectories=sampled,all_stochastic_trajectories=sum(sampled.values()),
        greedy_trajectories=greedy,all_greedy_trajectories=sum(greedy.values()),
        sampled_route_replays=verification['sampled_route_replays'],replayed_service_transitions=verification['replayed_service_transitions'],
        saved_geometry_rechecks=verification['saved_plan_geometry_checks'],
        final_verification_geometry_certificates=verification['verification_geometry_certificates_total'],
        first_attempt_geometry_certificates=verification['initial_attempt_additional_geometry_certificates'],
        all_verification_geometry_certificates=verification['verification_geometry_certificates_total']+verification['initial_attempt_additional_geometry_certificates'],
        new_case_measurements=dict(c0_development=480,c2_development=len(evaluation),small_exact=12,full_and_restricted_dp_calls=24),
        training_input_preparation_wall_s=sum(x['preparation_s'] for x in history),training_loop_elapsed_s=train['actual_elapsed_s'],
        baseline_replacement_directional_p=directional,
        notes=[
            'C2 cap refers to optimizer sampling: 16384. Seven earlier C1 stochastic test trajectories are reported separately, so total stochastic calls are not claimed to be 16384.',
            'C1 inference counts are posthoc from two executed six-test runs and the final seven-test run: 7 stochastic and 12 greedy trajectories, not independent training scenes.',
            'The training-summary calibration field counts only 64 forward/backward probe trajectories. Two additional single-instance timing probes are included here.',
            'Original baseline-update log stores two-sided p/2 even if candidate mean worsens. The actual replacement always also requires a lower mean; directional p is corrected here for interpretation without changing original logs or decisions.',
            'All counts are trajectories/episodes, not individual batched forward invocations. Numeric stored-route replays and geometric rechecks are additional verification work, without new optimization or neural inference.',
            'Development and selection are reused for screening; reserved independent test and cross-scale scenes were not queried.'
            ,'Verifier first attempt completed all numeric replays and543 geometry certificates, then hit default GBK decoding of UTF8 CLI data. Explicit completion rechecked544 certificates without repeating numeric environment replays. Both attempts/logs and posthoc recovery provenance retained;1087 actual verification certificates are not1087 independent scenarios.'
        ]))


if __name__=='__main__':main()
