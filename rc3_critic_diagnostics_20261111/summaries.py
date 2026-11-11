"""Preregistered descriptive arithmetic from saved offline tables only."""
import csv
import json
import math
from pathlib import Path
from .ledger import write_new


def boolean(value):
    return value == 'True' or value is True


def summarize(folder):
    folder=Path(folder)
    with (folder/'ACTOR_RANK_RESPONSE.csv').open(encoding='utf-8',newline='') as f:
        actor=list(csv.DictReader(f))
    with (folder/'CRITIC_MATCHED_CALIBRATION.csv').open(encoding='utf-8',newline='') as f:
        metrics=list(csv.DictReader(f))
    many=[r for r in actor if not boolean(r['single_legal_action'])]
    single=len(actor)-len(many)
    locked=sum(boolean(r['gap_gt_0_5']) for r in many)
    actual=[]
    for r in actor:
        prefix=r['source_after']+'__'
        actual.append({key:float(r[prefix+'residual_'+key]) for key in
                       ('mean_square','common_mean_square','centered_mean_square','span','min','max')})
    total=sum(x['mean_square'] for x in actual)
    common=sum(x['common_mean_square'] for x in actual)
    actor_stats=dict(states=len(actor),trajectories=len({r['trajectory_id'] for r in actor}),
        physical_parents=len({r['parent_id'] for r in actor}),single_legal_states=single,
        multiple_legal_states=len(many),gap_gt_0_5_states=locked,
        gap_gt_0_5_fraction_multiple=(locked/len(many) if many else None),
        gap_gt_0_5_fraction_all=locked/len(actor),
        single_or_strictly_locked_fraction=(single+locked)/len(actor),
        post_update_top1_changes=sum(boolean(r['pre_to_post_top1_changed']) for r in actor),
        post_update_any_rank_changes=sum(int(r['pre_to_post_rank_changed_candidates'])>0 for r in actor),
        residual_min=min(x['min'] for x in actual),residual_max=max(x['max'] for x in actual),
        actual_residual_state_equal_energy_common_fraction=common/total if total else None,
        actual_residual_centered_rms=math.sqrt(sum(x['centered_mean_square'] for x in actual)/len(actual)),
        actual_residual_rms=math.sqrt(total/len(actual)),
        actual_residual_max_span=max(x['span'] for x in actual),
        pre_update_ratio_min=min(float(r['pre_update_ratio']) for r in actor),
        pre_update_ratio_max=max(float(r['pre_update_ratio']) for r in actor),
        post_update_ratio_min=min(float(r['post_update_ratio']) for r in actor),
        post_update_ratio_max=max(float(r['post_update_ratio']) for r in actor),
        feature_switch_ON_top1_changes=sum(boolean(r['ON_feature_switch__top1_differs']) for r in actor),
        feature_switch_OFF_top1_changes=sum(boolean(r['OFF_feature_switch__top1_differs']) for r in actor),
        ON_OFF_weights_fixed_ON_top1_changes=sum(boolean(r['ON_OFF_weights_fixed_ON__top1_differs']) for r in actor),
        ON_OFF_weights_fixed_OFF_top1_changes=sum(boolean(r['ON_OFF_weights_fixed_OFF__top1_differs']) for r in actor))
    strata=[]
    for phase in sorted({r['phase'] for r in actor}):
        for size in sorted({(r['width'],r['height']) for r in actor}):
            rows=[r for r in actor if r['phase']==phase and (r['width'],r['height'])==size]
            if not rows: continue
            mult=[r for r in rows if not boolean(r['single_legal_action'])]
            strata.append(dict(phase=phase,width=int(size[0]),height=int(size[1]),states=len(rows),
                multiple_legal_states=len(mult),single_legal_states=len(rows)-len(mult),
                gap_gt_0_5_states=sum(boolean(r['gap_gt_0_5']) for r in mult),
                pre_post_top1_changes=sum(boolean(r['pre_to_post_top1_changed']) for r in rows)))
    value_metrics=[r for r in metrics if r['component']=='value']
    summary=dict(actor=actor_stats,actor_strata=strata,critic_value_metrics=value_metrics,
        interpretation='Offline mechanism evidence only; no new route quality, no R03 stopping-metric replacement')
    write_new(folder/'MECHANISM_SUMMARY.json',summary)
    return summary
