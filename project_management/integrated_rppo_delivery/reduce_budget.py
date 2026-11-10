"""The one allowed pre-science whole-task reduction; never edit original v1."""
import copy
from rc3_integrated_rppo.common import *


def main():
    events,_=read_chain(OUT/'ledger/events.jsonl')
    require(all(e.get('stage','').startswith(('P0','synthetic')) or e.get('role')=='P0_restore_fixture' or
        (e.get('role')=='residual_SL' and e.get('round')==0) for e in events if e['event']=='optimizer_attempt')
        and all(e['query']['kind']=='engineering' for e in events if e['event']=='request_start'),
        'Cannot reduce after scientific optimizer execution')
    require(not (HOME/'RESOURCE_REDUCTION_DECISION.json').exists(),'Exactly one whole-task reduction')
    gpu=read(HOME/'GPU_ENGINEERING_VERIFICATION.json');old=read(HOME/'PROTOCOL.json');new=copy.deepcopy(old)
    caps=dict(old['caps'],train=256*24*3,selection=3*(48+3*48*2)+96)
    new.update(schema='Integrated-RPPO-v1-small256-pre-science',caps=caps,top_request_total=sum(caps.values()),
        original_full_protocol_sha256=digest(HOME/'PROTOCOL.json'),original_task_sha256=digest(ROOT/'RC3_Integrated_DRL_Plan_and_Codex_Task_v1.md'))
    new['training'].update(rounds=256,checkpoints=[64,128,256],selection_candidates=[128,256])
    new['matched_SL']['selection_checkpoints']=[64,128,256]
    new['statistics']['P2']='Select RL and residual-SL independently from128/256 by parent mean gain vs ownSL, earlier ties. RLvsSL and RLvscontrol both mean>=.005. Same original gate; no added rule gate.'
    new['resource_revision']=dict(one_time=True,reason='Actual CPU/GPU P0 complete-round projections exceed14400s for512',
        only_changes=['Main rounds512→256 for every seed','Selection candidates128/256; no512 opportunity','Training schedule exact first256 prefix','Corresponding request ceilings reduced'],
        unchanged='Network,reward,features,Adam,KL,4epochs,gamma/lambda,H,mask,scope,seed,case splits,thresholds,P3/P4/P5 sizes',
        statistical_loss='Cannot test late512 gains;2 rather than3 selection candidates; half on-policy training episodes; no claim equivalence to original full512 protocol',
        training_device='CUDA FP32 residual head; CPU frozen base/critic and D1 inference; established before science by registered P0GPU')
    write_new(HOME/'PROTOCOL_SMALL256.json',new)
    data=read(HOME/'DATA_SPLITS.json')
    scope=dict(original_data_splits_sha256=digest(HOME/'DATA_SPLITS.json'),seeds=list(SEEDS),training_rounds_inclusive=[1,256],
        exact_training_prefix_sha256={str(s):identity(data['training'][str(s)][:256]) for s in SEEDS},
        exact_SL_batch_prefix_sha256={str(s):identity(data['SL_batches'][str(s)][:256]) for s in SEEDS},
        unchanged_split_hashes={n:identity(data[n]) for n in ('warm','selection','promotion','final_seen','final_unseen','timing_case_ids')},
        no_new_or_replaced_instances=True,unused_declared_rounds=[257,512],final_tasks_materialized=False)
    write_new(HOME/'DATA_EXECUTION_SCOPE_SMALL256.json',scope)
    b=gpu['benchmark'];per=b['rollout24_s']/24;main=256*b['round_s'];ablation=128*(b['rollout24_s']+b['PPO4_and_critic4_s'])
    estimate=1.25*(main+ablation+(192+336+96+96)*per+60)
    decision=dict(experiment=EXP,one_time_only=True,scientific_optimizer_updates_before=0,
        original_GPU_report_sha256=digest(HOME/'GPU_ENGINEERING_VERIFICATION.json'),original_CPU_report_sha256=digest(HOME/'ENGINEERING_VERIFICATION.json'),
        original_protocol_sha256=digest(HOME/'PROTOCOL.json'),protocol_sha256=digest(HOME/'PROTOCOL_SMALL256.json'),
        projected_first_seed_including_P5_s=estimate,stage_cap_s=14400,projection=dict(main256_s=main,two_ablation64_s=ablation,
            warm_selection_rule_and_ablation_evaluation_requests=720,conservative_factor=1.25,extra_critic_and_metadata_reserve_s=60),
        state='GO_SMALL256_BEFORE_SCIENCE',retained_engineering_requests=66,retained_engineering_optimizer_attempts=32,
        no_budget_reset=True,new_top_request_cap=sum(caps.values()),infrastructure_extra_pool=32,hard_possible_total=sum(caps.values())+32,
        task_authorization='User explicitly requested execution under attachment; section7 requires exactly one smaller whole-task version before first scientific training when resources do not support full plan.',
        exact_attachment_instruction='如果实际设备不能支持这些上限，必须在首次科学训练前给出一次缩小的整套任务版本及失去的统计能力，不能跑到一半改门槛。',
        user_pre_read_generated_hash_claim=False,utc=now())
    require(estimate<14400,'Small plan projection still exceeds budget');write_new(HOME/'RESOURCE_REDUCTION_DECISION.json',decision)
    write_new(HOME/'EXECUTION_PROTOCOL.json',dict(protocol_path=label(HOME/'PROTOCOL_SMALL256.json'),protocol_sha256=digest(HOME/'PROTOCOL_SMALL256.json'),
        resource_decision_sha256=digest(HOME/'RESOURCE_REDUCTION_DECISION.json'),original_protocol_preserved=True))
    print(json.dumps(decision,ensure_ascii=False))


if __name__=='__main__':main()
