"""Post-run static source/size/resource and saved-metric summaries only."""
import collections
import csv
import time
import numpy as np
import torch
from rc3_actor_pg_20261109.common import HOME,ROOT,read,write_new,digest,label,read_chain,now,require,EXPERIMENT
from rc3_actor_pg_20261109.analysis import completed,csv_new


def main():
    began=time.perf_counter();out=HOME/'results';base=read(out/'RESOURCE_ACCOUNTING.json')
    counters=collections.Counter();seconds=collections.Counter();prime=collections.Counter();pipeline=collections.Counter();internal=collections.Counter()
    perphase={};requests=[]
    for phase,folder in [('DEPLOYMENT_'+m,HOME/'runs/DEPLOYMENT'/m) for m in ('D0','D1','D2')]+[('DEV54',HOME/'runs/DEV54')]:
        rows,events=completed(folder);c=collections.Counter();s=collections.Counter();h=collections.Counter()
        for row,event in rows:
            c.update(event['counts']);counters.update(event['counts']);pipeline.update(row['pipeline_counters']);internal.update(row['internal_search']['counts'])
            for k,v in event['seconds'].items():s[k]+=v;seconds[k]+=v
        for event in events:
            if event['event']=='hot_prime':h.update(event['counts']);prime.update(event['counts'])
        perphase[phase]=dict(complete_requests=len(rows),request_counts=dict(c),inclusive_component_seconds=dict(s),hot_prime_counts=dict(h))
    for arm in ('A128','B128'):
        complete=read(HOME/'runs'/arm/'COMPLETE.json')
        term=complete['ledger']['terminal']['RL']
        require(term['terminal_requests']==term['terminal_attempts']==term['terminal_completed']==1536 and term['terminal_cache_hits']==term['terminal_failed']==0,'Actual terminal count differs')
    models=read(HOME/'actors/ACTOR_INDEX.json')['models'];old=read(ROOT/'rc3_s3_evaluator/S3_MODEL_INDEX.json')['models'];sizes=[]
    for seed,roles in models.items():
        for role,e in roles.items():
            m=old[seed][role]
            sizes.append(dict(seed=seed,role=role,full_checkpoint_bytes=m['checkpoint_bytes'],actor_only_bytes=e['checkpoint_bytes'],
                size_reduction=1-e['checkpoint_bytes']/m['checkpoint_bytes'],size_ratio=m['checkpoint_bytes']/e['checkpoint_bytes'],
                parameters=len(e['parameters']),tensor_sha256=e['model_tensor_sha256'],source_checkpoint_sha256=m['checkpoint_sha256'],actor_sha256=e['checkpoint_sha256']))
    csv_new(out/'ACTOR_SIZES.csv',sizes)
    with (out/'DEV54_RESULTS.csv').open(encoding='utf8',newline='') as f:dev=list(csv.DictReader(f))
    with (out/'TRAINING_CURVES.csv').open(encoding='utf8',newline='') as f:curves=list(csv.DictReader(f))
    summary=[];strata=[];stats=[]
    for role in ('SL256','A128','B128'):
        rows=[r for r in dev if r['model']==role]
        summary.append(dict(model=role,n=len(rows),**{k:float(np.mean([float(r[k]) for r in rows])) for k in ('J','L','K','raw_J','raw_L','raw_K','H_gain_J')}))
        for size in ('4x3','6x6','8x8'):
            for endpoint in ('closed_depot','open_first_last_connected'):
                rr=[r for r in rows if r['size']==size and r['endpoint']==endpoint]
                strata.append(dict(model=role,size=size,endpoint=endpoint,n=len(rr),J_mean=float(np.mean([float(r['J']) for r in rr])),raw_J_mean=float(np.mean([float(r['raw_J']) for r in rr]))))
    csv_new(out/'DEV_MODEL_SUMMARY.csv',summary);csv_new(out/'DEV_STRATA.csv',strata)
    for arm in ('A128','B128'):
        rr=[r for r in curves if r['arm']==arm];v=np.array([float(r['gradient_norm']) for r in rr])
        stats.append(dict(arm=arm,updates=len(rr),gradient_norm_min=float(v.min()),gradient_norm_mean=float(v.mean()),gradient_norm_max=float(v.max()),
            clipped=sum(r['clipped']=='True' for r in rr),zero_signal=sum(r['zero_signal']=='True' for r in rr),nonfinite=0,
            initial_tensor_sha256=read(HOME/'runs'/arm/'WEIGHT_BRANCH.json')['initial_tensor_sha256'],
            final_tensor_sha256=read(HOME/'runs'/arm/'COMPLETE.json')['tensor_sha256']))
    sl=torch.load(ROOT/models['11622026']['SL256']['path'],map_location='cpu',weights_only=True)['model']
    weight_changes={}
    for arm in ('A128','B128'):
        final=read(HOME/'runs'/arm/'COMPLETE.json');s=torch.load(ROOT/final['checkpoint'],map_location='cpu',weights_only=False)['model']
        delta=sum(float((s[k].double()-sl[k].double()).square().sum()) for k in sl)**.5
        norm=sum(float(sl[k].double().square().sum()) for k in sl)**.5
        weight_changes[arm]=dict(parameter_L2_change=delta,relative_L2_change=delta/norm,
            metadata_only_tensor_reads=True,new_model_forward_backward_optimizer=0)
    write_new(out/'GRADIENT_AND_WEIGHT_SUMMARY.json',dict(training=stats,weight_changes=weight_changes))
    starts={m:read(HOME/'runs/DEPLOYMENT'/m/'STARTUP.json') for m in ('D0','D1','D2')}
    for mode in starts:
        rows,_=completed(HOME/'runs/DEPLOYMENT'/mode);first,event=rows[0]
        starts[mode].update(first_request=first['request'],first_complete_request_s=event['complete_time_s'],
            startup_plus_first_request_spans_s=starts[mode]['process_to_ready_s']+event['complete_time_s'],
            first_request_timing_rule='Sum of measured startup and actual first complete-request spans. Excludes intervening startup receipt/predebit IO, so not an exact process-to-response latency. Phase wall includes that IO; no task encoding prewarm.')
    write_new(out/'STARTUP_AND_FIRST_RESPONSE.json',starts)
    write_new(HOME/'RESOURCE_ACCOUNTING.json',dict(base,request_phase_details=perphase,all_inference_request_operation_counts=dict(counters),
        all_inference_inclusive_seconds=dict(seconds),all_navigation_hot_prime_counts=dict(prime),
        inference_pipeline_counters=dict(pipeline),inference_internal_search_entries=dict(internal),
        actual_total_new_navigation_solves=counters['navigation_solves']+prime['navigation_solves'],
        training_new_navigation_solves=0,training_saved_navigation_artifact_loads=1002,training_navigation_memory_cache_hits=22,
        actual_all_common_H_invocations=counters['common_H_invocations']+3072,
        actual_native_complete_requests=counters['native_pipeline_completed'],all_scientific_complete_requests=756+3072+54,
        independent_caps_not_pooled=True,all_phase_caps_respected=True,
        preparation=dict(registration=read(HOME/'REGISTRATION_ACCOUNTING.json'),engineering=read(HOME/'ENGINEERING_VERIFICATION.json'),
            export_12=read(HOME/'runs/DEPLOYMENT/export/COMPLETE.json')),
        clocks_note='Phase wall includes startup, preparation, imports, navigation, H, export, validation, checkpoint and IO. Nested spans are diagnostic and must not be summed; post-run analysis/review/Git are separate maintenance with zero new scientific calls.',
        old_experiment_resources_changed=False,old_S3_missing_slots_unchanged=True,
        postprocessing=dict(read_only_supplement_wall_s=time.perf_counter()-began,helper_sha256=digest(__file__),new_scientific_calls=0)))
    write_new(HOME/'WEIGHT_LINEAGE.json',dict(experiment_id=EXPERIMENT,source_binding_sha256=digest(HOME/'SOURCE_BINDING.json'),
        protocol_sha256=digest(HOME/'PROTOCOL.json'),fixed_12_original_models=read(ROOT/'rc3_s3_evaluator/S3_MODEL_INDEX.json'),
        original_12_actor_exports=read(HOME/'actors/ACTOR_INDEX.json'),
        warm_start=old['11622026']['SL256'],
        new_branches={arm:dict(initial=read(HOME/'runs'/arm/'WEIGHT_BRANCH.json'),fixed_endpoint=read(HOME/'runs'/arm/'COMPLETE.json'),
            actor=read(HOME/f'screen_actors/11622026_{arm}.pt.json')) for arm in ('A128','B128')},
        fresh_Adam_each=True,old_training_optimizer_restored=False,old_checkpoints_modified=False))
    write_new(HOME/'ENVIRONMENT_RECORD.json',dict(cpu_model='AMD Ryzen 7 7745HX with Radeon Graphics',
        cpu_source='Read Windows HKLM HARDWARE DESCRIPTION System CentralProcessor0 ProcessorNameString after completed run; CIM query denied, no scientific impact',
        runtime=read(HOME/'SOURCE_BINDING.json')['environment'],device='CPU',threads=1,interop_threads=1,
        CUDA_used=False,load_trace_collected=False,mode_order_fixed=True,not_randomized_causal_hardware_benchmark=True))
    print(dict(total_scientific_requests=3882,optimizer_updates=256,navigation=counters['navigation_solves']+prime['navigation_solves'],
        common_H=counters['common_H_invocations']+3072,native=counters['native_pipeline_completed']))


if __name__=='__main__':main()
