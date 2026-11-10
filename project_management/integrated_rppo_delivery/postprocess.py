"""Metered post-run tables/plots/action and independent constraint verification."""
import time
from rc3_integrated_rppo.common import *
from rc3_integrated_rppo.data import Inputs
from rc3_global_local.validation import validate


def main():
    require((HOME/'RESOURCE_CLOSEOUT.json').exists(),'Core run must close first')
    require(not (HOME/'POSTPROCESS_COMPLETE.json').exists(),'Postprocessing already closed')
    from .extended_analysis import main as extended
    from .figures import main as figures
    from rc3_integrated_rppo.replay import verify
    binding={label(p):digest(p) for p in (Path(__file__),Path(__file__).with_name('extended_analysis.py'),Path(__file__).with_name('figures.py'))}
    source_record=dict(source_sha256=binding,science_source_sha256=digest(source_path()),
        preregistered_before_postprocess=True,new_complete_requests=0,new_model_forward=0,new_navigation=0,new_H=0,new_optimizer=0,
        planned='Raw/H paired contrasts; explicit Adam identity map; figures; all saved action rewards; every frozen-evaluation constraint certificate recomputed',wall='Same confirmation_and_analysis10800 cumulative; no reset')
    if (HOME/'POSTPROCESS_SOURCE_BINDING.json').exists():require(read(HOME/'POSTPROCESS_SOURCE_BINDING.json')==source_record,'Analysis source changed after its registration')
    else:write_new(HOME/'POSTPROCESS_SOURCE_BINDING.json',source_record)
    configure();log=Log();clock=Clock(log,'confirmation_and_analysis',10800);started=time.perf_counter()
    try:
        extended();clock.check();figures();clock.check();verify(True);clock.check()
        inputs=Inputs(log,clock);count=0;decisions=0;repeated={}
        starts={e['key']:e['query'] for e in log.events if e['event']=='request_start'}
        for e in list(log.events):
            if e['event']!='request_done':continue
            q=starts[e['key']]
            if q['kind'] not in ('selection','promotion','quality','timing'):continue
            r=object_read(e['object']);p,_=inputs.view(r['spec']);cert=validate(p,r['plan'])
            require(cert['valid_model_plan'] and cert['all_tasks_served'],'Independent saved-plan constraint check failed')
            raw=p.metrics(r['raw_route']);final=p.metrics(r['route'])
            require(all(abs(final[k]-r['metrics'][k])<1e-7 for k in ('J','L','K')),'Saved final metric mismatch')
            require(all(abs(raw[k]-r['metrics']['raw_'+k])<1e-7 for k in ('J','L','K')),'Saved raw metric mismatch')
            if q['kind'] in ('quality','timing'):
                k=(r['case_id'],r['seed'],r['study_method']);content=identity(dict(input=r['input_identity'],actor_sha=r['actor_sha'],
                    raw_route=r['raw_route'],final_route=r['route'],metrics=r['metrics']))
                require(k not in repeated or repeated[k]==content,'Quality/cold/hot repeated route or input identity mismatch');repeated[k]=content
            count+=1;decisions+=len(r['trace']);clock.check()
        record=dict(status='PASS',saved_frozen_evaluation_plans_checked=count,reported_raw_and_final_metrics_recomputed=True,
            full_coverage_same_arm_FA_navigation_endpoints=True,physical_robot_or_IK_certified=False,
            new_model_forward=0,new_navigation_solve=0,new_H=0,new_optimizer=0,new_complete_requests=0,
            postprocess_elapsed_s=time.perf_counter()-started,cumulative_analysis_wall_s=clock.elapsed(),source_binding_sha256=identity(binding),
            saved_input_and_navigation_load_counters=inputs.counts,quality_cold_hot_repeated_route_groups=len(repeated))
        packages=[];actor_index=read(HOME/'ACTOR_INDEX.json')
        for entry in actor_index['actors']:
            seed=entry['seed'];stage=entry.get('training_source',{}).get('stage');point=entry['update'];role=entry['checkpoint_role']
            base=inputs.ref['models'][str(seed)]
            if role=='SL256':training=dict(path=base['path'],sha256=base['checkpoint_sha256'],state_key='model')
            else:
                cp=OUT/'checkpoints'/str(seed)/stage/f'closed_{point:04}.pt'
                require(cp.exists(),'Missing fixed-endpoint recovery archive')
                training=dict(path=label(cp),sha256=digest(cp),state_key='control' if role=='residual_SL' else 'actor',
                    frozen_base_reference=base['path'],frozen_base_checkpoint_sha256=base['checkpoint_sha256'])
            packages.append(dict(actor=entry,protocol_path=label(protocol_path()),protocol_sha256=digest(protocol_path()),
                source_binding_path=label(source_path()),source_binding_sha256=digest(source_path()),H_identity=H_ID,
                registry_sha256=REGISTRY,training_checkpoint=training,inference_device='CPU FP32 D1',
                contains_optimizer_or_critic=False))
        write_new(HOME/'DEPLOYMENT_PACKAGE_MANIFEST.json',dict(packages=packages,
            recovery='Training checkpoints store residual/control, critic, Adam,RNG,cursors plus exact external frozen base identity; D1 actor files contain complete base+residual parameters'))
        write_new(HOME/'POSTPROCESS_COMPLETE.json',record);log(dict(event='postprocess_complete',receipt_sha256=digest(HOME/'POSTPROCESS_COMPLETE.json')))
    finally:clock.close();log.close()
    events,tail=read_chain(OUT/'ledger/events.jsonl');wall={}
    for e in events:
        if e['event']=='clock_close':wall[e['name']]=wall.get(e['name'],0.)+e['elapsed_s']
    training=[e for e in events if e['event']=='round_complete'];saved_label_replays=sum(24 for e in training if e['matched_SL'])
    core=read(HOME/'RESOURCE_ACCOUNTING.json');final=dict(core,wall_s=wall,core_accounting_sha256=digest(HOME/'RESOURCE_ACCOUNTING.json'),
        postprocess=read(HOME/'POSTPROCESS_COMPLETE.json'),closed_ledger_tail=tail,
        non_request_bootstrap_failures=read(HOME/'BOOTSTRAP_FAILURE_RESOURCE.json'),
        saved_teacher_label_terminal_metric_replays=saved_label_replays,
        saved_label_replay_note='Arithmetic on already saved teacher routes; no new teacher, H or oracle completion; label-state preparation distinct from repeated CE forward passes',
        resource_adjustments='CPU/GPU complete-round probes infeasible for512, one pre-science small256 protocol; old reports and budgets retained')
    write_new(HOME/'RESOURCE_ACCOUNTING_FINAL.json',final)
    print(json.dumps(record))


if __name__=='__main__':main()
