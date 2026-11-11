import copy
import math
import hashlib
import json
import torch
from rc3_r04_review_20261111.relative_actor import RelativeCorrection
from rc3_conservative_refinement.guard import restore_if_rejected
from rc3_conservative_refinement.runtime import tensor_digest
from .common import HOME,ROOT,DATA,read,put,sha,csvrow,archive
from .construct import construct,inspect,metrics

def adam_digest(obj):
    h=hashlib.sha256()
    def add(v):
        if isinstance(v,torch.Tensor):
            h.update(str((v.dtype,tuple(v.shape))).encode());h.update(v.detach().contiguous().cpu().numpy().tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str):h.update(str(k).encode());add(v[k])
        elif isinstance(v,(tuple,list)):
            for x in v:add(x)
        else:h.update(repr(v).encode())
    add(obj);return h.hexdigest()

def save(model,opt,step,name):
    with (HOME/'diagnostic'/name).open('xb') as f:
        torch.save(dict(model=model.state_dict(),Adam=opt.state_dict(),step=step,
            model_tensor_sha256=tensor_digest(model.state_dict()),diagnostic_only=True,
            forbidden_as_formal_initialization=True,forbidden_for_deployment=True,
            data_sha256=sha(ROOT/DATA),source_binding_sha256=sha(HOME/'SOURCE_BINDING.json')),f)

def run(ledger,trace):
    import io
    with trace.stage('exact_sealed32_state_and_zero_initialization_load'):
        data=torch.load(io.BytesIO(trace.read_bytes(ROOT/DATA)),map_location='cpu',weights_only=False)
        ledger.counts['saved_state_file_loads']+=1;states=data['states'];assert len(states)==32
        previous=trace.read_json(ROOT/'rc3_actor_positive_scale_20261111/FEASIBLE_RANK_WITNESSES.json')
        assert [s['state_id'] for s in states]==[s['state_id'] for s in previous['witnesses']]
        assert sha(ROOT/DATA)==previous['exact_selected_dataset_sha256']
        for s,w in zip(states,previous['witnesses']):
            assert s['x'].dtype==s['z0'].dtype==torch.float32 and s['mask'].dtype==torch.bool and s['x'].shape[1]==206
            assert bool(s['mask'][s['target']]) and bool(torch.isfinite(s['x'][s['mask']]).all())
            assert tensor_digest({'x':s['x'],'z0_legal':s['z0'][s['mask']],'mask':s['mask']})==w['source_feature_tensor_sha256']
            assert s['mask'].tolist()==w['legal_mask']
        initial=RelativeCorrection();initial.load_state_dict(data['initial_correction'],strict=True)
        assert tensor_digest(initial.state_dict())==data['initial_tensor_sha256']=='0fe392e9ecce47abbc957ad9942369ffd647a2b5f427bab7db5d642b55ca0d27'
        assert bool((initial.head[-1].weight==0).all() and (initial.head[-1].bias==0).all())
        ledger.counts['fixed_states']=32
        put(HOME/'DATA_LOAD_RECEIPT.json',dict(status='PASS',data_path=DATA,data_sha256=sha(ROOT/DATA),states=32,
            order=[s['state_id'] for s in states],initial_tensor_sha256=data['initial_tensor_sha256'],
            no_teacher_pool_scan_or_reselection=True,features_masks_candidate_order_frozen=True))
    with trace.stage('shared_network_parallel_bump_construction'):
        joint=construct(states,initial,ledger,trace)
    if not joint:
        put(HOME/'RANK_STAGE_STATUS.json',dict(status='NOT_RUN',reason='NO_SHARED_WITNESS',route_evaluation='NOT_RUN'))
        csvrow(HOME/'RANK_OBJECTIVE_CONTROL.csv',dict(status='NOT_RUN',step='',reason='NO_SHARED_WITNESS'))
        csvrow(HOME/'FAILURE_STATE_ANALYSIS.csv',dict(stage='shared_construction',step='',state_index='',state_id='',constraint='NO_SHARED_WITNESS',value='',bound='',candidate_parameter_sha256='',repeat_of_step=''))
        return
    model=copy.deepcopy(initial)
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    save(model,optimizer,0,'RANK_step0000.diagnostic.pt')
    accepted=0;rejected=0;consecutive=0;seen={};completed=0
    for step in range(1,129):
        ledger.check();before=copy.deepcopy(model.state_dict());before_opt=copy.deepcopy(optimizer.state_dict())
        before_adam=adam_digest(before_opt)
        with torch.no_grad():old=[model(s['z0'],s['x'],s['mask']).detach() for s in states]
        optimizer.zero_grad(set_to_none=True);loss=0.
        for s in states:
            z=model(s['z0'],s['x'],s['mask']);other=s['mask'].clone();other[s['target']]=False
            part=torch.relu(1e-4+z[other].max()-z[s['target']])/32
            if not torch.isfinite(part):raise ValueError('Nonfinite rank loss')
            part.backward();loss+=float(part.detach())
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite rank gradient')
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        ledger.optimizer('rank_lr3e-4',step);ledger.counts['diagnostic_target_presentations']+=32
        optimizer.step();ledger.counts['actor_optimizer_completed']+=1
        if any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('Nonfinite candidate weights')
        candidate_hash=tensor_digest(model.state_dict());candidate_adam=adam_digest(optimizer.state_dict())
        signature=(candidate_hash,candidate_adam);repeat=seen.get(signature);seen.setdefault(signature,step)
        candidate=inspect(model,states,old);ok=all(r['guard']['accepted'] for r in candidate)
        if not all(r['finite'] and r['mask_preserved'] and r['max_abs_delta']<=.25000001 for r in candidate):raise ValueError('Candidate finite/mask/amplitude failure')
        rollback=restore_if_rejected(model,optimizer,before,before_opt,{'accepted':ok})
        if not ok:
            ledger.counts['guard_rejections']+=1
            assert rollback['restored'] and adam_digest(optimizer.state_dict())==before_adam
            ledger.counts['exact_rollbacks']+=1
            for r in candidate:
                for k,b in [('max_step_kl',.001),('max_step_probability_change',.02),('max_base_total_variation',.1)]:
                    v=r['guard']['maxima'][k]
                    if v>b:csvrow(HOME/'FAILURE_STATE_ANALYSIS.csv',dict(stage='rank_candidate_guard',step=step,state_index=r['state_index'],state_id=r['state_id'],constraint=k,value=v,bound=b,candidate_parameter_sha256=candidate_hash,repeat_of_step='' if repeat is None else repeat))
        accepted+=int(ok);rejected+=int(not ok);consecutive=0 if ok else consecutive+1
        applied=inspect(model,states);m=metrics(applied);cm=metrics(candidate)
        row=dict(status='RUN',step=step,accepted=ok,accepted_total=accepted,rejected_total=rejected,
            loss_before_update=loss,gradient_norm=float(grad),**m,candidate_max_step_KL=cm['max_step_KL'],
            candidate_max_probability_change=cm['max_probability_change'],candidate_max_base_TV=cm['max_base_TV'],
            candidate_parameter_sha256=candidate_hash,candidate_Adam_sha256=candidate_adam,
            repeat_of_step='' if repeat is None else repeat,repeat_is_same_parameter_and_Adam=repeat is not None,
            applied_parameter_sha256=tensor_digest(model.state_dict()),rollback_exact=bool(rollback.get('restored')),wall_s=ledger.elapsed())
        csvrow(HOME/'RANK_OBJECTIVE_CONTROL.csv',row)
        archive(ledger.path/'RANK_STATE_DETAILS.jsonl.gz',dict(step=step,candidate_parameter_sha256=candidate_hash,
            candidate_Adam_sha256=candidate_adam,repeat_of_step=repeat,accepted=ok,candidate_states=candidate,applied_states=applied))
        completed=step
        if step%32==0:print(f'Rank step{step}: fit={m["strict_target_top1_count"]}/32 accepts={accepted}/{step}',flush=True)
        if consecutive>=16:break
    final=inspect(model,states);m=metrics(final)
    passed=completed==128 and m['strict_target_top1_count']/32>=.8 and accepted>=math.ceil(.9*128)
    save(model,optimizer,completed,f'RANK_step{completed:04d}_final.diagnostic.pt')
    for r in final:
        if r['margin']<=1e-5:csvrow(HOME/'FAILURE_STATE_ANALYSIS.csv',dict(stage='rank_applied_endpoint',step=completed,state_index=r['state_index'],state_id=r['state_id'],constraint='strict_target_margin',value=r['margin'],bound=1e-5,candidate_parameter_sha256=tensor_digest(model.state_dict()),repeat_of_step=''))
    # Fixed published CE reference: no CE rerun or loading its fitted weights.
    ce=trace.read_json(ROOT/'rc3_actor_positive_scale_20261111/POSITIVE_CONTROL_SUMMARY.json')['configurations'][1]
    assert ce['learning_rate']==3e-4 and ce['completed_attempts']==128 and ce['accepted']==128 and ce['final']['target_top1_count']==14
    put(HOME/'RANK_STAGE_STATUS.json',dict(status='PASS' if passed else 'FAIL',completed_attempts=completed,accepted=accepted,rejected=rejected,
        final=m,per_state=final,success_standard='Exact step128 strict margin>1e-5 in>=80%, accepted>=ceil(.9*128), finite/mask/amplitude/all applied guards',
        CE_fixed_reference=ce,CE_source_sha256=sha(ROOT/'rc3_actor_positive_scale_20261111/POSITIVE_CONTROL_SUMMARY.json'),
        original_zero_initialization=True,shared_witness_not_used_as_fit_initialization=True,route_evaluation='NOT_RUN',diagnostic_only=True))
    if not (HOME/'FAILURE_STATE_ANALYSIS.csv').exists():
        csvrow(HOME/'FAILURE_STATE_ANALYSIS.csv',dict(stage='NO_REJECTION_OR_ENDPOINT_MARGIN_FAILURE',step=completed,state_index='',state_id='',constraint='',value='',bound='',candidate_parameter_sha256='',repeat_of_step=''))
