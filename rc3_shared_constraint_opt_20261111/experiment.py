import copy
import io
import time
import numpy as np
import torch
from rc3_r04_review_20261111.relative_actor import RelativeCorrection
from rc3_conservative_refinement.runtime import tensor_digest
from rc3_shared_a1_rank_20261111.construct import inspect
from .common import HOME,ROOT,DATA,WITNESS,read,put,sha,csvrow,archive
from .solver import solve,slacks,MARGIN

def flatten(model):return torch.cat([p.detach().reshape(-1) for p in model.parameters()])

@torch.no_grad()
def assign(model,vec):
    offset=0
    for p in model.parameters():p.copy_(vec[offset:offset+p.numel()].reshape(p.shape));offset+=p.numel()

def save(model,name,extra):
    with (HOME/'diagnostic'/name).open('xb') as f:
        torch.save(dict(model=model.state_dict(),parameter_sha256=tensor_digest(model.state_dict()),diagnostic_only=True,
            forbidden_as_formal_initialization=True,forbidden_for_deployment=True,source_binding_sha256=sha(HOME/'SOURCE_BINDING.json'),**extra),f)

def state_rows(records,stage,outer,candidate):
    for r in records:
        csvrow(HOME/'STATE_MARGIN_RESULTS.csv',dict(stage=stage,outer=outer,candidate=candidate,state_index=r['state_index'],state_id=r['state_id'],
            margin=r['margin'],hinge_from_margin=max(0.,MARGIN-r['margin']),strict_teacher_top1=r['margin']>1e-5,unmet_target_margin=r['margin']<MARGIN,
            residual_range=r['residual_range'],max_abs_delta=r['max_abs_delta'],probability_change=r['probability_change'],
            incremental_KL=r['guard']['maxima']['max_step_kl'],base_TV=r['guard']['maxima']['max_base_total_variation'],
            guard_reference=r['guard_reference'],guard_accepted=r['guard']['accepted']))

def loss(records,states):
    total=0.
    for r,s in zip(records,states):
        z=torch.tensor(r['legal_logits'],dtype=torch.float32)
        ids=s['mask'].nonzero().flatten();t=int((ids==s['target']).nonzero()[0])
        other=torch.arange(len(z))!=t
        total+=float(torch.relu(1e-4+z[other].max()-z[t])/32)
    return total

def load_data(trace,ledger):
    data=torch.load(io.BytesIO(trace.read_bytes(ROOT/DATA)),map_location='cpu',weights_only=False)
    ledger.counts['saved_state_file_loads']+=1
    states=data['states'];prior=trace.read_json(ROOT/'rc3_actor_positive_scale_20261111/FEASIBLE_RANK_WITNESSES.json')
    assert len(states)==32 and sha(ROOT/DATA)==prior['exact_selected_dataset_sha256']
    assert [s['state_id'] for s in states]==[s['state_id'] for s in prior['witnesses']]
    for s,w in zip(states,prior['witnesses']):
        assert s['x'].dtype==s['z0'].dtype==torch.float32 and s['mask'].dtype==torch.bool
        assert s['mask'].tolist()==w['legal_mask']
        assert tensor_digest({'x':s['x'],'z0_legal':s['z0'][s['mask']],'mask':s['mask']})==w['source_feature_tensor_sha256']
    model=RelativeCorrection();model.load_state_dict(data['initial_correction'],strict=True)
    assert tensor_digest(model.state_dict())==data['initial_tensor_sha256']=='0fe392e9ecce47abbc957ad9942369ffd647a2b5f427bab7db5d642b55ca0d27'
    assert all(p.requires_grad for p in model.parameters()) and sum(p.numel() for p in model.parameters())==17473
    put(HOME/'DATA_LOAD_RECEIPT.json',dict(status='PASS',data_path=DATA,data_sha256=sha(ROOT/DATA),state_order=[s['state_id'] for s in states],
        initial_tensor_sha256=data['initial_tensor_sha256'],trainable_parameter_count=17473,all_original_parameters_included=True,
        no_teacher_pool_scan_or_state_reselection=True,shared_witness_not_read_before_optimization=True))
    return states,model

@torch.no_grad()
def isolated_path_probe(initial,states,trace,ledger,final_model):
    final_hash=tensor_digest(final_model.state_dict())
    payload=torch.load(io.BytesIO(trace.read_bytes(ROOT/WITNESS)),map_location='cpu',weights_only=False)
    ledger.counts['shared_witness_loads_after_optimization']+=1
    assert payload['diagnostic_only'] and payload['forbidden_as_formal_initialization']
    witness=payload['model'];model=copy.deepcopy(initial);orig=initial.state_dict()
    last=[s['z0'] for s in states];results=[]
    for position in range(22):
        ledger.check();state={}
        for name,value in orig.items():
            state[name]=((1-position/20)*value+(position/20)*witness[name]) if position<=20 and not name.startswith('head.4.') else (torch.zeros_like(value) if position<=20 else witness[name])
        # The conditional above keeps last-layer zero at all21 hidden positions.
        model.load_state_dict(state);ledger.counts['path_parameter_assignments']+=1
        records=inspect(model,states,last);finite=all(bool(torch.isfinite(p).all()) for p in model.parameters())
        zero_equivalent=all(r['max_abs_delta']==0. for r in records) if position<=20 else None
        valid=finite and all(r['finite'] and r['mask_preserved'] and r['max_abs_delta']<=.25000001 and r['guard']['accepted'] for r in records)
        results.append(dict(position=position,hidden_fraction=min(position/20,1.),output='zero' if position<=20 else 'shared_final',finite=finite,
            actual_guard_all_32=valid,zero_output_SL_equal=zero_equivalent,strict_teacher_top1=sum(r['margin']>1e-5 for r in records),
            parameter_sha256=tensor_digest(model.state_dict())))
        archive(ledger.path/'PATH_PROBE_STATES.jsonl.gz',dict(**results[-1],states=records))
        last=[model(s['z0'],s['x'],s['mask']).detach() for s in states]
    assert tensor_digest(final_model.state_dict())==final_hash
    put(HOME/'ISOLATED_PATH_PROBE.json',dict(status='PASS' if all(r['actual_guard_all_32'] for r in results) and results[-1]['strict_teacher_top1']==32 else 'FAIL',
        positions=results,parameter_assignments=22,learning=False,shared_weights_read_only_after_final_optimization=True,
        optimized_model_unchanged=True,not_a_formal_training_path=True,diagnostic_only=True))

def run(ledger,trace):
    with trace.stage('frozen32_state_zero_initialization_load'):states,model=load_data(trace,ledger)
    initial=copy.deepcopy(model);save(model,'JOINT_step0000.diagnostic.pt',dict(outer=0))
    records=inspect(model,states);state_rows(records,'INITIAL',0,0);accepted=0;terminal='OUTER_LIMIT';completed=0
    for outer in range(1,33):
        ledger.check();ledger.counts['subproblem_optimizations']+=1
        ledger.event('subproblem_optimizer_debit',outer=outer,count=ledger.counts['subproblem_optimizations'])
        with trace.stage('linearized_joint_subproblem_'+str(outer)):q=solve(model,states,outer,ledger)
        completed=outer
        if q['status']!='COMPLETE':terminal=q['status'];break
        old=inspect(model,states);old_loss=loss(old,states)
        if q['predicted_decrease']<=1e-9:terminal='NO_PREDICTED_DESCENT';break
        before=copy.deepcopy(model.state_dict());before_vec=flatten(model).clone();before_hash=tensor_digest(before)
        old_logits=[model(s['z0'],s['x'],s['mask']).detach() for s in states]
        direction=torch.from_numpy(q['direction']).to(torch.float32)
        applied=False
        for bt,scale in enumerate((1.,.5,.25,.125,.0625,.03125),1):
            tick=time.perf_counter();ledger.optimizer('joint_QP_parameter_candidate',outer)
            ledger.counts['actor_parameter_candidate_checks']+=1
            assign(model,before_vec+scale*direction)
            ledger.counts['actor_optimizer_completed']+=1
            candidate_hash=tensor_digest(model.state_dict());candidate=inspect(model,states,old_logits);new_loss=loss(candidate,states)
            finite=all(bool(torch.isfinite(p).all()) for p in model.parameters()) and all(r['finite'] for r in candidate)
            protection=finite and all(r['mask_preserved'] and r['max_abs_delta']<=.25000001 and r['guard']['accepted'] for r in candidate)
            descend=new_loss<old_loss-1e-9;ok=protection and descend
            pred=float(slacks(q['b']-scale*q['jv'],q['owners']).mean())
            norm=float(torch.linalg.vector_norm(flatten(model).double()-before_vec.double()))
            row=dict(outer=outer,backtrack=bt,scale=scale,accepted=ok,actual_hinge_before=old_loss,actual_hinge_after=new_loss,
                predicted_hinge_after=pred,predicted_decrease=old_loss-pred,actual_decrease=old_loss-new_loss,
                actual_parameter_update_norm=norm,direction_norm=float(np.linalg.norm(q['direction'])),
                strict_teacher_top1=sum(r['margin']>1e-5 for r in candidate),finite=finite,mask_amplitude_guard=protection,loss_descended=descend,
                max_incremental_KL=max(r['guard']['maxima']['max_step_kl'] for r in candidate),
                max_probability_change=max(r['probability_change'] for r in candidate),max_base_TV=max(r['guard']['maxima']['max_base_total_variation'] for r in candidate),
                candidate_parameter_sha256=candidate_hash,parent_parameter_sha256=before_hash,subproblem_status=q['status'],
                compute_wall_s=time.perf_counter()-tick,cumulative_wall_s=ledger.elapsed())
            csvrow(HOME/'ACTUAL_STEP_RESPONSE.csv',row);state_rows(candidate,'CANDIDATE_ACCEPT' if ok else 'CANDIDATE_REJECT',outer,bt)
            archive(ledger.path/'CANDIDATE_STATES.jsonl.gz',dict(**row,states=candidate))
            if ok:
                ledger.counts['actor_applied_updates']+=1;accepted+=1;applied=True;break
            ledger.counts['actor_rejected_candidates']+=1;model.load_state_dict(before,strict=True)
            assert tensor_digest(model.state_dict())==before_hash;ledger.counts['exact_restores']+=1
        current=inspect(model,states);state_rows(current,'APPLIED_OUTER',outer,0)
        ledger.event('outer_complete',outer=outer,accepted=applied,fit=sum(r['margin']>1e-5 for r in current),hinge=loss(current,states))
        print(dict(outer=outer,accepted=applied,fit=sum(r['margin']>1e-5 for r in current),hinge=loss(current,states)),flush=True)
        if not applied:terminal='ALL_SIX_BACKTRACKS_REJECTED';break
        if all(r['margin']>1e-5 for r in current) and loss(current,states)<=1e-7:terminal='ALL32_HINGE_CONVERGED';break
    final=inspect(model,states);state_rows(final,'FINAL',completed,0)
    hits=sum(r['margin']>1e-5 for r in final);incomplete=terminal.startswith('INCOMPLETE')
    status='INCOMPLETE' if incomplete else ('PASS' if hits>=26 else 'FAIL')
    save(model,'JOINT_final.diagnostic.pt',dict(outer=completed,quality_status=status,terminal=terminal))
    put(HOME/'OPTIMIZATION_RESULT.json',dict(status=status,terminal=terminal,outer_subproblems=completed,accepted_updates=accepted,
        final_teacher_top1=hits,final_hinge=loss(final,states),final_states=final,all_original_parameters_included=True,
        old_CE_fixed_teacher_top1=14,old_hinge_fixed_teacher_top1=18,equal_compute_claim=False,online='NOT_RUN',diagnostic_only=True))
    if not (HOME/'ACTUAL_STEP_RESPONSE.csv').exists():
        csvrow(HOME/'ACTUAL_STEP_RESPONSE.csv',dict(status='NOT_RUN',reason=terminal,outer=completed))
    with trace.stage('isolated22_position_existence_path_AFTER_optimization'):isolated_path_probe(initial,states,trace,ledger,model)
