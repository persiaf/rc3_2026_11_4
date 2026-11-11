import copy
import math
from collections import Counter
import torch
from rc3_conservative_refinement.guard import check_update,GuardLimits,restore_if_rejected
from rc3_conservative_refinement.runtime import tensor_digest,_assert_frozen
from rc3_frozen_reference_pg_20261111.actor import load_actor
from .common import HOME,ROOT,read,put,sha,csvrow,archive,identity,RATES
from . import witness

@torch.no_grad()
def restored_states(inputs,actor,index,ledger):
    from rc3_global_local.environment import Environment
    ledger.counts['existing_teacher_label_reads']+=1
    problem,actions,evidence=inputs.teacher(index)
    if not actions:
        if len(problem.C):raise ValueError('Empty saved teacher with tasks')
        return
    def no_terminal(*a,**kw):raise RuntimeError('Terminal/H forbidden in positive replay')
    env=Environment(problem,no_terminal);encoded=actor.encode(problem)
    for t,node in enumerate(actions):
        ledger.check()
        if env.done or not env.mask()[node-1]:raise ValueError('Saved teacher action not legal')
        f=actor.features(env,encoded)
        ledger.counts['restored_teacher_states']+=1
        sid=identity(dict(teacher=index,view=problem.identity,prefix=list(env.prefix),state=t))
        yield dict(state_id=sid,teacher_index=index,state_index=t,view_identity=problem.identity,
            physical_key=evidence['physical_key'],parent_identity=problem.parent.identity,
            prefix_local=list(env.prefix),target=node-1,target_global=int(problem.candidate_ids[node-1]),
            candidate_ids=problem.candidate_ids.tolist(),teacher_source=evidence['teacher_source'],
            x=f['x'].detach(),z0=f['z0'].detach(),mask=f['mask'].detach())
        after=env.remaining & ~problem.C[:,node-1]
        if t==len(actions)-1:
            if after.any():raise ValueError('Incomplete saved teacher prefix')
        else:
            if not after.any():raise ValueError('Saved teacher has trailing actions')
            env.step(node)

def select(inputs,actor,ledger,trace):
    selected=[];statistics=Counter()
    initial=copy.deepcopy(actor.correction)
    with trace.stage('restore_saved_teacher_states_and_validate_actual_A1_witnesses'):
        for index in range(256):
            for state in restored_states(inputs,actor,index,ledger):
                mask,z,target=state['mask'],state['z0'],state['target'];legal=mask.nonzero().flatten()
                order=legal[torch.argsort(z[legal],descending=True,stable=True)]
                rank=int((order==target).nonzero()[0])+1
                if rank==1:statistics['teacher_already_SL_top1']+=1;continue
                state['teacher_SL_rank']=rank
                found,reason=witness.construct(state,initial,ledger)
                if found is None:statistics[reason]+=1;continue
                statistics['feasible_states']+=1
                if rank==2:statistics['feasible_SL_second']+=1
                key=(0 if rank==2 else 1,state['state_id'])
                # Identity-only ordering within the priority class, no outcome,
                # learning rate, fit score, route quality or teacher-J sorting.
                state={k:(v.clone() if isinstance(v,torch.Tensor) else copy.deepcopy(v)) for k,v in state.items()}
                selected.append((key,state,found));selected.sort(key=lambda a:a[0]);selected=selected[:32]
            if (index+1)%32==0:print(f'Teacher pool {index+1}/256; verified={statistics["feasible_states"]}; retained={len(selected)}',flush=True)
    states=[r[1] for r in selected];witnesses=[r[2] for r in selected]
    if states:
        independent=witness.verify_selected(states,witnesses,initial)
        with (ledger.path/'SELECTED_TEACHER_STATES.pt').open('xb') as f:
            torch.save(dict(schema='actor-positive-fixed-state-set-v1',states=states,witnesses=witnesses,
                initial_correction=initial.state_dict(),initial_tensor_sha256=tensor_digest(initial.state_dict()),
                diagnostic_only=True,forbidden_as_online_initialization=True),f)
        dataset_sha=sha(ledger.path/'SELECTED_TEACHER_STATES.pt')
    else:independent=dict(status='NO_ELIGIBLE_STATES',checks=0);dataset_sha=None
    put(ledger.path/'WITNESS_INDEPENDENT_RECHECK.json',independent)
    records=[]
    for s,w in zip(states,witnesses):
        records.append(dict(**{k:v for k,v in s.items() if k not in ('x','z0','mask')},
            source_feature_tensor_sha256=tensor_digest({'x':s['x'],'z0_legal':s['z0'][s['mask']],'mask':s['mask']}),
            legal_mask=s['mask'].tolist(),base_legal_logits=s['z0'][s['mask']].tolist(),
            **{k:v for k,v in w.items() if k!='model'}))
    receipt=dict(schema='actual-A1-feasible-rank-witnesses-v1',selection='teacher SL-rank2 priority then fixed state_id SHA256, first32; fixed original256-label pool only',
        selected_count=len(states),minimum_for_interpretable_positive=16,selection_statistics=dict(statistics),
        exact_selected_dataset_sha256=dataset_sha,source_binding_sha256=sha(HOME/'SOURCE_BINDING.json'),
        no_new_teacher_navigation_H_or_route_requests=True,optimizer_attempts_at_selection=ledger.counts['actor_optimizer_attempts'],
        witness_scope='Actual A1 network parameters verified separately per selected state; not a shared memorization-weight proof',witnesses=records)
    put(HOME/'FEASIBLE_RANK_WITNESSES.json',receipt)
    ledger.event('selected_states_frozen_before_first_CE',count=len(states),dataset_sha256=dataset_sha,state_order=[s['state_id'] for s in states])
    return states,initial,dataset_sha

@torch.no_grad()
def fit_metrics(model,states):
    hits=0;ce=0.;span=0.;maximum=0.
    for s in states:
        d=model.details(s['z0'],s['x'],s['mask']);z=d['logits'];other=s['mask'].clone();other[s['target']]=False
        hits+=int(z[s['target']]>z[other].max()+1e-5)
        ce-=float(torch.log_softmax(z,-1)[s['target']])/len(states)
        delta=d['delta'][s['mask']];span=max(span,float(delta.max()-delta.min()));maximum=max(maximum,float(delta.abs().max()))
    return dict(target_top1_count=hits,target_top1_fraction=hits/len(states),CE=ce,max_effective_residual_range=span,max_abs_delta=maximum)

def run(ledger,trace,inputs):
    actor=load_actor('PG-cond');_assert_frozen(actor)
    states,initial,data_sha=select(inputs,actor,ledger,trace)
    results=[];(ledger.path/'models').mkdir()
    for rate in RATES:
        label=f'lr_{rate:.0e}';model=copy.deepcopy(initial);optimizer=torch.optim.Adam(model.parameters(),lr=rate,betas=(.9,.999),eps=1e-8,weight_decay=0.)
        initial_sha=tensor_digest(model.state_dict());accepted=0;rejected=0;consecutive=0;completed=0
        for step in (0,):
            with (ledger.path/'models'/f'{label}_step{step:04d}.diagnostic.pt').open('xb') as f:
                torch.save(dict(model=model.state_dict(),Adam=optimizer.state_dict(),learning_rate=rate,step=step,
                    diagnostic_only=True,forbidden_as_online_initialization=True,data_sha256=data_sha,source_binding_sha256=sha(HOME/'SOURCE_BINDING.json')),f)
        if states:
            with trace.stage('CE_'+label):
                for step in range(1,129):
                    ledger.check();before=copy.deepcopy(model.state_dict());before_opt=copy.deepcopy(optimizer.state_dict())
                    with torch.no_grad():old=[model(s['z0'],s['x'],s['mask']).detach() for s in states]
                    optimizer.zero_grad(set_to_none=True);loss_total=0.
                    for s in states:
                        z=model(s['z0'],s['x'],s['mask']);part=-torch.log_softmax(z,-1)[s['target']]/len(states)
                        if not torch.isfinite(part):raise ValueError('Nonfinite positive CE')
                        part.backward();loss_total+=float(part.detach())
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite positive gradient')
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    ledger.optimizer(label,step);ledger.counts['diagnostic_optimizer_attempts']+=1;optimizer.step();ledger.counts['actor_optimizer_completed']+=1
                    if any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('Nonfinite positive weights')
                    with torch.no_grad():
                        new=[model(s['z0'],s['x'],s['mask']).detach() for s in states]
                        guards=[check_update(s['z0'],o,n,s['mask'],GuardLimits()) for s,o,n in zip(states,old,new)]
                    ok=all(g['accepted'] for g in guards)
                    rollback=restore_if_rejected(model,optimizer,before,before_opt,{'accepted':ok})
                    accepted+=int(ok);rejected+=int(not ok);consecutive=0 if ok else consecutive+1
                    metrics=fit_metrics(model,states);assert metrics['max_abs_delta']<=.25000001
                    record=dict(stage_status='RUN',learning_rate=rate,step=step,selected_states=len(states),
                        accepted=ok,rollback_exact=bool(rollback.get('restored')),accepted_total=accepted,rejected_total=rejected,
                        loss_before_update=loss_total,gradient_norm=float(norm),**metrics,
                        max_step_KL=max(g['maxima']['max_step_kl'] for g in guards),
                        max_probability_change=max(g['maxima']['max_step_probability_change'] for g in guards),
                        max_base_TV=max(g['maxima']['max_base_total_variation'] for g in guards),
                        model_tensor_sha256=tensor_digest(model.state_dict()),wall_s=ledger.elapsed())
                    csvrow(HOME/'UPDATE_SCALE_RESULTS.csv',record)
                    archive(ledger.path/'UPDATE_GUARDS.jsonl.gz',dict(learning_rate=rate,step=step,accepted=ok,guards=guards))
                    completed=step
                    if step%32==0:print(f'{label} step{step}: fit={metrics["target_top1_count"]}/{len(states)}, accept={accepted}/{step}',flush=True)
                    if consecutive>=16:break
        final=fit_metrics(model,states) if states else None
        success=bool(len(states)>=16 and completed==128 and final['target_top1_fraction']>=.8 and accepted>=math.ceil(.9*128))
        results.append(dict(learning_rate=rate,status='PASS' if success else ('INSUFFICIENT_STATES' if len(states)<16 else 'FAIL'),
            success=success,completed_attempts=completed,accepted=accepted,rejected=rejected,initial_tensor_sha256=initial_sha,final=final))
        with (ledger.path/'models'/f'{label}_step{completed:04d}_final.diagnostic.pt').open('xb') as f:
            torch.save(dict(model=model.state_dict(),Adam=optimizer.state_dict(),learning_rate=rate,step=completed,diagnostic_only=True,
                forbidden_as_online_initialization=True,data_sha256=data_sha,source_binding_sha256=sha(HOME/'SOURCE_BINDING.json')),f)
    assert len({r['initial_tensor_sha256'] for r in results})==1
    # Independent endpoint recomputation from actual saved weights, no updates.
    checks=[]
    if states:
        for r in results:
            label=f'lr_{r["learning_rate"]:.0e}';file=ledger.path/'models'/f'{label}_step{r["completed_attempts"]:04d}_final.diagnostic.pt'
            saved=torch.load(file,map_location='cpu',weights_only=False);check=copy.deepcopy(initial);check.load_state_dict(saved['model'])
            measured=fit_metrics(check,states);assert measured==r['final'];checks.append(dict(learning_rate=r['learning_rate'],target_top1_count=measured['target_top1_count'],status='PASS'))
    put(ledger.path/'ENDPOINT_RECHECK.json',dict(status='PASS',checkpoints=checks,optimizer_attempts=0))
    put(HOME/'POSITIVE_CONTROL_SUMMARY.json',dict(status='COMPLETE',selected_states=len(states),configurations=results,
        success_standard='>=16 witnessed states; exact step128 target strict-top1>=80%, accepted>=90%, all applied amplitude/mask/guard/finite protection',
        scope='Known saved training states only, not generalization or route benefit',diagnostic_weights_forbidden_for_online=True))
    if not states:
        csvrow(HOME/'UPDATE_SCALE_RESULTS.csv',dict(stage_status='INSUFFICIENT_STATES',learning_rate='',step='',selected_states=0,
            target_top1_count='',target_top1_fraction='',CE='',accepted='',gradient_norm=''))
    chosen=None
    if len(states)>=16 and not results[0]['success']:
        chosen=next((r['learning_rate'] for r in results[1:] if r['success']),None)
    reason='ORIGINAL_RATE_ALREADY_PASS' if results[0]['success'] else ('INSUFFICIENT_WITNESSED_STATES' if len(states)<16 else ('NO_NEW_RATE_PASS' if chosen is None else 'ORIGINAL_FAIL_MINIMUM_NEW_RATE_PASS'))
    put(HOME/'TRIGGER_DECISION.json',dict(triggered=chosen is not None,selected_learning_rate=chosen,reason=reason,
        chosen_without_any_route_evaluation=True,positive_summary_sha256=sha(HOME/'POSITIVE_CONTROL_SUMMARY.json')))
    _assert_frozen(actor)
    return chosen
