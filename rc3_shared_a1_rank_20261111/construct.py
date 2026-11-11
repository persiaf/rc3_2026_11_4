import copy
import torch
import numpy as np
from rc3_r04_review_20261111.relative_actor import residual_details
from rc3_conservative_refinement.guard import check_update,GuardLimits
from rc3_conservative_refinement.runtime import tensor_digest
from .common import HOME,put,archive,csvrow

@torch.no_grad()
def inspect(model,states,old=None):
    records=[]
    for i,s in enumerate(states):
        d=model.details(s['z0'],s['x'],s['mask']);z=d['logits'];mask=s['mask'];t=s['target']
        other=mask.clone();other[t]=False
        g=check_update(s['z0'],s['z0'] if old is None else old[i],z,mask,GuardLimits())
        delta=d['delta'][mask]
        records.append(dict(state_index=i,state_id=s['state_id'],target=t,
            margin=float(z[t]-z[other].max()),target_top1=int(z.argmax())==t,
            residual_range=float(delta.max()-delta.min()),max_abs_delta=float(delta.abs().max()),
            probability_change=g['maxima']['max_step_probability_change'],guard=g,
            guard_reference='base_to_model' if old is None else 'previous_applied_to_candidate',
            mask_preserved=bool(torch.isneginf(z[~mask]).all()),
            finite=bool(torch.isfinite(z[mask]).all() and torch.isfinite(d['raw_scores']).all()),
            legal_logits=z[mask].tolist(),raw_legal_scores=d['raw_scores'][mask].tolist(),
            legal_delta=delta.tolist()))
    return records

def metrics(records):
    return dict(strict_target_top1_count=sum(r['margin']>1e-5 for r in records),
        minimum_margin=min(r['margin'] for r in records),
        max_residual_range=max(r['residual_range'] for r in records),
        max_abs_delta=max(r['max_abs_delta'] for r in records),
        max_step_KL=max(r['guard']['maxima']['max_step_kl'] for r in records),
        max_probability_change=max(r['probability_change'] for r in records),
        max_base_TV=max(r['guard']['maxima']['max_base_total_variation'] for r in records))

@torch.no_grad()
def independent_probability_audit(records,states,margin_bound):
    checks=0
    for r,s in zip(records,states):
        z0=s['z0'][s['mask']].double().numpy();z=np.array(r['legal_logits']);raw=np.array(r['raw_legal_scores'])
        ls=lambda v:v-v.max()-np.log(np.exp(v-v.max()).sum())
        lp,lq=ls(z0),ls(z);p,q=np.exp(lp),np.exp(lq)
        delta=.25*np.tanh(raw-raw.mean())
        assert np.allclose(delta,r['legal_delta'],rtol=1e-6,atol=1e-7)
        assert np.sum(p*(lp-lq))<=.001 and abs(p-q).max()<=.02 and .5*abs(p-q).sum()<=.1
        assert r['margin']>=margin_bound and r['mask_preserved'] and r['finite'] and max(abs(delta))<=.25
        checks+=7
    return dict(status='PASS',states=32,checks=checks,scope='Independent NumPy full-legal probability and centered-tanh arithmetic on actual single shared-network float32 outputs; not an independent GEMM implementation.')

@torch.no_grad()
def construct(states,initial,ledger,trace):
    # Each teacher feature defines one bump. Exact identical teacher rows merge;
    # duplicates elsewhere remain in every actual joint network check.
    groups=[]
    for i,s in enumerate(states):
        row=s['x'][s['target']]
        found=next((g for g in groups if torch.equal(g['target'],row)),None)
        if found is None:groups.append(dict(target=row.clone(),states=[i]))
        else:found['states'].append(i)
    assert len(groups)<=32
    X=torch.cat([s['x'][s['mask']] for s in states]).double().numpy()
    rng=np.random.Generator(np.random.PCG64(11622026))
    dense1=rng.choice([-1.,1.],size=(128,206));dense1/=np.sqrt(206)
    dense2=rng.standard_normal((128,206));dense2/=np.linalg.norm(dense2,axis=1)[:,None]
    projections=np.concatenate([np.eye(206),dense1,dense2]).astype(np.float32)
    Y=X@projections.T.astype(np.float64)
    error_scale=1+np.abs(X)@np.abs(projections.T.astype(np.float64))
    duplicate_records=[];near_records=[]
    for gi,g in enumerate(groups):
        ledger.check();t=g['target'].double().numpy();same=np.all(X==t,axis=1)
        center=t@projections.T.astype(np.float64)
        diff=np.abs(Y-center)
        diff[same,:]=np.inf
        separation=diff.min(axis=0)
        normalized=separation/error_scale.max(axis=0)
        order=np.argsort(-normalized,kind='stable')
        g['choices']=[dict(projection_index=int(k),center=float(np.float32(center[k])),
            half_width=float(separation[k]/3),normalized_separation=float(normalized[k]))
            for k in order[:8] if normalized[k]>=2e-6]
        # Isolated spike scale by frozen float32 residual arithmetic, no Adam.
        scales=[]
        for si in g['states']:
            s=states[si];raw=torch.zeros_like(s['z0']);raw[s['target']]=1
            lo,hi=0.,64.*int(s['mask'].sum())
            other=s['mask'].clone();other[s['target']]=False
            for _ in range(48):
                mid=(lo+hi)/2;z=s['z0']+residual_details(raw*mid,s['mask'])['delta']
                if float(z[s['target']]-z[other].max())>=2e-4:hi=mid
                else:lo=mid
            scales.append(hi)
        g['scale']=max(scales)
        duplicate_records.append(dict(group=gi,target_state_indices=g['states'],exact_target_feature_occurrences=int(same.sum())))
        near_records.append(dict(group=gi,minimum_other_feature_Linf=float(np.max(abs(X[~same]-t),axis=1).min()) if (~same).any() else None,
            viable_projection_choices=len(g['choices']),best_normalized_separation=float(normalized[order[0]])))
    put(HOME/'FEATURE_SEPARATION.json',dict(projection_count=462,fixed_seed=11622026,groups=len(groups),
        exact_duplicates=duplicate_records,nearest_distinct_features=near_records,
        tolerance='minimum projection gap / max(1+abs(X)@abs(projection)) >=2e-6; half-width=gap/3',
        no_state_ID_or_query_table_in_network=True))
    attempts=[]
    for attempt in range(8):
        ledger.check()
        if any(not g['choices'] for g in groups):break
        model=copy.deepcopy(initial)
        for p in model.parameters():p.zero_()
        choices=[]
        for j,g in enumerate(groups):
            c=g['choices'][min(attempt,len(g['choices'])-1)];p=torch.from_numpy(projections[c['projection_index']]);eps=c['half_width']
            model.head[0].weight[2*j]=p;model.head[0].bias[2*j]=-c['center']
            model.head[0].weight[2*j+1]=-p;model.head[0].bias[2*j+1]=c['center']
            model.head[2].weight[j,2*j]=-1/eps;model.head[2].weight[j,2*j+1]=-1/eps;model.head[2].bias[j]=1
            model.head[4].weight[0,j]=g['scale'];choices.append(dict(group=j,**c,output_scale=g['scale']))
        # Normalize the final coefficient by the actual float32 target bump
        # peak. No gradients/optimizer; still one fixed shared parameter set.
        peaks=[model.head[:4](s['x'].masked_fill(~s['mask'][:,None],0.)) for s in states]
        for j,g in enumerate(groups):
            peak=min(float(peaks[si][states[si]['target'],j]) for si in g['states'])
            if peak>0:model.head[4].weight[0,j]=g['scale']/peak
            choices[j].update(actual_target_peak_min=peak,output_scale=float(model.head[4].weight[0,j]))
        ledger.counts['shared_construction_attempts']+=1
        records=inspect(model,states)
        ok=all(r['margin']>=1e-4 and r['guard']['accepted'] and r['mask_preserved'] and r['finite'] and r['max_abs_delta']<=.25000001 for r in records)
        receipt=dict(attempt=attempt+1,parameter_sha256=tensor_digest(model.state_dict()),passed=ok,**metrics(records),construction=choices)
        with (ledger.path/f'SHARED_attempt{attempt+1:02d}.diagnostic.pt').open('xb') as f:
            torch.save(dict(model=model.state_dict(),parameter_sha256=receipt['parameter_sha256'],diagnostic_only=True,
                forbidden_as_formal_initialization=True,forbidden_for_deployment=True),f)
        attempts.append(receipt);archive(ledger.path/'SHARED_CONSTRUCTION_STATES.jsonl.gz',dict(**receipt,states=records))
        if not ok:
            for r in records:
                failures=[(k,r['guard']['maxima'][k],b) for k,b in [('max_step_kl',.001),('max_step_probability_change',.02),('max_base_total_variation',.1)] if r['guard']['maxima'][k]>b]
                if r['margin']<1e-4:failures.append(('shared_strict_margin',r['margin'],1e-4))
                for k,v,b in failures:
                    csvrow(HOME/'FAILURE_STATE_ANALYSIS.csv',dict(stage='shared_construction',step=attempt+1,state_index=r['state_index'],state_id=r['state_id'],constraint=k,value=v,bound=b,candidate_parameter_sha256=receipt['parameter_sha256'],repeat_of_step=''))
        ledger.event('shared_construction_candidate',attempt=attempt+1,passed=ok,parameter_sha256=receipt['parameter_sha256'])
        if ok:
            audit=independent_probability_audit(records,states,1e-4)
            with (ledger.path/'SHARED_WITNESS.diagnostic.pt').open('xb') as f:
                torch.save(dict(model=model.state_dict(),parameter_sha256=receipt['parameter_sha256'],diagnostic_only=True,
                    forbidden_as_formal_initialization=True,forbidden_for_deployment=True),f)
            put(HOME/'JOINT_FEASIBILITY.json',dict(status='PASS',shared_parameter_set=True,states=32,bumps=len(groups),first_hidden_used=2*len(groups),
                second_hidden_used=len(groups),all_32_checked_simultaneously=True,actual_float32_network=True,architecture=[206,64,64,1],
                witness_margin=1e-4,attempts=attempts,independent_probability_audit=audit,per_state=records,
                prior_witness_scope='Old witnesses were one separate parameter set per state, not one shared32-state model.',diagnostic_only=True))
            return True
    put(HOME/'JOINT_FEASIBILITY.json',dict(status='CONSTRUCTION_NOT_FOUND',shared_parameter_set=False,states=32,attempts=attempts,
        reason='Insufficient stable separating projections or actual joint float32 margin/guard failure within fixed8 attempts',
        impossibility_proof=False,feature_separation='FEATURE_SEPARATION.json',diagnostic_only=True))
    return False
