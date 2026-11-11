"""Actual 206-64-64-1 A1 network witnesses, not a gap-only eligibility rule."""
import copy
import math
import torch
from rc3_r04_review_20261111.relative_actor import RelativeCorrection,residual_details
from rc3_conservative_refinement.guard import check_update,GuardLimits
from rc3_conservative_refinement.runtime import tensor_digest

MARGIN=1e-4

def strict_margin(z,mask,target):
    others=mask.clone();others[target]=False
    return float(z[target]-z[others].max())

@torch.no_grad()
def construct(state,initial,ledger):
    x,z,mask,target=state['x'],state['z0'],state['mask'],state['target']
    legal=mask.nonzero(as_tuple=False).flatten();m=len(legal)
    if m<2 or not bool(mask[target]):return None,'illegal_or_single'
    gap=float(z[legal].max()-z[target])
    if gap+2*MARGIN>=.5:return None,'necessary_amplitude_bound'
    others=legal[legal!=target]
    separation=(x[others]-x[target]).abs().amin(0)
    safe=separation/(1+x[target].abs())
    dim=int(safe.argmax());eps=float(separation[dim])/3
    if float(safe[dim])<3e-5:return None,'no_stable_single_coordinate_bump'
    # Two first-layer hinges and one second-layer triangular bump. All other
    # units are zero; this is the SAME A1 network class/width/dtype/amplitude.
    model=copy.deepcopy(initial)
    for p in model.parameters():p.zero_()
    center=float(x[target,dim])
    model.head[0].weight[0,dim]=1;model.head[0].bias[0]=-center
    model.head[0].weight[1,dim]=-1;model.head[0].bias[1]=center
    model.head[2].weight[0,0]=-1/eps;model.head[2].weight[0,1]=-1/eps;model.head[2].bias[0]=1
    model.head[4].weight[0,0]=1
    bump=model.head(x.masked_fill(~mask[:,None],0.)).squeeze(-1)
    centered=bump[legal]-bump[legal].double().mean()
    target_pos=int((legal==target).nonzero()[0])
    if not (float(centered[target_pos])>0 and float(centered[torch.arange(m)!=target_pos].max())<0):return None,'unstable_bump_centering'
    # Search only a scalar of the explicit construction, never Adam/CE fitting.
    low,high=0.,64.*m
    def scores(scale):
        delta=residual_details(bump*scale,mask)['delta']
        return (z.masked_fill(~mask,0.)+delta).masked_fill(~mask,-torch.inf)
    if strict_margin(scores(high),mask,target)<MARGIN:return None,'constructed_bump_cannot_rank'
    for _ in range(48):
        ledger.check();mid=(low+high)/2
        if strict_margin(scores(mid),mask,target)>=2*MARGIN:high=mid
        else:low=mid
    model.head[4].weight[0,0]=high
    ledger.counts['witness_parameterizations']+=1
    actual=model.details(z,x,mask);new=actual['logits'];guard=check_update(z,z,new,mask,GuardLimits())
    margin=strict_margin(new,mask,target)
    valid=bool(new.argmax()==target and margin>=MARGIN and guard['accepted'] and float(actual['delta'][mask].abs().max())<=.25000001)
    if not valid:return None,'full_network_guard_or_rank_reject'
    ledger.counts['witness_verified']+=1
    return dict(model=copy.deepcopy(model.state_dict()),construction=dict(type='actual_A1_two_hinge_one_bump',feature_dimension=dim,
        feature_center=center,half_width=eps,output_scale=high),tensor_sha256=tensor_digest(model.state_dict()),
        strict_teacher_margin=margin,guard=guard,delta=actual['delta'].tolist(),raw_scores=actual['raw_scores'].tolist(),
        original_gap=gap,optimizer_attempts=0),None

@torch.no_grad()
def verify_selected(states,witnesses,initial):
    """Independent NumPy probability arithmetic on actual witness forwards."""
    import numpy as np
    checks=0
    for s,w in zip(states,witnesses):
        model=copy.deepcopy(initial);model.load_state_dict(w['model'])
        out=model.details(s['z0'],s['x'],s['mask']);ids=s['mask'].nonzero().flatten()
        base=s['z0'][ids].double().numpy();new=out['logits'][ids].double().numpy()
        logsoft=lambda v:v-v.max()-np.log(np.exp(v-v.max()).sum())
        lp,lq=logsoft(base),logsoft(new);p,q=np.exp(lp),np.exp(lq)
        assert int(out['logits'].argmax())==s['target'] and strict_margin(out['logits'],s['mask'],s['target'])>=MARGIN
        assert np.sum(p*(lp-lq))<=.001+1e-12 and np.max(np.abs(p-q))<=.02+1e-12 and .5*np.abs(p-q).sum()<=.1+1e-12
        raw=out['raw_scores'][ids].double().numpy();delta=.25*np.tanh(raw-raw.mean())
        assert np.max(np.abs(delta))<=.25 and np.allclose(delta,out['delta'][ids].numpy(),atol=1e-7,rtol=1e-6)
        checks+=7
    return dict(status='PASS',witnesses=len(states),checks=checks,scope='actual A1 parameter witnesses plus independent full-legal probability arithmetic',optimizer_attempts=0,
        joint_model_claim='Per-state existence witnesses; not a claim that one set of witness weights simultaneously fits every selected state. Actual CE fitting is tested jointly.')
