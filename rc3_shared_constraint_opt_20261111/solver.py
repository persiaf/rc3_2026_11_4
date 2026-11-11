"""One fixed dual projected-FISTA QP, with full-constraint separation."""
import time
import numpy as np
import torch
from .common import HOME,archive,csvrow

REG=1e-3
MARGIN=1e-4
GAP_TOL=1e-6
SEPARATION_TOL=1e-6

def project_blocks(v,groups):
    result=np.maximum(v,0.)
    for ids in groups:
        if result[ids].sum()<=1/32:continue
        u=np.sort(v[ids])[::-1];cs=np.cumsum(u)-1/32
        rho=np.nonzero(u>cs/np.arange(1,len(u)+1))[0][-1]
        result[ids]=np.maximum(v[ids]-cs[rho]/(rho+1),0.)
    return result

def slacks(values,owners):
    s=np.zeros(32);np.maximum.at(s,owners,values);return s

def dual_qp(J,b,owners,alpha,ledger):
    if not np.isfinite(J).all() or not np.isfinite(b).all():raise ValueError('Nonfinite local Jacobian or margin')
    G=J@J.T;K=G/REG
    eigen=np.linalg.eigvalsh(G);smax=np.sqrt(max(float(eigen[-1]),0.));threshold=max(1e-10,smax*1e-6)
    positive=eigen>threshold**2;rank=int(positive.sum())
    condition=smax/np.sqrt(float(eigen[positive][0])) if rank else None
    L=max(float(eigen[-1])/REG,1e-20)
    groups=[np.flatnonzero(owners==i) for i in range(32)]
    a=project_blocks(alpha,groups);y=a.copy();t=1.;done=False
    for iteration in range(1,5001):
        if iteration%25==1:ledger.check()
        ledger.counts['dual_projected_iterations']+=1
        nxt=project_blocks(y+(b-K@y)/L,groups)
        tn=(1+np.sqrt(1+4*t*t))/2
        yn=nxt+(t-1)/tn*(nxt-a)
        # Fixed adaptive restart, not a second optimizer/configuration.
        if float((y-nxt)@(nxt-a))>0:yn=nxt.copy();tn=1.
        a,y,t=nxt,yn,tn
        if iteration%25==0 or iteration==5000:
            Ka=K@a;norm2=float(a@Ka)/REG
            s=slacks(b-Ka,owners)
            primal=.5*REG*norm2+float(s.mean());dual=float(b@a)-.5*REG*norm2
            gap=primal-dual
            if gap<=GAP_TOL and gap>=-1e-8:done=True;break
    d=J.T@a/REG
    if not np.isfinite(d).all():raise ValueError('Nonfinite QP parameter direction')
    return dict(status='COMPLETE' if done else 'INCOMPLETE_DUAL_GAP',alpha=a,d=d,slack=s,
        primal=primal,dual=dual,gap=gap,iterations=iteration,G=G,rank=rank,
        singular_max=smax,effective_condition=condition,rank_threshold=threshold,
        parameter_direction_norm=float(np.linalg.norm(d)))

def margin_function(model,states,parameters):
    names=[n for n,_ in model.named_parameters()]
    def fn(*values):
        mapping=dict(zip(names,values));out=[]
        for s in states:
            z=torch.func.functional_call(model,mapping,(s['z0'],s['x'],s['mask']))
            ids=s['mask'].nonzero().flatten();ids=ids[ids!=s['target']]
            out.append(z[s['target']]-z[ids])
        return torch.cat(out)
    return fn

def gradients_for_rows(model,states,layout,indices,ledger):
    params=tuple(model.parameters());rows={}
    for si in range(32):
        selected=[k for k in indices if layout[k][0]==si]
        if not selected:continue
        s=states[si];z=model(s['z0'],s['x'],s['mask'])
        for pos,k in enumerate(selected):
            ledger.check();j=layout[k][1]
            grad=torch.autograd.grad(z[s['target']]-z[j],params,retain_graph=pos+1<len(selected),allow_unused=False)
            rows[k]=torch.cat([g.detach().reshape(-1).double() for g in grad]).numpy()
            ledger.counts['exact_Jacobian_rows']+=1
    return rows

def solve(model,states,outer,ledger):
    start=time.perf_counter();params=tuple(model.parameters())
    layout=[(i,int(j)) for i,s in enumerate(states) for j in s['mask'].nonzero().flatten() if int(j)!=s['target']]
    owners=np.array([a for a,b in layout],dtype=np.int64)
    fn=margin_function(model,states,params)
    with torch.no_grad():m=fn(*params).double().numpy()
    b=MARGIN-m
    active=[int(ids[np.argmax(b[ids])]) for i in range(32) if len(ids:=np.flatnonzero(owners==i))]
    cache=gradients_for_rows(model,states,layout,active,ledger);alpha=np.zeros(len(active));last=None;status='INCOMPLETE_CUT_LIMIT'
    for cutting in range(1,17):
        ledger.check();ledger.counts['cutting_rounds']+=1
        J=np.stack([cache[k] for k in active]);own=owners[active]
        q=dual_qp(J,b[active],own,alpha,ledger);last=q
        pieces=[];offset=0
        for p in params:
            pieces.append(torch.from_numpy(q['d'][offset:offset+p.numel()]).to(dtype=p.dtype).reshape(p.shape));offset+=p.numel()
        _,jv=torch.autograd.functional.jvp(fn,params,tuple(pieces),create_graph=False,strict=False)
        jv=jv.detach().double().numpy();ledger.counts['all_constraint_JVP_checks']+=1
        full_slack=slacks(b-jv,owners);viol=b-jv-q['slack'][owners]
        max_violation=float(max(0.,viol.max()))
        predicted_loss=float(full_slack.mean())
        full_primal=.5*REG*float(q['d']@q['d'])+predicted_loss
        global_gap=full_primal-q['dual']
        fully_complete=q['status']=='COMPLETE' and max_violation<=SEPARATION_TOL and -1e-8<=global_gap<=GAP_TOL
        # Every original parameter contributes to the exact active row gradients;
        # all other legal margins are checked via the exact AD Jacobian product.
        gram=q['G'];norms=np.sqrt(np.maximum(np.diag(gram[:32,:32]),0.));pair=gram[:32,:32]
        tri=np.triu_indices(len(norms),1);denom=np.outer(norms,norms)
        valid=denom[tri]>1e-20;cos=pair[tri][valid]/denom[tri][valid]
        log=dict(outer=outer,cutting_round=cutting,restricted_QP_status=q['status'],
            full_subproblem_status='COMPLETE' if fully_complete else 'INCOMPLETE_NEEDS_CUT_OR_SOLVE',active_constraints=len(active),
            all_legal_alternative_constraints=len(layout),all_legal_candidates=len(layout)+32,
            dual_iterations=q['iterations'],primal_objective=full_primal,restricted_primal_objective=q['primal'],dual_objective=q['dual'],duality_gap=global_gap,
            max_all_constraint_violation=max_violation,predicted_hinge=predicted_loss,old_hinge=float(slacks(b,owners).mean()),
            predicted_hinge_decrease=float(slacks(b,owners).mean())-predicted_loss,direction_norm=q['parameter_direction_norm'],
            active_Jacobian_effective_rank=q['rank'],active_Jacobian_rows=len(active),Jacobian_columns=sum(p.numel() for p in params),
            singular_max=q['singular_max'],effective_condition=q['effective_condition'],rank_threshold=q['rank_threshold'],
            rank_scope='Exact ACTIVE constraint rows, not an estimate of full all-legal Jacobian rank',
            initial_worst_state_gradient_negative_pairs=int((cos< -1e-12).sum()),initial_worst_state_gradient_valid_pairs=len(cos),
            compute_wall_s=time.perf_counter()-start)
        csvrow(HOME/'LOCAL_SUBPROBLEM_LOG.csv',log)
        archive(ledger.path/'SUBPROBLEM_CERTIFICATES.jsonl.gz',dict(**log,active_indices=active,
            active_layout=[layout[k] for k in active],alpha=q['alpha'].tolist(),Gram=q['G'].tolist(),b_active=b[active].tolist(),
            slack_per_state=q['slack'].tolist(),all_linearized_margins=(m+jv).tolist(),all_constraint_layout=layout,
            full_slack_per_state=full_slack.tolist(),direction=q['d'].tolist()))
        if q['status']!='COMPLETE':status=q['status'];break
        if fully_complete:
            status='COMPLETE';break
        new=[]
        for i in range(32):
            ids=np.flatnonzero(owners==i);k=int(ids[np.argmax(viol[ids])])
            if viol[k]>SEPARATION_TOL and k not in cache:new.append(k)
        if not new:status='INCOMPLETE_NUMERICAL_SEPARATION';break
        if cutting==16:break
        cache.update(gradients_for_rows(model,states,layout,new,ledger));active.extend(new)
        alpha=np.concatenate([q['alpha'],np.zeros(len(new))])
    return dict(status=status,direction=last['d'],predicted_hinge=predicted_loss,old_hinge=float(slacks(b,owners).mean()),
        predicted_decrease=float(slacks(b,owners).mean())-predicted_loss,linearized_margins=m+jv,
        layout=layout,owners=owners,b=b,jv=jv,compute_wall_s=time.perf_counter()-start,
        cutting_rounds=cutting,duality_gap=global_gap,max_all_constraint_violation=max_violation)
