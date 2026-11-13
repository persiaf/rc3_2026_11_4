"""Causal, unnormalised cost decomposition on saved source-bound distances."""
import math
import numpy as np

ATOL=1e-7
def decompose(p,r):
    actions=r['actions'];nodes=[a+1 for a in actions];raw=[0,*nodes,0]
    if raw!=r['raw_route'] or p.identity!=r['problem_identity']:raise ValueError('Raw actions/input identity')
    n=len(p.C);remaining=np.ones(n,dtype=bool);current=0;prefix_L=0.;prefix=[];inc=[];counts=[]
    for t,node in enumerate(nodes):
        prefix.append(prefix_L+p.penalty*t);counts.append(int(remaining.sum()))
        distance=float(p.D[current,node])
        if not math.isfinite(distance) or not np.isfinite(p.D[node,0]) or not (remaining&p.C[:,node-1]).any():raise ValueError('Nonfinite/zero-gain action')
        inc.append(distance+p.penalty);prefix_L+=distance;remaining &= ~p.C[:,node-1];current=node
    if remaining.any() or not nodes or n<=0:raise ValueError('Incomplete/empty registered episode')
    legs=[float(p.D[a,b]) for a,b in zip(raw,raw[1:])]
    savedlegs=r['raw_plan']['route_leg_costs_m']
    if len(legs)!=len(savedlegs) or max(abs(a-b) for a,b in zip(legs,savedlegs))>ATOL:raise ValueError('Raw saved navigation legs differ from D')
    raw_J=sum(legs)+p.penalty*len(nodes);J=float(r['metrics']['J']);raw_m=p.metrics(raw);H_m=p.metrics(r['route'])
    errors=[abs(raw_J-r['metrics']['raw_J'])]
    errors += [abs(raw_m[k]-r['metrics']['raw_'+k]) for k in ('J','L','K')]
    errors += [abs(H_m[k]-r['metrics'][k]) for k in ('J','L','K')]
    inc[-1]+=legs[-1]
    raw_sum=math.fsum(inc);correction=J-raw_J;after=list(inc);after[-1]+=correction
    errors += [abs(raw_sum-raw_J),abs(math.fsum(after)-J),abs(math.fsum(r['rewards'])+J)]
    if max(errors)>ATOL:raise ValueError('Complete raw/H cost decomposition failed: '+str(max(errors)))
    q=[k/n for k in counts];tail=[J-x for x in prefix]
    if prefix[0]!=0 or q[0]!=1:raise ValueError('First-action causal baseline contract')
    return dict(raw_prefix_cost_before=prefix,raw_step_cost_with_terminal_exit=inc,
        H_step_cost_with_terminal_correction=after,remaining_task_count_before=counts,initial_task_count=n,
        q_before=q,tail_cost=tail,H_cost_correction=correction,max_abs_decomposition_error=max(errors),
        units='Unnormalised J=L+0.75K; q is the sole division by initial task count',terminal_exit_cost=legs[-1])

def coefficients(credit,J,reference,algorithm):
    if algorithm=='A':return [J-reference]*len(credit['q_before'])
    if algorithm!='B':raise ValueError('Exact estimator A or B required')
    out=[tail-q*reference for tail,q in zip(credit['tail_cost'],credit['q_before'])]
    if abs(out[0]-(J-reference))>ATOL:raise ValueError('First action A/B coefficient differs')
    return out
