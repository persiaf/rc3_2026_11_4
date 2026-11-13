"""Registered R08-B menu; no prospective objective/H queries."""
from itertools import combinations,zip_longest
import hashlib,json,time
import numpy as np
from rc3_r08_joint_decision_design_20261113.contracts import JointAction,action_legality

def build_menu(state,assignments):
    tick=time.perf_counter();p=state.snapshot;order=state.incumbent.visit_ids;index={j:k for k,j in enumerate(p.candidate_ids)};cover=p.cover
    adjacent={tuple(sorted(x)) for x in zip(order,order[1:])};non=[]
    for a,b in combinations(sorted(order),2):
        if (a,b) not in adjacent and bool((cover[:,index[a]]&cover[:,index[b]]).any()):non.append((a,b))
    blocks=[]
    for a,b in zip_longest(sorted(adjacent),sorted(non)):
        for pair in (a,b):
            if pair is not None and len(blocks)<8:blocks.append(pair)
        if len(blocks)==8:break
    actions=[];seen=set();trace=[];selected=set(order);task_index={i:k for k,i in enumerate(p.task_ids)}
    for block in blocks:
        retained=selected-set(block);cols=[index[j] for j in retained];count=cover[:,cols].sum(1) if cols else np.zeros(len(p.task_ids),int)
        U=np.flatnonzero(count==0);served={a['task_id'] for a in assignments if a['candidate_id'] in block};related=sorted(set(U)|{task_index[i] for i in served})
        eligible=[j for j in p.candidate_ids if j not in retained and related and cover[related,index[j]].any()]
        def key(j):return (-int(cover[U,index[j]].sum()),min(float(p.D[index[j]+1,index[b]+1]) for b in block),j)
        pool=sorted(eligible,key=key)[:8]
        replacements=[()] if not len(U) else [*combinations(pool,1),*combinations(pool,2)]
        local={};reasons={}
        for add in replacements:
            if len(U) and (not (set(add)-selected) or not cover[np.ix_(U,[index[j] for j in add])].any(1).all()):continue
            common=set(block)&set(add);drop=tuple(sorted(set(block)-common));new=tuple(sorted(set(add)-common))
            if not drop and not new:continue
            action=JointAction(drop=drop,add=new);valid,why=action_legality(state,action)
            if not valid:reasons[why]=reasons.get(why,0)+1;continue
            signature=(drop,new);local[signature]=action
        ranked=sorted(local,key=lambda sig:(len(sig[1]),sig[1],sig[0]))[:2]
        appended=[]
        for sig in ranked:
            if sig in seen:continue
            seen.add(sig);a=local[sig];record=dict(menu_index=len(actions)+1,drop=list(a.drop),add=list(a.add),
                net_joint=len(a.drop)==2 or len(a.add)==2,block=list(block),block_kind='adjacent' if block in adjacent else 'coverage_overlap_nonadjacent',
                uncovered_task_ids=[p.task_ids[i] for i in U],pool_global_ids=pool,canonical_signature=[list(a.drop),list(a.add)],legal=True)
            actions.append(record);appended.append(record['menu_index'])
        trace.append(dict(block=list(block),U=[p.task_ids[i] for i in U],assigned_tasks=sorted(served),pool=pool,
            legal_net_candidates=len(local),first_two_before_global_dedup=[[list(d),list(a)] for d,a in ranked],appended_indices=appended,rejection_reasons=reasons))
    if len(actions)>16:raise AssertionError('Registered menu cap')
    body=dict(actions=actions,blocks=trace,stop_legal=True,missing_slots=16-len(actions),
        operationalization='Adjacent-first alternating sorted block classes; U-empty deletion only; net-add count then add IDs then drop IDs; per-block first2 before global dedup; duplicates not backfilled')
    body['menu_sha256']=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest();body['menu_build_s']=time.perf_counter()-tick
    return body
