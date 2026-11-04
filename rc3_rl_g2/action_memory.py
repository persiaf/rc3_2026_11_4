"""Shared deterministic proposal identity and pre-cap no-op/failure filtering."""
from __future__ import annotations
import math
from dataclasses import asdict
import numpy as np
from rc3_rl_research.types import Action, ActionMenu, SearchConfig
from rc3_rl_research.action_builder import (FEATURE_NAMES, local_context, stable_digest,
                                           validate_caps, _problem_identity)
from rc3_improve.repair import route_is_valid


def proposal_key(problem_id, route, action, config):
    """Solver identity deliberately excludes budget, best and replay history."""
    return stable_digest(dict(problem_id=problem_id,route=list(route),kind=action.kind,
        start=action.start,end=action.end,forbidden_node=action.forbidden_node,
        max_tasks=config.max_tasks,max_candidates=config.max_candidates,
        primitive='rc3-g01-dp-v1'))


def is_blocked(record):
    return bool(record.get('no_change',False) or record.get('failed',False))


def build_actions(problem,route,*,remaining_work,steps_remaining,history=(),
                  config=SearchConfig(),problem_id='',attempt_memory=None):
    """Same cheap feature/work semantics as G01, with memory filtering pre-cap.

    No expensive proposal is queried here. All scanned contexts remain charged,
    including known blocked contexts. Memory also filters certified over-bound
    proposals: the same-route acceptance bound can only tighten as best falls.
    """
    attempt_memory={} if attempt_memory is None else attempt_memory
    validate_caps(config)
    if not route_is_valid(problem,route):raise ValueError('invalid incumbent route')
    if type(remaining_work) is not int or type(steps_remaining) is not int:
        raise ValueError('work and steps must be integers')
    route=list(map(int,route));history=tuple(history)
    cover=problem.F.any(2);nodes=np.asarray(route,dtype=np.int64)
    counts=cover[:,nodes[1:-1]-1].sum(1);costs=problem.nav['costs']
    contexts_evaluated=candidate_checks=unsupported=0
    groups={'improve':[],'replace':[]}

    def evaluate(kind,start,end,forbidden=None):
        nonlocal contexts_evaluated,candidate_checks,unsupported
        context=local_context(problem,route,start,end,forbidden,cover=cover,counts=counts)
        contexts_evaluated+=1;candidate_checks+=context['candidate_checks']
        required,pool=context['required'],context['pool']
        supported=len(required)<=config.max_tasks and len(pool)<=config.max_candidates
        if len(required):supported=supported and bool(len(pool)) and bool(cover[np.ix_(required,pool-1)].any(1).all())
        if not supported:unsupported+=1;return None
        local_nodes=nodes[start-1:end+1]
        travel=float(costs[local_nodes[:-1],local_nodes[1:]].sum())
        direct=float(costs[local_nodes[0],local_nodes[-1]])
        detour=travel-direct if math.isfinite(direct) else 0.
        action=Action(kind,start,end,forbidden,f'{kind}:{start}:{end}:{forbidden if forbidden is not None else "-"}')
        return dict(action=action,context=context,travel=travel,
                    direct=direct if math.isfinite(direct) else 0.,detour=detour)

    if steps_remaining>0 and remaining_work>0:
        for start in range(1,len(route)-1):
            for width in range(min(config.max_width,len(route)-1-start),0,-1):
                row=evaluate('improve',start,start+width)
                if row is not None:groups['improve'].append(row);break
            row=evaluate('replace',start,start+1,route[start])
            if row is not None:groups['replace'].append(row)
    for rows in groups.values():rows.sort(key=lambda row:(-row['detour'],row['action'].start,row['action'].end))
    ordered=[]
    for i in range(max(map(len,groups.values()),default=0)):
        for kind in ('improve','replace'):
            if i<len(groups[kind]):ordered.append(groups[kind][i])
    count_before=len(ordered)
    ordered=[row for row in ordered if not is_blocked(attempt_memory.get(proposal_key(problem_id,route,row['action'],config),{}))]
    memory_filtered=count_before-len(ordered);available_before_cap=len(ordered)
    ordered=ordered[:config.max_actions]
    work_units=contexts_evaluated+candidate_checks
    context_key=stable_digest(dict(schema='rc3-g2-menu-v2',problem_id=_problem_identity(problem,problem_id),
        route=route,remaining_work=remaining_work,steps_remaining=steps_remaining,
        history=history,config=asdict(config),attempt_memory=attempt_memory))
    common=[math.log1p(len(route)-2),math.log1p(cover.shape[0]),math.log1p(cover.shape[1]),
            float(counts.mean()) if len(counts) else 0.,float(counts.max()) if len(counts) else 0.]
    dynamic=[max(remaining_work,0)/max(config.work_budget,1),
             max(steps_remaining,0)/max(config.horizon,1),float(len(history))]
    actions=[];features=[];details=[]
    for row in ordered:
        action,context=row['action'],row['context']
        removed_covered=cover[:,context['removed']-1].any(1)
        overlap=removed_covered & (context['retained_counts']>0)
        extra=cover[:,context['pool']-1].any(1) & (context['retained_counts']>0)
        actions.append(action)
        features.append([float(action.kind=='improve'),float(action.kind=='replace'),0.,
            (action.start-1)/max(len(route)-2,1),float(action.end-action.start),
            float(len(context['required'])),float(len(context['pool'])),
            row['travel']/config.physical_scale,row['direct']/config.physical_scale,
            row['detour']/config.physical_scale,row['detour']/max(row['travel'],1e-8),
            *common,float(removed_covered.sum()),float(overlap.sum()/max(removed_covered.sum(),1)),
            float(extra.sum()),*dynamic])
        details.append(dict(required_tasks=context['required'].tolist(),pool_nodes=context['pool'].tolist(),
            retained_coverage_counts=context['retained_counts'].tolist(),retained_nodes=context['retained'].tolist()))
    actions.append(Action('stop',key='stop'))
    features.append([0.,0.,1.,*([0.]*8),*common,0.,0.,0.,*dynamic]);details.append({})
    can_search=steps_remaining>0 and remaining_work>work_units
    mask=[can_search]*(len(actions)-1)+[True]
    if not np.isfinite(np.asarray(features)).all():raise ValueError('nonfinite menu features')
    keys=[stable_digest(dict(context=context_key,action=a.to_dict())) for a in actions]
    return ActionMenu(actions,mask,features,list(FEATURE_NAMES),context_key,keys,int(work_units),diagnostics=dict(
        contexts_evaluated=contexts_evaluated,candidate_checks=candidate_checks,
        unsupported_contexts=unsupported,supported_before_cap=count_before,
        memory_filtered=memory_filtered,omitted_by_menu_cap=max(available_before_cap-len(ordered),0),
        ordered_nonstop_count=len(ordered),route=route,coverage_counts=counts.tolist(),action_details=details,
        cache_enabled=False,attempt_memory_enabled=True,ordering='detour_desc_interleave_improve_replace'))
