"""Unchanged H source on explicit endpoint costs; complete plan export and accounting."""
import copy
import time
import numpy as np
from rc3_baseline.baseline import expand_route
from rc3_constructive import baselines
from rc3_fast import solver
from .validation import validate


def export(problem,route,method):
    metrics=problem.metrics(route);p=problem.parent;scene=p.scene
    global_nodes=problem.global_route(route)
    points,legs=expand_route(global_nodes,p.base.nav)
    visited=[int(problem.candidate_ids[n-1]) for n in route[1:-1]]
    left=set(problem.task_ids.tolist());assignments=[];stops=[];arrival=0.
    for k,j in enumerate(visited):
        served=[i for i in sorted(left) if p.effective[i,j].any()]
        if not served:raise ValueError('Export refuses zero-gain or duplicate service')
        for i in served:
            arm='L' if p.effective[i,j,0] else 'R'
            assignments.append(dict(task_id=i,candidate_id=j,arm=arm))
        left-=set(served);arrival+=legs[k]
        stops.append(dict(sequence=k+1,candidate_id=j,pose={key:scene['candidates'][j][key] for key in ('x','y','theta','h')},
            arrival_distance_m=arrival,tasks=[dict(task_id=a['task_id'],arm=a['arm']) for a in assignments if a['candidate_id']==j]))
    plan=dict(schema='rc3-global-local-plan-v1',method=method,status='complete',problem_identity=problem.identity,parent_identity=p.identity,
        entry_global_node=problem.entry,exit_global_node=problem.exit,window=problem.window,
        target_task_ids_global=problem.task_ids.tolist(),visit_order_candidate_ids=visited,selected_candidate_ids=visited,
        navigation_global_nodes=global_nodes,assignments=assignments,service_stops=stops,unserved_task_ids=sorted(left),
        route_polyline=points,route_leg_costs_m=legs,route_length_m=metrics['L'],
        objective_length_plus_stops_m=metrics['J'],stop_penalty_equivalent_m=.75,
        arm_joint_trajectories=None,physical_joint_trajectory_certified=False)
    return plan


def finish(problem,raw,stats=None):
    raw_m=problem.metrics(raw);tick=time.perf_counter()
    if raw[1:-1]:
        if stats is not None:stats['full_H_calls']+=1
        h0=baselines.postprocess(problem,raw)
        route=baselines.normalize(problem,solver.joint_search(list(h0),problem.C,problem.D,passes=2))
    else:route=[0,0]
    h_s=time.perf_counter()-tick
    final=problem.metrics(route)
    if final['J']>raw_m['J']+1e-7:raise ValueError('Frozen H worsened terminal objective')
    tick=time.perf_counter()
    if stats is not None:stats['export_attempts']+=1
    plan=export(problem,route,'S0_endpoint_construction')
    if stats is not None:stats['exports']+=1
    export_s=time.perf_counter()-tick;tick=time.perf_counter()
    if stats is not None:stats['validation_attempts']+=1
    certificate=validate(problem,plan)
    if not certificate['valid_model_plan']:raise ValueError(certificate['errors'])
    if stats is not None:stats['validation_completed']+=1
    plan['validation']=certificate
    return dict(**final,raw_J=raw_m['J'],raw_L=raw_m['L'],raw_K=raw_m['K'],route=route,raw_route=list(raw),
        plan=plan,H_s=h_s,export_s=export_s,validation_s=time.perf_counter()-tick)


class TerminalCache:
    def __init__(self,stats,record,h_identity,limit=80,deadline=None):
        self.stats=stats;self.record=record;self.limit=limit;self.cache={};self.deadline=deadline;self.h_identity=h_identity

    def __call__(self,p,raw):
        self.stats['terminal_requests']+=1
        key=(p.identity,self.h_identity,tuple(raw))
        if key in self.cache:
            self.stats['terminal_cache_hits']+=1
            self.record(dict(event='terminal_cache_hit',problem=p.identity,raw_route=list(raw)))
            result=copy.deepcopy(self.cache[key]);result.update(cache_hit=True,H_s=0.,export_s=0.,validation_s=0.)
            return result
        if self.deadline is not None and time.perf_counter()>=self.deadline:raise TimeoutError('S0 deadline before next terminal')
        if self.stats['terminal_attempts']>=self.limit:raise RuntimeError('Frozen S0 terminal budget reached')
        self.stats['terminal_attempts']+=1;call=self.stats['terminal_attempts']
        self.record(dict(event='terminal_start',call=call,problem=p.identity,raw_route=list(raw)))
        tick=time.perf_counter()
        try:
            result=finish(p,raw,self.stats);result['cache_hit']=False
            self.stats['terminal_completed']+=1
            self.record(dict(event='terminal_complete',call=call,problem=p.identity,elapsed_s=time.perf_counter()-tick,
                result={k:v for k,v in result.items() if k!='plan'}))
        except Exception as exc:
            self.stats['terminal_failed']+=1
            self.record(dict(event='terminal_error',call=call,problem=p.identity,error=type(exc).__name__,message=str(exc)))
            raise
        self.cache[key]=copy.deepcopy(result)
        return result
