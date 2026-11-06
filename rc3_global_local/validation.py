"""Independent endpoint-aware scalar record/geometry validation. No IK claims."""
import math
from rc3_baseline.validator import proxy_reachable,segment_hits_rectangle


def _point(parent,node):
    if node==0:return parent.scene['start']
    c=parent.scene['candidates'][node-1];return [c['x'],c['y']]


def validate(problem,plan):
    try:return _validate(problem,plan)
    except (ValueError,TypeError,KeyError,IndexError,OverflowError) as e:
        return dict(valid_model_plan=False,all_tasks_served=False,errors=[str(e)],physical_joint_trajectory_certified=False)


def _validate(p,plan):
    errors=[];parent=p.parent;scene=parent.scene
    def need(condition,message):
        if not condition:raise ValueError(message)
    if parent.mode=='external_tables':
        from rc3_v5.external import check_bundle
        F,A=check_bundle(parent.bundle)
        need((F==parent.F).all() and (A==parent.A).all(),'Parent external table mismatch')
    need(plan['problem_identity']==p.identity and plan['parent_identity']==parent.identity,'Wrong problem identity')
    need(plan['entry_global_node']==p.entry and plan['exit_global_node']==p.exit,'Wrong requested endpoint')
    visited=plan['visit_order_candidate_ids'];allowed=set(p.candidate_ids.tolist())
    need(all(type(j) is int and j in allowed for j in visited),'Unknown/unavailable service candidate')
    need(len(visited)==len(set(visited)),'Repeated service candidate')
    nodes=[p.entry,*[j+1 for j in visited],p.exit]
    need(nodes==plan['navigation_global_nodes'],'Global route/ID mapping mismatch')
    tasks=set(p.task_ids.tolist());assignments=plan['assignments'];assigned=[a['task_id'] for a in assignments]
    need(plan['target_task_ids_global']==p.task_ids.tolist(),'Wrong target task manifest')
    need(len(assigned)==len(set(assigned)) and set(assigned)==tasks,'Missing/duplicate/non-target task')
    for a in assignments:
        i,j,arm=a['task_id'],a['candidate_id'],a['arm']
        need(type(i) is int and type(j) is int and j in visited and arm in ('L','R'),'Invalid assignment')
        r=0 if arm=='L' else 1
        need(bool(parent.F[i,j,r]) and bool(parent.A[i,j,r]),'Same-arm F&A violation')
        if parent.mode=='proxy':need(proxy_reachable(scene['tasks'][i],scene['candidates'][j],arm,scene['robot']),'Independent proxy reachability failed')
    expected_left=set(tasks)
    for j in visited:
        served={i for i in expected_left if any(parent.F[i,j,r] and parent.A[i,j,r] for r in (0,1))}
        need(bool(served),'Zero-gain service stop')
        need({a['task_id'] for a in assignments if a['candidate_id']==j}==served,'Assignment differs from ordered dynamic service')
        expected_left-=served
    need(not expected_left,'Incomplete task coverage')
    stops=plan['service_stops'];need(len(stops)==len(visited),'Wrong stop record count')
    for k,(stop,j) in enumerate(zip(stops,visited)):
        need(stop['sequence']==k+1 and stop['candidate_id']==j,'Wrong stop order')
        for key in ('x','y','theta','h'):need(abs(stop['pose'][key]-scene['candidates'][j][key])<1e-9,'Wrong docking pose')
        need(stop['tasks']==[dict(task_id=a['task_id'],arm=a['arm']) for a in assignments if a['candidate_id']==j],'Wrong stop service list')
    legs=plan['route_leg_costs_m'];need(len(legs)==len(nodes)-1,'Wrong number of navigation legs')
    for k,(a,b) in enumerate(zip(nodes,nodes[1:])):
        need(math.isfinite(legs[k]) and abs(legs[k]-parent.D[a,b])<=1e-7,'Wrong/nonfinite navigation cost')
    points=plan['route_polyline']
    need(bool(points) and all(len(q)==2 and all(math.isfinite(float(x)) for x in q) for q in points),'Invalid navigation polyline')
    need(math.dist(points[0],_point(parent,p.entry))<=1e-8 and math.dist(points[-1],_point(parent,p.exit))<=1e-8,'Polyline endpoint mismatch')
    length=sum(math.dist(a,b) for a,b in zip(points,points[1:]))
    need(abs(length-sum(legs))<=1e-7 and abs(plan['route_length_m']-length)<=1e-7,'Polyline length mismatch')
    arcs=[0.]
    for a,b in zip(points,points[1:]):arcs.append(arcs[-1]+math.dist(a,b))
    arrival=0.
    for k,j in enumerate(visited):
        arrival+=legs[k]
        need(abs(stops[k]['arrival_distance_m']-arrival)<=1e-7,'Wrong arrival distance')
        pose=_point(parent,j+1)
        need(any(abs(s-arrival)<=1e-7 and math.dist(q,pose)<=1e-8 for s,q in zip(arcs,points)),'Polyline misses ordered service pose')
    radius=scene['robot']['base_radius']+scene['robot']['clearance']
    boxes=[(b['xmin']-radius,b['ymin']-radius,b['xmax']+radius,b['ymax']+radius) for b in [*scene['tables'],*scene.get('obstacles',[])]]
    segments=list(zip(points,points[1:])) or [(points[0],points[0])]
    collisions=sum(segment_hits_rectangle(a,b,box) for a,b in segments for box in boxes)
    need(collisions==0,'Base path intersects inflated obstacle')
    need(plan['stop_penalty_equivalent_m']==.75 and abs(plan['objective_length_plus_stops_m']-(length+.75*len(visited)))<=1e-7,'Objective mismatch')
    need(plan['unserved_task_ids']==[] and plan['status']=='complete','Wrong completion record')
    need(plan['arm_joint_trajectories'] is None and plan['physical_joint_trajectory_certified'] is False,'Unsupported physical trajectory claim')
    return dict(valid_model_plan=True,all_tasks_served=True,collision_count=0,errors=errors,
        independent_polyline_length_m=length,effective_same_arm_F_and_A=True,
        endpoint_constraints_satisfied=True,physical_joint_trajectory_certified=False,
        scope='Original global roadmap, circular-base inflated AABB navigation, and proxy/externally supplied per-arm service tables; no built-in IK, arm swept path or joint trajectory certificate')
