"""Small reproducible, non-learning tabletop coverage-and-routing benchmark.

All physical numbers are demonstration parameters, not measurements of a robot.
Navigation is a continuous polyline on a corridor roadmap, not task-to-task lines.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import hashlib
import math
import os
from pathlib import Path
import platform
import time
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
if __package__:
    from .validator import validate
else:
    from validator import validate

ROBOT = dict(base_radius=0.32, clearance=0.08, arm_forward_offset=0.12,
             arm_spacing=0.36, arm_min_reach=0.10, arm_max_reach=0.90,
             arm_sector_half_angle_deg=100.0, shoulder_height=0.75,
             fixed_lift_height=0.40)
GEOMETRY = dict(table_width=1.0, table_depth=0.7, aisle_gap=1.2,
                table_height=0.75, dock_gap=0.50, task_margin=0.04)
METHODS = ('nearest_feasible', 'nearest_feasible_2opt',
           'coverage_cost', 'coverage_cost_2opt')
DEFAULT_SCALES = ((4, 3), (6, 6), (8, 8), (10, 10))


def scene_parameters(geometry=None, robot=None):
    """Validate configurable demo parameters; height remains fixed metadata."""
    values = []
    for name, defaults, overrides in (('geometry', GEOMETRY, geometry),
                                       ('robot', ROBOT, robot)):
        if overrides is not None and not isinstance(overrides, dict):
            raise ValueError(f'{name} must be a JSON object')
        overrides = {} if overrides is None else overrides
        unknown = set(overrides) - set(defaults)
        if unknown:
            raise ValueError(f'unknown {name} parameters: {sorted(unknown)}')
        merged = {**defaults, **overrides}
        for key, value in merged.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{name}.{key} must be a finite number')
        values.append(merged)
    g, r = values
    for key in ('table_width', 'table_depth', 'aisle_gap', 'dock_gap'):
        if g[key] <= 0:
            raise ValueError(f'geometry.{key} must be positive')
    if g['table_height'] < 0 or not 0 <= g['task_margin'] < min(g['table_width'], g['table_depth']) / 2:
        raise ValueError('table height must be nonnegative and task margin must fit inside the table')
    for key in ('base_radius', 'clearance', 'arm_spacing', 'arm_min_reach',
                'shoulder_height', 'fixed_lift_height'):
        if r[key] < 0:
            raise ValueError(f'robot.{key} must be nonnegative')
    if r['arm_max_reach'] <= r['arm_min_reach']:
        raise ValueError('arm_max_reach must be greater than arm_min_reach')
    if not 0 < r['arm_sector_half_angle_deg'] <= 180:
        raise ValueError('arm_sector_half_angle_deg must be in (0, 180]')
    return g, r


def make_scene(nx, ny, seed, counts=None, *, geometry=None, robot=None):
    for name, value in (('nx', nx), ('ny', ny), ('seed', seed)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < (0 if name == 'seed' else 1):
            raise ValueError(f'{name} must be an integer >= {0 if name == "seed" else 1}')
    if counts is not None:
        if len(counts) != nx * ny or any(isinstance(c, bool) or not isinstance(c, (int, np.integer)) or not 0 <= c <= 3 for c in counts):
            raise ValueError('counts must contain one integer in [0, 3] per table')
    g, r = scene_parameters(geometry, robot)
    rng = np.random.default_rng(seed)
    width, depth, gap, table_z, dock_gap = (g[key] for key in
        ('table_width', 'table_depth', 'aisle_gap', 'table_height', 'dock_gap'))
    margin = g['task_margin']
    sx, sy = width + gap, depth + gap
    tables, tasks, candidates = [], [], []
    for iy in range(ny):
        for ix in range(nx):
            tid = iy * nx + ix
            cx, cy = ix * sx + gap / 2 + width / 2, iy * sy + gap / 2 + depth / 2
            table = dict(id=tid, ix=ix, iy=iy, cx=cx, cy=cy,
                         xmin=cx-width/2, xmax=cx+width/2,
                         ymin=cy-depth/2, ymax=cy+depth/2, z=table_z)
            tables.append(table)
            count = int(rng.integers(0, 4)) if counts is None else int(counts[tid])
            for _ in range(count):
                tasks.append(dict(id=len(tasks), table_id=tid,
                                  x=float(rng.uniform(table['xmin']+margin, table['xmax']-margin)),
                                  y=float(rng.uniform(table['ymin']+margin, table['ymax']-margin)), z=table_z))
            for side in ('bottom', 'top', 'left', 'right'):
                for frac in (-0.25, 0.25):
                    if side in ('bottom', 'top'):
                        x = cx + frac * width
                        y = cy + (-1 if side == 'bottom' else 1) * (depth/2 + dock_gap)
                        theta = math.pi/2 if side == 'bottom' else -math.pi/2
                        anchor = [x, iy * sy if side == 'bottom' else (iy+1)*sy]
                    else:
                        x = cx + (-1 if side == 'left' else 1) * (width/2 + dock_gap)
                        y = cy + frac * depth
                        theta = 0.0 if side == 'left' else math.pi
                        anchor = [ix*sx if side == 'left' else (ix+1)*sx, y]
                    candidates.append(dict(id=len(candidates), table_id=tid, side=side,
                                           x=x, y=y, theta=theta, h=r['fixed_lift_height'],
                                           corridor_anchor=anchor))
    return dict(schema_version=2, parameter_status='DEMONSTRATION_ONLY_NOT_MEASURED',
                nx=nx, ny=ny, seed=seed, start=[0.0,0.0], robot=r, geometry=g,
                table_width=width, table_depth=depth, aisle_gap=gap,
                x_lines=[i*sx for i in range(nx+1)], y_lines=[i*sy for i in range(ny+1)],
                tables=tables, tasks=tasks, candidates=candidates,
                model_scope='fixed-height spherical-sector arm proxy; independent sequential one-arm tasks; no IK, joint path, or hardware validation')


def feasibility(scene):
    n, k = len(scene['tasks']), len(scene['candidates'])
    F = np.zeros((n,k,2), dtype=bool)
    distances = np.full((n,k,2), np.inf)
    r = scene['robot']
    for j,c in enumerate(scene['candidates']):
        for i,t in enumerate(scene['tasks']):
            dx,dy = t['x']-c['x'],t['y']-c['y']
            xx = math.cos(c['theta'])*dx + math.sin(c['theta'])*dy - r['arm_forward_offset']
            yy = -math.sin(c['theta'])*dx + math.cos(c['theta'])*dy
            zz = t['z']-r['shoulder_height']
            for a, sign in enumerate((1,-1)):
                arm_y = yy-sign*r['arm_spacing']/2
                rr = math.sqrt(xx*xx+arm_y*arm_y+zz*zz)
                distances[i,j,a] = rr
                F[i,j,a] = (r['arm_min_reach'] <= rr <= r['arm_max_reach']
                             and abs(math.atan2(arm_y,xx)) <= math.radians(r['arm_sector_half_angle_deg']))
    sparse = [dict(task_id=i, left=np.flatnonzero(F[i,:,0]).tolist(),
                   right=np.flatnonzero(F[i,:,1]).tolist()) for i in range(n)]
    return F, distances, sparse


def navigation(scene):
    # Independent from validator.py: vectorized segment/AABB clipping is used
    # here to remove blocked graph edges before shortest-path computation.
    radius = scene['robot']['base_radius'] + scene['robot']['clearance']
    boxes = np.asarray([(t['xmin']-radius, t['ymin']-radius,
                         t['xmax']+radius, t['ymax']+radius)
                        for t in [*scene['tables'], *scene.get('obstacles', [])]], dtype=float).reshape(-1, 4)

    def blocked(a, b):
        lower, upper = np.zeros(len(boxes)), np.ones(len(boxes))
        possible = np.ones(len(boxes), dtype=bool)
        for axis in (0, 1):
            delta = b[axis] - a[axis]
            if abs(delta) < 1e-12:
                possible &= (a[axis] >= boxes[:, axis]) & (a[axis] <= boxes[:, axis+2])
            else:
                t0 = (boxes[:, axis] - a[axis]) / delta
                t1 = (boxes[:, axis+2] - a[axis]) / delta
                lower = np.maximum(lower, np.minimum(t0, t1))
                upper = np.minimum(upper, np.maximum(t0, t1))
        return bool(np.any(possible & (lower <= upper)))

    if blocked(scene['start'], scene['start']):
        raise ValueError('depot is inside an inflated obstacle; adjust geometry or robot dimensions')
    coordinates, ids = [], {}
    def node(point):
        key = (round(point[0],10),round(point[1],10))
        if key not in ids:
            ids[key] = len(coordinates)
            coordinates.append(list(key))
        return ids[key]
    for x in scene['x_lines']:
        for y in scene['y_lines']:
            node((x,y))
    connectors=[]
    service_nodes=[node(scene['start'])]
    for c in scene['candidates']:
        a,b = node((c['x'],c['y'])),node(c['corridor_anchor'])
        service_nodes.append(a)
        connectors.append((a,b))
    edges = set()
    for x in scene['x_lines']:
        lane = sorted((i for i,p in enumerate(coordinates) if abs(p[0]-x)<1e-8),key=lambda i:coordinates[i][1])
        edges.update(zip(lane,lane[1:]))
    for y in scene['y_lines']:
        lane = sorted((i for i,p in enumerate(coordinates) if abs(p[1]-y)<1e-8),key=lambda i:coordinates[i][0])
        edges.update(zip(lane,lane[1:]))
    edges.update(connectors)
    edges = sorted({tuple(sorted((a, b))) for a, b in edges if a != b})
    row,col,data=[],[],[]
    rejected = 0
    for a,b in edges:
        if blocked(coordinates[a], coordinates[b]):
            rejected += 1
            continue
        d=math.dist(coordinates[a],coordinates[b])
        row += [a,b]; col += [b,a]; data += [d,d]
    graph=csr_matrix((data,(row,col)),shape=(len(coordinates),len(coordinates)))
    dist,pred=dijkstra(graph,directed=False,indices=service_nodes,return_predecessors=True)
    costs=dist[:,service_nodes]
    return dict(coordinates=coordinates,service_nodes=service_nodes,costs=costs,
                predecessors=pred,edge_count=len(edges)-rejected,
                rejected_edge_count=rejected,node_count=len(coordinates))


def improve_2opt(route,costs,max_passes=60):
    route=list(route)
    for _ in range(max_passes):
        best_delta,best_move=-1e-9,None
        for i in range(1,len(route)-2):
            for j in range(i+1,len(route)-1):
                a,b,c,d=route[i-1],route[i],route[j],route[j+1]
                delta=costs[a,c]+costs[b,d]-costs[a,b]-costs[c,d]
                if delta<best_delta:
                    best_delta,best_move=delta,(i,j)
        if best_move is None:
            break
        i,j=best_move
        route[i:j+1]=reversed(route[i:j+1])
    return route


def expand_route(route,nav):
    polyline,leg_costs=[],[]
    for source,target in zip(route,route[1:]):
        a,b=nav['service_nodes'][source],nav['service_nodes'][target]
        path=[b]
        while path[-1]!=a:
            predecessor=int(nav['predecessors'][source,path[-1]])
            if predecessor<0:
                raise RuntimeError('Disconnected navigation leg')
            path.append(predecessor)
        points=[nav['coordinates'][v] for v in reversed(path)]
        polyline.extend(points if not polyline else points[1:])
        leg_costs.append(float(nav['costs'][source,target]))
    return polyline,leg_costs


def solve(scene,F,arm_dist,nav,method,stop_penalty=0.75,*,two_opt_fn=None):
    if method not in METHODS:
        raise ValueError(f'unknown method: {method}')
    if not math.isfinite(stop_penalty) or stop_penalty < 0:
        raise ValueError('stop_penalty must be finite and nonnegative')
    feasible=F.any(axis=2)
    navigable = np.isfinite(nav['costs'][0, 1:]) & np.isfinite(nav['costs'][1:, 0])
    unserved=np.ones(len(scene['tasks']),dtype=bool)
    route=[0]; selected=[]; assignments=[]
    while unserved.any():
        gains=feasible[unserved].sum(axis=0)
        travel=nav['costs'][route[-1],1:]
        eligible=(gains>0)&np.isfinite(travel)&navigable
        if not eligible.any():
            break
        if method in ('nearest_feasible', 'nearest_feasible_2opt'):
            scores=np.where(eligible,travel,np.inf)
            j=int(np.argmin(scores))
        else:
            denominator = travel + stop_penalty
            scores = np.full(len(gains), -np.inf)
            np.divide(gains, denominator, out=scores, where=eligible & (denominator > 0))
            scores[eligible & (denominator == 0)] = np.inf
            j=int(np.argmax(scores))
        served=np.flatnonzero(unserved & feasible[:,j])
        for i in served:
            arm=int(np.argmin(np.where(F[i,j],arm_dist[i,j],np.inf)))
            assignments.append(dict(task_id=int(i),candidate_id=j,arm='L' if arm==0 else 'R'))
        unserved[served]=False
        selected.append(j);route.append(j+1)
    route.append(0)
    before=float(sum(nav['costs'][a,b] for a,b in zip(route,route[1:])))
    if method.endswith('_2opt'):
        route=(two_opt_fn or improve_2opt)(route,nav['costs'])
    polyline,leg_costs=expand_route(route,nav)
    length=float(sum(leg_costs))
    service_stops = []
    arrival_distance = 0.0
    for step, (node_id, leg_cost) in enumerate(zip(route[1:-1], leg_costs), start=1):
        j = node_id - 1
        candidate = scene['candidates'][j]
        arrival_distance += leg_cost
        service_stops.append(dict(sequence=step, candidate_id=j,
                                  pose={key: candidate[key] for key in ('x', 'y', 'theta', 'h')},
                                  arrival_distance_m=arrival_distance,
                                  tasks=[dict(task_id=a['task_id'], arm=a['arm'])
                                         for a in assignments if a['candidate_id'] == j]))
    unserved_ids = np.flatnonzero(unserved).tolist()
    reasons = [dict(task_id=i, reason=('no_arm_feasible_candidate' if not feasible[i].any()
                                      else 'no_navigable_feasible_candidate')) for i in unserved_ids]
    return dict(method=method,status='complete' if not unserved_ids else 'partial',
                selected_candidate_ids=selected,
                visit_order_candidate_ids=[i-1 for i in route[1:-1]],
                assignments=assignments,service_stops=service_stops,
                unserved_task_ids=unserved_ids,unserved_reasons=reasons,
                route_polyline=polyline,route_leg_costs_m=leg_costs,
                route_length_m=length,route_length_before_2opt_m=before,
                stop_penalty_equivalent_m=stop_penalty,
                objective_length_plus_stops_m=length+stop_penalty*len(selected))


def run_case(nx,ny,seed,*,methods=METHODS,stop_penalty=0.75,geometry=None,robot=None):
    if not methods or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise ValueError('methods must be a nonempty, unique subset of METHODS')
    t=time.perf_counter();scene=make_scene(nx,ny,seed,geometry=geometry,robot=robot);scene_s=time.perf_counter()-t
    t=time.perf_counter();F,arm_dist,sparse=feasibility(scene);reach_s=time.perf_counter()-t
    t=time.perf_counter();nav=navigation(scene);nav_s=time.perf_counter()-t
    rows=[];plans={}
    navigable = np.isfinite(nav['costs'][0, 1:]) & np.isfinite(nav['costs'][1:, 0])
    reachable_tasks = int((F.any(axis=2) & navigable).any(axis=1).sum())
    for method in methods:
        t=time.perf_counter();plan=solve(scene,F,arm_dist,nav,method,stop_penalty);solve_s=time.perf_counter()-t
        t=time.perf_counter();check=validate(scene,plan,sparse);validate_s=time.perf_counter()-t
        plan['validation']=check;plans[method]=plan
        n=len(scene['tasks']);served=len(plan['assignments'])
        rows.append(dict(nx=nx,ny=ny,tables=nx*ny,seed=seed,method=method,
                         tasks=n,feasible_tasks=int(F.any(axis=(1,2)).sum()),served_tasks=served,
                         reachable_tasks=reachable_tasks,
                         coverage=served/n if n else 1.0,unserved_tasks=n-served,
                         stops=len(plan['selected_candidate_ids']),route_length_m=plan['route_length_m'],
                         objective_m=plan['objective_length_plus_stops_m'],
                         before_2opt_m=plan['route_length_before_2opt_m'],
                         scene_s=scene_s,reachability_s=reach_s,navigation_preprocess_s=nav_s,
                         solver_s=solve_s,validation_s=validate_s,
                         total_s=scene_s+reach_s+nav_s+solve_s+validate_s,
                         validator_pass=check['valid_model_plan'],collision_count=check['collision_count']))
    output=dict(scene=scene,feasibility=sparse,plans=plans,
                navigation=dict(model='undirected corridor centerline roadmap; exact Dijkstra shortest paths ON THIS ROADMAP, not globally shortest continuous paths',
                                nodes=nav['node_count'],edges=nav['edge_count'],
                                rejected_collision_edges=nav['rejected_edge_count'],
                                navigable_candidate_ids=np.flatnonzero(navigable).tolist()),metrics=rows)
    return output,rows


def edge_case_tests():
    results=[]
    for name,count in [('empty_tasks',0),('unreachable_task',1)]:
        scene=make_scene(1,1,999,counts=[count])
        if count:
            scene['tasks'][0]['z']=3.0
        F,ad,sparse=feasibility(scene);nav=navigation(scene)
        for method in METHODS:
            p=solve(scene,F,ad,nav,method);v=validate(scene,p,sparse)
            assert v['valid_model_plan'],v
            assert len(p['unserved_task_ids'])==count
            assert len(p['assignments'])==0
            assert v['all_tasks_served']==(count==0)
            results.append(dict(case=name,method=method,passed=True,tasks=count,
                                unserved_task_ids=p['unserved_task_ids'],validation=v))
    return results


def plot_scene(ax,output,title,method='coverage_cost_2opt'):
    from matplotlib.patches import Rectangle
    scene=output['scene'];plan=output['plans'][method]
    for t in scene['tables']:
        ax.add_patch(Rectangle((t['xmin'],t['ymin']),t['xmax']-t['xmin'],t['ymax']-t['ymin'],
                               facecolor='#e3e8ee',edgecolor='#9aa7b5',linewidth=0.5,zorder=1))
    for obstacle in scene.get('obstacles', []):
        ax.add_patch(Rectangle((obstacle['xmin'], obstacle['ymin']), obstacle['xmax']-obstacle['xmin'],
                               obstacle['ymax']-obstacle['ymin'], facecolor='#555555', edgecolor='black',
                               linewidth=0.6, zorder=3))
    p=np.array(plan['route_polyline']);ax.plot(p[:,0],p[:,1],color='#1762ad',lw=1.0,zorder=2,label='Roadmap route')
    for arm,color in [('L','#14856a'),('R','#b45230')]:
        ids={x['task_id'] for x in plan['assignments'] if x['arm']==arm}
        tasks=[t for t in scene['tasks'] if t['id'] in ids]
        if tasks:ax.scatter([t['x'] for t in tasks],[t['y'] for t in tasks],s=9,c=color,label=f'{arm}-arm proxy task',zorder=4)
    missing = set(plan['unserved_task_ids'])
    tasks = [t for t in scene['tasks'] if t['id'] in missing]
    if tasks:
        ax.scatter([t['x'] for t in tasks], [t['y'] for t in tasks], s=30,
                   marker='x', c='#cf2437', label='Unserved task', zorder=5)
    docks=[scene['candidates'][j] for j in plan['selected_candidate_ids']]
    ax.scatter([d['x'] for d in docks],[d['y'] for d in docks],s=13,marker='^',c='#7547b6',zorder=4,label='Selected dock')
    ax.scatter([scene['start'][0]],[scene['start'][1]],marker='*',s=80,c='#d59b20',zorder=5,label='Start / return')
    ax.set(xlabel='x [m]',ylabel='y [m]',title=title,aspect='equal')
    ax.set_xlim(-0.65,scene['x_lines'][-1]+0.65);ax.set_ylim(-0.65,scene['y_lines'][-1]+0.65)
    ax.grid(alpha=.13)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows, fieldnames=None):
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


SCHEDULE_COLUMNS = ['scale', 'seed', 'method', 'visit_index', 'candidate_id',
                    'task_id', 'table_id', 'arm', 'task_x_m', 'task_y_m', 'task_z_m',
                    'dock_x_m', 'dock_y_m', 'dock_theta_rad', 'fixed_lift_h_m',
                    'arrival_distance_m']


def schedule_rows(output):
    scene = output['scene']
    tasks = {task['id']: task for task in scene['tasks']}
    for method, plan in output['plans'].items():
        for stop in plan['service_stops']:
            for assignment in stop['tasks']:
                task = tasks[assignment['task_id']]
                pose = stop['pose']
                yield dict(scale=f'{scene["nx"]}x{scene["ny"]}', seed=scene['seed'], method=method,
                           visit_index=stop['sequence'], candidate_id=stop['candidate_id'],
                           task_id=task['id'], table_id=task['table_id'], arm=assignment['arm'],
                           task_x_m=task['x'], task_y_m=task['y'], task_z_m=task['z'],
                           dock_x_m=pose['x'], dock_y_m=pose['y'], dock_theta_rad=pose['theta'],
                           fixed_lift_h_m=pose['h'], arrival_distance_m=stop['arrival_distance_m'])


def save_plots(primary, output_dir, method):
    os.environ.setdefault('MPLCONFIGDIR', str(output_dir / 'mplcache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.titlesize':11,'figure.facecolor':'white'})
    columns = min(2, len(primary))
    rows = math.ceil(len(primary) / columns)
    fig,axes=plt.subplots(rows,columns,figsize=(6.5*columns,5.5*rows),squeeze=False,constrained_layout=True)
    for ax,out in zip(axes.flat,primary):
        p=out['plans'][method];s=out['scene']
        plot_scene(ax,out,f'{s["nx"]} x {s["ny"]} tables | {len(s["tasks"])} tasks | {len(p["selected_candidate_ids"])} docks | {p["route_length_m"]:.1f} m',method)
    for ax in list(axes.flat)[len(primary):]:
        ax.set_visible(False)
    fig.suptitle(f'Non-learning baseline | {method}\nDemonstration geometry; arm proxy only; seed {primary[0]["scene"]["seed"]}',fontsize=14)
    legend = {}
    for ax in list(axes.flat)[:len(primary)]:
        handles, labels = ax.get_legend_handles_labels()
        legend.update(zip(labels, handles))
    fig.legend(legend.values(),legend.keys(),loc='outside lower center',ncol=3,frameon=False)
    fig.savefig(output_dir/'scales.png',dpi=170);plt.close(fig)
    for out in primary:
        scene = out['scene']
        fig,ax=plt.subplots(figsize=(12,10),constrained_layout=True)
        plot_scene(ax,out,f'{scene["nx"]} x {scene["ny"]} tables | {method} | seed {scene["seed"]}',method)
        handles,labels=ax.get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False,fontsize=9)
        fig.savefig(output_dir/f'route_{scene["nx"]}x{scene["ny"]}_seed{scene["seed"]}.png',dpi=170)
        plt.close(fig)


def summarize(allrows, scales, methods):
    summary = []
    for nx, ny in scales:
        for method in methods:
            rr = [r for r in allrows if (r['nx'], r['ny'], r['method']) == (nx, ny, method)]
            summary.append(dict(scale=f'{nx}x{ny}', method=method, instances=len(rr),
                                avg_tasks=float(np.mean([r['tasks'] for r in rr])),
                                avg_stops=float(np.mean([r['stops'] for r in rr])),
                                avg_route_m=float(np.mean([r['route_length_m'] for r in rr])),
                                avg_objective_m=float(np.mean([r['objective_m'] for r in rr])),
                                avg_solver_s=float(np.mean([r['solver_s'] for r in rr])),
                                avg_total_s=float(np.mean([r['total_s'] for r in rr])),
                                all_valid=all(r['validator_pass'] for r in rr),
                                all_covered=all(r['coverage'] == 1 for r in rr)))
    return summary


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def scale_argument(value):
    try:
        nx, ny = value.lower().replace('\u00d7', 'x').split('x')
        return positive_int(nx), positive_int(ny)
    except (ValueError, argparse.ArgumentTypeError) as error:
        raise argparse.ArgumentTypeError('use a positive scale such as 4x3 or 10x10') from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='new or empty output directory; default: timestamped folder')
    parser.add_argument('--seeds', type=positive_int, default=5, help='number of random seeds per scale')
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--scales', nargs='+', type=scale_argument, default=list(DEFAULT_SCALES))
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--stop-penalty', type=float, default=0.75, help='nonnegative equivalent distance per stop [m]')
    parser.add_argument('--config', type=Path, help='JSON with geometry and/or robot overrides')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)
    if args.seed_start < 0:
        parser.error('--seed-start must be nonnegative')
    if not math.isfinite(args.stop_penalty) or args.stop_penalty < 0:
        parser.error('--stop-penalty must be finite and nonnegative')
    if len(set(args.scales)) != len(args.scales) or len(set(args.methods)) != len(args.methods):
        parser.error('--scales and --methods must not contain duplicates')
    try:
        config = json.loads(args.config.read_text(encoding='utf-8-sig')) if args.config else {}
        if not isinstance(config, dict) or set(config) - {'geometry', 'robot'}:
            raise ValueError('config must contain only geometry and/or robot objects')
        geometry, robot = scene_parameters(config.get('geometry'), config.get('robot'))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    started = datetime.now(timezone(timedelta(hours=8)))
    output_dir = (args.output or Path(__file__).parent / f'results_run_{started:%Y%m%d_%H%M%S_%f}').resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        parser.error('--output must be new or empty; existing results are never overwritten')
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(__file__).parent
    manifest = dict(started_at=started.isoformat(), status='running', scales=args.scales,
                    seeds=list(range(args.seed_start, args.seed_start+args.seeds)),
                    methods=args.methods, stop_penalty_equivalent_m=args.stop_penalty,
                    geometry=geometry, robot=robot, plots=not args.no_plots,
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(source_dir.glob('*.py'))})
    write_json(output_dir/'run_manifest.json', manifest)
    allrows, primary, schedule = [], [], []
    for nx, ny in args.scales:
        for seed in manifest['seeds']:
            try:
                out, rows = run_case(nx, ny, seed, methods=args.methods, stop_penalty=args.stop_penalty,
                                     geometry=geometry, robot=robot)
            except ValueError as error:
                manifest.update(status='failed', error=str(error))
                write_json(output_dir/'run_manifest.json', manifest)
                parser.exit(2, f'error: {error}\n')
            allrows.extend(rows)
            schedule.extend(schedule_rows(out))
            write_json(output_dir/f'scene_plan_{nx}x{ny}_seed{seed}.json', out)
            if seed == args.seed_start:
                primary.append(out)
            print(f'{nx}x{ny} seed={seed}: tasks={rows[0]["tasks"]}, ' + ', '.join(
                f'{r["method"]}: {r["served_tasks"]}/{r["tasks"]}, {r["stops"]} docks, '
                f'{r["route_length_m"]:.2f}m, solve={r["solver_s"]:.3f}s, valid={r["validator_pass"]}' for r in rows), flush=True)
    write_csv(output_dir/'metrics.csv', allrows)
    write_csv(output_dir/'service_schedule.csv', schedule, SCHEDULE_COLUMNS)
    summary = summarize(allrows, args.scales, args.methods)
    write_json(output_dir/'summary.json', summary)
    write_json(output_dir/'edge_case_tests.json', edge_case_tests())
    try:
        matplotlib_version = version('matplotlib')
    except PackageNotFoundError:
        matplotlib_version = None
    write_json(output_dir/'environment.json', dict(
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        matplotlib=matplotlib_version, platform=platform.platform(),
        note='CPU wall-clock times; no training. Timings exclude imports, plots, serialization and file I/O. '
             'Scene/reachability/navigation preprocessing is shared per scene and included in each method total.'))
    if not args.no_plots:
        plot_method = 'coverage_cost_2opt' if 'coverage_cost_2opt' in args.methods else args.methods[0]
        save_plots(primary, output_dir, plot_method)
    all_valid = all(r['validator_pass'] for r in allrows)
    all_covered = all(r['coverage'] == 1 for r in allrows)
    manifest.update(status='complete' if all_valid and all_covered else 'incomplete',
                    plan_count=len(allrows), all_valid=all_valid, all_tasks_served=all_covered,
                    finished_at=datetime.now(timezone(timedelta(hours=8))).isoformat())
    write_json(output_dir/'run_manifest.json', manifest)
    print(f'plans={len(allrows)}; all_valid={all_valid}; all_tasks_served={all_covered}', flush=True)
    print(f'Output: {output_dir}', flush=True)
    return 0 if all_valid and all_covered else 2


if __name__=='__main__':
    raise SystemExit(main())
