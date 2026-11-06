"""Independent saved-artifact audit. No planner/model/scene factories are imported.

This utility only hashes files, slices the three saved S0 arrays, and recomputes
scalar bookkeeping/geometry on saved plans. It does not rerun scientific calls.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import tarfile
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
BASE = '057830d584b6c71c79ef5118087c681c98135fe2'
QA = Counter()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def need(condition, message):
    QA['assertions'] += 1
    if not condition:
        raise AssertionError(message)


def near(a, b, message, atol=1e-7):
    need(math.isfinite(float(a)) and abs(float(a)-float(b)) <= atol, message)


def ah(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode()+a.tobytes()).hexdigest()


def journal(name):
    return [json.loads(line) for line in (OUT/name).read_text(encoding='utf-8').splitlines() if line]


def verify_sources(receipt):
    for name, digest in receipt['source_sha256'].items():
        path = ROOT/name if (ROOT/name).exists() else OUT/name
        need(sha(path) == digest, 'Source drift: '+name)
        QA['source_hash_checks'] += 1


def frozen_bytes():
    # Archive contains ordinary Git blob bytes, or LFS pointers, never runs code.
    archive = subprocess.check_output(['git', 'archive', BASE, 'rc3_global_local'], cwd=ROOT)
    manifest = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            reference = tf.extractfile(member).read()
            actual = (ROOT/member.name).read_bytes()
            if reference.startswith(b'version https://git-lfs.github.com/spec/v1\n'):
                fields = reference.decode().splitlines()
                expected = next(x[11:] for x in fields if x.startswith('oid sha256:'))
                need(hashlib.sha256(actual).hexdigest() == expected, 'Frozen LFS byte drift: '+member.name)
                QA['frozen_lfs_files'] += 1
            else:
                need(actual == reference, 'Frozen byte drift: '+member.name)
            manifest[member.name] = hashlib.sha256(actual).hexdigest()
    need(len(manifest) == 140, 'Frozen S0 subtree file count')
    QA['frozen_files_checked'] = len(manifest)
    return dict(files=len(manifest), manifest_sha256=hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest())


def physical_layouts():
    gate = read(OUT/'GEOMETRY_GATE.json')
    need(gate['status'] == 'PASS', 'Geometry gate')
    records = gate['records']
    need(len(records) == 111, '111 saved physical layouts')
    keys = []
    for record in records:
        path = ROOT/record['file']; payload = read(path)
        raw = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        need(path.read_bytes() == raw and sha(path) == record['map_key'], 'Canonical physical key')
        need(set(payload) == {'tables','candidates','start','x_lines','y_lines','robot','obstacles'}, 'Physical-only payload')
        need(record['task_records'] == 0, 'No saved task draw')
        keys.append(record['map_key'])
    need(len(set(keys)) == 111, 'Physical layout uniqueness')
    need(Counter(r['role'] for r in records) == {'train':96,'development':3,'unseen':12}, 'Physical split counts')
    events = journal('geometry_journal.jsonl')
    need(Counter(x['event'] for x in events) == {'start':111,'complete':111}, 'Geometry journal accounting')
    need(all(r['matches_existing_saved_S0'] for r in gate['references']), 'Saved S0 layout reference receipt')
    QA['saved_geometry_files_checked'] = 111
    return dict(physical_keys=111, target_task_feasibility_tested=False,
                limitation='Physical independence is established; no future task/F&A instance was generated or validated.')


def load_parents():
    result = {}
    for case in range(3):
        prefix = ROOT/'rc3_global_local/parents'/f'{case:02}'
        scene = read(str(prefix)+'_scene.json'); receipt = read(str(prefix)+'_receipt.json')
        arrays_path = Path(str(prefix)+'_arrays.npz')
        need(sha(Path(str(prefix)+'_scene.json')) == receipt['scene_sha256'], 'Saved scene hash')
        need(sha(arrays_path) == receipt['arrays_sha256'], 'Saved NPZ hash')
        with np.load(arrays_path, allow_pickle=False) as z:
            arrays = {name:z[name].copy() for name in ('F','A','D')}
        for name in arrays:
            need(ah(arrays[name]) == receipt[name+'_sha256'], 'Saved array hash')
        arrays['scene'] = scene
        result[case] = arrays
    QA['saved_parent_array_loads'] = 3
    return result


def derived_view(parent, target, entry, exit_):
    effective = parent['F'][target] & parent['A'][target]
    candidates = np.flatnonzero(effective.any((0,2)) & np.isfinite(parent['D'][entry,1:]) & np.isfinite(parent['D'][1:,exit_]))
    F = effective[:,candidates,:]; C = F.any(2); nodes = candidates+1
    D = np.zeros((len(nodes)+1,len(nodes)+1), dtype=np.float64)
    D[0,0] = parent['D'][entry,exit_]; D[0,1:] = parent['D'][entry,nodes]
    D[1:,0] = parent['D'][nodes,exit_]; D[1:,1:] = parent['D'][np.ix_(nodes,nodes)]
    return candidates,F,C,D


def collision_count(points, scene):
    # Independent vector slab implementation; no frozen validator import/call.
    p = np.asarray(points,dtype=float)
    segments = np.stack((p[:-1],p[1:]),axis=1) if len(p)>1 else np.stack((p,p),axis=1)
    radius = scene['robot']['base_radius']+scene['robot']['clearance']
    boxes = np.array([[b['xmin']-radius,b['ymin']-radius,b['xmax']+radius,b['ymax']+radius]
                      for b in scene['tables']+scene.get('obstacles',[])])
    count = 0
    for start in range(0,len(segments),512):
        batch = segments[start:start+512]; a = batch[:,0,None,:]; delta = batch[:,1,None,:]-a
        lower,upper = boxes[None,:,:2],boxes[None,:,2:]
        parallel = np.abs(delta)<1e-12
        safe = np.where(parallel,1.,delta)
        t1,t2 = (lower-a)/safe,(upper-a)/safe
        lo = np.where(parallel,-np.inf,np.minimum(t1,t2)).max(2)
        hi = np.where(parallel,np.inf,np.maximum(t1,t2)).min(2)
        outside = (parallel & ((a<lower)|(a>upper))).any(2)
        count += int((~outside & (np.maximum(lo,0.)<=np.minimum(hi,1.))).sum())
        QA['segment_box_pairs_checked'] += len(batch)*len(boxes)
    return count


def proxy_reach(task, candidate, arm, robot):
    dx,dy = task['x']-candidate['x'],task['y']-candidate['y']; angle=candidate['theta']
    forward = math.cos(angle)*dx+math.sin(angle)*dy-robot['arm_forward_offset']
    side = -math.sin(angle)*dx+math.cos(angle)*dy-robot['arm_spacing']/2*(1 if arm=='L' else -1)
    height = task['z']-robot['shoulder_height']; distance=math.sqrt(forward**2+side**2+height**2)
    return robot['arm_min_reach']-1e-10<=distance<=robot['arm_max_reach']+1e-10 and abs(math.atan2(side,forward))<=math.radians(robot['arm_sector_half_angle_deg'])+1e-10


def audit_plan(plan, parent, target, candidates, entry, exit_, native_reference=False):
    scene=parent['scene']; order=plan['visit_order_candidate_ids']; nodes=[entry,*[x+1 for x in order],exit_]
    need(len(order)==len(set(order)) and set(order)<=set(candidates.tolist()), 'Legal unique service stops')
    need(plan['selected_candidate_ids']==order and plan['status']=='complete' and not plan['unserved_task_ids'], 'Complete plan record')
    translated=[dict(a,task_id=target[a['task_id']] if native_reference else a['task_id']) for a in plan['assignments']]
    need(len(translated)==len(target) and {a['task_id'] for a in translated}==set(target), 'Exactly all target task assignments')
    left=set(target)
    for candidate in order:
        served={i for i in left if (parent['F'][i,candidate]&parent['A'][i,candidate]).any()}
        need(bool(served) and {a['task_id'] for a in translated if a['candidate_id']==candidate}==served, 'Ordered dynamic coverage')
        left-=served
    need(not left, 'Complete coverage')
    for assignment in translated:
        i,j,arm=assignment['task_id'],assignment['candidate_id'],assignment['arm']
        need(arm in ('L','R') and j in order, 'Arm/service assignment IDs')
        r=0 if arm=='L' else 1
        need(bool(parent['F'][i,j,r] and parent['A'][i,j,r]), 'Same-arm F&A')
        need(proxy_reach(scene['tasks'][i],scene['candidates'][j],arm,scene['robot']), 'Independent proxy reach')
        QA['assignment_checks']+=1
    legs=plan['route_leg_costs_m']; need(len(legs)==len(nodes)-1, 'Leg count')
    for a,b,cost in zip(nodes,nodes[1:],legs):near(cost,parent['D'][a,b],'Original parent D navigation')
    points=plan['route_polyline']; need(bool(points) and all(len(p)==2 and all(math.isfinite(x) for x in p) for p in points),'Finite polyline')
    def point(node):
        return scene['start'] if node==0 else [scene['candidates'][node-1][k] for k in ('x','y')]
    need(math.dist(points[0],point(entry))<=1e-8 and math.dist(points[-1],point(exit_))<=1e-8,'Actual navigation endpoints')
    length=sum(math.dist(a,b) for a,b in zip(points,points[1:]));near(length,sum(legs),'Polyline vs D length')
    near(plan['route_length_m'],length,'Recorded L');near(plan['objective_length_plus_stops_m'],length+.75*len(order),'Recorded J=L+.75K')
    near(plan['stop_penalty_equivalent_m'],.75,'Penalty')
    arcs=[0.]
    for a,b in zip(points,points[1:]):arcs.append(arcs[-1]+math.dist(a,b))
    need(len(plan['service_stops'])==len(order), 'Service stop count')
    arrival=0.
    for k,(candidate,stop) in enumerate(zip(order,plan['service_stops'])):
        arrival+=legs[k]; near(stop['arrival_distance_m'],arrival,'Arrival distance')
        need(stop['sequence']==k+1 and stop['candidate_id']==candidate,'Service stop order')
        stop_tasks=[dict(a,task_id=target[a['task_id']] if native_reference else a['task_id']) for a in stop['tasks']]
        need(stop_tasks==[dict(task_id=a['task_id'],arm=a['arm']) for a in translated if a['candidate_id']==candidate],'Service stop assignment records')
        need(any(abs(distance-arrival)<=1e-7 and math.dist(p,point(candidate+1))<=1e-8 for p,distance in zip(points,arcs)),'Ordered service pose occurs on path')
        for key in ('x','y','theta','h'):near(stop['pose'][key],scene['candidates'][candidate][key],'Dock pose',1e-9)
    need(collision_count(points,scene)==0,'Independent inflated-AABB collision check')
    need(plan['validation']['valid_model_plan'] is True,'Saved validator result')
    if not native_reference:
        need(plan['navigation_global_nodes']==nodes and plan['target_task_ids_global']==target,'Endpoint plan manifest')
        need(plan['entry_global_node']==entry and plan['exit_global_node']==exit_,'Requested endpoints')
        need(plan['arm_joint_trajectories'] is None and plan['physical_joint_trajectory_certified'] is False,'No unsupported IK/joint claim')
    QA['saved_plans_verified']+=1


def engineering(parents):
    result=read(OUT/'ENGINEERING_RESULT.json'); protocol=read(OUT/'ENGINEERING_CHECK_PROTOCOL.json')
    verify_sources(protocol); need(result['status']=='PASS' and result['error'] is None,'Bounded engineering completed')
    c=result['counters'];expected={'endpoint_baseline_attempts':36,'endpoint_baseline_completed':36,'closed_reference_attempts':18,'closed_reference_completed':18,'policy_trajectories':9,'policy_encode_calls':9,'backward_attempts':3,'backward_completed':3,'optimizer_updates':0,'S1_training_calls':0,'S2_training_calls':0,'S3_task_queries':0,'original60_reads':0,'new_scene_generation':0,'navigation_preparations':0}
    for key,value in expected.items():need(c[key]==value,'Bounded count '+key)
    need(result['elapsed_wall_s']<=protocol['limits']['wall_s'],'Wall cap')
    need(sha(ROOT/'rc3_global_local/initial_policy.pt')==protocol['initial_policy_sha256'],'Checkpoint bytes')
    old_hash=read(ROOT/'rc3_global_local/gradient_probe.json')['tensor_hash_before']
    need(result['model_hash_before']==result['model_hash_after']==old_hash,'Saved weight hashes unchanged')
    rows={(r['case'],r['method']):r for r in result['baseline_rows']};need(len(rows)==36,'36 baseline row keys')
    closed_count=open_count=0
    for path in sorted((ROOT/'rc3_global_local/views').glob('*.json')):
        v=read(path);parent=parents[v['parent_case']];target=v['task_ids_global'];entry,exit_=v['entry_global_node'],v['exit_global_node']
        ids,F,C,D=derived_view(parent,target,entry,exit_);bound=read(OUT/f"bound_view_{v['case']:02}.json")
        need(ids.tolist()==v['candidate_ids_global']==bound['candidate_ids_global'],'Numeric candidate join')
        for name,value in [('effective_F_sha256',F),('coverage_sha256',C),('endpoint_D_sha256',D)]:need(ah(value)==v[name]==bound[name],'S0 snapshot numeric hash '+name)
        need(bound['s0_view_identity']==v['identity'] and bound['bound_view_identity']==bound['identity'],'Explicit two-namespace join')
        need(bound['window']==v['window'] and bound['task_ids_global']==target and bound['entry_global_node']==entry and bound['exit_global_node']==exit_,'Window/target/endpoint join')
        for method in ('fixed3','detour16'):
            saved=read(OUT/'baseline_plans'/f"{v['case']:02}_{method}_endpoint.json");plan,info=saved['plan'],saved['info']
            audit_plan(plan,parent,target,ids,entry,exit_)
            near(plan['objective_length_plus_stops_m'],rows[v['case'],method]['J'],'Summary J matches file')
            need(info['common_H_added'] is False and not info['fallback_used'] and not info['failures'],'Unchanged native pipeline/no fallback')
            if entry==exit_==0:
                reference=read(OUT/'baseline_plans'/f"{v['case']:02}_{method}_closed_reference.json")
                audit_plan(reference['plan'],parent,target,np.arange(len(parent['D'])-1),0,0,True)
                need(reference['plan']['visit_order_candidate_ids']==plan['visit_order_candidate_ids'],'Full native closed global route equivalence')
                near(reference['plan']['objective_length_plus_stops_m'],plan['objective_length_plus_stops_m'],'Full native closed objective equivalence')
                need(reference['info']['original_export_and_proxy_validator_unpatched'] is True,'Native reference exporter unpatched')
                closed_count+=1
            else:open_count+=1
    events=journal('engineering_journal.jsonl');need(Counter(x['event'] for x in events)=={'endpoint_start':36,'endpoint_complete':36,'closed_reference_start':18,'closed_reference_complete':18},'Planning journal exact accounting')
    terminal=journal('calibration_terminal_journal.jsonl');need(Counter(x['event'] for x in terminal)=={'terminal_start':9,'terminal_complete':9},'Reward journal exact accounting')
    completed={tuple(x['result']['raw_route']):x for x in terminal if x['event']=='terminal_complete'}
    need(len(completed)==9,'Nine unique persisted terminal requests')
    for group in result['calibration']:
        for row in group['rows']:
            terminal_result=completed[tuple(row['raw_route'])]['result']
            near(terminal_result['J'],row['J'],'Terminal journal J')
            need(terminal_result['route']==row['final_route'],'Terminal journal final route')
    for field in ('terminal_requests','terminal_attempts','terminal_completed','full_H_calls','exports','validation_completed'):need(result['terminal_counters'][field]==9,'Reward count '+field)
    need(result['terminal_counters']['terminal_cache_hits']==0 and result['terminal_counters']['terminal_failed']==0,'No uncharged terminal fallback/cache')
    calibration_audit(result,parents)
    return dict(endpoint_plans=36,closed_full_native_plans=18,open_endpoint_plans=open_count,
                closed_objective_and_order_equal=closed_count,calibration_plans=9,backwards_saved=3,optimizer_updates=0,
                model_hash_unchanged=True,real_calls_not_repeated=True,
                gradient_limit='Static loss and saved finite gradients checked; independent audit did not recompute model forward/backward.')


def calibration_audit(result,parents):
    total_scores=0;seeds=set()
    for group in result['calibration']:
        case=group['parent'];parent=parents[case];target=list(range(len(parent['F'])));ids,F,C,D=derived_view(parent,target,0,0)
        returns=[];log_sums=[]
        for row in group['rows']:
            role=row['role'];saved=read(OUT/'calibration_plans'/f'{case}_{role}.json')
            need(saved['row']==row,'Saved trajectory matches summary')
            audit_plan(saved['plan'],parent,target,ids,0,0)
            remaining=np.ones(len(target),dtype=bool);current=0;raw_L=0.;prefix=[]
            need(len(row['trace'])==len(row['logprobs'])==len(row['raw_route'])-2,'Trajectory lengths')
            for step,event in enumerate(row['trace']):
                node=event['action_local_node'];j=node-1
                legal=C[remaining].any(0)&np.isfinite(D[current,1:])&np.isfinite(D[1:,0])
                need(0<=j<len(ids) and bool(legal[j]),'Saved raw action legal')
                served=np.flatnonzero(remaining&C[:,j]);remaining&=~C[:,j]
                need(event['served_tasks_global']==served.tolist() and event['remaining_tasks_global']==np.flatnonzero(remaining).tolist(),'Saved state transition')
                need(event['action_global_candidate']==int(ids[j]),'Raw global/local ID mapping')
                reward=-float(D[current,node])-.75;raw_L+=float(D[current,node]);current=node;prefix.append(node)
                done=not remaining.any();need(event['done']==done,'Terminal flag')
                correction=0.
                if done:
                    raw_J=raw_L+float(D[current,0])+.75*len(prefix);near(row['raw_J'],raw_J,'Raw terminal J')
                    correction=raw_J-row['J'];reward-=float(D[current,0]);reward+=correction
                near(event['terminal_H_correction'],correction,'Terminal H correction')
                near(event['reward'],reward,'Dense plus terminal reward')
                near(row['rewards'][step],reward,'Reward list matches transition')
                QA['saved_action_transitions_checked']+=1
            need(not remaining.any() and row['raw_route']==[0,*prefix,0],'Complete raw prefix')
            near(row['J'],saved['plan']['objective_length_plus_stops_m'],'Final H J')
            need(row['J']<=row['raw_J']+1e-7,'H nonincrease belongs to H, not NN')
            need([int(ids[n-1]) for n in row['final_route'][1:-1]]==saved['plan']['visit_order_candidate_ids'],'Final local route/global plan join')
            near(row['return_'],sum(row['rewards']),'Saved return sum');near(row['return_'],-row['J'],'Return=-JH')
            expected_seed=int.from_bytes(hashlib.sha256(f'preflight-v1/{case}/{role}/11632026'.encode()).digest()[:8],'little')%(2**63)
            need(row['sampling_seed']==expected_seed and expected_seed not in seeds,'Independent named stream IDs');seeds.add(expected_seed)
            need(all(x is not None and math.isfinite(x) for x in row['logprobs'][1:]),'Finite sampled logprobs')
            if role=='primary':need(row['forced_first'] is None and row['logprobs'][0] is not None,'Primary first action learned')
            else:need(row['forced_first']==prefix[0] and row['logprobs'][0] is None,'Forced first action excluded')
            near(row['logprob_sum'],sum(x for x in row['logprobs'] if x is not None),'Logprob saved accumulation',.01)
            total_scores+=sum(x is not None for x in row['logprobs']);returns.append(row['return_']);log_sums.append(row['logprob_sum'])
        need([r['role'] for r in group['rows']]==['primary','forced_low','forced_high'],'Three fixed roles')
        advantages=[]
        for index,value in enumerate(returns):
            baseline=(sum(returns)-value)/2;near(group['LOO_baseline'][index],baseline,'LOO reference baseline')
            advantages.append(value-baseline);near(group['advantages'][index],value-baseline,'Saved detached advantage value')
        expected_loss=-sum(w*a*lp for w,a,lp in zip((1.,.25,.25),advantages,log_sums))/1.5
        near(group['loss'],expected_loss,'Mixed primary/forced LOO loss',1e-6)
        need(all(math.isfinite(v) and v>=0 for v in group['gradient_norms'].values()),'Saved finite gradient norms')
        need(any(v>0 for v in group['gradient_norms'].values()) and group['model_hash_unchanged'] is True,'Nonzero probe gradient/unchanged model')
    need(total_scores==result['counters']['policy_score_calls']==1063,'Score count from saved sampled actions')


def proposal_metadata():
    receipt=read(OUT/'PROTOCOL_V3_PROPOSAL_RECEIPT.json');p=read(OUT/'PROPOSED_PROTOCOL_V3.json')
    old=read(ROOT/'rc3_global_local/future_seeds_v2.json');new=read(OUT/'PROPOSED_SEEDS_V3.json')
    for key,path in [('proposal_sha256',OUT/'PROPOSED_PROTOCOL_V3.json'),('proposal_registry_sha256',OUT/'PROPOSED_SEEDS_V3.json'),('v2_sha256',ROOT/'rc3_global_local/S2_S3_protocol_v2.json'),('old_registry_sha256',ROOT/'rc3_global_local/future_seeds_v2.json'),('active_pointer_unchanged_sha256',ROOT/'rc3_global_local/FUTURE_DESIGN_STATUS.json')]:
        need(sha(path)==receipt[key],'Proposal binding '+key)
    need(receipt['activated'] is False and receipt['optimizer_authorized'] is False,'Unactivated proposal')
    if 'prior_draft_preserved' in receipt:
        need(sha(OUT/receipt['prior_draft_preserved'])==receipt['prior_draft_sha256'],'Retained previous proposal draft bytes')
    need(p['status']=='PROPOSED_FROZEN_NOT_ACTIVE_NOT_AUTHORIZED','Proposal status')
    need(p['seeds_sha256']==receipt['proposal_registry_sha256'],'Proposal registry hash')
    for key in ('S2_train','S2_development','policy_seeds','sampling_seed','original60_protected'):
        need(new[key]==old[key],'Unchanged metadata bank '+key)
    need(len(new['S1_teachers'])==256 and len(new['S3'])==108,'Proposal counts')
    for previous,current in zip(old['S1_teachers'],new['S1_teachers']):
        need({k:v for k,v in previous.items() if k not in ('width','height','x0','y0','endpoints')}=={k:v for k,v in current.items() if k not in ('width','height','x0','y0','endpoints')},'Declared S1 amendment scope')
    for previous,current in zip(old['S3'],new['S3']):
        need({k:v for k,v in previous.items() if k!='A_condition'}=={k:v for k,v in current.items() if k!='A_condition'},'Declared S3 amendment scope')
    modes={x['parent']:x['mode'] for x in new['S2_train']}
    cells=Counter((modes[x['parent']],x['endpoints']) for x in new['S1_teachers'])
    need(len(cells)==4 and set(cells.values())=={64},'S1 four balanced mode/endpoint cells')
    by_parent={i:set() for i in range(96)}
    for x in new['S1_teachers']:by_parent[x['parent']].add((x['width'],x['height']))
    need(Counter(len(v) for v in by_parent.values())=={3:64,2:32},'S1 parent/size coverage')
    cells=Counter((x['group'],tuple(x['size']),x['A_condition']) for x in new['S3'])
    need(len(cells)==18,'S3 all group/size/A cells')
    need(all(v==(4 if k[0]=='seen_layout_new_tasks' else 8) for k,v in cells.items()),'S3 balanced A cells')
    need(p['resources_summary']['all3seeds_optimizer_updates_max']==3840,'Full update budget includes all controls')
    return dict(activated=False,optimizer_authorized=False,S1_balanced_cells=4,S3_balanced_group_size_A_cells=18,
                changed_distribution_explicit=True,no_actual_target_tasks_generated=True)


def documents(final):
    names=['IMPLEMENTATION_REVIEW.md','EXPERIMENT_REPORT.md','RESULT_TO_CLAIM.md','RESOURCE_ACCOUNTING.json','NEXT_STEP_DECISION.md','GIT_VERSION_AUDIT.json']
    present={name:sha(OUT/name) for name in names if (OUT/name).exists()}
    if final:
        need(len(present)==6,'All six preparation/Git deliverable files present')
        for name in ('IMPLEMENTATION_REVIEW.md','NEXT_STEP_DECISION.md'):
            need('REVISE' in (OUT/name).read_text(encoding='utf-8'),'Formal execution REVISE: '+name)
        status=read(ROOT/'rc3_global_local/FUTURE_DESIGN_STATUS.json')
        need(status['active_protocol']=='S2_S3_protocol_v2.json' and status['status']=='FROZEN_DESIGN_NOT_STARTED','v2 remains active, formal stages unstarted')
        resources=read(OUT/'RESOURCE_ACCOUNTING.json');engineering=read(OUT/'ENGINEERING_RESULT.json')
        totals=resources['scientific_execution_totals']
        for key,value in {'real_engineering_complete_plans':63,'endpoint_native_calls':36,'full_candidate_closed_native_reference_calls':18,'policy_complete_trajectories':9,'environment_actions':1069,'policy_score_calls':1063,'full_common_H_calls':9,'backward_calls':3,'optimizer_steps':0,'formal_S1_training':0,'formal_S2_training':0,'S3_target_task_queries':0,'original60_semantic_reads_or_evaluations':0}.items():
            need(totals[key]==value,'Final resource scientific totals '+key)
        endpoint_dp=reference_dp=0
        for path in (OUT/'baseline_plans').glob('*.json'):
            saved=read(path)
            if path.name.endswith('_endpoint.json'):endpoint_dp+=saved['info']['dp_calls']
            else:reference_dp+=saved['info']['dp_calls']
        need(resources['engineering']['native_DP_calls']=={'endpoint':endpoint_dp,'closed_full_reference':reference_dp,'total':endpoint_dp+reference_dp},'Independent native DP sum')
        need(endpoint_dp==269 and reference_dp==131,'400 actual native DP calls')
        attempts=engineering['native_counters']['endpoint_validation_attempts']+engineering['native_counters']['full_reference_validation_attempts']+engineering['terminal_counters']['validation_attempts']
        need(resources['engineering']['export_and_validation_attempts']==resources['engineering']['validation_passed']==attempts==88,'Actual validation accounting')
        near(resources['physical_geometry_only']['wall_s'],read(OUT/'GEOMETRY_GATE.json')['elapsed_wall_s'],'Geometry timing source')
        near(resources['saved_evidence_verification_and_maintenance']['layout_review_wall_s'],read(OUT/'LAYOUT_GATE_REVIEW.json')['elapsed_s'],'Independent layout verifier timing')
        need(resources['synthetic_unit_tests_separate']['endpoint_suite_all_runs_totals']['actual_native_runs']==61,'Synthetic native entries remain separate')
        git=read(OUT/'GIT_VERSION_AUDIT.json')
        def output(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True,encoding='utf-8').strip()
        need(output('branch','--show-current')==git['working_branch']=='codex/global-local-preflight','Actual review branch')
        need(output('remote','get-url','origin')==git['remote'],'Configured remote')
        for ref,digest in git['preserved_refs'].items():need(output('rev-parse',ref)==digest,'Preserved reference '+ref)
        need(git['payload_parent_commit']==BASE and git['pushed'] is False and git['push_attempts']==0 and git['merged_main'] is False,'Local-only review, no merge')
        need(git['remote_heads_verified_this_task']['refs/heads/main']==git['preserved_refs']['origin/main'],'Root live remote record agrees with fetched ref')
        need('external_postcommit_receipt' in git,'Postcommit SHA receipt path explicit')
    return dict(final_deliverables_checked=final,present_sha256=present,
                git_receipt='GIT_VERSION_AUDIT.json checked against actual branch/config/preserved refs. Root live remote receipt retained; no duplicate child fetch. Final actual commit SHA is necessarily postcommit and will be verified by root in its external receipt/reply; not falsely claimed checked here.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args()
    started=time.perf_counter();cpu_started=time.process_time();QA.clear()
    previous=read(OUT/'PREPARATION_AUDIT.json') if (OUT/'PREPARATION_AUDIT.json').exists() else {}
    result=dict(schema='rc3-independent-saved-preparation-audit-v1',reviewer='/root/preflight_independent_audit',base=BASE,
        status='RUNNING',final_review=args.final,scientific_call_counters=dict(new_scene_generation=0,navigation_preparations=0,planning=0,full_H=0,model_instantiations=0,model_forwards=0,backwards=0,optimizer_updates=0,S3_task_queries=0,original60_reads=0))
    try:
        result['frozen_S0']=frozen_bytes();verify_sources(read(OUT/'GEOMETRY_CHECK_PROTOCOL.json'))
        for name in ('GUARD_TEST_RECEIPT.json','BASELINE_UNIT_RECEIPT.json'):
            receipt=read(OUT/name);need(receipt['failures']==receipt['errors']==0,'Author unit receipt: '+name);verify_sources(receipt)
        result['physical_layouts']=physical_layouts();parents=load_parents();result['saved_engineering']=engineering(parents)
        result['v3_proposal']=proposal_metadata()
        result['documents']=documents(args.final);result['status']='PASS_FINAL' if args.final else 'PASS_SAVED_EVIDENCE_FINAL_DOCS_PENDING'
    except Exception as exc:
        result['status']='FAIL_STOP';result['error']=str(exc);raise
    finally:
        result['verification_accounting']=dict(QA,wall_s=time.perf_counter()-started,cpu_s=time.process_time()-cpu_started)
        result['script_sha256']=sha(__file__)
        history=previous.get('verification_run_history',[])
        for entry in history:
            entry.setdefault('script_sha256',previous.get('script_sha256'))
        history.append(dict(status=result['status'],final_review=args.final,script_sha256=result['script_sha256'],**result['verification_accounting']))
        result['verification_run_history']=history
        (OUT/'PREPARATION_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(status=result['status'],verification=result['verification_accounting'])))


if __name__=='__main__':
    main()
