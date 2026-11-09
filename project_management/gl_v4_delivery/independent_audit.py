"""Independent saved-evidence arithmetic. Imports no experiment implementation.

Torch is used only to deserialize/compare stored tensors; never a policy,
optimizer, autograd execution, scene generator, navigation or planner.
"""
import csv
import hashlib
import json
import time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
HOME=ROOT/'rc3_gl_v4_20261109'


def read(p):return json.loads(Path(p).read_text(encoding='utf8'),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def need(ok,msg):
    if not ok:raise ValueError(msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')
def ah(a):
    a=np.ascontiguousarray(a);return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+a.tobytes()).hexdigest()
def th(state):
    h=hashlib.sha256()
    for n,v in sorted(state.items()):h.update(n.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def chain(folder):
    p=folder/'events.jsonl';previous='0'*64;events=[]
    need(not (folder/'HEAD.pending').exists(),'Pending journal '+str(folder))
    for line in p.read_bytes().splitlines():
        e=json.loads(line);h=e.pop('sha256')
        need(e['sequence']==len(events) and e['previous']==previous and hashlib.sha256(canonical(e)).hexdigest()==h,'Broken event chain')
        e['sha256']=h;events.append(e);previous=h
    head=read(folder/'HEAD.json');need(head['sha256']==previous and head['sequence']==len(events),'Bad HEAD')
    return events


def audit():
    start=time.perf_counter();result={};binding=read(HOME/'SOURCE_BINDING.json')
    for n,h in binding['source_sha256'].items():need(sha(ROOT/n)==h,'Runtime source changed '+n)
    result['bound_source_files']=len(binding['source_sha256'])
    frozen=read(HOME/'FROZEN_PRIOR_EVIDENCE.json')['files']
    for n,v in frozen.items():need(sha(ROOT/n)==v['sha256'],'Old frozen evidence changed '+n)
    result['old_frozen_files_unchanged']=len(frozen)
    index=read(ROOT/'rc3_s3_evaluator/S3_MODEL_INDEX.json');model_sources=0
    for models in index['models'].values():
        for e in models.values():
            for pk,hk in (('path','checkpoint_sha256'),('sidecar_path','sidecar_sha256'),('commit_path','commit_sha256')):
                need(sha(ROOT/e[pk])==e[hk],'Original model source changed');model_sources+=1
    result['old_model_source_files']=model_sources
    reg=read(HOME/'INPUT_REGISTRY.json');bank=read(HOME/'runs/SCIENCE/NEW_TASK_BANK_FROZEN.json')['cases']
    bycase={c['id']:c for c in reg['deployment_cases']+bank};inputs={};navs={}
    for c in reg['deployment_cases']+bank:
        files=c['files'];scene=read(ROOT/files['scene']['path'])
        for n,e in files.items():need(sha(ROOT/e['path'])==e['sha256'],'Input source changed')
        with np.load(ROOT/files['arrays']['path'],allow_pickle=False) as z:F,A=z['F'].copy(),z['A'].copy()
        if c['id'].startswith('GLV4'):
            src=reg['new_development_parents'][c['spec']['parent']];navpath=ROOT/src['navigation_file'];old=read(ROOT/src['scene_file'])
            need(all(scene[k]==v for k,v in old.items() if k not in ('tasks','task_seed','task_generator')),'New task draw changed geometry/context')
            need(scene['task_seed']==c['spec']['task_seed'],'Wrong new task seed')
        else:navpath=ROOT/files['navigation']['path']
        if str(navpath) not in navs:
            with np.load(navpath,allow_pickle=False) as z:navs[str(navpath)]=z['costs'].copy()
        inputs[c['id']]=(scene,F,A,navs[str(navpath)])
    plan_count=0;assignments=0;mask_checks=0;polyline_checks=0;stable={};rows_by_folder={};eventsets={}
    for folder in (HOME/'runs/DEPLOYMENT',HOME/'runs/SCIENCE/DEV216'):
        events=chain(folder);eventsets[folder.name]=events;starts={e['request']['id']:e['request'] for e in events if e['event']=='request_start'}
        rows=[]
        for e in events:
            if e['event']!='request_complete':continue
            need(sha(ROOT/e['artifact_path'])==e['artifact_sha256'],'Result artifact altered');r=read(ROOT/e['artifact_path']);q=r['request']
            need(starts[q['id']]==q and r['valid'] and r['plan']['validation']['valid_model_plan'],'Unbilled/invalid plan')
            c=bycase[q['case_id']];scene,F,A,D=inputs[c['id']];p=r['plan'];ids=r['source']['global_task_ids'];cs=r['source']['global_candidate_ids']
            need(ah(F)==r['source']['F_sha256'] and ah(A)==r['source']['A_sha256'] and ah(D)==r['source']['D_sha256'],'Content digest mismatch')
            w=p['window'];tables={t['id'] for t in scene['tables'] if w['x0']<=t['ix']<w['x0']+w['width'] and w['y0']<=t['iy']<w['y0']+w['height']}
            target=[t['id'] for t in scene['tasks'] if t['table_id'] in tables];need(target==ids==p['target_task_ids_global'],'Window target changed')
            entry,exit_=p['entry_global_node'],p['exit_global_node']
            if c['spec']['endpoints']=='closed_depot':need(entry==exit_==0,'Wrong closed endpoints')
            else:
                nodes=np.flatnonzero(np.isfinite(D[0,1:])&np.isfinite(D[1:,0]))+1
                need(entry==int(nodes[0]) and exit_==int(nodes[-1]),'Wrong open endpoints')
            effective=F&A;legal=np.flatnonzero(effective[ids].any((0,2))&np.isfinite(D[entry,1:])&np.isfinite(D[1:,exit_]))
            need(legal.tolist()==cs,'Legal boundary candidate lost or foreign candidate added')
            C=effective[np.asarray(ids)[:,None],np.asarray(cs)[None,:]].any(2)
            for route,prefix in ((r['raw_route'],'raw_'),(r['route'],'')):
                need(route[0]==route[-1]==0 and len(set(route[1:-1]))==len(route)-2,'Bad/repeated route')
                global_nodes=[entry,*[cs[n-1]+1 for n in route[1:-1]],exit_]
                length=sum(float(D[x,y]) for x,y in zip(global_nodes,global_nodes[1:]));k=len(route)-2
                need(np.isfinite(length) and abs(length-r['metrics'][prefix+'L'])<1e-6 and k==r['metrics'][prefix+'K'] and abs(length+.75*k-r['metrics'][prefix+'J'])<1e-6,'J/L/K independent recomputation differs')
                need(C[:,np.asarray(route[1:-1],dtype=int)-1].any(1).all(),'Uncovered target')
            served=[]
            for a in p['assignments']:
                t,j,arm=a['task_id'],a['candidate_id'],{'L':0,'R':1}.get(a['arm'])
                need(t in ids and j in cs and j in p['visit_order_candidate_ids'] and arm is not None and F[t,j,arm] and A[t,j,arm],'Wrong same-arm service')
                served.append(t);assignments+=1
            need(len(served)==len(ids) and set(served)==set(ids) and not p['unserved_task_ids'],'Repeated/missing assignment')
            pts=np.asarray(p['route_polyline'],float);length=float(np.linalg.norm(np.diff(pts,axis=0),axis=1).sum())
            need(abs(length-r['metrics']['L'])<1e-6,'Polyline length differs');polyline_checks+=1
            if r['trace']:
                need([e['action_local_node'] for e in r['trace']]==r['raw_route'][1:-1],'Raw route/trace differs')
                remaining=np.ones(len(ids),bool);current=entry
                for event in r['trace']:
                    mask=C[remaining].any(0)&np.isfinite(D[current,np.asarray(cs)+1])&np.isfinite(D[np.asarray(cs)+1,exit_])
                    need((np.flatnonzero(mask)+1).tolist()==event['mask_local_nodes'] and ah(mask)==event['mask_sha256'],'Actual greedy mask differs')
                    node=event['action_local_node'];need(mask[node-1],'Illegal greedy action')
                    remaining &= ~C[:,node-1];current=cs[node-1]+1;mask_checks+=1
                need(not remaining.any(),'Greedy trace not complete')
            if folder.name=='DEPLOYMENT':
                key=(q['case_id'],q['role']);value=(r['raw_route'],r['route'],r['metrics'],r['trace'],p['assignments'])
                need(key not in stable or stable[key]==value,'Deployment lifecycle changed numeric decisions')
                stable[key]=value
            rows.append((r,e));plan_count+=1
        need(len(rows)==len(starts) and not any(e['event']=='request_error' for e in events),'Incomplete requests')
        rows_by_folder[folder.name]=rows
    need(plan_count==612 and len(stable)==12,'Wrong plan grid')
    result.update(plans_verified=plan_count,same_arm_assignments_verified=assignments,greedy_masks_recomputed=mask_checks,polylines_recomputed=polyline_checks)
    initial_hashes=[];update_count=0;terminal_requests=0;ab={};max_clip_error=0.
    for arm in ('A128','B128','C128'):
        folder=HOME/'runs/SCIENCE'/arm;events=chain(folder);full=read(folder/'COMPLETE.json')
        need(sum(e['event']=='update_start' for e in events)==sum(e['event']=='update_complete' for e in events)==128,'Update predebit mismatch')
        terminals=sum(e['event']=='terminal_request_predebit' for e in events);terminal_requests+=terminals
        need(terminals==(0 if arm=='C128' else 1536),'Wrong terminal debit count')
        init=torch.load(folder/'checkpoints/update_0000.pt',map_location='cpu',weights_only=False)
        need(not init['optimizer']['state'],'Optimizer inherited moments');initial_hashes.append(th(init['model']))
        need(initial_hashes[-1]==index['models']['11622026']['SL256']['model_tensor_sha256'],'Not shared SL256')
        rows=[]
        for u in range(1,129):
            d=read(folder/'updates'/f'update_{u:04}.json');rows.append(d);artifact=d['full_gradient_and_weight_artifact']
            need(sha(ROOT/artifact['path'])==artifact['sha256'],'Gradient/parameter artifact altered')
            s=torch.load(ROOT/artifact['path'],map_location='cpu',weights_only=True)
            need(th(s['model'])==d['model_tensor_sha256']==s['model_tensor_sha256'],'Parameter endpoint mismatch')
            norms=[]
            for name,g in s['gradient_before_clip'].items():
                h=s['gradient_after_clip'][name]
                if g is None:need(h is None,'Absent gradient appeared');continue
                gn=g.numpy().astype(np.float64);norms.append(float(np.linalg.norm(gn)))
                need(np.isfinite(gn).all() and np.isfinite(h.numpy()).all(),'Nonfinite saved gradient')
            norm=float(np.linalg.norm(norms));need(abs(norm-d['gradient_norm_before_clip'])<=max(1e-5,1e-5*norm),'Gradient norm record differs')
            coef=min(1.,1./(d['gradient_norm_before_clip']+1e-6))
            for name,g in s['gradient_before_clip'].items():
                if g is None:continue
                error=float(np.max(np.abs(s['gradient_after_clip'][name].numpy()-g.numpy()*coef)))
                max_clip_error=max(max_clip_error,error);need(error<2e-6,'Stored clipping differs from norm1 recipe')
            need(d['gradient_norm_after_clip']<=1.00001,'Gradient not clipped')
            if arm!='C128':
                for inst in d['instances']:
                    rr=inst['rows'];returns=np.array([x['return_'] for x in rr]);adv=returns-(returns.sum()-returns)/2
                    need(np.max(np.abs(adv-np.asarray(inst['advantages'])))<1e-9 and inst['mixture_weights']==[1.,.25,.25],'LOO/weight mismatch')
                    expected=sum(-w*a*x['logprob_sum'] for w,a,x in zip([1.,.25,.25],adv,rr))/1.5
                    need(abs(expected-inst['loss'])<1e-6,'Wrong objective')
                    for i,x in enumerate(rr):
                        need(abs(x['return_']+x['J'])<1e-7,'Wrong terminal reward')
                        if x['actions']:
                            if arm=='B128' or i==0:need(x['logprobs'][0] is not None and x['forced_first'] is None,'Natural first action missing')
                            else:need(x['logprobs'][0] is None and x['forced_first'] is not None,'A forcing changed')
            else:need(all(i['teacher_or_H_queries']==0 for i in d['instances']),'C queried an oracle')
            update_count+=1
        if arm!='C128':ab[arm]=rows
        end=torch.load(folder/'checkpoints/update_0128.pt',map_location='cpu',weights_only=False)
        need(th(end['model'])==full['tensor_sha256'],'Wrong fixed endpoint')
        need(all(g['lr']==1e-4 and list(g['betas'])==[.9,.999] and g['eps']==1e-8 and g['weight_decay']==0 for g in end['optimizer']['param_groups']),'Optimizer config changed')
        if arm!='C128':
            st=full['ledger']['terminal']['RL'];need(st['terminal_requests']==st['terminal_cache_hits']+st['terminal_attempts']==1536,'Terminal hit/attempt accounting differs')
    need(all(a['specs']==b['specs']==reg['training_batches'][i] and a['view_identities']==b['view_identities'] for i,(a,b) in enumerate(zip(ab['A128'],ab['B128']))),'A/B schedule diverged')
    need(update_count==384 and terminal_requests==3072,'Wrong training budget')
    result.update(optimizer_updates_audited=update_count,terminal_predebits=terminal_requests,identical_A_B_input_views=512,max_clipping_array_error=max_clip_error,
        fresh_Adam_initial_states=3,common_initial_tensor=initial_hashes[0])
    actor_count=0;parameter_tensors=0
    for folder in (HOME/'actors',HOME/'endpoint_actors'):
        for path in folder.glob('*.pt'):
            actor=torch.load(path,map_location='cpu',weights_only=True)
            need(set(actor)=={'schema','architecture','model','model_tensor_sha256','provenance'},'Actor contains training/cache state')
            src=actor['provenance'];need(sha(ROOT/src['training_checkpoint_path'])==src['training_checkpoint_sha256'],'Actor provenance changed')
            full=torch.load(ROOT/src['training_checkpoint_path'],map_location='cpu',weights_only=False)
            need(set(full['model'])==set(actor['model']) and th(full['model'])==th(actor['model'])==actor['model_tensor_sha256'],'Actor tensor identity differs')
            for name,x in actor['model'].items():
                y=full['model'][name];need(x.shape==y.shape and x.dtype==y.dtype and torch.equal(x,y),'Actor precision/value changed');parameter_tensors+=1
            actor_count+=1
    need(actor_count==5,'Wrong deployment/fixed endpoint package count')
    result.update(actor_packages_exact=actor_count,actor_parameter_tensors_compared=parameter_tensors)
    costs={(r['case_id'],r['request']['study_method']):r['metrics'] for r,e in rows_by_folder['DEV216']}
    original=read(HOME/'results/MECHANISM_COMPARISONS.json')['comparisons'];independent=[];maxdiff=0.
    parents=sorted({c['physical_key'] for c in bank});rng=np.random.Generator(np.random.PCG64(11632026));indices=rng.integers(3,size=(20000,3))
    for metric in ('raw_J','J'):
        for control in ('A128','SL256','C128','Fixed3','Detour16'):
            gains=[]
            for p in parents:
                ids=[c['id'] for c in bank if c['physical_key']==p];a=np.mean([costs[c,'B128'][metric] for c in ids]);b=np.mean([costs[c,control][metric] for c in ids]);gains.append((b-a)/b)
            means=np.asarray(gains)[indices].mean(1);v=dict(metric=metric,control=control,mean=float(np.mean(gains)),lower95=float(np.quantile(means,.025)),upper95=float(np.quantile(means,.975)))
            match=next(x for x in original if x['metric']==metric and x['control']==control)
            for k in ('mean','lower95','upper95'):maxdiff=max(maxdiff,abs(v[k]-match[k]));need(abs(v[k]-match[k])<1e-12,'Independent cluster statistic differs')
            independent.append(v)
    with (HOME/'DEPLOYMENT_BENCHMARK.csv').open(encoding='utf8',newline='') as f:csvrows=list(csv.DictReader(f))
    need(len(csvrows)==396,'Deployment CSV omitted rows')
    groups=defaultdict(list)
    for r in csvrows:groups[r['role'],r['mode'],r['navigation']].append(float(r['complete_time_s']))
    with (HOME/'results/DEPLOYMENT_TIMING.csv').open(encoding='utf8',newline='') as f:
        for r in csv.DictReader(f):
            ts=groups[r['role'],r['mode'],r['navigation']]
            need(abs(np.median(ts)-float(r['median_s']))<1e-12 and abs(np.quantile(ts,.95)-float(r['p95_s']))<1e-12,'Timing statistics differ')
    for phase in ('DEPLOYMENT','SCIENCE'):
        p=read(HOME/'runs'/phase/'PHASE_FINAL.json');need(p['status']=='COMPLETE' and p['wall_s']<p['cap_s'],'Unclosed/over-budget stage')
    result.update(status='PASS',independent_mechanism_comparisons=independent,max_cluster_stat_difference=maxdiff,
        source_binding_sha256=sha(HOME/'SOURCE_BINDING.json'),new_scientific_calls=0,policy_materializations=0,optimizer_step_calls=0,
        wall_s=time.perf_counter()-start,review_method='Separate read-only implementation and process; no experimental implementation imports or policy forward. Not external human peer review.',
        limitations=['One optimization seed and three already exposed DEV physical layouts; new tasks only.',
            'DEV216 saves actual first saved-navigation artifact load within its early request timing. Those latencies are descriptive;396 paired deployment grid alone is used for packaging-speed claims.',
            'Engineering packaging speed is not DRL quality gain; H improvement is separate. Old S3 exposure and two failed slots unchanged.'])
    write(HOME/'INDEPENDENT_AUDIT.json',result);print({k:result[k] for k in ('status','plans_verified','optimizer_updates_audited','old_frozen_files_unchanged','wall_s')})


if __name__=='__main__':audit()
