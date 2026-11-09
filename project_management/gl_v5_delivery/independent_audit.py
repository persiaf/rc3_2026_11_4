"""Independent arithmetic on saved arrays/tensors; no experiment implementation."""
import csv
import json
import math
import time
from collections import Counter,OrderedDict
from pathlib import Path
import numpy as np
import torch
from project_management.gl_v4_delivery.independent_audit import read,need,sha,ah,th,chain,write

ROOT=Path(__file__).resolve().parents[2];HOME=ROOT/'rc3_gl_v5_20261109'


class Saved:
    def __init__(self):self.inputs={};self.nav=OrderedDict()
    def get(self,scene_path,arrays_path,nav_path):
        k=str(scene_path)
        if k not in self.inputs:
            with np.load(arrays_path,allow_pickle=False) as z:F,A=z['F'].copy(),z['A'].copy()
            self.inputs[k]=(read(scene_path),F,A)
        if str(nav_path) not in self.nav:
            with np.load(nav_path,allow_pickle=False) as z:self.nav[str(nav_path)]=z['costs'].copy()
            if len(self.nav)>6:self.nav.popitem(last=False)
        else:self.nav.move_to_end(str(nav_path))
        return (*self.inputs[k],self.nav[str(nav_path)])


def verify_plan(spec,source,plan,raw,route,metrics,trace=(),description=None):
    scene,F,A,D=source;w=plan['window'];ids=plan['target_task_ids_global']
    tables={t['id'] for t in scene['tables'] if w['x0']<=t['ix']<w['x0']+w['width'] and w['y0']<=t['iy']<w['y0']+w['height']}
    need([t['id'] for t in scene['tasks'] if t['table_id'] in tables]==ids,'Window tasks changed')
    need(w=={k:spec[k] for k in ('x0','y0','width','height')},'Declared window changed')
    entry,exit_=plan['entry_global_node'],plan['exit_global_node']
    if spec['endpoints']=='closed_depot':need(entry==exit_==0,'Wrong closed endpoints')
    else:
        connected=np.flatnonzero(np.isfinite(D[0,1:])&np.isfinite(D[1:,0]))+1
        need(entry==int(connected[0]) and exit_==int(connected[-1]),'Wrong open endpoints')
    effective=F&A;cs=np.flatnonzero(effective[ids].any((0,2))&np.isfinite(D[entry,1:])&np.isfinite(D[1:,exit_]))
    C=effective[np.asarray(ids)[:,None],cs[None,:]].any(2)
    need(C.any(1).all(),'Infeasible target')
    if description:
        need(description['task_ids_global']==ids and description['candidate_ids_global']==cs.tolist(),'Descriptor lost legal/boundary candidates')
        need(ah(C)==description['coverage_sha256'],'Coverage identity differs')
    for r,prefix in ((raw,'raw_'),(route,'')):
        need(r[0]==r[-1]==0 and len(set(r[1:-1]))==len(r)-2,'Bad route')
        need(all(1<=n<=len(cs) for n in r[1:-1]),'Candidate index out of range')
        nodes=[entry,*[int(cs[n-1])+1 for n in r[1:-1]],exit_]
        L=sum(float(D[a,b]) for a,b in zip(nodes,nodes[1:]));K=len(r)-2
        need(np.isfinite(L) and abs(L-metrics[prefix+'L'])<1e-6 and K==metrics[prefix+'K'] and abs(L+.75*K-metrics[prefix+'J'])<1e-6,'Raw/final JLK differs')
        need(C[:,np.asarray(r[1:-1],int)-1].any(1).all(),'Incomplete coverage')
    served=[]
    for a in plan['assignments']:
        t,j,arm=a['task_id'],a['candidate_id'],{'L':0,'R':1}.get(a['arm'])
        need(t in ids and j in cs and j in plan['visit_order_candidate_ids'] and arm is not None and F[t,j,arm] and A[t,j,arm],'Wrong same-arm F&A')
        served.append(t)
    need(len(served)==len(ids) and set(served)==set(ids) and not plan['unserved_task_ids'],'Repeated/missing task assignment')
    need(plan['validation']['valid_model_plan'] and plan['validation']['all_tasks_served'],'Runtime validation failed')
    pts=np.asarray(plan['route_polyline'],float)
    need(abs(np.linalg.norm(np.diff(pts,axis=0),axis=1).sum()-metrics['L'])<1e-6,'Polyline length differs')
    if trace:
        remaining=np.ones(len(ids),bool);current=entry
        need([e['action_local_node'] for e in trace]==raw[1:-1],'Trace route differs')
        for e in trace:
            mask=C[remaining].any(0)&np.isfinite(D[current,cs+1])&np.isfinite(D[cs+1,exit_])
            need((np.flatnonzero(mask)+1).tolist()==e['mask_local_nodes'],'Actual mask changed')
            n=e['action_local_node'];need(mask[n-1],'Illegal sampled/greedy action')
            remaining &= ~C[:,n-1];current=int(cs[n-1])+1
        need(not remaining.any(),'Trace leaves tasks unserved')
    return len(ids),len(trace)


def audit():
    begin=time.perf_counter();out={};binding=read(HOME/'SOURCE_BINDING.json')
    for n,h in binding['source_sha256'].items():need(sha(ROOT/n)==h,'Frozen source changed '+n)
    frozen=read(HOME/'FROZEN_PRIOR_EVIDENCE.json')['files']
    for n,v in frozen.items():need(sha(ROOT/n)==v['sha256'],'Old evidence changed '+n)
    out.update(bound_sources=len(binding['source_sha256']),old_frozen_files_unchanged=len(frozen))
    reg=read(HOME/'INPUT_REGISTRY.json');folder=HOME/'runs/SCIENCE';G=folder/'G128';events=chain(G)
    need(sum(e['event']=='update_start' for e in events)==sum(e['event']=='update_complete' for e in events)==128,'Update journal incomplete')
    need(sum(e['event']=='terminal_request_predebit' for e in events)==1536 and sum(e['event']=='reference_request_predebit' for e in events)==512,'Request predebits wrong')
    need(not any(e['event'].endswith('_error') for e in events),'Execution failure hidden')
    saved=Saved();views={};sample_traces={};initial=torch.load(G/'checkpoints/update_0000.pt',map_location='cpu',weights_only=False)
    need(not initial['optimizer']['state'],'Not fresh Adam')
    need(th(initial['model'])==reg['actors']['SL256']['model_tensor_sha256'],'Wrong exact SL starting weights')
    previous=initial['model'];max_clip=0.;max_adam=0.;assignments=0;masks=0;samples=0
    for u in range(1,129):
        d=read(G/'updates'/f'update_{u:04}.json');b=read(ROOT/'rc3_gl_v4_20261109/runs/SCIENCE/B128/updates'/f'update_{u:04}.json')
        need(d['specs']==b['specs']==reg['training_batches'][u-1] and d['view_identities']==b['view_identities'],'G/B schedule/view changed')
        art=d['full_gradient_and_weight_artifact'];need(sha(ROOT/art['path'])==art['sha256'],'Gradient artifact changed')
        s=torch.load(ROOT/art['path'],map_location='cpu',weights_only=True);need(th(s['model'])==d['model_tensor_sha256'],'Update weight identity differs')
        norm=math.sqrt(sum(float(g.double().square().sum()) for g in s['gradient_before_clip'].values() if g is not None))
        need(abs(norm-d['gradient_norm_before_clip'])<max(1e-5,1e-5*norm),'Gradient norm differs')
        coef=min(1.,1/(d['gradient_norm_before_clip']+1e-6))
        for n,g in s['gradient_before_clip'].items():
            if g is None:continue
            need(bool(torch.isfinite(g).all()) and bool(torch.isfinite(s['gradient_after_clip'][n]).all()),'Nonfinite gradient')
            error=float((s['gradient_after_clip'][n]-g*coef).abs().max());max_clip=max(max_clip,error);need(error<2e-6,'Gradient clip changed')
        delta=math.sqrt(sum(float((s['model'][n].double()-previous[n].double()).square().sum()) for n in previous))
        max_adam=max(max_adam,abs(delta-d['Adam_step_l2']));need(abs(delta-d['Adam_step_l2'])<1e-10,'Adam actual step differs')
        need(d['Adam_internal_steps']==[u]*len(d['Adam_internal_steps']),'Adam step count differs')
        previous=s['model']
        for slot,(spec,i,bi) in enumerate(zip(d['specs'],d['instances'],b['instances'])):
            bank=ROOT/'rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01/data'
            f=bank/'task_draws'/f"{spec['map_seed']}_{spec['task_seed']}_A{spec['A_condition']}";receipt=read(f/'receipt.json')
            nav=bank/'navigation'/(receipt['map_key']+'.npz');views[i['problem']]=(spec,f/'scene.json',f/'arrays.npz',nav)
            rr=i['rows'];ref=i['reference'];need(sha(ROOT/ref['artifact_path'])==ref['artifact_sha256'],'Reference artifact changed')
            refart=read(ROOT/ref['artifact_path']);need(refart['problem']==i['problem'] and refart['result']['J']==ref['J'],'Not same input reference')
            need(ref['model_identity']['tensor_sha256']==reg['actors']['SL256']['model_tensor_sha256'] and ref['model_identity']['checkpoint_role']=='SL256','Reference promoted or wrong')
            need(refart['policy_sampling_RNG_used'] is False,'Reference used policy RNG')
            cost=np.array([x['J'] for x in rr]);adv=ref['J']-cost
            need(np.max(np.abs(adv-np.asarray(i['advantages'])))<1e-8 and i['mixture_weights']==[1.,.25,.25],'Fixed baseline/sign/weight changed')
            need(abs(sum(-w*a*x['logprob_sum'] for w,a,x in zip([1.,.25,.25],adv,rr))/1.5-i['loss'])<1e-6,'Wrong loss')
            for row,brow in zip(rr,bi['rows']):
                need(row['forced_first'] is None and row['sampling_seed']==brow['sampling_seed'] and all(x is not None for x in row['logprobs']),'First likelihood or RNG convention changed')
                accumulated=np.float32(0)
                for value in row['logprobs']:accumulated=np.float32(accumulated+np.float32(value))
                need(float(accumulated)==row['logprob_sum'] and abs(row['return_']+row['J'])<1e-7,'Wrong original float32 complete likelihood/reward')
                sample_traces[(i['problem'],tuple(row['raw_route']))]=[dict(action_local_node=n,mask_local_nodes=m) for n,m in zip(row['actions'],row['masks_local_nodes'])]
                samples+=1
    actual_plans=0
    for p in sorted((G/'terminal_plans').glob('*.json')):
        x=read(p);v=x['problem'];spec,scene,arrays,nav=views[v['identity']]
        trace=sample_traces[(v['identity'],tuple(x['result']['raw_route']))]
        a,m=verify_plan(spec,saved.get(scene,arrays,nav),x['result']['plan'],x['result']['raw_route'],x['result']['route'],x['result'],trace,description=v)
        assignments+=a;masks+=m;actual_plans+=1
    for p in sorted((G/'reference/greedy_results').glob('*.json')):
        x=read(p);spec,scene,arrays,nav=views[x['problem']]
        a,m=verify_plan(spec,saved.get(scene,arrays,nav),x['result']['plan'],x['raw_route'],x['result']['route'],x['result'],x['trace'])
        assignments+=a;masks+=m
    bank=read(folder/'NEW_TASK_BANK_FROZEN.json')['cases'];cases={c['id']:c for c in bank};evalevents=chain(folder/'DEV432');costs={};native=0
    actors=read(folder/'DEV432/ACTOR_INDEX.json')['models']['11622026']
    for role,entry in actors.items():
        need(entry['role']==entry['provenance']['role']==role,'Actor scientific role mismatch')
        need(sha(ROOT/entry['path'])==entry['checkpoint_sha256'],'Actor bytes changed')
        actor=torch.load(ROOT/entry['path'],map_location='cpu',weights_only=True);src=actor['provenance']
        need(sha(ROOT/src['training_checkpoint_path'])==src['training_checkpoint_sha256'],'Training source changed')
        full=torch.load(ROOT/src['training_checkpoint_path'],map_location='cpu',weights_only=False)
        need(th(actor['model'])==th(full['model'])==entry['model_tensor_sha256'],'Package tensors differ')
        need(all(torch.equal(v,full['model'][n]) and v.dtype==full['model'][n].dtype for n,v in actor['model'].items()),'Package precision/value changed')
    for e in evalevents:
        if e['event']!='request_complete':continue
        need(sha(ROOT/e['artifact_path'])==e['artifact_sha256'],'Evaluation artifact changed')
        x=read(ROOT/e['artifact_path']);role=x['study_method'];q=x['request'];c=cases[x['case_id']]
        need(role==q['study_method']==x['plan']['method'],'Wrong scientific method label')
        if role in actors:
            a=actors[role];need(x['weight']==a and x['checkpoint_role']==role==x['training_weight_role'],'Role/source/default mismatch')
            need(x['actor_file']==a['path'] and x['tensor_sha256']==a['model_tensor_sha256'] and x['fixed_update']==(256 if role=='SL256' else 128),'Endpoint identity mismatch')
            need(x['pipeline_counters']['full_H_calls']==1,'Missing/extra common H')
        else:need(x['weight'] is None and x['checkpoint_role'] is None and x['pipeline_counters']['full_H_calls']==0,'Native gained model/H');native+=1
        src=reg['new_development_parents'][c['spec']['parent']];files=c['files']
        for v in files.values():need(sha(ROOT/v['path'])==v['sha256'],'New input changed')
        source=saved.get(ROOT/files['scene']['path'],ROOT/files['arrays']['path'],ROOT/src['navigation_file'])
        a,m=verify_plan(c['spec'],source,x['plan'],x['raw_route'],x['route'],x['metrics'],x['trace']);assignments+=a;masks+=m
        costs[(x['physical_key'],x['case_id'],role)]=x['metrics']
    need(len(costs)==432 and native==144 and samples==1536,'Incomplete evaluation/training')
    keys=sorted(p['physical_key'] for p in reg['new_development_parents']);modes={p['physical_key']:p['mode'] for p in reg['new_development_parents']}
    rng=np.random.Generator(np.random.PCG64(11642026));draw=np.column_stack([rng.choice([i for i,k in enumerate(keys) if modes[k]==mode],size=(20000,6)) for mode in ('mixed','blocked')])
    summary=read(HOME/'results/SUMMARY.json');diff=0.
    for c in summary['comparisons']:
        gains=[]
        for k in keys:
            gr=[m[c['metric']] for (parent,case,role),m in costs.items() if parent==k and role=='G128'];cr=[m[c['metric']] for (parent,case,role),m in costs.items() if parent==k and role==c['control']]
            need(len(gr)==len(cr)==6,'Wrong parent cluster size');gains.append((np.mean(cr)-np.mean(gr))/np.mean(cr))
        gains=np.array(gains);ci=np.quantile(gains[draw].mean(1),[.025,.975],method='linear')
        err=max(abs(gains.mean()-c['mean']),abs(ci[0]-c['lower95']),abs(ci[1]-c['upper95']));diff=max(diff,float(err));need(err<1e-12,'Independent bootstrap differs')
    final=read(folder/'PHASE_FINAL.json');need(final['status']=='COMPLETE' and final['wall_s']<=7200,'Bad stage budget')
    out.update(status='PASS',sample_requests_checked=1536,reference_requests_checked=512,evaluation_requests_checked=432,
        actual_sample_H_plans=actual_plans,same_arm_assignments_checked=assignments,greedy_masks_recomputed=masks,
        updates_audited=128,matched_B_views=512,frozen_reference_identity=reg['actors']['SL256']['model_tensor_sha256'],
        max_clipping_error=max_clip,max_Adam_step_error=max_adam,max_statistics_error=diff,
        all4_actor_packages_equal_to_complete_sources=True,new_scientific_calls=0,model_materializations=0,optimizer_steps=0,
        wall_s=time.perf_counter()-begin,audit_source_sha256=sha(Path(__file__)),review_scope='Separate saved-array/tensor arithmetic; no scientific implementation import or new policy/optimizer/nav/native/H calls')
    write(HOME/'INDEPENDENT_AUDIT.json',out);print(out)


if __name__=='__main__':
    started=time.perf_counter()
    try:audit()
    except BaseException as exc:
        import traceback
        write(HOME/'READ_ONLY_AUDIT_FAILURE_02.json',dict(error=repr(exc),traceback=traceback.format_exc(),
            exact_wall_s=time.perf_counter()-started,source_sha256=sha(Path(__file__)),new_scientific_calls=0))
        raise
