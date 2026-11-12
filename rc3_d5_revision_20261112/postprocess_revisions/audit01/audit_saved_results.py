"""Read-only independent arithmetic/source audit; never plan or train."""
import json,gzip,collections,time,math,hashlib,csv,base64
from pathlib import Path
import numpy as np
from .common import ROOT,HOME,read,sha,put,unarchive,gzread

class ReadBudget:
    def __init__(self):self.counts=collections.Counter();self.start=time.perf_counter()
    def check(self):
        if time.perf_counter()-self.start>7200:raise TimeoutError('Independent saved-output audit time limit')

def main():
    start=time.perf_counter();b=ReadBudget()
    from .data import Pool
    pool=Pool(b);manifest=read(HOME/'FULL_TEACHER_MANIFEST.json');checks=0;full_physical=set()
    if len(manifest['records'])!=96:raise ValueError('Incomplete full-teacher count')
    for tid,entry in manifest['records'].items():
        b.check()
        if sha(ROOT/entry['path'])!=entry['sha256']:raise ValueError('Full label file digest')
        label=gzread(ROOT/entry['path']);p,_=pool.get(tid);full_physical.add(entry['physical_key'])
        if label['physical_key']!=entry['physical_key']:raise ValueError('Full physical-parent binding')
        if p.identity!=label['problem_identity']:raise ValueError('Teacher problem source')
        for prefix,route in [('raw_',label['raw_route']),('',label['route'])]:
            route=np.asarray(route,dtype=int);L=float(p.D[route[:-1],route[1:]].sum());K=len(route)-2;J=L+.75*K
            for name,value in [('L',L),('K',K),('J',J)]:
                if abs(value-label['metrics'][prefix+name])>1e-7:raise ValueError('Full teacher cost differs from saved D/route')
                checks+=1
        remaining=np.ones(len(p.C),dtype=bool);current=0
        for node in label['actions_local_nodes']:
            if not p.C[remaining,node-1].any() or not np.isfinite(p.D[current,node]) or not np.isfinite(p.D[node,0]):raise ValueError('Independent current-mask witness')
            remaining&=~p.C[:,node-1];current=node;checks+=1
        if remaining.any():raise ValueError('Full label misses targets')
        for key in ('raw_plan','plan'):
            plan=label[key];assigned=[]
            for item in plan['assignments']:
                arm=0 if item['arm']=='L' else 1;i=item['task_id'];j=item['candidate_id']
                if not (p.parent.F[i,j,arm] and p.parent.A[i,j,arm]):raise ValueError('Saved service violates same-arm F&A')
                assigned.append(i);checks+=1
            if sorted(assigned)!=sorted(p.task_ids.tolist()) or not plan['validation']['valid_model_plan']:raise ValueError('Saved coverage/certificate')
        if label['candidate_ids']!=p.candidate_ids.tolist() or label['task_ids']!=p.task_ids.tolist():raise ValueError('Candidate/task order changed')
    if len(full_physical)!=96:raise ValueError('Full teachers do not cover96 unique physical parents')
    original=read(HOME/'ORIGINAL_LOCAL_TEACHERS.json')
    for file in original['files']:
        if sha(ROOT/file['path'])!=file['sha256']:raise ValueError('Original256 teacher bytes changed')
        checks+=1
    exposure=read(HOME/'SCHEDULE_EXPOSURE.json');uses=[]
    for seed,source in exposure['schedules'].items():
        if sha(ROOT/source['path'])!=source['sha256']:raise ValueError('Schedule digest')
        batches=read(ROOT/source['path'])['batches'];u=collections.Counter()
        for op,batch in enumerate(batches):
            if len(batch)!=8 or not all(t.startswith('F') for t in batch[:4]):raise ValueError('Batch shape/full ratio')
            sizes=collections.Counter((pool.local[t]['spec']['width'],pool.local[t]['spec']['height']) for t in batch[4:])
            if [sizes[x] for x in ((4,3),(6,6),(8,8))]!=((2,1,1),(1,2,1),(1,1,2))[op%3]:raise ValueError('Local rotation')
            u.update(batch);checks+=1
        if len(u)!=352 or sum(u.values())!=32768 or dict(u)!=exposure['uses_by_seed'][seed]:raise ValueError('Unique tasks/exposure count')
        uses.append(u)
    if not uses[0]==uses[1]==uses[2]:raise ValueError('Seed exposure multiset changed')
    # Verify actual PG batch IDs, returns, masks, and fixed-source baseline.
    reference=read(HOME/'REFERENCE_MANIFEST.json');reference_J={}
    for tid,entry in reference['records'].items():
        if sha(ROOT/entry['path'])!=entry['sha256']:raise ValueError('Frozen reference bytes')
        rec=gzread(ROOT/entry['path']);reference_J[tid]=rec['metrics']['J']
        if rec['actor_tensor_sha256']!='1817f58aa6506456ff12f0c5fe0a914e156aef8af319f4687dde90782f55a63e':raise ValueError('Reference is not exact SL256')
        checks+=1
    streams={};teacher_lengths={}
    for folder in sorted((HOME/'streams').glob('*/*')) if (HOME/'streams').exists() else []:
        summary=folder/'STREAM_SUMMARY.json'
        if not summary.exists():continue
        data=read(summary);seed=str(data['seed']);batches=read(ROOT/exposure['schedules'][seed]['path'])['batches'];role=data['role'];actual=[];online={}
        if role!='D5-SL':
            with (folder/'REQUEST_POINTERS.jsonl').open(encoding='utf8') as f:
                for line in f:
                    b.check()
                    pointer=json.loads(line);record=unarchive(pointer['record']);parts=record['request_id'].split(':');op=int(parts[2]);slot=int(parts[3])
                    if record['task_id']!=batches[(op-1)%4096][slot] or record['method']!=role:raise ValueError('Actual PG schedule/role differs')
                    m=record['metrics']
                    if abs(record['complete_return']+m['J'])>1e-7 or abs(record['reward_sum']+m['J'])>1e-7:raise ValueError('Online return has truncation/renormalization')
                    for pre in ('raw_',''):
                        if abs(m[pre+'J']-m[pre+'L']-.75*m[pre+'K'])>1e-7:raise ValueError('Online cost component units')
                    p,_=pool.get(record['task_id'])
                    if p.identity!=record['problem_identity']:raise ValueError('Online problem source')
                    for pre,key in [('raw_','raw_route'),('','route')]:
                        route=np.asarray(record[key],dtype=int);L=float(p.D[route[:-1],route[1:]].sum());K=len(route)-2
                        for metric,value in [('L',L),('K',K),('J',L+.75*K)]:
                            if abs(value-m[pre+metric])>1e-7:raise ValueError('Online saved-route arithmetic')
                            checks+=1
                    if record['raw_route']!=[0,*[a+1 for a in record['actions']],0]:raise ValueError('Online action/route identity')
                    if not len(record['actions'])==len(record['mask_bits_b64'])==len(record['sample_log_prob'])==len(record['rewards']):raise ValueError('Online complete trace lengths')
                    for action,packed,logpi in zip(record['actions'],record['mask_bits_b64'],record['sample_log_prob']):
                        mask=base64.b64decode(packed)
                        if not 0<=action<len(p.candidate_ids) or not (mask[action//8]>>(action%8))&1 or not math.isfinite(logpi) or logpi>1e-6:raise ValueError('Saved sampled action violates saved legal mask/probability')
                    if abs(math.fsum(record['rewards'])+m['J'])>1e-7:raise ValueError('Saved rewards do not sum to complete return')
                    online[record['request_id']]=(record['actor_tensor_sha256'],m['J'])
                    actual.append((op,slot,record['task_id']));checks+=4
            required={(op,s,batches[(op-1)%4096][s]) for op in range(1,data['updates']+1) for s in range(8)}
            if not required<=set(actual):raise ValueError('Committed PG update lacks online trajectories')
        updates=folder/'UPDATES.jsonl.gz';logical=collections.Counter()
        if updates.exists():
            with gzip.open(updates,'rt',encoding='utf8') as f:
                for line in f:
                    b.check()
                    u=json.loads(line);logical[u['cursor']]+=1
                    if len(u['loss_sources'])!=8:raise ValueError('Scientific update is not logical batch8')
                    for source in u['loss_sources']:
                        parts=source['episode_id'].split(':');op=int(parts[2]);slot=int(parts[3]);tid=batches[(op-1)%4096][slot]
                        if op!=u['cursor'] or not 0<=slot<8 or parts[1]!=role:raise ValueError('Update/episode cursor or role changed')
                        if role!='D5-SL':
                            actor_hash,sample_cost=online[source['episode_id']]
                            if actor_hash!=u['before'] or abs(sample_cost-source['sample_J'])>1e-7:raise ValueError('Update uses foreign actor trajectory/cost')
                            if abs(source['reference_J']-reference_J[tid])>1e-7:raise ValueError('PG baseline is not same-problem frozen SL')
                            expected=float(np.float32(source['sample_J']-source['reference_J']))
                            if source['factor']!=expected:raise ValueError('PG objective normalization/sign changed')
                        else:
                            if tid not in teacher_lengths:teacher_lengths[tid]=len(pool.teacher(tid)[1])
                            if source['action_count']!=teacher_lengths[tid] or source['factor'] is not None:raise ValueError('SL slot differs from bound teacher sequence or objective')
                        checks+=1
        if not set(range(1,data['updates']+1))<=set(logical):raise ValueError('Committed logical update trace missing')
        streams[seed+':'+role]=dict(updates=data['updates'],complete=data.get('complete',False),actual_trajectory_records=len(actual),duplicate_actual_optimizer_updates=sum(max(0,v-1) for v in logical.values()))
    evaluated={}
    for file in sorted((HOME/'evaluations/development').glob('*.json')):
        report=read(file)
        if not report.get('complete'):continue
        rr=report['rows'];good=[r for r in rr if r['valid']];parents=collections.defaultdict(list)
        for row in good:
            b.check()
            rec=unarchive(row['record']);p,_=pool.get(row['case_id'])
            if rec['problem_identity']!=p.identity:raise ValueError('Evaluation problem source')
            for prefix,key in [('raw_','raw_route'),('','route')]:
                route=np.asarray(rec[key],dtype=int);L=float(p.D[route[:-1],route[1:]].sum());K=len(route)-2
                for name,value in [('L',L),('K',K),('J',L+.75*K)]:
                    if abs(value-row['metrics'][prefix+name])>1e-7:raise ValueError('Evaluation saved-route arithmetic')
                    checks+=1
            parents[row['physical_key']].append(row)
        saved=report['summary']['overall']
        if saved['completed_valid']!=len(good) or abs(saved['success_rate']-len(good)/len(rr))>1e-12:raise ValueError('Evaluation denominator')
        for metric in ('J','L','K','raw_J','raw_L','raw_K'):
            a=math.fsum(r['metrics'][metric] for r in good)/len(good)
            z=math.fsum(math.fsum(r['metrics'][metric] for r in group)/len(group) for group in parents.values())/len(parents)
            if abs(a-saved['request_mean'][metric])>1e-8 or abs(z-saved['parent_equal_mean'][metric])>1e-8:raise ValueError('Evaluation aggregation')
            checks+=2
        evaluated[file.name]=dict(requests=len(rr),valid=len(good),physical_parents=len(parents))
    # Tensor-only inspection, without constructing a model or running a forward.
    import torch
    from rc3_conservative_refinement.runtime import SL_PATH,SL_SHA256,tensor_digest
    if sha(ROOT/SL_PATH)!=SL_SHA256:raise ValueError('Exact SL256 archive digest')
    original_model=torch.load(ROOT/SL_PATH,map_location='cpu',weights_only=True)['model']
    allowed={'key.weight','dynamic.weight','nav.weight','query.weight','query.bias'};models={};fixed_hashes={}
    for file in sorted((HOME/'streams').glob('*/*/step*.actor.json')):
        b.check()
        entry=read(file)
        if sha(ROOT/entry['path'])!=entry['sha256']:raise ValueError('Scientific export bytes')
        package=torch.load(ROOT/entry['path'],map_location='cpu',weights_only=True)
        if package['engineering_only'] or package['critic_included'] or package['role']!=entry['role']:raise ValueError('Export role or usage')
        model=package['model'];changed=[k for k in model if not torch.equal(model[k],original_model[k])]
        if tensor_digest(model)!=package['tensor_sha256']:raise ValueError('Export tensor metadata differs from actual tensors')
        if set(model)!=set(original_model) or sum(v.numel() for v in model.values())!=65088:raise ValueError('Architecture changed')
        if entry['role']!='Full-PG' and not set(changed)<=allowed:raise ValueError('Frozen representation changed')
        if 'step00000' in file.name and changed:raise ValueError('Formal initialization is not exact SL256')
        models[file.relative_to(HOME).as_posix()]=dict(role=entry['role'],changed_tensors=changed,engineering_only=False)
        method='D5-SL-U' if entry['role']=='D5-SL' else entry['role']
        fixed_hashes[(int(file.parent.parent.name),method,int(file.name.split('.')[0].replace('step','')))]=package['tensor_sha256']
        checks+=1
    for file in sorted((HOME/'evaluations/development').glob('*.json')):
        report=read(file)
        if not report.get('complete'):continue
        expected='1817f58aa6506456ff12f0c5fe0a914e156aef8af319f4687dde90782f55a63e' if report['method']=='FrozenSL' else fixed_hashes.get((report['seed'],report['method'],report['endpoint']))
        if expected is not None:
            if any(row['valid'] and row['actor_tensor_sha256']!=expected for row in report['rows']):raise ValueError('Actual evaluation used another tensor version than frozen export')
            checks+=len(report['rows'])
    events=collections.Counter()
    with (HOME/'EVENTS.jsonl').open(encoding='utf8') as f:
        for line in f:events[json.loads(line)['event']]+=1
    ledger=read(HOME/'RESOURCE_ACCOUNTING.json')['counts']
    for event,key,prior in [('request_attempt','requests',15),('request_complete','requests_completed',14),('request_failure','requests_failed',1),('optimizer_attempt','actor_optimizer_attempts',5)]:
        if events[event]+prior!=ledger[key]:raise ValueError('Cumulative ledger differs from actual-attempt journal: '+key)
        checks+=1
    verdict=dict(status='PASS',checks=checks,full_teachers_checked=len(manifest['records']),physical_full_parents=len(full_physical),old256_hashes_unchanged=True,
        unique_joint_tasks=352,same_task_schedule_per_seed=True,actual_streams=streams,
        actual_evaluation_grids=evaluated,actual_actor_exports=models,actual_event_counts=dict(events),
        new_forward_calls=0,new_optimizer_attempts=0,new_planning_or_H_or_native=0,new_navigation_or_teacher=0,
        scope='New saved evidence only; no old historical CSV audit, original60 or oldS3 semantics',wall_s=time.perf_counter()-start)
    put(HOME/'INDEPENDENT_NUMERICAL_AUDIT.json',verdict)
    print(dict(status='PASS',checks=checks,streams=streams,wall_s=verdict['wall_s']))

if __name__=='__main__':main()
