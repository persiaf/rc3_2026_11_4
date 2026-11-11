"""Independent arithmetic over new request and online diagnostic archives only."""
import csv
import gzip
import json
import math
from collections import Counter
from .common import ROOT,TABLE,read

def run(path,ledger):
    rows=list(csv.DictReader((path/'REQUEST_INDEX.csv').open(encoding='utf-8',newline='')))
    byid={r['request_id']:r for r in rows};assert len(byid)==len(rows)
    count=0;checks=0
    with gzip.open(path/'REQUESTS.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            ledger.check();r=json.loads(line);i=byid[r['request_id']];count+=1
            for k in ('J','L','K','raw_J','raw_L','raw_K'):
                assert math.isclose(float(i[k]),r['metrics'][k],rel_tol=1e-12,abs_tol=1e-7);checks+=1
            for prefix in ('','raw_'):
                m=r['metrics'];assert math.isclose(m[prefix+'J'],m[prefix+'L']+.75*m[prefix+'K'],rel_tol=1e-12,abs_tol=1e-7);checks+=1
            for name in ('plan','raw_plan'):
                p=r[name];assert p['validation']['valid_model_plan'] and not p['unserved_task_ids']
                assert len(p['assignments'])==len(p['target_task_ids_global']);checks+=1
            if 'targets' in r:
                rewards=r['targets']['scaled_dense_rewards'];returns=r['targets']['returns']
                assert len(rewards)==len(returns)==len(r['actions'])
                tail=0.
                for reward,G in zip(reversed(rewards),reversed(returns)):
                    tail+=reward;assert math.isclose(tail,G,rel_tol=1e-6,abs_tol=2e-5);checks+=1
    assert count==len(rows)==ledger.counts['requests_completed']
    table=read(ROOT/TABLE);state_count=0;opportunities=set();unknown=0
    with gzip.open(path/'ONLINE_UPDATE_DIAGNOSTICS.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            ledger.check();r=json.loads(line);opportunities.add((r['role'],r['opportunity']))
            for s in r['states']:
                state_count+=1;conditional=table['conditional'].get(s['condition_key'],table['global'])
                compose=lambda c:-c[0]-.75*c[1]+c[2]
                assert math.isclose(s['B_global'],compose(table['global']),rel_tol=2e-6,abs_tol=2e-6)
                assert math.isclose(s['B_cond'],compose(conditional),rel_tol=2e-6,abs_tol=2e-6)
                assert -0.2500001<=s['delta_min']<=s['delta_max']<=.2500001
                if s['G'] is not None:
                    base=s['B_cond'] if r['role']=='PG-cond' else s['B_global']
                    assert math.isclose(s['A_sample'],s['G']-base,rel_tol=1e-6,abs_tol=2e-6)
                    assert math.isclose(s['centered_global'],s['G']-s['B_global'],rel_tol=1e-6,abs_tol=2e-6)
                    assert math.isclose(s['centered_cond'],s['G']-s['B_cond'],rel_tol=1e-6,abs_tol=2e-6)
                if not r['accepted']:assert math.isclose(s['applied_ratio'],1.,abs_tol=2e-6)
                if s['unknown_fallback']:unknown+=1
                checks+=8
    counts=Counter(r['kind'] for r in rows)
    if ledger.stage=='science':
        assert counts['training']==1536 and counts['evaluation']==792 and len(opportunities)==384
        assert ledger.counts['actor_optimizer_attempts']==384 and ledger.counts['teacher_replays']==768
        for endpoint in (64,128):
            e=[r for r in rows if r['kind']=='evaluation' and int(r['opportunity'])==endpoint]
            assert len(e)==396 and len({r['case_id'] for r in e})==66
        import numpy as np
        from collections import defaultdict
        timing={r['request_id']:float(r['full_request_s']) for r in csv.DictReader((path/'REQUEST_TIMING.csv').open(encoding='utf-8',newline=''))}
        for row in csv.DictReader((path/'QUALITY_CONTRASTS.csv').open(encoding='utf-8',newline='')):
            selected=[r for r in rows if r['kind']=='evaluation' and r['opportunity']==row['endpoint']
                and (row['cohort']=='ALL' or r['cohort']==row['cohort']) and (row['size']=='ALL' or r['width']+'x'+r['height']==row['size'])]
            groups=defaultdict(lambda:defaultdict(list))
            for r in selected:
                if r['role'] in (row['method'],row['control']):
                    value=timing[r['request_id']] if row['metric']=='request_wall_s' else float(r[row['metric']])
                    groups[r['physical_key']][r['role']].append(value)
            gains=[]
            for parent in sorted(groups):
                a=math.fsum(groups[parent][row['method']])/len(groups[parent][row['method']])
                b=math.fsum(groups[parent][row['control']])/len(groups[parent][row['control']])
                gains.append((b-a)/b)
            n=len(gains);draw=np.random.Generator(np.random.PCG64(11652026)).integers(n,size=(20000,n))
            # Alternate frequency accumulation versus production direct indexing.
            sampled=np.zeros(20000)
            for i,g in enumerate(gains):sampled+=np.count_nonzero(draw==i,axis=1)*g/n
            lo,hi=np.percentile(sampled,[2.5,97.5],method='linear')
            for key,value in (('relative_saving',math.fsum(gains)/n),('ci95_low',lo),('ci95_high',hi)):
                assert math.isclose(float(row[key]),value,rel_tol=1e-9,abs_tol=1e-10);checks+=1
        for endpoint in (64,128):
            d=read(path/f'ENDPOINT_{endpoint:04d}_DECISION.json')
            passed=all(r['relative_saving']>=.005 and r['ci95_low']>0 for r in d['checks']) and all(r['relative_saving']>=0 for r in d['size_checks'])
            assert (d['status']=='QUALITY_SCREEN_PASS')==passed and d['all_routes_valid'];checks+=1
    return dict(status='PASS',checks=checks,new_requests=count,online_update_state_rows=state_count,
        distinct_role_opportunities=len(opportunities),unknown_fallback_states=unknown,
        old_large_csv_read=False,critic_updates=0,scope='new archives only; independent return/table/advantage/cost/mask-bound arithmetic and exact request/update counts')
