import csv
import gzip
import json
import math
from collections import Counter,defaultdict
import numpy as np
from .common import HOME,read,put

def online(ledger):
    path=ledger.path;rows=list(csv.DictReader((path/'REQUEST_INDEX.csv').open(encoding='utf-8',newline='')))
    byid={r['request_id']:r for r in rows};counts=Counter();checks=0
    assert len(rows)==len(byid)==984
    with gzip.open(path/'REQUESTS.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            ledger.check();r=json.loads(line);i=byid[r['request_id']];counts[r['request_kind']]+=1
            for k in ('J','raw_J','L','raw_L','K','raw_K'):
                assert math.isclose(float(i[k]),r['metrics'][k],abs_tol=1e-7,rel_tol=1e-12);checks+=1
            for pre in ('','raw_'):
                m=r['metrics'];assert math.isclose(m[pre+'J'],m[pre+'L']+.75*m[pre+'K'],abs_tol=1e-7);checks+=1
            for k in ('plan','raw_plan'):assert r[k]['validation']['valid_model_plan'] and not r[k]['unserved_task_ids'];checks+=1
            if 'targets' in r:
                rewards=r['targets']['scaled_dense_rewards'];G=r['targets']['returns'];tail=0.
                for reward,target in zip(reversed(rewards),reversed(G)):
                    tail+=reward;assert math.isclose(tail,target,abs_tol=2e-5,rel_tol=1e-6);checks+=1
    assert counts['training']==768 and counts['evaluation']==216
    events=[json.loads(r) for r in (path/'EVENTS.jsonl').read_text(encoding='utf-8').splitlines()]
    updates=[e for e in events if e['event']=='actor_candidate_complete'];assert len(updates)==192
    assert Counter(e['role'] for e in updates)=={'A':64,'B':64,'C':64}
    contrast=list(csv.DictReader((HOME/'CONDITIONAL_ONLINE_RESULTS.csv').open(encoding='utf-8',newline='')))
    for r in contrast:
        ledger.check();groups=defaultdict(lambda:defaultdict(list))
        for i in rows:
            if i['kind']=='evaluation' and i['opportunity']==r['endpoint'] and (r['size']=='ALL' or i['width']+'x'+i['height']==r['size']) and i['role'] in ('B',r['control']):groups[i['physical_key']][i['role']].append(float(i[r['metric']]))
        gains=[]
        for parent in sorted(groups):
            v=groups[parent];a=math.fsum(v['B'])/len(v['B']);b=math.fsum(v[r['control']])/len(v[r['control']]);gains.append((b-a)/b)
        n=len(gains);draw=np.random.Generator(np.random.PCG64(11652026)).integers(n,size=(20000,n));sample=np.zeros(20000)
        for j,g in enumerate(gains):sample+=np.count_nonzero(draw==j,axis=1)*g/n
        lo,hi=np.percentile(sample,[2.5,97.5],method='linear')
        for key,v in [('relative_saving',math.fsum(gains)/n),('ci95_low',lo),('ci95_high',hi)]:assert math.isclose(float(r[key]),v,abs_tol=1e-10,rel_tol=1e-9);checks+=1
    table=read(HOME/'PROTOCOL.json')['table_sha256']
    from .common import ROOT,TABLE,sha
    assert sha(ROOT/TABLE)==table
    put(path/'INDEPENDENT_NUMERICAL_RECHECK.json',dict(status='PASS',new_requests=984,checks=checks,Actor_attempts=192,Critic_updates=0,
        scope='new records only: route costs/full returns/request counts and independently recomputed paired statistics'))
