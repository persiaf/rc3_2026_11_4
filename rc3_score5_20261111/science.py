"""Proposed1428-request development screen; separate execution receipt REQUIRED."""
from collections import defaultdict
import csv
import numpy as np
from .common import HOME,read,put,csvrow,METHODS,ROLES
from .actor import load_actor,export
from .trainer import Trainer
from .runtime import request,teacher_episode

def summarize(ledger,endpoint):
    data=list(csv.DictReader((ledger.path/'REQUEST_INDEX.csv').open(encoding='utf-8-sig')))
    rows=[r for r in data if r['kind']=='evaluation' and int(r['opportunity'])==endpoint]
    times={r['request_id']:float(r['full_request_s']) for r in csv.DictReader((ledger.path/'REQUEST_TIMING.csv').open(encoding='utf-8-sig'))}
    if len(rows)!=330:raise ValueError('Incomplete fixed endpoint; no success-only aggregation')
    checks=[]
    for cohort in sorted({r['cohort'] for r in rows}):
        for size in ('ALL','4x3','6x6','8x8'):
            chosen=[r for r in rows if r['cohort']==cohort and (size=='ALL' or r['width']+'x'+r['height']==size)]
            for method in METHODS:
                group=[r for r in chosen if r['role']==method];parents=defaultdict(list)
                for r in group:parents[r['physical_key']].append(r)
                record=dict(endpoint=endpoint,cohort=cohort,size=size,method=method,physical_parents=len(parents),requests=len(group),success_rate=sum(r['valid']=='True' for r in group)/len(group))
                for m in ('J','L','K','raw_J','raw_L','raw_K'):record[m]=float(np.mean([np.mean([float(r[m]) for r in rs]) for rs in parents.values()]))
                record.update(full_request_p50_s=float(np.median([times[r['request_id']] for r in group])),full_request_p95_s=float(np.quantile([times[r['request_id']] for r in group],.95)))
                csvrow(ledger.path/'COMPLETE_ROUTE_SUMMARY.csv',record)
            for control in ('FrozenSL','Score5-SL','Fixed3','Detour16'):
                for metric in ('raw_J','J'):
                    groups=defaultdict(lambda:defaultdict(list))
                    for r in chosen:
                        if r['role'] in ('Score5-PG',control):groups[r['physical_key']][r['role']].append(float(r[metric]))
                    delta=np.array([(np.mean(g[control])-np.mean(g['Score5-PG']))/np.mean(g[control]) for g in groups.values()])
                    draws=np.random.Generator(np.random.PCG64(11652026)).integers(0,len(delta),size=(20000,len(delta)))
                    lo,hi=np.quantile(delta[draws].mean(1),[.025,.975]);point=float(delta.mean())
                    csvrow(ledger.path/'PARENT_CONTRASTS.csv',dict(endpoint=endpoint,cohort=cohort,size=size,control=control,metric=metric,saving=point,ci95_low=float(lo),ci95_high=float(hi),parents=len(delta)))
                    if control in ('FrozenSL','Score5-SL'):checks.append(point>=.005 and lo>0 if size=='ALL' else point>=0)
    passed=all(checks) and all(r['valid']=='True' for r in rows)
    put(ledger.path/f'ENDPOINT_{endpoint:04d}.json',dict(status='DEVELOPMENT_INCREMENT_PASS' if passed else 'QUALITY_NOT_MET',independent_generalization=False,automatic_followup=False))
    return passed

def run(ledger,trace,inputs,identity):
    schedule=read(HOME/'SCHEDULE.json');models=ledger.path/'models';models.mkdir()
    trainers={r:Trainer(load_actor(r),ledger,identity) for r in ROLES}
    for r,t in trainers.items():t.save(models/f'{r}_0000.training.pt')
    decisions=[]
    for row in schedule['training']:
        op=row['opportunity'];pg=trainers['Score5-PG'];sl=trainers['Score5-SL']
        episodes=[request(ledger,inputs,s,f'train:Score5-PG:{op}:{slot}','Score5-PG',actor=pg.actor,seed=11622026+1000*op+slot,op=op,kind='training')['episode'] for slot,s in enumerate(row['specs'])]
        pg.update(episodes);del episodes
        labels=[teacher_episode(ledger,inputs,sl.actor,i,f'teacher:{op}:{slot}') for slot,i in enumerate(row['teacher_indices'])];sl.update(labels);del labels
        if max(t.consecutive_rejects for t in trainers.values())>=16:raise RuntimeError('Paired stop at16 consecutive rejections, no refill')
        if op in (64,128):
            entries={}
            for r,t in trainers.items():
                t.save(models/f'{r}_{op:04d}.training.pt');entries[r]=export(models/f'{r}_{op:04d}.actor.pt',t.actor,'science')
            for ci,case in enumerate(schedule['evaluation']):
                results={};k=(ci+op)%5
                for role in METHODS[k:]+METHODS[:k]:
                    results[role]=request(ledger,inputs,case['spec'],f'eval:{op}:{case["id"]}:{role}',role,entry=entries.get(role),greedy=True,op=op,kind='evaluation',case=case)['record']
                for role in ('Score5-PG','Score5-SL'):
                    for stage,key in [('raw','raw_plan'),('H','plan')]:
                        visits=results[role][key]['visit_order_candidate_ids'];base=results['FrozenSL'][key]['visit_order_candidate_ids']
                        csvrow(ledger.path/'STOP_SET_AND_ORDER_CHANGES.csv',dict(endpoint=op,case_id=case['id'],cohort=case['group'],method=role,stage=stage,
                            set_changed=set(visits)!=set(base),sequence_changed=visits!=base,same_set_order_changed=set(visits)==set(base) and visits!=base,control='FrozenSL'))
            decisions.append(summarize(ledger,op))
    put(ledger.path/'FINAL_DECISION.json',dict(status='CONSIDER_SEPARATE_FULL_ACTOR_PG_ATTRIBUTION' if decisions[-1] else 'STOP_SCORE5_CONFIGURATION',
        attribution_executed=False,automatic_extension=False,independent_heldout_evaluation_executed=False))
