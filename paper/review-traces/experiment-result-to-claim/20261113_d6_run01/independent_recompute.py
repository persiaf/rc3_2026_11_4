import json,gzip,hashlib,math,statistics,collections,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[4]
HOME=ROOT/'rc3_d6_20261113'
OUT=pathlib.Path(__file__).resolve().parent
load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
methods=['FrozenSL','A','B','SL','Fixed3','Detour16'];res={}; grids={};errors=[];nraw=0;maxerr=0.;timing={};scope=set()
def avg(v):return statistics.mean(v)
def gain(a,b,m,group=None):
 aa={r['case_id']:r for r in a if group is None or r['group']==group};bb={r['case_id']:r for r in b if group is None or r['group']==group}
 assert set(aa)==set(bb)
 parents=collections.defaultdict(list)
 for cid,r in aa.items():parents[r['physical_key']].append(cid);assert r['physical_key']==bb[cid]['physical_key']
 ps={p:(sum(bb[i]['metrics'][m] for i in ids)-sum(aa[i]['metrics'][m] for i in ids))/sum(bb[i]['metrics'][m] for i in ids) for p,ids in parents.items()}
 return dict(mean=avg(ps.values()),parents=ps)
for ep in [128,256]:
 grids[ep]={m:load(HOME/f'evaluations/development/{m}_{ep}.json')['rows'] for m in methods};gg=grids[ep];ids={r['case_id'] for r in gg['B']};assert len(ids)==66
 for method,rows in gg.items():
  assert len(rows)==66 and {r['case_id'] for r in rows}==ids and all(r['valid'] for r in rows)
  pc=collections.Counter(r['physical_key'] for r in rows);assert len(pc)==11 and set(pc.values())=={6}
  timing[f'{method}_{ep}']={'p50_s':statistics.median(r['full_request_s'] for r in rows),'mean_s':avg(r['full_request_s'] for r in rows)}
  for row in rows:
   ptr=row['record']
   with (ROOT/ptr['path']).open('rb') as f:f.seek(ptr['offset']);raw=gzip.decompress(f.read(ptr['compressed_bytes']))
   assert hashlib.sha256(raw).hexdigest()==ptr['raw_sha256'];r=json.loads(raw);nraw+=1;assert r['metrics']==row['metrics'] and r['task_id']==row['case_id']
   for prefix,pkey in [('', 'plan'),('raw_', 'raw_plan')]:
    met=r['metrics'];plan=r[pkey];cert=plan['validation'];scope.add(cert['scope'])
    assert cert['valid_model_plan'] and cert['all_tasks_served'] and cert['endpoint_constraints_satisfied'] and not cert['physical_joint_trajectory_certified'] and cert['collision_count']==0
    err=max(abs(met[prefix+'J']-(met[prefix+'L']+.75*met[prefix+'K'])),abs(math.fsum(plan['route_leg_costs_m'])-met[prefix+'L']),abs(cert['independent_polyline_length_m']-met[prefix+'L']))
    maxerr=max(maxerr,err);assert err<1e-7
   if method in ['Fixed3','Detour16']:assert 'Native final' in r['raw_semantics'] and r['raw_route']==r['route']
   else:
    assert r['H_identity']=='5db67cacc1128b6143c7a04b9b84a70c37bdc0ab917a0eebae0b8aca132c4d14'
    assert abs(math.fsum(r['rewards'])+r['metrics']['J'])<1e-7
    c=r['credit'];assert c['q_before'][0]==1 and c['raw_prefix_cost_before'][0]==0
    assert all(abs(q-k/c['initial_task_count'])<1e-14 for q,k in zip(c['q_before'],c['remaining_task_count_before']))
    assert abs(math.fsum(c['raw_step_cost_with_terminal_exit'])-r['metrics']['raw_J'])<1e-7
    assert abs(math.fsum(c['H_step_cost_with_terminal_correction'])-r['metrics']['J'])<1e-7
 res[ep]={}
 for control in ['FrozenSL','SL','A']:
  rr={m:gain(gg['B'],gg[control],m) for m in ['J','raw_J']};rr['groups']={g:gain(gg['B'],gg[control],'J',g) for g in sorted({r['group'] for r in gg['B']})};res[ep][control]=rr
 events=[json.loads(l) for l in (HOME/'EVENTS.jsonl').read_text().splitlines()]
account=load(HOME/'RESOURCE_SCIENTIFIC_FINAL_RAW.json');counts=collections.Counter(e['event'] for e in events)
assert counts['request_attempt']==4888 and counts['request_complete']==4888 and counts['optimizer_attempt']==768 and not counts['request_failure']
updates={}
for method in ['A','B','SL']:
 with gzip.open(HOME/f'streams/{method}/UPDATES.jsonl.gz','rt') as f:uu=[json.loads(l) for l in f]
 assert len(uu)==256 and [u['cursor'] for u in uu]==list(range(1,257))
 assert all(len(u['loss_sources'])==8 and math.isfinite(u['loss']) and math.isfinite(u['gradient_norm']) for u in uu)
 updates[method]={'updates':len(uu),'all_clipped':all(u['clipped'] for u in uu),'mean_gradient_norm':avg(u['gradient_norm'] for u in uu)}
 restores=[e for e in events if e['event']=='checkpoint_restored' and e['stream']==method];assert len(restores)==1 and restores[0]['cursor']==128 and restores[0]['Adam_rng_sampler_cursor_and_budget_restored']
result={'status':'PASS','formula':'For each parent: (sum control J - sum B J)/sum control J, then equal mean over parents; positive means B lower cost.','parent_paired':res,'complete_grids':12,'raw_records_checked':nraw,'max_abs_saved_cost_error':maxerr,'timing':timing,'certificate_scope':sorted(scope),'accounting':account,'event_counts':dict(counts),'updates':updates,'no_new_forward_optimizer_or_planning':True}
(OUT/'INDEPENDENT_RECOMPUTATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps({'status':'PASS','raw_checked':nraw,'max_cost_error':maxerr,'metrics':{ep:{c:{'HJ':v['J']['mean'],'rawJ':v['raw_J']['mean'],'groups':{g:z['mean'] for g,z in v['groups'].items()}} for c,v in val.items()} for ep,val in res.items()},'updates':updates},indent=2))
