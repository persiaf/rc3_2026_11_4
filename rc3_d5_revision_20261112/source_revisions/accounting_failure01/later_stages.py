"""Only the original P2-P5 conditions can enter these stages."""
import time,copy,collections,json,hashlib
import numpy as np
from .common import ROOT,HOME,H_ID,read,put,sha,state,identity,gzput,gzread,unarchive
from .science import train_stream,closeout,fast_budget
from .data import Pool
from .runtime import deployed_actor,frozen_actor,policy_request,native_request
from . import evaluation

def advance(budget,pool,base,pg,sl):
    seed=2026111101;state(phase='P2',status='MATCHED_CONTROLS')
    full=train_stream(budget,pool,'Full-PG',seed)
    sl=train_stream(budget,pool,'D5-SL',seed,sl_time_target=pg['training_compute_s'],until_C=True)
    cr=None
    if sl.get('SL_C'):
        cr=evaluation.evaluate(budget,pool,'D5-SL-C',seed,sl['SL_C']['cursor'],entry=sl['SL_C'])
    pgr=evaluation.evaluate(budget,pool,'D5-PG',seed,4096,entry=pg['endpoints']['4096'])
    fr=evaluation.evaluate(budget,pool,'Full-PG',seed,4096,entry=full['endpoints']['4096'])
    c_gain=None if cr is None else evaluation.gains(pgr['rows'],cr['rows'])
    p2=dict(compute_matched=bool(sl.get('compute_matched')),compute_ratio=sl.get('compute_ratio'),against_SL_C=c_gain,
        against_FullPG=evaluation.gains(pgr['rows'],fr['rows']),fixed_endpoints=4096)
    put(HOME/'P2_DECISION.json',p2,(HOME/'P2_DECISION.json').exists())
    if not sl.get('compute_matched'):closeout('COMPUTE_ATTRIBUTION_INCOMPLETE','SL-C cannot match registered PG training wall within2%,16384 updates and8h');return
    if not c_gain or not c_gain['complete'] or c_gain['mean']<=0:closeout('NO_INCREMENT_OVER_COMPUTE_MATCHED_SL','D5-PG not positive against SL-C at fixed development endpoints');return
    pgs=[pgr];us=[evaluation.evaluate(budget,pool,'D5-SL-U',seed,4096,entry=sl['endpoints']['4096'])];cs=[cr]
    for seed in (2026111102,2026111103):
        state(phase='P3',status='REPLICATION',seed=seed)
        p=train_stream(budget,pool,'D5-PG',seed);s=train_stream(budget,pool,'D5-SL',seed,sl_time_target=p['training_compute_s'])
        train_stream(budget,pool,'Full-PG',seed)
        s=train_stream(budget,pool,'D5-SL',seed,sl_time_target=p['training_compute_s'],until_C=True)
        if not s.get('compute_matched'):closeout('REPLICATION_COMPUTE_INCOMPLETE','A prescribed seed SL-C does not match time budget',failed_seed=seed);return
        pgs.append(evaluation.evaluate(budget,pool,'D5-PG',seed,4096,entry=p['endpoints']['4096']))
        us.append(evaluation.evaluate(budget,pool,'D5-SL-U',seed,4096,entry=s['endpoints']['4096']))
        cs.append(evaluation.evaluate(budget,pool,'D5-SL-C',seed,s['SL_C']['cursor'],entry=s['SL_C']))
    controls={'FrozenSL':[base['FrozenSL']]*3,'SL-U':us,'SL-C':cs};comparisons={};passed=all(all(r['valid'] for r in p['rows']) for p in pgs)
    for name,ctrl in controls.items():
        per=[evaluation.gains(p['rows'],c['rows']) for p,c in zip(pgs,ctrl)];avg=evaluation.gains(evaluation.average_seed_rows(pgs),evaluation.average_seed_rows(ctrl))
        positive=sum(x['complete'] and x['mean']>0 for x in per);ok=avg['complete'] and avg['mean']>0 and positive>=2
        comparisons[name]=dict(seed_results=per,pooled_parent_result=avg,seeds_positive=positive,passed=bool(ok));passed=passed and ok
    decision=dict(passed=bool(passed),comparisons=comparisons,deploy_seed=2026111101,seed_selection=False)
    put(HOME/'P3_DECISION.json',decision,(HOME/'P3_DECISION.json').exists())
    if not passed:closeout('NOT_STABLY_REPRODUCED','Prescribed three-seed development criteria not satisfied');return
    state(phase='P4',status='THREE_SEED_PASS',next_action='Bounded same-output deployment timing and method freeze')

def methods():
    entries={'FrozenSL':None,'Fixed3':None,'Detour16':None}
    for seed in (2026111101,2026111102,2026111103):
        pg=read(HOME/'streams'/str(seed)/'D5-PG/STREAM_SUMMARY.json');sl=read(HOME/'streams'/str(seed)/'D5-SL/STREAM_SUMMARY.json');full=read(HOME/'streams'/str(seed)/'Full-PG/STREAM_SUMMARY.json')
        entries[f'D5-PG_{seed}']=pg['endpoints']['4096'];entries[f'SL-U_{seed}']=sl['endpoints']['4096'];entries[f'SL-C_{seed}']=sl['SL_C'];entries[f'Full-PG_{seed}']=full['endpoints']['4096']
    return entries

class ColdPool:
    def __init__(self,factory,entries):self.factory=factory;self.entries=entries;self.instance=None
    def get(self,tid):
        if self.instance is None:self.instance=self.factory()
        return self.instance.get(tid)

def timing_grid(budget,pool,entries,tier,factory=None):
    labels=['FrozenSL','SL-U_2026111101','SL-C_2026111101','D5-PG_2026111101','Full-PG_2026111101','Fixed3','Detour16']
    file=HOME/'timing'/(tier+'.json');report=read(file) if file.exists() else dict(tier=tier,observations=[],contract='Warm model/map resident; cold model+map unloaded, runtime library already initialized; all task-specific work and output validation included')
    done={r['id'] for r in report['observations']};warm={};setup=time.perf_counter()
    for name in labels:
        if name not in ('Fixed3','Detour16'):warm[name]=frozen_actor(budget) if name=='FrozenSL' else deployed_actor(entries[name],budget)
    for case in pool.cases:pool.get(case['id'])
    report['warm_setup_s']=time.perf_counter()-setup
    rng=np.random.default_rng(2026111198)
    # A fixed single full-request warm-up for every method/mode; saved and charged.
    cases=[dict(pool.cases[0],id=pool.cases[0]['id'],warmup=True)]+pool.cases
    for i,case in enumerate(cases):
        order=list(rng.permutation(labels))
        for mode in ('warm','cold'):
            for name in order:
                for rep in range(1 if i==0 else 3):
                    rid=f'timing:{tier}:{"warmup" if i==0 else case["id"]}:{mode}:{name}:{rep}'
                    if rid in done:continue
                    used_pool=pool if mode=='warm' else ColdPool(factory or (lambda:Pool(budget)),pool.entries)
                    purpose='deployment' if tier=='development' else tier
                    if name in ('Fixed3','Detour16'):out=native_request(budget,used_pool,case['id'],rid,name,'timing/records',purpose)
                    else:
                        act=warm[name] if mode=='warm' else (lambda n=name:frozen_actor(budget) if n=='FrozenSL' else deployed_actor(entries[n],budget))
                        out=policy_request(budget,used_pool,case['id'],rid,act,purpose=purpose,folder='timing/records',greedy=True,method=name,seed=2026111101)
                    row=dict(id=rid,case_id=case['id'],physical_key=case['physical_key'],method=name,mode=mode,rep=rep,warmup=i==0,wall_s=out['full_request_s'],record=out['ref'])
                    report['observations'].append(row);put(file,report,file.exists());done.add(rid)
    # Output equality is verified using saved full outputs, without replanning.
    outputs={}
    for row in report['observations']:
        if row['warmup']:continue
        rec=unarchive(row['record']);key=(row['case_id'],row['method']);semantic=(rec['raw_route'],rec['route'],rec['metrics'])
        if key in outputs and outputs[key]!=semantic:raise ValueError('Deployment outputs changed by residency/timing mode')
        outputs[key]=semantic
    summary={}
    for mode in ('warm','cold'):
        summary[mode]={}
        for name in labels:
            times=collections.defaultdict(list)
            for row in report['observations']:
                if not row['warmup'] and row['mode']==mode and row['method']==name:times[row['case_id']].append(row['wall_s'])
            med=[float(np.median(x)) for x in times.values()]
            summary[mode][name]=dict(median_s=float(np.median(med)),p95_s=float(np.quantile(med,.95)),case_medians=dict(zip(times,med)))
    report.update(summary=summary,all_outputs_equivalent=True,complete=True);put(file,report,True);return report

def deployment(budget):
    fast_budget(budget)
    if not read(HOME/'P3_DECISION.json')['passed']:raise PermissionError('P3 not passed')
    pool=Pool(budget);entries=methods();timing=timing_grid(budget,pool,entries,'development')
    source=read(HOME/'SCIENTIFIC_SOURCE_BINDING.json')
    frozen=dict(identity=identity(),models=entries,source_sha256=source['source_sha256'],protocol_sha256=sha(HOME/'REVISION_PROTOCOL.json'),
        statistics=dict(bootstrap=20000,seed=2026111199),deployment_seed=2026111101,checkpoint=4096,
        optimization_rounds=0,reason='Production common immutable-input cache/model residency already implemented; no additional method-changing optimization',
        development_timing=timing['summary'],all_outputs_equivalent=True)
    text=json.dumps(frozen,sort_keys=True,separators=(',',':')).encode();frozen['freeze_id']=hashlib.sha256(text).hexdigest()
    put(HOME/'METHOD_FREEZE.json',frozen,(HOME/'METHOD_FREEZE.json').exists());state(phase='P5_CONFIRM',status='METHODS_FROZEN',freeze_id=frozen['freeze_id'])

class LayerPool(Pool):
    def __init__(self,budget,tier):
        super().__init__(budget);self.tier=tier;self.parent_cache={};idx=read(HOME/tier/'CASE_INDEX.json');self.cases=idx['cases'];self.layer_parents=idx['parents']
        self.entries.update({c['id']:c for c in self.cases})
    def get(self,tid):
        if tid not in {c['id'] for c in self.cases}:return super().get(tid)
        if tid in self.cache:return self.cache[tid]
        from rc3_global_local_impl_review.runtime_guards import snapshot_parent
        from rc3_global_local_trainer.navigation_compat import load_navigation_variants,select_navigation
        from rc3_s3_connectivity_20261109.scope import scoped_view
        e=self.entries[tid];pe=self.layer_parents[str(e['parent'])]
        if e['parent'] not in self.parent_cache:
            for info in pe['files'].values():
                if sha(ROOT/info['path'])!=info['sha256']:raise ValueError('Frozen independent input bytes')
            scene=read(ROOT/pe['files']['scene']['path'])
            with np.load(ROOT/pe['files']['arrays']['path'],allow_pickle=False) as arr:F,A=arr['F'].copy(),arr['A'].copy()
            nav=select_navigation(load_navigation_variants(scene,ROOT/pe['files']['navigation']['path']),scene,F,A,pe['parent_identity'])
            parent=snapshot_parent(scene,F,A,navigation=nav)
            if parent.identity!=pe['parent_identity']:raise ValueError('Independent parent identity')
            self.parent_cache[e['parent']]=parent
        p,evidence=scoped_view(self.parent_cache[e['parent']],e['spec']);evidence.update(physical_key=e['physical_key'],input_source_files=pe['files'])
        self.cache[tid]=(p,evidence);return self.cache[tid]

def generate_layer(budget,tier):
    if not (HOME/'METHOD_FREEZE.json').exists():raise PermissionError('Methods not frozen')
    if tier=='final' and not read(HOME/'CONFIRMATION_DECISION.json')['passed']:raise PermissionError('Final layer remains sealed')
    folder=HOME/tier;folder.mkdir(exist_ok=True);index=folder/'CASE_INDEX.json'
    if index.exists():return LayerPool(budget,tier)
    from rc3_fast.scenes import make_complex_scene
    from rc3_constructive.problem import resample_tasks
    from rc3_fast.solver import vectorized_feasibility
    from rc3_global_local.run_s0 import make_allow
    from rc3_baseline.baseline import navigation
    from rc3_fast.planner import map_key
    from rc3_global_local_impl_review.runtime_guards import bind_navigation,current_navigation_sources,snapshot_parent,extract_snapshot
    from rc3_global_local_trainer.navigation_compat import save_navigation_artifact
    refs=read(ROOT/'rc3_integrated_rppo/DATA_REFERENCES.json');known=set(refs['known_physical_keys']);parents={};cases=[]
    if tier=='final':known.update(x['physical_key'] for x in read(HOME/'confirmation/CASE_INDEX.json')['parents'].values())
    start_map=900000000000 if tier=='confirmation' else 910000000000;start_task=920000000000 if tier=='confirmation' else 930000000000
    for i in range(24):
        budget.check();record=folder/f'parent_{i:02}';record.mkdir(exist_ok=False);mode='mixed' if i%2==0 else 'blocked'
        budget.debit('new_scenes');scene=resample_tasks(make_complex_scene(10,10,start_map+i,mode),start_task+i);physical=map_key(scene)
        if physical in known:raise ValueError('New physical geometry collides with allowed registry; no reseed/replacement')
        known.add(physical);F,_,_=vectorized_feasibility(scene);A=make_allow(F,i%3)
        budget.debit('navigation_solves');tick=time.perf_counter();nav=navigation(scene);navpath=record/'navigation.npz';save_navigation_artifact(scene,nav,navpath,time.perf_counter()-tick)
        bound=bind_navigation(scene,nav,source_map_key=physical,source_sha256=current_navigation_sources(),provenance=dict(kind='D5_authorized_new_parent',tier=tier,parent=i))
        parent=snapshot_parent(scene,F,A,navigation=bound);extract_snapshot(parent,0,0,10,10,0,0)
        put(record/'scene.json',scene)
        with (record/'arrays.npz').open('xb') as f:np.savez_compressed(f,F=F,A=A)
        files={n:dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for n,p in [('scene',record/'scene.json'),('arrays',record/'arrays.npz'),('navigation',navpath),('navigation_receipt',navpath.with_suffix('.json'))]}
        parents[str(i)]=dict(parent_identity=parent.identity,physical_key=physical,files=files)
        for w,h in ((4,3),(6,6),(8,8)):
            for central in (False,True):
                spec=dict(role=tier,parent=i,map_seed=start_map+i,task_seed=start_task+i,mode=mode,A_condition=i%3,
                    width=w,height=h,x0=(10-w)//2 if central else 0,y0=(10-h)//2 if central else 0,endpoints='open_first_last_connected' if central else 'closed_depot')
                cases.append(dict(id=f'{tier}_{len(cases):03}',parent=i,spec=spec,physical_key=physical,group=tier))
        put(folder/f'parent_{i:02}_RECEIPT.json',dict(parent=i,physical_key=physical,parent_identity=parent.identity,files=files,no_replacement=True))
    put(index,dict(cases=cases,parents=parents,freeze_id=read(HOME/'METHOD_FREEZE.json')['freeze_id']));return LayerPool(budget,tier)

def independent_quality(budget,pool,tier):
    entries=read(HOME/'METHOD_FREEZE.json')['models'];out={}
    for label,entry in entries.items():
        seed=int(label.rsplit('_',1)[1]) if '_' in label else 0
        method='D5-SL-U' if label.startswith('SL-U') else 'D5-SL-C' if label.startswith('SL-C') else 'D5-PG' if label.startswith('D5-PG') else 'Full-PG' if label.startswith('Full-PG') else label
        out[label]=evaluation.evaluate(budget,pool,method,seed,4096,entry=entry,tier=tier,cases=pool.cases,purpose=tier,tag=label)
    pgs=[out[f'D5-PG_{s}'] for s in (2026111101,2026111102,2026111103)]
    controls={'FrozenSL':[out['FrozenSL']]*3,'SL-U':[out[f'SL-U_{s}'] for s in (2026111101,2026111102,2026111103)],'SL-C':[out[f'SL-C_{s}'] for s in (2026111101,2026111102,2026111103)]}
    return out,evaluation.independent_gate(pgs,controls)

def confirmation(budget):
    fast_budget(budget);pool=generate_layer(budget,'confirmation');out,gate=independent_quality(budget,pool,'confirmation')
    timing_grid(budget,pool,read(HOME/'METHOD_FREEZE.json')['models'],'confirmation',factory=lambda:LayerPool(budget,'confirmation'))
    put(HOME/'CONFIRMATION_DECISION.json',gate,(HOME/'CONFIRMATION_DECISION.json').exists())
    if not gate['passed']:closeout('NOT_INDEPENDENTLY_CONFIRMED','Predefined confirmation gate not passed; final layer remains sealed');return
    state(phase='P5_FINAL',status='CONFIRMATION_PASS',next_action='Exactly frozen final grid')

def final(budget):
    fast_budget(budget);pool=generate_layer(budget,'final');out,gate=independent_quality(budget,pool,'final')
    timing=timing_grid(budget,pool,read(HOME/'METHOD_FREEZE.json')['models'],'final',factory=lambda:LayerPool(budget,'final'))
    pg=out['D5-PG_2026111101'];detour=out['Detour16'];fixed=out['Fixed3'];g=evaluation.gains(pg['rows'],detour['rows'])
    upper=None
    if g['complete']:
        modes={r['physical_key']:r['mode'] for r in pg['rows']};groups=[[p for p in g['parents'] if modes[p]==m] for m in ('mixed','blocked')];rng=np.random.default_rng(2026111199)
        draws=[np.mean([-g['parents'][p] for grp in groups for p in rng.choice(grp,len(grp),replace=True)]) for _ in range(20000)];upper=float(np.quantile(draws,.95))
    warm=timing['summary']['warm'];P=warm['D5-PG_2026111101'];D=warm['Detour16'];F=warm['Fixed3']
    medratio=P['median_s']/D['median_s'];p95ratio=P['p95_s']/D['p95_s'];pJ=pg['summary']['overall']['parent_equal_mean']['J'];fJ=fixed['summary']['overall']['parent_equal_mean']['J']
    dominated=fJ<=pJ and F['median_s']<=P['median_s'] and (fJ<pJ or F['median_s']<P['median_s'])
    success=gate['passed'] and upper is not None and upper<=.005 and medratio<=.70 and p95ratio<=1 and not dominated
    decision=dict(quality=gate,Detour_relative_deterioration_upper95=upper,warm_median_ratio=medratio,warm_p95_ratio=p95ratio,dominated_by_Fixed3=bool(dominated),passed=bool(success),deployment_seed=2026111101,checkpoint=4096)
    put(HOME/'FINAL_DECISION.json',decision,(HOME/'FINAL_DECISION.json').exists())
    closeout('SCIENTIFIC_TARGET_PASSED' if success else 'FINAL_TARGET_NOT_MET','Fixed final gate evaluated without method or checkpoint selection',scientific_goal_achieved=bool(success))
