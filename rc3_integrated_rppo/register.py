"""Concretize the user's conditional task before any engineering/science call."""
import subprocess
import time
MODULE_STARTED=time.perf_counter()
from .common import *


def specs_for(seed,stage,rounds,parents,bank):
    rng=np.random.Generator(np.random.PCG64(seed_for(seed,stage,'data')));buckets={}
    for p in parents:
        for ti,ts in enumerate(p['task_seeds']):buckets.setdefault((p['mode'],(p['parent']+ti)%3),[]).append((p,ti,ts))
    order={k:[] for k in buckets};position={k:0 for k in buckets};out=[]
    for r in range(rounds):
        batch=[]
        for slot,(w,h) in enumerate([(10,10)]*6+[(4,3)]*2+[(6,6)]*2+[(8,8)]*2):
            mode=('mixed','blocked')[(r+slot)%2];a=(r+slot)%3;k=(mode,a)
            if position[k]>=len(order[k]):order[k]=rng.permutation(len(buckets[k])).tolist();position[k]=0
            p,ti,ts=buckets[k][order[k][position[k]]];position[k]+=1
            opened=(slot+r//2)%2;key=f"{p['map_seed']}_{ts}_A{a}"
            batch.append(dict(role='train',parent=p['parent'],map_seed=p['map_seed'],task_seed=ts,task_draw=ti,
                mode=mode,A_condition=a,width=w,height=h,x0=int(rng.integers(11-w)),y0=int(rng.integers(11-h)),
                endpoints='open_first_last_connected' if opened else 'closed_depot',source_family='bank' if key in bank else 'missing_bank'))
        out.append(batch)
    return out


def register():
    start=MODULE_STARTED;require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,encoding='utf8').strip()==BASE,'Wrong baseline')
    reg=read(ROOT/'rc3_global_local_execution_v3/REGISTRY.json');prior=read(ROOT/'project_management/GIT_MANIFEST.json')
    bank={};geometry={};source_files={}
    for folder in sorted((OLD/'data/task_draws').iterdir()):
        if not folder.is_dir() or not (folder/'receipt.json').exists():continue
        r=read(folder/'receipt.json');s=r['spec'];nav=OLD/'data/navigation'/(r['map_key']+'.npz')
        files={}
        for name,path in [('scene',folder/'scene.json'),('arrays',folder/'arrays.npz'),('receipt',folder/'receipt.json'),('navigation',nav),('navigation_receipt',nav.with_suffix('.json'))]:
            n=label(path);v=prior['files'][n];require(digest(path)==v['sha256'],'Original artifact changed before registration: '+n)
            files[name]=dict(path=n,sha256=v['sha256']);source_files[n]=v
        bank[folder.name]=dict(files=files,original_spec=s,parent_identity=r['parent_identity'])
        geometry.setdefault(str(s['map_seed']),dict(physical_key=r['map_key'],mode=s['mode'],scene_file=files['scene']['path'],scene_sha256=files['scene']['sha256'],navigation_file=files['navigation']['path']))
    parents=reg['S2_train'];splits={};groups={}
    require(len(parents)==96 and len({geometry[str(p['map_seed'])]['physical_key'] for p in parents})==96,'Training parents not 96 independent physical layouts')
    for mode in ('mixed','blocked'):
        ps=sorted([p for p in parents if p['mode']==mode],key=lambda p:geometry[str(p['map_seed'])]['physical_key'])
        groups[mode]=dict(selection=ps[:4],promotion=ps[4:8],final_seen=ps[8:20],warm_valid=ps[:12])
    for family,offset in [('selection',0),('promotion',100),('final_seen',200),('final_unseen',300)]:
        if family=='final_unseen':ps=[dict(parent=None,map_seed=26000000000+i,mode=('mixed','blocked')[i%2]) for i in range(12)]
        else:ps=[p for mode in ('mixed','blocked') for p in groups[mode][family]]
        cases=[]
        for i,p in enumerate(ps):
            ts=26100000000+offset+i
            require(all(ts not in par['task_seeds'] for par in parents),'Task seed collision')
            for w,h in ((4,3),(6,6),(8,8)):
                for opened in (False,True):
                    s=dict(role=family,parent=p['parent'],map_seed=p['map_seed'],task_seed=ts,A_condition=i%3,mode=p['mode'],width=w,height=h,
                        x0=(10-w)//2 if opened else 0,y0=(10-h)//2 if opened else 0,
                        endpoints='open_first_last_connected' if opened else 'closed_depot',source_family='unseen' if family=='final_unseen' else 'new')
                    cases.append(dict(id=f'{family}_{len(cases):03}',spec=s,size=[w,h],mode=p['mode'],endpoint=s['endpoints'],
                        physical_key=None if family=='final_unseen' else geometry[str(p['map_seed'])]['physical_key']))
        splits[family]=cases
    require([len(splits[k]) for k in ('selection','promotion','final_seen','final_unseen')]==[48,48,144,72],'Wrong split sizes')
    splits['training']={str(seed):specs_for(seed,'train',512,parents,bank) for seed in SEEDS}
    splits['warm']={str(seed):specs_for(seed,'warm',8,parents,bank) for seed in SEEDS}
    rngs={seed:np.random.Generator(np.random.PCG64(seed_for(seed,'SL_labels'))) for seed in SEEDS}
    splits['SL_batches']={str(seed):[rngs[seed].integers(0,256,24).tolist() for _ in range(512)] for seed in SEEDS}
    splits['warm_valid_parent_ids']=[p['parent'] for mode in ('mixed','blocked') for p in groups[mode]['warm_valid']]
    engineering=read(ROOT/'rc3_s3_evaluator/ENGINEERING_REGISTRATION.json')['cases'];splits['engineering']=engineering
    # A realistic preregistered old training-bank batch, not twelve views of one
    # cached development map; P0 weights are discarded before formal use.
    splits['engineering_benchmark']=splits['training'][str(SEEDS[0])][0]
    #36 final timing declarations: 3seen+3unseen physical layouts, each all6 windows.
    # 3seen(2mixed+1blocked),3unseen(1mixed+2blocked),each all6.
    timing_seen={0,1,12};timing_unseen={0,1,3}
    splits['timing_case_ids']=[c['id'] for fam in ('final_seen','final_unseen') for c in splits[fam]
        if int(c['id'].split('_')[-1])//6 in (timing_seen if fam=='final_seen' else timing_unseen)]
    models={str(seed):read(ROOT/'rc3_s3_evaluator/S3_MODEL_INDEX.json')['models'][str(seed)]['SL256'] for seed in SEEDS}
    labels=[];pool=read(OLD/'teachers256/POOL_INDEX.json')
    for e in pool['labels']:
        path=OLD/'teachers256'/e['file'];require(digest(path)==e['sha256'],'Teacher label source changed');labels.append(dict(path=label(path),sha256=e['sha256']))
    known=set(v['physical_key'] for v in geometry.values());geometry_roots={}
    def geometry_keys(d):
        if isinstance(d,dict):
            for k,v in d.items():
                if k in ('map_key','physical_key') and isinstance(v,str) and len(v)==64:known.add(v)
                else:geometry_keys(v)
        elif isinstance(d,list):
            for v in d:geometry_keys(v)
    for name in ('rc3_global_local_impl_review/GEOMETRY_GATE.json','rc3_gl_v4_20261109/INPUT_REGISTRY.json','rc3_gl_v5_20261109/INPUT_REGISTRY.json'):
        path=ROOT/name;require(digest(path)==prior['files'][name]['sha256'],'Historical geometry metadata root changed')
        geometry_keys(read(path));geometry_roots[name]=digest(path)
    refs=dict(bank=bank,geometry=geometry,models=models,teachers=labels,teacher_pool_index=dict(path=label(OLD/'teachers256/POOL_INDEX.json'),sha256=digest(OLD/'teachers256/POOL_INDEX.json')),
        known_physical_keys=sorted(known),known_geometry_metadata_roots=geometry_roots,source_files=source_files)
    write_new(HOME/'DATA_REFERENCES.json',refs);write_new(HOME/'DATA_SPLITS.json',splits)
    proposal=ROOT/'RC3_Integrated_DRL_Plan_and_Codex_Task_v1.md'
    p=dict(schema='Integrated-RPPO-v1-executable',experiment=EXP,baseline=BASE,user_task_sha256=digest(proposal),
        old_registry_sha256=REGISTRY,H_identity=H_ID,fixed_goal='Large-parent DRL training, frozen rapid variable-size complete local constraint planning',
        actor=dict(base='Exact own seed SL256 frozen/eval',hidden=[64,64],features=202,residual_amplitude=2.,output_zero=True),
        critic=dict(features=268,hidden=[128,128],independent=True),
        features=dict(actor='q64 + candidate64 + remaining-coverage pool64 + u10',critic='global64 + remaining64 + current64 + ordered-prefix GRU64 + context12',
            u10=['new_cover/n','sum_remaining(C_ij/max(1,sum_kC_ik))/n','D_current_j/10','D_j_exit/10','remaining/n','prefix_stops/n','raw_L/100','window_width/10','window_height/10','entry_ne_exit'],
            context12=['remaining/n','prefix_stops/n','raw_L/100','current_x/10','current_y/10','entry_x/10','entry_y/10','exit_x/10','exit_y/10','width/10','height/10','open_flag'],
            missing='Finite distance0 only for masked illegal entries; no task/route/H answer features',cache='input,prefix,base tensor and endpoint D hash; ephemeral per rollout/SL batch, none across inference requests'),
        optimizer=dict(actor_lr=3e-5,critic_lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=0.,clip=1.,actor_epochs=4,critic_epochs=4),
        objective=dict(gamma=1.,GAE_lambda=1.,reward='Exact frozen dense increments with terminal raw−H correction',scale='max(1,target_task_count)',
            note='Changes cross-instance weighting; reported J/L/K unnormalized',advantage='Frozen full RTG−V_old; divide ALL advantages by single RMS over valid sampled states +1e-8, no centering; empirical, not strictly unbiased',
            actor='1/E sum_episode sum_valid_t [-min(exp(new_lp_t−old_lp_t)*A_t,clip(.9,1.1)*A_t)+.02 KL(pi0||pi)]',
            critic='1/(2E) sum_episode sum_valid_t (V−G)^2',entropy=0.,padding='No padding in implementation; synthetic padded tests exclude padding'),
        KL=dict(mean_old_new=.01,p95_old_new=.05,mean_base_current=.10,statistics='Pooled valid states, not episode-summed KL; exact same mask',
            rollback='Restore failed epoch Actor+Adam+accepted cursor exactly; never refund attempts. Critic independent.',
            retries='At most one persistent LR-halving/retry per rollout batch, then carry lower LR to following batches;3 consecutive noncompliant batches stop'),
        warm=dict(episodes=192,train_valid='Physical parent split24 valid/72train balanced modes',initial_steps=224,max_steps_including_extra_epochs=256,
            extra='If valid episode-MSE worse than initial or >1.5train, at most2 full critic-only epochs within256 cap; no extra trajectories',no_quality_gate=True),
        training=dict(rounds=512,per_round_episodes=24,full_local=[6,2,2,2],natural_rows_per_input=2,checkpoints=[64,128,256,512],selection_candidates=[128,256,512],
            early_stop='Both64 and128: every size mean gain<0 AND all-layout gain<−.02. Neither single failure nor non-significant CI stops otherwise.'),
        matched_SL=dict(batch_episodes=24,per_round='Fixed24 existing teacher labels cached for this batch; one masked route-mean CE step for each accepted PPO Actor epoch',
            optimizer='Same residual parameters, fresh Actor Adam lr3e-5,clip1',no_equal_FLOPs_claim=True,selection_checkpoints=[64,128,256,512]),
        statistics=dict(cluster='true physical map_key; average all6 views then relative gain; no windows or seeds counted as independent maps',
            bootstrap=20000,generator='PCG64',seed=11652026,stratification='mixed/blocked physical parents',interval='percentile2.5/97.5 linear',
            multi_seed='Within each parent average paired gains across3 matched seeds, then parent bootstrap',
            timing='Within each group/size/cache/boundary report median and linear p95 over all matched-seed RL repeats vs deterministic native repeats; native not counted as three independent models. Fixed3 dominance requires no greater paired parent meanJ and no greater median time, at least one strict. N1/10/100 use first actual N measured service durations plus fresh common ready; insufficient streams stay missing, no extrapolation.',
            P2='Select RL and control independently by greatest selection-set mean gain vs ownSL; ties earlier128/256/512. Both RL-vsSL and RL-vsselectedControl mean>=.005; no CI requirement. Rule dominance reported, not additional research gate.',
            P3='Both comparisons aggregate mean>=.005 and lower95>0; each of3 seed point gains>0, all valid; then P4',
            deployment='Detour cost-degradation upper one-sided95<=.005; median ratio<=.7,p95<=1; no Fixed3 simultaneous dominance; byseen/unseen and size'),
        caps=CAPS,top_request_total=48672,infrastructure_retry_total=32,absolute_request_cap=50000,
        clocks=dict(P0=1800,per_seed=14400,confirmation_and_analysis=10800,P5='Resume seed11622026 cumulative remaining14400; no extra reset clock'),
        P0=dict(engineering_requests_cap=96,planned_requests=42,optimizer_calls_cap=32,scope='18zero residual equality +24 on-policy engineering trajectories and finite new-math/failure tests; no scientific weight reused'),
        P5=dict(trigger='P3 reproducible RL increment; only2 registered variants',rounds=64,terminal_requests=3168,
            warm='Reuse192 current pi0 warm episodes only because initial policy identical; refit value targets for raw scale without new H',
            sampling='Same first64 problem schedules and paired seeds, own on-policy actions; never reuse main post-update trajectories'),
        ready_service='One source/model/static-source validation ready step; D1 actor each request; all methods same input/nav service; no dynamic reuse; report service vs full audit boundary separately',
        preparation_allocation='Registration/import, immutable shared bank/split feasibility preparation charged to P0; each seed loads/warms/trains/controls/selects under its14400s. Cold/hot final input/nav/IO plus numerical analysis under confirmation10800s. Parent watchdog launch-to-ready charged on startup. Git/checksum/publication administration separate.',
        sealed='No original60; oldS3 exposed retained; final216 tasks materialize only after configs/candidates frozen; unseen actual geometry hashes checked without replacement')
    write_new(HOME/'PROTOCOL.json',p)
    write_new(HOME/'AUTHORIZATION.json',dict(authorized_by='user',source='Prior explicit full-task request plus supplied exact file path',
        original_task_file=label(proposal),original_task_sha256=digest(proposal),protocol_sha256=digest(HOME/'PROTOCOL.json'),
        experiment=EXP,conditional_stages=['P0','P1','P2','P3 if registered gate','P4 if registered gate','P5 if registered gate'],generated_hash_pre_read_by_user=False,old_budget_reused=False,utc=now()))
    write_new(HOME/'PRIOR_MANIFEST_REFERENCE.json',dict(commit=BASE,path='project_management/GIT_MANIFEST.json',sha256=digest(ROOT/'project_management/GIT_MANIFEST.json'),
        retrieval='git show BASE:project_management/GIT_MANIFEST.json; incremental new audit, full old checksum at closeout'))
    sources=read(ROOT/'rc3_s3_connectivity_20261109/SOURCE_BINDING.json')['source_sha256'].copy()
    sources.update({label(f):digest(f) for f in HOME.glob('*.py')})
    write_new(HOME/'SOURCE_BINDING.json',dict(source_sha256=sources,artifacts={label(f):digest(f) for f in [proposal,*[HOME/n for n in ('PROTOCOL.json','DATA_SPLITS.json','DATA_REFERENCES.json','AUTHORIZATION.json','PRIOR_MANIFEST_REFERENCE.json')]]},
        base_models=models,environment=dict(python=__import__('platform').python_version(),torch=torch.__version__,numpy=np.__version__,device='cpu',threads=1,deterministic=True),utc=now()))
    write_new(HOME/'REGISTRATION_RESOURCE.json',dict(wall_s=time.perf_counter()-start,new_episodes_optimizer_H_nav=0,charged_to_P0=True,source_binding_sha256=digest(HOME/'SOURCE_BINDING.json')))
    print(digest(HOME/'SOURCE_BINDING.json'))


if __name__=='__main__':register()
