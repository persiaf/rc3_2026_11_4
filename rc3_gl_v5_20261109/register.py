"""Read-only registration then exact-source freeze, before any science call."""
import subprocess
import time
from .common import *


def register():
    begin=time.perf_counter()
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,encoding='utf8').strip()==BASE,'Wrong baseline')
    old=ROOT/'rc3_gl_v4_20261109';r=read(old/'INPUT_REGISTRY.json');batches=r['training_batches']
    require(len(batches)==128 and sum(map(len,batches))==512,'Wrong B schedule')
    refs=r['source_files'].copy();seen={};proof=[]
    for u,batch in enumerate(batches):
        actual=read(old/'runs/SCIENCE/B128/updates'/f'update_{u+1:04}.json')
        require(actual['specs']==batch and len(actual['view_identities'])==4,'B actual schedule differs')
        for slot,s in enumerate(batch):
            folder=OLD_RUN/'data/task_draws'/f"{s['map_seed']}_{s['task_seed']}_A{s['A_condition']}"
            receipt=read(folder/'receipt.json');key=receipt['map_key']
            seen.setdefault(key,dict(spec=s,folder=folder,physical_key=key,first_update=u+1,first_slot=slot,
                actual_B_view=actual['view_identities'][slot]))
    # Deterministic identity ordering; no costs or success subsets inspected.
    selected=[]
    for mode in ('mixed','blocked'):
        candidates=sorted((v for v in seen.values() if v['spec']['mode']==mode),key=lambda v:(v['physical_key'],v['first_update'],v['first_slot']))
        require(len(candidates)>=6,'Insufficient actual training physical layouts')
        selected.extend(candidates[:6])
    parents=[]
    for v in selected:
        s=v['spec'];folder=v['folder'];nav=OLD_RUN/'data/navigation'/(v['physical_key']+'.npz')
        for path in (folder/'scene.json',folder/'receipt.json',nav,nav.with_suffix('.json')):
            refs[label(path)]=dict(sha256=digest(path),bytes=path.stat().st_size)
        parents.append(dict(map_seed=s['map_seed'],mode=s['mode'],physical_key=v['physical_key'],
            first_actual_B_update=v['first_update'],first_actual_B_slot=v['first_slot'],actual_B_view=v['actual_B_view'],
            scene_file=label(folder/'scene.json'),scene_sha256=digest(folder/'scene.json'),
            navigation_file=label(nav),navigation_sha256=digest(nav),navigation_receipt_sha256=digest(nav.with_suffix('.json'))))
    require(len({p['physical_key'] for p in parents})==12,'Physical layout collision')
    declared=[];oldseeds={s['task_seed'] for b in batches for s in b}|{c['spec']['task_seed'] for c in r['new_development']}
    for pi,parent in enumerate(parents):
        task_seed=61109050000+pi;require(task_seed not in oldseeds,'Task seed collision')
        for w,h in ((4,3),(6,6),(8,8)):
            for opened in (False,True):
                spec=dict(role='GLV5_NEW_DEVELOPMENT',parent=pi,draw=0,map_seed=parent['map_seed'],task_seed=task_seed,
                    mode=parent['mode'],A_condition=pi%3,width=w,height=h,x0=(10-w)//2 if opened else 0,
                    y0=(10-h)//2 if opened else 0,endpoints='open_first_last_connected' if opened else 'closed_depot')
                declared.append(dict(id=f'GLV5_DEV_{len(declared):03}',group='new_task_trained_physical_layout',
                    spec=spec,physical_key=parent['physical_key'],size=[w,h],endpoint=spec['endpoints']))
    actors=read(old/'runs/SCIENCE/DEV216/ACTOR_INDEX.json')['models'][str(SEED)]
    actors={role:actors[role] for role in ('SL256','B128','C128')}
    for e in actors.values():
        for path in (ROOT/e['path'],ROOT/e['provenance']['training_checkpoint_path']):
            refs[label(path)]=dict(sha256=digest(path),bytes=path.stat().st_size)
    for path in (old/'METHOD_IDENTITY_AUDIT.json',old/'METHOD_IDENTITY_CLARIFICATION.json',old/'SOURCE_BINDING.json',old/'INPUT_REGISTRY.json'):
        refs[label(path)]=dict(sha256=digest(path),bytes=path.stat().st_size)
    write_new(HOME/'INPUT_REGISTRY.json',dict(training_batches=batches,training_schedule_sha256=identity(batches),
        data_schedule_rng_protocol=OLD_PROTOCOL,new_development_parents=parents,new_development=declared,
        source_files=refs,actors=actors,selection_rule='Separate mixed/blocked actual B128 layouts; sort physical map_key lexicographically, earliest actual update/slot tie-break; first6 each. No route/cost filtering.',
        actual_training_unique_layouts=len(seen),evaluation_layouts=12,evaluation_task_draws=12,evaluation_cases=72,
        interpretation='New tasks on already trained physical layouts, single optimization seed; not unseen environment or S3.'))
    p=dict(schema='rc3-gl-v5-fixed-greedy-reference-v1',experiment_id=EXPERIMENT,baseline=BASE,
        fixed_goal='Large complex parent-scene DRL training, frozen rapid complete local planning; SL and rules auxiliary only',
        network_reward_H_precision_unchanged=True,H_identity=H_ID,registry_sha256=REGISTRY,old_protocol=OLD_PROTOCOL,
        training=dict(seed=SEED,start='Exact original SL256',fresh_Adam=True,updates=128,batch=4,
            trajectories=3,weights=[1.,.25,.25],normalize_weight_sum=1.5,all_first_logprobs=True,
            only_major_variable='Replace detached LOO by action-independent frozen SL256 greedy+sameH cost on identical input',
            objective='mean_batch sum_r w_r (J_sample_r-J_ref).detach()*logP(full trajectory_r)/1.5',
            baseline='Frozen SL256, D1 actor-only load per input, fresh greedy encode/prefix, sameH, no policy RNG consumption or promotion. Reference and sample terminals have disjoint exact128LRU caches.',
            negative_advantages='Retain all signs; no clamping/filtering',rng='Original v3 NamedStreams seed11622026 phaseRL/update/slot/primary,forced_low,forced_high. B slot names retained, no forced actions.',
            distribution='Exact B128 frozen512 views,64full and64local batches',
            optimizer=read(old/'PROTOCOL.json')['mechanism']['optimizer'],checkpoint='0,32,64,96,128 full Adam/RNG/ref+sample state; full gradients/new weights every update'),
        evaluation=dict(cases=72,physical_layouts=12,balanced_modes=[6,6],task_draws=12,seed=SEED,
            methods=['SL256','B128','C128','G128','Fixed3','Detour16'],requests=432,
            neural='D1 actor-only each request, greedy+sameH, explicit study_method/checkpoint_role/tensor/source',
            native='Original endpoint adapters, no extra common H',
            navigation='Identical saved physical D source for every method, only navigation cached. First saved-D read paid in actual request, timing descriptive, not paired deployment proof.',
            statistics='Per true physical layout average6 sizes/endpoints; compute (J_control-J_G)/J_control. Equal weight12 parents, bootstrap20000 draws stratified mixed6/blocked6, PCG64 seed11642026, percentile2.5/97.5 linear; raw separately.',
            gate=dict(controls=['SL256','B128','C128'],metric='J_after_H',mean_relative_gain_min_strict=0.,lower95_min_strict=0.,
                logical='AND all3 plus all432 feasible/finite/identity-valid; only single-seed mechanism candidate, no final deployment/generalization claim'),
            no_top1_gate=True,not_unseen_environments=True),
        budgets=dict(sample_requests=1536,reference_requests=512,evaluation_requests=432,total_requests=2480,
            optimizer_updates=128,shared_wall_s=7200,engineering_optimizer=0,
            engineering='Synthetic logits only, no real policy/nav/native/H, max120s inside shared7200',
            charging='Count requests including cache hits before work. Include registration CPU/IO, all runtime preparation, baseline greedy,H, training, validation, checkpoints and IO. Git publication/read-only post-run audit are separate maintenance.'),
        mathematics='State-only frozen baseline subtracts zero expected score, preserving expected J objective; clipping/Adam/finite-sample training do not guarantee monotonic greedy quality.',
        stop='Any source/role/infeasible/nonfinite/validation/IO/cap error retains all debits and last closed checkpoint, no replacement/reseed/free retry, no resetting experiment folders.',
        historical='Old role files and mapping preserved, old108 labels not rewritten; no A/B/C retrain, deployment-grid rerun, oldS3 missing refill or original60/newS3 semantic access.')
    write_new(HOME/'PROTOCOL.json',p)
    write_new(HOME/'AUTHORIZATION.json',dict(authorized_by='user',experiment_id=EXPERIMENT,protocol_sha256=digest(HOME/'PROTOCOL.json'),
        source='Current explicit GL-v5 task authorizes registration then bounded execution',generated_hash_pre_read_by_user=False,
        stages=['synthetic engineering','G128','72x6 fixed endpoint evaluation'],old_budget_reused=False,utc=now()))
    write_new(HOME/'READ_ONLY_DATA_SOURCE_BINDING.json',read(old/'READ_ONLY_DATA_SOURCE_BINDING.json'))
    oldmanifest=read(ROOT/'project_management/GIT_MANIFEST.json')
    write_new(HOME/'FROZEN_PRIOR_EVIDENCE.json',dict(baseline=BASE,files={n:v for n,v in oldmanifest['files'].items() if n.startswith('rc3_')}))
    (HOME/'IMPLEMENTATION_REVIEW.md').write_text('''# GL-v5实施评审

GO_FOR_AUTHORIZED_BOUNDED_EXECUTION，入口仍要求本轮合成数学/身份测试PASS。固定研究目标和全部旧证据保持。

唯一主要训练变量为冻结SL256贪心+H的状态成本基线；G仍从同一SL256 fresh Adam训练自然起点REINFORCE。三轨迹1/.25/.25、首动作全似然、batch4、lr1e-4、clip1、网络、MDP、终局-J、mask、H和50/50 full/local不变。参考无梯度、无晋升、独立前缀和终局缓存，不使用采样RNG。所有优势含负值均训练。状态基线保留期望成本目标，无贪心单调保证。

实际B128日程512输入精确复用；仅按实际出现物理身份排序选mixed/blocked各6布局，12新任务draw、72窗口与端点事前确定，全部输入在首优化前封存。完整父地图、边界候选与scope_r1代理检查保持。已训练布局不称未见环境；单种子结论仅机制筛选。三个已有对照固定终点只读取，旧角色默认值不重写。

来源使用冻结checkpoint、D1张量包、显式科学角色、原registry/H和旧兼容加载器。reference和G缓存分账；请求含命中先扣。G1536、参考512、评价432、总2480，最多128更新，注册与数学检查费用加运行共用7200秒。每32保存闭合权重/Adam/RNG/计数，失败和预算不可清零。独立新模块不回写旧协议/训练源码。

风险为固定基线导致优势/梯度方差或裁剪增多、单种子敏感、12个训练布局的新任务可能不可行；发生真实不可行必须停，不能换任务。F&A是代理可达与安全检查，不代表真实IK或关节轨迹认证。工程PASS不等于质量通过。正式结果须AND同时改善共同SL、B、C，不能只挑退化对照。
''',encoding='utf8')
    from rc3_s3_evaluator.provenance import environment
    source=read(old/'SOURCE_BINDING.json')['source_sha256'].copy();source.update({label(q):digest(q) for q in HOME.glob('*.py')})
    write_new(HOME/'SOURCE_BINDING.json',dict(schema='rc3-gl-v5-execution-binding-v1',source_sha256=source,
        artifacts={n:digest(HOME/n) for n in ('PROTOCOL.json','INPUT_REGISTRY.json','AUTHORIZATION.json','READ_ONLY_DATA_SOURCE_BINDING.json')},
        environment=environment(),H_identity=H_ID,old_GL_v4_source_sha256=digest(old/'SOURCE_BINDING.json'),utc=now()))
    write_new(HOME/'REGISTRATION_ACCOUNTING.json',dict(wall_s=time.perf_counter()-begin,charged_to_shared7200=True,
        model_forward_optimizer_navigation_native_H_teacher=0,actual_training_layouts=len(seen),selected_layouts=12,
        declared_new_cases=72,source_binding_sha256=digest(HOME/'SOURCE_BINDING.json'),utc=now()))
    print(dict(source_binding=digest(HOME/'SOURCE_BINDING.json'),layouts=12,cases=72,training_views=512))


if __name__=='__main__':register()
