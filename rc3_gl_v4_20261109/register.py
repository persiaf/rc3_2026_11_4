"""Freeze GL-v4 before export, scene generation, inference, or optimization."""
import time
import subprocess
import numpy as np
from .common import *


def register():
    began=time.perf_counter()
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,encoding='utf8').strip()==BASE,'Wrong baseline')
    from rc3_global_local_trainer.data import TrainingRegistry,TrainingSchedule
    from rc3_global_local_trainer.rng import NamedStreams
    reg=TrainingRegistry(ROOT/'rc3_global_local_execution_v3/REGISTRY.json')
    schedule=TrainingSchedule(reg,NamedStreams(OLD_PROTOCOL,SEED))
    batches=[schedule.batch(u) for u in range(128)]
    require(sum(s['width']==10 for b in batches for s in b)==256,'Wrong50/50 mix')
    c_rng=NamedStreams(OLD_PROTOCOL,SEED).numpy('continuedSL_data')
    teachers=[c_rng.integers(0,256,4).tolist() for _ in range(128)]
    cases=read(ROOT/'rc3_s3_evaluator/ENGINEERING_REGISTRATION.json')['cases']
    require(len(cases)==6,'Exactly six existing engineering DEV cases required')
    refs={}
    for spec in [s for b in batches for s in b]:
        folder=OLD_RUN/'data/task_draws'/f"{spec['map_seed']}_{spec['task_seed']}_A{spec['A_condition']}"
        for name in ('receipt.json','scene.json','arrays.npz'):
            path=folder/name;require(path.is_file(),'Prespecified saved train input missing')
            refs[label(path)]=dict(sha256=digest(path),bytes=path.stat().st_size)
    parents=[];seen=set()
    for spec in reg.development_specs():
        if spec['map_seed'] in seen:continue
        seen.add(spec['map_seed'])
        folder=OLD_RUN/'data/task_draws'/f"{spec['map_seed']}_{spec['task_seed']}_A{spec['A_condition']}"
        receipt=read(folder/'receipt.json');nav=OLD_RUN/'data/navigation'/(receipt['map_key']+'.npz')
        for path in (folder/'scene.json',folder/'receipt.json',nav,nav.with_suffix('.json')):
            refs[label(path)]=dict(sha256=digest(path),bytes=path.stat().st_size)
        parents.append(dict(map_seed=spec['map_seed'],mode=spec['mode'],physical_key=receipt['map_key'],
            scene_file=label(folder/'scene.json'),scene_sha256=digest(folder/'scene.json'),
            navigation_file=label(nav),navigation_sha256=digest(nav),navigation_receipt_sha256=digest(nav.with_suffix('.json'))))
    require(len(parents)==3,'Expected three original DEV physical layouts')
    declared=[]
    for pi,parent in enumerate(parents):
        for draw in range(2):
            task_seed=61109000000+10*pi+draw
            require(all(task_seed not in p['task_seeds'] for p in reg.parents) and
                task_seed not in [d['task_seed'] for d in reg.development],'New task seed collision')
            for w,h in ((4,3),(6,6),(8,8)):
                for center in (False,True):
                    spec=dict(role='GLV4_NEW_DEVELOPMENT',parent=pi,draw=draw,map_seed=parent['map_seed'],
                        task_seed=task_seed,mode=parent['mode'],A_condition=(pi+draw)%3,
                        width=w,height=h,x0=(10-w)//2 if center else 0,y0=(10-h)//2 if center else 0,
                        endpoints='open_first_last_connected' if center else 'closed_depot')
                    declared.append(dict(id=f'GLV4_DEV_{len(declared):03}',group='fresh_task_existing_DEV_layout',spec=spec,physical_key=parent['physical_key'],size=[w,h],endpoint=spec['endpoints']))
    require(len(declared)==36,'Wrong new DEV count')
    write_new(HOME/'INPUT_REGISTRY.json',dict(training_batches=batches,C_teacher_batches=teachers,
        deployment_cases=cases,new_development=declared,new_development_parents=parents,source_files=refs,
        data_schedule_rng_protocol=OLD_PROTOCOL,training_schedule_sha256=identity(batches),
        interpretation='512 identical A/B old training-bank views;36 NEW task-window cases from6 new task draws on3 previously exposed DEV layouts, not new S3 or held-out tests'))
    rng=np.random.Generator(np.random.PCG64(6110911622));requests=[]
    # Randomize lifecycle order per paired case/kind/repeat block; startup is separate.
    for c in cases:
        for kind,repeats in (('quality',1),('cold',5),('hot',5)):
            for rep in range(repeats):
                for mode,role in rng.permutation([(m,r) for m in ('D0','D1','D2') for r in ('SL256','RL512')]).tolist():
                    q=dict(case_id=c['id'],seed=SEED,mode=mode,role=role,
                        method='frozen_supervised' if role=='SL256' else 'RL_plus_H',kind=kind,repeat=rep)
                    requests.append(dict(q,id=identity(q)))
    require(len(requests)==396 and len({q['id'] for q in requests})==396,'Wrong deployment grid')
    write_new(HOME/'DEPLOYMENT_GRID.json',dict(order_seed=6110911622,requests=requests))
    protocol=dict(schema='rc3-gl-v4-natural-start-and-deployment-v1',experiment_id=EXPERIMENT,baseline=BASE,
        fixed_goal='Large complex parent-scene DRL training, frozen rapid complete variable-size local planning; SL/rules auxiliary only',
        old_v3_protocol_sha256=OLD_PROTOCOL,registry_sha256=REGISTRY,H_identity=H_ID,
        scope='New independent experiment; original S1/S2/S3 bytes, failures, exposure and negative claims remain frozen',
        deployment=dict(seed=SEED,cases=6,models=['SL256','RL512'],modes=['D0','D1','D2'],
            quality_repeats=1,cold_repeats=5,hot_repeats=5,requests_cap=396,wall_s_cap=1800,optimizer_cap=0,
            export='Policy tensors only, architecture configuration and exact source inventory. Two fixed seed11622026 endpoints; no quantization/compile/precision change.',
            lifecycle='D0 full checkpoint each request;D1 exact lightweight package each request;D2 same lightweight parameters resident. Identical dynamic task work every request.',
            startup='One common worker hosts all lifecycle catalogs; each catalog readiness/model startup measured separately before paired requests, paid inside1800. Process/import/admission startup also recorded. Model state and cold/hot navigation are separate factors.',
            order='Randomized6 lifecycle/role combinations in every case/cache/repeat block; PCG64 seed6110911622',
            navigation='Each lifecycle has its own identical-source navigation cache namespace; quality/cold never consult or populate hot cache. Only navigation warmed; prewarming paid outside request timer inside phase wall.',
            timing='Complete request includes input/source checks, navigation, full parent/window extraction, model access, fresh features/encoding, greedy,H,export,validation,weight verification,artifact IO. Startup never silently included/excluded as a cache label.',
            equivalence='Exact named tensors/dtypes/shape, actual mask/logits hashes, greedy raw/final route,assignments,J/L/K across all3 lifecycles; no task embedding,route,H result caches'),
        mechanism=dict(seed=SEED,start='Exact own SL256; three separate tensor storage branches and fresh Adam, old RL512 never continued',
            updates_each=128,arms=['A128','B128','C128'],total_updates_cap=384,batch=4,
            A='Original primary/forced_low/forced_high; normalized weights1/.25/.25; forced first logprobs excluded; detached LOO',
            B='Only forced-first sampling changes: all3 trajectories sample all actions naturally; all first-action logprobs included; same normalized1/.25/.25 and detached LOO',
            C='Original256 teacher pool, route-mean teacher-forcing masked CE, batch4,128 fresh-Adam updates; no teacher/native/H queries',
            training_schedule_sha256=identity(batches),schedule='Exactly original seed11622026 first128 S2 batches, frozen512 views; protocol hash changes never alter data schedule',
            rng='Separate NamedStreams objects using original immutable protocol SHA and seed. Same primary/forced_low/forced_high RNG slot names in A/B for common random numbers; names in B are only stream slots, not forced actions. Each row has distinct generator; record seeds/states.',
            mixture='64full10x10 batches,64local batches4x3/6x6/8x8; identical A/B parent,task,A,window,endpoints',
            optimizer=dict(algorithm='Adam',lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=0.,clip_global_norm=1.,gamma=1.,entropy_weight=0.,advantage_standardization=False),
            terminal_requests_per_RL_arm_cap=1536,RL_terminal_requests_cap=3072,
            terminal='Original full H/export/validation and128-entry exact input/H/raw-route LRU, separately owned per arm. Every trajectory request incl hit predebited; actual H,internal search,cache hits separately counted.',
            checkpoint='initial and every32 complete updates, full weights/Adam/RNG/ledger/terminal state; gradients and final parameters every update. No best checkpoint selection.'),
        fixed_endpoint_development=dict(cases=36,models=['SL256','A128','B128','C128','Fixed3','Detour16'],requests_cap=216,
            data='Six new task draws on3 original DEV physical geometries,36 exact window/endpoint declarations frozen before training. No exposed S3 or original60 task input is used. This is fresh-task DEV, not new-layout independent generalization.',
            freeze='Generate/save new scene,F/A,input receipt and36 view identities at SCIENCE start, before any optimizer. Preserve full parent geometry and boundary candidates,scope_r1 local target hard/full-parent diagnostic.',
            navigation='Identical saved physical-navigation source for all six methods; bank loading/preparation and all validation/IO paid in7200. No free task embeddings or routes.',
            neural='Frozen exact128 endpoint actor package per request,greedy+H,raw and postH J/L/K. NativeFixed3/Detour16 endpoint adapters, no extra commonH.',
            statistics='3 physical-parent clusters; average all12 cases per parent then parent-relative gains;20000 PCG64 parent bootstrap seed11632026. Report per-size/endpoint, no3-seed or final deployment inference.',
            candidate_screen='B postH mean improvement>0 and bootstrap95lower>0 versus EACH A128,sharedSL256,C128,all prescribed results feasible/numerically valid. Raw comparisons separately disclosed. Single-seed mechanism screen only, no final generalization/deployment success.'),
        science_budget=dict(training_and_evaluation_requests_cap=3288,total_optimizer_updates_cap=384,shared_wall_s_cap=7200,
            includes='New DEV scene/FA preparation,source reads,navigation,all3 training,checkpoint/export/validation/evaluation/IO and parent closure'),
        engineering=dict(new_tests='Synthetic detached LOO/weighted objective,first-action inclusion,gradient and schedule/source checks only; no real model forward/optimizer/nav/native/H',
            real_optimizer_cap=0,real_model_inference_cap=0,wall_s_cap=120,charged_inside_deployment1800=True),
        stop='Source mismatch,infeasible declared input,nonfinite,validation/IO failure,or budget cap preserves predebits and evidence and stops affected configuration; no replacement/reseed/free retry or resetting dirs. One-shot fixed experiment roots.',
        authority='Current user GL-v4 task explicitly authorizes registration then bounded execution; not claiming user pre-read generated hashes or changing old approved v3')
    write_new(HOME/'PROTOCOL.json',protocol)
    write_new(HOME/'AUTHORIZATION.json',dict(authorized_by='user',experiment_id=EXPERIMENT,protocol_sha256=digest(HOME/'PROTOCOL.json'),
        basis='Current message: GL-v4部署拆分与自然起点DRL对照；首次实际调用前冻结新协议、实例日程和资源上限，随后在本任务范围内直接执行。',
        phases=['DEPLOYMENT396','A128','B128','C128','NEW_DEV216'],old_budget_reused=False,generated_hash_pre_read_by_user=False,utc=now()))
    index=read(ROOT/'rc3_s3_evaluator/S3_MODEL_INDEX.json');side=read(ROOT/index['models'][str(SEED)]['SL256']['sidecar_path'])
    binding={k:v for k,v in side['binding'].items() if k not in ('optimizer_seed','run','phase')}
    write_new(HOME/'READ_ONLY_DATA_SOURCE_BINDING.json',dict(binding=binding,scope='Historical data interpretation only; new AUTHORIZATION supplies GL-v4 execution authority'))
    old=read(ROOT/'project_management/GIT_MANIFEST.json')
    write_new(HOME/'FROZEN_PRIOR_EVIDENCE.json',dict(baseline=BASE,files={n:v for n,v in old['files'].items() if n.startswith('rc3_')},old_S3_exposure_and_two_missing_preserved=True))
    write_new(HOME/'CODE_REUSE_PROVENANCE.json',dict(reference_commit='c5f47ec3518bd2b58f26e8efb2d4737676632dba',
        source_files=['rc3_actor_pg_20261109/common.py','rc3_actor_pg_20261109/artifacts.py','rc3_actor_pg_20261109/mechanism.py'],
        copied_scope='Engineering primitives only; no prior actor performance results/weights reused. B corrected to this task original1/.25/.25; all science new and independent of other branch.'))
    write_new(HOME/'REGISTRATION_ACCOUNTING.json',dict(utc=now(),wall_s=time.perf_counter()-began,model_forward_optimizer_navigation_native_H_teacher=0,
        saved_training_specs=512,new_DEV_declarations=36,deployment_requests=396))
    (HOME/'IMPLEMENTATION_REVIEW.md').write_text('''# GL-v4实施前评审

GO_FOR_AUTHORIZED_BOUNDED_EXECUTION，正式入口还须自动验证准确源码与新起点数学检查PASS，未执行阶段不记为成功。

固定研究目标保持大规模父场景DRL训练—冻结策略快速局部完整规划。v3、scope_r1、旧训练/模型/结果和两个缺失槽位原字节保留，新分支自925a0a1a建立。本任务明确授权新协议冻结后执行，不冒充用户事前读过生成哈希，不复用旧阶段授权或剩余额度。

部署：仅SL256/RL512两固定模型，参数/架构/来源轻量包；原完整checkpoint留存。三载入方式同精度、同实际mask/logits、贪心路线、J/L/K；模型常驻与导航冷热分开，396请求与1800秒含启动、导出、数学检查、准备和IO。只缓存参数/导航，不缓存任务编码/路径/H。

数学：A原三个起点与1/.25/.25；B只移除两强制首动作，三条完整自然策略轨迹的首动作log probability全部参与，保留加权1.5归一与detached LOO。独立随机流、共同512训练问题、50%full/local；C原256教师CE，三fresh Adam。学习率1e-4、clip1、网络/奖励/H均原样。纯合成枚举验证LOO期望梯度和实际A/B函数梯度，不增加真实optimizer或科学探针。

数据：训练与教师只读原来源，兼容身份重载不放松。36新任务窗口来自6个新task draw、3个既有DEV几何；任务种子、A、窗口、端点在首调用前固定，实际输入身份在训练前封存。它是新任务开发筛选，不是新布局泛化，也不将旧S3变回未见测试。保持全父地图与全局上下文，实际目标可服务性硬检查，所有合法边界候选保留。

资源与停止：每A/B最多1536次终局请求，含命中先扣；三组384更新、216终点评价，共3288请求与7200秒共享上限。每32闭合更新保存权重/Adam/RNG/资源。任何来源、不可行、非有限、验证或IO/预算错误保留原账并停止，不补抽、换seed、重置目录或临时改算法。

风险：仅单优化种子与3个DEV物理布局；时间受本机软件/IO影响；工程提速不属于RL质量增益；H收益独立列示；代理F&A不等于真实IK/关节轨迹认证。仅B优于A不足，必须同时检查共同SL与C，且不能宣称最终泛化或部署成功。
''',encoding='utf8')
    print(dict(training_views=512,new_DEV_cases=36,deployment_requests=396))


def freeze():
    from rc3_s3_evaluator.provenance import environment
    source=read(ROOT/'rc3_s3_connectivity_20261109/SOURCE_BINDING.json')['source_sha256'].copy()
    source.update({label(p):digest(p) for p in HOME.glob('*.py')})
    write_new(HOME/'SOURCE_BINDING.json',dict(schema='rc3-gl-v4-execution-binding-v1',source_sha256=source,
        protocol_sha256=digest(HOME/'PROTOCOL.json'),input_registry_sha256=digest(HOME/'INPUT_REGISTRY.json'),
        deployment_grid_sha256=digest(HOME/'DEPLOYMENT_GRID.json'),authorization_sha256=digest(HOME/'AUTHORIZATION.json'),
        data_source_binding_sha256=digest(HOME/'READ_ONLY_DATA_SOURCE_BINDING.json'),
        model_index_sha256=digest(ROOT/'rc3_s3_evaluator/S3_MODEL_INDEX.json'),
        registry_sha256=REGISTRY,H_identity=H_ID,environment=environment(),utc=now()))
    print(digest(HOME/'SOURCE_BINDING.json'))


if __name__=='__main__':
    import sys
    {'register':register,'freeze':freeze}[sys.argv[1]]()
