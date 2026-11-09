"""Evidence formatting/metadata only; never executes an S3 grid or navigation."""
import json
import time
from rc3_s3_evaluator.common import ROOT,read,write_new,digest,label,now
from rc3_s3_connectivity_20261109.scan import HOME


def geometry():
    start=time.perf_counter()
    import numpy as np
    from rc3_s3_connectivity_20261109.diagnose import inputs
    scene,F,A,spec=inputs();w=read(HOME/'CONNECTIVITY_WITNESS.json')
    with np.load(HOME/'diagnosis/navigation.npz',allow_pickle=False) as z:
        D=z['costs'].copy();coords=z['coordinates'].copy();services=z['service_nodes'].copy();pred=z['predecessors'].copy()
    radius=scene['robot']['base_radius']+scene['robot']['clearance'];records=[]
    for task in w['full_parent_closed_depot']['unservable_tasks']:
        for record in task['same_arm_FA_candidates']:
            j=record['global_candidate_id'];c=scene['candidates'][j];blockers=[]
            for kind,objects in (('table',scene['tables']),('obstacle',scene.get('obstacles',[]))):
                for i,o in enumerate(objects):
                    box=[o['xmin']-radius,o['ymin']-radius,o['xmax']+radius,o['ymax']+radius]
                    if box[0]<=c['x']<=box[2] and box[1]<=c['y']<=box[3]:
                        blockers.append(dict(kind=kind,index=i,original_bounds={k:o[k] for k in ('xmin','xmax','ymin','ymax')},inflated_bounds_xyxy=box))
            records.append(dict(task_id=task['global_task_id'],candidate_id=j,service_index=j+1,
                navigation_graph_node=int(services[j+1]),point_xy=[c['x'],c['y']],corridor_anchor=c['corridor_anchor'],
                depot_graph_node=int(services[0]),depot_predecessor_at_service=int(pred[0,services[j+1]]),
                depot_to_candidate_finite=bool(np.isfinite(D[0,j+1])),candidate_to_depot_finite=bool(np.isfinite(D[j+1,0])),
                same_arm_indices=record['same_arm_indices'],point_inside_inflated_blockers=blockers))
    ids=w['local_view']['candidate_ids_global'];nodes=[0,*[j+1 for j in ids]]
    assert np.isfinite(D[np.ix_(nodes,nodes)]).all()
    write_new(HOME/'CONNECTIVITY_GEOMETRY_EVIDENCE.json',dict(
        original_witness_sha256=digest(HOME/'CONNECTIVITY_WITNESS.json'),diagnostic_navigation_npz_sha256=digest(HOME/'diagnosis/navigation.npz'),
        base_radius=scene['robot']['base_radius'],clearance=scene['robot']['clearance'],inflation_radius=radius,
        records=records,target_candidate_count=len(ids),target_endpoint_candidate_submatrix_all_finite=True,
        existence_argument='Undirected unchanged navigation: each of23 targets has effective reachable candidate; all66 candidates and depot share finite pairwise D. A finite covering walk exists without constructing/evaluating any route.',
        original_D_recovery_claim=False,new_navigation_calls=0,new_routes_or_costs=0))
    write_new(HOME/'GEOMETRY_RESOURCE.json',dict(read_compute_save_wall_s=time.perf_counter()-start,
        navigation_calls=0,task_generation=0,FA_generation=0,model=0,native=0,H=0,optimizer=0))


def contract_and_reports():
    w=read(HOME/'CONNECTIVITY_WITNESS.json');b=read(HOME/'BASELINE_IMPORT.json');diag=read(HOME/'diagnosis/FINAL.json')
    assert w['outcome']=='A_OUTSIDE_TARGET_ONLY'
    amendment=dict(schema='rc3-s3-input-contract-amendment-r1',date='2026-11-09',
        status='IMPLEMENTED_UNDER_EXPLICIT_USER_CONDITIONAL_A_TASK',
        user_instruction_basis='2026-11-09 actual user task: if only outside-window tasks reject, propose and implement uniform S3 target-scope check; retain full-parent diagnostic and actual-target hard check',
        base_protocol_sha256='0c5bc7d7a9557edf2af76f51a9f418550f0b4ce1ec6a8b47ddea95c6256e78bb',
        old_evaluation_binding_sha256='b348fb7261c8e6b8c5b2391649c8bf8d19dfca31f9b26f473e62eb4b82232492',
        old_persistence_binding_sha256='767f13bd4619bef37501db997296982baf6b1565cb25ea8cb964c2e39d381bea',
        diagnosis_witness_sha256=digest(HOME/'CONNECTIVITY_WITNESS.json'),
        old_rule='Full parent10x10 closed-depot task bank must pass before every S3 local view',
        new_rule='Report all full-parent closed-depot unservable task IDs; enforce original same-arm F&A/navigation/endpoints for every declared local target; no case/method exceptions',
        hard_target_check_retained=True,full_parent_geometry_and_obstacles_retained=True,global_context_and_IDs_retained=True,
        all_effective_connected_service_candidates_retained_including_outside_window=True,
        methods=['Fixed3','Detour16','untrained','frozenSL','continuedSL','RL_raw','RL_plus_H'],
        scenario_or_task_seed_changed=False,A_changed=False,target_tasks_deleted=False,endpoints_changed=False,
        neural_architecture_reward_optimizer_H_changed=False,training_or_model_endpoints_changed=False,
        nominal_registry_changed=False,scientific_input_admission_changed=True,
        scientific_impact='Previously rejected outside-target infeasibility can now admit a declared window; these states had no accepted original result. This is an input-contract amendment, not solely engineering maintenance. Earlier accepted views unchanged; new version/tag and no unamended-v3 complete-grid claim.',
        original_v3_bytes_preserved=True,original_two_failed_slots_refilled=False,
        current_task_grid_execution=False,diagnostic_navigation_not_free_scientific_cache=True)
    write_new(HOME/'INPUT_SCOPE_AMENDMENT.json',amendment)
    (HOME/'S3_042_CONNECTIVITY_DIAGNOSIS.md').write_text(f'''# S3_042端点连通服务诊断

**目标4×3窗口可行，阻断任务只在窗口外。** 在同一保存scene/F/A和明确标记的诊断重算D下，23个目标任务全部具有同臂F&A且往返实际端点可达的候选；目标候选66个，与depot组成的子矩阵全部有限。未运行模型、规划器、H或计算路线J。

全父233任务中唯一不可服务的是全局task **221**、table **94**（ix=4,iy=9），不在corner窗口x=0..3、y=0..2的12张目标桌中。它只有候选752与753：752允许臂0/1，753允许臂0；端点过滤后均为0个候选。

| 候选ID | D中的服务索引 | 导航图节点 | depot前驱 | depot→候选 / 候选→depot |
|---|---:|---:|---:|---|
| 752 | 753 | 1625 | -9999 | ∞ / ∞ |
| 753 | 754 | 1627 | -9999 | ∞ / ∞ |

两个停靠点分别为(11.2067711873,21.3634121369)、(11.6889818665,21.3634121369)，均位于同一障碍物按机器人半径+clearance=0.4m膨胀后的矩形内：[11.0123035819,11.9323035819]×[20.4842645555,21.6842645555]。冻结导航器会剔除与该膨胀矩形相交的边；包含这些端点的连接均不可用。这与D=∞和前驱−9999一致，不能通过降低安全间隙修复。

D为801×801，坐标1721×2，前驱801×1721；全局task/table/candidate ID连续，服务索引j+1映射正确。有限性关系和数值对称、非负性、零对角及端点连通检查通过，769条depot到有限服务节点的前驱链长度与D一致。800候选中768个与depot连通、32个不连通；仅task221的全部有效候选都落入失联集合。

已先搜索99份带来源的导航产物及3份S0父产物，没有同map_key/导航源码的可验证完整产物。因此事前登记最多1次冻结navigation、300秒：**实际调用1次**，导航函数{w['navigation_wall_s']:.9f}秒，完整诊断{diag['all_diagnostic_wall_s']:.9f}秒。任务/FA生成、模型、native、H、教师、optimizer全0。没有重新生成或替换scene/F/A。

原D哈希不存在，`original_D_sha256=null`；当前D身份为`{w['diagnostic_D_sha256']}`，导航内容身份`{w['navigation_identity']}`。这是诊断重算，**不是原D逐字节恢复**。完整D、坐标、服务节点、前驱及边计数保存在[导航NPZ](diagnosis/navigation.npz)和[身份receipt](diagnosis/navigation.json)。

数字证据：[CONNECTIVITY_WITNESS.json](CONNECTIVITY_WITNESS.json)、[几何证据](CONNECTIVITY_GEOMETRY_EVIDENCE.json)、[调用预登记](DIAGNOSTIC_REGISTRATION.json)、[诊断完整时钟](diagnosis/FINAL.json)。本结论只适用于保存的S3_042及冻结代理约束，不构成DRL性能结果或其他未执行窗口的可行性保证。
''',encoding='utf8')
    (HOME/'INPUT_SCOPE_DECISION.md').write_text('''# S3输入范围修订决定

采用用户明确条件A：全父不可行仅来自窗口外task221，实际23个目标任务均可服务。新模块scope.py统一供七方法及全部108声明案例使用；没有单案例、方法或候选ID特判。

修改前，全父10×10闭合depot检查是局部输入的硬前提。修订后，该全父检查保留为完整诊断；实际声明窗口的全部目标任务仍必须通过原同臂F&A、导航、实际端点和覆盖检查。目标真正不可行仍拒绝并保留具体目标集合，绝不交给模型或规则伪造有效结果。

原父地图及障碍/通道不裁剪；全局上下文、全F/A、全局ID、端点和所有合法窗口外服务候选保留。S1/S2训练输入契约、MDP/奖励、网络、优化器、冻结权重终点、H和原生基线均不改。旧源码及v3原文件不回写。

**这确实改变S3输入接纳契约。** 原先因窗口外任务拒绝的状态会被接纳，不能称为完全无科学影响。对原全父检查已通过的输入，新旧局部视图的任务、候选、D、父上下文和身份保持；新接纳状态没有旧有效路线可供声称性能等价。名义108案例、种子、A、端点、门槛数字不变，后续结果标记v3+scope_r1，不伪称未修订v3完整结果。

依据：[独立修订JSON](INPUT_SCOPE_AMENDMENT.json)、[诊断](S3_042_CONNECTIVITY_DIAGNOSIS.md)、[源码](scope.py)。本轮只完成诊断、修正、回归和恢复处置，没有运行剩余科学网格；未来默认入口先--check，只有显式--execute才进行真实请求。
''',encoding='utf8')
    (HOME/'RECOVERY_DECISION.md').write_text(f'''# 一次恢复处置

将旧S3_042失败明确处置为“旧全父硬前提拒绝，原收费槽仍缺失；范围修订后允许其他未执行请求继续”。不重批未变的原研究设计，不覆盖v3，不静默跳过新的目标不可行。

继承原2059预扣、2057成功、2失败，精确预算账2896.9277158999903秒，原准确旧阶段墙钟仍null。首种子剩余3233请求/11503.07228410001秒；其他两种子各5292/14400。诊断1次navigation及其完整{diag['all_diagnostic_wall_s']:.9f}秒独立记账，不退款、不清账，也不把诊断D免费放入正式冷/热导航缓存。

两个旧失败ID均列入已消费终态：S3_003/Fixed3/cold/repeat1与S3_042/RL_plus_H/quality/repeat0。它们不进入新请求计算，缺失值不由其他重复、种子或均值填补。当前新增补齐尝试登记为0。若以后要补齐，必须另列付费尝试及预算变化，不混入原剩余额度。

新续段固定在本模块runs/同一实验ID/scope_r1；原persistence_r1日志/结果/收据不再写入。先核对2515冻结文件、源/模型/授权、两段哈希链和全部旧终态，再按原请求顺序只选UNEXECUTED。独立来源绑定、目录外activation、OS单写入者、失败独立收据和累计时钟防止换目录清账。新错误或脏边界继续停止，不自动免费重试。

显式执行入口：`.\\rc3_learning\\.venv\\Scripts\\python.exe -B -X utf8 -m rc3_s3_connectivity_20261109.resume --execute`。默认无--execute仅来源/预算检查。本轮未执行此命令，科学请求新增0；不使用已经看到的路线成本挑选输入、权重或方法。

即使余下全部请求成功，最多15874份有效结果/15876个原槽：quality最多2267/2268，cold6803/6804，hot6804/6804。质量缺失仍在unseen4×3，冷计时缺失仍在seen6×6；相关格为INCOMPLETE，不宣称完整总体AND或未修订v3部署通过。新/旧输入契约及计时版本分别标记，旧有效输入语义兼容与计时开销相等是不同命题；范围诊断开销属于完整计时，分层保留后才能解释合并时间。

准确的源码、回归和本处置绑定见[SOURCE_BINDING.json](SOURCE_BINDING.json)与[机器处置](RECOVERY_DECISION.json)。
''',encoding='utf8')
    print('Saved diagnosis, scientific input-contract amendment and explicit non-refill recovery plan.')


if __name__=='__main__':
    import sys
    {'geometry':geometry,'reports':contract_and_reports}[sys.argv[1]]()
