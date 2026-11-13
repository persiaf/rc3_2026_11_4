"""Close the completed D6 cycle and refresh only this task's delivery metadata."""
from pathlib import Path
import json,csv,time,datetime,subprocess,hashlib,sys

ROOT=Path(__file__).resolve().parent.parent;H=Path(__file__).resolve().parent;BASE='539ead5ff52cb110a91736d6550cb988171d4d9b'
TRACE=ROOT/'paper/review-traces/experiment-result-to-claim/20261113_d6_run01'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    g=read(H/'QUALITY_GATE_256.json');g128=read(H/'QUALITY_GATE_128.json');audit=read(H/'INDEPENDENT_NUMERICAL_AUDIT.json');review=read(TRACE/'REVIEWER_VERDICT.json')
    assert audit['status']=='PASS' and audit['complete_scientific_grid'] and not g['passed']
    source=read(H/'SOURCE_BINDING.json')
    for n,v in source['source_sha256'].items():
        if n.startswith(H.name+'/'):assert digest(ROOT/n)==v,n
    old=read(H/'RESOURCE_SCIENTIFIC_FINAL_RAW.json');r=read(H/'RESOURCE_ACCOUNTING.json');assert r['counts']==old['counts']
    final='--final' in sys.argv;upper=120 if final else 0
    protocol=read(H/'PROTOCOL.json');now=time.time();wall=now-protocol['origin_unix_s']
    assert wall+upper<=7200,'Original D6 wall cap reached'
    r['wall']['total_s']=wall+upper
    r['closeout_accounting']=dict(measured_global_calendar_through_unix_s=now,measured_global_calendar_s=wall,
        final_metadata_tail_charged_upper_bound_s=upper,
        scope='All task preparation,diagnostics,science,audit,parallel independent review and publication included in one cumulative calendar wall; no reset on resume',
        origin_scope=protocol['origin_scope'],science_counters_unchanged=True)
    save(H/'RESOURCE_ACCOUNTING.json',r)
    if final:save(H/'FINAL_WALL_ACCOUNTING.json',dict(wall=r['wall'],accounting=r['closeout_accounting'],limit_s=7200))
    f=g['comparisons']['FrozenSL']['J']['mean'];s=g['comparisons']['SL']['J']['mean'];a=g['comparisons']['A']['J']['mean']
    result=f'''# D6结果到主张

独立架构/审稿角色判定：**claim_supported=no，confidence=high**，范围仅是本预定seed、已暴露SEL8+DEV3开发筛选。`pending_external_review=false`。没有paper/PAPER_CLAIM_AUDIT.json，论文写作裁决仍为provisional；不将实验审查包装为论文级审计。

主终点256的B父等权H-J收益：FrozenSL {100*f:.6f}%，匹配继续SL {100*s:.6f}%，A {100*a:.6f}%。两个主比较和各自SEL/DEV均未达到原门槛。相对退化A的小幅改善不足以通过；B对A的DEV分组还退化1.250991%。128也没有通过，未用其选择模型。

可支持：32768旧轨迹的D/raw/H成本分解成立；基线只依赖动作前历史，q只除初始任务数一次；A/B/SL同起点、同任务序列完成256更新，当前完整代理规划可执行；本B配方没有获得要求的完整路线收益。

不能支持：方差降低、裁剪/Adam更新无偏、机制普遍优越、独立父布局泛化、部署公平加速、实机机械臂IK或关节轨迹安全认证。baseline零均值推导针对裁剪前score估计器；实际A/B256步均触发裁剪，不能只凭梯度范数或系数分布解释唯一根因。

数值复核30692项PASS，独立角色另从792条原始评价记录重算。新A前256步的权重与损失严格复现旧D5对应前缀；实际使用的是新采集轨迹、新Adam及新的H请求，没有用旧D5轨迹更新新Actor。数学成立、梯度有限、动作/停靠变化均没有替代路线指标。

独立审查完成时主复核/模型索引仍在生成，原审查快照保留这一时间事实；随后主复核已PASS，见[INDEPENDENT_NUMERICAL_AUDIT.json](INDEPENDENT_NUMERICAL_AUDIT.json)。两套审查主收益一致，原始审查不回写。共享skill附录缺失按内联契约留痕，没有伪造外部审计。

路由：关闭本估计器配方，不追加训练或其他算法。原D5 NO_DEVELOPMENT_SIGNAL、P2—P5 NOT_RUN及固定32/A1/Critic等封存边界保持。不能据此否定所有因果基线或DRL。

[独立审查prompt/raw/parsed/routing](../paper/review-traces/experiment-result-to-claim/20261113_d6_run01/TRACE_MANIFEST.json)
'''
    (H/'RESULT_TO_CLAIM.md').write_text(result,encoding='utf-8')
    (H/'findings.md').write_text(f'''# D6 findings

- intended_claim: causal remaining-cost PG obtains registered complete-route development gains overFrozenSL/matchedSL.
- claim_supported: no; confidence: high within fixed256/exposed66/single-seed scope.
- H-J parent gains: FrozenSL {100*f:.9f}%, SL {100*s:.9f}%, A {100*a:.9f}%.
- support: exact cost decomposition and real A/B/SL256 implementation; actual route-quality gate failed.
- missing_evidence: variance reduction, retained generalization, fair timing and physical certification are not established; paper claim audit unavailable.
- writing: provisional; pending_external_review=false.
- routing: close_this_recipe; no new experiments, budgets, seeds or old branch restoration.
''',encoding='utf-8')
    (H/'NEXT_STEP_DECISION.md').write_text(f'''# D6结束决定

**256主终点NO_DEVELOPMENT_SIGNAL，结束本候选。** B相对FrozenSL、匹配SL的父等权H-J收益为{100*f:.6f}%、{100*s:.6f}%，SEL和DEV均退化；对A仅{100*a:.6f}%，不能替代两个主比较。128同样没有通过。

实际已完成A/B/SL各256、4096新在线轨迹、2048标签重放及792次当前评价，共4888请求，全部代理可行，失败0，Critic0。没有技术中断被解释为方法失败。

成本分解和裁剪前基线因果性成立，但本次完整路线质量没有要求的收益。新A与旧D5前256权重/损失严格一致，说明对照实现保留原损失；它不改变旧D5在4096主终点的失败结论。

研究内容三目标保持。当前证据不能唯一判定问题来自表示、估计器方差、H修正结构或受裁剪优化；不自动追加这些诊断。本轮不增加学习率、宽度、次数、种子，不恢复固定32/A1/Critic/R03/旧M3，不运行独立测试，不把当前权重提升为生产默认。完成本周期分析和GitHub交付后停止。
''',encoding='utf-8')
    (H/'TEACHER_DECISION_BRIEF.md').write_text(f'''# 供老师决策：D6信用分配候选结束

在固定D5表示/五张量、SL256起点、任务序列、Adam1e-4、裁剪1和完整H回报下，完成原损失A、因果剩余成本B及匹配继续SL各256更新。实际4096条新在线轨迹、792次同期评价均完成。

结论是：B相对FrozenSL、匹配SL的父等权H-J分别退化{100*(-f):.2f}%、{100*(-s):.2f}%，相对A改善{100*a:.2f}%；训练父新任务SEL8和已暴露非训练父DEV3对两个主对照均退化。该估计器没有通过原0.5%主门槛，结束本配方。

可靠基础仍是来源准确的SL初始化、可微原Actor评分、完整任务/mask/代理验证、固定H、真实在线采样和恢复。32768旧轨迹逐步成本重建成立，终端出口和H校正没有遗漏或重复。旧完整10×10占50%轨迹和84.83%绝对标量损失；全部旧批次裁剪，但这些统计不能证明唯一根因或新的梯度方差改善。

尚未实现的是DRL对完整任务的可靠增益，以及冻结后的质量/速度确认。单seed、66个已暴露任务只作机制筛选；原生规则final与神经raw/H明确分开，缓存差异下不宣称公平部署加速。没有新教师/场景/导航，没有读取封存集。

本轮不提出新算法、损失、网络、学习率或实验预算，不恢复已关闭支线。详见EXPERIMENT_REPORT.md、RESULT_TO_CLAIM.md及独立审查trace。
''',encoding='utf-8')
    report=(H/'EXPERIMENT_REPORT.md').read_text(encoding='utf-8-sig');marker='\n## 分尺寸、父布局与停靠变化补表\n'
    if marker not in report:
        tail=[marker,'收益仍先父内均值再父等权，正值才改善；以下不是新门槛。\n','|终点|分组|B对FrozenSL H-J|B对SL H-J|B对A H-J|\n|---:|---|---:|---:|---:|']
        from .evaluation import gains
        for point in (128,256):
            reps={k:read(H/'evaluations/development'/f'{k}_{point}.json')['rows'] for k in ('FrozenSL','A','B','SL')}
            for name in ('4x3','6x6','8x8','SEL8','DEV3'):
                def take(rows):
                    if name in ('SEL8','DEV3'):return [x for x in rows if x['group']==('trained_parent_new_tasks_exposed_selection' if name=='SEL8' else 'independent_parent_exposed_development')]
                    w,hh=map(int,name.split('x'));return [x for x in rows if (x['width'],x['height'])==(w,hh)]
                vals=[100*gains(take(reps['B']),take(reps[c]))['mean'] for c in ('FrozenSL','SL','A')]
                tail.append(f'|{point}|{name}|{vals[0]:.4f}%|{vals[1]:.4f}%|{vals[2]:.4f}%|')
        rows=list(csv.DictReader((H/'STOP_SET_ORDER_CHANGES.csv').open(encoding='utf-8-sig',newline='')))
        tail+=['\n|终点|B对照|raw集合/顺序改变数|H后集合/顺序改变数|\n|---:|---|---:|---:|']
        for p in (128,256):
            for c in ('FrozenSL','SL','A'):
                rr=[x for x in rows if int(x['endpoint'])==p and x['control']==c]
                sums=[sum(x[k]=='True' for x in rr) for k in ('raw_stop_set_changed','raw_visit_order_changed','H_stop_set_changed','H_visit_order_changed')]
                tail.append(f'|{p}|{c}|{sums[0]}/{sums[1]}|{sums[2]}/{sums[3]}|')
        tail+=['\n共9个实际step0/128/256 Actor导出。准确SL256文件SHA60f5aa4e59983e92d565f0599823b710965cf2c96a041e7eb483d23a8eb0556c，tensor SHA1817f58aa6506456ff12f0c5fe0a914e156aef8af319f4687dde90782f55a63e。实际Torch2.12.1+cu126、Python3.10.20、NumPy2.2.6，CPU单线程和deterministic，真实来源检查保存在各jobs/ENVIRONMENT.json。三组128→256均实际恢复完整检查点；没有工程模型参与初始化。\n',
            '独立数值30692项PASS；所有32768旧轨迹逐动作成本与负奖励最大误差4.55e-13，tail与原日志未来奖励和最大误差2.96e-12。新A前256的模型/损失严格复现原D5前缀，同时完成了新的在线采样及H请求，旧轨迹没有进入新optimizer。\n',
            '[结果到主张](RESULT_TO_CLAIM.md) · [给老师的短材料](TEACHER_DECISION_BRIEF.md) · [独立角色审查](../paper/review-traces/experiment-result-to-claim/20261113_d6_run01/REVIEWER_RAW.md) · [逐动作成本复核](OLD_PER_ACTION_COST_AUDIT.json) · [新在线系数诊断](ONLINE_CREDIT_DIAGNOSTICS.csv)\n']
        (H/'EXPERIMENT_REPORT.md').write_text(report+'\n'.join(tail),encoding='utf-8')
    p=H/'README.md';text=p.read_text(encoding='utf-8-sig')
    if '## 已完成的实际交付' not in text:
        text+='''\n## 已完成的实际交付\n\nA/B/匹配SL各256完成，4888请求/768 Actor/2048标签重放，失败0。主终点B对FrozenSL、SL的父等权H-J收益−1.2305%、−1.5834%，对A+0.7985%，NO_DEVELOPMENT_SIGNAL，结束本配方。\n\n[完整路线报告](EXPERIMENT_REPORT.md) · [数学与实现](MATHEMATICS.md) · [9个实际模型](MODEL_INDEX.json) · [792请求表](COMPLETE_TASK_RESULTS.csv) · [两聚合/分尺寸](AGGREGATE_RESULTS.csv) · [父配对](PARENT_PAIRED_GAINS.csv) · [停靠与顺序](STOP_SET_ORDER_CHANGES.csv) · [成本分解](COST_DECOMPOSITION_VALIDATION.json) · [逐动作复核](OLD_PER_ACTION_COST_AUDIT.json) · [旧规模损失/裁剪](D5_TRAJECTORY_CREDIT.csv) · [新在线参考诊断](ONLINE_CREDIT_DIAGNOSTICS.csv) · [数值复核](INDEPENDENT_NUMERICAL_AUDIT.json) · [研究决定](NEXT_STEP_DECISION.md) · [结果到主张](RESULT_TO_CLAIM.md) · [给老师的短材料](TEACHER_DECISION_BRIEF.md)。\n\n首算前协议/源码提交c0e3cd08897615a4137c9efeecc2c5860fd104fe。正式A/B/SL只读同一个COMMON_SCHEDULE，当前A前256的模型和损失与旧D5前缀精确一致，但实际进行了新的采样和更新。没有借用旧D5轨迹训练。当前候选不提升为生产默认。\n'''
    if (H/'PUBLICATION_RECEIPT.json').exists() and '## 远端发布核验' not in text:
        q=read(H/'PUBLICATION_RECEIPT.json');text+=f"\n## 远端发布核验\n\n实际payload提交{q['payload_commit']}，新增{q['actual_new_LFS_objects']}个LFS对象、{q['actual_new_LFS_bytes']}字节已从远端下载并逐一核对SHA256和大小。详见[发布收据](PUBLICATION_RECEIPT.json)、[远端LFS逐对象核对](LFS_REMOTE_READBACK.json)、[冻结文件保持](PRESERVATION_CHECK.json)与[最终墙钟结账](FINAL_WALL_ACCOUNTING.json)。最终元数据提交不改变已核对的模型和轨迹。\n"
    p.write_text(text,encoding='utf-8')
    block='''<!-- D6_CURRENT_BEGIN -->
## 当前：D6完整路线对照完成，因果剩余成本候选结束

[D6实际交付](rc3_d6_20261113/README.md) · [完整路线报告](rc3_d6_20261113/EXPERIMENT_REPORT.md) · [给老师的短材料](rc3_d6_20261113/TEACHER_DECISION_BRIEF.md)。32768旧轨迹成本分解PASS；A/B/匹配SL各256真实更新、4096新在线轨迹、792同期评价全部完成。B在256主终点对FrozenSL、SL的父等权H-J收益−1.2305%、−1.5834%，对A+0.7985%，SEL/DEV对两个主对照均负，按原门槛NO_DEVELOPMENT_SIGNAL结束。累计新4888请求、Actor768、重放2048、Critic0，失败0；9个实际模型与逐请求/资源/独立审查保存。旧D5负结果及封存边界保持，不合并main。
<!-- D6_CURRENT_END -->

'''
    old_readme=subprocess.check_output(['git','show',BASE+':README.md'],cwd=ROOT)
    (ROOT/'README.md').write_bytes(block.encode('utf-8').replace(b'\n',b'\r\n')+old_readme)
    specs=[H.name,'paper/review-traces/experiment-result-to-claim/20261113_d6_run01','rc3_r04_m1_recovery_20261111/run_claims/rc3-d6-causal-tail-20261113-r1-*']
    out=subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard','--',*specs],cwd=ROOT)
    files=sorted({x.decode('utf-8') for x in out.split(b'\0') if x})+['.gitattributes','README.md']
    p=ROOT/'project_management/GIT_MANIFEST.json';m=read(p)
    for name in files:
        f=ROOT/name
        if f.is_file():m['files'][name]=dict(sha256=digest(f),bytes=f.stat().st_size)
    m['file_count']=len(m['files']);m['total_bytes']=sum(x['bytes'] for x in m['files'].values());m['version']='2026.11.13-D6-complete-negative'
    m['D6']=dict(base_commit=BASE,branch='codex/rc3-d6-causal-tail-20261113',preregistration_commit='c0e3cd08897615a4137c9efeecc2c5860fd104fe',
        status='NO_DEVELOPMENT_SIGNAL',A_B_SL_updates=[256,256,256],requests=4888,Actor=768,Critic=0,teacher_replay=2048,
        old_D5_preserved=True,user_files_not_refreshed=True,main_merged=False,report=H.name+'/EXPERIMENT_REPORT.md',resource=H.name+'/RESOURCE_ACCOUNTING.json',publication='PENDING_REMOTE_VERIFICATION')
    if (H/'PUBLICATION_RECEIPT.json').exists():
        q=read(H/'PUBLICATION_RECEIPT.json');m['D6'].update(publication=q['status'],payload_commit=q['payload_commit'],actual_new_LFS_objects=q['actual_new_LFS_objects'],actual_new_LFS_bytes=q['actual_new_LFS_bytes'],receipt=H.name+'/PUBLICATION_RECEIPT.json')
    with p.open('w',encoding='utf-8',newline='') as f:f.write(json.dumps(m,ensure_ascii=False,indent=2).replace('\n','\r\n')+'\r\n')
    print(dict(status='D6_DELIVERY_METADATA_READY',final=final,scoped_files=len(files),charged_wall_s=r['wall']['total_s'],remaining_s=7200-r['wall']['total_s']))
if __name__=='__main__':main()
