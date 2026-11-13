"""Render completed B results and close this fixed neighborhood."""
import csv,time
from .common import HOME,ROOT,read,put,sha,state

def main():
    result=read(HOME/'RESULTS.json');g=result['groups'];parents=list(csv.DictReader((HOME/'TASK_PARENT_RESULTS.csv').open(encoding='utf-8')))
    rows=list(csv.DictReader((HOME/'ACTION_RESULTS.csv').open(encoding='utf-8')));curves=list(csv.DictReader((HOME/'PREFIX_SUMMARY.csv').open(encoding='utf-8')))
    lines=['# R08-B联合配置编辑机会：实际完整处理结果\n',
        '**'+result['status']+'**。按R08-A登记原门槛结束本固定邻域；不执行C、D、E。所有48原R07训练任务完成（32完整、局部6/5/5），48个不同物理父。没有选择更容易案例、改变菜单或追加候选。\n',
        '## 实际处理与固定菜单\n',
        f"48次准确FrozenSL raw/H锚点实时重建与保存来源逐路线/分量一致；671次菜单编辑，共719完整请求，全部成功，失败0。正常最多816，本轮去重/合法菜单不足留下97空槽，未补。Actor/Critic更新0，新教师/场景/导航/native0，原H实际719次。Frozen Actor重新编码48次、评分3984次；这不是零前向实验，也没有新菜单神经头。\n",
        '每个编辑从同一FrozenSL raw起点独立执行，非串联累积；按已登记最多8块/8替代池/每块前2，再全局净编辑去重，不填重复。48个真实菜单与完整块/池/筛选次序见menus/。显式drop/add交集先相消，再计单/双配置。下层按ID插入、8pass原2opt及正增益normalize，随后原H与独立完整验证。\n',
        '## 原门槛逐项结果\n',
        '|条件|实际|门槛|结果|\n|---|---:|---:|---|',
        f"|完整父有≥1%改善|{g['10x10']['material_tasks']}/32|≥8|PASS|",
        f"|局部父有≥1%改善|{g['local']['material_tasks']}/16|≥4|PASS|",
        f"|任务改善比例|{100*g['ALL']['material_task_fraction']:.4f}%|≥30%|PASS|",
        f"|完整/局部改善比例|{100*g['10x10']['material_task_fraction']:.4f}% / {100*g['local']['material_task_fraction']:.4f}%|各≥25%|PASS|",
        f"|逐任务最有利编辑的父等权假设收益|{100*g['ALL']['oracle_parent_mean']:.6f}%|≥1.5%|FAIL|",
        f"|实质改善编辑中的净双配置比例|{g['ALL']['material_joint_edits']}/{g['ALL']['material_edits']}={100*result['material_net_joint_fraction']:.6f}%|≥50%|FAIL|",
        f"|D有界吞吐预测|{result['forecast']['estimated_D_wall_s']:.3f}s|≤21600s|PASS（预测）|",
        '|实际raw/H完整代理约束|全719请求有效|全部有效|PASS|',
        '\n联合AND失败，两项科学机会条件未通过；不是未跑完造成的技术FAIL，也不能因普遍性或时间一项通过而改判。\n',
        '## 分规模与动作结果\n',
        '|规模|任务|动作|≥1%改善任务|父等权假设最佳收益|改善/平/退化动作|净双配置实质改善/全部实质改善|编辑完整时间均值/p95秒|\n|---|---:|---:|---:|---:|---:|---:|---:|']
    for key in ('ALL','10x10','local','4x3','6x6','8x8'):
        v=g[key];t=v['edit_times'];lines.append(f"|{key}|{v['tasks']}|{v['actions']}|{v['material_tasks']}|{100*v['oracle_parent_mean']:.6f}%|{v['improved']}/{v['tied']}/{v['degraded']}|{v['material_joint_edits']}/{v['material_edits']}|{t['mean']:.6f}/{t['p95']:.6f}|")
    lines += ['\n671个显式净动作中，115为双配置，556退化为单配置；74个≥1%改善编辑中，仅11为净双配置（覆盖7个父）。raw最终集合出现至少2个配置差异的617个动作，H后相对锚点出现至少2个配置差异的283个动作，**不能**据此重标为双配置动作：normalize/H也会自动删除/替换其他配置。例如单配置动作可以在清理时删除数十个冗余配置，仍须按实际显式动作归类。\n',
        '本轮只能区分实际执行的显式单/双编辑，不能证明双配置交互的独立因果收益。每个候选还共同得到8pass排序及原H，未增加新的单配置归因或no-edit排序控制。不得把整套下层/H收益全部归于上层联合动作。\n',
        '## H是否抹平差异\n',
        'H后335/671成本持平（49.93%），其中332个raw与锚点成本不同但H后持平；208/671完整H路线相同，240/671配置集合相同。成本持平不等于路线相同。所有配置/任务/臂变化、显式动作存活、raw/H J/L/K和前后导航证据见ACTION_RESULTS.csv及每动作归档。H可改写完整原路线，不是已物理执行前缀的后缀优化。\n',
        '## 固定菜单前缀：查询机会与真实累计时间\n',
        '|实际菜单前k上限|实际动作合计|≥1%改善任务比例|父等权假设最佳收益|每任务累计编辑时间均值|含锚点+菜单均值秒|\n|---|---:|---:|---:|---:|---:|']
    for x in curves:
        if x['size']=='ALL':lines.append(f"|{x['prefix_budget']}|{x['actual_actions']}|{100*float(x['material_fraction']):.4f}%|{100*float(x['oracle_parent_mean']):.6f}%|{float(x['mean_edit_s']):.6f}|{float(x['mean_total_s']):.6f}|")
    lines += ['\n不足k只使用已有min(k,M)动作，不虚构第k个请求。先按登记次序执行，再统计有利选择；这是假设可看到候选真实成本的oracle，不是学成策略收益，也不是沿一条路径可累加的多步收益。每任务只有一个物理父，所有任务/尺度/父均分开报告；不把671动作当独立父布局。\n',
        '## 128步后续训练的成本边界\n']
    f=result['forecast'];lines += [f"实际编辑完整时间均值{g['ALL']['edit_times']['mean']:.6f}s，p95 {g['ALL']['edit_times']['p95']:.6f}s。按max(全部请求均值,编辑p95)×22428 +14656次菜单×max(菜单均值,p95) +1920梯度/存盘 +3000其他预留 +{f['source_model_misc_setup_s']:.3f}s已测源/模型/日志杂项，预测{f['estimated_D_wall_s']:.3f}s，低于21600s。没有把B一份菜单16候选的摊销错当成训练只选一个动作的特征开销。\n",
        '该项仅通过B的工程预算预测；400维新头、四步状态更新、梯度/恢复/部署均未执行，不能称已验证128训练吞吐。质量机会已失败，时间有余不构成继续许可。计时是本轮常驻模型、精确恢复输入条件，完整请求含下层/H/导出验证与压缩归档，指针/账本尾部和模型源恢复另在总作业账；不与旧R07冷/热耗时宣称公平加速。\n',
        'H内部2873次two_opt、719次joint_search由原监督器记录；新增下层每成功编辑另有1次8pass two_opt，共671次，计总3544次two_opt。下层总时间已测，但其中two_opt未单独拆时，不以H计时冒充其时间。内部inclusive/exclusive按原口径保留。\n',
        '## 研究决定\n',
        '关闭这个固定2配置/8块/8池/16动作邻域。可改善任务存在，但总体假设潜力不足且改善主要来自退化的单配置动作，未满足原联合机会要求。C不具科学准入，且用户未授权；不启动C/D/E、不加配置数/候选/排序轮次/训练，不回到D5/D6/A1/Critic/32诊断。结论只限本登记邻域，不证明所有联合规划学习不可能。\n',
        '完整代理验证不构成机械臂IK、关节轨迹或实机认证。原60/旧S3/确认最终任务无读取，历史资源和用户文件保持。独立审查与结果到主张见RESULT_TO_CLAIM.md。\n']
    (HOME/'EXPERIMENT_REPORT.md').write_text('\n'.join(lines).rstrip()+'\n',encoding='utf-8')
    (HOME/'NEXT_STEP_DECISION.md').write_text('# R08-B结束决定\n\n**OPPORTUNITY_INSUFFICIENT：关闭本固定联合邻域。**\n\n19/48任务有≥1%改善，但父等权假设最佳收益1.195478%<1.5%，实质改善动作净双配置比例11/74=14.864865%<50%；原联合门槛未过。719请求全完成且有效、失败0，非技术中断。\n\n吞吐预测12660.580s可容纳原D21600s预算，不抵消质量失败；预测不是实际训练吞吐。Actor/Critic0；C/D/E均NOT_RUN_NOT_AUTHORIZED，C质量不准入。未执行项为null，不填零实验结果。\n\n不追加菜单/配置数/求解轮次/训练，旧D5/D6/R07/固定32等停止保持；不自动提出或启动另一算法。若将来重新设计需独立新任务和事前协议，本次许可到B交付结束。\n',encoding='utf-8')
    (HOME/'README.md').write_text('# R08-B：固定联合配置编辑机会\n\n**已完成：OPPORTUNITY_INSUFFICIENT。关闭本邻域，不进C。**\n\n48个互异训练父任务、48次准确SL锚点+671编辑=719完整处理，失败0。两项机会门槛失败：oracle1.195478%<1.5%、净双配置实质改善11/74<50%。新Actor/Critic更新、教师/场景/导航/native均0，H719。\n\n[完整报告](EXPERIMENT_REPORT.md) · [逐动作](ACTION_RESULTS.csv) · [物理父/任务](TASK_PARENT_RESULTS.csv) · [前缀机会](PREFIX_SUMMARY.csv) · [菜单](menus/) · [前后完整方案归档](rollouts/REQUEST_POINTERS.jsonl) · [原门槛](PROTOCOL.json) · [资源](RESOURCE_ACCOUNTING.json) · [决定](NEXT_STEP_DECISION.md)\n\n已执行命令（身份已封闭，不允许重新执行重置账）：\n\n```powershell\n& .\\rc3_learning\\.venv\\Scripts\\python.exe -B -X utf8 -m rc3_r08_joint_runtime.run\n```\n\n来源/源码在首次实际请求前Git登记，PREREGISTRATION_GIT_RECEIPT.json绑定。只恢复准确SL256，未创建新菜单神经头；A旧文档和全部旧证据原字节保持。完整请求与归档各记录真实raw/H成本/配置顺序/任务臂分配/导航折线/验证；oracle不是已训练策略收益。\n',encoding='utf-8')
    r=read(HOME/'RESOURCE_ACCOUNTING.json');put(HOME/'RESOURCE_SCIENTIFIC_FINAL_RAW.json',r)
    r.update(derived_lower_two_opt_calls=len(rows),total_two_opt_calls=r['counts']['internal_two_opt_counts']+len(rows),
        lower_two_opt_timing='Not separately timed; included in per-edit lower_s and full request, never silently omitted from request count',
        internal_two_opt_count_scope='Instrumented original H only; lower.py imported callable counted separately by one actual call per successful edit; failures0',
        actual_normal_requests=719,unused_normal_slots=97,hard_request_cap=900,unused_hard_request_capacity=181,
        stage='B_ONLY',C_D_E='NOT_RUN_NOT_AUTHORIZED',new_menu_neural_heads=0,scientific_result=result['status'],
        supervised_worker_wall_s=read(HOME/'controls/SUPERVISOR_RECEIPT.json')['elapsed_s'],
        total_wall_origin_utc='2026-11-13T07:16:30+00:00',total_deadline_utc='2026-11-13T07:46:30+00:00',
        prior_ledger_hashes={x:sha(ROOT/x) for x in ['rc3_r07_20261113/RESOURCE_ACCOUNTING.json','rc3_r08_joint_decision_design_20261113/RESOURCE_ACCOUNTING.json']},
        wall_s=time.time()-read(HOME/'PROTOCOL.json')['origin_unix_s'])
    put(HOME/'RESOURCE_ACCOUNTING.json',r,True);state(status=result['status'],closed=True,scientific_quality_gate='FAIL_TWO_ORIGINAL_CONDITIONS',C_D_E='NOT_RUN_NOT_AUTHORIZED')
    print(dict(status=result['status'],report='EXPERIMENT_REPORT.md',wall_s=r['wall_s']),flush=True)

if __name__=='__main__':main()
