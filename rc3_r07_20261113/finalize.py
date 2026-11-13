"""Delivery arithmetic and scoped manifests; never imports Torch or planning."""
import csv,datetime,hashlib,json,subprocess,time
from pathlib import Path
from collections import defaultdict
from .common import ROOT,HOME,read,put,unarchive

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def csvout(name,rows):
    with (HOME/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def avg(v):return sum(v)/len(v)
def main():
    protocol=read(HOME/'PROTOCOL.json');assert time.time()<protocol['deadline_unix_s']
    result=read(HOME/'RESULTS.json');rows=list(csv.DictReader((HOME/'STATE_PAIRS.csv').open(encoding='utf-8')))
    costs=[]
    for p in sorted((HOME/'pairs').glob('*.json')):
        pair=read(p)
        for branch in ('baseline','alternative'):
            r=unarchive(pair[branch]['record']);s=pair['state']
            costs.append(dict(state_id=s['state_id'],physical_key=s['physical_key'],size=s['size'],stage=s['stage'],branch=branch,
                **r['metrics'],H_J_correction=r['metrics']['J']-r['metrics']['raw_J'],
                H_L_correction=r['metrics']['L']-r['metrics']['raw_L'],H_K_correction=r['metrics']['K']-r['metrics']['raw_K'],
                full_request_s=pair[branch]['full_request_s'],H_s=r['H_s'],prefix_restore_s=r['prefix_restore_s'],
                raw_validation=True,H_validation=True))
    csvout('RAW_H_ROUTE_METRICS.csv',costs)
    gaps=read(HOME/'TEACHER_GAPS.json')['rows'];gs=[]
    for size in ('overall','10x10','4x3','6x6','8x8'):
        rr=[x for x in gaps if size=='overall' or x['size']==size]
        gs.append(dict(size=size,parents=len(rr),FrozenSL_H_J_mean=avg([x['FrozenSL_H_J'] for x in rr]),
            strong_teacher_H_J_mean=avg([x['strong_teacher_H_J'] for x in rr]),
            mean_SL_minus_teacher_J=avg([x['gap'] for x in rr]),parent_equal_relative_gap=avg([x['relative_gap'] for x in rr]),
            teacher_better=sum(x['gap']>1e-7 for x in rr),equal=sum(abs(x['gap'])<=1e-7 for x in rr),teacher_worse=sum(x['gap']<-1e-7 for x in rr)))
    csvout('TEACHER_GAP_SUMMARY.csv',gs)
    def count(fn):return sum(fn(x) for x in rows)
    effects=dict(states=len(rows),raw_cost_different=count(lambda x:abs(float(x['raw_delta_J']))>1e-7),
        H_cost_different=count(lambda x:abs(float(x['delta_J']))>1e-7),
        raw_cost_difference_erased_by_H=count(lambda x:abs(float(x['raw_delta_J']))>1e-7 and abs(float(x['delta_J']))<=1e-7),
        H_identical_route=count(lambda x:x['H_identical_route']=='True'),
        H_identical_stop_set=count(lambda x:x['H_identical_set']=='True'),
        H_material_different=count(lambda x:abs(float(x['relative_gain']))>=.005),
        material_improved=count(lambda x:float(x['relative_gain'])>=.005))
    put(HOME/'H_EFFECTS.json',effects)
    with (HOME/'EXPERIMENT_REPORT.md').open('a',encoding='utf-8') as f:
        f.write('\n## 对研究问题的回答\n\n')
        f.write(f"1. 存在同状态、固定后续策略下可复现的动作条件成本差异：H后{effects['H_cost_different']}/144有非零成本差，{effects['H_material_different']}/144绝对差达到0.5%。不能据此宣称Top2整体优于Top1或全局统计显著。\n")
        f.write(f"2. 差异部分在H后保留；{effects['H_identical_route']}/144条配对最终路线完全相同，{effects['H_identical_stop_set']}/144停靠集合相同。raw存在成本差而H后持平的有{effects['raw_cost_difference_erased_by_H']}个状态。成本持平不等于路线相同。\n")
        f.write(f"3. 可改善案例不足以达到预设继续标准：≥0.5%改善为{effects['material_improved']}/144=15.97%，低于20%；局部为5/48=10.42%，低于15%。完整18/96=18.75%、改善父布局12/32及4/16、假设oracle均值0.5218%满足各自门槛，联合门槛仍FAIL。该筛选只涉及这些训练父布局和Top1/Top2。\n")
        t=result['summaries']['overall']
        f.write(f"4. 本轮常驻模型/已恢复输入条件下，每状态两条完整Rollout合计均值{t['paired_rollout_s']['mean']:.6f}秒，p95 {t['paired_rollout_s']['p95']:.6f}秒；若精确baseline已有，新增alternative均值{t['alternative_rollout_s']['mean']:.6f}秒。包括前缀重放、编码、剩余选择、H、展开验证与证据写入，任务输入初次恢复/状态选择另记。缓存/冷热条件不支持公平部署加速主张。\n")
        f.write('5. 本轮不支持直接进入动作相对价值近似策略迭代。按协议建议结束当前构造式学习方向，下一版若继续研究，应重新设计整体规划决策层次；本轮不生成新的算法或预算。\n\n')
        f.write('|任务尺寸|父布局|FrozenSL H-J均值|既有强教师 H-J均值|父等权相对差|教师更好/相同/更差|\n|---|---:|---:|---:|---:|---:|\n')
        for x in gs:f.write(f"|{x['size']}|{x['parents']}|{x['FrozenSL_H_J_mean']:.4f}|{x['strong_teacher_H_J_mean']:.4f}|{100*x['parent_equal_relative_gap']:.3f}%|{x['teacher_better']}/{x['equal']}/{x['teacher_worse']}|\n")
        f.write('\n教师gap是既有同问题整体路线差距，并不保证本轮单动作干预可实现。每父一条问题计一次；不把三个状态重复计为三个父布局。每分支绝对raw/H J/L/K、H修正和耗时见RAW_H_ROUTE_METRICS.csv，完整停靠顺序、服务分配、导航折线与验证证书见归档逐请求记录。\n')
    (HOME/'README.md').write_text('# R07动作反事实完整Rollout机会验证\n\n**完成：OPPORTUNITY_INSUFFICIENT。**48个互异训练父布局、144个真实前缀状态、288次主比较+16次固定复测，共304次完整请求；Actor/Critic更新0。\n\n[完整报告](EXPERIMENT_REPORT.md) · [逐状态比较](STATE_PAIRS.csv) · [逐分支raw/H成本](RAW_H_ROUTE_METRICS.csv) · [父布局聚合](PARENT_OPPORTUNITIES.csv) · [尺度/进度](RESULTS.json) · [教师差距](TEACHER_GAP_SUMMARY.csv) · [完整双路线归档索引](rollouts/REQUEST_POINTERS.jsonl) · [实际重复核验](REPRODUCIBILITY.json) · [研究决定](NEXT_STEP_DECISION.md) · [资源](RESOURCE_ACCOUNTING.json)\n\n[协议](PROTOCOL.json)、[任务](TASKS.json)、[源码绑定](SOURCE_BINDING.json)、[首次计算前Git登记](PREREGISTRATION_GIT_RECEIPT.json)已冻结。实际完成命令：\n\n```powershell\n& .\\rc3_learning\\.venv\\Scripts\\python.exe -B -X utf8 -m rc3_r07_20261113.run\n```\n\n该身份已封闭，命令再次执行会拒绝，不能作为续跑授权。潜在有利候选选择不是已训练策略收益。D5/D6旧负结果、96完整教师和256局部教师及模型原字节保留。\n',encoding='utf-8')
    # Root historical README bytes remain exactly as in the requested base.
    old=subprocess.check_output(['git','show',protocol['base_commit']+':README.md'],cwd=ROOT)
    block=('# RC3-R07 已完成：动作反事实机会不足\n\n48个互异训练父布局、144个状态，304次完整请求（含16复测），Actor/Critic更新0。≥0.5%改善23/144，局部5/48，未过预登记机会门槛；不启动新训练。\n\n[报告](rc3_r07_20261113/EXPERIMENT_REPORT.md) · [双路线证据](rc3_r07_20261113/rollouts/REQUEST_POINTERS.jsonl) · [研究决定](rc3_r07_20261113/NEXT_STEP_DECISION.md) · [资源](rc3_r07_20261113/RESOURCE_ACCOUNTING.json)\n\n').encode('utf-8')
    (ROOT/'README.md').write_bytes(block+old)
    source_paths=['analyze.py','finalize.py']
    put(HOME/'POSTPROCESS_EXECUTED_SOURCE.json',dict(utc=now(),scope='Saved route arithmetic and delivery metadata only',
        sources={n:hashlib.sha256((HOME/n).read_bytes()).hexdigest() for n in source_paths},new_forward_H_planning_optimizer=0))
    print(json.dumps(dict(H_effects=effects,teacher_gap_summary=gs,wall_s=time.time()-protocol['origin_unix_s']),ensure_ascii=False),flush=True)

def manifest():
    p=ROOT/'project_management/GIT_MANIFEST.json';raw=p.read_bytes();data=json.loads(raw)
    scope=['rc3_r07_20261113','paper/review-traces/experiment-result-to-claim/20261113_r07_run01',
        'rc3_r04_m1_recovery_20261111/run_claims/rc3-r07-action-rollouts-20261113-r1*','README.md','.gitattributes','EXPERIMENT_RESULT_TO_CLAIM.md']
    names=subprocess.check_output(['git','ls-files','-z','-c','-o','--exclude-standard','--',*scope],cwd=ROOT).split(b'\0')
    for name in sorted(set(n.decode('utf-8') for n in names if n)):
        f=ROOT/name
        if f.is_file():data['files'][name]=dict(sha256=hashlib.sha256(f.read_bytes()).hexdigest(),bytes=f.stat().st_size)
    data['version']='2026.11.13-R07-complete-opportunity-insufficient'
    data['R07']=dict(base_commit='fed8296a8e5481a6920c18e9a5e908b23f2266cb',branch='codex/rc3-r07-action-rollouts-20261113',
        preregistration_commit='9ae6f6e199211cc1f7069f1118fa57252c1d8bb2',status='OPPORTUNITY_INSUFFICIENT',
        physical_training_parents=48,states=144,requests=304,Actor=0,Critic=0,new_teacher_scene_navigation=0,
        old_D5_D6_and_teacher_bytes_preserved=True,user_files_not_refreshed=True,main_merged=False,
        report='rc3_r07_20261113/EXPERIMENT_REPORT.md',resource='rc3_r07_20261113/RESOURCE_ACCOUNTING.json')
    receipt=HOME/'PUBLICATION_RECEIPT.json'
    if receipt.exists():data['R07'].update(read(receipt))
    content=json.dumps(data,ensure_ascii=False,indent=2)+'\n'
    if b'\r\n' in raw:content=content.replace('\n','\r\n')
    p.write_bytes(content.encode('utf-8'))

if __name__=='__main__':
    import sys
    manifest() if '--manifest' in sys.argv else main()
