"""Summarize CLOSED new-run evidence. Never calls a model, optimizer or planner."""
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

ROOT=Path(__file__).resolve().parents[2]
HOME=ROOT/'rc3_frozen_reference_pg_20261111'
ACTIVE=HOME/'execution_v2'
SCI=ACTIVE/'science'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def put(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8',newline='')))
def savecsv(p,data):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def mean(a):return math.fsum(a)/len(a)
def quantile(a,q):
    a=sorted(a);u=(len(a)-1)*q;i=int(u);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(u-i)
def main():
    started=time.perf_counter()
    resource=read(SCI/'RESOURCE_ACCOUNTING.json');decision=read(SCI/'FINAL_DECISION.json');audit=read(SCI/'INDEPENDENT_NEW_OUTPUT_AUDIT.json')
    assert resource['status']=='COMPLETE' and audit['status']=='PASS'
    request_rows=rows(SCI/'REQUEST_INDEX.csv');contrasts=rows(SCI/'QUALITY_CONTRASTS.csv')
    times={r['request_id']:r for r in rows(SCI/'REQUEST_TIMING.csv')}
    timing=[]
    for kind in ('training','evaluation'):
        for role in sorted({r['role'] for r in request_rows if r['kind']==kind}):
            selected=[r for r in request_rows if (r['role'],r['kind'])==(role,kind)]
            full=[float(times[r['request_id']]['full_request_s']) for r in selected]
            timing.append(dict(kind=kind,role=role,requests=len(selected),full_total_s=math.fsum(full),
                full_median_s=statistics.median(full),full_p95_s=quantile(full,.95),
                saved_input_prepare_total_s=math.fsum(float(r['input_prepare_s']) for r in selected),
                model_load_total_s=math.fsum(float(r['model_load_s']) for r in selected),
                added_feature_total_s=math.fsum(float(r['added_feature_s']) for r in selected),
                added_feature_calls=sum(int(r['added_feature_calls']) for r in selected),
                archive_IO_s=math.fsum(float(r['io_s']) for r in selected),
                index_journal_IO_s=math.fsum(float(times[r['request_id']]['index_and_journal_s']) for r in selected)))
    savecsv(HOME/'REQUEST_TIMING_SUMMARY.csv',timing)
    groups={};diagnostic_states=0
    with gzip.open(SCI/'ONLINE_UPDATE_DIAGNOSTICS.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            for window in ('ALL',r['window']):
                g=groups.setdefault((r['role'],window),dict(G=[],bg=[],bc=[],ag=[],ac=[],delta=[],delta_multi=[],gap=[],ratio=[],applied_ratio=[],KL=[],TV=[],fallback=0,states=0,flips=0))
                for s in r['states']:
                    g['states']+=1;g['fallback']+=int(s['unknown_fallback']);g['flips']+=int(s['base_top1']!=s['applied_top1'])
                    for out,key in [('delta','effective_delta_range'),('ratio','candidate_ratio'),('applied_ratio','applied_ratio'),('KL','candidate_step_KL'),('TV','candidate_base_TV')]:g[out].append(s[key])
                    if s['base_gap'] is not None:g['gap'].append(s['base_gap'])
                    if s['legal_count']>1:g['delta_multi'].append(s['effective_delta_range'])
                    if s['G'] is not None:
                        for out,key in [('G','G'),('bg','B_global'),('bc','B_cond'),('ag','centered_global'),('ac','centered_cond')]:g[out].append(s[key])
            diagnostic_states+=len(r['states'])
    diagnostics=[]
    for (role,window),g in sorted(groups.items()):
        row=dict(role=role,window=window,states=g['states'],unknown_fallback=g['fallback'],base_to_applied_top1_changes=g['flips'],
            effective_delta_range_median=statistics.median(g['delta']),effective_delta_range_p95=quantile(g['delta'],.95),
            effective_delta_range_max=max(g['delta']),multilegal_states=len(g['delta_multi']),
            multilegal_delta_range_median=statistics.median(g['delta_multi']) if g['delta_multi'] else None,
            multilegal_delta_range_p95=quantile(g['delta_multi'],.95) if g['delta_multi'] else None,
            candidate_ratio_min=min(g['ratio']),candidate_ratio_max=max(g['ratio']),applied_ratio_min=min(g['applied_ratio']),applied_ratio_max=max(g['applied_ratio']),
            candidate_KL_max=max(g['KL']),candidate_base_TV_max=max(g['TV']),
            base_gap_median=statistics.median(g['gap']) if g['gap'] else None,
            multilegal_base_gap_above_half_fraction=mean([float(x>.5) for x in g['gap']]) if g['gap'] else None)
        for name in ('G','bg','bc','ag','ac'):
            a=g[name];row[name+'_mean']=mean(a) if a else None
            row[name+'_std']=statistics.pstdev(a) if a else None
            row[name+'_negative_fraction']=mean([float(x<0) for x in a]) if a else None
        diagnostics.append(row)
    savecsv(HOME/'ONLINE_REFERENCE_SUMMARY.csv',diagnostics)
    events=[json.loads(s) for s in (SCI/'EVENTS.jsonl').read_text(encoding='utf-8').splitlines()]
    teacher=[e for e in events if e['event']=='teacher_replay_complete']
    candidates=[e for e in events if e['event']=='actor_candidate_complete']
    put(HOME/'ONLINE_SUMMARY.json',dict(online_diagnostic_states=diagnostic_states,
        teacher_replays=len(teacher),teacher_states=sum(e['states'] for e in teacher),teacher_replay_wall_s=math.fsum(e['wall_s'] for e in teacher),
        candidates={r:dict(opportunities=sum(e['role']==r for e in candidates),accepted=sum(e['role']==r and e['accepted'] for e in candidates),
            rejected=sum(e['role']==r and not e['accepted'] for e in candidates),rollbacks=sum(e['role']==r and e['rollback_exact'] for e in candidates)) for r in ('PG-global','PG-cond','continued_SL_A1')},
        interpretation='These training diagnostics do not establish complete-route benefit'))
    lines=['# 固定终点完整路线结果','', '仅单seed11622026、既有暴露开发案例。每终点66案例×6方法，父布局配对区间为描述性证据。', '',
        '## PG-cond主比较：父布局等权相对节省','', '| 终点 | 案例组 | 指标 | 对照 | 相对节省% | 父布局配对95%区间% |','|---:|---|---|---|---:|---:|']
    for r in contrasts:
        if r['method']=='PG-cond' and r['size']=='ALL' and r['cohort']!='ALL' and r['metric'] in ('J','raw_J') and r['control'] in ('FrozenSL','continued_SL_A1','PG-global'):
            lines.append(f"| {r['endpoint']} | {r['cohort']} | {r['metric']} | {r['control']} | {100*float(r['relative_saving']):.6f} | [{100*float(r['ci95_low']):.6f}, {100*float(r['ci95_high']):.6f}] |")
    lines+=['','## 六方法完整成本与请求时间','', '| 终点 | 方法 | H-J均值 | raw-J均值 | L均值 | K均值 | 完整请求中位秒 | p95秒 |','|---:|---|---:|---:|---:|---:|---:|---:|']
    for endpoint in (64,128):
        for role in ('FrozenSL','continued_SL_A1','PG-global','PG-cond','Fixed3','Detour16'):
            rr=[r for r in request_rows if r['kind']=='evaluation' and r['opportunity']==str(endpoint) and r['role']==role]
            tt=[float(times[r['request_id']]['full_request_s']) for r in rr]
            vals=[mean([float(r[k]) for r in rr]) for k in ('J','raw_J','L','K')]
            lines.append(f"| {endpoint} | {role} | "+' | '.join(f'{v:.6f}' for v in vals)+f' | {statistics.median(tt):.6f} | {quantile(tt,.95):.6f} |')
    lines+=['', '本表成本均值是66案例均值；主推断采用上表的父布局分组。完整各尺寸、J/raw-J/L/K及完整请求时间对照见[QUALITY_CONTRASTS.csv](execution_v2/science/QUALITY_CONTRASTS.csv)。', '',
        '原生Fixed3/Detour16的raw栏沿用其最终原生搜索/H路线，不代表去掉原生内部H的额外实验；没有追加共同H。显式共享postprocess前后路线、原生最终路线和内部搜索调用均保留；原生全部H中间路线未单独捕获，见[记录边界](SCHEMA_AND_ACCOUNTING_NOTES.md)。', '',
        '完整请求计时包括输入/来源/已保存导航恢复、冷模型加载、provider/新增特征、策略或native、H、完整导出/验证、归档和索引/完成journal。计时收据自身写入计入阶段墙钟；文件系统缓存未控制，验证字节缓存真实记录。']
    (HOME/'ENDPOINT_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    put(HOME/'POSTPROCESS_ACCOUNTING.json',dict(wall_s=time.perf_counter()-started,scope='saved new request index/timing/diagnostics/events summary only; no model/planner/optimizer',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps(dict(status=decision['status'],requests=len(request_rows),audit=audit['status'],postprocess_s=time.perf_counter()-started)))
if __name__=='__main__':main()
