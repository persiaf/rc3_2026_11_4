"""Tabulate ONLY existing task-evaluation records; stdlib, no model/planner import."""
import csv
import hashlib
import json
from collections import defaultdict,Counter
from pathlib import Path
from statistics import fmean
import time

ROOT=Path(__file__).resolve().parents[1]
HOME=Path(__file__).resolve().parent
INPUTS={}
DEV='independent_parent_exposed_development'
SEL='trained_parent_new_tasks_exposed_selection'
R03='rc3_conservative_refinement/integration_20261110/runs/conservative_paired_11622026_v1'
PG='rc3_frozen_reference_pg_20261111/execution_v2/science'

def payload(name):
    p=ROOT/name;b=p.read_bytes();INPUTS[name]=dict(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b));return b
def rows(name):return list(csv.DictReader(payload(name).decode('utf-8-sig').splitlines()))
def read(name):return json.loads(payload(name).decode('utf-8-sig'))
def put(name,x):(HOME/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
def write(name,data):
    with (HOME/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def quantile(v,p):
    x=sorted(v);i=(len(x)-1)*p;k=int(i);return x[k]+(x[min(k+1,len(x)-1)]-x[k])*(i-k)
def exposure(cohort):
    return '训练父新任务；选择/筛选已暴露' if cohort==SEL else '训练父外DEV；已反复暴露，不是新独立确认'

def make_group(experiment,context,endpoint,method,model_endpoint,cohort,data,times,quality_source,time_source,clock):
    parents=defaultdict(list)
    for r in data:parents[r['parent']].append(r)
    assert len(data)==len(times) and len(parents)==(8 if cohort==SEL else 3)
    assert all(r['valid']=='True' for r in data)
    native=method in ('Fixed3','Detour16')
    family='FrozenSL' if method=='FrozenSL' else ('匹配监督' if method in ('residual_SL','continued_SL','continued_SL_A1') else ('原生基线' if native else '实际DRL'))
    out=dict(experiment=experiment,comparison_context=context,comparison_endpoint=endpoint,method=method,family=family,
        model_endpoint=model_endpoint,cohort=cohort,exposure=exposure(cohort),size='ALL',physical_parents=len(parents),cases=len(data),
        quality_aggregation='within-parent case mean, then equally weighted parent mean',
        raw_semantics='NATIVE_FINAL_NOT_PRE_H' if native else 'ACTOR_COMPLETE_BEFORE_COMMON_H',
        H_semantics='NATIVE_FINAL_NO_EXTRA_COMMON_H' if native else 'SAME_FIXED_COMMON_H_WITHIN_EXPERIMENT')
    for m in ['raw_J','raw_L','raw_K','J','L','K']:
        out[('H_'+m) if m in ('J','L','K') else m]=fmean(fmean(float(r[m]) for r in group) for group in parents.values())
    out.update(full_recorded_time_p50_s=quantile(times,.5),full_recorded_time_p95_s=quantile(times,.95),
        time_aggregation='request p50/p95 within SAME experiment/context/endpoint/cohort; not a mean of medians',
        timing_scope=clock,quality_source=quality_source,time_source=time_source,seed=11622026,
        independent_generalization=False,physical_joint_trajectory_certified=False)
    return out

def main():
    start=time.perf_counter();table=[];case_export=[];contrasts=[]
    # Earlier actual PPO: only the originally selected five methods on the
    # already exposed48 selection cases, not a newly selected checkpoint.
    path='rc3_integrated_rppo/ALL_ROUTE_RESULTS.csv';data=rows(path)
    specifications=[('SL256','256','FrozenSL'),('residual_SL','256','residual_SL'),('RL','128','RL'),('Fixed3','','Fixed3'),('Detour16','','Detour16')]
    for role,ep,method in specifications:
        chosen=[dict(r,parent=r['physical_parent']) for r in data if r['scope']=='selection' and r['role']==role and r['update']==ep]
        assert len(chosen)==48
        out=make_group('INTEGRATED_RPPO_SMALL256','original_selected_endpoints','RL128_vs_SL256',method,ep or 'NATIVE_FIXED',SEL,chosen,
            [float(r['full_audit_s']) for r in chosen],path,path,
            'saved development full_audit_s: preparation/load/features/route/H/export-validation/prescribed IO; NOT final paired cold/hot deployment timing')
        table.append(out)
    # Earlier conservative PPO ON/OFF. Use its saved through-record-IO clock;
    # never substitute its timings for later A1 PG.
    tr=f'{R03}/analysis/ALL_REQUEST_TIMING.csv';timing=[r for r in rows(tr) if r['kind']=='eval']
    assert len(timing)==792
    for ep in [64,128]:
        path=f'{R03}/evaluation_{ep:04d}/ALL_ROUTE_RESULTS.csv';data=rows(path)
        assert len(data)==396 and all(r['repeat']=='0' for r in data)
        for cohort in [DEV,SEL]:
            for method in ['FrozenSL','continued_SL','ON','OFF','Fixed3','Detour16']:
                chosen=[dict(r,parent=r['physical_key']) for r in data if r['cohort']==cohort and r['method']==method]
                ts=[r for r in timing if int(r['endpoint_or_round'])==ep and r['cohort']==cohort and r['method']==method]
                assert Counter((r['parent'],r['width'],r['height']) for r in chosen)==Counter((r['physical_key'],r['width'],r['height']) for r in ts)
                table.append(make_group('CONSERVATIVE_PPO_R03','fixed_saved_endpoint',ep,method,'SL256' if method=='FrozenSL' else ('NATIVE_FIXED' if method in ('Fixed3','Detour16') else ep),cohort,chosen,[float(r['through_record_IO_s']) for r in ts],path,tr,
                    'saved through_record_IO_s; input/model/features/route/H/export-validation plus saved record IO; caches uncontrolled; NOT later PG clock'))
    # Latest frozen-reference PG, only792 evaluation requests. Training1536
    # records remain excluded from all quality/time aggregates.
    path=f'{PG}/REQUEST_INDEX.csv';data=[r for r in rows(path) if r['kind']=='evaluation'];assert len(data)==792
    tp=f'{PG}/REQUEST_TIMING.csv';all_times={r['request_id']:r for r in rows(tp)}
    for ep in [64,128]:
        for cohort in [DEV,SEL]:
            for method in ['FrozenSL','continued_SL_A1','PG-global','PG-cond','Fixed3','Detour16']:
                chosen=[dict(r,parent=r['physical_key']) for r in data if int(r['opportunity'])==ep and r['cohort']==cohort and r['role']==method]
                table.append(make_group('FROZEN_REFERENCE_PG_A1','fixed_saved_endpoint',ep,method,'SL256' if method=='FrozenSL' else ('NATIVE_FIXED' if method in ('Fixed3','Detour16') else ep),cohort,chosen,
                    [float(all_times[r['request_id']]['full_request_s']) for r in chosen],path,tp,
                    'saved full_request_s through archive/index/completion journal; input/restored navigation/source/cold model/features/route/H/export-validation; not fresh Python process; FS cache uncontrolled'))
                for r in chosen:
                    case_export.append(dict(experiment='FROZEN_REFERENCE_PG_A1',endpoint=ep,cohort=cohort,method=method,case_id=r['case_id'],physical_key=r['parent'],size=r['width']+'x'+r['height'],
                        **{m:r[m] for m in ['raw_J','raw_L','raw_K','J','L','K','valid']},full_request_s=all_times[r['request_id']]['full_request_s'],source_request_id=r['request_id']))
    # Reuse saved descriptive parent bootstrap outputs. No bootstrap is rerun.
    qp=f'{PG}/QUALITY_CONTRASTS.csv'
    for r in rows(qp):
        if r['cohort'] not in (DEV,SEL) or r['size']!='ALL' or r['metric'] not in ('J','raw_J'):continue
        contrasts.append(dict(experiment='FROZEN_REFERENCE_PG_A1',endpoint=r['endpoint'],cohort=r['cohort'],method=r['method'],control=r['control'],metric=r['metric'],physical_parents=r['physical_parents'],relative_saving=r['relative_saving'],ci95_low=r['ci95_low'],ci95_high=r['ci95_high'],source=qp,inference='SAVED descriptive parent-paired interval; exposed single-seed development; no new bootstrap'))
    qp=f'{R03}/analysis/QUALITY_CONTRASTS.json'
    for r in read(qp):
        if r['cohort'] not in (DEV,SEL) or r['size']!='all' or r['metric'] not in ('J','raw_J'):continue
        contrasts.append(dict(experiment='CONSERVATIVE_PPO_R03',endpoint=r['endpoint'],cohort=r['cohort'],method=r['method'],control=r['control'],metric=r['metric'],physical_parents=r['physical_parents'],relative_saving=r['point'],ci95_low=r['ci95'][0],ci95_high=r['ci95'][1],source=qp,inference='SAVED descriptive parent-paired interval; exposed single-seed development; no new bootstrap'))
    assert len(table)==53
    write('TASK_LEVEL_COMPARISON.csv',table);write('SAVED_PARENT_CONTRASTS.csv',contrasts);write('LATEST_PG_EVALUATION_ROWS.csv',case_export)
    # Readable view of the ONE unified table, preserving experiment boundaries.
    out=['# 已有完整任务统一对照表','',
        '来源限定为三个已完成实验的已暴露任务评价。每行只汇总同一实验、端点、方法、cohort；没有跨实验均值。J=L+0.75K。质量按父布局等权，时间为该组原始完整记录时间p50/p95，不是同一次新计时。',
        '', 'DEV=训练父外但已暴露开发组3父/18例；SEL=训练父新任务选择组8父/48例。每父6例（3局部尺寸×2端点）；窗口/重复终点不是独立父样本。',
        '', 'Native raw†是原生完整搜索/内部H后的最终结果，不是去H路线；表中相同raw/H不能解释为H贡献为0。',
        '', '| 实验 | 比较终点/模型端点 | 案例组 | 方法 | 父/例 | raw J | H/最终 J | H L | H K | 完整记录p50/p95秒 |',
        '|---|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in table:
        label='DEV' if r['cohort']==DEV else 'SEL'
        out.append(f'| {r["experiment"]} | {r["comparison_endpoint"]}/{r["model_endpoint"]} | {label} | {r["method"]}{"†" if r["family"]=="原生基线" else ""} | {r["physical_parents"]}/{r["cases"]} | {r["raw_J"]:.6f} | {r["H_J"]:.6f} | {r["H_L"]:.6f} | {r["H_K"]:.6f} | {r["full_recorded_time_p50_s"]:.6f}/{r["full_recorded_time_p95_s"]:.6f} |')
    out+=['','[机器表（包含raw/H L/K、完整来源与计时口径）](TASK_LEVEL_COMPARISON.csv) · [原父布局配对区间](SAVED_PARENT_CONTRASTS.csv) · [最新PG792评价逐例转录](LATEST_PG_EVALUATION_ROWS.csv)',
        '', 'Integrated行使用原协议已经选中的RL128、监督256与SL256，所选更新步数不同；其full_audit_s是开发请求，不是未执行的正式冷/热部署测试。R03用through_record_IO_s，最新PG用经过索引和完成journal的full_request_s，三种时钟分别保留，不能跨实验比较速度。',
        '', '全部为已暴露开发证据；本轮没有重新求解、前向、训练、bootstrap或物理验证。代理约束有效不等于真实IK/关节轨迹认证。']
    (HOME/'TASK_LEVEL_COMPARISON.md').write_text('\n'.join(out)+'\n',encoding='utf-8',newline='\n')
    put('TABLE_BUILD_RECEIPT.json',dict(status='COMPLETE',table_rows=53,latest_PG_evaluation_rows=792,contrast_rows=len(contrasts),
        inputs=INPUTS,elapsed_s=time.perf_counter()-start,new_model_forwards=0,new_optimizer_calls=0,new_planning_requests=0,
        aggregation='quality parent means then equal parent weighting; time empirical request linear quantiles within each row only',bootstrap_rerun=False,
        all_three_experiments_kept_separate=True,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    primary=[r for r in table if r['experiment']=='FROZEN_REFERENCE_PG_A1' and r['comparison_endpoint']==128 and r['cohort']==DEV]
    print(json.dumps(primary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
