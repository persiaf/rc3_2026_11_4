"""Rebuild all numerical results from saved repeat plans, then apply frozen gates."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from rc3_baseline.baseline import write_json,write_csv
from rc3_baseline.validator import validate

BOOTSTRAP_SAMPLES=20000
BOOTSTRAP_SEED=731203


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reconstruct(results):
    """Ignore saved success booleans when rechecking every exported plan."""
    manifest=read(results/'manifest.json');rows=[];raw=[];errors=[];checks=[];valid_plans=0
    penalty=manifest.get('stop_penalty_equivalent_m')
    if not isinstance(penalty,(int,float)) or not np.isfinite(penalty) or penalty<0:
        raise ValueError('missing or invalid frozen stop penalty')
    for field in ('model_sha256','source_sha256','data_sha256'):
        for path,expected in manifest[field].items():
            actual=sha(path) if Path(path).is_file() else None
            checks.append(dict(path=path,kind=field,unchanged=actual==expected))
            if actual!=expected:errors.append('changed frozen file: '+path)
    files=sorted((results/'plans').glob('scene_plan_*.json'))
    for file in files:
        data=read(file);meta=data['metadata'];groups=defaultdict(list);row_start=len(rows)
        for label,record in data['runs'].items():
            spec=record['config'];item=dict(record);item.pop('runtime',None)
            item.update(**meta,label=label,config=spec['id'],kind=spec['kind'],horizon=spec['horizon'],
                        limit_s=spec['seconds'],repeat=record['repeat'])
            plan=data['plans'].get(label)
            if plan is not None:
                check=validate(data['scene'],plan,data['feasibility'])
                objective=float(check['independent_polyline_length_m'])+penalty*len(plan['selected_candidate_ids'])
                penalty_ok=abs(float(plan.get('stop_penalty_equivalent_m',np.nan))-penalty)<=1e-12
                valid=check['valid_model_plan'] and check['all_tasks_served'] and penalty_ok
                if not penalty_ok:errors.append(f'{file.name}:{label}: plan penalty differs from frozen request')
                if not valid:
                    errors.append(f'{file.name}:{label}: invalid independent plan {check["errors"]}')
                    item['status']='failed'
                if record['status']=='complete':
                    if abs(objective-record['objective_m'])>1e-6 or objective>record['reference_objective_m']+1e-7:
                        errors.append(f'{file.name}:{label}: cost reconstruction/reference bound mismatch')
                        item['status']='failed'
                    if abs(plan['objective_length_plus_stops_m']-objective)>1e-6:
                        errors.append(f'{file.name}:{label}: exported objective mismatch');item['status']='failed'
                item['objective_m']=objective
                item['independent_valid']=bool(valid)
                valid_plans+=int(valid)
            else:
                item['status']='failed';item['independent_valid']=False
                if record['status']=='complete':errors.append(f'{file.name}:{label}: missing successful plan')
            if item['status']!='complete':errors.append(f'{file.name}:{label}: failed run retained')
            groups[spec['id']].append(item);raw.append(item)
        for config,items in groups.items():
            first=items[0];spec=data['runs'][first['label']]['config']
            row=dict(**meta,config=config,kind=spec['kind'],repeats=len(items),expected_repeats=spec['repeats'],
                     horizon=spec['horizon'],limit_s=spec['seconds'],preprocessing_s=data.get('preprocessing_s'),
                     training_seed=spec.get('training_seed'),status='complete',successful_repeats=sum(r['status']=='complete' for r in items))
            if len(items)!=spec['repeats'] or len({r['repeat'] for r in items})!=len(items):
                errors.append(f'{file.name}:{config}: missing or duplicate repetition');row['status']='failed'
            if any(r['status']!='complete' for r in items):row['status']='failed'
            if row['status']=='complete':
                costs=np.array([r['objective_m'] for r in items]);refs=np.array([r['reference_objective_m'] for r in items])
                if spec['kind']!='wall' and np.ptp(costs)>1e-6:
                    errors.append(f'{file.name}:{config}: deterministic cost changes across repeats');row['status']='failed'
                if np.ptp(refs)>1e-6:
                    errors.append(f'{file.name}:{config}: reference changes across repeats');row['status']='failed'
                planning=float(np.median([r['planning_s'] for r in items]))
                row.update(objective_m=float(costs.mean()),objective_min_m=float(costs.min()),objective_max_m=float(costs.max()),
                    reference_objective_m=float(refs.mean()),improvement_pct=float(np.mean(100*(refs-costs)/np.maximum(refs,1e-8))),
                    planning_s=planning,total_s=planning+row['preprocessing_s'],
                    dp_calls=float(np.mean([r['dp_calls'] for r in items])),accepted=float(np.mean([r['accepted'] for r in items])),
                    overrun_rate=float(np.mean([r['deadline_exceeded'] for r in items])),
                    mean_overrun_s=float(np.mean([r['overrun_s'] for r in items])),max_overrun_s=max(r['overrun_s'] for r in items),
                    fallback_rate=float(np.mean([r['fallback_used'] for r in items])),
                    policy_stop_rate=float(np.mean([r['stop_reason']=='policy_stop' for r in items])),
                    graph_feature_s=float(np.median([r.get('policy_graph_s',0.) for r in items])),
                    graph_inference_s=float(np.median([r.get('policy_inference_s',0.) for r in items])))
            rows.append(row)
        # A separately validated saved Fixed3 plan anchors every method's bound;
        # do not rely solely on each method's self-reported reference field.
        local_rows=rows[row_start:]
        fixed=[r for r in local_rows if r['config']=='fixed3' and r['status']=='complete']
        if len(fixed)!=1:
            errors.append(f'{file.name}: no unique independently validated Fixed3 aggregate')
        else:
            reference=fixed[0]['objective_m']
            for row in local_rows:
                if row['status']=='complete' and (abs(row['reference_objective_m']-reference)>1e-6 or row['objective_m']>reference+1e-7):
                    errors.append(f'{file.name}:{row["config"]}: bound differs from saved independently validated Fixed3')
                    row['status']='failed'
    if len(files)!=manifest['cases']:errors.append('wrong number of saved scenes')
    if len(raw)!=manifest['planned_runs']:errors.append('wrong number of planned repeated runs')
    if manifest['status']!='complete':errors.append('evaluation manifest is not complete')
    verification=dict(all_passed=not errors,errors=errors,scene_files=len(files),repeated_runs=len(raw),
        valid_plans=valid_plans,scene_method_records=len(rows),hash_checks=checks,
        all_hashes_unchanged=all(c['unchanged'] for c in checks),
        note='All plans revalidated from geometry and service assignment. All objectives recomputed from independent polyline length and service-stop count. Failures stay in denominators.')
    return rows,raw,verification


def paired(current,reference):
    key=lambda r:(r['seed'],r['nx'],r['ny'],r['mode'])
    lookup={key(r):r for r in reference if r['status']=='complete'}
    pairs=sorted([(r,lookup[key(r)]) for r in current if r['status']=='complete' and key(r) in lookup],key=lambda p:key(p[0]))
    if not pairs:return None
    delta=np.array([100*(b['objective_m']-a['objective_m'])/max(b['objective_m'],1e-8) for a,b in pairs])
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    ci=np.quantile(delta[rng.integers(len(delta),size=(BOOTSTRAP_SAMPLES,len(delta)))].mean(1),[.025,.975])
    return dict(paired_n=len(pairs),mean_reduction_pct=float(delta.mean()),ci95_pct=ci.tolist(),
        wins=int((delta>1e-7).sum()),ties=int((np.abs(delta)<=1e-7).sum()),losses=int((delta< -1e-7).sum()),
        median_paired_time_ratio=float(np.median([a['planning_s']/max(b['planning_s'],1e-12) for a,b in pairs])))


def summarize(rows):
    groups={'main_100':lambda r:r['tables']==100,'transfer_36':lambda r:r['tables']==36,
            'transfer_196':lambda r:r['tables']==196,'stress_400':lambda r:r['tables']==400,'all_39':lambda r:True}
    summaries=[];comparisons=[]
    for group,predicate in groups.items():
        subset=[r for r in rows if predicate(r)];configs=sorted({r['config'] for r in subset})
        for config in configs:
            selected=[r for r in subset if r['config']==config];good=[r for r in selected if r['status']=='complete']
            record=dict(group=group,config=config,kind=selected[0]['kind'],total=len(selected),success=len(good),failed=len(selected)-len(good))
            if good:
                record.update(mean_objective_m=float(np.mean([r['objective_m'] for r in good])),
                    mean_improvement_pct=float(np.mean([r['improvement_pct'] for r in good])),
                    median_planning_ms=float(np.median([r['planning_s'] for r in good])*1000),
                    p95_planning_ms=float(np.quantile([r['planning_s'] for r in good],.95)*1000),
                    median_total_ms=float(np.median([r['total_s'] for r in good])*1000),
                    mean_dp_calls=float(np.mean([r['dp_calls'] for r in good])),
                    mean_overrun_rate=float(np.mean([r['overrun_rate'] for r in good])),
                    mean_overrun_ms=float(np.mean([r['mean_overrun_s'] for r in good])*1000),
                    max_overrun_ms=float(max(r['max_overrun_s'] for r in good)*1000),
                    mean_stop_rate=float(np.mean([r['policy_stop_rate'] for r in good])),
                    median_graph_feature_ms=float(np.median([r['graph_feature_s'] for r in good])*1000),
                    median_graph_inference_ms=float(np.median([r['graph_inference_s'] for r in good])*1000))
            summaries.append(record)
        specs=[('structured8',r) for r in ('control8','detour8','v3_mlp8','fixed3')]
        specs += [('control8','detour8'),('detour8','fixed3')]
        specs += [('structured_wall400',r) for r in ('control_wall400','detour_wall400')]
        for seed in (17,29,43):
            specs += [(f'structured_seed{seed}_calls8',r) for r in (f'control_seed{seed}_calls8','detour8')]
        for config,reference in specs:
            value=paired([r for r in subset if r['config']==config],[r for r in subset if r['config']==reference])
            if value:comparisons.append(dict(group=group,config=config,reference=reference,**value))
    return dict(summaries=summaries,comparisons=comparisons,bootstrap_samples=BOOTSTRAP_SAMPLES,
        bootstrap_seed=BOOTSTRAP_SEED,bootstrap_unit='scene; mean cost over repeated runs within scene',
        note='Positive reduction means current is cheaper. Confidence intervals omit training-seed uncertainty and multiplicity correction. all_39 is descriptive and mixes sizes; main_100 is the registered decision set. Seed runs are reported separately, never used to select test-best weights.')


def decision(summary,verification):
    comparisons={r['reference']:r for r in summary['comparisons'] if r['group']=='main_100' and r['config']=='structured8'}
    stats={r['config']:r for r in summary['summaries'] if r['group']=='main_100'}
    gates={}
    for reference in ('control8','detour8'):
        value=comparisons.get(reference,{})
        gates['quality_vs_'+reference]=(value.get('paired_n')==18 and value.get('mean_reduction_pct',-1)>=.2
                                       and value.get('ci95_pct',[-1])[0]>0)
    structured,detour=stats.get('structured8',{}),stats.get('detour8',{})
    ratio=(structured['median_planning_ms']/detour['median_planning_ms']) if structured.get('median_planning_ms') is not None and detour.get('median_planning_ms',0)>0 else None
    gates['median_time_ratio_at_most_1_25']=ratio is not None and ratio<=1.25
    gates['all_plans_valid']=bool(verification['all_passed'])
    adopted=all(gates.values())
    return dict(adopt_structured=adopted,gates=gates,median_time_ratio_structured_to_detour=ratio,
        default='structured_optional' if adopted else 'fixed3_or_detour',
        statement='The registered thresholds pass on this finite new test set; actual robot timing and service constraints remain unvalidated.' if adopted else 'The structural model did not satisfy every registered quality/time/reliability gate. Retain Fixed3 for fast plans and optional Detour repair; keep learned weights as research artifacts.',
        threshold_scope='0.2% and 1.25x are study decision thresholds, not measured application requirements.')


def _f(value,digits=3):return '—' if value is None else f'{value:.{digits}f}'


def markdown(summary,verdict,verification,manifest):
    lines=['# v4 结构输入消融：冻结新场景结果','',
        '**采用决策：'+('达到本轮预注册门槛，可作为可选研究部署。' if verdict['adopt_structured'] else '未满足全部预注册门槛；默认继续使用 Fixed3／Detour。')+'**','',
        f"本轮为 {verification['scene_files']} 个新场景、{verification['scene_method_records']} 组场景—方法记录、{verification['repeated_runs']} 次运行；独立验证通过 {verification['valid_plans']} 份。重复运行不能当作独立场景。",'',
        'control 与 structured 使用相同 38,438 参数网络、同监督标签、三训练种子和训练预算，只改变图输入是否屏蔽。主表检查点只在原18个验证场景选择，三个种子的新测试另列；没有根据测试结果改选权重。','',
        'v3_mlp8 保持第三版冻结权重不变，使用经过数值一致性验证的工程适配器，将线程控制从每动作移至整次规划；没有重新训练。此表比较的是优化后的同权重部署。','',
        '成本为 J=L+0.75K。每场景重复三次的成本取均值、完整规划耗时取中位，再在场景间汇总；冷总时间加入共同导航／可达性预处理。计时含初解构造、特征、模型、修复、终验和线程控制，不含模型加载、进程启动、场景生成、额外审计和写盘。','',
        '## 调用预算：最多8次修复','',
        '|组／场景数|方法|平均成本|相对Fixed3降本 %|规划中位 ms|冷总中位 ms|平均调用|',
        '|---|---|---:|---:|---:|---:|---:|']
    for group in ('main_100','transfer_36','transfer_196','stress_400','all_39'):
        for config in ('fixed3','detour8','v3_mlp8','control8','structured8'):
            r=next((v for v in summary['summaries'] if v['group']==group and v['config']==config),None)
            if r:lines.append(f"|{group} / {r['total']}|{config}|{_f(r.get('mean_objective_m'))}|{_f(r.get('mean_improvement_pct'))}|{_f(r.get('median_planning_ms'),2)}|{_f(r.get('median_total_ms'),2)}|{_f(r.get('mean_dp_calls'),2)}|")
    lines += ['', '## 主要配对比较：100桌18场景','', '|当前／参照|平均降本 %|95%场景区间|胜／平／负|','|---|---:|---|---|']
    for r in summary['comparisons']:
        if r['group']=='main_100' and r['config'] in ('structured8','control8','detour8'):
            lines.append(f"|{r['config']} / {r['reference']}|{r['mean_reduction_pct']:.4f}|[{r['ci95_pct'][0]:.4f}, {r['ci95_pct'][1]:.4f}]|{r['wins']}/{r['ties']}/{r['losses']}|")
    lines += ['',f'配对 bootstrap {BOOTSTRAP_SAMPLES:,} 次，固定种子 {BOOTSTRAP_SEED}，单位为场景。区间未包含训练种子不确定性，未做多重比较修正；规模分组属于补充结果。','',
        '## 400 ms软预算：100桌18场景','', '|方法|降本 %|实际规划中位 ms|平均调用|超时比例 %|最大超时 ms|','|---|---:|---:|---:|---:|---:|']
    for config in ('detour_wall400','control_wall400','structured_wall400'):
        r=next((v for v in summary['summaries'] if v['group']=='main_100' and v['config']==config),None)
        if r:lines.append(f"|{config}|{_f(r.get('mean_improvement_pct'))}|{_f(r.get('median_planning_ms'),2)}|{_f(r.get('mean_dp_calls'),2)}|{_f(100*r.get('mean_overrun_rate',0),2)}|{_f(r.get('max_overrun_ms'),2)}|")
    lines += ['', '软预算包含初解，但正在运行的DP和强制终验可以跨过截止时间；这不是硬实时承诺。此处256步安全上限超出训练8步范围。','',
        '## 三个训练种子：100桌18场景各运行一次','', '|方法／种子|降本 %|规划中位 ms|平均调用|','|---|---:|---:|---:|']
    for r in summary['summaries']:
        if r['group']=='main_100' and r['kind']=='seed':lines.append(f"|{r['config']}|{_f(r.get('mean_improvement_pct'))}|{_f(r.get('median_planning_ms'),2)}|{_f(r.get('mean_dp_calls'),2)}|")
    lines += ['', '上述单次种子耗时仅作补充，主计时结论使用三次重复的验证集选中检查点。','',
        '## 预注册验收','', '|检查|通过|','|---|---|']
    for name,value in verdict['gates'].items():lines.append(f'|{name}|{value}|')
    lines += ['',f"structured / Detour 完整规划中位时间比：{_f(verdict['median_time_ratio_structured_to_detour'])}。质量门槛为同时相对control和Detour平均降本≥0.2%、场景区间下界>0；耗时门槛≤1.25倍；全部独立验证通过。阈值是研究决策标准，实际机器人等待时间尚未给定。",'',
        'STOP触发比例、图特征／推理分量、全部配对与逐场景结果见 summary.json、metrics.json/csv。调用减少不自动表示STOP有效，也不能表示整体更快。','',
        '当前所有验证只支持离散候选、导航图和代理服务约束内的可靠性；没有真实双臂IK、互碰或实机执行证据。400桌只有3个压力场景。','',
        f"冻结文件一致性：{verification['all_hashes_unchanged']}；总体复验：{verification['all_passed']}。"]
    if verification['errors']:lines += ['', '审计错误：', *['- '+e for e in verification['errors']]]
    return '\n'.join(lines)+'\n'


def generate(results):
    results=Path(results);manifest=read(results/'manifest.json')
    rows,raw,verification=reconstruct(results)
    summary=summarize(rows);verdict=decision(summary,verification)
    write_json(results/'metrics.json',rows);write_json(results/'repeat_metrics.json',raw)
    write_csv(results/'metrics.csv',rows,list(dict.fromkeys(k for r in rows for k in r)))
    write_json(results/'summary.json',summary);write_json(results/'verification.json',verification)
    write_json(results/'decision.json',verdict)
    (results/'RESULTS.md').write_text(markdown(summary,verdict,verification,manifest),encoding='utf-8')
    return dict(summary=summary,decision=verdict,verification=verification)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path('rc3_v4/results'))
    args=parser.parse_args(argv);result=generate(args.results)
    print(json.dumps(dict(all_passed=result['verification']['all_passed'],adopt_structured=result['decision']['adopt_structured'])))
    return 0 if result['verification']['all_passed'] else 2


if __name__=='__main__':raise SystemExit(main())
