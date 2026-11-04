# 委派的 reviewer 任务与输入快照

角色：独立 paper architect/reviewer。父执行者请求审计新G0/G1代码与协议，重点三层非单调、全部retained覆盖、全历史重放、reward/fallback一致、终止/截断、预算与无免费oracle、共同F/A/D、防训练误称。不得改算法。

最终委派：对全32开发场景作原始账本审计、计数重算和结果到主张判断；从叶记录补充每场景最大total_reward、根STOP以及严格变差见证return比较。补充分析不改预注册J主指标。父执行者负责32个selected序列重新执行和总体回归；reviewer不得把读取日志说成亲自重跑全部。

预声明主张来自 design/2026-11-03_G01_PLAN.md：
1. 给定静态F（外部F∩A）和D，共享动作能产生服务配置与后续选择变化；预固定32个百桌状态中可能存在先变差再获新纪录。
2. 环境维护认证anchor/current/best，并在预算、终止、重放和回退时保持代价与奖励一致。

固定设置：32个10×10新开发场景，regular/mixed/blocked=11/11/10；每节点至多8分支含STOP，深度3；每个场景一份认证Fixed3初态，所有失败保留；未训练PPO。协议SHA256：5e1f1e919c4ba6b7914b12b11d26279798beec4537680ba9f9918a6504c3bb8c。

请求输出：claim_supported、what_results_support、what_results_dont_support、missing_evidence、suggested_claim_revision、next_experiments_needed、confidence和routing_action。区分阶段窄主张与更大的DRL主张。无PAPER_CLAIM_AUDIT.json时写作判定provisional。此代理继承上下文，是独立角色审阅而非零上下文。

输入：outputs/g0/protocol.json、所有case_*/menus.jsonl.gz与queries.jsonl.gz、input.json/summary.json/selected_replay.json；outputs/tests_final.log、delivery_verification.json、freeze_after.json、freeze_v51.json；新源码及原始用户设计。实际文件哈希见parsed_verdict.json。

## 独立账本复算程序

以下程序只重建问题、核对保存账本并另算统计，没有再运行完整诊断树。另行7项运行检查记录于outputs/review_independent_probes.json。

```python
import gzip,json,hashlib,math
from pathlib import Path
from collections import Counter
import numpy as np
from rc3_rl_research.adapter import ResearchAdapter
from rc3_improve.repair import route_is_valid
from rc3_fast.solver import route_objective
root=Path('rc3_rl_research/outputs/g0')
protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
digest=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()
assert digest==(root/'protocol.sha256').read_text().split()[0]
reports=[]
for case in protocol['cases']:
    d=root/f"case_{case['case_id']:02d}"
    if not (d/'summary.json').exists():continue
    summary=json.loads((d/'summary.json').read_text(encoding='utf-8'))
    if summary['status']!='complete':
        reports.append(dict(case_id=case['case_id'],status=summary['status']));continue
    assert summary['case']==case
    menus=[json.loads(l) for l in gzip.open(d/'menus.jsonl.gz','rt',encoding='utf-8')]
    rows=[json.loads(l) for l in gzip.open(d/'queries.jsonl.gz','rt',encoding='utf-8')]
    adapter=ResearchAdapter.from_payload(json.loads((d/'input.json').read_text(encoding='utf-8')))
    bypath={tuple(q['path']):q for q in rows}
    assert len(bypath)==len(rows)==summary['queries']
    assert len(menus)==summary['menus_expanded']
    anchor=adapter.anchor.cost; cfg=protocol['config']; tol=1e-7
    leaves=[q for q in rows if q['leaf']]
    checks=0
    for m in menus:
        o=m['observation']; ids=[i for i,v in enumerate(o['mask']) if v and o['actions'][i]['kind']!='stop'][:7]
        ids += [i for i,v in enumerate(o['mask']) if v and o['actions'][i]['kind']=='stop']
        assert ids==m['selected_action_ids'] and len(ids)<=8
        assert [q['action_id'] for q in rows if q['menu_id']==m['menu_id']]==ids
        for a in ids:
            q=bypath[tuple(m['path']+[a])];checks+=1
            assert q['before_current']==o['current'] and q['action']==o['actions'][a]
            assert q['info']['menu_work_units']==o['pending_menu_work']
            assert len(q['path'])<=3
            assert route_is_valid(adapter.problem,q['current']) and route_is_valid(adapter.problem,q['best'])
            assert abs(route_objective(adapter.problem,q['current'])-q['current_cost'])<tol
            assert abs(route_objective(adapter.problem,q['best'])-q['best_cost'])<tol
            assert q['best_cost']<=anchor+tol
            if q['info']['accepted']:
                assert q['current_cost']<=o['best_cost']+cfg['epsilon']*anchor+tol
            proposal=q['info']['proposal']
            if proposal is not None:assert proposal['work_units']==sum(proposal['counters'].values())
            cost=q['finalization']['returned_cost'] if q['leaf'] else q['best_cost']
            expected=100*(anchor-cost)/max(anchor,1e-8)-cfg['lambda_work']*q['work_used']/cfg['work_scale']
            assert abs(q['total_reward']-expected)<tol
    roots=[q for q in rows if len(q['path'])==1]
    one=min(q['best_cost'] for q in roots)
    future={}
    for q in roots:future[q['action_id']]=min(x['finalization']['returned_cost'] for x in leaves if x['path'][0]==q['action_id'])
    worse=[q for q in roots if q['info']['accepted'] and q['current_cost']>anchor+1e-8]
    neutral=[q for q in roots if q['info']['accepted'] and q['current_cost']>=anchor-1e-8 and q['action']['kind']!='stop']
    reversal=any(a['best_cost']<b['best_cost']-1e-8 and future[a['action_id']]>future[b['action_id']]+1e-8 for a in roots for b in roots)
    monotone=min(future[q['action_id']] for q in roots if q['current_cost']<=anchor+1e-8)
    derived=dict(worse_first_beats_one_step=any(future[q['action_id']]<one-1e-8 for q in worse),
      worse_first_then_record=any(future[q['action_id']]<anchor-1e-8 for q in worse),
      nonimproving_first_then_record=any(future[q['action_id']]<anchor-1e-8 for q in neutral),
      ranking_reversal=reversal,worse_first_beats_all_monotone_first=any(future[q['action_id']]<monotone-1e-8 for q in worse))
    assert all(summary[k]==v for k,v in derived.items())
    assert abs(summary['best_depth3_cost']-min(q['finalization']['returned_cost'] for q in leaves))<tol
    assert summary['leaves']==len(leaves)
    assert summary['physical_oracle_work']['proposal_work_units']==sum(q['info']['proposal_work_units'] for q in rows)
    assert summary['physical_oracle_work']['selected_edges']==len(rows)
    by_menu_path={tuple(m['path']):m for m in menus}
    expected_menu_work=by_menu_path[()]['observation']['pending_menu_work']
    expected_menu_builds=1
    for q in rows:
        parent_work=0 if len(q['path'])==1 else bypath[tuple(q['path'][:-1])]['work_used']
        if q['info']['finalization'] is not None:
            next_menu=0
        elif q['leaf']:
            next_menu=q['work_used']-parent_work-q['info']['work_units']
            expected_menu_builds+=1
        else:
            next_menu=by_menu_path[tuple(q['path'])]['observation']['pending_menu_work']
            expected_menu_builds+=1
        assert next_menu>=0
        expected_menu_work+=next_menu
    assert expected_menu_work==summary['physical_oracle_work']['menu_work_units']
    assert expected_menu_builds==summary['physical_oracle_work']['menu_builds']
    stop=next(q for q in roots if q['action']['kind']=='stop')
    max_leaf=max(leaves,key=lambda q:q['total_reward'])
    worse_ids={q['action_id'] for q in worse}
    strict_leaves=[q for q in leaves if q['path'][0] in worse_ids and q['finalization']['returned_cost']<one-1e-8]
    strict_best=max(strict_leaves,key=lambda q:q['total_reward']) if strict_leaves else None
    selected=bypath[tuple(summary['selected_path'])]
    reward_description=dict(root_stop_reward=stop['total_reward'],max_leaf_reward=max_leaf['total_reward'],
       max_leaf_reward_path=max_leaf['path'],max_leaf_reward_minus_stop=max_leaf['total_reward']-stop['total_reward'],
       max_leaf_reward_beats_stop=max_leaf['total_reward']>stop['total_reward']+1e-8,
       strict_leaf_count=len(strict_leaves),strict_max_reward=None if strict_best is None else strict_best['total_reward'],
       strict_max_reward_path=None if strict_best is None else strict_best['path'],
       strict_max_reward_minus_stop=None if strict_best is None else strict_best['total_reward']-stop['total_reward'],
       strict_reward_beats_stop=False if strict_best is None else strict_best['total_reward']>stop['total_reward']+1e-8,
       selected_path=summary['selected_path'],selected_path_is_strict_witness=summary['selected_is_strict_witness'],
       selected_path_reward=selected['total_reward'],selected_path_reward_minus_stop=selected['total_reward']-stop['total_reward'])
    reports.append(dict(case_id=case['case_id'],status='pass',queries=checks,menus=len(menus),leaves=len(leaves),metrics=derived,reward_description=reward_description,physical_menu_work_verified=expected_menu_work))
report=dict(scope='independent raw-ledger recomputation: full retained coverage, objectives, acceptance bounds, menu order, rewards and mechanism counts; no rerun of full search tree',protocol_sha256=digest,planned=32,reviewed=len(reports),cases=reports,
    all_reviewed_pass=all(r['status']=='pass' for r in reports),total_queries=sum(r.get('queries',0) for r in reports),
    recomputed_counts={k:sum(r.get('metrics',{}).get(k,False) for r in reports) for k in ('worse_first_beats_one_step','worse_first_then_record','nonimproving_first_then_record','ranking_reversal','worse_first_beats_all_monotone_first')})
report['supplementary_reward_scope']='Post-registration descriptive reward audit, not a changed G0 primary J criterion. Leaves are correlated within scenes; counts below use scenes.'
report['reward_scene_counts']={k:sum(r.get('reward_description',{}).get(k,False) for r in reports) for k in ('max_leaf_reward_beats_stop','strict_reward_beats_stop')}
Path('rc3_rl_research/outputs/review_g0_records.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='cases'}))
```
