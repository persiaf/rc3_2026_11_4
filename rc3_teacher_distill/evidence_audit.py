"""T0: read saved R0/R1 evidence only; no planner, inference, H or training imports.

Reconstructs union coverage from saved scene geometry using the frozen proxy
formula only. It does not compute navigation, generate scenes or new labels.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
TOL = 1e-7


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Unexpected JSON value: {type(value).__name__}')


def summary(values):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return dict(n=0, mean=None, median=None, p95=None, minimum=None, maximum=None)
    assert np.isfinite(x).all()
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)),
                p95=float(np.quantile(x, .95)), minimum=float(x.min()), maximum=float(x.max()))


class Audit:
    def __init__(self):
        self.inputs = {}
        self.scenes = {}
        self.covers = {}
        self.checks = Counter()

    def read(self, name, jsonl=False):
        p = ROOT/name
        assert p.is_relative_to(ROOT/'rc3_nav_value') or name in (
            'rc3_constructive/baselines.py', 'rc3_fast/solver.py')
        self.inputs[name] = dict(sha256=sha(p), bytes=p.stat().st_size)
        txt = p.read_text(encoding='utf-8')
        return [json.loads(s) for s in txt.splitlines() if s.strip()] if jsonl else json.loads(txt)

    def scene(self, family, case):
        key = (family, case)
        if key not in self.scenes:
            filename = (f'rc3_nav_value/r0/scenes/case{case:02}.json' if family == 'r0'
                        else f'rc3_nav_value/r1/scenes/train_{case:02}.json')
            s = self.read(filename)
            t = np.asarray([[t['x'], t['y'], t['z']] for t in s['tasks']], float).reshape(-1, 3)
            c = np.asarray([[c['x'], c['y'], c['theta']] for c in s['candidates']], float).reshape(-1, 3)
            r = s['robot']
            dx, dy = t[:, None, 0]-c[None, :, 0], t[:, None, 1]-c[None, :, 1]
            co, si = np.cos(c[:, 2]), np.sin(c[:, 2])
            x = co*dx + si*dy-r['arm_forward_offset']
            y = (-si*dx+co*dy)[..., None]-np.array([1., -1.])*r['arm_spacing']/2
            z = t[:, None, 2]-r['shoulder_height']
            d = np.sqrt(x[..., None]**2+y**2+z[..., None]**2)
            F = ((d >= r['arm_min_reach']) & (d <= r['arm_max_reach'])
                 & (np.abs(np.arctan2(y, x[..., None])) <= math.radians(r['arm_sector_half_angle_deg'])))
            self.scenes[key], self.covers[key] = s, F.any(2)
        return self.scenes[key], self.covers[key]

    def state(self, family, rec):
        scene, C = self.scene(family, rec['case'])
        remaining = np.ones(len(C), bool)
        for n in rec['prefix']:
            assert isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= C.shape[1]
            assert (remaining & C[:, n-1]).any()
            remaining &= ~C[:, n-1]
            self.checks['saved_prefix_positive_coverage_steps'] += 1
        if family == 'r0':
            assert int(remaining.sum()) == rec['remaining_tasks']
            self.checks['r0_saved_remaining_tasks_equal_geometry_replay'] += 1
        return dict(case=rec['case'], mode=scene['complexity'], prefix_length=len(rec['prefix']),
                    completed_fraction=1-float(remaining.mean()), remaining_tasks=int(remaining.sum()),
                    tasks=len(C), total_candidates=C.shape[1],
                    positive_coverage_candidate_count=int(C[remaining].any(0).sum()))

    def menus(self, records, family):
        per_state = []
        for r in records:
            if family == 'r0':
                ys = np.array([c['Y'] for c in r['candidates']], float)
                actions = [c['node'] for c in r['candidates']]
                ref = actions.index(r['reference_action'])
            else:
                ys, actions, ref = np.asarray(r['Y'], float), r['actions'], 0
                assert abs(ys[ref]-r['Yref']) <= TOL
                assert np.allclose(np.array(r['target']), (r['Yref']-ys)/r['Yref'], atol=1e-12, rtol=0)
                self.checks['r1_saved_reference_and_targets_recomputed'] += 1
            assert len(ys) == len(actions) == len(set(actions)) and len(ys) > 0
            assert np.isfinite(ys).all() and min(ys) > 0
            best = float(ys.min()); ordered = np.sort(ys)
            positive = ys[ys > best+TOL]
            regs = 100*(ys-best)/best
            x = dict(**self.state(family, r), menu_size=len(ys),
                     best_Y=best, first_second_gap_percent=100*(ordered[1]-best)/best if len(ys)>1 else 0.,
                     first_strictly_worse_gap_percent=100*(positive.min()-best)/best if len(positive) else None,
                     best_tie_count=int(np.sum(ys <= best+TOL)),
                     all_tied=bool((ys <= best+TOL).all()),
                     near_0_1_action_fraction=float(np.mean(regs <= .1+1e-12)),
                     near_0_5_action_fraction=float(np.mean(regs <= .5+1e-12)),
                     reference_regret_percent=float(regs[ref]),
                     best_over_reference_gain_percent=100*(ys[ref]-best)/ys[ref],
                     worst_regret_percent=float(regs.max()))
            if family == 'r0':
                x['immediate_rule_regret_percent'] = float(regs[actions.index(r['immediate_action'])])
                x['chosen_regret_percent'] = float(regs[actions.index(r['chosen_action'])])
                x['legal_action_count_saved'] = r['legal_actions']
                x['query_menu_legal_fraction'] = len(actions)/r['legal_actions']
            per_state.append(x)
        ans = dict(states=len(per_state), scenes=len(set(x['case'] for x in per_state)),
                   unique_full_prefixes=len(set((r['case'], tuple(r['prefix'])) for r in records)),
                   first_second_gap_percent=summary([x['first_second_gap_percent'] for x in per_state]),
                   first_strictly_worse_gap_percent=summary([x['first_strictly_worse_gap_percent'] for x in per_state if x['first_strictly_worse_gap_percent'] is not None]),
                   best_tie_states=sum(x['best_tie_count'] > 1 for x in per_state),
                   all_tied_states=sum(x['all_tied'] for x in per_state),
                   near_0_1_action_fraction=summary([x['near_0_1_action_fraction'] for x in per_state]),
                   near_0_5_action_fraction=summary([x['near_0_5_action_fraction'] for x in per_state]),
                   reference_regret_percent=summary([x['reference_regret_percent'] for x in per_state]),
                   best_over_reference_gain_percent=summary([x['best_over_reference_gain_percent'] for x in per_state]),
                   worst_regret_percent=summary([x['worst_regret_percent'] for x in per_state]),
                   state_distribution=self.distribution(per_state), per_state=per_state)
        if family == 'r0':
            ans.update(immediate_rule_regret_percent=summary([x['immediate_rule_regret_percent'] for x in per_state]),
                       chosen_regret_percent=summary([x['chosen_regret_percent'] for x in per_state]),
                       query_menu_legal_fraction=summary([x['query_menu_legal_fraction'] for x in per_state]))
        return ans

    def distribution(self, states):
        return dict(prefix_length=summary([s['prefix_length'] for s in states]),
                    completed_fraction=summary([s['completed_fraction'] for s in states]),
                    coverage_quarters=dict(Counter(str(min(3, int(s['completed_fraction']*4))) for s in states)),
                    modes=dict(Counter(s['mode'] for s in states)),
                    states_per_scene=dict(sorted(Counter(str(s['case']) for s in states).items(), key=lambda x:int(x[0]))),
                    positive_coverage_candidate_count=summary([s['positive_coverage_candidate_count'] for s in states]),
                    navigation_legality_note='No navigation recomputation: positive-coverage count is an upper bound, not the exact legal action count')

    def model_audit(self, rows, label_index):
        records, pair_correct, pair_count, squared = [], 0, 0, []
        for r in rows:
            key = (r['round'], r['group'], r['case'], tuple(r['prefix']))
            label = label_index[key]
            assert label['reference_hash'] == r['reference_hash']
            assert label['actions'] == r['queried']
            ys = np.array(label['Y'], float); ps = np.array(r['queried_prediction'], float)
            ts = (r['Yref']-ys)/r['Yref']
            assert np.allclose(ts, r['queried_target'], atol=1e-12, rtol=0)
            assert abs(ys.min()-r['known_best']) <= TOL
            unknown = r['selected'] not in r['queried']; assert unknown == r['unknown']
            if not unknown: assert abs(r['Yselected']-ys[r['queried'].index(r['selected'])]) <= TOL
            selected_index, canonical_best = int(ps.argmax()), int(ys.argmin())
            chosen_y, best = ys[selected_index], ys.min()
            for i in range(len(ys)):
                for j in range(i+1, len(ys)):
                    if abs(ts[i]-ts[j]) > 1e-8:
                        pair_count += 1
                        pair_correct += int(np.sign(ts[i]-ts[j]) == np.sign(ps[i]-ps[j]))
            squared.extend(((ps-ts)**2).tolist())
            true = (r['Yref']-r['Yselected'])/r['Yref']
            records.append(dict(case=r['case'], prefix_length=len(r['prefix']), unknown=unknown,
                canonical_argmin_top1=selected_index == canonical_best,
                tied_best_top1=chosen_y <= best+TOL,
                queried_selected_regret_percent=float(100*(chosen_y-best)/best),
                queried_selected_near_0_1=100*(chosen_y-best)/best <= .1+1e-12,
                queried_selected_near_0_5=100*(chosen_y-best)/best <= .5+1e-12,
                actual_selected_gap_to_queried_best_percent=100*(r['Yselected']-best)/best,
                actual_positive_regret_to_queried_best_percent=max(0.,100*(r['Yselected']-best)/best),
                actual_selected_positive_regret_Yref_percent=max(0.,100*(r['Yselected']-best)/r['Yref']),
                actual_advantage_percent=100*true,
                optimism_bias_percentage_points=100*(r['predicted']-true),
                predicted_positive_true_negative=r['predicted'] > 0 and true < 0))
            self.checks['r1_actual_action_audit_joined_and_consistent'] += 1
        unknown = [x for x in records if x['unknown']]
        return dict(states=len(rows), scope='Seen training states; no independent-development action labels or unseen-state claim',
            queried_advantage_MSE=float(np.mean(squared)), queried_pair_sign_accuracy=pair_correct/pair_count,
            queried_nontied_pairs=pair_count,
            canonical_argmin_top1_rate=float(np.mean([x['canonical_argmin_top1'] for x in records])),
            tied_best_top1_rate=float(np.mean([x['tied_best_top1'] for x in records])),
            queried_selected_regret_percent=summary([x['queried_selected_regret_percent'] for x in records]),
            queried_selected_near_0_1_rate=float(np.mean([x['queried_selected_near_0_1'] for x in records])),
            queried_selected_near_0_5_rate=float(np.mean([x['queried_selected_near_0_5'] for x in records])),
            actual_unknown_count=len(unknown), actual_unknown_rate=len(unknown)/len(records),
            actual_selected_gap_to_queried_best_percent=summary([x['actual_selected_gap_to_queried_best_percent'] for x in records]),
            actual_positive_regret_to_queried_best_percent=summary([x['actual_positive_regret_to_queried_best_percent'] for x in records]),
            unknown_selected_advantage_percent=summary([x['actual_advantage_percent'] for x in unknown]),
            unknown_positive_regret_Yref_percent=summary([x['actual_selected_positive_regret_Yref_percent'] for x in unknown]),
            unknown_optimism_bias_percentage_points=summary([x['optimism_bias_percentage_points'] for x in unknown]),
            unknown_predicted_positive_true_negative_rate=float(np.mean([x['predicted_positive_true_negative'] for x in unknown])) if unknown else None,
            per_state=records)


def main():
    started = time.perf_counter(); a = Audit()
    p0, p1 = a.read('rc3_nav_value/r0/protocol.json'), a.read('rc3_nav_value/r1/protocol.json')
    initial = a.read('rc3_nav_value/r1/labels_initial.jsonl', True)
    rounds = a.read('rc3_nav_value/r1/labels_round.jsonl', True)
    audits = a.read('rc3_nav_value/r1/audit.jsonl', True)
    r0 = a.read('rc3_nav_value/r0/checkpoints.jsonl', True)
    r0eval = a.read('rc3_nav_value/r0/evaluations.jsonl', True)
    terminal = a.read('rc3_nav_value/r1/terminal.jsonl', True)
    promotion = a.read('rc3_nav_value/r1/promotion.json')
    seeds0, seeds1 = a.read('rc3_nav_value/r0/seeds.json'), a.read('rc3_nav_value/r1/seeds.json')
    assert p0['H_sha256'] == p1['H_sha256']
    source_checks = {}
    for name, old in p1['source_sha256'].items():
        path = ROOT/name
        actual = sha(path); a.inputs[name] = dict(sha256=actual, bytes=path.stat().st_size)
        source_checks[name] = actual == old
    assert all(source_checks.values())
    ev0 = {x['key']:x for x in r0eval}
    for r in r0:
        for c in r['candidates']:
            ev = ev0[c['key']]
            assert ev['prefix'] == r['prefix'] and ev['action'] == c['node'] and abs(ev['final']['J']-c['Y']) <= TOL
            assert ev['raw_route'][1:len(r['prefix'])+1] == r['prefix']
            a.checks['r0_candidate_to_complete_prefix_terminal_join'] += 1
    b_hashes = {}
    for case in range(32):
        b = a.read(f'rc3_nav_value/r0/baselines/case{case:02}.json')
        assert signature(b['B']) == b['B_hash']
        b_hashes[str(case)] = b['B_hash']
    warm_by_case = {r['metadata']['case']:r for r in terminal
                    if r['kind']=='native' and r['metadata'].get('tag')=='train_referenceB'}
    terminal_cache = {t['cache_key_sha256']:t for t in terminal
                      if t['kind']=='full_H' and t.get('cache_key_sha256')}
    for r in initial+rounds:
        if r['reference'] == 'gB':
            assert r['reference_hash'] == signature(warm_by_case[r['case']]['final_route'])
        else:
            rep = 'base' if r['group'] in ('A','C') else 'nav'
            assert r['reference_hash'] == promotion[rep]['reference_tensor_hash']
            assert r['reference'] == rep+'_g0' and not promotion[rep]['promoted']
        for action, y in zip(r['actions'],r['Y']):
            cache_key = signature(dict(scene=r['scene_sha256'],prefix=r['prefix'],action=action,
                                       reference=r['reference_hash'],H=p1['H_sha256']))
            t = terminal_cache[signature(cache_key)]
            assert abs(t['J']-y) <= TOL
            assert t['source_versions']['H_sha256'] == p1['H_sha256']
            assert t['raw_route'][1:len(r['prefix'])+1] == r['prefix']
            a.checks['r1_label_to_complete_prefix_versioned_terminal_join'] += 1
    outputs = dict(r0=a.menus(r0,'r0'), r1_initial=a.menus(initial,'r1'), r1_rounds={}, r1_model_audit={})
    index = {(r['round'],r['group'],r['case'],tuple(r['prefix'])):r for r in rounds}
    for r in audits:
        label = index[(r['round'],r['group'],r['case'],tuple(r['prefix']))]
        cache_key = signature(dict(scene=label['scene_sha256'],prefix=r['prefix'],action=r['selected'],
                                   reference=r['reference_hash'],H=p1['H_sha256']))
        t = terminal_cache[signature(cache_key)]
        assert abs(t['J']-r['Yselected']) <= TOL
        assert t['raw_route'][1:len(r['prefix'])+1] == r['prefix']
        assert t['raw_route'][len(r['prefix'])+1] == r['selected']
        a.checks['r1_actual_selected_action_versioned_terminal_join'] += 1
    for round_id in (1,2):
        for group in 'ABCD':
            name = f'{group}{round_id}'
            records = [r for r in rounds if r['round']==round_id and r['group']==group]
            outputs['r1_rounds'][name] = a.menus(records,'r1')
            outputs['r1_model_audit'][name] = a.model_audit([r for r in audits if r['round']==round_id and r['group']==group],index)
    ref_counts = Counter((r['reference'],r['reference_hash']) for r in initial+rounds)
    old_seed_pairs = {(s['map_seed'],s['task_seed']) for s in seeds0}
    train_pairs = {(s['map_seed'],s['task_seed']) for s in seeds1['train']}
    dev_pairs = {(s['map_seed'],s['task_seed']) for s in seeds1['dev']}
    assert not old_seed_pairs & train_pairs and not old_seed_pairs & dev_pairs and not train_pairs & dev_pairs
    result = dict(schema='rc3-teacher-distill-t0-read-only-v1', maintenance_base='5d5fe25754ffa032f97937a3fb54f15901c2fb7d',
                  frozen_research_commit='df8cf28b0ece2271bd86d5eceee8ea5ca71a65d0',
                  new_scene_generation=0,new_planning=0,new_H=0,new_neural_inference=0,new_optimizer_updates=0,retained_test_reads=0,
                  formulas=dict(menu_regret_percent='100*(Yaction-min queried Y)/min queried Y',
                    near_optimal_tolerances_percent=[.1,.5],best_tie_absolute_J_tolerance=TOL,
                    top1='Canonical argmin uses saved action order and raw floating costs; tied-best uses absolute J tolerance1e-7',
                    warning='Unknown action signed gap to queried best may be negative; queried best is not a global oracle'),
                  reference_versions=dict(H_sha256=p1['H_sha256'],frozen_sources_match=source_checks,
                    R0_gB_hashes=b_hashes,R1_counts=[dict(name=k[0],hash=k[1],states=v) for k,v in sorted(ref_counts.items())],
                    distinction='R0 and R1 warm labels use scene-specific Fixed3 gB; R1 round labels use representation-specific frozen warm neural g0; do not mix these as one teacher'),
                  sample_independence=dict(R0_scenes=32,R1_train_scenes=32,R1_dev_scenes=32,
                    seed_pairs_disjoint=True,R1_action_audits='Training states only; the separate dev32 contain route outcomes, not per-action labels',
                    R1_dev32='Already used for promotion and screen; never untouched confirmation'),
                  checks=dict(a.checks),diagnostics=outputs,raw_inputs=a.inputs,
                  elapsed_s=time.perf_counter()-started,
                  limitations=['No saved full navigation matrix is regenerated; R0 legal-action counts are saved, R1 positive coverage counts only upper-bound legality.',
                    'No new model fit: feature separability/generalization cannot be established by T0 label diagnostics.',
                    'Queries are biased bounded menus on teacher or warm-network states; menu opportunity does not guarantee full-route student benefit.'])
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'t0_diagnostics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,
        allow_nan=False,default=json_default)+'\n',encoding='utf-8',newline='\n')
    d=outputs['r1_model_audit']['D2']; ri=outputs['r1_initial']; rz=outputs['r0']
    lines=['# T0：保存教师标签与动作可学习性诊断','',
      '本轮仅对保存的 R0/R1 证据做只读分析，未生成场景、导航、规划、后处理 H、神经推理或训练；原60个保留测试未读。诊断脚本只依赖标准库和NumPy。',
      '', '## 标签版本首先需要区分','',
      f"R0 的256个菜单及R1暖启动的508个菜单采用场景内固定原生Fixed3顺序gB。R1两轮A/C使用base_g0、B/D使用nav_g0，均为已冻结暖起点网络，并非gB。两种晋升均未通过，后轮参考仍为g0。共同H哈希为`{p1['H_sha256']}`；17个冻结源文件均匹配事前哈希。",
      '', '因此，新T1采用单一冻结gB时，不能直接把R1 round标签混入当作同一教师。保存的完整有序前缀已与实际终局记录逐动作核对；完整输入字节哈希与检查计数见JSON。',
      '', '## 菜单成本间隔与近似并列','',
      '| 保存数据 | 状态数 | 最佳并列状态 | 全菜单并列 | 最佳/次佳间隔均值% | 0.1%近优动作比例均值 | 参考动作遗憾均值% |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for name,x in [('R0教师轨迹',rz),('R1初始gB',ri)]+[(k,v) for k,v in outputs['r1_rounds'].items()]:
        lines.append(f"| {name} | {x['states']} | {x['best_tie_states']} | {x['all_tied_states']} | {x['first_second_gap_percent']['mean']:.6f} | {100*x['near_0_1_action_fraction']['mean']:.2f}% | {x['reference_regret_percent']['mean']:.6f} |")
    lines += ['',f"R0菜单内即时规则的平均遗憾为{rz['immediate_rule_regret_percent']['mean']:.6f}%，教师选择遗憾均值为{rz['chosen_regret_percent']['mean']:.9f}%。R0查询菜单仅占合法动作平均{100*rz['query_menu_legal_fraction']['mean']:.3f}%，已查询最优不是全动作最优。",
      '', '严格动作编号Top-1会受到近似并列和浮点最小差异影响。新研究应同时验收并列容差内Top-1、预定0.1%与0.5%近优率和真实成本遗憾，不能只优化编号命中。',
      '', '## R1模型实际选择与可靠性','',
      '| 模型 | 查询内原始argmin命中 | 容差并列最优命中 | 查询内0.1%近优 | 查询内平均遗憾% | 全菜单未知动作 |',
      '|---|---:|---:|---:|---:|---:|']
    for name,x in outputs['r1_model_audit'].items():
        lines.append(f"| {name} | {100*x['canonical_argmin_top1_rate']:.2f}% | {100*x['tied_best_top1_rate']:.2f}% | {100*x['queried_selected_near_0_1_rate']:.2f}% | {x['queried_selected_regret_percent']['mean']:.6f} | {x['actual_unknown_count']}/{x['states']} |")
    lines += ['',f"D2查询内优势MSE={d['queried_advantage_MSE']:.9f}，非并列动作对符号准确率{100*d['queried_pair_sign_accuracy']:.4f}%。只在已查询菜单取预测最大值时，真实成本遗憾均值{d['queried_selected_regret_percent']['mean']:.6f}%；在全合法菜单实际执行时，选中未知动作{d['actual_unknown_count']}/{d['states']}。",
      '',f"这些未知动作相对g0仍有平均{d['unknown_selected_advantage_percent']['mean']:.6f}%正优势，但相对已查询最优动作的正遗憾（按Yref归一）均值为{d['unknown_positive_regret_Yref_percent']['mean']:.6f}%；预测优势相对真实优势的乐观偏差均值{d['unknown_optimism_bias_percentage_points']['mean']:.6f}个百分点。未知动作中‘预测正优势、真实负优势’占{100*d['unknown_predicted_positive_true_negative_rate']:.3f}%。因此不能给未知动作补零或负标签，也不能将其全部判成错误。",
      '', '## 状态分布与解释范围','',
      'R0为32场景上的教师实际状态；R1初始数据为32训练场景上的gB实际状态，后两轮为同32场景上的暖网络实际状态。不同表示进入不同前缀，标签排序难度与参考策略共同变化，不能直接把暖标签与round标签差异归因于某一模型组件。',
      '',f"R0平均完成覆盖比例{rz['state_distribution']['completed_fraction']['mean']:.4f}、前缀长度{rz['state_distribution']['prefix_length']['mean']:.3f}；R1初始分别为{ri['state_distribution']['completed_fraction']['mean']:.4f}、{ri['state_distribution']['prefix_length']['mean']:.3f}。每组完整分位数、覆盖四分位桶、模式及逐场景状态数保存在JSON。覆盖比例由保存场景几何和原代理覆盖公式重算，不调用导航；R0保存剩余任务数逐状态匹配。",
      '', '所有R1动作审计都发生在已用于训练的状态池，不是新开发状态上的动作泛化证据；独立dev32只保存完整路线结果，且已用于晋升和最终筛选。T0不能据此证明输入特征已经足够或模型可学，不能把既有开发场景改称未触碰测试。',
      '', '## 对T1的事前设计建议','',
      '固定单一教师、完整前缀和H语义；部署可用信息生成有界菜单，规则与模型共用。仅对实际查询动作监督。先在新场景按菜单内并列Top-1、近优率、遗憾与实际选中动作误差验收，再决定是否进入独立场景完整路线；不得以R0旧均值代替新场景教师收益。T0支持开展这一受限检验，尚未建立网络教师压缩能力。',
      '', '复现：`python -B -m rc3_teacher_distill.evidence_audit`；仅在本目录重写本报告和JSON，不写旧证据。此次文件级分析不是历史百万项验收的重跑。','']
    (OUT/'T0_DIAGNOSTIC.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print(json.dumps(dict(status='complete',inputs=len(a.inputs),checks=dict(a.checks),
          r0_states=rz['states'],initial_states=ri['states'],D2={k:d[k] for k in ('canonical_argmin_top1_rate','tied_best_top1_rate','queried_selected_regret_percent','actual_unknown_rate')},new_research_calls=0),ensure_ascii=True))


if __name__ == '__main__':
    main()
