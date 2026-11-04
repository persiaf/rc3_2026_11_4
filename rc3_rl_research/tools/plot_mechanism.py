"""Reproduce the first numbered strict G0 witness; never rank cases by gain.

Reads frozen G0 artifacts and runs their deterministic replay before plotting.
Only this tool's outputs and MECHANISM_CASE.md are written; no research runtime,
experiment record, old paper figure or model file is changed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from rc3_rl_research.runtime_rl import replay_file


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs'/'mechanism'
CASE = ROOT/'outputs'/'g0'/'case_01'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_close(actual, expected, name):
    if not math.isclose(actual, expected, abs_tol=1e-8, rel_tol=1e-9):
        raise ValueError(f'{name} differs: {actual} != {expected}')


def extract(data, summary):
    state, records = data['state'], data['state']['transitions']
    anchor = state['anchor_cost']
    rows = [dict(stage='anchor', x=0., current_cost=anchor, best_cost=anchor,
                 step_quality=0., step_work_penalty=0., step_return=0.,
                 cumulative_quality=0., cumulative_work_penalty=0., cumulative_return=0.,
                 work_units=0, menu_work_units=0, proposal_work_units=0, cumulative_work=0,
                 current_stops=len(state['anchor'])-2, best_stops=len(state['anchor'])-2,
                 action_kind='anchor', action_id=None, removed_nodes=[], added_nodes=[],
                 pointers=['/state/anchor_cost', '/state/anchor'])]
    actions = []
    quality = penalty = total = work = 0.
    for i, transition in enumerate(records):
        info, act = transition['info'], transition['action']
        before, after = transition['observation']['current'], transition['next_observation']['current']
        quality += info['quality_reward']; penalty += info['work_penalty']
        total += transition['reward']; work += info['work_units']
        assert_close(quality-penalty, total, f'undiscounted return after step {i+1}')
        removed, added = sorted(set(before)-set(after)), sorted(set(after)-set(before))
        prefix = f'/state/transitions/{i}'
        rows.append(dict(stage=f'step_{i+1}', x=float(i+1), current_cost=info['current_cost'],
                         best_cost=info['best_cost'], step_quality=info['quality_reward'],
                         step_work_penalty=info['work_penalty'], step_return=transition['reward'],
                         cumulative_quality=quality, cumulative_work_penalty=penalty,
                         cumulative_return=total, work_units=info['work_units'],
                         menu_work_units=info['menu_work_units'], proposal_work_units=info['proposal_work_units'],
                         cumulative_work=int(work), current_stops=len(after)-2,
                         best_stops=len(transition['next_observation']['best'])-2,
                         action_kind=act['kind'], action_id=transition['action_id'],
                         removed_nodes=removed, added_nodes=added,
                         pointers=[prefix+'/info/current_cost', prefix+'/info/best_cost',
                                   prefix+'/info/quality_reward', prefix+'/info/work_penalty', prefix+'/reward']))
        actions.append(dict(step=i+1, action=act, action_id=transition['action_id'],
                            accepted=info['accepted'], proposal_valid=info['proposal_valid'],
                            before_window=before[act['start']:act['end']],
                            removed_nodes=removed, added_nodes=added,
                            removed_candidate_ids=[n-1 for n in removed],
                            added_candidate_ids=[n-1 for n in added],
                            before_route=before, after_route=after,
                            required_task_ids=info['proposal']['diagnostics']['required_tasks'],
                            json_pointers=dict(action=prefix+'/action',
                                before_route=prefix+'/observation/current',
                                after_route=prefix+'/next_observation/current',
                                required_tasks=prefix+'/info/proposal/diagnostics/required_tasks')))
    # The diagnostic depth is an explicit runtime termination. Its menu was
    # already built by the preceding step, and is charged only by this event.
    if len(state['events']) != 1:
        raise ValueError('expected exactly one independently represented finalization event')
    event = state['events'][0]
    if event['reason'] != 'diagnostic_depth' or event['after_steps'] != len(records):
        raise ValueError('unexpected finalization semantics')
    quality += event['quality_reward']; penalty += event['work_penalty']
    total += event['reward']; work += event['menu_work_units']
    rows.append(dict(stage='finalize', x=float(len(records))+.7,
                     current_cost=state['current_cost'], best_cost=state['returned_certificate']['cost'],
                     step_quality=event['quality_reward'], step_work_penalty=event['work_penalty'],
                     step_return=event['reward'], cumulative_quality=quality,
                     cumulative_work_penalty=penalty, cumulative_return=total,
                     work_units=event['menu_work_units'], menu_work_units=event['menu_work_units'],
                     proposal_work_units=0, cumulative_work=int(work),
                     current_stops=len(state['current'])-2,
                     best_stops=len(state['returned_certificate']['route'])-2,
                     action_kind='diagnostic_finalization', action_id=None,
                     removed_nodes=[], added_nodes=[],
                     pointers=['/state/events/0', '/state/finalization', '/state/returned_certificate/cost']))
    for actual, expected, name in (
        (quality, state['total_quality_reward'], 'quality'),
        (penalty, state['total_work_penalty'], 'penalty'),
        (total, state['total_reward'], 'return'),
        (work, state['work_used'], 'work'),
        (quality-penalty, total, 'Q-P'),
        (quality, 100*(anchor-state['returned_certificate']['cost'])/anchor, 'quality telescope')):
        assert_close(actual, expected, name)
    if state['config']['potential_weight'] != 0 or state['total_shaping_reward'] != 0:
        raise ValueError('this fixed figure requires the preregistered unshaped reward')
    if records[0]['info']['current_cost'] <= anchor:
        raise ValueError('fixed selected witness no longer has a worsening first action')
    if rows[-1]['best_cost'] >= summary['root_one_step_best']:
        raise ValueError('fixed selected witness no longer beats the queried one-step result')
    return rows, actions


def draw(rows):
    plt.rcParams.update({'font.family':'serif', 'font.serif':['Times New Roman','DejaVu Serif'],
        'font.size':10, 'axes.labelsize':10, 'xtick.labelsize':9, 'ytick.labelsize':9,
        'legend.fontsize':9, 'axes.spines.top':False, 'axes.spines.right':False,
        'mathtext.fontset':'stix', 'pdf.fonttype':42, 'ps.fonttype':42,
        'svg.fonttype':'none', 'savefig.dpi':300})
    blue, green, orange = '#0072B2', '#009E73', '#D55E00'
    x = [r['x'] for r in rows]
    fig, axes = plt.subplots(1,2,figsize=(7.5,3.25))
    left, right = axes
    left.plot(x, [r['current_cost'] for r in rows], color=blue, marker='o',
              ms=4.5, lw=1.5, label='Current solution')
    left.plot(x, [r['best_cost'] for r in rows], color=green, marker='s',
              mfc='white', ms=4.5, lw=1.5, ls='--', label='Best certified solution')
    left.set_ylabel(r'Cost $J=L+0.75K$ (equivalent m)')
    left.ticklabel_format(axis='y', style='plain', useOffset=False)
    left.yaxis.set_major_locator(MaxNLocator(5))
    lo, hi = min(r['best_cost'] for r in rows), max(r['current_cost'] for r in rows)
    left.set_ylim(lo-.25, hi+.65)
    left.legend(loc='upper right',frameon=False)
    left.annotate('Accepted worsening', xy=(x[1], rows[1]['current_cost']),
                  xytext=(.08, hi+.21), fontsize=8.5,
                  arrowprops=dict(arrowstyle='-',color='#555555',lw=.7))
    right.plot(x, [r['cumulative_quality'] for r in rows], color=green, marker='s',
               mfc='white', ms=4.5, lw=1.5, label=r'Quality reward $Q$')
    right.plot(x, [r['cumulative_work_penalty'] for r in rows], color=orange, marker='^',
               mfc='white', ms=4.5, lw=1.5, ls='--', label=r'Work penalty $P\geq0$')
    right.plot(x, [r['cumulative_return'] for r in rows], color=blue, marker='o',
               ms=4.5, lw=1.5, ls='-.', label=r'Return $Q-P$')
    right.axhline(0.,color='#b0b0b0',lw=.7,zorder=0)
    right.set_ylabel('Cumulative reward component')
    right.set_ylim(-.43,.94)
    right.legend(loc='upper left',frameon=False)
    terminal_charge=rows[-1]['step_work_penalty']
    right.annotate(f'Final menu charge\n{terminal_charge:.6f}',
                   xy=(x[-1],rows[-1]['cumulative_return']), xytext=(1.65,-.405),
                   fontsize=8.5, ha='left', va='bottom',
                   arrowprops=dict(arrowstyle='->',color='#555555',lw=.7))
    for letter, ax in zip(('a','b'), axes):
        ax.set_xlim(-.18,4.)
        ax.set_xticks(x, ['Anchor','1','2','3','End'])
        ax.set_xlabel('Search step / finalization')
        ax.axvline(3.35,color='#aaaaaa',ls=':',lw=.8,zorder=0)
        ax.text(-.18,1.01,f'({letter})',transform=ax.transAxes, fontsize=10,va='bottom')
        ax.tick_params(direction='out',length=3,width=.7)
    fig.subplots_adjust(left=.09,right=.985,bottom=.18,top=.94,wspace=.37)
    for extension in ('pdf','svg','png'):
        fig.savefig(OUT/f'case01_mechanism.{extension}',dpi=300)
    plt.close(fig)


def document(rows, actions, data, summary, sources):
    state=data['state']; case=summary['case']; end=rows[-1]
    stop=next(r for r in summary['root_action_summaries'] if r['action']['kind']=='stop')['descendant']
    text=[
        '# G0 固定机制案例：case_01', '',
        '本页从保存的 JSON 生成；制图前已运行 `runtime_rl.replay_file`。该案例是开发集有限深度 oracle 的机制展示，未训练或调用 PPO。', '',
        '## 选择规则与证据范围', '',
        f"按 case 编号递增选择第一个 `worse_first_beats_one_step=true` 的场景：case_00 为 false，case_01 为 true。没有按降本幅度挑选最优案例。本例 seed={case['seed']}，{case['nx']}×{case['ny']} 张桌、{summary['task_count']} 个任务、{state['transitions'][0]['observation']['candidate_count']} 个候选配置，保存动作索引为 `{summary['selected_path']}`。",
        '',
        f"这里的“严格见证”仅表示：所选序列第一步被接受且 current 变差，其深度 3 终值 {end['best_cost']:.9f} 小于已查询根菜单的一步最好值 {summary['root_one_step_best']:.9f}，差 {summary['root_one_step_best']-end['best_cost']:.9f}。它不表示必须先变差：本例 `best_monotone_first_depth3_cost` 同为 {summary['best_monotone_first_depth3_cost']:.9f}，`worse_first_beats_all_monotone_first=false`；该字段只限制第一步，并非完整单调搜索树证明。", '',
        '## 图与图注', '',
        '![current/best 与累计回报](outputs/mechanism/case01_mechanism.png)', '',
        '**图注：** 固定 G0 开发案例 case_01（深度 3、每层最多 8 个分支，含 STOP）的保存序列。左图为当前路线与最好已认证路线的加权成本；右图为累计质量奖励 Q、非负计算工作惩罚 P 和总回报 Q−P。Anchor 是认证 Fixed3 初解。End 是诊断深度结束后的最终验证与待结算菜单费用，不是第四次修复；虚线区分这两类阶段。目标 J=L+0.75K 是等效路程成本，不能解释为机器人任务执行时间。该图证明此查询范围内存在先变差后改善的有效序列，不证明变差的必要性、PPO 优势或部署速度。', '',
        '图输出：`outputs/mechanism/case01_mechanism.pdf`、同名 SVG 与 300 dpi PNG。曲线数据、完整精度表格和来源哈希分别在 `mechanism_data.json`、`trajectory.csv`、`source_manifest.json`。', '',
        '## 成本与累计回报', '',
        '| 阶段 | current J | best / 返回 J | 累计 Q | 累计 P | 累计 Q−P | 累计工作单元 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        text.append(f"| {r['stage']} | {r['current_cost']:.9f} | {r['best_cost']:.9f} | {r['cumulative_quality']:.9f} | {r['cumulative_work_penalty']:.9f} | {r['cumulative_return']:.9f} | {r['cumulative_work']:,} |")
    text.extend(['',
        f"采用原记录中的 epsilon={state['config']['epsilon']}、lambda_work={state['config']['lambda_work']}、work_scale={state['config']['work_scale']:g}、potential_weight=0。Q=100(J_anchor−J_best)/J_anchor；P=lambda_work×累计工作/work_scale。总质量改善为 {end['cumulative_quality']:.9f}%，但总回报为 {end['cumulative_return']:.9f}。同一根状态直接 STOP 的回报为 {stop['total_reward']:.9f}，因此该路径没有优于 STOP 的含工作惩罚回报；不能把质量见证写成已解决奖励权衡。", '',
        f"第三次动作结束后 Q−P={rows[-2]['cumulative_return']:.9f}。随后单独结算已经构建的菜单：{end['menu_work_units']:,} 工作单元，P 增加 {end['step_work_penalty']:.9f}，最终回报降至 {end['cumulative_return']:.9f}。本次终验成功，没有质量回退校正。表内累计工作仅是所选序列的记账，不是发现它所花的整棵 oracle 搜索成本。", '',
        '## 动作语义与服务点变化', '',
        '下列 node 为导航节点编号（depot=0，候选 id=node−1），route 切片采用 Python 半开索引，且 route[0] 是起点。节点集合差用于描述服务点增删；完整访问顺序保存在 JSON，不能仅由集合差恢复。', '',
        '| 步骤 | 动作 / route 切片 | 移除 node | 新增 node | 移除候选 id | 新增候选 id |',
        '|---|---|---|---|---|---|'])
    for a in actions:
        act=a['action']
        text.append(f"| {a['step']} | {act['kind']} / [{act['start']}:{act['end']}] | {a['removed_nodes']} | {a['added_nodes']} | {a['removed_candidate_ids']} | {a['added_candidate_ids']} |")
    text.extend(['',
        f"第一步 replace 在本次候选池禁用 node {actions[0]['action']['forbidden_node']}，新增 node {actions[0]['added_nodes']}；不修改 F/A。current 上升而 best 保持 anchor。第二、三步 improve 使用相同覆盖修复接口，在完整保留路线决定的必要任务上求解；第三步又移除了 node {sorted(set(actions[0]['added_nodes']) & set(actions[2]['removed_nodes']))}。这些动作是在虚拟规划搜索中发生，机器人不会执行中间较差路线。所有被接受的路线均经完整认证。", '',
        '## 原始 JSON 指针', '',
        '以下指针均相对于 `outputs/g0/case_01/selected_replay.json`，完整机器可读映射在 `mechanism_data.json`。', '',
        '| 数据 | JSON 指针 |', '|---|---|',
        '| 初解成本与路线 | `/state/anchor_cost`；`/state/anchor` |',
        '| 第 i 次动作（i=0,1,2） | `/state/transitions/i/action`；`/state/transitions/i/action_id` |',
        '| 动作前后完整 current 路线 | `/state/transitions/i/observation/current`；`/state/transitions/i/next_observation/current` |',
        '| 动作后的 current / best 成本 | `/state/transitions/i/info/current_cost`；`/state/transitions/i/info/best_cost` |',
        '| 单步质量、工作惩罚与回报 | `/state/transitions/i/info/quality_reward`；`/state/transitions/i/info/work_penalty`；`/state/transitions/i/reward` |',
        '| 必要任务 | `/state/transitions/i/info/proposal/diagnostics/required_tasks` |',
        '| 终止待结算菜单费 | `/state/events/0/menu_work_units`；`/state/events/0/work_penalty`；`/state/events/0/reward` |',
        '| 最终回报与完整方案 | `/state/total_reward`；`/state/returned_certificate`；`/plan` |',
        '| 严格见证的比较口径 | `case_01/summary.json` 中 `/root_one_step_best`、`/best_monotone_first_depth3_cost`、`/selected_is_strict_witness` |', '',
        '复现命令：`python -B -m rc3_rl_research.tools.plot_mechanism`。脚本再次重放原记录、核对奖励望远镜恒等式、终止费用和输入 SHA-256 后生成输出。它只重放已有序列，既不重新搜索整个树，也不训练或测量模型性能。', '',
        '## 来源文件哈希', ''])
    text += [f"- `{p}`：`{h}`" for p,h in sources.items()]
    (ROOT/'MECHANISM_CASE.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    earlier=read(ROOT/'outputs'/'g0'/'case_00'/'summary.json')
    summary=read(CASE/'summary.json')
    if earlier['worse_first_beats_one_step'] or not summary['worse_first_beats_one_step']:
        raise ValueError('predeclared smallest-numbered strict witness is no longer case_01')
    paths=[ROOT/'outputs'/'g0'/'protocol.json', ROOT/'outputs'/'g0'/'case_00'/'summary.json',
           CASE/'summary.json', CASE/'selected_replay.json']
    sources={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths}
    receipt=replay_file(CASE/'selected_replay.json')
    write(OUT/'replay_receipt.json',receipt)
    data=read(CASE/'selected_replay.json')
    rows,actions=extract(data,summary)
    write(OUT/'mechanism_data.json',dict(selection_rule='lowest case_id with worse_first_beats_one_step=true',
        case_id=1,source_manifest=sources,rows=rows,actions=actions,summary=summary,
        note='A finite development oracle witness, not a trained PPO result.'))
    write(OUT/'source_manifest.json',sources)
    with (OUT/'trajectory.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        fields=[k for k in rows[0] if k not in ('pointers','removed_nodes','added_nodes')]
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    table=['\\begin{table}[t]', '\\centering',
           '\\caption{Saved G0 case 01 trajectory; Q is quality reward, P is work penalty.}',
           '\\label{tab:g0-case01}', '\\begin{tabular}{lrrrrr}', '\\toprule',
           'Stage & Current $J$ & Best $J$ & $Q$ & $P$ & $Q-P$ \\\\', '\\midrule']
    for r in rows:
        stage=r['stage'].replace('_', '\\_')
        table.append(f"{stage} & {r['current_cost']:.6f} & {r['best_cost']:.6f} & "
                     f"{r['cumulative_quality']:.6f} & {r['cumulative_work_penalty']:.6f} & "
                     f"{r['cumulative_return']:.6f} \\\\")
    table.extend(['\\bottomrule','\\end{tabular}','\\end{table}'])
    (OUT/'TABLE_mechanism.tex').write_text('\n'.join(table)+'\n',encoding='utf-8')
    draw(rows)
    document(rows,actions,data,summary,sources)
    caption=('A fixed depth-3 G0 oracle witness (case 01). Current and best certified costs '
             '(left), and cumulative quality reward Q, nonnegative work penalty P and return Q-P '
             '(right). End separately charges the pending menu; it is not another repair action. '
             'The first accepted move worsens current while preserving best. This sequence does '
             'not establish the necessity of worsening, PPO superiority, or a runtime advantage.')
    (OUT/'latex_includes.tex').write_text('\\begin{figure}[t]\n\\centering\n'
        '\\includegraphics[width=0.95\\textwidth]{outputs/mechanism/case01_mechanism.pdf}\n'
        '\\caption{'+caption+'}\n\\label{fig:g0-case01-mechanism}\n\\end{figure}\n',encoding='utf-8')
    for p in paths:
        if sha(p)!=sources[str(p.relative_to(ROOT)).replace('\\','/')]:
            raise ValueError('an input changed during figure generation')
    write(OUT/'generation_receipt.json',dict(replay_valid=True,accounting_verified=True,
        source_files_unchanged=True,source_manifest=sources,
        selected_path=summary['selected_path'],current_sequence=[r['current_cost'] for r in rows],
        final_quality=rows[-1]['cumulative_quality'],final_penalty=rows[-1]['cumulative_work_penalty'],
        final_return=rows[-1]['cumulative_return'],terminal_pending_menu_penalty=rows[-1]['step_work_penalty'],
        outputs={p.name:dict(bytes=p.stat().st_size,sha256=sha(p))
                 for p in OUT.iterdir() if p.is_file() and p.suffix in ('.pdf','.svg','.png','.csv')}))
    print(json.dumps(dict(case_id=1,replay_valid=True,final_return=rows[-1]['cumulative_return'],
                         output=str(OUT)),ensure_ascii=False))


if __name__=='__main__':
    main()
