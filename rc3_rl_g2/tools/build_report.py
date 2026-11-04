"""Render the G2 report from registered raw results, not copied summary numbers."""
from pathlib import Path
import json
import statistics as st
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'
LABEL={'ppo_final':'PPO 32,768','bc_frozen':'冻结 BC','bc_continued_2048':'继续 BC 2,048',
       'original_fixed3':'原 Fixed3','original_detour':'原 Detour','stop':'共同菜单 STOP',
       'detour_memory':'Detour＋尝试记忆','ils':'ILS','random':'随机策略'}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def group(rows,name):return [r for r in rows if r['method']==name]
def mean(rows,key):return st.mean(r[key] for r in rows)
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])
def s(v,n=4):return '—' if v is None else f'{v:.{n}f}'

def main():
    receipt=read(OUT/'training/ppo_receipt.json')
    if receipt.get('actual_actions')!=32768 or not receipt.get('completed'):raise ValueError('Incomplete PPO')
    summary=read(OUT/'final_evaluation/summary.json')
    sets={name:read(OUT/f'final_evaluation/{name}/results.json') for name in
          ['controls_fixed_work','final_fixed_work','final_soft_time_0400ms']}
    if any(r['status']!='complete' for rows in sets.values() for r in rows):
        raise ValueError('Manual report required: failure rows must not be silently omitted')
    fixed=sets['final_fixed_work'];timed=sets['final_soft_time_0400ms'];controls=sets['controls_fixed_work']
    pp=group(fixed,'ppo_final');iterations=[json.loads(line) for line in (OUT/'training/ppo_iterations.jsonl').read_text(encoding='utf-8').splitlines()]
    episodes=[json.loads(line) for line in (OUT/'training/ppo_episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    teacher=[read(OUT/f'teacher_{name}/summary.json') for name in ['initial','extra']]
    bc=read(OUT/'training/bc_initial_training.json');continued=read(OUT/'training/bc_continued_training.json')
    fit=read(OUT/'teacher_fit_diagnostic/summary.json')
    fit_rows=[]
    for name in ['bc_frozen','bc_continued_2048','ppo_final']:
        g=next(x for x in fit['groups'] if x['model']==name and x['split']=='combined' and x['level']=='root')
        m=g['metrics'];n=g['samples']
        fit_rows.append([LABEL[name],f"{round(n*m['full_greedy_stop']['sample_mean'])}/{n}",
          f"{round(n*m['full_legal_greedy_observed_near_best_hit']['sample_mean'])}/{n}",
          f"{round(n*m['evidence_greedy_near_best_hit']['sample_mean'])}/{n}",
          s(m['stop_probability']['sample_mean']),f"{100*m['unknown_probability_mass']['sample_mean']:.2f}"])
    fit_table=table(['模型','根 STOP','全合法近优命中','仅已查动作近优命中','平均 STOP 概率','未知动作概率质量 %'],fit_rows)
    regression=read(OUT/'final_regression.json');audit=read(OUT/'g2_audit.json')
    if not regression['passed'] or not audit['all_requested_checks_passed']:raise ValueError('Final verification failed')
    all_stop=sum(r['stop_at_root'] for r in pp)==24
    outcome=('在固定工作预算评测中，最终 PPO 在 24／24 个验证场景根节点 STOP，返回成本与 Fixed3 相同。'
             '本轮没有得到优于强规则的学习规划器，不支持直接扩大到三种子或 65,536 步。'
             if all_stop else '本轮模型已完成预登记预算；增量与继续门槛须结合下列配对结果和独立审阅判断。')
    text=['# G2：预算感知局部图 PPO 单种子小试\n',
      '**完成了准备补丁、实际训练、同初始化对照、逐场景评测和模型导出。** '+outcome,
      '\n## 1. 本轮交付与范围\n',
      '本轮是二维静态任务＋代理可达性的受控开发实验。正式网络为 65,602 参数的两轮稀疏局部图 actor–critic，'
      '从经验证的 Fixed3 初解开始，选择覆盖修复、扰动或 STOP；它不直接生成整条底盘折线。'
      '默认规划方法仍为原 Fixed3／Detour。v4、v5、v5.1 和 G01 保持冻结；真实标定、IK、动态任务和跨规模实验不在本轮范围。',
      '\n公开工作借鉴及实际采用边界见 [OPEN_SOURCE_NOTES.md](OPEN_SOURCE_NOTES.md)：PPO-Clip／GAE、CleanRL 的采集契约、'
      'NeuOpt／RL4CO 的多步改进视角，以及 MPNN 的关系编码。当前是本地新写实现，未直接复现这些项目的整套模型或成绩。',
      '\n## 2. 实际训练预算\n',
      table(['项目','实际值'],[
       ['初始教师',f"{teacher[0]['completed_cases']} 场景／{teacher[0]['samples']} 状态／{teacher[0]['query_count']} 查询"],
       ['追加教师',f"{teacher[1]['completed_cases']} 场景／{teacher[1]['samples']} 状态／{teacher[1]['query_count']} 查询"],
       ['初始 BC',f"{bc['updates']} 更新，{bc['elapsed_s']:.2f} s"],
       ['继续 BC',f"{continued['updates']} 额外更新，{continued['elapsed_s']:.2f} s"],
       ['正式 PPO',f"{receipt['actual_actions']:,} 已提交动作／{len(iterations)} rollout／{receipt['optimizer_updates']} 已提交优化更新"],
       ['PPO 已记录墙钟',f"{receipt['elapsed_s']:.2f} s（含固定检查点验证 {receipt['validation_s']:.2f} s，不含中断间隔／未知尾段）"],
       ['完整 episode 记录',f'{len(episodes):,}；最终未结束 worker 状态单独保存在 checkpoint'],
       ['设备','CPU；单模型线程，8 个单线程 NumPy/SciPy 环境 worker'],
       ['训练与验证','384 个新百桌训练场景池反复采样；24 个互斥新百桌验证场景（3 种模式各 8）']]),
      '\n训练场景是预登记的有限 384 场景池，不能称作每个回合都使用从未出现的新场景。'
      '两个 1,024 动作管线检查合计 2,048 个调试动作不计入正式 32,768。'
      '中断的教师 v1 已知 2,496 查询及无法完全核定的尾部工作另存，未用于正式初始化。'
      '教师求解、监督优化、环境交互和参数更新分别记账；同更新次数不等于同计算预算。',
      '\n**中断与恢复口径：** 本轮进程在 18,432 步后随会话中断退出，之后从同源、同协议且与日志一致的检查点恢复。'
      '最终 32,768 指这条恢复链上已保存并用于正式更新的 on-policy 动作。中断时可能另有 0～256 个未落盘动作，'
      '以及最多一轮未提交的优化尝试；这些尾段的精确工作量和时间未知，不能将全部历史物理调用声称为恰好 32,768。'
      '训练器 elapsed_s 为已记录分段运行时间，不含中断间隔和未知尾段；详细证据见 `outputs/interruption_receipt.json` 与独立审计。',
      '\nBC 与 PPO 从同一初始权重分叉。教师采用含工作费和终止事件的有限深度剩余回报，'
      '只在已查询 evidence mask 内监督；未知动作不作为负样本，有限深度标签不用于监督 horizon=16 的 critic。'
      'PPO 固定 gamma=1、GAE λ=0.95、clip=0.2、Adam 1e−4、4 epoch、minibatch64，'
      '全部超参数、数据种子和最终 32,768 检查点在查看验证前固定。',
      '\n## 3. 固定工作条件的全部方法\n',
      '目标成本为 J=L+0.75K；不是机器人真实完成时间。共享搜索回报为 '
      '`100(J_anchor−J_return)/J_anchor − 0.01×work/10000`。共享环境使用 2,000,000 工作软预算和至多 16 次决策；'
      '原版方法保留自己的算法，不伪造可比的 G2 work 或 reward。原 Fixed3 不构建首菜单。']
    method_rows=[]
    for name in LABEL:
        gg=group(fixed,name);ret=mean(gg,'total_reward') if gg[0]['total_reward'] is not None else None
        work=mean(gg,'work_used') if gg[0]['work_used'] is not None else None
        method_rows.append([LABEL[name],s(mean(gg,'quality_gain_percent')),s(ret),
            f"{1000*st.median(r['complete_s'] for r in gg):.2f}",s(work,0),
            f"{mean(gg,'proposal_calls'):.2f}",f"{sum(r['stop_at_root'] for r in gg)}/24" if gg[0]['shared_action_protocol'] else '—'])
    text.append('\n'+table(['方法','均值降本 %','均值回报','完整时间中位 ms','均值工作量','均值提案数','根 STOP'],method_rows))
    text+=['\n## 4. 三个预登记检查点\n']
    stages=[]
    for steps,updates in [(8192,512),(16384,1024),(32768,2048)]:
        rr=read(OUT/f'validation/ppo_{steps}/results.json');cc=group(controls,f'bc_continued_{updates}')
        if len(rr)!=24 or any(r['status']!='complete' for r in rr):raise ValueError('Incomplete checkpoint validation')
        stages.append([f'{steps:,}',s(mean(rr,'total_reward')),f"{sum(r['stop_at_root'] for r in rr)}/24",
                       s(mean(group(controls,'bc_frozen'),'total_reward')),s(mean(cc,'total_reward')),updates])
    text+=[table(['PPO 动作','PPO 回报','PPO 根 STOP','冻结 BC 回报','继续 BC 回报','继续 BC 更新'],stages),
      '\n三个检查点是同一训练过程与同一验证集的重复观察，不能计为三个训练种子；'
      '继续 BC 的三档更新数也不是与 PPO 等算力。阶段比较只使用质量／回报／工作量，部署耗时以另行执行的最终比较为准。',
      '\n## 5. 最终模型的场景配对差\n']
    pairs=summary['datasets']['final_fixed_work']['ppo_vs_methods'];pairrows=[]
    for name in ['bc_frozen','bc_continued_2048','detour_memory','ils','stop']:
        pair=next(p for p in pairs if p['method_b']==name);metric=pair['metrics']['total_reward']
        lo,hi=metric['improvement_ci95_percentile']
        pairrows.append([LABEL[name],s(metric['a_improvement_mean']),f'[{lo:.4f}, {hi:.4f}]',
                         f"{metric['wins']}／{metric['ties']}／{metric['losses']}",
                         s(pair['metrics']['J']['relative_J_reduction_percent_mean'])])
    text+=[table(['PPO 对照','回报差均值','配对 95% 区间','胜／平／负','PPO 相对降本 %'],pairrows),
      '\n正值有利于 PPO。区间按 24 场景配对自助重采样 10,000 次，仅是单训练种子、重复使用验证集条件下的开发统计，'
      '不是跨种子稳定性结论，也没有做多重比较校正。分模式结果及全部逐场景结果保留在 `outputs/final_evaluation/`。',
      '\n冻结 BC 在这批验证上已全部根 STOP，PPO 没有突破该初始化行为。'
      'PPO 相对共享规则更高的回报来自不搜索、少付工作罚，简单 STOP 获得完全相同的结果；'
      '这不构成强化学习增量。原 Fixed3 还省去了共同菜单和网络开销。',
      '\n## 6. 400 ms 完整规划软预算\n']
    time_rows=[]
    for name in LABEL:
        gg=group(timed,name)
        time_rows.append([LABEL[name],s(mean(gg,'quality_gain_percent')),
            f"{1000*st.median(r['complete_s'] for r in gg):.2f}",
            f"{sum(r['time_overrun_s']>0 for r in gg)}/24",f"{1000*max(r['time_overrun_s'] for r in gg):.2f}"])
    text+=[table(['方法','均值降本 %','完整时间中位 ms','超时场景','最大超时 ms'],time_rows),
      '\n完整时间含场景预处理、初解、菜单、策略、修复、认证和最终验证；不含场景随机生成、模型载入及 JSON 写盘。'
      '模型载入单列测量，未清操作系统磁盘缓存。软截止允许正在执行的计算和最终验证超时，不能声称达到 400 ms 硬实时。'
      '当前模型观察不含剩余墙钟秒数：预算感知仅指工作与决策步数，时间实验由外层 deadline 执行。',
      '\n训练 episode 的 `complete_s` 包含等待采集器／优化器／阶段验证的墙钟时间，禁止用于部署耗时比较。'
      '训练准备缓存只复用不可变问题；所有最终比较都使用 fresh adapter。',
      '\n## 7. 正确性、复核和负结果边界\n',
      f"最终 3 组比较共有 {sum(len(x) for x in sets.values())} 条运行记录，另有 3×24 个阶段检查点记录。"
      '这些重复方法／检查点仍只对应 24 个独立验证场景。最终保存方案的离线账本与摘要核对、实际重放样本、'
      'checkpoint 恢复和回归测试需分开计数，详见 `G2_AUDIT.md` 与 `outputs/g2_audit.json`。',
      f"\n本轮实际执行当前 G2 回归 {regression['tests_run']}／{regression['tests_run']} 项通过；"
      f"独立审阅实际重放 {len(audit['actual_replay']['saved_plan_replays'])} 份预登记方案，"
      f"并对同一个保存 worker 做两次独立恢复，各采样 {len(audit['actual_replay']['sampled_actions'])} 个真实动作后自然终止，结果一致。"
      '全部保存记录的离线复算不冒称全部重新执行。',
      '\n状态恢复拒绝 NaN/Inf、非法计数及路线／菜单不一致并保持原子性；'
      '外部终止事件恰好一次并入最后真实动作，零动作仅记指标；采集长度截断继续保留环境；'
      '共享尝试记忆在菜单截断前屏蔽确定性无变化／失败动作并补位；'
      '重复路线认证可缓存，最终方案仍独立验证。已有 F∩A 共同约束继续通过原适配层使用。',
      '\n若出现几乎全 STOP，它可降低搜索工作，却不证明学会了高质量多步决策。'
      '训练教师的部分有限树存在更好分支，因此需要把奖励下的探索、监督拟合与跨场景泛化分开诊断。'
      '本轮没有匹配的单调 PPO 消融，不能归因非单调机制；没有多架构正式训练，不能断言图网络普遍无效。',
      '\n### 已有教师状态上的拟合诊断\n',fit_table,
      f"\n上表来自现有 96 个教师根状态，每根等权；完整诊断为 3 模型×768 状态，共 {fit['rows']} 条推理记录，新增环境查询为 0。"
      '这不是新的泛化测试。近优仅指已查有限深度分支；全合法选择了未知动作时不计已观察近优命中，'
      '但不能据此认定该未知动作质量差。结果支持先改进初始化和动作排序，再研究 PPO 探索；'
      '不能把增加轮数作为已得到验证的修复。逐状态 CSV 与概率记录见 `outputs/teacher_fit_diagnostic/`。',
      '\n## 8. 使用与后续判断\n',
      '最终研究权重为 `outputs/deployment/policy_ppo.pt`，另附冻结与继续 BC 的轻量参数包。'
      '它们保留研究和复现价值；生产默认入口保持原 Fixed3／Detour。运行说明见 [README_G2.md](README_G2.md)。'
      '图表脚本、PDF／SVG／PNG 与输入哈希位于 `figures/`。',
      '\n本轮不自动追加 65,536 步或三种子。若后续继续，应先在独立开发场景检查教师动作排序、未知动作概率和'
      'STOP／继续搜索的回报差，再预登记新的探索或监督方案；不能用当前验证集反复改奖励后继续当作未见数据。'
      '最终独立的主张判定与下一步路由见结果审阅文件。\n']
    (ROOT/'REPORT_G2.md').write_text('\n\n'.join(text),encoding='utf-8')
    print(json.dumps(dict(report=str(ROOT/'REPORT_G2.md'),actual_actions=receipt['actual_actions'],
                         final_evaluations=sum(map(len,sets.values())),ppo_root_stop=sum(r['stop_at_root'] for r in pp))))

if __name__=='__main__':main()
