"""Report saved S0 evidence only; no fresh model, navigation or trajectory calls."""
from collections import defaultdict
import json
import numpy as np
from .register import OUT,read,write,digest


def main():
    status=read(OUT/'STATUS.json');resources=read(OUT/'resources.json');s=resources['stats']
    complete=status.get('S0')=='complete_pass'
    rows=[json.loads(x) for x in (OUT/'episodes.jsonl').read_text(encoding='utf-8').splitlines()] if (OUT/'episodes.jsonl').exists() else []
    lines=['# S0：大场景训练—局部任务推理环境一致性报告','',
        '日期：2026-11-06。固定研究目标：大规模复杂父场景中的深度强化学习训练，冻结策略直接规划不同规模局部任务，输出完整约束可验证的路径。监督只作初始化/对照。',
        '',f"S0状态：**{status.get('S0')}**。S1/S2正式优化器更新0；S2/S3未启动。本报告不提供已训练DRL性能、部署优势或真实关节轨迹已验证的结论。",
        '', '## 任务提取与约束','',
        '4×3、6×6、8×8指100桌父地图中的任务窗口；其余桌子、障碍、通道和导航图仍保留，路径可以离开任务窗口。不是将地图物理切小，也不删除窗口外仍能服务目标任务的合法候选。',
        '有效覆盖为逐臂(F&A)后再对L/R取并集。全局task/candidate ID保持，局部节点1..M另存映射。入口与出口可不同，Dlocal[0,j]为entry→j，Dlocal[j,0]为j→exit；内部服务距离保留原对称D。',
        '父输入scene/F/A/D和导航前驱/节点均保存，支持仅从保存数据完整核对；旧H代码字节不变，在端点成本适配上执行。开路线由新端点验证器检查，未借旧闭环验证器冒充认证。',
        '', '## MDP与训练接口','',
        '状态包括全局上下文、局部有效F/C、剩余任务、当前位置、明确出口、完整有序前缀。动作仅能新增覆盖且从当前可达、可到出口；全覆盖后自动到出口，没有早停STOP。',
        '逐步奖励=-导航移动−0.75，最后加入出口代价及Jraw−JH修正，因此总回报=-JH；空任务直接entry→exit且K=0。',
        '策略使用全局表征、局部覆盖二部编码、真实导航及前缀GRU，无内置规则logit先验。缓存同时绑定问题/任务/F&A、模型实例与权重版本；不同checkpoint不能混用。',
        '', '## 有界一致性检查','',
        f"实际父场景{s['parent_scene_generation']}、全局导航准备{s['navigation_preparations']}、窗口{s['cropextractions']}；普通随机/规则/未训练神经轨迹{s['ordinary_episodes']}，额外多起点梯度探针{s['gradient_episodes']}。",
        f"终局请求{s['terminal_requests']}、真实尝试{s['terminal_attempts']}、完成{s['terminal_completed']}、失败{s['terminal_failed']}、精确重放命中{s['terminal_cache_hits']}；完整H调用{s['full_H_calls']}、导出{s['exports']}、验证完成{s['validation_completed']}。",
        '表中成本只描述未训练环境探针，时间包含一致性重放和验收，不是正式冻结DRL规划速度比较。','',
        '| 无训练策略 | 普通轨迹数 | 可行率 | 平均J | 平均K |','|---|---:|---:|---:|---:|']
    for method in ('random','rule','neural'):
        subset=[r for r in rows if r['method']==method and not r['gradient_probe']]
        if subset:lines.append(f"| {method} | {len(subset)} | {100*np.mean([r['valid'] for r in subset]):.2f}% | {np.mean([r['J'] for r in subset]):.6f} | {np.mean([r['K'] for r in subset]):.3f} |")
    lines+=['','## 梯度与完整轨迹','']
    if (OUT/'gradient_probe.json').exists():
        g=read(OUT/'gradient_probe.json')
        lines+=[f"策略梯度探针有限={g['all_finite']}，非零={g['nonzero']}，反向传播{s['backward_calls']}次；参数前后哈希一致={g['tensor_hash_before']==g['tensor_hash_after']}，优化器更新={g['optimizer_updates']}。",
            '同一实例三条路线用共享平均回报基线；强制首动作不计策略logprob。不同覆盖路线不被假定等价，未声称多起点必然降方差或提高质量。']
    else:lines+=['实际梯度探针尚未合格完成，保留部分结果，不认定S0通过。']
    lines+=['每份plans/记录含全局停靠位姿、访问顺序、导航折线/腿长、任务服务关系和逐臂分配。验证包含端点、覆盖、同臂F&A、代理可达、膨胀障碍碰撞、长度及目标一致性。',
        'arm_joint_trajectories为null，physical_joint_trajectory_certified=false。真实IK、机械臂扫掠碰撞、关节轨迹约束仍是同一研究目标内未满足的验证项，不能用代理/表一致性代替。',
        '', '## 协议与研究状态','',
        f"S0协议SHA256 `{digest(OUT/'S0_protocol.json')}`；S2/S3设计协议SHA256 `{digest(OUT/'S2_S3_protocol.json')}`。",
        '未来设计：96个10×10训练布局、每布局8个任务抽样；50%完整百桌episode+50%父图内局部episode，真正策略梯度训练为核心。监督初始化/继续监督仅为辅助与对照。',
        'S3已声明108局部任务案例，6个已训练布局与12个未训练布局两组分开、每组各尺寸和端点单独报告。统计聚类单位是父布局，不将108子场景冒充108独立环境。原60保留测试仍封存。',
        '正式运行前需新执行任务绑定训练runner/基线实现哈希、通过端点适配Fixed3/Detour一致性检查并保存执行收据。S0不自动启动S1/S2/S3，不能以本轮梯度流检查声称强化学习已有独立增量。',
        '', f"研究运行墙钟{resources['elapsed_wall_s']:.3f}秒。合成环境与神经契约单测的forward/backward单独记录在unit_receipt.json和policy_unit_receipt.json，不并入真实场景轨迹账。",
        '', '旧研究证据隔离在原提交；默认Fixed3／Detour保持。文献真实性和适用边界见LITERATURE_VERIFICATION.md，原始来源见literature_sources.json。','']
    if not complete:lines+=['**S0未完整完成：部分探针不替代全部事前检查，不启动正式训练。**','']
    if not complete and ('error' in status or 'message' in status):
        lines+=['失败原因：'+str(status.get('error',''))+' — '+str(status.get('message','')),'']
    (OUT/'REPORT_S0.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    write(OUT/'claims.json',dict(S0_consistency_passed=complete,DRL_research_goal_fixed=True,
        trained_DRL_performance_supported=False,independent_RL_gain_supported=False,
        S1_started=False,S2_started=False,S3_started=False,physical_joint_trajectory_certified=False))
    print(json.dumps(dict(report='rc3_global_local/REPORT_S0.md',status=status,stats=s)))


if __name__=='__main__':main()
