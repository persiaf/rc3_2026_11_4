# 当前系统：实际代码与决策权限

基准 `d0b021db8a7f74f50622c0feb68d5dda23ed6738`。本轮只读源码和训练/暴露开发证据；未调用这里列出的生成、策略、规划或H入口。R08是提案，不改写旧活动协议、模型或停止状态。

```mermaid
flowchart LR
 A[已保存scene/逐臂F和A/导航NPZ与receipt] --> B[LocalProblem:任务窗口、候选、端点D]
 B --> C[SL编码:父布局/任务/候选/前缀]
 C --> D[合法候选评分与选择]
 D --> E[自动服务全部新覆盖任务]
 E -->|尚未完成| D
 E --> F[完整raw配置序列]
 F --> G[共同H:去冗余/2-opt/单配置替换]
 G --> H[完整路线展开/按顺序服务分配/独立验证]
```

## 模块和接口

|代码文件/入口|真实输入→输出及数学语义|连接/复用与重构判断|
|---|---|---|
|`rc3_fast/scenes.py:make_complex_scene`；`rc3_baseline/baseline.py:make_scene`|10×10、layout seed、mixed/blocked→桌体、任务、每桌8个有朝向/固定高度的候选配置、通道锚点。复杂版本改桌体大小/位置并可加水平通道障碍|保留生成规则；R08-A不执行。将来新父布局必须新授权，不能生成失败后换seed|
|`rc3_constructive/problem.py:resample_tasks`|已有几何+task seed→每桌0—3任务及位置；不是改变物理父布局|复用；本轮读已保存scene，不重抽样|
|`rc3_global_local_trainer/data.py:TrainingSchedule.batch,SceneProvider.get`|登记父/任务seed、A条件、窗口、端点→实例；旧日程batch4交替完整/局部；provider缺失时有生成分支|历史说明，不用于新分析loader；不能误把D5新batch8日程说成旧batch4|
|`rc3_global_local/run_s0.py:make_allow`|F+条件0/1/2→逐臂允许表；条件非零时按索引确定性屏蔽并为每个可服务任务保留一个F合法对|复用已存A，不把它解释为真实IK；真实耦合双臂任务不在当前模型中|
|`rc3_fast/solver.py:vectorized_feasibility`|任务xyz、配置xyθ、肩高/臂间距/前偏置、球扇区范围→F[N,M,2]|同臂必须同时满足F和A。当前代理不验证关节极限/臂运动扫掠/双臂耦合|
|`rc3_baseline/baseline.py:navigation,expand_route`|膨胀桌体/障碍（base_radius+clearance）、走廊图、候选接入→无向Dijkstra距离/前驱；保存前驱→导航折线与各边长度|复用精确NPZ，不新求导航；不能用候选直线距离替代D|
|`rc3_global_local_impl_review/runtime_guards.py:snapshot_parent,extract_snapshot`|scene/F/A+带来源的导航→不可变父/局部快照，绑定物理/任务/导航/producer身份|正式运行复用；本轮轻量adapter只读已有数组和证据，未调用生成/验证入口|
|`rc3_conservative_refinement/saved_inputs.py:SavedInputs.view`|允许的spec+保存文件/receipt/hash→同问题窗口、global ID映射、来源证据；无生成fallback|复用来源规则；R08-A实现不导入历史运行时的独立读取器|
|`rc3_global_local/problem.py:LocalProblem,extract`|有效逐臂关系F∧A、目标任务/桌窗、入口出口→保留对目标有覆盖且连通的候选；C=(F∧A).any(arm)，局部D包含端点行列|保留任务、候选身份和完整父导航；窗口外候选可保留，路径可出窗，不能把局部任务误作裁剪物理地图|
|`rc3_global_local/policy.py:static_data,GlobalLocalPolicy.encode`|父桌token14维；task5维；candidate7维；C与D→64维任务/候选表示、全局均值、key与GRU缓存|可复用合法SL256表示。5维任务为相对xy/z/两臂可行候选比例；7维候选为相对xy、cosθ/sinθ/h、覆盖比例、端点距离。C消息传递使用合并覆盖，未保留完整逐臂边类型|
|`GlobalLocalPolicy.score`|剩余任务、当前配置、历史GRU、raw_L、返回端点距离→合法候选logits；g/remaining/current/h各64维+4标量，key+dynamic+nav后tanh与query内积，再10tanh|原单步构造评分真实存在，无显式完整方案编辑动作；参数版本/问题/实例不符缓存会拒绝|
|`rc3_global_local/environment.py:Environment.mask,step`|mask要求正新增覆盖且当前→候选及候选→出口有限；选1候选→**全部**尚未服务且可覆盖任务立即标记已服务，更新位置/前缀/长度|不单独选择任务子集或手臂；不提供STOP、删除前缀、回退分配动作；最多N个正增益动作。完整回报含出口和一次终端H修正，和为−J_H|
|`rc3_global_local/terminal.py:export`|完整有序候选ID→停靠记录、服务分配、导航折线、L/K/J|顺序最早可服务配置负责该任务，L合法则优先L，否则R。服务臂是合法确定性选择，不是Actor另一个输出|
|`terminal.finish`；`rc3_constructive/baselines.py:postprocess,normalize`|raw完整路线→两轮normalize/2-opt（每次最多60pass），再joint_search最多2pass及normalize→H路线/成本/证书|H不是仅排序，也不是只删点；需要保留前后集合/顺序/分配。可重写任意原前缀，不能当作物理执行已提交路线的后缀优化|
|`rc3_fast/solver.py:joint_search`|当前覆盖计数→删除无独占任务的停靠；替换一个停靠时保留其当前独占任务覆盖；随后2-opt|可复用独立H。单配置逐次替换不是显式同时替换两个配置；不能声称旧系统从不做集合优化|
|`rc3_global_local/validation.py:validate,_validate`|问题+完整plan→身份/全覆盖/无重复配置/同臂F&A/按序动态服务/端点/每边D/折线经过服务姿态/障碍膨胀碰撞/J证书；臂只查合法、不强制L优先|完整复用；L优先来自上述export，不能混称验证器硬约束。物理joint_trajectory_certified始终false|
|`rc3_global_local_impl_review/endpoint_baselines.py:run_endpoint`；`rc3_v4/runtime.py:run`|端点shim、有效逐臂F、原D→原生Fixed3或Detour16最终方案与内部计数|Fixed3运行recipe(0,1,7)构造+2-opt+joint_search并验最优合法方案；不是“只取三个点”。原生final不额外加共同H|
|`rc3_v3/environment.py:Env,adaptive_windows`；`rc3_improve/repair.py:repair_window`|完整Fixed3初解→最多4停靠的连续窗口；条件任务上限8、候选80，固定端点DP修复；Detour按detour_excess选最多16次，接受降本并缓存失败上下文|这是**已有**完整方案局部联合修复。R08若仅学窗口评分将重复旧方向；新候选应显式改变非相邻配置对与覆盖集合，且须与此强规则对照|
|`rc3_score5_20261111/actor.py:Score5Actor`；`rc3_decoder_program_20261111/actor.py`|准确SL256→五张量25152/65088标量可训练，其余39936冻结；新key在梯度路径重算，冻结前缀状态随动作改变|保留旧实现；新目录不改D5参数白名单或初始化。新方案仅复用**未更新SL256**编码/初解|
|`rc3_decoder_program_20261111/trainer.py:Trainer.update`；D5 `training.py,science.py,runtime.py`|8个当前Actor完整轨迹→完整成本减同问题FrozenSL参考的REINFORCE；Adam1e-4/clip1/一次更新；SL为教师CE|真实在线采样、无Critic、无A1幅度或永久SL guard；原损失和失败不改。D5正式部署经export/load_export后冻结贪心+同H|
|D6 `costs.py,training.py,runtime.py,science.py`|A保留D5损失；B逐动作(final_H_J−raw_prefix)−q×FrozenSL_H_J；各256。轨迹身份、五张量白名单和采样器独立|只复用日志/恢复/预算框架，不能把D6负终点作为初始化或继续其训练|

## K、距离与服务的关键语义

K=`len(route[1:-1])`，每个内部候选配置只允许出现一次。配置由ID及(x,y,θ,h)定义；相同xy但朝向等不同的配置仍可分别计K。入口/出口是导航端点，不自动算服务配置，即使其坐标与配置重合。每任务独立按一只合法手臂服务，无容量、服务时间、双臂同时作业或任务先后约束。

父D被代码明确要求无向；不同入口/出口使局部虚拟0行（入口出发）与0列（到出口）通常不同，所以局部D整体不能盲目当对称矩阵。服务节点子矩阵仍对称，旧2-opt利用这一条件。新公式可表达有向情形，但旧H尚不能因此宣称支持有向导航。

## Actor的控制权与边界

**能改变**：合法候选排序/采样，连续多次选择形成的停靠配置集合、访问顺序，以及由最早服务规则间接形成的任务/手臂分配。旧全Actor训练也已发生，不能把“解冻主Actor”称为新机制。

**不能直接改变**：候选位置/朝向/高度、F/A/D、入口出口、任务需求、覆盖规则、某次选点服务的子集、左/右臂优先规则、已选前缀的撤销/交换、H搜索。mask不保证最优集合或顺序，只保证当前动作合法且有新增覆盖。

因此逐点MDP是联合集合—排序问题的一种顺序求解方式，**并非数学上不能表达好的组合**。现有池化、正增益转移、长前缀与非局部H回报使相关多动作的学习可能困难；哪个因素为主仍未被因果证实。R07不足只针对固定后续Top1/Top2，不证明所有构造式DRL不可能。

## 本轮接口完成范围

`contracts.py`：不可变问题/已有路线/联合编辑/状态/预算/恢复身份，以及覆盖前检；`adapters.py`：精确hash和归档offset读取、只从已存NPZ恢复F/A/D；`test_interfaces.py`：12个纯内存测试PASS。`LowerOrderingSolver`只是Protocol，**没有**真实solver、网络或训练器。真实转移、求解、数值Adam/RNG恢复、冻结部署等留待获准R08-C。

分析未导入Torch、SciPy或任何历史规划模块（见ANALYSIS_SUMMARY.json）。这避免把只读数组算术说成新规划，也不把接口测试说成新DRL有效。
