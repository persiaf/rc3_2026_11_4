# 训练对象、文献与MATLAB代码的对应

检索日期：2026-11-13。只引用原始论文和作者代码；没有把外部模型的成绩转写为本项目预期成绩。

## 1 首选：覆盖节点子集与顺序的构造式DRL

[Li等，Deep Reinforcement Learning for Combinatorial Optimization: Covering Salesman Problems](https://arxiv.org/pdf/2102.05875)，IEEE Transactions on Cybernetics，在线2021，卷期2022，DOI [10.1109/TCYB.2021.3103811](https://doi.org/10.1109/TCYB.2021.3103811)。模型用多头注意力和动态覆盖嵌入，逐次选节点至全覆盖；REINFORCE使用完整路径成本和greedy rollout参考。这比普通“每点必访”的TSP更接近本项目。

原文Table I：128维、3层、8头、Adam1e-4、batch256；每epoch320000实例、50epochs，约1600万次实例呈现。规模实验20→50、100→150、200→300；20→300明显退化，不能引用成任意跨尺度泛化。作者代码：[仓库](https://github.com/kevin031060/CSP_Attention)、[train.py](https://github.com/kevin031060/CSP_Attention/blob/master/train.py)、[options.py](https://github.com/kevin031060/CSP_Attention/blob/master/options.py)。README含旧TSP示例，不能直接当CSP复现实验命令。

本项目的迁移是研究推断：以逐臂可达E替代半径覆盖，以障碍导航D替代欧氏距离，以完整JH替代纯路长；固定端点与任务支持必须进入环境。关系表示的有效性需要消融，不能由引用证明。

## 2 机器人参照：BaSeNet

[BaSeNet: A Learning-based Mobile Manipulator Base Pose Sequence Planning for Pickup Tasks](https://arxiv.org/abs/2406.08653)，IROS2024，[全文](https://arxiv.org/html/2406.08653v1)。低层SAC学习单对象底座位姿并冻结，上层GNN/REINFORCE学习抓取顺序，以导航和抓取成本为目标。主要顺序评测为5/10对象，各50场景，不是百桌父场景的直接证据。

本项目已有MATLAB可达候选及导航组件，首轮不再承担连续位姿策略训练。先让DRL学习离散配置集合和顺序；只有证据显示候选离散化主导误差，才单独提议BaSeNet式连续位姿低层。增加SAC不是当前默认工作。

## 3 训练配方参考与不采用的路线

|来源|真实方法|本计划取舍|
|---|---|---|
|[Kool等，Attention, Learn to Solve Routing Problems!](https://arxiv.org/abs/1803.08475)|注意力构造+REINFORCE+greedy rollout baseline|简洁的完整回报训练骨架；不依赖必须校准好的Critic|
|[POMO](https://arxiv.org/abs/2010.16011)|同题多起点与共享基线|借同题多轨迹比较；项目固定入口/出口不直接复制TSP对称起点|
|[LEHD](https://arxiv.org/abs/2310.07985)|轻encoder/重decoder，主实验监督重构，推理可RRC|参考剩余节点动态信息；不把SL/RRC当DRL增量，不先扩大模型|
|[H-TSP](https://arxiv.org/abs/2304.09395)|上层PPO构造子集，低层REINFORCE接入路线|单层在实际规模上不够时才评议；不复活已关闭固定路线小菜单|
|[Learning to Delegate](https://arxiv.org/abs/2107.04139)、[NeuroLKH](https://arxiv.org/abs/2110.07983)|学习辅助传统求解器，包含监督/非RL目标|可参考求解接口，但不能替代第五章固定DRL核心|

没有一篇论文支持“换成PPO/DDQN/GNN必然改善本项目”。其中模型规模、奖励、数据与搜索预算必须分别对应。

## 4 MATLAB准确代码入口

来源仓库：[persiaf/MobileManipulationPlanning](https://github.com/persiaf/MobileManipulationPlanning/tree/5c51a02bcc109a9b23af73c72a94255adb958f5e)。本轮只读准确Git对象，无新运行。

|入口|可带入下一研究的内容|
|---|---|
|[evaluate_task_reachability.m](https://github.com/persiaf/MobileManipulationPlanning/blob/5c51a02bcc109a9b23af73c72a94255adb958f5e/manipulator/evaluate_task_reachability.m)|左右臂任务条件采样IK、路径和残差，作为离线E来源|
|[main_task_navigation_setup.m](https://github.com/persiaf/MobileManipulationPlanning/blob/5c51a02bcc109a9b23af73c72a94255adb958f5e/experiments/main_task_navigation_setup.m)|候选池、闭合端点、有向导航、任务成本身份|
|[build_joint_navigation_graph.m](https://github.com/persiaf/MobileManipulationPlanning/blob/5c51a02bcc109a9b23af73c72a94255adb958f5e/experiments/build_joint_navigation_graph.m)|真实可行导航边与代价，不能用坐标直线距离替代|
|[enumerate_minimum_task_covers.m](https://github.com/persiaf/MobileManipulationPlanning/blob/5c51a02bcc109a9b23af73c72a94255adb958f5e/planning_strategy/joint/enumerate_minimum_task_covers.m)|有限候选覆盖参照；其逐区域最少覆盖条件不能静默移植RC3|
|[solve_fixed_set_atsp.py](https://github.com/persiaf/MobileManipulationPlanning/blob/5c51a02bcc109a9b23af73c72a94255adb958f5e/planning_strategy/joint/solve_fixed_set_atsp.py)|对固定集合求最优顺序，区分选位收益与重排不足|
|[最新归因报告](https://github.com/persiaf/MobileManipulationPlanning/blob/5c51a02bcc109a9b23af73c72a94255adb958f5e/results/joint_attribution_20261113/ANALYSIS.md)|27/27相对原集合最优的选位证据；新预算调度本身未占优|

第三章真实安装标定、自碰撞/双臂碰撞、连续轨迹与接触力尚不是这些离线结果的认证范围。该限制影响论文结论等级，不妨碍先开展明确标注的任务条件规划研究。
