# 接口细节与执行状态

新接口的action为0-based局部候选位置；plan中的candidate_id为0-based全局候选ID；D中服务节点为candidate_id+1，0为基地。MATLAB矩阵下标为1-based，但导出字段值明确用0-based。

输入中保留：场景几何、桌子/任务/候选ID、候选theta/h、F、A、D、坐标、service_nodes、predecessors、来源模型与rejected_edge_count。Parent仅将数组设为只读；调用者不能修改其scene字典后继续沿用旧身份，应重新构建Parent并生成新identity。

当前接口假设无向导航，允许D=+inf拒绝不可达目标。没有仅凭欧氏距离补不可达边。输出折线由predecessors重建并核对长度，然后检查矩形障碍膨胀。若真实场景是多边形footprint、非矩形障碍或有转向/升降扫掠约束，应接入现有独立几何验证器，不能沿用本教学验证器声称完整实机安全。

同臂F&A、入口出口、空任务、窗外可用候选均有测试。所有目标已覆盖才终止；非法action失败不改变状态。无STOP动作，不允许网络提前终止省成本。

本包不实现正式H；postprocessor字段显式none_not_the_frozen_research_H。因此demo成本不能与使用冻结H的科研结果直接对比。

learning_math.py的PPO损失返回逐状态项，轨迹与规模的最终归约由正式trainer明确实现；当前不宣称它已完成训练预算、PPOepoch、KL回滚或critic训练。

fetch_locked_sources.py仅获取源码，默认跳过LFS，只有--execute才网络克隆；不会运行main.py或训练器，也不会修改现有目录。
