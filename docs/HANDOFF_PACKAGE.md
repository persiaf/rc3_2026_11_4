# 教师交接包整合与版本差异

来源为用户提供的[RC3_Teacher_Handoff_v1.zip](../交接/RC3_Teacher_Handoff_v1.zip)。原包此前是源工作区未跟踪文件，本轮按原字节纳入Git LFS，未覆盖任何RC3模块；尚未推送。

原包SHA256为`5a4e414e0af41bf1138372c238ee78c1f60f8afb158243debfe86a73d830f914`，1,634,947字节。六份需求/接口/来源/原测试资料置于`docs/handoff_reference/`，逐项匹配包内MANIFEST哈希。[整合收据](evidence/SOURCE_INTEGRATION.json)

| 材料 | 来源与验证 | 不能据此声称 |
|---|---|---|
| 当前RC3规则/科研模块 | 精确提交、源码绑定、原报告与审查；本轮仅验Fixed3/Detour16代理入口 | 全部历史实验本轮重跑、真实双臂认证完成 |
| 包内legacy_rc3_runtime | 旧rc3_v5_delivery.zip提取的源码子集，有来源清单 | 是GL-v5或最新RPPO完整仓库 |
| 包内src/rc3_handoff | 新建教学接口，原包26项测试PASS | 是原历史模型或完整正式训练器 |
| learning_math.py、maps.py | 新建数学/地图处理辅助 | 已复现作者DDQN或已训练PPO |
| gain-cost demo | 新规则示例，无科研H和DRL权重 | 是Fixed3/Detour或论文主实验结果 |
| MATLAB导出模板 | 新字段转换示例，未在MATLAB执行 | 已取得旧完整.m工程 |
| uavSim/icar两份源码节选 | 包内锁定作者提交与BSD许可 | 完整上游运行或重训已复现 |

本轮未导入示例模块，未重跑原包测试/演示或合成梯度。26项PASS仅为原交接Linux环境报告，和本轮Windows验收分别记账。[原测试记录](handoff_reference/HANDOFF_TEST_REPORT.json)

## 已整合的接口要求

输入须有scene、任务/候选全局ID、坐标单位、theta/h、同臂F/A、D、coordinates、service_nodes、predecessors与来源receipt，保留identity使用的全部导航字段。不能只有D就声称有绕障完整折线。

科研动作节点1-based、0为符号端点；示例动作0-based，D服务节点为candidate_id+1。MATLAB下标与导出ID另行明确，不能猜测转换。示例raw回报没有科研H，不能直接比较H后科研成本。[原接口说明](handoff_reference/INTERFACE_NOTES.md)

输出须含目标ID、候选停靠位姿、访问顺序、任务—候选—臂指派、端点、导航折线、J/L/K、方法/权重/H来源及验证。`arm_joint_trajectories=null`、`physical_joint_trajectory_certified=false`等边界须保留。

## 来源时点的区别

原包固定RC3提交`7d2cfe75919b25bac59c5274321baca4d8e98f77`。“RPPO未执行”对应该时点；后续`0c268426f`已有实现及未闭合部分运行证据。本轮未回写原包文字，当前解释看[进度页](RESEARCH_PROGRESS.md)。

原包记录private，本轮GitHub API返回public；本轮未改变权限。旧MATLAB源码、真实标定和IK仍缺。[来源缺项](handoff_reference/SOURCE_AVAILABILITY.md)、[来源锁](handoff_reference/SOURCE_LOCKS.json)

[完整原始需求](handoff_reference/REQUIREMENTS_TEACHER.md)和[代码地图](handoff_reference/CODE_MAP.md)按原字节保留，其中相对路径按ZIP根目录解释。其数字和未执行提案保留原有来源边界。
