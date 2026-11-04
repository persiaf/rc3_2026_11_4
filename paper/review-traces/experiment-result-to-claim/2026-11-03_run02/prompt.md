# Result-to-claim 审阅请求

审阅者：/root/audit_maintenance_v4。日期：2026-11-03。

任务：在第四版全部39场景计时完成、manifest.status=complete且summary/decision/verification落盘后，独立评价两个主张。

C1：离散服务接口、认证初解/最终验证回退，及小规模全局最优质量参照。
C2：同网络同监督同预算下，局部结构输入优于统计输入控制和Detour，同时满足预设时间与有效性门槛。

来源：paper/EXPERIMENT_PLAN_20261103_v8.md；rc3_v4/training/summary.json；results/plans原始方案、metrics、summary、decision、verification；small_results/summary.json与READOUT.md；verification/summary.json；training/parity.json与stop_diagnostic.json；ROBOT_INPUT_STATUS.md。

必须重新核算100桌18场景配对均值/20000次bootstrap（seed731203）以及17/29/43训练种子结果；检查39/357/855分母与400桌仅3例的范围。整体采用门槛须全部满足；部分质量获益不能否认，速度失败也不能隐藏。训练为收益监督，不称RL；局部输入不称充分状态；合成接口不称真实IK。

要求输出claim_supported、支持/不支持内容、缺项、修订主张、后续路由与置信度。没有paper claim audit时为provisional。明确内部独立于训练实现，非外部零上下文审查。不能改原始数据、模型、主代码或主报告。

已知角色限制：审阅者此前维护哈希审计并修复外部验证的罚项请求绑定；未参与本轮特征/网络实现、训练或主报告编制。数字复核脚本独立于主report统计实现，共用既有独立几何验证器。
