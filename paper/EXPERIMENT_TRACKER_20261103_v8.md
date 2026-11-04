# v4 执行记录

日期：2026-11-03。预注册方案见 EXPERIMENT_PLAN_20261103_v8.md。

|工作|状态|证据|
|---|---|---|
|附件复核与旧冻结审计|完成，31/31匹配|rc3_v4/audit_results|
|外部服务证据接口|完成，真实机器人配置仍缺|rc3_v4/EXTERNAL_SCHEMA.md、ROBOT_INPUT_STATUS.md|
|精确参照|完成，225/225有效，44/45组合最优|rc3_v4/small_results|
|图输入向量化|完成，384状态逐位一致，微测过门槛|rc3_v4/feature_equivalence.json、profile_vectorized|
|六次80轮训练|完成，23040更新，951.59秒|rc3_v4/training/summary.json|
|跨后端复现|完成，288状态无动作差异|rc3_v4/training/parity.json、portable_validation.json|
|新冻结测试|完成，39场景855方案全部通过|rc3_v4/results/verification.json|
|全部回归|完成，40项通过|rc3_v4/verification/summary.json|
|采用决策|不替换默认Fixed3/Detour；质量正信号但输入增益CI跨零且时间超门槛|rc3_v4/results/decision.json|
|结果审阅|完成：partial，按预注册门槛保留默认规则|rc3_v4/claims_review.md|
|交付打包|资料齐备；压缩包验收见同名delivery.manifest.json与delivery.smoke.json|rc3_v4_delivery.zip|

没有新增PPO训练。真实IK/实机约束不标记为已完成；接口与合成验证已交付。
