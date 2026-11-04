# v5 执行记录

|工作|状态|位置|
|---|---|---|
|上传证据安全导入|83/83 清单文件一致；Linux108方案与全部统计重建通过|rc3_v5/import_integrity.json、uploaded_*audit.json|
|共同约束修复|F∩A贯穿构造/修复/规范化/分配；28测试、5 CLI通过|rc3_v5/external_results|
|图嵌入缓存与等价性|144步逐位一致、36等价计划；单次108计时协议全部有效，17快/1慢|rc3_v5/cache_results|
|本机旧MLP同预算补评|108/108有效；+0.260507%，CI[-0.008684,0.529545]跨零；与Linux分列|rc3_v5/mlp_budget_windows、native_*audit.json|
|机器人资产和配置preflight|两URDF/19STL完整；15合成测试通过；真实配置缺项明确blocked_inputs，无真实IK/F|rc3_v5/robot_input.py、ROBOT_INPUT_STATUS.md|
|集成与原版回归|v5 54/54、v4 40/40；四模式CPU入口与冷暖缓存通过|rc3_v5/verification、v4_regression.log|
|证据整理与交付|结果与三部分证据索引已齐，独立审阅见run03；整包与搬迁校验以根目录交付回执为准|rc3_v5/RESULTS.md、EVIDENCE_MAP.md；rc3_v5_delivery.manifest.json/.smoke.json|

没有新增训练、改变原采用门槛或择优重跑性能。默认Fixed3/Detour，旧MLP可选，Structured研究模式。
后续真实闭环仍依赖有效标定配置及IK/碰撞见证，不能将资产齐备或输入检查通过当作完成实体执行认证。
