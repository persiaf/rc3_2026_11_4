# 独立保存数据数值与来源审查

数值、实际模型来源与路线审查：**GO／PASS，附一项已披露元数据缺陷**。此处“独立”指独立实现并在独立进程执行的只读数值/来源核对；没有导入实验实现、实例化策略、运行autograd/optimizer或调用导航/规划器，不冒充外部人类同行评审。

实际核查：203源码文件身份、49638旧冻结文件原字节、36份原模型来源；5份轻量包逐张量与完整来源参数/精度一致。612份实际计划逐例从原F/A和保存D重算合法候选、mask、同臂分配、raw/final J/L/K及导航polyline长度；实际服务分配34002条，greedy mask 22149个。

384次更新的日志预扣/闭合、每次实际梯度范数、clip1前后张量、更新后权重哈希、共同SL起点和fresh Adam初态通过。A/B512问题与view相同；LOO、1/.25/.25、B首动作参与和3072终局预扣一致。梯度裁剪最大数组差1.49e-08；父布局收益/区间与独立重算最大差0。

**GLV4_META_001：通用adapter默认角色未覆写。** A/B/C的108份原始结果仍有`training_weight_role=SL256`；`plan.method=frozen_supervised`表示共用执行管线，不能表示实际学习方法。代码根因是`complete_request`以通用method调用旧接口，同时通过Catalog.override_role正确载入各组模型，随后未重标角色。独立追加审计对全部216份原始artifact和Journal哈希、声明study_method、actor文件/张量哈希、固定128来源及CSV分组/成本进行严格匹配，144份神经计划和72份native均符合准确来源。没有改动任何原始结果或科学源码，也未重跑、补值或隐藏此字段错误；[澄清](METHOD_IDENTITY_CLARIFICATION.json)和[逐份映射](METHOD_IDENTITY_AUDIT.json)供复核。数值结论不变，不声称原始角色元数据全部正确。

部署等价工程GO，时间减幅只能按实际role/cache表报告。自然起点的单种子candidate判定为REVISE，不能用B仅优于A、后处理或工程提速替代学习收益。完整成本与限制见[目标对照](START_OBJECTIVE_ABLATION.md)。

限制：新任务仅3个已暴露DEV物理布局；未验证新的未见布局、跨优化种子或实机IK。DEV216请求计时含第一次保存导航产物加载，不能拿它作严格配对部署速度比较；396网格的冷热和模型状态独立、可比较。原S3两个缺失和负结论保留，不回填、不换任务或改A。

审计墙钟474.036010秒，新增科学请求/策略forward/optimizer/nav/native/H均0。[完整审计数据](INDEPENDENT_AUDIT.json)及[独立脚本](../project_management/gl_v4_delivery/independent_audit.py)。
