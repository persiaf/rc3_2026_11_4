# 实验跟踪

| Run ID | 阶段 | 方法/范围 | 划分 | 状态 |
|---|---|---|---|---|
| T0 | 诊断 | 旧R0/R1版本化保存证据 | 已知证据 | COMPLETE，无新研究调用 |
| UNIT | 接口 | 独立合成菜单测试19项 | 合成小图 | PASS |
| T1 | 动作 | 规则/线性/MLP，同菜单匹配监督 | train96/dev24 | PENDING_REGISTRATION |
| T2 | 路线 | 7方法与独立配对有界教师 | 新36场景 | CONDITIONAL_ON_T1 |
| T3 | RL | 未授权 | 无 | NOT_STARTED |

所有事前配置以protocol.json及registration_receipt.json为准，运行状态以STATUS.json和resources.json为准。
