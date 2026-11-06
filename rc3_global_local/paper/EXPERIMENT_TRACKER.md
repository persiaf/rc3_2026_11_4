# 实验跟踪

| 阶段 | 当前状态 | 证据/范围 |
|---|---|---|
| 研究目标 | FIXED_DRL | RESEARCH_OBJECTIVE.md |
| S0环境单测 | PASS 38项 | 合成F/A/D，无真实scene/nav/H |
| S0策略单测 | PASS 13项 | 合成结构forward/backward单独记账，optimizer0 |
| S0事前登记 | COMPLETE | 本地提交89af4e0，源与种子/版本已钉住 |
| S0真实探针 | COMPLETE_PASS | 3父图18窗口，54普通+9梯度轨迹，63重放 |
| S1 | NOT_STARTED | 有限辅助监督，不是最终方法 |
| S2 | NOT_STARTED | 未来真正大场景DRL，执行前绑定runner |
| S3 | NOT_STARTED | 冻结局部完整规划与质量时间验证 |

原60测试封存，旧C/R/T证据冻结，默认Fixed3／Detour。本地独立研究分支不自动推送。

S0实际63次完整终局/H，0失败；63精确重放缓存命中，所有轨迹代理验证通过。真实神经27次编码、1546次评分、1次backward，参数哈希不变、optimizer0，研究运行墙钟9.405554s。合成单测历史另计38/13最新验收及此前必要修复回归。S0只支撑环境/梯度接口，不声称训练后质量或效率，S1/S2/S3仍未启动。

未来设计有效入口更新为FUTURE_DESIGN_STATUS.json→v2。v1的regular新seed不构成新物理布局；v2仅预登记mixed/blocked父图及canonical map_key独立性门槛，尚未实际验证未来布局或生成S3任务。正式执行资格只属于v2，原v1/S0字节均保持。
