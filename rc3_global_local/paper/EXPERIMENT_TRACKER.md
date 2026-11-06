# 实验跟踪

| 阶段 | 当前状态 | 证据/范围 |
|---|---|---|
| 研究目标 | FIXED_DRL | RESEARCH_OBJECTIVE.md |
| S0环境单测 | PASS 38项 | 合成F/A/D，无真实scene/nav/H |
| S0策略单测 | PASS 13项 | 合成结构forward/backward单独记账，optimizer0 |
| S0事前登记 | PENDING | S0_protocol.json / registration_receipt.json |
| S0真实探针 | PENDING | 3父图18窗口，计划54普通+9梯度轨迹 |
| S1 | NOT_STARTED | 有限辅助监督，不是最终方法 |
| S2 | NOT_STARTED | 未来真正大场景DRL，执行前绑定runner |
| S3 | NOT_STARTED | 冻结局部完整规划与质量时间验证 |

原60测试封存，旧C/R/T证据冻结，默认Fixed3／Detour。本地独立研究分支不自动推送。
