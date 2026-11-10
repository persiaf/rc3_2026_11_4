# 执行跟踪

| 项目 | 状态 | 实际证据 |
|---|---|---|
| 新206维真实PPO/事务guard/双Adam/部署 | IMPLEMENTED | ../trainer.py、runtime.py、runner.py |
| 工程R01 | FAILED_UNIT_ONLY | 86测试4错误，0请求/optimizer，失败原账 |
| 工程R02 | FAILED_INPUT_LINEAGE_CHECK | 87测试PASS，1输入失败，0规划/optimizer |
| 工程R03 | PASS | 90测试、20请求、5Actor/4Critic，含raw/H独立验证 |
| 独立源码/账本审查 | COMPLETE | INDEPENDENT_TRAINER_REVIEW.md |
| 具体协议R03 | APPROVED | authorizations/下真实回复、协议批准和独立阶段授权 |
| 正式Critic预热 | COMPLETE | ON/OFF各24条自身轨迹、32次Critic更新，固定同一累计账 |
| ON/OFF各最多256机会 | STOPPED_AT128 | 各128接受/0拒绝；按两规则停止，256未开始 |
| 匹配继续监督最多256机会 | CLOSED_AT128 | 128接受/0拒绝；匹配机会/结构/guard，不声称FLOPs匹配 |
| 六方法66视图配对评价 | COMPLETE_64_128 | 全部有效；ON/OFF raw/H路线与SL全部相同 |
| 正式Critic校准/完整时间比较 | COMPLETE_NO_EFFICACY | value MSE超常数，完整时间无优势；独立判读no |

研究阶段与工程分账，旧small256失败门槛和封存边界保持。所有正式任务仅在确切协议与阶段分别获准后执行；负结果提前收尾，不追加轮数。
