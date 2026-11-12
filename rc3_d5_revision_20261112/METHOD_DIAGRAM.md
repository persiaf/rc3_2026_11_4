# 本候选的训练与冻结路径

```mermaid
flowchart LR
  I["已登记的同一问题输入\nscene / F / A / D / 窗口与端点"] --> E["冻结原表示和prefix GRU\n39,936标量不变"]
  E --> S["原Actor评分路径\n仅key / dynamic / nav / query五张量\n25,152可训练标量"]
  S --> M["原合法mask\n正覆盖增益、当前位置及出口可达"]
  M --> A["训练：己方on-policy采样\n冻结评价：贪心选择"]
  A --> T["原任务转移\n覆盖全部目标并到出口"]
  T --> H["原固定H + 独立模型约束验证\n完整raw/H路线及服务分配"]
  H --> G["完整G = -J_H\nJ = L + 0.75 K"]
  I --> B["准确FrozenSL同问题贪心+H\n永久冻结的完整J参考缓存"]
  G --> P["PG：mean(detach(J_sample-J_ref) × sum logpi)\n单批8路线、一次Adam"]
  B --> P
  P --> S
  I --> C["共同问题日程\n每批4完整 + 4局部"]
  C --> A
  C --> U["匹配SL-U\n该问题已绑定强教师完整动作\nmean_episode mean_action CE"]
  U --> S
```

两个训练流是独立的模型、Adam和采样状态，均从准确SL256重新克隆。图中共享评分框只表示相同结构与参数范围，不表示共享更新后的权重或轨迹。PG不读取教师动作作为提示。没有神经Critic、额外A1残差、PPO多epoch或永久贴近SL的guard。

Actor能改变合法停靠选择和访问顺序，进而影响覆盖路径及固定H结果；F/A、导航输入、mask、任务转移、成本、服务规则与H均保持原版。模型约束证书不是实际机械臂标定/IK或关节轨迹认证。

P1未通过，实际权重仅交付为科学证据；未替换项目默认方法，未进入部署优化或新父布局确认。
