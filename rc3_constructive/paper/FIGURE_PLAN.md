# C2 图表计划

| ID | 类型 | 数据 | 用途 |
| --- | --- | --- | --- |
| F1 | 双面板学习曲线 | outputs/validation_curve.json | 48个选择场景 raw／post mean J 对更新数，显示初始值；不称独立测试 |
| F2 | 双面板质量—时间散点 | outputs/analysis.json | 同一轮672次交错测量的冷热比较，7方法相同坐标单位；mean J 与median time是边际统计 |
| T1 | MD／LaTeX 表 | outputs/analysis.json | 逐方法 L、K、J、raw J、完整中位／p95与回退 |

使用 paper-figure 技能生成矢量PDF及300dpi PNG，再由独立审查者复核图注／比较范围和数值。机制图、CSP／BC消融和跨规模图本轮不生成，以免制造尚未取得的证据。
