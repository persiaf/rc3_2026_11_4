# 实验追踪：已完成

| ID | 工作 | 状态 | 可核查产物 |
|---|---|---|---|
| R001 | 向量化可达性、2-opt 与原实现对照 | 完成 | `../rc3_fast/test_fast.py`；192 个测试场景向量化控制与原基线等目标 |
| R002 | 复杂场景、障碍和覆盖保持局部搜索 | 完成 | `../rc3_fast/scenes.py`、`solver.py`；底盘障碍独立校验 |
| R003 | 初试数据与 MLP | 完成 | `../rc3_fast/data_20261102/`、`run_selector_20261102/`；7,500 次更新 |
| R004 | 扩大数据与两类 MLP | 完成 | 1,152/192/192 场景；`run_extended_20261102/`、`run_portfolio_20261102/`；各 27,000 次更新 |
| R004b | 树模型与测试前模型选择 | 完成 | `run_forest_20261102/`、`deployment/selection.json`；九个树模型配置 |
| R005 | 独立留出测试 | 完成 | `benchmark_20261102/`；192 场景×9 方法，1,728 计划有效 |
| R005b | 144/196 桌外推 | 完成 | `benchmark_ood_20261102/`；24 场景×9 方法，216 计划有效 |
| R006 | NumPy 模型、CLI、缓存与独立复验 | 完成 | `deployment/`、`demo_final/`、两测试目录的 `independent_validation.json` |
| R007 | 结果归因、图表和使用说明 | 完成 | `../rc3_fast/EXPERIMENT_REPORT.md`、`README.md`、`figures/` |

判定：C1 在百桌快速模式下部分支持；C2 学习相对同预算固定策略的额外收益未证实。保留负面结果，不因本轮结束而改称模型已优于强优化基线。
