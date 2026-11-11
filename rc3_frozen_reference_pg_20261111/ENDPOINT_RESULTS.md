# 固定终点完整路线结果

仅单seed11622026、既有暴露开发案例。每终点66案例×6方法，父布局配对区间为描述性证据。

## PG-cond主比较：父布局等权相对节省

| 终点 | 案例组 | 指标 | 对照 | 相对节省% | 父布局配对95%区间% |
|---:|---|---|---|---:|---:|
| 64 | independent_parent_exposed_development | J | FrozenSL | 0.000000 | [0.000000, 0.000000] |
| 64 | independent_parent_exposed_development | raw_J | FrozenSL | 0.000000 | [0.000000, 0.000000] |
| 64 | independent_parent_exposed_development | J | continued_SL_A1 | 0.000000 | [0.000000, 0.000000] |
| 64 | independent_parent_exposed_development | raw_J | continued_SL_A1 | 0.000000 | [0.000000, 0.000000] |
| 64 | independent_parent_exposed_development | J | PG-global | 0.000000 | [0.000000, 0.000000] |
| 64 | independent_parent_exposed_development | raw_J | PG-global | 0.000000 | [0.000000, 0.000000] |
| 64 | trained_parent_new_tasks_exposed_selection | J | FrozenSL | 0.000000 | [0.000000, 0.000000] |
| 64 | trained_parent_new_tasks_exposed_selection | raw_J | FrozenSL | 0.000000 | [0.000000, 0.000000] |
| 64 | trained_parent_new_tasks_exposed_selection | J | continued_SL_A1 | 0.000000 | [0.000000, 0.000000] |
| 64 | trained_parent_new_tasks_exposed_selection | raw_J | continued_SL_A1 | 0.000000 | [0.000000, 0.000000] |
| 64 | trained_parent_new_tasks_exposed_selection | J | PG-global | 0.000000 | [0.000000, 0.000000] |
| 64 | trained_parent_new_tasks_exposed_selection | raw_J | PG-global | 0.000000 | [0.000000, 0.000000] |
| 128 | independent_parent_exposed_development | J | FrozenSL | 0.000000 | [0.000000, 0.000000] |
| 128 | independent_parent_exposed_development | raw_J | FrozenSL | 0.000000 | [0.000000, 0.000000] |
| 128 | independent_parent_exposed_development | J | continued_SL_A1 | -0.004556 | [-0.013668, 0.000000] |
| 128 | independent_parent_exposed_development | raw_J | continued_SL_A1 | 0.000333 | [0.000000, 0.001000] |
| 128 | independent_parent_exposed_development | J | PG-global | 0.000000 | [0.000000, 0.000000] |
| 128 | independent_parent_exposed_development | raw_J | PG-global | 0.000000 | [0.000000, 0.000000] |
| 128 | trained_parent_new_tasks_exposed_selection | J | FrozenSL | 0.000000 | [0.000000, 0.000000] |
| 128 | trained_parent_new_tasks_exposed_selection | raw_J | FrozenSL | 0.000000 | [0.000000, 0.000000] |
| 128 | trained_parent_new_tasks_exposed_selection | J | continued_SL_A1 | 0.000000 | [0.000000, 0.000000] |
| 128 | trained_parent_new_tasks_exposed_selection | raw_J | continued_SL_A1 | 0.000000 | [0.000000, 0.000000] |
| 128 | trained_parent_new_tasks_exposed_selection | J | PG-global | 0.000000 | [0.000000, 0.000000] |
| 128 | trained_parent_new_tasks_exposed_selection | raw_J | PG-global | 0.000000 | [0.000000, 0.000000] |

## 六方法完整成本与请求时间

| 终点 | 方法 | H-J均值 | raw-J均值 | L均值 | K均值 | 完整请求中位秒 | p95秒 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 64 | FrozenSL | 137.889648 | 168.174374 | 110.912375 | 35.969697 | 0.440820 | 0.653239 |
| 64 | continued_SL_A1 | 137.889648 | 168.174374 | 110.912375 | 35.969697 | 0.524497 | 0.878056 |
| 64 | PG-global | 137.889648 | 168.174374 | 110.912375 | 35.969697 | 0.510387 | 0.847863 |
| 64 | PG-cond | 137.889648 | 168.174374 | 110.912375 | 35.969697 | 0.491635 | 0.927906 |
| 64 | Fixed3 | 134.764308 | 134.764308 | 109.150672 | 34.151515 | 0.375219 | 0.484835 |
| 64 | Detour16 | 132.926705 | 132.926705 | 107.142614 | 34.378788 | 0.431336 | 0.691474 |
| 128 | FrozenSL | 137.889648 | 168.174374 | 110.912375 | 35.969697 | 0.417793 | 0.608961 |
| 128 | continued_SL_A1 | 137.888000 | 168.174536 | 110.910727 | 35.969697 | 0.488458 | 0.830537 |
| 128 | PG-global | 137.889648 | 168.174374 | 110.912375 | 35.969697 | 0.481923 | 0.804287 |
| 128 | PG-cond | 137.889648 | 168.174374 | 110.912375 | 35.969697 | 0.510359 | 0.826442 |
| 128 | Fixed3 | 134.764308 | 134.764308 | 109.150672 | 34.151515 | 0.363377 | 0.490348 |
| 128 | Detour16 | 132.926705 | 132.926705 | 107.142614 | 34.378788 | 0.431446 | 0.615311 |

本表成本均值是66案例均值；主推断采用上表的父布局分组。完整各尺寸、J/raw-J/L/K及完整请求时间对照见[QUALITY_CONTRASTS.csv](execution_v2/science/QUALITY_CONTRASTS.csv)。

原生Fixed3/Detour16的raw栏沿用其最终原生搜索/H路线，不代表去掉原生内部H的额外实验；没有追加共同H。显式共享postprocess前后路线、原生最终路线和内部搜索调用均保留；原生全部H中间路线未单独捕获，见[记录边界](SCHEMA_AND_ACCOUNTING_NOTES.md)。

完整请求计时包括输入/来源/已保存导航恢复、冷模型加载、provider/新增特征、策略或native、H、完整导出/验证、归档和索引/完成journal。计时收据自身写入计入阶段墙钟；文件系统缓存未控制，验证字节缓存真实记录。
