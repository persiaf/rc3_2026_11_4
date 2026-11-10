# 代码结构与阅读顺序

本轮保留全部现有模块与历史目录，不移动代码或实验产物。`docs/`提供科研协作导航。

| 路径 | 作用 | 当前阅读/执行边界 |
|---|---|---|
| `rc3_baseline/baseline.py`、`validator.py` | 代理场景、导航、独立标量验证 | 验证器可直接重核本轮输出，不依赖训练数据 |
| `rc3_fast/scenes.py`、`solver.py` | 复杂场景生成、F和路径、导出 | 最小示例由此新建场景，不读取保存测试 |
| `rc3_improve/repair.py` | 覆盖保持的局部修复 | 历史规则内核，保持原字节 |
| `rc3_v3/environment.py` | 修复环境、特征、参考recipe | 被v4/v5原生规则调用 |
| `rc3_v4/runtime.py`、`plan.py` | 独立认证锚点、规则修复、失败回退 | 当前v5继承的规则路径 |
| `rc3_v5/plan.py`、`external.py` | 默认代理CLI与外部证书入口 | 新手只运行QUICK_START指定的代理CLI |
| `rc3_v5_1/` | 机器人输入预检 | 模板blocked不等于实机失败实验 |
| `rc3_global_local/problem.py`、`environment.py` | 父场景/局部问题与构造式MDP | 科研动作索引、端点和mask以此为准 |
| `rc3_global_local/policy.py` | 表格全局token、覆盖聚合、GRU | 不是原论文CNN架构 |
| `rc3_global_local/terminal.py`、`validation.py` | 冻结H、导出、代理验证 | 保留无真实关节认证字段 |
| `rc3_global_local_impl_review/runtime_guards.py` | 不可变输入、缓存及来源约束 | 旧schema不能跳过身份验证 |
| `rc3_global_local_trainer/` | 训练、RNG、checkpoint与分账 | 本轮不运行 |
| `rc3_global_local_execution_v3/` | 冻结协议、独立批准与阶段收据 | v3状态与旧v2设计指针分别保留 |
| `rc3_s3_evaluator/`、`rc3_s3_connectivity_20261109/` | 冻结评价、scope_r1准入 | 只读报告；本轮不解封或重跑S3 |
| `rc3_s1*_results_*`、`rc3_s2_results_*`、`rc3_s3*_results_*` | 阶段报告、账本、结果索引 | 历史目录不覆盖 |
| `rc3_gl_v4_20261109/`、`rc3_gl_v5_20261109/` | 已结束的机制筛选 | 必读负结果及NEXT_STEP_DECISION |
| `rc3_integrated_rppo/` | 后续独立RPPO模块及部分证据 | 本轮不启动或接管活动运行 |
| `rc3_teacher_distill/`、`rc3_constructive/`、`rc3_nav_value/`、G系列 | 历史探索 | 按实验索引理解；不当作当前默认方法 |
| `交接/RC3_Teacher_Handoff_v1.zip` | 教学接口、需求、来源与原测试材料 | 保留原包；未向RC3命名空间复制任何示例代码 |
| `project_management/` | Git清单、来源与交付收据 | 历史状态按所属轮次解读 |
| `paper/`、论文Word、`paper_figures_v1/` | 论文展示和草稿 | 不修改当前用户Word编辑 |
| `docs/evidence/` | 本轮审计及两个代理方案证据 | 维护/工程账，不是训练结果 |

建议代码阅读顺序：`rc3_v5/plan.py` → `rc3_v4/runtime.py` → `rc3_fast/solver.py` → `rc3_baseline/validator.py`，之后再读科研`problem/environment/policy/terminal/validation`及训练器。先弄清规则完整输出，再比较DRL替换的是哪一环。

本轮新建的`docs/tools/reproduce_baselines.py`只是两个CLI和独立验证器的有界编排，拒绝覆盖输出目录，不实现任何规划算法。它没有checkpoint参数，不调用训练、H科研入口、S3或保存测试集。
