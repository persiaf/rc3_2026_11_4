# 固定展示入口（现有记录，不新增实验）

选择规则已固定：使用指定反例、冻结百桌列表首项、默认Fixed3集成示例。没有按学习收益或耗时择优选图；没有运行规划、推理或绘图。完整相对路径、SHA-256及经过读取确认的RFC 6901 JSON指针在 [display_manifest.json](display_manifest.json)。所有路径相对于交付根目录。

## ① 共同约束的四任务规范化反例

来源 `rc3_v5/external_results/canonicalization_counterexample.json`；2桌、4任务、32候选，seed=870001。场景 `/bundle/scene`，F `/bundle/planning/F`，A `/bundle/audit/F`。

展示 `/f_only_canonical_route` 的 `[0,1,0]` 与 `/common_canonical_route` 的 `[0,1,24,0]`，以及 `/direct_export/v5/plan/route_polyline` 的成功完整路径。导航节点0是起点，节点i+1对应候选i，不能将导航节点ID直接当候选ID。保留候选0和23，L=11.3、K=2、J=12.8；任务分配在 `/direct_export/v5/plan/assignments`。Fixed3/Detour的保存计划另在 `/planning/v5/{fixed3,detour}/plans/{fixed3,detour}`。

适合画共同覆盖关系及“为何不能提前删掉后站”；这是合成双表反例，不是测得的真实机械臂工作空间，不能补画原失败算法不存在的已认证完整路径。

## ② 原百桌列表第一场景：固定调用数方法对照

按 `rc3_v4/data/test_cases.json` 的保存顺序取首个百桌项 `/0`：case_index=0、seed=1565649625、regular，100桌、158任务、800候选。唯一方案文件为 `rc3_v4/results/plans/scene_plan_000_1565649625.json`。各方法一律取rep0，没有看结果后挑重复。

| 保存方法标签 | L / m | K | J / 等效m |
|---|---:|---:|---:|
| fixed3_rep0 | 209.550 | 94 | 280.050 |
| detour8_rep0 | 205.700 | 94 | 276.200 |
| v3_mlp8_rep0 | 205.100 | 93 | 274.850 |
| control8_rep0 | 205.100 | 93 | 274.850 |
| structured8_rep0 | 207.300 | 94 | 277.800 |

读取 `/plans/{标签}/route_polyline`、`assignments`、`service_stops` 和 `selected_candidate_ids`；目标成本为同前缀下 `objective_length_plus_stops_m`。场景位于 `/scene`，原种子位于 `/metadata/seed`，预算配置位于 `/runs/{标签}/config`。Fixed3是0次修复参照，其余为固定8次；不能把这组与wall400标签混合。精确原浮点值保留在JSON索引。

适合五幅同场景路线图及L/K/J表。此例Structured的J=277.8高于Detour8的276.2，应如实展示；不能用单例推断总体胜负。总体结论仍引用冻结全组统计。

## ③ 3×3集成格式与几何示例

来源 `rc3_v5/verification/fixed3.json`：seed=771903、blocked，9桌、14任务、72候选。场景 `/scene`；路径 `/plans/fixed3/route_polyline`；分配 `/plans/fixed3/assignments`；顺序及姿态 `/plans/fixed3/service_stops`。L=20.1558174996、K=6、J=24.6558174996。

适合放大显示桌面任务、左右臂标签、停靠姿态和完整底盘折线。该入口用于读图与数据格式，不把保存的单次耗时作为新增性能样本。

## 可复用的原图与文件指纹

原v4统计总览SVG：`rc3_v4_review_import/rc3_v4_review_evidence/experiments/v4_evidence_audit.svg`。保持原v4标题、平台、分组和图注；不要将其重标为v5.1或新缓存测试。此轮未生成任何新图。

| 索引source ID | SHA-256 |
|---|---|
| external | `29d402191cbae69d42ccdc4864711ba011ebfded3bc3f71a57f4c7910cbb72f0` |
| case_order | `6dfd2cbc8ffa2b0dfe8a9ab192db3775f562dd7bc96c4ff13cac834720f51219` |
| main100 | `491551177f3ba509a4e714db724fb27e99acb59dc46c45db589c4c0ce03cb79d` |
| v4_protocol | `ce6181c0977d322781a7c3d490f9b289cf4689d43ed59b19ff6d03af850cf829` |
| integration | `311c9618e62b68c158433915e634bec36883a1b8856b7fd326cb6aff51a01287` |
| existing_svg | `c78bb8aa4522e382b9f677b38b2ddade4cfc93496059afdd0717a89558e4cad9` |

`J=L+0.75K`是等效路程，不是实测服务总时间。以上三个入口均不能替代真实标定、IK/碰撞验证或独立统计样本；v5.1预检维护不改变这些冻结路线。
