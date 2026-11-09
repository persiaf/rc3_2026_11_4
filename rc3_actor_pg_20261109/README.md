# actor-only部署与匹配策略梯度小试

**已实际完成：756部署请求、A/B各128更新、3072训练终局请求和54固定DEV评价。** 十二actor逐参数精确相等，部署756请求raw/final路线与J/L/K完全保持。RL+H D2相对完整checkpoint的冷/热中位提速1.924/2.376倍，启动成本另列。B128在H后未通过预登记双对照筛选，不能宣布稳定改善SL起点；极小绝对均值下降也完整公开。

- [实际报告和四项回答](EXPERIMENT_REPORT.md)、[结果到主张](RESULT_TO_CLAIM.md)、[下一步](NEXT_STEP_DECISION.md)
- [12推理制品及来源](actors/ACTOR_INDEX.json)、[每模型制品大小](results/ACTOR_SIZES.csv)、[全部权重谱系](WEIGHT_LINEAGE.json)
- [756请求](results/DEPLOYMENT_REQUESTS.csv)、[七方法42格完整median/p95](results/DEPLOYMENT_TIMING.csv)、[提速与削减比例](results/DEPLOYMENT_SPEEDUP.csv)、[启动记录](results/STARTUP_AND_FIRST_RESPONSE.json)
- [18×3完整raw/H成本](results/DEV54_RESULTS.csv)、[匹配逐案例收益](results/DEV_MATCHED_PER_CASE.csv)、[绝对均值](results/DEV_MODEL_SUMMARY.csv)、[预登记父布局相对收益与区间](results/MECHANISM_COMPARISONS.json)
- [训练曲线](results/TRAINING_CURVES.csv)、[完整A记录](runs/A128/)、[完整B记录](runs/B128/)、[梯度与参数变化](results/GRADIENT_AND_WEIGHT_SUMMARY.json)
- [协议](PROTOCOL.json)、[用户授权范围](AUTHORIZATION.json)、[执行源码](SOURCE_BINDING.json)、[实例日程](INPUT_REGISTRY.json)、[零科学工程验收](ENGINEERING_VERIFICATION.json)
- [全部资源](RESOURCE_ACCOUNTING.json)、[运行环境](ENVIRONMENT_RECORD.json)、[独立审查](INDEPENDENT_REVIEW.md)、[Git记录](GIT_VERSION_AUDIT.json)

制品可通过`rc3_actor_pg_20261109.artifacts.load_actor`校验后加载；`Catalog('D2', models, seed=...)`仅驻留该seed四个actor，不保存编码。完整科学入口`python -B -X utf8 -m rc3_actor_pg_20261109.run --execute`已消费本轮固定日程；重复运行只识别已完成阶段，不提供额外免费请求。不要把合成负测夹具当模型。

独立实验`rc3_actor_pg_screen_20261109_01`，基准`925a0a1a8957299e3e849b29ac501a99212aa902`，源码SHA256`57b0d0d2236b4b1f942a393ce2fc41e6408f4e7ec899df038bdfc4b19a65eee6`。请检出Git提交后`git lfs pull`取得.pt等对象。S0—S3及旧失败原字节保留、旧v3指针不改、原60不读，不新增整包备份，不合并main。
