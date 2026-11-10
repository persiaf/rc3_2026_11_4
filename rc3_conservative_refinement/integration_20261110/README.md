# 保守长期规划DRL：真实训练器集成

## 实际正式交付

已完成ON/OFF及匹配继续监督各128次接受更新、两组Critic各32次预热+128次在线更新，2376/2376请求完成且有效，0请求失败。完整评价为64/128的六方法×66视图；不把256未执行写成通过或失败。按同一账在64实际暂停/恢复，保存后分析20.918780秒，累计正式墙钟3733.860712秒。

优先看[正式报告与完整raw/H表](FORMAL_EXPERIMENT_REPORT.md)、[主张判读](RESULT_TO_CLAIM.md)、[资源账](FORMAL_RESOURCE_ACCOUNTING.json)、[部署权重索引](DEPLOYMENT_ARTIFACTS.json)和[下一步](NEXT_STEP_DECISION.md)。自动正式报告是独立判读前的事实汇总；当前科学判断以RESULT_TO_CLAIM及独立审查为准。部署只用约74KB修正包和准确SL256引用，Critic/Adam不进入推理；这些文件是负结果研究制品，不晋升默认规划器。

本轮已交付可运行的独立206维PPO训练入口，并完成准确SL256、真实保存输入的有账工程验收。**用户随后已明确批准准确R03参数及阶段，正式ON/OFF已在128按两项预定停止条件结束：64/128全部66视图的raw/H路线及成本均与FrozenSL相同，两组Critic均未超过固定常数基线；256未执行。** [实际用户批准](authorizations/USER_APPROVAL_SOURCE.txt)、[协议批准](authorizations/PROTOCOL_APPROVAL.json)和[独立阶段授权](authorizations/STAGE_AUTHORIZATION.json)分别保存。不能把下列工程Adam调用或两个native示例当作性能提升。

优先阅读[协议评审摘要](PROTOCOL_REVIEW.md)、[实际工程报告](EXPERIMENT_REPORT.md)、[训练器](../trainer.py)、[独立训练器审查](INDEPENDENT_TRAINER_REVIEW.md)和[下一步](NEXT_STEP_DECISION.md)。

实现链路：准确SL256冻结 → 原202维特征加停靠后4维摘要 → 零输出206维修正 → 各自在线完整轨迹 → 剩余L/K/H信用Critic → 独立PPO候选更新 → 同mask逐状态guard → 接受或严格恢复Actor/Adam → 持久日志/完整恢复 → actor-only部署。监督和原生基线保留为初始化及对照，研究核心仍是大父场景DRL训练、冻结后变尺度局部完整规划。

90项测试通过。20个真实请求通过，包括full10×10、4×3、6×6、8×8的ON/OFF/plainSL零修正一致性、真实PPO接受/拒绝、连续与恢复一致、部署重载和Fixed3/Detour16验证。实际5次Actor和4次Critic调用、1个既有教师标签重放，仅属工程；前两轮失败保留在R01/R02目录，累计未重置。[完整工程逐请求表](ENGINEERING_ROUTE_TABLE.csv)、[guard/优势摘要](ENGINEERING_UPDATE_SUMMARY.csv)、[资源账](RESOURCE_ACCOUNTING.json)。

运行环境实际为Windows、Python3.10.20、Torch2.12.1+cu126、NumPy2.2.6，CPU单线程和确定性算法；`pip check`通过。没有安装或升级环境。仍是二维导航/覆盖/同臂F&A与安全代理验证，未完成真实双臂IK或实机认证。

## 入口和复现

在仓库根目录使用已验收解释器：

```powershell
.\rc3_learning\.venv\Scripts\python.exe -B -X utf8 -m rc3_conservative_refinement.runner --help
```

本机该解释器链接现有conda环境，其他机器须依据[准确环境绑定](SOURCE_BINDING_R03.json)恢复相应Python/Torch/NumPy，不能把本机venv路径当作便携环境。

已执行的工程命令如下，输出不可覆盖；R01/R02失败也属于累计账。对已交付目录再次执行会拒绝，不能换目录重置预算：

```powershell
.\rc3_learning\.venv\Scripts\python.exe -B -X utf8 -m rc3_conservative_refinement.engineering --plan rc3_conservative_refinement/integration_20261110/ENGINEERING_PLAN_R03.json --output rc3_conservative_refinement/integration_20261110/engineering_R03
```

正式训练命令的真实入口如下。所列两份授权文件已依据用户对准确R03的真实确认分别创建。入口先验证准确源码、环境、固定日程、工程PASS及两份独立批准，缺任一项均在optimizer前拒绝（批准前已验证拒绝路径）：

```powershell
.\rc3_learning\.venv\Scripts\python.exe -B -X utf8 -m rc3_conservative_refinement.runner --protocol rc3_conservative_refinement/integration_20261110/PROTOCOL_DRAFT_R03.json --protocol-approval rc3_conservative_refinement/integration_20261110/authorizations/PROTOCOL_APPROVAL.json --stage-authorization rc3_conservative_refinement/integration_20261110/authorizations/STAGE_AUTHORIZATION.json
```

干净暂停可附`--pause-after 64`；后续使用同一协议/授权和`--resume`，不改变预算。每个完整机会闭合checkpoint；脏事务拒绝自动重放。恢复包含两网络、两Adam、RNG、资源、已用轨迹和持久账本前缀。

Actor部署API为`runtime.load_deployed_actor(path, expected_sha256=...)`，只载新修正包及准确冻结SL256引用；`runtime.plan_request(problem_factory, actor_factory=..., greedy=True)`计入输入准备、冷模型加载、provider、动态特征、选择、H、raw/H导出与独立验证。Critic不进入该推理链路；日志持久化计时另记。工程部署包为[actor_only.pt](engineering_R03/actor_only.pt)，仅用于工程等价证明，不作正式模型。

## 依赖、LFS和证据

使用现有`.gitattributes`，新增`.pt`继续Git LFS；不改历史规则、不生成整包ZIP。准确SL导出、上游checkpoint、sidecar/commit、输入scene/F/A/导航及旧schema收据均校验。仅下载源码ZIP会缺LFS权重和数组；恢复后须检查准确hash，而非只看文件存在。

```powershell
git lfs install
git lfs pull --include="rc3_integrated_rppo/runs/rc3_integrated_rppo_v1_20261110_01/actors/11622026/SL256_0256_1817f58aa650.pt,rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01/seed_11622026/S1/update_0256.pt,rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01/data/task_draws/*/arrays.npz,rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01/data/navigation/*.npz,rc3_integrated_rppo/runs/rc3_integrated_rppo_v1_20261110_01/inputs/*/arrays.npz,rc3_integrated_rppo/runs/rc3_integrated_rppo_v1_20261110_01/inputs/*/navigation.npz,rc3_conservative_refinement/integration_20261110/engineering_R03/*.pt"
```

本机所需LFS已恢复，未重新下载上述已有输入。具体每个输入的允许文件与SHA来自`SavedInputs`固定anchors及本轮[SCHEDULE_R03.json](SCHEDULE_R03.json)，无任务/导航生成后备。原60/S3/未批准final输入不在允许集合。

正式输出固定在`runs/conservative_paired_11622026_v1`，已保存完整raw/H计划、逐次PPO/guard/优势及checkpoint；64/128六方法评价、Critic校准、actor-only部署和父布局聚类对照均完成。256未执行，原目录不覆盖、不继续续跑。详见[实际复现与部署边界](DEPLOYMENT_AND_REPRODUCTION.md)、[文件索引](DELIVERY_FILE_INDEX.md)和[独立正式审查](INDEPENDENT_FORMAL_REVIEW.md)。
