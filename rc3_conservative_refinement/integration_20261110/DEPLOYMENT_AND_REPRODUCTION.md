# 部署与本轮复现边界

实际正式入口为`python -m rc3_conservative_refinement.runner`，新206维网络不经过旧202维训练入口。本轮准确源码/依赖/输入/H/种子由[SOURCE_BINDING_R03.json](SOURCE_BINDING_R03.json)、[R03协议](PROTOCOL_DRAFT_R03.json)及[固定日程](SCHEDULE_R03.json)绑定；协议和阶段批准分别见[authorizations](authorizations)。

已实际执行并返回0的两段正式命令，使用本机Python3.10.20、Torch2.12.1+cu126、NumPy2.2.6、CPU单线程：

```powershell
.\rc3_learning\.venv\Scripts\python.exe -B -X utf8 -m rc3_conservative_refinement.runner --protocol rc3_conservative_refinement/integration_20261110/PROTOCOL_DRAFT_R03.json --protocol-approval rc3_conservative_refinement/integration_20261110/authorizations/PROTOCOL_APPROVAL.json --stage-authorization rc3_conservative_refinement/integration_20261110/authorizations/STAGE_AUTHORIZATION.json --pause-after 64
.\rc3_learning\.venv\Scripts\python.exe -B -X utf8 -m rc3_conservative_refinement.runner --protocol rc3_conservative_refinement/integration_20261110/PROTOCOL_DRAFT_R03.json --protocol-approval rc3_conservative_refinement/integration_20261110/authorizations/PROTOCOL_APPROVAL.json --stage-authorization rc3_conservative_refinement/integration_20261110/authorizations/STAGE_AUTHORIZATION.json --resume
```

64后实际干净暂停、校验checkpoint并恢复至65，未重置Adam/RNG/游标、已用轨迹和累计资源。证据为[暂停观察](FORMAL_PAUSE_OBSERVATION.json)、[恢复观察](FORMAL_RESUME_OBSERVATION.json)和原账本。工程R03另有连续两步与暂停恢复精确一致检查；没有额外执行一套完整连续128对照。

本轮128后按两个终点无增益及Critic误差规则终闭合，256未执行。现有目录是不可覆盖的证据；再次使用原命令不能获得新预算，保存后分析也使训练恢复边界关闭。新的重复训练须另行登记身份、来源、累计预算与授权，不通过换目录覆盖本轮结果。

已实际执行一次、返回0的保存后分析命令：

```powershell
.\rc3_learning\.venv\Scripts\python.exe -B -X utf8 rc3_conservative_refinement/integration_20261110/postprocess_formal.py --plan rc3_conservative_refinement/integration_20261110/ANALYSIS_PLAN.json
.\rc3_learning\.venv\Scripts\python.exe -B -X utf8 rc3_conservative_refinement/integration_20261110/write_formal_report.py --analysis rc3_conservative_refinement/integration_20261110/runs/conservative_paired_11622026_v1/analysis
```

分析为同一账本一次执行，实际20.918780秒、0新请求/模型加载/规划/H/验证器/optimizer；报告工具仅汇总已审计表，维护耗时另记。分析输出和报告都不可覆盖，失败不得通过新输出路径重置额度。

## 实际部署制品

[DEPLOYMENT_ARTIFACTS.json](DEPLOYMENT_ARTIFACTS.json)列出6个actor-only包的精确SHA/大小和准确SL256引用。每个74,389字节，分别对应ON/OFF/继续监督的64、128终点，各在正式评价中实际冷加载66次；严格loader拒绝Critic、Adam、旧schema或错误SL来源。另保存138个完整训练checkpoint，用于追溯已接受更新，不进入部署。

部署API为`runtime.load_deployed_actor(path, expected_sha256=...)`；`runtime.plan_request(problem_factory, actor_factory=..., greedy=True)`负责完整请求计时。这些API已在工程等价和正式actor-only评价中实际调用。工程导出重载与直接Actor的raw/H路线及J/L/K精确一致；正式六文件的实际路线与原验证收据在[逐请求表](runs/conservative_paired_11622026_v1/analysis/ALL_RAW_H_ROUTES.csv)中。

包只保存修正权重和准确冻结SL引用，部署仍需原SL256、上游S1来源收据及所用输入的来源文件。本机`.venv`不是便携环境；其他机器须恢复准确依赖与LFS内容，并核SHA，不能把Git指针当模型。所有`.pt`沿用原Git LFS；本run闭合大日志采用目录内专属LFS规则，根`.gitattributes`原字节保留。

## 时间与限制

完整请求从保存输入和导航D/路径加载、来源检查开始，包含模型加载、provider/特征、Actor、H、导出及raw/H独立代理验证。原输入的导航解算不在本轮重新执行；落盘时间另列，Critic只用于训练或单独诊断。冷加载指模型对象加载，文件系统缓存未控制，不能解读成冷OS进程启动。

全部2376请求完成且有效。当前模型没有性能晋升；Fixed3/Detour仍为默认。验证范围是二维导航、覆盖、同臂F&A与安全代理，未完成真实双臂IK、关节轨迹或实机认证。原60、旧S3及未批准final保持未访问。
