# Integrated-RPPO-v1

本模块实现冻结SL256骨干、零输出残差Actor、独立Critic及真正逐动作clipped PPO。固定目标为大规模父场景DRL训练，冻结后快速规划不同局部规模的完整约束任务。原H、同臂F&A、mask、端点和scope_r1复用旧验证内核；原60与旧S3失败槽保持。

执行协议及来源在首次调用前由 `register` 冻结并提交；`runner --execute` 按P0—P5条件自动推进。工程权重隔离，选择集、晋级集、最终确认集有各自身份。模型选择固定128/256/512机会，不能用训练loss或H改善替代RL收益。

```powershell
.\rc3_learning\.venv\Scripts\python.exe -B -X utf8 -m rc3_integrated_rppo.runner --execute
```

冻结Actor独立生成一份方案（**这是新增的显式用户推理**，单独预扣1请求、300秒，不能写回科学结果或解封尚未生成的最终任务）：

```powershell
.\rc3_learning\.venv\Scripts\python.exe -B -X utf8 -m rc3_integrated_rppo.infer --execute-one --seed 11622026 --case selection_000 --output rc3_integrated_rppo/consumer_plan_001.json
```

只读验证全部对象／Actor哈希及已存动作奖励，无新增推理、导航或H：

```powershell
.\rc3_learning\.venv\Scripts\python.exe -B -X utf8 -m rc3_integrated_rppo.replay --actions
```

完整训练checkpoint保留Adam/RNG/游标，Actor-only包只供D1推理。恢复只允许证据闭合的轮边界；已采样请求可以原动作重建，未闭合optimizer不得重放。资源失败不退款，时钟未闭合尾段必须继承。所有新研究对象位于本模块，旧证据不改写。

结果以 `EXPERIMENT_REPORT.md`、逐例CSV、`RESULT_SUMMARY.json`、独立审查与 `RESULT_TO_CLAIM.md` 为准。未触发的阶段明确记NOT_STARTED，开发筛选不能写成最终泛化／部署成功。
