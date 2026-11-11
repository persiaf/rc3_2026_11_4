# R04实际工程验收报告

状态：**PASS_ENGINEERING_ONLY_SCIENTIFIC_TRAINING_NOT_STARTED**。这是设计和工程记录，不是新的规划性能实验。

准确基准为853eb6b016df3aee08c233894ab6c24beb3765d8，首算前源码/设计/工程登记提交为5f12d5329eafddbfdae90993ed69769e9aadf1d5。登记SHA256为`5a3153c279610eac9092863b07a204ec8eb753029cfd012b7e599930b3fb18d2`，覆盖619份源码、32项原制品/CSV/协议锚点；900秒，全部optimizer和新科学请求cap为0。R04科学消融协议SHA为`f1e8165ffe426c5e2843e01e7d99c27089bbd7d96ef1a9f3330e8a7b6777f130`，其身份仍是NOT_APPROVED_NOT_ACTIVE。

实际命令均成功返回0：

```powershell
.\rc3_learning\.venv\Scripts\python.exe -B -X utf8 -m rc3_r04_review_20261111.register_engineering
.\rc3_learning\.venv\Scripts\python.exe -B -X utf8 -m rc3_r04_review_20261111.run_engineering
```

环境：Python3.10.20、Torch2.12.1+cu126、NumPy2.2.6、CPU单线程、确定性模式。实际累计150.8392281秒，小于900秒上限；没有失败重跑或重新登记。

| 项目 | 实际结果 | 覆盖及边界 |
| --- | --- | --- |
| 独立unit case | 37/37 PASS，约0.516秒 | Actor18、Critic19；人工张量、人工fit折记录，不是场景 |
| 离线接口 | PASS，约0.331秒 | 固定前8个保存base-logit状态，人工零206维features和冻结fixture encoder；PPO/C1 stateless loss有限与梯度隔离 |
| 保存CSV审计 | 41,624,938检查PASS，失败0，约130.162秒 | 全候选/状态/分量/日程与父隔离；新增全量399350个预测分量行、1568条窗口阶段summary、784组的streaming统计 |
| 字节复核 | 619源码/32锚点PASS | 旧R03、前轮诊断和原H/SL字节未变；不修改原审计收据 |

来源读取1327次、hash1316次，nn.Module调用256次、反向4次、autograd.grad0次。Actor/Critic optimizer均0，真实权重反序列化0，新场景/导航/教师/native/H、科学训练、原60和旧S3访问均0。4次反向仅验证梯度数学接口，没有参数学习；手工修改参数只是测试fixture。源码/原权重可做字节hash核验，该维护读取不等于加载真实权重进行推理。

旧审计函数复用的部分是维护复核，不伪称第二份独立实现；窗口/阶段矩统计另有新实现。所有算术PASS只说明保存数值与接口一致，不能改变原C1/C2或Actor的科研结论。分项时间不含全部启动、hash、日记写盘，也不应直接相加代替总墙钟。

有意未执行：新C1或条件残差拟合、任何Actor更新、联合PPO、真实SL/场景模型推理、完整raw/H路径验证、真实Adam回滚/恢复、生产文件部署或实机验证。这些未执行项不写成成功或失败，生产/科学G2/G3继续阻断。

证据：[UNIT_TEST_RESULTS](UNIT_TEST_RESULTS.json)、[OFFLINE_INTERFACE_RESULTS](OFFLINE_INTERFACE_RESULTS.json)、[EVIDENCE_AUDIT](EVIDENCE_AUDIT.json)、[RESOURCE_ACCOUNTING](RESOURCE_ACCOUNTING.json)、[FROZEN_BYTES_CHECK](FROZEN_BYTES_CHECK.json)。独立静态初始REVISE及修复保留，终审见[FINAL_INDEPENDENT_REVIEW.md](FINAL_INDEPENDENT_REVIEW.md)。
