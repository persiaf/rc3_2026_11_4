# T0～T2：教师价值蒸馏与可靠动作决策

独立研究分支：`codex/teacher-distill-t0-t2`。维护基准`5d5fe25`，R1研究证据`df8cf28`保持冻结。本轮按用户要求仅本地归档，不自动推送，不更改main、冻结标签或旧研究文件。T3、PPO/API及其他强化学习扩展未授权。

先读[T0诊断](T0_DIAGNOSTIC.md)、[实验计划](paper/EXPERIMENT_PLAN.md)、[事前审查](PREREG_REVIEW.md)，再看`protocol.json`、`seeds.json`、`registration_receipt.json`。运行结果汇总为[完整报告](REPORT_T012.md)、`STATUS.json`、`T1_GATE.json`和`resources.json`。T2必须通过T1全部门槛才能执行；未启动不是缺失结果。

使用项目已有解释器`rc3_learning/.venv/Scripts/python.exe -B`，无需安装LightGBM。统一菜单的独立单测：`-m unittest rc3_teacher_distill.test_candidate_menu -v`。

`-m rc3_teacher_distill.register`只允许一次事前登记；`-m rc3_teacher_distill.run`只允许一次正式研究运行，存在run_started.json即拒绝重放，不做替换种子或隐式续训。`-m rc3_teacher_distill.summarize`只读保存结果并更新本目录报告，没有研究调用。T0诊断脚本会重写本目录诊断文件，登记冻结后不应原地重跑以免改变已登记哈希；复核使用保存记录与独立审查。

全部菜单动作是原约束下的合法服务点；K≤8为部署接口限制，不删除原问题的合法服务点或任务。参考动作始终保留，规则/模型使用同一生成器。标签仅使用同一gB补全+完整H，模型不接触未查询动作伪标签。原60保留测试不生成、不读取、不查询。

完整路线计时包含获得gB所需的原生Fixed3规划，不能免费提供参考路线。即使动作学习通过，也可能因该固定开销不具备部署优势；质量与效率门槛分别报告。所有学习都是监督学习，不作为强化学习创新。
