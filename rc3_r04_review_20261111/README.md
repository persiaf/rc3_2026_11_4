# RC3-R04长期价值与相对策略：设计和工程审查

独立分支基于`853eb6b016df3aee08c233894ab6c24beb3765d8`；固定大父布局DRL训练与冻结局部完整规划目标。本轮不训练，仅完成方案、接口原型、独立测试和保存CSV审查。

优先阅读：

1. [R04_RESEARCH_DESIGN.md](R04_RESEARCH_DESIGN.md)：历史边界、依赖、可证伪假设与四项回答。
2. [R04_MATHEMATICAL_MODEL.md](R04_MATHEMATICAL_MODEL.md)：C1单位、H前缀信用、条件基线和pre-tanh居中数学。
3. [R04_IMPLEMENTATION_PLAN.md](R04_IMPLEMENTATION_PLAN.md)：真实trainer、loader及checkpoint的迁移阻断。
4. [R04_ABLATION_PROTOCOL.json](R04_ABLATION_PROTOCOL.json)：待批准的最小消融、预算、停止规则；不构成执行许可。
5. [R04_ACCEPTANCE_CRITERIA.md](R04_ACCEPTANCE_CRITERIA.md)：联合PPO门槛，现有证据尚未通过。
6. [R04_CODE_REVIEW.md](R04_CODE_REVIEW.md)：逐项代码/CSV检查与实际风险。

新原型为relative_actor.py、critic_interfaces.py；它们没有场景、规划、H或优化器入口。旧生产runtime导出器不能直接用于相对head，会错标语义；新的mapping接口测试不代表已修复旧生产器。

实际工程入口（只允许已登记的一次运行）：

```powershell
.\rc3_learning\.venv\Scripts\python.exe -B -X utf8 -m rc3_r04_review_20261111.register_engineering
.\rc3_learning\.venv\Scripts\python.exe -B -X utf8 -m rc3_r04_review_20261111.run_engineering
```

必须使用原准确Python3.10.20/Torch2.12.1+cu126/NumPy2.2.6环境。现有闭合/失败账本存在即拒绝重启，不能改目录重置消费。代码准备、Git/LFS维护与工程计算分账；结果见UNIT_TEST_RESULTS、OFFLINE_INTERFACE_RESULTS、EVIDENCE_AUDIT和RESOURCE_ACCOUNTING。

37个人工单元用例及8个保存base-logit状态的人工feature接口，不重建真实状态、不加载真实SL权重、不生成完整路线。所有Actor/Critic optimizer、新场景、导航、教师、native、H、原60和旧S3请求均为0。R03的128停止、256未运行及前轮全部负结论原样保留。
