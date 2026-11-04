# Research findings

## 2026-11-03：v3冻结实验到结论

- Experiment: `run_v3_stopfix_20261102` + `results/`；另有独立验证`verification/`与18场景工程诊断`engineering_profile.json`。
- Intended claim: 在可达性约束下，学习选择局部修复窗口能以受控计算预算改善服务路线，并提供快速、可靠输出。
- Verdict: **partial**。可支持代理模型内的可验证覆盖修复、Fixed3成本上界及本机无学习流程提速；不支持学习选择器全面优于detour或更快。
- Confidence: medium；论文级结论暂定，没有paper claim audit，测试只含一个选定训练检查点。
- Routing: narrow current claim and supplement future independent experiments；本轮保持冻结，不再用已查看测试调参。默认Fixed3，学习模式可选。
- Detailed evidence/limitations: [claims_review.md](claims_review.md)。
