# RC3-D6长序列信用分配对照

从539ead5ff52cb110a91736d6550cb988171d4d9b建立的独立研究。复用D5的32768条保存轨迹做成本分解，随后从准确SL256重新训练A（原D5损失）、B（因果剩余成本）和匹配继续SL，各256更新。旧D54096负结果、教师、权重和用户文件保持。

[数学](MATHEMATICS.md) · [实施审查](IMPLEMENTATION_REVIEW.md) · [精确协议](PROTOCOL.json) · [共同日程](COMMON_SCHEDULE.json) · [来源](REGISTERED_INPUTS.json) · [资源账](RESOURCE_ACCOUNTING.json)

```powershell
& .\rc3_learning\.venv\Scripts\python.exe -B -X utf8 -m rc3_d6_20261113.run --run rc3-d6-causal-tail-20261113-r1
& .\rc3_learning\.venv\Scripts\python.exe -B -X utf8 -m rc3_d6_20261113.run --resume rc3-d6-causal-tail-20261113-r1
& .\rc3_learning\.venv\Scripts\python.exe -B -X utf8 -m rc3_d6_20261113.run --status
```

恢复不会清零任何账；已成功的phase跳过，失败收据保留并要求具体修复，不把失败标成PASS。固定128段结束保存完整检查点，256段实际重启恢复。若发生优化器已扣但无完整检查点，停止为技术未完成，禁止免费重做。

首个optimizer前固定：256是主终点；B对FrozenSL和匹配SL父等权H后J均改善≥0.5%，SEL与DEV均不退化且完整代理可行。128只固定报告；B对A另报，不能以A退化替代两个主比较。未通过结束本候选，不加学习率、宽度、更新或种子。当前66案例均已暴露，仅开发筛选。
